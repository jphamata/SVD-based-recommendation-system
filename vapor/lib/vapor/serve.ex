defmodule Vapor.Serve do
  @moduledoc """
  An OpenAI-compatible HTTP/1.1 server on `:gen_tcp` (no dependencies).

      {:ok, _} = Vapor.Serve.start_link(engine: engine, tokenizer: tk, port: 8000)

  Endpoints: `GET /v1/models`, `POST /v1/completions`,
  `POST /v1/chat/completions` (both with `"stream": true` as server-sent
  events over chunked transfer encoding, ending in `data: [DONE]`),
  `POST /v1/embeddings` (with an `embedder:`, see `Vapor.Embed`),
  `GET /health`, and the operator console at `/` with its `/v1/vapor/*`
  endpoints (`Vapor.Console`: documents, search with provenance, the
  model's contract, the noise gate). Without an `engine:` the server runs
  the console and the document library alone. Request fields: `prompt` (text, or a list of token ids),
  `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`,
  `stop`; unknown fields are ignored, as OpenAI's servers do.

  Agents' fields: `tools` with `tool_choice` (`"auto"`, `"none"`,
  `"required"`, or a named function) and `response_format`
  (`{"type": "json_object"}` or `{"type": "json_schema", "json_schema":
  {"schema": …, "strict": …}}`). The prompt is rendered by the model's own
  chat template when there is one (`template:`, see `Vapor.Chat`), and the
  output is *constrained* (`Vapor.Grammar`): a structured output is valid
  JSON for its schema, a tool call names a declared tool with arguments
  valid for it (`Vapor.Tools`) — by construction, not by retry. Calls come
  back as `message.tool_calls` with `finish_reason: "tool_calls"`; with
  tools and `stream: true` the reply is sent as one final chunk.

  Every completion carries `x-vapor-receipt`: the canonical digest
  (`Vapor.Canonical`) of the model's identity, the prompt token ids, the
  sampling parameters and the generated ids. The engine's outputs are a
  deterministic function of exactly these, so anyone with the same model can
  re-derive the response and check the receipt.

  Concurrency is the BEAM's: an acceptor per listening socket hands each
  connection to its own process, which submits to the engine and receives
  that request's tokens as messages — so many clients stream at once while
  the engine batches all of their sequences into each step. Streaming text
  is cut only at UTF-8 character boundaries.

  TLS and HTTP/2-3 belong in a reverse proxy in front (see
  docs/ECOSSISTEMA.md); this server speaks plain HTTP/1.1 with keep-alive.
  """
  use GenServer
  require Logger
  alias Vapor.{Chat, Engine, JSON, Tokenizer}

  @max_body 16 * 1024 * 1024

  def start_link(opts), do: GenServer.start_link(__MODULE__, opts, Keyword.take(opts, [:name]))

  @doc "The bound port (useful with `port: 0`)."
  def port(server), do: GenServer.call(server, :port)

  @impl true
  def init(opts) do
    {:ok, ls} =
      :gen_tcp.listen(Keyword.get(opts, :port, 8000),
                      [:binary, packet: :http_bin, active: false, reuseaddr: true, backlog: 128,
                       ip: Keyword.get(opts, :ip, {127, 0, 0, 1})])

    {:ok, port} = :inet.port(ls)
    ctx = context(opts)

    for _ <- 1..Keyword.get(opts, :acceptors, 4), do: spawn_link(fn -> accept(ls, ctx) end)
    {:ok, %{ls: ls, port: port}}
  end

  @impl true
  def handle_call(:port, _from, s), do: {:reply, s.port, s}

  @doc """
  The request context — engine, tokenizer, template, embedder, clock —
  from the options of `start_link/1`. Transports other than this module's
  HTTP/1.1 listener (a `Plug` in a Phoenix router, see
  `integrations/vapor_plug`) build it once and call `dispatch/3`.
  """
  def context(opts) do
    tk = Keyword.get(opts, :tokenizer)

    # the console's library (`Vapor.Console`): given, or a fresh one
    library = Keyword.get_lazy(opts, :library, fn -> {:ok, h} = Vapor.Console.Holder.start_link(); h end)
    # the transparency log the console anchors receipts in (`Vapor.Tlog`)
    tlog = Keyword.get_lazy(opts, :tlog, fn -> {:ok, h} = Vapor.Tlog.Holder.start_link(); h end)

    %{engine: Keyword.get(opts, :engine), tk: tk, library: library, tlog: tlog, model: Keyword.get(opts, :model_name, "vapor"),
      template: Keyword.get(opts, :template), embedder: Keyword.get(opts, :embedder),
      model_digest: Keyword.get(opts, :model_digest, Keyword.get(opts, :model_name, "vapor")),
      now: Keyword.get(opts, :now, fn -> DateTime.utc_now() end),
      # with a token, every request but /health must carry it (Bearer, or the
      # cookie set by opening /?token=…): required when listening beyond loopback
      token: Keyword.get(opts, :token),
      vocab: if(tk, do: vocab(tk))}
  end

  @doc """
  Serve one request through a *responder* `{module, ref}`: `module.json(ref,
  status, body, headers)`, `module.stream_start(ref, headers)`,
  `module.chunk(ref, iodata)` and `module.stream_end(ref)` — the transport
  (this module's `:gen_tcp` listener, or a `Plug.Conn`). `request` is
  `%{method: :GET | :POST, path: binary, body: binary}`.
  """
  def dispatch(responder, request, ctx) do
    Process.delete(:vapor_serve_streaming)

    case authorize(request, ctx) do
      :ok -> handle(responder, request, ctx)
      {:login, token} -> Vapor.Console.login(responder, token)
      :denied -> reply(responder, 401, %{error: %{message: "a token is required (Authorization: Bearer …, or open /?token=… once)", type: "authentication_error"}})
    end
  rescue
    e ->
      Logger.error("vapor serve: " <> Exception.format(:error, e, __STACKTRACE__))
      error = %{error: %{message: "internal error", type: "server_error"}}

      # once a stream has begun, the status line is gone: the error is the
      # stream's last event, and the connection is not reused
      if Process.delete(:vapor_serve_streaming) do
        _ = event(responder, error)
        _ = chunk(responder, "data: [DONE]\n\n")
        _ = stream_end(responder)
        false
      else
        reply(responder, 500, error)
      end
  end

  defp accept(ls, ctx) do
    case :gen_tcp.accept(ls) do
      {:ok, sock} ->
        pid = spawn(fn -> receive(do: (:go -> conn(sock, ctx))) end)
        :ok = :gen_tcp.controlling_process(sock, pid)
        send(pid, :go)
        accept(ls, ctx)

      {:error, :closed} ->
        :ok
    end
  end

  # ---------------------------------------------------------- connection --

  defp conn(sock, ctx) do
    case read_request(sock) do
      {:ok, req} ->
        keep = dispatch({__MODULE__.TCP, sock}, req, ctx)

        if keep and req.keep_alive, do: conn(sock, ctx), else: :gen_tcp.close(sock)

      _ ->
        :gen_tcp.close(sock)
    end
  end

  defp read_request(sock) do
    with {:ok, {:http_request, method, {:abs_path, path}, _v}} <- :gen_tcp.recv(sock, 0, 300_000),
         {:ok, headers} <- read_headers(sock, %{}),
         len = String.to_integer(Map.get(headers, "content-length", "0")),
         true <- len <= @max_body || {:error, :too_large},
         :ok <- :inet.setopts(sock, packet: :raw),
         {:ok, body} <- if(len > 0, do: :gen_tcp.recv(sock, len, 60_000), else: {:ok, ""}),
         :ok <- :inet.setopts(sock, packet: :http_bin) do
      {:ok, %{method: method, path: path, headers: headers, body: body,
              keep_alive: String.downcase(Map.get(headers, "connection", "keep-alive")) != "close"}}
    end
  rescue
    ArgumentError -> {:error, :bad_request}
  end

  defp read_headers(sock, acc) do
    case :gen_tcp.recv(sock, 0, 60_000) do
      {:ok, {:http_header, _, name, _, value}} -> read_headers(sock, Map.put(acc, String.downcase(to_string(name)), value))
      {:ok, :http_eoh} -> {:ok, acc}
      other -> {:error, other}
    end
  end

  # returns whether the connection may be kept
  defp handle(sock, %{method: :GET, path: "/health"}, _ctx), do: reply(sock, 200, %{status: "ok"})

  defp handle(sock, %{method: :GET, path: "/v1/models"}, ctx),
    do: reply(sock, 200, %{object: "list", data: [%{id: ctx.model, object: "model", created: 0, owned_by: "vapor"}]})

  defp handle(sock, %{method: :POST, path: "/v1/embeddings", body: body}, ctx) do
    with {:ok, req} <- decode(body),
         e when e != nil <- ctx.embedder || {:error, "this server has no embedding model"},
         inputs = List.wrap(req["input"]),
         true <- (inputs != [] and Enum.all?(inputs, &(is_binary(&1) or is_list(&1)))) || {:error, "input: text or a list of texts"},
         {:ok, vecs} <- Vapor.Embed.embed(e, inputs) do
      tokens = inputs |> Enum.map(&length(Vapor.Embed.ids(e, &1))) |> Enum.sum()

      reply(sock, 200, %{object: "list", model: ctx.model,
                         data: vecs |> Enum.with_index() |> Enum.map(fn {v, i} -> %{object: "embedding", index: i, embedding: v} end),
                         usage: %{prompt_tokens: tokens, total_tokens: tokens}},
            [{"x-vapor-receipt", Vapor.Canonical.hex_digest({:embeddings, e.digest, inputs})}])
    else
      {:error, why} -> reply(sock, 400, %{error: %{message: describe(why), type: "invalid_request_error"}})
    end
  end

  defp handle(sock, %{method: :POST, path: path}, %{engine: nil}) when path in ["/v1/completions", "/v1/chat/completions"],
    do: reply(sock, 503, %{error: %{message: "no model is loaded on this server (start it with --model)", type: "server_error"}})

  defp handle(sock, %{method: :POST, path: path, body: body}, ctx) when path in ["/v1/completions", "/v1/chat/completions"] do
    chat? = path == "/v1/chat/completions"

    with {:ok, req, oreq} <- decode_both(body),
         {:ok, plan} <- tools_plan(req, oreq, chat?, ctx),
         {:ok, prompt, add_bos, stop_ids} <- prompt(req, oreq, chat?, plan, ctx),
         ids = if(is_list(prompt), do: prompt, else: Tokenizer.encode(ctx.tk, prompt, add_bos: add_bos)),
         opts = gen_opts(req) ++ [stop_ids: stop_ids, constraint: plan.constraint],
         {:ok, ref} <- Engine.generate(ctx.engine, ids, opts) do
      # an engine that dies mid-request ends the request (503), never hangs it
      mon = Process.monitor(GenServer.whereis(ctx.engine))
      id = (if chat?, do: "chatcmpl-", else: "cmpl-") <> Base.url_encode64(:crypto.strong_rand_bytes(9))
      receipt = fn out_ids -> Vapor.Canonical.hex_digest({:completion, ctx.model_digest, ids, Keyword.drop(opts, [:constraint]), plan.digest, out_ids}) end
      meta = %{id: id, created: System.os_time(:second), model: ctx.model, chat?: chat?, plan: plan, receipt: receipt, engine: ctx.engine, mon: mon}

      try do
        if req["stream"] == true and plan.dialect == nil and plan.prefill == "", do: stream(sock, ref, meta), else: whole(sock, ref, meta, req["stream"] == true)
      after
        # a long-lived transport process (a Plug adapter's) keeps no stale monitor
        Process.demonitor(mon, [:flush])
      end
    else
      {:error, why} -> reply(sock, 400, %{error: %{message: describe(why), type: "invalid_request_error"}})
    end
  end

  defp handle(sock, req, ctx) do
    case Vapor.Console.handle(sock, req, ctx) do
      :pass -> reply(sock, 404, %{error: %{message: "not found", type: "invalid_request_error"}})
      keep -> keep
    end
  end

  @doc false
  # constant-time comparison; /health stays open (a load balancer's probe)
  def authorize(_req, %{token: t}) when t in [nil, ""], do: :ok
  def authorize(%{path: "/health"}, _ctx), do: :ok

  def authorize(req, %{token: token}) do
    headers = Map.get(req, :headers, %{})
    bearer = case headers["authorization"] do
      "Bearer " <> b -> String.trim(b)
      _ -> nil
    end

    cookie = (headers["cookie"] || "") |> String.split(";") |> Enum.map(&String.trim/1)
             |> Enum.find_value(fn "vapor_token=" <> v -> v; _ -> nil end)
    query = case req.path do
      "/?token=" <> q -> URI.decode_www_form(q)
      _ -> nil
    end

    eq = fn x -> is_binary(x) and byte_size(x) == byte_size(token) and :crypto.hash_equals(x, token) end

    cond do
      eq.(query) -> {:login, token}
      eq.(bearer) or eq.(cookie) -> :ok
      true -> :denied
    end
  end

  defp decode(body) do
    case JSON.decode(body) do
      {:ok, %{} = m} -> {:ok, m}
      _ -> {:error, "body: a JSON object"}
    end
  end

  # the plain request, and the order-preserving one (for chat templates)
  defp decode_both(body) do
    with {:ok, req} <- decode(body),
         {:ok, {:dict, pairs}} <- JSON.decode(body, ordered: true) do
      {:ok, req, Map.new(pairs)}
    end
  end

  # the vocabulary prepared for constraints, once per tokenizer
  defp vocab(tk) do
    key = {__MODULE__, :vocab, :erlang.phash2(tk.surface)}

    case :persistent_term.get(key, nil) do
      nil -> v = Vapor.Grammar.Vocab.build(tk); :persistent_term.put(key, v); v
      v -> v
    end
  end

  @no_plan %{dialect: nil, constraint: nil, prefill: "", digest: nil}

  # what constrains this request: a structured output, tool calls, or nothing
  defp tools_plan(req, oreq, chat?, ctx) do
    tools = req["tools"]
    choice = req["tool_choice"] || "auto"
    eos = Vapor.Chat.stop_ids(ctx.tk)

    cond do
      fmt = req["response_format"] ->
        case grammar_for(fmt) do
          {:ok, nil} -> {:ok, @no_plan}
          {:ok, g} -> {:ok, %{@no_plan | constraint: Vapor.Grammar.Constraint.new(g, ctx.vocab, eos, :strict), digest: {:format, fmt}}}
          err -> err
        end

      chat? and is_list(tools) and tools != [] and choice != "none" ->
        dialect = Vapor.Tools.dialect(ctx.template && ctx.template.source)
        only = case choice do
          %{"function" => %{"name" => n}} -> n
          _ -> nil
        end

        case Vapor.Tools.call_grammar(dialect, tools, only) do
          {:error, why} -> {:error, why}
          g ->
            {mode, prefill} =
              if choice == "auto",
                do: {{:lazy, Vapor.Tools.opener(dialect)}, ""},
                else: {:strict, Vapor.Tools.opener(dialect) <> if(dialect in [:hermes, :generic], do: "\n", else: "")}

            _ = oreq
            {:ok, %{dialect: dialect, constraint: Vapor.Grammar.Constraint.new(g, ctx.vocab, eos, mode), prefill: prefill,
                    digest: {:tools, Vapor.Canonical.hex_digest(Vapor.Tools.plain(tools)), choice}}}
        end

      true ->
        {:ok, @no_plan}
    end
  end

  defp grammar_for(%{"type" => "text"}), do: {:ok, nil}
  defp grammar_for(%{"type" => "json_object"}), do: {:ok, Vapor.Grammar.new({:ref, :object})}

  defp grammar_for(%{"type" => "json_schema", "json_schema" => %{"schema" => schema} = js}) do
    case Vapor.Grammar.JSONSchema.compile(schema, lenient: js["strict"] != true) do
      {:ok, g} -> {:ok, g}
      {:error, r} -> {:error, "response_format: #{r.bound} (#{inspect(r.node)})"}
    end
  end

  defp grammar_for(_), do: {:error, "response_format: text, json_object or json_schema"}

  defp prompt(req, oreq, true, plan, ctx) do
    with msgs when is_list(msgs) <- oreq["messages"] || {:error, "messages is required"},
         {:ok, {text, bos, stop}} <- chat_text(req, oreq, msgs, plan, ctx) do
      {:ok, text <> plan.prefill, bos, stop}
    end
  end

  defp prompt(%{"prompt" => p}, _o, false, _plan, _ctx) when is_binary(p), do: {:ok, p, true, []}

  defp prompt(%{"prompt" => [i | _] = ids}, _o, false, _plan, _ctx) when is_integer(i),
    do: if(Enum.all?(ids, &is_integer/1), do: {:ok, ids, false, []}, else: {:error, "prompt: text or token ids"})

  defp prompt(_req, _o, false, _plan, _ctx), do: {:error, "prompt: text or token ids"}

  # the model's template when it has one; else the built-in formats, with
  # tools taught by a system note
  defp chat_text(req, oreq, msgs, plan, %{template: %{} = tpl} = ctx) do
    extra = Map.take(req, ["enable_thinking"]) |> Map.merge(Map.get(req, "chat_template_kwargs", %{}))
    tools = if plan.dialect, do: oreq["tools"], else: nil
    Chat.render(ctx.tk, msgs, tpl, tools: tools, now: ctx.now.(), vars: extra)
  end

  defp chat_text(req, _oreq, _msgs, plan, ctx) do
    msgs = Vapor.Tools.plain(req["messages"])

    msgs =
      if plan.dialect do
        note = Vapor.Tools.system_note(req["tools"])
        case msgs do
          [%{"role" => "system", "content" => c} = s | rest] -> [%{s | "content" => c <> "\n\n" <> note} | rest]
          _ -> [%{"role" => "system", "content" => note} | msgs]
        end
      else
        msgs
      end

    Chat.render(ctx.tk, Enum.map(msgs, &flatten_message/1))
  end

  # the built-in formats know text turns only: calls and results become text
  defp flatten_message(%{"role" => "tool", "content" => c}), do: %{"role" => "user", "content" => "<tool_response>\n#{c}\n</tool_response>"}

  defp flatten_message(%{"role" => "assistant", "tool_calls" => calls} = m) when is_list(calls) do
    text = Enum.map_join(calls, "\n", fn c -> "<tool_call>\n" <> Vapor.JSON.encode(Map.take(c["function"] || c, ["name", "arguments"])) <> "\n</tool_call>" end)
    %{"role" => "assistant", "content" => (m["content"] || "") <> text}
  end

  defp flatten_message(%{"content" => nil} = m), do: %{m | "content" => ""}
  defp flatten_message(m), do: m

  defp gen_opts(r) do
    [max_tokens: r["max_tokens"] || r["max_completion_tokens"] || 128, temperature: num(r["temperature"], 1.0),
     top_p: num(r["top_p"], 1.0), top_k: r["top_k"] || 0, seed: r["seed"] || 0, stop: List.wrap(r["stop"])]
  end

  defp num(nil, d), do: d
  defp num(x, _d) when is_number(x), do: x * 1.0
  defp num(_, d), do: d

  defp describe({:context_length, n, max}), do: "#{n} tokens exceed the context of #{max}"
  defp describe(why) when is_binary(why), do: why
  defp describe(why), do: inspect(why)

  defp finish(:length), do: "length"
  defp finish(_), do: "stop"

  # ------------------------------------------------------------ responses --

  @lost %{error: %{message: "the worker computing this request was lost; retry", type: "server_error"}}

  defp whole(sock, ref, meta, as_stream?) do
    case gather(ref, meta.mon, "", []) do
      {_, _, :error, _} -> reply(sock, 503, @lost)
      {text, ids, reason, usage} -> whole(sock, meta, text, ids, reason, usage, as_stream?)
    end
  end

  defp whole(sock, meta, text, ids, reason, usage, as_stream?) do
    text = valid_prefix(meta.plan.prefill <> text) |> elem(0)
    usage = Map.put(usage, :total_tokens, usage.prompt_tokens + usage.completion_tokens)
    receipt = meta.receipt.(ids)

    {message, fin} =
      case meta.plan.dialect do
        nil -> {%{role: "assistant", content: text}, finish(reason)}
        d ->
          case Vapor.Tools.parse(d, text, receipt) do
            {content, []} -> {%{role: "assistant", content: content}, finish(reason)}
            {content, calls} -> {%{role: "assistant", content: if(content == "", do: nil, else: content), tool_calls: Vapor.Tools.openai(calls)}, "tool_calls"}
          end
      end

    if as_stream? do
      stream_start(sock, [{"x-vapor-receipt", receipt}])
      event(sock, %{id: meta.id, object: "chat.completion.chunk", created: meta.created, model: meta.model,
                    choices: [%{index: 0, delta: message, finish_reason: fin}]})
      chunk(sock, "data: [DONE]\n\n")
      stream_end(sock)
    else
      choice =
        if meta.chat?,
          do: %{index: 0, message: message, finish_reason: fin},
          else: %{index: 0, text: text, logprobs: nil, finish_reason: fin}

      reply(sock, 200, %{id: meta.id, object: if(meta.chat?, do: "chat.completion", else: "text_completion"),
                         created: meta.created, model: meta.model, choices: [choice], usage: usage},
            [{"x-vapor-receipt", receipt}])
    end
  end

  # an engine that dies mid-request ends the request (503), never hangs it
  defp gather(ref, mon, acc, ids) do
    receive do
      {:vapor, ^ref, {:token, id, bytes}} -> gather(ref, mon, acc <> bytes, [id | ids])
      {:vapor, ^ref, {:done, reason, usage}} -> {acc, Enum.reverse(ids), reason, usage}
      {:DOWN, ^mon, :process, _, _} -> {acc, Enum.reverse(ids), :error, %{prompt_tokens: 0, completion_tokens: 0}}
    end
  end

  defp stream(sock, ref, meta) do
    stream_start(sock, [])
    object = if meta.chat?, do: "chat.completion.chunk", else: "text_completion"
    base = %{id: meta.id, object: object, created: meta.created, model: meta.model}
    if meta.chat?, do: event(sock, Map.put(base, :choices, [%{index: 0, delta: %{role: "assistant"}, finish_reason: nil}]))

    delta = fn text, fin ->
      if meta.chat?,
        do: %{index: 0, delta: if(text == "", do: %{}, else: %{content: text}), finish_reason: fin},
        else: %{index: 0, text: text, logprobs: nil, finish_reason: fin}
    end

    # the last chunk carries the receipt (the ids are known only then)
    sent =
      stream_loop(sock, {ref, meta.mon}, "", [], fn text, fin, ids ->
        ev = Map.put(base, :choices, [delta.(text, fin)])
        event(sock, if(fin, do: Map.put(ev, :vapor_receipt, meta.receipt.(ids)), else: ev))
      end)

    case sent do
      :ok ->
        chunk(sock, "data: [DONE]\n\n")
        stream_end(sock)

      # the client hung up: the request leaves the engine's batch now
      {:error, _} ->
        Engine.cancel(meta.engine, ref)
        false
    end
  end

  # :ok, or the transport's error once the client is gone
  defp stream_loop(sock, {ref, mon} = refs, pending, ids, send_delta) do
    receive do
      {:vapor, ^ref, {:token, id, bytes}} ->
        {ready, rest} = valid_prefix(pending <> bytes)

        case if(ready != "", do: send_delta.(ready, nil, nil), else: :ok) do
          :ok -> stream_loop(sock, refs, rest, [id | ids], send_delta)
          error -> error
        end

      # mid-stream, a lost worker can only be reported in the stream
      {:vapor, ^ref, {:done, :error, _usage}} ->
        event(sock, @lost)

      {:DOWN, ^mon, :process, _, _} ->
        event(sock, @lost)

      # an unfinished character left at the end is not text: dropped
      {:vapor, ^ref, {:done, reason, _usage}} ->
        _ = pending
        send_delta.("", finish(reason), Enum.reverse(ids))
    end
  end

  defp event(sock, obj), do: chunk(sock, ["data: ", JSON.encode(obj), "\n\n"])

  # the responder: `{module, ref}` (see dispatch/3); chunk returns :ok or
  # {:error, reason} once the client is gone
  defp chunk({m, ref}, data), do: m.chunk(ref, data)
  defp stream_start({m, ref}, headers) do
    Process.put(:vapor_serve_streaming, true)
    m.stream_start(ref, headers)
  end
  defp stream_end({m, ref}) do
    Process.delete(:vapor_serve_streaming)
    m.stream_end(ref)
  end

  @doc false
  # the longest prefix that is complete UTF-8, and the rest (at most 3 bytes
  # of an unfinished character — or invalid bytes, which are dropped)
  def valid_prefix(bin) do
    case String.chunk(bin, :valid) do
      [] -> {"", ""}
      chunks ->
        {init, [last]} = Enum.split(chunks, -1)
        good = init |> Enum.filter(&String.valid?/1) |> Enum.join()

        cond do
          String.valid?(last) -> {good <> last, ""}
          byte_size(last) <= 3 and unfinished?(last) -> {good, last}
          true -> {good, ""}
        end
    end
  end

  defp unfinished?(<<c, rest::binary>>) when c >= 0xC0, do: byte_size(rest) < lead_len(c) - 1 and Enum.all?(:binary.bin_to_list(rest), &(&1 in 0x80..0xBF))
  defp unfinished?(_), do: false
  defp lead_len(c) when c >= 0xF0, do: 4
  defp lead_len(c) when c >= 0xE0, do: 3
  defp lead_len(_), do: 2

  defp reply({m, ref}, status, obj, headers \\ []), do: m.json(ref, status, JSON.encode(obj), headers)
end

defmodule Vapor.Serve.TCP do
  @moduledoc false
  # the responder of Vapor.Serve's own HTTP/1.1 listener (returns whether the
  # connection may be kept)

  @phrase %{200 => "OK", 400 => "Bad Request", 404 => "Not Found", 406 => "Not Acceptable", 422 => "Unprocessable Entity",
            500 => "Internal Server Error", 503 => "Service Unavailable"}

  def json(sock, status, body, headers), do: send_body(sock, status, "application/json", body, headers)

  def send_body(sock, status, ctype, body, headers) do
    extra = Enum.map(headers, fn {k, v} -> [k, ": ", v, "\r\n"] end)

    :gen_tcp.send(sock, ["HTTP/1.1 #{status} #{@phrase[status]}\r\ncontent-type: ", ctype, "\r\n", extra, "content-length: ",
                         Integer.to_string(byte_size(body)), "\r\n\r\n", body])

    true
  end

  def stream_start(sock, headers) do
    extra = Enum.map(headers, fn {k, v} -> [k, ": ", v, "\r\n"] end)
    :ok = :gen_tcp.send(sock, ["HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\n", extra, "transfer-encoding: chunked\r\n\r\n"])
  end

  def chunk(sock, data) do
    data = IO.iodata_to_binary(data)
    :gen_tcp.send(sock, [Integer.to_string(byte_size(data), 16), "\r\n", data, "\r\n"])
  end

  def stream_end(sock) do
    chunk(sock, "")
    true
  end
end
