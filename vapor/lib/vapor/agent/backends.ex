defmodule Vapor.Agent.Backend do
  @moduledoc """
  Where an agent's decisions come from. `complete/4` takes OpenAI-shaped
  messages and tools and returns the assistant's content and tool calls,
  whether the decision is *re-derivable* (`deterministic`), and what to
  record about it.

  The distinction is the honest boundary of verifiability:

    * `Vapor.Agent.Backend.Local` — a vapor engine. Its output is a function
      of (model, prompt tokens, sampling parameters, seed) on every
      substrate, so a replay recomputes it and must get the same tokens.
    * `Vapor.Agent.Backend.OpenAI` (any OpenAI-compatible server: OpenAI,
      vLLM, llama.cpp, Ollama, or vapor's own `Vapor.Serve`) and
      `Vapor.Agent.Backend.Anthropic` (the Messages API) — frontier models
      as *observations*: their answers are recorded and replay reads the
      record. Nothing claims they are reproducible; what is guaranteed is
      that the record cannot be changed unnoticed.
  """

  @type result :: %{content: binary, calls: [map], deterministic: boolean, record: map}
  @callback complete(term, [map], [map], map) :: {:ok, result} | {:error, term}

  @doc "Dispatch to the backend struct's module."
  def complete(%mod{} = b, messages, tools, opts), do: mod.complete(b, messages, tools, opts)
end

defmodule Vapor.Agent.Backend.Local do
  @moduledoc """
  A local model behind `Vapor.Engine`: the model's own chat template
  (`Vapor.Chat.load_template/1`) renders tools and history, tool calls are
  constrained to the declared tools and schemas (`Vapor.Tools`), and the
  clock the template may read is the run's recorded clock.
  """
  @behaviour Vapor.Agent.Backend
  alias Vapor.{Chat, Engine, Tokenizer, Tools}

  defstruct [:engine, :tk, :template, :vocab, :model_id]

  @doc "Options: `engine:`, `tokenizer:`, `template:` (optional), `model_id:` (the spec's model id)."
  def new(opts) do
    tk = Keyword.fetch!(opts, :tokenizer)
    %__MODULE__{engine: Keyword.fetch!(opts, :engine), tk: tk, template: Keyword.get(opts, :template),
                vocab: Vapor.Grammar.Vocab.build(tk), model_id: Keyword.get(opts, :model_id)}
  end

  @impl true
  def complete(%__MODULE__{} = b, messages, tools, opts) do
    dialect = Tools.dialect(b.template && b.template.source)

    with {:ok, {text, add_bos, stop}} <- render(b, messages, tools, opts) do
      ids = Tokenizer.encode(b.tk, text, add_bos: add_bos)

      constraint =
        if tools != [] do
          case Tools.call_grammar(dialect, tools) do
            {:error, _} -> nil
            g -> Vapor.Grammar.Constraint.new(g, b.vocab, stop, {:lazy, Tools.opener(dialect)})
          end
        end

      gen = [max_tokens: opts["max_tokens"], temperature: opts["temperature"] * 1.0, seed: opts["seed"], stop_ids: stop, constraint: constraint]

      with {:ok, out, reason, _usage} <- Engine.complete(b.engine, ids, gen) do
        out_text = Tokenizer.decode(b.tk, out)
        receipt = Vapor.Canonical.hex_digest({:completion, b.model_id, ids, Keyword.drop(gen, [:constraint]), out})
        {content, calls} = if tools != [], do: Tools.parse(dialect, out_text, receipt), else: {out_text, []}

        {:ok, %{content: content, calls: calls, deterministic: true,
                record: %{"tokens" => out, "prompt" => Vapor.Canonical.hex_digest(ids), "finish" => Atom.to_string(reason), "receipt" => receipt}}}
      end
    end
  end

  defp render(%{template: nil} = b, messages, tools, _opts) do
    msgs = if tools != [], do: with_note(messages, tools), else: messages
    Chat.render(b.tk, Enum.map(msgs, &flatten/1))
  end

  defp render(b, messages, tools, opts) do
    Chat.render(b.tk, messages, b.template, tools: if(tools != [], do: tools), now: opts["now"])
  end

  defp with_note([%{"role" => "system", "content" => c} = s | rest], tools), do: [%{s | "content" => c <> "\n\n" <> Tools.system_note(tools)} | rest]
  defp with_note(msgs, tools), do: [%{"role" => "system", "content" => Tools.system_note(tools)} | msgs]

  defp flatten(%{"role" => "tool", "content" => c}), do: %{"role" => "user", "content" => "<tool_response>\n#{c}\n</tool_response>"}

  defp flatten(%{"role" => "assistant", "tool_calls" => calls} = m) when is_list(calls) and calls != [] do
    text = Enum.map_join(calls, "\n", fn c -> f = c["function"]; "<tool_call>\n" <> Vapor.JSON.encode(%{"name" => f["name"], "arguments" => f["arguments"]}) <> "\n</tool_call>" end)
    %{"role" => "assistant", "content" => (m["content"] || "") <> text}
  end

  defp flatten(m), do: Map.take(Map.update(m, "content", "", &(&1 || "")), ["role", "content"])
end

defmodule Vapor.Agent.HTTP do
  @moduledoc false
  # JSON over HTTPS with OTP alone (:httpc, :ssl): certificates verified
  # against the system store, hostnames checked
  def post_json(url, headers, body, timeout \\ 300_000) do
    {:ok, _} = Application.ensure_all_started(:inets)
    {:ok, _} = Application.ensure_all_started(:ssl)

    ssl =
      if String.starts_with?(url, "https"),
        do: [ssl: [verify: :verify_peer, cacerts: :public_key.cacerts_get(), depth: 4,
                   customize_hostname_check: [match_fun: :public_key.pkix_verify_hostname_match_fun(:https)]]],
        else: []

    hs = Enum.map(headers, fn {k, v} -> {String.to_charlist(k), String.to_charlist(v)} end)

    case :httpc.request(:post, {String.to_charlist(url), hs, ~c"application/json", Vapor.JSON.encode(body)}, [timeout: timeout] ++ ssl, body_format: :binary) do
      {:ok, {{_, 200, _}, _h, resp}} -> Vapor.JSON.decode(resp)
      {:ok, {{_, status, _}, _h, resp}} -> {:error, {:http, status, String.slice(resp, 0, 500)}}
      {:error, why} -> {:error, why}
    end
  end
end

defmodule Vapor.Agent.Backend.OpenAI do
  @moduledoc """
  Any OpenAI-compatible chat completions endpoint — OpenAI itself, vLLM,
  llama.cpp's server, Ollama, or vapor's `Vapor.Serve` (whose
  `x-vapor-receipt` makes even a remote vapor model checkable by whoever
  holds the same model). Decisions are recorded as observations.
  """
  @behaviour Vapor.Agent.Backend
  defstruct base_url: "https://api.openai.com/v1", model: nil, api_key: nil

  @impl true
  def complete(%__MODULE__{} = b, messages, tools, opts) do
    body =
      %{"model" => b.model, "messages" => messages, "temperature" => opts["temperature"], "max_tokens" => opts["max_tokens"], "seed" => opts["seed"]}
      |> then(fn m -> if tools != [], do: Map.merge(m, %{"tools" => tools, "tool_choice" => "auto"}), else: m end)

    headers = if b.api_key, do: [{"authorization", "Bearer " <> b.api_key}], else: []

    with {:ok, %{"choices" => [%{"message" => msg} | _]} = resp} <- Vapor.Agent.HTTP.post_json(b.base_url <> "/chat/completions", headers, body) do
      calls =
        for c <- msg["tool_calls"] || [] do
          f = c["function"]
          args = case f["arguments"] do
            s when is_binary(s) -> case Vapor.JSON.decode(s) do {:ok, a} -> a; _ -> %{"$unparsed" => s} end
            m -> m
          end
          %{"id" => c["id"], "name" => f["name"], "arguments" => args}
        end

      {:ok, %{content: msg["content"] || "", calls: calls, deterministic: false,
              record: %{"provider" => "openai-compatible", "model" => resp["model"] || b.model, "id" => resp["id"],
                        "usage" => resp["usage"]}}}
    end
  end
end

defmodule Vapor.Agent.Backend.Anthropic do
  @moduledoc """
  Anthropic's Messages API (`POST /v1/messages`, `anthropic-version:
  2023-06-01`): tools as `{name, description, input_schema}`, calls as
  `tool_use` blocks, results sent back as `tool_result` blocks. Decisions
  are recorded as observations.
  """
  @behaviour Vapor.Agent.Backend
  defstruct base_url: "https://api.anthropic.com", model: nil, api_key: nil, version: "2023-06-01"

  @impl true
  def complete(%__MODULE__{} = b, messages, tools, opts) do
    {system, rest} = Enum.split_with(messages, &(&1["role"] == "system"))

    body =
      %{"model" => b.model, "max_tokens" => opts["max_tokens"], "temperature" => opts["temperature"], "messages" => convert(rest)}
      |> then(fn m -> if system != [], do: Map.put(m, "system", Enum.map_join(system, "\n\n", & &1["content"])), else: m end)
      |> then(fn m ->
        if tools != [],
          do: Map.merge(m, %{"tools" => Enum.map(tools, fn %{"function" => f} -> %{"name" => f["name"], "description" => f["description"] || "", "input_schema" => f["parameters"]} end),
                             "tool_choice" => %{"type" => "auto"}}),
          else: m
      end)

    headers = [{"x-api-key", b.api_key || ""}, {"anthropic-version", b.version}]

    with {:ok, %{"content" => blocks} = resp} <- Vapor.Agent.HTTP.post_json(b.base_url <> "/v1/messages", headers, body) do
      text = blocks |> Enum.filter(&(&1["type"] == "text")) |> Enum.map_join(& &1["text"])
      calls = for %{"type" => "tool_use"} = u <- blocks, do: %{"id" => u["id"], "name" => u["name"], "arguments" => u["input"] || %{}}

      {:ok, %{content: text, calls: calls, deterministic: false,
              record: %{"provider" => "anthropic", "model" => resp["model"] || b.model, "id" => resp["id"], "stop_reason" => resp["stop_reason"],
                        "usage" => resp["usage"]}}}
    end
  end

  # OpenAI-shaped history → Messages API turns (tool results merge into one user turn)
  defp convert(msgs) do
    msgs
    |> Enum.map(fn
      %{"role" => "assistant", "tool_calls" => calls} = m when is_list(calls) and calls != [] ->
        text = if (m["content"] || "") != "", do: [%{"type" => "text", "text" => m["content"]}], else: []
        uses = for c <- calls, do: %{"type" => "tool_use", "id" => c["id"], "name" => c["function"]["name"], "input" => c["function"]["arguments"]}
        %{"role" => "assistant", "content" => text ++ uses}

      %{"role" => "tool"} = m ->
        %{"role" => "user", "content" => [%{"type" => "tool_result", "tool_use_id" => m["tool_call_id"], "content" => m["content"]}]}

      m ->
        %{"role" => m["role"], "content" => m["content"] || ""}
    end)
    |> Enum.chunk_while(nil, fn
      %{"role" => "user", "content" => [%{"type" => "tool_result"} | _] = c}, %{"role" => "user", "content" => [%{"type" => "tool_result"} | _] = acc} ->
        {:cont, %{"role" => "user", "content" => acc ++ c}}
      m, nil -> {:cont, m}
      m, acc -> {:cont, acc, m}
    end, fn nil -> {:cont, nil}; acc -> {:cont, acc, nil} end)
    |> Enum.reject(&is_nil/1)
  end
end
