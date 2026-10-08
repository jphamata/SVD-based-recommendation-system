defmodule Vapor.Console.Hall do
  @moduledoc """
  The HTTP face of conversations (`Vapor.Majlis`) and of the terminal
  (`Vapor.Diwan`). Every endpoint is a thin call; the logic is where the CLI
  and the TUI reach it too.

  | endpoint | what it does |
  |---|---|
  | `GET /v1/vapor/threads` | threads (newest first), the configured models, the tools a thread may enable |
  | `POST /v1/vapor/threads` | `{title, system, model, tools, budget}` → `{id}` |
  | `POST /v1/vapor/threads/import` | `{data}`: a vapor, ChatGPT or Claude export → `{threads}` |
  | `GET /v1/vapor/threads/:id` | settings and the current path (every message with its versions) |
  | `GET /v1/vapor/threads/:id/tree` · `/context` | every branch · exactly what the model will see |
  | `GET /v1/vapor/threads/:id/export?format=json\\|markdown` | the export |
  | `POST /v1/vapor/threads/:id/:op` | `say` `{text, reply}`, `reply`, `edit` `{node, text, reply}`, `regenerate` `{node}`, `switch`/`rewind` `{node}`, `fork` `{node, title}`, `pin` `{node, on}`, `compact` `{upto, text}`, `uncompact`, `settings` `{…}`, `share`, `revoke`, `delete` |
  | `POST /v1/vapor/chat/search` | `{query}`: every thread |
  | `GET /v1/vapor/journal/:id` | an agent run's journal (verifiable) |
  | `GET /v1/vapor/shared/:id?cap=…` · `GET /shared/:id?cap=…` | a shared thread, read-only (JSON · a page) — **no console token needed: the capability is the authority** |
  | `POST /v1/vapor/diwan` | `{session, line}` → `{session, out, err, code, codes, files}` |
  | `GET /v1/vapor/diwan/file?session=…&name=…` · `POST /v1/vapor/diwan/file` `{session, name, text}` | a session file |
  | `POST /v1/vapor/diwan/complete` | `{session, line}` → completions for the last word |
  """
  alias Vapor.{Diwan, JSON}
  alias Vapor.Majlis, as: M

  # --------------------------------------------------------------- dispatch

  def handle(sock, %{method: :GET, path: "/shared/" <> rest}, ctx), do: shared_page(sock, rest, ctx)
  def handle(sock, %{method: :GET, path: "/v1/vapor/shared/" <> rest}, ctx), do: with_m(sock, ctx, fn m -> shared_json(sock, m, rest) end)

  def handle(sock, %{method: :GET, path: "/v1/vapor/threads"}, ctx) do
    with_m(sock, ctx, fn m ->
      json(sock, 200, %{threads: M.threads(m), models: M.backends(m), tools: Vapor.Majlis.Tools.names()})
    end)
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/threads", body: body}, ctx) do
    with_m(sock, ctx, fn m ->
      with_body(sock, body, fn r ->
        fields = for {k, v} <- [title: r["title"], system: r["system"], model: r["model"], tools: r["tools"], budget: r["budget"]], v != nil, do: {k, v}
        reply(sock, M.new(m, fields), &%{id: &1})
      end)
    end)
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/threads/import", body: body}, ctx) do
    with_m(sock, ctx, fn m -> with_body(sock, body, fn r -> reply(sock, M.import(m, to_string(r["data"] || "")), &%{threads: &1}) end) end)
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/chat/search", body: body}, ctx) do
    with_m(sock, ctx, fn m -> with_body(sock, body, fn r -> json(sock, 200, %{hits: M.search(m, to_string(r["query"] || ""), 30)}) end) end)
  end

  def handle(sock, %{method: :GET, path: "/v1/vapor/journal/" <> id}, ctx) do
    with_m(sock, ctx, fn m ->
      case M.journal(m, id) do
        {:ok, j} -> json(sock, 200, %{run_id: j.run_id, head: j.head, verified: Vapor.Agent.Journal.verify(j) == :ok, events: j.events})
        {:error, e} -> json(sock, 404, %{error: %{message: to_string(e)}})
      end
    end)
  end

  def handle(sock, %{method: :GET, path: "/v1/vapor/threads/" <> rest}, ctx) do
    with_m(sock, ctx, fn m ->
      {path, query} = split_query(rest)

      case String.split(path, "/") do
        [tid] ->
          with {:ok, t} <- M.thread(m, tid), {:ok, p} <- M.path(m, tid) do
            json(sock, 200, %{thread: t, path: p})
          else
            e -> fail(sock, e)
          end

        [tid, "tree"] -> reply(sock, M.tree(m, tid), & &1)
        [tid, "context"] -> reply(sock, M.context(m, tid), &Map.delete(&1, :messages))

        [tid, "export"] ->
          fmt = if query["format"] in ["markdown", "md"], do: :markdown, else: :json
          case M.export(m, tid, fmt) do
            {:ok, text} -> raw(sock, 200, if(fmt == :json, do: "application/json", else: "text/markdown; charset=utf-8"), text)
            e -> fail(sock, e)
          end

        _ -> :pass
      end
    end)
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/threads/" <> rest, body: body}, ctx) do
    with_m(sock, ctx, fn m ->
      case String.split(rest, "/") do
        [tid, op] -> with_body(sock, body, fn r -> thread_op(sock, m, tid, op, r) end)
        _ -> :pass
      end
    end)
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/diwan", body: body}, ctx) do
    with_body(sock, body, fn r ->
      line = to_string(r["line"] || "")

      if byte_size(line) > 100_000 do
        bad(sock, "line: at most 100 000 bytes")
      else
        case checkout(r["session"], ctx) do
          {:ok, sid, st} ->
            {res, st} =
              try do
                Diwan.eval(line, st)
              after
                :ok
              end

            checkin(sid, st)
            json(sock, 200, Map.merge(res, %{session: sid, files: Map.keys(st.files) |> Enum.sort()}))

          {:busy, sid} ->
            json(sock, 409, %{error: %{message: "a command is still running in this session"}, session: sid})
        end
      end
    end)
  end

  def handle(sock, %{method: :GET, path: "/v1/vapor/diwan/file?" <> q}, ctx) do
    query = URI.decode_query(q)

    case peek(query["session"], ctx) do
      {:ok, st} ->
        case Map.fetch(st.files, Diwan.clean_path(query["name"] || "")) do
          {:ok, d} -> json(sock, 200, %{name: query["name"], text: if(String.valid?(d), do: d), base64: if(String.valid?(d), do: nil, else: Base.encode64(d))})
          :error -> json(sock, 404, %{error: %{message: "no such file"}})
        end

      :error ->
        json(sock, 404, %{error: %{message: "no such session"}})
    end
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/diwan/file", body: body}, ctx) do
    with_body(sock, body, fn r ->
      data =
        case r do
          %{"base64" => b} when is_binary(b) -> Base.decode64(b)
          %{"text" => t} when is_binary(t) -> {:ok, t}
          _ -> :error
        end

      with {:ok, d} <- data,
           {:ok, sid, st} <- checkout(r["session"], ctx) do
        case Diwan.write_file(st, to_string(r["name"] || ""), d, :write) do
          {:ok, st} -> checkin(sid, st); json(sock, 200, %{session: sid, files: Map.keys(st.files) |> Enum.sort()})
          {:error, why} -> checkin(sid, st); bad(sock, why)
        end
      else
        :error -> bad(sock, "give text or base64")
        {:busy, _} -> json(sock, 409, %{error: %{message: "a command is still running in this session"}})
      end
    end)
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/diwan/complete", body: body}, ctx) do
    with_body(sock, body, fn r ->
      files = case peek(r["session"], ctx) do {:ok, st} -> Map.keys(st.files); :error -> [] end
      json(sock, 200, %{completions: complete(to_string(r["line"] || ""), files)})
    end)
  end

  def handle(_sock, _req, _ctx), do: :pass

  # --------------------------------------------------------------- threads

  defp thread_op(sock, m, tid, op, r) do
    gen = for {k, v} <- [temperature: r["temperature"], max_tokens: r["max_tokens"], seed: r["seed"], backend: r["backend"]], v != nil, do: {k, v}

    case op do
      "say" ->
        case M.say(m, tid, to_string(r["text"] || "")) do
          {:ok, id} ->
            if r["reply"] == false, do: json(sock, 200, %{node: id}), else: reply(sock, M.reply(m, tid, gen), &Map.put(&1, :question, id))

          e -> fail(sock, e)
        end

      "reply" -> reply(sock, M.reply(m, tid, gen), & &1)

      "edit" ->
        case M.edit(m, tid, to_string(r["node"]), to_string(r["text"] || "")) do
          {:ok, id} ->
            {:ok, n} = M.node(m, id)
            if r["reply"] != false and n.role == "user", do: reply(sock, M.reply(m, tid, gen), &Map.put(&1, :question, id)), else: json(sock, 200, %{node: id})

          e -> fail(sock, e)
        end

      "regenerate" -> reply(sock, M.regenerate(m, tid, to_string(r["node"]), gen), & &1)
      "switch" -> reply(sock, M.switch(m, tid, to_string(r["node"])), &%{head: &1})
      "rewind" -> reply(sock, M.rewind(m, tid, to_string(r["node"])), &%{head: &1})
      "fork" -> reply(sock, M.fork(m, tid, [at: r["node"], title: r["title"]] |> Enum.reject(fn {_, v} -> v == nil end)), &%{id: &1})
      "pin" -> reply(sock, M.pin(m, tid, to_string(r["node"]), r["on"] != false), fn _ -> %{ok: true} end)

      "compact" ->
        result = if is_binary(r["text"]), do: with(:ok <- M.compact(m, tid, to_string(r["upto"]), r["text"]), do: {:ok, %{upto: r["upto"], text: r["text"]}}), else: M.summarize(m, tid, r["upto"])
        reply(sock, result, & &1)

      "uncompact" -> reply(sock, M.uncompact(m, tid), fn _ -> %{ok: true} end)

      "settings" ->
        fields = for {k, v} <- [title: r["title"], system: r["system"], model: r["model"], tools: r["tools"], budget: r["budget"], gen: r["gen"]], v != nil, do: {k, v}
        reply(sock, M.set(m, tid, fields), fn _ -> %{ok: true} end)

      "share" -> reply(sock, M.share(m, tid), &%{cap: &1, url: "/shared/#{tid}?cap=#{&1}"})
      "revoke" -> reply(sock, M.revoke(m, tid), fn _ -> %{ok: true} end)
      "delete" -> reply(sock, M.delete(m, tid), fn _ -> %{ok: true} end)
      _ -> :pass
    end
  end

  defp shared_json(sock, m, rest) do
    {tid, q} = split_query(rest)

    case M.shared(m, tid, q["cap"]) do
      {:ok, v} -> json(sock, 200, v)
      {:error, :forbidden} -> json(sock, 403, %{error: %{message: "this link is not valid (it may have been revoked)"}})
    end
  end

  # a shared thread as a page anyone with the link can read: no script, every text escaped
  defp shared_page(sock, rest, ctx) do
    {tid, q} = split_query(rest)

    case ctx[:majlis] && M.shared(ctx.majlis, tid, q["cap"]) do
      {:ok, v} ->
        body =
          Enum.map_join(v.messages, "\n", fn msg ->
            ~s(<article class="#{esc(msg.role)}"><h2>#{esc(msg.role)} <small>#{esc(msg.t)}</small></h2><div>#{esc(msg.content)}</div></article>)
          end)

        html = """
        <!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
        <meta name="referrer" content="no-referrer"><title>#{esc(v.title)} · vapor</title>
        <style>
        :root{--bg:#f4f1e8;--fg:#1b1a17;--mut:#6b675e;--acc:#7a5c12;--card:#fffdf7}
        @media (prefers-color-scheme:dark){:root{--bg:#14110d;--fg:#ece6d8;--mut:#a39d8e;--acc:#d6a93a;--card:#1d1914}}
        body{background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,sans-serif;max-width:760px;margin:0 auto;padding:24px 16px}
        h1{font-family:Georgia,serif;font-weight:400}article{background:var(--card);border-radius:10px;padding:12px 16px;margin:12px 0}
        article.user{border-left:3px solid var(--acc)}h2{font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin:0 0 6px}
        small{text-transform:none;letter-spacing:0}div{white-space:pre-wrap;overflow-wrap:anywhere}footer{color:var(--mut);font-size:.8rem;margin-top:24px}
        </style></head><body><h1>#{esc(v.title)}</h1>
        #{body}
        <footer>A read-only conversation shared from vapor. Every message is content-addressed; the owner can revoke this link.</footer></body></html>
        """

        raw(sock, 200, "text/html; charset=utf-8", html)

      _ ->
        raw(sock, 403, "text/plain; charset=utf-8", "This link is not valid (it may have been revoked).\n")
    end
  end

  # --------------------------------------------------------------- sessions

  defp sessions do
    case Process.whereis(__MODULE__.Sessions) do
      nil ->
        case Agent.start(fn -> %{} end, name: __MODULE__.Sessions) do
          {:ok, pid} -> pid
          {:error, {:already_started, pid}} -> pid
        end

      pid ->
        pid
    end
  end

  # a session is taken for the length of a command; a second command in the same
  # session meanwhile is refused (409) rather than racing over its files
  defp checkout(sid, ctx) do
    Agent.get_and_update(sessions(), fn all ->
      case is_binary(sid) && all[sid] do
        %{busy: true} -> {{:busy, sid}, all}
        %{st: st} = s -> {{:ok, sid, st}, Map.put(all, sid, %{s | busy: true, used: System.monotonic_time()})}
        _ ->
          new = Vapor.Entropy.token(12)
          st = Diwan.new(majlis: ctx[:majlis], jail: true, tty: true)
          all = if map_size(all) >= 256, do: Map.delete(all, all |> Enum.min_by(fn {_, s} -> s.used end) |> elem(0)), else: all
          {{:ok, new, st}, Map.put(all, new, %{st: st, busy: true, used: System.monotonic_time()})}
      end
    end)
  end

  defp checkin(sid, st), do: Agent.update(sessions(), &Map.put(&1, sid, %{st: st, busy: false, used: System.monotonic_time()}))

  defp peek(sid, _ctx) do
    case is_binary(sid) && Agent.get(sessions(), &Map.get(&1, sid)) do
      %{st: st} -> {:ok, st}
      _ -> :error
    end
  end

  # the terminal completes every verb but those that make no sense in a jail (models from disk, the editors' server)
  # siphon: fetching is the person's act at their own terminal, never a page's (docs/SIPHON.md); render writes
  # a file on the server (the console has its render desk); qalam needs a terminal of one's own
  @verbs Vapor.Main.verbs() -- ~w(lsp version search palingenesis siphon render qalam)

  @doc "Completions for the last word of a line: verbs and builtins first, then the session's files."
  def complete(line, files) do
    words = String.split(line, ~r/\s+/)
    last = List.last(words) || ""
    pool = if length(words) <= 1 or List.last(Enum.drop(words, -1)) == "|", do: @verbs ++ Diwan.builtins(), else: files ++ @verbs

    pool |> Enum.filter(&String.starts_with?(&1, last)) |> Enum.uniq() |> Enum.sort() |> Enum.take(30)
  end

  # ---------------------------------------------------------------- helpers

  defp with_m(sock, ctx, f) do
    case ctx[:majlis] do
      nil -> json(sock, 503, %{error: %{message: "conversations are not enabled on this server (start it with --data DIR)"}})
      m -> f.(m)
    end
  end

  defp with_body(sock, body, f) do
    case JSON.decode(body || "") do
      {:ok, r} when is_map(r) -> f.(r)
      _ -> if body in [nil, ""], do: f.(%{}), else: bad(sock, "body: a JSON object")
    end
  end

  defp reply(sock, {:ok, v}, f), do: json(sock, 200, f.(v))
  defp reply(sock, :ok, f), do: json(sock, 200, f.(:ok))
  defp reply(sock, e, _f), do: fail(sock, e)

  defp fail(sock, {:error, :forbidden}), do: json(sock, 403, %{error: %{message: "forbidden"}})
  defp fail(sock, {:error, why}) when is_binary(why), do: json(sock, if(why =~ ~r/^no (thread|message)/, do: 404, else: 422), %{error: %{message: why}})
  defp fail(sock, other), do: json(sock, 422, %{error: %{message: inspect(other)}})

  defp bad(sock, why), do: json(sock, 400, %{error: %{message: why}})

  defp split_query(s) do
    case String.split(s, "?", parts: 2) do
      [p, q] -> {p, URI.decode_query(q)}
      [p] -> {p, %{}}
    end
  end

  defp json({m, ref}, status, obj), do: m.json(ref, status, JSON.encode(Vapor.Main.jsonable(obj)), [])

  defp raw({m, ref} = sock, status, ctype, body) do
    if Code.ensure_loaded?(m) and function_exported?(m, :send_body, 5),
      do: m.send_body(ref, status, ctype, body, []),
      else: json(sock, 406, %{error: %{message: "this transport serves JSON only"}})
  end

  defp esc(nil), do: ""
  defp esc(s), do: s |> to_string() |> String.replace("&", "&amp;") |> String.replace("<", "&lt;") |> String.replace(">", "&gt;") |> String.replace("\"", "&quot;")
end
