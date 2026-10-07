defmodule Vapor.Main.ChatCli do
  @moduledoc false
  # `vapor chat` — conversations (Vapor.Majlis) from the terminal. Inside the
  # console's terminal it talks to the server's majlis; from a shell, to the
  # store in $VAPOR_HOME/majlis (default ~/.vapor/majlis), with the model of
  # VAPOR_MIND (anthropic:…, openai:…@URL, script:FILE).
  import Vapor.Main
  alias Vapor.Majlis, as: M

  @usage """
  vapor chat                                   the threads, newest first
  vapor chat new [--title T --system S --tools a,b --budget N --model M]
  vapor chat say ID TEXT|- [--no-reply --temperature X --max-tokens N --seed N]
  vapor chat show ID [--tree]                  the current path (or every branch)
  vapor chat edit ID MSG TEXT|- [--no-reply]   a new version of a message: a branch
  vapor chat regen ID [MSG]                    another answer to the same question
  vapor chat switch ID MSG · rewind ID MSG     follow another branch · move back
  vapor chat fork ID [MSG] [--title T]         a new thread sharing the past up to MSG
  vapor chat pin ID MSG · unpin ID MSG         always keep a message in the context
  vapor chat context ID                        exactly what the model will see, and what is dropped
  vapor chat compact ID [MSG] [--text T]       summarise the path up to MSG (by the model, or your text)
  vapor chat set ID [--title --system --tools --budget --model]
  vapor chat search WORDS                      every thread
  vapor chat export ID [--md] · import FILE    vapor JSON (hashes re-checked), ChatGPT or Claude exports
  vapor chat share ID · revoke ID              a read-only link's capability · kill every link
  vapor chat rm ID · gc · tools                delete a thread · compact the store · the tools a thread may use
  MSG is a message id (6+ hex digits are enough).
  """

  def run([]), do: with_majlis(fn m -> list(m, []) end)
  def run(["help" | _]), do: (out(@usage); 0)

  def run([cmd | rest]) do
    case opts(rest, [title: :string, system: :string, tools: :string, budget: :integer, model: :string, reply: :boolean, temperature: :float,
                     max_tokens: :integer, seed: :integer, tree: :boolean, md: :boolean, text: :string]) do
      :usage -> 2
      {:ok, o, args} -> with_majlis(fn m -> dispatch(cmd, args, o, m) end)
    end
  end

  defp with_majlis(f) do
    case Process.get(:vapor_majlis) do
      nil ->
        dir = Path.join(System.get_env("VAPOR_HOME") || Path.join(System.user_home!(), ".vapor"), "majlis")

        backends =
          case Vapor.Mind.from_env() do
            nil -> %{}
            mind -> %{mind.name => mind.backend}
          end

        case M.start_link(dir: dir, backends: backends) do
          {:ok, m} ->
            try do
              f.(m)
            after
              GenServer.stop(m)
            end

          {:error, why} ->
            err("chat: #{inspect(why)}")
            4
        end

      m ->
        f.(m)
    end
  end

  defp dispatch("new", [], o, m) do
    fields = [title: o[:title], system: o[:system], tools: split(o[:tools]), budget: o[:budget], model: o[:model]] |> Enum.reject(fn {_, v} -> v in [nil, []] end)

    case M.new(m, fields) do
      {:ok, tid} -> if json?(o), do: emit_json(%{thread: tid}), else: out(tid); 0
      {:error, e} -> fail(e, 3)
    end
  end

  defp dispatch("list", _, o, m), do: list(m, o)

  defp dispatch("say", [tid | words], o, m) do
    with {:ok, text} <- text_arg(words) do
      case M.say(m, tid, text) do
        {:ok, id} -> if Keyword.get(o, :reply, true), do: answer(m, tid, o), else: done_id(id, o)
        {:error, e} -> fail(e, 3)
      end
    else
      {:error, e} -> fail(e, 3)
    end
  end

  defp dispatch("show", [tid], o, m) do
    if o[:tree] do
      case M.tree(m, tid) do
        {:ok, t} -> if json?(o), do: emit_json(t), else: Enum.each(t.nodes, &out(tree_line(&1))); 0
        {:error, e} -> fail(e, 3)
      end
    else
      case M.path(m, tid) do
        {:ok, p} -> if json?(o), do: emit_json(p), else: Enum.each(p.messages, &out(message(&1))); 0
        {:error, e} -> fail(e, 3)
      end
    end
  end

  defp dispatch("edit", [tid, msg | words], o, m) do
    with {:ok, text} <- text_arg(words), {:ok, id} <- M.edit(m, tid, msg, text) do
      {:ok, n} = M.node(m, id)
      if n.role == "user" and Keyword.get(o, :reply, true), do: answer(m, tid, o), else: done_id(id, o)
    else
      {:error, e} -> fail(e, 3)
    end
  end

  defp dispatch("regen", [tid | rest], o, m) do
    target =
      case rest do
        [msg] -> {:ok, msg}
        [] -> with {:ok, p} <- M.path(m, tid), %{} = last <- Enum.find(Enum.reverse(p.messages), &(&1.role == "assistant")) || {:error, "no answer to regenerate yet"}, do: {:ok, last.id}
      end

    with {:ok, msg} <- target do
      case M.regenerate(m, tid, msg, gen(o)) do
        {:ok, r} -> show_reply(r, o)
        {:error, e} -> fail(e, 4)
      end
    else
      {:error, e} -> fail(e, 3)
    end
  end

  defp dispatch(cmd, [tid, msg], o, m) when cmd in ["switch", "rewind"] do
    f = if cmd == "switch", do: &M.switch/3, else: &M.rewind/3
    case f.(m, tid, msg) do
      {:ok, id} -> done_id(id, o)
      {:error, e} -> fail(e, 3)
    end
  end

  defp dispatch("fork", [tid | rest], o, m) do
    case M.fork(m, tid, [at: List.first(rest), title: o[:title]] |> Enum.reject(fn {_, v} -> v == nil end)) do
      {:ok, new} -> if json?(o), do: emit_json(%{thread: new}), else: out(new); 0
      {:error, e} -> fail(e, 3)
    end
  end

  defp dispatch(cmd, [tid, msg], o, m) when cmd in ["pin", "unpin"] do
    case M.pin(m, tid, msg, cmd == "pin") do
      :ok -> if json?(o), do: emit_json(%{ok: true}), else: out(dim("#{cmd}ned")); 0
      {:error, e} -> fail(e, 3)
    end
  end

  defp dispatch("context", [tid], o, m) do
    case M.context(m, tid) do
      {:ok, c} ->
        if json?(o) do
          emit_json(Map.delete(c, :messages))
        else
          out(bold("#{c.tokens}#{if c.exact, do: "", else: " ≈"} / #{c.budget} tokens") <> dim("  (system and summary: #{c.system_tokens})"))
          if c.summary, do: out(dim("  summary of #{c.summary["covers"]} messages up to #{String.slice(c.summary["upto"], 0, 12)}, by #{c.summary["by"]}"))
          Enum.each(c.items, fn i ->
            mark = case i.status do "sent" -> good("●"); "pinned" -> good("◆"); "summarized" -> dim("≡"); _ -> warn("○") end
            out("  #{mark} #{String.slice(i.id, 0, 10)} #{String.pad_trailing(i.role, 9)} #{String.pad_leading(Integer.to_string(i.tokens), 6)}  #{i.status}")
          end)
        end
        0

      {:error, e} -> fail(e, 3)
    end
  end

  defp dispatch("compact", [tid | rest], o, m) do
    result =
      case o[:text] do
        nil -> M.summarize(m, tid, List.first(rest))
        text ->
          with {:ok, p} <- M.path(m, tid), upto = List.first(rest) || (Enum.at(p.messages, -3) || %{id: nil}).id,
               true <- upto != nil || {:error, "nothing to compact yet"}, :ok <- M.compact(m, tid, upto, text), do: {:ok, %{upto: upto, text: text}}
      end

    case result do
      {:ok, s} -> if json?(o), do: emit_json(s), else: out(good("summary installed") <> dim("  up to #{String.slice(s.upto, 0, 12)}") <> "\n" <> s.text); 0
      {:error, e} -> fail(e, 3)
    end
  end

  defp dispatch("set", [tid], o, m) do
    fields = [title: o[:title], system: o[:system], tools: o[:tools] && split(o[:tools]), budget: o[:budget], model: o[:model]] |> Enum.reject(fn {_, v} -> v == nil end)
    case M.set(m, tid, fields) do
      :ok -> if json?(o), do: emit_json(%{ok: true}), else: out(dim("set")); 0
      {:error, e} -> fail(e, 3)
    end
  end

  defp dispatch("search", words, o, m) when words != [] do
    hits = M.search(m, Enum.join(words, " "), 20)
    if json?(o), do: emit_json(hits), else: Enum.each(hits, &out("#{bold(String.slice(&1.id, 0, 10))} #{dim(&1.role)} #{dim(Enum.join(&1.threads, ","))}  #{String.replace(&1.snippet, "\n", " ")}"))
    if hits == [], do: 1, else: 0
  end

  defp dispatch("export", [tid], o, m) do
    case M.export(m, tid, if(o[:md], do: :markdown, else: :json)) do
      {:ok, text} -> IO.write(text); 0
      {:error, e} -> fail(e, 3)
    end
  end

  defp dispatch("import", [file], o, m) do
    with {:ok, data} <- read_input(file), {:ok, ids} <- M.import(m, data) do
      if json?(o), do: emit_json(%{threads: ids}), else: Enum.each(ids, &out/1)
      0
    else
      {:error, e} -> fail(e, 3)
    end
  end

  defp dispatch("share", [tid], o, m) do
    case M.share(m, tid) do
      {:ok, tok} -> if json?(o), do: emit_json(%{thread: tid, cap: tok, path: "/shared/#{tid}?cap=#{tok}"}), else: out("/shared/#{tid}?cap=#{tok}"); 0
      {:error, e} -> fail(e, 3)
    end
  end

  defp dispatch("revoke", [tid], o, m) do
    case M.revoke(m, tid) do
      :ok -> if json?(o), do: emit_json(%{ok: true}), else: out(dim("every link to #{tid} is dead")); 0
      {:error, e} -> fail(e, 3)
    end
  end

  defp dispatch("rm", [tid], o, m) do
    case M.delete(m, tid) do
      :ok -> if json?(o), do: emit_json(%{ok: true}), else: out(dim("deleted (vapor chat gc reclaims the space)")); 0
      {:error, e} -> fail(e, 3)
    end
  end

  defp dispatch("gc", [], o, m) do
    case M.gc(m) do
      {:ok, rep} -> if json?(o), do: emit_json(rep), else: out(dim("kept #{rep.kept}, dropped #{rep.dropped}; #{rep.bytes_before} → #{rep.bytes_after} bytes")); 0
      {:error, e} -> fail(e, 4)
    end
  end

  defp dispatch("tools", [], o, _m) do
    names = Vapor.Majlis.Tools.names()
    if json?(o), do: emit_json(names), else: Enum.each(names, &out/1)
    0
  end

  defp dispatch(_, _, _, _), do: (err(@usage); 2)

  # ------------------------------------------------------------------ output

  defp list(m, o) do
    ts = M.threads(m)

    if json?(o) do
      emit_json(ts)
    else
      if ts == [], do: out(dim("no threads yet: vapor chat new --title …"))
      Enum.each(ts, &out("#{bold(&1.id)}  #{&1.title}  #{dim("#{&1.messages} messages · #{&1.updated}")}" <> if(&1.preview != "", do: "\n    " <> dim(String.replace(&1.preview, "\n", " ")), else: "")))
    end

    0
  end

  defp answer(m, tid, o) do
    case M.reply(m, tid, gen(o)) do
      {:ok, r} -> show_reply(r, o)
      {:error, e} -> fail(e, 4)
    end
  end

  defp show_reply(r, o) do
    if json?(o) do
      emit_json(r)
    else
      out(r.content)
      steps = get_in(r.meta, ["agent", "steps"]) || []
      tail = if steps == [], do: "", else: " · tools: " <> Enum.map_join(steps, ", ", &"#{&1["tool"]}#{if &1["ok"], do: "", else: " ✗"}")
      out(dim("— #{String.slice(r.node, 0, 10)} · #{r.meta["backend"]} · #{r.meta["ms"]} ms#{tail}"))
    end

    0
  end

  defp message(n) do
    sib = if length(n.siblings) > 1, do: dim(" ‹#{n.index}/#{length(n.siblings)}›"), else: ""
    pin = if n.pinned, do: good(" ◆"), else: ""
    bold(String.pad_trailing(n.role, 9)) <> dim(String.slice(n.id, 0, 10)) <> sib <> pin <> "\n" <> n.content <> "\n"
  end

  defp tree_line(n) do
    "#{if n.on_path, do: good("●"), else: dim("○")} #{String.slice(n.id, 0, 10)} #{dim("← " <> String.slice(n.parent || "", 0, 10))} #{n.role}: " <>
      (n.content |> String.replace("\n", " ") |> String.slice(0, 70))
  end

  defp done_id(id, o), do: (if json?(o), do: emit_json(%{node: id}), else: out(id); 0)

  defp gen(o), do: [temperature: o[:temperature], max_tokens: o[:max_tokens], seed: o[:seed]] |> Enum.reject(fn {_, v} -> v == nil end)

  defp text_arg(["-"]), do: read_input("-")
  defp text_arg([]), do: {:error, "what to say? (a TEXT argument, or - for standard input)"}
  defp text_arg(words), do: {:ok, Enum.join(words, " ")}

  defp split(nil), do: []
  defp split(s), do: s |> String.split(",", trim: true) |> Enum.map(&String.trim/1)

  defp fail(e, code), do: (err("chat: #{if is_binary(e), do: e, else: inspect(e)}"); code)
end
