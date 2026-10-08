defmodule Vapor.LSP do
  @moduledoc """
  `vapor lsp` — a Language Server (LSP 3.17, JSON-RPC over stdio) for
  vapor's languages, so any editor that speaks LSP — VS Code, Neovim,
  Emacs (eglot), Helix, Zed, Kakoune — gets the same help (docs/EDITORS.md):

  | | Almizan (`.wzn`) | Alembic (`.nbq`) |
  |---|---|---|
  | diagnostics | syntax and the morphological rules as you type; on open and save, every obligation **decided** — a refuted claim is an error at its line, with the point that refutes it | parse errors with line and column |
  | hover | a claim's verdict and decider; a root's meaning and abjad value; a keyword in both scripts | a builtin's signature |
  | completion | keywords in the file's script, roots, claims defined above | builtins and constants |
  | symbols, go to definition | claims | definitions |
  | formatting | the canonical printing, in the file's script | — |
  | inlay hints | each proved claim's verdict and decider at the end of its first line (`✓ proved · SAT + DRUP`) | — |
  | commands | `vapor.almizan.toArabic` / `vapor.almizan.toLatin`: the same program in the other script (the manifesto's projection, as an edit the person chooses) | — |

  `handle/2` is the whole server — a message and the state in, messages out
  — so it is tested without an editor; `serve/0` frames it on stdio
  (`Content-Length` headers, bytes).
  """
  alias Vapor.{Alembic, Almizan}
  alias Vapor.Almizan.{Abjad, Syntax}

  @kw_latin ~w(claim root wazn inputs field box step init invariant proof body import as fail maful burhan conserved nonneg pos identity bounded q int f64 f32 bool and or not if true false
               graph do independent identifiable adjustment separated)

  # ------------------------------------------------------------------ stdio

  @doc "Serve on standard input/output until `exit`."
  def serve do
    :io.setopts(:standard_io, [:binary, encoding: :latin1])
    loop(%{docs: %{}, shutdown: false})
  end

  defp loop(st) do
    case read_message() do
      {:ok, msg} ->
        {out, st} = handle(msg, st)
        Enum.each(out, &write_message/1)
        if msg["method"] == "exit", do: :ok, else: loop(st)

      :eof ->
        :ok
    end
  end

  defp read_message do
    case read_headers(%{}) do
      {:ok, %{"content-length" => n}} ->
        case IO.binread(:stdio, String.to_integer(n)) do
          data when is_binary(data) ->
            case Vapor.JSON.decode(data) do
              {:ok, m} -> {:ok, m}
              _ -> {:ok, %{"jsonrpc" => "2.0", "method" => "$/invalid"}}
            end

          _ ->
            :eof
        end

      _ ->
        :eof
    end
  end

  defp read_headers(acc) do
    case IO.binread(:stdio, :line) do
      line when is_binary(line) ->
        case String.trim(line) do
          "" -> {:ok, acc}
          h ->
            case String.split(h, ":", parts: 2) do
              [k, v] -> read_headers(Map.put(acc, String.downcase(String.trim(k)), String.trim(v)))
              _ -> read_headers(acc)
            end
        end

      _ ->
        :eof
    end
  end

  defp write_message(m) do
    body = Vapor.JSON.encode(m)
    IO.binwrite(:stdio, ["Content-Length: ", Integer.to_string(byte_size(body)), "\r\n\r\n", body])
  end

  # ---------------------------------------------------------------- protocol

  @doc "One message → `{[messages to send], state}`."
  def handle(%{"method" => "initialize", "id" => id}, st) do
    caps = %{
      "textDocumentSync" => %{"openClose" => true, "change" => 1, "save" => %{"includeText" => true}},
      "hoverProvider" => true,
      "completionProvider" => %{"triggerCharacters" => ["(", " "]},
      "documentSymbolProvider" => true,
      "definitionProvider" => true,
      "documentFormattingProvider" => true,
      "inlayHintProvider" => true,
      "executeCommandProvider" => %{"commands" => ["vapor.almizan.toArabic", "vapor.almizan.toLatin"]}
    }

    {[reply(id, %{"capabilities" => caps, "serverInfo" => %{"name" => "vapor", "version" => to_string(Application.spec(:vapor, :vsn) || "dev")}})], st}
  end

  def handle(%{"method" => "shutdown", "id" => id}, st), do: {[reply(id, nil)], %{st | shutdown: true}}
  def handle(%{"method" => m}, st) when m in ["initialized", "exit", "$/cancelRequest", "$/setTrace", "$/invalid"], do: {[], st}

  def handle(%{"method" => "textDocument/didOpen", "params" => %{"textDocument" => d}}, st) do
    st = put_in(st, [:docs, d["uri"]], %{text: d["text"], lang: lang(d["uri"], d["languageId"])})
    {[diagnostics(d["uri"], st, true)], st}
  end

  def handle(%{"method" => "textDocument/didChange", "params" => %{"textDocument" => d, "contentChanges" => changes}}, st) do
    case List.last(changes) do
      %{"text" => text} ->
        st = update_in(st, [:docs, d["uri"]], &Map.put(&1 || %{lang: lang(d["uri"], nil)}, :text, text))
        {[diagnostics(d["uri"], st, false)], st}

      _ ->
        {[], st}
    end
  end

  def handle(%{"method" => "textDocument/didSave", "params" => %{"textDocument" => d} = p}, st) do
    st = if p["text"], do: update_in(st, [:docs, d["uri"]], &Map.put(&1, :text, p["text"])), else: st
    {[diagnostics(d["uri"], st, true)], st}
  end

  def handle(%{"method" => "textDocument/didClose", "params" => %{"textDocument" => d}}, st) do
    {[notify("textDocument/publishDiagnostics", %{"uri" => d["uri"], "diagnostics" => []})], %{st | docs: Map.delete(st.docs, d["uri"])}}
  end

  def handle(%{"method" => "textDocument/hover", "id" => id, "params" => p}, st) do
    {[reply(id, with_doc(st, p, &hover/3))], st}
  end

  def handle(%{"method" => "textDocument/completion", "id" => id, "params" => p}, st) do
    {[reply(id, with_doc(st, p, &completion/3) || [])], st}
  end

  # the verdict of every proved claim, at the end of its first line: the evidence beside the code
  def handle(%{"method" => "textDocument/inlayHint", "id" => id, "params" => p}, st) do
    {[reply(id, with_doc(st, p, fn doc, _pos, _uri -> hints(doc) end) || [])], st}
  end

  def handle(%{"method" => "textDocument/documentSymbol", "id" => id, "params" => p}, st) do
    {[reply(id, with_doc(st, p, fn doc, _pos, _uri -> symbols(doc) end) || [])], st}
  end

  def handle(%{"method" => "textDocument/definition", "id" => id, "params" => p}, st) do
    {[reply(id, with_doc(st, p, &definition/3))], st}
  end

  def handle(%{"method" => "textDocument/formatting", "id" => id, "params" => %{"textDocument" => %{"uri" => uri}}}, st) do
    edits =
      case st.docs[uri] do
        # comments kept (glosses): the printer alone would drop them
        %{lang: :almizan, text: text} ->
          case Vapor.Almizan.Format.format(text) do
            {:ok, ^text} -> []
            {:ok, canonical} -> [%{"range" => whole(text), "newText" => canonical}]
            _ -> []
          end

        _ -> []
      end

    {[reply(id, edits)], st}
  end

  def handle(%{"method" => "workspace/executeCommand", "id" => id, "params" => %{"command" => cmd, "arguments" => [uri | _]}}, st)
      when cmd in ["vapor.almizan.toArabic", "vapor.almizan.toLatin"] do
    case st.docs[uri] do
      %{text: text} ->
        proj = if cmd == "vapor.almizan.toArabic", do: :arabic, else: :latin

        case Vapor.Almizan.Format.format(text, proj) do
          {:ok, projected} ->
            edit = %{"changes" => %{uri => [%{"range" => whole(text), "newText" => projected}]}}
            {[reply(id, nil), request("workspace/applyEdit", %{"label" => "Almizan: #{proj}", "edit" => edit})], st}

          {:error, why} ->
            {[error(id, -32602, why)], st}
        end

      nil ->
        {[error(id, -32602, "the document is not open")], st}
    end
  end

  def handle(%{"id" => id, "method" => m}, st), do: {[error(id, -32601, "method not found: #{m}")], st}
  def handle(%{"id" => _id, "result" => _}, st), do: {[], st}
  def handle(_, st), do: {[], st}

  # ---------------------------------------------------------------- features

  defp lang(uri, lid) do
    cond do
      lid in ["almizan", "wzn"] or String.ends_with?(uri, ".wzn") -> :almizan
      lid == "alembic" or String.ends_with?(uri, ".nbq") -> :alembic
      true -> :other
    end
  end

  defp with_doc(st, %{"textDocument" => %{"uri" => uri}} = p, f) do
    case st.docs[uri] do
      nil -> nil
      doc -> f.(doc, p["position"], uri)
    end
  end

  # diagnostics: syntax always; with `decide`, the obligations too (they may take a moment)
  defp diagnostics(uri, st, decide) do
    doc = st.docs[uri]
    notify("textDocument/publishDiagnostics", %{"uri" => uri, "diagnostics" => diags(doc, decide)})
  end

  defp diags(%{lang: :almizan, text: text}, decide) do
    case Almizan.parse(text) do
      {:ok, m} ->
        if decide do
          lines = claim_lines(text)

          for r <- Almizan.check(m, depth: 14), r.verdict in ["refuted", "unknown"] do
            line = Map.get(lines, r.claim, 1)
            cex = if r[:counterexample], do: " — at " <> Enum.map_join(r.counterexample, ", ", fn {k, v} -> "#{k} = #{Almizan.show(v)}" end), else: ""
            diag(line, if(r.verdict == "refuted", do: 1, else: 2), "#{r.claim}: #{r.verdict} (#{r.decider}) #{r.detail}#{cex}", text)
          end
        else
          []
        end

      {:error, why} ->
        line = case Regex.run(~r/^line (\d+)/, why) do [_, l] -> String.to_integer(l); _ -> 1 end
        [diag(line, 1, why, text)]
    end
  end

  defp diags(%{lang: :alembic, text: text}, _decide) do
    case Alembic.load(text) do
      {:ok, _} -> []
      {:error, %{message: msg, line: l, col: c}} -> [diag(max(l, 1), 1, msg, text, max(c - 1, 0))]
      {:error, other} -> [diag(1, 1, Alembic.format_error(other), text)]
    end
  end

  defp diags(_, _), do: []

  defp diag(line, severity, msg, text, col \\ 0) do
    l = max(line - 1, 0)
    len = text |> String.split("\n") |> Enum.at(l, "") |> utf16_len()
    %{"range" => %{"start" => %{"line" => l, "character" => min(col, len)}, "end" => %{"line" => l, "character" => len}},
      "severity" => severity, "source" => "vapor", "message" => msg}
  end

  # LSP counts UTF-16 code units
  defp utf16_len(s), do: s |> :unicode.characters_to_binary(:utf8, :utf16) |> byte_size() |> div(2)

  defp whole(text) do
    lines = String.split(text, "\n")
    %{"start" => %{"line" => 0, "character" => 0}, "end" => %{"line" => length(lines), "character" => 0}}
  end

  # the word under the cursor (letters of any script, digits, - . / @ and the Buckwalter marks)
  defp word_at(text, %{"line" => l, "character" => ch}) do
    line = text |> String.split("\n") |> Enum.at(l, "")
    units = :unicode.characters_to_binary(line, :utf8, :utf16)
    {before, after_} = :erlang.split_binary(units, min(ch * 2, byte_size(units)))
    b = :unicode.characters_to_binary(before, :utf16, :utf8)
    a = :unicode.characters_to_binary(after_, :utf16, :utf8)
    left = case Regex.run(~r/[^\s()]*$/u, b) do [w] -> w; _ -> "" end
    right = case Regex.run(~r/^[^\s()]*/u, a) do [w] -> w; _ -> "" end
    left <> right
  end

  defp hover(%{lang: :almizan, text: text}, pos, _uri) do
    w = word_at(text, pos)

    info =
      cond do
        id = Syntax.root(w) ->
          {l, a} = Syntax.roots()[id]
          meaning = %{"hsb" => "arithmetic", "hfz" => "conservation (invariance)", "nql" => "transition", "ktb" => "record"}[id]
          "**root** `#{l}` · `#{a}` — #{meaning}\n\nabjad value #{Abjad.value(a)} (a value, not an address: #{Abjad.collisions().sharing} of #{Abjad.collisions().roots} roots share theirs)"

        k = Syntax.keyword(w) ->
          "**#{k}** · `#{Syntax.arabic(k)}`"

        true ->
          case Almizan.parse(text) do
            {:ok, m} ->
              case Enum.find(Almizan.check(m, depth: 12), &(&1.claim == w)) do
                nil -> nil
                r -> "**#{r.claim}** (#{r.root}, #{r.wazn}) — **#{r.verdict}**#{if r.decider, do: " by #{r.decider}", else: ""}\n\n#{r.detail}"
              end

            _ -> nil
          end
      end

    info && %{"contents" => %{"kind" => "markdown", "value" => info}}
  end

  defp hover(%{lang: :alembic, text: text}, pos, _uri) do
    w = word_at(text, pos)
    if w in Vapor.Alembic.Builtins.names(), do: %{"contents" => %{"kind" => "markdown", "value" => "`#{w}` — Alembic builtin (see `vapor alembic --card`)"}}
  end

  defp hover(_, _, _), do: nil

  defp completion(%{lang: :almizan, text: text}, _pos, _uri) do
    arabic = Syntax.projection(text) == :arabic
    kws = Enum.map(@kw_latin, &if(arabic, do: Syntax.arabic(&1), else: &1))
    roots = Enum.map(Syntax.roots(), fn {_, {l, a}} -> if arabic, do: a, else: l end)
    claims = claim_lines(text) |> Map.keys()
    Enum.map(kws, &item(&1, 14)) ++ Enum.map(roots, &item(&1, 20)) ++ Enum.map(claims, &item(&1, 3))
  end

  defp completion(%{lang: :alembic}, _pos, _uri), do: Enum.map(Vapor.Alembic.Builtins.names(), &item(&1, 3))
  defp completion(_, _, _), do: []

  defp item(label, kind), do: %{"label" => label, "kind" => kind}

  defp symbols(%{lang: :almizan, text: text}) do
    for {name, line} <- claim_lines(text) do
      r = %{"start" => %{"line" => line - 1, "character" => 0}, "end" => %{"line" => line - 1, "character" => 0}}
      %{"name" => name, "kind" => 12, "range" => r, "selectionRange" => r}
    end
  end

  defp symbols(%{lang: :alembic, text: text}) do
    case Alembic.load(text) do
      {:ok, p} ->
        for d <- p.defs do
          r = %{"start" => %{"line" => d.line - 1, "character" => 0}, "end" => %{"line" => d.line - 1, "character" => 0}}
          %{"name" => d.name, "kind" => 12, "range" => r, "selectionRange" => r}
        end

      _ -> []
    end
  end

  defp symbols(_), do: []

  defp definition(%{lang: :almizan, text: text}, pos, uri) do
    w = word_at(text, pos)
    name = case Syntax.ident(w) do {:ok, n} -> n; _ -> w end

    case claim_lines(text)[name] do
      nil -> nil
      line -> %{"uri" => uri, "range" => %{"start" => %{"line" => line - 1, "character" => 0}, "end" => %{"line" => line - 1, "character" => 0}}}
    end
  end

  defp definition(_, _, _), do: nil

  # claim name → line, from the reader (positions survive even when the checker refuses)
  defp hints(%{lang: :almizan, text: text}) do
    with {:ok, m} <- Almizan.parse(text) do
      lines = claim_lines(text)
      rows = String.split(text, "\n")

      for r <- Almizan.check(m, depth: 12), r.verdict != "none", line = lines[r.claim], line != nil do
        width = rows |> Enum.at(line - 1, "") |> :unicode.characters_to_binary(:utf8, {:utf16, :little}) |> byte_size() |> div(2)
        mark = case r.verdict do "proved" -> "✓ proved"; "refuted" -> "✗ refuted"; v -> "? " <> v end
        %{"position" => %{"line" => line - 1, "character" => width}, "label" => "#{mark} · #{r.decider}", "paddingLeft" => true, "tooltip" => r.detail}
      end
    else
      _ -> []
    end
  end

  defp hints(_), do: []

  defp claim_lines(text) do
    case Syntax.read(text) do
      {:ok, forms} ->
        for {:list, [{:atom, kw, _}, {:atom, name, _} | _], line} <- forms, Syntax.keyword(kw) == "claim", into: %{} do
          {case Syntax.ident(name) do {:ok, n} -> n; _ -> name end, line}
        end

      _ -> %{}
    end
  end

  # ------------------------------------------------------------------ framing

  defp reply(id, result), do: %{"jsonrpc" => "2.0", "id" => id, "result" => result}
  defp error(id, code, msg), do: %{"jsonrpc" => "2.0", "id" => id, "error" => %{"code" => code, "message" => msg}}
  defp notify(method, params), do: %{"jsonrpc" => "2.0", "method" => method, "params" => params}
  defp request(method, params), do: %{"jsonrpc" => "2.0", "id" => "vapor-#{System.unique_integer([:positive])}", "method" => method, "params" => params}
end
