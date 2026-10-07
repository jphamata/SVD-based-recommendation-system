defmodule Vapor.TUI do
  @moduledoc """
  The terminal face of the console (`mix vapor.tui`): the same readers and
  measurements, in a line-oriented session that works over SSH, in a CI log
  or on a machine without a browser — no dependency, no curses.

      vapor> read scan.pdf          OCR of a picture or of a PDF's scanned pages
      vapor> add ./papers           ingest files into the session's library
      vapor> search multa contratual
      vapor> quality <text>         the calibrated noise gate
      vapor> chat new --title notas ; chat say t… "…"   conversations (vapor chat)
      vapor> wzn check energy.wzn | head 3              any vapor verb, with pipes and redirection
      vapor> lang pt | lang en      messages in Portuguese or English

  Every answer carries its measurement: OCR lines with their confidence (a
  `?` marks a line below 80 %), the passages with their scores, the gate's
  verdict. Every other line is `Vapor.Diwan`'s — the same interpreter as the
  console's terminal and `bin/vapor`, so the three answer alike. Colour
  (ANSI) only on a terminal and never with `NO_COLOR` set.

  `eval/2` is the whole interpreter — `(line, state) → {output, state}` —
  so it is tested without a terminal.
  """
  alias Vapor.Docs.Library

  @words %{
    en: %{help: "commands: read FILE · add PATH · search QUERY · quality TEXT · lang en|pt · quit — and every vapor verb (chat, wzn, alembic, athanor, rebis, aludel, tabula, …) with | > < ; (type: vapor help)",
          unknown: "unknown command — type help", conf: "confidence", check: "check this line",
          added: "added", passages: "passages", nothing: "nothing found",
          signal: "signal", noise: "noise", short: "too short to judge", bye: "bye", lib_empty: "the library is empty — add PATH first", no_text: "no text found"},
    pt: %{help: "comandos: read ARQUIVO · add CAMINHO · search CONSULTA · quality TEXTO · lang en|pt · quit — e todo verbo do vapor (chat, wzn, alembic, athanor, rebis, aludel, tabula, …) com | > < ; (digite: vapor help)",
          unknown: "comando desconhecido — digite help", conf: "confiança", check: "confira esta linha",
          added: "adicionado", passages: "passagens", nothing: "nada encontrado",
          signal: "sinal", noise: "ruído", short: "curto demais para julgar", bye: "até logo", lib_empty: "a biblioteca está vazia — use add CAMINHO antes", no_text: "nenhum texto encontrado"}
  }

  @doc """
  A fresh session: `%{lang, tty, color, library, worker}`. On a terminal
  (`tty`) the loop shows a prompt; in a pipe it reads lines and writes only
  answers.
  """
  def new(opts \\ []) do
    tty = Keyword.get_lazy(opts, :tty, &tty?/0)
    %{lang: Keyword.get(opts, :lang, :en), tty: tty, color: Keyword.get(opts, :color, tty and System.get_env("NO_COLOR") in [nil, ""]),
      library: Library.new(), worker: Keyword.get(opts, :worker)}
  end

  defp tty?, do: match?({:ok, true}, :io.getopts(:standard_io) |> Keyword.fetch(:terminal))

  @doc "The interactive loop on standard I/O, from a session."
  def loop(st) do
    case IO.gets(if st[:tty], do: "vapor> ", else: "") do
      :eof -> :ok
      {:error, _} -> :ok
      line ->
        case eval(String.trim(line), st) do
          {:quit, out} -> if st[:tty], do: IO.puts(out)
          {[], st} -> loop(st)
          {out, st} -> IO.puts(out); loop(st)
        end
    end
  end

  defp w(st, k), do: @words[st.lang][k]

  @doc "Interpret one line: `{output_iodata, state}` or `{:quit, output}`."
  def eval(line, st) do
    case String.split(line, " ", parts: 2) do
      [""] -> {[], st}
      ["help"] -> {w(st, :help), st}
      [q] when q in ["quit", "exit", "q"] -> {:quit, w(st, :bye)}
      ["lang", l] when l in ["en", "pt"] -> {"ok", %{st | lang: String.to_atom(l)}}
      ["read", path] -> {read(String.trim(path), st), st}
      ["add", path] -> add(String.trim(path), st)
      ["search", q] -> {search(q, st), st}
      ["quality", text] -> {quality(text, st), st}
      _ -> diwan(line, st)
    end
  rescue
    e -> {paint(st, :red, "error: " <> Exception.message(e)), st}
  catch
    {:refused, msg} -> {paint(st, :red, msg), st}
  end

  # ------------------------------------------------------------- commands --

  # every other line: the same interpreter as the console's terminal, on the
  # person's own files (not jailed: this is their terminal)
  defp diwan(line, st) do
    d = st[:diwan] || Vapor.Diwan.new(jail: false, tty: st.tty)
    {r, d} = Vapor.Diwan.eval(line, d)
    out = String.trim_trailing(r.out <> r.err)
    out = if r.code in [0, 1], do: out, else: out <> "\n(exit #{r.code})"
    {if(r.codes == [2] and r.err =~ "unknown command", do: w(st, :unknown), else: out), Map.put(st, :diwan, d)}
  end

  defp read(path, st) do
    bytes = File.read!(path)

    pictures =
      case Vapor.Docs.sniff(Path.basename(path), bytes) do
        :pdf ->
          {:ok, pages, _} = ok!(Vapor.Docs.PDF.pages(bytes))
          empty = for {t, i} <- Enum.with_index(pages, 1), String.trim(t) == "", do: i
          {:ok, imgs, _} = ok!(Vapor.Docs.PDF.images(bytes, empty))
          for {i, list} <- Enum.sort(imgs), img <- list, do: {"page #{i}", img}

        kind when kind in [:png, :jpeg, :pnm] ->
          {:ok, pic} = ok!(Vapor.Docs.Pictures.read(kind, bytes))
          if pic[:image], do: [{Path.basename(path), pic.image}], else: throw({:refused, Enum.join(List.wrap(pic[:warnings]), "; ")})

        other ->
          throw({:refused, "#{other}: not a picture or a PDF"})
      end

    for {label, img} <- pictures do
      {:ok, r} = ok!(Vapor.Vision.OCR.read(img, worker: worker(st)))
      head =
        if r.lines == [],
          do: paint(st, :dim, "── #{label} · #{w(st, :no_text)}"),
          else: paint(st, :dim, "── #{label} · #{w(st, :conf)} #{pct(r.confidence)}")

      lines =
        for l <- r.lines do
          flag = if l.confidence < 0.8, do: paint(st, :yellow, " ? "), else: "   "
          [flag, l.text, if(l.confidence < 0.8, do: paint(st, :dim, "  (#{w(st, :check)}, #{pct(l.confidence)})"), else: "")]
        end

      Enum.intersperse([head | lines], "\n")
    end
    |> Enum.intersperse("\n")
  end

  defp add(path, st) do
    files = if File.dir?(path), do: Path.wildcard(Path.join(path, "**/*")) |> Enum.filter(&File.regular?/1), else: [path]

    {lib, outs} =
      Enum.reduce(files, {st.library, []}, fn f, {lib, outs} ->
        case Library.add(lib, f) do
          {:ok, lib2, rep} -> {lib2, ["#{Path.basename(f)}: #{w(st, :added)}, #{rep.added} #{w(st, :passages)}" <> warn(rep) | outs]}
          {:error, %Vapor.Rejection{bound: b}} -> {lib, [paint(st, :red, "#{Path.basename(f)}: #{b}") | outs]}
        end
      end)

    {outs |> Enum.reverse() |> Enum.intersperse("\n"), %{st | library: lib}}
  end

  defp warn(%{warnings: [_ | _] = ws}), do: " · " <> Enum.join(ws, "; ")
  defp warn(_), do: ""

  defp search(q, st) do
    if Library.stats(st.library).files == 0, do: throw({:refused, w(st, :lib_empty)})
    r = Library.search(st.library, q, k: 5)

    case r.hits do
      [] -> w(st, :nothing)
      hits ->
        top = hd(hits).score
        Enum.map(hits, fn h ->
          [paint(st, :cyan, h.doc), "  ", meter(h.score / top, st), "\n  ", h.text |> String.replace("\n", " ") |> String.slice(0, 160), "\n"]
        end) ++ [paint(st, :dim, "receipt #{r.library_receipt}")]
    end
  end

  defp quality(text, st) do
    {ref, hold} = Vapor.Quality.Suite.corpus_pt_raw()
    p = Vapor.Quality.Text.profile(ref)
    {:ok, g} = Vapor.Quality.Text.gate(p, hold, len: 120, count: 24)
    j = Vapor.Quality.Text.judge(text, p, g)

    case j.verdict do
      :pass -> paint(st, :green, w(st, :signal) <> " (#{j.noise})")
      :fail -> paint(st, :red, w(st, :noise))
      _ -> paint(st, :yellow, w(st, :short))
    end
  end

  # -------------------------------------------------------------- drawing --

  # an 8×8 image (0–16) as 8 rows of 2-character cells on the 24-step grey ramp

  defp meter(x, st) do
    n = round(min(max(x, 0.0), 1.0) * 20)
    paint(st, :cyan, String.duplicate("█", n)) <> paint(st, :dim, String.duplicate("·", 20 - n))
  end

  defp pct(x, decimals \\ 0), do: :erlang.float_to_binary(x * 100.0, decimals: decimals) <> " %"

  @codes %{red: 31, green: 32, yellow: 33, cyan: 36, dim: 2}
  defp paint(%{color: true}, c, s), do: "\e[#{@codes[c]}m#{s}\e[0m"
  defp paint(_, _c, s), do: s

  defp worker(%{worker: w}) when w != nil, do: w
  defp worker(_), do: Vapor.Vision.OCR.worker()

  defp ok!({:ok, _} = v), do: v
  defp ok!({:ok, _, _} = v), do: v
  defp ok!({:ok, _, _, _} = v), do: v
  defp ok!({:error, %Vapor.Rejection{} = r}), do: throw({:refused, "refused: #{r.bound}"})
  defp ok!({:error, e}), do: throw({:refused, "refused: #{inspect(e)}"})
end
