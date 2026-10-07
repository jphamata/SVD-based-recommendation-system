defmodule Vapor.TUI do
  @moduledoc """
  The terminal face of the console (`mix vapor.tui`): the same readers and
  measurements, in a line-oriented session that works over SSH, in a CI log
  or on a machine without a browser — no dependency, no curses.

      vapor> read scan.pdf          OCR of a picture or of a PDF's scanned pages
      vapor> listen seven.wav       a spoken digit
      vapor> draw 7 3               a handwritten 7 by diffusion (seed 3), drawn in the terminal
      vapor> add ./papers           ingest files into the session's library
      vapor> search multa contratual
      vapor> quality <text>         the calibrated noise gate
      vapor> merge                  the fusion laboratory (regime, every method measured)
      vapor> lang pt | lang en      messages in Portuguese or English

  Every answer carries its measurement: OCR lines with their confidence (a
  `?` marks a line below 80 %), the speech reader's certainty, the drawn
  digit read back by the real-data classifier with its distance to the
  nearest training image. Colour (ANSI) only on a terminal and never with
  `NO_COLOR` set; otherwise plain text.

  `eval/2` is the whole interpreter — `(line, state) → {output, state}` —
  so it is tested without a terminal.
  """
  alias Vapor.Docs.Library

  @words %{
    en: %{help: "commands: read FILE · listen FILE.wav · draw DIGIT [SEED] · add PATH · search QUERY · quality TEXT · merge · athanor run FILE · game FILE solve · crucible KIND FILE · assay TOOL FILE · alembic -e EXPR · lang en|pt · quit",
          unknown: "unknown command — type help", conf: "confidence", check: "check this line", certain: "certain",
          reads: "the real-data classifier reads", nearest: "distance to the nearest training image", copy: "a copy would be ≈ 0",
          added: "added", passages: "passages", nothing: "nothing found", regime: "regime", chosen: "chosen on validation",
          signal: "signal", noise: "noise", short: "too short to judge", bye: "bye", lib_empty: "the library is empty — add PATH first", no_text: "no text found"},
    pt: %{help: "comandos: read ARQUIVO · listen ARQUIVO.wav · draw DÍGITO [SEMENTE] · add CAMINHO · search CONSULTA · quality TEXTO · merge · athanor run ARQUIVO · game ARQUIVO solve · crucible TIPO ARQUIVO · assay FERRAMENTA ARQUIVO · alembic -e EXPR · lang en|pt · quit",
          unknown: "comando desconhecido — digite help", conf: "confiança", check: "confira esta linha", certain: "de certeza",
          reads: "o classificador de dados reais lê", nearest: "distância à imagem de treino mais próxima", copy: "uma cópia daria ≈ 0",
          added: "adicionado", passages: "passagens", nothing: "nada encontrado", regime: "regime", chosen: "escolhido na validação",
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
      ["listen", path] -> {listen(String.trim(path), st), st}
      ["draw" | rest] -> {draw(rest, st), st}
      ["add", path] -> add(String.trim(path), st)
      ["search", q] -> {search(q, st), st}
      ["quality", text] -> {quality(text, st), st}
      ["merge"] -> {merge(st), st}
      [verb, rest] when verb in ~w(alembic athanor search game crucible assay mind scene solve verify) -> {bench(verb, rest), st}
      [verb] when verb in ~w(crucible assay) -> {bench(verb, ""), st}
      _ -> {w(st, :unknown), st}
    end
  rescue
    e -> {paint(st, :red, "error: " <> Exception.message(e)), st}
  catch
    {:refused, msg} -> {paint(st, :red, msg), st}
  end

  # ------------------------------------------------------------- commands --

  # the open bench (0.14): the command line's verbs, their standard output captured
  defp bench(verb, rest) do
    args = [verb | OptionParser.split(rest)]
    {:ok, io} = StringIO.open("")
    old = Process.group_leader()
    Process.group_leader(self(), io)
    code =
      try do
        Vapor.Main.run(args)
      after
        Process.group_leader(self(), old)
      end
    {:ok, {_, out}} = StringIO.close(io)
    String.trim_trailing(out) <> if(code in [0, 1], do: "", else: "\n(exit #{code})")
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

  defp listen(path, st) do
    {:ok, clip} = ok!(Vapor.Modal.Audio.read(path))
    {:ok, model} = ok!(Vapor.Modal.Speech.load())
    {:ok, r} = Vapor.Modal.Speech.classify(clip, model, worker: worker(st))
    color = if r.p >= 0.6, do: :green, else: :yellow
    [paint(st, color, r.label), "  ", pct(r.p, 1), " ", w(st, :certain), "\n", bar_line(r.probs, st)]
  end

  defp draw(args, st) do
    {d, seed} =
      case Enum.flat_map(args, &String.split/1) |> Enum.map(&Integer.parse/1) do
        [{d, ""}] -> {d, 0}
        [{d, ""}, {s, ""}] -> {d, s}
        _ -> throw({:refused, "draw DIGIT [SEED]"})
      end

    unless d in 0..9, do: throw({:refused, "draw DIGIT [SEED] — a digit from 0 to 9"})
    r = Vapor.Modal.Digits.draw(d, 1, seed: seed, worker: worker(st))
    [reading] = r.readings
    color = if reading.digit == d, do: :green, else: :red

    [pixels(hd(r.images), st), "\n", w(st, :reads), " ", paint(st, color, Integer.to_string(reading.digit)), " (", pct(reading.p, 1), ")  ·  ",
     w(st, :nearest), " ", :erlang.float_to_binary(hd(r.nearest), decimals: 1), " (", w(st, :copy), ")"]
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

  defp merge(st) do
    r = Vapor.Quality.Suite.merge_real(worker: worker(st))

    for {label, part} <- [{"fine-tunes of one base", r.fine_tune}, {"trained separately", r.independent}] do
      rows = Enum.map(part.rows, fn x -> "  #{String.pad_trailing(x.method, 24)} #{:erlang.float_to_binary(x.mean, decimals: 3)} bits/char" <> if(x.method == part.chosen, do: paint(st, :green, "  ← #{w(st, :chosen)}"), else: "") end)
      [paint(st, :cyan, "#{label} · #{w(st, :regime)} #{part.diag.regime}"), "\n", Enum.intersperse(rows, "\n"), "\n"]
    end
  end

  # -------------------------------------------------------------- drawing --

  # an 8×8 image (0–16) as 8 rows of 2-character cells on the 24-step grey ramp
  defp pixels(levels, st) do
    levels
    |> Enum.chunk_every(8)
    |> Enum.map(fn row ->
      for v <- row do
        g = round(min(max(v, 0.0), 16.0) / 16 * 23)
        if st.color, do: "\e[48;5;#{232 + g}m  \e[0m", else: String.at(" .:-=+*#%@", min(div(g * 10, 24), 9)) |> String.duplicate(2)
      end
    end)
    |> Enum.intersperse("\n")
  end

  defp bar_line(probs, st) do
    probs
    |> Enum.with_index()
    |> Enum.map(fn {p, i} -> "#{i} #{meter(p, st)} #{pct(p)}" end)
    |> Enum.intersperse("\n")
  end

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
