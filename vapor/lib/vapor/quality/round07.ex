defmodule Vapor.Quality.Round07 do
  @moduledoc """
  Quality checks for the 0.7 round — the scanned office page, structured
  output with patterns, fusion from disk — in the suite's discipline: every
  check has a value, a **control** that a broken implementation (or the
  naive one) would produce, and a threshold that separates them.

  | check | value | control (must fail) |
  |---|---|---|
  | CCITT fax | committed Group 4 / Group 3 2-D / MH streams decoded to libtiff's bitmap (SHA-256) | the same streams decoded with the coding (K) misdeclared |
  | reading order | mean CER of the eight scanned pages (2–3 columns, title, footer) through PDF → CCITT → layout → reader | the same pages with the layout cut off (lines across columns) |
  | language model | CER of the held-out lines, beam search with the model | the same with a model of the shuffled corpus (same letters, no language) |
  | the model abstains | lines of random strings whose reading the model changed | the same without the abstention gate |
  | codes on a page | CER of lines mixing words with amounts, dates and IDs, with the model | greedy decoding (the model must not make them worse) |
  | formats | generated `date`/`date-time`/`ipv4`/`uuid` values that OTP's own parsers refuse | a naive pattern (`\\d{4}-\\d{2}-\\d{2}`) |
  | fusion from disk | `Merge.stream/3`'s output root against `merge/2`'s; its peak memory against the models' size | the roots of a fusion at another `t` (the comparison sees a difference) |
  """
  alias Vapor.Docs.{CCITT, PDF}
  alias Vapor.Grammar
  alias Vapor.Grammar.Regex, as: Pattern
  alias Vapor.Vision.{CharLM, OCR}

  @priv "priv/quality"

  def run(opts \\ []) do
    w = Keyword.get(opts, :worker)
    checks = [ccitt(), formats(), fusion(), layout(w), language(w)] |> List.flatten()
    %{checks: checks}
  end

  defp check(name, value, control, threshold, pass), do: %{name: name, value: value, control: control, threshold: threshold, pass: pass}
  # priv/quality/… from the application; test fixtures from the checkout (absent in a release: skipped)
  defp p("priv/" <> rel), do: Path.join(to_string(:code.priv_dir(:vapor)), rel)
  defp p(rel), do: Path.expand(rel)

  # ------------------------------------------------------------- CCITT --

  @streams [{"page_g4.bin", -1, 1240, "5fb0679aa971c843695ce37ef1fb49bb6c91ec1918eb0918ca3c9623b56c0b87"},
            {"runs_g3_2dfill.bin", 1, 2700, "12f8edc0e6f92557ac4613cc958cad2f051154a7bfc947f7aeda148b40496ddc"},
            {"noise5_g3.bin", 0, 101, "318cd021cc515c06095230083aa6abed94f408768f305b9bcd213d7c32c26ade"}]

  defp ccitt do
    dir = p("test/fixtures/docs/ccitt")

    if File.dir?(dir) do
      run = fn kfun ->
        Enum.count(@streams, fn {f, k, cols, sha} ->
          {:ok, r} = CCITT.decode(File.read!(Path.join(dir, f)), k: kfun.(k), columns: cols, black_is_1: true)
          Base.encode16(:crypto.hash(:sha256, r.data), case: :lower) == sha
        end)
      end

      good = run.(& &1)
      bad = run.(fn k -> if k < 0, do: 0, else: -1 end)
      check("CCITT fax: streams decoded to libtiff's bitmap (G4, G3 2-D, G3 1-D)", "#{good}/3", "#{bad}/3 (K misdeclared)", "3/3, control 0/3",
            good == 3 and bad == 0)
    else
      []
    end
  end

  # ------------------------------------------------------------ formats --

  defp formats do
    :rand.seed(:exsss, {7, 0, 7})
    valid = %{
      "date" => fn s -> match?({:ok, _}, Date.from_iso8601(s)) end,
      "ipv4" => fn s -> match?({:ok, _}, :inet.parse_strict_address(String.to_charlist(s))) and not Regex.match?(~r/(^|\.)0\d/, s) end,
      "uuid" => fn s -> Regex.match?(~r/^[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$/, s) end,
      # RFC 3339 §5.6: "t" and "z" may be lower case, and a leap second is :60 — OTP's parser takes neither
      "date-time" => fn s -> match?({:ok, _, _}, s |> String.upcase() |> String.replace(~r/:60(?=[.Z+-])/, ":59") |> DateTime.from_iso8601()) end
    }

    gen = fn pattern, n ->
      {:ok, ast} = Pattern.parse(pattern)
      for _ <- 1..n, do: List.to_string(sample(ast))
    end

    bad = Enum.sum(for {f, ok?} <- valid, s <- gen.(Pattern.formats()[f], 150), not ok?.(s), do: 1)
    naive = Enum.count(gen.("^\\d{4}-\\d{2}-\\d{2}$", 150), &(not valid["date"].(&1)))

    # and the grammar admits exactly its pattern: a sampled value is accepted, its corruption refused
    {:ok, g} = Pattern.json_format("date")
    acc = fn s -> match?({:ok, g2} when is_struct(g2), Grammar.advance(Grammar.new(g), Vapor.JSON.encode(s))) and Grammar.complete?(elem(Grammar.advance(Grammar.new(g), Vapor.JSON.encode(s)), 1)) end
    ok = acc.("2024-02-29") and not acc.("2023-02-29") and not acc.("2024-13-01")

    check("formats: generated date / date-time / ipv4 / uuid that OTP's parsers refuse (of 600)", bad, "#{naive} of 150 (naive \\d{4}-\\d{2}-\\d{2})",
          "0, control > 0", bad == 0 and naive > 0 and ok)
  end

  # a random member of a parsed pattern (ASCII where possible)
  defp sample({:set, ranges}) do
    ascii = for {a, b} <- ranges, a <= 0x7E, do: {a, min(b, 0x7E)}
    {a, b} = Enum.random(if ascii != [], do: ascii, else: ranges)
    [a + :rand.uniform(b - a + 1) - 1]
  end

  defp sample({:cat, xs}), do: Enum.flat_map(xs, &sample/1)
  defp sample({:alt, xs}), do: sample(Enum.random(xs))
  defp sample({:rep, a, mn, mx}), do: Enum.flat_map(1..(mn + :rand.uniform((if mx == :inf, do: mn + 2, else: mx) - mn + 1) - 1)//1, fn _ -> sample(a) end)

  # ------------------------------------------------------------- fusion --

  defp fusion do
    import Vapor.Merge, only: [stream: 3, merge: 2]
    tmp = Path.join(System.tmp_dir!(), "vapor-q07-#{System.unique_integer([:positive])}")
    map = %{"model_type" => "llama", "vocab_size" => 500, "hidden_size" => 192, "intermediate_size" => 384, "num_hidden_layers" => 6,
            "num_attention_heads" => 4, "num_key_value_heads" => 2, "max_position_embeddings" => 32, "rms_norm_eps" => 1.0e-5,
            "rope_theta" => 10_000.0, "hidden_act" => "silu", "tie_word_embeddings" => false}
    {:ok, c} = Vapor.Model.Config.from_map(map)

    dirs =
      for seed <- 1..2 do
        d = Path.join(tmp, "m#{seed}")
        File.mkdir_p!(d)
        File.write!(Path.join(d, "config.json"), Vapor.JSON.encode(map))
        ws = for {name, shape, _} <- Vapor.Model.Llama.expected_weights(c), into: %{}, do: {name, Vapor.Tensor.random(:f32, shape, :erlang.phash2({seed, name}), scale: 0.2)}
        {:ok, _} = Vapor.Ingest.Safetensors.write_sharded(d, ws, 2_000_000)
        d
      end

    size = dirs |> Enum.flat_map(fn d -> d |> File.ls!() |> Enum.map(&File.stat!(Path.join(d, &1)).size) end) |> Enum.sum()
    open = fn d -> {:ok, m} = Vapor.Lock.open(d); %{spec: m.spec, weights: m.weights} end
    {:ok, mem} = merge(Enum.map(dirs, open), method: :slerp, t: 0.3)
    {:ok, other} = merge(Enum.map(dirs, open), method: :slerp, t: 0.31)
    :erlang.garbage_collect()
    {{:ok, s}, peak} = peak(fn -> stream(dirs, Path.join(tmp, "out"), method: :slerp, t: 0.3) end)
    File.rm_rf!(tmp)
    same = s.receipt.payload.output == mem.receipt.payload.output
    ctl = other.receipt.payload.output == mem.receipt.payload.output

    check("fusion from disk: output root = in-memory fusion; peak memory / models' size", "#{same} · #{Float.round(peak / size, 3)}",
          "#{ctl} (another t)", "same root, memory < 0.25 of the models, control differs", same and not ctl and peak < 0.25 * size)
  end

  defp peak(f) do
    parent = self()
    # every process collected first: the baseline holds no garbage the measured call could free
    for pid <- Process.list(), do: :erlang.garbage_collect(pid)
    base = :erlang.memory(:binary)

    s = spawn(fn ->
      loop = fn loop, mx ->
        receive do
          :stop -> send(parent, {:peak, mx})
        after
          2 -> loop.(loop, max(mx, :erlang.memory(:binary)))
        end
      end

      loop.(loop, base)
    end)

    r = f.()
    send(s, :stop)
    receive do: ({:peak, mx} -> {r, max(mx - base, 0)})
  end

  # ------------------------------------------------- the scanned page --

  defp layout(nil), do: []

  defp layout(w) do
    dir = p(@priv <> "/scans")
    {:ok, truth} = Vapor.JSON.decode(File.read!(Path.join(dir, "truth.json")))
    {:ok, %{"readings" => tess}} = Vapor.JSON.decode(File.read!(Path.join(dir, "tesseract.json")))

    rows =
      for {name, t} <- Enum.sort(truth) do
        {:ok, %{1 => [img]}, _} = PDF.images(File.read!(Path.join(dir, name <> ".pdf")), [1])
        ref = t["blocks"] |> List.flatten() |> Enum.join("\n")
        {:ok, r} = OCR.read(img, worker: w)
        {:ok, flat} = OCR.read(img, worker: w, columns: false)
        {OCR.cer(r.text, ref), OCR.cer(flat.text, ref), OCR.cer(tess[name], ref)}
      end

    mean = fn i -> Enum.sum(Enum.map(rows, &elem(&1, i))) / length(rows) end
    {a, b, t} = {mean.(0), mean.(1), mean.(2)}

    check("reading order: mean CER of 8 scanned pages (PDF → CCITT → layout → reader; Tesseract #{Float.round(t * 100, 1)} %)",
          "#{Float.round(a * 100, 2)} %", "#{Float.round(b * 100, 1)} % (no layout: lines across columns)", "< 3 %, control > 20 %",
          a < 0.03 and b > 0.20)
  end

  defp language(nil), do: []

  defp language(w) do
    {:ok, m} = OCR.default()
    lm = OCR.language_model(m, [])
    alphabet = m.labels |> Tuple.to_list() |> tl() |> Enum.reject(&(&1 == " "))
    corpus = Enum.map_join(["pt_reference.txt", "en_reference.txt"], " ", &File.read!(p(@priv <> "/" <> &1))) |> String.replace(~r/[`*#|]/, "")
    shuffled = %{lm | lm: CharLM.shuffled(corpus, alphabet, lm.lm.order)}
    ungated = %{lm | gate: nil}

    lines = fn dir ->
      {:ok, meta} = Vapor.JSON.decode(File.read!(Path.join(dir, "labels.json")))

      for {file, %{"text" => text}} <- Enum.sort(meta["lines"]),
          {:ok, pic} = Vapor.Docs.Pictures.read(:png, File.read!(Path.join(dir, file))),
          [lps] <- [OCR.line_logprobs(pic.image, m, w)], lps != nil,
          do: {String.trim(Regex.replace(~r/ +/, text, " ")), lps}
    end

    cer = fn items, dec ->
      {e, n} = Enum.reduce(items, {0, 0}, fn {ref, lps}, {e, n} ->
        hyp = dec.(lps) |> Enum.map_join(&elem(m.labels, &1)) |> String.trim()
        {e + OCR.levenshtein(String.graphemes(hyp), String.graphemes(ref)), n + String.length(ref)}
      end)

      e / max(n, 1)
    end

    greedy = &OCR.greedy_labels/1
    beam = fn cfg -> &OCR.ctc_beam(&1, m.labels, cfg) end
    pct = fn x -> "#{Float.round(x * 100, 2)} %" end

    held = lines.(p(@priv <> "/ocr"))
    {g, b, s} = {cer.(held, greedy), cer.(held, beam.(lm)), cer.(held, beam.(shuffled))}

    rand = lines.(p(@priv <> "/ocr_controls/random"))
    changed = fn cfg -> Enum.count(rand, fn {_, lps} -> beam.(cfg).(lps) != greedy.(lps) end) end
    {rc, ru} = {changed.(lm), changed.(ungated)}

    codes = lines.(p(@priv <> "/ocr_controls/codes"))
    {cg, cb} = {cer.(codes, greedy), cer.(codes, beam.(lm))}

    [check("language model: CER of the held-out lines, beam + model (greedy #{pct.(g)})", pct.(b), "#{pct.(s)} (shuffled corpus)",
           "< 0.75 × greedy, control ≥ 0.95 × greedy", b < 0.75 * g and s >= 0.95 * g),
     check("the model abstains: random-string lines whose reading it changed (of #{length(rand)})", rc, "#{ru} (no gate)", "0, control > 0", rc == 0 and ru > 0),
     check("codes on a page (amounts, dates, IDs): CER with the model", pct.(cb), "#{pct.(cg)} (greedy)", "≤ greedy", cb <= cg)]
  end
end
