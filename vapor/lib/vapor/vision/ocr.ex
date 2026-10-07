defmodule Vapor.Vision.OCR do
  @moduledoc """
  **Optical character recognition by a model admitted through the airlock**
  — not by a dependency.

  A printed line is a sequence of column frames. Reading it is the same
  problem as reading speech: a bidirectional encoder classifies every
  frame, and CTC (connectionist temporal classification) collapses the
  frames into characters — no character segmentation, so touching glyphs,
  kerning and ligatures are the model's problem, not the geometry's.

      picture ─ Segment: ink, components, lines ─▶ line bitmaps (32 px, baseline row 22)
              ─ frames (8 columns every 2) ─▶ vapor_encoder (head: rows) on the substrate
              ─ CTC greedy ─▶ text, with per-character confidence and the frames behind it

  The reader is an ordinary `vapor_encoder` checkpoint (`config.json` +
  `model.safetensors`, `head: "rows"`, `labels` = the charset, blank = 0),
  trained by `test/python/train_ocr.py` on lines rendered in fonts it never
  sees at test time; vapor ships one in `priv/ocr` (its training recipe,
  data and measured error rates in `docs/OCR.md`). Any other checkpoint of
  the same contract — another script, another language — plugs in the
  same way.

  The encoder runs on the native worker (a line is one program run); on
  the oracle alone it is exact but slow (seconds per line).
  """
  alias Vapor.{Lock, Tensor}
  alias Vapor.Lock.Adapters.Encoder
  alias Vapor.Modal.Runner
  alias Vapor.Vision.Segment

  @doc """
  An OCR model shipped with vapor, admitted: `{:ok, model}` or `{:error,
  rejection}`. `script` is `:latin` (`priv/ocr`, printed Latin — the
  default), `:arabic` (`priv/ocr-arabic`, printed Arabic, right to left)
  or `:cyrillic` (`priv/ocr-cyrillic`, printed Russian); `:cursive` is
  refused with the measurement that kept it out.
  Chinese, Japanese and Korean have their own reader (`Vapor.Vision.CJK`).
  """
  def default(script \\ :latin)

  # Measured, not shipped: a reader trained on 17 handwriting typefaces read
  # the hands of 4 others at 64 % CER (the printed reader: 76 %) — noise, by
  # the suite's own standard (docs/OCR.md §3j). The contract stays: any
  # checkpoint trained on real handwriting loads with `load/1`.
  def default(:cursive) do
    {:error, Vapor.Rejection.new({:ocr, :cursive}, "a handwriting reader measured on hands it never saw",
                                 "none is shipped (64 % CER on unseen hands when trained on typefaces); train one on real handwriting with test/python/train_ocr.py and load it")}
  end

  def default(script) do
    dir = %{latin: "ocr", arabic: "ocr-arabic", cyrillic: "ocr-cyrillic"} |> Map.fetch!(script)

    case :persistent_term.get({__MODULE__, :default, script}, nil) do
      nil ->
        with {:ok, m} <- load(Path.join(to_string(:code.priv_dir(:vapor)), dir)) do
          :persistent_term.put({__MODULE__, :default, script}, m)
          {:ok, m}
        end

      m ->
        {:ok, m}
    end
  end

  @doc """
  A native worker kept for OCR across calls (`nil` on a host without one):
  ingesting a folder of scans must not start a process per page.
  """
  def worker do
    case :persistent_term.get({__MODULE__, :worker}, nil) do
      pid when is_pid(pid) ->
        if Process.alive?(pid), do: pid, else: start_worker()

      _ ->
        start_worker()
    end
  end

  defp start_worker do
    case Runner.worker() do
      nil ->
        nil

      pid ->
        Process.unlink(pid)
        :persistent_term.put({__MODULE__, :worker}, pid)
        pid
    end
  end

  @doc "Admit an OCR checkpoint (a `vapor_encoder` with `head: \"rows\"` and `labels`) and build its program."
  def load(dir) do
    with {:ok, m} <- Lock.open(dir),
         :ok <- need(m.spec.interface == :encoder and :row_logits in m.spec.features, "an encoder with a per-row head (head: \"rows\")"),
         labels when is_list(labels) <- m.spec.config.raw["labels"] || {:error, :labels},
         :ok <- need(m.spec.in_width == Segment.frame_geometry().row_width, "row_width #{Segment.frame_geometry().row_width} (32×8 frames)"),
         {:ok, programs} <- buckets(m) do
      {:ok, %{spec: m.spec, programs: programs, labels: List.to_tuple(["" | labels]), max_frames: m.spec.rows, dir: dir,
              direction: if(m.spec.config.raw["direction"] == "rtl", do: :rtl, else: :ltr)}}
    else
      {:error, :labels} -> need(false, "a \"labels\" list in config.json (the charset; CTC blank is index 0)")
      other -> other
    end
  end

  @doc """
  Read the text of a picture (`Vapor.Modal.Image` or `%{w, h, gray}`).
  Returns `%{text, lines: [%{box, block, text, greedy, confidence, chars:
  [%{char, p, frames}]}], tables, blocks: [box], block_kinds, confidence,
  decoder}` — lines in reading order (`block` indexes `blocks`), `greedy`
  the frames' own reading (what the language model changed is the
  difference). Ruled tables are read cell by cell (`Vapor.Vision.Table`):
  `tables` holds their structure and cells, and `text` holds each one as
  Markdown at its place in the reading order.
  Figures (`Vapor.Vision.Figure`: charts and pictures, with their
  captions) are set apart before the text is cut into blocks: `figures`
  holds `%{box, kind, caption}` (and, with `digitize: true`, each chart's
  data or its refusal), and their marks — tick labels, a curve — do not
  pollute the text.
  Options: `columns: false` (one block), `lm: false` (greedy decoding),
  `tables: false` (no table detection: the 0.7 reading), `figures: false`
  (no figure detection: the 0.9 reading), `digitize: true`.
  Options: `model` (a loaded model, or a script of `default/1`: `:latin` —
  the default —, `:arabic`, `:cursive`), `worker` (default: a native
  worker if the host has one), and those of `Segment.analyse/2`.
  """
  def read(picture, opts \\ []) do
    with {:ok, model} <- model_of(opts) do
      w = Keyword.get_lazy(opts, :worker, &worker/0)
      lm = language_model(model, opts)
      # a right-to-left reader reads columns right to left too
      opts = Keyword.put_new(opts, :direction, Map.get(model, :direction, :ltr))
      g = Segment.gray(picture)
      mask = Segment.ink(g, opts)
      # tables first (their rules declare them), then the rest of the page
      # in reading order: blocks (columns, bands), then the lines of each
      {tables, comps} = mask |> Segment.components(opts) |> Vapor.Vision.Table.detect(opts)
      # figures next (charts, pictures): their marks are not text; their captions are
      {figures, comps} =
        if Keyword.get(opts, :figures, true),
          do: Vapor.Vision.Figure.detect(comps, picture: picture, ocr: model, worker: w, digitize: Keyword.get(opts, :digitize, false)),
          else: {[], comps}

      text_blocks = Segment.blocks(comps, opts) |> Enum.map(&Map.put(&1, :kind, :text))
      table_blocks = Enum.map(tables, &%{box: &1.box, kind: :table, table: &1})
      blocks = place(text_blocks, table_blocks)

      read_blocks =
        blocks
        |> Enum.with_index()
        |> Enum.map(fn
          {%{kind: :table, table: t} = b, i} -> Map.put(b, :table, t |> Vapor.Vision.Table.read(model, w, lm, opts) |> Map.put(:block, i))
          {b, i} -> Map.put(b, :lines, b.comps |> Segment.lines(opts) |> Enum.map(&(&1 |> Map.put(:block, i) |> read_line(model, w, lm, opts))) |> Enum.filter(&reported?/1))
        end)

      read_lines = Enum.flat_map(read_blocks, &Map.get(&1, :lines, []))
      read_tables = for %{kind: :table, table: t} <- read_blocks, do: Map.put(t, :markdown, Vapor.Vision.Table.to_markdown(t))

      text =
        read_blocks
        |> Enum.map(fn
          %{kind: :table, table: t} -> Vapor.Vision.Table.to_markdown(t)
          %{lines: ls} -> Enum.map_join(ls, "\n", & &1.text)
        end)
        |> Enum.reject(&(&1 == ""))
        |> Enum.join("\n")

      scored = read_lines ++ Enum.flat_map(read_tables, fn t -> Enum.reject(t.cells, &(&1.text == "")) end)
      conf = if scored == [], do: 0.0, else: Enum.reduce(scored, 0.0, &(&1.confidence + &2)) / length(scored)
      {:ok, %{text: text, lines: read_lines, tables: read_tables, figures: Enum.map(figures, &Map.delete(&1, :comps)), confidence: conf, width: g.w, height: g.h,
              blocks: Enum.map(blocks, & &1.box), block_kinds: Enum.map(blocks, & &1.kind),
              decoder: if(lm, do: %{order: lm.lm.order, weight: lm.weight, bonus: lm.bonus, beam: lm.beam, gate: lm.gate}, else: :greedy)}}
    end
  end

  defp model_of(opts) do
    case opts[:model] do
      nil -> default()
      s when is_atom(s) -> default(s)
      m -> {:ok, m}
    end
  end

  # a table takes its place in the reading order before the first text
  # block that starts below its top and overlaps it horizontally (or at the end)
  defp place(text, []), do: text

  defp place(text, [t | more]) do
    {tx0, ty0, tx1, _} = t.box

    i =
      Enum.find_index(text, fn %{box: {x0, y0, x1, _}} -> y0 >= ty0 and min(x1, tx1) > max(x0, tx0) end) ||
        Enum.find_index(text, fn %{box: {_, y0, _, _}} -> y0 >= ty0 end) || length(text)

    place(List.insert_at(text, i, t), more)
  end

  @doc """
  Read one line of `Segment.lines/2` (cut where it exceeds the encoder's
  rows): `%{box, text, greedy, confidence, chars}` (and `block` when the
  line has one).
  """
  def read_line(line, model, w, lm, opts \\ []) do
    parts = split(line, model.max_frames)

    read =
      parts
      |> Enum.with_index()
      |> Enum.map(fn {part, i} ->
        bm = Segment.line_bitmap(part)
        logits = run(model, Segment.frames(bm), w)
        {cs, gr} = read_bitmap(bm, model, w, lm: lm, both: true, allowed: opts[:allowed], logits: logits)
        {if(i > 0, do: {[%{char: " ", k: nil, p: 1.0, frames: []} | cs], " " <> gr}, else: {cs, gr}), logits}
      end)

    {chars, greedy} = read |> Enum.map(&elem(&1, 0)) |> Enum.unzip()

    # the frames themselves, for a decoder that constrains them afterwards
    # (`Vapor.Vision.Template`); a line cut in parts has none
    lps =
      case {read, Keyword.get(opts, :logprobs, false)} do
        {[{_, logits}], true} -> Enum.map(logits, &log_softmax/1)
        _ -> nil
      end

    chars = List.flatten(chars)
    greedy = greedy |> Enum.join() |> String.trim()
    text = chars |> Enum.map_join(& &1.char) |> String.trim()
    ink = Enum.reject(chars, &(&1.char == " "))
    conf = if ink == [], do: 0.0, else: Enum.reduce(ink, 0.0, &(&1.p + &2)) / length(ink)

    # the frames are read left to right: for a right-to-left script that is
    # the visual order, turned back into the stored order (`chars` stay in
    # the order of the frames, with them)
    {text, greedy, extra} =
      if Map.get(model, :direction) == :rtl,
        do: {Vapor.Vision.Bidi.logical(text), Vapor.Vision.Bidi.logical(greedy), %{visual: text, direction: :rtl}},
        else: {text, greedy, %{}}

    %{box: line.box, block: Map.get(line, :block), text: text, greedy: greedy, confidence: conf, chars: chars}
    |> Map.merge(extra)
    |> then(fn r -> if lps, do: Map.put(r, :lps, lps), else: r end)
  end

  @doc """
  Read a set of components as text (a table cell, a region): their lines
  top to bottom, each read; `%{text, lines, confidence}`. Lines that are
  marks rather than text (`reported?/1`) are dropped, unless `marks: true`
  (the components are known to be content: a table cell). `allowed:` (a
  set of characters) restricts the decoding to them; `logprobs: true` keeps
  each single-part line's frame log-probabilities (`lps`).
  """
  def read_components(comps, model, w, lm, opts \\ []) do
    keep? = if opts[:marks], do: &(&1.text != ""), else: &reported?/1
    lines = comps |> Segment.lines(opts) |> Enum.map(&read_line(&1, model, w, lm, opts)) |> Enum.filter(keep?)
    conf = if lines == [], do: 0.0, else: Enum.reduce(lines, 0.0, &(&1.confidence + &2)) / length(lines)
    %{text: Enum.map_join(lines, " ", & &1.text), lines: lines, confidence: conf}
  end

  @doc """
  The language model a reading uses: `lm: :default` (the default — the
  shipped `priv/ocr/lm.json`, when the model's alphabet is the shipped
  reader's), `lm: false` (greedy CTC, no model), or a model map
  (`%{lm: %Vapor.Vision.CharLM{}, weight, bonus, beam}`).
  """
  def language_model(model, opts) do
    case Keyword.get(opts, :lm, :default) do
      :default ->
        alphabet = model.labels |> Tuple.to_list() |> tl() |> Enum.reject(&(&1 == " "))

        case Vapor.Vision.CharLM.default(alphabet) do
          {:ok, m} -> m
          _ -> nil
        end

      v when v in [nil, false] -> nil
      %{lm: _} = m -> m
    end
  end

  @doc """
  Whether a read line is text rather than a mark: it holds a word (two
  letters or digits in a row), or a lone symbol the reader is sure of
  (≥ 0.8: a page number, a bullet). A bar, a rule or a photo's edge read as
  a single unsure character is dropped — not reported as text.
  """
  def reported?(%{text: text, confidence: c}), do: text != "" and (word?(text) or c >= 0.8)

  @doc "Two letters or digits in a row somewhere in `text`."
  def word?(text), do: Regex.match?(~r/[[:alnum:]]{2}/u, text)

  @doc """
  Reading of one normalised line bitmap: `[%{char, p, frames}]`. With a
  language model (`lm:` — `%{lm, weight, bonus, beam}`, see
  `Vapor.Vision.CharLM.default/1`) the frames are decoded by a CTC prefix
  beam search scored by the model; otherwise greedily. Returns
  `{chars, greedy_text}` when `both: true`.
  """
  def read_bitmap(bitmap, model, worker \\ nil, opts \\ []) do
    rows = Segment.frames(bitmap)
    [t, _] = rows.shape
    logits = Keyword.get_lazy(opts, :logits, fn -> run(model, rows, worker) end) |> restrict(model.labels, opts[:allowed])
    greedy = ctc_greedy(logits, t, model.labels)

    chars =
      case opts[:lm] do
        nil -> greedy
        lm ->
          lps = Enum.map(logits, &log_softmax/1)
          labels = ctc_beam(lps, model.labels, lm)
          if labels == Enum.map(greedy, & &1.k), do: greedy, else: align(lps, labels, model.labels)
      end

    if opts[:both], do: {chars, Enum.map_join(greedy, & &1.char)}, else: chars
  end

  # `allowed`: a set of characters — every other label is impossible (its
  # logit −∞; the blank always stays): decoding restricted to a type, as a
  # grammar restricts a language model's tokens
  defp restrict(logits, _labels, nil), do: logits

  defp restrict(logits, labels, allowed) do
    keep = for k <- 0..(tuple_size(labels) - 1), into: MapSet.new(), do: (if k == 0 or MapSet.member?(allowed, elem(labels, k)), do: k, else: -1)

    Enum.map(logits, fn row ->
      row |> Enum.with_index() |> Enum.map(fn {v, k} -> if MapSet.member?(keep, k), do: v, else: -1.0e30 end)
    end)
  end

  @doc false
  # per-frame log-probabilities of every line of a picture (calibration, tests)
  def line_logprobs(picture, model, worker) do
    picture
    |> Segment.gray()
    |> Segment.ink()
    |> Segment.components()
    |> Segment.lines()
    |> Enum.map(fn line ->
      rows = line |> Segment.line_bitmap() |> Segment.frames()
      [t, _] = rows.shape
      if t > model.max_frames, do: nil, else: run(model, rows, worker) |> Enum.map(&log_softmax/1)
    end)
  end

  @doc false
  def greedy_labels(lps), do: lps |> Enum.map(fn lp -> lp |> Tuple.to_list() |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1) end) |> Enum.chunk_by(& &1) |> Enum.map(&hd/1) |> Enum.reject(&(&1 == 0))

  defp log_softmax(row) do
    m = Enum.max(row)
    z = m + :math.log(Enum.reduce(row, 0.0, &(&2 + :math.exp(&1 - m))))
    row |> Enum.map(&(&1 - z)) |> List.to_tuple()
  end

  @neg_inf -1.0e300

  defp lse(a, b) when a < b, do: lse(b, a)
  defp lse(a, _b) when a <= @neg_inf, do: a
  defp lse(a, b), do: a + :math.log(1.0 + :math.exp(b - a))

  @doc """
  CTC **prefix beam search** (Graves 2006; Hannun et al. 2014) with a
  character language model: every prefix keeps the probability of ending
  in a blank and in its last label; extending it by a character `c` adds
  `log P_frames + weight · log P_lm(c | prefix) + bonus`.

  **The model chooses among what the page could say, never against it.**
  At every frame only the labels the frames find plausible (probability ≥
  10⁻³, and the most probable one) — the blank included — can be taken: a
  letter read at 0.97 cannot be deleted because the model finds it rare in
  its context, and a confident gap cannot become a letter. Returns the best
  label sequence (indices into `labels`).

  **The model abstains where there is no language to model** (`gate`):
  when the greedy reading of the line costs more than `gate` bits per
  character under the model — random strings cost ≈ 12, held-out prose ≈ 2.6
  — the greedy reading is returned unchanged. Without the gate the model
  made random strings *worse* (CER 16,0 % → 22,5 %): it rewrote them toward
  what is common. Measured in docs/OCR.md §3b.
  """
  def ctc_beam(lps, labels, %{lm: lm} = cfg) do
    greedy = greedy_labels(lps)
    gate = cfg[:gate]

    if gate && greedy != [] && Vapor.Vision.CharLM.bits_per_char(lm, Enum.map_join(greedy, &elem(labels, &1))) > gate do
      greedy
    else
      beam(lps, labels, cfg)
    end
  end

  defp beam(lps, labels, %{lm: lm, weight: wt, bonus: bonus, beam: width}) do
    floor = :math.log(1.0e-3)
    n = tuple_size(labels)

    final =
      Enum.reduce(lps, %{[] => {0.0, @neg_inf}}, fn lp, beams ->
        # what the frame could plausibly be — blank included: a reading the
        # frame rules out (a confident letter deleted, a confident blank made
        # a letter) is not offered to the language model at all
        top = Enum.max_by(0..(n - 1), &elem(lp, &1))
        cands = for k <- 1..(n - 1)//1, elem(lp, k) >= floor or k == top, do: k
        blank = if elem(lp, 0) >= floor or top == 0, do: elem(lp, 0), else: @neg_inf

        next =
          Enum.reduce(beams, %{}, fn {prefix, {pb, pnb}}, acc ->
            tot = lse(pb, pnb)
            acc = if blank > @neg_inf, do: bump(acc, prefix, tot + blank, :b), else: acc
            last = List.first(prefix)

            Enum.reduce(cands, acc, fn k, acc ->
              p = elem(lp, k)
              ext = wt * lm_logp(lm, prefix, labels, k) + bonus

              if k == last do
                acc |> bump(prefix, pnb + p, :nb) |> bump([k | prefix], pb + p + ext, :nb)
              else
                bump(acc, [k | prefix], tot + p + ext, :nb)
              end
            end)
          end)

        next |> Enum.sort_by(fn {_, {b, nb}} -> -lse(b, nb) end) |> Enum.take(width) |> Map.new()
      end)

    {best, _} = Enum.max_by(final, fn {_, {b, nb}} -> lse(b, nb) end)
    Enum.reverse(best)
  end

  defp bump(acc, key, v, which) do
    {b, nb} = Map.get(acc, key, {@neg_inf, @neg_inf})
    Map.put(acc, key, if(which == :b, do: {lse(b, v), nb}, else: {b, lse(nb, v)}))
  end

  # the history of a prefix (label indices, most recent first) as characters
  defp lm_logp(lm, prefix, labels, k) do
    hist = prefix |> Enum.take(lm.order - 1) |> Enum.map(&elem(labels, &1))
    hist = if length(hist) < lm.order - 1, do: hist ++ [" "], else: hist
    Vapor.Vision.CharLM.log_prob_rev(lm, hist, elem(labels, k))
  end

  @doc """
  CTC Viterbi alignment of a label sequence to the frames: the frames each
  character occupies and their mean probability — the evidence behind a
  character the beam search chose.
  """
  def align(lps, seq, labels) do
    states = List.to_tuple(Enum.flat_map(seq, &[0, &1]) ++ [0])
    s = tuple_size(states)
    lps_t = List.to_tuple(lps)
    t = tuple_size(lps_t)
    first = elem(lps_t, 0)

    init =
      for(i <- 0..(s - 1), do: if(i < 2, do: {elem(first, elem(states, i)), nil}, else: {@neg_inf, nil}))
      |> List.to_tuple()

    # trellis rows: {score, back pointer}
    rows =
      Enum.reduce(1..(t - 1)//1, [init], fn f, [prev | _] = acc ->
        lp = elem(lps_t, f)

        row =
          for i <- 0..(s - 1) do
            cands = [{elem(elem(prev, i), 0), i}] ++
                      if(i >= 1, do: [{elem(elem(prev, i - 1), 0), i - 1}], else: []) ++
                      if(i >= 2 and elem(states, i) != 0 and elem(states, i) != elem(states, i - 2), do: [{elem(elem(prev, i - 2), 0), i - 2}], else: [])

            {sc, from} = Enum.max_by(cands, &elem(&1, 0))
            {sc + elem(lp, elem(states, i)), from}
          end
          |> List.to_tuple()

        [row | acc]
      end)

    last = hd(rows)
    endi = if s >= 2 and elem(elem(last, s - 2), 0) > elem(elem(last, s - 1), 0), do: s - 2, else: s - 1

    {path, _} =
      Enum.reduce(rows, {[], endi}, fn row, {path, i} -> {[i | path], elem(elem(row, i), 1) || i} end)

    path
    |> Enum.with_index()
    |> Enum.reject(fn {i, _} -> rem(i, 2) == 0 end)
    |> Enum.chunk_by(&elem(&1, 0))
    |> Enum.map(fn [{i, _} | _] = run ->
      k = elem(states, i)
      ps = Enum.map(run, fn {_, f} -> :math.exp(elem(elem(lps_t, f), k)) end)
      %{char: elem(labels, k), k: k, p: Enum.sum(ps) / length(ps), frames: Enum.map(run, &elem(&1, 1))}
    end)
  end

  # programs over 64, 128, 256 … rows (up to the model's): a line runs on
  # the smallest that holds it — attention and the rows it pads cost
  # quadratically and linearly in that size
  defp buckets(m) do
    sizes = Stream.iterate(64, &(&1 * 2)) |> Enum.take_while(&(&1 < m.spec.rows)) |> Kernel.++([m.spec.rows])

    Enum.reduce_while(sizes, {:ok, []}, fn t, {:ok, acc} ->
      case Lock.build(m.spec, m.weights, rows: t) do
        {:ok, p} -> {:cont, {:ok, acc ++ [{t, p}]}}
        err -> {:halt, err}
      end
    end)
  end

  # frames → per-frame logits [t, classes]
  defp run(model, %Tensor{shape: [t, _]} = rows, worker) do
    {size, program} = Enum.find(model.programs, fn {size, _} -> size >= t end)
    env = Encoder.input(model.spec, rows, size)
    out = Runner.run(program, env, worker: worker)
    c = tuple_size(model.labels)
    out.row_logits |> Tensor.to_floats() |> Enum.chunk_every(c) |> Enum.take(t)
  end

  @doc """
  Greedy CTC decoding of per-frame logits: the argmax of every frame,
  repeats collapsed, blanks (index 0) dropped. Each character carries the
  mean probability of its frames and their indices.
  """
  def ctc_greedy(logits, _t, labels) do
    logits
    |> Enum.with_index()
    |> Enum.map(fn {row, i} ->
      m = Enum.max(row)
      es = Enum.map(row, &:math.exp(&1 - m))
      z = Enum.sum(es)
      {best, k} = es |> Enum.with_index() |> Enum.max_by(&elem(&1, 0))
      {k, best / z, i}
    end)
    |> Enum.chunk_by(&elem(&1, 0))
    |> Enum.reject(fn [{k, _, _} | _] -> k == 0 end)
    |> Enum.map(fn [{k, _, _} | _] = run ->
      %{char: elem(labels, k), k: k, p: Enum.sum(Enum.map(run, &elem(&1, 1))) / length(run), frames: Enum.map(run, &elem(&1, 2))}
    end)
  end

  # a line longer than the encoder's rows is cut at its widest gap nearest
  # the middle, recursively (the cut falls between words)
  defp split(line, max_frames) do
    %{w: _} = bm = Segment.line_bitmap(line)
    geo = Segment.frame_geometry()
    frames = div(max(bm.w, geo.window) - geo.window, geo.stride) + 1

    if frames <= max_frames or length(line.glyphs) < 2 do
      [line]
    else
      gs = line.glyphs
      n = length(gs)

      cut =
        gs
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.with_index()
        |> Enum.max_by(fn {[%{box: {_, _, a1, _}}, %{box: {b0, _, _, _}}], i} -> (b0 - a1) - abs(i - n / 2) * 0.01 end)
        |> elem(1)

      {left, right} = Enum.split(gs, cut + 1)
      Enum.flat_map([left, right], &split(sub_line(line, &1), max_frames))
    end
  end

  defp sub_line(line, gs) do
    box = gs |> Enum.map(& &1.box) |> Enum.reduce(fn {a0, b0, a1, b1}, {c0, d0, c1, d1} -> {min(a0, c0), min(b0, d0), max(a1, c1), max(b1, d1)} end)
    %{line | glyphs: gs, box: box}
  end

  # ------------------------------------------------------------- datasets --

  @doc """
  Build a training/evaluation set from rendered lines (`test/python/ocr_render.py`):
  every `DIR/*.png` that segments into exactly one line becomes its
  normalised bitmap. Writes `out.bin` (`<<w::32-little, 32·w bytes>>` per
  line) and `out.json` (`{charset, texts, files}`). Returns counts.
  """
  def dataset(dir, out) do
    {:ok, meta} = Vapor.JSON.decode(File.read!(Path.join(dir, "labels.json")))

    rows =
      meta["lines"]
      |> Enum.sort()
      |> Task.async_stream(fn {file, %{"text" => text}} ->
        {:ok, pic} = Vapor.Docs.Pictures.read(:png, File.read!(Path.join(dir, file)))
        g = Segment.gray(pic.image)

        case g |> Segment.ink() |> Segment.components() |> Segment.lines() do
          [line] -> {:ok, file, String.trim(Regex.replace(~r/ +/, text, " ")), Segment.line_bitmap(line)}
          other -> {:skip, file, length(other)}
        end
      end, timeout: :infinity, ordered: true)
      |> Enum.map(fn {:ok, r} -> r end)

    ok = for {:ok, f, t, bm} <- rows, do: {f, t, bm}
    File.write!(out <> ".bin", Enum.map(ok, fn {_, _, bm} -> [<<bm.w::32-little>>, bm.data] end))
    File.write!(out <> ".json", Vapor.JSON.encode(%{charset: meta["charset"], texts: Enum.map(ok, &elem(&1, 1)), files: Enum.map(ok, &elem(&1, 0))}))
    %{lines: length(rows), kept: length(ok), skipped: length(rows) - length(ok)}
  end

  # ------------------------------------------------------------ measuring --

  @doc """
  Tesseract's frozen reading of an image in an evaluation set (`nil` when
  there is none): `DIR/tesseract.json`, written by
  `test/python/ocr_tesseract.py`. vapor never runs an external tool; the
  comparison is a file, reproducible without Tesseract installed.
  """
  def tesseract(path) do
    ref = Path.join(Path.dirname(path), "tesseract.json")

    with true <- File.exists?(ref), {:ok, %{"readings" => r}} <- Vapor.JSON.decode(File.read!(ref)) do
      r[Path.basename(path)]
    else
      _ -> nil
    end
  end

  @doc "Character error rate: Levenshtein distance / reference length (graphemes)."
  def cer(hyp, ref) do
    {h, r} = {String.graphemes(hyp), String.graphemes(ref)}
    if r == [], do: (if h == [], do: 0.0, else: 1.0), else: levenshtein(h, r) / length(r)
  end

  @doc "Word error rate: Levenshtein distance over words / reference words."
  def wer(hyp, ref) do
    {h, r} = {String.split(hyp), String.split(ref)}
    if r == [], do: (if h == [], do: 0.0, else: 1.0), else: levenshtein(h, r) / length(r)
  end

  @doc false
  def levenshtein(a, b) do
    bt = List.to_tuple(b)
    nb = tuple_size(bt)
    first = Enum.to_list(0..nb)

    a
    |> Enum.with_index(1)
    |> Enum.reduce(first, fn {x, i}, prev ->
      prev_t = List.to_tuple(prev)

      {row, _} =
        Enum.reduce(1..nb//1, {[i], i}, fn j, {acc, left} ->
          cost = if x == elem(bt, j - 1), do: 0, else: 1
          v = Enum.min([left + 1, elem(prev_t, j) + 1, elem(prev_t, j - 1) + cost])
          {[v | acc], v}
        end)

      Enum.reverse(row)
    end)
    |> List.last()
  end

  defp need(true, _), do: :ok
  defp need(false, what), do: {:error, Vapor.Rejection.new(:ocr_model, what, "use an OCR checkpoint of the vapor_encoder contract (docs/OCR.md)")}
end
