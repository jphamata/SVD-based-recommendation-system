defmodule Vapor.Vision.CJK do
  @moduledoc """
  **Reading Chinese, Japanese and Korean** — thousands of classes without
  a trained network, by first principles of the scripts.

  Three facts carry the design:

  1. **The writing is a grid.** Every hanzi, kanji, kana and hangul
     syllable occupies the same square cell, as tall as the line. A
     character made of several pieces (好 = 女 + 子) is not segmented by
     its connected components but by *cells*: candidate cuts sit in the
     ink gaps, and a candidate segment is judged in a cell of the line's
     height centred on it — so the left half of 好, seen in a full cell,
     looks like a narrow 女, not like the full-width 女 of the templates.
  2. **A typeface is its own training set.** The class templates are
     rendered from the national character standards (GB 2312, JIS X 0208,
     KS X 1001) in several typefaces and summarised by *directional
     element features* — the gradient of the stroke edges binned in eight
     directions over an 8×8 grid (512 values, square-rooted and
     normalised): the representation printed-CJK recognition has relied on
     since the 1990s because a stroke's direction survives a change of
     font far better than its pixels do. Reading is then a nearest
     template by cosine — one `linear` on the native worker against the
     whole class matrix (`2·x·t` with unit templates: the VQ trick again).
  3. **Recognition decides the segmentation.** Every segmentation of a
     line into cells is scored by how well its cells are recognised, minus
     a fixed price per cell, and the best path wins (dynamic programming
     over the gaps); an optional character language model
     (`Vapor.Vision.CharLM`, Witten–Bell) reranks only among visually
     plausible candidates and never rescues an implausible one.

  What is measured — on fonts never used for the templates, with the
  control of random characters — is in `docs/OCR.md §3g`.
  """
  alias Vapor.{Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Vision.{CharLM, Segment}

  @cell 64
  @grid 8
  @dirs 8
  @dim @grid * @grid * @dirs

  defstruct [:lang, :classes, :templates, :index, :lm, :fonts, :price]

  def dim, do: @dim

  # ------------------------------------------------------------ features --

  @doc """
  The 512 directional element features of the cell centred on columns
  `xa..xb` of a line whose rows are `y0..y1`, from an ink map (`Segment.ink`).
  The cell is the line's height wide; it is sampled to #{@cell}×#{@cell}
  by area coverage.
  """
  def features(mask, {y0, y1}, {xa, xb}) do
    h = y1 - y0 + 1
    cx = (xa + xb + 1) / 2
    # only the segment's own columns: a narrow digit or bracket centred in
    # a full cell must not see half of its neighbours
    cell = coverage(mask, cx - h / 2, y0, h, {xa, xb})
    directional(cell)
  end

  # area coverage of the source square [sx, sx+side) × [sy, sy+side) on a
  # @cell grid: a tuple of rows of floats in [0, 1]
  defp coverage(mask, sx, sy, side, {ca, cb}) do
    mh = tuple_size(mask)
    mw = if mh > 0, do: tuple_size(elem(mask, 0)), else: 0
    s = @cell / side
    acc = :counters.new(@cell * @cell, [])
    {x_lo, x_hi} = {max(max(floor_i(sx), 0), ca), min(min(ceil_i(sx + side) - 1, mw - 1), cb)}
    {y_lo, y_hi} = {max(floor_i(sy), 0), min(ceil_i(sy + side) - 1, mh - 1)}

    for y <- y_lo..y_hi//1, row = elem(mask, y), x <- x_lo..x_hi//1, elem(row, x) == 1 do
      {fx0, fy0} = {(x - sx) * s, (y - sy) * s}
      {fx1, fy1} = {fx0 + s, fy0 + s}

      for cy <- max(floor_i(fy0), 0)..min(ceil_i(fy1) - 1, @cell - 1)//1, cx <- max(floor_i(fx0), 0)..min(ceil_i(fx1) - 1, @cell - 1)//1 do
        a = (min(fx1, cx + 1) - max(fx0, cx)) * (min(fy1, cy + 1) - max(fy0, cy))
        if a > 0, do: :counters.add(acc, cy * @cell + cx + 1, round(a * 1024))
      end
    end

    for y <- 0..(@cell - 1) do
      for(x <- 0..(@cell - 1), do: min(:counters.get(acc, y * @cell + x + 1) / 1024, 1.0)) |> List.to_tuple()
    end
    |> List.to_tuple()
  end

  # a [1 4 6 4 1]/16 binomial blur (σ ≈ 1 px of the 64-px cell), rows
  # then columns: stroke weight and serifs matter less than stroke direction
  defp smooth(img) do
    k = {1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16}
    n = tuple_size(img)
    pass = fn rows ->
      for r <- Tuple.to_list(rows) do
        for(x <- 0..(n - 1), do: Enum.reduce(-2..2, 0.0, fn d, a -> xx = x + d; if xx < 0 or xx >= n, do: a, else: a + elem(k, d + 2) * elem(r, xx) end))
        |> List.to_tuple()
      end
      |> List.to_tuple()
    end
    img |> pass.() |> transpose() |> pass.() |> transpose()
  end

  defp transpose(t) do
    n = tuple_size(t)
    for(x <- 0..(n - 1), do: for(y <- 0..(n - 1), do: elem(elem(t, y), x)) |> List.to_tuple()) |> List.to_tuple()
  end

  # Sobel gradients, each split between its two nearest of eight
  # directions, pooled bilinearly into 8×8 blocks; √ then unit length
  defp directional(img0) do
    img = smooth(img0)
    n = @cell
    at = fn x, y -> if x < 0 or y < 0 or x >= n or y >= n, do: 0.0, else: elem(elem(img, y), x) end
    bins = :counters.new(@dim, [])
    bw = n / @grid
    q = 4096.0

    for y <- 0..(n - 1), x <- 0..(n - 1) do
      gx = at.(x + 1, y - 1) + 2 * at.(x + 1, y) + at.(x + 1, y + 1) - at.(x - 1, y - 1) - 2 * at.(x - 1, y) - at.(x - 1, y + 1)
      gy = at.(x - 1, y + 1) + 2 * at.(x, y + 1) + at.(x + 1, y + 1) - at.(x - 1, y - 1) - 2 * at.(x, y - 1) - at.(x + 1, y - 1)
      m = :math.sqrt(gx * gx + gy * gy)

      if m > 1.0e-6 do
        t = :math.atan2(gy, gx) / (:math.pi() / 4)
        t = if t < 0, do: t + @dirs, else: t
        d0 = floor_i(t) |> rem(@dirs)
        d1 = rem(d0 + 1, @dirs)
        f = t - floor_i(t)
        # bilinear position among block centres
        bx = (x + 0.5) / bw - 0.5
        by = (y + 0.5) / bw - 0.5
        {ix, iy} = {floor_i(bx), floor_i(by)}
        {fx, fy} = {bx - ix, by - iy}

        for {jx, wx} <- [{ix, 1 - fx}, {ix + 1, fx}], jx >= 0 and jx < @grid, {jy, wy} <- [{iy, 1 - fy}, {iy + 1, fy}], jy >= 0 and jy < @grid do
          base = (jy * @grid + jx) * @dirs
          w = m * wx * wy
          :counters.add(bins, base + d0 + 1, round(w * (1 - f) * q))
          :counters.add(bins, base + d1 + 1, round(w * f * q))
        end
      end
    end

    v = for i <- 1..@dim, do: :math.sqrt(:counters.get(bins, i) / q)
    norm = :math.sqrt(Enum.reduce(v, 0.0, &(&1 * &1 + &2)))
    if norm > 0, do: Enum.map(v, &(&1 / norm)), else: v
  end

  defp floor_i(v) do
    t = trunc(v)
    if v < t, do: t - 1, else: t
  end

  defp ceil_i(v) do
    t = trunc(v)
    if v > t, do: t + 1, else: t
  end

  # ----------------------------------------------------------- templates --

  @doc """
  Class templates from rendered template lines (`test/python/ocr_render_cjk.py
  LANG DIR templates`): each character's features in each font, averaged
  per class and normalised. Returns a pack (`save/2`, `load/1`). Option
  `lm`: a corpus (string) for the character language model.
  """
  def build(dir, opts \\ []) do
    {:ok, meta} = Vapor.JSON.decode(File.read!(Path.join(dir, "labels.json")))
    classes = meta["classes"]
    index = classes |> Enum.with_index() |> Map.new()

    sums =
      meta["lines"]
      |> Enum.sort()
      |> Task.async_stream(fn {file, %{"text" => text, "cells" => cells}} ->
        {:ok, pic} = Vapor.Docs.Pictures.read(:png, File.read!(Path.join(dir, file)))
        g = Segment.gray(pic.image)
        mask = Segment.ink(g)
        {y0, y1} = rows_of(mask)

        Enum.zip(String.graphemes(text), cells)
        |> Enum.flat_map(fn {ch, [x0, x1]} ->
          case ink_extent(mask, {y0, y1}, {round(x0), round(x1) - 1}) do
            nil -> []
            ext -> [{index[ch], features(mask, {y0, y1}, ext)}]
          end
        end)
      end, timeout: :infinity, ordered: false)
      |> Enum.flat_map(fn {:ok, fs} -> fs end)
      |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))

    rows =
      for i <- 0..(length(classes) - 1) do
        case sums[i] do
          nil -> List.duplicate(0.0, @dim)
          vs -> vs |> Enum.zip_with(fn col -> Enum.sum(col) / length(vs) end) |> unit()
        end
      end

    fonts = meta["lines"] |> Map.values() |> Enum.map(& &1["font"]) |> Enum.uniq() |> Enum.sort()
    lm = if opts[:lm], do: CharLM.build(opts[:lm], Enum.uniq(classes ++ [" "]), Keyword.get(opts, :order, 3))

    %__MODULE__{lang: meta["lang"], classes: List.to_tuple(classes), index: index, fonts: fonts, lm: lm,
                templates: Tensor.from_list(:f32, [length(classes), @dim], List.flatten(rows)), price: Keyword.get(opts, :price, 0.72)}
  end

  defp unit(v) do
    n = :math.sqrt(Enum.reduce(v, 0.0, &(&1 * &1 + &2)))
    if n > 0, do: Enum.map(v, &(&1 / n)), else: v
  end

  # the line's ink rows (first and last row with ink)
  defp rows_of(mask) do
    rows = for {row, y} <- Tuple.to_list(mask) |> Enum.with_index(), Enum.any?(Tuple.to_list(row), &(&1 == 1)), do: y
    {Enum.min(rows), Enum.max(rows)}
  end

  # columns with ink, within x0..x1 and the rows of the line
  defp ink_extent(mask, {y0, y1}, {x0, x1}) do
    w = tuple_size(elem(mask, 0))
    cols = for x <- max(x0, 0)..min(x1, w - 1)//1, Enum.any?(y0..y1, &(elem(elem(mask, &1), x) == 1)), do: x
    if cols == [], do: nil, else: {Enum.min(cols), Enum.max(cols)}
  end

  @doc """
  Save a pack: `templates.safetensors` (the class matrix, as 8-bit levels
  of the non-negative unit features — a quarter of the bytes, the cosine
  barely moved) and `config.json` (classes, fonts, the per-cell price).
  """
  def save(%__MODULE__{} = p, dir) do
    File.mkdir_p!(dir)
    q = for(<<v::float-32-little <- p.templates.data>>, into: <<>>, do: <<min(round(v * 255 * 4), 255)>>)
    :ok = Vapor.Ingest.Safetensors.write(Path.join(dir, "templates.safetensors"), %{"templates" => Tensor.new(:u8, p.templates.shape, q)})
    cfg = %{"model_type" => "vapor_cjk_templates", "lang" => p.lang, "classes" => Tuple.to_list(p.classes), "fonts" => p.fonts,
            "features" => %{"cell" => @cell, "grid" => @grid, "directions" => @dirs, "scale" => 4.0 / 255}, "price" => p.price}
    File.write!(Path.join(dir, "config.json"), Vapor.JSON.encode(cfg))
    {:ok, dir}
  end

  @doc "Load a pack written by `save/2` (rows renormalised after dequantisation). Option `lm`: corpus text."
  def load(dir, opts \\ []) do
    {:ok, cfg} = Vapor.JSON.decode(File.read!(Path.join(dir, "config.json")))
    {:ok, %{"templates" => t}} = Vapor.Ingest.Safetensors.read(Path.join(dir, "templates.safetensors"))
    [c, d] = t.shape
    scale = cfg["features"]["scale"]
    rows = for(<<b <- t.data>>, do: b * scale) |> Enum.chunk_every(d) |> Enum.map(&unit/1)
    classes = cfg["classes"]
    lm = if opts[:lm], do: CharLM.build(opts[:lm], Enum.uniq(classes ++ [" "]), Keyword.get(opts, :order, 3))

    {:ok, %__MODULE__{lang: cfg["lang"], classes: List.to_tuple(classes), index: classes |> Enum.with_index() |> Map.new(),
                      fonts: cfg["fonts"], lm: lm, price: cfg["price"], templates: Tensor.from_list(:f32, [c, d], List.flatten(rows))}}
  end

  @doc """
  The pack shipped for `lang` (`:zh` simplified Chinese, `:ja`, `:ko`):
  `priv/ocr-cjk-<lang>`, with its character language model built from the
  pack's `lm_corpus.txt`. Cached after the first load.
  """
  def default(lang) when lang in [:zh, :ja, :ko] do
    key = {__MODULE__, :default, lang}

    case :persistent_term.get(key, nil) do
      nil ->
        dir = Path.join(to_string(:code.priv_dir(:vapor)), "ocr-cjk-#{lang}")
        corpus = Path.join(dir, "lm_corpus.txt")

        with true <- File.exists?(Path.join(dir, "config.json")) || {:error, {:missing, dir}},
             {:ok, p} <- load(dir, lm: if(File.exists?(corpus), do: File.read!(corpus))) do
          :persistent_term.put(key, p)
          {:ok, p}
        end

      p ->
        {:ok, p}
    end
  end

  # --------------------------------------------------------------- reading --

  @doc """
  Read a line or a page (horizontal text): `{:ok, %{text, lines: [%{box,
  text, chars: [%{char, score, alts, cols}]}]}}`. Options: `worker` (a
  native worker; default the oracle — exact, slower), `lm: false`,
  `price` (the per-cell price of the segmentation), `top` (candidates kept
  per cell, 5).
  """
  def read(%__MODULE__{} = p, picture, opts \\ []) do
    g = Segment.gray(picture)
    mask = Segment.ink(g)
    read_lines = mask |> lines() |> Enum.map(&read_line(p, mask, &1, opts))
    {:ok, %{text: Enum.map_join(read_lines, "\n", & &1.text), lines: read_lines}}
  end

  @doc """
  Text lines of a CJK page by the horizontal projection of the ink: rows
  with ink form bands, and a band separated from the next by a gap smaller
  than a third of the larger one is the same line (a hangul syllable
  stacks its jamo with gaps between them; the components-and-cores line
  finder of `Vapor.Vision.Segment`, made for Latin, splits such a line in
  two). Returns `[%{box: {x0, y0, x1, y1}}]`, top to bottom.
  """
  def lines(mask) do
    h = tuple_size(mask)
    w = if h > 0, do: tuple_size(elem(mask, 0)), else: 0
    inked = for y <- 0..(h - 1)//1, do: Enum.any?(Tuple.to_list(elem(mask, y)), &(&1 == 1))

    bands =
      inked
      |> Enum.with_index()
      |> Enum.chunk_by(&elem(&1, 0))
      |> Enum.filter(fn [{v, _} | _] -> v end)
      |> Enum.map(fn run -> {elem(hd(run), 1), elem(List.last(run), 1)} end)

    merged =
      Enum.reduce(bands, [], fn
        {a, b}, [{pa, pb} | rest] = acc ->
          if a - pb - 1 < max(b - a + 1, pb - pa + 1) / 3, do: [{pa, b} | rest], else: [{a, b} | acc]

        band, [] ->
          [band]
      end)
      |> Enum.reverse()
      # specks: bands far thinner than the text are noise
      |> then(fn bs -> (m = bs |> Enum.map(fn {a, b} -> b - a + 1 end) |> Enum.max(fn -> 0 end); Enum.filter(bs, fn {a, b} -> b - a + 1 >= m / 4 end)) end)

    for {y0, y1} <- merged do
      cols = for x <- 0..(w - 1)//1, Enum.any?(y0..y1, &(elem(elem(mask, &1), x) == 1)), do: x
      %{box: {Enum.min(cols), y0, Enum.max(cols), y1}}
    end
  end

  @doc false
  def read_line(%__MODULE__{} = p, mask, %{box: {lx0, y0, lx1, y1}} = line, opts) do
    h = y1 - y0 + 1
    atoms = atoms(mask, {y0, y1}, {lx0, lx1})
    n = length(atoms)
    at = List.to_tuple(atoms)
    # candidate segments: consecutive atoms no wider than 1.3 cells
    segs = for i <- 0..(n - 1)//1, j <- i..(n - 1)//1, elem(elem(at, j), 1) - elem(elem(at, i), 0) + 1 <= max(1.3 * h, elem(elem(at, i), 1) - elem(elem(at, i), 0) + 1), do: {i, j}
    feats = Enum.map(segs, fn {i, j} -> features(mask, {y0, y1}, {elem(elem(at, i), 0), elem(elem(at, j), 1)}) end)
    top = Keyword.get(opts, :top, 5)
    cands = classify(p, feats, top, opts) |> then(&Enum.zip(segs, &1)) |> Map.new()
    price = Keyword.get(opts, :price, p.price)
    greedy = best_path(n, cands, price, nil, p, opts)
    lm = if Keyword.get(opts, :lm, true), do: p.lm
    gate = Keyword.get(opts, :gate, 10.0)

    # the language model abstains where the frames read no language: a
    # visual reading that the model finds this unlikely is kept as read
    path =
      cond do
        lm == nil -> greedy
        CharLM.bits_per_char(lm, Enum.map_join(greedy, "", fn {_, {ch, _}} -> ch end)) > gate -> greedy
        true -> best_path(n, cands, price, lm, p, opts)
      end

    chars =
      Enum.map(path, fn {{i, j}, {ch, score}} ->
        %{char: ch, score: score, cols: {elem(elem(at, i), 0), elem(elem(at, j), 1)},
          alts: cands[{i, j}] |> Enum.map(fn {k, s} -> {elem(p.classes, k), s} end)}
      end)

    %{box: line.box, text: Enum.map_join(chars, "", & &1.char), chars: chars,
      confidence: if(chars == [], do: 0.0, else: Enum.sum(Enum.map(chars, & &1.score)) / length(chars))}
  end

  # maximal runs of ink columns between empty columns; a run wider than a
  # cell and a quarter (glyphs that touch, in a tight typeface) is cut
  # where the column ink is thinnest near each multiple of the cell width
  defp atoms(mask, {y0, y1}, {x0, x1}) do
    h = y1 - y0 + 1
    ink = for x <- x0..x1, do: Enum.count(y0..y1, &(elem(elem(mask, &1), x) == 1))
    inkt = List.to_tuple(ink)
    col = fn x -> elem(inkt, x - x0) end

    ink
    |> Enum.with_index(x0)
    |> Enum.chunk_by(&(elem(&1, 0) > 0))
    |> Enum.filter(fn [{v, _} | _] -> v > 0 end)
    |> Enum.map(fn run -> {elem(hd(run), 1), elem(List.last(run), 1)} end)
    |> Enum.flat_map(fn {a, b} = run ->
      wid = b - a + 1

      if wid <= 1.25 * h do
        [run]
      else
        k = max(round(wid / (0.95 * h)), 2)
        step = wid / k
        cuts =
          for j <- 1..(k - 1) do
            ideal = a + round(j * step)
            win = max(round(step / 4), 1)
            Enum.min_by(max(ideal - win, a + 1)..min(ideal + win, b), fn x -> {col.(x), abs(x - ideal)} end)
          end
          |> Enum.uniq()
          |> Enum.sort()

        Enum.zip([a | cuts], Enum.map(cuts, &(&1 - 1)) ++ [b])
      end
    end)
  end

  # top-k classes per feature row by cosine against the unit templates
  defp classify(_p, [], _top, _opts), do: []

  defp classify(p, feats, top, opts) do
    nrows = length(feats)
    padded = nrows + rem(16 - rem(nrows, 16), 16)
    x = Tensor.from_list(:f32, [padded, @dim], List.flatten(feats) ++ List.duplicate(0.0, (padded - nrows) * @dim))
    scores = run_scores(p, x, opts)

    scores
    |> Tensor.to_floats()
    |> Enum.chunk_every(elem(p.templates.shape |> List.to_tuple(), 0))
    |> Enum.take(nrows)
    |> Enum.map(fn row -> row |> Enum.with_index() |> Enum.sort_by(&(-elem(&1, 0))) |> Enum.take(top) |> Enum.map(fn {s, k} -> {k, s} end) end)
  end

  defp run_scores(p, x, opts) do
    comp = compiled(p)
    env = %{x: x}

    case Keyword.get(opts, :worker) do
      nil ->
        {:ok, r} = Vapor.Runtime.Native.run_oracle(comp, env)
        r.outputs.s

      w ->
        {:ok, r} = Vapor.Runtime.Native.run(w, comp, env, isa: Vapor.Runtime.Substrates.host_isa(), mode: :native)
        r.outputs.s
    end
  end

  defp compiled(p) do
    key = {__MODULE__, :erlang.phash2(p.templates.data)}

    case :persistent_term.get(key, nil) do
      nil ->
        prog = Program.new(s: T.linear(T.input(:x, :f32, [T.dyn(:n, 4096), @dim]), T.const(p.templates)))
        {:ok, c} = Vapor.Compile.Lower.lower(prog)
        :persistent_term.put(key, c)
        c

      c ->
        c
    end
  end

  # dynamic programming over atom boundaries: each segment contributes its
  # best (visual + language) candidate minus the price of a cell; with a
  # language model the state carries the last characters (a small beam)
  defp best_path(0, _cands, _price, _lm, _p, _opts), do: []

  defp best_path(n, cands, price, lm, p, opts) do
    beam = Keyword.get(opts, :beam, 8)
    weight = Keyword.get(opts, :lm_weight, 0.04)
    margin = Keyword.get(opts, :margin, 0.06)

    # best[e] = list of {score, rev_path, history} ending after atom e-1
    init = %{0 => [{0.0, [], []}]}

    final =
      Enum.reduce(1..n, init, fn e, acc ->
        hyps =
          for s <- 0..(e - 1), Map.has_key?(acc, s), cs = Map.get(cands, {s, e - 1}), cs != nil, {sc, path, hist} <- acc[s] do
            {k0, best} = hd(cs)
            # language: only candidates within `margin` of the visual best
            options = if lm, do: Enum.filter(cs, fn {_, v} -> v >= best - margin end), else: [{k0, best}]

            for {k, v} <- options do
              ch = elem(p.classes, k)
              lmv = if lm, do: weight * CharLM.log_prob_rev(lm, hist, ch), else: 0.0
              {sc + v - price + lmv, [{{s, e - 1}, {ch, v}} | path], Enum.take([ch | hist], 4)}
            end
          end
          |> List.flatten()
          |> Enum.sort_by(&(-elem(&1, 0)))
          |> Enum.uniq_by(fn {_, _, hist} -> if lm, do: hist, else: :one end)
          |> Enum.take(if(lm, do: beam, else: 1))

        Map.put(acc, e, hyps)
      end)

    case final[n] do
      [{_, path, _} | _] -> Enum.reverse(path)
      _ -> []
    end
  end

end
