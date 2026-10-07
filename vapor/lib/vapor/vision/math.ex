defmodule Vapor.Vision.Math do
  @moduledoc """
  **Typeset formulas to LaTeX** — symbols by their shape, structure by
  geometry.

  A printed formula is two problems that a sequence reader conflates: *what*
  each mark is, and *where* it sits. Here they are kept apart.

  1. **Symbols.** Marks are connected components, merged where one symbol
     is several marks (the two bars of `=`, the dot of an `i`). Two kinds
     are told by geometry alone, because their shape *is* their meaning: a
     flat bar with marks above and below it is a fraction bar (otherwise a
     minus); a hollow mark that encloses others is a radical. Every other
     mark is classified by its nearest template — directional edge
     features of the mark in a square cell, plus its aspect — averaged
     from the classes rendered in three type families (the templates never
     include the families measured on).
  2. **Structure.** Outermost fraction bars first (numerator: the marks
     above the bar within its span; denominator: below), then radicals
     (the marks inside), then the limits of a `∑` (above and below it);
     what remains is read left to right, each mark either on the line of
     the base before it or raised or lowered from it — a superscript, a
     subscript, or both — by where its centre sits relative to the base's
     body. Each group is parsed by the same rules, recursively.

  The output spelling is canonical (`x^{2}`, `a_{i}^{n}`, `\\frac{a}{b}`,
  `\\sqrt{x}`, `\\sum_{i=1}^{n}`, `\\alpha`), and `confidence` is the
  smallest template similarity of the formula's marks — the mark most
  likely to be wrong. Measured on formulas set in Computer Modern and STIX
  (never in the templates): `docs/OCR.md §3i`. What the grammar does not
  cover (matrices, accents, `\\left…\\right` delimiters, multi-line
  layouts) is outside this reader.
  """
  alias Vapor.Vision.Segment

  defstruct [:names, :latex, :templates]

  @grid 6
  @cell 24

  # ------------------------------------------------------------- symbols --

  @doc "The symbols of a formula image: `[%{box, pixels, area}]`, marks merged into symbols."
  def symbols(picture) do
    picture |> Segment.gray() |> Segment.ink() |> Segment.components() |> Enum.filter(&(&1.area >= 3)) |> merge()
  end

  # `=`: two flat bars of one width, one just over the other with nothing
  # between (two stacked fraction bars have a line of marks between them);
  # `i`: a dot over a stem
  defp merge(cs) do
    cs = Enum.sort_by(cs, fn %{box: {x0, y0, _, _}} -> {x0, y0} end)

    Enum.reduce(cs, [], fn c, acc ->
      case Enum.find_index(acc, &(pair?(&1, c) and nothing_between?(&1, c, cs))) do
        nil -> [c | acc]
        i -> List.update_at(acc, i, &join(&1, c))
      end
    end)
    |> Enum.reverse()
  end

  defp nothing_between?(a, b, cs) do
    {top, bot} = if elem(a.box, 1) < elem(b.box, 1), do: {a, b}, else: {b, a}
    {gy0, gy1} = {elem(top.box, 3), elem(bot.box, 1)}
    {gx0, gx1} = {max(elem(a.box, 0), elem(b.box, 0)), min(elem(a.box, 2), elem(b.box, 2))}

    not Enum.any?(cs, fn c ->
      c != a and c != b and elem(c.box, 1) < gy1 and elem(c.box, 3) > gy0 and elem(c.box, 0) <= gx1 and elem(c.box, 2) >= gx0
    end)
  end

  defp pair?(a, b) do
    {a0, a1, b0, b1} = {elem(a.box, 0), elem(a.box, 2), elem(b.box, 0), elem(b.box, 2)}
    {aw, bw} = {a1 - a0 + 1, b1 - b0 + 1}
    {ah, bh} = {h(a), h(b)}
    ov = min(a1, b1) - max(a0, b0) + 1
    gap = max(elem(b.box, 1) - elem(a.box, 3), elem(a.box, 1) - elem(b.box, 3))
    flat? = fn w, hh -> hh * 3 <= w end

    (flat?.(aw, ah) and flat?.(bw, bh) and abs(aw - bw) <= 0.3 * max(aw, bw) and ov >= 0.7 * min(aw, bw) and gap <= 0.5 * max(aw, bw)) or
      (ov >= 0.5 * min(aw, bw) and gap >= 0 and gap <= 0.6 * max(ah, bh) and (dot_over?(a, b) or dot_over?(b, a)))
  end

  # a dot over a stem: the dot small both ways (no wider than the stem's
  # width by much, at most half its height), the stem tall and narrow — not
  # a digit over a fraction bar
  defp dot_over?(d, stem) do
    elem(d.box, 3) < elem(stem.box, 1) and h(stem) >= 1.5 * w(stem) and max(w(d), h(d)) <= 1.8 * w(stem) and h(d) <= 0.5 * h(stem)
  end

  defp join(a, b) do
    {a0, a1, a2, a3} = a.box
    {b0, b1, b2, b3} = b.box
    %{box: {min(a0, b0), min(a1, b1), max(a2, b2), max(a3, b3)}, pixels: a.pixels ++ b.pixels, area: a.area + b.area}
  end

  defp h(%{box: {_, y0, _, y1}}), do: y1 - y0 + 1
  defp w(%{box: {x0, _, x1, _}}), do: x1 - x0 + 1
  defp cx(%{box: {x0, _, x1, _}}), do: (x0 + x1) / 2
  defp cy(%{box: {_, y0, _, y1}}), do: (y0 + y1) / 2

  @doc """
  A symbol's features: its marks in a #{@cell}×#{@cell} square cell (aspect
  kept, centred), edge directions (horizontal, vertical, two diagonals) on
  a #{@grid}×#{@grid} grid, the coverage on the same grid, unit length; then
  the log-aspect, compared apart (`similarity/2`), so `-`, `1` and `o` stay
  apart.
  """
  def features(%{box: {x0, y0, x1, y1}, pixels: px}) do
    {bw, bh} = {x1 - x0 + 1, y1 - y0 + 1}
    side = max(bw, bh)
    s = (@cell - 2) / side
    {ox, oy} = {(@cell - bw * s) / 2, (@cell - bh * s) / 2}

    set =
      for {x, y} <- px, ix <- trunc(ox + (x - x0) * s)..max(trunc(ox + (x - x0 + 1) * s) - 1, trunc(ox + (x - x0) * s)),
          iy <- trunc(oy + (y - y0) * s)..max(trunc(oy + (y - y0 + 1) * s) - 1, trunc(oy + (y - y0) * s)), into: MapSet.new(), do: {ix, iy}

    on = fn x, y -> MapSet.member?(set, {x, y}) end
    step = div(@cell, @grid)

    edges =
      for gy <- 0..(@grid - 1), gx <- 0..(@grid - 1), dir <- [{1, 0}, {0, 1}, {1, 1}, {1, -1}] do
        {dx, dy} = dir
        Enum.count(for(y <- (gy * step)..(gy * step + step - 1), x <- (gx * step)..(gx * step + step - 1), on.(x, y) and not on.(x + dx, y + dy), do: 1))
      end

    cover = for gy <- 0..(@grid - 1), gx <- 0..(@grid - 1), do: Enum.count(for(y <- (gy * step)..(gy * step + step - 1), x <- (gx * step)..(gx * step + step - 1), on.(x, y), do: 1))
    v = unit(Enum.map(edges ++ cover, &:math.sqrt/1))
    v ++ [:math.log(bw / bh)]
  end

  defp unit(v) do
    n = :math.sqrt(Enum.reduce(v, 0.0, &(&1 * &1 + &2)))
    if n > 0, do: Enum.map(v, &(&1 / n)), else: v
  end

  # ----------------------------------------------------------- templates --

  @doc "Templates from `test/python/math_render.py DIR templates`: one feature vector per rendered symbol."
  def build(dir) do
    {:ok, meta} = Vapor.JSON.decode(File.read!(Path.join(dir, "truth.json")))
    names = meta["classes"] |> Map.keys() |> Enum.sort()

    # one template per class and type family: the mean over its sizes
    templates =
      meta["items"]
      |> Enum.sort()
      |> Enum.flat_map(fn {file, %{"class" => cls, "font" => font}} ->
        {:ok, pic} = Vapor.Docs.Pictures.read(:png, File.read!(Path.join(dir, file)))

        case symbols(pic.image) do
          [sym] -> [{{Enum.find_index(names, &(&1 == cls)), font}, features(sym)}]
          _ -> []
        end
      end)
      |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
      |> Enum.sort()
      |> Enum.map(fn {{c, _font}, vs} ->
        mean = Enum.zip_with(vs, fn col -> Enum.sum(col) / length(col) end)
        {shape, [asp]} = Enum.split(mean, -1)
        {c, unit(shape) ++ [asp]}
      end)

    %__MODULE__{names: List.to_tuple(names), latex: List.to_tuple(Enum.map(names, &meta["classes"][&1])), templates: templates}
  end

  @doc "Save a pack (`config.json` with the classes and the template vectors, rounded to 4 decimals)."
  def save(%__MODULE__{} = p, dir) do
    File.mkdir_p!(dir)
    cfg = %{"model_type" => "vapor_math_templates", "classes" => Tuple.to_list(p.names), "latex" => Tuple.to_list(p.latex),
            "templates" => Enum.map(p.templates, fn {c, v} -> [c, Enum.map(v, &Float.round(&1, 4))] end)}
    File.write!(Path.join(dir, "config.json"), Vapor.JSON.encode(cfg))
    {:ok, dir}
  end

  @doc "Load a pack written by `save/2`."
  def load(dir) do
    with {:ok, body} <- File.read(Path.join(dir, "config.json")),
         {:ok, cfg} <- Vapor.JSON.decode(body) do
      {:ok, %__MODULE__{names: List.to_tuple(cfg["classes"]), latex: List.to_tuple(cfg["latex"]),
                        templates: Enum.map(cfg["templates"], fn [c, v] -> {c, Enum.map(v, &(&1 * 1.0))} end)}}
    end
  end

  @doc "The pack shipped in `priv/math` (cached)."
  def default do
    case :persistent_term.get({__MODULE__, :default}, nil) do
      nil ->
        with {:ok, p} <- load(Path.join(to_string(:code.priv_dir(:vapor)), "math")) do
          :persistent_term.put({__MODULE__, :default}, p)
          {:ok, p}
        end

      p ->
        {:ok, p}
    end
  end

  # cosine of the shape features, less half the difference of log-aspects
  # (so a `-` and an `=`, a `)` and an `∫`, differ by their proportions
  # without the proportion swamping the shape)
  defp similarity(f, t) do
    {fa, [la]} = Enum.split(f, -1)
    {ta, [lb]} = Enum.split(t, -1)
    (Enum.zip_with(fa, ta, &(&1 * &2)) |> Enum.sum()) - 0.5 * abs(la - lb)
  end

  # nearest template: {class index, similarity, margin to the best other class}
  defp classify(p, sym, skip) do
    f = features(sym)
    banned = for {n, i} <- Enum.with_index(Tuple.to_list(p.names)), n in skip, into: MapSet.new(), do: i

    best =
      p.templates
      |> Enum.reject(fn {c, _} -> MapSet.member?(banned, c) end)
      |> Enum.map(fn {c, t} -> {c, similarity(f, t)} end)
      |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
      |> Enum.map(fn {c, ss} -> {c, Enum.max(ss)} end)
      |> Enum.sort_by(&(-elem(&1, 1)))

    [{c, s} | rest] = best
    {c, s, s - (case rest do [{_, s2} | _] -> s2; [] -> 0.0 end)}
  end

  # --------------------------------------------------------------- reading --

  @doc """
  Read a formula image: `{:ok, %{latex, confidence, symbols}}` (`symbols`:
  `[%{box, latex, score}]`), or `{:error, :empty}`. Option `pack` (default
  `default/0`).
  """
  def read(picture, opts \\ []) do
    with {:ok, p} <- (if opts[:pack], do: {:ok, opts[:pack]}, else: default()) do
      case symbols(picture) do
        [] -> {:error, :empty}
        syms ->
          hs = syms |> Enum.reject(&(h(&1) * 4 <= w(&1))) |> Enum.map(&h/1) |> Enum.sort()
          med = Enum.at(hs, div(length(hs), 2)) || 1
          nodes = Enum.map(syms, &node(p, &1, syms, med))
          atoms = for %{kind: :atom} = n <- nodes, do: n
          conf = if atoms == [], do: 1.0, else: atoms |> Enum.map(& &1.score) |> Enum.min()
          {:ok, %{latex: parse(nodes), confidence: conf, symbols: Enum.map(nodes, &Map.take(&1, [:box, :latex, :score, :kind]))}}
      end
    end
  end

  # a symbol as a node: a bar (fraction or minus, decided by the parse), a radical, or a classified atom
  defp node(p, sym, all, med) do
    others = Enum.reject(all, &(&1 == sym))
    {x0, y0, x1, y1} = sym.box

    cond do
      h(sym) * 4 <= w(sym) and w(sym) >= 4 ->
        %{kind: :bar, box: sym.box, latex: "-", score: 1.0}

      # a radical: hollow, and wholly enclosing at least one other mark
      Enum.any?(others, fn %{box: {a0, b0, a1, b1}} -> a0 > x0 and a1 <= x1 and b0 >= y0 and b1 <= y1 end) and sym.area < 0.35 * w(sym) * h(sym) ->
        %{kind: :radical, box: sym.box, latex: "\\sqrt", score: 1.0}

      # a big operator is big, and a sum carries its limits above and below
      h(sym) >= 1.3 * med and Enum.any?(others, &above?(&1, sym)) and Enum.any?(others, &below?(&1, sym)) ->
        %{kind: :atom, box: sym.box, class: "sum", latex: "\\sum", score: 1.0, margin: 1.0}

      true ->
        # ∑ and ∫ only for marks taller than the formula's typical mark
        skip = if h(sym) < 1.3 * med, do: ["sum", "int"], else: []
        {c, s, m} = classify(p, sym, skip)
        %{kind: :atom, box: sym.box, class: elem(p.names, c), latex: elem(p.latex, c), score: s, margin: m}
    end
  end

  defp above?(o, s), do: elem(o.box, 3) < elem(s.box, 1) and cx(o) >= elem(s.box, 0) and cx(o) <= elem(s.box, 2) and elem(s.box, 1) - elem(o.box, 3) < 0.6 * h(s)
  defp below?(o, s), do: elem(o.box, 1) > elem(s.box, 3) and cx(o) >= elem(s.box, 0) and cx(o) <= elem(s.box, 2) and elem(o.box, 1) - elem(s.box, 3) < 0.6 * h(s)

  @doc false
  def parse([]), do: ""

  def parse(nodes) do
    nodes |> structure() |> big_ops() |> Enum.sort_by(fn %{box: {x0, _, _, _}} -> x0 end) |> line()
  end

  # Fractions and radicals, outermost first: of every bar with marks above
  # and below it and every radical, the widest is taken apart first (a
  # fraction's bar spans its numerator and denominator, a radical its
  # contents, so the outer one is always the wider), its contents parsed on their own (a radical over a
  # fraction, a fraction of radicals, fractions of fractions).
  defp structure(nodes) do
    fracs =
      for %{kind: :bar} = bar <- nodes,
          {x0, y0, x1, _} = bar.box,
          within = fn n -> n != bar and cx(n) >= x0 - 1 and cx(n) <= x1 + 1 end,
          num = Enum.filter(nodes, &(within.(&1) and cy(&1) < y0)),
          den = Enum.filter(nodes, &(within.(&1) and cy(&1) > y0)),
          num != [] and den != [] do
        box = Enum.reduce(num ++ den, bar.box, fn n, b -> union(b, n.box) end)
        {w(bar), fn -> [%{kind: :group, box: box, latex: "\\frac{#{parse(num)}}{#{parse(den)}}"} | (nodes -- num) -- [bar | den]] end}
      end

    rads =
      for %{kind: :radical} = r <- nodes do
        {x0, y0, x1, y1} = r.box
        inside = Enum.filter(nodes, fn n -> n != r and cx(n) > x0 and cx(n) < x1 and cy(n) > y0 and cy(n) < y1 end)
        {w(r), fn -> [%{kind: :group, box: r.box, latex: "\\sqrt{#{parse(inside)}}"} | nodes -- [r | inside]] end}
      end

    case Enum.sort_by(fracs ++ rads, &(-elem(&1, 0))) do
      [] -> nodes
      [{_, take} | _] -> structure(take.())
    end
  end


  # a sum's limits sit above and below it
  defp big_ops(nodes) do
    case Enum.find(nodes, &(Map.get(&1, :class) == "sum" and not Map.get(&1, :done, false))) do
      nil -> nodes
      s ->
        {x0, y0, x1, y1} = s.box
        sh = y1 - y0 + 1
        # a limit is a row of marks just above or below, reaching past the
        # operator's sides when it is wider (i = 1 under a narrow ∑)
        reach = fn n -> elem(n.box, 2) >= x0 - 0.8 * sh and elem(n.box, 0) <= x1 + 0.8 * sh end
        up = Enum.filter(nodes, &(&1 != s and reach.(&1) and elem(&1.box, 3) < y0 and y0 - elem(&1.box, 3) < 0.7 * sh))
        down = Enum.filter(nodes, &(&1 != s and reach.(&1) and elem(&1.box, 1) > y1 and elem(&1.box, 1) - y1 < 0.7 * sh))
        tex = "\\sum" <> if(down != [], do: "_{#{parse(down)}}", else: "") <> if(up != [], do: "^{#{parse(up)}}", else: "")
        box = Enum.reduce(up ++ down, s.box, fn n, b -> union(b, n.box) end)
        big_ops([Map.merge(s, %{kind: :group, latex: tex, box: {elem(box, 0), y0, elem(box, 2), y1}, done: true}) | ((nodes -- [s]) -- (up ++ down))])
    end
  end

  # left to right: each node on the base's line, or a script of it
  defp line([]), do: ""

  defp line([base | rest]) do
    {scripts, rest} = Enum.split_while(rest, &(level(base, &1) != :line))
    {sup, sub} = Enum.split_with(scripts, &(level(base, &1) == :sup))
    tex = latex(base) <> if(sub != [], do: "_{#{parse(sub)}}", else: "") <> if(sup != [], do: "^{#{parse(sup)}}", else: "")
    tex <> line(rest)
  end

  defp latex(%{kind: :bar}), do: "-"
  defp latex(%{latex: l}), do: l

  # where n sits relative to the base's body — its box without the
  # descender of a y, μ, β, γ or a parenthesis, without the ascender of a
  # d, k, t, λ, β or a digit — raised, lowered or on its line
  @descends ["y", "beta", "gamma", "mu", "(", ")", "int"]
  @ascends ["b", "d", "k", "t", "lambda", "beta", "theta", "(", ")", "int", "sum"] ++ Enum.map(0..9, &Integer.to_string/1)

  defp body(%{box: {x0, y0, x1, y1}} = n) do
    hh = y1 - y0 + 1
    cls = Map.get(n, :class)
    y1 = if cls in @descends, do: y1 - 0.25 * hh, else: y1
    y0 = if cls in @ascends, do: y0 + 0.3 * hh, else: y0
    {x0, y0, x1, y1}
  end

  defp level(base, n) do
    {_, b0, _, b1} = body(base)
    {_, n0, _, n1} = body(n)
    bh = b1 - b0 + 1
    c = (b0 + b1) / 2
    nc = (n0 + n1) / 2
    k = if Map.get(base, :class) == "int", do: 0.2, else: 0.35

    cond do
      Map.get(n, :kind) in [:group] and h(n) >= 0.8 * h(base) -> :line
      nc < c - k * bh and n1 < b1 - 0.2 * bh -> :sup
      nc > c + k * bh and n0 > b0 + 0.2 * bh -> :sub
      true -> :line
    end
  end

  defp union({a0, b0, a1, b1}, {c0, d0, c1, d1}), do: {min(a0, c0), min(b0, d0), max(a1, c1), max(b1, d1)}

  @doc "LaTeX tokens (commands, braces, scripts, single characters) — for edit distances."
  def tokens(tex), do: Regex.scan(~r/\\[a-zA-Z]+|[{}^_]|\S/u, tex) |> List.flatten()
end
