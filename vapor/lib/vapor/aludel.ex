defmodule Vapor.Aludel do
  @moduledoc """
  The **aludel** — the vessel in which a volatile substance is *fixed* —
  for claims about polynomials on boxes: "`p(x) ≥ 0` for every `x` in this
  box", decided in exact integer arithmetic, with a witness anyone can
  replay (docs/ALUDEL.md). Absorbed from PALADIN's kernel; rewritten here
  on the BEAM's integers.

  The pain it answers is the one behind every "provably safe controller"
  claim: a sampled check says nothing between the samples, and floating
  point says nothing near zero. The procedure:

    * **Bernstein enclosure.** On a box, `p` is a combination of Bernstein
      polynomials, which are non-negative and sum to one; so `p` lies between
      its smallest and largest Bernstein coefficient, and the coefficients at
      the box's vertices *are* `p` there. One exact conversion from the power
      basis (rationals, then integers over a common denominator).
    * **Three verdicts, never two.** All coefficients `≥ 0` (or `> 0` for
      strict claims): **certified** on the cell. A vertex coefficient `< 0`:
      **refuted**, with that vertex — an exact point and value. Otherwise the
      cell is split at the midpoint of its widest axis by de Casteljau's
      algorithm (integer additions and shifts only) and both halves are
      decided. A cell that reaches the budget is **exhausted**: a verdict with
      a name and the cell, never a longer wait and never a guess.
    * **The witness is the subdivision tree** (one bit per cell, depth-first).
      `check/4` replays it without searching and, at each leaf, recomputes the
      Bernstein coefficients **directly** from the polynomial on that leaf's
      box — a different computation from the search's subdivisions, with the
      same theorem behind it.

  On top: `enclose/3` (a rigorous range), and `barrier/2` + `synthesize/2`
  — **barrier certificates** for polynomial vector fields (Prajna &
  Jadbabaie, 2004): a function `B` with `B ≤ 0` on the initial set, `B > 0`
  on the unsafe set and `λB − ∇B·f ≥ 0` on the domain proves that no
  trajectory from the initial set reaches the unsafe set without first
  leaving the domain. Each condition is a positivity claim on a box. A
  candidate `B` comes from the person, from a model, or from `synthesize/2`
  (an exact LP over `B`'s coefficients whose rows are Bernstein
  coefficients, grown by constraint generation) — and in every case is
  accepted only by the same decision.

  What this is **not**: a model of an aircraft, a plasma or a cortex. It
  proves properties *of the polynomial system it is given*; whether that
  system describes the world is a separate claim, and is stated as such.
  """
  import Bitwise
  alias Vapor.Logic.LP

  # ===================================================== polynomials

  defmodule P do
    @moduledoc "A polynomial in `n` variables: `%{exponent tuple => rational {num, den}}`, zero terms absent."
    defstruct n: 0, terms: %{}
  end

  @max_degree 24
  @max_terms 20_000

  @doc "The polynomial of a rational constant."
  def const(n, c), do: norm(%P{n: n, terms: %{zeros(n) => LP.rat(c)}})
  @doc "The polynomial `xᵢ`."
  def var(n, i), do: %P{n: n, terms: %{put_elem(zeros(n), i, 1) => {1, 1}}}

  defp zeros(n), do: List.to_tuple(List.duplicate(0, n))

  @doc "Sum."
  def add(%P{n: n} = a, %P{n: n} = b), do: norm(%P{n: n, terms: Map.merge(a.terms, b.terms, fn _, x, y -> LP.qadd(x, y) end)})
  @doc "Difference."
  def sub(a, b), do: add(a, scale(b, {-1, 1}))
  @doc "Scalar multiple."
  def scale(%P{} = a, c), do: norm(%{a | terms: Map.new(a.terms, fn {e, x} -> {e, LP.qmul(x, LP.rat(c))} end)})

  @doc "Product (refused past #{@max_degree} per variable or #{@max_terms} terms)."
  def mul(%P{n: n} = a, %P{n: n} = b) do
    terms =
      for {e1, c1} <- a.terms, {e2, c2} <- b.terms, reduce: %{} do
        acc ->
          e = add_exp(e1, e2)
          Map.update(acc, e, LP.qmul(c1, c2), &LP.qadd(&1, LP.qmul(c1, c2)))
      end

    p = norm(%P{n: n, terms: terms})
    guard!(p)
  end

  @doc "Integer power."
  def pow(%P{n: n}, 0), do: const(n, 1)
  def pow(p, k) when k > 0, do: Enum.reduce(2..k//1, p, fn _, acc -> mul(acc, p) end)

  @doc "Partial derivative with respect to variable `i`."
  def diff(%P{} = p, i) do
    terms =
      for {e, c} <- p.terms, elem(e, i) > 0, into: %{} do
        {put_elem(e, i, elem(e, i) - 1), LP.qmul(c, {elem(e, i), 1})}
      end

    norm(%{p | terms: terms})
  end

  @doc "Degree in each variable."
  def degrees(%P{n: n, terms: t}), do: Enum.reduce(Map.keys(t), List.duplicate(0, n), fn e, ds -> Enum.zip_with(ds, Tuple.to_list(e), &max/2) end)

  @doc "Exact value at a point of rationals."
  def eval(%P{terms: t}, point) do
    Enum.reduce(t, {0, 1}, fn {e, c}, acc ->
      m = e |> Tuple.to_list() |> Enum.zip(point) |> Enum.reduce({1, 1}, fn {k, x}, m -> LP.qmul(m, qpow(x, k)) end)
      LP.qadd(acc, LP.qmul(c, m))
    end)
  end

  defp qpow(_x, 0), do: {1, 1}
  defp qpow(x, k), do: LP.qmul(x, qpow(x, k - 1))

  defp add_exp(a, b), do: a |> Tuple.to_list() |> Enum.zip_with(Tuple.to_list(b), &+/2) |> List.to_tuple()
  defp norm(%P{} = p), do: %{p | terms: Map.reject(p.terms, fn {_, c} -> LP.qzero?(c) end)}

  defp guard!(p) do
    cond do
      map_size(p.terms) > @max_terms -> throw({:aludel, "more than #{@max_terms} terms"})
      Enum.any?(degrees(p), &(&1 > @max_degree)) -> throw({:aludel, "degree above #{@max_degree} in one variable"})
      true -> p
    end
  end

  # ===================================================== the text form

  @doc """
  Parse a polynomial over the named variables: `+ - * ^` (non-negative
  integer powers), parentheses, constants as integers, decimals or
  fractions (`3/4`; division only by constants). `{:ok, p}` or `{:error, why}`.
  """
  def parse(text, vars) when is_binary(text) and is_list(vars) do
    n = length(vars)
    index = vars |> Enum.with_index() |> Map.new()

    with {:ok, toks} <- tokens(String.replace(text, "−", "-"), []),
         {:ok, p, []} <- sum(toks, n, index) do
      {:ok, p}
    else
      {:ok, _, [t | _]} -> {:error, "unexpected #{inspect(t)}"}
      {:error, _} = e -> e
    end
  catch
    {:aludel, why} -> {:error, why}
  end

  defp tokens("", acc), do: {:ok, Enum.reverse(acc)}
  defp tokens(<<c, r::binary>>, acc) when c in [?\s, ?\t, ?\n], do: tokens(r, acc)
  defp tokens(<<c, r::binary>>, acc) when c in [?+, ?-, ?*, ?/, ?^, ?(, ?)], do: tokens(r, [<<c>> | acc])

  defp tokens(<<c, _::binary>> = s, acc) when c in ?0..?9 or c == ?. do
    [num] = Regex.run(~r/^\d*\.?\d+(?:[eE][-+]?\d+)?|^\d+\.?/, s)
    tokens(binary_part(s, byte_size(num), byte_size(s) - byte_size(num)), [{:num, num} | acc])
  end

  defp tokens(<<c, _::binary>> = s, acc) when c in ?a..?z or c in ?A..?Z or c == ?_ do
    [w] = Regex.run(~r/^[A-Za-z_][A-Za-z0-9_]*/, s)
    tokens(binary_part(s, byte_size(w), byte_size(s) - byte_size(w)), [{:id, w} | acc])
  end

  defp tokens(<<c::utf8, _::binary>>, _), do: {:error, "unexpected character #{inspect(<<c::utf8>>)}"}

  defp sum(toks, n, ix) do
    {sign, toks} =
      case toks do
        ["-" | r] -> {-1, r}
        ["+" | r] -> {1, r}
        _ -> {1, toks}
      end

    with {:ok, t, rest} <- product(toks, n, ix), do: sum_rest(scale(t, sign), rest, n, ix)
  end

  defp sum_rest(acc, [op | r], n, ix) when op in ["+", "-"] do
    with {:ok, t, rest} <- product(r, n, ix), do: sum_rest(if(op == "+", do: add(acc, t), else: sub(acc, t)), rest, n, ix)
  end

  defp sum_rest(acc, rest, _n, _ix), do: {:ok, acc, rest}

  defp product(toks, n, ix) do
    with {:ok, f, rest} <- power(toks, n, ix), do: product_rest(f, rest, n, ix)
  end

  defp product_rest(acc, ["*" | r], n, ix), do: with({:ok, f, rest} <- power(r, n, ix), do: product_rest(mul(acc, f), rest, n, ix))

  defp product_rest(acc, ["/" | r], n, ix) do
    with {:ok, f, rest} <- power(r, n, ix) do
      case Map.keys(f.terms) do
        [] -> {:error, "division by zero"}
        [e] -> if Tuple.to_list(e) |> Enum.all?(&(&1 == 0)), do: product_rest(scale(acc, LP.qdiv({1, 1}, f.terms[e])), rest, n, ix), else: {:error, "division only by constants"}
        _ -> {:error, "division only by constants"}
      end
    end
  end

  # implicit product: "2x", "3(x+1)"
  defp product_rest(acc, [t | _] = toks, n, ix) when t == "(" or (is_tuple(t) and elem(t, 0) == :id),
    do: with({:ok, f, rest} <- power(toks, n, ix), do: product_rest(mul(acc, f), rest, n, ix))

  defp product_rest(acc, rest, _n, _ix), do: {:ok, acc, rest}

  defp power(toks, n, ix) do
    with {:ok, b, rest} <- atom(toks, n, ix) do
      case rest do
        ["^", {:num, k} | r] ->
          case Integer.parse(k) do
            {e, ""} when e >= 0 and e <= @max_degree -> {:ok, pow(b, e), r}
            _ -> {:error, "exponent #{k}: a non-negative integer up to #{@max_degree}"}
          end

        ["^" | _] -> {:error, "an exponent must be an integer literal"}
        _ -> {:ok, b, rest}
      end
    end
  end

  defp atom([{:num, s} | r], n, _ix), do: {:ok, const(n, LP.rat(s)), r}

  defp atom([{:id, v} | r], n, ix) do
    case Map.fetch(ix, v) do
      {:ok, i} -> {:ok, var(n, i), r}
      :error -> {:error, "unknown variable #{v} (declared: #{Enum.join(Map.keys(ix), ", ")})"}
    end
  end

  defp atom(["(" | r], n, ix) do
    case sum(r, n, ix) do
      {:ok, p, [")" | rest]} -> {:ok, p, rest}
      {:ok, _, _} -> {:error, "missing )"}
      e -> e
    end
  end

  defp atom(["-" | r], n, ix), do: with({:ok, p, rest} <- atom(r, n, ix), do: {:ok, scale(p, -1), rest})
  defp atom([t | _], _n, _ix), do: {:error, "unexpected #{inspect(t)}"}
  defp atom([], _n, _ix), do: {:error, "expression ends early"}

  @doc "A box from `[{lo, hi}]` (rationals, decimals or fraction strings): `{:ok, box}` or `{:error, why}`."
  def box(pairs) do
    bx = Enum.map(pairs, fn {lo, hi} -> {LP.rat(lo), LP.rat(hi)} end)
    if Enum.all?(bx, fn {lo, hi} -> LP.qcmp(lo, hi) < 0 end), do: {:ok, bx}, else: {:error, "every interval needs lo < hi"}
  end

  # ===================================================== Bernstein form

  # dense from the start: the power-basis array over `dims`, then per axis
  # the affine change x = lo + w·t (a Taylor shift of every line), then per
  # axis the Bernstein transform b_j = Σ_{i≤j} C(j,i)/C(d,i) a_i; rationals,
  # finally integers over one common denominator
  defp bernstein(%P{n: n} = p, box, dims) do
    strides = strides(dims)
    size = Enum.reduce(dims, 1, &(&2 * (&1 + 1)))

    arr =
      Enum.reduce(p.terms, :array.new(size, default: {0, 1}), fn {e, c}, a ->
        el = Tuple.to_list(e)
        if Enum.zip(el, dims) |> Enum.any?(fn {k, d} -> k > d end), do: throw({:aludel, "degree exceeds the declared multi-degree"})
        :array.set(index(el, strides), c, a)
      end)

    arr =
      Enum.reduce(Enum.with_index(box), arr, fn {{lo, hi}, k}, a -> shift_axis(a, dims, strides, k, lo, LP.qsub(hi, lo)) end)

    arr = Enum.reduce(0..(n - 1)//1, arr, fn k, a -> transform_axis(a, dims, strides, k) end)
    vals = :array.to_list(arr)
    den = Enum.reduce(vals, 1, fn {_, d}, l -> div(l * d, Integer.gcd(l, d)) end)
    {Enum.map(vals, fn {num, d} -> num * div(den, d) end) |> List.to_tuple(), den}
  end

  # along axis k: a(x) → a(lo + w·t), c_j = w^j · Σ_{i≥j} C(i,j) lo^(i−j) a_i
  defp shift_axis(arr, dims, strides, k, lo, w) do
    d = Enum.at(dims, k)
    sk = Enum.at(strides, k)
    lop = Enum.scan(1..d//1, {1, 1}, fn _, acc -> LP.qmul(acc, lo) end) |> then(&List.to_tuple([{1, 1} | &1]))
    wp = Enum.scan(1..d//1, {1, 1}, fn _, acc -> LP.qmul(acc, w) end) |> then(&List.to_tuple([{1, 1} | &1]))

    Enum.reduce(line_bases(dims, strides, k), arr, fn base, a ->
      line = for(j <- 0..d, do: :array.get(base + j * sk, a)) |> List.to_tuple()

      if Enum.all?(Tuple.to_list(line), &LP.qzero?/1) do
        a
      else
        Enum.reduce(0..d, a, fn j, a2 ->
          s = Enum.reduce(j..d, {0, 1}, fn i, acc -> LP.qadd(acc, LP.qmul({binom(i, j), 1}, LP.qmul(elem(lop, i - j), elem(line, i)))) end)
          :array.set(base + j * sk, LP.qmul(s, elem(wp, j)), a2)
        end)
      end
    end)
  end

  defp strides(dims) do
    {ss, _} = dims |> Enum.reverse() |> Enum.map_reduce(1, fn d, s -> {s, s * (d + 1)} end)
    Enum.reverse(ss)
  end

  defp index(js, strides), do: Enum.zip_reduce(js, strides, 0, fn j, s, acc -> acc + j * s end)

  defp line_bases(dims, strides, k) do
    Enum.reduce(Enum.with_index(dims), [0], fn {d, i}, acc ->
      if i == k, do: acc, else: for(b <- acc, j <- 0..d, do: b + j * Enum.at(strides, i))
    end)
  end

  defp transform_axis(arr, dims, strides, k) do
    d = Enum.at(dims, k)
    sk = Enum.at(strides, k)

    Enum.reduce(line_bases(dims, strides, k), arr, fn base, a ->
      line = for j <- 0..d, do: :array.get(base + j * sk, a)
      lt = List.to_tuple(line)

      Enum.reduce(0..d, a, fn j, a2 ->
        v = Enum.reduce(0..j, {0, 1}, fn i, s -> LP.qadd(s, LP.qmul(LP.q(binom(j, i), binom(d, i)), elem(lt, i))) end)
        :array.set(base + j * sk, v, a2)
      end)
    end)
  end

  defp binom(n, k) when k < 0 or k > n, do: 0
  defp binom(n, k), do: Enum.reduce(1..k//1, 1, fn i, acc -> div(acc * (n - k + i), i) end)

  # vertex coefficients: every index with each coordinate 0 or d_i
  defp vertices(dims) do
    Enum.reduce(dims, [[]], fn d, acc -> for v <- acc, j <- Enum.uniq([0, d]), do: v ++ [j] end)
  end

  # midpoint de Casteljau along axis k: integers, the denominator ×2^d
  defp split({coeffs, den}, dims, k) do
    strides = strides(dims)
    d = Enum.at(dims, k)
    sk = Enum.at(strides, k)

    {left, right} =
      Enum.reduce(line_bases(dims, strides, k), {coeffs, coeffs}, fn base, {l, r} ->
        tri = for j <- 0..d, do: elem(coeffs, base + j * sk)
        {ls, rs} = casteljau(tri, d)
        l = Enum.reduce(Enum.with_index(ls), l, fn {v, j}, t -> put_elem(t, base + j * sk, v) end)
        r = Enum.reduce(Enum.with_index(rs), r, fn {v, j}, t -> put_elem(t, base + j * sk, v) end)
        {l, r}
      end)

    {{left, den <<< d}, {right, den <<< d}}
  end

  # L_r = v⁽ʳ⁾₀·2^(d−r), R_r = v⁽ᵈ⁻ʳ⁾_r·2^r with v⁽ʳ⁾ᵢ = v⁽ʳ⁻¹⁾ᵢ + v⁽ʳ⁻¹⁾ᵢ₊₁
  defp casteljau(tri, d) do
    levels = Enum.scan(1..d//1, tri, fn _, cur -> Enum.zip_with(cur, tl(cur), &+/2) end)
    levels = [tri | levels]
    ls = for r <- 0..d, do: hd(Enum.at(levels, r)) <<< (d - r)
    rs = for r <- 0..d, do: Enum.at(Enum.at(levels, d - r), r) <<< r
    {ls, rs}
  end

  # ===================================================== the decision

  @doc """
  Decide `p ≥ 0` (`sense: :nonneg`, default) or `p > 0` (`sense: :pos`) on
  `box`. Returns

    * `{:certified, %{witness, cells, depth}}` — the witness is replayed by `check/4`;
    * `{:refuted, %{point, value}}` — an exact point (a vertex of some cell) where the claim fails;
    * `{:exhausted, %{cell, cells, depth}}` — the budget ran out on `cell` (a sub-box).

  Options: `depth:` (24), `cells:` (65 536), `dims:` (the multi-degree; default `p`'s own).
  """
  def decide(%P{} = p, box, opts \\ []) do
    sense = Keyword.get(opts, :sense, :nonneg)
    dims = Keyword.get(opts, :dims) || degrees(p)
    maxd = Keyword.get(opts, :depth, 24)
    maxc = Keyword.get(opts, :cells, 65_536)
    root = bernstein(p, box, dims)
    unit = List.duplicate({{0, 1}, {1, 1}}, p.n)

    case walk([{root, unit, 0}], dims, sense, box, maxd, maxc, %{bits: [], cells: 0, depth: 0, pending: nil}) do
      {:refuted, pt, v} -> {:refuted, %{point: pt, value: v}}
      %{pending: nil} = st -> {:certified, %{witness: encode_bits(Enum.reverse(st.bits)), cells: st.cells, depth: st.depth}}
      %{pending: cell} = st -> {:exhausted, %{cell: to_box(cell, box), cells: st.cells, depth: st.depth}}
    end
  catch
    {:aludel, why} -> {:error, why}
  end

  defp walk([], _dims, _sense, _box, _maxd, _maxc, st), do: st

  defp walk([{bf, cell, depth} | rest], dims, sense, box, maxd, maxc, st) do
    {coeffs, den} = bf
    st = %{st | cells: st.cells + 1, depth: max(st.depth, depth)}
    strides = strides(dims)

    bad_vertex =
      Enum.find(vertices(dims), fn v -> c = elem(coeffs, index(v, strides)); if sense == :pos, do: c <= 0, else: c < 0 end)

    cond do
      bad_vertex != nil ->
        pt = to_point(bad_vertex, dims, cell, box)
        {:refuted, pt, LP.q(elem(coeffs, index(bad_vertex, strides)), den)}

      ok?(coeffs, sense) ->
        walk(rest, dims, sense, box, maxd, maxc, %{st | bits: [0 | st.bits]})

      depth >= maxd or st.cells >= maxc ->
        # a pending cell does not stop the search: a refutation elsewhere is the stronger answer
        st = %{st | bits: [0 | st.bits], pending: st.pending || cell}
        walk(rest, dims, sense, box, maxd, maxc, st)

      true ->
        k = widest(cell)
        {l, r} = split(bf, dims, k)
        {lc, rc} = halve(cell, k)
        walk([{l, lc, depth + 1}, {r, rc, depth + 1} | rest], dims, sense, box, maxd, maxc, %{st | bits: [1 | st.bits]})
    end
  end

  defp ok?(coeffs, :pos), do: coeffs |> Tuple.to_list() |> Enum.all?(&(&1 > 0))
  defp ok?(coeffs, _), do: coeffs |> Tuple.to_list() |> Enum.all?(&(&1 >= 0))

  # the widest axis of a cell (in unit coordinates), ties to the lowest index
  defp widest(cell) do
    cell
    |> Enum.with_index()
    |> Enum.reduce({-1, nil}, fn {{lo, hi}, i}, {best, w} ->
      wi = LP.qsub(hi, lo)
      if w == nil or LP.qcmp(wi, w) > 0, do: {i, wi}, else: {best, w}
    end)
    |> elem(0)
  end

  defp halve(cell, k) do
    {lo, hi} = Enum.at(cell, k)
    mid = LP.qmul(LP.qadd(lo, hi), {1, 2})
    {List.replace_at(cell, k, {lo, mid}), List.replace_at(cell, k, {mid, hi})}
  end

  defp to_box(cell, box) do
    Enum.zip_with(cell, box, fn {ulo, uhi}, {lo, hi} ->
      w = LP.qsub(hi, lo)
      {LP.qadd(lo, LP.qmul(ulo, w)), LP.qadd(lo, LP.qmul(uhi, w))}
    end)
  end

  defp to_point(vertex, dims, cell, box) do
    Enum.zip([vertex, dims, to_box(cell, box)]) |> Enum.map(fn {j, d, {lo, hi}} -> if j == 0 and d >= 0, do: lo, else: hi end)
  end

  defp encode_bits(bits) do
    pad = rem(8 - rem(length(bits), 8), 8)
    bin = for b <- bits ++ List.duplicate(0, pad), into: <<>>, do: <<b::1>>
    %{bits: length(bits), tree: Base.encode16(bin, case: :lower)}
  end

  defp decode_bits(%{bits: n, tree: hex}) do
    with {:ok, bin} <- Base.decode16(hex, case: :mixed),
         true <- bit_size(bin) >= n do
      {:ok, for(<<b::1 <- bin>>, do: b) |> Enum.take(n)}
    else
      _ -> :error
    end
  end

  # ===================================================== the check

  @doc """
  Replay a witness without searching: walk the subdivision tree it encodes
  and, at every leaf, convert `p` to the Bernstein basis **on that leaf's
  box directly** and confirm the sign condition. `:ok` or `{:error, why}`.
  """
  def check(%P{} = p, box, witness, opts \\ []) do
    sense = Keyword.get(opts, :sense, :nonneg)
    dims = Keyword.get(opts, :dims) || degrees(p)

    with {:ok, bits} <- decode_bits(witness) do
      unit = List.duplicate({{0, 1}, {1, 1}}, p.n)

      case replay(bits, [unit], p, box, dims, sense) do
        {:ok, []} -> :ok
        {:ok, _} -> {:error, "the witness has bits left over"}
        e -> e
      end
    else
      :error -> {:error, "an unreadable witness"}
    end
  catch
    {:aludel, why} -> {:error, why}
  end

  defp replay(bits, [], _p, _box, _dims, _sense), do: {:ok, bits}
  defp replay([], [_ | _], _p, _box, _dims, _sense), do: {:error, "the witness ends before the tree does"}

  defp replay([1 | bits], [cell | rest], p, box, dims, sense) do
    {l, r} = halve(cell, widest(cell))
    replay(bits, [l, r | rest], p, box, dims, sense)
  end

  defp replay([0 | bits], [cell | rest], p, box, dims, sense) do
    {coeffs, _} = bernstein(p, to_box(cell, box), dims)
    if ok?(coeffs, sense), do: replay(bits, rest, p, box, dims, sense), else: {:error, "a leaf of the witness is not certified"}
  end

  @doc """
  The leaf cells of a witness, as boxes of rationals in the original
  coordinates (what a picture of the subdivision draws). Does not check.
  """
  def cells(witness, box) do
    with {:ok, bits} <- decode_bits(witness) do
      unit = List.duplicate({{0, 1}, {1, 1}}, length(box))
      {:ok, leaves(bits, [unit], []) |> Enum.map(&to_box(&1, box))}
    else
      :error -> {:error, "an unreadable witness"}
    end
  end

  defp leaves(_bits, [], acc), do: Enum.reverse(acc)
  defp leaves([], _cells, acc), do: Enum.reverse(acc)

  defp leaves([1 | bits], [cell | rest], acc) do
    {l, r} = halve(cell, widest(cell))
    leaves(bits, [l, r | rest], acc)
  end

  defp leaves([0 | bits], [cell | rest], acc), do: leaves(bits, rest, [cell | acc])

  @doc "The polynomial as text over the named variables (exact coefficients)."
  def to_text(%P{terms: t}, vars) when map_size(t) == 0 and is_list(vars), do: "0"

  def to_text(%P{terms: t}, vars) do
    t
    |> Enum.sort_by(fn {e, _} -> {-Enum.sum(Tuple.to_list(e)), Tuple.to_list(e) |> Enum.map(&(-&1))} end)
    |> Enum.with_index()
    |> Enum.map_join("", fn {{e, c}, i} ->
      mono = e |> Tuple.to_list() |> Enum.zip(vars) |> Enum.flat_map(fn {k, v} -> case k do 0 -> []; 1 -> [v]; _ -> ["#{v}^#{k}"] end end)
      neg = LP.qsign(c) < 0
      a = if neg, do: LP.qneg(c), else: c
      body = cond do
        mono == [] -> LP.show(a)
        a == {1, 1} -> Enum.join(mono, "*")
        true -> LP.show(a) <> "*" <> Enum.join(mono, "*")
      end
      cond do
        i == 0 and neg -> "-" <> body
        i == 0 -> body
        neg -> " - " <> body
        true -> " + " <> body
      end
    end)
  end

  @doc "The value at a point of floats (for pictures; decisions are exact)."
  def eval_float(%P{terms: t}, xs) do
    Enum.reduce(t, 0.0, fn {e, {n, d}}, acc ->
      acc + n / d * (e |> Tuple.to_list() |> Enum.zip(xs) |> Enum.reduce(1.0, fn {k, x}, m -> m * :math.pow(x, k) end))
    end)
  end

  # ===================================================== ranges

  @doc """
  A rigorous enclosure `{lo, hi}` of `p` on `box`: the extreme Bernstein
  coefficients over a uniform subdivision of `depth` halvings per axis
  (default 3). Rationals.
  """
  def enclose(%P{} = p, box, depth \\ 3) do
    dims = degrees(p)
    cells = uniform(box, depth)

    cells
    |> Enum.map(fn c -> {cs, den} = bernstein(p, c, dims); l = Tuple.to_list(cs); {LP.q(Enum.min(l), den), LP.q(Enum.max(l), den)} end)
    |> Enum.reduce(fn {a, b}, {lo, hi} -> {if(LP.qcmp(a, lo) < 0, do: a, else: lo), if(LP.qcmp(b, hi) > 0, do: b, else: hi)} end)
  end

  defp uniform(box, depth) do
    Enum.reduce(box, [[]], fn {lo, hi}, acc ->
      k = 1 <<< depth
      w = LP.qsub(hi, lo)
      pieces = for i <- 0..(k - 1), do: {LP.qadd(lo, LP.qmul(w, LP.q(i, k))), LP.qadd(lo, LP.qmul(w, LP.q(i + 1, k)))}
      for c <- acc, piece <- pieces, do: c ++ [piece]
    end)
  end

  # ===================================================== barrier certificates

  @doc """
  Check a barrier certificate. `sys` is `%{vars, field, domain, init,
  unsafe}` (`field` a list of polynomials — `ẋᵢ = fᵢ(x)`; boxes as lists
  of `{lo, hi}`), `b` the candidate polynomial, option `lambda:` (a
  rational, default 0). Returns `%{verdict: :proved | :refuted | :unknown,
  conditions: [%{name, claim, result}]}`; `:proved` only when all three
  conditions are certified.
  """
  def barrier(%{field: f, domain: d, init: i, unsafe: u} = _sys, %P{} = b, opts \\ []) do
    lambda = LP.rat(Keyword.get(opts, :lambda, 0))
    lie = lie_derivative(b, f)
    budget = Keyword.take(opts, [:depth, :cells])

    conds = [
      {"initial", "B ≤ 0 on the initial set", scale(b, -1), i, :nonneg},
      {"unsafe", "B > 0 on the unsafe set", b, u, :pos},
      {"flow", "λ·B − ∇B·f ≥ 0 on the domain", sub(scale(b, lambda), lie), d, :nonneg}
    ]

    results =
      for {name, claim, poly, bx, sense} <- conds do
        {:ok, bx} = box(bx)
        %{name: name, claim: claim, result: decide(poly, bx, Keyword.merge(budget, sense: sense)), polynomial: poly, box: bx, sense: sense}
      end

    verdict =
      cond do
        Enum.all?(results, &match?({:certified, _}, &1.result)) -> :proved
        Enum.any?(results, &match?({:refuted, _}, &1.result)) -> :refuted
        true -> :unknown
      end

    %{verdict: verdict, conditions: results, lambda: lambda}
  end

  @doc "`∇B · f` — the derivative of `B` along the flow."
  def lie_derivative(%P{} = b, field) do
    field |> Enum.with_index() |> Enum.reduce(const(b.n, 0), fn {fi, i}, acc -> add(acc, mul(diff(b, i), fi)) end)
  end

  @doc """
  Look for a barrier certificate of total degree `degree` (default 2): an
  exact LP over `B`'s coefficients whose rows say "every Bernstein
  coefficient of each condition, on each cell of a `split`-times halved
  subdivision, has the right sign" (`B ≥ 1` on the unsafe set fixes the
  scale). Rows are added by **constraint generation** — solve on a few,
  find the most violated of all, add it, repeat — so the LP stays small.
  A solution is then **re-decided** by `barrier/3`: the LP proposes, the
  decision accepts. Returns `{:ok, b, report}` or `{:error, why}`.
  """
  def synthesize(%{vars: vars, field: f, domain: d, init: i, unsafe: u} = sys, opts \\ []) do
    n = length(vars)
    deg = Keyword.get(opts, :degree, 2)
    lambda = LP.rat(Keyword.get(opts, :lambda, 0))
    splits = Keyword.get(opts, :split, 1)
    bound = LP.rat(Keyword.get(opts, :bound, 100))
    monos = monomials(n, deg)
    names = for k <- 0..(length(monos) - 1), do: "c#{k}"

    # per monomial m: its contribution to each condition polynomial
    contrib = fn m ->
      mp = %P{n: n, terms: %{m => {1, 1}}}
      %{initial: scale(mp, -1), unsafe: mp, flow: sub(scale(mp, lambda), lie_derivative(mp, f))}
    end

    parts = Enum.map(monos, contrib)

    rows =
      for {cname, bx, rhs} <- [{:initial, i, {0, 1}}, {:unsafe, u, {1, 1}}, {:flow, d, {0, 1}}],
          {:ok, bxq} = box(bx),
          cell <- uniform(bxq, splits) do
        polys = Enum.map(parts, & &1[cname])
        dims = polys |> Enum.map(&degrees/1) |> Enum.reduce(fn a, b -> Enum.zip_with(a, b, &max/2) end)
        # Bernstein coefficients of every monomial's contribution on this cell, as rationals
        cols = Enum.map(polys, fn p -> {cs, den} = bernstein(p, cell, dims); Enum.map(Tuple.to_list(cs), &LP.q(&1, den)) end)
        cols |> Enum.zip_with(& &1) |> Enum.map(fn coeffs -> {Map.new(Enum.zip(names, coeffs)) |> Map.reject(fn {_, c} -> LP.qzero?(c) end), :ge, rhs} end)
      end
      |> List.flatten()
      |> Enum.reject(fn {coeffs, _, rhs} -> coeffs == %{} and LP.qcmp(rhs, {0, 1}) <= 0 end)

    boxes = for nm <- names, sgn <- [1, -1], do: {%{nm => {sgn, 1}}, :ge, LP.qneg(bound)}

    case generate(rows, boxes, names, Keyword.get(opts, :rounds, 60)) do
      {:ok, x, rounds, used} ->
        b = %P{n: n, terms: Map.new(Enum.zip(monos, Enum.map(names, &x[&1])))} |> norm()
        report = barrier(sys, b, Keyword.merge(opts, lambda: lambda))
        if report.verdict == :proved,
          do: {:ok, b, Map.merge(report, %{lp_rounds: rounds, lp_rows: used, lp_rows_total: length(rows)})},
          else: {:error, "the LP's candidate was not accepted by the decision (#{report.verdict})"}

      {:error, _} = e ->
        e
    end
  catch
    {:aludel, why} -> {:error, why}
  end

  defp monomials(n, deg) do
    all = Enum.reduce(1..n//1, [[]], fn _, acc -> for e <- acc, k <- 0..deg, do: e ++ [k] end)
    all |> Enum.filter(&(Enum.sum(&1) <= deg)) |> Enum.sort_by(&{Enum.sum(&1), &1}) |> Enum.map(&List.to_tuple/1)
  end

  # constraint generation over an exact LP: start with every vertex-like row
  # cheaply (the first of each cell is enough to bound), add the most violated
  defp generate(rows, boxes, names, max_rounds) do
    start = rows |> Enum.chunk_every(max(div(length(rows), 24), 1)) |> Enum.map(&hd/1)
    loop(start, rows, boxes, names, 1, max_rounds)
  end

  defp loop(active, rows, boxes, names, round, max_rounds) do
    lp = %{sense: :max, vars: names, c: %{}, c0: {0, 1}, rows: active ++ boxes, free: names}

    case LP.solve(lp) do
      {:ok, %{status: :optimal, x: x}} ->
        violated =
          rows
          |> Enum.map(fn {coeffs, :ge, rhs} = row -> {row, LP.qsub(Enum.reduce(coeffs, {0, 1}, fn {v, c}, s -> LP.qadd(s, LP.qmul(c, x[v])) end), rhs)} end)
          |> Enum.filter(fn {_, slack} -> LP.qsign(slack) < 0 end)
          |> Enum.sort_by(fn {_, s} -> LP.to_float(s) end)
          |> Enum.take(8)
          |> Enum.map(&elem(&1, 0))

        cond do
          violated == [] -> {:ok, x, round, length(active)}
          round >= max_rounds -> {:error, "no barrier of this degree within #{max_rounds} rounds of constraint generation"}
          true -> loop(active ++ violated, rows, boxes, names, round + 1, max_rounds)
        end

      {:ok, %{status: :infeasible}} ->
        {:error, "no barrier of this degree and coefficient bound exists on this subdivision (the LP is infeasible, with a Farkas certificate)"}

      {:ok, %{status: other}} ->
        {:error, "the LP is #{other}"}
    end
  end
end
