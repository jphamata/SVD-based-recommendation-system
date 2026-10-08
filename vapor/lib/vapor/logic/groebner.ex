defmodule Vapor.Logic.Groebner do
  @moduledoc """
  Polynomial algebra over the rationals, exactly (docs/LOGIC.md §5):
  Buchberger's algorithm with his two criteria, reduced Gröbner bases in
  lex or graded reverse-lex order, ideal membership with the cofactors
  that prove it, and **implication between polynomial statements** by
  the Rabinowitsch trick — a hypothesis set H implies g = 0 (where the
  non-degeneracy conditions d ≠ 0 hold) exactly when
  1 ∈ ⟨H, 1 − t·g·d⟩, i.e. when the reduced basis is {1}.

  This is the prover of 0.11's geometry (`Vapor.Prove`) opened up: any
  hypotheses and any conclusion typed as polynomials, not a fixed list of
  constructions —

      vars x y a b
      hyp x^2 + y^2 - 1            # P = (x, y) on the unit circle
      hyp a + 1                     # A = (−1, 0)
      hyp b - 1                     # B = (1, 0)
      claim (x - a)*(x - b) + y*y   # AP ⊥ BP  (Thales)
      unless y                      # non-degenerate: P ≠ A, B

  and the answer is **proved** (with the basis {1} as the certificate,
  re-checkable by reduction) or **not implied**, with a witness point
  where the hypotheses hold and the claim fails when the solver finds one.
  """
  alias Vapor.Expr

  # a polynomial: %{exponents tuple => {num, den}} with no zero coefficients

  # ============================================================== rationals

  defp qn({a, b}) when b < 0, do: qn({-a, -b})
  defp qn({0, _}), do: {0, 1}
  defp qn({a, b}), do: (g = Integer.gcd(a, b); {div(a, g), div(b, g)})
  defp qadd({a, b}, {c, d}), do: qn({a * d + c * b, b * d})
  defp qmul({a, b}, {c, d}), do: qn({a * c, b * d})
  defp qdiv({a, b}, {c, d}), do: qn({a * d, b * c})
  defp qneg({a, b}), do: {-a, b}

  # ============================================================ polynomials

  @doc "A polynomial from an expression tree over `vars` (integer and decimal constants, + − × ^ with natural exponents, ÷ by constants)."
  def from_tree(tree, vars) do
    n = length(vars)
    idx = vars |> Enum.with_index() |> Map.new()
    pt(tree, idx, n)
  catch
    {:poly, w} -> {:error, w}
  else
    p -> {:ok, p}
  end

  defp zero_exp(n), do: Tuple.duplicate(0, n)
  defp const(c, n), do: if(c == {0, 1}, do: %{}, else: %{zero_exp(n) => c})

  defp pt({:n, x}, _, n), do: const(rational(x), n)
  defp pt({:v, v}, idx, n), do: (case Map.fetch(idx, v) do {:ok, i} -> %{put_elem(zero_exp(n), i, 1) => {1, 1}}; :error -> throw({:poly, "#{v} is not a declared variable"}) end)
  defp pt({:neg, a}, idx, n), do: pneg(pt(a, idx, n))
  defp pt({:+, a, b}, idx, n), do: padd(pt(a, idx, n), pt(b, idx, n))
  defp pt({:-, a, b}, idx, n), do: padd(pt(a, idx, n), pneg(pt(b, idx, n)))
  defp pt({:*, a, b}, idx, n), do: pmul(pt(a, idx, n), pt(b, idx, n))
  defp pt({:/, a, {:n, c}}, idx, n), do: pscale(pt(a, idx, n), qdiv({1, 1}, rational(c)))
  defp pt({:^, a, {:n, k}}, idx, n) when k == trunc(k) and k >= 0 and k <= 64, do: Enum.reduce(1..max(trunc(k), 1)//1, (if k == 0, do: const({1, 1}, n), else: nil), fn _, acc -> if acc, do: pmul(acc, pt(a, idx, n)), else: pt(a, idx, n) end)
  defp pt(t, _, _), do: throw({:poly, "not a polynomial: #{Expr.to_text(t)}"})

  # a float as the rational its shortest decimal shows (0.1 → 1/10)
  defp rational(x) when x == trunc(x), do: {trunc(x), 1}
  defp rational(x) do
    s = :erlang.float_to_binary(x, [:short])
    case String.split(s, ["e", "E"]) do
      [m] -> dec(m)
      [m, e] -> (({a, b} = dec(m)); ex = String.to_integer(e); if ex >= 0, do: qn({a * Integer.pow(10, ex), b}), else: qn({a, b * Integer.pow(10, -ex)}))
    end
  end
  defp dec(m) do
    case String.split(m, ".") do
      [i] -> {String.to_integer(i), 1}
      [i, f] -> qn({String.to_integer(i <> f), Integer.pow(10, String.length(f))})
    end
  end

  defp padd(a, b), do: Map.merge(a, b, fn _, x, y -> qadd(x, y) end) |> Map.reject(fn {_, c} -> c == {0, 1} end)
  defp pneg(a), do: Map.new(a, fn {e, c} -> {e, qneg(c)} end)
  defp pscale(a, c), do: if(c == {0, 1}, do: %{}, else: Map.new(a, fn {e, x} -> {e, qmul(x, c)} end))
  defp pmul(a, b) do
    for {e1, c1} <- a, {e2, c2} <- b, reduce: %{} do
      acc -> (e = addexp(e1, e2); Map.update(acc, e, qmul(c1, c2), &qadd(&1, qmul(c1, c2))))
    end
    |> Map.reject(fn {_, c} -> c == {0, 1} end)
  end
  defp addexp(a, b), do: Enum.zip_with(Tuple.to_list(a), Tuple.to_list(b), &(&1 + &2)) |> List.to_tuple()
  defp subexp(a, b), do: Enum.zip_with(Tuple.to_list(a), Tuple.to_list(b), &(&1 - &2)) |> List.to_tuple()
  defp divides?(a, b), do: Enum.zip(Tuple.to_list(a), Tuple.to_list(b)) |> Enum.all?(fn {x, y} -> x <= y end)
  defp lcmexp(a, b), do: Enum.zip_with(Tuple.to_list(a), Tuple.to_list(b), &max/2) |> List.to_tuple()

  # monomial orders: a key that sorts larger monomials last
  defp key(e, :lex), do: Tuple.to_list(e)
  defp key(e, :grevlex), do: [Enum.sum(Tuple.to_list(e)) | e |> Tuple.to_list() |> Enum.reverse() |> Enum.map(&(-&1))]

  defp lead(p, ord), do: Enum.max_by(p, fn {e, _} -> key(e, ord) end)

  defp monic(p, ord), do: (case Map.to_list(p) do [] -> p; _ -> (({_, c} = lead(p, ord)); pscale(p, qdiv({1, 1}, c))) end)

  @doc "Divide `f` by the list `gs`: `{quotients, remainder}` with f = Σ qᵢ gᵢ + r, no term of r divisible by a leading monomial."
  def divide(f, gs, ord) do
    leads = Enum.map(gs, &lead(&1, ord))
    divide_loop(f, gs, leads, ord, List.duplicate(%{}, length(gs)), %{})
  end

  defp divide_loop(f, gs, leads, ord, qs, r) when map_size(f) == 0, do: (_ = {gs, leads, ord}; {qs, r})
  defp divide_loop(f, gs, leads, ord, qs, r) do
    {e, c} = lead(f, ord)
    case Enum.find_index(leads, fn {le, _} -> divides?(le, e) end) do
      nil -> divide_loop(Map.delete(f, e), gs, leads, ord, qs, Map.put(r, e, c))
      i ->
        {le, lc} = Enum.at(leads, i)
        t = %{subexp(e, le) => qdiv(c, lc)}
        f = padd(f, pneg(pmul(t, Enum.at(gs, i))))
        divide_loop(f, gs, leads, ord, List.update_at(qs, i, &padd(&1, t)), r)
    end
  end

  defp spoly(f, g, ord) do
    {ef, cf} = lead(f, ord)
    {eg, cg} = lead(g, ord)
    l = lcmexp(ef, eg)
    padd(pmul(%{subexp(l, ef) => qdiv({1, 1}, cf)}, f), pneg(pmul(%{subexp(l, eg) => qdiv({1, 1}, cg)}, g)))
  end

  @doc "A reduced Gröbner basis (Buchberger with the coprime and chain criteria). `{basis, stats}`."
  def basis(polys, ord \\ :grevlex, max_pairs \\ 20_000) do
    gs = polys |> Enum.reject(&(map_size(&1) == 0)) |> Enum.map(&monic(&1, ord))
    pairs = for i <- 0..(length(gs) - 1)//1, j <- (i + 1)..(length(gs) - 1)//1, do: {i, j}
    {g, st} = buch(gs, pairs, ord, %{pairs: 0, reductions_to_zero: 0, skipped: 0}, max_pairs)
    {reduce_basis(g, ord), st}
  end

  defp buch(g, [], _ord, st, _max), do: {g, st}
  defp buch(_g, _, _ord, %{pairs: p}, max) when p > max, do: throw({:poly, "more than #{max} S-pairs: the computation is too large here"})
  defp buch(g, [{i, j} | rest], ord, st, max) do
    gi = Enum.at(g, i)
    gj = Enum.at(g, j)
    {ei, _} = lead(gi, ord)
    {ej, _} = lead(gj, ord)
    l = lcmexp(ei, ej)
    cond do
      # Buchberger's first criterion: coprime leading monomials
      l == addexp(ei, ej) -> buch(g, rest, ord, %{st | skipped: st.skipped + 1}, max)
      # the chain criterion: some g_k's leading monomial divides lcm and both pairs (i,k),(j,k) are already gone
      Enum.any?(Enum.with_index(g), fn {gk, k} -> k != i and k != j and divides?(elem(lead(gk, ord), 0), l) and not ({min(i, k), max(i, k)} in rest) and not ({min(j, k), max(j, k)} in rest) end) ->
        buch(g, rest, ord, %{st | skipped: st.skipped + 1}, max)
      true ->
        {_, r} = divide(spoly(gi, gj, ord), g, ord)
        st = %{st | pairs: st.pairs + 1}
        if map_size(r) == 0 do
          buch(g, rest, ord, %{st | reductions_to_zero: st.reductions_to_zero + 1}, max)
        else
          n = length(g)
          buch(g ++ [monic(r, ord)], rest ++ for(k <- 0..(n - 1), do: {k, n}), ord, st, max)
        end
    end
  end

  defp reduce_basis(g, ord) do
    # drop elements whose leading monomial another's divides, then reduce each by the rest
    g = Enum.uniq(g)
    minimal = Enum.reject(Enum.with_index(g), fn {p, i} ->
      {e, _} = lead(p, ord)
      Enum.any?(Enum.with_index(g), fn {q, j} -> j != i and (({f, _} = lead(q, ord)); divides?(f, e) and (f != e or j < i)) end)
    end) |> Enum.map(&elem(&1, 0))
    Enum.map(minimal, fn p -> (({_, r} = divide(p, List.delete(minimal, p), ord)); r = if map_size(r) == 0, do: p, else: r; monic(r, ord)) end)
    |> Enum.sort_by(fn p -> key(elem(lead(p, ord), 0), ord) end)
  end

  @doc "Show a polynomial over `vars`."
  def show(p, vars, ord \\ :grevlex) do
    case p |> Enum.sort_by(fn {e, _} -> key(e, ord) end, :desc) do
      [] -> "0"
      terms ->
        terms |> Enum.with_index() |> Enum.map_join("", fn {{e, {a, b}}, i} ->
          mono = Enum.zip(vars, Tuple.to_list(e)) |> Enum.reject(fn {_, k} -> k == 0 end) |> Enum.map_join("·", fn {v, 1} -> v; {v, k} -> "#{v}^#{k}" end)
          coef = if b == 1, do: "#{abs(a)}", else: "#{abs(a)}/#{b}"
          body = cond do mono == "" -> coef; abs(a) == 1 and b == 1 -> mono; true -> coef <> "·" <> mono end
          cond do i == 0 and a < 0 -> "−" <> body; i == 0 -> body; a < 0 -> " − " <> body; true -> " + " <> body end
        end)
    end
  end

  # ================================================================ problems

  @doc """
  Solve a problem written as text (see the moduledoc). Kinds, by the
  lines present: `claim` → implication; `member` → ideal membership with
  cofactors; otherwise the reduced basis of the `hyp`s (with `order lex`
  for elimination). `{:ok, result}` or `{:error, why}`.
  """
  def run(text) do
    with {:ok, p} <- parse(text) do
      cond do
        p.claim -> implies(p)
        p.member -> membership(p)
        true ->
          {g, st} = basis(p.hyps, p.order)
          {:ok, %{kind: "basis", basis: Enum.map(g, &show(&1, p.vars, p.order)), order: p.order, stats: st, consistent: g != [const({1, 1}, length(p.vars))] and g != [%{zero_exp(length(p.vars)) => {1, 1}}]}}
      end
    end
  catch
    {:poly, w} -> {:error, w}
  end

  defp parse(text) do
    lines = text |> String.split("\n") |> Enum.map(&(&1 |> String.split("#", parts: 2) |> hd() |> String.trim())) |> Enum.reject(&(&1 == ""))
    vars = Enum.find_value(lines, fn l -> if l =~ ~r/^vars\b/, do: l |> String.split() |> tl() end)
    order = if Enum.any?(lines, &(&1 =~ ~r/^order\s+lex/)), do: :lex, else: :grevlex
    exprs = for l <- lines, m = Regex.run(~r/^(hyp|hipótese|claim|tese|unless|member)\s+(.+)$/u, l), do: {Enum.at(m, 1), Enum.at(m, 2)}
    trees = Enum.map(exprs, fn {k, s} -> {k, Expr.parse(s)} end)
    case Enum.find(trees, fn {_, r} -> match?({:error, _}, r) end) do
      {_, {:error, w}} -> {:error, w}
      nil ->
        trees = Enum.map(trees, fn {k, {:ok, t}} -> {k, t} end)
        vars = vars || (trees |> Enum.flat_map(fn {_, t} -> Expr.vars(t) end) |> Enum.uniq() |> Enum.sort())
        conv = fn t -> case from_tree(t, vars) do {:ok, p} -> p; {:error, w} -> throw({:poly, w}) end end
        {:ok, %{vars: vars, order: order,
                hyps: for({k, t} <- trees, k in ["hyp", "hipótese"], do: conv.(t)),
                claim: Enum.find_value(trees, fn {k, t} -> if k in ["claim", "tese"], do: conv.(t) end),
                member: Enum.find_value(trees, fn {k, t} -> if k == "member", do: conv.(t) end),
                unless: for({"unless", t} <- trees, do: conv.(t))}}
    end
  end

  # H ⇒ g = 0 wherever every d ≠ 0: 1 ∈ ⟨H, 1 − t·g·Πd⟩ over vars + t
  defp implies(p) do
    n = length(p.vars)
    lift = fn poly -> Map.new(poly, fn {e, c} -> {:erlang.append_element(e, 0), c} end) end
    t = %{:erlang.append_element(zero_exp(n), 1) => {1, 1}}
    gd = Enum.reduce(p.unless, lift.(p.claim), fn d, acc -> pmul(acc, lift.(d)) end)
    rab = padd(const({1, 1}, n + 1), pneg(pmul(t, gd)))
    {g, st} = basis(Enum.map(p.hyps, lift) ++ [rab], :grevlex)
    one = [const({1, 1}, n + 1)]
    proved = g == one
    # the plain (non-radical) membership too: g itself reduces to 0 modulo H?
    {gh, _} = basis(p.hyps, :grevlex)
    {_, rem} = divide(p.claim, gh, :grevlex)
    {:ok, %{kind: "implication", verdict: if(proved, do: "proved", else: "not implied"), stats: st,
            certificate: if(proved, do: "the reduced Gröbner basis of ⟨H, 1 − t·g#{if p.unless != [], do: "·d"}⟩ is {1}", else: nil),
            in_ideal: map_size(rem) == 0, remainder: show(rem, p.vars),
            basis_of_hypotheses: Enum.map(gh, &show(&1, p.vars)), conditions: Enum.map(p.unless, &(show(&1, p.vars) <> " ≠ 0"))}}
  end

  defp membership(p) do
    {qs, r} = divide(p.member, p.hyps, :grevlex)
    {g, _} = basis(p.hyps, :grevlex)
    {_, rg} = divide(p.member, g, :grevlex)
    {:ok, %{kind: "membership", member: map_size(rg) == 0, remainder: show(rg, p.vars),
            division_by_hypotheses: %{quotients: Enum.map(qs, &show(&1, p.vars)), remainder: show(r, p.vars)}}}
  end
end
