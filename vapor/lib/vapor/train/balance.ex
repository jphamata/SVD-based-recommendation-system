defmodule Vapor.Train.Balance do
  @moduledoc """
  **Quantile Balancing** of a mixture of experts (Kimi K3 report, §2.3.3
  and Appendix C), in exact rational arithmetic, beside the problem it
  claims to solve, so that the claim can be checked rather than believed.

  A router scores `m` tokens over `n` experts (`s : m×n`), each token takes
  its top `k` of `s + b`, and the bias `b` is tuned so that every expert
  receives `q = mk/n` tokens. The report derives the bias from the
  balanced assignment

      max Σ xᵢⱼ sᵢⱼ   s.t.  Σⱼ xᵢⱼ = k,  Σᵢ xᵢⱼ = q,  0 ≤ x ≤ 1        (Eq. 20)

  whose relaxation is exact (a bipartite b-matching: the constraint matrix
  is totally unimodular, every vertex is integral) and whose dual is
  minimised one coordinate block at a time; each block has a closed form,
  a quantile, so the update is

      αᵢ = the (k+1)-th largest of sᵢ + b
      b̂ⱼ = −(the (q+1)-th largest of s:,ⱼ − α),   b = b̂ − mean(b̂)       (Eq. 14)

  and repeating it on one batch is the alternating solver (Algorithm 1).

  Here: `step/4` is Eq. 14, `alternate/4` Algorithm 1, `route/3` the top-k
  of Eq. 13 (ties to the lower index), `assignment/3` the exact optimum of
  Eq. 20 by `Vapor.Logic.LP` (simplex over ℚ, with its dual certificate
  checked). What the test file measures with them is in docs/KIMI.md.
  Scores and biases are rationals `{n, d}`.
  """
  alias Vapor.Logic.LP

  @type rat :: {integer, pos_integer}

  @doc "Each token's top `k` experts of `s + b` (ties to the lower index)."
  def route(s, b, k) do
    for row <- s do
      row |> Enum.zip(b) |> Enum.with_index() |> Enum.map(fn {{x, y}, j} -> {LP.qadd(x, y), j} end)
      |> Enum.sort(fn {x, i}, {y, j} -> c = LP.qcmp(x, y); c > 0 or (c == 0 and i < j) end)
      |> Enum.take(k) |> Enum.map(&elem(&1, 1)) |> Enum.sort()
    end
  end

  @doc "Tokens per expert."
  def loads(routes, n), do: Enum.reduce(List.flatten(routes), List.duplicate(0, n), fn j, acc -> List.update_at(acc, j, &(&1 + 1)) end)

  @doc "Σ of the chosen scores (the objective of Eq. 20 for a routing)."
  def score(s, routes), do: Enum.zip(s, routes) |> Enum.reduce({0, 1}, fn {row, js}, acc -> Enum.reduce(js, acc, &LP.qadd(&2, Enum.at(row, &1))) end)

  @doc """
  One Quantile Balancing update of the bias `b` from a batch (Eq. 14),
  mean-centred. `threshold:` `:order` (the report's convention: the
  (k+1)-th and (q+1)-th largest entries themselves) or `:midpoint` (half
  way between the k-th and (k+1)-th, and the q-th and (q+1)-th — the same
  coordinate minimiser, by the report's own derivation).
  """
  def step(s, b, k, opts \\ []) do
    {m, n} = {length(s), length(b)}
    q = div(m * k, n)
    cut = threshold(Keyword.get(opts, :threshold, :order))
    alpha = for row <- s, do: cut.(Enum.zip_with(row, b, &LP.qadd/2), k)
    cols = Enum.zip(s) |> Enum.map(&Tuple.to_list/1)
    bh = for col <- cols, do: LP.qneg(cut.(Enum.zip_with(col, alpha, &LP.qsub/2), q))
    mean = LP.qdiv(Enum.reduce(bh, {0, 1}, &LP.qadd/2), {n, 1})
    Enum.map(bh, &LP.qsub(&1, mean))
  end

  defp threshold(:order), do: fn xs, k -> kth(xs, k + 1) end
  defp threshold(:midpoint), do: fn xs, k -> (s = sorted(xs); LP.qdiv(LP.qadd(Enum.at(s, k - 1), Enum.at(s, k)), {2, 1})) end

  @doc """
  Algorithm 1: `step/4` repeated on one batch from `b = 0`, at most
  `iters` times, stopping at a fixed point of the bias. Returns `%{bias,
  routes, loads, balanced, rounds, fixed}`: `rounds` is the first round
  after which the routing was balanced (nil if never), `fixed` the round at
  which the bias stopped moving (nil if it did not).
  """
  def alternate(s, k, iters, opts \\ []) do
    n = length(hd(s))
    q = div(length(s) * k, n)

    {b, first, fixed} =
      Enum.reduce_while(1..iters, {List.duplicate({0, 1}, n), nil, nil}, fn i, {b, first, _} ->
        b2 = step(s, b, k, opts)
        first = first || (Enum.all?(loads(route(s, b2, k), n), &(&1 == q)) && i) || nil
        if b2 == b, do: {:halt, {b2, first, i}}, else: {:cont, {b2, first, nil}}
      end)

    routes = route(s, b, k)
    l = loads(routes, n)
    %{bias: b, routes: routes, loads: l, balanced: Enum.all?(l, &(&1 == q)), rounds: first, fixed: fixed}
  end

  @doc """
  The ties of a routing: tokens whose k-th and (k+1)-th largest `s + b`
  are equal, so that top-k breaks the tie by index rather than by score.
  """
  def ties(s, b, k) do
    for {row, i} <- Enum.with_index(s), v = sorted(Enum.zip_with(row, b, &LP.qadd/2)), Enum.at(v, k - 1) == Enum.at(v, k), do: i
  end

  @doc """
  The exact optimum of the balanced assignment (Eq. 20, relaxed) by the
  rational simplex: `{:ok, %{objective, x, integral, certified}}`, `x` the
  assignment as `{i, j}` pairs with `xᵢⱼ = 1` when integral.
  """
  def assignment(s, k) do
    {m, n} = {length(s), length(hd(s))}
    q = div(m * k, n)
    v = fn i, j -> "x#{i}_#{j}" end
    vars = for i <- 0..(m - 1), j <- 0..(n - 1), do: v.(i, j)
    c = for i <- 0..(m - 1), j <- 0..(n - 1), into: %{}, do: {v.(i, j), s |> Enum.at(i) |> Enum.at(j)}
    rows =
      (for i <- 0..(m - 1), do: {Map.new(0..(n - 1), &{v.(i, &1), {1, 1}}), :eq, {k, 1}}) ++
        (for j <- 0..(n - 1), do: {Map.new(0..(m - 1), &{v.(&1, j), {1, 1}}), :eq, {q, 1}}) ++
        (for x <- vars, do: {%{x => {1, 1}}, :le, {1, 1}})

    with {:ok, %{status: :optimal} = r} <- LP.solve(%{sense: :max, vars: vars, c: c, c0: {0, 1}, rows: rows, free: []}) do
      integral = Enum.all?(r.x, fn {_, x} -> x in [{0, 1}, {1, 1}] end)
      pairs = for i <- 0..(m - 1), j <- 0..(n - 1), r.x[v.(i, j)] == {1, 1}, do: {i, j}
      {:ok, %{objective: r.objective, x: pairs, integral: integral, certified: r.check.accepted}}
    end
  end

  # the k-th largest (1-based) of a list of rationals
  defp kth(xs, k), do: xs |> sorted() |> Enum.at(k - 1)
  defp sorted(xs), do: Enum.sort(xs, &(LP.qcmp(&1, &2) >= 0))
end
