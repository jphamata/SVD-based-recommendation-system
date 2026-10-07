defmodule Vapor.Crucible.Evolution do
  @moduledoc """
  The fate of a mutant in a population the user describes (docs/CRUCIBLE.md
  §6): haploid Wright–Fisher with population size `N`, selection `s`,
  starting count `i0` — answered three independent ways that must agree:

    * **exactly**: the absorption probabilities of the Markov chain on
      0…N, by solving (I − Q)u = r (Gaussian elimination, N ≤ 600);
    * **by simulation**: `replicates` runs with a seeded generator — the
      exact value must lie within 3 standard errors;
    * **by Kimura's diffusion**: u(p) = (1 − e^{−2Nsp})/(1 − e^{−2Ns}) —
      an approximation whose error is reported, not hidden.

  The control is the neutral case (s = 0), whose answer is p = i0/N.
  Simulation also gives the mean time to fixation (conditional on it).
  """
  alias Vapor.Crucible.Sheet

  def run(text) do
    s = Sheet.parse(text)
    n = s |> Sheet.const("N", 100.0) |> round()
    sel = Sheet.const(s, "s", 0.01)
    i0 = s |> Sheet.const("i0", 1.0) |> round()
    reps = s |> Sheet.const("replicates", 2000.0) |> round() |> min(20_000)
    seed = s |> Sheet.const("seed", 1.0) |> round()

    cond do
      s.errors != [] -> {:error, Enum.join(s.errors, "; ")}
      n < 2 or n > 5000 -> {:error, "N between 2 and 5000"}
      i0 < 1 or i0 >= n -> {:error, "i0 between 1 and N − 1"}
      sel <= -1 -> {:error, "s > −1"}
      true ->
        p0 = i0 / n
        exact = if n <= 600, do: exact(n, sel, i0)
        {sim, times} = simulate(n, sel, i0, reps, seed)
        neutral = elem(simulate(n, 0.0, i0, reps, seed + 1), 0)
        kim = kimura(n, sel, p0)
        se = :math.sqrt(max(sim * (1 - sim), 1.0e-12) / reps)
        se0 = :math.sqrt(p0 * (1 - p0) / reps)
        evidence =
          [exact && %{check: "simulation vs exact chain", ok: abs(sim - exact) <= 3 * se, detail: "#{f(sim)} ± #{f(se)} against #{f(exact)} (|Δ| = #{f(abs(sim - exact) / se)} SE)"},
           exact && %{check: "diffusion approximation", ok: true, detail: "Kimura gives #{f(kim)}: #{f(abs(kim - exact) / exact * 100)} % from the exact chain"},
           %{check: "neutral control", ok: abs(neutral - p0) <= 3 * se0, detail: "with s = 0 the simulated probability is #{f(neutral)}; theory: i0/N = #{f(p0)}"}]
          |> Enum.filter(& &1)

        {:ok, %{kind: "evolution", n: n, s: sel, i0: i0, replicates: reps, exact: exact, simulated: sim, standard_error: se, kimura: kim, neutral: neutral,
                fixation_time: if(times == [], do: nil, else: Enum.sum(times) / length(times)), evidence: evidence,
                says: "fixation probability #{f(exact || sim)}" <> if(exact, do: " (exact chain)", else: " (simulated)") <> "; neutral would be #{f(p0)} — selection multiplies it by #{f((exact || sim) / p0)}"}}
    end
  end

  defp f(nil), do: "—"
  defp f(x), do: :erlang.float_to_binary(x * 1.0, [{:decimals, 5}, :compact])

  defp next_p(i, n, s), do: i * (1 + s) / (i * (1 + s) + n - i)

  @doc "Kimura's diffusion approximation for a haploid population."
  def kimura(_n, s, p) when s == 0, do: p
  def kimura(n, s, p), do: (1 - :math.exp(-2 * n * s * p)) / (1 - :math.exp(-2 * n * s))

  defp pmf_row(n, p) do
    # binomial pmf by logs, stable for large n
    lp = if p > 0, do: :math.log(p), else: nil
    lq = if p < 1, do: :math.log(1 - p), else: nil
    lf = Enum.scan(1..n, 0.0, fn k, acc -> acc + :math.log(k) end) |> then(&List.to_tuple([0.0 | &1]))
    for k <- 0..n do
      cond do
        p == 0 -> if k == 0, do: 1.0, else: 0.0
        p == 1 -> if k == n, do: 1.0, else: 0.0
        true -> :math.exp(elem(lf, n) - elem(lf, k) - elem(lf, n - k) + k * lp + (n - k) * lq)
      end
    end
  end

  @doc "Exact fixation probability from i0 copies: solve the absorbing chain."
  def exact(n, s, i0) do
    # unknowns u_1 … u_{N−1}: u_i − Σ_j P_ij u_j = P_iN
    rows =
      for i <- 1..(n - 1) do
        row = pmf_row(n, next_p(i, n, s))
        coeffs = for j <- 1..(n - 1), do: (if i == j, do: 1.0, else: 0.0) - Enum.at(row, j)
        List.to_tuple(coeffs ++ [Enum.at(row, n)])
      end
    sol = gauss(List.to_tuple(rows), n - 1)
    elem(sol, i0 - 1)
  end

  defp gauss(a, m) do
    a =
      Enum.reduce(0..(m - 1), a, fn c, a ->
        p = Enum.max_by(c..(m - 1), fn r -> abs(elem(elem(a, r), c)) end)
        a = if p != c, do: a |> put_elem(c, elem(a, p)) |> put_elem(p, elem(a, c)), else: a
        piv = elem(a, c)
        pv = elem(piv, c)
        Enum.reduce((c + 1)..(m - 1)//1, a, fn r, a ->
          row = elem(a, r)
          fac = elem(row, c) / pv
          if fac == 0.0, do: a, else: put_elem(a, r, List.to_tuple(Enum.zip_with(Tuple.to_list(row), Tuple.to_list(piv), &(&1 - fac * &2))))
        end)
      end)
    Enum.reduce((m - 1)..0//-1, %{}, fn r, x ->
      row = elem(a, r)
      s = Enum.reduce((r + 1)..(m - 1)//1, 0.0, fn j, s -> s + elem(row, j) * x[j] end)
      Map.put(x, r, (elem(row, m) - s) / elem(row, r))
    end)
    |> then(fn x -> List.to_tuple(for i <- 0..(m - 1), do: x[i]) end)
  end

  defp simulate(n, s, i0, reps, seed) do
    rng = :rand.seed_s(:exsss, {seed, 2718, 31})
    {fixed, times, _} =
      Enum.reduce(1..reps, {0, [], rng}, fn _, {fx, ts, r} ->
        {outcome, gens, r} = run_one(n, s, i0, r, 0)
        if outcome == :fixed, do: {fx + 1, [gens | ts], r}, else: {fx, ts, r}
      end)
    {fixed / reps, times}
  end

  defp run_one(n, _s, i, r, g) when i >= n, do: {:fixed, g, r}
  defp run_one(_n, _s, 0, r, g), do: {:lost, g, r}
  defp run_one(_n, _s, _i, r, g) when g > 200_000, do: {:lost, g, r}

  defp run_one(n, s, i, r, g) do
    {k, r} = binomial(n, next_p(i, n, s), r)
    run_one(n, s, k, r, g + 1)
  end

  # inverse transform for small n·p, normal approximation with continuity only when both tails are large
  defp binomial(n, p, r) do
    {u, r} = :rand.uniform_s(r)
    if n * p < 30 or n * (1 - p) < 30 do
      {inv(n, p, u), r}
    else
      {z, r} = :rand.normal_s(r)
      {(n * p + z * :math.sqrt(n * p * (1 - p))) |> round() |> max(0) |> min(n), r}
    end
  end

  defp inv(n, p, u) do
    q = 1 - p
    if p > 0.5 do
      n - inv(n, q, u)
    else
      pk = :math.pow(q, n)
      walk(0, n, p / q, pk, pk, u)
    end
  end

  defp walk(k, n, ratio, pk, cdf, u) do
    if u <= cdf or k >= n do
      k
    else
      pk2 = pk * ratio * (n - k) / (k + 1)
      walk(k + 1, n, ratio, pk2, cdf + pk2, u)
    end
  end
end
