defmodule Vapor.Assay.Scaling do
  @moduledoc """
  Scaling laws with their uncertainty and a test of their predictive power
  (docs/ASSAY.md §6).

  With columns N (parameters), D (tokens) and L (final loss), the model is
  Hoffmann et al.'s L = E + A/N^α + B/D^β, fitted the way their "approach
  3" did — log L̂ = LSE(a − α log N, b − β log D, e), Huber loss (δ = 10⁻³)
  on the log residuals, many starts (a grid, the best twenty refined by
  Nelder–Mead) — since the replication by Besiroglu et al. (2024) showed
  how fragile a single fit is. With only N (or a compute column C):
  L = E + A/N^α.

  Evidence:

    * **bootstrap** over runs: intervals for α, β, E and the compute-optimal
      exponents a = β/(α+β) (N* ∝ C^a) and b = α/(α+β);
    * **leave-the-largest-out**: fit without the largest ~20 % of runs and
      predict them — a law that cannot predict its own largest runs should
      not be extrapolated further;
    * the residuals' size against the loss's own scale.
  """
  alias Vapor.Assay.Stats
  alias Vapor.Crucible.Regress
  import Stats, only: [f: 1]

  def fit(h, rows, o) do
    col = fn names -> Enum.find_index(h, &(&1 in names)) end
    ni = col.(~w(N n params parameters))
    di = col.(~w(D d tokens data))
    ci = col.(~w(C c compute flops))
    li = col.(~w(L l loss))
    x1 = ni || ci

    cond do
      li == nil or x1 == nil -> {:error, "columns N (or C) and L, and optionally D"}
      true ->
        pts =
          rows
          |> Enum.map(fn r -> {Stats.num(Enum.at(r, x1)), di && Stats.num(Enum.at(r, di)), Stats.num(Enum.at(r, li))} end)
          |> Enum.reject(fn {x, d, l} -> x == nil or l == nil or x <= 0 or l <= 0 or (di != nil and (d == nil or d <= 0)) end)

        two = di != nil and ni != nil
        if length(pts) < (if two, do: 7, else: 5) do
          {:error, "too few runs to fit #{if two, do: "5", else: "3"} parameters with a check: at least #{if two, do: 7, else: 5}"}
        else
          run(pts, two, Keyword.get(o, :seed, 1), if(ni, do: "N", else: "C"))
        end
    end
  end

  defp run(pts, two, seed, xname) do
    best = fit_points(pts, two, nil)
    params = unpack(best, two)
    pt = List.to_tuple(pts)
    n = length(pts)
    {boots, _} =
      Enum.map_reduce(1..120, Stats.rng(seed), fn _, r ->
        {idx, r} = Enum.map_reduce(1..n, r, fn _, rr -> :rand.uniform_s(n, rr) end)
        sample = Enum.map(idx, &elem(pt, &1 - 1))
        {unpack(fit_points(sample, two, best), two), r}
      end)
    ci = fn key -> vals = boots |> Enum.map(&Map.get(&1, key)) |> Enum.reject(&is_nil/1) |> Enum.sort() |> List.to_tuple(); if tuple_size(vals) > 4, do: [Stats.pct(vals, 0.025), Stats.pct(vals, 0.975)] end

    # leave the largest out
    order = Enum.sort_by(pts, fn {x, d, _} -> if two, do: x * d, else: x end)
    hold = max(1, div(n, 5)) |> min(n - (if two, do: 6, else: 4))
    {train, test} = Enum.split(order, n - hold)
    pfit = unpack(fit_points(train, two, best), two)
    preds = Enum.map(test, fn {x, d, l} -> %{x: x, d: d, loss: l, predicted: predict(pfit, x, d), error: abs(predict(pfit, x, d) - l) / l} end)
    worst = preds |> Enum.map(& &1.error) |> Enum.max()
    resid = Enum.map(pts, fn {x, d, l} -> abs(:math.log(predict(params, x, d)) - :math.log(l)) end)

    optimal = if two, do: (try do compute_optimal(params, boots) rescue _ -> nil end), else: nil

    {:ok, %{model: if(two, do: "L = E + A/N^α + B/D^β", else: "L = E + A/#{xname}^α"), params: Map.delete(params, :raw),
            ci95: %{alpha: ci.(:alpha), beta: ci.(:beta), e: ci.(:e), a_opt: ci.(:a_opt)},
            runs: n, holdout: %{held_out: hold, predictions: preds, worst_relative_error: worst}, compute_optimal: optimal,
            fit_rms_log: :math.sqrt(Stats.mean(Enum.map(resid, &(&1 * &1)))),
            says: "α = #{f(params.alpha)} #{interval(ci.(:alpha))}" <> if(two, do: ", β = #{f(params.beta)} #{interval(ci.(:beta))}, N* ∝ C^#{f(params.a_opt)} #{interval(ci.(:a_opt))}", else: "") <>
              ", E = #{f(params.e)}; fitted without the largest #{hold} run(s), the law predicts them within #{f(worst * 100)} %",
            evidence: [
              %{check: "predicts its largest runs", ok: worst < 0.02, detail: "worst relative error #{f(worst * 100)} % on the #{hold} largest run(s), fitted without them"},
              %{check: "parameter stability", ok: ci.(:alpha) != nil and Enum.at(ci.(:alpha), 1) - Enum.at(ci.(:alpha), 0) < 0.5 * max(params.alpha, 1.0e-9), detail: "α's 95 % interval #{interval(ci.(:alpha))} (width relative to α: #{f((Enum.at(ci.(:alpha) || [0, 0], 1) - Enum.at(ci.(:alpha) || [0, 0], 0)) / max(params.alpha, 1.0e-9))})"}
            ]}}
  end

  defp interval(nil), do: ""
  defp interval([a, b]), do: "[#{f(a)}, #{f(b)}]"

  # x = (e, a, α) or (e, a, α, b, β) in log space
  defp loss(x, pts, two) do
    Enum.reduce(pts, 0.0, fn {n, d, l}, s -> s + huber(log_pred(x, n, d, two) - :math.log(l), 1.0e-3) end)
  end

  defp log_pred([e, a, al], n, _d, false), do: lse([a - al * :math.log(n), e])
  defp log_pred([e, a, al, b, be], n, d, true), do: lse([a - al * :math.log(n), b - be * :math.log(d), e])

  defp lse(xs), do: (m = Enum.max(xs); m + :math.log(Enum.reduce(xs, 0.0, &(&2 + :math.exp(&1 - m)))))
  defp huber(r, d), do: (if abs(r) <= d, do: 0.5 * r * r, else: d * (abs(r) - 0.5 * d))

  defp fit_points(pts, two, warm) do
    f = fn x -> (try do loss(x, pts, two) rescue _ -> 1.0e300 end) end
    starts =
      if warm do
        [warm]
      else
        grid = for e <- [-1.0, -0.5, 0.0, 0.5, 1.0], a <- [0.0, 5.0, 10.0, 15.0, 20.0, 25.0], al <- [0.1, 0.25, 0.4, 0.6, 0.8], do: [e, a, al]
        grid = if two, do: for(g <- grid, b <- [0.0, 5.0, 10.0, 15.0, 20.0, 25.0], be <- [0.1, 0.25, 0.4, 0.6], do: g ++ [b, be]), else: grid
        grid |> Enum.map(&{f.(&1), &1}) |> Enum.sort() |> Enum.take(20) |> Enum.map(&elem(&1, 1))
      end
    starts |> Enum.map(fn s -> x = Regress.nelder_mead(f, s, if(warm, do: 400, else: 900)); {f.(x), x} end) |> Enum.min_by(&elem(&1, 0)) |> elem(1)
  end

  # a fit on data without structure can wander to absurd logs; the numbers stay finite (and the
  # holdout then says the law predicts nothing) instead of overflowing
  defp sexp(x), do: :math.exp(min(max(x, -700.0), 700.0))

  defp unpack([e, a, al] = raw, false), do: %{e: sexp(e), a: sexp(a), alpha: al, b: nil, beta: nil, a_opt: nil, raw: raw}
  defp unpack([e, a, al, b, be] = raw, true), do: %{e: sexp(e), a: sexp(a), alpha: al, b: sexp(b), beta: be, a_opt: if(al + be != 0, do: be / (al + be)), raw: raw}

  defp predict(%{raw: raw, beta: be}, x, d), do: sexp(log_pred(raw, x, d, be != nil))

  # C = 6ND: N*(C) = G (C/6)^a, D*(C) = (C/6)^b / G, G = (αA/(βB))^(1/(α+β))
  defp compute_optimal(p, _boots) do
    a = p.beta / (p.alpha + p.beta)
    b = p.alpha / (p.alpha + p.beta)
    g = :math.pow(p.alpha * p.a / (p.beta * p.b), 1 / (p.alpha + p.beta))
    budgets = [1.0e20, 1.0e21, 1.0e22, 1.0e23, 1.0e24]
    %{a: a, b: b, budgets: Enum.map(budgets, fn c -> n = g * :math.pow(c / 6, a); d = :math.pow(c / 6, b) / g; %{flops: c, params: n, tokens: d, tokens_per_param: d / n, loss: predict(p, n, d)} end)}
  end
end
