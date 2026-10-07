defmodule Vapor.InfoGeom do
  @moduledoc """
  Information geometry where there is a statistical manifold to measure
  (docs/GEOMETRIA.md) — model outputs, estimators, ensembles — and not where
  there is none (a compiler's cost surface over registers is a discrete
  lattice; DIRETRIZ §19 says why geodesics do not belong there).

  On the probability simplex the Fisher information metric is, under the map
  `p ↦ 2√p`, the round metric of a sphere. Everything below follows exactly
  from that:

    * `fisher_rao/2` — the geodesic distance `2·arccos Σ√(pᵢqᵢ)`: a true
      metric (symmetric, triangle inequality — tested), unlike KL; the
      distance between two models' predictive distributions that can be
      averaged, clustered and bounded;
    * `geodesic/3` and `frechet_mean/2` — interpolation along the great
      circle and the Karcher mean: the *geometric* ensemble of several
      models' predictions, which stays a distribution and is invariant to
      relabelling;
    * `gaussian_fisher_rao/4` — for univariate normals the Fisher metric is
      hyperbolic (the Poincaré half-plane in (μ/√2, σ)), so their distance
      has a closed form;
    * `natural_logistic/3` — logistic regression by natural-gradient steps
      (the Fisher matrix `Xᵀ diag(p(1−p)) X`): its iterates do not depend on
      how the features are scaled, while plain gradient descent's do —
      *invariance to reparametrisation*, measured (`plain_logistic/3` is the
      control).
  """

  @eps 1.0e-300

  # ------------------------------------------------------------- simplex

  @doc "Normalise non-negative weights into a distribution."
  def normalize(xs) do
    s = Enum.sum(xs)
    if s <= 0, do: raise(ArgumentError, "a distribution needs positive mass"), else: Enum.map(xs, &(&1 / s))
  end

  @doc "The Bhattacharyya coefficient Σ√(pᵢqᵢ) ∈ [0, 1]."
  def bc(p, q), do: Enum.zip_reduce(p, q, 0.0, fn a, b, acc -> acc + :math.sqrt(max(a, 0.0) * max(b, 0.0)) end)

  @doc "The Fisher–Rao geodesic distance on the simplex: 2·arccos(BC), in [0, π]."
  def fisher_rao(p, q), do: 2 * :math.acos(min(max(bc(p, q), 0.0), 1.0))

  @doc "The Hellinger distance √(1 − BC) — the chord to Fisher–Rao's arc."
  def hellinger(p, q), do: :math.sqrt(max(1 - bc(p, q), 0.0))

  @doc "Kullback–Leibler divergence KL(p‖q) in nats (asymmetric; ∞ where q = 0 < p)."
  def kl(p, q) do
    Enum.zip_reduce(p, q, 0.0, fn a, b, acc ->
      cond do
        a <= 0 -> acc
        b <= 0 -> :infinity
        acc == :infinity -> acc
        true -> acc + a * :math.log(a / b)
      end
    end)
  end

  @doc "Jensen–Shannon divergence (symmetric; its square root is a metric)."
  def js(p, q) do
    m = Enum.zip_with(p, q, &((&1 + &2) / 2))
    (kl(p, m) + kl(q, m)) / 2
  end

  @doc "The point at fraction `t` along the Fisher–Rao geodesic from p to q (slerp of √p, √q)."
  def geodesic(p, q, t) do
    a = Enum.map(p, &:math.sqrt(max(&1, 0.0)))
    b = Enum.map(q, &:math.sqrt(max(&1, 0.0)))
    theta = :math.acos(min(max(bc(p, q), -1.0), 1.0))

    u =
      if theta < 1.0e-12,
        do: Enum.zip_with(a, b, &((1 - t) * &1 + t * &2)),
        else: Enum.zip_with(a, b, &((:math.sin((1 - t) * theta) * &1 + :math.sin(t * theta) * &2) / :math.sin(theta)))

    normalize(Enum.map(u, &(&1 * &1)))
  end

  @doc """
  The weighted Karcher (Fréchet) mean of distributions under the Fisher
  metric: the point minimising Σ wᵢ·d(m, pᵢ)², found on the sphere by
  Riemannian gradient steps (log map, average, exp map). Returns `{mean,
  iterations}`.
  """
  def frechet_mean(ps, weights \\ nil, opts \\ []) do
    n = length(ps)
    w = normalize(weights || List.duplicate(1.0, n))
    xs = Enum.map(ps, fn p -> Enum.map(p, &:math.sqrt(max(&1, 0.0))) end)
    start = xs |> Enum.zip(w) |> Enum.map(fn {x, wi} -> Enum.map(x, &(&1 * wi)) end) |> Enum.zip_with(&Enum.sum/1) |> unit()
    iterate(start, xs, w, Keyword.get(opts, :iters, 100), Keyword.get(opts, :tol, 1.0e-12), 0)
  end

  defp iterate(m, xs, w, max, tol, k) do
    # the weighted average of the log maps at m
    v =
      Enum.zip(xs, w)
      |> Enum.map(fn {x, wi} ->
        c = min(max(dot(m, x), -1.0), 1.0)
        th = :math.acos(c)
        if th < 1.0e-15, do: Enum.map(x, fn _ -> 0.0 end), else: Enum.zip_with(x, m, fn xi, mi -> wi * th / :math.sin(th) * (xi - c * mi) end)
      end)
      |> Enum.zip_with(&Enum.sum/1)

    nv = :math.sqrt(dot(v, v))
    m2 = if nv < 1.0e-300, do: m, else: Enum.zip_with(m, v, fn mi, vi -> :math.cos(nv) * mi + :math.sin(nv) * vi / nv end) |> unit()

    if nv < tol or k + 1 >= max, do: {normalize(Enum.map(m2, &(&1 * &1))), k + 1}, else: iterate(m2, xs, w, max, tol, k + 1)
  end

  defp dot(a, b), do: Enum.zip_reduce(a, b, 0.0, &(&3 + &1 * &2))
  defp unit(v), do: (n = :math.sqrt(dot(v, v)); Enum.map(v, &(&1 / max(n, @eps))))

  # ------------------------------------------------------------- normals

  @doc """
  Fisher–Rao distance between N(μ₁, σ₁²) and N(μ₂, σ₂²): the hyperbolic
  distance √2·arccosh(1 + ((μ₁−μ₂)²/2 + (σ₁−σ₂)²) / (2σ₁σ₂)).
  """
  def gaussian_fisher_rao(m1, s1, m2, s2) when s1 > 0 and s2 > 0 do
    arg = 1 + ((m1 - m2) * (m1 - m2) / 2 + (s1 - s2) * (s1 - s2)) / (2 * s1 * s2)
    :math.sqrt(2) * :math.acosh(max(arg, 1.0))
  end

  # ----------------------------------------------------- natural gradient

  @doc """
  Logistic regression `P(y=1|x) = σ(θ·x)` (add a constant column for an
  intercept) by natural-gradient steps θ ← θ + F⁻¹∇ℓ, F = Xᵀdiag(p(1−p))X —
  for the canonical link this is Fisher scoring. Options: `steps:`
  (default 50), `tol:` on the gradient's norm. Returns `%{theta, steps,
  loss, grad_norm}`.
  """
  def natural_logistic(x, y, opts \\ []) do
    d = length(hd(x))
    run(x, y, List.duplicate(0.0, d), Keyword.get(opts, :steps, 50), Keyword.get(opts, :tol, 1.0e-9), :natural, Keyword.get(opts, :rate, 1.0), 0)
  end

  @doc "The control: plain gradient ascent at a fixed rate (default 0.5) — the same model, parametrisation-dependent."
  def plain_logistic(x, y, opts \\ []) do
    d = length(hd(x))
    run(x, y, List.duplicate(0.0, d), Keyword.get(opts, :steps, 50), Keyword.get(opts, :tol, 1.0e-9), :plain, Keyword.get(opts, :rate, 0.5), 0)
  end

  defp run(x, y, th, max, tol, how, rate, k) do
    p = Enum.map(x, &sigmoid(dot(&1, th)))
    n = length(y)
    g = x |> Enum.zip(Enum.zip(y, p)) |> Enum.map(fn {xi, {yi, pi}} -> Enum.map(xi, &(&1 * (yi - pi) / n)) end) |> Enum.zip_with(&Enum.sum/1)
    gn = :math.sqrt(dot(g, g))

    if gn < tol or k >= max do
      %{theta: th, steps: k, loss: loss(p, y), grad_norm: gn}
    else
      step =
        case how do
          :plain -> Enum.map(g, &(&1 * rate))
          :natural ->
            f = fisher(x, p, n)
            {:ok, l} = Vapor.Linalg.cholesky(add_ridge(f, 1.0e-12))
            Vapor.Linalg.chol_solve(l, List.to_tuple(g)) |> Tuple.to_list() |> Enum.map(&(&1 * rate))
        end

      run(x, y, Enum.zip_with(th, step, &(&1 + &2)), max, tol, how, rate, k + 1)
    end
  end

  defp fisher(x, p, n) do
    d = length(hd(x))
    rows = for i <- 0..(d - 1), do: (for j <- 0..(d - 1), do: Enum.zip_reduce(x, p, 0.0, fn xi, pi, acc -> acc + Enum.at(xi, i) * Enum.at(xi, j) * pi * (1 - pi) / n end))
    Vapor.Linalg.from_rows(rows)
  end

  defp add_ridge(a, r) do
    a |> Vapor.Linalg.to_rows() |> Enum.with_index() |> Enum.map(fn {row, i} -> List.update_at(row, i, &(&1 + r)) end) |> Vapor.Linalg.from_rows()
  end

  defp sigmoid(z) when z >= 0, do: 1 / (1 + :math.exp(-z))
  defp sigmoid(z), do: (e = :math.exp(z); e / (1 + e))

  defp loss(p, y), do: -Enum.zip_reduce(p, y, 0.0, fn pi, yi, acc -> acc + yi * :math.log(max(pi, @eps)) + (1 - yi) * :math.log(max(1 - pi, @eps)) end) / length(y)
end
