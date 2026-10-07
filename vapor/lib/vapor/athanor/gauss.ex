defmodule Vapor.Athanor.Gauss do
  @moduledoc """
  Bayesian optimisation for the Athanor (docs/ATHANOR.md §3): a Gaussian
  process with a Matérn-5/2 kernel on the unit cube, hyperparameters
  (length scale, noise) chosen by the log marginal likelihood over a grid,
  and **expected improvement** maximised over random and local points.
  A batch is built by the *kriging believer*: each chosen point is added
  with its predicted mean before choosing the next.

  This is the strategy for objectives that cost minutes or are measured
  by a person — tens of evaluations, not thousands.
  """

  @doc "Suggest `n` candidates from the evaluated points `pts` (%{x, fitness})."
  def suggest(space, pts, n, rng) do
    {lo, hi} = {space.lo * 1.0, space.hi * 1.0}
    xs = Enum.map(pts, fn p -> Enum.map(p.x, &((&1 - lo) / (hi - lo))) end)
    ys = Enum.map(pts, & &1.fitness)
    {mu, sd} = {mean(ys), max(stdev(ys), 1.0e-12)}
    zs = Enum.map(ys, &((&1 - mu) / sd))
    model = fit(xs, zs)

    {chosen, rng, _model} =
      Enum.reduce(1..min(n, 4), {[], rng, model}, fn _, {acc, r, m} ->
        {x, r} = argmax_ei(m, space.n, r)
        {pm, _} = predict(m, x)
        {[x | acc], r, refit(m, x, pm)}
      end)

    out = chosen |> Enum.reverse() |> Enum.map(fn u -> to_space(space, u) end)
    {out, rng}
  end

  defp to_space(%{kind: :reals, lo: lo, hi: hi}, u), do: Enum.map(u, &(lo + &1 * (hi - lo)))
  defp to_space(%{kind: :ints, lo: lo, hi: hi}, u), do: Enum.map(u, &round(lo + &1 * (hi - lo)))

  @doc "Fit a GP to unit-cube points and standardised values; picks hyperparameters by marginal likelihood."
  def fit(xs, zs) do
    grid = for l <- [0.05, 0.1, 0.2, 0.35, 0.6, 1.0], s2 <- [1.0e-6, 1.0e-3, 1.0e-2, 0.1], do: {l, s2}

    grid
    |> Enum.map(fn {l, s2} -> build(xs, zs, l, s2) end)
    |> Enum.reject(&is_nil/1)
    |> Enum.max_by(& &1.lml)
  end

  defp refit(m, x, y), do: build(m.xs ++ [x], m.zs ++ [y], m.l, m.s2) || m

  defp build(xs, zs, l, s2) do
    k = for a <- xs, do: for(b <- xs, do: kern(a, b, l))
    k = k |> Enum.with_index() |> Enum.map(fn {row, i} -> List.update_at(row, i, &(&1 + s2 + 1.0e-9)) end)

    case cholesky(k) do
      nil -> nil
      lmat ->
        alpha = backsub_t(lmat, forward(lmat, zs))
        logdet = lmat |> Enum.with_index() |> Enum.map(fn {r, i} -> :math.log(Enum.at(r, i)) end) |> Enum.sum()
        lml = -0.5 * dot(zs, alpha) - logdet - 0.5 * length(zs) * :math.log(2 * :math.pi())
        %{xs: xs, zs: zs, l: l, s2: s2, lmat: lmat, alpha: alpha, lml: lml, best: Enum.max(zs)}
    end
  end

  @doc "Posterior mean and standard deviation at a unit-cube point."
  def predict(m, x) do
    ks = Enum.map(m.xs, &kern(&1, x, m.l))
    mean = dot(ks, m.alpha)
    v = forward(m.lmat, ks)
    var = max(1.0 + m.s2 - dot(v, v), 1.0e-12)
    {mean, :math.sqrt(var)}
  end

  defp ei(m, x) do
    {mu, s} = predict(m, x)
    z = (mu - m.best) / s
    (mu - m.best) * cdf(z) + s * pdf(z)
  end

  defp argmax_ei(m, n, rng) do
    {rand_pts, rng} = Enum.map_reduce(1..600, rng, fn _, r -> Enum.map_reduce(1..n, r, fn _, r2 -> :rand.uniform_s(r2) end) end)
    tops = m.xs |> Enum.zip(m.zs) |> Enum.sort_by(&elem(&1, 1), :desc) |> Enum.take(5) |> Enum.map(&elem(&1, 0))
    {local, rng} =
      Enum.map_reduce(1..200, rng, fn i, r ->
        base = Enum.at(tops, rem(i, length(tops)))
        Enum.map_reduce(base, r, fn c, r2 -> {z, r2} = :rand.normal_s(r2); {min(max(c + 0.05 * z, 0.0), 1.0), r2} end)
      end)
    best = Enum.max_by(rand_pts ++ local, &ei(m, &1))
    {best, rng}
  end

  defp kern(a, b, l) do
    r = :math.sqrt(Enum.zip_with(a, b, &((&1 - &2) * (&1 - &2))) |> Enum.sum()) / l
    s5 = :math.sqrt(5) * r
    (1 + s5 + 5 * r * r / 3) * :math.exp(-s5)
  end

  defp cdf(z), do: 0.5 * :math.erfc(-z / :math.sqrt(2))
  defp pdf(z), do: :math.exp(-z * z / 2) / :math.sqrt(2 * :math.pi())
  defp dot(a, b), do: Enum.zip_with(a, b, &(&1 * &2)) |> Enum.sum()
  defp mean(xs), do: Enum.sum(xs) / length(xs)
  defp stdev(xs), do: (m = mean(xs); :math.sqrt(Enum.sum(Enum.map(xs, &((&1 - m) ** 2))) / max(length(xs) - 1, 1)))

  @doc "Lower Cholesky factor of a symmetric positive-definite matrix (rows), or nil."
  def cholesky(a) do
    n = length(a)
    at = a |> Enum.map(&List.to_tuple/1) |> List.to_tuple()

    try do
      l =
        Enum.reduce(0..(n - 1), %{}, fn i, l ->
          Enum.reduce(0..i, l, fn j, l ->
            s = Enum.reduce(0..(j - 1)//1, 0.0, fn k, s -> s + Map.fetch!(l, {i, k}) * Map.fetch!(l, {j, k}) end)
            if i == j do
              d = elem(elem(at, i), i) - s
              if d <= 0, do: throw(:not_pd)
              Map.put(l, {i, j}, :math.sqrt(d))
            else
              Map.put(l, {i, j}, (elem(elem(at, i), j) - s) / Map.fetch!(l, {j, j}))
            end
          end)
        end)

      for i <- 0..(n - 1), do: for(j <- 0..(n - 1), do: Map.get(l, {i, j}, 0.0))
    catch
      :not_pd -> nil
    end
  end

  defp forward(l, b) do
    {ys, _} =
      Enum.reduce(Enum.with_index(l), {[], []}, fn {row, i}, {acc, _} ->
        s = Enum.zip(Enum.take(row, i), Enum.reverse(acc)) |> Enum.reduce(0.0, fn {lij, yj}, s -> s + lij * yj end)
        {[(Enum.at(b, i) - s) / Enum.at(row, i) | acc], nil}
      end)
    Enum.reverse(ys)
  end

  defp backsub_t(l, y) do
    n = length(l)
    lt = l |> Enum.map(&List.to_tuple/1) |> List.to_tuple()
    yt = List.to_tuple(y)
    Enum.reduce((n - 1)..0//-1, %{}, fn i, x ->
      s = Enum.reduce((i + 1)..(n - 1)//1, 0.0, fn j, s -> s + elem(elem(lt, j), i) * Map.fetch!(x, j) end)
      Map.put(x, i, (elem(yt, i) - s) / elem(elem(lt, i), i))
    end)
    |> then(fn x -> for i <- 0..(n - 1), do: x[i] end)
  end
end
