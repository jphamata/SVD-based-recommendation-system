defmodule Vapor.RecommendTest do
  use ExUnit.Case, async: true
  alias Vapor.{Dense, Recommend}

  @moduletag timeout: 300_000

  defp matrix(m, n, seed) do
    {rows, _} = Enum.map_reduce(1..m, :rand.seed_s(:exsss, {seed, 3, 5}), fn _, s -> Enum.map_reduce(1..n, s, fn _, s -> :rand.normal_s(s) end) end)
    rows
  end

  # ratings from a planted model: mean, biases, a rank-r interaction and noise; `frac` of the cells observed
  defp planted(users, items, r, noise, frac, seed) do
    s0 = :rand.seed_s(:exsss, {seed, 7, 9})
    draw = fn s, n, sc -> Enum.map_reduce(1..n, s, fn _, s -> {x, s} = :rand.normal_s(s); {sc * x, s} end) end
    {bu, s} = draw.(s0, users, 0.5)
    {bi, s} = draw.(s, items, 0.5)
    {p, s} = Enum.map_reduce(1..users, s, fn _, s -> draw.(s, r, 1.0) end)
    {q, s} = Enum.map_reduce(1..items, s, fn _, s -> draw.(s, r, 1.0) end)

    {ratings, _} =
      for(u <- 0..(users - 1), i <- 0..(items - 1), do: {u, i})
      |> Enum.flat_map_reduce(s, fn {u, i}, s ->
        {keep, s} = :rand.uniform_s(s)
        {e, s} = :rand.normal_s(s)
        v = 3.0 + Enum.at(bu, u) + Enum.at(bi, i) + Dense.dot(Enum.at(p, u), Enum.at(q, i)) + noise * e
        {if(keep < frac, do: [{u, i, v}], else: []), s}
      end)

    %{users: Enum.map(0..(users - 1), &"u#{&1}"), items: Enum.map(0..(items - 1), &"i#{&1}"), ratings: ratings}
  end

  test "SVD by one-sided Jacobi: the certificate holds, the values agree with eigh(AᵀA), Eckart–Young is exact" do
    for {m, n} <- [{9, 5}, {4, 7}, {6, 6}] do
      a = matrix(m, n, m * 10 + n)
      f = Dense.svd(a)
      c = Dense.svd_residual(a, f)
      assert c.residual < 1.0e-13 and c.u_orth < 1.0e-12 and c.v_orth < 1.0e-12
      assert f.s == Enum.sort(f.s, :desc)
      # an independent route to the same values: the symmetric eigenproblem of AᵀA (or AAᵀ)
      g = if m >= n, do: Dense.matmul(Dense.transpose(a), a), else: Dense.matmul(a, Dense.transpose(a))
      {vals, _} = Dense.eigh(g)
      ev = vals |> Enum.map(&:math.sqrt(max(&1, 0.0))) |> Enum.sort(:desc)
      assert Enum.zip_with(f.s, ev, &abs(&1 - &2)) |> Enum.max() < 1.0e-10

      {:ok, t} = Recommend.truncated_svd(a, 2)
      assert abs(t.error2 - t.tail2) <= 1.0e-10 * max(t.tail2, 1.0)
    end
  end

  test "planted interactions are found: signal, near the noise level, against both baselines and the shuffled control" do
    data = planted(50, 36, 3, 0.25, 0.45, 1)
    {:ok, r} = Recommend.evaluate(data, ranks: [0, 1, 2, 3, 5], lambdas: [0.3, 1.0, 5.0])
    assert r.verdict == "signal", inspect(r)
    assert r.rank >= 2
    assert r.rmse.model < r.rmse.biases and r.rmse.biases < r.rmse.mean
    assert r.rmse.model < 0.6
    assert r.paired.p_value <= 0.05
    assert r.rmse.shuffled_control >= 0.95 * r.rmse.mean
  end

  test "no interactions planted: no claim that they help (the test does not invent a signal)" do
    data = planted(50, 36, 3, 0.25, 0.45, 2)
    # biases plus unstructured noise (a hash of the cell): nothing of low rank to find
    flat = %{data | ratings: Enum.map(data.ratings, fn {u, i, _} -> {u, i, 3.0 + 0.4 * rem(u, 5) - 0.3 * rem(i, 4) + 0.5 * (:erlang.phash2({u, i}, 1000) / 1000 - 0.5)} end)}
    {:ok, r} = Recommend.evaluate(flat, ranks: [0, 1, 2], lambdas: [1.0, 5.0])
    refute r.verdict == "signal"
  end

  test "overwriting missing ratings with the scale's midpoint (the old script) costs accuracy, measured" do
    data = planted(50, 36, 3, 0.25, 0.45, 3)
    {train, test} = Recommend.split(data.ratings, 0.2, 1)
    dims = [users: 50, items: 36, k: 3, lambda: 1.0]
    honest = Recommend.fit(train, dims)
    # the script's move: 60 % of the observed training ratings replaced by the midpoint, then trained on as data
    {lo, hi} = train |> Enum.map(&elem(&1, 2)) |> Enum.min_max()
    mid = (lo + hi) / 2
    imputed = Enum.map(train, fn {u, i, v} -> if :erlang.phash2({u, i}, 100) < 60, do: {u, i, mid}, else: {u, i, v} end)
    damaged = Recommend.fit(imputed, dims)
    assert Recommend.rmse(damaged, test) > 1.5 * Recommend.rmse(honest, test)
  end

  test "the same data, rank, λ and seed give the same model, bit for bit" do
    data = planted(20, 15, 2, 0.2, 0.5, 4)
    a = Recommend.fit(data.ratings, k: 2, seed: 9)
    b = Recommend.fit(data.ratings, k: 2, seed: 9)
    assert a == b
  end

  test "the original repository's table: quoted names, missing cells, a full report with recommendations" do
    {:ok, d} = Recommend.parse_csv(File.read!(Path.expand("../fixtures/recommend/genres_films.csv", __DIR__)))
    assert length(d.users) == 6 and length(d.items) == 70
    assert "Cinema \"Arte\"/Cult" in d.users
    # the original table's last row is two values short: those two cells are missing, as they should be
    assert length(d.ratings) == 6 * 70 - 2
    # cells removed by the old script are missing here, not 50
    sparse = %{d | ratings: Enum.reject(d.ratings, fn {u, i, _} -> :erlang.phash2({u, i}, 10) < 4 end)}
    {:ok, r} = Recommend.evaluate(sparse, ranks: [0, 1, 2], lambdas: [1.0, 5.0, 20.0], top: 3)
    assert r.verdict in ["signal", "no evidence the interactions help (the biases explain what is predictable)", "flawed: the shuffled control beats the mean"]
    assert map_size(r.recommendations) == 6
    assert Enum.all?(Map.values(r.recommendations), &(length(&1) == 3))
    assert {:error, _} = Recommend.parse_csv("a,b\nx,abc\n")
  end
end
