defmodule Vapor.InfoGeomTest do
  use ExUnit.Case, async: true
  alias Vapor.InfoGeom, as: G

  defp rand_dist(rng, k) do
    {xs, rng} = Enum.map_reduce(1..k, rng, fn _, r -> {u, r} = Vapor.Entropy.float(r); {u + 1.0e-3, r} end)
    {G.normalize(xs), rng}
  end

  test "Fisher–Rao is a metric on the simplex (symmetric, zero only at equality, triangle inequality, bounded by π)" do
    rng = Vapor.Entropy.rng({:fr, :metric})

    Enum.reduce(1..500, rng, fn _, rng ->
      {p, rng} = rand_dist(rng, 5)
      {q, rng} = rand_dist(rng, 5)
      {r, rng} = rand_dist(rng, 5)
      assert_in_delta G.fisher_rao(p, q), G.fisher_rao(q, p), 1.0e-12
      assert G.fisher_rao(p, p) < 1.0e-6
      assert G.fisher_rao(p, r) <= G.fisher_rao(p, q) + G.fisher_rao(q, r) + 1.0e-12
      assert G.fisher_rao(p, q) <= :math.pi() + 1.0e-12
      rng
    end)

    assert_in_delta G.fisher_rao([1.0, 0.0], [0.0, 1.0]), :math.pi(), 1.0e-12
  end

  test "the control: KL is asymmetric and breaks the triangle inequality, so it cannot rank 'closer'" do
    p = [0.98, 0.01, 0.01]
    q = [0.4, 0.3, 0.3]
    r = [0.01, 0.01, 0.98]
    refute_in_delta G.kl(p, q), G.kl(q, p), 0.1
    assert G.kl(p, r) > G.kl(p, q) + G.kl(q, r)
    assert G.fisher_rao(p, r) <= G.fisher_rao(p, q) + G.fisher_rao(q, r)
  end

  test "geodesics: the midpoint is halfway in distance; the Karcher mean of two points is that midpoint; relabelling commutes" do
    p = [0.7, 0.2, 0.1]
    q = [0.1, 0.3, 0.6]
    m = G.geodesic(p, q, 0.5)
    d = G.fisher_rao(p, q)
    assert_in_delta G.fisher_rao(p, m), d / 2, 1.0e-9
    assert_in_delta G.fisher_rao(m, q), d / 2, 1.0e-9
    {mean, _} = G.frechet_mean([p, q])
    assert G.fisher_rao(mean, m) < 1.0e-8
    perm = fn x -> [Enum.at(x, 2), Enum.at(x, 0), Enum.at(x, 1)] end
    {mp, _} = G.frechet_mean([perm.(p), perm.(q), perm.([0.3, 0.3, 0.4])])
    {mo, _} = G.frechet_mean([p, q, [0.3, 0.3, 0.4]])
    assert G.fisher_rao(mp, perm.(mo)) < 1.0e-8
  end

  test "normals: the hyperbolic closed form gives √2·|ln(σ₂/σ₁)| at equal means and grows with the mean gap" do
    for {s1, s2} <- [{1.0, 2.0}, {0.5, 3.0}, {2.0, 2.0}] do
      assert_in_delta G.gaussian_fisher_rao(0.0, s1, 0.0, s2), :math.sqrt(2) * abs(:math.log(s2 / s1)), 1.0e-12
    end

    assert G.gaussian_fisher_rao(0.0, 1.0, 1.0, 1.0) < G.gaussian_fisher_rao(0.0, 1.0, 2.0, 1.0)
    # distance between narrow distributions grows: the same mean gap matters more when σ is small
    assert G.gaussian_fisher_rao(0.0, 0.1, 1.0, 0.1) > G.gaussian_fisher_rao(0.0, 10.0, 1.0, 10.0)
  end

  test "natural gradient is invariant to how a feature is scaled; plain gradient — the control — is not" do
    rng = Vapor.Entropy.rng({:logistic})

    {rows, _} =
      Enum.map_reduce(1..120, rng, fn _, r ->
        {a, r} = Vapor.Entropy.normal(r)
        {b, r} = Vapor.Entropy.normal(r)
        {u, r} = Vapor.Entropy.float(r)
        y = if u < 1 / (1 + :math.exp(-(1.5 * a - 0.8 * b + 0.3))), do: 1, else: 0
        {{[1.0, a, b], y}, r}
      end)

    x = Enum.map(rows, &elem(&1, 0))
    y = Enum.map(rows, &elem(&1, 1))
    scaled = Enum.map(x, fn [c, a, b] -> [c, a * 1000.0, b] end)
    pred = fn th, xs -> Enum.map(xs, fn xi -> 1 / (1 + :math.exp(-Enum.zip_reduce(xi, th, 0.0, &(&3 + &1 * &2)))) end) end

    n1 = G.natural_logistic(x, y)
    n2 = G.natural_logistic(scaled, y)
    assert n1.steps <= 10 and n1.steps == n2.steps
    diff = Enum.zip_reduce(pred.(n1.theta, x), pred.(n2.theta, scaled), 0.0, &max(&3, abs(&1 - &2)))
    assert diff < 1.0e-9

    p1 = G.plain_logistic(x, y, steps: 200)
    p2 = G.plain_logistic(scaled, y, steps: 200, rate: 1.0e-6)
    gap = Enum.zip_reduce(pred.(p1.theta, x), pred.(p2.theta, scaled), 0.0, &max(&3, abs(&1 - &2)))
    assert gap > 1.0e-3
    assert p1.loss > n1.loss
  end

  test "assay geometry: the tuned model is near its base, the unrelated one far; the metric and the control are checked" do
    {:ok, r} = Vapor.Assay.run("geometry", Vapor.Assay.example("geometry"))
    [closest | _] = r.pairs
    assert {closest.a, closest.b} == {"base", "tuned"}
    assert Enum.all?(r.evidence, & &1.ok)
    assert hd(r.consensus).model == "other"
    assert {:error, msg} = Vapor.Assay.run("geometry", "model,item,p\nA,1,0.5")
    assert msg =~ "two models"
  end
end
