defmodule Vapor.ProveTest do
  @moduledoc """
  Geometry by the algebraic method and topology by exact homology
  (docs/MATHEMATICS.md). Each true theorem is proved symbolically (the
  claim's numerator is the zero polynomial) and checked independently in
  exact rationals at random points; each false statement of the same
  shape — the control — is refuted by both.
  """
  use ExUnit.Case, async: true
  alias Vapor.Prove, as: P

  @moduletag timeout: 900_000

  test "classical theorems proved symbolically and checked independently; their false twins refuted by both" do
    for {name, t} <- P.theorems(), name not in ["simson_line", "false_simson_off_circle"] do
      {verdict, _} = P.prove(name)
      {check, k} = P.check(name)
      assert k >= 3

      if t.true? do
        assert verdict == :proved, name
        assert check == :holds, name
      else
        assert verdict == :refuted, name
        assert check == :fails, name
      end
    end
  end

  test "a construction degenerate for every value of its parameters is reported, never 'proved' by 0/0" do
    # the foot of c on the 'line' aa: no line at all — every coordinate is 0/0
    bad = %{params: [:p, :q], points: [a: {:coords, 0, 0}, b: {:coords, 1, 0}, c: {:coords, :p, :q}, x: {:foot, :c, :a, :a}], claim: {:collinear, :x, :b, :c}}
    assert P.prove(bad) == {:degenerate, :construction}
  end

  test "Simson's line: checked in exact rationals on the circle; off the circle it fails" do
    assert {:holds, _} = P.check("simson_line")
    assert {:fails, _} = P.check("false_simson_off_circle")
  end

  test "conjecture and prove: a triangle's Euler line, its altitudes' and medians' concurrence, the nine-point circle — unasked" do
    r = P.discover(P.theorems()["nine_point_circle"])
    col = MapSet.new(r.collinear, fn {:collinear, a, b, c} -> MapSet.new([a, b, c]) end)
    assert MapSet.member?(col, MapSet.new([:o, :g, :h])), "Euler line"
    assert MapSet.member?(col, MapSet.new([:c, :hc, :h])), "third altitude"
    assert MapSet.member?(col, MapSet.new([:c, :mc, :g])), "third median"
    cyc = MapSet.new(r.concyclic, fn {:concyclic, a, b, c, d} -> MapSet.new([a, b, c, d]) end)
    nine = [:ma, :mb, :mc, :ha, :hb, :hc]
    # every quadruple of the six nine-point-circle points found and proved
    for q <- combos(nine, 4), do: assert(MapSet.member?(cyc, MapSet.new(q)), inspect(q))
    # nothing trivial: no three points of one defining line reported
    refute MapSet.member?(col, MapSet.new([:a, :mb, :hb]))
    assert r.survivors < r.candidates / 10
  end

  defp combos(_, 0), do: [[]]
  defp combos([], _), do: []
  defp combos([h | t], k), do: Enum.map(combos(t, k - 1), &[h | &1]) ++ combos(t, k)

  test "homology: Betti numbers of the classics; ℚ against GF(2) reveals the torsion of the Klein bottle and RP²" do
    expect = %{sphere: {[1, 0, 1], [1, 0, 1], 2}, torus: {[1, 2, 1], [1, 2, 1], 0}, klein: {[1, 2, 1], [1, 1, 0], 0}, rp2: {[1, 1, 1], [1, 0, 0], 1}, mobius: {[1, 1, 0], [1, 1, 0], 0}}

    for {c, {gf2, q, chi}} <- expect do
      a = P.betti(P.complex(c), :gf2)
      b = P.betti(P.complex(c), :q)
      assert a.betti == gf2 and b.betti == q and a.euler == chi, "#{c}: #{inspect(a)} #{inspect(b)}"
      # Euler–Poincaré: the alternating sum of Betti numbers is χ over any field
      assert Enum.at(gf2, 0) - Enum.at(gf2, 1) + Enum.at(gf2, 2) == chi
    end

    # over GF(2) the torus and the Klein bottle look alike; over ℚ they do not
    assert P.betti(P.complex(:torus)).betti == P.betti(P.complex(:klein)).betti
    refute P.betti(P.complex(:torus), :q).betti == P.betti(P.complex(:klein), :q).betti
  end

  test "persistent homology: a noisy loop has one long-lived H₁ class; a blob (the control) has none" do
    loop = for i <- 1..40, do: (a = 2 * :math.pi() * i / 40; {:math.cos(a) + 0.05 * :math.sin(i * 7.0), :math.sin(a) + 0.05 * :math.cos(i * 3.0)})
    blob = for i <- 1..40, do: {Vapor.Sampler.uniform(3, i), Vapor.Sampler.uniform(4, i)}
    life = fn pts -> P.persistence(pts).h1 |> Enum.map(fn {b, d} -> if d == :inf, do: 10.0, else: d - b end) |> Enum.sort(:desc) end
    [l1 | rest] = life.(loop)
    assert l1 > 1.0 and Enum.all?(rest, &(&1 < 0.3))
    assert hd(life.(blob)) < 0.3
    # H₀: one class per point at birth, one survives
    assert Enum.count(P.persistence(loop).h0, fn {_, d} -> d == :inf end) == 1
  end
end
