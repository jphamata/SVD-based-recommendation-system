defmodule Vapor.MIPTest do
  use ExUnit.Case, async: true
  alias Vapor.Logic.{LP, MIP}

  # the relaxation takes 3/5 of a: the integer optimum needs branching
  @knap """
  maximize 8a + 11b + 6c + 4d
  5a + 7b + 4c + 3d <= 14
  bin a, b, c, d
  """

  # brute force over a box: an oracle that shares nothing with branch and bound
  defp brute(text, box) do
    {:ok, p} = MIP.parse(text)
    vars = p.lp.vars
    pts = Enum.reduce(vars, [%{}], fn v, acc -> for m <- acc, k <- box, do: Map.put(m, v, {k, 1}) end)
    feas = Enum.filter(pts, &feasible?(p, &1))
    best = if p.lp.sense == :max, do: Enum.max_by(feas, &LP.to_float(value(p, &1)), fn -> nil end), else: Enum.min_by(feas, &LP.to_float(value(p, &1)), fn -> nil end)
    best && value(p, best)
  end

  defp value(p, x), do: Enum.reduce(p.lp.vars, p.lp.c0, fn v, s -> LP.qadd(s, LP.qmul(Map.get(p.lp.c, v, {0, 1}), x[v])) end)

  defp feasible?(p, x) do
    Enum.all?(p.lp.rows, fn {co, op, rhs} ->
      s = LP.qcmp(Enum.reduce(co, {0, 1}, fn {v, a}, acc -> LP.qadd(acc, LP.qmul(a, x[v])) end), rhs)
      case op do :le -> s <= 0; :ge -> s >= 0; :eq -> s == 0 end
    end)
  end

  test "a knapsack: the optimum agrees with brute force, and the certificate checks" do
    {:ok, r} = MIP.solve(@knap)
    assert r.status == :optimal
    assert r.check.accepted, r.check.reason
    assert r.objective == brute(@knap, 0..1)
    assert r.objective == {21, 1}
    # the relaxation is fractional here, so the answer really needed branching
    {:ok, p} = MIP.parse(@knap)
    {:ok, lp} = LP.solve(p.lp)
    assert LP.qcmp(lp.objective, r.objective) > 0
    assert match?({:branch, _, _, _, _}, r.certificate.tree)
  end

  test "random small integer programs agree with brute force (min and max, ≤ ≥ =)" do
    :rand.seed(:exsss, {17, 29, 31})

    for _ <- 1..40 do
      coef = fn -> :rand.uniform(9) - 3 end
      sense = Enum.random(["maximize", "minimize"])
      rows = for _ <- 1..3, do: "#{coef.()}x + #{coef.()}y + #{coef.()}z #{Enum.random(["<=", ">="])} #{:rand.uniform(12) - 4}"
      bounds = ["x <= 4", "y <= 4", "z <= 4"]
      text = Enum.join(["#{sense} #{coef.()}x + #{coef.()}y + #{coef.()}z"] ++ rows ++ bounds ++ ["int x, y, z"], "\n")

      expected = brute(text, 0..4)
      {:ok, r} = MIP.solve(text)

      case expected do
        nil -> assert r.status == :infeasible, text
        v -> assert {r.status, r.objective} == {:optimal, v}, text
      end

      assert r.check.accepted, text <> "\n" <> r.check.reason
    end
  end

  test "infeasible only because of integrality: every leaf is a Farkas certificate" do
    text = "maximize x\n2x = 1\nint x"
    {:ok, lp} = LP.solve("maximize x\n2x = 1")
    assert lp.status == :optimal
    {:ok, r} = MIP.solve(text)
    assert r.status == :infeasible
    assert r.check.accepted
    assert r.check.reason =~ "no integer point"
  end

  test "binaries and free integer variables" do
    {:ok, r} = MIP.solve("minimize 3d + 2e - f\nd + e >= 1\nf <= 2\nf >= -3\nbin d, e\nint f\nfree f")
    assert {r.status, r.objective} == {:optimal, {0, 1}}
    assert r.x["e"] == {1, 1} and r.x["d"] == {0, 1} and r.x["f"] == {2, 1}
    assert r.check.accepted
  end

  test "a forged optimum is refused: the honest tree bounds every leaf by the true value" do
    {:ok, p} = MIP.parse(@knap)
    {:ok, r} = MIP.solve(p)
    # claim a better value: the incumbent's objective no longer matches
    forged = put_in(r.certificate, [:incumbent, :objective], LP.qadd(r.objective, {1, 1}))
    refute MIP.check(p, forged).accepted
    # a worse incumbent with the same tree: some leaf's bound beats it
    worse = %{r.certificate | incumbent: %{x: Map.new(p.lp.vars, &{&1, {0, 1}}), objective: {0, 1}}}
    %{accepted: false, reason: why} = MIP.check(p, worse)
    assert why =~ "could beat the incumbent"
  end

  test "a tampered tree is refused" do
    {:ok, p} = MIP.parse(@knap)
    {:ok, r} = MIP.solve(p)
    assert {:branch, v, k, lo, hi} = r.certificate.tree

    # a split that skips integer points (v ≤ k and v ≥ k + 2) cannot be expressed: the checker
    # derives both sides itself — so the tamper is a split on a continuous variable instead
    {:ok, p2} = MIP.parse(@knap |> String.replace("bin a, b, c, d", "bin a, b\nbin c, d"))
    assert MIP.check(p2, r.certificate).accepted
    bad_var = %{r.certificate | tree: {:branch, "nope", k, lo, hi}}
    refute MIP.check(p, bad_var).accepted

    # drop a leaf's certificate
    broken = %{r.certificate | tree: {:branch, v, k, {:leaf, :infeasible, List.duplicate({0, 1}, 4)}, hi}}
    refute MIP.check(p, broken).accepted
  end

  test "a node budget ends in :exhausted with the gap, never a guess" do
    text = """
    maximize 12a + 11b + 10c + 9d + 8e + 7f + 6g + 5h
    7a + 6b + 6c + 5d + 5e + 4f + 4g + 3h <= 19
    a <= 1
    b <= 1
    c <= 1
    d <= 1
    e <= 1
    f <= 1
    g <= 1
    h <= 1
    int a, b, c, d, e, f, g, h
    """

    {:ok, full} = MIP.solve(text)
    assert full.status == :optimal and full.check.accepted
    {:ok, cut} = MIP.solve(text, max_nodes: 3)
    assert cut.status == :exhausted
    assert LP.qcmp(cut.bound, full.objective) >= 0
    if cut.objective, do: assert(LP.qcmp(cut.objective, full.objective) <= 0)
  end

  test "an unbounded relaxation is refused, as are malformed declarations" do
    assert {:error, why} = MIP.solve("maximize x\nx >= 1\nint x")
    assert why =~ "unbounded"
    assert {:error, _} = MIP.parse("maximize x\nx <= 3")
    assert {:error, _} = MIP.parse("maximize x\nx <= 3\nint y")
  end

  test "the logic desk solves it, and checks anyone's proposal as JSON" do
    {:ok, v} = Vapor.Logic.run(@knap)
    assert v.kind == "integer linear" and v.verdict == "optimal" and v.certified
    # round trip through JSON: the presented tree is a valid proposal
    prop = v |> Vapor.JSON.encode() |> Vapor.JSON.decode!()
    proposal = %{"incumbent" => Map.new(prop["x"], fn {k, x} -> {k, x["exact"]} end), "objective" => prop["objective"]["exact"], "tree" => prop["tree"]}
    assert {:ok, %{accepted: true}} = Vapor.Logic.check(@knap, proposal)
    assert {:ok, %{accepted: false}} = Vapor.Logic.check(@knap, %{proposal | "objective" => "22"})
    assert {:error, _} = Vapor.Logic.check(@knap, %{"tree" => %{"leaf" => "magic", "y" => []}})
  end

  test "the presentation is JSON-ready" do
    j = Vapor.JSON.encode(MIP.present(MIP.solve(@knap)))
    assert j =~ ~s("status":"optimal")
  end
end
