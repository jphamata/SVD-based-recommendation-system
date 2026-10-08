defmodule Vapor.CausalTest do
  use ExUnit.Case, async: true
  alias Vapor.Logic.{Causal, LP}

  # ---------------------------------------------------------------- models

  # a random structural causal model over binary variables that fits the diagram: one hidden
  # binary parent per bidirected edge, every probability a rational strictly inside (0, 1)
  defp scm(g, seed) do
    :rand.seed(:exsss, {seed, 7, 11})
    pr = fn -> LP.q(:rand.uniform(9), 10) end
    hidden = for {a, b} <- Enum.sort(g.bi), do: {"u_#{a}_#{b}", a, b}
    hpar = fn v -> for {h, a, b} <- hidden, v in [a, b], do: h end

    cpt =
      Map.new(g.nodes, fn v ->
        ps = Causal.parents(g, v) ++ hpar.(v)
        {v, {ps, Map.new(Causal.assignments(ps), fn a -> {a, pr.()} end)}}
      end)

    %{nodes: g.nodes, hidden: Map.new(hidden, fn {h, _, _} -> {h, pr.()} end), cpt: cpt}
  end

  defp bern(p, 1), do: p
  defp bern(p, 0), do: LP.qsub({1, 1}, p)

  # P(observed) with the variables in `fixed` set by intervention (their mechanisms cut)
  defp joint(m, fixed \\ %{}) do
    hs = Map.keys(m.hidden)

    for obs <- Causal.assignments(m.nodes), Enum.all?(fixed, fn {k, v} -> obs[k] == v end), into: %{} do
      p =
        Enum.reduce(Causal.assignments(hs), {0, 1}, fn ha, acc ->
          a = Map.merge(obs, ha)
          ph = Enum.reduce(hs, {1, 1}, fn h, s -> LP.qmul(s, bern(m.hidden[h], ha[h])) end)

          po =
            Enum.reduce(m.nodes, {1, 1}, fn v, s ->
              if Map.has_key?(fixed, v) do
                s
              else
                {ps, t} = m.cpt[v]
                LP.qmul(s, bern(t[Map.take(a, ps)], a[v]))
              end
            end)

          LP.qadd(acc, LP.qmul(ph, po))
        end)

      {obs, p}
    end
  end

  defp prob(j, fixed), do: Enum.reduce(j, {0, 1}, fn {a, p}, s -> if Enum.all?(fixed, fn {k, v} -> a[k] == v end), do: LP.qadd(s, p), else: s end)

  # the truth: cut the model at X, read P(Y); against the estimand evaluated on the observed joint
  defp agrees?(g, ys, xs, estimand, seed) do
    m = scm(g, seed)
    obs = joint(m)

    # the estimand must give the effect at every value of its other free variables
    free = Causal.free_vars(estimand) -- (xs ++ ys)

    Enum.all?(Causal.assignments(xs), fn xa ->
      cut = joint(m, xa)

      Enum.all?(Causal.assignments(ys), fn ya ->
        Enum.all?(Causal.assignments(free), fn fa -> Causal.eval(estimand, obs, Map.merge(fa, Map.merge(xa, ya))) == prob(cut, ya) end)
      end)
    end)
  end

  defp g!(edges), do: (fn {:ok, g} -> g end).(Causal.graph(edges))

  # ----------------------------------------------------------------- tests

  test "front door: identifiable through the mediator, exact on random models; the naive P(y | x) is not" do
    g = g!([{:dir, "x", "m"}, {:dir, "m", "y"}, {:bi, "x", "y"}])
    assert {:ok, e} = Causal.identify(g, ["y"], ["x"])
    assert Causal.to_text(e) =~ "Σ"
    for seed <- 1..5, do: assert(agrees?(g, ["y"], ["x"], e, seed))
    # the control: conditioning is not intervening under a hidden confounder
    refute Enum.all?(1..5, &agrees?(g, ["y"], ["x"], {:p, ["y"], ["x"]}, &1))
  end

  test "the napkin: identifiable only as a ratio (line 7 of ID)" do
    g = g!([{:dir, "w", "z"}, {:dir, "z", "x"}, {:dir, "x", "y"}, {:bi, "w", "x"}, {:bi, "w", "y"}])
    assert {:ok, e} = Causal.identify(g, ["y"], ["x"])
    assert Causal.to_text(e) =~ "/"
    for seed <- 1..4, do: assert(agrees?(g, ["y"], ["x"], e, seed))
  end

  test "the bow is not identifiable: the hedge checks, and two models agree on P(x, y) but not on P(y | do(x))" do
    g = g!([{:dir, "x", "y"}, {:bi, "x", "y"}])
    assert {:fail, h} = Causal.identify(g, ["y"], ["x"])
    assert Causal.hedge?(g, h.ys, h.xs, h.f, h.f_prime)
    refute Causal.hedge?(g, h.ys, h.xs, h.f_prime, h.f)

    # two models for the bow, computed: x copies a hidden coin u; in A, y copies u; in B, y copies x
    cpt = fn ypar -> %{"x" => {["u_x_y"], %{%{"u_x_y" => 0} => {0, 1}, %{"u_x_y" => 1} => {1, 1}}},
                       "y" => {["x", "u_x_y"], Map.new(Causal.assignments(["x", "u_x_y"]), fn a -> {a, {a[ypar], 1}} end)}} end
    a = %{nodes: ["x", "y"], hidden: %{"u_x_y" => {1, 2}}, cpt: cpt.("u_x_y")}
    b = %{a | cpt: cpt.("x")}
    assert joint(a) == joint(b)
    assert prob(joint(a, %{"x" => 1}), %{"y" => 1}) == {1, 2}
    assert prob(joint(b, %{"x" => 1}), %{"y" => 1}) == {1, 1}
  end

  test "random diagrams: every identified estimand equals the true intervention; every failure carries a checked hedge" do
    :rand.seed(:exsss, {3, 5, 8})
    names = ~w(a b c d e)

    {ok, failed} =
      Enum.reduce(1..30, {0, 0}, fn i, {ok, failed} ->
        dir = for {p, j} <- Enum.with_index(names), {q, k} <- Enum.with_index(names), j < k, :rand.uniform() < 0.4, do: {:dir, p, q}
        bi = for {p, j} <- Enum.with_index(names), {q, k} <- Enum.with_index(names), j < k, :rand.uniform() < 0.2, do: {:bi, p, q}
        {:ok, g} = Causal.graph(dir ++ bi, names)
        [x, y] = Enum.take_random(names, 2) |> Enum.sort_by(&Enum.find_index(Causal.topo(g), fn n -> n == &1 end))

        case Causal.identify(g, [y], [x]) do
          {:ok, e} ->
            assert agrees?(g, [y], [x], e, i), "#{inspect(dir ++ bi)}: P(#{y} | do(#{x})) = #{Causal.to_text(e)}"
            {ok + 1, failed}

          {:fail, h} ->
            assert Causal.hedge?(g, h.ys, h.xs, h.f, h.f_prime), inspect({dir ++ bi, x, y, h})
            {ok, failed + 1}
        end
      end)

    # both branches were exercised
    assert ok > 5 and failed > 0
  end

  test "back-door: a valid set, the empty set refused with the open path, a descendant refused" do
    g = g!([{:dir, "z", "x"}, {:dir, "z", "y"}, {:dir, "x", "y"}, {:dir, "x", "m"}])
    assert :ok = Causal.backdoor(g, ["x"], ["y"], ["z"])
    assert {:refuted, why} = Causal.backdoor(g, ["x"], ["y"], [])
    assert why =~ "x – z – y" or why =~ "z"
    assert {:refuted, why2} = Causal.backdoor(g, ["x"], ["y"], ["z", "m"])
    assert why2 =~ "descends"
    # the adjustment formula is the true effect
    e = Causal.adjustment_estimand(["x"], ["y"], ["z"])
    for seed <- 1..3, do: assert(agrees?(g, ["y"], ["x"], e, seed))
  end

  test "d-separation: chain, fork, collider, and a hidden confounder" do
    g = g!([{:dir, "a", "b"}, {:dir, "b", "c"}, {:dir, "a", "d"}, {:dir, "c", "d"}, {:bi, "c", "e"}])
    assert Causal.dsep?(g, ["a"], ["c"], ["b"])
    refute Causal.dsep?(g, ["a"], ["c"], [])
    # conditioning on a collider opens it
    refute Causal.dsep?(g, ["a"], ["c"], ["b", "d"])
    # b → c ← u → e: c is a collider on the hidden confounder's path, closed until c is given
    assert Causal.dsep?(g, ["b"], ["e"], [])
    refute Causal.dsep?(g, ["b"], ["e"], ["c"])
    assert Causal.dsep?(g, ["a"], ["e"], ["b"])
  end

  test "the desk's text form" do
    text = """
    causal
    x -> m
    m -> y
    x <-> y
    identify y | do(x)
    backdoor x -> y | m
    dsep x; y | m
    """

    {:ok, r} = Causal.run(text)
    assert [%{verdict: "identifiable"}, %{verdict: "invalid"}, %{verdict: "d-connected"}] = r.results
    {:ok, bow} = Causal.run("x -> y\nx <-> y\nidentify y | do(x)")
    assert [%{verdict: "not identifiable", hedge_checked: true}] = bow.results
    assert {:error, _} = Causal.run("x -> y\ny -> x\nidentify y | do(x)")
    assert {:error, why} = Causal.run("x -> y\nidentify q | do(x)")
    assert why =~ "not in the diagram"
  end
end
