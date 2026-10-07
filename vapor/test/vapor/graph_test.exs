defmodule Vapor.GraphTest do
  @moduledoc """
  `Vapor.Graph` against theory, against its nulls, and against networkx:

    * the Barabási–Albert degrees pass the power-law tests; the
      Erdős–Rényi degrees (the control) are not called scale-free;
    * the Erdős–Rényi giant component equals S = 1 − e^{−cS};
    * a small world's clustering is hundreds of σ above its
      configuration null; a random graph's is not;
    * Louvain recovers a planted partition (NMI 1) and finds little
      structure in its degree-preserving null;
    * epidemics: below the heterogeneous mean-field threshold outbreaks
      die out, above it they become large;
    * robustness: a scale-free network shrugs off random failures and
      falls apart under attack;
    * PageRank: the vapor program (oracle and native: the same bits) ranks
      as the host iteration does; with networkx present, PageRank,
      clustering, assortativity and betweenness equal networkx's.
  """
  use ExUnit.Case, async: true
  alias Vapor.Graph, as: G

  @moduletag timeout: 600_000

  test "power laws: Barabási–Albert passes (α near 3), Erdős–Rényi is not called scale-free" do
    ba = G.barabasi_albert(1000, 3, 1)
    er = G.erdos_renyi(1000, 6 / 999, 2)
    pb = G.power_law(G.degrees(ba), boot: 40)
    pe = G.power_law(G.degrees(er), boot: 40)
    assert pb.verdict == :power_law and pb.alpha > 2.4 and pb.alpha < 3.3, inspect(Map.drop(pb, []))
    assert pe.verdict in [:rejected, :exponential], inspect(pe)
    # deterministic: the same seed, the same verdict to the last digit
    assert G.power_law(G.degrees(ba), boot: 40) == pb
  end

  test "the giant component of G(n, c/n) is the root of S = 1 − e^{−cS}; none below c = 1" do
    for c <- [1.5, 2.0, 3.0] do
      s = G.giant(G.erdos_renyi(3000, c / 2999, round(c * 10)))
      assert abs(s - G.er_giant(c)) < 0.03, "c=#{c}: #{s} vs #{G.er_giant(c)}"
    end

    assert G.giant(G.erdos_renyi(3000, 0.5 / 2999, 9)) < 0.02
  end

  test "small world: clustering far above its configuration null; a random graph's is not" do
    ws = G.zscore(G.watts_strogatz(500, 10, 0.05, 4), &G.avg_clustering/1, 8, 5)
    er = G.zscore(G.erdos_renyi(500, 10 / 499, 1), &G.avg_clustering/1, 8, 5)
    assert ws.z > 50
    assert abs(er.z) < 3
    # the null keeps every degree
    g = G.erdos_renyi(300, 0.03, 3)
    assert G.degrees(G.rewire(g, 3000, 1)) == G.degrees(g)
  end

  test "Louvain recovers a planted partition; its null has far less modularity" do
    {g, truth} = G.planted(4, 50, 0.3, 0.02, 7)
    labels = G.communities(g)
    assert G.nmi(labels, truth) > 0.95
    q = G.modularity(g, labels)
    null = G.rewire(g, 10 * G.edge_count(g), 3)
    q0 = G.modularity(null, G.communities(null))
    assert q > 0.5 and q > q0 + 0.25, "Q #{q} vs null #{q0}"
  end

  test "epidemics: dying out below the mean-field threshold, large above it" do
    g = G.erdos_renyi(1500, 8 / 1499, 11)
    tc = G.threshold(g)
    gamma = 0.5
    # β for a transmissibility T: T = β / (1 − (1 − β)(1 − γ)) ⇒ β = Tγ / (1 − T(1 − γ))
    beta = fn t -> t * gamma / (1 - t * (1 - gamma)) end
    below = for s <- 1..5, do: G.sir(g, {beta.(0.5 * tc), gamma}, seed: s, seeds: 5).final_size
    above = for s <- 1..5, do: G.sir(g, {beta.(2.5 * tc), gamma}, seed: s, seeds: 5).final_size
    assert Enum.max(below) < 0.05, inspect(below)
    assert Enum.sum(above) / 5 > 0.3, inspect(above)
  end

  test "a scale-free network survives random failure and breaks under attack" do
    ba = G.barabasi_albert(2000, 2, 3)
    fail = G.percolation(ba, 0.15, :failure)
    attack = G.percolation(ba, 0.15, :attack)
    assert fail > 0.9 and attack < 0.6 * fail, "failure #{fail}, attack #{attack}"
  end

  @tag :native
  test "PageRank as a vapor program: the same bits on oracle and native, the host's ranking" do
    g = G.barabasi_albert(200, 2, 5)
    {:ok, comp} = Vapor.Compile.Lower.lower(G.pagerank_program(g, 60))
    {:ok, o} = Vapor.Runtime.Native.run_oracle(comp, %{})
    {:ok, w} = Vapor.Runtime.Worker.start_link(exec: Vapor.TestHelpers.worker_exec(:host))
    {:ok, n} = Vapor.Runtime.Native.run(w, comp, %{}, isa: Vapor.Runtime.Substrates.host_isa(), mode: :native)
    assert o.outputs.rank == n.outputs.rank
    prog = o.outputs.rank |> Vapor.Tensor.to_floats() |> Enum.take(200)
    host = G.pagerank(g)
    assert Enum.zip_with(prog, host, &abs(&1 - &2)) |> Enum.max() < 1.0e-5
    top = fn xs -> xs |> Enum.with_index() |> Enum.sort_by(fn {v, i} -> {-v, i} end) |> Enum.take(10) |> Enum.map(&elem(&1, 1)) end
    assert top.(prog) == top.(host)
  end

  @tag :networkx
  test "networkx computes the same PageRank, clustering, assortativity and betweenness" do
    g = G.watts_strogatz(120, 6, 0.2, 9)
    edges = Enum.map_join(G.edges(g), "\n", fn {a, b} -> "#{a} #{b}" end)

    out =
      Vapor.TestHelpers.py!("""
      import sys, json, networkx as nx
      g = nx.Graph(); g.add_nodes_from(range(120))
      g.add_edges_from(tuple(map(int, l.split())) for l in sys.stdin.read().splitlines() if l.strip())
      pr = nx.pagerank(g, alpha=0.85, tol=1e-14, max_iter=1000)
      bc = nx.betweenness_centrality(g, normalized=True)
      print(json.dumps({"pr": [pr[i] for i in range(120)], "cc": [nx.clustering(g, i) for i in range(120)],
                        "as": nx.degree_assortativity_coefficient(g), "bc": [bc[i] for i in range(120)]}))
      """, [], edges)

    {:ok, ref} = Vapor.JSON.decode(out)
    close = fn a, b, tol -> Enum.zip_with(a, b, &abs(&1 - &2)) |> Enum.max() < tol end
    assert close.(G.pagerank(g), ref["pr"], 1.0e-9)
    assert close.(G.clustering(g), ref["cc"], 1.0e-12)
    assert abs(G.assortativity(g) - ref["as"]) < 1.0e-9
    assert close.(G.betweenness(g), ref["bc"], 1.0e-9)
  end
end
