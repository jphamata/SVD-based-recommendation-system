defmodule Vapor.ClusterTest do
  @moduledoc """
  Orchestration on real peer nodes (`Vapor.Cluster`): results equal to the
  oracle wherever they run, memoized cluster-wide, audited by redundant
  execution (a node flipping one bit is caught and quarantined — and,
  without the audit, its wrong answers would have gone through), failover
  and hedging that change no bit, re-admission only by measurement, and
  training across nodes with the bits of one machine.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Cluster, Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Runtime.Native

  @moduletag :native
  @moduletag timeout: 600_000

  setup_all do
    unless Node.alive?() do
      System.cmd("epmd", ["-daemon"])
      {:ok, _} = Node.start(:"vapor_cluster_test@127.0.0.1", :longnames)
    end

    paths = Enum.flat_map(:code.get_path(), &[~c"-pa", &1])

    peers = for _ <- 1..3, do: start_peer(paths)
    on_exit(fn -> for {pid, _} <- peers, do: stop_peer(pid) end)
    {:ok, peers: peers, nodes: Enum.map(peers, &elem(&1, 1)), paths: paths}
  end

  defp start_peer(paths) do
    {:ok, pid, node} = :peer.start(%{name: :"vapor_cl_#{System.unique_integer([:positive])}", host: ~c"127.0.0.1", longnames: true, args: [~c"-connect_all", ~c"false" | paths]})
    {:ok, _} = :erpc.call(node, Application, :ensure_all_started, [:vapor])
    {pid, node}
  end

  defp stop_peer(pid), do: (try do :peer.stop(pid) catch _, _ -> :ok end)

  defp prog(seed) do
    w = Tensor.random(:f32, [48, 64], seed, scale: 0.2)
    p = Program.new(y: T.silu(T.linear(T.input(:x, :f32, [4, 64]), T.const(w))))
    {:ok, c} = Lower.lower(p)
    c
  end

  defp env(seed), do: %{x: Tensor.random(:f32, [4, 64], 100 + seed)}
  defp oracle(c, e), do: elem(Native.run_oracle(c, e), 1).outputs
  defp node_pid(cl, n), do: :sys.get_state(cl).nodes[n].pid

  test "every node gives the oracle's bits; a repeated job is a cache hit", %{nodes: nodes} do
    {:ok, cl} = Cluster.start_link(nodes: nodes, audit_rate: 0.0)
    jobs = for i <- 1..9, do: {prog(rem(i, 3)), env(i)}
    results = Cluster.map(cl, jobs)

    for {{c, e}, {:ok, outs, meta}} <- Enum.zip(jobs, results) do
      assert outs == oracle(c, e)
      assert meta.node in nodes and not meta.cached
    end

    # spread over the nodes, the programs remembered by content
    used = results |> Enum.map(fn {:ok, _, m} -> m.node end) |> Enum.uniq()
    assert length(used) >= 2
    {c, e} = hd(jobs)
    assert {:ok, outs, %{cached: true}} = Cluster.run(cl, c, e)
    assert outs == oracle(c, e)
    assert Cluster.status(cl).stats.cache_hits == 1
  end

  test "redundant execution catches a node that flips one bit; without it the error goes through", %{nodes: [a, b, c]} do
    # control: no audit, one liar among three — some answers are wrong
    {:ok, open} = Cluster.start_link(nodes: [a, b, c], audit_rate: 0.0)
    Vapor.Cluster.Node.corrupt(node_pid(open, b), true)
    jobs = for i <- 1..12, do: {prog(i), env(i)}
    wrong = for {{p, e}, {:ok, outs, _}} <- Enum.zip(jobs, Cluster.map(open, jobs)), outs != oracle(p, e), do: 1
    assert length(wrong) > 0

    # audited: every answer right, the liar quarantined with the evidence
    {:ok, cl} = Cluster.start_link(nodes: [a, b, c], audit_rate: 1.0)
    Vapor.Cluster.Node.corrupt(node_pid(cl, b), true)
    results = Cluster.map(cl, jobs, concurrency: 1)
    for {{p, e}, {:ok, outs, _}} <- Enum.zip(jobs, results), do: assert(outs == oracle(p, e))
    st = Cluster.status(cl)
    assert st.nodes[b].state == :quarantined
    assert st.stats.disagreements >= 1
    assert [%{agree: false, wrong: [^b]} | _] = Enum.filter(st.audits, &(not &1.agree))
    # after quarantine the liar receives nothing
    {:ok, _, meta} = Cluster.run(cl, prog(99), env(99))
    refute meta.node == b or meta[:with] == b
    # and nothing it answered unaudited is served from the cache any more
    refute Enum.any?(:sys.get_state(cl).cache, fn {_, {_, node}} -> node == b end)

    # back only by measurement: refused while it lies, admitted once honest
    assert {:refused, _} = Cluster.readmit(cl, b)
    Vapor.Cluster.Node.corrupt(node_pid(cl, b), false)
    assert {:ok, %{verdict: :canonical}} = Cluster.readmit(cl, b)
    assert Cluster.status(cl).nodes[b].state == :healthy
  end

  test "the audited sample is a keyed hash: recomputable by anyone with the salt", %{nodes: nodes} do
    salt = "fixed salt"
    {:ok, cl} = Cluster.start_link(nodes: nodes, audit_rate: 0.5, salt: salt)
    metas = for i <- 1..16, do: (({:ok, _, m} = Cluster.run(cl, prog(i), env(i))); m)
    assert Enum.all?(metas, fn m -> m.audited == Cluster.audited?(m.key, salt, 0.5) end)
    n = Enum.count(metas, & &1.audited)
    assert n > 2 and n < 14
  end

  test "a node lost mid-run: its jobs run elsewhere with the same bits", %{nodes: [a, b | _], paths: paths} do
    # a node of its own to lose, so the shared peers stay up for the other tests
    {victim_pid, victim} = start_peer(paths)
    {:ok, cl} = Cluster.start_link(nodes: [a, b, victim], audit_rate: 0.0)
    assert Cluster.status(cl).nodes[victim].state == :healthy
    stop_peer(victim_pid)
    Process.sleep(200)
    assert Cluster.status(cl).nodes[victim].state == :down
    jobs = for i <- 1..6, do: {prog(i + 20), env(i + 20)}
    for {{p, e}, {:ok, outs, m}} <- Enum.zip(jobs, Cluster.map(cl, jobs)), do: (assert outs == oracle(p, e); refute m.node == victim)

    # a cluster started with the dead node lists it as down instead of failing
    {:ok, cl2} = Cluster.start_link(nodes: [a, victim], audit_rate: 0.0)
    assert %{state: :down, why: {:unreachable, _}} = Cluster.status(cl2).nodes[victim]
    assert {:ok, _, %{node: ^a}} = Cluster.run(cl2, prog(5), env(5))
  end

  test "hedging: a straggler is overtaken, and the answer is the same", %{nodes: [n1, n2 | _]} do
    {:ok, cl} = Cluster.start_link(nodes: [n1, n2], audit_rate: 0.0, hedge_ms: 50)
    # a fresh cluster places a new program on the first node by name: make that one the straggler
    [slow, fast] = Enum.sort_by([n1, n2], &to_string/1)
    Vapor.Cluster.Node.delay(node_pid(cl, slow), 2_000)
    Vapor.Cluster.Node.delay(node_pid(cl, fast), 0)
    {p, e} = {prog(7), env(7)}
    t0 = System.monotonic_time(:millisecond)
    {:ok, outs, meta} = Cluster.run(cl, p, e)
    assert System.monotonic_time(:millisecond) - t0 < 1_500
    assert outs == oracle(p, e) and meta.node == fast
    assert Cluster.status(cl).stats.hedged >= 1

    # control: without hedging the same job waits for the straggler
    {:ok, plain} = Cluster.start_link(nodes: [n1, n2], audit_rate: 0.0)
    Vapor.Cluster.Node.delay(node_pid(plain, slow), 600)
    t0 = System.monotonic_time(:millisecond)
    {:ok, ^outs, %{node: ^slow}} = Cluster.run(plain, p, e)
    assert System.monotonic_time(:millisecond) - t0 >= 600
  end

  test "training across nodes has the bits of one machine", %{nodes: [a, b | _]} do
    alias Vapor.Train.LM
    c = LM.new(d: 32, layers: 1, heads: 2, ff: 64, seq: 16, seqs: 1)
    corpus = String.duplicate("o worker executa, a eclusa admite; the worker runs, the airlock admits. ", 8)
    run = LM.start(c, corpus, micro: 4, chunk: 1, steps: 10, seed: 3)
    {:ok, cl} = Cluster.start_link(nodes: [a, b], workers: 1, audit_rate: 0.0)
    remote = Cluster.workers(cl)
    assert length(remote) == 2 and Enum.all?(remote, &(node(&1) in [a, b]))
    {:ok, local} = Vapor.Runtime.Worker.start_link(exec: Vapor.TestHelpers.worker_exec(:host))
    assert LM.digest(LM.train(run, corpus, remote, 2)) == LM.digest(LM.train(run, corpus, [local], 2))
  end
end
