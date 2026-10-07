defmodule Vapor.ShardClusterTest do
  @moduledoc """
  Tensor parallelism across BEAM nodes (`Vapor.Shard.Cluster`), on real
  peer nodes (`:peer`, Distributed Erlang): the bits of one worker, any
  number of nodes; a lost node's shards placed again with no change in the
  answer; replicas compared bit for bit, so a faulty node is caught; a
  corrupted shard refused at the door.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Shard.{Cluster, Host}
  alias Vapor.Runtime.{Native, Substrates, Worker}
  import Vapor.TestHelpers

  @moduletag :native
  @moduletag timeout: 300_000

  setup_all do
    unless Node.alive?() do
      System.cmd("epmd", ["-daemon"])
      {:ok, _} = Node.start(:"vapor_shard_test@127.0.0.1", :longnames)
    end

    paths = Enum.flat_map(:code.get_path(), &[~c"-pa", &1])

    peers =
      for i <- 1..3 do
        {:ok, pid, node} = :peer.start(%{name: :"vapor_peer#{i}_#{System.unique_integer([:positive])}", host: ~c"127.0.0.1", longnames: true, args: [~c"-connect_all", ~c"false" | paths]})
        {:ok, _} = :erpc.call(node, Application, :ensure_all_started, [:vapor])
        {pid, node}
      end

    on_exit(fn -> for {pid, _} <- peers, do: (try do :peer.stop(pid) catch _, _ -> :ok end) end)
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    {:ok, peers: peers, worker: w}
  end

  defp one(wk, x, w) do
    {:ok, c} = Vapor.Compile.Lower.lower(Program.new(y: T.linear(T.input(:x, :f32, x.shape), T.const(w))))
    {:ok, r} = Native.run(wk, c, %{x: x}, isa: Substrates.host_isa(), mode: :native)
    r.outputs.y
  end

  test "a matrix split over 1, 2 and 3 nodes = one worker, bit for bit; an MLP too", %{peers: peers, worker: wk} do
    x = Tensor.random(:f32, [5, 256], 1)
    w = Tensor.random(:f32, [100, 256], 2, scale: 0.1)
    ref = one(wk, x, w)

    for n <- 1..3 do
      {:ok, c} = Cluster.start(Enum.map(Enum.take(peers, n), &elem(&1, 1)))
      {:ok, c} = Cluster.load(c, :w, w)
      assert {:ok, y, _c, %{replaced: []}} = Cluster.linear(c, :w, x)
      assert y == ref, "#{n} nodes"
    end

    # SwiGLU MLP: gate/up column-parallel, intermediate all-gathered, down column-parallel
    g = Tensor.random(:f32, [96, 256], 3, scale: 0.1)
    u = Tensor.random(:f32, [96, 256], 4, scale: 0.1)
    d = Tensor.random(:f32, [256, 96], 5, scale: 0.1)
    {:ok, local} = Vapor.Shard.mlp([wk], x, {g, u, d})
    {:ok, c} = Cluster.start(Enum.map(peers, &elem(&1, 1)))
    {:ok, c} = Cluster.load(c, {:mlp, :gate}, g)
    {:ok, c} = Cluster.load(c, {:mlp, :up}, u)
    {:ok, c} = Cluster.load(c, {:mlp, :down}, d)
    assert {:ok, y, _, _} = Cluster.mlp(c, :mlp, x)
    assert y == local
  end

  test "a node lost mid-run: its shards move to the survivors and the answer keeps its bits", %{worker: wk} do
    paths = Enum.flat_map(:code.get_path(), &[~c"-pa", &1])
    nodes =
      for i <- 1..2 do
        {:ok, pid, node} = :peer.start(%{name: :"vapor_doomed#{i}_#{System.unique_integer([:positive])}", host: ~c"127.0.0.1", longnames: true, args: [~c"-connect_all", ~c"false" | paths]})
        {:ok, _} = :erpc.call(node, Application, :ensure_all_started, [:vapor])
        {pid, node}
      end

    x = Tensor.random(:f32, [3, 128], 11)
    w = Tensor.random(:f32, [70, 128], 12, scale: 0.1)
    ref = one(wk, x, w)
    {:ok, c} = Cluster.start([Node.self() | Enum.map(nodes, &elem(&1, 1))])
    {:ok, c} = Cluster.load(c, :w, w)
    {:ok, y0, c, _} = Cluster.linear(c, :w, x)
    assert y0 == ref

    {doomed, dnode} = hd(nodes)
    :peer.stop(doomed)
    assert {:ok, y1, c2, rep} = Cluster.linear(c, :w, x)
    assert y1 == ref
    assert [{:w, _, [_]}] = rep.replaced
    refute dnode in Enum.map(c2.hosts, &elem(&1, 0))
    :peer.stop(elem(List.last(nodes), 0))
  end

  test "replicas compared bit for bit: a node that corrupts one bit is caught, not averaged in", %{peers: peers} do
    x = Tensor.random(:f32, [2, 64], 21)
    w = Tensor.random(:f32, [30, 64], 22, scale: 0.1)
    {:ok, c} = Cluster.start(Enum.map(peers, &elem(&1, 1)), replicas: 2)
    {:ok, c} = Cluster.load(c, :w, w)
    assert {:ok, _y, _, %{checked: checked}} = Cluster.linear(c, :w, x)
    assert length(checked) == 3

    {bad_node, bad_host} = Enum.at(c.hosts, 1)
    :ok = Host.corrupt(bad_host, true)
    assert {:error, {:disagreement, _shard, nodes}} = Cluster.linear(c, :w, x)
    assert bad_node in nodes
    :ok = Host.corrupt(bad_host, false)
  end

  test "a shard whose bytes do not match its digest is refused", %{peers: [{_, node} | _]} do
    {:ok, h} = Host.start(node)
    w = Tensor.random(:f32, [8, 32], 1)
    assert {:error, :digest_mismatch} = Host.load(h, :w, w, :crypto.hash(:sha256, "something else"))
    assert :ok = Host.load(h, :w, w, :crypto.hash(:sha256, w.data))
  end
end
