defmodule Vapor.Shard.Host do
  @moduledoc """
  One node's share of a sharded model (`Vapor.Shard.Cluster`): a native
  worker on that node and the weight shards placed there, **resident** —
  compiled once with the shard as a constant, so a call carries only the
  activations in and the shard's columns out.

  A shard arrives with its SHA-256; the host recomputes it and refuses a
  shard that does not match (a corrupted transfer is not computed on).
  """
  use GenServer
  alias Vapor.{Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Runtime.{Native, Substrates, Worker}

  @max_rows 256

  @doc "Start a host on `node` (the local node when `node() == node`). `{:ok, pid}`."
  def start(node, opts \\ []) do
    if node == Node.self(),
      do: GenServer.start(__MODULE__, opts),
      else: :erpc.call(node, GenServer, :start, [__MODULE__, opts], 30_000)
  end

  @doc "Place shard `id` (`W : f32[n, k]`) on the host, checked against `sha256`."
  def load(host, id, %Tensor{} = w, sha256), do: GenServer.call(host, {:load, id, w, sha256}, 120_000)

  @doc "`x · Wᵀ` for the host's shard `id`: `{:ok, y}`."
  def linear(host, id, %Tensor{} = x), do: GenServer.call(host, {:linear, id, x}, 120_000)

  @doc false
  # test hook: make the host corrupt one bit of every result (a faulty or lying node)
  def corrupt(host, on?), do: GenServer.call(host, {:corrupt, on?})

  @impl true
  def init(opts) do
    exec = Keyword.get(opts, :exec) || [Substrates.binary("vapor-worker", "native")]
    {:ok, w} = Worker.start_link(exec: exec, threads: Keyword.get(opts, :threads, 1))
    {:ok, %{worker: w, shards: %{}, corrupt: false}}
  end

  @impl true
  def handle_call({:load, id, w, sha}, _from, s) do
    if :crypto.hash(:sha256, w.data) != sha do
      {:reply, {:error, :digest_mismatch}, s}
    else
      [_n, k] = w.shape
      p = Program.new(y: T.linear(T.input(:x, :f32, [T.dyn(:b, @max_rows), k]), T.const(w)))

      case Vapor.Compile.Lower.lower(p) do
        {:ok, c} -> {:reply, :ok, put_in(s.shards[id], c)}
        err -> {:reply, err, s}
      end
    end
  end

  def handle_call({:linear, id, x}, _from, s) do
    case s.shards do
      %{^id => c} ->
        case Native.run(s.worker, c, %{x: x}, isa: Substrates.host_isa(), mode: :native) do
          {:ok, r} -> {:reply, {:ok, if(s.corrupt, do: flip(r.outputs.y), else: r.outputs.y)}, s}
          err -> {:reply, err, s}
        end

      _ ->
        {:reply, {:error, {:no_shard, id}}, s}
    end
  end

  def handle_call({:corrupt, on?}, _from, s), do: {:reply, :ok, %{s | corrupt: on?}}

  defp flip(%Tensor{data: <<a::binary-size(4), x, rest::binary>>} = t), do: %{t | data: <<a::binary, Bitwise.bxor(x, 1), rest::binary>>}
end

defmodule Vapor.Shard.Cluster do
  @moduledoc """
  **Tensor parallelism across BEAM nodes, with the bits of one machine.**
  `Vapor.Shard` proved the exact form (column-parallel + all-gather) across
  worker processes; this takes it across nodes of an Erlang cluster
  (Distributed Erlang: authenticated by the cluster cookie — and by TLS
  with `-proto_dist inet_tls`).

  Shards are placed once (`load/4`: each node receives its rows of `W`
  with their SHA-256 and keeps them resident); a call (`linear/3`) sends
  the activations to every node in parallel and concatenates the columns.
  Because every output element is computed whole, by the same
  instructions, on exactly one node, the result is **bit-identical** to
  the one-worker product — whatever the number of nodes, the split, or
  where each shard lives.

  Two consequences of exactness that an inexact runtime cannot have:

    * **failover without drift** — when a node is lost, its shards are
      placed again on the survivors and the call is redone: the answer is
      the same bits, so nothing downstream can tell;
    * **replication as verification** — with `replicas: 2` every shard is
      computed on two different nodes and the results are compared *bit
      for bit*: a faulty (or lying) node is detected, not averaged in.
      Floating point that is not reproducible could only compare within a
      tolerance, which a careful adversary stays inside.
  """
  alias Vapor.Shard.Host
  alias Vapor.Tensor

  defstruct hosts: [], shards: %{}, replicas: 1

  @doc "Start a host on each node. Options: `replicas` (1 or 2), `threads`. `{:ok, cluster}`."
  def start(nodes, opts \\ []) when nodes != [] do
    hosts = for n <- nodes, do: {n, elem(Host.start(n, Keyword.take(opts, [:threads])), 1)}
    replicas = min(Keyword.get(opts, :replicas, 1), length(nodes))
    {:ok, %__MODULE__{hosts: hosts, replicas: replicas}}
  end

  @doc """
  Place matrix `id` (`W : f32[n, k]`): its rows split into one shard per
  node, each shard on `replicas` distinct nodes. The cluster keeps the
  shards, to place them again if a node is lost.
  """
  def load(%__MODULE__{} = c, id, %Tensor{} = w) do
    shards = split_rows(w, length(c.hosts))
    nh = length(c.hosts)

    placed =
      shards
      |> Enum.with_index()
      |> Enum.map(fn {s, i} ->
        homes = for r <- 0..(c.replicas - 1), do: Enum.at(c.hosts, rem(i + r, nh))
        sha = :crypto.hash(:sha256, s.data)
        for {_n, h} <- homes, do: :ok = Host.load(h, {id, i}, s, sha)
        %{tensor: s, sha: sha, homes: homes}
      end)

    {:ok, %{c | shards: Map.put(c.shards, id, placed)}}
  end

  @doc """
  `x · Wᵀ` for matrix `id`, across the cluster: `{:ok, y, cluster, report}`
  — `report.replaced` lists shards moved off lost nodes, `report.checked`
  the shards whose replicas agreed bit for bit. A replica disagreement is
  `{:error, {:disagreement, shard, nodes}}`.
  """
  def linear(%__MODULE__{} = c, id, %Tensor{shape: [b, _]} = x) do
    shards = Map.fetch!(c.shards, id)

    results =
      shards
      |> Enum.with_index()
      |> Enum.map(fn {sh, i} -> Task.async(fn -> {i, run_shard(sh, {id, i}, x)} end) end)
      |> Enum.map(&Task.await(&1, :infinity))

    case Enum.find(results, fn {_, r} -> match?({:error, _}, r) end) do
      {i, {:error, {:disagreement, nodes}}} ->
        {:error, {:disagreement, i, nodes}}

      {_i, {:error, _}} ->
        # a node was lost: place its shards on the survivors, then redo the call
        case drop_dead(c, id) do
          {:ok, c2, moved} when moved != [] ->
            with {:ok, y, c3, rep} <- linear(c2, id, x), do: {:ok, y, c3, %{rep | replaced: moved ++ rep.replaced}}

          {:ok, _, []} -> {:error, :shard_failed}
          err -> err
        end

      nil ->
        ys = Enum.map(results, fn {_, {:ok, y, _}} -> y end)
        checked = for {i, {:ok, _, n}} <- results, n > 1, do: i
        {:ok, concat_columns(ys, b), c, %{replaced: [], checked: checked}}
    end
  end

  @doc """
  A SwiGLU MLP across the cluster, exactly (`Vapor.Shard.mlp/4`'s form):
  gate and up column-parallel, the intermediate all-gathered (it travels
  back to the coordinator and out again), down column-parallel.
  Matrices must have been loaded as `{name, :gate}`, `{name, :up}`, `{name, :down}`.
  """
  def mlp(%__MODULE__{} = c, name, %Tensor{shape: [b, _]} = x) do
    with {:ok, g, c, r1} <- linear(c, {name, :gate}, x),
         {:ok, u, c, r2} <- linear(c, {name, :up}, x) do
      [_, n] = g.shape
      p = Vapor.Program.new(y: Vapor.Algebra.Term.mul(Vapor.Algebra.Term.silu(Vapor.Algebra.Term.input(:g, :f32, [b, n])),
                                                      Vapor.Algebra.Term.input(:u, :f32, [b, n])))
      inter = Vapor.Runtime.Oracle.eval_program(p, %{g: g, u: u}).y

      with {:ok, y, c, r3} <- linear(c, {name, :down}, inter),
           do: {:ok, y, c, %{replaced: r1.replaced ++ r2.replaced ++ r3.replaced, checked: r1.checked ++ r2.checked ++ r3.checked}}
    end
  end

  # every replica of a shard computes it; results must agree bit for bit
  defp run_shard(%{homes: homes}, key, x) do
    rs = Enum.map(homes, fn {n, h} -> {n, safe(fn -> Host.linear(h, key, x) end)} end)

    case Enum.split_with(rs, fn {_, r} -> match?({:ok, _}, r) end) do
      {[], _} -> {:error, :lost}
      {_ok, [_ | _]} -> {:error, :lost}
      {oks, []} ->
        case oks |> Enum.map(fn {_, {:ok, y}} -> y.data end) |> Enum.uniq() do
          [_one] -> {:ok, elem(elem(hd(oks), 1), 1), length(oks)}
          _ -> {:error, {:disagreement, Enum.map(oks, &elem(&1, 0))}}
        end
    end
  end

  defp safe(f) do
    f.()
  catch
    :exit, why -> {:error, {:exit, why}}
  end

  # hosts whose process is gone (its node down, or the host crashed) leave;
  # every shard that lived there is placed again on a survivor
  defp drop_dead(c, _id) do
    alive = Enum.filter(c.hosts, fn {_n, h} -> alive?(h) end)

    if alive == [] do
      {:error, :no_nodes}
    else
      {shards, moved} =
        Enum.map_reduce(c.shards, [], fn {mid, list}, moved ->
          {list2, moved} =
            list
            |> Enum.with_index()
            |> Enum.map_reduce(moved, fn {sh, i}, moved ->
              live = Enum.filter(sh.homes, &(&1 in alive))
              lost = length(sh.homes) - length(live)

              if lost == 0 do
                {sh, moved}
              else
                spare = (alive -- live) |> Enum.take(lost)
                homes = live ++ spare
                homes = if homes == [], do: [hd(alive)], else: homes
                for {_n, h} <- homes -- live, do: :ok = Host.load(h, {mid, i}, sh.tensor, sh.sha)
                {%{sh | homes: homes}, [{mid, i, Enum.map(homes -- live, &elem(&1, 0))} | moved]}
              end
            end)

          {{mid, list2}, moved}
        end)

      {:ok, %{c | hosts: alive, shards: Map.new(shards)}, Enum.reverse(moved)}
    end
  end

  defp alive?(pid) when node(pid) == node(), do: Process.alive?(pid)

  defp alive?(pid) do
    :erpc.call(node(pid), Process, :alive?, [pid], 5_000)
  catch
    _, _ -> false
  end

  defp split_rows(%Tensor{shape: [n, k]} = w, parts) do
    w = Tensor.widen(w)
    sizes = for i <- 0..(parts - 1), do: div(n, parts) + if(i < rem(n, parts), do: 1, else: 0)

    {shards, _} =
      Enum.map_reduce(Enum.reject(sizes, &(&1 == 0)), 0, fn s, at ->
        {Tensor.new(:f32, [s, k], binary_part(w.data, at * k * 4, s * k * 4)), at + s}
      end)

    shards
  end

  defp concat_columns(parts, b) do
    rows =
      for r <- 0..(b - 1), into: <<>> do
        for %Tensor{shape: [_, n]} = t <- parts, into: <<>>, do: binary_part(t.data, r * n * 4, n * 4)
      end

    n = parts |> Enum.map(fn %Tensor{shape: [_, n]} -> n end) |> Enum.sum()
    Tensor.new(:f32, [b, n], rows)
  end
end
