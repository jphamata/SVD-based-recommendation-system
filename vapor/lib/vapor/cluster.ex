defmodule Vapor.Cluster do
  @moduledoc """
  **Orchestration of a vapor cluster, built on the one property clusters
  never have: every node computes the same bits.**

  In a conventional cluster, two machines running the same job return
  slightly different numbers (thread counts, reduction orders, libraries),
  so results can be neither compared nor reused across machines. vapor's
  canonical programs give the same bits on every substrate, and that turns
  four hard problems of distributed computing into equalities:

  * **cluster-wide memoization** — a result is a function of
    `(program, inputs)`: its content key (SHA-256 of both) finds it in the
    cache whichever node computed it, whenever;
  * **redundant execution as an audit** — a sampled fraction of jobs runs
    on a second node and the two results must be *equal*, not close: a
    node that corrupts one bit is caught the first time it is audited, the
    oracle settles who is right, and the liar is quarantined. The sample is
    a keyed hash of the job (`audit_rate`, `salt`), so which jobs were
    audited is itself checkable afterwards;
  * **failover without drift** — a lost node's jobs are re-run elsewhere
    with identical results;
  * **hedging against stragglers** — after `hedge_ms` a duplicate starts on
    another node; the first answer is returned, and when the second
    arrives it is compared for free.

  A quarantined node comes back only by measurement: `readmit/2` runs the
  substrate airlock's battery (`Vapor.Substrate`) on it.

  Nodes are BEAM nodes (Distributed Erlang); each runs a
  `Vapor.Cluster.Node` with its own native workers, which also serve
  training: `workers/1` gives their pids to `Vapor.Train.LM.train/5`, whose
  bits do not depend on how many there are or where they live.
  """
  use GenServer
  alias Vapor.Cluster.Node, as: CNode
  alias Vapor.Compiled

  defmodule Job do
    @moduledoc false
    defstruct [:id, :key, :program, :env, :opts]
  end

  # ------------------------------------------------------------------ API --

  @doc """
  Start a coordinator over `nodes` (atoms; `node()` is allowed). Options:
  `workers` per node (1), `audit_rate` (0.1), `salt` (a binary; random),
  `hedge_ms` (nil: no hedging), `cache` (entries kept, 1024), `name`.
  """
  def start_link(opts), do: GenServer.start_link(__MODULE__, opts, Keyword.take(opts, [:name]))

  @doc """
  Run a compiled program on the cluster: `{:ok, outputs, meta}` where meta
  says `node`, `cached`, `audited` (and with whom), `attempts`. A canonical
  program is required for auditing and caching to mean anything; a `:fast`
  one is run without either.
  """
  def run(c, %Compiled{} = comp, env, opts \\ []), do: GenServer.call(c, {:run, comp, env, opts}, :infinity)

  @doc "Run many jobs (`[{compiled, env}]`) concurrently; results in order."
  def map(c, jobs, opts \\ []) do
    jobs
    |> Task.async_stream(fn {comp, env} -> run(c, comp, env, opts) end, timeout: :infinity, max_concurrency: Keyword.get(opts, :concurrency, 16))
    |> Enum.map(fn {:ok, r} -> r end)
  end

  @doc "Nodes, health, load, quarantine, counters, the audit log."
  def status(c), do: GenServer.call(c, :status)

  @doc "The native workers of every healthy node (for `Vapor.Train.LM.train/5`)."
  def workers(c), do: GenServer.call(c, :workers)

  @doc "Re-admit a quarantined node if it passes the substrate airlock on its own host."
  def readmit(c, node), do: GenServer.call(c, {:readmit, node}, :infinity)

  @doc "Add a node to a running cluster."
  def join(c, node), do: GenServer.call(c, {:join, node}, 60_000)

  # -------------------------------------------------------------- server --

  @impl true
  def init(opts) do
    :ok = :net_kernel.monitor_nodes(true)
    per = Keyword.get(opts, :workers, 1)

    nodes =
      for n <- Keyword.fetch!(opts, :nodes), into: %{} do
        # a node that cannot be reached at start is listed as down, with the
        # reason, rather than refusing the whole cluster
        case (try do CNode.start(n, workers: per) catch k, why -> {:error, {k, why}} end) do
          {:ok, pid} ->
            Process.monitor(pid)
            {n, %{pid: pid, state: :healthy, inflight: 0, done: 0, programs: MapSet.new(), why: nil}}

          {:error, why} ->
            {n, %{pid: nil, state: :down, inflight: 0, done: 0, programs: MapSet.new(), why: {:unreachable, why}}}
        end
      end

    {:ok,
     %{nodes: nodes, per: per, audit_rate: Keyword.get(opts, :audit_rate, 0.1), salt: Keyword.get_lazy(opts, :salt, fn -> :crypto.strong_rand_bytes(16) end),
       hedge_ms: Keyword.get(opts, :hedge_ms), cache: %{}, order: :queue.new(), cap: Keyword.get(opts, :cache, 1024),
       audits: [], stats: %{jobs: 0, cache_hits: 0, audited: 0, disagreements: 0, retries: 0, hedged: 0}, digests: %{}}}
  end

  @impl true
  def handle_call({:run, comp, env, opts}, from, s) do
    {pkey, s} = program_key(comp, s)
    key = job_key(pkey, env, opts)
    s = update_in(s.stats.jobs, &(&1 + 1))

    case Map.fetch(s.cache, key) do
      {:ok, {outs, node}} when comp.policy == :canonical ->
        {:reply, {:ok, outs, %{node: node, cached: true, audited: false, attempts: 0, key: key}}, update_in(s.stats.cache_hits, &(&1 + 1))}

      _ ->
        audited = comp.policy == :canonical and audited?(key, s)
        job = %Job{id: make_ref(), key: key, program: {pkey, comp}, env: env, opts: opts}
        coordinator = self()
        candidates = placement(s, pkey)
        hedge = s.hedge_ms

        Task.start(fn ->
          reply = execute(job, candidates, audited, hedge, coordinator)
          send(coordinator, {:job_done, from, key, comp.policy, reply})
        end)

        # optimistic load accounting for placement
        s = Enum.reduce(Enum.take(candidates, if(audited, do: 2, else: 1)), s, fn {n, _}, s -> update_in(s.nodes[n].inflight, &(&1 + 1)) end)
        s = Enum.reduce(Enum.take(candidates, 1), s, fn {n, _}, s -> update_in(s.nodes[n].programs, &MapSet.put(&1, pkey)) end)
        {:noreply, s}
    end
  end

  def handle_call(:status, _from, s) do
    nodes = for {n, i} <- s.nodes, into: %{}, do: {n, Map.drop(i, [:pid, :programs]) |> Map.put(:programs, MapSet.size(i.programs))}
    {:reply, %{nodes: nodes, stats: s.stats, audits: Enum.reverse(s.audits), audit_rate: s.audit_rate, cached: map_size(s.cache)}, s}
  end

  def handle_call(:workers, _from, s) do
    ws = for {_n, %{state: :healthy, pid: p}} <- s.nodes, w <- safe(fn -> CNode.workers(p) end, []), do: w
    {:reply, ws, s}
  end

  def handle_call({:readmit, node}, _from, s) do
    case s.nodes[node] do
      nil ->
        {:reply, {:error, :unknown_node}, s}

      %{pid: p} ->
        case safe(fn -> CNode.admit(p) end, {:error, :unreachable}) do
          {:ok, %{verdict: :canonical} = a} -> {:reply, {:ok, a}, put_in(s.nodes[node].state, :healthy) |> put_in([:nodes, node, :why], nil)}
          {:ok, a} -> {:reply, {:refused, a}, s}
          err -> {:reply, err, s}
        end
    end
  end

  def handle_call({:join, node}, _from, s) do
    case (try do CNode.start(node, workers: s.per) catch k, why -> {:error, {k, why}} end) do
      {:ok, pid} ->
        Process.monitor(pid)
        {:reply, :ok, put_in(s.nodes[node], %{pid: pid, state: :healthy, inflight: 0, done: 0, programs: MapSet.new(), why: nil})}

      err ->
        {:reply, err, s}
    end
  end

  @impl true
  def handle_info({:job_done, from, key, policy, reply}, s) do
    {result, events} = reply
    s = Enum.reduce(events, s, &apply_event/2)

    s =
      case result do
        {:ok, outs, meta} when policy == :canonical -> cache_put(s, key, {outs, meta.node})
        _ -> s
      end

    GenServer.reply(from, result)
    {:noreply, s}
  end

  def handle_info({:nodedown, n}, s) do
    {:noreply, if(Map.has_key?(s.nodes, n), do: s |> put_in([:nodes, n, :state], :down) |> put_in([:nodes, n, :why], :nodedown), else: s)}
  end

  def handle_info({:DOWN, _, :process, pid, why}, s) do
    case Enum.find(s.nodes, fn {_, i} -> i.pid == pid end) do
      {n, _} -> {:noreply, s |> put_in([:nodes, n, :state], :down) |> put_in([:nodes, n, :why], {:exited, why})}
      nil -> {:noreply, s}
    end
  end

  def handle_info(_, s), do: {:noreply, s}

  # ------------------------------------------------------------- policy --

  defp apply_event({:done, n}, s), do: if(s.nodes[n], do: update_in(s.nodes[n], &%{&1 | inflight: max(&1.inflight - 1, 0), done: &1.done + 1}), else: s)
  defp apply_event({:failed, n, why}, s), do: if(s.nodes[n], do: s |> update_in([:nodes, n], &%{&1 | inflight: max(&1.inflight - 1, 0), state: :down, why: why}) |> update_in([:stats, :retries], &(&1 + 1)), else: s)
  # a node caught lying: its unaudited answers leave the cache with it
  defp apply_event({:quarantine, n, why}, s),
    do: if(s.nodes[n], do: s |> put_in([:nodes, n, :state], :quarantined) |> put_in([:nodes, n, :why], why) |> evict(n), else: s)

  defp apply_event({:audit, entry}, s), do: s |> update_in([:stats, :audited], &(&1 + 1)) |> Map.update!(:audits, &[entry | &1]) |> then(fn s -> if entry.agree, do: s, else: update_in(s.stats.disagreements, &(&1 + 1)) end)
  defp apply_event(:hedged, s), do: update_in(s.stats.hedged, &(&1 + 1))

  defp evict(s, n) do
    cache = Map.reject(s.cache, fn {_, {_, node}} -> node == n end)
    %{s | cache: cache, order: :queue.filter(&Map.has_key?(cache, &1), s.order)}
  end

  # healthy nodes, those already holding the program first, then the least loaded
  defp placement(s, pkey) do
    s.nodes
    |> Enum.filter(fn {_, i} -> i.state == :healthy end)
    |> Enum.sort_by(fn {n, i} -> {if(MapSet.member?(i.programs, pkey), do: 0, else: 1), i.inflight, to_string(n)} end)
    |> Enum.map(fn {n, i} -> {n, i.pid} end)
  end

  defp audited?(key, s), do: audited?(key, s.salt, s.audit_rate)

  @doc """
  Whether a job (its content key) is audited at `rate` under `salt`:
  `H(salt ‖ key) < rate · 2²⁵⁶`. Anyone holding the salt can recompute
  which jobs were checked — the sample cannot be steered after the fact.
  """
  def audited?(_key, _salt, rate) when rate <= 0, do: false
  def audited?(_key, _salt, rate) when rate >= 1, do: true
  def audited?(key, salt, rate), do: :binary.decode_unsigned(:crypto.hash(:sha256, [salt, key])) < trunc(rate * :math.pow(2, 256))

  defp cache_put(s, key, v) do
    if map_size(s.cache) >= s.cap do
      {{:value, old}, q} = :queue.out(s.order)
      %{s | cache: s.cache |> Map.delete(old) |> Map.put(key, v), order: :queue.in(key, q)}
    else
      %{s | cache: Map.put(s.cache, key, v), order: :queue.in(key, s.order)}
    end
  end

  # ---------------------------------------------------------- execution --
  # Runs in a task; returns {reply, events} for the coordinator to apply.

  defp execute(job, candidates, audited, hedge, _coord) do
    case candidates do
      [] -> {{:error, :no_healthy_node}, []}
      _ when audited and length(candidates) >= 2 -> audit(job, candidates)
      _ -> single(job, candidates, hedge, [], 1)
    end
  end

  # one node; on failure the next; optionally hedged after `hedge` ms
  defp single(_job, [], _hedge, events, _attempts), do: {{:error, :all_nodes_failed}, events}

  defp single(job, [{n, pid} | rest], hedge, events, attempts) do
    primary = Task.async(fn -> {n, call(pid, job)} end)

    {first, events} =
      case hedge && rest != [] && Task.yield(primary, hedge) do
        {:ok, r} -> {r, events}
        nil when is_integer(hedge) and rest != [] ->
          [{n2, pid2} | _] = rest
          backup = Task.async(fn -> {n2, call(pid2, job)} end)
          winner = wait_first([primary, backup])
          {winner, [:hedged | events]}
        _ -> {Task.await(primary, :infinity), events}
      end

    case first do
      {node, {:ok, outs}} -> {{:ok, outs, %{node: node, cached: false, audited: false, attempts: attempts, key: job.key}}, [{:done, node} | events]}
      {node, {:error, why}} -> single(job, Enum.reject(rest, &(elem(&1, 0) == node)), hedge, [{:failed, node, why} | events], attempts + 1)
    end
  end

  defp wait_first(tasks) do
    receive do
      {ref, {_n, {:ok, _}} = r} when is_reference(ref) ->
        if Enum.any?(tasks, &(&1.ref == ref)) do
          Process.demonitor(ref, [:flush])
          for t <- tasks, t.ref != ref, do: Task.shutdown(t, :brutal_kill)
          r
        else
          wait_first(tasks)
        end

      {ref, {_n, {:error, _}} = r} when is_reference(ref) ->
        rest = Enum.reject(tasks, &(&1.ref == ref))
        Process.demonitor(ref, [:flush])
        if rest == [], do: r, else: wait_first(rest)
    end
  end

  # two nodes; equal bits or an incident settled by the exact oracle
  defp audit(job, [{a, pa}, {b, pb} | rest]) do
    [ra, rb] = Task.await_many([Task.async(fn -> call(pa, job) end), Task.async(fn -> call(pb, job) end)], :infinity)

    case {ra, rb} do
      {{:ok, oa}, {:ok, ob}} when oa == ob ->
        entry = %{key: hex(job.key), nodes: [a, b], agree: true}
        {{:ok, oa, %{node: a, cached: false, audited: true, with: b, attempts: 1, key: job.key}}, [{:done, a}, {:done, b}, {:audit, entry}]}

      {{:ok, oa}, {:ok, ob}} ->
        {pkey, comp} = job.program
        _ = pkey
        {:ok, truth} = Vapor.Runtime.Native.run_oracle(comp, job.env, job.opts)
        t = truth.outputs
        liars = for {n, o} <- [{a, oa}, {b, ob}], o != t, do: n
        entry = %{key: hex(job.key), nodes: [a, b], agree: false, wrong: liars, digest_a: digest(oa), digest_b: digest(ob), digest_oracle: digest(t)}
        events = [{:done, a}, {:done, b}, {:audit, entry}] ++ for(n <- liars, do: {:quarantine, n, {:disagreed_with_oracle, hex(job.key)}})
        {{:ok, t, %{node: :oracle, cached: false, audited: true, with: [a, b], incident: entry, attempts: 1, key: job.key}}, events}

      {{:ok, oa}, {:error, why}} ->
        {{:ok, oa, %{node: a, cached: false, audited: false, attempts: 2, key: job.key}}, [{:done, a}, {:failed, b, why}]}

      {{:error, why}, _} ->
        {res, ev} = single(job, [{b, pb} | rest], nil, [{:failed, a, why}], 2)
        {res, ev}
    end
  end

  defp call(pid, job) do
    {pkey, comp} = job.program

    case safe(fn -> CNode.run(pid, pkey, nil, job.env, job.opts) end, {:error, :unreachable}) do
      {:need, ^pkey} -> safe(fn -> CNode.run(pid, pkey, comp, job.env, job.opts) end, {:error, :unreachable})
      other -> other
    end
  end

  defp safe(f, default) do
    f.()
  catch
    :exit, _ -> default
  end

  defp digest(outs), do: hex(:crypto.hash(:sha256, for({k, t} <- Enum.sort(outs), do: [to_string(k), t.data])))
  defp hex(b), do: Base.encode16(b, case: :lower) |> binary_part(0, 16)

  # ---------------------------------------------------------------- keys --

  defp program_key(%Compiled{} = comp, s) do
    ref = :erlang.phash2({comp.policy, comp.schedule, map_size(comp.slots)})

    case Map.fetch(s.digests, ref) do
      {:ok, {^comp, k}} -> {k, s}
      _ ->
        k = Vapor.Canonical.digest({comp.policy, comp.program})
        {k, %{s | digests: Map.put(s.digests, ref, {comp, k})}}
    end
  end

  defp job_key(pkey, env, opts) do
    :crypto.hash(:sha256, [pkey, :erlang.term_to_binary(Keyword.take(opts, [:iterations, :sequence])),
                           for({k, t} <- Enum.sort(env), do: [to_string(k), to_string(t.dtype), :erlang.term_to_binary(t.shape), t.data])])
  end
end

defmodule Vapor.Cluster.Node do
  @moduledoc """
  One node of a `Vapor.Cluster`: its native workers, the programs it has
  been sent (by content key: a program crosses the network once), and the
  jobs it runs. A test hook makes it lie (one flipped bit per result).
  """
  use GenServer
  alias Vapor.Runtime.{Native, Substrates, Worker}

  @doc "Start on `node` (locally when it is this node)."
  def start(node, opts) do
    if node == Node.self(), do: GenServer.start(__MODULE__, opts), else: :erpc.call(node, GenServer, :start, [__MODULE__, opts], 30_000)
  end

  @doc "Run program `pkey` (send `comp` when the node replies `{:need, pkey}`)."
  def run(pid, pkey, comp, env, opts), do: GenServer.call(pid, {:run, pkey, comp, env, opts}, :infinity)

  def workers(pid), do: GenServer.call(pid, :workers)
  def admit(pid), do: GenServer.call(pid, :admit, :infinity)

  @doc false
  def corrupt(pid, on?), do: GenServer.call(pid, {:corrupt, on?})

  @doc false
  # test hook: answer every job `ms` late (a straggler)
  def delay(pid, ms), do: GenServer.call(pid, {:delay, ms})

  @impl true
  def init(opts) do
    exec = [Substrates.binary("vapor-worker", "native")]
    ws = for _ <- 1..max(Keyword.get(opts, :workers, 1), 1), do: elem(Worker.start_link(exec: exec), 1)
    {:ok, %{workers: ws, next: 0, programs: %{}, corrupt: false, delay: 0}}
  end

  @impl true
  def handle_call({:run, pkey, comp, env, opts}, from, s) do
    case {Map.get(s.programs, pkey), comp} do
      {nil, nil} ->
        {:reply, {:need, pkey}, s}

      {have, sent} ->
        c = have || sent
        w = Enum.at(s.workers, rem(s.next, length(s.workers)))
        corrupt = s.corrupt
        delay = s.delay

        Task.start(fn ->
          if delay > 0, do: Process.sleep(delay)

          reply =
            case Native.run(w, c, env, [isa: Substrates.host_isa(), mode: :native] ++ Keyword.take(opts, [:iterations, :sequence])) do
              {:ok, r} -> {:ok, if(corrupt, do: flip(r.outputs), else: r.outputs)}
              {:error, why} -> {:error, why}
            end

          GenServer.reply(from, reply)
        end)

        {:noreply, %{s | next: s.next + 1, programs: Map.put(s.programs, pkey, c)}}
    end
  end

  def handle_call(:workers, _from, s), do: {:reply, s.workers, s}
  def handle_call({:corrupt, on?}, _from, s), do: {:reply, :ok, %{s | corrupt: on?}}
  def handle_call({:delay, ms}, _from, s), do: {:reply, :ok, %{s | delay: ms}}

  # the battery runs through the very path jobs take (corruption included)
  def handle_call(:admit, _from, s) do
    w = hd(s.workers)
    corrupt = s.corrupt

    runner = fn c, env, o ->
      with {:ok, r} <- Native.run(w, c, env, [isa: Substrates.host_isa(), mode: :native] ++ o),
           do: {:ok, %{r | outputs: if(corrupt, do: flip(r.outputs), else: r.outputs)}}
    end

    a = Vapor.Substrate.admit(%{id: node(), kind: :native, isa: Substrates.host_isa(), mode: :native, server: w}, runner: runner, device: "#{node()}")
    {:reply, {:ok, a}, s}
  end

  defp flip(outs) do
    {k, t} = outs |> Enum.sort() |> hd()
    <<x, rest::binary>> = t.data
    Map.put(outs, k, %{t | data: <<Bitwise.bxor(x, 1), rest::binary>>})
  end
end
