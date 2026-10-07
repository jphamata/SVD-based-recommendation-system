defmodule Vapor.Engine.Pool do
  @moduledoc """
  Data parallelism over requests: `n` replicas of `Vapor.Engine`, each with
  its own worker process, behind the engine's own API (`Vapor.Engine.generate/3`,
  `complete/4`, `info/1` work on a pool unchanged, and so does `Vapor.Serve`).

    * **One compilation, one copy of the weights.** The model is prepared
      once (`Vapor.Engine.prepare/1`); every replica opens a session on the
      same compiled program, whose large constants live in content-addressed
      shared memory — `n` workers map the same physical pages.
    * **Routing:** a new request goes to the replica with the fewest
      outstanding requests (ties: the lowest index). Every replica computes
      the same bits for a request — same program, batch invariance — so the
      routing decides only *when* a request runs, never *what* it yields
      (tested).
    * **Isolation:** a replica that dies takes only its own requests with it
      (each ends with `{:done, :error, usage}`); the pool starts a new
      replica in its place and keeps serving. A worker crash inside a
      replica is handled by the engine itself.

  Threads: `:threads` is the total; each replica gets `max(1, threads ÷ n)`
  for its intra-op pool. Messages from replicas pass through the pool
  (one extra local hop per token), which is how it knows what is outstanding.
  """
  use GenServer
  alias Vapor.Engine

  def start_link(opts), do: GenServer.start_link(__MODULE__, opts, Keyword.take(opts, [:name]))

  @impl true
  def init(opts) do
    Process.flag(:trap_exit, true)
    n = Keyword.fetch!(opts, :replicas)
    per = max(1, div(Keyword.get(opts, :threads, 1), n))

    with {:ok, prep} <- Engine.prepare(opts) do
      ropts = opts |> Keyword.drop([:name, :replicas, :config, :weights, :model]) |> Keyword.merge(prepared: prep, threads: per, replicas: 1)

      replicas =
        Enum.reduce_while(0..(n - 1), {:ok, %{}}, fn i, {:ok, acc} ->
          case Engine.start_link(ropts) do
            {:ok, pid} -> {:cont, {:ok, Map.put(acc, i, pid)}}
            {:error, why} -> {:halt, {:error, why}}
          end
        end)

      with {:ok, reps} <- replicas do
        {:ok, %{opts: ropts, replicas: reps, load: Map.new(0..(n - 1), &{&1, 0}), refs: %{}, restarts: 0}}
      end
    else
      {:error, why} -> {:stop, why}
    end
  end

  @impl true
  def handle_call({:generate, prompt, opts}, _from, st) do
    {i, _} = Enum.min_by(st.load, fn {i, l} -> {l, i} end)
    to = Keyword.fetch!(opts, :to)

    case GenServer.call(st.replicas[i], {:generate, prompt, Keyword.put(opts, :to, self())}) do
      {:ok, ref} ->
        # the replica's receiver is the pool; the pool watches the real one
        mon = Process.monitor(to)
        {:reply, {:ok, ref}, %{st | refs: Map.put(st.refs, ref, {i, to, mon}), load: Map.update!(st.load, i, &(&1 + 1))}}

      other ->
        {:reply, other, st}
    end
  end

  def handle_call(:info, _from, st) do
    infos = st.replicas |> Enum.sort() |> Enum.map(fn {_, pid} -> Engine.info(pid) end)
    first = hd(infos)
    sum = fn key -> infos |> Enum.map(&Map.fetch!(&1, key)) |> Enum.sum() end
    stats = Enum.reduce(infos, %{}, fn i, acc -> Map.merge(acc, i.stats, fn _, a, b -> a + b end) end)

    {:reply,
     %{first | sequences: sum.(:sequences), pages: sum.(:pages), active: sum.(:active), queued: sum.(:queued), stats: stats}
     |> Map.merge(%{replicas: map_size(st.replicas), load: st.load, restarts: st.restarts}), st}
  end

  @impl true
  def handle_cast({:cancel, ref}, st) do
    with {i, _, _} <- st.refs[ref], do: Engine.cancel(st.replicas[i], ref)
    {:noreply, st}
  end

  @impl true
  def handle_info({:vapor, ref, {:token, _, _}} = m, st) do
    with {_, to, _} <- st.refs[ref], do: send(to, m)
    {:noreply, st}
  end

  def handle_info({:vapor, ref, {:done, _, _}} = m, st) do
    case Map.pop(st.refs, ref) do
      {{i, to, mon}, refs} ->
        Process.demonitor(mon, [:flush])
        send(to, m)
        {:noreply, %{st | refs: refs, load: Map.update!(st.load, i, &(&1 - 1))}}

      {nil, _} ->
        {:noreply, st}
    end
  end

  # a receiver is gone: its requests are withdrawn from their replicas
  def handle_info({:DOWN, mon, :process, _, _}, st) do
    {gone, refs} = Enum.split_with(st.refs, fn {_, {_, _, m}} -> m == mon end)
    for {ref, {i, _, _}} <- gone, do: Engine.cancel(st.replicas[i], ref)
    load = Enum.reduce(gone, st.load, fn {_, {i, _, _}}, l -> Map.update!(l, i, &(&1 - 1)) end)
    {:noreply, %{st | refs: Map.new(refs), load: load}}
  end

  # a replica died: its requests end with :error, a new replica takes its place
  def handle_info({:EXIT, pid, reason}, st) do
    case Enum.find(st.replicas, fn {_, p} -> p == pid end) do
      nil ->
        if reason == :normal, do: {:noreply, st}, else: {:stop, reason, st}

      {i, _} ->
        {lost, refs} = Enum.split_with(st.refs, fn {_, {j, _, _}} -> j == i end)
        for {ref, {_, to, mon}} <- lost do
          Process.demonitor(mon, [:flush])
          send(to, {:vapor, ref, {:done, :error, %{prompt_tokens: 0, completion_tokens: 0}}})
        end
        {:ok, new} = Engine.start_link(st.opts)

        {:noreply, %{st | replicas: Map.put(st.replicas, i, new), refs: Map.new(refs), load: Map.put(st.load, i, 0),
                          restarts: st.restarts + 1}}
    end
  end

  @impl true
  def format_status(_reason, [_pdict, st]),
    do: [data: [{~c"State", %{replicas: map_size(st.replicas), load: st.load, restarts: st.restarts}}]]
end
