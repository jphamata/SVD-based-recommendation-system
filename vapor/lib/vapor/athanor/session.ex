defmodule Vapor.Athanor.Session do
  @moduledoc """
  A live Athanor run that a person steers (docs/ATHANOR.md §4): the search
  advances a round at a time in its own process (heap capped by the VM),
  and between rounds it takes what the person does — propose candidates,
  pin one to keep it alive in the archive, ban one, ask the model for
  ideas, give measurements for a measured problem, pause, resume, give
  more budget. Everything the person and the model put in is recorded in
  the certificate's `outside_proposals`, so the run stays replayable.

  Sessions are named by a random id, registered in `Vapor.Athanor.Registry`
  and end after 30 idle minutes.
  """
  use GenServer
  alias Vapor.Athanor
  alias Vapor.Athanor.Spec

  @idle 30 * 60_000

  @doc "Start a session for a problem; `{:ok, id}` or `{:error, why}`."
  def start(text, opts \\ []) do
    with {:ok, spec} <- Spec.parse(text, Keyword.take(opts, [:consts, :budget, :seed])) do
      id = Vapor.Entropy.token(9)
      case DynamicSupervisor.start_child(Vapor.Athanor.Sessions, %{id: id, start: {GenServer, :start_link, [__MODULE__, {id, spec, opts}, [name: via(id)]]}, restart: :temporary}) do
        {:ok, _pid} -> {:ok, id}
        {:error, e} -> {:error, inspect(e)}
      end
    end
  end

  defp via(id), do: {:via, Registry, {Vapor.Athanor.Registry, id}}

  @doc "Call a session: `:snapshot`, `{:propose, [values]}`, `{:pin, key}`, `{:ban, key}`, `:stop`, `:resume`, `{:extend, n}`, `{:measure, key, v}`, `:certificate`, `{:mind, n}`."
  def call(id, msg, timeout \\ 60_000) do
    case Registry.lookup(Vapor.Athanor.Registry, id) do
      [{pid, _}] -> GenServer.call(pid, msg, timeout)
      [] -> {:error, "no session #{id} (it may have ended after 30 idle minutes)"}
    end
  catch
    :exit, _ -> {:error, "the session ended"}
  end

  def close(id) do
    case Registry.lookup(Vapor.Athanor.Registry, id) do
      [{pid, _}] -> GenServer.stop(pid, :normal)
      [] -> :ok
    end
  end

  @impl true
  def init({id, spec, opts}) do
    Vapor.Hermetic.cap_self(Keyword.get(opts, :heap_mb, 1024))
    run = Athanor.init(spec, Keyword.drop(opts, [:measure]))
    send(self(), :tick)
    {:ok, %{id: id, run: run, opts: opts, last: System.monotonic_time(:millisecond)}, @idle}
  end

  @impl true
  def handle_info(:tick, st) do
    run = Athanor.step(st.run, 1)
    if run.status == :running and run.pending == [], do: send(self(), :tick)
    {:noreply, %{st | run: run}, @idle}
  end

  def handle_info(:timeout, st), do: {:stop, :normal, st}
  def handle_info(_, st), do: {:noreply, st, @idle}

  @impl true
  def handle_call(:snapshot, _from, st), do: {:reply, {:ok, Athanor.snapshot(st.run) |> Map.put(:id, st.id)}, st, @idle}
  def handle_call({:snapshot, since}, _from, st), do: {:reply, {:ok, Athanor.snapshot(st.run) |> Map.put(:id, st.id) |> Map.put(:sparks, Athanor.sparks(st.run, since))}, st, @idle}

  def handle_call({:propose, values}, _from, st) do
    run = Athanor.propose(st.run, values, :human)
    {:reply, {:ok, %{queued: length(run.queue), rejected: run.error_samples}}, kick(st, run), @idle}
  end

  def handle_call({:pin, key}, _from, st), do: {:reply, :ok, %{st | run: Athanor.pin(st.run, key)}, @idle}
  def handle_call({:ban, key}, _from, st), do: {:reply, :ok, %{st | run: Athanor.ban(st.run, key)}, @idle}
  def handle_call(:stop, _from, st), do: {:reply, :ok, %{st | run: Athanor.stop(st.run)}, @idle}
  def handle_call(:resume, _from, st), do: {:reply, :ok, kick(st, Athanor.resume(st.run)), @idle}
  def handle_call({:extend, n}, _from, st), do: {:reply, :ok, kick(st, Athanor.extend(st.run, n)), @idle}

  def handle_call({:measure, key, v}, _from, st) do
    case Athanor.measure(st.run, key, v) do
      {:ok, run} -> {:reply, :ok, kick(st, run), @idle}
      e -> {:reply, e, st, @idle}
    end
  end

  def handle_call({:mind, n}, _from, st) do
    case st.opts[:mind] do
      nil -> {:reply, {:error, "no language model is configured (see `vapor mind` / VAPOR_MIND)"}, st, @idle}
      mind ->
        ctx = %{spec: st.run.spec, archive: st.run.archive, best: st.run.best, observed: st.run.observed, elites: st.run.elites}
        case Vapor.Mind.propose(mind, ctx, n) do
          {:ok, xs} ->
            run = Athanor.propose(st.run, xs, :mind)
            {:reply, {:ok, %{proposed: length(xs)}}, kick(st, run), @idle}
          e -> {:reply, e, st, @idle}
        end
    end
  end

  def handle_call(:certificate, _from, st), do: {:reply, {:ok, Athanor.certificate(st.run)}, st, @idle}

  defp kick(st, run) do
    if st.run.status != :running or st.run.pending != [], do: send(self(), :tick)
    %{st | run: run}
  end
end
