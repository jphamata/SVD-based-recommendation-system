defmodule Vapor.Runtime.Worker do
  @moduledoc """
  OTP owner of one `vapor-worker` OS process (Axiom 3 for the CPU substrate).

  The worker is spawned as a port with `{:packet, 4}` framing and
  `:exit_status`. If generated code faults (SIGSEGV, SIGILL, SIGBUS), hits
  the seccomp allowlist (SIGSYS) or its watchdog (SIGALRM), only that process
  dies: this GenServer reports `{:error, {:worker_crashed, status}}` for the
  unit in flight, spawns a fresh worker, and stays up. The BEAM never maps,
  jumps into, or links against generated code.

  `exec` is the command line — `["path/vapor-worker"]` natively, or under a
  user-mode emulator for foreign ISAs, e.g.
  `["qemu-riscv64", "-cpu", "rv64,v=true,vlen=256,vext_spec=v1.0", "path/vapor-worker"]`.

  `threads: n` gives the worker a pool of `n` threads for intra-operation
  data parallelism (row ranges of one kernel call; results are identical
  for every `n`).
  """
  use GenServer
  require Logger
  alias Vapor.Runtime.Plan

  @default_timeout 60_000

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, Keyword.take(opts, [:name]))
  end

  @doc """
  Run a plan. Options (besides `Plan.encode/2`'s): `:on_emit` — a 2-arity
  function receiving `(t, bytes)` per iteration; `:timeout` (ms).
  Returns `{:ok, %{returns, emits, elapsed_ns, retired}}` or `{:error, reason}`.
  """
  def run(server, %Plan{} = plan, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, @default_timeout)
    GenServer.call(server, {:run, plan, opts, timeout}, timeout + 5_000)
  end

  @doc """
  Open a session (a `Plan.encode_open/3` frame). Returns `{:ok, ref}`; the
  ref names this session in `step/4` and becomes invalid if the worker is
  restarted (a crash takes the session's memory with it).
  """
  def open(server, frame, timeout \\ @default_timeout),
    do: GenServer.call(server, {:open, frame, timeout}, timeout + 5_000)

  @doc "One step of session `ref` (a `Plan.encode_step/4` frame)."
  def step(server, ref, frame, timeout \\ @default_timeout),
    do: GenServer.call(server, {:step, ref, frame, timeout}, timeout + 5_000)

  @doc "Close the worker's session."
  def close_session(server), do: GenServer.call(server, :close)

  @doc "Host facts reported by the worker: arch, sandbox status, restarts."
  def info(server), do: GenServer.call(server, :info)

  # ------------------------------------------------------------ server --

  @impl true
  def init(opts) do
    # the port is linked to this process: a write to a worker that just died
    # (EPIPE) must arrive as a message, not as an exit signal that would take
    # this process — and whoever is linked to it — down
    Process.flag(:trap_exit, true)
    exec = Keyword.fetch!(opts, :exec)
    sandbox = Keyword.get(opts, :sandbox, true)

    threads = Keyword.get(opts, :threads, 1)

    case spawn_worker(exec, sandbox, threads) do
      {:ok, port, hello} ->
        {:ok, %{exec: exec, sandbox: sandbox, threads: threads, port: port, hello: hello, restarts: 0, session: nil}}

      {:error, reason} ->
        {:stop, reason}
    end
  end

  @impl true
  def handle_call(:info, _from, s), do: {:reply, Map.put(s.hello, :restarts, s.restarts), s}

  def handle_call({:open, frame, timeout}, _from, s) do
    case unit(s, frame, nil, timeout) do
      {{:ok, _}, s} -> ref = make_ref(); {:reply, {:ok, ref}, %{s | session: ref}}
      {err, s} -> {:reply, err, %{s | session: nil}}
    end
  end

  def handle_call({:step, ref, frame, timeout}, _from, %{session: ref} = s) when ref != nil do
    {reply, s} = unit(s, frame, nil, timeout)
    {:reply, reply, s}
  end

  def handle_call({:step, _ref, _frame, _timeout}, _from, s), do: {:reply, {:error, :session_lost}, s}

  def handle_call(:close, _from, s) do
    {reply, s} = unit(s, Plan.encode_close(), nil, @default_timeout)
    {:reply, reply, %{s | session: nil}}
  end

  def handle_call({:run, plan, opts, timeout}, _from, s) do
    send_frame(s.port, Plan.encode(plan, Keyword.put_new(opts, :deadline_ms, timeout)))
    on_emit = Keyword.get(opts, :on_emit)

    # the in-worker watchdog fires at `timeout`; the BEAM-side deadline is a
    # backstop with a grace period, so a runaway unit reports as SIGALRM
    case collect(s.port, on_emit, [], timeout + 2_000) do
      {:ok, _} = ok ->
        {:reply, ok, s}

      {:error, {:worker_crashed, _} = why} = err ->
        Logger.warning("vapor-worker #{inspect(why)} — contained; respawning")
        {:reply, err, respawn(s)}

      {:error, :timeout} = err ->
        close(s.port)
        {:reply, err, respawn(s)}

      {:error, _} = err ->
        {:reply, err, s}
    end
  end

  # send one frame, collect its reply; a crash respawns and drops the session
  defp unit(s, frame, on_emit, timeout) do
    send_frame(s.port, frame)

    case collect(s.port, on_emit, [], timeout + 2_000) do
      {:ok, _} = ok ->
        {ok, s}

      {:error, {:worker_crashed, _} = why} = err ->
        Logger.warning("vapor-worker #{inspect(why)} — contained; respawning")
        {err, respawn(s)}

      {:error, :timeout} = err ->
        close(s.port)
        {err, respawn(s)}

      {:error, _} = err ->
        {err, s}
    end
  end

  @impl true
  def handle_info({port, {:exit_status, st}}, %{port: port} = s) do
    Logger.warning("vapor-worker exited #{st} while idle — respawning")
    {:noreply, respawn(s)}
  end

  def handle_info({:EXIT, port, why}, %{port: port} = s) do
    Logger.warning("vapor-worker port closed (#{inspect(why)}) while idle — respawning")
    {:noreply, respawn(s)}
  end

  def handle_info(_msg, s), do: {:noreply, s}

  # a worker that died between two requests: the write fails or the port
  # exits; either way the reply is collected as a crash below
  defp send_frame(port, frame) do
    Port.command(port, frame)
  rescue
    ArgumentError -> send(self(), {:EXIT, port, :closed})
  end

  defp collect(port, on_emit, emits, timeout) do
    receive do
      {^port, {:data, frame}} ->
        case Plan.decode(frame) do
          {:emit, t, bytes} ->
            if on_emit, do: on_emit.(t, bytes)
            collect(port, on_emit, if(on_emit, do: emits, else: [{t, bytes} | emits]), timeout)

          {:done, res} ->
            {:ok, Map.put(res, :emits, Enum.reverse(emits))}

          {:error, e} ->
            {:error, {:unit_fault, e}}
        end

      {^port, {:exit_status, st}} ->
        {:error, {:worker_crashed, describe(st)}}

      {:EXIT, ^port, why} ->
        {:error, {:worker_crashed, {:port, why}}}
    after
      timeout -> {:error, :timeout}
    end
  end

  defp close(port) do
    Port.close(port)
  rescue
    ArgumentError -> :already_closed
  end

  defp describe(st) when st > 128, do: {:signal, signal_name(st - 128)}
  defp describe(st), do: {:exit, st}

  defp signal_name(4), do: :sigill
  defp signal_name(6), do: :sigabrt
  defp signal_name(7), do: :sigbus
  defp signal_name(8), do: :sigfpe
  defp signal_name(11), do: :sigsegv
  defp signal_name(14), do: :sigalrm
  defp signal_name(31), do: :sigsys
  defp signal_name(n), do: n

  defp respawn(s) do
    case spawn_worker(s.exec, s.sandbox, s.threads) do
      {:ok, port, hello} -> %{s | port: port, hello: hello, restarts: s.restarts + 1, session: nil}
      {:error, reason} -> exit({:worker_unavailable, reason})
    end
  end

  defp spawn_worker([prog | args], sandbox, threads) do
    with path when is_binary(path) <- System.find_executable(prog) || {:error, {:not_found, prog}} do
      port =
        Port.open({:spawn_executable, path}, [
          {:packet, 4}, :binary, :exit_status, :use_stdio, args: args
        ])

      Port.command(port, Plan.hello(sandbox: sandbox, threads: threads))

      receive do
        {^port, {:data, frame}} ->
          {:hello, hello} = Plan.decode(frame)
          {:ok, port, hello}

        {^port, {:exit_status, st}} ->
          {:error, {:worker_exited, st}}
      after
        10_000 -> {:error, :hello_timeout}
      end
    end
  end
end
