defmodule Vapor.Hermetic do
  @moduledoc """
  **The hermetic seal** (the alchemists' vessel closed against the world,
  Hermes' seal, whence *hermetic*): the one place where vapor contains work
  on untrusted input. That means a stranger's file (PDF, JBIG2, JPEG, zip, GGUF), a
  program written in Alembic by a person or a model, or a verb typed into
  the console's terminal. Every such job runs in a fresh BEAM process that:

    * has its memory capped by the VM (`max_heap_size` with
      `include_shared_binaries: true`) and is killed when it crosses the
      cap;
    * is killed after a wall-clock deadline;
    * cannot take its caller down: a crash, an exhausted cap or a missed
      deadline comes back as a value.

  Containment counts **off-heap binaries**, not only the process heap.
  Before 0.17 the four sandboxes vapor carried (the Alembic sandbox, the
  Dīwān jail, the Athanor session, the interactive furnace) capped only the
  heap. Every binary above 64 bytes lives outside it, so a decompression
  bomb, or a verb that built a large binary, could allocate past any cap
  and exhaust the node. Measured on OTP 28: under a 64 MB cap, 512 MB of
  binaries survived without the flag and are killed with it
  (`test/vapor/hermetic_test.exs`).

  The admission boundaries (`Vapor.Lock` for models, `Vapor.Docs` for
  files, `Vapor.Substrate` for accelerators) decide *what* gets in; the seal
  bounds *what it can cost*. The proposal this replaces was to move the
  parsers into the Zig worker behind seccomp. That trades a memory-safe language for a memory-unsafe
  one to gain an isolation the BEAM already gives per process. The real
  exposures of an Elixir parser are memory, time and crashes, and those are
  contained here. Machine code still runs only in the isolated workers, as
  before.
  """

  @type failure :: :memory | :timeout | {:crash, term()}

  @default_heap_mb 256
  @default_timeout 30_000

  @doc """
  Run `fun` contained. Options: `heap_mb` (default #{@default_heap_mb}),
  `timeout` in ms (default #{@default_timeout}, or `:infinity`).

  Returns `{:ok, result}` or `{:error, :memory | :timeout | {:crash, reason}}`.
  The result is copied to the caller (large binaries are shared, not copied).
  """
  @spec seal((-> term()), keyword()) :: {:ok, term()} | {:error, failure()}
  def seal(fun, opts \\ []) when is_function(fun, 0) do
    heap_mb = Keyword.get(opts, :heap_mb, @default_heap_mb)
    timeout = Keyword.get(opts, :timeout, @default_timeout)
    parent = self()
    ref = make_ref()

    {pid, mon} =
      spawn_monitor(fn ->
        cap_self(heap_mb)
        send(parent, {ref, fun.()})
      end)

    receive do
      {^ref, result} ->
        Process.demonitor(mon, [:flush])
        {:ok, result}

      {:DOWN, ^mon, :process, ^pid, :killed} ->
        {:error, :memory}

      {:DOWN, ^mon, :process, ^pid, reason} ->
        {:error, {:crash, reason}}
    after
      timeout ->
        Process.exit(pid, :kill)
        Process.demonitor(mon, [:flush])
        # a reply that raced the deadline must not leak into the caller's mailbox
        receive do
          {^ref, _} -> :ok
        after
          0 -> :ok
        end

        {:error, :timeout}
    end
  end

  @doc """
  Cap the calling process: its heap **and** the off-heap binaries it holds
  together stay under `heap_mb`, or the VM kills it. Use this in a
  long-lived process (a GenServer that runs a search); use `seal/2` for
  a single job.
  """
  @spec cap_self(pos_integer()) :: :ok
  def cap_self(heap_mb) when is_integer(heap_mb) and heap_mb > 0 do
    words = div(heap_mb * 1_048_576, :erlang.system_info(:wordsize))
    Process.flag(:max_heap_size, %{size: words, kill: true, error_logger: false, include_shared_binaries: true})
    :ok
  end

  @doc "A failure as one line of text, for people and for logs."
  @spec describe(failure(), keyword()) :: String.t()
  def describe(failure, opts \\ [])
  def describe(:memory, opts), do: "used more than #{Keyword.get(opts, :heap_mb, @default_heap_mb)} MB and was stopped"
  def describe(:timeout, opts), do: "took longer than #{format_ms(Keyword.get(opts, :timeout, @default_timeout))} and was stopped"
  def describe({:crash, {%{__exception__: true} = e, _stack}}, _opts), do: "stopped: " <> Exception.message(e)
  def describe({:crash, reason}, _opts), do: "stopped: " <> String.slice(inspect(reason), 0, 200)

  defp format_ms(:infinity), do: "its deadline"
  defp format_ms(ms) when ms >= 1000, do: "#{div(ms, 1000)} s"
  defp format_ms(ms), do: "#{ms} ms"
end
