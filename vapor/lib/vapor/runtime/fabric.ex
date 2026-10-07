defmodule Vapor.Runtime.Fabric do
  @moduledoc """
  OTP owner of the `vapor-fabric` Vulkan daemon (Substrate II, Axiom 3).

  The daemon is a separate OS process speaking `{packet, 4}` frames on its
  stdio. A driver fault or lost device terminates only that process: the
  unit in flight returns `{:error, {:fabric_crashed, status}}` (or
  `{:device_lost, …}`), the daemon is respawned, and `Vapor.Runtime.Dispatch`
  reroutes the unit to the native substrate.

  Plans are built from the same `Vapor.Compiled` schedule as the native
  path: slot arguments become storage-buffer bindings, immediates become
  push constants, sequence inputs become per-iteration *windows* (the
  daemon copies slice `t` into a window buffer before iteration `t`).

  The same owner drives `vapor-metal` (Apple GPUs, and its Linux stand-in
  `vapor-metal-sim`): the protocol is identical, the HELLO reply ends with
  a format byte, and modules travel as MSL source (`Vapor.Emit.MSL`, the
  same kernel library translated) instead of SPIR-V words — see
  `module_entry/3`.
  """
  use GenServer
  require Logger
  alias Vapor.{Compiled, Tensor}
  alias Vapor.Emit.{MSL, SpirvKernels}
  alias Vapor.Runtime.Plan

  @timeout 60_000

  def start_link(opts), do: GenServer.start_link(__MODULE__, opts, Keyword.take(opts, [:name]))

  @doc "Device facts: name, vendor, denormal preservation, host-import, coop-matrix."
  def info(server), do: GenServer.call(server, :info)

  @doc """
  Open a resident session (an OPEN frame, see `Vapor.Runtime.Session`).
  Returns `{:ok, handle, %{staged, device_local, resident_bytes}}`; the
  handle is invalidated by a daemon restart (the device memory is gone).
  """
  def open_session(server, frame, timeout \\ @timeout),
    do: GenServer.call(server, {:open, frame, timeout}, timeout + 10_000)

  @doc "One STEP of a session (`Plan.decode/1`-shaped result)."
  def step_session(server, handle, frame, timeout \\ @timeout),
    do: GenServer.call(server, {:step, handle, frame, timeout}, timeout + 10_000)

  @doc "Close a session (its device memory is freed)."
  def close_session(server, handle), do: GenServer.call(server, {:close, handle})

  @doc "Test hook: ask a daemon started with `--fault-injection` to crash."
  def inject_fault(server), do: GenServer.call(server, :inject_fault)

  @doc "Run a compiled program on the GPU. Same options as `Vapor.Runtime.Native.run/4`."
  def run(server, %Compiled{} = c, env, opts \\ []) do
    t_count = Keyword.get(opts, :iterations, 1)
    seq = MapSet.new(Keyword.get(opts, :sequence, []))
    step_env = Map.new(env, fn {k, v} -> {k, if(MapSet.member?(seq, k), do: Vapor.Runtime.Native.slice(v, 0), else: v)} end)

    with {:ok, dims} <- Compiled.dims(c, step_env),
         :ok <- fits(c, dims) do
      hello = GenServer.call(server, :info)
      {frame, out_ids} = encode(c, env, dims, seq, t_count, Keyword.put(opts, :device, hello))

      case GenServer.call(server, {:run, frame, Keyword.get(opts, :timeout, @timeout)}, :infinity) do
        {:ok, %{returns: rets, emits: emits, elapsed_ns: ns}} ->
          decode = fn bytes -> split(c, dims, out_ids, bytes) end

          {:ok,
           %{outputs: c.outputs |> Enum.zip(rets) |> Map.new(fn {{n, id}, b} -> {n, tensor(c, id, dims, b)} end),
             steps: Enum.map(emits, fn {_t, b} -> decode.(b) end),
             elapsed_ns: ns, retired: nil}}

        err ->
          err
      end
    end
  end

  # every dispatch within its module's static limits, or the unit is refused
  # (and `Vapor.Runtime.Dispatch` reroutes it)
  defp fits(c, dims) do
    Enum.find_value(c.schedule, :ok, fn %{kernel: key, args: args} ->
      push = for {:imm, v} <- Enum.map(args, &Compiled.resolve_arg(c, &1, dims)), do: v
      if not SpirvKernels.fits?(key, push), do: {:error, {:fabric_limit, key, push}}
    end)
  end

  # -------------------------------------------------------------- server --

  @impl true
  def init(opts) do
    # a write to a daemon that just died must be a message, not an exit
    # signal through the port's link (see Vapor.Runtime.Worker)
    Process.flag(:trap_exit, true)
    exec = Keyword.fetch!(opts, :exec)

    case spawn_daemon(exec) do
      {:ok, port, hello} -> {:ok, %{exec: exec, port: port, hello: hello, restarts: 0, gen: 0}}
      {:error, reason} -> {:stop, reason}
    end
  end

  @impl true
  def handle_call(:info, _from, s), do: {:reply, Map.put(s.hello, :restarts, s.restarts), s}

  def handle_call(:inject_fault, _from, s) do
    send_frame(s.port, <<9>>)
    reply = await(s.port, 10_000)
    {:reply, reply, if(match?({:error, {:fabric_crashed, _}}, reply), do: respawn(s), else: s)}
  end

  def handle_call({op, _frame, _t}, _from, %{hello: %{ready: false}} = s) when op in [:open],
    do: {:reply, {:error, {:fabric_unavailable, s.hello.name}}, s}

  def handle_call({:open, frame, timeout}, _from, s) do
    send_frame(s.port, frame)

    case collect(s.port, [], timeout + 5_000) do
      {:opened, sid, info} -> {:reply, {:ok, {sid, s.gen}, info}, s}
      other -> reply_fault(other, s)
    end
  end

  # a session opened before the daemon restarted lived in device memory that is gone
  def handle_call({:step, {_sid, gen}, _frame, _t}, _from, %{gen: g} = s) when gen != g,
    do: {:reply, {:error, :session_lost}, s}

  def handle_call({:step, {sid, _gen}, frame, timeout}, _from, s) do
    send_frame(s.port, [<<11, sid::32-little>>, frame])
    reply_fault(collect(s.port, [], timeout + 5_000), s)
  end

  def handle_call({:close, {_sid, gen}}, _from, %{gen: g} = s) when gen != g, do: {:reply, :ok, s}

  def handle_call({:close, {sid, _gen}}, _from, s) do
    send_frame(s.port, <<12, sid::32-little>>)
    {reply, s} = reply_fault(collect(s.port, [], 30_000), s) |> then(fn {:reply, r, s} -> {r, s} end)
    {:reply, if(match?({:ok, _}, reply), do: :ok, else: reply), s}
  end

  def handle_call({:run, _frame, _t}, _from, %{hello: %{ready: false}} = s),
    do: {:reply, {:error, {:fabric_unavailable, s.hello.name}}, s}

  def handle_call({:run, frame, timeout}, _from, s) do
    send_frame(s.port, frame)
    reply_fault(collect(s.port, [], timeout + 5_000), s)
  end

  defp reply_fault(result, s) do
    case result do
      {:ok, _} = ok ->
        {:reply, ok, s}

      {:error, {:fabric_crashed, _}} = err ->
        Logger.warning("vapor-fabric #{inspect(err)} — driver fault contained; respawning")
        {:reply, err, respawn(s)}

      {:error, {:unit_fault, %{code: :device_lost}}} = err ->
        {:reply, err, respawn(s)}

      {:error, :timeout} = err ->
        try do
          Port.close(s.port)
        rescue
          ArgumentError -> :already_closed
        end

        {:reply, err, respawn(s)}

      err ->
        {:reply, err, s}
    end
  end

  @impl true
  def handle_info({port, {:exit_status, st}}, %{port: port} = s) do
    Logger.warning("vapor-fabric exited #{st} while idle — respawning")
    {:noreply, respawn(s)}
  end

  def handle_info({:EXIT, port, why}, %{port: port} = s) do
    Logger.warning("vapor-fabric port closed (#{inspect(why)}) while idle — respawning")
    {:noreply, respawn(s)}
  end

  def handle_info(_, s), do: {:noreply, s}

  defp send_frame(port, frame) do
    Port.command(port, frame)
  rescue
    ArgumentError -> send(self(), {:EXIT, port, :closed})
  end

  defp collect(port, emits, timeout) do
    receive do
      {^port, {:data, <<3, t::32-little, bytes::binary>>}} ->
        collect(port, [{t, bytes} | emits], timeout)

      {^port, {:data, <<4, _::binary>> = f}} ->
        {:done, res} = Plan.decode(f)
        {:ok, Map.put(res, :emits, Enum.reverse(emits))}

      {^port, {:data, <<10, sid::32-little, staged, local, bytes::64-little>>}} ->
        {:opened, sid, %{staged: staged == 1, device_local: local == 1, resident_bytes: bytes}}

      {^port, {:data, <<5, code::32-little, vkres::signed-64-little, _::32, msg::binary>>}} ->
        {:error, {:unit_fault, %{code: if(code == 2, do: :device_lost, else: :vulkan_error),
                                 vk_result: vkres, message: msg}}}

      {^port, {:exit_status, st}} ->
        {:error, {:fabric_crashed, describe(st)}}

      {:EXIT, ^port, why} ->
        {:error, {:fabric_crashed, {:port, why}}}
    after
      timeout -> {:error, :timeout}
    end
  end

  defp await(port, timeout) do
    receive do
      {^port, {:data, _}} -> :ok
      {^port, {:exit_status, st}} -> {:error, {:fabric_crashed, describe(st)}}
      {:EXIT, ^port, why} -> {:error, {:fabric_crashed, {:port, why}}}
    after
      timeout -> {:error, :timeout}
    end
  end

  defp describe(st) when st > 128, do: {:signal, st - 128}
  defp describe(st), do: {:exit, st}

  defp respawn(s) do
    case spawn_daemon(s.exec) do
      {:ok, port, hello} -> %{s | port: port, hello: hello, restarts: s.restarts + 1, gen: s.gen + 1}
      {:error, reason} -> exit({:fabric_unavailable, reason})
    end
  end

  defp spawn_daemon([prog | args]) do
    with path when is_binary(path) <- System.find_executable(prog) || {:error, {:not_found, prog}} do
      port = Port.open({:spawn_executable, path}, [{:packet, 4}, :binary, :exit_status, :use_stdio, args: args])
      Port.command(port, <<1, 0::32>>)

      receive do
        {^port, {:data, <<1, ready, api::32-little, vendor::32-little, device::32-little,
                          denorm, host, coop, nlen::16-little, name::binary-size(nlen), rest::binary>>}} ->
          {:ok, port,
           %{ready: ready == 1, api: api, vendor: vendor, device: device, name: name,
             denorm_preserve32: denorm == 1, host_import: host == 1, coop_i8: coop == 1,
             format: if(match?(<<1, _::binary>>, rest), do: :msl, else: :spirv)}}

        {^port, {:exit_status, st}} ->
          {:error, {:fabric_exited, st}}
      after
        20_000 -> {:error, :hello_timeout}
      end
    end
  end

  # ------------------------------------------------------------ planning --

  @doc """
  Kernel variant for a dispatch on a given device: an int8 GEMM whose
  extents are multiples of 16 runs on cooperative-matrix hardware when the
  device reports the exact (s8, s8, s32, 16×16×16, subgroup) configuration.
  Integer accumulation is order-free (Theorem 7.1), so both variants are
  bit-identical and the choice is purely a performance one.
  """
  def variant(:gemm_i8, [m, n, k], %{coop_i8: true}) when rem(m, 16) == 0 and rem(n, 16) == 0 and rem(k, 16) == 0,
    do: :gemm_i8_coop

  def variant(key, _push, _device), do: key

  @doc """
  One module of a RUN or OPEN frame, in the daemon's format: SPIR-V words
  for Vulkan, MSL source for Metal (`len, bytes, nbind, npush` either way).
  """
  def module_entry(%Compiled{} = c, key, %{format: :msl}) do
    %{src: src, nbind: nb, npush: np} = MSL.compile(key, c.policy) || raise(ArgumentError, "no MSL form for #{inspect(key)}")
    [<<byte_size(src)::32-little>>, src, <<nb::32-little, np::32-little>>]
  end

  def module_entry(%Compiled{} = c, key, _device) do
    %{bin: bin, nbind: nb, npush: np} = Map.get(c.spirv, key) || SpirvKernels.compile(key, c.policy)
    [<<byte_size(bin)::32-little>>, bin, <<nb::32-little, np::32-little>>]
  end

  defp encode(c, env, dims, seq, t_count, opts) do
    ids = c.slots |> Map.keys() |> Enum.sort()
    index = ids |> Enum.with_index() |> Map.new()
    device = Keyword.get(opts, :device, %{})

    push_of = fn args ->
      for({:imm, v} <- Enum.map(args, &Compiled.resolve_arg(c, &1, dims)), do: v)
    end

    sched = Enum.map(c.schedule, fn %{kernel: key, args: args} = call -> %{call | kernel: variant(key, push_of.(args), device)} end)
    keys = sched |> Enum.map(& &1.kernel) |> Enum.uniq()
    mod_index = keys |> Enum.with_index() |> Map.new()

    modules = Enum.map(keys, &module_entry(c, &1, device))

    buffers =
      Enum.map(ids, fn id ->
        len = Compiled.nbytes(c, id, dims)

        case c.slots[id].role do
          {:input, name} ->
            %Tensor{data: d} = Map.fetch!(env, name)
            [<<1, 0, byte_size(d)::64-little>>, d]

          {:const, %Tensor{data: d}} ->
            case byte_size(d) >= 64 * 1024 and Vapor.Runtime.Shm.put(d) do
              {:ok, path} -> [<<2, 0, byte_size(d)::64-little, byte_size(path)::16-little>>, path, <<0::64>>]
              _ -> [<<1, 0, byte_size(d)::64-little>>, d]
            end

          :tmp ->
            <<0, 1, len::64-little>>
        end
      end)

    seq_slot = fn id ->
      case c.slots[id].role do
        {:input, name} -> MapSet.member?(seq, name)
        _ -> false
      end
    end

    dispatches =
      Enum.map(sched, fn %{kernel: key, args: args} ->
        resolved = Enum.map(args, &Compiled.resolve_arg(c, &1, dims))
        push = for {:imm, v} <- resolved, do: v
        slots = for {:slot, id} <- resolved, do: id
        {gx, gy, gz} = SpirvKernels.groups(key, push)

        binds =
          Enum.map(slots, fn id ->
            n = Compiled.nbytes(c, id, dims)
            if seq_slot.(id), do: <<2, index[id]::32-little, 0::64, n::64-little, n::64-little>>,
                              else: <<1, index[id]::32-little>>
          end)

        [<<mod_index[key]::32-little, gx::32-little, gy::32-little, gz::32-little, length(push)::32-little>>,
         Enum.map(push, &<<&1::32-little>>), <<length(binds)::32-little>>, binds]
      end)

    out_ids = Enum.map(c.outputs, &elem(&1, 1))
    emits = if t_count > 1, do: Enum.map(out_ids, &<<index[&1]::32-little, 0::64, 0::64, Compiled.nbytes(c, &1, dims)::64-little>>), else: []

    copies =
      Enum.map(c.state, fn {in_id, out_id} ->
        <<index[out_id]::32-little, 0::64, index[in_id]::32-little, 0::64, Compiled.nbytes(c, out_id, dims)::64-little>>
      end)

    frame =
      IO.iodata_to_binary([
        <<2, Keyword.get(opts, :timeout, @timeout)::32-little>>,
        <<length(modules)::32-little>>, modules,
        <<length(buffers)::32-little>>, buffers,
        <<length(dispatches)::32-little>>, dispatches,
        <<t_count::32-little, length(emits)::32-little>>, emits,
        <<length(copies)::32-little>>, copies,
        <<length(out_ids)::32-little>>, Enum.map(out_ids, &<<index[&1]::32-little>>)
      ])

    {frame, out_ids}
  end

  defp split(c, dims, out_ids, bytes) do
    {m, _} =
      Enum.reduce(Enum.zip(c.outputs, out_ids), {%{}, bytes}, fn {{name, id}, id}, {m, rest} ->
        n = Compiled.nbytes(c, id, dims)
        <<b::binary-size(n), rest::binary>> = rest
        {Map.put(m, name, tensor(c, id, dims, b)), rest}
      end)

    m
  end

  defp tensor(c, id, dims, bin), do: Tensor.new(c.slots[id].dtype, Compiled.shape(c, id, dims), bin)
end
