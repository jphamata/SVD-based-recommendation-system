defmodule Vapor.Runtime.Session do
  @moduledoc """
  A compiled program resident in a worker: weights mapped once, every slot
  allocated at its certified maximal extent, state (KV caches) kept in
  worker memory between steps. A step writes only the inputs it is given,
  runs the schedule resolved for the step's extents, and returns only the
  outputs asked for — for autoregressive decoding the per-token traffic is
  a few ids in and one row of logits out, independent of context length.

  Inputs not given to a step keep their contents (an in-place state such as
  a KV cache or a page pool is never written by the BEAM at all; it starts
  zeroed). State that is not updated in place — the recurrent state of an
  SSM, `h ← h_next` — is copied back inside the worker after each step, so
  it never crosses to the BEAM either. Dimensions are unified from the given inputs; an input omitted
  from a step must have a static shape.

  If the worker dies, the session dies with it (`{:error, :session_lost}`
  thereafter): its caches existed only there. The caller decides whether to
  reopen and recompute.

  **On the GPU** (`isa: :spirv` or `:msl`, `worker` a `Vapor.Runtime.Fabric`), the
  same contract holds on the Vulkan daemon: buffers live in device memory
  for the life of the session (weights uploaded or imported once, caches
  never leave the device), pipelines are created at OPEN, and a step's
  recorded command buffer is replayed whenever its geometry repeats (a
  decode loop at a steady batch). The results are bit-identical to the CPU
  session — same certified semantics, same SPIR-V as a one-shot run.
  `staging: true` forces the discrete-GPU path (device memory the host
  cannot map, every byte through a staging buffer).
  """
  alias Vapor.{Compiled, Tensor}
  alias Vapor.Emit.SpirvKernels
  alias Vapor.Runtime.{Fabric, Native, Plan, Worker}

  defstruct [:worker, :ref, :compiled, :index, :entries, kind: :native, modules: %{}, device: %{}, info: %{}]

  @type t :: %__MODULE__{}

  @doc """
  Open `c` on `worker`. Options: `:isa` (required), `:mode`, `:vlen`,
  `:poison` (as `Vapor.Runtime.Native.run/4`), `:init` — initial contents
  for inputs (`%{name => Tensor}` at maximal shape; default zero),
  `:consts` — the constant buffers from `const_buffers/1`, when several
  sessions open the same program (the weights are hashed and placed in
  shared memory once).
  """
  def open(worker, %Compiled{} = c, opts) do
    if Keyword.fetch!(opts, :isa) in [:spirv, :msl], do: open_fabric(worker, c, opts), else: open_native(worker, c, opts)
  end

  defp open_native(worker, %Compiled{} = c, opts) do
    isa = Keyword.fetch!(opts, :isa)
    %{blob: blob, entries: entries} = Map.fetch!(c.code, isa)
    init = Keyword.get(opts, :init, %{})
    ids = c.slots |> Map.keys() |> Enum.sort()
    index = ids |> Enum.with_index() |> Map.new()
    max = max_dims(c)

    scratch = Native.scratch_bytes(c, Worker.info(worker).threads)
    consts = Keyword.get_lazy(opts, :consts, fn -> const_buffers(c) end)

    buffers =
      Enum.map(ids, fn id ->
        slot = c.slots[id]
        len = max(Compiled.nbytes(c, id, max), Map.get(scratch, id, 0))

        case slot.role do
          {:const, _} -> Map.fetch!(consts, id)
          {:input, name} ->
            case init do
              %{^name => %Tensor{data: d}} when byte_size(d) == len -> %{kind: :inline, data: d, len: len, writable: true}
              _ -> %{kind: :zero, len: len, writable: true}
            end
          :tmp -> %{kind: :zero, len: len, writable: true}
        end
      end)

    with {:ok, ref} <- Worker.open(worker, Plan.encode_open(blob, buffers, opts)) do
      {:ok, %__MODULE__{worker: worker, ref: ref, compiled: c, index: index, entries: entries}}
    end
  end

  @doc "The buffers of a program's constants (large ones in content-addressed shared memory), by slot."
  def const_buffers(%Compiled{} = c) do
    for {id, %{role: {:const, %Tensor{data: d}}}} <- c.slots, into: %{}, do: {id, Native.const_buffer(d)}
  end

  @doc """
  One step: write `inputs`, run, return `outputs` (names) as tensors.
  Options: `:timeout` (ms).
  """
  def step(s, inputs, outputs, opts \\ [])

  def step(%__MODULE__{kind: :fabric} = s, inputs, outputs, opts), do: step_fabric(s, inputs, outputs, opts)

  def step(%__MODULE__{compiled: c} = s, inputs, outputs, opts) do
    timeout = Keyword.get(opts, :timeout, 60_000)

    with {:ok, dims} <- Compiled.dims(c, inputs, partial: true) do
      by_name = for %{role: {:input, n}} = slot <- Map.values(c.slots), into: %{}, do: {n, slot.id}
      writes = for {name, %Tensor{data: d}} <- inputs, do: {s.index[Map.fetch!(by_name, name)], 0, d}
      calls = Native.calls(c, dims, s.entries, s.index)
      outs = Enum.map(outputs, fn name -> {name, c.outputs |> List.keyfind(name, 0) |> elem(1)} end)
      returns = Enum.map(outs, fn {_, id} -> {s.index[id], 0, Compiled.nbytes(c, id, dims)} end)
      # state not updated in place (a recurrent model's h ← h_next) is fed
      # back inside the worker after the pass
      copies = for {in_id, out_id} <- c.state, in_id != out_id, do: {s.index[out_id], 0, s.index[in_id], 0, Compiled.nbytes(c, out_id, dims)}

      case Worker.step(s.worker, s.ref, Plan.encode_step(writes, calls, returns, timeout, copies), timeout) do
        {:ok, %{returns: bins} = res} ->
          tensors =
            outs
            |> Enum.zip(bins)
            |> Map.new(fn {{name, id}, bin} -> {name, Tensor.new(c.slots[id].dtype, Compiled.shape(c, id, dims), bin)} end)

          {:ok, tensors, %{elapsed_ns: res.elapsed_ns, counters: res.counters}}

        err ->
          err
      end
    end
  end

  @doc "Close the session (the worker or the device frees its memory)."
  def close(%__MODULE__{kind: :fabric, worker: f, ref: h}), do: Fabric.close_session(f, h)
  def close(%__MODULE__{worker: w}), do: Worker.close_session(w)

  @doc "Where the session lives: `%{kind, device, staged, device_local, resident_bytes}` (GPU) or `%{kind: :native}`."
  def info(%__MODULE__{kind: :fabric, info: i, device: d}), do: i |> Map.put(:kind, :fabric) |> Map.put(:device, d[:name])
  def info(%__MODULE__{kind: k, info: i}), do: Map.put(i, :kind, k)

  # ----------------------------------------------------------- the GPU --

  defp open_fabric(fabric, %Compiled{} = c, opts) do
    device = Fabric.info(fabric)
    max = max_dims(c)
    ids = c.slots |> Map.keys() |> Enum.sort()
    index = ids |> Enum.with_index() |> Map.new()

    # every kernel the schedule may dispatch, with the device's variants
    # (an int8 GEMM may take the cooperative-matrix path at some extents)
    keys =
      c.schedule
      |> Enum.flat_map(fn %{kernel: k} -> if k == :gemm_i8 and device.coop_i8, do: [k, :gemm_i8_coop], else: [k] end)
      |> Enum.uniq()

    with :ok <- supported(keys, device) do
      modules = keys |> Enum.with_index() |> Map.new()

      mods = Enum.map(keys, &Fabric.module_entry(c, &1, device))

      init = Keyword.get(opts, :init, %{})

      buffers =
        Enum.map(ids, fn id ->
          len = Compiled.nbytes(c, id, max)

          case c.slots[id].role do
            {:const, %Tensor{data: d}} ->
              case byte_size(d) >= 64 * 1024 and Vapor.Runtime.Shm.put(d) do
                {:ok, path} -> [<<2, 0, byte_size(d)::64-little, byte_size(path)::16-little>>, path, <<0::64>>]
                _ -> [<<1, 0, byte_size(d)::64-little>>, d]
              end

            {:input, name} ->
              case init do
                %{^name => %Tensor{data: d}} when byte_size(d) == len -> [<<1, 1, len::64-little>>, d]
                _ -> <<0, 1, len::64-little>>
              end

            :tmp ->
              <<0, 1, len::64-little>>
          end
        end)

      flags = if Keyword.get(opts, :staging, false), do: 1, else: 0
      timeout = Keyword.get(opts, :timeout, 120_000)

      frame =
        IO.iodata_to_binary([<<10, timeout::32-little, flags::32-little, length(mods)::32-little>>, mods,
                             <<length(buffers)::32-little>>, buffers])

      with {:ok, handle, info} <- Fabric.open_session(fabric, frame, timeout) do
        {:ok, %__MODULE__{kind: :fabric, worker: fabric, ref: handle, compiled: c, index: index, modules: modules,
                          device: device, info: info}}
      end
    end
  end

  defp supported(keys, device) do
    ok? = if device[:format] == :msl, do: &(Vapor.Emit.MSL.compile(&1, :canonical) != nil), else: &SpirvKernels.supported?/1

    case Enum.reject(keys, ok?) do
      [] -> :ok
      miss -> {:error, {:fabric_unsupported, miss}}
    end
  end

  defp step_fabric(%__MODULE__{compiled: c} = s, inputs, outputs, opts) do
    timeout = Keyword.get(opts, :timeout, 60_000)

    with {:ok, dims} <- Compiled.dims(c, inputs, partial: true),
         {:ok, disps} <- dispatches(s, dims) do
      by_name = for %{role: {:input, n}} = slot <- Map.values(c.slots), into: %{}, do: {n, slot.id}
      writes = for {name, %Tensor{data: d}} <- inputs, do: [<<s.index[Map.fetch!(by_name, name)]::32-little, 0::64, byte_size(d)::64-little>>, d]
      outs = Enum.map(outputs, fn name -> {name, c.outputs |> List.keyfind(name, 0) |> elem(1)} end)
      returns = Enum.map(outs, fn {_, id} -> <<s.index[id]::32-little, 0::64, Compiled.nbytes(c, id, dims)::64-little>> end)

      copies =
        for {in_id, out_id} <- c.state, in_id != out_id,
            do: <<s.index[out_id]::32-little, 0::64, s.index[in_id]::32-little, 0::64, Compiled.nbytes(c, out_id, dims)::64-little>>

      frame =
        [<<timeout::32-little, length(writes)::32-little>>, writes,
         <<length(returns)::32-little>>, returns,
         <<length(disps)::32-little>>, disps,
         <<length(copies)::32-little>>, copies]

      case Fabric.step_session(s.worker, s.ref, frame, timeout) do
        {:ok, %{returns: bins} = res} ->
          tensors =
            outs
            |> Enum.zip(bins)
            |> Map.new(fn {{name, id}, bin} -> {name, Tensor.new(c.slots[id].dtype, Compiled.shape(c, id, dims), bin)} end)

          {:ok, tensors, %{elapsed_ns: res.elapsed_ns, counters: res.counters}}

        err ->
          err
      end
    end
  end

  # the schedule resolved under the step's extents: module, geometry, push
  # constants, bound session buffers — every dispatch within its module's limits
  defp dispatches(%__MODULE__{compiled: c} = s, dims) do
    Enum.reduce_while(c.schedule, {:ok, []}, fn %{kernel: key, args: args}, {:ok, acc} ->
      resolved = Enum.map(args, &Compiled.resolve_arg(c, &1, dims))
      push = for {:imm, v} <- resolved, do: v
      key = Fabric.variant(key, push, s.device)

      if SpirvKernels.fits?(key, push) do
        {gx, gy, gz} = SpirvKernels.groups(key, push)
        binds = for {:slot, id} <- resolved, do: <<s.index[id]::32-little>>

        d = [<<Map.fetch!(s.modules, key)::32-little, gx::32-little, gy::32-little, gz::32-little, length(push)::32-little>>,
             Enum.map(push, &<<&1::32-little>>), <<length(binds)::32-little>>, binds]

        {:cont, {:ok, [d | acc]}}
      else
        {:halt, {:error, {:fabric_limit, key, push}}}
      end
    end)
    |> case do
      {:ok, ds} -> {:ok, Enum.reverse(ds)}
      err -> err
    end
  end

  # every dynamic extent at its certified maximum
  defp max_dims(c) do
    for %{shape: shape} <- Map.values(c.slots), {:dyn, sym, max} <- shape, into: %{}, do: {sym, max}
  end
end
