defmodule Vapor.Runtime.Native do
  @moduledoc """
  Substrate I execution: turns a `Vapor.Compiled` schedule plus concrete
  inputs into a `Vapor.Runtime.Plan` for a worker, runs it, and decodes the
  typed results.

  Weights cross the process boundary zero-copy: constants above
  `@shm_threshold` bytes are written once to a content-addressed file in
  `/dev/shm` and mapped read-only (copy-on-write) by every worker, native or
  emulated; small ones travel inline.

  Recurrent programs (state feedback) run all `T` iterations inside one
  worker call; per-iteration outputs stream back as they are produced.
  """
  alias Vapor.{Compiled, Tensor}
  alias Vapor.Runtime.{Oracle, Plan, Worker}

  @shm_threshold 64 * 1024

  @doc """
  Run on a worker. Options:

    * `:isa` — which code blob to use (`:x86_64`, `:aarch64`, `:riscv64`)
    * `:mode` — `:native` (execute) or `:emulate` (RVV interpreter)
    * `:iterations`, `:sequence` — recurrence: inputs named in `:sequence`
      carry a leading dimension `T` and are consumed one slice per iteration
    * `:on_emit` — `fn t, %{name => Tensor} -> any end`
    * `:stream` — `false` returns only the final outputs (no per-step frames)
    * any `Vapor.Runtime.Plan.encode/2` option (`:vlen`, `:poison`, `:fuel`)
  """
  def run(worker, %Compiled{} = c, env, opts \\ []) do
    t_count = Keyword.get(opts, :iterations, 1)
    seq = MapSet.new(Keyword.get(opts, :sequence, []))
    step_env = Map.new(env, fn {k, v} -> {k, if(MapSet.member?(seq, k), do: slice(v, 0), else: v)} end)

    with {:ok, dims} <- Compiled.dims(c, step_env) do
      opts = Keyword.put_new_lazy(opts, :threads, fn -> Worker.info(worker).threads end)
      {plan, decode} = plan(c, env, dims, seq, t_count, opts)

      on_emit =
        case opts[:on_emit] do
          nil -> nil
          f -> fn t, bytes -> f.(t, decode.(bytes)) end
        end

      case Worker.run(worker, plan, Keyword.put(opts, :on_emit, on_emit)) do
        {:ok, res} ->
          {:ok,
           %{outputs: decode_returns(c, dims, res.returns),
             steps: Enum.map(res.emits, fn {_t, b} -> decode.(b) end),
             elapsed_ns: res.elapsed_ns, retired: res.retired, counters: res.counters}}

        err ->
          err
      end
    end
  end

  @doc "The same computation on the oracle substrate (exact binary32, no hardware)."
  def run_oracle(%Compiled{} = c, env, opts \\ []) do
    t_count = Keyword.get(opts, :iterations, 1)
    seq = MapSet.new(Keyword.get(opts, :sequence, []))
    {steps, _state} =
      Enum.map_reduce(0..(t_count - 1), env, fn t, cur ->
        env_t = Map.new(cur, fn {k, v} -> {k, if(MapSet.member?(seq, k), do: slice(env[k], t), else: v)} end)
        outs = Oracle.eval_program(c.program, env_t, c.policy)
        next = Enum.reduce(c.program.state, cur, fn {in_n, out_n}, acc -> Map.put(acc, in_n, outs[out_n]) end)
        if f = opts[:on_emit], do: f.(t, outs)
        {outs, next}
      end)

    {:ok, %{outputs: List.last(steps), steps: steps, elapsed_ns: nil, retired: nil}}
  end

  # ------------------------------------------------------------- planning --

  defp plan(c, env, dims, seq, t_count, opts) do
    isa = Keyword.fetch!(opts, :isa)
    %{blob: blob, entries: entries} = Map.fetch!(c.code, isa)
    ids = c.slots |> Map.keys() |> Enum.sort()
    index = ids |> Enum.with_index() |> Map.new()
    state_in = MapSet.new(Enum.map(c.state, &elem(&1, 0)))

    scratch = scratch_bytes(c, Keyword.get(opts, :threads, 1))

    buffers =
      Enum.map(ids, fn id ->
        slot = c.slots[id]
        len = max(Compiled.nbytes(c, id, dims), Map.get(scratch, id, 0))

        case slot.role do
          {:input, name} ->
            %Tensor{data: d} = Map.fetch!(env, name)
            writable = MapSet.member?(state_in, id) or MapSet.member?(c.inplace, id)
            if MapSet.member?(seq, name), do: %{kind: :inline, data: d, len: byte_size(d)},
                                          else: %{kind: :inline, data: d, len: len, writable: writable}

          {:const, %Tensor{data: d}} ->
            const_buffer(d)

          :tmp ->
            %{kind: :zero, len: len, writable: true}
        end
      end)

    seq_slot = fn id ->
      case c.slots[id].role do
        {:input, name} -> MapSet.member?(seq, name)
        _ -> false
      end
    end

    calls = calls(c, dims, entries, index, seq_slot)

    out_ids = Enum.map(c.outputs, &elem(&1, 1))
    stream? = Keyword.get(opts, :stream, true) and t_count > 1
    emits = if stream?, do: Enum.map(out_ids, &{index[&1], 0, 0, Compiled.nbytes(c, &1, dims)}), else: []

    # state updated in place (in_id == out_id) needs no copy between iterations
    copies =
      for {in_id, out_id} <- c.state, in_id != out_id do
        {index[out_id], 0, index[in_id], 0, Compiled.nbytes(c, out_id, dims)}
      end

    plan = %Plan{code: blob, entries: entries, buffers: buffers, calls: calls, iterations: t_count,
                 emits: emits, copies: copies, returns: Enum.map(out_ids, &index[&1])}

    decode = fn bytes -> split_outputs(c, dims, bytes) end
    {plan, decode}
  end

  @doc false
  # the schedule with symbolic arguments resolved under `dims`
  def calls(c, dims, entries, index, seq_slot \\ fn _ -> false end) do
    Enum.map(c.schedule, fn %{kernel: key, args: args} = e ->
      %{entry: Map.fetch!(entries, key),
        split: split(c, dims, Map.get(e, :split)),
        args: Enum.map(args, fn a ->
          case Compiled.resolve_arg(c, a, dims) do
            {:slot, id} ->
              if seq_slot.(id), do: {:iter, index[id], 0, Compiled.nbytes(c, id, dims)}, else: {:buf, index[id], 0}

            imm ->
              imm
          end
        end)}
    end)
  end

  defp split(_c, _dims, nil), do: nil
  defp split(c, dims, list) when is_list(list), do: Enum.map(list, &split(c, dims, &1))

  defp split(c, dims, %{count: i, grain: g, ptrs: ptrs, scratch: scr} = sp) do
    stride = fn
      {:bytes, b} -> b
      {:last_bytes, id} -> List.last(Compiled.shape(c, id, dims)) * Tensor.elem_bytes(c.slots[id].dtype)
    end

    %{count: i, grain: g, ptrs: Enum.map(ptrs, fn {a, st} -> {a, stride.(st)} end), scratch: scr, guard: Map.get(sp, :guard)}
  end

  @doc false
  # per-thread scratch: slots named by a partition descriptor are sized for
  # every thread of the worker's pool
  def scratch_bytes(c, threads) do
    for %{args: args, split: sp} <- c.schedule, sp != nil, %{scratch: scr} <- List.wrap(sp),
        {a, bytes} <- scr, {:slot, id} = Enum.at(args, a),
        reduce: %{} do
      acc -> Map.update(acc, id, bytes * threads, &max(&1, bytes * threads))
    end
  end

  @doc false
  def const_buffer(d) when byte_size(d) >= @shm_threshold do
    case Vapor.Runtime.Shm.put(d) do
      {:ok, path} -> %{kind: :file, path: path, offset: 0, len: byte_size(d)}
      :unavailable -> %{kind: :inline, data: d, len: byte_size(d)}
    end
  end

  def const_buffer(d), do: %{kind: :inline, data: d, len: byte_size(d)}

  defp decode_returns(c, dims, returns) do
    c.outputs
    |> Enum.zip(returns)
    |> Map.new(fn {{name, id}, bin} -> {name, tensor(c, id, dims, bin)} end)
  end

  defp split_outputs(c, dims, bytes) do
    {m, <<>>} =
      Enum.reduce(c.outputs, {%{}, bytes}, fn {name, id}, {m, rest} ->
        n = Compiled.nbytes(c, id, dims)
        <<b::binary-size(n), rest::binary>> = rest
        {Map.put(m, name, tensor(c, id, dims, b)), rest}
      end)

    m
  end

  defp tensor(c, id, dims, bin), do: Tensor.new(c.slots[id].dtype, Compiled.shape(c, id, dims), bin)

  @doc "Slice `t` of a sequence tensor (leading dimension)."
  def slice(%Tensor{shape: [n | rest]} = x, t) when t < n do
    sz = div(byte_size(x.data), n)
    Tensor.new(x.dtype, rest, binary_part(x.data, t * sz, sz))
  end
end
