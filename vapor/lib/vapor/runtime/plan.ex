defmodule Vapor.Runtime.Plan do
  @moduledoc """
  A self-contained unit of native work — the only thing that crosses the
  process boundary to `vapor-worker`.

  Stateless by design: a plan carries its code, its buffers (inline, zeroed,
  or zero-copy file mappings for weights) and its call schedule, so a worker
  that crashed and was restarted needs no state replay before the unit is
  retried or rerouted.

  `iterations` repeats the call list; `{:iter, buf, base, stride}` arguments
  advance by `stride` bytes per iteration (token t reads its input at
  `base + t·stride`), `emits` stream slices back after every iteration and
  `copies` realise state feedback (`h ← h_next`) between iterations — the
  whole autoregressive loop runs behind one BEAM crossing.
  """
  @enforce_keys [:code, :buffers, :calls]
  defstruct code: <<>>, entries: %{}, buffers: [], calls: [], iterations: 1,
            emits: [], copies: [], returns: []

  @type arg :: {:imm, non_neg_integer} | {:buf, non_neg_integer, non_neg_integer}
               | {:iter, non_neg_integer, non_neg_integer, non_neg_integer}
  @type buffer :: %{required(:kind) => :zero | :inline | :file, required(:len) => non_neg_integer,
                    optional(:data) => binary, optional(:path) => String.t(),
                    optional(:offset) => non_neg_integer, optional(:writable) => boolean}
  @type t :: %__MODULE__{}

  @op_hello 1
  @op_run 2
  @op_open 6
  @op_step 7
  @op_close 8

  @doc """
  HELLO frame: bit 0 of flags requests the seccomp sandbox; `:threads` sizes
  the worker's thread pool (created before the filter is installed).
  """
  def hello(opts \\ []),
    do: <<@op_hello, if(Keyword.get(opts, :sandbox, true), do: 1, else: 0)::32-little,
          Keyword.get(opts, :threads, 1)::32-little>>

  @doc """
  Encode a RUN frame. Options: `:mode` (`:native` | `:emulate`), `:vlen`
  (emulated VLEN in bits), `:poison` (agnostic-tail poisoning),
  `:fuel` (emulated instruction budget, 0 = unbounded), `:deadline_ms`
  (in-worker SIGALRM watchdog for native code, 0 = none).
  """
  def encode(%__MODULE__{} = p, opts \\ []) do
    mode = if Keyword.get(opts, :mode, :native) == :native, do: 0, else: 1
    poison = if Keyword.get(opts, :poison, false), do: 1, else: 0

    IO.iodata_to_binary([
      <<@op_run, mode, Keyword.get(opts, :vlen, 128)::32-little, poison::32-little,
        Keyword.get(opts, :fuel, 0)::64-little, Keyword.get(opts, :deadline_ms, 0)::32-little>>,
      <<byte_size(p.code)::32-little>>, p.code,
      <<length(p.buffers)::32-little>>, Enum.map(p.buffers, &buffer/1),
      <<length(p.calls)::32-little>>, Enum.map(p.calls, &call/1),
      <<p.iterations::32-little>>,
      <<length(p.emits)::32-little>>,
      Enum.map(p.emits, fn {b, base, stride, len} -> <<b::32-little, base::64-little, stride::64-little, len::64-little>> end),
      <<length(p.copies)::32-little>>,
      Enum.map(p.copies, fn {s, so, d, dof, len} ->
        <<s::32-little, so::64-little, d::32-little, dof::64-little, len::64-little>>
      end),
      <<length(p.returns)::32-little>>, Enum.map(p.returns, &<<&1::32-little>>)
    ])
  end

  @doc """
  OPEN frame: a session's machine — code and buffers, kept by the worker
  until CLOSE or the next OPEN. Options: `:mode`, `:vlen`, `:poison`, `:fuel`.
  """
  def encode_open(code, buffers, opts \\ []) do
    mode = if Keyword.get(opts, :mode, :native) == :native, do: 0, else: 1
    poison = if Keyword.get(opts, :poison, false), do: 1, else: 0

    IO.iodata_to_binary([
      <<@op_open, mode, Keyword.get(opts, :vlen, 128)::32-little, poison::32-little, Keyword.get(opts, :fuel, 0)::64-little>>,
      <<byte_size(code)::32-little>>, code,
      <<length(buffers)::32-little>>, Enum.map(buffers, &buffer/1)
    ])
  end

  @doc """
  STEP frame: `writes` `[{buf, offset, bytes}]` into session buffers, one
  pass over `calls`, and `returns` `[{buf, offset, len}]` slices back;
  `copies` `[{src, soff, dst, doff, len}]` (state feedback after the pass,
  for state that is not updated in place — a recurrent model's) are
  appended only when there are any, so the frame of every other session is
  unchanged.
  """
  def encode_step(writes, calls, returns, deadline_ms, copies \\ []) do
    IO.iodata_to_binary([
      <<@op_step, deadline_ms::32-little, length(writes)::32-little>>,
      Enum.map(writes, fn {b, off, bytes} -> [<<b::32-little, off::64-little, byte_size(bytes)::64-little>>, bytes] end),
      <<length(calls)::32-little>>, Enum.map(calls, &call/1),
      <<length(returns)::32-little>>, Enum.map(returns, fn {b, off, len} -> <<b::32-little, off::64-little, len::64-little>> end),
      if(copies == [], do: [], else: [<<length(copies)::32-little>>, Enum.map(copies, fn {s, so, d, dof, len} ->
        <<s::32-little, so::64-little, d::32-little, dof::64-little, len::64-little>> end)])
    ])
  end

  @doc "CLOSE frame."
  def encode_close, do: <<@op_close>>

  defp buffer(%{kind: :zero, len: len} = b), do: <<0, w(b), len::64-little>>
  defp buffer(%{kind: :inline, data: d} = b), do: [<<1, w(b), byte_size(d)::64-little>>, d]

  defp buffer(%{kind: :file, path: path, len: len} = b),
    do: [<<2, w(b), len::64-little, byte_size(path)::16-little>>, path, <<Map.get(b, :offset, 0)::64-little>>]

  defp w(b), do: if(Map.get(b, :writable, false), do: 1, else: 0)

  defp call(%{entry: e, args: args} = c) do
    [<<e::32-little, length(args)::32-little>>,
     Enum.map(args, fn
       {:imm, v} -> <<0, v::64-little>>
       {:buf, i, off} -> <<1, i::32-little, off::64-little>>
       {:iter, i, base, stride} -> <<2, i::32-little, base::64-little, stride::64-little>>
     end),
     split(Map.get(c, :split))]
  end

  # 0, 1 or 2 partition descriptors; the worker uses the first applicable
  defp split(nil), do: <<0>>
  defp split(%{} = one), do: split([one])

  defp split(list) when is_list(list) do
    [<<length(list)>>,
     Enum.map(list, fn %{count: i, grain: g, ptrs: ptrs, scratch: scr} = sp ->
       [<<i, Map.get(sp, :guard) || 255, g::32-little, length(ptrs)>>, Enum.map(ptrs, fn {a, st} -> <<a, st::64-little>> end),
        <<length(scr)>>, Enum.map(scr, fn {a, b} -> <<a, b::64-little>> end)]
     end)]
  end

  # ------------------------------------------------------------- replies --

  @doc "Decode one reply frame from the worker."
  def decode(<<1, arch, sandbox, version::32-little, threads::32-little>>),
    do: {:hello, %{arch: arch_name(arch), sandbox: sandbox_name(sandbox), version: version, threads: threads}}

  def decode(<<3, t::32-little, bytes::binary>>), do: {:emit, t, bytes}

  def decode(<<4, elapsed::64-little, retired::64-little, n, rest::binary>>) do
    <<evs::binary-size(n * 9), rest::binary>> = rest
    counters = for <<id, v::64-little <- evs>>, into: %{}, do: {counter(id), v}
    {:done, %{elapsed_ns: elapsed, retired: retired, counters: counters, returns: returns(rest, [])}}
  end


  def decode(<<5, code::32-little, pc::64-little, word::32-little, msg::binary>>),
    do: {:error, %{code: err_name(code), pc: pc, word: word, message: msg}}

  defp counter(1), do: :cycles
  defp counter(2), do: :instructions
  defp counter(3), do: :cache_misses
  defp counter(4), do: :task_clock_ns
  defp counter(5), do: :page_faults
  defp counter(6), do: :context_switches
  defp counter(7), do: :gpu_recording_reused
  defp counter(8), do: :gpu_host_bytes
  defp counter(n), do: n

  defp returns(<<>>, acc), do: Enum.reverse(acc)
  defp returns(<<n::64-little, b::binary-size(n), rest::binary>>, acc), do: returns(rest, [b | acc])

  defp arch_name(1), do: :x86_64
  defp arch_name(2), do: :aarch64
  defp arch_name(3), do: :riscv64
  defp arch_name(_), do: :unknown
  defp sandbox_name(0), do: :off
  defp sandbox_name(1), do: :seccomp
  defp sandbox_name(_), do: :unsupported

  @errors {:ok, :illegal_instruction, :memory_fault, :fuel_exhausted, :misaligned_fetch,
            :bad_frame, :io, :unsupported}
  defp err_name(c) when c >= 0 and c < tuple_size(@errors), do: elem(@errors, c)
  defp err_name(_), do: :unknown
end
