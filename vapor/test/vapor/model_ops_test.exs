defmodule Vapor.ModelOpsTest do
  @moduledoc """
  Model operators (phase P1b): embedding gather, RoPE, KV-cache writes and
  grouped-query attention — bit-identical on every CPU substrate, batch
  invariant (a prompt processed at once equals the same tokens decoded one
  by one), total on out-of-range indices, and certified by the ladder.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Runtime.{Native, Worker}
  import Vapor.TestHelpers

  defp lower(p) do
    {:ok, c} = Lower.lower(p)
    c
  end

  defp decode_env(n) do
    env = attention_env(n)
    %{env | tok: Tensor.new(:s32, [n, 1], env.tok.data), pos: Tensor.new(:s32, [n, 1], env.pos.data)}
  end

  test "the attention block lowers with both caches updated in place" do
    c = lower(attention_block())
    assert MapSet.size(c.inplace) == 2
    # every operator has a SPIR-V module: the fabric admits the block
    assert Vapor.Compiled.fabric_complete?(c)

    # reading the old cache elsewhere forces the copying form
    kc = T.input(:k, :f32, [8, 16])
    p = Program.new(old: T.mul(kc, T.splat(1.0)), new: T.kv_write(kc, T.input(:pos, :s32, [1]), T.input(:r, :f32, [1, 16])))
    c2 = lower(p)
    assert MapSet.size(c2.inplace) == 0
    assert Enum.any?(c2.schedule, &(&1.kernel == {:kv_write, :copy}))
  end

  test "rejections: head geometry, table shapes, index sorts" do
    x = T.input(:x, :f32, [2, 24])
    pos = T.input(:pos, :s32, [2])
    kv = T.input(:k, :f32, [8, 24])
    assert {:error, _} = Program.check(Program.new(y: T.attention(x, kv, kv, pos, 2, 1)))
    assert {:error, _} = Program.check(Program.new(y: T.rope(x, T.input(:c, :f32, [8, 5]), T.input(:s, :f32, [8, 5]), pos, 2)))
    assert {:error, _} = Program.check(Program.new(y: T.gather_row(kv, T.input(:i, :f32, [2]))))
    assert {:error, _} = Program.check(Program.new(y: T.kv_write(kv, T.input(:p, :s32, [3]), x)))
  end

  test "out-of-range indices are total and defined: clamp (gather, rope, attention), skip (cache write)" do
    table = Tensor.random(:f32, [5, 16], 1)
    idx = Tensor.from_list(:s32, [3], [2, 99, -1])
    c = lower(Program.new(y: T.gather_row(T.const(table), T.input(:i, :s32, [3]))))
    {:ok, %{outputs: %{y: y}}} = Native.run_oracle(c, %{i: idx})
    rows = y.data |> :binary.bin_to_list() |> Enum.chunk_every(64)
    assert Enum.at(rows, 1) == :binary.bin_to_list(Tensor.row(table, 4))
    assert Enum.at(rows, 2) == :binary.bin_to_list(Tensor.row(table, 4))

    cache = Tensor.random(:f32, [4, 16], 2)
    w = lower(Program.new(y: T.kv_write(T.input(:c, :f32, [4, 16]), T.input(:p, :s32, [1]), T.input(:r, :f32, [1, 16]))))
    {:ok, %{outputs: %{y: kept}}} = Native.run_oracle(w, %{c: cache, p: Tensor.from_list(:s32, [1], [4]), r: Tensor.random(:f32, [1, 16], 3)})
    assert kept.data == cache.data
  end

  @tag :native
  test "prefill and recurrent decode: host and RVV interpreter bit-identical to the oracle, and to each other" do
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    n = 11

    pre = lower(attention_block())
    env = attention_env(n)
    {:ok, ref} = Native.run_oracle(pre, env)
    for isa <- Vapor.Runtime.Substrates.host_isas() do
      {:ok, got} = Native.run(w, pre, env, isa: isa, mode: :native)
      assert got.outputs == ref.outputs, inspect(isa)
    end

    for vlen <- [128, 512] do
      {:ok, emu} = Native.run(w, pre, env, isa: :riscv64, mode: :emulate, poison: true, vlen: vlen)
      assert emu.outputs == ref.outputs, "interpreter, VLEN #{vlen}"
    end

    # decoding the same tokens one per iteration, caches fed back as state
    dec = lower(attention_block(t: 1, recurrent: true))
    denv = decode_env(n)
    run = [iterations: n, sequence: [:tok, :pos]]
    {:ok, dref} = Native.run_oracle(dec, denv, run)
    for isa <- Vapor.Runtime.Substrates.host_isas() do
      {:ok, dgot} = Native.run(w, dec, denv, [isa: isa, mode: :native] ++ run)
      assert dgot.steps == dref.steps, inspect(isa)
    end

    # batch invariance: row t of the prefill is bit-identical to decode step t
    prefill_rows = ref.outputs.y.data |> :binary.bin_to_list() |> Enum.chunk_every(64 * 4)
    decode_rows = Enum.map(dref.steps, &:binary.bin_to_list(&1.y.data))
    assert prefill_rows == decode_rows
    assert List.last(dref.steps).k_next == ref.outputs.k_next
  end

  @tag :qemu
  @tag timeout: 900_000
  test "AArch64 and RVV (VLEN 128, 256) under QEMU are bit-identical to the oracle" do
    c = lower(attention_block())
    env = attention_env(7)
    {:ok, ref} = Native.run_oracle(c, env)

    for {target, isa} <- [{:aarch64, :aarch64}, {{:riscv64, 128}, :riscv64}, {{:riscv64, 256}, :riscv64}] do
      {:ok, w} = Worker.start_link(exec: worker_exec(target))
      {:ok, got} = Native.run(w, c, env, isa: isa, mode: :native)
      assert got.outputs == ref.outputs, inspect(target)
    end
  end

  test "the ladder certifies the attention block (bit parity on every substrate present, the fabric included)" do
    assert {:ok, c} = Vapor.compile(attention_block(), probe_dims: %{t: 5})
    p = c.certificate.payload
    assert p.parity.bit_identical == :all_outputs
    assert (:fabric in p.parity.substrates) == Enum.any?(Vapor.Runtime.Substrates.list(), &(&1.kind == :fabric))
    assert p.parity.envelope_max_abs_error.y == :bit_parity
  end

  # ------------------------------------------------------ batched sb4 --

  defp qprog(x) do
    w = T.const(Vapor.Quant.Sb4.quantize(Tensor.random(:f32, [40, 512], 31, scale: 0.3)))
    Program.new(y: T.qgemv(w, x))
  end

  @tag :native
  test "batched 4-bit GEMV: row i of W·X equals W·x_i (oracle), on every CPU substrate" do
    b = 5
    xs = Tensor.random(:f32, [b, 512], 32)
    batched = lower(qprog(T.input(:x, :f32, [T.dyn(:b, 8), 512])))
    single = lower(qprog(T.input(:x, :f32, [512])))

    {:ok, %{outputs: %{y: y}}} = Native.run_oracle(batched, %{x: xs})
    assert y.shape == [b, 40]

    for i <- 0..(b - 1) do
      xi = Tensor.new(:f32, [512], Tensor.row(xs, i))
      {:ok, %{outputs: %{y: yi}}} = Native.run_oracle(single, %{x: xi})
      assert Tensor.row(y, i) == yi.data
    end

    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    for isa <- Vapor.Runtime.Substrates.host_isas() do
      {:ok, got} = Native.run(w, batched, %{x: xs}, isa: isa, mode: :native)
      assert got.outputs.y == y, inspect(isa)
    end

    for vlen <- [128, 512] do
      {:ok, emu} = Native.run(w, batched, %{x: xs}, isa: :riscv64, mode: :emulate, poison: true, vlen: vlen)
      assert emu.outputs.y == y
    end
  end

  @tag :vulkan
  test "batched 4-bit GEMV on the Vulkan fabric is bit-identical to the oracle" do
    fabric = Enum.find(Vapor.Runtime.Substrates.list(), &(&1.kind == :fabric))
    c = lower(qprog(T.input(:x, :f32, [T.dyn(:b, 8), 512])))
    xs = Tensor.random(:f32, [3, 512], 33)
    {:ok, ref} = Native.run_oracle(c, %{x: xs})
    {:ok, got} = Vapor.Runtime.Dispatch.run_on(fabric, c, %{x: xs}, [])
    assert got.outputs == ref.outputs
  end

  # ----------------------------------------------------------- paged KV --

  defp pool(rows), do: Tensor.from_list(:f32, [rows, 32], List.duplicate(0.0, rows * 32))

  @tag :native
  test "paged KV: permuted pages and two interleaved sequences = contiguous caches, bit for bit, any thread count" do
    contig = lower(attention_block())
    paged = lower(attention_block(paged: {8, 10, 2}))
    assert MapSet.size(paged.inplace) == 2

    # sequence A: 10 tokens at 0..9; sequence B: 6 tokens at 0..5
    a = attention_env(10)
    b = %{attention_env(6) | tok: Tensor.random(:s32, [6], 77, max: 97)}
    {:ok, ra} = Native.run_oracle(contig, a)
    {:ok, rb} = Native.run_oracle(contig, b)

    # A owns pages 7, 2, 9, 0; B owns 4, 5, 1, 3 — rows interleaved in one batch
    table = Tensor.from_list(:s32, [2, 4], [7, 2, 9, 0, 4, 5, 1, 3])
    order = Enum.map(0..9, &{:a, &1}) |> Enum.zip_with(Enum.map(0..5, &{:b, &1}) ++ List.duplicate(nil, 4), &[&1, &2]) |> List.flatten() |> Enum.reject(&is_nil/1)
    pick = fn {sq, i}, key -> Tensor.to_list(if(sq == :a, do: a, else: b)[key]) |> Enum.at(i) end
    n = length(order)

    env = %{tok: Tensor.from_list(:s32, [n], Enum.map(order, &pick.(&1, :tok))),
            pos: Tensor.from_list(:s32, [n], Enum.map(order, &pick.(&1, :pos))),
            slot: Tensor.from_list(:s32, [n], Enum.map(order, fn {sq, _} -> if sq == :a, do: 0, else: 1 end)),
            table: table, k: pool(80), v: pool(80)}

    {:ok, ref} = Native.run_oracle(paged, env)
    rows = Enum.chunk_every(:binary.bin_to_list(ref.outputs.y.data), 64 * 4)

    for {{sq, i}, row} <- Enum.zip(order, rows) do
      want = Tensor.row(if(sq == :a, do: ra, else: rb).outputs.y, i)
      assert :binary.list_to_bin(row) == want, "#{sq} row #{i}"
    end

    for threads <- [1, 3] do
      {:ok, w} = Worker.start_link(exec: worker_exec(:host), threads: threads)
      for isa <- Vapor.Runtime.Substrates.host_isas() do
        {:ok, got} = Native.run(w, paged, env, isa: isa, mode: :native)
        assert got.outputs == ref.outputs, "host, #{threads} threads (#{isa})"
      end
      {:ok, emu} = Native.run(w, paged, env, isa: :riscv64, mode: :emulate, poison: true, vlen: 256)
      assert emu.outputs == ref.outputs, "RVV interpreter"
    end
  end

  test "paged KV: out-of-range slots, pages and positions are total (skipped writes, clamped reads)" do
    paged = lower(attention_block(paged: {8, 4, 1}))
    env = %{tok: Tensor.from_list(:s32, [3], [1, 2, 3]), pos: Tensor.from_list(:s32, [3], [0, 40, 1]),
            slot: Tensor.from_list(:s32, [3], [0, 0, 9]), table: Tensor.from_list(:s32, [1, 4], [3, 99, 0, 1]),
            k: pool(32), v: pool(32)}
    assert {:ok, %{outputs: %{y: y}}} = Native.run_oracle(paged, env)
    assert y.shape == [3, 64]
  end

  test "a paged pool must be updated in place" do
    kp = T.input(:k, :f32, [16, 16])
    tb = T.input(:tb, :s32, [1, 2])
    i = T.input(:i, :s32, [1])
    p = Program.new(old: T.mul(kp, T.splat(1.0)), new: T.kv_write_paged(kp, tb, i, i, T.input(:r, :f32, [1, 16]), 8))
    assert {:error, %Vapor.Rejection{bound: "the paged pool is updated in place" <> _}} = Lower.lower(p)
  end

  @tag :qemu
  @tag timeout: 900_000
  test "paged KV under QEMU (AArch64, RVV VLEN 128) is bit-identical to the oracle" do
    c = lower(attention_block(paged: {8, 4, 1}))
    env = %{tok: Tensor.from_list(:s32, [5], [1, 2, 3, 4, 5]), pos: Tensor.from_list(:s32, [5], [0, 1, 2, 3, 4]),
            slot: Tensor.from_list(:s32, [5], [0, 0, 0, 0, 0]), table: Tensor.from_list(:s32, [1, 4], [2, 0, 3, 1]),
            k: pool(32), v: pool(32)}
    {:ok, ref} = Native.run_oracle(c, env)

    for {target, isa} <- [{:aarch64, :aarch64}, {{:riscv64, 128}, :riscv64}] do
      {:ok, w} = Worker.start_link(exec: worker_exec(target))
      {:ok, got} = Native.run(w, c, env, isa: isa, mode: :native)
      assert got.outputs == ref.outputs, inspect(target)
    end
  end

  # ------------------------------------------------------ sampling --

  @tag :native
  test "sampling on the substrate: greedy and categorical rows bit-identical to the oracle, any thread count" do
    v = 4096
    c = lower(Program.new(next: T.sample(T.input(:l, :f32, [T.dyn(:b, 8), v]), T.input(:p, :f32, [T.dyn(:b, 8), 2]))))
    logits = Tensor.random(:f32, [6, v], 90, scale: 6.0)
    # rows: greedy, T = 1, T = 0.5, T = 2, u near 0, u near 1
    params = Tensor.from_list(:f32, [6, 2], [0.0, 0.0, 1.0, 0.37, 2.0, 0.81, 0.5, 0.5, 1.0, 0.0, 1.0, 0.99999994])
    {:ok, ref} = Native.run_oracle(c, %{l: logits, p: params})
    [g | _] = Tensor.to_list(ref.outputs.next)
    assert g == logits |> Tensor.row(0) |> Vapor.Sampler.floats() |> Vapor.Sampler.argmax()

    for threads <- [1, 2] do
      {:ok, w} = Worker.start_link(exec: worker_exec(:host), threads: threads)
      for isa <- Vapor.Runtime.Substrates.host_isas() do
        {:ok, got} = Native.run(w, c, %{l: logits, p: params}, isa: isa, mode: :native)
        assert got.outputs == ref.outputs, inspect(isa)
      end
      {:ok, emu} = Native.run(w, c, %{l: logits, p: params}, isa: :riscv64, mode: :emulate, poison: true, vlen: 512)
      assert emu.outputs == ref.outputs
    end
  end
end
