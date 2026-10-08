defmodule Vapor.ModelTest do
  @moduledoc """
  Phase P2, models: `config.json` + weights → a vapor program of the
  Llama family (Llama, Mistral, Qwen2), bit-identical on every CPU
  substrate, batch invariant, certified by the ladder — and let-bindings,
  which keep the program's terms layer-sized however deep the model.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Program, Rejection, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Model.{Config, Decoder}
  alias Vapor.Runtime.{Native, Oracle, Substrates, Worker}
  import Vapor.TestHelpers

  defp model(arch, over \\ %{}, opts \\ []) do
    {:ok, c} = Config.from_map(tiny_config(arch, over))
    {:ok, p} = Decoder.program(c, tiny_weights(c), [max_seq: 16] ++ opts)
    {c, p}
  end

  defp env(c, toks, s \\ 16) do
    n = length(toks)
    Map.merge(Decoder.empty_caches(c, s), %{tok: Tensor.from_list(:s32, [n], toks), pos: Tensor.from_list(:s32, [n], Enum.to_list(0..(n - 1)))})
  end

  defp lower!(p) do
    {:ok, c} = Lower.lower(p)
    c
  end

  # size of a term as a tree (no sharing): what hashing or copying it costs
  defp tree_size(t), do: 1 + (t |> T.children() |> Enum.map(&tree_size/1) |> Enum.sum())

  # ------------------------------------------------------------ bindings --

  test "let-bindings: terms stay layer-sized, so depth costs linear time" do
    {_, p2} = model("llama", %{"num_hidden_layers" => 2})
    {_, p12} = model("llama", %{"num_hidden_layers" => 12})

    # the enumerated DAG grows linearly …
    n2 = length(Program.order(p2))
    n12 = length(Program.order(p12))
    assert n12 - n2 == 5 * div(n12 - n2, 5) and n12 < 7 * n2

    # … and no single term grows with depth (without bindings the residual
    # stream makes the unshared tree of an L-layer model grow like k^L)
    biggest = &(&1 |> Program.roots() |> Enum.map(fn t -> tree_size(t) end) |> Enum.max())
    assert biggest.(p12) == biggest.(p2)

    {t, {:ok, c}} = :timer.tc(fn -> Lower.lower(p12) end)
    assert length(c.schedule) > 12 * 10
    assert t < 20_000_000
  end

  test "let-bindings are substitution: the bound program equals the inlined one" do
    x = T.input(:x, :f32, [3, 16])
    w = Tensor.random(:f32, [16, 16], 4)
    a = T.add(x, T.linear(x, T.const(w)))
    inlined = Program.new(y: T.mul(a, T.rsqrt(T.add(T.reduce(:sum, T.mul(a, a)), T.splat(1.0)))))

    wr = T.ref(:w, T.const(w))
    ar = T.ref(:a, T.add(x, T.linear(x, wr)))
    bound = Program.new([y: T.mul(ar, T.rsqrt(T.add(T.reduce(:sum, T.mul(ar, ar)), T.splat(1.0))))],
                        lets: [w: T.const(w), a: T.add(x, T.linear(x, wr))])

    envx = %{x: Tensor.random(:f32, [3, 16], 5)}
    assert :ok = Program.check(bound)
    assert Program.inputs(bound) == [x]
    assert Oracle.eval_program(bound, envx) == Oracle.eval_program(inlined, envx)
    # one kernel schedule either way: bindings do not cut fusion
    assert lower!(bound).regions == lower!(inlined).regions
  end

  test "Rung 1 checks bindings: order, sorts, names" do
    x = T.input(:x, :f32, [4])
    later = Program.new([y: T.ref(:b, x)], lets: [a: T.neg(T.ref(:b, x)), b: T.neg(x)])
    assert {:error, %Rejection{node: {:let, :a, _}}} = Program.check(later)

    wrong_sort = Program.new([y: T.input(:a, :f32, [5])], lets: [a: T.neg(x)])
    assert {:error, %Rejection{node: {:let, :y, _}}} = Program.check(wrong_sort)

    dup = Program.new([y: x], lets: [a: T.neg(x), a: T.relu(x)])
    assert {:error, %Rejection{node: :lets}} = Program.check(dup)
  end

  # --------------------------------------------------------------- models --

  test "missing or misshapen weights are refused by name" do
    {:ok, c} = Config.from_map(tiny_config("qwen2"))
    ws = tiny_weights(c)
    name = "model.layers.1.self_attn.k_proj.bias"

    assert {:error, %Rejection{node: {:weight, ^name}}} = Decoder.program(c, Map.delete(ws, name))
    bad = Map.put(ws, name, Tensor.random(:f32, [3], 1))
    assert {:error, %Rejection{node: {:weight, ^name}, bound: "f32[32], got f32[3]"}} = Decoder.program(c, bad)
  end

  # a window that binds is executed (it used to be refused): the program's
  # attention carries it, and a window the cache cannot exceed is left out
  test "sliding windows: a binding window is in the program, a non-binding one is not" do
    {:ok, c} = Config.from_map(tiny_config("mistral"))
    ws = tiny_weights(c)
    windows = fn p -> for {:attention, _, _, _, _, h} <- Vapor.Algebra.Term.postorder(Enum.map(p.outputs, &elem(&1, 1)) ++ Enum.map(p.lets, &elem(&1, 1))), uniq: true, do: Vapor.Algebra.Term.attention_window(h) end
    {:ok, p8} = Decoder.program(%{c | sliding_window: 8}, ws, max_seq: 16)
    {:ok, p32} = Decoder.program(%{c | sliding_window: 32}, ws, max_seq: 16)
    assert windows.(p8) == [8]
    assert windows.(p32) == [nil]
  end

  test "logits: :last computes exactly the last row of :all" do
    {c, all} = model("mistral")
    {_, last} = model("mistral", %{}, logits: :last)
    e = env(c, [5, 17, 3, 90])
    %{logits: la} = Oracle.eval_program(all, e)
    %{logits: ll} = Oracle.eval_program(last, Map.put(e, :last, Tensor.from_list(:s32, [1], [3])))
    assert ll.shape == [1, 96]
    assert ll.data == Tensor.row(la, 3)
  end

  @tag :native
  test "Llama (untied, biased), Mistral, Qwen2 (tied): host and RVV interpreter bit-identical to the oracle" do
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))

    for {arch, over} <- [{"llama", %{"attention_bias" => true}}, {"mistral", %{}}, {"qwen2", %{}}] do
      {c, p} = model(arch, over)
      comp = lower!(p)
      e = env(c, [1, 95, 7, 7, 42, 0])
      {:ok, ref} = Native.run_oracle(comp, e)
      for isa <- Vapor.Runtime.Substrates.host_isas() do
        {:ok, got} = Native.run(w, comp, e, isa: isa, mode: :native)
        assert got.outputs == ref.outputs, "#{arch} on the host (#{isa})"
      end

      for vlen <- [128, 256] do
        {:ok, emu} = Native.run(w, comp, e, isa: :riscv64, mode: :emulate, poison: true, vlen: vlen)
        assert emu.outputs == ref.outputs, "#{arch} on the RVV interpreter, VLEN #{vlen}"
      end
    end
  end

  @tag :qemu
  @tag timeout: 900_000
  test "a whole model under QEMU (AArch64, RVV VLEN 128 and 512) is bit-identical to the oracle" do
    {c, p} = model("qwen2")
    comp = lower!(p)
    e = env(c, [9, 8, 7, 6])
    {:ok, ref} = Native.run_oracle(comp, e)

    for {target, isa} <- [{:aarch64, :aarch64}, {{:riscv64, 128}, :riscv64}, {{:riscv64, 512}, :riscv64}] do
      {:ok, w} = Worker.start_link(exec: worker_exec(target))
      {:ok, got} = Native.run(w, comp, e, isa: isa, mode: :native)
      assert got.outputs == ref.outputs, inspect(target)
    end
  end

  @tag :native
  test "batch invariance: a prompt processed at once equals the same tokens decoded one by one" do
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    {c, pre} = model("qwen2")
    {_, dec} = model("qwen2", %{}, max_tokens: 1)
    toks = [3, 1, 4, 1, 5, 9, 2]
    n = length(toks)
    e = env(c, toks)

    {:ok, ref} = Native.run(w, lower!(pre), e, isa: Substrates.host_isa(), mode: :native)

    de = %{e | tok: Tensor.new(:s32, [n, 1], e.tok.data), pos: Tensor.new(:s32, [n, 1], e.pos.data)}
    run = [iterations: n, sequence: [:tok, :pos], isa: Substrates.host_isa(), mode: :native]
    {:ok, got} = Native.run(w, lower!(dec), de, run)

    rows = for i <- 0..(n - 1), do: Tensor.row(ref.outputs.logits, i)
    assert Enum.map(got.steps, & &1.logits.data) == rows
    assert List.last(got.steps).k1_next == ref.outputs.k1_next
  end

  @tag :vulkan
  test "whole models on the Vulkan fabric = the oracle: prefill, the engine's paged sampling step, recurrent decoding" do
    fabric = Enum.find(Substrates.list(), &(&1.kind == :fabric))
    run = fn comp, e, opts -> Vapor.Runtime.Dispatch.run_on(fabric, comp, e, opts) end

    for {arch, over} <- [{"llama", %{"attention_bias" => true}}, {"mistral", %{}}, {"qwen2", %{}}] do
      {c, p} = model(arch, over)
      comp = lower!(p)
      e = env(c, [1, 95, 7, 7, 42, 0])
      {:ok, ref} = Native.run_oracle(comp, e)
      assert {:ok, %{outputs: got}} = run.(comp, e, [])
      assert got == ref.outputs, arch
    end

    # the engine's step: paged pools, permuted pages, two interleaved
    # sequences, greedy and sampled rows chosen on the GPU
    {c, p} = model("qwen2", %{}, kv: {:paged, 4, 8, 2}, logits: :last, sample: true, max_tokens: 8)
    comp = lower!(p)
    pool = Tensor.new(:f32, [32, c.kv_heads * c.head_dim], :binary.copy(<<0::32>>, 32 * c.kv_heads * c.head_dim))
    ids = &Tensor.from_list(:s32, [length(&1)], &1)

    e =
      Map.merge(for(l <- 0..(c.layers - 1), n <- [:"k#{l}", :"v#{l}"], into: %{}, do: {n, pool}),
                %{tok: ids.([5, 9, 11, 3, 70]), pos: ids.([0, 1, 0, 1, 2]), slot: ids.([0, 0, 1, 1, 0]),
                  table: Tensor.from_list(:s32, [2, 4], [3, 1, 6, 0, 2, 5, 7, 4]), last: ids.([3, 4]),
                  sampling: Tensor.from_list(:f32, [2, 2], [0.0, 0.0, 1.0, 0.37])})

    {:ok, ref} = Native.run_oracle(comp, e)
    assert {:ok, %{outputs: got}} = run.(comp, e, [])
    assert got == ref.outputs

    # recurrent decoding: iterations on the device, caches fed back as state
    {c, dec} = model("llama", %{}, max_tokens: 1)
    toks = [3, 1, 4, 1, 5]
    n = length(toks)
    e = env(c, toks)
    de = %{e | tok: Tensor.new(:s32, [n, 1], e.tok.data), pos: Tensor.new(:s32, [n, 1], e.pos.data)}
    opts = [iterations: n, sequence: [:tok, :pos]]
    {:ok, ref} = Native.run_oracle(lower!(dec), de, opts)
    assert {:ok, got} = run.(lower!(dec), de, opts)
    assert got.steps == ref.steps
  end

  test "the ladder certifies a tiny model: bit parity on every CPU substrate" do
    {_, p} = model("llama", %{"num_hidden_layers" => 1})
    assert {:ok, c} = Vapor.compile(p, probe_dims: %{t: 3})
    pay = c.certificate.payload
    assert pay.parity.bit_identical == :all_outputs
    # every projection passed the adjoint identity, weights resolved through bindings
    assert length(pay.adjoint) == 8
  end

  # ------------------------------------------------------------ bf16 --

  @tag :native
  test "storage: :bf16 is the f32 program over bf16-rounded weights, bit for bit, at half the weight bytes" do
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))

    for arch <- ["llama", "qwen2"] do
      {:ok, c} = Config.from_map(tiny_config(arch, %{"attention_bias" => true}))
      ws = tiny_weights(c)
      {:ok, pb} = Decoder.program(c, ws, max_seq: 16, storage: :bf16)
      # the same values, rounded and widened back, in an f32 program
      rounded = Map.new(ws, fn {k, t} -> {k, if(length(t.shape) == 2, do: Tensor.widen(Tensor.to_bf16(t)), else: t)} end)
      {:ok, pf} = Decoder.program(c, rounded, max_seq: 16)
      {cb, cf} = {lower!(pb), lower!(pf)}

      bytes = fn comp -> for({_, %{role: {:const, t}}} <- comp.slots, do: byte_size(t.data)) |> Enum.sum() end
      assert bytes.(cb) < 0.6 * bytes.(cf)

      e = env(c, [1, 95, 7, 7, 42, 0])
      {:ok, ref} = Native.run_oracle(cf, e)
      assert {:ok, %{outputs: o}} = Native.run_oracle(cb, e)
      assert o == ref.outputs

      for isa <- Substrates.host_isas() do
        {:ok, got} = Native.run(w, cb, e, isa: isa, mode: :native)
        assert got.outputs == ref.outputs, "#{arch} #{isa}"
      end

      {:ok, emu} = Native.run(w, cb, e, isa: :riscv64, mode: :emulate, poison: true, vlen: 256)
      assert emu.outputs == ref.outputs
    end
  end

  # ------------------------------------------------------------ 4-bit --

  test "quantize: :sb4 needs contraction widths ≡ 0 mod 256" do
    {:ok, c} = Config.from_map(tiny_config("llama"))
    assert {:error, %Rejection{node: {:quantize, :sb4}}} = Decoder.program(c, tiny_weights(c), quantize: :sb4)
  end

  @tag :native
  test "a 4-bit model: every projection is a qgemv, bit-identical across substrates, batch invariant" do
    over = %{"hidden_size" => 256, "intermediate_size" => 512, "vocab_size" => 128}
    {c, pre} = model("qwen2", over, quantize: :sb4)
    {_, dec} = model("qwen2", over, quantize: :sb4, max_tokens: 1)

    kinds = pre |> Program.order() |> Enum.map(&elem(&1, 0)) |> Enum.frequencies()
    # 7 projections per layer and the head; nothing multiplies an f32 matrix
    assert kinds[:qgemv] == 2 * 7 + 1 and kinds[:linear] == nil

    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    toks = [5, 3, 127, 0, 64]
    n = length(toks)
    e = env(c, toks)
    comp = lower!(pre)
    {:ok, ref} = Native.run_oracle(comp, e)
    for isa <- Vapor.Runtime.Substrates.host_isas() do
      {:ok, got} = Native.run(w, comp, e, isa: isa, mode: :native)
      assert got.outputs == ref.outputs, inspect(isa)
    end
    {:ok, emu} = Native.run(w, comp, e, isa: :riscv64, mode: :emulate, poison: true, vlen: 256)
    assert emu.outputs == ref.outputs

    de = %{e | tok: Tensor.new(:s32, [n, 1], e.tok.data), pos: Tensor.new(:s32, [n, 1], e.pos.data)}
    for isa <- Vapor.Runtime.Substrates.host_isas() do
      {:ok, steps} = Native.run(w, lower!(dec), de, iterations: n, sequence: [:tok, :pos], isa: isa, mode: :native)
      assert Enum.map(steps.steps, & &1.logits.data) == for(i <- 0..(n - 1), do: Tensor.row(ref.outputs.logits, i))
    end
  end
end
