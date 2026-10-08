defmodule Vapor.SparseSb4Test do
  @moduledoc """
  Sparse 4-bit experts (open since 0.6: "the MoE in 4 bits still runs
  dense"). `qgemv_masked` is to `qgemv` what `linear_masked` is to
  `linear`: active rows run the dense kernel's very instructions — the same
  bits — and a masked row reads no weight superblock at all. So a
  quantized mixture of experts reads only the selected experts' weights,
  with output bits equal to the dense 4-bit program on every substrate.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Model.{Config, Decoder}
  alias Vapor.Quant.Sb4
  alias Vapor.Runtime.{Dispatch, Native, Oracle, Substrates, Worker}
  import Vapor.TestHelpers

  @s 16
  @toks [1, 95, 7, 7, 42, 0]
  @mixtral %{"hidden_size" => 256, "intermediate_size" => 256, "num_attention_heads" => 4, "num_key_value_heads" => 2,
             "num_local_experts" => 4, "num_experts_per_tok" => 2, "num_hidden_layers" => 2}
  @qwen3 %{"hidden_size" => 256, "intermediate_size" => 256, "num_attention_heads" => 4, "num_key_value_heads" => 2,
           "head_dim" => 64, "num_experts" => 6, "num_experts_per_tok" => 2, "moe_intermediate_size" => 256,
           "norm_topk_prob" => true, "mlp_only_layers" => [0]}

  defp lower!(p), do: elem(Lower.lower(p), 1)

  defp env(c, toks) do
    n = length(toks)
    Map.merge(Decoder.empty_caches(c, @s), %{tok: Tensor.from_list(:s32, [n], toks), pos: Tensor.from_list(:s32, [n], Enum.to_list(0..(n - 1)))})
  end

  defp programs(arch, over) do
    {:ok, c} = Config.from_map(tiny_config(arch, over))
    ws = tiny_weights(c, 3)
    {:ok, dense} = Decoder.program(c, ws, max_seq: @s, moe: :dense, quantize: :sb4)
    {:ok, sparse} = Decoder.program(c, ws, max_seq: @s, moe: :sparse, quantize: :sb4)
    {c, dense, sparse}
  end

  defp masked_case do
    w = Sb4.quantize(Tensor.random(:f32, [40, 512], 3, scale: 0.3))
    x = T.input(:x, :f32, [T.dyn(:t, 8), 512])
    m = T.input(:m, :f32, [T.dyn(:t, 8), 1])
    e = %{x: Tensor.random(:f32, [5, 512], 4), m: Tensor.from_list(:f32, [5, 1], [1.0, 0.0, -0.0, 2.0, 0.0])}
    {Program.new(y: T.qgemv_masked(T.const(w), x, m)), Program.new(y: T.qgemv(T.const(w), x)), e}
  end

  test "qgemv_masked: active rows are qgemv's bits, inactive rows +0 (either sign of a zero mask)" do
    {pm, pd, e} = masked_case()
    %{y: masked} = Oracle.eval_program(pm, e)
    %{y: dense} = Oracle.eval_program(pd, %{x: e.x})
    rows = fn t -> t.data |> :binary.bin_to_list() |> Enum.chunk_every(40 * 4) end
    zero = List.duplicate(0, 40 * 4)

    for {{mr, dr}, active} <- Enum.zip(Enum.zip(rows.(masked), rows.(dense)), [true, false, false, true, false]),
        do: assert(mr == if(active, do: dr, else: zero))

    # shape errors are refused at the term, not discovered in a kernel
    bad = T.qgemv_masked(T.const(Sb4.quantize(Tensor.random(:f32, [4, 256], 1))), T.input(:x, :f32, [3, 256]), T.input(:m, :f32, [2, 1]))
    assert {:error, %Vapor.Rejection{}} = T.infer(bad)
  end

  @tag :native
  test "the predicated 4-bit kernel: host ISAs, threads, the poisoned RVV interpreter and the fabric = the oracle" do
    {pm, _pd, e} = masked_case()
    c = lower!(pm)
    {:ok, ref} = Native.run_oracle(c, e)

    for threads <- [1, 2] do
      {:ok, wk} = Worker.start_link(exec: worker_exec(:host), threads: threads)
      for isa <- Substrates.host_isas(), do: assert(elem(Native.run(wk, c, e, isa: isa, mode: :native), 1).outputs == ref.outputs)
      {:ok, emu} = Native.run(wk, c, e, isa: :riscv64, mode: :emulate, poison: true, vlen: 256)
      assert emu.outputs == ref.outputs
    end

    if fabric = Enum.find(Substrates.list(), &(&1.kind == :fabric)),
      do: assert(elem(Dispatch.run_on(fabric, c, e, []), 1).outputs == ref.outputs)
  end

  test "a 4-bit MoE: moe: :sparse and :dense are the same function (oracle), and sparse uses the predicated kernel" do
    for {arch, over} <- [{"mixtral", @mixtral}, {"qwen3_moe", @qwen3}] do
      {c, dense, sparse} = programs(arch, over)
      assert Oracle.eval_program(dense, env(c, Enum.take(@toks, 3))) == Oracle.eval_program(sparse, env(c, Enum.take(@toks, 3))), arch
      kernels = lower!(sparse).schedule |> Enum.map(& &1.kernel) |> MapSet.new()
      assert :gemv_sb4_masked in kernels
      refute :gemv_sb4_masked in (lower!(dense).schedule |> Enum.map(& &1.kernel))
    end
  end

  @tag :native
  @tag timeout: 900_000
  test "sparse 4-bit dispatch: dense bits on every host ISA and the RVV interpreter, with fewer instructions retired" do
    {:ok, wk} = Worker.start_link(exec: worker_exec(:host))

    for {arch, over} <- [{"mixtral", @mixtral}, {"qwen3_moe", @qwen3}] do
      {c, dense, sparse} = programs(arch, over)
      e = env(c, Enum.take(@toks, 4))
      {cd, cs} = {lower!(dense), lower!(sparse)}
      {:ok, ref} = Native.run(wk, cd, e, isa: Substrates.host_isa(), mode: :native)

      for isa <- Substrates.host_isas(), comp <- [cd, cs],
          do: assert(elem(Native.run(wk, comp, e, isa: isa, mode: :native), 1).outputs == ref.outputs, "#{arch} #{isa}")

      {:ok, ed} = Native.run(wk, cd, e, isa: :riscv64, mode: :emulate, poison: true, vlen: 256)
      {:ok, es} = Native.run(wk, cs, e, isa: :riscv64, mode: :emulate, poison: true, vlen: 256)
      assert ed.outputs == ref.outputs and es.outputs == ref.outputs
      assert es.retired < ed.retired, "#{arch}: #{es.retired} ≥ #{ed.retired}"
    end
  end

  @tag :native
  test "a 4-bit expert no row selects may hold NaN scales: not one output bit changes" do
    {:ok, wk} = Worker.start_link(exec: worker_exec(:host))
    {:ok, c} = Config.from_map(tiny_config("mixtral", @mixtral))
    ws = tiny_weights(c, 3)
    {:ok, p} = Decoder.program(c, ws, max_seq: @s, quantize: :sb4)
    {:ok, ref} = Native.run(wk, lower!(p), env(c, [42]), isa: Substrates.host_isa(), mode: :native)

    ranks = for e <- 0..3, do: :"layers.0.rank#{e}"
    probe = %{p | outputs: p.outputs ++ Enum.map(ranks, &{&1, T.input(&1, :f32, [T.dyn(:t, @s), 1])})}
    {:ok, r} = Native.run(wk, lower!(probe), env(c, [42]), isa: Substrates.host_isa(), mode: :native)
    idle = for {n, e} <- Enum.with_index(ranks), hd(Tensor.to_floats(r.outputs[n])) >= 2, do: e
    assert length(idle) == 2

    # the idle experts' superblocks get NaN scales (c1 = c0 = NaN in the
    # execution form): any read of them would spread NaN into the output
    nan_sb = fn <<head::binary-144, _::binary-8>> -> head <> :binary.copy(<<0x7FC0_0000::32-little>>, 2) end
    idle? = fn name -> Enum.any?(idle, &String.contains?(Atom.to_string(name), "layers.0.block_sparse_moe.experts.#{&1}.")) end

    lets =
      Enum.map(p.lets, fn
        {name, {:const, %Tensor{dtype: :sb4} = t}} = l ->
          if idle?.(name) do
            x = Sb4.to_exec(t)
            {name, {:const, %{x | data: for(<<sb::binary-152 <- x.data>>, into: <<>>, do: nan_sb.(sb))}}}
          else
            l
          end

        l ->
          l
      end)

    assert Enum.count(lets, fn {n, _} -> idle?.(n) end) == 6
    pp = %{p | lets: lets}
    {:ok, got} = Native.run(wk, lower!(pp), env(c, [42]), isa: Substrates.host_isa(), mode: :native)
    assert got.outputs == ref.outputs
  end

  @tag :qemu
  @tag timeout: 1_800_000
  test "the predicated 4-bit kernel under QEMU: AArch64 NEON and RVV (VLEN 128, 512) = the oracle" do
    {pm, _pd, e} = masked_case()
    c = lower!(pm)
    {:ok, ref} = Native.run_oracle(c, e)

    for {target, isa} <- [{:aarch64, :aarch64}, {{:riscv64, 128}, :riscv64}, {{:riscv64, 512}, :riscv64}] do
      {:ok, w} = Worker.start_link(exec: worker_exec(target))
      assert elem(Native.run(w, c, e, isa: isa, mode: :native), 1).outputs == ref.outputs, inspect(target)
    end
  end
end
