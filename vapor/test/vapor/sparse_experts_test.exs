defmodule Vapor.SparseExpertsTest do
  @moduledoc """
  Sparse mixture-of-experts dispatch by **row predication**
  (`Vapor.Algebra.Term.linear_masked/3`): an expert's projections run only on
  the rows that selected it, inside the weight-stationary GEMV — no
  permutation, no gather/scatter, no data-dependent shape. The dense program
  stays the definition: the selected rows execute the very instructions of
  the dense kernel and `sel` discards the others either way, so `moe:
  :sparse` and `moe: :dense` produce the same bits on every substrate —
  checked here, not argued — while the sparse one reads only the selected
  experts' weights.

  Also the block-diagonal `linear_grouped/3` (MLA's per-head maps): bit for
  bit `g` separate `linear`s, on every substrate.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Model.{Config, Llama}
  alias Vapor.Runtime.{Dispatch, Native, Oracle, Substrates, Worker}
  import Vapor.TestHelpers

  @moe [
    {"qwen3_moe", %{"head_dim" => 16, "num_experts" => 6, "num_experts_per_tok" => 2, "moe_intermediate_size" => 32,
                    "norm_topk_prob" => true, "mlp_only_layers" => [0]}},
    {"mixtral", %{"num_local_experts" => 4, "num_experts_per_tok" => 2}},
    {"deepseek_v3", %{"q_lora_rank" => 32, "kv_lora_rank" => 32, "qk_nope_head_dim" => 16, "qk_rope_head_dim" => 16,
                      "v_head_dim" => 24, "n_routed_experts" => 8, "num_experts_per_tok" => 3, "moe_intermediate_size" => 32,
                      "n_shared_experts" => 1, "first_k_dense_replace" => 1, "n_group" => 4, "topk_group" => 2,
                      "routed_scaling_factor" => 2.5, "num_key_value_heads" => 4, "max_position_embeddings" => 64}}
  ]
  @s 16
  @toks [1, 95, 7, 7, 42, 0]

  defp env(c, toks) do
    n = length(toks)
    Map.merge(Llama.empty_caches(c, @s), %{tok: Tensor.from_list(:s32, [n], toks), pos: Tensor.from_list(:s32, [n], Enum.to_list(0..(n - 1)))})
  end

  defp programs(arch, over) do
    {:ok, c} = Config.from_map(tiny_config(arch, over))
    ws = tiny_weights(c, 3)
    {:ok, dense} = Llama.program(c, ws, max_seq: @s, moe: :dense)
    {:ok, sparse} = Llama.program(c, ws, max_seq: @s, moe: :sparse)
    {c, dense, sparse}
  end

  defp lower!(p) do
    {:ok, c} = Lower.lower(p)
    c
  end

  # ------------------------------------------------------------ kernels --

  test "linear_masked: active rows are linear's bits, inactive rows +0 (either sign of a zero mask)" do
    w = Tensor.random(:f32, [24, 32], 3)
    x = T.input(:x, :f32, [T.dyn(:t, 8), 32])
    m = T.input(:m, :f32, [T.dyn(:t, 8), 1])
    xv = Tensor.random(:f32, [5, 32], 4)
    mv = Tensor.from_list(:f32, [5, 1], [1.0, 0.0, -0.0, 2.0, 0.0])

    %{y: masked} = Oracle.eval_program(Program.new(y: T.linear_masked(x, T.const(w), m)), %{x: xv, m: mv})
    %{y: dense} = Oracle.eval_program(Program.new(y: T.linear(x, T.const(w))), %{x: xv})
    rows = fn t -> t.data |> :binary.bin_to_list() |> Enum.chunk_every(24 * 4) end
    zero = List.duplicate(0, 24 * 4)

    for {{mr, dr}, active} <- Enum.zip(Enum.zip(rows.(masked), rows.(dense)), [true, false, false, true, false]),
        do: assert(mr == if(active, do: dr, else: zero))
  end

  @tag :native
  test "masked and grouped GEMV: host ISAs, the poisoned RVV interpreter and the fabric = the oracle" do
    {:ok, wk} = Worker.start_link(exec: worker_exec(:host), threads: 2)
    fabric = Enum.find(Substrates.list(), &(&1.kind == :fabric))

    cases =
      for {g, n, k, b} <- [{3, 5, 32, 4}, {4, 16, 48, 1}, {2, 7, 16, 3}], dt <- [:f32, :bf16] do
        w = Tensor.random(:f32, [g * n, k], 7 + g, scale: 0.5)
        w = if dt == :bf16, do: Tensor.to_bf16(w), else: w
        p = Program.new(y: T.linear_grouped(T.input(:x, :f32, [T.dyn(:t, 8), g * k]), T.const(w), g))
        {p, %{x: Tensor.random(:f32, [b, g * k], 9 + n)}}
      end

    mp = Program.new(y: T.linear_masked(T.input(:x, :f32, [T.dyn(:t, 8), 32]), T.const(Tensor.random(:f32, [24, 32], 3)),
                                        T.input(:m, :f32, [T.dyn(:t, 8), 1])))
    cases = cases ++ [{mp, %{x: Tensor.random(:f32, [5, 32], 4), m: Tensor.from_list(:f32, [5, 1], [1.0, 0.0, -0.0, 2.0, 0.0])}}]

    for {p, e} <- cases do
      c = lower!(p)
      {:ok, ref} = Native.run_oracle(c, e)
      for isa <- Substrates.host_isas(), do: assert({:ok, %{outputs: o}} = Native.run(wk, c, e, isa: isa, mode: :native)) && assert(o == ref.outputs)
      {:ok, emu} = Native.run(wk, c, e, isa: :riscv64, mode: :emulate, poison: true, vlen: 256)
      assert emu.outputs == ref.outputs
      if fabric, do: assert(elem(Dispatch.run_on(fabric, c, e, []), 1).outputs == ref.outputs)
    end
  end

  test "linear_grouped = g separate linears, bit for bit (oracle)" do
    {g, n, k} = {3, 5, 32}
    w = Tensor.random(:f32, [g * n, k], 1)
    xv = Tensor.random(:f32, [2, g * k], 2)
    %{y: y} = Oracle.eval_program(Program.new(y: T.linear_grouped(T.input(:x, :f32, [2, g * k]), T.const(w), g)), %{x: xv})

    sep =
      for b <- 0..1, i <- 0..(g - 1), into: <<>> do
        xr = Tensor.new(:f32, [1, k], binary_part(xv.data, (b * g * k + i * k) * 4, k * 4))
        wr = Tensor.new(:f32, [n, k], binary_part(w.data, i * n * k * 4, n * k * 4))
        Oracle.eval_program(Program.new(y: T.linear(T.input(:x, :f32, [1, k]), T.const(wr))), %{x: xr}).y.data
      end

    assert y.data == sep
  end

  # -------------------------------------------------------------- models --

  test "moe: :sparse and :dense are the same function, bit for bit (oracle)" do
    for {arch, over} <- @moe do
      {c, dense, sparse} = programs(arch, over)
      assert Oracle.eval_program(dense, env(c, @toks)) == Oracle.eval_program(sparse, env(c, @toks)), arch
    end
  end

  @tag :native
  @tag timeout: 900_000
  test "sparse dispatch: same bits as dense on every host ISA and the RVV interpreter, with fewer instructions" do
    {:ok, wk} = Worker.start_link(exec: worker_exec(:host))

    for {arch, over} <- @moe do
      {c, dense, sparse} = programs(arch, over)
      e = env(c, @toks)
      {cd, cs} = {lower!(dense), lower!(sparse)}
      {:ok, ref} = Native.run_oracle(cd, e)

      for isa <- Substrates.host_isas(), comp <- [cd, cs] do
        {:ok, got} = Native.run(wk, comp, e, isa: isa, mode: :native)
        assert got.outputs == ref.outputs, "#{arch} on #{isa}"
      end

      # the interpreter counts retired instructions exactly: the predicated
      # experts must cost less, and the bits must not move
      {:ok, ed} = Native.run(wk, cd, e, isa: :riscv64, mode: :emulate, poison: true, vlen: 256)
      {:ok, es} = Native.run(wk, cs, e, isa: :riscv64, mode: :emulate, poison: true, vlen: 256)
      assert ed.outputs == ref.outputs and es.outputs == ref.outputs
      assert es.retired < ed.retired, "#{arch}: #{es.retired} ≥ #{ed.retired}"
    end
  end

  @tag :native
  test "sparse dispatch reads only the selected experts: a NaN expert no row selects changes nothing" do
    {:ok, wk} = Worker.start_link(exec: worker_exec(:host))
    {"mixtral", over} = Enum.at(@moe, 1)
    {:ok, c} = Config.from_map(tiny_config("mixtral", over))
    ws = tiny_weights(c, 3)
    {:ok, p} = Llama.program(c, ws, max_seq: @s)
    {:ok, ref} = Native.run(wk, lower!(p), env(c, [42]), isa: Substrates.host_isa(), mode: :native)

    # poison every expert of layer 0 except the two the token selects
    ranks = for e <- 0..3, do: :"layers.0.rank#{e}"
    probe = %{p | outputs: p.outputs ++ Enum.map(ranks, &{&1, T.input(&1, :f32, [T.dyn(:t, @s), 1])})}
    {:ok, r} = Native.run(wk, lower!(probe), env(c, [42]), isa: Substrates.host_isa(), mode: :native)
    idle = for {n, e} <- Enum.with_index(ranks), hd(Tensor.to_floats(r.outputs[n])) >= 2, do: e
    assert length(idle) == 2

    poisoned =
      Map.new(ws, fn {k, t} ->
        if Enum.any?(idle, &String.contains?(k, "layers.0.block_sparse_moe.experts.#{&1}.")),
          do: {k, Tensor.new(:f32, t.shape, :binary.copy(<<0x7FC0_0000::32-little>>, Enum.product(t.shape)))},
          else: {k, t}
      end)

    {:ok, pp} = Llama.program(c, poisoned, max_seq: @s)
    {:ok, got} = Native.run(wk, lower!(pp), env(c, [42]), isa: Substrates.host_isa(), mode: :native)
    assert got.outputs == ref.outputs
  end

  test "DeepSeek with one expert group (V2-Lite's n_group = 1): the group limit is the identity, same bits as all groups kept" do
    base = elem(List.last(@moe), 1)
    one = Map.merge(base, %{"n_group" => 1, "topk_group" => 1})
    all = Map.merge(base, %{"n_group" => 4, "topk_group" => 4})
    {c1, d1, s1} = programs("deepseek_v3", one)
    {_c4, _d4, s4} = programs("deepseek_v3", all)
    e = env(c1, @toks)
    r1 = Oracle.eval_program(s1, e)
    assert r1 == Oracle.eval_program(s4, e)
    assert r1 == Oracle.eval_program(d1, e)
  end

  @tag :vulkan
  @tag timeout: 900_000
  test "sparse experts on the Vulkan fabric = the oracle" do
    fabric = Enum.find(Substrates.list(), &(&1.kind == :fabric))

    for {arch, over} <- @moe do
      {c, _dense, sparse} = programs(arch, over)
      comp = lower!(sparse)
      e = env(c, Enum.take(@toks, 4))
      {:ok, ref} = Native.run_oracle(comp, e)
      assert elem(Dispatch.run_on(fabric, comp, e, []), 1).outputs == ref.outputs, arch
    end
  end
end
