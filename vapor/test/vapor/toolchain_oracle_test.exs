defmodule Vapor.ToolchainOracleTest do
  @moduledoc """
  External oracles used *only in tests*: GNU binutils must decode every
  emitted kernel without a single unknown word, and `spirv-val` must accept
  every SPIR-V module. The product itself never invokes them (Axiom 4).
  """
  use ExUnit.Case, async: true
  alias Vapor.KIR.Kernels
  alias Vapor.Emit.{ARM, Machine, RVV, SpirvKernels, X86}
  alias Vapor.Emit.X86.AVX512

  @ew %{inputs: 2, outputs: [{:t, 2}, {:t, 0}],
        ops: [{{:t, 0}, :fma, [{:in, 0}, {:in, 1}, {:splat, 0x3F00_0000}]}, {{:t, 1}, :neg, [{:t, 0}]},
              {{:t, 2}, :relu, [{:t, 1}]}]}

  # every canonical function, every operand class and every primitive
  defp canon_ew do
    {ops, r1, k} = Vapor.Canon.expand(:silu, [{:in, 0}], 0)
    {ops2, r2, k} = Vapor.Canon.expand(:rsqrt, [{:in, 1}], k)
    {ops3, r3, k} = Vapor.Canon.expand(:div, [r1, {:in, 2}], k)
    {ops4, r4, _} = Vapor.Canon.expand(:max, [r3, {:in, 3}], k)
    %{inputs: [:full, :col, :row, :scalar], outputs: [r4, r2], ops: ops ++ ops2 ++ ops3 ++ ops4}
  end

  defp kernels,
    do: [Kernels.sb_sums(), Kernels.gemv_sb4(), Kernels.gemv_sb4_masked(), Kernels.gemm_i8(), Kernels.ew(@ew), Kernels.ew(canon_ew()),
         Kernels.reduce(:sum), Kernels.reduce(:max), Kernels.gemv_f32(), Kernels.gemv_bf16(), Kernels.gather_row(),
         Kernels.gather_row_bf16(), Kernels.rope(),
         Kernels.kv_write(:inplace), Kernels.kv_write(:copy), Kernels.kv_write_paged(4), Kernels.attention(0x3E00_0000),
         Kernels.attention(0x3E00_0000, {:paged, 4}), Kernels.sample(), Kernels.transpose()]

  defp tmp(name), do: Path.join(System.tmp_dir!(), "vapor-#{System.unique_integer([:positive])}-#{name}")

  defp disassemble(:x86_64_avx512, bin), do: disassemble(:x86_64, bin)

  defp disassemble(:x86_64, bin) do
    f = tmp("x.bin")
    File.write!(f, bin)
    {out, 0} = System.cmd("objdump", ["-D", "-b", "binary", "-m", "i386:x86-64", f])
    out
  end

  defp disassemble(:aarch64, bin) do
    f = tmp("a.bin")
    File.write!(f, bin)
    {out, 0} = System.cmd("aarch64-linux-gnu-objdump", ["-D", "-b", "binary", "-m", "aarch64", f])
    out
  end

  defp disassemble(:riscv64, bin) do
    s = tmp("r.s")
    o = tmp("r.o")
    insns = for <<w::32-little <- bin>>, do: ".insn 4, 0x#{Integer.to_string(w, 16)}\n"
    File.write!(s, [".text\n", insns])
    {_, 0} = System.cmd("riscv64-linux-gnu-as", ["-march=rv64gcv", "-o", o, s])
    {out, 0} = System.cmd("riscv64-linux-gnu-objdump", ["-d", o])
    out
  end

  @tag :binutils
  test "binutils decodes every emitted instruction on all four backends" do
    for k <- kernels(), be <- [X86, AVX512, ARM, RVV], pol <- [:canonical, :fast], g <- be.group_factors() do
      case Machine.compile(k, be, policy: pol, g: g) do
        {:ok, c} ->
          out = disassemble(c.isa, binary_part(c.bin, 0, c.text))
          refute out =~ "(bad)", "#{c.isa} #{inspect(k.name)} g=#{g}: undecodable x86"
          refute out =~ ~r/\.insn|\.word|udf|unknown/, "#{c.isa} #{inspect(k.name)} g=#{g}: unknown word\n#{out}"

        {:error, _} ->
          :ok
      end
    end
  end

  @tag :spirv_tools
  test "spirv-val accepts every SPIR-V module (incl. cooperative matrix and the model operators)" do
    for key <- [{:ew, @ew}, {:ew, canon_ew()}, {:reduce, :sum}, {:reduce, :max}, :gemv_f32,
                :sb_sums, :gemv_sb4, :gemv_sb4_masked, {:gemv_masked, :f32}, :gemm_i8, :gemm_i8_coop, :gather_row, :gather_row_bf16, :gemv_bf16, :rope, {:kv_write, :copy}, {:kv_write, :inplace},
                {:kv_write_paged, 4}, {:attention, 0x3E00_0000}, {:attention_paged, 0x3E00_0000, {:paged, 4}}, :sample,
                :transpose], pol <- [:canonical, :fast] do
      f = tmp("k.spv")
      File.write!(f, SpirvKernels.binary(key, pol))
      assert {_, 0} = System.cmd("spirv-val", ["--target-env", "vulkan1.3", f], stderr_to_stdout: true)
    end
  end
end
