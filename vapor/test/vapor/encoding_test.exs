defmodule Vapor.EncodingTest do
  @moduledoc "Bit-level encodings, pinned against GNU binutils' decoding of the same words."
  use ExUnit.Case, async: true
  alias Vapor.Emit.{ARM, RVV, X86}

  defp r, do: fn {:phys, n} -> n end
  defp p(n), do: {:phys, n}

  test "x86-64: VEX, REX, ModR/M, SIB" do
    assert X86.encode(:vfmadd231ps, [p(1), p(2), p(3)], r()) == <<0xC4, 0xE2, 0x6D, 0xB8, 0xCB>>
    assert X86.encode(:vpmovzxbd, [p(15), {:mem, p(5), 8}], r()) == <<0xC4, 0x62, 0x7D, 0x31, 0xBD, 8, 0, 0, 0>>
    assert X86.encode(:mov_load, [p(1), {:mem, p(12), 16}], r()) == <<0x49, 0x8B, 0x8C, 0x24, 16, 0, 0, 0>>
    assert X86.encode(:vextractf128, [p(15), p(2), 1], r()) == <<0xC4, 0xC3, 0x7D, 0x19, 0xD7, 1>>
    assert X86.epilogue(%{g: [3]}) == <<0x5B, 0xC5, 0xF8, 0x77, 0xC3>>
  end

  test "AArch64 / NEON" do
    assert ARM.encode(:fmla4s, [p(0), p(1), p(2)], r()) == <<0x4E22CC20::32-little>>
    assert ARM.encode(:fmul4s_elem, [p(20), p(20), p(16)], r()) == <<0x4F909294::32-little>>
    assert ARM.encode(:uxtl2_4s, [p(21), p(20)], r()) == <<0x6F10A695::32-little>>
    assert ARM.encode(:ldr_q, [p(31), p(9), 0], r()) == <<0x3DC0013F::32-little>>
    assert ARM.epilogue(%{}) == <<0xD65F03C0::32-little>>
  end

  test "RISC-V RVV 1.0: vtype layout vlmul[2:0] | vsew[5:3] | vta | vma" do
    assert RVV.vtypei({32, 4, :ta}) == 0b1101_0010
    assert RVV.encode(:vsetivli, [p(0), 16, RVV.vtypei({32, 4, :ta})], r()) == <<0xCD287057::32-little>>
    assert RVV.encode(:vzext_vf4, [p(8), p(2)], r()) == <<0x4A222457::32-little>>
    assert RVV.encode(:vwmul_vv, [p(16), p(2), p(4)], r()) == <<0xEE222857::32-little>>
    assert RVV.encode(:vfmv_f_s, [p(1), p(12)], r()) == <<0x42C010D7::32-little>>
  end

  test "RVV unit-stride loads live in LOAD-FP with a width field (not OP-V)" do
    <<w::32-little>> = RVV.encode(:vle8, [p(1), p(16)], r())
    assert Bitwise.band(w, 0x7F) == 0b0000111
    assert w == 0x02080087
  end

  test "RVV ret is jalr x0, 0(ra)" do
    assert binary_part(RVV.epilogue(%{}), 0, 4) == <<0x00008067::32-little>>
  end
end
