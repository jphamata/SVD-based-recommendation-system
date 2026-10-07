defmodule Vapor.F32Test do
  use ExUnit.Case, async: true
  import Bitwise, only: [<<<: 2]
  alias Vapor.F32

  test "signed zeros are preserved and follow IEEE 754 §6.3" do
    nz = 0x8000_0000
    assert F32.add(nz, 0) == 0, "(−0) + (+0) = +0 under round-to-nearest"
    assert F32.add(nz, nz) == nz
    assert F32.mul(nz, 0x3F80_0000) == nz
    assert F32.fma(nz, 0x3F80_0000, nz) == nz
    assert F32.fma(0x3F80_0000, 0x3F80_0000, 0xBF80_0000) == 0, "exact cancellation gives +0"
  end

  test "round_dyadic: ties to even, gradual underflow, overflow to infinity" do
    # 1 + 2^-24 is a tie between 1 and 1 + 2^-23: rounds to even (1.0)
    assert F32.round_dyadic((1 <<< 24) + 1, -24) == 0x3F80_0000
    # 1 + 3·2^-24 is a tie: rounds up to 1 + 2^-22
    assert F32.round_dyadic((1 <<< 24) + 3, -24) == 0x3F80_0002
    assert F32.round_dyadic(1, -149) == 1, "smallest subnormal"
    assert F32.round_dyadic(1, -150) == 0, "half the smallest subnormal ties to even (0)"
    assert F32.round_dyadic(3, -151) == 1
    assert F32.round_dyadic(1, 128) == 0x7F80_0000
    assert F32.round_dyadic(-1, 128) == 0xFF80_0000
  end

  test "fma rounds once (differs from mul-then-add where it must)" do
    a = F32.from_float(1.0 + :math.pow(2, -12))
    c = F32.neg(F32.from_float(1.0 + :math.pow(2, -11)))
    # a·a = 1 + 2^-11 + 2^-24; the separate product rounds the 2^-24 away
    assert F32.to_float(F32.fma(a, a, c)) == :math.pow(2, -24)
    assert F32.add(F32.mul(a, a), c) == 0
  end

  test "to_dyadic / round_dyadic round-trip every class of finite value" do
    for b <- [0, 1, 0x7F_FFFF, 0x80_0000, 0x3F80_0000, 0x7F7F_FFFF, 0x8000_0001, 0xFF7F_FFFF] do
      {m, e} = F32.to_dyadic(b)
      if m != 0, do: assert(F32.round_dyadic(m, e) == b)
    end
  end

  test "non-finite operands are refused: certificates cover finite executions" do
    assert_raise ArithmeticError, fn -> F32.add(0x7F80_0000, 0) end
  end
end
