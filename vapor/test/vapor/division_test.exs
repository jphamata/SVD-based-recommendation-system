defmodule Vapor.DivisionTest do
  @moduledoc """
  The canonical division is **correctly rounded**: `a/b` is the IEEE-754
  round-to-nearest-even quotient whenever the operands and the result are
  normal, with the GPU convention outside (subnormal inputs read as zero,
  results below 2⁻¹²⁶ flushed to ±0) and IEEE's specials — so `+ − × ÷`
  are all IEEE operations on every substrate, by construction from `+ − ×`
  and integer lane operations (`Vapor.Canon`, Markstein's correction with
  an exact Dekker residual, no FMA).

  The reference here is independent: an integer computation from the
  binary64 quotient (exact enough: double rounding is innocuous for
  division, 53 ≥ 2·24 + 2), and NumPy's binary32 division for exhaustive
  sweeps over every significand.
  """
  use ExUnit.Case, async: false
  import Bitwise
  alias Vapor.{Canon, F32, Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Runtime.{Dispatch, Native, Substrates, Worker}
  import Vapor.TestHelpers

  # RN-even of the quotient with an unbounded exponent; then FTZ/overflow
  def reference(a, b) do
    daz = fn x -> if (x &&& 0x7F80_0000) == 0, do: x &&& 0x8000_0000, else: x end
    {a, b} = {daz.(a), daz.(b)}
    s = bxor(a, b) &&& 0x8000_0000
    {aa, ab} = {a &&& 0x7FFF_FFFF, b &&& 0x7FFF_FFFF}

    cond do
      aa > 0x7F80_0000 or ab > 0x7F80_0000 -> 0x7FC0_0000
      aa == 0 and ab == 0 -> 0x7FC0_0000
      aa == 0x7F80_0000 and ab == 0x7F80_0000 -> 0x7FC0_0000
      aa == 0x7F80_0000 or ab == 0 -> s ||| 0x7F80_0000
      aa == 0 or ab == 0x7F80_0000 -> s
      true ->
        <<fa::float-32>> = <<aa::32>>
        <<fb::float-32>> = <<ab::32>>
        <<0::1, e64::11, m64::52>> = <<fa / fb::float-64>>
        sig = (1 <<< 52) ||| m64
        {hi, rest} = {sig >>> 29, sig &&& ((1 <<< 29) - 1)}

        hi =
          cond do
            rest > 1 <<< 28 -> hi + 1
            rest < 1 <<< 28 -> hi
            (hi &&& 1) == 1 -> hi + 1
            true -> hi
          end

        {hi, e} = if hi == 1 <<< 24, do: {1 <<< 23, e64 - 1022}, else: {hi, e64 - 1023}

        cond do
          e + 127 <= 0 -> s
          e + 127 >= 255 -> s ||| 0x7F80_0000
          true -> s ||| ((e + 127) <<< 23) ||| (hi &&& 0x7F_FFFF)
        end
    end
  end

  @edge [0, 0x8000_0000, 1, 0x007F_FFFF, 0x0080_0000, 0x0080_0001, 0x3F80_0000, 0x3F80_0001, 0x3FFF_FFFF,
         0x7F7F_FFFF, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0000, 0x7F80_0001, 0x4000_0000, 0x3F7F_FFFF, 0xBF80_0000,
         0x7F00_0000, 0x0100_0000, 0x4B40_0000, 0x3EAA_AAAB]

  defp pairs(n, seed) do
    :rand.seed(:exsss, {seed, seed + 1, seed + 2})
    rnd = fn -> :rand.uniform(0x1_0000_0000) - 1 end
    sig = fn -> (rnd.() &&& 0x807F_FFFF) ||| 0x3F80_0000 end
    (for a <- @edge, b <- @edge, do: {a, b}) ++
      for(_ <- 1..n, do: {rnd.(), rnd.()}) ++ for(_ <- 1..n, do: {sig.(), sig.()}) ++
      # quotients near the ends of the normal range
      for(_ <- 1..div(n, 4), do: {(rnd.() &&& 0x80FF_FFFF) ||| 0x0080_0000, (rnd.() &&& 0x807F_FFFF) ||| 0x7E80_0000}) ++
      for(_ <- 1..div(n, 4), do: {(rnd.() &&& 0x807F_FFFF) ||| 0x7F00_0000, (rnd.() &&& 0x80FF_FFFF) ||| 0x0080_0000})
  end

  test "the microprogram is IEEE round-to-nearest division (with DAZ/FTZ) on 100 000 pairs and every special" do
    div = Canon.compile(:div)
    bad = for {a, b} <- pairs(40_000, 1), div.([a, b]) != reference(a, b), do: {a, b}
    assert bad == []
  end

  test "the old a·rcp(b) was not: it misrounds about a fifth of random quotients" do
    rcp = Canon.compile(:rcp)
    ps = for {a, b} <- pairs(5_000, 2), r = reference(a, b), (r &&& 0x7F80_0000) not in [0, 0x7F80_0000], do: {a, b}
    wrong = Enum.count(ps, fn {a, b} -> F32.mul(a, rcp.([b])) != reference(a, b) end)
    assert wrong / length(ps) > 0.1
  end

  defp div_program(n), do: Program.new(y: T.divide(T.input(:a, :f32, [n]), T.input(:b, :f32, [n])))

  @tag :native
  @tag timeout: 900_000
  test "every substrate divides alike: host ISAs, RVV interpreter, fabric = the microprogram = IEEE" do
    {:ok, w} = Worker.start_link(exec: worker_exec(:host), threads: 2)
    ps = pairs(20_000, 3)
    n = length(ps)
    {as, bs} = Enum.unzip(ps)
    e = %{a: Tensor.new(:f32, [n], F32.encode(as)), b: Tensor.new(:f32, [n], F32.encode(bs))}
    want = Tensor.new(:f32, [n], F32.encode(Enum.map(ps, fn {a, b} -> reference(a, b) end)))
    {:ok, c} = Lower.lower(div_program(n))

    for isa <- Substrates.host_isas(), do: assert(elem(Native.run(w, c, e, isa: isa, mode: :native), 1).outputs.y == want, "#{isa}")
    {:ok, emu} = Native.run(w, c, e, isa: :riscv64, mode: :emulate, vlen: 256)
    assert emu.outputs.y == want

    case Enum.find(Substrates.list(), &(&1.kind == :fabric)) do
      nil -> :ok
      fabric -> assert elem(Dispatch.run_on(fabric, c, e, []), 1).outputs.y == want
    end
  end

  @tag :native
  @tag :python
  @tag timeout: 1_800_000
  test "exhaustive over the significand: all 2²³ dividends ÷ six divisors = NumPy's binary32 division" do
    {:ok, w} = Worker.start_link(exec: worker_exec(:host), threads: 2)
    n = 1 <<< 23
    a = Tensor.new(:f32, [n], for(m <- 0..(n - 1), into: <<>>, do: <<(0x3F80_0000 ||| m)::32-little>>))
    {:ok, c} = Lower.lower(Program.new(y: T.divide(T.input(:a, :f32, [n]), T.input(:b, :f32, [1]))))
    dir = Path.join(System.tmp_dir!(), "vapor-div-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)
    File.write!(Path.join(dir, "a.bin"), a.data)

    divisors = [0x3F80_0001, 0x3FC0_0000, 0x3FFF_FFFF, 0x3FA2_F983, 0x3F9E_3779, 0x3FB5_04F3]

    for b <- divisors do
      {:ok, r} = Native.run(w, c, %{a: a, b: Tensor.new(:f32, [1], <<b::32-little>>)}, isa: Substrates.host_isa(), mode: :native)
      File.write!(Path.join(dir, "y_#{b}.bin"), r.outputs.y.data)
    end

    out =
      py!("""
      import numpy as np, sys
      d = sys.argv[1]
      a = np.fromfile(d + "/a.bin", dtype=np.float32)
      bad = 0
      for b in [#{Enum.join(divisors, ", ")}]:
          y = np.fromfile(f"{d}/y_{b}.bin", dtype=np.uint32)
          ref = (a / np.array([b], dtype=np.uint32).view(np.float32)[0]).astype(np.float32).view(np.uint32)
          bad += int(np.sum(y != ref))
      print(bad)
      """, [dir])

    assert String.trim(out) == "0"
  end
end
