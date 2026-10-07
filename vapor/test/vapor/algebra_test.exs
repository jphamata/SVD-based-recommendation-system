defmodule Vapor.AlgebraTest do
  use ExUnit.Case, async: true
  alias Vapor.{Program, Rejection, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Quant.Sb4

  test "rung 1 infers sorts through every generator, with dynamic extents" do
    x = T.input(:x, :f32, [T.dyn(:s, 128)])
    assert {:ok, {:f32, [{:dyn, :s, 128}]}} = T.infer(T.fma(x, x, T.splat(1.0)))
    a = T.input(:a, :s8, [T.dyn(:m, 64), 32])
    w = T.const(Tensor.random(:s8, [8, 32], 1))
    assert {:ok, {:s32, [{:dyn, :m, 64}, 8]}} = T.infer(T.gemm_i8(a, w))
  end

  test "rejections carry node, bound and repair (Axiom 1)" do
    x = T.input(:x, :f32, [4])
    y = T.input(:y, :f32, [5])
    assert {:error, %Rejection{bound: b, repair: r}} = T.infer(T.add(x, y))
    assert b =~ "broadcast" and is_binary(r)

    w = T.const(Sb4.quantize(Tensor.random(:f32, [4, 256], 1)))
    assert {:error, %Rejection{bound: "contraction axis" <> _}} = T.infer(T.qgemv(w, T.input(:x, :f32, [255])))
    assert {:error, %Rejection{}} = T.infer(T.ew(:frobnicate, [x]))
  end

  test "state feedback must preserve the sort" do
    h = T.input(:h, :f32, [4])
    good = Program.new([h_next: T.neg(h)], state: [h: :h_next])
    assert :ok = Program.check(good)
    bad = Program.new([h_next: T.neg(T.input(:z, :f32, [3]))], state: [h: :h_next])
    assert {:error, %Rejection{}} = Program.check(bad)
  end

  test "postorder is children-first and hash-consed" do
    x = T.input(:x, :f32, [4])
    s = T.add(x, x)
    order = T.postorder([T.mul(s, s)])
    assert order == [x, s, T.mul(s, s)]
  end

  test ":sb4 storage density is 4.6875 bit/w; execution layout 4.75 bit/w; repack is exact" do
    assert Sb4.bits_per_weight(:sb4) == 4.6875
    assert Sb4.bits_per_weight(:sb4x) == 4.75
    w = Sb4.quantize(Tensor.random(:f32, [3, 512], 7))
    assert byte_size(w.data) == 3 * 2 * 150
    x = Sb4.to_exec(w)
    assert byte_size(x.data) == 3 * 2 * 152

    # the execution form dequantises to exactly the storage semantics' α, β
    <<blk::binary-150, _::binary>> = w.data
    %{q: q, u: u, v: v, d: d, m0: m0, mr: mr} = Sb4.decode(blk)
    <<xblk::binary-152, _::binary>> = x.data
    {qx, ab} = Sb4.exec_block(xblk)
    assert Tuple.to_list(qx) == q
    [{a0, b0} | _] = ab
    alpha = Vapor.F32.mul(Vapor.F32.from_float(mr * d), Vapor.F32.from_float(hd(u)))
    assert a0 == alpha
    assert b0 == Vapor.F32.sub(Vapor.F32.from_float(mr * m0), Vapor.F32.mul(alpha, Vapor.F32.from_float(hd(v))))
  end
end
