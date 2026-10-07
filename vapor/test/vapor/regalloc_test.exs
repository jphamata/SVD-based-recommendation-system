defmodule Vapor.RegAllocTest do
  use ExUnit.Case, async: true
  alias Vapor.KIR.{Kernels, Liveness, RegAlloc}
  alias Vapor.Emit.{ARM, Machine, RVV, X86}

  defp iv(id, size, a, b), do: %{reg: {:vr, id, :v, size}, start: a, stop: b}
  @rvv_v %{v: %{count: 32, order: Enum.to_list(1..31), callee_saved: []}}

  defp nested(ps), do: for({a, b, p, s} <- ps, do: {a, {b, {p, s}}})

  test "LMUL groups are aligned and disjoint while live" do
    ivs = [iv(1, 4, 0, 10), iv(2, 2, 1, 5), iv(3, 8, 2, 9), iv(4, 1, 3, 4)]
    {:ok, %{assign: a}} = RegAlloc.allocate(ivs, @rvv_v)

    for %{reg: {:vr, _, _, s} = r} <- ivs, do: assert(rem(a[r], s) == 0 and a[r] >= 1)
    ps = RegAlloc.placements(ivs, a)[:v]
    assert Vapor.Extracted.check_alloc(32, [0], nested(ps))
  end

  test "pressure beyond the register file is reported, never spilled" do
    ivs = for i <- 1..4, do: iv(i, 8, 0, 10)
    assert {:error, %{need: 8, reg: {:vr, 4, :v, 8}}} = RegAlloc.allocate(ivs, @rvv_v)
  end

  test "early clobber: a widening destination never shares a source group" do
    code = [
      {:def_a, [{:vr, 1, :v, 1}], [], %{}},
      {:widen, [{:vr, 2, :v, 2}], [{:vr, 1, :v, 1}], %{ec: true}},
      {:use, [], [{:vr, 2, :v, 2}], %{}}
    ]

    ivs = Liveness.intervals(code)
    {:ok, %{assign: a}} = RegAlloc.allocate(ivs, @rvv_v)
    assert a[{:vr, 1, :v, 1}] not in a[{:vr, 2, :v, 2}]..(a[{:vr, 2, :v, 2}] + 1)
  end

  test "values live around a loop back-edge span the whole loop" do
    acc = {:vr, 1, :v, 1}
    tmp = {:vr, 2, :v, 1}

    code = [
      {:init, [acc], [], %{}},
      {:label, [], [], %{label: :l}},
      {:t, [tmp], [], %{}},
      {:acc, [acc], [acc, tmp], %{}},
      {:br, [], [], %{br: :l}},
      {:use, [], [acc], %{}}
    ]

    [a_iv] = Liveness.intervals(code) |> Enum.filter(&(&1.reg == acc))
    assert a_iv.start <= 2 and a_iv.stop >= 10
  end

  test "the verified checker rejects corrupted assignments" do
    ok = [{0, 10, 8, 4}, {2, 6, 12, 4}, {5, 9, 1, 1}]
    assert Vapor.Extracted.check_alloc(32, [0], nested(ok))
    refute Vapor.Extracted.check_alloc(32, [0], nested([{0, 10, 8, 4}, {2, 6, 10, 2}])), "overlap while live"
    refute Vapor.Extracted.check_alloc(32, [0], nested([{0, 10, 6, 4}])), "misaligned group"
    refute Vapor.Extracted.check_alloc(32, [0], nested([{0, 10, 0, 1}])), "reserved v0"
    refute Vapor.Extracted.check_alloc(32, [0], nested([{0, 10, 30, 4}])), "outside the file"
    assert Vapor.Extracted.check_alloc(32, [0], nested([{0, 5, 8, 4}, {5, 9, 8, 4}])), "disjoint lifetimes may share"
  end

  test "fuzz: every allocation the heuristic accepts, the checker accepts" do
    :rand.seed(:exsss, {7, 7, 7})

    for _ <- 1..300 do
      ivs =
        for i <- 1..Enum.random(1..12) do
          a = Enum.random(0..40)
          iv(i, Enum.random([1, 1, 2, 4, 8]), a, a + Enum.random(1..15))
        end

      case RegAlloc.allocate(ivs, @rvv_v) do
        {:ok, %{assign: a}} -> assert Vapor.Extracted.check_alloc(32, [0], nested(RegAlloc.placements(ivs, a)[:v]))
        {:error, _} -> :ok
      end
    end
  end

  test "every kernel compiles on every backend; the group factor adapts to pressure" do
    ks = [Kernels.sb_sums(), Kernels.gemv_sb4(), Kernels.gemv_sb4_masked(), Kernels.gemm_i8(),
          Kernels.ew(%{inputs: 3, outputs: [{:t, 0}], ops: [{{:t, 0}, :fma, [{:in, 0}, {:in, 1}, {:in, 2}]}]})]

    for k <- ks, be <- [X86, ARM, RVV], pol <- [:canonical, :fast] do
      assert {:ok, %Machine.Code{bin: bin, g: g}} = Machine.compile(k, be, policy: pol)
      assert byte_size(bin) > 0 and g in be.group_factors()
    end

    # 3-input fma strips: RVV fits m8 (3 groups of 8 in v1..v31); AVX2 needs g ≤ 2
    fma = List.last(ks)
    assert {:ok, %{g: 8}} = Machine.compile(fma, RVV)
    assert {:ok, %{g: 2}} = Machine.compile(fma, X86)
  end
end
