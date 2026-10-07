defmodule Vapor.ArbiterTest do
  use ExUnit.Case, async: true
  alias Vapor.Arbiter
  alias Vapor.Arbiter.Profile
  alias Vapor.Compile.Lower
  import Vapor.TestHelpers

  setup_all do
    {:ok, c} = Lower.lower(ssm_block())
    {:ok, c: c}
  end

  test "work is counted from the schedule and the emitted code", %{c: c} do
    w = Arbiter.work(c, %{}, :x86_64)
    gemv = Enum.filter(w, &(&1.kernel == :gemv_sb4))
    assert length(gemv) == 2
    assert Enum.all?(w, &(&1.kernel == :state_feedback or (&1.instructions > 0 and &1.bytes > 0)))
    assert [%{kernel: :state_feedback, bytes: b}] = Enum.filter(w, &(&1.kernel == :state_feedback))
    assert b == 2 * 4 * 256, "h ← h_next: read + write of the state per step"
    # the RVV code (R = 4 rows per iteration, m4 groups) issues fewer instructions
    rvv = Arbiter.work(c, %{}, :riscv64) |> Enum.map(& &1.instructions) |> Enum.sum()
    x86 = w |> Enum.map(& &1.instructions) |> Enum.sum()
    assert rvv < x86
  end

  test "the 4-bit GEMV is issue-bound on one AVX2 core: the third roof binds", %{c: c} do
    [g | _] = Arbiter.work(c, %{}, :x86_64) |> Enum.filter(&(&1.kernel == :gemv_sb4))
    p = Arbiter.default_profiles().native
    assert g.instructions / p.issue_rate > g.bytes / p.bandwidth
    assert g.instructions / p.issue_rate > g.flops / p.peak_flops
  end

  test "routing is argmin of predicted time; the crest is reported, not decisive", %{c: c} do
    w = Arbiter.work(c, %{})
    slow = %Profile{name: :fabric, peak_flops: 1.0e9, bandwidth: 1.0e9, call_overhead_s: 1.0}
    fast = %Profile{name: :fabric, peak_flops: 1.0e14, bandwidth: 1.0e13}
    assert Arbiter.decide(w, 8, %{native: Arbiter.default_profiles().native, fabric: slow}).target == :native
    d = Arbiter.decide(w, 8, %{native: Arbiter.default_profiles().native, fabric: fast})
    assert d.target == :fabric and d.intensity < d.crest_fabric, "faster although below its crest"
  end

  test "cooperative-matrix GEMM is selected only for the exact device configuration" do
    alias Vapor.Runtime.Fabric
    assert Fabric.variant(:gemm_i8, [64, 32, 256], %{coop_i8: true}) == :gemm_i8_coop
    assert Fabric.variant(:gemm_i8, [64, 17, 256], %{coop_i8: true}) == :gemm_i8
    assert Fabric.variant(:gemm_i8, [64, 32, 256], %{coop_i8: false}) == :gemm_i8
    assert Fabric.variant(:gemv_sb4, [64, 2], %{coop_i8: true}) == :gemv_sb4
  end

  test "predictions are deterministic functions of counted work and declared profiles", %{c: c} do
    w = Arbiter.work(c, %{})
    assert Arbiter.decide(w, 64) == Arbiter.decide(w, 64)
  end
end
