defmodule Vapor.SubstrateTest do
  @moduledoc """
  The substrate airlock: a device is admitted by measurement. Every way a
  device can depart from the canonical semantics must be caught and
  named — tested here against devices built to break the rules.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Program, Substrate, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Runtime.{Dispatch, Native, Substrates}

  # a device that rounds every binary32 operand to bfloat16 before computing
  # (a matrix engine fed bf16), simulated exactly on the oracle
  defp bf16_device do
    fn c, env, opts ->
      round = fn %Tensor{dtype: :f32} = t -> t |> Tensor.to_bf16() |> Tensor.widen(); t -> t end
      prog = deep_map(c.program, round)
      {:ok, c2} = Lower.lower(prog, policy: c.policy)
      Native.run_oracle(c2, Map.new(env, fn {k, v} -> {k, round.(v)} end), opts)
    end
  end

  defp deep_map(%Tensor{} = t, f), do: f.(t)
  defp deep_map(%{__struct__: m} = s, f), do: s |> Map.from_struct() |> Map.new(fn {k, v} -> {k, deep_map(v, f)} end) |> then(&struct(m, &1))
  defp deep_map(t, f) when is_tuple(t), do: t |> Tuple.to_list() |> Enum.map(&deep_map(&1, f)) |> List.to_tuple()
  defp deep_map(l, f) when is_list(l), do: Enum.map(l, &deep_map(&1, f))
  defp deep_map(x, _f), do: x

  # a device whose int8 GEMM saturates instead of wrapping (and is otherwise exact)
  defp saturating_device do
    fn c, env, opts ->
      {:ok, r} = Native.run_oracle(c, env, opts)
      outs = Map.new(r.outputs, fn {k, %Tensor{dtype: :s32} = t} -> {k, Tensor.from_list(:s32, t.shape, Enum.map(Tensor.to_list(t), &(&1 + 1)))}
                                    {k, t} -> {k, t} end)
      {:ok, %{r | outputs: outs}}
    end
  end

  @tag :native
  test "the host worker and the exact oracle are canonical" do
    a = Substrate.admit(Substrates.get(:host))
    assert a.verdict == :canonical and a.reasons == []
    assert a.fingerprint == %{contraction: false, flush_to_zero: false, denormals_are_zero: false, reduction_order: :canonical,
                              significand_bits: 24, signed_zero: :ieee, nan_select: :ieee, division: :correctly_rounded, functions: :canonical}
    assert Substrate.admit(%{id: :oracle, kind: :oracle}).verdict == :canonical
  end

  @tag :vulkan
  test "the Vulkan device (lavapipe here) is admitted canonical on arrival, by the dispatcher itself" do
    {:ok, f} = Vapor.Runtime.Fabric.start_link(exec: [Substrates.binary("vapor-fabric", "native")])
    dev = %{id: :fabric, kind: :fabric, isa: :spirv, mode: :gpu, server: f}
    assert Substrate.admission(dev) == nil
    {:ok, c} = Lower.lower(Program.new(y: T.mul(T.input(:x, :f32, [16]), T.splat(3.0))))
    {:ok, r, _} = Dispatch.run(c, %{x: Tensor.random(:f32, [16], 1)}, substrates: [dev], prefer: :fabric)
    assert r.substrate == :fabric
    a = Substrate.admission(dev)
    assert a.verdict == :canonical and a.device == Vapor.Runtime.Fabric.info(f).name
  end

  @tag :msl_shim
  @tag :native
  test "three Metal devices: one canonical, one that fuses a·b+c, one that flushes subnormals" do
    keys = for {_n, p, _e, _o} <- Substrate.probes(), {:ok, c} = Lower.lower(p), %{kernel: k} <- c.schedule, uniq: true, do: k

    verdicts =
      for mode <- [:conforming, :contracting, :ftz] do
        Vapor.MSLShim.ensure_keys(keys, :canonical, mode)
        {:ok, f} = Vapor.MSLShim.start(mode)
        a = Substrate.admit(%{id: :metal, kind: :fabric, server: f})
        assert a.device == Vapor.MSLShim.device_name(mode)
        {mode, a}
      end
      |> Map.new()

    assert verdicts.conforming.verdict == :canonical

    c = verdicts.contracting
    assert c.verdict == :envelope and c.fingerprint.contraction == true
    assert c.fingerprint.flush_to_zero == false and c.fingerprint.significand_bits == 24
    # the contraction shows up in the kernels, and every analytic output is inside its bound
    assert c.probes.linear_f32.envelope == :within and not c.probes.linear_f32.equal
    assert c.probes.gemm_i8.equal

    z = verdicts.ftz
    assert z.verdict == :envelope and z.fingerprint.flush_to_zero == true and z.fingerprint.denormals_are_zero == true
    assert z.fingerprint.contraction == false
  end

  test "the envelope covers denormals-are-zero (found by the airlock on XLA's CPU), and stays tight" do
    sub = Vapor.F32.from_float(:math.pow(2, -130))
    x = Tensor.new(:f32, [16], :binary.copy(<<sub::32-little>>, 16))
    prog = Program.new(mn: T.min(T.input(:x, :f32, [16]), T.splat(0.25)), big: T.mul(T.input(:x, :f32, [16]), T.splat(:math.pow(2, 100))))
    {:ok, c} = Lower.lower(prog)
    [b] = Vapor.Verify.Envelope.bounds(c, %{x: x})
    zero = Tensor.from_list(:f32, [16], List.duplicate(0.0, 16))
    # a DAZ device reads x as 0: min → +0, x·2¹⁰⁰ → +0 — both inside
    assert :ok == Vapor.Verify.Envelope.check(b.mn, zero)
    assert :ok == Vapor.Verify.Envelope.check(b.big, zero)
    # … tightly: the exact product is 2⁻³⁰, so 0 and 2⁻³⁰ pass and twice it does not
    assert {:error, _} = Vapor.Verify.Envelope.check(b.big, Tensor.from_list(:f32, [16], List.duplicate(:math.pow(2, -28), 16)))
  end

  test "operands rounded below binary32 are refused, with the precision measured" do
    a = Substrate.admit(%{id: :bf16_engine}, runner: bf16_device(), device: "bf16 matrix engine (simulated)")
    assert a.verdict == :refused
    assert a.fingerprint.significand_bits == 8
    assert [reason] = a.reasons
    assert reason =~ "8 significand bits"
  end

  test "integer kernels that do not wrap are refused" do
    a = Substrate.admit(%{id: :bad_int}, runner: saturating_device(), only: [:gemm_i8, :linear_f32])
    assert a.verdict == :refused
    assert Enum.any?(a.reasons, &(&1 =~ "integer kernels"))
  end

  test "a crashing device is refused with the error, not trusted" do
    a = Substrate.admit(%{id: :dead}, runner: fn _, _, _ -> exit(:device_gone) end, only: [:contraction])
    assert a.verdict == :refused
    assert hd(a.reasons) =~ "device_gone"
  end

  test "an answer out of protocol, or missing an output, is refused — the airlock never crashes" do
    for runner <- [fn _, _, _ -> :weird end, fn _, _, _ -> {:ok, %{outputs: %{}}} end] do
      a = Substrate.admit(%{id: :odd}, runner: runner, only: [:contraction])
      assert a.verdict == :refused
    end
  end

  test "a difference no bound covers, with no measured cause, is refused; unrun probes are unmeasured" do
    oracle = fn c, env, ro -> Vapor.Runtime.Native.run_oracle(c, env, ro) end
    off = fn c, env, ro ->
      {:ok, want} = oracle.(c, env, ro)
      {:ok, %{outputs: Map.new(want.outputs, fn {k, t} -> {k, Tensor.from_list(:f32, t.shape, Enum.map(Tensor.to_floats(t), &(&1 * (1 + 1 / 1024))))} end)}}
    end

    a = Substrate.admit(%{id: :odd_div}, runner: off, only: [:division])
    assert a.verdict == :refused
    # the control: the exact answers on the same probe
    b = Substrate.admit(%{id: :ok_div}, runner: oracle, only: [:division])
    assert b.verdict == :canonical
    assert b.fingerprint.contraction == :unmeasured and b.fingerprint.reduction_order == :unmeasured
  end

  test "the admission is a signed, canonical record; one changed byte is caught" do
    a = Substrate.admit(%{id: :oracle, kind: :oracle})
    key = Vapor.Certificate.keygen()
    {bytes, sig} = Substrate.attest(a, key)
    assert Substrate.verify({bytes, sig}, key.public)
    assert bytes == Vapor.Canonical.encode(Substrate.record(Substrate.admit(%{id: :oracle, kind: :oracle})))
    <<h, rest::binary>> = bytes
    refute Substrate.verify({<<Bitwise.bxor(h, 1), rest::binary>>, sig}, key.public)
  end

  @tag :native
  test "the dispatcher honours admissions: canonical programs skip an envelope device, fast ones may use it" do
    {:ok, c_can} = Lower.lower(Program.new(y: T.mul(T.input(:x, :f32, [16]), T.splat(3.0))))
    {:ok, c_fast} = Lower.lower(Program.new(y: T.mul(T.input(:x, :f32, [16]), T.splat(3.0))), policy: :fast)
    env = %{x: Tensor.random(:f32, [16], 1)}
    # a stand-in "device" in the Metal slot (the oracle underneath) with a recorded admission
    dev = %{id: :metal, kind: :oracle, server: nil, isa: :oracle, mode: :exact}
    host = Substrates.get(:host)

    Substrate.register(%Substrate.Admission{substrate: :metal, verdict: :envelope, fingerprint: %{}, probes: %{}, reasons: []})
    on_exit(fn -> :persistent_term.erase({Substrate, :metal}) end)

    assert Substrate.allowed?(dev, :fast) and not Substrate.allowed?(dev, :canonical)
    # preferred first, yet skipped for a canonical program …
    {:ok, r, trace} = Dispatch.run(c_can, env, substrates: [dev, host], prefer: :fabric)
    assert r.substrate == :host and not Enum.any?(trace, &match?({:metal, _}, &1))
    # … and used for a fast one
    {:ok, r, _} = Dispatch.run(c_fast, env, substrates: [dev, host], prefer: :fabric)
    assert r.substrate == :metal

    Substrate.register(%Substrate.Admission{substrate: :metal, verdict: :refused, fingerprint: %{}, probes: %{}, reasons: ["x"]})
    refute Substrate.allowed?(dev, :fast)
  end
end
