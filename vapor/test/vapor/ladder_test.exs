defmodule Vapor.LadderTest do
  use ExUnit.Case, async: false
  alias Vapor.{Bundle, Certificate, Program, Rejection, Tensor}
  alias Vapor.Algebra.Term, as: T
  import Vapor.TestHelpers

  @moduletag timeout: 600_000

  setup_all do
    {:ok, k1: Certificate.keygen(), k2: Certificate.keygen(), k3: Certificate.keygen()}
  end

  test "the SSM block is certified on every available substrate", %{k1: k1} do
    {:ok, c} = Vapor.compile(ssm_block(), key: k1)
    p = c.certificate.payload
    assert p.parity.bit_identical == :all_outputs
    assert :oracle in p.parity.substrates
    assert Enum.all?(p.adjoint, &(&1.identity == :holds))
    assert p.admission.registers |> Map.keys() |> Enum.sort() == [:aarch64, :riscv64, :x86_64, :x86_64_avx512]
    assert p.lean_sources == Vapor.Extracted.source_digest()
    assert :ok = Certificate.verify(c.certificate, [k1.public])
  end

  test "independent nodes produce byte-identical payloads and form a quorum", %{k1: k1, k2: k2, k3: k3} do
    {:ok, a} = Vapor.compile(gemm_program(), key: k1)
    {:ok, b} = Vapor.compile(gemm_program(), key: k2)
    assert Certificate.canonical(a.certificate) == Certificate.canonical(b.certificate)
    {:ok, cert} = Certificate.cosign(a.certificate, b.certificate, k2)
    trusted = [k1.public, k2.public, k3.public]
    assert :ok = Certificate.verify(cert, trusted, 2)
    assert {:error, {:quorum, 2, 3}} = Certificate.verify(cert, trusted, 3)
    assert {:error, {:quorum, 0, 1}} = Certificate.verify(cert, [k3.public], 1)
  end

  test "tampering is detected: payload, signature, and shipped machine code", %{k1: k1} do
    {:ok, c} = Vapor.compile(gemm_program(), key: k1)
    forged = put_in(c.certificate.payload.parity.bit_identical, :forged).certificate
    assert {:error, _} = Certificate.verify(forged, [k1.public])

    bin = Bundle.pack(c)
    assert {:ok, _} = Bundle.unpack(bin, [k1.public])
    evil = update_in(c.code[:x86_64].blob, fn b -> <<0xCC>> <> binary_part(b, 1, byte_size(b) - 1) end)
    assert {:error, :artifact_digest_mismatch} = Bundle.unpack(Bundle.pack(evil), [k1.public])
  end

  test "the no-wrap bound rejects inadmissible int8 contractions with a repair" do
    a = T.input(:a, :s8, [4, 140_000])
    w = T.const(Tensor.random(:s8, [2, 140_000], 1))
    assert {:error, %Rejection{bound: bound, repair: repair}} = Vapor.compile(Program.new(c: T.gemm_i8(a, w)))
    assert bound =~ "2³¹" and repair =~ "strip-mine K"
  end

  test "Vapor.run executes the certified recurrent program with failover tracing", %{k1: k1} do
    {:ok, c} = Vapor.compile(ssm_block(), key: k1)
    env = ssm_env(6)
    {:ok, res, trace} = Vapor.run(c, env)
    assert length(res.steps) == 6
    assert List.last(trace) == {res.substrate, :ok}
    {:ok, ref} = Vapor.Runtime.Native.run_oracle(c, env, iterations: 6, sequence: [:x])
    assert res.steps == ref.steps
  end
end
