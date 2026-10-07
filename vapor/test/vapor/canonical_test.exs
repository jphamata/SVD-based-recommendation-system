defmodule Vapor.CanonicalTest do
  @moduledoc """
  Canonical bytes (`Vapor.Canonical`, deterministic CBOR): the RFC 8949
  appendix vectors, one accepted encoding per value, and — the point of a
  standard encoding — certificates that a verifier outside the BEAM
  checks with off-the-shelf libraries.
  """
  use ExUnit.Case, async: true
  import Bitwise
  alias Vapor.{Canonical, Certificate}
  import Vapor.TestHelpers

  # RFC 8949, Appendix A (the values with a canonical form here)
  @vectors [{0, "00"}, {23, "17"}, {24, "1818"}, {1000, "1903e8"}, {1_000_000_000_000, "1b000000e8d4a51000"},
            {18_446_744_073_709_551_615, "1bffffffffffffffff"}, {18_446_744_073_709_551_616, "c249010000000000000000"},
            {-18_446_744_073_709_551_616, "3bffffffffffffffff"}, {-18_446_744_073_709_551_617, "c349010000000000000000"},
            {-1, "20"}, {-1000, "3903e7"}, {0.0, "f90000"}, {1.0, "f93c00"}, {1.1, "fb3ff199999999999a"}, {1.5, "f93e00"},
            {65504.0, "f97bff"}, {100_000.0, "fa47c35000"}, {3.4028234663852886e38, "fa7f7fffff"},
            {1.0e300, "fb7e37e43c8800759c"}, {5.960464477539063e-8, "f90001"}, {0.00006103515625, "f90400"},
            {-4.0, "f9c400"}, {-4.1, "fbc010666666666666"}, {false, "f4"}, {true, "f5"}, {nil, "f6"}, {"", "40"},
            {<<1, 2, 3, 4>>, "4401020304"}, {[], "80"}, {[1, 2, 3], "83010203"}, {%{}, "a0"}, {%{1 => 2, 3 => 4}, "a201020304"}]

  test "RFC 8949 appendix vectors" do
    for {t, hex} <- @vectors, do: assert(Base.encode16(Canonical.encode(t), case: :lower) == hex, inspect(t))
  end

  test "round trip, and exactly one accepted encoding per value" do
    t = %{a: [1, {2, :b}, "x"], z: 1.5, n: nil, big: 1 <<< 70, neg: -(1 <<< 80), t: Vapor.Tensor.from_list(:f32, [2], [1.0, 2.0])}
    assert Canonical.decode(Canonical.encode(t)) == {:ok, t}
    # map order is part of the encoding: equal maps, equal bytes
    assert Canonical.encode(Map.new(Enum.shuffle(Map.to_list(t)))) == Canonical.encode(t)
    # a non-shortest integer, a float wider than needed, a text string: refused
    assert {:error, :not_canonical} = Canonical.decode(<<0x18, 5>>)
    assert {:error, :not_canonical} = Canonical.decode(<<0xFB, 1.0::float-64>>)
    assert {:error, :not_canonical} = Canonical.decode(<<0x61, ?a>>)
    assert {:error, :trailing_bytes} = Canonical.decode(<<0x01, 0x02>>)
    # unknown atom names are not created from untrusted bytes
    bin = IO.iodata_to_binary([<<0xD8, 39, 0x78, 40>>, String.duplicate("z", 40)])
    assert {:error, _} = Canonical.decode(bin)
    assert_raise ArgumentError, fn -> Canonical.encode(self()) end
  end

  @tag :cbor2
  test "a Python verifier recomputes the canonical bytes and checks the certificate's Ed25519 signature" do
    {:ok, c} = Vapor.compile(gemm_program(), key: Certificate.keygen() |> tap(&Process.put(:k, &1)))
    k = Process.get(:k)
    [{pub, sig}] = c.certificate.signatures
    assert pub == k.public

    out =
      py!("""
      import sys, cbor2
      from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
      raw = sys.stdin.buffer.read()
      signed = cbor2.loads(raw)                         # tuple tag 30305: (name, 2, payload, sigs)
      assert cbor2.dumps(signed, canonical=True) == raw  # Python's canonical bytes are vapor's
      _, version, payload, sigs = signed.value
      # what is signed: the tuple {:vapor_certificate, 2, payload}, canonically
      msg = cbor2.dumps(cbor2.CBORTag(30305, [cbor2.CBORTag(39, "vapor_certificate"), version, payload]), canonical=True)
      ok = 0
      for s in sigs:                                    # each a tuple {public_key, signature}
          pub, sig = s.value
          Ed25519PublicKey.from_public_bytes(pub).verify(sig, msg); ok += 1
      print(ok, version, sorted(str(k.value) if hasattr(k, "value") else str(k) for k in payload)[:3])
      """, [], Certificate.encode(c.certificate))

    assert String.starts_with?(out, "1 2 "), out
    _ = sig
  end
end
