defmodule Vapor.TlogTest do
  @moduledoc """
  The transparency log against outside references: the RFC 6962 roots and
  node hashes of transparency-dev's reference tree, and transparency-dev's
  inclusion and consistency probes — mostly *negative* (wrong roots,
  truncated, extended, shifted proofs…), which a verifier that only tests
  the happy path would accept.
  """
  use ExUnit.Case, async: true
  alias Vapor.Tlog
  alias Vapor.Tlog.{Note, Witness}

  @leaves ["", <<0>>, <<0x10>>, <<0x20, 0x21>>, <<0x30, 0x31>>, <<0x40, 0x41, 0x42, 0x43>>,
           <<0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57>>, Base.decode16!("606162636465666768696A6B6C6D6E6F")]

  # transparency-dev/merkle testonly.RootHashes()
  @roots ~w(e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
            6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d
            fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125
            aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77
            d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
            4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4
            76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef
            ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c
            5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328)

  defp log_of(entries), do: Enum.reduce(entries, Tlog.new("test.vapor/log"), &elem(Tlog.append(&2, &1), 0))

  test "roots of every prefix equal the RFC 6962 reference tree" do
    log = log_of(@leaves)
    for {hex, n} <- Enum.with_index(@roots), do: assert(Base.encode16(Tlog.root(log, n), case: :lower) == hex, "size #{n}")
    # and the existing corpus/journal Merkle module agrees
    assert Tlog.root(log) == Vapor.Merkle.root(Enum.map(@leaves, &Vapor.Merkle.leaf/1))
  end

  test "every inclusion and consistency proof of the reference tree verifies; altered ones do not" do
    log = log_of(@leaves)

    for n <- 1..8, i <- 0..(n - 1) do
      p = Tlog.inclusion(log, i, n)
      assert Tlog.verify_inclusion(Tlog.leaf_hash(Enum.at(@leaves, i)), i, n, p, Tlog.root(log, n))
      refute Tlog.verify_inclusion(Tlog.leaf_hash(Enum.at(@leaves, i)), rem(i + 1, n), n, p, Tlog.root(log, n)) and n > 1
    end

    for m <- 1..8, n <- m..8 do
      p = Tlog.consistency(log, m, n)
      assert Tlog.verify_consistency(m, n, p, Tlog.root(log, m), Tlog.root(log, n)), "#{m} → #{n}"
      if m < n, do: refute(Tlog.verify_consistency(m, n, p, Tlog.root(log, n), Tlog.root(log, m)))
    end
  end

  test "transparency-dev probes: positives accepted, every negative refused" do
    probes = "../fixtures/tlog/probes.json" |> Path.expand(__DIR__) |> File.read!() |> Vapor.JSON.decode!()
    b64 = fn s -> case Base.decode64(s) do {:ok, b} -> b; :error -> s end end

    results =
      for p <- probes["consistency"] do
        got = Tlog.verify_consistency(p["size1"], p["size2"], Enum.map(p["proof"] || [], b64), b64.(p["root1"] || ""), b64.(p["root2"] || ""))
        {p["file"], got == not Map.get(p, "wantErr", false)}
      end ++
        for p <- probes["inclusion"] do
          got = Tlog.verify_inclusion(b64.(p["leafHash"] || ""), p["leafIdx"], p["treeSize"], Enum.map(p["proof"] || [], b64), b64.(p["root"] || ""))
          {p["file"], got == not Map.get(p, "wantErr", false)}
        end

    assert length(results) == 196
    assert Enum.reject(results, &elem(&1, 1)) == []
  end

  test "proofs on larger trees: every pair verifies, against roots computed from scratch" do
    entries = for i <- 0..200, do: :crypto.hash(:sha256, <<i::32>>)
    log = log_of(entries)
    fresh = fn n -> Vapor.Merkle.root(entries |> Enum.take(n) |> Enum.map(&Vapor.Merkle.leaf/1)) end

    for n <- [1, 2, 3, 7, 64, 65, 127, 128, 129, 201] do
      assert Tlog.root(log, n) == fresh.(n)
      for i <- Enum.uniq([0, div(n, 2), n - 1]), do: assert(Tlog.verify_inclusion(Tlog.leaf_hash(Enum.at(entries, i)), i, n, Tlog.inclusion(log, i, n), fresh.(n)))
      for m <- Enum.uniq([1, div(n, 3) + 1, n]), m <= n,
          do: assert(Tlog.verify_consistency(m, n, Tlog.consistency(log, m, n), fresh.(m), fresh.(n)))
    end

    # a log that rewrote entry 5 is inconsistent with its past checkpoint
    forked = log_of(List.replace_at(entries, 5, "rewritten"))
    refute Tlog.verify_consistency(10, 201, Tlog.consistency(forked, 10, 201), Tlog.root(log, 10), Tlog.root(forked, 201))
  end

  test "signed notes: format, key strings, unknown keys ignored, forgeries refused" do
    k = Note.keygen("vapor.test/log")
    other = Note.keygen("someone.else")
    assert k.verifier =~ ~r/^vapor\.test\/log\+[0-9a-f]{8}\+[A-Za-z0-9+\/=]+$/
    note = Note.sign("hello\nworld\n", [k.signer, other.signer])
    assert note =~ "hello\nworld\n\n— vapor.test/log "
    assert {:ok, %{text: "hello\nworld\n", signers: [%{name: "vapor.test/log", kind: :note}]}} = Note.open(note, [k.verifier])
    assert {:ok, %{signers: [_, _]}} = Note.open(note, [k.verifier, other.verifier])
    assert {:error, :no_known_signature} = Note.open(note, [Note.keygen("x").verifier])
    tampered = String.replace(note, "world", "w0rld")
    assert {:error, {:bad_signature, "vapor.test/log"}} = Note.open(tampered, [k.verifier])
  end

  test "a receipt proves inclusion offline; a witness cosigns only consistent growth" do
    log_key = Note.keygen("vapor.test/attestations")
    wit_key = Note.keygen("witness.test", :cosignature)
    log = Tlog.new("vapor.test/attestations")
    {log, _} = Tlog.append(log, "first")
    {log, r} = Tlog.anchor(log, "an attestation", log_key.signer)

    assert {:ok, %{size: 2, witnesses: []}} = Tlog.verify_receipt(r, "an attestation", log_keys: [log_key.verifier])
    assert {:error, :not_included} = Tlog.verify_receipt(r, "another attestation", log_keys: [log_key.verifier])
    assert {:error, :no_known_signature} = Tlog.verify_receipt(r, "an attestation", log_keys: [Note.keygen("x").verifier])
    assert {:error, :wrong_origin} = Tlog.verify_receipt(r, "an attestation", log_keys: [log_key.verifier], origin: "elsewhere")

    w = Witness.new(wit_key.signer, %{"vapor.test/attestations" => log_key.verifier})
    {:ok, w, cosigned} = Witness.cosign(w, r["checkpoint"], [], 1_700_000_000)
    r2 = %{r | "checkpoint" => cosigned}
    opts = [log_keys: [log_key.verifier], witnesses: [wit_key.verifier]]
    assert {:ok, %{witnesses: ["witness.test"]}} = Tlog.verify_receipt(r2, "an attestation", opts ++ [quorum: 1])
    assert {:error, {:witness_quorum, 1}} = Tlog.verify_receipt(r2, "an attestation", opts ++ [quorum: 2])

    # the log grows: the witness needs the consistency proof from what it saw
    {log, _} = Tlog.append(log, "third")
    cp3 = Tlog.checkpoint(log, log_key.signer)
    assert {:error, :inconsistent} = Witness.cosign(w, cp3, [])
    assert {:ok, w, _} = Witness.cosign(w, cp3, Tlog.consistency(log, 2, 3))

    # a split view: a fork of the same log with entry 0 rewritten
    {fork, _} = Tlog.new("vapor.test/attestations") |> Tlog.append("FIRST")
    {fork, _} = Tlog.append(fork, "an attestation")
    {fork, _} = Tlog.append(fork, "third")
    {fork, _} = Tlog.append(fork, "fourth")
    assert {:error, :inconsistent} = Witness.cosign(w, Tlog.checkpoint(fork, log_key.signer), Tlog.consistency(fork, 3, 4))
    # and a rollback is refused outright
    assert {:error, :rollback} = Witness.cosign(w, r["checkpoint"], [])
  end

  # the console's verifier (JavaScript, WebCrypto) is a second, independent
  # implementation: run it under Node against the probes and our own proofs
  @tag :tmp_dir
  test "the console's in-browser verifier agrees: probes, receipts, consistency, signatures", %{tmp_dir: dir} do
    node = System.find_executable("node")

    if node do
      k = Note.keygen("vapor.test/console")
      entries = for i <- 0..40, do: "vapor-note/1\nentry #{i}"
      log = log_of(entries)
      receipts = for i <- [0, 1, 7, 20, 40], do: Map.put(Tlog.receipt(log, i, k.signer), "entry", Base.encode64(Enum.at(entries, i)))
      hex = &Base.encode16(&1, case: :lower)
      receipts = Enum.map(receipts, fn r -> Map.update!(r, "proof", fn ps -> Enum.map(ps, &hex.(Base.decode64!(&1))) end) end)
      cons = for {m, n} <- [{1, 41}, {3, 41}, {8, 41}, {16, 17}, {41, 41}, {13, 40}],
                 do: %{from: m, to: n, proof: Enum.map(Tlog.consistency(log, m, n), hex), root1: hex.(Tlog.root(log, m)), root2: hex.(Tlog.root(log, n))}

      cases = Path.join(dir, "cases.json")
      File.write!(cases, Vapor.JSON.encode(%{checkpoint: Tlog.checkpoint(log, k.signer), verifier: k.verifier,
                                              receipts: Enum.map(receipts, &Map.new(&1, fn {a, b} -> {String.to_atom(a), b} end)),
                                              consistency: cons}))
      {out, 0} = System.cmd(node, [Path.expand("../js/tlog_verify.mjs", __DIR__), Path.expand("../../priv/console/index.html", __DIR__),
                                   Path.expand("../fixtures/tlog/probes.json", __DIR__), cases])
      assert out =~ ~r/^ok \d+/, out
    end
  end

  @tag :tmp_dir
  test "the log file is append-only and reopens to the same tree; a torn tail is dropped", %{tmp_dir: dir} do
    path = Path.join(dir, "a.tlog")
    log = Enum.reduce(@leaves, Tlog.new("test.vapor/file", path: path), &elem(Tlog.append(&2, &1), 0))
    {:ok, again} = Tlog.open(path)
    assert {again.size, Tlog.root(again)} == {8, Tlog.root(log)}
    assert Tlog.entry(again, 7) == List.last(@leaves)
    File.write!(path, <<0, 0, 1, 0, "torn">>, [:append])
    assert {:ok, %{size: 8}} = Tlog.open(path)
    assert_raise ArgumentError, fn -> Tlog.new("x", path: path) end
  end
end
