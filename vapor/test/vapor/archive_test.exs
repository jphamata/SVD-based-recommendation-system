defmodule Vapor.ArchiveTest do
  @moduledoc """
  Save and export (`Vapor.Archive`): an archive's identity is the hash of
  its manifest; every file is checked against it; replayable kinds are
  computed again and compared; a forged file is caught; and an archive
  names a kind, never code — opening one runs nothing it chose.
  """
  use ExUnit.Case, async: true
  alias Vapor.Archive

  test "pack, verify, replay; the same result is the same archive" do
    {:ok, r} = Archive.produce("prove.homology", %{"complex" => "torus"})
    a = Archive.pack("prove.homology", %{"complex" => "torus"}, r, %{"note.txt" => "hello"})
    b = Archive.pack("prove.homology", %{"complex" => "torus"}, r, %{"note.txt" => "hello"})
    assert a.id == b.id
    assert {:ok, %{files: %{"note.txt" => "hello"}, manifest: %{"kind" => "prove.homology"}}} = Archive.verify(a.zip)
    assert Archive.replay(a.zip) == {:ok, :same}
  end

  test "a forged result is caught; a result that no longer matches its recipe does not replay" do
    {:ok, r} = Archive.produce("prove.homology", %{"complex" => "klein"})
    a = Archive.pack("prove.homology", %{"complex" => "klein"}, r)
    {:ok, files} = :zip.unzip(a.zip, [:memory])
    forged = Enum.map(files, fn {n, b} -> if n == ~c"result.json", do: {n, String.replace(b, "1", "2")}, else: {n, b} end)
    {:ok, {_, z}} = :zip.create(~c"x.zip", forged, [:memory])
    assert Archive.verify(z) == {:error, {:tampered, ["result.json"]}}

    # a lie told consistently (the manifest re-hashed) verifies — and replay is what catches it
    lie = Archive.pack("prove.homology", %{"complex" => "klein"}, %{"gf2" => %{"betti" => [9, 9, 9]}})
    assert {:ok, _} = Archive.verify(lie.zip)
    assert {:error, {:differs, _}} = Archive.replay(lie.zip)
  end

  test "an archive names a kind, not a function: unknown kinds verify but are not run" do
    a = Archive.pack("System.cmd", %{"cmd" => "rm"}, %{})
    assert {:ok, _} = Archive.verify(a.zip)
    assert Archive.replay(a.zip) == {:error, {:not_replayable, "System.cmd"}}
    assert {:error, {:not_an_archive, _}} = Archive.verify("not a zip")
  end

  test "untrusted bytes are bounded: a zip bomb, a malformed manifest, a recipe that would take the machine" do
    # 300 MB of zeros deflates to ~1 MB: refused at the bound, never inflated past it
    {:ok, {_, bomb}} = :zip.create(~c"b.zip", [{~c"manifest.json", "{}"}, {~c"z", :binary.copy(<<0>>, 300_000_000)}], [:memory])
    assert byte_size(bomb) < 2_000_000
    assert Archive.verify(bomb) == {:error, :too_large}

    {:ok, {_, list}} = :zip.create(~c"c.zip", [{~c"manifest.json", "[1, 2]"}], [:memory])
    assert Archive.verify(list) == {:error, :bad_manifest}
    {:ok, {_, nores}} = :zip.create(~c"c.zip", [{~c"manifest.json", ~s({"kind": "x", "files": {}})}], [:memory])
    assert Archive.verify(nores) == {:error, :no_result}

    # every recipe parameter is bounded before anything runs
    # kinds whose producers left in 0.16 (discovery, self-play) are refused by name, not run
    assert {:error, {:not_replayable, "discover.sorting_network"}} = Archive.produce("discover.sorting_network", %{"n" => 5})
    assert {:error, {:not_replayable, "games.selfplay"}} = Archive.produce("games.selfplay", %{"games" => 1, "sims" => 4, "seed" => 1})
    assert {:error, _} = Archive.produce("science", %{"experiment" => "nope"})
    assert {:error, _} = Archive.produce("prove.homology", %{"complex" => "nope"})
    assert {:error, _} = Archive.produce("prove.geometry", %{"name" => "nope"})
  end

  describe "signed archives (0.13)" do
    test "a signature by the operator's key; the identity unchanged; unsigned, untrusted, forged — refused" do
      k = Vapor.Certificate.keygen(); other = Vapor.Certificate.keygen()
      {:ok, r} = Vapor.Finance.replay("finance.calendar", %{"text" => "du 2025-01-02 2026-01-02"})
      a = Archive.pack("finance.calendar", %{"text" => "du 2025-01-02 2026-01-02"}, r)
      {:ok, z} = Archive.sign(a.zip, k)
      {:ok, b} = Archive.verify(z, trusted: [k.public])
      assert b.id == a.id and b.signature.valid and b.signature.trusted
      assert Archive.replay(z) == {:ok, :same}
      assert Archive.verify(a.zip, trusted: [k.public]) == {:error, :unsigned}
      assert {:error, {:untrusted_key, _}} = Archive.verify(z, trusted: [other.public])
      # a coherent lie: the result and the manifest rewritten together, the old signature kept
      {:ok, entries} = :zip.unzip(z, [:memory])
      ent = Map.new(entries, fn {n, b} -> {to_string(n), b} end)
      fake = Vapor.JSON.encode(%{"lines" => [%{"value" => 251}]})
      m = Vapor.JSON.decode(ent["manifest.json"]) |> elem(1) |> put_in(["files", "result.json", "sha256"], Base.encode16(:crypto.hash(:sha256, fake), case: :lower))
      list = [{~c"manifest.json", Vapor.JSON.encode(m)}, {~c"result.json", fake}, {~c"signature.json", ent["signature.json"]}]
      {:ok, {_, forged}} = :zip.create(~c"x.zip", list, [:memory])
      assert Archive.verify(forged) == {:error, :bad_signature}
      # re-signed by the forger: the signature holds, the key is not trusted
      {:ok, resigned} = Archive.sign(forged, other)
      assert {:error, {:untrusted_key, _}} = Archive.verify(resigned, trusted: [k.public])
    end

    test "the finance desk's deterministic kinds replay; timings are scrubbed" do
      for {kind, text} <- [{"finance.backtest", "data = gbm n=300 seed=4\nsignal = sign(sma(ret(close), 3))"}, {"finance.book", "buy 10 @ 1.00 owner=a\nsell 4 @ 0.99 owner=b"},
                           {"finance.arbitrage", "states = u, d\nb bid=0.9 ask=0.91 payoff = 1, 1\ns bid=1 ask=1.01 payoff = 1.2, 0.9"}] do
        {:ok, r} = Archive.produce(kind, %{"text" => text})
        a = Archive.pack(kind, %{"text" => text}, r)
        assert Archive.replay(a.zip) == {:ok, :same}, kind
      end
    end
  end
end
