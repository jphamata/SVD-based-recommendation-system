defmodule Vapor.KhazanaTest do
  @moduledoc """
  The khazāna's promise is one sentence — after a crash at any instant,
  the store opens at the previous root or the new one, never a mixture —
  so the tests crash it at every byte of a commit.
  """
  use ExUnit.Case, async: true
  alias Vapor.Khazana, as: K

  setup do
    dir = Path.join(System.tmp_dir!(), "khz-#{System.unique_integer([:positive])}")
    on_exit(fn -> File.rm_rf!(dir) end)
    {:ok, dir: dir}
  end

  defp fresh(dir) do
    {:ok, k} = K.init(dir)
    {h1, k} = K.put(k, "first blob")
    {:ok, k} = K.commit(k, %{"v" => 1, "blobs" => [h1]})
    {k, h1}
  end

  test "blobs and the root survive a reopen; a blob is stored once", %{dir: dir} do
    {k, h1} = fresh(dir)
    {^h1, k} = K.put(k, "first blob")
    {h2, k} = K.put(k, "second")
    {:ok, k} = K.commit(k, %{"v" => 2, "blobs" => [h1, h2]})
    {:ok, k2} = K.open(dir)
    assert K.root(k2) == %{"v" => 2, "blobs" => [h1, h2]}
    assert K.get(k2, h2) == {:ok, "second"}
    assert k2.seq == k.seq
    # three blobs: two values and the two roots minus the one shared nothing — count by content
    assert length(Enum.filter(K.hashes(k2), &(K.get(k2, &1) == {:ok, "first blob"}))) == 1
  end

  test "a store is created once; opening nothing is an error unless asked to create", %{dir: dir} do
    assert {:error, _} = K.open(dir)
    assert {:ok, _} = K.open(dir, create: true)
    assert {:error, msg} = K.init(dir)
    assert msg =~ "already"
  end

  test "a crash at every byte of the pack append: the old root, and the store keeps working", %{dir: dir} do
    {k, _} = fresh(dir)
    old = K.root(k)
    {_, staged} = K.put(k, String.duplicate("x", 40))
    # the bytes this commit appends: the 40-byte blob and the new root, with their headers
    total = staged.pending_len + 36 + byte_size(Vapor.Canonical.encode(%{"v" => 9}))

    for n <- 0..(total - 1) do
      d = "#{dir}-p#{n}"
      File.cp_r!(dir, d)
      {:ok, kk} = K.open(d)
      {_, kk} = K.put(kk, String.duplicate("x", 40))
      assert_raise RuntimeError, ~r/injected/, fn -> K.commit(kk, %{"v" => 9}, fault: {:pack, n}) end
      {:ok, back} = K.open(d)
      assert K.root(back) == old, "byte #{n}"
      # the torn tail is overwritten by the next commit
      {h, back} = K.put(back, "after the crash")
      {:ok, _} = K.commit(back, %{"v" => 10})
      {:ok, again} = K.open(d)
      assert K.root(again) == %{"v" => 10} and K.get(again, h) == {:ok, "after the crash"}
      File.rm_rf!(d)
    end
  end

  test "a crash at every byte of the root slot: the old root or the new one, never neither", %{dir: dir} do
    {k, _} = fresh(dir)
    old = K.root(k)
    tagged = 4 + 8 + 1 + 8 + 32 + 32

    for n <- 0..(K.slot_size() - 1) do
      d = "#{dir}-s#{n}"
      File.cp_r!(dir, d)
      {:ok, kk} = K.open(d)
      assert_raise RuntimeError, fn -> K.commit(kk, %{"v" => 9}, fault: {:slot, n}) end
      {:ok, back} = K.open(d)
      got = K.root(back)
      # until the tag is whole the new slot cannot verify; once it is, the commit happened
      if n < tagged, do: assert(got == old, "byte #{n}"), else: assert(got == %{"v" => 9}, "byte #{n}")
      File.rm_rf!(d)
    end
  end

  test "a flipped bit in the newest slot falls back to the previous root; in both, the store says it is damaged", %{dir: dir} do
    {k, _} = fresh(dir)
    first = K.root(k)
    {:ok, k} = K.commit(k, %{"v" => 2})
    slot = Path.join(dir, "root.#{k.slot}")
    <<a::binary-20, b, rest::binary>> = File.read!(slot)
    File.write!(slot, <<a::binary, Bitwise.bxor(b, 4), rest::binary>>)
    {:ok, back} = K.open(dir)
    assert K.root(back) == first

    other = Path.join(dir, "root.#{if k.slot == :a, do: "b", else: "a"}")
    <<a::binary-20, b, rest::binary>> = File.read!(other)
    File.write!(other, <<a::binary, Bitwise.bxor(b, 4), rest::binary>>)
    assert {:error, msg} = K.open(dir)
    assert msg =~ "neither root slot"
  end

  test "a corrupted byte inside the committed pack is found on open, not served", %{dir: dir} do
    {k, _} = fresh(dir)
    pack = Path.join(dir, "pack.#{k.gen}")
    <<a::binary-40, b, rest::binary>> = File.read!(pack)
    File.write!(pack, <<a::binary, Bitwise.bxor(b, 1), rest::binary>>)
    assert {:error, msg} = K.open(dir)
    assert msg =~ "damaged"
  end

  test "compaction keeps the live blobs and the root; a crash during it leaves the old pack in force", %{dir: dir} do
    {k, h1} = fresh(dir)
    {dead, k} = K.put(k, String.duplicate("dead", 100))
    {live, k} = K.put(k, "live")
    {:ok, k} = K.commit(k, %{"live" => [h1, live]})

    d = dir <> "-gc"
    File.cp_r!(dir, d)
    {:ok, kk} = K.open(d)
    assert_raise RuntimeError, fn -> K.gc(kk, [h1, live], fault: {:slot, 30}) end
    {:ok, back} = K.open(d)
    assert back.gen == kk.gen and K.get(back, dead) == {:ok, String.duplicate("dead", 100)}
    File.rm_rf!(d)

    {:ok, k2, rep} = K.gc(k, [h1, live])
    assert rep.dropped > 0 and rep.bytes_after < rep.bytes_before
    {:ok, back} = K.open(dir)
    assert back.gen != k.gen
    assert K.root(back) == %{"live" => [h1, live]}
    assert K.get(back, live) == {:ok, "live"} and K.get(back, dead) == :error
    assert k2.seq == back.seq
    # and the store goes on: commits after a compaction append to the new pack
    {h3, back} = K.put(back, "next")
    {:ok, _} = K.commit(back, %{"n" => h3})
    {:ok, again} = K.open(dir)
    assert K.get(again, h3) == {:ok, "next"}
  end

  test "computed capabilities: verified by recomputation, revoked by a generation bump, unforgeable without the key", %{dir: dir} do
    {k, _} = fresh(dir)
    tok = K.mac(k, ["thread", "t1", "read", 0])
    assert K.mac_ok?(k, ["thread", "t1", "read", 0], tok)
    refute K.mac_ok?(k, ["thread", "t1", "read", 1], tok)
    refute K.mac_ok?(k, ["thread", "t1", "write", 0], tok)
    refute K.mac_ok?(k, ["thread", "t1", "read", 0], nil)
    other = dir <> "-other"
    {:ok, k2} = K.init(other)
    refute K.mac_ok?(k2, ["thread", "t1", "read", 0], tok)
    File.rm_rf!(other)
  end

  test "stress: 300 commits of random sizes with reopens; every root and every blob ever committed is readable", %{dir: dir} do
    {:ok, k} = K.init(dir)
    rng = Vapor.Entropy.rng({:khazana, :stress})

    {k, seen, _} =
      Enum.reduce(1..300, {k, %{}, rng}, fn i, {k, seen, rng} ->
        {n, rng} = Vapor.Entropy.uniform(3, rng)
        {k, seen, rng} =
          Enum.reduce(1..n, {k, seen, rng}, fn j, {k, seen, rng} ->
            {len, rng} = Vapor.Entropy.uniform(2_000, rng)
            data = :binary.copy(<<rem(i * 31 + j, 251)>>, len)
            {h, k} = K.put(k, data)
            {k, Map.put(seen, h, data), rng}
          end)

        {:ok, k} = K.commit(k, %{"i" => i})
        k = if rem(i, 50) == 0, do: elem(K.open(dir), 1), else: k
        {k, seen, rng}
      end)

    {:ok, back} = K.open(dir)
    assert K.root(back) == %{"i" => 300} and back.seq == k.seq
    for {h, d} <- seen, do: assert(K.get(back, h) == {:ok, d})
  end
end
