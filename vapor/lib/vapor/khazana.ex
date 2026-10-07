defmodule Vapor.Khazana do
  @moduledoc """
  The **khazāna** (خزانة, root خ-ز-ن *kh-z-n*, "to store") — a
  content-addressed store whose current state advances by one crash-atomic
  root commit (docs/KHAZANA.md). The persistence protocol is ASAS §8.4
  (*write content, then persist; write the inactive root slot; publish by
  sequence, not by overwrite; recover by validation*), carried from NVRAM to
  a POSIX directory.

  The pain: every application that keeps state in files reinvents "write a
  temp file, fsync, rename, fsync the directory" — and OTP cannot fsync a
  directory, so on the BEAM a rename is not durable. The khazāna never
  needs one: **after `init/1` no file is ever created, renamed or deleted.**

      DIR/pack.0, DIR/pack.1   blobs, appended:  [u32 length][32-byte SHA-256][bytes]
      DIR/root.a, DIR/root.b   one fixed 128-byte record each, rewritten in place
      DIR/key                  a 32-byte secret (capabilities, `mac/2`)

  A root record is `magic · seq (u64) · pack generation (u8) · committed
  pack length (u64) · root blob hash (32) · tag (32)`, the tag being the
  SHA-256 of everything before it. `commit/2`:

    1. appends the new blobs (and the encoded root term) to the active pack
       and `datasync`s it — content is durable before anything names it;
    2. writes the **inactive** slot with `seq + 1` and `datasync`s it — the
       active slot is never touched, so a crash here loses only the
       half-written slot;
    3. the current root is the valid slot with the highest sequence.

  `open/1` reads both slots, discards any whose tag does not verify (a torn
  write fails its tag), takes the highest surviving sequence, and reads the
  pack only up to the length that root committed: a torn append past it is
  ignored. Every blob inside the committed prefix is re-hashed on open, so
  corruption is found, not served.

  Compaction (`gc/2`) writes the live blobs into the *other* pack (both
  exist from `init/1`), then commits a root naming that pack: the same
  protocol, so a crash during compaction leaves the previous pack and root
  in force.

  What it does **not** do, stated: it assumes `datasync` returns only when
  the data is durable (true for local filesystems with honest disks; not for
  some network filesystems or disks that lie about their caches), and that a
  128-byte write is not *silently* corrupted in a way that still matches its
  tag (a SHA-256 collision). Concurrency is the caller's: one process owns a
  store (`Vapor.Majlis` is a GenServer for that reason).
  """

  @magic "KHZ1"
  @slot 128
  @names %{0 => "pack.0", 1 => "pack.1"}
  @max_blob 64 * 1024 * 1024

  defstruct [:dir, :seq, :gen, :len, :root, :index, :slot, :key, pending: [], pending_len: 0]

  @type t :: %__MODULE__{}

  # ------------------------------------------------------------ lifecycle

  @doc """
  Create a store in `dir` (which must not hold one already): two empty
  packs, two empty slots, a secret key — each fsynced. The one moment files
  are created; its directory entries are as durable as the filesystem makes
  a new file's (the store is unusable, not inconsistent, if a crash loses
  them, and `init/1` can be run again).
  """
  def init(dir) do
    File.mkdir_p!(dir)

    if File.exists?(Path.join(dir, "root.a")) do
      {:error, "#{dir} already holds a store"}
    else
      for {_, n} <- @names, do: write_new(Path.join(dir, n), "")
      write_new(Path.join(dir, "key"), Vapor.Entropy.bytes(32))
      File.chmod(Path.join(dir, "key"), 0o600)
      # an empty store is a committed root too: seq 1, pack 0, nothing in it
      {:ok, k} = open_slots(dir)
      write_new(Path.join(dir, "root.b"), :binary.copy(<<0>>, @slot))
      write_new(Path.join(dir, "root.a"), :binary.copy(<<0>>, @slot))
      commit(%{k | slot: :b}, nil)
    end
  end

  @doc "Open an existing store, or create it when `create: true` and absent."
  def open(dir, opts \\ []) do
    cond do
      File.exists?(Path.join(dir, "root.a")) -> load(dir)
      Keyword.get(opts, :create, false) -> init(dir)
      true -> {:error, "#{dir}: no store (open with create: true to make one)"}
    end
  end

  defp write_new(path, data) do
    {:ok, f} = :file.open(path, [:write, :binary, :raw])
    :ok = :file.write(f, data)
    :ok = :file.sync(f)
    :ok = :file.close(f)
  end

  defp open_slots(dir) do
    key = File.read!(Path.join(dir, "key"))
    {:ok, %__MODULE__{dir: dir, seq: 0, gen: 0, len: 0, root: nil, index: %{}, slot: :b, key: key}}
  end

  defp load(dir) do
    key = File.read!(Path.join(dir, "key"))

    slots =
      for s <- [:a, :b], rec = read_slot(dir, s), rec != nil, do: Map.put(rec, :slot, s)

    case Enum.max_by(slots, & &1.seq, fn -> nil end) do
      nil ->
        {:error, "#{dir}: neither root slot verifies — the store is damaged"}

      r ->
        with {:ok, index} <- scan(Path.join(dir, @names[r.gen]), r.len) do
          if r.root != nil and not Map.has_key?(index, r.root) do
            {:error, "#{dir}: the committed root names a blob the pack does not hold"}
          else
            {:ok, %__MODULE__{dir: dir, seq: r.seq, gen: r.gen, len: r.len, root: r.root, index: index, slot: r.slot, key: key}}
          end
        end
    end
  end

  defp read_slot(dir, s) do
    case File.read(Path.join(dir, "root.#{s}")) do
      {:ok, <<@magic, seq::64, gen, len::64, root::binary-32, tag::binary-32, _::binary>> = bin} when byte_size(bin) == @slot ->
        body = binary_part(bin, 0, 4 + 8 + 1 + 8 + 32)

        if :crypto.hash(:sha256, body) == tag and gen in [0, 1] do
          %{seq: seq, gen: gen, len: len, root: if(root == <<0::256>>, do: nil, else: root)}
        end

      _ ->
        nil
    end
  end

  # every record of the committed prefix is re-hashed: corruption is found on open
  defp scan(path, len) do
    case File.open(path, [:read, :binary, :raw]) do
      {:ok, f} ->
        try do
          scan_records(f, 0, len, %{})
        after
          File.close(f)
        end

      {:error, why} ->
        {:error, "#{path}: #{inspect(why)}"}
    end
  end

  defp scan_records(_f, pos, len, acc) when pos == len, do: {:ok, acc}

  defp scan_records(f, pos, len, acc) do
    with {:ok, <<n::32, h::binary-32>>} <- :file.pread(f, pos, 36),
         true <- pos + 36 + n <= len,
         {:ok, data} when byte_size(data) == n <- read_exact(f, pos + 36, n),
         true <- :crypto.hash(:sha256, data) == h do
      scan_records(f, pos + 36 + n, len, Map.put_new(acc, h, {pos + 36, n}))
    else
      _ -> {:error, "the pack is damaged at byte #{pos} (inside the committed length #{len})"}
    end
  end

  defp read_exact(_f, _pos, 0), do: {:ok, ""}
  defp read_exact(f, pos, n), do: :file.pread(f, pos, n)

  # ---------------------------------------------------------------- blobs

  @doc """
  Stage a blob; returns `{hash, store}`. Nothing is durable until the next
  `commit/2`. A blob already present (committed or staged) is not stored
  twice.
  """
  def put(%__MODULE__{} = k, data) when is_binary(data) do
    if byte_size(data) > @max_blob, do: raise(ArgumentError, "a blob of #{byte_size(data)} bytes (the limit is #{@max_blob})")
    h = :crypto.hash(:sha256, data)

    cond do
      Map.has_key?(k.index, h) -> {h, k}
      Enum.any?(k.pending, &(elem(&1, 0) == h)) -> {h, k}
      true -> {h, %{k | pending: [{h, data} | k.pending], pending_len: k.pending_len + 36 + byte_size(data)}}
    end
  end

  @doc "Stage a term as canonical CBOR (`Vapor.Canonical`); `{hash, store}`."
  def put_term(k, term), do: put(k, Vapor.Canonical.encode(term))

  @doc "A blob's bytes, committed or staged: `{:ok, data}` or `:error`."
  def get(%__MODULE__{} = k, h) do
    case Map.fetch(k.index, h) do
      {:ok, {pos, n}} ->
        {:ok, f} = File.open(Path.join(k.dir, @names[k.gen]), [:read, :binary, :raw])

        try do
          {:ok, data} = read_exact(f, pos, n)
          {:ok, data}
        after
          File.close(f)
        end

      :error ->
        case List.keyfind(k.pending, h, 0) do
          {_, data} -> {:ok, data}
          nil -> :error
        end
    end
  end

  @doc "A term stored by `put_term/2`."
  def get_term(k, h) do
    with {:ok, bin} <- get(k, h), do: Vapor.Canonical.decode(bin)
  end

  @doc "Whether a blob is held."
  def has?(%__MODULE__{} = k, h), do: Map.has_key?(k.index, h) or List.keymember?(k.pending, h, 0)

  @doc "Every committed blob hash, in the order they were written (oldest first)."
  def hashes(%__MODULE__{index: idx}), do: idx |> Enum.sort_by(fn {_, {pos, _}} -> pos end) |> Enum.map(&elem(&1, 0))

  @doc "A blob's position in the pack: the order in which blobs were committed."
  def position(%__MODULE__{index: idx}, h), do: (case Map.fetch(idx, h) do {:ok, {pos, _}} -> pos; :error -> nil end)

  # ---------------------------------------------------------------- roots

  @doc "The committed root term (nil for an empty store)."
  def root(%__MODULE__{root: nil}), do: nil
  def root(%__MODULE__{root: h} = k), do: (case get_term(k, h) do {:ok, t} -> t; :error -> nil end)

  @doc """
  Make the staged blobs and `term` (the new root) durable, atomically: after
  a crash at any instant, `open/1` returns either this root or the previous
  one, never a mixture. Options: `fault: {stage, bytes}` stops after
  writing `bytes` bytes of the pack append (`:pack`) or of the slot record
  (`:slot`) and raises — the test harness for torn writes.
  """
  def commit(%__MODULE__{} = k, term, opts \\ []) do
    {k, root} =
      case term do
        nil -> {k, nil}
        t -> (fn {h, k2} -> {k2, h} end).(put_term(k, t))
      end

    fault = opts[:fault]
    pack = Path.join(k.dir, @names[k.gen])
    records = k.pending |> Enum.reverse() |> Enum.map(fn {h, d} -> [<<byte_size(d)::32>>, h, d] end) |> IO.iodata_to_binary()

    {:ok, f} = :file.open(pack, [:read, :write, :binary, :raw])

    index =
      try do
        :ok = write_prefix(f, k.len, records, fault, :pack)
        :ok = :file.datasync(f)
        index_records(records, k.len, k.index)
      after
        :file.close(f)
      end

    new_len = k.len + byte_size(records)
    target = if k.slot == :a, do: :b, else: :a
    write_slot(k.dir, target, k.seq + 1, k.gen, new_len, root, fault)
    {:ok, %{k | seq: k.seq + 1, len: new_len, root: root, index: index, slot: target, pending: [], pending_len: 0}}
  end

  defp write_prefix(f, pos, data, {stage, n}, stage) do
    :ok = :file.pwrite(f, pos, binary_part(data, 0, min(n, byte_size(data))))
    :file.datasync(f)
    raise "khazana: injected fault after #{n} bytes of the #{stage}"
  end

  defp write_prefix(_f, _pos, "", _fault, _stage), do: :ok
  defp write_prefix(f, pos, data, _fault, _stage), do: :file.pwrite(f, pos, data)

  defp index_records(<<>>, _pos, idx), do: idx

  defp index_records(<<n::32, h::binary-32, _data::binary-size(n), rest::binary>>, pos, idx),
    do: index_records(rest, pos + 36 + n, Map.put_new(idx, h, {pos + 36, n}))

  defp write_slot(dir, s, seq, gen, len, root, fault) do
    body = <<@magic, seq::64, gen, len::64, (root || <<0::256>>)::binary>>
    rec = body <> :crypto.hash(:sha256, body)
    rec = rec <> :binary.copy(<<0>>, @slot - byte_size(rec))
    {:ok, f} = :file.open(Path.join(dir, "root.#{s}"), [:read, :write, :binary, :raw])

    try do
      :ok = write_prefix(f, 0, rec, fault, :slot)
      :ok = :file.datasync(f)
    after
      :file.close(f)
    end
  end

  # ------------------------------------------------------------ compaction

  @doc """
  Keep only the blobs in `live` (an enumerable of hashes; the root is
  always kept): they are copied into the other pack, which is then
  committed with the same root. Returns `{:ok, store, %{kept, dropped,
  bytes_before, bytes_after}}`. Staged blobs must be committed first.
  """
  def gc(k, live, opts \\ [])

  def gc(%__MODULE__{pending: []} = k, live, opts) do
    live = MapSet.new(live) |> then(&if(k.root, do: MapSet.put(&1, k.root), else: &1))
    other = 1 - k.gen
    path = Path.join(k.dir, @names[other])

    kept = k |> hashes() |> Enum.filter(&MapSet.member?(live, &1))
    records = kept |> Enum.map(fn h -> {:ok, d} = get(k, h); [<<byte_size(d)::32>>, h, d] end) |> IO.iodata_to_binary()

    {:ok, f} = :file.open(path, [:read, :write, :binary, :raw])

    try do
      :ok = write_prefix(f, 0, records, opts[:fault], :pack)
      :ok = :file.datasync(f)
    after
      :file.close(f)
    end

    target = if k.slot == :a, do: :b, else: :a
    write_slot(k.dir, target, k.seq + 1, other, byte_size(records), k.root, opts[:fault])
    k2 = %{k | seq: k.seq + 1, gen: other, len: byte_size(records), index: index_records(records, 0, %{}), slot: target}
    {:ok, k2, %{kept: length(kept), dropped: map_size(k.index) - length(kept), bytes_before: k.len, bytes_after: byte_size(records)}}
  end

  def gc(%__MODULE__{}, _live, _opts), do: {:error, "commit the staged blobs before compacting"}

  # ---------------------------------------------------------- capabilities

  @doc """
  A computed capability (ASAS §6.2): `MAC_K(parts)` under the store's secret
  key, as 22 URL-safe characters (128 bits). Nothing is stored: a presented
  token is checked by recomputing it (`mac_ok?/3`, constant time), and
  bumping a generation counter that is one of the `parts` revokes every
  token issued under the old value at once.
  """
  def mac(%__MODULE__{key: key}, parts) when is_list(parts) do
    :crypto.mac(:hmac, :sha256, key, Vapor.Canonical.encode(parts)) |> binary_part(0, 16) |> Base.url_encode64(padding: false)
  end

  @doc "Whether `token` is the capability for `parts` (constant-time comparison)."
  def mac_ok?(%__MODULE__{} = k, parts, token) when is_binary(token) do
    expect = mac(k, parts)
    byte_size(token) == byte_size(expect) and
      :crypto.hash_equals(expect, token)
  end

  def mac_ok?(_, _, _), do: false

  @doc "Hex of a hash (for display and ids)."
  def hex(h) when is_binary(h), do: Base.encode16(h, case: :lower)

  @doc "A hash from its hex (full 64 digits)."
  def unhex(s) when is_binary(s) and byte_size(s) == 64 do
    case Base.decode16(s, case: :mixed) do
      {:ok, h} -> {:ok, h}
      :error -> :error
    end
  end

  def unhex(_), do: :error

  @doc false
  def slot_size, do: @slot
end
