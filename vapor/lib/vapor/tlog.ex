defmodule Vapor.Tlog do
  @moduledoc """
  A transparency log: an append-only, verifiable record of the artifacts
  vapor signs — run attestations, compilation certificates, merge
  receipts, any bytes. Anchoring an attestation here turns "the operator
  signed it" into "the operator signed it **and cannot show anyone a
  different history without being caught**": every reader can check that
  an entry is in the log (inclusion proof) and that today's log extends
  yesterday's (consistency proof); witnesses co-sign a checkpoint only
  after checking that consistency, so a split view needs the witnesses to
  collude too.

  Nothing here is invented: the tree is RFC 9162's (the RFC 6962 hashing
  already used by `Vapor.Merkle`), proofs are generated and verified by the
  RFC's algorithms, checkpoints are C2SP `tlog-checkpoint` documents signed
  in the `signed-note` format (Ed25519), and witnesses add C2SP
  `tlog-cosignature/v1` lines — the formats of the Go checksum database,
  Sigstore's Rekor v2 and the transparency-dev witness network, so a log
  kept by vapor can be witnessed and audited by tools that have never
  heard of vapor. The verifiers are checked against the transparency-dev
  test probes (`test/fixtures/tlog/probes.json`), negative cases included.

  A blockchain adds nothing to these properties but cost and latency: what
  anchoring needs is an append-only commitment plus independent observers,
  which is exactly a log with witnesses. (`docs/TRANSPARENCIA.md`.)

  The log is a value (`%Vapor.Tlog{}`); with `path:` every append is also
  written to an append-only file (length-prefixed entries, `fsync`ed) from
  which `open/2` rebuilds — and re-verifies — the tree.
  """
  import Bitwise
  alias Vapor.Tlog.Note

  defstruct origin: nil, size: 0, levels: {}, entries: :array.new(), path: nil

  @type t :: %__MODULE__{}

  # ------------------------------------------------------------- hashing --

  @doc "RFC 9162 leaf hash: `SHA-256(0x00 ‖ entry)`."
  def leaf_hash(entry) when is_binary(entry), do: :crypto.hash(:sha256, <<0, entry::binary>>)

  defp node(l, r), do: :crypto.hash(:sha256, <<1, l::binary, r::binary>>)

  @empty :crypto.hash(:sha256, "")

  # ---------------------------------------------------------- construction --

  @doc """
  A new, empty log named `origin` (the checkpoint's first line, e.g.
  `"vapor.example/attestations"`). Options: `path` — the append-only file
  (created; must not exist).
  """
  def new(origin, opts \\ []) when is_binary(origin) do
    valid_origin!(origin)
    log = %__MODULE__{origin: origin}

    case Keyword.get(opts, :path) do
      nil ->
        log

      path ->
        if File.exists?(path), do: raise(ArgumentError, "#{path} exists: use Vapor.Tlog.open/2")
        File.write!(path, header(origin))
        %{log | path: path}
    end
  end

  @doc "Reopen a log file, rebuilding the tree from its entries."
  def open(path) do
    with {:ok, bin} <- File.read(path),
         {:ok, origin, rest} <- parse_header(bin),
         {:ok, entries} <- parse_entries(rest, []) do
      log = Enum.reduce(entries, %__MODULE__{origin: origin}, fn e, acc -> elem(push(acc, e), 0) end)
      {:ok, %{log | path: path}}
    end
  end

  defp header(origin), do: <<"vapor-tlog/1\n", origin::binary, "\n">>

  defp parse_header(<<"vapor-tlog/1\n", rest::binary>>) do
    case :binary.split(rest, "\n") do
      [origin, entries] -> {:ok, origin, entries}
      _ -> {:error, :bad_header}
    end
  end

  defp parse_header(_), do: {:error, :not_a_tlog}

  defp parse_entries(<<>>, acc), do: {:ok, Enum.reverse(acc)}
  defp parse_entries(<<n::32, e::binary-size(n), rest::binary>>, acc), do: parse_entries(rest, [e | acc])
  # a torn final record (a crash mid-append) is not part of the log
  defp parse_entries(_torn, acc), do: {:ok, Enum.reverse(acc)}

  defp valid_origin!(o) do
    if o == "" or String.contains?(o, "\n") or not String.valid?(o),
      do: raise(ArgumentError, "an origin is one non-empty line of UTF-8")
  end

  @doc "Append `entry`; returns `{log, index}`."
  def append(%__MODULE__{} = log, entry) when is_binary(entry) do
    if log.path do
      {:ok, f} = :file.open(log.path, [:append, :raw, :binary])
      :ok = :file.write(f, <<byte_size(entry)::32, entry::binary>>)
      :ok = :file.sync(f)
      :ok = :file.close(f)
    end

    push(log, entry)
  end

  # the levels are the complete subtrees: level l holds the hashes of the
  # aligned 2^l-leaf subtrees completed so far (a dense Merkle store)
  defp push(%__MODULE__{size: n} = log, entry) do
    levels = carry(log.levels, 0, leaf_hash(entry))
    {%{log | size: n + 1, levels: levels, entries: :array.set(n, entry, log.entries)}, n}
  end

  defp carry(levels, l, h) do
    levels = if tuple_size(levels) <= l, do: Tuple.append(levels, :array.new()), else: levels
    lvl = elem(levels, l)
    c = :array.size(lvl)
    lvl = :array.set(c, h, lvl)
    levels = put_elem(levels, l, lvl)
    if rem(c + 1, 2) == 0, do: carry(levels, l + 1, node(:array.get(c - 1, lvl), h)), else: levels
  end

  @doc "Entry `i`."
  def entry(%__MODULE__{size: n} = log, i) when i >= 0 and i < n, do: :array.get(i, log.entries)

  # ---------------------------------------------------------------- roots --

  @doc "The root of the first `size` entries (default: all) — RFC 9162 MTH."
  def root(%__MODULE__{} = log, size \\ nil) do
    size = size || log.size
    if size == 0, do: @empty, else: mth(log, 0, size)
  end

  # MTH(D[a:b]); aligned power-of-two ranges are read from the levels
  defp mth(log, a, b) do
    n = b - a

    if (n &&& n - 1) == 0 and rem(a, n) == 0 do
      l = log2(n)
      :array.get(div(a, n), elem(log.levels, l))
    else
      k = split(n)
      node(mth(log, a, a + k), mth(log, a + k, b))
    end
  end

  # the largest power of two strictly below n (n ≥ 2)
  defp split(n), do: 1 <<< (log2(n - 1))
  defp log2(n), do: length(Integer.digits(n, 2)) - 1

  # --------------------------------------------------------------- proofs --

  @doc "RFC 9162 inclusion proof of entry `i` in the tree of `size` entries (leaf to root)."
  def inclusion(%__MODULE__{} = log, i, size \\ nil) do
    size = size || log.size
    if i < 0 or i >= size or size > log.size, do: raise(ArgumentError, "no entry #{i} in a tree of #{size}")
    path(log, i, 0, size)
  end

  defp path(_log, _m, a, b) when b - a == 1, do: []

  defp path(log, m, a, b) do
    k = split(b - a)
    if m < k, do: path(log, m, a, a + k) ++ [mth(log, a + k, b)], else: path(log, m - k, a + k, b) ++ [mth(log, a, a + k)]
  end

  @doc "RFC 9162 consistency proof between the trees of `m` and `n` entries (`0 < m ≤ n`)."
  def consistency(%__MODULE__{} = log, m, n \\ nil) do
    n = n || log.size
    if m < 1 or m > n or n > log.size, do: raise(ArgumentError, "no consistency proof #{m} → #{n}")
    if m == n, do: [], else: subproof(log, m, 0, n, true)
  end

  defp subproof(_log, m, a, b, true) when m == b - a, do: []
  defp subproof(log, m, a, b, false) when m == b - a, do: [mth(log, a, b)]

  defp subproof(log, m, a, b, flag) do
    k = split(b - a)
    if m <= k, do: subproof(log, m, a, a + k, flag) ++ [mth(log, a + k, b)], else: subproof(log, m - k, a + k, b, false) ++ [mth(log, a, a + k)]
  end

  @doc """
  RFC 9162 §2.1.3.2: whether `proof` places `leaf_hash` at `index` of a tree
  of `size` leaves with root `root`. The directions come from the index,
  never from the proof, so a proof cannot be re-labelled.
  """
  def verify_inclusion(leaf_hash, index, size, proof, root) do
    if index >= size or index < 0 or not hashes?([leaf_hash, root | proof]) do
      false
    else
      {r, sn} =
        Enum.reduce_while(proof, {leaf_hash, {index, size - 1}}, fn p, {r, {fn_, sn}} ->
          cond do
            sn == 0 ->
              {:halt, {:fail, 0}}

            (fn_ &&& 1) == 1 or fn_ == sn ->
              r = node(p, r)
              {fn_, sn} = if (fn_ &&& 1) == 0, do: shift_until(fn_, sn), else: {fn_, sn}
              {:cont, {r, {fn_ >>> 1, sn >>> 1}}}

            true ->
              {:cont, {node(r, p), {fn_ >>> 1, sn >>> 1}}}
          end
        end)
        |> case do
          {:fail, _} -> {:fail, 1}
          {r, {_fn, sn}} -> {r, sn}
        end

      r != :fail and sn == 0 and r == root
    end
  end

  # right-shift both until LSB(fn) is set or fn is 0
  defp shift_until(0, sn), do: {0, sn}
  defp shift_until(fn_, sn) when (fn_ &&& 1) == 1, do: {fn_, sn}
  defp shift_until(fn_, sn), do: shift_until(fn_ >>> 1, sn >>> 1)

  @doc """
  RFC 9162 §2.1.4.2: whether `proof` shows that the tree of `size2` with
  root `root2` extends the tree of `size1` with root `root1`.
  """
  def verify_consistency(size1, size2, proof, root1, root2) do
    # (from the empty tree a "consistency proof" is meaningless: refused, as
    # the reference implementation does)
    cond do
      size1 <= 0 or size2 < size1 -> false
      size1 == size2 -> proof == [] and root1 == root2
      proof == [] or not hashes?([root1, root2 | proof]) -> false
      true -> consistent?(size1, size2, proof, root1, root2)
    end
  end

  defp consistent?(size1, size2, proof, root1, root2) do
    proof = if (size1 &&& size1 - 1) == 0, do: [root1 | proof], else: proof
    {fn_, sn} = shift_while_set(size1 - 1, size2 - 1)
    [c0 | rest] = proof

    result =
      Enum.reduce_while(rest, {c0, c0, fn_, sn}, fn c, {fr, sr, fn_, sn} ->
        cond do
          sn == 0 ->
            {:halt, :fail}

          (fn_ &&& 1) == 1 or fn_ == sn ->
            {fr, sr} = {node(c, fr), node(c, sr)}
            {fn_, sn} = if (fn_ &&& 1) == 0, do: shift_until(fn_, sn), else: {fn_, sn}
            {:cont, {fr, sr, fn_ >>> 1, sn >>> 1}}

          true ->
            {:cont, {fr, node(sr, c), fn_ >>> 1, sn >>> 1}}
        end
      end)

    case result do
      {fr, sr, _fn, sn} -> fr == root1 and sr == root2 and sn == 0
      :fail -> false
    end
  end

  defp hashes?(list), do: Enum.all?(list, &(is_binary(&1) and byte_size(&1) == 32))

  defp shift_while_set(fn_, sn) when (fn_ &&& 1) == 1, do: shift_while_set(fn_ >>> 1, sn >>> 1)
  defp shift_while_set(fn_, sn), do: {fn_, sn}

  # ------------------------------------------------------------ checkpoints --

  @doc "The C2SP checkpoint body of the current tree: origin, size, base64 root."
  def checkpoint_body(%__MODULE__{} = log, size \\ nil) do
    size = size || log.size
    "#{log.origin}\n#{size}\n#{Base.encode64(root(log, size))}\n"
  end

  @doc "A signed checkpoint (a signed note) by `signer` (`Vapor.Tlog.Note.keygen/1`)."
  def checkpoint(%__MODULE__{} = log, signer), do: Note.sign(checkpoint_body(log), [signer])

  @doc "Parse a checkpoint body: `{:ok, %{origin, size, root}}`."
  def parse_checkpoint(text) do
    case String.split(text, "\n") do
      [origin, size, root | _ext] when origin != "" ->
        with {n, ""} <- Integer.parse(size),
             true <- n >= 0 and Integer.to_string(n) == size,
             {:ok, r} <- Base.decode64(root),
             32 <- byte_size(r) do
          {:ok, %{origin: origin, size: n, root: r}}
        else
          _ -> {:error, :bad_checkpoint}
        end

      _ ->
        {:error, :bad_checkpoint}
    end
  end

  # -------------------------------------------------------------- receipts --

  @doc """
  Append `entry` and return `{log, receipt}`: everything an offline reader
  needs to check that `entry` is in the log the checkpoint commits to —
  `%{"index", "size", "proof" (base64 list), "checkpoint" (signed note)}`.
  """
  def anchor(%__MODULE__{} = log, entry, signer) do
    {log, i} = append(log, entry)
    {log, receipt(log, i, signer)}
  end

  @doc "A receipt for entry `i` against the current tree."
  def receipt(%__MODULE__{} = log, i, signer) do
    %{"index" => i, "size" => log.size, "proof" => Enum.map(inclusion(log, i), &Base.encode64/1),
      "checkpoint" => checkpoint(log, signer)}
  end

  @doc """
  Verify a receipt for `entry` offline: the checkpoint is signed by one of
  `log_keys` (verifier keys, `Vapor.Tlog.Note`), names `origin` when given,
  and the inclusion proof leads from the entry to its root. With
  `witnesses: keys, quorum: k` at least `k` distinct witnesses must have
  cosigned it. Returns `{:ok, %{origin, size, root, witnesses}}` or
  `{:error, reason}`.
  """
  def verify_receipt(%{"index" => i, "size" => n, "proof" => proof, "checkpoint" => note}, entry, opts) do
    log_keys = Keyword.fetch!(opts, :log_keys)

    with {:ok, %{text: text, signers: signers}} <- Note.open(note, log_keys ++ Keyword.get(opts, :witnesses, [])),
         true <- Enum.any?(log_keys, &(Note.key_name(&1) in Enum.map(signers, fn s -> s.name end) and
                                       Enum.any?(signers, fn s -> s.key == &1 and s.kind == :note end))) || {:error, :not_signed_by_log},
         {:ok, cp} <- parse_checkpoint(text),
         true <- (Keyword.get(opts, :origin) in [nil, cp.origin]) || {:error, :wrong_origin},
         true <- cp.size == n || {:error, :size_mismatch},
         {:ok, hashes} <- decode_all(proof),
         true <- verify_inclusion(leaf_hash(entry), i, n, hashes, cp.root) || {:error, :not_included},
         witnesses = signers |> Enum.filter(&(&1.kind == :cosignature)) |> Enum.map(& &1.name) |> Enum.uniq(),
         true <- length(witnesses) >= Keyword.get(opts, :quorum, 0) || {:error, {:witness_quorum, length(witnesses)}} do
      {:ok, Map.put(cp, :witnesses, witnesses)}
    end
  end

  defp decode_all(list) do
    Enum.reduce_while(list, {:ok, []}, fn b, {:ok, acc} ->
      case Base.decode64(b) do
        {:ok, h} when byte_size(h) == 32 -> {:cont, {:ok, acc ++ [h]}}
        _ -> {:halt, {:error, :bad_proof}}
      end
    end)
  end
end
