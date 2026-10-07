defmodule Vapor.Agent.Journal do
  @moduledoc """
  The record of one run: an append-only sequence of events, each bound to
  its predecessor by hash, the whole committed to by a Merkle root.

      hash₀ = SHA-256(canonical({run_id, 0, kind, data}))
      hashᵢ = SHA-256(canonical({hashᵢ₋₁, i, kind, data}))

  Changing, inserting, dropping or reordering any event changes every hash
  after it (`verify/1` names the first bad one). `root/1` is the RFC 6962
  Merkle root over the event hashes, so a single event can be shown to
  belong to a run with `⌈log₂ n⌉` hashes (`proof/2`) — e.g. to an auditor
  who must not see the others. Encoding is `Vapor.Canonical` (deterministic
  CBOR), so any language can recompute the hashes.

  ## Erasure without breaking the chain (crypto-shredding)

  An immutable log of personal data collides with the right to erasure
  (GDPR art. 17, LGPD art. 18). Values that are personal are stored
  *sealed* (`seal/4`: AES-256-GCM under a key per data subject, the run id
  as associated data); the chain hashes the ciphertext. Erasure destroys the
  subject's key (`Vapor.Agent.Keys.shred/2`): the plaintext becomes
  unrecoverable everywhere at once — backups included — while every hash,
  the root and every proof stay valid. Replay reports those events as
  redacted instead of failing.
  """
  alias Vapor.{Canonical, Merkle}

  defstruct run_id: nil, events: [], head: nil

  @type t :: %__MODULE__{}

  @doc "An empty journal for a run."
  def new(run_id) when is_binary(run_id), do: %__MODULE__{run_id: run_id, events: [], head: run_id}

  @doc "Append an event (`kind` a string, `data` a canonical-encodable map)."
  def append(%__MODULE__{} = j, kind, data) when is_binary(kind) and is_map(data) do
    seq = length(j.events)
    hash = event_hash(j.head, seq, kind, data)
    %{j | events: j.events ++ [%{"seq" => seq, "kind" => kind, "data" => data, "hash" => hash}], head: hash}
  end

  defp event_hash(prev, seq, kind, data), do: Canonical.hex_digest({prev, seq, kind, data})

  @doc "`:ok`, or `{:error, {:broken_at, seq}}` for the first event whose hash does not follow."
  def verify(%__MODULE__{run_id: run, events: evs} = j) do
    {result, head} =
      Enum.reduce_while(Enum.with_index(evs), {:ok, run}, fn {e, i}, {:ok, prev} ->
        if e["seq"] == i and event_hash(prev, i, e["kind"], e["data"]) == e["hash"],
          do: {:cont, {:ok, e["hash"]}},
          else: {:halt, {{:error, {:broken_at, i}}, prev}}
      end)
      |> case do
        {{:error, _} = e, h} -> {e, h}
        {:ok, h} -> {:ok, h}
      end

    if result == :ok and head != j.head, do: {:error, {:broken_at, length(evs)}}, else: result
  end

  @doc "Merkle root (hex) over the event hashes."
  def root(%__MODULE__{events: evs}), do: evs |> Enum.map(&Merkle.leaf(&1["hash"])) |> Merkle.root() |> Base.encode16(case: :lower)

  @doc "Inclusion proof of event `seq`."
  def proof(%__MODULE__{events: evs}, seq), do: evs |> Enum.map(&Merkle.leaf(&1["hash"])) |> Merkle.proof(seq)

  @doc "Check an event's membership against a root (hex)."
  def member?(event, proof, root_hex), do: Merkle.verify(Merkle.leaf(event["hash"]), proof, Base.decode16!(root_hex, case: :lower))

  # --------------------------------------------------------- attestation --

  @doc """
  Attest a run: the node that ran it signs (Ed25519, a
  `Vapor.Certificate.keygen/0` key) its id, length, head and Merkle root.
  A hash chain proves integrity only relative to a head one already trusts;
  the attestation is that trust, portable — publish it (or anchor it in a
  transparency log) and the journal can no longer be rewritten, even by
  whoever stores it.
  """
  def attest(%__MODULE__{} = j, %{public: pub, private: priv}) do
    att = %{"run_id" => j.run_id, "events" => length(j.events), "head" => j.head, "root" => root(j), "key" => pub}
    Map.put(att, "sig", :crypto.sign(:eddsa, :none, attestation_bytes(att), [priv, :ed25519]))
  end

  @doc "Whether `att` is a valid attestation of this journal by one of the `trusted` public keys."
  def attested?(%__MODULE__{} = j, %{"key" => pub, "sig" => sig} = att, trusted) do
    verify(j) == :ok and pub in trusted and
      att["run_id"] == j.run_id and att["events"] == length(j.events) and att["head"] == j.head and att["root"] == root(j) and
      :crypto.verify(:eddsa, :none, attestation_bytes(att), sig, [pub, :ed25519])
  end

  defp attestation_bytes(att), do: Canonical.encode({:vapor_run, 1, att["run_id"], att["events"], att["head"], att["root"], att["key"]})

  # ------------------------------------------------------------- sealing --

  @doc "Seal a value for `subject` (AES-256-GCM, AAD = run id): a canonical map, safe to hash and store."
  def seal(value, subject, keys, run_id) do
    key = Vapor.Agent.Keys.key_for(keys, subject)
    # the nonce is derived, not drawn: sealing is deterministic, so a replayed
    # run reproduces the same ciphertext and the same hashes
    nonce = binary_part(:crypto.hash(:sha256, [key, run_id, Canonical.encode(value)]), 0, 12)
    {ct, tag} = :crypto.crypto_one_time_aead(:aes_256_gcm, key, nonce, Canonical.encode(value), run_id, true)
    %{"sealed" => subject, "nonce" => nonce, "ct" => ct, "tag" => tag}
  end

  @doc "Open a sealed value: `{:ok, value}` or `{:redacted, subject}` once the subject's key is gone."
  def unseal(%{"sealed" => subject, "nonce" => n, "ct" => ct, "tag" => tag}, keys, run_id) do
    case Vapor.Agent.Keys.fetch(keys, subject) do
      {:ok, key} ->
        case :crypto.crypto_one_time_aead(:aes_256_gcm, key, n, ct, run_id, tag, false) do
          :error -> {:error, :tampered}
          plain -> Canonical.decode(plain, atoms: :existing)
        end

      :error ->
        {:redacted, subject}
    end
  end

  def unseal(value, _keys, _run_id), do: {:ok, value}

  # ----------------------------------------------------------- transport --

  @doc "Bytes of a journal (canonical CBOR) — for files, databases, queues."
  def encode(%__MODULE__{} = j), do: Canonical.encode({:vapor_journal, 1, j.run_id, j.events, j.head})

  @doc "Parse bytes from `encode/1` (verify separately)."
  def decode(bin) do
    case Canonical.decode(bin) do
      {:ok, {:vapor_journal, 1, run, events, head}} -> {:ok, %__MODULE__{run_id: run, events: events, head: head}}
      _ -> {:error, :malformed}
    end
  end
end
