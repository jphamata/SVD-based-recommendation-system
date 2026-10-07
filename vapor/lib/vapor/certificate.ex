defmodule Vapor.Certificate do
  @moduledoc """
  Rung 6 — proof-carrying code: the ladder's evidence as a signed artifact.

  The payload names *what* was certified (SHA-256 of the program, of every
  code blob and SPIR-V module, of the Lean sources the checkers were
  extracted from) and *what was established* (allocations accepted by the
  verified checker, admissibility bounds, parity digests per substrate,
  envelope maxima, the dispatch decision). It is encoded canonically
  (`Vapor.Canonical`: deterministic CBOR, the same bytes on every OTP
  release and recomputable outside the BEAM) and signed with Ed25519
  (`:crypto`, no dependency). Everything in the payload is a deterministic
  function of the program and the toolchain — no timestamps, no host
  measurements — so independent verifier nodes that re-run the ladder
  produce *byte-identical* payloads and can co-sign.

  Quorum: `verify/3` accepts a certificate only with at least `quorum`
  valid signatures from distinct trusted keys (e.g. `f + 1` of `2f + 1`),
  so no single compromised compiler node can certify alone.
  """
  alias Vapor.Verify.Digest

  @enforce_keys [:payload]
  defstruct payload: %{}, signatures: []

  @type key :: %{public: binary, private: binary}
  @type t :: %__MODULE__{payload: map, signatures: [{binary, binary}]}

  @doc "Fresh Ed25519 key pair."
  @spec keygen() :: key
  def keygen do
    {pub, priv} = :crypto.generate_key(:eddsa, :ed25519)
    %{public: pub, private: priv}
  end

  @doc "Key id: first 16 hex digits of SHA-256(public key)."
  def key_id(pub), do: pub |> Digest.sha256() |> binary_part(0, 16)

  @doc "Canonical bytes of the payload (what is signed)."
  def canonical(%__MODULE__{payload: p}), do: canonical(p)
  def canonical(p) when is_map(p), do: Vapor.Canonical.encode({:vapor_certificate, 2, p})

  @doc "Sign (or co-sign) a certificate."
  @spec sign(t, key) :: t
  def sign(%__MODULE__{} = c, %{public: pub, private: priv}) do
    sig = :crypto.sign(:eddsa, :none, canonical(c), [priv, :ed25519])
    %{c | signatures: Enum.uniq_by(c.signatures ++ [{pub, sig}], &elem(&1, 0))}
  end

  @doc """
  Co-sign only if an independently produced certificate has a byte-identical
  payload — the verifier node re-ran the ladder itself and agrees.
  """
  def cosign(%__MODULE__{} = c, %__MODULE__{} = independent, key) do
    if canonical(c) == canonical(independent),
      do: {:ok, sign(c, key)},
      else: {:error, :payload_mismatch}
  end

  @doc """
  Offline verification at an edge node: at least `quorum` valid signatures
  from distinct keys in `trusted` (a list of public keys).
  """
  @spec verify(t, [binary], pos_integer) :: :ok | {:error, term}
  def verify(%__MODULE__{} = c, trusted, quorum \\ 1) do
    msg = canonical(c)
    trusted = MapSet.new(trusted)

    valid =
      c.signatures
      |> Enum.filter(fn {pub, sig} ->
        MapSet.member?(trusted, pub) and :crypto.verify(:eddsa, :none, msg, sig, [pub, :ed25519])
      end)
      |> Enum.uniq_by(&elem(&1, 0))
      |> length()

    if valid >= quorum, do: :ok, else: {:error, {:quorum, valid, quorum}}
  end

  @doc "Serialise for transport (signatures included)."
  def encode(%__MODULE__{payload: p, signatures: s}),
    do: Vapor.Canonical.encode({:vapor_signed_certificate, 2, p, s})

  @doc "Parse a transported certificate (safe decoding: no atom creation)."
  def decode(bin) do
    case Vapor.Canonical.decode(bin) do
      {:ok, {:vapor_signed_certificate, 2, p, s}} when is_map(p) and is_list(s) ->
        {:ok, %__MODULE__{payload: p, signatures: s}}

      _ ->
        {:error, :malformed}
    end
  end
end
