defmodule Vapor.Agent.Keys do
  @moduledoc """
  Per-subject data keys for crypto-shredding (`Vapor.Agent.Journal.seal/4`).

  A process holding 256-bit keys by subject id. `shred/2` destroys a key:
  every sealed value of that subject, in every journal and backup, becomes
  ciphertext nobody can open. In production the keys belong in an HSM or a
  KMS (the same three operations); this in-memory holder states the
  contract and is what the tests use.
  """
  use Agent

  def start_link(opts \\ []), do: Agent.start_link(fn -> %{} end, Keyword.take(opts, [:name]))

  @doc """
  The subject's key, created on first use. An erased subject stays erased:
  sealing for it again raises, rather than minting a new key that would make
  the old ciphertexts look tampered with instead of erased.
  """
  def key_for(keys, subject) do
    Agent.get_and_update(keys, fn m ->
      case m do
        %{^subject => :shredded} -> {:shredded, m}
        %{^subject => k} -> {k, m}
        _ -> k = Vapor.Entropy.bytes(32); {k, Map.put(m, subject, k)}
      end
    end)
    |> case do
      :shredded -> raise ArgumentError, "subject #{inspect(subject)} was erased; its data cannot be sealed again"
      k -> k
    end
  end

  @doc "`{:ok, key}` or `:error` (unknown or shredded)."
  def fetch(keys, subject) do
    case Agent.get(keys, &Map.fetch(&1, subject)) do
      {:ok, :shredded} -> :error
      other -> other
    end
  end

  @doc "Destroy a subject's key (erasure); the subject is remembered as erased."
  def shred(keys, subject), do: Agent.update(keys, &Map.put(&1, subject, :shredded))
end
