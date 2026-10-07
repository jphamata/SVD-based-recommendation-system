defmodule Vapor.Tlog.Witness do
  @moduledoc """
  A witness: it cosigns a log's checkpoint only if the checkpoint is signed
  by the log, does not shrink the tree, and is **consistent** with the last
  checkpoint it cosigned for that log (an RFC 9162 consistency proof). A
  log that shows two readers two different histories must therefore get
  two different, mutually inconsistent checkpoints cosigned — which honest
  witnesses refuse. This is the C2SP `tlog-witness` discipline as a value:
  `%Witness{}` holds the trusted log keys and the latest cosigned
  checkpoint per origin.
  """
  alias Vapor.Tlog
  alias Vapor.Tlog.Note

  defstruct signer: nil, logs: %{}, seen: %{}

  @doc "A witness signing with `signer` (a `:cosignature` key) for `logs`: `%{origin => verifier key}`."
  def new(signer, logs), do: %__MODULE__{signer: signer, logs: logs}

  @doc """
  Cosign `note` (a signed checkpoint) given `proof`, the consistency proof
  from the size this witness last cosigned for the log (`[]` the first
  time). Returns `{:ok, witness, cosigned_note}` or `{:error, reason}` —
  `:inconsistent` is the alarm a split view raises.
  """
  def cosign(%__MODULE__{} = w, note, proof, t \\ System.os_time(:second)) do
    {text, _} = Note.split_note(note)

    with {:ok, cp} <- Tlog.parse_checkpoint(text),
         {:ok, vkey} <- Map.fetch(w.logs, cp.origin) |> known(),
         {:ok, %{signers: signers}} <- Note.open(note, [vkey]),
         true <- Enum.any?(signers, &(&1.kind == :note)) || {:error, :not_signed_by_log},
         :ok <- extends(Map.get(w.seen, cp.origin), cp, proof) do
      {:ok, %{w | seen: Map.put(w.seen, cp.origin, cp)}, Note.cosign(note, w.signer, t)}
    end
  end

  defp known({:ok, k}), do: {:ok, k}
  defp known(:error), do: {:error, :unknown_log}

  defp extends(nil, _cp, _proof), do: :ok

  defp extends(old, cp, proof) do
    cond do
      cp.size < old.size -> {:error, :rollback}
      Tlog.verify_consistency(old.size, cp.size, proof, old.root, cp.root) -> :ok
      true -> {:error, :inconsistent}
    end
  end

end
