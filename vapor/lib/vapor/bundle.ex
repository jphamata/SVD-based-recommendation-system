defmodule Vapor.Bundle do
  @moduledoc """
  A deployable unit: compiled artifacts + signed certificate. Edge nodes
  `unpack/3` it — checking the signature quorum and that every code blob and
  SPIR-V module hashes to what the certificate names — and run it without
  re-running the verification ladder (proof-carrying code: checking the
  evidence is cheap, producing it is not).
  """
  alias Vapor.{Certificate, Compiled}
  alias Vapor.Verify.Digest

  @spec pack(Compiled.t()) :: binary
  def pack(%Compiled{certificate: %Certificate{} = cert} = c) do
    artifacts = %{c | certificate: nil}
    :erlang.term_to_binary({:vapor_bundle, 1, artifacts, Certificate.encode(cert)}, [:deterministic])
  end

  @spec unpack(binary, [binary], pos_integer) :: {:ok, Compiled.t()} | {:error, term}
  def unpack(bin, trusted, quorum \\ 1) do
    with {:vapor_bundle, 1, %Compiled{} = c, cert_bin} <- safe_decode(bin),
         {:ok, cert} <- Certificate.decode(cert_bin),
         :ok <- Certificate.verify(cert, trusted, quorum),
         :ok <- matches?(c, cert) do
      {:ok, %{c | certificate: cert}}
    else
      {:error, _} = e -> e
      _ -> {:error, :malformed_bundle}
    end
  end

  @doc "Content addresses of a compiled program, as certified."
  def digests(%Compiled{} = c) do
    %{
      program: Vapor.Canonical.digest(c.program),
      code: Map.new(c.code, fn {isa, %{blob: b}} -> {isa, Digest.sha256(b)} end),
      spirv: Map.new(c.spirv, fn {k, %{bin: b}} -> {inspect(k), Digest.sha256(b)} end)
    }
  end

  defp matches?(c, %Certificate{payload: p}) do
    if digests(c) == Map.take(p, [:program, :code, :spirv]), do: :ok, else: {:error, :artifact_digest_mismatch}
  end

  defp safe_decode(bin) do
    # Compiled contains atoms (dtypes, kernel names) that exist in this VM;
    # :safe refuses to create new ones.
    :erlang.binary_to_term(bin, [:safe])
  rescue
    ArgumentError -> {:error, :malformed_bundle}
  end
end
