defmodule Vapor.Verify.Digest do
  @moduledoc """
  FNV-1a/64 — the parity digest of Rung 5 (cheap, streaming, identical on
  every node) — and SHA-256 for the certificate's content addresses.
  """
  import Bitwise

  @offset 0xCBF29CE484222325
  @prime 0x100000001B3
  @mask 0xFFFF_FFFF_FFFF_FFFF

  @spec fnv1a64(iodata, non_neg_integer) :: non_neg_integer
  def fnv1a64(data, h \\ @offset)

  def fnv1a64(data, h) when is_binary(data) do
    for <<b <- data>>, reduce: h do
      acc -> bxor(acc, b) * @prime &&& @mask
    end
  end

  def fnv1a64(parts, h), do: parts |> IO.iodata_to_binary() |> fnv1a64(h)

  def hex64(h), do: h |> Integer.to_string(16) |> String.downcase() |> String.pad_leading(16, "0")

  @doc "FNV-1a/64 hex of a tensor's bytes (the parity digest)."
  def tensor(%Vapor.Tensor{data: d}), do: hex64(fnv1a64(d))

  @doc "SHA-256 hex of any binary."
  def sha256(bin), do: :crypto.hash(:sha256, bin) |> Base.encode16(case: :lower)
end
