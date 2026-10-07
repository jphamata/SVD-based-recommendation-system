defmodule Vapor.Merkle do
  @moduledoc """
  Merkle trees with the hashing of RFC 6962 (Certificate Transparency):
  leaves are `SHA-256(0x00 ‖ data)`, inner nodes `SHA-256(0x01 ‖ left ‖
  right)`, and a tree of `n` leaves splits at the largest power of two
  below `n` — so the root commits to the exact sequence, and inclusion
  proofs have `⌈log₂ n⌉` hashes. The empty tree's root is `SHA-256("")`.

  Used for corpus snapshots (`Vapor.RAG`: a retrieved chunk proves it
  belongs to the corpus the answer claims) and for agent journals
  (`Vapor.Agent.Journal`: any event proves it belongs to the run).
  Verification needs nothing but this module and SHA-256, so a verifier in
  any language can check proofs.
  """

  @doc "Leaf hash of a binary."
  def leaf(data) when is_binary(data), do: :crypto.hash(:sha256, <<0, data::binary>>)

  defp node(l, r), do: :crypto.hash(:sha256, <<1, l::binary, r::binary>>)

  @doc "Root of a list of leaf hashes (already hashed with `leaf/1`)."
  def root([]), do: :crypto.hash(:sha256, "")
  def root([h]), do: h

  def root(hashes) do
    k = split(length(hashes))
    {l, r} = Enum.split(hashes, k)
    node(root(l), root(r))
  end

  @doc "Inclusion proof of leaf `i` (0-based) among `hashes`: the audit path, leaf to root."
  def proof(hashes, i) when i >= 0 and i < length(hashes), do: path(hashes, i)

  defp path([_], 0), do: []

  defp path(hashes, i) do
    k = split(length(hashes))
    {l, r} = Enum.split(hashes, k)
    if i < k, do: path(l, i) ++ [{:right, root(r)}], else: path(r, i - k) ++ [{:left, root(l)}]
  end

  @doc "Check that `leaf_hash` sits at a position whose audit path is `proof` under `root`."
  def verify(leaf_hash, proof, root) do
    Enum.reduce(proof, leaf_hash, fn
      {:right, h}, acc -> node(acc, h)
      {:left, h}, acc -> node(h, acc)
    end) == root
  end

  # the largest power of two strictly below n
  defp split(n), do: Bitwise.bsl(1, length(Integer.digits(n - 1, 2)) - 1)
end
