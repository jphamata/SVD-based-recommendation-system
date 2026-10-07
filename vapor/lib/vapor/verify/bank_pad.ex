defmodule Vapor.Verify.BankPad do
  @moduledoc """
  Theorem 8.1 at run time: shared-memory row padding for tiled kernels,
  using the Lean-extracted `pad_stride/2` (sufficient by `padStride_coprime`,
  minimal by `padStride_minimal`) for power-of-two bank counts.
  """
  import Bitwise

  @doc "Padded row stride (in words) for `banks = 2^m`."
  def padded_stride(stride, banks) when banks > 0 and (banks &&& banks - 1) == 0 do
    stride + Vapor.Extracted.pad_stride(stride, log2(banks))
  end

  @doc "Banks touched by lanes `0..banks-1` reading column `j` (distinct iff conflict-free)."
  def banks_touched(stride, banks, j \\ 0),
    do: for(lane <- 0..(banks - 1), do: rem(Vapor.Extracted.bank_of(stride, lane, banks) + j, banks))

  defp log2(1), do: 0
  defp log2(n), do: 1 + log2(n >>> 1)
end
