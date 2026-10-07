defmodule Vapor.Emit.Link do
  @moduledoc """
  Concatenates compiled kernels into one position-independent code blob.
  Every kernel starts on a 16-byte boundary; padding is a trap instruction
  of the target ISA (`int3`, `udf #0`, the all-zero illegal RISC-V word), so
  a stray jump into padding faults instead of sliding into the next kernel.
  """
  alias Vapor.Emit.Machine.Code

  @spec link([{term, %Code{}}]) :: {binary, %{term => non_neg_integer}}
  def link(codes) do
    {bin, entries} =
      Enum.reduce(codes, {<<>>, %{}}, fn {key, %Code{isa: isa, bin: b}}, {acc, ents} ->
        acc = acc <> pad(isa, rem(16 - rem(byte_size(acc), 16), 16))
        {acc <> b, Map.put(ents, key, byte_size(acc))}
      end)

    {bin, entries}
  end

  defp pad(isa, n) when isa in [:x86_64, :x86_64_avx512], do: :binary.copy(<<0xCC>>, n)
  defp pad(_, n), do: :binary.copy(<<0>>, n)
end
