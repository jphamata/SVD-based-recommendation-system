defmodule Vapor.Quant.MXFP4 do
  @moduledoc """
  **MXFP4** (OCP Microscaling Formats v1.0): blocks of 32 FP4 E2M1
  elements sharing one E8M0 scale. Kimi K3 stores its routed experts this
  way (MXFP4 weights, MXFP8 activations, quantization-aware from SFT on);
  gpt-oss does as well.

  An element is a sign, two exponent bits and one mantissa bit: the
  magnitudes `0, ½, 1, 1½, 2, 3, 4, 6`. A scale byte `s` means `2^(s − 127)`
  (`s = 255` is NaN). Every product `element · 2^(s − 127)` is a dyadic
  number with a two-bit significand, so it is **exactly** a binary32 value
  whenever it lies in binary32's range — from `2^-128` (`½ · 2^-127`, a
  subnormal) up to `1½ · 2^127`. Decoding is therefore exact: the f32 matrix
  *is* the MXFP4 matrix, and a model computes the same bits from it as from
  any other copy of the same values. Each element is checked: a NaN scale,
  or an element whose value overflows binary32 (a magnitude of 6 under
  `s ≥ 253`, of 2 or more under `s = 254`), is refused by name, never
  rounded.

  Layout read (gpt-oss's, which transformers' MXFP4 path uses): a matrix
  `X : [rows, k]` arrives as `X_blocks : u8[rows, k/32, 16]` (two elements
  per byte, the **low nibble first**) and `X_scales : u8[rows, k/32]`.
  Kimi K3's own on-disk layout is not known on this machine (its weights
  are unreachable from here): a checkpoint that packs otherwise is not
  read silently wrong, because `expand/1` only touches tensors named
  `…_blocks` / `…_scales` with exactly these shapes.

  MXFP8 activations are a serving-time choice of the reference stack, not
  part of the weights; vapor keeps activations in binary32 (the canonical
  policy) and says so.
  """
  import Bitwise
  alias Vapor.{F32, Rejection, Tensor}

  # (−1)^sign · magnitude for the 16 codes, as {sign, numerator, exponent}: value = ±num · 2^exp
  @codes for c <- 0..15,
             do: (fn s, m -> {s, elem({0, 1, 1, 3, 1, 3, 1, 3}, m), elem({0, -1, 0, -1, 1, 0, 2, 1}, m)} end).(c >>> 3, c &&& 7)

  @doc "The 16 E2M1 values, in code order."
  def values, do: Enum.map(@codes, fn {s, n, e} -> (if s == 1, do: -1, else: 1) * n * :math.pow(2, e) end)

  @doc """
  Decode one matrix: `blocks : u8[rows, nb, 16]`, `scales : u8[rows, nb]`
  → `{:ok, f32[rows, 32·nb]}`, or a rejection naming the first value that
  is not a finite binary32.
  """
  def decode(name, %Tensor{dtype: :u8, shape: [rows, nb, 16], data: blocks}, %Tensor{dtype: :u8, shape: [rows, nb], data: scales}) do
    table = for s <- 0..255, into: %{}, do: {s, scale_row(s)}

    try do
      data =
        for r <- 0..(rows - 1)//1, b <- 0..(nb - 1)//1, into: <<>> do
          s = :binary.at(scales, r * nb + b)
          row = table[s]
          bytes = binary_part(blocks, (r * nb + b) * 16, 16)
          for <<byte <- bytes>>, into: <<>>, do: <<at(row, byte &&& 15, s)::32-little, at(row, byte >>> 4, s)::32-little>>
        end

      {:ok, Tensor.new(:f32, [rows, nb * 32], data)}
    catch
      {:scale, s} ->
        {:error, Rejection.new({:weight, name <> "_scales"}, "E8M0 scales whose blocks are finite binary32 (got #{s}: #{if s == 255, do: "NaN", else: "overflow"})",
                               "re-export the checkpoint")}
    end
  end

  def decode(name, blocks, scales),
    do: {:error, Rejection.new({:weight, name}, "#{name}_blocks : u8[rows, k/32, 16] and #{name}_scales : u8[rows, k/32] (got #{shape(blocks)}, #{shape(scales)})",
                               "check the checkpoint's MXFP4 layout")}

  defp shape(%Tensor{dtype: d, shape: s}), do: "#{d}#{inspect(s, charlists: :as_lists)}"
  defp shape(nil), do: "absent"

  # the 16 binary32 patterns of one scale (nil where the value is not a finite binary32)
  defp scale_row(255), do: List.to_tuple(List.duplicate(nil, 16))
  defp scale_row(s), do: List.to_tuple(for({sign, n, e} <- @codes, do: exact(sign, n, e + s - 127)))

  defp at(row, code, s), do: elem(row, code) || throw({:scale, s})

  defp exact(sign, 0, _e), do: if(sign == 1, do: 0x8000_0000, else: 0)

  defp exact(sign, n, e) do
    v = n * :math.pow(2, e)
    b = F32.from_float(if sign == 1, do: -v, else: v)
    if F32.finite?(b) and abs(F32.to_float(b)) == v, do: b
  end

  @doc """
  Replace every `X_blocks` / `X_scales` pair of a weight map by `X`, the
  decoded f32 matrix (tensors without a pair are untouched): `{:ok, ws}` or
  the first rejection.
  """
  def expand(ws) when is_map(ws) do
    pairs = for {k, _} <- ws, is_binary(k), String.ends_with?(k, "_blocks"), do: String.replace_suffix(k, "_blocks", "")

    Enum.reduce_while(pairs, {:ok, ws}, fn base, {:ok, acc} ->
      case decode(base, acc[base <> "_blocks"], acc[base <> "_scales"]) do
        {:ok, t} -> {:cont, {:ok, acc |> Map.drop([base <> "_blocks", base <> "_scales"]) |> Map.put(base, t)}}
        err -> {:halt, err}
      end
    end)
  end
end
