defmodule Vapor.Ingest.GGML do
  @moduledoc """
  ggml tensor payloads → binary32, bit-identical to llama.cpp's reference
  dequantisation (`gguf-py`'s `quants.py`, tested against it):

  `F32 F16 BF16 Q8_0 Q4_0 Q4_1 Q5_0 Q5_1` (32-weight blocks) and
  `Q4_K Q5_K Q6_K` (256-weight super-blocks).

  The reference computes in binary32, one rounding per operation; here each
  operation is evaluated in binary64 and rounded to binary32 at once, which
  yields the same bits for `+ − ×` (binary64 has more than twice binary32's
  precision plus two bits, so the double rounding is innocuous).
  """
  import Bitwise
  alias Vapor.Ingest.Safetensors

  @doc "Dequantise `bytes` of ggml `type` (`Vapor.Ingest.GGUF` names) to f32 little-endian."
  def dequantize(:f32, bytes), do: bytes
  def dequantize(:f16, bytes), do: Safetensors.f16_to_f32(bytes)
  def dequantize(:bf16, bytes), do: Safetensors.bf16_to_f32(bytes)

  def dequantize(:q8_0, bytes),
    do: for(<<d::binary-2, qs::binary-32 <- bytes>>, into: <<>>, do: block(h(d), for(<<q::signed <- qs>>, do: q), &(&1 * &2)))

  def dequantize(:q4_0, bytes),
    do: for(<<d::binary-2, qs::binary-16 <- bytes>>, into: <<>>, do: block(h(d), Enum.map(nib32(qs), &(&1 - 8)), &(&1 * &2)))

  def dequantize(:q4_1, bytes) do
    for <<d::binary-2, m::binary-2, qs::binary-16 <- bytes>>, into: <<>> do
      {dd, mm} = {h(d), h(m)}
      encode(for q <- nib32(qs), do: f(f(dd * q) + mm))
    end
  end

  def dequantize(:q5_0, bytes) do
    for <<d::binary-2, qh::32-little, qs::binary-16 <- bytes>>, into: <<>> do
      block(h(d), Enum.map(Enum.with_index(nib32(qs)), fn {q, j} -> (q ||| (qh >>> j &&& 1) <<< 4) - 16 end), &(&1 * &2))
    end
  end

  def dequantize(:q5_1, bytes) do
    for <<d::binary-2, m::binary-2, qh::32-little, qs::binary-16 <- bytes>>, into: <<>> do
      {dd, mm} = {h(d), h(m)}
      encode(for {q, j} <- Enum.with_index(nib32(qs)), do: f(f(dd * (q ||| (qh >>> j &&& 1) <<< 4)) + mm))
    end
  end

  def dequantize(:q4_k, bytes) do
    for <<d::binary-2, dmin::binary-2, scales::binary-12, qs::binary-128 <- bytes>>, into: <<>> do
      {sc, mn} = scale_min(scales)
      {dd, dm} = {h(d), h(dmin)}
      # 4 chunks of 32 bytes: low nibbles are sub-block 2c, high nibbles 2c+1
      chunks = for <<c::binary-32 <- qs>>, do: :binary.bin_to_list(c)

      encode(
        for {c, ci} <- Enum.with_index(chunks), {shift, s} <- [{0, 2 * ci}, {4, 2 * ci + 1}], q <- c do
          f(f(f(dd * Enum.at(sc, s)) * (q >>> shift &&& 15)) - f(dm * Enum.at(mn, s)))
        end
      )
    end
  end

  def dequantize(:q5_k, bytes) do
    for <<d::binary-2, dmin::binary-2, scales::binary-12, qh::binary-32, qs::binary-128 <- bytes>>, into: <<>> do
      {sc, mn} = scale_min(scales)
      {dd, dm} = {h(d), h(dmin)}
      hs = :binary.bin_to_list(qh)
      chunks = for <<c::binary-32 <- qs>>, do: :binary.bin_to_list(c)

      encode(
        for {c, ci} <- Enum.with_index(chunks), {shift, s} <- [{0, 2 * ci}, {4, 2 * ci + 1}], {q, l} <- Enum.with_index(c) do
          hi = Enum.at(hs, l) >>> s &&& 1
          f(f(f(dd * Enum.at(sc, s)) * ((q >>> shift &&& 15) ||| hi <<< 4)) - f(dm * Enum.at(mn, s)))
        end
      )
    end
  end

  def dequantize(:q6_k, bytes) do
    for <<ql::binary-128, qh::binary-64, scales::binary-16, d::binary-2 <- bytes>>, into: <<>> do
      dd = h(d)
      sc = for <<s::signed <- scales>>, do: s
      # ql: 2 halves × 64 bytes, each giving 2 × 32 low nibbles; qh: 2 × 32
      # bytes, each giving 4 × 32 two-bit fields
      # groups of 32, in the reference's order: (half, shift, bytes 0–31 | 32–63)
      lows = for <<half::binary-64 <- ql>>, shift <- [0, 4], <<part::binary-32 <- half>>, do: for(<<b <- part>>, do: b >>> shift &&& 15)
      highs = for <<half::binary-32 <- qh>>, shift <- [0, 2, 4, 6], do: for(<<b <- half>>, do: b >>> shift &&& 3)

      qs =
        Enum.zip(lows, highs)
        |> Enum.flat_map(fn {l, hb} -> Enum.zip_with(l, hb, fn a, b -> (a ||| b <<< 4) - 32 end) end)

      encode(for {q, i} <- Enum.with_index(qs), do: f(f(dd * Enum.at(sc, div(i, 16))) * q))
    end
  end

  # 6-bit scales and mins of the 8 sub-blocks (llama.cpp get_scale_min_k4)
  defp scale_min(<<d::binary-4, m::binary-4, md::binary-4>>) do
    [d, m, md] = Enum.map([d, m, md], &:binary.bin_to_list/1)
    sc = Enum.map(d, &(&1 &&& 0x3F)) ++ Enum.zip_with(md, d, fn x, y -> (x &&& 0x0F) ||| (y >>> 2 &&& 0x30) end)
    mn = Enum.map(m, &(&1 &&& 0x3F)) ++ Enum.zip_with(md, m, fn x, y -> x >>> 4 ||| (y >>> 2 &&& 0x30) end)
    {sc, mn}
  end

  # the 32 weights of a 16-byte nibble block: low nibbles first, then high
  defp nib32(qs) do
    bytes = :binary.bin_to_list(qs)
    Enum.map(bytes, &(&1 &&& 15)) ++ Enum.map(bytes, &(&1 >>> 4))
  end

  defp block(d, qs, op), do: encode(for q <- qs, do: f(op.(d, q)))

  defp h(<<x::16-little>>), do: f32_of(Safetensors.f16_bits(x))

  defp f32_of(bits) do
    <<x::float-32>> = <<bits::32>>
    x
  end

  # round to binary32
  defp f(x) do
    <<y::float-32>> = <<x::float-32>>
    y
  end

  defp encode(xs), do: for(x <- xs, into: <<>>, do: <<x::float-32-little>>)

  @doc """
  Quantise f32 little-endian `bytes` (a multiple of 32 values, all finite)
  to Q8_0, bit-identical to gguf-py's reference (`d = max|x|/127` and
  `q = roundf(x · (1/d))` in binary32, ties away from zero; `d` stored as
  binary16, rounded to nearest even).
  """
  def quantize(:q8_0, bytes) do
    for <<blk::binary-128 <- bytes>>, into: <<>> do
      xs = for <<x::float-32-little <- blk>>, do: x
      d = f(Enum.max(Enum.map(xs, &abs/1)) / 127)
      id = if d == 0, do: 0.0, else: f(1 / d)
      qs = for x <- xs, into: <<>>, do: <<roundf(f(x * id))::signed-8>>
      <<d::float-16-little, qs::binary>>
    end
  end

  # C roundf: half away from zero
  defp roundf(x) when x >= 0, do: trunc(Float.floor(x + 0.5))
  defp roundf(x), do: -roundf(-x)

end
