defmodule Vapor.Quant.Sb4 do
  @moduledoc """
  The `:sb4`-affine superblock (Section 9.3) — storage *and* execution forms.

  ## Storage form `:sb4` (150 B / 256 weights = 4.6875 bit/w, spec Figure 6)

      nibbles[128] | u[8] | v[8] | D:f16 | M0:f16 | MR:f16
      w_i = MR · (M0 + D · u_s · (q_i − v_s)),   s = i div 32

  Nibble order: byte j holds element 2j in the high and 2j+1 in the low nibble.

  ## Execution form `:sb4x` (152 B / 256 weights = 4.75 bit/w)

  Repacked once at ingest so that every substrate runs the same branch-free
  inner loop with no f16 arithmetic and contiguous activation loads:

      per sub-block s (×8): 16 B, byte j = q_{32s+j} | q_{32s+16+j} << 4
      u[8] | v[8] | c1:f32 = MR·D | c0:f32 = MR·M0

  Since `w = c0 + α_s·q − α_s·v_s` with `α_s = c1·u_s`, the declared
  (canonical) dequantization evaluated identically on every substrate is

      α̃_s = fl(c1 · u_s)      β̃_s = fl(c0 − fl(α̃_s · v_s))      w̃_i = fl(α̃_s · q_i)

  and a row product is  `y = Σ_i w̃_i x_i + Σ_s β̃_s X_s`  with `X_s = Σ_{i∈s} x_i`.
  `c1`, `c0` are *exact*: a product of two binary16 values has ≤ 22
  significant bits and fits binary32 without rounding.
  """
  import Bitwise
  alias Vapor.{F32, Tensor}

  @storage_bytes 150
  @exec_bytes 152

  def storage_bytes, do: @storage_bytes
  def exec_bytes, do: @exec_bytes
  def bits_per_weight(:sb4), do: @storage_bytes * 8 / 256
  def bits_per_weight(:sb4x), do: @exec_bytes * 8 / 256

  # ----------------------------------------------------------- storage form --

  @doc "Pack one storage superblock from its raw fields."
  @spec encode([0..15], [0..255], [0..255], float, float, float) :: binary
  def encode(q, u, v, d, m0, mr) when length(q) == 256 and length(u) == 8 and length(v) == 8 do
    unless Enum.all?(q, &(&1 in 0..15)), do: raise(ArgumentError, "weights must be nibbles")
    nib = q |> Enum.chunk_every(2) |> Enum.map(fn [a, b] -> <<a::4, b::4>> end)

    IO.iodata_to_binary([
      nib,
      Enum.map(u, &<<&1::8>>),
      Enum.map(v, &<<&1::8>>),
      <<d::float-16-little, m0::float-16-little, mr::float-16-little>>
    ])
  end

  @doc "Unpack one storage superblock."
  def decode(<<nib::binary-128, u::binary-8, v::binary-8, d::float-16-little,
               m0::float-16-little, mr::float-16-little>>) do
    %{
      q: for(<<a::4, b::4 <- nib>>, x <- [a, b], do: x),
      u: :binary.bin_to_list(u),
      v: :binary.bin_to_list(v),
      d: d,
      m0: m0,
      mr: mr
    }
  end

  @doc """
  Quantize a `[rows, k]` f32 tensor to `:sb4` (k ≡ 0 mod 256). Per sub-block
  affine grid `lo_s + step_s·q`, expressed in the spec's parameters with
  `MR = 1`, `M0 = max_s lo_s`, `D = max_s step_s / 255`.
  """
  @spec quantize(Tensor.t()) :: Tensor.t()
  def quantize(%Tensor{dtype: :f32, shape: [rows, k]} = t) when rem(k, 256) == 0 do
    data =
      for r <- 0..(rows - 1), into: <<>> do
        row = Tensor.row(t, r) |> F32.decode() |> Enum.map(&F32.to_float/1)
        for sb <- Enum.chunk_every(row, 256), into: <<>>, do: quantize_superblock(sb)
      end

    Tensor.new(:sb4, [rows, k], data)
  end

  defp quantize_superblock(w) do
    subs = Enum.chunk_every(w, 32)
    los = Enum.map(subs, &Enum.min/1)
    steps = Enum.map(subs, fn s -> max((Enum.max(s) - Enum.min(s)) / 15, 1.0e-6) end)
    d = f16(Enum.max(steps) / 255)
    m0 = f16(Enum.max(los))
    u = Enum.map(steps, &clamp(round(&1 / d), 1, 255))
    v = Enum.zip_with(los, u, fn lo, us -> clamp(round((m0 - lo) / (d * us)), 0, 255) end)

    q =
      Enum.zip([subs, u, v])
      |> Enum.flat_map(fn {sub, us, vs} ->
        Enum.map(sub, fn x -> clamp(round((x - m0) / (d * us) + vs), 0, 15) end)
      end)

    encode(q, u, v, d, m0, 1.0)
  end

  defp f16(x) do
    <<y::float-16>> = <<x::float-16>>
    if y == 0.0, do: 6.103515625e-05, else: y
  end

  defp clamp(x, lo, hi), do: x |> max(lo) |> min(hi)

  # --------------------------------------------------------- execution form --

  @doc "Repack `:sb4` storage into the `:sb4x` execution layout (exact)."
  @spec to_exec(Tensor.t()) :: Tensor.t()
  def to_exec(%Tensor{dtype: :sb4x} = t), do: t

  def to_exec(%Tensor{dtype: :sb4, shape: shape, data: data}) do
    out =
      for <<blk::binary-@storage_bytes <- data>>, into: <<>> do
        %{q: q, u: u, v: v, d: d, m0: m0, mr: mr} = decode(blk)
        c1 = F32.from_float(mr * d)
        c0 = F32.from_float(mr * m0)
        IO.iodata_to_binary([repack(q), u, v, <<c1::32-little, c0::32-little>>])
      end

    Tensor.new(:sb4x, shape, out)
  end

  defp repack(q) do
    tq = List.to_tuple(q)

    for s <- 0..7, j <- 0..15, into: <<>> do
      lo = elem(tq, 32 * s + j)
      hi = elem(tq, 32 * s + 16 + j)
      <<(lo ||| hi <<< 4)::8>>
    end
  end

  @doc """
  Decode one `:sb4x` superblock into `{nibbles_tuple(256), [{α̃, β̃}] (8)}`,
  with α̃, β̃ computed in the canonical binary32 order.
  """
  def exec_block(<<nib::binary-128, u::binary-8, v::binary-8, c1::32-little, c0::32-little>>) do
    q =
      for <<sub::binary-16 <- nib>>, into: [] do
        bytes = :binary.bin_to_list(sub)
        Enum.map(bytes, &(&1 &&& 0xF)) ++ Enum.map(bytes, &(&1 >>> 4))
      end
      |> List.flatten()
      |> List.to_tuple()

    ab =
      Enum.zip_with(:binary.bin_to_list(u), :binary.bin_to_list(v), fn us, vs ->
        a = F32.mul(c1, F32.from_float(us))
        {a, F32.sub(c0, F32.mul(a, F32.from_float(vs)))}
      end)

    {q, ab}
  end

  @doc "Dequantized weights w̃ of one execution row (binary32 bit patterns)."
  def dequant_row(row_bin) do
    for <<blk::binary-@exec_bytes <- row_bin>>, reduce: [] do
      acc ->
        {q, ab} = exec_block(blk)

        ws =
          for {{a, _b}, s} <- Enum.with_index(ab), i <- 0..31,
              do: F32.mul(a, F32.from_float(elem(q, 32 * s + i)))

        acc ++ ws
    end
  end
end
