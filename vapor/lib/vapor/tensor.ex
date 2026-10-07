defmodule Vapor.Tensor do
  @moduledoc """
  A dense tensor: dtype + shape + one contiguous little-endian binary.

  There are no linked lists of numbers anywhere in the data path: element
  access is `binary_part/3` arithmetic, decoding is typed per dtype
  (the predecessor decoded every tensor as `s8`, destroying IEEE-754 data).

  Dtypes:

    * `:f32` `:f16` — IEEE-754 (f32 values surface as bit patterns, see `Vapor.F32`)
    * `:s32` `:s8` `:u8` — two's complement / unsigned integers
    * `:sb4`  — storage superblock, 150 B per 256 weights (4.6875 bit/w), shape `[rows, k]`
    * `:sb4x` — execution superblock, 152 B per 256 weights (4.75 bit/w), see `Vapor.Quant.Sb4`
  """
  import Bitwise
  alias Vapor.F32

  @enforce_keys [:dtype, :shape, :data]
  defstruct [:dtype, :shape, :data]

  @type dtype :: :f32 | :bf16 | :f16 | :s32 | :s8 | :u8 | :sb4 | :sb4x
  @type t :: %__MODULE__{dtype: dtype, shape: [pos_integer], data: binary}

  @dtypes [:f32, :bf16, :f16, :s32, :s8, :u8, :sb4, :sb4x]
  def dtypes, do: @dtypes

  @doc "Bytes for a dense tensor of `dtype` and `shape` (superblocks for :sb4/:sb4x)."
  @spec nbytes(dtype, [pos_integer]) :: non_neg_integer
  def nbytes(:sb4, [rows, k]) when rem(k, 256) == 0, do: rows * div(k, 256) * 150
  def nbytes(:sb4x, [rows, k]) when rem(k, 256) == 0, do: rows * div(k, 256) * 152
  def nbytes(q, shape) when q in [:sb4, :sb4x],
    do: raise(ArgumentError, "#{q} needs shape [rows, k] with k ≡ 0 (mod 256), got #{inspect(shape, charlists: :as_lists)}")

  def nbytes(dtype, shape), do: Enum.product(shape) * elem_bytes(dtype)

  def elem_bytes(:f32), do: 4
  def elem_bytes(:s32), do: 4
  def elem_bytes(:f16), do: 2
  def elem_bytes(:bf16), do: 2
  def elem_bytes(:s8), do: 1
  def elem_bytes(:u8), do: 1

  @spec new(dtype, [pos_integer], binary) :: t
  def new(dtype, shape, data) when dtype in @dtypes and is_binary(data) do
    unless valid_shape?(shape), do: raise(ArgumentError, "invalid shape #{inspect(shape, charlists: :as_lists)}")
    expect = nbytes(dtype, shape)

    unless byte_size(data) == expect,
      do: raise(ArgumentError, "#{dtype}#{inspect(shape, charlists: :as_lists)} needs #{expect} B, got #{byte_size(data)}")

    %__MODULE__{dtype: dtype, shape: shape, data: data}
  end

  def valid_shape?(s), do: is_list(s) and s != [] and Enum.all?(s, &(is_integer(&1) and &1 > 0))

  @doc "Build from a flat list. `:f32` accepts numbers (rounded once) or `{:bits, b}`."
  @spec from_list(dtype, [pos_integer], list) :: t
  def from_list(:f32, shape, xs) do
    new(:f32, shape, F32.encode(Enum.map(xs, &f32_bits/1)))
  end

  def from_list(:f16, shape, xs), do: new(:f16, shape, for(x <- xs, into: <<>>, do: <<x * 1.0::float-16-little>>))
  def from_list(:s32, shape, xs), do: new(:s32, shape, for(x <- xs, into: <<>>, do: <<x::signed-32-little>>))
  def from_list(:s8, shape, xs), do: new(:s8, shape, for(x <- xs, into: <<>>, do: <<x::signed-8>>))
  def from_list(:u8, shape, xs), do: new(:u8, shape, for(x <- xs, into: <<>>, do: <<x::8>>))

  defp f32_bits({:bits, b}), do: b
  defp f32_bits(x) when is_number(x), do: F32.from_float(x)

  @doc "Typed decode: f32 → bit patterns, integers → integers, f16 → floats."
  @spec to_list(t) :: list
  def to_list(%__MODULE__{dtype: :f32, data: d}), do: F32.decode(d)
  def to_list(%__MODULE__{dtype: :f16, data: d}), do: for(<<x::float-16-little <- d>>, do: x)
  def to_list(%__MODULE__{dtype: :s32, data: d}), do: for(<<x::signed-32-little <- d>>, do: x)
  def to_list(%__MODULE__{dtype: :s8, data: d}), do: for(<<x::signed-8 <- d>>, do: x)
  def to_list(%__MODULE__{dtype: :u8, data: d}), do: for(<<x::8 <- d>>, do: x)

  def to_list(%__MODULE__{dtype: q}) when q in [:sb4, :sb4x],
    do: raise(ArgumentError, "#{q} is packed; use Vapor.Quant.Sb4")

  def to_list(%__MODULE__{dtype: :bf16} = t), do: t |> widen() |> to_list()

  @doc "f32 tensor as Elixir floats (exact widening)."
  def to_floats(%__MODULE__{dtype: :f32} = t), do: Enum.map(to_list(t), &F32.to_float/1)
  def to_floats(%__MODULE__{dtype: :bf16} = t), do: t |> widen() |> to_floats()

  @doc """
  `bf16` weights as the `f32` values they denote — exact (bfloat16 is the
  high half of a binary32): the meaning of a `bf16` constant everywhere.
  """
  def widen(%__MODULE__{dtype: :bf16, shape: s, data: d}), do: new(:f32, s, Vapor.Ingest.Safetensors.bf16_to_f32(d))
  def widen(%__MODULE__{} = t), do: t

  @doc "`f32` → `bf16`, rounding to nearest-even (the identity on widened `bf16` values)."
  def to_bf16(%__MODULE__{dtype: :f32, shape: s, data: d}), do: new(:bf16, s, Vapor.Ingest.Safetensors.f32_to_bf16(d))
  def to_bf16(%__MODULE__{dtype: :bf16} = t), do: t

  @doc "Row `i` of a 2-D tensor as a binary slice (O(1), no copy)."
  @spec row(t, non_neg_integer) :: binary
  def row(%__MODULE__{shape: [rows, _]} = t, i) when i >= 0 and i < rows do
    rb = div(byte_size(t.data), rows)
    binary_part(t.data, i * rb, rb)
  end

  @doc "Maximum absolute value of an integer tensor (for the no-wrap bound)."
  def max_abs(%__MODULE__{dtype: d} = t) when d in [:s8, :s32, :u8] do
    t |> to_list() |> Enum.reduce(0, fn x, m -> max(m, abs(x)) end)
  end

  @doc "Deterministic pseudo-random tensor (splitmix64) — reproducible test data."
  @spec random(dtype, [pos_integer], non_neg_integer, keyword) :: t
  def random(dtype, shape, seed, opts \\ []) do
    n = Enum.product(shape)
    {vals, _} = Enum.map_reduce(1..n, seed, fn _, s -> splitmix(s) end)

    case dtype do
      :f32 ->
        scale = Keyword.get(opts, :scale, 1.0)
        # 24-bit uniform in [-scale, scale): exactly representable inputs
        from_list(:f32, shape, for(v <- vals, do: ((v >>> 40) - 8_388_608) / 8_388_608 * scale))

      :s8 ->
        lim = Keyword.get(opts, :max, 127)
        from_list(:s8, shape, for(v <- vals, do: rem(v, 2 * lim + 1) - lim))

      :u8 ->
        from_list(:u8, shape, for(v <- vals, do: v &&& 0xFF))

      :s32 ->
        lim = Keyword.get(opts, :max, 64)
        from_list(:s32, shape, for(v <- vals, do: rem(v, lim)))
    end
  end

  @doc false
  def splitmix(s) do
    s = s + 0x9E3779B97F4A7C15 &&& 0xFFFFFFFFFFFFFFFF
    z = bxor(s, s >>> 30) * 0xBF58476D1CE4E5B9 &&& 0xFFFFFFFFFFFFFFFF
    z = bxor(z, z >>> 27) * 0x94D049BB133111EB &&& 0xFFFFFFFFFFFFFFFF
    {bxor(z, z >>> 31), s}
  end
end
