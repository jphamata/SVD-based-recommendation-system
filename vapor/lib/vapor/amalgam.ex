defmodule Vapor.Amalgam do
  @moduledoc """
  Exact, order-independent reduction: the **amalgam** of many partial sums
  is the same whatever the order, the grouping, the number of workers or
  the topology that produced them (docs/AMALGAMA.md).

  The pain: floating-point addition is commutative but not associative, so
  a distributed sum (an all-reduce of gradients, a merge of partial
  results from nodes that come and go) has bits that depend on *who added
  what first*. The canonical policy of 0.14 answers by **fixing the
  shape** of the reduction (16 lanes, a fixed binary tree over micro-batch
  indices, a power-of-two count): correct, but a constraint on everything
  around it — the number of micro-batches, the schedule, failover, the
  row-parallel split that `Vapor.Shard` had to refuse.

  First principles: every binary format value is an integer multiple of
  its smallest subnormal, `2^qmin`. So a sum of `n` values is an integer
  times `2^qmin`, and integer addition **is** associative. The BEAM's
  arbitrary-precision integers make that integer a Kulisch accumulator
  with no limbs to manage: a cell is `Σ mᵢ·2^(eᵢ − qmin)`, exact; `merge/2`
  adds cells; `round/1` rounds **once**, to nearest-even, with gradual
  underflow and overflow to ±∞. The result is the correctly rounded value
  of the true real sum — a definition that names no order at all, so it is
  also the canonical one.

  IEEE special values keep their algebra, order-free: a NaN or `+∞` with
  `−∞` gives NaN; an infinity dominates finite terms; the sum of only `−0`
  terms is `−0`, any cancellation or `+0` is `+0` (IEEE 754 §6.3). A cell is
  `:empty | :nzero | integer | :pinf | :ninf | :nan`, and `merge/2` is a
  commutative monoid on cells with `:empty` as identity.

  Not a free lunch, and stated: finite terms whose true sum is finite never
  overflow (IEEE left-to-right may, on the way); the cost is a bignum
  add per element (≈ 0.1 µs on the BEAM), so this is a reduction for the
  control plane — gradients across micro-batches and nodes, partial results
  across a cluster — not for the inner loop of a kernel, whose
  exact form needs integer operations in all five emitters (docs/TODO.md).
  """
  import Bitwise
  alias Vapor.{Canonical, Tensor}

  # {precision p (with the hidden bit), exponent bits, bytes}
  @formats %{f16: {11, 5, 2}, bf16: {8, 8, 2}, f32: {24, 8, 4}, f64: {53, 11, 8}}

  defstruct format: :f32, scale: 149, n: 0, count: 0, cells: []

  @type cell :: :empty | :nzero | integer | :pinf | :ninf | :nan
  @type t :: %__MODULE__{format: atom, scale: non_neg_integer, n: non_neg_integer, count: non_neg_integer, cells: [cell]}

  @doc "The formats an amalgam reads and rounds to."
  def formats, do: Map.keys(@formats)

  @doc "`-qmin` of a format: every value is an integer multiple of `2^-scale`."
  def scale(fmt) do
    {p, eb, _} = fmt!(fmt)
    bias = (1 <<< (eb - 1)) - 1
    bias - 1 + (p - 1)
  end

  @doc """
  An empty amalgam of `n` cells. Option `products: true` makes the cells
  hold sums of **products** of two values of the format (scale doubled),
  for exact dot products.
  """
  def new(fmt, n, opts \\ []) when is_integer(n) and n >= 0 do
    s = scale(fmt)
    s = if Keyword.get(opts, :products, false), do: 2 * s, else: s
    %__MODULE__{format: fmt, scale: s, n: n, count: 0, cells: List.duplicate(:empty, n)}
  end

  # ------------------------------------------------------------ decoding --

  @doc """
  The cell of one value given by its bits: its exact value as an integer
  multiple of `2^-scale(fmt)`, or the special it is.
  """
  def cell(fmt, bits) do
    {p, eb, _} = fmt!(fmt)
    fbits = p - 1
    emask = (1 <<< eb) - 1
    sign = bits >>> (fbits + eb) &&& 1
    e = bits >>> fbits &&& emask
    f = bits &&& (1 <<< fbits) - 1

    cond do
      e == emask and f != 0 -> :nan
      e == emask -> if sign == 1, do: :ninf, else: :pinf
      e == 0 and f == 0 -> if sign == 1, do: :nzero, else: 0
      true ->
        m = if e == 0, do: f, else: (f ||| 1 <<< fbits) <<< (e - 1)
        if sign == 1, do: -m, else: m
    end
  end

  @doc "Decode a little-endian binary of a format into its list of bit patterns."
  def bits_of(fmt, bin) do
    {_, _, w} = fmt!(fmt)
    size = w * 8
    if rem(byte_size(bin), w) != 0, do: raise(ArgumentError, "#{byte_size(bin)} bytes is not a whole number of #{fmt} values")
    for <<b::size(size)-little <- bin>>, do: b
  end

  # ------------------------------------------------------- accumulating --

  @doc """
  Add one vector of `n` values (a `%Tensor{}` of the format, a binary, or
  a list of bit patterns) into every cell. `count` grows by one.
  """
  def add(%__MODULE__{} = a, %Tensor{dtype: dt, data: d}) do
    if dt != a.format, do: raise(ArgumentError, "a #{dt} tensor added to a #{a.format} amalgam")
    add(a, d)
  end

  def add(%__MODULE__{} = a, bin) when is_binary(bin), do: add(a, bits_of(a.format, bin))

  def add(%__MODULE__{n: n} = a, bits) when is_list(bits) do
    if length(bits) != n, do: raise(ArgumentError, "#{length(bits)} values added to an amalgam of #{n}")
    shift = a.scale - scale(a.format)
    fmt = a.format
    cells = Enum.zip_with(a.cells, bits, fn c, b -> join(c, lift(cell(fmt, b), shift)) end)
    %{a | cells: cells, count: a.count + 1}
  end

  @doc """
  Add the elementwise **products** `x[i]·y[i]` (exact) into the cells of a
  `products: true` amalgam; `count` grows by one.
  """
  def add_products(%__MODULE__{n: n, format: fmt} = a, xs, ys) do
    xs = as_bits(fmt, xs)
    ys = as_bits(fmt, ys)
    if a.scale != 2 * scale(fmt), do: raise(ArgumentError, "add_products/3 needs an amalgam made with products: true")
    if length(xs) != n or length(ys) != n, do: raise(ArgumentError, "vectors of #{length(xs)}, #{length(ys)} into an amalgam of #{n}")
    cells = Enum.zip_with([a.cells, xs, ys], fn [c, x, y] -> join(c, product(cell(fmt, x), cell(fmt, y))) end)
    %{a | cells: cells, count: a.count + 1}
  end

  @doc """
  Merge two amalgams of the same shape: exact, associative and commutative
  — any reduction tree, any arrival order, any regrouping gives the same
  cells. `count`s add.
  """
  def merge(%__MODULE__{format: f, scale: s, n: n} = a, %__MODULE__{format: f, scale: s, n: n} = b),
    do: %{a | cells: Enum.zip_with(a.cells, b.cells, &join/2), count: a.count + b.count}

  def merge(%__MODULE__{} = a, %__MODULE__{} = b),
    do: raise(ArgumentError, "merging #{a.format}[#{a.n}]/2^-#{a.scale} with #{b.format}[#{b.n}]/2^-#{b.scale}")

  @doc "Merge a non-empty list of amalgams (the order is irrelevant by construction)."
  def merge_all([a | rest]), do: Enum.reduce(rest, a, &merge(&2, &1))

  @doc "The order-free sum of a list of vectors of a format: `{:ok, amalgam}`."
  def sum(fmt, [first | _] = vectors) do
    n = length(as_bits(fmt, first))
    Enum.reduce(vectors, new(fmt, n), &add(&2, &1))
  end

  # the commutative monoid on cells
  defp join(:empty, c), do: c
  defp join(c, :empty), do: c
  defp join(:nan, _), do: :nan
  defp join(_, :nan), do: :nan
  defp join(:pinf, :ninf), do: :nan
  defp join(:ninf, :pinf), do: :nan
  defp join(:pinf, _), do: :pinf
  defp join(_, :pinf), do: :pinf
  defp join(:ninf, _), do: :ninf
  defp join(_, :ninf), do: :ninf
  defp join(:nzero, :nzero), do: :nzero
  defp join(:nzero, c), do: c
  defp join(c, :nzero), do: c
  defp join(a, b), do: a + b

  defp lift(c, 0), do: c
  defp lift(c, k) when is_integer(c), do: c <<< k
  defp lift(c, _), do: c

  # IEEE products of the specials: 0·∞ = NaN, signs multiply
  defp product(:nan, _), do: :nan
  defp product(_, :nan), do: :nan
  defp product(x, y) when x in [:pinf, :ninf] or y in [:pinf, :ninf] do
    if zero_cell?(x) or zero_cell?(y), do: :nan, else: if(neg_cell?(x) == neg_cell?(y), do: :pinf, else: :ninf)
  end
  # finite factors: a zero factor gives a zero whose sign is the xor of the signs
  defp product(x, y) do
    if zero_cell?(x) or zero_cell?(y) do
      if neg_cell?(x) != neg_cell?(y), do: :nzero, else: 0
    else
      x * y
    end
  end

  defp zero_cell?(c), do: c == 0 or c == :nzero
  defp neg_cell?(c), do: c == :ninf or c == :nzero or (is_integer(c) and c < 0)

  defp as_bits(_fmt, %Tensor{data: d, dtype: dt}), do: as_bits(dt, d)
  defp as_bits(fmt, bin) when is_binary(bin), do: bits_of(fmt, bin)
  defp as_bits(_fmt, list) when is_list(list), do: list

  # ------------------------------------------------------------ rounding --

  @doc "Round every cell once to the format: a `%Tensor{}` (shape `[1, n]`, or `shape:`) for f32/bf16/f16, a binary for f64."
  def round(%__MODULE__{} = a, opts \\ []) do
    bits = Enum.map(a.cells, &round_cell(&1, a.scale, 1, a.format))
    pack(a.format, bits, Keyword.get(opts, :shape, [1, a.n]))
  end

  @doc """
  The correctly rounded **mean**: every cell divided by `count` (or
  `by:`), rounded once — not the rounded sum divided again.
  """
  def mean(%__MODULE__{} = a, opts \\ []) do
    d = Keyword.get(opts, :by, a.count)
    if not (is_integer(d) and d > 0), do: raise(ArgumentError, "mean of an amalgam of #{d} terms")
    bits = Enum.map(a.cells, &round_cell(&1, a.scale, d, a.format))
    pack(a.format, bits, Keyword.get(opts, :shape, [1, a.n]))
  end

  @doc "The bit patterns of the rounded cells (no packing)."
  def round_bits(%__MODULE__{} = a, by \\ 1), do: Enum.map(a.cells, &round_cell(&1, a.scale, by, a.format))

  defp pack(fmt, bits, shape) do
    {_, _, w} = fmt!(fmt)
    size = w * 8
    bin = for b <- bits, into: <<>>, do: <<b::size(size)-little>>
    if fmt == :f64, do: bin, else: Tensor.new(fmt, shape, bin)
  end

  defp round_cell(:empty, _s, _d, _fmt), do: 0
  defp round_cell(:nzero, _s, _d, fmt), do: sign_bit(fmt)
  defp round_cell(:nan, _s, _d, fmt), do: quiet_nan(fmt)
  defp round_cell(:pinf, _s, _d, fmt), do: inf(fmt)
  defp round_cell(:ninf, _s, _d, fmt), do: sign_bit(fmt) ||| inf(fmt)
  defp round_cell(0, _s, _d, _fmt), do: 0
  defp round_cell(m, s, d, fmt), do: round_rational(m, s, d, fmt)

  @doc """
  Round the rational `m / (d · 2^s)` (`d > 0`) to a binary format:
  round-to-nearest-even, gradual underflow, ±∞ on overflow. Returns the
  bit pattern. The one rounding of an amalgam.
  """
  def round_rational(0, _s, _d, _fmt), do: 0

  def round_rational(m, s, d, fmt) when is_integer(m) and is_integer(s) and is_integer(d) and d > 0 do
    {p, eb, _} = fmt!(fmt)
    bias = (1 <<< (eb - 1)) - 1
    qmin = 1 - bias - (p - 1)
    sign = if m < 0, do: sign_bit(fmt), else: 0
    num = abs(m)
    den = d <<< max(s, 0)
    num = num <<< max(-s, 0)
    # e = ⌊log₂(num/den)⌋
    e0 = bits(num) - bits(den)
    e = if cmp_shift(num, den, e0) == :lt, do: e0 - 1, else: e0
    q = max(e - (p - 1), qmin)
    # mant = round(num / (den · 2^q))
    {n2, d2} = if q >= 0, do: {num, den <<< q}, else: {num <<< -q, den}
    t = div(n2, d2)
    r2 = 2 * rem(n2, d2)
    mant = if r2 > d2 or (r2 == d2 and (t &&& 1) == 1), do: t + 1, else: t
    {mant, q} = if mant == 1 <<< p, do: {1 <<< (p - 1), q + 1}, else: {mant, q}
    efield = q + (p - 1) + bias

    cond do
      mant < 1 <<< (p - 1) -> sign ||| mant
      efield >= (1 <<< eb) - 1 -> sign ||| inf(fmt)
      true -> sign ||| efield <<< (p - 1) ||| (mant - (1 <<< (p - 1)))
    end
  end

  # compare num with den·2^k
  defp cmp_shift(num, den, k) when k >= 0, do: cmp(num, den <<< k)
  defp cmp_shift(num, den, k), do: cmp(num <<< -k, den)
  defp cmp(a, b) when a < b, do: :lt
  defp cmp(a, b) when a > b, do: :gt
  defp cmp(_, _), do: :eq

  defp bits(0), do: 0
  defp bits(a), do: bits(a, 0)
  defp bits(a, n) when a >= 1 <<< 256, do: bits(a >>> 256, n + 256)
  defp bits(a, n) when a >= 1 <<< 16, do: bits(a >>> 16, n + 16)
  defp bits(0, n), do: n
  defp bits(a, n), do: bits(a >>> 1, n + 1)

  defp sign_bit(fmt) do
    {_, _, w} = fmt!(fmt)
    1 <<< (8 * w - 1)
  end

  defp inf(fmt) do
    {p, eb, _} = fmt!(fmt)
    ((1 <<< eb) - 1) <<< (p - 1)
  end

  defp quiet_nan(fmt) do
    {p, _, _} = fmt!(fmt)
    inf(fmt) ||| 1 <<< (p - 2)
  end

  # ------------------------------------------------------- dot products --

  @doc """
  The exact dot product of two vectors of a format, rounded once: the
  correctly rounded value of `Σ xᵢyᵢ` — independent of lane count, tile
  size, sharding and order. Returns the bit pattern.
  """
  def dot(fmt, xs, ys) do
    xs = as_bits(fmt, xs)
    ys = as_bits(fmt, ys)
    if length(xs) != length(ys), do: raise(ArgumentError, "dot of #{length(xs)} and #{length(ys)} values")
    s = 2 * scale(fmt)
    acc = Enum.zip_with(xs, ys, fn x, y -> product(cell(fmt, x), cell(fmt, y)) end) |> Enum.reduce(:empty, &join(&2, &1))
    round_cell(acc, s, 1, fmt)
  end

  @doc """
  A partial dot product as a one-cell amalgam: shards of a contraction
  (row-parallel, Megatron's second half) each return one, and their merge
  — in any order, on any node — rounds to the bits of the unsharded `dot/3`.
  """
  def partial_dot(fmt, xs, ys) do
    xs = as_bits(fmt, xs)
    ys = as_bits(fmt, ys)
    a = new(fmt, 1, products: true)
    acc = Enum.zip_with(xs, ys, fn x, y -> product(cell(fmt, x), cell(fmt, y)) end) |> Enum.reduce(:empty, &join(&2, &1))
    %{a | cells: [acc], count: 1}
  end

  # ------------------------------------------------------------ the wire --

  @doc """
  Canonical bytes of an amalgam (CBOR, `Vapor.Canonical`): what one node
  sends another, what a certificate signs. Two amalgams with the same
  cells have the same bytes, whatever produced them.
  """
  def to_wire(%__MODULE__{} = a) do
    Canonical.encode(%{"amalgam" => 1, "format" => Atom.to_string(a.format), "scale" => a.scale, "count" => a.count,
                       "cells" => Enum.map(a.cells, &wire_cell/1)})
  end

  @doc """
  Read the bytes of `to_wire/1` from an untrusted peer: `{:ok, amalgam}` or
  `{:error, why}`. A cell larger than `count` terms of the largest finite
  value could produce is refused — a peer cannot inject a sum it could
  not have computed.
  """
  def from_wire(bin) when is_binary(bin) do
    with {:ok, %{"amalgam" => 1, "format" => f, "scale" => s, "count" => c, "cells" => cells}} <- decode(bin),
         {:ok, fmt} <- format_named(f),
         true <- s in [scale(fmt), 2 * scale(fmt)] || {:error, "scale #{s} is not one of #{fmt}'s"},
         true <- (is_integer(c) and c >= 0) || {:error, "bad count"},
         {:ok, cells} <- read_cells(cells, cap(fmt, s, c)) do
      {:ok, %__MODULE__{format: fmt, scale: s, n: length(cells), count: c, cells: cells}}
    else
      {:error, _} = e -> e
      _ -> {:error, "not an amalgam"}
    end
  end

  defp decode(bin) do
    case Canonical.decode(bin) do
      {:ok, t} -> {:ok, t}
      {:error, why} -> {:error, "undecodable: #{inspect(why)}"}
    end
  rescue
    _ -> {:error, "undecodable bytes"}
  end

  defp format_named(f) do
    case Enum.find(formats(), &(Atom.to_string(&1) == f)) do
      nil -> {:error, "unknown format #{inspect(f)}"}
      fmt -> {:ok, fmt}
    end
  end

  # |cell| ≤ count · (largest finite)·2^scale  (products: the square)
  defp cap(fmt, s, c) do
    {p, eb, _} = fmt!(fmt)
    bias = (1 <<< (eb - 1)) - 1
    maxv = ((1 <<< p) - 1) <<< (bias - (p - 1) + scale(fmt))
    if s == scale(fmt), do: c * maxv, else: c * maxv * maxv
  end

  defp read_cells(cells, cap) when is_list(cells) do
    Enum.reduce_while(cells, {:ok, []}, fn c, {:ok, acc} ->
      case c do
        i when is_integer(i) and abs(i) <= cap -> {:cont, {:ok, [i | acc]}}
        i when is_integer(i) -> {:halt, {:error, "a cell exceeds what its count of terms can sum to"}}
        s when s in ["empty", "-0", "nan", "+inf", "-inf"] -> {:cont, {:ok, [special(s) | acc]}}
        _ -> {:halt, {:error, "a cell is neither an integer nor a special"}}
      end
    end)
    |> case do
      {:ok, l} -> {:ok, Enum.reverse(l)}
      e -> e
    end
  end

  defp read_cells(_, _), do: {:error, "cells is not a list"}

  defp wire_cell(i) when is_integer(i), do: i
  defp wire_cell(:empty), do: "empty"
  defp wire_cell(:nzero), do: "-0"
  defp wire_cell(:nan), do: "nan"
  defp wire_cell(:pinf), do: "+inf"
  defp wire_cell(:ninf), do: "-inf"

  defp special("empty"), do: :empty
  defp special("-0"), do: :nzero
  defp special("nan"), do: :nan
  defp special("+inf"), do: :pinf
  defp special("-inf"), do: :ninf

  defp fmt!(fmt) do
    case Map.fetch(@formats, fmt) do
      {:ok, v} -> v
      :error -> raise ArgumentError, "unknown format #{inspect(fmt)} (one of #{inspect(formats())})"
    end
  end
end
