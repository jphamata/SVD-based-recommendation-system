defmodule Vapor.Modal.Image do
  @moduledoc """
  Images as rows: the image codec of the any-to-any layer.

  An image is `w × h` pixels of `c` channels, values in binary64 (nominally
  `[0, 1]`), stored row-major, channels last (the order of PPM/PGM files).
  Everything a model needs from it is an *exact permutation* of its values:

    * `patches/3` cuts it into `p × p` patches, row-major across the image,
      each flattened channel-major `(c, i, j)` — exactly the dot-product
      order of a convolution with weight `[d, c, p, p]` and stride `p`, so
      a ViT's patch embedding is `linear(patches, W.reshape(d, c·p·p))`;
    * `from_patches/5` is its inverse (bit-exact round trip).

  No dependency: binary PPM (`P6`) and PGM (`P5`), 8-bit, are read and
  written here (`read/1`, `write/2`); any image tool converts to them.

  Generators for tests and controls are deterministic: `scene/3` (soft-edged
  shapes over a gradient — the 1/f-like statistics of natural images),
  `noise/4` (white noise) and `shuffle/2` (the same pixels, permuted —
  the histogram of an image with none of its structure).
  """
  alias Vapor.{Rejection, Tensor}

  @enforce_keys [:w, :h, :c, :px]
  defstruct [:w, :h, :c, :px]

  @type t :: %__MODULE__{w: pos_integer, h: pos_integer, c: 1 | 3, px: tuple}

  @doc "An image from a flat list of `w·h·c` values (row-major, channels last)."
  def new(w, h, c, values) when length(values) == w * h * c, do: %__MODULE__{w: w, h: h, c: c, px: List.to_tuple(Enum.map(values, &(&1 * 1.0)))}

  @doc "Pixel value at column `x`, row `y`, channel `ch`."
  def at(%__MODULE__{w: w, c: c, px: px}, x, y, ch), do: elem(px, (y * w + x) * c + ch)

  def values(%__MODULE__{px: px}), do: Tuple.to_list(px)

  # ------------------------------------------------------------------ files --

  @doc "Read a binary PPM (P6) or PGM (P5) with maxval ≤ 255."
  def read(path) do
    with {:ok, bin} <- File.read(path) do
      parse(bin)
    else
      {:error, why} -> {:error, Rejection.new({:image, path}, "readable (#{inspect(why)})", "check the path")}
    end
  end

  @doc false
  def parse(bin) do
    with {:ok, magic, rest} <- token(bin),
         true <- magic in ["P5", "P6"] || bad("magic P5 or P6"),
         {:ok, w, rest} <- int_token(rest),
         {:ok, h, rest} <- int_token(rest),
         {:ok, maxv, <<_ws, data::binary>>} <- int_token(rest),
         true <- (maxv > 0 and maxv <= 255) || bad("maxval in 1..255"),
         c = if(magic == "P6", do: 3, else: 1),
         true <- byte_size(data) >= w * h * c || bad("#{w * h * c} bytes of pixels") do
      {:ok, new(w, h, c, for(<<b <- binary_part(data, 0, w * h * c)>>, do: b / maxv))}
    end
  end

  defp bad(b), do: {:error, Rejection.new(:pnm, b, "write the image as binary PPM/PGM")}

  defp token(bin) do
    bin = skip(bin)
    case :binary.match(bin, [" ", "\n", "\r", "\t"]) do
      {i, _} -> {:ok, binary_part(bin, 0, i), binary_part(bin, i, byte_size(bin) - i)}
      :nomatch -> bad("a header")
    end
  end

  defp int_token(bin) do
    with {:ok, t, rest} <- token(bin) do
      case Integer.parse(t) do
        {n, ""} when n > 0 -> {:ok, n, rest}
        _ -> bad("a positive integer in the header")
      end
    end
  end

  # whitespace and comments
  defp skip(<<c, rest::binary>>) when c in [?\s, ?\n, ?\r, ?\t], do: skip(rest)
  defp skip(<<?#, rest::binary>>), do: rest |> String.split("\n", parts: 2) |> Enum.at(1, "") |> skip()
  defp skip(bin), do: bin

  @doc "Write an 8-bit binary PPM (3 channels) or PGM (1 channel); values are clamped to [0, 1]."
  def write(path, %__MODULE__{} = img), do: File.write(path, encode(img))

  @doc false
  def encode(%__MODULE__{w: w, h: h, c: c} = img) do
    magic = if c == 3, do: "P6", else: "P5"
    "#{magic}\n#{w} #{h}\n255\n" <> for(v <- values(img), into: <<>>, do: <<round(min(1.0, max(0.0, v)) * 255)>>)
  end

  @doc """
  Write an 8-bit PNG (RGB or grey), each pixel repeated `scale × scale`
  times — zlib and CRC-32 from OTP, nothing else.
  """
  def write_png(path, %__MODULE__{} = img, scale \\ 1), do: File.write(path, png(img, scale))

  @doc false
  def png(%__MODULE__{w: w, h: h, c: c} = img, scale \\ 1) do
    rows = values(img) |> Enum.map(&round(min(1.0, max(0.0, &1)) * 255)) |> Enum.chunk_every(w * c)

    raw =
      for row <- rows, line = row |> Enum.chunk_every(c) |> Enum.flat_map(&List.duplicate(&1, scale)) |> List.flatten(),
          _ <- 1..scale, into: <<>>, do: <<0>> <> :binary.list_to_bin(line)

    colour = if c == 3, do: 2, else: 0
    ihdr = <<w * scale::32, h * scale::32, 8, colour, 0, 0, 0>>
    <<137, 80, 78, 71, 13, 10, 26, 10>> <> chunk("IHDR", ihdr) <> chunk("IDAT", :zlib.compress(raw)) <> chunk("IEND", <<>>)
  end

  defp chunk(type, data), do: <<byte_size(data)::32>> <> type <> data <> <<:erlang.crc32(type <> data)::32>>

  # ---------------------------------------------------------------- patches --

  @doc """
  Patches of `p × p` as rows `f32[n, c·p·p]`, `n = (w/p)·(h/p)`, row-major,
  each `(c, i, j)`. Option `normalize: {mean, std}` (per channel lists, or
  numbers) maps `v ↦ (v − mean)/std` first (in binary64, rounded once).
  """
  def patches(%__MODULE__{w: w, h: h, c: c} = img, p, opts \\ []) when rem(w, p) == 0 and rem(h, p) == 0 do
    {mean, std} = norm_params(Keyword.get(opts, :normalize), c)

    rows =
      for py <- 0..(div(h, p) - 1), px <- 0..(div(w, p) - 1), ch <- 0..(c - 1), i <- 0..(p - 1), j <- 0..(p - 1) do
        (at(img, px * p + j, py * p + i, ch) - elem(mean, ch)) / elem(std, ch)
      end

    Tensor.from_list(:f32, [div(w, p) * div(h, p), c * p * p], rows)
  end

  @doc "The inverse of `patches/3` (without normalisation): rows → image."
  def from_patches(%Tensor{} = rows, w, h, c, p) do
    vals = Tensor.to_floats(rows) |> List.to_tuple()
    k = c * p * p
    per_row = div(w, p)

    px =
      for y <- 0..(h - 1), x <- 0..(w - 1), ch <- 0..(c - 1) do
        r = div(y, p) * per_row + div(x, p)
        elem(vals, r * k + ch * p * p + rem(y, p) * p + rem(x, p))
      end

    new(w, h, c, px)
  end

  defp norm_params(nil, c), do: {Tuple.duplicate(0.0, c), Tuple.duplicate(1.0, c)}
  defp norm_params({m, s}, c) when is_number(m) and is_number(s), do: {Tuple.duplicate(m * 1.0, c), Tuple.duplicate(s * 1.0, c)}
  defp norm_params({m, s}, _c) when is_list(m) and is_list(s), do: {List.to_tuple(m), List.to_tuple(s)}

  # ------------------------------------------------------------- generators --

  @doc """
  A deterministic scene: a vertical gradient `bg = {top, bottom}` (colours
  as lists of `c` values) and shapes painted in order, each with a soft
  one-pixel edge — `{:rect, x0, y0, x1, y1, colour}`,
  `{:disc, cx, cy, r, colour}`.
  """
  def scene(w, h, opts) do
    c = Keyword.get(opts, :channels, 3)
    {top, bottom} = Keyword.get(opts, :bg, {List.duplicate(0.9, c), List.duplicate(0.6, c)})
    shapes = Keyword.get(opts, :shapes, [])

    px =
      for y <- 0..(h - 1), x <- 0..(w - 1) do
        t = if h > 1, do: y / (h - 1), else: 0.0
        base = Enum.zip_with(top, bottom, fn a, b -> a + (b - a) * t end)

        Enum.reduce(shapes, base, fn shape, col ->
          {cov, colour} = coverage(shape, x + 0.5, y + 0.5)
          Enum.zip_with(col, colour, fn a, b -> a + (b - a) * cov end)
        end)
      end

    new(w, h, c, List.flatten(px))
  end

  defp coverage({:rect, x0, y0, x1, y1, col}, x, y) do
    d = Enum.min([x - x0, x1 - x, y - y0, y1 - y])
    {clamp01(d + 0.5), col}
  end

  defp coverage({:disc, cx, cy, r, col}, x, y), do: {clamp01(r - :math.sqrt((x - cx) ** 2 + (y - cy) ** 2) + 0.5), col}

  defp clamp01(v), do: min(1.0, max(0.0, v))

  @doc "White noise in [0, 1) (splitmix64, deterministic)."
  def noise(w, h, c, seed), do: new(w, h, c, Vapor.Modal.Rng.uniform(seed, w * h * c))

  @doc "The same pixels in a seeded random order (structure destroyed, histogram kept)."
  def shuffle(%__MODULE__{w: w, h: h, c: c} = img, seed) do
    pixels = values(img) |> Enum.chunk_every(c)
    new(w, h, c, pixels |> Vapor.Modal.Rng.permute(seed) |> List.flatten())
  end

  @doc "Values as a flat f32 tensor `[1, w·h·c]` (for linear maps over whole images)."
  def to_row(%__MODULE__{} = img), do: Tensor.from_list(:f32, [1, img.w * img.h * img.c], values(img))

  @doc "A flat f32 tensor (or float list) back into an image."
  def from_row(%Tensor{} = t, w, h, c), do: new(w, h, c, Tensor.to_floats(t))
  def from_row(vals, w, h, c) when is_list(vals), do: new(w, h, c, vals)
end
