defmodule Vapor.Vision.Upscale do
  @moduledoc """
  **An AI upscaler that cannot contradict its input.**

  Learned super-resolution invents detail — that is its job — and nothing
  stops it from inventing detail the low-resolution image rules out: a
  letter that becomes another, an edge that moves. Here every result
  satisfies, by construction,

      D(upscale(x)) = x

  where `D` is the declared degradation — the mean of each 2×2 block (area
  downsampling): shrink the output and you get the input back, to binary64
  rounding. The network only chooses *among the images consistent with
  what was seen*. The projection is exact and cheap for this `D`: the
  residual `x − mean(block)` is added to the four pixels of the block (and
  redistributed among the unsaturated ones when a pixel would leave
  [0, 1]). Standard interpolators are not consistent: Lanczos and bicubic
  shrink back to an image that differs from the input (the control in the
  tests).

  The network is small and trained **by vapor itself** (`Vapor.Learn`:
  autodiff and AdamW as one recurrent program, bit-reproducible): a 7×7
  neighbourhood of the low-resolution luma (relative to its centre) → two
  hidden layers of 64 → a correction of the four sub-pixels of the
  Lanczos upscaling (on smooth regions the network learns to add nothing). Colour
  images: the luma through the network, the chroma by Lanczos, all three
  projected — `D` is linear and commutes with the colour transform, so the
  RGB result is consistent too (up to clipping into [0, 1]).

  Trained on photographs, scientific images and text pages (scikit-image's
  bundled data and rendered fonts); measured on images and fonts never seen
  (`mix vapor.upscale eval`, `docs/bench/QUALITY.md` §5e).
  """
  alias Vapor.{Learn, Rejection}
  alias Vapor.Modal.Image
  alias Vapor.Studio.Resample

  @r 3
  @sizes [49, 64, 64, 4]

  # ------------------------------------------------------------- features --

  @doc "The degradation `D`: the mean of each 2×2 block (sides must be even)."
  def downsample(%Image{w: w, h: h, c: c, px: px}) when rem(w, 2) == 0 and rem(h, 2) == 0 do
    w2 = div(w, 2)
    vals =
      for y <- 0..(div(h, 2) - 1), x <- 0..(w2 - 1), k <- 0..(c - 1) do
        at = fn dx, dy -> elem(px, ((2 * y + dy) * w + 2 * x + dx) * c + k) end
        (at.(0, 0) + at.(1, 0) + at.(0, 1) + at.(1, 1)) / 4
      end

    %Image{w: w2, h: div(h, 2), c: c, px: List.to_tuple(vals)}
  end

  @doc false
  # one row per low-resolution pixel (or per `{x, y}` in `at`): the 7×7
  # neighbourhood minus its centre, the centre value in the middle slot
  def features(%Image{w: w, h: h, c: 1} = img, at \\ nil) do
    at = at || for(y <- 0..(h - 1), x <- 0..(w - 1), do: {x, y})
    Enum.map(at, &feature(img, &1))
  end

  defp feature(%Image{w: w, h: h, px: px}, {x, y}) do
    c = elem(px, y * w + x)
    for dy <- -@r..@r, dx <- -@r..@r do
      if dx == 0 and dy == 0, do: c, else: elem(px, refl(y + dy, h) * w + refl(x + dx, w)) - c
    end
  end

  defp refl(i, n) when i < 0, do: min(-i, n - 1)
  defp refl(i, n) when i >= n, do: max(2 * (n - 1) - i, 0)
  defp refl(i, _), do: i

  @doc false
  # the four sub-pixels (2×2, row-major) of a low-resolution pixel, relative
  # to the Lanczos upscaling there (the network learns a correction)
  def target(%Image{w: w, px: hr}, %Image{px: base}, {x, y}) do
    for dy <- 0..1, dx <- 0..1, do: (fn i -> elem(hr, i) - elem(base, i) end).((2 * y + dy) * w + 2 * x + dx)
  end

  @doc "The base the network corrects: Lanczos-3 ×2 (`Vapor.Studio.Resample`)."
  def base(%Image{w: w, h: h} = lr, opts \\ []), do: Resample.resize(lr, 2 * w, 2 * h, "lanczos", worker: Keyword.get(opts, :worker))

  # ------------------------------------------------------------ the model --

  @doc "The shipped model (`priv/upscale`)."
  def default do
    dir = Path.join(to_string(:code.priv_dir(:vapor)), "upscale")
    load(dir)
  end

  @doc "Load a model directory (`model.safetensors`, `config.json`)."
  def load(dir) do
    with {:ok, ts} <- Vapor.Ingest.Safetensors.read(Path.join(dir, "model.safetensors")),
         {:ok, cfg} <- File.read(Path.join(dir, "config.json")),
         {:ok, cfg} <- Vapor.JSON.decode(cfg) do
      {:ok, %{net: Learn.from_tensors(cfg["sizes"], ts), config: cfg}}
    else
      _ -> {:error, Rejection.new({:upscale, dir}, "model.safetensors and config.json", "train one: mix vapor.upscale train")}
    end
  end

  @doc "Write a model directory."
  def save(%Learn{} = net, dir, info) do
    File.mkdir_p!(dir)
    File.write!(Path.join(dir, "model.safetensors"), Vapor.Ingest.Safetensors.encode(Learn.tensors(net)))
    cfg = Map.merge(%{"kind" => "vapor-consistent-upscaler", "scale" => 2, "sizes" => net.sizes, "radius" => @r, "degradation" => "mean of 2x2 blocks",
                      "weights_sha256" => Learn.digest(net)}, info)
    File.write!(Path.join(dir, "config.json"), Vapor.JSON.encode(cfg))
    cfg
  end

  # --------------------------------------------------------------- upscale --

  @doc """
  Upscale ×2 (or ×4: twice). Options: `model` (default: the shipped one),
  `project` (true — the consistency projection; false only for
  measurement), `worker`.
  """
  def upscale(img, opts \\ [])

  def upscale(%Image{} = img, opts) do
    case Keyword.get(opts, :factor, 2) do
      4 -> img |> upscale(Keyword.put(opts, :factor, 2)) |> upscale(Keyword.put(opts, :factor, 2))
      2 -> x2(img, opts)
    end
  end

  defp x2(%Image{c: 1} = img, opts) do
    model = Keyword.get_lazy(opts, :model, fn -> {:ok, m} = default(); m end)
    rows = features(img)
    pred = Learn.predict(model.net, rows, worker: Keyword.get(opts, :worker), rows: 4096)
    hr = assemble(base(img, opts), pred)
    if Keyword.get(opts, :project, true), do: project(hr, img), else: hr
  end

  defp x2(%Image{c: 3} = img, opts) do
    {y, cb, cr} = to_ycc(img)
    yu = x2(y, opts)
    up = fn ch -> ch |> Resample.resize(2 * img.w, 2 * img.h, "lanczos", worker: Keyword.get(opts, :worker)) |> then(&if Keyword.get(opts, :project, true), do: project(&1, ch, false), else: &1) end
    from_ycc(yu, up.(cb), up.(cr))
  end

  # the base (2w × 2h) plus each low-resolution pixel's four corrections
  defp assemble(%Image{w: w2, h: h2, px: b} = base, pred) do
    {w, h} = {div(w2, 2), div(h2, 2)}
    pt = pred |> Enum.map(&List.to_tuple/1) |> List.to_tuple()

    rows =
      for y <- 0..(h - 1), dy <- 0..1 do
        for x <- 0..(w - 1), dx <- 0..1, do: elem(b, (2 * y + dy) * w2 + 2 * x + dx) + elem(elem(pt, y * w + x), dy * 2 + dx)
      end

    %{base | px: rows |> List.flatten() |> List.to_tuple()}
  end

  @doc """
  The consistency projection: the closest image (in the least-squares
  sense, per block) to `hr` whose 2×2 block means are `lr`, kept inside
  [0, 1] (`clip`) by redistributing among the unsaturated pixels.
  """
  def project(%Image{w: w2, c: 1, px: hp} = hr, %Image{w: w, h: h, px: lp}, clip \\ true) do
    # row pairs of the output, block by block, then back to row-major order
    rows =
      for y <- 0..(h - 1) do
        fixed =
          for x <- 0..(w - 1) do
            idx = [(2 * y) * w2 + 2 * x, (2 * y) * w2 + 2 * x + 1, (2 * y + 1) * w2 + 2 * x, (2 * y + 1) * w2 + 2 * x + 1]
            fix(Enum.map(idx, &elem(hp, &1)), elem(lp, y * w + x), clip)
          end

        top = Enum.flat_map(fixed, fn [a, b, _, _] -> [a, b] end)
        bottom = Enum.flat_map(fixed, fn [_, _, c, d] -> [c, d] end)
        [top, bottom]
      end

    %{hr | px: rows |> List.flatten() |> List.to_tuple()}
  end

  # ------------------------------------------------------------- temporal --

  @doc """
  **Temporal upscaling that cannot contradict any frame.** `lr` is the
  current low-resolution frame (one channel), `prev` the previous output (or
  `nil` for the first frame), and `motion` how the picture moved since then,
  in output pixels: `{dx, dy}` for the whole frame, or a function
  `{x, y} -> {dx, dy}` per output pixel (what a renderer's motion vectors
  give).

  The history (`prev` moved by `motion`) is clamped, pixel by pixel, to the
  range of the current frame's single-frame upscaling over a 3×3
  neighbourhood, so history that does not fit what this frame shows is
  rejected (as temporal anti-aliasing does); it is blended with that
  upscaling (`alpha`, the current frame's weight, 0.25); and the result is
  projected so that `D(y) = lr` exactly. Whatever the history holds (an
  object that has since gone, wrong motion vectors), the output agrees with
  this frame's input. What the history can add is detail inside the 2×2
  blocks: new information when successive frames sample different sub-pixel
  positions (the picture moved by a fraction of an input pixel), and none
  when they sample the same ones.

  Options: `alpha`; `spatial`, the single-frame upscaling of `lr` (default
  `upscale(lr, opts)`); and those of `upscale/2`.
  """
  def temporal(%Image{c: 1} = lr, prev, motion, opts \\ []) do
    spatial = Keyword.get_lazy(opts, :spatial, fn -> upscale(lr, opts) end)

    case prev do
      nil -> spatial
      %Image{} -> prev |> warp(motion, spatial) |> clamp_to(spatial) |> mix(spatial, Keyword.get(opts, :alpha, 0.25)) |> project(lr)
    end
  end

  # the previous output where each pixel now is; outside the old frame, the current one
  defp warp(%Image{w: w, h: h, px: p}, motion, %Image{px: s} = cur) do
    at = fn x, y -> case motion do {dx, dy} -> {x - dx, y - dy}; f -> {dx, dy} = f.({x, y}); {x - dx, y - dy} end end

    px =
      for y <- 0..(h - 1), x <- 0..(w - 1) do
        {sx, sy} = at.(x, y)
        if sx in 0..(w - 1) and sy in 0..(h - 1), do: elem(p, sy * w + sx), else: elem(s, y * w + x)
      end

    %{cur | px: List.to_tuple(px)}
  end

  defp clamp_to(%Image{px: hp} = hist, %Image{w: w, h: h, px: s}) do
    px =
      for y <- 0..(h - 1), x <- 0..(w - 1) do
        nb = for j <- max(y - 1, 0)..min(y + 1, h - 1), i <- max(x - 1, 0)..min(x + 1, w - 1), do: elem(s, j * w + i)
        elem(hp, y * w + x) |> max(Enum.min(nb)) |> min(Enum.max(nb))
      end

    %{hist | px: List.to_tuple(px)}
  end

  defp mix(%Image{px: hp} = hist, %Image{px: s}, alpha) do
    %{hist | px: Enum.zip_with(Tuple.to_list(s), Tuple.to_list(hp), fn a, b -> alpha * a + (1 - alpha) * b end) |> List.to_tuple()}
  end

  defp fix(vals, target, clip, round \\ 0) do
    r = target - Enum.sum(vals) / 4
    free = if clip, do: Enum.map(vals, fn v -> if (r > 0 and v >= 1.0) or (r < 0 and v <= 0.0), do: false, else: true end), else: [true, true, true, true]
    n = Enum.count(free, & &1)

    cond do
      r == 0.0 or n == 0 or round > 6 -> vals
      true ->
        add = r * 4 / n
        vals = Enum.zip_with(vals, free, fn v, f -> if f, do: v + add, else: v end)
        vals = if clip, do: Enum.map(vals, &min(1.0, max(0.0, &1))), else: vals
        if clip, do: fix(vals, target, clip, round + 1), else: vals
    end
  end

  # BT.601 full range (the transform is linear, so it commutes with D)
  defp to_ycc(%Image{w: w, h: h, px: px}) do
    n = w * h
    trip = for i <- 0..(n - 1), do: {elem(px, 3 * i), elem(px, 3 * i + 1), elem(px, 3 * i + 2)}
    y = Enum.map(trip, fn {r, g, b} -> 0.299 * r + 0.587 * g + 0.114 * b end)
    cb = Enum.zip_with(trip, y, fn {_, _, b}, l -> (b - l) / 1.772 + 0.5 end)
    cr = Enum.zip_with(trip, y, fn {r, _, _}, l -> (r - l) / 1.402 + 0.5 end)
    mk = &%Image{w: w, h: h, c: 1, px: List.to_tuple(&1)}
    {mk.(y), mk.(cb), mk.(cr)}
  end

  defp from_ycc(%Image{w: w, h: h, px: y}, %Image{px: cb}, %Image{px: cr}) do
    vals =
      for i <- 0..(w * h - 1) do
        {l, b, r} = {elem(y, i), elem(cb, i) - 0.5, elem(cr, i) - 0.5}
        [l + 1.402 * r, l - 0.344136286201022 * b - 0.714136286201022 * r, l + 1.772 * b] |> Enum.map(&min(1.0, max(0.0, &1)))
      end

    %Image{w: w, h: h, c: 3, px: vals |> List.flatten() |> List.to_tuple()}
  end

  # -------------------------------------------------------------- training --

  @doc """
  Training rows from high-resolution greyscale images: every image is
  degraded by `D`; rows are sampled deterministically — half uniformly,
  half among the most textured neighbourhoods (flat regions teach nothing).
  Options: `per_image` (4000).
  """
  def dataset(images, opts \\ []) do
    per = Keyword.get(opts, :per_image, 4000)

    images
    |> Enum.with_index()
    |> Enum.flat_map(fn {hr, i} ->
      %Image{w: w, h: h, px: px} = lr = downsample(hr)
      b = base(lr, opts)
      # activity: squared differences to the 8 neighbours
      keyed =
        for y <- 0..(h - 1), x <- 0..(w - 1) do
          c = elem(px, y * w + x)
          a = for(dy <- -1..1, dx <- -1..1, do: (elem(px, refl(y + dy, h) * w + refl(x + dx, w)) - c) |> then(&(&1 * &1))) |> Enum.sum()
          {Vapor.Modal.Rng.key([:upscale, i, x, y]), a, {x, y}}
        end

      k = div(min(per, w * h), 2)
      uniform = keyed |> Enum.sort_by(&elem(&1, 0)) |> Enum.take(k) |> Enum.map(&elem(&1, 2))
      textured = keyed |> Enum.sort_by(&{-elem(&1, 1), elem(&1, 0)}) |> Enum.take(k) |> Enum.map(&elem(&1, 2))
      for pt <- uniform ++ textured, do: {feature(lr, pt), target(hr, b, pt)}
    end)
    |> Enum.unzip()
  end

  @doc "Train the network on `images` (high-resolution greyscale). Options as `Vapor.Learn.train/5` (`steps`, `batch`, `lr`, `worker`, `seed`)."
  def train(images, opts \\ []) do
    {xs, ys} = dataset(images, opts)
    net = Learn.new(@sizes, Keyword.get(opts, :seed, 1))
    Learn.train(net, xs, ys, Keyword.get(opts, :steps, 4000), Keyword.merge([batch: 256, lr: 2.0e-3, lr_end: 0.02, chunk: 250], opts))
  end

  # --------------------------------------------------------------- metrics --

  @doc "PSNR (dB) between two one-channel images of one size, in [0, 1]."
  def psnr(%Image{px: a}, %Image{px: b}) do
    n = tuple_size(a)
    mse = Enum.reduce(0..(n - 1), 0.0, fn i, s -> d = min(1.0, max(0.0, elem(a, i))) - elem(b, i); s + d * d end) / n
    if mse == 0.0, do: :infinity, else: 10 * :math.log10(1.0 / mse)
  end

  @doc "The largest |D(y) − x| over the image (0 for a consistent upscaling, up to rounding)."
  def inconsistency(%Image{} = y, %Image{} = x) do
    %Image{px: d} = downsample(y)
    Enum.reduce(0..(tuple_size(d) - 1), 0.0, fn i, m -> max(m, abs(elem(d, i) - elem(x.px, i))) end)
  end

end
