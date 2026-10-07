defmodule Vapor.Studio.Nodes.Image do
  @moduledoc """
  Image nodes: load, make, resize, crop, pad, flip, rotate, blur, sharpen,
  invert, greyscale, levels, blend, composite, masks.

  Resizing and filtering are matrix products run as vapor programs
  (`Vapor.Studio.Resample`), the same bits on every substrate. Pixelwise
  nodes run in the BEAM using only operations whose result IEEE-754 fixes
  (`+ − × ÷`, comparisons) and the correctly rounded `pow` of `Vapor.CR`, so
  they too are the same bits on every machine.
  """
  @behaviour Vapor.Studio.Node
  alias Vapor.Modal.Image
  alias Vapor.Rejection
  alias Vapor.Studio.Resample

  @methods Resample.methods()

  defp n(type, title, doc, inputs, outputs, params),
    do: {__MODULE__, %{type: type, version: 1, category: "image", title: title, doc: doc, inputs: inputs, outputs: outputs, params: params}}

  @impl true
  def nodes do
    [n("image.load", "Load image", "PNG, JPEG (baseline and progressive), GIF (first frame), PPM/PGM — from `data` (bytes) or `path` (inside the studio's directory).",
       [], [image: :image], [data: {:data, nil}, path: {:string, ""}]),
     n("image.scene", "Scene", "A deterministic picture to start from: a sky-to-ground gradient and seeded discs and blocks with soft edges (the same seed, the same pixels).",
       [], [image: :image], [width: {:int, 8, 4096, 384}, height: {:int, 8, 4096, 256}, seed: {:int, 0, 1_000_000, 1}, shapes: {:int, 0, 64, 9}]),
     n("image.solid", "Solid image", "A width × height image of one colour.", [], [image: :image],
       [width: {:int, 1, 8192, 512}, height: {:int, 1, 8192, 512}, red: {:float, 0.0, 1.0, 0.0}, green: {:float, 0.0, 1.0, 0.0}, blue: {:float, 0.0, 1.0, 0.0}]),
     n("image.resize", "Resize", "To width × height (0 keeps the aspect ratio from the other side); `crop: center` fills the frame without distortion.",
       [image: :image], [image: :image],
       [width: {:int, 0, 16384, 512}, height: {:int, 0, 16384, 0}, method: {:enum, @methods, "lanczos"}, antialias: {:bool, false}, crop: {:enum, ~w(disabled center), "disabled"}]),
     n("image.scale_by", "Scale by", "Multiply both sides by `factor`.", [image: :image], [image: :image],
       [factor: {:float, 0.01, 16.0, 2.0}, method: {:enum, @methods, "lanczos"}, antialias: {:bool, false}]),
     n("image.crop", "Crop", "The rectangle at (x, y) of width × height (clipped to the image).", [image: :image], [image: :image],
       [x: {:int, 0, 16384, 0}, y: {:int, 0, 16384, 0}, width: {:int, 1, 16384, 512}, height: {:int, 1, 16384, 512}]),
     n("image.pad", "Pad", "Add borders; the mask marks the new area (for outpainting).", [image: :image], [image: :image, mask: :mask],
       [left: {:int, 0, 4096, 0}, top: {:int, 0, 4096, 0}, right: {:int, 0, 4096, 0}, bottom: {:int, 0, 4096, 0}, mode: {:enum, ~w(constant edge reflect), "constant"},
        value: {:float, 0.0, 1.0, 0.0}]),
     n("image.flip", "Flip", "Mirror horizontally or vertically.", [image: :image], [image: :image], [axis: {:enum, ~w(horizontal vertical), "horizontal"}]),
     n("image.rotate", "Rotate", "By a multiple of 90° (exact).", [image: :image], [image: :image], [degrees: {:enum, ~w(90 180 270), "90"}]),
     n("image.blur", "Gaussian blur", "Separable Gaussian, radius and sigma in pixels, reflected edges.", [image: :image], [image: :image],
       [radius: {:int, 1, 64, 2}, sigma: {:float, 0.1, 32.0, 1.0}]),
     n("image.sharpen", "Sharpen", "Unsharp masking: (1 + amount)·x − amount·blur(x).", [image: :image], [image: :image],
       [radius: {:int, 1, 64, 2}, sigma: {:float, 0.1, 32.0, 1.0}, amount: {:float, 0.0, 10.0, 0.6}]),
     n("image.invert", "Invert", "1 − x.", [image: :image], [image: :image], []),
     n("image.grayscale", "Greyscale", "BT.709 luma (0.2126 R + 0.7152 G + 0.0722 B), one channel.", [image: :image], [image: :image], []),
     n("image.levels", "Levels", "Map [black, white] to [0, 1], then the gamma.", [image: :image], [image: :image],
       [black: {:float, 0.0, 1.0, 0.0}, white: {:float, 0.0, 1.0, 1.0}, gamma: {:float, 0.05, 10.0, 1.0}]),
     n("image.blend", "Blend", "Mix two images of one size: normal, multiply, screen, difference, add.", [a: :image, b: :image], [image: :image],
       [factor: {:float, 0.0, 1.0, 0.5}, mode: {:enum, ~w(normal multiply screen difference add), "normal"}]),
     n("image.composite", "Composite", "Paste `source` onto `destination` at (x, y), weighted by `mask` (all of it without one).",
       [destination: :image, source: :image, mask: {:mask, :optional}], [image: :image], [x: {:int, -16384, 16384, 0}, y: {:int, -16384, 16384, 0}]),
     n("mask.threshold", "Mask from image", "1 where the channel (or the luma) is at least the threshold.", [image: :image], [mask: :mask],
       [channel: {:enum, ~w(luma red green blue), "luma"}, threshold: {:float, 0.0, 1.0, 0.5}]),
     n("mask.rect", "Rectangle mask", "1 inside the rectangle, 0 outside.", [], [mask: :mask],
       [width: {:int, 1, 8192, 512}, height: {:int, 1, 8192, 512}, x: {:int, 0, 8192, 0}, y: {:int, 0, 8192, 0}, w: {:int, 1, 8192, 256}, h: {:int, 1, 8192, 256}]),
     n("mask.invert", "Invert mask", "1 − m.", [mask: :mask], [mask: :mask], []),
     n("mask.feather", "Feather mask", "Soften the edges (a Gaussian blur of the mask).", [mask: :mask], [mask: :mask], [radius: {:int, 1, 64, 4}, sigma: {:float, 0.1, 32.0, 2.0}])]
  end

  # ------------------------------------------------------------------ run --

  @impl true
  def run("image.load", _ins, %{data: data, path: path}, ctx) do
    with {:ok, bytes} <- source(data, path, ctx), do: (with {:ok, img} <- decode(bytes), do: {:ok, %{image: img}})
  end

  def run("image.scene", _, p, _) do
    u = Vapor.Modal.Rng.uniform(p.seed, p.shapes * 8 + 6)
    {sky, rest} = Enum.split(u, 6)
    [a, b, c, d, e, f] = sky
    top = [0.35 + 0.4 * a, 0.5 + 0.4 * b, 0.75 + 0.25 * c]
    bottom = [0.75 + 0.25 * d, 0.55 + 0.35 * e, 0.3 + 0.3 * f]
    {w, h} = {p.width, p.height}

    shapes =
      for [k, x, y, r, s, cr, cg, cb] <- Enum.chunk_every(rest, 8, 8, :discard) do
        col = [cr, cg, cb]
        if k < 0.6,
          do: {:disc, x * w, y * h, (0.04 + 0.16 * r) * min(w, h), col},
          else: (fn x0, y0 -> {:rect, x0, y0, x0 + (0.1 + 0.3 * r) * w, y0 + (0.1 + 0.3 * s) * h, col} end).(x * w * 0.8, y * h * 0.8)
      end

    {:ok, %{image: Image.scene(w, h, bg: {top, bottom}, shapes: shapes)}}
  end

  def run("image.solid", _, p, _), do: {:ok, %{image: solid(p.width, p.height, [p.red, p.green, p.blue])}}

  def run("image.resize", %{image: img}, p, ctx) do
    {w2, h2} = target(img, p.width, p.height)
    img = if p.crop == "center", do: center_crop(img, w2 / h2), else: img
    {:ok, %{image: Resample.resize(img, w2, h2, p.method, worker: ctx.worker, antialias: p.antialias)}}
  end

  def run("image.scale_by", %{image: img}, p, ctx) do
    {w2, h2} = {max(1, round(img.w * p.factor)), max(1, round(img.h * p.factor))}
    {:ok, %{image: Resample.resize(img, w2, h2, p.method, worker: ctx.worker, antialias: p.antialias)}}
  end

  def run("image.crop", %{image: img}, p, _) do
    x = min(p.x, img.w - 1)
    y = min(p.y, img.h - 1)
    {:ok, %{image: crop(img, x, y, min(p.width, img.w - x), min(p.height, img.h - y))}}
  end

  def run("image.pad", %{image: img}, p, _), do: {:ok, pad(img, p)}

  def run("image.flip", %{image: %Image{w: w, h: h} = img}, %{axis: a}, _) do
    {:ok, %{image: remap(img, w, h, fn x, y -> if a == "horizontal", do: {w - 1 - x, y}, else: {x, h - 1 - y} end)}}
  end

  def run("image.rotate", %{image: %Image{w: w, h: h} = img}, %{degrees: d}, _) do
    out =
      case d do
        "90" -> remap(img, h, w, fn x, y -> {y, h - 1 - x} end)
        "180" -> remap(img, w, h, fn x, y -> {w - 1 - x, h - 1 - y} end)
        "270" -> remap(img, h, w, fn x, y -> {w - 1 - y, x} end)
      end

    {:ok, %{image: out}}
  end

  def run("image.blur", %{image: img}, p, ctx), do: {:ok, %{image: Resample.blur(img, p.radius, p.sigma, worker: ctx.worker)}}
  def run("image.sharpen", %{image: img}, p, ctx), do: {:ok, %{image: clamp(Resample.sharpen(img, p.radius, p.sigma, p.amount, worker: ctx.worker))}}
  def run("image.invert", %{image: img}, _, _), do: {:ok, %{image: map(img, &(1.0 - &1))}}
  def run("image.grayscale", %{image: img}, _, _), do: {:ok, %{image: luma(img)}}

  def run("image.levels", %{image: img}, p, _) do
    span = max(p.white - p.black, 1.0e-6)
    inv = 1.0 / p.gamma
    f = fn v -> t = min(1.0, max(0.0, (v - p.black) / span)); if p.gamma == 1.0 or t == 0.0, do: t, else: Vapor.CR.pow_f64(t, inv) end
    {:ok, %{image: map(img, f)}}
  end

  def run("image.blend", %{a: a, b: b}, p, _) do
    with :ok <- same_size(a, b, "blend") do
      {a, b} = match_channels(a, b)
      t = p.factor

      f =
        case p.mode do
          "normal" -> fn x, y -> x + (y - x) * t end
          "multiply" -> fn x, y -> x + (x * y - x) * t end
          "screen" -> fn x, y -> x + (1.0 - (1.0 - x) * (1.0 - y) - x) * t end
          "difference" -> fn x, y -> x + (abs(x - y) - x) * t end
          "add" -> fn x, y -> min(1.0, x + y * t) end
        end

      {:ok, %{image: zip(a, b, f)}}
    end
  end

  def run("image.composite", %{destination: d, source: s} = ins, %{x: x0, y: y0}, _) do
    {d, s} = match_channels(d, s)
    m = Map.get(ins, :mask)
    c = d.c
    dt = d.px
    st = s.px

    vals =
      for y <- 0..(d.h - 1), x <- 0..(d.w - 1), ch <- 0..(c - 1) do
        dv = elem(dt, (y * d.w + x) * c + ch)
        {sx, sy} = {x - x0, y - y0}

        if sx < 0 or sy < 0 or sx >= s.w or sy >= s.h do
          dv
        else
          sv = elem(st, (sy * s.w + sx) * c + ch)
          a = if m, do: (if sx < m.w and sy < m.h, do: elem(m.px, sy * m.w + sx), else: 0.0), else: 1.0
          dv + (sv - dv) * a
        end
      end

    {:ok, %{image: %Image{w: d.w, h: d.h, c: c, px: List.to_tuple(vals)}}}
  end

  def run("mask.threshold", %{image: img}, %{channel: ch, threshold: t}, _) do
    src = if ch == "luma" or img.c == 1, do: luma(img), else: channel(img, Enum.find_index(~w(red green blue), &(&1 == ch)))
    {:ok, %{mask: map(src, &(if &1 >= t, do: 1.0, else: 0.0))}}
  end

  def run("mask.rect", _, p, _) do
    vals = for y <- 0..(p.height - 1), x <- 0..(p.width - 1), do: if(x >= p.x and x < p.x + p.w and y >= p.y and y < p.y + p.h, do: 1.0, else: 0.0)
    {:ok, %{mask: %Image{w: p.width, h: p.height, c: 1, px: List.to_tuple(vals)}}}
  end

  def run("mask.invert", %{mask: m}, _, _), do: {:ok, %{mask: map(m, &(1.0 - &1))}}
  def run("mask.feather", %{mask: m}, p, ctx), do: {:ok, %{mask: clamp(Resample.blur(m, p.radius, p.sigma, worker: ctx.worker))}}

  # -------------------------------------------------------------- helpers --

  @doc false
  def source(data, path, ctx) do
    cond do
      is_binary(data) and data != "" -> {:ok, data}
      path != "" ->
        dir = Path.expand(ctx.dir)
        full = Path.expand(path, dir)
        if String.starts_with?(full, dir <> "/") or full == dir,
          do: File.read(full) |> then(fn {:ok, b} -> {:ok, b}; {:error, why} -> {:error, Rejection.new({:path, path}, "a readable file", inspect(why))} end),
          else: {:error, Rejection.new({:path, path}, "a path inside the studio's directory", "copy the file there")}

      true -> {:error, Rejection.new(:source, "data (bytes) or a path", "give one of them")}
    end
  end

  @doc "Decode PNG, JPEG, GIF (first frame) or PPM/PGM bytes."
  def decode(<<0x89, "PNG", _::binary>> = b), do: picture(:png, b)
  def decode(<<0xFF, 0xD8, _::binary>> = b), do: picture(:jpeg, b)
  def decode(<<"P", d, _::binary>> = b) when d in [?5, ?6], do: Image.parse(b)
  def decode(<<"GIF8", _::binary>> = b), do: with({:ok, g} <- Vapor.Media.GIF.decode(b), do: {:ok, hd(g.frames)})
  def decode(_), do: {:error, Rejection.new(:image, "PNG, JPEG, GIF or PPM/PGM bytes", "convert the picture")}

  defp picture(fmt, b) do
    case Vapor.Docs.Pictures.read(fmt, b) do
      {:ok, %{image: %Image{} = img}} -> {:ok, img}
      {:ok, _} -> {:error, Rejection.new(:image, "a picture small enough to decode (≤ 4 MP)", "resize it first")}
      {:error, _} = e -> e
    end
  end

  defp target(%Image{w: w, h: h}, 0, 0), do: {w, h}
  defp target(%Image{w: w, h: h}, w2, 0), do: {w2, max(1, round(h * w2 / w))}
  defp target(%Image{w: w, h: h}, 0, h2), do: {max(1, round(w * h2 / h)), h2}
  defp target(_, w2, h2), do: {w2, h2}

  defp center_crop(%Image{w: w, h: h} = img, aspect) do
    cond do
      w / h > aspect -> (fn nw -> crop(img, div(w - nw, 2), 0, nw, h) end).(max(1, round(h * aspect)))
      w / h < aspect -> (fn nh -> crop(img, 0, div(h - nh, 2), w, nh) end).(max(1, round(w / aspect)))
      true -> img
    end
  end

  @doc false
  def crop(%Image{w: w, c: c, px: px}, x0, y0, cw, ch) do
    vals = for y <- y0..(y0 + ch - 1), x <- x0..(x0 + cw - 1), k <- 0..(c - 1), do: elem(px, (y * w + x) * c + k)
    %Image{w: cw, h: ch, c: c, px: List.to_tuple(vals)}
  end

  defp pad(%Image{w: w, h: h, c: c, px: px}, p) do
    {nw, nh} = {w + p.left + p.right, h + p.top + p.bottom}
    idx = fn i, n ->
      cond do
        i >= 0 and i < n -> i
        p.mode == "edge" -> min(max(i, 0), n - 1)
        p.mode == "reflect" -> reflect(i, n)
        true -> nil
      end
    end

    {vals, mask} =
      for y <- 0..(nh - 1), x <- 0..(nw - 1), reduce: {[], []} do
        {vs, ms} ->
          {sx, sy} = {x - p.left, y - p.top}
          inside = sx >= 0 and sx < w and sy >= 0 and sy < h
          {ix, iy} = {idx.(sx, w), idx.(sy, h)}
          pix = for k <- 0..(c - 1), do: if(ix && iy, do: elem(px, (iy * w + ix) * c + k), else: p.value)
          {Enum.reverse(pix) ++ vs, [if(inside, do: 0.0, else: 1.0) | ms]}
      end

    %{image: %Image{w: nw, h: nh, c: c, px: vals |> Enum.reverse() |> List.to_tuple()},
      mask: %Image{w: nw, h: nh, c: 1, px: mask |> Enum.reverse() |> List.to_tuple()}}
  end

  defp reflect(_i, 1), do: 0
  defp reflect(i, n) when i < 0, do: reflect(-i, n)
  defp reflect(i, n) when i >= n, do: reflect(2 * (n - 1) - i, n)
  defp reflect(i, _), do: i

  defp remap(%Image{w: w, c: c, px: px}, nw, nh, f) do
    vals = for y <- 0..(nh - 1), x <- 0..(nw - 1), {sx, sy} = f.(x, y), k <- 0..(c - 1), do: elem(px, (sy * w + sx) * c + k)
    %Image{w: nw, h: nh, c: c, px: List.to_tuple(vals)}
  end

  @doc false
  def solid(w, h, rgb), do: %Image{w: w, h: h, c: 3, px: List.to_tuple(List.flatten(List.duplicate(Enum.map(rgb, &(&1 * 1.0)), w * h)))}

  defp map(%Image{px: px} = img, f), do: %{img | px: px |> Tuple.to_list() |> Enum.map(f) |> List.to_tuple()}
  defp clamp(img), do: map(img, &min(1.0, max(0.0, &1)))
  defp zip(%Image{px: a} = img, %Image{px: b}, f), do: %{img | px: Enum.zip_with(Tuple.to_list(a), Tuple.to_list(b), f) |> List.to_tuple()}

  defp luma(%Image{c: 1} = img), do: img
  defp luma(%Image{w: w, h: h, px: px}) do
    vals = for i <- 0..(w * h - 1), do: 0.2126 * elem(px, 3 * i) + 0.7152 * elem(px, 3 * i + 1) + 0.0722 * elem(px, 3 * i + 2)
    %Image{w: w, h: h, c: 1, px: List.to_tuple(vals)}
  end

  defp channel(%Image{w: w, h: h, px: px}, k), do: %Image{w: w, h: h, c: 1, px: List.to_tuple(for(i <- 0..(w * h - 1), do: elem(px, 3 * i + k)))}

  defp rgb(%Image{c: 3} = img), do: img
  defp rgb(%Image{w: w, h: h, px: px}), do: %Image{w: w, h: h, c: 3, px: List.to_tuple(Enum.flat_map(Tuple.to_list(px), &[&1, &1, &1]))}

  defp match_channels(%Image{c: c} = a, %Image{c: c} = b), do: {a, b}
  defp match_channels(a, b), do: {rgb(a), rgb(b)}

  defp same_size(%Image{w: w, h: h}, %Image{w: w, h: h}, _), do: :ok
  defp same_size(a, b, what), do: {:error, Rejection.new({what, :size}, "two images of one size", "#{a.w}×#{a.h} and #{b.w}×#{b.h}: resize one first")}
end
