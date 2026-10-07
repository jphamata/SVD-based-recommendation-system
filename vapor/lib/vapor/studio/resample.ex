defmodule Vapor.Studio.Resample do
  @moduledoc """
  **Resizing and filtering as certified matrix products.** Every separable
  image operator — resizing with any kernel, Gaussian blur, unsharp
  masking — is `Y = R_y · X · R_xᵀ` per channel: two products with constant
  matrices. So it runs as a vapor program (`linear`, `transpose`) on the
  native worker (or the GPU), bit-identical on every substrate, and gets
  the same certificate as any other program.

  The matrices follow the references they are compared with
  (`test/vapor/studio_media_test.exs`):

    * `nearest` — the source pixel under the destination pixel's centre
      (`torch` *nearest-exact*);
    * `bilinear`, `bicubic` — `torch.nn.functional.interpolate`,
      `align_corners=False`, bicubic with `a = −0.75`, **no antialiasing**
      (so they alias when shrinking, as in torch; `antialias: true` widens
      the kernel by the scale, as Pillow does);
    * `area` — `adaptive_avg_pool` (each output the mean of the source
      pixels it covers);
    * `lanczos` — Pillow's Lanczos-3, antialiased (support `3·max(1, scale)`),
      with weights from the correctly rounded `sin` (`Vapor.CR`), so the
      constants are the same bits on every machine.

  Weights are computed in binary64, normalized, rounded once to binary32.
  `separable/4` runs the program on a worker when one is given; otherwise a
  sparse evaluator in the BEAM that reproduces the canonical 16-lane
  summation order exactly (only non-zero weights contribute, and adding a
  signed zero to a lane changes nothing) — the same bits, without dense
  matrices.
  """
  alias Vapor.{F32, Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Modal.Image
  alias Vapor.Runtime.{Native, Oracle, Substrates}

  @methods ~w(nearest bilinear bicubic area lanczos)
  def methods, do: @methods

  # ------------------------------------------------------------ the matrices --

  @doc """
  Sparse resampling weights from `n_in` to `n_out` samples: one list of
  `{source, weight_f32_bits}` per output, in increasing source order.
  """
  def weights(method, n_in, n_out, opts \\ []) do
    # a window of the source: `offset` (first source coordinate, fractional)
    # and `span` (its length); the whole source by default
    off = Keyword.get(opts, :offset, 0.0) * 1.0
    scale = Keyword.get(opts, :span, n_in) / n_out
    aa = Keyword.get(opts, :antialias, false)

    for i <- 0..(n_out - 1) do
      raw =
        case method do
          "nearest" -> [{min(n_in - 1, max(0, trunc(Float.floor(off + (i + 0.5) * scale)))), 1.0}]
          "area" when off == 0.0 -> area(i, n_in, n_out)
          "area" -> filtered(i, n_in, scale, 0.5, &box/1, off)
          "bilinear" -> if aa and scale > 1, do: filtered(i, n_in, scale, 1.0, &tri/1, off), else: bilinear(i, n_in, scale, off)
          "bicubic" -> if aa and scale > 1, do: filtered(i, n_in, scale, 2.0, &cubic_aa/1, off), else: bicubic(i, n_in, scale, off)
          "lanczos" -> filtered(i, n_in, scale, 3.0, &lanczos3/1, off)
          "lanczos:" <> a -> (fn a -> filtered(i, n_in, scale, a, &lanczos(&1, a), off) end).(String.to_integer(a) * 1.0)
          "gauss:" <> _ = g -> gauss(i, n_in, g)
        end

      raw
      |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
      |> Enum.map(fn {j, ws} -> {j, Enum.reduce(ws, 0.0, &+/2)} end)
      |> Enum.sort()
      |> Enum.reject(fn {_, w} -> w == 0.0 end)
      |> Enum.map(fn {j, w} -> {j, F32.from_float(w)} end)
    end
  end

  defp area(i, n_in, n_out) do
    s = div(i * n_in, n_out)
    e = div((i + 1) * n_in + n_out - 1, n_out)
    for j <- s..(e - 1), do: {j, 1.0 / (e - s)}
  end

  defp bilinear(i, n_in, scale, off) do
    x = max(off + (i + 0.5) * scale - 0.5, 0.0)
    x0 = trunc(Float.floor(x))
    t = x - x0
    x1 = min(x0 + 1, n_in - 1)
    x0 = min(x0, n_in - 1)
    [{x0, 1.0 - t}, {x1, t}]
  end

  # Keys' cubic with a = −0.75 (torch), taps x0−1 … x0+2 clamped to the edge
  defp bicubic(i, n_in, scale, off) do
    x = off + (i + 0.5) * scale - 0.5
    x0 = trunc(Float.floor(x))
    t = x - x0
    a = -0.75
    w = [cubic2(t + 1.0, a), cubic1(t, a), cubic1(1.0 - t, a), cubic2(2.0 - t, a)]
    for {wk, k} <- Enum.with_index(w), do: {min(max(x0 - 1 + k, 0), n_in - 1), wk}
  end

  defp cubic1(x, a), do: ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
  defp cubic2(x, a), do: ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a

  # Pillow's scheme: support widened by the scale when shrinking, weights normalized
  defp filtered(i, n_in, scale, support, k, off) do
    fs = max(scale, 1.0)
    sup = support * fs
    center = off + (i + 0.5) * scale
    lo = max(trunc(Float.floor(center - sup + 0.5)), 0)
    hi = min(trunc(Float.floor(center + sup + 0.5)), n_in)
    ws = for j <- lo..(hi - 1)//1, do: {j, k.((j - center + 0.5) / fs)}
    total = ws |> Enum.map(&elem(&1, 1)) |> Enum.reduce(0.0, &+/2)
    if total == 0.0, do: [{min(max(trunc(center), 0), n_in - 1), 1.0}], else: Enum.map(ws, fn {j, w} -> {j, w / total} end)
  end

  defp box(x), do: if(abs(x) <= 0.5, do: 1.0, else: 0.0)

  defp tri(x), do: max(0.0, 1.0 - abs(x))
  defp cubic_aa(x), do: (fn ax -> if ax < 1.0, do: cubic1(ax, -0.5), else: if(ax < 2.0, do: cubic2(ax, -0.5), else: 0.0) end).(abs(x))

  defp lanczos3(x), do: lanczos(x, 3.0)

  defp lanczos(x, a) do
    cond do
      x == 0.0 -> 1.0
      abs(x) >= a -> 0.0
      true ->
        px = :math.pi() * x
        Vapor.CR.sin_f64(px) * Vapor.CR.sin_f64(px / a) / (px * px / a)
    end
  end

  # "gauss:<radius>:<sigma>" — a normalized Gaussian, reflected at the edges
  defp gauss(i, n, "gauss:" <> rest) do
    [r, s] = String.split(rest, ":")
    {r, s} = {String.to_integer(r), String.to_float(s)}
    taps = for d <- -r..r, do: {d, Vapor.CR.exp_f64(-(d * d) / (2.0 * s * s))}
    total = taps |> Enum.map(&elem(&1, 1)) |> Enum.reduce(0.0, &+/2)
    for {d, w} <- taps, do: {reflect(i + d, n), w / total}
  end

  defp reflect(j, n) when n == 1, do: 0 * j
  defp reflect(j, n) when j < 0, do: reflect(-j, n)
  defp reflect(j, n) when j >= n, do: reflect(2 * (n - 1) - j, n)
  defp reflect(j, _), do: j

  # ---------------------------------------------------------------- apply --

  @doc """
  `Y = R_y · X · R_xᵀ` per channel, where `rx` and `ry` are weight lists
  (`weights/4`), optionally followed by `Y ← (1 + a)·X − a·Y` (unsharp
  masking, `unsharp: a`, only when the size is unchanged). Options:
  `worker` (run as a program there), `unsharp`.
  """
  def separable(%Image{c: c} = img, rx, ry, opts \\ []) do
    {w2, h2} = {length(rx), length(ry)}
    a = Keyword.get(opts, :unsharp)

    planes =
      case Keyword.get(opts, :worker) do
        nil -> beam(img, rx, ry, a)
        wk -> program(wk, img, rx, ry, a)
      end

    # back to an image, channels last
    tuples = Enum.map(planes, &List.to_tuple/1)
    vals = for p <- 0..(w2 * h2 - 1), ch <- 0..(c - 1), do: F32.to_float(elem(Enum.at(tuples, ch), p))
    %Image{w: w2, h: h2, c: c, px: List.to_tuple(vals)}
  end

  # f32 bit planes of each channel, row-major
  defp planes(%Image{w: w, h: h, c: c, px: px}) do
    for ch <- 0..(c - 1), do: for(p <- 0..(w * h - 1), do: F32.from_float(elem(px, p * c + ch)))
  end

  # the canonical dot over the non-zero taps: lane j mod 16 accumulates in
  # increasing j, then the 16-lane tree
  defp cdot(taps, get) do
    lanes =
      Enum.reduce(taps, %{}, fn {j, wt}, acc ->
        l = rem(j, 16)
        Map.put(acc, l, F32.add(Map.get(acc, l, 0), F32.mul(wt, get.(j))))
      end)

    Oracle.reduce16(for l <- 0..15, do: Map.get(lanes, l, 0))
  end

  defp beam(%Image{w: w, h: h} = img, rx, ry, a) do
    w2 = length(rx)

    for plane <- planes(img) do
      pt = List.to_tuple(plane)
      # horizontal pass: rows of X against rows of R_x
      horiz = for y <- 0..(h - 1), {taps} <- Enum.map(rx, &{&1}), do: cdot(taps, fn j -> elem(pt, y * w + j) end)
      ht = List.to_tuple(horiz)
      # vertical pass on the transpose
      out = for {taps} <- Enum.map(ry, &{&1}), x <- 0..(w2 - 1), do: cdot(taps, fn j -> elem(ht, j * w2 + x) end)
      if a, do: unsharp_beam(plane, out, a), else: out
    end
  end

  defp unsharp_beam(xs, ys, a) do
    ka = F32.from_float(1.0 + a)
    aa = F32.from_float(a)
    Enum.zip_with(xs, ys, fn x, y -> F32.sub(F32.mul(x, ka), F32.mul(y, aa)) end)
  end

  # the same operator as a vapor program, compiled once per shape: the two
  # matrices are inputs, so any kernel, scale or window of the same shape
  # reuses the compiled program
  defp program(wk, %Image{w: w, h: h, c: c} = img, rx, ry, a) do
    {w2, h2} = {length(rx), length(ry)}
    {wp, hp} = {Vapor.Spatial.pad16(w), Vapor.Spatial.pad16(h)}
    key = {:studio_resample, w, h, c, w2, h2, a}

    comp =
      cached(key, fn ->
        mx = T.input(:mx, :f32, [w2, wp])
        my = T.input(:my, :f32, [h2, hp])
        outs =
          for ch <- 0..(c - 1) do
            x = T.input(:"x#{ch}", :f32, [hp, wp])
            y = T.transpose(T.linear(T.transpose(T.linear(x, mx)), my))
            y = if a, do: T.sub(T.mul(crop_term(x, h, w), T.splat(1.0 + a)), T.mul(y, T.splat(a))), else: y
            {:"y#{ch}", y}
          end

        {:ok, comp} = Vapor.Compile.Lower.lower(Program.new(outs))
        comp
      end)

    env =
      img
      |> planes()
      |> Enum.with_index()
      |> Map.new(fn {plane, ch} ->
        rows = plane |> Enum.chunk_every(w) |> Enum.map(fn r -> for(v <- r, into: <<>>, do: <<v::32-little>>) <> :binary.copy(<<0::32>>, wp - w) end)
        data = IO.iodata_to_binary([rows, :binary.copy(<<0::32>>, (hp - h) * wp)])
        {:"x#{ch}", Tensor.new(:f32, [hp, wp], data)}
      end)
      |> Map.merge(%{mx: dense(rx, wp), my: dense(ry, hp)})

    {:ok, r} = Native.run(wk, comp, env, isa: Substrates.host_isa(), mode: :native)
    for ch <- 0..(c - 1), do: for(<<v::32-little <- r.outputs[:"y#{ch}"].data>>, do: v)
  end

  # unsharp masking keeps the size: the unpadded X, as a [h, w] term
  defp crop_term(x, h, w) do
    {:ok, {:f32, [hp, wp]}} = T.infer(x)
    if hp == h and wp == w, do: x, else: T.transpose(T.linear(T.transpose(T.linear(x, T.const(eye(w, wp)))), T.const(eye(h, hp))))
  end

  defp eye(n, np), do: dense(for(i <- 0..(n - 1), do: [{i, F32.from_float(1.0)}]), np)

  defp dense(rows, k) do
    data =
      for taps <- rows, into: <<>> do
        m = Map.new(taps)
        for j <- 0..(k - 1), into: <<>>, do: <<Map.get(m, j, 0)::32-little>>
      end

    Tensor.new(:f32, [length(rows), k], data)
  end

  defp cached(key, f) do
    k = {__MODULE__, key}
    case :persistent_term.get(k, nil) do
      nil ->
        v = f.()
        keys = :persistent_term.get({__MODULE__, :keys}, [])
        {keep, drop} = Enum.split([k | keys], 32)
        Enum.each(drop, &:persistent_term.erase/1)
        :persistent_term.put({__MODULE__, :keys}, keep)
        :persistent_term.put(k, v)
        v

      v -> v
    end
  end

  # ------------------------------------------------------------- operators --

  @doc "Resize to `w2 × h2` with `method` (see the module doc). Options: `worker`, `antialias`."
  def resize(%Image{w: w, h: h} = img, w2, h2, method, opts \\ []) when method in @methods do
    separable(img, weights(method, w, w2, opts), weights(method, h, h2, opts), opts)
  end

  @doc "Gaussian blur: radius `r` pixels, `sigma` pixels, reflected edges. Options: `worker`."
  def blur(%Image{w: w, h: h} = img, r, sigma, opts \\ []) do
    g = "gauss:#{r}:#{:erlang.float_to_binary(sigma * 1.0, [:short])}"
    separable(img, weights(g, w, w), weights(g, h, h), opts)
  end

  @doc "Unsharp masking: `(1 + amount)·X − amount·blur(X)`. Options: `worker`."
  def sharpen(%Image{w: w, h: h} = img, r, sigma, amount, opts \\ []) do
    g = "gauss:#{r}:#{:erlang.float_to_binary(sigma * 1.0, [:short])}"
    separable(img, weights(g, w, w), weights(g, h, h), Keyword.put(opts, :unsharp, amount * 1.0))
  end
end
