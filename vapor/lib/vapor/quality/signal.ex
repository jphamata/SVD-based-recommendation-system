defmodule Vapor.Quality.Signal do
  @moduledoc """
  Is this image or sound a signal, or noise? And how close is it to a
  reference? Measurements in binary64 on the BEAM (they judge outputs;
  they are not part of any certified program).

  Images (`Vapor.Modal.Image`), on luminance:

    * `neighbour_corr` — mean lag-1 Pearson correlation, horizontal and
      vertical. Natural images ≈ 0.8–0.99; white noise and shuffled pixels
      ≈ 0. **The main image noise score.**
    * `spectral_slope` — slope of the radially averaged log power spectrum
      against log frequency (natural ≈ −2, white noise ≈ 0);
    * `psnr`, `ssim` (8×8 windows) against a reference.

  Audio (`Vapor.Modal.Audio`):

    * `flatness` — Wiener entropy of the power spectrum, mean over frames:
      geometric mean / arithmetic mean, 1 for white noise, → 0 for tones.
      **The main audio noise score** (lower = more structure);
    * `dominant_hz` — the strongest frequency; `snr` against a reference.
  """
  alias Vapor.Modal.{Audio, Image}

  # ------------------------------------------------------------------ images --

  @doc "Luminance plane as a tuple of row tuples."
  def luma(%Image{w: w, h: h, c: c} = img) do
    vals = Image.values(img) |> Enum.chunk_every(c) |> Enum.map(fn
      [r, g, b] -> 0.299 * r + 0.587 * g + 0.114 * b
      [y] -> y
    end)

    vals |> Enum.chunk_every(w) |> Enum.map(&List.to_tuple/1) |> List.to_tuple() |> then(&{&1, w, h})
  end

  @doc "Mean lag-1 correlation (horizontal and vertical) of the luminance."
  def neighbour_corr(%Image{} = img) do
    {l, w, h} = luma(img)
    hp = for y <- 0..(h - 1), x <- 0..(w - 2), do: {px(l, x, y), px(l, x + 1, y)}
    vp = for y <- 0..(h - 2), x <- 0..(w - 1), do: {px(l, x, y), px(l, x, y + 1)}
    (pearson(hp) + pearson(vp)) / 2
  end

  defp px(l, x, y), do: l |> elem(y) |> elem(x)

  defp pearson(pairs) do
    n = length(pairs)
    {sa, sb} = Enum.reduce(pairs, {0.0, 0.0}, fn {a, b}, {x, y} -> {x + a, y + b} end)
    {ma, mb} = {sa / n, sb / n}
    {cov, va, vb} = Enum.reduce(pairs, {0.0, 0.0, 0.0}, fn {a, b}, {c, x, y} -> {c + (a - ma) * (b - mb), x + (a - ma) ** 2, y + (b - mb) ** 2} end)
    if va == 0 or vb == 0, do: 0.0, else: cov / :math.sqrt(va * vb)
  end

  @doc "PSNR in dB (peak 1.0); `:infinity` for identical images."
  def psnr(%Image{} = a, %Image{} = b) do
    mse = Enum.zip_reduce(Image.values(a), Image.values(b), 0.0, fn x, y, s -> s + (x - y) ** 2 end) / (a.w * a.h * a.c)
    if mse == 0, do: :infinity, else: 10 * :math.log10(1.0 / mse)
  end

  @doc "Mean SSIM over non-overlapping 8×8 luminance windows (C₁ = 0.01², C₂ = 0.03²)."
  def ssim(%Image{} = a, %Image{} = b, win \\ 8) do
    {la, w, h} = luma(a)
    {lb, _, _} = luma(b)
    {c1, c2} = {0.0001, 0.0009}

    scores =
      for wy <- 0..(div(h, win) - 1), wx <- 0..(div(w, win) - 1) do
        ps = for y <- 0..(win - 1), x <- 0..(win - 1), do: {px(la, wx * win + x, wy * win + y), px(lb, wx * win + x, wy * win + y)}
        n = length(ps)
        {ma, mb} = Enum.reduce(ps, {0.0, 0.0}, fn {p, q}, {x, y} -> {x + p / n, y + q / n} end)
        {va, vb, cov} = Enum.reduce(ps, {0.0, 0.0, 0.0}, fn {p, q}, {x, y, z} -> {x + (p - ma) ** 2 / n, y + (q - mb) ** 2 / n, z + (p - ma) * (q - mb) / n} end)
        (2 * ma * mb + c1) * (2 * cov + c2) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
      end

    Enum.sum(scores) / max(length(scores), 1)
  end

  @doc """
  Slope of log radial power against log frequency over the mid band
  (frequencies 2 … n/4). Width and height must be powers of two.
  """
  def spectral_slope(%Image{} = img) do
    {l, w, h} = luma(img)
    mean = (for y <- 0..(h - 1), x <- 0..(w - 1), do: px(l, x, y)) |> then(&(Enum.sum(&1) / length(&1)))
    rows = for y <- 0..(h - 1), do: fft(for x <- 0..(w - 1), do: {px(l, x, y) - mean, 0.0})
    cols = rows |> Enum.zip_with(& &1) |> Enum.map(&fft/1)

    bins =
      for {col, x} <- Enum.with_index(cols), {{re, im}, y} <- Enum.with_index(col), reduce: %{} do
        acc ->
          fx = if x <= div(w, 2), do: x, else: x - w
          fy = if y <= div(h, 2), do: y, else: y - h
          r = round(:math.sqrt(fx * fx + fy * fy))
          Map.update(acc, r, [re * re + im * im], &[re * re + im * im | &1])
      end

    pts =
      for r <- 2..max(2, div(min(w, h), 4)), ps = bins[r], ps != nil, m = Enum.sum(ps) / length(ps), m > 0,
          do: {:math.log(r), :math.log(m)}

    slope(pts)
  end

  defp slope(pts) when length(pts) < 2, do: 0.0

  defp slope(pts) do
    n = length(pts)
    {sx, sy} = Enum.reduce(pts, {0.0, 0.0}, fn {x, y}, {a, b} -> {a + x, b + y} end)
    {mx, my} = {sx / n, sy / n}
    {num, den} = Enum.reduce(pts, {0.0, 0.0}, fn {x, y}, {a, b} -> {a + (x - mx) * (y - my), b + (x - mx) ** 2} end)
    if den == 0, do: 0.0, else: num / den
  end

  # ------------------------------------------------------------------- audio --

  @doc "Mean spectral flatness over Hann-windowed frames of `n` (power of two) samples."
  def flatness(%Audio{} = a, n \\ 512) do
    n = fit_pow2(n, length(a.samples))

    a
    |> frame_spectra(n)
    |> Enum.map(fn ps ->
      ps = Enum.map(ps, &max(&1, 1.0e-20))
      g = :math.exp(Enum.reduce(ps, 0.0, &(&2 + :math.log(&1))) / length(ps))
      g / (Enum.sum(ps) / length(ps))
    end)
    |> then(&(Enum.sum(&1) / max(length(&1), 1)))
  end

  @doc "Frequency (Hz) of the strongest bin, over the whole clip's frames."
  def dominant_hz(%Audio{rate: rate} = a, n \\ 2048) do
    n = fit_pow2(n, length(a.samples))
    spectra = frame_spectra(a, n)
    total = spectra |> Enum.zip_with(&Enum.sum/1)
    {_, k} = total |> Enum.with_index() |> Enum.drop(1) |> Enum.max_by(&elem(&1, 0))
    k * rate / n
  end

  @doc "SNR in dB of `out` against `ref` (the shorter length)."
  def snr(%Audio{samples: ref}, %Audio{samples: out}) do
    {s, e} = Enum.zip_reduce(ref, out, {0.0, 0.0}, fn r, o, {s, e} -> {s + r * r, e + (r - o) ** 2} end)
    if e == 0, do: :infinity, else: 10 * :math.log10(s / e)
  end

  # the largest power of two ≤ min(n, len) (at least 2)
  defp fit_pow2(n, len), do: Enum.reduce_while(Stream.iterate(2, &(&1 * 2)), 2, fn p, acc -> if p <= min(n, len), do: {:cont, p}, else: {:halt, acc} end)

  defp frame_spectra(%Audio{samples: s}, n) do
    win = for i <- 0..(n - 1), do: 0.5 - 0.5 * :math.cos(2 * :math.pi() * i / n)

    s
    |> Enum.chunk_every(n, div(n, 2), :discard)
    |> Enum.map(fn fr ->
      fr |> Enum.zip_with(win, &{&1 * &2, 0.0}) |> fft() |> Enum.take(div(n, 2)) |> Enum.map(fn {re, im} -> re * re + im * im end)
    end)
  end

  # ------------------------------------------------------------------- fft --

  @doc "Radix-2 FFT of a list of `{re, im}` (length a power of two)."
  def fft([x]), do: [x]

  def fft(xs) do
    n = length(xs)
    {ev, od} = xs |> Enum.with_index() |> Enum.split_with(fn {_, i} -> rem(i, 2) == 0 end)
    e = fft(Enum.map(ev, &elem(&1, 0)))
    o = fft(Enum.map(od, &elem(&1, 0)))

    tw =
      Enum.zip(o, 0..(div(n, 2) - 1))
      |> Enum.map(fn {{ore, oim}, k} ->
        a = -2 * :math.pi() * k / n
        {c, s} = {:math.cos(a), :math.sin(a)}
        {ore * c - oim * s, ore * s + oim * c}
      end)

    first = Enum.zip_with(e, tw, fn {a, b}, {c, d} -> {a + c, b + d} end)
    second = Enum.zip_with(e, tw, fn {a, b}, {c, d} -> {a - c, b - d} end)
    first ++ second
  end
end
