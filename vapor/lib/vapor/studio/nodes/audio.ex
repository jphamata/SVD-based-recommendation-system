defmodule Vapor.Studio.Nodes.Audio do
  @moduledoc """
  Audio nodes: load, synthesize, gain, mix, concatenate, trim, fade,
  normalize, reverse, resample, spectrogram.

  Resampling uses windowed-sinc (Lanczos, `a = 8`) weights, widened by the
  ratio when the rate goes down (so it filters before it decimates), summed
  in the canonical 16-lane order (`Vapor.Studio.Resample`) — a function of
  the samples alone. Oscillators use the correctly rounded `sin` and gains
  the correctly rounded `pow` (`Vapor.CR`): the same samples on every
  machine.
  """
  @behaviour Vapor.Studio.Node
  alias Vapor.{CR, F32, Rejection}
  alias Vapor.Modal.Audio
  alias Vapor.Studio.Resample

  defp n(type, title, doc, inputs, outputs, params),
    do: {__MODULE__, %{type: type, version: 1, category: "audio", title: title, doc: doc, inputs: inputs, outputs: outputs, params: params}}

  @impl true
  def nodes do
    [n("audio.load", "Load audio", "PCM WAV (8/16/24/32-bit integer or 32-bit float, mono or mixed down) from `data` or `path`.", [], [audio: :audio],
       [data: {:data, nil}, path: {:string, ""}]),
     n("audio.tone", "Tone", "An oscillator: sine, square, sawtooth or triangle.", [], [audio: :audio],
       [frequency: {:float, 1.0, 20_000.0, 440.0}, seconds: {:float, 0.01, 600.0, 1.0}, rate: {:int, 4000, 192_000, 16_000},
        amplitude: {:float, 0.0, 1.0, 0.5}, waveform: {:enum, ~w(sine square sawtooth triangle), "sine"}]),
     n("audio.noise", "Noise", "White noise from a seed.", [], [audio: :audio],
       [seconds: {:float, 0.01, 600.0, 1.0}, rate: {:int, 4000, 192_000, 16_000}, amplitude: {:float, 0.0, 1.0, 0.3}, seed: {:int, 0, 2_147_483_647, 0}]),
     n("audio.gain", "Gain", "Multiply by 10^(dB/20).", [audio: :audio], [audio: :audio], [db: {:float, -96.0, 48.0, 0.0}]),
     n("audio.mix", "Mix", "a + gain·b, at one rate (the longer length; the shorter padded with silence).", [a: :audio, b: :audio], [audio: :audio],
       [gain: {:float, 0.0, 4.0, 1.0}]),
     n("audio.concat", "Concatenate", "a, then b (one rate).", [a: :audio, b: :audio], [audio: :audio], []),
     n("audio.trim", "Trim", "Keep [start, end) seconds (end 0 = to the end).", [audio: :audio], [audio: :audio],
       [start: {:float, 0.0, 36_000.0, 0.0}, end: {:float, 0.0, 36_000.0, 0.0}]),
     n("audio.fade", "Fade", "Linear fade in and out, in seconds.", [audio: :audio], [audio: :audio], [fade_in: {:float, 0.0, 600.0, 0.05}, fade_out: {:float, 0.0, 600.0, 0.05}]),
     n("audio.normalize", "Normalize", "Scale so the peak is at `peak_db` dBFS.", [audio: :audio], [audio: :audio], [peak_db: {:float, -60.0, 0.0, -1.0}]),
     n("audio.reverse", "Reverse", "Play backwards.", [audio: :audio], [audio: :audio], []),
     n("audio.resample", "Resample", "To another rate: windowed sinc (Lanczos a = 8), antialiased when going down.", [audio: :audio], [audio: :audio],
       [rate: {:int, 4000, 192_000, 16_000}]),
     n("audio.spectrogram", "Spectrogram", "Log-magnitude STFT (Hann window) as a greyscale image, low frequencies at the bottom.", [audio: :audio], [image: :image],
       [window: {:enum, ~w(128 256 512 1024), "256"}, hop: {:int, 16, 1024, 128}, floor_db: {:float, -140.0, -20.0, -80.0}])]
  end

  @impl true
  def run("audio.load", _, %{data: d, path: p}, ctx) do
    with {:ok, bytes} <- Vapor.Studio.Nodes.Image.source(d, p, ctx), {:ok, a} <- Audio.parse(bytes), do: {:ok, %{audio: a}}
  end

  def run("audio.tone", _, p, _) do
    n = round(p.seconds * p.rate)
    k = p.frequency / p.rate

    s =
      for i <- 0..(n - 1) do
        ph = k * i - Float.floor(k * i)
        v =
          case p.waveform do
            "sine" -> CR.sin_f64(2.0 * :math.pi() * ph)
            "square" -> if ph < 0.5, do: 1.0, else: -1.0
            "sawtooth" -> 2.0 * ph - 1.0
            "triangle" -> 1.0 - 4.0 * abs(ph - 0.5)
          end

        p.amplitude * v
      end

    {:ok, %{audio: %Audio{rate: p.rate, samples: s}}}
  end

  def run("audio.noise", _, p, _), do: {:ok, %{audio: Audio.noise(p.rate, round(p.seconds * p.rate), p.seed, p.amplitude)}}

  def run("audio.gain", %{audio: a}, %{db: db}, _) do
    g = CR.pow_f64(10.0, db / 20.0)
    {:ok, %{audio: %{a | samples: Enum.map(a.samples, &(&1 * g))}}}
  end

  def run("audio.mix", %{a: a, b: b}, %{gain: g}, _) do
    with :ok <- same_rate(a, b, "mix") do
      n = max(length(a.samples), length(b.samples))
      pad = fn s -> s ++ List.duplicate(0.0, n - length(s)) end
      {:ok, %{audio: %Audio{rate: a.rate, samples: Enum.zip_with(pad.(a.samples), pad.(b.samples), &(&1 + g * &2))}}}
    end
  end

  def run("audio.concat", %{a: a, b: b}, _, _) do
    with :ok <- same_rate(a, b, "concat"), do: {:ok, %{audio: %Audio{rate: a.rate, samples: a.samples ++ b.samples}}}
  end

  def run("audio.trim", %{audio: a}, %{start: s, end: e}, _) do
    i0 = min(round(s * a.rate), length(a.samples))
    i1 = if e == 0.0, do: length(a.samples), else: min(round(e * a.rate), length(a.samples))
    {:ok, %{audio: %{a | samples: Enum.slice(a.samples, i0, max(0, i1 - i0))}}}
  end

  def run("audio.fade", %{audio: a}, %{fade_in: fi, fade_out: fo}, _) do
    n = length(a.samples)
    {ni, no} = {round(fi * a.rate), round(fo * a.rate)}

    s =
      a.samples
      |> Enum.with_index()
      |> Enum.map(fn {v, i} ->
        gi = if ni > 0 and i < ni, do: i / ni, else: 1.0
        go = if no > 0 and i >= n - no, do: (n - 1 - i) / no, else: 1.0
        v * gi * go
      end)

    {:ok, %{audio: %{a | samples: s}}}
  end

  def run("audio.normalize", %{audio: a}, %{peak_db: db}, _) do
    peak = a.samples |> Enum.map(&abs/1) |> Enum.max(fn -> 0.0 end)
    g = if peak == 0.0, do: 1.0, else: CR.pow_f64(10.0, db / 20.0) / peak
    {:ok, %{audio: %{a | samples: Enum.map(a.samples, &(&1 * g))}}}
  end

  def run("audio.reverse", %{audio: a}, _, _), do: {:ok, %{audio: %{a | samples: Enum.reverse(a.samples)}}}

  def run("audio.resample", %{audio: %Audio{rate: r} = a}, %{rate: r}, _), do: {:ok, %{audio: a}}

  def run("audio.resample", %{audio: a}, %{rate: r2}, _) do
    n = length(a.samples)
    n2 = max(1, round(n * r2 / a.rate))
    # samples are points in time (not pixel areas): output i sits at input i·n/n2
    ws = Resample.weights("lanczos:8", n, n2, antialias: true, offset: 0.5 - 0.5 * n / n2)
    st = a.samples |> Enum.map(&F32.from_float/1) |> List.to_tuple()
    out = Enum.map(ws, fn taps -> F32.to_float(dot(taps, st)) end)
    {:ok, %{audio: %Audio{rate: r2, samples: out}}}
  end

  def run("audio.spectrogram", %{audio: a}, p, _) do
    nw = String.to_integer(p.window)
    hop = p.hop
    s = List.to_tuple(a.samples)
    n = tuple_size(s)
    frames = max(1, div(max(n - nw, 0), hop) + 1)
    bins = div(nw, 2) + 1
    hann = for i <- 0..(nw - 1), do: 0.5 - 0.5 * CR.cos_f64(2.0 * :math.pi() * i / nw)
    # nw distinct angles: one correctly rounded cos and sin each
    cos_t = List.to_tuple(for(m <- 0..(nw - 1), do: CR.cos_f64(2.0 * :math.pi() * m / nw)))
    sin_t = List.to_tuple(for(m <- 0..(nw - 1), do: CR.sin_f64(2.0 * :math.pi() * m / nw)))
    cs = for k <- 0..(bins - 1), do: List.to_tuple(for(i <- 0..(nw - 1), do: elem(cos_t, rem(k * i, nw))))
    sn = for k <- 0..(bins - 1), do: List.to_tuple(for(i <- 0..(nw - 1), do: elem(sin_t, rem(k * i, nw))))
    tables = Enum.zip(cs, sn)

    cols =
      for f <- 0..(frames - 1) do
        x = for {w, i} <- Enum.with_index(hann), do: w * (if f * hop + i < n, do: elem(s, f * hop + i), else: 0.0)
        xt = List.to_tuple(x)

        for {c, sv} <- tables do
          {re, im} = Enum.reduce(0..(nw - 1), {0.0, 0.0}, fn i, {r, m} -> {r + elem(xt, i) * elem(c, i), m - elem(xt, i) * elem(sv, i)} end)
          mag = (re * re + im * im) / (nw * nw)
          db = if mag <= 0.0, do: p.floor_db, else: max(p.floor_db, 10.0 * CR.log_f64(mag) / CR.log_f64(10.0))
          (db - p.floor_db) / -p.floor_db
        end
      end

    ct = List.to_tuple(Enum.map(cols, &List.to_tuple/1))
    vals = for row <- 0..(bins - 1), f <- 0..(frames - 1), do: elem(elem(ct, f), bins - 1 - row)
    {:ok, %{image: Vapor.Modal.Image.new(frames, bins, 1, vals)}}
  end

  # the canonical dot (lane j mod 16, then the tree) over the non-zero taps
  defp dot(taps, st) do
    lanes = Enum.reduce(taps, %{}, fn {j, w}, acc -> (fn p -> Map.update(acc, rem(j, 16), F32.add(0, p), &F32.add(&1, p)) end).(F32.mul(w, elem(st, j))) end)
    Vapor.Runtime.Oracle.reduce16(for l <- 0..15, do: Map.get(lanes, l, 0))
  end

  defp same_rate(%Audio{rate: r}, %Audio{rate: r}, _), do: :ok
  defp same_rate(a, b, what), do: {:error, Rejection.new({what, :rate}, "two clips at one rate", "#{a.rate} Hz and #{b.rate} Hz: resample one first")}
end
