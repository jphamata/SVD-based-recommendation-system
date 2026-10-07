defmodule Vapor.Modal.Audio do
  @moduledoc """
  Audio as rows: the audio codec of the any-to-any layer.

  A clip is a sample rate and mono samples in binary64 (nominally
  `[−1, 1]`). Models read it through two exact or certified steps:

    * `frames/3` cuts it into frames of `n` samples every `hop` samples
      (rows `f32[T, n]`, the last frame zero-padded) — a copy;
    * `spectrum_program/2` is the short-time power spectrum as a program
      of the certified algebra: `P = (X·Cᵀ)² + (X·Sᵀ)²` with the Hann
      window folded into the cosine and sine tables `C, S : f32[bins, n]`.
      The tables are computed with correctly rounded `cos`/`sin`
      (`Vapor.CR`), so the spectrum is the same bits on every substrate —
      a Whisper-style front end with a certificate. A mel filterbank is one
      more `linear`.

  Synthesis goes the other way: `sinusoids_program/3` renders frames from
  per-frame oscillator amplitudes (additive synthesis, one `linear`).

  No dependency: 16-bit PCM WAV (mono or stereo, mixed to mono) is read and
  written here. Generators: `tone/4`, `chord/4`, `noise/3`, `shuffle/2`.
  """
  alias Vapor.{CR, Program, Rejection, Tensor}
  alias Vapor.Algebra.Term, as: T

  @enforce_keys [:rate, :samples]
  defstruct [:rate, :samples]

  @type t :: %__MODULE__{rate: pos_integer, samples: [float]}

  # ------------------------------------------------------------------ files --

  @doc "Read a 16-bit PCM WAV file."
  def read(path) do
    case File.read(path) do
      {:ok, bin} -> parse(bin)
      {:error, why} -> {:error, Rejection.new({:audio, path}, "readable (#{inspect(why)})", "check the path")}
    end
  end

  @doc false
  def parse(<<"RIFF", _::32, "WAVE", rest::binary>>), do: chunks(rest, nil)
  def parse(_), do: bad("a RIFF/WAVE file")

  defp chunks(<<"fmt ", n::32-little, body::binary-size(n), rest::binary>>, _fmt) do
    <<format::16-little, ch::16-little, rate::32-little, _::32, _::16, bits::16-little, _::binary>> = body
    if format == 1 and bits == 16 and ch in [1, 2], do: chunks(pad(rest, n), {ch, rate}), else: bad("16-bit PCM, mono or stereo")
  end

  defp chunks(<<"data", n::32-little, rest::binary>>, {ch, rate}) do
    data = binary_part(rest, 0, min(n, byte_size(rest)))
    raw = for <<s::signed-16-little <- data>>, do: s / 32768
    samples = if ch == 2, do: raw |> Enum.chunk_every(2) |> Enum.map(fn [l, r] -> (l + r) / 2; [l] -> l end), else: raw
    {:ok, %__MODULE__{rate: rate, samples: samples}}
  end

  defp chunks(<<_id::binary-size(4), n::32-little, rest::binary>>, fmt) when byte_size(rest) >= n,
    do: chunks(pad(binary_part(rest, n, byte_size(rest) - n), n), fmt)

  defp chunks(_, _), do: bad("fmt and data chunks")

  defp pad(rest, n) when rem(n, 2) == 1 and byte_size(rest) > 0, do: binary_part(rest, 1, byte_size(rest) - 1)
  defp pad(rest, _), do: rest

  defp bad(b), do: {:error, Rejection.new(:wav, b, "write the audio as 16-bit PCM WAV")}

  @doc "Write a 16-bit PCM mono WAV (samples × 32768, clamped to the int16 range)."
  def write(path, %__MODULE__{} = a), do: File.write(path, encode(a))

  @doc false
  def encode(%__MODULE__{rate: rate, samples: s}) do
    # the inverse of reading (s/32768), clamped to the int16 range
    data = for x <- s, into: <<>>, do: <<min(32767, max(-32768, round(x * 32768)))::signed-16-little>>
    n = byte_size(data)

    <<"RIFF", 36 + n::32-little, "WAVE", "fmt ", 16::32-little, 1::16-little, 1::16-little, rate::32-little,
      rate * 2::32-little, 2::16-little, 16::16-little, "data", n::32-little>> <> data
  end

  # ------------------------------------------------------------- generators --

  @doc "A sine at `freq` Hz, `n` samples, amplitude `amp`."
  def tone(freq, rate, n, amp \\ 0.5), do: chord([freq], rate, n, amp)

  @doc "Equal-amplitude sines (total amplitude `amp`)."
  def chord(freqs, rate, n, amp \\ 0.5) do
    k = length(freqs)

    samples =
      for i <- 0..(n - 1) do
        Enum.reduce(freqs, 0.0, fn f, acc -> acc + CR.sin_f64(2 * :math.pi() * f * i / rate) end) * amp / k
      end

    %__MODULE__{rate: rate, samples: samples}
  end

  @doc "White noise in [−amp, amp)."
  def noise(rate, n, seed, amp \\ 0.5),
    do: %__MODULE__{rate: rate, samples: Enum.map(Vapor.Modal.Rng.uniform(seed, n), &((2 * &1 - 1) * amp))}

  @doc "The same samples in a seeded random order."
  def shuffle(%__MODULE__{} = a, seed), do: %{a | samples: Vapor.Modal.Rng.permute(a.samples, seed)}

  @doc "Sample-wise sum (lengths: the shorter one)."
  def mix(%__MODULE__{} = a, %__MODULE__{samples: b}), do: %{a | samples: Enum.zip_with(a.samples, b, &+/2)}

  # ----------------------------------------------------------------- frames --

  @doc "Frames of `n` samples every `hop` samples as rows `f32[T, n]` (the tail zero-padded)."
  def frames(%__MODULE__{samples: s}, n, hop) do
    len = length(s)
    count = max(1, div(max(len - n, 0) + hop - 1, hop) + 1)
    tup = List.to_tuple(s)
    vals = for f <- 0..(count - 1), i <- 0..(n - 1), do: (if f * hop + i < len, do: elem(tup, f * hop + i), else: 0.0)
    Tensor.from_list(:f32, [count, n], vals)
  end

  @doc "Concatenate frames (hop = frame length): rows `f32[T, n]` → a clip."
  def from_frames(%Tensor{} = rows, rate), do: %__MODULE__{rate: rate, samples: Tensor.to_floats(rows)}

  # ---------------------------------------------------------------- spectra --

  @doc """
  The short-time power spectrum of frames `f32[T, n]` as a program:
  output `power : f32[T, bins]`, bin `k` at `k·rate/n` Hz, Hann-windowed.
  `bins` ≤ n/2 + 1 (default n/2, so that a power of two stays one).
  """
  def spectrum_program(n, t, bins \\ nil) do
    bins = bins || div(n, 2)
    {c, s} = dft_tables(n, bins)
    x = T.input(:rows, :f32, [t, n])
    re = T.linear(x, T.ref(:dft_cos, T.const(c)))
    im = T.linear(x, T.ref(:dft_sin, T.const(s)))
    Program.new([power: T.add(T.mul(re, re), T.mul(im, im))], lets: [dft_cos: T.const(c), dft_sin: T.const(s)])
  end

  @doc "Hann-windowed DFT tables `C, S : f32[bins, n]` (correctly rounded)."
  def dft_tables(n, bins) do
    win = for i <- 0..(n - 1), do: 0.5 - 0.5 * CR.cos_f64(2 * :math.pi() * i / n)

    {cs, ss} =
      for k <- 0..(bins - 1), {w, i} <- Enum.with_index(win), reduce: {[], []} do
        {cs, ss} ->
          a = 2 * :math.pi() * rem(k * i, n) / n
          {[w * CR.cos_f64(a) | cs], [-w * CR.sin_f64(a) | ss]}
      end

    {Tensor.from_list(:f32, [bins, n], Enum.reverse(cs)), Tensor.from_list(:f32, [bins, n], Enum.reverse(ss))}
  end

  @doc """
  Additive synthesis as a program: per-frame amplitudes `amps : f32[T, m]`
  of `m` oscillators at `freqs` (Hz) → frames `rows : f32[T, n]`, each a
  sum of sines in phase with the clip's time origin (frame `f` starts at
  sample `f·n`, so consecutive frames join without a seam).
  """
  def sinusoids_program(freqs, rate, n, t) do
    m = length(freqs)
    mp = div(m + 15, 16) * 16
    # frame-relative basis; phase continuity holds when every frequency
    # completes whole cycles per frame (choose n accordingly)
    basis =
      for i <- 0..(n - 1), j <- 0..(mp - 1) do
        if j < m, do: CR.sin_f64(2 * :math.pi() * Enum.at(freqs, j) * i / rate), else: 0.0
      end

    b = Tensor.from_list(:f32, [n, mp], basis)
    amps = T.input(:amps, :f32, [t, mp])
    Program.new([rows: T.linear(amps, T.ref(:basis, T.const(b)))], lets: [basis: T.const(b)])
  end

  @doc "The frequency (Hz) of bin `k` of an `n`-sample frame."
  def bin_hz(k, n, rate), do: k * rate / n
end
