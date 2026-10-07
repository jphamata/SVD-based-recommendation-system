defmodule Vapor.Modal.Speech do
  @moduledoc """
  Speech as rows: the front end that turns a recording into the frames an
  encoder reads, with every multiply-add certified.

      clip (8 kHz) ─ frames of 256 samples every 128 (32 ms / 16 ms)
                   ─ Hann power spectrum  P = (X·Cᵀ)² + (X·Sᵀ)²     ┐ one program of the
                   ─ 32 mel bands         M = P·Fᵀ (triangular filters) ┘ certified algebra
                   ─ log, then each band minus its mean over the clip (binary64, `Vapor.CR`)
                   ─ rows f32[64, 32] (1.024 s; shorter clips padded, the horizon hides it)

  The DFT and filterbank tables are computed with correctly rounded
  `cos`/`sin`/`log`, so the features are the same bits on every substrate
  and every host; the per-clip mean (cepstral-style normalisation) removes
  the channel and much of the speaker. `classify/3` runs a `vapor_encoder`
  checkpoint (`head: "pooled"`, `labels`) over them — the spoken-digit
  reader in `priv/speech` is trained on five speakers of the Free Spoken
  Digit Dataset and measured on a sixth (`docs/ANY_TO_ANY.md`).
  """
  alias Vapor.{CR, Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Modal.{Audio, Runner}

  @rate 8000
  @n 256
  @hop 128
  @bands 32
  @frames 64

  @doc "Feature geometry: `%{rate, n, hop, bands, frames}`."
  def geometry, do: %{rate: @rate, n: @n, hop: @hop, bands: @bands, frames: @frames}

  @doc """
  The log-mel rows of a clip: `{rows : f32[real, 32], real}` with `real ≤ 64`
  frames (a longer clip is cut at 1.024 s). Option `worker` (the spectrum
  and filterbank run there; the oracle otherwise — the same bits).
  """
  def features(clip, opts \\ [])

  def features(%Audio{rate: @rate} = clip, opts) do
    fr = Audio.frames(clip, @n, @hop)
    [t, _] = fr.shape
    real = min(t, @frames)
    rows = Tensor.new(:f32, [real, @n], binary_part(fr.data, 0, real * @n * 4))
    mel = Runner.run(program(real), %{rows: rows}, worker: opts[:worker]).mel

    logs =
      mel
      |> Tensor.to_floats()
      |> Enum.map(&CR.log_f64(max(&1, 0.0) + 1.0e-6))
      |> Enum.chunk_every(@bands)

    means = logs |> Enum.zip_with(& &1) |> Enum.map(&(Enum.sum(&1) / real))
    norm = Enum.map(logs, fn row -> Enum.zip_with(row, means, &(&1 - &2)) end)
    {Tensor.from_list(:f32, [real, @bands], List.flatten(norm)), real}
  end

  def features(%Audio{rate: r}, _opts), do: {:error, Vapor.Rejection.new(:speech, "audio at #{@rate} Hz (got #{r})", "resample to 8 kHz")}

  @doc "The certified part of the front end as a program over `t` frames: `mel : f32[t, 32]`."
  def program(t) do
    key = {__MODULE__, :program, t}

    case :persistent_term.get(key, nil) do
      nil ->
        bins = div(@n, 2)
        {c, s} = Audio.dft_tables(@n, bins)
        fb = filterbank(bins)
        x = T.input(:rows, :f32, [t, @n])
        re = T.linear(x, T.ref(:dft_cos, T.const(c)))
        im = T.linear(x, T.ref(:dft_sin, T.const(s)))
        p = T.add(T.mul(re, re), T.mul(im, im))
        mel = T.linear(p, T.ref(:mel_fb, T.const(fb)))
        prog = Program.new([mel: mel], lets: [dft_cos: T.const(c), dft_sin: T.const(s), mel_fb: T.const(fb)])
        :persistent_term.put(key, prog)
        prog

      p ->
        p
    end
  end

  @doc "Triangular mel filters `f32[32, bins]` over 0 … rate/2 (HTK mel scale, correctly rounded logs)."
  def filterbank(bins) do
    mel = fn hz -> 2595.0 * CR.log_f64(1.0 + hz / 700.0) / CR.log_f64(10.0) end
    inv = fn m -> 700.0 * (CR.exp_f64(m / 2595.0 * CR.log_f64(10.0)) - 1.0) end
    top = mel.(@rate / 2)
    edges = for i <- 0..(@bands + 1), do: inv.(top * i / (@bands + 1))
    hz = fn k -> k * @rate / @n end

    vals =
      for b <- 1..@bands, k <- 0..(bins - 1) do
        {lo, mid, hi} = {Enum.at(edges, b - 1), Enum.at(edges, b), Enum.at(edges, b + 1)}
        f = hz.(k)

        cond do
          f > lo and f <= mid -> (f - lo) / (mid - lo)
          f > mid and f < hi -> (hi - f) / (hi - mid)
          true -> 0.0
        end
      end

    Tensor.from_list(:f32, [@bands, bins], vals)
  end

  @doc """
  Classify a clip with a `vapor_encoder` checkpoint of this front end
  (`row_width` 32, `cls`, a pooled head, `labels`): `{:ok, %{label, p,
  probs}}`. `model` is `%{spec, program}` (see `load/1`).
  """
  def classify(%Audio{} = clip, model, opts \\ []) do
    {rows, _real} = features(clip, opts)
    env = Vapor.Lock.Adapters.Encoder.input(model.spec, rows)
    logits = Runner.run(model.program, env, worker: opts[:worker]).logits |> Tensor.to_floats()
    m = Enum.max(logits)
    es = Enum.map(logits, &:math.exp(&1 - m))
    z = Enum.sum(es)
    probs = Enum.map(es, &(&1 / z))
    {p, i} = probs |> Enum.with_index() |> Enum.max_by(&elem(&1, 0))
    {:ok, %{label: Enum.at(model.labels, i), p: p, probs: probs}}
  end

  @doc """
  Features of every recording of a Free Spoken Digit Dataset checkout
  (`DIR/recordings/<digit>_<speaker>_<n>.wav`), for training elsewhere:
  writes `out` (safetensors: `x : f32[N, 64, 32]` zero-padded, `len :
  s32[N]`, `digit : s32[N]`, `speaker : s32[N]`, `index : s32[N]`) and
  returns the speakers in id order.
  """
  def dataset(dir, out, opts \\ []) do
    files = Path.wildcard(Path.join([dir, "recordings", "*.wav"])) |> Enum.sort()
    meta = Enum.map(files, fn f -> [d, spk, i] = f |> Path.basename(".wav") |> String.split("_"); {f, String.to_integer(d), spk, String.to_integer(i)} end)
    speakers = meta |> Enum.map(&elem(&1, 2)) |> Enum.uniq() |> Enum.sort()
    w = opts[:worker]

    rows =
      meta
      |> Task.async_stream(fn {f, d, spk, i} ->
        {:ok, clip} = Audio.read(f)
        {x, real} = features(clip, worker: w)
        pad = :binary.copy(<<0::32>>, (@frames - real) * @bands)
        {x.data <> pad, real, d, Enum.find_index(speakers, &(&1 == spk)), i}
      end, timeout: :infinity, ordered: true, max_concurrency: if(w, do: 1, else: System.schedulers_online()))
      |> Enum.map(fn {:ok, r} -> r end)

    n = length(rows)
    t = fn l -> Tensor.from_list(:s32, [n], l) end

    :ok = Vapor.Ingest.Safetensors.write(out, %{
      "x" => Tensor.new(:f32, [n, @frames, @bands], rows |> Enum.map(&elem(&1, 0)) |> IO.iodata_to_binary()),
      "len" => t.(Enum.map(rows, &elem(&1, 1))), "digit" => t.(Enum.map(rows, &elem(&1, 2))),
      "speaker" => t.(Enum.map(rows, &elem(&1, 3))), "index" => t.(Enum.map(rows, &elem(&1, 4)))})

    speakers
  end

  @doc "Admit a speech classifier checkpoint (`priv/speech` by default)."
  def load(dir \\ Path.join(to_string(:code.priv_dir(:vapor)), "speech")) do
    with {:ok, m} <- Vapor.Lock.open(dir),
         {:ok, p} <- Vapor.Lock.build(m.spec, m.weights) do
      {:ok, %{spec: m.spec, program: p, labels: m.spec.config.raw["labels"] || Enum.map(0..9, &Integer.to_string/1)}}
    end
  end
end
