defmodule Vapor.Modal.Digits do
  @moduledoc """
  Real handwriting, both ways: the routes image → text and text → image
  over 1 797 real 8×8 handwritten digits (UCI, via scikit-learn's
  `load_digits`; 1 300 for training, 497 never seen), with models admitted
  through the airlock (`priv/digits`, trained by `test/python/train_digits.py`):

    * `read/2` — a `vapor_mlp` classifier, image → digit;
    * `draw/3` — a `vapor_mlp` noise predictor sampled by DDIM
      (`Vapor.Modal.Diffusion`), digit → image, deterministic in its seed;
    * every generated image is then *read back* by the classifier — which
      was trained on real images only — and compared with its nearest
      training image (a copy would sit at distance ≈ 0).

  With `Vapor.Modal.Speech` this closes a chain on real signals: a spoken
  digit (a real voice) → text → a generated handwritten digit.
  """
  alias Vapor.{Lock, Tensor}
  alias Vapor.Modal.{Diffusion, Runner}

  @doc "The models and the data (cached): `%{classifier, denoiser, images, labels, train}`."
  def models do
    case :persistent_term.get({__MODULE__, :models}, nil) do
      nil ->
        dir = Path.join(to_string(:code.priv_dir(:vapor)), "digits")

        with {:ok, c} <- Lock.open(Path.join(dir, "classifier")),
             {:ok, cp} <- Lock.build(c.spec, c.weights, rows: 1),
             {:ok, den} <- Diffusion.load(Path.join(dir, "denoiser")),
             {:ok, data} <- Vapor.Ingest.Safetensors.read(Path.join(dir, "data.safetensors")) do
          imgs = data["images"].data |> :binary.bin_to_list() |> Enum.chunk_every(64)
          m = %{classifier: %{spec: c.spec, weights: c.weights, program: cp}, denoiser: den, images: imgs,
                labels: Tensor.to_list(data["labels"]), train: Tensor.to_list(data["train"])}
          :persistent_term.put({__MODULE__, :models}, m)
          {:ok, m}
        end

      m ->
        {:ok, m}
    end
  end

  @doc "Images of a split (`:train` | `:test`) with their labels: `[{pixels 0–16, label}]`."
  def split(m, which) do
    want = if which == :train, do: 1, else: 0
    for {{img, l}, t} <- Enum.zip(Enum.zip(m.images, m.labels), m.train), t == want, do: {img, l}
  end

  @doc """
  Read digits: `pixels` are 64 grey levels in 0–16 (or a list of such
  images). Returns `[%{digit, p, probs}]`. Option `worker`.
  """
  def read(images, opts \\ []) do
    {:ok, m} = models()
    images = if is_list(hd(images)), do: images, else: [images]
    n = length(images)
    p = if n == 1, do: m.classifier.program, else: elem(Lock.build(m.classifier.spec, m.classifier.weights, rows: n), 1)
    x = Tensor.from_list(:f32, [n, 64], Enum.flat_map(images, fn img -> Enum.map(img, &(&1 / 16)) end))
    out = Runner.run(p, %{rows: x}, worker: opts[:worker]).out |> Tensor.to_floats() |> Enum.chunk_every(10)
    Enum.map(out, &softmax_pick/1)
  end

  defp softmax_pick(logits) do
    mx = Enum.max(logits)
    es = Enum.map(logits, &:math.exp(&1 - mx))
    z = Enum.sum(es)
    probs = Enum.map(es, &(&1 / z))
    {p, d} = probs |> Enum.with_index() |> Enum.max_by(&elem(&1, 0))
    %{digit: d, p: p, probs: probs}
  end

  @doc """
  Draw `n` handwritten images of `digit` (0–9). Options: `seed` (0),
  `steps` (25), `guidance` (2.0), `worker`, `trace` (keep x̂₀ of every
  step). Returns `%{images: [[0–16 levels as floats]], trace, readings,
  nearest}` — each image read back by the classifier and its distance to
  the closest training image (in grey levels).
  """
  def draw(digit, n \\ 1, opts \\ []) when digit in 0..9 do
    {:ok, m} = models()
    r = Diffusion.sample(m.denoiser, digit, n, opts)
    to_levels = fn xs -> Enum.map(xs, &((&1 + 1) * 8)) end
    images = Enum.map(r.samples, to_levels)
    train = m |> split(:train) |> Enum.map(fn {img, _} -> img end)

    %{images: images, readings: read(Enum.map(images, fn img -> Enum.map(img, &min(max(&1, 0.0), 16.0)) end), opts),
      nearest: Diffusion.nearest(images, train), timesteps: r.timesteps,
      trace: Enum.map(r.trace, fn step -> Enum.map(step, to_levels) end)}
  end
end
