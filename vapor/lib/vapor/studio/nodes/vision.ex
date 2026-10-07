defmodule Vapor.Studio.Nodes.Vision do
  @moduledoc "Model nodes on the studio's media: the consistent upscaler, OCR (with tables), spoken digits, digits drawn by diffusion."
  @behaviour Vapor.Studio.Node
  alias Vapor.Modal.Image
  alias Vapor.Rejection
  alias Vapor.Studio.Resample
  alias Vapor.Vision.Upscale

  defp n(type, cat, title, doc, inputs, outputs, params),
    do: {__MODULE__, %{type: type, version: 1, category: cat, title: title, doc: doc, inputs: inputs, outputs: outputs, params: params}}

  @impl true
  def nodes do
    [n("image.upscale", "image", "Upscale (AI, consistent)",
       "×2 or ×4. `vapor`: the network trained by vapor, then projected so that shrinking the result gives the input back exactly (it cannot contradict what was seen). `lanczos+consistency`: Lanczos with the same projection. `lanczos`: plain (for comparison). Odd sides are padded by one edge pixel first.",
       [image: :image], [image: :image], [factor: {:enum, ~w(2 4), "2"}, method: {:enum, ["vapor", "lanczos+consistency", "lanczos"], "vapor"}]),
     n("video.upscale", "video", "Upscale video", "Every frame through `image.upscale` (the same model and projection).", [video: :video], [video: :video],
       [factor: {:enum, ~w(2 4), "2"}, method: {:enum, ["vapor", "lanczos+consistency", "lanczos"], "vapor"}]),
     n("vision.ocr", "text", "Read text (OCR)", "Printed text in reading order, tables as Markdown (`Vapor.Vision.OCR`).", [image: :image], [text: :text, confidence: :number], []),
     n("audio.digit", "text", "Spoken digit", "The digit said in a clip (the speech model in priv/speech; resampled to 8 kHz first).", [audio: :audio],
       [text: :text, confidence: :number], []),
     n("image.digit", "image", "Draw a digit", "A handwritten digit by diffusion (8×8, priv/digits), read back by the classifier.", [], [image: :image, reading: :text],
       [digit: {:int, 0, 9, 7}, seed: {:int, 0, 1_000_000, 0}, steps: {:int, 2, 200, 25}, guidance: {:float, 0.0, 8.0, 2.0}])]
  end

  @impl true
  def run("image.upscale", %{image: img}, p, ctx), do: {:ok, %{image: upscale(img, p, ctx)}}

  def run("video.upscale", %{video: v}, p, ctx), do: {:ok, %{video: %{v | frames: Enum.map(v.frames, &upscale(&1, p, ctx))}}}

  def run("vision.ocr", %{image: img}, _, _) do
    case Vapor.Vision.OCR.read(img) do
      {:ok, r} -> {:ok, %{text: r.text, confidence: r.confidence}}
      {:error, _} = e -> e
    end
  end

  def run("audio.digit", %{audio: a}, _, ctx) do
    a = if a.rate == 8000, do: a, else: elem(Vapor.Studio.Nodes.Audio.run("audio.resample", %{audio: a}, %{rate: 8000}, ctx), 1).audio

    with {:ok, model} <- Vapor.Modal.Speech.load(),
         {:ok, r} <- Vapor.Modal.Speech.classify(a, model, worker: ctx.worker || Vapor.Vision.OCR.worker()) do
      {:ok, %{text: to_string(r.label), confidence: r.p}}
    else
      {:error, _} = e -> e
      _ -> {:error, Rejection.new(:speech, "the speech model (priv/speech)", "check the installation")}
    end
  end

  def run("image.digit", _, p, ctx) do
    r = Vapor.Modal.Digits.draw(p.digit, 1, seed: p.seed, steps: p.steps, guidance: p.guidance, worker: ctx.worker || Vapor.Vision.OCR.worker())
    [img] = r.images
    [reading] = r.readings
    {:ok, %{image: Image.new(8, 8, 1, Enum.map(img, &(min(16.0, max(0.0, &1)) / 16))), reading: to_string(reading[:digit] || reading["digit"] || inspect(reading))}}
  end

  defp upscale(img, %{factor: f, method: m}, ctx) do
    times = if f == "4", do: 2, else: 1
    Enum.reduce(1..times, img, fn _, im -> x2(even(im), m, ctx) end)
  end

  defp x2(img, "vapor", ctx), do: Upscale.upscale(img, worker: ctx.worker)
  defp x2(%Image{c: 1} = img, "lanczos+consistency", ctx), do: Upscale.project(Resample.resize(img, 2 * img.w, 2 * img.h, "lanczos", worker: ctx.worker), img)

  defp x2(%Image{} = img, "lanczos+consistency", ctx) do
    up = Resample.resize(img, 2 * img.w, 2 * img.h, "lanczos", worker: ctx.worker)
    chans = for k <- 0..(img.c - 1), do: Upscale.project(channel(up, k), channel(img, k))
    merge(chans)
  end

  defp x2(img, "lanczos", ctx), do: Resample.resize(img, 2 * img.w, 2 * img.h, "lanczos", worker: ctx.worker)

  # odd sides: repeat the last column/row (the projection needs whole 2×2 blocks)
  defp even(%Image{w: w, h: h} = img) when rem(w, 2) == 0 and rem(h, 2) == 0, do: img

  defp even(%Image{w: w, h: h, c: c, px: px}) do
    {w2, h2} = {w + rem(w, 2), h + rem(h, 2)}
    vals = for y <- 0..(h2 - 1), x <- 0..(w2 - 1), k <- 0..(c - 1), do: elem(px, (min(y, h - 1) * w + min(x, w - 1)) * c + k)
    %Image{w: w2, h: h2, c: c, px: List.to_tuple(vals)}
  end

  defp channel(%Image{w: w, h: h, c: c, px: px}, k), do: %Image{w: w, h: h, c: 1, px: List.to_tuple(for(i <- 0..(w * h - 1), do: elem(px, i * c + k)))}

  defp merge([%Image{w: w, h: h} | _] = chans) do
    ts = Enum.map(chans, & &1.px)
    %Image{w: w, h: h, c: length(chans), px: List.to_tuple(for(i <- 0..(w * h - 1), t <- ts, do: elem(t, i)))}
  end
end
