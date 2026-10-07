defmodule Mix.Tasks.Vapor.Upscale do
  @shortdoc "Train, measure and apply the consistent AI upscaler"
  @moduledoc """
      mix vapor.upscale apply IN OUT [--factor 2|4] [--method vapor|lanczos+consistency|lanczos]
      mix vapor.upscale train DATA [--steps 30000] [--out priv/upscale] [--seed 1]
      mix vapor.upscale eval DATA [--model DIR]

  `apply` reads PNG/JPEG/GIF/PPM and writes PNG (or PPM by extension).
  `DATA` is a directory written by `test/python/upscale_data.py` (`train/`,
  `val/`, `test/` of greyscale PGMs). `train` writes the model and a
  receipt of how it was made (data digest, seed, schedule, loss curve,
  weights digest): run it again with the same data and seed and the
  weights are the same bits. `eval` prints PSNR against Lanczos, bicubic
  and Lanczos with the same consistency projection, and the largest
  |D(y) − x| (0 for a consistent upscaling).
  """
  use Mix.Task
  alias Vapor.Modal.Image
  alias Vapor.Studio.Resample
  alias Vapor.Vision.Upscale

  @switches [steps: :integer, out: :string, seed: :integer, model: :string, factor: :integer, method: :string]

  @impl true
  def run(argv) do
    {o, args, _} = OptionParser.parse(argv, strict: @switches)
    Mix.Task.run("app.start")
    w = Vapor.Vision.OCR.worker()

    case args do
      ["apply", input, output] -> apply_one(input, output, o, w)
      ["train", data] -> train(data, o, w)
      ["eval", data] -> eval(data, o, w)
      _ -> Mix.raise("usage: mix vapor.upscale apply IN OUT | train DATA | eval DATA")
    end
  end

  defp load(dir, split), do: Path.wildcard(Path.join([dir, split, "*.pgm"])) |> Enum.sort() |> Enum.map(fn f -> {:ok, i} = Image.read(f); {Path.basename(f, ".pgm"), i} end)

  defp apply_one(input, output, o, w) do
    {:ok, img} = Vapor.Studio.Nodes.Image.decode(File.read!(input))
    p = %{factor: to_string(o[:factor] || 2), method: o[:method] || "vapor"}
    {:ok, %{image: out}} = Vapor.Studio.Nodes.Vision.run("image.upscale", %{image: img}, p, %{worker: w})
    bytes = if Path.extname(output) in [".ppm", ".pgm"], do: Image.encode(out), else: Image.png(out)
    File.write!(output, bytes)
    Mix.shell().info("#{input} #{img.w}×#{img.h} → #{output} #{out.w}×#{out.h} (#{p.method})")
  end

  defp train(data, o, w) do
    steps = o[:steps] || 30_000
    seed = o[:seed] || 1
    imgs = load(data, "train")
    t0 = System.monotonic_time(:millisecond)
    {net, info} = Upscale.train(Enum.map(imgs, &elem(&1, 1)), worker: w, steps: steps, seed: seed, per_image: 6000,
                                 log: fn {s, l} -> if rem(s, 5000) == 0, do: Mix.shell().info("step #{s}: loss #{Float.round(l, 6)}") end)
    cfg = Upscale.save(net, o[:out] || "priv/upscale",
      %{"steps" => steps, "seed" => seed, "batch" => info.batch, "lr" => 2.0e-3, "lr_end" => 0.02, "per_image" => 6000,
        "train_images" => Enum.map(imgs, &elem(&1, 0)), "data_sha256" => info.data_digest,
        "loss_curve" => Enum.take_every(info.losses, max(1, div(length(info.losses), 24))) |> Enum.map(&Tuple.to_list/1),
        "seconds" => div(System.monotonic_time(:millisecond) - t0, 1000)})
    Mix.shell().info("model #{cfg["weights_sha256"]} → #{o[:out] || "priv/upscale"}")
  end

  defp eval(data, o, w) do
    model = case o[:model] do
      nil -> elem(Upscale.default(), 1)
      d -> elem(Upscale.load(d), 1)
    end

    Mix.shell().info("| image | bicubic | Lanczos | Lanczos + consistency | vapor | max ‖D(y) − x‖ Lanczos | vapor |")
    Mix.shell().info("|---|---:|---:|---:|---:|---:|---:|")

    for split <- ["val", "test"], {name, hr} <- load(data, split) do
      lr = Upscale.downsample(hr)
      lz = Resample.resize(lr, hr.w, hr.h, "lanczos", worker: w)
      bc = Resample.resize(lr, hr.w, hr.h, "bicubic", worker: w)
      v = Upscale.upscale(lr, model: model, worker: w)
      f = &:erlang.float_to_binary(Upscale.psnr(&1, hr), decimals: 2)
      Mix.shell().info("| #{split}/#{name} | #{f.(bc)} | #{f.(lz)} | #{f.(Upscale.project(lz, lr))} | #{f.(v)} | #{Float.round(Upscale.inconsistency(lz, lr), 4)} | #{Upscale.inconsistency(v, lr)} |")
    end
  end
end
