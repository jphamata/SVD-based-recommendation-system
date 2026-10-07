defmodule Vapor.Studio.Nodes.Diffusion do
  @moduledoc """
  Latent diffusion in the studio (`Vapor.Diffusion.Pipeline`): the
  ComfyUI-shaped nodes — checkpoint, text conditioning, empty latent,
  sampler, VAE decode/encode — and one all-in-one `diffusion.generate`
  (text → image, image → image, inpainting) for agents and short graphs.

  A checkpoint is a **diffusers directory** (unet/, vae/, text_encoder/,
  scheduler/, tokenizer/) inside the studio's directory or under
  `VAPOR_MODELS`. Single-file `.safetensors`/`.ckpt` checkpoints are
  refused: convert them once with diffusers (`from_single_file(...)
  .save_pretrained(dir)`).

  The cache keys a checkpoint by its path: replacing the weights at the
  same path is not noticed by the cache — use a new directory name.
  """
  @behaviour Vapor.Studio.Node
  alias Vapor.{Rejection, Tensor}
  alias Vapor.Diffusion.Pipeline

  @samplers ~w(dpmpp_2m euler ddim)

  @impl true
  def nodes do
    sampling = [steps: {:int, 1, 200, 25}, cfg: {:float, 0.0, 30.0, 7.5}, sampler: {:enum, @samplers, "dpmpp_2m"}, seed: {:int, 0, 4_294_967_295, 0}]

    [
      {__MODULE__, %{type: "diffusion.checkpoint", version: 1, category: "diffusion", title: "Checkpoint",
                     doc: "A Stable Diffusion checkpoint in diffusers layout (text encoder, U-Net, VAE, scheduler), admitted by the airlock.",
                     inputs: [], outputs: [model: :json], params: [path: {:string, ""}]}},
      {__MODULE__, %{type: "diffusion.text", version: 1, category: "diffusion", title: "Text conditioning",
                     doc: "CLIP's hidden rows for a prompt (ComfyUI's CLIPTextEncode).",
                     inputs: [model: :json], outputs: [conditioning: :tensor], params: [text: {:string, ""}]}},
      {__MODULE__, %{type: "diffusion.empty_latent", version: 1, category: "diffusion", title: "Empty latent",
                     doc: "Zero latents for a picture of width × height (ComfyUI's EmptyLatentImage); `factor` is the VAE's (8 for SD).",
                     inputs: [], outputs: [latent: :latent],
                     params: [width: {:int, 16, 4096, 512}, height: {:int, 16, 4096, 512}, channels: {:int, 1, 16, 4}, factor: {:int, 1, 16, 8}]}},
      {__MODULE__, %{type: "diffusion.sample", version: 1, category: "diffusion", title: "Sampler",
                     doc: "Denoise latents under the conditioning (ComfyUI's KSampler). `denoise` < 1 starts from the input latents (image to image); a mask repaints only where it is 1.",
                     inputs: [model: :json, positive: :tensor, negative: {:tensor, :optional}, latent: :latent, mask: {:mask, :optional}],
                     outputs: [latent: :latent], params: sampling ++ [denoise: {:float, 0.0, 1.0, 1.0}]}},
      {__MODULE__, %{type: "diffusion.decode", version: 1, category: "diffusion", title: "VAE decode",
                     doc: "Latents → image.", inputs: [model: :json, latent: :latent], outputs: [image: :image], params: []}},
      {__MODULE__, %{type: "diffusion.encode", version: 1, category: "diffusion", title: "VAE encode",
                     doc: "Image → latents (the encoder's mean: deterministic).", inputs: [model: :json, image: :image], outputs: [latent: :latent], params: []}},
      {__MODULE__, %{type: "diffusion.generate", version: 1, category: "diffusion", title: "Generate",
                     doc: "Prompt → image in one node; with an image, image to image (`strength`); with a mask too, inpainting.",
                     inputs: [model: :json, image: {:image, :optional}, mask: {:mask, :optional}], outputs: [image: :image, latent: :latent],
                     params: [prompt: {:string, ""}, negative: {:string, ""}, width: {:int, 0, 4096, 0}, height: {:int, 0, 4096, 0},
                              strength: {:float, 0.0, 1.0, 0.8}] ++ sampling}}
    ]
  end

  @impl true
  def run("diffusion.checkpoint", _, %{path: path}, ctx) do
    with {:ok, dir} <- resolve(path, ctx),
         {:ok, p} <- Pipeline.load(dir) do
      {:ok, %{model: %{"dir" => dir, "digest" => p.digest, "factor" => p.factor, "latent" => p.latent}}}
    end
  end

  def run("diffusion.text", %{model: m}, %{text: text}, ctx) do
    with {:ok, p, w} <- pipe(m, ctx), do: {:ok, %{conditioning: Pipeline.encode(p, text, w)}}
  end

  def run("diffusion.empty_latent", _, p, _) do
    {h, w} = {div(p.height, p.factor), div(p.width, p.factor)}
    {:ok, %{latent: Tensor.from_list(:f32, [p.channels, h, w], List.duplicate(0.0, p.channels * h * w))}}
  end

  def run("diffusion.sample", ins, prm, ctx) do
    with {:ok, p, w} <- pipe(ins.model, ctx),
         {:ok, {c, h, wd}} <- latent_shape(ins.latent, p) do
      x0 = Tensor.to_floats(ins.latent)
      blank? = Enum.all?(x0, &(&1 == 0.0))
      mask = with %{} = m <- ins[:mask], do: Pipeline.mask_latents(p, Vapor.Studio.Resample.resize(m, wd * p.factor, h * p.factor, "nearest"))
      init = if blank? and mask == nil, do: nil, else: x0

      {x, _} = Pipeline.sample(p, ins.positive, ins[:negative], worker: w, size: {h, wd}, steps: prm.steps, guidance: prm.cfg,
                               sampler: String.to_existing_atom(prm.sampler), seed: prm.seed, init_latents: init,
                               strength: prm.denoise, latent_mask: mask)

      {:ok, %{latent: Tensor.from_list(:f32, [c, h, wd], x)}}
    end
  end

  def run("diffusion.decode", %{model: m, latent: z}, _, ctx) do
    with {:ok, p, w} <- pipe(m, ctx), {:ok, {_, h, wd}} <- latent_shape(z, p) do
      {:ok, %{image: Pipeline.decode(p, Tensor.to_floats(z), {h, wd}, w)}}
    end
  end

  def run("diffusion.encode", %{model: m, image: img}, _, ctx) do
    with {:ok, p, w} <- pipe(m, ctx) do
      {h, wd} = {div(img.h, p.factor), div(img.w, p.factor)}
      img = if {img.w, img.h} == {wd * p.factor, h * p.factor}, do: img, else: Vapor.Studio.Nodes.Image.crop(img, 0, 0, wd * p.factor, h * p.factor)
      {:ok, %{latent: Tensor.from_list(:f32, [p.latent, h, wd], Pipeline.encode_image(img, p, w))}}
    end
  end

  def run("diffusion.generate", ins, prm, ctx) do
    with {:ok, p, w} <- pipe(ins.model, ctx) do
      size = for {k, v} <- [width: prm.width, height: prm.height], v > 0, do: {k, v}
      img = ins[:image]
      mask = ins[:mask]

      opts = [worker: w, prompt: prm.prompt, negative: prm.negative, guidance: prm.cfg, steps: prm.steps, seed: prm.seed,
              sampler: String.to_existing_atom(prm.sampler)] ++ size ++
             (if img, do: [init: img, strength: (if mask, do: 1.0, else: prm.strength)], else: []) ++ (if img && mask, do: [mask: mask], else: [])

      with {:ok, r} <- Pipeline.generate(p, opts) do
        {lh, lw} = {div(r.image.h, p.factor), div(r.image.w, p.factor)}
        {:ok, %{image: r.image, latent: Tensor.from_list(:f32, [p.latent, lh, lw], r.latents)}}
      end
    end
  end

  defp pipe(%{"dir" => dir}, ctx) do
    cond do
      ctx[:worker] == nil -> {:error, Rejection.new(:worker, "a native worker (diffusion runs compiled programs)", "start the studio with a worker")}
      true -> with {:ok, p} <- Pipeline.load(dir), do: {:ok, p, ctx.worker}
    end
  end

  defp pipe(_, _), do: {:error, Rejection.new(:model, "a checkpoint (diffusion.checkpoint)", "wire a checkpoint node")}

  defp latent_shape(%Tensor{shape: [c, h, w]}, %Pipeline{latent: c}), do: {:ok, {c, h, w}}
  defp latent_shape(%Tensor{shape: s}, p), do: {:error, Rejection.new(:latent, "latents [#{p.latent}, h, w]", "got #{inspect(s)}")}

  @doc "A checkpoint path: inside the studio's directory, or under `VAPOR_MODELS`; a diffusers directory."
  def resolve(path, ctx) do
    roots = [ctx[:dir] || ".", System.get_env("VAPOR_MODELS")] |> Enum.reject(&is_nil/1) |> Enum.map(&Path.expand/1)

    found =
      Enum.find_value(roots, fn root ->
        full = Path.expand(path, root)
        if (String.starts_with?(full, root <> "/") or full == root) and File.dir?(full), do: full
      end)

    cond do
      path == "" -> {:error, Rejection.new({:checkpoint, path}, "a checkpoint path", "set path")}
      String.ends_with?(path, [".safetensors", ".ckpt"]) ->
        {:error, Rejection.new({:checkpoint, path}, "a diffusers directory (unet/, vae/, text_encoder/, scheduler/)",
                               "convert the single file once: StableDiffusionPipeline.from_single_file(f).save_pretrained(dir)")}
      found == nil -> {:error, Rejection.new({:checkpoint, path}, "a directory inside the studio's directory or VAPOR_MODELS", "copy or link it there")}
      not File.dir?(Path.join(found, "unet")) -> {:error, Rejection.new({:checkpoint, path}, "a diffusers layout with unet/", "check the directory")}
      true -> {:ok, found}
    end
  end

  # ------------------------------------------------------------ ComfyUI --

  @doc "The ComfyUI classes translated to these nodes."
  def comfy_types, do: ~w(CheckpointLoaderSimple CLIPTextEncode EmptyLatentImage KSampler VAEDecode VAEEncode)

  @doc "A ComfyUI class's outputs by index."
  def comfy_outputs("CheckpointLoaderSimple"), do: ["model", "model", "model"]
  def comfy_outputs("CLIPTextEncode"), do: ["conditioning"]
  def comfy_outputs(t) when t in ["EmptyLatentImage", "KSampler", "VAEEncode"], do: ["latent"]
  def comfy_outputs("VAEDecode"), do: ["image"]
  def comfy_outputs(_), do: []

  @comfy_schedulers ~w(normal simple ddim_uniform sgm_uniform)
  @comfy_samplers %{"euler" => "euler", "ddim" => "ddim", "dpmpp_2m" => "dpmpp_2m"}

  @doc """
  One ComfyUI node as a studio node — translated by its documented
  meaning, not by reproducing ComfyUI: the sampler families are diffusers'
  (DDIM, Euler, DPM++ 2M over the checkpoint's own schedule), so a
  translated KSampler gives diffusers' image, not ComfyUI's pixels (its
  sigma spacing differs).
  """
  def from_comfy("CheckpointLoaderSimple", %{"ckpt_name" => name}),
    do: {:ok, %{"type" => "diffusion.checkpoint", "params" => %{"path" => name}}, "CheckpointLoaderSimple: #{name} must be a diffusers directory"}

  def from_comfy("CLIPTextEncode", %{"text" => t, "clip" => c}),
    do: {:ok, %{"type" => "diffusion.text", "params" => %{"text" => t}, "inputs" => %{"model" => link(c)}}, nil}

  def from_comfy("EmptyLatentImage", %{"width" => w, "height" => h} = i) do
    if Map.get(i, "batch_size", 1) != 1,
      do: {:error, "batch_size > 1"},
      else: {:ok, %{"type" => "diffusion.empty_latent", "params" => %{"width" => w, "height" => h}}, nil}
  end

  def from_comfy("KSampler", %{"model" => m, "positive" => pos, "negative" => neg, "latent_image" => l} = i) do
    sampler = @comfy_samplers[i["sampler_name"] || "euler"]
    sched = i["scheduler"] || "normal"

    cond do
      sampler == nil -> {:error, "sampler #{i["sampler_name"]}"}
      sched not in @comfy_schedulers -> {:error, "scheduler #{sched}"}
      true ->
        {:ok, %{"type" => "diffusion.sample",
                "params" => %{"seed" => rem(i["seed"] || 0, 4_294_967_296), "steps" => i["steps"] || 20, "cfg" => i["cfg"] || 8.0,
                              "sampler" => sampler, "denoise" => i["denoise"] || 1.0},
                "inputs" => %{"model" => link(m), "positive" => link(pos), "negative" => link(neg), "latent" => link(l)}},
         "KSampler #{sampler}/#{sched}: diffusers' #{sampler} over the checkpoint's schedule (not ComfyUI's sigmas); the noise is vapor's seeded generator"}
    end
  end

  def from_comfy("VAEDecode", %{"samples" => s, "vae" => v}),
    do: {:ok, %{"type" => "diffusion.decode", "inputs" => %{"model" => link(v), "latent" => link(s)}}, nil}

  def from_comfy("VAEEncode", %{"pixels" => px, "vae" => v}),
    do: {:ok, %{"type" => "diffusion.encode", "inputs" => %{"model" => link(v), "image" => link(px)}}, "VAEEncode takes the encoder's mean (ComfyUI's too)"}

  def from_comfy(t, _), do: {:error, "#{t}: missing inputs"}

  # a ComfyUI link [src, index] → [src, port] by the source's class (resolved by Vapor.Studio.Comfy)
  defp link([src, i]) when is_integer(i) do
    ports = Process.get({Vapor.Studio.Comfy, :ports}, %{})
    [to_string(src), Enum.at(Map.get(ports, to_string(src), []), i, "output#{i}")]
  end

  defp link(other), do: other
end
