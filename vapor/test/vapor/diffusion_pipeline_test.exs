defmodule Vapor.DiffusionPipelineTest do
  @moduledoc """
  Stable Diffusion on vapor (`Vapor.Diffusion.Pipeline`), piece by piece and
  whole, against diffusers itself (random tiny weights — this machine
  downloads none):

    * the schedulers: DDIM, Euler and DPM-Solver++ 2M, each with leading,
      linspace and trailing spacing, 10 and 25 steps — the same timesteps
      and the same final latents (to float32: diffusers steps in float32);
    * the U-Net (`UNet2DConditionModel`, SD 1.x-like and SD 2.x-like) and
      the VAE encoder: one forward pass, relative error ~1e-6;
    * the pipelines: text to image under each sampler, image to image,
      inpainting — the decoded image to ~1e-6; the **control**: the same
      run under another sampler is off by orders of magnitude more, so the
      comparison discriminates;
    * the CLIP tokenizer assembled from the slow files (vocab.json +
      merges.txt, what SD checkpoints ship) gives transformers' ids;
    * an image is a function of the seed: the same seed, the same bits.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Lock, Tensor}
  alias Vapor.Diffusion.{Pipeline, Scheduler}
  alias Vapor.Lock.Adapters.UNet
  alias Vapor.Modal.Image
  import Vapor.TestHelpers

  @moduletag :diffusers
  @moduletag :native
  @moduletag timeout: 900_000

  setup_all do
    {:ok, w} = Vapor.Runtime.Worker.start_link(exec: worker_exec(:host))
    tmp = Path.join(System.tmp_dir!(), "vapor-sd-#{System.unique_integer([:positive])}")
    File.mkdir_p!(tmp)
    on_exit(fn -> File.rm_rf!(tmp) end)
    {:ok, worker: w, tmp: tmp}
  end

  defp script(name, args), do: System.cmd(python(), [Path.expand("../python/#{name}", __DIR__) | args], stderr_to_stdout: false)
  defp rel(a, b), do: (Enum.zip_with(a, b, &abs(&1 - &2)) |> Enum.max()) / (b |> Enum.map(&abs/1) |> Enum.max())

  test "the schedulers take diffusers' timesteps and reach diffusers' latents" do
    for kind <- [:ddim, :euler, :dpmpp_2m], spacing <- ~w(leading linspace trailing), steps <- [10, 25] do
      {out, 0} = script("diffusers_scheduler.py", [Atom.to_string(kind), spacing, "#{steps}"])
      ref = Vapor.JSON.decode!(out |> String.split("\n", trim: true) |> List.last())
      s = Scheduler.new(kind, steps, timestep_spacing: spacing)
      assert Enum.map(s.timesteps, &(&1 * 1.0)) == ref["timesteps"], "#{kind} #{spacing} #{steps}"
      assert abs(Scheduler.init_sigma(s) - ref["init_sigma"]) / ref["init_sigma"] < 1.0e-6
      x0 = for i <- 0..15, do: (rem(i * 37, 17) / 8.0 - 1.0) * Scheduler.init_sigma(s)

      {x, _} =
        Enum.reduce(Enum.with_index(s.timesteps), {x0, %{}}, fn {t, i}, {x, st} ->
          Scheduler.step(s, Enum.map(Scheduler.scale(s, x, i), &(0.3 * &1 + t / 4000.0)), x, i, st)
        end)

      err = Enum.zip(x, ref["x"]) |> Enum.map(fn {a, b} -> abs(a - b) / max(1.0, abs(b)) end) |> Enum.max()
      assert err < 2.0e-6, "#{kind} #{spacing} #{steps}: #{err}"
    end
  end

  test "the U-Net (SD 1.x and 2.x shapes) and the VAE encoder are diffusers' to ~1e-6", %{worker: w, tmp: tmp} do
    for v <- ~w(sd1 sd2) do
      dir = Path.join(tmp, "unet_#{v}")
      {_, 0} = script("diffusers_unet.py", [dir, "7", v])
      {:ok, ref} = Vapor.Ingest.Safetensors.read(Path.join(dir, "reference.safetensors"))
      {:ok, m} = Lock.open(dir)
      [_, h, ww] = ref["z"].shape
      {:ok, p} = Lock.build(m.spec, m.weights, latent: {h, ww}, context: hd(ref["ctx"].shape))
      {:ok, c} = Vapor.Compile.Lower.lower(p)
      [t] = Tensor.to_floats(ref["t"])
      {:ok, r} = Vapor.Runtime.Native.run(w, c, UNet.input(m.spec, ref["z"], t, ref["ctx"]), isa: Vapor.Runtime.Substrates.host_isa(), mode: :native)
      assert rel(Tensor.to_floats(UNet.output(m.spec, r.outputs.out, {h, ww})), Tensor.to_floats(ref["eps"])) < 1.0e-5, v
    end

    dir = Path.join(tmp, "sd")
    {_, 0} = script("diffusers_pipeline.py", [dir])
    {:ok, p} = Pipeline.load(dir)
    {:ok, ref} = Vapor.Ingest.Safetensors.read(Path.join(dir, "reference.safetensors"))
    init = img(ref["init"])
    # the encoder's mean, through diffusers' own VAE
    out = py!("""
    import sys, torch, json
    from diffusers import AutoencoderKL
    from safetensors.torch import load_file
    v = AutoencoderKL.from_pretrained(sys.argv[1] + '/vae')
    x = load_file(sys.argv[1] + '/reference.safetensors')['init'].permute(2, 0, 1)[None] * 2 - 1
    with torch.no_grad():
        print(json.dumps((v.encode(x).latent_dist.mean * v.config.scaling_factor).flatten().tolist()))
    """, [dir])
    assert rel(Pipeline.encode_image(init, p, w), Vapor.JSON.decode!(out)) < 1.0e-5
  end

  test "text to image, image to image and inpainting are diffusers' pipelines; another sampler is not", %{worker: w, tmp: tmp} do
    dir = Path.join(tmp, "sd")
    unless File.exists?(Path.join(dir, "reference.safetensors")), do: {_, 0} = script("diffusers_pipeline.py", [dir])
    {:ok, p} = Pipeline.load(dir)
    {:ok, ref} = Vapor.Ingest.Safetensors.read(Path.join(dir, "reference.safetensors"))
    ids = fn k -> ref[k] |> Tensor.to_floats() |> Enum.map(&trunc/1) end
    common = [prompt: ids.("ids"), negative: ids.("neg"), guidance: 3.0, worker: w]
    err = fn name, %Image{} = a -> Enum.zip_with(Image.values(a), Tensor.to_floats(ref[name]), &abs(&1 - &2)) |> Enum.max() end

    runs =
      for k <- [:ddim, :euler, :dpmpp_2m] do
        {:ok, r} = Pipeline.generate(p, common ++ [sampler: k, steps: 6, noise: ref["latents"], width: 16, height: 16])
        assert err.("txt2img_#{k}", r.image) < 1.0e-5, "#{k}"
        {k, r.image}
      end
      |> Map.new()

    # the control: DPM++ against the DDIM reference
    assert err.("txt2img_ddim", runs.dpmpp_2m) > 1.0e-3

    init = img(ref["init"])
    {:ok, r} = Pipeline.generate(p, common ++ [sampler: :ddim, steps: 10, noise: ref["noise"], init: init, strength: 0.6])
    assert err.("img2img", r.image) < 1.0e-5
    [h, wd] = ref["mask"].shape
    mask = Image.new(wd, h, 1, Tensor.to_floats(ref["mask"]))
    {:ok, r} = Pipeline.generate(p, common ++ [sampler: :ddim, steps: 10, noise: ref["latents"], init: init, mask: mask])
    assert err.("inpaint", r.image) < 1.0e-5

    # the seed is the image
    {:ok, a} = Pipeline.generate(p, common ++ [steps: 4, seed: 9])
    {:ok, b} = Pipeline.generate(p, common ++ [steps: 4, seed: 9])
    {:ok, c} = Pipeline.generate(p, common ++ [steps: 4, seed: 10])
    assert a.image == b.image and a.image != c.image
  end

  test "the CLIP tokenizer from vocab.json + merges.txt gives transformers' ids", %{tmp: tmp} do
    dir = Path.join(tmp, "sd_tk")
    File.mkdir_p!(Path.join(dir, "tokenizer"))
    lines = ["a photo of an astronaut riding a horse", "A CAT, isn't it?  42 ✓", "", "ab abab ba"]
    out = py!("""
    import json, sys
    d = sys.argv[1] + '/tokenizer'
    chars = list('abcdefghijklmnopqrstuvwxyz0123456789,.?!\\'') + ['âľĵ', 'âľ', 'ĵ', 'Ģ']
    toks = []
    for c in chars:
        toks += [c, c + '</w>']
    merges = ['a b', 'a b</w>', 'ab ab', 'ab ab</w>', 'p h', 'h o', 'o r', 's e', 'a t</w>', 'i t</w>']
    toks += [m.replace(' ', '') for m in merges] + ['<|startoftext|>', '<|endoftext|>']
    json.dump({t: i for i, t in enumerate(dict.fromkeys(toks))}, open(d + '/vocab.json', 'w'), ensure_ascii=False)
    open(d + '/merges.txt', 'w').write('#version: 0.2\\n' + '\\n'.join(merges) + '\\n')
    json.dump({'model_max_length': 16, 'pad_token': '<|endoftext|>', 'bos_token': '<|startoftext|>', 'eos_token': '<|endoftext|>'}, open(d + '/tokenizer_config.json', 'w'))
    from transformers import CLIPTokenizer
    tk = CLIPTokenizer(d + '/vocab.json', d + '/merges.txt')
    print(json.dumps([tk(l, padding='max_length', max_length=16, truncation=True).input_ids for l in json.loads(sys.argv[2])]))
    """, [dir, Vapor.JSON.encode(lines)])

    ref = Vapor.JSON.decode!(out)
    # a pipeline whose only real part is the tokenizer: the ids are compared before any network runs
    text = %{config: %{rows: 16}}
    {tk, max, pad, bos, eos} = Pipeline.__tokenizer__(Path.join(dir, "tokenizer"), text)
    p = %Pipeline{text: text, tokenizer: tk, max_len: max, pad: pad, bos: bos, eos: eos}
    for {l, r} <- Enum.zip(lines, ref), do: assert(Pipeline.ids(p, l) == r, l)
  end

  test "a ComfyUI txt2img workflow imports, runs in the studio, equals the pipeline and verifies", %{worker: w, tmp: tmp} do
    dir = Path.join(tmp, "sd")
    unless File.exists?(Path.join(dir, "reference.safetensors")), do: {_, 0} = script("diffusers_pipeline.py", [dir])

    comfy = %{
      "4" => %{"class_type" => "CheckpointLoaderSimple", "inputs" => %{"ckpt_name" => "sd"}},
      "6" => %{"class_type" => "CLIPTextEncode", "inputs" => %{"text" => "a cat on a dog", "clip" => ["4", 1]}},
      "7" => %{"class_type" => "CLIPTextEncode", "inputs" => %{"text" => "", "clip" => ["4", 1]}},
      "5" => %{"class_type" => "EmptyLatentImage", "inputs" => %{"width" => 16, "height" => 16, "batch_size" => 1}},
      "3" => %{"class_type" => "KSampler", "inputs" => %{"seed" => 5, "steps" => 6, "cfg" => 3.0, "sampler_name" => "euler", "scheduler" => "normal",
                                                          "denoise" => 1.0, "model" => ["4", 0], "positive" => ["6", 0], "negative" => ["7", 0], "latent_image" => ["5", 0]}},
      "8" => %{"class_type" => "VAEDecode", "inputs" => %{"samples" => ["3", 0], "vae" => ["4", 2]}},
      "9" => %{"class_type" => "SaveImage", "inputs" => %{"filename_prefix" => "out", "images" => ["8", 0]}}
    }

    {:ok, g, notes} = Vapor.Studio.Comfy.import(comfy)
    assert Enum.any?(notes, &(&1 =~ "not ComfyUI's sigmas"))
    g = put_in(g, ["nodes", "5", "params", "factor"], 2)
    {:ok, cache} = Vapor.Studio.Cache.start_link()
    {:ok, r} = Vapor.Studio.run(g, worker: w, dir: tmp, cache: cache)
    {:ok, p} = Pipeline.load(dir)
    {:ok, ref} = Pipeline.generate(p, prompt: "a cat on a dog", negative: "", guidance: 3.0, steps: 6, sampler: :euler, seed: 5, worker: w)
    assert r.outputs["9"]["value"] == ref.image
    {:ok, r2} = Vapor.Studio.run(g, worker: w, dir: tmp, cache: cache)
    assert r2.executed == [] and r2.root == r.root
    assert :ok == Vapor.Studio.verify(g, r.root, worker: w, dir: tmp)
    # a single-file checkpoint is refused with the conversion to do
    {:error, %Vapor.Rejection{} = why} = Vapor.Studio.Nodes.Diffusion.resolve("model.safetensors", %{dir: tmp})
    assert inspect(why) =~ "from_single_file"
  end

  defp img(t), do: (fn [h, wd, 3] -> Image.new(wd, h, 3, Tensor.to_floats(t)) end).(t.shape)
end
