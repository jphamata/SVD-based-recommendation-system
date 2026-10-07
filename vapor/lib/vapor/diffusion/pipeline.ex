defmodule Vapor.Diffusion.Pipeline do
  @moduledoc """
  **Stable Diffusion on vapor** — text to image, image to image and
  inpainting over a diffusers checkpoint directory, every network admitted
  by the model airlock and run as a certified program:

      text_encoder/   CLIPTextModel        → Vapor.Lock.Adapters.Encoder (clip_text)
      unet/           UNet2DConditionModel → Vapor.Lock.Adapters.UNet
      vae/            AutoencoderKL        → Vapor.Lock.Adapters.VAE (encoder and decoder)
      scheduler/      scheduler_config.json → Vapor.Diffusion.Scheduler (DDIM, Euler, DPM++ 2M)
      tokenizer/      tokenizer.json, or vocab.json + merges.txt (CLIP BPE)

  What runs where:

    * the three networks are programs (float32, the same bits on every
      substrate the worker runs them on), compiled once per shape and kept;
    * the sampler — the schedule, classifier-free guidance, the update, the
      latent blending of inpainting — is binary64 on the BEAM with
      correctly rounded `exp`/`log` (`Vapor.Diffusion.Scheduler`);
    * the starting noise comes from `Vapor.Modal.Rng` (splitmix64 +
      Box–Muller without libm): an image is a function of (weights, prompt,
      seed, steps, guidance, sampler), and anyone recomputes it.

  `test/vapor/diffusion_pipeline_test.exs` compares the whole chain with
  diffusers' own `StableDiffusionPipeline`, `…Img2ImgPipeline` and
  `…InpaintPipeline` on a tiny random checkpoint (the same ids, the same
  starting noise): the decoded images agree to float32 accuracy.

  Two deliberate differences from diffusers: the VAE encoder's latent is
  its distribution's **mean** (diffusers samples it: a second, hidden
  source of randomness), and inpainting is latent blending with a
  4-channel U-Net (a 9-channel inpainting U-Net is refused by the adapter).
  """
  alias Vapor.{Lock, Tensor, Tokenizer}
  alias Vapor.Lock.Adapters.{Encoder, UNet, VAE}
  alias Vapor.Diffusion.Scheduler
  alias Vapor.Modal.{Image, Rng}
  alias Vapor.Runtime.{Native, Substrates}

  defstruct [:dir, :unet, :vae, :text, :tokenizer, :max_len, :pad, :eos, :bos, :sched, :scaling, :factor, :latent, :cache, :digest]

  @doc """
  Open a diffusers Stable Diffusion directory (cached by path: a second
  `load` returns the same pipeline and its compiled programs).
  """
  def load(dir) do
    dir = Path.expand(dir)
    key = {__MODULE__, dir}

    case :persistent_term.get(key, nil) do
      %__MODULE__{} = p -> {:ok, p}
      nil -> with {:ok, p} <- open(dir), do: (:persistent_term.put(key, p); {:ok, p})
    end
  end

  defp open(dir) do
    with {:ok, u} <- Lock.open(Path.join(dir, "unet")),
         {:ok, v} <- Lock.open(Path.join(dir, "vae")),
         {:ok, t} <- Lock.open(Path.join(dir, "text_encoder")),
         {:ok, vc} <- json(Path.join([dir, "vae", "config.json"])) do
      sc = case json(Path.join([dir, "scheduler", "scheduler_config.json"])) do
        {:ok, m} -> m
        _ -> %{}
      end

      {tk, max, pad, bos, eos} = tokenizer(Path.join(dir, "tokenizer"), t)
      {:ok, cache} = Agent.start(fn -> %{} end)

      digest = :crypto.hash(:sha256, :erlang.term_to_binary({u.spec, v.spec, t.spec, sc})) |> Base.encode16(case: :lower)

      {:ok, %__MODULE__{dir: dir, unet: u, vae: v, text: t, tokenizer: tk, max_len: max, pad: pad, bos: bos, eos: eos,
                        sched: sc, scaling: (vc["scaling_factor"] || 0.18215) * 1.0, factor: VAE.factor(v.config),
                        latent: vc["latent_channels"] || 4, cache: cache, digest: digest}}
    end
  end

  defp json(path) do
    with {:ok, b} <- File.read(path), do: Vapor.JSON.decode(b)
  end

  # tokenizer.json when present; otherwise the slow CLIP files (vocab.json +
  # merges.txt) assembled into the same tokenizer transformers would write
  defp tokenizer(dir, text) do
    rows = text.config.rows
    cfg = case json(Path.join(dir, "tokenizer_config.json")) do
      {:ok, m} -> m
      _ -> %{}
    end

    max = min(rows, if(is_integer(cfg["model_max_length"]), do: cfg["model_max_length"], else: rows))

    tk =
      cond do
        File.regular?(Path.join(dir, "tokenizer.json")) -> elem(Tokenizer.load(Path.join(dir, "tokenizer.json")), 1)
        File.regular?(Path.join(dir, "vocab.json")) -> clip_tokenizer(dir)
        true -> nil
      end

    tok = fn name, default ->
      s = case cfg[name] do
        %{"content" => c} -> c
        c when is_binary(c) -> c
        _ -> default
      end

      if tk, do: List.first(Tokenizer.encode(tk, s, add_bos: false, add_eos: false)), else: nil
    end

    {tk, max, tok.("pad_token", "<|endoftext|>"), tok.("bos_token", "<|startoftext|>"), tok.("eos_token", "<|endoftext|>")}
  end

  @doc false
  def __tokenizer__(dir, text), do: tokenizer(dir, text)

  defp clip_tokenizer(dir) do
    {:ok, vocab} = json(Path.join(dir, "vocab.json"))
    merges = File.read!(Path.join(dir, "merges.txt")) |> String.split("\n", trim: true) |> Enum.reject(&String.starts_with?(&1, "#version")) |> Enum.map(&String.split(&1, " "))
    special = fn s -> %{"id" => vocab[s], "content" => s, "single_word" => false, "lstrip" => false, "rstrip" => false, "normalized" => false, "special" => true} end

    {:ok, tk} =
      Tokenizer.from_hf(%{
        "added_tokens" => [special.("<|startoftext|>"), special.("<|endoftext|>")],
        "normalizer" => %{"type" => "Sequence", "normalizers" => [%{"type" => "NFC"}, %{"type" => "Replace", "pattern" => %{"Regex" => "\\s+"}, "content" => " "}, %{"type" => "Lowercase"}]},
        "pre_tokenizer" => %{"type" => "Sequence", "pretokenizers" => [
          %{"type" => "Split", "pattern" => %{"Regex" => "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"}, "behavior" => "Removed", "invert" => true},
          %{"type" => "ByteLevel", "add_prefix_space" => false, "trim_offsets" => true, "use_regex" => true}]},
        "post_processor" => %{"type" => "RobertaProcessing", "sep" => ["<|endoftext|>", vocab["<|endoftext|>"]], "cls" => ["<|startoftext|>", vocab["<|startoftext|>"]], "trim_offsets" => false, "add_prefix_space" => false},
        "decoder" => %{"type" => "ByteLevel", "add_prefix_space" => true, "trim_offsets" => true, "use_regex" => true},
        "model" => %{"type" => "BPE", "dropout" => nil, "unk_token" => "<|endoftext|>", "continuing_subword_prefix" => "", "end_of_word_suffix" => "</w>",
                     "fuse_unk" => false, "byte_fallback" => false, "ignore_merges" => false, "vocab" => vocab, "merges" => merges}
      })

    tk
  end

  @doc """
  Token ids for a prompt, as diffusers prepares them: start and end of text,
  truncated to the model's length (the end-of-text token kept), padded with
  the tokenizer's pad token.
  """
  def ids(%__MODULE__{tokenizer: nil}, text) when is_binary(text), do: raise(ArgumentError, "this checkpoint has no tokenizer/: pass token ids")

  def ids(%__MODULE__{} = p, text) when is_binary(text) do
    ids = Tokenizer.encode(p.tokenizer, text)
    ids = if length(ids) > p.max_len, do: Enum.take(ids, p.max_len - 1) ++ [p.eos], else: ids
    pad(p, ids)
  end

  def ids(%__MODULE__{} = p, ids) when is_list(ids), do: pad(p, Enum.take(ids, p.text.config.rows))

  defp pad(p, ids), do: ids ++ List.duplicate(p.pad || p.eos || List.last(ids), p.text.config.rows - length(ids))

  @doc "The text encoder's hidden rows `[S, D]` for a prompt (text or ids)."
  def encode(%__MODULE__{} = p, prompt, w) do
    ids = ids(p, prompt)
    c = program(p, :text, fn -> Lock.build(p.text.spec, p.text.weights, []) end)
    r = run(w, c, Encoder.text_input(p.text.spec, ids))
    d = p.text.config.width
    %Tensor{shape: [s, dp]} = h = r.outputs.hidden
    if dp == d, do: h, else: Tensor.from_list(:f32, [s, d], h |> Tensor.to_floats() |> Enum.chunk_every(dp) |> Enum.flat_map(&Enum.take(&1, d)))
  end

  @doc """
  Generate an image. Options:

    * `prompt` (text or ids), `negative` (default the empty prompt),
      `guidance` (7.5; ≤ 1 runs the conditional branch only);
    * `sampler` `:ddim` | `:euler` | `:dpmpp_2m` (default `:dpmpp_2m`),
      `steps` (25), and the scheduler config of the checkpoint (spacing,
      offset, betas) unless overridden with `scheduler: [..]`;
    * `width`, `height` (the VAE's sample size by default; multiples of
      the VAE factor);
    * `seed` (0) or `noise` (the starting latents, a list or `[C, h, w]`);
    * `init` (a `Vapor.Modal.Image`) and `strength` (0.8) for image to image;
      with `mask` (one channel, 1 = repaint) for inpainting;
    * `worker` — the native worker the programs run on (required).

  Returns `{:ok, %{image: Image, latents: [float], timesteps: [..]}}`.
  """
  def generate(%__MODULE__{} = p, opts) do
    w = Keyword.fetch!(opts, :worker) || raise ArgumentError, "diffusion needs the native worker"
    init = Keyword.get(opts, :init)
    {iw, ih} = if init, do: {init.w, init.h}, else: {p.vae.config.sample, p.vae.config.sample}
    {wd, ht} = {Keyword.get(opts, :width, iw), Keyword.get(opts, :height, ih)}
    g = Keyword.get(opts, :guidance, 7.5) * 1.0
    cond_h = encode(p, Keyword.get(opts, :prompt, ""), w)
    unc_h = if g > 1.0, do: encode(p, Keyword.get(opts, :negative, ""), w)
    image = if init, do: init |> fit(wd, ht) |> encode_image(p, w)
    mask = with %Image{} = m <- Keyword.get(opts, :mask), do: m |> fit(wd, ht) |> latent_mask(p.factor, p.latent)
    {lh, lw} = {div(ht, p.factor), div(wd, p.factor)}
    default_strength = if mask, do: 1.0, else: 0.8

    opts = Keyword.merge(opts, init_latents: image, latent_mask: mask, size: {lh, lw},
                         strength: if(image, do: Keyword.get(opts, :strength, default_strength), else: 1.0))

    {x, ts} = sample(p, cond_h, unc_h, opts)
    # latents travel as float32 (diffusers', and the studio's latent wires)
    x = Tensor.from_list(:f32, [length(x)], x) |> Tensor.to_floats()
    {:ok, %{image: decode(p, x, {lh, lw}, w), latents: x, timesteps: ts}}
  end

  @doc """
  The sampling loop alone (ComfyUI's KSampler): conditioning rows `cond`
  and `uncond` (nil: no guidance) → final latents `[C·h·w]` and the
  timesteps run. Options as `generate/2`, plus `init_latents` (scaled
  latents to start from, with `strength` < 1), `latent_mask` (per latent
  element, 1 = repaint) and `size` `{h, w}` of the latents.
  """
  def sample(%__MODULE__{} = p, %Tensor{} = cond_h, unc_h, opts) do
    w = Keyword.fetch!(opts, :worker) || raise ArgumentError, "diffusion needs the native worker"
    {lh, lw} = Keyword.fetch!(opts, :size)
    n = p.latent * lh * lw
    kind = Keyword.get(opts, :sampler, :dpmpp_2m)
    steps = Keyword.get(opts, :steps, 25)
    s = Scheduler.new(kind, steps, sched_opts(p.sched) |> Keyword.merge(Keyword.get(opts, :scheduler, [])))
    g = Keyword.get(opts, :guidance, 7.5) * 1.0
    image = Keyword.get(opts, :init_latents)
    mask = Keyword.get(opts, :latent_mask)

    noise = case Keyword.get(opts, :noise) do
      nil -> Rng.normal(Keyword.get(opts, :seed, 0), n)
      %Tensor{} = t -> Tensor.to_floats(t)
      l when is_list(l) -> l
    end

    true = length(noise) == n
    strength = if image, do: Keyword.get(opts, :strength, 1.0) * 1.0, else: 1.0
    start = if strength >= 1.0, do: 0, else: max(steps - min(trunc(steps * strength), steps), 0)

    x =
      if strength >= 1.0 do
        Enum.map(noise, &(&1 * Scheduler.init_sigma(s)))
      else
        Scheduler.add_noise(s, image, noise, start)
      end

    unet = program(p, {:unet, lh, lw, cond_h.shape}, fn -> Lock.build(p.unet.spec, p.unet.weights, latent: {lh, lw}, context: hd(cond_h.shape)) end)

    eps_of = fn xi, t, h ->
      r = run(w, unet, UNet.input(p.unet.spec, Tensor.from_list(:f32, [p.latent, lh, lw], xi), t, h))
      UNet.output(p.unet.spec, r.outputs.out, {lh, lw}) |> Tensor.to_floats()
    end

    idx = Enum.to_list(start..(steps - 1)//1)

    {x, _} =
      Enum.reduce(idx, {x, %{}}, fn i, {x, st} ->
        t = Enum.at(s.timesteps, i)
        xi = Scheduler.scale(s, x, i)
        ec = eps_of.(xi, t, cond_h)
        eps = if unc_h && g > 1.0, do: Enum.zip_with(eps_of.(xi, t, unc_h), ec, fn u, c -> u + g * (c - u) end), else: ec
        {x2, st} = Scheduler.step(s, eps, x, i, st)

        x2 =
          if mask && image do
            # outside the mask: the source, noised to the next step's level
            known = if i < steps - 1, do: Scheduler.add_noise(s, image, noise, i + 1), else: image
            Enum.zip_with([x2, known, mask], fn [a, k, m] -> (1.0 - m) * k + m * a end)
          else
            x2
          end

        {x2, st}
      end)

    {x, Enum.map(idx, &Enum.at(s.timesteps, &1))}
  end

  @doc "A one-channel mask at image size → per-latent-element weights (torch's nearest, binarized at ½)."
  def mask_latents(%__MODULE__{} = p, %Image{} = m), do: latent_mask(m, p.factor, p.latent)

  defp sched_opts(c) do
    for {k, a} <- [{"num_train_timesteps", :num_train_timesteps}, {"beta_start", :beta_start}, {"beta_end", :beta_end},
                   {"beta_schedule", :beta_schedule}, {"timestep_spacing", :timestep_spacing}, {"steps_offset", :steps_offset},
                   {"set_alpha_to_one", :set_alpha_to_one}],
        Map.has_key?(c, k), do: {a, c[k]}
  end

  @doc "Latents `[C·h·w]` (scaled) → an image in [0, 1]."
  def decode(%__MODULE__{} = p, latents, {lh, lw}, w) do
    c = program(p, {:vae_dec, lh, lw}, fn -> Lock.build(p.vae.spec, p.vae.weights, latent: {lh, lw}) end)
    z = Tensor.from_list(:f32, [p.latent, lh, lw], Enum.map(latents, &(&1 / p.scaling)))
    r = run(w, c, VAE.input(p.vae.spec, z))
    t = VAE.image(p.vae.spec, r.outputs.out, {lh, lw})
    [3, h, wd] = t.shape
    v = t |> Tensor.to_floats() |> List.to_tuple()
    Image.new(wd, h, 3, for(y <- 0..(h - 1), x <- 0..(wd - 1), ch <- 0..2, do: min(1.0, max(0.0, elem(v, (ch * h + y) * wd + x) / 2 + 0.5))))
  end

  @doc "An image in [0, 1] → its latents (the encoder's mean, scaled)."
  def encode_image(%Image{w: wd, h: h} = img, %__MODULE__{} = p, w) do
    img = if img.c == 3, do: img, else: Image.new(wd, h, 3, img |> Image.values() |> Enum.flat_map(&[&1, &1, &1]))
    c = program(p, {:vae_enc, h, wd}, fn -> Lock.build(p.vae.spec, p.vae.weights, part: :encoder, image: {h, wd}) end)
    px = img.px
    t = Tensor.from_list(:f32, [3, h, wd], for(ch <- 0..2, y <- 0..(h - 1), x <- 0..(wd - 1), do: elem(px, (y * wd + x) * 3 + ch) * 2.0 - 1.0))
    r = run(w, c, VAE.encode_input(p.vae.spec, t))
    VAE.latents(p.vae.spec, r.outputs.out, {div(h, p.factor), div(wd, p.factor)}) |> Tensor.to_floats() |> Enum.map(&(&1 * p.scaling))
  end

  # torch's nearest interpolation to the latent size, binarized at ½ (diffusers' mask processor), one plane per channel
  defp latent_mask(%Image{w: wd, h: h} = m, f, ch) do
    plane = for y <- 0..(div(h, f) - 1), x <- 0..(div(wd, f) - 1), do: (if Image.at(m, x * f, y * f, 0) >= 0.5, do: 1.0, else: 0.0)
    List.duplicate(plane, ch) |> List.flatten()
  end

  defp fit(%Image{w: w, h: h} = img, w, h), do: img
  defp fit(img, w, h), do: Vapor.Studio.Resample.resize(img, w, h, "bicubic", antialias: true)

  defp program(%__MODULE__{cache: a}, key, build) do
    case Agent.get(a, &Map.get(&1, key)) do
      nil ->
        {:ok, prog} = build.()
        {:ok, c} = Vapor.Compile.Lower.lower(prog)
        Agent.update(a, &Map.put(&1, key, c))
        c

      c ->
        c
    end
  end

  defp run(w, c, env) do
    {:ok, r} = Native.run(w, c, env, isa: Substrates.host_isa(), mode: :native)
    r
  end
end
