defmodule Vapor.Lock.Adapters.VAE do
  @moduledoc """
  Tier 3 of the model airlock: the **continuous latent decoder** of latent
  diffusion — diffusers' `AutoencoderKL` (the VAE of Stable Diffusion 1.x/2.x,
  SDXL, and, with `latent_channels: 16`, the shape Flux and SD3 use) —
  latents in, pixels out, built from `Vapor.Spatial` (convolution by
  gather + reshape + GEMV, GroupNorm by exact selector contractions, nearest
  upsampling, pixel attention by the encoder's horizon trick): no new
  operator, the same kernels and certificates as the language models.

      z → post_quant_conv (1×1) → conv_in (3×3)
        → mid: resnet, self-attention over the pixels, resnet
        → up blocks (top channels first): (layers_per_block + 1) resnets,
          then nearest ×2 + conv, except the last
        → GroupNorm → SiLU → conv_out (3×3) → pixels in about [−1, 1]

  a resnet being `x + conv₂(silu(gn₂(conv₁(silu(gn₁(x))))))` (a 1×1
  `conv_shortcut` when the channel count changes).

  Configurations: diffusers' `config.json` with `"_class_name":
  "AutoencoderKL"` (`block_out_channels`, `layers_per_block`,
  `latent_channels`, `norm_num_groups`, `act_fn: "silu"`; refused, by
  name: other activations, `mid_block_add_attention: false` is honoured,
  `use_post_quant_conv: false` too). The **encoder** (since 0.9, for
  img2img and inpainting) is built with `part: :encoder`: pixels → the
  mean of the latent distribution (`latent_dist.mean`), its downsamplers
  padded right and bottom as diffusers does; `encode_input/2` and
  `latents/3` convert.

  Contract `:map`: `rows : f32[h·w, pad16(latent_channels)]` (latents,
  already divided by `scaling_factor` and shifted back, as diffusers'
  pipelines do before `decode`) → `out : f32[(f·h)·(f·w), 16]`, the first
  three columns RGB. Build option `latent: {h, w}` (default: `sample_size`
  divided by the upsampling factor); `h·w` must be a multiple of 16.
  `input/2` and `image/2` convert from and to diffusers' `NCHW`.
  """
  @behaviour Vapor.Lock.Adapter
  alias Vapor.{Program, Rejection, Spatial, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Lock.Spec

  defmodule Config do
    @moduledoc "An admitted AutoencoderKL decoder."
    defstruct [:blocks, :layers, :latent, :groups, :eps, :attention, :post_quant, :sample, :scaling, :shift, :raw]
  end

  @impl true
  def id, do: "vae"

  @impl true
  def claim(%{config: %{"_class_name" => "AutoencoderKL"}}), do: {:claim, 100}

  def claim(%{config: %{"_class_name" => c}}) when c in ["AutoencoderTiny", "AutoencoderKLWan", "AutoencoderKLHunyuanVideo", "AutoencoderKLCogVideoX"],
    do: {:near, "#{c} is a different autoencoder topology (tiny or causal-3D): not this adapter"}

  def claim(_), do: :no

  @impl true
  def owns?(%Config{}), do: true
  def owns?(_), do: false

  @impl true
  def admit(%{config: c}, ws, _opts) do
    blocks = c["block_out_channels"] || [64]

    cond do
      not (is_list(blocks) and blocks != [] and Enum.all?(blocks, &(is_integer(&1) and &1 > 0))) -> no("block_out_channels", "a list of positive integers")
      (c["act_fn"] || "silu") != "silu" -> no("act_fn", "silu (got #{inspect(c["act_fn"])})")
      Enum.any?(c["up_block_types"] || [], &(&1 != "UpDecoderBlock2D")) -> no("up_block_types", "UpDecoderBlock2D")
      Enum.any?(blocks, &(rem(&1, c["norm_num_groups"] || 32) != 0)) -> no("norm_num_groups", "a divisor of every block width")
      true ->
        cfg = %Config{blocks: blocks, layers: c["layers_per_block"] || 1, latent: c["latent_channels"] || 4,
                      groups: c["norm_num_groups"] || 32, eps: 1.0e-6, attention: c["mid_block_add_attention"] != false,
                      post_quant: c["use_post_quant_conv"] != false, sample: c["sample_size"] || 32,
                      scaling: c["scaling_factor"] || 0.18215, shift: c["shift_factor"], raw: c}

        spec = spec(cfg)

        case Enum.find(expected(spec), fn {n, shape, _} -> not match?(%Tensor{shape: ^shape}, ws[n]) end) do
          nil -> {:ok, spec, ws}
          {n, shape, _} -> {:error, Rejection.new({:weight, n}, "#{inspect(shape, charlists: :as_lists)}", "check the checkpoint against config.json")}
        end
    end
  end

  defp no(f, b), do: {:error, Rejection.new({:config, f}, b, "use a supported checkpoint")}

  @impl true
  def spec(%Config{} = c) do
    %Spec{adapter: __MODULE__, family: "autoencoder_kl", lineage: ["autoencoder_kl", "vae"], interface: :map, config: c,
          width: 16, in_width: Spatial.pad16(c.latent), features: [:decode, :encode], parts: %{encoder: :map},
          modality: %{in: [:latent], out: [:image]}, digest: Vapor.Canonical.hex_digest({:vae, c.raw})}
  end

  @doc "The upsampling factor between latents and pixels: 2^(blocks − 1)."
  def factor(%Config{blocks: b}), do: Integer.pow(2, length(b) - 1)

  # ---------------------------------------------------------- tensor names --

  @impl true
  def expected(%Spec{config: c}) do
    top = List.last(c.blocks)
    rev = Enum.reverse(c.blocks)
    conv = fn n, o, i, k -> [{n <> ".weight", [o, i, k, k], :matrix}, {n <> ".bias", [o], :bias}] end
    gn = fn n, ch -> [{n <> ".weight", [ch], :norm}, {n <> ".bias", [ch], :bias}] end

    resnet = fn p, i, o ->
      gn.(p <> ".norm1", i) ++ conv.(p <> ".conv1", o, i, 3) ++ gn.(p <> ".norm2", o) ++ conv.(p <> ".conv2", o, o, 3) ++
        if(i != o, do: conv.(p <> ".conv_shortcut", o, i, 1), else: [])
    end

    pq = if c.post_quant, do: conv.("post_quant_conv", c.latent, c.latent, 1), else: []

    attn =
      if c.attention do
        a = "decoder.mid_block.attentions.0."
        gn.(a <> "group_norm", top) ++
          Enum.flat_map(~w(to_q to_k to_v to_out.0), &[{a <> &1 <> ".weight", [top, top], :matrix}, {a <> &1 <> ".bias", [top], :bias}])
      else
        []
      end

    {ups, _} =
      Enum.flat_map_reduce(Enum.with_index(rev), top, fn {o, bi}, prev ->
        rs = for j <- 0..c.layers, do: resnet.("decoder.up_blocks.#{bi}.resnets.#{j}", if(j == 0, do: prev, else: o), o)
        up = if bi < length(rev) - 1, do: conv.("decoder.up_blocks.#{bi}.upsamplers.0.conv", o, o, 3), else: []
        {List.flatten(rs) ++ up, o}
      end)

    pq ++ conv.("decoder.conv_in", top, c.latent, 3) ++ resnet.("decoder.mid_block.resnets.0", top, top) ++ attn ++
      resnet.("decoder.mid_block.resnets.1", top, top) ++ ups ++ gn.("decoder.conv_norm_out", hd(c.blocks)) ++
      conv.("decoder.conv_out", 3, hd(c.blocks), 3)
  end

  # ----------------------------------------------------------------- build --

  @impl true
  def build(%Spec{} = spec, ws, opts) do
    if Keyword.get(opts, :part) == :encoder, do: build_encoder(spec, ws, opts), else: build_decoder(spec, ws, opts)
  end

  defp build_decoder(%Spec{config: c}, ws, opts) do
    f = factor(c)
    {h, w} = Keyword.get(opts, :latent, {div(c.sample, f), div(c.sample, f)})

    if rem(h * w, 16) != 0 do
      {:error, Rejection.new({:vae, :latent}, "h·w a multiple of 16 (got #{h}×#{w})", "choose a latent of 4×4, 8×8, 64×64…")}
    else
      x = T.input(:rows, :f32, [h * w, Spatial.pad16(c.latent)])
      lets = []
      d = {h, w, c.latent}
      conv = fn lets, x, d, n, k -> Spatial.conv2d(lets, x, d, n, ws[n <> ".weight"], ws[n <> ".bias"], padding: div(k - 1, 2)) end

      {lets, x, d} = if c.post_quant, do: conv.(lets, x, d, "post_quant_conv", 1), else: {lets, x, d}
      {lets, x, d} = conv.(lets, x, d, "decoder.conv_in", 3)
      {lets, x, d} = resnet(lets, x, d, "decoder.mid_block.resnets.0", ws, c)

      {lets, x} =
        if c.attention do
          a = "decoder.mid_block.attentions.0."
          {hh, ww, ch} = d
          {lets, hn} = Spatial.group_norm(lets, x, hh * ww, ch, c.groups, a <> "group_norm", ws[a <> "group_norm.weight"], ws[a <> "group_norm.bias"], c.eps)
          p = &{ws[a <> &1 <> ".weight"], ws[a <> &1 <> ".bias"]}
          {lets, at} = Spatial.pixel_attention(lets, hn, hh * ww, ch, a <> "attn", p.("to_q"), p.("to_k"), p.("to_v"), p.("to_out.0"))
          Spatial.name(lets, a <> "out", T.add(x, at))
        else
          {lets, x}
        end

      {lets, x, d} = resnet(lets, x, d, "decoder.mid_block.resnets.1", ws, c)
      n_up = length(c.blocks)

      {lets, x, d} =
        Enum.reduce(0..(n_up - 1), {lets, x, d}, fn bi, {lets, x, d} ->
          {lets, x, d} = Enum.reduce(0..c.layers, {lets, x, d}, fn j, {lets, x, d} -> resnet(lets, x, d, "decoder.up_blocks.#{bi}.resnets.#{j}", ws, c) end)

          if bi < n_up - 1 do
            {lets, x, d} = Spatial.upsample_nearest(lets, x, d, "decoder.up_blocks.#{bi}.upsampled")
            conv.(lets, x, d, "decoder.up_blocks.#{bi}.upsamplers.0.conv", 3)
          else
            {lets, x, d}
          end
        end)

      {hh, ww, ch} = d
      {lets, x} = Spatial.group_norm(lets, x, hh * ww, ch, c.groups, "decoder.conv_norm_out", ws["decoder.conv_norm_out.weight"], ws["decoder.conv_norm_out.bias"], c.eps)
      {lets, out, _} = conv.(lets, T.silu(x), d, "decoder.conv_out", 3)
      {:ok, Program.new([out: out], lets: Enum.reverse(lets))}
    end
  end

  # pixels [H·W, 16] (RGB in about [−1, 1]) → latent means [h·w, pad16(latent)]
  defp build_encoder(%Spec{config: c}, ws, opts) do
    f = factor(c)
    {hh, ww} = Keyword.get(opts, :image, {c.sample, c.sample})
    {h, w} = {div(hh, f), div(ww, f)}

    cond do
      rem(hh, f) != 0 or rem(ww, f) != 0 -> {:error, Rejection.new({:vae, :image}, "sides divisible by #{f}", "resize the image")}
      rem(h * w, 16) != 0 -> {:error, Rejection.new({:vae, :image}, "a latent of a multiple of 16 pixels", "use sides multiple of #{4 * f}")}
      not Map.has_key?(ws, "encoder.conv_in.weight") -> {:error, Rejection.new({:vae, :encoder}, "the encoder's weights", "use a full AutoencoderKL checkpoint")}
      true ->
        x = T.input(:rows, :f32, [hh * ww, 16])
        conv = fn lets, x, d, n, k, o -> Spatial.conv2d(lets, x, d, n, ws[n <> ".weight"], ws[n <> ".bias"], Keyword.merge([padding: div(k - 1, 2)], o)) end
        lets = []
        {lets, x, d} = conv.(lets, x, {hh, ww, 3}, "encoder.conv_in", 3, [])
        n_down = length(c.blocks)

        {lets, x, d} =
          Enum.reduce(0..(n_down - 1), {lets, x, d}, fn bi, {lets, x, d} ->
            {lets, x, d} = Enum.reduce(0..(c.layers - 1), {lets, x, d}, fn j, {lets, x, d} -> resnet(lets, x, d, "encoder.down_blocks.#{bi}.resnets.#{j}", ws, c) end)
            if bi < n_down - 1,
              do: conv.(lets, x, d, "encoder.down_blocks.#{bi}.downsamplers.0.conv", 3, stride: 2, pads: {0, 1, 0, 1}),
              else: {lets, x, d}
          end)

        {lets, x, d} = resnet(lets, x, d, "encoder.mid_block.resnets.0", ws, c)

        {lets, x} =
          if c.attention do
            a = "encoder.mid_block.attentions.0."
            {mh, mw, ch} = d
            {lets, hn} = Spatial.group_norm(lets, x, mh * mw, ch, c.groups, a <> "group_norm", ws[a <> "group_norm.weight"], ws[a <> "group_norm.bias"], c.eps)
            p = &{ws[a <> &1 <> ".weight"], ws[a <> &1 <> ".bias"]}
            {lets, at} = Spatial.pixel_attention(lets, hn, mh * mw, ch, a <> "attn", p.("to_q"), p.("to_k"), p.("to_v"), p.("to_out.0"))
            Spatial.name(lets, a <> "out", T.add(x, at))
          else
            {lets, x}
          end

        {lets, x, d} = resnet(lets, x, d, "encoder.mid_block.resnets.1", ws, c)
        {mh, mw, ch} = d
        {lets, x} = Spatial.group_norm(lets, x, mh * mw, ch, c.groups, "encoder.conv_norm_out", ws["encoder.conv_norm_out.weight"], ws["encoder.conv_norm_out.bias"], c.eps)
        {lets, x, d} = conv.(lets, T.silu(x), d, "encoder.conv_out", 3, [])
        {lets, x, _} = if ws["quant_conv.weight"], do: conv.(lets, x, d, "quant_conv", 1, []), else: {lets, x, d}
        # the mean: the first `latent` of the 2·latent moments (an exact selection)
        l = c.latent
        {mp, lp} = {Spatial.pad16(2 * l), Spatial.pad16(l)}
        sel = Tensor.from_list(:f32, [lp, mp], for(o <- 0..(lp - 1), i <- 0..(mp - 1), do: if(o < l and o == i, do: 1.0, else: 0.0)))
        {lets, sr} = Spatial.bind(lets, "$vae.mean", sel)
        {:ok, Program.new([out: T.linear(x, sr)], lets: Enum.reverse(lets))}
    end
  end

  @doc "diffusers' image `[3, H, W]` (values in about [−1, 1]) as the encoder's environment."
  def encode_input(%Spec{config: %Config{}}, %Tensor{} = img) do
    {rows, _} = Spatial.from_nchw(img)
    %{rows: rows}
  end

  @doc "The encoder's output as diffusers' `[latent, h, w]` (the distribution's mean, not yet scaled)."
  def latents(%Spec{config: c}, %Tensor{} = mean, {h, w}), do: Spatial.to_nchw(mean, {h, w, c.latent})

  defp resnet(lets, x, {hh, ww, i} = d, p, ws, c) do
    o = hd(ws[p <> ".conv1.weight"].shape)
    n = hh * ww
    {lets, h1} = Spatial.group_norm(lets, x, n, i, c.groups, p <> ".norm1", ws[p <> ".norm1.weight"], ws[p <> ".norm1.bias"], c.eps)
    {lets, h1, d1} = Spatial.conv2d(lets, T.silu(h1), d, p <> ".conv1", ws[p <> ".conv1.weight"], ws[p <> ".conv1.bias"], padding: 1)
    {lets, h2} = Spatial.group_norm(lets, h1, n, o, c.groups, p <> ".norm2", ws[p <> ".norm2.weight"], ws[p <> ".norm2.bias"], c.eps)
    {lets, h2, _} = Spatial.conv2d(lets, T.silu(h2), d1, p <> ".conv2", ws[p <> ".conv2.weight"], ws[p <> ".conv2.bias"], padding: 1)

    {lets, skip} =
      if i != o do
        {lets, s, _} = Spatial.conv2d(lets, x, d, p <> ".conv_shortcut", ws[p <> ".conv_shortcut.weight"], ws[p <> ".conv_shortcut.bias"])
        {lets, s}
      else
        {lets, x}
      end

    {lets, y} = Spatial.name(lets, p <> ".out", T.add(skip, h2))
    {lets, y, {hh, ww, o}}
  end

  # ---------------------------------------------------------------- helpers --

  @doc "diffusers' latents `[C, h, w]` (or `[1, C, h, w]`) as the program's environment."
  def input(%Spec{config: %Config{}}, %Tensor{} = z) do
    {rows, _} = Spatial.from_nchw(z)
    %{rows: rows}
  end

  @doc "The program's output as an image `[3, H, W]` (diffusers' layout, values about [−1, 1])."
  def image(%Spec{config: c}, %Tensor{shape: [n, 16]} = out, {h, w} \\ nil) do
    f = factor(c)
    side = round(:math.sqrt(n))
    {hh, ww} = if h, do: {h * f, w * f}, else: {side, div(n, side)}
    Spatial.to_nchw(out, {hh, ww, 3})
  end
end
