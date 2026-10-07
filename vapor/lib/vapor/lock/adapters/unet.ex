defmodule Vapor.Lock.Adapters.UNet do
  @moduledoc """
  Tier 3 of the model airlock: the **denoiser of Stable-Diffusion-style
  latent diffusion** — diffusers' `UNet2DConditionModel` (SD 1.x and 2.x
  layouts) — built from `Vapor.Spatial` and the algebra's attention: no new
  operator, the same kernels and certificates as everything else.

      latents → conv_in
        → down blocks: [resnet (+ time) → transformer (self-attention,
          cross-attention to the text, GEGLU)] × layers, then a stride-2
          conv; every output kept as a skip
        → mid: resnet, transformer, resnet
        → up blocks: [concat(skip) → resnet → transformer] × (layers + 1),
          then nearest ×2 + conv
        → GroupNorm → SiLU → conv_out → predicted noise (ε)

  The timestep enters as its sinusoidal features (`timestep_features/2`,
  diffusers' `get_timestep_embedding` with `flip_sin_to_cos` and
  `freq_shift`), computed with the correctly rounded `exp`, `sin`, `cos`;
  the text as the encoder's hidden states (`ctx`, e.g. CLIP's last layer).
  Heads of any width are padded to the 16-lane contraction (zero rows in
  q/k/v, zero columns in the output projection; the score scale stays
  1/√dh). Channel concatenation is an exact 0/1 contraction.

  Contract `:map` with three inputs: `rows : f32[h·w, pad16(in)]`, `temb :
  f32[1, pad16(C₀)]`, `ctx : f32[S, pad16(cross_attention_dim)]` → `out :
  f32[h·w, pad16(out)]`. Build options `latent: {h, w}` (default
  `sample_size`²), `context: S` (77). The lowest resolution must hold a
  multiple of 16 pixels (GroupNorm's canonical reduction).

  Refused by name: SDXL's added conditioning (`addition_embed_type`),
  class embeddings, `resnet_time_scale_shift: "scale_shift"`,
  `transformer_layers_per_block ≠ 1`, dual cross-attention, other
  activations.
  """
  @behaviour Vapor.Lock.Adapter
  alias Vapor.{CR, Program, Rejection, Spatial, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Lock.Spec

  defmodule Config do
    @moduledoc "An admitted UNet2DConditionModel."
    defstruct [:in, :out, :blocks, :layers, :down, :up, :mid, :cross, :heads, :groups, :eps, :flip, :shift, :linear_proj, :sample, :raw]
  end

  @impl true
  def id, do: "unet"

  @impl true
  def claim(%{config: %{"_class_name" => "UNet2DConditionModel"}}), do: {:claim, 100}
  def claim(%{config: %{"_class_name" => c}}) when c in ["UNet2DModel", "UNetSpatioTemporalConditionModel", "ControlNetModel"],
    do: {:near, "#{c} is a U-Net relative (unconditional, video or ControlNet): not this adapter yet"}
  def claim(_), do: :no

  @impl true
  def owns?(%Config{}), do: true
  def owns?(_), do: false

  @impl true
  def admit(%{config: c}, ws, _opts) do
    blocks = c["block_out_channels"] || [320, 640, 1280, 1280]
    n = length(blocks)
    layers = c["layers_per_block"] || 2
    heads = c["num_attention_heads"] || c["attention_head_dim"] || 8

    cond do
      c["addition_embed_type"] != nil -> no("addition_embed_type", "none (SDXL's added text/time conditioning is not built yet)")
      c["class_embed_type"] != nil -> no("class_embed_type", "none")
      (c["resnet_time_scale_shift"] || "default") != "default" -> no("resnet_time_scale_shift", "default")
      not (c["transformer_layers_per_block"] in [nil, 1]) -> no("transformer_layers_per_block", "1")
      c["dual_cross_attention"] == true -> no("dual_cross_attention", "false")
      (c["act_fn"] || "silu") != "silu" -> no("act_fn", "silu")
      not is_integer(c["cross_attention_dim"] || 768) -> no("cross_attention_dim", "one integer")
      not is_integer(layers) -> no("layers_per_block", "one integer")
      (c["mid_block_type"] || "UNetMidBlock2DCrossAttn") != "UNetMidBlock2DCrossAttn" -> no("mid_block_type", "UNetMidBlock2DCrossAttn")
      true ->
        cfg = %Config{in: c["in_channels"] || 4, out: c["out_channels"] || 4, blocks: blocks, layers: layers,
                      down: c["down_block_types"] || List.duplicate("CrossAttnDownBlock2D", n - 1) ++ ["DownBlock2D"],
                      up: c["up_block_types"] || ["UpBlock2D" | List.duplicate("CrossAttnUpBlock2D", n - 1)],
                      mid: true, cross: c["cross_attention_dim"] || 768, heads: if(is_list(heads), do: heads, else: List.duplicate(heads, n)),
                      groups: c["norm_num_groups"] || 32, eps: (c["norm_eps"] || 1.0e-5) * 1.0, flip: c["flip_sin_to_cos"] != false,
                      shift: (c["freq_shift"] || 0) * 1.0, linear_proj: c["use_linear_projection"] == true, sample: c["sample_size"] || 64, raw: c}

        missing = ws |> Map.keys() |> Enum.empty?()
        if missing, do: no("weights", "diffusion_pytorch_model.safetensors"), else: {:ok, spec(cfg), ws}
    end
  end

  defp no(f, b), do: {:error, Rejection.new({:config, f}, b, "use a supported checkpoint")}

  @impl true
  def spec(%Config{} = c) do
    %Spec{adapter: __MODULE__, family: "unet2d_condition", lineage: ["unet2d_condition", "unet"], interface: :map, config: c,
          width: Spatial.pad16(c.out), in_width: Spatial.pad16(c.in), features: [:denoise],
          modality: %{in: [:latent, :text_embedding], out: [:latent]}, digest: Vapor.Canonical.hex_digest({:unet, c.raw})}
  end

  # the names are discovered from the checkpoint during the build; every one is read
  @impl true
  def expected(%Spec{}), do: []

  # --------------------------------------------------------------- timestep --

  @doc "diffusers' sinusoidal timestep features `[1, pad16(dim)]` (correctly rounded exp/sin/cos)."
  def timestep_features(%Config{} = c, t) do
    dim = hd(c.blocks)
    half = div(dim, 2)
    freqs = for i <- 0..(half - 1), do: CR.exp_f64(-:math.log(10_000.0) * i / (half - c.shift))
    args = Enum.map(freqs, &(t * &1))
    sin = Enum.map(args, &CR.sin_f64/1)
    cos = Enum.map(args, &CR.cos_f64/1)
    v = if c.flip, do: cos ++ sin, else: sin ++ cos
    Tensor.from_list(:f32, [1, Spatial.pad16(dim)], v ++ List.duplicate(0.0, Spatial.pad16(dim) - dim))
  end

  # ----------------------------------------------------------------- build --

  @impl true
  def build(%Spec{config: c}, ws, opts) do
    {h, w} = Keyword.get(opts, :latent, {c.sample, c.sample})
    s = Keyword.get(opts, :context, 77)
    lowest = div(h, Integer.pow(2, length(c.blocks) - 1)) * div(w, Integer.pow(2, length(c.blocks) - 1))

    if rem(lowest, 16) != 0 or rem(h * w, 16) != 0 do
      {:error, Rejection.new({:unet, :latent}, "a multiple of 16 pixels at every resolution (got #{h}×#{w})", "use a latent side multiple of #{4 * Integer.pow(2, length(c.blocks) - 1)}")}
    else
      ws = Map.new(ws, fn {k, t} -> {k, if(is_binary(k), do: Tensor.widen(t), else: t)} end)
      g = %{ws: ws, c: c}
      x = T.input(:rows, :f32, [h * w, Spatial.pad16(c.in)])
      temb = T.input(:temb, :f32, [1, Spatial.pad16(hd(c.blocks))])
      ctx = T.input(:ctx, :f32, [s, Spatial.pad16(c.cross)])
      lets = []

      # time embedding: linear → silu → linear
      {lets, e1} = lin(lets, g, "time_embedding.linear_1", temb)
      {lets, emb} = lin(lets, g, "time_embedding.linear_2", T.silu(e1))
      {lets, emb} = Spatial.name(lets, "temb.out", emb)
      g = Map.merge(g, %{emb: T.silu(emb), ctx: ctx, s: s})

      {lets, x, d} = conv(lets, g, x, {h, w, c.in}, "conv_in", 3)
      skips = [{x, d}]

      {lets, x, d, skips} =
        c.down
        |> Enum.with_index()
        |> Enum.reduce({lets, x, d, skips}, fn {type, bi}, {lets, x, d, skips} ->
          {lets, x, d, skips} =
            Enum.reduce(0..(c.layers - 1), {lets, x, d, skips}, fn j, {lets, x, d, skips} ->
              p = "down_blocks.#{bi}"
              {lets, x, d} = resnet(lets, g, x, d, "#{p}.resnets.#{j}")
              {lets, x} = if type == "CrossAttnDownBlock2D", do: transformer(lets, g, x, d, "#{p}.attentions.#{j}", Enum.at(c.heads, bi)), else: {lets, x}
              {lets, x, d, skips ++ [{x, d}]}
            end)

          if Map.has_key?(ws, "down_blocks.#{bi}.downsamplers.0.conv.weight") do
            {lets, x, d} = conv(lets, g, x, d, "down_blocks.#{bi}.downsamplers.0.conv", 3, stride: 2, padding: 1)
            {lets, x, d, skips ++ [{x, d}]}
          else
            {lets, x, d, skips}
          end
        end)

      {lets, x, d} = resnet(lets, g, x, d, "mid_block.resnets.0")
      {lets, x} = transformer(lets, g, x, d, "mid_block.attentions.0", List.last(c.heads))
      {lets, x, d} = resnet(lets, g, x, d, "mid_block.resnets.1")

      rev_heads = Enum.reverse(c.heads)

      {lets, x, d, _} =
        c.up
        |> Enum.with_index()
        |> Enum.reduce({lets, x, d, skips}, fn {type, bi}, {lets, x, d, skips} ->
          {lets, x, d, skips} =
            Enum.reduce(0..c.layers, {lets, x, d, skips}, fn j, {lets, x, d, skips} ->
              p = "up_blocks.#{bi}"
              {{sx, {_, _, sc}}, skips} = List.pop_at(skips, -1)
              {lets, x, d} = concat(lets, x, d, sx, sc, "#{p}.concat.#{j}")
              {lets, x, d} = resnet(lets, g, x, d, "#{p}.resnets.#{j}")
              {lets, x} = if type == "CrossAttnUpBlock2D", do: transformer(lets, g, x, d, "#{p}.attentions.#{j}", Enum.at(rev_heads, bi)), else: {lets, x}
              {lets, x, d, skips}
            end)

          if Map.has_key?(ws, "up_blocks.#{bi}.upsamplers.0.conv.weight") do
            {lets, x, d} = Spatial.upsample_nearest(lets, x, d, "up_blocks.#{bi}.upsampled")
            {lets, x, d} = conv(lets, g, x, d, "up_blocks.#{bi}.upsamplers.0.conv", 3)
            {lets, x, d, skips}
          else
            {lets, x, d, skips}
          end
        end)

      {hh, ww, ch} = d
      {lets, x} = Spatial.group_norm(lets, x, hh * ww, ch, c.groups, "conv_norm_out", ws["conv_norm_out.weight"], ws["conv_norm_out.bias"], c.eps)
      {lets, out, _} = conv(lets, g, T.silu(x), d, "conv_out", 3)
      {:ok, Program.new([out: out], lets: Enum.reverse(lets))}
    end
  end

  defp conv(lets, g, x, d, name, k, opts \\ []),
    do: Spatial.conv2d(lets, x, d, name, g.ws[name <> ".weight"], g.ws[name <> ".bias"], Keyword.merge([padding: div(k - 1, 2)], opts))

  # a linear layer from a [o, i] weight (and [o] bias) on f32[r, pad16(i)] rows
  defp lin(lets, g, name, x, bias? \\ true) do
    %Tensor{shape: [o, i]} = wt = g.ws[name <> ".weight"]
    {lets, wr} = Spatial.bind(lets, name <> ".weight", pad_matrix(wt, Spatial.pad16(o), Spatial.pad16(i)))
    y = T.linear(x, wr)
    if bias? and g.ws[name <> ".bias"] do
      {lets, br} = Spatial.bind(lets, name <> ".bias", pad_row(g.ws[name <> ".bias"], Spatial.pad16(o)))
      {lets, T.add(y, br)}
    else
      {lets, y}
    end
  end

  defp resnet(lets, g, x, {hh, ww, i} = d, p) do
    c = g.c
    ws = g.ws
    o = hd(ws[p <> ".conv1.weight"].shape)
    n = hh * ww
    {lets, h1} = Spatial.group_norm(lets, x, n, i, c.groups, p <> ".norm1", ws[p <> ".norm1.weight"], ws[p <> ".norm1.bias"], c.eps)
    {lets, h1, d1} = Spatial.conv2d(lets, T.silu(h1), d, p <> ".conv1", ws[p <> ".conv1.weight"], ws[p <> ".conv1.bias"], padding: 1)
    {lets, t} = lin(lets, g, p <> ".time_emb_proj", g.emb)
    {lets, h1} = Spatial.name(lets, p <> ".with_time", T.add(h1, t))
    {lets, h2} = Spatial.group_norm(lets, h1, n, o, c.groups, p <> ".norm2", ws[p <> ".norm2.weight"], ws[p <> ".norm2.bias"], c.eps)
    {lets, h2, _} = Spatial.conv2d(lets, T.silu(h2), d1, p <> ".conv2", ws[p <> ".conv2.weight"], ws[p <> ".conv2.bias"], padding: 1)

    {lets, skip} =
      if ws[p <> ".conv_shortcut.weight"] do
        {lets, sk, _} = Spatial.conv2d(lets, x, d, p <> ".conv_shortcut", ws[p <> ".conv_shortcut.weight"], ws[p <> ".conv_shortcut.bias"])
        {lets, sk}
      else
        {lets, x}
      end

    {lets, y} = Spatial.name(lets, p <> ".out", T.add(skip, h2))
    {lets, y, {hh, ww, o}}
  end

  defp transformer(lets, g, x, {hh, ww, ch}, p, heads) do
    c = g.c
    ws = g.ws
    n = hh * ww
    {lets, hn} = Spatial.group_norm(lets, x, n, ch, c.groups, p <> ".norm", ws[p <> ".norm.weight"], ws[p <> ".norm.bias"], 1.0e-6)
    {lets, h} = proj(lets, g, hn, {hh, ww, ch}, p <> ".proj_in")
    b = p <> ".transformer_blocks.0"

    {lets, n1} = layer_norm(lets, h, ch, b <> ".norm1", ws, 1.0e-5)
    {lets, a1} = attention(lets, g, n1, ch, n1, ch, n, b <> ".attn1", heads)
    {lets, h} = Spatial.name(lets, b <> ".after_self", T.add(h, a1))
    {lets, n2} = layer_norm(lets, h, ch, b <> ".norm2", ws, 1.0e-5)
    {lets, a2} = attention(lets, g, n2, ch, g.ctx, c.cross, g.s, b <> ".attn2", heads)
    {lets, h} = Spatial.name(lets, b <> ".after_cross", T.add(h, a2))
    {lets, n3} = layer_norm(lets, h, ch, b <> ".norm3", ws, 1.0e-5)
    {lets, f} = geglu(lets, g, n3, b <> ".ff")
    {lets, h} = Spatial.name(lets, b <> ".after_ff", T.add(h, f))
    {lets, o} = proj(lets, g, h, {hh, ww, ch}, p <> ".proj_out")
    Spatial.name(lets, p <> ".out", T.add(o, x))
  end

  # proj_in/proj_out: a 1×1 convolution, or a linear layer (use_linear_projection)
  defp proj(lets, g, x, d, name) do
    case g.ws[name <> ".weight"] do
      %Tensor{shape: [_, _, 1, 1]} -> (fn {lets, y, _} -> {lets, y} end).(Spatial.conv2d(lets, x, d, name, g.ws[name <> ".weight"], g.ws[name <> ".bias"]))
      %Tensor{shape: [_, _]} -> lin(lets, g, name, x)
    end
  end

  # LayerNorm over the c real channels of f32[n, pad16(c)] (padded columns masked out of the variance)
  defp layer_norm(lets, x, c, name, ws, eps) do
    cp = Spatial.pad16(c)
    {lets, mask} = Spatial.bind(lets, "$ln.mask.#{c}", Tensor.from_list(:f32, [1, cp], for(i <- 0..(cp - 1), do: if(i < c, do: 1.0, else: 0.0))))
    mean = T.mul(T.reduce(:sum, x), T.splat(1.0 / c))
    {lets, xc} = Spatial.name(lets, name <> ".centred", T.mul(T.sub(x, mean), mask))
    var = T.mul(T.reduce(:sum, T.mul(xc, xc)), T.splat(1.0 / c))
    {lets, wr} = Spatial.bind(lets, name <> ".weight", pad_row(ws[name <> ".weight"], cp))
    {lets, br} = Spatial.bind(lets, name <> ".bias", pad_row(ws[name <> ".bias"], cp))
    Spatial.name(lets, name <> ".out", T.add(T.mul(wr, T.mul(xc, T.rsqrt(T.add(var, T.splat(eps))))), br))
  end

  # multi-head attention, heads padded to the 16-lane width; full attention (every row sees all S keys)
  defp attention(lets, g, x, c, src, cs, s, name, heads) do
    dh = div(c, heads)
    dhp = Spatial.pad16(dh)
    {cp, csp} = {Spatial.pad16(c), Spatial.pad16(cs)}
    head_rows = fn %Tensor{shape: [o, i]} = wt, ip ->
      v = wt |> Tensor.to_floats() |> Enum.chunk_every(i) |> List.to_tuple()
      rows = for hd <- 0..(heads - 1), r <- 0..(dhp - 1), do: (if r < dh, do: elem(v, hd * dh + r) ++ List.duplicate(0.0, ip - i), else: List.duplicate(0.0, ip))
      _ = o
      Tensor.from_list(:f32, [heads * dhp, ip], List.flatten(rows))
    end

    {lets, wq} = Spatial.bind(lets, name <> ".to_q", head_rows.(g.ws[name <> ".to_q.weight"], cp))
    {lets, wk} = Spatial.bind(lets, name <> ".to_k", head_rows.(g.ws[name <> ".to_k.weight"], csp))
    {lets, wv} = Spatial.bind(lets, name <> ".to_v", head_rows.(g.ws[name <> ".to_v.weight"], csp))
    # the output projection reads each head's dh columns out of its padded block
    wo = g.ws[name <> ".to_out.0.weight"]
    ov = wo |> Tensor.to_floats() |> Enum.chunk_every(c)
    wo_rows = for r <- 0..(cp - 1), do: (if r < c, do: (fn row -> for(hd <- 0..(heads - 1), j <- 0..(dhp - 1), do: if(j < dh, do: Enum.at(row, hd * dh + j), else: 0.0)) end).(Enum.at(ov, r)), else: List.duplicate(0.0, heads * dhp))
    {lets, wor} = Spatial.bind(lets, name <> ".to_out", Tensor.from_list(:f32, [cp, heads * dhp], List.flatten(wo_rows)))
    {lets, bo} = Spatial.bind(lets, name <> ".to_out.bias", pad_row(g.ws[name <> ".to_out.0.bias"], cp))
    {lets, hz} = Spatial.bind(lets, "$unet.horizon.#{elem(dims(x), 0)}.#{s}", Tensor.from_list(:s32, [elem(dims(x), 0)], List.duplicate(s - 1, elem(dims(x), 0))))
    {lets, k} = Spatial.name(lets, name <> ".k", T.linear(src, wk))
    {lets, v} = Spatial.name(lets, name <> ".v", T.linear(src, wv))
    att = T.attention(T.linear(x, wq), k, v, hz, heads, heads, 1.0 / :math.sqrt(dh))
    Spatial.name(lets, name <> ".out", T.add(T.linear(att, wor), bo))
  end

  defp dims(t), do: (fn {:ok, {:f32, [r, cc]}} -> {r, cc} end).(T.infer(t))

  # GEGLU: (x·Wₕ) ⊙ gelu(x·W_g), then the output linear
  defp geglu(lets, g, x, name) do
    %Tensor{shape: [two_inner, c]} = wt = g.ws[name <> ".net.0.proj.weight"]
    inner = div(two_inner, 2)
    {ip, cp} = {Spatial.pad16(inner), Spatial.pad16(c)}
    rows = wt |> Tensor.to_floats() |> Enum.chunk_every(c)
    bias = g.ws[name <> ".net.0.proj.bias"] |> Tensor.to_floats()
    part = fn from -> Tensor.from_list(:f32, [ip, cp], List.flatten(for r <- 0..(ip - 1), do: if(r < inner, do: Enum.at(rows, from + r) ++ List.duplicate(0.0, cp - c), else: List.duplicate(0.0, cp)))) end
    pb = fn from -> Tensor.from_list(:f32, [1, ip], for(r <- 0..(ip - 1), do: if(r < inner, do: Enum.at(bias, from + r), else: 0.0))) end
    {lets, wh} = Spatial.bind(lets, name <> ".hidden", part.(0))
    {lets, wg} = Spatial.bind(lets, name <> ".gate", part.(inner))
    {lets, bh} = Spatial.bind(lets, name <> ".hidden.bias", pb.(0))
    {lets, bg} = Spatial.bind(lets, name <> ".gate.bias", pb.(inner))
    {lets, act} = Spatial.name(lets, name <> ".act", T.mul(T.add(T.linear(x, wh), bh), T.gelu(T.add(T.linear(x, wg), bg))))
    lin(lets, g, name <> ".net.2", act)
  end

  # channel concatenation [x | skip] as exact 0/1 contractions onto the joint padded width
  defp concat(lets, x, {hh, ww, c1}, sx, c2, name) do
    {p1, p2, pj} = {Spatial.pad16(c1), Spatial.pad16(c2), Spatial.pad16(c1 + c2)}
    e1 = Tensor.from_list(:f32, [pj, p1], for(o <- 0..(pj - 1), i <- 0..(p1 - 1), do: if(o < c1 and o == i, do: 1.0, else: 0.0)))
    e2 = Tensor.from_list(:f32, [pj, p2], for(o <- 0..(pj - 1), i <- 0..(p2 - 1), do: if(i < c2 and o == c1 + i, do: 1.0, else: 0.0)))
    {lets, r1} = Spatial.bind(lets, "$cat.#{c1}.#{c2}.a", e1)
    {lets, r2} = Spatial.bind(lets, "$cat.#{c1}.#{c2}.b", e2)
    {lets, y} = Spatial.name(lets, name, T.add(T.linear(x, r1), T.linear(sx, r2)))
    {lets, y, {hh, ww, c1 + c2}}
  end

  defp pad_row(%Tensor{} = t, n) do
    v = Tensor.to_floats(Tensor.widen(t))
    Tensor.from_list(:f32, [1, n], v ++ List.duplicate(0.0, n - length(v)))
  end

  defp pad_matrix(%Tensor{shape: [m, k]} = t, mp, kp) do
    v = t |> Tensor.widen() |> Tensor.to_floats() |> Enum.chunk_every(k)
    Tensor.from_list(:f32, [mp, kp], List.flatten(for(r <- 0..(mp - 1), do: if(r < m, do: Enum.at(v, r) ++ List.duplicate(0.0, kp - k), else: List.duplicate(0.0, kp)))))
  end

  # ------------------------------------------------------------- helpers --

  @doc "The program's environment: diffusers' latents `[C, h, w]`, the timestep, the text states `[S, D]`."
  def input(%Spec{config: c}, %Tensor{} = z, t, %Tensor{shape: [s, dd]} = ctx) do
    {rows, _} = Spatial.from_nchw(z)
    dp = Spatial.pad16(dd)
    cv = ctx |> Tensor.widen() |> Tensor.to_floats() |> Enum.chunk_every(dd) |> Enum.flat_map(&(&1 ++ List.duplicate(0.0, dp - dd)))
    %{rows: rows, temb: timestep_features(c, t), ctx: Tensor.from_list(:f32, [s, dp], cv)}
  end

  @doc "The predicted noise as diffusers' `[C, h, w]`."
  def output(%Spec{config: c}, %Tensor{} = out, {h, w}), do: Spatial.to_nchw(out, {h, w, c.out})
end
