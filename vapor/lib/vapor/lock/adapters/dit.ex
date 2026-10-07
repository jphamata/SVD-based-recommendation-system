defmodule Vapor.Lock.Adapters.DiT do
  @moduledoc """
  Tier 3 of the model airlock: the **diffusion transformer** — diffusers'
  `DiTTransformer2DModel` (Peebles & Xie's DiT, class-conditional, adaLN-Zero)
  — the denoiser of latent diffusion, as a program of the certified algebra
  with no new operator: a transformer is what vapor already runs; what is
  specific here is conditioning by *modulation*, and modulation is a
  per-row broadcast of vectors computed from the conditioning.

      x  = patches·Wₚᵀ + b + pos                     (the patch convolution is a linear
                                                      map over patch rows; pos = 2-D sin-cos)
      per block, c = silu(MLP(temb) + E[label]) and its six vectors
        x += gate₁ · attn(LN(x)·(1 + scale₁) + shift₁)
        x += gate₂ · FF(LN(x)·(1 + scale₂) + shift₂)  (FF = GELU-tanh MLP)
      out = (LN(x)·(1 + scale) + shift)·W_outᵀ + b   (shift, scale from block 0's c)

  Program contract `:map`: inputs `rows : f32[N, C·p·p]` (the latent cut into
  `p×p` patches, channel-major inside a patch — `input/2`), `temb : f32[1,
  256]` (the sinusoidal timestep features, `timestep_features/1`) and
  `label : s32[1]`; output `out : f32[N, p·p·C_out]` (`image/3` puts the
  patches back as `[C_out, H, W]`: the noise prediction and, for learned
  sigma, the variance channels).

  The sinusoidal timestep features are an *input*: diffusers computes them
  with the host's libm in float32, and reproducing them bit for bit is a
  property of that library, not of the model. vapor's own
  `timestep_features/1` follows PyTorch's binary32 steps with correctly
  rounded `exp`/`sin`/`cos` (within libm's last-ulp errors of PyTorch's);
  the parity test feeds diffusers' own features to compare the model alone,
  and measures vapor's features separately.
  """
  @behaviour Vapor.Lock.Adapter
  alias Vapor.{Program, Rejection, Spatial, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Lock.Spec

  defmodule Config do
    @moduledoc "An admitted DiT."
    defstruct [:heads, :head_dim, :width, :in_ch, :out_ch, :layers, :patch, :sample, :classes, :norm_eps, :raw]
  end

  @impl true
  def id, do: "dit"

  @impl true
  def claim(%{config: %{"_class_name" => "DiTTransformer2DModel"}}), do: {:claim, 100}

  def claim(%{config: %{"_class_name" => c}}) when c in ["PixArtTransformer2DModel", "SD3Transformer2DModel", "FluxTransformer2DModel"],
    do: {:near, "#{c} is a DiT relative (cross-attention to text, joint attention or rotary 2-D positions): not this adapter yet"}

  def claim(_), do: :no

  @impl true
  def owns?(%Config{}), do: true
  def owns?(_), do: false

  @impl true
  def admit(%{config: c}, ws, _opts) do
    cfg = %Config{heads: c["num_attention_heads"] || 16, head_dim: c["attention_head_dim"] || 72, in_ch: c["in_channels"] || 4,
                  out_ch: c["out_channels"] || c["in_channels"] || 4, layers: c["num_layers"] || 28, patch: c["patch_size"] || 2,
                  sample: c["sample_size"] || 32, classes: c["num_embeds_ada_norm"] || 1000, norm_eps: (c["norm_eps"] || 1.0e-5) * 1.0,
                  raw: c}

    cfg = %{cfg | width: cfg.heads * cfg.head_dim}

    cond do
      (c["norm_type"] || "ada_norm_zero") != "ada_norm_zero" -> no("norm_type", "ada_norm_zero")
      (c["activation_fn"] || "gelu-approximate") != "gelu-approximate" -> no("activation_fn", "gelu-approximate")
      c["norm_elementwise_affine"] == true -> no("norm_elementwise_affine", "false")
      rem(cfg.head_dim, 16) != 0 -> no("attention_head_dim", "a multiple of 16")
      rem(cfg.in_ch * cfg.patch * cfg.patch, 16) != 0 -> no("in_channels·patch_size²", "a multiple of 16")
      rem(cfg.sample, cfg.patch) != 0 -> no("sample_size", "a multiple of patch_size")
      true ->
        spec = spec(cfg)

        case Enum.find(expected(spec), fn {n, shape, _} -> not match?(%Tensor{shape: ^shape}, ws[n]) end) do
          nil -> {:ok, spec, ws}
          {n, shape, _} -> {:error, Rejection.new({:weight, n}, inspect(shape, charlists: :as_lists), "check the checkpoint against config.json")}
        end
    end
  end

  defp no(f, b), do: {:error, Rejection.new({:config, f}, b, "use a supported checkpoint")}

  @impl true
  def spec(%Config{} = c) do
    %Spec{adapter: __MODULE__, family: "dit", lineage: ["dit", "transformer"], interface: :map, config: c,
          width: c.patch * c.patch * c.out_ch, in_width: c.in_ch * c.patch * c.patch, features: [:denoise, :class_conditional],
          modality: %{in: [:latent], out: [:latent]}, digest: Vapor.Canonical.hex_digest({:dit, c.raw})}
  end

  @impl true
  def expected(%Spec{config: c}) do
    d = c.width
    lin = fn n, o, i -> [{n <> ".weight", [o, i], :matrix}, {n <> ".bias", [o], :bias}] end

    blocks =
      for l <- 0..(c.layers - 1) do
        b = "transformer_blocks.#{l}."
        e = b <> "norm1.emb."
        lin.(e <> "timestep_embedder.linear_1", d, 256) ++ lin.(e <> "timestep_embedder.linear_2", d, d) ++
          [{e <> "class_embedder.embedding_table.weight", [c.classes + 1, d], :matrix}] ++ lin.(b <> "norm1.linear", 6 * d, d) ++
          Enum.flat_map(~w(to_q to_k to_v to_out.0), &lin.(b <> "attn1." <> &1, d, d)) ++
          lin.(b <> "ff.net.0.proj", 4 * d, d) ++ lin.(b <> "ff.net.2", d, 4 * d)
      end

    [{"pos_embed.proj.weight", [d, c.in_ch, c.patch, c.patch], :matrix}, {"pos_embed.proj.bias", [d], :bias}] ++
      List.flatten(blocks) ++ lin.("proj_out_1", 2 * d, d) ++ lin.("proj_out_2", c.patch * c.patch * c.out_ch, d)
  end

  # ----------------------------------------------------------------- build --

  @impl true
  def build(%Spec{config: c} = spec, ws, opts) do
    d = c.width
    side = div(Keyword.get(opts, :sample, c.sample), c.patch)
    n = side * side
    k = c.in_ch * c.patch * c.patch
    w = fn name -> Tensor.widen(ws[name]) end
    rowv = fn t -> Tensor.new(:f32, [1, hd(t.shape)], Tensor.widen(t).data) end
    flat = fn t -> %{Tensor.widen(t) | shape: [hd(t.shape), div(Enum.product(t.shape), hd(t.shape))]} end

    lets = []
    {lets, wp} = Spatial.bind(lets, "pos_embed.proj.weight", flat.(ws["pos_embed.proj.weight"]))
    {lets, bp} = Spatial.bind(lets, "pos_embed.proj.bias", rowv.(ws["pos_embed.proj.bias"]))
    {lets, pos} = Spatial.bind(lets, "$dit.pos#{side}", pos_table(d, side, div(c.sample, c.patch)))

    rows = T.input(:rows, :f32, [n, k])
    temb = T.input(:temb, :f32, [1, 256])
    label = T.input(:label, :s32, [1])
    {lets, x} = Spatial.name(lets, :x_embed, T.add(T.add(T.linear(rows, wp), bp), pos))

    lin = fn lets, name, x ->
      {lets, wr} = Spatial.bind(lets, name <> ".weight", w.(name <> ".weight"))
      {lets, br} = Spatial.bind(lets, name <> ".bias", rowv.(ws[name <> ".bias"]))
      {lets, T.add(T.linear(x, wr), br)}
    end

    # the six modulation vectors of a block: one linear map split by rows
    mods = fn lets, b, cond ->
      Enum.map_reduce(0..5, lets, fn j, lets ->
        wt = w.(b <> "norm1.linear.weight")
        part = Tensor.new(:f32, [d, d], binary_part(wt.data, j * d * d * 4, d * d * 4))
        bias = Tensor.new(:f32, [1, d], binary_part(w.(b <> "norm1.linear.bias").data, j * d * 4, d * 4))
        {lets, wr} = Spatial.bind(lets, b <> "norm1.linear.#{j}.weight", part)
        {lets, br} = Spatial.bind(lets, b <> "norm1.linear.#{j}.bias", bias)
        {T.add(T.linear(cond, wr), br), lets}
      end)
      |> then(fn {vs, lets} -> {lets, vs} end)
    end

    {lets, x, cond0} =
      Enum.reduce(0..(c.layers - 1), {lets, x, nil}, fn l, {lets, x, cond0} ->
        b = "transformer_blocks.#{l}."
        e = b <> "norm1.emb."
        {lets, t1} = lin.(lets, e <> "timestep_embedder.linear_1", temb)
        {lets, t2} = lin.(lets, e <> "timestep_embedder.linear_2", T.silu(t1))
        {lets, table} = Spatial.bind(lets, e <> "class_embedder.embedding_table.weight", w.(e <> "class_embedder.embedding_table.weight"))
        {lets, cond} = Spatial.name(lets, b <> "cond", T.add(t2, T.gather_row(table, label)))
        {lets, [sh1, sc1, g1, sh2, sc2, g2]} = mods.(lets, b, T.silu(cond))

        {lets, h} = Spatial.name(lets, b <> "attn_in", modulate(layer_norm(x, d, 1.0e-6), sh1, sc1))
        {lets, q} = lin.(lets, b <> "attn1.to_q", h)
        {lets, kk} = lin.(lets, b <> "attn1.to_k", h)
        {lets, vv} = lin.(lets, b <> "attn1.to_v", h)
        {lets, hz} = Spatial.bind(lets, "$dit.horizon#{n}", Tensor.from_list(:s32, [n], List.duplicate(n - 1, n)))
        {lets, kk} = Spatial.name(lets, b <> "k", kk)
        {lets, vv} = Spatial.name(lets, b <> "v", vv)
        {lets, a} = lin.(lets, b <> "attn1.to_out.0", T.attention(q, kk, vv, hz, c.heads, c.heads))
        {lets, x} = Spatial.name(lets, b <> "attn_out", T.add(x, T.mul(g1, a)))

        {lets, h2} = Spatial.name(lets, b <> "ff_in", modulate(layer_norm(x, d, c.norm_eps), sh2, sc2))
        {lets, f1} = lin.(lets, b <> "ff.net.0.proj", h2)
        {lets, f2} = lin.(lets, b <> "ff.net.2", T.gelu_tanh(f1))
        {lets, x} = Spatial.name(lets, b <> "out", T.add(x, T.mul(g2, f2)))
        {lets, x, cond0 || cond}
      end)

    # the final modulation: shift and scale from block 0's conditioning
    wt = w.("proj_out_1.weight")
    bt = w.("proj_out_1.bias")

    {[shift, scale], lets} =
      Enum.map_reduce(0..1, lets, fn j, lets ->
        {lets, wr} = Spatial.bind(lets, "proj_out_1.#{j}.weight", Tensor.new(:f32, [d, d], binary_part(wt.data, j * d * d * 4, d * d * 4)))
        {lets, br} = Spatial.bind(lets, "proj_out_1.#{j}.bias", Tensor.new(:f32, [1, d], binary_part(bt.data, j * d * 4, d * 4)))
        {T.add(T.linear(T.silu(cond0), wr), br), lets}
      end)

    {lets, out} = lin.(lets, "proj_out_2", modulate(layer_norm(x, d, 1.0e-6), shift, scale))
    _ = spec
    {:ok, Program.new([out: out], lets: Enum.reverse(lets))}
  end

  # LayerNorm without affine parameters (two passes, biased variance)
  defp layer_norm(x, d, eps) do
    inv = T.splat(1.0 / d)
    mean = T.mul(T.reduce(:sum, x), inv)
    xc = T.sub(x, mean)
    var = T.mul(T.reduce(:sum, T.mul(xc, xc)), inv)
    T.mul(xc, T.rsqrt(T.add(var, T.splat(eps))))
  end

  defp modulate(x, shift, scale), do: T.add(T.mul(x, T.add(scale, T.splat(1.0))), shift)

  # diffusers' 2-D sin-cos table: grid coordinates in binary32 (scaled by the
  # base grid), frequencies, products and sin/cos in binary64, rounded once
  defp pos_table(d, side, base) do
    q = d |> div(2) |> div(2)
    omega = for i <- 0..(q - 1), do: 1.0 / :math.pow(10_000, i / (q * 1.0))
    coord = fn v -> Vapor.F32.to_float(Vapor.F32.from_float(v / (side / base))) end
    one_d = fn p -> Enum.map(omega, &:math.sin(p * &1)) ++ Enum.map(omega, &:math.cos(p * &1)) end

    vals = for r <- 0..(side - 1), cc <- 0..(side - 1), do: one_d.(coord.(cc)) ++ one_d.(coord.(r))
    Tensor.from_list(:f32, [side * side, d], List.flatten(vals))
  end

  # ---------------------------------------------------------------- helpers --

  @doc """
  diffusers' sinusoidal timestep features (`Timesteps(256, flip_sin_to_cos:
  true, downscale_freq_shift: 1)`): `[cos(t·fᵢ) …, sin(t·fᵢ) …]` with
  PyTorch's binary32 steps — `e = f32(−ln 10⁴)·i`, `e/127`, `fᵢ = exp(e)`,
  `t·fᵢ` each rounded to binary32 — and `exp`, `sin`, `cos` correctly
  rounded (`Vapor.CR`), so it differs from PyTorch's libm by at most its
  last-ulp errors.
  """
  def timestep_features(t) when is_number(t) do
    f32 = &Vapor.CR.to_f32/1
    c = f32.(-:math.log(10_000))

    f =
      for i <- 0..127 do
        e = f32.(f32.(c * i) / 127)
        f32.(Vapor.CR.exp_f64(e))
      end

    args = Enum.map(f, &f32.(f32.(t * 1.0) * &1))
    Tensor.from_list(:f32, [1, 256], Enum.map(args, &Vapor.CR.cos_f32/1) ++ Enum.map(args, &Vapor.CR.sin_f32/1))
  end

  @doc "The program's inputs for latents `[C, H, W]`, the timestep features and a class label."
  def input(%Spec{config: c}, %Tensor{shape: [ch, h, w]} = z, %Tensor{} = temb, label) do
    p = c.patch
    v = z |> Tensor.widen() |> Tensor.to_floats() |> List.to_tuple()

    rows =
      for py <- 0..(div(h, p) - 1), px <- 0..(div(w, p) - 1), cc <- 0..(ch - 1), dy <- 0..(p - 1), dx <- 0..(p - 1),
          do: elem(v, (cc * h + py * p + dy) * w + px * p + dx)

    %{rows: Tensor.from_list(:f32, [div(h, p) * div(w, p), ch * p * p], rows), temb: temb, label: Tensor.from_list(:s32, [1], [label])}
  end

  @doc "The output rows back to `[C_out, H, W]` (diffusers' unpatchify: each row is p × p × C_out)."
  def image(%Spec{config: c}, %Tensor{shape: [n, _]} = out, side \\ nil) do
    p = c.patch
    side = side || round(:math.sqrt(n))
    v = out |> Tensor.to_floats() |> List.to_tuple()
    width = p * p * c.out_ch

    vals =
      for cc <- 0..(c.out_ch - 1), y <- 0..(side * p - 1), x <- 0..(side * p - 1) do
        {py, dy, px, dx} = {div(y, p), rem(y, p), div(x, p), rem(x, p)}
        elem(v, (py * side + px) * width + (dy * p + dx) * c.out_ch + cc)
      end

    Tensor.from_list(:f32, [c.out_ch, side * p, side * p], vals)
  end
end
