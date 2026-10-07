defmodule Vapor.Lock.Adapters.Encoder do
  @moduledoc """
  Tier 3 of the model airlock, a **topology**: the bidirectional
  transformer encoder over rows — image patches (ViT, CLIP/SigLIP vision
  towers), audio frames, any modality cut into rows — built from the
  existing operators of the algebra, with no new kernel:

      x = rows·Wₚᵀ + bₚ                 (a patch / frame embedding: the convolution
                                          whose stride equals its kernel *is* this
                                          linear map over rows cut by the codec)
      x = sel(cls_mask, ½, x, cls)      (the learned [CLS] row, when declared)
      x = x + pos                       (learned absolute positions)
      per layer (pre-norm):
        h = norm(x)
        x = x + attention(h·Wq+b, h·Wk+b, h·Wv+b, horizon)·Woᵀ + b
        x = x + act(norm(x)·W₁ᵀ + b₁)·W₂ᵀ + b₂
      hidden = norm(x)                  (+ pooled row 0 and classifier logits)

  **Bidirectional attention is causal attention whose horizon is the last
  row.** The algebra's attention lets row `t` see rows `0 … pos[t]` of a
  key/value table. An encoder passes its own keys and values as that table
  (`T` static rows) and `horizon[t] = n − 1` for every row: each row then
  sees all `n` real rows, and padding rows `≥ n` are never seen. The same
  operator, the same kernels on every substrate, the same certificate.

  Configurations:

    * `model_type: "vit"` — a Hugging Face ViT (`ViTModel` or
      `ViTForImageClassification`, with or without the `vit.` prefix):
      LayerNorm with bias, exact GELU, q/k/v biases, [CLS], learned
      positions; `image_size`, `patch_size`, `num_channels`; a
      `classifier` head when present.
    * `model_type: "vapor_encoder"` — the same topology described
      directly (`rows`, `row_width`, `hidden_size`, `num_hidden_layers`,
      `num_attention_heads`, `intermediate_size`, `norm` `"layer"` |
      `"rms"`, `hidden_act`, `cls`, `num_labels`, `head` `"pooled"` |
      `"rows"`, `modality`), tensors
      named `embed.weight`, `embed.bias`, `cls`, `pos`,
      `layers.{l}.{norm1,norm2}.{weight,bias}`,
      `layers.{l}.{q,k,v,o,fc1,fc2}.{weight,bias}`, `norm.{weight,bias}`,
      `head.{weight,bias}`.

  Program contract `:encoder`: inputs `rows : f32[T, k]` and
  `horizon : s32[T]`, output `hidden : f32[T, d]`, plus `pooled : f32[1, d]`
  and `logits : f32[1, labels]` when the model has a head — or, with
  `head: "rows"`, `row_logits : f32[T, labels]`: every row classified (a
  frame of a text line or of speech, decoded by CTC — `Vapor.Vision.OCR`).
  """
  @behaviour Vapor.Lock.Adapter
  alias Vapor.{Program, Rejection, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Lock.Spec

  defmodule Config do
    @moduledoc "An admitted encoder configuration (opaque to the core)."
    defstruct [:arch, :rows, :row_width, :width, :layers, :heads, :intermediate, :eps, :norm, :act, :cls, :labels,
               :names, :prefix, :modality, :image, :raw,
               # CLIP-style towers: a norm before the layers, the final norm on
               # the pooled row only, no patch bias, a projection of the pooled row
               pre_norm: false, final_norm: :all, embed_bias: true, pooler: nil, projection: nil,
               # where the classifier head reads: the pooled row 0, or every
               # row (token / frame classification: OCR and speech with CTC)
               head_on: :pooled,
               # text towers (CLIP): rows are token embeddings, attention is
               # causal (the horizon of row t is t) and the pooled row is the
               # end-of-text token's, chosen per input (`pick`)
               embed: :rows, vocab: nil, eos: nil]
  end

  @impl true
  def id, do: "encoder"

  @impl true
  def claim(%{config: %{"model_type" => t}}) when t in ["vit", "vapor_encoder", "clip_vision_model", "clip_text_model", "clip"],
    do: {:claim, 100}

  def claim(%{config: %{"model_type" => t}}) when t in ["siglip_vision_model", "dinov2", "deit"],
    do: {:near, "#{t} is this encoder topology under other names — register an alias, or extend the encoder's name map"}

  def claim(_), do: :no

  @impl true
  def owns?(%Config{}), do: true
  def owns?(_), do: false

  # ------------------------------------------------------------- admission --

  @impl true
  def admit(%{config: %{"model_type" => "vit"} = c}, ws, _opts) do
    prefix = if Map.has_key?(ws, "vit.embeddings.cls_token"), do: "vit.", else: ""

    with {:ok, [d, l, h, ff, img, p, ch]} <- ints(c, ~w(hidden_size num_hidden_layers num_attention_heads intermediate_size image_size patch_size num_channels), %{"num_channels" => 3}),
         :ok <- need(rem(img, p) == 0, "image_size", "a multiple of patch_size"),
         {:ok, act} <- activation(c["hidden_act"] || "gelu"),
         :ok <- need(c["qkv_bias"] != false, "qkv_bias", "true"),
         labels = if(Map.has_key?(ws, "classifier.weight"), do: hd(ws["classifier.weight"].shape)),
         n = div(img, p) * div(img, p) do
      # ViTModel's pooler: tanh(dense(hidden[0])), when the checkpoint has one
      pooler = if Map.has_key?(ws, prefix <> "pooler.dense.weight"), do: :tanh_dense
      cfg = %Config{arch: "vit", rows: n + 1, row_width: ch * p * p, width: d, layers: l, heads: h, intermediate: ff,
                    eps: (c["layer_norm_eps"] || 1.0e-12) * 1.0, norm: :layer, act: act, cls: true, labels: labels,
                    names: :vit, prefix: prefix, modality: :image, image: %{size: img, patch: p, channels: ch},
                    pooler: pooler, raw: Map.drop(c, ~w(torch_dtype dtype transformers_version))}

      with :ok <- shapes(cfg), do: {:ok, spec(cfg), ws}
    end
  end

  # CLIP's vision tower (CLIPVisionModel / CLIPVisionModelWithProjection, or
  # the vision half of a CLIPModel under `vision_model.`): [CLS] from
  # class_embedding, no patch bias, LayerNorm *before* the layers
  # (pre_layrnorm), the final LayerNorm on the pooled [CLS] row only
  # (post_layernorm), quick_gelu, and the optional visual_projection
  def admit(%{config: %{"model_type" => "clip_vision_model"} = c}, ws, _opts) do
    with {:ok, [d, l, h, ff, img, p, ch]} <- ints(c, ~w(hidden_size num_hidden_layers num_attention_heads intermediate_size image_size patch_size num_channels), %{"num_channels" => 3}),
         :ok <- need(rem(img, p) == 0, "image_size", "a multiple of patch_size"),
         {:ok, act} <- activation(c["hidden_act"] || "quick_gelu"),
         n = div(img, p) * div(img, p) do
      proj = case ws["visual_projection.weight"] do
        %Tensor{shape: [pd, ^d]} -> pd
        _ -> nil
      end

      cfg = %Config{arch: "clip_vision", rows: n + 1, row_width: ch * p * p, width: d, layers: l, heads: h, intermediate: ff,
                    eps: (c["layer_norm_eps"] || 1.0e-5) * 1.0, norm: :layer, act: act, cls: true, labels: nil,
                    names: :clip, prefix: "vision_model.", modality: :image, image: %{size: img, patch: p, channels: ch},
                    pre_norm: true, final_norm: :pooled, embed_bias: false, pooler: :row0, projection: proj,
                    raw: Map.drop(c, ~w(torch_dtype dtype transformers_version))}

      with :ok <- shapes(cfg), do: {:ok, spec(cfg), ws}
    end
  end

  # CLIP's text tower (CLIPTextModel / CLIPTextModelWithProjection, or the
  # text half of a CLIPModel): token and learned position embeddings, causal
  # pre-norm layers (quick_gelu), the final LayerNorm over every row, the
  # pooled row at the end-of-text token, the optional text_projection
  def admit(%{config: %{"model_type" => "clip_text_model"} = c}, ws, _opts) do
    with {:ok, [v, d, l, h, ff, t]} <- ints(c, ~w(vocab_size hidden_size num_hidden_layers num_attention_heads intermediate_size max_position_embeddings), %{}),
         {:ok, act} <- activation(c["hidden_act"] || "quick_gelu") do
      prefix = if Map.has_key?(ws, "text_model.embeddings.token_embedding.weight"), do: "text_model.", else: ""

      proj = case ws["text_projection.weight"] do
        %Tensor{shape: [pd, ^d]} -> pd
        _ -> nil
      end

      cfg = %Config{arch: "clip_text", rows: t, row_width: d, width: d, layers: l, heads: h, intermediate: ff,
                    eps: (c["layer_norm_eps"] || 1.0e-5) * 1.0, norm: :layer, act: act, cls: false, labels: nil,
                    names: :clip_text, prefix: prefix, modality: :text, embed: :tokens, vocab: v, embed_bias: false,
                    # transformers: eos_token_id 2 (the pre-#24773 configs) pools at
                    # the largest id, otherwise at the first eos_token_id
                    eos: (if c["eos_token_id"] in [nil, 2], do: :argmax, else: c["eos_token_id"]),
                    pooler: :pick, projection: proj, raw: Map.drop(c, ~w(torch_dtype dtype transformers_version))}

      with :ok <- shapes(cfg), do: {:ok, spec(cfg), ws}
    end
  end

  # a CLIPModel holds both towers: admitted one at a time (`tower:`)
  def admit(%{config: %{"model_type" => "clip"} = c} = m, ws, opts) do
    case Keyword.get(opts, :tower) do
      t when t in [:text, "text"] ->
        admit(%{m | config: Map.merge(c["text_config"] || %{}, %{"model_type" => "clip_text_model"})}, ws, opts)

      t when t in [:vision, "vision"] ->
        admit(%{m | config: Map.merge(c["vision_config"] || %{}, %{"model_type" => "clip_vision_model"})}, ws, opts)

      _ ->
        {:error, Rejection.new({:config, "model_type"}, "one tower of a CLIPModel", "open it with tower: :text or tower: :vision")}
    end
  end

  def admit(%{config: %{"model_type" => "vapor_encoder"} = c}, ws, _opts) do
    with {:ok, [t, k, d, l, h, ff]} <- ints(c, ~w(rows row_width hidden_size num_hidden_layers num_attention_heads intermediate_size), %{}),
         {:ok, act} <- activation(c["hidden_act"] || "gelu"),
         {:ok, norm} <- (case c["norm"] || "layer" do
           "layer" -> {:ok, :layer}
           "rms" -> {:ok, :rms}
           x -> {:error, Rejection.new({:config, "norm"}, "layer or rms (got #{inspect(x)})", "check the config")}
         end),
         modality = c["modality"] || "rows",
         :ok <- need(modality in ~w(rows image audio text_image), "modality", "rows, image, audio or text_image"),
         :ok <- need(c["head"] in [nil, "pooled", "rows"], "head", "pooled or rows"),
         :ok <- need(c["head"] != "rows" or is_integer(c["num_labels"]), "num_labels", "an integer when head is rows") do
      cfg = %Config{arch: "vapor_encoder", rows: t, row_width: k, width: d, layers: l, heads: h, intermediate: ff,
                    eps: (c["eps"] || 1.0e-6) * 1.0, norm: norm, act: act, cls: c["cls"] == true, labels: c["num_labels"],
                    names: :vapor, prefix: "", modality: String.to_atom(modality), image: c["image"],
                    head_on: if(c["head"] == "rows", do: :rows, else: :pooled),
                    raw: Map.drop(c, ~w(torch_dtype dtype transformers_version))}

      with :ok <- shapes(cfg), do: {:ok, spec(cfg), ws}
    end
  end

  defp activation("gelu"), do: {:ok, :gelu}
  defp activation(a) when a in ["gelu_pytorch_tanh", "gelu_new", "gelu_fast"], do: {:ok, :gelu_tanh}
  defp activation(a) when a in ["silu", "swish"], do: {:ok, :silu}
  defp activation("relu"), do: {:ok, :relu}
  defp activation("quick_gelu"), do: {:ok, :quick_gelu}
  defp activation(a), do: {:error, Rejection.new({:config, "hidden_act"}, "gelu, gelu_pytorch_tanh, quick_gelu, silu or relu (got #{inspect(a)})", "use a supported checkpoint")}

  defp shapes(c) do
    with :ok <- need(rem(c.width, c.heads) == 0, "num_attention_heads", "a divisor of hidden_size"),
         :ok <- need(rem(div(c.width, c.heads), 16) == 0, "head_dim", "hidden_size/num_attention_heads ≡ 0 (mod 16), got #{div(c.width, c.heads)}"),
         :ok <- need(rem(c.width, 16) == 0 and rem(c.intermediate, 16) == 0, "hidden_size/intermediate_size", "multiples of 16"),
         :ok <- need(rem(c.row_width, 16) == 0, "row_width", "a multiple of 16 (patch_size²·channels), got #{c.row_width}") do
      :ok
    end
  end

  @impl true
  def spec(%Config{} = c) do
    %Spec{adapter: __MODULE__, family: c.arch, lineage: [c.arch, "encoder"], interface: :encoder, config: c,
          width: c.width, in_width: if(c.embed == :tokens, do: nil, else: c.row_width), rows: c.rows, max_pos: c.rows,
          vocab: c.vocab,
          features: [:hidden] ++ if(c.labels, do: [if(c.head_on == :rows, do: :row_logits, else: :logits)], else: []) ++ if(c.pooler, do: [:pooled], else: []) ++
                      if(c.projection, do: [:embeds], else: []),
          modality: %{in: [c.modality], out: [:rows] ++ if(c.labels, do: [:labels], else: []) ++ if(c.projection, do: [:embedding], else: [])},
          digest: Vapor.Canonical.hex_digest({:encoder, c.raw})}
  end

  # ---------------------------------------------------------- tensor names --

  @impl true
  def expected(%Spec{config: c}) do
    n = names(c)
    {d, ff} = {c.width, c.intermediate}

    top =
      [{n.embed_w, embed_shape(c), :matrix}, {n.pos, pos_shape(c), :vector}, {n.norm_w, [d], :norm}, {n.norm_b, [d], :bias}] ++
        if(c.embed_bias, do: [{n.embed_b, [d], :bias}], else: []) ++
        if(c.pre_norm, do: [{n.pre_w, [d], :norm}, {n.pre_b, [d], :bias}], else: []) ++
        if(c.cls, do: [{n.cls, cls_shape(c), :vector}], else: []) ++
        if(c.labels, do: [{n.head_w, [c.labels, d], :matrix}, {n.head_b, [c.labels], :bias}], else: []) ++
        if(c.pooler == :tanh_dense, do: [{n.pool_w, [d, d], :matrix}, {n.pool_b, [d], :bias}], else: []) ++
        if(c.projection, do: [{n.proj_w, [c.projection, d], :matrix}], else: [])

    layers =
      for l <- 0..(c.layers - 1), {key, shape, kind} <- [
            {:n1w, [d], :norm}, {:n1b, [d], :bias}, {:n2w, [d], :norm}, {:n2b, [d], :bias},
            {:qw, [d, d], :matrix}, {:qb, [d], :bias}, {:kw, [d, d], :matrix}, {:kb, [d], :bias},
            {:vw, [d, d], :matrix}, {:vb, [d], :bias}, {:ow, [d, d], :matrix}, {:ob, [d], :bias},
            {:f1w, [ff, d], :matrix}, {:f1b, [ff], :bias}, {:f2w, [d, ff], :matrix}, {:f2b, [d], :bias}],
          do: {n.layer.(l, key), shape, kind}

    top ++ layers
  end

  defp embed_shape(%Config{embed: :tokens, width: d, vocab: v}), do: [v, d]
  defp embed_shape(%Config{names: :vit, width: d, image: %{patch: p, channels: ch}}), do: [d, ch, p, p]
  defp embed_shape(%Config{names: :clip, width: d, image: %{patch: p, channels: ch}}), do: [d, ch, p, p]
  defp embed_shape(%Config{width: d, row_width: k}), do: [d, k]
  defp pos_shape(%Config{names: :vit, rows: t, width: d}), do: [1, t, d]
  defp pos_shape(%Config{rows: t, width: d}), do: [t, d]
  defp cls_shape(%Config{names: :vit, width: d}), do: [1, 1, d]
  defp cls_shape(%Config{names: :clip, width: d}), do: [d]
  defp cls_shape(%Config{width: d}), do: [1, d]

  defp names(%Config{names: :vit, prefix: px}) do
    lay = %{n1w: "layernorm_before.weight", n1b: "layernorm_before.bias", n2w: "layernorm_after.weight",
            n2b: "layernorm_after.bias", qw: "attention.attention.query.weight", qb: "attention.attention.query.bias",
            kw: "attention.attention.key.weight", kb: "attention.attention.key.bias",
            vw: "attention.attention.value.weight", vb: "attention.attention.value.bias",
            ow: "attention.output.dense.weight", ob: "attention.output.dense.bias",
            f1w: "intermediate.dense.weight", f1b: "intermediate.dense.bias", f2w: "output.dense.weight", f2b: "output.dense.bias"}

    %{embed_w: px <> "embeddings.patch_embeddings.projection.weight", embed_b: px <> "embeddings.patch_embeddings.projection.bias",
      pos: px <> "embeddings.position_embeddings", cls: px <> "embeddings.cls_token",
      norm_w: px <> "layernorm.weight", norm_b: px <> "layernorm.bias", head_w: "classifier.weight", head_b: "classifier.bias",
      pool_w: px <> "pooler.dense.weight", pool_b: px <> "pooler.dense.bias",
      layer: fn l, key -> px <> "encoder.layer.#{l}." <> Map.fetch!(lay, key) end}
  end

  defp names(%Config{names: :clip, prefix: px}) do
    lay = %{n1w: "layer_norm1.weight", n1b: "layer_norm1.bias", n2w: "layer_norm2.weight", n2b: "layer_norm2.bias",
            qw: "self_attn.q_proj.weight", qb: "self_attn.q_proj.bias", kw: "self_attn.k_proj.weight", kb: "self_attn.k_proj.bias",
            vw: "self_attn.v_proj.weight", vb: "self_attn.v_proj.bias", ow: "self_attn.out_proj.weight", ob: "self_attn.out_proj.bias",
            f1w: "mlp.fc1.weight", f1b: "mlp.fc1.bias", f2w: "mlp.fc2.weight", f2b: "mlp.fc2.bias"}

    %{embed_w: px <> "embeddings.patch_embedding.weight", embed_b: nil, pos: px <> "embeddings.position_embedding.weight",
      cls: px <> "embeddings.class_embedding", pre_w: px <> "pre_layrnorm.weight", pre_b: px <> "pre_layrnorm.bias",
      norm_w: px <> "post_layernorm.weight", norm_b: px <> "post_layernorm.bias", proj_w: "visual_projection.weight",
      layer: fn l, key -> px <> "encoder.layers.#{l}." <> Map.fetch!(lay, key) end}
  end

  defp names(%Config{names: :clip_text, prefix: px}) do
    lay = %{n1w: "layer_norm1.weight", n1b: "layer_norm1.bias", n2w: "layer_norm2.weight", n2b: "layer_norm2.bias",
            qw: "self_attn.q_proj.weight", qb: "self_attn.q_proj.bias", kw: "self_attn.k_proj.weight", kb: "self_attn.k_proj.bias",
            vw: "self_attn.v_proj.weight", vb: "self_attn.v_proj.bias", ow: "self_attn.out_proj.weight", ob: "self_attn.out_proj.bias",
            f1w: "mlp.fc1.weight", f1b: "mlp.fc1.bias", f2w: "mlp.fc2.weight", f2b: "mlp.fc2.bias"}

    %{embed_w: px <> "embeddings.token_embedding.weight", embed_b: nil, pos: px <> "embeddings.position_embedding.weight",
      norm_w: px <> "final_layer_norm.weight", norm_b: px <> "final_layer_norm.bias", proj_w: "text_projection.weight",
      layer: fn l, key -> px <> "encoder.layers.#{l}." <> Map.fetch!(lay, key) end}
  end

  defp names(%Config{names: :vapor}) do
    lay = %{n1w: "norm1.weight", n1b: "norm1.bias", n2w: "norm2.weight", n2b: "norm2.bias", qw: "q.weight", qb: "q.bias",
            kw: "k.weight", kb: "k.bias", vw: "v.weight", vb: "v.bias", ow: "o.weight", ob: "o.bias",
            f1w: "fc1.weight", f1b: "fc1.bias", f2w: "fc2.weight", f2b: "fc2.bias"}

    %{embed_w: "embed.weight", embed_b: "embed.bias", pos: "pos", cls: "cls", norm_w: "norm.weight", norm_b: "norm.bias",
      head_w: "head.weight", head_b: "head.bias", layer: fn l, key -> "layers.#{l}." <> Map.fetch!(lay, key) end}
  end

  # ----------------------------------------------------------------- build --

  @doc """
  Build the encoder program. Options: `storage` (`:f32` | `:bf16` for the
  matrices), `rows` (a program over the first `rows` positions only; at
  most the model's). Inputs `rows : f32[T, k]` (row 0 is ignored when the model
  has a [CLS] row — the codec leaves it zero) and `horizon : s32[T]`.
  """
  @impl true
  def build(%Spec{config: c} = spec, ws, opts) do
    exp = expected(spec)

    with :ok <- present(ws, exp) do
      n = names(c)
      store = Keyword.get(opts, :storage, :f32)
      mat = fn t -> t = flat2(t); if store == :bf16, do: Tensor.to_bf16(t), else: Tensor.widen(t) end
      row = fn t -> t = Tensor.widen(t); Tensor.new(:f32, [1, Enum.product(t.shape)], t.data) end
      # `rows: t` builds a shorter program over the first t positions (a
      # short line or utterance need not pay for the longest one)
      t = Keyword.get(opts, :rows, c.rows)
      true = is_integer(t) and t >= 1 and t <= c.rows
      d = c.width

      {lets, get} = binder()
      {lets, wp} = get.(lets, n.embed_w, mat.(ws[n.embed_w]))
      {lets, pos} = get.(lets, n.pos <> if(t == c.rows, do: "", else: "$#{t}"), Tensor.widen(ws[n.pos]) |> then(&Tensor.new(:f32, [t, d], binary_part(&1.data, 0, t * d * 4))))

      horizon = T.input(:horizon, :s32, [t])

      {x, lets} =
        cond do
          # text: token embeddings (CLIP's text tower)
          c.embed == :tokens ->
            {T.gather_row(wp, T.input(:tok, :s32, [t])), lets}

          c.embed_bias ->
            rows = T.input(:rows, :f32, [t, c.row_width])
            {lets, bp} = get.(lets, n.embed_b, row.(ws[n.embed_b]))
            {T.add(T.linear(rows, wp), bp), lets}

          true ->
            {T.linear(T.input(:rows, :f32, [t, c.row_width]), wp), lets}
        end

      {x, lets} =
        if c.cls do
          {lets, cls} = get.(lets, n.cls, row.(ws[n.cls]))
          mask = Tensor.from_list(:f32, [t, 1], [1.0 | List.duplicate(0.0, t - 1)])
          {lets, m} = get.(lets, "$encoder.cls_mask", mask)
          {T.sel(m, T.splat(0.5), x, cls), lets}
        else
          {x, lets}
        end

      {x, lets} = bind(lets, :x_embed, T.add(x, pos))

      {x, lets} =
        if c.pre_norm do
          {lets, pw} = get.(lets, n.pre_w, row.(ws[n.pre_w]))
          {lets, pb} = get.(lets, n.pre_b, row.(ws[n.pre_b]))
          bind(lets, :x_pre, norm(x, pw, pb, c))
        else
          {x, lets}
        end

      {x, lets} =
        Enum.reduce(0..(c.layers - 1), {x, lets}, fn l, {x, lets} ->
          w = fn key, lets, f -> get.(lets, n.layer.(l, key), f.(ws[n.layer.(l, key)])) end
          {lets, n1w} = w.(:n1w, lets, row)
          {lets, n1b} = w.(:n1b, lets, row)
          {lets, n2w} = w.(:n2w, lets, row)
          {lets, n2b} = w.(:n2b, lets, row)

          {pairs, lets} =
            Enum.map_reduce([{:qw, :qb}, {:kw, :kb}, {:vw, :vb}, {:ow, :ob}, {:f1w, :f1b}, {:f2w, :f2b}], lets, fn {mw, mb}, lets ->
              {lets, a} = w.(mw, lets, mat)
              {lets, b} = w.(mb, lets, row)
              {{a, b}, lets}
            end)

          [{qw, qb}, {kw, kb}, {vw, vb}, {ow, ob}, {f1w, f1b}, {f2w, f2b}] = pairs
          name = &:"layers.#{l}.#{&1}"

          {h, lets} = bind(lets, name.(:attn_in), norm(x, n1w, n1b, c))
          {k, lets} = bind(lets, name.(:k), T.add(T.linear(h, kw), kb))
          {v, lets} = bind(lets, name.(:v), T.add(T.linear(h, vw), vb))
          q = T.add(T.linear(h, qw), qb)
          att = T.attention(q, k, v, horizon, c.heads, c.heads)
          {x, lets} = bind(lets, name.(:attn_out), T.add(x, T.add(T.linear(att, ow), ob)))
          {h2, lets} = bind(lets, name.(:mlp_in), norm(x, n2w, n2b, c))
          f = T.add(T.linear(act(T.add(T.linear(h2, f1w), f1b), c.act), f2w), f2b)
          bind(lets, name.(:mlp_out), T.add(x, f))
        end)

      {lets, nw} = get.(lets, n.norm_w, row.(ws[n.norm_w]))
      {lets, nb} = get.(lets, n.norm_b, row.(ws[n.norm_b]))
      # the pooled row: row 0 ([CLS]), or the row an input names (CLIP text:
      # the end-of-text token's position, `text_input/3`)
      {lets, first} =
        if c.pooler == :pick,
          do: {lets, T.input(:pick, :s32, [1])},
          else: get.(lets, "$encoder.row0", Tensor.from_list(:s32, [1], [0]))

      # ViT: the final norm over every row; CLIP: over the pooled [CLS] row only
      {hidden, row0, lets} =
        case c.final_norm do
          :all ->
            {hidden, lets} = bind(lets, :hidden, norm(x, nw, nb, c))
            {hidden, T.gather_row(hidden, first), lets}

          :pooled ->
            {hidden, lets} = bind(lets, :hidden, x)
            {hidden, norm(T.gather_row(hidden, first), nw, nb, c), lets}
        end

      {head, lets} =
        cond do
          c.labels && c.head_on == :rows ->
            {lets, hw} = get.(lets, n.head_w, mat.(ws[n.head_w]))
            {lets, hb} = get.(lets, n.head_b, row.(ws[n.head_b]))
            {[row_logits: T.add(T.linear(hidden, hw), hb)], lets}

          c.labels ->
            {lets, hw} = get.(lets, n.head_w, mat.(ws[n.head_w]))
            {lets, hb} = get.(lets, n.head_b, row.(ws[n.head_b]))
            {[logits: T.add(T.linear(row0, hw), hb)], lets}

          true ->
            {[], lets}
        end

      {pooled, lets} =
        case c.pooler do
          :tanh_dense ->
            {lets, pw} = get.(lets, n.pool_w, mat.(ws[n.pool_w]))
            {lets, pb} = get.(lets, n.pool_b, row.(ws[n.pool_b]))
            {T.tanh(T.add(T.linear(row0, pw), pb)), lets}

          _ ->
            {row0, lets}
        end

      {proj, lets} =
        if c.projection do
          {lets, pw} = get.(lets, n.proj_w, mat.(ws[n.proj_w]))
          {pooled, lets} = bind(lets, :pooled, pooled)
          {[pooled: pooled, embeds: T.linear(pooled, pw)], lets}
        else
          {if((c.labels && c.head_on == :pooled) || c.pooler, do: [pooled: pooled], else: []), lets}
        end

      {:ok, Program.new([hidden: hidden] ++ head ++ proj, lets: Enum.reverse(lets))}
    end
  end

  # constants bound once by name (weights never inside a term key)
  defp binder do
    get = fn lets, name, %Tensor{} = t ->
      atom = String.to_atom(name)
      {[{atom, T.const(t)} | lets], T.ref(atom, T.const(t))}
    end

    {[], get}
  end

  defp bind(lets, name, term), do: {T.ref(name, term), [{name, term} | lets]}

  # [d, C, p, p] (a convolution whose stride is its kernel) → [d, C·p·p]
  defp flat2(%Tensor{shape: [n | rest]} = t) when length(rest) > 1, do: %{t | shape: [n, Enum.product(rest)]}
  defp flat2(t), do: t

  defp present(ws, exp) do
    case Enum.find(exp, fn {name, shape, _} -> not match?(%Tensor{shape: ^shape}, ws[name]) end) do
      nil -> :ok
      {name, shape, _} ->
        got = case ws[name] do
          %Tensor{shape: s} -> "got #{inspect(s, charlists: :as_lists)}"
          nil -> "missing"
        end

        {:error, Rejection.new({:weight, name}, "#{inspect(shape, charlists: :as_lists)} (#{got})", "check the checkpoint against config.json")}
    end
  end

  # LayerNorm (two-pass, biased variance) or RMSNorm, then weight and bias
  defp norm(x, w, b, %Config{norm: :layer, eps: eps, width: d}) do
    inv = T.splat(1.0 / d)
    mean = T.mul(T.reduce(:sum, x), inv)
    xc = T.sub(x, mean)
    var = T.mul(T.reduce(:sum, T.mul(xc, xc)), inv)
    T.add(T.mul(w, T.mul(xc, T.rsqrt(T.add(var, T.splat(eps))))), b)
  end

  defp norm(x, w, b, %Config{norm: :rms, eps: eps, width: d}) do
    ms = T.mul(T.reduce(:sum, T.mul(x, x)), T.splat(1.0 / d))
    T.add(T.mul(w, T.mul(x, T.rsqrt(T.add(ms, T.splat(eps))))), b)
  end

  defp act(x, :gelu), do: T.gelu(x)
  defp act(x, :gelu_tanh), do: T.gelu_tanh(x)
  defp act(x, :silu), do: T.silu(x)
  defp act(x, :relu), do: T.relu(x)
  # CLIP's quick_gelu: x · σ(1.702 x)
  defp act(x, :quick_gelu), do: T.mul(x, T.sigmoid(T.mul(x, T.splat(1.702))))

  # ------------------------------------------------------------ helpers --

  @doc """
  The environment of an encoder program for real rows `f32[n, k]`: a
  zero [CLS] row in front when the model has one, zero padding up to the
  model's `T` rows, and the horizon that hides the padding.
  """
  def input(spec, rows, t \\ nil)

  def input(%Spec{config: %Config{} = c}, %Tensor{shape: [n, k]} = rows, t) when k == c.row_width do
    t = t || c.rows
    lead = if c.cls, do: 1, else: 0
    real = n + lead
    true = real <= t
    data = :binary.copy(<<0::32>>, lead * k) <> Tensor.widen(rows).data <> :binary.copy(<<0::32>>, (t - real) * k)
    %{rows: Tensor.new(:f32, [t, k], data), horizon: horizon(t, real)}
  end

  @doc """
  The environment of a text tower (CLIP) for token ids (already with the
  start and end-of-text tokens, as the tokenizer's post-processor adds
  them): ids padded to the model's rows with the end-of-text id, the
  causal horizon (row `t` sees rows `0 … t`) and `pick`, the pooled row —
  the first end-of-text token (or the largest id, for configurations that
  still declare `eos_token_id: 2`, as transformers does).
  """
  def text_input(%Spec{config: %Config{embed: :tokens} = c}, ids, t \\ nil) when is_list(ids) and ids != [] do
    t = t || c.rows
    n = length(ids)
    true = n <= t
    pick = case c.eos do
      :argmax -> ids |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1)
      eos -> Enum.find_index(ids, &(&1 == eos)) || n - 1
    end

    pad = if is_integer(c.eos), do: c.eos, else: Enum.max(ids)

    %{tok: Tensor.from_list(:s32, [t], ids ++ List.duplicate(pad, t - n)),
      horizon: Tensor.from_list(:s32, [t], Enum.to_list(0..(t - 1))),
      pick: Tensor.from_list(:s32, [1], [pick])}
  end

  @doc """
  The `horizon` input for `n` real rows of a `t`-row encoder: every row
  sees rows `0 … n − 1` (bidirectional attention over the real rows).
  """
  def horizon(t, n) when n >= 1 and n <= t, do: Tensor.from_list(:s32, [t], List.duplicate(n - 1, t))

  defp ints(c, keys, defaults) do
    vals = Enum.map(keys, &(c[&1] || defaults[&1]))

    case Enum.find(Enum.zip(keys, vals), fn {_, v} -> not (is_integer(v) and v > 0) end) do
      nil -> {:ok, vals}
      {k, _} -> {:error, Rejection.new({:config, k}, "a positive integer", "check the config")}
    end
  end

  defp need(true, _f, _b), do: :ok
  defp need(false, f, b), do: {:error, Rejection.new({:config, f}, b, "use a supported checkpoint")}
end
