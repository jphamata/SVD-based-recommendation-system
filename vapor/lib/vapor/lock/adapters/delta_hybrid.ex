defmodule Vapor.Lock.Adapters.DeltaHybrid do
  @moduledoc """
  Tier 3 of the model airlock: the **delta-rule hybrid** — linear
  attention by a gated delta rule beside latent global attention, with
  residuals that attend over depth and a latent mixture of experts — as a
  causal language model of one token per step whose memory is a recurrent
  state *and* a latent cache.

  The topology is named by what it computes. The family that brought it is
  a spelling: Kimi K3 (Kimi Team, *Kimi K3: Open Frontier Intelligence*,
  2026) is the built-in alias `kimi_k3` (`Vapor.Lock.Alias`), and this
  adapter claims only vapor's own spelling, `model_type:
  "vapor_delta_hybrid"`. docs/AIRLOCK.md says why: the core is mathematics,
  families live in the airlock's data.

  Per module (every layer has two: attention, then the feed-forward), with
  `u = RMSNorm(h)` its normalised input:

      depth     h = Σᵢ softmaxᵢ(w · RMSNorm(vᵢ)) vᵢ          Block Attention Residuals (Eq. 8–10):
                                                          vᵢ = the embedding, each finished block's sum
                                                          of module outputs, the current block's partial sum
      KDA       q, k = L2(silu(conv(u·Wq/k))), v = silu(conv(u·Wv)), β = σ(u·Wb)
                g = g_min · σ(e^A ⊙ (u·Wαᵀ·Wαᵀ + b))  ∈ (g_min, 0),  α = e^g        (Eq. 2, 5)
                S = (I − β k kᵀ) Diag(α) S + β k vᵀ,  o = Sᵀ q                       (Eq. 1)
                y = Wo [σ(u·Wg) ⊙ RMSNorm_head(o)]                                   (Eq. 6)
      MLA       c = RMSNorm(u·Wkv_a)  (cached), q = u·Wq (or the low-rank pair), NoPE
                y = Wo [σ(u·Wg) ⊙ attention(q, k = W_uk c, v = W_uv c)]              (Eq. 7)
      MoE       y = shared(u) + W↑ RMSNorm(Σ_{top-k} pᵢ Eᵢ(W↓ u))                    (Eq. 11)
                s = σ(u·Wr), top-k of s + b (the frozen balancing bias), pᵢ = sᵢ/Σ s (Eq. 13)
      SiTU-GLU  β₁ tanh(g/β₁) ⊙ σ(g) ⊙ β₂ tanh(u/β₂)                               (Eq. 12)

  and the final aggregation over every block, `RMSNorm`, the output head.

  **How it runs.** The KDA state `S` (`f32[heads·d_k, d_v]` per layer) and
  its short-convolution windows are a recurrence, as in Mamba; the MLA
  layers keep a cache of the compressed latent `c` (`f32[context, r]` per
  layer), written at `pos` with `kv_write`. The key map is absorbed into
  the query (`q' = W_ukᵀ q`, one block-diagonal product per head, no
  product of weights computed at build time), so the cache holds `r`
  numbers per token and layer — the point of MLA. Prefill and decoding are
  the same step (`Vapor.Recurrent`), so they give the same bits; the
  context is fixed when the model is opened (`context:`, default 4096).
  Top-k selection is exact (ranks by comparison, ties to the lower index)
  and the experts are dispatched by row predication (`linear_masked`), as
  in the decoder family.

  **What is canonical.** The recurrence of Eq. 1 is the semantics. The
  report's chunkwise form (Eq. 4) computes the same function in another
  order; vapor's reference (`test/python/kimi_k3_reference.py`) checks the
  two agree, and the bounded decay of Eq. 5 is why the chunkwise form
  stays finite. Activations are binary32; the report serves MXFP8.

  **MXFP4 experts.** A routed-expert matrix may arrive as MXFP4 blocks
  and scales; `Vapor.Quant.MXFP4` decodes it exactly (every MXFP4 value is
  a binary32 value).

  **The configuration** is vapor's spelling (`model_type:
  "vapor_delta_hybrid"`, or `kimi_k3` through the alias). Moonshot's
  `config.json` is not reachable from the machine that wrote this adapter:
  the field names follow DeepSeek-V3's where the K3 report says the module
  is DeepSeek's (MLA, the MoE router), and every other one is named here.
  A real K3 checkpoint whose spelling differs is refused with the field
  named; reconciling it is the alias's data, not code. Tensors are checked
  when the program is built (an alias renames them after admission).
  Required: `vocab_size`, `hidden_size`,
  `num_hidden_layers`, `kda_num_heads`, `kda_head_dim`, `kda_alpha_rank`,
  `num_attention_heads`, `kv_lora_rank`, `qk_nope_head_dim`,
  `v_head_dim`, `intermediate_size`, `n_routed_experts`,
  `num_experts_per_tok`, `moe_intermediate_size`, `moe_latent_size`,
  `attn_res_block_size`. Defaulted: `layer_types` (three KDA then one MLA,
  and MLA last — the report's 69 + 24 for 93 layers), `rms_norm_eps`
  (1e-6), `l2norm_eps` (1e-6), `short_conv_kernel_size` (4), `kda_g_min`
  (−5), `q_lora_rank` (none), `first_k_dense_replace` (1),
  `n_shared_experts` (2), `shared_expert_intermediate_size`
  (`moe_intermediate_size`), `routed_scaling_factor` (1), `situ_beta`
  ([4, 25]).

  Refused by name: a rotary part in MLA (`qk_rope_head_dim` > 0 — K3 is
  NoPE; a rotary MLA is `deepseek_v3`), `hidden_act` other than
  `situ_glu`, widths that are not multiples of 16, `num_experts_per_tok`
  outside `1 … n_routed_experts`. Not read: the vision tower (MoonViT-V2)
  and the multi-token-prediction layer; named in `docs/KIMI.md`, with
  what remains owed.
  """
  @behaviour Vapor.Lock.Adapter
  alias Vapor.{F32, Program, Rejection, Spatial, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Lock.Spec
  alias Vapor.Quant.MXFP4

  defmodule Config do
    @moduledoc "An admitted delta-rule hybrid configuration."
    defstruct [:vocab, :hidden, :layers, :types, :eps, :l2_eps, :tie, :bos, :eos, :context, :block,
               :kda, :mla, :dense, :moe, :beta, :raw]
  end

  @impl true
  def id, do: "delta_hybrid"

  @impl true
  def claim(%{config: %{"model_type" => "vapor_delta_hybrid"}}), do: {:claim, 100}
  def claim(_), do: :no

  @impl true
  def owns?(%Config{}), do: true
  def owns?(_), do: false

  # ------------------------------------------------------------ admission --

  @impl true
  def admit(%{config: c}, ws, opts) do
    with {:ok, ws} <- MXFP4.expand(ws),
         {:ok, cfg} <- config(c, ws, opts),
         do: {:ok, spec(cfg), ws}
  end

  # every tensor the program reads, with its shape
  defp shapes(spec, ws) do
    case Enum.find(expected(spec), fn {n, shape, _} -> not match?(%Tensor{shape: ^shape}, ws[n]) end) do
      nil -> :ok
      {n, shape, _} -> {:error, Rejection.new({:weight, n}, "#{inspect(shape, charlists: :as_lists)} (got #{got(ws[n])})", "check the checkpoint against config.json")}
    end
  end

  defp got(%Tensor{dtype: d, shape: s}), do: "#{d}#{inspect(s, charlists: :as_lists)}"
  defp got(nil), do: "absent"

  defp config(c, ws, opts) do
    with {:ok, [v, d, l]} <- ints(c, ~w(vocab_size hidden_size num_hidden_layers)),
         {:ok, [kh, kd, kr]} <- ints(c, ~w(kda_num_heads kda_head_dim kda_alpha_rank)),
         {:ok, [mh, rkv, dqk, mdv]} <- ints(c, ~w(num_attention_heads kv_lora_rank qk_nope_head_dim v_head_dim)),
         {:ok, [inter, e, k, ie, lat, blk]} <-
           ints(c, ~w(intermediate_size n_routed_experts num_experts_per_tok moe_intermediate_size moe_latent_size attn_res_block_size)),
         {:ok, types} <- types(c["layer_types"], l),
         :ok <- need(c["qk_rope_head_dim"] in [nil, 0], "qk_rope_head_dim", "0 (this topology's latent attention has no rotary part; a rotary MLA is the decoder's deepseek_v3)"),
         :ok <- need(c["hidden_act"] in [nil, "situ_glu"], "hidden_act", "situ_glu (got #{inspect(c["hidden_act"])})"),
         :ok <- need(k >= 1 and k <= e, "num_experts_per_tok", "1 … n_routed_experts (#{e})"),
         :ok <- need(c["q_lora_rank"] == nil or (is_integer(c["q_lora_rank"]) and c["q_lora_rank"] > 0), "q_lora_rank", "null or a positive integer"),
         {:ok, beta} <- beta(c["situ_beta"]) do
      rq = c["q_lora_rank"]
      ns = c["n_shared_experts"] || 2
      is = c["shared_expert_intermediate_size"] || ie
      context = Keyword.get(opts, :context) || c["vapor_context"] || 4096
      widths = [hidden_size: d, kda_head_dim: kd, kda_alpha_rank: kr, kv_lora_rank: rkv, qk_nope_head_dim: dqk, v_head_dim: mdv,
                intermediate_size: inter, moe_intermediate_size: ie, moe_latent_size: lat, shared_expert_intermediate_size: is,
                n_routed_experts: e] ++ if(rq, do: [q_lora_rank: rq], else: [])

      case Enum.find(widths, fn {_, w} -> rem(w, 16) != 0 end) do
        {f, w} ->
          no("#{f}", "a multiple of 16 (got #{w})")

        nil ->
          {:ok,
           %Config{vocab: v, hidden: d, layers: l, types: types, eps: num(c["rms_norm_eps"], 1.0e-6), l2_eps: num(c["l2norm_eps"], 1.0e-6),
                   tie: not Map.has_key?(ws, "lm_head.weight") and c["tie_word_embeddings"] == true,
                   bos: c["bos_token_id"], eos: c["eos_token_id"], context: context, block: blk, beta: beta,
                   kda: %{heads: kh, dk: kd, dv: kd, rank: kr, conv: c["short_conv_kernel_size"] || 4, g_min: num(c["kda_g_min"], -5.0)},
                   mla: %{heads: mh, q_lora: rq, kv_lora: rkv, dqk: dqk, dv: mdv},
                   dense: %{first: c["first_k_dense_replace"] || 1, inter: inter},
                   moe: %{experts: e, top_k: k, inter: ie, latent: lat, shared: ns * is, scale: num(c["routed_scaling_factor"], 1.0)},
                   raw: c}}
      end
    end
  end

  defp ints(c, keys) do
    case Enum.find(keys, fn k -> not (is_integer(c[k]) and c[k] > 0) end) do
      nil -> {:ok, Enum.map(keys, &c[&1])}
      k -> no(k, "a positive integer (got #{inspect(c[k])})")
    end
  end

  # the report's pattern: three KDA layers, then one Gated MLA; the last layer is MLA
  defp types(nil, l), do: {:ok, for(i <- 0..(l - 1), do: if(rem(i, 4) == 3 or i == l - 1, do: :mla, else: :kda))}

  defp types(list, l) when is_list(list) and length(list) == l do
    if Enum.all?(list, &(&1 in ["kda", "mla"])),
      do: {:ok, Enum.map(list, &String.to_existing_atom/1)},
      else: no("layer_types", "a list of \"kda\" and \"mla\"")
  end

  defp types(other, l), do: no("layer_types", "#{l} entries of \"kda\" or \"mla\" (got #{inspect(other)})")

  defp beta(nil), do: {:ok, {4.0, 25.0}}
  defp beta([a, b]) when is_number(a) and is_number(b) and a > 0 and b > 0, do: {:ok, {a * 1.0, b * 1.0}}
  defp beta(other), do: no("situ_beta", "[β₁ > 0, β₂ > 0] (got #{inspect(other)})")

  defp num(nil, d), do: d
  defp num(x, _d) when is_number(x), do: x * 1.0

  defp need(true, _f, _b), do: :ok
  defp need(false, f, b), do: no(f, b)

  defp no(f, b), do: {:error, Rejection.new({:config, f}, b, "use a supported checkpoint, or an alias that maps its spelling")}

  @impl true
  def spec(%Config{} = c) do
    %Spec{adapter: __MODULE__, family: "vapor_delta_hybrid", lineage: ["vapor_delta_hybrid"], interface: :causal_lm, config: c,
          vocab: c.vocab, width: c.hidden, max_pos: c.context, bos: c.bos, eos: c.eos |> List.wrap() |> Enum.reject(&is_nil/1),
          features: [:recurrent, :hidden], modality: %{in: [:text], out: [:text]},
          digest: Vapor.Canonical.hex_digest({:delta_hybrid, c.raw, c.context})}
  end

  @impl true
  def expected(%Spec{config: c}) do
    d = c.hidden
    %{heads: kh, dk: dk, dv: dv, rank: kr, conv: kc} = c.kda
    %{heads: mh, q_lora: rq, kv_lora: rkv, dqk: dqk, dv: mdv} = c.mla
    %{experts: e, inter: ie, latent: lat, shared: is} = c.moe

    layers =
      for {t, l} <- Enum.with_index(c.types) do
        p = "model.layers.#{l}."
        a = p <> "self_attn."
        m = p <> "mlp."

        attn =
          case t do
            :kda ->
              [{a <> "q_proj.weight", [kh * dk, d], :matrix}, {a <> "k_proj.weight", [kh * dk, d], :matrix}, {a <> "v_proj.weight", [kh * dv, d], :matrix},
               {a <> "q_conv1d.weight", [kh * dk, 1, kc], :matrix}, {a <> "k_conv1d.weight", [kh * dk, 1, kc], :matrix},
               {a <> "v_conv1d.weight", [kh * dv, 1, kc], :matrix}, {a <> "b_proj.weight", [kh, d], :matrix},
               {a <> "f_a_proj.weight", [kr, d], :matrix}, {a <> "f_b_proj.weight", [kh * dk, kr], :matrix},
               {a <> "dt_bias", [kh * dk], :bias}, {a <> "A_log", [kh], :vector}, {a <> "g_proj.weight", [kh * dv, d], :matrix},
               {a <> "o_norm.weight", [dv], :norm}, {a <> "o_proj.weight", [d, kh * dv], :matrix}]

            :mla ->
              q = if rq, do: [{a <> "q_a_proj.weight", [rq, d], :matrix}, {a <> "q_a_layernorm.weight", [rq], :norm}, {a <> "q_b_proj.weight", [mh * dqk, rq], :matrix}],
                         else: [{a <> "q_proj.weight", [mh * dqk, d], :matrix}]

              q ++ [{a <> "kv_a_proj_with_mqa.weight", [rkv, d], :matrix}, {a <> "kv_a_layernorm.weight", [rkv], :norm},
                    {a <> "kv_b_proj.weight", [mh * (dqk + mdv), rkv], :matrix}, {a <> "g_proj.weight", [mh * mdv, d], :matrix},
                    {a <> "o_proj.weight", [d, mh * mdv], :matrix}]
          end

        ffn =
          if l < c.dense.first do
            n = c.dense.inter
            [{m <> "gate_proj.weight", [n, d], :matrix}, {m <> "up_proj.weight", [n, d], :matrix}, {m <> "down_proj.weight", [d, n], :matrix}]
          else
            [{m <> "gate.weight", [e, d], :matrix}, {m <> "gate.e_score_correction_bias", [e], :bias},
             {m <> "latent_down.weight", [lat, d], :matrix}, {m <> "latent_norm.weight", [lat], :norm}, {m <> "latent_up.weight", [d, lat], :matrix},
             {m <> "shared_experts.gate_proj.weight", [is, d], :matrix}, {m <> "shared_experts.up_proj.weight", [is, d], :matrix},
             {m <> "shared_experts.down_proj.weight", [d, is], :matrix}] ++
              for(j <- 0..(e - 1), x = m <> "experts.#{j}.",
                  do: [{x <> "gate_proj.weight", [ie, lat], :matrix}, {x <> "up_proj.weight", [ie, lat], :matrix}, {x <> "down_proj.weight", [lat, ie], :matrix}])
              |> List.flatten()
          end

        [{p <> "input_layernorm.weight", [d], :norm}, {p <> "post_attention_layernorm.weight", [d], :norm},
         {p <> "attn_res.attn_query", [d], :vector}, {p <> "attn_res.ffn_query", [d], :vector}] ++ attn ++ ffn
      end

    [{"model.embed_tokens.weight", [c.vocab, d], :embedding}, {"model.norm.weight", [d], :norm}, {"model.attn_res.final_query", [d], :vector}] ++
      List.flatten(layers) ++ if(c.tie, do: [], else: [{"lm_head.weight", [c.vocab, d], :matrix}])
  end

  # ----------------------------------------------------------------- state --

  @doc "The state's names and shapes: `kda{l}` and the convolution windows of KDA layers, `lat{l}` (the latent cache) of MLA layers."
  def state(%Config{} = c) do
    %{heads: h, dk: dk, dv: dv, conv: k} = c.kda

    for {t, l} <- Enum.with_index(c.types) do
      case t do
        :kda -> [{:"kda#{l}", [h * dk, dv]} | for({x, w} <- [q: h * dk, k: h * dk, v: h * dv], j <- 0..(k - 2)//1, do: {:"conv#{l}_#{x}_#{j}", [1, w]})]
        :mla -> [{:"lat#{l}", [c.context, c.mla.kv_lora]}]
      end
    end
    |> List.flatten()
  end

  @doc "A zero state (the start of every sequence)."
  def empty_state(%Config{} = c),
    do: Map.new(state(c), fn {n, shape} -> {n, Tensor.new(:f32, shape, :binary.copy(<<0::32>>, Enum.product(shape)))} end)

  # ----------------------------------------------------------------- build --

  @impl true
  def build(%Spec{config: c} = spec, ws, _opts) do
    with :ok <- shapes(spec, ws), do: program(c, ws)
  end

  defp program(c, ws) do
    get = fn name -> Tensor.widen(ws[name]) end
    {lets, embed} = Spatial.bind([], "model.embed_tokens.weight", get.("model.embed_tokens.weight"))
    ctx = %{c: c, get: get, pos: T.input(:pos, :s32, [1])}
    {lets, emb} = source(lets, T.gather_row(embed, T.input(:tok, :s32, [1])), "emb", c.eps)

    {done, nil, lets, nexts} =
      Enum.reduce(Enum.with_index(c.types), {[emb], nil, lets, []}, fn {t, l}, {done, partial, lets, nexts} ->
        p = "model.layers.#{l}."

        {partial, lets, nexts} =
          Enum.reduce([:attn, :ffn], {partial, lets, nexts}, fn kind, {partial, lets, nexts} ->
            {lets, q} = Spatial.bind(lets, p <> "attn_res.#{kind}_query", row(get.(p <> "attn_res.#{kind}_query")))
            {lets, h} = depth(lets, done ++ List.wrap(partial), q, "#{p}#{kind}_res")
            norm = if kind == :attn, do: "input_layernorm", else: "post_attention_layernorm"
            {lets, nw} = Spatial.bind(lets, p <> norm, row(get.(p <> norm <> ".weight")))
            {lets, u} = Spatial.name(lets, "#{p}#{kind}_in", rms(h, nw, c.eps))

            {lets, y, nx} =
              case {kind, t} do
                {:attn, :kda} -> kda(lets, u, l, ctx)
                {:attn, :mla} -> mla(lets, u, l, ctx)
                {:ffn, _} -> {lets, y} = ffn(lets, u, l, ctx); {lets, y, []}
              end

            {lets, partial} = source(lets, if(partial, do: T.add(elem(partial, 0), y), else: y), "#{p}#{kind}_sum", c.eps)
            {partial, lets, nexts ++ nx}
          end)

        if rem(l + 1, c.block) == 0 or l == c.layers - 1, do: {done ++ [partial], nil, lets, nexts}, else: {done, partial, lets, nexts}
      end)

    {lets, qf} = Spatial.bind(lets, "model.attn_res.final_query", row(get.("model.attn_res.final_query")))
    {lets, h} = depth(lets, done, qf, "final_res")
    {lets, nf} = Spatial.bind(lets, "model.norm", row(get.("model.norm.weight")))
    {lets, hid} = Spatial.name(lets, "hidden_v", rms(h, nf, c.eps))
    {lets, head} = if c.tie, do: {lets, embed}, else: Spatial.bind(lets, "lm_head", get.("lm_head.weight"))

    outputs = [logits: T.linear(hid, head), hidden: hid] ++ nexts
    state = for {name, _} <- state(c), do: {name, :"#{name}_next"}
    {:ok, Program.new(outputs, state: state, lets: Enum.reverse(lets))}
  end

  # a source of the depth attention: its value and its key RMSNorm(v), named once
  defp source(lets, v, name, eps) do
    {lets, v} = Spatial.name(lets, name, v)
    {lets, k} = Spatial.name(lets, name <> "_key", rms(v, nil, eps))
    {lets, {v, k}}
  end

  # Eq. 9: h = Σ softmax(q·RMSNorm(vᵢ)) vᵢ; a single source is itself (weight exactly 1)
  defp depth(lets, [{v, _}], _q, _name), do: {lets, v}

  defp depth(lets, sources, q, name) do
    named = fn lets, n, t -> {lets, r} = Spatial.name(lets, n, t); {r, lets} end

    {scores, lets} =
      sources
      |> Enum.with_index()
      |> Enum.map_reduce(lets, fn {{_, k}, i}, lets -> named.(lets, "#{name}_s#{i}", T.reduce(:sum, T.mul(k, q))) end)

    {lets, mx} = Spatial.name(lets, "#{name}_max", Enum.reduce(tl(scores), hd(scores), &T.max(&2, &1)))
    {es, lets} = scores |> Enum.with_index() |> Enum.map_reduce(lets, fn {s, i}, lets -> named.(lets, "#{name}_e#{i}", T.exp(T.sub(s, mx))) end)
    {lets, den} = Spatial.name(lets, "#{name}_den", Enum.reduce(tl(es), hd(es), &T.add(&2, &1)))

    sum =
      Enum.zip(sources, es)
      |> Enum.map(fn {{v, _}, e} -> T.mul(v, T.divide(e, den)) end)
      |> then(fn [t | ts] -> Enum.reduce(ts, t, &T.add(&2, &1)) end)

    Spatial.name(lets, name, sum)
  end

  # ------------------------------------------------------------------ KDA --

  defp kda(lets, u, l, %{c: c, get: get}) do
    %{heads: h, dk: dk, dv: dv, conv: k, g_min: gmin} = c.kda
    a = "model.layers.#{l}.self_attn."
    b = fn lets, name, t -> Spatial.bind(lets, a <> name, t) end

    # projection, causal depthwise convolution (k − 1 past inputs are state), silu
    {lets, conv, nexts} =
      Enum.reduce([q: h * dk, k: h * dk, v: h * dv], {lets, %{}, []}, fn {x, w}, {lets, acc, nexts} ->
        {lets, wp} = b.(lets, "#{x}_proj", get.(a <> "#{x}_proj.weight"))
        {lets, cur} = Spatial.name(lets, "#{a}#{x}_in", T.linear(u, wp))
        cw = get.(a <> "#{x}_conv1d.weight") |> Tensor.to_floats() |> List.to_tuple()
        inputs = (for j <- 0..(k - 2)//1, do: T.input(:"conv#{l}_#{x}_#{j}", :f32, [1, w])) ++ [cur]

        {lets, terms} =
          Enum.reduce(Enum.with_index(inputs), {lets, []}, fn {inp, j}, {lets, ts} ->
            {lets, wr} = b.(lets, "#{x}_conv_w#{j}", Tensor.from_list(:f32, [1, w], for(ch <- 0..(w - 1), do: elem(cw, ch * k + j))))
            {lets, ts ++ [T.mul(inp, wr)]}
          end)

        {lets, y} = Spatial.name(lets, "#{a}#{x}_conv", T.silu(Enum.reduce(tl(terms), hd(terms), &T.add(&2, &1))))
        nx = for j <- 0..(k - 2)//1, do: {:"conv#{l}_#{x}_#{j}_next", Enum.at(inputs, j + 1)}
        {lets, Map.put(acc, x, y), nexts ++ nx}
      end)

    {lets, q} = Spatial.name(lets, "#{a}q", l2(conv.q, h, dk, c.l2_eps))
    {lets, kk} = Spatial.name(lets, "#{a}k", l2(conv.k, h, dk, c.l2_eps))

    # β per head, α per key channel: g = g_min·σ(e^A ⊙ z) ∈ (g_min, 0)
    {lets, wb} = b.(lets, "b_proj", get.(a <> "b_proj.weight"))
    {lets, beta} = Spatial.name(lets, "#{a}beta", T.transpose(T.sigmoid(T.linear(u, wb))))
    {lets, fa} = b.(lets, "f_a_proj", get.(a <> "f_a_proj.weight"))
    {lets, fb} = b.(lets, "f_b_proj", get.(a <> "f_b_proj.weight"))
    {lets, dtb} = b.(lets, "dt_bias", row(get.(a <> "dt_bias")))
    ea = get.(a <> "A_log") |> Tensor.to_floats() |> Enum.map(&F32.to_float(F32.from_float(Vapor.CR.exp_f64(&1))))
    {lets, ea} = b.(lets, "exp_A", Tensor.from_list(:f32, [1, h * dk], for(r <- 0..(h * dk - 1), do: Enum.at(ea, div(r, dk)))))
    z = T.add(T.linear(T.linear(u, fa), fb), dtb)
    {lets, alpha} = Spatial.name(lets, "#{a}alpha", T.transpose(T.exp(T.mul(T.splat(gmin), T.sigmoid(T.mul(ea, z))))))

    # S = Diag(α)S;  Δ = β (v − kᵀ S) per head;  S += k Δ;  o = Sᵀ q
    heads = T.const(Tensor.from_list(:s32, [h * dk], for(r <- 0..(h * dk - 1), do: div(r, dk))))
    s_in = T.input(:"kda#{l}", :f32, [h * dk, dv])
    {lets, as} = Spatial.name(lets, "#{a}decayed", T.mul(s_in, alpha))
    {lets, kc} = Spatial.name(lets, "#{a}k_col", T.transpose(kk))
    ks = head_dot(as, kc, h, dk, dv)
    delta = T.mul(beta, T.sub(T.reshape(conv.v, [h, dv]), ks))
    {lets, delta} = Spatial.name(lets, "#{a}delta", delta)
    {lets, s} = Spatial.name(lets, "kda#{l}_next_v", T.add(as, T.mul(kc, T.gather_row(delta, heads))))
    o = head_dot(s, T.transpose(q), h, dk, dv)

    # head-wise RMSNorm, the full-rank sigmoid gate, the output projection
    {lets, onw} = b.(lets, "o_norm", row(get.(a <> "o_norm.weight")))
    on = T.reshape(rms(o, onw, c.eps), [1, h * dv])
    {lets, wg} = b.(lets, "g_proj", get.(a <> "g_proj.weight"))
    {lets, wo} = b.(lets, "o_proj", get.(a <> "o_proj.weight"))
    {lets, y} = Spatial.name(lets, "#{a}out", T.linear(T.mul(T.sigmoid(T.linear(u, wg)), on), wo))
    {lets, y, nexts ++ [{:"kda#{l}_next", s}]}
  end

  # per head: (m ⊙ col) summed over the head's d_k rows, as f32[h, d_v]
  defp head_dot(m, col, h, dk, dv) do
    m |> T.mul(col) |> T.transpose() |> T.reshape([dv * h, dk]) |> then(&T.reduce(:sum, &1)) |> T.reshape([dv, h]) |> T.transpose()
  end

  # L2 normalisation of every head's d_k channels: x · rsqrt(Σx² + ε)
  defp l2(x, h, dk, eps) do
    g = T.reshape(x, [h, dk])
    T.reshape(T.mul(g, T.rsqrt(T.add(T.reduce(:sum, T.mul(g, g)), T.splat(eps)))), [1, h * dk])
  end

  # ------------------------------------------------------------------ MLA --

  defp mla(lets, u, l, %{c: c, get: get, pos: pos}) do
    %{heads: h, q_lora: rq, kv_lora: r, dqk: dqk, dv: dv} = c.mla
    a = "model.layers.#{l}.self_attn."
    b = fn lets, name, t -> Spatial.bind(lets, a <> name, t) end

    {lets, q} =
      if rq do
        {lets, qa} = b.(lets, "q_a_proj", get.(a <> "q_a_proj.weight"))
        {lets, qn} = b.(lets, "q_a_layernorm", row(get.(a <> "q_a_layernorm.weight")))
        {lets, qb} = b.(lets, "q_b_proj", get.(a <> "q_b_proj.weight"))
        {lets, T.linear(rms(T.linear(u, qa), qn, c.eps), qb)}
      else
        {lets, wq} = b.(lets, "q_proj", get.(a <> "q_proj.weight"))
        {lets, T.linear(u, wq)}
      end

    {lets, q} = Spatial.name(lets, "#{a}q", q)
    {lets, kva} = b.(lets, "kv_a_proj", get.(a <> "kv_a_proj_with_mqa.weight"))
    {lets, kvn} = b.(lets, "kv_a_layernorm", row(get.(a <> "kv_a_layernorm.weight")))
    {lets, lat} = Spatial.name(lets, "#{a}latent", rms(T.linear(u, kva), kvn, c.eps))
    cache = T.kv_write(T.input(:"lat#{l}", :f32, [c.context, r]), pos, lat)
    {lets, cache} = Spatial.name(lets, "lat#{l}_next_v", cache)

    # kv_b_proj, per head: d_qk key rows then d_v value rows over the latent
    kvb = get.(a <> "kv_b_proj.weight") |> Tensor.to_floats() |> List.to_tuple()
    at = fn row, col -> elem(kvb, row * r + col) end
    uk_t = for i <- 0..(h - 1), j <- 0..(r - 1), t <- 0..(dqk - 1), do: at.(i * (dqk + dv) + t, j)
    uv = for i <- 0..(h - 1), j <- 0..(dv - 1), t <- 0..(r - 1), do: at.(i * (dqk + dv) + dqk + j, t)
    {lets, wuk} = b.(lets, "kv_b_proj.absorbed_key", Tensor.from_list(:f32, [h * r, dqk], uk_t))
    {lets, wuv} = b.(lets, "kv_b_proj.value", Tensor.from_list(:f32, [h * dv, r], uv))

    qa = T.linear_grouped(q, wuk, h)
    o = T.attention(qa, cache, cache, pos, h, 1, Vapor.CR.pow_f64(dqk * 1.0, -0.5))
    {lets, o} = Spatial.name(lets, "#{a}attn", T.linear_grouped(o, wuv, h))
    {lets, wg} = b.(lets, "g_proj", get.(a <> "g_proj.weight"))
    {lets, wo} = b.(lets, "o_proj", get.(a <> "o_proj.weight"))
    {lets, y} = Spatial.name(lets, "#{a}out", T.linear(T.mul(T.sigmoid(T.linear(u, wg)), o), wo))
    {lets, y, [{:"lat#{l}_next", cache}]}
  end

  # ------------------------------------------------------------------ FFN --

  defp ffn(lets, u, l, %{c: c, get: get}) do
    m = "model.layers.#{l}.mlp."
    b = fn lets, name -> Spatial.bind(lets, m <> name, get.(m <> name <> ".weight")) end

    if l < c.dense.first do
      {lets, wg} = b.(lets, "gate_proj")
      {lets, wu} = b.(lets, "up_proj")
      {lets, wd} = b.(lets, "down_proj")
      Spatial.name(lets, "#{m}out", T.linear(situ(T.linear(u, wg), T.linear(u, wu), c.beta), wd))
    else
      moe(lets, u, m, c, get, b)
    end
  end

  defp moe(lets, u, m, c, get, b) do
    %{experts: e, top_k: k, scale: scale} = c.moe
    {lets, wr} = b.(lets, "gate")
    {lets, bias} = Spatial.bind(lets, m <> "gate.e_score_correction_bias", row(get.(m <> "gate.e_score_correction_bias")))
    {lets, s} = Spatial.name(lets, "#{m}router_s", T.sigmoid(T.linear(u, wr)))
    {lets, ch} = Spatial.name(lets, "#{m}router_choice", T.add(s, bias))
    kk = T.splat(k * 1.0)

    # expert i is chosen iff fewer than k experts beat it (a higher choice, or an equal one at a lower index)
    {picks, lets} =
      Enum.map_reduce(0..(e - 1), lets, fn i, lets ->
        oh = T.const(Tensor.from_list(:f32, [1, e], for(j <- 0..(e - 1), do: if(j == i, do: 1.0, else: 0.0))))
        tie = T.const(Tensor.from_list(:f32, [1, e], for(j <- 0..(e - 1), do: if(j < i, do: 1.0, else: 0.0))))
        col = T.linear(ch, oh)
        count = T.sel(col, ch, T.splat(1.0), T.sel(ch, col, T.splat(0.0), tie))
        {lets, rank} = Spatial.name(lets, "#{m}rank#{i}", T.reduce(:sum, count))
        {lets, sc} = Spatial.name(lets, "#{m}score#{i}", T.linear(s, oh))
        {{rank, sc}, lets}
      end)

    den = picks |> Enum.map(fn {r, sc} -> T.sel(r, kk, sc, T.splat(0.0)) end) |> then(fn [t | ts] -> Enum.reduce(ts, t, &T.add(&2, &1)) end)
    {lets, den} = Spatial.name(lets, "#{m}router_den", den)
    {lets, wdown} = b.(lets, "latent_down")
    {lets, z} = Spatial.name(lets, "#{m}latent", T.linear(u, wdown))

    {acc, lets} =
      picks
      |> Enum.with_index()
      |> Enum.reduce({nil, lets}, fn {{r, sc}, i}, {acc, lets} ->
        x = "experts.#{i}."
        {lets, wg} = b.(lets, x <> "gate_proj")
        {lets, wu} = b.(lets, x <> "up_proj")
        {lets, wd} = b.(lets, x <> "down_proj")
        {lets, mk} = Spatial.name(lets, "#{m}mask#{i}", T.sel(r, kk, T.splat(1.0), T.splat(0.0)))
        y = T.linear_masked(situ(T.linear_masked(z, wg, mk), T.linear_masked(z, wu, mk), c.beta), wd, mk)
        g = T.divide(sc, den)
        g = if scale != 1.0, do: T.mul(g, T.splat(scale)), else: g
        contrib = T.mul(y, g)
        next = if acc, do: T.sel(r, kk, T.add(acc, contrib), acc), else: T.sel(r, kk, contrib, T.splat(0.0))
        {lets, next} = Spatial.name(lets, "#{m}acc#{i}", next)
        {next, lets}
      end)

    {lets, ln} = Spatial.bind(lets, m <> "latent_norm", row(get.(m <> "latent_norm.weight")))
    {lets, wup} = b.(lets, "latent_up")
    {lets, sg} = b.(lets, "shared_experts.gate_proj")
    {lets, su} = b.(lets, "shared_experts.up_proj")
    {lets, sd} = b.(lets, "shared_experts.down_proj")
    shared = T.linear(situ(T.linear(u, sg), T.linear(u, su), c.beta), sd)
    Spatial.name(lets, "#{m}out", T.add(shared, T.linear(rms(acc, ln, c.eps), wup)))
  end

  # Eq. 12: β₁ tanh(g/β₁) ⊙ σ(g) ⊙ β₂ tanh(u/β₂)
  defp situ(g, u, {b1, b2}) do
    gate = T.mul(T.mul(T.splat(b1), T.tanh(T.divide(g, T.splat(b1)))), T.sigmoid(g))
    T.mul(gate, T.mul(T.splat(b2), T.tanh(T.divide(u, T.splat(b2)))))
  end

  # ---------------------------------------------------------------- shared --

  # x · rsqrt(Σx²·(1/n) + ε), then the weight (nil: none), over the last axis
  defp rms(x, w, eps) do
    {:ok, {:f32, shape}} = T.infer(x)
    ms = T.mul(T.reduce(:sum, T.mul(x, x)), T.splat(1.0 / List.last(shape)))
    y = T.mul(x, T.rsqrt(T.add(ms, T.splat(eps))))
    if w, do: T.mul(w, y), else: y
  end

  defp row(%Tensor{shape: [n]} = t), do: Tensor.new(:f32, [1, n], Tensor.widen(t).data)
end
