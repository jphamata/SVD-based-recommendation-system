defmodule Vapor.Model.Decoder do
  @moduledoc """
  The **pre-norm decoder** topology as a vapor program (eight families
  read it: `Vapor.Model.Config`, `Vapor.Lock.Adapters.Decoder`; the module
  is named for what it computes, not for the first family that used
  it) — every operation a term
  of the certified algebra (`Vapor.Algebra.Term`), so the whole model
  inherits the canonical semantics, the ladder and the substrates:

      x      = embed[tok]                           (· √d for Gemma)
      per layer:
        h    = rmsnorm(x) ⊙ w₁
        q,k,v = h·Wqᵀ (+b), h·Wkᵀ (+b), h·Wvᵀ (+b)  (MLA: low-rank q, latent kv)
        q,k  ← per-head RMSNorm (Qwen3, Gemma 3), then RoPE(pos)
        K,V  ← write rows pos of the caches (in place)
        a    = attention(q, K, V, pos)·Woᵀ (+b)     (Gemma: rmsnorm(a))
        x    = x + a
        f    = MLP(rmsnorm(x) ⊙ w₂)                 (gated: act(·Wgᵀ) ⊙ (·Wuᵀ), then ·Wdᵀ;
                                                     or a mixture of experts)
        x    = x + f                                (Gemma: + rmsnorm(f))
      logits = (rmsnorm(x) ⊙ w_f)·Wₗₘᵀ               (Wₗₘ = embed when tied; optional soft-cap)

  RMSNorm is `x · rsqrt(Σx²·(1/d) + ε)` with the canonical 16-lane sum and
  canonical `rsqrt`, then the weight (`1 + w` for Gemma) — the Hugging Face
  order.

  ## Frontier operators without new kernels

  Three constructions keep the operator set of the algebra unchanged, so
  every substrate (x86, AVX-512, NEON, RVV, SPIR-V, the oracle) runs the new
  families bit-identically with no new machine code:

    * **Per-head normalisation by exact selector contractions.** The algebra
      has no reshape. The per-head sums of squares of `x : f32[T, h·dh]` are
      `(x⊙x)·Bᵀ` with `B` the 0/1 block indicator; because `dh ≡ 0 (mod 16)`,
      every 16-lane accumulator of that dot product receives exactly the
      elements a per-head canonical reduction would, plus `+0` terms, so the
      sums are *bit for bit* those of a per-head reduction. The inverse root
      returns to the columns by `r·Eᵀ` (one `1·r` term, the rest `+0`).
    * **Mixture of experts by ranks and selection.** For each expert `e`,
      `rank_e = Σ_j [s_j > s_e or (s_j = s_e and j < e)]` (an exact sum of
      0/1 values); the expert is selected iff `rank_e < k` — the top-k of
      `torch.topk` with ties to the lower index. The output accumulates in
      expert-index order, `acc ← sel(rank_e < k, acc + w_e·y_e, acc)`:
      *selection*, not multiplication by a 0 gate, so a non-selected expert
      contributes nothing — not even a NaN or the sign of a zero. Hence a
      runtime that skips non-selected experts produces the same bits: dense
      evaluation is the definition, sparse dispatch is an optimisation.
    * **Multi-head latent attention by layout.** DeepSeek's MLA rotates only
      `qk_rope_head_dim` of each head and shares the rotated key across heads.
      The head is laid out so that rotate-half pairs `(j, j + dh/2)` are
      either both rotary or both pass-through (cos = 1, sin = 0 there — an
      exact identity up to the sign of zero); the interleaved pairing of
      `rope_interleave` is a permutation folded into the weights; the shared
      rotary key is placed into every head by an exact 0/1 contraction; the
      value head is padded with zero columns that the output projection
      ignores.

  Program interface: inputs `tok, pos : s32[T]` and one cache pair
  `k{i}, v{i} : f32[S, hkv·dh]` per layer; outputs `logits` and the updated
  caches `k{i}_next, v{i}_next`, declared as state so a recurrent run feeds
  them back. With `logits: :last` an input `last : s32[1]` selects the row
  whose logits are computed (the next-token distribution) — the vocabulary
  projection, usually the largest matrix, then runs once, not `T` times.
  With `hidden: true` the final normalised hidden state is an output too
  (`hidden : f32[T, d]`, embeddings for retrieval).
  """
  alias Vapor.{CR, F32, Program, Rejection, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Model.Config

  @neg_max -3.4028234663852886e38
  # DeepSeek's latent norms are built with the RMSNorm default, not rms_norm_eps
  @latent_eps 1.0e-6

  @doc """
  Build the program. Options: `:max_seq` (cache rows, default
  `min(max_position_embeddings, 2048)`), `:sample` (also output `next`, the
  token chosen on the substrate — see `Vapor.Engine`), `:kv` (`:contiguous` or
  `{:paged, page, pages, sequences}`, see `kv_layout/4`), `:quantize` (`nil` | `:sb4`: every
  projection matrix and the output head stored as 4-bit superblocks,
  `Vapor.Quant.Sb4`; needs contraction widths ≡ 0 mod 256), `:storage` (`:f32` |
  `:bf16`: the embedding table and every matrix stored as bfloat16 — half the
  bytes read per token, and the very bits of the `:f32` program over the
  same values rounded to bfloat16; a bfloat16 checkpoint loses nothing),
  `:max_tokens` (bound of `T`, default `:max_seq`), `:logits` (`:all` | `:last`, default `:all`),
  `:hidden` (also output the final hidden state), `:head` (default `true`;
  `false` drops the vocabulary projection — an embedding program),
  `:inject` (also take `soft : f32[T, d]` and `soft_mask : f32[T, 1]`: rows
  with mask 1 enter as `soft` instead of a token embedding — how any
  modality's rows reach a text decoder, see `Vapor.Modal`).
  """
  @spec program(Config.t(), %{String.t() => Tensor.t()}, keyword) :: {:ok, Program.t()} | {:error, Rejection.t()}
  def program(%Config{} = c, weights, opts \\ []) do
    s = Keyword.get(opts, :max_seq, min(c.max_pos, 2048))
    t = T.dyn(:t, Keyword.get(opts, :max_tokens, s))
    c = latent_flag(c, opts)

    with :ok <- quantizable(c, Keyword.get(opts, :quantize)),
         {:ok, w, lets} <- weights(c, weights, Keyword.get(opts, :quantize), Keyword.get(opts, :storage, :f32)) do
      {tables, lets} = tables(c, s, Map.get(weights, :rope_freqs), lets)
      tok = T.input(:tok, :s32, [t])
      pos = T.input(:pos, :s32, [t])
      kv = kv_layout(Keyword.get(opts, :kv, :contiguous), s, t, cache_width(c))
      x = T.gather_row(w.embed, tok)
      x = if c.embed_scale, do: T.mul(x, T.splat(c.embed_scale)), else: x

      # `inject: true`: rows whose `soft_mask` is 1 take the row of `soft`
      # instead of the token's embedding — a projected image patch, an audio
      # frame, any modality's row in the model's own space (selection, so a
      # replaced row carries nothing of the token, not even a NaN)
      x =
        if Keyword.get(opts, :inject, false),
          do: T.sel(T.input(:soft_mask, :f32, [t, 1]), T.splat(0.5), x, T.input(:soft, :f32, [t, c.hidden])),
          else: x

      {x, lets} = bind(lets, :x_embed, x)
      ctx = %{c: c, s: s, pos: pos, kv: kv, tables: tables, sel: w.sel, moe: Keyword.get(opts, :moe, :sparse)}

      {x, lets, caches} =
        Enum.reduce(0..(c.layers - 1), {x, lets, []}, fn l, {x, lets, caches} ->
          lw = w.layers |> elem(l)
          name = &:"layers.#{l}.#{&1}"

          {h, lets} = bind(lets, name.(:attn_in), norm(x, lw.ln1, c))
          {att, kn, vn, lets} = attention(h, lw, l, name, ctx, lets)
          att = if c.sandwich, do: norm(att, lw.post_attn, c), else: att
          {x, lets} = bind(lets, name.(:attn_out), T.add(x, residual(att, c)))

          {h2, lets} = bind(lets, name.(:mlp_in), norm(x, lw.ln2, c))
          {ff, lets} = mlp(h2, lw.mlp, ctx, name, lets)
          ff = if c.sandwich, do: norm(ff, lw.post_ff, c), else: ff
          {x, lets} = bind(lets, name.(:mlp_out), T.add(x, residual(ff, c)))
          {x, lets, [{l, kn, vn} | caches]}
        end)

      x =
        case Keyword.get(opts, :logits, :all) do
          :all -> x
          :last -> T.gather_row(x, T.input(:last, :s32, [T.dyn(:b, Keyword.get(opts, :max_tokens, s))]))
        end

      {xf, lets} = bind(lets, :final_norm, norm(x, w.norm, c))
      logits = lin(xf, w.head)

      logits = if c.logit_divisor, do: T.divide(logits, T.splat(c.logit_divisor)), else: logits

      logits =
        case c.final_softcap do
          nil -> logits
          cap -> T.mul(T.tanh(T.mul(logits, T.splat(1.0 / cap))), T.splat(cap * 1.0))
        end

      caches = Enum.reverse(caches)

      # `sample: true`: the next token of every logits row is also chosen on
      # the substrate (`Term.sample/2`), with `sampling : f32[B, 2]` = (1/T, u)
      sampled =
        if Keyword.get(opts, :sample, false) and rem(c.vocab, 16) == 0 do
          {:ok, {:f32, [b, _]}} = T.infer(logits)
          [next: T.sample(logits, T.input(:sampling, :f32, [b, 2]))]
        else
          []
        end

      hidden = if Keyword.get(opts, :hidden, false), do: [hidden: xf], else: []
      # `head: false` (embeddings): no vocabulary projection at all
      head = if Keyword.get(opts, :head, true), do: [logits: logits] ++ sampled, else: []

      # a latent (MLA) layer has one cache, read as both keys and values
      outputs =
        head ++ hidden ++
          Enum.flat_map(caches, fn
            {l, kn, nil} -> [{:"k#{l}_next", kn}]
            {l, kn, vn} -> [{:"k#{l}_next", kn}, {:"v#{l}_next", vn}]
          end)

      state =
        Enum.flat_map(caches, fn
          {l, _, nil} -> [{:"k#{l}", :"k#{l}_next"}]
          {l, _, _} -> [{:"k#{l}", :"k#{l}_next"}, {:"v#{l}", :"v#{l}_next"}]
        end)
      {:ok, Program.new(outputs, state: state, lets: Enum.reverse(lets))}
    end
  end

  @doc """
  Every checkpoint tensor a configuration needs, as `{name, shape, kind}`
  with `kind` in `:matrix | :norm | :bias | :vector` — what `program/3`
  reads, in Hugging Face's spelling.
  """
  def expected_weights(%Config{} = c) do
    d = c.hidden

    top =
      [{"model.embed_tokens.weight", [c.vocab, d], :matrix}, {"model.norm.weight", [d], :norm}] ++
        if(c.tie, do: [], else: [{"lm_head.weight", [c.vocab, d], :matrix}])

    top ++ Enum.flat_map(0..(c.layers - 1), &layer_weights(c, &1))
  end

  defp layer_weights(c, l) do
    p = "model.layers.#{l}."
    d = c.hidden

    norms =
      [{p <> "input_layernorm.weight", [d], :norm}, {p <> "post_attention_layernorm.weight", [d], :norm}] ++
        if(c.sandwich, do: [{p <> "pre_feedforward_layernorm.weight", [d], :norm}, {p <> "post_feedforward_layernorm.weight", [d], :norm}], else: [])

    a = p <> "self_attn."

    attn =
      case c.mla do
        nil ->
          {qw, kvw} = {c.heads * c.head_dim, c.kv_heads * c.head_dim}

          [{a <> "q_proj.weight", [qw, d], :matrix}, {a <> "k_proj.weight", [kvw, d], :matrix},
           {a <> "v_proj.weight", [kvw, d], :matrix}, {a <> "o_proj.weight", [d, qw], :matrix}] ++
            if(c.qkv_bias, do: [{a <> "q_proj.bias", [qw], :bias}, {a <> "k_proj.bias", [kvw], :bias}, {a <> "v_proj.bias", [kvw], :bias}], else: []) ++
            if(c.o_bias, do: [{a <> "o_proj.bias", [d], :bias}], else: []) ++
            if(c.qk_norm, do: [{a <> "q_norm.weight", [c.head_dim], :norm}, {a <> "k_norm.weight", [c.head_dim], :norm}], else: [])

        %{q_lora: rq, kv_lora: rkv, nope: dn, rope: dr, v: dv} ->
          h = c.heads

          if(rq, do: [{a <> "q_a_proj.weight", [rq, d], :matrix}, {a <> "q_a_layernorm.weight", [rq], :norm},
                      {a <> "q_b_proj.weight", [h * (dn + dr), rq], :matrix}],
                 else: [{a <> "q_proj.weight", [h * (dn + dr), d], :matrix}]) ++
            [{a <> "kv_a_proj_with_mqa.weight", [rkv + dr, d], :matrix}, {a <> "kv_a_layernorm.weight", [rkv], :norm},
             {a <> "kv_b_proj.weight", [h * (dn + dv), rkv], :matrix}, {a <> "o_proj.weight", [d, h * dv], :matrix}]
      end

    mlp =
      if Config.moe_layer?(c, l) do
        m = c.moe

        {router, expert, names} =
          case m.names do
            :mixtral -> {p <> "block_sparse_moe.gate.weight", &(p <> "block_sparse_moe.experts.#{&1}.#{&2}.weight"), {"w1", "w3", "w2"}}
            :qwen -> {p <> "mlp.gate.weight", &(p <> "mlp.experts.#{&1}.#{&2}_proj.weight"), {"gate", "up", "down"}}
          end

        {g, u, dn} = names

        [{router, [m.experts, d], :matrix}] ++
          Enum.flat_map(0..(m.experts - 1), fn e ->
            [{expert.(e, g), [m.inter, d], :matrix}, {expert.(e, u), [m.inter, d], :matrix}, {expert.(e, dn), [d, m.inter], :matrix}]
          end) ++
          if(m.kind == :sigmoid_group, do: [{p <> "mlp.gate.e_score_correction_bias", [m.experts], :vector}], else: []) ++
          if(m.shared > 0,
            do: [{p <> "mlp.shared_experts.gate_proj.weight", [m.inter * m.shared, d], :matrix},
                 {p <> "mlp.shared_experts.up_proj.weight", [m.inter * m.shared, d], :matrix},
                 {p <> "mlp.shared_experts.down_proj.weight", [d, m.inter * m.shared], :matrix}],
            else: [])
      else
        ff = c.intermediate
        [{p <> "mlp.gate_proj.weight", [ff, d], :matrix}, {p <> "mlp.up_proj.weight", [ff, d], :matrix},
         {p <> "mlp.down_proj.weight", [d, ff], :matrix}]
      end

    norms ++ attn ++ mlp
  end

  defp quantizable(_c, nil), do: :ok

  defp quantizable(%Config{mla: m}, :sb4) when m != nil,
    do: {:error, Rejection.new({:quantize, :sb4}, "a model without latent attention", "keep this model in f32 or bf16")}

  defp quantizable(c, :sb4) do
    widths = [c.hidden, c.heads * c.head_dim, c.intermediate] ++ if(c.moe, do: [c.moe.inter], else: [])

    if Enum.all?(widths, &(rem(&1, 256) == 0)),
      do: :ok,
      else: {:error, Rejection.new({:quantize, :sb4}, "contraction widths #{inspect(widths, charlists: :as_lists)} ≡ 0 mod 256",
                                   "keep this model in f32")}
  end

  # Contiguous caches `k{l}, v{l} : f32[S, n]` for one sequence, or paged
  # pools shared by many: `{:paged, page, pages, sequences}` gives pools
  # `f32[pages·page, n]`, a block table `table : s32[sequences, S/page]` and a
  # sequence index `slot : s32[T]` per row (continuous batching).
  defp kv_layout(:contiguous, s, _t, kvw) do
    %{write: fn name, pos, rows -> T.kv_write(T.input(name, :f32, [s, kvw]), pos, rows) end,
      attend: fn q, k, v, pos, h, hkv, scale, win -> T.attention(q, k, v, pos, h, hkv, scale, win) end}
  end

  # Streaming (`{:stream, sinks, window}`, s = sinks + window): the cache
  # holds *unrotated* keys in s rows — the first `sinks` tokens pinned, the
  # rest a ring of `window` — written at `wrow`; every step gathers the
  # rows in logical order (`gidx : s32[s]`) and rotates them at their
  # cache-relative positions 0…s−1, the queries at theirs (`pos`, ≤ s−1).
  # Positions never exceed the cache: the stream is unbounded, the memory
  # constant, and every query–key distance one the model was trained on
  # (StreamingLLM, Xiao et al. 2024, with RoPE applied in the cache frame).
  defp kv_layout({:stream, ns, nw}, s, t, kvw) when ns + nw == s do
    %{write: fn name, _pos, rows -> T.kv_write(T.input(name, :f32, [s, kvw]), T.input(:wrow, :s32, [t]), rows) end,
      attend: fn q, k, v, pos, h, hkv, scale, _win -> T.attention(q, k, v, pos, h, hkv, scale, nil) end,
      stream: %{gidx: T.input(:gidx, :s32, [s]), rel: T.const(Vapor.Tensor.from_list(:s32, [s], Enum.to_list(0..(s - 1))))}}
  end

  defp kv_layout({:paged, page, pages, ns}, s, t, kvw) when rem(s, page) == 0 do
    table = T.input(:table, :s32, [ns, div(s, page)])
    slot = T.input(:slot, :s32, [t])

    %{write: fn name, pos, rows -> T.kv_write_paged(T.input(name, :f32, [pages * page, kvw]), table, slot, pos, rows, page) end,
      attend: fn q, k, v, pos, h, hkv, scale, win -> T.attention_paged(q, k, v, table, slot, pos, h, hkv, page, scale, win) end}
  end

  defp residual(branch, %Config{residual_scale: nil}), do: branch
  defp residual(branch, %Config{residual_scale: r}), do: T.mul(branch, T.splat(r))

  # name a value (see Vapor.Program, let-bindings); lets accumulate reversed
  defp bind(lets, name, term), do: {T.ref(name, term), [{name, term} | lets]}

  @doc """
  Zero caches for every layer: `%{k0: …, v0: …, …}` — or, for latent
  attention (the default for MLA models, see `program/3`'s `:mla`), one
  cache per layer, `%{k0: f32[S, kv_lora_rank + qk_rope_head_dim], …}`.
  """
  def empty_caches(%Config{} = c, s, opts \\ []) do
    c = latent_flag(c, opts)
    w = cache_width(c)
    z = Tensor.new(:f32, [s, w], :binary.copy(<<0::32>>, s * w))
    for l <- 0..(c.layers - 1), name <- cache_names(c, l), into: %{}, do: {name, z}
  end

  @doc "Names of a layer's cache inputs: `[:k{l}, :v{l}]`, or `[:k{l}]` for a latent layer."
  def cache_names(%Config{} = c, l, opts \\ []) do
    if latent?(latent_flag(c, opts)), do: [:"k#{l}"], else: [:"k#{l}", :"v#{l}"]
  end

  @doc """
  Floats per cached token per layer: `2·hkv·dh`, or `kv_lora_rank +
  qk_rope_head_dim` for latent attention (DeepSeek-V3: 576 against 49 152
  in the expanded layout — 85× less).
  """
  def cache_floats(%Config{} = c, opts \\ []) do
    c = latent_flag(c, opts)
    if latent?(c), do: cache_width(c), else: 2 * cache_width(c)
  end

  # :mla — :latent (default: cache the compressed latent and the shared
  # rotary key; absorb the key and value up-projections into the query and
  # the output, per head) or :expanded (cache full per-head keys and values)
  defp latent_flag(%Config{mla: nil} = c, _opts), do: c
  defp latent_flag(%Config{mla: %{latent: _}} = c, opts) when opts == [], do: c
  defp latent_flag(%Config{mla: m} = c, opts), do: %{c | mla: Map.put(m, :latent, Keyword.get(opts, :mla, :latent) == :latent)}

  defp latent?(%Config{mla: %{latent: true}}), do: true
  defp latent?(_), do: false

  defp cache_width(%Config{mla: %{latent: true, kv_lora: rkv, rope: dr}}), do: rkv + dr
  defp cache_width(%Config{} = c), do: c.kv_heads * c.head_dim

  # x · rsqrt(Σx²·(1/d) + ε), then the weight (Hugging Face order)
  defp norm(x, w, %Config{eps: eps}), do: norm(x, w, eps)

  defp norm(x, w, eps) when is_float(eps) do
    {:ok, {:f32, shape}} = T.infer(x)
    d = List.last(shape)
    ms = T.mul(T.reduce(:sum, T.mul(x, x)), T.splat(1.0 / d))
    T.mul(w, T.mul(x, T.rsqrt(T.add(ms, T.splat(eps)))))
  end

  # per-head RMSNorm of x : f32[T, h·dh] by exact selector contractions
  # (see the moduledoc): sums (x⊙x)·Bᵀ, back to columns r·Eᵀ
  defp head_norm(x, w_row, {b, e}, eps, dh) do
    ss = T.linear(T.mul(x, x), b)
    r = T.rsqrt(T.add(T.mul(ss, T.splat(1.0 / dh)), T.splat(eps)))
    T.mul(w_row, T.mul(x, T.linear(r, e)))
  end

  defp proj(x, w, nil), do: lin(x, w)
  defp proj(x, w, b), do: T.add(lin(x, w), b)

  # a projection by a (bound) f32 or 4-bit matrix
  defp lin(x, {:input, _, q, _} = w) when q in [:sb4, :sb4x], do: T.qgemv(w, x)
  defp lin(x, w), do: T.linear(x, w)

  # a row-predicated projection (sparse expert dispatch): f32/bf16 or 4-bit,
  # a token reads only the weights of the experts it selected
  defp lin_m(x, {:input, _, q, _} = w, m) when q in [:sb4, :sb4x], do: T.qgemv_masked(w, x, m)
  defp lin_m(x, w, m), do: T.linear_masked(x, w, m)

  defp act(x, %Config{act: :silu}), do: T.silu(x)
  defp act(x, %Config{act: :gelu_tanh}), do: T.gelu_tanh(x)
  defp act(x, %Config{act: :gelu}), do: T.gelu(x)

  # ------------------------------------------------------------- attention --

  defp attention(h, %{attn: :mla} = lw, l, name, ctx, lets), do: mla(h, lw, l, name, ctx, lets)
  defp attention(h, %{attn: :mla_latent} = lw, l, name, ctx, lets), do: mla_latent(h, lw, l, name, ctx, lets)

  defp attention(h, lw, l, name, %{c: c, pos: pos, kv: kv} = ctx, lets) do
    {cos, sin} = layer_tables(ctx, l)
    q = proj(h, lw.wq, lw.bq)
    k = proj(h, lw.wk, lw.bk)

    {q, k} =
      if c.qk_norm,
        do: {head_norm(q, lw.qn, ctx.sel.q, c.eps, c.head_dim), head_norm(k, lw.kn, ctx.sel.k, c.eps, c.head_dim)},
        else: {q, k}

    q = T.rope(q, cos, sin, pos, c.heads)

    case kv do
      %{stream: st} ->
        # raw keys in the cache; rotated in logical order, in the cache's frame
        {kn, lets} = bind(lets, name.(:k_cache), kv.write.(:"k#{l}", pos, k))
        {vn, lets} = bind(lets, name.(:v_cache), kv.write.(:"v#{l}", pos, proj(h, lw.wv, lw.bv)))
        {kw, lets} = bind(lets, name.(:k_frame), T.rope(T.gather_row(kn, st.gidx), cos, sin, st.rel, c.kv_heads))
        {vw, lets} = bind(lets, name.(:v_frame), T.gather_row(vn, st.gidx))
        att = kv.attend.(q, kw, vw, pos, c.heads, c.kv_heads, c.attn_scale, nil)
        {proj(att, lw.wo, lw.bo), kn, vn, lets}

      _ ->
        k = T.rope(k, cos, sin, pos, c.kv_heads)
        {kn, lets} = bind(lets, name.(:k_cache), kv.write.(:"k#{l}", pos, k))
        {vn, lets} = bind(lets, name.(:v_cache), kv.write.(:"v#{l}", pos, proj(h, lw.wv, lw.bv)))
        att = kv.attend.(q, kn, vn, pos, c.heads, c.kv_heads, c.attn_scale, window(ctx, l))
        {proj(att, lw.wo, lw.bo), kn, vn, lets}
    end
  end

  # multi-head latent attention, in the head layout of `mla_layout/1`
  defp mla(h, lw, l, name, %{c: c, pos: pos, kv: kv} = ctx, lets) do
    {cos, sin} = layer_tables(ctx, l)

    q =
      case lw.qa do
        nil -> lin(h, lw.qb)
        qa -> lin(norm(lin(h, qa), lw.qa_ln, @latent_eps), lw.qb)
      end

    {cn, lets} = bind(lets, name.(:kv_latent), norm(lin(h, lw.kv_c), lw.kv_ln, @latent_eps))
    k = T.add(lin(cn, lw.kb), T.linear(lin(h, lw.kv_r), ctx.sel.place))
    q = T.rope(q, cos, sin, pos, c.heads)
    k = T.rope(k, cos, sin, pos, c.heads)
    {kn, lets} = bind(lets, name.(:k_cache), kv.write.(:"k#{l}", pos, k))
    {vn, lets} = bind(lets, name.(:v_cache), kv.write.(:"v#{l}", pos, lin(cn, lw.vb)))
    att = kv.attend.(q, kn, vn, pos, c.heads, c.heads, c.attn_scale, window(ctx, l))
    {lin(att, lw.wo), kn, vn, lets}
  end

  # Latent MLA (DeepSeek's absorbed form). With c the normalised latent and
  # kᴿ the shared rotated key, head i scores and reads
  #   qᵢ·kᵢ = (W_UKᵢᵀ qᵢᴺ)·c + qᵢᴿ·kᴿ,   oᵢ = W_UVᵢ (Σ pᵢ c)
  # so the cache row is [c | kᴿ] (kv_lora + rope floats, shared by every
  # head: multi-query attention over the latent), the query is mapped into
  # latent space per head (`linear_grouped`), and the value up-projection is
  # applied after attention, per head. The cache is read as both K and V
  # (the rotary columns of the "value" meet zero columns of W_UV). Real
  # arithmetic is that of the expanded form; the rounding differs (another
  # association), so this is a different program: certified in its own
  # right, compared with the expanded form and with transformers within
  # tolerance, not bit for bit.
  defp mla_latent(h, lw, l, name, %{c: c, pos: pos, kv: kv} = ctx, lets) do
    {cos, sin} = layer_tables(ctx, l)
    sel = ctx.sel

    q =
      case lw.qa do
        nil -> lin(h, lw.qb)
        qa -> lin(norm(lin(h, qa), lw.qa_ln, @latent_eps), lw.qb)
      end

    q = T.rope(q, cos, sin, pos, c.heads)
    {qt, lets} = bind(lets, name.(:q_latent), T.linear_grouped(q, lw.qabs, c.heads))

    {cn, lets} = bind(lets, name.(:kv_latent), norm(lin(h, lw.kv_c), lw.kv_ln, @latent_eps))
    # the shared rotary key: placed into one head of the layout, rotated by
    # the layer's own tables, read back in raw order (exact 0/1 contractions)
    kr = T.linear(T.rope(T.linear(lin(h, lw.kv_r), sel.place1), cos, sin, pos, 1), sel.unplace)
    row = T.add(T.linear(cn, sel.lat_c), T.linear(kr, sel.lat_r))
    {kn, lets} = bind(lets, name.(:k_cache), kv.write.(:"k#{l}", pos, row))
    att = kv.attend.(qt, kn, kn, pos, c.heads, 1, c.attn_scale, window(ctx, l))
    {lin(T.linear_grouped(att, lw.vabs, c.heads), lw.wo), kn, nil, lets}
  end

  # --------------------------------------------------------------------- MLP --

  defp mlp(h, {:dense, wg, wu, wd}, %{c: c}, _name, lets), do: {lin(T.mul(act(lin(h, wg), c), lin(h, wu)), wd), lets}

  defp mlp(h, {:moe, m}, %{c: c, sel: sel} = ctx, name, lets) do
    m = Map.merge(m, sel.moe)
    %{experts: e, top_k: k} = c.moe
    ep = pad16(e)

    logits = lin(h, m.router)
    logits = if ep > e, do: T.add(logits, m.pad), else: logits

    {score, choice, lets} =
      case c.moe.kind do
        :softmax ->
          mx = T.reduce(:max, logits)
          ex = T.exp(T.sub(logits, mx))
          {p, lets} = bind(lets, name.(:router_p), T.mul(ex, T.rcp(T.reduce(:sum, ex))))
          {p, p, lets}

        :sigmoid_group ->
          {sg, lets} = bind(lets, name.(:router_s), T.sigmoid(logits))
          {ch, lets} = bind(lets, name.(:router_choice), T.add(sg, m.bias))
          {mc, lets} = group_limit(ch, m, c, name, lets)
          {sg, mc, lets}
      end

    cols = for i <- 0..(e - 1), do: T.linear(choice, m.onehot |> elem(i))
    scols = if score == choice, do: cols, else: (for i <- 0..(e - 1), do: T.linear(score, m.onehot |> elem(i)))

    {ranks, lets} =
      Enum.map_reduce(0..(e - 1), lets, fn i, lets ->
        col = Enum.at(cols, i)
        count = T.sel(col, choice, T.splat(1.0), T.sel(choice, col, T.splat(0.0), m.tie |> elem(i)))
        bind(lets, name.(:"rank#{i}"), T.reduce(:sum, count))
      end)

    kk = T.splat(k * 1.0)

    # gate weights: the selected scores, renormalised (and scaled) as declared
    {den, lets} =
      if c.moe.norm do
        s = Enum.zip(ranks, scols) |> Enum.reduce(nil, fn {r, sc}, acc ->
          term = T.sel(r, kk, sc, T.splat(0.0))
          if acc, do: T.add(acc, term), else: term
        end)

        s = if c.moe.kind == :sigmoid_group, do: T.add(s, T.splat(1.0e-20)), else: s
        {d, lets} = bind(lets, name.(:router_den), s)
        {d, lets}
      else
        {nil, lets}
      end

    dispatch = Map.get(ctx, :moe, :sparse)

    {acc, lets} =
      Enum.zip([Enum.to_list(0..(e - 1)), ranks, scols])
      |> Enum.reduce({nil, lets}, fn {i, r, sc}, {acc, lets} ->
        {wg, wu, wd} = m.experts |> elem(i)

        # sparse dispatch: the expert's rows are predicated on its selection
        # (`linear_masked`), so a token reads only its top-k experts' weights;
        # the selected rows run the dense instructions, the others are
        # discarded by `sel` below either way — the same bits as :dense
        {y, lets} =
          case dispatch do
            :dense ->
              {lin(T.mul(act(lin(h, wg), c), lin(h, wu)), wd), lets}

            :sparse ->
              {mk, lets} = bind(lets, name.(:"moe_mask#{i}"), T.sel(r, kk, T.splat(1.0), T.splat(0.0)))
              {lin_m(T.mul(act(lin_m(h, wg, mk), c), lin_m(h, wu, mk)), wd, mk), lets}
          end

        g = if den, do: T.divide(sc, den), else: sc
        g = if c.moe.scale != 1.0, do: T.mul(g, T.splat(c.moe.scale)), else: g
        contrib = T.mul(y, g)
        next = if acc, do: T.sel(r, kk, T.add(acc, contrib), acc), else: T.sel(r, kk, contrib, T.splat(0.0))
        bind(lets, name.(:"moe_acc#{i}"), next)
      end)

    case m.shared do
      nil -> {acc, lets}
      {wg, wu, wd} -> {T.add(acc, lin(T.mul(act(lin(h, wg), c), lin(h, wu)), wd)), lets}
    end
  end

  # DeepSeek's group-limited choice: a group's score is the sum of its two
  # best choice scores; the best `topk_group` groups keep their experts, the
  # others are masked to −FLT_MAX. All counts and sums are exact.
  # every group kept (n_group 1 — DeepSeek-V2-Lite — or topk_group ≥ n_group):
  # the mask is all ones and `sel` would return `ch` itself, so the limit is
  # the identity (same bits) and is not built — with one group there is no
  # other group to rank against, and the general form below has no terms
  defp group_limit(ch, _m, %{moe: %{groups: g, topk_group: kg}}, _name, lets) when kg >= g, do: {ch, lets}

  defp group_limit(ch, m, c, name, lets) do
    %{experts: e, groups: g, topk_group: kg} = c.moe
    n = div(e, g)
    cols = for i <- 0..(e - 1), do: T.linear(ch, m.onehot |> elem(i))

    {gscores, lets} =
      Enum.map_reduce(0..(g - 1), lets, fn gi, lets ->
        members = (gi * n)..(gi * n + n - 1)

        top2 =
          Enum.reduce(members, nil, fn i, acc ->
            col = Enum.at(cols, i)
            count = T.sel(col, ch, m.gmask |> elem(i), T.sel(ch, col, T.splat(0.0), m.gtie |> elem(i)))
            term = T.sel(T.reduce(:sum, count), T.splat(2.0), col, T.splat(0.0))
            if acc, do: T.add(acc, term), else: term
          end)

        bind(lets, name.(:"group#{gi}"), top2)
      end)

    # a group is kept iff fewer than topk_group groups beat it (ties: lower index first)
    {keep, lets} =
      Enum.map_reduce(0..(g - 1), lets, fn gi, lets ->
        mine = Enum.at(gscores, gi)

        rank =
          Enum.reduce(Enum.reject(0..(g - 1), &(&1 == gi)), nil, fn gj, acc ->
            other = Enum.at(gscores, gj)
            beats = if gj < gi, do: T.sel(other, mine, T.splat(0.0), T.splat(1.0)), else: T.sel(mine, other, T.splat(1.0), T.splat(0.0))
            if acc, do: T.add(acc, beats), else: beats
          end)

        bind(lets, name.(:"group_keep#{gi}"), T.sel(rank, T.splat(kg * 1.0), T.splat(1.0), T.splat(0.0)))
      end)

    # the kept-group mask over the expert columns: Σ_g keep_g · member_row_g
    mask = Enum.zip(keep, Tuple.to_list(m.grows)) |> Enum.reduce(nil, fn {kp, row}, acc ->
      term = T.mul(kp, row)
      if acc, do: T.add(acc, term), else: term
    end)

    bind(lets, name.(:router_masked), T.sel(mask, T.splat(0.5), T.splat(@neg_max), ch))
  end

  defp pad16(n), do: div(n + 15, 16) * 16

  # --------------------------------------------------------------- weights --

  @matrices [:wq, :wk, :wv, :wo]

  defp weights(c, ws, quantize, wdt) do
    d = c.hidden
    get = fn name, shape -> fetch(ws, name, shape) end

    # norms and biases broadcast as one row [1, n]; Gemma's norms mean 1 + w
    row = fn t -> t = Tensor.widen(t); Tensor.new(:f32, [1 | t.shape], t.data) end
    nrow = fn t -> if c.norm_offset, do: offset_row(row.(t)), else: row.(t) end
    # matrices in the requested storage (bf16 → f32 is exact; f32 → bf16 rounds)
    mat = fn t -> if wdt == :bf16, do: Tensor.to_bf16(t), else: Tensor.widen(t) end
    q = fn t -> if quantize == :sb4, do: Vapor.Quant.Sb4.quantize(Tensor.widen(t)), else: mat.(t) end

    with {:ok, embed} <- get.("model.embed_tokens.weight", [c.vocab, d]),
         {:ok, normw} <- get.("model.norm.weight", [d]),
         {:ok, head} <- if(c.tie, do: {:ok, nil}, else: get.("lm_head.weight", [c.vocab, d])),
         {:ok, layers} <- all(0..(c.layers - 1), &layer(c, ws, &1, get)) do
      {embed_r, lets} = bind([], :"model.embed_tokens.weight", T.const(mat.(embed)))
      {norm_r, lets} = bind(lets, :"model.norm.weight", T.const(nrow.(normw)))

      {head_r, lets} =
        cond do
          head -> bind(lets, :"lm_head.weight", T.const(q.(head)))
          quantize == :sb4 -> bind(lets, :"lm_head.weight", T.const(q.(embed)))
          true -> {embed_r, lets}
        end

      {sel, lets} = selectors(c, lets)

      {layers, lets} =
        Enum.map_reduce(layers, lets, fn fields, lets ->
          Enum.map_reduce(fields, lets, fn
            {k, nil}, lets -> {{k, nil}, lets}
            {k, atom}, lets when is_atom(atom) -> {{k, atom}, lets}
            {k, {:mlp, mlp}}, lets -> {mlp, lets} = bind_mlp(mlp, q, lets); {{k, mlp}, lets}
            {k, {:mla, name, t}}, lets -> {r, lets} = bind(lets, String.to_atom(name), T.const(mat.(t))); {{k, r}, lets}
            {k, {:norm, name, t}}, lets -> {r, lets} = bind(lets, String.to_atom(name), T.const(nrow.(t))); {{k, r}, lets}
            {k, {:row, name, t}}, lets -> {r, lets} = bind(lets, String.to_atom(name), T.const(t)); {{k, r}, lets}
            {k, {name, t}}, lets ->
              t = cond do
                k in @matrices -> q.(t)
                true -> row.(t)
              end

              {r, lets} = bind(lets, String.to_atom(name), T.const(t))
              {{k, r}, lets}
          end)
          |> then(fn {kv, lets} -> {Map.new(kv), lets} end)
        end)

      {:ok, %{embed: embed_r, norm: norm_r, head: head_r, layers: List.to_tuple(layers), sel: sel}, lets}
    end
  end

  defp fetch(ws, name, shape) do
    case Map.fetch(ws, name) do
      {:ok, %Tensor{dtype: dt, shape: ^shape} = t} when dt in [:f32, :bf16] -> {:ok, t}
      {:ok, %Tensor{} = t} -> {:error, Rejection.new({:weight, name}, "f32#{inspect(shape, charlists: :as_lists)}, got #{t.dtype}#{inspect(t.shape, charlists: :as_lists)}", "check the checkpoint against config.json")}
      :error -> {:error, Rejection.new({:weight, name}, "present in the checkpoint", "check the checkpoint against config.json")}
    end
  end

  defp offset_row(%Tensor{} = t), do: Tensor.new(:f32, t.shape, t.data |> F32.decode() |> Enum.map(&F32.add(F32.from_float(1.0), &1)) |> F32.encode())

  # one layer's fields: {key, {name, tensor}} (norms, biases, standard
  # projections), {:norm, …}, {:row, …} (exact constants), {:mla, …}
  # (derived matrices), {:mlp, …}, or atoms (markers)
  defp layer(c, ws, l, get) do
    p = "model.layers.#{l}."
    d = c.hidden

    with {:ok, norms} <- layer_norms(c, p, d, get),
         {:ok, attn} <- if(c.mla, do: mla_weights(c, ws, p, get), else: std_attention(c, p, d, get)),
         {:ok, mlp} <- mlp_weights(c, l, p, get) do
      {:ok, norms ++ attn ++ [mlp: {:mlp, mlp}]}
    end
  end

  defp layer_norms(%Config{sandwich: true}, p, d, get) do
    with {:ok, ln1} <- get.(p <> "input_layernorm.weight", [d]),
         {:ok, pa} <- get.(p <> "post_attention_layernorm.weight", [d]),
         {:ok, pf} <- get.(p <> "pre_feedforward_layernorm.weight", [d]),
         {:ok, po} <- get.(p <> "post_feedforward_layernorm.weight", [d]) do
      {:ok, [ln1: {:norm, p <> "input_layernorm.weight", ln1}, post_attn: {:norm, p <> "post_attention_layernorm.weight", pa},
             ln2: {:norm, p <> "pre_feedforward_layernorm.weight", pf}, post_ff: {:norm, p <> "post_feedforward_layernorm.weight", po}]}
    end
  end

  defp layer_norms(_c, p, d, get) do
    with {:ok, ln1} <- get.(p <> "input_layernorm.weight", [d]),
         {:ok, ln2} <- get.(p <> "post_attention_layernorm.weight", [d]) do
      {:ok, [ln1: {:norm, p <> "input_layernorm.weight", ln1}, ln2: {:norm, p <> "post_attention_layernorm.weight", ln2}]}
    end
  end

  defp std_attention(c, p, d, get) do
    qw = c.heads * c.head_dim
    kvw = c.kv_heads * c.head_dim
    opt = fn name, shape, on -> if on, do: get.(name, shape), else: {:ok, nil} end
    a = p <> "self_attn."

    with {:ok, wq} <- get.(a <> "q_proj.weight", [qw, d]),
         {:ok, wk} <- get.(a <> "k_proj.weight", [kvw, d]),
         {:ok, wv} <- get.(a <> "v_proj.weight", [kvw, d]),
         {:ok, wo} <- get.(a <> "o_proj.weight", [d, qw]),
         {:ok, bq} <- opt.(a <> "q_proj.bias", [qw], c.qkv_bias),
         {:ok, bk} <- opt.(a <> "k_proj.bias", [kvw], c.qkv_bias),
         {:ok, bv} <- opt.(a <> "v_proj.bias", [kvw], c.qkv_bias),
         {:ok, bo} <- opt.(a <> "o_proj.bias", [d], c.o_bias),
         {:ok, qn} <- opt.(a <> "q_norm.weight", [c.head_dim], c.qk_norm),
         {:ok, kn} <- opt.(a <> "k_norm.weight", [c.head_dim], c.qk_norm) do
      tile = fn t, n -> if t, do: tiled(t, n, c.norm_offset) end
      # partial rotary: q/k rows of each head moved into the rotate-half
      # layout (attention scores are invariant to one permutation of both)
      {wq, wk, bq, bk} =
        if c.rotary_dim do
          slot_of = rotary_layout(c)
          src = Enum.sort_by(0..(c.head_dim - 1), slot_of) |> List.to_tuple()
          perm = fn t, h -> t && permute_heads(t, h, c.head_dim, src) end
          {perm.(wq, c.heads), perm.(wk, c.kv_heads), perm.(bq, c.heads), perm.(bk, c.kv_heads)}
        else
          {wq, wk, bq, bk}
        end

      {:ok,
       [attn: :std, wq: {a <> "q_proj.weight", wq}, wk: {a <> "k_proj.weight", wk}, wv: {a <> "v_proj.weight", wv},
        wo: {a <> "o_proj.weight", wo}, bq: bq && {a <> "q_proj.bias", bq}, bk: bk && {a <> "k_proj.bias", bk},
        bv: bv && {a <> "v_proj.bias", bv}, bo: bo && {a <> "o_proj.bias", bo},
        qn: qn && {:row, a <> "q_norm.weight", tile.(qn, c.heads)}, kn: kn && {:row, a <> "k_norm.weight", tile.(kn, c.kv_heads)}]}
    end
  end

  @doc false
  # The partial-rotary head layout (dh slots, half = dh/2, rotate-half pairs
  # (j, j + half)): raw rotary dim j < r forms the pair (j mod r/2) of the
  # first r dims, placed at (half − r/2 + pair, dh − r/2 + pair); the dh − r
  # pass-through dims fill [0, half − r/2) and [half, dh − r/2) in order.
  # Returns the slot of raw dim i.
  def rotary_layout(%Config{head_dim: dh, rotary_dim: r}) do
    {half, hr} = {div(dh, 2), div(r, 2)}
    first = half - hr

    fn
      i when i < r -> if div(i, hr) == 0, do: first + rem(i, hr), else: dh - hr + rem(i, hr)
      i -> m = i - r; if m < first, do: m, else: half + (m - first)
    end
  end

  # rows (or entries, for a bias) of each of h heads reordered: slot s ← raw src[s]
  defp permute_heads(%Tensor{shape: shape, dtype: dt, data: data} = t, h, dh, src) do
    k = case shape do
      [_, k] -> k
      [_] -> 1
    end

    es = div(byte_size(data), Enum.product(shape))
    row = k * es
    out = for hh <- 0..(h - 1), s <- 0..(dh - 1), into: <<>>, do: binary_part(data, (hh * dh + elem(src, s)) * row, row)
    %{t | data: out, dtype: dt}
  end

  # a per-head norm weight repeated over h heads, as a row [1, h·dh]
  defp tiled(%Tensor{} = t, h, offset) do
    bits = t |> Tensor.widen() |> Map.fetch!(:data) |> F32.decode()
    bits = if offset, do: Enum.map(bits, &F32.add(F32.from_float(1.0), &1)), else: bits
    Tensor.new(:f32, [1, h * length(bits)], bits |> List.duplicate(h) |> List.flatten() |> F32.encode())
  end

  # ------------------------------------------------------------------ MLA --

  @doc false
  # The MLA head layout (dh slots, half = dh/2): rotate-half pairs (j, j+half).
  # Rotary pair p sits at (half − dr/2 + p, dh − dr/2 + p); the dh − dr
  # pass-through slots fill [0, half − dr/2) and [half, dh − dr/2) in order.
  # Returns {nope slot of i (i < dn), slot of raw rotary dim j (j < dr)}.
  def mla_layout(%Config{head_dim: dh, mla: %{nope: dn, rope: dr, interleave: il}}) do
    half = div(dh, 2)
    first = half - div(dr, 2)
    nope = fn i -> if i < first, do: i, else: half + (i - first) end

    rope = fn j ->
      {pair, member} = if il, do: {div(j, 2), rem(j, 2)}, else: {rem(j, div(dr, 2)), div(j, div(dr, 2))}
      if member == 0, do: first + pair, else: dh - div(dr, 2) + pair
    end

    _ = dn
    {nope, rope}
  end

  defp mla_weights(c, _ws, p, get) do
    %{q_lora: rq, kv_lora: rkv, nope: dn, rope: dr, v: dv} = c.mla
    {h, d, dh} = {c.heads, c.hidden, c.head_dim}
    a = p <> "self_attn."
    {nope, rope} = mla_layout(c)

    with {:ok, q} <- if(rq, do: get.(a <> "q_b_proj.weight", [h * (dn + dr), rq]), else: get.(a <> "q_proj.weight", [h * (dn + dr), d])),
         {:ok, qa} <- if(rq, do: get.(a <> "q_a_proj.weight", [rq, d]), else: {:ok, nil}),
         {:ok, qa_ln} <- if(rq, do: get.(a <> "q_a_layernorm.weight", [rq]), else: {:ok, nil}),
         {:ok, kva} <- get.(a <> "kv_a_proj_with_mqa.weight", [rkv + dr, d]),
         {:ok, kv_ln} <- get.(a <> "kv_a_layernorm.weight", [rkv]),
         {:ok, kvb} <- get.(a <> "kv_b_proj.weight", [h * (dn + dv), rkv]),
         {:ok, wo} <- get.(a <> "o_proj.weight", [d, h * dv]) do
      # q rows into the layout (pad slots are zero rows)
      q_src = for hh <- 0..(h - 1), into: %{} do
        {hh, Map.new(Enum.map(0..(dn - 1), &{nope.(&1), hh * (dn + dr) + &1}) ++ Enum.map(0..(dr - 1), &{rope.(&1), hh * (dn + dr) + dn + &1}))}
      end

      qrows = place_rows(q, h, dh, fn hh, slot -> q_src[hh][slot] end)
      kb = place_rows(kvb, h, dh, fn hh, slot -> Enum.find_value(0..(dn - 1), &(nope.(&1) == slot && hh * (dn + dv) + &1)) end)
      kv_c = rows_of(kva, 0, rkv)
      kv_r = rows_of(kva, rkv, dr)

      common =
        [qb: {:mla, a <> "q_b_proj.weight", qrows}, qa: qa && {:mla, a <> "q_a_proj.weight", qa},
         qa_ln: qa_ln && {:norm, a <> "q_a_layernorm.weight", qa_ln}, kv_c: {:mla, a <> "kv_a_proj.latent", kv_c},
         kv_r: {:mla, a <> "kv_a_proj.rope", kv_r}, kv_ln: {:norm, a <> "kv_a_layernorm.weight", kv_ln}]

      if latent?(c) do
        {:ok,
         [attn: :mla_latent] ++ common ++
           [qabs: {:mla, a <> "kv_b_proj.absorbed_key", absorbed_key(kb, h, dh, rkv, dr, rope)},
            vabs: {:mla, a <> "kv_b_proj.absorbed_value", absorbed_value(kvb, h, dn, dv, rkv, dr)},
            wo: {:mla, a <> "o_proj.weight", wo}]}
      else
        vb = place_rows(kvb, h, dh, fn hh, slot -> if slot < dv, do: hh * (dn + dv) + dn + slot end)

        {:ok,
         [attn: :mla] ++ common ++
           [kb: {:mla, a <> "kv_b_proj.key", kb}, vb: {:mla, a <> "kv_b_proj.value", vb},
            wo: {:mla, a <> "o_proj.weight", place_cols(wo, h, dh, dv)}]}
      end
    end
  end

  # per head, the absorbed query map [kv_lora + rope, dh] of the layout:
  # rows r < kv_lora are column r of the head's key up-projection (W_UKᵀ),
  # rows kv_lora + j select the rotary slot of raw rope dim j
  defp absorbed_key(%Tensor{shape: [_, rkv]} = kb, h, dh, rkv, dr, rope) do
    kb = Tensor.widen(kb)
    one = F32.from_float(1.0)

    data =
      for hh <- 0..(h - 1), into: <<>> do
        block = binary_part(kb.data, hh * dh * rkv * 4, dh * rkv * 4) |> F32.decode() |> Enum.chunk_every(rkv)
        cols = block |> Enum.zip() |> Enum.map(&Tuple.to_list/1)
        lat = for col <- cols, into: <<>>, do: F32.encode(col)
        rot = for j <- 0..(dr - 1), into: <<>>, do: F32.encode(for(s <- 0..(dh - 1), do: if(s == rope.(j), do: one, else: 0)))
        lat <> rot
      end

    Tensor.new(:f32, [h * (rkv + dr), dh], data)
  end

  # per head, the value up-projection over the latent columns of a cache
  # row [dv, kv_lora + rope] (zero on the rotary columns)
  defp absorbed_value(%Tensor{shape: [_, rkv]} = kvb, h, dn, dv, rkv, dr) do
    kvb = Tensor.widen(kvb)
    zeros = :binary.copy(<<0::32>>, dr)

    data =
      for hh <- 0..(h - 1), i <- 0..(dv - 1), into: <<>> do
        binary_part(kvb.data, (hh * (dn + dv) + dn + i) * rkv * 4, rkv * 4) <> zeros
      end

    Tensor.new(:f32, [h * dv, rkv + dr], data)
  end

  # rows r of a matrix [n, k] → [h·dh, k]: slot s of head hh takes source row src.(hh, s) (nil: zero)
  defp place_rows(%Tensor{shape: [_, k]} = t, h, dh, src) do
    t = Tensor.widen(t)
    zero = :binary.copy(<<0::32>>, k)
    data = for hh <- 0..(h - 1), s <- 0..(dh - 1), into: <<>>, do: (case src.(hh, s) do
      nil -> zero
      r -> binary_part(t.data, r * k * 4, k * 4)
    end)

    Tensor.new(:f32, [h * dh, k], data)
  end

  defp rows_of(%Tensor{shape: [_, k]} = t, from, n) do
    t = Tensor.widen(t)
    Tensor.new(:f32, [n, k], binary_part(t.data, from * k * 4, n * k * 4))
  end

  # o_proj [d, h·dv] → [d, h·dh]: column hh·dh + i ← hh·dv + i (i < dv), else 0
  defp place_cols(%Tensor{shape: [d, _]} = t, h, dh, dv) do
    t = Tensor.widen(t)

    data =
      for r <- 0..(d - 1), into: <<>> do
        row = binary_part(t.data, r * h * dv * 4, h * dv * 4)
        for hh <- 0..(h - 1), into: <<>>, do: binary_part(row, hh * dv * 4, dv * 4) <> :binary.copy(<<0::32>>, dh - dv)
      end

    Tensor.new(:f32, [d, h * dh], data)
  end

  # ------------------------------------------------------------------ MLP weights --

  defp mlp_weights(c, l, p, get) do
    if Config.moe_layer?(c, l), do: moe_weights(c, p, get), else: dense_weights(c, p, get)
  end

  defp dense_weights(c, p, get) do
    d = c.hidden
    ff = c.intermediate

    with {:ok, wg} <- get.(p <> "mlp.gate_proj.weight", [ff, d]),
         {:ok, wu} <- get.(p <> "mlp.up_proj.weight", [ff, d]),
         {:ok, wd} <- get.(p <> "mlp.down_proj.weight", [d, ff]) do
      {:ok, {:dense, {p <> "mlp.gate_proj.weight", wg}, {p <> "mlp.up_proj.weight", wu}, {p <> "mlp.down_proj.weight", wd}}}
    end
  end

  defp moe_weights(%Config{moe: m} = c, p, get) do
    d = c.hidden
    mi = m.inter

    {router, expert} =
      case m.names do
        :mixtral -> {p <> "block_sparse_moe.gate.weight", fn e, w -> p <> "block_sparse_moe.experts.#{e}.#{w}.weight" end}
        :qwen -> {p <> "mlp.gate.weight", fn e, w -> p <> "mlp.experts.#{e}.#{w}_proj.weight" end}
      end

    names = if m.names == :mixtral, do: {"w1", "w3", "w2"}, else: {"gate", "up", "down"}

    with {:ok, r} <- get.(router, [m.experts, d]),
         {:ok, experts} <- all(0..(m.experts - 1), fn e ->
           {g, u, dn} = names
           with {:ok, wg} <- get.(expert.(e, g), [mi, d]),
                {:ok, wu} <- get.(expert.(e, u), [mi, d]),
                {:ok, wd} <- get.(expert.(e, dn), [d, mi]) do
             {:ok, {{expert.(e, g), wg}, {expert.(e, u), wu}, {expert.(e, dn), wd}}}
           end
         end),
         {:ok, bias} <- if(m.kind == :sigmoid_group, do: get.(p <> "mlp.gate.e_score_correction_bias", [m.experts]), else: {:ok, nil}),
         {:ok, shared} <- shared_weights(c, p, get) do
      {:ok, {:moe, %{router: {router, r}, experts: experts, bias: bias && {p <> "mlp.gate.e_score_correction_bias", bias}, shared: shared}}}
    end
  end

  defp shared_weights(%Config{moe: %{shared: 0}}, _p, _get), do: {:ok, nil}

  defp shared_weights(%Config{moe: m} = c, p, get) do
    {d, si} = {c.hidden, m.inter * m.shared}

    with {:ok, wg} <- get.(p <> "mlp.shared_experts.gate_proj.weight", [si, d]),
         {:ok, wu} <- get.(p <> "mlp.shared_experts.up_proj.weight", [si, d]),
         {:ok, wd} <- get.(p <> "mlp.shared_experts.down_proj.weight", [d, si]) do
      {:ok, {{p <> "mlp.shared_experts.gate_proj.weight", wg}, {p <> "mlp.shared_experts.up_proj.weight", wu},
             {p <> "mlp.shared_experts.down_proj.weight", wd}}}
    end
  end

  # bind the MLP's matrices; the router stays f32 (it decides, it is not
  # summed into the stream), padded to a multiple of 16 experts
  defp bind_mlp({:dense, {gn, g}, {un, u}, {dn, d}}, q, lets) do
    {g, lets} = bind(lets, String.to_atom(gn), T.const(q.(g)))
    {u, lets} = bind(lets, String.to_atom(un), T.const(q.(u)))
    {d, lets} = bind(lets, String.to_atom(dn), T.const(q.(d)))
    {{:dense, g, u, d}, lets}
  end

  defp bind_mlp({:moe, m}, q, lets) do
    {rname, r} = m.router
    [e, dd] = r.shape
    ep = pad16(e)
    r = Tensor.widen(r)
    rpad = Tensor.new(:f32, [ep, dd], r.data <> :binary.copy(<<0::32>>, (ep - e) * dd))
    {router, lets} = bind(lets, String.to_atom(rname), T.const(rpad))

    {experts, lets} =
      Enum.map_reduce(m.experts, lets, fn {{gn, g}, {un, u}, {dn, d}}, lets ->
        {g, lets} = bind(lets, String.to_atom(gn), T.const(q.(g)))
        {u, lets} = bind(lets, String.to_atom(un), T.const(q.(u)))
        {d, lets} = bind(lets, String.to_atom(dn), T.const(q.(d)))
        {{g, u, d}, lets}
      end)

    {shared, lets} =
      case m.shared do
        nil -> {nil, lets}
        sh -> {{:dense, g, u, d}, lets} = bind_mlp({:dense, elem(sh, 0), elem(sh, 1), elem(sh, 2)}, q, lets); {{g, u, d}, lets}
      end

    # the choice bias (DeepSeek's e_score_correction_bias), −FLT_MAX on padding
    {bias, lets} =
      case m.bias do
        nil -> {nil, lets}
        {bn, b} ->
          vals = (b |> Tensor.widen() |> Map.fetch!(:data) |> F32.decode()) ++ List.duplicate(F32.from_float(@neg_max), ep - e)
          bind(lets, String.to_atom(bn), T.const(Tensor.new(:f32, [1, ep], F32.encode(vals))))
      end

    {{:moe, %{router: router, experts: List.to_tuple(experts), shared: shared, bias: bias}}, lets}
  end

  # --------------------------------------------------------- selector constants --

  # Exact 0/1 constants shared by all layers: per-head sum/expand matrices
  # for q and k (B : [hp, h·dh], E : [h·dh, hp]); the MLA rotary-key
  # placement [h·dh, dr]; the MoE one-hot rows, tie rows, group masks.
  defp selectors(c, lets) do
    {sel, lets} =
      if c.qk_norm do
        {q, lets} = head_selectors(c.heads, c.head_dim, lets)
        {k, lets} = head_selectors(c.kv_heads, c.head_dim, lets)
        {%{q: q, k: k}, lets}
      else
        {%{}, lets}
      end

    {sel, lets} =
      if c.mla do
        {_nope, rope} = mla_layout(c)
        dr = c.mla.rope
        dh = c.head_dim
        h = c.heads
        ones = MapSet.new(for hh <- 0..(h - 1), j <- 0..(dr - 1), do: {hh * dh + rope.(j), j})
        if latent?(c) do
          rkv = c.mla.kv_lora
          slot = MapSet.new(for j <- 0..(dr - 1), do: {rope.(j), j})
          {p1, lets} = bind(lets, :"$mla.place1", T.const(matrix(dh, dr, &MapSet.member?(slot, {&1, &2}))))
          {up, lets} = bind(lets, :"$mla.unplace", T.const(matrix(dr, dh, &MapSet.member?(slot, {&2, &1}))))
          {lc, lets} = bind(lets, :"$mla.latent_c", T.const(matrix(rkv + dr, rkv, &(&1 == &2))))
          {lr, lets} = bind(lets, :"$mla.latent_r", T.const(matrix(rkv + dr, dr, &(&1 == rkv + &2))))
          {Map.merge(sel, %{place1: p1, unplace: up, lat_c: lc, lat_r: lr}), lets}
        else
          place = matrix(h * dh, dr, &MapSet.member?(ones, {&1, &2}))
          {r, lets} = bind(lets, :"$mla.place", T.const(place))
          {Map.put(sel, :place, r), lets}
        end
      else
        {sel, lets}
      end

    if c.moe, do: moe_selectors(c, sel, lets), else: {sel, lets}
  end

  defp head_selectors(h, dh, lets) do
    hp = pad16(h)
    key = "#{h}x#{dh}"

    case List.keyfind(lets, :"$sel.sum.#{key}", 0) do
      {_, b} ->
        {_, e} = List.keyfind(lets, :"$sel.expand.#{key}", 0)
        {{T.ref(:"$sel.sum.#{key}", b), T.ref(:"$sel.expand.#{key}", e)}, lets}

      nil ->
        {b, lets} = bind(lets, :"$sel.sum.#{key}", T.const(matrix(hp, h * dh, fn i, j -> div(j, dh) == i end)))
        {e, lets} = bind(lets, :"$sel.expand.#{key}", T.const(matrix(h * dh, hp, fn j, i -> div(j, dh) == i end)))
        {{b, e}, lets}
    end
  end

  defp moe_selectors(%Config{moe: m}, sel, lets) do
    e = m.experts
    ep = pad16(e)
    row = fn f -> Tensor.new(:f32, [1, ep], for(j <- 0..(ep - 1), into: <<>>, do: <<F32.from_float(if(f.(j), do: 1.0, else: 0.0))::32-little>>)) end
    bind_rows = fn prefix, f, lets ->
      Enum.map_reduce(0..(e - 1), lets, fn i, lets -> bind(lets, :"$moe.#{prefix}#{i}", T.const(row.(&f.(i, &1)))) end)
    end

    {onehot, lets} = bind_rows.("onehot", fn i, j -> j == i end, lets)
    {tie, lets} = bind_rows.("tie", fn i, j -> j < i end, lets)
    {pad, lets} = bind(lets, :"$moe.pad", T.const(Tensor.new(:f32, [1, ep], F32.encode(List.duplicate(0, e) ++ List.duplicate(F32.from_float(@neg_max), ep - e)))))
    # onehot rows become [1, ep] linear weights (a column selector)
    m_sel = %{onehot: List.to_tuple(onehot), tie: List.to_tuple(tie), pad: pad}

    {m_sel, lets} =
      if m.kind == :sigmoid_group do
        n = div(e, m.groups)
        {gmask, lets} = bind_rows.("gmask", fn i, j -> j < e and div(j, n) == div(i, n) end, lets)
        {gtie, lets} = bind_rows.("gtie", fn i, j -> j < i and div(j, n) == div(i, n) end, lets)
        {grows, lets} = Enum.map_reduce(0..(m.groups - 1), lets, fn g, lets -> bind(lets, :"$moe.grow#{g}", T.const(row.(&(&1 < e and div(&1, n) == g)))) end)
        {Map.merge(m_sel, %{gmask: List.to_tuple(gmask), gtie: List.to_tuple(gtie), grows: List.to_tuple(grows)}), lets}
      else
        {m_sel, lets}
      end

    {Map.put(sel, :moe, m_sel), lets}
  end

  defp matrix(n, k, f) do
    one = F32.from_float(1.0)
    Tensor.new(:f32, [n, k], for(i <- 0..(n - 1), j <- 0..(k - 1), into: <<>>, do: <<if(f.(i, j), do: one, else: 0)::32-little>>))
  end

  defp all(range, f) do
    Enum.reduce_while(range, {:ok, []}, fn i, {:ok, acc} ->
      case f.(i) do
        {:ok, v} -> {:cont, {:ok, acc ++ [v]}}
        err -> {:halt, err}
      end
    end)
  end

  # ------------------------------------------------------------ positions --

  # the tables a program needs, bound once: global, and local for models
  # whose sliding layers use their own base (Gemma 3)
  defp tables(c, s, freq_factors, lets) do
    {cos, sin} = rope_tables(c, s, freq_factors)
    {cg, lets} = bind(lets, :rope_cos, T.const(cos))
    {sg, lets} = bind(lets, :rope_sin, T.const(sin))

    case c.rope_local do
      nil ->
        {%{global: {cg, sg}}, lets}

      {theta, scaling} ->
        {cl, sl} = rope_tables(%{c | rope_theta: theta, rope_scaling: scaling, rope_local: nil}, s, nil)
        {cl, lets} = bind(lets, :rope_cos_local, T.const(cl))
        {sl, lets} = bind(lets, :rope_sin_local, T.const(sl))
        {%{global: {cg, sg}, local: {cl, sl}}, lets}
    end
  end

  # a layer's sliding window — Hugging Face's `kv > q − w`: the last `w`
  # positions — or nil (full attention; or a window the cache cannot exceed,
  # which never binds and is left out so the program stays the same)
  defp window(%{c: %Config{sliding_window: w} = c, s: s}, l) when is_integer(w) and w < s do
    if c.layer_types == nil or Enum.at(c.layer_types, l) == :sliding, do: w, else: nil
  end

  defp window(_ctx, _l), do: nil

  defp layer_tables(%{c: %Config{layer_types: types}, tables: %{local: local, global: global}}, l) when is_list(types) do
    if Enum.at(types, l) == :sliding, do: local, else: global
  end

  defp layer_tables(%{tables: tables}, _l), do: tables.global

  @doc """
  RoPE tables `cos, sin : f32[S, dh/2]`, following the Hugging Face
  computation in binary32: `inv_freq[i] = 1/θ^(2i/dim)` (with `linear`,
  `llama3` or `yarn` scaling), `freq = pos · inv_freq`, then cos/sin, each
  step rounded to binary32. Every transcendental step is **correctly
  rounded** (`Vapor.CR`), so the tables — constants inside certified
  programs — are the same bits on every host, whatever its libm. They are
  computed once, in the BEAM, and are part of the certified program.

  For latent attention (`mla`) the table spans the whole head layout
  (`mla_layout/1`): the rotary pairs get the frequencies of
  `qk_rope_head_dim`, the pass-through pairs cos = 1, sin = 0.
  """
  def rope_tables(c, s, freq_factors \\ nil)

  def rope_tables(%Config{mla: %{rope: dr}} = c, s, freq_factors) do
    half = div(c.head_dim, 2)
    pass = half - div(dr, 2)
    {inv, af} = inv_freq(c, dr, freq_factors)
    rot = for p <- 0..(s - 1), do: Enum.map(inv, &angle(p, &1, af))
    one = {1.0, 0.0}

    rows = for r <- rot, do: List.duplicate(one, pass) ++ r
    flat = List.flatten(rows)
    {Tensor.from_list(:f32, [s, half], Enum.map(flat, &elem(&1, 0))), Tensor.from_list(:f32, [s, half], Enum.map(flat, &elem(&1, 1)))}
  end

  # partial rotary: the rotating pairs get the frequencies of `rotary_dim`,
  # the pass-through pairs cos = 1, sin = 0 — the layout of `rotary_layout/1`,
  # folded into the q/k rows by `std_attention`
  def rope_tables(%Config{rotary_dim: r} = c, s, freq_factors) when is_integer(r) do
    half = div(c.head_dim, 2)
    pass = half - div(r, 2)
    {inv, af} = inv_freq(c, r, freq_factors)
    one = {1.0, 0.0}
    rows = for p <- 0..(s - 1), do: List.duplicate(one, pass) ++ Enum.map(inv, &angle(p, &1, af))
    flat = List.flatten(rows)
    {Tensor.from_list(:f32, [s, half], Enum.map(flat, &elem(&1, 0))), Tensor.from_list(:f32, [s, half], Enum.map(flat, &elem(&1, 1)))}
  end

  def rope_tables(c, s, freq_factors) do
    half = div(c.head_dim, 2)
    {inv, af} = inv_freq(c, c.head_dim, freq_factors)
    cs = for p <- 0..(s - 1), iv <- inv, do: angle(p, iv, af)
    {Tensor.from_list(:f32, [s, half], Enum.map(cs, &elem(&1, 0))), Tensor.from_list(:f32, [s, half], Enum.map(cs, &elem(&1, 1)))}
  end

  # freq = p·inv rounded to binary32, then cos/sin correctly rounded
  # (× the YaRN attention factor, rounded again, when there is one)
  defp angle(p, iv, af) do
    a = f32(p * iv)
    {c, s} = {CR.cos_f32(a), CR.sin_f32(a)}
    if af == 1.0, do: {c, s}, else: {f32(c * af), f32(s * af)}
  end

  # inverse frequencies over `dim` rotary dimensions, and the attention factor
  defp inv_freq(c, dim, freq_factors) do
    half = div(dim, 2)
    theta = f32(c.rope_theta)
    # pos_freqs[i] = θ^(f32(2i)/dim) in binary32, correctly rounded
    pf = for i <- 0..(half - 1), do: CR.pow_f32(theta, f32(f32(2.0 * i) / dim))
    base = Enum.map(pf, &f32(1.0 / &1))

    case {c.rope_scaling, freq_factors} do
      {{:yarn, y}, _} -> yarn_inv(pf, base, y, dim, c.rope_theta)
      {sc, ff} -> {scale(base, sc, ff), 1.0}
    end
  end

  defp scale(inv, nil, _), do: inv
  defp scale(inv, {:linear, factor}, _), do: Enum.map(inv, &f32(&1 / factor))

  # GGUF's Llama 3 form: a divisor per frequency (llama.cpp's rope_freqs)
  defp scale(inv, :freq_factors, %Tensor{} = f), do: Enum.zip_with(inv, Tensor.to_floats(f), &f32(&1 / &2))

  defp scale(inv, {:llama3, factor, low, high, orig}, _) do
    low_wl = orig / low
    high_wl = orig / high

    Enum.map(inv, fn iv ->
      wl = 2 * :math.pi() / iv

      cond do
        wl < high_wl -> iv
        wl > low_wl -> f32(iv / factor)
        true ->
          smooth = (orig / wl - low) / (high - low)
          f32((1 - smooth) * iv / factor + smooth * iv)
      end
    end)
  end

  # transformers' `_compute_yarn_parameters`, step for step: the correction
  # range in binary64 (logs correctly rounded), the ramp and the blend in binary32
  defp yarn_inv(pf, extrap, y, dim, base) do
    factor = y.factor
    interp = Enum.map(pf, &f32(1.0 / f32(f32(factor) * &1)))
    corr = fn rot -> dim * CR.log_f64(y.orig / (rot * 2 * :math.pi())) / (2 * CR.log_f64(base * 1.0)) end
    {low, high} = {corr.(y.beta_fast), corr.(y.beta_slow)}
    {low, high} = if y.truncate, do: {Float.floor(low), Float.ceil(high)}, else: {low, high}
    {low, high} = {max(low, 0), min(high, dim - 1)}
    high = if low == high, do: high + 0.001, else: high
    span = f32(high - low)
    lo32 = f32(low * 1.0)

    ramp = for i <- 0..(div(dim, 2) - 1), do: f32(f32(f32(i * 1.0) - lo32) / span) |> max(0.0) |> min(1.0)
    ef = Enum.map(ramp, &f32(1.0 - &1))

    inv =
      Enum.zip_with([interp, extrap, ef], fn [i, e, w] ->
        f32(f32(i * f32(1.0 - w)) + f32(e * w))
      end)

    mscale = fn s, m -> if s <= 1, do: 1.0, else: 0.1 * m * CR.log_f64(s) + 1.0 end

    af =
      cond do
        is_number(y.attention_factor_given) -> y.attention_factor_given * 1.0
        y.mscale && y.mscale_all_dim -> mscale.(factor, y.mscale) / mscale.(factor, y.mscale_all_dim)
        true -> mscale.(factor, 1)
      end

    {inv, f32(af)}
  end

  defp f32(x), do: Vapor.F32.to_float(Vapor.F32.from_float(x * 1.0))
end
