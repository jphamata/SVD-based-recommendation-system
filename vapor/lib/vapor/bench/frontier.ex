defmodule Vapor.Bench.Frontier do
  @moduledoc """
  Measurements for the 0.6 round (`mix vapor.bench --frontier`, written to
  `docs/bench/FRONTIER.md`): what each new capability costs or saves, on
  this machine, now — and, next to every speed, whether the bits held.

    * sparse experts: decode and prefill of a Mixtral-shaped model, dense
      against predicated dispatch (same output bits checked);
    * latent attention: cache floats per token and layer, decode-step time,
      latent against expanded (DeepSeek-V3-shaped heads, scaled down);
    * sliding window: attention time over a long cache, windowed against full;
    * correctly rounded division: microprogram length and throughput,
      against the old `a·rcp(b)`;
    * spatial: a VAE decoder (latents → pixels) and a DiT step;
    * the transparency log: appends, proofs and verifications per second;
    * tensor parallelism: one worker against column shards (bits checked);
    * state-space decoding (Mamba): per-token cost and memory at any
      context, against attention's growing cache;
    * the circular KV cache: pages a sequence holds, ring against uncapped;
    * Whisper: the encoder over 30 s of audio at whisper-tiny's widths, and
      a decoding step;
    * tree speculation: tokens per target step, linear against tree, prompt
      lookup against a draft that is always wrong (output equality checked).

  Random weights throughout: these are measurements of the machinery, not
  of model quality (that is `mix vapor.quality`).
  """
  alias Vapor.{Program, Spatial, Tensor, Tlog}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Model.{Config, Llama}
  alias Vapor.Runtime.{Native, Substrates, Worker}

  @reps 7

  def run(opts \\ []) do
    out = Keyword.get(opts, :out, "docs/bench")
    log = Keyword.get(opts, :log, &IO.puts/1)
    File.mkdir_p!(out)
    {:ok, w} = Worker.start_link(exec: [Substrates.binary("vapor-worker", "native")], threads: System.schedulers_online())
    isa = Substrates.host_isa()

    log.("sparse experts…")
    moe = moe(w, isa)
    log.("latent attention…")
    mla = mla(w, isa)
    log.("sliding window…")
    win = window(w, isa)
    log.("division…")
    dv = division(w, isa)
    log.("spatial…")
    sp = spatial(w, isa)
    log.("transparency log…")
    tl = tlog()
    log.("tensor parallelism…")
    sh = shard(isa)
    log.("state-space decoding…")
    ssm = ssm(w, isa)
    log.("circular cache…")
    ring = ring()
    log.("whisper…")
    wh = whisper(w, isa)
    log.("tree speculation…")
    tree = tree(isa)

    File.write!(Path.join(out, "FRONTIER.md"), report(moe, mla, win, dv, sp, tl, sh) <> report2(ssm, ring, wh, tree))
    :ok
  end

  # median wall time inside the worker over @reps runs; and the outputs
  defp timed(w, comp, env, isa) do
    runs = for _ <- 1..@reps, do: elem(Native.run(w, comp, env, isa: isa, mode: :native), 1)
    {runs |> Enum.map(& &1.elapsed_ns) |> Enum.sort() |> Enum.at(div(@reps, 2)) |> Kernel./(1.0e6), hd(runs).outputs}
  end

  defp lower!(p), do: elem(Lower.lower(p), 1)

  defp tiny(arch, over) do
    {:ok, c} = Config.from_map(Map.merge(%{"model_type" => arch, "vocab_size" => 256, "max_position_embeddings" => 512,
                                           "rms_norm_eps" => 1.0e-5, "rope_theta" => 10_000.0, "hidden_act" => "silu"}, over))
    ws = for {name, shape, kind} <- Llama.expected_weights(c), into: %{} do
      t = Tensor.random(:f32, shape, :erlang.phash2(name), scale: if(kind == :norm, do: 0.05, else: 0.2))
      {name, if(kind == :norm, do: Tensor.from_list(:f32, shape, Enum.map(Tensor.to_floats(t), &(&1 + 1.0))), else: t)}
    end

    {c, ws}
  end

  defp env(c, toks, s, opts \\ []) do
    n = length(toks)
    Map.merge(Llama.empty_caches(c, s, opts), %{tok: Tensor.from_list(:s32, [n], toks), pos: Tensor.from_list(:s32, [n], Enum.to_list(0..(n - 1)))})
  end

  defp moe(w, isa) do
    {c, ws} = tiny("mixtral", %{"hidden_size" => 512, "intermediate_size" => 1024, "num_hidden_layers" => 2, "num_attention_heads" => 8,
                                "num_key_value_heads" => 2, "num_local_experts" => 8, "num_experts_per_tok" => 2})

    for t <- [1, 8, 32] do
      toks = Enum.map(1..t, &rem(&1 * 37, 250))
      [{dm, d}, {sm, s}] =
        for mode <- [:dense, :sparse] do
          {:ok, p} = Llama.program(c, ws, max_seq: 64, moe: mode)
          timed(w, lower!(p), env(c, toks, 64), isa)
        end

      %{tokens: t, dense_ms: dm, sparse_ms: sm, same_bits: d == s}
    end
  end

  defp mla(w, isa) do
    over = %{"hidden_size" => 256, "intermediate_size" => 512, "num_hidden_layers" => 2, "num_attention_heads" => 16,
             "num_key_value_heads" => 16, "q_lora_rank" => 64, "kv_lora_rank" => 128, "qk_nope_head_dim" => 32,
             "qk_rope_head_dim" => 16, "v_head_dim" => 32, "n_routed_experts" => 4, "num_experts_per_tok" => 2,
             "moe_intermediate_size" => 128, "n_shared_experts" => 1, "first_k_dense_replace" => 2, "n_group" => 1, "topk_group" => 1}
    {c, ws} = tiny("deepseek_v3", over)
    s = 256

    for mode <- [:expanded, :latent] do
      {:ok, p} = Llama.program(c, ws, max_seq: s, mla: mode)
      comp = lower!(p)
      # decode one token at position s − 1 against a full cache
      e = Map.merge(Llama.empty_caches(c, s, mla: mode), %{tok: Tensor.from_list(:s32, [1], [7]), pos: Tensor.from_list(:s32, [1], [s - 1])})
      {ms, _} = timed(w, comp, e, isa)
      %{mode: mode, floats: Llama.cache_floats(c, mla: mode), ms: ms, cache_mb: Llama.cache_floats(c, mla: mode) * 4 * s * c.layers / 1.0e6}
    end
  end

  defp window(w, isa) do
    {h, hkv, dh, s} = {8, 2, 64, 4096}
    e = %{q: Tensor.random(:f32, [1, h * dh], 1), k: Tensor.random(:f32, [s, hkv * dh], 2), v: Tensor.random(:f32, [s, hkv * dh], 3),
          pos: Tensor.from_list(:s32, [1], [s - 1])}

    for win <- [nil, 1024, 256] do
      q = T.input(:q, :f32, [1, h * dh]); k = T.input(:k, :f32, [s, hkv * dh]); v = T.input(:v, :f32, [s, hkv * dh])
      p = Program.new(o: T.attention(q, k, v, T.input(:pos, :s32, [1]), h, hkv, nil, win))
      {ms, _} = timed(w, lower!(p), e, isa)
      %{window: win, ms: ms}
    end
  end

  defp division(w, isa) do
    n = 1 <<< 20
    a = Tensor.random(:f32, [n], 4)
    b = Tensor.random(:f32, [n], 5)
    len = fn op, args -> {ops, _, _} = Vapor.Canon.expand(op, args, 0); length(ops) end
    new_len = len.(:div, [{:in, 0}, {:in, 1}])
    old_len = len.(:rcp, [{:in, 0}]) + 1

    p_new = Program.new(y: T.divide(T.input(:a, :f32, [n]), T.input(:b, :f32, [n])))
    p_old = Program.new(y: T.mul(T.input(:a, :f32, [n]), T.rcp(T.input(:b, :f32, [n]))))
    {new_ms, _} = timed(w, lower!(p_new), %{a: a, b: b}, isa)
    {old_ms, _} = timed(w, lower!(p_old), %{a: a, b: b}, isa)
    %{n: n, new_ops: new_len, old_ops: old_len, new_ms: new_ms, old_ms: old_ms}
  end

  defp spatial(w, isa) do
    # a VAE decoder block stack at 16×16 latents (64 channels), and a DiT-like
    # conv + attention layer: the machinery's cost on this machine
    x = T.input(:x, :f32, [256, 64])
    wt = fn i, o, ci, k -> Tensor.random(:f32, [o, ci, k, k], i, scale: 0.05) end
    bias = fn i, o -> Tensor.random(:f32, [o], i, scale: 0.05) end
    one = fn i -> Tensor.from_list(:f32, [64], List.duplicate(1.0, 64)) |> then(fn t -> {t, bias.(i, 64)} end) end
    {gw, gb} = one.(1)
    {lets, y} = Spatial.group_norm([], x, 256, 64, 8, "gn", gw, gb, 1.0e-6)
    {lets, y, d} = Spatial.conv2d(lets, T.silu(y), {16, 16, 64}, "c1", wt.(2, 64, 64, 3), bias.(3, 64), padding: 1)
    {lets, y, d} = Spatial.upsample_nearest(lets, y, d, "up")
    {lets, y, _} = Spatial.conv2d(lets, y, d, "c2", wt.(4, 64, 64, 3), bias.(5, 64), padding: 1)
    comp = lower!(Program.new([y: y], lets: Enum.reverse(lets)))
    {ms, _} = timed(w, comp, %{x: Tensor.random(:f32, [256, 64], 6)}, isa)
    flops = 2 * (256 * 64 * 576) + 2 * (1024 * 64 * 576)
    [%{what: "GroupNorm + SiLU + conv 3×3 (16×16×64) + upsample ×2 + conv 3×3 (32×32×64)", ms: ms, gflops: flops / (ms * 1.0e6)}]
  end

  defp tlog do
    n = 4096
    {t_app, log} = :timer.tc(fn -> Enum.reduce(1..n, Tlog.new("bench.vapor/log"), &elem(Tlog.append(&2, "entry #{&1}"), 0)) end)
    root = Tlog.root(log)
    {t_proof, proofs} = :timer.tc(fn -> for i <- 0..(n - 1)//16, do: {i, Tlog.inclusion(log, i)} end)
    {t_ver, oks} = :timer.tc(fn -> for {i, p} <- proofs, do: Tlog.verify_inclusion(Tlog.leaf_hash("entry #{i + 1}"), i, n, p, root) end)
    {t_cons, ok_c} = :timer.tc(fn -> Tlog.verify_consistency(1000, n, Tlog.consistency(log, 1000), Tlog.root(log, 1000), root) end)
    %{entries: n, append_per_s: n / (t_app / 1.0e6), proof_per_s: length(proofs) / (t_proof / 1.0e6),
      verify_per_s: length(proofs) / (t_ver / 1.0e6), all_ok: Enum.all?(oks) and ok_c, consistency_ms: t_cons / 1000, proof_len: length(elem(hd(proofs), 1))}
  end

  defp shard(isa) do
    workers = for _ <- 1..2, do: elem(Worker.start_link(exec: [Substrates.binary("vapor-worker", "native")]), 1)
    x = Tensor.random(:f32, [8, 1024], 1)
    wt = Tensor.random(:f32, [2048, 1024], 2, scale: 0.05)
    {t1, {:ok, y1}} = :timer.tc(fn -> Vapor.Shard.linear([hd(workers)], x, wt, isa: isa) end)
    {t2, {:ok, y2}} = :timer.tc(fn -> Vapor.Shard.linear(workers, x, wt, isa: isa) end)
    {drift, total} = Vapor.Shard.row_parallel_drift(workers, x, wt, isa: isa)
    %{one_ms: t1 / 1000, two_ms: t2 / 1000, same_bits: y1 == y2, drift: drift, total: total}
  end

  # a session step's median wall time (ms) inside the worker
  defp step_ms(session, inputs, outs) do
    runs = for _ <- 1..@reps, do: elem(Vapor.Runtime.Session.step(session, inputs, outs), 2).elapsed_ns
    runs |> Enum.sort() |> Enum.at(div(@reps, 2)) |> Kernel./(1.0e6)
  end

  defp ssm(w, isa) do
    alias Vapor.Lock.Adapters.Mamba
    alias Vapor.Runtime.Session
    cfg = %{"model_type" => "mamba", "vocab_size" => 256, "hidden_size" => 256, "intermediate_size" => 512, "num_hidden_layers" => 4,
            "state_size" => 16, "time_step_rank" => 16, "conv_kernel" => 4}
    spec = Mamba.spec(struct(Mamba.Config, vocab: 256, hidden: 256, inner: 512, layers: 4, state: 16, rank: 16, conv: 4, eps: 1.0e-5,
                                            bias: false, conv_bias: true, tie: true, raw: cfg))
    ws = for {name, shape, kind} <- Vapor.Lock.expected(spec), into: %{} do
      t = Tensor.random(:f32, shape, :erlang.phash2(name), scale: 0.1)
      {name, if(kind == :norm, do: Tensor.from_list(:f32, shape, Enum.map(Tensor.to_floats(t), &(&1 + 1.0))), else: t)}
    end

    {:ok, g} = Vapor.Recurrent.open(spec, ws, worker: w, isa: isa)
    tok = %{tok: Tensor.from_list(:s32, [1], [7])}
    # warm the session (first touches of its buffers), then a short context
    {_, g, _} = Vapor.Recurrent.prefill(g, [1, 2, 3])
    early = step_ms(g.session, tok, [:logits])
    {_, g, _} = Vapor.Recurrent.prefill(g, Enum.map(1..1000, &rem(&1 * 31, 256)))
    late = step_ms(g.session, tok, [:logits])
    state_floats = Mamba.state(spec.config) |> Enum.map(fn {_, sh} -> Enum.product(sh) end) |> Enum.sum()

    # attention at the same widths: a Llama of d 256, 4 layers, 8 heads (2 KV)
    {c, lws} = tiny("llama", %{"hidden_size" => 256, "intermediate_size" => 512, "num_hidden_layers" => 4, "num_attention_heads" => 8,
                               "num_key_value_heads" => 2, "max_position_embeddings" => 8192})

    attn =
      for s <- [64, 1024, 8192] do
        {:ok, p} = Llama.program(c, lws, max_seq: s, max_tokens: 1)
        {:ok, sess} = Session.open(w, lower!(p), isa: isa)
        ms = step_ms(sess, %{tok: Tensor.from_list(:s32, [1], [7]), pos: Tensor.from_list(:s32, [1], [s - 1])}, [:logits])
        Session.close(sess)
        %{context: s, ms: ms, kv_floats: Llama.cache_floats(c) * c.layers * s}
      end

    %{early_ms: early, late_ms: late, state_floats: state_floats, attn: attn}
  end

  defp ring do
    # the engine's own accounting on a small model: a window of 32 binding on every layer
    {:ok, c} = Config.from_map(%{"model_type" => "mistral", "vocab_size" => 128, "hidden_size" => 64, "intermediate_size" => 96,
                                 "num_hidden_layers" => 2, "num_attention_heads" => 4, "num_key_value_heads" => 2,
                                 "max_position_embeddings" => 512, "rms_norm_eps" => 1.0e-5, "rope_theta" => 10_000.0,
                                 "sliding_window" => 32})
    ws = for {name, shape, _} <- Llama.expected_weights(c), into: %{}, do: {name, Tensor.random(:f32, shape, :erlang.phash2(name), scale: 0.1)}
    {:ok, e} = Vapor.Engine.start_link(config: c, weights: ws, max_seq: 512, page: 16, sequences: 4, step_tokens: 32)
    info = Vapor.Engine.info(e)
    GenServer.stop(e)
    ring = fn w, step, page -> div(w + step - 1 + page - 1, page) end
    %{engine: %{window: info.window, ring: info.ring_pages, uncapped: div(512, 16)},
      examples: for({ctx, w} <- [{32_768, 4096}, {131_072, 4096}, {8192, 2047}], do: %{context: ctx, window: w, ring: ring.(w, 64, 16), uncapped: div(ctx, 16)})}
  end

  defp whisper(w, isa) do
    alias Vapor.Lock.Adapters.Whisper
    raw = %{"model_type" => "whisper", "d_model" => 384, "encoder_layers" => 4, "decoder_layers" => 4, "encoder_attention_heads" => 6,
            "decoder_attention_heads" => 6, "encoder_ffn_dim" => 1536, "decoder_ffn_dim" => 1536, "num_mel_bins" => 80,
            "max_source_positions" => 1500, "max_target_positions" => 448, "vocab_size" => 4096, "decoder_start_token_id" => 1, "eos_token_id" => 2}
    spec0 = Whisper.spec(struct(Whisper.Config, vocab: 4096, d: 384, mels: 80, src: 1500, tgt: 448, enc_layers: 4, dec_layers: 4, enc_heads: 6,
                                                dec_heads: 6, enc_ffn: 1536, dec_ffn: 1536, eps: 1.0e-5, start: 1, eos: 2, tie: true, raw: raw))
    ws = for {name, shape, kind} <- Vapor.Lock.expected(spec0), into: %{} do
      t = Tensor.random(:f32, shape, :erlang.phash2(name), scale: 0.05)
      {name, if(kind == :norm, do: Tensor.from_list(:f32, shape, Enum.map(Tensor.to_floats(t), &(&1 + 1.0))), else: t)}
    end

    {:ok, spec, _} = Whisper.admit(%{config: raw}, ws, [])
    {:ok, ep} = Vapor.Lock.build(spec, ws, part: :encoder)
    {t_lower, comp} = :timer.tc(fn -> lower!(ep) end)
    env = Whisper.encoder_input(spec, Tensor.random(:f32, [80, 3000], 3))
    {enc_ms, out} = timed(w, comp, env, isa)

    {:ok, dp} = Vapor.Lock.build(spec, ws, max_seq: 448, max_tokens: 1)
    {:ok, sess} = Vapor.Runtime.Session.open(w, lower!(dp), isa: isa)
    cross = for l <- 0..3, kv <- ["xk", "xv"], into: %{}, do: {:"#{kv}#{l}", out[:"#{kv}#{l}"]}
    ids = &Tensor.from_list(:s32, [1], [&1])
    dec_ms = step_ms(sess, Map.merge(cross, %{tok: ids.(5), pos: ids.(100), xh: ids.(1499)}), [:logits])
    Vapor.Runtime.Session.close(sess)
    %{encoder_ms: enc_ms, lower_ms: t_lower / 1000, decode_ms: dec_ms}
  end

  defp tree(isa) do
    alias Vapor.Speculative.Tree
    {:ok, w} = Worker.start_link(exec: [Substrates.binary("vapor-worker", "native")])
    text = String.to_charlist("o gato subiu no muro. o gato comeu o rato. o rato subiu no telhado. ")
    pl = Vapor.Quality.Planted.bigram(text, 128)
    {:ok, pc} = Config.from_map(pl.config)
    {rc, rws} = tiny("llama", %{"vocab_size" => 128, "hidden_size" => 128, "intermediate_size" => 256, "num_hidden_layers" => 2,
                                "num_attention_heads" => 4, "num_key_value_heads" => 2})
    junk = fn ctx, b, k -> for i <- 1..b, do: List.duplicate(rem(List.last(ctx) + 50 + i, 128), k) end
    n = 96

    for {model, c, ws} <- [{"planted bigram", pc, pl.weights}, {"random weights", rc, rws}],
        {label, opts} <- [{"no draft", [branches: 1, depth: 0]}, {"prompt lookup, linear", [branches: 1, depth: 6]},
                          {"prompt lookup, tree of 4", [branches: 4, depth: 6]}, {"draft always wrong, tree of 4", [branches: 4, depth: 6, draft: junk]}] do
      {:ok, tr} = Tree.open(c, ws, w, max_seq: 256, page: 8, branches: 4, max_tokens: 64, isa: isa)
      {t, {ids, st}} = :timer.tc(fn -> Tree.generate(tr, text, n, opts) end)
      Vapor.Runtime.Session.close(tr.session)
      %{model: model, label: label, ids: ids, steps: st.target_steps - 1, per_step: n / max(st.target_steps - 1, 1), ms: t / 1000, rows: st.rows}
    end
    |> Enum.group_by(& &1.model)
    |> Enum.map(fn {m, rs} -> {m, Enum.map(rs, &Map.put(&1, :same, &1.ids == hd(rs).ids))} end)
  end

  # ---------------------------------------------------------------- report --

  defp f(x, d \\ 2), do: :erlang.float_to_binary(x * 1.0, decimals: d)

  defp report(moe, mla, win, dv, sp, tl, sh) do
    """
    # Round 0.6 measurements — vapor

    Generated by `mix vapor.bench --frontier` on #{Date.utc_today()}, on this machine
    (#{System.schedulers_online()} cores). Random weights: they measure the machinery, not the quality.
    Times are medians of #{@reps} runs inside the worker; they depend on the host's load.

    ## Sparse experts (scaled-down Mixtral: d 512, 8 experts, top-2, 2 layers)

    | tokens in the step | dense ms | sparse ms | speedup | same bits |
    |---|---|---|---|---|
    #{Enum.map_join(moe, "\n", &"| #{&1.tokens} | #{f(&1.dense_ms)} | #{f(&1.sparse_ms)} | #{f(&1.dense_ms / &1.sparse_ms)}× | #{&1.same_bits} |")}

    ## Latent attention (MLA, 16 heads, decode with the cache full at 256 positions)

    | form | floats/token/layer | cache (MB, 256 pos.) | step ms |
    |---|---|---|---|
    #{Enum.map_join(mla, "\n", &"| #{&1.mode} | #{&1.floats} | #{f(&1.cache_mb, 3)} | #{f(&1.ms)} |")}

    At DeepSeek-V3's shapes (128 heads, 512 + 64 latents): 576 against 49,152 floats — 85×.

    ## Sliding window (8 heads, dh 64, a cache of 4,096 positions, one query at the last)

    | window | ms |
    |---|---|
    #{Enum.map_join(win, "\n", &"| #{&1.window || "none"} | #{f(&1.ms, 3)} |")}

    ## Correctly rounded division (#{dv.n} quotients)

    | microprogram | primitives | ms |
    |---|---|---|
    | `a·rcp(b)` (up to 0.5, gets the rounding wrong in ~20 %) | #{dv.old_ops} | #{f(dv.old_ms)} |
    | correct (IEEE, DAZ/FTZ) | #{dv.new_ops} | #{f(dv.new_ms)} |

    ## Spatial (convolution without a convolution kernel)

    | block | ms | GFLOP/s |
    |---|---|---|
    #{Enum.map_join(sp, "\n", &"| #{&1.what} | #{f(&1.ms)} | #{f(&1.gflops)} |")}

    ## Transparency log (#{tl.entries} entries)

    | operation | per second |
    |---|---|
    | append (with the dense tree) | #{f(tl.append_per_s, 0)} |
    | inclusion proof (#{tl.proof_len} hashes) | #{f(tl.proof_per_s, 0)} |
    | inclusion verification | #{f(tl.verify_per_s, 0)} |

    Consistency proof 1,000 → #{tl.entries} verified in #{f(tl.consistency_ms, 3)} ms; all verifications: #{tl.all_ok}.

    ## Tensor parallelism (GEMV 2048×1024, 8 rows)

    | | ms (end to end, with compilation) |
    |---|---|
    | 1 worker | #{f(sh.one_ms)} |
    | 2 workers, columns | #{f(sh.two_ms)} |

    Equal bits between 1 and 2 workers: #{sh.same_bits}. The row-wise form (k split, sum of the partials)
    would change #{sh.drift} of #{sh.total} elements — which is why the exact MLP uses all-gather.
    """
  end

  defp report2(ssm, ring, wh, tree) do
    """

    ## State-space decoding (Mamba: d 256, inner 512, state 16, 4 layers)

    | | step ms | memory per sequence (floats) |
    |---|---|---|
    | Mamba, a context of ~10 tokens | #{f(ssm.early_ms, 3)} | #{ssm.state_floats} (fixed state) |
    | Mamba, after 1,000 tokens | #{f(ssm.late_ms, 3)} | #{ssm.state_floats} |
    #{Enum.map_join(ssm.attn, "\n", &"| attention (Llama, the same widths), context #{&1.context} | #{f(&1.ms, 3)} | #{&1.kv_floats} (KV cache) |")}

    An SSM's step does not depend on the context; attention's grows with it (and the cache, linearly).

    ## Circular KV cache (a window on every layer)

    Real engine (scaled-down Mistral, window #{ring.engine.window}, context 512, pages of 16, 32 tokens per step):
    #{ring.engine.ring} pages per sequence in the ring against #{ring.engine.uncapped} without it.

    | context | window | pages in the ring (page 16, 64 tokens/step) | without ring | extra sequences in the same memory |
    |---|---|---|---|---|
    #{Enum.map_join(ring.examples, "\n", &"| #{&1.context} | #{&1.window} | #{&1.ring} | #{&1.uncapped} | #{f(&1.uncapped / &1.ring, 1)}× |")}

    ## Whisper (whisper-tiny's widths: d 384, 4 + 4 layers, 6 heads; vocabulary reduced to 4,096)

    | | ms |
    |---|---|
    | compile the encoder (once) | #{f(wh.lower_ms, 0)} |
    | encoder over 30 s of audio (3,000 mel frames → 1,500 positions) | #{f(wh.encoder_ms, 1)} |
    | one decoder step (cross-attention over 1,500 positions) | #{f(wh.decode_ms, 3)} |

    ## Tree speculation (#{tree |> hd() |> elem(1) |> hd() |> Map.get(:ids) |> length()} greedy tokens; pages of 8)

    #{Enum.map_join(tree, "\n\n", fn {m, rs} -> "**#{m}**\n\n| draft | target steps | tokens/step | rows computed | ms | output = greedy |\n|---|---|---|---|---|---|\n" <> Enum.map_join(rs, "\n", &"| #{&1.label} | #{&1.steps} | #{f(&1.per_step)} | #{&1.rows} | #{f(&1.ms, 0)} | #{&1.same} |") end)}

    The output is the target's greedy decoding in every row; the draft changes only the number of steps.
    With random weights the output does not copy the context and the lookup rarely hits — measured, not hidden.
    """
  end

  defp a <<< b, do: Bitwise.bsl(a, b)
end
