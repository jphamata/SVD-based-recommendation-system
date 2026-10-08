defmodule Vapor.Quality.Round06 do
  @moduledoc """
  Quality checks for the 0.6 round, in the suite's discipline: every check
  has a value, a **control** a broken implementation would produce, and a
  threshold that separates them — a check whose control passes too is not
  evidence of anything.

  | check | value | control (must fail) |
  |---|---|---|
  | sparse experts = dense | elements differing between `moe: :sparse` and `:dense` (0) | the same after perturbing one weight of an expert the tokens *select* (> 0: the comparison sees the experts) |
  | latent MLA = expanded | argmax agreement over every position (1.0) | agreement with a model whose key up-projection is permuted (low) |
  | sliding window | bits differing from the unwindowed attention over the window moved to the front (0) | bits differing from full attention (> 0) |
  | correctly rounded ÷ | IEEE mismatches in random pairs (0) | mismatches of the old `a·rcp(b)` (> 10 %) |
  | convolution | relative error against a direct binary64 convolution (≤ 10⁻⁶) | the same against the flipped kernel — convolution confused with correlation (≫) |
  | transparency log | transparency-dev probes answered correctly (all) | a verifier that trusts the proof's own count of steps (accepts truncated proofs) |
  | state-space step | relative distance of a Mamba's logits, the native session (state fed back inside the worker) against the oracle's recurrence (0) | the oracle with the state reset before each token — a Mamba without memory (≫ 0) |
  | tree speculation | tokens differing from the target's plain greedy decoding (0) | a verifier that accepts the drafts unchecked (≫ 0) |
  """
  alias Vapor.{Canon, F32, Program, Spatial, Tensor, Tlog}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Model.{Config, Decoder}
  alias Vapor.Runtime.Oracle
  import Bitwise

  def run(opts \\ []) do
    w = Keyword.get(opts, :worker)
    checks = [experts(), latent(), window(), division(), convolution(), tlog(), ssm(w), tree(w)] |> List.flatten()
    %{checks: checks}
  end

  defp check(name, value, control, threshold, pass), do: %{name: name, value: value, control: control, threshold: threshold, pass: pass}

  defp cfg(arch, over) do
    {:ok, c} = Config.from_map(Map.merge(%{"model_type" => arch, "vocab_size" => 96, "hidden_size" => 64, "intermediate_size" => 96,
                                           "num_hidden_layers" => 2, "num_attention_heads" => 4, "num_key_value_heads" => 2,
                                           "max_position_embeddings" => 64, "rms_norm_eps" => 1.0e-5, "rope_theta" => 10_000.0,
                                           "hidden_act" => "silu"}, over))
    ws = for {name, shape, kind} <- Decoder.expected_weights(c), into: %{} do
      t = Tensor.random(:f32, shape, :erlang.phash2(name), scale: 0.2)
      {name, if(kind == :norm, do: Tensor.from_list(:f32, shape, Enum.map(Tensor.to_floats(t), &(&1 + 1.0))), else: t)}
    end

    {c, ws}
  end

  defp env(c, toks, opts \\ []) do
    n = length(toks)
    Map.merge(Decoder.empty_caches(c, 16, opts), %{tok: Tensor.from_list(:s32, [n], toks), pos: Tensor.from_list(:s32, [n], Enum.to_list(0..(n - 1)))})
  end

  defp diff(a, b), do: Enum.count(Enum.zip(F32.decode(a.data), F32.decode(b.data)), fn {x, y} -> x != y end)

  defp experts do
    {c, ws} = cfg("mixtral", %{"num_local_experts" => 4, "num_experts_per_tok" => 2})
    toks = [3, 50, 7, 81, 12]
    run = fn ws, mode -> {:ok, p} = Decoder.program(c, ws, max_seq: 16, moe: mode); Oracle.eval_program(p, env(c, toks)).logits end
    dense = run.(ws, :dense)
    sparse = run.(ws, :sparse)
    # perturb an expert every token routes through: expert 0 of layer 0 is
    # selected by some token with these weights (checked below by the effect)
    name = "model.layers.0.block_sparse_moe.experts.0.w2.weight"
    t = ws[name]
    bumped = Map.put(ws, name, Tensor.from_list(:f32, t.shape, Enum.map(Tensor.to_floats(t), &(&1 * 1.5))))
    control = diff(run.(bumped, :sparse), dense)
    check("experts: elements differing, sparse dispatch vs dense", diff(sparse, dense), control, "0, control > 0",
          diff(sparse, dense) == 0 and control > 0)
  end

  defp latent do
    over = %{"q_lora_rank" => 32, "kv_lora_rank" => 32, "qk_nope_head_dim" => 16, "qk_rope_head_dim" => 16, "v_head_dim" => 24,
             "n_routed_experts" => 4, "num_experts_per_tok" => 2, "moe_intermediate_size" => 32, "n_shared_experts" => 1,
             "first_k_dense_replace" => 1, "n_group" => 1, "topk_group" => 1, "num_key_value_heads" => 4}
    {c, ws} = cfg("deepseek_v3", over)
    toks = [1, 95, 7, 7, 42, 0, 33, 64]
    am = fn l -> l |> Tensor.to_floats() |> Enum.chunk_every(c.vocab) |> Enum.map(fn r -> r |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1) end) end
    run = fn ws, mode -> {:ok, p} = Decoder.program(c, ws, max_seq: 16, mla: mode); am.(Oracle.eval_program(p, env(c, toks, mla: mode)).logits) end
    exp = run.(ws, :expanded)
    lat = run.(ws, :latent)
    kv = "model.layers.0.self_attn.kv_b_proj.weight"
    t = ws[kv]
    [rows, k] = t.shape
    shuffled = Tensor.new(:f32, t.shape, t.data |> F32.decode() |> Enum.chunk_every(k) |> Enum.reverse() |> List.flatten() |> F32.encode())
    _ = rows
    ctl = run.(Map.put(ws, kv, shuffled), :latent)
    agree = fn a, b -> Enum.count(Enum.zip(a, b), fn {x, y} -> x == y end) / length(a) end
    check("latent MLA: argmax agreement with the expanded form", agree.(lat, exp), agree.(ctl, exp), "1.0, control < 1.0",
          agree.(lat, exp) == 1.0 and agree.(ctl, exp) < 1.0)
  end

  defp window do
    {h, hkv, dh, s} = {4, 2, 16, 32}
    mk = fn win, srows -> Program.new(o: T.attention(T.input(:q, :f32, [1, h * dh]), T.input(:k, :f32, [srows, hkv * dh]), T.input(:v, :f32, [srows, hkv * dh]), T.input(:pos, :s32, [1]), h, hkv, nil, win)) end
    {q, k, v} = {Tensor.random(:f32, [1, h * dh], 1), Tensor.random(:f32, [s, hkv * dh], 2), Tensor.random(:f32, [s, hkv * dh], 3)}
    {p, w} = {27, 6}
    win = Oracle.eval_program(mk.(w, s), %{q: q, k: k, v: v, pos: Tensor.from_list(:s32, [1], [p])}).o
    rw = hkv * dh * 4
    s0 = p - w + 1
    front = fn t -> Tensor.new(:f32, [s, hkv * dh], binary_part(t.data, s0 * rw, (s - s0) * rw) <> :binary.copy(<<0>>, s0 * rw)) end
    moved = Oracle.eval_program(mk.(nil, s), %{q: q, k: front.(k), v: front.(v), pos: Tensor.from_list(:s32, [1], [w - 1])}).o
    full = Oracle.eval_program(mk.(nil, s), %{q: q, k: k, v: v, pos: Tensor.from_list(:s32, [1], [p])}).o
    check("sliding window: bits differing from attention over the window moved to the front", diff(win, moved), diff(win, full),
          "0, control > 0", diff(win, moved) == 0 and diff(win, full) > 0)
  end

  defp division do
    :rand.seed(:exsss, {6, 6, 6})
    pairs = for _ <- 1..20_000, do: {:rand.uniform(0x1_0000_0000) - 1, :rand.uniform(0x1_0000_0000) - 1}
    normal = fn r -> (r &&& 0x7F80_0000) not in [0, 0x7F80_0000] end
    ref = &ieee/2
    pairs = Enum.filter(pairs, fn {a, b} -> normal.(ref.(a, b)) end)
    div = Canon.compile(:div)
    rcp = Canon.compile(:rcp)
    bad = Enum.count(pairs, fn {a, b} -> div.([a, b]) != ref.(a, b) end)
    old = Enum.count(pairs, fn {a, b} -> F32.mul(a, rcp.([b])) != ref.(a, b) end) / length(pairs)
    check("÷: IEEE mismatches over #{length(pairs)} random normal quotients", bad, Float.round(old, 3), "0, old a·rcp(b) > 0.1",
          bad == 0 and old > 0.1)
  end

  # IEEE quotient of normal operands via binary64 (double rounding is
  # innocuous for division)
  defp ieee(a, b) do
    {aa, ab} = {a &&& 0x7FFF_FFFF, b &&& 0x7FFF_FFFF}

    if aa in 0x0080_0000..0x7F7F_FFFF and ab in 0x0080_0000..0x7F7F_FFFF do
      <<fa::float-32>> = <<aa::32>>
      <<fb::float-32>> = <<ab::32>>
      q = fa / fb
      r = if q >= 3.4028235677973366e38 or q < 1.1754943508222875e-38, do: 0, else: F32.from_float(q)
      (bxor(a, b) &&& 0x8000_0000) ||| r
    else
      0
    end
  end

  defp convolution do
    {h, w, cin, cout} = {6, 7, 5, 4}
    x = Tensor.random(:f32, [cin, h, w], 11)
    wt = Tensor.random(:f32, [cout, cin, 3, 3], 12, scale: 0.3)
    {rows, d} = Spatial.from_nchw(x)
    {lets, y, d2} = Spatial.conv2d([], T.input(:x, :f32, rows.shape), d, "c", wt, nil, padding: 1)
    got = Oracle.eval_program(Program.new([y: y], lets: Enum.reverse(lets)), %{x: rows}).y |> Spatial.to_nchw(d2) |> Tensor.to_floats()
    xv = x |> Tensor.to_floats() |> List.to_tuple()
    wv = wt |> Tensor.to_floats() |> List.to_tuple()

    direct = fn flip ->
      for co <- 0..(cout - 1), oy <- 0..(h - 1), ox <- 0..(w - 1) do
        Enum.sum(for ci <- 0..(cin - 1), ky <- 0..2, kx <- 0..2, iy = oy + ky - 1, ix = ox + kx - 1, iy in 0..(h - 1), ix in 0..(w - 1) do
          {fy, fx} = if flip, do: {2 - ky, 2 - kx}, else: {ky, kx}
          elem(xv, (ci * h + iy) * w + ix) * elem(wv, ((co * cin + ci) * 3 + fy) * 3 + fx)
        end)
      end
    end

    rel = fn ref -> sc = ref |> Enum.map(&abs/1) |> Enum.max(); (Enum.zip_with(got, ref, &abs(&1 - &2)) |> Enum.max()) / sc end
    {e, ctl} = {rel.(direct.(false)), rel.(direct.(true))}
    check("convolution: relative error against a direct binary64 convolution", e, ctl, "≤ 1e-6, control ≥ 0.1", e <= 1.0e-6 and ctl >= 0.1)
  end

  defp tlog do
    probes = Path.join([to_string(:code.priv_dir(:vapor)), "quality", "tlog_probes.json"]) |> File.read!() |> Vapor.JSON.decode!()
    b64 = fn s -> case Base.decode64(s || "") do {:ok, b} -> b; :error -> s end end

    answer = fn verify ->
      Enum.count(probes["inclusion"], fn p ->
        verify.(b64.(p["leafHash"]), p["leafIdx"], p["treeSize"], Enum.map(p["proof"] || [], b64), b64.(p["root"])) == not Map.get(p, "wantErr", false)
      end) +
        Enum.count(probes["consistency"], fn p ->
          Tlog.verify_consistency(p["size1"], p["size2"], Enum.map(p["proof"] || [], b64), b64.(p["root1"]), b64.(p["root2"])) == not Map.get(p, "wantErr", false)
        end)
    end

    total = length(probes["inclusion"]) + length(probes["consistency"])
    good = answer.(&Tlog.verify_inclusion/5)
    # the control: fold the proof without checking that it ends at the root's
    # level — what a naïve verifier does (it accepts truncated proofs)
    naive = answer.(fn leaf, i, n, proof, root ->
      Enum.reduce(Enum.with_index(proof), {leaf, i}, fn {p, _}, {acc, idx} ->
        if rem(idx, 2) == 1 or idx == n - 1, do: {node(p, acc), div(idx, 2)}, else: {node(acc, p), div(idx, 2)}
      end) |> elem(0) |> Kernel.==(root)
    end)

    check("transparency log: probes answered correctly", "#{good}/#{total}", "#{naive}/#{total}", "all; control fewer", good == total and naive < total)
  end

  defp node(l, r), do: :crypto.hash(:sha256, <<1, l::binary, r::binary>>)

  # a random-weight Mamba: the worker's session, whose state is fed back
  # inside the worker after every step, against the oracle's recurrence;
  # the control drops the state (a Mamba without memory)
  defp ssm(nil), do: []

  defp ssm(w) do
    alias Vapor.Lock.Adapters.Mamba
    spec = Mamba.spec(struct(Mamba.Config, vocab: 64, hidden: 32, inner: 64, layers: 2, state: 16, rank: 16, conv: 4, eps: 1.0e-5,
                                            bias: false, conv_bias: true, tie: true, raw: %{"model_type" => "mamba"}))
    ws = for {name, shape, kind} <- Vapor.Lock.expected(spec), into: %{} do
      t = Tensor.random(:f32, shape, :erlang.phash2(name), scale: 0.3)
      {name, if(kind == :norm, do: Tensor.from_list(:f32, shape, Enum.map(Tensor.to_floats(t), &(&1 + 1.0))), else: t)}
    end

    toks = [3, 17, 42, 8, 8, 60, 1, 33]
    rows = fn worker -> {:ok, g} = Vapor.Recurrent.open(spec, ws, worker: worker); elem(Vapor.Recurrent.prefill(g, toks), 2) end
    {native, oracle} = {rows.(w), rows.(nil)}
    {:ok, p} = Vapor.Lock.build(spec, ws, [])
    zero = Mamba.empty_state(spec.config)
    amnesic = for t <- toks, do: Oracle.eval_program(p, Map.put(zero, :tok, Tensor.from_list(:s32, [1], [t]))).logits.data
    check("state-space: native session vs the oracle's recurrence (relative distance of logits)", rel_rows(native, oracle),
          Float.round(rel_rows(amnesic, oracle), 3), "0, control > 0.05", rel_rows(native, oracle) == 0.0 and rel_rows(amnesic, oracle) > 0.05)
  end

  defp rel_rows(as, bs) do
    Enum.zip(as, bs)
    |> Enum.map(fn {a, b} ->
      {fa, fb} = {Vapor.Sampler.floats(a), Vapor.Sampler.floats(b)}
      s = fb |> Enum.map(&abs/1) |> Enum.max()
      (Enum.zip_with(fa, fb, &abs(&1 - &2)) |> Enum.max()) / s
    end)
    |> Enum.max()
  end

  # tree speculation on a planted bigram: lookup drafts in 4 branches must
  # leave the greedy output untouched; accepting the drafts unchecked would not
  defp tree(nil), do: []

  defp tree(w) do
    alias Vapor.Speculative.Tree
    text = String.to_charlist("o gato subiu no muro. o gato comeu o rato. o rato subiu. ")
    pl = Vapor.Quality.Planted.bigram(text, 128)
    {:ok, c} = Config.from_map(pl.config)
    n = 40
    {:ok, plain} = Tree.open(c, pl.weights, w, max_seq: 128, page: 8, branches: 1, max_tokens: 64)
    {want, _} = Tree.generate(plain, text, n, branches: 1, depth: 0)
    {:ok, tr} = Tree.open(c, pl.weights, w, max_seq: 128, page: 8, branches: 4, max_tokens: 64)
    {got, st} = Tree.generate(tr, text, n, branches: 4, depth: 4)
    # the broken verifier: whatever the first lookup branch says, accepted
    blind = Stream.unfold(text, fn ctx -> case Tree.lookup(ctx, 1, 4) do
      [br | _] -> {br, ctx ++ br}
      [] -> {[0], ctx ++ [0]}
    end end) |> Enum.take(n) |> List.flatten() |> Enum.take(n)
    differ = fn a -> Enum.count(Enum.zip(a, want), fn {x, y} -> x != y end) end
    check("tree speculation: tokens differing from plain greedy (#{Float.round(n / max(st.target_steps - 1, 1), 1)} tokens/step)",
          differ.(got), differ.(blind), "0, control > 0", differ.(got) == 0 and differ.(blind) > 0)
  end
end
