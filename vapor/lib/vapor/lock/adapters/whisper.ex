defmodule Vapor.Lock.Adapters.Whisper do
  @moduledoc """
  Tier 3 of the model airlock: **Whisper** (`WhisperForConditionalGeneration`,
  `model_type: "whisper"`: tiny … large-v3, distil-whisper's layout) —
  speech recognition as two programs of the same algebra, no new operator:

    * **encoder** (`part: :encoder`, contract `:encoder`): log-mel frames
      as rows `f32[2S, mels]` (time-major: the transpose of transformers'
      `input_features[mels, 2S]`) → `conv1d(k 3) → gelu → conv1d(k 3,
      stride 2) → gelu` (`Vapor.Spatial`: gather + reshape + GEMV) → `+`
      the stored positional table → pre-LN layers (bidirectional attention:
      every row's horizon is the last row) → LayerNorm → `hidden : f32[S, d]`;
      and, once per utterance, every decoder layer's **cross keys and
      values** `xk{l}, xv{l} : f32[S, d]` — so a decoding step never
      recomputes them.
    * **decoder** (default, contract `:causal_lm`): tokens + learned
      positions → pre-LN layers of causal self-attention (KV cache as
      state, `k{l}`/`v{l}`), **cross-attention** over `xk{l}, xv{l}` (the
      attention operator with every query's horizon at the last encoder
      row, input `xh`), GELU MLP → LayerNorm → logits (the head tied to the
      token embeddings, as transformers writes it).

  The query is scaled by `dh^−½` before the product, as transformers does
  (and the attention's own scale is then 1). The log-mel front end is not
  part of the program (`features` are what transformers' feature extractor
  produces); `Vapor.Modal.Speech` is vapor's own front end.

  `Vapor.Lock.Adapters.Whisper.transcribe/4` runs encoder + greedy decoding;
  the test suite compares hidden states, logits and greedy tokens with
  transformers (`Vapor.WhisperHFTest`).
  """
  @behaviour Vapor.Lock.Adapter
  alias Vapor.{F32, Program, Rejection, Spatial, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Lock.Spec
  alias Vapor.Runtime.Oracle

  defmodule Config do
    @moduledoc "An admitted Whisper configuration."
    defstruct [:vocab, :d, :mels, :src, :tgt, :enc_layers, :dec_layers, :enc_heads, :dec_heads, :enc_ffn, :dec_ffn,
               :eps, :start, :eos, :tie, :raw]
  end

  @impl true
  def id, do: "whisper"

  @impl true
  def claim(%{config: %{"model_type" => "whisper"}}), do: {:claim, 100}
  def claim(_), do: :no

  @impl true
  def owns?(%Config{}), do: true
  def owns?(_), do: false

  @impl true
  def admit(%{config: c}, ws, _opts) do
    d = c["d_model"]

    cond do
      not is_integer(d) or rem(d, 16) != 0 -> no("d_model", "a multiple of 16")
      (c["activation_function"] || "gelu") != "gelu" -> no("activation_function", "gelu (got #{inspect(c["activation_function"])})")
      c["scale_embedding"] == true -> no("scale_embedding", "false (as every released Whisper)")
      Enum.any?(~w(encoder_attention_heads decoder_attention_heads), &(not is_integer(c[&1]) or rem(d, c[&1]) != 0)) ->
        no("encoder/decoder_attention_heads", "divisors of d_model")
      true ->
        cfg = %Config{vocab: c["vocab_size"], d: d, mels: c["num_mel_bins"] || 80, src: c["max_source_positions"] || 1500,
                      tgt: c["max_target_positions"] || 448, enc_layers: c["encoder_layers"], dec_layers: c["decoder_layers"],
                      enc_heads: c["encoder_attention_heads"], dec_heads: c["decoder_attention_heads"],
                      enc_ffn: c["encoder_ffn_dim"], dec_ffn: c["decoder_ffn_dim"], eps: 1.0e-5,
                      start: c["decoder_start_token_id"], eos: c["eos_token_id"],
                      tie: not Map.has_key?(ws, "proj_out.weight") or c["tie_word_embeddings"] != false, raw: c}

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
    %Spec{adapter: __MODULE__, family: "whisper", lineage: ["whisper", "encoder_decoder"], interface: :causal_lm, config: c,
          vocab: c.vocab, width: c.d, max_pos: c.tgt, bos: c.start, eos: List.wrap(c.eos) |> Enum.reject(&is_nil/1),
          features: [:encoder_decoder], parts: %{encoder: :encoder}, in_width: c.mels, rows: 2 * c.src,
          modality: %{in: [:audio], out: [:text]}, digest: Vapor.Canonical.hex_digest({:whisper, c.raw})}
  end

  # ---------------------------------------------------------- tensor names --

  @impl true
  def expected(%Spec{config: c}) do
    d = c.d
    lin = fn n, o, i, bias? -> [{n <> ".weight", [o, i], :matrix}] ++ if(bias?, do: [{n <> ".bias", [o], :bias}], else: []) end
    ln = fn n -> [{n <> ".weight", [d], :norm}, {n <> ".bias", [d], :bias}] end
    attn = fn p -> lin.(p <> ".q_proj", d, d, true) ++ lin.(p <> ".k_proj", d, d, false) ++ lin.(p <> ".v_proj", d, d, true) ++ lin.(p <> ".out_proj", d, d, true) end
    mlp = fn p, f -> lin.(p <> ".fc1", f, d, true) ++ lin.(p <> ".fc2", d, f, true) end

    enc =
      [{"model.encoder.conv1.weight", [d, c.mels, 3], :matrix}, {"model.encoder.conv1.bias", [d], :bias},
       {"model.encoder.conv2.weight", [d, d, 3], :matrix}, {"model.encoder.conv2.bias", [d], :bias},
       {"model.encoder.embed_positions.weight", [c.src, d], :embedding}] ++
        Enum.flat_map(0..(c.enc_layers - 1), fn l ->
          p = "model.encoder.layers.#{l}"
          attn.(p <> ".self_attn") ++ ln.(p <> ".self_attn_layer_norm") ++ mlp.(p, c.enc_ffn) ++ ln.(p <> ".final_layer_norm")
        end) ++ ln.("model.encoder.layer_norm")

    dec =
      [{"model.decoder.embed_tokens.weight", [c.vocab, d], :embedding}, {"model.decoder.embed_positions.weight", [c.tgt, d], :embedding}] ++
        Enum.flat_map(0..(c.dec_layers - 1), fn l ->
          p = "model.decoder.layers.#{l}"
          attn.(p <> ".self_attn") ++ ln.(p <> ".self_attn_layer_norm") ++ attn.(p <> ".encoder_attn") ++
            ln.(p <> ".encoder_attn_layer_norm") ++ mlp.(p, c.dec_ffn) ++ ln.(p <> ".final_layer_norm")
        end) ++ ln.("model.decoder.layer_norm") ++ if(c.tie, do: [], else: [{"proj_out.weight", [c.vocab, d], :matrix}])

    enc ++ dec
  end

  # ----------------------------------------------------------------- build --

  @impl true
  def build(%Spec{config: c}, ws, opts) do
    case Keyword.get(opts, :part, :decoder) do
      :encoder -> encoder(c, ws)
      :decoder -> decoder(c, ws, Keyword.get(opts, :max_seq, c.tgt), Keyword.get(opts, :max_tokens, Keyword.get(opts, :max_seq, c.tgt)))
      other -> {:error, Rejection.new({:whisper, :part}, ":encoder or :decoder (got #{inspect(other)})", "choose a part")}
    end
  end

  defp encoder(c, ws) do
    {d, s} = {c.d, c.src}
    get = fn n -> Tensor.widen(ws[n]) end
    rows = T.input(:rows, :f32, [2 * s, Spatial.pad16(c.mels)])
    conv_w = fn n -> t = get.(n); [o, i, k] = t.shape; Tensor.new(:f32, [o, i, 1, k], t.data) end

    {lets, y, {1, w1, _}} = Spatial.conv2d([], rows, {1, 2 * s, c.mels}, "enc.conv1", conv_w.("model.encoder.conv1.weight"),
                                           get.("model.encoder.conv1.bias"), padding: {0, 1})
    {lets, y} = Spatial.name(lets, "enc.gelu1", T.gelu(y))
    {lets, y, {1, ^s, _}} = Spatial.conv2d(lets, y, {1, w1, d}, "enc.conv2", conv_w.("model.encoder.conv2.weight"),
                                           get.("model.encoder.conv2.bias"), padding: {0, 1}, stride: {1, 2})
    {lets, pos} = Spatial.bind(lets, "model.encoder.embed_positions.weight", get.("model.encoder.embed_positions.weight"))
    {lets, x} = Spatial.name(lets, "enc.x0", T.add(T.gelu(y), pos))
    horizon = T.input(:horizon, :s32, [s])

    {x, lets} =
      Enum.reduce(0..(c.enc_layers - 1), {x, lets}, fn l, {x, lets} ->
        p = "model.encoder.layers.#{l}"

        {x, lets, []} =
          layer(lets, x, p, get, fn lets, h ->
            {lets, q, k, v} = qkv(lets, h, p <> ".self_attn", c.enc_heads, get)
            {lets, out} = out_proj(lets, T.attention(q, k, v, horizon, c.enc_heads, c.enc_heads, 1.0), p <> ".self_attn", get)
            {lets, out, []}
          end, nil)

        {x, lets}
      end)

    {lets, hn} = ln(lets, x, "model.encoder.layer_norm", get)
    {lets, hidden} = Spatial.name(lets, "enc.hidden", hn)

    # every decoder layer's cross keys and values, once per utterance
    {cross, lets} =
      Enum.flat_map_reduce(0..(c.dec_layers - 1), lets, fn l, lets ->
        p = "model.decoder.layers.#{l}.encoder_attn"
        {lets, wk} = Spatial.bind(lets, p <> ".k_proj.weight", get.(p <> ".k_proj.weight"))
        {lets, wv} = Spatial.bind(lets, p <> ".v_proj.weight", get.(p <> ".v_proj.weight"))
        {lets, bv} = Spatial.bind(lets, p <> ".v_proj.bias", row(get.(p <> ".v_proj.bias")))
        {[{:"xk#{l}", T.linear(hidden, wk)}, {:"xv#{l}", T.add(T.linear(hidden, wv), bv)}], lets}
      end)

    {:ok, Program.new([hidden: hidden] ++ cross, lets: Enum.reverse(lets))}
  end

  defp decoder(c, ws, s, tmax) do
    d = c.d
    get = fn n -> Tensor.widen(ws[n]) end
    t = T.dyn(:t, tmax)
    tok = T.input(:tok, :s32, [t])
    pos = T.input(:pos, :s32, [t])
    xh = T.input(:xh, :s32, [t])

    {lets, emb} = Spatial.bind([], "model.decoder.embed_tokens.weight", get.("model.decoder.embed_tokens.weight"))
    {lets, pe} = Spatial.bind(lets, "model.decoder.embed_positions.weight", get.("model.decoder.embed_positions.weight"))
    {lets, x} = Spatial.name(lets, "dec.x0", T.add(T.gather_row(emb, tok), T.gather_row(pe, pos)))

    {x, lets, nexts} =
      Enum.reduce(0..(c.dec_layers - 1), {x, lets, []}, fn l, {x, lets, nexts} ->
        p = "model.decoder.layers.#{l}"

        self_attn = fn lets, h ->
          {lets, q, k, v} = qkv(lets, h, p <> ".self_attn", c.dec_heads, get)
          kn = T.kv_write(T.input(:"k#{l}", :f32, [s, d]), pos, k)
          vn = T.kv_write(T.input(:"v#{l}", :f32, [s, d]), pos, v)
          {lets, kn} = Spatial.name(lets, "#{p}.k_cache", kn)
          {lets, vn} = Spatial.name(lets, "#{p}.v_cache", vn)
          {lets, out} = out_proj(lets, T.attention(q, kn, vn, pos, c.dec_heads, c.dec_heads, 1.0), p <> ".self_attn", get)
          {lets, out, [{:"k#{l}_next", kn}, {:"v#{l}_next", vn}]}
        end

        # cross-attention: the encoder's keys and values, every horizon at its last row
        cross = fn lets, h ->
          {lets, q} = query(lets, h, p <> ".encoder_attn", c.dec_heads, get)
          att = T.attention(q, T.input(:"xk#{l}", :f32, [c.src, d]), T.input(:"xv#{l}", :f32, [c.src, d]), xh, c.dec_heads, c.dec_heads, 1.0)
          out_proj(lets, att, p <> ".encoder_attn", get)
        end

        {x, lets, n} = layer(lets, x, p, get, self_attn, cross)
        {x, lets, nexts ++ n}
      end)

    {lets, hn} = ln(lets, x, "model.decoder.layer_norm", get)
    {lets, hidden} = Spatial.name(lets, "dec.hidden", hn)
    {lets, head} = if c.tie, do: {lets, emb}, else: Spatial.bind(lets, "proj_out.weight", get.("proj_out.weight"))
    state = for l <- 0..(c.dec_layers - 1), kv <- ["k", "v"], do: {:"#{kv}#{l}", :"#{kv}#{l}_next"}
    {:ok, Program.new([logits: T.linear(hidden, head)] ++ nexts, state: state, lets: Enum.reverse(lets))}
  end

  # x + attn(LN₁ x); x + cross(LN₂ x) when present; x + MLP(LN₃ x)
  defp layer(lets, x, p, get, attn, cross) do
    {lets, h} = ln(lets, x, p <> ".self_attn_layer_norm", get)
    {lets, h} = Spatial.name(lets, "#{p}.h1", h)
    {lets, a, nexts} = attn.(lets, h)
    {lets, x} = Spatial.name(lets, "#{p}.x1", T.add(x, a))

    {lets, x} =
      if cross do
        {lets, h} = ln(lets, x, p <> ".encoder_attn_layer_norm", get)
        {lets, h} = Spatial.name(lets, "#{p}.h2", h)
        {lets, a} = cross.(lets, h)
        Spatial.name(lets, "#{p}.x2", T.add(x, a))
      else
        {lets, x}
      end

    {lets, h} = ln(lets, x, p <> ".final_layer_norm", get)
    {lets, h} = Spatial.name(lets, "#{p}.h3", h)
    {lets, w1} = Spatial.bind(lets, p <> ".fc1.weight", get.(p <> ".fc1.weight"))
    {lets, b1} = Spatial.bind(lets, p <> ".fc1.bias", row(get.(p <> ".fc1.bias")))
    {lets, w2} = Spatial.bind(lets, p <> ".fc2.weight", get.(p <> ".fc2.weight"))
    {lets, b2} = Spatial.bind(lets, p <> ".fc2.bias", row(get.(p <> ".fc2.bias")))
    f = T.add(T.linear(T.gelu(T.add(T.linear(h, w1), b1)), w2), b2)
    {lets, x} = Spatial.name(lets, "#{p}.x3", T.add(x, f))
    {x, lets, nexts}
  end

  # (h·Wqᵀ + b)·dh^−½ — transformers scales the query before the product,
  # and the attention's own scale is then 1
  defp query(lets, h, p, heads, get) do
    wq = get.(p <> ".q_proj.weight")
    [d, _] = wq.shape
    {lets, wq} = Spatial.bind(lets, p <> ".q_proj.weight", wq)
    {lets, bq} = Spatial.bind(lets, p <> ".q_proj.bias", row(get.(p <> ".q_proj.bias")))
    Spatial.name(lets, "#{p}.q", T.mul(T.add(T.linear(h, wq), bq), T.splat(:math.pow(div(d, heads), -0.5))))
  end

  defp qkv(lets, h, p, heads, get) do
    {lets, q} = query(lets, h, p, heads, get)
    {lets, wk} = Spatial.bind(lets, p <> ".k_proj.weight", get.(p <> ".k_proj.weight"))
    {lets, wv} = Spatial.bind(lets, p <> ".v_proj.weight", get.(p <> ".v_proj.weight"))
    {lets, bv} = Spatial.bind(lets, p <> ".v_proj.bias", row(get.(p <> ".v_proj.bias")))
    {lets, k} = Spatial.name(lets, "#{p}.k", T.linear(h, wk))
    {lets, v} = Spatial.name(lets, "#{p}.v", T.add(T.linear(h, wv), bv))
    {lets, q, k, v}
  end

  defp out_proj(lets, a, p, get) do
    {lets, wo} = Spatial.bind(lets, p <> ".out_proj.weight", get.(p <> ".out_proj.weight"))
    {lets, bo} = Spatial.bind(lets, p <> ".out_proj.bias", row(get.(p <> ".out_proj.bias")))
    {lets, T.add(T.linear(a, wo), bo)}
  end

  # LayerNorm: (x − μ)·rsqrt(σ² + ε)·w + b, μ and σ² by canonical reductions
  defp ln(lets, x, p, get) do
    {lets, w} = Spatial.bind(lets, p <> ".weight", row(get.(p <> ".weight")))
    {lets, b} = Spatial.bind(lets, p <> ".bias", row(get.(p <> ".bias")))
    {:ok, {:f32, shape}} = T.infer(x)
    inv = T.splat(1.0 / List.last(shape))
    mean = T.mul(T.reduce(:sum, x), inv)
    xc = T.sub(x, mean)
    var = T.mul(T.reduce(:sum, T.mul(xc, xc)), inv)
    {lets, T.add(T.mul(w, T.mul(xc, T.rsqrt(T.add(var, T.splat(1.0e-5))))), b)}
  end

  # -------------------------------------------------------------- driving --

  @doc """
  The encoder's inputs for transformers' `input_features : f32[mels, 2S]`:
  `%{rows: f32[2S, pad16(mels)], horizon}` (time-major rows; every row
  sees every row).
  """
  def encoder_input(%Spec{config: c}, %Tensor{shape: [mels, frames]} = f) when mels == c.mels and frames == 2 * c.src do
    v = f |> Tensor.widen() |> Tensor.to_list() |> List.to_tuple()
    mp = Spatial.pad16(mels)
    pad = List.duplicate(0, mp - mels)
    data = for tt <- 0..(frames - 1), into: <<>>, do: F32.encode(for(m <- 0..(mels - 1), do: elem(v, m * frames + tt)) ++ pad)
    %{rows: Tensor.new(:f32, [frames, mp], data), horizon: Tensor.from_list(:s32, [c.src], List.duplicate(c.src - 1, c.src))}
  end

  @doc """
  Encode `features` and decode greedily from `prompt` (default: the start
  token) for at most `n` tokens, stopping at EOS. `run` evaluates a program
  on an environment (default: the oracle; `Vapor.Runtime.Native.run/4`
  through a closure for a worker). Returns `{:ok, ids, encoder_outputs}`.
  """
  def transcribe(%Spec{config: c} = spec, ws, features, opts \\ []) do
    run = Keyword.get(opts, :run, fn p, env -> Oracle.eval_program(p, env) end)
    n = Keyword.get(opts, :max_tokens, c.tgt - 1)
    prompt = Keyword.get(opts, :prompt, [c.start])

    with {:ok, ep} <- Vapor.Lock.build(spec, ws, part: :encoder),
         {:ok, dp} <- Vapor.Lock.build(spec, ws, max_seq: c.tgt) do
      enc = run.(ep, encoder_input(spec, features))
      cross = for l <- 0..(c.dec_layers - 1), kv <- ["xk", "xv"], into: %{}, do: {:"#{kv}#{l}", enc[:"#{kv}#{l}"]}
      caches = for {i, _} <- dp.state, into: %{}, do: {i, Tensor.new(:f32, [c.tgt, c.d], :binary.copy(<<0::32>>, c.tgt * c.d))}

      step = fn caches, toks, p0 ->
        k = length(toks)
        env = Map.merge(Map.merge(cross, caches), %{tok: Tensor.from_list(:s32, [k], toks), pos: Tensor.from_list(:s32, [k], Enum.to_list(p0..(p0 + k - 1))),
                                                   xh: Tensor.from_list(:s32, [k], List.duplicate(c.src - 1, k))})
        out = run.(dp, env)
        last = out.logits.data |> binary_part((k - 1) * c.vocab * 4, c.vocab * 4)
        {Vapor.Sampler.argmax(last), Map.new(dp.state, fn {i, o} -> {i, out[o]} end)}
      end

      {first, caches} = step.(caches, prompt, 0)
      eos = List.wrap(c.eos)

      {ids, _} =
        Enum.reduce_while(1..n//1, {[first], caches}, fn i, {acc, caches} ->
          [last | _] = acc
          if last in eos or length(prompt) + i > c.tgt - 1, do: {:halt, {acc, caches}}, else: (
            {nx, caches} = step.(caches, [last], length(prompt) + i - 1)
            {:cont, {[nx | acc], caches}})
        end)

      {:ok, Enum.reverse(ids) |> Enum.take(n), enc}
    end
  end

  defp row(%Tensor{shape: [n]} = t), do: Tensor.new(:f32, [1, n], t.data)
end
