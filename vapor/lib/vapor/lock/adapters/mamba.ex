defmodule Vapor.Lock.Adapters.Mamba do
  @moduledoc """
  Tier 3 of the model airlock: **Mamba** (selective state-space models,
  Gu & Dao 2023) — Hugging Face's `MambaForCausalLM` (`model_type:
  "mamba"`: state-spaces/mamba-130m … 2.8b, Falcon-Mamba's layout) — as a
  causal language model whose memory is a fixed-size **state**, not a KV
  cache: a step costs the same at token 10 and at token 10⁶.

  The program is one recurrent step (`t = 1`); per layer, in transformers'
  order of operations:

      h   = RMSNorm(x)
      xs  = h·W_inᵀ[0:di]        z = h·W_inᵀ[di:2di]          (+ biases)
      u   = silu(conv1d(…, x_{t−k+2}, xs) + b)   depthwise, causal: the last
                                    k − 1 inputs are state (`conv{l}_{j}`)
      Δ   = softplus(u·W_xᵀ[0:R] · W_Δᵀ + b_Δ)   B = u·W_xᵀ[R:R+N]   C = …
      s   = s·exp(Δᵀ ⊙ A) + (Δᵀ ⊙ B) ⊙ uᵀ        s : f32[di, N] (`ssm{l}`)
      y   = (s·C + u ⊙ D) ⊙ silu(z)
      x  += y·W_outᵀ (+ bias)

  then the final RMSNorm and the head (tied to the embeddings unless
  `lm_head.weight` is present). `A = −exp(A_log)` is formed at build time
  with the correctly rounded exponential (`Vapor.CR`), so it does not
  depend on the host's libm. Extents the 16-lane contraction cannot take
  (Δ rank `R`, state size `N`) are zero-padded: padded state columns stay
  exactly zero (`exp(0·Δ) = 1` times a zero state plus a zero input).

  A prompt is consumed one token per step — the recurrence *is* the
  definition, so prefill and decoding are the same instructions and give
  the same bits. The state stays in the session between steps (the worker
  feeds `s ← s_next` back itself; `Vapor.Runtime.Session`).
  `Vapor.Recurrent` drives generation. Not served by `Vapor.Engine`, whose
  memory model is the paged KV cache (the spec has no `:paged` feature,
  so the engine refuses it by contract).

  Configurations refused by name: `hidden_act` other than `silu`,
  `time_step_rank` other than an integer (`"auto"` is resolved by
  transformers when it writes the config). Mamba-2 (`model_type: mamba2`)
  is `Vapor.Lock.Adapters.Mamba2`.
  """
  @behaviour Vapor.Lock.Adapter
  alias Vapor.{F32, Program, Rejection, Spatial, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Lock.Spec

  defmodule Config do
    @moduledoc "An admitted Mamba configuration."
    defstruct [:vocab, :hidden, :inner, :layers, :state, :rank, :conv, :eps, :bias, :conv_bias, :tie, :bos, :eos, :raw]
  end

  @impl true
  def id, do: "mamba"

  @impl true
  def claim(%{config: %{"model_type" => "mamba"}}), do: {:claim, 100}
  def claim(%{config: %{"model_type" => "falcon_mamba"}}),
    do: {:near, "Falcon-Mamba normalises B, C and Δ (RMS without weights) inside the mixer: not yet built"}
  def claim(%{config: %{"model_type" => t}}) when t in ["jamba", "zamba", "zamba2", "bamba", "nemotron_h", "falcon_h1"],
    do: {:near, "#{t} is a hybrid of state-space and attention layers (a KV cache and a state per sequence): not yet built"}
  def claim(_), do: :no

  @impl true
  def owns?(%Config{}), do: true
  def owns?(_), do: false

  @impl true
  def admit(%{config: c}, ws, _opts) do
    d = c["hidden_size"]
    inner = c["intermediate_size"] || (d && (c["expand"] || 2) * d)

    cond do
      not is_integer(d) or not is_integer(c["vocab_size"]) -> no("hidden_size/vocab_size", "integers")
      (c["hidden_act"] || "silu") != "silu" -> no("hidden_act", "silu (got #{inspect(c["hidden_act"])})")
      not is_integer(c["time_step_rank"]) -> no("time_step_rank", "an integer (got #{inspect(c["time_step_rank"])})")
      rem(d, 16) != 0 or rem(inner, 16) != 0 -> no("hidden_size/intermediate_size", "multiples of 16")
      true ->
        cfg = %Config{vocab: c["vocab_size"], hidden: d, inner: inner, layers: c["num_hidden_layers"], state: c["state_size"] || 16,
                      rank: c["time_step_rank"], conv: c["conv_kernel"] || 4, eps: (c["layer_norm_epsilon"] || 1.0e-5) * 1.0,
                      bias: c["use_bias"] == true, conv_bias: c["use_conv_bias"] != false,
                      tie: not Map.has_key?(ws, "lm_head.weight") and c["tie_word_embeddings"] != false,
                      bos: c["bos_token_id"], eos: c["eos_token_id"], raw: c}

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
    %Spec{adapter: __MODULE__, family: "mamba", lineage: ["mamba", "ssm"], interface: :causal_lm, config: c,
          vocab: c.vocab, width: c.hidden, max_pos: :unbounded, bos: c.bos, eos: c.eos |> List.wrap() |> Enum.reject(&is_nil/1),
          features: [:recurrent, :hidden], modality: %{in: [:text], out: [:text]},
          digest: Vapor.Canonical.hex_digest({:mamba, c.raw})}
  end

  @impl true
  def expected(%Spec{config: c}) do
    {d, di, n, r, k} = {c.hidden, c.inner, c.state, c.rank, c.conv}

    layer =
      for l <- 0..(c.layers - 1), p = "backbone.layers.#{l}." do
        m = p <> "mixer."

        [{p <> "norm.weight", [d], :norm}, {m <> "A_log", [di, n], :vector}, {m <> "D", [di], :vector},
         {m <> "conv1d.weight", [di, 1, k], :matrix}, {m <> "in_proj.weight", [2 * di, d], :matrix},
         {m <> "x_proj.weight", [r + 2 * n, di], :matrix}, {m <> "dt_proj.weight", [di, r], :matrix},
         {m <> "dt_proj.bias", [di], :bias}, {m <> "out_proj.weight", [d, di], :matrix}] ++
          if(c.conv_bias, do: [{m <> "conv1d.bias", [di], :bias}], else: []) ++
          if(c.bias, do: [{m <> "in_proj.bias", [2 * di], :bias}, {m <> "out_proj.bias", [d], :bias}], else: [])
      end

    [{"backbone.embeddings.weight", [c.vocab, d], :embedding}] ++ List.flatten(layer) ++
      [{"backbone.norm_f.weight", [d], :norm}] ++ if(c.tie, do: [], else: [{"lm_head.weight", [c.vocab, d], :matrix}])
  end

  @impl true
  def taps(%Spec{config: c}) do
    for l <- 0..(c.layers - 1), do: {:"ssm#{l}_u", ["backbone.layers.#{l}.mixer.x_proj.weight"]}
  end

  # ----------------------------------------------------------------- state --

  @doc "The recurrent state's names: `[ssm0, conv0_0, …]`, each with its sort."
  def state(%Config{} = c) do
    np = Spatial.pad16(c.state)

    for l <- 0..(c.layers - 1) do
      [{:"ssm#{l}", [c.inner, np]} | for(j <- 0..(c.conv - 2), do: {:"conv#{l}_#{j}", [1, c.inner]})]
    end
    |> List.flatten()
  end

  @doc "A zero state (the start of every sequence)."
  def empty_state(%Config{} = c),
    do: Map.new(state(c), fn {n, shape} -> {n, Tensor.new(:f32, shape, :binary.copy(<<0::32>>, Enum.product(shape)))} end)

  # ----------------------------------------------------------------- build --

  @impl true
  def build(%Spec{config: c}, ws, _opts) do
    {_d, di, n, r, k} = {c.hidden, c.inner, c.state, c.rank, c.conv}
    {np, rp} = {Spatial.pad16(n), Spatial.pad16(r)}
    get = fn name -> Tensor.widen(ws[name]) end
    rows = fn t, from, len -> rows(get.(t), from, len) end

    {lets, embed} = Spatial.bind([], "backbone.embeddings.weight", get.("backbone.embeddings.weight"))
    x = T.gather_row(embed, T.input(:tok, :s32, [1]))

    {x, lets, nexts} =
      Enum.reduce(0..(c.layers - 1), {x, lets, []}, fn l, {x, lets, nexts} ->
        p = "backbone.layers.#{l}."
        m = p <> "mixer."
        b = fn lets, name, t -> Spatial.bind(lets, "#{p}#{name}", t) end

        {lets, nw} = b.(lets, "norm", row(get.(p <> "norm.weight")))
        {lets, h} = Spatial.name(lets, "#{p}h", norm(x, nw, c.eps))

        bias = fn lets, name, from, len ->
          if c.bias, do: b.(lets, name, row(slice(get.(m <> "in_proj.bias"), from, len))), else: {lets, nil}
        end

        {lets, w_x} = b.(lets, "in_x", rows.(m <> "in_proj.weight", 0, di))
        {lets, w_z} = b.(lets, "in_z", rows.(m <> "in_proj.weight", di, di))
        {lets, b_x} = bias.(lets, "in_x_bias", 0, di)
        {lets, b_z} = bias.(lets, "in_z_bias", di, di)
        {lets, xs} = Spatial.name(lets, "#{p}xs", add_bias(T.linear(h, w_x), b_x))
        {lets, z} = Spatial.name(lets, "#{p}z", add_bias(T.linear(h, w_z), b_z))

        # depthwise causal convolution: Σ_j w[:, j]·input(t − k + 1 + j), oldest first
        cw = m <> "conv1d.weight" |> get.() |> Map.fetch!(:data) |> F32.decode() |> List.to_tuple()
        cols = for j <- 0..(k - 1), do: Tensor.new(:f32, [1, di], F32.encode(for(i <- 0..(di - 1), do: elem(cw, i * k + j))))
        {lets, cols} = Enum.reduce(Enum.with_index(cols), {lets, []}, fn {t, j}, {lets, acc} -> {lets, ref} = b.(lets, "conv_w#{j}", t); {lets, acc ++ [ref]} end)
        past = for j <- 0..(k - 2), do: T.input(:"conv#{l}_#{j}", :f32, [1, di])
        inputs = past ++ [xs]
        conv = Enum.zip(inputs, cols) |> Enum.map(fn {a, w} -> T.mul(a, w) end) |> Enum.reduce(fn t, acc -> T.add(acc, t) end)
        {lets, conv} = if c.conv_bias, do: (fn {lets, cb} -> {lets, T.add(conv, cb)} end).(b.(lets, "conv_b", row(get.(m <> "conv1d.bias")))), else: {lets, conv}
        {lets, u} = Spatial.name(lets, "ssm#{l}_u", T.silu(conv))

        # Δ, B, C (padded extents are zero rows: zero outputs)
        xw = get.(m <> "x_proj.weight")
        {lets, w_dt} = b.(lets, "x_dt", pad_rows(rows(xw, 0, r), rp))
        {lets, w_b} = b.(lets, "x_b", pad_rows(rows(xw, r, n), np))
        {lets, w_c} = b.(lets, "x_c", pad_rows(rows(xw, r + n, n), np))
        {lets, w_d} = b.(lets, "dt_proj", pad_cols(get.(m <> "dt_proj.weight"), rp))
        {lets, b_d} = b.(lets, "dt_bias", row(get.(m <> "dt_proj.bias")))
        {lets, dtl} = Spatial.name(lets, "#{p}dt_low", T.linear(u, w_dt))
        {lets, dt} = Spatial.name(lets, "#{p}dt", T.softplus(T.add(T.linear(dtl, w_d), b_d)))
        {lets, bb} = Spatial.name(lets, "#{p}B", T.linear(u, w_b))
        {lets, cc} = Spatial.name(lets, "#{p}C", T.linear(u, w_c))

        # the selective scan, one step
        {lets, a} = b.(lets, "A", pad_cols(a_matrix(get.(m <> "A_log")), np))
        {lets, dtc} = Spatial.name(lets, "#{p}dt_col", T.transpose(dt))
        {lets, uc} = Spatial.name(lets, "#{p}u_col", T.transpose(u))
        da = T.exp(T.mul(dtc, a))
        dbx = T.mul(T.mul(dtc, bb), uc)
        s_in = T.input(:"ssm#{l}", :f32, [di, np])
        {lets, s} = Spatial.name(lets, "ssm#{l}_next_v", T.add(T.mul(s_in, da), dbx))
        ys = T.transpose(T.reduce(:sum, T.mul(s, cc)))
        {lets, dd} = b.(lets, "D", row(get.(m <> "D")))
        y = T.mul(T.add(ys, T.mul(u, dd)), T.silu(z))

        {lets, w_o} = b.(lets, "out", get.(m <> "out_proj.weight"))
        {lets, b_o} = if c.bias, do: b.(lets, "out_bias", row(get.(m <> "out_proj.bias"))), else: {lets, nil}
        {lets, x} = Spatial.name(lets, "#{p}x", T.add(x, add_bias(T.linear(y, w_o), b_o)))

        # state outputs: the scan state, and the convolution window shifted by one
        conv_next = for j <- 0..(k - 2), do: {:"conv#{l}_#{j}_next", Enum.at(inputs, j + 1)}
        {x, lets, nexts ++ [{:"ssm#{l}_next", s} | conv_next]}
      end)

    {lets, nf} = Spatial.bind(lets, "backbone.norm_f", row(get.("backbone.norm_f.weight")))
    {lets, hid} = Spatial.name(lets, "hidden_v", norm(x, nf, c.eps))
    {lets, head} = if c.tie, do: {lets, embed}, else: Spatial.bind(lets, "lm_head", get.("lm_head.weight"))

    outputs = [logits: T.linear(hid, head), hidden: hid] ++ nexts
    state = for {name, _} <- state(c), do: {name, :"#{name}_next"}
    {:ok, Program.new(outputs, state: state, lets: Enum.reverse(lets))}
  end

  # x · rsqrt(Σx²·(1/d) + ε), then the weight (as `Vapor.Model.Llama`)
  defp norm(x, w, eps) do
    {:ok, {:f32, shape}} = T.infer(x)
    ms = T.mul(T.reduce(:sum, T.mul(x, x)), T.splat(1.0 / List.last(shape)))
    T.mul(w, T.mul(x, T.rsqrt(T.add(ms, T.splat(eps)))))
  end

  defp add_bias(t, nil), do: t
  defp add_bias(t, b), do: T.add(t, b)

  # A = −exp(A_log), correctly rounded (binary64 CR exp, then binary32)
  defp a_matrix(%Tensor{shape: shape} = t) do
    Tensor.from_list(:f32, shape, Enum.map(Tensor.to_floats(t), &(-F32.to_float(F32.from_float(Vapor.CR.exp_f64(&1))))))
  end

  defp row(%Tensor{shape: [n]} = t), do: Tensor.new(:f32, [1, n], t.data)

  defp slice(%Tensor{shape: [_]} = t, from, len), do: Tensor.new(:f32, [len], binary_part(t.data, from * 4, len * 4))

  defp rows(%Tensor{shape: [_, k]} = t, from, len), do: Tensor.new(:f32, [len, k], binary_part(t.data, from * k * 4, len * k * 4))

  defp pad_rows(%Tensor{shape: [n, k]} = t, n), do: Tensor.new(:f32, [n, k], t.data)
  defp pad_rows(%Tensor{shape: [n, k]} = t, np), do: Tensor.new(:f32, [np, k], t.data <> :binary.copy(<<0::32>>, (np - n) * k))

  defp pad_cols(%Tensor{shape: [_, k]} = t, k), do: t

  defp pad_cols(%Tensor{shape: [n, k]} = t, kp) do
    z = :binary.copy(<<0::32>>, kp - k)
    Tensor.new(:f32, [n, kp], for(i <- 0..(n - 1), into: <<>>, do: binary_part(t.data, i * k * 4, k * 4) <> z))
  end
end
