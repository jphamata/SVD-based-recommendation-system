defmodule Vapor.Lock.Adapters.Mamba2 do
  @moduledoc """
  Tier 3 of the model airlock: **Mamba-2** (state-space duality, Dao & Gu
  2024) — Hugging Face's `Mamba2ForCausalLM` (`model_type: "mamba2"`:
  state-spaces/mamba2-130m … 2.7b, Mamba-Codestral's layout) — as a
  causal language model whose memory is a fixed-size state.

  Mamba-2 differs from Mamba in three places, and the step follows each:

    * **multi-head, scalar decay** — `A` is one number per head, `Δ` one
      per head (`dt = softplus(h·W_dt + dt_bias)`, clamped to
      `time_step_limit` when it is not `(0, ∞)`), the state is
      `f32[heads·head_dim, N]`;
    * **B and C shared by groups of heads** (`n_groups`), produced by the
      same input projection and passed through the same causal
      convolution as `x`;
    * **a gated RMSNorm** before the output projection:
      `RMSNorm(y ⊙ silu(z))`.

  Per layer, one token:

      h        = RMSNorm(x)
      z | x | B | C | dt = h·W_inᵀ (+ b)                split by rows
      x, B, C  = silu(conv1d(…, ·) + b)                 depthwise, causal, k−1 past inputs as state
      dt       = clamp(softplus(dt + dt_bias))          f32[1, heads]
      s        = s ⊙ exp(dt·A) + (dt·B) ⊙ x             s : f32[heads·head_dim, N]   (`ssm{l}`)
      y        = s·C + x ⊙ D
      x       += RMSNormGated(y, z)·W_outᵀ (+ b)

  Head-to-channel expansion is exact: `dt` reaches its head's channels
  through a product with a one-hot matrix (`dt·1 + Σ 0` — one non-zero
  term, no rounding), `A` and `D` are expanded at build time, and a
  group's `B`/`C` row is fetched per channel with `gather_row` (a copy).
  `A = −exp(A_log)` uses the correctly rounded exponential (`Vapor.CR`).
  State size `N` and the head count are zero-padded to the 16-lane
  contraction; padded state columns stay exactly zero.

  **The gated norm, and a disagreement found on the way.** The training
  code (`mamba_ssm`'s `RMSNormGated(group_size = d_inner / n_groups)`)
  normalises each group separately; transformers' `MambaRMSNormGated`
  normalises the whole width, whatever `n_groups` is. They agree only for
  `n_groups = 1`. The default here is the training code's (`gated_norm:
  :group`); `gated_norm: :whole` reproduces transformers (both tested
  against their reference, `test/vapor/mamba2_hf_test.exs`). A second one:
  transformers' cached decoding step skips the `time_step_limit` clamp
  that its chunked scan applies; the step here always clamps (the trained
  semantics), so it matches transformers' full forward pass, not its
  cached `generate()` when the limit bites.

  Prefill and decoding are the same instructions (one token per step;
  `Vapor.Recurrent`), so they give the same bits. Not served by
  `Vapor.Engine` (no `:paged` feature).

  Refused by name: `hidden_act` other than `silu`, `hidden_size·expand ≠
  num_heads·head_dim`, `num_heads` not a multiple of `n_groups`, widths
  not multiples of 16.
  """
  @behaviour Vapor.Lock.Adapter
  alias Vapor.{F32, Program, Rejection, Spatial, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Lock.Spec

  defmodule Config do
    @moduledoc "An admitted Mamba-2 configuration."
    defstruct [:vocab, :hidden, :inner, :heads, :head_dim, :groups, :layers, :state, :conv, :eps, :bias, :conv_bias,
               :tie, :limit, :gated_norm, :bos, :eos, :raw]
  end

  @impl true
  def id, do: "mamba2"

  @impl true
  def claim(%{config: %{"model_type" => "mamba2"}}), do: {:claim, 100}
  def claim(_), do: :no

  @impl true
  def owns?(%Config{}), do: true
  def owns?(_), do: false

  @impl true
  def admit(%{config: c}, ws, opts) do
    d = c["hidden_size"]
    {nh, hd, ng} = {c["num_heads"], c["head_dim"], c["n_groups"] || 1}
    di = is_integer(d) && (c["expand"] || 2) * d
    norm = Keyword.get(opts, :gated_norm) || gated_norm(c["vapor_gated_norm"])

    cond do
      not is_integer(d) or not is_integer(c["vocab_size"]) -> no("hidden_size/vocab_size", "integers")
      not (is_integer(nh) and is_integer(hd) and is_integer(ng) and nh > 0 and hd > 0 and ng > 0) ->
        no("num_heads/head_dim/n_groups", "positive integers")
      (c["hidden_act"] || "silu") != "silu" -> no("hidden_act", "silu (got #{inspect(c["hidden_act"])})")
      di != nh * hd -> no("expand", "hidden_size·expand = num_heads·head_dim (#{di} ≠ #{nh * hd})")
      rem(nh, ng) != 0 -> no("n_groups", "a divisor of num_heads (#{nh})")
      rem(d, 16) != 0 or rem(di, 16) != 0 -> no("hidden_size/expand", "hidden and inner widths multiples of 16")
      norm not in [:group, :whole] -> no("gated_norm", ":group (mamba_ssm) or :whole (transformers)")
      true ->
        with {:ok, limit} <- limit(c["time_step_limit"]) do
          cfg = %Config{vocab: c["vocab_size"], hidden: d, inner: di, heads: nh, head_dim: hd, groups: ng, layers: c["num_hidden_layers"],
                        state: c["state_size"] || 128, conv: c["conv_kernel"] || 4, eps: (c["layer_norm_epsilon"] || 1.0e-5) * 1.0,
                        bias: c["use_bias"] == true, conv_bias: c["use_conv_bias"] != false,
                        tie: not Map.has_key?(ws, "lm_head.weight") and c["tie_word_embeddings"] == true,
                        limit: limit, gated_norm: if(ng == 1, do: :group, else: norm), bos: c["bos_token_id"], eos: c["eos_token_id"], raw: c}

          spec = spec(cfg)

          case Enum.find(expected(spec), fn {n, shape, _} -> not match?(%Tensor{shape: ^shape}, ws[n]) end) do
            nil -> {:ok, spec, ws}
            {n, shape, _} -> {:error, Rejection.new({:weight, n}, "#{inspect(shape, charlists: :as_lists)}", "check the checkpoint against config.json")}
          end
        end
    end
  end

  defp gated_norm(nil), do: :group
  defp gated_norm("group"), do: :group
  defp gated_norm("whole"), do: :whole
  defp gated_norm(other), do: other

  # time_step_limit: [lo, hi], hi possibly ∞ (`Infinity`, or transformers 5's {"__float__": "Infinity"})
  defp limit(nil), do: {:ok, {0.0, :infinity}}

  defp limit([lo, hi]) do
    case {bound(lo), bound(hi)} do
      {l, h} when is_float(l) and (is_float(h) or h == :infinity) and l >= 0 -> {:ok, {l, h}}
      _ -> no("time_step_limit", "[lo ≥ 0, hi] (got #{inspect([lo, hi])})")
    end
  end

  defp limit(other), do: no("time_step_limit", "[lo, hi] (got #{inspect(other)})")

  defp bound(x) when is_number(x), do: x * 1.0
  defp bound(:infinity), do: :infinity
  defp bound(%{"__float__" => "Infinity"}), do: :infinity
  defp bound(_), do: :bad

  defp no(f, b), do: {:error, Rejection.new({:config, f}, b, "use a supported checkpoint")}

  @impl true
  def spec(%Config{} = c) do
    %Spec{adapter: __MODULE__, family: "mamba2", lineage: ["mamba2", "ssm"], interface: :causal_lm, config: c,
          vocab: c.vocab, width: c.hidden, max_pos: :unbounded, bos: c.bos, eos: c.eos |> List.wrap() |> Enum.reject(&is_nil/1),
          features: [:recurrent, :hidden], modality: %{in: [:text], out: [:text]},
          digest: Vapor.Canonical.hex_digest({:mamba2, c.raw, c.gated_norm})}
  end

  defp conv_dim(c), do: c.inner + 2 * c.groups * c.state
  defp proj(c), do: c.inner + conv_dim(c) + c.heads

  @impl true
  def expected(%Spec{config: c}) do
    {d, di, nh, k} = {c.hidden, c.inner, c.heads, c.conv}

    layer =
      for l <- 0..(c.layers - 1), p = "backbone.layers.#{l}." do
        m = p <> "mixer."

        [{p <> "norm.weight", [d], :norm}, {m <> "A_log", [nh], :vector}, {m <> "D", [nh], :vector}, {m <> "dt_bias", [nh], :bias},
         {m <> "conv1d.weight", [conv_dim(c), 1, k], :matrix}, {m <> "in_proj.weight", [proj(c), d], :matrix},
         {m <> "norm.weight", [di], :norm}, {m <> "out_proj.weight", [d, di], :matrix}] ++
          if(c.conv_bias, do: [{m <> "conv1d.bias", [conv_dim(c)], :bias}], else: []) ++
          if(c.bias, do: [{m <> "in_proj.bias", [proj(c)], :bias}, {m <> "out_proj.bias", [d], :bias}], else: [])
      end

    [{"backbone.embeddings.weight", [c.vocab, d], :embedding}] ++ List.flatten(layer) ++
      [{"backbone.norm_f.weight", [d], :norm}] ++ if(c.tie, do: [], else: [{"lm_head.weight", [c.vocab, d], :matrix}])
  end

  @impl true
  def taps(%Spec{config: c}) do
    for l <- 0..(c.layers - 1), do: {:"backbone.layers.#{l}.h", ["backbone.layers.#{l}.mixer.in_proj.weight"]}
  end

  # ----------------------------------------------------------------- state --

  @doc "The recurrent state's names: `ssm{l}` and the convolution windows of x, B and C, each with its sort."
  def state(%Config{} = c) do
    np = Spatial.pad16(c.state)
    widths = [x: c.inner, b: c.groups * np, c: c.groups * np]

    for l <- 0..(c.layers - 1) do
      [{:"ssm#{l}", [c.inner, np]} | for({part, w} <- widths, j <- 0..(c.conv - 2), do: {:"conv#{l}_#{part}_#{j}", [1, w]})]
    end
    |> List.flatten()
  end

  @doc "A zero state (the start of every sequence)."
  def empty_state(%Config{} = c),
    do: Map.new(state(c), fn {n, shape} -> {n, Tensor.new(:f32, shape, :binary.copy(<<0::32>>, Enum.product(shape)))} end)

  # ----------------------------------------------------------------- build --

  @impl true
  def build(%Spec{config: c}, ws, _opts) do
    {di, nh, hd, ng, n, k} = {c.inner, c.heads, c.head_dim, c.groups, c.state, c.conv}
    {np, nhp} = {Spatial.pad16(n), Spatial.pad16(nh)}
    get = fn name -> Tensor.widen(ws[name]) end
    head = fn r -> div(r, hd) end
    group = fn r -> div(head.(r), div(nh, ng)) end

    # channel indices of each part, into in_proj's rows and conv1d's channels (nil = a zero pad)
    padded = fn base -> for g <- 0..(ng - 1), j <- 0..(np - 1), do: if(j < n, do: base + g * n + j) end
    parts = [x: {Enum.map(0..(di - 1), &(di + &1)), Enum.to_list(0..(di - 1))},
             b: {padded.(2 * di), padded.(di)},
             c: {padded.(2 * di + ng * n), padded.(di + ng * n)}]
    dt_rows = for h <- 0..(nhp - 1), do: if(h < nh, do: 2 * di + 2 * ng * n + h)

    {lets, embed} = Spatial.bind([], "backbone.embeddings.weight", get.("backbone.embeddings.weight"))
    x = T.gather_row(embed, T.input(:tok, :s32, [1]))

    # build-time constants shared by every layer
    expand = Tensor.from_list(:f32, [di, nhp], for(r <- 0..(di - 1), h <- 0..(nhp - 1), do: if(head.(r) == h, do: 1.0, else: 0.0)))
    {lets, expand} = Spatial.bind(lets, "mamba2.head_expand", expand)
    gidx = T.const(Tensor.from_list(:s32, [di], Enum.map(0..(di - 1), group)))

    {x, lets, nexts} =
      Enum.reduce(0..(c.layers - 1), {x, lets, []}, fn l, {x, lets, nexts} ->
        p = "backbone.layers.#{l}."
        m = p <> "mixer."
        b = fn lets, name, t -> Spatial.bind(lets, "#{p}#{name}", t) end
        w_in = get.(m <> "in_proj.weight")
        b_in = if c.bias, do: get.(m <> "in_proj.bias")
        cw = m <> "conv1d.weight" |> get.() |> Tensor.to_floats() |> List.to_tuple()
        cb = if c.conv_bias, do: m <> "conv1d.bias" |> get.() |> Tensor.to_floats() |> List.to_tuple()

        {lets, nw} = b.(lets, "norm", row(get.(p <> "norm.weight")))
        {lets, h} = Spatial.name(lets, "#{p}h", norm(x, nw, c.eps))

        proj = fn lets, name, idxs ->
          {lets, w} = b.(lets, "in_#{name}", pick_rows(w_in, idxs))
          t = T.linear(h, w)
          if b_in, do: (fn {lets, bb} -> {lets, T.add(t, bb)} end).(b.(lets, "in_#{name}_bias", pick(b_in, idxs))), else: {lets, t}
        end

        {lets, z} = proj.(lets, "z", Enum.to_list(0..(di - 1)))
        {lets, z} = Spatial.name(lets, "#{p}z", z)

        # x, B and C: projection, then the causal depthwise convolution (k − 1 past inputs are state), then silu
        {lets, convs, nexts} =
          Enum.reduce(parts, {lets, %{}, nexts}, fn {part, {rows_in, chans}}, {lets, acc, nexts} ->
            {lets, cur} = proj.(lets, "#{part}", rows_in)
            {lets, cur} = Spatial.name(lets, "#{p}#{part}_in", cur)
            w = length(chans)
            past = for j <- 0..(k - 2), do: T.input(:"conv#{l}_#{part}_#{j}", :f32, [1, w])
            inputs = past ++ [cur]

            {lets, terms} =
              Enum.reduce(Enum.with_index(inputs), {lets, []}, fn {inp, j}, {lets, ts} ->
                col = Tensor.from_list(:f32, [1, w], for(ch <- chans, do: if(ch, do: elem(cw, ch * k + j), else: 0.0)))
                {lets, wr} = b.(lets, "conv_#{part}_w#{j}", col)
                {lets, ts ++ [T.mul(inp, wr)]}
              end)

            conv = Enum.reduce(terms, fn t, a -> T.add(a, t) end)

            {lets, conv} =
              if cb,
                do: (fn {lets, r} -> {lets, T.add(conv, r)} end).(b.(lets, "conv_#{part}_b", Tensor.from_list(:f32, [1, w], for(ch <- chans, do: if(ch, do: elem(cb, ch), else: 0.0))))),
                else: {lets, conv}

            {lets, u} = Spatial.name(lets, "ssm#{l}_#{part}", T.silu(conv))
            nx = for j <- 0..(k - 2), do: {:"conv#{l}_#{part}_#{j}_next", Enum.at(inputs, j + 1)}
            {lets, Map.put(acc, part, u), nexts ++ nx}
          end)

        u = convs.x

        # dt per head, clamped, then expanded to its head's channels (exactly)
        {lets, dt_lin} = proj.(lets, "dt", dt_rows)
        {lets, dtb} = b.(lets, "dt_bias", Tensor.from_list(:f32, [1, nhp], pad_list(Tensor.to_floats(get.(m <> "dt_bias")), nhp)))
        dt = T.softplus(T.add(dt_lin, dtb))
        dt = clamp(dt, c.limit)
        {lets, dt} = Spatial.name(lets, "#{p}dt", dt)
        {lets, dtc} = Spatial.name(lets, "#{p}dt_col", T.transpose(T.linear(dt, expand)))

        # the scan, one step
        a = get.(m <> "A_log") |> Tensor.to_floats() |> Enum.map(&(-F32.to_float(F32.from_float(Vapor.CR.exp_f64(&1)))))
        {lets, a_col} = b.(lets, "A", Tensor.from_list(:f32, [di, 1], for(r <- 0..(di - 1), do: Enum.at(a, head.(r)))))
        {bb, cc} = if ng == 1, do: {convs.b, convs.c}, else: {T.gather_row(T.reshape(convs.b, [ng, np]), gidx), T.gather_row(T.reshape(convs.c, [ng, np]), gidx)}
        {lets, uc} = Spatial.name(lets, "#{p}u_col", T.transpose(u))
        da = T.exp(T.mul(dtc, a_col))
        dbx = T.mul(T.mul(dtc, bb), uc)
        s_in = T.input(:"ssm#{l}", :f32, [di, np])
        {lets, s} = Spatial.name(lets, "ssm#{l}_next_v", T.add(T.mul(s_in, da), dbx))
        ys = T.transpose(T.reduce(:sum, T.mul(s, cc)))
        dvals = get.(m <> "D") |> Tensor.to_floats()
        {lets, dd} = b.(lets, "D", Tensor.from_list(:f32, [1, di], for(r <- 0..(di - 1), do: Enum.at(dvals, head.(r)))))
        y = T.add(ys, T.mul(u, dd))

        # gated RMSNorm, per group (mamba_ssm) or over the whole width (transformers)
        {lets, gw} = b.(lets, "gated_norm", row(get.(m <> "norm.weight")))
        {lets, g} = Spatial.name(lets, "#{p}gated", T.mul(y, T.silu(z)))
        yn = if c.gated_norm == :whole or ng == 1, do: norm(g, gw, c.eps), else: group_norm(g, gw, ng, div(di, ng), c.eps)

        {lets, w_o} = b.(lets, "out", get.(m <> "out_proj.weight"))
        o = T.linear(yn, w_o)
        {lets, o} = if c.bias, do: (fn {lets, bo} -> {lets, T.add(o, bo)} end).(b.(lets, "out_bias", row(get.(m <> "out_proj.bias")))), else: {lets, o}
        {lets, x} = Spatial.name(lets, "#{p}x", T.add(x, o))
        {x, lets, nexts ++ [{:"ssm#{l}_next", s}]}
      end)

    {lets, nf} = Spatial.bind(lets, "backbone.norm_f", row(get.("backbone.norm_f.weight")))
    {lets, hid} = Spatial.name(lets, "hidden_v", norm(x, nf, c.eps))
    {lets, head_w} = if c.tie, do: {lets, embed}, else: Spatial.bind(lets, "lm_head", get.("lm_head.weight"))

    outputs = [logits: T.linear(hid, head_w), hidden: hid] ++ nexts
    state = for {name, _} <- state(c), do: {name, :"#{name}_next"}
    {:ok, Program.new(outputs, state: state, lets: Enum.reverse(lets))}
  end

  defp clamp(dt, {lo, hi}) do
    dt = if lo > 0.0, do: T.max(dt, T.splat(lo)), else: dt
    if hi == :infinity, do: dt, else: T.min(dt, T.splat(hi))
  end

  # x · rsqrt(Σx²·(1/d) + ε), then the weight (as `Vapor.Model.Llama`)
  defp norm(x, w, eps) do
    {:ok, {:f32, shape}} = T.infer(x)
    ms = T.mul(T.reduce(:sum, T.mul(x, x)), T.splat(1.0 / List.last(shape)))
    T.mul(w, T.mul(x, T.rsqrt(T.add(ms, T.splat(eps)))))
  end

  # the same, each group of `gs` channels on its own
  defp group_norm(x, w, ng, gs, eps) do
    g = T.reshape(x, [ng, gs])
    ms = T.mul(T.reduce(:sum, T.mul(g, g)), T.splat(1.0 / gs))
    T.mul(w, T.reshape(T.mul(g, T.rsqrt(T.add(ms, T.splat(eps)))), [1, ng * gs]))
  end

  defp row(%Tensor{shape: [n]} = t), do: Tensor.new(:f32, [1, n], t.data)

  defp pad_list(l, n), do: l ++ List.duplicate(0.0, n - length(l))

  defp pick_list(%Tensor{shape: [_]} = t, idxs) do
    v = t |> Tensor.to_floats() |> List.to_tuple()
    for i <- idxs, do: if(i, do: elem(v, i), else: 0.0)
  end

  defp pick(%Tensor{shape: [_]} = t, idxs), do: Tensor.from_list(:f32, [1, length(idxs)], pick_list(t, idxs))

  defp pick_rows(%Tensor{shape: [_, k]} = t, idxs) do
    z = :binary.copy(<<0::32>>, k)
    Tensor.new(:f32, [length(idxs), k], for(i <- idxs, into: <<>>, do: if(i, do: binary_part(t.data, i * k * 4, k * 4), else: z)))
  end
end
