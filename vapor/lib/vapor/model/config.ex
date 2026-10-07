defmodule Vapor.Model.Config do
  @moduledoc """
  A Hugging Face `config.json`, checked. Every supported family is one
  pre-norm decoder — RMSNorm, rotary positions, grouped-query attention,
  gated MLP — and differs in details this struct makes explicit:

  | `model_type` | attention | MLP | notes |
  |---|---|---|---|
  | `llama`   | GQA, `attention_bias` on q/k/v/o | SwiGLU | `rope_scaling` none, `linear`, `llama3` or `yarn` |
  | `mistral` | GQA | SwiGLU | `sliding_window` executed exactly (attention over the last `w` positions) |
  | `qwen2`   | GQA, q/k/v bias | SwiGLU | |
  | `qwen3`   | GQA + per-head q/k RMSNorm | SwiGLU | explicit `head_dim` |
  | `qwen3_moe` | as `qwen3` | top-k softmax mixture of experts | `norm_topk_prob`, `decoder_sparse_step`, `mlp_only_layers` |
  | `mixtral` | as `mistral` | top-k softmax mixture of experts (renormalised) | |
  | `gemma3_text` | GQA + q/k norm, `query_pre_attn_scalar` | GeGLU (tanh) | `(1 + w)` norms, sandwich norms, √d embedding scale, local/global RoPE per layer type |
  | `deepseek_v3` | multi-head latent attention (MLA) | dense first layers, then sigmoid group-limited MoE with shared experts | `rope_interleave`, YaRN with `mscale_all_dim` |

  Anything else (other activations, MLP biases, attention logit soft-capping,
  dynamic or LongRoPE scaling, …) is
  a rejection naming the field, never a silently wrong model.

  `raw` keeps the checked map: `to_map/1` gives it back for the families
  whose spelling vapor does not rebuild field by field.
  """
  alias Vapor.Rejection

  @enforce_keys [:arch, :vocab, :hidden, :intermediate, :layers, :heads, :kv_heads, :head_dim, :eps,
                 :rope_theta, :max_pos, :tie]
  defstruct [:arch, :vocab, :hidden, :intermediate, :layers, :heads, :kv_heads, :head_dim, :eps,
             :rope_theta, :max_pos, :tie, rope_scaling: nil, qkv_bias: false, o_bias: false,
             sliding_window: nil, bos: nil, eos: nil,
             # frontier families (defaults are the Llama decoder)
             act: :silu, qk_norm: false, norm_offset: false, embed_scale: nil, sandwich: false,
             attn_scale: nil, layer_types: nil, rope_local: nil, final_softcap: nil,
             moe: nil, mla: nil, raw: nil,
             # blueprint knobs (Granite-style multipliers): residual branches
             # scaled before the add, logits divided at the end
             residual_scale: nil, logit_divisor: nil,
             # partial rotary embeddings (Phi-4-mini, GPT-NeoX style): the
             # first `rotary_dim` dimensions of each head rotate, the rest pass
             rotary_dim: nil]

  @type t :: %__MODULE__{}

  @families ~w(llama mistral qwen2 qwen3 qwen3_moe mixtral gemma3_text deepseek_v3)
  @doc "The `model_type` values vapor builds."
  def families, do: @families

  @spec from_map(map) :: {:ok, t} | {:error, Rejection.t()}
  def from_map(c) when is_map(c) do
    arch = c["model_type"]

    with :ok <- need(arch in @families, "model_type", "one of #{Enum.join(@families, ", ")} (got #{inspect(arch)})"),
         {:ok, act} <- activation(arch, c),
         :ok <- need(c["mlp_bias"] in [nil, false], "mlp_bias", "false"),
         :ok <- need(c["attn_logit_softcapping"] == nil, "attn_logit_softcapping",
                     "null (soft-capped attention scores are not an operator of the algebra yet)"),
         {:ok, rope, local} <- rope_params(arch, c),
         {:ok, scaling} <- scaling(rope, c),
         :ok <- rope_keys(rope),
         {:ok, ints} <- ints(c, ~w(vocab_size hidden_size intermediate_size num_hidden_layers num_attention_heads)) do
      [v, d, ff, l, h] = ints
      hkv = c["num_key_value_heads"] || h
      dh = c["head_dim"] || div(d, h)
      bias = c["attention_bias"] == true

      cfg = %__MODULE__{
        arch: arch, vocab: v, hidden: d, intermediate: ff, layers: l, heads: h, kv_heads: hkv, head_dim: dh,
        eps: (c["rms_norm_eps"] || 1.0e-6) * 1.0, rope_theta: theta(c, rope), max_pos: c["max_position_embeddings"] || 2048,
        tie: tie(arch, c), rope_scaling: scaling, act: act,
        # Qwen2/3 declare a window they do not use unless use_sliding_window
        sliding_window: if(c["use_sliding_window"] == false, do: nil, else: c["sliding_window"]),
        qkv_bias: arch == "qwen2" or (arch in ~w(llama qwen3 qwen3_moe) and bias),
        o_bias: arch in ~w(llama qwen3 qwen3_moe) and bias,
        bos: c["bos_token_id"], eos: c["eos_token_id"],
        # only the frontier families are written back from the map they came from
        raw: if(arch in ~w(llama mistral qwen2), do: nil, else: Map.drop(c, ~w(torch_dtype dtype transformers_version)))
      }

      with {:ok, cfg} <- family(cfg, c, local),
           {:ok, cfg} <- layer_types(cfg, c),
           {:ok, cfg} <- partial_rotary(cfg, rope, c),
           :ok <- need(is_integer(cfg.kv_heads) and cfg.kv_heads > 0 and rem(cfg.heads, cfg.kv_heads) == 0,
                       "num_key_value_heads", "a divisor of num_attention_heads"),
           :ok <- need(is_integer(cfg.head_dim) and rem(cfg.head_dim, 16) == 0, "head_dim", "a multiple of 16 (got #{inspect(cfg.head_dim)})"),
           :ok <- need(rem(d, 16) == 0 and rem(ff, 16) == 0, "hidden_size/intermediate_size", "multiples of 16") do
        {:ok, cfg}
      end
    end
  end

  defp tie("gemma3_text", c), do: c["tie_word_embeddings"] != false
  defp tie(_, c), do: c["tie_word_embeddings"] == true

  defp theta(c, rope), do: ((rope && rope["rope_theta"]) || c["rope_theta"] || 10_000.0) * 1.0

  defp activation("gemma3_text", c) do
    a = c["hidden_activation"] || c["hidden_act"]
    if a in ["gelu_pytorch_tanh", "gelu_new", "gelu_fast"],
      do: {:ok, :gelu_tanh},
      else: {:error, Rejection.new({:config, "hidden_activation"}, "gelu_pytorch_tanh (got #{inspect(a)})", "use a supported checkpoint")}
  end

  defp activation(_arch, c) do
    case c["hidden_act"] do
      a when a in [nil, "silu", "swish"] -> {:ok, :silu}
      a when a in ["gelu_pytorch_tanh", "gelu_new"] -> {:ok, :gelu_tanh}
      "gelu" -> {:ok, :gelu}
      a -> {:error, Rejection.new({:config, "hidden_act"}, "silu, gelu or gelu_pytorch_tanh (got #{inspect(a)})", "use a supported checkpoint")}
    end
  end

  # transformers ≥ 5 writes `rope_parameters` (θ inside), possibly nested by
  # layer type (Gemma 3); earlier versions `rope_theta` and `rope_scaling`
  defp rope_params("gemma3_text", c) do
    case c["rope_parameters"] do
      %{"full_attention" => full} = nested when is_map(full) ->
        {:ok, full, nested["sliding_attention"]}

      _ ->
        local = %{"rope_theta" => c["rope_local_base_freq"] || 10_000.0, "rope_type" => "default"}
        {:ok, c["rope_scaling"] && Map.put_new(c["rope_scaling"], "rope_theta", c["rope_theta"]), local}
    end
  end

  defp rope_params(_arch, c), do: {:ok, c["rope_parameters"] || c["rope_scaling"], nil}

  # ------------------------------------------------------------ families --

  defp family(%{arch: a} = cfg, _c, _local) when a in ~w(llama mistral qwen2), do: {:ok, cfg}


  defp family(%{arch: "qwen3"} = cfg, _c, _local), do: {:ok, %{cfg | qk_norm: true}}

  defp family(%{arch: "qwen3_moe"} = cfg, c, _local) do
    # Qwen's checkpoints say num_experts; transformers ≥ 5 writes num_local_experts
    c = Map.put_new(c, "num_experts", c["num_local_experts"])

    with {:ok, [e, k, mi]} <- ints(c, ~w(num_experts num_experts_per_tok moe_intermediate_size)),
         :ok <- need(k <= e, "num_experts_per_tok", "at most num_experts"),
         :ok <- need(rem(mi, 16) == 0, "moe_intermediate_size", "a multiple of 16") do
      step = c["decoder_sparse_step"] || 1
      dense = MapSet.new(c["mlp_only_layers"] || [])
      sparse = for l <- 0..(cfg.layers - 1), do: not MapSet.member?(dense, l) and rem(l + 1, step) == 0

      {:ok, %{cfg | qk_norm: true,
                    moe: %{kind: :softmax, experts: e, top_k: k, norm: c["norm_topk_prob"] != false, inter: mi,
                           sparse: sparse, names: :qwen, shared: 0, scale: 1.0}}}
    end
  end

  defp family(%{arch: "mixtral"} = cfg, c, _local) do
    with {:ok, [e, k]} <- ints(c, ~w(num_local_experts num_experts_per_tok)),
         :ok <- need(k <= e, "num_experts_per_tok", "at most num_local_experts") do
      {:ok, %{cfg | moe: %{kind: :softmax, experts: e, top_k: k, norm: true, inter: cfg.intermediate,
                           sparse: List.duplicate(true, cfg.layers), names: :mixtral, shared: 0, scale: 1.0}}}
    end
  end

  defp family(%{arch: "gemma3_text"} = cfg, c, local) do
    types = c["layer_types"] || gemma_pattern(cfg.layers, c["sliding_window_pattern"] || c["_sliding_window_pattern"] || 6)

    with {:ok, local_scaling} <- scaling(local, c),
         :ok <- need(Enum.all?(types, &(&1 in ["sliding_attention", "full_attention"])), "layer_types",
                     "sliding_attention or full_attention") do
      qpas = c["query_pre_attn_scalar"] || cfg.head_dim

      {:ok, %{cfg | qk_norm: true, norm_offset: true, sandwich: true,
                    embed_scale: Vapor.CR.to_f32(:math.sqrt(cfg.hidden)),
                    attn_scale: Vapor.CR.pow_f64(qpas * 1.0, -0.5),
                    layer_types: Enum.map(types, &if(&1 == "sliding_attention", do: :sliding, else: :full)),
                    rope_local: {((local && local["rope_theta"]) || 10_000.0) * 1.0, local_scaling},
                    final_softcap: c["final_logit_softcapping"]}}
    end
  end

  defp family(%{arch: "deepseek_v3"} = cfg, c, _local) do
    with {:ok, [kvr, dn, dr, dv]} <- ints(c, ~w(kv_lora_rank qk_nope_head_dim qk_rope_head_dim v_head_dim)),
         {:ok, [e, k, mi, g, kg]} <- ints(c, ~w(n_routed_experts num_experts_per_tok moe_intermediate_size n_group topk_group)),
         :ok <- need(c["attention_bias"] in [nil, false], "attention_bias", "false"),
         :ok <- need(rem(dr, 2) == 0 and rem(dn, 2) == 0, "qk_rope_head_dim/qk_nope_head_dim", "even"),
         :ok <- need(rem(kvr, 16) == 0 and rem(dr, 16) == 0, "kv_lora_rank/qk_rope_head_dim", "multiples of 16"),
         :ok <- need(c["q_lora_rank"] == nil or rem(c["q_lora_rank"], 16) == 0, "q_lora_rank", "null or a multiple of 16"),
         :ok <- need(rem(e, g) == 0 and kg <= g and k <= kg * div(e, g), "n_group/topk_group",
                     "groups divide the experts and the kept groups hold at least num_experts_per_tok"),
         :ok <- need(rem(mi, 16) == 0, "moe_intermediate_size", "a multiple of 16") do
      # head layout: dn + dr slots, widened to hold dv and to a multiple of 16
      dh = (max(dn + dr, dv) + 15) |> div(16) |> Kernel.*(16)
      first = c["first_k_dense_replace"] || 0
      step = c["moe_layer_freq"] || 1
      sparse = for l <- 0..(cfg.layers - 1), do: l >= first and rem(l, step) == 0

      {:ok, %{cfg | kv_heads: cfg.heads, head_dim: dh, attn_scale: mla_scale(dn + dr, cfg.rope_scaling),
                    mla: %{q_lora: c["q_lora_rank"], kv_lora: kvr, nope: dn, rope: dr, v: dv,
                           interleave: c["rope_interleave"] != false},
                    moe: %{kind: :sigmoid_group, experts: e, top_k: k, norm: c["norm_topk_prob"] != false, inter: mi,
                           sparse: sparse, names: :qwen, shared: c["n_shared_experts"] || 0,
                           groups: g, topk_group: kg, scale: (c["routed_scaling_factor"] || 1.0) * 1.0}}}
    end
  end

  # DeepSeek: qk_head_dim^-½, times mscale² when YaRN declares mscale_all_dim
  defp mla_scale(qk, {:yarn, %{mscale_all_dim: m, factor: f}}) when is_number(m) and m != 0 do
    ms = if f <= 1, do: 1.0, else: 0.1 * m * Vapor.CR.log_f64(f * 1.0) + 1.0
    Vapor.CR.pow_f64(qk * 1.0, -0.5) * ms * ms
  end

  defp mla_scale(qk, _), do: Vapor.CR.pow_f64(qk * 1.0, -0.5)

  defp gemma_pattern(n, p), do: for(l <- 0..(n - 1), do: if(rem(l + 1, p) == 0, do: "full_attention", else: "sliding_attention"))

  # ---------------------------------------------------------------- to_map --

  @doc """
  The `config.json` map of a configuration — the inverse of `from_map/1`
  (`from_map(to_map(c)) = c`, tested), in the form transformers reads.
  Refused when the configuration has no Hugging Face spelling: rotary
  frequency factors that arrived as a GGUF tensor, or biases a family
  cannot declare. Frontier families are written back as they were read.
  """
  def to_map(%__MODULE__{arch: a, raw: raw}) when a not in ~w(llama mistral qwen2) and is_map(raw),
    do: {:ok, Map.put(raw, "torch_dtype", "float32")}

  # a Llama-topology model with a sliding window (Phi-3 through its alias)
  # has the Hugging Face spelling of Mistral: same layers, no biases, a window
  def to_map(%__MODULE__{arch: "llama", sliding_window: w, qkv_bias: false, o_bias: false} = c) when is_integer(w),
    do: to_map(%{c | arch: "mistral"})

  def to_map(%__MODULE__{} = c) do
    arch_class = %{"llama" => "LlamaForCausalLM", "mistral" => "MistralForCausalLM", "qwen2" => "Qwen2ForCausalLM"}

    with :ok <- need(Map.has_key?(arch_class, c.arch), "model_type", "a family with a rebuilt spelling, or a configuration read from config.json"),
         {:ok, rope} <- rope_map(c.rope_scaling),
         :ok <- biases(c) do
      base = %{
        "architectures" => [arch_class[c.arch]], "model_type" => c.arch, "vocab_size" => c.vocab,
        "hidden_size" => c.hidden, "intermediate_size" => c.intermediate, "num_hidden_layers" => c.layers,
        "num_attention_heads" => c.heads, "num_key_value_heads" => c.kv_heads, "head_dim" => c.head_dim,
        "rms_norm_eps" => c.eps, "rope_theta" => c.rope_theta, "max_position_embeddings" => c.max_pos,
        "tie_word_embeddings" => c.tie, "hidden_act" => "silu", "rope_scaling" => rope,
        "bos_token_id" => c.bos, "eos_token_id" => c.eos, "torch_dtype" => "float32"
      }

      base = if c.rotary_dim, do: Map.put(base, "partial_rotary_factor", c.rotary_dim / c.head_dim), else: base

      {:ok,
       Map.merge(base, case c.arch do
         "llama" -> %{"attention_bias" => c.qkv_bias, "mlp_bias" => false}
         "mistral" -> %{"sliding_window" => c.sliding_window} |> then(&if(c.layer_types,
                        do: Map.put(&1, "layer_types", Enum.map(c.layer_types, fn t -> "#{t}_attention" end)),
                        else: &1))
         "qwen2" -> %{"use_sliding_window" => false, "sliding_window" => c.sliding_window}
       end)}
    end
  end

  defp rope_map(nil), do: {:ok, nil}
  defp rope_map({:linear, f}), do: {:ok, %{"rope_type" => "linear", "factor" => f}}

  defp rope_map({:llama3, f, lo, hi, orig}),
    do: {:ok, %{"rope_type" => "llama3", "factor" => f, "low_freq_factor" => lo, "high_freq_factor" => hi,
                "original_max_position_embeddings" => orig}}

  defp rope_map({:yarn, y}) do
    {:ok,
     %{"rope_type" => "yarn", "factor" => y.factor, "original_max_position_embeddings" => y.orig,
       "beta_fast" => y.beta_fast, "beta_slow" => y.beta_slow, "attention_factor" => y.attention_factor_given,
       "mscale" => y.mscale, "mscale_all_dim" => y.mscale_all_dim, "truncate" => y.truncate}
     |> Map.reject(fn {_, v} -> v == nil end)}
  end

  defp rope_map(:freq_factors),
    do: {:error, Rejection.new({:config, "rope_scaling"}, "a Hugging Face rope_scaling (the frequency factors came as a GGUF tensor)",
                               "export to GGUF instead")}

  defp biases(%{arch: "llama", qkv_bias: b, o_bias: b}), do: :ok
  defp biases(%{arch: "mistral", qkv_bias: false, o_bias: false}), do: :ok
  defp biases(%{arch: "qwen2", qkv_bias: true, o_bias: false}), do: :ok
  defp biases(c), do: {:error, Rejection.new({:config, "attention_bias"}, "biases #{c.arch} can declare", "export to GGUF instead")}

  @doc "Parse and check a `config.json` file."
  def load(path) do
    with {:ok, bin} <- File.read(path),
         {:ok, map} <- Vapor.JSON.decode(bin) do
      from_map(map)
    else
      {:error, %Rejection{}} = e -> e
      {:error, why} -> {:error, Rejection.new({:config, path}, "readable JSON (#{inspect(why)})", "check the file")}
    end
  end

  @doc """
  Whether a sliding window binds within a cache of `s` rows (`w < s` on some
  sliding layer). A binding window is executed exactly — attention over the
  last `w` positions (`Vapor.Algebra.Term.attention/8`); one that cannot
  bind is left out of the program.
  """
  def window_binds?(%__MODULE__{sliding_window: w, layer_types: types}, s) when is_integer(w) and w < s,
    do: types == nil or :sliding in types

  def window_binds?(_cfg, _s), do: false

  @doc """
  The window `w` when it binds on **every** layer within a cache of `s` rows
  (Mistral 7B v0.1, Phi-3-mini's 2 047…), else `nil`. Only then can a
  serving cache be circular: no layer ever reads a position older than
  `w`, so pages behind the window may be overwritten (`Vapor.Engine`).
  A hybrid (Gemma 2/3: global layers between local ones) keeps every
  position and is `nil`.
  """
  def ring_window(%__MODULE__{sliding_window: w, layer_types: types}, s) when is_integer(w) and w < s,
    do: if(types == nil or Enum.all?(types, &(&1 == :sliding)), do: w, else: nil)

  def ring_window(_cfg, _s), do: nil

  @doc "Whether layer `l` is a mixture-of-experts layer."
  def moe_layer?(%__MODULE__{moe: nil}, _l), do: false
  def moe_layer?(%__MODULE__{moe: %{sparse: sparse}}, l), do: Enum.at(sparse, l, false)

  # ------------------------------------------------------------ rope scaling --

  # Every key a rope map may carry, by meaning. An unknown key is refused:
  # transformers ≥ 5 moved `partial_rotary_factor` *inside* rope_parameters,
  # where 0.4.0 did not look — a Phi-4-mini-style checkpoint was admitted and
  # computed wrong (found against transformers itself, 2026-10-02). Keys that
  # change the math must be read or refused, never skipped.
  @rope_keys ~w(rope_type type rope_theta factor low_freq_factor high_freq_factor original_max_position_embeddings
                beta_fast beta_slow attention_factor mscale mscale_all_dim truncate partial_rotary_factor)

  defp rope_keys(nil), do: :ok

  defp rope_keys(%{} = r) do
    case Enum.find(Map.keys(r), &(&1 not in @rope_keys)) do
      nil -> :ok
      k -> {:error, Rejection.new({:config, "rope_parameters.#{k}"}, "a key whose meaning vapor implements (#{Enum.join(@rope_keys, ", ")})",
                                  "an unknown rope key could change the positions; extend Vapor.Model.Config or use a supported checkpoint")}
    end
  end

  defp rope_keys(_), do: :ok

  # `layer_types` (transformers ≥ 4.5x writes it for Mistral, Qwen2/3…):
  # which layers slide. A key that changes the attention is read or
  # refused, never skipped — a hybrid read as all-sliding would compute
  # the global layers wrong (and let the engine recycle pages they read)
  defp layer_types(%{layer_types: t} = cfg, _c) when is_list(t), do: {:ok, cfg}
  defp layer_types(cfg, %{"layer_types" => nil}), do: {:ok, cfg}

  defp layer_types(cfg, %{"layer_types" => types}) do
    cond do
      not (is_list(types) and length(types) == cfg.layers and Enum.all?(types, &(&1 in ["sliding_attention", "full_attention"]))) ->
        {:error, Rejection.new({:config, "layer_types"}, "one of sliding_attention / full_attention per layer",
                                     "fix layer_types (#{cfg.layers} entries)")}

      cfg.sliding_window == nil and "sliding_attention" in types ->
        {:error, Rejection.new({:config, "layer_types"}, "sliding layers only with a sliding_window", "set sliding_window")}

      cfg.sliding_window == nil or Enum.all?(types, &(&1 == "sliding_attention")) ->
        {:ok, cfg}

      true ->
        {:ok, %{cfg | layer_types: Enum.map(types, &if(&1 == "sliding_attention", do: :sliding, else: :full))}}
    end
  end

  defp layer_types(cfg, _c), do: {:ok, cfg}

  # `partial_rotary_factor` at the top level (transformers < 5) or inside the
  # rope map (≥ 5); rotary_dim = int(head_dim · factor), as transformers
  defp partial_rotary(cfg, rope, c) do
    f = (is_map(rope) && rope["partial_rotary_factor"]) || c["partial_rotary_factor"]

    cond do
      f in [nil, 1, 1.0] -> {:ok, cfg}
      not is_number(f) or f <= 0 or f > 1 -> {:error, Rejection.new({:config, "partial_rotary_factor"}, "a number in (0, 1]", "check the config")}
      cfg.mla != nil or cfg.qk_norm or cfg.rope_local != nil ->
        {:error, Rejection.new({:config, "partial_rotary_factor"}, "1 for #{cfg.arch} (partial rotary is built for the plain decoder layout)", "use a supported checkpoint")}
      true ->
        r = trunc(cfg.head_dim * f)
        if r >= 2 and rem(r, 2) == 0 and r < cfg.head_dim,
          do: {:ok, %{cfg | rotary_dim: r}},
          else: {:error, Rejection.new({:config, "partial_rotary_factor"}, "int(head_dim · factor) even and in [2, head_dim) (got #{r})", "check the config")}
    end
  end

  defp scaling(nil, _c), do: {:ok, nil}

  defp scaling(%{} = r, c) do
    case r["rope_type"] || r["type"] do
      t when t in [nil, "default"] -> {:ok, nil}
      "linear" -> {:ok, {:linear, r["factor"] * 1.0}}
      "llama3" -> {:ok, {:llama3, r["factor"] * 1.0, r["low_freq_factor"] * 1.0, r["high_freq_factor"] * 1.0,
                         r["original_max_position_embeddings"]}}
      "yarn" -> yarn(r, c)
      other -> {:error, Rejection.new({:config, "rope_scaling"}, "rope_type default, linear, llama3 or yarn (got #{inspect(other)}; " <>
                                     "dynamic and longrope change with the sequence length)", "use a supported checkpoint")}
    end
  end

  # the parameters transformers' `_compute_yarn_parameters` reads, with its defaults
  defp yarn(r, c) do
    orig = r["original_max_position_embeddings"] || c["original_max_position_embeddings"] || c["max_position_embeddings"]
    factor = r["factor"] || (c["max_position_embeddings"] && orig && c["max_position_embeddings"] / orig)

    if is_number(factor) and is_integer(orig) do
      {:ok,
       {:yarn, %{factor: factor * 1.0, orig: orig, beta_fast: (r["beta_fast"] || 32) * 1.0, beta_slow: (r["beta_slow"] || 1) * 1.0,
                 attention_factor_given: r["attention_factor"], mscale: r["mscale"], mscale_all_dim: r["mscale_all_dim"],
                 truncate: r["truncate"] != false}}}
    else
      {:error, Rejection.new({:config, "rope_scaling"}, "yarn with factor and original_max_position_embeddings", "check the config")}
    end
  end

  defp ints(c, keys) do
    vals = Enum.map(keys, &c[&1])
    bad = Enum.find(Enum.zip(keys, vals), fn {_, v} -> not (is_integer(v) and v > 0) end)
    if bad, do: {:error, Rejection.new({:config, elem(bad, 0)}, "a positive integer", "check the config")}, else: {:ok, vals}
  end

  defp need(true, _field, _bound), do: :ok
  defp need(false, field, bound), do: {:error, Rejection.new({:config, field}, bound, "use a supported checkpoint")}
end
