defmodule Vapor.Model.GGUF do
  @moduledoc """
  A llama.cpp GGUF model → `Vapor.Model.Config` + weights with Hugging Face
  names, through the GGUF airlock (`Vapor.Ingest.GGUF`) and the reference
  dequantisation (`Vapor.Ingest.GGML`).

  Architectures `llama` (Llama, Mistral) and `qwen2`. Two conventions of
  llama.cpp's converter are undone so the weights mean what the
  `rotate-half` RoPE of `Vapor.Model.Decoder` expects:

    * for `llama`, the rows of `attn_q`/`attn_k` (weights and biases) are
      permuted within each head from `(pair, half)` to `(half, pair)` order
      — the converter's `permute`, inverted;
    * Llama 3 frequency scaling arrives as a `rope_freqs` tensor of per-pair
      divisors (applied to θ^(−2i/d) as llama.cpp does).

  Tensors are read one at a time from the file (positioned reads), so the
  BEAM never holds the raw file.

  `write/4` is the inverse: a configuration and Hugging Face–named weights
  → a GGUF that llama.cpp loads, with the converter's conventions applied
  (q/k row order for `llama`, Llama 3 scaling as `rope_freqs`, Mistral
  written as `llama`, whose GGUF has no sliding window: exact while caches
  stay within the window) and the vocabulary's `tokenizer.*` metadata
  copied in. `load(write(m)) = m` bit for bit in `f32` (tested).
  """
  alias Vapor.{Rejection, Tensor}
  alias Vapor.Ingest.{GGML, GGUF}
  alias Vapor.Model.Config

  @names %{"token_embd.weight" => "model.embed_tokens.weight", "output_norm.weight" => "model.norm.weight",
           "output.weight" => "lm_head.weight"}
  @block %{"attn_norm.weight" => "input_layernorm.weight", "ffn_norm.weight" => "post_attention_layernorm.weight",
           "attn_q.weight" => "self_attn.q_proj.weight", "attn_k.weight" => "self_attn.k_proj.weight",
           "attn_v.weight" => "self_attn.v_proj.weight", "attn_output.weight" => "self_attn.o_proj.weight",
           "attn_q.bias" => "self_attn.q_proj.bias", "attn_k.bias" => "self_attn.k_proj.bias",
           "attn_v.bias" => "self_attn.v_proj.bias", "attn_output.bias" => "self_attn.o_proj.bias",
           "ffn_gate.weight" => "mlp.gate_proj.weight", "ffn_up.weight" => "mlp.up_proj.weight",
           "ffn_down.weight" => "mlp.down_proj.weight"}

  @doc "Load: `{:ok, %{config, weights, tokenizer, rope_freqs}}`."
  def load(path) do
    with {:ok, g} <- GGUF.read(path),
         {:ok, cfg} <- config(g),
         {:ok, ws} <- weights(path, g, cfg) do
      tk = case Vapor.Tokenizer.from_gguf(g.metadata) do
        {:ok, tk} -> tk
        _ -> nil
      end

      {:ok, %{config: cfg, weights: ws, tokenizer: tk}}
    end
  end

  defp config(%{metadata: m, tensors: ts}) do
    arch = m["general.architecture"]
    key = &m["#{arch}.#{&1}"]
    names = MapSet.new(ts, & &1.name)
    vocab = key.("vocab_size") || length(m["tokenizer.ggml.tokens"] || [])
    heads = key.("attention.head_count")

    scaling =
      case key.("rope.scaling.type") do
        "linear" -> %{"rope_type" => "linear", "factor" => key.("rope.scaling.factor")}
        _ -> nil
      end

    with true <- arch in ["llama", "qwen2"] || reject("general.architecture", "llama or qwen2 (got #{inspect(arch)})") do
      map = %{
        "model_type" => arch, "vocab_size" => vocab, "hidden_size" => key.("embedding_length"),
        "intermediate_size" => key.("feed_forward_length"), "num_hidden_layers" => key.("block_count"),
        "num_attention_heads" => heads, "num_key_value_heads" => key.("attention.head_count_kv") || heads,
        "head_dim" => key.("attention.key_length"), "rms_norm_eps" => key.("attention.layer_norm_rms_epsilon"),
        "rope_theta" => key.("rope.freq_base") || 10_000.0, "max_position_embeddings" => key.("context_length"),
        "tie_word_embeddings" => not MapSet.member?(names, "output.weight"),
        "attention_bias" => MapSet.member?(names, "blk.0.attn_q.bias"), "rope_scaling" => scaling,
        "bos_token_id" => m["tokenizer.ggml.bos_token_id"], "eos_token_id" => m["tokenizer.ggml.eos_token_id"]
      }

      with {:ok, c} <- Config.from_map(map) do
        # Llama 3 frequency scaling travels as a tensor of per-pair divisors
        case Enum.find(ts, &(&1.name == "rope_freqs.weight")) do
          nil -> {:ok, c}
          _ -> {:ok, %{c | rope_scaling: :freq_factors}}
        end
      end
    end
  end

  defp weights(path, g, cfg) do
    {:ok, f} = File.open(path, [:read, :binary])

    try do
      Enum.reduce_while(g.tensors, {:ok, %{}}, fn t, {:ok, acc} ->
        case hf_name(t.name) do
          nil when t.name == "rope_freqs.weight" -> {:cont, {:ok, Map.put(acc, :rope_freqs, read(f, g, t))}}
          nil -> {:cont, {:ok, acc}}
          name -> {:cont, {:ok, Map.put(acc, name, read(f, g, t) |> unpermute(name, cfg))}}
        end
      end)
    after
      File.close(f)
    end
  end

  defp read(f, g, t) do
    {:ok, raw} = :file.pread(f, g.data_start + t.offset, t.bytes)
    Tensor.new(:f32, Enum.reverse(t.dims), GGML.dequantize(t.type, raw))
  end

  defp hf_name(n) do
    case Map.fetch(@names, n) do
      {:ok, hf} ->
        hf

      :error ->
        case Regex.run(~r/^blk\.(\d+)\.(.+)$/, n) do
          [_, l, rest] -> (hf = @block[rest]) && "model.layers.#{l}.#{hf}"
          _ -> nil
        end
    end
  end

  # llama.cpp's converter reorders q/k rows of `llama` models within each
  # head: row (pair i, half j) of HF becomes row 2i + j. Invert it.
  defp unpermute(%Tensor{} = t, name, %Config{arch: "llama"} = c) do
    cond do
      String.ends_with?(name, "q_proj.weight") or String.ends_with?(name, "q_proj.bias") -> permute_back(t, c.heads, c.head_dim)
      String.ends_with?(name, "k_proj.weight") or String.ends_with?(name, "k_proj.bias") -> permute_back(t, c.kv_heads, c.head_dim)
      true -> t
    end
  end

  defp unpermute(t, _name, _c), do: t

  defp permute_back(%Tensor{shape: shape, data: data} = t, heads, dh) do
    rows = hd(shape)
    rb = div(byte_size(data), rows)
    half = div(dh, 2)
    row = fn r -> binary_part(data, r * rb, rb) end

    out =
      for h <- 0..(heads - 1), j <- 0..1, i <- 0..(half - 1), into: <<>> do
        row.(h * dh + 2 * i + j)
      end

    %{t | data: out}
  end

  @doc """
  Write `weights` (Hugging Face names, f32) of configuration `c` to a GGUF
  file. Options: `type` (`:f32`, default, or `:q8_0` — matrices whose rows
  are whole 32-blocks; vectors stay f32, as in llama.cpp's converter),
  `vocab` (a `Vapor.Ingest.GGUF.read/1` result, whose `tokenizer.*` keys
  are copied with their value types, or a bare metadata map), `name`.
  """
  def write(path, %Config{} = c, weights, opts \\ []) do
    with :ok <- writable(c), do: do_write(path, c, weights, opts)
  end

  # GGUF's `llama` and `qwen2` carry the plain decoder only: anything with
  # per-head norms, experts, latent attention, sandwich norms, multipliers
  # or another activation would lose tensors or knobs in silence
  defp writable(%Config{} = c) do
    extras = [qk_norm: c.qk_norm, moe: c.moe != nil, mla: c.mla != nil, sandwich: c.sandwich, norm_offset: c.norm_offset,
              embed_scale: c.embed_scale != nil, attn_scale: c.attn_scale != nil, residual_scale: c.residual_scale != nil,
              logit_divisor: c.logit_divisor != nil, final_softcap: c.final_softcap != nil, layer_types: c.layer_types != nil,
              act: c.act != :silu, rotary_dim: c.rotary_dim != nil]

    case for({k, true} <- extras, do: k) do
      [] -> :ok
      ks -> {:error, Rejection.new({:gguf, c.arch}, "a plain decoder (llama/mistral/qwen2 layout); #{c.arch} has #{inspect(ks)}",
                                   "export to safetensors (mix vapor.export --dtype …) instead")}
    end
  end

  defp do_write(path, %Config{} = c, weights, opts) do
    type = Keyword.get(opts, :type, :f32)
    arch = if c.arch == "qwen2", do: "qwen2", else: "llama"
    back = Map.new(@names, fn {g, h} -> {h, g} end)
    blk = Map.new(@block, fn {g, h} -> {h, g} end)

    gname = fn hf ->
      case Map.fetch(back, hf) do
        {:ok, g} -> g
        :error ->
          case Regex.run(~r/^model\.layers\.(\d+)\.(.+)$/, hf) do
            [_, l, rest] -> (g = blk[rest]) && "blk.#{l}.#{g}"
            _ -> nil
          end
      end
    end

    {freqs, scaling} =
      case c.rope_scaling do
        {:linear, f} -> {nil, [{"#{arch}.rope.scaling.type", "linear"}, {"#{arch}.rope.scaling.factor", {:f32, f}}]}
        {:llama3, _, _, _, _} = l3 -> {Tensor.from_list(:f32, [div(c.head_dim, 2)], llama3_factors(c, l3)), []}
        _ -> {weights[:rope_freqs], []}
      end

    tensors =
      weights
      |> Enum.filter(fn {k, _} -> is_binary(k) and gname.(k) != nil end)
      |> Enum.reject(fn {k, _} -> k == "lm_head.weight" and c.tie end)
      |> Enum.sort_by(fn {k, _} -> order(gname.(k)) end)
      |> Enum.map(fn {k, t} -> tensor(gname.(k), permute(t, k, c, arch), type) end)
      |> Kernel.++(if freqs, do: [tensor("rope_freqs.weight", freqs, :f32)], else: [])

    # the vocabulary's metadata, with its value types when known (`GGUF.read/1`)
    {vmeta, vtypes} =
      case Keyword.get(opts, :vocab, %{}) do
        %{metadata: m, types: t} -> {m, t}
        m -> {m, %{}}
      end

    vocab = for {k, v} <- vmeta, String.starts_with?(k, "tokenizer."), do: {k, if(t = vtypes[k], do: {t, v}, else: v)}
    key = &"#{arch}.#{&1}"

    meta =
      [{"general.architecture", arch}, {"general.name", Keyword.get(opts, :name, arch)},
       {"general.file_type", if(type == :q8_0, do: 7, else: 0)},
       {key.("vocab_size"), c.vocab}, {key.("context_length"), c.max_pos}, {key.("embedding_length"), c.hidden},
       {key.("block_count"), c.layers}, {key.("feed_forward_length"), c.intermediate},
       {key.("attention.head_count"), c.heads}, {key.("attention.head_count_kv"), c.kv_heads},
       {key.("attention.key_length"), c.head_dim}, {key.("attention.value_length"), c.head_dim},
       {key.("rope.dimension_count"), c.head_dim}, {key.("rope.freq_base"), {:f32, c.rope_theta}},
       {key.("attention.layer_norm_rms_epsilon"), {:f32, c.eps}}] ++ scaling ++
        Enum.sort(vocab) ++
        Enum.reject([{"tokenizer.ggml.bos_token_id", c.bos}, {"tokenizer.ggml.eos_token_id", c.eos}],
                    fn {k, v} -> not is_integer(v) or List.keymember?(vocab, k, 0) end)

    GGUF.write(path, meta, tensors)
  end

  # llama.cpp's converter: token_embd, then blocks in order, then output_norm/output
  defp order("token_embd.weight"), do: {0, 0, ""}
  defp order("blk." <> rest), do: (([l, n] = String.split(rest, ".", parts: 2)); {1, String.to_integer(l), n})
  defp order(n), do: {2, 0, n}

  defp tensor(name, %Tensor{shape: shape, data: data}, type) do
    dims = Enum.reverse(shape)

    if type == :q8_0 and length(shape) == 2 and rem(hd(dims), 32) == 0,
      do: {name, :q8_0, dims, GGML.quantize(:q8_0, data)},
      else: {name, :f32, dims, data}
  end

  defp permute(t, name, c, "llama") do
    cond do
      String.ends_with?(name, "q_proj.weight") or String.ends_with?(name, "q_proj.bias") -> permute_fwd(t, c.heads, c.head_dim)
      String.ends_with?(name, "k_proj.weight") or String.ends_with?(name, "k_proj.bias") -> permute_fwd(t, c.kv_heads, c.head_dim)
      true -> t
    end
  end

  defp permute(t, _name, _c, _arch), do: t

  # HF row (half j, pair i) → GGUF row 2i + j
  defp permute_fwd(%Tensor{shape: shape, data: data} = t, heads, dh) do
    rb = div(byte_size(data), hd(shape))
    half = div(dh, 2)
    out = for h <- 0..(heads - 1), i <- 0..(half - 1), j <- 0..1, into: <<>>, do: binary_part(data, (h * dh + j * half + i) * rb, rb)
    %{t | data: out}
  end

  # llama.cpp's converter (LlamaModel.generate_extra_tensors): a divisor per frequency
  defp llama3_factors(c, {:llama3, factor, low, high, orig}) do
    {low_wl, high_wl} = {orig / low, orig / high}

    for i <- 0..(div(c.head_dim, 2) - 1) do
      freq = 1.0 / Vapor.CR.pow_f64(c.rope_theta, 2 * i / c.head_dim)
      wl = 2 * :math.pi() / freq

      cond do
        wl < high_wl -> 1.0
        wl > low_wl -> factor
        true -> 1 / ((1 - (orig / wl - low) / (high - low)) / factor + (orig / wl - low) / (high - low))
      end
    end
  end

  defp reject(field, bound), do: {:error, Rejection.new({:gguf, field}, bound, "use a supported model")}
end
