defmodule Vapor.Quality.Planted do
  @moduledoc """
  **Planted models**: transformers whose weights are written, not trained,
  so that the right output is known in closed form. They are the ground
  truth that makes "the stack does not produce noise" a falsifiable claim.

  `bigram/3` embeds a counted bigram model — the maximum-likelihood
  order-1 model of a corpus, with additive smoothing — *exactly* into the
  Llama topology:

      embed[i]      = e_i                       (one-hot; width d = V padded to 16)
      every q,k,v,o = 0,  gate, up, down = 0    (both residual branches add +0)
      norms         = 1
      lm_head[j, i] = log P(j | i) / s          s = 1/√(1/d + ε): what RMSNorm makes of e_i

  so the final hidden row of token `i` is `s·e_i` and its logits are
  `log P(· | i)` up to binary32 rounding. The checkpoint is an ordinary
  `config.json` map and safetensors-named weights: it goes through the
  model airlock, the compiler, the ladder and any substrate like a real
  model, and every one of them must reproduce the table (tested to 10⁻⁵).
  Counting *is* training here: the model is the corpus's own statistics.

  A planted model is not a language model; it is an instrument. Real
  checkpoints are judged by the same gates and the same bits-per-token.
  """
  alias Vapor.Tensor

  @doc """
  The bigram model of `ids` over a vocabulary of `v` tokens. Options:
  `alpha` (additive smoothing, default 0.05), `eps` (1.0e-6), `table` (a
  `vp × vp` table of logits to plant instead of the counted one — e.g. a
  `finetune/4` result; `ids` then only sets nothing but the size).
  Returns `%{config, weights, logp}` (`logp[i][j]` = log P(j | i), binary64,
  for the first `v` ids; padding ids get log P = −30).
  """
  def bigram(ids, v, opts \\ []) do
    alpha = Keyword.get(opts, :alpha, 0.05)
    eps = Keyword.get(opts, :eps, 1.0e-6)
    vp = div(v + 15, 16) * 16
    d = vp

    counts = ids |> Enum.chunk_every(2, 1, :discard) |> Enum.frequencies()
    totals = Enum.reduce(counts, %{}, fn {[i, _], c}, acc -> Map.update(acc, i, c, &(&1 + c)) end)

    logp =
      opts[:table] ||
      for i <- 0..(vp - 1) do
        for j <- 0..(vp - 1) do
          cond do
            j >= v -> -30.0
            i >= v -> -:math.log(v)
            true -> :math.log((Map.get(counts, [i, j], 0) + alpha) / (Map.get(totals, i, 0) + alpha * v))
          end
        end
      end

    s = 1.0 / :math.sqrt(1.0 / d + eps)
    lt = List.to_tuple(Enum.map(logp, &List.to_tuple/1))
    # lm_head row j, column i = logp[i][j] / s
    head = for j <- 0..(vp - 1), i <- 0..(d - 1), do: elem(elem(lt, i), j) / s

    config = %{"model_type" => "llama", "vocab_size" => vp, "hidden_size" => d, "intermediate_size" => 16,
               "num_hidden_layers" => 1, "num_attention_heads" => 1, "num_key_value_heads" => 1, "head_dim" => d,
               "rms_norm_eps" => eps, "rope_theta" => 10_000.0, "max_position_embeddings" => 4096,
               "tie_word_embeddings" => false, "hidden_act" => "silu"}

    z = fn shape -> Tensor.new(:f32, shape, :binary.copy(<<0::32>>, Enum.product(shape))) end
    one = fn n -> Tensor.from_list(:f32, [n], List.duplicate(1.0, n)) end
    eye = Tensor.from_list(:f32, [vp, d], for(i <- 0..(vp - 1), j <- 0..(d - 1), do: if(i == j, do: 1.0, else: 0.0)))
    p = "model.layers.0."

    weights = %{
      "model.embed_tokens.weight" => eye, "model.norm.weight" => one.(d), "lm_head.weight" => Tensor.from_list(:f32, [vp, d], head),
      (p <> "input_layernorm.weight") => one.(d), (p <> "post_attention_layernorm.weight") => one.(d),
      (p <> "self_attn.q_proj.weight") => z.([d, d]), (p <> "self_attn.k_proj.weight") => z.([d, d]),
      (p <> "self_attn.v_proj.weight") => z.([d, d]), (p <> "self_attn.o_proj.weight") => z.([d, d]),
      (p <> "mlp.gate_proj.weight") => z.([16, d]), (p <> "mlp.up_proj.weight") => z.([16, d]),
      (p <> "mlp.down_proj.weight") => z.([d, 16])
    }

    %{config: config, weights: weights, logp: logp, vocab: v}
  end

  @doc """
  **Fine-tuning, in closed form**: `steps` of full-batch gradient descent on
  the cross-entropy of `ids` (the next-token loss), starting from the table
  `logits` (`vp × vp`, a `bigram/3` result's `logp`), with learning rate
  `lr`. The gradient of the loss of context `i` with respect to its logits
  is `nᵢ·softmax(zᵢ) − cᵢ` (counts), divided by the number of pairs.
  Padding rows and columns (ids ≥ `v`) are left as they are. Returns the
  new table — a fine-tune whose delta from its base is small and dense,
  the regime task-vector methods were designed for.
  """
  def finetune(ids, logits, v, opts \\ []) do
    steps = Keyword.get(opts, :steps, 20)
    lr = Keyword.get(opts, :lr, 1.0)
    pairs = Enum.chunk_every(ids, 2, 1, :discard)
    total = max(length(pairs), 1)
    counts = Enum.frequencies(pairs)
    rows = Enum.reduce(counts, %{}, fn {[i, j], c}, acc -> Map.update(acc, i, %{j => c}, &Map.put(&1, j, c)) end)

    Enum.reduce(1..steps//1, logits, fn _, z ->
      z
      |> Enum.with_index()
      |> Enum.map(fn {zi, i} ->
        case rows[i] do
          nil -> zi
          ci when i < v ->
            ni = ci |> Map.values() |> Enum.sum()
            real = Enum.take(zi, v)
            m = Enum.max(real)
            es = Enum.map(real, &:math.exp(&1 - m))
            se = Enum.sum(es)
            new = es |> Enum.with_index() |> Enum.zip(real) |> Enum.map(fn {{e, j}, zij} -> zij - lr * (ni * e / se - Map.get(ci, j, 0)) / total end)
            new ++ Enum.drop(zi, v)
          _ -> zi
        end
      end)
    end)
  end

  @doc "Bits per token of `ids` under the analytic table (what the model must reproduce)."
  def table_bits(%{logp: logp}, ids) do
    lt = List.to_tuple(Enum.map(logp, &List.to_tuple/1))
    pairs = Enum.chunk_every(ids, 2, 1, :discard)
    -Enum.reduce(pairs, 0.0, fn [i, j], s -> s + elem(elem(lt, i), j) end) / :math.log(2) / max(length(pairs), 1)
  end

  @doc "A word vocabulary: `%{words, id}` with `id.(word)` and `word.(id)`."
  def words(list) do
    t = List.to_tuple(list)
    index = list |> Enum.with_index() |> Map.new()
    %{words: list, size: length(list), id: &Map.fetch!(index, &1), word: &elem(t, &1)}
  end

  @doc """
  Bytes → ids over a compact alphabet: lowercase ASCII letters, digits,
  space, `.,;:!?-'\\n` and one id for everything else — enough for the
  statistics of prose, small enough for the exact oracle.
  """
  def alphabet do
    chars = Enum.to_list(?a..?z) ++ Enum.to_list(?0..?9) ++ ~c" .,;:!?-'\n"
    index = chars |> Enum.with_index() |> Map.new()
    other = length(chars)

    %{size: other + 1, chars: chars,
      encode: fn text -> text |> String.downcase() |> :binary.bin_to_list() |> Enum.map(&Map.get(index, &1, other)) end,
      decode: fn ids -> ids |> Enum.map(&Enum.at(chars, &1, ?_)) |> List.to_string() end}
  end
end
