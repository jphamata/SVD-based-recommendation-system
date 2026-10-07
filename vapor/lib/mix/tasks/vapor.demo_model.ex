defmodule Mix.Tasks.Vapor.DemoModel do
  @shortdoc "Write a small random checkpoint with a real vocabulary (for the end-to-end pipeline)"
  @moduledoc """
      mix vapor.demo_model OUT_DIR [--vocab test/fixtures/vocab/ggml-vocab-qwen2.gguf]
                                   [--hidden 128] [--layers 2] [--heads 4] [--kv-heads 2] [--seed 1]

  Writes `config.json` (Qwen2 architecture), `model.safetensors` (random
  weights at unit activation scale) and `tokenizer.gguf` (the vocabulary,
  copied). The text it generates is meaningless — the point is that every
  stage of the real pipeline runs: airlocks, tokenizer, compiler, engine,
  server. A real checkpoint directory (config.json + safetensors +
  tokenizer.json) takes its place unchanged.
  """
  use Mix.Task
  import Bitwise
  alias Vapor.Tensor

  @switches [vocab: :string, hidden: :integer, layers: :integer, heads: :integer, kv_heads: :integer, seed: :integer]

  @impl true
  def run(argv) do
    {o, [out | _], _} = OptionParser.parse(argv, strict: @switches)
    vocab_file = o[:vocab] || "test/fixtures/vocab/ggml-vocab-qwen2.gguf"
    {:ok, g} = Vapor.Ingest.GGUF.read(vocab_file)
    {:ok, tk} = Vapor.Tokenizer.from_gguf(g.metadata)
    {d, l, h, hkv} = {o[:hidden] || 128, o[:layers] || 2, o[:heads] || 4, o[:kv_heads] || 2}
    v = Vapor.Tokenizer.vocab_size(tk)
    ff = 2 * d
    dh = div(d, h)
    :rand.seed(:exsss, {o[:seed] || 1, 7, 7})

    cfg = %{"architectures" => ["Qwen2ForCausalLM"], "model_type" => "qwen2", "vocab_size" => v, "hidden_size" => d,
            "intermediate_size" => ff, "num_hidden_layers" => l, "num_attention_heads" => h, "num_key_value_heads" => hkv,
            "max_position_embeddings" => 1024, "rms_norm_eps" => 1.0e-6, "rope_theta" => 1_000_000.0,
            "tie_word_embeddings" => true, "hidden_act" => "silu", "bos_token_id" => tk.bos, "eos_token_id" => tk.eos}

    w = fn shape, scale -> uniform(shape, scale) end
    ones = fn n -> Tensor.from_list(:f32, [n], List.duplicate(1.0, n)) end

    tensors =
      %{"model.embed_tokens.weight" => w.([v, d], 0.05), "model.norm.weight" => ones.(d)}
      |> Map.merge(
        Map.new(
          for i <- 0..(l - 1),
              {n, t} <- [{"input_layernorm.weight", ones.(d)}, {"post_attention_layernorm.weight", ones.(d)},
                         {"self_attn.q_proj.weight", w.([d, d], 1 / :math.sqrt(d))},
                         {"self_attn.k_proj.weight", w.([hkv * dh, d], 1 / :math.sqrt(d))},
                         {"self_attn.v_proj.weight", w.([hkv * dh, d], 1 / :math.sqrt(d))},
                         {"self_attn.q_proj.bias", w.([d], 0.1)}, {"self_attn.k_proj.bias", w.([hkv * dh], 0.1)},
                         {"self_attn.v_proj.bias", w.([hkv * dh], 0.1)},
                         {"self_attn.o_proj.weight", w.([d, d], 1 / :math.sqrt(d))},
                         {"mlp.gate_proj.weight", w.([ff, d], 1 / :math.sqrt(d))},
                         {"mlp.up_proj.weight", w.([ff, d], 1 / :math.sqrt(d))},
                         {"mlp.down_proj.weight", w.([d, ff], 1 / :math.sqrt(ff))}],
              do: {"model.layers.#{i}.#{n}", t}
        )
      )

    File.mkdir_p!(out)
    File.write!(Path.join(out, "config.json"), Vapor.JSON.encode(cfg))
    :ok = Vapor.Ingest.Safetensors.write(Path.join(out, "model.safetensors"), tensors, %{"format" => "pt"})
    File.cp!(vocab_file, Path.join(out, "tokenizer.gguf"))
    Mix.shell().info("wrote #{out}: #{v} tokens, hidden #{d}, #{l} layers (#{h} heads, #{hkv} kv heads)")
  end

  # uniform in [-√3·s, √3·s) (standard deviation s), from 23 random bits a value
  defp uniform(shape, s) do
    n = Enum.product(shape)
    k = :math.sqrt(3) * s * 2 / 8_388_608

    data =
      for <<b::32 <- :rand.bytes(4 * n)>>, into: <<>> do
        <<((b &&& 0x7F_FFFF) - 4_194_304) * k::float-32-little>>
      end

    Tensor.new(:f32, shape, data)
  end
end
