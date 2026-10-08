defmodule Vapor.GGUFWriteTest do
  @moduledoc """
  GGUF export — the inverse of the airlock. A model written by
  `Vapor.Model.GGUF.write/4` loads back to the very same configuration and
  weights (f32: bit for bit, through the q/k permutation and the name map);
  Q8_0 quantisation is gguf-py's to the bit, and gguf-py reads our files.
  """
  use ExUnit.Case, async: true
  alias Vapor.Tensor
  alias Vapor.Ingest.{GGML, GGUF}
  alias Vapor.Model.Config
  import Vapor.TestHelpers
  import Bitwise

  @vocab Path.expand("../fixtures/vocab/ggml-vocab-llama-bpe.gguf", __DIR__)

  defp tmp(name) do
    dir = Path.join(System.tmp_dir!(), "vapor-gguf-w-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)
    Path.join(dir, name)
  end

  test "metadata of every type survives write → read" do
    path = tmp("m.gguf")
    meta = [{"a.u32", 7}, {"a.neg", -3}, {"a.wide", 1 <<< 40}, {"a.f32", 0.5}, {"a.f64", {:f64, 0.1}}, {"a.bool", true},
            {"a.str", "çã ✓"}, {"a.ints", [1, 2, 3]}, {"a.strs", ["x", "", "ü"]}, {"a.floats", [0.25, -1.5]},
            {"a.i8", {:i8, -5}}, {"a.u16", {:u16, 65_535}}]
    t = Tensor.from_list(:f32, [2, 32], Enum.map(1..64, &(&1 / 8)))
    :ok = GGUF.write(path, meta, [{"t", :f32, [32, 2], t.data}, {"q", :q8_0, [32, 2], GGML.quantize(:q8_0, t.data)}])
    {:ok, g} = GGUF.read(path)

    assert g.metadata == %{"a.u32" => 7, "a.neg" => -3, "a.wide" => 1 <<< 40, "a.f32" => 0.5, "a.f64" => 0.1, "a.bool" => true,
                           "a.str" => "çã ✓", "a.ints" => [1, 2, 3], "a.strs" => ["x", "", "ü"], "a.floats" => [0.25, -1.5],
                           "a.i8" => -5, "a.u16" => 65_535}
    assert g.types == %{"a.u32" => :u32, "a.neg" => :i32, "a.wide" => :u64, "a.f32" => :f32, "a.f64" => :f64, "a.bool" => :bool,
                        "a.str" => :string, "a.ints" => {:array, :u32}, "a.strs" => {:array, :string},
                        "a.floats" => {:array, :f32}, "a.i8" => :i8, "a.u16" => :u16}
    assert [%{name: "t", type: :f32, dims: [32, 2]}, %{name: "q", type: :q8_0, dims: [32, 2], bytes: 68}] = g.tensors
  end

  for arch <- ["llama", "qwen2"] do
    test "#{arch}: write (f32) → load is the identity on configuration and weights" do
      over = if unquote(arch) == "llama", do: %{"attention_bias" => true, "bos_token_id" => 1, "eos_token_id" => 2}, else: %{}
      {:ok, c} = Config.from_map(tiny_config(unquote(arch), over))
      w = tiny_weights(c)
      path = tmp("m.gguf")
      :ok = Vapor.Model.GGUF.write(path, c, w)
      {:ok, m} = Vapor.Model.GGUF.load(path)

      # GGUF stores ε in binary32 — the precision the program uses anyway
      assert m.config == %{c | eps: Vapor.F32.to_float(Vapor.F32.from_float(c.eps))}
      assert Map.keys(m.weights) |> Enum.sort() == Map.keys(w) |> Enum.sort()
      for {k, t} <- w, do: assert(m.weights[k] == t, k)
    end
  end

  # |x − q·d₁₆| ≤ d/2 + 127·|d − d₁₆| ≤ d·(1/2 + 127·2⁻¹¹)
  test "q8_0: matrices quantised (within half a step and the f16 scale's rounding), vectors kept exact" do
    {:ok, c} = Config.from_map(tiny_config("qwen2"))
    w = tiny_weights(c)
    path = tmp("m.gguf")
    :ok = Vapor.Model.GGUF.write(path, c, w, type: :q8_0)
    {:ok, g} = GGUF.read(path)
    {:ok, m} = Vapor.Model.GGUF.load(path)

    for %{name: n, type: t, dims: dims} <- g.tensors, do: assert(t == if(length(dims) == 2, do: :q8_0, else: :f32), n)

    for {k, t} <- w do
      got = m.weights[k]

      if length(t.shape) == 2 do
        assert got.data == GGML.dequantize(:q8_0, GGML.quantize(:q8_0, t.data))

        for {a, b} <- Enum.zip(Enum.chunk_every(Tensor.to_floats(t), 32), Enum.chunk_every(Tensor.to_floats(got), 32)) do
          step = Enum.max(Enum.map(a, &abs/1)) / 127
          assert Enum.zip_with(a, b, &abs(&1 - &2)) |> Enum.max() <= step * (0.5 + 127 / 2048) * (1 + 1.0e-6), k
        end
      else
        assert got == t, k
      end
    end
  end

  test "Llama 3 frequency scaling travels as rope_freqs and gives the same RoPE tables to binary32 rounding" do
    {:ok, c} =
      Config.from_map(tiny_config("llama", %{"rope_theta" => 500_000.0, "max_position_embeddings" => 64,
        "rope_scaling" => %{"rope_type" => "llama3", "factor" => 8.0, "low_freq_factor" => 1.0, "high_freq_factor" => 4.0,
                            "original_max_position_embeddings" => 16}}))

    path = tmp("m.gguf")
    :ok = Vapor.Model.GGUF.write(path, c, tiny_weights(c))
    {:ok, m} = Vapor.Model.GGUF.load(path)
    assert m.config.rope_scaling == :freq_factors
    {cos0, sin0} = Vapor.Model.Decoder.rope_tables(c, 64)
    {cos1, sin1} = Vapor.Model.Decoder.rope_tables(m.config, 64, m.weights[:rope_freqs])

    for {a, b} <- [{cos0, cos1}, {sin0, sin1}] do
      err = Enum.zip_with(Tensor.to_floats(a), Tensor.to_floats(b), &abs(&1 - &2)) |> Enum.max()
      assert err <= 64 * 4.0e-7
    end
  end

  # a real 150k-entry vocabulary and a model with that many rows: minutes on
  # a small machine, not seconds
  @tag :vocab
  @tag timeout: 600_000
  test "the vocabulary travels with the model" do
    {:ok, g} = GGUF.read(@vocab)
    {:ok, want} = Vapor.Tokenizer.from_gguf(g.metadata)
    {:ok, c} = Config.from_map(tiny_config("llama", %{"vocab_size" => Vapor.Tokenizer.vocab_size(want)}))
    path = tmp("m.gguf")
    :ok = Vapor.Model.GGUF.write(path, c, tiny_weights(c), vocab: g)
    {:ok, m} = Vapor.Model.GGUF.load(path)
    text = "Hello, world! Ça va? 12345"
    assert Vapor.Tokenizer.encode(m.tokenizer, text) == Vapor.Tokenizer.encode(want, text)
    # the vocabulary's value types travel too (llama.cpp checks them)
    {:ok, h} = GGUF.read(path)
    for {k, t} <- g.types, String.starts_with?(k, "tokenizer."), do: assert(h.types[k] == t, k)
    assert {:ok, %{tokenizer: %Vapor.Tokenizer{}}} = Vapor.Model.open(path)
  end

  # llama.cpp itself (libllama from the llama-cpp-python sdist) runs our
  # export: its tokenizer reads our vocabulary metadata, its graph our
  # weights; logits against vapor's on the same file
  @tag :llama_cpp_lib
  @tag timeout: 900_000
  test "llama.cpp runs vapor's GGUF export: same tokens, same logits (f32 to 1e-4, q8_0 to its quantisation)" do
    {:ok, w} = Vapor.Runtime.Worker.start_link(exec: worker_exec(:host))
    text = "Hello, world! The quick brown fox — ça va? 12345"

    for {arch, vocab} <- [{"llama", "llama-bpe"}, {"qwen2", "qwen2"}], {type, tol} <- [{:f32, 1.0e-4}, {:q8_0, 0.05}] do
      {:ok, g} = GGUF.read(Path.expand("../fixtures/vocab/ggml-vocab-#{vocab}.gguf", __DIR__))
      {:ok, tk} = Vapor.Tokenizer.from_gguf(g.metadata)
      over = %{"vocab_size" => Vapor.Tokenizer.vocab_size(tk), "max_position_embeddings" => 256, "num_hidden_layers" => 2}
      over = if arch == "llama", do: Map.put(over, "attention_bias", false), else: over
      {:ok, c} = Config.from_map(tiny_config(arch, over))
      ws = tiny_weights(c, 3)
      path = tmp("#{arch}-#{type}.gguf")
      :ok = Vapor.Model.GGUF.write(path, c, ws, type: type, vocab: g)
      {:ok, m} = Vapor.Model.GGUF.load(path)

      ids = Vapor.Tokenizer.encode(m.tokenizer, text, add_bos: arch == "llama")
      n = length(ids)
      {:ok, p} = Vapor.Model.Decoder.program(m.config, m.weights, max_seq: 64, logits: :all)
      {:ok, comp} = Vapor.Compile.Lower.lower(p)
      env = Map.merge(Vapor.Model.Decoder.empty_caches(m.config, 64), %{tok: Tensor.from_list(:s32, [n], ids), pos: Tensor.from_list(:s32, [n], Enum.to_list(0..(n - 1)))})
      {:ok, r} = Vapor.Runtime.Native.run(w, comp, env, isa: Vapor.Runtime.Substrates.best_isa(), mode: :native)
      ours = Tensor.to_floats(r.outputs.logits)

      out =
        py!("""
        import sys, json, numpy as np, llama_cpp
        from llama_cpp import Llama
        llm = Llama(model_path=sys.argv[1], n_ctx=64, n_batch=64, n_threads=1, logits_all=True, verbose=False,
                    type_k=llama_cpp.GGML_TYPE_F32, type_v=llama_cpp.GGML_TYPE_F32, flash_attn=False)
        toks = llm.tokenize(sys.argv[2].encode(), add_bos=sys.argv[3] == "1", special=False)
        ids = json.loads(sys.argv[4])
        llm.eval(ids)
        print(json.dumps(toks))
        print(np.asarray(llm.scores[:len(ids)], dtype=np.float32).tobytes().hex())
        """, [path, text, if(arch == "llama", do: "1", else: "0"), Vapor.JSON.encode(ids)])

      [toks, hex] = String.split(out, "\n", trim: true)
      assert {:ok, ^ids} = Vapor.JSON.decode(toks), "#{arch}: llama.cpp tokenizes our vocabulary the same"
      theirs = for <<x::float-32-little <- Base.decode16!(hex, case: :lower)>>, do: x
      scale = theirs |> Enum.map(&abs/1) |> Enum.max()
      err = Enum.zip_with(ours, theirs, &abs(&1 - &2)) |> Enum.max()
      assert err <= tol * scale, "#{arch} #{type}: max |Δ| #{err} (max |logit| #{scale})"
    end
  end

  @tag :gguf_py
  test "Q8_0 quantisation = gguf-py's, bit for bit; gguf-py reads our files" do
    path = tmp("m.gguf")
    {:ok, c} = Config.from_map(tiny_config("llama"))
    :ok = Vapor.Model.GGUF.write(path, c, tiny_weights(c), type: :q8_0)

    out =
      py!("""
      import sys, numpy as np
      from gguf.quants import quantize
      from gguf.constants import GGMLQuantizationType as Q
      from gguf import GGUFReader
      rng = np.random.default_rng(1)
      x = rng.standard_normal(32 * 64).astype(np.float32)
      x[:32] = np.arange(32, dtype=np.float32) - 15.5          # exact halves after scaling
      x[32:64] = 0                                             # an all-zero block
      x[64:96] *= 1e-30                                        # tiny scales (f16 subnormal / zero)
      print(x.tobytes().hex(), quantize(x.reshape(64, 32), Q.Q8_0).tobytes().hex())
      r = GGUFReader(sys.argv[1])
      for t in r.tensors:
          print(t.name, t.tensor_type.name, ",".join(map(str, t.shape)), bytes(t.data.tobytes()).hex())
      """, [path])

    [first | tensors] = String.split(out, "\n", trim: true)
    [x, q] = String.split(first, " ")
    assert GGML.quantize(:q8_0, Base.decode16!(x, case: :lower)) == Base.decode16!(q, case: :lower)

    {:ok, g} = GGUF.read(path)
    {:ok, f} = File.open(path, [:read, :binary])

    for {line, t} <- Enum.zip(tensors, g.tensors) do
      [name, type, dims, hex] = String.split(line, " ")
      {:ok, raw} = :file.pread(f, g.data_start + t.offset, t.bytes)
      assert {name, String.downcase(type), dims} == {t.name, Atom.to_string(t.type), Enum.join(t.dims, ",")}
      assert Base.decode16!(hex, case: :lower) == raw, name
    end

    File.close(f)
  end
end
