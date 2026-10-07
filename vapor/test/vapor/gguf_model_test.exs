defmodule Vapor.GGUFModelTest do
  @moduledoc """
  Phase P8 — GGUF weights. Dequantisation of every supported ggml type is
  bit-identical to gguf-py's reference; a model converted by llama.cpp's
  own `convert_hf_to_gguf.py` (from the llama-cpp-python sdist on PyPI,
  `$VAPOR_LLAMA_CPP`) loads with its q/k permutation undone and matches
  transformers' logits (f32 to 1e-5 relative; q8_0 to its quantisation).
  """
  use ExUnit.Case, async: false
  alias Vapor.Tensor
  alias Vapor.Ingest.GGML
  import Vapor.TestHelpers

  @moduletag timeout: 900_000

  @tag :gguf_py
  test "ggml dequantisation = gguf-py, bit for bit (F16 BF16 Q8_0 Q4_0 Q4_1 Q5_0 Q5_1 Q4_K Q5_K Q6_K)" do
    out = py!(File.read!(Path.expand("../python/ggml_dequant.py", __DIR__)))

    for line <- String.split(out, "\n", trim: true) do
      [t, raw, want] = String.split(line, " ")
      got = GGML.dequantize(String.to_atom(t), Base.decode16!(raw, case: :lower))
      assert got == Base.decode16!(want, case: :lower), t
    end
  end

  @tag :llama_cpp
  test "a GGUF from llama.cpp's converter (f32, q8_0) reproduces transformers' logits" do
    lcpp = System.fetch_env!("VAPOR_LLAMA_CPP")
    out = Path.join(System.tmp_dir!(), "vapor-gguf-#{System.unique_integer([:positive])}")
    File.mkdir_p!(out)
    on_exit(fn -> File.rm_rf!(out) end)
    tok = Path.join(out, "llama3.json")
    py!("""
    import json, sys
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(sys.argv[1], gguf_file="ggml-vocab-llama-bpe.gguf")
    j = json.loads(t.backend_tokenizer.to_str()); j["model"]["ignore_merges"] = True
    json.dump(j, open(sys.argv[2], "w"))
    """, [Path.expand("../fixtures/vocab", __DIR__), tok])
    py!(File.read!(Path.expand("../python/hf_gguf.py", __DIR__)), [lcpp, tok, out])
    {:ok, ref} = Vapor.Ingest.Safetensors.read(Path.join(out, "reference.safetensors"))
    {:ok, w} = Vapor.Runtime.Worker.start_link(exec: worker_exec(:host))
    want = Tensor.to_floats(ref["logits"])
    scale = want |> Enum.map(&abs/1) |> Enum.max()

    for {t, tol} <- [{"f32", 1.0e-5}, {"q8_0", 0.05}] do
      {:ok, m} = Vapor.Model.GGUF.load(Path.join(out, "model-#{t}.gguf"))
      assert m.config.arch == "llama" and m.tokenizer != nil
      {:ok, p} = Vapor.Model.Llama.program(m.config, m.weights, max_seq: 32)
      {:ok, c} = Vapor.Compile.Lower.lower(p)
      n = hd(ref["prompt"].shape)
      env = Map.merge(Vapor.Model.Llama.empty_caches(m.config, 32), %{tok: ref["prompt"], pos: Tensor.from_list(:s32, [n], Enum.to_list(0..(n - 1)))})
      {:ok, r} = Vapor.Runtime.Native.run(w, c, env, isa: Vapor.Runtime.Substrates.host_isa(), mode: :native)
      err = Enum.zip_with(Tensor.to_floats(r.outputs.logits), want, &abs(&1 - &2)) |> Enum.max()
      assert err <= tol * scale, "#{t}: max |Δ| #{err}"
    end
  end
end
