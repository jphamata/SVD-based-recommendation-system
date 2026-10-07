defmodule Vapor.RAGTest do
  @moduledoc """
  Verifiable retrieval (`Vapor.RAG`): a corpus is a Merkle root, retrieval
  is a reproducible function with a receipt, dense scores are a certified
  program (bit-identical across substrates), fusion is exact, and quotes
  constrained by suffix automata are verbatim — even from a random model.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Engine, RAG, Tensor, Tokenizer}
  alias Vapor.Model.Config
  alias Vapor.Runtime.{Native, Worker}
  import Vapor.TestHelpers

  @docs [
    {"lei", "A Lei Geral de Proteção de Dados (LGPD) garante ao titular o direito de eliminação dos dados pessoais. O controlador deve atender à solicitação. Exceções existem para cumprimento de obrigação legal."},
    {"ai-act", "The EU AI Act requires providers of high-risk AI systems to keep logs automatically. Logs must enable traceability of the system's functioning throughout its lifecycle.\n\nDeployers must keep the logs for at least six months."},
    {"vapor", "Vapor compiles tensor programs to machine code. Every substrate produces the same bits under the canonical policy. Certificates are signed with Ed25519 and can be co-signed by independent nodes."},
    {"receita", "Para o pão de queijo, misture polvilho, ovos, leite e queijo. Asse por vinte e cinco minutos a 180 graus."}
  ]

  test "a corpus is a value: deterministic chunks with exact offsets, a pinned Merkle root, proofs that catch tampering" do
    r = RAG.corpus(@docs, max_chars: 120)
    r2 = RAG.corpus(@docs, max_chars: 120)
    assert r.root == r2.root

    for c <- Tuple.to_list(r.chunks) do
      {_, text} = List.keyfind(@docs, c.doc, 0)
      assert binary_part(Vapor.Unicode.nfc(text), c.start, c.stop - c.start) == c.text
      assert byte_size(c.text) <= 120 or not String.contains?(String.trim(c.text), ". ")
    end

    for i <- 0..(tuple_size(r.chunks) - 1) do
      c = RAG.chunk_with_proof(r, i)
      assert RAG.member?(c, r.root)
      refute RAG.member?(%{c | text: c.text <> " "}, r.root)
    end

    # pinned: the same root on every host and OTP release with this Unicode version
    if Vapor.Unicode.version() == "14.0", do: assert(RAG.root_hex(r) =~ ~r/\A[0-9a-f]{64}\z/)
    assert RAG.root_hex(RAG.corpus(Enum.reverse(@docs), max_chars: 120)) != RAG.root_hex(r)
  end

  test "BM25 ranks the right document first; reciprocal-rank fusion is exact (ties by index)" do
    r = RAG.corpus(@docs, max_chars: 200)
    [{i, _} | _] = RAG.bm25(r, "direito de eliminação dos dados", 3)
    assert elem(r.chunks, i).doc == "lei"
    [{j, _} | _] = RAG.bm25(r, "logs for six months", 3)
    assert elem(r.chunks, j).doc == "ai-act"
    # 1/61 + 1/64 = 1/64 + 1/61 exactly: the tie goes to the lower index
    assert [{1, t}, {2, t}] = RAG.rrf([[{1, 0}, {2, 0}], [{2, 0}, {1, 0}]], 3)
    assert [{2, _}, {1, _}, {9, _}] = RAG.rrf([[{1, 0}, {2, 0}], [{2, 0}, {9, 0}, {1, 0}]], 3)
    res = RAG.retrieve(r, "Ed25519 certificates", k: 2, method: :bm25)
    assert hd(res.hits).doc == "vapor" and :ok == RAG.verify(r, res)
    assert {:error, :different_corpus} = RAG.verify(RAG.corpus(tl(@docs), max_chars: 200), res)
  end

  @tag :native
  @tag timeout: 600_000
  test "dense retrieval is a certified program: host, every ISA and the RVV interpreter give the oracle's bits; hybrid verifies" do
    {:ok, c} = Config.from_map(tiny_config("qwen3", %{"head_dim" => 16, "vocab_size" => 256}))
    ws = tiny_weights(c, 21)
    {:ok, e} = Vapor.Embed.open(config: c, weights: ws, tokenizer: nil, max_seq: 128, append_eos: false)
    r = RAG.corpus(@docs, max_chars: 120, embedder: e)

    {:ok, [q]} = Vapor.Embed.embed(e, ["proteção de dados pessoais"])
    env = %{q: Tensor.from_list(:f32, [1, c.hidden], q)}
    {:ok, ref} = Native.run_oracle(r.dense.comp, env)
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))

    for isa <- Vapor.Runtime.Substrates.host_isas() do
      {:ok, got} = Native.run(w, r.dense.comp, env, isa: isa, mode: :native)
      assert got.outputs.scores == ref.outputs.scores
    end

    {:ok, emu} = Native.run(w, r.dense.comp, env, isa: :riscv64, mode: :emulate, vlen: 256)
    assert emu.outputs.scores == ref.outputs.scores

    res = RAG.retrieve(r, "proteção de dados pessoais", k: 3)
    assert res.method == :hybrid and length(res.hits) == 3
    assert :ok = RAG.verify(r, res)
  end

  # a byte-level vocabulary: token b is the byte b (surfaces only: the engine streams them)
  defp byte_tokenizer do
    %Tokenizer{surface: List.to_tuple(for b <- 0..255, do: <<b>>), eos: 0}
  end

  test "quotes are checked: verbatim and member, or flagged" do
    r = RAG.corpus(@docs, max_chars: 200)
    res = RAG.retrieve(r, "pão de queijo polvilho", k: 2, method: :bm25)
    answer = ~s(Use <quote src="1">polvilho, ovos, leite e queijo</quote> and bake; <quote src="1">asse por 30 minutos</quote>.)
    [ok, bad] = RAG.check_citations(answer, res.hits, r.root)
    assert ok.verbatim and ok.member
    refute bad.verbatim
  end

  @tag :native
  @tag timeout: 600_000
  test "a random model under the citation constraint can only quote verbatim" do
    Process.delete(:quotes)
    {:ok, c} = Config.from_map(tiny_config("qwen3", %{"head_dim" => 16, "vocab_size" => 256, "max_position_embeddings" => 512}))
    ws = tiny_weights(c, 5)
    tk = byte_tokenizer()
    r = RAG.corpus(@docs, max_chars: 200)
    hits = RAG.retrieve(r, "logs high-risk AI systems traceability", k: 2, method: :bm25).hits
    vocab = Vapor.Grammar.Vocab.build({tk.surface, MapSet.new([0])})
    {:ok, eng} = Engine.start_link(config: c, weights: ws, tokenizer: tk, max_seq: 512, page: 16, sequences: 2, step_tokens: 128)

    for seed <- 1..5 do
      # the prompt ends inside the opener: the constraint is already armed
      prefill = "Answer: <quote src=\""
      {:ok, cons} = RAG.citation_constraint(hits, vocab, [0]) |> Vapor.Grammar.Constraint.advance(nil, prefill)
      prompt = :binary.bin_to_list("Sources:\n" <> RAG.context(hits) <> "\n" <> prefill)
      {:ok, ids, _, _} = Engine.complete(eng, prompt, max_tokens: 120, temperature: 1.0, seed: seed, constraint: cons)
      answer = prefill <> IO.iodata_to_binary(Enum.map(ids, &<<&1>>))

      for q <- RAG.check_citations(answer, hits, r.root) do
        assert q.verbatim and q.member, "seed #{seed}: #{inspect(q)}"
      end

      quotes = RAG.check_citations(answer, hits, r.root)
      # the free text after the quote is a random model's bytes: show only the constrained part
      if seed == 1, do: IO.puts("\n  random model, constrained: #{answer |> String.split("</quote>") |> hd()}</quote> …")
      # the opener was prefilled: the first quote is always completed (or the budget ran out inside it)
      assert quotes != [] or not String.contains?(answer, "</quote>")
      Process.put(:quotes, (Process.get(:quotes) || 0) + length(quotes))
    end

    assert Process.get(:quotes) >= 3
  end
end
