defmodule Vapor.TokenizerTest do
  @moduledoc """
  Phase P4, the tokenizer. Two references:

    * llama.cpp's vocabulary GGUFs of Llama 3, Qwen2, GPT-2 and Llama 2 with
      the test vectors Hugging Face produced for them (`make fixtures`,
      SHA-256 pinned): `from_gguf/1` must reproduce every vector;
    * Hugging Face `tokenizers` itself, on the same vocabularies converted
      to `tokenizer.json` (and patched back to each release's original
      options), over a generated corpus that stresses the pre-tokenizer
      patterns, Unicode (combining marks, ZWJ emoji, CJK, exotic spaces),
      digits, contractions, special-token strings and long unsplit text:
      `from_hf/1` must give the same ids and decode to the same bytes.
  """
  use ExUnit.Case, async: false
  import Bitwise
  alias Vapor.Tokenizer
  alias Vapor.Ingest.GGUF
  import Vapor.TestHelpers

  @dir Path.expand("../fixtures/vocab", __DIR__)
  @names ~w(llama-bpe qwen2 gpt-2 llama-spm)

  @moduletag timeout: 900_000

  defp gguf(name) do
    {:ok, g} = GGUF.read(Path.join(@dir, "ggml-vocab-#{name}.gguf"))
    {:ok, tk} = Tokenizer.from_gguf(g.metadata)
    tk
  end

  # ------------------------------------------------------------ fixtures --

  @tag :vocab
  test "GGUF vocabularies reproduce Hugging Face's test vectors exactly" do
    for name <- @names do
      tk = gguf(name)
      path = Path.join(@dir, "ggml-vocab-#{name}.gguf")
      inputs = File.read!(path <> ".inp") |> String.split("\n__ggml_vocab_test__\n") |> Enum.drop(-1)
      wants = File.read!(path <> ".out") |> String.split("\n") |> Enum.take(length(inputs))

      for {text, want} <- Enum.zip(inputs, wants) do
        ids = want |> String.split(" ", trim: true) |> Enum.map(&String.to_integer/1)
        assert Tokenizer.encode(tk, text, add_bos: false) == ids, "#{name}: #{inspect(text)}"
      end

      assert length(inputs) > 40
    end
  end

  @tag :vocab
  test "decoding inverts encoding (byte-level exactly; SentencePiece up to its prefix space)" do
    for name <- @names, tk = gguf(name), text <- corpus(60, 11) do
      ids = Tokenizer.encode(tk, text, add_bos: false, special: false)
      assert Tokenizer.decode(tk, ids) == text, "#{name}: #{inspect(text)}"
    end
  end

  @tag :vocab
  test "BOS, special tokens, byte fallback" do
    l3 = gguf("llama-bpe")
    [bos | rest] = Tokenizer.encode(l3, "Hi")
    assert bos == 128_000 and rest == Tokenizer.encode(l3, "Hi", add_bos: false)
    # a special-token string is one id — unless specials are plain text
    assert Tokenizer.encode(l3, "a<|end_of_text|>b", add_bos: false) |> Enum.member?(128_001)
    refute Tokenizer.encode(l3, "a<|end_of_text|>b", add_bos: false, special: false) |> Enum.member?(128_001)

    spm = gguf("llama-spm")
    # a character outside the vocabulary becomes its UTF-8 bytes as <0xXX> tokens
    ids = Tokenizer.encode(spm, "𒀀", add_bos: false)
    assert length(ids) >= 4 and Tokenizer.decode(spm, ids) == "𒀀"
  end

  test "GGUF airlock: malformed headers are refused" do
    tmp = Path.join(System.tmp_dir!(), "vapor-gguf-#{System.unique_integer([:positive])}.gguf")
    on_exit(fn -> File.rm(tmp) end)
    str = fn s -> <<byte_size(s)::64-little, s::binary>> end
    head = fn nt, nkv -> <<"GGUF", 3::32-little, nt::64-little, nkv::64-little>> end
    try_ = fn bin -> File.write!(tmp, bin); GGUF.read(tmp) end

    assert {:ok, %{metadata: %{"a" => 7, "b" => ["x", "y"]}}} =
             try_.(head.(0, 2) <> str.("a") <> <<4::32-little, 7::32-little>> <>
                     str.("b") <> <<9::32-little, 8::32-little, 2::64-little>> <> str.("x") <> str.("y"))

    for {bin, why} <- [
          {"GGUX" <> <<0::160>>, "GGUF magic"},
          {<<"GGUF", 9::32-little, 0::128>>, "version"},
          {head.(0, 1 <<< 40), "fit in the file"},
          {head.(0, 1) <> str.("a") <> <<99::32-little>>, "known GGUF value type"},
          {head.(0, 2) <> str.("a") <> <<4::32-little, 1::32-little>> <> str.("a") <> <<4::32-little, 1::32-little>>, "unique"},
          {head.(0, 1) <> str.("a") <> <<9::32-little, 8::32-little, (1 <<< 50)::64-little>>, "fits in the file"},
          {head.(0, 1) <> str.("a") <> <<8::32-little>> <> str.(<<0xFF>>), "UTF-8"},
          {head.(0, 1) <> <<50::64-little>> <> String.duplicate("a", 10), "complete header"},
          {head.(1, 0) <> str.("t") <> <<1::32-little, 4::64-little, 0::32-little, 0::64-little>>, "inside the file"}
        ] do
      assert {:error, %Vapor.Rejection{bound: bound}} = try_.(bin)
      assert bound =~ why, "#{why}: #{bound}"
    end
  end

  # ------------------------------------------------------ HF differential --

  @tag :hf_tokenizers
  test "tokenizer.json (converted and original options): same ids and same decoded bytes as HF tokenizers" do
    out = Path.join(System.tmp_dir!(), "vapor-tok-#{System.unique_integer([:positive])}")
    File.mkdir_p!(out)
    on_exit(fn -> File.rm_rf!(out) end)
    py!(File.read!(Path.expand("../python/hf_vocab.py", __DIR__)), [@dir, out])

    texts = corpus(400, 7)
    input = Enum.map_join(texts, &(&1 <> <<0>>))

    for file <- Enum.sort(File.ls!(out)) do
      json = Path.join(out, file)
      {:ok, tk} = Tokenizer.load(json)
      lines = py!(File.read!(Path.expand("../python/hf_tokenize.py", __DIR__)), [json], input) |> String.split("\n", trim: false)

      mism =
        texts
        |> Enum.zip(lines)
        |> Enum.reject(fn {text, line} ->
          [ids, hex] = String.split(line, "\t")
          want = ids |> String.split(" ", trim: true) |> Enum.map(&String.to_integer/1)
          got = Tokenizer.encode(tk, text)
          got == want and hf_decode_equal?(Tokenizer.decode(tk, got), Base.decode16!(hex, case: :lower))
        end)

      assert mism == [], "#{file}: #{length(mism)} mismatches, first #{inspect(hd(mism ++ [nil]))}"
    end
  end

  # HF decodes to a string (invalid UTF-8 → U+FFFD); vapor returns bytes
  defp hf_decode_equal?(bytes, hf), do: bytes == hf or scrub(bytes) == hf
  defp scrub(b), do: b |> String.chunk(:valid) |> Enum.map_join(&if(String.valid?(&1), do: &1, else: String.duplicate("�", byte_size(&1))))

  # ------------------------------------------------------------- corpus --

  @words ~w(Hello world the The tokenizer Việt Cửa Äpfel нещо Български 日本語 中文字符 🦙 😶‍🌫️ 👨‍👩‍👧 é ﬁ Ⅻ ½ ١٢٣ ३४ ǅ İstanbul straße ΣΊΣΥΦΟΣ ❤️ 🇧🇷 ‍ 가나다)
  @seps [" ", "  ", "   ", "\t", "\n", "\n\n", "\r\n", " \n ", " ", "　", " ", "\u0085", "", "'s", "'LL", "'d",
         "...", "!!!", "(", ")", "-", "—", "\"", "```", "é", "​", "  \t\n"]
  @specials ["<|endoftext|>", "<s>", "</s>", "<|begin_of_text|>", "<|im_start|>", "<unk>"]

  @doc false
  def corpus(n, seed) do
    :rand.seed(:exsss, {seed, 17, 29})
    fixed = ["", " ", "   ", "\n", "Hello", " Hello", "Hello   ", "3333333", "1234567890123", "  leading and trailing  ",
             String.duplicate("a", 3000), String.duplicate("日本", 700), String.duplicate(" ab", 900), " x", "x\u0085y"]
    fixed ++ for(_ <- 1..n, do: random_text())
  end

  defp random_text do
    for _ <- 1..:rand.uniform(30), into: "" do
      case :rand.uniform(10) do
        n when n <= 4 -> Enum.random(@words)
        n when n <= 7 -> Enum.random(@seps)
        8 -> Integer.to_string(:rand.uniform(1 <<< (:rand.uniform(40))))
        9 -> if :rand.uniform(4) == 1, do: Enum.random(@specials), else: random_chars()
        10 -> random_chars()
      end
    end
  end

  defp random_chars do
    ranges = [0x20..0x7E, 0xA0..0x17F, 0x300..0x36F, 0x400..0x4FF, 0x4E00..0x4E80, 0x1F600..0x1F64F, 0x0E00..0x0E5B, 0x01..0x1F]
    for _ <- 1..:rand.uniform(8), into: "", do: <<Enum.random(Enum.random(ranges))::utf8>>
  end

end
