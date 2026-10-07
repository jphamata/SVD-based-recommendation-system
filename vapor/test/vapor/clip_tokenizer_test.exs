defmodule Vapor.ClipTokenizerTest do
  @moduledoc """
  CLIP's tokenizer — lowercased, whitespace-folded, split by CLIP's own
  pattern (gaps dropped), byte-level, BPE whose word-final symbols carry
  the `</w>` suffix, wrapped in start/end-of-text — read from the
  `tokenizer.json` transformers writes and compared, id for id, with
  Hugging Face `tokenizers` (Rust). The vocabulary is rebuilt from OpenAI's
  published merges (`make fixtures`, SHA-256 pinned).
  """
  use ExUnit.Case, async: false
  import Vapor.TestHelpers

  @moduletag :clip_vocab
  @moduletag timeout: 600_000

  @lines [
    "A photo of a CAT sitting on the mat, isn't it?  Ünïcödé 123 ✓!",
    "a diagram of a neural network",
    "O'Neill's  dog   ran 42km — très vite!!!",
    "東京の夜景 and 서울 at night",
    "emoji 🐈‍⬛ test 😀😀 ok",
    "<|startoftext|>raw specials<|endoftext|> inside",
    "tabs\tand\u00A0non-breaking   spaces",
    "x",
    "I'm, you're, they've, we'll, he'd, she's, it'll",
    "   leading and trailing   ",
    "",
    "uma fotografia de um gato preto em cima do sofá",
    "Ελληνικά κείμενα και αριθμοί 2024",
    "Привет, мир! Это тест.",
    "مرحبا بالعالم",
    "x² + y² = z²; ∀ε>0 ∃δ>0",
    "www.example.com/path?query=1&b=2#frag",
    "snake_case and camelCase and SCREAMING_CASE",
    "3.14159265358979323846",
    "a1b2c3 d4e5f6"
  ]

  test "every line encodes to transformers' ids" do
    dir = Path.join(System.tmp_dir!(), "vapor-clip-tk-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)
    bpe = Path.expand("../fixtures/vocab/bpe_simple_vocab_16e6.txt.gz", __DIR__)
    out = py!(File.read!(Path.expand("../python/clip_tokenizer.py", __DIR__)), [bpe, dir], Enum.join(@lines, "\n"))
    want = out |> String.split("\n", trim: true) |> Enum.map(&Vapor.JSON.decode!/1)

    {:ok, tk} = Vapor.Tokenizer.load(Path.join(dir, "tokenizer.json"))
    assert {tk.bos, tk.eos, tk.suffix} == {49_406, 49_407, "</w>"}
    got = Enum.map(@lines, &Vapor.Tokenizer.encode(tk, &1))
    assert Enum.zip(@lines, got) |> Enum.zip(want) |> Enum.reject(fn {{_, g}, w} -> g == w end) == []

    # a word-final token decodes with its space
    assert Vapor.Tokenizer.decode(tk, Vapor.Tokenizer.encode(tk, "A photo of a cat", add_bos: false, add_eos: false)) ==
             "a photo of a cat "
  end
end
