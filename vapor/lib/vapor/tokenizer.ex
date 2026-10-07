defmodule Vapor.Tokenizer do
  @moduledoc """
  Text ⇄ token ids for the model families vapor runs, in pure Elixir.

  One algorithm covers both vocabularies in use:

    * **byte-level BPE** (GPT-2, Llama 3, Qwen2): the text is split by a
      regular expression, each piece starts as its bytes, and ranked merges
      are applied;
    * **SentencePiece BPE** (Llama 2, Mistral v1): spaces become `▁`, a
      `▁` may be prepended, each piece starts as its characters, and pairs
      merge by vocabulary score (from a GGUF) or by rank (from a
      `tokenizer.json`); characters without a token fall back to `<0xXX>`
      byte tokens.

  In both, a word is merged by the same priority procedure — repeatedly
  merge the best-priority adjacent pair, the leftmost among equals — in
  `O(n log n)` with a heap and a doubly linked symbol list, so a long
  unsplit text (SentencePiece has no pre-tokenizer) costs no more than its
  length warrants.

  Tokens are stored by their *surface bytes*: byte-level tokens un-mapped
  from GPT-2's printable alias alphabet, SentencePiece pieces with `▁` as a
  space and byte tokens as the byte. Decoding is then concatenation (minus
  the one prefix space SentencePiece added), and it returns bytes — a
  partial UTF-8 sequence at the end of a stream stays intact for the next
  token.

  Sources: `from_gguf/1` (llama.cpp metadata; the pre-tokenizer regex is
  chosen by `tokenizer.ggml.pre`, as llama.cpp does) and `from_hf/1`
  (`tokenizer.json`: the normalizers, pre-tokenizers, models and decoders
  these families use; anything else is a rejection naming the component).
  Special tokens are matched in the raw text first, leftmost-longest, as
  Hugging Face does.
  """
  alias Vapor.Rejection

  defstruct kind: nil, vocab: %{}, surface: {}, ranks: nil, scores: nil, byte_fallback: false,
            ignore_merges: false, normalize: [], pre: [], special: [], special_re: nil,
            strip_prefix: false, bos: nil, eos: nil, unk: nil, add_bos: false, byte_ids: nil,
            # end-of-word suffix of the BPE model (CLIP: "</w>" marks a word's
            # last symbol); add_eos: the post-processor appends EOS (CLIP)
            suffix: nil, add_eos: false

  @type t :: %__MODULE__{}

  # ------------------------------------------------------------ patterns --

  @gpt2 "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
  @llama3 "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
  @qwen2 "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

  @doc "The pre-tokenizer patterns llama.cpp associates with a `tokenizer.ggml.pre` name."
  def gguf_pre(p) when p in ["llama-bpe", "llama3", "llama-v3", "smaug-bpe"], do: {:ok, [@llama3]}
  def gguf_pre("qwen2"), do: {:ok, [@qwen2]}
  def gguf_pre(p) when p in ["gpt-2", "default", nil], do: {:ok, [@gpt2]}
  def gguf_pre(p), do: {:error, p}

  # -------------------------------------------------------------- encode --

  @doc """
  Encode text. Options: `:add_bos` (default: the tokenizer's own setting),
  `:special` (`true`: special-token strings in the text become their ids,
  as Hugging Face does; `false`: they are ordinary text).
  """
  @spec encode(t, binary, keyword) :: [non_neg_integer]
  def encode(%__MODULE__{} = tk, text, opts \\ []) when is_binary(text) do
    bos = if Keyword.get(opts, :add_bos, tk.add_bos) and tk.bos, do: [tk.bos], else: []

    parts = if Keyword.get(opts, :special, true), do: split_special(tk, text), else: [{:text, text}]

    {ids, _cache} =
      parts
      |> Enum.with_index()
      |> Enum.flat_map_reduce(%{}, fn
        {{:special, id}, _}, cache -> {[id], cache}
        {{:text, s}, i}, cache -> encode_text(tk, s, i == 0, cache)
      end)

    eos = if Keyword.get(opts, :add_eos, tk.add_eos) and tk.eos, do: [tk.eos], else: []
    bos ++ ids ++ eos
  end

  defp encode_text(tk, text, first?, cache) do
    text
    |> normalize(tk.normalize, first?)
    |> pre_tokenize(tk.pre, first?)
    |> Enum.flat_map_reduce(cache, fn piece, cache ->
      case cache do
        %{^piece => ids} -> {ids, cache}
        _ -> ids = word(tk, piece); {ids, Map.put(cache, piece, ids)}
      end
    end)
  end

  # special tokens: leftmost, and longest at a position (one compiled
  # alternation ordered by decreasing length)
  defp split_special(%{special_re: nil}, text), do: [{:text, text}]

  defp split_special(tk, text) do
    case :re.run(text, tk.special_re, [:global, capture: :first]) do
      :nomatch ->
        [{:text, text}]

      {:match, ms} ->
        {parts, at} =
          Enum.flat_map_reduce(ms, 0, fn [{s, l}], at ->
            content = binary_part(text, s, l)
            {id, lstrip, rstrip} = List.keyfind(tk.special, content, 0) |> then(fn {_, id, ls, rs} -> {id, ls, rs} end)
            gap = binary_part(text, at, s - at)
            gap = if lstrip, do: String.trim_trailing(gap), else: gap
            {[{:text, gap}, {:special, id, rstrip}], s + l}
          end)

        tail = binary_part(text, at, byte_size(text) - at)

        (parts ++ [{:text, tail}])
        |> rstrip_after_special()
        |> Enum.reject(&(&1 == {:text, ""}))
    end
  end

  defp rstrip_after_special([{:special, id, true}, {:text, t} | rest]),
    do: [{:special, id} | rstrip_after_special([{:text, String.trim_leading(t)} | rest])]

  defp rstrip_after_special([{:special, id, _} | rest]), do: [{:special, id} | rstrip_after_special(rest)]
  defp rstrip_after_special([x | rest]), do: [x | rstrip_after_special(rest)]
  defp rstrip_after_special([]), do: []

  # ---------------------------------------------------------- normalizer --

  defp normalize(text, steps, first?), do: Enum.reduce(steps, text, &norm(&1, &2, first?))

  defp norm(:nfc, t, _), do: Vapor.Unicode.nfc(t)
  defp norm(:nfkc, t, _), do: Vapor.Unicode.nfkc(t)
  defp norm(:lowercase, t, _), do: String.downcase(t)
  defp norm({:prepend, _p}, "", _), do: ""
  defp norm({:prepend, p}, t, _), do: p <> t
  defp norm({:replace, from, to}, t, _) when is_binary(from), do: String.replace(t, from, to)
  defp norm({:replace, re, to}, t, _), do: :re.replace(t, re, to, [:global, {:return, :binary}])

  # ------------------------------------------------------- pre-tokenizer --

  defp pre_tokenize(text, steps, first?),
    do: Enum.reduce(steps, [text], fn step, pieces -> Enum.flat_map(pieces, &pre(step, &1, first?)) end)

  # "Isolated": matches and the gaps between them are pieces
  defp pre({:split, re}, text, _) do
    case :re.run(text, re, [:global, capture: :first]) do
      :nomatch ->
        [text]

      {:match, ms} ->
        {pieces, at} =
          Enum.flat_map_reduce(ms, 0, fn [{s, l}], at ->
            {[binary_part(text, at, s - at), binary_part(text, s, l)], s + l}
          end)

        Enum.reject(pieces ++ [binary_part(text, at, byte_size(text) - at)], &(&1 == ""))
    end
  end

  # "Removed" with invert: the matches are the pieces, the gaps are dropped
  defp pre({:keep_matches, re}, text, _) do
    case :re.run(text, re, [:global, capture: :first]) do
      :nomatch -> []
      {:match, ms} -> for [{s, l}] <- ms, l > 0, do: binary_part(text, s, l)
    end
  end

  defp pre({:prefix_space, :always}, <<?\s, _::binary>> = t, _), do: [t]
  defp pre({:prefix_space, :always}, "", _), do: [""]
  defp pre({:prefix_space, :always}, t, _), do: [" " <> t]

  # Metaspace: spaces become ▁ and every piece starts at a ▁
  defp pre({:metaspace, rep, scheme, split}, t, first?) do
    t = String.replace(t, " ", rep)

    t =
      if (scheme == :always or (scheme == :first and first?)) and not String.starts_with?(t, rep) and t != "",
        do: rep <> t,
        else: t

    if split do
      t
      |> String.split(rep)
      |> Enum.with_index()
      |> Enum.flat_map(fn
        {"", 0} -> []
        {p, 0} -> [p]
        {p, _} -> [rep <> p]
      end)
    else
      [t]
    end
  end

  defp pre(:digits, t, _) do
    Regex.split(~r/\p{N}/u, t, include_captures: true, trim: true)
  end

  # --------------------------------------------------------------- words --

  # one pre-tokenized piece → ids
  defp word(tk, piece) do
    case tk.ignore_merges && Map.fetch(tk.vocab, piece) do
      {:ok, id} -> [id]
      _ -> piece |> symbols(tk.kind) |> suffixed(tk.suffix) |> merge(tk) |> Enum.flat_map(&to_ids(tk, &1))
    end
  end

  defp symbols(piece, :bytes), do: for(<<b <- piece>>, do: <<b>>)
  defp symbols(piece, :chars), do: String.codepoints(piece)

  # the word's last symbol carries the end-of-word suffix (CLIP's "</w>")
  defp suffixed(syms, nil), do: syms
  defp suffixed([], _), do: []
  defp suffixed(syms, suffix), do: List.update_at(syms, -1, &(&1 <> suffix))

  defp to_ids(tk, sym) do
    case tk.vocab do
      %{^sym => id} -> [id]
      _ when tk.byte_fallback -> for <<b <- sym>>, do: elem(tk.byte_ids, b)
      _ when tk.unk != nil -> [tk.unk]
      _ -> raise ArgumentError, "no token for #{inspect(sym)} and no byte fallback or <unk>"
    end
  end

  # The merge procedure. Symbols live in a map i → {bin, prev, next}; the
  # heap holds {priority, i, left, right} candidates, checked when popped
  # (a merge may have consumed either side since).
  defp merge([_] = syms, _tk), do: syms
  defp merge([], _tk), do: []

  defp merge(syms, tk) do
    n = length(syms)
    tup = List.to_tuple(syms)
    nodes = Map.new(0..(n - 1), fn i -> {i, {elem(tup, i), i - 1, if(i + 1 < n, do: i + 1, else: -1)}} end)

    heap =
      Enum.reduce(0..(n - 2), :gb_sets.empty(), fn i, h ->
        push(h, tk, i, elem(tup, i), elem(tup, i + 1))
      end)

    nodes |> run(heap, tk) |> collect(0)
  end

  defp push(h, tk, i, a, b) do
    case prio(tk, a, b) do
      nil -> h
      p -> :gb_sets.add({p, i, a, b}, h)
    end
  end

  defp prio(%{ranks: ranks}, a, b) when ranks != nil, do: Map.get(ranks, {a, b})

  defp prio(%{scores: scores, vocab: v}, a, b) do
    case Map.fetch(v, a <> b) do
      {:ok, id} -> -elem(scores, id)
      :error -> nil
    end
  end

  defp run(nodes, heap, tk) do
    if :gb_sets.is_empty(heap) do
      nodes
    else
      {{_p, i, a, b}, heap} = :gb_sets.take_smallest(heap)

      case nodes do
        %{^i => {^a, prev, j}} when j >= 0 ->
          case nodes do
            %{^j => {^b, _, next}} ->
              ab = a <> b
              nodes = nodes |> Map.put(i, {ab, prev, next}) |> Map.delete(j)
              nodes = if next >= 0, do: Map.update!(nodes, next, fn {s, _, nn} -> {s, i, nn} end), else: nodes
              heap = if prev >= 0, do: push(heap, tk, prev, elem(nodes[prev], 0), ab), else: heap
              heap = if next >= 0, do: push(heap, tk, i, ab, elem(nodes[next], 0)), else: heap
              run(nodes, heap, tk)

            _ ->
              run(nodes, heap, tk)
          end

        _ ->
          run(nodes, heap, tk)
      end
    end
  end

  defp collect(_nodes, -1), do: []
  defp collect(nodes, i), do: [elem(nodes[i], 0) | collect(nodes, elem(nodes[i], 2))]

  # -------------------------------------------------------------- decode --

  @doc """
  Token ids → bytes (surface forms concatenated; SentencePiece's added
  prefix space removed). Unknown ids are an `ArgumentError`.
  """
  @spec decode(t, [non_neg_integer]) :: binary
  def decode(%__MODULE__{} = tk, ids) do
    out = IO.iodata_to_binary(for id <- ids, do: surface(tk, id))
    if tk.strip_prefix, do: strip_one_space(out), else: out
  end

  @doc "Surface bytes of one token (for streaming)."
  def surface(%__MODULE__{surface: s}, id) when is_integer(id) and id >= 0 and id < tuple_size(s), do: elem(s, id)
  def surface(_tk, id), do: raise(ArgumentError, "token id #{inspect(id)} out of range")

  defp strip_one_space(<<?\s, rest::binary>>), do: rest
  defp strip_one_space(b), do: b

  def vocab_size(%__MODULE__{surface: s}), do: tuple_size(s)

  # ------------------------------------------------------ GPT-2 byte map --

  @printable Enum.to_list(?!..?~) ++ Enum.to_list(0xA1..0xAC) ++ Enum.to_list(0xAE..0xFF)

  @byte_to_char (fn ->
                   {extra, _} =
                     Enum.map_reduce(Enum.reject(0..255, &(&1 in @printable)), 0, fn b, n -> {{b, 256 + n}, n + 1} end)

                   Map.new(Enum.map(@printable, &{&1, &1}) ++ extra)
                 end).()
  @char_to_byte Map.new(@byte_to_char, fn {b, c} -> {c, b} end)

  @doc "GPT-2 alias string → raw bytes, or `:error` when a character has no byte."
  def unalias(s) do
    s
    |> String.to_charlist()
    |> Enum.reduce_while([], fn c, acc ->
      case @char_to_byte do
        %{^c => b} -> {:cont, [b | acc]}
        _ -> {:halt, :error}
      end
    end)
    |> case do
      :error -> :error
      acc -> acc |> Enum.reverse() |> :binary.list_to_bin()
    end
  end

  # ------------------------------------------------------------ builders --

  @doc "From GGUF metadata (`Vapor.Ingest.GGUF.read/1`)."
  @spec from_gguf(map) :: {:ok, t} | {:error, Rejection.t()}
  def from_gguf(%{"tokenizer.ggml.model" => model, "tokenizer.ggml.tokens" => tokens} = m) when is_list(tokens) do
    types = m["tokenizer.ggml.token_type"] || List.duplicate(1, length(tokens))
    base = %{bos: m["tokenizer.ggml.bos_token_id"], eos: m["tokenizer.ggml.eos_token_id"],
             unk: m["tokenizer.ggml.unknown_token_id"]}

    case model do
      "gpt2" ->
        with {:ok, pats} <- gguf_pre(m["tokenizer.ggml.pre"]) |> named("tokenizer.ggml.pre"),
             {:ok, merges} <- gguf_merges(m["tokenizer.ggml.merges"]) do
          surface = Enum.zip_with(tokens, types, fn t, ty -> if ty in [3, 4], do: t, else: unalias_or(t) end)

          # Llama 3 emits a whole piece that is itself a token without merging
          # (tokenizer.json `ignore_merges`; llama.cpp sets it for these pre
          # names — not for `smaug-bpe`, which shares only the pattern)
          build(:bytes, surface, types, Map.merge(base, %{
            ranks: merges, pre: Enum.map(pats, &{:split, compile!(&1)}),
            ignore_merges: m["tokenizer.ggml.pre"] in ["llama-bpe", "llama3", "llama-v3"],
            add_bos: Map.get(m, "tokenizer.ggml.add_bos_token", true)}))
        end

      "llama" ->
        scores = m["tokenizer.ggml.scores"] || List.duplicate(0.0, length(tokens))
        prefix = Map.get(m, "tokenizer.ggml.add_space_prefix", true)
        surface = Enum.zip_with(tokens, types, fn t, ty -> spm_surface(t, ty) end)

        build(:chars, tokens, types, Map.merge(base, %{
          surface_override: surface, scores: List.to_tuple(scores), byte_fallback: true,
          # llama.cpp prefixes every text fragment (the first, and any after a
          # special token) — as Hugging Face's Prepend normalizer does
          normalize: [{:replace, " ", "▁"}] ++ if(prefix, do: [{:prepend, "▁"}], else: []),
          strip_prefix: prefix, add_bos: Map.get(m, "tokenizer.ggml.add_bos_token", true)}))

      other ->
        {:error, Rejection.new({:tokenizer, "tokenizer.ggml.model"}, "gpt2 or llama (got #{inspect(other)})", "use a BPE or SentencePiece vocabulary")}
    end
  end

  def from_gguf(_), do: {:error, Rejection.new({:tokenizer, :gguf}, "tokenizer.ggml.model and tokenizer.ggml.tokens", "use a GGUF with a vocabulary")}

  defp spm_surface(t, 6), do: byte_token(t) || t
  defp spm_surface(t, _), do: String.replace(t, "▁", " ")

  defp byte_token(<<"<0x", h::binary-2, ">">>) do
    case Integer.parse(h, 16) do
      {b, ""} -> <<b>>
      _ -> nil
    end
  end

  defp byte_token(_), do: nil

  defp unalias_or(t), do: (case unalias(t) do :error -> t; b -> b end)

  defp gguf_merges(nil), do: {:error, Rejection.new({:tokenizer, "tokenizer.ggml.merges"}, "present for a gpt2 vocabulary", "use a complete GGUF")}

  defp gguf_merges(list) do
    ranks =
      list
      |> Enum.with_index()
      |> Map.new(fn {m, i} ->
        [a, b] = String.split(m, " ", parts: 2)
        {{unalias_or(a), unalias_or(b)}, i}
      end)

    {:ok, ranks}
  end

  # common construction: vocab by surface (bytes) or by piece (chars)
  defp build(kind, keys, types, o) do
    surface = Map.get(o, :surface_override, keys)
    specials = for {t, id, ty} <- Enum.zip([keys, 0..(length(keys) - 1), types]), ty in [3, 4], do: {t, id, false, false}

    vocab =
      keys
      |> Enum.with_index()
      |> Enum.zip(types)
      |> Enum.reduce(%{}, fn {{k, id}, ty}, acc ->
        # control tokens never arise from merges; first id wins on duplicates
        if ty == 3, do: acc, else: Map.put_new(acc, k, id)
      end)

    byte_ids =
      if Map.get(o, :byte_fallback) do
        idx = Map.new(Enum.with_index(keys))
        List.to_tuple(for b <- 0..255, do: Map.get(idx, "<0x" <> String.pad_leading(Integer.to_string(b, 16), 2, "0") <> ">"))
      end

    if byte_ids && Enum.any?(Tuple.to_list(byte_ids), &is_nil/1) do
      {:error, Rejection.new({:tokenizer, :byte_fallback}, "all 256 <0xXX> tokens", "use a complete vocabulary")}
    else
      {:ok,
       %__MODULE__{
         kind: kind, vocab: vocab, surface: List.to_tuple(surface), ranks: o[:ranks], scores: o[:scores],
         byte_fallback: Map.get(o, :byte_fallback, false), ignore_merges: Map.get(o, :ignore_merges, false),
         normalize: Map.get(o, :normalize, []), pre: Map.get(o, :pre, []), strip_prefix: Map.get(o, :strip_prefix, false),
         bos: o[:bos], eos: o[:eos], unk: o[:unk], add_bos: o[:add_bos] || false, byte_ids: byte_ids,
         suffix: o[:suffix], add_eos: o[:add_eos] || false
       }
       |> with_special(Map.get(o, :special, specials))}
    end
  end

  defp with_special(tk, []), do: tk

  defp with_special(tk, specials) do
    sorted = Enum.sort_by(specials, fn {c, _, _, _} -> -byte_size(c) end)
    re = sorted |> Enum.map_join("|", fn {c, _, _, _} -> escape(c) end) |> compile!()
    %{tk | special: sorted, special_re: re}
  end

  defp escape(s), do: Regex.escape(s)

  defp compile!(pat) do
    {:ok, re} = :re.compile(pat, [:unicode, :ucp])
    re
  end

  defp named({:ok, _} = ok, _field), do: ok

  defp named({:error, v}, field),
    do: {:error, Rejection.new({:tokenizer, field}, "a known pre-tokenizer (got #{inspect(v)})", "add its pattern to Vapor.Tokenizer")}

  # ------------------------------------------------------ tokenizer.json --

  @doc "From a parsed Hugging Face `tokenizer.json`."
  @spec from_hf(map) :: {:ok, t} | {:error, Rejection.t()}
  def from_hf(%{"model" => %{"type" => "BPE"} = model} = j) do
    with :ok <- hf_model_ok(model),
         {:ok, norm} <- hf_normalizer(j["normalizer"]),
         {:ok, {pre, byte_level?}} <- hf_pre(j["pre_tokenizer"]),
         {:ok, dec} <- hf_decoder(j["decoder"]) do
      byte_level? = byte_level? or dec == :byte_level
      vocab = model["vocab"]
      n = vocab |> Map.values() |> Enum.max(fn -> -1 end)
      added = j["added_tokens"] || []
      n = Enum.max([n | Enum.map(added, & &1["id"])]) + 1
      by_id = Map.new(vocab, fn {t, id} -> {id, t} end) |> Map.merge(Map.new(added, &{&1["id"], &1["content"]}))
      added_ids = MapSet.new(added, & &1["id"])
      added? = &MapSet.member?(added_ids, &1)

      keys =
        for id <- 0..(n - 1) do
          t = Map.get(by_id, id, "")
          if byte_level? and not added?.(id), do: unalias_or(t), else: t
        end

      suffix = if model["end_of_word_suffix"] in [nil, ""], do: nil, else: model["end_of_word_suffix"]

      surface =
        for {k, id} <- Enum.with_index(keys) do
          cond do
            # a word-final token decodes with the space its suffix stands for
            suffix && not added?.(id) && String.ends_with?(k, suffix) -> String.replace_suffix(k, suffix, " ")
            added?.(id) or byte_level? -> k
            model["byte_fallback"] == true and byte_token(k) -> byte_token(k)
            true -> String.replace(k, "▁", " ")
          end
        end

      merges = model["merges"] || []
      ranks =
        merges
        |> Enum.with_index()
        |> Map.new(fn {m, i} ->
          [a, b] = if is_list(m), do: m, else: String.split(m, " ", parts: 2)
          if byte_level?, do: {{unalias_or(a), unalias_or(b)}, i}, else: {{a, b}, i}
        end)

      specials = for a <- added, do: {a["content"], a["id"], a["lstrip"] == true, a["rstrip"] == true}
      {bos, add_bos} = hf_bos(j["post_processor"], vocab, added)
      {eos, add_eos} = hf_eos(j["post_processor"], vocab, added)
      unk = model["unk_token"] && (vocab[model["unk_token"]] || Enum.find_value(added, &(&1["content"] == model["unk_token"] && &1["id"])))

      types = for id <- 0..(n - 1), do: if(added?.(id), do: 3, else: 1)

      build(if(byte_level?, do: :bytes, else: :chars), keys, types, %{
        surface_override: surface, ranks: ranks, byte_fallback: model["byte_fallback"] == true,
        ignore_merges: model["ignore_merges"] == true, normalize: norm, pre: pre,
        strip_prefix: dec == {:spm, true}, bos: bos, add_bos: add_bos, unk: unk, special: specials,
        eos: eos, add_eos: add_eos, suffix: suffix})
    end
  end

  def from_hf(%{"model" => %{"type" => t}}),
    do: {:error, Rejection.new({:tokenizer, "model.type"}, "BPE (got #{inspect(t)})", "WordPiece/Unigram are not supported")}

  defp hf_model_ok(m) do
    bad =
      Enum.find([{"dropout", nil}, {"continuing_subword_prefix", nil}], fn {k, ok} ->
        m[k] not in [ok, ""]
      end)

    if bad, do: {:error, Rejection.new({:tokenizer, "model." <> elem(bad, 0)}, "null", "unsupported BPE option")}, else: :ok
  end

  defp hf_normalizer(nil), do: {:ok, []}
  defp hf_normalizer(%{"type" => "Sequence", "normalizers" => ns}), do: all(ns, &hf_normalizer/1)
  defp hf_normalizer(%{"type" => "NFC"}), do: {:ok, [:nfc]}
  defp hf_normalizer(%{"type" => "NFKC"}), do: {:ok, [:nfkc]}
  defp hf_normalizer(%{"type" => "Lowercase"}), do: {:ok, [:lowercase]}
  defp hf_normalizer(%{"type" => "Prepend", "prepend" => p}), do: {:ok, [{:prepend, p}]}
  defp hf_normalizer(%{"type" => "Replace", "pattern" => %{"String" => s}, "content" => c}), do: {:ok, [{:replace, s, c}]}

  defp hf_normalizer(%{"type" => "Replace", "pattern" => %{"Regex" => r}, "content" => c}) do
    case :re.compile(r, [:unicode, :ucp]) do
      {:ok, re} -> {:ok, [{:replace, re, c}]}
      {:error, why} -> {:error, Rejection.new({:tokenizer, "normalizer.Replace"}, "a PCRE-compatible pattern (#{inspect(why)})", "report the pattern")}
    end
  end
  defp hf_normalizer(%{"type" => t}), do: unsupported("normalizer", t)

  defp hf_pre(nil), do: {:ok, {[], false}}

  defp hf_pre(%{"type" => "Sequence", "pretokenizers" => ps}) do
    Enum.reduce_while(ps, {:ok, {[], false}}, fn p, {:ok, {acc, bl}} ->
      case hf_pre(p) do
        {:ok, {steps, b}} -> {:cont, {:ok, {acc ++ steps, bl or b}}}
        err -> {:halt, err}
      end
    end)
  end

  defp hf_pre(%{"type" => "Split", "pattern" => pat, "behavior" => "Isolated", "invert" => false}) do
    re = case pat do
      %{"Regex" => r} -> r
      %{"String" => s} -> Regex.escape(s)
    end

    case :re.compile(re, [:unicode, :ucp]) do
      {:ok, c} -> {:ok, {[{:split, c}], false}}
      {:error, why} -> {:error, Rejection.new({:tokenizer, "pre_tokenizer.Split"}, "a PCRE-compatible pattern (#{inspect(why)})", "report the pattern")}
    end
  end

  defp hf_pre(%{"type" => "Split", "pattern" => %{"Regex" => re}, "behavior" => "Removed", "invert" => true}) do
    case :re.compile(re, [:unicode, :ucp]) do
      {:ok, c} -> {:ok, {[{:keep_matches, c}], false}}
      {:error, why} -> {:error, Rejection.new({:tokenizer, "pre_tokenizer.Split"}, "a PCRE-compatible pattern (#{inspect(why)})", "report the pattern")}
    end
  end

  defp hf_pre(%{"type" => "ByteLevel"} = p) do
    steps = if p["add_prefix_space"] == true, do: [{:prefix_space, :always}], else: []
    steps = if p["use_regex"] != false, do: steps ++ [{:split, compile!(@gpt2)}], else: steps
    {:ok, {steps, true}}
  end

  defp hf_pre(%{"type" => "Metaspace"} = p) do
    scheme = case p["prepend_scheme"] do
      "first" -> :first
      "never" -> :never
      _ -> if p["add_prefix_space"] == false, do: :never, else: :always
    end

    {:ok, {[{:metaspace, p["replacement"] || "▁", scheme, p["split"] != false}], false}}
  end

  defp hf_pre(%{"type" => "Digits", "individual_digits" => true}), do: {:ok, {[:digits], false}}
  defp hf_pre(%{"type" => t}), do: unsupported("pre_tokenizer", t)

  defp hf_decoder(nil), do: {:ok, :plain}
  defp hf_decoder(%{"type" => "ByteLevel"}), do: {:ok, :byte_level}
  # the suffix is turned into a space in the surfaces (see from_hf/1)
  defp hf_decoder(%{"type" => "BPEDecoder"}), do: {:ok, :plain}
  defp hf_decoder(%{"type" => "Metaspace"} = d), do: {:ok, {:spm, d["prepend_scheme"] != "never" and d["add_prefix_space"] != false}}

  defp hf_decoder(%{"type" => "Sequence", "decoders" => ds}) do
    # the SentencePiece sequence: Replace(▁, " "), ByteFallback, Fuse, Strip(" ", 1, 0)
    types = Enum.map(ds, & &1["type"])

    cond do
      "ByteLevel" in types -> {:ok, :byte_level}
      "Replace" in types -> {:ok, {:spm, Enum.any?(ds, &(&1["type"] == "Strip" and &1["start"] == 1))}}
      true -> unsupported("decoder", inspect(types))
    end
  end

  defp hf_decoder(%{"type" => t}), do: unsupported("decoder", t)

  # BOS from a TemplateProcessing post-processor: single = [SpecialToken, $A]
  defp hf_bos(%{"type" => "TemplateProcessing", "single" => [%{"SpecialToken" => %{"id" => tok}} | _]}, vocab, added) do
    id = vocab[tok] || Enum.find_value(added, &(&1["content"] == tok && &1["id"]))
    {id, id != nil}
  end

  defp hf_bos(%{"type" => "Sequence", "processors" => ps}, vocab, added) do
    Enum.find_value(ps, {nil, false}, fn p ->
      case hf_bos(p, vocab, added) do
        {_, true} = found -> found
        _ -> nil
      end
    end)
  end

  # BERT/RoBERTa-style: [CLS] $A [SEP] (CLIP: <|startoftext|> … <|endoftext|>)
  defp hf_bos(%{"type" => "RobertaProcessing", "cls" => [_, id]}, _, _) when is_integer(id), do: {id, true}
  defp hf_bos(%{"type" => "BertProcessing", "cls" => [_, id]}, _, _) when is_integer(id), do: {id, true}
  defp hf_bos(_, _, _), do: {nil, false}

  # EOS appended by the post-processor: Roberta/Bert's sep, or a template
  # single = [$A, SpecialToken] / [SpecialToken, $A, SpecialToken]
  defp hf_eos(%{"type" => t, "sep" => [_, id]}, _, _) when t in ["RobertaProcessing", "BertProcessing"] and is_integer(id), do: {id, true}

  defp hf_eos(%{"type" => "TemplateProcessing", "single" => single}, vocab, added) when is_list(single) and length(single) >= 2 do
    case List.last(single) do
      %{"SpecialToken" => %{"id" => tok}} ->
        id = vocab[tok] || Enum.find_value(added, &(&1["content"] == tok && &1["id"]))
        {id, id != nil}

      _ ->
        {nil, false}
    end
  end

  defp hf_eos(%{"type" => "Sequence", "processors" => ps}, vocab, added) do
    Enum.find_value(ps, {nil, false}, fn p ->
      case hf_eos(p, vocab, added) do
        {_, true} = found -> found
        _ -> nil
      end
    end)
  end

  defp hf_eos(_, _, _), do: {nil, false}

  defp all(xs, f) do
    Enum.reduce_while(xs, {:ok, []}, fn x, {:ok, acc} ->
      case f.(x) do
        {:ok, v} -> {:cont, {:ok, acc ++ v}}
        err -> {:halt, err}
      end
    end)
  end

  defp unsupported(what, t),
    do: {:error, Rejection.new({:tokenizer, what}, "a supported #{what} (got #{inspect(t)})", "extend Vapor.Tokenizer")}

  @doc "Load a `tokenizer.json` file."
  def load(path) do
    with {:ok, bin} <- File.read(path),
         {:ok, j} <- Vapor.JSON.decode(bin) do
      from_hf(j)
    else
      {:error, %Rejection{}} = e -> e
      {:error, why} -> {:error, Rejection.new({:tokenizer, path}, "readable JSON (#{inspect(why)})", "check the file")}
    end
  end
end
