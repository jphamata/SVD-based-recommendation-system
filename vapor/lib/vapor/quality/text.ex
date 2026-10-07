defmodule Vapor.Quality.Text do
  @moduledoc """
  Is this text, or noise? Measurements that need no model, and one that
  needs the model's own probabilities.

  Reference-based (a `profile/2` of a corpus in the target language):

    * `trigram_hit` — the fraction of the sample's byte trigrams that occur
      in the reference. Random bytes almost never hit; text with the right
      letter frequencies but no order (the unigram control) hits some;
      real text hits most. **The main noise score.**
    * `cond_compression` — extra zlib bytes per sample byte when the sample
      is appended to the reference: how predictable it is *given the
      language* (≈ 1 for random bytes).

  Self-contained:

    * `self_compression` — zlib ratio of the sample alone; very low means
      degenerate repetition (a sampler stuck in a loop);
    * `distinct2` — distinct byte bigrams over bigrams; `utf8` — validity,
      `utf8_fraction` — bytes inside valid characters (the gate needs ≥ 95 %:
      a byte-level model may cut one multibyte character, a broken
      detokeniser cuts many);
      `entropy` — bits per byte of the sample's unigram distribution.

  Model-based: `bits_per_token/2` — the cross-entropy a model assigns to
  held-out text, against the uniform (`log₂ V`) and unigram baselines. A
  model that does not beat the unigram baseline has learned no order.

  `gate/2` fits the two gates (noise, collapse) to controls drawn from the
  reference itself (`Vapor.Quality.Gate`).
  """
  alias Vapor.Quality.Gate
  alias Vapor.Modal.Rng

  defstruct [:trigrams, :unigram, :corpus, :reference]

  @doc """
  A reference profile of `corpus` (bytes). The compression reference is the
  last `ref_bytes` (default 16 KiB) of it.
  """
  def profile(corpus, opts \\ []) when is_binary(corpus) do
    bytes = :binary.bin_to_list(corpus)
    tri = bytes |> Enum.chunk_every(3, 1, :discard) |> MapSet.new()
    freq = Enum.frequencies(bytes)
    total = length(bytes)
    ref = binary_part(corpus, max(0, byte_size(corpus) - Keyword.get(opts, :ref_bytes, 16_384)), min(byte_size(corpus), Keyword.get(opts, :ref_bytes, 16_384)))
    %__MODULE__{trigrams: tri, unigram: Map.new(freq, fn {b, c} -> {b, c / total} end), corpus: corpus, reference: ref}
  end

  @doc "Every measurement of a sample."
  def metrics(sample, %__MODULE__{} = p) when is_binary(sample) do
    bytes = :binary.bin_to_list(sample)
    tris = Enum.chunk_every(bytes, 3, 1, :discard)
    bis = Enum.chunk_every(bytes, 2, 1, :discard)
    n = max(byte_size(sample), 1)

    %{
      bytes: byte_size(sample),
      utf8: String.valid?(sample),
      utf8_fraction: utf8_fraction(sample),
      trigram_hit: if(tris == [], do: 0.0, else: Enum.count(tris, &MapSet.member?(p.trigrams, &1)) / length(tris)),
      cond_compression: (z(p.reference <> sample) - z(p.reference)) / n,
      self_compression: z(sample) / n,
      distinct2: if(bis == [], do: 0.0, else: length(Enum.uniq(bis)) / length(bis)),
      entropy: entropy(bytes)
    }
  end

  # bytes inside valid UTF-8 characters over all bytes
  defp utf8_fraction(""), do: 1.0
  defp utf8_fraction(s), do: (s |> String.chunk(:valid) |> Enum.filter(&String.valid?/1) |> Enum.map(&byte_size/1) |> Enum.sum()) / byte_size(s)

  defp z(bin), do: bin |> :zlib.compress() |> byte_size()

  defp entropy([]), do: 0.0

  defp entropy(bytes) do
    n = length(bytes)
    bytes |> Enum.frequencies() |> Map.values() |> Enum.reduce(0.0, fn c, h -> p = c / n; h - p * :math.log2(p) end)
  end

  # ---------------------------------------------------------------- controls --

  @doc """
  Controls of `len` bytes, `count` of each, from `holdout` (text the
  profile has not seen): `%{negatives: %{uniform, unigram, loop}, positives: %{holdout}}`.
  """
  def controls(holdout, len, count, seed \\ 1) do
    hb = :binary.bin_to_list(holdout)
    max_start = max(length(hb) - len, 1)
    starts = Rng.uniform(seed, count) |> Enum.map(&trunc(&1 * max_start))
    windows = Enum.map(starts, fn s -> hb |> Enum.slice(s, len) |> :binary.list_to_bin() end)

    %{negatives: %{
        uniform: for(i <- 1..count, do: Rng.uniform(seed * 1000 + i, len) |> Enum.map(&trunc(&1 * 256)) |> :binary.list_to_bin()),
        unigram: for({w, i} <- Enum.with_index(windows), do: w |> :binary.bin_to_list() |> Rng.permute(seed * 7919 + i) |> :binary.list_to_bin()),
        loop: for(w <- windows, do: w |> binary_part(0, min(8, byte_size(w))) |> :binary.copy(div(len, 8) + 1) |> binary_part(0, len))
      },
      positives: %{holdout: windows}}
  end

  @doc """
  Fit the text gates on controls of `len` bytes: `{:ok, %{noise: gate,
  collapse: gate}}` — or the reason a gate cannot be trusted at this length.
  """
  def gate(%__MODULE__{} = p, holdout, opts \\ []) do
    len = Keyword.get(opts, :len, 200)
    cs = controls(holdout, len, Keyword.get(opts, :count, 24), Keyword.get(opts, :seed, 1))
    score = fn samples, key -> Enum.map(samples, &metrics(&1, p)[key]) end

    with {:ok, noise} <- Gate.calibrate(:text_noise, Map.take(cs.negatives, [:uniform, :unigram]) |> Map.new(fn {k, v} -> {k, score.(v, :trigram_hit)} end),
                                        %{holdout: score.(cs.positives.holdout, :trigram_hit)}),
         {:ok, collapse} <- Gate.calibrate(:text_collapse, %{loop: score.(cs.negatives.loop, :self_compression)},
                                           %{holdout: score.(cs.positives.holdout, :self_compression)}) do
      {:ok, %{noise: noise, collapse: collapse, len: len}}
    end
  end

  @doc """
  Judge a sample: `%{verdict, noise, collapse, metrics}` where `verdict` is
  `:pass` (structured or natural, not collapsed) or `:fail`. Samples
  shorter than the calibration length are judged on their own length only
  if at least half of it (otherwise `:too_short`).
  """
  def judge(sample, %__MODULE__{} = p, %{noise: ng, collapse: cg, len: len}) do
    m = metrics(sample, p)

    if m.bytes < div(len, 2) do
      %{verdict: :too_short, metrics: m}
    else
      noise = Gate.judge(ng, m.trigram_hit)
      collapse = Gate.judge(cg, m.self_compression)
      verdict = if Gate.signal?(noise) and Gate.signal?(collapse) and m.utf8_fraction >= 0.95, do: :pass, else: :fail
      %{verdict: verdict, noise: noise, collapse: collapse, metrics: m}
    end
  end

  # ------------------------------------------------------------ model-based --

  @doc """
  Bits per token a model assigns to `ids` (teacher forcing): `logprobs.(prefix)`
  returns the model's log-probabilities (natural log) of the next token as a
  map or list indexed by id. Returns `%{bits, uniform, tokens}`.
  """
  def bits_per_token(logprobs, ids, vocab) when length(ids) >= 2 do
    {sum, n} =
      ids
      |> Enum.with_index()
      |> Enum.drop(1)
      |> Enum.reduce({0.0, 0}, fn {id, i}, {s, n} ->
        lp = logprobs.(Enum.take(ids, i))
        {s - at(lp, id) / :math.log(2), n + 1}
      end)

    %{bits: sum / n, uniform: :math.log2(vocab), tokens: n}
  end

  defp at(lp, id) when is_list(lp), do: Enum.at(lp, id)
  defp at(lp, id) when is_map(lp), do: Map.fetch!(lp, id)
  defp at(lp, id) when is_tuple(lp), do: elem(lp, id)

  @doc "Log-softmax (natural log) of a list of logits, in binary64."
  def log_softmax(logits) do
    m = Enum.max(logits)
    lse = m + :math.log(Enum.reduce(logits, 0.0, &(&2 + :math.exp(&1 - m))))
    Enum.map(logits, &(&1 - lse))
  end

  @doc "Unigram cross-entropy (bits per byte) of `text` under the profile's byte frequencies (unseen bytes: 1/(N+256))."
  def unigram_bits(text, %__MODULE__{unigram: u}) do
    bytes = :binary.bin_to_list(text)
    floor = 1.0e-6
    Enum.reduce(bytes, 0.0, fn b, s -> s - :math.log2(Map.get(u, b, floor)) end) / max(length(bytes), 1)
  end
end
