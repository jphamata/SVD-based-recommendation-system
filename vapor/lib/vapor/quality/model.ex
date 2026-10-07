defmodule Vapor.Quality.Model do
  @moduledoc """
  The quality gate applied to **a real checkpoint**: is this model's output
  language, or noise?

  Two independent tests, both against baselines that need no model:

    1. **Bits per byte** on held-out text (teacher forcing, windows of
       `window` tokens): `Σ −log₂ p(token) / bytes`. A model that has learned
       anything about the language beats the unigram byte model of a
       reference corpus (≈ 4–5 bits/byte for prose); random weights do not
       even beat the uniform 8 bits/byte. Logits are read row by row from
       the binary, so a 150 000-token vocabulary costs no float lists.
    2. **Generations through the calibrated text gate**
       (`Vapor.Quality.Text`): `samples` continuations of prompts cut from
       the held-out text must pass (`:structured` or `:natural`, not
       collapsed).

  `verdict: :signal` needs both. The reference corpus defaults to vapor's
  own Portuguese docs; pass `text:` (and `reference:`) in the model's
  language.
  """
  alias Vapor.{Lock, Rejection, Tensor, Tokenizer}
  alias Vapor.Modal.Runner
  alias Vapor.Quality.{Suite, Text}

  @doc """
  Judge a checkpoint (`path`) or `%{spec, weights, tokenizer}`. Options:
  `text` (held-out text, default: vapor's Portuguese docs), `reference`
  (corpus for the profile and the gate), `bytes` (text scored, 1500),
  `window` (128), `samples` (3), `sample_bytes` (200), `temperature` (0.8),
  `worker`.
  """
  def judge(model, opts \\ []) do
    with {:ok, m} <- open(model),
         :ok <- need(m.spec.interface == :causal_lm, "the :causal_lm contract"),
         :ok <- need(m.tokenizer != nil, "a tokenizer (tokenizer.json or tokenizer.gguf)") do
      {ref_default, hold_default} = Suite.corpus_pt_raw()
      reference = Keyword.get(opts, :reference, ref_default)
      text = Keyword.get(opts, :text, hold_default)
      ro = [worker: Keyword.get_lazy(opts, :worker, &Runner.worker/0)]
      profile = Text.profile(reference)
      sb = Keyword.get(opts, :sample_bytes, 200)

      scored = binary_part(text, 0, min(byte_size(text), Keyword.get(opts, :bytes, 1500))) |> valid_prefix()
      bpb = bits_per_byte(m, scored, Keyword.get(opts, :window, 128), ro)
      unigram = Text.unigram_bits(scored, profile)

      gate = Text.gate(profile, text, len: sb, count: 16)

      samples =
        case gate do
          {:ok, g} ->
            for i <- 1..Keyword.get(opts, :samples, 3) do
              start = rem(i * 997, max(byte_size(text) - 64, 1))
              prompt = text |> binary_part(start, 48) |> valid_prefix()
              ids = Tokenizer.encode(m.tokenizer, prompt, add_bos: false)
              out = Vapor.Modal.Text.generate(m.spec, m.weights, ids, max(div(sb, 2), 16),
                                              Keyword.merge(ro, temperature: Keyword.get(opts, :temperature, 0.8), seed: i, max_seq: length(ids) + sb + 2))
              # ids beyond the tokenizer (a padded embedding matrix) decode to
              # nothing and count against the sample: they are not text
              known = Tokenizer.vocab_size(m.tokenizer)
              {ok_ids, bad} = Enum.split_with(out, &(&1 < known))
              s = Tokenizer.decode(m.tokenizer, ok_ids)
              j = Text.judge(s, profile, g)
              j = if bad != [] and length(bad) * 20 > length(out), do: %{j | verdict: :fail}, else: j
              j |> Map.put(:sample, s) |> Map.put(:unknown_ids, length(bad))
            end

          {:error, why} ->
            [%{verdict: :gate_unavailable, why: why}]
        end

      passed = Enum.count(samples, &(&1.verdict == :pass))
      learned = bpb < unigram

      {:ok,
       %{family: m.spec.family, bits_per_byte: bpb, unigram_bits_per_byte: unigram, uniform_bits_per_byte: 8.0,
         scored_bytes: byte_size(scored), samples: samples, samples_passed: passed,
         verdict: if(learned and passed == length(samples), do: :signal, else: :noise),
         reasons: Enum.reject([if(not learned, do: "bits/byte #{Float.round(bpb, 3)} ≥ unigram #{Float.round(unigram, 3)}: no order learned"),
                               if(passed < length(samples), do: "#{length(samples) - passed} of #{length(samples)} generations fail the noise gate")], &is_nil/1)}}
    end
  end

  defp open(path) when is_binary(path), do: Lock.open(path)
  defp open(%{spec: _, weights: _} = m), do: {:ok, Map.put_new(m, :tokenizer, nil)}

  @doc """
  Held-out **bits per byte** of `text` under a causal LM `%{spec, weights,
  tokenizer}`: Σ −log₂ p(token) over windows of `window` tokens (teacher
  forcing), divided by the bytes of the text. `ro`: `[worker: w]` or `[]`.
  Deterministic: the substrate's bits are the oracle's.
  """
  def bits_per_byte(m, text, window \\ 128, ro \\ []) do
    ids = Tokenizer.encode(m.tokenizer, text, add_bos: false)
    {:ok, p} = Lock.build(m.spec, m.weights, max_seq: window)
    v = m.spec.vocab

    bits =
      ids
      |> Enum.chunk_every(window)
      |> Enum.filter(&(length(&1) >= 2))
      |> Enum.reduce(0.0, fn win, acc ->
        n = length(win)
        env = Map.merge(Lock.zero_state(p), %{tok: Tensor.from_list(:s32, [n], win), pos: Tensor.from_list(:s32, [n], Enum.to_list(0..(n - 1)))})
        logits = Runner.run(p, env, ro).logits.data

        win
        |> Enum.drop(1)
        |> Enum.with_index()
        |> Enum.reduce(acc, fn {next, row}, acc -> acc - log2_softmax_at(binary_part(logits, row * v * 4, v * 4), next) end)
      end)

    bits / max(byte_size(text), 1)
  end

  defp log2_softmax_at(row, j) do
    {mx, _} = for(<<x::float-32-little <- row>>, reduce: {-1.0e300, 0}, do: ({m, i} -> {max(m, x), i + 1}))
    lse = mx + :math.log(for(<<x::float-32-little <- row>>, reduce: 0.0, do: (s -> s + :math.exp(x - mx))))
    <<_::binary-size(j * 4), x::float-32-little, _::binary>> = row
    (x - lse) / :math.log(2)
  end

  defp valid_prefix(bin) do
    if String.valid?(bin), do: bin, else: valid_prefix(binary_part(bin, 0, byte_size(bin) - 1))
  end

  defp need(true, _), do: :ok
  defp need(false, b), do: {:error, Rejection.new(:quality_model, b, "judge a causal LM with its tokenizer")}
end
