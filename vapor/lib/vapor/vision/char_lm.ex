defmodule Vapor.Vision.CharLM do
  @moduledoc """
  A **character n-gram language model** for reading: what makes "rn" read
  as "m" unlikely in *"modern"* and "0" unlikely between two letters.

  Interpolated Witten–Bell smoothing (Witten & Bell 1991; the estimator
  character-level OCR and speech systems have used for decades because it
  needs no tuned discount):

      P(c | h) = (C(h·c) + T(h) · P(c | h⁻)) / (C(h) + T(h))

  where `h⁻` drops the oldest character of the history, `T(h)` is the
  number of distinct characters seen after `h`, and the recursion ends in
  the uniform distribution over the alphabet — so every character keeps a
  non-zero probability: the model can make a reading *more or less likely*,
  never impossible. That is the property that keeps it from overwriting
  what the page says with what is common (`test/vapor/reading_test.exs`,
  and the random-strings control of `mix vapor.quality`).

  The model *is* its corpus: `build/2` counts it, deterministically, and
  nothing else is stored. vapor's reader uses the reference corpora
  (`priv/ocr/lm.json` names them) — the same text the reader was trained
  on, never the held-out text it is measured on — **as printed**: the
  corpora are Markdown, and a model counted with its backticks and
  asterisks wrote backticks into scanned pages that had none (measured,
  docs/OCR.md §3b), so the markup is deleted before counting.
  """

  defstruct order: 5, alphabet: 0, table: %{}, chars: 0, source: nil

  @doc """
  Count `text` into an order-`order` model over `alphabet` (a list of
  characters; anything outside it is read as a space, runs of spaces
  collapse — as the OCR reader's training text was prepared).
  """
  def build(text, alphabet, order \\ 5) when order >= 1 do
    allowed = MapSet.new([" " | alphabet])

    chars =
      text
      |> String.replace(~r/\s+/u, " ")
      |> String.graphemes()
      |> Enum.map(&if(MapSet.member?(allowed, &1), do: &1, else: " "))
      |> Enum.chunk_by(&(&1 == " "))
      |> Enum.flat_map(fn [" " | _] -> [" "]; run -> run end)

    tab = :ets.new(:char_lm, [:set, :private])

    try do
      # every position, every history length 0..order-1 (histories start with a space)
      chars
      |> Enum.reduce([" "], fn c, hist ->
        Enum.each(0..(order - 1), fn k ->
          if k <= length(hist) do
            h = hist |> Enum.take(k) |> Enum.reverse() |> Enum.join()
            if :ets.update_counter(tab, {h, c}, {2, 1}, {{h, c}, 0}) == 1, do: :ets.update_counter(tab, {:types, h}, {2, 1}, {{:types, h}, 0})
            :ets.update_counter(tab, {:total, h}, {2, 1}, {{:total, h}, 0})
          end
        end)

        Enum.take([c | hist], order - 1)
      end)

      table =
        :ets.tab2list(tab)
        |> Enum.reduce(%{}, fn
          {{:types, h}, n}, acc -> Map.update(acc, h, {0, n, %{}}, fn {t, _, m} -> {t, n, m} end)
          {{:total, h}, n}, acc -> Map.update(acc, h, {n, 0, %{}}, fn {_, ty, m} -> {n, ty, m} end)
          {{h, c}, n}, acc -> Map.update(acc, h, {0, 0, %{c => n}}, fn {t, ty, m} -> {t, ty, Map.put(m, c, n)} end)
        end)

      %__MODULE__{order: order, alphabet: length(alphabet) + 1, table: table, chars: length(chars)}
    after
      :ets.delete(tab)
    end
  end

  @doc "log P(c | history) — history a string (only its last `order − 1` characters count)."
  def log_prob(%__MODULE__{} = lm, history, c) do
    hs = history |> String.graphemes() |> Enum.take(-(lm.order - 1))
    :math.log(prob(lm, hs, c))
  end

  @doc "The same, with the history as a list of characters, most recent first (the beam search keeps it so)."
  def log_prob_rev(%__MODULE__{} = lm, rev_hist, c) do
    hs = rev_hist |> Enum.take(lm.order - 1) |> Enum.reverse()
    :math.log(prob(lm, hs, c))
  end

  # recursion from the empty history up, by interpolation
  defp prob(lm, hs, c) do
    base = 1.0 / lm.alphabet

    Enum.reduce(0..length(hs), base, fn k, lower ->
      h = hs |> Enum.take(-k) |> Enum.join()
      h = if k == 0, do: "", else: h

      case Map.get(lm.table, h) do
        nil -> lower
        {total, types, m} -> (Map.get(m, c, 0) + types * lower) / (total + types)
      end
    end)
  end

  @doc "Bits per character of `text` under the model (lower = more expected)."
  def bits_per_char(%__MODULE__{} = lm, text) do
    cs = String.graphemes(text)

    {sum, _} =
      Enum.reduce(cs, {0.0, [" "]}, fn c, {s, rev} -> {s - log_prob_rev(lm, rev, c) / :math.log(2), [c | rev]} end)

    if cs == [], do: 0.0, else: sum / length(cs)
  end

  @doc """
  The model shipped with the OCR reader: `priv/ocr/lm.json` names its
  corpora, order and decoding weights. Built once and cached.
  `{:ok, %{lm, weight, bonus, beam, gate}}` or `{:error, reason}`.
  """
  def default(alphabet) do
    key = {__MODULE__, :default}

    case :persistent_term.get(key, nil) do
      nil ->
        dir = Path.join(to_string(:code.priv_dir(:vapor)), "ocr")

        with {:ok, raw} <- File.read(Path.join(dir, "lm.json")),
             {:ok, cfg} <- Vapor.JSON.decode(raw) do
          text = cfg["corpora"] |> Enum.map_join(" ", &File.read!(Path.expand(&1, dir)))
          # the corpora are Markdown; a printed page shows no markup: it is deleted
          text = if cfg["strip"], do: String.replace(text, Regex.compile!(cfg["strip"], "u"), ""), else: text
          lm = %{build(text, alphabet, cfg["order"]) | source: cfg["corpora"]}
          v = %{lm: lm, weight: cfg["weight"] * 1.0, bonus: cfg["bonus"] * 1.0, beam: cfg["beam"], gate: cfg["gate"] && cfg["gate"] * 1.0}
          :persistent_term.put(key, v)
          {:ok, v}
        end

      v ->
        {:ok, v}
    end
  end

  @doc """
  The control model: the same corpus with its characters shuffled (a fixed
  seed) — the same character frequencies, no language. A reading gain that
  survives this model was not the language's.
  """
  def shuffled(text, alphabet, order \\ 5) do
    :rand.seed(:exsss, {7, 7, 7})
    text |> String.graphemes() |> Enum.shuffle() |> Enum.join() |> build(alphabet, order)
  end
end
