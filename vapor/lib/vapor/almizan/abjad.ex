defmodule Vapor.Almizan.Abjad do
  @moduledoc """
  *Ḥisāb al-jummal* (حساب الجُمَّل), the abjad numerals: every letter a fixed
  number, in the Mashriqi order (ا 1 … غ 1000). The manifesto proposed the
  value of a function's root as its *address* — the dispatch key, the
  register, the CAM line. `collisions/0` measures why it cannot be: the sum
  ignores the order of the letters (every anagram collides), and with 28
  letters and values from 3 to 3000 most of the 28³ roots share their value
  with another. An address must be injective; the abjad value is shown
  (it is beautiful, and it is history) while identity is the hash.
  """

  @values [
    {"ا", 1}, {"ب", 2}, {"ج", 3}, {"د", 4}, {"ه", 5}, {"و", 6}, {"ز", 7}, {"ح", 8}, {"ط", 9}, {"ي", 10},
    {"ك", 20}, {"ل", 30}, {"م", 40}, {"ن", 50}, {"س", 60}, {"ع", 70}, {"ف", 80}, {"ص", 90}, {"ق", 100},
    {"ر", 200}, {"ش", 300}, {"ت", 400}, {"ث", 500}, {"خ", 600}, {"ذ", 700}, {"ض", 800}, {"ظ", 900}, {"غ", 1000}
  ]
  @table Map.new(@values)
  # letter forms that count as their base letter
  @forms %{"أ" => "ا", "إ" => "ا", "آ" => "ا", "ٱ" => "ا", "ة" => "ه", "ى" => "ي", "ؤ" => "و", "ئ" => "ي"}

  @doc "The 28 letters with their values."
  def letters, do: @values

  @doc "The abjad value of a word or of a root written `ح-ف-ظ` (marks and hyphens are skipped)."
  def value(word) do
    word
    |> String.graphemes()
    |> Enum.map(&Map.get(@forms, &1, &1))
    |> Enum.reduce(0, fn ch, acc -> acc + Map.get(@table, String.first(ch) || "", 0) end)
  end

  @doc """
  Over every root of three letters (28³ ordered triples): how many distinct
  values there are, how many roots share their value with another, and the
  largest class. The measured refutation of "the gematric value as address".
  """
  def collisions do
    vals = for {_, a} <- @values, {_, b} <- @values, {_, c} <- @values, do: a + b + c
    freq = Enum.frequencies(vals)
    shared = Enum.count(vals, &(freq[&1] > 1))
    %{roots: length(vals), distinct_values: map_size(freq), sharing: shared, fraction_sharing: shared / length(vals), largest_class: Enum.max(Map.values(freq))}
  end
end
