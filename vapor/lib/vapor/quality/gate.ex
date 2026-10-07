defmodule Vapor.Quality.Gate do
  @moduledoc """
  A **calibrated, self-validating** decision on one score.

  A quality threshold picked by hand certifies nothing: it may pass noise
  or fail signal and nobody would know. A gate here is *fitted to
  controls* and refuses to exist when the controls do not separate:

      negatives (must fail): white noise, shuffled signal (same marginals,
                             no structure), degenerate repetition …
      positives (must pass): held-out real signal

      t_noise   = the highest negative score      (anything ≤ is :noise)
      t_natural = the lowest positive score       (anything ≥ is :natural)
      valid  ⇔  t_noise < t_natural                (else {:error, :inseparable})

  Between the two lies `:structured` — more structure than any noise
  control, less than every real sample (a bigram model's text, a blurred
  image). The verdict "not noise" is `:structured` or `:natural`.

  Scores are oriented so that higher means more structure (`direction:
  :up`) or the opposite (`:down`, e.g. spectral flatness, where noise is
  high). The gate records its controls' statistics and its margin, so a
  report can show *why* it is trusted.
  """
  @enforce_keys [:name, :direction, :t_noise, :t_natural]
  defstruct [:name, :direction, :t_noise, :t_natural, :margin, controls: %{}]

  @type t :: %__MODULE__{}

  @doc """
  Fit a gate. `negatives` and `positives` are `%{label => [score]}`.
  Returns `{:ok, gate}` or `{:error, {:inseparable, details}}`.
  """
  def calibrate(name, negatives, positives, direction \\ :up) do
    o = orient(direction)
    neg = negatives |> Map.values() |> List.flatten() |> Enum.map(o)
    pos = positives |> Map.values() |> List.flatten() |> Enum.map(o)

    {t_noise, t_natural} = {Enum.max(neg), Enum.min(pos)}
    stats = Map.new(Map.merge(negatives, positives), fn {k, v} -> {k, summary(v)} end)

    if t_noise < t_natural do
      {:ok, %__MODULE__{name: name, direction: direction, t_noise: o.(t_noise), t_natural: o.(t_natural),
                        margin: t_natural - t_noise, controls: stats}}
    else
      {:error, {:inseparable, %{gate: name, worst_negative: o.(t_noise), worst_positive: o.(t_natural), controls: stats}}}
    end
  end

  @doc "`:noise | :structured | :natural` for a score."
  def judge(%__MODULE__{direction: d, t_noise: tn, t_natural: tp}, score) do
    o = orient(d)

    cond do
      o.(score) <= o.(tn) -> :noise
      o.(score) >= o.(tp) -> :natural
      true -> :structured
    end
  end

  @doc "Whether a verdict means \"not noise\"."
  def signal?(v), do: v in [:structured, :natural]

  defp orient(:up), do: & &1
  defp orient(:down), do: &(-&1)

  @doc "min / mean / max of a list of scores."
  def summary([]), do: %{n: 0}

  def summary(xs) do
    n = length(xs)
    %{n: n, min: Enum.min(xs), mean: Enum.sum(xs) / n, max: Enum.max(xs)}
  end
end
