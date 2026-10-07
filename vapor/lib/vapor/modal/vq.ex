defmodule Vapor.Modal.VQ do
  @moduledoc """
  Fitting a vector-quantised codec (`Vapor.Lock.Adapters.Codec`) to data:
  k-means (Lloyd) in binary64, deterministic — initialisation by
  farthest-point traversal from the row nearest the mean (no randomness at
  all), then `iters` Lloyd steps; ties go to the lower index everywhere.

      {:ok, spec, weights} = Vapor.Modal.VQ.fit(rows, 64, modality: "image", iters: 12)
      {:ok, enc} = Vapor.Lock.build(spec, weights, direction: :encode, rows: n)

  The result is a checkpoint like any other: a `config.json` map and a
  `codebook` tensor, admitted through the model airlock.
  """
  alias Vapor.Tensor

  @doc """
  Fit `k` codewords to rows `f32[n, w]` (or a list of float lists).
  Options: `iters` (10), `modality` (`"rows"`), `extra` (more config keys).
  Returns `{:ok, spec, weights}` (admitted), plus the config map as
  `spec.config.raw`.
  """
  def fit(rows, k, opts \\ []) do
    pts = points(rows)
    w = length(hd(pts))
    k = min(k, length(Enum.uniq(pts)))
    centres = init(pts, k)
    centres = Enum.reduce(1..Keyword.get(opts, :iters, 10)//1, centres, fn _, cs -> step(pts, cs) end)
    book = Tensor.from_list(:f32, [k, w], List.flatten(centres))

    config =
      Map.merge(%{"model_type" => "vapor_vq", "codebook_size" => k, "row_width" => w, "modality" => Keyword.get(opts, :modality, "rows")},
                Keyword.get(opts, :extra, %{}))

    Vapor.Lock.from_map(config, %{"codebook" => book})
  end

  defp points(%Tensor{shape: [_, w]} = t), do: t |> Tensor.to_floats() |> Enum.chunk_every(w)
  defp points(list) when is_list(list), do: list

  defp init(pts, k) do
    mean = centroid(pts)
    first = Enum.min_by(Enum.with_index(pts), fn {p, i} -> {d2(p, mean), i} end) |> elem(0)
    dists = Enum.map(pts, &d2(&1, first))

    {cs, _} =
      Enum.reduce(2..k//1, {[first], dists}, fn _, {cs, ds} ->
        {far, _} = Enum.zip(pts, ds) |> Enum.with_index() |> Enum.max_by(fn {{_, d}, i} -> {d, -i} end) |> elem(0)
        {[far | cs], Enum.zip_with(pts, ds, fn p, d -> min(d, d2(p, far)) end)}
      end)

    Enum.reverse(cs)
  end

  defp step(pts, cs) do
    groups = Enum.group_by(pts, &nearest(&1, cs))
    for {c, j} <- Enum.with_index(cs), do: (case groups[j] do
      nil -> c
      members -> centroid(members)
    end)
  end

  @doc "Index of the nearest codeword (ties: lower index)."
  def nearest(p, cs) do
    cs |> Enum.with_index() |> Enum.min_by(fn {c, j} -> {d2(p, c), j} end) |> elem(1)
  end

  defp centroid(pts) do
    n = length(pts)
    pts |> Enum.zip_with(& &1) |> Enum.map(&(Enum.sum(&1) / n))
  end

  defp d2(a, b), do: Enum.zip_reduce(a, b, 0.0, fn x, y, s -> s + (x - y) * (x - y) end)
end
