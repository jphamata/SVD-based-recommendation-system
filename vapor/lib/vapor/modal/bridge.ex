defmodule Vapor.Modal.Bridge do
  @moduledoc """
  A linear bridge between two row spaces, fitted in **closed form**:
  ridge regression `W = argmin ‖XWᵀ + b − Y‖² + λ‖W‖²` by the normal
  equations and a Cholesky factorisation, in binary64, deterministic.

  This is how frozen models of different modalities are connected without
  training a joint model: a linear map from one model's representation to
  another's (the LLaVA projector, linear probes, "linearly mapping from
  image to text space"). Whether a linear bridge *can* carry the signal is
  an empirical question the quality gate answers per bridge; when the
  structure is compositional and additive, a linear bridge generalises to
  combinations it never saw (tested).

  The result is a `vapor_linear` checkpoint admitted through the model
  airlock (`Vapor.Lock.Adapters.Linear`).
  """
  alias Vapor.Tensor

  @doc """
  Fit `ys ≈ xs·Wᵀ + b` over paired rows (lists of float lists, or `f32`
  tensors). Options: `ridge` (λ, default 1.0e-6), `from`, `to` (modality
  labels). Returns `{:ok, spec, weights}`.
  """
  def fit(xs, ys, opts \\ []) do
    xs = rows(xs)
    ys = rows(ys)
    true = length(xs) == length(ys)
    n = length(xs)
    dx = length(hd(xs))
    k = div(dx + 15, 16) * 16
    m = length(hd(ys))
    lam = Keyword.get(opts, :ridge, 1.0e-6)

    # centre: the bias absorbs the means and is not regularised
    mx = mean(xs)
    my = mean(ys)
    xc = Enum.map(xs, &Enum.zip_with(&1, mx, fn a, b -> a - b end))
    yc = Enum.map(ys, &Enum.zip_with(&1, my, fn a, b -> a - b end))

    # W : m × dx. Primal (XᵀX + λI)Wᵀ = XᵀY when n > dx, dual
    # Wᵀ = Xᵀ(XXᵀ + λI)⁻¹Y otherwise — the same minimiser, the smaller system
    w =
      if n > dx do
        l = cholesky(add_ridge(gram(transpose(xc)), lam))
        xty = for col <- transpose(yc), do: for(xcol <- transpose(xc), do: dot(xcol, col))
        Enum.map(xty, &solve(l, &1))
      else
        l = cholesky(add_ridge(gram(xc), lam))
        alphas = for col <- transpose(yc), do: solve(l, col)
        xt = transpose(xc)
        for a <- alphas, do: for(xcol <- xt, do: dot(xcol, a))
      end

    bias = Enum.zip_with(w, my, fn row, y -> y - dot(row, mx) end)
    weight = for row <- w, do: row ++ List.duplicate(0.0, k - dx)

    config = %{"model_type" => "vapor_linear", "in_width" => k, "out_width" => m,
               "from" => Keyword.get(opts, :from, "rows"), "to" => Keyword.get(opts, :to, "rows"), "fitted_in" => dx}

    Vapor.Lock.from_map(config, %{"weight" => Tensor.from_list(:f32, [m, k], List.flatten(weight)),
                                  "bias" => Tensor.from_list(:f32, [m], bias)})
  end

  defp mean(rows) do
    n = length(rows)
    rows |> transpose() |> Enum.map(&(Enum.sum(&1) / n))
  end

  defp transpose(rows), do: Enum.zip_with(rows, & &1)
  defp dot(a, b), do: Enum.zip_reduce(a, b, 0.0, fn x, y, s -> s + x * y end)

  # G = A·Aᵀ for the rows of A
  defp gram(a) do
    for r <- a, do: for(q <- a, do: dot(r, q))
  end

  defp add_ridge(g, lam), do: g |> Enum.with_index() |> Enum.map(fn {row, i} -> List.update_at(row, i, &(&1 + lam)) end)

  @doc "Pad rows with zeros to the bridge's input width."
  def pad(rows, k), do: Enum.map(rows(rows), &(&1 ++ List.duplicate(0.0, k - length(&1))))

  defp rows(%Tensor{shape: [_, w]} = t), do: t |> Tensor.to_floats() |> Enum.chunk_every(w)
  defp rows(list) when is_list(list), do: list

  # A = L·Lᵀ (A symmetric positive definite after the ridge term)
  defp cholesky(a) do
    n = length(a)
    at = a |> Enum.map(&List.to_tuple/1) |> List.to_tuple()

    Enum.reduce(0..(n - 1), %{}, fn j, l ->
      s = Enum.reduce(0..(j - 1)//1, 0.0, fn k, s -> s + l[{j, k}] * l[{j, k}] end)
      d = elem(elem(at, j), j) - s
      ljj = :math.sqrt(max(d, 1.0e-300))
      l = Map.put(l, {j, j}, ljj)

      Enum.reduce((j + 1)..(n - 1)//1, l, fn i, l ->
        s = Enum.reduce(0..(j - 1)//1, 0.0, fn k, s -> s + l[{i, k}] * l[{j, k}] end)
        Map.put(l, {i, j}, (elem(elem(at, i), j) - s) / ljj)
      end)
    end)
    |> then(&{&1, n})
  end

  defp solve({l, n}, b) do
    bt = List.to_tuple(b)

    y =
      Enum.reduce(0..(n - 1), %{}, fn i, y ->
        s = Enum.reduce(0..(i - 1)//1, 0.0, fn k, s -> s + l[{i, k}] * y[k] end)
        Map.put(y, i, (elem(bt, i) - s) / l[{i, i}])
      end)

    x =
      Enum.reduce((n - 1)..0//-1, %{}, fn i, x ->
        s = Enum.reduce((i + 1)..(n - 1)//1, 0.0, fn k, s -> s + l[{k, i}] * x[k] end)
        Map.put(x, i, (y[i] - s) / l[{i, i}])
      end)

    for i <- 0..(n - 1), do: x[i]
  end
end
