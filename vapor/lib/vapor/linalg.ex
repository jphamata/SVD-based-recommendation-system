defmodule Vapor.Linalg do
  @moduledoc """
  Small dense linear algebra in binary64 on the BEAM — the same bits on
  every host (IEEE `+ − × ÷ √`, fixed summation order, no libm, no BLAS).

  Matrices are tuples of row tuples. Used where a closed form needs a
  solve: the least-squares fusion of `Vapor.Merge` (`:regmean`), Gram
  accumulation. Cost is that of the BEAM (Cholesky `n³/3` multiply-adds:
  a 512 × 512 system in a few seconds); larger systems belong in a program.
  """

  @doc "`n × n` zero matrix."
  def zeros(n, m \\ nil), do: Tuple.duplicate(Tuple.duplicate(0.0, m || n), n)

  @doc "Element `(i, j)`."
  def at(a, i, j), do: elem(elem(a, i), j)

  @doc "Rows as lists → tuple matrix."
  def from_rows(rows), do: rows |> Enum.map(&List.to_tuple(Enum.map(&1, fn x -> x * 1.0 end))) |> List.to_tuple()

  @doc "Tuple matrix → rows as lists."
  def to_rows(a), do: a |> Tuple.to_list() |> Enum.map(&Tuple.to_list/1)

  @doc "A row-major f32 binary of `rows × cols` → tuple matrix (binary64)."
  def from_f32(bin, rows, cols) do
    for r <- 0..(rows - 1) do
      for <<x::float-32-little <- binary_part(bin, r * cols * 4, cols * 4)>>, do: x
    end
    |> Enum.map(&List.to_tuple/1)
    |> List.to_tuple()
  end

  @doc "Tuple matrix → row-major f32 binary (each entry rounded once)."
  def to_f32(a), do: for(row <- Tuple.to_list(a), x <- Tuple.to_list(row), into: <<>>, do: <<x::float-32-little>>)

  @doc "`a + b`."
  def add(a, b), do: map2(a, b, &(&1 + &2))

  @doc "`s · a`."
  def scale(a, s), do: a |> Tuple.to_list() |> Enum.map(fn r -> r |> Tuple.to_list() |> Enum.map(&(&1 * s)) |> List.to_tuple() end) |> List.to_tuple()

  defp map2(a, b, f) do
    Enum.zip_with(Tuple.to_list(a), Tuple.to_list(b), fn ra, rb ->
      Enum.zip_with(Tuple.to_list(ra), Tuple.to_list(rb), f) |> List.to_tuple()
    end)
    |> List.to_tuple()
  end

  @doc "`a · b` (`n × k` by `k × m`)."
  def mul(a, b) do
    bt = transpose(b) |> Tuple.to_list()
    a |> Tuple.to_list() |> Enum.map(fn ra -> bt |> Enum.map(&dot(ra, &1)) |> List.to_tuple() end) |> List.to_tuple()
  end

  @doc "Transpose."
  def transpose(a) do
    case tuple_size(a) do
      0 -> {}
      _ -> a |> Tuple.to_list() |> Enum.map(&Tuple.to_list/1) |> Enum.zip_with(&List.to_tuple/1) |> List.to_tuple()
    end
  end

  defp dot(ra, rb), do: dot(ra, rb, 0, tuple_size(ra), 0.0)
  defp dot(_ra, _rb, i, n, s) when i == n, do: s
  defp dot(ra, rb, i, n, s), do: dot(ra, rb, i + 1, n, s + elem(ra, i) * elem(rb, i))

  @doc """
  Accumulate `G += Xᵀ X` for the rows of `x` (a list of row lists or tuples
  of width `n`); `g` a tuple matrix or `nil` (zeros).
  """
  def gram_add(nil, [r | _] = x), do: gram_add(zeros(tuple_or_len(r)), x)

  def gram_add(g, x) do
    xs = Enum.map(x, &to_tuple/1)
    n = tuple_size(g)

    for i <- 0..(n - 1) do
      gi = elem(g, i)

      for j <- 0..(n - 1) do
        Enum.reduce(xs, elem(gi, j), fn r, s -> s + elem(r, i) * elem(r, j) end)
      end
      |> List.to_tuple()
    end
    |> List.to_tuple()
  end

  defp to_tuple(r) when is_tuple(r), do: r
  defp to_tuple(r) when is_list(r), do: List.to_tuple(r)
  defp tuple_or_len(r) when is_tuple(r), do: tuple_size(r)
  defp tuple_or_len(r) when is_list(r), do: length(r)

  @doc """
  Cholesky factor `L` (lower; row `i` a tuple of `i + 1` entries) of a
  symmetric positive definite `a`, by Cholesky–Banachiewicz (row by row,
  entries left to right, sums in increasing index): `{:ok, l}` or
  `{:error, :not_positive_definite, j}` at the first non-positive pivot.
  """
  def cholesky(a) do
    n = tuple_size(a)

    Enum.reduce_while(0..(n - 1)//1, {}, fn j, l ->
      aj = elem(a, j)

      row =
        Enum.reduce(0..(j - 1)//1, {}, fn k, row ->
          lk = elem(l, k)
          Tuple.append(row, (elem(aj, k) - dot(row, lk, 0, k, 0.0)) / elem(lk, k))
        end)

      d = elem(aj, j) - dot(row, row, 0, j, 0.0)
      if d > 0.0, do: {:cont, Tuple.append(l, Tuple.append(row, :math.sqrt(d)))}, else: {:halt, {:error, :not_positive_definite, j}}
    end)
    |> case do
      {:error, _, _} = e -> e
      l -> {:ok, l}
    end
  end

  @doc """
  Solve `A x = b` given `L` (`A = L Lᵀ`), for one right-hand side `b`
  (a tuple or list): forward then backward substitution.
  """
  def chol_solve(l, b) do
    n = tuple_size(l)
    b = to_tuple(b)

    y =
      Enum.reduce(0..(n - 1)//1, {}, fn i, y ->
        li = elem(l, i)
        s = Enum.reduce(0..(i - 1)//1, 0.0, fn k, s -> s + elem(li, k) * elem(y, k) end)
        Tuple.append(y, (elem(b, i) - s) / elem(li, i))
      end)

    lt = transpose_lower(l, n)

    Enum.reduce((n - 1)..0//-1, %{}, fn i, x ->
      ci = elem(lt, i)
      s = Enum.reduce((i + 1)..(n - 1)//1, 0.0, fn k, s -> s + elem(ci, k) * Map.fetch!(x, k) end)
      Map.put(x, i, (elem(y, i) - s) / elem(ci, i))
    end)
    |> then(fn x -> for(i <- 0..(n - 1), do: Map.fetch!(x, i)) |> List.to_tuple() end)
  end

  # Lᵀ as full rows (row i of Lᵀ = column i of L), zeros below the diagonal
  defp transpose_lower(l, n) do
    for i <- 0..(n - 1) do
      for k <- 0..(n - 1), do: (if k >= i, do: elem(elem(l, k), i), else: 0.0)
    end
    |> Enum.map(&List.to_tuple/1)
    |> List.to_tuple()
  end
end
