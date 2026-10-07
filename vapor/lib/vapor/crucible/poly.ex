defmodule Vapor.Crucible.Poly do
  @moduledoc """
  Exact multivariate polynomials over the rationals (docs/CRUCIBLE.md §4):
  the arithmetic behind conservation laws that are **proved**, not fitted.
  A polynomial is a map from exponent tuples to rationals `{num, den}`;
  a number typed as `0.1` is the rational 1/10 (its shortest decimal),
  never the binary float's 3602879701896397/36028797018963968.
  """

  # ============================================================ rationals

  def q(a), do: {a, 1}
  def qn({a, b}) when b < 0, do: qn({-a, -b})
  def qn({0, _}), do: {0, 1}
  def qn({a, b}), do: (g = Integer.gcd(a, b); {div(a, g), div(b, g)})
  def qadd({a, b}, {c, d}), do: qn({a * d + c * b, b * d})
  def qsub({a, b}, {c, d}), do: qn({a * d - c * b, b * d})
  def qmul({a, b}, {c, d}), do: qn({a * c, b * d})
  def qdiv({a, b}, {c, d}) when c != 0, do: qn({a * d, b * c})
  def qneg({a, b}), do: {-a, b}
  def qzero?({a, _}), do: a == 0
  def qfloat({a, b}), do: a / b

  @doc "A float as the rational of its shortest decimal form."
  def rational(x) when is_integer(x), do: {x, 1}

  def rational(x) when is_float(x) do
    s = :erlang.float_to_binary(x, [:short])
    {mant, exp} =
      case String.split(String.downcase(s), "e") do
        [m, e] -> {m, String.to_integer(e)}
        [m] -> {m, 0}
      end
    {neg, mant} = if String.starts_with?(mant, "-"), do: {true, String.trim_leading(mant, "-")}, else: {false, mant}
    {ip, fp} = case String.split(mant, ".") do [i, f] -> {i, f}; [i] -> {i, ""} end
    digits = String.to_integer(ip <> fp)
    scale = exp - String.length(fp)
    r = if scale >= 0, do: {digits * Integer.pow(10, scale), 1}, else: qn({digits, Integer.pow(10, -scale)})
    if neg, do: qneg(r), else: r
  end

  @doc "A rational as text: 3, -1/2."
  def qtext({a, 1}), do: Integer.to_string(a)
  def qtext({a, b}), do: "#{a}/#{b}"

  # ============================================================ polynomials

  def zero, do: %{}
  def const(c, n), do: norm(%{List.to_tuple(List.duplicate(0, n)) => c})
  def var(i, n), do: %{List.to_tuple(for(j <- 0..(n - 1), do: if(j == i, do: 1, else: 0))) => {1, 1}}

  defp norm(p), do: p |> Enum.reject(fn {_, c} -> qzero?(c) end) |> Map.new()

  def add(p, q), do: Map.merge(p, q, fn _, a, b -> qadd(a, b) end) |> norm()
  def sub(p, q), do: add(p, scale(q, {-1, 1}))
  def scale(p, c), do: p |> Map.new(fn {e, a} -> {e, qmul(a, c)} end) |> norm()

  def mul(p, q) do
    for({e1, a} <- p, {e2, b} <- q, do: {add_exp(e1, e2), qmul(a, b)})
    |> Enum.reduce(%{}, fn {e, c}, acc -> Map.update(acc, e, c, &qadd(&1, c)) end)
    |> norm()
  end

  defp add_exp(a, b), do: Enum.zip_with(Tuple.to_list(a), Tuple.to_list(b), &+/2) |> List.to_tuple()

  def pow(_p, 0, n), do: const({1, 1}, n)
  def pow(p, k, n) when k > 0, do: mul(p, pow(p, k - 1, n))

  @doc "∂p/∂x_i."
  def diff(p, i) do
    p
    |> Enum.flat_map(fn {e, c} ->
      k = elem(e, i)
      if k == 0, do: [], else: [{put_elem(e, i, k - 1), qmul(c, {k, 1})}]
    end)
    |> Enum.reduce(%{}, fn {e, c}, acc -> Map.update(acc, e, c, &qadd(&1, c)) end)
    |> norm()
  end

  def zero?(p), do: map_size(norm(p)) == 0
  def degree(p), do: p |> Map.keys() |> Enum.map(&Enum.sum(Tuple.to_list(&1))) |> Enum.max(fn -> 0 end)

  @doc "Exact division by the variable x_i, or :error when some term lacks it."
  def div_var(p, i) do
    if Enum.all?(p, fn {e, _} -> elem(e, i) >= 1 end),
      do: {:ok, Map.new(p, fn {e, c} -> {put_elem(e, i, elem(e, i) - 1), c} end)},
      else: :error
  end

  @doc "Every monomial (exponent tuple) of total degree 1..d in n variables, graded order."
  def monomials(n, d) do
    for k <- 1..d, e <- compositions(k, n), do: List.to_tuple(e)
  end

  defp compositions(k, 1), do: [[k]]
  defp compositions(k, n), do: for(i <- k..0//-1, rest <- compositions(k - i, n - 1), do: [i | rest])

  @doc "Convert a `Vapor.Expr` tree to a polynomial in `vars` (constants substituted), or `:not_polynomial`."
  def from_expr(t, vars, consts \\ %{}) do
    n = length(vars)
    idx = Map.new(Enum.with_index(vars))
    {:ok, conv(t, idx, n, consts)}
  catch
    :not_polynomial -> :not_polynomial
  end

  defp conv({:n, x}, _idx, n, _c), do: const(rational(x), n)
  defp conv({:q, x, _, _}, _idx, n, _c), do: const(rational(x), n)

  defp conv({:v, name}, idx, n, c) do
    case Map.fetch(idx, name) do
      {:ok, i} -> var(i, n)
      :error ->
        case Map.fetch(c, name) do
          {:ok, v} -> const(rational(v), n)
          :error -> if name in ["pi", "e"], do: throw(:not_polynomial), else: throw(:not_polynomial)
        end
    end
  end

  defp conv({:neg, a}, idx, n, c), do: scale(conv(a, idx, n, c), {-1, 1})
  defp conv({:+, a, b}, idx, n, c), do: add(conv(a, idx, n, c), conv(b, idx, n, c))
  defp conv({:-, a, b}, idx, n, c), do: sub(conv(a, idx, n, c), conv(b, idx, n, c))
  defp conv({:*, a, b}, idx, n, c), do: mul(conv(a, idx, n, c), conv(b, idx, n, c))

  defp conv({:/, a, b}, idx, n, c) do
    pb = conv(b, idx, n, c)
    case Map.to_list(pb) do
      [{e, k}] -> if Enum.all?(Tuple.to_list(e), &(&1 == 0)) and not qzero?(k), do: scale(conv(a, idx, n, c), qdiv({1, 1}, k)), else: throw(:not_polynomial)
      _ -> throw(:not_polynomial)
    end
  end

  defp conv({:^, a, b}, idx, n, c) do
    case conv(b, idx, n, c) |> Map.to_list() do
      [] -> const({1, 1}, n)
      [{e, {k, 1}}] when k >= 0 and k <= 12 -> if Enum.all?(Tuple.to_list(e), &(&1 == 0)), do: pow(conv(a, idx, n, c), k, n), else: throw(:not_polynomial)
      _ -> throw(:not_polynomial)
    end
  end

  defp conv({:f, "sq", [a]}, idx, n, c), do: (p = conv(a, idx, n, c); mul(p, p))
  defp conv(_, _, _, _), do: throw(:not_polynomial)

  @doc "A polynomial as text in the given variable names."
  def text(p, vars) do
    if map_size(p) == 0 do
      "0"
    else
      p
      |> Enum.sort_by(fn {e, _} -> {-Enum.sum(Tuple.to_list(e)), e |> Tuple.to_list() |> Enum.map(&(-&1))} end)
      |> Enum.with_index()
      |> Enum.map_join("", fn {{e, {a, b}}, i} ->
        mono = e |> Tuple.to_list() |> Enum.zip(vars) |> Enum.filter(fn {k, _} -> k > 0 end) |> Enum.map_join("·", fn {1, v} -> v; {k, v} -> "#{v}^#{k}" end)
        coef = qtext({abs(a), b})
        body = cond do mono == "" -> coef; coef == "1" -> mono; true -> coef <> "·" <> mono end
        sign = cond do i == 0 and a < 0 -> "−"; i == 0 -> ""; a < 0 -> " − "; true -> " + " end
        sign <> body
      end)
    end
  end

  @doc "Evaluate at a point (floats)."
  def eval(p, xs) do
    Enum.reduce(p, 0.0, fn {e, c}, s ->
      s + qfloat(c) * (Enum.zip(Tuple.to_list(e), xs) |> Enum.reduce(1.0, fn {k, x}, m -> m * :math.pow(x, k) end))
    end)
  end

  # ============================================================ linear algebra over Q

  @doc "A basis of the null space of a matrix of rationals (rows), as integer vectors."
  def null_space(rows, ncols) do
    {rref, pivots} = rref(rows, ncols)
    free = Enum.reject(0..(ncols - 1)//1, &(&1 in pivots))

    for f <- free do
      v =
        for col <- 0..(ncols - 1) do
          cond do
            col == f -> {1, 1}
            col in pivots -> rref |> Enum.at(Enum.find_index(pivots, &(&1 == col))) |> Enum.at(f) |> qneg()
            true -> {0, 1}
          end
        end

      l = Enum.reduce(v, 1, fn {_, d}, acc -> div(acc * d, Integer.gcd(acc, d)) end)
      ints = Enum.map(v, fn {a, d} -> div(a * l, d) end)
      g = Enum.reduce(ints, 0, &Integer.gcd(&1, &2)) |> max(1)
      ints = Enum.map(ints, &div(&1, g))
      first = Enum.find(ints, &(&1 != 0))
      if first < 0, do: Enum.map(ints, &(-&1)), else: ints
    end
  end

  @doc "Reduced row echelon form over Q: {rows, pivot columns}."
  def rref(rows, ncols) do
    {rows, piv, _} =
      Enum.reduce(0..(ncols - 1)//1, {rows, [], 0}, fn c, {rows, piv, r} ->
        case Enum.find_index(Enum.drop(rows, r), fn row -> not qzero?(Enum.at(row, c)) end) do
          nil -> {rows, piv, r}
          i0 ->
            i = i0 + r
            rows = swap(rows, r, i)
            prow = Enum.at(rows, r)
            pv = Enum.at(prow, c)
            prow = Enum.map(prow, &qdiv(&1, pv))
            rows =
              rows
              |> List.replace_at(r, prow)
              |> Enum.with_index()
              |> Enum.map(fn {row, k} ->
                if k == r do
                  row
                else
                  f = Enum.at(row, c)
                  if qzero?(f), do: row, else: Enum.zip_with(row, prow, fn a, b -> qsub(a, qmul(f, b)) end)
                end
              end)
            {rows, piv ++ [c], r + 1}
        end
      end)

    {Enum.take(rows, length(piv)), piv}
  end

  defp swap(rows, a, b) when a == b, do: rows
  defp swap(rows, a, b), do: rows |> List.replace_at(a, Enum.at(rows, b)) |> List.replace_at(b, Enum.at(rows, a))
end
