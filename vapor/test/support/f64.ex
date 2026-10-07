defmodule Vapor.TestF64 do
  @moduledoc false
  # A binary64 evaluator of the differentiable term subset: the reference
  # for finite-difference gradient checks (independent of the f32 oracle).
  alias Vapor.Algebra.Term, as: T

  def eval(root, env) do
    root |> T.postorder() |> Enum.reduce(%{}, fn t, m -> Map.put(m, t, node(t, m, env)) end) |> Map.fetch!(root)
  end

  # tensors are {shape, tuple of floats}, row-major
  defp node({:input, n, _, _}, _m, env), do: Map.fetch!(env, n)
  defp node({:const, t}, _m, _env), do: {t.shape, t |> Vapor.Tensor.to_floats() |> List.to_tuple()}

  defp node({:ew, op, args}, m, _env) do
    vals = Enum.map(args, fn
      {:splat, b} -> {[1], {Vapor.F32.to_float(b)}}
      a -> m[a]
    end)

    shape = vals |> Enum.map(&elem(&1, 0)) |> Enum.filter(&(length(&1) > 1 or &1 != [1])) |> Enum.max_by(&Enum.product/1, fn -> [1] end)
    shape = if Enum.all?(vals, &(elem(&1, 0) == [1])), do: [1], else: shape
    n = Enum.product(shape)
    at = fn {s, v}, i -> elem(v, bidx(s, shape, i)) end
    {shape, List.to_tuple(for i <- 0..(n - 1), do: f(op, Enum.map(vals, &at.(&1, i))))}
  end

  defp node({:reduce, :sum, x}, m, _env) do
    {s, v} = m[x]
    c = List.last(s)
    rows = div(tuple_size(v), c)
    {List.replace_at(s, -1, 1), List.to_tuple(for r <- 0..(rows - 1), do: Enum.sum(for j <- 0..(c - 1), do: elem(v, r * c + j)))}
  end

  defp node({:linear, x, w}, m, _env) do
    {[t, k], xv} = m[x]
    {[n, ^k], wv} = m[w]
    {[t, n], List.to_tuple(for i <- 0..(t - 1), o <- 0..(n - 1), do: Enum.sum(for j <- 0..(k - 1), do: elem(xv, i * k + j) * elem(wv, o * k + j)))}
  end

  defp node({:transpose, x}, m, _env) do
    {[r, c], v} = m[x]
    {[c, r], List.to_tuple(for j <- 0..(c - 1), i <- 0..(r - 1), do: elem(v, i * c + j))}
  end

  # index of element i of `shape` in an operand of shape s (broadcasting)
  defp bidx([1], _shape, _i), do: 0
  defp bidx(s, s, i), do: i
  defp bidx([r, 1], [_, c], i), do: div(i, c) |> min(r - 1)
  defp bidx([1, c], [_, c], i), do: rem(i, c)
  defp bidx([1, 1], _, _), do: 0

  defp f(:add, [a, b]), do: a + b
  defp f(:sub, [a, b]), do: a - b
  defp f(:mul, [a, b]), do: a * b
  defp f(:neg, [a]), do: -a
  defp f(:div, [a, b]), do: a / b
  defp f(:exp, [a]), do: :math.exp(a)
  defp f(:rcp, [a]), do: 1 / a
  defp f(:rsqrt, [a]), do: 1 / :math.sqrt(a)
  defp f(:sigmoid, [a]), do: 1 / (1 + :math.exp(-a))
  defp f(:silu, [a]), do: a / (1 + :math.exp(-a))
end
