defmodule Vapor.Autodiff do
  @moduledoc """
  Reverse-mode differentiation of terms into terms (phase P7).

  `grad(y, seed, wrt)` returns, for each term in `wrt`, the term computing
  `(∂y/∂w)ᵀ·seed` — a vector–Jacobian product, built from the same
  operators as the forward pass, so gradients are programs like any other:
  lowered, run on every substrate, bit-identical across them and
  certified. Contributions to one adjoint are summed in a fixed order (the
  reverse topological order of the forward DAG), so the gradient program —
  and its bits — are deterministic.

  Differentiable operators: `add sub mul neg div exp rcp rsqrt sigmoid
  silu tanh fma` and, piecewise, `max min relu sel` (with broadcasting), `reduce(:sum)`, `linear` (x : f32[T, k]),
  `transpose`; leaves are inputs and constants. Anything else on a path to
  a requested leaf is a rejection naming the operator.

  The rules, with `o` the node's value:

  | node | adjoint of each operand |
  |---|---|
  | `a + b`, `a − b` | `ḡ`, `±ḡ` (summed over broadcast axes) |
  | `a·b` | `ḡ·b`, `ḡ·a` |
  | `a / b` | `ḡ/b`, `−ḡ·o/b` |
  | `exp a` | `ḡ·o` |
  | `rcp a` | `−ḡ·o·o` |
  | `rsqrt a` | `−½·ḡ·o·o·o` |
  | `σ(a)` | `ḡ·o·(1 − o)` |
  | `tanh a` | `ḡ·(1 − o²)` |
  | `max(a, b)`, `min(a, b)` | `ḡ` to the operand chosen (a tie: the first), `0` to the other |
  | `relu a` | `ḡ` where `a > 0`, else `0` |
  | `(a < b) ? x : y` | `0`, `0`, `ḡ` where taken, `ḡ` where taken |
  | `silu a` | `ḡ·σ(a)·(1 + a·(1 − σ(a)))` |
  | `Σ_last a` | `ḡ` broadcast along the reduced axis |
  | `x·Wᵀ` | `ḡ·W` (as `linear(ḡ, Wᵀ)`), `ḡᵀ·x` (as `linear(ḡᵀ, xᵀ)`) |
  | `xᵀ` | `ḡᵀ` |

  Summing over a broadcast axis uses `reduce(:sum)` (rows) or a
  transpose–reduce–transpose (columns): extents reduced that way must be
  multiples of 16, like every canonical reduction.

  `detach:` names nodes through which no gradient flows (their value is a
  constant for the backward pass). The one use that matters: the row
  maximum subtracted before a softmax's `exp` — the softmax does not depend
  on it, so its gradient is exactly zero in the reals, and a `reduce(:max)`
  (not differentiable here) never needs one.
  """
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Rejection

  @spec grad(term, term, [term], keyword) :: {:ok, [term]} | {:error, Rejection.t()}
  def grad(y, seed, wrt, opts \\ []) do
    order = T.postorder([y])
    detached = MapSet.new(Keyword.get(opts, :detach, []))
    needed = needed(order, MapSet.new(wrt), detached)

    adj =
      order
      |> Enum.reverse()
      |> Enum.reduce_while(%{y => seed}, fn node, adj ->
        case Map.fetch(adj, node) do
          {:ok, g} -> if T.leaf?(node), do: {:cont, adj}, else: back(node, g, adj, needed)
          :error -> {:cont, adj}
        end
      end)

    case adj do
      {:error, _} = e -> e
      adj -> {:ok, Enum.map(wrt, fn w -> Map.get(adj, w) || zeros_like(w) end)}
    end
  end

  @doc """
  Reverse mode over a program with let-bindings — what keeps the backward
  pass of a deep network linear in size. Terms are trees: a value used
  twice is two references to one subterm in memory, but every traversal
  (hashing, comparison) walks it twice, and the backward pass of a
  transformer, built as one term, grows exponentially with depth. Here
  every binding is differentiated on its own, with the refs to earlier
  bindings as leaves; each binding's adjoint is itself let-bound
  (`:"adj.<name>"`) before it is propagated, so the backward program has
  one binding per forward binding, plus one per gradient.

  `lets` are the forward bindings in order (`[{name, body}]`), `y` the
  differentiated output (its leaves may be refs), `seed` its adjoint, and
  `wrt` the input terms (`{:input, …}`) whose gradients are wanted.
  Contributions to one adjoint are summed in a fixed order (reverse binding
  order, then leaf order), so the bits are deterministic. Returns
  `{:ok, grads, lets}`: `grads` the gradient terms (refs to
  `:"grad.<input>"` bindings) in the order of `wrt`, `lets` the forward
  bindings followed by the backward ones. Option `detach:` as `grad/4`.
  """
  def grad_lets(lets, y, seed, wrt, opts \\ []) do
    bound = Map.new(lets)
    wanted = MapSet.new(for {:input, n, _, _} <- wrt, do: n)

    leaves = fn t ->
      t
      |> T.postorder()
      |> Enum.filter(fn
        {:input, n, _, _} -> Map.has_key?(bound, n) or MapSet.member?(wanted, n)
        _ -> false
      end)
      |> Enum.uniq()
    end

    # one term's contributions to its leaves: [{leaf_name, term}], in leaf order
    contrib = fn term, adj ->
      ls = leaves.(term)

      case grad(term, adj, ls, opts) do
        {:ok, gs} -> {:ok, for({{:input, n, _, _} = l, g} <- Enum.zip(ls, gs), g != zeros_like(l), do: {n, g})}
        err -> err
      end
    end

    with {:ok, first} <- contrib.(y, seed) do
      acc = Enum.reduce(first, %{}, fn {n, g}, a -> Map.update(a, n, [g], &[g | &1]) end)

      {acc, back} =
        lets
        |> Enum.reverse()
        |> Enum.reduce_while({acc, []}, fn {name, body}, {acc, back} ->
          case Map.pop(acc, name) do
            {nil, acc} ->
              {:cont, {acc, back}}

            {terms, acc} ->
              # contributions arrived latest-first; sum them in arrival order
              adj_name = :"adj.#{name}"
              sum = terms |> Enum.reverse() |> Enum.reduce(&T.add(&2, &1))
              adj = T.ref(adj_name, sum)

              case contrib.(body, adj) do
                {:ok, cs} -> {:cont, {Enum.reduce(cs, acc, fn {n, g}, a -> Map.update(a, n, [g], &[g | &1]) end), [{adj_name, sum} | back]}}
                {:error, _} = e -> {:halt, e}
              end
          end
        end)
        |> case do
          {:error, _} = e -> e
          ok -> ok
        end
        |> case do
          {:error, _} = e -> throw(e)
          ok -> ok
        end

      {grads, glets} =
        Enum.map_reduce(wrt, [], fn {:input, n, _, _} = w, gl ->
          case Map.get(acc, n) do
            nil -> {zeros_like(w), gl}
            terms ->
              gname = :"grad.#{n}"
              sum = terms |> Enum.reverse() |> Enum.reduce(&T.add(&2, &1))
              {T.ref(gname, sum), [{gname, sum} | gl]}
          end
        end)

      {:ok, grads, lets ++ Enum.reverse(back) ++ Enum.reverse(glets)}
    end
  catch
    {:error, _} = e -> e
  end

  # nodes on a path to some requested leaf (a detached node is on none)
  defp needed(order, wrt, detached) do
    Enum.reduce(order, %{}, fn n, acc ->
      cond do
        MapSet.member?(detached, n) -> acc
        MapSet.member?(wrt, n) or Enum.any?(T.children(n), &Map.has_key?(acc, &1)) -> Map.put(acc, n, true)
        true -> acc
      end
    end)
  end

  defp back(node, g, adj, needed) do
    case contributions(node, g) do
      {:ok, pairs} ->
        {:cont,
         Enum.reduce(pairs, adj, fn {child, c}, acc ->
           if Map.has_key?(needed, child), do: Map.update(acc, child, c, &T.add(&1, c)), else: acc
         end)}

      {:error, _} = e ->
        {:halt, e}
    end
  end

  # ------------------------------------------------------------- the rules --

  defp contributions({:ew, op, args} = node, g) do
    with {:ok, local} <- local(op, args, node, g) do
      # one adjoint per operand position; splats are constants
      pairs = for {a, d} <- Enum.zip(args, local), not splat?(a), do: {a, unbroadcast(d, a)}
      {:ok, pairs}
    end
  end

  defp contributions({:reduce, :sum, x}, g), do: {:ok, [{x, T.mul(g, ones_like(x))}]}
  defp contributions({:linear, x, w}, g), do: {:ok, [{x, T.linear(g, T.transpose(w))}, {w, T.linear(T.transpose(g), T.transpose(x))}]}
  defp contributions({:transpose, x}, g), do: {:ok, [{x, T.transpose(g)}]}
  defp contributions({:reshape, x, _}, g), do: (with {:ok, {_, s}} <- T.infer(x), do: {:ok, [{x, T.reshape(g, s)}]})

  defp contributions(node, _g),
    do: {:error, Rejection.new(node, "a differentiable operator (#{inspect(elem(node, 0))} is not)", "keep it out of the differentiated path")}

  defp local(:add, [_, _], _n, g), do: {:ok, [g, g]}
  defp local(:sub, [_, _], _n, g), do: {:ok, [g, T.neg(g)]}
  defp local(:mul, [a, b], _n, g), do: {:ok, [T.mul(g, b), T.mul(g, a)]}
  defp local(:neg, [_], _n, g), do: {:ok, [T.neg(g)]}
  defp local(:div, [_a, b], o, g), do: {:ok, [T.divide(g, b), T.neg(T.divide(T.mul(g, o), b))]}
  defp local(:exp, [_], o, g), do: {:ok, [T.mul(g, o)]}
  defp local(:rcp, [_], o, g), do: {:ok, [T.neg(T.mul(g, T.mul(o, o)))]}
  defp local(:rsqrt, [_], o, g), do: {:ok, [T.mul(T.mul(g, T.splat(-0.5)), T.mul(o, T.mul(o, o)))]}
  defp local(:sigmoid, [_], o, g), do: {:ok, [T.mul(g, T.mul(o, T.sub(T.splat(1.0), o)))]}

  defp local(:silu, [a], _o, g) do
    s = T.sigmoid(a)
    {:ok, [T.mul(g, T.mul(s, T.add(T.splat(1.0), T.mul(a, T.sub(T.splat(1.0), s)))))]}
  end

  # piecewise operators: the (sub)gradient goes to the operand that was
  # chosen (a tie goes to the first); contact, clamping and ReLU become
  # differentiable almost everywhere — the gradient of a simulation with
  # ground contact is the gradient of the branch it took
  defp local(:max, [a, b], _o, g), do: {:ok, [T.sel(a, b, zeros_like(g), g), T.sel(a, b, g, zeros_like(g))]}
  defp local(:min, [a, b], _o, g), do: {:ok, [T.sel(a, b, g, zeros_like(g)), T.sel(a, b, zeros_like(g), g)]}
  defp local(:relu, [a], _o, g), do: {:ok, [T.sel(T.splat(0.0), a, g, zeros_like(g))]}
  defp local(:sel, [a, b, _x, _y], _o, g), do: {:ok, [T.mul(a, T.splat(0.0)), T.mul(b, T.splat(0.0)), T.sel(a, b, g, zeros_like(g)), T.sel(a, b, zeros_like(g), g)]}
  defp local(:tanh, [_], o, g), do: {:ok, [T.mul(g, T.sub(T.splat(1.0), T.mul(o, o)))]}
  defp local(:fma, [a, b, _c], _o, g), do: {:ok, [T.mul(g, b), T.mul(g, a), g]}

  defp local(op, _args, node, _g),
    do: {:error, Rejection.new(node, "a differentiable operator (#{op} is not)", "keep it out of the differentiated path")}

  # sum the adjoint over the axes along which the operand was broadcast
  defp unbroadcast(d, a) do
    {:ok, {_, ds}} = T.infer(d)
    {:ok, {_, as}} = T.infer(a)

    case {ds, as} do
      {s, s} -> d
      {[_, _], [_, 1]} -> T.reduce(:sum, d)
      {[_, _], [1, _]} -> T.transpose(T.reduce(:sum, T.transpose(d)))
      {[_, _], [1, 1]} -> T.reduce(:sum, T.transpose(T.reduce(:sum, d)))
      {[_], [1]} -> T.reduce(:sum, d)
    end
  end

  defp splat?({:splat, _}), do: true
  defp splat?(_), do: false

  # 1 in the shape of x (x·0 + 1 is exactly 1 for finite x)
  defp ones_like(x), do: T.add(T.mul(x, T.splat(0.0)), T.splat(1.0))
  defp zeros_like(x), do: T.mul(x, T.splat(0.0))
end
