defmodule Vapor.Rebis.Ideal do
  @moduledoc """
  Word-level identities of a circuit proved by **algebra over ℤ** — the
  procedure where the round's "Gröbner bases" proposal is right, once
  corrected in one detail.

  The detail: over GF(2) the ANF decides equivalence but says nothing about
  *arithmetic* (that 64 output wires are the product of two 32-bit words).
  Over ℤ with the Boolean constraint `x² = x`, every gate is a polynomial —
  `¬a = 1 − a`, `a∧b = ab`, `a∨b = a + b − ab`, `a⊕b = a + b − 2ab`,
  `mux(s,a,b) = sa + b − sb` — and under a lexicographic order that puts
  every gate above its inputs (reverse topological), the gate polynomials
  already **are** a Gröbner basis of the circuit's ideal (their leading
  terms are distinct variables). Reducing a specification by them is just
  substituting gates backwards, from the outputs to the inputs: the
  remainder is `0` iff the identity holds for every input (Lv, Kalla &
  Enescu 2013; Ritirc, Biere & Kauers 2017).

  Why it matters: multiplier commutativity is exponential for resolution
  (measured in 0.15: CDCL with a checked DRUP proof needs 2 963 conflicts at
  5 bits and does not finish in minutes at 6), while backward rewriting of
  `Σ2ᵏmₖ − (Σ2ⁱaᵢ)(Σ2ʲbⱼ)` through an array multiplier stays polynomial.

  A non-zero remainder is a polynomial over the inputs; its monomial of
  least degree, set to 1 with everything else 0, is a point where it does
  not vanish — a **counterexample**, re-evaluated on the circuit before it
  is reported. Monomials are BEAM integers (bit `i` = node `i`), so a
  product of monomials is an OR.
  """
  import Bitwise
  alias Vapor.Rebis
  alias Vapor.Rebis.Circuit

  @doc "`Σ 2ⁱ · prefixᵢ` for `i < n`: a word as a polynomial (`[{coeff, [names]}]`)."
  def word(prefix, n), do: for(i <- 0..(n - 1), do: {1 <<< i, ["#{prefix}#{i}"]})

  @doc "Product of two polynomials given as term lists (names multiply, `x² = x`)."
  def mul(p, q), do: for({c1, n1} <- p, {c2, n2} <- q, do: {c1 * c2, Enum.uniq(n1 ++ n2)})

  @doc "Difference of two term lists."
  def sub(p, q), do: p ++ Enum.map(q, fn {c, ns} -> {-c, ns} end)

  @doc """
  Prove that the polynomial `spec` (terms over input and output names)
  vanishes on every input of `circuit`. `{:proved, stats}`,
  `{:refuted, %{counterexample, value}}` (the value of `spec` there, ≠ 0,
  computed by simulating the circuit), or `{:error, why}`. Option
  `max_terms:` (default 2 000 000) bounds the intermediate polynomial —
  past it the answer is `{:unknown, …}`, never a guess.
  """
  def prove(%Circuit{} = c, spec, opts \\ []) do
    names = Map.new(c.outputs) |> Map.merge(input_nodes(c))
    max = Keyword.get(opts, :max_terms, 2_000_000)

    with {:ok, poly} <- to_poly(spec, names) do
      inputs = c.inputs |> Enum.map(&names[&1]) |> MapSet.new()
      t0 = System.monotonic_time(:millisecond)

      result =
        Enum.reduce_while((c.size - 1)..0//-1, {poly, 0, map_size(poly)}, fn g, {p, steps, peak} ->
          cond do
            MapSet.member?(inputs, g) -> {:cont, {p, steps, peak}}
            true ->
              p = substitute(p, g, gate_poly(elem(c.nodes, g)))
              size = map_size(p)
              if size > max, do: {:halt, {:unknown, size}}, else: {:cont, {p, steps + 1, max(peak, size)}}
          end
        end)

      ms = System.monotonic_time(:millisecond) - t0

      case result do
        {:unknown, size} -> {:unknown, "the intermediate polynomial passed #{size} terms"}
        {p, steps, peak} when map_size(p) == 0 -> {:proved, %{substitutions: steps, peak_terms: peak, ms: ms}}
        {p, steps, peak} -> refute(c, spec, p, %{substitutions: steps, peak_terms: peak, ms: ms})
      end
    end
  end

  defp input_nodes(c) do
    for i <- 0..(c.size - 1), match?({:in, _}, elem(c.nodes, i)), into: %{}, do: {elem(elem(c.nodes, i), 1), i}
  end

  defp to_poly(spec, names) do
    Enum.reduce_while(spec, {:ok, %{}}, fn {coef, ns}, {:ok, acc} ->
      case Enum.find(ns, &(not Map.has_key?(names, &1))) do
        nil ->
          m = Enum.reduce(ns, 0, fn n, m -> m ||| 1 <<< names[n] end)
          {:cont, {:ok, add_term(acc, m, coef)}}

        missing ->
          {:halt, {:error, "#{missing} is neither an input nor an output of the circuit"}}
      end
    end)
  end

  defp add_term(p, _m, 0), do: p

  defp add_term(p, m, c) do
    case Map.get(p, m, 0) + c do
      0 -> Map.delete(p, m)
      v -> Map.put(p, m, v)
    end
  end

  # the gate as a polynomial over its fanins: [{monomial, coeff}]
  defp gate_poly({:const, v}), do: [{0, v}]
  defp gate_poly({:not, a}), do: [{0, 1}, {1 <<< a, -1}]
  defp gate_poly({:and, a, b}), do: [{1 <<< a ||| 1 <<< b, 1}]
  defp gate_poly({:or, a, b}), do: [{1 <<< a, 1}, {1 <<< b, 1}, {1 <<< a ||| 1 <<< b, -1}]
  defp gate_poly({:xor, a, b}) when a == b, do: [{0, 0}]
  defp gate_poly({:xor, a, b}), do: [{1 <<< a, 1}, {1 <<< b, 1}, {1 <<< a ||| 1 <<< b, -2}]
  defp gate_poly({:mux, s, a, b}), do: [{1 <<< s ||| 1 <<< a, 1}, {1 <<< b, 1}, {1 <<< s ||| 1 <<< b, -1}]

  # replace variable g by its gate polynomial in every monomial that contains it
  defp substitute(p, g, gp) do
    bit = 1 <<< g

    Enum.reduce(p, p, fn {m, c}, acc ->
      if (m &&& bit) == 0 do
        acc
      else
        rest = bxor(m, bit)
        acc = Map.delete(acc, m)
        Enum.reduce(gp, acc, fn {gm, gc}, a -> add_term(a, rest ||| gm, c * gc) end)
      end
    end)
  end

  defp refute(c, spec, p, stats) do
    # a monomial of least degree: setting exactly its variables to 1 gives its coefficient
    {m, _} = Enum.min_by(p, fn {m, _} -> {Rebis.popcount(m), m} end)
    nodes = input_nodes(c)
    cex = Map.new(nodes, fn {n, i} -> {n, m >>> i &&& 1} end)
    value = evaluate(c, spec, cex)
    if value == 0, do: raise("Rebis.Ideal: a non-zero remainder gave a point where the specification vanishes")
    {:refuted, Map.merge(stats, %{counterexample: cex, value: value, remainder_terms: map_size(p)})}
  end

  @doc "The value of `spec` on the circuit at one input assignment (by simulation)."
  def evaluate(c, spec, cex) do
    vals = Map.merge(cex, Rebis.eval(c, cex))
    Enum.reduce(spec, 0, fn {coef, ns}, acc -> acc + coef * Enum.reduce(ns, 1, &(&2 * Map.fetch!(vals, &1))) end)
  end
end
