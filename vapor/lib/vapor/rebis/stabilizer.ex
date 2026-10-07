defmodule Vapor.Rebis.Stabilizer do
  @moduledoc """
  Clifford circuits on thousands of qubits, exactly, on a classical machine
  (Gottesman–Knill): the state is not `2ⁿ` amplitudes but `2n` Pauli
  operators that stabilize it — a binary matrix over the symplectic space
  GF(2)²ⁿ plus a sign per row (Aaronson & Gottesman, *Improved simulation
  of stabilizer circuits*, 2004, the "CHP" tableau).

  Each row is two BEAM integers (`x`, `z`: bit `j` is qubit `j`) and a
  sign bit, so a row operation is a handful of word-parallel XORs. The
  phase of a row product (`rowsum`, the `g` function of the paper) is
  computed for all qubits at once: the qubits where the product gains `+i`
  and `−i` are two bit masks, and the phase is the difference of their
  popcounts mod 4 — no loop over qubits.

  What this is **not**: a universal quantum simulator. A `T` gate leaves
  the stabilizer formalism; it is refused by name. The measured outcomes
  of random measurements come from a seed, so a run is reproducible.

  Circuits as text, one gate per line: `h q`, `s q`, `sdg q`, `x q`, `y q`,
  `z q`, `cx c t` (also `cnot`), `cz a b`, `swap a b`, `m q` (measure in
  the Z basis), `reset q`; `#` comments.
  """
  import Bitwise

  defstruct n: 0, rows: {}, seed: 1, outcomes: []

  @doc "|0…0⟩ on `n` qubits: destabilizers Xᵢ, stabilizers Zᵢ."
  def new(n, opts \\ []) when is_integer(n) and n > 0 do
    rows = for(i <- 0..(n - 1), do: {1 <<< i, 0, 0}) ++ for(i <- 0..(n - 1), do: {0, 1 <<< i, 0}) ++ [{0, 0, 0}]
    %__MODULE__{n: n, rows: List.to_tuple(rows), seed: Keyword.get(opts, :seed, 1) &&& 0xFFFF_FFFF_FFFF_FFFF}
  end

  # ------------------------------------------------------------- gates

  @doc "Hadamard on qubit `a`."
  def h(st, a), do: map_rows(st, fn {x, z, r} -> {xa, za} = {bit(x, a), bit(z, a)}; {set(x, a, za), set(z, a, xa), bxor(r, xa &&& za)} end)
  @doc "Phase gate S on qubit `a`."
  def s(st, a), do: map_rows(st, fn {x, z, r} -> {xa, za} = {bit(x, a), bit(z, a)}; {x, set(z, a, bxor(za, xa)), bxor(r, xa &&& za)} end)
  @doc "S† = S³."
  def sdg(st, a), do: st |> s(a) |> s(a) |> s(a)
  @doc "Pauli X on qubit `a` (flips the sign of rows with Z there)."
  def x(st, a), do: map_rows(st, fn {x, z, r} -> {x, z, bxor(r, bit(z, a))} end)
  @doc "Pauli Z on qubit `a`."
  def z(st, a), do: map_rows(st, fn {x, z, r} -> {x, z, bxor(r, bit(x, a))} end)
  @doc "Pauli Y on qubit `a`."
  def y(st, a), do: map_rows(st, fn {x, z, r} -> {x, z, bxor(r, bxor(bit(x, a), bit(z, a)))} end)

  @doc "CNOT with control `a`, target `b`."
  def cx(_st, a, a), do: raise(ArgumentError, "cx on a single qubit")

  def cx(st, a, b) do
    map_rows(st, fn {x, z, r} ->
      {xa, za, xb, zb} = {bit(x, a), bit(z, a), bit(x, b), bit(z, b)}
      r = bxor(r, xa &&& zb &&& bxor(bxor(xb, za), 1))
      {set(x, b, bxor(xb, xa)), set(z, a, bxor(za, zb)), r}
    end)
  end

  @doc "CZ = (I⊗H)·CX·(I⊗H)."
  def cz(st, a, b), do: st |> h(b) |> cx(a, b) |> h(b)
  @doc "SWAP = three CNOTs."
  def swap(st, a, b), do: st |> cx(a, b) |> cx(b, a) |> cx(a, b)

  defp map_rows(%__MODULE__{rows: rows} = st, f), do: %{st | rows: rows |> Tuple.to_list() |> Enum.map(f) |> List.to_tuple()}
  defp bit(v, a), do: v >>> a &&& 1
  defp set(v, a, 1), do: v ||| 1 <<< a
  defp set(v, a, 0), do: v &&& bnot(1 <<< a)

  # -------------------------------------------------------- measurement

  @doc """
  Measure qubit `a` in the Z basis: `{outcome, state, kind}` with `kind`
  `:deterministic` or `:random` (the outcome then drawn from the seed, or
  forced by `force: 0 | 1`).
  """
  def measure(%__MODULE__{n: n, rows: rows} = st, a, opts \\ []) do
    case Enum.find(n..(2 * n - 1), &(bit(elem(rows, &1) |> elem(0), a) == 1)) do
      nil ->
        # deterministic: the scratch row accumulates the stabilizers whose destabilizer has X on a
        scratch = 2 * n
        rows = put_elem(rows, scratch, {0, 0, 0})

        rows =
          Enum.reduce(0..(n - 1), rows, fn i, rows ->
            if bit(elem(elem(rows, i), 0), a) == 1, do: put_elem(rows, scratch, rowsum(elem(rows, scratch), elem(rows, i + n), true)), else: rows
          end)

        {_, _, out} = elem(rows, scratch)
        {out, %{st | rows: rows, outcomes: [out | st.outcomes]}, :deterministic}

      p ->
        {out, seed} =
          case Keyword.fetch(opts, :force) do
            {:ok, v} when v in [0, 1] -> {v, st.seed}
            :error -> draw(st.seed)
          end

        rp = elem(rows, p)

        rows =
          Enum.reduce(0..(2 * n - 1), rows, fn i, rows ->
            # destabilizer phases are not tracked (their product with a stabilizer may be imaginary)
            if i != p and bit(elem(elem(rows, i), 0), a) == 1, do: put_elem(rows, i, rowsum(elem(rows, i), rp, i >= n)), else: rows
          end)

        rows = rows |> put_elem(p - n, elem(rows, p)) |> put_elem(p, {0, 1 <<< a, out})
        {out, %{st | rows: rows, seed: seed, outcomes: [out | st.outcomes]}, :random}
    end
  end

  @doc "Reset qubit `a` to |0⟩ (measure, then X if the outcome was 1)."
  def reset(st, a) do
    {out, st, _} = measure(st, a)
    st = %{st | outcomes: tl(st.outcomes)}
    if out == 1, do: x(st, a), else: st
  end

  # row h ← row i · row h, with the phase of the product (the g function, bit-parallel)
  defp rowsum({x2, z2, r2}, {x1, z1, r1}, strict) do
    y1 = x1 &&& z1
    xo = x1 &&& bnot(z1)
    zo = z1 &&& bnot(x1)
    plus = (y1 &&& z2 &&& bnot(x2)) ||| (xo &&& z2 &&& x2) ||| (zo &&& x2 &&& bnot(z2))
    minus = (y1 &&& x2 &&& bnot(z2)) ||| (xo &&& z2 &&& bnot(x2)) ||| (zo &&& x2 &&& z2)
    phase = Integer.mod(2 * r2 + 2 * r1 + popcount(plus) - popcount(minus), 4)
    if strict and phase not in [0, 2], do: raise("Stabilizer: odd phase in a product of stabilizers (tableau corrupted)")
    {bxor(x1, x2), bxor(z1, z2), div(phase, 2) &&& 1}
  end

  @popc for b <- 0..255, do: Enum.sum(for i <- 0..7, do: b >>> i &&& 1)
  @popt List.to_tuple(@popc)

  defp popcount(0), do: 0
  defp popcount(v), do: for(<<b <- :binary.encode_unsigned(v)>>, reduce: 0, do: (acc -> acc + elem(@popt, b)))

  defp draw(seed) do
    {v, s} = Vapor.Tensor.splitmix(seed)
    {v >>> 63 &&& 1, s}
  end

  # ---------------------------------------------------------- reading it

  @doc "The stabilizer generators as signed Pauli strings, qubit 0 first: `[\"+XX\", \"-ZZ\"…]`."
  def stabilizers(%__MODULE__{n: n, rows: rows}), do: for(i <- n..(2 * n - 1), do: pauli(elem(rows, i), n))

  @doc "The destabilizers, same format."
  def destabilizers(%__MODULE__{n: n, rows: rows}), do: for(i <- 0..(n - 1), do: pauli(elem(rows, i), n))

  defp pauli({x, z, r}, n) do
    (if r == 1, do: "-", else: "+") <>
      for(j <- 0..(n - 1), into: "", do: (case {bit(x, j), bit(z, j)} do {0, 0} -> "I"; {1, 0} -> "X"; {0, 1} -> "Z"; {1, 1} -> "Y" end))
  end

  # ------------------------------------------------------------ circuits

  @doc """
  Parse and run a circuit on `n` qubits: `{:ok, %{outcomes, kinds, state}}`
  (outcomes in measurement order) or `{:error, why}`. Option `seed:`.
  """
  def run(text, n, opts \\ []) when is_binary(text) do
    with {:ok, ops} <- parse(text, n) do
      {st, kinds} =
        Enum.reduce(ops, {new(n, opts), []}, fn
          {:m, q}, {st, ks} -> {_, st, k} = measure(st, q); {st, [k | ks]}
          {:reset, q}, {st, ks} -> {reset(st, q), ks}
          {g, q}, {st, ks} -> {apply(__MODULE__, g, [st, q]), ks}
          {g, a, b}, {st, ks} -> {apply(__MODULE__, g, [st, a, b]), ks}
        end)

      {:ok, %{outcomes: Enum.reverse(st.outcomes), kinds: Enum.reverse(kinds), state: st}}
    end
  end

  @one %{"h" => :h, "s" => :s, "sdg" => :sdg, "x" => :x, "y" => :y, "z" => :z, "m" => :m, "reset" => :reset}
  @two %{"cx" => :cx, "cnot" => :cx, "cz" => :cz, "swap" => :swap}

  @doc "Parse a circuit: `{:ok, [op]}` or `{:error, why}` (non-Clifford gates refused by name)."
  def parse(text, n) do
    text
    |> String.split("\n")
    |> Enum.with_index(1)
    |> Enum.map(fn {l, i} -> {l |> String.split("#", parts: 2) |> hd() |> String.trim() |> String.downcase(), i} end)
    |> Enum.reject(&(elem(&1, 0) == ""))
    |> Enum.reduce_while({:ok, []}, fn {line, ln}, {:ok, acc} ->
      case op(String.split(line), n) do
        {:ok, o} -> {:cont, {:ok, [o | acc]}}
        {:error, why} -> {:halt, {:error, "line #{ln}: #{why}"}}
      end
    end)
    |> case do
      {:ok, ops} -> {:ok, Enum.reverse(ops)}
      e -> e
    end
  end

  defp op([g, q], n) when is_map_key(@one, g), do: with({:ok, q} <- qubit(q, n), do: {:ok, {@one[g], q}})

  defp op([g, a, b], n) when is_map_key(@two, g) do
    with {:ok, a} <- qubit(a, n), {:ok, b} <- qubit(b, n), true <- a != b || {:error, "#{g} on one qubit"}, do: {:ok, {@two[g], a, b}}
  end

  defp op([g | _], _n) when g in ~w(t tdg ccx toffoli rx ry rz u u3),
    do: {:error, "#{g} is not a Clifford gate: it leaves the stabilizer formalism (Gottesman–Knill covers H, S, CNOT and Paulis)"}

  defp op(words, _n), do: {:error, "not a gate: #{Enum.join(words, " ") |> String.slice(0, 30)}"}

  defp qubit(s, n) do
    case Integer.parse(s) do
      {q, ""} when q >= 0 and q < n -> {:ok, q}
      _ -> {:error, "qubit #{inspect(String.slice(s, 0, 12))} is not in 0…#{n - 1}"}
    end
  end
end
