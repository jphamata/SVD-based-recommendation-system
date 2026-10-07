defmodule Vapor.Rebis do
  @moduledoc """
  The **rebis** — the alchemists' "double thing", two natures shown to be
  one — for combinational circuits over GF(2) (docs/REBIS.md).

  The pain: a netlist after synthesis, a third-party IP block, a chip back
  from the foundry — is it the specification? Simulation answers for the
  patterns it tried; a hardware trojan whose trigger is a 64-bit
  coincidence survives every test bench ever run. Equivalence is a
  theorem or it is nothing.

  The algebra, stated exactly. Over GF(2), XOR is `+`, AND is `·`, NOT is
  `+1`; every Boolean function of `n` inputs has **one** multilinear
  polynomial — its algebraic normal form (Zhegalkin) — in
  `GF(2)[x₁…xₙ]/⟨xᵢ² − xᵢ⟩`. Two circuits are equivalent iff their ANFs are
  equal. The proposal this round came with ("prove `P_A − P_B ≡ 0` with
  Gröbner bases") is correct and the wrong tool: the ANF *is* the normal
  form modulo that ideal, computed by the Möbius transform in `O(n·2ⁿ)`
  word operations — no Buchberger — and beyond ~20 inputs no normal form
  is cheap (equivalence is coNP-complete). So, two procedures, each with
  a checkable output:

    * **`n ≤ 16`** (default): the truth table of every output as one BEAM
      integer of `2ⁿ` bits (all patterns at once), compared; the ANF by
      Möbius in `n` shift-and-xor steps. A difference is a counterexample.
    * **beyond**: 4096 random patterns at once (a counterexample is
      usually found here), then a **miter** — `OR(outᵃᵢ ⊕ outᵇᵢ)` — by
      Tseitin into CNF for `Vapor.Logic.SAT`: UNSAT comes with a DRUP
      proof that `Vapor.Logic.DRUP` (code sharing nothing with the solver)
      checks before the answer is "equivalent"; SAT comes with a model,
      re-simulated on both circuits before it is called a counterexample.

  For **word-level arithmetic** (that 2n wires are the product of two
  words) GF(2) is the wrong ring: `Vapor.Rebis.Ideal` proves such identities
  over ℤ by reducing the specification modulo the circuit's own Gröbner
  basis — polynomial for multipliers, where CDCL is exponential (and the
  other way round for parallel-prefix adders: the two procedures are
  complementary, measured in docs/REBIS.md).

  A counterexample is **shrunk** (fewest inputs at 1, greedily) — what the
  property-testing tradition calls shrinking — so the trigger of a trojan
  reads as the trigger.

  Input is a small netlist language (`input`, `output`, `w = a & ~b ^ c`,
  `mux(s, a, b)`, `maj(a, b, c)`) or AIGER ASCII (`aag`, the format of
  the hardware model-checking competitions). Names never become atoms;
  sizes are bounded.
  """
  import Bitwise
  alias Vapor.Logic.{DRUP, SAT}

  defmodule Circuit do
    @moduledoc """
    A combinational circuit: `nodes` in topological order, each
    `{:in, name} | {:const, 0 | 1} | {:not, a} | {op, a, b}` (`op` ∈ `:and
    :or :xor`) or `{:mux, s, a, b}` with operands as node indices;
    `outputs` are `{name, index}`.
    """
    defstruct inputs: [], outputs: [], nodes: {}, size: 0
  end

  @max_nodes 500_000
  @max_name 64

  # ===================================================== the text format

  @doc """
  Parse a netlist:

      # a full adder
      input a b cin
      output s cout
      s = a ^ b ^ cin
      cout = maj(a, b, cin)

  Operators by precedence: `~`/`!` (not), `&`, `^`, `|`; constants `0`,
  `1`; functions `mux(s, a, b)` (s ? a : b), `maj(a, b, c)`. A wire is
  defined once, before use. `{:ok, circuit}` or `{:error, why}`.
  """
  def parse(text) when is_binary(text) do
    lines =
      text
      |> String.split("\n")
      |> Enum.with_index(1)
      |> Enum.map(fn {l, i} -> {l |> String.split("#", parts: 2) |> hd() |> String.trim(), i} end)
      |> Enum.reject(&(elem(&1, 0) == ""))

    st = %{nodes: [], count: 0, names: %{}, inputs: [], outputs: [], memo: %{}}

    Enum.reduce_while(lines, {:ok, st}, fn {line, ln}, {:ok, st} ->
      case statement(line, st) do
        {:ok, st} -> {:cont, {:ok, st}}
        {:error, why} -> {:halt, {:error, "line #{ln}: #{why}"}}
      end
    end)
    |> case do
      {:ok, st} -> finish(st)
      e -> e
    end
  catch
    {:rebis, why} -> {:error, why}
  end

  defp statement(line, st) do
    cond do
      Regex.match?(~r/^inputs?\s/, line) ->
        line |> String.split() |> tl() |> Enum.reduce_while({:ok, st}, fn n, {:ok, st} ->
          with :ok <- name_ok(n), true <- not Map.has_key?(st.names, n) || {:error, "#{n} defined twice"} do
            {i, st} = node(st, {:in, n})
            {:cont, {:ok, %{st | names: Map.put(st.names, n, i), inputs: st.inputs ++ [n]}}}
          else
            e -> {:halt, e}
          end
        end)

      Regex.match?(~r/^outputs?\s/, line) ->
        outs = line |> String.split() |> tl()
        case Enum.find(outs, &(name_ok(&1) != :ok)) do
          nil -> {:ok, %{st | outputs: st.outputs ++ outs}}
          bad -> {:error, "bad output name #{inspect(bad)}"}
        end

      String.contains?(line, "=") ->
        [lhs, rhs] = String.split(line, "=", parts: 2)
        lhs = String.trim(lhs)

        with :ok <- name_ok(lhs),
             true <- not Map.has_key?(st.names, lhs) || {:error, "#{lhs} defined twice"},
             {:ok, toks} <- lex(rhs),
             {:ok, i, st, []} <- expr(toks, st) do
          {:ok, %{st | names: Map.put(st.names, lhs, i)}}
        else
          {:ok, _, _, [t | _]} -> {:error, "unexpected #{inspect(t)}"}
          {:error, _} = e -> e
        end

      true ->
        {:error, "not a statement: #{String.slice(line, 0, 40)}"}
    end
  end

  defp name_ok(n) do
    if Regex.match?(~r/^[A-Za-z_][A-Za-z0-9_\[\]\.]*$/, n) and byte_size(n) <= @max_name,
      do: :ok,
      else: {:error, "bad name #{inspect(String.slice(n, 0, 20))}"}
  end

  defp lex(s), do: lex(String.to_charlist(s), [])
  defp lex([], acc), do: {:ok, Enum.reverse(acc)}
  defp lex([c | r], acc) when c in [?\s, ?\t], do: lex(r, acc)
  defp lex([c | r], acc) when c in [?~, ?!], do: lex(r, [:not | acc])
  defp lex([?& | r], acc), do: lex(r, [:and | acc])
  defp lex([?| | r], acc), do: lex(r, [:or | acc])
  defp lex([?^ | r], acc), do: lex(r, [:xor | acc])
  defp lex([?( | r], acc), do: lex(r, [:lp | acc])
  defp lex([?) | r], acc), do: lex(r, [:rp | acc])
  defp lex([?, | r], acc), do: lex(r, [:comma | acc])

  defp lex([c | _] = s, acc) when c in ?a..?z or c in ?A..?Z or c == ?_ or c in ?0..?9 do
    {word, rest} = Enum.split_while(s, &(&1 in ?a..?z or &1 in ?A..?Z or &1 in ?0..?9 or &1 in [?_, ?[, ?], ?.]))
    lex(rest, [{:w, List.to_string(word)} | acc])
  end

  defp lex([c | _], _), do: {:error, "unexpected character #{inspect(<<c::utf8>>)}"}

  # | over ^ over & over unary
  defp expr(t, st), do: binop(t, st, [:or, :xor, :and])

  defp binop(t, st, []), do: unary(t, st)

  defp binop(t, st, [op | lower] = ops) do
    with {:ok, a, st, rest} <- binop(t, st, lower), do: binop_rest(a, st, rest, op, ops, lower)
  end

  defp binop_rest(a, st, [op | rest], op, ops, lower) do
    with {:ok, b, st, rest} <- binop(rest, st, lower) do
      {i, st} = node(st, {op, a, b})
      binop_rest(i, st, rest, op, ops, lower)
    end
  end

  defp binop_rest(a, st, rest, _op, _ops, _lower), do: {:ok, a, st, rest}

  defp unary([:not | r], st) do
    with {:ok, a, st, rest} <- unary(r, st) do
      {i, st} = node(st, {:not, a})
      {:ok, i, st, rest}
    end
  end

  defp unary([:lp | r], st) do
    case expr(r, st) do
      {:ok, a, st, [:rp | rest]} -> {:ok, a, st, rest}
      {:ok, _, _, _} -> {:error, "missing )"}
      e -> e
    end
  end

  defp unary([{:w, f}, :lp | r], st) when f in ["mux", "maj"] do
    with {:ok, a, st, [:comma | r]} <- expr(r, st),
         {:ok, b, st, [:comma | r]} <- expr(r, st),
         {:ok, c, st, [:rp | rest]} <- expr(r, st) do
      {i, st} =
        case f do
          "mux" -> node(st, {:mux, a, b, c})
          "maj" ->
            {ab, st} = node(st, {:and, a, b})
            {ac, st} = node(st, {:and, a, c})
            {bc, st} = node(st, {:and, b, c})
            {o, st} = node(st, {:or, ab, ac})
            node(st, {:or, o, bc})
        end

      {:ok, i, st, rest}
    else
      {:ok, _, _, _} -> {:error, "#{f} takes three arguments"}
      e -> e
    end
  end

  defp unary([{:w, "0"} | r], st), do: (fn {i, st} -> {:ok, i, st, r} end).(node(st, {:const, 0}))
  defp unary([{:w, "1"} | r], st), do: (fn {i, st} -> {:ok, i, st, r} end).(node(st, {:const, 1}))

  defp unary([{:w, n} | r], st) do
    case Map.fetch(st.names, n) do
      {:ok, i} -> {:ok, i, st, r}
      :error -> {:error, "#{n} used before it is defined"}
    end
  end

  defp unary([t | _], _st), do: {:error, "unexpected #{inspect(t)}"}
  defp unary([], _st), do: {:error, "expression ends early"}

  # hash-consing: a structurally equal node is the same node
  defp node(st, n) do
    case Map.fetch(st.memo, n) do
      {:ok, i} when elem(n, 0) != :in -> {i, st}
      _ ->
        if st.count >= @max_nodes, do: throw({:rebis, "more than #{@max_nodes} nodes"})
        {st.count, %{st | nodes: [n | st.nodes], count: st.count + 1, memo: Map.put(st.memo, n, st.count)}}
    end
  end

  defp finish(st) do
    case Enum.find(st.outputs, &(not Map.has_key?(st.names, &1))) do
      nil ->
        if st.outputs == [], do: {:error, "no outputs"},
        else: {:ok, %Circuit{inputs: st.inputs, outputs: Enum.map(st.outputs, &{&1, st.names[&1]}), nodes: st.nodes |> Enum.reverse() |> List.to_tuple(), size: st.count}}

      missing ->
        {:error, "output #{missing} is never defined"}
    end
  catch
    {:rebis, why} -> {:error, why}
  end

  @doc "Parse or raise (for generators and tests)."
  def parse!(text) do
    case parse(text) do
      {:ok, c} -> c
      {:error, why} -> raise ArgumentError, why
    end
  catch
    {:rebis, why} -> raise ArgumentError, why
  end

  # ====================================================== AIGER (ASCII)

  @doc """
  Read AIGER ASCII (`aag M I L O A`): combinational only (latches are
  refused — this is equivalence of combinational logic). Symbol lines
  name inputs and outputs; unnamed ones are `i0…`, `o0…`.
  """
  def from_aiger(text) do
    lines = text |> String.split("\n") |> Enum.map(&String.trim/1)

    with [hdr | rest] <- lines,
         ["aag" | nums] <- String.split(hdr),
         [m, ni, nl, no, na] <- Enum.map(nums, &int/1),
         true <- Enum.all?([m, ni, nl, no, na], &(is_integer(&1) and &1 >= 0)) || {:error, "bad header"},
         true <- nl == 0 || {:error, "#{nl} latches: only combinational circuits are compared"},
         true <- m + 1 <= @max_nodes || {:error, "too large"},
         {ins, rest} <- Enum.split(rest, ni),
         {outs, rest} <- Enum.split(rest, no),
         {ands, rest} <- Enum.split(rest, na),
         ins = Enum.map(ins, &int/1),
         outs = Enum.map(outs, &int/1),
         ands = Enum.map(ands, fn l -> l |> String.split() |> Enum.map(&int/1) end),
         true <- Enum.all?(ins ++ outs, &is_integer/1) and Enum.all?(ands, &match?([_, _, _], &1)) || {:error, "malformed body"} do
      syms =
        for l <- rest, m = Regex.run(~r/^([io])(\d+)\s+(\S.{0,63})$/, l), m != nil, into: %{} do
          [_, k, idx, name] = m
          {{k, String.to_integer(idx)}, String.trim(name)}
        end

      in_names = for {_, i} <- Enum.with_index(ins), do: Map.get(syms, {"i", i}, "i#{i}")
      out_names = for {_, i} <- Enum.with_index(outs), do: Map.get(syms, {"o", i}, "o#{i}")

      cond do
        length(Enum.uniq(in_names)) != length(in_names) -> {:error, "two inputs share a name"}
        length(Enum.uniq(out_names)) != length(out_names) -> {:error, "two outputs share a name"}
        true -> build_aiger(ins, outs, ands, in_names, out_names)
      end
    else
      {:error, _} = e -> e
      _ -> {:error, "not AIGER ASCII (aag)"}
    end
  end

  defp int(s) do
    case Integer.parse(s) do
      {v, ""} -> v
      _ -> nil
    end
  end

  defp build_aiger(ins, outs, ands, in_names, out_names) do
    st = %{nodes: [], count: 0, names: %{}, inputs: [], outputs: [], memo: %{}}
    {false_i, st} = node(st, {:const, 0})
    {st, var} = Enum.zip(ins, in_names) |> Enum.reduce({st, %{0 => false_i}}, fn {lit, n}, {st, var} ->
      {i, st} = node(st, {:in, n})
      {%{st | inputs: st.inputs ++ [n]}, Map.put(var, div(lit, 2), i)}
    end)

    lit = fn st, var, l ->
      case Map.fetch(var, div(l, 2)) do
        {:ok, i} -> if rem(l, 2) == 1, do: node(st, {:not, i}), else: {i, st}
        :error -> throw({:rebis, "literal #{l} used before it is defined"})
      end
    end

    # AIGER ASCII allows gates in any order: Kahn's algorithm over the variables they define
    {st, var} = topo_ands(ands, st, var, lit)

    {outs_i, st} = Enum.map_reduce(outs, st, fn l, st -> lit.(st, var, l) end)
    {:ok, %Circuit{inputs: st.inputs, outputs: Enum.zip(out_names, outs_i), nodes: st.nodes |> Enum.reverse() |> List.to_tuple(), size: st.count}}
  catch
    {:rebis, why} -> {:error, why}
  end

  defp topo_ands(ands, st, var, lit) do
    by_lhs = Map.new(ands, fn [lhs, a, b] -> {div(lhs, 2), {lhs, a, b}} end)
    if map_size(by_lhs) != length(ands), do: throw({:rebis, "a variable is defined by two AND gates"})
    if Enum.any?(ands, fn [lhs, _, _] -> rem(lhs, 2) == 1 or Map.has_key?(var, div(lhs, 2)) end), do: throw({:rebis, "an AND gate redefines an input or a negated literal"})
    deps = Map.new(by_lhs, fn {v, {_, a, b}} -> {v, Enum.uniq(for l <- [a, b], Map.has_key?(by_lhs, div(l, 2)), do: div(l, 2))} end)
    users = Enum.reduce(deps, %{}, fn {v, ds}, acc -> Enum.reduce(ds, acc, &Map.update(&2, &1, [v], fn us -> [v | us] end)) end)
    pending = Map.new(deps, fn {v, ds} -> {v, length(ds)} end)
    ready = for {v, 0} <- pending, do: v

    {st, var, done} = kahn(Enum.sort(ready), pending, users, by_lhs, st, var, lit, 0)
    if done != map_size(by_lhs), do: throw({:rebis, "the AND gates form a cycle"})
    {st, var}
  end

  defp kahn([], _pending, _users, _by, st, var, _lit, done), do: {st, var, done}

  defp kahn([v | rest], pending, users, by, st, var, lit, done) do
    {_lhs, a, b} = by[v]
    {ia, st} = lit.(st, var, a)
    {ib, st} = lit.(st, var, b)
    {g, st} = node(st, {:and, ia, ib})
    var = Map.put(var, v, g)

    {pending, newly} =
      Enum.reduce(Map.get(users, v, []), {pending, []}, fn u, {p, nw} ->
        p = Map.update!(p, u, &(&1 - 1))
        if p[u] == 0, do: {p, [u | nw]}, else: {p, nw}
      end)

    kahn(rest ++ Enum.sort(newly), pending, users, by, st, var, lit, done + 1)
  end

  @doc "Write a circuit as AIGER ASCII (AND/NOT only; XOR, OR and MUX expanded)."
  def to_aiger(%Circuit{} = c) do
    # literal of every node: 2·var or 2·var+1
    {lits, ands, next} =
      Enum.reduce(0..(c.size - 1), {%{}, [], length(c.inputs) + 1}, fn i, {lits, ands, nv} ->
        case elem(c.nodes, i) do
          {:in, n} -> {Map.put(lits, i, 2 * (Enum.find_index(c.inputs, &(&1 == n)) + 1)), ands, nv}
          {:const, v} -> {Map.put(lits, i, v), ands, nv}
          {:not, a} -> {Map.put(lits, i, bxor(lits[a], 1)), ands, nv}
          {:and, a, b} -> {Map.put(lits, i, 2 * nv), [{2 * nv, lits[a], lits[b]} | ands], nv + 1}
          {:or, a, b} -> {Map.put(lits, i, 2 * nv + 1), [{2 * nv, bxor(lits[a], 1), bxor(lits[b], 1)} | ands], nv + 1}
          {:xor, a, b} -> xor_aig(lits, ands, nv, i, lits[a], lits[b])
          {:mux, s, a, b} ->
            # s·a + ¬s·b = ¬(¬(s·a) · ¬(¬s·b))
            g1 = {2 * nv, lits[s], lits[a]}
            g2 = {2 * (nv + 1), bxor(lits[s], 1), lits[b]}
            g3 = {2 * (nv + 2), 2 * nv + 1, 2 * (nv + 1) + 1}
            {Map.put(lits, i, 2 * (nv + 2) + 1), [g3, g2, g1 | ands], nv + 3}
        end
      end)

    ands = Enum.reverse(ands)
    m = next - 1
    head = "aag #{m} #{length(c.inputs)} 0 #{length(c.outputs)} #{length(ands)}\n"
    ins = Enum.map_join(1..length(c.inputs)//1, "", &"#{2 * &1}\n")
    outs = Enum.map_join(c.outputs, "", fn {_, i} -> "#{lits[i]}\n" end)
    gates = Enum.map_join(ands, "", fn {l, a, b} -> "#{l} #{a} #{b}\n" end)
    syms = Enum.map_join(Enum.with_index(c.inputs), "", fn {n, i} -> "i#{i} #{n}\n" end) <> Enum.map_join(Enum.with_index(c.outputs), "", fn {{n, _}, i} -> "o#{i} #{n}\n" end)
    head <> ins <> outs <> gates <> syms
  end

  # a ⊕ b = ¬(a·b) · ¬(¬a·¬b)
  defp xor_aig(lits, ands, nv, i, a, b) do
    g1 = {2 * nv, a, b}
    g2 = {2 * (nv + 1), bxor(a, 1), bxor(b, 1)}
    g3 = {2 * (nv + 2), 2 * nv + 1, 2 * (nv + 1) + 1}
    {Map.put(lits, i, 2 * (nv + 2)), [g3, g2, g1 | ands], nv + 3}
  end

  # ====================================================== simulation

  @doc """
  Evaluate on words: `inputs` maps each input name to a non-negative
  integer whose bit `j` is that input in pattern `j`; `mask` is
  `2^patterns − 1`. Returns `%{output_name => word}`. Every pattern at
  once — the BEAM integer is the bit vector.
  """
  def simulate(%Circuit{} = c, inputs, mask) do
    vals =
      Enum.reduce(0..(c.size - 1)//1, %{}, fn i, v ->
        w =
          case elem(c.nodes, i) do
            {:in, n} -> Map.fetch!(inputs, n) &&& mask
            {:const, 0} -> 0
            {:const, 1} -> mask
            {:not, a} -> bxor(v[a], mask)
            {:and, a, b} -> v[a] &&& v[b]
            {:or, a, b} -> v[a] ||| v[b]
            {:xor, a, b} -> bxor(v[a], v[b])
            {:mux, s, a, b} -> (v[s] &&& v[a]) ||| (bxor(v[s], mask) &&& v[b])
          end

        Map.put(v, i, w)
      end)

    Map.new(c.outputs, fn {n, i} -> {n, vals[i]} end)
  end

  @doc "Evaluate one assignment `%{name => 0 | 1}`: `%{output => 0 | 1}`."
  def eval(%Circuit{} = c, assignment), do: simulate(c, assignment, 1)

  @doc "The word of input `i` of `n` in a full truth table: bit `j` = bit `i` of `j`."
  def projection(i, n) when i < n do
    block = 1 <<< (i + 1)
    unit = ((1 <<< (1 <<< i)) - 1) <<< (1 <<< i)
    total = 1 <<< n
    double(unit, block, total)
  end

  defp double(w, len, total) when len >= total, do: w
  defp double(w, len, total), do: double(w ||| w <<< len, 2 * len, total)

  @doc "Truth tables of every output (`2ⁿ`-bit integers, pattern `j` = the inputs as the bits of `j` in declaration order)."
  def truth_tables(%Circuit{inputs: ins} = c) do
    n = length(ins)
    if n > 24, do: raise(ArgumentError, "#{n} inputs: a truth table of 2^#{n} bits is not computed (use equivalent/3, which goes to SAT)")
    words = ins |> Enum.with_index() |> Map.new(fn {name, i} -> {name, projection(i, n)} end)
    simulate(c, words, (1 <<< (1 <<< n)) - 1)
  end

  # ====================================================== the ANF

  @doc """
  Algebraic normal form of a truth table of `n` inputs by the Möbius
  transform (`n` shift-and-xor steps over one integer): bit `m` of the
  result is the coefficient of the monomial `∏_{i ∈ m} xᵢ`.
  """
  def mobius(table, n) do
    mask = (1 <<< (1 <<< n)) - 1

    Enum.reduce(0..(n - 1)//1, table, fn i, f ->
      low = bxor(projection(i, n), mask)
      bxor(f, (f &&& low) <<< (1 <<< i)) &&& mask
    end)
  end

  @doc """
  The ANF of every output: `%{output => %{monomials, degree, terms, text}}`
  — `text` like `x0·x2 + x1 + 1` with the input names, cut at `limit`
  monomials (default 64).
  """
  def anf(%Circuit{inputs: ins} = c, opts \\ []) do
    n = length(ins)
    limit = Keyword.get(opts, :limit, 64)

    c
    |> truth_tables()
    |> Map.new(fn {o, t} ->
      coeffs = mobius(t, n)
      monos = set_bits(coeffs)
      deg = monos |> Enum.map(&popcount/1) |> Enum.max(fn -> 0 end)
      {o, %{monomials: monos, degree: deg, terms: length(monos), text: anf_text(monos, ins, limit)}}
    end)
  end

  defp anf_text([], _ins, _), do: "0"

  defp anf_text(monos, ins, limit) do
    shown = monos |> Enum.sort_by(&{-popcount(&1), &1}) |> Enum.take(limit)
    body = Enum.map_join(shown, " + ", fn
      0 -> "1"
      m -> ins |> Enum.with_index() |> Enum.filter(fn {_, i} -> (m >>> i &&& 1) == 1 end) |> Enum.map_join("·", &elem(&1, 0))
    end)
    if length(monos) > limit, do: body <> " + … (#{length(monos) - limit} more)", else: body
  end

  @doc "Positions of the set bits of a non-negative integer, ascending."
  def set_bits(0), do: []

  def set_bits(x) when x > 0 do
    bin = :binary.encode_unsigned(x, :little)

    {acc, _} =
      for <<byte <- bin>>, reduce: {[], 0} do
        {acc, base} ->
          acc = if byte == 0, do: acc, else: Enum.reduce(0..7, acc, fn b, a -> if (byte >>> b &&& 1) == 1, do: [base + b | a], else: a end)
          {acc, base + 8}
      end

    Enum.reverse(acc)
  end

  @doc "Number of set bits."
  def popcount(x), do: popcount(x, 0)
  defp popcount(0, n), do: n
  defp popcount(x, n), do: popcount(x &&& x - 1, n + 1)

  # ====================================================== equivalence

  @doc """
  Are `a` and `b` the same function? Inputs are matched by name, outputs
  by name (or position, when the names differ but the counts agree).

  `{:equivalent, evidence}` — `evidence.method` is `:truth_table` (with
  the SHA-256 of the shared tables) or `:sat` (with the DRUP proof's
  lemma count, as checked); `{:different, %{counterexample, a, b}}` with
  the counterexample shrunk and re-simulated; `{:unknown, why}` when the
  SAT budget runs out (never a guess); `{:error, why}` when the
  interfaces differ.

  Options: `truth_table: 16` (largest `n` for the table), `patterns:
  4096`, `seed: 1`, `conflicts:` (SAT budget).
  """
  def equivalent(%Circuit{} = a, %Circuit{} = b, opts \\ []) do
    with {:ok, pairs} <- interface(a, b) do
      n = length(a.inputs)

      if n <= Keyword.get(opts, :truth_table, 16) do
        by_table(a, b, pairs, n)
      else
        by_search(a, b, pairs, opts)
      end
    end
  end

  defp interface(a, b) do
    an = Enum.map(a.outputs, &elem(&1, 0))
    bn = Enum.map(b.outputs, &elem(&1, 0))

    cond do
      Enum.sort(a.inputs) != Enum.sort(b.inputs) ->
        {:error, "inputs differ: #{inspect(a.inputs -- b.inputs)} vs #{inspect(b.inputs -- a.inputs)}"}

      Enum.sort(an) == Enum.sort(bn) ->
        {:ok, Enum.map(an, &{&1, &1})}

      length(an) == length(bn) ->
        {:ok, Enum.zip(an, bn)}

      true ->
        {:error, "#{length(an)} outputs vs #{length(bn)}"}
    end
  end

  defp by_table(a, b, pairs, n) do
    # b's tables over a's input order
    ta = truth_tables(a)
    tb = b |> reorder(a.inputs) |> truth_tables()
    diff = Enum.reduce(pairs, 0, fn {oa, ob}, acc -> acc ||| bxor(ta[oa], tb[ob]) end)

    if diff == 0 do
      digest = :crypto.hash(:sha256, Enum.map(pairs, fn {oa, _} -> :binary.encode_unsigned(ta[oa]) end)) |> Base.encode16(case: :lower)
      {:equivalent, %{method: :truth_table, inputs: n, patterns: 1 <<< n, tables_sha256: digest}}
    else
      # the counterexample with fewest ones, exactly (all patterns are known)
      j = diff |> set_bits() |> Enum.min_by(&{popcount(&1), &1})
      cex = a.inputs |> Enum.with_index() |> Map.new(fn {nm, i} -> {nm, j >>> i &&& 1} end)
      different(a, b, pairs, cex, :truth_table)
    end
  end

  # the same circuit with its inputs listed in another order (names unchanged)
  defp reorder(%Circuit{} = c, order), do: %{c | inputs: order}

  defp by_search(a, b, pairs, opts) do
    case random_cex(a, b, pairs, Keyword.get(opts, :patterns, 4096), Keyword.get(opts, :seed, 1)) do
      {:ok, cex} -> different(a, b, pairs, shrink(a, b, pairs, cex), :simulation)
      :none -> by_sat(a, b, pairs, opts)
    end
  end

  defp random_cex(a, b, pairs, k, seed) do
    :rand.seed(:exsss, {seed, 0xBEE, 0xCAFE})
    mask = (1 <<< k) - 1
    words = Map.new(a.inputs, fn n -> {n, rand_word(k)} end)
    oa = simulate(a, words, mask)
    ob = simulate(b, words, mask)
    diff = Enum.reduce(pairs, 0, fn {x, y}, acc -> acc ||| bxor(oa[x], ob[y]) end)

    if diff == 0 do
      :none
    else
      j = diff |> set_bits() |> hd()
      {:ok, Map.new(words, fn {n, w} -> {n, w >>> j &&& 1} end)}
    end
  end

  defp rand_word(k), do: :rand.bytes(div(k + 7, 8)) |> :binary.decode_unsigned() |> band((1 <<< k) - 1)

  defp by_sat(a, b, pairs, opts) do
    {cnf, _} = miter(a, b, pairs)

    case SAT.solve(cnf, conflicts: Keyword.get(opts, :conflicts, 2_000_000)) do
      {:unsat, proof, stats} ->
        case DRUP.check(cnf, proof) do
          {:ok, chk} ->
            {:equivalent, %{method: :sat, inputs: length(a.inputs), cnf_vars: cnf.vars, cnf_clauses: length(cnf.clauses),
                            proof_lemmas: chk.lemmas, checked_lemmas: chk.checked, conflicts: stats.conflicts,
                            cnf_sha256: :crypto.hash(:sha256, SAT.to_dimacs(cnf)) |> Base.encode16(case: :lower)}}

          {:error, why} ->
            # a solver that claims UNSAT with a proof that does not check is a bug, said as such
            {:unknown, "the SAT solver's proof did not check: #{inspect(why)}"}
        end

      {:sat, model, _} ->
        {_, var_of} = miter(a, b, pairs)
        cex = Map.new(a.inputs, fn n -> {n, if(Map.get(model, var_of[n], false), do: 1, else: 0)} end)
        different(a, b, pairs, shrink(a, b, pairs, cex), :sat)

      {:unknown, :budget, _} ->
        {:unknown, "SAT budget exhausted"}
    end
  end

  defp different(a, b, pairs, cex, method) do
    # the counterexample is re-simulated on both circuits, never trusted
    if not distinguishes?(a, b, pairs, cex), do: raise("Rebis: a #{method} counterexample does not distinguish the circuits")
    {:different, %{method: method, counterexample: cex, a: eval(a, cex), b: eval(b, cex), ones: cex |> Map.values() |> Enum.sum()}}
  end

  @doc """
  Shrink a distinguishing assignment: turn ones into zeros while the two
  circuits still disagree (greedy, in input order). The result is locally
  minimal — no single one can be cleared.
  """
  def shrink(a, b, pairs, cex) do
    Enum.reduce(a.inputs, cex, fn n, cur ->
      if cur[n] == 1 do
        try = Map.put(cur, n, 0)
        if distinguishes?(a, b, pairs, try), do: try, else: cur
      else
        cur
      end
    end)
  end

  defp distinguishes?(a, b, pairs, asg) do
    oa = eval(a, asg)
    ob = eval(b, asg)
    Enum.any?(pairs, fn {x, y} -> oa[x] != ob[y] end)
  end

  # ====================================================== the miter

  @doc """
  The miter of two circuits as CNF (Tseitin): satisfiable iff some input
  makes some output pair differ. Returns `{%{vars, clauses}, var_of_input}`.
  """
  def miter(a, b, pairs) do
    ivar = a.inputs |> Enum.with_index(1) |> Map.new()
    {la, next, ca} = tseitin(a, ivar, length(a.inputs) + 1)
    {lb, next, cb} = tseitin(b, ivar, next)
    outa = Map.new(a.outputs, fn {n, i} -> {n, la[i]} end)
    outb = Map.new(b.outputs, fn {n, i} -> {n, lb[i]} end)

    {ds, next, cd} =
      Enum.reduce(pairs, {[], next, []}, fn {x, y}, {ds, v, cs} ->
        {ds ++ [v], v + 1, xor_clauses(v, outa[x], outb[y]) ++ cs}
      end)

    # normal clauses for the solver: no repeated literal (x ⊕ x after
    # hash-consing gives one), no tautology
    clauses =
      (ca ++ cb ++ cd ++ [ds])
      |> Enum.map(&Enum.uniq/1)
      |> Enum.reject(fn c -> Enum.any?(c, &(-&1 in c)) end)

    {%{vars: next - 1, clauses: clauses}, ivar}
  end

  # literal of every node; constants as a fixed variable
  defp tseitin(c, ivar, first) do
    Enum.reduce(0..(c.size - 1)//1, {%{}, first, []}, fn i, {lit, v, cs} ->
      case elem(c.nodes, i) do
        {:in, n} -> {Map.put(lit, i, ivar[n]), v, cs}
        {:const, 1} -> {Map.put(lit, i, v), v + 1, [[v] | cs]}
        {:const, 0} -> {Map.put(lit, i, v), v + 1, [[-v] | cs]}
        {:not, x} -> {Map.put(lit, i, -lit[x]), v, cs}
        {:and, x, y} -> {Map.put(lit, i, v), v + 1, [[-v, lit[x]], [-v, lit[y]], [v, -lit[x], -lit[y]] | cs]}
        {:or, x, y} -> {Map.put(lit, i, v), v + 1, [[v, -lit[x]], [v, -lit[y]], [-v, lit[x], lit[y]] | cs]}
        {:xor, x, y} -> {Map.put(lit, i, v), v + 1, xor_clauses(v, lit[x], lit[y]) ++ cs}
        {:mux, s, x, y} ->
          {Map.put(lit, i, v), v + 1,
           [[-v, -lit[s], lit[x]], [v, -lit[s], -lit[x]], [-v, lit[s], lit[y]], [v, lit[s], -lit[y]] | cs]}
      end
    end)
  end

  # v ↔ a ⊕ b
  defp xor_clauses(v, a, b), do: [[-v, a, b], [-v, -a, -b], [v, -a, b], [v, a, -b]]

  # ====================================================== reporting

  @doc "Counts of a circuit: inputs, outputs, gates by kind."
  def stats(%Circuit{} = c) do
    kinds = c.nodes |> Tuple.to_list() |> Enum.frequencies_by(&elem(&1, 0))
    %{inputs: length(c.inputs), outputs: length(c.outputs), nodes: c.size, gates: kinds |> Map.drop([:in, :const]) |> Map.values() |> Enum.sum(), by_kind: kinds}
  end
end
