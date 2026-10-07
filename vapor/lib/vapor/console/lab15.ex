defmodule Vapor.Console.Lab15 do
  @moduledoc """
  The **Opus** desks of the console (docs/CONSOLE.md §0.15), the same calls
  the terminal (`vapor rebis|aludel|tabula|cupel|amalgam`) and the MCP
  server make: circuits proved equal or told apart (Rebis), polynomial
  claims and barrier certificates decided exactly (Aludel), contracts
  checked for antinomies (Tabula), silent data corruption caught by the
  adjoint identity (Cupel), and sums that do not depend on their order
  (Amalgam). Every request is bounded; every answer carries what lets
  someone else check it.
  """
  import Bitwise
  alias Vapor.{Aludel, Amalgam, Cupel, F32, Rebis, Tabula, Tensor}
  alias Vapor.Rebis.{Gen, Ideal, Stabilizer}
  alias Vapor.Logic.LP

  @max_text 200_000

  # ------------------------------------------------------------ examples

  @full_adder """
  # a full adder
  input a b cin
  output s cout
  s = a ^ b ^ cin
  cout = maj(a, b, cin)
  """

  @contract """
  # a sale, with an override for force majeure
  parties buyer seller
  facts delivered late defective force_majeure
  exclusive pay withhold
  assume not (late and not delivered)
  C1: if delivered and not defective then buyer must pay seller
  C2: if late then buyer may withhold
  C3: if defective then buyer must not pay
  C4: if force_majeure then seller is exempt from deliver
  C5: seller must deliver buyer
  C6: if late and delivered then buyer must withhold
  C4 overrides C5
  """

  @doc "What the desks offer: starting points for each."
  def info do
    %{
      rebis: [
        %{id: "adders", title_pt: "Dois somadores de 8 bits", about_pt: "ripple-carry contra Kogge–Stone: a mesma função, provada por tabela-verdade (2¹⁶ padrões)", title: "Two 8-bit adders", about: "ripple-carry against Kogge–Stone: the same function, proved by truth table (2¹⁶ patterns)",
          op: "equivalent", a: Gen.ripple(8), b: Gen.kogge_stone(8)},
        %{id: "adders16", title_pt: "Dois somadores de 16 bits", about_pt: "além da tabela-verdade: simulação aleatória, depois um miter cuja prova UNSAT (DRUP) é conferida", title: "Two 16-bit adders", about: "beyond the truth table: random simulation, then a miter whose UNSAT proof (DRUP) is checked",
          op: "equivalent", a: Gen.ripple(16), b: Gen.kogge_stone(16)},
        %{id: "trojan", title_pt: "Um cavalo de Troia de 32 bits", about_pt: "uma saída invertida quando a = 0xDEADBEEF: 4096 padrões aleatórios não o veem, o miter acha o gatilho, o encolhimento deixa só ele", title: "A 32-bit trojan", about: "one output flipped when a = 0xDEADBEEF: 4096 random patterns miss it, the miter finds the trigger, shrinking leaves exactly it",
          op: "equivalent", a: Gen.ripple(32), b: Gen.ripple(32, trojan: 0xDEADBEEF)},
        %{id: "multiplier", title_pt: "Um multiplicador de 8 bits", about_pt: "m = a·b provado por álgebra sobre ℤ (a base de Gröbner do circuito), onde o SAT é exponencial", title: "An 8-bit multiplier", about: "m = a·b proved by algebra over ℤ (the circuit's Gröbner basis), where SAT is exponential",
          op: "identity", a: Gen.multiplier(8), spec: "m[16] = a[8] * b[8]"},
        %{id: "anf", title_pt: "A FNA de um somador completo", about_pt: "o polinômio de Zhegalkin de cada saída, pela transformada de Möbius", title: "A full adder's ANF", about: "the Zhegalkin polynomial of each output, by the Möbius transform", op: "anf", a: @full_adder},
        %{id: "ghz", title_pt: "Um estado GHZ de 64 qubits", about_pt: "circuitos de Clifford pelo tableau de estabilizadores: um resultado aleatório, 63 que o seguem", title: "A 64-qubit GHZ state", about: "Clifford circuits by the stabilizer tableau: one random outcome, 63 that follow it",
          op: "stabilizer", n: 64, a: "h 0\n" <> Enum.map_join(1..63, "\n", &"cx 0 #{&1}") <> "\n" <> Enum.map_join(0..63, "\n", &"m #{&1}")}
      ],
      aludel: [
        %{id: "motzkin", title_pt: "Motzkin + 1/1000 > 0", about_pt: "não negativo mas não é soma de quadrados: a subdivisão de Bernstein certifica a forma estrita numa caixa", title: "Motzkin + 1/1000 > 0", about: "non-negative but no sum of squares: Bernstein subdivision certifies the strict form on a box",
          op: "decide", vars: "x, y", poly: "x^4*y^2 + x^2*y^4 - 3*x^2*y^2 + 1 + 1/1000", box: [["-2", "2"], ["-2", "2"]], sense: "pos"},
        %{id: "motzkin0", title_pt: "Motzkin ≥ 0", about_pt: "seus zeros ficam onde os envoltórios de Bernstein não fecham: o veredito é 'esgotado', com a célula — nunca um palpite", title: "Motzkin ≥ 0", about: "its zeros sit where Bernstein enclosures cannot close: the verdict is 'exhausted', with the cell — never a guess",
          op: "decide", vars: "x, y", poly: "x^4*y^2 + x^2*y^4 - 3*x^2*y^2 + 1", box: [["-2", "2"], ["-2", "2"]], sense: "nonneg"},
        %{id: "refute", title_pt: "Uma afirmação que falha", about_pt: "refutada num vértice exato com o seu valor exato", title: "A claim that fails", about: "refuted at an exact vertex with its exact value",
          op: "decide", vars: "x, y", poly: "x^2*y - 3/2 + y^2", box: [["-1", "1"], ["-1", "1"]], sense: "nonneg"},
        %{id: "oscillator", title_pt: "Um oscilador amortecido, barreira dada", about_pt: "ẋ = y, ẏ = −x − y: B = x² + y² − 1 separa o conjunto inicial do inseguro", title: "A damped oscillator, barrier given", about: "ẋ = y, ẏ = −x − y: B = x² + y² − 1 separates the initial from the unsafe set",
          op: "barrier", vars: "x, y", field: ["y", "-x - y"], domain: [["-2", "2"], ["-2", "2"]], init: [["-1/2", "1/2"], ["-1/2", "1/2"]],
          unsafe: [["3/2", "2"], ["3/2", "2"]], barrier: "x^2 + y^2 - 1"},
        %{id: "synth", title_pt: "Um sistema não linear, barreira achada", about_pt: "ẋ = −x + y², ẏ = −y: um LP exato propõe B, a decisão aceita", title: "A nonlinear system, barrier found", about: "ẋ = −x + y², ẏ = −y: an exact LP proposes B, the decision accepts it",
          op: "barrier", vars: "x, y", field: ["-x + y^2", "-y"], domain: [["-1", "1"], ["-1", "1"]], init: [["-1/4", "1/4"], ["-1/4", "1/4"]],
          unsafe: [["3/4", "1"], ["3/4", "1"]], synthesize: true, degree: 2}
      ],
      tabula: [%{id: "sale", title_pt: "Um contrato de compra e venda", about_pt: "antinomias com o cenário que as dispara, provas de consistência, uma prevalência, lacunas", title: "A sale contract", about: "antinomies with the scenario that triggers them, proofs of consistency, an override, gaps", text: @contract}],
      cupel: [%{id: "drill", title_pt: "Um exercício de falha", about_pt: "inverta um bit de um produto correto: quais posições a aritmética correta nunca produziria?", title: "A fault drill", about: "flip one bit of a correct product: which bit positions can correct arithmetic never produce?", n: 32, k: 64, seed: 5}],
      amalgam: [%{id: "cancel", title_pt: "Cancelamento", about_pt: "1e16, 1, −1e16, 1 … em ordens diferentes: as somas da esquerda para a direita discordam; a amálgama, não", title: "Cancellation", about: "1e16, 1, −1e16, 1 … in different orders: left-to-right float sums disagree; the amalgam does not",
                  numbers: "1e16 1 -1e16 1 3.25 -2.5e-3 7e15 -7e15 0.1 0.2 0.3", format: "f64"}]
    }
  end

  defp text(req, key) do
    case req[key] do
      t when is_binary(t) and byte_size(t) <= @max_text -> {:ok, t}
      t when is_binary(t) -> {:error, "#{key}: at most #{div(@max_text, 1000)} kB"}
      _ -> {:error, "#{key}: a string"}
    end
  end

  defp int(req, key, default, lo, hi) do
    case req[key] do
      nil -> default
      v when is_integer(v) -> v |> max(lo) |> min(hi)
      v when is_binary(v) -> (case Integer.parse(v) do {i, _} -> i |> max(lo) |> min(hi); :error -> default end)
      _ -> default
    end
  end

  # ================================================================ Rebis

  @doc "Circuits: `op` equivalent | anf | identity | stabilizer | aiger."
  def rebis(req) do
    case req["op"] do
      "equivalent" ->
        with {:ok, a} <- circuit(req, "a"), {:ok, b} <- circuit(req, "b") do
          t0 = System.monotonic_time(:millisecond)
          r = Rebis.equivalent(a, b, conflicts: int(req, "conflicts", 2_000_000, 1, 5_000_000))
          ms = System.monotonic_time(:millisecond) - t0

          case r do
            {:equivalent, ev} -> {:ok, %{verdict: "equivalent", evidence: ev, ms: ms, a: Rebis.stats(a), b: Rebis.stats(b)}}
            {:different, d} -> {:ok, %{verdict: "different", method: d.method, counterexample: d.counterexample, a_out: d.a, b_out: d.b, ones: d.ones, ms: ms,
                                        a: Rebis.stats(a), b: Rebis.stats(b)}}
            {:unknown, why} -> {:ok, %{verdict: "unknown", why: why, ms: ms}}
            {:error, why} -> {:error, why}
          end
        end

      "anf" ->
        with {:ok, a} <- circuit(req, "a") do
          if length(a.inputs) > 16 do
            {:error, "#{length(a.inputs)} inputs: the ANF is shown up to 16 (equivalence beyond that goes to SAT)"}
          else
            outs = a |> Rebis.anf(limit: 48) |> Enum.map(fn {o, x} -> %{output: o, degree: x.degree, terms: x.terms, text: x.text} end) |> Enum.sort_by(& &1.output)
            {:ok, %{inputs: a.inputs, outputs: outs}}
          end
        end

      "identity" ->
        with {:ok, a} <- circuit(req, "a"), {:ok, sp} <- text(req, "spec"), {:ok, terms} <- Ideal.spec(sp) do
          case Ideal.prove(a, terms, max_terms: 400_000) do
            {:proved, st} -> {:ok, %{verdict: "proved", spec: sp, stats: st}}
            {:refuted, r} -> {:ok, %{verdict: "refuted", spec: sp, counterexample: r.counterexample, value: r.value, stats: Map.drop(r, [:counterexample, :value])}}
            {:unknown, why} -> {:ok, %{verdict: "unknown", why: why}}
            {:error, why} -> {:error, why}
          end
        end

      "stabilizer" ->
        with {:ok, t} <- text(req, "a") do
          n = int(req, "n", 2, 1, 2_000)

          case Stabilizer.run(t, n, seed: int(req, "seed", 1, 1, 1 <<< 40)) do
            {:ok, r} ->
              {:ok, %{n: n, outcomes: Enum.take(r.outcomes, 512), kinds: Enum.take(r.kinds, 512), measured: length(r.outcomes),
                      stabilizers: if(n <= 16, do: Stabilizer.stabilizers(r.state), else: nil)}}

            e ->
              e
          end
        end

      "aiger" ->
        with {:ok, a} <- circuit(req, "a"), do: {:ok, %{aiger: Rebis.to_aiger(a)}}

      _ ->
        {:error, "op: equivalent, anf, identity, stabilizer or aiger"}
    end
  end

  defp circuit(req, key) do
    with {:ok, t} <- text(req, key) do
      result = if String.starts_with?(String.trim_leading(t), "aag "), do: Rebis.from_aiger(t), else: Rebis.parse(t)

      case result do
        {:ok, c} -> {:ok, c}
        {:error, why} -> {:error, "#{key}: #{why}"}
      end
    end
  end

  # ================================================================ Aludel

  @doc "Polynomials on boxes: `op` decide | enclose | barrier."
  def aludel(req) do
    with {:ok, vars} <- vars(req) do
      case req["op"] do
        "decide" ->
          with {:ok, p} <- poly(req, "poly", vars), {:ok, box} <- box(req, "box", vars) do
            sense = if req["sense"] == "pos", do: :pos, else: :nonneg
            opts = [sense: sense, depth: int(req, "depth", 20, 1, 30), cells: int(req, "cells", 20_000, 1, 100_000)]
            t0 = System.monotonic_time(:millisecond)
            r = Aludel.decide(p, box, opts)
            ms = System.monotonic_time(:millisecond) - t0
            {:ok, Map.merge(decision(r, p, box, sense, vars), %{ms: ms, polynomial: Aludel.to_text(p, vars)})}
          end

        "enclose" ->
          with {:ok, p} <- poly(req, "poly", vars), {:ok, box} <- box(req, "box", vars) do
            {lo, hi} = Aludel.enclose(p, box, int(req, "depth", 3, 0, 6))
            {:ok, %{lo: LP.show(lo), hi: LP.show(hi), lo_f: LP.to_float(lo), hi_f: LP.to_float(hi)}}
          end

        "barrier" ->
          barrier(req, vars)

        _ ->
          {:error, "op: decide, enclose or barrier"}
      end
    end
  end

  defp vars(req) do
    names = (req["vars"] || "x, y") |> to_string() |> String.split(~r/[\s,]+/, trim: true)

    cond do
      names == [] or length(names) > 4 -> {:error, "vars: one to four names"}
      Enum.any?(names, &(not Regex.match?(~r/^[A-Za-z_][A-Za-z0-9_]{0,15}$/, &1))) -> {:error, "vars: names of letters, digits and _"}
      length(Enum.uniq(names)) != length(names) -> {:error, "vars: a name twice"}
      true -> {:ok, names}
    end
  end

  defp poly(req, key, vars) do
    with {:ok, t} <- text(req, key) do
      case Aludel.parse(t, vars) do
        {:ok, p} -> {:ok, p}
        {:error, why} -> {:error, "#{key}: #{why}"}
      end
    end
  end

  defp box(req, key, vars) do
    case req[key] do
      pairs when is_list(pairs) and length(pairs) == length(vars) ->
        try do
          Aludel.box(Enum.map(pairs, fn [lo, hi] -> {rat(lo), rat(hi)} end))
        rescue
          _ -> {:error, "one [lo, hi] per variable, numbers or fractions"}
        end
        |> case do
          {:ok, b} -> {:ok, b}
          {:error, why} -> {:error, "#{key}: #{why}"}
        end

      _ ->
        {:error, "#{key}: one [lo, hi] per variable (#{Enum.join(vars, ", ")})"}
    end
  end

  defp rat(v) when is_integer(v), do: {v, 1}
  defp rat(v) when is_float(v), do: LP.rat(v)
  defp rat(v) when is_binary(v), do: LP.rat(String.trim(v))

  defp decision({:certified, w}, p, box, sense, vars) do
    check = Aludel.check(p, box, w.witness, sense: sense)
    cells = if length(vars) == 2 and w.cells <= 4_000, do: cells_f(w.witness, box), else: nil
    %{verdict: "certified", cells: w.cells, depth: w.depth, witness: w.witness, replayed: check == :ok, leaves: cells}
  end

  defp decision({:refuted, r}, _p, _box, _sense, vars) do
    %{verdict: "refuted", point: Enum.zip(vars, Enum.map(r.point, &LP.show/1)) |> Map.new(), point_f: Enum.map(r.point, &LP.to_float/1),
      value: LP.show(r.value), value_f: LP.to_float(r.value)}
  end

  defp decision({:exhausted, r}, _p, _box, _sense, _vars) do
    %{verdict: "exhausted", cell: Enum.map(r.cell, fn {lo, hi} -> [LP.show(lo), LP.show(hi)] end),
      cell_f: Enum.map(r.cell, fn {lo, hi} -> [LP.to_float(lo), LP.to_float(hi)] end), cells: r.cells, depth: r.depth}
  end

  defp decision({:error, why}, _p, _box, _sense, _vars), do: %{verdict: "error", why: why}

  defp cells_f(witness, box) do
    case Aludel.cells(witness, box) do
      {:ok, cs} -> Enum.map(cs, fn c -> Enum.flat_map(c, fn {lo, hi} -> [LP.to_float(lo), LP.to_float(hi)] end) end)
      _ -> nil
    end
  end

  defp barrier(req, vars) do
    with {:ok, field} <- field(req, vars),
         {:ok, d} <- box(req, "domain", vars), {:ok, i} <- box(req, "init", vars), {:ok, u} <- box(req, "unsafe", vars) do
      sys = %{vars: vars, field: field, domain: unq(d), init: unq(i), unsafe: unq(u)}
      lambda = req["lambda"] || 0
      budget = [depth: int(req, "depth", 16, 1, 24), cells: int(req, "cells", 20_000, 1, 100_000)]

      result =
        if req["synthesize"] == true do
          case Aludel.synthesize(sys, [degree: int(req, "degree", 2, 1, 4), lambda: rat(lambda), split: int(req, "split", 1, 0, 2)] ++ budget) do
            {:ok, b, rep} -> {:ok, b, rep}
            {:error, why} -> {:error, why}
          end
        else
          with {:ok, b} <- poly(req, "barrier", vars), do: {:ok, b, Aludel.barrier(sys, b, [lambda: rat(lambda)] ++ budget)}
        end

      case result do
        {:ok, b, rep} ->
          conds = Enum.map(rep.conditions, fn c -> %{name: c.name, claim: c.claim, result: decision(c.result, c.polynomial, c.box, c.sense, [])} end)
          plot = if length(vars) == 2, do: plot(field, b, d), else: nil
          {:ok, %{verdict: Atom.to_string(rep.verdict), barrier: Aludel.to_text(b, vars), conditions: conds, plot: plot,
                  lp: Map.take(rep, [:lp_rounds, :lp_rows, :lp_rows_total])}}

        {:error, why} ->
          {:ok, %{verdict: "not found", why: why}}
      end
    end
  end

  defp unq(box), do: Enum.map(box, fn {lo, hi} -> {lo, hi} end)

  defp field(req, vars) do
    case req["field"] do
      fs when is_list(fs) and length(fs) == length(vars) ->
        Enum.reduce_while(Enum.with_index(fs), {:ok, []}, fn {f, k}, {:ok, acc} ->
          case is_binary(f) && Aludel.parse(f, vars) do
            {:ok, p} -> {:cont, {:ok, acc ++ [p]}}
            {:error, why} -> {:halt, {:error, "field[#{k}]: #{why}"}}
            _ -> {:halt, {:error, "field[#{k}]: a polynomial"}}
          end
        end)

      _ ->
        {:error, "field: one polynomial per variable (ẋᵢ = fᵢ)"}
    end
  end

  # a picture's worth of floats: the field on a 15×15 grid, B on a 41×41 grid
  defp plot(field, b, [{x0, x1}, {y0, y1}]) do
    {a0, a1, b0, b1} = {LP.to_float(x0), LP.to_float(x1), LP.to_float(y0), LP.to_float(y1)}
    grid = fn n, f -> for j <- 0..(n - 1), i <- 0..(n - 1), do: f.(a0 + (a1 - a0) * i / (n - 1), b0 + (b1 - b0) * j / (n - 1)) end
    %{box: [a0, a1, b0, b1],
      arrows: grid.(15, fn x, y -> Enum.map(field, &Float.round(Aludel.eval_float(&1, [x, y]), 6)) end),
      b: %{n: 41, values: grid.(41, fn x, y -> Float.round(Aludel.eval_float(b, [x, y]), 6) end)}}
  end

  # ================================================================ Tabula

  @doc "A contract: the analysis, and the positions under `facts` (a map of fact → boolean) when given."
  def tabula(req) do
    with {:ok, t} <- text(req, "text"), {:ok, tab} <- Tabula.parse(t) do
      a = Tabula.analyze(tab)

      pos =
        case req["facts"] do
          f when is_map(f) -> (case Tabula.positions(tab, f) do {:ok, p} -> p; {:error, why} -> %{error: why} end)
          _ -> nil
        end

      {:ok, %{verdict: Atom.to_string(a.verdict), findings: a.findings, resolved: a.resolved, silences: a.silences, checked_pairs: a.checked_pairs,
              clauses: Enum.map(tab.clauses, &%{id: &1.id, text: &1.text, party: &1.party, modality: Tabula.modality_text(&1.modality), action: &1.action}),
              facts: tab.facts, parties: tab.parties, positions: pos}}
    end
  end

  # ================================================================ Cupel

  @doc """
  A fault drill on `x·Wᵀ` (W: n × k, random, seeded): the detection rate of
  a flipped bit at every position (f32), and the same for the exact int8
  check; one example flip at `bit`.
  """
  def cupel(req) do
    n = int(req, "n", 32, 4, 128)
    k = int(req, "k", 64, 16, 256) |> div(16) |> Kernel.*(16)
    seed = int(req, "seed", 5, 1, 1_000_000)
    trials = int(req, "trials", 8, 2, 24)
    bit = int(req, "bit", 26, 0, 31)
    w = Tensor.random(:f32, [n, k], seed)
    xs = for s <- 1..trials, do: Tensor.random(:f32, [2, k], seed * 1000 + s)
    t0 = System.monotonic_time(:millisecond)
    profile = Cupel.sensitivity(w, xs, seed: seed + 7) |> Enum.map(fn {b, hit, tot} -> %{bit: b, detected: hit, trials: tot} end)
    p = Cupel.probe(w, seed: seed + 7)
    x = hd(xs)
    y = Cupel.oracle_linear(w, x)
    {_, clean} = Cupel.assay(p, x, y)
    {verdict, flipped} = Cupel.assay(p, x, Cupel.flip(y, 3, bit))
    old = y.data |> binary_part(12, 4) |> then(fn <<v::32-little>> -> v end)

    # int8: exact, every bit
    wi = Tensor.from_list(:s8, [n, k], for(i <- 1..(n * k), do: rem(i * 37 + seed, 255) - 127))
    xi = Tensor.from_list(:s8, [2, k], for(i <- 1..(2 * k), do: rem(i * 91 + seed, 255) - 127))
    yi = Cupel.int_linear(wi, xi)
    pi = Cupel.probe(wi, seed: seed)
    int_bits = Enum.count(0..31, fn b -> match?({:corrupt, _}, Cupel.assay(pi, xi, Cupel.flip(yi, 1, b))) end)

    {:ok, %{n: n, k: k, trials: trials, profile: profile, ms: System.monotonic_time(:millisecond) - t0,
            example: %{bit: bit, element: 3, before: F32.to_float(old), after: F32.to_float(bxor(old, 1 <<< bit)), verdict: Atom.to_string(verdict),
                       worst_ratio: ratio(flipped.worst), clean_ratio: ratio(clean.worst)},
            int8: %{bits_detected: int_bits, of: 32},
            check_cost: "O(b·(n + k)) per product after an O(n·k) probe, against O(b·n·k) for the product"}}
  end

  defp ratio(:infinity), do: "infinity"
  defp ratio(r), do: r

  # ================================================================ Amalgam

  @doc "Sum a list of numbers in several orders, left to right, and once exactly (`format` f32 or f64)."
  def amalgam(req) do
    fmt = if req["format"] == "f32", do: :f32, else: :f64

    with {:ok, t} <- text(req, "numbers") do
      toks = String.split(t, ~r/[\s,;]+/, trim: true)

      parsed =
        Enum.map(toks, fn s ->
          case Float.parse(s) do
            {f, ""} -> f
            _ -> (case Integer.parse(s) do {i, ""} -> i * 1.0; _ -> nil end)
          end
        end)

      cond do
        toks == [] -> {:error, "numbers: at least one"}
        length(toks) > 20_000 -> {:error, "numbers: at most 20 000"}
        Enum.any?(parsed, &is_nil/1) -> {:error, "numbers: #{Enum.at(toks, Enum.find_index(parsed, &is_nil/1))} is not a number"}
        true -> {:ok, sums(parsed, fmt)}
      end
    end
  end

  defp sums(xs, fmt) do
    bits = Enum.map(xs, &to_bits(&1, fmt))
    add = fn a, b -> to_bits(from_bits(a, fmt) + from_bits(b, fmt), fmt) end
    lr = fn list -> Enum.reduce(list, to_bits(0.0, fmt), &add.(&2, &1)) end
    :rand.seed(:exsss, {7, 7, 7})
    shuffles = for k <- 1..4, do: {"shuffled #{k}", Enum.shuffle(bits)}

    orders =
      [{"left to right", bits}, {"right to left", Enum.reverse(bits)}, {"ascending", Enum.sort_by(bits, &from_bits(&1, fmt))},
       {"descending", Enum.sort_by(bits, &from_bits(&1, fmt), :desc)}, {"by magnitude", Enum.sort_by(bits, &abs(from_bits(&1, fmt)))}] ++ shuffles

    naive = Enum.map(orders, fn {name, l} -> r = safe(fn -> lr.(l) end); %{order: name, value: show(r, fmt), bits: hex(r, fmt)} end)
    a = Enum.reduce(bits, Amalgam.new(fmt, 1), &Amalgam.add(&2, [&1]))
    exact = hd(a.cells)
    r = a |> Amalgam.round_bits() |> hd()

    %{format: Atom.to_string(fmt), count: length(xs), naive: naive, distinct_naive: naive |> Enum.map(& &1.bits) |> Enum.uniq() |> length(),
      amalgam: %{value: show(r, fmt), bits: hex(r, fmt)}, exact: exact_text(exact, a.scale)}
  end

  defp safe(f) do
    f.()
  rescue
    ArithmeticError -> :overflow
  end

  defp to_bits(x, :f64) do
    <<b::64>> = <<x::float-64>>
    b
  end

  defp to_bits(x, :f32), do: F32.from_float(x)

  defp from_bits(b, :f64) do
    <<x::float-64>> = <<b::64>>
    x
  end

  defp from_bits(b, :f32), do: F32.to_float(b)

  defp show(:overflow, _), do: "overflow"
  defp show(b, :f64) when (b >>> 52 &&& 0x7FF) == 0x7FF, do: if((b &&& 0xF_FFFF_FFFF_FFFF) != 0, do: "NaN", else: if(b >>> 63 == 1, do: "-inf", else: "inf"))
  defp show(b, :f32) when (b >>> 23 &&& 0xFF) == 0xFF, do: if((b &&& 0x7F_FFFF) != 0, do: "NaN", else: if(b >>> 31 == 1, do: "-inf", else: "inf"))
  defp show(b, fmt), do: from_bits(b, fmt) |> :erlang.float_to_binary([:short])

  defp hex(:overflow, _), do: "—"
  defp hex(b, :f64), do: "0x" <> String.pad_leading(Integer.to_string(b, 16), 16, "0")
  defp hex(b, :f32), do: "0x" <> String.pad_leading(Integer.to_string(b, 16), 8, "0")

  # the exact sum as a decimal (the cell is an integer multiple of 2^-scale; every such number has a finite decimal)
  defp exact_text(c, _scale) when not is_integer(c), do: to_string(c)
  defp exact_text(0, _), do: "0"

  defp exact_text(c, scale) do
    sign = if c < 0, do: "-", else: ""
    m = abs(c) * Integer.pow(5, scale)
    s = Integer.to_string(m) |> String.pad_leading(scale + 1, "0")
    {int, frac} = String.split_at(s, String.length(s) - scale)
    frac = String.trim_trailing(frac, "0")
    digits = if frac == "", do: int, else: int <> "." <> frac
    if String.length(digits) > 120, do: sign <> String.slice(digits, 0, 120) <> "…", else: sign <> digits
  end
end
