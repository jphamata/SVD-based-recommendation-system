defmodule Vapor.Mizan do
  @moduledoc """
  **Al-Mīzān** (الميزان, "the balance"; root و-ز-ن *w-z-n*, "to weigh" —
  files `.wzn`) — vapor's formal dialect for claims that must be *decided*,
  not trusted (docs/MIZAN.md). The manifesto that proposed it is kept where
  it is right and corrected where it is not (DIRETRIZ §19):

    * **One neutral tree, two projections.** The program is a tree; Latin
      and Arabic are bijective printings of it (`Vapor.Mizan.Syntax`). Its
      identity is the hash of the tree, the same in both scripts.
    * **Roots are domains, weights are regimes** (the morphological type
      system, kept): the root says *what kind of claim* —
      `ح-س-ب` H-s-b arithmetic, `ح-ف-ظ` H-f-Z conservation, `ن-ق-ل` n-q-l
      transition, `ك-ت-ب` k-t-b record — and the weight says *how it may be
      used*: `فاعل` fāʿil (transient: a pure function, lowered to vapor's
      compiler), `مفعول` mafʿūl (persistent: its value is stored and named
      by its hash), `برهان` burhān (proved: nothing runs until the
      obligation is discharged). Some pairs are refused: a conservation law
      without a proof is not a conservation law.
    * **Proofs by decision procedures, not by a person at an ITP.** Every
      obligation the language can state has a decider vapor already ships,
      with a certificate checked by separate code: polynomial identities
      over ℚ (exact normal form), positivity and bounds on a box
      (`Vapor.Aludel`: Bernstein, a replayable witness), invariants of
      Boolean transition systems (`Vapor.Logic.Formula`: SAT with a DRUP
      proof). The verdict is *proved*, *refuted* with a counterexample, or
      *unknown* — and *unknown* does not compile. The same statements can be
      exported to Lean 4 (`Vapor.Mizan.Lower.lean/2`) for anyone who wants a
      second, independent kernel.
    * **Abjad is shown, not used as an address**: the gematria of a root
      collides (every anagram, and most roots besides — measured in
      `Vapor.Mizan.Abjad`); identity is the hash.
  """
  alias Vapor.Aludel
  alias Vapor.Logic.{Formula, LP}
  alias Vapor.Mizan.Syntax, as: S

  @types ~w(q int f64 f32 bool)
  @wazns ~w(fail maful burhan)
  @arith ~w(+ - * / ^)
  @cmp ~w(< <= > >= =)
  @logic ~w(and or not if)
  @max_decls 512

  # ================================================================ parsing

  @doc "Text (either projection) → `{:ok, module}` or `{:error, why}` (with a line)."
  def parse(text) do
    with {:ok, forms} <- S.read(text),
         true <- length(forms) <= @max_decls || {:error, "at most #{@max_decls} declarations"} do
      Enum.reduce_while(forms, {:ok, []}, fn f, {:ok, acc} ->
        case decl(f) do
          {:ok, d} -> {:cont, {:ok, acc ++ [d]}}
          e -> {:halt, e}
        end
      end)
      |> case do
        {:ok, decls} -> check_module(%{"mizan" => 1, "decls" => decls})
        e -> e
      end
    end
  catch
    {:mzn, line, why} -> {:error, "line #{line}: #{why}"}
  end

  defp decl({:list, [{:atom, w, line} | rest], _}) do
    case S.keyword(w) do
      "claim" -> claim(rest, line)
      "import" -> import_decl(rest, line)
      _ -> throw({:mzn, line, "a declaration starts with claim/دعوى or import/استيراد, not #{inspect(w)}"})
    end
  end

  defp decl({_, _, line}), do: throw({:mzn, line, "expected a declaration in parentheses"})

  defp import_decl([{:atom, h, _}, {:atom, as_kw, _}, {:atom, name, _}], line) do
    if S.keyword(as_kw) != "as", do: throw({:mzn, line, "import HASH as NAME"})
    unless Regex.match?(~r/^[0-9a-f]{64}$/, h), do: throw({:mzn, line, "import names a module by its 64-hex-digit hash"})
    {:ok, %{"import" => h, "as" => ident!(name, line)}}
  end

  defp import_decl(_, line), do: throw({:mzn, line, "import HASH as NAME"})

  defp claim([{:atom, name, _} | clauses], line) do
    c = %{"claim" => ident!(name, line), "root" => nil, "wazn" => nil, "inputs" => [], "body" => nil}

    c =
      Enum.reduce(clauses, c, fn
        {:list, [{:atom, head, l} | args], _}, c ->
          case S.keyword(head) do
            "root" -> one(args, l, "root", fn {:atom, r, _} -> S.root(r) || throw({:mzn, l, "unknown root #{r} (H-s-b, H-f-Z, n-q-l, k-t-b)"}) end) |> put(c, "root", l)
            "wazn" -> one(args, l, "wazn", fn {:atom, w, _} -> (k = S.keyword(w)) in @wazns && k || throw({:mzn, l, "unknown wazn #{w} (fail, maful, burhan)"}) end) |> put(c, "wazn", l)
            "inputs" -> put(Enum.map(args, &input(&1, l)), c, "inputs", l)
            "field" -> put(Enum.map(args, &binding(&1, l)), c, "field", l)
            "step" -> put(Enum.map(args, &binding(&1, l)), c, "step", l)
            "box" -> put(Enum.map(args, &interval(&1, l)), c, "box", l)
            "init" -> one(args, l, "init", &expr/1) |> put(c, "init", l)
            "invariant" -> one(args, l, "invariant", &expr/1) |> put(c, "invariant", l)
            k when k in ["proof", "burhan"] -> one(args, l, "proof", &proof(&1, l)) |> put(c, "proof", l)
            "body" -> one(args, l, "body", &expr/1) |> put(c, "body", l)
            _ -> throw({:mzn, l, "unknown clause #{inspect(head)}"})
          end

        {_, _, l}, _ -> throw({:mzn, l, "a clause is a list: (root …), (wazn …), (inputs …), (body …)…"})
      end)

    for k <- ["root", "wazn"], c[k] == nil, do: throw({:mzn, line, "claim #{name}: no #{k}"})
    if c["body"] == nil, do: throw({:mzn, line, "claim #{name}: no body"})
    {:ok, Map.put(c, "line", line)}
  end

  defp claim(_, line), do: throw({:mzn, line, "a claim has a name"})

  defp put(v, c, k, line) do
    if Map.has_key?(c, k) and c[k] not in [nil, []] and k != "inputs", do: throw({:mzn, line, "#{k} given twice"})
    Map.put(c, k, v)
  end

  defp one([x], _l, _what, f), do: f.(x)
  defp one(_, l, what, _), do: throw({:mzn, l, "(#{what} …) takes one argument"})

  defp input({:list, [{:atom, v, _}, {:atom, t, _}], _}, l) do
    type = S.keyword(t)
    if type in @types, do: [ident!(v, l), type], else: throw({:mzn, l, "type #{t}: one of q, int, f64, f32, bool"})
  end

  defp input(_, l), do: throw({:mzn, l, "an input is (name type)"})

  defp binding({:list, [{:atom, v, _}, e], _}, l), do: [ident!(v, l), expr(e)]
  defp binding(_, l), do: throw({:mzn, l, "a binding is (name expression)"})

  defp interval({:list, [{:atom, v, _}, {:num, lo, _}, {:num, hi, _}], _}, l) do
    if LP.qcmp(lo, hi) >= 0, do: throw({:mzn, l, "an interval needs lo < hi"})
    [ident!(v, l), lo, hi]
  end

  defp interval(_, l), do: throw({:mzn, l, "an interval is (name lo hi) with numbers"})

  defp proof({:atom, w, _}, l) do
    case S.keyword(w) do
      k when k in ["conserved", "nonneg", "pos", "invariant"] -> %{"kind" => k}
      _ -> throw({:mzn, l, "unknown proof #{w} (conserved, nonneg, pos, invariant, (identity E), (bounded LO HI))"})
    end
  end

  defp proof({:list, [{:atom, w, _} | args], _}, l) do
    case {S.keyword(w), args} do
      {"identity", [e]} -> %{"kind" => "identity", "rhs" => expr(e)}
      {"bounded", [{:num, lo, _}, {:num, hi, _}]} -> %{"kind" => "bounded", "lo" => lo, "hi" => hi}
      _ -> throw({:mzn, l, "a proof is conserved | nonneg | pos | invariant | (identity E) | (bounded LO HI)"})
    end
  end

  defp proof({_, _, l}, _), do: throw({:mzn, l, "a proof is a word or (identity E) / (bounded LO HI)"})

  defp expr({:num, {n, d}, _}), do: ["q", n, d]

  defp expr({:atom, w, l}) do
    case S.keyword(w) do
      "true" -> ["b", true]
      "false" -> ["b", false]
      _ -> ["v", ident!(w, l)]
    end
  end

  defp expr({:list, [{:atom, w, l} | args], _}) do
    op = S.operator(w)
    kw = S.keyword(w)

    cond do
      op in @arith or op in @cmp -> ["op", op, Enum.map(args, &expr/1)] |> arity!(l)
      kw in @logic -> ["op", kw, Enum.map(args, &expr/1)] |> arity!(l)
      # a claim of an imported module: alias.name
      String.contains?(w, ".") ->
        case String.split(w, ".") do
          [a, n] -> ["call", ident!(a, l) <> "." <> ident!(n, l), Enum.map(args, &expr/1)]
          _ -> throw({:mzn, l, "#{w}: a qualified name is alias.name"})
        end

      true -> ["call", ident!(w, l), Enum.map(args, &expr/1)]
    end
  end

  defp expr({:list, [], l}), do: throw({:mzn, l, "an empty ()"})
  defp expr({:list, [_ | _], l}), do: throw({:mzn, l, "a call starts with an operator or a claim's name"})

  defp arity!(["op", op, args] = e, l) do
    n = length(args)

    ok =
      case op do
        o when o in ["+", "*", "and", "or"] -> n >= 2
        "-" -> n in [1, 2]
        o when o in ["/", "^"] -> n == 2
        o when o in @cmp -> n == 2
        "not" -> n == 1
        "if" -> n == 3
      end

    if ok, do: e, else: throw({:mzn, l, "#{op} with #{n} argument(s)"})
  end

  defp ident!(w, l) do
    if S.keyword(w) != nil, do: throw({:mzn, l, "#{w} is a keyword, not a name"})

    case S.ident(w) do
      {:ok, n} -> n
      {:error, why} -> throw({:mzn, l, why})
    end
  end

  # ============================================================ the checker

  # the morphological rules: which roots go with which weights and clauses, and names in scope
  defp check_module(%{"decls" => decls} = m) do
    Enum.reduce_while(decls, {:ok, %{}}, fn
      %{"import" => _, "as" => a}, {:ok, seen} -> {:cont, {:ok, Map.put(seen, a, :module)}}
      c, {:ok, seen} ->
        case check_claim(c, seen) do
          :ok -> if Map.has_key?(seen, c["claim"]), do: {:halt, {:error, "line #{c["line"]}: #{c["claim"]} is defined twice"}}, else: {:cont, {:ok, Map.put(seen, c["claim"], c)}}
          {:error, why} -> {:halt, {:error, "line #{c["line"]}: claim #{c["claim"]}: #{why}"}}
        end
    end)
    |> case do
      {:ok, _} -> {:ok, %{m | "decls" => Enum.map(decls, &Map.delete(&1, "line"))}}
      e -> e
    end
  end

  defp check_claim(c, seen) do
    vars = Enum.map(c["inputs"], &hd/1)
    types = c["inputs"] |> Enum.map(&List.last/1) |> Enum.uniq()
    bool? = types == ["bool"]

    cond do
      length(Enum.uniq(vars)) != length(vars) -> {:error, "an input is named twice"}
      length(types) > 1 -> {:error, "inputs mix types #{Enum.join(types, ", ")} (one numeric type, or bool)"}
      c["root"] == "hfz" and c["wazn"] != "burhan" -> {:error, "the root ح-ف-ظ (conservation) takes the weight burhān: a conservation law without its proof is not one"}
      c["root"] == "hfz" and c["field"] == nil -> {:error, "a conservation claim needs (field (x dx/dt) …)"}
      c["root"] == "hfz" and Enum.sort(Enum.map(c["field"], &hd/1)) != Enum.sort(vars) -> {:error, "the field must give d/dt of every input, and only of them"}
      c["root"] == "nql" and (c["step"] == nil or c["invariant"] == nil) -> {:error, "a transition claim needs (step …) and (invariant …)"}
      c["root"] == "nql" and not bool? -> {:error, "transition claims are over bool inputs (numeric transition systems: state them with H-f-Z or H-s-b and a box)"}
      c["root"] == "nql" and Enum.sort(Enum.map(c["step"], &hd/1)) != Enum.sort(vars) -> {:error, "the step must give the next value of every input, and only of them"}
      c["root"] == "ktb" and c["wazn"] != "maful" -> {:error, "the root ك-ت-ب (record) takes the weight mafʿūl: a record is kept and named by its hash"}
      c["root"] == "ktb" and vars != [] -> {:error, "a record has no inputs: it is a value"}
      c["wazn"] == "burhan" and c["root"] == "hsb" and c["proof"] == nil -> {:error, "the weight burhān needs (proof …)"}
      c["wazn"] != "burhan" and c["proof"] != nil -> {:error, "a proof clause needs the weight burhān (fail and maful claims make no claim to prove)"}
      c["proof"]["kind"] in ["nonneg", "pos", "bounded"] and c["box"] == nil -> {:error, "a positivity or bound claim is made on a box: add (box (x lo hi) …)"}
      c["box"] != nil and Enum.sort(Enum.map(c["box"] || [], &hd/1)) != Enum.sort(vars) -> {:error, "the box must give an interval for every input, and only for them"}
      true -> check_scope(c, vars, seen)
    end
  end

  defp check_scope(c, vars, seen) do
    scope = MapSet.new(vars)
    exprs = [c["body"], c["init"], c["invariant"]] ++ Enum.map(c["field"] || [], &List.last/1) ++ Enum.map(c["step"] || [], &List.last/1) ++
              if(c["proof"]["kind"] == "identity", do: [c["proof"]["rhs"]], else: [])

    Enum.reduce_while(Enum.reject(exprs, &is_nil/1), :ok, fn e, :ok ->
      case scope_errors(e, scope, seen) do
        nil -> {:cont, :ok}
        why -> {:halt, {:error, why}}
      end
    end)
  end

  defp scope_errors(["v", n], scope, _seen), do: if(MapSet.member?(scope, n), do: nil, else: "#{n} is not an input")
  defp scope_errors(["op", _, args], scope, seen), do: Enum.find_value(args, &scope_errors(&1, scope, seen))

  defp scope_errors(["call", f, args], scope, seen) do
    case String.split(f, ".") do
      [a, _] -> if seen[a] == :module, do: Enum.find_value(args, &scope_errors(&1, scope, seen)), else: "#{a} is not an imported module"
      _ -> scope_call(f, args, scope, seen)
    end
  end

  defp scope_errors(_, _, _), do: nil

  defp scope_call(f, args, scope, seen) do
    case seen[f] do
      nil -> "#{f} is not a claim defined above"
      :module -> nil
      %{"inputs" => ins} when length(ins) != length(args) -> "#{f} takes #{length(ins)} argument(s), not #{length(args)}"
      %{"wazn" => "burhan", "root" => r} when r != "hsb" -> "#{f} is a proved claim, not a function to call"
      _ -> Enum.find_value(args, &scope_errors(&1, scope, seen))
    end
  end

  # ================================================================ linking

  @doc """
  Resolve a module's imports from a library directory of `HASH.wzn` files:
  each imported text is parsed (in either script), its hash recomputed and
  compared with the one named — one bit of difference refuses it — and its
  obligations are decided again: a module whose proofs do not hold is not
  imported. Returns the module with the imported claims added under
  `alias.name`, for `check/2` and `run/3`.
  """
  def link(%{"decls" => decls} = m, lib) do
    Enum.reduce_while(decls, {:ok, []}, fn
      %{"import" => h, "as" => a}, {:ok, acc} ->
        with {:ok, text} <- File.read(Path.join(lib, h <> ".wzn")) |> then(fn {:error, _} -> {:error, "import #{String.slice(h, 0, 12)}…: not in #{lib}"}; ok -> ok end),
             {:ok, im} <- parse(text),
             true <- hash(im) == h || {:error, "import #{String.slice(h, 0, 12)}…: the file's tree hashes to #{String.slice(hash(im), 0, 12)}… — refused"},
             {:ok, linked} <- link(im, lib),
             [] <- Enum.reject(check(linked), &(&1.verdict in ["proved", "none"])) do
          renamed = for %{"claim" => n} = c <- linked["decls"], do: c |> Map.put("claim", a <> "." <> n) |> Map.update!("body", &qualify(&1, a, linked))
          {:cont, {:ok, acc ++ renamed}}
        else
          {:error, why} -> {:halt, {:error, why}}
          [bad | _] -> {:halt, {:error, "import #{String.slice(h, 0, 12)}…: #{bad.claim} is #{bad.verdict} — a module whose proofs do not hold is not imported"}}
        end

      d, {:ok, acc} ->
        {:cont, {:ok, acc ++ [d]}}
    end)
    |> case do
      {:ok, ds} -> {:ok, %{m | "decls" => ds}}
      e -> e
    end
  end

  # calls inside an imported module refer to its own claims: qualify them
  defp qualify(["call", f, args], a, im) do
    f2 = if Enum.any?(im["decls"], &(&1["claim"] == f)), do: a <> "." <> f, else: f
    ["call", f2, Enum.map(args, &qualify(&1, a, im))]
  end

  defp qualify(["op", op, args], a, im), do: ["op", op, Enum.map(args, &qualify(&1, a, im))]
  defp qualify(e, _, _), do: e

  # ================================================================ identity

  @doc "The module's identity: SHA-256 of the canonical tree — the same whichever script it was written in."
  def hash(%{"decls" => _} = m), do: Vapor.Canonical.hex_digest(m)

  @doc "The module printed in a projection (`:latin` or `:arabic`)."
  def print(m, proj \\ :latin), do: S.print(m, proj)

  # ============================================================ obligations

  @doc """
  Discharge every obligation of a module. Returns a list of
  `%{claim, root, wazn, verdict, decider, detail}` with verdict `"proved"`,
  `"refuted"` (and a counterexample), `"unknown"`, or `"none"` (claims with
  nothing to prove).
  """
  def check(%{"decls" => decls} = m, opts \\ []) do
    env = claims(m)
    for %{"claim" => _} = c <- decls, do: obligation(c, env, opts)
  end

  defp claims(%{"decls" => decls}), do: for(%{"claim" => n} = c <- decls, into: %{}, do: {n, c})

  defp obligation(%{"wazn" => w} = c, _env, _opts) when w != "burhan",
    do: %{claim: c["claim"], root: c["root"], wazn: w, verdict: "none", decider: nil, detail: "a #{w} claim states no theorem"}

  defp obligation(%{"root" => "hfz"} = c, env, _opts) do
    vars = Enum.map(c["inputs"], &hd/1)
    h = poly!(c["body"], vars, env)
    lie = Enum.reduce(Enum.with_index(vars), Aludel.const(length(vars), 0), fn {v, i}, acc ->
      f = c["field"] |> Enum.find(&(hd(&1) == v)) |> List.last() |> poly!(vars, env)
      Aludel.add(acc, Aludel.mul(Aludel.diff(h, i), f))
    end)

    zero_verdict(c, lie, vars, "dH/dt = ∇H·f is the zero polynomial over ℚ", "dH/dt ≠ 0")
  rescue
    e in ArgumentError -> unknown(c, "polynomial identity", Exception.message(e))
  end

  defp obligation(%{"root" => "nql"} = c, env, _opts) do
    vars = Enum.map(c["inputs"], &hd/1)
    step = Map.new(c["step"], fn [v, e] -> {v, formula!(e, env)} end)
    inv = formula!(c["invariant"], env)
    next = subst_formula(inv, step)
    inductive = Formula.valid?({:imp, inv, next})
    initial = if c["init"], do: Formula.valid?({:imp, formula!(c["init"], env), inv}), else: :absent

    case {inductive, initial} do
      {{:proved, ci}, init} when init == :absent or elem(init, 0) == :proved ->
        lemmas = ci.proof_lemmas + if(init == :absent, do: 0, else: elem(init, 1).proof_lemmas)
        %{claim: c["claim"], root: "nql", wazn: "burhan", verdict: "proved", decider: "SAT + DRUP",
          detail: "invariant #{if init == :absent, do: "inductive (no initial condition given)", else: "holds initially and is inductive"}; #{lemmas} proof lemmas checked over #{length(vars)} variables"}

      {{:refuted, r}, _} ->
        %{claim: c["claim"], root: "nql", wazn: "burhan", verdict: "refuted", decider: "SAT", counterexample: r.counterexample,
          detail: "a state satisfying the invariant whose successor does not"}

      {_, {:refuted, r}} ->
        %{claim: c["claim"], root: "nql", wazn: "burhan", verdict: "refuted", decider: "SAT", counterexample: r.counterexample,
          detail: "an initial state outside the invariant"}

      other ->
        unknown(c, "SAT", inspect(other) |> String.slice(0, 120))
    end
  rescue
    e in ArgumentError -> unknown(c, "SAT", Exception.message(e))
  end

  defp obligation(%{"proof" => %{"kind" => "identity", "rhs" => r}} = c, env, _opts) do
    vars = Enum.map(c["inputs"], &hd/1)
    d = Aludel.sub(poly!(c["body"], vars, env), poly!(r, vars, env))
    zero_verdict(c, d, vars, "body − rhs is the zero polynomial over ℚ", "body ≠ rhs")
  rescue
    e in ArgumentError -> unknown(c, "polynomial identity", Exception.message(e))
  end

  defp obligation(%{"proof" => %{"kind" => k}} = c, env, opts) when k in ["nonneg", "pos", "bounded"] do
    vars = Enum.map(c["inputs"], &hd/1)
    p = poly!(c["body"], vars, env)
    {:ok, box} = Aludel.box(Enum.map(vars, fn v -> [_, lo, hi] = Enum.find(c["box"], &(hd(&1) == v)); {lo, hi} end))
    depth = Keyword.get(opts, :depth, 18)

    claims =
      case k do
        "nonneg" -> [{p, :nonneg, "body ≥ 0"}]
        "pos" -> [{p, :pos, "body > 0"}]
        "bounded" -> [{Aludel.sub(p, Aludel.const(length(vars), c["proof"]["lo"])), :nonneg, "body ≥ lo"},
                      {Aludel.sub(Aludel.const(length(vars), c["proof"]["hi"]), p), :nonneg, "body ≤ hi"}]
      end

    results =
      Enum.map(claims, fn {q, sense, what} ->
        case Aludel.decide(q, box, sense: sense, depth: depth) do
          {:certified, w} -> {:proved, what, w.cells, Aludel.check(q, box, w.witness, sense: sense) == :ok}
          {:refuted, r} -> {:refuted, what, r}
          {:exhausted, r} -> {:unknown, what, r}
          {:error, why} -> {:unknown, what, why}
        end
      end)

    cond do
      Enum.all?(results, &(elem(&1, 0) == :proved and elem(&1, 3))) ->
        cells = results |> Enum.map(&elem(&1, 2)) |> Enum.sum()
        %{claim: c["claim"], root: c["root"], wazn: "burhan", verdict: "proved", decider: "Bernstein (Aludel)",
          detail: Enum.map_join(results, "; ", &elem(&1, 1)) <> " on the box — #{cells} cells, witness replayed"}

      r = Enum.find(results, &(elem(&1, 0) == :refuted)) ->
        {_, what, ref} = r
        point = Map.new(Enum.zip(vars, ref.point), fn {v, q} -> {v, q} end)
        %{claim: c["claim"], root: c["root"], wazn: "burhan", verdict: "refuted", decider: "Bernstein (Aludel)", counterexample: point,
          detail: "#{what} fails at an exact point (value #{inspect(ref.value)})"}

      true ->
        unknown(c, "Bernstein (Aludel)", "the subdivision budget ran out (depth #{depth}); the claim may be true and touch its bound")
    end
  rescue
    e in ArgumentError -> unknown(c, "Bernstein (Aludel)", Exception.message(e))
  end

  defp zero_verdict(c, p, vars, proved_text, refuted_text) do
    if map_size(p.terms) == 0 do
      %{claim: c["claim"], root: c["root"], wazn: "burhan", verdict: "proved", decider: "polynomial normal form over ℚ", detail: proved_text}
    else
      pt = nonzero_point(p, vars)
      %{claim: c["claim"], root: c["root"], wazn: "burhan", verdict: "refuted", decider: "polynomial normal form over ℚ",
        counterexample: Map.new(Enum.zip(vars, pt)), detail: "#{refuted_text}: #{Aludel.to_text(p, vars)}"}
    end
  end

  # a non-zero polynomial of degree ≤ d per variable is non-zero somewhere on {0..d}ⁿ
  defp nonzero_point(p, vars) do
    d = p |> Aludel.degrees() |> Enum.max(fn -> 0 end)
    grid = Enum.reduce(vars, [[]], fn _, acc -> for pt <- acc, k <- 0..d, do: pt ++ [{k, 1}] end)
    Enum.find(grid, fn pt -> LP.qsign(Aludel.eval(p, pt)) != 0 end) || Enum.map(vars, fn _ -> {0, 1} end)
  end

  defp unknown(c, decider, why), do: %{claim: c["claim"], root: c["root"], wazn: c["wazn"], verdict: "unknown", decider: decider, detail: why}

  # --------------------------------------------- expressions → polynomials

  @doc false
  def poly!(e, vars, env), do: poly(e, vars, env, length(vars), %{})

  defp poly(["q", n, d], _vars, _env, k, _sub), do: Aludel.const(k, {n, d})
  defp poly(["v", x], vars, _env, k, sub), do: Map.get(sub, x) || Aludel.var(k, Enum.find_index(vars, &(&1 == x)) || raise(ArgumentError, "#{x} is not an input"))
  defp poly(["op", "+", args], vars, env, k, sub), do: args |> Enum.map(&poly(&1, vars, env, k, sub)) |> Enum.reduce(&Aludel.add(&2, &1))
  defp poly(["op", "*", args], vars, env, k, sub), do: args |> Enum.map(&poly(&1, vars, env, k, sub)) |> Enum.reduce(&Aludel.mul(&2, &1))
  defp poly(["op", "-", [a]], vars, env, k, sub), do: Aludel.scale(poly(a, vars, env, k, sub), -1)
  defp poly(["op", "-", [a, b]], vars, env, k, sub), do: Aludel.sub(poly(a, vars, env, k, sub), poly(b, vars, env, k, sub))

  defp poly(["op", "/", [a, ["q", n, d]]], vars, env, k, sub) when n != 0, do: Aludel.scale(poly(a, vars, env, k, sub), LP.q(d, n))
  defp poly(["op", "/", _], _, _, _, _), do: raise(ArgumentError, "division is by a non-zero number only, for a decision over polynomials")
  defp poly(["op", "^", [a, ["q", n, 1]]], vars, env, k, sub) when n >= 0 and n <= 24, do: Aludel.pow(poly(a, vars, env, k, sub), n)
  defp poly(["op", "^", _], _, _, _, _), do: raise(ArgumentError, "a power is a whole number from 0 to 24, for a decision over polynomials")

  defp poly(["call", f, args], vars, env, k, sub) do
    case env[f] do
      %{"inputs" => ins, "body" => body} ->
        inner = Map.new(Enum.zip(Enum.map(ins, &hd/1), Enum.map(args, &poly(&1, vars, env, k, sub))))
        # the callee's variables are replaced by the caller's polynomials
        poly_in(body, inner, env, k)

      _ -> raise ArgumentError, "#{f} cannot be expanded into a polynomial here"
    end
  end

  defp poly(["op", op, _], _, _, _, _), do: raise(ArgumentError, "#{op} is not polynomial arithmetic")
  defp poly(["b", _], _, _, _, _), do: raise(ArgumentError, "a truth value is not a polynomial")

  # a callee's body, with its own variables bound to polynomials of the caller's
  defp poly_in(["v", x], bound, _env, _k), do: Map.fetch!(bound, x)
  defp poly_in(e, bound, env, k), do: poly(e, Map.keys(bound), env, k, bound)

  # ------------------------------------------------ expressions → formulas

  defp formula!(["b", b], _env), do: b
  defp formula!(["v", x], _env), do: {:var, x}
  defp formula!(["op", "not", [a]], env), do: {:not, formula!(a, env)}
  defp formula!(["op", "and", args], env), do: args |> Enum.map(&formula!(&1, env)) |> Enum.reduce(&{:and, &2, &1})
  defp formula!(["op", "or", args], env), do: args |> Enum.map(&formula!(&1, env)) |> Enum.reduce(&{:or, &2, &1})
  defp formula!(["op", "=", [a, b]], env), do: {:iff, formula!(a, env), formula!(b, env)}
  defp formula!(["op", "if", [c, a, b]], env), do: (fc = formula!(c, env); {:or, {:and, fc, formula!(a, env)}, {:and, {:not, fc}, formula!(b, env)}})

  defp formula!(["call", f, args], env) do
    case env[f] do
      %{"inputs" => ins, "body" => body} -> subst_formula(formula!(body, env), Map.new(Enum.zip(Enum.map(ins, &hd/1), Enum.map(args, &formula!(&1, env)))))
      _ -> raise ArgumentError, "#{f} is not a claim defined above"
    end
  end

  defp formula!(other, _), do: raise(ArgumentError, "#{inspect(other)} is not Boolean")

  defp subst_formula({:var, x} = v, m), do: Map.get(m, x, v)
  defp subst_formula({:not, a}, m), do: {:not, subst_formula(a, m)}
  defp subst_formula({op, a, b}, m), do: {op, subst_formula(a, m), subst_formula(b, m)}
  defp subst_formula(b, _m), do: b

  # ================================================================ running

  @doc """
  Evaluate a claim's body on arguments. `q`/`int` inputs are exact
  rationals (`{num, den}`), `f64` floats, `f32` the canonical binary32 of
  vapor's oracle, `bool` booleans. A burhān claim runs only when its
  obligation is proved; a mafʿūl claim's value comes back with its hash.
  """
  def run(m, name, args) do
    env = claims(m)

    with %{} = c <- env[name] || {:error, "no claim #{name}"},
         true <- length(args) == length(c["inputs"]) || {:error, "#{name} takes #{length(c["inputs"])} argument(s)"},
         :ok <- runnable(c, env) do
      type = case c["inputs"] do [[_, t] | _] -> t; [] -> "q" end
      vals = Enum.zip(Enum.map(c["inputs"], &hd/1), Enum.map(args, &coerce(&1, type)))
      v = ev(c["body"], Map.new(vals), env, type)
      out = %{value: v, type: type}
      if c["wazn"] == "maful", do: {:ok, Map.put(out, :hash, Vapor.Canonical.hex_digest({:mizan_record, name, v}))}, else: {:ok, out}
    end
  end

  defp runnable(%{"wazn" => "burhan"} = c, env) do
    case obligation(c, env, []) do
      %{verdict: "proved"} -> :ok
      %{verdict: v, detail: d} -> {:error, "#{c["claim"]} is a burhān claim whose obligation is #{v} (#{d}): it does not run"}
    end
  end

  defp runnable(_, _), do: :ok

  defp coerce(x, t) when t in ["q", "int"] and is_integer(x), do: {x, 1}
  defp coerce({n, d}, t) when t in ["q", "int"], do: LP.q(n, d)
  defp coerce(x, t) when t in ["q", "int"] and is_binary(x), do: LP.rat(x)
  defp coerce(x, t) when t in ["q", "int"] and is_float(x), do: LP.rat(x)
  defp coerce(x, "f64") when is_number(x), do: x * 1.0
  defp coerce(x, "f64") when is_binary(x), do: String.to_float(if String.contains?(x, "."), do: x, else: x <> ".0")
  defp coerce(x, "f32"), do: x |> coerce("f64") |> Vapor.F32.from_float() |> Vapor.F32.to_float()
  defp coerce(x, "bool") when is_boolean(x), do: x
  defp coerce(x, "bool") when x in ["true", "1", "صواب"], do: true
  defp coerce(x, "bool") when x in ["false", "0", "خطأ"], do: false
  defp coerce(x, t), do: raise(ArgumentError, "#{inspect(x)} is not a #{t}")

  defp ev(["q", n, d], _b, _env, t) when t in ["f64", "f32"], do: rnd(n / d, t)
  defp ev(["q", n, d], _b, _env, _t), do: {n, d}
  defp ev(["b", v], _b, _env, _t), do: v
  defp ev(["v", x], b, _env, _t), do: Map.fetch!(b, x)

  defp ev(["op", op, args], b, env, t) when op in ["+", "*"] do
    vs = Enum.map(args, &ev(&1, b, env, t))
    Enum.reduce(tl(vs), hd(vs), fn y, x -> arith(op, x, y, t) end)
  end

  defp ev(["op", "-", [a]], b, env, t), do: arith("-", ev(["q", 0, 1], b, env, t), ev(a, b, env, t), t)
  defp ev(["op", op, [a, x]], b, env, t) when op in ["-", "/", "^"], do: arith(op, ev(a, b, env, t), ev(x, b, env, t), t)
  defp ev(["op", op, [a, x]], b, env, t) when op in @cmp, do: compare(op, ev(a, b, env, t), ev(x, b, env, t))
  defp ev(["op", "and", args], b, env, t), do: Enum.all?(args, &ev(&1, b, env, t))
  defp ev(["op", "or", args], b, env, t), do: Enum.any?(args, &ev(&1, b, env, t))
  defp ev(["op", "not", [a]], b, env, t), do: not ev(a, b, env, t)
  defp ev(["op", "if", [c, x, y]], b, env, t), do: if(ev(c, b, env, t), do: ev(x, b, env, t), else: ev(y, b, env, t))

  defp ev(["call", f, args], b, env, t) do
    %{"inputs" => ins, "body" => body} = env[f]
    ev(body, Map.new(Enum.zip(Enum.map(ins, &hd/1), Enum.map(args, &ev(&1, b, env, t)))), env, t)
  end

  defp arith("+", x, y, t) when t in ["q", "int"], do: LP.qadd(x, y)
  defp arith("-", x, y, t) when t in ["q", "int"], do: LP.qsub(x, y)
  defp arith("*", x, y, t) when t in ["q", "int"], do: LP.qmul(x, y)
  defp arith("/", _x, {0, _}, t) when t in ["q", "int"], do: raise(ArgumentError, "division by zero")
  defp arith("/", x, y, t) when t in ["q", "int"], do: LP.qdiv(x, y)
  defp arith("^", {n, d}, {k, 1}, t) when t in ["q", "int"] and k >= 0 and k <= 4096, do: LP.q(Integer.pow(n, k), Integer.pow(d, k))
  defp arith("^", _, _, t) when t in ["q", "int"], do: raise(ArgumentError, "exact powers take a whole exponent from 0 to 4096")
  defp arith(op, x, y, t), do: rnd(farith(op, x, y), t)

  defp farith("+", x, y), do: x + y
  defp farith("-", x, y), do: x - y
  defp farith("*", x, y), do: x * y
  defp farith("/", x, y), do: x / y
  defp farith("^", x, y), do: :math.pow(x, y)

  defp rnd(x, "f32"), do: x |> Vapor.F32.from_float() |> Vapor.F32.to_float()
  defp rnd(x, _), do: x

  defp compare(op, {_, _} = x, {_, _} = y), do: cmp(op, LP.qcmp(x, y))
  defp compare(op, x, y), do: cmp(op, cond do x < y -> -1; x > y -> 1; true -> 0 end)
  defp cmp("<", c), do: c < 0
  defp cmp("<=", c), do: c <= 0
  defp cmp(">", c), do: c > 0
  defp cmp(">=", c), do: c >= 0
  defp cmp("=", c), do: c == 0

  @doc "A value as text (rationals as n/d)."
  def show({n, 1}), do: Integer.to_string(n)
  def show({n, d}), do: "#{n}/#{d}"
  def show(x), do: to_string(x)
end
