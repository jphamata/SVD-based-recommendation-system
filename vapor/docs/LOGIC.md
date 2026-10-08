# Logic — a procedure proposes, a verifier decides

> Request (0.12), translated: "fine control […] for CS, mathematics, physics (Fields,
> Turing, Nobel and frontier level), using AI masterfully, with or without a
> human in the loop, for arbitrary problems". Scrutiny:
> [DIRECTIVE.md §15](DIRECTIVE.md).

What makes a system that produces mathematics trustworthy — be it a
classical prover or a language model trained by reinforcement over
a proof assistant — is always the same separation: **whoever proposes does not
decide**. The logic desk implements this separation for four decidable
logics and opens it to any proposer: a person, a search, or a
language model via MCP. Acceptance depends only on the verifier.

`Vapor.Logic.run/1` (the desk decides) · `Vapor.Logic.check/2` (the desk
checks someone else's proposal) · console *Solve → Logic* · MCP
`logic_check`.

## 1. The four logics

| input | logic | procedure | certificate and verifier |
|---|---|---|---|
| `p cnf …` (DIMACS) | propositional | **CDCL** (2 watched literals, 1-UIP, minimization, Luby restarts, activity) | model (evaluated clause by clause) or **DRUP** refutation, checked by `Vapor.Logic.DRUP` (backwards, core first) — a separate program that shares no code with the solver |
| `valid: φ`, `sat: φ`, `equiv: a ; b` | formulas | Tseitin → CDCL | the same; a counterexample evaluated on the original formula |
| `schur k`, `vdw k r`, `ramsey s t`, `pigeonhole p h`, `queens n` | finite combinatorics | encoding + CDCL, threshold search | the number: a witness **below** (checked against the definition) and a DRUP refutation **at** the threshold |
| equations + `decide s = t` | equational theories | **Knuth–Bendix** with lexicographic path ordering | convergent rewriting system; normal forms with the derivations |
| `vars` / `hyp` / `claim` | polynomial geometry | **Buchberger** (criteria, reduced basis), **Rabinowitsch** trick | basis {1} for the implication; the remainder, when it does not follow; non-degeneracy conditions |

Results checked (`logic_test.exs`, §5h): the solver agrees with
exhaustive enumeration on 150 random 3-CNFs at the threshold; **S(3) = 13**,
**W(3; 2) = 9**, **R(3, 3) = 6**, each with witness and refutation
checked; a tampered proof is rejected (the control); Knuth–Bendix
completes the three group axioms into the **ten classic rules** and decides
i(x·y) = i(y)·i(x) — the merely oriented axioms (control) do not decide; the
lex basis of the Cox–Little–O'Shea example; **Thales' theorem** proved and
a false variant "not implied".

## 2. Human or AI in the loop: `logic_check` with a proposal

An external proposer sends the claim **and** a candidate:

| claim | proposal | accepted when |
|---|---|---|
| DIMACS | `{"model": [1, -2, 3]}` | every clause has a true literal |
| DIMACS | `{"drup": [[…], …, []]}` | the DRUP verifier derives the empty clause by unit propagation |
| `schur k` | `{"witness": [colours of 1..n]}` | colours in 1..k and no monochromatic x + y = z → S(k) ≥ n |
| `vdw k r` | `{"witness": [colours of 1..n]}` | no monochromatic progression of k terms → W(k; r) > n |
| `ramsey s t` | `{"n": n, "red": [[a, b], …]}` | neither a red K_s nor a blue K_t → R(s, t) > n |
| `sat: φ` / `valid: φ` | `{"assignment": {…}}` | φ true (model) / φ false (counterexample to validity) |

This is how a language model takes part without needing to be trusted:
it can propose a larger Schur colouring, a counterexample, a
refutation that another solver produced — and the desk answers ACCEPTED or
REJECTED with the reason. The tests (`mcp_server_test.exs`, `logic_test.exs`)
show the right colouring accepted and the same one with **one** colour swapped
rejected; the whole refutation accepted and the truncated one rejected.

## 3. Comparison with what exists

- RL provers over Lean (the state-of-the-art olympiad level):
  they propose steps in a general proof assistant. Here there is no Lean and no
  trained model; there are four **decidable** logics in which both proposing
  and checking are complete. The proposer/verifier separation is the
  same; the reach is smaller and the verdict is always definitive.
- Industrial SAT solvers (Kissat, CaDiCaL): orders of magnitude
  faster. The refutation format (DRUP) is the same one that SAT
  competitions require; the verifier here checks proofs from those solvers too.

## 4. Honest limits

- DRUP checking is in Elixir: R(3, 4) = 9 is refuted in seconds but
  its refutation takes ~2 minutes to check — outside the fast suite.
- No general first-order logic, no arithmetic; Gröbner over ℚ, no
  inequalities (ordered real geometry is not decided here). Since 0.17
  the Lean development builds here (Lean 4.34.1), but the desk's
  certificates are not yet exported to it.
- Knuth–Bendix may not terminate for a theory without a finite
  convergent system: there is a limit of 4000 steps and the answer says that completion did not terminate (instead of spinning forever).

## 5. Exact linear arithmetic: rational simplex with certificates (0.13)

```
maximize 3x + 2y
subject to
x + y <= 4
x + 3y <= 6
x <= 3
free z
```

`Vapor.Logic.LP` solves linear programs in **exact rationals** (two
phases, Bland's rule — it does not cycle), so that "optimal", "infeasible" and
"unbounded" are decided, not estimated. Each verdict comes with the object
that proves it, found by solving the **alternative system** and checked by
`LP.check/2`, which only multiplies and compares:

| verdict | certificate | the check |
|---|---|---|
| optimal | the primal x and a dual y | x feasible; y dual-feasible (Aᵀy ≥ c with the rows' signs); cᵀx = bᵀy |
| infeasible | the Farkas vector y | Aᵀy ≥ 0, y ≥ 0 on the ≤ rows, ≤ 0 on the ≥ rows, bᵀy < 0 |
| unbounded | a feasible x and a ray d ≥ 0 | A·d with the sign of each row and cᵀd > 0 |

The logic desk decides (`maximize …`/`minimize …` on the first line) and
**checks proposals** via `logic_check`: `{"x": {…}, "y": […]}` for
optimality, `{"farkas": […]}` for infeasibility, `{"x": …, "ray": …}`
for unboundedness — numbers as `"p/q"`. Six random LPs agree with
SciPy's HiGHS to 10⁻⁹; a wrong proposal (an x that violates the second
constraint) is rejected with the row.

The same simplex decides **arbitrage** (the fundamental theorem of asset
pricing is Farkas' lemma): [FINANCE.md §8](FINANCE.md). This
closes the item "linear arithmetic (Simplex with Farkas certificate)" of the
0.12 TODO.

## 6. Integer programs: branch and bound with a certificate anyone can check (0.17)

```
maximize 8a + 11b + 6c + 4d
5a + 7b + 4c + 3d <= 14
bin a, b, c, d          # 0/1 variables; `int x, y` for general integers
```

`Vapor.Logic.MIP` adds integer and binary variables to the same text
form and solves by **branch and bound** over the exact simplex of §5: a
fractional integer variable `v = 7/2` splits its node into `v ≤ 3` and
`v ≥ 4`, which keep every integer point of the parent between them. The
answer is **optimal**, **infeasible** or, when the node budget
(`max_nodes`, 20,000) runs out, **exhausted** with the incumbent and the
best open bound: a gap, never a guess.

The certificate is the search tree itself, the incumbent, and at each leaf
the object that closes it. `MIP.check/2` shares no code with the search:

| leaf | object | the check |
|---|---|---|
| infeasible | a Farkas `y` for the node's rows | `LP.check/2` on the node (the original rows plus the branching bounds) |
| bounded | a dual-feasible `y` for the node's rows | weak duality: `Aᵀy ≥ c` (free variables `=`), `bᵀy + c₀ ≤ z*` in the max form |

It re-derives every node from the branching decisions, so a split can
only be `v ≤ k` / `v ≥ k + 1` on a declared integer variable, and
then the leaves cover every integer point. A wrong optimum cannot pass,
because the leaf that holds the better point is bounded by `z*`. This is
the shape of certificate that VIPR standardised for exact MIP solvers
(Cheung, Gleixner & Steffy, 2017). It uses branching only; cutting planes
(Gomory, with their Chvátal–Gomory derivations) are not implemented.

Measured (`test/vapor/mip_test.exs`, §5m of the quality suite): on 40
random integer programs (max and min, `≤` and `≥` rows), the optimum
equals brute force on every one and every certificate checks; the
relaxation rounded down is right on 12 of 40. A forged objective is
refused, and so is a worse incumbent presented with the honest tree,
while the naive check (feasible and integral) accepts that worse point.
The desk takes proposals as JSON (`vapor logic check FILE PROPOSAL.json`,
or `logic_check` over MCP): `incumbent`, `objective`, and the tree as
nested `{"split", "at", "le", "ge"}` / `{"leaf", "y"}` objects.

A relaxation that is unbounded at the root is refused. With rational data
the integer program is then infeasible or unbounded (Meyer, 1974), and
telling which needs an integer ray that this solver does not search for.

## 7. Causal claims on a stated diagram (0.17)

```
causal
x -> m
m -> y
x <-> y              # a hidden common cause of x and y
identify y | do(x)
backdoor x -> y | z
dsep a; b | c
```

`Vapor.Logic.Causal` decides three kinds of question about a
semi-Markovian diagram: directed edges, and bidirected edges that each
stand for a hidden common cause.

- **Identifiability** of `P(y | do(x))` by the ID algorithm (Shpitser &
  Pearl, 2006), which is **complete**: when it fails, no estimate from the
  observed joint exists for any model with this diagram. Success returns
  the **estimand**, an expression in `P(v)` alone (sums, products,
  conditionals, ratios). For the front door it is
  `Σ_m P(m | x) Σ_x′ P(x′) P(y | x′, m)`. Failure returns a **hedge**,
  checked structurally by `Causal.hedge?/5`: two bidirected-connected node
  sets `F′ ⊂ F`, `F` meets X and `F′` does not, and each contains a
  forest with one root set inside `An(Y)`.
- **Back-door adjustment**: a set `Z` admits `Σ_z P(y | x, z) P(z)` when no
  element of Z descends from X and Z d-separates X from Y once the edges
  out of X are cut. A refusal names the descendant or the open path.
- **d-separation**, decided on the moralised ancestral graph, with each
  bidirected edge made an explicit hidden parent; a refusal names the
  connecting path.

Soundness is **tested in exact arithmetic** (`test/vapor/causal_test.exs`).
Random structural causal models with binary variables and explicit hidden
parents give the observed joint and, by cutting the model, the true
intervention. Both are rationals, so the estimand must equal the truth
exactly, and on 30 random diagrams it does. The naive `P(y | x)` is the
control, and it is wrong under confounding. For the bow (`x → y`, `x ↔ y`)
the test builds two models that agree on `P(x, y)` and disagree on the
effect (1/2 against 1), which is what "not identifiable" means.

What these decisions do not do: they do not find the diagram. Every
verdict is conditional on the diagram a person states, and the module
infers nothing about causes from data. The same decisions are an
Almizan root (`س-ب-ب s-b-b`, [ALMIZAN.md](ALMIZAN.md)), so a causal claim
can sit in a file next to the conservation laws and be decided with them.
