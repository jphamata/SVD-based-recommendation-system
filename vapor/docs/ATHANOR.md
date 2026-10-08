# Athanor — the search furnace, and the Touchstone that checks it

> Since 0.14.0. Code: `lib/vapor/athanor.ex`, `lib/vapor/athanor/` (`space.ex`, `spec.ex`,
> `strategy.ex`, `gauss.ex`, `touchstone.ex`, `game.ex`, `session.ex`, `examples.ex`) and
> `lib/vapor/mind.ex`. Tests: `test/vapor/athanor_test.exs`, `mind_test.exs`,
> `workspace_test.exs`. Quality: `mix vapor.quality --only round14`.

The athanor is the alchemist's furnace that keeps the fire constant for a long time. Here it is a
**general searcher**: it receives a problem written in [Alembic](ALEMBIC.md) — a space and an
objective, or a claim — and returns a **certificate** that anyone can re-check without
trusting the search. There are no categories: Golomb, Ramsey, sorting networks, travelling salesman,
portfolios, hyperparameters, adversarial examples, *trading* rules and formulas are just
starting examples (`Examples.all/0`), written in the same language the person uses.

## 1. A problem

```
# the Golomb ruler with 7 marks (the optimum is 25)
space = subset(1..30, 6)
ruler(r) = [0] ++ r
violation(r) = let d = [b - a for (a, b) in pairs(ruler(r))] in len(d) - len(distinct(d))
minimize(r) = max(r)
target = 25
budget = 30000
```

Reserved names: `space`, `minimize`/`maximize`/`claim`, `valid`, `violation`, `margin`,
`target`, `budget`, `seed`, `start`, `show`, `describe`, `holdout`, `neighbor`, `measured`.
Spaces: `bits`, `ints`, `reals`, `perm`, `subset`, `subsets`, `seq`, `graph`, `partition` and
`program` (expression trees — the candidate reaches your functions as a function).

## 2. The furnace

- **Portfolio of strategies** — exhaustive (resumable, when the space fits), random,
  annealing, evolution with a MAP-Elites archive (`describe`), CMA-ES (real spaces, with
  Jacobi eigenvalues), Bayesian (Matérn-5/2 Gaussian process + expected improvement, batches
  via the *kriging believer*), **mind** (a language model proposes) and **human** (the person
  proposes). A discounted UCB bandit (γ = 0.97) shares out the budget by what each one yields.
- **Ranked infeasibles**: with `violation`, an invalid candidate still has a rank (−violation),
  and the search climbs up to validity (Schur, Golomb 8).
- **Control**: the same budget spent on uniform samples. If random never reaches the
  best, the certificate gives the upper bound of the per-sample chance (rule of three, 95%).
- **Holdout**: with `holdout(x)`, the finalists are re-evaluated on an objective the search did not see;
  the rank correlation (Spearman) and the noise's max-z (√(2 ln N)) say whether the winner is
  signal or selection bias. On a random walk, moving-average rules: ρ ≈ −0.33; with
  planted AR(1) momentum: ρ ≈ 0.73.
- **Journal**: each evaluation enters a SHA-256 chain over the canonical encoding; the root
  depends only on the text and the seed.

## 3. The certificate and the Touchstone

The certificate states the best (candidate, value, who found it), the reason for stopping (`exhausted`,
`target`, `counterexample`, `found`, `budget`, `time`), the verdict in words ("optimum proved
by enumeration", "claim proved over the whole space"), the control, the holdout and the
outside proposals. `Touchstone.verify/3` re-checks **without trusting the search**: it re-evaluates the
candidate, checks the value and membership of the space; with `full: true` it redoes the enumeration; with
`replay: true` it redoes the run and compares the journal root. A forged value is rejected with the
check that failed.

```
vapor athanor run golomb.nbq > cert.json      # exit code 0/1 according to the result
vapor verify golomb.nbq cert.json --replay     # the touchstone
```

## 4. Human and model in the loop

- **Sessions** (`Athanor.Session`, under `DynamicSupervisor`): the search runs in the background; the
  person watches the sparks, **proposes** candidates (checked and evaluated like the others),
  **pins** and **bans** finalists, extends the budget, stops and resumes. In the console, this is the
  live furnace; in the terminal, `vapor athanor run --interactive`.
- **Measured** (`measured = true`): the objective is outside the machine — a laboratory
  experiment, a training run, a person. `vapor athanor ask` proposes, the person measures and types it in;
  `--measure 'command'` measures through an external program. The Bayesian strategy works in batches.
- **Mind** (`Vapor.Mind`, `VAPOR_MIND=anthropic:MODEL | openai:MODEL[@URL] | script:FILE`):
  `formalize` translates words into Alembic, compiles, repairs up to three times with the
  compiler's error and returns a **back-translation** for the person to check what was understood;
  `propose` suggests candidates — which come in through the same door as anyone's. The model
  never decides: the Touchstone decides.

## 5. Games

With `init`, `player`, `moves`, `play` and `winner`, any two-player game:
`solve` (negamax with a transposition table — tic-tac-toe is a draw over 5,478 positions),
`search` (UCT/MCTS), `learn` (linear value tanh(w·features) by self-play) and `play`
(a game against the human), and `match` with a Wilson interval.

## 6. What it is not

It is not a theorem prover: "proved" here means **complete enumeration of a finite
space**, and the certificate states the size. Infinite spaces give evidence, not proof. It is not
parallel across nodes (yet). The default budget is modest; the furnace is honest about what
it did not find.
