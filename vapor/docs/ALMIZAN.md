# Almizan — a dialect for claims that get decided

> الميزان *al-mīzān*, "the balance"; root و-ز-ن *w-z-n*, "to weigh". Files **`.wzn`** (not `.mzn`: that is the
> MiniZinc extension). `Vapor.Almizan`, `Vapor.Almizan.Syntax`, `Vapor.Almizan.Lower`,
> `Vapor.Almizan.Abjad`. Tests: `almizan_test.exs`; §5l. The weighing of the manifesto that proposed it:
> [DIRECTIVE §19](DIRECTIVE.md).

## The name (0.17)

Until 0.16 the language was written Mīzān, Mizān and Al-Mizān in different places. It is now
**Almizan**, with the Arabic article fused as English fuses it in every word it took from Arabic
(alembic, alchemy, algebra, alkali, almanac). The two languages now read as a pair named the same
way: **Alembic** (*al-anbīq*, the still) distils and runs, **Almizan** (*al-mīzān*, the balance)
weighs and decides. Both file extensions are their triliteral roots: `.nbq` (ن-ب-ق *n-b-q*) and
`.wzn` (و-ز-ن *w-z-n*). The tree's format tag stays `"mizan" => 1`, since renaming it would change
the hash, and so the identity, of every module written in 0.16.

## In one sentence

A small language in which each claim says **what kind it is** (the root) and **how it may be
used** (the *wazn*), and in which a claim marked as proved **does not compile** until a decision
procedure of vapor proves it — or refutes it with the point that breaks it.

## The tree and the two scripts

The program is a tree. Latin and Arabic are **bijective printings** of it — the same tree, the same
hash:

```lisp
(claim energy (root H-f-Z) (wazn burhan)
  (inputs (x q) (v q))
  (field (x v) (v (- x)))
  (proof conserved)
  (body (+ (kinetic v) (* 1/2 x x))))
```

```lisp
(دعوى energy (جذر ح-ف-ظ) (وزن برهان)
  (مدخلات (x نسبي) (v نسبي))
  (حقل (x v) (v (- x)))
  (برهان محفوظ)
  (تنفيذ (+ (kinetic v) (* ١/٢ x x))))
```

- Keywords have a Latin and an Arabic form (`claim` دعوى, `root` جذر, `wazn` وزن, `inputs` مدخلات,
  `field` حقل, `box` صندوق, `step` خطوة, `init` بداية, `invariant` ثابت, `proof` برهان, `body`
  تنفيذ, `import` استيراد; types `q` نسبي, `int` صحيح, `f64` عائم٦٤, `f32` عائم٣٢, `bool` منطقي;
  `and` و, `or` أو, `not` ليس, `if` إذا; and the spelled-out operators جمع طرح ضرب قسمة أس).
- Numbers: Western digits in one script, Arabic-Indic (٠–٩) in the other; exact fractions.
- Names: Latin (letters, digits, `-`, `_`) **or** Arabic (letters, `-`, Arabic-Indic digits)
  — **never mixed**. Roots in Latin form use Buckwalter (`H-f-Z` = ح-ف-ظ): lossless,
  unlike a "pretty" romanization that merges the forms of hamza (7 names → 2, measured).
- `read(print(t)) = t` in both scripts: tested on 300 random programs.
- The hash (`vapor wzn hash`) is the SHA-256 of the canonical tree, the same in both scripts.

## Roots and *awzān*: a morphological type system

| root | domain | what can be claimed |
|---|---|---|
| ح-س-ب H-s-b | calculation | a value; identities; bounds and positivity on a box |
| ح-ف-ظ H-f-Z | conservation | a quantity constant along a field `ẋ = f(x)` |
| ن-ق-ل n-q-l | transition | a system of boolean steps and its invariant |
| ك-ت-ب k-t-b | record | a stored value, named by its hash |
| س-ب-ب s-b-b | cause (0.17) | an interventional query on a stated causal diagram |

| *wazn* | regime | consequence |
|---|---|---|
| فاعل *fāʿil* | transient | a pure function, lowered to vapor's compiler; states no theorem |
| مفعول *mafʿūl* | persistent | the value is stored and named by its hash |
| برهان *burhān* | proved | nothing runs until the obligation is met |

Meaningless pairs are refused at checking: a conservation (ح-ف-ظ) without `burhān` is not a law;
a record (ك-ت-ب) with `proof` has nothing to prove.

## Proofs by decision, not by a person in an assistant

| `proof` | decider | certificate |
|---|---|---|
| `(identity E)` | exact polynomial normal form over ℚ | both sides normalize to the same polynomial |
| `conserved` | dH/dt = ∇H·f, normalized over ℚ | the zero polynomial — or the remainder and a point where it is ≠ 0 |
| `nonneg`, `pos`, `(bounded a b)` | Aludel (Bernstein in exact integers) | a replayable subdivision — or the exact point that refutes |
| `invariant` | SAT (induction: `init ⇒ I`, `I ∧ step ⇒ I'`) | a DRUP proof checked by separate code — or the counterexample |

The verdict is *proved*, *refuted* (with the point) or *unknown*; **unknown does not compile**.
The language is cut down to fit the deciders. Lean 4 is a second opinion:
`transmute --to lean` generates the theorems (`ring`/`decide`); closing them is in the ledger as
**owed**, because Lean is not installed on this machine.

## The verbs (the manifesto's `ikseer`)

```sh
vapor wzn check FILE           # distils and decides: ✓ proved · ✗ refuted (with the point) · ? unknown
vapor wzn show FILE --arabic   # the other script (the same tree)
vapor wzn hash FILE            # the identity
vapor wzn run FILE CLAIM ARGS  # evaluates (exact in ℚ); a burhān only after it is proved
vapor wzn transmute FILE CLAIM --to vapor|aiger|lean [--out DIR]
                               # vapor's compiler (x86-64, AVX-512, AArch64, RISC-V, SPIR-V) · a circuit · Lean 4
vapor wzn assay FILE CLAIM     # the fāʿil in binary32 on vapor's oracle against ℚ, in ULPs
vapor wzn abjad كتب            # the abjad value, and why it is not an address
```

Exit: 0 everything proved · 1 something refuted or unknown · 3 bad input. Examples:
`priv/almizan/oscillator.wzn` (and its Arabic version), `bounds.wzn` (Motzkin + 1/1000, a cubic
bound, a square), `handshake.wzn` (an invariant by SAT).

## *Abjad*, measured

The manifesto proposed addressing by *abjad* (the numerical value of the letters). Of the 21,952
three-letter roots, **21,950 share their value with another** (99.99%); the largest class has 82 roots; every
anagram collides. The value is shown, never used as an address. The identity is the hash: zero
collisions on the same roots.

## Causes: the root س-ب-ب s-b-b (0.17)

*Sabab* is cause. A causal claim states its diagram and a query, and its proof names the
decision:

```
(claim front-door (root s-b-b) (wazn burhan)
  (graph (-> x m) (-> m y) (<-> x y))
  (proof identifiable)
  (body (do (y) (x))))

(claim adjust-z (root s-b-b) (wazn burhan)
  (graph (-> z x) (-> z y) (-> x m) (-> m y))
  (proof (adjustment z))
  (body (do (y) (x))))

(claim collider (root s-b-b) (wazn burhan)
  (graph (-> a c) (-> b c))
  (proof separated)
  (body (independent (a) (b) ())))
```

| proof | decider | proved | refuted |
|---|---|---|---|
| `identifiable` | the ID algorithm (Shpitser–Pearl, complete) | with the estimand, e.g. `Σ_{m} P(m \| x) (Σ_{x} P(x) P(y \| x, m))` | with a hedge, checked |
| `(adjustment z …)` | the back-door criterion | with `Σ_z P(y \| x, z) P(z)` | naming the descendant or the open path |
| `separated` | d-separation | in every model with this diagram | naming the connecting path |

The morphology holds here too. A causal claim takes `burhān`, since a causal claim without its
decision is an opinion. It states a `(graph …)` and has no inputs, because its variables are the
diagram's. Every name it mentions must be in the diagram, and `(do …)` and `(independent …)` belong
to s-b-b only. The Arabic projection is complete: `(دعوى front-door (جذر س-ب-ب) (وزن برهان)
(مخطط (-> x m) …) (برهان معرف) (تنفيذ (افعل (y) (x))))` is the same tree, with the same hash.
A causal claim is decided, never run, and no Lean statement is emitted for it, since core Lean has
no causal calculus. `priv/almizan/causes.wzn` holds five claims, two of them refuted. The
procedures are those of the logic desk ([LOGIC.md §7](LOGIC.md)).

## Editors

The language server (`vapor lsp`) diagnoses (each obligation decided on save), explains
(*hover*: the verdict and the decider of a claim; the meaning and the abjad value of a root; a
keyword in both scripts), completes, goes to the definition, formats and switches script.
[EDITORS.md](EDITORS.md).

## What it is not

It is not a general-purpose language, nor does it replace Alembic (the *kernels*) — Almizan states
claims and decides them; execution belongs to vapor's compiler. It does not prove what its deciders do not
decide: it refuses.
