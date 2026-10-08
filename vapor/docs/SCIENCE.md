# Science: from quantum mechanics to the genome, against references (0.11)

> Request, translated: "physical simulation (physics on its many fronts, from
> classical, relativistic, quantum, tokamak, chemistry (batteries, molecules,
> materials and beyond), biology (mutations, genomics, something similar to
> AlphaFold and beyond))". Scrutiny: [DIRECTIVE.md §14](DIRECTIVE.md).
> Tests: `science_test.exs`. Console: *Simulate → Science*. Classical
> physics (pendulums, chaos, digital twins) is that of 0.10:
> [PHYSICS.md](PHYSICS.md).

> **Since 0.14.0** these fixed experiments appear in the console as **Calibration**: they are the
> calibration of the instruments, with known answers. To put *your* system on the workbench —
> with evidence that needs no answer key — use the [Crucible](CRUCIBLE.md).

## 1. The rule

Each experiment has a **reference** — a closed form or a published
value — and, where a good one exists, a **control**: what a
wrong or naive implementation would produce, and which must fail. Not every
"control" in the table has the same strength, and this is stated: the Euler
integrator, the Cartesian Laplacian, the shuffled sites and the random
conformations are **wrong implementations, run** and rejected; the
classical particle for tunnelling is a **computed prediction** (the part
of the packet above the barrier), not a simulation; the norm drift of the
coherent state is a **conservation**, which a unitary scheme with the
wrong physics would also maintain; E×B and HeH⁺ have no control (—). Everything in binary64 on the BEAM, deterministic:
the same numbers on any machine, each one a function of its
parameters (`Vapor.Science.run/1`), savable and recomputable as an archive.

## 2. The experiments

| front | experiment | method | reference | control | measured |
|---|---|---|---|---|---|
| quantum | coherent state of the oscillator | 1-D Schrödinger by Fourier *split-step* (Feit, Fleck & Steiger 1982), own radix-2 FFT | ⟨x⟩(t) = x₀ cos t | the norm drift (unitarity) | max. error **4.0·10⁻⁵**; norm to 3·10⁻¹⁴ |
| quantum | tunnelling | packet below the barrier | ∫T(k)\|φ(k)\|²dk, exact plane-wave T | the classical particle: only the part of the packet above the barrier gets through | **0.5431** against **0.5445**; classical 0.0013 |
| relativity | gyration at 0.9 c | Boris pusher (the integrator of PIC codes) on u = γv | period 2πγ/B (γ = 2.294) | explicit Euler gains energy on every turn | 14.41464 against 14.41462; \|u\| to 2·10⁻¹⁵; Euler +6.7% |
| plasma | E×B drift | Boris with E ⊥ B | v = E/B | — | 0.2992 against 0.3 |
| fusion | equilibrium of a tokamak | Grad–Shafranov (Δ*ψ = −μ₀R²p′ − FF′) by conservative finite differences and SOR | the exact **Solov'ev** solution (p′, FF′ constant) | the Cartesian Laplacian (without the 1/R term of the toroidal geometry) | flux to 2·10⁻¹² (the scheme has no truncation error on these polynomials); the **magnetic axis** (vertex of a parabola through the grid maximum) converges as h² (1.25·10⁻³ → 3.1·10⁻⁴) — the rate is the locator's, not the scheme's; the control is off by 8·10⁻³ and does not converge |
| chemistry | H₂ | restricted Hartree–Fock, STO-3G, closed-form integrals (Boys function via `erf`) — Szabo & Ostlund §3.5 | **−1.1167** hartree (R = 1.4 bohr) | — | **−1.11671**; orbital energies −0.578 and 0.670 (the book's) |
| chemistry | HeH⁺ | the same | **−2.860662** hartree | — | **−2.860659** |
| chemistry | the known failure | separated H₂ (R = 10) | two H atoms: −0.9332 | — | RHF ends up **0.34 hartree above** — a single determinant does not describe two separated electrons: that is why configuration interaction exists |
| materials | Lennard-Jones liquid | 64 atoms, periodic box, velocity Verlet | energy conservation; the first peak of g(r) near 2^{1/6}σ | explicit Euler | fluctuation **5·10⁻⁴** over 300 steps; g(r) peak at 1.11σ; Euler blows up (10³³) |
| evolution | fixation of a mutant | haploid Wright–Fisher, 4,000 replicates with counter-based draws | the **exact** probability of the Markov chain (solved) | the neutral mutant fixes at 1/N | 0.0955 against 0.0941 (Kimura: 0.0958); neutral 0.019 against 0.02 |
| genomics | tree rebuilt from genomes | Jukes–Cantor along a known tree; corrected distances; *neighbour joining* | Robinson–Foulds 0 | the sites of each sequence shuffled | RF **0**; distances to 3.3%; control RF 6 (the maximum) |
| proteins | HP folding (Dill 1985) | Monte Carlo with pivots, corners and ends, annealing | the published optimum of the classic 20-mer: **−9** (Unger & Moult 1993); on a 12-mer, exact enumeration | random conformations | **−9**; 12-mer equal to the enumeration (−5); random −1.3 |

## 3. What was found along the way

- **Precedence**: in Elixir, `-(x - x0) ** 2` is `(x0 - x)²`, not
  `−(x − x0)²` — the Gaussian packet was born as a growing exponential.
  The reference (the coherent state) caught it on the first run.
- **The barrier misaligned with the grid** (5.1 cells for a width of 1)
  gave 0.447 against 0.544: the barrier now has exactly a/dx cells.
- **The Jukes–Cantor mutation** with `|> min(2)` in the wrong place drew
  bases that were sometimes the same: the distances came out 40% short. The
  check of the distances against the true lengths caught it.

## 4. What it is not, and why

- **AlphaFold**: predicting real protein structures requires the trained
  model and its databases (PDB, multiple alignments) — none of that
  exists here. The HP model is the physical toy that makes the search
  problem exact; stated as such.
- **Real batteries and materials**: DFT of solids (plane waves,
  pseudopotentials) and electrolyte chemistry are far beyond what an
  honest round measures; Hartree–Fock of two-electron molecules and the
  Lennard-Jones liquid are the first steps, checked.
- **Tokamak**: the equilibrium, not stability or transport;
  Solov'ev is the standard test case of equilibrium codes.
- **General relativity**: out.
- Nothing here is a vapor program yet (binary64 on the BEAM, deterministic,
  but not the compiler's "same bits on every substrate"): the *split-step*
  is linear (DFT as `linear`) and is the natural candidate to become a program.
