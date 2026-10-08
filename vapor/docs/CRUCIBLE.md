# Crucible — open science, with evidence that needs no answer key

> Since 0.14.0. Code: `lib/vapor/crucible.ex` and `lib/vapor/crucible/` (`sheet`, `poly`,
> `laws`, `quantum`, `hamiltonian`, `reactions`, `evolution`, `phylo`, `molecule`, `fold`,
> `fields`, `regress`). Tests: `test/vapor/crucible_test.exs`.

The "Science" panel of the earlier rounds showed fixed experiments with known answers —
solved problems, presented. They remain (renamed **Calibration** in the console: they are the
calibration of the instruments). The Crucible is the opposite: **the person brings the system** — their
equation, their molecule, their sequences, their data — and the answer comes with evidence that
holds **without a reference solution**:

- **observed order** (Richardson): the error falls at the rate the method promises, or it does not;
- **theorems that hold for any input**: virial, Ehrenfest, work–energy,
  unitarity, the SCF commutator;
- **two methods that must agree** (exact chain × simulation × Kimura);
- **controls that a wrong method would fail**: RK4 against the symplectic one, Euler against Boris,
  shuffled columns, shuffled target, generic terms added.

## The domains

| domain | input | what it returns | the evidence |
|---|---|---|---|
| `laws` | a system of ODEs | **conservation laws proved over ℚ** (including with `ln`) | exact null space of the Lie-derivative coefficient; they survive pushing each coefficient by 1 %; the generic control has none |
| `quantum` | V(x), box, grid | eigenvalues (Sturm bisection), dynamics (split-step Fourier) | observed order 2, virial, Ehrenfest, norm |
| `hamiltonian` | H(q, p) | symplectic integration (Yoshida 4, Verlet, implicit midpoint) | order, reversibility, drift against RK4, laws |
| `reactions` | chemical reactions | ODE and Gillespie | mass invariants |
| `evolution` | N, s, i₀ | Wright–Fisher fixation | exact chain = simulation = Kimura |
| `phylogeny` | FASTA | NJ tree with *bootstrap* | the shuffled-columns control without support |
| `molecule` | H/He atoms | N-centre RHF/STO-3G | converged commutator, virial; H₂ = −1.1167 |
| `fold` | HP sequence | lattice folding | parity bound; optimum proved by enumeration when it fits |
| `fields` | E, B, charge | Boris | \|u\| conserved, work–energy; Euler as control |
| `regress` | table | symbolic regression (Keijzer linear scaling + Nelder–Mead) | test R²; shuffled target as control |

Example — an SIR epidemic:

```
S' = -0.3*S*I
I' = 0.3*S*I - 0.1*I
R' = 0.1*I
S(0) = 0.99; I(0) = 0.01; R(0) = 0
t = 0 .. 100
degree = 2
```

→ `S + I + R` and `3·S + 3·I − ln(S)`, both **proved** by exact cancellation, both
structural (they survive perturbed coefficients), relative drift < 10⁻⁴ along the
trajectory. A damped oscillator: no law — and that is proved too.

## Limits

Everything runs in `Alembic.sandbox` (1 GB, 4 min): an absurd grid comes back as a time or
memory error, it never brings the server down. The chemistry is closed-shell STO-3G with H and He;
p orbitals are refused with the reason. Laws with non-polynomial functions (`basis = [cos(q)]`) are
verified numerically and **labelled** "not a proof".
