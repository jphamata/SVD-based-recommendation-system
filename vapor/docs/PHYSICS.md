# Physics for reinforcement learning and digital twins (0.10)

> Request, translated: "think about a physics engine in vapor" — in the sense of
> *reinforcement learning* and *digital twins*. Scrutiny:
> [DIRECTIVE.md §13](DIRECTIVE.md). Tests: `physics_test.exs`. Console:
> *Simulate → Physics*.

## 1. Why a physics engine inside a tensor compiler

Simulators disagree with themselves: the same model, seed and actions give
another trajectory on another GPU, with another number of *threads* or
another version of the library, because floating-point sums are reordered.
For a smooth system it is a nuisance; for a **chaotic** one — a double
pendulum, a walking robot, a flow — it is total: two runs separate within
seconds. Hence the two pains:

- **RL does not reproduce.** A training curve, a "good" episode, a rare
  failure: none of them can be redone on another machine.
- **A digital twin cannot be audited.** "The twin predicted X at time t"
  only holds if someone else, with the model and the inputs, recomputes X.

In vapor, a world's step is a **program** like any other: it runs under the
canonical semantics and has **the same bits on every substrate** — the
exact oracle, the native workers, the admitted GPUs — for any batch of
worlds.

## 2. The method (`Vapor.Physics`)

Extended position-based dynamics (XPBD; Macklin, Müller & Chentanez 2016)
with many substeps and one constraint pass each (Müller et al.
2020): particles with inverse mass, **rods** (distance constraints with
compliance), **rails** (one fixed coordinate — the cart on the rail),
**ground** and **actuators** (forces coming from the action). The constraint
graph is a signed incidence matrix `D` (rod c: +1 at i, −1 at j), and one
Jacobi pass over all rods of all worlds is two matrix products:

    d = p·Dᵀ                           rod vectors, all worlds
    λ = −(|d| − ℓ) / (wᵢ + wⱼ + α/h²)
    Δp = ((λ/|d|)·d)·D ∘ w ∘ 1/degree  each particle's share

— `linear` operations the airlock already certifies. Worlds are rows
(`f32[B, N]` per axis): a thousand cart-pendulums are one run.

**Finding (0.10):** computing the velocity as `(p − p_before)/h` loses three
digits in single precision when `h` is small (cancellation): the pendulum's
period **got worse** above 8 substeps. The velocity is now
`v + (corrections)/h` — no cancellation — and the error halves each time the
substep is halved, as it should in a first-order method.

## 3. What is measured

| measure | value | control |
|---|---|---|
| pendulum period (θ₀ = 0.5) against the exact `4√(L/g)·K(sin θ₀/2)` | error 6.5·10⁻⁴ → 3.4·10⁻⁴ → 1.7·10⁻⁴ → 8.5·10⁻⁵ (4 → 32 substeps) | first order: each halving of `h`, half the error |
| double pendulum, oracle × native | **bit for bit** | the same world **one ulp** away: separated (> 10⁻²) within 20 s |
| trajectory gradient (`Vapor.Autodiff`) | = finite differences within 3% | — |
| twin identification: rod and damping from measurements with 2 mm noise | **1.0002 m (truth 1) and 0.300 (0.3)** | the same measurements shuffled in time: 0.72 m and 1.55 — nothing recovered |
| cart-pendulum by random search (ARS, linear policy, ±1 directions) | **200/200** on 8 never-seen starts | null policy: ~50 |
| twin watching the plant (CUSUM of the residuals) | alarm **18 steps** after the rod stretches 0.5% (5 mm) | no fault: no alarm in 150 steps |
| twin's ledger (chain of *hashes*) | redone from the model and the actions: every prediction recomputed | a forged prediction: caught at the exact entry (measurements and residuals: only with an external anchor, §4) |

vapor's cart-pendulum is a particle system (a cart on a rail, a mass at the
tip of a rod), not gymnasium's equations; gymnasium's analytic CartPole
remains in `Vapor.RL` (equal to it within 10⁻¹²).

## 4. The digital twin (`Physics.twin/2`, `observe/3`, `replay/1`)

The twin runs the model alongside the plant, step by step, with the plant's
actions; it compares predicted and measured positions (residual in units of
the sensor noise) and accumulates the excess in a **CUSUM** (Page 1954): a
slow drift — a rod that lengthens, a bearing that wears — adds up until it
crosses the threshold, where a per-step threshold would let it through.
Each step enters a **hash chain** `(t, actions, prediction digest,
measurement digest, residual)`. Since the simulation is exact, whoever has
the model and the actions redoes the predictions and checks the chain
(`replay/1`): the record of "what we expected at t" cannot be rewritten
afterwards. **What the chain alone does not guarantee:** the measurements
and the residuals (hence the history of alarms) enter only as *digests*;
whoever rewrites a measurement and recomputes the whole chain produces
another head, equally valid. For that the head needs an external anchor —
a *checkpoint* of the transparency log (`Vapor.Tlog`) or a signature at the
time — not yet wired in here.

And if the twin's model is wrong? `sysid/3` fits its parameters (lengths,
damping) to the plant's measurements by **gradient descent through the
simulator** — the gradient of the whole trajectory, computed as a program
(`sysid_program/2`), also bit for bit on any substrate.

## 5. Limits (and what is left for the [TODO](TODO.md))

- Particles and rods, 2-D and 3-D; **no rigid bodies with rotation**
  (inertia, angular joints), no collision between bodies (only the ground),
  no friction. They are the natural next step (XPBD handles joints and
  contacts the same way), and each needs its own measurement against a
  reference.
- Jacobi with 1/degree relaxation: stable, but converges more slowly than
  Gauss-Seidel on long chains; substeps compensate.
- ARS runs the policy in the BEAM between steps (one round trip per step,
  a small one); embedding the policy in the program would make the whole
  episode a single run.
- Plant and twin here are the same simulator (with the fault injected): the
  real case — the twin of a physical machine — has model error, and the
  CUSUM threshold has to be calibrated on it.
