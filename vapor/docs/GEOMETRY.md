# Information geometry — where there is a manifold to measure

> `Vapor.InfoGeom`, `Vapor.Assay.Geometry` (`vapor assay geometry`). Tests: `info_geom_test.exs`;
> §5l. The critiques and the scope: [DIRECTIVE §19](DIRECTIVE.md), critique 1.

## Where it applies, where it does not

Information geometry measures **distributions**: model outputs, estimators, sets. There a statistical
manifold exists, and the Fisher metric is the only one invariant under reparametrisation
(Čencov). Where there is **no** distribution — a compiler's cost over register allocations,
the surface of a *kernel* — the space is a discrete lattice, and geodesics mean nothing:
vapor does not use them there.

And one does not run a manifold on silicon: one runs the **closed form** that the geometry delivers.

## What there is

On the simplex, `p ↦ 2√p` takes the Fisher metric to the round metric of a sphere. Everything follows from there:

| function | what | cost |
|---|---|---|
| `fisher_rao(p, q)` | `2·arccos Σ√(pᵢqᵢ)`: the geodesic distance — **a metric** (symmetric, triangle inequality), bounded by π | one sum and one arccosine |
| `geodesic(p, q, t)` | the great-circle interpolation (*slerp* on the roots) | O(k) |
| `frechet_mean(ps)` | the Karcher mean: the geometric *ensemble*, which stays a distribution and commutes with renaming the classes | fixed-point iterations |
| `gaussian_fisher_rao(μ₁, σ₁, μ₂, σ₂)` | between normals the metric is hyperbolic (Poincaré half-plane in (μ/√2, σ)): closed form | O(1) |
| `natural_logistic(X, y)` | logistic regression by natural-gradient steps (the Fisher matrix `Xᵀ diag(p(1−p)) X`) | Newton |
| `kl`, `js`, `hellinger` | the controls and the neighbours | O(k) |

## Measured, with controls (§5l)

- **Metric**: on 1,000 random triples of distributions, Fisher–Rao violates the triangle
  inequality **0** times; KL, **217**. With KL, "A is closer to B than to C" means
  nothing.
- **Invariance**: one variable rescaled ×1000 — the natural gradient gives the same predictions
  (largest difference 7·10⁻¹²); the plain gradient (the control) does not (0.54).
- **Normals**: with equal means the distance is `√2·|ln(σ₂/σ₁)|`; the same difference in means weighs
  more when σ is small — which the Euclidean distance on the parameters does not see.

## In Assay

`vapor assay geometry` takes one row per (model, item) with the predicted probabilities (CSV
`model,item,<class>,…` or JSON lines) and returns the mean distance of each pair with a *bootstrap*
interval over the items, the distance of each model to the geometric consensus (the Fréchet mean,
item by item) — the *outlier* is the model the others do not believe — and two checks: the
triangle inequality on every triple and a **control**: the same models with the items
shuffled. Two models are only "close" if they are closer than models that answer
unrelated questions. [ASSAY.md](ASSAY.md).
