# Mathematics by machine: proofs with certificates (0.11)

> Request, translated: "weigh something similar to AlphaProof for mathematics
> (geometry, topology, and beyond)". Scrutiny:
> [DIRECTIVE.md §14](DIRECTIVE.md). Tests: `prove_test.exs`. Console:
> *Discover → Mathematics*.

## 1. What AlphaProof and AlphaGeometry are, and what can be done here

AlphaProof is reinforcement learning over Lean: a language model
proposes steps, Lean's kernel checks them. AlphaGeometry combines a
symbolic deduction engine with a language model that proposes
**auxiliary constructions**. What the two have in common, and what matters,
is the division: **search proposes, a formal verifier decides**. With no GPU and
no weights here, vapor does the part that does not need a trained network — the
verifier and a complete symbolic search for a large class of
theorems — and is honest about what is missing (readable proofs, learned
auxiliary constructions).

## 2. Geometry by the algebraic method (`Vapor.Prove.prove/1`)

A figure is a **construction**: free parameters, and points defined
from the previous ones (midpoint, intersection of two lines, foot of a
perpendicular, circumcentre, rational point of the unit circle
`((1−t²)/(1+t²), 2t/(1+t²))`, point at a ratio). Each coordinate is then
an **exact rational function** of the parameters, with integer coefficients, and
each claim (collinear, parallel, perpendicular, equal
lengths, concyclic, the same point) is a polynomial in the coordinates. The
theorem holds for every figure in general position **if and only if the numerator
of the claim is the zero polynomial** — the family of Wu's methods (1978) and
of Gröbner bases, with no need to triangularize because the constructions
are explicit.

- **Certificate**: the expanded numerator (zero) and the **non-degeneracy
  conditions** — the denominators that appeared (for example, for the
  Euler line: `u²v² + u²w² − 2uv³ − 2uvw² + v⁴ + 2v²w² + w⁴ ≠ 0`, the
  non-degenerate triangle).
- **Checking in another arithmetic**: the same construction in **exact
  rationals** at random integer points up to 10⁹ (Schwartz–Zippel: a
  nonzero polynomial of degree *D* vanishes at a point drawn from a
  set of size *N* with probability ≤ D/N). It is independent of the
  polynomial algebra, **not of the construction formulas** (the same
  code builds them in both arithmetics): an error in a formula — the foot of the
  perpendicular, say — would pass both. What catches it are the
  known theorems (a wrong formula breaks the Euler line) and the
  false controls.
- **Degenerate constructions**: a figure that is 0/0 for every value of the
  parameters (the foot of a perpendicular onto the "line" through a single point) is
  reported as `{:degenerate, :construction}`, never "proved" (tested).
- **Measured**: concurrent medians, altitudes and perpendicular bisectors, **Euler
  line**, the 1 : 2 ratio of the centroid on OH, **nine-point circle**
  (two forms), midline, Varignon, Thales, **Pappus**, **Simson** —
  proved; the **controls**, false statements of the same form (centroid
  on the circumcircle, orthocentre = circumcentre, perpendicular midline,
  Simson off the circle, the nine-point circle through a vertex) —
  refuted by the symbolic proof **and** by the checking. All in < 0.1 s,
  except symbolic Simson (~100 s: the growth of the degrees without polynomial
  GCD — checked only in exact rationals in the tests).

## 3. Conjecture and prove (`Vapor.Prove.discover/2`)

What AlphaGeometry calls "deduction": without anything being asked,
every triple of points of a triangle's figure (vertices, midpoints,
feet of the altitudes, centroid, orthocentre, circumcentre, nine-point
centre) is tested for a line and every quadruple for a circle, in floating
point on two random figures; the **trivial** triples (three points
of a defining line) are discarded; the rest is **proved
symbolically** (zero numerator, as in §2) and checked in exact
rationals. Measured: of 1,001 candidates, 29 survive and 29 are proved
(~7 s) — among them the **Euler line** (O, G, H and the centre N), the
**third median** and the **third altitude**, and the **15 quadrilaterals** of the
nine-point circle, plus less famous circles (B, C and the feet
of the altitudes from B and C; B, the midpoints of AB and BC, and O).

## 4. Topology

**Homology** (`betti/2`): Betti numbers of simplicial complexes
over GF(2) (elimination with bitsets) and over ℚ (Bareiss elimination,
exact integers), and the Euler characteristic. Measured:

| complex | χ | GF(2) | ℚ |
|---|---|---|---|
| sphere | 2 | 1 0 1 | 1 0 1 |
| torus | 0 | 1 2 1 | 1 2 1 |
| Klein bottle | 0 | 1 2 1 | **1 1 0** |
| RP² | 1 | 1 1 1 | **1 0 0** |
| Möbius strip | 0 | 1 1 0 | 1 1 0 |

Over GF(2) the torus and the Klein bottle look the same; over ℚ they do not — the
difference is the **torsion** ℤ/2, and that is how the machine tells the
non-orientable surfaces apart.

**Persistent homology** (`persistence/2`): Vietoris–Rips up to
triangles, column reduction over GF(2). A noisy loop has **one**
long H₁ bar (1.42); a blob of random points (the control), bars
of at most 0.13 — the topological data analysis tool, with the
control that separates structure from noise.

## 5. Limits

- Geometry of **equalities** (incidence, perpendicularity,
  lengths, circles); inequalities and oriented angles are out. The
  constructions must be explicit (a point defined by two
  quadratic conditions — two circles — would require Wu's full
  triangularization).
- No **readable** step-by-step proof (which AlphaGeometry produces); the
  certificate is algebraic.
- No Lean in this round (`lake` is not on this machine): the
  geometric proofs are not exported to a proof assistant.
- Number theory, analysis, combinatorics: out. "And beyond" was
  weighed (DIRECTIVE §14) and stayed in the TODO with the path — export
  certificates to Lean and a model-guided search once there are
  weights.
