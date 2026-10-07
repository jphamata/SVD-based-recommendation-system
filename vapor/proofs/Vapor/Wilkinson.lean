/-!
# Theorem 7.2 — the Wilkinson envelope, as the runtime decides it

Higham's inner-product bound `|fl(Σ aᵢbᵢ) − Σ aᵢbᵢ| ≤ γₙ Σ|aᵢbᵢ|`,
`γₙ = n·u/(1 − n·u)`, `u = 2⁻²⁴`, is proved in `Higham.lean` (Lemma 3.1
and (3.4), for any summation order, from the standard model of rounding),
together with the bridge `withinEnvelope_of_bound`. What the runtime
actually executes is an exact integer decision over dyadic rationals scaled
to a common power of two, and this file proves things about that decision:

* `withinEnvelope d s a n` ≡ `d·(2²⁴ − n) ≤ n·s + a·(2²⁴ − n)` ⇔ `d ≤ γₙ·s + a`
  (cross-multiplied, no rounding in the checker itself);
* the decision is monotone in `n`: a result certified for the maximal
  extent of a semi-dynamic dimension is certified for every smaller one;
* the two-sided form: two substrates each within `ε` of the exact value
  are within `2ε` of each other.
-/
namespace Vapor

/-- `u⁻¹ = 2²⁴` for binary32 round-to-nearest. -/
def invUnitRoundoff : Nat := 16777216

/-- Exact envelope decision `d ≤ γₙ·s + a` with `γₙ = n/(2²⁴ − n)`, i.e.
`d·(2²⁴ − n) ≤ n·s + a·(2²⁴ − n)`. The absolute term `a` covers underflow
(gradual or flush-to-zero), which Higham's relative bound does not. -/
def withinEnvelope (d s a n : Nat) : Bool :=
  decide (n < invUnitRoundoff) &&
    decide (d * (invUnitRoundoff - n) ≤ n * s + a * (invUnitRoundoff - n))

/-- Monotonicity in the extent: certifying at the maximum covers all smaller extents. -/
theorem withinEnvelope_mono {d s a n n' : Nat} (h : withinEnvelope d s a n = true) (hn : n ≤ n')
    (hn' : n' < invUnitRoundoff) : withinEnvelope d s a n' = true := by
  simp only [withinEnvelope, Bool.and_eq_true, decide_eq_true_eq] at h ⊢
  refine ⟨hn', ?_⟩
  obtain ⟨_, h⟩ := h
  by_cases hda : d ≤ a
  · exact Nat.le_trans (Nat.mul_le_mul_right _ hda) (Nat.le_add_left _ _)
  · -- (d − a)(2²⁴ − n) ≤ n·s, and the left side shrinks while the right grows
    have e1 : d * (invUnitRoundoff - n) = (d - a) * (invUnitRoundoff - n) + a * (invUnitRoundoff - n) := by
      rw [← Nat.add_mul]; congr 1; omega
    have e2 : d * (invUnitRoundoff - n') = (d - a) * (invUnitRoundoff - n') + a * (invUnitRoundoff - n') := by
      rw [← Nat.add_mul]; congr 1; omega
    have h1 : (d - a) * (invUnitRoundoff - n) ≤ n * s := by omega
    have h2 : (d - a) * (invUnitRoundoff - n') ≤ (d - a) * (invUnitRoundoff - n) :=
      Nat.mul_le_mul_left _ (by omega)
    have h3 : n * s ≤ n' * s := Nat.mul_le_mul_right s hn
    omega

/-- Monotonicity in the magnitude sum `s` and in the absolute term `a`. -/
theorem withinEnvelope_mono_sa {d s s' a a' n : Nat} (h : withinEnvelope d s a n = true)
    (hs : s ≤ s') (ha : a ≤ a') : withinEnvelope d s' a' n = true := by
  simp only [withinEnvelope, Bool.and_eq_true, decide_eq_true_eq] at h ⊢
  refine ⟨h.1, Nat.le_trans h.2 (Nat.add_le_add (Nat.mul_le_mul_left n hs) (Nat.mul_le_mul_right _ ha))⟩

/-- Two-sided envelope: both substrates within `ε` of the exact value ⇒ within
`2ε` of each other (values scaled to a common dyadic unit). -/
theorem two_sided (hw dec exact ε : Int)
    (h₁ : hw - exact ≤ ε) (h₂ : exact - hw ≤ ε) (h₃ : dec - exact ≤ ε) (h₄ : exact - dec ≤ ε) :
    hw - dec ≤ 2 * ε ∧ dec - hw ≤ 2 * ε := by
  omega

end Vapor
