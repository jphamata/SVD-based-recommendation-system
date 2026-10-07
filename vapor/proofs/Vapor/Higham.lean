import Vapor.Wilkinson
/-!
# Higham's Lemma 3.1 and the inner-product bound, machine-checked

`Wilkinson.lean` proves things about the *decision* the runtime executes
(`withinEnvelope`), and until now the real-analysis lemma behind it —
Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Lemma 3.1
and (3.4) — was cited. Here it is proved, in core Lean (no Mathlib), over
exact rationals written with a common denominator:

* a rounding factor `1 + δ` with `|δ| ≤ u = k/D` is a natural `f` with
  `D − k ≤ f ≤ D + k` (`f = D·(1 + δ)`);
* a product of `n` such factors is `D^n·(1 + θₙ)`, and Lemma 3.1 states
  `|θₙ| ≤ γₙ = n·u/(1 − n·u)`, i.e. cross-multiplied, with no division:
  `|∏ f − Dⁿ| · (D − n·k) ≤ n·k · Dⁿ` (`higham_upper`, `higham_lower`);
* an inner product evaluated in *any* order under the standard model is
  `Σ tᵢ·∏ⱼ (1 + δᵢⱼ)` with at most `n` factors per term (pad with exact
  factors `f = D`), so `|ŝ − s| ≤ γₙ·Σ|tᵢ|` (`inner_product`) — the bound
  the envelope uses, valid for every summation tree, the canonical 16-lane
  tree included;
* that bound, at `u = 2⁻²⁴`, is exactly what `withinEnvelope` decides
  (`withinEnvelope_of_bound`).

What remains outside the proof is the *standard model* itself — that an
IEEE-754 binary32 round-to-nearest operation returns `(x op y)(1 + δ)`,
`|δ| ≤ 2⁻²⁴`, away from underflow (the absolute term `a` of the envelope
covers underflow) — a property of the hardware's specification.
-/
namespace Vapor

/-- The product of a list of naturals. -/
def prodN : List Nat → Nat
  | [] => 1
  | f :: fs => f * prodN fs

theorem prodN_le_pow (D k : Nat) : ∀ fs : List Nat, (∀ f ∈ fs, f ≤ D + k) → prodN fs ≤ (D + k) ^ fs.length
  | [], _ => by simp [prodN]
  | f :: fs, h => by
    have hf : f ≤ D + k := h f (List.mem_cons_self ..)
    have ih := prodN_le_pow D k fs (fun g hg => h g (List.mem_cons_of_mem _ hg))
    simp only [prodN, List.length_cons, Nat.pow_succ]
    calc f * prodN fs ≤ (D + k) * (D + k) ^ fs.length := Nat.mul_le_mul hf ih
      _ = (D + k) ^ fs.length * (D + k) := Nat.mul_comm _ _

theorem pow_le_prodN (D k : Nat) : ∀ fs : List Nat, (∀ f ∈ fs, D ≤ f + k) → (D - k) ^ fs.length ≤ prodN fs
  | [], _ => by simp [prodN]
  | f :: fs, h => by
    have hf : D - k ≤ f := by have := h f (List.mem_cons_self ..); omega
    have ih := pow_le_prodN D k fs (fun g hg => h g (List.mem_cons_of_mem _ hg))
    simp only [prodN, List.length_cons, Nat.pow_succ]
    calc (D - k) ^ fs.length * (D - k) = (D - k) * (D - k) ^ fs.length := Nat.mul_comm _ _
      _ ≤ f * prodN fs := Nat.mul_le_mul hf ih

/-- `(1 + u)ⁿ (1 − n·u) ≤ 1`, scaled by `D^(n+1)`. -/
theorem pow_succ_mul_le (D k : Nat) : ∀ n : Nat, n * k ≤ D → (D + k) ^ n * (D - n * k) ≤ D ^ (n + 1)
  | 0, _ => by simp
  | n + 1, h => by
    have h' : n * k ≤ D := by rw [Nat.succ_mul] at h; omega
    have ih := pow_succ_mul_le D k n h'
    -- e = D − (n+1)k, so D − nk = e + k and (D + k)·e ≤ D·(e + k)
    have hsplit : D - n * k = (D - (n + 1) * k) + k := by rw [Nat.succ_mul] at h ⊢; omega
    have key : (D + k) * (D - (n + 1) * k) ≤ D * (D - n * k) := by
      rw [hsplit, Nat.add_mul, Nat.mul_add, Nat.mul_comm k (D - (n + 1) * k)]
      apply Nat.add_le_add_left
      exact Nat.mul_le_mul_right k (Nat.sub_le _ _)
    calc (D + k) ^ (n + 1) * (D - (n + 1) * k)
        = (D + k) ^ n * ((D + k) * (D - (n + 1) * k)) := by rw [Nat.pow_succ, Nat.mul_assoc]
      _ ≤ (D + k) ^ n * (D * (D - n * k)) := Nat.mul_le_mul_left _ key
      _ = D * ((D + k) ^ n * (D - n * k)) := by rw [Nat.mul_left_comm]
      _ ≤ D * D ^ (n + 1) := Nat.mul_le_mul_left _ ih
      _ = D ^ (n + 1 + 1) := by rw [Nat.pow_succ D (n + 1), Nat.mul_comm]

/-- Bernoulli's inequality `(1 − u)ⁿ ≥ 1 − n·u`, scaled: `Dⁿ (D − n·k) ≤ D·(D − k)ⁿ`. -/
theorem bernoulli (D k : Nat) : ∀ n : Nat, n * k ≤ D → D ^ n * (D - n * k) ≤ D * (D - k) ^ n
  | 0, _ => by simp
  | n + 1, h => by
    have h' : n * k ≤ D := by rw [Nat.succ_mul] at h; omega
    have ih := bernoulli D k n h'
    -- with e = D − (n+1)k: D·e ≤ (e + k)(e + n·k) = (D − n·k)(D − k)
    have e1 : D - n * k = (D - (n + 1) * k) + k := by rw [Nat.succ_mul] at h ⊢; omega
    have e2 : D - k = (D - (n + 1) * k) + n * k := by rw [Nat.succ_mul] at h ⊢; omega
    have e3 : D = (D - (n + 1) * k) + n * k + k := by rw [Nat.succ_mul] at h ⊢; omega
    have key : D * (D - (n + 1) * k) ≤ (D - n * k) * (D - k) := by
      generalize hE : D - (n + 1) * k = E at e1 e2 e3 ⊢
      rw [e1, e2]
      conv => lhs; rw [e3]
      -- (E + nk + k)·E ≤ (E + k)(E + nk)
      have : (E + n * k + k) * E ≤ (E + k) * (E + n * k) := by
        have a1 : (E + n * k + k) * E = E * E + n * k * E + k * E := by rw [Nat.add_mul, Nat.add_mul]
        have a2 : (E + k) * (E + n * k) = E * E + E * (n * k) + (k * E + k * (n * k)) := by
          rw [Nat.add_mul, Nat.mul_add, Nat.mul_add]
        rw [a1, a2, Nat.mul_comm (n * k) E]
        omega
      exact this
    calc D ^ (n + 1) * (D - (n + 1) * k)
        = D ^ n * (D * (D - (n + 1) * k)) := by rw [Nat.pow_succ, Nat.mul_assoc, Nat.mul_comm D]
      _ ≤ D ^ n * ((D - n * k) * (D - k)) := Nat.mul_le_mul_left _ key
      _ = (D ^ n * (D - n * k)) * (D - k) := by rw [Nat.mul_assoc]
      _ ≤ (D * (D - k) ^ n) * (D - k) := Nat.mul_le_mul_right _ ih
      _ = D * (D - k) ^ (n + 1) := by rw [Nat.mul_assoc, Nat.pow_succ]

/-- Lemma 3.1, upper half: `(∏ f − Dⁿ)(D − n·k) ≤ n·k·Dⁿ`, i.e. `θₙ ≤ γₙ`. -/
theorem higham_upper (D k : Nat) (fs : List Nat) (hk : fs.length * k ≤ D) (h : ∀ f ∈ fs, f ≤ D + k) :
    prodN fs * (D - fs.length * k) ≤ D ^ fs.length * (D - fs.length * k) + fs.length * k * D ^ fs.length := by
  have hp := Nat.mul_le_mul_right (D - fs.length * k) (prodN_le_pow D k fs h)
  have ha := pow_succ_mul_le D k fs.length hk
  have hd : D ^ (fs.length + 1) = D ^ fs.length * (D - fs.length * k) + fs.length * k * D ^ fs.length := by
    rw [Nat.pow_succ, Nat.mul_comm (fs.length * k), ← Nat.mul_add]
    congr 1
    omega
  omega

/-- Lemma 3.1, lower half: `(Dⁿ − ∏ f)(D − n·k) ≤ n·k·Dⁿ`, i.e. `−θₙ ≤ γₙ`. -/
theorem higham_lower (D k : Nat) (fs : List Nat) (hk : fs.length * k ≤ D) (h : ∀ f ∈ fs, D ≤ f + k) :
    D ^ fs.length * (D - fs.length * k) ≤ prodN fs * (D - fs.length * k) + fs.length * k * D ^ fs.length := by
  generalize hn : fs.length = n at hk ⊢
  have hP : (D - k) ^ n ≤ prodN fs := hn ▸ pow_le_prodN D k fs h
  have hB := bernoulli D k n hk
  -- D·P ≥ D·(D − k)ⁿ ≥ Dⁿ(D − n·k)
  have hDP : D ^ n * (D - n * k) ≤ D * prodN fs := Nat.le_trans hB (Nat.mul_le_mul_left _ hP)
  by_cases hc : prodN fs ≤ D ^ n
  · -- P(D − nk) + nk·Dⁿ ≥ P(D − nk) + nk·P = P·D
    have : prodN fs * D ≤ prodN fs * (D - n * k) + n * k * D ^ n := by
      have e : prodN fs * D = prodN fs * (D - n * k) + n * k * prodN fs := by
        rw [Nat.mul_comm (n * k), ← Nat.mul_add]; congr 1; omega
      rw [e]; exact Nat.add_le_add_left (Nat.mul_le_mul_left _ hc) _
    rw [Nat.mul_comm D] at hDP
    exact Nat.le_trans hDP this
  · exact Nat.le_trans (Nat.mul_le_mul_right _ (Nat.le_of_lt (Nat.lt_of_not_le hc))) (Nat.le_add_right _ _)

/-- Lemma 3.1 in one statement: `|∏ f − Dⁿ| · (D − n·k) ≤ n·k·Dⁿ`. -/
theorem higham (D k : Nat) (fs : List Nat) (hk : fs.length * k ≤ D)
    (h : ∀ f ∈ fs, D ≤ f + k ∧ f ≤ D + k) :
    ((prodN fs : Int) - (D : Int) ^ fs.length).natAbs * (D - fs.length * k) ≤ fs.length * k * D ^ fs.length := by
  have hu := higham_upper D k fs hk (fun f hf => (h f hf).2)
  have hl := higham_lower D k fs hk (fun f hf => (h f hf).1)
  rw [← Int.natCast_pow]
  generalize prodN fs = P at hu hl
  generalize D ^ fs.length = Q at hu hl
  generalize D - fs.length * k = R at hu hl
  generalize fs.length * k = M at hu hl
  by_cases hc : Q ≤ P
  · have : ((P : Int) - (Q : Int)).natAbs = P - Q := by omega
    rw [this, Nat.sub_mul]
    omega
  · have : ((P : Int) - (Q : Int)).natAbs = Q - P := by omega
    rw [this, Nat.sub_mul]
    omega

/-- A term of an inner product: the exact product `t` and its rounding factors. -/
structure Term where
  t : Int
  fs : List Nat

/-- The computed sum, scaled by `Dⁿ`: `Σ tᵢ·∏ⱼ fᵢⱼ`. -/
def computed : List Term → Int
  | [] => 0
  | x :: xs => x.t * (prodN x.fs : Int) + computed xs

/-- The exact sum `Σ tᵢ`. -/
def exact : List Term → Int
  | [] => 0
  | x :: xs => x.t + exact xs

/-- `Σ |tᵢ|`. -/
def absSum : List Term → Nat
  | [] => 0
  | x :: xs => x.t.natAbs + absSum xs

/-- Higham (3.4), any summation order: every term carries `n` factors with
`|δ| ≤ k/D` (terms with fewer roundings are padded with exact factors
`f = D`), so `|ŝ − s| ≤ γₙ Σ |tᵢ|`, cross-multiplied:
`|Σ tᵢ∏fᵢⱼ − Dⁿ Σ tᵢ| · (D − n·k) ≤ n·k·Dⁿ·Σ|tᵢ|`. -/
theorem inner_product (D k n : Nat) (hk : n * k ≤ D) :
    ∀ xs : List Term, (∀ x ∈ xs, x.fs.length = n ∧ ∀ f ∈ x.fs, D ≤ f + k ∧ f ≤ D + k) →
      (computed xs - (D : Int) ^ n * exact xs).natAbs * (D - n * k) ≤ n * k * D ^ n * absSum xs
  | [], _ => by simp [computed, exact, absSum]
  | x :: xs, h => by
    have ⟨hlen, hf⟩ := h x (List.mem_cons_self ..)
    have ih := inner_product D k n hk xs (fun y hy => h y (List.mem_cons_of_mem _ hy))
    have hx := higham D k x.fs (hlen ▸ hk) hf
    rw [hlen] at hx
    -- split off the first term, then the triangle inequality
    have e : computed (x :: xs) - (D : Int) ^ n * exact (x :: xs)
        = x.t * ((prodN x.fs : Int) - (D : Int) ^ n) + (computed xs - (D : Int) ^ n * exact xs) := by
      simp only [computed, exact]
      rw [Int.mul_sub, Int.mul_add]
      rw [Int.mul_comm ((D : Int) ^ n) x.t]
      omega
    rw [e]
    have tri := Int.natAbs_add_le (x.t * ((prodN x.fs : Int) - (D : Int) ^ n)) (computed xs - (D : Int) ^ n * exact xs)
    rw [Int.natAbs_mul] at tri
    have step := Nat.mul_le_mul_right (D - n * k) tri
    rw [Nat.add_mul, Nat.mul_assoc] at step
    have h1 : x.t.natAbs * (((prodN x.fs : Int) - (D : Int) ^ n).natAbs * (D - n * k)) ≤ x.t.natAbs * (n * k * D ^ n) :=
      Nat.mul_le_mul_left _ hx
    simp only [absSum]
    rw [Nat.mul_add]
    calc _ ≤ _ := step
      _ ≤ x.t.natAbs * (n * k * D ^ n) + n * k * D ^ n * absSum xs := Nat.add_le_add h1 ih
      _ = n * k * D ^ n * x.t.natAbs + n * k * D ^ n * absSum xs := by rw [Nat.mul_comm x.t.natAbs]

/-- The bridge to the runtime: at `u = 2⁻²⁴` (`D = 2²⁴`, `k = 1`), the
inner-product bound `d·(2²⁴ − n) ≤ n·s` is exactly an accepting instance of
the decision `withinEnvelope` executes, for every absolute term `a`. -/
theorem withinEnvelope_of_bound (d s a n : Nat) (hn : n < invUnitRoundoff)
    (h : d * (invUnitRoundoff - n) ≤ n * s) : withinEnvelope d s a n = true := by
  simp only [withinEnvelope, Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨hn, Nat.le_trans h (Nat.le_add_right _ _)⟩

end Vapor
