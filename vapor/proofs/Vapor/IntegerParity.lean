/-!
# Theorem 7.1 — integer parity across substrates

Machine accumulation over `s32` is arithmetic in `ℤ/2³²ℤ`. We prove:

* reduction modulo `2³²` is a ring homomorphism, so **every** reduction tree
  of wrapped additions computes `wrap32` of the exact sum of its leaves;
* the exact sum is invariant under permutation of the leaves;
* the counted admissibility bound `K · |A|max · |B|max < 2³¹` keeps the exact
  dot product inside the `s32` range, so the machine result *is* the exact
  integer — on every substrate, in every accumulation order.

Everything is constructive; no axioms beyond Lean's core.
-/
namespace Vapor

/-- Residue modulo `2³²` (the bit pattern of an `s32`, as a natural residue). -/
def wrap32 (x : Int) : Int := x % 4294967296

/-- The `s32` value whose bit pattern is `x mod 2³²` (two's complement). -/
def wrapS32 (x : Int) : Int :=
  if x % 4294967296 < 2147483648 then x % 4294967296 else x % 4294967296 - 4294967296

theorem wrap32_add (a b : Int) : wrap32 (wrap32 a + wrap32 b) = wrap32 (a + b) := by
  unfold wrap32; rw [← Int.add_emod]

theorem wrap32_mul (a b : Int) : wrap32 (wrap32 a * wrap32 b) = wrap32 (a * b) := by
  unfold wrap32; rw [← Int.mul_emod]

theorem wrap32_idem (a : Int) : wrap32 (wrap32 a) = wrap32 a := by
  unfold wrap32; exact Int.emod_emod a _

/-- Addition in `ℤ/2³²ℤ` is associative. -/
theorem add32_assoc (a b c : Int) :
    wrap32 (wrap32 (a + b) + c) = wrap32 (a + wrap32 (b + c)) := by
  unfold wrap32
  rw [Int.emod_add_emod, Int.add_emod a ((b + c) % 4294967296), Int.emod_emod,
    ← Int.add_emod, Int.add_assoc]

/-- Addition in `ℤ/2³²ℤ` is commutative. -/
theorem add32_comm (a b : Int) : wrap32 (a + b) = wrap32 (b + a) := by
  rw [Int.add_comm]

/-- Multiplication in `ℤ/2³²ℤ` is associative. -/
theorem mul32_assoc (a b c : Int) :
    wrap32 (wrap32 (a * b) * c) = wrap32 (a * wrap32 (b * c)) := by
  unfold wrap32
  rw [Int.mul_emod, Int.emod_emod, ← Int.mul_emod,
    Int.mul_emod a ((b * c) % 4294967296), Int.emod_emod, ← Int.mul_emod, Int.mul_assoc]

/-- `wrapS32` only depends on the residue. -/
theorem wrapS32_wrap32 (x : Int) : wrapS32 (wrap32 x) = wrapS32 x := by
  unfold wrapS32 wrap32; rw [Int.emod_emod]

/-- Inside the `s32` range the machine value is the exact value. -/
theorem wrapS32_of_small {x : Int} (h1 : -2147483648 ≤ x) (h2 : x < 2147483648) : wrapS32 x = x := by
  unfold wrapS32
  by_cases hx : 0 ≤ x
  · have : x % 4294967296 = x := Int.emod_eq_of_lt hx (by omega)
    rw [this]; simp [h2]
  · have e : x % 4294967296 = x + 4294967296 := by
      have h0 : (x + 4294967296) % 4294967296 = x + 4294967296 :=
        Int.emod_eq_of_lt (by omega) (by omega)
      rw [← h0, Int.add_emod_right]
    rw [e]
    have : ¬ (x + 4294967296 < 2147483648) := by omega
    simp [this]

/-! ## Reduction trees: any accumulation order -/

/-- A reduction tree over integer leaves (a vector lane layout, a pairwise
tree, a sequential fold — every accumulation order is one of these). -/
inductive RTree where
  | leaf (v : Int)
  | node (l r : RTree)

namespace RTree

def exact : RTree → Int
  | leaf v => v
  | node l r => exact l + exact r

/-- Evaluation with a wrapping `s32` adder at every node. -/
def machine : RTree → Int
  | leaf v => wrap32 v
  | node l r => wrap32 (machine l + machine r)

def leaves : RTree → List Int
  | leaf v => [v]
  | node l r => leaves l ++ leaves r

theorem machine_eq_wrap_exact : ∀ t : RTree, machine t = wrap32 (exact t)
  | leaf _ => rfl
  | node l r => by
      simp only [machine, exact]
      rw [machine_eq_wrap_exact l, machine_eq_wrap_exact r, wrap32_add]

theorem sum_append : ∀ (a b : List Int), (a ++ b).sum = a.sum + b.sum
  | [], b => by simp
  | x :: a, b => by simp [List.sum_cons, sum_append a b, Int.add_assoc]

theorem exact_eq_sum : ∀ t : RTree, exact t = (leaves t).sum
  | leaf v => by simp [exact, leaves]
  | node l r => by simp [exact, leaves, exact_eq_sum l, exact_eq_sum r]

end RTree

/-- Sums are invariant under permutation (proved here; core Lean has no
`List.Perm.sum_eq` for `Int`). -/
theorem perm_sum_eq {l₁ l₂ : List Int} (h : l₁.Perm l₂) : l₁.sum = l₂.sum := by
  induction h with
  | nil => rfl
  | cons x _ ih => simp [List.sum_cons, ih]
  | swap x y l => simp [List.sum_cons]; omega
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

/-! ## The counted bound -/

/-- Exact dot product of paired operands. -/
def dot (ps : List (Int × Int)) : Int := (ps.map (fun p => p.1 * p.2)).sum

/-- The admissibility test carried by certificates: `K · A · B < 2³¹`. -/
def admissible (k a b : Nat) : Bool := decide (k * a * b < 2147483648)

/-- Monotone in every argument: certifying at the maximal extent of a
semi-dynamic dimension covers every smaller extent. -/
theorem admissible_mono {k k' a a' b b' : Nat} (h : admissible k' a' b' = true)
    (hk : k ≤ k') (ha : a ≤ a') (hb : b ≤ b') : admissible k a b = true := by
  simp only [admissible, decide_eq_true_eq] at h ⊢
  exact Nat.lt_of_le_of_lt (Nat.mul_le_mul (Nat.mul_le_mul hk ha) hb) h

theorem dot_bound (A B : Nat) :
    ∀ ps : List (Int × Int), (∀ p ∈ ps, p.1.natAbs ≤ A ∧ p.2.natAbs ≤ B) →
      (dot ps).natAbs ≤ ps.length * A * B
  | [], _ => by simp [dot]
  | p :: ps, h => by
      have hp := h p (by simp)
      have ih := dot_bound A B ps (fun q hq => h q (by simp [hq]))
      have hm : (p.1 * p.2).natAbs ≤ A * B := by
        rw [Int.natAbs_mul]; exact Nat.mul_le_mul hp.1 hp.2
      have hs : (dot (p :: ps)).natAbs ≤ (p.1 * p.2).natAbs + (dot ps).natAbs := by
        simp only [dot, List.map_cons, List.sum_cons]; exact Int.natAbs_add_le _ _
      have e : (p :: ps).length * A * B = A * B + ps.length * A * B := by
        simp [List.length_cons, Nat.succ_mul, Nat.add_mul]; omega
      omega

/-- **Theorem 7.1 (integer parity).** For any reduction tree whose leaves are
a permutation of the products `aᵢ·bᵢ`, if the counted bound holds, the
wrapping machine evaluation read back as `s32` equals the exact dot product.
Hence all substrates agree bit for bit, whatever their accumulation order. -/
theorem integer_parity (A B : Nat) (ps : List (Int × Int)) (t : RTree)
    (hleaves : t.leaves.Perm (ps.map (fun p => p.1 * p.2)))
    (hbound : ∀ p ∈ ps, p.1.natAbs ≤ A ∧ p.2.natAbs ≤ B)
    (hadm : admissible ps.length A B = true) :
    wrapS32 t.machine = dot ps := by
  have hlt : ps.length * A * B < 2147483648 := by simpa [admissible] using hadm
  have hsum : t.exact = dot ps := by
    rw [RTree.exact_eq_sum, perm_sum_eq hleaves]; rfl
  have hb := dot_bound A B ps hbound
  rw [RTree.machine_eq_wrap_exact, wrapS32_wrap32, hsum]
  apply wrapS32_of_small <;> omega

end Vapor
