/-
  Field embedding: why an integer computation and its image in a prime
  field are the same computation (used by `Vapor.ZK`).

  A zero-knowledge circuit computes in ℤ/pℤ. Read back by the signed
  embedding (values above p/2 are negatives), a field element names a
  unique integer of (−p/2, p/2). So if the exact integer result is known to
  lie in that interval — which the integer-parity bound of
  `Vapor.IntegerParity` establishes for int8 contractions — the circuit's
  value and the kernel's value coincide: the witness the certified kernels
  compute is the only one the circuit accepts for that wire.
-/
import Vapor.IntegerParity

namespace Vapor

/-- The signed embedding ℤ → ℤ/pℤ is injective on (−p/2, p/2). -/
theorem embed_injective (p a b : Int) (hp : 0 < p)
    (ha : -p < 2 * a ∧ 2 * a < p) (hb : -p < 2 * b ∧ 2 * b < p)
    (h : p ∣ a - b) : a = b := by
  obtain ⟨k, hk⟩ := h
  have hk0 : k = 0 := by
    rcases Int.lt_trichotomy k 0 with hneg | hz | hpos
    · have h1 : p * k ≤ p * (-1) := Int.mul_le_mul_of_nonneg_left (by omega) (by omega)
      generalize p * k = m at hk h1
      omega
    · exact hz
    · have h1 : p * 1 ≤ p * k := Int.mul_le_mul_of_nonneg_left (by omega) (by omega)
      generalize p * k = m at hk h1
      omega
  subst hk0
  simp at hk
  omega

/-- **Field parity.** If the counted bound of the contraction stays below
p/2, any field value congruent to it — read in (−p/2, p/2) — *is* the exact
dot product: the circuit over ℤ/pℤ computes the integers. -/
theorem field_parity (p : Int) (A B : Nat) (ps : List (Int × Int)) (z : Int)
    (hbound : ∀ q ∈ ps, q.1.natAbs ≤ A ∧ q.2.natAbs ≤ B)
    (hp : 2 * ((ps.length * A * B : Nat) : Int) < p)
    (hz : -p < 2 * z ∧ 2 * z < p)
    (hcong : p ∣ z - dot ps) : z = dot ps := by
  have hb := dot_bound A B ps hbound
  apply embed_injective p z (dot ps) (by omega) hz _ hcong
  constructor <;> omega

/-- Together with `integer_parity`: when both bounds hold, the wrapping
32-bit machine, the exact integers and the field agree on the same value. -/
theorem machine_field_agree (p : Int) (A B : Nat) (ps : List (Int × Int)) (t : RTree) (z : Int)
    (hleaves : t.leaves.Perm (ps.map (fun q => q.1 * q.2)))
    (hbound : ∀ q ∈ ps, q.1.natAbs ≤ A ∧ q.2.natAbs ≤ B)
    (hadm : admissible ps.length A B = true)
    (hp : 2 * ((ps.length * A * B : Nat) : Int) < p)
    (hz : -p < 2 * z ∧ 2 * z < p)
    (hcong : p ∣ z - wrapS32 t.machine) : z = wrapS32 t.machine := by
  rw [integer_parity A B ps t hleaves hbound hadm] at hcong ⊢
  exact field_parity p A B ps z hbound hp hz hcong

end Vapor
