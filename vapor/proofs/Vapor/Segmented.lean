/-!
# Theorem 9.1 — the affine SSM monoid and its segmented lifting

The state-space recurrence `h ↦ a·h + b` composes as an affine map. The
composition `(a₂, b₂) ⊙ (a₁, b₁) = (a₂a₁, a₂b₁ + b₂)` ("apply 1, then 2") is
associative with neutral `(1, 0)` — which is what licenses parallel
(tree-shaped) scans of the recurrence. The segmented lifting
`F_seg(M) = M × Bool` restarts at segment boundaries (ragged batches
without padding) and is again a monoid, for *any* monoid `M`.

We state the affine monoid over `Int` (exact integer/fixed-point states);
the float instantiation is not associative, which is precisely why float
scans are certified by envelope rather than by these equalities.
-/
namespace Vapor

/-- Composition of affine maps: `x` first, then `y`. -/
def affineOp (x y : Int × Int) : Int × Int := (y.1 * x.1, y.1 * x.2 + y.2)

def affineNeutral : Int × Int := (1, 0)

/-- Applying an affine map to a state. -/
def affineApply (f : Int × Int) (h : Int) : Int := f.1 * h + f.2

theorem affineOp_assoc (x y z : Int × Int) :
    affineOp (affineOp x y) z = affineOp x (affineOp y z) := by
  obtain ⟨a₁, b₁⟩ := x; obtain ⟨a₂, b₂⟩ := y; obtain ⟨a₃, b₃⟩ := z
  simp only [affineOp, Prod.mk.injEq]
  constructor
  · rw [Int.mul_assoc]
  · rw [Int.mul_add, Int.add_assoc, Int.mul_assoc]

theorem affineOp_neutral_left (x : Int × Int) : affineOp affineNeutral x = x := by
  obtain ⟨a, b⟩ := x; simp [affineOp, affineNeutral]

theorem affineOp_neutral_right (x : Int × Int) : affineOp x affineNeutral = x := by
  obtain ⟨a, b⟩ := x; simp [affineOp, affineNeutral]

/-- The monoid is the right one: composing then applying equals applying in
sequence (so a scan of `affineOp` computes the recurrence). -/
theorem affineOp_apply (x y : Int × Int) (h : Int) :
    affineApply (affineOp x y) h = affineApply y (affineApply x h) := by
  obtain ⟨a₁, b₁⟩ := x; obtain ⟨a₂, b₂⟩ := y
  simp only [affineApply, affineOp]
  rw [Int.mul_add, Int.mul_assoc, Int.add_assoc]

/-! ## Segmented lifting -/

section Seg
variable {M : Type} (op : M → M → M) (e : M)

/-- `(a, fa) ⊗ (b, fb) = (fb ? b : a ⊙ b, fa ∨ fb)`: a set flag starts a new
segment, discarding the carried prefix. -/
def segOp (x y : M × Bool) : M × Bool :=
  (if y.2 then y.1 else op x.1 y.1, x.2 || y.2)

theorem segOp_assoc (hassoc : ∀ a b c, op (op a b) c = op a (op b c)) (x y z : M × Bool) :
    segOp op (segOp op x y) z = segOp op x (segOp op y z) := by
  obtain ⟨a, fa⟩ := x; obtain ⟨b, fb⟩ := y; obtain ⟨c, fc⟩ := z
  cases fa <;> cases fb <;> cases fc <;> simp [segOp, hassoc]

theorem segOp_neutral_left (hl : ∀ a, op e a = a) (x : M × Bool) : segOp op (e, false) x = x := by
  obtain ⟨a, fa⟩ := x; cases fa <;> simp [segOp, hl]

theorem segOp_neutral_right (hr : ∀ a, op a e = a) (x : M × Bool) : segOp op x (e, false) = x := by
  obtain ⟨a, fa⟩ := x; simp [segOp, hr]

end Seg

/-- The segmented affine SSM combine, specialised for extraction. -/
def segAffine (x y : (Int × Int) × Bool) : (Int × Int) × Bool :=
  (if y.2 then y.1 else affineOp x.1 y.1, x.2 || y.2)

theorem segAffine_eq (x y : (Int × Int) × Bool) : segAffine x y = segOp affineOp x y := rfl

/-- **Theorem 9.1.** The segmented affine monoid is associative, with
neutral `((1, 0), false)` on both sides. -/
theorem segAffine_monoid :
    (∀ x y z, segAffine (segAffine x y) z = segAffine x (segAffine y z)) ∧
    (∀ x, segAffine (affineNeutral, false) x = x) ∧
    (∀ x, segAffine x (affineNeutral, false) = x) :=
  ⟨fun x y z => segOp_assoc affineOp affineOp_assoc x y z,
   fun x => segOp_neutral_left affineOp affineNeutral affineOp_neutral_left x,
   fun x => segOp_neutral_right affineOp affineNeutral affineOp_neutral_right x⟩

end Vapor
