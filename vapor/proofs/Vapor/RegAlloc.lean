/-!
# Register-allocation checker (spill-freedom, per region)

The linear-scan allocator (`lib/vapor/kir/regalloc.ex`) is a heuristic and
is not trusted. Every allocation it produces is re-validated by `checkAlloc`,
extracted from this file into Elixir. A placement is
`(start, stop, reg, size)`: the value is live on `[start, stop)` and occupies
physical registers `reg … reg + size − 1` (an RVV LMUL group, or its AVX2/NEON
generalisation).

**Soundness:** if `checkAlloc` accepts, every group is aligned, inside the
register file and clear of reserved registers, and no two values that are
live at the same time occupy a common register. Since the backends emit no
spill code at all, an accepted region executes with every live value in its
register: spill-free and clobber-free by construction.
-/
namespace Vapor.RegAlloc

/-- `(start, stop, reg, size)` -/
abbrev Place := Nat × Nat × Nat × Nat

def overlaps (p q : Place) : Bool := decide (p.1 < q.2.1) && decide (q.1 < p.2.1)

def disjoint (p q : Place) : Bool :=
  decide (p.2.2.1 + p.2.2.2 ≤ q.2.2.1) || decide (q.2.2.1 + q.2.2.2 ≤ p.2.2.1)

def compatible (p q : Place) : Bool := !overlaps p q || disjoint p q

def placed (count : Nat) (reserved : List Nat) (p : Place) : Bool :=
  decide (0 < p.2.2.2) && decide (p.2.2.1 % p.2.2.2 = 0) && decide (p.2.2.1 + p.2.2.2 ≤ count) &&
    decide (p.1 < p.2.1) && (List.range p.2.2.2).all (fun k => !reserved.contains (p.2.2.1 + k))

/-- Pairwise compatibility in one right fold (carries the suffix seen so far). -/
def pairwiseOk (ps : List Place) : Bool :=
  (ps.foldr (fun p acc => (p :: acc.1, acc.2 && acc.1.all (compatible p))) ([], true)).2

def checkAlloc (count : Nat) (reserved : List Nat) (ps : List Place) : Bool :=
  ps.all (placed count reserved) && pairwiseOk ps

/-! ## Semantics -/

def live (p : Place) (t : Nat) : Prop := p.1 ≤ t ∧ t < p.2.1
def occupies (p : Place) (r : Nat) : Prop := p.2.2.1 ≤ r ∧ r < p.2.2.1 + p.2.2.2

/-- Two placements never hold a common register while both are live. -/
def NoClash (p q : Place) : Prop := ∀ t r, live p t → live q t → occupies p r → ¬ occupies q r

theorem compatible_sound {p q : Place} (h : compatible p q = true) : NoClash p q := by
  obtain ⟨a₁, b₁, r₁, s₁⟩ := p
  obtain ⟨a₂, b₂, r₂, s₂⟩ := q
  intro t r hl₁ hl₂ ho₁ ho₂
  simp only [live, occupies] at hl₁ hl₂ ho₁ ho₂
  simp only [compatible, overlaps, disjoint, Bool.or_eq_true, Bool.not_eq_true',
    Bool.and_eq_false_iff, decide_eq_false_iff_not, decide_eq_true_eq] at h
  omega

theorem foldr_fst (ps : List Place) :
    (ps.foldr (fun p acc => (p :: acc.1, acc.2 && acc.1.all (compatible p))) ([], true)).1 = ps := by
  induction ps with
  | nil => rfl
  | cons p ps ih => simp only [List.foldr_cons, ih]

theorem pairwiseOk_sound : ∀ ps : List Place, pairwiseOk ps = true → ps.Pairwise NoClash
  | [], _ => List.Pairwise.nil
  | p :: ps, h => by
      unfold pairwiseOk at h
      simp only [List.foldr_cons, foldr_fst, Bool.and_eq_true] at h
      obtain ⟨hrest, hall⟩ := h
      refine List.pairwise_cons.mpr ⟨fun q hq => compatible_sound ?_, pairwiseOk_sound ps hrest⟩
      exact (List.all_eq_true.mp hall) q hq

theorem placed_sound {count : Nat} {reserved : List Nat} {p : Place}
    (h : placed count reserved p = true) :
    0 < p.2.2.2 ∧ p.2.2.1 % p.2.2.2 = 0 ∧ p.2.2.1 + p.2.2.2 ≤ count ∧ p.1 < p.2.1 ∧
      ∀ r, occupies p r → r ∉ reserved := by
  simp only [placed, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨h0, h1⟩, h2⟩, h3⟩, h4⟩ := h
  refine ⟨h0, h1, h2, h3, fun r ho hr => ?_⟩
  have hk : r - p.2.2.1 ∈ List.range p.2.2.2 := List.mem_range.mpr (by simp only [occupies] at ho; omega)
  have := (List.all_eq_true.mp h4) _ hk
  have e : p.2.2.1 + (r - p.2.2.1) = r := by simp only [occupies] at ho; omega
  rw [e] at this
  simp only [Bool.not_eq_true'] at this
  have hc : reserved.contains r = true := List.contains_iff_mem.mpr hr
  rw [hc] at this
  exact Bool.noConfusion this

/-- **Soundness of the checker.** -/
theorem checkAlloc_sound {count : Nat} {reserved : List Nat} {ps : List Place}
    (h : checkAlloc count reserved ps = true) :
    (∀ p ∈ ps, 0 < p.2.2.2 ∧ p.2.2.1 % p.2.2.2 = 0 ∧ p.2.2.1 + p.2.2.2 ≤ count ∧ p.1 < p.2.1 ∧
        ∀ r, occupies p r → r ∉ reserved) ∧
      ps.Pairwise NoClash := by
  simp only [checkAlloc, Bool.and_eq_true] at h
  exact ⟨fun p hp => placed_sound ((List.all_eq_true.mp h.1) p hp), pairwiseOk_sound ps h.2⟩

end Vapor.RegAlloc
