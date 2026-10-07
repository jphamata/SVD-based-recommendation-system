/-!
# Binary32 and the bit-exact rewrite rules (`Vapor.Compile.Rewrite`)

The rewriter admits an identity only if it holds *bit for bit* for every
input. This file proves the rules it uses, and refutes the one it refuses,
against an IEEE-754 model built from first principles in core Lean (no
reals, no Mathlib): a binary32 datum is its 32-bit pattern, its value is an
integer count of the smallest subnormal `2⁻¹⁴⁹`, and an operation's result
is specified the way the standard specifies it — **any** finite pattern
nearest to the exact result (§4.3), with the sign rules for exact zeros
(§6.3). The nearest relation here omits the ties-to-even clause; that only
admits *more* candidate results, so "every admissible result equals `x`"
is a stronger statement than the standard needs.

  * `x · 1 → x`, `1 · x → x`: the only correctly rounded result is `x`
    (`mul_one_unique`, `one_mul_unique`), and `x` is one (`mul_one_admissible`);
  * `x + (−0) → x`, `(−0) + x → x`, `x − (+0) → x` (`add_negZero_unique`, …);
  * `x + (+0) → x` is **unsound**: `(−0) + (+0)` can only be `+0`
    (`add_posZero_refuted`);
  * `neg (neg x) = x` (`neg_neg`).

Scope, stated rather than hidden: finite operands (zeros and subnormals
included). NaN and ∞ are outside the model on purpose: hardware disagrees
on NaN bits (x86 quiets a signalling NaN and keeps a payload, RISC-V
returns the canonical NaN, Arm with default-NaN mode too), so no rewrite
can be bit-exact on NaN across substrates — and vapor's oracle refuses
non-finite values, so its certificates are statements about finite
executions only.

`units` is extracted to Elixir and checked against `Vapor.F32`'s decoder
on generated patterns (`test/vapor/extracted_conformance_test.exs`): the
model's reading of a bit pattern is the implementation's.
-/
namespace Vapor.Binary32

/-- Sign bit, biased exponent field, trailing significand field. -/
def sgn (b : Nat) : Nat := b / 2147483648
def ex (b : Nat) : Nat := (b / 8388608) % 256
def man (b : Nat) : Nat := b % 8388608

/-- A finite binary32 pattern (zeros, subnormals and normals; not ∞ or NaN). -/
def Finite (b : Nat) : Prop := b < 4294967296 ∧ ex b < 255

/-- `|value| / 2⁻¹⁴⁹`: the significand, shifted by the exponent. -/
def mag (b : Nat) : Nat :=
  if ex b = 0 then man b else (man b + 8388608) * 2 ^ (ex b - 1)

/-- The exact value, in units of `2⁻¹⁴⁹`. -/
def units (b : Nat) : Int :=
  if sgn b = 1 then -((mag b : Nat) : Int) else ((mag b : Nat) : Int)

def posZero : Nat := 0
def negZero : Nat := 2147483648
def one : Nat := 1065353216

/-- Negation flips the sign bit. -/
def neg (b : Nat) : Nat := if b < 2147483648 then b + 2147483648 else b - 2147483648

/-- `f` is a finite pattern at least as close to the exact value `v · 2^(−149−s)`
as any other finite pattern (round to nearest, §4.3; ties unconstrained). -/
def Nearest (v : Int) (s : Nat) (f : Nat) : Prop :=
  Finite f ∧ ∀ g, Finite g → Int.natAbs (v - units f * 2 ^ s) ≤ Int.natAbs (v - units g * 2 ^ s)

/-- An admissible result of `x · y` (exact product scaled by `2⁻²⁹⁸`); an exact
zero takes the exclusive-or of the operands' signs (§6.3). -/
def MulRN (x y f : Nat) : Prop :=
  if units x * units y = 0 then f = ((sgn x + sgn y) % 2) * 2147483648 else Nearest (units x * units y) 149 f

/-- An admissible result of `x + y`; an exact zero is `+0` unless both operands
are zeros of one sign (§6.3, round to nearest). -/
def AddRN (x y f : Nat) : Prop :=
  if units x + units y = 0 then f = (if sgn x = sgn y then sgn x * 2147483648 else 0) else Nearest (units x + units y) 0 f

/-- `x − y` is `x + (−y)` (§5.4.1). -/
def SubRN (x y f : Nat) : Prop := AddRN x (neg y) f

-- ------------------------------------------------------------- the model --

theorem sgn_le (b : Nat) (h : b < 4294967296) : sgn b = 0 ∨ sgn b = 1 := by
  unfold sgn; omega

theorem bits_eq (b : Nat) : b = sgn b * 2147483648 + ex b * 8388608 + man b := by
  unfold sgn ex man; omega

theorem man_lt (b : Nat) : man b < 8388608 := by unfold man; omega

theorem two_pow_pos (n : Nat) : 0 < 2 ^ n := Nat.two_pow_pos n

theorem mag_normal_ge (b : Nat) (h : ex b ≠ 0) : 8388608 ≤ mag b := by
  unfold mag
  rw [if_neg h]
  have := two_pow_pos (ex b - 1)
  calc 8388608 = 8388608 * 1 := by omega
    _ ≤ (man b + 8388608) * 2 ^ (ex b - 1) := Nat.mul_le_mul (by omega) this

theorem mag_zero (b : Nat) (h : mag b = 0) : ex b = 0 ∧ man b = 0 := by
  by_cases e : ex b = 0
  · unfold mag at h; rw [if_pos e] at h; exact ⟨e, h⟩
  · have := mag_normal_ge b e; omega

/-- A finite value has one encoding (up to the sign of zero): exponent and
significand are determined by the magnitude. -/
theorem mag_inj (a b : Nat) (h : mag a = mag b) : ex a = ex b ∧ man a = man b := by
  have ma := man_lt a
  have mb := man_lt b
  by_cases ea : ex a = 0 <;> by_cases eb : ex b = 0
  · unfold mag at h; rw [if_pos ea, if_pos eb] at h; omega
  · have := mag_normal_ge b eb
    have e1 : mag a = man a := by unfold mag; rw [if_pos ea]
    omega
  · have := mag_normal_ge a ea
    have e1 : mag b = man b := by unfold mag; rw [if_pos eb]
    omega
  · unfold mag at h
    rw [if_neg ea, if_neg eb] at h
    -- a larger exponent puts the magnitude in a higher binade: [2²³·2ᵉ⁻¹, 2²⁴·2ᵉ⁻¹)
    have binade : ∀ p q m n : Nat, m < 8388608 → n < 8388608 → p < q →
        (m + 8388608) * 2 ^ p ≠ (n + 8388608) * 2 ^ q := by
      intro p q m n hm _ hpq heq
      have hq : 2 ^ q = 2 ^ p * 2 ^ (q - p) := by rw [← Nat.pow_add]; congr 1; omega
      have h2 : 2 ≤ 2 ^ (q - p) := by
        have : 2 ^ 1 ≤ 2 ^ (q - p) := Nat.pow_le_pow_right (by decide) (by omega)
        simpa using this
      have hp := two_pow_pos p
      have lhs : (m + 8388608) * 2 ^ p < 16777216 * 2 ^ p := Nat.mul_lt_mul_of_pos_right (by omega) hp
      have rhs : 16777216 * 2 ^ p ≤ (n + 8388608) * 2 ^ q := by
        rw [hq]
        calc 16777216 * 2 ^ p = 8388608 * (2 ^ p * 2) := by omega
          _ ≤ (n + 8388608) * (2 ^ p * 2 ^ (q - p)) :=
              Nat.mul_le_mul (by omega) (Nat.mul_le_mul_left (2 ^ p) h2)
      omega
    have same : ex a - 1 = ex b - 1 := by
      rcases Nat.lt_trichotomy (ex a - 1) (ex b - 1) with lt | eq | gt
      · exact absurd h (binade _ _ _ _ ma mb lt)
      · exact eq
      · exact absurd h.symm (binade _ _ _ _ mb ma gt)
    rw [same] at h
    have := Nat.eq_of_mul_eq_mul_right (two_pow_pos (ex b - 1)) h
    omega

/-- Equal non-zero values are equal patterns. -/
theorem units_inj (a b : Nat) (fa : Finite a) (fb : Finite b) (h : units a = units b) (nz : units a ≠ 0) : a = b := by
  have ra := bits_eq a
  have rb := bits_eq b
  have hm : mag a = mag b ∧ sgn a = sgn b := by
    unfold units at h nz
    rcases sgn_le a fa.1 with sa | sa <;> rcases sgn_le b fb.1 with sb | sb <;> simp [sa, sb] at h nz ⊢ <;> omega
  have ⟨he, hn⟩ := mag_inj a b hm.1
  omega

theorem units_zero (b : Nat) (h : units b = 0) : b = sgn b * 2147483648 := by
  have hm : mag b = 0 := by unfold units at h; split at h <;> omega
  have ⟨e, m⟩ := mag_zero b hm
  have := bits_eq b
  omega

theorem units_one : units one = 2 ^ 149 := by simp [units, sgn, mag, ex, man, one]

theorem pow149_ne : (2 : Int) ^ 149 ≠ 0 := by
  have : (0 : Int) < 2 ^ 149 := Int.pow_pos (by decide)
  omega

theorem finite_zero_neg : Finite negZero := by unfold Finite ex negZero; decide
theorem finite_zero_pos : Finite posZero := by unfold Finite ex posZero; decide
theorem units_negZero : units negZero = 0 := by decide
theorem units_posZero : units posZero = 0 := by decide

/-- The exact value is always nearest to itself. -/
theorem nearest_self (x : Nat) (s : Nat) (fx : Finite x) : Nearest (units x * 2 ^ s) s x :=
  ⟨fx, fun _ _ => by simp⟩

/-- …and, if non-zero, nothing else is. -/
theorem nearest_exact (x f : Nat) (s : Nat) (fx : Finite x) (nz : units x ≠ 0)
    (h : Nearest (units x * 2 ^ s) s f) : f = x := by
  have d := h.2 x fx
  simp only [Int.sub_self, Int.natAbs_zero] at d
  have e : units x * 2 ^ s = units f * 2 ^ s := by omega
  have p : (2 : Int) ^ s ≠ 0 := by
    have : (0 : Int) < 2 ^ s := Int.pow_pos (by decide)
    omega
  have := Int.eq_of_mul_eq_mul_right p e
  exact (units_inj x f fx h.1 this nz).symm

-- --------------------------------------------------------- the rewrites --

theorem mul_one_admissible (x : Nat) (fx : Finite x) : MulRN x one x := by
  unfold MulRN
  rw [units_one]
  by_cases z : units x * 2 ^ 149 = 0
  · rw [if_pos z]
    have : units x = 0 := by
      rcases Int.mul_eq_zero.mp z with h | h
      · exact h
      · exact absurd h pow149_ne
    have := units_zero x this
    have s1 : sgn one = 0 := by decide
    rcases sgn_le x fx.1 with s | s <;> rw [s1] <;> simp [s] at this ⊢ <;> omega
  · rw [if_neg z]; exact nearest_self x 149 fx

/-- `x · 1 → x` is exact: `x` is the only correctly rounded product. -/
theorem mul_one_unique (x f : Nat) (fx : Finite x) (h : MulRN x one f) : f = x := by
  unfold MulRN at h
  rw [units_one] at h
  by_cases z : units x * 2 ^ 149 = 0
  · rw [if_pos z] at h
    have ux : units x = 0 := by
      rcases Int.mul_eq_zero.mp z with h | h
      · exact h
      · exact absurd h pow149_ne
    have := units_zero x ux
    have s1 : sgn one = 0 := by decide
    rw [s1] at h
    rcases sgn_le x fx.1 with s | s <;> simp [s] at h this ⊢ <;> omega
  · rw [if_neg z] at h
    have nz : units x ≠ 0 := fun e => z (by rw [e]; simp)
    exact nearest_exact x f 149 fx nz h

theorem mul_comm_rn (x y f : Nat) : MulRN x y f ↔ MulRN y x f := by
  unfold MulRN
  rw [Int.mul_comm (units x), Nat.add_comm (sgn x)]

/-- `1 · x → x`. -/
theorem one_mul_unique (x f : Nat) (fx : Finite x) (h : MulRN one x f) : f = x :=
  mul_one_unique x f fx ((mul_comm_rn one x f).mp h)

/-- `x + (−0) → x` is exact, signed zeros included. -/
theorem add_negZero_unique (x f : Nat) (fx : Finite x) (h : AddRN x negZero f) : f = x := by
  unfold AddRN at h
  rw [units_negZero, Int.add_zero] at h
  have sn : sgn negZero = 1 := by decide
  rw [sn] at h
  by_cases z : units x = 0
  · rw [if_pos z] at h
    have := units_zero x z
    rcases sgn_le x fx.1 with s | s <;> simp [s] at h this ⊢ <;> omega
  · rw [if_neg z] at h
    have h' : Nearest (units x * 2 ^ 0) 0 f := by simpa using h
    exact nearest_exact x f 0 fx z h'

theorem add_negZero_admissible (x : Nat) (fx : Finite x) : AddRN x negZero x := by
  unfold AddRN
  rw [units_negZero, Int.add_zero]
  have sn : sgn negZero = 1 := by decide
  rw [sn]
  by_cases z : units x = 0
  · rw [if_pos z]
    have := units_zero x z
    rcases sgn_le x fx.1 with s | s <;> simp [s] at this ⊢ <;> omega
  · rw [if_neg z]
    have := nearest_self x 0 fx
    simpa using this

theorem add_comm_rn (x y f : Nat) : AddRN x y f ↔ AddRN y x f := by
  unfold AddRN
  rw [Int.add_comm (units x)]
  by_cases z : units y + units x = 0
  · rw [if_pos z, if_pos z]
    by_cases s : sgn x = sgn y
    · rw [if_pos s, if_pos s.symm, s]
    · rw [if_neg s, if_neg (Ne.symm s)]
  · rw [if_neg z, if_neg z]

/-- `(−0) + x → x`. -/
theorem negZero_add_unique (x f : Nat) (fx : Finite x) (h : AddRN negZero x f) : f = x :=
  add_negZero_unique x f fx ((add_comm_rn negZero x f).mp h)

/-- `x − (+0) → x`: subtracting `+0` adds `−0`. -/
theorem sub_posZero_unique (x f : Nat) (fx : Finite x) (h : SubRN x posZero f) : f = x := by
  unfold SubRN at h
  have : neg posZero = negZero := by decide
  rw [this] at h
  exact add_negZero_unique x f fx h

/-- `x + (+0) → x` is **unsound**: for `x = −0` the only admissible sum is `+0`. -/
theorem add_posZero_refuted : ∀ f, AddRN negZero posZero f → f = posZero ∧ f ≠ negZero := by
  intro f h
  unfold AddRN at h
  rw [units_negZero, units_posZero] at h
  have : sgn negZero ≠ sgn posZero := by decide
  simp [this] at h
  subst h
  decide

theorem neg_neg (x : Nat) (h : x < 4294967296) : neg (neg x) = x := by
  unfold neg
  by_cases a : x < 2147483648
  · rw [if_pos a, if_neg (by omega)]; omega
  · rw [if_neg a, if_pos (by omega)]; omega

end Vapor.Binary32
