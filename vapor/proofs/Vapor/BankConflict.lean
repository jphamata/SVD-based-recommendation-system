/-!
# Theorem 8.1 — bank-conflict neutralisation by coprime stride

Shared memory with `b` banks maps word address `a` to bank `a mod b`. Lanes
`k = 0 … b−1` reading column `j` of a tile with row stride `s` touch banks
`(s·k + j) mod b`. If `gcd s b = 1` this map is injective on the lanes: one
cycle, no serialisation. For `b = 2ᵐ` the minimal pad is closed-form
(`padStride`), and proven both sufficient and minimal.

Core Lean has no `Nat.Coprime`; Euclid's lemma is derived here from `gcd`.
-/
namespace Vapor

/-- Euclid's lemma: `gcd a b = 1 ∧ b ∣ a·c → b ∣ c`. -/
theorem dvd_of_gcd_one_dvd_mul {a b c : Nat} (h : Nat.gcd a b = 1) (hd : b ∣ a * c) : b ∣ c := by
  have e : Nat.gcd (c * a) (c * b) = c := by rw [Nat.gcd_mul_left, h, Nat.mul_one]
  have h1 : b ∣ c * a := by rw [Nat.mul_comm]; exact hd
  have h2 : b ∣ c * b := Nat.dvd_mul_left b c
  have := Nat.dvd_gcd h1 h2
  rwa [e] at this

/-- **Theorem 8.1.** A stride coprime to the bank count maps distinct lanes
to distinct banks. -/
theorem bank_injective {s b j k₁ k₂ : Nat} (hcop : Nat.gcd s b = 1)
    (h₁ : k₁ < b) (h₂ : k₂ < b) (h : (s * k₁ + j) % b = (s * k₂ + j) % b) : k₁ = k₂ := by
  have key : ∀ {x y : Nat}, x ≤ y → y < b → (s * x + j) % b = (s * y + j) % b → x = y := by
    intro x y hxy hy hm
    have hsub : (s * y + j - (s * x + j)) % b = 0 := Nat.sub_mod_eq_zero_of_mod_eq hm.symm
    have e : s * y + j - (s * x + j) = s * (y - x) := by rw [Nat.mul_sub]; omega
    rw [e] at hsub
    have hd : b ∣ y - x := dvd_of_gcd_one_dvd_mul hcop (Nat.dvd_of_mod_eq_zero hsub)
    have : y - x = 0 := Nat.eq_zero_of_dvd_of_lt hd (by omega)
    omega
  rcases Nat.le_total k₁ k₂ with hle | hle
  · exact key hle h₂ h
  · exact (key hle h₁ h.symm).symm

/-- Bank touched by lane `lane` for a given stride. -/
def bankOf (stride lane banks : Nat) : Nat := (stride * lane) % banks

theorem gcd_two_of_odd {x : Nat} (hx : x % 2 = 1) : Nat.gcd x 2 = 1 := by
  rw [Nat.gcd_comm, Nat.gcd_rec, hx]; rfl

/-- An odd number is coprime to every power of two. -/
theorem gcd_odd_pow_two {x : Nat} (hx : x % 2 = 1) : ∀ m : Nat, Nat.gcd x (2 ^ m) = 1
  | 0 => by simp
  | m + 1 => by
      have ih := gcd_odd_pow_two hx m
      apply Nat.dvd_antisymm _ (Nat.one_dvd _)
      -- g = gcd x (2·2ᵐ) divides x, is coprime to 2, hence divides 2ᵐ
      have hg2 : Nat.gcd (Nat.gcd x (2 ^ (m + 1))) 2 = 1 := by
        apply Nat.dvd_antisymm _ (Nat.one_dvd _)
        have : Nat.gcd (Nat.gcd x (2 ^ (m + 1))) 2 ∣ Nat.gcd x 2 :=
          Nat.dvd_gcd (Nat.dvd_trans (Nat.gcd_dvd_left _ _) (Nat.gcd_dvd_left _ _)) (Nat.gcd_dvd_right _ _)
        rwa [gcd_two_of_odd hx] at this
      have hdm : Nat.gcd x (2 ^ (m + 1)) ∣ 2 ^ m := by
        have : Nat.gcd x (2 ^ (m + 1)) ∣ 2 * 2 ^ m := by
          rw [← Nat.pow_succ']; exact Nat.gcd_dvd_right _ _
        exact dvd_of_gcd_one_dvd_mul (Nat.gcd_comm _ _ ▸ hg2) this
      have := Nat.dvd_gcd (Nat.gcd_dvd_left x (2 ^ (m + 1))) hdm
      rwa [ih] at this

/-- Minimal coprime pad for `2ᵐ` banks (closed form of the search). -/
def padStride (s m : Nat) : Nat := if m = 0 then 0 else if s % 2 = 0 then 1 else 0

/-- The padded stride is coprime to the bank count. -/
theorem padStride_coprime (s m : Nat) : Nat.gcd (s + padStride s m) (2 ^ m) = 1 := by
  unfold padStride
  by_cases hm : m = 0
  · subst hm; simp
  · by_cases hs : s % 2 = 0
    · simp only [hm, hs, if_false, if_true]
      exact gcd_odd_pow_two (by omega) m
    · simp only [hm, hs, if_false]
      exact gcd_odd_pow_two (by omega) m

/-- The pad is minimal: a non-zero pad is only chosen when the unpadded
stride conflicts. -/
theorem padStride_minimal (s m : Nat) (hp : padStride s m ≠ 0) : Nat.gcd s (2 ^ m) ≠ 1 := by
  unfold padStride at hp
  by_cases hm : m = 0
  · simp [hm] at hp
  · by_cases hs : s % 2 = 0
    · intro h
      have h2 : 2 ∣ Nat.gcd s (2 ^ m) :=
        Nat.dvd_gcd (Nat.dvd_of_mod_eq_zero hs)
          (by obtain ⟨k, rfl⟩ : ∃ k, m = k + 1 := ⟨m - 1, by omega⟩
              exact ⟨2 ^ k, by rw [Nat.pow_succ, Nat.mul_comm]⟩)
      rw [h] at h2
      exact absurd (Nat.le_of_dvd Nat.one_pos h2) (by decide)
    · simp [hm, hs] at hp

/-- **Corollary.** With the computed pad, the `2ᵐ` lanes of a warp hit `2ᵐ`
distinct banks for any column offset. -/
theorem conflict_free (s m j k₁ k₂ : Nat) (h₁ : k₁ < 2 ^ m) (h₂ : k₂ < 2 ^ m)
    (h : ((s + padStride s m) * k₁ + j) % 2 ^ m = ((s + padStride s m) * k₂ + j) % 2 ^ m) :
    k₁ = k₂ :=
  bank_injective (padStride_coprime s m) h₁ h₂ h

/-- Decided instance of the standard 32-bank configuration: stride 32 (a
32-way conflict) is padded by exactly one word. -/
theorem pad_32_banks : padStride 32 5 = 1 ∧ Nat.gcd (32 + 1) (2 ^ 5) = 1 := by decide

end Vapor
