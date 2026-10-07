/-!
# Definition 6.1 — the fixed Estrin tree for `eʳ − 1`, degree 7

The parenthesisation is data, so its multiplicative depth is a computed
fact rather than an assertion: `⌈log₂(7+1)⌉ = 3`, against 7 for Horner.
-/
namespace Vapor

inductive PExpr where
  | r
  | c (i : Nat)
  | add (a b : PExpr)
  | mul (a b : PExpr)
deriving DecidableEq

def PExpr.muldepth : PExpr → Nat
  | .mul a b => 1 + max a.muldepth b.muldepth
  | .add a b => max a.muldepth b.muldepth
  | _ => 0

/-- `P₀₇ = (P₀₁ + r²·P₂₃) + r⁴·(P₄₅ + r²·P₆₇)`. -/
def estrin7 : PExpr :=
  let u1 := PExpr.mul .r .r
  let u2 := PExpr.mul u1 u1
  let p (i : Nat) := PExpr.add (.c i) (.mul (.c (i + 1)) .r)
  .add (.add (p 0) (.mul u1 (p 2))) (.mul u2 (.add (p 4) (.mul u1 (p 6))))

def horner : Nat → PExpr
  | 0 => .c 7
  | n + 1 => .add (.c (6 - n)) (.mul .r (horner n))

theorem estrin7_depth : estrin7.muldepth = 3 := by decide
theorem horner7_depth : (horner 7).muldepth = 7 := by decide

end Vapor
