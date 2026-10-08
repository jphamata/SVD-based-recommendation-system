import Vapor
import Extract
/-!
`lake exe vapor-extract <extracted.ex> <conformance_test.exs>`

1. Extracts the proven definitions into `Vapor.Extracted` (Elixir).
2. Evaluates the *same* definitions in Lean on generated inputs and writes an
   ExUnit suite asserting the Elixir extraction reproduces every value.
3. Stamps both files with the FNV-1a/64 digest of the Lean sources, so the
   Elixir test suite can detect a stale extraction without a Lean toolchain.
-/
open Lean

/-- Source files covered by the digest (sorted, relative to `proofs/`). -/
def sources : List String :=
  ["Extract.lean", "Main.lean", "Vapor.lean", "Vapor/BankConflict.lean", "Vapor/Binary32.lean", "Vapor/Estrin.lean",
   "Vapor/FieldEmbedding.lean", "Vapor/Higham.lean", "Vapor/IntegerParity.lean", "Vapor/RegAlloc.lean", "Vapor/Segmented.lean",
   "Vapor/Wilkinson.lean"]

def fnv1a64 (bytes : ByteArray) (h : UInt64 := 0xcbf29ce484222325) : UInt64 :=
  bytes.foldl (fun h b => (h ^^^ b.toUInt64) * 0x100000001b3) h

def hex16 (x : UInt64) : String :=
  let s := String.ofList (Nat.toDigits 16 x.toNat)
  "".pushn '0' (16 - s.length) ++ s

/-- Roots of extraction and the theorems that justify them. -/
def roots : List (Name × String) :=
  [(`Vapor.RegAlloc.checkAlloc,
      "Register-allocation checker. Sound by `Vapor.RegAlloc.checkAlloc_sound`: an accepted\n  assignment is aligned, in range, avoids reserved registers, and never maps two\n  simultaneously-live values to a common register."),
   (`Vapor.admissible,
      "Counted no-wrap test `k·a·b < 2³¹`. By `Vapor.integer_parity`, admissible int8\n  contractions are bit-identical on every substrate in every accumulation order;\n  by `Vapor.admissible_mono` certifying at the maximal extent covers all smaller ones."),
   (`Vapor.wrapS32,
      "Two's-complement reading of the low 32 bits (`Vapor.wrapS32_of_small`,\n  `Vapor.wrapS32_wrap32`)."),
   (`Vapor.padStride,
      "Minimal coprime pad for 2ᵐ banks: sufficient by `Vapor.padStride_coprime`,\n  minimal by `Vapor.padStride_minimal`, conflict-free by `Vapor.conflict_free`."),
   (`Vapor.bankOf, "Bank touched by a lane (`Vapor.bank_injective`)."),
   (`Vapor.affineOp,
      "Affine SSM composition, associative with neutral {1, 0} (`Vapor.affineOp_assoc`,\n  `Vapor.affineOp_apply`)."),
   (`Vapor.segAffine,
      "Segmented affine combine for ragged batches; a monoid by `Vapor.segAffine_monoid`."),
   (`Vapor.Binary32.units,
      "Value of a finite binary32 pattern in units of 2⁻¹⁴⁹ — the model under which\n  `Vapor.Binary32.mul_one_unique`, `add_negZero_unique`, `sub_posZero_unique` prove the\n  rewriter's rules bit-exact and `add_posZero_refuted` refutes `x + (+0) → x`."),
   (`Vapor.withinEnvelope,
      "Exact Wilkinson envelope decision `d ≤ γₙ·s + a` as `d·(2²⁴ − n) ≤ n·s + a·(2²⁴ − n)`,\n  monotone in the extent (`Vapor.withinEnvelope_mono`) and in `s`, `a`.")]

class ToEx (α : Type) where
  ex : α → String

open ToEx in
instance : ToEx Nat := ⟨toString⟩
instance : ToEx Int := ⟨toString⟩
instance : ToEx Bool := ⟨fun b => if b then "true" else "false"⟩
instance {α β : Type} [ToEx α] [ToEx β] : ToEx (α × β) := ⟨fun p => "{" ++ ToEx.ex p.1 ++ ", " ++ ToEx.ex p.2 ++ "}"⟩
instance {α : Type} [ToEx α] : ToEx (List α) := ⟨fun l => "[" ++ ", ".intercalate (l.map ToEx.ex) ++ "]"⟩

/-- Deterministic generator (64-bit LCG, Knuth MMIX constants). -/
def lcg (s : UInt64) : UInt64 := s * 6364136223846793005 + 1442695040888963407

def stream (seed n : Nat) : List Nat := Id.run do
  let mut s : UInt64 := seed.toUInt64
  let mut out := #[]
  for _ in [0:n] do
    s := lcg s
    out := out.push (s >>> 16).toNat
  return out.toList

def ints (seed n bound : Nat) : List Int :=
  (stream seed n).map fun x => (x % (2 * bound + 1) : Nat) - (bound : Int)

/-- One ExUnit test: `args` renders the argument list, `f` is the Lean function. -/
def cases {α β : Type} [ToEx β] (fn : String) (xs : List α) (args : α → List String) (f : α → β) : String :=
  let rows := xs.map fun x => "      {[" ++ ", ".intercalate (args x) ++ "], " ++ ToEx.ex (f x) ++ "}"
  s!"  test \"{fn} agrees with Lean on {xs.length} vectors\" do\n    for \{args, want} <- [\n{",\n".intercalate rows}\n    ] do\n      assert apply(Vapor.Extracted, :{fn}, args) == want, \"{fn}(#\{inspect(args)})\"\n    end\n  end\n"

/-- Half the cases are valid by construction (distinct aligned blocks, or
disjoint lifetimes on one block); the other half are random and mostly
invalid — so both verdicts of the checker are exercised. -/
def placementsFor (seed : Nat) : List (Nat × Nat × Nat × Nat) :=
  let r := stream seed 25
  let n := r[0]! % 3 + 1
  match seed % 4 with
  | 0 => (List.range n).map fun i =>
      let size := [1, 2, 4, 8][r[i + 1]! % 4]!
      (r[i + 5]! % 10, 10 + r[i + 9]! % 30, 8 * (i + 1), size)
  | 1 => (List.range n).map fun i =>
      let size := [1, 2, 4, 8][r[1]! % 4]!
      (10 * i, 10 * i + 1 + r[i + 5]! % 9, 8, size)
  | _ =>
    let n := r[0]! % 6 + 1
    (List.range n).map fun i =>
      let a := r[4 * i + 1]! % 40
      let len := r[4 * i + 2]! % 20 + (if r[4 * i + 3]! % 7 == 0 then 0 else 1)
      let size := [1, 2, 4, 8][r[4 * i + 4]! % 4]!
      let reg := (r[4 * i + 4]! / 4) % 33
      (a, a + len, reg, size)

def conformance (digest : String) : String :=
  let wrapIn : List Int :=
    [0, 1, -1, 2147483647, 2147483648, -2147483648, -2147483649, 4294967296, 4294967301,
     -4294967296, 12884901895, -12884901895] ++ ints 11 40 1099511627776
  let admIn : List (Nat × Nat × Nat) :=
    (([1, 16, 256, 4096, 131071, 131072, 131073].map fun k =>
      [(k, 127, 127), (k, 128, 128), (k, 127, 128)]).flatten) ++
    ((stream 5 30).map fun x => (x % 200000, x % 129, (x / 7) % 129))
  let padIn : List (Nat × Nat) := ((List.range 70).map fun s => (List.range 7).map fun m => (s, m)).flatten
  let bankIn : List (Nat × Nat × Nat) := (stream 9 40).map fun x => (x % 97, (x / 97) % 64, 32)
  let aff := ints 21 80 1000
  let affIn : List ((Int × Int) × (Int × Int)) :=
    (List.range 20).map fun i => ((aff[4 * i]!, aff[4 * i + 1]!), (aff[4 * i + 2]!, aff[4 * i + 3]!))
  let flags := stream 23 40
  let segIn : List (((Int × Int) × Bool) × ((Int × Int) × Bool)) :=
    (List.range 20).map fun i =>
      (((aff[4 * i]!, aff[4 * i + 1]!), flags[2 * i]! % 2 == 0), ((aff[4 * i + 2]!, aff[4 * i + 3]!), flags[2 * i + 1]! % 2 == 0))
  let envIn : List (Nat × Nat × Nat × Nat) :=
    ((stream 31 40).map fun x => (x % 5000, (x / 5000) % 100000000, (x / 11) % 3, (x / 3) % 20000000)) ++
    [(0, 0, 0, 0), (1, 16777215, 0, 1), (10, 100, 0, 16777216), (5, 0, 0, 3), (5, 0, 5, 3), (6, 0, 5, 3)]
  let f32In : List Nat :=
    [0, 2147483648, 1, 2147483649, 8388607, 8388608, 2139095039, 4286578687, 1065353216, 3212836864, 1, 16777216] ++
    ((stream 41 40).map fun x => let b := x % 4294967296; if (b / 8388608) % 256 == 255 then b - 8388608 else b)
  let allocIn : List (Nat × List Nat × List (Nat × Nat × Nat × Nat)) :=
    (List.range 60).map fun i => (32, if i % 3 == 0 then [0] else [15, 31], placementsFor (100 + i))
  "# GENERATED by `lake exe vapor-extract` (proofs/Main.lean) — do not edit.\n" ++
  s!"# lean-source-fnv1a64: {digest}\n" ++
  "defmodule Vapor.ExtractedConformanceTest do\n" ++
  "  @moduledoc \"Values computed by Lean for the extracted definitions; the Elixir\\n  extraction must reproduce every one (validates the printer and the prelude).\"\n" ++
  "  use ExUnit.Case, async: true\n\n" ++
  s!"  test \"extraction matches the Lean sources it was generated from\" do\n    assert Vapor.Extracted.source_digest() == \"{digest}\"\n  end\n\n" ++
  let e := fun {γ : Type} [ToEx γ] (x : γ) => ToEx.ex x
  cases "wrap_s32" wrapIn (fun x => [e x]) Vapor.wrapS32 ++ "\n" ++
  cases "admissible" admIn (fun (k, a, b) => [e k, e a, e b]) (fun (k, a, b) => Vapor.admissible k a b) ++ "\n" ++
  cases "pad_stride" padIn (fun (s, m) => [e s, e m]) (fun (s, m) => Vapor.padStride s m) ++ "\n" ++
  cases "bank_of" bankIn (fun (s, l, b) => [e s, e l, e b]) (fun (s, l, b) => Vapor.bankOf s l b) ++ "\n" ++
  cases "affine_op" affIn (fun (x, y) => [e x, e y]) (fun (x, y) => Vapor.affineOp x y) ++ "\n" ++
  cases "seg_affine" segIn (fun (x, y) => [e x, e y]) (fun (x, y) => Vapor.segAffine x y) ++ "\n" ++
  cases "within_envelope" envIn (fun (d, s, a, n) => [e d, e s, e a, e n]) (fun (d, s, a, n) => Vapor.withinEnvelope d s a n) ++ "\n" ++
  cases "units" f32In (fun b => [e b]) Vapor.Binary32.units ++ "\n" ++
  cases "check_alloc" allocIn (fun (c, r, ps) => [e c, e r, e ps]) (fun (c, r, ps) => Vapor.RegAlloc.checkAlloc c r ps) ++
  "end\n"

def main (args : List String) : IO UInt32 := do
  let (outEx, outTest) := match args with
    | [a, b] => (a, b)
    | _ => ("../lib/vapor/extracted.ex", "../test/vapor/extracted_conformance_test.exs")
  let mut h : UInt64 := 0xcbf29ce484222325
  for f in sources do
    h := fnv1a64 (← IO.FS.readBinFile f) h
  let digest := hex16 h
  initSearchPath (← findSysroot)
  let env ← importModules #[{ module := `Vapor }] {} (trustLevel := 1024)
  let body ← VaporExtract.run env roots
  let header :=
    "# GENERATED by `lake exe vapor-extract` from proofs/Vapor/*.lean — do not edit.\n" ++
    s!"# lean-source-fnv1a64: {digest}\n" ++
    "defmodule Vapor.Extracted do\n" ++
    "  @moduledoc \"\"\"\n" ++
    "  Definitions extracted from the Lean 4 development in `proofs/` (see\n" ++
    "  `proofs/Extract.lean`). Each function is the elaborated Lean term that the\n" ++
    "  cited theorems are about, printed as Elixir; the conformance suite\n" ++
    "  `test/vapor/extracted_conformance_test.exs` holds values computed by Lean.\n" ++
    "  Nested pairs follow Lean's `α × β × γ = α × (β × γ)`: `{a, {b, {c, d}}}`.\n" ++
    "  \"\"\"\n\n" ++
    s!"  @doc \"FNV-1a/64 of the Lean sources this module was extracted from.\"\n  def source_digest, do: \"{digest}\"\n\n"
  IO.FS.writeFile outEx (header ++ body ++ "end\n")
  IO.FS.writeFile outTest (conformance digest)
  IO.println s!"extracted {roots.length} roots → {outEx} (sources {digest})"
  return 0
