# Rebis — two circuits, one function?

> Since 0.15.0. Code: `lib/vapor/rebis.ex`, `lib/vapor/rebis/{gen,ideal,field,gcm,stabilizer}.ex`.
> Tests: `test/vapor/rebis_test.exs`. Console: *Opus → Rebis*. Terminal: `vapor rebis equiv|anf|identity|stabilizer|aiger`.
> MCP: `rebis_check`.

The *rebis* is the alchemists' "double thing": two natures shown as one.

## The pain

A *netlist* after synthesis, a third-party IP block, a chip back from the foundry — is it the
specification? Simulation answers for the patterns it tried; a hardware Trojan horse
whose trigger is a 64-bit coincidence survives every testbench ever run.
Equivalence is a theorem or it is nothing.

## The algebra, stated exactly — and the correction to the proposal

Over GF(2), XOR is `+`, AND is `·`, NOT is `+1`; every boolean function of `n` inputs has **one**
multilinear polynomial — the algebraic normal form (Zhegalkin) — in `GF(2)[x₁…xₙ]/⟨xᵢ² − xᵢ⟩`.
The round's proposal ("prove `P_A − P_B ≡ 0` with Gröbner bases") is right and is the
wrong tool: the ANF **already is** the normal form modulo that ideal, computed by the Möbius
transform in `O(n·2ⁿ)` word operations — no Buchberger — and beyond ~20 inputs no
normal form is cheap (equivalence is coNP-complete). So, two procedures, each with
checkable output:

- **`n ≤ 16`**: the truth table of each output as **one** BEAM integer of `2ⁿ` bits (all
  patterns at once), compared; the ANF by Möbius in `n` shift-and-XOR steps.
- **beyond**: 4,096 random patterns at once, then a **miter** — `OR(outᵃᵢ ⊕ outᵇᵢ)` — via
  Tseitin into CNF for `Vapor.Logic.SAT`. UNSAT comes with a DRUP proof that
  `Vapor.Logic.DRUP` (code that shares nothing with the solver) checks before the
  answer is "equivalent"; SAT comes with a model, re-simulated on both circuits before being
  called a counterexample.

A counterexample is **shrunk** (the fewest inputs at 1, greedily), so a
Trojan's trigger reads as the trigger.

## Word arithmetic: Gröbner where Gröbner is the right tool

For word identities (64 wires are the product of two 32-bit words), GF(2) is the wrong
ring. `Vapor.Rebis.Ideal` works **over ℤ** with `x² = x`: each gate is a polynomial
(`¬a = 1 − a`, `a∧b = ab`, `a⊕b = a + b − 2ab`, …) and, in a lexicographic order that puts each
gate above its inputs, the gate polynomials **already are** a Gröbner basis of the circuit's
ideal (the leading terms are distinct variables). Reducing the specification by them is
substituting gates from back to front: the remainder is `0` iff the identity holds for every input
(Lv, Kalla & Enescu 2013; Ritirc, Biere & Kauers 2017). A non-zero remainder gives a point where the
identity fails — re-evaluated on the circuit before being reported. Specifications as text:
`m[16] = a[8] * b[8]`, `s[8] + 2^8*cout = a[8] + b[8]`.

The two procedures are **complementary**, measured:

| | CDCL + DRUP | algebra over ℤ |
|---|---|---|
| multiplier, commutativity | 2,963 conflicts at 5 bits; does not finish in minutes at 6 | 16 bits: 2,748 substitutions, peak of 522 terms; **32 bits: 1.4 s, peak of 2,058** |
| 64-bit *ripple* adder | — | linear: peak < 1,000 terms |
| *ripple* × Kogge–Stone | 16 bits: DRUP proof checked; 64 bits: 88 s (7,541 conflicts, 5,380 lemmas checked) | Kogge–Stone 32 bits: goes past 50,000 terms → `:unknown`, never a guess |

## Binary fields, AES and GCM

`Vapor.Rebis.Field`: GF(2ⁿ) with the **carry-less** product (what `PCLMULQDQ`, `PMULL` and `vclmul`
compute), reduction, inverse by extended Euclid, Rabin's irreducibility test. The AES S-box
is **derived** (the inverse in GF(2⁸) followed by the affine map), not tabulated — and each output bit
has algebraic degree 7, recomputed by Möbius. GHASH is done two ways (NIST's
algorithm and the reflected product) that agree. `Vapor.Rebis.GCM`: the whole of AES-GCM on top of that, equal
to OpenSSL (`:crypto`) on every tested combination of message length, AAD and IV; a
tampered tag is rejected. It is a **checker**, not a library for encrypting
production data (no constant time).

## Stabilizers

`Vapor.Rebis.Stabilizer`: Clifford circuits on thousands of qubits, exact, on a
classical machine (Gottesman–Knill, Aaronson–Gottesman's CHP tableau). Each row is two BEAM integers
(`x`, `z`) and a sign; the phase of a product of rows is computed for all qubits at once
(the positions that gain `+i` and `−i` are two masks, the phase is the difference of the *popcounts*
mod 4). Checked against a dense state-vector simulator on 60 random circuits of up to
5 qubits; a 400-qubit GHZ state gives 1 random measurement and 399 determined ones. `T` leaves the
formalism and is rejected by name.

## Input

A small *netlist* language (`input`, `output`, `w = a & ~b ^ c`, `mux(s, a, b)`,
`maj(a, b, c)`, with *hash-consing*) or ASCII AIGER (`aag`, the format of the hardware
verification competitions), read in Kahn topological order (out-of-order gates accepted, cycles
rejected). Names never become atoms; sizes have a ceiling.

## Found along the way

- A miter with **repeated literals** in a clause hung the solver; clauses are
  normalised (repetitions removed, tautologies discarded) before SAT.
- AIGER with out-of-order gates was rejected; real files do not guarantee the order.
- The phase of CHP's destabilizer rows can be odd (only the stabilizer rows are
  Hermitian); row addition accepts that for them.

## What it is not

It is not synthesis or *place-and-route*; sequential circuits (with registers) come in only as
their combinational part (the one-step *miter*). An emulator of old chips as networks over
GF(2⁸) was not built ([DIRECTIVE §18](DIRECTIVE.md)).
