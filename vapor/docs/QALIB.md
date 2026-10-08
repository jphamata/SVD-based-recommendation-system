# Qālib — from a Boolean term to standard cells, every step proved

> `Vapor.Qalib` (`lib/vapor/qalib.ex`), `vapor qalib map|check`, the MCP tool `qalib_check`. Tests:
> `qalib_test.exs`; §5m of the quality suite. Scrutiny: [DIRECTIVE §20](DIRECTIVE.md).

*Al-qālib* (القالب) is the mould, where the metal takes its shape.

## What was asked, and what was built

The request asked for `Vapor.Silicon`, a pure-Elixir hardware compiler that emits GDSII masks for
SkyWater 130 nm and so "ends the dependence on proprietary EDA". Two premises fail:

- **The sky130 flow is not proprietary.** Yosys (synthesis), OpenROAD (placement and routing),
  Magic and KLayout (layout, DRC, LVS) are open source, and Tiny Tapeout and Efabless use them.
- **Layout is the worst place for a home-grown tool.** A GDS that has not passed the foundry's
  design-rule check and layout-versus-schematic is not a chip. Re-implementing placement, routing
  and DRC would add the least trustworthy code in the flow, not remove it.

What the open flow lacks, and vapor has, is a **checker that does not trust the flow's tools**. So
Qālib reads what the flow writes, maps vapor's circuits onto the same cells, and proves equivalence
at every step. That is *translation validation* of the whole flow.

```
spec (Rebis netlist, AIGER, an Almizan n-q-l claim lowered to AIGER)
  │  vapor qalib map ──────────────► sky130 Verilog (proved equal before it is printed)
  │                                        │ Yosys / ABC
  │  vapor qalib check spec yosys.v ◄──────┘  (proved, or a counterexample)
  │                                        │ OpenROAD → GDS → Magic extracts a netlist
  └─ vapor qalib check spec extracted.v ◄──┘  (the layout's logic, proved against the spec)
```

A trojan inserted by any tool, or a synthesis bug, shows up as a counterexample however rare its
trigger.

## Reading

- **Structural Verilog**: `module` with plain or ANSI ports, `input`/`output`/`wire` with ranges
  (`[3:0]` becomes `a[3] … a[0]`), escaped identifiers, gate primitives (`and or xor nand nor xnor
  not buf`, any arity), `assign` with `~ ! & | ^ ?:`, parentheses, bit selects and concatenations,
  one-bit constants, and instances of the `sky130_fd_sc_hd` cells with named pins.
- **BLIF**: `.model`, `.inputs`, `.outputs`, `.names` (on-set or off-set covers, `-` for don't
  care, constants) and `.gate` with the same cells.

The cell table is explicit: inv, buf, clkinv, clkbuf, nand2–4, nor2–4, and2–4, or2–4, xor2, xor3,
xnor2, xnor3, nand2b, nor2b, and2b, or2b, mux2, mux2i, maj3, a21o, a21oi, o21a, o21ai, a22oi,
o22ai, a211oi, o211ai, a31oi, o31ai, conb. Any drive strength (`_1`, `_2`, `_4`…) is accepted.
**An unknown cell is refused, not guessed**, and so are latches and flip-flops, combinational
loops, nets driven twice, outputs never driven, and constants wider than one bit. Each cell's
function is tested against its Boolean formula.

## Mapping and writing

`Qalib.to_verilog/2` writes a circuit as structural Verilog over the same cells, with
`style: :cells` (INV, AND2, OR2, XOR2, MUX2, CONB, one per gate) or `style: :nand` (NAND2 and
INV only). `vapor qalib map` reads its own output back and proves it equal to the source before
printing it.

## Proving

`Qalib.certify/3` is `Vapor.Rebis.equivalent/3`. Up to 16 inputs it compares truth tables. Beyond
that, 4,096 random patterns run first, then a SAT miter whose UNSAT answer carries a DRUP proof
checked by code that shares nothing with the solver. A difference comes with a counterexample,
shrunk and re-simulated on both circuits.

Measured: a 10-bit ripple-carry adder (21 inputs) written as Verilog gate primitives is proved
equal to its specification by SAT with a checked DRUP proof. The same adder with a trojan, sum bit
0 flipped when `a = 0x2B5` and `b = 0x14A`, is told apart with exactly that input. 4,096 random
patterns do not find it (the control).

## What is not done

Sequential logic (equivalence of state machines needs induction; [TODO](TODO.md)), timing,
power and area (the liberty files hold them; cell counts are reported, µm² are not), placement,
routing and GDSII.
