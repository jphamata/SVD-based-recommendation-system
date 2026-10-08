# Engineering — the engineer's text, a certificate independent of the solver

> Request (0.12), translated: "solve real pains of electrical, mechanical,
> chemical and civil engineering". Scrutiny: [DIRECTIVE.md §15](DIRECTIVE.md).

The pain common to the four engineering fields is not the lack of a solver — it is **trusting**
the number it returns. Each tool here reads the text the
engineer would already write (a netlist, a bus list, nodes and members,
a mesh, pipes, reactions) and returns, alongside the answer, an
**independently computed certificate**: the physical law
re-evaluated on the solution, not the method's internal residual.

`Vapor.Engineering.*` · console *Solve → Engineering* · MCP `engineering_run`.

## 1. Electrical

### Circuits (`Vapor.Engineering.Circuit`) — a subset of SPICE

```
* a diode biased by a divider
V1 in 0 DC 5
R1 in a 1k
R2 a 0 2k
R3 a d 1k
D1 d 0 IS=1e-14
.op
```

Elements R C L V I D E G F H O (ideal op-amp); SPICE values (`1k`,
`2.2u`, `10meg`); sources `DC`, `AC`, `SIN(…)`, `PULSE(…)`, `PWL(…)`;
analyses `.op` (Newton with *pnjlim* and *source stepping*), `.dc`, `.ac`
(complex phasors), `.tran` (trapezoidal; `.method be`). MNA.

**Certificate**: Kirchhoff's current law re-evaluated at each node
from the voltages and the models (not from the linear system), the power
balance (Σ supplied = Σ dissipated), op-amp outputs within the rail.
**Warnings** name nodes with no DC path to ground; a loop of voltage
sources is refused as singular.

Verified: the Shockley equation satisfied at the solved point; RC by
trapezoidal rule **order 1.97**, implicit Euler (control) **0.99**; series RLC —
peak current at 1/2π√LC, peak capacitor voltage at
ω₀√(1 − 1/2Q²) with the exact value Q/√(1 − 1/4Q²); inverting amplifier
−R2/R1 to 10⁻¹².

### Transistors (0.13)

```
* a common-source stage and a CMOS inverter
VDD vdd 0 DC 5
VG g 0 DC 1.5 AC 1
RD vdd d 10k
M1 d g 0 0 NMOS KP=50u VTO=0.7 LAMBDA=0.02 W=10u L=1u
Q1 c b e NPN IS=1e-15 BF=150 BR=2
```

`M d g s [b] NMOS|PMOS` is the **level-1 MOSFET** (Shichman–Hodges: cutoff,
triode and saturation with channel-length modulation λ; a symmetric device — the
lower potential between drain and source acts as the source); `Q c b e NPN|PNP` is the
**Ebers–Moll bipolar** in transport form (`IS`, `BF`, `BR`), the
junctions limited like the diode's. Each transistor enters the MNA as a
three-terminal element whose current depends on the three voltages; the
Newton conductances come from central differences with h = 10⁻⁷ V (error
~(h/V_T)² ≈ 10⁻¹¹), and a GMIN of 10⁻¹² S crosses the channel or the junction,
as in SPICE, so that a cut-off device does not leave a node floating. The
bipolar is **two branches** (collector → emitter and base → emitter): the
emitter current is the sum, and the Kirchhoff certificate — which re-evaluates
each current by the device's own equation — remains valid without
changing a line.

Verified **against ngspice 42** (`engineering_test.exs`, when the
binary is present): the common-source stage (3.2945736 V at the drain), the
common-emitter stage with divider and emitter resistor (V_C, V_B, V_E equal to the
6th decimal place — with `.temp`/`TNOM` equal to the 300.00 K used here; with the default TNOM
of 27 °C ngspice rescales IS and the 2.4 mV difference at the collector is the
physics of the model, not an error) and the CMOS inverter (3.1436422 V); the
small-signal gain of the common-source stage in `.ac` equal to ngspice's to 10⁻⁶. By hand:
the square law Id = β/2·(V_GS − V_t)²·(1 + λV_DS) closes to 3·10⁻¹² A — the
GMIN current.

### Power flow (`Vapor.Engineering.Power`)

```
base 100
bus 1 slack V=1.06
bus 2 pq P=20 Q=20
line 1 2 r=0.02 x=0.06 b=0.06
shunt 4 b=0.15
```

Newton–Raphson in polar coordinates (slack, PV, PQ buses, taps,
shunts); Gauss–Seidel as the control. **Certificate**: the maximum mismatch
at each bus recomputed from the admittances, and the active power
balance (generation = load + losses).

Verified on the five-bus system of **Stagg & El-Abiad** (1968): the
published voltages and angles to 6·10⁻⁴ pu and 6·10⁻³ °, the slack bus
power 129.59 MW, mismatch < 10⁻¹⁰ in 4 iterations; Gauss–Seidel reaches
the same voltages in 119.

## 2. Civil and mechanical

### 2-D frames and trusses (`Vapor.Engineering.Structure`)

```
node 1 0 0
node 2 3[m] 0
support 1 fixed            # fixed | pinned | roller-x | roller-y | combinations x y rz
beam 1 2 E=200[GPa] A=0.01 I=1e-4 rho=7850 n=8
truss 2 3 E=200e9 A=1e-3
load 2 fy=-10[kN]          # fx fy mz
udl 1 2 w=-5[kN/m]
modes 3
```

Direct stiffness with member subdivision (`n=`), distributed loads as
equivalent nodal forces, **consistent mass** and the generalised eigenvalue
problem for the modes, rotations of truss-only nodes
restrained automatically (and reported), **mechanism refused**,
**reverse Cuthill–McKee** numbering and banded Cholesky.
**Certificate**: global equilibrium — Σ loads + Σ reactions = 0 in force and
moment, relative to the largest load.

Verified: cantilever PL³/3EI and fixed-end moment PL **exact**; simply
supported beam 5wL⁴/384EI and wL²/8; truss by the method of joints; cantilever
frequencies against Euler–Bernoulli (1.8751² and 4.6941² √(EI/ρAL⁴)) to 10⁻⁶ with
10 elements — a single element (control) is off by 4.8·10⁻³.

### Plane stress (`Vapor.Engineering.FEM`)

```
plate x=0..10 y=-0.5..0.5 nx=20 ny=4
material E=1000 nu=0.3 t=1        # strain for plane strain
element qm6                       # q4 | qm6 (incompatible modes, default)
fix x=0
traction x=10 ty=-1
```

Q4 and **QM6** (Wilson–Taylor, condensed incompatible modes), nodal
stresses and von Mises, bandwidth reduced by RCM. **Certificate**: residual
K·u − f and equilibrium of the reactions.

Verified: **patch test** passed by Q4 and QM6 on distorted elements
(displacement 10⁻¹⁴, stress 10⁻¹²); on the slender plate in bending QM6 is within
**0.9%** of the Timoshenko beam, Q4 (control) **locks** and is off by 29%.

### Pipe networks (`Vapor.Engineering.Pipes`)

```
reservoir R head=60[m]
junction A elev=10[m] demand=15[L/s]
pipe 1 R A L=800[m] D=250[mm] eps=0.1[mm] K=2
```

Global Newton on flows and heads (Todini–Pilati gradient method),
Colebrook–White solved by Newton (laminar 64/Re), minor losses.
**Certificate**: continuity at each node, energy around each **loop**
(found by a spanning tree), paths between reservoirs.

Verified: a pipe between two reservoirs equal to **SciPy**'s
`brentq` to 10⁻¹⁰; a network with two loops closing continuity to 10⁻¹⁸ and
energy to 10⁻¹².

## 3. Chemical

### Kinetics and reactors (`Vapor.Engineering.Process.reactions/1`)

```
A + B -> C ; k = 2
C <-> D ; kf = 1, kb = 0.2
2 A -> E ; k = 0.1
A0 = 1; B0 = 0.8
t = 0 .. 20
reactor cstr tau=5     # batch (default) or CSTR with feed
```

Mass action → ODEs (with the workbench's automatic switch to
Rosenbrock). **Invariants**: the left null space of the stoichiometric
matrix, in exact rational arithmetic — the conserved
quantities come **from the stoichiometry alone** and are checked along the
solution.

Verified: A → B → C against Bateman's closed form to 10⁻⁸; the network
above has exactly 2 invariants with drift < 10⁻⁹ ([A] alone, the
control, changes by 0.93); CSTR at steady state C₀/(1 + kτ).

### Flash and distillation (`Vapor.Engineering.Process.flash/1`, `distill/1`)

Isothermal flash by **Rachford–Rice** with Antoine (bubble and
dew pressures, a single phase reported when that is the case); certificate: mass
balances, Σx = Σy = 1, Rachford–Rice residual. Binary column by
**McCabe–Thiele** (stages, feed tray), **Fenske**,
**Underwood** and **Gilliland**; control: with reflux 10⁴ the
McCabe–Thiele staircase gives exactly ⌈Fenske⌉, and a reflux below the minimum is
refused.

## 4. Honest limits

- Circuits: transistors are Ebers–Moll and MOSFET level 1 since 0.13 (§1),
  without device capacitances in transients; no noise, no `.subckt`. The diode is Shockley with zero series resistance.
- Power flow: balanced, single-phase equivalent; no reactive
  limits on PV buses, no short-circuit.
- Structures: 2-D, linear, small displacements; no buckling, no
  geometric or material non-linearity; FEM only on quadrilaterals and
  structured rectangles.
- Pipes: steady state; no pumps or controlled valves.
- Processes: ideality (Raoult), binary distillation at constant α.
