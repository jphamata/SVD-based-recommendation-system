# Workbench — any equation, with units and checks

> Request (0.12), translated: "solve real pains of electrical, mechanical,
> chemical and civil engineering, as well as physics, chemistry, mathematics, CS and biology,
> maximum HPC […] for arbitrary problems and not just predefined
> categories". Scrutiny: [DIRECTIVE.md §15](DIRECTIVE.md).

The pain the workbench attacks is that of someone who needs a reliable number and
today has to choose between a spreadsheet (fast, no units, no
checking), a script (flexible, no guarantee at all) and a
heavy package (licence, installation, one form per problem type). Here the
problem is **written as on paper**, the workbench recognizes the type, checks
the **units before running** and returns, together with each answer, the
numbers that let you judge it.

`Vapor.Solve.run/1` · console *Solve → Workbench* · MCP `workbench_solve`.

## 1. The language

An expression is ordinary text with implicit multiplication (`2x`, `3(x+1)`),
functions (`sin cos tan atan sinh cosh tanh exp ln sqrt cbrt erf asinh abs
min max …`) and **quantities with units** in square brackets:

```
F = 3[kN]
L = 2.5[m]
E = 200[GPa]
I = 8.33e-6[m^4]
M = F*L in [kN*m]                 → 7.5 kN·m
d = F*L^3/(3*E*I) in [mm]         → 9.0036 mm
k = 3*E*I/L^3 in [kN/mm]
f = sqrt(k/120[kg])/(2*pi) in [Hz]
p = 101325[Pa] in [psi]           → 14.6959 psi
bad = F + L                       → refused: "adding force (N) to length (m)"
```

The seven SI base dimensions are tracked through every operation; the
named derived units (N, Pa, J, W, V, C, F, Ω, S, Wb, T, H, Hz) are
recognized on output; prefixes from y to Y. A transcendental function of a
dimensioned quantity is refused (`sin(3[m])`). Each tree is **compiled
to a BEAM module** (`Vapor.Expr.compile/2`, cached by the tree's sha-256),
differentiated symbolically (`Vapor.Expr.diff/2`) and printed as
text and LaTeX.

### Affine scales: °C and °F (0.13)

Up to 0.12, `degC` and `degF` were only **intervals** (a difference of one
degree) and the scales with an offset were refused — honest, but
awkward: nobody writes a room temperature in kelvin. Now `°C`,
`°F`, `celsius` and `fahrenheit` are **readings** on a thermometer, accepted
exactly where they have a single meaning:

```
T = 98.6[°F]
x = T in [°C]                          → 37 °C
p = 1[mol]*8.314[J/(mol*K)]*20[°C]/1[L]
q = p in [kPa]                         → 2437.25 kPa   (20 °C = 293.15 K)
d = 30[°C] - 20[°C]                    → 10 K          (a difference)
c = 4186[J/(kg*°C)]                    → refused: "°C is a reading… inside a compound unit write degC"
```

After a number, the reading becomes absolute kelvin (25[°C] = 298.15 K) and
all the physics works; `in [°C]` subtracts the offset back; inside
a compound unit or after an expression the scale is refused with
the reason — the classic error is treating a reading as a difference.

## 2. What the workbench solves

| written | kind | method | evidence returned |
|---|---|---|---|
| `x' = …`, `x(0) = …`, `t = 0 .. 10[s]` | ODEs | Dormand–Prince 5(4) with Hairer's dense output; automatic switch to **Rosenbrock ode23s** (symbolic Jacobian) when stiffness exhausts 20,000 steps; fixed-step RK4 as an option | steps, rejected steps, evaluations, the switch and why, units of each state |
| `stop when y < 0` | events | bisection over the dense output | the instant of the event (within 10⁻¹² of 2v/g for the launch) |
| `E := …` | derived outputs | evaluated on the output grid | — |
| `u_t = …`, `u_x(1, t) = 0` | 1-D parabolic (variable coefficients, advection, nonlinear reaction, Neumann) | Crank–Nicolson + Adams–Bashforth 2 on the reaction; ghost nodes | the **observed order** with `verify` |
| `u_tt = c^2*u_xx` | hyperbolic | leapfrog | the CFL number; unstable step **refused** |
| `poisson -(u_xx + u_yy) [+ c*u] = f` | 2-D elliptic | five-point stencil + conjugate gradients | residual, iterations |
| `exact u = …` + `verify` | verification | **manufactured solution**: the source term is derived symbolically, three grids | error per grid and the observed order against the expected one |
| `unknowns x = 2, y = 0.5` + equations + `search = [a, b]` | nonlinear systems | Newton with Halton multistart | **all** roots found in the box, the residual of each |
| `fit y = a*exp(-b*x) + c` + `data` | nonlinear least squares | Levenberg–Marquardt | standard error of each parameter, R², RMSE, AIC, residuals |
| `minimize …`, `subject to …` | constrained optimization | augmented Lagrangian + BFGS; **box bounds by projection** | KKT conditions (stationarity, feasibility, complementarity), multipliers, verdict |
| `k ~ normal(4, 0.2)`, `members = 4096` | uncertainty (ensemble) | unrolled RK4, compiled for the **native worker** | percentile bands, bit-for-bit parity with the oracle, agreement with binary64, speed-up |

## 3. How it is verified

`test/vapor/workbench_test.exs` (20 tests) and §5h of the quality report:

- **Oscillator** with units: maximum error < 10⁻⁸ against cos 2t, energy
  conserved to 10⁻⁷; RK4 with step 0.1 (control) is off by 2.3·10⁻⁴.
- **Robertson** (stiff) at t = 40: a = 0.7158271, b = 9.185535·10⁻⁶ —
  the values of Hairer & Wanner — and the switch to Rosenbrock reported.
- **SciPy** (`solve_ivp`, DOP853 at 10⁻¹²) agrees to 10⁻⁶ on the
  Lotka–Volterra cycle.
- **Heat** by manufactured solution: Crank–Nicolson order 2.00; implicit
  Euler (control) 1.05. With a variable coefficient, advection, nonlinear
  reaction and Neumann: still > 1.85. **2-D Poisson**: 1.99.
- **Wave** with a step above CFL: refused by name.
- **Roots**: the four intersections of the circle with the hyperbola, residual
  < 10⁻¹²; **fit** recovers planted parameters within 4σ.
- **Optimization**: Rosenbrock to 10⁻⁸; the can of least area with volume 1 and
  r, h ≥ 0.01 gives r = (1/2π)^⅓ to 10⁻⁶ with KKT; **without the bounds** the same
  formulation descends to r < 0 — and the result is **"diverged"**, never an
  answer (this was a real defect found in this round: 0.12.0-dev
  returned −6·10⁶² as the optimum).
- **Native ensemble** (4096 members × 1000 steps): 236 ms on the x86-64 worker,
  **53×** the f64 estimate on the BEAM, bit-for-bit parity with the oracle,
  median relative deviation 6·10⁻⁷ against binary64.

## 4. Honest limits

- PDEs: 1-D in time and 2-D Poisson on rectangles; no unstructured
  meshes (the planar FEM of *Engineering* covers 2-D elasticity),
  no 3-D, no coupled systems of PDEs.
- The native ensemble accepts the worker's algebra (`+ − × ÷`, `sqrt`, `exp`,
  `log`, `tanh`, selection); `sin` is refused by name (approximating it
  silently would break parity with the oracle).
- Optimization is local (BFGS); multistart exists for systems, not
  for global minima.
- Units: affine scales (°C, °F) are admitted as readings since 0.13 (see "Affine scales" above); they are not multiplicative quantities and never enter a product.
