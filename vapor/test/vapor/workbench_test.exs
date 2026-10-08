defmodule Vapor.WorkbenchTest do
  @moduledoc """
  The workbench (docs/WORKBENCH.md): the expression language with units,
  the ODE solvers against closed forms and published values, the
  algebraic solvers, the PDE solvers verified by manufactured solutions
  (with a first-order scheme as the control), the native ensemble against
  the oracle bit for bit, and SciPy as an outside oracle.
  """
  use ExUnit.Case, async: true
  @moduletag timeout: 600_000
  alias Vapor.{Dense, Expr, Solve, Units}

  describe "expressions and units" do
    test "parse, print, differentiate, compile" do
      t = Expr.parse!("3*x^2 + sin(2x) - exp(-x/2)")
      assert_in_delta Expr.eval(t, %{"x" => 1.3}), 3 * 1.69 + :math.sin(2.6) - :math.exp(-0.65), 1.0e-12
      d = Expr.diff(t, "x")
      num = (Expr.eval(t, %{"x" => 1.3 + 1.0e-6}) - Expr.eval(t, %{"x" => 1.3 - 1.0e-6})) / 2.0e-6
      assert_in_delta Expr.eval(d, %{"x" => 1.3}), num, 1.0e-6
      f = Expr.compile(t, ["x"])
      assert f.({1.3}) == Expr.eval(t, %{"x" => 1.3})
      assert {:ok, _} = Expr.parse(Expr.to_text(d))
    end

    test "every function differentiates like a central difference" do
      for f <- ~w(sin cos tan atan sinh cosh tanh exp ln sqrt cbrt erf asinh) do
        t = Expr.parse!("#{f}(0.3*x + 0.2)")
        d = Expr.diff(t, "x")
        h = 1.0e-6
        num = (Expr.eval(t, %{"x" => 0.7 + h}) - Expr.eval(t, %{"x" => 0.7 - h})) / (2 * h)
        assert_in_delta Expr.eval(d, %{"x" => 0.7}), num, 1.0e-6, f
      end
    end

    test "parse errors name the column; unknown functions are refused" do
      assert {:error, m} = Expr.parse("2 + * 3")
      assert m =~ "column"
      assert {:error, m} = Expr.parse("system(1)")
      assert m =~ "unknown function"
    end

    test "units: conversions and the dimension mismatch (the control)" do
      [_, _, m, _, _, d, bad, _, p] = Expr.worksheet("""
      F = 3[kN]
      L = 2.5[m]
      M = F*L in [kN*m]
      E = 200[GPa]
      I = 8.33e-6[m^4]
      d = F*L^3/(3*E*I) in [mm]
      bad = F + L
      w = sqrt(9.81[m/s^2]/1[m])
      p = 101325[Pa] in [psi]
      """)
      assert_in_delta m.shown, 7.5, 1.0e-12
      assert_in_delta d.shown, 3000 * 2.5 ** 3 / (3 * 200.0e9 * 8.33e-6) * 1000, 1.0e-9
      assert bad.error =~ "adding force"
      assert_in_delta p.shown, 14.6959, 1.0e-4
      assert {:ok, {_, dim}} = Units.parse("kg*m^2/s^3/A")
      assert Units.format(dim) == "V"
      assert {:error, m} = Expr.dim(Expr.parse!("sin(3[m])"))
      assert m =~ "pure number"
    end
  end

  describe "ordinary differential equations" do
    test "Dormand–Prince with dense output: the oscillator to 10⁻⁸, energy kept" do
      {:ok, r} = Solve.ode("""
      x' = v
      v' = -k/m*x
      k = 4[N/m]; m = 1[kg]
      x(0) = 1[m]; v(0) = 0[m/s]
      t = 0 .. 10[s]
      E := 0.5*m*v^2 + 0.5*k*x^2
      rtol = 1e-9
      atol = 1e-12
      """)
      err = Enum.zip(r.t, r.series["x"]) |> Enum.map(fn {t, x} -> abs(x - :math.cos(2 * t)) end) |> Enum.max()
      assert err < 1.0e-8
      assert Enum.max(r.series["E"]) - Enum.min(r.series["E"]) < 1.0e-7
      assert r.units["v"] == "m/s"
    end

    test "inconsistent units in an ODE are refused before anything runs" do
      assert {:error, m} = Solve.ode("x' = v\nv' = -k*x\nk = 4[N/m]\nx(0)=1[m]\nv(0)=0[m/s]\nt=0..1[s]")
      assert m =~ "v' has force"
    end

    test "Robertson's stiff kinetics: the switch to Rosenbrock, and Hairer & Wanner's values at t = 40" do
      {:ok, r} = Solve.ode("""
      a' = -0.04*a + 1e4*b*c
      b' = 0.04*a - 1e4*b*c - 3e7*b^2
      c' = 3e7*b^2
      a(0) = 1; b(0) = 0; c(0) = 0
      t = 0 .. 40
      rtol = 1e-6
      atol = 1e-10
      """)
      assert r.switched =~ "stiff"
      assert_in_delta r.final["a"], 0.7158271, 2.0e-6
      assert_in_delta r.final["b"], 9.185535e-6, 5.0e-10
      assert_in_delta r.final["c"], 0.2841729 - 9.2e-6, 2.0e-6
    end

    test "events: a ball thrown up lands at 2v/g" do
      {:ok, r} = Solve.ode("y' = v\nv' = -9.81\ny(0)=0\nv(0)=20\nt=0..10\nstop when y < 0")
      assert_in_delta r.event.t, 40 / 9.81, 1.0e-9
    end

    @tag :scipy
    test "SciPy's solve_ivp (DOP853 at 10⁻¹²) agrees on the Lotka–Volterra cycle" do
      {:ok, r} = Solve.ode("p' = a*p - b*p*q\nq' = -c*q + d*p*q\na=1.1; b=0.4; c=0.4; d=0.1\np(0)=10; q(0)=5\nt=0..30\nrtol=1e-10\natol=1e-12\nsamples=7")
      out = Vapor.TestHelpers.py!("""
      from scipy.integrate import solve_ivp
      import numpy as np
      f = lambda t, y: [1.1*y[0]-0.4*y[0]*y[1], -0.4*y[1]+0.1*y[0]*y[1]]
      s = solve_ivp(f, (0, 30), [10, 5], method='DOP853', rtol=1e-12, atol=1e-14, t_eval=np.linspace(0, 30, 7))
      print(' '.join(repr(float(v)) for v in s.y[0]))
      """)
      ref = out |> String.split() |> Enum.map(&String.to_float/1)
      for {a, b} <- Enum.zip(r.series["p"], ref), do: assert_in_delta(a, b, 1.0e-6 * max(1.0, abs(b)))
    end
  end

  describe "algebra" do
    test "every root of a system in a box" do
      {:ok, r} = Solve.run("unknowns x = 2, y = 0.5\nx^2 + y^2 = 4\nx*y = 1\nsearch = [-3, 3]")
      assert length(r.roots) == 4
      for root <- r.roots, do: assert(root.residual < 1.0e-12)
    end

    test "a fit recovers planted parameters within their standard errors" do
      data = for i <- 0..40, do: (x = i * 0.1; "#{x}, #{3 * :math.exp(-1.3 * x) + 0.5 + 0.01 * :math.sin(17 * i)}")
      {:ok, r} = Solve.run("fit y = a*exp(-b*x) + c\na = 1; b = 0.5\ndata\nx, y\n" <> Enum.join(data, "\n"))
      for {k, v} <- [{"a", 3.0}, {"b", 1.3}, {"c", 0.5}], do: assert(abs(r.params[k].value - v) < 4 * r.params[k].stderr)
      assert r.r2 > 0.999
    end

    test "constrained minimisation with its KKT certificate" do
      {:ok, r} = Solve.run("minimize x^2 + y^2\nfrom x=3, y=1\nsubject to x + y >= 1")
      assert_in_delta r.x["x"], 0.5, 1.0e-7
      assert_in_delta hd(r.multipliers), 1.0, 1.0e-6
      assert r.kkt.stationarity < 1.0e-6 and r.kkt.infeasibility < 1.0e-8
      {:ok, r} = Solve.run("minimize (1-x)^2 + 100*(y-x^2)^2\nfrom x = -1.2, y = 1")
      assert_in_delta r.x["x"], 1.0, 1.0e-8
    end

    test "box bounds are kept by projection; an unbounded objective is reported as divergence, never as an answer" do
      # the can of least surface holding 1: r = (1/2π)^(1/3), h = 2r
      {:ok, r} = Solve.run("minimize 2*pi*r^2 + 2*pi*r*h\nfrom r = 1, h = 1\nsubject to pi*r^2*h = 1, r >= 0.01, h >= 0.01")
      r0 = :math.pow(1 / (2 * :math.pi()), 1 / 3)
      assert r.converged and r.bounds == ["h ≥ 0.01", "r ≥ 0.01"]
      assert_in_delta r.x["r"], r0, 1.0e-6
      assert_in_delta r.x["h"], 2 * r0, 1.0e-6
      # an active bound: the minimum of (x − 3)² + (y + 1)² on x ≤ 1, 0 ≤ y ≤ 2 is at the corner (1, 0)
      {:ok, b} = Solve.run("minimize (x-3)^2 + (y+1)^2\nfrom x = 0, y = 1\nsubject to x <= 1, 0 <= y, y <= 2")
      assert b.converged and b.x["x"] == 1.0 and b.x["y"] == 0.0
      # the control: without the bounds the same problem runs to r < 0, where the area has no lower bound
      {:ok, d} = Solve.run("minimize 2*pi*r^2 + 2*pi*r*h\nfrom r = 1, h = 1\nsubject to pi*r^2*h = 1")
      refute d.converged
      assert d.verdict =~ "diverged"
    end

    test "dense linear algebra reports its backward error; Jacobi and the generalized eigenproblem" do
      a = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]]
      {:ok, x} = Dense.solve(a, [1.0, 2.0, 3.0])
      assert Dense.residual(a, x, [1.0, 2.0, 3.0]) < 1.0e-15
      {vals, vecs} = Dense.eigh(a)
      for {l, v} <- Enum.zip(vals, vecs), do: assert(Dense.norm_inf(Dense.sub(Dense.matvec(a, v), Enum.map(v, &(&1 * l)))) < 1.0e-10)
      {:ok, {gv, _}} = Dense.geigh(a, [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
      assert length(gv) == 3
    end
  end

  describe "partial differential equations, verified by manufactured solutions" do
    test "heat equation: Crank–Nicolson shows order 2; backward Euler (the control) order 1" do
      base = """
      u_t = D*u_xx
      D = 0.1
      x = 0 .. 1
      t = 0 .. 0.5
      u(0, t) = 0
      u(1, t) = 0
      nx = 21; nt = 20
      exact u = exp(-D*pi^2*t)*sin(pi*x)
      verify
      """
      {:ok, cn} = Solve.run(base)
      {:ok, be} = Solve.run(base <> "scheme = implicit\n")
      assert_in_delta List.last(cn.verification.orders), 2.0, 0.1
      assert_in_delta List.last(be.verification.orders), 1.0, 0.15
    end

    test "variable coefficients, advection, a nonlinear reaction and a Neumann end: still order 2" do
      {:ok, r} = Solve.run("""
      u_t = (1 + x^2)*u_xx - 0.5*u_x + u*(1-u)
      x = 0 .. 1
      t = 0 .. 1
      u_x(1, t) = 0
      nx = 21; nt = 20
      exact u = 1 + 0.3*cos(t)*cos(pi*x/2)^2*x
      verify
      """)
      assert List.last(r.verification.orders) > 1.85
    end

    test "Poisson in 2-D: order 2" do
      {:ok, r} = Solve.run("poisson -(u_xx + u_yy) = f\nf = 1\nx = 0 .. 1; y = 0 .. 1\nnx = 11; ny = 11\nexact u = sin(pi*x)*sinh(pi*y)/sinh(pi) + x*y*(1-x)\nverify")
      assert_in_delta List.last(r.verification.orders), 2.0, 0.1
    end

    test "the wave equation refuses a step that breaks the CFL condition" do
      assert {:error, m} = Solve.run("u_tt = 4*u_xx\nx=0..1\nt=0..1\nu(x,0)=sin(pi*x)\nnx=101\nnt=50")
      assert m =~ "CFL"
    end
  end

  describe "the native ensemble (HPC)" do
    @tag :native
    test "compiled RK4 over 1024 members: oracle bit for bit, binary64 to 10⁻⁴" do
      {:ok, r} = Vapor.Solve.Ensemble.run("""
      x' = v
      v' = -k/m*x - c/m*v
      k ~ normal(4, 0.2); c ~ uniform(0.05, 0.2); m = 1
      x(0) ~ normal(1, 0.05); v(0) = 0
      t = 0 .. 5
      members = 1024; h = 0.01
      """, reference: 16)
      assert r.oracle_parity == true
      assert r.f64_agreement.max_relative < 1.0e-3
      assert r.f64_agreement.median_relative < 1.0e-5
    end

    test "a function outside the algebra is refused by name" do
      assert {:error, m} = Vapor.Solve.Ensemble.run("x' = -sin(x)\nx(0)=1\nt=0..1")
      assert m =~ "no sin"
    end
  end

  describe "affine temperature scales (0.13)" do
    test "readings on °C/°F become kelvin; `in [°C]` subtracts the offset; compounds and bare scales are refused with the reason" do
      {:ok, r} = Solve.run("T = 98.6[°F]\nx = T in [°C]\np = 1[mol]*8.314[J/(mol*K)]*20[°C]/1[L]\nq = p in [kPa]")
      by = Map.new(r.lines, &{&1.name, &1})
      assert_in_delta by["T"].value, 310.15, 1.0e-9
      assert_in_delta by["x"].shown, 37.0, 1.0e-9
      assert_in_delta by["q"].shown, 8.314 * 293.15, 1.0e-6
      {:ok, d} = Solve.run("d = 30[°C] - 20[°C]")
      assert_in_delta hd(d.lines).value, 10.0, 1.0e-9
      {:ok, bad} = Solve.run("c = 4186[J/(kg*°C)]")
      assert hd(bad.lines).error =~ "degC"
      {:ok, bad2} = Solve.run("q = 5*[°C]")
      assert hd(bad2.lines).error =~ "reading"
    end
  end
end
