defmodule Vapor.EngineeringTest do
  @moduledoc """
  Engineering solvers (docs/ENGINEERING.md), each against a closed form, a
  published case or an outside oracle, with the certificate checked and
  a control where one exists: circuits (RC, RLC, diode, op-amp), power
  flow (Stagg & El-Abiad's five buses; Gauss–Seidel as the control),
  frames and trusses (beam formulas, method of joints, cantilever modes),
  plane-stress FEM (patch test; Q4 locking as the control), pipe
  networks (Colebrook against SciPy), chemical kinetics, flash and
  distillation.
  """
  use ExUnit.Case, async: true
  @moduletag timeout: 600_000
  alias Vapor.Engineering.{Circuit, FEM, Pipes, Power, Process, Structure}

  describe "circuits" do
    test "operating point with a diode: KCL and power balance certify it" do
      {:ok, r} = Circuit.run("V1 in 0 DC 5\nR1 in a 1k\nR2 a 0 2k\nR3 a d 1k\nD1 d 0 IS=1e-14\n.op")
      assert r.op.certificate.kcl_max < 1.0e-12
      assert r.op.certificate.power_balance < 1.0e-10
      # the diode's own equation at the solved voltage
      vd = r.op.nodes["d"]
      id = (r.op.nodes["a"] - vd) / 1000
      assert_in_delta id, 1.0e-14 * (:math.exp(vd / 0.025851991024560) - 1), 1.0e-9
    end

    test "RC charging: trapezoidal is second order; backward Euler (the control) first" do
      net = fn m, h -> "V1 in 0 PULSE(0 1 0 1n 1n 1 2)\nR1 in out 1k\nC1 out 0 1u\n.tran #{h} 3m\n" <> if(m == :be, do: ".method be\n", else: "") end
      err = fn m, h ->
        {:ok, r} = Circuit.run(net.(m, h))
        Enum.zip(r.tran.t, r.tran.nodes["out"]) |> Enum.filter(fn {t, _} -> t > 1.0e-3 end) |> Enum.map(fn {t, v} -> abs(v - (1 - :math.exp(-t / 1.0e-3))) end) |> Enum.max()
      end
      tr = :math.log2(err.(:trap, "20u") / err.(:trap, "10u"))
      be = :math.log2(err.(:be, "20u") / err.(:be, "10u"))
      assert tr > 1.6 and be < 1.3
    end

    test "series RLC: the current peaks at 1/2π√LC; the capacitor voltage at ω₀√(1 − 1/2Q²)" do
      {:ok, r} = Circuit.run("V1 in 0 AC 1\nR1 in a 10\nL1 a b 1m\nC1 b 0 1u\n.ac lin 2001 4k 6k")
      f0 = 1 / (2 * :math.pi() * :math.sqrt(1.0e-9))
      q = :math.sqrt(1.0e-3 / 1.0e-6) / 10
      pi = Enum.max_by(r.ac.points, & &1.branches["V1"].mag)
      assert_in_delta pi.f, f0, 1.0
      assert_in_delta pi.branches["V1"].mag, 0.1, 1.0e-6
      pv = Enum.max_by(r.ac.points, & &1.nodes["b"].mag)
      assert_in_delta pv.f, f0 * :math.sqrt(1 - 1 / (2 * q * q)), 1.0
      assert_in_delta pv.nodes["b"].mag, q / :math.sqrt(1 - 1 / (4 * q * q)), 1.0e-4
    end

    test "an inverting amplifier with an ideal op-amp: gain −R2/R1" do
      {:ok, r} = Circuit.run("V1 in 0 DC 1\nR1 in m 1k\nR2 m out 10k\nO1 out 0 m\n.op")
      assert_in_delta r.op.nodes["out"], -10.0, 1.0e-12
    end

    test "a node with no DC path to ground is named; a loop of voltage sources is refused" do
      {:ok, r} = Circuit.run("V1 a 0 DC 1\nR1 a b 1k\nC1 b c 1u\n.op")
      assert Enum.any?(r.op.warnings, &(&1 =~ "c"))
      assert {:error, m} = Circuit.run("V1 a 0 DC 1\nV2 a 0 DC 2\n.op")
      assert m =~ "singular"
    end
  end

  describe "transistors (0.13)" do
    @cs "VDD vdd 0 DC 5\nVG g 0 DC 1.5 AC 1\nRD vdd d 10k\nM1 d g 0 0 NMOS KP=50u VTO=0.7 LAMBDA=0.02 W=10u L=1u\n"
    @ce "VCC vcc 0 DC 12\nR1 vcc b 100k\nR2 b 0 20k\nRC vcc c 4.7k\nRE e 0 1k\nQ1 c b e NPN IS=1e-15 BF=150 BR=2\n"
    @inv "VDD vdd 0 DC 3.3\nVIN in 0 DC 1.2\nM1 out in 0 0 NMOS KP=100u VTO=0.6 LAMBDA=0.05 W=4u L=1u\nM2 out in vdd vdd PMOS KP=50u VTO=-0.7 LAMBDA=0.05 W=8u L=1u\n"

    test "a common-source stage: the square law by hand; KCL certified with the transistor's own equation" do
      {:ok, r} = Circuit.run(@cs <> ".op")
      # saturation: Id = β/2 (Vgs − Vt)² (1 + λVds), β = 500 µA/V²; solve Vd = 5 − 10k·Id
      vd = r.op.nodes["d"]
      # (the 10⁻¹² S GMIN across the channel carries 3·10⁻¹² A: the only difference)
      assert_in_delta (5 - vd) / 10_000, 250.0e-6 * 0.8 * 0.8 * (1 + 0.02 * vd), 1.0e-11
      assert r.op.certificate.kcl_max < 1.0e-9
    end

    test "a CMOS inverter and a common-emitter stage converge; the emitter current is base + collector" do
      {:ok, r} = Circuit.run(@inv <> ".op")
      assert r.op.nodes["out"] > 3.0 and r.op.certificate.kcl_max < 1.0e-9
      {:ok, q} = Circuit.run(@ce <> ".op")
      ve = q.op.nodes["e"]; vc = q.op.nodes["c"]; vb = q.op.nodes["b"]
      ie = ve / 1000; ic = (12 - vc) / 4700; ib = (12 - vb) / 100_000 - vb / 20_000
      assert_in_delta ie, ic + ib, 1.0e-12
      assert_in_delta ic / ib, 150 * (1 - 1.0e-3), 3.0
    end

    @tag :ngspice
    test "operating points and the small-signal gain equal ngspice (level-1 MOSFET, Gummel–Poon at its Ebers–Moll defaults)" do
      run_ng = fn netlist, prints ->
        dir = Path.join(System.tmp_dir!(), "vapor-ng-#{System.unique_integer([:positive])}")
        File.mkdir_p!(dir)
        f = Path.join(dir, "c.cir")
        # SPICE device cards: the model goes in a .model line; same temperature as ours (300.00 K, TNOM too)
        cards = netlist |> String.split("\n", trim: true) |> Enum.map(fn l ->
          cond do
            l =~ ~r/^M/ -> (([name, d, g, s, b, typ | opts] = String.split(l)); "#{name} #{d} #{g} #{s} #{b} m#{name}\n.model m#{name} #{typ} LEVEL=1 #{Enum.join(opts, " ")}")
            l =~ ~r/^Q/ -> (([name, c, b, e, typ | opts] = String.split(l)); "#{name} #{c} #{b} #{e} q#{name}\n.model q#{name} #{typ} #{Enum.join(opts, " ")}")
            true -> l
          end
        end)
        File.write!(f, "* vapor\n" <> Enum.join(cards, "\n") <> "\n.temp 26.85\n.options TNOM=26.85\n.control\n#{prints}\n.endc\n.end\n")
        {out, _} = System.cmd("ngspice", ["-b", f], stderr_to_stdout: true)
        File.rm_rf!(dir)
        Regex.scan(~r/^(\S+)\s*=\s*([-+\d.e]+)/m, out) |> Map.new(fn [_, k, v] -> {k, elem(Float.parse(v), 0)} end)
      end
      for {net, nodes} <- [{@cs, ["d"]}, {@ce, ["c", "b", "e"]}, {@inv, ["out"]}] do
        ng = run_ng.(net, "op\nprint " <> Enum.map_join(nodes, " ", &"v(#{&1})"))
        {:ok, r} = Circuit.run(net <> ".op")
        for n <- nodes, do: assert_in_delta(r.op.nodes[n], ng["v(#{n})"], 2.0e-6 * max(1.0, abs(ng["v(#{n})"])))
      end
      ng = run_ng.(@cs, "ac lin 1 1k 1k\nprint vm(d)")
      {:ok, r} = Circuit.run(@cs <> ".ac lin 1 1k 1k")
      gain = hd(r.ac.points).nodes["d"].mag
      assert_in_delta gain, ng["vm(d)"], 1.0e-6 * ng["vm(d)"]
    end
  end

  describe "power flow" do
    @stagg """
    base 100
    bus 1 slack V=1.06
    bus 2 pq P=20 Q=20
    bus 3 pq P=-45 Q=-15
    bus 4 pq P=-40 Q=-5
    bus 5 pq P=-60 Q=-10
    line 1 2 r=0.02 x=0.06 b=0.06
    line 1 3 r=0.08 x=0.24 b=0.05
    line 2 3 r=0.06 x=0.18 b=0.04
    line 2 4 r=0.06 x=0.18 b=0.04
    line 2 5 r=0.04 x=0.12 b=0.03
    line 3 4 r=0.01 x=0.03 b=0.02
    line 4 5 r=0.08 x=0.24 b=0.05
    """

    test "Stagg & El-Abiad's five-bus system: the published voltages, angles and slack power" do
      {:ok, r} = Power.run(@stagg)
      b = Map.new(r.buses, &{&1.id, &1})
      for {id, v, a} <- [{"2", 1.047, -2.807}, {"3", 1.024, -4.997}, {"4", 1.024, -5.329}, {"5", 1.018, -6.150}] do
        assert_in_delta b[id].v, v, 6.0e-4
        assert_in_delta b[id].angle_deg, a, 6.0e-3
      end
      assert_in_delta b["1"].p_mw, 129.59, 0.01
      assert r.certificate.max_mismatch_pu < 1.0e-10 and r.certificate.balance_mw < 1.0e-9
      assert r.iterations <= 6
    end

    test "Gauss–Seidel (the control) reaches the same voltages in many more iterations" do
      {:ok, n} = Power.run(@stagg)
      {:ok, g} = Power.run(@stagg, method: :gauss_seidel)
      assert Enum.zip(n.buses, g.buses) |> Enum.all?(fn {a, b} -> abs(a.v - b.v) < 1.0e-9 end)
      assert g.iterations > 10 * n.iterations
    end
  end

  describe "frames and trusses" do
    test "cantilever tip deflection PL³/3EI and the fixed-end moment PL, exactly" do
      {:ok, r} = Structure.run("node 1 0 0\nnode 2 3[m] 0\nsupport 1 fixed\nbeam 1 2 E=200[GPa] A=0.01 I=1e-4 n=4\nload 2 fy=-10[kN]")
      assert_in_delta r.nodes["2"].uy, -10.0e3 * 27 / (3 * 200.0e9 * 1.0e-4), 1.0e-12
      assert_in_delta r.reactions["1"].mz, 30.0e3, 1.0e-6
      assert r.certificate.relative < 1.0e-12
    end

    test "simply supported beam under a uniform load: 5wL⁴/384EI and wL²/8" do
      {:ok, r} = Structure.run("node 1 0 0\nnode 2 10 0\nsupport 1 pinned\nsupport 2 roller-x\nbeam 1 2 E=200e9 A=0.01 I=1e-4 n=2\nudl 1 2 w=-5e3")
      assert_in_delta r.nodes["1~2~1"].uy, -5 * 5.0e3 * 1.0e4 / (384 * 200.0e9 * 1.0e-4), 1.0e-12
      assert_in_delta r.members |> Enum.flat_map(& &1.moment) |> Enum.max(), 5.0e3 * 100 / 8, 1.0e-6
    end

    test "a triangular truss: forces of the method of joints" do
      {:ok, r} = Structure.run("node 1 0 0\nnode 2 4 0\nnode 3 2 3\nsupport 1 pinned\nsupport 2 roller-x\ntruss 1 2 E=200e9 A=1e-3\ntruss 1 3 E=200e9 A=1e-3\ntruss 2 3 E=200e9 A=1e-3\nload 3 fy=-10e3")
      f = Map.new(r.members, &{&1.member, hd(&1.axial)})
      # joint 3: 2·F·(3/√13) = −10 kN → F = −5√13/3 kN (compression); bottom chord: F·(2/√13) in tension
      assert_in_delta f["1-3"], -5.0e3 * :math.sqrt(13) / 3, 1.0e-6
      assert_in_delta f["1-2"], 5.0e3 * :math.sqrt(13) / 3 * 2 / :math.sqrt(13), 1.0e-6
    end

    test "cantilever natural frequencies converge to 1.8751² and 4.6941² √(EI/ρAL⁴)" do
      {:ok, r} = Structure.run("node 1 0 0\nnode 2 2 0\nsupport 1 fixed\nbeam 1 2 E=200e9 A=1e-3 I=1e-7 rho=7850 n=10\nmodes 2")
      base = :math.sqrt(200.0e9 * 1.0e-7 / (7850 * 1.0e-3 * 16))
      [m1, m2] = r.modes
      assert_in_delta m1.omega / base, 1.875104 ** 2, 1.0e-4
      assert_in_delta m2.omega / base, 4.694091 ** 2, 2.0e-3
    end

    test "a mechanism is refused" do
      assert {:error, m} = Structure.run("node 1 0 0\nnode 2 1 0\nsupport 1 pinned\nbeam 1 2 E=1 A=1 I=1")
      assert m =~ "mechanism"
    end
  end

  describe "plane-stress FEM" do
    test "the patch test passes for Q4 and QM6 on distorted elements" do
      for k <- [:q4, :qm6] do
        p = FEM.patch_test(k)
        assert p.displacement_error < 1.0e-14 and p.stress_error < 1.0e-12
      end
    end

    test "QM6 does not lock in bending; Q4 (the control) does" do
      tip = fn el ->
        {:ok, r} = FEM.run("plate x=0..10 y=-0.5..0.5 nx=10 ny=2\nmaterial E=1000 nu=0.3 t=1\nelement #{el}\nfix x=0\ntraction x=10 ty=-1")
        r.nodes |> Map.values() |> Enum.filter(&(&1.x == 10.0)) |> Enum.map(& &1.uy) |> then(&(Enum.sum(&1) / length(&1)))
      end
      beam = -4.0 - 10 / (5 / 6 * 1000 / 2.6)
      assert abs(tip.("qm6") / beam - 1) < 0.02
      assert abs(tip.("q4") / beam - 1) > 0.2
    end
  end

  describe "pipe networks" do
    @tag :scipy
    test "one pipe between two reservoirs: Colebrook solved like SciPy's root finder" do
      {:ok, r} = Pipes.run("reservoir A head=100\nreservoir B head=80\npipe 1 A B L=1000 D=0.3 eps=0.00026")
      out = Vapor.TestHelpers.py!("""
      from scipy.optimize import brentq
      import math
      nu=1e-6;D=0.3;L=1000;e=0.00026;g=9.80665
      def f(Re): return brentq(lambda f: 1/math.sqrt(f)+2*math.log10(e/D/3.7+2.51/(Re*math.sqrt(f))),1e-4,1,xtol=1e-15)
      def h(Q):
        A=math.pi*D*D/4;V=Q/A;Re=V*D/nu; return f(Re)*L/D*V*V/(2*g)
      print(repr(brentq(lambda Q: h(Q)-20,1e-4,2,xtol=1e-15)))
      """)
      assert_in_delta hd(r.pipes).flow, String.to_float(String.trim(out)), 1.0e-10
    end

    test "a looped network: continuity and every loop's energy close" do
      {:ok, r} = Pipes.run("""
      reservoir R head=60[m]
      junction A elev=10[m] demand=15[L/s]
      junction B elev=12[m] demand=20[L/s]
      junction C elev=8[m] demand=10[L/s]
      pipe 1 R A L=800[m] D=250[mm] eps=0.1[mm]
      pipe 2 A B L=600[m] D=200[mm] eps=0.1[mm] K=2
      pipe 3 R B L=1200[m] D=200[mm] eps=0.1[mm]
      pipe 4 A C L=500[m] D=150[mm] eps=0.1[mm]
      pipe 5 C B L=400[m] D=150[mm] eps=0.1[mm]
      """)
      assert r.loops == 2
      assert r.certificate.continuity_max < 1.0e-12 and r.certificate.loop_energy_max < 1.0e-9
    end
  end

  describe "chemical processes" do
    test "A → B → C against Bateman's closed form; the conserved total certified" do
      {:ok, r} = Process.reactions("A -> B ; k = 1\nB -> C ; k = 0.5\nA0 = 1\nt = 0 .. 10\nrtol = 1e-10\natol = 1e-12")
      b = fn t -> 1 / (0.5 - 1) * (:math.exp(-t) - :math.exp(-0.5 * t)) end
      assert Enum.zip(r.t, r.series["B"]) |> Enum.all?(fn {t, v} -> abs(v - b.(t)) < 1.0e-8 end)
      assert [%{combination: "A + B + C", drift: d}] = r.invariants
      assert d < 1.0e-12
    end

    test "a network's conserved moieties come from its stoichiometry alone" do
      {:ok, r} = Process.reactions("A + B -> C ; k = 2\nC <-> D ; kf = 1, kb = 0.2\n2 A -> E ; k = 0.1\nA0 = 1; B0 = 0.8\nt = 0 .. 20")
      assert length(r.invariants) == 2
      assert Enum.all?(r.invariants, &(&1.drift < 1.0e-9))
    end

    test "a CSTR reaches its steady state C = C₀/(1 + kτ)" do
      {:ok, r} = Process.reactions("A -> B ; k = 2\nreactor cstr tau=5\nfeed A=1\nt = 0 .. 100")
      assert_in_delta r.final["A"], 1 / 11, 1.0e-6
    end

    test "a benzene–toluene flash closes its balances; Raoult's bubble pressure" do
      {:ok, r} = Process.flash("component benzene z=0.4 A=6.90565 B=1211.033 C=220.79\ncomponent toluene z=0.6 A=6.95464 B=1344.8 C=219.482\nT = 100\nP = 760")
      assert r.phase == "two phases"
      assert r.certificate.balance < 1.0e-15 and abs(r.certificate.sum_x - 1) < 1.0e-12 and abs(r.certificate.sum_y - 1) < 1.0e-12
      assert r.vapour_fraction > 0 and r.vapour_fraction < 1
    end

    test "McCabe–Thiele at total reflux falls to Fenske (the control)" do
      {:ok, r} = Process.distill("alpha = 2.5; xF = 0.45; xD = 0.95; xB = 0.05; q = 1; Rfactor = 1.3")
      assert r.control.total_reflux_stages == ceil(r.fenske)
      assert r.stages > r.fenske and r.stages < 2 * r.gilliland
      assert {:error, m} = Process.distill("alpha = 2.5; xF = 0.45; xD = 0.95; xB = 0.05; q = 1; R = 1.0")
      assert m =~ "minimum reflux"
    end
  end
end
