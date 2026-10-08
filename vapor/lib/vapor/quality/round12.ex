defmodule Vapor.Quality.Round12 do
  @moduledoc """
  Quality checks for the 0.12 round, in the suite's discipline: a value, a
  **control** that a broken or naive implementation would produce, and a
  threshold that separates them. Each line answers "is this signal or
  noise?" with a number that could have come out otherwise.

  | check | value | control (must fail, or be caught) |
  |---|---|---|
  | ODE | Dormand–Prince on the oscillator to 10⁻⁸; Robertson at Hairer & Wanner's values | fixed-step RK4, h = 0.1: error > 10⁻⁵ |
  | PDE | Crank–Nicolson observed order 2 by manufactured solution | backward Euler: order 1 |
  | units | a consistent system runs | the same system with a force where a velocity belongs: refused before running |
  | optimisation | the can of least surface, KKT holding, bounds kept | without bounds: divergence reported, not an answer |
  | circuits | trapezoidal RC: order 2 | backward Euler: order 1 |
  | power flow | Stagg & El-Abiad's voltages, Newton in ≤ 6 iterations | Gauss–Seidel: > 10× the iterations |
  | frames | cantilever ω₁ to 10⁻⁴ of Euler–Bernoulli | one element: error > 10⁻³ |
  | plane FEM | QM6 within 2 % of beam theory | Q4: locks, off by > 20 % |
  | kinetics | conserved moieties drift < 10⁻⁹ | a non-conserved combination drifts |
  | distillation | McCabe–Thiele at total reflux = ⌈Fenske⌉ | below minimum reflux: refused |
  | SAT/DRUP | S(3) = 13: witness checked, refutation checked | a truncated proof: rejected |
  | rewriting | Knuth–Bendix: the ten group rules; i(x·y) = i(y)·i(x) decided | the axioms merely oriented: undecided |
  | Gröbner | Thales proved | a false variant: not implied |
  | proposals | a correct colouring accepted | one changed colour: rejected |
  | chess | perft 8 902 (start, 3), 2 039 (Kiwipete, 2); a mate-in-2 proof replayed | mate in 1 where there is none: not claimed |
  | shogi | perft 30 · 900 · 25 470 | — |
  | Go | 57 legal 2×2 positions (Tromp) | — |
  | poker | Kuhn: exploitability < 0.01, value −1/18 | uniform play: exploitability 0.458 |
  | proteins | NMR models of 1LCD: TM > 0.8; DCA precision > 0.9; fold TM > 0.6 | shuffled alignment: precision at chance, fold TM < 0.3 |
  | render | white and gradient furnaces | the biased estimator: caught |
  | ink (0.17) | the ball's hard shadow at the ambient level, its outline on the silhouette | the same scene without the ball: the ground lit |
  """
  alias Vapor.{Logic, Render, Solve}
  alias Vapor.Bio.{Coevolution, Structure}
  alias Vapor.Engineering.{Circuit, FEM, Power, Process}
  alias Vapor.Play.{Chess, Go, MNK, Poker, Shogi}

  def run(_opts \\ []) do
    %{checks: List.flatten([bench(), engineering(), logic(), boards(), proteins(), render(), ink()])}
  end

  defp check(name, value, control, threshold, pass), do: %{name: name, value: value, control: control, threshold: threshold, pass: pass}
  defp p(rel), do: Path.join(to_string(:code.priv_dir(:vapor)), rel)
  defp f(x) when is_float(x) and x != 0.0 and abs(x) < 1.0e-3, do: String.to_float(:erlang.float_to_binary(x, [{:scientific, 3}]))
  defp f(x), do: if(is_float(x), do: Float.round(x, 6), else: x)

  # ============================================================ workbench

  defp bench do
    osc = "x' = v\nv' = -4*x\nx(0) = 1; v(0) = 0\nt = 0 .. 10\nrtol = 1e-10\natol = 1e-12"
    err = fn r -> Enum.zip(r.t, r.series["x"]) |> Enum.map(fn {t, x} -> abs(x - :math.cos(2 * t)) end) |> Enum.max() end
    {:ok, dp} = Solve.ode(osc)
    {:ok, rk} = Solve.ode(osc <> "\nmethod = rk4\nh = 0.1")
    {:ok, rob} = Solve.ode("a' = -0.04*a + 1e4*b*c\nb' = 0.04*a - 1e4*b*c - 3e7*b^2\nc' = 3e7*b^2\na(0) = 1; b(0) = 0; c(0) = 0\nt = 0 .. 40\nrtol = 1e-6\natol = 1e-10")
    heat = "u_t = D*u_xx\nD = 0.1\nx = 0 .. 1\nt = 0 .. 0.5\nu(0, t) = 0\nu(1, t) = 0\nnx = 21; nt = 20\nexact u = exp(-D*pi^2*t)*sin(pi*x)\nverify\n"
    {:ok, cn} = Solve.run(heat)
    {:ok, be} = Solve.run(heat <> "scheme = implicit\n")
    good_units = Solve.ode("x' = v\nv' = -k/m*x\nk = 4[N/m]; m = 1[kg]\nx(0)=1[m]\nv(0)=0[m/s]\nt=0..1[s]")
    bad_units = Solve.ode("x' = v\nv' = -k*x\nk = 4[N/m]\nx(0)=1[m]\nv(0)=0[m/s]\nt=0..1[s]")
    {:ok, can} = Solve.run("minimize 2*pi*r^2 + 2*pi*r*h\nfrom r = 1, h = 1\nsubject to pi*r^2*h = 1, r >= 0.01, h >= 0.01")
    {:ok, div} = Solve.run("minimize 2*pi*r^2 + 2*pi*r*h\nfrom r = 1, h = 1\nsubject to pi*r^2*h = 1")
    r0 = :math.pow(1 / (2 * :math.pi()), 1 / 3)

    [check("workbench: Dormand–Prince 5(4) with dense output on x'' = −4x", f(err.(dp)), f(err.(rk)), "< 10⁻⁸; RK4 at h = 0.1 > 10⁻⁵", err.(dp) < 1.0e-8 and err.(rk) > 1.0e-5),
     check("workbench: Robertson's stiff kinetics at t = 40 (Hairer & Wanner), switch to Rosenbrock", f(rob.final["a"]), rob.switched, "a = 0.7158271 ± 2·10⁻⁶; stiffness detected", abs(rob.final["a"] - 0.7158271) < 2.0e-6 and rob.switched != nil),
     check("workbench: heat equation, observed order by manufactured solution", f(List.last(cn.verification.orders)), f(List.last(be.verification.orders)), "Crank–Nicolson 2 ± 0.1; backward Euler 1 ± 0.15",
           abs(List.last(cn.verification.orders) - 2) < 0.1 and abs(List.last(be.verification.orders) - 1) < 0.15),
     check("workbench: dimensions checked before integrating", inspect(elem(good_units, 0)), inspect(bad_units) |> String.slice(0, 80), "consistent system runs; v' with a force refused", match?({:ok, _}, good_units) and match?({:error, _}, bad_units)),
     check("workbench: the can of least surface (equality + box bounds, projected BFGS)", f(can.x["r"]), div.verdict |> String.slice(0, 40), "r = (1/2π)^⅓ to 10⁻⁶, KKT holds; unbounded: reported diverged",
           abs(can.x["r"] - r0) < 1.0e-6 and can.converged and not div.converged)]
  end

  # =========================================================== engineering

  defp engineering do
    net = fn m, h -> "V1 in 0 PULSE(0 1 0 1n 1n 1 2)\nR1 in out 1k\nC1 out 0 1u\n.tran #{h} 3m\n" <> if(m == :be, do: ".method be\n", else: "") end
    cerr = fn m, h ->
      {:ok, r} = Circuit.run(net.(m, h))
      Enum.zip(r.tran.t, r.tran.nodes["out"]) |> Enum.filter(fn {t, _} -> t > 1.0e-3 end) |> Enum.map(fn {t, v} -> abs(v - (1 - :math.exp(-t / 1.0e-3))) end) |> Enum.max()
    end
    tr = :math.log2(cerr.(:trap, "20u") / cerr.(:trap, "10u"))
    be = :math.log2(cerr.(:be, "20u") / cerr.(:be, "10u"))

    stagg = "base 100\nbus 1 slack V=1.06\nbus 2 pq P=20 Q=20\nbus 3 pq P=-45 Q=-15\nbus 4 pq P=-40 Q=-5\nbus 5 pq P=-60 Q=-10\nline 1 2 r=0.02 x=0.06 b=0.06\nline 1 3 r=0.08 x=0.24 b=0.05\nline 2 3 r=0.06 x=0.18 b=0.04\nline 2 4 r=0.06 x=0.18 b=0.04\nline 2 5 r=0.04 x=0.12 b=0.03\nline 3 4 r=0.01 x=0.03 b=0.02\nline 4 5 r=0.08 x=0.24 b=0.05"
    {:ok, nr} = Power.run(stagg)
    {:ok, gs} = Power.run(stagg, method: :gauss_seidel)
    v5 = Enum.find(nr.buses, &(&1.id == "5")).v

    base = :math.sqrt(200.0e9 * 1.0e-7 / (7850 * 1.0e-3 * 16))
    w1 = fn n -> {:ok, r} = Vapor.Engineering.Structure.run("node 1 0 0\nnode 2 2 0\nsupport 1 fixed\nbeam 1 2 E=200e9 A=1e-3 I=1e-7 rho=7850 n=#{n}\nmodes 1"); abs(hd(r.modes).omega / base / 1.875104 ** 2 - 1) end

    tip = fn el ->
      {:ok, r} = FEM.run("plate x=0..10 y=-0.5..0.5 nx=10 ny=2\nmaterial E=1000 nu=0.3 t=1\nelement #{el}\nfix x=0\ntraction x=10 ty=-1")
      r.nodes |> Map.values() |> Enum.filter(&(&1.x == 10.0)) |> Enum.map(& &1.uy) |> then(&(Enum.sum(&1) / length(&1)))
    end
    beam = -4.0 - 10 / (5 / 6 * 1000 / 2.6)
    {q6, q4} = {abs(tip.("qm6") / beam - 1), abs(tip.("q4") / beam - 1)}

    {:ok, rx} = Process.reactions("A + B -> C ; k = 2\nC <-> D ; kf = 1, kb = 0.2\n2 A -> E ; k = 0.1\nA0 = 1; B0 = 0.8\nt = 0 .. 20")
    drift = rx.invariants |> Enum.map(& &1.drift) |> Enum.max()
    a_drift = abs(List.last(rx.series["A"]) - hd(rx.series["A"]))
    {:ok, ds} = Process.distill("alpha = 2.5; xF = 0.45; xD = 0.95; xB = 0.05; q = 1; Rfactor = 1.3")
    low = Process.distill("alpha = 2.5; xF = 0.45; xD = 0.95; xB = 0.05; q = 1; R = 1.0")

    [check("engineering: RC transient, observed order", f(tr), f(be), "trapezoidal > 1.6; backward Euler < 1.3", tr > 1.6 and be < 1.3),
     check("engineering: Stagg & El-Abiad five-bus power flow", "|V5| #{f(v5)} in #{nr.iterations} it.", "Gauss–Seidel #{gs.iterations} it.", "|V5| = 1.018 ± 6·10⁻⁴, mismatch < 10⁻¹⁰; GS > 10× iterations",
           abs(v5 - 1.018) < 6.0e-4 and nr.certificate.max_mismatch_pu < 1.0e-10 and gs.iterations > 10 * nr.iterations),
     check("engineering: cantilever first natural frequency (consistent mass)", f(w1.(10)), f(w1.(1)), "10 elements < 10⁻⁴; 1 element > 10⁻³", w1.(10) < 1.0e-4 and w1.(1) > 1.0e-3),
     check("engineering: slender plate in bending, tip deflection vs Timoshenko beam", f(q6), f(q4), "QM6 < 2 %; Q4 (locking) > 20 %", q6 < 0.02 and q4 > 0.2),
     check("engineering: conserved moieties of a reaction network (from stoichiometry alone)", f(drift), f(a_drift), "2 invariants, drift < 10⁻⁹; [A] alone changes > 0.1", length(rx.invariants) == 2 and drift < 1.0e-9 and a_drift > 0.1),
     check("engineering: McCabe–Thiele at total reflux = ⌈Fenske⌉", ds.control.total_reflux_stages, inspect(low) |> String.slice(0, 60), "equal; R below the minimum refused", ds.control.total_reflux_stages == ceil(ds.fenske) and match?({:error, _}, low))]
  end

  # ================================================================= logic

  defp logic do
    {:ok, s3} = Logic.run("schur 3")
    p = Logic.Problems.schur(14, 3)
    {:unsat, proof, _} = Logic.SAT.solve(p)
    cut = Logic.DRUP.check(p, Enum.drop(proof, -1))
    {:ok, kb} = Logic.Rewrite.complete("vars x y z\nprecedence i > * > e\ne * x = x\ni(x) * x = e\n(x * y) * z = x * (y * z)")
    {:ok, d} = Logic.Rewrite.decide(kb.rules, "i(x * y)", "i(y) * i(x)")
    {:ok, eqs, _, _} = Logic.Rewrite.parse("vars x y z\ne * x = x\ni(x) * x = e\n(x * y) * z = x * (y * z)")
    {:ok, d0} = Logic.Rewrite.decide(eqs, "i(x * y)", "i(y) * i(x)")
    {:ok, th} = Logic.run("vars x y a b\nhyp x^2 + y^2 - 1\nhyp a + 1\nhyp b - 1\nclaim (x - a)*(x - b) + y*y")
    {:ok, fv} = Logic.run("vars x y a b\nhyp x^2 + y^2 - 1\nhyp a + 1\nhyp b - 1\nclaim (x - a)*(x - b) + y")
    w = s3.below.witness
    {:ok, good} = Logic.check("schur 3", %{"witness" => w})
    {:ok, bad} = Logic.check("schur 3", %{"witness" => List.replace_at(w, 0, Enum.at(w, 1))})

    [check("logic: Schur S(3) — witness at 13 checked, DRUP refutation at 14 checked", s3.value, inspect(cut), "13, both certificates valid; the proof without its last lemma rejected",
           s3.value == 13 and s3.below.checked and s3.refutation.drup.valid and cut == {:error, :no_empty_clause}),
     check("logic: Knuth–Bendix completes the group axioms; i(x·y) = i(y)·i(x) decided", length(kb.rules), d0.equal, "10 rules, equal; oriented axioms alone: not decided", length(kb.rules) == 10 and d.equal and not d0.equal),
     check("logic: Thales' theorem by Gröbner basis (Rabinowitsch)", th.verdict, fv.verdict, "proved; the altered claim not implied", th.verdict == "proved" and fv.verdict == "not implied"),
     check("logic: an outside proposal is checked, not trusted", good.accepted, bad.accepted, "the solver's colouring accepted; one colour changed rejected", good.accepted and not bad.accepted)]
  end

  # ================================================================ boards

  defp boards do
    start = Chess.start()
    {:ok, kiwi} = Chess.from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    {:ok, ladder} = Chess.from_fen("7k/8/8/8/8/8/1R6/R5K1 w - - 0 1")
    {:mate, proof} = Chess.prove_mate(ladder, 2)
    replay = Chess.verify_mate(ladder, proof, 2)
    m1 = Chess.prove_mate(ladder, 1)
    sh = Shogi.start()
    shp = Enum.map(1..3, &Shogi.perft(sh, &1))
    go2 = Go.legal_positions(2)
    k = Poker.solve(Poker.Kuhn, 1000)
    u = Poker.exploitability(Poker.Kuhn, Poker.uniform())
    {v33, _} = Vapor.Play.solve(MNK, MNK.new(3, 3, 3))

    [check("boards: chess perft (start depth 3, Kiwipete depth 2)", "#{Chess.perft(start, 3)} · #{Chess.perft(kiwi, 2)}", "—", "8 902 · 2 039 (published)", Chess.perft(start, 3) == 8902 and Chess.perft(kiwi, 2) == 2039),
     check("boards: mate in 2 proved and replayed by the independent checker", inspect(replay), inspect(m1), "{:ok, …}; no mate in 1 claimed", match?({:ok, _}, replay) and m1 == :no_mate),
     check("boards: shogi perft from the initial position", Enum.join(shp, " · "), "—", "30 · 900 · 25 470 (published)", shp == [30, 900, 25470]),
     check("boards: Go, legal positions on 2×2 (Tromp–Taylor)", go2, "—", "57 (Tromp & Farnebäck)", go2 == 57),
     check("boards: Kuhn poker by CFR+ — exploitability and game value", f(k.exploitability), f(u), "< 0.01 and value −1/18 ± 0.01; uniform 0.458", k.exploitability < 0.01 and abs(k.value + 1 / 18) < 0.01 and abs(u - 0.4583) < 0.001),
     check("boards: tic-tac-toe solved exactly", v33, "—", "a draw (0)", v33 == 0)]
  end

  # ============================================================== proteins

  defp proteins do
    read = fn n -> File.read!(p("quality/protein/#{n}")) end
    [m1, m2 | _] = read.("1LCD.pdb") |> Structure.read_models() |> Enum.map(& &1.ca)
    tm12 = Structure.tm_score(m2, m1).tm
    {:ok, nat} = Structure.read_pdb(read.("1A8O.pdb"))
    l = length(nat.ca)
    ss = Structure.secondary(nat.ca)
    allc = Structure.contacts(nat.ca, 8.0, 3)
    truth = Structure.contacts(nat.ca, 8.0, 6)
    k = length(truth)
    local = Enum.filter(allc, fn {i, j} -> j - i < 6 end)
    msa = Coevolution.sample(l, allc, n: 2000, coupling: 0.6, sweeps: 3, seed: 1)
    shuf = Coevolution.shuffle(msa)
    {dca, dcs} = {Coevolution.dca(msa), Coevolution.dca(shuf)}
    {pd, ps} = {Structure.precision(dca, truth, k), Structure.precision(dcs, truth, k)}
    chance = k / div((l - 6) * (l - 5), 2)
    good = Structure.fold(l, local ++ Enum.take(dca, k), helices: ss, restarts: 2)
    bad = Structure.fold(l, local ++ Enum.take(dcs, k), helices: ss, restarts: 2)
    {tg, tb} = {Structure.tm_score(good.ca, nat.ca).tm, Structure.tm_score(bad.ca, nat.ca).tm}

    [check("proteins: TM-score between NMR models 1 and 2 of 1LCD", f(tm12), "—", "> 0.8 (the same fold); equals TM-align in proteins_test", tm12 > 0.8),
     check("proteins: contacts from a planted co-evolution alignment (DCA, top-k)", f(pd), f(ps), "> 0.9; shuffled alignment < 3× chance (#{Float.round(chance, 3)})", pd > 0.9 and ps < 3 * chance),
     check("proteins: the pipeline alignment → DCA → distance geometry on 1A8O", f(tg), f(tb), "TM > 0.6; from the shuffled alignment's contacts < 0.3", tg > 0.6 and tb < 0.3)]
  end

  # ================================================================ render

  defp render do
    u = Render.furnace(0.8, spp: 8)
    g = Render.furnace_gradient(0.8)
    b = Render.furnace_gradient(0.8, biased: true)
    [check("render: the white furnace (energy conservation)", f(u.max_error), "—", "< 10⁻⁹ on every sphere pixel", u.max_error < 1.0e-9),
     check("render: the gradient furnace a(½ + n_y/3) (the estimator's distribution)", f(g.mean_error), f(b.mean_error), "|error| < 0.005; the biased estimator < −0.03", abs(g.mean_error) < 0.005 and b.mean_error < -0.03)]
  end

  # ================================================================= ink

  @ink_scene """
  camera pos=0,3,5 look=0,0,0 fov=45
  sun dir=0,1,0 color=1,1,1 power=2
  plane y=0 mat=diffuse albedo=0.8,0.8,0.8
  sphere c=0,1.2,0 r=0.6 mat=diffuse albedo=0.2,0.4,0.8
  """

  # the camera looks at the origin: the image's centre is the ground straight below the ball, in its shadow
  defp ink do
    {:ok, s} = Render.parse(@ink_scene)
    {:ok, bare} = Render.parse(@ink_scene |> String.split("\n") |> Enum.reject(&String.starts_with?(&1, "sphere")) |> Enum.join("\n"))
    centre = fn r -> r.linear |> Enum.at(45) |> Enum.at(60) |> elem(0) end
    a = Render.ink(s, width: 120, height: 90, bands: 4)
    b = Render.ink(bare, width: 120, height: 90, bands: 4)
    [check("ink: a hard sun shadow at the ambient level (0.8 × 0.3), the ball outlined", "#{f(centre.(a))}, #{a.edges} outline pixels", "without the ball: #{f(centre.(b))}",
           "shadow = 0.24 to 10⁻⁹ with an outline; without the ball the same pixel lit at 0.8", abs(centre.(a) - 0.24) < 1.0e-9 and a.edges > 0 and centre.(b) == 0.8)]
  end
end
