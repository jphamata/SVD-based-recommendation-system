defmodule Vapor.Science do
  @moduledoc """
  The science laboratory of the 0.11 round: physics past the classical
  (quantum, relativistic, a tokamak's equilibrium), chemistry and matter,
  and biology — each experiment a function of its parameters with a
  closed-form or published reference and a control that must fail
  (docs/SCIENCE.md). `experiments/0` lists them; `run/1` runs one and
  returns `%{name, value, reference, control, threshold, pass, data}` —
  the shape of the quality suite's checks, so the suite and the console
  run the same code.
  """
  alias Vapor.Science.{Biology, Chemistry, Plasma, Quantum, Relativity}

  @doc "The experiments, in the order the console shows them."
  def experiments do
    ~w(quantum_coherent quantum_tunneling relativistic_gyration exb_drift tokamak_equilibrium h2_hartree_fock heh_hartree_fock lennard_jones wright_fisher phylogeny hp_folding)
  end

  @doc "Run one experiment by name."
  def run("quantum_coherent") do
    r = Quantum.coherent()
    %{name: "quantum: a coherent state follows the classical orbit, ⟨x⟩ = x₀ cos t (split-step Fourier)", value: r.max_error, reference: 0.0,
      control: r.norm_drift, threshold: "max |⟨x⟩ − x₀cos t| < 10⁻³ over a period; norm drift (the unitarity check) < 10⁻¹⁰",
      pass: r.max_error < 1.0e-3 and r.norm_drift < 1.0e-10, data: %{trace: Enum.map(r.trace, &Tuple.to_list/1)}}
  end

  def run("quantum_tunneling") do
    r = Quantum.tunneling()
    %{name: "quantum: a packet below the barrier tunnels with the exact probability ∫T(k)|φ(k)|²dk", value: r.simulated, reference: r.exact, control: r.classical,
      threshold: "|simulated − exact| < 0.01; the classical particle (the control): < 0.01 crosses",
      pass: abs(r.simulated - r.exact) < 0.01 and r.classical < 0.01, data: Map.take(r, [:energy, :barrier])}
  end

  def run("relativistic_gyration") do
    r = Relativity.gyration(0.9)
    %{name: "relativity: gyration at v = 0.9c takes 2πγ/B (γ = 2.29), |u| kept by the Boris pusher", value: r.boris.period, reference: r.theory, control: r.euler.u_drift,
      threshold: "period within 10⁻⁴ (relative); |u| drift < 10⁻¹²; explicit Euler (the control) gains > 1 %",
      pass: abs(r.boris.period / r.theory - 1) < 1.0e-4 and r.boris.u_drift < 1.0e-12 and r.euler.u_drift > 0.01, data: %{gamma: r.gamma}}
  end

  def run("exb_drift") do
    r = Relativity.drift(0.3, 1.0)
    %{name: "relativity: the E×B drift velocity is E/B", value: r.measured, reference: r.theory, control: nil, threshold: "within 1 %",
      pass: abs(r.measured / r.theory - 1) < 0.01, data: %{}}
  end

  def run("tokamak_equilibrium") do
    [c, f] = for n <- [15, 31], do: Plasma.solve(n)
    cart = Plasma.solve(31, toroidal: false)
    {ra, _} = c.axis
    {rb, _} = f.axis
    {rx, _} = f.axis_exact
    %{name: "tokamak: Grad–Shafranov solved against Solov'ev's exact equilibrium; the axis (parabola through the grid maximum) converges as h²", value: f.max_error, reference: 0.0, control: cart.max_error,
      threshold: "flux error < 10⁻⁸; axis error falls ≥ 3× when h halves; the Cartesian Laplacian (the control) misses by > 10⁻³",
      pass: f.max_error < 1.0e-8 and abs(ra - rx) / max(abs(rb - rx), 1.0e-15) >= 3 and cart.max_error > 1.0e-3,
      data: %{axis_errors: [abs(ra - rx), abs(rb - rx)], axis: Tuple.to_list(f.axis), exact_axis: Tuple.to_list(f.axis_exact)}}
  end

  def run("h2_hartree_fock") do
    r = Chemistry.h2()
    far = Chemistry.h2(10.0).energy
    atoms = 2 * Chemistry.h_atom()
    %{name: "chemistry: H₂ (STO-3G, R = 1.4 bohr) by restricted Hartree–Fock vs Szabo & Ostlund's −1.1167 hartree", value: r.energy, reference: -1.1167, control: far - atoms,
      threshold: "within 10⁻⁴ hartree; pulled apart, RHF stays > 0.2 hartree above two atoms (the method's known failure, shown)",
      pass: abs(r.energy + 1.1167) < 1.0e-4 and far - atoms > 0.2, data: %{orbital_energies: r.orbital_energies}}
  end

  def run("heh_hartree_fock") do
    r = Chemistry.heh()
    %{name: "chemistry: HeH⁺ (STO-3G, R = 1.4632 bohr) vs Szabo & Ostlund's −2.860662 hartree", value: r.energy, reference: -2.860662, control: nil,
      threshold: "within 10⁻⁵ hartree", pass: abs(r.energy + 2.860662) < 1.0e-5, data: %{}}
  end

  def run("lennard_jones") do
    v = Chemistry.lj(steps: 300)
    e = Chemistry.lj(steps: 300, integrator: :euler)
    {rpeak, _} = Enum.max_by(v.rdf, &elem(&1, 1))
    %{name: "matter: a Lennard-Jones liquid (64 atoms) — velocity Verlet keeps the energy; its g(r) peaks near 2^{1/6}σ", value: v.fluctuation, reference: 0.0, control: e.drift,
      threshold: "energy fluctuation < 10⁻³ (relative) over 300 steps; explicit Euler (the control) > 10 %; first g(r) peak in [1.0, 1.25]σ",
      pass: v.fluctuation < 1.0e-3 and e.drift > 0.1 and rpeak >= 1.0 and rpeak <= 1.25,
      data: %{rdf: Enum.map(v.rdf, &Tuple.to_list/1), energies: Enum.take_every(v.energies, 5)}}
  end

  def run("wright_fisher") do
    {n, s} = {50, 0.05}
    exact = Biology.fixation_exact(n, s)
    sim = Biology.fixation_sim(n, s, 4000)
    neutral = Biology.fixation_sim(n, 0.0, 4000)
    %{name: "evolution: a beneficial mutant (N = 50, s = 0.05) fixes with the Wright–Fisher chain's exact probability", value: sim, reference: exact, control: neutral,
      threshold: "within 3 standard errors; the neutral control fixes at ≈ 1/N", pass: abs(sim - exact) < 3 * :math.sqrt(exact * (1 - exact) / 4000) and abs(neutral - 1 / n) < 3 * :math.sqrt(0.02 * 0.98 / 4000),
      data: %{kimura: Biology.kimura(n, s), neutral_exact: 1 / n}}
  end

  def run("phylogeny") do
    r = Biology.phylogeny()
    %{name: "genomics: a 6-taxon tree rebuilt from evolved sequences (Jukes–Cantor + neighbour joining)", value: r.rf, reference: 0, control: r.rf_shuffled,
      threshold: "Robinson–Foulds 0; distances within 10 %; shuffled sites (the control): RF > 0", pass: r.rf == 0 and r.distance_error < 0.1 and r.rf_shuffled > 0,
      data: %{distance_error: r.distance_error}}
  end

  def run("hp_folding") do
    f = Biology.fold(Biology.benchmark20())
    g = Biology.ground_state("HPHPPHHPHPPH")
    m = Biology.fold("HPHPPHHPHPPH", steps: 50_000)
    rnd = Biology.random_energy(Biology.benchmark20())
    %{name: "folding: the HP 20-mer reaches the published optimum −9; on a 12-mer the search equals exact enumeration", value: f.energy, reference: -9, control: rnd,
      threshold: "E = −9 (Unger & Moult 1993); 12-mer: search = enumeration; random conformations (the control) > −3 on average",
      pass: f.energy == -9 and m.energy == g.energy and rnd > -3, data: %{coords: Enum.map(f.coords, &Tuple.to_list/1), sequence: Biology.benchmark20()}}
  end

  @doc false
  def replay("science", %{"experiment" => name}) do
    if name in experiments(), do: {:ok, run(name)}, else: {:error, {:unknown_experiment, name}}
  end
  def replay(_, _), do: {:error, :bad_recipe}
end
