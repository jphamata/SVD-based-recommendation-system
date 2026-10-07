defmodule Vapor.Science.Quantum do
  @moduledoc """
  The time-dependent Schrödinger equation in one dimension (ħ = m = 1),
  by the split-step Fourier method (Feit, Fleck & Steiger 1982): half a
  step of the potential, a full kinetic step in momentum space, half a
  step of the potential — unitary, second order in dt. Binary64 on the
  BEAM, deterministic.

  Measured against closed forms (docs/CIENCIA.md):

    * a **coherent state** of the harmonic oscillator follows the classical
      orbit, ⟨x⟩(t) = x₀ cos t, and keeps its norm;
    * a wave packet **tunnels** through a square barrier with the
      probability ∫ T(k)|φ(k)|² dk, T the exact plane-wave transmission —
      while the classical particle, with less energy than the barrier,
      never crosses (the control).
  """

  @doc "Radix-2 FFT of a list of `{re, im}` (length a power of two); `inverse: true` scales by 1/N."
  def fft(xs, inverse \\ false) do
    n = length(xs)
    out = do_fft(List.to_tuple(xs), n, if(inverse, do: 1.0, else: -1.0))
    if inverse, do: Enum.map(out, fn {a, b} -> {a / n, b / n} end), else: out
  end

  defp do_fft(t, 1, _sign), do: [elem(t, 0)]

  defp do_fft(t, n, sign) do
    half = div(n, 2)
    even = do_fft(List.to_tuple(for(i <- 0..(half - 1), do: elem(t, 2 * i))), half, sign)
    odd = do_fft(List.to_tuple(for(i <- 0..(half - 1), do: elem(t, 2 * i + 1))), half, sign)

    tw =
      Enum.zip(even, odd)
      |> Enum.with_index()
      |> Enum.map(fn {{{er, ei}, {or_, oi}}, k} ->
        a = sign * 2 * :math.pi() * k / n
        {c, s} = {:math.cos(a), :math.sin(a)}
        {tr, ti} = {c * or_ - s * oi, c * oi + s * or_}
        {{er + tr, ei + ti}, {er - tr, ei - ti}}
      end)

    Enum.map(tw, &elem(&1, 0)) ++ Enum.map(tw, &elem(&1, 1))
  end

  @doc """
  A grid: `n` points on [−L/2, L/2) and the matching wavenumbers.
  """
  def grid(n, l) do
    dx = l / n
    xs = for j <- 0..(n - 1), do: -l / 2 + j * dx
    ks = for j <- 0..(n - 1), do: 2 * :math.pi() / l * if(j < div(n, 2), do: j, else: j - n)
    %{n: n, l: l, dx: dx, x: xs, k: ks}
  end

  @doc "A Gaussian packet centred at x0 with mean wavenumber k0 and width σ, normalised."
  def packet(g, x0, k0, sigma) do
    psi = for x <- g.x, do: (a = :math.exp(-((x - x0) ** 2) / (2 * sigma * sigma)); {a * :math.cos(k0 * x), a * :math.sin(k0 * x)})
    normalize(psi, g.dx)
  end

  def normalize(psi, dx) do
    nrm = :math.sqrt(norm(psi, dx))
    Enum.map(psi, fn {a, b} -> {a / nrm, b / nrm} end)
  end

  def norm(psi, dx), do: Enum.reduce(psi, 0.0, fn {a, b}, s -> s + a * a + b * b end) * dx

  @doc "Advance `steps` split steps of size dt in the potential `v` (list on the grid); calls `observe.(t, psi)` every `every` steps."
  def evolve(g, psi, v, dt, steps, observe \\ nil, every \\ 1) do
    half = Enum.map(v, fn vx -> phase(-vx * dt / 2) end)
    kin = Enum.map(g.k, fn k -> phase(-k * k * dt / 2) end)

    Enum.reduce(1..steps, {psi, []}, fn s, {psi, obs} ->
      psi = cmul(psi, half) |> fft() |> cmul(kin) |> fft(true) |> cmul(half)
      obs = if observe && rem(s, every) == 0, do: [observe.(s * dt, psi) | obs], else: obs
      {psi, obs}
    end)
    |> then(fn {psi, obs} -> {psi, Enum.reverse(obs)} end)
  end

  defp phase(a), do: {:math.cos(a), :math.sin(a)}
  defp cmul(a, b), do: Enum.zip_with(a, b, fn {ar, ai}, {br, bi} -> {ar * br - ai * bi, ar * bi + ai * br} end)

  def mean_x(g, psi), do: Enum.zip(g.x, psi) |> Enum.reduce(0.0, fn {x, {a, b}}, s -> s + x * (a * a + b * b) end) |> Kernel.*(g.dx)

  @doc """
  The coherent state of the oscillator V = x²/2 started at x0: the largest
  |⟨x⟩(t) − x0 cos t| over one period and the norm's drift.
  """
  def coherent(opts \\ []) do
    {n, l, x0, dt} = {Keyword.get(opts, :n, 128), Keyword.get(opts, :l, 20.0), Keyword.get(opts, :x0, 2.0), Keyword.get(opts, :dt, 0.01)}
    g = grid(n, l)
    psi0 = packet(g, x0, 0.0, 1.0)
    v = Enum.map(g.x, &(&1 * &1 / 2))
    steps = round(2 * :math.pi() / dt)
    {psi, obs} = evolve(g, psi0, v, dt, steps, fn t, p -> {t, mean_x(g, p)} end, 10)
    err = obs |> Enum.map(fn {t, m} -> abs(m - x0 * :math.cos(t)) end) |> Enum.max()
    %{max_error: err, norm_drift: abs(norm(psi, g.dx) - 1.0), trace: obs}
  end

  @doc "Exact transmission of a plane wave of wavenumber k through a barrier of height v0 and width a."
  def transmission(k, v0, a) do
    e = k * k / 2

    cond do
      e < v0 ->
        kappa = :math.sqrt(2 * (v0 - e))
        1 / (1 + v0 * v0 * :math.pow(:math.sinh(kappa * a), 2) / (4 * e * (v0 - e)))

      e > v0 ->
        kp = :math.sqrt(2 * (e - v0))
        1 / (1 + v0 * v0 * :math.pow(:math.sin(kp * a), 2) / (4 * e * (e - v0)))

      true ->
        1 / (1 + v0 * a * a / 2)
    end
  end

  @doc """
  A packet (mean energy below the barrier) against a square barrier: the
  simulated transmitted probability, the exact expectation
  ∫ T(k)|φ(k)|² dk, and the classical answer (0 below the barrier, by
  energy conservation, for every k in the packet below √(2V₀)).
  """
  def tunneling(opts \\ []) do
    # dx = 0.1: the barrier is exactly a/dx grid cells wide
    {n, l} = {Keyword.get(opts, :n, 2048), Keyword.get(opts, :l, 204.8)}
    {v0, a, k0, sigma, x0} = {Keyword.get(opts, :v0, 1.0), Keyword.get(opts, :a, 1.0), Keyword.get(opts, :k0, 1.2), Keyword.get(opts, :sigma, 10.0), -60.0}
    dt = Keyword.get(opts, :dt, 0.05)
    g = grid(n, l)
    v = Enum.map(g.x, fn x -> if x >= -1.0e-9 and x < a - 1.0e-9, do: v0, else: 0.0 end)
    psi0 = packet(g, x0, k0, sigma)
    t_end = (abs(x0) + 60) / k0
    {psi, _} = evolve(g, psi0, v, dt, round(t_end / dt))
    sim = Enum.zip(g.x, psi) |> Enum.reduce(0.0, fn {x, {re, im}}, s -> if x > a, do: s + re * re + im * im, else: s end) |> Kernel.*(g.dx)

    # |φ(k)|² of the packet: a Gaussian about k0 with standard deviation 1/(σ√2)
    sk = 1 / (sigma * :math.sqrt(2))
    ks = for i <- -400..400, do: k0 + i * sk * 6 / 400
    w = Enum.map(ks, fn k -> :math.exp(-((k - k0) ** 2) / (2 * sk * sk)) end)
    expect = Enum.zip(ks, w) |> Enum.reduce(0.0, fn {k, wk}, s -> if k > 0, do: s + wk * transmission(k, v0, a), else: s end) |> Kernel./(Enum.sum(w))
    above = Enum.zip(ks, w) |> Enum.reduce(0.0, fn {k, wk}, s -> if k * k / 2 > v0, do: s + wk, else: s end) |> Kernel./(Enum.sum(w))
    %{simulated: sim, exact: expect, classical: above, energy: k0 * k0 / 2, barrier: v0}
  end
end
