defmodule Vapor.Diffusion.Scheduler do
  @moduledoc """
  The noise schedules and samplers of latent diffusion, as diffusers defines
  them (compared step for step in `test/vapor/diffusion_pipeline_test.exs`):

    * betas `scaled_linear` (Stable Diffusion: √β linear from 0.00085 to
      0.012 over 1000 steps) or `linear`; ᾱ the cumulative product;
    * timestep spacing `leading` (SD 1.x, with `steps_offset`), `linspace`
      or `trailing`;
    * **DDIM** (η = 0), **Euler** (`EulerDiscreteScheduler`), **DPM-Solver++
      2M** (`DPMSolverMultistepScheduler`, second order, first order on the
      last step when there are fewer than 15).

  Arithmetic is binary64 on lists of floats (the latents); the only
  transcendental functions are the correctly rounded `exp`, `log` and
  square root, so a schedule is the same numbers on every machine.
  """
  alias Vapor.CR

  defstruct [:kind, :steps, :timesteps, :sigmas, :ac, :n, :final_ac, :offset, :spacing, :init_sigma]

  @doc """
  A scheduler of `kind` (`:ddim`, `:euler`, `:dpmpp_2m`) for `steps` steps.
  Options (diffusers' scheduler_config keys, as atoms): `num_train_timesteps`
  (1000), `beta_start` (0.00085), `beta_end` (0.012), `beta_schedule`
  ("scaled_linear"), `timestep_spacing` ("leading"), `steps_offset` (1),
  `set_alpha_to_one` (false).
  """
  def new(kind, steps, opts \\ []) do
    n = Keyword.get(opts, :num_train_timesteps, 1000)
    {b0, b1} = {Keyword.get(opts, :beta_start, 0.00085), Keyword.get(opts, :beta_end, 0.012)}

    betas =
      case Keyword.get(opts, :beta_schedule, "scaled_linear") do
        "scaled_linear" -> for i <- 0..(n - 1), do: (fn r -> r * r end).(:math.sqrt(b0) + (:math.sqrt(b1) - :math.sqrt(b0)) * i / (n - 1))
        "linear" -> for i <- 0..(n - 1), do: b0 + (b1 - b0) * i / (n - 1)
      end

    {ac, _} = Enum.map_reduce(betas, 1.0, fn b, p -> v = p * (1.0 - b); {v, v} end)
    ac = List.to_tuple(ac)
    offset = Keyword.get(opts, :steps_offset, 1)
    spacing = Keyword.get(opts, :timestep_spacing, "leading")
    ts = timesteps(kind, spacing, n, steps, offset)
    final_ac = if Keyword.get(opts, :set_alpha_to_one, false), do: 1.0, else: elem(ac, 0)
    sigma_all = for i <- 0..(n - 1), do: :math.sqrt((1 - elem(ac, i)) / elem(ac, i))
    sigmas = Enum.map(ts, &interp(sigma_all, &1)) ++ [0.0]
    smax = Enum.max(Enum.drop(sigmas, -1))
    init = if kind == :euler and spacing == "leading", do: :math.sqrt(smax * smax + 1), else: (if kind == :euler, do: smax, else: 1.0)
    %__MODULE__{kind: kind, steps: steps, timesteps: ts, sigmas: sigmas, ac: ac, n: n, final_ac: final_ac, offset: offset, spacing: spacing, init_sigma: init}
  end

  # each diffusers scheduler spaces its timesteps its own way: DPM-Solver
  # divides by steps + 1, Euler keeps linspace fractional (σ interpolated)
  defp timesteps(:dpmpp_2m, "leading", n, steps, offset) do
    ratio = div(n, steps + 1)
    for(i <- steps..1//-1, do: i * ratio + offset)
  end

  defp timesteps(_, "leading", n, steps, offset) do
    ratio = div(n, steps)
    (for(i <- 0..(steps - 1), do: i * ratio) |> Enum.reverse()) |> Enum.map(&(&1 + offset))
  end

  defp timesteps(:dpmpp_2m, "linspace", n, steps, _offset) do
    for(i <- 0..steps, do: round_half_even((n - 1) * i / steps)) |> Enum.reverse() |> Enum.drop(-1)
  end

  defp timesteps(:euler, "linspace", n, steps, _offset) do
    for(i <- (steps - 1)..0//-1, do: linspace(n - 1, steps, i))
  end

  defp timesteps(_, "linspace", n, steps, _offset) do
    for(i <- (steps - 1)..0//-1, do: round_half_even(linspace(n - 1, steps, i)))
  end

  defp timesteps(_, "trailing", n, steps, _offset) do
    ratio = n / steps
    for(i <- steps..1//-1, do: round_half_even(i * ratio) - 1)
  end

  # numpy's round: half to even
  defp round_half_even(x) do
    f = Float.floor(x)
    d = x - f
    cond do
      d > 0.5 -> trunc(f) + 1
      d < 0.5 -> trunc(f)
      rem(trunc(f), 2) == 0 -> trunc(f)
      true -> trunc(f) + 1
    end
  end

  # numpy.linspace(0, stop, num)[i]: i·step, the last point exactly stop
  defp linspace(stop, num, i) when i == num - 1, do: stop * 1.0
  defp linspace(stop, num, i), do: i * (stop / (num - 1))

  defp interp(table, t) when is_integer(t), do: Enum.at(table, t)

  defp interp(table, t) do
    i = trunc(Float.floor(t))
    if i == t, do: Enum.at(table, i), else: (fn a, b -> a + (t - i) * (b - a) end).(Enum.at(table, i), Enum.at(table, i + 1))
  end

  @doc "The multiplier applied to the noise that starts sampling (diffusers' `init_noise_sigma`)."
  def init_sigma(%__MODULE__{init_sigma: s}), do: s

  @doc "The model input for step `i` (Euler scales by 1/√(σ²+1); the others pass it through)."
  def scale(%__MODULE__{kind: :euler, sigmas: sg}, x, i) do
    k = 1.0 / :math.sqrt(Enum.at(sg, i) * Enum.at(sg, i) + 1.0)
    Enum.map(x, &(&1 * k))
  end

  def scale(_, x, _), do: x

  @doc """
  One step from the model's noise prediction `eps` at step index `i`;
  `state` carries the multistep history (DPM++). Returns `{x_prev, state}`.
  """
  def step(s, eps, x, i, state \\ %{})

  def step(%__MODULE__{kind: :ddim} = s, eps, x, i, state) do
    t = Enum.at(s.timesteps, i)
    prev = t - div(s.n, s.steps)
    a = elem(s.ac, t)
    ap = if prev >= 0, do: elem(s.ac, prev), else: s.final_ac
    {sa, s1a, sap, s1ap} = {:math.sqrt(a), :math.sqrt(1 - a), :math.sqrt(ap), :math.sqrt(1 - ap)}
    {Enum.zip_with(x, eps, fn xv, e -> x0 = (xv - s1a * e) / sa; sap * x0 + s1ap * e end), state}
  end

  def step(%__MODULE__{kind: :euler} = s, eps, x, i, state) do
    {sig, nxt} = {Enum.at(s.sigmas, i), Enum.at(s.sigmas, i + 1)}
    dt = nxt - sig
    {Enum.zip_with(x, eps, fn xv, e -> xv + e * dt end), state}
  end

  def step(%__MODULE__{kind: :dpmpp_2m} = s, eps, x, i, state) do
    sig = Enum.at(s.sigmas, i)
    {at, st} = alpha_sigma(sig)
    x0 = Enum.zip_with(x, eps, fn xv, e -> (xv - st * e) / at end)
    nxt = Enum.at(s.sigmas, i + 1)
    {an, sn} = alpha_sigma(nxt)
    lam = fn a, sg -> CR.log_f64(a) - CR.log_f64(sg) end
    lam_s = lam.(at, st)
    lower_final = i == s.steps - 1 and s.steps < 15

    out =
      cond do
        nxt == 0.0 ->
          # the last step to σ = 0: the data prediction itself (diffusers: final_sigmas_type "zero")
          x0

        state[:prev_x0] == nil or lower_final ->
          h = lam.(an, sn) - lam_s
          k = an * (CR.exp_f64(-h) - 1.0)
          Enum.zip_with(x, x0, fn xv, d -> sn / st * xv - k * d end)

        true ->
          {lam_prev, x0_prev} = {state.prev_lambda, state.prev_x0}
          h = lam.(an, sn) - lam_s
          h0 = lam_s - lam_prev
          r0 = h0 / h
          k = an * (CR.exp_f64(-h) - 1.0)
          Enum.zip_with([x, x0, x0_prev], fn [xv, d0, d1p] ->
            d1 = (d0 - d1p) / r0
            sn / st * xv - k * d0 - 0.5 * k * d1
          end)
      end

    {out, %{prev_x0: x0, prev_lambda: lam_s}}
  end

  defp alpha_sigma(sig), do: (fn a -> {a, sig * a} end).(1.0 / :math.sqrt(sig * sig + 1.0))

  @doc "Noise added to clean latents at step index `i` (img2img, inpainting): √ᾱ·x₀ + √(1−ᾱ)·ε, or x₀ + σ·ε for Euler."
  def add_noise(%__MODULE__{kind: :euler, sigmas: sg}, x0, noise, i) do
    sig = Enum.at(sg, i)
    Enum.zip_with(x0, noise, &(&1 + sig * &2))
  end

  def add_noise(%__MODULE__{} = s, x0, noise, i) do
    a = elem(s.ac, Enum.at(s.timesteps, i))
    {sa, s1} = {:math.sqrt(a), :math.sqrt(1 - a)}
    Enum.zip_with(x0, noise, &(sa * &1 + s1 * &2))
  end
end
