defmodule Vapor.Modal.Diffusion do
  @moduledoc """
  **Diffusion sampling** on vapor's substrate: deterministic DDIM over any
  noise predictor admitted by the model airlock, with classifier-free
  guidance — and an analytic denoiser that makes the sampler itself
  falsifiable.

  The sampler (Song et al., η = 0) walks a schedule ᾱ₁ > … backwards:

      ε̂ = ε(x_t, t, ∅) + w·(ε(x_t, t, c) − ε(x_t, t, ∅))       (guidance, one program run for both)
      x̂₀ = clamp((x_t − √(1 − ᾱ_t)·ε̂) / √ᾱ_t, −1, 1)            (images live in [−1, 1])
      ε̂ ← (x_t − √ᾱ_t·x̂₀) / √(1 − ᾱ_t)                          (consistent with the clamped x̂₀)
      x_{t'} = √ᾱ_{t'}·x̂₀ + √(1 − ᾱ_{t'})·ε̂

  The network is a program (certified: the same bits on every substrate);
  the update is binary64 on the BEAM; the starting noise comes from
  `Vapor.Modal.Rng` (splitmix64 + libm-free Box–Muller). So a sample is a
  function of (model, class, seed, steps, guidance): anyone recomputes it.

  **Why this is checkable.** For data that is a mixture of Gaussians the
  optimal noise predictor has a closed form: `analytic/1` builds it —
  and, for a finite training set, the posterior mean of the clean image is
  *softmax attention* with the noisy image as the query and the training
  images as keys and values. `sample/4` driven by it must (i) put the
  mixture's modes at their weights and spreads, and (ii) with the
  empirical set, return training images: the textbook memorisation that a
  learned denoiser must *not* show. `docs/ANY_TO_ANY.md` measures both,
  and a trained denoiser against them.

  Models: a `vapor_mlp` (or any `:map` contract) whose input row is
  `[x_t (d) | 16 time features | class one-hot padded to 16]` and whose
  output is ε; its `config.json` carries the schedule (`schedule.abar`).
  """
  alias Vapor.{CR, Lock, Tensor}
  alias Vapor.Modal.{Rng, Runner}

  @doc "Admit a denoiser checkpoint: `{:ok, %{spec, weights, abar, d, programs}}`."
  def load(dir) do
    with {:ok, m} <- Lock.open(dir),
         %{"schedule" => %{"abar" => abar}} <- m.spec.config.raw || {:error, :schedule} do
      {:ok, %{spec: m.spec, weights: m.weights, abar: List.to_tuple(abar), d: m.spec.width, programs: %{}}}
    else
      {:error, :schedule} -> {:error, Vapor.Rejection.new(:diffusion, "a schedule (schedule.abar) in config.json", "export the training schedule with the model")}
      {:error, _} = e -> e
      _ -> {:error, Vapor.Rejection.new(:diffusion, "a schedule (schedule.abar) in config.json", "export the training schedule with the model")}
    end
  end

  @doc "The 16 time features of step `t` of `big_t` (8 sines, 8 cosines of t/T·π·2ᵏ, correctly rounded)."
  def time_features(t, big_t) do
    us = for k <- 0..7, do: t / big_t * :math.pow(2, k) * :math.pi()
    Enum.map(us, &CR.sin_f64/1) ++ Enum.map(us, &CR.cos_f64/1)
  end

  @doc """
  The DDIM timesteps for `steps` steps over a schedule of `big_t`: evenly
  spaced, descending, from `big_t` down to 1.
  """
  def timesteps(big_t, steps) do
    for i <- (steps - 1)..0//-1, uniq: true, do: 1 + div(i * (big_t - 1), max(steps - 1, 1))
  end

  @doc """
  Draw `n` samples of `class` (an integer, or `nil` for unconditional).
  Options: `seed` (0), `steps` (25), `guidance` (2.0), `worker`,
  `denoiser` (a function `(xs, t, class) → ε rows` replacing the model —
  how `analytic/2` plugs in), `clip` (true: x̂₀ clamped to [−1, 1], the
  range of images), `trace` (keep every x̂₀). Returns
  `%{samples: [[float]], trace}` in [−1, 1].
  """
  def sample(model, class, n, opts \\ []) do
    big_t = tuple_size(model.abar)
    d = model.d
    steps = Keyword.get(opts, :steps, 25)
    w = Keyword.get(opts, :guidance, 2.0)
    seed = Keyword.get(opts, :seed, 0)
    eps_fn = Keyword.get(opts, :denoiser) || model_denoiser(model, n, w, opts[:worker])
    clip = if Keyword.get(opts, :clip, true), do: fn v -> v |> max(-1.0) |> min(1.0) end, else: & &1

    x = Rng.normal(Rng.key({:diffusion, seed, class}), n * d) |> Enum.chunk_every(d)
    ts = timesteps(big_t, steps)

    {x, trace} =
      ts
      |> Enum.with_index()
      |> Enum.reduce({x, []}, fn {t, i}, {x, trace} ->
        a = elem(model.abar, t - 1)
        a_prev = case Enum.at(ts, i + 1) do
          nil -> 1.0
          tp -> elem(model.abar, tp - 1)
        end

        eps = eps_fn.(x, t, class)
        {sa, s1a, sp, s1p} = {:math.sqrt(a), :math.sqrt(1 - a), :math.sqrt(a_prev), :math.sqrt(1 - a_prev)}

        {x0s, xs} =
          Enum.zip_with(x, eps, fn xr, er ->
            x0 = Enum.zip_with(xr, er, fn xv, ev -> clip.((xv - s1a * ev) / sa) end)
            # ε̂ re-derived from the clamped x̂₀, so the pair stays consistent: without
            # this, the first steps (ᾱ ≈ 0, where x̂₀ amplifies ε̂'s error 300×)
            # leave a clamped x̂₀ and an ε̂ that disagree, and DDIM drifts off the
            # data (measured: 17 % → 99 % of generated digits read back right)
            e2 = Enum.zip_with(xr, x0, fn xv, x0v -> (xv - sa * x0v) / s1a end)
            {x0, Enum.zip_with(x0, e2, fn x0v, ev -> sp * x0v + s1p * ev end)}
          end)
          |> Enum.unzip()

        {xs, if(opts[:trace], do: [x0s | trace], else: trace)}
      end)

    %{samples: x, trace: Enum.reverse(trace), timesteps: ts}
  end

  # the model as ε(xs, t, class): one program run over 2n rows (n conditional, n unconditional)
  defp model_denoiser(model, n, w, worker) do
    {:ok, p} = Lock.build(model.spec, model.weights, rows: 2 * n)
    big_t = tuple_size(model.abar)
    k = model.spec.in_width
    d = model.d

    fn xs, t, class ->
      tf = time_features(t, big_t)
      cond_oh = one_hot(class)
      rows = Enum.map(xs, &(&1 ++ tf ++ cond_oh)) ++ Enum.map(xs, &(&1 ++ tf ++ List.duplicate(0.0, 16)))
      rows = Enum.map(rows, &(&1 ++ List.duplicate(0.0, k - length(&1))))
      out = Runner.run(p, %{rows: Tensor.from_list(:f32, [2 * n, k], List.flatten(rows))}, worker: worker).out
      {c, u} = out |> Tensor.to_floats() |> Enum.chunk_every(d) |> Enum.split(n)
      if class == nil, do: u, else: Enum.zip_with(c, u, fn cr, ur -> Enum.zip_with(cr, ur, fn cv, uv -> uv + w * (cv - uv) end) end)
    end
  end

  defp one_hot(nil), do: List.duplicate(0.0, 16)
  defp one_hot(c), do: for(i <- 0..15, do: if(i == c, do: 1.0, else: 0.0))

  # ------------------------------------------------------------- analytic --

  @doc """
  The optimal noise predictor of a Gaussian mixture `[{weight, mean, var}]`
  (isotropic components; `var` 0 = the empirical distribution of points),
  per class: `components` is `%{class => [{w, mean, var}]}` (`nil` key: the
  whole mixture). For x_t = √ᾱ x₀ + √(1−ᾱ) ε,

      p(k | x_t) ∝ w_k · N(x_t; √ᾱ μ_k, (ᾱ v_k + 1 − ᾱ) I)          (a softmax over components)
      E[x₀ | x_t] = Σ_k p(k | x_t) · (μ_k + √ᾱ v_k (x_t − √ᾱ μ_k) / (ᾱ v_k + 1 − ᾱ))
      ε̂ = (x_t − √ᾱ E[x₀ | x_t]) / √(1 − ᾱ)

  With v = 0 the posterior is attention: query x_t, keys √ᾱ·μ_k, values
  μ_k, temperature 1 − ᾱ. Returns a `denoiser` for `sample/4`.
  """
  def analytic(model, components) do
    fn xs, t, class ->
      a = elem(model.abar, t - 1)
      comps = Map.fetch!(components, class)

      Enum.map(xs, fn x ->
        logits =
          Enum.map(comps, fn {wk, mu, v} ->
            s2 = a * v + 1 - a
            d2 = Enum.zip_reduce(x, mu, 0.0, fn xv, mv, acc -> acc + (xv - :math.sqrt(a) * mv) ** 2 end)
            :math.log(wk) - d2 / (2 * s2) - length(x) / 2 * :math.log(s2)
          end)

        m = Enum.max(logits)
        ps = Enum.map(logits, &:math.exp(&1 - m))
        z = Enum.sum(ps)

        x0 =
          Enum.zip(ps, comps)
          |> Enum.reduce(List.duplicate(0.0, length(x)), fn {p, {_w, mu, v}}, acc ->
            s2 = a * v + 1 - a
            post = Enum.zip_with(x, mu, fn xv, mv -> mv + :math.sqrt(a) * v * (xv - :math.sqrt(a) * mv) / s2 end)
            Enum.zip_with(acc, post, fn s, pv -> s + p / z * pv end)
          end)

        Enum.zip_with(x, x0, fn xv, x0v -> (xv - :math.sqrt(a) * x0v) / :math.sqrt(1 - a) end)
      end)
    end
  end

  # ---------------------------------------------------------------- measures --

  @doc "Euclidean distance from each sample to its nearest neighbour in `pool` (lists of floats)."
  def nearest(samples, pool) do
    Enum.map(samples, fn s -> pool |> Enum.map(&dist(s, &1)) |> Enum.min() end)
  end

  @doc "Euclidean distance."
  def dist(a, b), do: :math.sqrt(Enum.zip_reduce(a, b, 0.0, fn x, y, s -> s + (x - y) * (x - y) end))
end
