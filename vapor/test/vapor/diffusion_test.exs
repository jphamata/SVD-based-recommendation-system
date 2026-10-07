defmodule Vapor.DiffusionTest do
  @moduledoc """
  The sampler checked against closed forms: with the optimal denoiser of a
  Gaussian mixture (`Diffusion.analytic/2`), DDIM must put the modes at
  their weights and spreads; with the empirical set (variance 0 — the
  denoiser is softmax attention over the training points) it must return
  training points: memorisation, the behaviour a learned denoiser is
  measured against.
  """
  use ExUnit.Case, async: true
  alias Vapor.Modal.Diffusion

  # the cosine schedule of improved DDPM (as train_digits.py)
  defp schedule(t \\ 200, s \\ 0.008) do
    f = fn i -> :math.pow(:math.cos((i / t + s) / (1 + s) * :math.pi() / 2), 2) end
    for(i <- 1..t, do: (f.(i) / f.(0)) |> max(1.0e-5) |> min(0.9999)) |> List.to_tuple()
  end

  test "timesteps: evenly spaced, descending, ending at 1" do
    assert Diffusion.timesteps(200, 5) == [200, 150, 100, 50, 1]
    assert Diffusion.timesteps(10, 10) == Enum.to_list(10..1)
  end

  test "a Gaussian mixture: modes at their weights, spreads at their variances" do
    model = %{abar: schedule(), d: 2}
    comps = [{0.5, [-2.0, 0.0], 0.04}, {0.3, [2.0, 1.0], 0.09}, {0.2, [0.0, -2.5], 0.01}]
    r = Diffusion.sample(model, nil, 600, steps: 60, seed: 3, clip: false, denoiser: Diffusion.analytic(model, %{nil => comps}))

    groups = Enum.group_by(r.samples, fn x -> comps |> Enum.with_index() |> Enum.min_by(fn {{_, mu, _}, _} -> Diffusion.dist(x, mu) end) |> elem(1) end)

    for {{w, mu, v}, k} <- Enum.with_index(comps) do
      xs = Map.get(groups, k, [])
      assert_in_delta length(xs) / 600, w, 0.06
      # per-axis standard deviation around the mode
      for axis <- 0..1 do
        vals = Enum.map(xs, &Enum.at(&1, axis))
        m = Enum.sum(vals) / length(vals)
        sd = :math.sqrt(Enum.sum(Enum.map(vals, &((&1 - m) ** 2))) / length(vals))
        assert_in_delta m, Enum.at(mu, axis), 0.08
        assert_in_delta sd, :math.sqrt(v), 0.25 * :math.sqrt(v) + 0.02
      end
    end
  end

  test "the empirical set: the optimal denoiser is attention over the training points, and it memorises them" do
    model = %{abar: schedule(), d: 4}
    train = [[0.5, -0.5, 0.2, 0.9], [-0.7, 0.1, 0.4, -0.3], [0.0, 0.8, -0.6, 0.1]]
    comps = Enum.map(train, &{1 / 3, &1, 0.0})
    r = Diffusion.sample(model, nil, 30, steps: 50, seed: 1, denoiser: Diffusion.analytic(model, %{nil => comps}))
    assert Enum.all?(Diffusion.nearest(r.samples, train), &(&1 < 1.0e-3))
  end

  test "deterministic in its seed" do
    model = %{abar: schedule(), d: 2}
    den = Diffusion.analytic(model, %{nil => [{1.0, [0.3, -0.2], 0.05}]})
    a = Diffusion.sample(model, nil, 4, seed: 9, denoiser: den)
    assert a == Diffusion.sample(model, nil, 4, seed: 9, denoiser: den)
    refute a.samples == Diffusion.sample(model, nil, 4, seed: 10, denoiser: den).samples
  end
end
