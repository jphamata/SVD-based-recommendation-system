defmodule Vapor.Modal.World do
  @moduledoc """
  A small, fully specified world in three modalities — the fixture on which
  the any-to-any layer is *measured*, not demonstrated.

    * four colours and four notes, and a bijection between them
      (`red ↔ do`, `green ↔ mi`, `blue ↔ sol`, `yellow ↔ si`);
    * images: 16×16 RGB scenes, a soft-edged disc on the left and one on
      the right over a vertical gradient — caption `"<left> <right>"`;
    * sounds: a sine at the note's frequency (C4, E4, G4, B4), 1024 samples
      at 8 kHz, with seeded phase and amplitude;
    * text: a word vocabulary (colours, notes, `.`, padding).

  Every ground truth is known in closed form, so every output can be scored
  against it; and the controls (noise, shuffles) come from the same world.
  Four of the sixteen colour pairs are **held out** of every fit, so the
  measurements are of generalisation, not recall.
  """
  alias Vapor.Modal.{Audio, Image}

  @colours %{"red" => [0.85, 0.15, 0.1], "green" => [0.1, 0.7, 0.2], "blue" => [0.15, 0.25, 0.85], "yellow" => [0.9, 0.8, 0.1]}
  @notes %{"do" => 261.63, "mi" => 329.63, "sol" => 392.0, "si" => 493.88}
  @pairs [{"red", "do"}, {"green", "mi"}, {"blue", "sol"}, {"yellow", "si"}]

  def colours, do: ~w(red green blue yellow)
  def notes, do: ~w(do mi sol si)
  def note_of(colour), do: @pairs |> List.keyfind(colour, 0) |> elem(1)
  def colour_of(note), do: @pairs |> List.keyfind(note, 1) |> elem(0)
  def hz(note), do: Map.fetch!(@notes, note)
  def rgb(colour), do: Map.fetch!(@colours, colour)

  @doc "The word vocabulary (`Vapor.Quality.Planted.words/1`)."
  def vocab, do: Vapor.Quality.Planted.words(colours() ++ notes() ++ ["."])

  @doc "Every (left, right) colour pair."
  def all_pairs, do: for(a <- colours(), b <- colours(), do: {a, b})

  @doc "The held-out pairs: each colour once on each side, never in a fit."
  def held_out, do: [{"red", "green"}, {"green", "blue"}, {"blue", "yellow"}, {"yellow", "red"}]
  def train_pairs, do: all_pairs() -- held_out()

  @doc """
  The scene of a caption. `seed` 0 is the canonical rendering; any other
  seed is a *variant*: disc centres jittered by up to ±1.5 px, radii by
  ±0.6 px, global illumination by ±8 %, and sensor noise σ = 0.01 — so no
  two renderings share a patch, and a fit cannot succeed by recall.
  """
  def scene({left, right}, seed \\ 0, size \\ 16) do
    [jx1, jy1, jx2, jy2, jr1, jr2, jl] = if seed == 0, do: List.duplicate(0.5, 7), else: Vapor.Modal.Rng.uniform(seed * 7_919 + 3, 7)
    j = fn u, a -> (2 * u - 1) * a end
    gain = 1.0 + j.(jl, 0.08)
    r = size * 0.2
    col = fn name -> Enum.map(rgb(name), &min(1.0, &1 * gain)) end

    img =
      Image.scene(size, size, bg: {Enum.map([0.55, 0.55, 0.6], &(&1 * gain)), Enum.map([0.35, 0.35, 0.4], &(&1 * gain))},
                  shapes: [{:disc, size * 0.27 + j.(jx1, 1.5), size * 0.5 + j.(jy1, 1.5), r + j.(jr1, 0.6), col.(left)},
                           {:disc, size * 0.73 + j.(jx2, 1.5), size * 0.5 + j.(jy2, 1.5), r + j.(jr2, 0.6), col.(right)}])

    if seed == 0, do: img, else: noisy(img, 0.01, seed * 31 + 1)
  end

  @doc "A note as sound: `n` samples at `rate`, seeded phase (0 … 2π) and amplitude (0.3 … 0.6)."
  def sound(note, seed \\ 0, n \\ 1024, rate \\ 8000) do
    [u1, u2] = Vapor.Modal.Rng.uniform(seed + 17, 2)
    {ph, amp} = {2 * :math.pi() * u1, 0.3 + 0.3 * u2}
    f = hz(note)
    %Audio{rate: rate, samples: for(i <- 0..(n - 1), do: amp * Vapor.CR.sin_f64(2 * :math.pi() * f * i / rate + ph))}
  end

  @doc "The reference rendering of a note (phase 0, amplitude 0.5)."
  def reference_sound(note, n \\ 1024, rate \\ 8000),
    do: %Audio{rate: rate, samples: for(i <- 0..(n - 1), do: 0.5 * Vapor.CR.sin_f64(2 * :math.pi() * hz(note) * i / rate))}

  @doc "Gaussian noise added to an image (σ), clamped to [0, 1]."
  def noisy(%Image{} = img, sigma, seed) do
    vals = Enum.zip_with(Image.values(img), Vapor.Modal.Rng.normal(seed, img.w * img.h * img.c), fn v, z -> min(1.0, max(0.0, v + sigma * z)) end)
    Image.new(img.w, img.h, img.c, vals)
  end

  @doc "White noise added to a sound at `snr_db`."
  def noisy(%Audio{samples: s} = a, snr_db, seed, :snr) do
    p = Enum.reduce(s, 0.0, &(&1 * &1 + &2)) / length(s)
    sigma = :math.sqrt(p / :math.pow(10, snr_db / 10))
    %{a | samples: Enum.zip_with(s, Vapor.Modal.Rng.normal(seed, length(s)), fn x, z -> x + sigma * z end)}
  end
end
