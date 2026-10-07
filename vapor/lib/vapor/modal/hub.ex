defmodule Vapor.Modal.Hub do
  @moduledoc """
  **Any-to-any as a hub, not a matrix.**

  `N` modalities joined pairwise need `N·(N − 1)` converters. Joined through
  a pivot they need `N` codecs: each modality says how to reach the pivot
  and how to come back (`to_pivot`, `from_pivot`), and every route
  `a → b` is `from_pivot_b ∘ translate ∘ to_pivot_a`. Adding a modality is
  one codec; it then talks to all the others. Direct routes (an image
  codec's own round trip, a soft-token injection) are registered next to
  the pivot routes and preferred when present.

  Here the pivot is **text** — words a decoder reads and writes — because
  that is where the most capable models live; every step is a program
  built through the model airlock (`Vapor.Lock`) and run on a substrate
  (`Vapor.Modal.Runner`): projectors (`vapor_linear`), codecs (`vapor_vq`),
  the spectrum and synthesis programs, decoders (`:causal_lm`).

  `fit_world/1` fits every codec of `Vapor.Modal.World` in closed form
  (ridge bridges, k-means codebooks, counted bigram translators) on the
  training pairs only; `convert/4` routes. The same hub takes real
  checkpoints: register a modality whose `to_pivot` runs a ViT through
  `Vapor.Lock.Adapters.Encoder` and a projector, and nothing else changes.
  """
  alias Vapor.{Lock, Tensor}
  alias Vapor.Modal.{Audio, Bridge, Image, Runner, Text, VQ, World}
  alias Vapor.Quality.Planted

  defstruct mods: %{}, direct: %{}, translators: %{}, models: %{}, opts: []

  @doc "Register a modality: `%{concept: atom, to_pivot: (x -> [word]), from_pivot: ([word] -> x)}`."
  def register(%__MODULE__{} = h, modality, codec), do: %{h | mods: Map.put(h.mods, modality, codec)}

  @doc "Register a direct route `from → to` (preferred over the pivot)."
  def route(%__MODULE__{} = h, from, to, fun), do: %{h | direct: Map.put(h.direct, {from, to}, fun)}

  @doc "Register a translator between pivot concepts (`:colour → :note`, …)."
  def translator(%__MODULE__{} = h, from, to, fun), do: %{h | translators: Map.put(h.translators, {from, to}, fun)}

  @doc "Every modality pair the hub can route."
  def routes(%__MODULE__{mods: m} = h) do
    for a <- Map.keys(m), b <- Map.keys(m), routable?(h, a, b), do: {a, b}
  end

  defp routable?(h, a, b),
    do: Map.has_key?(h.direct, {a, b}) or (h.mods[a].concept == h.mods[b].concept or Map.has_key?(h.translators, {h.mods[a].concept, h.mods[b].concept}))

  @doc "Convert `x` from modality `from` to modality `to`: `{:ok, y, route}`."
  def convert(%__MODULE__{} = h, from, to, x) do
    cond do
      f = h.direct[{from, to}] ->
        {:ok, f.(x), [:direct]}

      Map.has_key?(h.mods, from) and Map.has_key?(h.mods, to) and routable?(h, from, to) ->
        words = h.mods[from].to_pivot.(x)
        {ca, cb} = {h.mods[from].concept, h.mods[to].concept}
        words = if ca == cb, do: words, else: h.translators[{ca, cb}].(words)
        {:ok, h.mods[to].from_pivot.(words), [from, :pivot, to]}

      true ->
        {:error, Vapor.Rejection.new({:hub, from, to}, "a route #{from} → #{to}", "register a codec or a direct route")}
    end
  end

  # ---------------------------------------------------------------- the world --

  @doc """
  Fit every codec of `Vapor.Modal.World` (training pairs only). Options:
  `worker` (a native worker; default: the oracle). Returns the hub; fitted
  models are in `hub.models` (each `{spec, weights}`, admitted).
  """
  def fit_world(opts \\ []) do
    v = World.vocab()
    run = fn p, env -> Runner.run(p, env, opts) end

    # text: bigram translators colour → note and note → colour (counted)
    c2n = Enum.flat_map(1..8, fn _ -> Enum.flat_map(World.colours(), &[v.id.(&1), v.id.(World.note_of(&1)), v.id.(".")]) end)
    n2c = Enum.flat_map(1..8, fn _ -> Enum.flat_map(World.notes(), &[v.id.(&1), v.id.(World.colour_of(&1)), v.id.(".")]) end)
    {:ok, ts, tw} = planted(c2n, v.size)
    {:ok, rs, rw} = planted(n2c, v.size)

    # the text encoder: the decoder's final hidden rows (no vocabulary head)
    {:ok, enc} = Lock.build(ts, tw, max_seq: 16, hidden: true, head: false)

    hidden = fn words ->
      ids = Enum.map(words, v.id)
      n = length(ids)
      env = Map.merge(Lock.zero_state(enc), %{tok: Tensor.from_list(:s32, [n], ids), pos: Tensor.from_list(:s32, [n], Enum.to_list(0..(n - 1)))})
      run.(enc, env).hidden |> Tensor.to_floats()
    end

    # image ↔ text: bridges from the caption's hidden rows to pixels, and
    # from pixels to one-hot slots (left colour, right colour)
    # every training pair in several variants (seeds 1…k): no two images share a patch
    k = Keyword.get(opts, :variants, 4)
    train = for pair <- World.train_pairs(), seed <- 1..k, do: {pair, seed}
    imgs = Enum.map(train, fn {pair, seed} -> pair |> World.scene(seed) |> Image.values() end)
    caps = Enum.map(train, fn {{a, b}, _} -> hidden.([a, b]) end)
    {:ok, t2i, t2i_w} = Bridge.fit(caps, imgs, ridge: 1.0e-6, from: "text", to: "image")
    slots = Enum.map(train, fn {{a, b}, _} -> onehot(a, World.colours()) ++ onehot(b, World.colours()) end)
    {:ok, i2t, i2t_w} = Bridge.fit(imgs, slots, ridge: 1.0e-3, from: "image", to: "text")

    # image → soft token: pixels → the embedding row of the left colour,
    # injected into the colour → note decoder (LLaVA's projector)
    emb = fn w -> for(j <- 0..(ts.width - 1), do: if(j == v.id.(w), do: 1.0, else: 0.0)) end
    {:ok, i2s, i2s_w} = Bridge.fit(imgs, Enum.map(train, fn {{a, _}, _} -> emb.(a) end), ridge: 1.0e-3, from: "image", to: "text")

    # audio ↔ text: spectrum features → note slots; note → oscillator amplitudes
    spec_p = Audio.spectrum_program(256, 4, 128)
    feats = fn %Audio{} = a ->
      p = run.(spec_p, %{rows: Audio.frames(a, 256, 256)}).power |> Tensor.to_floats() |> Enum.chunk_every(128)
      p |> Enum.zip_with(& &1) |> Enum.map(fn col -> :math.log(1.0e-6 + Enum.sum(col) / length(col)) end)
    end

    sounds = for note <- World.notes(), seed <- 1..6, do: {note, World.sound(note, seed)}
    {:ok, a2t, a2t_w} = Bridge.fit(Enum.map(sounds, &feats.(elem(&1, 1))), Enum.map(sounds, &onehot(elem(&1, 0), World.notes())),
                                   ridge: 1.0e-2, from: "audio", to: "text")

    freqs = Enum.map(World.notes(), &World.hz/1)
    synth = Audio.sinusoids_program(freqs, 8000, 1024, 1)
    note_rows = for n <- World.notes(), do: hidden.([n])
    amps = for n <- World.notes(), do: Enum.map(World.notes(), &if(&1 == n, do: 0.5, else: 0.0))
    {:ok, t2a, t2a_w} = Bridge.fit(note_rows, amps, ridge: 1.0e-6, from: "text", to: "audio")

    # direct codecs: VQ over image patches and over audio frames
    patches = train |> Enum.map(fn {pair, seed} -> World.scene(pair, seed) end) |> Enum.flat_map(&(&1 |> Image.patches(4) |> Tensor.to_floats() |> Enum.chunk_every(48)))
    {:ok, ivq, ivq_w} = VQ.fit(patches, 48, modality: "image", iters: 8)
    frames = sounds |> Enum.flat_map(fn {_, s} -> s |> Audio.frames(64, 64) |> Tensor.to_floats() |> Enum.chunk_every(64) end)
    {:ok, avq, avq_w} = VQ.fit(frames, 64, modality: "audio", iters: 8)

    m = %{translate_c2n: {ts, tw}, translate_n2c: {rs, rw}, text_to_image: {t2i, t2i_w}, image_to_text: {i2t, i2t_w},
          image_to_soft: {i2s, i2s_w}, audio_to_text: {a2t, a2t_w}, text_to_audio: {t2a, t2a_w},
          image_vq: {ivq, ivq_w}, audio_vq: {avq, avq_w}}

    apply_map = fn {spec, w}, rows ->
      t = length(rows)
      k = spec.in_width
      {:ok, p} = Lock.build(spec, w, rows: t)
      run.(p, %{rows: Tensor.from_list(:f32, [t, k], List.flatten(Bridge.pad(rows, k)))}).out |> Tensor.to_floats() |> Enum.chunk_every(spec.width)
    end

    argmax_slot = fn row, names -> names |> Enum.zip(row) |> Enum.max_by(&elem(&1, 1)) |> elem(0) end

    translate = fn {spec, w}, words ->
      Enum.map(words, fn word ->
        [id] = Text.generate(spec, w, [v.id.(word)], 1, opts)
        v.word.(id)
      end)
    end

    vq_round = fn {spec, w}, rows ->
      t = length(rows)
      {:ok, e} = Lock.build(spec, w, direction: :encode, rows: t)
      {:ok, d} = Lock.build(spec, w, direction: :decode, rows: t)
      codes = run.(e, %{rows: Tensor.from_list(:f32, [t, spec.width], List.flatten(rows))}).codes
      {Tensor.to_list(codes), run.(d, %{codes: codes}).rows}
    end

    hub =
      %__MODULE__{models: m, opts: opts}
      |> register(:text_colour, %{concept: :colour, to_pivot: & &1, from_pivot: & &1})
      |> register(:text_note, %{concept: :note, to_pivot: & &1, from_pivot: & &1})
      |> register(:image, %{concept: :colour,
           to_pivot: fn %Image{} = img ->
             [row] = apply_map.(m.image_to_text, [Image.values(img)])
             {l, r} = Enum.split(row, 4)
             [argmax_slot.(l, World.colours()), argmax_slot.(r, World.colours())]
           end,
           from_pivot: fn words ->
             words = case words do
               [a] -> [a, a]
               [a, b | _] -> [a, b]
             end
             [row] = apply_map.(m.text_to_image, [hidden.(words)])
             Image.new(16, 16, 3, Enum.map(row, &min(1.0, max(0.0, &1))))
           end})
      |> register(:audio, %{concept: :note,
           to_pivot: fn %Audio{} = a ->
             [row] = apply_map.(m.audio_to_text, [feats.(a)])
             [argmax_slot.(row, World.notes())]
           end,
           from_pivot: fn [note | _] ->
             [amps] = apply_map.(m.text_to_audio, [hidden.([note])])
             padded = amps ++ List.duplicate(0.0, 16 - length(amps))
             rows = run.(synth, %{amps: Tensor.from_list(:f32, [1, 16], padded)}).rows
             Audio.from_frames(rows, 8000)
           end})
      |> translator(:colour, :note, &translate.(m.translate_c2n, &1))
      |> translator(:note, :colour, &translate.(m.translate_n2c, &1))
      |> route(:image, :image, fn %Image{} = img ->
           rows = img |> Image.patches(4) |> Tensor.to_floats() |> Enum.chunk_every(48)
           {_codes, back} = vq_round.(m.image_vq, rows)
           Image.from_patches(back, 16, 16, 3, 4)
         end)
      |> route(:audio, :audio, fn %Audio{} = a ->
           rows = a |> Audio.frames(64, 64) |> Tensor.to_floats() |> Enum.chunk_every(64)
           {_codes, back} = vq_round.(m.audio_vq, rows)
           %{Audio.from_frames(back, a.rate) | samples: Enum.take(Tensor.to_floats(back), length(a.samples))}
         end)

    # the injection route: image → projector → soft token → decoder → note
    {:ok, inj} = Lock.build(ts, tw, max_seq: 4, logits: :last, inject: true)

    route(hub, :image, :text_note_soft, fn %Image{} = img ->
      [row] = apply_map.(m.image_to_soft, [Image.values(img)])
      env = Map.merge(Lock.zero_state(inj), %{tok: Tensor.from_list(:s32, [1], [0]), pos: Tensor.from_list(:s32, [1], [0]),
                                              last: Tensor.from_list(:s32, [1], [0]),
                                              soft: Tensor.from_list(:f32, [1, ts.width], row), soft_mask: Tensor.from_list(:f32, [1, 1], [1.0])})
      [v.word.(Vapor.Sampler.argmax(run.(inj, env).logits.data))]
    end)
  end

  defp planted(ids, v) do
    m = Planted.bigram(ids, v, alpha: 0.01)
    Lock.from_map(m.config, m.weights)
  end

  defp onehot(x, names), do: Enum.map(names, &if(&1 == x, do: 1.0, else: 0.0))
end
