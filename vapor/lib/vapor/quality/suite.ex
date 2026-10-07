defmodule Vapor.Quality.Suite do
  @moduledoc """
  The quality benchmark: every output vapor can produce, scored against a
  ground truth **and** against a control that a noise generator would
  match — so a pass means "signal", and a fail names the gap.

  Sections of `run/1`:

    * `gates` — the calibrated noise gates (text, image, audio) with the
      statistics of their controls (`Vapor.Quality.Gate`); a gate that
      cannot separate its controls aborts the suite;
    * `text` — a planted bigram of Portuguese prose
      (`Vapor.Quality.Planted`) through airlock → compiler → substrate:
      agreement with the analytic table, bits per character against the
      unigram and uniform baselines, the gate's verdict on its generations
      and on those of the same model with random weights (which must fail);
    * `any_to_any` — every route of the world hub (`Vapor.Modal.Hub`) on
      held-out inputs, with its metric, its control and its threshold;
    * `merge` — fusing a Portuguese and an English specialist, per method;
    * `substrates` — when a native worker is present, every modal program
      on it against the oracle, bit for bit.

  Each check is `%{name, value, control, threshold, pass}`; `passed?/1`
  is the conjunction. `mix vapor.quality` writes the report to
  `docs/bench/QUALITY.md` and `docs/bench/quality.json`.
  """
  alias Vapor.{Lock, Merge, Tensor}
  alias Vapor.Modal.{Audio, Hub, Image, Runner, World}
  alias Vapor.Quality.{Gate, Planted, Signal, Text}

  @doc "Run everything. Options: `worker` (default: a native worker if the host has one), `oracle: true` (force the oracle)."
  def run(opts \\ []) do
    w = if opts[:oracle], do: nil, else: Keyword.get_lazy(opts, :worker, &Runner.worker/0)
    ro = [worker: w]
    t0 = System.monotonic_time(:millisecond)

    {gates, gt} = timed(fn -> gates() end)
    {text, tt} = timed(fn -> text(gates.text, ro) end)
    {hub, ht} = timed(fn -> Hub.fit_world(ro) end)
    {a2a, at} = timed(fn -> any_to_any(hub, gates) end)
    {merge, mt} = timed(fn -> merge(ro) end)
    {merge_real, mrt} = timed(fn -> merge_real(ro) end)
    {real, rt} = timed(fn -> real_data(ro) end)
    {subs, st} = timed(fn -> substrates(hub, w) end)
    {lock, lt} = timed(fn -> lock_bench() end)
    {round06, r6t} = timed(fn -> Vapor.Quality.Round06.run(ro) end)
    {round07, r7t} = timed(fn -> Vapor.Quality.Round07.run(ro) end)
    {round08, r8t} = timed(fn -> Vapor.Quality.Round08.run(ro) end)
    {round09, r9t} = timed(fn -> Vapor.Quality.Round09.run(ro) end)
    {round10, r10t} = timed(fn -> Vapor.Quality.Round10.run(ro) end)
    {round11, r11t} = timed(fn -> Vapor.Quality.Round11.run(ro) end)
    {round12, r12t} = timed(fn -> Vapor.Quality.Round12.run(ro) end)
    {round13, r13t} = timed(fn -> Vapor.Quality.Round13.run(ro) end)
    {round14, r14t} = timed(fn -> Vapor.Quality.Round14.run(ro) end)
    {round15, r15t} = timed(fn -> Vapor.Quality.Round15.run(ro) end)
    {round16, r16t} = timed(fn -> Vapor.Quality.Round16.run(ro) end)
    if dir = opts[:gallery], do: gallery(hub, dir)

    %{substrate: if(w, do: :native, else: :oracle),
      gates: summarize_gates(gates), text: text, any_to_any: a2a, merge: merge, merge_real: merge_real, real: real, substrates: subs, lock: lock, round06: round06, round07: round07, round08: round08, round09: round09, round10: round10, round11: round11, round12: round12, round13: round13, round14: round14, round15: round15, round16: round16,
      timings_ms: %{gates: gt, text: tt, hub_fit: ht, any_to_any: at, merge: mt, merge_real: mrt, real: rt, substrates: st, lock: lt, round06: r6t, round07: r7t, round08: r8t, round09: r9t, round10: r10t, round11: r11t, round12: r12t, round13: r13t, round14: r14t, round15: r15t, round16: r16t,
                    total: System.monotonic_time(:millisecond) - t0}}
  end

  @doc "Whether every check passed."
  def passed?(report) do
    checks(report) |> Enum.all?(& &1.pass)
  end

  @doc "Every check of a report, flattened."
  def checks(r),
    do: r.text.checks ++ Enum.flat_map(r.any_to_any, & &1.checks) ++ r.merge.checks ++ Map.get(r, :merge_real, %{checks: []}).checks ++
          Map.get(r, :real, %{checks: []}).checks ++ r.substrates.checks ++ Map.get(r, :round06, %{checks: []}).checks ++
          Map.get(r, :round07, %{checks: []}).checks ++ Map.get(r, :round08, %{checks: []}).checks ++
          Map.get(r, :round09, %{checks: []}).checks ++ Map.get(r, :round10, %{checks: []}).checks ++ Map.get(r, :round11, %{checks: []}).checks ++
          Map.get(r, :round12, %{checks: []}).checks ++ Map.get(r, :round13, %{checks: []}).checks ++
          Map.get(r, :round14, %{checks: []}).checks ++ Map.get(r, :round15, %{checks: []}).checks ++
          Map.get(r, :round16, %{checks: []}).checks

  defp timed(f) do
    t = System.monotonic_time(:millisecond)
    v = f.()
    {v, System.monotonic_time(:millisecond) - t}
  end

  defp check(name, value, control, threshold, pass), do: %{name: name, value: value, control: control, threshold: threshold, pass: pass}

  # ------------------------------------------------------------------ corpora --

  # The corpora are frozen in priv/quality (the repository's 0.3.0 docs in
  # Portuguese, the moduledocs of its sources in English): editing the docs
  # must not move the benchmark.
  @doc "Portuguese prose in the planted alphabet: `{train, holdout}`."
  def corpus_pt do
    {ref, hold} = corpus_pt_raw()
    {norm(ref), norm(hold)}
  end

  @doc "Portuguese prose as written (bytes): `{reference, holdout}`."
  def corpus_pt_raw, do: {corpus("pt_reference.txt"), corpus("pt_holdout.txt")}

  @doc "English prose (vapor's own moduledocs) in the planted alphabet: `{train, holdout}`."
  def corpus_en, do: {norm(corpus("en_reference.txt")), norm(corpus("en_holdout.txt"))}

  @doc "A frozen corpus file of `priv/quality`."
  def corpus(name), do: File.read!(Path.join([to_string(:code.priv_dir(:vapor)), "quality", name]))

  defp norm(t) do
    a = Planted.alphabet()
    a.decode.(a.encode.(t))
  end

  # -------------------------------------------------------------------- gates --

  @doc "The three calibrated gates."
  def gates do
    {train, hold} = corpus_pt()
    profile = Text.profile(train)
    {:ok, tg} = Text.gate(profile, hold, len: 200, count: 24)

    pos_img = for pair <- World.all_pairs(), seed <- [1, 2], do: World.scene(pair, seed)
    neg_img = %{noise: for(s <- 1..12, do: Image.noise(16, 16, 3, s)), shuffled: for({img, s} <- Enum.with_index(Enum.take(pos_img, 12)), do: Image.shuffle(img, s + 1))}
    {:ok, ig} = Gate.calibrate(:image_noise, Map.new(neg_img, fn {k, v} -> {k, Enum.map(v, &Signal.neighbour_corr/1)} end),
                               %{scenes: Enum.map(pos_img, &Signal.neighbour_corr/1)})

    tones = for n <- World.notes(), s <- 1..3, do: World.sound(n, s)
    chords = for s <- 1..4, do: Audio.chord([261.63, 329.63, 392.0], 8000, 1024, 0.4 + 0.02 * s)
    neg_aud = %{noise: for(s <- 1..12, do: Audio.noise(8000, 1024, s)), shuffled: for({t, s} <- Enum.with_index(tones), do: Audio.shuffle(t, s + 1))}
    {:ok, ag} = Gate.calibrate(:audio_noise, Map.new(neg_aud, fn {k, v} -> {k, Enum.map(v, &Signal.flatness/1)} end),
                               %{tones: Enum.map(tones, &Signal.flatness/1), chords: Enum.map(chords, &Signal.flatness/1)}, :down)

    %{text: %{profile: profile, gate: tg, holdout: hold}, image: ig, audio: ag}
  end

  defp summarize_gates(g) do
    %{text_noise: gate_summary(g.text.gate.noise), text_collapse: gate_summary(g.text.gate.collapse),
      image_noise: gate_summary(g.image), audio_noise: gate_summary(g.audio)}
  end

  defp gate_summary(%Gate{} = g), do: Map.take(g, [:name, :direction, :t_noise, :t_natural, :margin, :controls])

  # --------------------------------------------------------------------- text --

  defp text(%{profile: profile, gate: g, holdout: hold}, ro) do
    a = Planted.alphabet()
    {train, _} = corpus_pt()
    m = Planted.bigram(a.encode.(train), a.size)
    {:ok, spec, ws} = Lock.from_map(m.config, m.weights)

    ids = a.encode.(binary_part(hold, 2000, 300))
    rows = Vapor.Modal.Text.logits(spec, ws, ids, ro)

    table_err =
      rows |> Enum.zip(ids) |> Enum.map(fn {row, i} ->
        row |> Enum.take(a.size) |> Enum.zip(m.logp |> Enum.at(i) |> Enum.take(a.size)) |> Enum.map(fn {x, y} -> abs(x - y) end) |> Enum.max()
      end) |> Enum.max()

    bits = Text.bits_per_token(fn p -> Text.log_softmax(Enum.at(rows, length(p) - 1)) end, ids, a.size).bits
    unigram = Text.unigram_bits(a.decode.(ids), profile)

    gen = fn spec, ws, seed -> spec |> Vapor.Modal.Text.generate(ws, a.encode.("o motor "), 220, Keyword.merge(ro, temperature: 1.0, seed: seed)) |> a.decode.() end
    planted = for s <- 1..4, do: Text.judge(gen.(spec, ws, s), profile, g)
    rw = Map.new(m.weights, fn {k, t} -> {k, Tensor.random(:f32, t.shape, :erlang.phash2(k))} end)
    {:ok, rspec, rws} = Lock.from_map(m.config, rw)
    random = for s <- 1..4, do: Text.judge(gen.(rspec, rws, s), profile, g)
    sample = gen.(spec, ws, 1)

    %{bits_per_char: bits, table_bits: Planted.table_bits(m, ids), unigram_bits: unigram, uniform_bits: :math.log2(a.size),
      max_table_error: table_err, sample: binary_part(sample, 0, min(160, byte_size(sample))),
      planted_verdicts: Enum.map(planted, &{&1.verdict, &1.noise}), random_verdicts: Enum.map(random, &{&1.verdict, &1.noise}),
      checks: [
        check("text: logits = analytic bigram table (max |Δ|)", table_err, nil, 1.0e-5, table_err <= 1.0e-5),
        check("text: bits/char below the unigram baseline", bits, unigram, "< unigram − 0.5", bits < unigram - 0.5),
        check("text: planted generations pass the noise gate", Enum.count(planted, &(&1.verdict == :pass)), length(planted), "all", Enum.all?(planted, &(&1.verdict == :pass))),
        check("text: random-weight generations fail the noise gate (control)", Enum.count(random, &(&1.verdict == :fail)), length(random), "all", Enum.all?(random, &(&1.verdict == :fail)))
      ]}
  end

  # --------------------------------------------------------------- any-to-any --

  defp any_to_any(hub, gates) do
    held = World.held_out()
    conv = fn from, to, x -> {:ok, y, route} = Hub.convert(hub, from, to, x); {y, route} end
    acc = fn hits -> Enum.count(hits, & &1) / length(hits) end
    img_sig = fn img -> Gate.judge(gates.image, Signal.neighbour_corr(img)) end
    aud_sig = fn a -> Gate.judge(gates.audio, Signal.flatness(a)) end

    # image → text (captions of unseen pairs, unseen variants, plus sensor noise σ = 0.05)
    inputs = for pair <- held, seed <- 101..103, noise <- [0.0, 0.05], do: {pair, World.scene(pair, seed) |> then(&if(noise > 0, do: World.noisy(&1, noise, seed), else: &1))}
    i2t = Enum.map(inputs, fn {{a, b}, img} -> conv.(:image, :text_colour, img) |> elem(0) == [a, b] end)

    # text → image (and the control: the image of a *different* caption)
    t2i = for {a, b} <- held do
      {img, _} = conv.(:text_colour, :image, [a, b])
      truth = World.scene({a, b})
      {other, _} = conv.(:text_colour, :image, [b, a])
      {Signal.psnr(truth, img), Signal.psnr(truth, other), img_sig.(img)}
    end

    # image → image (VQ round trip of unseen variants) and its random-codebook control
    i2i = for pair <- held, seed <- [201, 202] do
      img = World.scene(pair, seed)
      {back, _} = conv.(:image, :image, img)
      {Signal.psnr(img, back), img_sig.(back), img}
    end

    {rspec, _} = hub.models.image_vq
    rand_book = %{"codebook" => Tensor.random(:f32, [rspec.vocab, rspec.width], 77, scale: 0.5) |> then(&Tensor.from_list(:f32, &1.shape, Enum.map(Tensor.to_floats(&1), fn x -> x + 0.5 end)))}
    i2i_ctrl = Enum.map(i2i, fn {_, _, img} -> Signal.psnr(img, vq_round(rspec, rand_book, img)) end)

    # audio → text (unseen phases and amplitudes, at 10 dB SNR)
    a2t = for note <- World.notes(), seed <- 100..103, do: conv.(:audio, :text_note, World.noisy(World.sound(note, seed), 10, seed, :snr)) |> elem(0) == [note]

    # text → audio
    t2a = for note <- World.notes() do
      {snd, _} = conv.(:text_note, :audio, [note])
      hz = Signal.dominant_hz(snd)
      {abs(hz - World.hz(note)) / World.hz(note), aud_sig.(snd), Signal.snr(World.reference_sound(note), snd)}
    end

    # audio → audio (VQ round trip) and its control (white noise of equal power through nothing: SNR 0)
    a2a = for note <- World.notes(), seed <- [300, 301] do
      s = World.sound(note, seed)
      {back, _} = conv.(:audio, :audio, s)
      {Signal.snr(s, back), aud_sig.(back)}
    end

    # translations and composites through the pivot
    c2n = for c <- World.colours(), do: conv.(:text_colour, :text_note, [c]) |> elem(0) == [World.note_of(c)]
    n2c = for n <- World.notes(), do: conv.(:text_note, :text_colour, [n]) |> elem(0) == [World.colour_of(n)]
    i2a = for {a, b} <- held do
      {snd, _} = conv.(:image, :audio, World.scene({a, b}, 400))
      abs(Signal.dominant_hz(snd) - World.hz(World.note_of(a))) / World.hz(World.note_of(a)) < 0.03
    end
    a2i = for note <- World.notes() do
      {img, _} = conv.(:audio, :image, World.sound(note, 500))
      c = World.colour_of(note)
      {Signal.psnr(World.scene({c, c}), img), img_sig.(img)}
    end
    soft = for {a, b} <- held, seed <- [601, 602], do: conv.(:image, :text_note_soft, World.scene({a, b}, seed)) |> elem(0) == [World.note_of(a)]

    mean = fn xs -> Enum.sum(xs) / length(xs) end
    sig_all = fn vs -> Enum.all?(vs, &Gate.signal?/1) end

    [
      route("image → text", "caption accuracy (unseen pairs, variants, σ=0.05 noise)", [
        check("image→text accuracy", acc.(i2t), 0.25, "≥ 0.95 (chance 0.25 per slot)", acc.(i2t) >= 0.95)]),
      route("text → image", "PSNR vs the canonical scene; control: the swapped caption", [
        check("text→image PSNR (dB)", mean.(Enum.map(t2i, &elem(&1, 0))), mean.(Enum.map(t2i, &elem(&1, 1))), "≥ 18 and ≥ control + 3", mean.(Enum.map(t2i, &elem(&1, 0))) >= 18 and mean.(Enum.map(t2i, &elem(&1, 0))) >= mean.(Enum.map(t2i, &elem(&1, 1))) + 3),
        check("text→image outputs pass the image gate", Enum.map(t2i, &elem(&1, 2)), nil, "signal", sig_all.(Enum.map(t2i, &elem(&1, 2))))]),
      route("image → image", "VQ round trip PSNR on unseen variants; control: random codebook", [
        check("image→image PSNR (dB)", mean.(Enum.map(i2i, &elem(&1, 0))), mean.(i2i_ctrl), "≥ 20 and ≥ control + 6", mean.(Enum.map(i2i, &elem(&1, 0))) >= 20 and mean.(Enum.map(i2i, &elem(&1, 0))) >= mean.(i2i_ctrl) + 6),
        check("image→image outputs pass the image gate", Enum.map(i2i, &elem(&1, 1)), nil, "signal", sig_all.(Enum.map(i2i, &elem(&1, 1))))]),
      route("audio → text", "note accuracy (unseen phase/amplitude, 10 dB SNR)", [
        check("audio→text accuracy", acc.(a2t), 0.25, "≥ 0.95 (chance 0.25)", acc.(a2t) >= 0.95)]),
      route("text → audio", "pitch error, audio gate, SNR vs the reference tone", [
        check("text→audio max pitch error", Enum.max(Enum.map(t2a, &elem(&1, 0))), nil, "≤ 3 %", Enum.max(Enum.map(t2a, &elem(&1, 0))) <= 0.03),
        check("text→audio outputs pass the audio gate", Enum.map(t2a, &elem(&1, 1)), nil, "signal", sig_all.(Enum.map(t2a, &elem(&1, 1)))),
        check("text→audio SNR vs reference (dB)", mean.(Enum.map(t2a, &elem(&1, 2))), 0.0, "≥ 20", mean.(Enum.map(t2a, &elem(&1, 2))) >= 20)]),
      route("audio → audio", "VQ round trip SNR on unseen phases", [
        check("audio→audio SNR (dB)", mean.(Enum.map(a2a, &elem(&1, 0))), 0.0, "≥ 10", mean.(Enum.map(a2a, &elem(&1, 0))) >= 10),
        check("audio→audio outputs pass the audio gate", Enum.map(a2a, &elem(&1, 1)), nil, "signal", sig_all.(Enum.map(a2a, &elem(&1, 1))))]),
      route("text → text", "translation exact match (colour ↔ note, counted bigram decoders)", [
        check("colour→note exact match", acc.(c2n), 0.25, "= 1", acc.(c2n) == 1.0),
        check("note→colour exact match", acc.(n2c), 0.25, "= 1", acc.(n2c) == 1.0)]),
      route("image → audio", "via the pivot: the note of the left colour, pitch within 3 %", [
        check("image→audio correct pitch", acc.(i2a), 0.25, "= 1", acc.(i2a) == 1.0)]),
      route("audio → image", "via the pivot: PSNR vs the scene of the note's colour", [
        check("audio→image PSNR (dB)", mean.(Enum.map(a2i, &elem(&1, 0))), nil, "≥ 18", mean.(Enum.map(a2i, &elem(&1, 0))) >= 18),
        check("audio→image outputs pass the image gate", Enum.map(a2i, &elem(&1, 1)), nil, "signal", sig_all.(Enum.map(a2i, &elem(&1, 1))))]),
      route("image → soft token → decoder", "projector output injected as a row of the decoder (inject: true)", [
        check("image→soft-token→note accuracy", acc.(soft), 0.25, "= 1", acc.(soft) == 1.0)])
    ]
  end

  defp route(name, what, checks), do: %{route: name, measures: what, checks: checks}

  defp vq_round(spec, ws, img) do
    rows = img |> Image.patches(4) |> Tensor.to_floats() |> Enum.chunk_every(48)
    t = length(rows)
    {:ok, e} = Lock.build(spec, ws, direction: :encode, rows: t)
    {:ok, d} = Lock.build(spec, ws, direction: :decode, rows: t)
    codes = Runner.run(e, %{rows: Tensor.from_list(:f32, [t, 48], List.flatten(rows))}).codes
    Image.from_patches(Runner.run(d, %{codes: codes}).rows, 16, 16, 3, 4)
  end

  # -------------------------------------------------------------------- merge --

  defp merge(ro) do
    a = Planted.alphabet()
    {pt, pt_ho} = corpus_pt()
    {en, en_ho} = corpus_en()
    mk = fn txt -> m = Planted.bigram(a.encode.(txt), a.size); {:ok, s, w} = Lock.from_map(m.config, m.weights); %{spec: s, weights: w} end
    {ma, mb, base} = {mk.(pt), mk.(en), mk.("")}

    bits = fn m, txt ->
      ids = a.encode.(binary_part(txt, 0, 400))
      rows = Vapor.Modal.Text.logits(m.spec, m.weights, ids, ro)
      Text.bits_per_token(fn p -> Text.log_softmax(Enum.at(rows, length(p) - 1)) end, ids, a.size).bits
    end

    score = fn m -> %{pt: bits.(m, pt_ho), en: bits.(m, en_ho)} end
    {sa, sb} = {score.(ma), score.(mb)}

    rows =
      for {name, opts} <- [linear: [method: :linear], slerp: [method: :slerp, t: 0.5],
                           task_arithmetic: [method: :task_arithmetic, base: base, weights: [0.5, 0.5]],
                           ties: [method: :ties, base: base, density: 0.5], dare_linear: [method: :dare_linear, base: base, density: 0.7, weights: [0.5, 0.5], seed: 3]] do
        {t, {:ok, mm}} = :timer.tc(fn -> Merge.merge([ma, mb], opts) end)
        s = score.(mm)
        %{method: name, pt: s.pt, en: s.en, mean: (s.pt + s.en) / 2, ms: div(t, 1000), receipt_ok: Merge.verify_receipt(mm.receipt, mm.weights) == :ok}
      end

    lin = Enum.find(rows, &(&1.method == :linear))
    best_single = min((sa.pt + sa.en) / 2, (sb.pt + sb.en) / 2)

    # identities that must hold exactly
    {:ok, same} = Merge.merge([ma, ma], method: :linear)
    {:ok, s0} = Merge.merge([ma, mb], method: :slerp, t: 0.0)

    %{specialists: %{pt: sa, en: sb}, methods: rows,
      checks: [
        check("merge: linear beats the wrong specialist on each domain", {lin.pt, lin.en}, {sb.pt, sa.en}, "pt < B@pt and en < A@en", lin.pt < sb.pt and lin.en < sa.en),
        check("merge: linear's mean bits beat both specialists'", lin.mean, best_single, "<", lin.mean < best_single),
        check("merge: merge(A, A) = A bit for bit", same.weights == ma.weights, true, "identity", Map.drop(same.weights, []) == ma.weights),
        check("merge: slerp(t = 0) = A bit for bit", s0.weights == ma.weights, true, "identity", s0.weights == ma.weights),
        check("merge: every receipt verifies", Enum.all?(rows, & &1.receipt_ok), true, "all", Enum.all?(rows, & &1.receipt_ok))
      ]}
  end

  # ------------------------------------------------- merge: trained transformers --

  @doc """
  Fusion of **trained** transformers (not planted): the character-level
  Llama decoders of `priv/quality/merge` (`test/python/train_merge_models.py`,
  PyTorch, frozen with their SHA-256): a base trained on mixed Portuguese
  and English, two fine-tunes of it (`ft_pt`, `ft_en`) and two models
  trained from scratch on one language each (`solo_pt`, `solo_en`, other
  initialisations). Bits per character on held-out text: candidates are
  *selected* on a validation slice and *reported* on a disjoint test slice.
  """
  def merge_real(ro) do
    a = Planted.alphabet()
    dir = Path.join([to_string(:code.priv_dir(:vapor)), "quality", "merge"])
    open = fn n -> {:ok, m} = Lock.open(Path.join(dir, n)); %{spec: m.spec, weights: m.weights} end
    ms = Map.new(~w(base ft_pt ft_en solo_pt solo_en), &{&1, open.(&1)})
    {pt_ref, pt_ho} = corpus_pt_raw()
    {en_ref, en_ho} = {corpus("en_reference.txt"), corpus("en_holdout.txt")}
    slice = fn txt, from -> txt |> a.encode.() |> Enum.drop(from) |> Enum.take(1024) end
    val = %{pt: slice.(pt_ho, 0), en: slice.(en_ho, 0)}
    test = %{pt: slice.(pt_ho, 2048), en: slice.(en_ho, 2048)}

    bits = fn m, ids ->
      ids
      |> Enum.chunk_every(256)
      |> Enum.map(fn ch ->
        rows = Vapor.Modal.Text.logits(m.spec, m.weights, ch, ro)
        {Text.bits_per_token(fn p -> Text.log_softmax(Enum.at(rows, length(p) - 1)) end, ch, 48).bits * (length(ch) - 1), length(ch) - 1}
      end)
      |> Enum.reduce({0.0, 0}, fn {b, n}, {sb, sn} -> {sb + b, sn + n} end)
      |> then(fn {b, n} -> b / n end)
    end

    score = fn m, set -> {p, e} = {bits.(m, set.pt), bits.(m, set.en)}; %{pt: p, en: e, mean: (p + e) / 2} end
    models = Map.new(ms, fn {k, m} -> {k, score.(m, test)} end)
    batches = fn txt -> txt |> a.encode.() |> Enum.chunk_every(128) |> Enum.take(16) end
    w = ro[:worker]

    run = fn pair_names, base ->
      [ma, mb] = Enum.map(pair_names, &ms[&1])
      diag = Merge.diagnose([ma, mb], base: base)
      {:ok, ga} = Merge.calibrate(ma, batches.(pt_ref), worker: w)
      {:ok, gb} = Merge.calibrate(mb, batches.(en_ref), worker: w)

      cands =
        [linear: [method: :linear], slerp: [method: :slerp], regmean: [method: :regmean, grams: [ga, gb]]] ++
          if(base, do: [task_arithmetic: [method: :task_arithmetic, base: base], "task_arithmetic λ=0.7": [method: :task_arithmetic, base: base, lambda: 0.7],
                        "ties 0.2": [method: :ties, base: base, density: 0.2], "ties 0.5": [method: :ties, base: base, density: 0.5],
                        "dare_linear 0.5": [method: :dare_linear, base: base, density: 0.5, seed: 1], "dare_ties 0.5": [method: :dare_ties, base: base, density: 0.5, seed: 1]],
                   else: [])

      {:ok, sel} = Merge.select([ma, mb], cands, fn m -> score.(m, val).mean end, eval: "pt_holdout[0:1024] + en_holdout[0:1024]")

      rows =
        for {label, mo} <- cands do
          {:ok, m} = Merge.merge([ma, mb], mo)
          v = Enum.find(sel.table, &(&1.label == to_string(label)))
          Map.merge(%{method: to_string(label), val: v.score, ms: v.ms}, score.(m, test))
        end

      %{diag: Map.take(diag, [:regime, :advice, :models, :pairs, :density]), rows: rows, chosen: sel.best,
        chosen_test: Enum.find(rows, &(&1.method == sel.best)), receipt_kind: sel.receipt.payload.kind}
    end

    ft = run.(["ft_pt", "ft_en"], ms["base"])
    solo = run.(["solo_pt", "solo_en"], nil)
    lin_ft = Enum.find(ft.rows, &(&1.method == "linear"))
    lin_solo = Enum.find(solo.rows, &(&1.method == "linear"))
    best_spec_ft = min(models["ft_pt"].mean, models["ft_en"].mean)
    best_test = ft.rows |> Enum.map(& &1.mean) |> Enum.min()

    %{models: models, fine_tune: ft, independent: solo,
      checks: [
        check("merge (trained): linear fusion of the fine-tunes beats both specialists and the base (mean bits/char, test)",
              lin_ft.mean, {best_spec_ft, models["base"].mean}, "< min(specialists, base)", lin_ft.mean < best_spec_ft and lin_ft.mean < models["base"].mean),
        check("merge (trained): diagnose calls fine-tunes of one base related, not :unrelated", ft.diag.regime, :small_deltas, "≠ :unrelated", ft.diag.regime != :unrelated),
        check("merge (trained): diagnose calls independently trained models :unrelated", solo.diag.regime, :unrelated, "= :unrelated", solo.diag.regime == :unrelated),
        check("merge (trained): …and it was right — their linear average is worse than either specialist",
              lin_solo.mean, min(models["solo_pt"].mean, models["solo_en"].mean), ">", lin_solo.mean > min(models["solo_pt"].mean, models["solo_en"].mean)),
        check("merge (trained): selection on validation lands within 0.02 bits of the best test score",
              ft.chosen_test.mean, best_test, "≤ best + 0.02", ft.chosen_test.mean <= best_test + 0.02)
      ]}
  end

  # ------------------------------------------------------------ real data --

  @doc """
  The routes on **real signals**, with models admitted through the airlock
  and data never seen in training (thresholds declared before the models
  were trained):

    * OCR — 40 lines in five typefaces absent from training, text from the
      held-out corpora; a real photograph of a printed page (scikit-image's
      `page`); control: a reader that answers fluent but wrong text (each
      line read as the previous line's truth); Tesseract alongside when
      installed (information, not a check);
    * speech — 100 recordings of a speaker absent from training (FSDD);
      control: chance, and the same clips reversed in time;
    * handwriting — the classifier on 497 held-out digits; 50 generated
      digits read back by it; their distance to the nearest training image
      against held-out real images' (a copy sits at ≈ 0 — the analytic
      denoiser over the training set shows it);
    * a chain — a held-out voice says a digit → text → a drawn digit → read back;
    * JPEG — the committed fixtures decode to Pillow's pixels.
  """
  def real_data(ro) do
    w = ro[:worker]
    priv = Path.join(to_string(:code.priv_dir(:vapor)), "quality")
    {ocr, ocr_checks} = real_ocr(Path.join(priv, "ocr"), w)
    {speech, speech_checks, clips} = real_speech(Path.join(priv, "speech"), w)
    {digits, digit_checks} = real_digits(w)
    {chain, chain_checks} = real_chain(clips, w)
    {jpeg, jpeg_checks} = real_jpeg()

    %{ocr: ocr, speech: speech, digits: digits, chain: chain, jpeg: jpeg,
      checks: ocr_checks ++ speech_checks ++ digit_checks ++ chain_checks ++ jpeg_checks}
  end

  defp real_ocr(dir, w) do
    alias Vapor.Vision.OCR
    {:ok, meta} = Vapor.JSON.decode(File.read!(Path.join(dir, "labels.json")))
    items = Enum.sort(meta["lines"])

    rows =
      for {file, %{"text" => text, "font" => font}} <- items do
        {:ok, pic} = Vapor.Docs.Pictures.read(:png, File.read!(Path.join(dir, file)))
        {:ok, r} = OCR.read(pic.image, worker: w)
        tess = OCR.tesseract(Path.join(dir, file))
        %{font: font, ref: String.trim(Regex.replace(~r/ +/, text, " ")), hyp: r.text, tess: tess}
      end

    cer = fn rs, key ->
      {e, n} = Enum.reduce(rs, {0, 0}, fn r, {e, n} -> {e + OCR.levenshtein(String.graphemes(Map.fetch!(r, key) || ""), String.graphemes(r.ref)), n + String.length(r.ref)} end)
      e / max(n, 1)
    end

    shifted = Enum.zip(rows, tl(rows) ++ [hd(rows)]) |> Enum.map(fn {a, b} -> %{a | hyp: b.ref} end)
    {all, control} = {cer.(rows, :hyp), cer.(shifted, :hyp)}
    fonts = rows |> Enum.group_by(& &1.font) |> Map.new(fn {f, rs} -> {f, cer.(rs, :hyp)} end)
    tess_all = if Enum.all?(rows, & &1.tess), do: cer.(rows, :tess)

    {:ok, gt} = Vapor.JSON.decode(File.read!(Path.join(dir, "page.json")))
    {:ok, pic} = Vapor.Docs.Pictures.read(:png, File.read!(Path.join(dir, "page.png")))
    {:ok, pr} = OCR.read(pic.image, worker: w)
    ref = Enum.join(gt["lines"], "\n")
    page_hyp = pr.lines |> Enum.take(length(gt["lines"])) |> Enum.map_join("\n", & &1.text)
    page_cer = OCR.cer(page_hyp, ref)
    page_tess = OCR.tesseract(Path.join(dir, "page.png"))
    page_tess_cer = page_tess && OCR.cer(page_tess |> String.split("\n", trim: true) |> Enum.take(length(gt["lines"])) |> Enum.join("\n"), ref)

    {%{lines: length(rows), cer: all, control_cer: control, per_font: fonts, tesseract_cer: tess_all,
       page: %{cer: page_cer, wer: OCR.wer(page_hyp, ref), text: page_hyp, tesseract_cer: page_tess_cer, tesseract_text: page_tess},
       samples: rows |> Enum.take(6) |> Enum.map(&Map.take(&1, [:font, :ref, :hyp]))},
     [check("OCR: CER on typefaces never seen in training", all, control, "≤ 0.10 (control: fluent wrong text)", all <= 0.10 and control > 0.5),
      check("OCR: CER on a real photographed page (uneven light)", page_cer, page_tess_cer, "≤ 0.15", page_cer <= 0.15)]}
  end

  defp real_speech(dir, w) do
    alias Vapor.Modal.{Audio, Speech}
    {:ok, model} = Speech.load()
    files = Path.wildcard(Path.join(dir, "*.wav")) |> Enum.sort()
    clips = for f <- files, do: {f |> Path.basename() |> String.split("_") |> hd(), elem(Audio.read(f), 1)}
    acc = fn cs -> Enum.count(cs, fn {d, clip} -> {:ok, r} = Speech.classify(clip, model, worker: w); r.label == d end) / length(cs) end
    a = acc.(clips)
    rev = acc.(Enum.map(clips, fn {d, c} -> {d, %{c | samples: Enum.reverse(c.samples)}} end))

    {%{clips: length(clips), accuracy: a, reversed_accuracy: rev, chance: 0.1, speaker: "theo (held out)",
       model: model.spec.config.raw["training"]},
     [check("speech: spoken-digit accuracy on a voice never heard", a, 0.1, "≥ 0.85 (chance 0.10)", a >= 0.85)], clips}
  end

  defp real_digits(w) do
    alias Vapor.Modal.{Diffusion, Digits}
    {:ok, m} = Digits.models()
    test = Digits.split(m, :test)
    readings = Digits.read(Enum.map(test, &elem(&1, 0)), worker: w)
    acc = Enum.zip(readings, test) |> Enum.count(fn {r, {_, l}} -> r.digit == l end) |> Kernel./(length(test))

    gen = for d <- 0..9, do: {d, Digits.draw(d, 5, seed: 100 + d, worker: w)}
    read_back = gen |> Enum.flat_map(fn {d, r} -> Enum.map(r.readings, &(&1.digit == d)) end)
    gen_acc = Enum.count(read_back, & &1) / length(read_back)
    median = fn xs -> xs |> Enum.sort() |> Enum.at(div(length(xs), 2)) end
    gen_near = gen |> Enum.flat_map(fn {_, r} -> r.nearest end) |> median.()
    train = m |> Digits.split(:train) |> Enum.map(&elem(&1, 0))
    real_near = test |> Enum.take(100) |> Enum.map(&elem(&1, 0)) |> Diffusion.nearest(train) |> median.()

    # the control: the optimal denoiser of the training set itself (attention
    # over its 1 300 images) — what perfect memorisation looks like
    to_unit = fn img -> Enum.map(img, &(&1 / 8 - 1)) end
    comps = train |> Enum.map(&{1.0, to_unit.(&1), 0.0})
    copies = Diffusion.sample(m.denoiser, nil, 6, seed: 7, steps: 25, denoiser: Diffusion.analytic(m.denoiser, %{nil => comps}))
    copy_near = copies.samples |> Enum.map(fn x -> Enum.map(x, &((&1 + 1) * 8)) end) |> Diffusion.nearest(train) |> median.()

    {%{held_out: length(test), accuracy: acc, generated: length(read_back), read_back_accuracy: gen_acc,
       nearest_train: %{generated: gen_near, held_out_real: real_near, memorising_control: copy_near},
       gallery: Enum.map(gen, fn {d, r} -> {d, hd(r.images)} end)},
     [check("handwriting: image → digit accuracy on 497 held-out real digits", acc, 0.1, "≥ 0.95 (chance 0.10)", acc >= 0.95),
      check("handwriting: digit → image, generated digits read back by the real-data classifier", gen_acc, 0.1, "≥ 0.80 (chance 0.10)", gen_acc >= 0.8),
      check("handwriting: generated digits are not copies (median distance to the nearest training image)", gen_near, {real_near, copy_near},
            "≥ ½ of held-out real images'; the memorising control < 1", gen_near >= 0.5 * real_near and copy_near < 1.0)]}
  end

  defp real_chain(clips, w) do
    alias Vapor.Modal.{Digits, Speech}
    {:ok, model} = Speech.load()
    picked = clips |> Enum.chunk_every(10) |> Enum.map(&hd/1)

    steps =
      for {{d, clip}, i} <- Enum.with_index(picked) do
        {:ok, heard} = Speech.classify(clip, model, worker: w)
        drawn = Digits.draw(String.to_integer(heard.label), 1, seed: 500 + i, worker: w)
        %{said: d, heard: heard.label, read_back: hd(drawn.readings).digit}
      end

    ok = Enum.count(steps, &(Integer.to_string(&1.read_back) == &1.said)) / length(steps)
    {%{steps: steps, accuracy: ok},
     [check("chain: held-out voice → text → drawn digit → read back = what was said", ok, 0.1, "≥ 0.7 (chance 0.10)", ok >= 0.7)]}
  end

  defp real_jpeg do
    fx = fn n -> Path.join([to_string(:code.priv_dir(:vapor)), "quality", "jpeg", n]) end
    cases = [{"scene.jpg", "ef114fad3707daa5ec37ab16a550c8bcf0a4fb2d9fd8dec701fb04372497791f"},
             {"scene_prog.jpg", "91bd3506eeee1e6324e504f2f07209849d5f70bf6a47b38fee0394fc41eb5ae7"}]

    got =
      for {n, want} <- cases, File.exists?(fx.(n)) do
        {:ok, j} = Vapor.Docs.JPEG.decode(File.read!(fx.(n)))
        {n, Base.encode16(:crypto.hash(:sha256, j.pixels), case: :lower) == want}
      end

    {%{fixtures: got},
     if(got == [], do: [], else: [check("JPEG: baseline and progressive pixels = libjpeg-turbo (Pillow), SHA-256", Enum.count(got, &elem(&1, 1)), length(got), "all", Enum.all?(got, &elem(&1, 1)))])}
  end

  # --------------------------------------------------------------- the lock --

  @doc """
  What the airlock costs and builds, per tier: admission (claim + checks +
  weight rewrites), build (program + contract check) and lowering times,
  and the size of the program's DAG — on tiny instances of each adapter.
  """
  def lock_bench do
    tiny = fn arch, over -> Map.merge(%{"model_type" => arch, "vocab_size" => 96, "hidden_size" => 64, "intermediate_size" => 96,
                                        "num_hidden_layers" => 2, "num_attention_heads" => 4, "num_key_value_heads" => 2,
                                        "max_position_embeddings" => 32, "rms_norm_eps" => 1.0e-5, "rope_theta" => 10_000.0,
                                        "tie_word_embeddings" => false, "hidden_act" => "silu"}, over) end

    m = Planted.bigram(Enum.to_list(0..47) ++ [0], 48)
    {:ok, vq, vqw} = Vapor.Modal.VQ.fit(for(i <- 0..63, do: for(j <- 0..15, do: :math.sin(i + j * 0.1))), 16)
    {:ok, lin, linw} = Vapor.Modal.Bridge.fit(for(i <- 1..8, do: for(j <- 1..16, do: i * j * 0.01)), for(i <- 1..8, do: [i * 1.0]))

    cases = [
      {"decoder (llama)", tiny.("llama", %{}), :decoder},
      {"decoder (qwen2)", tiny.("qwen2", %{}), :decoder},
      {"blueprint (granite)", tiny.("granite", %{"residual_multiplier" => 0.2, "logits_scaling" => 2.0}), :decoder},
      {"planted bigram (llama)", m.config, {:given, m.weights}},
      {"codec (vapor_vq)", vq.config.raw, {:given, vqw}},
      {"projector (vapor_linear)", lin.config.raw, {:given, linw}}
    ]

    for {name, map, src} <- cases do
      ws =
        case src do
          {:given, ws} -> ws
          :decoder ->
            # random weights in exactly the names and shapes the airlock expects for this family
            {:ok, spec0, _} = Lock.from_map(map, %{})
            spec0 |> Lock.expected() |> Enum.with_index() |> Map.new(fn {{n, sh, _}, i} -> {n, Tensor.random(:f32, sh, i, scale: 0.2)} end)
        end

      {ta, {:ok, spec, ws2}} = :timer.tc(fn -> Lock.from_map(map, ws) end)
      build_opts = if spec.interface == :causal_lm, do: [max_seq: 16], else: []
      {tb, {:ok, p}} = :timer.tc(fn -> Lock.build(spec, ws2, build_opts) end)
      {tl, {:ok, _}} = :timer.tc(fn -> Vapor.Compile.Lower.lower(p) end)
      %{adapter: name, family: spec.family, interface: spec.interface, admit_us: ta, build_us: tb, lower_us: tl, nodes: length(Vapor.Program.order(p))}
    end
  end

  # ---------------------------------------------------------------- gallery --

  @doc "Write the any-to-any outputs of the held-out inputs as PNG (×8) and WAV files into `dir`."
  def gallery(hub, dir) do
    File.mkdir_p!(dir)
    conv = fn f, t, x -> {:ok, y, _} = Hub.convert(hub, f, t, x); y end

    for {a, b} = pair <- World.held_out() do
      src = World.scene(pair, 101)
      Image.write_png(Path.join(dir, "scene_#{a}_#{b}.png"), src, 8)
      Image.write_png(Path.join(dir, "text_to_image_#{a}_#{b}.png"), conv.(:text_colour, :image, [a, b]), 8)
      Image.write_png(Path.join(dir, "image_vq_#{a}_#{b}.png"), conv.(:image, :image, src), 8)
      Audio.write(Path.join(dir, "image_to_audio_#{a}_#{b}.wav"), conv.(:image, :audio, src))
    end

    for note <- World.notes() do
      Audio.write(Path.join(dir, "text_to_audio_#{note}.wav"), conv.(:text_note, :audio, [note]))
      Image.write_png(Path.join(dir, "audio_to_image_#{note}.png"), conv.(:audio, :image, World.sound(note, 500)), 8)
    end

    Image.write_png(Path.join(dir, "control_noise.png"), Image.noise(16, 16, 3, 1), 8)
    Image.write_png(Path.join(dir, "control_shuffled.png"), Image.shuffle(World.scene({"red", "green"}, 101), 1), 8)
    :ok
  end

  # --------------------------------------------------------------- substrates --

  defp substrates(_hub, nil), do: %{programs: [], checks: []}

  defp substrates(hub, w) do
    {ispec, iw} = hub.models.image_vq
    {lspec, lw} = hub.models.image_to_text
    img = World.scene({"red", "blue"}, 9)
    rows = Image.patches(img, 4)
    {:ok, enc} = Lock.build(ispec, iw, direction: :encode, rows: 16)
    {:ok, dec} = Lock.build(ispec, iw, direction: :decode, rows: 16)
    {:ok, lin} = Lock.build(lspec, lw, rows: 1)
    codes = Runner.run(enc, %{rows: rows}).codes

    cases = [
      {"vq encode", enc, %{rows: rows}},
      {"vq decode", dec, %{codes: codes}},
      {"linear bridge", lin, %{rows: Image.to_row(img)}},
      {"spectrum (Hann STFT)", Audio.spectrum_program(256, 4, 128), %{rows: Audio.frames(World.sound("mi", 3), 256, 256)}},
      {"additive synthesis", Audio.sinusoids_program([261.63, 329.63, 392.0, 493.88], 8000, 1024, 1), %{amps: Tensor.from_list(:f32, [1, 16], [0.1, 0.5, 0.2, 0.0] ++ List.duplicate(0.0, 12))}}
    ]

    progs = for {name, p, env} <- cases do
      same = Runner.run(p, env) == Runner.run(p, env, worker: w)
      %{program: name, bit_identical: same}
    end

    %{programs: progs, checks: [check("substrates: native = oracle bit for bit on every modal program", Enum.count(progs, & &1.bit_identical), length(progs), "all", Enum.all?(progs, & &1.bit_identical))]}
  end
end
