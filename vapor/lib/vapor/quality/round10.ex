defmodule Vapor.Quality.Round10 do
  @moduledoc """
  Quality checks for the 0.10 round, in the suite's discipline: each check
  has a value, a **control** that a broken, naive or lucky implementation
  would produce, and a threshold that separates them.

  | check | value | control (must fail) |
  |---|---|---|
  | substrate airlock | the host's verdict | a bf16-operand device (simulated): refused, 8 significand bits measured |
  | DAZ envelope | a denormals-are-zero result inside the bound | twice the exact value: outside |
  | language model | held-out bits/byte of the shipped model, through the inference stack | Witten–Bell order 5; the same model on byte-shuffled text |
  | unbounded stream | bits/byte 14× past the training length, 64 cache rows | the same model with growing positions |
  | physics | pendulum period error at 16 substeps | at 4 substeps (first order: ≥ 1.6× larger per halving) |
  | chaos | oracle = native for 200 steps | a one-ulp start: parted |
  | twin identification | rod length and damping recovered | time-shuffled measurements |
  | reinforcement | a cart-pole balanced 200/200 on unseen starts | the zero policy |
  | digital twin | alarm within 30 steps of a 0.5 % fault | no fault: no alarm |
  | networks | BA: power law; WS: clustering z ≫ 0; Louvain on a planted partition | ER: not a power law, z ≈ 0; the partition's null |
  | figures | charts within 1 % of span | permuted tick labels: every chart refused |
  | formulas | LaTeX token error (unseen type families) | the same symbols read flat |
  | CJK | CER on unseen typefaces | random characters: the language model must not help |
  | Arabic | CER on unseen typefaces; logical order = python-bidi's | the Latin reader on the same lines |
  | Cyrillic | CER on unseen typefaces | the Latin reader on the same lines |
  """
  alias Vapor.{Graph, Physics, Tensor}
  alias Vapor.Vision.{CJK, Figure, OCR}

  def run(opts \\ []) do
    w = Keyword.get(opts, :worker)
    checks = List.flatten([airlock(), lm(w), physics(w), networks(), figures(w), formulas(), cjk(w), arabic(w), cyrillic(w)])
    %{checks: checks}
  end

  defp check(name, value, control, threshold, pass), do: %{name: name, value: value, control: control, threshold: threshold, pass: pass}
  defp p(rel), do: Path.join(to_string(:code.priv_dir(:vapor)), rel)

  # ------------------------------------------------------------- airlock --

  defp airlock do
    host = Vapor.Substrate.admit(%{id: :oracle, kind: :oracle})

    bf16 = fn c, env, opts ->
      round = fn %Tensor{dtype: :f32} = t -> t |> Tensor.to_bf16() |> Tensor.widen(); t -> t end
      prog = deep_map(c.program, round)
      {:ok, c2} = Vapor.Compile.Lower.lower(prog, policy: c.policy)
      Vapor.Runtime.Native.run_oracle(c2, Map.new(env, fn {k, v} -> {k, round.(v)} end), opts)
    end

    dev = Vapor.Substrate.admit(%{id: :bf16_engine}, runner: bf16, only: [:input_precision, :linear_f32])

    sub = Vapor.F32.from_float(:math.pow(2, -130))
    x = Tensor.new(:f32, [16], :binary.copy(<<sub::32-little>>, 16))
    prog = Vapor.Program.new(big: Vapor.Algebra.Term.mul(Vapor.Algebra.Term.input(:x, :f32, [16]), Vapor.Algebra.Term.splat(:math.pow(2, 100))))
    {:ok, c} = Vapor.Compile.Lower.lower(prog)
    [b] = Vapor.Verify.Envelope.bounds(c, %{x: x})
    zero = Tensor.from_list(:f32, [16], List.duplicate(0.0, 16))
    twice = Tensor.from_list(:f32, [16], List.duplicate(:math.pow(2, -28), 16))

    [check("airlock: the exact oracle admitted canonical; a bf16-operand device refused", host.verdict, dev.verdict, "canonical vs refused (8 bits measured)",
           host.verdict == :canonical and dev.verdict == :refused and dev.fingerprint.significand_bits == 8),
     check("envelope: a DAZ result inside the bound; twice the exact value outside", Vapor.Verify.Envelope.check(b.big, zero), elem(Vapor.Verify.Envelope.check(b.big, twice), 0),
           "ok vs error", Vapor.Verify.Envelope.check(b.big, zero) == :ok and match?({:error, _}, Vapor.Verify.Envelope.check(b.big, twice)))]
  end

  defp deep_map(%Tensor{} = t, f), do: f.(t)
  defp deep_map(%{__struct__: m} = s, f), do: s |> Map.from_struct() |> Map.new(fn {k, v} -> {k, deep_map(v, f)} end) |> then(&struct(m, &1))
  defp deep_map(t, f) when is_tuple(t), do: t |> Tuple.to_list() |> Enum.map(&deep_map(&1, f)) |> List.to_tuple()
  defp deep_map(l, f) when is_list(l), do: Enum.map(l, &deep_map(&1, f))
  defp deep_map(x, _f), do: x

  # ------------------------------------------------------------------- LM --

  defp lm(nil), do: []

  defp lm(w) do
    dir = p("lm/model")
    {:ok, receipt} = Vapor.JSON.decode(File.read!(p("lm/receipt.json")))
    train = Enum.map_join(receipt["corpus"], &Vapor.Quality.Suite.corpus(&1["file"]))
    hold = Vapor.Quality.Suite.corpus("pt_holdout.txt") <> Vapor.Quality.Suite.corpus("en_holdout.txt")
    text = binary_part(hold, 0, min(byte_size(hold), 4096))
    toks = :binary.bin_to_list(text)
    shuffled = Vapor.Modal.Rng.permute(toks, 7)
    win = fn ts -> bits_windows(dir, ts, w) end
    model = win.(toks)
    shuf = win.(shuffled)
    # the baseline on the very bytes the model is scored on (the receipt's is on the whole held-out set)
    wb5 = Vapor.Train.LM.baselines(train, text).witten_bell_5

    {:ok, st} = Vapor.Streaming.open(dir, sinks: 4, window: 60, worker: w)
    long = Enum.take(toks, 900)
    {st, rows} = Vapor.Streaming.feed(st, long)
    Vapor.Streaming.close(st)
    stream = Vapor.Streaming.bits(rows, long, 64)
    dense = Vapor.Streaming.bits(causal(dir, long, 1024, w), long, 64)

    [check("trained LM: held-out bits/byte (inference stack) vs Witten–Bell order 5", model, wb5, "model < WB-5", model < wb5),
     check("trained LM: byte-shuffled held-out text (the model reads order, not frequencies)", model, shuf, "shuffled > model + 1.5", shuf > model + 1.5),
     check("stream: bits/byte 14× past the training length in 64 rows vs growing positions", stream, dense, "dense > stream + 1.5", dense > stream + 1.5)]
  end

  # windows of 64 bytes, each read fresh (the training length)
  defp bits_windows(dir, toks, w) do
    {:ok, %{program: p}} = Vapor.Model.load(dir, max_seq: 64)
    {:ok, comp} = Vapor.Compile.Lower.lower(p)
    {:ok, s} = Vapor.Runtime.Session.open(w, comp, isa: Vapor.Runtime.Substrates.host_isa())
    ids = fn xs -> Tensor.from_list(:s32, [length(xs)], xs) end

    {sum, n} =
      toks
      |> Enum.chunk_every(64, 63, :discard)
      |> Enum.reduce({0.0, 0}, fn win, {acc, n} ->
        {:ok, o, _} = Vapor.Runtime.Session.step(s, %{tok: ids.(win), pos: ids.(Enum.to_list(0..63))}, [:logits])
        rows = o.logits |> Tensor.to_floats() |> Enum.chunk_every(256)
        {acc + Vapor.Streaming.bits(rows, win) * 63, n + 63}
      end)

    Vapor.Runtime.Session.close(s)
    sum / n
  end

  defp causal(dir, toks, max_seq, w) do
    {:ok, %{program: p}} = Vapor.Model.load(dir, max_seq: max_seq)
    {:ok, comp} = Vapor.Compile.Lower.lower(p)
    {:ok, s} = Vapor.Runtime.Session.open(w, comp, isa: Vapor.Runtime.Substrates.host_isa())
    ids = fn xs -> Tensor.from_list(:s32, [length(xs)], xs) end
    {:ok, o, _} = Vapor.Runtime.Session.step(s, %{tok: ids.(toks), pos: ids.(Enum.to_list(0..(length(toks) - 1)))}, [:logits])
    Vapor.Runtime.Session.close(s)
    o.logits |> Tensor.to_floats() |> Enum.chunk_every(256)
  end

  # -------------------------------------------------------------- physics --

  defp physics(nil), do: []

  defp physics(w) do
    exact = Physics.pendulum_period(1.0, 0.5)
    err = fn sub -> abs(period(Physics.pendulum(theta: 0.5, substeps: sub, dt: 0.01), 400, w) - exact) / exact end
    [e4, e8, e16] = Enum.map([4, 8, 16], err)

    wd = Physics.double_pendulum(substeps: 8, dt: 0.01)
    x2 = wd.pos0 |> Enum.at(2) |> hd()
    <<i::32>> = <<x2::float-32>>
    <<x2u::float-32>> = <<i + 1::32>>
    st = Physics.state(wd, 2, perturb: fn b, q, a -> if b == 1 and q == 2 and a == :x, do: x2u - x2, else: 0.0 end)
    # the two substrates step by step from the same start
    {n, _, identical} =
      Enum.reduce(1..20, {Physics.start(wd, 2, worker: w, state: st, steps: 10), Physics.start(wd, 2, state: st, steps: 10), true}, fn _, {a, b, ok} ->
        {a, sa} = Physics.advance(a)
        {b, sb} = Physics.advance(b)
        {a, b, ok and sa == sb}
      end)

    {n, last} = Enum.reduce(1..180, {n, nil}, fn _, {n, _} -> Physics.advance(n) end)
    Physics.stop(n)
    [xa, xb] = Physics.rows(n, last.p_x)
    gap = abs(Enum.at(xa, 2) - Enum.at(xb, 2))

    {k, _} = Physics.cartpole_ars(worker: w)
    trained = Physics.cartpole_returns(List.duplicate(k, 8), worker: w, seed: 999)
    zero = Physics.cartpole_returns(List.duplicate([0.0, 0.0, 0.0, 0.0], 8), worker: w, seed: 999)

    {fit, ctrl} = sysid(w)
    {quiet, faulty} = twins(w)

    [check("physics: pendulum period, relative error at 16 substeps (first order: 4 → 8 → 16)", e16, e4, "e16 < 2e-4, each halving ≥ 1.6× better",
           e16 < 2.0e-4 and e4 > 1.6 * e8 and e8 > 1.6 * e16),
     check("chaos: oracle = native, bit for bit, for 200 steps; a one-ulp start parts by 20 s", identical, gap, "identical and gap > 1e-2", identical and gap > 1.0e-2),
     check("twin identification: rod length recovered from noisy measurements (error)", abs(hd(fit.rest) - 1.0), abs(hd(ctrl.rest) - 1.0),
           "< 0.01 vs > 0.05 (time-shuffled)", abs(hd(fit.rest) - 1.0) < 0.01 and abs(hd(ctrl.rest) - 1.0) > 0.05),
     check("reinforcement: cart-pole returns on unseen starts (random search)", Enum.min(trained), Enum.max(zero), "all 200 vs zero policy < 100",
           Enum.all?(trained, &(&1 == 200)) and Enum.max(zero) < 100),
     check("digital twin: steps from a 0.5 % fault to the alarm", faulty.alarm && faulty.alarm - 60, quiet.alarm, "≤ 30 vs no alarm without the fault",
           quiet.alarm == nil and faulty.alarm != nil and faulty.alarm - 60 <= 30)]
  end

  defp period(wd, steps, w) do
    sim = Physics.start(wd, 1, worker: w)
    {xs, sim} = Enum.map_reduce(1..steps, sim, fn _, s -> {s, st} = Physics.advance(s); {s |> Physics.rows(st.p_x) |> hd() |> Enum.at(1), s} end)
    Physics.stop(sim)
    cross = xs |> Enum.with_index(1) |> Enum.chunk_every(2, 1, :discard) |> Enum.filter(fn [{a, _}, {b, _}] -> a > 0 and b <= 0 end) |> Enum.map(fn [{a, i}, {b, _}] -> (i + a / (a - b)) * wd.dt end)
    (List.last(cross) - hd(cross)) / (length(cross) - 1)
  end

  defp sysid(w) do
    plant = Physics.pendulum(theta: 0.8, substeps: 4, dt: 0.02, damping: 0.3)
    sim = Physics.start(plant, 1, worker: w)

    {obs, sim} =
      Enum.map_reduce(1..40, sim, fn t, s ->
        {s, st} = Physics.advance(s)
        noise = fn k -> (Vapor.Sampler.uniform(7, t * 10 + k) - 0.5) * 0.004 end
        {%{x: Enum.with_index(hd(Physics.rows(s, st.p_x)), fn v, i -> v + noise.(i) end), y: Enum.with_index(hd(Physics.rows(s, st.p_y)), fn v, i -> v + noise.(i + 5) end)}, s}
      end)

    Physics.stop(sim)
    twin = %{plant | damping: 0.0}
    {Physics.sysid(twin, obs, worker: w, rest: [0.85], damping: 0.0, iters: 100, lr: 0.02),
     Physics.sysid(twin, Vapor.Modal.Rng.permute(obs, 3), worker: w, rest: [0.85], damping: 0.0, iters: 100, lr: 0.02)}
  end

  defp twins(w) do
    wd = Physics.pendulum(theta: 0.6, substeps: 4, dt: 0.02)
    noise = fn t, k -> (Vapor.Sampler.uniform(11, t * 10 + k) - 0.5) * 0.004 end
    meas = fn s, st, t -> %{x: Enum.with_index(hd(Physics.rows(s, st.p_x)), fn v, i -> v + noise.(t, i) end), y: Enum.with_index(hd(Physics.rows(s, st.p_y)), fn v, i -> v + noise.(t, i + 5) end)} end
    unstate = fn sim -> Map.new(sim.state, fn {k, t} -> [kind, a] = String.split(to_string(k), "_"); {{String.to_atom(kind), String.to_atom(a)}, Enum.chunk_every(Tensor.to_floats(t), sim.world.np)} end) end

    run = fn fault ->
      Enum.reduce(1..150, {Physics.start(wd, 1), Physics.twin(wd, worker: w)}, fn t, {plant, tw} ->
        plant = if t == fault, do: Physics.start(wd, 1, params: [rest: [1.005]], state: unstate.(plant)), else: plant
        {plant, st} = Physics.advance(plant)
        {plant, Physics.observe(tw, nil, meas.(plant, st, t))}
      end)
      |> elem(1)
      |> tap(&Physics.stop(&1.sim))
    end

    {run.(nil), run.(60)}
  end

  # -------------------------------------------------------------- networks --

  defp networks do
    ba = Graph.barabasi_albert(1000, 3, 1)
    er = Graph.erdos_renyi(1000, 6 / 999, 2)
    pb = Graph.power_law(Graph.degrees(ba), boot: 40)
    pe = Graph.power_law(Graph.degrees(er), boot: 40)
    ws = Graph.zscore(Graph.watts_strogatz(500, 10, 0.05, 4), &Graph.avg_clustering/1, 8, 5)
    erz = Graph.zscore(Graph.erdos_renyi(500, 10 / 499, 1), &Graph.avg_clustering/1, 8, 5)
    {g, truth} = Graph.planted(4, 50, 0.3, 0.02, 7)
    labels = Graph.communities(g)
    null = Graph.rewire(g, 10 * Graph.edge_count(g), 3)
    q = Graph.modularity(g, labels)
    q0 = Graph.modularity(null, Graph.communities(null))
    sf = Graph.barabasi_albert(2000, 2, 3)

    [check("networks: Barabási–Albert degrees, power-law verdict (Clauset–Shalizi–Newman)", pb.verdict, pe.verdict, "power_law vs not (Erdős–Rényi)",
           pb.verdict == :power_law and pe.verdict in [:rejected, :exponential]),
     check("networks: small-world clustering, z against the configuration null", ws.z, erz.z, "> 50 vs |z| < 3 (random graph)", ws.z > 50 and abs(erz.z) < 3),
     check("networks: Louvain on a planted partition (NMI; modularity vs its null)", Graph.nmi(labels, truth), q0, "NMI > 0.95, Q > Q_null + 0.25",
           Graph.nmi(labels, truth) > 0.95 and q > q0 + 0.25),
     check("networks: scale-free robustness — giant component after 15 % random failure", Graph.percolation(sf, 0.15, :failure), Graph.percolation(sf, 0.15, :attack),
           "> 0.9 vs attack < 0.6× failure", Graph.percolation(sf, 0.15, :failure) > 0.9 and Graph.percolation(sf, 0.15, :attack) < 0.6 * Graph.percolation(sf, 0.15, :failure))]
  end

  # --------------------------------------------------------------- figures --

  defp figures(w) do
    dir = p("quality/figures")
    run = fn split ->
      {:ok, truth} = Vapor.JSON.decode(File.read!(Path.join([dir, split, "truth.json"])))

      for {f, t} <- Enum.sort(truth) do
        {:ok, pic} = Vapor.Docs.Pictures.read(:png, File.read!(Path.join([dir, split, f])))
        r = Figure.digitize(pic.image, worker: w)
        {r, Vapor.Quality.Figures.chart(r, t)}
      end
    end

    charts = run.("charts")
    shuffled = run.("shuffled")
    ok = Enum.count(charts, fn {_, s} -> s.ok end)
    refused = Enum.count(shuffled, fn {r, _} -> match?({:error, _}, r) end)

    [check("figures: charts read within 1 % of the axis span (held-out, default style)", ok / length(charts), refused / length(shuffled),
           "≥ 0.8; permuted tick labels: all refused", ok >= 0.8 * length(charts) and refused == length(shuffled))]
  end

  # -------------------------------------------------------------- formulas --

  defp formulas do
    dir = p("quality/math/test")
    {:ok, meta} = Vapor.JSON.decode(File.read!(Path.join(dir, "truth.json")))

    pairs =
      for {f, t} <- Enum.sort(meta["items"]) do
        {:ok, pic} = Vapor.Docs.Pictures.read(:png, File.read!(Path.join(dir, f)))
        {:ok, r} = Vapor.Vision.Math.read(pic.image)
        flat = r.symbols |> Enum.sort_by(fn %{box: {x0, _, _, _}} -> x0 end) |> Enum.map_join(& &1.latex)
        {t["latex"], r.latex, flat}
      end

    ter = fn sel -> Enum.sum(for p <- pairs, do: OCR.levenshtein(Vapor.Vision.Math.tokens(sel.(p)), Vapor.Vision.Math.tokens(elem(p, 0)))) /
                      Enum.sum(for p <- pairs, do: length(Vapor.Vision.Math.tokens(elem(p, 0)))) end

    structured = ter.(&elem(&1, 1))
    flat = ter.(&elem(&1, 2))
    [check("formulas → LaTeX: token error on Computer Modern and STIX (never in the templates)", structured, flat, "< 0.08 vs flat reading > 3×",
           structured < 0.08 and flat > 3 * structured)]
  end

  # ------------------------------------------------------------------- CJK --

  defp cjk(nil), do: []

  defp cjk(w) do
    for lang <- [:zh, :ja, :ko] do
      {:ok, pack} = CJK.default(lang)
      cer = fn split, lm -> cer_cjk(pack, Path.join([p("quality/cjk"), Atom.to_string(lang), split]), lm, w) end
      {t0, t1, r0, r1} = {cer.("test", false), cer.("test", true), cer.("random", false), cer.("random", true)}
      bound = %{zh: 0.15, ja: 0.08, ko: 0.2}[lang]

      check("CJK #{lang}: CER on unseen typefaces (greedy → with the language model)", "#{Float.round(t0, 4)} → #{Float.round(t1, 4)}", r1 - r0,
            "≤ #{bound}; ja, ko: the model helps (> 0.01); on random characters it changes nothing (≤ +0.005)",
            min(t0, t1) <= bound and r1 - r0 <= 0.005 and (lang == :zh or t1 < t0 - 0.01))
    end
  end

  defp cer_cjk(pack, dir, lm, w) do
    {:ok, meta} = Vapor.JSON.decode(File.read!(Path.join(dir, "labels.json")))

    {e, n} =
      Enum.reduce(Enum.sort(meta["lines"]), {0, 0}, fn {f, %{"text" => truth}}, {e, n} ->
        {:ok, pic} = Vapor.Docs.Pictures.read(:png, File.read!(Path.join(dir, f)))
        {:ok, r} = CJK.read(pack, pic.image, worker: w, lm: lm)
        {e + OCR.levenshtein(String.graphemes(r.text), String.graphemes(truth)), n + String.length(truth)}
      end)

    e / n
  end

  # ---------------------------------------------------------------- Arabic --

  defp arabic(nil), do: []

  defp arabic(w) do
    dir = p("quality/arabic/test")
    {:ok, meta} = Vapor.JSON.decode(File.read!(Path.join(dir, "labels.json")))
    lines = Enum.sort(meta["lines"])
    bidi_ok = Enum.all?(lines, fn {_, l} -> Vapor.Vision.Bidi.logical(l["text"]) == l["logical"] end)
    {:ok, ar} = OCR.default(:arabic)
    {:ok, la} = OCR.default(:latin)

    v = line_cer(ar, dir, Enum.take(lines, 60), "logical", w)
    c = line_cer(la, dir, Enum.take(lines, 20), "logical", w)

    [check("Arabic: CER on unseen typefaces, read in visual order and returned to logical order", v, c, "< 0.25 vs the Latin reader > 0.8; inverse bidi = python-bidi on every line",
           v < 0.25 and c > 0.8 and bidi_ok)]
  end

  # -------------------------------------------------------------- Cyrillic --

  defp cyrillic(nil), do: []

  defp cyrillic(w) do
    dir = p("quality/cyrillic/test")
    {:ok, meta} = Vapor.JSON.decode(File.read!(Path.join(dir, "labels.json")))
    lines = Enum.sort(meta["lines"])
    {:ok, cy} = OCR.default(:cyrillic)
    {:ok, la} = OCR.default(:latin)
    v = line_cer(cy, dir, lines, "text", w)
    c = line_cer(la, dir, Enum.take(lines, 20), "text", w)

    [check("Cyrillic: CER on unseen typefaces (Russian, a vocabulary half the training never saw)", v, c, "< 0.08 vs the Latin reader > 0.6",
           v < 0.08 and c > 0.6)]
  end

  defp line_cer(model, dir, lines, key, w) do
    {e, n} =
      Enum.reduce(lines, {0, 0}, fn {f, l}, {e, n} ->
        {:ok, pic} = Vapor.Docs.Pictures.read(:png, File.read!(Path.join(dir, f)))
        {:ok, r} = OCR.read(pic.image, model: model, worker: w, lm: false, figures: false, tables: false)
        ref = String.trim(Regex.replace(~r/ +/, l[key], " "))
        {e + OCR.levenshtein(String.graphemes(r.text), String.graphemes(ref)), n + String.length(ref)}
      end)

    e / n
  end
end
