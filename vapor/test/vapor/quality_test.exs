defmodule Vapor.QualityTest do
  @moduledoc """
  The quality gate tests itself before it tests anything else: it must
  fail every noise control and pass every real sample, or refuse to exist.
  Then planted models with known answers, and the whole benchmark.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Lock, Tensor}
  alias Vapor.Modal.{Audio, Image, World}
  alias Vapor.Quality.{Gate, Planted, Signal, Suite, Text}

  test "a gate is fitted to controls and refuses to exist when they overlap" do
    assert {:ok, g} = Gate.calibrate(:x, %{noise: [0.1, 0.2]}, %{real: [0.8, 0.9]})
    assert Gate.judge(g, 0.15) == :noise and Gate.judge(g, 0.5) == :structured and Gate.judge(g, 0.95) == :natural
    assert {:error, {:inseparable, %{worst_negative: 0.85, worst_positive: 0.8}}} = Gate.calibrate(:x, %{noise: [0.1, 0.85]}, %{real: [0.8, 0.9]})
    # direction :down (lower = more structure)
    assert {:ok, d} = Gate.calibrate(:y, %{noise: [0.9, 0.8]}, %{real: [0.1]}, :down)
    assert Gate.judge(d, 0.85) == :noise and Gate.judge(d, 0.05) == :natural
  end

  test "text: the gate separates held-out prose from uniform bytes, shuffled letters and loops" do
    {train, hold} = Suite.corpus_pt()
    p = Text.profile(train)
    assert {:ok, g} = Text.gate(p, hold, len: 200, count: 16)
    cs = Text.controls(hold, 200, 6, 42)
    for s <- cs.negatives.uniform ++ cs.negatives.unigram ++ cs.negatives.loop, do: assert(Text.judge(s, p, g).verdict == :fail)
    for s <- cs.positives.holdout, do: assert(Text.judge(s, p, g).verdict == :pass)
    assert Text.judge("curto", p, g).verdict == :too_short
  end

  test "image and audio: noise and shuffles score as noise, scenes and tones as signal" do
    scene = World.scene({"red", "blue"}, 3)
    assert Signal.neighbour_corr(scene) > 0.8
    assert abs(Signal.neighbour_corr(Image.noise(16, 16, 3, 1))) < 0.2
    assert abs(Signal.neighbour_corr(Image.shuffle(scene, 1))) < 0.2
    assert Signal.spectral_slope(World.scene({"red", "blue"}, 0, 32)) < -1.5
    assert abs(Signal.spectral_slope(Image.noise(32, 32, 1, 2))) < 0.6
    assert Signal.psnr(scene, scene) == :infinity
    assert_in_delta Signal.ssim(scene, scene), 1.0, 1.0e-12

    tone = World.sound("mi", 1)
    assert Signal.flatness(tone) < 0.01
    assert Signal.flatness(Audio.noise(8000, 1024, 3)) > 0.4
    assert Signal.flatness(Audio.shuffle(tone, 2)) > 0.4
    assert_in_delta Signal.dominant_hz(tone), 329.63, 8000 / 1024
  end

  test "the planted bigram reproduces its analytic table through airlock, compiler and oracle" do
    a = Planted.alphabet()
    {train, hold} = Suite.corpus_pt()
    m = Planted.bigram(a.encode.(train), a.size)
    {:ok, spec, ws} = Lock.from_map(m.config, m.weights)
    ids = a.encode.(binary_part(hold, 500, 40))
    rows = Vapor.Modal.Text.logits(spec, ws, ids)

    for {row, i} <- Enum.zip(rows, ids), {x, y} <- Enum.zip(Enum.take(row, a.size), Enum.take(Enum.at(m.logp, i), a.size)),
        do: assert(abs(x - y) < 1.0e-5)

    bits = Text.bits_per_token(fn p -> Text.log_softmax(Enum.at(rows, length(p) - 1)) end, ids, a.size).bits
    assert_in_delta bits, Planted.table_bits(m, ids), 1.0e-4
    assert bits < Text.unigram_bits(a.decode.(ids), Text.profile(train))
  end

  test "a random-weight model is noise to the gate; the planted model is not" do
    a = Planted.alphabet()
    {train, hold} = Suite.corpus_pt()
    p = Text.profile(train)
    {:ok, g} = Text.gate(p, hold, len: 160, count: 16)
    m = Planted.bigram(a.encode.(train), a.size)
    {:ok, spec, ws} = Lock.from_map(m.config, m.weights)
    rw = Map.new(m.weights, fn {k, t} -> {k, Tensor.random(:f32, t.shape, :erlang.phash2(k))} end)
    {:ok, rspec, rws} = Lock.from_map(m.config, rw)
    gen = fn s, w -> s |> Vapor.Modal.Text.generate(w, a.encode.("o "), 170, temperature: 1.0, seed: 3) |> a.decode.() end
    assert Text.judge(gen.(spec, ws), p, g).verdict == :pass
    assert Text.judge(gen.(rspec, rws), p, g).verdict == :fail
  end

  @tag :native
  @tag timeout: 3_600_000
  test "a checkpoint with a real tokenizer: a planted byte bigram is signal, the same model with random weights is noise" do
    tk = Vapor.TestHelpers.byte_bpe_tokenizer()
    {reference, hold} = Suite.corpus_pt_raw()
    ids = Vapor.Tokenizer.encode(tk, reference, add_bos: false)
    m = Planted.bigram(ids, 259)
    {:ok, spec, ws} = Lock.from_map(m.config, m.weights)
    assert {:ok, r} = Vapor.Quality.Model.judge(%{spec: spec, weights: ws, tokenizer: tk}, bytes: 600, samples: 2, text: hold)
    assert r.verdict == :signal and r.bits_per_byte < r.unigram_bits_per_byte

    rw = Map.new(m.weights, fn {k, t} -> {k, Tensor.random(:f32, t.shape, :erlang.phash2(k))} end)
    {:ok, rspec, rws} = Lock.from_map(m.config, rw)
    assert {:ok, bad} = Vapor.Quality.Model.judge(%{spec: rspec, weights: rws, tokenizer: tk}, bytes: 600, samples: 2, text: hold)
    assert bad.verdict == :noise and bad.bits_per_byte > bad.unigram_bits_per_byte
  end

  @tag :native
  @tag timeout: 3_600_000
  test "the whole benchmark: every check passes (text, any-to-any, fusion, substrates)" do
    r = Suite.run()
    failed = for c <- Suite.checks(r), not c.pass, do: {c.name, c.value, c.threshold}
    assert failed == []
    assert r.substrate == :native and length(r.any_to_any) == 10
  end
end
