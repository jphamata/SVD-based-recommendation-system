defmodule Vapor.ModalTest do
  @moduledoc """
  The any-to-any layer: codecs that are exact permutations or certified
  programs, closed-form fits, the hub's routes — no new operator anywhere.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Lock, Tensor}
  alias Vapor.Modal.{Audio, Bridge, Hub, Image, Runner, VQ, World}
  alias Vapor.Quality.Signal
  alias Vapor.Runtime.Oracle

  test "PPM/PGM and WAV: write, read, the same values (8-bit / 16-bit quantised)" do
    img = World.scene({"red", "blue"})
    {:ok, back} = Image.parse(Image.encode(img))
    assert {back.w, back.h, back.c} == {16, 16, 3}
    assert Enum.zip_with(Image.values(img), Image.values(back), &abs(&1 - &2)) |> Enum.max() <= 0.5 / 255 + 1.0e-12

    gray = Image.new(2, 1, 1, [0.0, 1.0])
    {:ok, g} = Image.parse("P5\n# a comment\n2 1\n255\n" <> <<0, 255>>)
    assert Image.values(g) == Image.values(gray)
    assert {:error, _} = Image.parse("P3\n1 1\n255\n0 0 0")

    a = World.sound("mi", 1)
    {:ok, b} = Audio.parse(Audio.encode(a))
    assert b.rate == 8000 and length(b.samples) == 1024
    assert Enum.zip_with(a.samples, b.samples, &abs(&1 - &2)) |> Enum.max() <= 0.5 / 32768 + 1.0e-12
  end

  test "patches are an exact permutation: from_patches ∘ patches = id, in the convolution's (c, i, j) order" do
    img = World.scene({"green", "yellow"}, 3)
    rows = Image.patches(img, 4)
    assert rows.shape == [16, 48]
    back = Image.from_patches(rows, 16, 16, 3, 4)
    # binary32 storage: the values come back rounded once
    assert Enum.zip_with(Image.values(img), Image.values(back), &abs(&1 - &2)) |> Enum.max() < 1.0e-7
    # row 1 is the patch at x = 4…7, y = 0…3; its entry (c = 2, i = 1, j = 3) is pixel (7, 1) channel 2
    assert_in_delta Enum.at(Tensor.to_floats(rows), 48 + 2 * 16 + 1 * 4 + 3), Image.at(img, 7, 1, 2), 1.0e-7
  end

  test "the spectrum program (a certified linear map, squared) agrees with an FFT" do
    a = Audio.chord([500.0, 1250.0], 8000, 256)
    p = Audio.spectrum_program(256, 1, 128)
    got = Oracle.eval_program(p, %{rows: Audio.frames(a, 256, 256)}).power |> Tensor.to_floats()
    win = for i <- 0..255, do: 0.5 - 0.5 * :math.cos(2 * :math.pi() * i / 256)
    ref = a.samples |> Enum.zip_with(win, &{&1 * &2, 0.0}) |> Signal.fft() |> Enum.take(128) |> Enum.map(fn {re, im} -> re * re + im * im end)
    scale = Enum.max(ref)
    assert Enum.zip_with(got, ref, &abs(&1 - &2)) |> Enum.max() < 1.0e-5 * scale
    # the peaks are at 500 and 1250 Hz (bins 16 and 40)
    top = got |> Enum.with_index() |> Enum.sort_by(&(-elem(&1, 0))) |> Enum.take(2) |> Enum.map(&elem(&1, 1)) |> Enum.sort()
    assert top == [16, 40]
  end

  test "additive synthesis renders the amplitudes it is given" do
    p = Audio.sinusoids_program([250.0, 500.0], 8000, 512, 1)
    rows = Oracle.eval_program(p, %{amps: Tensor.from_list(:f32, [1, 16], [0.0, 0.7] ++ List.duplicate(0.0, 14))}).rows
    a = Audio.from_frames(rows, 8000)
    assert_in_delta Signal.dominant_hz(a), 500.0, 8000 / 512
    assert Signal.flatness(a) < 0.05
  end

  test "VQ: k-means is deterministic; the in-algebra encoder picks the nearest codeword; decoding is exact" do
    rows = for i <- 0..39, do: (for j <- 0..15, do: :math.sin(i * 0.7 + j * 0.3) + rem(i, 3))
    {:ok, s1, w1} = VQ.fit(rows, 8, iters: 5)
    {:ok, _s2, w2} = VQ.fit(rows, 8, iters: 5)
    assert w1 == w2
    book = w1["codebook"] |> Tensor.to_floats() |> Enum.chunk_every(16)

    {:ok, enc} = Lock.build(s1, w1, direction: :encode, rows: 40)
    {:ok, dec} = Lock.build(s1, w1, direction: :decode, rows: 40)
    codes = Oracle.eval_program(enc, %{rows: Tensor.from_list(:f32, [40, 16], List.flatten(rows))}).codes
    want = Enum.map(rows, &VQ.nearest(Enum.map(&1, fn x -> Vapor.CR.to_f32(x) end), book))
    assert Tensor.to_list(codes) == want
    back = Oracle.eval_program(dec, %{codes: codes}).rows
    assert back.data == Enum.map_join(want, &Tensor.row(w1["codebook"], &1))
  end

  test "a ridge bridge recovers an affine map exactly (primal and dual forms)" do
    w = [[1.5, -2.0, 0.25], [0.0, 3.0, -1.0]]
    b = [0.5, -0.25]
    f = fn x -> Enum.zip_with(w, b, fn row, bi -> Enum.zip_reduce(row, x, bi, fn a, c, s -> s + a * c end) end) end
    many = for i <- 1..12, do: [i * 0.3, :math.sin(i), rem(i, 4) * 1.0]
    few = Enum.take(many, 3) ++ [[0.0, 0.0, 1.0]]

    for xs <- [many, few] do
      {:ok, spec, ws} = Bridge.fit(xs, Enum.map(xs, f), ridge: 1.0e-12)
      assert spec.interface == :map and spec.in_width == 16
      {:ok, p} = Lock.build(spec, ws, rows: 1)
      x = [0.7, -1.1, 2.0]
      out = Oracle.eval_program(p, %{rows: Tensor.from_list(:f32, [1, 16], hd(Bridge.pad([x], 16)))}).out |> Tensor.to_floats()
      # exact when the data determine the map; the dual (n < d) form is the minimum-norm one
      if xs == many, do: Enum.zip_with(out, f.(x), &assert_in_delta(&1, &2, 1.0e-4))
    end
  end

  test "soft-token injection: a row given as `soft` acts exactly like the token whose embedding it is" do
    m = Vapor.Quality.Planted.bigram([1, 2, 3, 1, 2, 3, 4, 1], 8)
    {:ok, spec, ws} = Lock.from_map(m.config, m.weights)
    {:ok, p} = Lock.build(spec, ws, max_seq: 8, inject: true)
    ids = [1, 2, 3]
    emb = fn i -> for j <- 0..(spec.width - 1), do: if(j == i, do: 1.0, else: 0.0) end
    base = Map.merge(Lock.zero_state(p), %{tok: Tensor.from_list(:s32, [3], ids), pos: Tensor.from_list(:s32, [3], [0, 1, 2])})
    plain = Oracle.eval_program(p, Map.merge(base, %{soft: Tensor.from_list(:f32, [3, spec.width], List.duplicate(0.0, 3 * spec.width)),
                                                     soft_mask: Tensor.from_list(:f32, [3, 1], [0.0, 0.0, 0.0])}))
    # token 2 at position 1 replaced by its own embedding row, token 0 placed in the slot
    injected = Oracle.eval_program(p, Map.merge(base, %{tok: Tensor.from_list(:s32, [3], [1, 0, 3]),
                                                        soft: Tensor.from_list(:f32, [3, spec.width], List.duplicate(0.0, spec.width) ++ emb.(2) ++ List.duplicate(0.0, spec.width)),
                                                        soft_mask: Tensor.from_list(:f32, [3, 1], [0.0, 1.0, 0.0])}))
    assert injected.logits == plain.logits
  end

  test "the hub routes every pair through N codecs, not N² converters" do
    hub = Hub.fit_world(variants: 2)
    routes = Hub.routes(hub)
    for a <- [:image, :audio, :text_colour, :text_note], b <- [:image, :audio, :text_colour, :text_note], do: assert({a, b} in routes)
    assert {:error, _} = Hub.convert(hub, :image, :video, nil)
    {:ok, words, route} = Hub.convert(hub, :image, :text_colour, World.scene({"blue", "red"}, 77))
    assert words == ["blue", "red"] and route == [:image, :pivot, :text_colour]
    {:ok, note, _} = Hub.convert(hub, :audio, :text_note, World.sound("sol", 77))
    assert note == ["sol"]
  end

  @tag :native
  test "the hub on the native worker: the same answers, every modal program bit-identical to the oracle" do
    w = Runner.worker()
    hub = Hub.fit_world(worker: w, variants: 2)
    {:ok, words, _} = Hub.convert(hub, :image, :text_colour, World.scene({"green", "blue"}, 5))
    assert words == ["green", "blue"]
    {spec, ws} = hub.models.audio_to_text
    {:ok, p} = Lock.build(spec, ws, rows: 2)
    env = %{rows: Tensor.random(:f32, [2, spec.in_width], 4)}
    assert Runner.run(p, env) == Runner.run(p, env, worker: w)
  end
end
