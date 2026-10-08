defmodule Vapor.UpscaleTest do
  @moduledoc """
  The consistent upscaler (`Vapor.Vision.Upscale`), shipped in priv/upscale:

    * **consistency by construction** — shrinking the result gives the input
      back (|D(y) − x| at binary64 rounding), grey and colour; the control:
      Lanczos and bicubic are not consistent;
    * **quality on held-out images** (fonts and photographs never trained
      on): on text and graphics it beats Lanczos with the same projection
      by more than 1 dB; on photographs it stays within half a dB of it;
      it never loses to plain Lanczos;
    * **training is reproducible**: the oracle and the native worker train
      to the same weights, bit for bit;
    * **temporal upscaling** keeps every frame consistent with its own input
      whatever its history holds, gains only when frames sample new sub-pixel
      positions (the whole-pixel pan is the control), and rejects the ghost
      of an object that has gone, which a naive blend keeps.
  """
  use ExUnit.Case, async: false
  alias Vapor.Modal.Image
  alias Vapor.Studio.Resample
  alias Vapor.Vision.Upscale
  import Vapor.TestHelpers

  @crops Path.expand("../../priv/quality/upscale", __DIR__)

  setup_all do
    w = if Vapor.Runtime.Substrates.binary("vapor-worker", "native"), do: elem(Vapor.Runtime.Worker.start_link(exec: worker_exec(:host)), 1)
    {:ok, model} = Upscale.default()
    {:ok, worker: w, model: model}
  end

  defp crop(name), do: elem(Image.read(Path.join(@crops, name <> ".pgm")), 1)

  @tag :native
  test "consistency: D(upscale(x)) = x; Lanczos and bicubic are not consistent (the control)", %{worker: w, model: m} do
    lr = Upscale.downsample(crop("camera"))
    up = Upscale.upscale(lr, model: m, worker: w)
    assert Upscale.inconsistency(up, lr) < 1.0e-12
    assert Upscale.inconsistency(Resample.resize(lr, 2 * lr.w, 2 * lr.h, "lanczos", worker: w), lr) > 0.01
    assert Upscale.inconsistency(Resample.resize(lr, 2 * lr.w, 2 * lr.h, "bicubic", worker: w), lr) > 0.01

    # colour: the luma by the network, the chroma by Lanczos, all projected (D commutes with the colour transform)
    rgb = Image.scene(48, 32, seed: 2)
    small = Upscale.downsample(rgb)
    big = Upscale.upscale(small, model: m, worker: w)
    assert {big.w, big.h, big.c} == {48, 32, 3}
    assert Upscale.inconsistency(big, small) < 1.0e-9

    # ×4 is ×2 twice: consistent with the input at the 4×4 scale too
    x4 = Upscale.upscale(small, model: m, worker: w, factor: 4)
    assert Upscale.inconsistency(Upscale.downsample(x4), small) < 1.0e-9
  end

  @tag :native
  test "held-out quality: text and graphics +1 dB over Lanczos + consistency; photographs within 0.5 dB; never below Lanczos", %{worker: w, model: m} do
    for name <- ~w(text_0 text_1 shepp_logan_phantom camera chelsea) do
      hr = crop(name)
      lr = Upscale.downsample(hr)
      lz = Resample.resize(lr, hr.w, hr.h, "lanczos", worker: w)
      v = Upscale.psnr(Upscale.upscale(lr, model: m, worker: w), hr)
      lzp = Upscale.psnr(Upscale.project(lz, lr), hr)
      assert v > Upscale.psnr(lz, hr), name

      if name in ~w(camera chelsea),
        do: assert(v > lzp - 0.5, "#{name}: #{v} vs #{lzp}"),
        else: assert(v > lzp + 1.0, "#{name}: #{v} vs #{lzp}")
    end
  end

  @tag :native
  test "training is reproducible: the oracle and the native worker reach the same weights", %{worker: w} do
    imgs = [crop("text_0"), crop("camera")]
    {a, ia} = Upscale.train(imgs, worker: w, steps: 4, chunk: 2, batch: 32, per_image: 64)
    {b, ib} = Upscale.train(imgs, steps: 4, chunk: 2, batch: 32, per_image: 64)
    assert Vapor.Learn.digest(a) == Vapor.Learn.digest(b)
    assert ia.losses == ib.losses and ia.data_digest == ib.data_digest
  end

  test "the shipped model's receipt matches its weights" do
    dir = Path.expand("../../priv/upscale", __DIR__)
    {:ok, %{net: net, config: cfg}} = Upscale.load(dir)
    assert cfg["weights_sha256"] == Vapor.Learn.digest(net)
    assert cfg["kind"] == "vapor-consistent-upscaler" and cfg["steps"] > 0 and is_binary(cfg["data_sha256"])
  end

  @tag :native
  test "the studio node: ×2 and ×4, odd sides, three methods", %{worker: w} do
    img = Image.scene(21, 15, seed: 5)
    for {f, m} <- [{"2", "vapor"}, {"4", "lanczos+consistency"}, {"2", "lanczos"}] do
      {:ok, %{image: out}} = Vapor.Studio.Nodes.Vision.run("image.upscale", %{image: img}, %{factor: f, method: m}, %{worker: w})
      k = String.to_integer(f)
      assert {out.w, out.h} == {22 * k, 16 * k}
    end
  end

  # ------------------------------------------------------------- temporal --

  # a striped world with a bright disc, seen through a 48 × 48 window
  defp world(x, y, disc? \\ true) do
    stripes = if rem(div(x, 3) + div(y, 7), 2) == 0, do: 0.8, else: 0.2
    v = if disc? and (x - 70) ** 2 + (y - 22) ** 2 < 120, do: 0.95, else: stripes
    min(1.0, 0.15 + 0.7 * v + 0.1 * :math.sin(x / 5.0))
  end

  defp view(f), do: %Image{w: 48, h: 48, c: 1, px: (for y <- 0..47, x <- 0..47, do: f.(x, y)) |> List.to_tuple()}

  defp film(frames, motion, sp) do
    lrs = Enum.map(frames, &Upscale.downsample/1)
    {ys, _} = Enum.map_reduce(lrs, nil, fn lr, prev -> y = Upscale.temporal(lr, prev, motion, spatial: sp.(lr)); {y, y} end)
    {ys, lrs}
  end

  @tag :native
  test "temporal: the gain comes from new sub-pixel samples, every frame stays consistent, wrong vectors cost quality not consistency", %{worker: w, model: m} do
    sp = &Upscale.upscale(&1, worker: w, model: m)

    gain = fn s ->
      frames = for t <- 0..5, do: view(fn x, y -> world(x + s * t, y) end)
      {ys, lrs} = film(frames, {-s, 0}, sp)
      assert Enum.zip_with(ys, lrs, &Upscale.inconsistency/2) |> Enum.max() < 1.0e-12
      t = ys |> Enum.zip_with(frames, &Upscale.psnr/2) |> Enum.drop(2) |> Enum.sum()
      single = lrs |> Enum.map(sp) |> Enum.zip_with(frames, &Upscale.psnr/2) |> Enum.drop(2) |> Enum.sum()
      (t - single) / 4
    end

    # half an input pixel per frame: each frame samples positions the last did not
    assert gain.(1) > 0.5
    # a whole input pixel per frame: the same samples again; nothing to gain (the control)
    assert abs(gain.(2)) < 0.25

    frames = for t <- 0..3, do: view(fn x, y -> world(x + t, y) end)
    {ys, lrs} = film(frames, {7, -5}, sp)
    assert Enum.zip_with(ys, lrs, &Upscale.inconsistency/2) |> Enum.max() < 1.0e-12
  end

  @tag :native
  test "temporal: the ghost of an object that has gone is rejected; a naive blend keeps it and contradicts the input", %{worker: w, model: m} do
    sp = &Upscale.upscale(&1, worker: w, model: m)
    {before, now} = {view(fn x, y -> world(x + 40, y) end), view(fn x, y -> world(x + 40, y, false) end)}
    {lr0, lr1} = {Upscale.downsample(before), Upscale.downsample(now)}
    y0 = sp.(lr0)
    single = sp.(lr1)
    ours = Upscale.temporal(lr1, y0, {0, 0}, spatial: single)
    naive = %{y0 | px: Enum.zip_with(Tuple.to_list(single.px), Tuple.to_list(y0.px), fn a, b -> 0.25 * a + 0.75 * b end) |> List.to_tuple()}
    disc = for y <- 0..47, x <- 0..47, (x + 40 - 70) ** 2 + (y - 22) ** 2 < 120, do: y * 48 + x
    err = fn im -> disc |> Enum.map(&abs(elem(im.px, &1) - elem(now.px, &1))) |> Enum.max() end

    assert err.(naive) > 0.4 and Upscale.inconsistency(naive, lr1) > 0.3
    assert err.(ours) <= err.(single) + 0.02 and Upscale.inconsistency(ours, lr1) < 1.0e-12
  end

end
