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
      to the same weights, bit for bit.
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
end
