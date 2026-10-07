defmodule Vapor.StudioMediaTest do
  @moduledoc """
  The studio's media operations against their references:

    * resizing is a certified program whose bits equal the BEAM's sparse
      evaluator; its values equal `torch.nn.functional.interpolate`
      (nearest-exact, bilinear, bicubic, area) and Pillow's Lanczos (float
      images) to binary32 rounding;
    * GIF: what we encode Pillow decodes to our pixels exactly, and what
      Pillow/ffmpeg encode we decode to Pillow's pixels exactly;
    * MJPEG-AVI frames = libjpeg (through Pillow) bit for bit; Y4M planes =
      ffmpeg's raw planes, and our RGB within 3 levels of ffmpeg's converter;
    * audio resampling keeps a tone (SNR) and removes what would alias.
  """
  use ExUnit.Case, async: false
  alias Vapor.Modal.{Audio, Image}
  alias Vapor.Media.{GIF, Video}
  alias Vapor.Studio
  alias Vapor.Studio.Resample
  import Vapor.TestHelpers

  @tmp Path.join(System.tmp_dir!(), "vapor-studio-media")

  setup_all do
    File.mkdir_p!(@tmp)
    w = if Vapor.Runtime.Substrates.binary("vapor-worker", "native"), do: elem(Vapor.Runtime.Worker.start_link(exec: worker_exec(:host)), 1)
    {:ok, worker: w}
  end

  defp f64(img), do: for(v <- Image.values(img), into: <<>>, do: <<v::float-64-little>>)
  defp u8(img), do: for(v <- Image.values(img), into: <<>>, do: <<round(min(1.0, max(0.0, v)) * 255)>>)

  @tag :native
  test "resize, blur, sharpen: the native program and the BEAM's sparse evaluator give the same bits", %{worker: w} do
    img = Image.noise(37, 23, 3, 5)
    for m <- Resample.methods(), {w2, h2} <- [{61, 40}, {17, 9}] do
      assert Resample.resize(img, w2, h2, m) == Resample.resize(img, w2, h2, m, worker: w), m
    end
    assert Resample.blur(img, 3, 1.5) == Resample.blur(img, 3, 1.5, worker: w)
    assert Resample.sharpen(img, 2, 1.0, 0.7) == Resample.sharpen(img, 2, 1.0, 0.7, worker: w)
    # a window (camera move) of the same shape reuses the compiled program, and still equals the BEAM
    rx = Resample.weights("lanczos", 37, 37, offset: 3.25, span: 20.5)
    ry = Resample.weights("lanczos", 23, 23, offset: 1.5, span: 12.0)
    assert Resample.separable(img, rx, ry) == Resample.separable(img, rx, ry, worker: w)
  end

  @tag :torch
  test "resize values = torch interpolate and Pillow's Lanczos, to binary32 rounding" do
    img = Image.noise(37, 23, 3, 9)
    File.write!(Path.join(@tmp, "in.bin"), f64(img))

    for {tag, {w2, h2}} <- [{"up", {61, 40}}, {"down", {17, 9}}], m <- Resample.methods() do
      File.write!(Path.join(@tmp, "#{tag}_#{m}.bin"), f64(Resample.resize(img, w2, h2, m)))
    end

    out = py!("""
    import numpy as np, torch, torch.nn.functional as F, sys
    from PIL import Image
    d = sys.argv[1]
    x = np.fromfile(d + '/in.bin', '<f8').reshape(23, 37, 3)
    t = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)[None]
    worst = 0.0
    for tag, (w2, h2) in [('up', (61, 40)), ('down', (17, 9))]:
        for m in ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos']:
            ours = np.fromfile(f'{d}/{tag}_{m}.bin', '<f8').reshape(h2, w2, 3)
            if m == 'lanczos':
                ref = np.stack([np.asarray(Image.fromarray(x[:, :, c].astype(np.float32), mode='F').resize((w2, h2), Image.LANCZOS)) for c in range(3)], -1)
            else:
                kw = {} if m in ('nearest', 'area') else {'align_corners': False}
                ref = F.interpolate(t, size=(h2, w2), mode='nearest-exact' if m == 'nearest' else m, **kw)[0].permute(1, 2, 0).numpy()
            worst = max(worst, float(np.abs(ours - ref).max()))
    print(worst)
    """, [@tmp])

    assert String.to_float(String.trim(out)) < 5.0e-6
  end

  @tag :pillow
  test "GIF: ours decodes in Pillow to our pixels; Pillow's GIFs decode here to Pillow's pixels" do
    frames = for t <- 0..3, do: Image.noise(30, 20, 3, t)
    gif = GIF.encode(frames, fps: 8)
    File.write!(Path.join(@tmp, "ours.gif"), gif)
    {:ok, back} = GIF.decode(gif)
    assert length(back.frames) == 4 and back.delays_cs == [13, 13, 13, 13]
    File.write!(Path.join(@tmp, "ours.u8"), Enum.map_join(back.frames, &u8/1))

    out = py!("""
    import numpy as np, sys
    from PIL import Image, ImageSequence
    d = sys.argv[1]
    im = Image.open(d + '/ours.gif')
    fr = np.stack([np.asarray(f.convert('RGB')) for f in ImageSequence.Iterator(im)])
    ours = np.frombuffer(open(d + '/ours.u8', 'rb').read(), np.uint8).reshape(fr.shape)
    print(int((fr == ours).all()))
    # Pillow's own animated GIF (adaptive palette, a different size per frame region)
    rng = np.random.default_rng(1)
    imgs = [Image.fromarray(rng.integers(0, 255, (21, 33, 3), dtype=np.uint8)).quantize(64) for _ in range(3)]
    imgs[0].save(d + '/pil.gif', save_all=True, append_images=imgs[1:], duration=120, loop=0)
    ref = np.stack([np.asarray(f.convert('RGB')) for f in ImageSequence.Iterator(Image.open(d + '/pil.gif'))])
    ref.tofile(d + '/pil.u8')
    print(ref.shape[0])
    """, [@tmp])

    [same, n] = String.split(String.trim(out), "\n")
    assert same == "1"
    {:ok, pil} = GIF.decode(File.read!(Path.join(@tmp, "pil.gif")))
    assert length(pil.frames) == String.to_integer(n)
    assert Enum.map_join(pil.frames, &u8/1) == File.read!(Path.join(@tmp, "pil.u8"))
  end

  @tag :ffmpeg
  @tag :pillow
  test "MJPEG-AVI = libjpeg frame by frame; Y4M planes = ffmpeg's; RGB within 3 levels of ffmpeg's converter" do
    run = fn args -> {_, 0} = System.cmd("ffmpeg", ["-v", "error", "-y" | args], cd: @tmp) end
    run.(~w(-f lavfi -i testsrc=size=64x48:rate=8 -frames:v 5 -c:v mjpeg -q:v 3 t.avi))
    File.rm_rf!(Path.join(@tmp, "fr")); File.mkdir_p!(Path.join(@tmp, "fr"))
    run.(~w(-i t.avi -c:v copy -f image2 fr/f%02d.jpg))
    {:ok, v} = Video.read(File.read!(Path.join(@tmp, "t.avi")))
    assert v.fps == 8.0 and length(v.frames) == 5
    File.write!(Path.join(@tmp, "avi.u8"), Enum.map_join(v.frames, &u8/1))

    run.(~w(-f lavfi -i testsrc=size=64x48:rate=8 -frames:v 5 -pix_fmt yuv420p t.y4m))
    run.(~w(-i t.y4m -pix_fmt rgb24 -f rawvideo t.rgb))
    {:ok, y} = Video.read(File.read!(Path.join(@tmp, "t.y4m")))
    File.write!(Path.join(@tmp, "y4m.u8"), Enum.map_join(y.frames, &u8/1))
    # our writer: ffmpeg reads its planes back unchanged
    File.write!(Path.join(@tmp, "ours.y4m"), Video.y4m(y))
    run.(~w(-i ours.y4m -f rawvideo ours.yuv))
    "YUV4MPEG2" <> _ = ours = File.read!(Path.join(@tmp, "ours.y4m"))
    planes = ours |> :binary.split("\n") |> List.last() |> String.split("FRAME\n", trim: true) |> Enum.join()
    assert File.read!(Path.join(@tmp, "ours.yuv")) == planes

    out = py!("""
    import numpy as np, glob, sys
    from PIL import Image
    d = sys.argv[1]
    a = np.frombuffer(open(d + '/avi.u8', 'rb').read(), np.uint8).reshape(5, 48, 64, 3)
    ref = np.stack([np.asarray(Image.open(f).convert('RGB')) for f in sorted(glob.glob(d + '/fr/*.jpg'))])
    y = np.frombuffer(open(d + '/y4m.u8', 'rb').read(), np.uint8).reshape(5, 48, 64, 3).astype(int)
    r = np.frombuffer(open(d + '/t.rgb', 'rb').read(), np.uint8).reshape(5, 48, 64, 3).astype(int)
    print(int((a == ref).all()), int(np.abs(y - r).max()))
    """, [@tmp])

    [avi_same, worst] = String.split(String.trim(out))
    assert avi_same == "1"
    assert String.to_integer(worst) <= 3
  end

  test "audio: resampling keeps a tone and removes what would alias; the spectrogram finds the tone" do
    tone = fn f, rate -> {:ok, %{audio: a}} = Studio.Nodes.Audio.run("audio.tone", %{}, %{frequency: f, seconds: 0.25, rate: rate, amplitude: 0.5, waveform: "sine"}, %{}); a end
    rs = fn a, r -> {:ok, %{audio: b}} = Studio.Nodes.Audio.run("audio.resample", %{audio: a}, %{rate: r}, %{}); b end

    # 1 kHz from 48 kHz to 16 kHz: the ideal tone, up to the filter's edge effects
    b = rs.(tone.(1000.0, 48_000), 16_000)
    ideal = tone.(1000.0, 16_000)
    mid = fn s -> Enum.slice(s, 400, 3200) end
    err = Enum.zip_with(mid.(b.samples), mid.(ideal.samples), &((&1 - &2) * (&1 - &2))) |> Enum.sum()
    sig = mid.(ideal.samples) |> Enum.map(&(&1 * &1)) |> Enum.sum()
    assert 10 * :math.log10(sig / err) > 50

    # 7 kHz at 48 kHz down to 8 kHz (Nyquist 4 kHz): it must not fold back to 1 kHz
    folded = rs.(tone.(7000.0, 48_000), 8000)
    rms = :math.sqrt((mid.(folded.samples) |> Enum.map(&(&1 * &1)) |> Enum.sum()) / 3200)
    assert 20 * :math.log10(rms / (0.5 / :math.sqrt(2))) < -40

    {:ok, %{image: spec}} = Studio.Nodes.Audio.run("audio.spectrogram", %{audio: tone.(2000.0, 8000)}, %{window: "256", hop: 128, floor_db: -80.0}, %{})
    # brightest row of the middle column ↔ 2 kHz (bin 64 of 128, counted from the bottom)
    col = div(spec.w, 2)
    {_, row} = Enum.max_by(for(r <- 0..(spec.h - 1), do: {Image.at(spec, col, r, 0), r}), &elem(&1, 0))
    assert spec.h - 1 - row == 64
  end

  test "a whole media workflow: image → camera move → GIF; tone → fade → WAV; all exported" do
    graph = %{"nodes" => %{
      "img" => %{"type" => "image.solid", "params" => %{"width" => 48, "height" => 32, "blue" => 0.7}},
      "mask" => %{"type" => "mask.rect", "params" => %{"width" => 48, "height" => 32, "x" => 8, "y" => 8, "w" => 16, "h" => 10}},
      "red" => %{"type" => "image.solid", "params" => %{"width" => 48, "height" => 32, "red" => 1.0}},
      "comp" => %{"type" => "image.composite", "inputs" => %{"destination" => ["img", "image"], "source" => ["red", "image"], "mask" => ["mask", "mask"]}},
      "cam" => %{"type" => "video.camera", "params" => %{"seconds" => 1.0, "fps" => 6.0, "zoom_end" => 2.0, "cx" => 0.3, "cy" => 0.4}, "inputs" => %{"image" => ["comp", "image"]}},
      "tone" => %{"type" => "audio.tone", "params" => %{"seconds" => 0.2, "rate" => 8000}},
      "fade" => %{"type" => "audio.fade", "inputs" => %{"audio" => ["tone", "audio"]}},
      "o1" => %{"type" => "studio.output", "params" => %{"name" => "clip"}, "inputs" => %{"value" => ["cam", "video"]}},
      "o2" => %{"type" => "studio.output", "params" => %{"name" => "sound"}, "inputs" => %{"value" => ["fade", "audio"]}}}}

    {:ok, r} = Studio.run(graph)
    clip = r.results["clip"]
    assert length(clip.frames) == 6
    # zooming towards the red rectangle: more red in the last frame than in the first
    red = fn f -> Enum.count(0..(f.w * f.h - 1), fn p -> elem(f.px, 3 * p) > 0.5 end) end
    assert red.(List.last(clip.frames)) > red.(hd(clip.frames))
    {"image/gif", "gif", gif} = Studio.Export.encode(clip)
    assert {:ok, %{frames: fs}} = GIF.decode(gif)
    assert length(fs) == 6
    {"audio/wav", "wav", wav} = Studio.Export.encode(r.results["sound"])
    assert {:ok, %Audio{rate: 8000}} = Audio.parse(wav)
  end
end
