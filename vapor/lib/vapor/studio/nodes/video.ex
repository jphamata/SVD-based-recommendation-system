defmodule Vapor.Studio.Nodes.Video do
  @moduledoc """
  Video nodes: load (Y4M, MJPEG-AVI, GIF), stills and camera moves, frames,
  per-frame subgraphs, resize, trim, reverse, concatenate, crossfade,
  frame-rate change.

  `video.map` runs a subgraph (a workflow with a `studio.input` named
  `frame` and a `studio.output` named `frame`) on every frame, sharing one
  cache — identical frames are computed once — and the frames' receipts
  roll up into the node's. `video.camera` (a Ken Burns move) resamples a
  moving window with the same compiled program for every frame: the
  window enters as the resampling matrices, not as code.
  """
  @behaviour Vapor.Studio.Node
  alias Vapor.Modal.Image
  alias Vapor.Rejection
  alias Vapor.Studio
  alias Vapor.Studio.{Resample, Video}

  defp n(type, title, doc, inputs, outputs, params),
    do: {__MODULE__, %{type: type, version: 1, category: "video", title: title, doc: doc, inputs: inputs, outputs: outputs, params: params}}

  @impl true
  def nodes do
    [n("video.load", "Load video", "Y4M (any tool pipes it: ffmpeg -f yuv4mpegpipe), Motion-JPEG AVI, animated GIF — `data` or `path`.", [], [video: :video],
       [data: {:data, nil}, path: {:string, ""}]),
     n("video.still", "Still", "An image held for `seconds`.", [image: :image], [video: :video], [seconds: {:float, 0.04, 600.0, 2.0}, fps: {:float, 1.0, 120.0, 12.0}]),
     n("video.camera", "Camera move", "Ken Burns: zoom from `zoom_start` to `zoom_end` towards (cx, cy), same frame size, Lanczos.", [image: :image], [video: :video],
       [seconds: {:float, 0.1, 120.0, 2.0}, fps: {:float, 1.0, 120.0, 12.0}, zoom_start: {:float, 1.0, 16.0, 1.0}, zoom_end: {:float, 1.0, 16.0, 1.5},
        cx: {:float, 0.0, 1.0, 0.5}, cy: {:float, 0.0, 1.0, 0.5}]),
     n("video.frame", "Frame", "One frame as an image (index from 0; negative counts from the end).", [video: :video], [image: :image], [index: {:int, -100_000, 100_000, 0}]),
     n("video.map", "Map frames", "Run a subgraph on every frame: its `studio.input` and `studio.output` are both named \"frame\".", [video: :video], [video: :video],
       [graph: {:string, ""}]),
     n("video.resize", "Resize video", "Every frame to width × height (0 keeps the aspect).", [video: :video], [video: :video],
       [width: {:int, 0, 8192, 320}, height: {:int, 0, 8192, 0}, method: {:enum, Resample.methods(), "lanczos"}]),
     n("video.trim", "Trim video", "Frames [start, start + count) (count 0 = to the end).", [video: :video], [video: :video],
       [start: {:int, 0, 1_000_000, 0}, count: {:int, 0, 1_000_000, 0}]),
     n("video.reverse", "Reverse video", "Frames backwards.", [video: :video], [video: :video], []),
     n("video.concat", "Concatenate videos", "a, then b (one size; b takes a's rate).", [a: :video, b: :video], [video: :video], []),
     n("video.crossfade", "Crossfade", "The last `frames` of a dissolve into the first of b.", [a: :video, b: :video], [video: :video], [frames: {:int, 1, 10_000, 8}]),
     n("video.fps", "Frame rate", "Retime to `fps` (each output frame takes the source frame showing at its time).", [video: :video], [video: :video],
       [fps: {:float, 1.0, 240.0, 12.0}])]
  end

  @impl true
  def run("video.load", _, %{data: d, path: p}, ctx) do
    with {:ok, bytes} <- Studio.Nodes.Image.source(d, p, ctx), {:ok, v} <- Vapor.Media.Video.read(bytes), do: {:ok, %{video: v}}
  end

  def run("video.still", %{image: img}, %{seconds: s, fps: fps}, _), do: {:ok, %{video: %Video{fps: fps, frames: List.duplicate(img, max(1, round(s * fps)))}}}

  def run("video.camera", %{image: %Image{w: w, h: h} = img}, p, ctx) do
    nf = max(1, round(p.seconds * p.fps))

    # frames are independent: their windows (the costly part — Lanczos
    # weights through the correctly rounded sine) are computed in parallel,
    # each frame's bits unchanged
    frames =
      0..(nf - 1)
      |> Task.async_stream(fn f ->
        t = if nf == 1, do: 0.0, else: f / (nf - 1)
        # smoothstep easing, then the window: zoom z shows 1/z of each side
        e = t * t * (3.0 - 2.0 * t)
        z = p.zoom_start + (p.zoom_end - p.zoom_start) * e
        {sw, sh} = {w / z, h / z}
        x0 = min(max(p.cx * w - sw / 2, 0.0), w - sw)
        y0 = min(max(p.cy * h - sh / 2, 0.0), h - sh)
        rx = Resample.weights("lanczos", w, w, offset: x0, span: sw)
        ry = Resample.weights("lanczos", h, h, offset: y0, span: sh)
        Resample.separable(img, rx, ry, worker: ctx.worker)
      end, max_concurrency: System.schedulers_online(), timeout: :infinity, ordered: true)
      |> Enum.map(fn {:ok, fr} -> fr end)

    {:ok, %{video: %Video{fps: p.fps, frames: frames}}}
  end

  def run("video.frame", %{video: %Video{frames: fs}}, %{index: i}, _) do
    n = length(fs)
    k = if i < 0, do: n + i, else: i
    if k >= 0 and k < n, do: {:ok, %{image: Enum.at(fs, k)}}, else: {:error, Rejection.new({:frame, i}, "an index in 0..#{n - 1}", "the video has #{n} frames")}
  end

  def run("video.map", %{video: v}, %{graph: text}, ctx) do
    with {:ok, g} <- Studio.from_json(text),
         {:ok, _} <- Studio.validate(g) do
      {frames, _cache} =
        Enum.map_reduce(v.frames, %{}, fn f, cache ->
          case Studio.run(g, inputs: %{"frame" => f}, cache: cache, worker: ctx.worker, dir: ctx.dir) do
            {:ok, r} -> {r.results["frame"], r.cache}
            {:error, e} -> throw({:map_failed, e})
          end
        end)

      if Enum.all?(frames, &match?(%Image{}, &1)),
        do: {:ok, %{video: %{v | frames: frames}}},
        else: {:error, Rejection.new(:video_map, "a subgraph whose output \"frame\" is an image", "name its studio.output \"frame\"")}
    end
  catch
    {:map_failed, e} -> {:error, e}
  end

  def run("video.resize", %{video: %Video{frames: [f | _]} = v}, p, ctx) do
    {w2, h2} = case {p.width, p.height} do
      {0, 0} -> {f.w, f.h}
      {w, 0} -> {w, max(1, round(f.h * w / f.w))}
      {0, h} -> {max(1, round(f.w * h / f.h)), h}
      wh -> wh
    end

    rx = Resample.weights(p.method, f.w, w2)
    ry = Resample.weights(p.method, f.h, h2)
    {:ok, %{video: %{v | frames: Enum.map(v.frames, &Resample.separable(&1, rx, ry, worker: ctx.worker))}}}
  end

  def run("video.trim", %{video: v}, %{start: s, count: c}, _) do
    fs = Enum.drop(v.frames, s)
    fs = if c == 0, do: fs, else: Enum.take(fs, c)
    if fs == [], do: {:error, Rejection.new(:trim, "at least one frame left", "the video has #{length(v.frames)} frames")}, else: {:ok, %{video: %{v | frames: fs}}}
  end

  def run("video.reverse", %{video: v}, _, _), do: {:ok, %{video: %{v | frames: Enum.reverse(v.frames)}}}

  def run("video.concat", %{a: a, b: b}, _, _) do
    with :ok <- same_size(a, b), do: {:ok, %{video: %{a | frames: a.frames ++ b.frames}}}
  end

  def run("video.crossfade", %{a: a, b: b}, %{frames: k}, _) do
    with :ok <- same_size(a, b) do
      k = Enum.min([k, length(a.frames), length(b.frames)])
      {head, tail_a} = Enum.split(a.frames, length(a.frames) - k)
      {head_b, rest_b} = Enum.split(b.frames, k)

      mixed =
        Enum.zip(tail_a, head_b)
        |> Enum.with_index()
        |> Enum.map(fn {{fa, fb}, i} ->
          t = (i + 1) / (k + 1)
          %{fa | px: Enum.zip_with(Tuple.to_list(fa.px), Tuple.to_list(fb.px), &(&1 + (&2 - &1) * t)) |> List.to_tuple()}
        end)

      {:ok, %{video: %{a | frames: head ++ mixed ++ rest_b}}}
    end
  end

  def run("video.fps", %{video: v}, %{fps: fps}, _) do
    n = length(v.frames)
    dur = n / v.fps
    m = max(1, round(dur * fps))
    ft = List.to_tuple(v.frames)
    {:ok, %{video: %Video{fps: fps, frames: for(i <- 0..(m - 1), do: elem(ft, min(n - 1, trunc(Float.floor((i + 0.5) / fps * v.fps)))))}}}
  end

  defp same_size(a, b) do
    if Video.size(a) == Video.size(b) and hd(a.frames).c == hd(b.frames).c,
      do: :ok,
      else: {:error, Rejection.new(:video_size, "two videos of one frame size", "#{inspect(Video.size(a))} and #{inspect(Video.size(b))}: resize one")}
  end
end
