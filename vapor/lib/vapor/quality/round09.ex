defmodule Vapor.Quality.Round09 do
  @moduledoc """
  Quality checks for the 0.9 round (the studio), in the suite's discipline:
  each check has a value, a **control** that a broken or naive
  implementation would produce, and a threshold that separates them.

  | check | value | control (must fail) |
  |---|---|---|
  | Stable Diffusion | the pipeline's images (txt2img × 3 samplers, img2img, inpainting) against diffusers' own on a shipped tiny checkpoint | the DPM++ image against the DDIM reference |
  | studio determinism | two cache-free runs of a template: one Merkle root | the same graph with another seed |
  | studio cache | nodes recomputed after editing one parameter | a run without the cache (every node) |
  | consistent upscaler | dB over Lanczos + projection on held-out text; |D(y) − x| | Lanczos: inconsistency ≥ 0.01 |
  | reinforcement learning | Q-learning's FrozenLake success = value iteration's; the shipped CartPole policy's return | always-left (0 %); an untrained policy |
  | 3D | marching-tetrahedra sphere: watertight, volume error | the same mesh with one face removed (not watertight) |
  | audio resampling | SNR of a 1 kHz tone, 48 → 16 kHz | nearest-sample decimation |
  | MCP | a run, the same run again (all cache), verify | verify with a wrong root |
  """
  alias Vapor.{Geom, RL, Studio}
  alias Vapor.Diffusion.Pipeline
  alias Vapor.Modal.Image
  alias Vapor.Studio.Resample
  alias Vapor.Vision.Upscale

  def run(opts \\ []) do
    w = Keyword.get(opts, :worker)
    checks = List.flatten([diffusion(w), studio(w), upscale(w), rl(w), geom(), audio(), mcp(w)])
    %{checks: checks}
  end

  defp check(name, value, control, threshold, pass), do: %{name: name, value: value, control: control, threshold: threshold, pass: pass}
  defp p(rel), do: Path.join(to_string(:code.priv_dir(:vapor)), rel)
  defp r(x, d \\ 3), do: Float.round(x * 1.0, d)

  # ------------------------------------------------------------ diffusion --

  defp diffusion(nil), do: []

  defp diffusion(w) do
    dir = p("quality/sd_tiny")
    {:ok, ref} = Vapor.Ingest.Safetensors.read(Path.join(dir, "reference.safetensors"))
    {:ok, pl} = Pipeline.load(dir)
    ids = fn k -> ref[k] |> Vapor.Tensor.to_floats() |> Enum.map(&trunc/1) end
    common = [prompt: ids.("ids"), negative: ids.("neg"), guidance: 3.0, worker: w]
    img = fn t -> (fn [h, wd, 3] -> Image.new(wd, h, 3, Vapor.Tensor.to_floats(t)) end).(t.shape) end
    err = fn name, %Image{} = a -> Enum.zip_with(Image.values(a), Vapor.Tensor.to_floats(ref[name]), &abs(&1 - &2)) |> Enum.max() end

    gen = fn k -> elem(Pipeline.generate(pl, common ++ [sampler: k, steps: 6, noise: ref["latents"], width: 16, height: 16]), 1).image end
    runs = Map.new([:ddim, :euler, :dpmpp_2m], &{&1, gen.(&1)})
    init = img.(ref["init"])
    {:ok, i2i} = Pipeline.generate(pl, common ++ [sampler: :ddim, steps: 10, noise: ref["noise"], init: init, strength: 0.6])
    [h, wd] = ref["mask"].shape
    {:ok, inp} = Pipeline.generate(pl, common ++ [sampler: :ddim, steps: 10, noise: ref["latents"], init: init, mask: Image.new(wd, h, 1, Vapor.Tensor.to_floats(ref["mask"]))])
    errs = [err.("txt2img_ddim", runs.ddim), err.("txt2img_euler", runs.euler), err.("txt2img_dpmpp_2m", runs.dpmpp_2m), err.("img2img", i2i.image), err.("inpaint", inp.image)]
    worst = Enum.max(errs)
    control = err.("txt2img_ddim", runs.dpmpp_2m)

    [check("Stable Diffusion = diffusers (tiny checkpoint: txt2img DDIM/Euler/DPM++ 2M, img2img, inpainting): largest pixel difference",
           Float.round(worst, 9), "#{Float.round(control, 4)} (DPM++ against the DDIM reference)", "< 1e-5, control > 1e-3", worst < 1.0e-5 and control > 1.0e-3)]
  end

  # --------------------------------------------------------------- studio --

  defp studio(w) do
    g = Enum.find(Studio.Templates.all(), &(&1.id == "mask")).graph
    {:ok, a} = Studio.run(g, worker: w, cache: nil)
    {:ok, b} = Studio.run(g, worker: w, cache: nil)
    other = put_in(g, ["nodes", "2", "params", "seed"], 100)
    {:ok, c} = Studio.run(other, worker: w, cache: nil)

    {:ok, cache} = Studio.Cache.start_link()
    {:ok, _} = Studio.run(g, worker: w, cache: cache)
    edited = put_in(g, ["nodes", "4", "params", "sigma"], 2.0)
    {:ok, e} = Studio.run(edited, worker: w, cache: cache)
    {:ok, full} = Studio.run(edited, worker: w, cache: nil)
    Agent.stop(cache)

    [check("studio: two cache-free runs of a template give one Merkle root", a.root == b.root, "#{c.root != a.root} (another seed changes the root)",
           "equal, control differs", a.root == b.root and c.root != a.root),
     check("studio: nodes recomputed after editing one parameter (feather σ)", length(e.executed), "#{length(full.executed)} (no cache)",
           "only the edited node and what depends on it (3 of 6)", length(e.executed) == 3 and length(full.executed) == 6 and e.root == full.root)]
  end

  # -------------------------------------------------------------- upscale --

  defp upscale(nil), do: []

  defp upscale(w) do
    {:ok, m} = Upscale.default()
    crop = fn n -> elem(Image.read(p("quality/upscale/#{n}.pgm")), 1) end

    gains =
      for n <- ~w(text_0 text_1) do
        hr = crop.(n)
        lr = Upscale.downsample(hr)
        lz = Resample.resize(lr, hr.w, hr.h, "lanczos", worker: w)
        Upscale.psnr(Upscale.upscale(lr, model: m, worker: w), hr) - Upscale.psnr(Upscale.project(lz, lr), hr)
      end

    lr = Upscale.downsample(crop.("camera"))
    inc = Upscale.inconsistency(Upscale.upscale(lr, model: m, worker: w), lr)
    lzinc = Upscale.inconsistency(Resample.resize(lr, 2 * lr.w, 2 * lr.h, "lanczos", worker: w), lr)
    g = Enum.min(gains)

    [check("consistent upscaler: dB over Lanczos + the same projection, held-out text (worst of 2)", r(g, 2), "0 (Lanczos + projection itself)", "≥ +1 dB", g >= 1.0),
     check("consistent upscaler: inconsistency max(D(y) − x) on a held-out photograph", inc, "#{r(lzinc, 4)} (Lanczos)", "≤ 1e-12, control ≥ 0.01", inc <= 1.0e-12 and lzinc >= 0.01)]
  end

  # ------------------------------------------------------------------- RL --

  defp rl(w) do
    {_, vi} = RL.value_iteration()
    {_, ql} = RL.q_learning(20_000)
    rate = RL.lake_success(ql, 1000)
    left = RL.lake_success(List.duplicate(0, 16), 1000)

    lake = check("RL: tabular Q-learning on FrozenLake finds value iteration's policy; its success rate", r(rate), "#{r(left)} (always left)",
                 "policy = optimum, 0.7–0.8, control 0", ql == vi and rate > 0.7 and rate < 0.8 and left == 0.0)

    cart =
      if w do
        {:ok, cp} = RL.load(:cartpole)
        s = RL.cartpole_score(cp.net, 20, 50_000, w)
        c = RL.cartpole_score(Vapor.Learn.new([4, 32, 32, 2], 12_345), 20, 50_000, w, greedy: false)
        [check("RL: the shipped CartPole policy (REINFORCE), mean return on 20 unseen starts", r(s, 1), "#{r(c, 1)} (untrained)", "≥ 450 of 500, control < 50", s >= 450 and c < 50)]
      else
        []
      end

    [lake | cart]
  end

  # ------------------------------------------------------------------- 3D --

  defp geom do
    m = Geom.sdf(fn {x, y, z} -> :math.sqrt(x * x + y * y + z * z) - 0.7 end, 32)
    v_true = 4 / 3 * :math.pi() * 0.7 ** 3
    err = abs(Geom.volume(m) - v_true) / v_true
    holed = %{m | faces: Enum.drop(m.faces, 3)}

    [check("3D: marching-tetrahedra sphere — watertight, relative volume error", r(err, 5), "#{Geom.watertight?(holed)} (one face removed: watertight?)",
           "watertight, < 0.5 %, control not watertight", Geom.watertight?(m) and err < 0.005 and not Geom.watertight?(holed))]
  end

  # ---------------------------------------------------------------- audio --

  defp audio do
    tone = fn f, rate -> {:ok, %{audio: a}} = Studio.Nodes.Audio.run("audio.tone", %{}, %{frequency: f, seconds: 0.25, rate: rate, amplitude: 0.5, waveform: "sine"}, %{}); a end
    {:ok, %{audio: b}} = Studio.Nodes.Audio.run("audio.resample", %{audio: tone.(1000.0, 48_000)}, %{rate: 16_000}, %{})
    ideal = tone.(1000.0, 16_000).samples
    src = tone.(1000.0, 48_000).samples |> List.to_tuple()
    # the control: the nearest source sample of each output instant, a third of a sample late (a misaligned grid)
    nearest = for i <- 0..(length(ideal) - 1), do: elem(src, min(tuple_size(src) - 1, round(i * 3 + 1)))
    snr = fn s -> mid = fn l -> Enum.slice(l, 400, 3200) end
      e = Enum.zip_with(mid.(s), mid.(ideal), &((&1 - &2) ** 2)) |> Enum.sum()
      10 * :math.log10((mid.(ideal) |> Enum.map(&(&1 * &1)) |> Enum.sum()) / max(e, 1.0e-300)) end
    {v, c} = {snr.(b.samples), snr.(nearest)}

    [check("audio: resampling a 1 kHz tone 48 → 16 kHz, SNR against the ideal tone (dB)", r(v, 1), "#{r(c, 1)} (decimation one source sample late)", "> 50 dB, control < 30 dB", v > 50 and c < 30)]
  end

  # ------------------------------------------------------------------ MCP --

  defp mcp(w) do
    dir = Path.join(System.tmp_dir!(), "vapor-q09-mcp-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)

    try do
      st = Vapor.MCP.Server.new(dir: dir, worker: w)
      g = Enum.find(Studio.Templates.all(), &(&1.id == "mask")).graph
      call = fn name, args -> elem(Vapor.MCP.Server.handle(%{"jsonrpc" => "2.0", "id" => 1, "method" => "tools/call", "params" => %{"name" => name, "arguments" => args}}, st), 0)["result"] end
      r1 = call.("studio_run", %{"graph" => g})
      r2 = call.("studio_run", %{"graph" => g})
      root = r1["structuredContent"]["root"]
      ok = call.("studio_verify", %{"graph" => g, "root" => root})
      bad = call.("studio_verify", %{"graph" => g, "root" => String.duplicate("0", 64)})
      again = r2["structuredContent"]["executed"]

      [check("MCP server: the same run again is all cache; verify accepts the root", "#{length(again)} recomputed, verified #{not ok["isError"]}",
             "#{not bad["isError"]} (a wrong root accepted?)", "0 recomputed, verified, control refused", again == [] and ok["isError"] == false and bad["isError"] == true)]
    after
      File.rm_rf!(dir)
    end
  end
end
