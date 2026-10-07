defmodule Vapor.Console.Lab11 do
  @moduledoc """
  The console's laboratories for the 0.11 round, each a function from a
  small request to a JSON-ready map (docs/CONSOLE.md): the living scene
  (analysis, rig, direction, standalone export), sketch → drawing / plan,
  mathematics (proofs, conjectures, homology), algorithm discovery, the
  science laboratory, self-play and domain randomisation — and saving any
  of them as a verifiable archive (`Vapor.Archive`).
  """
  alias Vapor.{Archive, Discover, Games, Prove, Scene, Science, Sketch}

  defp memo(key, f) do
    case :persistent_term.get({__MODULE__, key}, nil) do
      nil ->
        :global.trans({{__MODULE__, :lab}, self()}, fn ->
          case :persistent_term.get({__MODULE__, key}, nil) do
            nil -> v = f.(); :persistent_term.put({__MODULE__, key}, v); v
            v -> v
          end
        end, [node()], :infinity)

      v ->
        v
    end
  end

  @samples %{"outdoor" => "scene/outdoor.png", "guild" => "scene/guild.png", "figure" => "scene/figure.png", "shapes" => "sketch/shapes.png", "plan" => "sketch/plan.png"}

  # a sample picture by name (the quality fixtures), or the uploaded one
  defp picture("sample:" <> name, _data) do
    case Map.fetch(@samples, name) do
      {:ok, rel} ->
        bytes = File.read!(Path.join([to_string(:code.priv_dir(:vapor)), "quality", rel]))
        {:ok, %{image: img}} = Vapor.Docs.Pictures.read(:png, bytes)
        {:ok, img}

      :error ->
        {:error, "sample: one of #{Enum.join(Map.keys(@samples), ", ")}"}
    end
  end

  defp picture(name, data) do
    with {:ok, bytes} <- Base.decode64(data, ignore: :whitespace) |> ok("data: base64"),
         {:ok, %{image: img}} <- Vapor.Docs.Pictures.read(Vapor.Docs.sniff(name, bytes), bytes) |> ok("a PNG, JPEG or PPM picture") do
      {:ok, img}
    else
      {:error, why} -> {:error, why}
      # read, but not decoded (too many pixels, or a format only described): said, not crashed on
      {:ok, _meta} -> {:error, "a PNG, JPEG or PPM picture of at most 4 megapixels"}
    end
  end

  defp ok({:ok, v}, _), do: {:ok, v}
  defp ok(:error, why), do: {:error, why}
  defp ok({:error, _}, why), do: {:error, why}

  # ------------------------------------------------------------------ scene

  def scene_analyze(name, data) do
    with {:ok, img} <- picture(name, data) do
      t0 = System.monotonic_time(:millisecond)
      s = Scene.analyze(img)
      {:ok, Map.put(s, :ms, System.monotonic_time(:millisecond) - t0)}
    end
  end

  def scene_rig(name, data) do
    with {:ok, img} <- picture(name, data), do: {:ok, Scene.rig(img)}
  end

  def scene_direct(prompt, known \\ [])

  def scene_direct(prompt, known) when is_binary(prompt) and byte_size(prompt) <= 2000 do
    known = if is_list(known), do: known |> Enum.filter(&is_binary/1) |> Enum.take(200), else: []
    {:ok, Scene.direct(prompt, known)}
  end

  def scene_direct(_, _), do: {:error, "prompt: a string up to 2000 bytes"}

  def scene_export(scene, title) when is_map(scene), do: {:ok, Scene.standalone(Vapor.JSON.encode(scene), title || "vapor — cena viva")}
  def scene_export(_, _), do: {:error, "scene: the scene object"}

  # ----------------------------------------------------------------- sketch

  def sketch(name, data, mode, opts) do
    with {:ok, img} <- picture(name, data) do
      # the sketch itself, as the vectoriser saw it (for the side-by-side)
      seen = "data:image/png;base64," <> Base.encode64(Vapor.Modal.Image.png(Vapor.Scene.fit(img, 640)))
      with {:ok, r} <- sketch_of(img, mode, opts), do: {:ok, Map.put(r, :image, seen)}
    end
  end

  defp sketch_of(img, "plan", opts), do: do_plan(img, opts)

  defp sketch_of(img, _mode, opts) do
    v = Sketch.vectorize(img, snap: opts["snap"] != false)
    {:ok, Map.merge(v, %{mode: "vector", svg: Sketch.svg(v), dxf: Sketch.dxf(v, opts["scale"] || 1.0)})}
  end

  defp do_plan(img, opts) do
    p = Sketch.plan(img, plan_opts(opts))
    mesh = %{vertices: p.mesh.vertices, faces: p.mesh.faces, colors: p.mesh.colors}

    {:ok, %{mode: "plan", w: p.w, h: p.h, walls: p.walls, doors: Enum.map(p.doors, &Map.put(&1, :metres, &1.width * p.scale)),
            rooms: Enum.map(p.rooms, fn r -> %{polygon: Enum.map(r.polygon, &Tuple.to_list/1), area: r.area} end), scale: p.scale, glb: p.glb, mesh: mesh}}
  end


  defp plan_opts(o) do
    [longest: o["longest"] || 8.0, height: o["height"] || 2.7] ++ if(o["scale"], do: [scale: o["scale"]], else: [])
  end

  # ------------------------------------------------------------ mathematics

  def theorems do
    memo(:theorems, fn ->
      for {name, t} <- Enum.sort(Prove.theorems()), do: %{name: name, doc: t.doc, true: t.true?, claim: Tuple.to_list(t.claim) |> Enum.map(&to_string/1)}
    end)
  end

  def prove(name) do
    case Map.fetch(Prove.theorems(), name) do
      {:ok, t} ->
        {us, verdict} = :timer.tc(fn -> if name =~ "simson", do: {:skipped_symbolic, %{}}, else: Prove.prove(name) end)
        {check, k} = Prove.check(name)
        values = for i <- 0..(length(t.params) - 1), do: Enum.at([0.9, 0.35, 0.55, 0.2, -0.6, 0.45, 1.3], i, 0.3)
        pts = Prove.coordinates(t, values)
        {kind, cert} = verdict

        {:ok, %{name: name, doc: t.doc, expected: t.true?, verdict: kind, certificate: cert, check: check, checked_points: k, ms: div(us, 1000),
                points: Map.new(pts, fn {p, {x, y}} -> {p, [x, y]} end), claim: Tuple.to_list(t.claim) |> Enum.map(&to_string/1), construction: Enum.map(t.points, fn {p, spec} -> [to_string(p), inspect(spec)] end)}}

      :error ->
        {:error, "theorem: one of #{Enum.join(Map.keys(Prove.theorems()), ", ")}"}
    end
  end

  def conjectures do
    memo(:conjectures, fn ->
      fig = Prove.theorems()["nine_point_circle"]
      r = Prove.discover(fig)
      pts = Prove.coordinates(fig, [0.9, 0.35, 0.55])
      %{collinear: Enum.map(r.collinear, &(&1 |> Tuple.to_list() |> Enum.map(fn x -> to_string(x) end))), concyclic: Enum.map(r.concyclic, &(&1 |> Tuple.to_list() |> Enum.map(fn x -> to_string(x) end))),
        candidates: r.candidates, survivors: r.survivors, points: Map.new(pts, fn {p, {x, y}} -> {p, [x, y]} end)}
    end)
  end

  def homology do
    memo(:homology, fn ->
      for c <- [:sphere, :torus, :klein, :rp2, :mobius, :disk] do
        a = Prove.betti(Prove.complex(c), :gf2)
        b = Prove.betti(Prove.complex(c), :q)
        %{complex: c, gf2: a.betti, q: b.betti, euler: a.euler, counts: a.counts, torsion: a.betti != b.betti}
      end
    end)
  end

  def persistence(kind, seed) do
    pts =
      case kind do
        "loop" -> for i <- 1..40, do: (a = 2 * :math.pi() * i / 40; {:math.cos(a) + 0.08 * (Vapor.Sampler.uniform(seed, i) - 0.5), :math.sin(a) + 0.08 * (Vapor.Sampler.uniform(seed + 1, i) - 0.5)})
        "two_loops" -> for i <- 1..50, do: (a = 2 * :math.pi() * i / 25; c = if(i <= 25, do: -1.2, else: 1.2); {c + 0.8 * :math.cos(a) + 0.06 * (Vapor.Sampler.uniform(seed, i) - 0.5), 0.8 * :math.sin(a) + 0.06 * (Vapor.Sampler.uniform(seed + 1, i) - 0.5)})
        _ -> for i <- 1..40, do: {Vapor.Sampler.uniform(seed, i) * 2 - 1, Vapor.Sampler.uniform(seed + 1, i) * 2 - 1}
      end

    r = Prove.persistence(pts)
    fin = fn l -> Enum.map(l, fn {b, d} -> [b, if(d == :inf, do: nil, else: d)] end) end
    %{points: Enum.map(pts, &Tuple.to_list/1), h0: fin.(r.h0), h1: fin.(r.h1)}
  end

  # -------------------------------------------------------------- discovery

  def discover("network", n) when n in 3..9 do
    memo({:net, n}, fn ->
      {us, r} = :timer.tc(fn -> Discover.network(n) end)
      c = Discover.random_network(n, 1)
      %{task: "network", n: n, net: Enum.map(r.net, &Tuple.to_list/1), size: r.size, depth: r.depth, sorts: r.sorts, known: r.known,
        control: %{size: c.size, depth: c.depth}, ms: div(us, 1000)}
    end)
  end

  def discover("matmul", _) do
    memo(:matmul, fn ->
      {us, {:ok, s}} = :timer.tc(fn -> Discover.matmul(2, 7, tries: 30, seed: 1) end)
      {:error, {:not_found, k6}} = Discover.matmul(2, 6, tries: 10, seed: 1)
      counts = Discover.recursive_counts(s, 8)
      %{task: "matmul", u: s.u, v: s.v, w: s.w, mults: s.mults, adds: s.adds, tries: s.tries, verified: Discover.bilinear_ok?(2, s), rank6_tries: k6,
        counts: Enum.map(counts, &Map.take(&1, [:n, :mults, :adds])), naive: for(%{n: n} <- counts, do: n * n * n),
        class: Discover.complexity(for(%{n: n, mults: m} <- counts, n >= 2, do: {n, m})).class, ms: div(us, 1000)}
    end)
  end

  def discover("synth", spec) do
    if Map.has_key?(Discover.specs(), spec) do
      memo({:synth, spec}, fn ->
        {us, r} = :timer.tc(fn -> Discover.synthesize(spec) end)
        case r do
          {:ok, x} -> %{task: "synth", spec: spec, doc: Discover.specs()[spec].doc, text: x.text, ops: x.ops, width: x.width, explored: x.explored, verified: x.verified, ms: div(us, 1000)}
          {:error, e} -> %{task: "synth", spec: spec, error: inspect(e)}
        end
      end)
    else
      %{error: "spec: one of #{Enum.join(Map.keys(Discover.specs()), ", ")}"}
    end
  end

  def discover(_, _), do: %{error: "task: network (n 3–9), matmul or synth"}

  # ---------------------------------------------------------------- science

  def science(name) do
    if name in Science.experiments() do
      memo({:science, name}, fn -> {us, r} = :timer.tc(fn -> Science.run(name) end); Map.put(r, :ms, div(us, 1000)) end)
    else
      %{error: "experiment: one of #{Enum.join(Science.experiments(), ", ")}"}
    end
  end

  # ------------------------------------------------------------------ games

  # "alphazero" is the 0.11 name of the task, still accepted
  def games("alphazero"), do: games("selfplay")

  def games("selfplay") do
    memo(:selfplay, fn ->
      {:ok, n, recipe} = Games.load()
      n0 = Games.net(1)
      # every optimal line of the perfect player, both sides: exhaustive, not sampled
      curve =
        for s <- [8, 16, 32, 64, 128] do
          {a, b} = {Games.versus_every_optimal_line(n, s), Games.versus_every_optimal_line(n0, s)}
          %{sims: s, trained: 100 * a.losses / a.lines, untrained: 100 * b.losses / b.lines, trained_lines: [a.losses, a.lines], untrained_lines: [b.losses, b.lines]}
        end

      %{recipe: recipe, digest: Games.digest(n), curve: curve, versus_random: Games.versus_random(n, 32, 20)}
    end)
  end

  def games("randomization") do
    memo(:randomization, fn ->
      runs = for seed <- 1..3 do
        a = Games.train_cartpole(seed: seed)
        b = Games.train_cartpole(seed: seed, randomize: true)
        %{seed: seed, nominal: Games.robustness(a.w), randomized: Games.robustness(b.w), curve_nominal: a.curve, curve_randomized: b.curve}
      end
      %{runs: runs, shifted: Enum.map(Games.shifted(), &Map.take(&1, [:length, :masspole, :masscart, :force]))}
    end)
  end

  @doc "The agent's move on a board (list of 9: 1 X, −1 O, 0 empty), with the search's visit counts."
  def games_move(board, sims) when is_list(board) and length(board) == 9 and is_integer(sims) do
    if Enum.all?(board, &(&1 in [-1, 0, 1])), do: games_move_ok(board, sims), else: {:error, "board: 9 cells of 1, -1 or 0"}
  end

  def games_move(_, _), do: {:error, "board: 9 cells of 1, -1 or 0"}

  defp games_move_ok(board, sims) do
    {:ok, n, _} = Games.load()
    b = List.to_tuple(board)
    case Games.winner(b) do
      nil ->
        visits = Games.mcts(n, b, min(max(sims, 1), 400))
        {move, _} = Enum.max_by(visits, fn {a, c} -> {c, -a} end)
        {:ok, %{move: move, visits: Map.new(visits, fn {a, c} -> {to_string(a), c} end), value: Games.solve(b)}}

      w ->
        {:ok, %{over: w}}
    end
  end



  def archive(kind, recipe, result) when is_binary(kind) and byte_size(kind) <= 64 do
    # a replayable kind is packed with the result its producer computes now, so replay compares like with like
    result =
      if kind in Archive.replayable() and is_map(recipe) do
        case Archive.produce(kind, recipe) do
          {:ok, r} -> r
          _ -> result
        end
      else
        result
      end

    a = Archive.pack(kind, recipe || %{}, result || %{})
    # signed with the operator's key when the server has one (VAPOR_ARCHIVE_KEY: a `mix vapor.audit keygen` file)
    {zip, signed} =
      case System.get_env("VAPOR_ARCHIVE_KEY") do
        nil -> {a.zip, nil}
        path -> (k = Archive.load_key(path); {:ok, z} = Archive.sign(a.zip, k); {z, Vapor.Certificate.key_id(k.public)})
      end
    {:ok, %{id: a.id, name: "vapor-#{kind |> String.replace(~r/[^a-z0-9._-]/, "-")}-#{String.slice(a.id, 0, 12)}.zip", data: Base.encode64(zip), replayable: kind in Archive.replayable(), signed_by: signed}}
  end

  def archive(_, _, _), do: {:error, "kind: a short string"}

  def archive_check(data) do
    with {:ok, zip} <- Base.decode64(data, ignore: :whitespace) |> ok("data: base64") do
      case Archive.verify(zip) do
        {:ok, b} ->
          replay = if b.manifest["kind"] in Archive.replayable(), do: inspect(Archive.replay(zip)), else: "not replayable (#{b.manifest["kind"]})"
          {:ok, %{id: b.id, kind: b.manifest["kind"], vapor: b.manifest["vapor"], files: Map.keys(b.manifest["files"]), recipe: b.manifest["recipe"], intact: true, replay: replay,
                  signature: b.signature}}

        {:error, why} ->
          {:ok, %{intact: false, why: inspect(why)}}
      end
    end
  end
end
