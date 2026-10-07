defmodule Vapor.Quality.Round11 do
  @moduledoc """
  Quality checks for the 0.11 round, in the suite's discipline: a value, a
  **control** that a broken or naive implementation would produce, and a
  threshold that separates them. (Algorithm discovery and self-play left
  with their demonstrations in 0.16 — DIRETRIZ §19.)

  | check | value | control (must fail) |
  |---|---|---|
  | geometry | true theorems proved, symbolically and in exact rationals | false statements of the same shape: refuted by both |
  | conjectures | Euler line and the nine-point circle found unasked | trivially collinear triples: none reported |
  | homology | Betti numbers of the classics; torsion of Klein and RP² | GF(2) alone cannot tell torus from Klein |
  | persistence | one long H₁ bar for a noisy loop | a blob: no long bar |
  | science (11) | each experiment against its closed form or published value | each its own control |
  | living scene | sky at infinity, horizon, ground walkable, warm light at the hearth | — |
  | drawing rig | four limb ends where they were drawn | — |
  | direction | prompts → operations, unknown words reported | a nonsense word: reported, not guessed |
  | sketch | an askew rectangle squared, a circle a circle | the same fit without constraints stays askew |
  | floor plan | rooms of 12 and 20 m², doors of 0.9 and 1.0 m | — |
  | archives | intact archive verifies and replays to the same result | one flipped byte: caught |
  """
  import Bitwise
  alias Vapor.{Archive, Prove, Scene, Science, Sketch}

  def run(_opts \\ []) do
    %{checks: List.flatten([mathematics(), science(), scene(), sketch(), archives()])}
  end

  defp check(name, value, control, threshold, pass), do: %{name: name, value: value, control: control, threshold: threshold, pass: pass}
  defp p(rel), do: Path.join(to_string(:code.priv_dir(:vapor)), rel)
  defp picture(rel), do: elem(Vapor.Docs.Pictures.read(:png, File.read!(p(rel))), 1).image


  defp mathematics do
    ths = Prove.theorems() |> Enum.reject(fn {n, _} -> n =~ "simson" end) |> Map.new()
    right = Enum.count(ths, fn {n, t} -> {v, _} = Prove.prove(n); {c, _} = Prove.check(n); if t.true?, do: v == :proved and c == :holds, else: v == :refuted and c == :fails end)
    simson = {elem(Prove.check("simson_line"), 0), elem(Prove.check("false_simson_off_circle"), 0)}
    d = Prove.discover(Prove.theorems()["nine_point_circle"])
    col = MapSet.new(d.collinear, fn {_, a, b, c} -> MapSet.new([a, b, c]) end)
    cyc = MapSet.new(d.concyclic, fn {_, a, b, c, e} -> MapSet.new([a, b, c, e]) end)
    found = MapSet.member?(col, MapSet.new([:o, :g, :h])) and MapSet.member?(cyc, MapSet.new([:ma, :mb, :mc, :ha]))
    trivial = MapSet.member?(col, MapSet.new([:a, :mb, :hb]))
    betti = for c <- [:torus, :klein, :rp2], do: {c, Prove.betti(Prove.complex(c), :gf2).betti, Prove.betti(Prove.complex(c), :q).betti}
    loop = for i <- 1..40, do: (a = 2 * :math.pi() * i / 40; {:math.cos(a) + 0.05 * :math.sin(i * 7.0), :math.sin(a) + 0.05 * :math.cos(i * 3.0)})
    blob = for i <- 1..40, do: {Vapor.Sampler.uniform(3, i), Vapor.Sampler.uniform(4, i)}
    life = fn pts -> Prove.persistence(pts).h1 |> Enum.map(fn {b, e} -> if e == :inf, do: 10.0, else: e - b end) |> Enum.max(fn -> 0.0 end) end

    [check("geometry: classical theorems proved (numerator ≡ 0) and checked in exact rationals; their false twins refuted by both", "#{right}/#{map_size(ths)}", "Simson: #{inspect(simson)}",
           "all; Simson holds on the circle, fails off it", right == map_size(ths) and simson == {:holds, :fails}),
     check("conjecture and prove: Euler line and the nine-point circle found among #{d.candidates} candidates", found, trivial, "found; no trivially collinear triple reported", found and not trivial),
     check("homology: Betti numbers over GF(2) and ℚ — torsion tells the Klein bottle and RP² from the torus", inspect(betti), "GF(2): torus = Klein", "torus (1,2,1)/(1,2,1); Klein (1,2,1)/(1,1,0); RP² (1,1,1)/(1,0,0)",
           betti == [{:torus, [1, 2, 1], [1, 2, 1]}, {:klein, [1, 2, 1], [1, 1, 0]}, {:rp2, [1, 1, 1], [1, 0, 0]}]),
     check("persistent homology: a noisy loop's longest H₁ bar", life.(loop), life.(blob), "loop > 1; blob (the control) < 0.3", life.(loop) > 1 and life.(blob) < 0.3)]
  end

  defp science do
    for name <- Science.experiments() do
      r = Science.run(name)
      check("science: " <> r.name, r.value, r.control, r.threshold, r.pass)
    end
  end


  defp scene do
    s = Scene.analyze(picture("quality/scene/outdoor.png"))
    g = Scene.analyze(picture("quality/scene/guild.png"))
    "#" <> hex = g.light.color
    warm = String.to_integer(String.slice(hex, 0, 2), 16) - String.to_integer(String.slice(hex, 4, 2), 16)
    ok_s = hd(s.layers).kind == "sky" and abs(s.horizon - 175 / 420) < 0.06 and List.last(s.layers).kind == "ground" and Enum.sum(s.walk.cells) > 50
    rig = Scene.rig(picture("quality/scene/figure.png"))
    d = Scene.direct("uma noite de chuva forte com vento, três pessoas e vagalumes; orbite devagar, xyzzy")

    [check("living scene: sky at infinity, horizon (truth 0.417), ground walkable; the guild's light warm at the hearth", "horizon #{Float.round(s.horizon, 3)}, #{length(s.layers)} layers", nil,
           "sky first, |horizon − truth| < 0.06, ground last, walkable cells; light x ∈ (0.12, 0.28), red − blue > 60", ok_s and g.light.x > 0.12 and g.light.x < 0.28 and warm > 60),
     check("drawing rig: limb ends of a stick figure", rig.endpoints, nil, "4", rig.endpoints == 4),
     check("direction: a Portuguese prompt as operations; a nonsense word reported", length(d.ops), inspect(d.unknown), "night, heavy rain, wind, 3 people, fireflies, orbit, slow; unknown = [xyzzy]",
           %{"weather" => "rain", "intensity" => 1.0} in d.ops and %{"spawn" => "people", "count" => 3} in d.ops and d.unknown == ["xyzzy"])]
  end

  defp sketch do
    v = Sketch.vectorize(picture("quality/sketch/shapes.png"))
    raw = Sketch.vectorize(picture("quality/sketch/shapes.png"), snap: false)
    deg = fn l -> a = :math.atan2(l.y1 - l.y0, l.x1 - l.x0) * 180 / :math.pi(); if(a < 0, do: a + 180, else: a) end
    rect = fn ls -> Enum.filter(ls, fn l -> max(l.y0, l.y1) < 260 and max(l.x0, l.x1) < 300 end) end
    square = Enum.all?(rect.(v.lines), fn l -> deg.(l) in [0.0, 90.0] or abs(deg.(l) - 180) < 1.0e-9 end) and length(rect.(v.lines)) == 4
    askew = Enum.any?(rect.(raw.lines), fn l -> a = deg.(l); a > 0.5 and a < 89.5 end)
    plan = Sketch.plan(picture("quality/sketch/plan.png"))
    areas = plan.rooms |> Enum.map(& &1.area) |> Enum.sort()
    doors = plan.doors |> Enum.map(&(&1.width * plan.scale)) |> Enum.sort()

    [check("sketch: a rectangle drawn 2.2° askew comes back square and closed; the circle a circle", square and length(v.circles) == 1, askew, "square; without constraints (the control) still askew", square and length(v.circles) == 1 and askew),
     check("floor plan: rooms (m²) and doors (m) at the scale of the longest wall", "#{inspect(Enum.map(areas, &Float.round(&1, 2)))} · #{inspect(Enum.map(doors, &Float.round(&1, 2)))}", nil,
           "rooms 12 and 20 within 3 %; doors 0.9 and 1.0 within 0.08 m",
           length(areas) == 2 and Enum.zip(areas, [12.0, 20.0]) |> Enum.all?(fn {a, b} -> abs(a - b) / b < 0.03 end) and length(doors) == 2 and Enum.zip(doors, [0.9, 1.0]) |> Enum.all?(fn {a, b} -> abs(a - b) < 0.08 end))]
  end

  defp archives do
    a = Archive.pack("science", %{"experiment" => "heh_hartree_fock"}, elem(Archive.produce("science", %{"experiment" => "heh_hartree_fock"}), 1))
    ok = match?({:ok, _}, Archive.verify(a.zip))
    replay = Archive.replay(a.zip)
    # the control: the same archive with one byte of its result changed and re-zipped (a well-formed zip that lies)
    {:ok, files} = :zip.unzip(a.zip, [:memory])
    files = Enum.map(files, fn {n, b} -> if n == ~c"result.json", do: {n, flip(b)}, else: {n, b} end)
    {:ok, {_, forged}} = :zip.create(~c"x.zip", files, [:memory])
    bad = Archive.verify(forged)

    [check("archives: an archive verifies and replays to the same result; one changed byte, re-zipped, is caught", inspect(replay), inspect(bad, limit: 3), "{:ok, :same}; the forged one {:tampered, [result.json]}",
           ok and replay == {:ok, :same} and bad == {:error, {:tampered, ["result.json"]}})]
  end

  defp flip(<<b, rest::binary>>), do: <<bxor(b, 1), rest::binary>>
end
