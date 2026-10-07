defmodule Vapor.OpusTest do
  @moduledoc """
  The 0.15 desks through their three surfaces: the console's API
  (`Vapor.Console.Lab15`, every shelf example run), the terminal (`vapor
  rebis|aludel|tabula|cupel|amalgam`, exit statuses with their meaning,
  JSON in pipes) and bounded, refused inputs.
  """
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO
  alias Vapor.Console.Lab15

  setup do
    System.put_env("VAPOR_TTY", "1")
    System.put_env("NO_COLOR", "1")
    on_exit(fn -> System.delete_env("VAPOR_TTY"); System.delete_env("NO_COLOR") end)
    dir = Path.join(System.tmp_dir!(), "vapor-opus-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    {:ok, dir: dir}
  end

  defp req(ex), do: ex |> Map.drop([:id, :title, :about]) |> Map.new(fn {k, v} -> {to_string(k), v} end)

  test "every shelf example of every desk runs and says what it found" do
    expect = %{
      "adders" => "equivalent", "adders16" => "equivalent", "trojan" => "different", "multiplier" => "proved",
      "motzkin" => "certified", "motzkin0" => "exhausted", "refute" => "refuted", "oscillator" => "proved", "synth" => "proved",
      "sale" => "antinomies"
    }

    for {desk, exs} <- Lab15.info(), ex <- exs do
      assert {:ok, r} = apply(Lab15, desk, [req(ex)]), "#{desk}/#{ex.id}"
      if Map.has_key?(expect, ex.id), do: assert(r.verdict == expect[ex.id], "#{desk}/#{ex.id}: #{inspect(r[:verdict])}")
      # every result is JSON-able, as the console and MCP send it
      assert is_binary(Vapor.JSON.encode(Vapor.Main.jsonable(r)))
    end
  end

  test "the drawings the console needs: the subdivision's leaves, a vector field, the barrier's values" do
    {:ok, r} = Lab15.aludel(%{"op" => "decide", "vars" => "x, y", "poly" => "x^4*y^2 + x^2*y^4 - 3*x^2*y^2 + 1001/1000", "box" => [["-2", "2"], ["-2", "2"]], "sense" => "pos"})
    assert r.replayed and length(r.leaves) > 10
    assert Enum.all?(r.leaves, &match?([_, _, _, _], &1))
    [ex] = Enum.filter(Lab15.info().aludel, &(&1.id == "oscillator"))
    {:ok, b} = Lab15.aludel(req(ex))
    assert length(b.plot.arrows) == 15 * 15 and length(b.plot.b.values) == 41 * 41
  end

  test "inputs are bounded and refused with a reason, never crash" do
    assert {:error, _} = Lab15.rebis(%{"op" => "frobnicate", "a" => ""})
    assert {:error, msg} = Lab15.rebis(%{"op" => "equivalent", "a" => "input a\noutput y\ny = a & b\n", "b" => ""})
    assert msg =~ "a:"
    assert {:error, _} = Lab15.rebis(%{"op" => "equivalent", "a" => String.duplicate("x", 300_000), "b" => ""})
    assert {:error, _} = Lab15.aludel(%{"op" => "decide", "vars" => "x y z w v", "poly" => "x"})
    assert {:error, _} = Lab15.aludel(%{"op" => "decide", "vars" => "x", "poly" => "x", "box" => [["1", "0"]]})
    assert {:error, _} = Lab15.aludel(%{"op" => "decide", "vars" => "x", "poly" => "x", "box" => "0..1"})
    assert {:error, _} = Lab15.tabula(%{"text" => "X: p wants go\n"})
    assert {:error, _} = Lab15.amalgam(%{"numbers" => "1 2 three"})
    assert {:error, _} = Lab15.rebis(%{"op" => "anf", "a" => Vapor.Rebis.Gen.ripple(9)})
    assert {:ok, %{n: 128, k: 256}} = Lab15.cupel(%{"n" => 10_000, "k" => 10_000, "trials" => 2})
  end

  test "the terminal: proofs exit 0, refutations 1, bad input 3; JSON in a pipe", %{dir: dir} do
    a = Path.join(dir, "r.net")
    b = Path.join(dir, "k.net")
    t = Path.join(dir, "t.net")
    File.write!(a, Vapor.Rebis.Gen.ripple(8))
    File.write!(b, Vapor.Rebis.Gen.kogge_stone(8))
    File.write!(t, Vapor.Rebis.Gen.ripple(8, trojan: 0xA5))
    assert capture_io(fn -> assert Vapor.Main.run(["rebis", "equiv", a, b]) == 0 end) =~ "equivalent"
    out = capture_io(fn -> assert Vapor.Main.run(["rebis", "equiv", a, t]) == 1 end)
    assert out =~ "a = 0xA5" and out =~ "b = 0x0"
    m = Path.join(dir, "m.net")
    File.write!(m, Vapor.Rebis.Gen.multiplier(6))
    assert capture_io(fn -> assert Vapor.Main.run(["rebis", "identity", m, "--spec", "m[12] = a[6] * b[6]"]) == 0 end) =~ "proved"
    assert capture_io(fn -> assert Vapor.Main.run(["rebis", "identity", m, "--spec", "m[12] = a[6] * b[6] + 1"]) == 1 end) =~ "refuted"
    assert capture_io(fn -> assert Vapor.Main.run(["aludel", "decide", "x^2 - x + 1/4", "--vars", "x", "--box", "0,1"]) == 0 end) =~ "certified"
    assert capture_io(fn -> assert Vapor.Main.run(["aludel", "decide", "x^2 - 2", "--vars", "x", "--box", "1,2"]) == 1 end) =~ "refuted"
    c = Path.join(dir, "c.tab")
    File.write!(c, "facts a b\nassume not (a and b)\nX: if a then p must go\nY: if b then p must not go\n")
    assert capture_io(fn -> assert Vapor.Main.run(["tabula", c]) == 0 end) =~ "consistent"
    n = Path.join(dir, "n.txt")
    File.write!(n, "1e16 1 -1e16 1")
    assert capture_io(fn -> assert Vapor.Main.run(["amalgam", n]) == 0 end) =~ "amalgam"
    assert capture_io(:stderr, fn -> assert Vapor.Main.run(["rebis", "equiv", a, Path.join(dir, "missing")]) == 3 end) =~ "missing"
    assert capture_io(:stderr, fn -> assert Vapor.Main.run(["tabula", Path.join(dir, "n.txt")]) == 3 end) =~ "tabula"
    System.put_env("VAPOR_TTY", "0")
    json = capture_io(fn -> Vapor.Main.run(["rebis", "equiv", a, t]) end)
    assert {:ok, %{"verdict" => "different", "counterexample" => _}} = Vapor.JSON.decode(json)
  end

  @tag :playwright
  @tag timeout: 600_000
  # every page script merges its strings into one I18N table; a key defined twice with
  # different text silently relabels another desk (opus.js once turned the Bancada's
  # "bias detected" into "caught")
  test "the Opus strings never redefine another script's key with other text" do
    keys = fn text ->
      # the per-script additions, and the base table of index.html
      (Regex.scan(~r/Object\.assign\(I18N\.(en|pt), \{(.*?)\n\}\);/s, text) ++ Regex.scan(~r/\n  (en|pt): \{(.*?)\n  \}/s, text))
      |> Enum.flat_map(fn [_, lang, body] -> for [_, k, v] <- Regex.scan(~r/(?:^|[{,]\s*)([A-Za-z_]\w*): "([^"]*)"/m, body), do: {{lang, k}, v} end)
      |> Map.new()
    end

    dir = Path.join(:code.priv_dir(:vapor), "console")
    opus = keys.(File.read!(Path.join(dir, "opus.js")))
    others = for f <- ["index.html", "athanor.js", "bancada.js", "mercado.js"], reduce: %{}, do: (acc -> Map.merge(acc, keys.(File.read!(Path.join(dir, f)))))
    assert map_size(opus) > 100
    clashes = for {k, v} <- opus, Map.has_key?(others, k), others[k] != v, do: {k, v, others[k]}
    assert clashes == [], inspect(clashes)
  end

  test "the five desks through the page in headless Chromium: every shelf example, a toggled fact, Portuguese, dark" do
    {:ok, srv} = Vapor.Serve.start_link(port: 0, model_name: "none")
    base = "http://127.0.0.1:#{Vapor.Serve.port(srv)}/"
    js = Path.expand("../js/console_opus.mjs", __DIR__)
    {out, 0} = System.cmd("node", [js, base] ++ List.wrap(System.get_env("VAPOR_SHOTS")), stderr_to_stdout: true)
    {:ok, j} = out |> String.split("\n", trim: true) |> List.last() |> Vapor.JSON.decode()
    assert j["failures"] == [] and j["errors"] == [], inspect(j)
    assert length(j["checks"]) >= 18
  end
end
