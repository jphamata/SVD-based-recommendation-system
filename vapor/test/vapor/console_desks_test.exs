defmodule Vapor.ConsoleDesksTest do
  @moduledoc """
  The console's 0.12 desks over HTTP (docs/CONSOLE.md): each route
  answers its example with a certificate that holds, refuses what it
  cannot do with a reason (the controls), the page stays one
  self-contained document, and — where headless Chromium exists — every
  example of every desk is run through the page itself
  (test/js/console_desks.mjs).
  """
  use ExUnit.Case, async: false
  @moduletag timeout: 900_000
  alias Vapor.JSON

  setup_all do
    {:ok, srv} = Vapor.Serve.start_link(port: 0, model_name: "none")
    %{base: "http://127.0.0.1:#{Vapor.Serve.port(srv)}"}
  end

  defp post(base, path, obj) do
    {:ok, {{_, status, _}, _h, body}} =
      :httpc.request(:post, {String.to_charlist(base <> path), [], ~c"application/json", JSON.encode(obj)}, [timeout: 300_000], body_format: :binary)
    {:ok, j} = JSON.decode(body)
    {status, j}
  end

  defp get(base, path) do
    {:ok, {{_, status, _}, _h, body}} = :httpc.request(:get, {String.to_charlist(base <> path), []}, [timeout: 300_000], body_format: :binary)
    {status, body}
  end

  test "the page inlines its scripts: one self-contained document, the desks and the GPU tracer inside it", %{base: b} do
    {200, html} = get(b, "/")
    refute html =~ ~r/<script[^>]+src=/
    assert html =~ "const GPUTracer" and html =~ ~s{mount("bench"} and html =~ ~s{mount("render"}
  end

  test "workbench: units refused before running; an ODE with its event located", %{base: b} do
    {400, e} = post(b, "/v1/vapor/solve", %{text: "x' = v\nv' = -k*x\nk = 4[N/m]\nx(0)=1[m]\nv(0)=0[m/s]\nt=0..1[s]"})
    assert e["error"]["message"] =~ "force"
    {200, r} = post(b, "/v1/vapor/solve", %{text: "y' = v\nv' = -9.81\ny(0)=0\nv(0)=20\nt=0..10\nstop when y < 0"})
    assert r["kind"] == "ode"
    assert_in_delta r["event"]["t"], 40 / 9.81, 1.0e-9
  end

  test "engineering: every kind answers with its certificate; an unknown kind is refused", %{base: b} do
    cases = [
      {"circuit", "V1 in 0 DC 5\nR1 in a 1k\nR2 a 0 2k\n.op", fn r -> r["op"]["certificate"]["kcl_max"] < 1.0e-12 end},
      {"power", "base 100\nbus 1 slack V=1.06\nbus 2 pq P=-40 Q=-10\nline 1 2 r=0.02 x=0.06 b=0.06", fn r -> r["certificate"]["max_mismatch_pu"] < 1.0e-8 end},
      {"structure", "node 1 0 0\nnode 2 3 0\nsupport 1 fixed\nbeam 1 2 E=200e9 A=0.01 I=1e-4\nload 2 fy=-10e3", fn r -> r["certificate"]["relative"] < 1.0e-12 end},
      {"fem", "plate x=0..2 y=0..1 nx=4 ny=2\nmaterial E=1000 nu=0.3 t=1\nfix x=0\ntraction x=2 tx=1", fn r -> r["certificate"]["residual"] < 1.0e-9 end},
      {"pipes", "reservoir A head=100\nreservoir B head=80\npipe 1 A B L=1000 D=0.3 eps=0.00026", fn r -> r["certificate"]["continuity_max"] < 1.0e-12 end},
      {"reactions", "A -> B ; k = 1\nA0 = 1\nt = 0 .. 5", fn r -> hd(r["invariants"])["drift"] < 1.0e-9 end},
      {"flash", "component benzene z=0.4 A=6.90565 B=1211.033 C=220.79\ncomponent toluene z=0.6 A=6.95464 B=1344.8 C=219.482\nT = 100\nP = 760", fn r -> r["certificate"]["balance"] < 1.0e-12 end},
      {"distill", "alpha = 2.5; xF = 0.45; xD = 0.95; xB = 0.05; q = 1; Rfactor = 1.3", fn r -> r["control"]["total_reflux_stages"] == ceil(r["fenske"]) end}
    ]
    for {k, text, ok} <- cases do
      {200, r} = post(b, "/v1/vapor/engineering", %{kind: k, text: text})
      assert ok.(r), k
    end
    {400, e} = post(b, "/v1/vapor/engineering", %{kind: "reactor-core", text: "x"})
    assert e["error"]["message"] =~ "kind"
  end

  test "logic: a number with a checked witness and a checked refutation; a false claim refuted with a counterexample", %{base: b} do
    {200, r} = post(b, "/v1/vapor/logic", %{text: "vdw 3 2"})
    assert r["value"] == 9 and r["below"]["checked"] and r["refutation"]["drup"]["valid"]
    {200, f} = post(b, "/v1/vapor/logic", %{text: "valid: (p -> q) -> (q -> p)"})
    assert f["verdict"] == "refuted" and f["counterexample"] == %{"p" => false, "q" => true}
  end

  test "boards: a legal move is played and an illegal one refused; a mate proved and checked; Go's superko and poker's equilibrium", %{base: b} do
    {200, r} = post(b, "/v1/vapor/chess", %{action: "move", move: "e2e4"})
    assert r["played"] == "e4" and r["turn"] == "b"
    {400, _} = post(b, "/v1/vapor/chess", %{action: "move", move: "e2e5"})
    {200, m} = post(b, "/v1/vapor/chess", %{action: "mate", fen: "7k/8/8/8/8/8/1R6/R5K1 w - - 0 1", n: 2})
    assert m["mate"] and m["check"] =~ ":ok"
    {200, p} = post(b, "/v1/vapor/chess", %{action: "perft", depth: 3})
    assert p["nodes"] == 8902
    {200, s} = post(b, "/v1/vapor/shogi", %{action: "perft", depth: 2})
    assert s["nodes"] == 900
    {400, _} = post(b, "/v1/vapor/go", %{size: 5, moves: [12, 12]})
    {200, k} = post(b, "/v1/vapor/mnk", %{m: 3, n: 3, k: 3})
    assert k["solved"] and k["value"] == 0
    {200, pk} = post(b, "/v1/vapor/poker", %{game: "kuhn", iterations: 300})
    assert pk["exploitability"] < 0.01 and pk["uniform_exploitability"] > 0.4
    assert_in_delta pk["value"], -1 / 18, 0.01
  end

  test "proteins: two NMR models compared; chains of different length refused (no silent correspondence)", %{base: b} do
    {200, r} = post(b, "/v1/vapor/protein", %{action: "compare", model_sample: "1LCD#2", native_sample: "1LCD"})
    assert r["tm"] > 0.8 and r["tm"] < 1.0 and r["rmsd"] > 0.1
    {400, e} = post(b, "/v1/vapor/protein", %{action: "compare", model_sample: "1A8O", native_sample: "1LCD"})
    assert e["error"]["message"] =~ "differ in length"
  end

  test "render: a picture with its mean radiance; a request beyond the server's budget refused; the furnace", %{base: b} do
    scene = "camera pos=0,0,3 look=0,0,0 fov=40\nsky color=1,1,1\nsphere c=0,0,0 r=1 mat=diffuse albedo=0.5,0.5,0.5"
    {200, r} = post(b, "/v1/vapor/render", %{text: scene, width: 32, height: 20, spp: 4})
    assert String.starts_with?(r["png"], "data:image/png;base64,")
    {400, e} = post(b, "/v1/vapor/render", %{text: scene, width: 480, height: 320, spp: 256})
    assert e["error"]["message"] =~ "6 million"
    {200, f} = get(b, "/v1/vapor/render/furnace")
    {:ok, f} = JSON.decode(f)
    assert f["uniform"]["max_error"] < 1.0e-9 and abs(f["gradient"]["mean_error"]) < 0.005 and f["control"]["mean_error"] < -0.03
  end

  @tag :playwright
  test "every example of every desk, run through the page in headless Chromium", %{base: b} do
    js = Path.expand("../js/console_desks.mjs", __DIR__)
    {out, 0} = System.cmd("node", [js, b <> "/"], stderr_to_stdout: true)
    {:ok, j} = out |> String.split("\n", trim: true) |> List.last() |> JSON.decode()
    assert j["failures"] == [] and j["errors"] == [], inspect(j)
    assert length(j["checks"]) >= 60
  end
end
