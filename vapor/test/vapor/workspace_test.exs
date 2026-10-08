defmodule Vapor.WorkspaceTest do
  @moduledoc "The open bench end to end: the console's workspace over HTTP, the command line, the MCP tools, the TUI."
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO
  alias Vapor.JSON
  alias Vapor.Athanor.Examples

  @moduletag timeout: 600_000

  setup_all do
    {:ok, srv} = Vapor.Serve.start_link(port: 0, model_name: "none")
    dir = Path.join(System.tmp_dir!(), "vapor-ws-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    %{base: "http://127.0.0.1:#{Vapor.Serve.port(srv)}", dir: dir}
  end

  defp req(base, method, path, obj \\ nil) do
    r = if method == :get, do: {String.to_charlist(base <> path), []}, else: {String.to_charlist(base <> path), [], ~c"application/json", JSON.encode(obj)}
    {:ok, {{_, status, _}, _h, body}} = :httpc.request(method, r, [timeout: 300_000], body_format: :binary)
    {:ok, j} = JSON.decode(body)
    {status, j}
  end

  describe "the console's workspace" do
    test "the bench lists its starting points and detects what a text is", %{base: b} do
      {200, info} = req(b, :get, "/v1/vapor/workspace")
      assert length(info["athanor"]) >= 15 and length(info["crucible"]) == 10 and length(info["assay"]) == 11
      assert info["card"] =~ "ATHANOR"
      for {text, kind} <- [{Examples.get("golomb").text, "athanor"}, {Examples.get("tictactoe").text, "game"}, {"H = p^2/2 + q^2/2", "crucible"}, {"f(x) = x + 1", "alembic"}] do
        assert {200, %{"kind" => ^kind}} = req(b, :post, "/v1/vapor/detect", %{text: text})
      end
    end

    test "a session: start, poll to the end, certify, verify — and a forged certificate fails", %{base: b} do
      text = Examples.get("maxcut").text
      {200, s} = req(b, :post, "/v1/vapor/athanor", %{text: text})
      id = s["id"]
      wait = fn wait, n -> {200, x} = req(b, :get, "/v1/vapor/athanor/#{id}?since=0"); if x["status"] == "running" and n > 0, do: (Process.sleep(100); wait.(wait, n - 1)), else: x end
      done = wait.(wait, 300)
      assert done["status"] == "done" and length(done["sparks"]) > 0
      {200, cert} = req(b, :post, "/v1/vapor/athanor/#{id}", %{action: "certificate"})
      assert cert["reason"] == "exhausted"
      {200, v} = req(b, :post, "/v1/vapor/athanor/verify", %{text: text, certificate: cert})
      assert v["verified"]
      {200, v2} = req(b, :post, "/v1/vapor/athanor/verify", %{text: text, certificate: put_in(cert, ["best", "value"], 99)})
      refute v2["verified"]
    end

    test "a person's proposal enters the search; a bad one is refused with the reason", %{base: b} do
      {200, s} = req(b, :post, "/v1/vapor/athanor", %{text: "space = ints(2, 0, 99)\nminimize(v) = abs(v[0] - 61) + abs(v[1] - 7)\nbudget = 5000"})
      {200, r} = req(b, :post, "/v1/vapor/athanor/#{s["id"]}", %{action: "propose", candidates: ["[61, 7]", "[500, 1]"]})
      assert r["rejected"] |> hd() =~ "not a member"
      Process.sleep(500)
      {200, c} = req(b, :post, "/v1/vapor/athanor/#{s["id"]}", %{action: "certificate"})
      assert c["best"]["value"] == 0
    end

    test "errors in a problem come back with their line", %{base: b} do
      assert {422, %{"error" => %{"message" => m}}} = req(b, :post, "/v1/vapor/athanor", %{text: "space = perm(5)\nminimize(p) = p[0] +"})
      assert m =~ "line" or m =~ "unexpected"
    end

    test "games, the Crucible, the Assay and Alembic over HTTP", %{base: b} do
      g = Examples.get("tictactoe").text
      {200, v} = req(b, :post, "/v1/vapor/game", %{text: g, action: "play", move: "4"})
      {200, r} = req(b, :post, "/v1/vapor/game", %{text: g, action: "reply", state: v["state"]})
      assert r["reply"] in ["0", "2", "6", "8"]
      {200, c} = req(b, :post, "/v1/vapor/crucible", %{kind: "laws", text: "x' = y\ny' = -x"})
      assert hd(c["laws"])["law"] == "x^2 + y^2"
      {200, a} = req(b, :post, "/v1/vapor/assay", %{tool: "judge", text: Vapor.Assay.example("judge")})
      assert is_number(a["p_position_bias"])
      {200, e} = req(b, :post, "/v1/vapor/alembic", %{text: "f(n) = n * n", expr: "f(12)"})
      assert e["value"] == "144"
    end

    test "without a model, drafting from words says how to get one", %{base: b} do
      assert {422, %{"error" => %{"message" => m}}} = req(b, :post, "/v1/vapor/formalize", %{words: "a ruler"})
      assert m =~ "VAPOR_MIND"
    end

    test "the page carries the open bench, its touchstone and the alembic mark", %{base: b} do
      {:ok, {{_, 200, _}, _, html}} = :httpc.request(:get, {String.to_charlist(b <> "/"), []}, [], body_format: :binary)
      assert html =~ ~s(id="p-work") and html =~ "function touchstone(" and html =~ "an alembic"
    end
  end

  describe "the command line" do
    test "alembic, athanor, verify and exit statuses", %{dir: d} do
      p = Path.join(d, "euler.nbq")
      File.write!(p, Examples.get("euler").text)
      assert capture_io(fn -> assert Vapor.Main.run(["alembic", "-e", "sum([x^2 for x in 1..10])"]) == 0 end) =~ "385"
      out = capture_io(fn -> assert Vapor.Main.run(["athanor", "run", p]) == 1 end)
      {:ok, cert} = JSON.decode(out)
      assert cert["reason"] == "counterexample"
      c = Path.join(d, "euler.json")
      File.write!(c, out)
      v = capture_io(fn -> assert Vapor.Main.run(["verify", p, c]) == 0 end)
      assert v =~ ~s("verified":true)
      assert capture_io(:stderr, fn -> assert Vapor.Main.run(["frobnicate"]) == 2 end) =~ "unknown command"
    end

    test "crucible, assay and render through files and pipes", %{dir: d} do
      f = Path.join(d, "osc.txt")
      File.write!(f, "x' = v\nv' = -9*x")
      assert capture_io(fn -> assert Vapor.Main.run(["crucible", "laws", f]) == 0 end) =~ "9·x^2 + v^2"
      j = Path.join(d, "judge.csv")
      File.write!(j, Vapor.Assay.example("judge"))
      # the example judge prefers the first slot by construction: the check fails, and so does the status
      out = capture_io(fn -> assert Vapor.Main.run(["assay", "judge", j]) == 1 end)
      assert out =~ "p_position_bias" and out =~ "position bias, p ="
      sc = Path.join(d, "ball.txt")
      File.write!(sc, "camera pos=0,0,4 look=0,0,0 fov=40\nsun dir=0.3,1,0.4 color=1,1,1 power=2\nsphere c=0,0,0 r=1 mat=diffuse albedo=0.8,0.2,0.2")
      {:ok, ink} = JSON.decode(capture_io(fn -> assert Vapor.Main.run(["render", sc, "--ink", "--width", "40", "--height", "30", "--json"]) == 0 end))
      assert ink["style"] == "ink" and ink["outline_pixels"] > 0 and File.read!(Path.join(d, "ball.png")) |> binary_part(1, 3) == "PNG"
      assert capture_io(:stderr, fn -> assert Vapor.Main.run(["render", sc <> ".missing"]) == 3 end) =~ "render"
    end
  end

  describe "agents and the terminal console" do
    test "MCP: the agent proposes, the Athanor checks", _ do
      tools = Vapor.MCP.Server.tools() |> Enum.map(& &1["name"])
      assert Enum.all?(~w(alembic_eval athanor_run athanor_verify game_query crucible_run assay_run render_scene), &(&1 in tools))
      st = Vapor.MCP.Server.new(dir: System.tmp_dir!())
      r = Vapor.MCP.Server.handle(%{"jsonrpc" => "2.0", "id" => 1, "method" => "tools/call", "params" => %{"name" => "athanor_run",
            "arguments" => %{"text" => "space = ints(2, 0, 99)\nminimize(v) = abs(v[0] - 61) + abs(v[1] - 7)\nbudget = 50", "proposals" => ["[61, 7]"]}}}, st)
      {resp, _} = r
      data = get_in(resp, ["result", "structuredContent"])
      assert data["best"]["value"] == 0 and data["best"]["by"] == "mind"
    end

    test "the TUI speaks the bench's verbs" do
      st = Vapor.TUI.new(lang: :en)
      {out, _} = Vapor.TUI.eval(~S|alembic -e "factorial(10)"|, st)
      assert out =~ "3628800"
    end
  end
end
