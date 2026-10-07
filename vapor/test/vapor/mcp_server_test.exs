defmodule Vapor.MCPServerTest do
  @moduledoc """
  vapor as an MCP server (`Vapor.MCP.Server`, `mix vapor.mcp`):

    * the protocol by hand: initialize, tools/list, notifications get no
      reply, unknown methods are JSON-RPC errors, a crashing tool is a tool
      error and the server lives on;
    * the official MCP Python SDK as the client, through `mix vapor.mcp`:
      the thirteen tools listed; a mistyped graph refused with the node and the
      repair; a run writes its outputs as files named by digest and returns
      the image inline; **the edited graph re-runs only what depends on the
      edit** (the cache lives across calls); verify accepts the true root
      and refuses it for another graph (the control); retrieval hits carry
      the corpus root; a path outside the studio directory is refused.
  """
  use ExUnit.Case, async: false
  alias Vapor.MCP.Server
  import Vapor.TestHelpers

  setup do
    dir = Path.join(System.tmp_dir!(), "vapor-mcp-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)
    {:ok, dir: dir}
  end

  test "the protocol by hand", %{dir: dir} do
    st = Server.new(dir: dir)
    {r, st} = Server.handle(%{"jsonrpc" => "2.0", "id" => 1, "method" => "initialize", "params" => %{"protocolVersion" => "2025-06-18"}}, st)
    assert r["result"]["serverInfo"]["name"] == "vapor" and r["result"]["capabilities"]["tools"]
    assert {nil, st} = Server.handle(%{"jsonrpc" => "2.0", "method" => "notifications/initialized"}, st)
    {r, st} = Server.handle(%{"jsonrpc" => "2.0", "id" => 2, "method" => "tools/list"}, st)
    assert length(r["result"]["tools"]) == 25 and Enum.all?(r["result"]["tools"], &match?(%{"inputSchema" => %{"type" => "object"}}, &1))
    {r, st} = Server.handle(%{"jsonrpc" => "2.0", "id" => 3, "method" => "nope"}, st)
    assert r["error"]["code"] == -32601
    {r, st} = Server.handle(%{"jsonrpc" => "2.0", "id" => 4, "method" => "tools/call", "params" => %{"name" => "studio_run", "arguments" => %{"graph" => 7}}}, st)
    assert r["result"]["isError"] == true
    {r, _} = Server.handle(%{"jsonrpc" => "2.0", "id" => 5, "method" => "tools/call", "params" => %{"name" => "studio_catalogue", "arguments" => %{}}}, st)
    assert r["result"]["isError"] == false and length(r["result"]["structuredContent"]["nodes"]) > 40
  end

  @tag :mcp
  test "the official MCP SDK drives `mix vapor.mcp`", %{dir: dir} do
    File.write!(Path.join(dir, "notes.md"), "# Geometry\n\nThe sphere is built by marching tetrahedra over a signed distance field; the mesh is welded and checked watertight.\n")
    File.write!(Path.join(dir, "other.md"), "Unrelated text about upscaling and Lanczos kernels.\n")
    {out, 0} = System.cmd(python(), [Path.expand("../python/mcp_client.py", __DIR__), File.cwd!(), dir], env: [{"MIX_ENV", to_string(Mix.env())}])
    d = Vapor.JSON.decode!(out |> String.split("\n", trim: true) |> List.last())

    assert d["tools"] == ~w(alembic_eval aludel_decide amalgam_sum arbitrage_check assay_run athanor_run athanor_verify board_query comfy_import context_search crucible_run cupel_drill engineering_run finance_run game_query logic_check rebis_check render_scene scene_ops studio_catalogue studio_run studio_validate studio_verify tabula_analyze workbench_solve)
    assert d["catalogue"]["structured"]["nodes"] |> Enum.map(& &1["type"]) |> Enum.all?(&String.starts_with?(&1, "diffusion."))
    assert d["invalid"]["isError"] and hd(d["invalid"]["content"])["text"] =~ "audio"
    refute d["valid"]["isError"]

    r1 = d["run1"]
    assert r1["structured"]["executed"] |> length() == 5
    assert Enum.any?(r1["content"], &(&1["type"] == "image" and &1["mime"] == "image/png"))
    for {_, o} <- r1["structured"]["outputs"], do: assert(File.read!(o["path"]) |> then(&(:crypto.hash(:sha256, &1) |> byte_size())) == 32)

    # only the edited tone and its output run again
    r2 = d["run2"]["structured"]
    assert Enum.sort(r2["executed"]) == ["3", "5"] and length(r2["cached"]) == 3
    assert r2["outputs"]["picture"]["digest"] == r1["structured"]["outputs"]["picture"]["digest"]
    refute r2["outputs"]["beep"]["digest"] == r1["structured"]["outputs"]["beep"]["digest"]

    refute d["verify"]["isError"]
    assert d["verify_wrong"]["isError"] and d["verify_wrong"]["structured"]["verified"] == false

    [hit | _] = d["search"]["structured"]["hits"]
    assert hit["doc"] == "notes.md" and hit["text"] =~ "marching tetrahedra"
    rag = Vapor.RAG.corpus([{"notes.md", File.read!(Path.join(dir, "notes.md"))}, {"other.md", File.read!(Path.join(dir, "other.md"))}])
    assert d["search"]["structured"]["root"] == Vapor.RAG.root_hex(rag)
    assert d["escape"]["isError"]
  end

  test "the laboratories as tools: the workbench, engineering, boards, the renderer, and logic_check deciding a proposal", %{dir: dir} do
    st = Server.new(dir: dir)
    call = fn name, args -> {r, _} = Server.handle(%{"jsonrpc" => "2.0", "id" => 9, "method" => "tools/call", "params" => %{"name" => name, "arguments" => args}}, st); r["result"] end
    w = call.("workbench_solve", %{"text" => "y' = v\nv' = -9.81\ny(0)=0\nv(0)=20\nt=0..10\nstop when y < 0"})
    refute w["isError"]
    assert_in_delta w["structuredContent"]["event"]["t"], 40 / 9.81, 1.0e-9
    assert call.("workbench_solve", %{"text" => "bad = 1[m] + 1[s]"})["structuredContent"]["lines"] |> hd() |> Map.get("error") =~ "adding"
    e = call.("engineering_run", %{"kind" => "circuit", "text" => "V1 a 0 DC 1\nR1 a 0 1k\n.op"})
    assert e["structuredContent"]["op"]["certificate"]["kcl_max"] < 1.0e-12
    assert call.("board_query", %{"game" => "chess", "action" => "perft", "depth" => 2})["structuredContent"]["nodes"] == 400
    # the checker decides, whoever proposes: a correct colouring accepted, a wrong one rejected (the control)
    good = call.("logic_check", %{"text" => "schur 3", "proposal" => %{"witness" => [1, 3, 3, 1, 2, 2, 2, 2, 2, 1, 3, 3, 1]}})
    assert good["structuredContent"]["accepted"] and good["structuredContent"]["claim"] == "S(3) ≥ 13"
    bad = call.("logic_check", %{"text" => "schur 3", "proposal" => %{"witness" => [1, 1, 3, 1, 2, 2, 2, 2, 2, 1, 3, 3, 1]}})
    refute bad["structuredContent"]["accepted"]
    {:unsat, proof, _} = Vapor.Logic.SAT.solve(Vapor.Logic.Problems.pigeonhole(4, 3))
    cnf = Vapor.Logic.SAT.to_dimacs(Vapor.Logic.Problems.pigeonhole(4, 3))
    assert call.("logic_check", %{"text" => cnf, "proposal" => %{"drup" => proof}})["structuredContent"]["accepted"]
    refute call.("logic_check", %{"text" => cnf, "proposal" => %{"drup" => Enum.drop(proof, -1)}})["structuredContent"]["accepted"]
    assert call.("logic_check", %{"text" => "valid: (p -> q) -> (q -> p)", "proposal" => %{"assignment" => %{"p" => false, "q" => true}}})["structuredContent"]["accepted"]
    r = call.("render_scene", %{"text" => "camera pos=0,0,3 look=0,0,0\nsky color=1,1,1\nsphere c=0,0,0 r=1 mat=diffuse albedo=0.5,0.5,0.5", "width" => 24, "height" => 16, "spp" => 2})
    assert [%{"type" => "text"}, %{"type" => "image", "mimeType" => "image/png"}] = r["content"]
    assert File.exists?(r["structuredContent"]["file"])
  end

  test "finance as tools (0.13): the desk answers with its certificate; a proposed arbitrage is checked, never trusted", %{dir: dir} do
    st = Server.new(dir: dir)
    call = fn name, args -> {r, _} = Server.handle(%{"jsonrpc" => "2.0", "id" => 9, "method" => "tools/call", "params" => %{"name" => name, "arguments" => args}}, st); r["result"] end
    c = call.("finance_run", %{"kind" => "curve", "text" => "date = 2025-01-02\ndi1 F26 = 15%\ndi1 F27 = 14.5%"})
    refute c["isError"]
    assert c["structuredContent"]["certificate"]["repriced"]
    b = call.("finance_run", %{"kind" => "backtest", "text" => "data = gbm n=600 seed=2\nsignal = sign(lead(close) - close)"})
    refute b["structuredContent"]["lookahead"]["clean"]
    q = "states = up, down\nbond bid=0.95 ask=0.96 payoff = 1, 1\nstock bid=100 ask=100.5 payoff = 120, 90\ncall bid=14 ask=14.5 payoff = 20, 0"
    good = call.("arbitrage_check", %{"text" => q, "proposal" => %{"portfolio" => %{"bond" => -1, "stock" => "1/90", "call" => "-1/60"}}})
    assert good["structuredContent"]["accepted"]
    # the control: a portfolio that merely pays is not free
    refute call.("arbitrage_check", %{"text" => q, "proposal" => %{"portfolio" => %{"stock" => 1, "call" => -1}}})["structuredContent"]["accepted"]
    assert call.("arbitrage_check", %{"text" => q, "proposal" => %{"portfolio" => %{"gold" => 1}}})["isError"]
  end

  test "the Opus desks as tools: a trojan found, a polynomial claim certified, a contract's antinomy, a drill, an order-free sum" do
    st = Server.new([])
    call = fn name, args ->
      {r, _} = Server.handle(%{"jsonrpc" => "2.0", "id" => 9, "method" => "tools/call", "params" => %{"name" => name, "arguments" => args}}, st)
      r["result"]
    end

    r = call.("rebis_check", %{"op" => "equivalent", "a" => Vapor.Rebis.Gen.ripple(12), "b" => Vapor.Rebis.Gen.ripple(12, trojan: 0x5A5)})
    assert r["isError"] == false and r["structuredContent"]["verdict"] == "different"
    r = call.("aludel_decide", %{"op" => "decide", "vars" => "x", "poly" => "x^2 - x + 1/4", "box" => [["0", "1"]]})
    assert r["structuredContent"]["verdict"] == "certified" and r["structuredContent"]["replayed"] == true
    r = call.("tabula_analyze", %{"text" => "facts a\nX: if a then p must go\nY: if a then p must not go\n"})
    assert r["structuredContent"]["verdict"] == "antinomies"
    r = call.("cupel_drill", %{"n" => 8, "k" => 16, "trials" => 2})
    assert r["structuredContent"]["int8"]["bits_detected"] == 32
    r = call.("amalgam_sum", %{"numbers" => "1e16 1 -1e16 1"})
    assert r["structuredContent"]["amalgam"]["value"] == "2.0"
    assert call.("aludel_decide", %{"op" => "decide", "vars" => "x", "poly" => "x^"})["isError"] == true
  end
end
