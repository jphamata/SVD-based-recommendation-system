defmodule Vapor.LSPTest do
  use ExUnit.Case, async: true
  alias Vapor.LSP

  @osc """
  (claim kinetic (root H-s-b) (wazn fail)
    (inputs (v q))
    (body (* 1/2 v v)))

  (claim energy (root H-f-Z) (wazn burhan)
    (inputs (x q) (v q))
    (field (x v) (v (- x)))
    (proof conserved)
    (body (+ (kinetic v) (* 1/2 x x))))

  (claim damped (root H-f-Z) (wazn burhan)
    (inputs (x q) (v q))
    (field (x v) (v (- (- x) (* 1/10 v))))
    (proof conserved)
    (body (+ (kinetic v) (* 1/2 x x))))
  """

  defp start do
    {[init], st} = LSP.handle(%{"jsonrpc" => "2.0", "id" => 1, "method" => "initialize", "params" => %{}}, %{docs: %{}, shutdown: false})
    {init, st}
  end

  defp open(st, uri, text, lang) do
    LSP.handle(%{"method" => "textDocument/didOpen", "params" => %{"textDocument" => %{"uri" => uri, "languageId" => lang, "version" => 1, "text" => text}}}, st)
  end

  defp req(st, id, method, params), do: LSP.handle(%{"id" => id, "method" => method, "params" => params}, st)

  test "capabilities; diagnostics decide the obligations: the refuted claim is an error at its own line, with its point" do
    {init, st} = start()
    assert init["result"]["capabilities"]["hoverProvider"]
    {[%{"params" => %{"diagnostics" => [d]}}], _} = open(st, "file:///a.wzn", @osc, "mizan")
    assert d["severity"] == 1 and d["range"]["start"]["line"] == 10
    assert d["message"] =~ "damped: refuted" and d["message"] =~ "at "
  end

  test "as you type only syntax is checked; a syntax error points at its line" do
    {_, st} = start()
    {_, st} = open(st, "file:///b.wzn", "(claim f (root H-s-b) (wazn fail) (body 1))", "mizan")
    {[%{"params" => %{"diagnostics" => [d]}}], _} =
      LSP.handle(%{"method" => "textDocument/didChange", "params" => %{"textDocument" => %{"uri" => "file:///b.wzn"}, "contentChanges" => [%{"text" => "\n\n(claim f (root H-s-b) (wazn fail) (body (+ 1)))"}]}}, st)
    assert d["range"]["start"]["line"] == 2 and d["message"] =~ "argument"
  end

  test "hover: a root, a keyword in both scripts, a claim's verdict; completion in the file's script; symbols; definition" do
    {_, st} = start()
    {_, st} = open(st, "file:///a.wzn", @osc, "mizan")
    {[h1], st} = req(st, 2, "textDocument/hover", %{"textDocument" => %{"uri" => "file:///a.wzn"}, "position" => %{"line" => 4, "character" => 22}})
    assert h1["result"]["contents"]["value"] =~ "conservation" and h1["result"]["contents"]["value"] =~ "abjad"
    {[h2], st} = req(st, 3, "textDocument/hover", %{"textDocument" => %{"uri" => "file:///a.wzn"}, "position" => %{"line" => 4, "character" => 9}})
    assert h2["result"]["contents"]["value"] =~ "proved"
    {[c], st} = req(st, 4, "textDocument/completion", %{"textDocument" => %{"uri" => "file:///a.wzn"}, "position" => %{"line" => 0, "character" => 0}})
    labels = Enum.map(c["result"], & &1["label"])
    assert "claim" in labels and "H-f-Z" in labels and "energy" in labels
    {_, st} = open(st, "file:///ar.wzn", "(دعوى طاقة (جذر ح-س-ب) (وزن فاعل) (تنفيذ ١))", "mizan")
    {[c2], st} = req(st, 5, "textDocument/completion", %{"textDocument" => %{"uri" => "file:///ar.wzn"}, "position" => %{"line" => 0, "character" => 0}})
    assert "دعوى" in Enum.map(c2["result"], & &1["label"])
    {[s], st} = req(st, 6, "textDocument/documentSymbol", %{"textDocument" => %{"uri" => "file:///a.wzn"}})
    assert Enum.map(s["result"], & &1["name"]) |> Enum.sort() == ["damped", "energy", "kinetic"]
    {[d], _} = req(st, 7, "textDocument/definition", %{"textDocument" => %{"uri" => "file:///a.wzn"}, "position" => %{"line" => 8, "character" => 13}})
    assert d["result"]["range"]["start"]["line"] == 0
  end

  test "formatting keeps the script; the projection command offers the other one as an edit" do
    {_, st} = start()
    {_, st} = open(st, "file:///a.wzn", "(claim   f (root H-s-b)  (wazn fail) (body 1))", "mizan")
    {[f], st} = req(st, 8, "textDocument/formatting", %{"textDocument" => %{"uri" => "file:///a.wzn"}})
    assert [%{"newText" => "(claim f (root H-s-b) (wazn fail)\n  (body 1))\n"}] = f["result"]
    {[ok, edit], _} = req(st, 9, "workspace/executeCommand", %{"command" => "vapor.mizan.toArabic", "arguments" => ["file:///a.wzn"]})
    assert ok["result"] == nil
    assert edit["method"] == "workspace/applyEdit"
    assert edit["params"]["edit"]["changes"]["file:///a.wzn"] |> hd() |> Map.get("newText") =~ "دعوى f"
  end

  test "Alembic: parse errors with line and column; definitions as symbols; builtins completed" do
    {_, st} = start()
    {[%{"params" => %{"diagnostics" => [d]}}], st} = open(st, "file:///p.alb", "f(x) = x * 2\ny = = 3\n", "alembic")
    assert d["range"]["start"]["line"] == 1
    {_, st} = open(st, "file:///q.alb", "f(x) = x * 2\ng = f(3)\n", "alembic")
    {[s], st} = req(st, 10, "textDocument/documentSymbol", %{"textDocument" => %{"uri" => "file:///q.alb"}})
    assert Enum.map(s["result"], & &1["name"]) == ["f", "g"]
    {[c], _} = req(st, 11, "textDocument/completion", %{"textDocument" => %{"uri" => "file:///q.alb"}, "position" => %{"line" => 0, "character" => 0}})
    assert length(c["result"]) > 20
  end

  test "unknown requests get method-not-found; shutdown answers; notifications get no reply" do
    {_, st} = start()
    {[e], st} = req(st, 12, "textDocument/rename", %{})
    assert e["error"]["code"] == -32601
    {[], st} = LSP.handle(%{"method" => "$/cancelRequest", "params" => %{}}, st)
    {[r], st} = req(st, 13, "shutdown", nil)
    assert Map.has_key?(r, "result") and st.shutdown
  end

  @tag :node
  test "end to end over stdio: a Node client and `bin/vapor lsp`, framed as editors frame it" do
    node = System.find_executable("node")
    if node == nil, do: flunk("node is needed for this test")
    root = Path.expand("../..", __DIR__)
    {out, 0} = System.cmd(node, [Path.join(root, "test/js/lsp_client.mjs"), Path.join(root, "bin/vapor")], env: [{"MIX_ENV", "test"}], stderr_to_stdout: false)
    {:ok, r} = Vapor.JSON.decode(out |> String.split("\n", trim: true) |> List.last())
    assert "hoverProvider" in r["capabilities"]
    assert [%{"line" => 0, "severity" => 1, "message" => msg}] = r["diagnostics"]
    assert msg =~ "refuted"
    assert r["hover"] =~ "conservation"
    assert r["formatted"] == 1
    assert r["arabic"] =~ "دعوى"
  end
end
