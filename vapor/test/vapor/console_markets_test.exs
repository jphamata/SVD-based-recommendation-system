defmodule Vapor.ConsoleMarketsTest do
  @moduledoc """
  The console's 0.13 desks (Finance, Trading desk): the endpoint answers
  every kind; a request is bounded; a malformed one is refused with the
  reason; the page carries the new script inlined (one self-contained
  document); and — where headless Chromium exists — every example of both
  desks, in English and Portuguese, is run through the page itself
  (test/js/console_markets.mjs).
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

  test "every kind answers through /v1/vapor/finance; bad requests are refused with the reason", %{base: b} do
    for {kind, text} <- [{"calendar", "du 2025-01-02 2026-01-02"}, {"curve", "date = 2025-01-02\ndi1 F26 = 15%"}, {"options", "price call S=100 K=100 T=1 vol=20%"},
                         {"risk", "data = t n=600 seed=2\nwindow = 250"}, {"portfolio", "assets = 4 n = 200"}, {"backtest", "data = gbm n=300\nsignal = sign(ret(close))"},
                         {"arbitrage", "fx\nUSD/BRL bid=5.40 ask=5.41\nEUR/USD bid=1.08 ask=1.081\nEUR/BRL bid=5.90 ask=5.91"}, {"book", "buy 10 @ 1.00 owner=a\nsell 10 @ 1.00 owner=b"},
                         {"exchange", "steps=200 seed=1"}, {"micro", "execution X=1e5 N=10"}] do
      {200, j} = post(b, "/v1/vapor/finance", %{kind: kind, text: text})
      assert j["kind"] == kind and is_integer(j["ms"]), kind
    end
    {400, e} = post(b, "/v1/vapor/finance", %{kind: "astrology", text: "x"})
    assert e["error"]["message"] =~ "kind"
    {400, e2} = post(b, "/v1/vapor/finance", %{kind: "options", text: "iv call S=100 K=50 T=1 r=5% price=40"})
    assert e2["error"]["message"] =~ "no-arbitrage bound"
    {400, e3} = post(b, "/v1/vapor/finance", %{kind: "book", text: "buy lots"})
    assert e3["error"]["message"] =~ "line 1"
  end

  test "the page inlines the markets script and declares both desks", %{base: b} do
    {:ok, {{_, 200, _}, _, html}} = :httpc.request(:get, {String.to_charlist(b <> "/"), []}, [], body_format: :binary)
    refute html =~ ~s(<script src="/market.js">)
    assert html =~ "market.js — the console's 0.13 desks"
    assert html =~ ~s(id="p-fin") and html =~ ~s(id="p-hft")
  end

  @tag :playwright
  test "every example of both desks, in two languages, run through the page in headless Chromium", %{base: b} do
    js = Path.expand("../js/console_markets.mjs", __DIR__)
    {out, 0} = System.cmd("node", [js, b <> "/"], stderr_to_stdout: true)
    {:ok, j} = out |> String.split("\n", trim: true) |> List.last() |> JSON.decode()
    assert j["failures"] == [] and j["errors"] == [], inspect(j)
    assert length(j["checks"]) >= 80
  end
end
