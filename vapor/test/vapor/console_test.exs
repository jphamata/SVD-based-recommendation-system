defmodule Vapor.ConsoleTest do
  @moduledoc "The operator console over HTTP: page, library, search with receipts, citations, the noise gate — without a model."
  use ExUnit.Case, async: false
  alias Vapor.JSON

  @dir Path.expand("../fixtures/docs", __DIR__)

  setup_all do
    {:ok, srv} = Vapor.Serve.start_link(port: 0, model_name: "none")
    %{base: "http://127.0.0.1:#{Vapor.Serve.port(srv)}"}
  end

  defp get(base, path) do
    {:ok, {{_, status, _}, headers, body}} = :httpc.request(:get, {String.to_charlist(base <> path), []}, [], body_format: :binary)
    {status, Map.new(headers, fn {k, v} -> {to_string(k), to_string(v)} end), body}
  end

  defp post(base, path, obj) do
    {:ok, {{_, status, _}, _h, body}} =
      :httpc.request(:post, {String.to_charlist(base <> path), [], ~c"application/json", JSON.encode(obj)}, [timeout: 120_000], body_format: :binary)

    {:ok, j} = JSON.decode(body)
    {status, j}
  end

  defp upload(base, name), do: post(base, "/v1/vapor/library", %{name: name, data: Base.encode64(File.read!(Path.join(@dir, name)))})

  test "the page is served, self-contained (no external script, style or font)", %{base: b} do
    {200, h, html} = get(b, "/")
    assert h["content-type"] =~ "text/html"
    assert html =~ "<title>vapor"
    refute html =~ ~r/<script[^>]+src=|<link[^>]+href=["']https?:|@import|url\(https?:/
  end

  test "English by default with Portuguese beside it; logo, favicon and an installable app manifest", %{base: b} do
    {200, _, html} = get(b, "/")
    assert html =~ ~s(<html lang="en">) and html =~ "const I18N = {" and html =~ "pt: {" and html =~ ~s(id="lang")
    assert {200, %{"content-type" => "image/x-icon"}, ico} = get(b, "/favicon.ico")
    assert <<0, 0, 1, 0, _::binary>> = ico
    assert {200, %{"content-type" => "image/svg+xml"}, svg} = get(b, "/logo.svg")
    assert svg =~ "<svg"
    {200, _, man} = get(b, "/manifest.webmanifest")
    {:ok, m} = JSON.decode(man)
    assert m["display"] == "standalone" and Enum.any?(m["icons"], &(&1["sizes"] == "512x512"))
    assert {200, %{"content-type" => "image/png"}, <<137, "PNG", _::binary>>} = get(b, "/icon-192.png")
  end

  test "draw: a generated digit, read back by the real-data classifier, never a copy; bad requests refused", %{base: b} do
    assert {200, j} = post(b, "/v1/vapor/draw", %{digit: 4, seed: 2, steps: 20})
    assert j["reading"]["digit"] == 4 and length(j["image"]) == 64 and length(j["trace"]) == 20
    assert j["nearest_train"] > 5.0
    # the same request is the same image
    assert {200, ^j} = post(b, "/v1/vapor/draw", %{digit: 4, seed: 2, steps: 20}) |> then(fn {s, x} -> {s, %{x | "ms" => j["ms"]}} end)
    assert {400, _} = post(b, "/v1/vapor/draw", %{digit: 12})
    assert {400, _} = post(b, "/v1/vapor/ocr", %{name: "x.bin", data: Base.encode64("hello")})
  end

  test "without a model: the console works, chat says why it cannot", %{base: b} do
    {200, _, body} = get(b, "/v1/vapor/info")
    {:ok, info} = JSON.decode(body)
    assert info["spec"] == nil and "decoder" in info["adapters"]
    assert {503, %{"error" => %{"message" => m}}} = post(b, "/v1/chat/completions", %{messages: [%{role: "user", content: "oi"}]})
    assert m =~ "no model"
  end

  test "upload, search with provenance and a receipt that recomputes, citations checked verbatim", %{base: b} do
    assert {200, %{"report" => %{"added" => 6}}} = upload(b, "bundle.zip")
    assert {200, %{"report" => %{"added" => 2}}} = upload(b, "gs.pdf")
    assert {200, %{"report" => %{"duplicate" => "gs.pdf"}}} = upload(b, "gs.pdf")
    assert {422, %{"error" => %{"message" => m}}} = upload(b, "bomb.zip")
    assert m =~ "zip bomb"

    {200, r} = post(b, "/v1/vapor/search", %{query: "Página dois PostScript", k: 3})
    [top | _] = r["hits"]
    assert top["doc"] == "gs.pdf#p2" and top["container_sha256"] == :crypto.hash(:sha256, File.read!(Path.join(@dir, "gs.pdf"))) |> Base.encode16(case: :lower)
    assert {200, %{"verified" => true}} = post(b, "/v1/vapor/verify", Map.take(r, ~w(query k method library_receipt)))
    assert {200, %{"verified" => false}} = post(b, "/v1/vapor/verify", %{r | "library_receipt" => "0"} |> Map.take(~w(query k method library_receipt)))

    answer = ~s(A fonte diz <quote src="1">Página dois do PostScript.</quote> e <quote src="1">algo inventado</quote>.)
    {200, c} = post(b, "/v1/vapor/citations", %{answer: answer, query: "Página dois PostScript", k: 3})
    assert [%{"verbatim" => true}, %{"verbatim" => false}] = c["citations"]
    refute c["all_verbatim"]

    {200, lib} = get(b, "/v1/vapor/library") |> then(fn {s, _, body} -> {s, elem(JSON.decode(body), 1)} end)
    assert Enum.any?(lib["files"], &(&1["path"] == "bundle.zip!/inner.zip!/deep/notes.md"))
    assert Enum.any?(lib["warnings"], &(&1 =~ "binario.bin"))

    {200, h, png} = get(b, "/v1/vapor/thumb?doc=" <> URI.encode_www_form("bundle.zip!/pasta/scene.png"))
    assert h["content-type"] == "image/png" and match?(<<0x89, "PNG", _::binary>>, png)
    {200, s} = post(b, "/v1/vapor/search_image", %{name: "q.png", data: Base.encode64(File.read!(Path.join(@dir, "rgba.png")))})
    assert [%{"doc" => "bundle.zip!/pasta/scene.png"} | _] = s["hits"]
  end

  test "the noise gate over HTTP", %{base: b} do
    {200, good} = post(b, "/v1/vapor/quality", %{text: "A eclusa de documentos reconhece cada arquivo pelos bytes e devolve o texto com o caminho até a página, para que a busca possa citar."})
    assert good["verdict"] == "pass"
    {200, bad} = post(b, "/v1/vapor/quality", %{text: "qx7#zv!kW pl0 9g^^jw ;bnm zzqv 1xx@ wvu kkq jjz qpwmx zrt vvbn mqqk xzjp wwq ttzk bbnq ppqx mmzw kkzr"})
    assert bad["verdict"] == "fail" and bad["noise"] == "noise"
    assert {400, _} = post(b, "/v1/vapor/quality", %{nope: 1})
  end

  test "with a token: everything but /health needs it — Bearer, or the HttpOnly cookie set by /?token=" do
    {:ok, srv} = Vapor.Serve.start_link(port: 0, model_name: "none", token: "s3cret")
    b = "http://127.0.0.1:#{Vapor.Serve.port(srv)}"
    assert {200, _, _} = get(b, "/health")
    assert {401, _, _} = get(b, "/")
    assert {401, _, _} = get(b, "/v1/vapor/info")
    assert {401, _, _} = get(b, "/?token=wrong")
    {200, h, html} = get(b, "/?token=s3cret")
    assert html =~ "<title>vapor" and h["set-cookie"] =~ "vapor_token=s3cret; HttpOnly; SameSite=Strict"
    {:ok, {{_, 200, _}, _, _}} = :httpc.request(:get, {String.to_charlist(b <> "/v1/vapor/info"), [{~c"authorization", ~c"Bearer s3cret"}]}, [], [])
    {:ok, {{_, 200, _}, _, _}} = :httpc.request(:get, {String.to_charlist(b <> "/v1/vapor/info"), [{~c"cookie", ~c"vapor_token=s3cret"}]}, [], [])
  end

  # the ledger: search receipts are anchored; the checkpoint, the receipts
  # and the consistency proofs verify with nothing but the published key
  test "ledger: searches anchored, checkpoint signed, proofs and consistency verifiable offline", %{base: b} do
    {200, _} = upload(b, "simple.pdf")
    {200, j0} = post(b, "/v1/vapor/search", %{query: "hello", k: 2})
    assert %{"index" => i0, "proof" => _, "checkpoint" => _} = j0["tlog"]
    {200, n} = post(b, "/v1/vapor/tlog", %{text: "a decision, recorded"})
    assert n["receipt"]["index"] == i0 + 1
    {200, _, body} = get(b, "/v1/vapor/tlog")
    {:ok, l} = JSON.decode(body)
    assert l["size"] >= 2 and hd(l["entries"])["kind"] == "vapor-note/1"
    {:ok, %{text: text}} = Vapor.Tlog.Note.open(l["checkpoint"], [l["verifier"]])
    {:ok, cp} = Vapor.Tlog.parse_checkpoint(text)

    {200, _, pb} = get(b, "/v1/vapor/tlog/proof?index=#{i0}")
    {:ok, p} = JSON.decode(pb)
    entry = Base.decode64!(p["data"])
    assert entry =~ ~s("query":"hello")
    proof = Enum.map(p["receipt"]["proof"], &Base.decode16!(&1, case: :lower))
    assert Vapor.Tlog.verify_inclusion(Vapor.Tlog.leaf_hash(entry), i0, cp.size, proof, cp.root)

    # the tree the search's receipt committed to is extended by today's
    {:ok, %{text: t0}} = Vapor.Tlog.Note.open(j0["tlog"]["checkpoint"], [l["verifier"]])
    {:ok, cp0} = Vapor.Tlog.parse_checkpoint(t0)
    {200, _, cb} = get(b, "/v1/vapor/tlog/consistency?from=#{cp0.size}")
    {:ok, c} = JSON.decode(cb)
    cproof = Enum.map(c["proof"], &Base.decode16!(&1, case: :lower))
    assert Vapor.Tlog.verify_consistency(cp0.size, cp.size, cproof, cp0.root, cp.root)
    refute Vapor.Tlog.verify_consistency(cp0.size, cp.size, cproof, cp.root, cp0.root)
    assert {400, _} = post(b, "/v1/vapor/tlog", %{text: ""})
  end
  test "vision: a table on the page comes back with its structure, cells and renderings", %{base: b} do
    png = Path.expand("../../priv/quality/tables/t01_grid.png", __DIR__)
    assert {200, j} = post(b, "/v1/vapor/ocr", %{name: "t01_grid.png", data: Base.encode64(File.read!(png))})
    [page] = j["pages"]
    assert [t | _] = page["tables"]
    assert t["kind"] == "ruled" and t["rows"] >= 3 and t["cols"] >= 3
    assert length(t["cells"]) >= t["rows"] * t["cols"] - 4
    assert t["markdown"] =~ "|" and t["html"] =~ "<table" and t["csv"] =~ ","
    assert Enum.all?(t["cells"], &(length(&1["box"]) == 4))
  end

  test "audit: the demonstration dossier verifies; an uploaded one with a flipped byte does not, by name", %{base: b} do
    assert {200, j} = post(b, "/v1/vapor/audit/demo", %{})
    assert j["ok"] and j["root_ok"] and length(j["items"]) >= 4
    assert Enum.all?(j["items"], &(&1["clauses"] != [])) and map_size(j["clauses"]) >= 12
    assert j["html"] =~ "<script>"
    assert [%{"check" => "ok"}] = j["anchors"]

    bin = Base.decode64!(j["dossier"])
    assert {200, again} = post(b, "/v1/vapor/audit", %{data: j["dossier"], log_key: j["log_key"]})
    assert again["ok"]
    pos = div(byte_size(bin), 2)
    <<a::binary-size(pos), x, rest::binary>> = bin
    assert {200, bad} = post(b, "/v1/vapor/audit", %{data: Base.encode64(a <> <<Bitwise.bxor(x, 1)>> <> rest)})
    refute bad["ok"]
    assert {400, _} = post(b, "/v1/vapor/audit", %{nope: 1})
  end

  test "studio: templates listed; a run previews every output; the second run is all cache; the seal verifies", %{base: b} do
    {200, _, body} = get(b, "/v1/vapor/studio/nodes")
    {:ok, cat} = JSON.decode(body)
    assert Enum.any?(cat["nodes"], &(&1["type"] == "diffusion.sample")) and Enum.any?(cat["templates"], &(&1["id"] == "mask"))
    g = Enum.find(cat["templates"], &(&1["id"] == "mask"))["graph"]

    {200, r} = post(b, "/v1/vapor/studio/run", %{graph: g})
    assert length(r["executed"]) == 6 and r["nodes"]["5"]["outputs"]["image"]["src"] =~ "data:image/png;base64,"
    {200, r2} = post(b, "/v1/vapor/studio/run", %{graph: g})
    assert r2["executed"] == [] and r2["root"] == r["root"]
    assert {200, %{"verified" => true}} = post(b, "/v1/vapor/studio/verify", %{graph: g, root: r["root"]})
    assert {200, %{"verified" => false}} = post(b, "/v1/vapor/studio/verify", %{graph: g, root: String.duplicate("0", 64)})

    # a mistyped wire: refused with the node and the repair
    bad = put_in(g, ["nodes", "5", "inputs", "mask"], ["1", "image"]) |> put_in(["nodes", "1", "type"], "audio.tone")
    assert {422, %{"error" => %{"repair" => _}}} = post(b, "/v1/vapor/studio/run", %{graph: bad})
    assert {422, %{"error" => %{"repair" => why}}} = post(b, "/v1/vapor/studio/comfy", %{workflow: %{"1" => %{"class_type" => "UpscaleModelLoader", "inputs" => %{}}}})
    assert why =~ "UpscaleModelLoader"
  end

  @tag :native
  test "0.11 over HTTP: a living scene analysed, directed and exported; sketch, proof, discovery, archive round trip", %{base: b} do
    {200, sc} = post(b, "/v1/vapor/scene/analyze", %{name: "sample:outdoor"})
    assert hd(sc["layers"])["kind"] == "sky" and sc["walk"]["cols"] > 0
    {200, d} = post(b, "/v1/vapor/scene/direct", %{prompt: "noite de chuva, xyzzy"})
    assert %{"time" => "night"} in d["ops"] and d["unknown"] == ["xyzzy"]
    # the export is a page, not JSON
    {:ok, {{_, 200, _}, h, html}} = :httpc.request(:post, {String.to_charlist(b <> "/v1/vapor/scene/export"), [], ~c"application/json", JSON.encode(%{scene: Map.put(sc, "ops", d["ops"])})}, [], body_format: :binary)
    assert List.keyfind(h, ~c"content-type", 0) |> elem(1) |> to_string() =~ "text/html"
    assert html =~ "SceneEngine.create" and html =~ "data:image/png;base64,"
    {200, sk} = post(b, "/v1/vapor/sketch", %{name: "sample:plan", mode: "plan"})
    assert length(sk["rooms"]) == 2 and String.starts_with?(sk["image"], "data:image/png")
    {200, pr} = post(b, "/v1/vapor/prove", %{theorem: "euler_line"})
    assert pr["verdict"] == "proved" and pr["check"] == "holds"
    {200, net} = post(b, "/v1/vapor/discover", %{task: "network", n: 5})
    assert net["size"] == 9 and net["sorts"]
    {200, ar} = post(b, "/v1/vapor/archive", %{kind: "discover.sorting_network", recipe: %{n: 5}})
    {200, ck} = post(b, "/v1/vapor/archive/check", %{data: ar["data"]})
    assert ck["intact"] and ck["replay"] == "{:ok, :same}"
    # refusals by name
    assert {400, %{"error" => %{"message" => m}}} = post(b, "/v1/vapor/prove", %{theorem: "fermat"})
    assert m =~ "theorem: one of"
    assert {400, _} = post(b, "/v1/vapor/scene/analyze", %{name: "sample:nope"})
    assert {400, _} = post(b, "/v1/vapor/games/move", %{board: [1, 2]})
  end
end
