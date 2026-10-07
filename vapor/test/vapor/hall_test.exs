defmodule Vapor.HallTest do
  @moduledoc "Conversations and the terminal over HTTP — the API the console page and any client use."
  use ExUnit.Case, async: false
  alias Vapor.JSON

  setup do
    dir = Path.join(System.tmp_dir!(), "hall-#{System.unique_integer([:positive])}")
    {:ok, m} = Vapor.Majlis.start_link(dir: dir, backends: %{"echo" => %Vapor.MajlisTest.Echo{}})
    {:ok, srv} = Vapor.Serve.start_link(port: 0, model_name: "none", majlis: m, token: "s3cret")
    on_exit(fn -> File.rm_rf!(dir) end)
    {:ok, base: "http://127.0.0.1:#{Vapor.Serve.port(srv)}"}
  end

  @auth [{~c"authorization", ~c"Bearer s3cret"}]

  defp get(b, path, headers \\ @auth) do
    {:ok, {{_, status, _}, hs, body}} = :httpc.request(:get, {String.to_charlist(b <> path), headers}, [], body_format: :binary)
    ctype = hs |> Enum.find_value(fn {k, v} -> if to_string(k) == "content-type", do: to_string(v) end)
    {status, if(ctype =~ "json", do: elem(JSON.decode(body), 1), else: body)}
  end

  defp post(b, path, obj) do
    {:ok, {{_, status, _}, _, body}} =
      :httpc.request(:post, {String.to_charlist(b <> path), @auth, ~c"application/json", JSON.encode(obj)}, [timeout: 60_000], body_format: :binary)

    {status, elem(JSON.decode(body), 1)}
  end

  test "a whole conversation over HTTP: say, edit, regenerate, fork, context, export, import, search", %{base: b} do
    {200, %{"id" => t}} = post(b, "/v1/vapor/threads", %{"title" => "http", "system" => "S"})
    {200, %{"content" => c, "question" => q}} = post(b, "/v1/vapor/threads/#{t}/say", %{"text" => "first"})
    assert c == "echo: first (2 msgs)"
    {200, %{"content" => "echo: second (2 msgs)"}} = post(b, "/v1/vapor/threads/#{t}/edit", %{"node" => q, "text" => "second"})
    {200, %{"path" => %{"messages" => [m1, m2]}}} = get(b, "/v1/vapor/threads/#{t}")
    assert length(m1["siblings"]) == 2 and m1["index"] == 2
    {200, %{"node" => _}} = post(b, "/v1/vapor/threads/#{t}/regenerate", %{"node" => m2["id"]})
    {200, %{"id" => f}} = post(b, "/v1/vapor/threads/#{t}/fork", %{"node" => m1["id"], "title" => "fork"})
    {200, %{"items" => items}} = get(b, "/v1/vapor/threads/#{f}/context")
    assert [%{"status" => "sent"}] = items
    {200, %{"nodes" => nodes}} = get(b, "/v1/vapor/threads/#{t}/tree")
    assert length(nodes) == 5
    {200, md} = get(b, "/v1/vapor/threads/#{t}/export?format=markdown")
    assert md =~ "# http"
    {200, json} = get(b, "/v1/vapor/threads/#{t}/export")
    {200, %{"threads" => [_]}} = post(b, "/v1/vapor/threads/import", %{"data" => JSON.encode(json)})
    {200, %{"hits" => [_ | _]}} = post(b, "/v1/vapor/chat/search", %{"query" => "second"})
    {200, %{"threads" => ts, "models" => %{"names" => ["echo"]}, "tools" => tools}} = get(b, "/v1/vapor/threads")
    assert length(ts) == 3 and "alembic_eval" in tools
    {404, _} = get(b, "/v1/vapor/threads/tnope")
    {404, _} = post(b, "/v1/vapor/threads/#{t}/edit", %{"node" => "zz", "text" => "x"})
    {422, _} = post(b, "/v1/vapor/threads/#{t}/settings", %{"budget" => 1})
  end

  test "a shared link needs no console token — the capability is the authority — and dies when revoked", %{base: b} do
    {200, %{"id" => t}} = post(b, "/v1/vapor/threads", %{"title" => "<b>public</b>"})
    {200, _} = post(b, "/v1/vapor/threads/#{t}/say", %{"text" => "<script>alert(1)</script>"})
    {200, %{"url" => url}} = post(b, "/v1/vapor/threads/#{t}/share", %{})
    {200, html} = get(b, url, [])
    assert html =~ "&lt;script&gt;" and not (html =~ "<script>") and html =~ "&lt;b&gt;public"
    {200, %{"messages" => [_, _]}} = get(b, String.replace(url, "/shared/", "/v1/vapor/shared/"), [])
    # the rest of the API still demands the token
    {401, _} = get(b, "/v1/vapor/threads", [])
    {200, _} = post(b, "/v1/vapor/threads/#{t}/revoke", %{})
    {403, _} = get(b, url, [])
  end

  test "the terminal over HTTP: a session, its files, pipes, completion; another session cannot see them", %{base: b} do
    {200, %{"session" => s, "code" => 0}} = post(b, "/v1/vapor/diwan", %{"line" => "echo 1e16 1 -1e16 > n.txt"})
    {200, %{"out" => out, "files" => ["n.txt"]}} = post(b, "/v1/vapor/diwan", %{"session" => s, "line" => "amalgam n.txt | cat"})
    assert {:ok, %{"amalgam" => %{"value" => "1.0"}}} = JSON.decode(out)
    {200, _} = post(b, "/v1/vapor/diwan/file", %{"session" => s, "name" => "p.alb", "text" => "2^64"})
    {200, %{"text" => "2^64"}} = get(b, "/v1/vapor/diwan/file?session=#{s}&name=p.alb")
    {200, %{"completions" => cs}} = post(b, "/v1/vapor/diwan/complete", %{"session" => s, "line" => "amalgam n"})
    assert "n.txt" in cs
    {200, %{"completions" => vs}} = post(b, "/v1/vapor/diwan/complete", %{"line" => "reb"})
    assert vs == ["rebis"]
    {200, %{"session" => s2, "files" => []}} = post(b, "/v1/vapor/diwan", %{"line" => "ls"})
    refute s2 == s
    {200, %{"code" => 3, "err" => err}} = post(b, "/v1/vapor/diwan", %{"session" => s2, "line" => "amalgam n.txt"})
    assert err =~ "no such file"
    {200, %{"out" => tid}} = post(b, "/v1/vapor/diwan", %{"session" => s, "line" => "chat new --title from-terminal"})
    assert String.trim(tid) =~ ~r/^t\w+$/
  end

  test "without a majlis the conversation endpoints say how to enable them" do
    {:ok, srv} = Vapor.Serve.start_link(port: 0, model_name: "none")
    {:ok, {{_, 503, _}, _, body}} = :httpc.request(:get, {String.to_charlist("http://127.0.0.1:#{Vapor.Serve.port(srv)}/v1/vapor/threads"), []}, [], body_format: :binary)
    assert body =~ "--data"
  end
end
