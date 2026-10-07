defmodule Vapor.ConsoleMajlisTest do
  @moduledoc "The console's conversations and terminal (priv/console/majlis.js) driven in headless Chromium, and its strings."
  use ExUnit.Case, async: false

  # the per-script I18N additions and the base table of index.html, as {lang, key} → text
  defp keys(text) do
    (Regex.scan(~r/Object\.assign\(I18N\.(en|pt), \{(.*?)\n\}\);/s, text) ++ Regex.scan(~r/\n  (en|pt): \{(.*?)\n  \}/s, text))
    |> Enum.flat_map(fn [_, lang, body] -> for [_, k, v] <- Regex.scan(~r/(?:^|[{,]\s*)([A-Za-z_]\w*): "([^"]*)"/m, body), do: {{lang, k}, v} end)
    |> Map.new()
  end

  test "the conversation strings never redefine another script's key with other text, and every English key has a Portuguese one" do
    dir = Path.join(:code.priv_dir(:vapor), "console")
    mine = keys(File.read!(Path.join(dir, "majlis.js")))
    others = for f <- ["index.html", "athanor.js", "bancada.js", "mercado.js", "opus.js"], reduce: %{}, do: (acc -> Map.merge(acc, keys(File.read!(Path.join(dir, f)))))
    assert map_size(mine) > 100
    clashes = for {k, v} <- mine, Map.has_key?(others, k), others[k] != v, do: {k, v, others[k]}
    assert clashes == [], inspect(clashes)
    src = File.read!(Path.join(dir, "majlis.js"))
    [_, en] = Regex.run(~r/Object\.assign\(I18N\.en, \{(.*?)\n\}\);/s, src)
    [_, pt] = Regex.run(~r/Object\.assign\(I18N\.pt, \{(.*?)\n\}\);/s, src)
    names = fn body -> Regex.scan(~r/(?:^|[{,]\s*)([A-Za-z_]\w*):/m, body) |> Enum.map(&List.last/1) |> MapSet.new() end
    assert MapSet.difference(names.(en), names.(pt)) |> MapSet.to_list() == []
  end

  @tag :playwright
  @tag timeout: 600_000
  test "a conversation and a terminal session through the page: branches, pins, fork, compaction, search, a share link revoked; the jail" do
    dir = Path.join(System.tmp_dir!(), "console-majlis-#{System.unique_integer([:positive])}")
    {:ok, m} = Vapor.Majlis.start_link(dir: dir, backends: %{"echo" => %Vapor.Quality.Round16.Echo{}}, default: "echo")
    {:ok, srv} = Vapor.Serve.start_link(port: 0, model_name: "none", majlis: m)
    base = "http://127.0.0.1:#{Vapor.Serve.port(srv)}/"
    js = Path.expand("../js/console_majlis.mjs", __DIR__)

    try do
      {out, 0} = System.cmd("node", [js, base] ++ List.wrap(System.get_env("VAPOR_SHOTS")), stderr_to_stdout: true)
      {:ok, j} = out |> String.split("\n", trim: true) |> List.last() |> Vapor.JSON.decode()
      assert j["failures"] == [] and j["errors"] == [], inspect(j)
      assert length(j["checks"]) >= 19
    after
      File.rm_rf!(dir)
    end
  end
end
