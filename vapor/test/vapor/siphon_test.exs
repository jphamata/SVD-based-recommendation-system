defmodule Vapor.SiphonTest do
  @moduledoc """
  The network airlock (`Vapor.Siphon`): fetchers the person declares, run
  only by the person, through the format airlocks, with a receipt; agents
  only propose. Every fetcher here is local (`cp`, `sh`): the airlock does
  not care where bytes come from, and the tests need no network.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Lock, Siphon, Tensor}
  alias Vapor.Ingest.Safetensors
  import Vapor.TestHelpers

  setup do
    home = Path.join(System.tmp_dir!(), "vapor-siphon-test-#{System.unique_integer([:positive])}")
    File.mkdir_p!(home)
    on_exit(fn -> File.rm_rf!(home) end)

    doc = %{"fetchers" => [
      %{"name" => "cp", "argv" => ["cp", "{ref}", "{out}/"]},
      %{"name" => "slow", "argv" => ["sh", "-c", "sleep 30; : {out}"], "timeout_s" => 1},
      %{"name" => "big", "argv" => ["sh", "-c", "head -c 5000000 /dev/zero > {out}/blob; sleep 10"], "max_bytes" => 1_000_000},
      %{"name" => "env", "argv" => ["sh", "-c", "env > {out}/env.txt"], "env" => ["VAPOR_SIPHON_ALLOWED"]},
      %{"name" => "range", "argv" => ["sh", "-c", "a=${0%-*}; b=${0#*-}; tail -c +$((a+1)) \"$1\" | head -c $((b-a+1)) > \"$2/part\"", "{range}", "{ref}", "{out}"]},
      %{"name" => "fail", "argv" => ["sh", "-c", "echo nope; exit 7; : {out}"]}
    ]}

    File.write!(Path.join(home, "siphons.json"), Vapor.JSON.encode(doc))
    {:ok, home: home, opts: [home: home]}
  end

  defp f(name, opts), do: elem(Siphon.find(name, opts), 1)

  defp checkpoint(dir) do
    File.mkdir_p!(dir)
    {:ok, c} = Vapor.Model.Config.from_map(tiny_config("qwen2"))
    path = Path.join(dir, "model.safetensors")
    :ok = Safetensors.write(path, tiny_weights(c))
    File.write!(Path.join(dir, "config.json"), Vapor.JSON.encode(tiny_config("qwen2")))
    path
  end

  test "a fetch the person runs lands with its receipt: digests, formats, the exact argv", %{home: home, opts: o} do
    src = checkpoint(Path.join(home, "src"))
    into = Path.join(home, "into")
    assert {:ok, r} = Siphon.run(f("cp", o), src, Keyword.put(o, :into, into))
    assert [%{"path" => "model.safetensors", "format" => "safetensors", "sha256" => sha}] = r["files"]
    assert sha == :crypto.hash(:sha256, File.read!(src)) |> Base.encode16(case: :lower)
    assert r["argv"] == [System.find_executable("cp"), src, "{out}/"] and r["proposal"] == nil
    assert File.read!(Path.join(into, "model.safetensors")) == File.read!(src)
    assert {:ok, %{"id" => _}} = Vapor.JSON.decode(File.read!(Path.join(into, "RECEIPT.json")))
  end

  test "nothing lands from a malformed file, a wrong digest or a failing fetcher", %{home: home, opts: o} do
    bad = Path.join(home, "bad.safetensors")
    File.write!(bad, <<200::64-little, "not json">>)
    into = Path.join(home, "into")
    assert {:error, %{node: {:siphon, "bad.safetensors"}}} = Siphon.run(f("cp", o), bad, Keyword.put(o, :into, into))
    refute File.exists?(into)

    src = checkpoint(Path.join(home, "src"))
    assert {:error, r} = Siphon.run(f("cp", o), src, o ++ [into: into, sha256: String.duplicate("0", 64)])
    assert r.bound =~ "SHA-256" and not File.exists?(into)

    assert {:error, r} = Siphon.run(f("fail", o), "x", o)
    assert r.bound =~ "status 7" and r.repair =~ "nope"
  end

  test "the deadline and the byte cap kill the fetcher", %{opts: o} do
    {t, {:error, r}} = :timer.tc(fn -> Siphon.run(f("slow", o), "x", o) end)
    assert r.bound =~ "timeout_s" and t < 10_000_000
    {t, {:error, r}} = :timer.tc(fn -> Siphon.run(f("big", o), "x", o) end)
    assert r.bound =~ "max_bytes" and t < 9_000_000
  end

  test "the fetcher's environment is what its declaration lists, and nothing else", %{home: home, opts: o} do
    System.put_env("VAPOR_SIPHON_ALLOWED", "yes")
    System.put_env("VAPOR_SIPHON_SECRET", "no")

    try do
      {:ok, r} = Siphon.run(f("env", o), "x", Keyword.put(o, :into, Path.join(home, "e")))
      env = File.read!(Path.join(r["into"], "env.txt"))
      assert env =~ "VAPOR_SIPHON_ALLOWED=yes" and env =~ "PATH="
      refute env =~ "VAPOR_SIPHON_SECRET"
    after
      System.delete_env("VAPOR_SIPHON_ALLOWED")
      System.delete_env("VAPOR_SIPHON_SECRET")
    end
  end

  test "a ref that could be an option is refused; a ref is one argument, never a shell word", %{home: home, opts: o} do
    assert {:error, %{node: {:siphon, :ref}}} = Siphon.run(f("cp", o), "-rf", o)
    assert {:error, %{node: {:siphon, :ref}}} = Siphon.propose("cp", "a\nb", "why", o)
    marker = Path.join(home, "marker")
    File.write!(marker, "here")
    # with a shell this would remove the marker; as one argument it is a file that does not exist
    assert {:error, _} = Siphon.run(f("cp", o), "nothing; rm #{marker}", o)
    assert File.exists?(marker)
  end

  test "an agent only proposes: one MCP tool, which fetches nothing; the person approves", %{home: home, opts: o} do
    names = Vapor.MCP.Server.tools() |> Enum.map(& &1["name"]) |> Enum.filter(&String.contains?(&1, "siphon"))
    assert names == ["siphon_propose"]

    src = checkpoint(Path.join(home, "src"))
    {:ok, p} = Siphon.propose("cp", src, "the tiny model for the test", Keyword.put(o, :by, "an agent over MCP"))
    assert [%{"id" => id}] = Siphon.queue(o)
    assert id == p["id"]
    refute File.exists?(Path.join([home, "siphon", "store"]))

    {:ok, r} = Siphon.approve(id, o)
    assert r["proposal"]["by"] == "an agent over MCP" and r["approved_by"] != nil
    assert File.exists?(Path.join(r["into"], "model.safetensors"))
    assert Siphon.queue(o) == []
    assert {:error, _} = Siphon.approve(id, o)
  end

  test "outside the siphon, only the agent backends open connections", %{} do
    root = Path.expand("../../lib", __DIR__)
    opening = for p <- Path.wildcard(Path.join(root, "**/*.ex")),
                  File.read!(p) =~ ~r/:httpc\.request|:gen_tcp\.connect|:ssl\.connect|:socket\.connect/,
                  do: Path.relative_to(p, root)

    assert opening == ["vapor/agent/backends.ex"]
  end

  test "headers before data: a checkpoint admitted from its config and its header alone", %{home: home, opts: o} do
    src = checkpoint(Path.join(home, "src"))
    # the remote file, as far as the fetcher is concerned; only its first bytes are ever read
    {:ok, table, data_len} = Siphon.headers(f("range", o), src, o)
    assert data_len == File.stat!(src).size - (8 + (File.read!(src) |> binary_part(0, 8) |> :binary.decode_unsigned(:little)))
    {:ok, ws} = Safetensors.read(src)
    assert Map.new(ws, fn {n, %Tensor{shape: s}} -> {n, s} end) == Map.new(table, fn {n, {_, s}} -> {n, s} end)

    cfg = Vapor.JSON.decode!(File.read!(Path.join(Path.dirname(src), "config.json")))
    assert {:ok, %{missing: [], unread: [], spec: %{family: "qwen2"}}} = Lock.preflight(cfg, table)
    # a missing tensor and a wrong shape are named from the headers alone
    table2 = table |> Map.delete("model.norm.weight") |> Map.put("lm_head.weight", {"F32", [3, 3]})
    {:ok, v} = Lock.preflight(cfg, table2)
    assert {"model.norm.weight", [64], nil} in v.missing
    # and a configuration the airlock refuses is refused before any weight
    assert {:error, _} = Lock.preflight(Map.put(cfg, "attn_logit_softcapping", 30.0), table)
    # a ranged fetcher used without a range, and a whole-file fetcher asked for one, are refused by name
    assert {:error, %{bound: b}} = Siphon.run(f("range", o), src, o)
    assert b =~ "byte range"
    assert {:error, %{bound: b}} = Siphon.run(f("cp", o), src, Keyword.put(o, :range, "0-7"))
    assert b =~ "{range}"
  end

  test "declarations are checked", %{opts: o} do
    bad = fn d -> Siphon.fetchers(Keyword.put(o, :fetchers, %{"fetchers" => [d]})) end
    assert {:error, _} = bad.(%{"name" => "x", "argv" => ["cp", "{ref}"]})
    assert {:error, _} = bad.(%{"name" => "Bad Name", "argv" => ["cp", "{out}"]})
    assert {:error, _} = bad.(%{"name" => "x", "argv" => ["cp", "{out}"], "env" => ["$(rm)"]})
    assert {:error, _} = bad.(%{"name" => "x", "argv" => ["cp", "{out}"], "max_bytes" => 0})
    assert {:ok, [%{name: "x"}]} = bad.(%{"name" => "x", "argv" => ["cp", "{ref}", "{out}"]})
    assert {:error, %{bound: b}} = Siphon.find("nope", o)
    assert b =~ "cp"
  end

  test "bin/vapor reads the person's $VAPOR_HOME, not its own directory", %{home: home} do
    # the launcher once named the repository VAPOR_HOME, which replaced an exported one for every verb
    root = Path.expand("../..", __DIR__)
    {out, 0} = System.cmd(Path.join(root, "bin/vapor"), ["siphon", "fetchers", "--json"], env: [{"VAPOR_HOME", home}, {"MIX_ENV", "test"}])
    {:ok, fs} = Vapor.JSON.decode(out)
    assert "cp" in Enum.map(fs, & &1["name"])
  end

end
