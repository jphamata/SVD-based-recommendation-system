defmodule Vapor.Main.SiphonCli do
  @moduledoc false
  # vapor siphon — the network airlock (docs/SIPHON.md): the person's own surface
  import Vapor.Main
  alias Vapor.{JSON, Lock, Siphon}

  def run(argv) do
    case opts(argv, [into: :string, sha256: :string, json: :boolean, by: :string]) do
      :usage -> 2
      {:ok, _, []} -> out(help()); 0
      {:ok, o, ["fetchers"]} -> fetchers(o)
      {:ok, o, ["queue"]} -> show(Siphon.queue(), o)
      {:ok, o, ["propose", name, ref | why]} -> result(Siphon.propose(name, ref, Enum.join(why, " "), by: o[:by] || "the command line"), o)
      {:ok, o, ["approve", id]} -> result(Siphon.approve(id, take(o)), o)
      {:ok, _, ["reject", id]} -> Siphon.reject(id); out("dropped #{id}"); 0
      {:ok, o, ["run", name, ref]} -> with_fetcher(name, &result(Siphon.run(&1, ref, take(o)), o))
      {:ok, o, ["headers", name, ref]} -> with_fetcher(name, &headers(&1, ref, o))
      {:ok, o, ["preflight", name, config_ref | shards]} when shards != [] -> with_fetcher(name, &preflight(&1, config_ref, shards, o))
      {:ok, _, _} -> err(help()); 2
    end
  end

  defp take(o), do: Keyword.take(o, [:into, :sha256])

  defp with_fetcher(name, fun) do
    case Siphon.find(name) do
      {:ok, f} -> fun.(f)
      {:error, r} -> err("siphon: #{r.bound}"); 3
    end
  end

  defp fetchers(o) do
    case Siphon.fetchers() do
      {:ok, fs} ->
        if json?(o), do: emit_json(Enum.map(fs, &Map.from_struct/1)), else: Enum.each(fs, &out("#{bold(&1.name)}  #{Enum.join(&1.argv, " ")}  env: #{Enum.join(&1.env, ", ")}"))
        if fs == [], do: out("no fetchers declared: write #{Path.join(Siphon.home(), "siphons.json")} (docs/SIPHON.md)")
        0

      {:error, r} ->
        err("siphon: #{r.bound}"); 3
    end
  end

  defp show(list, o) do
    if json?(o), do: emit_json(list), else: Enum.each(list, &out("#{bold(&1["id"])}  #{&1["fetcher"]} #{&1["ref"]}  — #{&1["why"]} (#{&1["by"]}, #{&1["at"]})"))
    0
  end

  defp result({:ok, r}, o) do
    if json?(o), do: emit_json(r), else: out(JSON.encode(r))
    0
  end

  defp result({:error, r}, _o), do: (err("siphon: #{r.bound} — #{r.repair}"); 1)

  defp headers(f, ref, o) do
    case Siphon.headers(f, ref) do
      {:ok, table, len} ->
        rows = table |> Enum.sort() |> Enum.map(fn {n, {d, s}} -> %{"name" => n, "dtype" => d, "shape" => s} end)
        if json?(o), do: emit_json(%{"tensors" => rows, "data_bytes" => len}), else: out("#{length(rows)} tensors, #{len} bytes of data (not fetched)")
        0

      {:error, r} ->
        err("siphon: #{r.bound}"); 1
    end
  end

  # the configuration (a small file), every shard's header — and the airlock's verdict, before any weight
  defp preflight(f, config_ref, shards, o) do
    fetch_config = fn ->
      case Siphon.find(f.name <> "-file") do
        {:ok, ff} -> Siphon.run(ff, config_ref, into: Path.join(System.tmp_dir!(), "vapor-preflight-#{System.unique_integer([:positive])}"))
        _ -> {:error, Vapor.Rejection.new({:siphon, f.name <> "-file"}, "a whole-file fetcher named #{f.name}-file for config.json", "declare one")}
      end
    end

    with {:ok, r} <- fetch_config.(),
         {:ok, cfg} <- JSON.decode(File.read!(Path.join(r["into"], hd(r["files"])["path"]))),
         {:ok, table} <- Enum.reduce_while(shards, {:ok, %{}}, fn s, {:ok, acc} ->
                           case Siphon.headers(f, s) do
                             {:ok, t, _} -> {:cont, {:ok, Map.merge(acc, t)}}
                             err -> {:halt, err}
                           end
                         end),
         {:ok, v} <- Lock.preflight(cfg, table) do
      summary = %{"admitted" => v.spec.family, "adapter" => Lock.id(v.spec.adapter), "expected" => v.expected,
                  "missing" => Enum.map(v.missing, fn {n, w, g} -> %{"name" => n, "want" => w, "got" => g} end), "unread" => v.unread}
      if json?(o), do: emit_json(summary), else: out(JSON.encode(summary))
      if v.missing == [], do: 0, else: 1
    else
      {:error, %Vapor.Rejection{} = r} -> err("siphon: #{inspect(r.node)} — #{r.bound}"); 1
      {:error, why} -> err("siphon: #{inspect(why)}"); 1
    end
  end

  defp help do
    bold("vapor siphon") <> " — the network airlock: fetchers you declare, run only when you say so (docs/SIPHON.md)\n" <>
      "  vapor siphon fetchers                     the declared fetchers ($VAPOR_HOME/siphons.json)\n" <>
      "  vapor siphon run NAME REF [--into DIR] [--sha256 HEX]   fetch now, through the format airlocks, with a receipt\n" <>
      "  vapor siphon queue | approve ID | reject ID             requests agents proposed; nothing is fetched until approved\n" <>
      "  vapor siphon propose NAME REF WHY…        queue a request (what an agent's siphon_propose does)\n" <>
      "  vapor siphon headers NAME REF             a remote .safetensors header by two ranged fetches, no data\n" <>
      "  vapor siphon preflight NAME CONFIG SHARD… admit a checkpoint from its config and headers alone"
  end
end
