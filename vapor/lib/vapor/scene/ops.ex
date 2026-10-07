defmodule Vapor.Scene.Ops do
  @moduledoc """
  Scenes as documents edited by operations (docs/CENA.md §9). A scene is
  its base (an analysed picture, or a blank canvas) plus a log of
  operations; the engine replays the log on load, so the document is the
  whole history and every edit is a line of text — from a person, a
  pipeline (`vapor scene edit`) or a model (`Vapor.Mind.direct/3`).

      add circle sun { x: 0.8, y: 0.15 + 0.02*sin(t/3), r: 0.06, color: "#ffcc55", glow: 1 }
      add particles snow { count: 120, x: noise(i, 1), y: fract(noise(i, 2) + t*0.05), r: 0.003, color: "#fff" }
      add trail orbit { x: 0.5 + 0.3*cos(t), y: 0.5 + 0.3*sin(t), length: 2, color: hsl(200 + 40*sin(t), 80, 60) }
      add text title { x: 0.5, y: 0.08, text: "vapor", size: 0.06, color: "#e9efef" }
      set sun.r = 0.08
      set world.weather = "rain"           # weather, time, wind, camera, entropy, cycle, speed
      remove sun
      at 5: set sun.color = "#ff7744"
      direct "night, light rain, three people walking"

  Every numeric property may be an expression of `t` (seconds), `i`
  (index within `count`), `n` (the count), `u` (= i/(n−1)) and `aspect`
  — Alembic's numeric subset (`Vapor.Alembic.Tree`), sent to the browser
  as a tree it interprets. Positions are fractions of the frame (x across,
  y down); `z` is depth (1 = nearest, larger is farther).
  """
  alias Vapor.Alembic.Tree

  @kinds ~w(circle ring rect line text glow particles trail)
  @vars ~w(t i n u aspect)
  @world ~w(weather time wind camera entropy cycle speed animate)
  @string_props ~w(color text font stroke fill)

  def kinds, do: @kinds

  @doc "A blank scene (no picture): a sky gradient over a frame of w × h."
  def blank(opts \\ []) do
    w = Keyword.get(opts, :w, 960)
    h = Keyword.get(opts, :h, 600)
    %{"w" => w, "h" => h, "horizon" => Keyword.get(opts, :horizon, 0.62), "layers" => [], "walk" => %{"cols" => 0, "rows" => 0, "cells" => []},
      "bg" => Keyword.get(opts, :bg, ["#0b1424", "#2a3550"]), "seed" => Keyword.get(opts, :seed, 1), "entropy" => 0.5, "ops" => [], "entities" => []}
  end

  @doc """
  Parse operations text: `{:ok, ops, problems}` — `ops` are maps the
  engine applies; `problems` name each line not understood (never guessed).
  """
  def parse(text) do
    {ops, probs} =
      text
      |> String.split("\n")
      |> Enum.with_index(1)
      |> Enum.reduce({[], []}, fn {line, n}, {ops, probs} ->
        l = line |> String.replace(~r/(^|\s)#.*$/, "") |> String.trim()
        if l == "" do
          {ops, probs}
        else
          case op(l) do
            {:ok, o} -> {ops ++ List.wrap(o), probs}
            {:error, m} -> {ops, probs ++ ["line #{n}: #{m}"]}
          end
        end
      end)

    {:ok, ops, probs}
  end

  defp op("at " <> rest) do
    case Regex.run(~r/^([\d.]+)\s*s?\s*:\s*(.+)$/, String.trim(rest)) do
      [_, t, inner] ->
        with {:ok, o} <- op(String.trim(inner)) do
          {:ok, o |> List.wrap() |> Enum.map(&Map.put(&1, "at", elem(Float.parse(t), 0)))}
        end
      _ -> {:error, "`at T: operation`"}
    end
  end

  defp op("add " <> rest) do
    case Regex.run(~r/^([a-z]+)\s+([\p{L}_][\w\-]*)\s*(\{.*\})?\s*$/u, String.trim(rest)) do
      [_, kind | more] ->
        id = hd(more)
        body = Enum.at(more, 1) || "{}"
        cond do
          kind not in @kinds -> {:error, "unknown kind #{kind} (#{Enum.join(@kinds, ", ")})"}
          true -> with {:ok, props} <- props(body), do: {:ok, %{"entity" => %{"id" => id, "kind" => kind, "props" => props}}}
        end
      _ -> {:error, "`add KIND NAME { key: value, … }`"}
    end
  end

  defp op("set " <> rest) do
    case Regex.run(~r/^([\p{L}_][\w\-]*)\.([\p{L}_]\w*)\s*=\s*(.+)$/u, String.trim(rest)) do
      [_, "world", key, value] ->
        if key in @world, do: world(key, value), else: {:error, "world.#{key}: one of #{Enum.join(@world, ", ")}"}
      [_, id, key, value] -> with {:ok, v} <- value(key, value), do: {:ok, %{"set" => %{"id" => id, "key" => key, "value" => v}}}
      _ -> {:error, "`set NAME.key = value`"}
    end
  end

  defp op("remove " <> id), do: {:ok, %{"remove_entity" => String.trim(id)}}
  defp op("clear"), do: {:ok, %{"clear_entities" => true}}

  defp op("direct " <> words) do
    w = words |> String.trim() |> String.trim("\"")
    r = Vapor.Scene.direct(w)
    if r.ops == [], do: {:error, "the vocabulary understood nothing in #{inspect(w)}" <> unknown(r.unknown)}, else: {:ok, r.ops}
  end

  defp op(other), do: {:error, "not an operation: #{String.slice(other, 0, 60)} (add, set, remove, clear, at, direct)"}

  defp unknown([]), do: ""
  defp unknown(ws), do: " (unknown: #{Enum.join(ws, ", ")})"

  defp world(key, value) do
    v = value |> String.trim() |> String.trim("\"")
    case Float.parse(v) do
      {x, ""} -> {:ok, %{key => x}}
      _ -> if v in ["true", "false"], do: {:ok, %{key => v == "true"}}, else: {:ok, %{key => v}}
    end
  end

  # { key: value, … } — values may contain commas inside parentheses
  defp props(body) do
    inner = body |> String.trim() |> String.trim_leading("{") |> String.trim_trailing("}")
    inner
    |> split_top(",")
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == ""))
    |> Enum.reduce_while({:ok, %{}}, fn kv, {:ok, acc} ->
      case String.split(kv, ":", parts: 2) do
        [k, v] ->
          k = String.trim(k)
          if Regex.match?(~r/^[a-z_][a-z0-9_]*$/, k) do
            case value(k, v) do
              {:ok, val} -> {:cont, {:ok, Map.put(acc, k, val)}}
              {:error, m} -> {:halt, {:error, "#{k}: #{m}"}}
            end
          else
            {:halt, {:error, "property names are lowercase words: #{k}"}}
          end
        _ -> {:halt, {:error, "`key: value` expected, got #{kv}"}}
      end
    end)
  end

  defp split_top(s, sep) do
    {parts, cur, _} =
      s |> String.graphemes() |> Enum.reduce({[], "", 0}, fn ch, {parts, cur, d} ->
        cond do
          ch in ["(", "["] -> {parts, cur <> ch, d + 1}
          ch in [")", "]"] -> {parts, cur <> ch, d - 1}
          ch == "\"" -> {parts, cur <> ch, d}
          ch == sep and d == 0 -> {parts ++ [cur], "", d}
          true -> {parts, cur <> ch, d}
        end
      end)
    parts ++ [cur]
  end

  @doc "A property value: a string, a colour function, or a numeric expression tree."
  def value(key, text) do
    v = String.trim(text)
    cond do
      String.starts_with?(v, "\"") and String.ends_with?(v, "\"") and byte_size(v) >= 2 ->
        s = String.slice(v, 1..-2//1)
        if String.length(s) > 400, do: {:error, "text longer than 400 characters"}, else: {:ok, s}

      key in @string_props and Regex.match?(~r/^#[0-9a-fA-F]{3,8}$/, v) -> {:ok, v}

      m = Regex.run(~r/^(hsl|rgb)\((.*)\)$/, v) ->
        [_, f, args] = m
        parts = split_top(args, ",")
        if length(parts) in [3, 4] do
          trees = Enum.map(parts, &Tree.parse(String.trim(&1), @vars))
          case Enum.find(trees, &match?({:error, _}, &1)) do
            nil -> {:ok, %{f => Enum.map(trees, &elem(&1, 1))}}
            {:error, e} -> {:error, e}
          end
        else
          {:error, "#{f}(a, b, c[, alpha])"}
        end

      key in @string_props -> {:ok, v}
      true -> Tree.parse(v, @vars)
    end
  end

  @doc "Append operations to a scene document (validated); `{:ok, scene, problems}`."
  def apply_text(scene, text) do
    {:ok, ops, probs} = parse(text)
    {:ok, Map.update(scene, "ops", ops, &(&1 ++ ops)), probs}
  end

  @doc "What a scene holds, in a few lines (for people and for a model directing it)."
  def summary(scene) do
    ops = scene["ops"] || []
    ents = ops |> Enum.reduce(%{}, fn
      %{"entity" => e}, acc -> Map.put(acc, e["id"], e["kind"])
      %{"remove_entity" => id}, acc -> Map.delete(acc, id)
      %{"clear_entities" => true}, _ -> %{}
      _, acc -> acc
    end)
    layers = length(scene["layers"] || [])
    "frame #{scene["w"]}×#{scene["h"]}; #{if layers > 0, do: "#{layers} picture layers", else: "blank canvas"}; entities: " <>
      if(ents == %{}, do: "none", else: Enum.map_join(ents, ", ", fn {id, k} -> "#{id} (#{k})" end)) <> "; #{length(ops)} operations so far"
  end

  @doc "The reference for operations (people and models)."
  def card do
    """
    SCENE OPERATIONS — one per line
      add KIND NAME { key: value, … }      KIND: #{Enum.join(@kinds, " ")}
      set NAME.key = value                 change one property
      set world.KEY = value                KEY: #{Enum.join(@world, " ")} (weather: clear rain storm snow fog; time: dawn day dusk night)
      remove NAME · clear · at SECONDS: operation · direct "words for the vocabulary"
    Properties: x y (fractions of the frame, y down) z (depth, 1 nearest) r w h x2 y2 size alpha rot glow count length
      color fill stroke ("#rrggbb" or hsl(h, s, l[, a]) or rgb(r, g, b[, a])) text font
    Numbers may be expressions of t (seconds), i, n, u (=i/(n−1)) and aspect, using + − * / ^ % and
      #{Enum.join(Tree.functions(), " ")} — e.g. y: 0.5 + 0.1*sin(2*t + u*tau)
    """
  end
end
