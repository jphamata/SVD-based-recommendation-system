defmodule Vapor.TemplateTest do
  @moduledoc """
  Chat templates (`Vapor.Template`) against jinja2 in the environment
  transformers renders them in (`test/python/jinja_render.py`): real
  templates of 20 model families (from llama.cpp's `models/templates`) on
  scenarios with system prompts, multi-turn chats, tool definitions, tool
  calls (also parallel, also with string arguments), tool results,
  reasoning, multimodal list content and documents; and feature snippets
  (scoping, loop controls, macros, filters, tests, slices, Python `repr`).
  Equality is byte for byte, errors included.
  """
  use ExUnit.Case, async: true
  alias Vapor.Template
  import Vapor.TestHelpers

  @dir Path.expand("../fixtures/templates", __DIR__)
  @now ~N[2026-10-01 12:00:00]

  test "pure: whitespace control, scoping, and the clock as an explicit input" do
    assert {:ok, "a|b"} = Template.render("{{- 'a' -}}  |  {{- 'b' }}", %{})
    assert {:ok, "1|0"} = Template.render("{% set x = 0 %}{% for i in [1] %}{% set x = i %}{{ x }}{% endfor %}|{{ x }}", %{})
    assert {:error, "strftime_now needs an explicit clock" <> _} = Template.render("{{ strftime_now('%Y') }}", %{})
    assert {:ok, "2026"} = Template.render("{{ strftime_now('%Y') }}", %{}, now: @now)
    assert {:error, {:raised, "no"}} = Template.render("{{ raise_exception('no') }}", %{})
    assert {:error, {:syntax, _}} = Template.compile("{% if x %}")
    assert {:error, {:syntax, "unsupported tag {% include %}"}} = Template.compile("{% include 'x' %}")
    # dicts keep their order (as Python's do)
    {:ok, vars} = Vapor.JSON.decode(~s({"d": {"b": 1, "a": 2}}), ordered: true)
    assert {:ok, ~s({"b": 1, "a": 2})} = Template.render("{{ d | tojson }}", Map.new(elem(vars, 1)))
  end

  @tag :jinja2
  test "20 real chat templates × 12 scenarios: byte-identical to jinja2 under transformers' environment" do
    names = @dir |> File.ls!() |> Enum.filter(&String.ends_with?(&1, ".jinja")) |> Enum.sort()
    templates = Map.new(names, &{&1, File.read!(Path.join(@dir, &1))})
    scen_src = File.read!(Path.expand("../fixtures/template_scenarios.json", __DIR__))
    {:ok, scenarios} = Vapor.JSON.decode(scen_src, ordered: true)
    req = "{\"templates\": " <> Vapor.JSON.encode(templates) <> ", \"scenarios\": " <> scen_src <> "}"
    ref = Vapor.JSON.decode!(py!(File.read!(Path.expand("../python/jinja_render.py", __DIR__)), [], req))

    results =
      for name <- names, {:dict, sc} <- scenarios do
        {_, sname} = List.keyfind(sc, "name", 0)
        {_, {:dict, vars}} = List.keyfind(sc, "vars", 0)
        {name, sname, Template.render(templates[name], Map.new(vars), now: @now), ref[name][sname]}
      end

    bad = Enum.reject(results, fn {_, _, got, want} -> same?(got, want) end)
    rendered = Enum.count(results, &match?({_, _, {:ok, _}, _}, &1))
    IO.puts("\n  templates: #{length(results)} renderings (#{rendered} texts, #{length(results) - rendered} template-raised errors), #{length(bad)} different")
    assert bad == [], inspect(Enum.take(bad, 2), limit: :infinity, printable_limit: 400)
  end

  @tag :jinja2
  test "feature snippets: byte-identical to jinja2" do
    src = File.read!(Path.expand("../fixtures/template_snippets.json", __DIR__))
    snippets = Vapor.JSON.decode!(src)
    ref = Vapor.JSON.decode!(py!(File.read!(Path.expand("../python/jinja_render.py", __DIR__)), ["snippets"], src))

    bad =
      for {[name, tsrc], want} <- Enum.zip(snippets, ref),
          got = Template.render(tsrc, %{}, now: @now),
          not same?(got, want),
          do: {name, got, want}

    assert bad == [], inspect(bad, limit: :infinity, printable_limit: 600)
  end

  defp same?({:ok, t}, %{"ok" => t}), do: true
  defp same?({:error, {:raised, m}}, %{"error" => m, "raised" => true}), do: true
  defp same?({:error, _}, %{"error" => _, "raised" => false}), do: true
  defp same?(_, _), do: false
end
