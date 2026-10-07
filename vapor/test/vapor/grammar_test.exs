defmodule Vapor.GrammarTest do
  @moduledoc """
  Byte-level grammars (`Vapor.Grammar`), JSON Schema compilation, token
  masks over a real vocabulary, and tool-call dialects (`Vapor.Tools`).
  """
  use ExUnit.Case, async: true
  alias Vapor.Grammar
  alias Vapor.Grammar.{Constraint, JSONSchema, Vocab}
  alias Vapor.Tools

  defp accepts?(g, text) do
    case Grammar.advance(g, text) do
      {:ok, g2} -> Grammar.complete?(g2)
      :reject -> false
    end
  end

  # random JSON values (depth-bounded)
  defp rand_json(0), do: Enum.random([nil, true, false, :rand.uniform(1000) - 500, :rand.uniform() * 1.0e3, "s#{:rand.uniform(99)}é\"\\\n"])

  defp rand_json(d) do
    case :rand.uniform(3) do
      1 -> for _ <- 1..:rand.uniform(3), do: rand_json(d - 1)
      2 -> Map.new(1..:rand.uniform(3), fn i -> {"k#{i}", rand_json(d - 1)} end)
      3 -> rand_json(0)
    end
  end

  test "any JSON value: 300 random documents accepted, malformed ones rejected" do
    :rand.seed(:exsss, {1, 1, 1})
    g = Grammar.new({:ref, :value})
    for _ <- 1..300, do: assert(accepts?(g, Vapor.JSON.encode(rand_json(3))))
    for bad <- ["{", "[1,]", "{\"a\" 1}", "01", "1.", "\"\\x\"", "\"\x01\"", "tru", "{\"a\":1,}", "[1 2]", "\"\xC3(\"",
                "\"\xED\xA0\x80\"", "\"\xE0\x80\x80\"", "\"\xF4\x90\x80\x80\"", "\"\xC0\xAF\""],
        do: refute(accepts?(g, bad), inspect(bad))
  end

  test "JSON Schema: types, enums, required/optional order, bounds, refs, rejections" do
    {:ok, g} =
      JSONSchema.compile(%{"type" => "object", "required" => ["b", "a"],
                           "properties" => %{"a" => %{"type" => "integer"}, "b" => %{"enum" => ["x", 2]},
                                             "c" => %{"type" => "array", "items" => %{"$ref" => "#/$defs/n"}, "minItems" => 1, "maxItems" => 2},
                                             "d" => %{"type" => "string", "minLength" => 2, "maxLength" => 3}},
                           "$defs" => %{"n" => %{"type" => ["number", "null"]}}})

    assert accepts?(g, ~s({"b": "x", "a": -3}))
    assert accepts?(g, ~s({"b":2,"a":0,"c":[1.5e3, null],"d":"ab"}))
    refute accepts?(g, ~s({"a": 1, "b": "x"}))
    refute accepts?(g, ~s({"b": "y", "a": 1}))
    refute accepts?(g, ~s({"b": "x", "a": 1.5}))
    refute accepts?(g, ~s({"b": "x", "a": 1, "c": []}))
    refute accepts?(g, ~s({"b": "x", "a": 1, "c": [1, 2, 3]}))
    refute accepts?(g, ~s({"b": "x", "a": 1, "d": "abcd"}))
    refute accepts?(g, ~s({"b": "x", "a": 1, "d": "a"}))
    refute accepts?(g, ~s({"b": "x", "a": 1, "z": 0}))

    # pattern is enforced since 0.7 (test/vapor/regex_test.exs); what the grammar cannot express is still refused
    assert {:ok, pg} = JSONSchema.compile(%{"type" => "string", "pattern" => "^a"})
    assert accepts?(pg, ~s("abc")) and not accepts?(pg, ~s("ba"))
    assert {:error, %Vapor.Rejection{node: {:json_schema, "pattern"}}} = JSONSchema.compile(%{"type" => "string", "pattern" => "(?=a)"})
    assert {:error, %Vapor.Rejection{node: {:json_schema, "minimum"}}} = JSONSchema.compile(%{"type" => "integer", "minimum" => 3})
    assert {:ok, _} = JSONSchema.compile(%{"type" => "integer", "minimum" => 3}, lenient: true)
  end

  # the bytes the grammar accepts from here
  defp live_bytes(g), do: Enum.filter(0..255, &match?({:ok, _}, Grammar.advance(g, <<&1>>)))

  test "no dead ends: every accepted prefix can still be completed (escapes at the length bound, surrogates)" do
    g1 = fn max -> elem(JSONSchema.compile(%{"type" => "string", "maxLength" => max}), 1) end

    # an escape is a character: at the bound it cannot start
    refute accepts?(g1.(3), ~s("abc\\n"))
    assert Grammar.advance(g1.(3), ~s("abc\\)) == :reject or Grammar.dead?(elem(Grammar.advance(g1.(3), ~s("abc\\)), 1))
    assert accepts?(g1.(3), ~s("ab\\n"))

    # a surrogate pair is one character; lone surrogates are refused at once
    assert accepts?(g1.(1), ~s("\\ud83d\\ude00"))
    refute accepts?(g1.(1), ~s("\\ud83d\\ude00x"))
    assert {:ok, %{"s" => "😀"}} = Vapor.JSON.decode(~s({"s":"\\ud83d\\ude00"}))
    for bad <- [~s("\\udc00"), ~s("\\ud83d"), ~s("\\ud83dx"), ~s("\\ud83d\\u0041")], do: refute(accepts?(g1.(4), bad), bad)
    assert {:ok, _} = Grammar.advance(g1.(4), "\"\\ud8")
    assert Grammar.advance(g1.(4), "\"\\udc") == :reject

    # random walks biased toward the delicate bytes: never stuck before the end
    :rand.seed(:exsss, {7, 7, 7})
    special = ~c"\\u\"dDcCeEfF89aAbB0}]:,"

    schema = %{"type" => "object", "required" => ["s", "t", "e"],
               "properties" => %{"s" => %{"type" => "string", "maxLength" => 2}, "t" => %{"type" => "string", "minLength" => 1, "maxLength" => 3},
                                 "e" => %{"enum" => ["\u00e9", "a\\b"]}, "n" => %{"type" => "array", "items" => %{"type" => "integer"}, "maxItems" => 2}}}

    {:ok, gs} = JSONSchema.compile(schema)

    for {g0, check} <- [{g1.(0), nil}, {g1.(1), nil}, {g1.(2), nil}, {gs, schema}], _ <- 1..60 do
      {text, g} =
        Enum.reduce_while(1..400, {"", g0}, fn _, {text, g} ->
          live = live_bytes(g)
          cond do
            Grammar.complete?(g) and (live == [] or :rand.uniform() < 0.15) -> {:halt, {text, g}}
            true ->
              assert live != [], "stuck after #{inspect(text)}"
              pool = Enum.filter(live, &(&1 in special))
              b = if pool != [] and :rand.uniform() < 0.7, do: Enum.random(pool), else: Enum.random(live)
              {:ok, g} = Grammar.advance(g, <<b>>)
              {:cont, {text <> <<b>>, g}}
          end
        end)

      if Grammar.complete?(g) do
        assert {:ok, v} = Vapor.JSON.decode(text), inspect(text)
        if check, do: assert(Vapor.Agent.Validate.check(check, v) == :ok, inspect(text))
      end
    end
  end

  test "string lengths are code points, in the grammar, the vocabulary's fast path and validation" do
    # "कि" is one grapheme but two code points; "é" (e + U+0301) likewise
    toks = {{"\"", "a", "कि", "e\u0301", "ab"}, MapSet.new()}
    v = Vocab.build(toks)
    schema = %{"type" => "string", "maxLength" => 1}
    {:ok, g} = JSONSchema.compile(schema)
    {:ok, g} = Grammar.advance(g, "\"")
    allowed = Vocab.allowed(v, g)
    assert MapSet.member?(allowed, 1) and MapSet.member?(allowed, 0)
    refute MapSet.member?(allowed, 2) or MapSet.member?(allowed, 3) or MapSet.member?(allowed, 4)
    assert Vapor.Agent.Validate.check(schema, "कि") =~ "at most 1"
  end

  test "tool dialects: detection from templates, grammars, parsing with content-derived ids" do
    dir = Path.expand("../fixtures/templates", __DIR__)
    assert Tools.dialect(File.read!(Path.join(dir, "Qwen-Qwen3-0.6B.jinja"))) == :hermes
    assert Tools.dialect(File.read!(Path.join(dir, "mistralai-Mistral-Nemo-Instruct-2407.jinja"))) == :mistral
    assert Tools.dialect(File.read!(Path.join(dir, "meta-llama-Llama-3.1-8B-Instruct.jinja"))) == :llama3
    assert Tools.dialect(nil) == :generic

    tools = [%{"type" => "function", "function" => %{"name" => "f", "parameters" => %{"type" => "object", "properties" => %{"x" => %{"type" => "integer"}}, "required" => ["x"]}}}]
    for {d, call} <- [{:hermes, "\n{\"name\": \"f\", \"arguments\": {\"x\": 1}}"}, {:mistral, "[{\"name\": \"f\", \"arguments\": {\"x\": 2}}]"},
                      {:llama3, "{\"name\": \"f\", \"parameters\": {\"x\": 3}}"}] do
      g = Tools.call_grammar(d, tools)
      assert accepts?(g, call), "#{d}"
      refute accepts?(g, String.replace(call, "\"f\"", "\"g\"")), "#{d}: unknown tool"
    end

    {content, [c1, c2]} = Tools.parse(:hermes, "Sure.<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\": 1}}\n</tool_call><tool_call>{\"name\":\"f\",\"arguments\":{\"x\":2}}</tool_call>", "salt")
    assert content == "Sure." and c1["arguments"] == %{"x" => 1} and c2["arguments"] == %{"x" => 2}
    assert c1["id"] != c2["id"]
    assert {_, [^c1, ^c2]} = Tools.parse(:hermes, "Sure.<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\": 1}}\n</tool_call><tool_call>{\"name\":\"f\",\"arguments\":{\"x\":2}}</tool_call>", "salt")
    assert {"", [%{"name" => "f", "arguments" => %{"x" => 3}}]} = Tools.parse(:llama3, "{\"name\": \"f\", \"parameters\": {\"x\": 3}}")
    assert {"", [%{"name" => "f"}]} = Tools.parse(:mistral, "[TOOL_CALLS][{\"name\": \"f\", \"arguments\": {}}]")
  end

  @tag :vocab
  test "masks over the Qwen2 vocabulary: every token of a valid document allowed, a wrong enum blocked at once, EOS only at the end" do
    {:ok, g} = Vapor.Ingest.GGUF.read(Path.expand("../fixtures/vocab/ggml-vocab-qwen2.gguf", __DIR__))
    {:ok, tk} = Vapor.Tokenizer.from_gguf(g.metadata)
    v = Vocab.build(tk)
    {:ok, gr} = JSONSchema.compile(%{"type" => "object", "properties" => %{"name" => %{"enum" => ["get_weather"]}, "q" => %{"type" => "string", "maxLength" => 40}}, "required" => ["name", "q"]})
    text = ~s({"name": "get_weather", "q": "São Paulo, \\"SP\\" — 3 dias"})
    ids = Vapor.Tokenizer.encode(tk, text, add_bos: false)

    c =
      Enum.reduce(ids, Constraint.new(gr, v, [tk.eos]), fn id, c ->
        {{:only, a}, c} = Constraint.allowed(c)
        assert MapSet.member?(a, id)
        refute MapSet.member?(a, tk.eos)
        {:ok, c} = Constraint.advance(c, id, Vapor.Tokenizer.surface(tk, id))
        c
      end)

    assert {{:only, only}, _} = Constraint.allowed(c)
    assert MapSet.equal?(only, MapSet.new([tk.eos]))

    [first | _] = bad = Vapor.Tokenizer.encode(tk, ~s({"name": "rm), add_bos: false)
    _ = first
    blocked = Enum.reduce_while(bad, Constraint.new(gr, v, [tk.eos]), fn id, c ->
      {{:only, a}, c} = Constraint.allowed(c)
      if MapSet.member?(a, id), do: ({:ok, c} = Constraint.advance(c, id, Vapor.Tokenizer.surface(tk, id)); {:cont, c}), else: {:halt, :blocked}
    end)

    assert blocked == :blocked

    # lazy: free until the opener, constrained after it
    {:ok, call} = JSONSchema.compile(%{"type" => "object", "properties" => %{"x" => %{"type" => "integer"}}, "required" => ["x"]})
    lazy = Constraint.new(call, v, [tk.eos], {:lazy, "<tool_call>"})
    assert {:all, _} = Constraint.allowed(lazy)
    {:ok, lazy} = Constraint.advance(lazy, 0, "thinking… <tool_call>")
    {{:only, a}, _} = Constraint.allowed(lazy)
    assert MapSet.size(a) < 1000
  end
end
