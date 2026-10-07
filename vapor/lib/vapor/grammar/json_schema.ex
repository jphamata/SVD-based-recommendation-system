defmodule Vapor.Grammar.JSONSchema do
  @moduledoc """
  JSON Schema → `Vapor.Grammar`. The subset structured outputs and tool
  calls use in practice:

    * `type` (a name or a list), `enum`, `const`, `anyOf`/`oneOf`,
      `$ref` to `#/$defs/…` or `#/definitions/…` (recursion allowed);
    * objects: `properties`, `required`, `additionalProperties` (false, or
      a schema when there are no `properties`); members are emitted in a
      fixed order — the required ones as `required` lists them, then the
      optional ones by name — and optional ones may be left out;
    * arrays: `items`, `minItems`, `maxItems`;
    * strings: `minLength`, `maxLength`, `enum`, **`pattern`** (ECMA-262,
      compiled to bytes by `Vapor.Grammar.Regex`) and **`format`** (`date`,
      `time`, `date-time`, `uuid`, `ipv4`, `email`, `hostname`); numbers
      and integers; booleans; null; `{}` or `true` for any value. A
      `pattern` together with a length bound or a `format` is refused (the
      grammar does not intersect languages) — `lenient: true` keeps the
      pattern and drops the rest.

  Keywords that constrain values in ways a grammar here does not express
  (`format` on numbers, `minimum`, `maximum`, `multipleOf`, `not`,
  `if`/`then`/`else`, `patternProperties`, `uniqueItems`, …) are a
  rejection naming the keyword — unless `lenient: true`, which drops them
  (the output is then valid for the schema without them). Annotations
  (`title`, `description`, `default`, `examples`, `$schema`, `$id`) are ignored.
  """
  alias Vapor.{Grammar, Rejection}

  @unsupported ~w(minimum maximum exclusiveMinimum exclusiveMaximum multipleOf not if then else
                  patternProperties uniqueItems contains minContains maxContains propertyNames dependentRequired
                  dependentSchemas minProperties maxProperties prefixItems unevaluatedProperties unevaluatedItems allOf)

  @doc "Compile a schema (a decoded JSON map) to `{:ok, %Vapor.Grammar{}}`."
  @spec compile(map | boolean, keyword) :: {:ok, Grammar.t()} | {:error, Rejection.t()}
  def compile(schema, opts \\ []) do
    defs = Map.merge(schema_map(schema, "$defs"), schema_map(schema, "definitions"))
    st = %{lenient: Keyword.get(opts, :lenient, false), defs: defs}

    with {:ok, root} <- node(schema, st),
         {:ok, rules} <- rules(defs, st) do
      {:ok, Grammar.new(root, rules)}
    end
  catch
    {:reject, field, why} -> {:error, Rejection.new({:json_schema, field}, why, "simplify the schema or pass lenient: true")}
  end

  defp schema_map(%{} = s, key), do: Map.get(s, key, %{})
  defp schema_map(_, _), do: %{}

  defp rules(defs, st) do
    Enum.reduce_while(defs, {:ok, %{}}, fn {name, s}, {:ok, acc} ->
      case node(s, st) do
        {:ok, g} -> {:cont, {:ok, Map.put(acc, {:def, name}, g)}}
        e -> {:halt, e}
      end
    end)
  end

  defp node(true, _st), do: {:ok, {:ref, :value}}
  defp node(false, _st), do: throw({:reject, "false", "a schema that admits a value"})

  defp node(%{} = s, st) do
    for k <- Map.keys(s), k in @unsupported, not st.lenient, do: throw({:reject, k, "a keyword the grammar can enforce (#{k} is not)"})

    # pattern and format constrain strings: a schema with one and no type is a string schema
    s = if (Map.has_key?(s, "pattern") or Map.has_key?(s, "format")) and s["type"] == nil and not Map.has_key?(s, "properties"), do: Map.put(s, "type", "string"), else: s
    t = s["type"]

    if Map.has_key?(s, "format") and t not in ["string", nil] and not (is_list(t) and "string" in t) and not st.lenient,
      do: throw({:reject, "format", "a format on strings (got #{inspect(t)})"})

    cond do
      ref = s["$ref"] -> {:ok, {:ref, {:def, ref_name(ref, st)}}}
      Map.has_key?(s, "const") -> {:ok, {:lit, Vapor.JSON.encode(s["const"])}}
      enum = s["enum"] -> {:ok, {:alt, Enum.map(enum, &{:lit, Vapor.JSON.encode(&1)})}}
      any = s["anyOf"] || s["oneOf"] -> alts(any, st)
      is_list(s["type"]) -> alts(Enum.map(s["type"], &Map.put(s, "type", &1)), st)
      true -> typed(s["type"], s, st)
    end
  end

  defp node(other, _st), do: throw({:reject, inspect(other), "a schema object"})

  defp ref_name("#/$defs/" <> name, st), do: known(name, st)
  defp ref_name("#/definitions/" <> name, st), do: known(name, st)
  defp ref_name(ref, _st), do: throw({:reject, "$ref", "a local reference #/$defs/… (got #{ref})"})

  defp known(name, st), do: if(Map.has_key?(st.defs, name), do: name, else: throw({:reject, "$ref", "a defined name (#{name})"}))

  defp alts(list, st) do
    {:ok, {:alt, Enum.map(list, fn s -> {:ok, g} = node(s, st); g end)}}
  end

  defp typed("object", s, st) do
    props = s["properties"] || %{}
    req = MapSet.new(s["required"] || [])
    addl = Map.get(s, "additionalProperties", if(props == %{}, do: true, else: false))

    cond do
      props == %{} and addl in [true, nil] -> {:ok, {:ref, :object}}
      props == %{} and is_map(addl) -> {:ok, map_of(node!(addl, st))}
      props == %{} -> {:ok, {:seq, [{:lit, "{"}, Grammar.ws(), {:lit, "}"}]}}
      true ->
        order = ordered_keys(s)
        members = for k <- order, do: {k, node!(props[k], st), MapSet.member?(req, k)}
        {:ok, {:seq, [{:lit, "{"}, Grammar.ws(), body(members, true), Grammar.ws(), {:lit, "}"}]}}
    end
  end

  defp typed("array", s, st) do
    item = node!(Map.get(s, "items", true), st)
    min = s["minItems"] || 0
    max = s["maxItems"] || :inf
    sep = {:seq, [Grammar.ws(), {:lit, ","}, Grammar.ws(), item]}

    elems =
      cond do
        max == 0 -> {:lit, ""}
        min == 0 -> {:alt, [{:lit, ""}, {:seq, [item, {:rep, sep, 0, dec(max)}]}]}
        true -> {:seq, [item, {:rep, sep, min - 1, dec(max)}]}
      end

    {:ok, {:seq, [{:lit, "["}, Grammar.ws(), elems, Grammar.ws(), {:lit, "]"}]}}
  end

  defp typed("string", s, st) do
    pattern = s["pattern"]
    format = s["format"]
    lengths? = Map.has_key?(s, "minLength") or Map.has_key?(s, "maxLength")

    cond do
      pattern == nil and format == nil ->
        {:ok, {:str, %{min: s["minLength"] || 0, max: s["maxLength"]}}}

      not st.lenient and pattern != nil and format != nil ->
        throw({:reject, "format", "pattern or format, not both (the grammar does not intersect them)"})

      not st.lenient and lengths? ->
        throw({:reject, if(pattern, do: "pattern", else: "format"), "a length bound written into the pattern ({m,n}), not beside it"})

      true ->
        compiled = if pattern, do: Vapor.Grammar.Regex.json_string(pattern), else: Vapor.Grammar.Regex.json_format(format)

        case compiled do
          {:ok, g} -> {:ok, g}
          {:error, _} when st.lenient -> {:ok, {:str, %{min: s["minLength"] || 0, max: s["maxLength"]}}}
          {:error, %Rejection{bound: b}} -> throw({:reject, if(pattern, do: "pattern", else: "format"), b})
        end
    end
  end
  defp typed("number", _s, _st), do: {:ok, {:num, :number}}
  defp typed("integer", _s, _st), do: {:ok, {:num, :integer}}
  defp typed("boolean", _s, _st), do: {:ok, {:alt, [{:lit, "true"}, {:lit, "false"}]}}
  defp typed("null", _s, _st), do: {:ok, {:lit, "null"}}
  defp typed(nil, s, st), do: if(Map.has_key?(s, "properties"), do: typed("object", s, st), else: {:ok, {:ref, :value}})
  defp typed(t, _s, _st), do: throw({:reject, "type", "a JSON type name (got #{inspect(t)})"})

  defp node!(s, st), do: (fn {:ok, g} -> g end).(node(s, st))
  defp dec(:inf), do: :inf
  defp dec(n), do: n - 1

  # members in declared order: after the first present member each one is
  # preceded by a comma; optional members may be skipped
  defp body([], _first?), do: {:lit, ""}

  defp body([{k, g, required?} | rest], first?) do
    sep = if first?, do: {:lit, ""}, else: {:seq, [Grammar.ws(), {:lit, ","}, Grammar.ws()]}
    present = {:seq, [sep, {:lit, Vapor.JSON.encode(k)}, Grammar.ws(), {:lit, ":"}, Grammar.ws(), g, body(rest, false)]}
    if required?, do: present, else: {:alt, [present, body(rest, first?)]}
  end

  defp map_of(v) do
    kv = {:seq, [{:str, %{min: 0, max: nil}}, Grammar.ws(), {:lit, ":"}, Grammar.ws(), v]}

    {:seq, [{:lit, "{"}, Grammar.ws(),
            {:alt, [{:lit, "}"}, {:seq, [kv, {:rep, {:seq, [Grammar.ws(), {:lit, ","}, Grammar.ws(), kv]}, 0, :inf}, Grammar.ws(), {:lit, "}"}]}]}]}
  end

  # A decoded JSON object has no member order, so the order is fixed by the
  # schema's own data: required members in the order `required` lists them
  # (which is how schemas for tools and structured outputs are written),
  # then the optional ones by name.
  defp ordered_keys(%{"properties" => props} = s) do
    req = Enum.filter(s["required"] || [], &Map.has_key?(props, &1))
    req ++ Enum.sort(Map.keys(props) -- req)
  end
end
