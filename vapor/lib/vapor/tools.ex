defmodule Vapor.Tools do
  @moduledoc """
  Tool calling for local models: the dialect a model was trained to call
  tools in, a grammar that makes its calls well-formed by construction,
  and the parser that turns the generated text back into OpenAI-shaped
  `tool_calls`.

  The dialect is read from the model's own chat template (the authority on
  how the model writes calls), not configured by hand:

  | dialect | recognised by | a call looks like |
  |---|---|---|
  | `:hermes` (Qwen 2.5/3, Hermes, QwQ, many fine-tunes) | `<tool_call>` in the template | `<tool_call>\\n{"name": …, "arguments": {…}}\\n</tool_call>` |
  | `:mistral` | `[TOOL_CALLS]` | `[TOOL_CALLS][{"name": …, "arguments": {…}}]` |
  | `:llama3` (Llama 3.1–3.3) | `"parameters": dictionary` | `{"name": …, "parameters": {…}}` as the whole reply |
  | `:generic` | anything else, or no template | Hermes's form, taught by a system note |

  The constraint (`Vapor.Grammar.Constraint`) follows `tool_choice`:
  `"auto"` leaves the text free until the dialect's opener appears and then
  admits only a well-formed call to a declared tool with arguments valid for
  its schema (lazy mode); `"required"` or a named function prefills the
  opener into the prompt and admits only calls (strict mode). A call can
  therefore never name an unknown tool or carry arguments its schema
  refuses — the failure mode that makes small local models unusable as
  agents is removed by construction, not by retrying.
  """
  alias Vapor.Grammar
  alias Vapor.Grammar.JSONSchema

  @doc "The dialect a chat template speaks (`nil` template → `:generic`)."
  def dialect(nil), do: :generic

  def dialect(src) when is_binary(src) do
    cond do
      String.contains?(src, "<tool_call>") -> :hermes
      String.contains?(src, "[TOOL_CALLS]") -> :mistral
      String.contains?(src, "\"parameters\": dictionary") or String.contains?(src, "<|python_tag|>") -> :llama3
      true -> :generic
    end
  end

  @doc "The text that opens a call in a dialect."
  def opener(:hermes), do: "<tool_call>"
  def opener(:generic), do: "<tool_call>"
  def opener(:mistral), do: "[TOOL_CALLS]"
  def opener(:llama3), do: "{\"name\""

  @doc """
  The grammar of one call body (what follows the opener) for a list of
  OpenAI tool definitions, and whether `tool_choice` narrows it to one name.
  """
  def call_grammar(dialect, tools, only \\ nil) do
    tools = for t <- tools, f = fun(t), only == nil or f["name"] == only, do: f
    if tools == [], do: throw({:tools, "no tool matches tool_choice"})

    alts =
      for f <- tools do
        {:ok, args} = JSONSchema.compile(Map.get(f, "parameters", %{"type" => "object"}), lenient: true)
        key = if dialect == :llama3, do: "parameters", else: "arguments"

        {{:seq, [{:lit, "{"}, Grammar.ws(), {:lit, "\"name\""}, Grammar.ws(), {:lit, ":"}, Grammar.ws(),
                 {:lit, Vapor.JSON.encode(f["name"])}, Grammar.ws(), {:lit, ","}, Grammar.ws(),
                 {:lit, "\"#{key}\""}, Grammar.ws(), {:lit, ":"}, Grammar.ws(), args.root, Grammar.ws(), {:lit, "}"}]},
         args.defs}
      end

    # schema-local definitions of every tool, kept apart by tool name
    {bodies, defs} = Enum.unzip(alts)
    defs = Enum.reduce(defs, %{}, &Map.merge(&2, Map.drop(&1, Map.keys(Grammar.builtin_defs()))))
    body = {:alt, bodies}

    root =
      case dialect do
        d when d in [:hermes, :generic] -> {:seq, [Grammar.ws(), body]}
        :llama3 -> body
        :mistral -> {:seq, [{:lit, "["}, body, {:rep, {:seq, [{:lit, ","}, Grammar.ws(), body]}, 0, :inf}, {:lit, "]"}]}
      end

    Grammar.new(root, defs)
  catch
    {:tools, why} -> {:error, why}
  end

  # the function object of an OpenAI tool ({type: function, function: {…}}) or a bare one
  defp fun({:dict, _} = d), do: fun(plain(d))
  defp fun(%{"type" => "function", "function" => f}), do: plain(f)
  defp fun(%{"name" => _} = f), do: plain(f)
  defp fun(_), do: nil

  @doc "An ordered dict (`{:dict, pairs}`) as a plain map, recursively."
  def plain({:dict, pairs}), do: Map.new(pairs, fn {k, v} -> {k, plain(v)} end)
  def plain(l) when is_list(l), do: Enum.map(l, &plain/1)
  def plain(%{} = m), do: Map.new(m, fn {k, v} -> {k, plain(v)} end)
  def plain(v), do: v

  @doc """
  A system note teaching the generic dialect, for models without a
  template that knows tools.
  """
  def system_note(tools) do
    defs = Enum.map_join(tools, "\n", &Vapor.JSON.encode(plain(&1)))

    "# Tools\n\nYou may call one or more functions. Their signatures are in <tools></tools>:\n<tools>\n" <>
      defs <> "\n</tools>\n\nTo call a function, reply with a JSON object within <tool_call></tool_call> tags:\n" <>
      "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
  end

  @doc """
  Split generated text into `{content, calls}`: the text before the first
  call, and every call as `%{"name" => n, "arguments" => map}` in order.
  Call ids are derived from the content they identify (`id_salt` plus the
  call's index and canonical bytes), so the same generation always yields
  the same ids.
  """
  def parse(dialect, text, id_salt \\ "") do
    {content, bodies} =
      case dialect do
        d when d in [:hermes, :generic] ->
          case String.split(text, "<tool_call>") do
            [only] -> {only, []}
            [c | calls] -> {c, Enum.map(calls, &(&1 |> String.split("</tool_call>") |> hd()))}
          end

        :mistral ->
          case String.split(text, "[TOOL_CALLS]", parts: 2) do
            [only] -> {only, []}
            [c, json] -> {c, json |> decode_list()}
          end

        :llama3 ->
          t = String.trim_leading(text)
          if String.starts_with?(t, "{\"name\""), do: {"", [t]}, else: {text, []}
      end

    calls =
      bodies
      |> Enum.flat_map(fn
        %{} = m -> [m]
        b when is_binary(b) -> case Vapor.JSON.decode(String.trim(b)) do
            {:ok, %{} = m} -> [m]
            _ -> []
          end
      end)
      |> Enum.filter(&is_binary(&1["name"]))
      |> Enum.with_index()
      |> Enum.map(fn {m, i} ->
        args = m["arguments"] || m["parameters"] || %{}
        id = "call_" <> String.slice(Vapor.Canonical.hex_digest({id_salt, i, m["name"], args}), 0, 24)
        %{"id" => id, "name" => m["name"], "arguments" => args}
      end)

    {String.trim_trailing(content), calls}
  end

  defp decode_list(json) do
    case Vapor.JSON.decode(String.trim(json)) do
      {:ok, l} when is_list(l) -> l
      _ -> []
    end
  end

  @doc "Calls in the OpenAI response shape (`arguments` as a JSON string)."
  def openai(calls) do
    for c <- calls, do: %{id: c["id"], type: "function", function: %{name: c["name"], arguments: Vapor.JSON.encode(c["arguments"])}}
  end
end
