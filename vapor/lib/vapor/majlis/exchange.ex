defmodule Vapor.Majlis.Exchange do
  @moduledoc """
  Conversations in and out.

    * **vapor** (`"format": "vapor-majlis/1"`): the thread's whole tree —
      every branch — with each message's id. On import every id is
      recomputed from the message (whose hash covers its parent), so an
      export altered in any byte is refused, and importing it twice shares
      the messages instead of copying them.
    * **Markdown**: the current path, for reading and pasting.
    * **ChatGPT** (`conversations.json`): a list of conversations whose
      `mapping` is a tree of `{message, parent, children}` with a
      `current_node` — edited prompts and regenerated answers are branches
      there too, and they arrive as branches.
    * **Claude** (`conversations.json`): a list of conversations with
      `chat_messages` (`sender` human/assistant, `text` or `content` parts),
      linked by `parent_message_uuid` when present, else in order.

  Imported text keeps its timestamps; non-text parts (images, files, tool
  payloads of other products) are noted as `[non-text part: kind]`, never
  silently dropped.
  """

  @max 200 * 1024 * 1024

  # ---------------------------------------------------------------- export

  @doc "Export one thread: `:json` (vapor) or `:markdown`."
  def export(:json, tid, t, _path, tree) do
    Vapor.JSON.encode(%{
      "format" => "vapor-majlis/1",
      "thread" => %{"id" => tid, "title" => t["title"], "system" => t["system"], "head" => t["head"], "pins" => t["pins"],
                    "summary" => t["summary"], "created" => t["created"], "updated" => t["updated"]},
      "nodes" => tree
    })
  end

  def export(:markdown, _tid, t, path, _tree) do
    head = "# #{t["title"]}\n\n" <> if(t["system"] not in [nil, ""], do: "> **system** — #{String.replace(t["system"], "\n", "\n> ")}\n\n", else: "")

    body =
      Enum.map_join(path.messages, "\n\n", fn m ->
        "### #{m.role} · #{m.t} · `#{String.slice(m.id, 0, 12)}`" <>
          (if length(m.siblings) > 1, do: " (version #{m.index} of #{length(m.siblings)})", else: "") <> "\n\n" <> m.content
      end)

    head <> body <> "\n"
  end

  def export(other, _, _, _, _), do: raise(ArgumentError, "export format: json or markdown, not #{inspect(other)}")

  # ---------------------------------------------------------------- import

  @doc """
  Parse an export into conversations: `{:ok, [%{title, head, nodes, source,
  created}]}`, nodes as `%{"key", "parent_key", "role", "content", "t",
  "meta"}` in parent-before-child order.
  """
  def parse(data, format \\ :auto)

  def parse(data, _format) when is_binary(data) and byte_size(data) > @max, do: {:error, "the export is larger than #{div(@max, 1024 * 1024)} MB"}

  def parse(data, format) when is_binary(data) do
    case Vapor.JSON.decode(data) do
      {:ok, v} -> parse(v, format)
      _ -> {:error, "not JSON"}
    end
  end

  def parse(v, :auto) do
    cond do
      is_map(v) and v["format"] == "vapor-majlis/1" -> parse(v, :vapor)
      is_list(v) and v != [] and is_map(hd(v)) and Map.has_key?(hd(v), "mapping") -> parse(v, :chatgpt)
      is_list(v) and v != [] and is_map(hd(v)) and Map.has_key?(hd(v), "chat_messages") -> parse(v, :claude)
      is_map(v) and Map.has_key?(v, "mapping") -> parse([v], :chatgpt)
      true -> {:error, "unrecognised export (vapor-majlis/1, ChatGPT or Claude conversations.json)"}
    end
  end

  def parse(%{"nodes" => nodes, "thread" => th}, :vapor) when is_list(nodes) do
    ids = MapSet.new(nodes, & &1["id"])
    ordered = topo(nodes, fn n -> n["id"] end, fn n -> n["parent"] end)

    entries =
      for n <- ordered do
        if n["parent"] != nil and not MapSet.member?(ids, n["parent"]) and n["parent"] != nil,
          do: throw({:bad, "message #{String.slice(n["id"], 0, 12)} names a parent outside the export"})

        %{"key" => n["id"], "parent_key" => n["parent"], "role" => n["role"], "content" => n["content"], "t" => n["t"], "meta" => n["meta"],
          "verify" => n["id"], "verify_node" => Map.take(n, ["k", "role", "content", "parent", "t", "meta", "o"])}
      end

    {:ok, [%{title: th["title"] || "imported", head: th["head"], nodes: entries, source: "vapor", created: th["created"], system: th["system"]}]}
  catch
    {:bad, why} -> {:error, why}
  end

  def parse(convs, :chatgpt) when is_list(convs) do
    {:ok, Enum.map(convs, &chatgpt/1)}
  rescue
    e -> {:error, "ChatGPT export: #{Exception.message(e)}"}
  end

  def parse(convs, :claude) when is_list(convs) do
    {:ok, Enum.map(convs, &claude/1)}
  rescue
    e -> {:error, "Claude export: #{Exception.message(e)}"}
  end

  def parse(_, f), do: {:error, "cannot read this as #{inspect(f)}"}

  # ChatGPT: a tree in `mapping`; empty and hidden nodes are skipped, their
  # children re-attached to the nearest kept ancestor
  defp chatgpt(c) do
    mapping = c["mapping"] || %{}
    order = topo(Map.values(mapping), & &1["id"], & &1["parent"])

    {entries, keep} =
      Enum.reduce(order, {[], %{}}, fn n, {acc, keep} ->
        parent_kept = keep[n["parent"]]

        case chatgpt_message(n["message"]) do
          nil ->
            {acc, Map.put(keep, n["id"], parent_kept)}

          {role, text, t} ->
            e = %{"key" => n["id"], "parent_key" => parent_kept, "role" => role, "content" => text, "t" => t, "meta" => %{"source" => "chatgpt"}}
            {acc ++ [e], Map.put(keep, n["id"], n["id"])}
        end
      end)

    head = keep[c["current_node"]] || (List.last(entries) || %{})["key"]
    %{title: c["title"] || "ChatGPT conversation", head: head, nodes: entries, source: "chatgpt", created: unix(c["create_time"])}
  end

  defp chatgpt_message(nil), do: nil

  defp chatgpt_message(m) do
    role = get_in(m, ["author", "role"])
    parts = get_in(m, ["content", "parts"]) || []
    text = Enum.map_join(parts, "\n", fn p when is_binary(p) -> p; p when is_map(p) -> "[non-text part: #{p["content_type"] || "object"}]"; _ -> "" end) |> String.trim()
    hidden = get_in(m, ["metadata", "is_visually_hidden_from_conversation"]) == true

    cond do
      hidden or text == "" -> nil
      role in ["user", "assistant", "system", "tool"] -> {role, text, unix(m["create_time"])}
      true -> nil
    end
  end

  defp claude(c) do
    msgs = c["chat_messages"] || []
    linked? = Enum.any?(msgs, &Map.has_key?(&1, "parent_message_uuid"))

    {entries, _} =
      Enum.reduce(msgs, {[], nil}, fn m, {acc, prev} ->
        text =
          case m["content"] do
            parts when is_list(parts) and parts != [] ->
              Enum.map_join(parts, "\n", fn %{"type" => "text", "text" => t} -> t; p -> "[non-text part: #{p["type"] || "object"}]" end)
            _ -> m["text"] || ""
          end

        role = if m["sender"] == "human", do: "user", else: "assistant"
        parent = if linked?, do: blank_to_nil(m["parent_message_uuid"]), else: prev
        e = %{"key" => m["uuid"], "parent_key" => parent, "role" => role, "content" => String.trim(text), "t" => m["created_at"], "meta" => %{"source" => "claude"}}
        {acc ++ [e], m["uuid"]}
      end)

    entries = if linked?, do: topo(entries, & &1["key"], & &1["parent_key"]), else: entries
    keys = MapSet.new(entries, & &1["key"])
    entries = Enum.map(entries, fn e -> if MapSet.member?(keys, e["parent_key"]), do: e, else: %{e | "parent_key" => nil} end)
    %{title: c["name"] || "Claude conversation", head: (List.last(entries) || %{})["key"], nodes: entries, source: "claude", created: c["created_at"]}
  end

  defp blank_to_nil(v) when v in [nil, "", "00000000-0000-4000-8000-000000000000"], do: nil
  defp blank_to_nil(v), do: v

  defp unix(nil), do: nil
  defp unix(t) when is_number(t), do: t |> Kernel.*(1000) |> round() |> DateTime.from_unix!(:millisecond) |> DateTime.to_iso8601()
  defp unix(t) when is_binary(t), do: t

  # parents before children; nodes whose parent is absent are roots
  defp topo(nodes, id, parent) do
    by_id = Map.new(nodes, &{id.(&1), &1})
    kids = Enum.group_by(nodes, &(if Map.has_key?(by_id, parent.(&1)), do: parent.(&1), else: :root))
    walk(Map.get(kids, :root, []), kids, id, MapSet.new())
  end

  defp walk(level, kids, id, seen) do
    Enum.flat_map(level, fn n ->
      k = id.(n)
      if MapSet.member?(seen, k), do: raise(ArgumentError, "the conversation has a cycle at #{k}"), else: [n | walk(Map.get(kids, k, []), kids, id, MapSet.put(seen, k))]
    end)
  end
end
