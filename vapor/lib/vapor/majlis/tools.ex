defmodule Vapor.Majlis.Tools do
  @moduledoc """
  The tools a chat thread may give its model: vapor's own, through the same
  declarations the MCP server publishes (`Vapor.MCP.Server.tools/0`), called
  in-process. A registry is `%{name => %{decl, effect, call}}`.

  The allowlist is the point. A tool that reads or writes the server's files
  (`context_search` over paths, `studio_run` and `render_scene`, which write
  outputs) is **not** offered to a conversation: text the model reads can
  steer it, and nothing a conversation says should reach the filesystem.
  What remains computes on its arguments only — Alembic in its sandbox (the
  code interpreter: fuel, a memory ceiling, a time limit, no I/O), the
  deciders (Rebis, Aludel, Tabula, logic, arbitrage), the solvers and the
  searches.

  Effect classes follow `Vapor.Agent.Spec`: `pure` tools are deterministic
  and re-executed when a run is replayed; tools whose answer may depend on
  wall-clock budgets are `observe` — recorded, read back on replay.
  """

  @pure ~w(alembic_eval athanor_verify rebis_check aludel_decide tabula_analyze amalgam_sum cupel_drill logic_check arbitrage_check workbench_solve studio_validate studio_catalogue)
  @observe ~w(athanor_run game_query crucible_run assay_run engineering_run finance_run board_query scene_ops)

  @doc "Every tool a thread may enable."
  def names, do: @pure ++ @observe

  @doc "vapor's own tools, called in-process through the MCP server's handler."
  def default do
    decls = Map.new(Vapor.MCP.Server.tools(), &{&1["name"], &1})
    state = Vapor.MCP.Server.new(dir: System.tmp_dir!())

    for name <- names(), Map.has_key?(decls, name), into: %{} do
      {name, %{decl: decls[name], effect: if(name in @pure, do: "pure", else: "observe"), call: fn args -> mcp_call(state, name, args) end}}
    end
  end

  defp mcp_call(state, name, args) do
    msg = %{"jsonrpc" => "2.0", "id" => 1, "method" => "tools/call", "params" => %{"name" => name, "arguments" => args}}

    case Vapor.MCP.Server.handle(msg, state) do
      {%{"result" => %{"isError" => true, "content" => [%{"text" => t} | _]}}, _} -> {:error, t}
      {%{"result" => %{"structuredContent" => data}}, _} when data != nil -> {:ok, data}
      {%{"result" => %{"content" => [%{"text" => t} | _]}}, _} -> {:ok, t}
      {%{"error" => %{"message" => m}}, _} -> {:error, m}
      other -> {:error, "unexpected answer from #{name}: #{inspect(other) |> String.slice(0, 200)}"}
    end
  end

  @doc "A registry from explicit entries (tests, embedders): `[{name, effect, description, schema, fun(args)}]`."
  def custom(entries) do
    for {name, effect, desc, schema, fun} <- entries, into: %{} do
      {name, %{decl: %{"name" => name, "description" => desc, "inputSchema" => schema}, effect: effect, call: fun}}
    end
  end

  @doc "The `Vapor.Agent.Spec` tool entries for the enabled names."
  def specs(registry, names) do
    Enum.map(names, fn n ->
      case registry[n] do
        nil -> raise ArgumentError, "no tool #{inspect(n)} (available: #{Enum.join(Enum.sort(Map.keys(registry)), ", ")})"
        e -> %{name: n, description: e.decl["description"], parameters: e.decl["inputSchema"], effect: e.effect}
      end
    end)
  end

  @doc "The implementations `Vapor.Agent.run/3` calls."
  def impls(registry, names) do
    for n <- names, e = registry[n], e != nil, into: %{} do
      {n, fn args, _ctx -> e.call.(args) end}
    end
  end
end
