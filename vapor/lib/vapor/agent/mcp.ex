defmodule Vapor.Agent.MCP do
  @moduledoc """
  A Model Context Protocol client over stdio (JSON-RPC 2.0, one message per
  line), with OTP alone: `initialize`, `tools/list`, `tools/call`.

  An MCP server's tools become agent tools (`agent_tools/2`). MCP does not
  say whether a tool reads or changes the world, so the effect class is the
  operator's declaration, and an undeclared tool is `"act"` — the most
  restrictive class: it runs only if the spec grants it, and replay never
  re-executes it. The server process is isolated like vapor's workers: if it
  dies, calls fail with an error the agent records; the BEAM is untouched.
  """
  use GenServer

  @protocol "2025-06-18"

  @doc "Start a server process (`cmd: [executable | args]`) and initialise the session."
  def start_link(opts), do: GenServer.start_link(__MODULE__, opts)

  @doc "The server's tools: `[%{\"name\", \"description\", \"inputSchema\"}]`."
  def tools(pid), do: GenServer.call(pid, :tools, 60_000)

  @doc "Call a tool: `{:ok, value}` (structured content, or the text) or `{:error, message}`."
  def call(pid, name, args), do: GenServer.call(pid, {:call, name, args}, 300_000)

  @doc """
  Agent tool entries and implementations for the server's tools. Options:
  `effects: %{name => "pure" | "observe" | "act"}` (default `"act"`),
  `version:` (recorded in each entry; default the server's name and
  version).
  """
  def agent_tools(pid, opts \\ []) do
    effects = Keyword.get(opts, :effects, %{})
    info = GenServer.call(pid, :info)
    version = Keyword.get(opts, :version, "mcp:#{info["name"]}@#{info["version"]}")

    {:ok, list} = tools(pid)

    entries =
      for t <- list do
        %{"name" => t["name"], "description" => t["description"] || "", "parameters" => t["inputSchema"] || %{"type" => "object"},
          "effect" => Map.get(effects, t["name"], "act"), "version" => version}
      end

    impls = Map.new(list, fn t -> {t["name"], fn args, _ctx -> call(pid, t["name"], args) end} end)
    {entries, impls}
  end

  # ------------------------------------------------------------- server --

  @impl true
  def init(opts) do
    # the server's death is an answer ("the tool is gone"), not a signal that
    # takes this process and its linked caller down
    Process.flag(:trap_exit, true)
    [exe | args] = Keyword.fetch!(opts, :cmd)
    path = System.find_executable(exe) || exe
    port = Port.open({:spawn_executable, path}, [:binary, :exit_status, {:line, 4_000_000}, args: args])
    st = %{port: port, next: 1, waiting: %{}, buf: "", info: %{}, gone: nil}

    {:ok, st, {:continue, :initialize}}
  end

  @impl true
  def handle_continue(:initialize, st) do
    {reply, st} = request_sync(st, "initialize", %{"protocolVersion" => @protocol, "capabilities" => %{},
                                                   "clientInfo" => %{"name" => "vapor", "version" => "0.3.0"}})

    case reply do
      {:ok, res} ->
        send_msg(st.port, %{"jsonrpc" => "2.0", "method" => "notifications/initialized"})
        {:noreply, %{st | info: res["serverInfo"] || %{}}}

      {:error, why} ->
        {:stop, {:mcp_initialize, why}, st}
    end
  end

  @impl true
  def handle_call(:info, _from, st), do: {:reply, st.info, st}

  def handle_call(_req, _from, %{gone: why} = st) when why != nil, do: {:reply, {:error, "MCP server exited (#{why})"}, st}

  def handle_call(:tools, from, st), do: {:noreply, request(st, "tools/list", %{}, from, &{:ok, &1["tools"] || []})}

  def handle_call({:call, name, args}, from, st) do
    {:noreply, request(st, "tools/call", %{"name" => name, "arguments" => args}, from, &tool_result/1)}
  end

  defp tool_result(%{"isError" => true} = r), do: {:error, text(r)}
  defp tool_result(%{"structuredContent" => %{"result" => v}}), do: {:ok, v}
  defp tool_result(%{"structuredContent" => v}) when v != nil, do: {:ok, v}
  defp tool_result(r), do: {:ok, text(r)}

  defp text(r), do: (r["content"] || []) |> Enum.filter(&(&1["type"] == "text")) |> Enum.map_join("\n", & &1["text"])

  @impl true
  def handle_info({port, {:data, {:noeol, part}}}, %{port: port} = st), do: {:noreply, %{st | buf: st.buf <> part}}

  def handle_info({port, {:data, {:eol, line}}}, %{port: port} = st) do
    full = st.buf <> line
    {:noreply, dispatch(Vapor.JSON.decode(full), %{st | buf: ""})}
  end

  def handle_info({port, {:exit_status, code}}, %{port: port} = st), do: {:noreply, gone(st, code)}
  def handle_info({:EXIT, port, why}, %{port: port} = st), do: {:noreply, gone(st, inspect(why))}
  def handle_info(_other, st), do: {:noreply, st}

  # pending and future calls fail with the reason; the process stays
  defp gone(st, why) do
    for {_id, {from, _}} <- st.waiting, do: GenServer.reply(from, {:error, "MCP server exited (#{why})"})
    %{st | waiting: %{}, gone: why}
  end

  defp dispatch({:ok, %{"id" => id} = msg}, st) when is_map_key(st.waiting, id) do
    {{from, k}, waiting} = Map.pop(st.waiting, id)
    reply = if msg["error"], do: {:error, msg["error"]["message"] || inspect(msg["error"])}, else: k.(msg["result"] || %{})
    GenServer.reply(from, reply)
    %{st | waiting: waiting}
  end

  # notifications and server→client requests are not needed by a tool client
  defp dispatch(_other, st), do: st

  defp request(st, method, params, from, k) do
    send_msg(st.port, %{"jsonrpc" => "2.0", "id" => st.next, "method" => method, "params" => params})
    %{st | next: st.next + 1, waiting: Map.put(st.waiting, st.next, {from, k})}
  end

  # the handshake, before the process serves calls
  defp request_sync(st, method, params) do
    id = st.next
    send_msg(st.port, %{"jsonrpc" => "2.0", "id" => id, "method" => method, "params" => params})
    {await(st.port, id, ""), %{st | next: id + 1}}
  end

  defp await(port, id, buf) do
    receive do
      {^port, {:data, {:noeol, part}}} -> await(port, id, buf <> part)
      {^port, {:data, {:eol, line}}} ->
        case Vapor.JSON.decode(buf <> line) do
          {:ok, %{"id" => ^id, "result" => r}} -> {:ok, r}
          {:ok, %{"id" => ^id, "error" => e}} -> {:error, e}
          _ -> await(port, id, "")
        end
      {^port, {:exit_status, code}} -> {:error, {:exit, code}}
    after
      30_000 -> {:error, :timeout}
    end
  end

  defp send_msg(port, msg) do
    Port.command(port, [Vapor.JSON.encode(msg), "\n"])
  rescue
    ArgumentError -> send(self(), {:EXIT, port, :closed})
  end
end
