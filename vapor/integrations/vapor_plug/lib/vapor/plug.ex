defmodule Vapor.Plug do
  @moduledoc """
  vapor's OpenAI-compatible API as a `Plug`: mount it in a Phoenix router
  (or any Plug pipeline, under Bandit or Cowboy) and the same handlers as
  `Vapor.Serve` answer — tools, structured outputs, embeddings, streaming,
  `x-vapor-receipt` — over your endpoint's TLS, HTTP/2, auth and telemetry.

      # application.ex — the engine, then the serving context (built once:
      # the vocabulary index for constrained decoding is not rebuilt per request)
      children = [
        {Vapor.Engine, name: MyApp.Engine, model: "/models/qwen3-0.6b", tokenizer: tk},
        {Vapor.Plug, name: MyApp.LLM, engine: MyApp.Engine, tokenizer: tk, template: tpl, model_name: "qwen3"},
        MyAppWeb.Endpoint
      ]

      # router.ex — behind whatever authenticates your API
      scope "/llm" do
        pipe_through :api_auth
        forward "/", Vapor.Plug, name: MyApp.LLM
      end

  Paths are relative to the mount point (`/llm/v1/chat/completions`).

  **Mount it before `Plug.Parsers`** (or outside the `:api` pipeline): it
  reads the raw body, because JSON-schema property order is part of the
  request (`response_format`, tool parameters) and a parsed map has lost it.
  If a parser already consumed the body, the parsed params are re-encoded —
  everything works, but object-key order in schemas becomes alphabetical.

  A client that disconnects mid-stream cancels its request in the engine
  (`Vapor.Engine.cancel/2`): the batch slot is freed at the next step.
  """
  @behaviour Plug
  import Plug.Conn

  @max_body 16 * 1024 * 1024

  @doc false
  def child_spec(opts), do: %{id: {__MODULE__, Keyword.fetch!(opts, :name)}, start: {__MODULE__, :start_link, [opts]}}

  @doc """
  Build the serving context (`Vapor.Serve.context/1`: `engine:`,
  `tokenizer:`, `template:`, `embedder:`, `model_name:`, `model_digest:`) and
  publish it under `name:`. The engine should be a registered name, so a
  restarted engine is found again. Returns `:ignore` — there is no process.
  """
  def start_link(opts) do
    :persistent_term.put({__MODULE__, Keyword.fetch!(opts, :name)}, Vapor.Serve.context(opts))
    :ignore
  end

  @impl Plug
  def init(opts) do
    case Keyword.fetch(opts, :context) do
      {:ok, ctx} -> {:context, ctx}
      :error -> {:name, Keyword.fetch!(opts, :name)}
    end
  end

  @impl Plug
  def call(conn, where) do
    ctx = context(where)

    case body(conn) do
      {:ok, body, conn} ->
        key = make_ref()
        Process.put({__MODULE__, key}, conn)

        request = %{method: method(conn.method), path: "/" <> Enum.join(conn.path_info, "/"), body: body,
                    headers: Map.new(conn.req_headers)}

        Vapor.Serve.dispatch({__MODULE__.Responder, key}, request, ctx)
        conn = Process.delete({__MODULE__, key})
        halt(conn)

      {:error, status, msg, conn} ->
        conn
        |> put_resp_content_type("application/json")
        |> send_resp(status, Vapor.JSON.encode(%{error: %{message: msg, type: "invalid_request_error"}}))
        |> halt()
    end
  end

  defp context({:context, ctx}), do: ctx
  defp context({:name, name}), do: :persistent_term.get({__MODULE__, name})

  # only the two methods the API has become atoms
  defp method("GET"), do: :GET
  defp method("POST"), do: :POST
  defp method(other), do: other

  defp body(%{body_params: %Plug.Conn.Unfetched{}} = conn), do: read_all(conn, [])
  defp body(%{body_params: params} = conn) when map_size(params) > 0, do: {:ok, Vapor.JSON.encode(params), conn}
  defp body(conn), do: read_all(conn, [])

  defp read_all(conn, acc) do
    case read_body(conn, length: 1_000_000) do
      {:ok, data, conn} -> {:ok, IO.iodata_to_binary([acc, data]), conn}
      {:more, data, conn} ->
        if IO.iodata_length([acc, data]) > @max_body,
          do: {:error, 413, "request body too large", conn},
          else: read_all(conn, [acc, data])
      {:error, why} -> {:error, 400, "could not read the body: #{inspect(why)}", conn}
    end
  end
end

defmodule Vapor.Plug.Responder do
  @moduledoc false
  # Vapor.Serve's responder over a Plug.Conn. Conns are values, and the
  # handlers do not thread them: the current one lives in the process
  # dictionary of the request process, under a key unique to the request.
  import Plug.Conn

  defp get(key), do: Process.get({Vapor.Plug, key})
  defp put(key, conn), do: Process.put({Vapor.Plug, key}, conn)

  def json(key, status, body, headers) do
    conn = get(key) |> put_resp_content_type("application/json") |> merge_resp_headers(headers) |> send_resp(status, body)
    put(key, conn)
    true
  end

  # any content type (the console page, thumbnails)
  def send_body(key, status, ctype, body, headers) do
    conn = get(key) |> put_resp_header("content-type", ctype) |> merge_resp_headers(headers) |> send_resp(status, body)
    put(key, conn)
    true
  end

  def stream_start(key, headers) do
    conn =
      get(key)
      |> put_resp_content_type("text/event-stream")
      |> put_resp_header("cache-control", "no-cache")
      |> merge_resp_headers(headers)
      |> send_chunked(200)

    put(key, conn)
    :ok
  end

  def chunk(key, data) do
    case Plug.Conn.chunk(get(key), data) do
      {:ok, conn} -> put(key, conn); :ok
      {:error, why} -> {:error, why}
    end
  end

  def stream_end(_key), do: true
end
