defmodule VaporPlug.Router do
  @moduledoc false
  # how an application mounts it: under a prefix, behind its own plugs
  use Plug.Router
  plug :match
  plug :dispatch
  forward "/llm", to: Vapor.Plug, init_opts: [name: VaporPlug.LLM]
  match _, do: send_resp(conn, 404, "app")
end

defmodule VaporPlug.Parsed do
  @moduledoc false
  # a pipeline that parsed the body first (Phoenix's :api pipeline does)
  use Plug.Builder
  plug Plug.Parsers, parsers: [:json], json_decoder: {Vapor.JSON, :decode!, []}, pass: ["*/*"]
  plug Vapor.Plug, name: VaporPlug.LLM
end

defmodule VaporPlugTest do
  @moduledoc """
  `Vapor.Plug` answers exactly as `Vapor.Serve` does — same bodies, same
  receipts — through Plug.Test, behind a Plug.Router prefix, after a body
  parser, and over a real HTTP server (Bandit), including streaming, tool
  calls constrained to their schemas, and a client hanging up mid-stream.
  """
  use ExUnit.Case, async: false
  import Plug.Test
  import Plug.Conn
  import Vapor.TestHelpers
  alias Vapor.{Engine, JSON, Serve}
  alias Vapor.Model.Config

  @moduletag timeout: 600_000

  setup_all do
    tk = byte_bpe_tokenizer()
    {:ok, c} = Config.from_map(tiny_config("qwen2", %{"vocab_size" => 259, "max_position_embeddings" => 1024}))
    {:ok, _} = Engine.start_link(config: c, weights: tiny_weights(c, 7), tokenizer: tk, max_seq: 1024, page: 16, sequences: 4,
                                 step_tokens: 32, name: VaporPlug.Engine)

    opts = [name: VaporPlug.LLM, engine: VaporPlug.Engine, tokenizer: tk, model_name: "tiny"]
    :ignore = Vapor.Plug.start_link(opts)
    {:ok, srv} = Serve.start_link(Keyword.merge(opts, name: nil, port: 0))
    {:ok, bandit} = Bandit.start_link(plug: VaporPlug.Router, port: 0, ip: :loopback, startup_log: false)
    {:ok, {_ip, port}} = ThousandIsland.listener_info(bandit)
    :inets.start()
    {:ok, serve: "http://127.0.0.1:#{Serve.port(srv)}", port: port}
  end

  defp post(url, body) do
    {:ok, {{_, status, _}, headers, resp}} =
      :httpc.request(:post, {~c"#{url}", [], ~c"application/json", JSON.encode(body)}, [timeout: 120_000], body_format: :binary)

    {status, Map.new(headers, fn {k, v} -> {to_string(k), to_string(v)} end), resp}
  end

  defp plug(path, body, via \\ VaporPlug.Router) do
    conn(:post, path, JSON.encode(body)) |> put_req_header("content-type", "application/json") |> via.call(via.init([]))
  end

  # the data events of a server-sent event stream
  defp events(sse), do: for("data: " <> d <- String.split(sse, "\n\n", trim: true), d != "[DONE]", do: JSON.decode!(d))

  defp strip(r), do: Map.drop(r, ["id", "created"])

  test "same bodies and receipts as Vapor.Serve; prefix, 404s and a pre-parsed body", %{serve: serve} do
    req = %{prompt: "Once upon a time", max_tokens: 12, temperature: 0.8, seed: 3}
    {200, h, want} = post(serve <> "/v1/completions", req)

    conn = plug("/llm/v1/completions", req)
    assert conn.status == 200 and conn.halted
    assert get_resp_header(conn, "x-vapor-receipt") == [h["x-vapor-receipt"]]
    assert strip(JSON.decode!(conn.resp_body)) == strip(JSON.decode!(want))

    # after Plug.Parsers: the params are re-encoded, same result
    conn = conn(:post, "/v1/completions", JSON.encode(req)) |> put_req_header("content-type", "application/json") |> VaporPlug.Parsed.call([])
    assert get_resp_header(conn, "x-vapor-receipt") == [h["x-vapor-receipt"]]

    chat = %{messages: [%{role: "user", content: "Olá!"}], max_tokens: 8, temperature: 0}
    {200, h, want} = post(serve <> "/v1/chat/completions", chat)
    conn = plug("/llm/v1/chat/completions", chat)
    assert get_resp_header(conn, "x-vapor-receipt") == [h["x-vapor-receipt"]]
    assert strip(JSON.decode!(conn.resp_body)) == strip(JSON.decode!(want))

    assert %{"data" => [%{"id" => "tiny"}]} = JSON.decode!(conn(:get, "/llm/v1/models") |> VaporPlug.Router.call([]) |> Map.fetch!(:resp_body))
    assert (conn(:get, "/llm/nope") |> VaporPlug.Router.call([])).status == 404
    assert (conn(:get, "/elsewhere") |> VaporPlug.Router.call([])).resp_body == "app"
    assert plug("/llm/v1/completions", %{nope: 1}).status == 400
  end

  test "streaming: the same deltas and the receipt in the last chunk", %{serve: serve} do
    req = %{prompt: "The sea", max_tokens: 16, temperature: 0.9, seed: 11, stream: true}
    {200, _, want} = post(serve <> "/v1/completions", req)
    conn = plug("/llm/v1/completions", req)
    assert conn.state == :chunked and hd(get_resp_header(conn, "content-type")) =~ "text/event-stream"

    got = events(conn.resp_body)
    assert Enum.map(got, &strip/1) == Enum.map(events(want), &strip/1)
    assert List.last(got)["vapor_receipt"] =~ ~r/^[0-9a-f]{64}$/
  end

  test "over a real server (Bandit): tool calls valid for their schemas; a client hanging up is cancelled", %{port: port} do
    url = "http://127.0.0.1:#{port}/llm"

    tools = [%{type: "function", function: %{name: "get_weather", description: "Weather for a city",
                                               parameters: %{type: "object", properties: %{city: %{type: "string", maxLength: 12},
                                                                                         unit: %{enum: ["c", "f"]}},
                                                             required: ["city", "unit"]}}}]

    for seed <- 1..3 do
      {200, h, body} = post(url <> "/v1/chat/completions", %{messages: [%{role: "user", content: "Weather in Recife?"}], tools: tools,
                                                              tool_choice: "required", max_tokens: 400, temperature: 1.0, seed: seed})
      assert h["x-vapor-receipt"] =~ ~r/^[0-9a-f]{64}$/
      %{"choices" => [%{"finish_reason" => "tool_calls", "message" => %{"tool_calls" => [call | _]}}]} = JSON.decode!(body)
      args = JSON.decode!(call["function"]["arguments"])
      assert call["function"]["name"] == "get_weather" and args["unit"] in ["c", "f"] and is_binary(args["city"])
    end

    before = Engine.info(VaporPlug.Engine).stats.cancelled
    {:ok, s} = :gen_tcp.connect(~c"127.0.0.1", port, [:binary, active: false])
    body = JSON.encode(%{prompt: "Long story:", max_tokens: 200, temperature: 0.8, seed: 2, stream: true})
    :ok = :gen_tcp.send(s, "POST /llm/v1/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: #{byte_size(body)}\r\n\r\n" <> body)
    {:ok, first} = :gen_tcp.recv(s, 0, 30_000)
    assert first =~ "text/event-stream"
    :gen_tcp.close(s)

    wait = fn wait, n -> if Engine.info(VaporPlug.Engine).stats.cancelled > before or n == 0, do: :ok, else: (Process.sleep(20); wait.(wait, n - 1)) end
    wait.(wait, 250)
    assert Engine.info(VaporPlug.Engine).stats.cancelled == before + 1
  end
end
