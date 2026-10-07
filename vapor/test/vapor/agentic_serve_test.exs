defmodule Vapor.AgenticServeTest do
  @moduledoc """
  The server's agent features end to end, on a model that knows nothing:
  random weights, the real Qwen2 vocabulary (151 936 tokens) and the real
  Qwen3 chat template. Whatever the weights, a constrained output is a
  member of its language — so `tool_choice: "required"` returns calls to
  declared tools with schema-valid arguments, a `json_schema` response is
  valid for its schema, and `/v1/embeddings` returns unit vectors. Every
  response carries a receipt that re-deriving the generation reproduces.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Engine, Serve, Tensor, Tokenizer}
  alias Vapor.Ingest.GGUF
  alias Vapor.Model.Config
  import Vapor.TestHelpers

  @moduletag :native
  @moduletag :vocab
  @moduletag timeout: 900_000

  @tools [%{"type" => "function", "function" => %{"name" => "get_weather", "description" => "Forecast for a city",
             "parameters" => %{"type" => "object", "properties" => %{"city" => %{"type" => "string", "maxLength" => 12},
                                                                     "days" => %{"type" => "integer"},
                                                                     "units" => %{"enum" => ["c", "f"]}},
                               "required" => ["city", "days"]}}},
          %{"type" => "function", "function" => %{"name" => "search_docs", "description" => "Search",
             "parameters" => %{"type" => "object", "properties" => %{"query" => %{"type" => "string", "maxLength" => 16}}, "required" => ["query"]}}}]

  setup_all do
    {:ok, g} = GGUF.read(Path.expand("../fixtures/vocab/ggml-vocab-qwen2.gguf", __DIR__))
    {:ok, tk} = Tokenizer.from_gguf(g.metadata)
    v = Tokenizer.vocab_size(tk)
    {:ok, c} = Config.from_map(tiny_config("qwen3", %{"vocab_size" => v, "head_dim" => 16, "max_position_embeddings" => 1024}))
    # 8192 random rows, tiled over the vocabulary (bounded exponent)
    :rand.seed(:exsss, {4, 5, 6})
    block = for(<<b::32 <- :rand.bytes(8192 * 64 * 4)>>, into: <<>>, do: <<Bitwise.bor(Bitwise.band(b, 0x807F_FFFF), 0x3E00_0000)::32-little>>)
    embed = binary_part(:binary.copy(block, div(v, 8192) + 1), 0, v * 64 * 4)
    ws = Map.put(tiny_weights(c), "model.embed_tokens.weight", Tensor.new(:f32, [v, 64], embed))

    {:ok, tpl} = Vapor.Chat.load_template(Path.expand("../fixtures/templates", __DIR__) |> then(&write_cfg/1))
    {:ok, e} = Engine.start_link(config: c, weights: ws, tokenizer: tk, max_seq: 1024, page: 16, sequences: 2, step_tokens: 256)
    {:ok, emb} = Vapor.Embed.open(config: c, weights: ws, tokenizer: tk, max_seq: 64)
    {:ok, srv} = Serve.start_link(engine: e, tokenizer: tk, port: 0, model_name: "tiny-qwen3", template: tpl, embedder: emb,
                                  now: fn -> ~N[2026-10-01 12:00:00] end)
    :inets.start()
    {:ok, url: "http://127.0.0.1:#{Serve.port(srv)}", engine: e, tk: tk}
  end

  # a directory whose chat_template.jinja is Qwen3's
  defp write_cfg(dir) do
    out = Path.join(System.tmp_dir!(), "vapor-qwen3-tpl")
    File.mkdir_p!(out)
    File.cp!(Path.join(dir, "Qwen-Qwen3-0.6B.jinja"), Path.join(out, "chat_template.jinja"))
    out
  end

  defp post(url, path, body) do
    {:ok, {{_, status, _}, headers, resp}} =
      :httpc.request(:post, {~c"#{url}#{path}", [], ~c"application/json", Vapor.JSON.encode(body)}, [timeout: 600_000], body_format: :binary)

    {status, Map.new(headers, fn {k, v} -> {to_string(k), to_string(v)} end), resp}
  end

  defp valid_args?("get_weather", a), do: is_binary(a["city"]) and String.length(a["city"]) <= 12 and is_integer(a["days"]) and a["units"] in [nil, "c", "f"] and Map.keys(a) -- ["city", "days", "units"] == []
  defp valid_args?("search_docs", a), do: is_binary(a["query"]) and String.length(a["query"]) <= 16 and Map.keys(a) == ["query"]

  test "tool_choice required: calls to declared tools, arguments valid for their schemas, for any seed", %{url: url} do
    for seed <- 1..4 do
      {200, h, body} =
        post(url, "/v1/chat/completions", %{messages: [%{role: "user", content: "Weather in Recife, 2 days?"}], tools: @tools,
                                            tool_choice: "required", max_tokens: 240, temperature: 1.0, seed: seed})

      r = Vapor.JSON.decode!(body)
      [%{"finish_reason" => "tool_calls", "message" => %{"tool_calls" => [_ | _] = calls}}] = r["choices"]
      assert String.length(h["x-vapor-receipt"]) == 64

      for %{"type" => "function", "id" => "call_" <> _, "function" => %{"name" => n, "arguments" => args}} <- calls do
        assert n in ["get_weather", "search_docs"]
        assert valid_args?(n, Vapor.JSON.decode!(args)), "seed #{seed}: #{n} #{args}"
      end
    end

    # a named function narrows the choice
    {200, _, body} =
      post(url, "/v1/chat/completions", %{messages: [%{role: "user", content: "find docs"}], tools: @tools, max_tokens: 200,
                                          tool_choice: %{type: "function", function: %{name: "search_docs"}}, temperature: 0.8, seed: 3})

    [%{"message" => %{"tool_calls" => calls}}] = Vapor.JSON.decode!(body)["choices"]
    assert Enum.all?(calls, &(&1["function"]["name"] == "search_docs"))
  end

  test "json_schema structured output is valid for the schema; the receipt is reproducible", %{url: url} do
    schema = %{"type" => "object", "properties" => %{"answer" => %{"type" => "string", "maxLength" => 10},
                                                     "confidence" => %{"type" => "number"}, "tags" => %{"type" => "array", "items" => %{"enum" => ["a", "b"]}, "maxItems" => 3}},
               "required" => ["answer", "confidence", "tags"]}

    req = %{messages: [%{role: "user", content: "Answer in JSON."}], max_tokens: 150, temperature: 0.9, seed: 11,
            response_format: %{type: "json_schema", json_schema: %{name: "a", strict: true, schema: schema}}}

    {200, h1, body} = post(url, "/v1/chat/completions", req)
    [%{"message" => %{"content" => content}, "finish_reason" => fin}] = Vapor.JSON.decode!(body)["choices"]

    if fin == "stop" do
      out = Vapor.JSON.decode!(content)
      assert is_binary(out["answer"]) and String.length(out["answer"]) <= 10 and is_number(out["confidence"])
      assert Enum.all?(out["tags"], &(&1 in ["a", "b"])) and length(out["tags"]) <= 3
    end

    # same request ⇒ same bits ⇒ same receipt
    {200, h2, body2} = post(url, "/v1/chat/completions", req)
    assert h1["x-vapor-receipt"] == h2["x-vapor-receipt"]
    assert Vapor.JSON.decode!(body2)["choices"] |> hd() |> get_in(["message", "content"]) == content
  end

  test "tool_choice auto stays free text unless a call opens; json_object; embeddings", %{url: url} do
    {200, _, body} = post(url, "/v1/chat/completions", %{messages: [%{role: "user", content: "hi"}], tools: @tools, max_tokens: 12, temperature: 0})
    [%{"message" => m}] = Vapor.JSON.decode!(body)["choices"]
    assert Map.has_key?(m, "content")

    {200, _, body} = post(url, "/v1/chat/completions", %{messages: [%{role: "user", content: "JSON please"}], max_tokens: 60, temperature: 0.7, seed: 2,
                                                         response_format: %{type: "json_object"}})
    [%{"message" => %{"content" => c}, "finish_reason" => fin}] = Vapor.JSON.decode!(body)["choices"]
    if fin == "stop", do: assert({:ok, %{}} = Vapor.JSON.decode(c))

    {200, _, body} = post(url, "/v1/embeddings", %{input: ["Olá mundo", "hello world"]})
    %{"data" => [%{"embedding" => a}, %{"embedding" => b}]} = Vapor.JSON.decode!(body)
    norm = fn v -> :math.sqrt(Enum.reduce(v, 0.0, &(&1 * &1 + &2))) end
    assert abs(norm.(a) - 1.0) < 1.0e-6 and abs(norm.(b) - 1.0) < 1.0e-6 and a != b
  end
end
