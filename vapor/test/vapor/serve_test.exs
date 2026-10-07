defmodule Vapor.ServeTest do
  @moduledoc """
  Phase P5 — the OpenAI-compatible server, end to end: a Qwen2-architecture
  model with the real Qwen2 vocabulary (random weights) behind
  `Vapor.Engine`, driven over HTTP by OTP's `:httpc` and by the official
  `openai` Python client (streaming and not), with concurrent clients.
  Outputs must equal the engine's own for the same request.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Engine, Serve, Tensor, Tokenizer}
  alias Vapor.Ingest.GGUF
  alias Vapor.Model.Config
  import Vapor.TestHelpers

  @moduletag :native
  @moduletag :vocab
  @moduletag timeout: 900_000

  setup_all do
    {:ok, g} = GGUF.read(Path.expand("../fixtures/vocab/ggml-vocab-qwen2.gguf", __DIR__))
    {:ok, tk} = Tokenizer.from_gguf(g.metadata)
    v = Tokenizer.vocab_size(tk)
    {:ok, c} = Config.from_map(tiny_config("qwen2", %{"vocab_size" => v, "max_position_embeddings" => 256}))

    # a large embedding is drawn from random bits directly (bounded exponent)
    :rand.seed(:exsss, {1, 2, 3})
    embed = for(<<b::32 <- :rand.bytes(v * 64 * 4)>>, into: <<>>, do: <<Bitwise.bor(Bitwise.band(b, 0x807F_FFFF), 0x3E00_0000)::32-little>>)
    ws = Map.put(tiny_weights(c), "model.embed_tokens.weight", Tensor.new(:f32, [v, 64], embed))

    {:ok, e} = Engine.start_link(config: c, weights: ws, tokenizer: tk, max_seq: 128, page: 16, sequences: 4, step_tokens: 32)
    {:ok, srv} = Serve.start_link(engine: e, tokenizer: tk, port: 0, model_name: "tiny-qwen2")
    :inets.start()
    {:ok, url: "http://127.0.0.1:#{Serve.port(srv)}", engine: e, tk: tk}
  end

  defp post(url, path, body) do
    {:ok, {{_, status, _}, _h, resp}} =
      :httpc.request(:post, {~c"#{url}#{path}", [], ~c"application/json", Vapor.JSON.encode(body)}, [timeout: 300_000], body_format: :binary)

    {status, resp}
  end

  test "models, completions (text and ids), chat, errors", %{url: url, engine: e, tk: tk} do
    {:ok, {{_, 200, _}, _, models}} = :httpc.request(~c"#{url}/v1/models")
    assert %{"data" => [%{"id" => "tiny-qwen2"}]} = Vapor.JSON.decode!(to_string(models))

    {200, body} = post(url, "/v1/completions", %{prompt: "Hello, world", max_tokens: 6, temperature: 0})
    r = Vapor.JSON.decode!(body)
    [choice] = r["choices"]
    assert r["object"] == "text_completion" and choice["finish_reason"] in ["length", "stop"]

    # the same request straight to the engine gives the same text
    {:ok, ids, _, usage} = Engine.complete(e, Tokenizer.encode(tk, "Hello, world"), max_tokens: 6, temperature: 0.0)
    assert choice["text"] == Serve.valid_prefix(Tokenizer.decode(tk, ids)) |> elem(0)
    assert r["usage"]["completion_tokens"] == usage.completion_tokens

    {200, body} = post(url, "/v1/completions", %{prompt: [9707, 11, 1879], max_tokens: 3, temperature: 0.7, seed: 5})
    assert %{"choices" => [_]} = Vapor.JSON.decode!(body)

    {200, body} = post(url, "/v1/chat/completions", %{messages: [%{role: "user", content: "Hi!"}], max_tokens: 5, temperature: 0})
    assert %{"object" => "chat.completion", "choices" => [%{"message" => %{"role" => "assistant"}}]} = Vapor.JSON.decode!(body)

    assert {400, _} = post(url, "/v1/chat/completions", %{messages: []})
    assert {400, _} = post(url, "/v1/completions", %{prompt: "x", max_tokens: 1000})
    assert {400, _} = post(url, "/v1/completions", %{nope: 1})
  end

  test "UTF-8 is cut only at character boundaries" do
    assert Serve.valid_prefix("ab\xE6\x97") == {"ab", "\xE6\x97"}
    assert Serve.valid_prefix("ab\xE6\x97\xA5") == {"ab日", ""}
    assert Serve.valid_prefix("a\xFFb") == {"ab", ""}
  end

  test "a streaming client that hangs up takes its request out of the engine", %{url: url, engine: e} do
    before = Engine.info(e).stats.cancelled
    %{port: port} = URI.parse(url)
    {:ok, s} = :gen_tcp.connect(~c"127.0.0.1", port, [:binary, active: false])
    body = Vapor.JSON.encode(%{prompt: [9707, 11, 1879], max_tokens: 120, temperature: 0.8, seed: 1, stream: true})
    :ok = :gen_tcp.send(s, "POST /v1/completions HTTP/1.1\r\nhost: x\r\ncontent-length: #{byte_size(body)}\r\n\r\n" <> body)
    {:ok, first} = :gen_tcp.recv(s, 0, 30_000)
    assert first =~ "text/event-stream"
    :gen_tcp.close(s)

    # the next writes fail; the server cancels instead of generating 120 tokens for nobody
    wait = fn wait, n -> if Engine.info(e).stats.cancelled > before or n == 0, do: :ok, else: (Process.sleep(20); wait.(wait, n - 1)) end
    wait.(wait, 250)
    info = Engine.info(e)
    assert info.stats.cancelled == before + 1 and info.active == 0
  end

  @tag :openai
  test "the official openai client: completions, chat, streaming, 6 concurrent clients", %{url: url} do
    script = """
    import sys, json, concurrent.futures as cf
    from openai import OpenAI
    c = OpenAI(base_url=sys.argv[1] + "/v1", api_key="none")
    out = {}
    out["models"] = [m.id for m in c.models.list().data]
    r = c.completions.create(model="tiny-qwen2", prompt="The capital of", max_tokens=8, temperature=0)
    out["text"] = r.choices[0].text
    s = c.completions.create(model="tiny-qwen2", prompt="The capital of", max_tokens=8, temperature=0, stream=True)
    out["streamed"] = "".join(ch.choices[0].text for ch in s)
    r = c.chat.completions.create(model="tiny-qwen2", messages=[{"role": "user", "content": "Olá"}], max_tokens=8, temperature=0)
    out["chat"] = r.choices[0].message.content
    s = c.chat.completions.create(model="tiny-qwen2", messages=[{"role": "user", "content": "Olá"}], max_tokens=8, temperature=0, stream=True)
    out["chat_streamed"] = "".join(ch.choices[0].delta.content or "" for ch in s)
    def one(i):
        r = c.completions.create(model="tiny-qwen2", prompt=f"Story {i}:", max_tokens=10, temperature=0.9, seed=i)
        return r.choices[0].text
    with cf.ThreadPoolExecutor(6) as ex:
        out["concurrent"] = list(ex.map(one, range(6)))
    out["sequential"] = [one(i) for i in range(6)]
    print(json.dumps(out))
    """

    out = py!(script, [url]) |> String.trim() |> String.split("\n") |> List.last() |> Vapor.JSON.decode!()
    assert out["models"] == ["tiny-qwen2"]
    assert out["text"] == out["streamed"]
    assert out["chat"] == out["chat_streamed"]
    # batch invariance seen from outside: concurrent = one at a time
    assert out["concurrent"] == out["sequential"]
  end
end
