defmodule Vapor.AgentTest do
  @moduledoc """
  Immutable agents (`Vapor.Agent`): the spec is a value with a lineage, the
  journal is hash-chained and Merkle-committed, replay re-derives every
  deterministic decision and never touches the world, resume is replay,
  capabilities are fixed by the spec, sealed data can be erased without
  breaking the chain — with a local vapor model, a remote OpenAI-compatible
  model (vapor's own server), Anthropic's Messages API (a stand-in server
  speaking its format) and tools from an MCP server (the official SDK).
  """
  use ExUnit.Case, async: false
  alias Vapor.Agent
  alias Vapor.Agent.{Journal, Keys, Spec}
  alias Vapor.AgentTest.Script

  @now ~U[2026-10-01 15:00:00Z]

  defp spec(extra \\ []) do
    Spec.new([name: "assistant", instructions: "Be precise.", model: %{"kind" => "local", "id" => "script"},
              tools: [%{name: "add", effect: "pure", parameters: %{"type" => "object", "properties" => %{"a" => %{"type" => "integer"}, "b" => %{"type" => "integer"}}, "required" => ["a", "b"]}},
                      %{name: "weather", effect: "observe", parameters: %{"type" => "object", "properties" => %{"city" => %{"type" => "string"}}, "required" => ["city"]}},
                      %{name: "send_report", effect: "act", parameters: %{"type" => "object", "properties" => %{"to" => %{"type" => "string"}}, "required" => ["to"]}}],
              grants: ["send_report"]] ++ extra)
  end

  defp impls(counter) do
    %{"add" => fn %{"a" => a, "b" => b}, _ -> {:ok, a + b} end,
      "weather" => fn %{"city" => c}, _ -> {:ok, %{"city" => c, "temp" => 21.5}} end,
      "send_report" => fn %{"to" => to}, ctx -> Elixir.Agent.update(counter, &[{to, ctx.idempotency_key} | &1]); {:ok, "sent"} end}
  end

  defp script, do: %Script{calls: [{"add", %{"a" => 2, "b" => 40}}, {"weather", %{"city" => "Recife"}}, {"send_report", %{"to" => "ops"}}]}

  setup do
    {:ok, counter} = Elixir.Agent.start_link(fn -> [] end)
    {:ok, counter: counter}
  end

  test "a run is a journal; replay verifies decisions and pure tools, reads observations, repeats no action", %{counter: c} do
    s = spec()
    {:ok, run} = Agent.run(s, "Add 2+40, check Recife, report.", backend: script(), impls: impls(c), now: @now, nonce: "n1")
    assert run.status == "final"
    assert run.answer == "done: 42 | {\"city\":\"Recife\",\"temp\":21.5} | \"sent\""
    assert [{"ops", key}] = Elixir.Agent.get(c, & &1)
    kinds = Enum.map(run.journal.events, & &1["kind"])
    # the action is announced (and, with a store, persisted) before it happens
    assert kinds == ["start", "model", "tool", "tool", "intent", "tool", "model", "final"]
    assert Enum.at(run.journal.events, 4)["data"] == %{"call_id" => "c2", "name" => "send_report", "idempotency_key" => key, "approval" => "granted"}
    assert :ok = Journal.verify(run.journal)
    assert List.last(Enum.filter(run.journal.events, &(&1["kind"] == "tool")))["data"]["idempotency_key"] == key

    # replay: 2 model decisions + 1 pure tool re-derived; observe + act read; nothing executed
    assert {:ok, %{verified: 3, observed: 2, mismatches: [], head: true}} = Agent.replay(s, run.journal, backend: script(), impls: impls(c))
    assert length(Elixir.Agent.get(c, & &1)) == 1

    # the same inputs give the same run, event for event
    {:ok, c2} = Elixir.Agent.start_link(fn -> [] end)
    {:ok, again} = Agent.run(s, "Add 2+40, check Recife, report.", backend: script(), impls: impls(c2), now: @now, nonce: "n1")
    assert again.journal.head == run.journal.head

    # a pure tool whose implementation changed is caught
    bad = Map.put(impls(c), "add", fn %{"a" => a, "b" => b}, _ -> {:ok, a + b + 1} end)
    assert {:error, %{mismatches: [%{what: :tool}]}} = Agent.replay(s, run.journal, backend: script(), impls: bad)

    # a different decision process is caught
    assert {:error, %{mismatches: [_ | _]}} = Agent.replay(s, run.journal, backend: %{script() | answer: "other: "}, impls: impls(c))
  end

  test "tampering, Merkle proofs, transport", %{counter: c} do
    {:ok, run} = Agent.run(spec(), "x", backend: script(), impls: impls(c), now: @now, nonce: "n2")
    j = run.journal
    root = Journal.root(j)

    forged = %{j | events: List.update_at(j.events, 2, &put_in(&1, ["data", "result"], 41))}
    assert {:error, {:broken_at, 2}} = Journal.verify(forged)
    assert {:error, %{tampered: {:broken_at, 2}}} = Agent.replay(spec(), forged, backend: script(), impls: impls(c))
    dropped = %{j | events: List.delete_at(j.events, 3)}
    assert {:error, {:broken_at, 3}} = Journal.verify(dropped)

    for e <- j.events, do: assert(Journal.member?(e, Journal.proof(j, e["seq"]), root))

    # the node that ran it attests; a rewritten journal (even re-chained) is not attested
    key = Vapor.Certificate.keygen()
    att = Journal.attest(j, key)
    assert Journal.attested?(j, att, [key.public])
    refute Journal.attested?(j, att, [Vapor.Certificate.keygen().public])
    rechained = Enum.reduce(Enum.drop(forged.events, 0), Journal.new(j.run_id), &Journal.append(&2, &1["kind"], &1["data"]))
    assert Journal.verify(rechained) == :ok
    refute Journal.attested?(rechained, att, [key.public])
    {:ok, back} = Journal.decode(Journal.encode(j))
    assert back == j and Journal.root(back) == root
  end

  test "resume is replay: a crashed run continues without repeating its action", %{counter: c} do
    s = spec()
    {:ok, full} = Agent.run(s, "go", backend: script(), impls: impls(c), now: @now, nonce: "n3")
    cut = fn n -> j = %{full.journal | events: Enum.take(full.journal.events, n)}; %{j | head: List.last(j.events)["hash"]} end

    # the process died right after the three tool results were written
    {:ok, resumed} = Agent.run(s, nil, backend: script(), impls: impls(c), journal: cut.(6))
    assert resumed.journal.head == full.journal.head
    assert length(Elixir.Agent.get(c, & &1)) == 1

    # it died between announcing the action and recording its result: the
    # action may have happened, so it is attempted again — recorded as a
    # retry, with the same idempotency key — and the journal still verifies
    {:ok, retried} = Agent.run(s, nil, backend: script(), impls: impls(c), journal: cut.(5))
    assert Enum.map(retried.journal.events, & &1["kind"]) == ["start", "model", "tool", "tool", "intent", "retry", "tool", "model", "final"]
    assert [{"ops", k}, {"ops", k}] = Elixir.Agent.get(c, & &1)
    assert retried.answer == full.answer
    assert {:ok, %{mismatches: []}} = Agent.replay(s, retried.journal, backend: script(), impls: impls(c))
  end

  test "a resumed run goes live only from a prefix that verifies; a halted run replays", %{counter: c} do
    s = spec()
    {:ok, full} = Agent.run(s, "go", backend: script(), impls: impls(c), now: @now, nonce: "n9")
    prefix = %{full.journal | events: Enum.take(full.journal.events, 3)}
    prefix = %{prefix | head: List.last(prefix.events)["hash"]}

    # another agent (an evolved spec), or a decision process that changed: no new event, no action
    assert {:error, {:diverged, [%{what: :spec} | _]}, _} = Agent.run(Spec.evolve(s, policy: %{"max_steps" => 9}), nil, backend: script(), impls: impls(c), journal: prefix)
    assert {:error, {:diverged, [%{what: :model} | _]}, _} = Agent.run(s, nil, backend: %{script() | calls: [{"add", %{"a" => 1, "b" => 1}}]}, impls: impls(c), journal: prefix)
    assert length(Elixir.Agent.get(c, & &1)) == 1
    assert {:error, :empty_journal, _} = Agent.run(s, nil, backend: script(), impls: impls(c), journal: %{prefix | events: []})

    # a prefix ending in an announced action that is not the one re-derived
    # (an edited, re-chained store): no retry, no action
    {:ok, c2} = Elixir.Agent.start_link(fn -> [] end)
    evs = Enum.take(full.journal.events, 5)
    forged = Enum.reduce(evs, Journal.new(full.journal.run_id), fn e, j ->
      d = if e["kind"] == "intent", do: Map.put(e["data"], "idempotency_key", String.duplicate("0", 64)), else: e["data"]
      Journal.append(j, e["kind"], d)
    end)
    assert {:error, {:diverged, [%{what: :intent} | _]}, _} = Agent.run(s, nil, backend: script(), impls: impls(c2), journal: forged)
    assert Elixir.Agent.get(c2, & &1) == []

    # a run stopped by max_steps verifies like any other
    short = Spec.evolve(s, policy: %{"max_steps" => 1})
    {:ok, halted} = Agent.run(short, "go", backend: script(), impls: impls(c), now: @now, nonce: "n10")
    assert halted.status == "halt" and List.last(halted.journal.events)["data"] == %{"reason" => "max_steps"}
    assert {:ok, %{mismatches: [], head: true}} = Agent.replay(short, halted.journal, backend: script(), impls: impls(c))
  end

  test "capabilities are part of the spec; arguments are checked; specs have a lineage", %{counter: c} do
    s = spec(grants: [])
    {:ok, run} = Agent.run(s, "go", backend: script(), impls: impls(c), now: @now, nonce: "n4")
    act = Enum.find(run.journal.events, &(&1["data"]["name"] == "send_report"))
    assert act["data"]["status"] == "denied"
    assert Elixir.Agent.get(c, & &1) == []

    # a human gate, recorded either way
    {:ok, run} = Agent.run(spec(), "go", backend: script(), impls: impls(c), now: @now, nonce: "n5", confirm: fn _ -> false end)
    assert Enum.find(run.journal.events, &(&1["data"]["name"] == "send_report"))["data"]["error"] == "not approved"
    {:ok, run} = Agent.run(spec(), "go", backend: script(), impls: impls(c), now: @now, nonce: "n5", confirm: fn _ -> true end)
    assert Enum.find(run.journal.events, &(&1["kind"] == "intent"))["data"]["approval"] == "approved"
    Elixir.Agent.update(c, fn _ -> [] end)

    bad_args = %Script{calls: [{"add", %{"a" => "2", "b" => 40}}, {"nope", %{}}]}
    {:ok, run} = Agent.run(spec(), "go", backend: bad_args, impls: impls(c), now: @now, nonce: "n6")
    [t1, t2] = for %{"kind" => "tool", "data" => d} <- run.journal.events, do: d
    assert t1["status"] == "error" and t1["error"] =~ "$.a"
    assert t2["error"] == "unknown tool"

    v2 = Spec.evolve(spec(), instructions: "Be precise and brief.")
    assert v2.parent == Spec.digest(spec()) and Spec.digest(v2) != Spec.digest(spec())
    {:ok, run1} = Agent.run(spec(), "go", backend: script(), impls: impls(c), now: @now, nonce: "n7")
    assert {:error, %{mismatches: [%{what: :spec} | _]}} = Agent.replay(v2, run1.journal, backend: script(), impls: impls(c))
  end

  test "erasure: shredding a subject's key redacts the data, the chain and root stay valid", %{counter: c} do
    {:ok, keys} = Keys.start_link()
    s = spec()
    {:ok, run} = Agent.run(s, "Maria's balance is 1.234,56", backend: script(), impls: impls(c), now: @now, nonce: "n8", keys: keys, subject: "maria")

    # a ciphertext that does not open under an existing key is tampering, not erasure
    bad = Enum.reduce(run.journal.events, Journal.new(run.journal.run_id), fn e, j ->
      d = if e["seq"] == 2, do: update_in(e["data"], ["result", "ct"], &(:crypto.exor(&1, :binary.copy(<<1>>, byte_size(&1))))), else: e["data"]
      Journal.append(j, e["kind"], d)
    end)
    assert {:error, %{mismatches: ms, redacted: 0}} = Agent.replay(s, bad, backend: script(), impls: impls(c), keys: keys)
    assert Enum.any?(ms, &(&1.what == :sealed))
    start = hd(run.journal.events)
    assert %{"sealed" => "maria"} = start["data"]["input"]
    # the input, the model's texts, call arguments and results are sealed; names and structure are not
    for secret <- ["1.234,56", "Recife", "\"temp\""], do: assert(:binary.match(Journal.encode(run.journal), secret) == :nomatch, secret)
    assert Enum.any?(run.journal.events, &(&1["data"]["name"] == "weather"))
    assert {:ok, %{verified: 3}} = Agent.replay(s, run.journal, backend: script(), impls: impls(c), keys: keys)

    root = Journal.root(run.journal)
    Keys.shred(keys, "maria")
    assert :ok = Journal.verify(run.journal)
    assert Journal.root(run.journal) == root
    assert {:ok, %{redacted: r}} = Agent.replay(s, run.journal, backend: script(), impls: impls(c), keys: keys)
    assert r > 0

    # erased stays erased: no new key for the subject, and the old run cannot be continued
    assert_raise ArgumentError, ~r/erased/, fn -> Agent.run(s, "again", backend: script(), impls: impls(c), keys: keys, subject: "maria") end
    unfinished = %{run.journal | events: Enum.take(run.journal.events, 3)}
    unfinished = %{unfinished | head: List.last(unfinished.events)["hash"]}
    assert {:error, :history_redacted, _} = Agent.run(s, nil, backend: script(), impls: impls(c), keys: keys, journal: unfinished)
  end
end

defmodule Vapor.AgentBackendsTest do
  @moduledoc false
  use ExUnit.Case, async: false
  alias Vapor.Agent
  alias Vapor.Agent.{Backend, Spec}
  alias Vapor.{Engine, Serve, Tensor, Tokenizer}
  import Vapor.TestHelpers

  @moduletag timeout: 900_000

  # a minimal HTTP server answering like the Messages API, recording requests
  defp fake_anthropic(test) do
    {:ok, ls} = :gen_tcp.listen(0, [:binary, packet: :http_bin, active: false, reuseaddr: true])
    {:ok, port} = :inet.port(ls)

    spawn_link(fn ->
      Enum.each(1..2, fn n ->
        {:ok, s} = :gen_tcp.accept(ls)
        {:ok, {:http_request, :POST, {:abs_path, "/v1/messages"}, _}} = :gen_tcp.recv(s, 0)
        headers = read_headers(s, %{})
        :ok = :inet.setopts(s, packet: :raw)
        {:ok, body} = :gen_tcp.recv(s, String.to_integer(headers["content-length"]))
        send(test, {:anthropic_request, headers, Vapor.JSON.decode!(body)})

        reply =
          if n == 1,
            do: %{"id" => "msg_1", "model" => "claude-x", "stop_reason" => "tool_use",
                  "content" => [%{"type" => "text", "text" => "Checking."}, %{"type" => "tool_use", "id" => "toolu_1", "name" => "add", "input" => %{"a" => 1, "b" => 2}}]},
            else: %{"id" => "msg_2", "model" => "claude-x", "stop_reason" => "end_turn", "content" => [%{"type" => "text", "text" => "It is 3."}]}

        b = Vapor.JSON.encode(reply)
        :gen_tcp.send(s, "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: #{byte_size(b)}\r\nconnection: close\r\n\r\n" <> b)
        :gen_tcp.close(s)
      end)
    end)

    "http://127.0.0.1:#{port}"
  end

  defp read_headers(s, acc) do
    case :gen_tcp.recv(s, 0) do
      {:ok, {:http_header, _, k, _, v}} -> read_headers(s, Map.put(acc, String.downcase(to_string(k)), v))
      {:ok, :http_eoh} -> acc
    end
  end

  defp add_spec(model) do
    Spec.new(name: "adder", model: model, instructions: "Use tools.",
             tools: [%{name: "add", effect: "pure", parameters: %{"type" => "object", "properties" => %{"a" => %{"type" => "integer"}, "b" => %{"type" => "integer"}}, "required" => ["a", "b"]}}])
  end

  @add %{"add" => &__MODULE__.add/2}
  def add(%{"a" => a, "b" => b}, _), do: {:ok, a + b}

  test "Anthropic Messages API: request shape, tool_use → tool_result, decisions recorded as observations" do
    base = fake_anthropic(self())
    s = add_spec(%{"kind" => "anthropic", "model" => "claude-x"})
    b = %Backend.Anthropic{base_url: base, model: "claude-x", api_key: "k"}
    {:ok, run} = Agent.run(s, "1+2?", backend: b, impls: @add, now: ~U[2026-10-01 15:00:00Z], nonce: "a")
    assert run.answer == "It is 3."

    assert_receive {:anthropic_request, h1, r1}
    assert h1["x-api-key"] == "k" and h1["anthropic-version"] == "2023-06-01"
    assert r1["system"] == "Use tools."
    assert [%{"name" => "add", "input_schema" => %{"type" => "object"}}] = r1["tools"]
    assert_receive {:anthropic_request, _, r2}
    [_, %{"role" => "assistant", "content" => [_, %{"type" => "tool_use", "id" => "toolu_1"}]}, %{"role" => "user", "content" => [%{"type" => "tool_result", "tool_use_id" => "toolu_1", "content" => "3"}]}] = r2["messages"]

    # replay needs no network: decisions are read, the pure tool re-run
    assert {:ok, %{observed: 2, verified: 1, mismatches: []}} = Agent.replay(s, run.journal, impls: @add)
  end

  @tag :native
  @tag :vocab
  test "a local vapor model: every decision re-derived bit for bit; the same agent through vapor's OpenAI server is an observation" do
    {:ok, g} = Vapor.Ingest.GGUF.read(Path.expand("../fixtures/vocab/ggml-vocab-qwen2.gguf", __DIR__))
    {:ok, tk} = Tokenizer.from_gguf(g.metadata)
    v = Tokenizer.vocab_size(tk)
    {:ok, c} = Vapor.Model.Config.from_map(tiny_config("qwen3", %{"vocab_size" => v, "head_dim" => 16, "max_position_embeddings" => 1024}))
    :rand.seed(:exsss, {7, 8, 9})
    block = for(<<x::32 <- :rand.bytes(8192 * 64 * 4)>>, into: <<>>, do: <<Bitwise.bor(Bitwise.band(x, 0x807F_FFFF), 0x3E00_0000)::32-little>>)
    ws = Map.put(tiny_weights(c), "model.embed_tokens.weight", Tensor.new(:f32, [v, 64], binary_part(:binary.copy(block, div(v, 8192) + 1), 0, v * 64 * 4)))
    dir = Path.join(System.tmp_dir!(), "vapor-agent-tpl")
    File.mkdir_p!(dir)
    File.cp!(Path.expand("../fixtures/templates/Qwen-Qwen3-0.6B.jinja", __DIR__), Path.join(dir, "chat_template.jinja"))
    {:ok, tpl} = Vapor.Chat.load_template(dir)
    {:ok, e} = Engine.start_link(config: c, weights: ws, tokenizer: tk, max_seq: 1024, page: 16, sequences: 2, step_tokens: 256)
    local = Backend.Local.new(engine: e, tokenizer: tk, template: tpl, model_id: "tiny-qwen3")

    s = Spec.new(name: "local", model: %{"kind" => "local", "id" => "tiny-qwen3"}, instructions: "Use tools.",
                 tools: [%{name: "add", effect: "pure", parameters: %{"type" => "object", "properties" => %{"a" => %{"type" => "integer"}}, "required" => ["a"]}}],
                 policy: %{"temperature" => 0.9, "max_tokens" => 24, "max_steps" => 2})

    {:ok, run} = Agent.run(s, "Say something.", backend: local, impls: @add, now: ~U[2026-10-01 15:00:00Z], nonce: "l")
    assert run.status in ["final", "halt"]
    [m | _] = for %{"kind" => "model", "data" => d} <- run.journal.events, do: d
    assert m["deterministic"] and is_list(m["record"]["tokens"])

    # replay re-runs the model and must meet the same tokens — and the recorded head
    assert {:ok, %{verified: n, mismatches: [], head: true}} = Agent.replay(s, run.journal, backend: local, impls: @add)
    assert n >= 1

    # the same model behind vapor's OpenAI-compatible server: a remote, observed decision
    {:ok, srv} = Serve.start_link(engine: e, tokenizer: tk, port: 0, model_name: "tiny-qwen3", template: tpl, now: fn -> ~N[2026-10-01 15:00:00] end)
    remote = %Backend.OpenAI{base_url: "http://127.0.0.1:#{Serve.port(srv)}/v1", model: "tiny-qwen3"}
    rs = %{s | model: %{"kind" => "openai", "model" => "tiny-qwen3"}}
    {:ok, rrun} = Agent.run(rs, "Say something.", backend: remote, impls: @add, now: ~U[2026-10-01 15:00:00Z], nonce: "r")
    assert {:ok, %{observed: o, mismatches: []}} = Agent.replay(rs, rrun.journal, impls: @add)
    assert o >= 1
  end

  @tag :mcp
  test "MCP: the official SDK's server; tools declared pure are re-run on replay through the server" do
    srv = Path.join(System.tmp_dir!(), "vapor_mcp_server.py")

    File.write!(srv, """
    from mcp.server.mcpserver import MCPServer
    s = MCPServer("calc")
    @s.tool()
    def add(a: int, b: int) -> int:
        \"\"\"Add two integers.\"\"\"
        return a + b
    @s.tool()
    def shout(text: str) -> str:
        \"\"\"Upper-case a text.\"\"\"
        return text.upper()
    s.run()
    """)

    {:ok, mcp} = Vapor.Agent.MCP.start_link(cmd: [python(), srv])
    {tools, impls} = Vapor.Agent.MCP.agent_tools(mcp, effects: %{"add" => "pure"})
    assert Enum.map(tools, & &1["name"]) |> Enum.sort() == ["add", "shout"]
    assert Enum.find(tools, &(&1["name"] == "shout"))["effect"] == "act"
    assert {:ok, 42} = Vapor.Agent.MCP.call(mcp, "add", %{"a" => 2, "b" => 40})

    s = Spec.new(name: "mcp", model: %{"kind" => "local", "id" => "script"}, tools: tools)
    script = %Vapor.AgentTest.Script{calls: [{"add", %{"a" => 20, "b" => 22}}, {"shout", %{"text" => "hi"}}]}
    {:ok, run} = Agent.run(s, "go", backend: script, impls: impls, now: ~U[2026-10-01 15:00:00Z], nonce: "m")
    assert run.answer =~ "42" and run.answer =~ "capability not granted"
    assert {:ok, %{verified: 3, mismatches: []}} = Agent.replay(s, run.journal, backend: script, impls: impls)

    # the server dies: calls fail with a reason, the client (and its caller) live on
    {:os_pid, os} = Port.info(:sys.get_state(mcp).port, :os_pid)
    System.cmd("kill", ["-9", Integer.to_string(os)])
    Process.sleep(200)
    assert {:error, "MCP server exited" <> _} = Vapor.Agent.MCP.call(mcp, "add", %{"a" => 1, "b" => 1})
    assert Process.alive?(mcp)
  end
end
