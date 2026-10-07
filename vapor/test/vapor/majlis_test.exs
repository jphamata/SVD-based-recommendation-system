defmodule Vapor.MajlisTest do
  use ExUnit.Case, async: true
  alias Vapor.Majlis, as: M
  alias Vapor.MajlisTest.Echo

  setup do
    dir = Path.join(System.tmp_dir!(), "majlis-#{System.unique_integer([:positive])}")
    on_exit(fn -> File.rm_rf!(dir) end)
    {:ok, m} = start(dir)
    {:ok, dir: dir, m: m}
  end

  defp start(dir, extra \\ []) do
    clock = fn -> "2026-10-07T12:00:#{String.pad_leading(to_string(rem(System.unique_integer([:positive, :monotonic]), 60)), 2, "0")}.000Z" end
    M.start_link([dir: dir, backends: %{"echo" => %Echo{sink: self()}}, default: "echo", clock: clock] ++ extra)
  end

  defp contents(m, tid), do: m |> M.path(tid) |> elem(1) |> Map.fetch!(:messages) |> Enum.map(&{&1.role, &1.content})

  test "say and reply; the conversation and its ids survive a restart", %{m: m, dir: dir} do
    {:ok, t} = M.new(m, title: "first", system: "be brief")
    {:ok, %{node: a1}} = M.ask(m, t, "hello")
    assert contents(m, t) == [{"user", "hello"}, {"assistant", "echo: hello (2 msgs)"}]
    {:ok, p1} = M.path(m, t)
    GenServer.stop(m)
    {:ok, m2} = start(dir)
    {:ok, p2} = M.path(m2, t)
    assert Enum.map(p2.messages, & &1.id) == Enum.map(p1.messages, & &1.id)
    assert p2.head == a1
    assert [%{title: "first", messages: 2}] = M.threads(m2)
  end

  test "a message's id commits to its whole history: the same words after a different past are another message", %{m: m} do
    {:ok, t1} = M.new(m)
    {:ok, t2} = M.new(m)
    {:ok, _} = M.say(m, t1, "x")
    {:ok, a} = M.say(m, t1, "same words")
    {:ok, _} = M.say(m, t2, "y")
    {:ok, b} = M.say(m, t2, "same words")
    refute a == b
  end

  test "editing makes a branch; both versions stay; switching follows the newest leaf", %{m: m} do
    {:ok, t} = M.new(m)
    {:ok, %{node: _}} = M.ask(m, t, "what is 2+2?")
    {:ok, p} = M.path(m, t)
    [q, _a] = p.messages
    {:ok, q2} = M.edit(m, t, String.slice(q.id, 0, 10), "what is 3+3?")
    {:ok, _} = M.reply(m, t)
    assert contents(m, t) == [{"user", "what is 3+3?"}, {"assistant", "echo: what is 3+3? (1 msgs)"}]
    {:ok, n} = M.node(m, q2)
    assert n.index == 2 and length(n.siblings) == 2
    # back to the first version: the head goes to its answer, not to the bare question
    {:ok, leaf} = M.switch(m, t, q.id)
    assert contents(m, t) == [{"user", "what is 2+2?"}, {"assistant", "echo: what is 2+2? (1 msgs)"}]
    assert {:ok, %{role: "assistant"}} = M.node(m, leaf)
  end

  test "regenerate: a sibling answer to the same question; the head follows it", %{m: m} do
    {:ok, t} = M.new(m)
    {:ok, %{node: a1}} = M.ask(m, t, "tell me")
    {:ok, %{node: a2}} = M.regenerate(m, t, a1, temperature: 0.9)
    refute a1 == a2
    {:ok, n2} = M.node(m, a2)
    assert n2.siblings == [a1, a2] and n2.index == 2
    assert {:ok, %{head: ^a2}} = M.path(m, t)
    assert {:error, msg} = M.regenerate(m, t, n2.parent)
    assert msg =~ "assistant"
    # an earlier answer regenerated on the path in view: the head moves to the new one (the later turns stay on their branch)
    {:ok, %{node: b1}} = M.ask(m, t, "and then?")
    {:ok, %{node: a3}} = M.regenerate(m, t, a2)
    assert {:ok, %{head: ^a3}} = M.path(m, t)
    # a question not in view (the person is on another branch): the answer is kept, the head stays
    {:ok, _} = M.switch(m, t, a1)
    {:ok, b} = M.node(m, b1)
    {:ok, %{node: b2}} = M.reply(m, t, at: b.parent)
    assert {:ok, %{head: ^a1}} = M.path(m, t)
    assert {:ok, %{siblings: [^b1, ^b2]}} = M.node(m, b2)
  end

  test "fork is a pointer: nothing is copied, and the two threads share their past and diverge after it", %{m: m} do
    {:ok, t} = M.new(m, title: "trunk")
    {:ok, _} = M.ask(m, t, "one")
    {:ok, _} = M.ask(m, t, "two")
    before = M.stats(m).messages
    {:ok, p} = M.path(m, t)
    {:ok, f} = M.fork(m, t, at: Enum.at(p.messages, 1).id, title: "branch")
    assert M.stats(m).messages == before
    {:ok, _} = M.ask(m, f, "three, in the fork")
    assert length(contents(m, t)) == 4
    assert contents(m, f) |> Enum.take(2) == contents(m, t) |> Enum.take(2)
    assert {:ok, %{settings: %{"forked_from" => %{"thread" => ^t}}}} = M.thread(m, f)
  end

  test "context: the last turn and the pins always go; the oldest drop first; the dropped are listed", %{m: m} do
    {:ok, t} = M.new(m, system: "S", budget: 256)
    for i <- 1..12, do: {:ok, _} = M.ask(m, t, "message #{i} " <> String.duplicate("word ", 40))
    {:ok, p} = M.path(m, t)
    first = hd(p.messages).id
    :ok = M.pin(m, t, first)
    {:ok, _} = M.say(m, t, "final question")
    {:ok, ctx} = M.context(m, t)
    sent = for i <- ctx.items, i.status in ["sent", "pinned"], do: i.id
    assert first in sent
    assert ctx.dropped != [] and ctx.tokens <= 256
    assert List.last(ctx.messages)["content"] == "final question"
    assert hd(ctx.messages) == %{"role" => "system", "content" => "S"}
    refute ctx.exact
  end

  test "compaction: the summary replaces the prefix it names, and the model sees summary + the rest", %{m: m} do
    {:ok, t} = M.new(m, system: "S")
    for i <- 1..4, do: {:ok, _} = M.ask(m, t, "turn #{i}")
    {:ok, p} = M.path(m, t)
    upto = Enum.at(p.messages, 3).id
    :ok = M.compact(m, t, upto, "the user counted turns 1 and 2")
    {:ok, ctx} = M.context(m, t)
    assert Enum.count(ctx.items, &(&1.status == "summarized")) == 4
    assert hd(ctx.messages)["content"] =~ "the user counted turns 1 and 2"
    assert hd(ctx.messages)["content"] =~ "covers 4 messages"
    # a model-written summary goes through the same path
    {:ok, s} = M.summarize(m, t)
    assert s.text =~ "echo"
    :ok = M.uncompact(m, t)
    {:ok, ctx2} = M.context(m, t)
    assert Enum.all?(ctx2.items, &(&1.status != "summarized"))
  end

  test "search finds messages in any thread and says which threads hold them", %{m: m} do
    {:ok, t1} = M.new(m)
    {:ok, t2} = M.new(m)
    {:ok, _} = M.say(m, t1, "the Kulisch accumulator is exact")
    {:ok, _} = M.say(m, t2, "a cupel catches silent corruption")
    {:ok, _} = M.say(m, t2, "acumulador de Kulisch, em português também")
    [hit | _] = M.search(m, "kulisch exact")
    assert hit.snippet =~ "Kulisch" and hit.threads == [t1]
    assert length(M.search(m, "kulisch")) == 2
    assert M.search(m, "   ") == []
  end

  test "vapor export round-trips with the same ids; a tampered export is refused", %{m: m} do
    {:ok, t} = M.new(m, title: "exported")
    {:ok, _} = M.ask(m, t, "q1")
    {:ok, p} = M.path(m, t)
    {:ok, _} = M.edit(m, t, hd(p.messages).id, "q1, edited")
    {:ok, _} = M.reply(m, t)
    {:ok, json} = M.export(m, t, :json)
    {:ok, [t2]} = M.import(m, json)
    assert M.path(m, t2) |> elem(1) |> Map.get(:head) == M.path(m, t) |> elem(1) |> Map.get(:head)
    {:ok, tree} = M.tree(m, t2)
    assert length(tree.nodes) == 4
    bad = String.replace(json, "q1, edited", "q1, EDITED")
    assert {:error, msg} = M.import(m, bad)
    assert msg =~ "altered"
    {:ok, md} = M.export(m, t, :markdown)
    assert md =~ "# exported" and md =~ "version 2 of 2"
  end

  test "a ChatGPT export: its tree, its hidden nodes and its current branch", %{m: m} do
    conv = %{
      "title" => "Recife weather", "create_time" => 1_700_000_000.5, "current_node" => "a2",
      "mapping" => %{
        "root" => %{"id" => "root", "message" => nil, "parent" => nil, "children" => ["sys"]},
        "sys" => %{"id" => "sys", "parent" => "root", "children" => ["u1", "u1b"],
                   "message" => %{"author" => %{"role" => "system"}, "content" => %{"content_type" => "text", "parts" => [""]}, "metadata" => %{"is_visually_hidden_from_conversation" => true}}},
        "u1" => %{"id" => "u1", "parent" => "sys", "children" => ["a1"], "message" => msg("user", "weather in Recife?", 1)},
        "a1" => %{"id" => "a1", "parent" => "u1", "children" => [], "message" => msg("assistant", "Sunny.", 2)},
        "u1b" => %{"id" => "u1b", "parent" => "sys", "children" => ["a2"], "message" => msg("user", "weather in Recife tomorrow?", 3)},
        "a2" => %{"id" => "a2", "parent" => "u1b", "children" => [], "message" => msg("assistant", "Rain, then sun.", 4)}
      }
    }

    {:ok, [t]} = M.import(m, Vapor.JSON.encode([conv]))
    assert contents(m, t) == [{"user", "weather in Recife tomorrow?"}, {"assistant", "Rain, then sun."}]
    {:ok, p} = M.path(m, t)
    assert length(hd(p.messages).siblings) == 2
    assert [%{title: "Recife weather"}] = Enum.filter(M.threads(m), &(&1.id == t))
  end

  test "a Claude export, linear and with non-text parts noted", %{m: m} do
    conv = %{"uuid" => "c1", "name" => "notes", "created_at" => "2026-01-01T00:00:00Z",
             "chat_messages" => [
               %{"uuid" => "m1", "sender" => "human", "text" => "summarise this", "content" => [%{"type" => "text", "text" => "summarise this"}, %{"type" => "image"}], "created_at" => "2026-01-01T00:00:01Z"},
               %{"uuid" => "m2", "sender" => "assistant", "text" => "Done.", "content" => [], "created_at" => "2026-01-01T00:00:02Z"}
             ]}

    {:ok, [t]} = M.import(m, Vapor.JSON.encode([conv]))
    assert contents(m, t) == [{"user", "summarise this\n[non-text part: image]"}, {"assistant", "Done."}]
    assert {:error, _} = M.import(m, ~s({"neither": 1}))
  end

  test "share: a computed read-only capability; revoking kills every link at once", %{m: m} do
    {:ok, t} = M.new(m, title: "public")
    {:ok, _} = M.ask(m, t, "hi")
    {:ok, tok} = M.share(m, t)
    assert {:ok, %{title: "public", messages: [_, _]}} = M.shared(m, t, tok)
    assert {:error, :forbidden} = M.shared(m, t, tok <> "x")
    :ok = M.revoke(m, t)
    assert {:error, :forbidden} = M.shared(m, t, tok)
    {:ok, tok2} = M.share(m, t)
    assert tok2 != tok and match?({:ok, _}, M.shared(m, t, tok2))
  end

  test "a reply lands under the question it answers, even if the head moved while the model was thinking", %{m: m} do
    {:ok, t} = M.new(m)
    {:ok, q} = M.say(m, t, "slow question")
    {:ok, job} = GenServer.call(m, {:prepare, t, %{}})
    {:ok, other} = M.say(m, t, "I kept typing")
    {:ok, %{node: a}} = GenServer.call(m, {:append_reply, t, job.parent, "late answer", %{}, nil})
    {:ok, n} = M.node(m, a)
    assert n.parent == q
    assert M.path(m, t) |> elem(1) |> Map.get(:head) == other
  end

  test "agent mode: tools run, the journal is stored, named by the answer and verifiable", %{dir: dir} do
    registry = Vapor.Majlis.Tools.custom([{"add", "pure", "add two integers", %{"type" => "object", "properties" => %{"a" => %{"type" => "integer"}, "b" => %{"type" => "integer"}}}, fn %{"a" => a, "b" => b} -> {:ok, a + b} end}])
    {:ok, m} = M.start_link(dir: dir <> "-agent", backends: %{"s" => %Vapor.AgentTest.Script{calls: [{"add", %{"a" => 20, "b" => 22}}]}}, tools: registry)
    {:ok, t} = M.new(m, tools: ["add"])
    {:ok, r} = M.ask(m, t, "what is 20 + 22?")
    assert r.content =~ "42"
    assert r.meta["agent"]["steps"] == [%{"tool" => "add", "ok" => true}]
    {:ok, j} = M.journal(m, r.meta["journal"])
    assert :ok = Vapor.Agent.Journal.verify(j)
    :ok = M.set(m, t, tools: ["nope"])
    assert {:error, msg} = M.ask(m, t, "again")
    assert msg =~ "no tool"
    File.rm_rf!(dir <> "-agent")
  end

  test "the code interpreter is Alembic in its sandbox: a real tool call through vapor's own registry", %{dir: dir} do
    {:ok, m} = M.start_link(dir: dir <> "-alb", backends: %{"s" => %Vapor.AgentTest.Script{calls: [{"alembic_eval", %{"expr" => "2^100 + 1"}}]}})
    {:ok, t} = M.new(m, tools: ["alembic_eval"])
    {:ok, r} = M.ask(m, t, "compute 2^100 + 1")
    assert r.content =~ "1267650600228229401496703205377"
    File.rm_rf!(dir <> "-alb")
  end

  test "garbage collection drops what no thread reaches and keeps what forks share", %{m: m} do
    {:ok, t} = M.new(m)
    {:ok, _} = M.ask(m, t, "shared past")
    {:ok, f} = M.fork(m, t)
    {:ok, _} = M.ask(m, t, "only in the trunk " <> String.duplicate("z", 5000))
    :ok = M.delete(m, t)
    {:ok, rep} = M.gc(m)
    assert rep.dropped > 0
    assert contents(m, f) |> length() == 2
    assert M.stats(m).messages == 2
  end

  test "concurrent replies on 16 threads: every answer under its own question", %{m: m} do
    tids = for i <- 1..16, do: elem(M.new(m, title: "t#{i}"), 1)

    tids
    |> Task.async_stream(fn t -> M.ask(m, t, "q #{t}") end, max_concurrency: 16)
    |> Enum.each(fn {:ok, {:ok, _}} -> :ok end)

    for t <- tids, do: assert(contents(m, t) == [{"user", "q #{t}"}, {"assistant", "echo: q #{t} (1 msgs)"}])
  end

  test "refusals: unknown threads, ambiguous or short ids, non-UTF-8, budget out of range, no model", %{m: m, dir: dir} do
    assert {:error, _} = M.say(m, "tnope", "x")
    {:ok, t} = M.new(m)
    assert {:error, msg} = M.edit(m, t, "ab", "x")
    assert msg =~ "6 hex"
    assert {:error, _} = M.say(m, t, <<0xFF, 0xFE>>)
    assert {:error, _} = M.set(m, t, budget: 10)
    assert {:error, msg} = M.reply(m, t)
    assert msg =~ "empty"
    {:ok, m2} = M.start_link(dir: dir <> "-none")
    {:ok, t2} = M.new(m2)
    {:ok, _} = M.say(m2, t2, "anyone?")
    assert {:error, msg} = M.reply(m2, t2)
    assert msg =~ "no model"
    File.rm_rf!(dir <> "-none")
  end

  defp msg(role, text, t), do: %{"author" => %{"role" => role}, "content" => %{"content_type" => "text", "parts" => [text]}, "create_time" => 1_700_000_000 + t}
end
