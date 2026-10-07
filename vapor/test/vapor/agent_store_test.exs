defmodule Vapor.AgentStoreTest do
  @moduledoc """
  Durable runs (`Vapor.Agent.Store`): events are persisted write-ahead
  through `on_event:`, a crashed run resumes from the store without
  repeating what was recorded, an unrecorded action is retried with the
  same idempotency key, two resumers cannot both write a step, and a
  half-written event is never read back as one.
  """
  use ExUnit.Case, async: false
  alias Vapor.Agent
  alias Vapor.Agent.{Journal, Spec, Store}
  alias Vapor.AgentTest.Script

  @now ~U[2026-10-01 15:00:00Z]

  defp spec do
    Spec.new(name: "assistant", instructions: "Be precise.", model: %{"kind" => "local", "id" => "script"},
             tools: [%{name: "add", effect: "pure", parameters: %{"type" => "object", "properties" => %{"a" => %{"type" => "integer"}, "b" => %{"type" => "integer"}}, "required" => ["a", "b"]}},
                     %{name: "weather", effect: "observe", parameters: %{"type" => "object", "properties" => %{"city" => %{"type" => "string"}}, "required" => ["city"]}},
                     %{name: "send_report", effect: "act", parameters: %{"type" => "object", "properties" => %{"to" => %{"type" => "string"}}, "required" => ["to"]}}],
             grants: ["send_report"])
  end

  defp script, do: %Script{calls: [{"add", %{"a" => 2, "b" => 40}}, {"weather", %{"city" => "Recife"}}, {"send_report", %{"to" => "ops"}}]}

  defp impls(counter) do
    %{"add" => fn %{"a" => a, "b" => b}, _ -> {:ok, a + b} end,
      "weather" => fn %{"city" => c}, _ -> {:ok, %{"city" => c, "temp" => 21.5}} end,
      "send_report" => fn %{"to" => to}, ctx -> Elixir.Agent.update(counter, &[{to, ctx.idempotency_key} | &1]); {:ok, "sent"} end}
  end

  setup do
    dir = Path.join(System.tmp_dir!(), "vapor-store-#{System.unique_integer([:positive])}")
    on_exit(fn -> File.rm_rf!(dir) end)
    {:ok, counter} = Elixir.Agent.start_link(fn -> [] end)
    {:ok, store: Store.File.new(dir), counter: counter}
  end

  # the reference: the same run, uninterrupted, in memory
  defp reference(nonce) do
    {:ok, c} = Elixir.Agent.start_link(fn -> [] end)
    {:ok, run} = Agent.run(spec(), "go", backend: script(), impls: impls(c), now: @now, nonce: nonce)
    run
  end

  test "every event is persisted as it happens; the store gives back the journal", %{store: st, counter: c} do
    me = self()
    ui = fn e, _j -> send(me, {:event, e["seq"], e["kind"]}); :ok end

    {:ok, run} = Store.run(st, spec(), "go", backend: script(), impls: impls(c), now: @now, nonce: "s1", on_event: ui)
    assert run.journal.head == reference("s1").journal.head

    {:ok, loaded} = Store.load(st, run.run_id)
    assert loaded == run.journal and Journal.verify(loaded) == :ok
    assert Store.runs(st) == [run.run_id] and Store.unfinished(st) == []

    # the UI saw every event, in order (a LiveView would get these over PubSub)
    seen = for _ <- run.journal.events, do: (receive do {:event, s, k} -> {s, k} after 1000 -> nil end)
    assert seen == Enum.map(run.journal.events, &{&1["seq"], &1["kind"]})

    # verifying emits nothing: no event of a replay is new
    assert {:ok, %{mismatches: []}} = Agent.replay(spec(), loaded, backend: script(), impls: impls(c), on_event: ui)
    refute_received {:event, _, _}

    # a finished run resumes to its result without acting
    {:ok, again} = Store.resume(st, spec(), run.run_id, backend: script(), impls: impls(c))
    assert again.answer == run.answer and length(Elixir.Agent.get(c, & &1)) == 1
  end

  test "a crash before the action: resume replays the prefix and acts once", %{store: st, counter: c} do
    # the store stops accepting writes after the `weather` event (seq 3)
    crash = fn e, _j -> if e["seq"] >= 3, do: {:error, :disk_gone}, else: :ok end

    assert {:error, {:on_event, :disk_gone}, _} =
             Agent.run(spec(), "go", backend: script(), impls: impls(c), now: @now, nonce: "s2", on_event: [crash, Store.hook(st)])

    [id] = Store.unfinished(st)
    assert {:ok, %{events: evs}} = Store.load(st, id)
    assert Enum.map(evs, & &1["kind"]) == ["start", "model", "tool"]
    assert Elixir.Agent.get(c, & &1) == []

    {:ok, done} = Store.resume(st, spec(), id, backend: script(), impls: impls(c))
    assert done.status == "final" and done.journal.head == reference("s2").journal.head
    assert [{"ops", _}] = Elixir.Agent.get(c, & &1)
    assert Store.unfinished(st) == [] and Store.load(st, id) == {:ok, done.journal}
  end

  test "a crash after the action, before its result was stored: retried, recorded as such, same idempotency key", %{store: st, counter: c} do
    # seq 4 is the intent (stored), seq 5 the action's result (lost)
    lost = fn e, _j -> if e["seq"] == 5, do: {:error, :network}, else: :ok end

    assert {:error, {:on_event, :network}, _} =
             Agent.run(spec(), "go", backend: script(), impls: impls(c), now: @now, nonce: "s3", on_event: [lost, Store.hook(st)])

    [id] = Store.unfinished(st)
    {:ok, stale} = Store.load(st, id)
    assert List.last(stale.events)["kind"] == "intent"
    {:ok, done} = Store.resume(st, spec(), id, backend: script(), impls: impls(c))
    assert done.answer == reference("s3").answer and "retry" in Enum.map(done.journal.events, & &1["kind"])
    assert {:ok, %{mismatches: []}} = Agent.replay(spec(), done.journal, backend: script(), impls: impls(c))

    # executed twice by the agent, with one key: a receiver that honours it applies it once
    assert [{"ops", k}, {"ops", k}] = Elixir.Agent.get(c, & &1)
    assert Enum.find(done.journal.events, &(&1["kind"] == "tool" and &1["data"]["name"] == "send_report"))["data"]["idempotency_key"] == k

    # a second resumer holding the same stale prefix collides on the retry: it does not act
    assert {:error, {:on_event, :conflict}, _} = Store.run(st, spec(), nil, backend: script(), impls: impls(c), journal: stale)
    assert length(Elixir.Agent.get(c, & &1)) == 2
  end

  test "two resumers of a run that crashed before its action: exactly one acts", %{store: st, counter: c} do
    crash = fn e, _j -> if e["seq"] >= 4, do: {:error, :killed}, else: :ok end
    {:error, _, _} = Agent.run(spec(), "go", backend: script(), impls: impls(c), now: @now, nonce: "s6", on_event: [crash, Store.hook(st)])
    [id] = Store.unfinished(st)
    {:ok, prefix} = Store.load(st, id)

    {:ok, _} = Store.run(st, spec(), nil, backend: script(), impls: impls(c), journal: prefix)
    # the second writes its intent at the same seq: refused, before the action
    assert {:error, {:on_event, :conflict}, _} = Store.run(st, spec(), nil, backend: script(), impls: impls(c), journal: prefix)
    assert length(Elixir.Agent.get(c, & &1)) == 1
  end

  test "two writers of one step: the second is refused before it acts", %{store: st, counter: c} do
    {:ok, run} = Store.run(st, spec(), "go", backend: script(), impls: impls(c), now: @now, nonce: "s4")
    ev = Enum.at(run.journal.events, 2)
    assert Store.append(st, run.run_id, ev) == {:error, :conflict}

    # a stale resumer holding only the first two events: its next write (the
    # first tool event, seq 2) collides with the one already stored, so it
    # stops after computing that tool — before the action at seq 4
    stale = %{run.journal | events: Enum.take(run.journal.events, 2)}
    stale = %{stale | head: List.last(stale.events)["hash"]}
    assert {:error, {:on_event, :conflict}, _} = Store.run(st, spec(), nil, backend: script(), impls: impls(c), journal: stale)
    assert length(Elixir.Agent.get(c, & &1)) == 1
  end

  test "a half-written or forged event is not an event; run ids cannot escape the directory", %{store: st, counter: c} do
    crash = fn e, _j -> if e["seq"] >= 2, do: {:error, :killed}, else: :ok end
    {:error, _, _} = Agent.run(spec(), "go", backend: script(), impls: impls(c), now: @now, nonce: "s5", on_event: [crash, Store.hook(st)])
    [id] = Store.unfinished(st)
    rdir = Path.join(st.dir, id)

    # leftovers of a crash mid-write (temporary files) and a forged next event
    File.write!(Path.join(rdir, ".000000002.cbor.99.tmp"), "partial")
    File.write!(Path.join(rdir, "000000002.cbor"), Vapor.Canonical.encode(%{"seq" => 2, "kind" => "tool", "data" => %{}, "hash" => String.duplicate("0", 64)}))
    assert {:ok, %{events: [_, _]}} = Store.load(st, id)

    assert_raise ArgumentError, fn -> Store.load(st, "../../etc") end
    assert Store.load(st, String.duplicate("ab", 32)) == {:error, :not_found}

    # a run whose start never reached the disk did not begin
    File.mkdir_p!(Path.join(st.dir, String.duplicate("cd", 32)))
    assert Store.unfinished(st) == [id]
  end
end
