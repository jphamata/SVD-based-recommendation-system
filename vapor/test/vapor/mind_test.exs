defmodule Vapor.MindTest do
  use ExUnit.Case, async: true
  alias Vapor.{Athanor, Mind}

  @moduletag timeout: 300_000

  test "formalize: a broken draft goes back to the model with its error; the fixed one loads and is read back" do
    broken = "```alembic\nspace = subset(1..20, 4\nminimize(r) = max(r)\n```"
    fixed = "```alembic\nspace = subset(1..20, 4)\nruler(r) = [0] ++ r\nviolation(r) = let d = [b - a for (a, b) in pairs(ruler(r))] in len(d) - len(distinct(d))\nminimize(r) = max(r)\n```"
    m = Mind.script([broken, fixed, "A ruler with marks at 0 and four of 1..20, all distances distinct, shortest."])
    assert {:ok, r} = Mind.formalize(m, "the shortest Golomb ruler with 5 marks")
    assert r.attempts == 2 and r.kind == :search
    assert r.back_translation =~ "ruler"
    assert [%{ok: false, error: e}, %{ok: true}] = r.log
    assert e =~ "unexpected"
    assert length(Mind.transcript(m)) == 3
  end

  test "formalize gives up after three failed drafts and says so" do
    m = Mind.script(["space = nonsense(", "still (", "no"])
    assert {:error, why, log} = Mind.formalize(m, "anything")
    assert why =~ "3 attempts" and length(log) == 3
  end

  test "a model as one more proposer: its candidates are parsed, checked and scored — nonsense is dropped" do
    m = Mind.script(fn _prompt -> "1. [37, 11]\n- [99, 99]\nnot a candidate\n[36, 12]" end)
    {:ok, c} = Athanor.run("space = ints(2, 0, 50)\nminimize(v) = (v[0] - 37)^2 + (v[1] - 11)^2\nbudget = 300", mind: m, seed: 3)
    assert c.best.value == 0
    assert Map.has_key?(c.strategies, "mind")
    assert c.strategies["mind"].evals >= 1
  end

  test "a model spec that cannot be understood is refused" do
    assert {:error, _} = Mind.parse("telepathy:v1")
    assert {:ok, %Mind{name: "anthropic:x"}} = Mind.parse("anthropic:x")
  end
end
