defmodule Vapor.Main.MeasureTest do
  use ExUnit.Case, async: true
  alias Vapor.Main.Measure

  test "the candidate arrives in the environment and the output comes back" do
    assert {:ok, "7\n"} = Measure.run(~s|echo "$VAPOR_CANDIDATE"|, [{"VAPOR_CANDIDATE", "7"}])
  end

  test "a non-zero exit is an error, not a measurement" do
    assert {:error, msg} = Measure.run("exit 3", [])
    assert msg =~ "3"
  end

  test "a command that hangs is killed at the deadline instead of freezing the furnace" do
    {t, r} = :timer.tc(fn -> Measure.run("sleep 30; echo 1", [], timeout: 1_000) end)
    assert {:error, _} = r
    assert t < 6_000_000
  end

  test "the Athanor's shell measure reads the last number the command prints" do
    m = Vapor.Main.AthanorCli.shell_measure(~s|echo "score: 1.5e2"|)
    assert {:ok, 150.0} = m.(3)
  end
end
