defmodule Vapor.NotebookTest do
  @moduledoc """
  The Livebook tour (`notebooks/vapor_tour.livemd`) is executable
  documentation: its cells run here in order, one binding threaded through
  them as Livebook does (the `Mix.install` cell aside), and its claims are
  re-checked on the values the cells return.
  """
  use ExUnit.Case, async: false

  @moduletag :native
  @moduletag timeout: 900_000

  test "every cell of the tour runs, and says what the text says" do
    path = Path.expand("../../notebooks/vapor_tour.livemd", __DIR__)

    cells =
      ~r/^```elixir\n(.*?)^```$/ms
      |> Regex.scan(File.read!(path), capture: :all_but_first)
      |> Enum.map(&hd/1)
      |> Enum.reject(&String.contains?(&1, "Mix.install("))

    {values, _} =
      Enum.map_reduce(cells, {[], %{__ENV__ | file: path}}, fn code, {binding, env} ->
        {value, binding, env} = Code.eval_quoted_with_env(Code.string_to_quoted!(code, file: path), binding, env)
        {value, {binding, env}}
      end)

    [certified, _run, _info, {invariant?, _}, json, rag, _spec, resumed, replay] = values
    assert certified.substrates_bit_identical == :all_outputs
    assert invariant?
    assert Enum.all?(json, fn {check, _text} -> check == :ok end)
    assert rag.reverify == :ok and rag.documents == ["ai-act"]
    assert resumed.notifications == 1 and resumed.answer == "done: 42 | \"sent\""
    assert replay.replay.mismatches == [] and replay.notifications_after == 1
    assert replay.tampered == {:error, {:broken_at, 2}}
  end
end
