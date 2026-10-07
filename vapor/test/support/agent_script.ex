defmodule Vapor.AgentTest.Script do
  @moduledoc false
  # A deterministic backend that decides from the conversation alone: the
  # first turn calls tools, the next answers from their results. Being
  # deterministic, replay recomputes its decisions like a local model's.
  @behaviour Vapor.Agent.Backend
  defstruct calls: [], answer: nil

  @impl true
  def complete(%__MODULE__{} = s, messages, _tools, opts) do
    results = for %{"role" => "tool", "content" => c} <- messages, do: c

    if results == [] do
      calls = s.calls |> Enum.with_index() |> Enum.map(fn {{n, a}, i} -> %{"id" => "c#{i}", "name" => n, "arguments" => a} end)
      {:ok, %{content: "", calls: calls, deterministic: true, record: %{"seed" => opts["seed"]}}}
    else
      {:ok, %{content: (s.answer || "done: ") <> Enum.join(results, " | "), calls: [], deterministic: true, record: %{"seed" => opts["seed"]}}}
    end
  end
end
