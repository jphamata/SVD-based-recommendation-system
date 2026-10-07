defmodule Vapor.Mind.Script do
  @moduledoc "A canned model: answers in order (the last repeats), or computed from the prompt by a function."
  @behaviour Vapor.Agent.Backend
  defstruct answers: [], agent: nil, fun: nil

  @impl true
  def complete(%__MODULE__{fun: f}, messages, _tools, _opts) when is_function(f, 1) do
    user = messages |> Enum.filter(&(&1["role"] == "user")) |> Enum.map_join("\n", & &1["content"])
    {:ok, %{content: f.(user), calls: [], deterministic: true, record: %{"provider" => "script"}}}
  end

  def complete(%__MODULE__{answers: as, agent: a}, _messages, _tools, _opts) do
    i = Agent.get_and_update(a, &{&1, &1 + 1})
    text = Enum.at(as, min(i, length(as) - 1)) || ""
    {:ok, %{content: text, calls: [], deterministic: true, record: %{"provider" => "script", "index" => i}}}
  end
end
