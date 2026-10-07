defmodule Vapor.MajlisTest.Echo do
  @moduledoc false
  # answers with what it was shown: the last user turn and how many messages it saw
  @behaviour Vapor.Agent.Backend
  defstruct tag: "echo", sink: nil

  @impl true
  def complete(%__MODULE__{} = b, messages, _tools, _opts) do
    if b.sink, do: send(b.sink, {:seen, messages})
    last = messages |> Enum.filter(&(&1["role"] == "user")) |> List.last()
    {:ok, %{content: "#{b.tag}: #{last["content"]} (#{length(messages)} msgs)", calls: [], deterministic: true, record: %{"n" => length(messages)}}}
  end
end
