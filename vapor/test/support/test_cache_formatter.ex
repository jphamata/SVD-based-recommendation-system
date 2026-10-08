defmodule Vapor.TestCache.Formatter do
  @moduledoc false
  # `mix vapor.test`'s ExUnit formatter: a test file whose every test passed
  # (excluded tiers are part of its key) is recorded under the key the task
  # computed before the run (Vapor.TestCache).
  use GenServer

  @impl true
  def init(_opts), do: {:ok, %{}}

  @impl true
  def handle_cast({:test_finished, %ExUnit.Test{tags: %{file: file}, state: state}}, st) do
    ok = match?(nil, state) or match?({:excluded, _}, state) or match?({:skipped, _}, state)
    {:noreply, Map.update(st, rel(file), {ok, 1}, fn {o, n} -> {o and ok, n + 1} end)}
  end

  def handle_cast({:module_finished, %ExUnit.TestModule{file: file, state: {:failed, _}}}, st),
    do: {:noreply, Map.update(st, rel(file), {false, 0}, fn {_, n} -> {false, n} end)}

  def handle_cast({:suite_finished, _}, st) do
    keys = Application.get_env(:vapor, :test_cache_keys, %{})
    Vapor.TestCache.record(for {file, {true, n}} <- st, key = keys[file], do: {file, key, n})
    {:noreply, st}
  end

  def handle_cast(_, st), do: {:noreply, st}

  defp rel(file), do: Path.relative_to(file, File.cwd!())
end
