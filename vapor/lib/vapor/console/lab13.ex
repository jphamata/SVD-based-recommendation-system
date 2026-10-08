defmodule Vapor.Console.Lab13 do
  @moduledoc """
  The console's desks for the 0.13 round (docs/CONSOLE.md, docs/FINANCE.md):
  finance (calendars and money, curves, options, Monte Carlo on the native
  worker, risk, portfolios, backtests with noise gates, arbitrage) and the
  trading desk (an order book with its journal, an exchange session,
  microstructure). Each request is bounded and timed before it runs.
  """
  alias Vapor.Console.Lab12

  @max_text 200_000

  def finance(req) do
    kind = req["kind"]
    text = req["text"] || ""
    cond do
      kind not in Vapor.Finance.kinds() -> {:error, "kind: #{Enum.join(Vapor.Finance.kinds(), ", ")}"}
      not is_binary(text) or byte_size(text) > @max_text -> {:error, "text: at most 200 kB"}
      true ->
        t0 = System.monotonic_time(:millisecond)
        case timed(fn -> Vapor.Finance.run(kind, text) end) do
          {:ok, r} -> {:ok, r |> Lab12.slim() |> Map.put(:ms, System.monotonic_time(:millisecond) - t0) |> Map.put(:kind, kind)}
          {:error, w} -> {:error, if(is_binary(w), do: w, else: inspect(w))}
        end
    end
  end

  defp timed(fun, ms \\ 120_000) do
    task = Task.async(fn -> try do fun.() rescue e -> {:error, Exception.message(e)} catch kind, why -> {:error, "#{kind}: #{inspect(why)}"} end end)
    case Task.yield(task, ms) || Task.shutdown(task, :brutal_kill) do
      {:ok, r} -> r
      nil -> {:error, "the computation took longer than #{div(ms, 1000)} s and was stopped (make the problem smaller)"}
    end
  end
end
