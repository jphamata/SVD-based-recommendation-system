defmodule Vapor.QualityRound15Test do
  @moduledoc "The 0.15 quality round runs in seconds, so the suite runs it on every test pass: each value beats its control."
  use ExUnit.Case, async: false

  @tag timeout: 300_000
  test "every check of round 0.15 passes, and each carries a control" do
    %{checks: checks} = Vapor.Quality.Round15.run()
    assert length(checks) == 12
    for c <- checks, do: assert(c.pass and c.control not in [nil, "", "—"], inspect(c))
  end
end
