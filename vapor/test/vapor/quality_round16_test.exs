defmodule Vapor.QualityRound16Test do
  @moduledoc "The 0.16 quality round runs in seconds, so the suite runs it on every test pass: each value beats its control."
  use ExUnit.Case, async: false

  @tag timeout: 300_000
  test "every check of round 0.16 passes, and each carries a control" do
    %{checks: checks} = Vapor.Quality.Round16.run()
    assert length(checks) == 11
    for c <- checks, do: assert(c.pass and c.control not in [nil, "", "—"], inspect(c))
  end
end
