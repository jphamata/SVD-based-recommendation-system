defmodule Vapor.QualityRound12Test do
  @moduledoc "The 0.12 quality round (Vapor.Quality.Round12, report §5h): every check passes, and every control is a value that differs from the check's."
  use ExUnit.Case, async: true
  @moduletag timeout: 600_000

  test "every 0.12 check passes against its control" do
    %{checks: cs} = Vapor.Quality.Round12.run()
    assert length(cs) >= 25
    failed = Enum.reject(cs, & &1.pass)
    assert failed == [], inspect(failed, pretty: true)
    # a control is never the value itself: the check could have failed
    assert Enum.all?(cs, &(&1.control == "—" or &1.control != &1.value))
  end
end
