defmodule Vapor.QualityRound13Test do
  @moduledoc "The 0.13 quality round (Vapor.Quality.Round13, report §5i): every check passes, and every control is a value that differs from the check's."
  use ExUnit.Case, async: true
  @moduletag timeout: 600_000

  test "every 0.13 check passes against its control" do
    w = if Vapor.Runtime.Substrates.binary("vapor-worker", "native"), do: Vapor.Modal.Runner.worker()
    %{checks: cs} = Vapor.Quality.Round13.run(worker: w)
    assert length(cs) >= 21
    failed = Enum.reject(cs, & &1.pass)
    assert failed == [], inspect(failed, pretty: true)
    # a control is never the value itself: the check could have failed
    assert Enum.all?(cs, &(&1.control == "—" or &1.control != &1.value))
  end
end
