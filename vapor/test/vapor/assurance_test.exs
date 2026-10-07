defmodule Vapor.AssuranceTest do
  use ExUnit.Case, async: true

  test "every piece of evidence the ledger cites exists, and the document is the ledger's rendering" do
    assert Vapor.Assurance.missing() == []
    assert File.read!("docs/GARANTIAS.md") == Vapor.Assurance.markdown(), "run mix vapor.assurance"
  end

  test "every proved claim cites a Lean file; every owed claim says what is missing" do
    for c <- Vapor.Assurance.claims() do
      if c.status == :proved, do: assert(Enum.any?(c.evidence, &String.ends_with?(&1, ".lean")), c.claim)
      if c.status == :owed, do: assert(c.limits != nil or c.basis != nil, c.claim)
    end
  end
end
