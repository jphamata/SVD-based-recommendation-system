defmodule Vapor.AuditTest do
  @moduledoc """
  Prohibitions enforced on the source text (checked, not advertised):
  no generated code ever runs in the BEAM, no external toolchain is invoked
  by the product, no dependency permeates the control plane, and the Lean
  development contains no axiom, `sorry`, or trusted escape hatch — and the
  extracted Elixir is provably fresh with respect to it.
  """
  use ExUnit.Case, async: true
  alias Vapor.Verify.Digest

  @lib Path.wildcard("lib/**/*.ex")
  @lean ~w(Extract.lean Main.lean Vapor.lean Vapor/BankConflict.lean Vapor/Binary32.lean Vapor/Estrin.lean
           Vapor/FieldEmbedding.lean Vapor/Higham.lean Vapor/IntegerParity.lean Vapor/RegAlloc.lean Vapor/Segmented.lean
           Vapor/Wilkinson.lean)

  test "no NIF, no shelling out: generated code can never execute inside the BEAM" do
    for path <- @lib, src = File.read!(path),
        re <- [~r/load_nif/, ~r/@on_load/, ~r/System\.cmd/, ~r/:os\.cmd/, ~r/\bclang\b|\bgcc\b|\bllc\b/] do
      refute src =~ re, "#{inspect(re)} in #{path}"
    end

    # OS processes: the two substrates that run generated code (isolated), the
    # MCP client, which runs an operator-named tool server, and the Athanor's
    # external measure, an operator-named command with a deadline — never
    # generated code
    ports = for p <- @lib, File.read!(p) =~ "Port.open", do: p
    assert Enum.sort(ports) == ["lib/vapor/agent/mcp.ex", "lib/vapor/main/measure.ex", "lib/vapor/runtime/fabric.ex", "lib/vapor/runtime/worker.ex"]
    mcp = File.read!("lib/vapor/agent/mcp.ex")
    assert [_] = Regex.scan(~r/Port\.open/, mcp)
    assert mcp =~ "Keyword.fetch!(opts, :cmd)"
    measure = File.read!("lib/vapor/main/measure.ex")
    assert [_] = Regex.scan(~r/Port\.open/, measure)
    assert measure =~ "deadline"
  end

  test "one entropy boundary (ASAS §6): the OS generator only through Vapor.Entropy; the process generator only seeded" do
    for path <- @lib, path != "lib/vapor/entropy.ex", src = File.read!(path) do
      refute src =~ ~r/strong_rand_bytes|:crypto\.rand_|:crypto\.strong_rand/, "#{path} draws OS entropy outside Vapor.Entropy"
      uses = src =~ ~r/\bEnum\.(random|shuffle|take_random)\(|:rand\.(uniform|bytes|normal)\(/
      if uses, do: assert(src =~ ~r/:rand\.seed\(/, "#{path} uses the process generator without seeding it")
    end
  end

  test "the control plane has no dependencies" do
    assert Mix.Project.config()[:deps] == []
  end

  test "the Lean development is axiom-free and escape-hatch-free" do
    for f <- @lean do
      src = File.read!(Path.join("proofs", f)) |> String.replace(~r/--.*$/m, "") |> String.replace(~r|/-.*?-/|s, "")

      for word <- ~w(axiom sorry admit native_decide implemented_by extern unsafe) do
        refute src =~ ~r/\b#{word}\b/, "#{word} in proofs/#{f}"
      end
    end

    lakefile = File.read!("proofs/lakefile.toml") |> String.replace(~r/#.*$/m, "")
    refute lakefile =~ ~r/require/i, "the proofs must build on core Lean alone"
  end

  test "the extracted module matches the Lean sources byte for byte (digest)" do
    h = Enum.reduce(@lean, 0xCBF29CE484222325, fn f, h -> Digest.fnv1a64(File.read!(Path.join("proofs", f)), h) end)
    assert Digest.hex64(h) == Vapor.Extracted.source_digest(),
           "lib/vapor/extracted.ex is stale: run `make extract`"

    assert File.read!("test/vapor/extracted_conformance_test.exs") =~ Vapor.Extracted.source_digest()
  end

  @tag :lean
  @tag timeout: 900_000
  test "lake build is warning-free and re-extraction is byte-identical" do
    {out, 0} = System.cmd("lake", ["build"], cd: "proofs", stderr_to_stdout: true)
    refute out =~ ~r/warning|error|sorry/i, out

    tmp = System.tmp_dir!()
    ex = Path.join(tmp, "vapor-extracted-#{System.unique_integer([:positive])}.ex")
    t = Path.join(tmp, "vapor-conf-#{System.unique_integer([:positive])}.exs")
    {_, 0} = System.cmd("lake", ["exe", "vapor-extract", ex, t], cd: "proofs", stderr_to_stdout: true)
    assert File.read!(ex) == File.read!("lib/vapor/extracted.ex")
    assert File.read!(t) == File.read!("test/vapor/extracted_conformance_test.exs")
  end
end
