defmodule Vapor.TestCacheTest do
  @moduledoc """
  `mix vapor.test`'s cache keys (`Vapor.TestCache`): a test file's key
  covers the modules it can reach and nothing else, so an edit re-runs the
  files that could see it, and only those; anything shared (fixtures,
  priv/, the toolchain) re-runs everything.
  """
  use ExUnit.Case, async: true
  alias Vapor.TestCache

  test "the modules a file names: full names, grouped aliases, and their parents" do
    mods = TestCache.named_modules("alias Vapor.{Lock, Siphon}\nVapor.Assay.Detect.run(x)")
    assert Vapor.Lock in mods and Vapor.Siphon in mods and Vapor.Assay.Detect in mods and Vapor.Assay in mods
  end

  test "reach is selective: the detection metric reaches the simplex, not the network airlock" do
    det = TestCache.reach("Vapor.Assay.Detect.run(text)")
    assert Vapor.Logic.LP in det
    refute Vapor.Siphon in det
    sip = TestCache.reach("Vapor.Siphon.run(f, ref)")
    assert Vapor.Ingest.Safetensors in sip
    refute Vapor.Assay.Detect in sip
  end

  test "a file reaches what the test helpers it uses reach (tiny_weights builds through the decoder)" do
    assert Vapor.Model.Decoder in TestCache.reach("import Vapor.TestHelpers\nx = tiny_weights(c)")
  end

  test "keys: a file's own text moves only its key; a shared input moves every key" do
    root = Path.join(System.tmp_dir!(), "vapor-tc-#{System.unique_integer([:positive])}")
    File.mkdir_p!(Path.join(root, "priv"))
    File.write!(Path.join(root, "a_test.exs"), "Vapor.Assay.Detect.run(x)")
    File.write!(Path.join(root, "b_test.exs"), "Vapor.Siphon.run(f, r)")
    File.write!(Path.join(root, "priv/data"), "1")

    try do
      k1 = TestCache.keys(["a_test.exs", "b_test.exs"], root, [:torch])
      File.write!(Path.join(root, "a_test.exs"), "Vapor.Assay.Detect.run(y)")
      k2 = TestCache.keys(["a_test.exs", "b_test.exs"], root, [:torch])
      assert k1["a_test.exs"] != k2["a_test.exs"] and k1["b_test.exs"] == k2["b_test.exs"]
      File.write!(Path.join(root, "priv/data"), "2")
      k3 = TestCache.keys(["a_test.exs", "b_test.exs"], root, [:torch])
      assert k3["a_test.exs"] != k2["a_test.exs"] and k3["b_test.exs"] != k2["b_test.exs"]
      # tooling that appears (a tier no longer excluded) changes what runs, so it moves the keys
      refute TestCache.keys(["b_test.exs"], root, [])["b_test.exs"] == k3["b_test.exs"]
    after
      File.rm_rf!(root)
    end
  end

  test "a file that reads repository files at run time is keyed on them: docs for the docs test, sources for the audit" do
    assert TestCache.reads(~s|Path.wildcard(Path.join(@root, "docs/*.md")) ++ [Path.join(@root, "README.md")]|) == ["docs", "README.md"]
    assert TestCache.reads(~s|@lib Path.wildcard("lib/**/*.ex")|) == ["lib"]
    assert TestCache.reads(~s|Path.expand("../../notebooks/tour.livemd", __DIR__)|) == ["notebooks"]
    assert TestCache.reads("Vapor.Assay.Detect.run(x)") == []

    root = Path.join(System.tmp_dir!(), "vapor-tc-#{System.unique_integer([:positive])}")
    File.mkdir_p!(Path.join(root, "docs"))
    File.write!(Path.join(root, "d_test.exs"), ~s|File.read!("docs/A.md")|)
    File.write!(Path.join(root, "e_test.exs"), "Vapor.Siphon.run(f, r)")
    File.write!(Path.join(root, "docs/A.md"), "one")

    try do
      k1 = TestCache.keys(["d_test.exs", "e_test.exs"], root, [])
      File.write!(Path.join(root, "docs/A.md"), "two")
      k2 = TestCache.keys(["d_test.exs", "e_test.exs"], root, [])
      assert k1["d_test.exs"] != k2["d_test.exs"] and k1["e_test.exs"] == k2["e_test.exs"]
    after
      File.rm_rf!(root)
    end
  end

end
