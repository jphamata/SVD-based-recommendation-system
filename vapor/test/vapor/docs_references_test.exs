defmodule Vapor.DocsReferencesTest do
  @moduledoc """
  Documentation that lies does not pass: every `Vapor.…` module and function
  the docs and the README name exists (a module, a namespace of modules, a
  module of the integrations, or a function with that name and arity), and
  the Almizan pair that docs/ALMIZAN.md shows as one tree in two scripts
  has one hash. Found by the first run: `Vapor.Alembic.sandbox` (replaced by
  `Vapor.Hermetic.seal` this round) and a misspelled JBIG2 module. Every
  repository path the docs name in code font (`lib/…`, `test/…`, `priv/…`)
  exists too; found by its first run: `priv/games`, gone since 0.16.
  DIRECTIVE.md is the record of past rounds and keeps the paths they had.
  """
  use ExUnit.Case, async: true

  @root Path.expand("../..", __DIR__)

  # named in the docs as retired, refused or renamed — on purpose
  @not_modules ~w(Vapor.Graph Vapor.Discover Vapor.Games Vapor.Silicon Vapor.Scene Vapor.Alembic.Tree Vapor.Model.Llama)

  defp known do
    {:ok, mods} = :application.get_key(:vapor, :modules)
    Enum.each(mods, &Code.ensure_loaded/1)

    integrations =
      for f <- Path.wildcard(Path.join(@root, "integrations/*/lib/**/*.ex")), text = File.read!(f),
          [_, m] <- Regex.scan(~r/defmodule\s+(Vapor(?:\.[A-Z][A-Za-z0-9_]*)+)/, text), into: %{},
          do: {m, for([_, fun] <- Regex.scan(~r/\bdef\s+([a-z_][a-z0-9_]*[?!]?)/, text), uniq: true, do: fun)}

    app = Map.new(mods, fn m -> {inspect(m), for({f, _} <- m.__info__(:functions) ++ m.__info__(:macros), uniq: true, do: Atom.to_string(f))} end)
    {Map.merge(app, integrations), mods}
  end

  test "every Vapor module and function the docs name exists" do
    {known, mods} = known()
    names = Map.keys(known)

    bad =
      for f <- Path.wildcard(Path.join(@root, "docs/*.md")) ++ [Path.join(@root, "README.md")], text = File.read!(f),
          [ref, mod, fun] <- Regex.scan(~r/\b(Vapor(?:\.[A-Z][A-Za-z0-9_]*)+)(?:\.([a-z_][a-z0-9_]*[?!]?)(?:\/\d+)?)?/, text, capture: :all) |> Enum.map(&pad/1),
          not ok?(mod, fun, known, names, mods), uniq: true, do: "#{Path.basename(f)}: #{ref}"

    assert bad == []
  end

  defp pad([r, m]), do: [r, m, nil]
  defp pad([r, m, f]), do: [r, m, if(f == "", do: nil, else: f)]

  defp ok?(mod, fun, known, names, _mods) do
    cond do
      Enum.any?(@not_modules, &(mod == &1 or String.starts_with?(mod, &1 <> "."))) -> true
      Map.has_key?(known, mod) -> fun == nil or fun in known[mod]
      # a namespace: some module lives under it
      Enum.any?(names, &String.starts_with?(&1, mod <> ".")) -> fun == nil
      true -> false
    end
  end

  test "every repository path the docs name exists" do
    bad =
      for f <- Path.wildcard(Path.join(@root, "docs/*.md")) ++ [Path.join(@root, "README.md")], Path.basename(f) != "DIRECTIVE.md",
          [_, p] <- Regex.scan(~r/`((?:lib|test|priv|native|scripts)\/[A-Za-z0-9_.\/-]+)`/, File.read!(f)),
          p = String.trim_trailing(p, "."), not String.contains?(p, "*"), not File.exists?(Path.join(@root, p)), uniq: true,
          do: "#{Path.basename(f)}: #{p}"

    assert bad == []
  end

  test "docs/ALMIZAN.md's two scripts are one tree: the same hash" do
    blocks = Regex.scan(~r/```lisp\n(.*?)```/s, File.read!(Path.join(@root, "docs/ALMIZAN.md")), capture: :all_but_first) |> Enum.map(&hd/1)
    assert [latin, arabic] = blocks
    assert arabic =~ ~r/\p{Arabic}/u
    # the excerpt calls `kinetic`, defined above it in the full file: the same definition, in each script
    kinetic = "(claim kinetic (root H-s-b) (wazn fail) (inputs (v q)) (body (* 1/2 v v)))\n"
    {:ok, kinetic_ar} = Vapor.Almizan.Format.format(kinetic, :arabic)
    {:ok, a} = Vapor.Almizan.parse(kinetic <> latin)
    {:ok, b} = Vapor.Almizan.parse(kinetic_ar <> arabic)
    assert Vapor.Almizan.hash(a) == Vapor.Almizan.hash(b)
  end
end
