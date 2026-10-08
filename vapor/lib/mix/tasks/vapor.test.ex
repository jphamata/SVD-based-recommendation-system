defmodule Mix.Tasks.Vapor.Test do
  @shortdoc "Run only the test files whose inputs changed since they passed"
  @moduledoc """
  The test suite through a content-addressed cache (`Vapor.TestCache`): a
  test file that passed, with every module it can reach, the fixtures, the
  support files, `priv/` and the toolchain unchanged, is not run again.

      mix vapor.test                    # every test file, the unchanged ones skipped
      mix vapor.test test/vapor/kimi_test.exs test/vapor/lock_test.exs
      mix vapor.test --dry              # what would run, and why the rest would not
      mix vapor.test --all              # forget the cache (the files that pass are recorded again)

  Other options are passed to `mix test`. The release gate is `mix test`,
  which ignores the cache.
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, rest, _} = OptionParser.parse(args, strict: [all: :boolean, dry: :boolean])
    {files, passthrough} = Enum.split_with(rest, &String.ends_with?(&1, "_test.exs"))
    Mix.Task.run("compile")
    Mix.Task.run("loadpaths")
    :ok = Application.ensure_loaded(:vapor)
    files = if files == [], do: Path.wildcard("test/**/*_test.exs") |> Enum.sort(), else: files
    excluded = Vapor.TestTiers.excluded()
    keys = Vapor.TestCache.keys(files, File.cwd!(), excluded)
    cache = if opts[:all], do: %{}, else: Vapor.TestCache.load()
    {cached, todo} = Enum.split_with(files, &Map.has_key?(cache, keys[&1]))

    Mix.shell().info("vapor.test: #{length(cached)} of #{length(files)} test files unchanged since they passed; running #{length(todo)}")
    if opts[:dry], do: Enum.each(todo, &Mix.shell().info("  run  " <> &1))

    cond do
      opts[:dry] == true or todo == [] -> :ok
      true ->
        Application.put_env(:vapor, :test_cache_keys, keys)
        Mix.Task.run("test", todo ++ passthrough ++ ["--formatter", "ExUnit.CLIFormatter", "--formatter", "Vapor.TestCache.Formatter"])
    end
  end
end
