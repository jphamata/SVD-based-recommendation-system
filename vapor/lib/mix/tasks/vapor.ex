defmodule Mix.Tasks.Vapor do
  @shortdoc "The vapor command line (alembic, athanor, game, crucible, assay, mind, scene, verify, solve)"
  @moduledoc """
      mix vapor <command> [args…]

  The same commands as `bin/vapor` (docs/CLI.md), through Mix. `bin/vapor`
  starts faster (no Mix) and is what pipelines should call.
  """
  use Mix.Task

  @impl true
  def run(argv) do
    Mix.Task.run("app.config")
    code = Vapor.Main.run(argv)
    if code != 0, do: exit({:shutdown, code})
  end
end
