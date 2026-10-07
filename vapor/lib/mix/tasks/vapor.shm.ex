defmodule Mix.Tasks.Vapor.Shm do
  @shortdoc "Prune unmapped weight files from shared memory (/dev/shm/vapor-*)"
  @moduledoc """
      mix vapor.shm

  Removes the content-addressed weight files that no process maps
  (`Vapor.Runtime.Shm.prune/0`) and reports what was freed.
  """
  use Mix.Task

  @impl true
  def run(_argv) do
    %{removed: n, kept: k, bytes: b} = Vapor.Runtime.Shm.prune()
    Mix.shell().info("removed #{n} files (#{Float.round(b / 1.0e6, 1)} MB), #{k} in use")
  end
end
