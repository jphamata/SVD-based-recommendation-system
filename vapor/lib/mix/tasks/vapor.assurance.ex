defmodule Mix.Tasks.Vapor.Assurance do
  @shortdoc "Write docs/ASSURANCE.md from the assurance ledger (Vapor.Assurance)"
  @moduledoc """
      mix vapor.assurance            # writes docs/ASSURANCE.md; fails if cited evidence is missing
      mix vapor.assurance --check    # writes nothing; fails if evidence is missing or the document drifted
  """
  use Mix.Task

  @impl true
  def run(argv) do
    Mix.Task.run("compile")

    case Vapor.Assurance.missing() do
      [] -> :ok
      gone -> Mix.raise("evidence missing: " <> Enum.map_join(gone, "; ", fn {c, p} -> "#{c}: #{p}" end))
    end

    if "--check" in argv do
      if File.read("docs/ASSURANCE.md") == {:ok, Vapor.Assurance.markdown()},
        do: Mix.shell().info("docs/ASSURANCE.md agrees with the ledger"),
        else: Mix.raise("docs/ASSURANCE.md differs from the ledger: run mix vapor.assurance")
    else
      File.write!("docs/ASSURANCE.md", Vapor.Assurance.markdown())
      Mix.shell().info("docs/ASSURANCE.md")
    end
  end
end
