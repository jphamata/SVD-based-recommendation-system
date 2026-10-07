defmodule Vapor.Main.AssayCli do
  @moduledoc false
  # vapor assay — the AI research suite from the terminal
  import Vapor.Main
  alias Vapor.Assay

  def run(argv) do
    case opts(argv, [example: :boolean, seed: :integer, threshold: :float, keep: :boolean, ngram: :integer, reps: :integer]) do
      :usage -> 2
      {:ok, _o, []} ->
        out(bold("vapor assay TOOL [FILE|-]") <> dim("   (--example prints a starting point)"))
        Enum.each(Assay.tools(), fn t -> out("  #{String.pad_trailing(t.tool, 14)} #{t.about}") end)
        0
      {:ok, o, [tool | rest]} ->
        cond do
          Assay.example(tool) == nil -> err("assay: unknown tool #{tool} (vapor assay lists them)"); 2
          o[:example] -> out(Assay.example(tool)); 0
          true ->
            with {:ok, text} <- input(tool, rest) do
              opts = Keyword.take(o, [:seed, :threshold, :ngram, :reps])
              case Assay.run(tool, text, opts) do
                {:ok, r} ->
                  cond do
                    tool == "dedup" and o[:keep] ->
                      docs = Vapor.Assay.Data.docs(text) |> List.to_tuple()
                      lines = text |> String.split("\n") |> Enum.reject(&(String.trim(&1) == "")) |> List.to_tuple()
                      _ = docs
                      Enum.each(r.keep, &out(elem(lines, &1)))
                      err(dim(r.says))
                    json?(o) -> emit_json(r)
                    true ->
                      out(bold("assay · #{tool}") <> dim("  #{r.ms} ms"))
                      out(r.says)
                      Enum.each(r[:evidence] || [], fn e -> out("  #{if e.ok, do: good("✓"), else: warn("!")} #{bold(e.check)}: #{e.detail}") end)
                  end
                  # 0 when every check holds (a real difference, a stable leader, an unbiased judge …), 1 otherwise
                  if Enum.all?(r[:evidence] || [], & &1.ok), do: 0, else: 1
                {:error, e} -> err("assay #{tool}: #{e}"); 3
              end
            else
              {:error, e} -> err("assay: " <> e); 3
            end
        end
    end
  end

  # contamination takes TRAIN TEST files as an alternative to one JSON document
  defp input("contamination", [train, test | _]) do
    with {:ok, tr} <- read_input(train), {:ok, te} <- read_input(test) do
      lines = fn t -> t |> String.split("\n") |> Enum.reject(&(String.trim(&1) == "")) end
      {:ok, Vapor.JSON.encode(%{"train" => lines.(tr), "test" => lines.(te)})}
    end
  end

  defp input(_tool, rest), do: read_input(List.first(rest))
end
