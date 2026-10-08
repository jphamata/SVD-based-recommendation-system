defmodule Mix.Tasks.Vapor.Lock do
  @shortdoc "Explain how the model airlock sees a checkpoint (who claims it, what is missing)"
  @moduledoc """
      mix vapor.lock PATH [--alias FILE.json]...
      mix vapor.lock --list

  For a Hugging Face directory or a `.gguf` file: every adapter's answer
  (claim, near miss, no), the spec of the winner (family, lineage,
  contract, widths, features), and the tensors its builder needs that are
  missing or misshapen, and those it will not read. `--alias` registers
  alias descriptors first (as `VAPOR_LOCK_ALIASES` does). `--list` prints
  the registered adapters.
  """
  use Mix.Task

  @impl true
  def run(argv) do
    {o, args, _} = OptionParser.parse(argv, strict: [alias: :keep, list: :boolean])
    for f <- Keyword.get_values(o, :alias), do: {:ok, _} = Vapor.Lock.register_json(f)

    case {o[:list], args} do
      {true, _} ->
        for a <- Vapor.Lock.adapters(), do: Mix.shell().info("  #{Vapor.Lock.id(a)}  (#{inspect(if is_tuple(a), do: elem(a, 0), else: a)})")

      {_, [path | _]} ->
        r = Vapor.Lock.explain(path)
        Mix.shell().info("claims:")
        for c <- r.claims, do: Mix.shell().info("  #{String.pad_trailing(c.adapter, 14)} #{answer(c.answer)}")

        case r.admitted do
          {:ok, s} ->
            Mix.shell().info("admitted: #{s.family} (#{Enum.join(s.lineage, " → ")}), contract #{s.interface}, " <>
                             "vocab #{inspect(s.vocab)}, width #{inspect(s.width)}, features #{inspect(s.features)}")
            t = r.tensors
            Mix.shell().info("tensors: #{t.expected} expected, #{length(t.missing)} missing or misshapen, #{length(t.unused)} unused")
            for {n, want, got} <- Enum.take(t.missing, 20), do: Mix.shell().info("  missing #{n}: want #{inspect(want)}, got #{inspect(got)}")
            for n <- Enum.take(t.unused, 20), do: Mix.shell().info("  unused  #{n}")

          {:error, rej} ->
            Mix.shell().error("refused at #{inspect(rej.node)}: #{rej.bound}\n  repair: #{rej.repair}")
            exit({:shutdown, 1})
        end

      _ ->
        Mix.raise("usage: mix vapor.lock PATH | --list")
    end
  end

  defp answer({:claim, s}), do: "claims (score #{s})"
  defp answer({:near, why}), do: "near miss — #{why}"
  defp answer(:no), do: "—"
end
