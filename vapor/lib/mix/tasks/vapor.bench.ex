defmodule Mix.Tasks.Vapor.Bench do
  @shortdoc "Measure kernels, engine, accuracy and tokenizer; write docs/bench"
  @moduledoc """
      mix vapor.bench [--out docs/bench]
      mix vapor.bench --frontier          # the 0.6 round: experts, latent cache, windows, ÷, spatial, ledger, shards
      mix vapor.bench --round08           # the 0.8 round: GPU sessions, sparse 4-bit experts, …

  See `Vapor.Bench`, `Vapor.Bench.Frontier` and `Vapor.Bench.Round08`. Needs the native worker (`make native`).
  """
  use Mix.Task

  @impl true
  def run(argv) do
    Mix.Task.run("app.start")
    {o, _, _} = OptionParser.parse(argv, strict: [out: :string, frontier: :boolean, round08: :boolean, only: :string])

    cond do
      o[:frontier] -> Vapor.Bench.Frontier.run(out: o[:out] || "docs/bench")
      o[:round08] ->
        only = if o[:only], do: [sections: o[:only] |> String.split(",") |> Enum.map(&String.to_atom/1)], else: []
        Vapor.Bench.Round08.run([out: o[:out] || "docs/bench"] ++ only)
      true -> Vapor.Bench.run(out: o[:out] || "docs/bench")
    end
  end
end
