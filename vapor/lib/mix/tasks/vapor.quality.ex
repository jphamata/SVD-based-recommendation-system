defmodule Mix.Tasks.Vapor.Quality do
  @shortdoc "Run the output-quality benchmark (text, any-to-any, fusion) and write docs/bench/QUALITY.md"
  @moduledoc """
      mix vapor.quality [--oracle] [--out docs/bench] [--no-gallery]
      mix vapor.quality --model PATH [--text FILE] [--reference FILE] [--samples 3]
      mix vapor.quality --only round14                      # one round's checks, printed (no files written)

  Runs `Vapor.Quality.Suite`: calibrated noise gates, a planted model through
  the whole stack, every any-to-any route of the world hub on held-out
  inputs, model fusion, substrate parity and the airlock's own cost. Writes
  `QUALITY.md`, `quality.json` and (unless `--no-gallery`) the outputs as
  PNG/WAV under `modal/`. Exits with status 1 when any check fails, so CI
  can gate on "the outputs are signal, not noise".

  With `--model`, judges a real checkpoint instead (`Vapor.Quality.Model`):
  bits per byte on held-out text against the unigram and uniform baselines,
  and its generations through the calibrated text gate. `--text` is the
  held-out text and `--reference` the corpus of the gate's profile (both
  default to vapor's Portuguese docs — give text in the model's language).
  Exits 1 when the verdict is noise.
  """
  use Mix.Task

  @impl true
  def run(argv) do
    {o, _, _} = OptionParser.parse(argv, strict: [oracle: :boolean, out: :string, gallery: :boolean, model: :string,
                                                  text: :string, reference: :string, samples: :integer, only: :string])
    Mix.Task.run("app.start")
    cond do
      o[:model] -> judge(o)
      o[:only] -> only(o[:only])
      true -> suite(o)
    end
  end

  defp judge(o) do
    opts = Enum.reject([text: o[:text] && File.read!(o[:text]), reference: o[:reference] && File.read!(o[:reference]),
                        samples: o[:samples]], fn {_, v} -> v == nil end)

    case Vapor.Quality.Model.judge(o[:model], opts) do
      {:ok, r} ->
        Mix.shell().info("#{r.family}: #{Float.round(r.bits_per_byte, 3)} bits/byte on #{r.scored_bytes} held-out bytes " <>
                         "(unigram #{Float.round(r.unigram_bits_per_byte, 3)}, uniform 8.0)")
        for s <- r.samples, do: Mix.shell().info("  #{s.verdict} #{inspect(Map.get(s, :noise))} #{inspect(String.slice(Map.get(s, :sample, ""), 0, 90))}")
        Mix.shell().info("verdict: #{r.verdict}" <> if(r.reasons == [], do: "", else: " — " <> Enum.join(r.reasons, "; ")))
        if r.verdict != :signal, do: exit({:shutdown, 1})

      {:error, rej} ->
        Mix.raise("#{inspect(rej.node)}: #{rej.bound}")
    end
  end

  defp only(round) do
    mod =
      case Regex.run(~r/^round(\d\d)$/, round) do
        [_, n] -> Module.concat(Vapor.Quality, "Round" <> n)
        _ -> Mix.raise("--only roundNN (round06 … round14)")
      end
    unless Code.ensure_loaded?(mod), do: Mix.raise("no #{inspect(mod)}")
    w = Vapor.Modal.Runner.worker()
    t0 = System.monotonic_time(:millisecond)
    r = mod.run(worker: w)
    for c <- r.checks, do: Mix.shell().info("#{if c.pass, do: "ok  ", else: "FAIL"} #{c.name}\n       value #{inspect(c.value)} · control #{inspect(c.control)} · #{c.threshold}")
    Mix.shell().info("#{Enum.count(r.checks, & &1.pass)}/#{length(r.checks)} checks passed in #{System.monotonic_time(:millisecond) - t0} ms")
    unless Enum.all?(r.checks, & &1.pass), do: exit({:shutdown, 1})
  end

  defp suite(o) do
    out = o[:out] || "docs/bench"
    File.mkdir_p!(out)
    gallery = if Keyword.get(o, :gallery, true), do: Path.join(out, "modal")
    r = Vapor.Quality.Suite.run(oracle: o[:oracle] == true, gallery: gallery)
    File.write!(Path.join(out, "QUALITY.md"), Vapor.Quality.Report.markdown(r))
    File.write!(Path.join(out, "quality.json"), Vapor.Quality.Report.json(r))

    checks = Vapor.Quality.Suite.checks(r)
    for c <- checks, do: Mix.shell().info("#{if c.pass, do: "ok  ", else: "FAIL"} #{c.name}")
    Mix.shell().info("#{Enum.count(checks, & &1.pass)}/#{length(checks)} checks passed (#{r.substrate}); wrote #{out}/QUALITY.md")
    unless Vapor.Quality.Suite.passed?(r), do: exit({:shutdown, 1})
  end
end
