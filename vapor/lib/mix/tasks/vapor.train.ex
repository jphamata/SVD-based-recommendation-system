defmodule Mix.Tasks.Vapor.Train do
  @shortdoc "Pre-train a byte-level Llama from scratch, reproducibly, on any number of workers"
  @moduledoc """
      mix vapor.train --corpus FILE[,FILE…] --holdout FILE[,FILE…] --out DIR
                      [--steps 1000] [--workers 2] [--every 100] [--seed 1]
                      [--d 128 --layers 2 --heads 4 --ff 384 --seq 64 --seqs 2]
                      [--micro 8 --chunk 4 --lr 0.003 --warmup 50]
                      [--resume DIR/state.safetensors]

  Trains `Vapor.Train.LM` on the bytes of the corpus files (concatenated
  with a newline), measures held-out bits per byte every `--every` steps,
  and writes into DIR:

    * `state.safetensors` — parameters and AdamW moments, resumable
      (`--resume`): the continuation has the bits of an uninterrupted run;
    * `model/` — the trained model as a Hugging Face Llama checkpoint
      (`Vapor.Model`, the engine and `transformers` load it);
    * `receipt.json` — the configuration, the SHA-256 of every corpus and
      held-out file, the schedule, the parameters' digest at every
      evaluation, the held-out curve and the n-gram baselines (byte
      frequencies; Witten–Bell orders 3 and 5) on the same held-out bytes.

  The digests do not depend on `--workers`: re-running with another count
  reproduces them (that is the claim the receipt lets anyone check).
  """
  use Mix.Task
  alias Vapor.Train.LM

  @switches [corpus: :string, holdout: :string, out: :string, steps: :integer, workers: :integer, every: :integer, seed: :integer,
             d: :integer, layers: :integer, heads: :integer, ff: :integer, seq: :integer, seqs: :integer,
             micro: :integer, chunk: :integer, lr: :float, warmup: :integer, resume: :string]

  @impl true
  def run(argv) do
    Mix.Task.run("app.start")
    {o, _, _} = OptionParser.parse(argv, strict: @switches)
    files = fn key -> (o[key] || Mix.raise("--#{key} FILE[,FILE…] is required")) |> String.split(",") end
    out = o[:out] || Mix.raise("--out DIR is required")
    File.mkdir_p!(out)
    cfiles = files.(:corpus)
    hfiles = files.(:holdout)
    corpus = cfiles |> Enum.map(&File.read!/1) |> Enum.join("\n")
    hold = hfiles |> Enum.map(&File.read!/1) |> Enum.join("\n")
    steps = o[:steps] || 1000
    every = o[:every] || 100

    c = LM.new(Keyword.take(o, [:d, :layers, :heads, :ff, :seq, :seqs]))
    nparams = LM.shapes(c) |> Enum.map(fn {_, s} -> Enum.product(s) end) |> Enum.sum()
    Mix.shell().info("model: #{nparams} parameters (d #{c.d}, #{c.layers} layers, #{c.heads} heads, ff #{c.ff}); compiling…")
    run = LM.start(c, corpus, Keyword.take(o, [:micro, :chunk, :lr, :warmup, :seed]) ++ [steps: steps])
    run = if o[:resume], do: LM.resume(run, o[:resume]), else: run
    exec = [Vapor.Runtime.Substrates.binary("vapor-worker", "native")]
    ws = for _ <- 1..(o[:workers] || 2), do: elem(Vapor.Runtime.Worker.start_link(exec: exec), 1)
    {b0, nbytes} = LM.bits_per_byte(run, hold, ws)
    Mix.shell().info("step #{run.step}: held-out #{fmt(b0)} bits/byte over #{nbytes} bytes")

    {run, curve} =
      Stream.iterate(run.step, &(&1 + every))
      |> Enum.take_while(&(&1 < steps))
      |> Enum.reduce({run, [%{step: run.step, heldout_bpb: b0, digest: LM.digest(run)}]}, fn from, {r, curve} ->
        n = min(every, steps - from)
        r = LM.train(r, corpus, ws, n)
        {b, _} = LM.bits_per_byte(r, hold, ws)
        {_, l} = hd(r.losses)
        Mix.shell().info("step #{r.step}: train #{fmt(l / :math.log(2))} bits/byte, held-out #{fmt(b)}")
        LM.checkpoint(r, Path.join(out, "state.safetensors"))
        {r, [%{step: r.step, heldout_bpb: b, train_bpb: l / :math.log(2), digest: LM.digest(r)} | curve]}
      end)

    {:ok, _} = LM.export(run, Path.join(out, "model"))
    baselines = LM.baselines(corpus, hold)
    sha = fn path -> %{file: Path.basename(path), sha256: Base.encode16(:crypto.hash(:sha256, File.read!(path)), case: :lower)} end

    receipt = %{
      vapor: to_string(Application.spec(:vapor, :vsn)), semantics: Vapor.Canon.version(),
      config: Map.from_struct(c), parameters: nparams,
      schedule: %{steps: steps, micro: run.micro, chunk: run.chunk, lr: run.lr, warmup: run.warmup, lr_end: run.lr_end,
                  beta1: run.beta1, beta2: run.beta2, seed: run.seed, tokens_per_step: run.micro * LM.rows(c)},
      corpus: Enum.map(cfiles, sha), heldout: Enum.map(hfiles, sha), heldout_bytes: nbytes,
      curve: Enum.reverse(curve), baselines: baselines, digest: LM.digest(run),
      sample: LM.generate(run, "A ", 80, ws)
    }

    File.write!(Path.join(out, "receipt.json"), Vapor.JSON.encode(receipt))
    Mix.shell().info("baselines on the same held-out bytes: " <> Enum.map_join(baselines, ", ", fn {k, v} -> "#{k} #{fmt(v)}" end))
    Mix.shell().info("wrote #{out}/ (state.safetensors, model/, receipt.json); parameters #{LM.digest(run)}")
  end

  defp fmt(x), do: :erlang.float_to_binary(x * 1.0, decimals: 3)
end
