defmodule Mix.Tasks.Vapor.Export do
  @shortdoc "Export a model to GGUF, a Hugging Face directory, or a StableHLO module"
  @moduledoc """
      mix vapor.export --model PATH --out FILE.gguf [--type f32|q8_0]
      mix vapor.export --model PATH --out DIR [--dtype f32|bf16|f16] [--shard-size BYTES]
      mix vapor.export --model PATH --out FILE.mlir [--tokens T] [--max-seq S]

  Reads a checkpoint directory or a GGUF file (`Vapor.Model.open/1`) and
  writes either

    * a GGUF that llama.cpp loads (`--out` ending in `.gguf`,
      `Vapor.Model.GGUF.write/4`): the converter's conventions applied;
    * a Hugging Face checkpoint directory (`Vapor.Model.write/4`):
      `config.json` and safetensors weights, sharded above `--shard-size`
      (default 5 GB) with `model.safetensors.index.json`.

  The vocabulary comes along when the source has one: `tokenizer.json`
  (copied into a directory), or a GGUF vocabulary (`tokenizer.gguf` or the
  source file itself).

  `--out FILE.mlir` writes the model's decoder step (`T` tokens against a
  contiguous KV cache of `S` rows) as a StableHLO module
  (`Vapor.Export.StableHLO`) — what Tenstorrent's `tt-xla`, XLA or IREE
  compile. The device that runs it is admitted with `mix vapor.substrate kit`.
  """
  use Mix.Task

  @switches [model: :string, out: :string, type: :string, dtype: :string, shard_size: :integer, tokens: :integer, max_seq: :integer]

  @impl true
  def run(argv) do
    {o, _, _} = OptionParser.parse(argv, strict: @switches)
    src = o[:model] || Mix.raise("--model PATH (a checkpoint directory or a .gguf file) is required")
    out = o[:out] || Mix.raise("--out FILE.gguf, FILE.mlir or --out DIR is required")

    if String.ends_with?(out, ".mlir") do
      %{program: p} = ok!(Vapor.Model.load(src, max_seq: o[:max_seq] || 512))
      ex = ok!(Vapor.Export.StableHLO.export(p, dims: %{t: o[:tokens] || 1}))
      File.write!(out, ex.mlir)
      sig = fn xs -> Enum.map_join(xs, ", ", fn {n, dt, s} -> "#{n}: #{dt}#{inspect(s)}" end) end
      Mix.shell().info("wrote #{out} (#{byte_size(ex.mlir)} bytes)\n  inputs:  #{sig.(ex.inputs)}\n  outputs: #{sig.(ex.outputs)}")
      exit(:normal)
    end

    m = ok!(Vapor.Model.open(src))

    vocab =
      Enum.find_value([src, Path.join(src, "tokenizer.gguf")], nil, fn p ->
        with true <- File.regular?(p) and String.ends_with?(p, ".gguf"), {:ok, g} <- Vapor.Ingest.GGUF.read(p), do: g, else: (_ -> nil)
      end)

    json = Path.join(src, "tokenizer.json")
    json = if File.regular?(json), do: json

    if String.ends_with?(out, ".gguf") do
      type = pick(o[:type] || "f32", %{"f32" => :f32, "q8_0" => :q8_0}, "--type")
      if vocab == nil, do: Mix.shell().info("note: no GGUF vocabulary in #{src}; the file carries weights and configuration only")
      :ok = ok!(Vapor.Model.GGUF.write(out, m.config, m.weights, type: type, vocab: vocab || %{}, name: Path.basename(Path.expand(src))))
      Mix.shell().info("wrote #{out} (#{type}, #{File.stat!(out).size} bytes)")
    else
      dtype = pick(o[:dtype] || "f32", %{"f32" => "F32", "bf16" => "BF16", "f16" => "F16"}, "--dtype")
      opts = [dtype: dtype, shard_bytes: o[:shard_size] || 5_000_000_000, tokenizer_json: json, vocab: vocab]
      :ok = ok!(Vapor.Model.write(out, m.config, m.weights, opts))
      Mix.shell().info("wrote #{out}/ (#{dtype}: #{out |> File.ls!() |> Enum.sort() |> Enum.join(", ")})")
    end
  end

  defp pick(v, table, flag), do: Map.get(table, v) || Mix.raise("#{flag} #{table |> Map.keys() |> Enum.join("|")} (got #{v})")

  defp ok!({:ok, x}), do: x
  defp ok!(:ok), do: :ok
  defp ok!({:error, r}), do: Mix.raise(inspect(r))
end
