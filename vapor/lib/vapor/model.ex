defmodule Vapor.Model do
  @moduledoc """
  A model on disk → a vapor program.

      {:ok, %{config: cfg, program: p}} = Vapor.Model.load("path/to/Qwen2-0.5B", max_seq: 512)
      {:ok, %{config: cfg, program: p}} = Vapor.Model.load("path/to/model-q8_0.gguf")

  Two forms, told apart by the path:

    * a Hugging Face checkpoint directory: `config.json` and either
      `model.safetensors` or the sharded form (`model.safetensors.index.json`
      + shards), with `tokenizer.json` or `tokenizer.gguf`;
    * a llama.cpp `.gguf` file, which carries configuration, weights and
      vocabulary together (`Vapor.Model.GGUF`).

  Every file passes its airlock (`Vapor.JSON`, `Vapor.Ingest.Safetensors`,
  `Vapor.Ingest.GGUF`, `Vapor.Model.Config`) before anything is built;
  shard names from the index must be plain file names inside the directory.
  """
  alias Vapor.{Rejection, JSON}
  alias Vapor.Ingest.Safetensors
  alias Vapor.Model.Config

  @doc """
  Load and build, through the model airlock (`Vapor.Lock`): the adapter
  that claims the checkpoint builds the program (options are its own; for
  decoders those of `Vapor.Model.Llama.program/3`).
  """
  def load(path, opts \\ []) do
    keep = if Keyword.get(opts, :storage) == :bf16, do: [bf16: :keep], else: []

    with {:ok, %{spec: spec, weights: ws}} <- Vapor.Lock.open(path, keep),
         {:ok, p} <- Vapor.Lock.build(spec, ws, opts) do
      {:ok, %{config: spec.config, spec: spec, program: p}}
    end
  end

  @doc """
  Configuration, weights and tokenizer of a checkpoint directory or a
  `.gguf` file, admitted by the model airlock: `{:ok, %{config, spec,
  weights, tokenizer}}` (`tokenizer` is `nil` when the model carries none;
  `config` is the adapter's admitted configuration, `spec` the contract the
  core reads). Option `bf16: :keep` keeps bfloat16 safetensors as `:bf16`
  tensors (for `storage: :bf16` programs).
  """
  def open(path, opts \\ []), do: Vapor.Lock.open(path, opts)

  @doc """
  Write a model as a Hugging Face checkpoint directory: `config.json`
  (`Config.to_map/1`), the weights as safetensors — one file, or shards of
  at most `:shard_bytes` (default 5 GB) with `model.safetensors.index.json`
  — stored as `:dtype` `"F32"` (default), `"BF16"` or `"F16"` (round to
  nearest-even). The vocabulary comes along when given: `:tokenizer_json`
  (a file, copied) or `:vocab` (a GGUF read, written as `tokenizer.gguf`).
  `open(dir)` then gives back the configuration and, in F32, the very
  weights (tested); transformers loads the directory (tested).
  """
  def write(dir, %Config{} = cfg, weights, opts \\ []) do
    tensors = for {k, %Vapor.Tensor{} = t} <- weights, is_binary(k), into: %{}, do: {k, t}
    dtype = Keyword.get(opts, :dtype, "F32")
    # norms and biases stay binary32: narrowing them buys nothing
    as = fn name -> if dtype != "F32" and length(tensors[name].shape) == 2, do: dtype, else: "F32" end

    with {:ok, map} <- Config.to_map(cfg),
         :ok <- File.mkdir_p(dir),
         :ok <- File.write(Path.join(dir, "config.json"), JSON.encode(%{map | "torch_dtype" => torch_dtype(dtype)})),
         {:ok, _} <- Safetensors.write_sharded(dir, tensors, Keyword.get(opts, :shard_bytes, 5_000_000_000), as: as) do
      cond do
        src = opts[:tokenizer_json] -> File.cp(src, Path.join(dir, "tokenizer.json"))
        g = opts[:vocab] -> write_vocab(Path.join(dir, "tokenizer.gguf"), g)
        true -> :ok
      end
    end
  end

  defp torch_dtype("F32"), do: "float32"
  defp torch_dtype("BF16"), do: "bfloat16"
  defp torch_dtype("F16"), do: "float16"

  # a vocabulary-only GGUF (the `tokenizer.*` keys, with their types)
  defp write_vocab(path, %{metadata: m, types: t}) do
    meta = for {k, v} <- Enum.sort(m), String.starts_with?(k, "tokenizer."), do: {k, if(ty = t[k], do: {ty, v}, else: v)}
    Vapor.Ingest.GGUF.write(path, [{"general.architecture", "llama"} | meta], [])
  end

  defp gguf?(path), do: File.regular?(path) and String.ends_with?(path, ".gguf")

  @doc "All tensors of a checkpoint directory (single file or sharded); options as `Safetensors.read/2`."
  def weights(dir, opts \\ []) do
    # transformers writes model.safetensors; diffusers (VAEs, U-Nets, DiTs)
    # diffusion_pytorch_model.safetensors — same format, same sharding
    found =
      Enum.find_value(["model", "diffusion_pytorch_model"], fn stem ->
        single = Path.join(dir, stem <> ".safetensors")
        index = Path.join(dir, stem <> ".safetensors.index.json")

        cond do
          File.regular?(single) -> {:single, single}
          File.regular?(index) -> {:index, index}
          true -> nil
        end
      end)

    case found do
      {:single, f} -> Safetensors.read(f, opts)
      {:index, f} -> sharded(dir, f, opts)
      nil -> {:error, Rejection.new({:checkpoint, dir}, "model.safetensors (or diffusion_pytorch_model.safetensors), single or sharded", "download the safetensors weights")}
    end
  end

  @doc """
  The tokenizer of a checkpoint directory (`tokenizer.json` from Hugging
  Face or `tokenizer.gguf`, a llama.cpp vocabulary) or of a `.gguf` model.
  """
  def tokenizer(path) do
    if gguf?(path), do: gguf_tokenizer(path), else: dir_tokenizer(path)
  end

  defp gguf_tokenizer(path) do
    with {:ok, g} <- Vapor.Ingest.GGUF.read(path), do: Vapor.Tokenizer.from_gguf(g.metadata)
  end

  defp dir_tokenizer(dir) do
    json = Path.join(dir, "tokenizer.json")
    gguf = Path.join(dir, "tokenizer.gguf")

    cond do
      File.regular?(json) -> Vapor.Tokenizer.load(json)
      File.regular?(gguf) -> with {:ok, g} <- Vapor.Ingest.GGUF.read(gguf), do: Vapor.Tokenizer.from_gguf(g.metadata)
      true -> {:error, Rejection.new({:checkpoint, dir}, "tokenizer.json or tokenizer.gguf", "add the model's tokenizer")}
    end
  end

  defp sharded(dir, index, opts) do
    with {:ok, bin} <- File.read(index),
         {:ok, %{"weight_map" => map}} when is_map(map) <- JSON.decode(bin),
         :ok <- plain_names(Map.values(map), index) do
      map
      |> Enum.group_by(&elem(&1, 1), &elem(&1, 0))
      |> Enum.sort()
      |> Enum.reduce_while({:ok, %{}}, fn {file, names}, {:ok, acc} ->
        case Safetensors.read(Path.join(dir, file), Keyword.put(opts, :only, names)) do
          {:ok, ts} when map_size(ts) == length(names) -> {:cont, {:ok, Map.merge(acc, ts)}}
          {:ok, ts} -> {:halt, reject(index, "#{file} holds #{inspect(hd(names -- Map.keys(ts)))}")}
          err -> {:halt, err}
        end
      end)
    else
      {:error, %Rejection{}} = e -> e
      _ -> reject(index, "an index with a weight_map object")
    end
  end

  defp plain_names(files, index) do
    case Enum.find(files, &(not is_binary(&1) or Path.basename(&1) != &1 or &1 in [".", ".."])) do
      nil -> :ok
      bad -> reject(index, "shard names are plain file names (got #{inspect(bad)})")
    end
  end

  defp reject(node, bound), do: {:error, Rejection.new({:checkpoint, node}, bound, "check the checkpoint files")}
end
