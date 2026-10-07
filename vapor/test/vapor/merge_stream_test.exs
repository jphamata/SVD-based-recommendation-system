defmodule Vapor.MergeStreamTest do
  @moduledoc """
  Fusion from disk to disk (`Vapor.Merge.stream/3`): the files are byte for
  byte those the in-memory fusion writes, the receipt carries the same
  Merkle roots, compatibility is refused on what the files declare — and
  the memory it takes is that of a tensor, not of the models.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Lock, Merge}
  alias Vapor.Ingest.Safetensors
  alias Vapor.Model.Config
  import Vapor.TestHelpers

  @moduletag timeout: 600_000

  defp tmp, do: Path.join(System.tmp_dir!(), "vapor-mstream-#{System.unique_integer([:positive])}")

  # a checkpoint directory: config.json + safetensors (sharded when max_shard is small)
  defp checkpoint(seed, over \\ %{}, max_shard \\ 200_000, dtype \\ "F32") do
    map = tiny_config("llama", Map.merge(%{"hidden_size" => 128, "intermediate_size" => 256, "num_hidden_layers" => 3, "vocab_size" => 300}, over))
    {:ok, c} = Config.from_map(map)
    dir = tmp()
    File.mkdir_p!(dir)
    File.write!(Path.join(dir, "config.json"), Vapor.JSON.encode(map))
    {:ok, _} = Safetensors.write_sharded(dir, tiny_weights(c, seed), max_shard, as: dtype)
    dir
  end

  defp in_memory(dirs, out, opts) do
    open = fn d -> {:ok, m} = Lock.open(d); %{spec: m.spec, weights: m.weights} end
    base = opts[:base] && open.(opts[:base])
    {:ok, m} = Merge.merge(Enum.map(dirs, open), Keyword.put(opts, :base, base))
    {:ok, files} = Safetensors.write_sharded(out, Map.filter(m.weights, fn {k, _} -> is_binary(k) end), opts[:max_shard], as: opts[:dtype] || "F32")
    {files, m.receipt}
  end

  defp same_files!(a, b, files) do
    for f <- files, do: assert(File.read!(Path.join(a, f)) == File.read!(Path.join(b, f)), "#{f} differs")
  end

  test "every element-wise method: the same bytes as merge/2 + write_sharded/4, and the same roots in the receipt" do
    {base, a, b} = {checkpoint(1), checkpoint(2), checkpoint(3)}

    for opts <- [[method: :linear, weights: [3, 1]], [method: :task_arithmetic, base: base, lambda: 0.7],
                 [method: :slerp, t: 0.3], [method: :ties, base: base, density: 0.3],
                 [method: :dare_ties, base: base, density: 0.5, seed: 9], [method: :linear, dtype: "BF16"]] do
      opts = Keyword.put(opts, :max_shard, 300_000)
      {mem, str} = {tmp(), tmp()}
      {files, receipt} = in_memory([a, b], mem, opts)
      assert {:ok, r} = Merge.stream([a, b], str, opts)
      assert r.files == files and length(files) > 1, inspect(opts)
      same_files!(mem, str, files)
      assert r.receipt.payload.output == receipt.payload.output
      assert Enum.map(r.receipt.payload.inputs, & &1.weights) == Enum.map(receipt.payload.inputs, & &1.weights)
      assert r.receipt.payload.params == receipt.payload.params
      # and the fused directory is an ordinary checkpoint again
      File.cp!(Path.join(a, "config.json"), Path.join(str, "config.json"))
      assert {:ok, _} = Lock.open(str)
      Enum.each([mem, str], &File.rm_rf!/1)
    end

    # bf16 inputs: widened exactly, fused, as in memory
    {x, y} = {checkpoint(4, %{}, 10_000_000, "BF16"), checkpoint(5, %{}, 10_000_000, "BF16")}
    {mem, str} = {tmp(), tmp()}
    {files, _} = in_memory([x, y], mem, max_shard: 10_000_000)
    assert {:ok, %{files: ^files}} = Merge.stream([x, y], str, max_shard: 10_000_000)
    same_files!(mem, str, files)
  end

  test "refused on what the files declare: another configuration, another shape, no base, regmean" do
    a = checkpoint(1)
    other = checkpoint(2, %{"rms_norm_eps" => 1.0e-6})
    wider = checkpoint(3, %{"intermediate_size" => 320})
    out = tmp()
    assert {:error, %Vapor.Rejection{node: {:merge, :config}}} = Merge.stream([a, other], out)
    assert {:ok, _} = Merge.stream([a, other], out, allow_config_mismatch: true)
    assert {:error, %Vapor.Rejection{}} = Merge.stream([a, wider], out, allow_config_mismatch: true)
    assert {:error, %Vapor.Rejection{node: {:merge, :base}}} = Merge.stream([a, a], out, method: :ties)
    assert {:error, %Vapor.Rejection{node: {:merge, :method}}} = Merge.stream([a, a], out, method: :regmean)
    assert {:error, %Vapor.Rejection{}} = Merge.stream([a, tmp()], out)
  end

  # peak of the VM's binary memory while `f` runs, sampled every 2 ms
  defp peak(f) do
    for pid <- Process.list(), do: :erlang.garbage_collect(pid)
    parent = self()
    base = :erlang.memory(:binary)

    sampler =
      spawn(fn ->
        loop = fn loop, mx ->
          receive do
            :stop -> send(parent, {:peak, mx})
          after
            2 -> loop.(loop, max(mx, :erlang.memory(:binary)))
          end
        end

        loop.(loop, base)
      end)

    r = f.()
    send(sampler, :stop)
    receive do: ({:peak, mx} -> {r, mx - base})
  end

  test "memory: a tensor's worth, not the models'" do
    # 8 layers: ~28 MB of f32 per model, the largest tensor ~1 MB
    big = %{"hidden_size" => 256, "intermediate_size" => 512, "num_hidden_layers" => 8, "vocab_size" => 1000}
    {a, b, c} = {checkpoint(1, big, 4_000_000), checkpoint(2, big, 4_000_000), checkpoint(3, big, 4_000_000)}
    total = [a, b, c] |> Enum.map(fn d -> d |> File.ls!() |> Enum.map(&File.stat!(Path.join(d, &1)).size) |> Enum.sum() end) |> Enum.sum()

    {{:ok, _}, streamed} = peak(fn -> Merge.stream([a, b, c], tmp(), max_shard: 4_000_000) end)
    {_, loaded} = peak(fn -> in_memory([a, b, c], tmp(), max_shard: 4_000_000) end)
    IO.puts("\n  merge of 3 × #{div(total, 3 * 1_000_000)} MB: peak binary memory streamed #{div(streamed, 1_000_000)} MB, in memory #{div(loaded, 1_000_000)} MB")
    assert streamed < total / 3
    assert streamed * 4 < loaded
  end
end
