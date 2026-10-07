defmodule Vapor.MambaHFTest do
  @moduledoc """
  Mamba (`Vapor.Lock.Adapters.Mamba`) against transformers itself:
  `MambaForCausalLM` checkpoints written by it (`test/python/hf_mamba.py`),
  compared against its own forward pass (its sequential selective scan)
  and greedy decoding — on the oracle and on the native worker, whose bits
  must agree. The control that must fail: the same program with the state
  reset at every token (a Mamba without memory) — so the comparison sees
  the recurrence.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Lock, Recurrent, Tensor}
  alias Vapor.Ingest.Safetensors
  alias Vapor.Runtime.{Oracle, Substrates, Worker}
  import Vapor.TestHelpers

  @moduletag :torch
  @moduletag timeout: 900_000

  setup_all do
    root = Path.join(System.tmp_dir!(), "vapor-mamba-#{System.unique_integer([:positive])}")
    script = File.read!(Path.expand("../python/hf_mamba.py", __DIR__))

    for v <- ~w(mamba mamba-odd) do
      File.mkdir_p!(Path.join(root, v))
      py!(script, [Path.join(root, v), v, "11"])
    end

    on_exit(fn -> File.rm_rf!(root) end)
    w = if Substrates.binary("vapor-worker", "native"), do: elem(Worker.start_link(exec: worker_exec(:host)), 1)
    {:ok, root: root, worker: w}
  end

  defp rel(rows, want) do
    Enum.zip(rows, want)
    |> Enum.map(fn {a, b} ->
      s = b |> Enum.map(&abs/1) |> Enum.max()
      (Enum.zip_with(a, b, &abs(&1 - &2)) |> Enum.max()) / s
    end)
    |> Enum.max()
  end

  for v <- ~w(mamba mamba-odd) do
    test "#{v}: every tensor read; prefill logits and greedy decoding = transformers; native bits = oracle bits; without its state it fails",
         %{root: root, worker: w} do
      dir = Path.join(root, unquote(v))
      {:ok, ref} = Safetensors.read(Path.join(dir, "reference.safetensors"))
      {:ok, m} = Lock.open(dir)
      assert m.spec.adapter == Vapor.Lock.Adapters.Mamba
      expected = m.spec |> Lock.expected() |> Enum.map(&elem(&1, 0)) |> MapSet.new()
      assert m.weights |> Map.keys() |> Enum.filter(&is_binary/1) |> Enum.reject(&MapSet.member?(expected, &1)) == []

      prompt = Tensor.to_list(ref["prompt"])
      want = ref["logits"] |> Tensor.to_floats() |> Enum.chunk_every(m.spec.vocab)

      rows = fn worker ->
        {:ok, g} = Recurrent.open(m.spec, m.weights, worker: worker)
        {_, _, rows} = Recurrent.prefill(g, prompt)
        rows
      end

      oracle = rows.(nil)
      assert rel(Enum.map(oracle, &Vapor.Sampler.floats/1), want) < 1.0e-5
      if w, do: assert(rows.(w) == oracle)

      {:ok, g} = Recurrent.open(m.spec, m.weights, worker: w)
      {:ok, ids, _} = Recurrent.generate(g, prompt, 16)
      assert prompt ++ ids == Tensor.to_list(ref["greedy"])

      # the control: the state reset before every token — no memory
      {:ok, p} = Lock.build(m.spec, m.weights, [])
      zero = m.spec.adapter.empty_state(m.spec.config)
      amnesic = for t <- prompt, do: Oracle.eval_program(p, Map.put(zero, :tok, Tensor.from_list(:s32, [1], [t]))).logits |> Tensor.to_floats()
      assert rel(amnesic, want) > 0.05
    end
  end
end
