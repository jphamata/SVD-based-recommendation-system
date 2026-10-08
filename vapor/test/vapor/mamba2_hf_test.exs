defmodule Vapor.Mamba2HFTest do
  @moduledoc """
  Mamba-2 (`Vapor.Lock.Adapters.Mamba2`) against transformers itself:
  `Mamba2ForCausalLM` checkpoints written by it (`test/python/hf_mamba2.py`),
  compared against its own chunked-scan forward pass and greedy decoding —
  on the oracle and on the native worker, whose bits must agree.

  Two groups of heads exercise the gated-norm disagreement between the
  training code (per group) and transformers (whole width): each setting
  matches its own reference and **fails the other's** — the comparison is
  sharp enough to tell them apart. The control that must fail as well:
  the state reset at every token.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Lock, Recurrent, Tensor}
  alias Vapor.Ingest.Safetensors
  alias Vapor.Runtime.{Oracle, Substrates, Worker}
  import Vapor.TestHelpers

  @moduletag :torch
  @moduletag timeout: 900_000

  setup_all do
    root = Path.join(System.tmp_dir!(), "vapor-mamba2-#{System.unique_integer([:positive])}")
    script = File.read!(Path.expand("../python/hf_mamba2.py", __DIR__))

    for v <- ~w(mamba2 mamba2-g2) do
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

  defp rows(m, prompt, worker) do
    {:ok, g} = Recurrent.open(m.spec, m.weights, worker: worker)
    {_, _, rows} = Recurrent.prefill(g, prompt)
    Recurrent.close(g)
    rows
  end

  defp floats(rows), do: Enum.map(rows, &Vapor.Sampler.floats/1)

  test "one group: every tensor read; logits and greedy decoding = transformers; native bits = oracle bits; amnesia fails",
       %{root: root, worker: w} do
    dir = Path.join(root, "mamba2")
    {:ok, ref} = Safetensors.read(Path.join(dir, "reference.safetensors"))
    {:ok, m} = Lock.open(dir)
    assert m.spec.adapter == Vapor.Lock.Adapters.Mamba2
    expected = m.spec |> Lock.expected() |> Enum.map(&elem(&1, 0)) |> MapSet.new()
    assert m.weights |> Map.keys() |> Enum.filter(&is_binary/1) |> Enum.reject(&MapSet.member?(expected, &1)) == []

    prompt = Tensor.to_list(ref["prompt"])
    want = ref["logits"] |> Tensor.to_floats() |> Enum.chunk_every(m.spec.vocab)
    oracle = rows(m, prompt, nil)
    assert rel(floats(oracle), want) < 1.0e-5
    if w, do: assert(rows(m, prompt, w) == oracle)

    {:ok, g} = Recurrent.open(m.spec, m.weights, worker: w)
    {:ok, ids, _} = Recurrent.generate(g, prompt, 16)
    assert prompt ++ ids == Tensor.to_list(ref["greedy"])

    {:ok, p} = Lock.build(m.spec, m.weights, [])
    zero = m.spec.adapter.empty_state(m.spec.config)
    amnesic = for t <- prompt, do: Oracle.eval_program(p, Map.put(zero, :tok, Tensor.from_list(:s32, [1], [t]))).logits |> Tensor.to_floats()
    assert rel(amnesic, want) > 0.05
  end

  test "two groups, time_step_limit biting: per-group norm = the training code, whole-width = transformers, each fails the other",
       %{root: root, worker: w} do
    dir = Path.join(root, "mamba2-g2")
    {:ok, ref} = Safetensors.read(Path.join(dir, "reference.safetensors"))
    prompt = Tensor.to_list(ref["prompt"])
    {:ok, grouped} = Lock.open(dir)
    {:ok, whole} = Lock.open(dir, gated_norm: :whole)
    assert grouped.spec.config.gated_norm == :group and grouped.spec.config.limit == {0.0, 0.6}
    refute grouped.spec.digest == whole.spec.digest
    v = grouped.spec.vocab
    want_g = ref["logits_grouped"] |> Tensor.to_floats() |> Enum.chunk_every(v)
    want_w = ref["logits"] |> Tensor.to_floats() |> Enum.chunk_every(v)

    og = rows(grouped, prompt, nil)
    ow = rows(whole, prompt, nil)
    assert rel(floats(og), want_g) < 1.0e-5
    assert rel(floats(ow), want_w) < 1.0e-5
    assert rel(floats(og), want_w) > 0.05
    assert rel(floats(ow), want_g) > 0.05

    if w do
      assert rows(grouped, prompt, w) == og
      assert rows(whole, prompt, w) == ow
    end

    for {m, key} <- [{grouped, "greedy_grouped"}, {whole, "greedy"}] do
      {:ok, g} = Recurrent.open(m.spec, m.weights, worker: w)
      {:ok, ids, _} = Recurrent.generate(g, prompt, 16)
      assert prompt ++ ids == Tensor.to_list(ref[key]), key
    end
  end

  test "configurations refused by name" do
    base = %{"model_type" => "mamba2", "vocab_size" => 96, "hidden_size" => 64, "num_heads" => 8, "head_dim" => 16, "n_groups" => 1,
             "state_size" => 16, "num_hidden_layers" => 1, "expand" => 2}
    for {patch, field} <- [{%{"head_dim" => 12}, "expand"}, {%{"n_groups" => 3}, "n_groups"}, {%{"hidden_act" => "gelu"}, "hidden_act"},
                           {%{"time_step_limit" => [-1.0, 1.0]}, "time_step_limit"}] do
      assert {:error, %Vapor.Rejection{} = r} = Lock.admit(%{source: :memory, config: Map.merge(base, patch)}, %{})
      assert inspect(r) =~ field
    end
    # Python's bare Infinity (older configs) is read; strict JSON stays strict
    assert {:ok, [+0.0, :infinity]} = Vapor.JSON.decode("[0.0, Infinity]", nonfinite: true)
    assert {:error, _} = Vapor.JSON.decode("[0.0, Infinity]")
  end
end
