defmodule Vapor.GpuSessionTest do
  @moduledoc """
  Resident sessions on the Vulkan fabric (P0 of 0.7): the program's buffers
  live in device memory across steps, pipelines are created once, recorded
  command buffers are replayed when a step's geometry repeats — and every
  bit equals the CPU session's, on both memory paths (direct and staged).
  """
  use ExUnit.Case, async: false
  alias Vapor.Tensor
  alias Vapor.Model.{Config, Llama}
  alias Vapor.Runtime.{Fabric, Session, Substrates, Worker}
  import Vapor.TestHelpers

  @moduletag :vulkan
  @moduletag :native

  setup_all do
    {:ok, f} = Fabric.start_link(exec: [Substrates.binary("vapor-fabric", "native"), "--fault-injection"])
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    {:ok, fabric: f, worker: w}
  end

  defp model(arch \\ "qwen2", over \\ %{}, opts \\ []) do
    {:ok, c} = Config.from_map(tiny_config(arch, over))
    {:ok, p} = Llama.program(c, tiny_weights(c), Keyword.merge([max_seq: 16], opts))
    {:ok, comp} = Vapor.Compile.Lower.lower(p)
    {c, comp}
  end

  defp ids(xs), do: Tensor.from_list(:s32, [length(xs)], xs)

  # chunked prefill, then one token per step; returns every logits row and the final cache
  defp drive(s, toks) do
    {:ok, %{logits: first}, m0} = Session.step(s, %{tok: ids(Enum.take(toks, 3)), pos: ids([0, 1, 2])}, [:logits])

    {rows, metas} =
      for i <- 3..(length(toks) - 1), reduce: {Enum.map(0..2, &Tensor.row(first, &1)), [m0]} do
        {rows, metas} ->
          {:ok, %{logits: l}, m} = Session.step(s, %{tok: ids([Enum.at(toks, i)]), pos: ids([i])}, [:logits])
          {rows ++ [l.data], metas ++ [m]}
      end

    {:ok, %{k1_next: k1}, _} = Session.step(s, %{tok: ids([0]), pos: ids([15])}, [:k1_next])
    {rows, k1, metas}
  end

  test "a GPU session = the CPU session, bit for bit, on the direct and the staged memory path", %{fabric: f, worker: w} do
    toks = [3, 1, 4, 1, 5, 9, 2, 6, 5]

    for {arch, over} <- [{"qwen2", %{}}, {"llama", %{"attention_bias" => true}}, {"mistral", %{}}] do
      {_c, comp} = model(arch, over)
      {:ok, cpu} = Session.open(w, comp, isa: Substrates.host_isa())
      {ref_rows, ref_k1, _} = drive(cpu, toks)

      for staging <- [false, true] do
        {:ok, gpu} = Session.open(f, comp, isa: :spirv, staging: staging)
        info = Session.info(gpu)
        assert info.kind == :fabric and info.staged == staging
        assert info.resident_bytes > 0
        {rows, k1, metas} = drive(gpu, toks)
        assert rows == ref_rows, "#{arch} staging=#{staging}"
        assert k1 == ref_k1

        # single-token steps share one geometry: recorded once, replayed after
        reused = metas |> Enum.drop(1) |> Enum.map(& &1.counters.gpu_recording_reused)
        assert hd(reused) == 0 and Enum.all?(tl(reused), &(&1 == 1)), inspect(reused)
        # per-token traffic: the ids in and one row of logits out, not the cache
        decode = Enum.at(metas, 2).counters.gpu_host_bytes
        assert decode == 2 * 4 + 96 * 4
        :ok = Session.close(gpu)
      end
    end
  end

  test "the engine's paged sampling step and a recurrent state stay on the device", %{fabric: f, worker: w} do
    # paged pools + native sampling: the engine's own program
    {c, comp} = model("qwen2", %{}, kv: {:paged, 4, 8, 2}, logits: :last, sample: true, max_tokens: 8)
    _ = c

    steps = [
      %{tok: ids([5, 9, 11, 3, 70]), pos: ids([0, 1, 0, 1, 2]), slot: ids([0, 0, 1, 1, 0]), last: ids([3, 4]),
        table: Tensor.from_list(:s32, [2, 4], [3, 1, 6, 0, 2, 5, 7, 4]), sampling: Tensor.from_list(:f32, [2, 2], [0.0, 0.0, 1.0, 0.37])},
      %{tok: ids([8, 2]), pos: ids([2, 3]), slot: ids([1, 0]), last: ids([0, 1]),
        table: Tensor.from_list(:s32, [2, 4], [3, 1, 6, 0, 2, 5, 7, 4]), sampling: Tensor.from_list(:f32, [2, 2], [0.0, 0.0, 0.8, 0.11])}
    ]

    run = fn s -> for e <- steps, do: (elem(Session.step(s, e, [:next, :logits]), 1)) end
    {:ok, cpu} = Session.open(w, comp, isa: Substrates.host_isa())
    {:ok, gpu} = Session.open(f, comp, isa: :spirv)
    assert run.(gpu) == run.(cpu)
    Session.close(gpu)

    # SSM state h ← h_next is fed back inside the device after each step
    {:ok, ssm} = Vapor.Compile.Lower.lower(ssm_block())
    xs = for t <- 0..3, do: Tensor.random(:f32, [512], 100 + t)
    feed = fn s -> for x <- xs, do: elem(Session.step(s, %{x: x}, [:y]), 1).y end
    {:ok, cpu} = Session.open(w, ssm, isa: Substrates.host_isa())
    {:ok, gpu} = Session.open(f, ssm, isa: :spirv)
    assert feed.(gpu) == feed.(cpu)
  end

  test "the engine serves on the GPU: every request's tokens = the CPU engine's", %{fabric: f} do
    {:ok, c} = Config.from_map(tiny_config("llama", %{"vocab_size" => 128}))
    ws = tiny_weights(c, 3)
    base = [config: c, weights: ws, max_seq: 64, page: 8, sequences: 4, step_tokens: 16]

    reqs = [{[5, 9, 2, 7, 1], [max_tokens: 12, temperature: 0.0]},
            {[3, 3, 8], [max_tokens: 10, temperature: 0.9, seed: 7]},
            {Enum.to_list(10..30), [max_tokens: 6, temperature: 0.7, top_k: 5, seed: 3]}]

    serve = fn e ->
      refs = for {p, o} <- reqs, do: elem(Vapor.Engine.generate(e, p, o), 1)
      for r <- refs, do: Vapor.Engine.collect(r, [], 120_000)
    end

    {:ok, cpu} = Vapor.Engine.start_link(base)
    want = serve.(cpu)
    assert Enum.all?(want, &match?({:ok, [_ | _], _, _}, &1))

    for staging <- [false, true] do
      {:ok, gpu} = Vapor.Engine.start_link(base ++ [isa: :spirv, fabric: f, staging: staging])
      assert %{substrate: %{kind: :fabric, staged: ^staging}} = Vapor.Engine.info(gpu)
      assert serve.(gpu) == want
      GenServer.stop(gpu)
    end
  end

  test "a state-space model decodes on the GPU: state never leaves the device, tokens = the CPU's", %{fabric: f, worker: w} do
    cfg = %{"model_type" => "mamba", "vocab_size" => 32, "hidden_size" => 32, "intermediate_size" => 64, "num_hidden_layers" => 2,
            "state_size" => 16, "time_step_rank" => 16, "conv_kernel" => 4}
    alias Vapor.Lock.Adapters.Mamba
    probe = struct(Mamba.Config, vocab: 32, hidden: 32, inner: 64, layers: 2, state: 16, rank: 16, conv: 4, eps: 1.0e-5,
                   bias: false, conv_bias: true, tie: true, bos: nil, eos: nil, raw: cfg)
    ws =
      for {name, shape, kind} <- Vapor.Lock.expected(Mamba.spec(probe)), into: %{} do
        t = Tensor.random(:f32, shape, :erlang.phash2(name), scale: 0.2)
        {name, if(kind == :norm, do: Tensor.from_list(:f32, shape, Enum.map(Tensor.to_floats(t), &(&1 + 1.0))), else: t)}
      end

    {:ok, spec, _} = Mamba.admit(%{config: cfg}, ws, [])
    {:ok, gc} = Vapor.Recurrent.open(spec, ws, worker: w)
    {:ok, gg} = Vapor.Recurrent.open(spec, ws, worker: f, isa: :spirv)
    {:ok, a, _} = Vapor.Recurrent.generate(gc, [1, 2, 3, 4], 12, temperature: 0.8, seed: 5)
    {:ok, b, _} = Vapor.Recurrent.generate(gg, [1, 2, 3, 4], 12, temperature: 0.8, seed: 5)
    assert a == b and length(a) == 12
  end

  test "Mamba-2 (two groups, gated norm per group, clamped Δ) decodes on the GPU: logits bit-identical to the CPU's", %{fabric: f, worker: w} do
    cfg = %{"model_type" => "mamba2", "vocab_size" => 32, "hidden_size" => 32, "num_heads" => 8, "head_dim" => 8, "n_groups" => 2,
            "expand" => 2, "num_hidden_layers" => 2, "state_size" => 8, "conv_kernel" => 4, "time_step_limit" => [0.0, 0.5],
            "use_bias" => true, "tie_word_embeddings" => true}
    alias Vapor.Lock.Adapters.Mamba2
    probe = struct(Mamba2.Config, vocab: 32, hidden: 32, inner: 64, heads: 8, head_dim: 8, groups: 2, layers: 2, state: 8, conv: 4,
                   eps: 1.0e-5, bias: true, conv_bias: true, tie: true, limit: {0.0, 0.5}, gated_norm: :group, raw: cfg)
    ws =
      for {name, shape, kind} <- Vapor.Lock.expected(Mamba2.spec(probe)), into: %{} do
        t = Tensor.random(:f32, shape, :erlang.phash2(name), scale: 0.3)
        {name, if(kind == :norm, do: Tensor.from_list(:f32, shape, Enum.map(Tensor.to_floats(t), &(&1 + 1.0))), else: t)}
      end

    {:ok, spec, _} = Mamba2.admit(%{config: cfg}, ws, [])
    {:ok, gc} = Vapor.Recurrent.open(spec, ws, worker: w)
    {:ok, gg} = Vapor.Recurrent.open(spec, ws, worker: f, isa: :spirv)
    {_, _, rc} = Vapor.Recurrent.prefill(gc, [1, 2, 3, 4, 5, 6])
    {_, _, rg} = Vapor.Recurrent.prefill(gg, [1, 2, 3, 4, 5, 6])
    assert rc == rg
  end

  test "a driver crash ends the session, never the BEAM; a reopened session serves on", %{fabric: f} do
    {_c, comp} = model()
    {:ok, s} = Session.open(f, comp, isa: :spirv)
    assert {:ok, _, _} = Session.step(s, %{tok: ids([1]), pos: ids([0])}, [:logits])
    assert {:error, {:fabric_crashed, {:signal, 11}}} = Fabric.inject_fault(f)
    assert {:error, :session_lost} = Session.step(s, %{tok: ids([1]), pos: ids([1])}, [:logits])
    {:ok, s2} = Session.open(f, comp, isa: :spirv)
    assert {:ok, _, _} = Session.step(s2, %{tok: ids([1]), pos: ids([0])}, [:logits])
  end

  test "hostile frames are refused, not executed", %{fabric: f} do
    {_c, comp} = model()
    {:ok, s} = Session.open(f, comp, isa: :spirv)
    # a write to a constant (read-only) buffer, an out-of-range buffer, an unknown session
    const = Enum.find_value(comp.slots, fn {id, %{role: r}} -> match?({:const, _}, r) && id end)
    idx = s.index[const]
    bad_write = [<<1000::32-little, 1::32-little, idx::32-little, 0::64, 4::64-little>>, <<0::32>>, <<0::32, 0::32, 0::32>>]
    assert {:error, {:unit_fault, %{code: :vulkan_error}}} = Fabric.step_session(f, s.ref, bad_write)
    oob = [<<1000::32-little, 0::32, 0::32, 0::32, 1::32-little, 99_999::32-little, 0::32, 0::32, 0::32, 0::32, 0::32, 1::32-little, 0::32>>, <<0::32>>]
    assert {:error, {:unit_fault, _}} = Fabric.step_session(f, s.ref, oob)
    {sid, gen} = s.ref
    assert {:error, {:unit_fault, _}} = Fabric.step_session(f, {sid + 5, gen}, [<<1000::32-little, 0::32, 0::32, 0::32, 0::32>>])
    # and the session is still intact
    assert {:ok, _, _} = Session.step(s, %{tok: ids([1]), pos: ids([0])}, [:logits])
  end
end
