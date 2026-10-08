defmodule Vapor.KimiTest do
  @moduledoc """
  Kimi K3 through the airlock — the alias `kimi_k3` onto the delta-rule
  hybrid topology (`Vapor.Lock.Adapters.DeltaHybrid`) — against an
  independent reference written from the technical report's equations
  (`test/python/kimi_k3_reference.py`, numpy in float64, no shared code,
  different algorithms: no caches, decompressed MLA, the whole sequence at
  every step). Two checkpoints: the report's constants, and a variant whose
  soft caps bite, whose query is low-rank and whose experts are MXFP4.

  The controls that must fail: forgetting the KDA state, or the MLA cache,
  at every token.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Lock, Recurrent, Tensor}
  alias Vapor.Ingest.Safetensors
  alias Vapor.Quant.MXFP4
  alias Vapor.Runtime.{Oracle, Substrates, Worker}
  import Vapor.TestHelpers

  @moduletag :python
  @moduletag timeout: 900_000

  setup_all do
    root = Path.join(System.tmp_dir!(), "vapor-kimi-#{System.unique_integer([:positive])}")
    script = File.read!(Path.expand("../python/kimi_k3_reference.py", __DIR__))

    for v <- ~w(k3 k3-tight) do
      File.mkdir_p!(Path.join(root, v))
      py!(script, [Path.join(root, v), v, "7"])
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

  # prefill, then greedy decoding on the same generator (what `Recurrent.generate/4` does at
  # temperature 0): one program lowered per run, not one per call
  defp run(m, prompt, n, worker) do
    {:ok, g} = Recurrent.open(m.spec, m.weights, worker: worker)
    {last, g, rows} = Recurrent.prefill(g, prompt)
    params = Vapor.Sampler.params([])

    {ids, _, g} =
      Enum.reduce(0..(n - 1), {[], last, g}, fn i, {ids, row, g} ->
        id = Vapor.Sampler.sample(row, params, i)
        if i == n - 1, do: {[id | ids], row, g}, else: (fn {row, g} -> {[id | ids], row, g} end).(Recurrent.feed(g, id))
      end)

    Recurrent.close(g)
    {rows, Enum.reverse(ids)}
  end

  defp floats(rows), do: Enum.map(rows, &Vapor.Sampler.floats/1)

  # the oracle, with the state whose names `forget` matches reset to zero before every token
  # (on the first six tokens: enough to tell the functions apart)
  defp forgetful(m, prompt, forget) do
    prompt = Enum.take(prompt, 6)
    {:ok, p} = Lock.build(m.spec, m.weights, [])
    zero = m.spec.adapter.empty_state(m.spec.config)

    {rows, _} =
      prompt
      |> Enum.with_index()
      |> Enum.map_reduce(zero, fn {t, i}, st ->
        st = Map.new(st, fn {k, v} -> {k, if(forget.(Atom.to_string(k)), do: zero[k], else: v)} end)
        out = Oracle.eval_program(p, Map.merge(st, %{tok: Tensor.from_list(:s32, [1], [t]), pos: Tensor.from_list(:s32, [1], [i])}))
        {Tensor.to_floats(out.logits), Map.new(p.state, fn {i, o} -> {i, out[o]} end)}
      end)

    rows
  end

  # `w`: a native worker, whose bits must equal the oracle's (nil: the oracle alone)
  defp check(dir, w, tol) do
    {:ok, ref} = Safetensors.read(Path.join(dir, "reference.safetensors"))
    {:ok, m} = Lock.open(dir, context: 32)
    assert m.spec.adapter == Vapor.Lock.Adapters.DeltaHybrid and m.spec.family == "kimi_k3"
    assert m.spec.lineage == ["kimi_k3", "vapor_delta_hybrid"]
    assert m.spec.config.types == [:kda, :kda, :kda, :mla, :kda, :mla]
    expected = m.spec |> Lock.expected() |> Enum.map(&elem(&1, 0)) |> MapSet.new()
    assert m.weights |> Map.keys() |> Enum.filter(&is_binary/1) |> Enum.reject(&MapSet.member?(expected, &1)) == []

    prompt = Tensor.to_list(ref["prompt"])
    want = ref["logits"] |> Tensor.to_floats() |> Enum.chunk_every(m.spec.vocab)
    {rows, ids} = run(m, prompt, 10, w)
    assert rel(floats(rows), want) < tol
    assert prompt ++ ids == Tensor.to_list(ref["greedy"])

    if w do
      {:ok, g} = Recurrent.open(m.spec, m.weights, worker: nil)
      {_, _, oracle} = Recurrent.prefill(g, prompt)
      assert rows == oracle
    end

    {m, prompt, want}
  end

  test "K3: every tensor read; logits and greedy decoding = the reference; native bits = oracle; both memories matter",
       %{root: root, worker: w} do
    {m, prompt, want} = check(Path.join(root, "k3"), w, 2.0e-5)
    # controls: without the recurrent state, or without the latent cache, the model is another function
    assert rel(forgetful(m, prompt, &String.match?(&1, ~r/^(kda|conv)\d/)), want) > 0.05
    assert rel(forgetful(m, prompt, &String.starts_with?(&1, "lat")), want) > 0.05
  end

  test "soft caps biting, a low-rank query, MXFP4 experts decoded exactly: = the reference", %{root: root} do
    dir = Path.join(root, "k3-tight")
    {:ok, raw} = Safetensors.read(Path.join(dir, "model.safetensors"))
    assert Map.has_key?(raw, "model.layers.1.mlp.experts.0.gate_proj.weight_blocks")
    refute Map.has_key?(raw, "model.layers.1.mlp.experts.0.gate_proj.weight")
    {m, _, _} = check(dir, nil, 5.0e-5)
    assert m.spec.config.beta == {0.5, 0.8} and m.spec.config.mla.q_lora == 16
  end

  test "the report's chunkwise form (Eq. 4) = its recurrence (Eq. 1); the bounded decay keeps it finite", %{root: root} do
    for v <- ~w(k3 k3-tight) do
      {:ok, ref} = Safetensors.read(Path.join([root, v, "reference.safetensors"]))
      assert hd(Tensor.to_floats(ref["chunk_gap"])) < 1.0e-12
      # over a 16-token tile in float32: g ∈ (−5, 0) keeps 1/Γ < e^80; −e^A·softplus(z) overflows
      assert Tensor.to_floats(ref["overflow"]) == [1.0, 0.0]
    end
  end

  test "a positioned model refuses to leave its cache", %{root: root} do
    {:ok, m} = Lock.open(Path.join(root, "k3"), context: 2)
    assert m.spec.max_pos == 2
    {:ok, g} = Recurrent.open(m.spec, m.weights, worker: nil)
    assert {:error, %Vapor.Rejection{node: {:context, "kimi_k3"}}} = Recurrent.generate(g, [5, 6], 2)
    {_, g, _} = Recurrent.prefill(g, [5, 6])
    assert Recurrent.room(g) == 0
    assert_raise ArgumentError, ~r/holds 2 positions/, fn -> Recurrent.feed(g, 7) end
  end

  test "refusals name the field", %{root: root} do
    dir = Path.join(root, "k3")
    {:ok, ws} = Vapor.Model.weights(dir)
    cfg = dir |> Path.join("config.json") |> File.read!() |> Vapor.JSON.decode!()
    refused = fn c -> {:error, r} = Lock.from_map(c, ws); r.node end

    assert refused.(Map.put(cfg, "qk_rope_head_dim", 64)) == {:config, "qk_rope_head_dim"}
    assert refused.(Map.put(cfg, "hidden_act", "silu")) == {:config, "hidden_act"}
    assert refused.(Map.put(cfg, "moe_latent_size", 24)) == {:config, "moe_latent_size"}
    assert refused.(Map.put(cfg, "num_experts_per_tok", 17)) == {:config, "num_experts_per_tok"}
    assert refused.(Map.put(cfg, "layer_types", ~w(kda mla))) == {:config, "layer_types"}
    # tensors are checked when the program is built (an alias renames them after admission)
    {:ok, spec, partial} = Lock.from_map(cfg, Map.delete(ws, "model.layers.3.self_attn.kv_b_proj.weight"))
    assert {:error, %{node: {:weight, "model.layers.3.self_attn.kv_b_proj.weight"}}} = Lock.build(spec, partial)
    # Kimi Linear is a near miss with the reason, not a silent K3
    assert {:error, r} = Lock.from_map(Map.put(cfg, "model_type", "kimi_linear"), ws)
    assert inspect(r) =~ "Kimi Linear"
  end

  describe "MXFP4" do
    defp block(codes, scale) do
      bytes = for [lo, hi] <- Enum.chunk_every(codes, 2), into: <<>>, do: <<Bitwise.bor(lo, Bitwise.bsl(hi, 4))>>
      {Tensor.new(:u8, [1, 1, 16], bytes), Tensor.new(:u8, [1, 1], <<scale>>)}
    end

    test "the sixteen E2M1 values, decoded exactly under every finite scale" do
      assert MXFP4.values() == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
      codes = Enum.to_list(0..15) ++ Enum.to_list(15..0//-1)

      for s <- [0, 1, 100, 127, 130, 252] do
        {b, sc} = block(codes, s)
        {:ok, t} = MXFP4.decode("w", b, sc)
        assert t.shape == [1, 32]
        assert Tensor.to_floats(t) == Enum.map(codes, &(Enum.at(MXFP4.values(), &1) * :math.pow(2, s - 127)))
      end

      # the smallest magnitude, ½·2^-127 = 2^-128, is a binary32 subnormal: bits 1 << 21
      {b, sc} = block([1 | List.duplicate(0, 31)], 0)
      {:ok, t} = MXFP4.decode("w", b, sc)
      assert <<0x0020_0000::32-little, _::binary>> = t.data
    end

    test "a NaN scale, or an element that overflows binary32, is refused; the rest of a scale is not" do
      {b, sc} = block(List.duplicate(1, 32), 255)
      assert {:error, %{node: {:weight, "w_scales"}}} = MXFP4.decode("w", b, sc)
      # under s = 254 (2^127): 1½ fits, 6 does not
      {b, sc} = block(List.duplicate(3, 32), 254)
      assert {:ok, t} = MXFP4.decode("w", b, sc)
      assert hd(Tensor.to_floats(t)) == 1.5 * :math.pow(2, 127)
      {b, sc} = block([7 | List.duplicate(0, 31)], 254)
      assert {:error, _} = MXFP4.decode("w", b, sc)
      # a misshapen pair is named, and tensors without a pair are untouched
      assert {:error, %{node: {:weight, "x"}}} = MXFP4.expand(%{"x_blocks" => Tensor.new(:u8, [1, 16], :binary.copy(<<0>>, 16))})
      {b, sc} = block(List.duplicate(2, 32), 127)
      {:ok, ws} = MXFP4.expand(%{"m.weight_blocks" => b, "m.weight_scales" => sc, "n" => :kept})
      assert Map.keys(ws) |> Enum.sort() == ["m.weight", "n"] and ws["n"] == :kept
      assert Tensor.to_floats(ws["m.weight"]) == List.duplicate(1.0, 32)
    end
  end
end
