defmodule Vapor.PalingenesisTest do
  use ExUnit.Case, async: false
  alias Vapor.{Certificate, Lock, Palingenesis, Tensor}
  alias Vapor.Model.Config
  import Vapor.TestHelpers

  @moduletag timeout: 600_000
  @plank "model.layers.1.mlp"

  setup_all do
    map = tiny_config("qwen2")
    {:ok, c} = Config.from_map(map)
    {:ok, spec, ws} = Lock.from_map(map, tiny_weights(c, 1))
    :rand.seed(:exsss, {1, 2, 3})
    anchors = for _ <- 1..3, do: for(_ <- 1..6, do: :rand.uniform(95))

    targets =
      # samples of the clean model itself: in expectation no model predicts them better than it does
      for i <- 1..16 do
        prompt = for _ <- 1..3, do: :rand.uniform(95)
        prompt ++ Vapor.Modal.Text.generate(spec, ws, prompt, 12, temperature: 1.0, seed: i)
      end

    %{spec: spec, clean: ws, anchors: anchors, targets: targets}
  end

  defp noisy(t, seed, s), do: Tensor.from_list(:f32, t.shape, Enum.zip_with(Tensor.to_floats(t), Tensor.to_floats(Tensor.random(:f32, t.shape, seed, scale: s)), &(&1 + &2)))
  defp names(ws), do: Palingenesis.planks(ws)[@plank]
  defp corrupt(ws, s), do: Map.new(ws, fn {k, v} -> if k in names(ws), do: {k, noisy(v, 900 + :erlang.phash2(k, 1000), s)}, else: {k, v} end)

  # the same block with its hidden units in another order: the same function
  defp permuted(ws, seed) do
    :rand.seed(:exsss, {seed, 5, 9})
    g = ws[@plank <> ".gate_proj.weight"]
    [n, k] = g.shape
    perm = Enum.shuffle(0..(n - 1))
    rows = fn t -> rs = t |> Tensor.to_floats() |> Enum.chunk_every(k) |> List.to_tuple(); Tensor.from_list(:f32, t.shape, Enum.flat_map(perm, &elem(rs, &1))) end
    d = ws[@plank <> ".down_proj.weight"]
    cols = d |> Tensor.to_floats() |> Enum.chunk_every(n) |> Enum.flat_map(fn r -> rt = List.to_tuple(r); Enum.map(perm, &elem(rt, &1)) end)
    %{@plank <> ".gate_proj.weight" => rows.(g), @plank <> ".up_proj.weight" => rows.(ws[@plank <> ".up_proj.weight"]),
      @plank <> ".down_proj.weight" => Tensor.from_list(:f32, d.shape, cols)}
  end

  defp launch(name, spec, ws, opts \\ []) do
    Palingenesis.retire(name)
    {:ok, g} = Palingenesis.launch(name, %{spec: spec, weights: ws}, opts)
    g
  end

  test "planks cover every tensor once, and the hull's root is the root of its planks", %{clean: ws} do
    ps = Palingenesis.planks(ws)
    covered = ps |> Map.values() |> List.flatten() |> Enum.sort()
    assert covered == ws |> Map.keys() |> Enum.filter(&is_binary/1) |> Enum.sort()
    assert ps[@plank] == Enum.map(~w(down_proj gate_proj up_proj), &"#{@plank}.#{&1}.weight")
    assert ps["model.layers.0.self_attn"] |> length() >= 4
    r = Palingenesis.root(ws, ps)
    refute r == Palingenesis.root(corrupt(ws, 0.01), ps)
  end

  test "the target gate admits a plank that restores the ship and refuses a sham of the same norm", ctx do
    old = corrupt(ctx.clean, 0.2)
    launch(:restore, ctx.spec, old)
    clean = Map.take(ctx.clean, names(old))
    opts = [anchors: ctx.anchors, targets: ctx.targets, epsilon: :math.pi()]

    # the control first: a random change of the same size, from the same generation
    sham =
      Map.new(names(old), fn k ->
        d = Enum.zip_with(Tensor.to_floats(clean[k]), Tensor.to_floats(old[k]), &(&1 - &2))
        r = Tensor.to_floats(Tensor.random(:f32, old[k].shape, 4242 + :erlang.phash2(k, 1000)))
        scale = :math.sqrt(Enum.sum(Enum.map(d, &(&1 * &1)))) / :math.sqrt(Enum.sum(Enum.map(r, &(&1 * &1))))
        {k, Tensor.from_list(:f32, old[k].shape, Enum.zip_with(Tensor.to_floats(old[k]), r, &(&1 + &2 * scale)))}
      end)

    assert {:error, %{gate: "target"} = refused} = Palingenesis.propose(:restore, @plank, sham, opts)
    assert refused.target.bits_new >= refused.target.bits_old or refused.target.p_value > 0.05
    {:ok, still} = Palingenesis.checkout(:restore)
    assert still.gen == 0

    assert {:ok, g1, report} = Palingenesis.propose(:restore, @plank, clean, opts)
    assert g1.gen == 1
    assert report.target.bits_new < report.target.bits_old and report.target.p_value <= 0.05
    assert report.record.payload.plank == @plank
    assert :ok = Palingenesis.verify(g1)
  end

  test "the brake: the same restoration is refused when its drift exceeds ε, and nothing is published", ctx do
    old = corrupt(ctx.clean, 0.2)
    g0 = launch(:brake, ctx.spec, old)
    clean = Map.take(ctx.clean, names(old))
    assert {:error, %{gate: "drift", drift: d}} = Palingenesis.propose(:brake, @plank, clean, anchors: ctx.anchors, targets: ctx.targets, epsilon: 0.5)
    assert d.max > 0.5 and d.positions == 18
    {:ok, g} = Palingenesis.checkout(:brake)
    assert g.root == g0.root and g.gen == 0
    assert {:error, %{gate: "drift"}} = Palingenesis.propose(:brake, @plank, clean, anchors: [])
  end

  test "a whole block with its hidden units permuted is the same plank to the network: drift at rounding level", ctx do
    launch(:perm, ctx.spec, ctx.clean)
    assert {:ok, g1, report} = Palingenesis.propose(:perm, @plank, permuted(ctx.clean, 3), anchors: ctx.anchors, epsilon: 1.0e-3)
    assert report.drift.max < 1.0e-3
    # the bytes changed, so the root did, while the behaviour did not
    refute g1.root == hd(tl(g1.records)).payload.root
  end

  test "blending needs alignment: aligned, the blend stays near the ship; unaligned, it tears the block", ctx do
    fresh = permuted(Map.merge(ctx.clean, Map.new(names(ctx.clean), fn k -> {k, noisy(ctx.clean[k], 77 + :erlang.phash2(k, 100), 0.05)} end)), 4)
    launch(:blend, ctx.spec, ctx.clean)
    {:ok, _, aligned} = Palingenesis.propose(:blend, @plank, fresh, anchors: ctx.anchors, mode: {:blend, 0.5}, epsilon: :math.pi())
    launch(:blend, ctx.spec, ctx.clean)
    {:ok, _, unaligned} = Palingenesis.propose(:blend, @plank, fresh, anchors: ctx.anchors, mode: {:blend, 0.5}, align: false, epsilon: :math.pi())
    assert aligned.record.payload.alignment.moved |> hd() > 0
    assert unaligned.record.payload.alignment == nil
    assert aligned.drift.mean * 3 < unaligned.drift.mean
  end

  test "the contract gate: same names, shapes and dtypes, and a real change", ctx do
    launch(:contract, ctx.spec, ctx.clean)
    one = Map.take(ctx.clean, names(ctx.clean))
    assert {:error, %{gate: "contract", reason: r1}} = Palingenesis.propose(:contract, @plank, Map.delete(one, hd(names(ctx.clean))), anchors: ctx.anchors)
    assert r1 =~ "the plank is"
    bad = Map.update!(one, hd(names(ctx.clean)), fn t -> Tensor.from_list(:f32, [1], [0.0]) |> Map.put(:dtype, t.dtype) end)
    assert {:error, %{gate: "contract", reason: r2}} = Palingenesis.propose(:contract, @plank, bad, anchors: ctx.anchors)
    assert r2 =~ "shape"
    assert {:error, %{gate: "contract", reason: r3}} = Palingenesis.propose(:contract, @plank, one, anchors: ctx.anchors)
    assert r3 =~ "identical"
    assert {:error, %{gate: "contract"}} = Palingenesis.propose(:contract, "no.such.plank", one, anchors: ctx.anchors)
  end

  test "read-copy-update: a reader keeps its generation, whole, while new ones are published", ctx do
    g0 = launch(:rcu, ctx.spec, ctx.clean)
    parent = self()

    readers =
      for i <- 1..8 do
        spawn_link(fn ->
          seen =
            for _ <- 1..30 do
              {:ok, g} = Palingenesis.checkout(:rcu)
              # every generation a reader holds hashes to its own root: no torn reads
              true = Palingenesis.root(g.weights, g.planks) == g.root
              g.gen
            end

          send(parent, {:seen, i, seen})
        end)
      end

    held = g0
    for s <- [0.001, 0.002] do
      ps = Map.new(names(ctx.clean), fn k -> {k, noisy(ctx.clean[k], 11 + :erlang.phash2({k, s}, 1000), s)} end)
      {:ok, _, _} = Palingenesis.propose(:rcu, @plank, ps, anchors: ctx.anchors, epsilon: :math.pi())
    end

    seen = for _ <- readers, do: (receive do {:seen, _, s} -> s end)
    # generations only move forward for every reader
    assert Enum.all?(seen, fn s -> s == Enum.sort(s) end)
    # the reader that kept generation 0 still has generation 0's exact weights
    assert Palingenesis.root(held.weights, held.planks) == g0.root
    {:ok, now} = Palingenesis.checkout(:rcu)
    assert now.gen == 2 and length(now.records) == 3
    assert :ok = Palingenesis.verify(now)
  end

  test "lineage: signed records chain from launch; a tampered record, root or key is caught", ctx do
    key = Certificate.keygen()
    launch(:lineage, ctx.spec, ctx.clean, key: key)
    ps = Map.new(names(ctx.clean), fn k -> {k, noisy(ctx.clean[k], 5 + :erlang.phash2(k, 1000), 0.001)} end)
    {:ok, g, _} = Palingenesis.propose(:lineage, @plank, ps, anchors: ctx.anchors, epsilon: :math.pi(), key: key)
    assert :ok = Palingenesis.verify(g, trusted: [key.public])
    assert {:error, _} = Palingenesis.verify(g, trusted: [Certificate.keygen().public])

    [last, first] = g.records
    forged_first = %{first | payload: Map.put(first.payload, :planks, 999)}
    assert {:error, why} = Palingenesis.verify(%{g | records: [last, forged_first]})
    assert why =~ "does not name the record before it"

    swapped = Map.put(g.weights, hd(names(ctx.clean)), ctx.clean[hd(names(ctx.clean))])
    assert {:error, why2} = Palingenesis.verify(%{g | weights: swapped})
    assert why2 =~ "do not hash"
  end
end
