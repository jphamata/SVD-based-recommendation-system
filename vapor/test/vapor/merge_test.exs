defmodule Vapor.MergeTest do
  @moduledoc "Model fusion: definitions, identities, determinism, receipts, compatibility at the airlock."
  use ExUnit.Case, async: true
  alias Vapor.{Certificate, Lock, Merge, Rejection, Tensor}
  alias Vapor.Model.Config
  import Vapor.TestHelpers

  defp model(seed, arch \\ "qwen2", over \\ %{}) do
    map = tiny_config(arch, over)
    {:ok, c} = Config.from_map(map)
    {:ok, spec, ws} = Lock.from_map(map, tiny_weights(c, seed))
    %{spec: spec, weights: ws}
  end

  defp vals(m, name), do: Tensor.to_floats(m.weights[name])
  @w "model.layers.0.mlp.up_proj.weight"

  test "linear: the weighted mean, rounded once; merge(A, A) = A" do
    {a, b} = {model(1), model(2)}
    {:ok, m} = Merge.merge([a, b], method: :linear, weights: [3, 1])
    want = Enum.zip_with(vals(a, @w), vals(b, @w), &Vapor.CR.to_f32((3 * &1 + &2) / 4))
    assert vals(m, @w) == want
    {:ok, same} = Merge.merge([a, a])
    assert same.weights == a.weights
  end

  test "task arithmetic: base + λ Σ wᵢ(θᵢ − base); λ = 0 gives the base" do
    {base, a, b} = {model(1), model(2), model(3)}
    {:ok, m} = Merge.merge([a, b], method: :task_arithmetic, base: base, lambda: 0.5, weights: [1, 1])
    want = Enum.zip_with([vals(base, @w), vals(a, @w), vals(b, @w)], fn [x, y, z] -> Vapor.CR.to_f32(x + 0.5 * (y - x + (z - x))) end)
    assert vals(m, @w) == want
    {:ok, z} = Merge.merge([a, b], method: :task_arithmetic, base: base, lambda: 0.0)
    assert z.weights == base.weights
  end

  test "slerp: endpoints exact, the midpoint of orthogonal vectors on the arc, host-independent acos" do
    {a, b} = {model(1), model(2)}
    {:ok, s0} = Merge.merge([a, b], method: :slerp, t: 0.0)
    {:ok, s1} = Merge.merge([a, b], method: :slerp, t: 1.0)
    assert s0.weights == a.weights and s1.weights == b.weights

    [x, y] = Merge.slerp([1.0, 0.0], [0.0, 1.0], 0.5)
    assert_in_delta x, :math.sqrt(0.5), 1.0e-15
    assert_in_delta y, :math.sqrt(0.5), 1.0e-15
    for c <- [-1.0, -0.3, 0.0, 0.5, 0.99, 1.0], do: assert_in_delta(Merge.acos(c), :math.acos(c), 1.0e-15)
  end

  test "ties: trimmed task vectors, elected sign, disjoint mean" do
    base = model(1)
    {a, b} = {model(2), model(3)}
    {:ok, m} = Merge.merge([a, b], method: :ties, base: base, density: 1.0)
    # with density 1 and agreeing signs it is the mean of the deltas; with opposite signs, the larger one wins
    for {x, y, z, got} <- Enum.zip([vals(base, @w), vals(a, @w), vals(b, @w), vals(m, @w)]) do
      {ta, tb} = {y - x, z - x}
      want =
        cond do
          ta * tb > 0 -> x + (ta + tb) / 2
          ta + tb > 0 -> x + Enum.max([ta, tb])
          ta + tb < 0 -> x + Enum.min([ta, tb])
          true -> x
        end

      assert_in_delta got, want, 1.0e-6
    end
  end

  test "dare: deterministic by seed, keeps about `density` of each task vector, rescaled" do
    {base, a} = {model(1), model(2)}
    {:ok, m1} = Merge.merge([a], method: :dare_linear, base: base, density: 0.3, seed: 9)
    {:ok, m2} = Merge.merge([a], method: :dare_linear, base: base, density: 0.3, seed: 9)
    {:ok, m3} = Merge.merge([a], method: :dare_linear, base: base, density: 0.3, seed: 10)
    assert m1.weights == m2.weights and m1.weights != m3.weights
    kept = Enum.zip_with(vals(base, @w), vals(m1, @w), &(&1 != &2)) |> Enum.count(& &1)
    assert_in_delta kept / length(vals(base, @w)), 0.3, 0.05
  end

  test "compatibility is checked at the airlock: other topology, other shapes, other config" do
    {a, b} = {model(1), model(2, "qwen2", %{"intermediate_size" => 112})}
    assert {:error, %Rejection{node: {:merge, :config}}} = Merge.merge([a, b])
    assert {:error, %Rejection{node: {:merge, {:weight, _}}}} = Merge.merge([a, b], allow_config_mismatch: true)
    assert {:error, %Rejection{node: {:merge, :base}}} = Merge.merge([a, model(3)], method: :ties)
    assert {:error, %Rejection{node: {:merge, :models}}} = Merge.merge([a, a, a], method: :slerp)
  end

  test "streaming over blocks = the whole-tensor definitions (tensors larger than a block)" do
    spec = %Vapor.Lock.Spec{adapter: :x, family: "x", interface: :map, digest: "d"}
    # more entries than one block (262 144), so blocks meet inside the tensor
    big = fn seed -> %{spec: spec, weights: %{"w" => Tensor.random(:f32, [2100, 128], seed)}} end
    {base, a, b} = {big.(1), big.(2), big.(3)}
    [xb, xa, xbb] = Enum.map([base, a, b], &Tensor.to_floats(&1.weights["w"]))

    # TIES with a magnitude cut: the kept set is the global top-k by (|τ| desc, index asc)
    {:ok, m} = Merge.merge([a, b], method: :ties, base: base, density: 0.3)
    k = round(0.3 * length(xb))
    top = fn xs -> xs |> Enum.zip_with(xb, &(&1 - &2)) |> Enum.with_index() |> Enum.sort_by(fn {t, i} -> {-abs(t), i} end) |> Enum.take(k) |> Map.new(fn {t, i} -> {i, t} end) end
    {ta, tb} = {top.(xa), top.(xbb)}

    want =
      for {x, i} <- Enum.with_index(xb) do
        col = [Map.get(ta, i, 0.0), Map.get(tb, i, 0.0)]
        el = case Enum.sum(col) do s when s > 0 -> 1; s when s < 0 -> -1; _ -> 0 end
        agree = Enum.filter(col, &(&1 != 0.0 and ((&1 > 0 and el == 1) or (&1 < 0 and el == -1))))
        Vapor.CR.to_f32(x + if(agree == [], do: 0.0, else: Enum.sum(agree) / length(agree)))
      end

    assert Tensor.to_floats(m.weights["w"]) == want

    # SLERP's angle is a whole-tensor statistic, taken in a first pass
    {:ok, s} = Merge.merge([a, b], method: :slerp, t: 0.3)
    assert Tensor.to_floats(s.weights["w"]) == Enum.map(Merge.slerp(xa, xbb, 0.3), &Vapor.CR.to_f32/1)
  end

  test "DARE across blocks: the generator jumps ahead (a counter), the same draws as one sequential pass; any schedule, same bits" do
    spec = %Vapor.Lock.Spec{adapter: :x, family: "x", interface: :map, digest: "d"}
    big = fn seed -> %{spec: spec, weights: %{"w" => Tensor.random(:f32, [2100, 128], seed)}} end
    {base, a} = {big.(1), big.(2)}
    {:ok, m} = Merge.merge([a], method: :dare_linear, base: base, density: 0.4, seed: 5)
    {:ok, m1} = Merge.merge([a], method: :dare_linear, base: base, density: 0.4, seed: 5, concurrency: 1)
    assert m.weights == m1.weights

    key = Vapor.Modal.Rng.key({:dare, 5, 0, "w"})
    {want, _} =
      Enum.zip(Tensor.to_floats(a.weights["w"]), Tensor.to_floats(base.weights["w"]))
      |> Enum.map_reduce(key, fn {x, y}, st ->
        {v, st} = Tensor.splitmix(st)
        u = Bitwise.bsr(v, 11) / 9_007_199_254_740_992
        {Vapor.CR.to_f32(y + 1.0 * (0.0 + 1.0 * if(u < 0.4, do: (x - y) / 0.4, else: 0.0))), st}
      end)

    assert Tensor.to_floats(m.weights["w"]) == want
  end

  test "a NaN or an infinity in a weight is a rejection naming the tensor, not a crash" do
    {a, b} = {model(1), model(2)}
    bad = Map.update!(b.weights, @w, fn t -> %{t | data: <<0, 0, 192, 127>> <> binary_part(t.data, 4, byte_size(t.data) - 4)} end)
    assert {:error, %Rejection{node: {:merge, {:weight, @w}}}} = Merge.merge([a, %{b | weights: bad}])
  end

  test "diagnose: unrelated networks, a dense pair and small deltas told apart from the weights" do
    {a, b} = {model(1), model(2)}
    assert %{regime: :unrelated, pairs: [%{weight_cosine: c}]} = Merge.diagnose([a, b])
    assert abs(c) < 0.2

    # fine-tunes: the base plus small perturbations
    nudge = fn m, seed, eps ->
      %{m | weights: Map.new(m.weights, fn
        {k, %Tensor{dtype: :f32} = t} when is_binary(k) ->
          n = Tensor.random(:f32, t.shape, seed + :erlang.phash2(k, 1000))
          {k, Tensor.from_list(:f32, t.shape, Enum.zip_with(Tensor.to_floats(t), Tensor.to_floats(n), &(&1 + eps * &2)))}
        kv -> kv
      end)}
    end

    small = Merge.diagnose([nudge.(a, 10, 0.01), nudge.(a, 20, 0.01)], base: a)
    assert small.regime == :small_deltas and hd(small.models).relative_delta < 0.05
    dense = Merge.diagnose([nudge.(a, 10, 0.15), nudge.(a, 20, 0.15)], base: a)
    assert dense.regime == :dense and Enum.any?(dense.advice, &(&1 =~ "TIES and DARE"))
  end

  test "regmean on planted bigrams: one-hot inputs make the Gram diagonal, and the fusion the count-weighted mean, exactly" do
    alias Vapor.Quality.Planted
    al = Planted.alphabet()
    {ta, tb} = {"o gato subiu no telhado e o cao ficou olhando", "the cat climbed the roof and the dog kept looking"}
    mk = fn txt -> m = Planted.bigram(al.encode.(txt), al.size); {:ok, s, w} = Lock.from_map(m.config, m.weights); %{spec: s, weights: w} end
    {ma, mb} = {mk.(ta), mk.(tb)}
    {:ok, ga} = Merge.calibrate(ma, [al.encode.(ta)])
    {:ok, gb} = Merge.calibrate(mb, [al.encode.(tb)])
    assert Map.has_key?(ga.grams, "lm_head.weight") and ga.tokens == byte_size(ta)
    {:ok, m} = Merge.merge([ma, mb], method: :regmean, grams: [ga, gb], alpha: 1.0, ridge: 1.0e-4)
    assert [_] = m.receipt.payload.params.grams |> Enum.take(1)

    # W*[:, i] = (gA_ii·A[:, i] + gB_ii·B[:, i] + λ·mean[:, i]) / (gA_ii + gB_ii + λ)
    [vp, d] = ma.weights["lm_head.weight"].shape
    {wa, wb, got} = {ma, mb, m} |> Tuple.to_list() |> Enum.map(&(&1.weights["lm_head.weight"] |> Tensor.to_floats() |> Enum.chunk_every(d))) |> List.to_tuple()
    diag = fn g, i -> Vapor.Linalg.at(g.grams["lm_head.weight"].g, i, i) end
    lam = 1.0e-4 * (Enum.sum(for i <- 0..(d - 1), do: diag.(ga, i) + diag.(gb, i)) / d)

    for j <- 0..(vp - 1), i <- 0..(d - 1) do
      {x, y} = {Enum.at(Enum.at(wa, j), i), Enum.at(Enum.at(wb, j), i)}
      want = (diag.(ga, i) * x + diag.(gb, i) * y + lam * (x + y) / 2) / (diag.(ga, i) + diag.(gb, i) + lam)
      assert_in_delta Enum.at(Enum.at(got, j), i), want, 2.0e-5 * max(1.0, abs(want))
    end
  end

  test "select: every candidate scored, the lowest kept, the table and the evaluation digest in a receipt" do
    {base, a, b} = {model(1), model(2), model(3)}
    target = Tensor.to_floats(a.weights[@w])
    # score = distance of one tensor to model a: linear (½a + ½b) is farther than slerp at t = 0.1
    score = fn m -> Enum.zip_reduce(Tensor.to_floats(m.weights[@w]), target, 0.0, fn x, y, s -> s + (x - y) * (x - y) end) end
    cands = [linear: [method: :linear], "slerp 0.1": [method: :slerp, t: 0.1], bad: [method: :ties]]
    assert {:ok, sel} = Merge.select([a, b], cands, score, eval: "abc", key: Certificate.keygen())
    assert sel.best == "slerp 0.1"
    assert [%{label: "linear"}, %{label: "slerp 0.1"}, %{label: "bad", score: nil, refused: _}] = sel.table
    assert sel.receipt.payload.kind == "vapor.merge.select/1" and sel.receipt.payload.eval == "abc"
    assert sel.receipt.payload.merge == sel.merged.receipt.payload
    _ = base
  end

  test "the receipt: signed, co-signable by an independent node, bound to the output weights" do
    {a, b} = {model(1), model(2)}
    k1 = Certificate.keygen()
    k2 = Certificate.keygen()
    {:ok, m} = Merge.merge([a, b], method: :slerp, t: 0.3, key: k1)
    {:ok, again} = Merge.merge([a, b], method: :slerp, t: 0.3)
    assert {:ok, both} = Certificate.cosign(m.receipt, again.receipt, k2)
    assert :ok = Certificate.verify(both, [k1.public, k2.public], 2)
    assert :ok = Merge.verify_receipt(m.receipt, m.weights)
    tampered = Map.update!(m.weights, @w, fn t -> %{t | data: <<0::32>> <> binary_part(t.data, 4, byte_size(t.data) - 4)} end)
    assert {:error, :mismatch} = Merge.verify_receipt(m.receipt, tampered)
    assert m.receipt.payload.method == "slerp" and length(m.receipt.payload.inputs) == 2
  end
end
