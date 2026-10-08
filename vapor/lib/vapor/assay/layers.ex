defmodule Vapor.Assay.Layers do
  @moduledoc """
  The Assay's `layers` tool: **which layer's features should a probe
  read?** The question of the Perception Encoder paper (Bolya et al.,
  Meta FAIR, 2025: "the best visual embeddings are not at the output of
  the network"), as a protocol any encoder can be put through.

  Input (JSON): `{"labels": [y, …], "layers": [[[f, …] per item] per
  layer]}`, the layers in depth order, the last one the network's output.

  Protocol:

  1. items are split by a keyed hash into train (60%), validation (20%)
     and test (20%), so the split does not depend on their order;
  2. on every layer, a linear probe: one-vs-rest ridge regression on
     standardised features (closed form), λ from a grid;
  3. the layer **and** λ are chosen on validation. The test split is read
     once, after the choice;
  4. reported on test: the chosen layer against the last one, with an
     exact McNemar test on the items where they disagree, and against
     chance (the majority class's rate), with an exact binomial test;
  5. the **control**: the same protocol on labels shuffled across items
     must find no layer above chance.

  The evidence checks are the signal (the chosen layer beats chance at
  α = 0.05) and the control (with shuffled labels it does not).
  """
  alias Vapor.Assay.Stats
  alias Vapor.Dense

  @lambdas [1.0e-3, 1.0e-1, 10.0]

  @doc "Run on JSON text. Options: `seed`."
  def run(text, opts \\ []) do
    with {:ok, %{"labels" => ys, "layers" => [_ | _] = layers}} when is_list(ys) <- Vapor.JSON.decode(text),
         :ok <- shape(ys, layers) do
      {:ok, evaluate(layers, ys, Keyword.get(opts, :seed, 1))}
    else
      {:ok, _} -> {:error, ~s(a JSON object {"labels": [...], "layers": [[[...], ...], ...]})}
      {:error, _} = e -> e
    end
  end

  defp shape(ys, layers) do
    n = length(ys)

    cond do
      n < 20 -> {:error, "at least 20 labelled items"}
      not Enum.all?(layers, &(is_list(&1) and length(&1) == n)) -> {:error, "every layer has one feature row per label"}
      not Enum.all?(layers, fn l -> d = length(hd(l)); d > 0 and Enum.all?(l, &(is_list(&1) and length(&1) == d and Enum.all?(&1, fn x -> is_number(x) end))) end) ->
        {:error, "rows of numbers, of one width per layer"}
      true -> :ok
    end
  end

  @doc "The protocol on decoded data (see the moduledoc)."
  def evaluate(layers, ys, seed) do
    classes = ys |> Enum.uniq() |> Enum.sort()
    y = Enum.map(ys, fn v -> Enum.find_index(classes, &(&1 == v)) end)
    split = Enum.map(0..(length(ys) - 1), fn i -> part(:erlang.phash2({seed, i}, 1000)) end)
    main = protocol(layers, y, split, length(classes))

    {perm, _} = shuffle(y, Stats.rng(seed + 7919))
    control = protocol(layers, perm, split, length(classes))

    chance_p = binom_greater(main.chosen.correct, main.test_n, main.chance)
    ctrl_p = binom_greater(control.chosen.correct, control.test_n, control.chance)
    b = main.disagree.chosen_only
    c = main.disagree.last_only

    %{classes: length(classes), items: length(ys), layers: length(layers),
      split: %{train: main.train_n, validation: main.val_n, test: main.test_n},
      validation_accuracy: main.val_acc, chosen_layer: main.chosen.layer, lambda: main.chosen.lambda,
      test: %{chosen: main.chosen.acc, last: main.last.acc, chance: main.chance},
      chosen_vs_last: %{chosen_only: b, last_only: c, p_mcnemar: Stats.binom_two_sided(b, b + c)},
      control: %{chosen_layer: control.chosen.layer, test: control.chosen.acc, chance: control.chance, p_above_chance: ctrl_p},
      says: "layer #{main.chosen.layer} of #{length(layers) - 1} (chosen on validation): test accuracy #{Stats.f(main.chosen.acc)} " <>
              "against #{Stats.f(main.last.acc)} at the output and #{Stats.f(main.chance)} by chance",
      evidence: [
        %{check: "signal", ok: chance_p <= 0.05, detail: "the chosen layer beats the majority-class rate on test: exact binomial p = #{Stats.f(chance_p)}"},
        %{check: "control", ok: ctrl_p > 0.05, detail: "with labels shuffled, the same protocol finds layer #{control.chosen.layer} at #{Stats.f(control.chosen.acc)} (p = #{Stats.f(ctrl_p)} against chance)"}
      ]}
  end

  defp part(h) when h < 600, do: :train
  defp part(h) when h < 800, do: :val
  defp part(_), do: :test

  defp protocol(layers, y, split, k) do
    idx = fn p -> for {s, i} <- Enum.with_index(split), s == p, do: i end
    {tr, va, te} = {idx.(:train), idx.(:val), idx.(:test)}
    fits = for {feats, l} <- Enum.with_index(layers), lam <- @lambdas, do: {l, lam, fit(feats, y, tr, lam, k)}
    scored = for {l, lam, model} <- fits, do: {l, lam, model, accuracy(model, Enum.at(layers, l), y, va)}

    {l, lam, model, _} = Enum.max_by(scored, fn {l, lam, _, a} -> {a, -l, lam} end)
    last = length(layers) - 1
    {_, _, last_model, _} = scored |> Enum.filter(&(elem(&1, 0) == last)) |> Enum.max_by(fn {_, lam, _, a} -> {a, lam} end)
    right = fn model, layer -> for i <- te, do: predict(model, Enum.at(Enum.at(layers, layer), i)) == Enum.at(y, i) end
    rc = right.(model, l)
    rl = right.(last_model, last)
    counts = Enum.frequencies(for i <- tr, do: Enum.at(y, i))
    majority = counts |> Enum.max_by(fn {_, n} -> n end) |> elem(0)

    %{train_n: length(tr), val_n: length(va), test_n: length(te),
      val_acc: for(layer <- 0..last, do: scored |> Enum.filter(&(elem(&1, 0) == layer)) |> Enum.map(&elem(&1, 3)) |> Enum.max()),
      chosen: %{layer: l, lambda: lam, acc: rate(rc), correct: Enum.count(rc, & &1)},
      last: %{acc: rate(rl)},
      chance: rate(for i <- te, do: Enum.at(y, i) == majority),
      disagree: %{chosen_only: Enum.zip(rc, rl) |> Enum.count(&(&1 == {true, false})), last_only: Enum.zip(rc, rl) |> Enum.count(&(&1 == {false, true}))}}
  end

  defp rate([]), do: 0.0
  defp rate(bs), do: Enum.count(bs, & &1) / length(bs)

  # one-vs-rest ridge on standardised features, with a bias column (not penalised)
  defp fit(feats, y, tr, lam, k) do
    rows = for i <- tr, do: Enum.at(feats, i)
    d = length(hd(rows))
    mu = for j <- 0..(d - 1), do: Stats.mean(Enum.map(rows, &Enum.at(&1, j)))
    sd = for j <- 0..(d - 1), do: max(Stats.sd(Enum.map(rows, &Enum.at(&1, j))), 1.0e-12)
    x = Enum.map(rows, &[1.0 | standard(&1, mu, sd)])
    t = for i <- tr, do: for(c <- 0..(k - 1), do: if(Enum.at(y, i) == c, do: 1.0, else: -1.0))
    xt = Dense.transpose(x)
    a = Dense.matmul(xt, x) |> Enum.with_index() |> Enum.map(fn {r, i} -> if i == 0, do: r, else: List.update_at(r, i, &(&1 + lam)) end)
    {:ok, w} = Dense.solve(a, Dense.matmul(xt, t))
    %{mu: mu, sd: sd, w: w}
  end

  defp standard(r, mu, sd), do: Enum.zip_with([r, mu, sd], fn [v, m, s] -> (v - m) / s end)

  defp predict(%{mu: mu, sd: sd, w: w}, row) do
    x = [1.0 | standard(row, mu, sd)]
    scores = for col <- Dense.transpose(w), do: Enum.zip_with(x, col, &(&1 * &2)) |> Enum.sum()
    scores |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1)
  end

  defp accuracy(model, feats, y, items), do: rate(for(i <- items, do: predict(model, Enum.at(feats, i)) == Enum.at(y, i)))

  defp shuffle(list, rng) do
    {keys, rng} = Enum.map_reduce(list, rng, fn _, r -> :rand.uniform_s(r) end)
    {Enum.zip(keys, list) |> Enum.sort() |> Enum.map(&elem(&1, 1)), rng}
  end

  # P(X ≥ k) for X ~ Binomial(n, p), exactly
  defp binom_greater(_k, 0, _p), do: 1.0

  defp binom_greater(k, n, p) do
    k..n//1 |> Enum.map(fn j -> choose(n, j) * :math.pow(p, j) * :math.pow(1 - p, n - j) end) |> Enum.sum() |> min(1.0)
  end

  defp choose(n, j), do: Enum.reduce(1..j//1, 1, fn i, acc -> div(acc * (n - j + i), i) end)

  @doc """
  A planted network (the example): items in ℝ⁸, three classes read off a
  linear map; layer 0 is the input with noise, layers 1–2 keep the classes
  linearly readable, layer 3 squares the class coordinates (folding them,
  so no linear probe can tell a class from its mirror), layer 4 mixes
  again: the classes are best read in the middle, not at the output.
  """
  def example(n \\ 300, seed \\ 3) do
    rng = Stats.rng(seed)
    {xs, rng} = Enum.map_reduce(1..n, rng, fn _, r -> Enum.map_reduce(1..8, r, fn _, rr -> :rand.normal_s(rr) end) end)
    {m1, rng} = mat(8, 8, rng)
    {m2, rng} = mat(8, 8, rng)
    {m4, _} = mat(8, 8, rng)
    label = fn x -> x |> Enum.take(3) |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1) end
    l0 = Enum.map(xs, fn x -> Enum.map(x, &(&1 + 0.8 * :math.sin(7.0 * &1))) end)
    l1 = Enum.map(xs, fn x -> x |> Enum.zip_with(Enum.map(m1, &dot(&1, x)), fn a, b -> a + 0.3 * :math.tanh(b) end) end)
    l2 = Enum.map(l1, fn h -> h |> Enum.zip_with(Enum.map(m2, &dot(&1, h)), fn a, b -> a + 0.2 * :math.tanh(b) end) end)
    l3 = Enum.map(l2, fn h -> Enum.with_index(h) |> Enum.map(fn {v, i} -> if i < 3, do: v * v, else: v end) end)
    l4 = Enum.map(l3, fn h -> Enum.map(m4, &:math.tanh(dot(&1, h))) end)
    round4 = fn l -> Enum.map(l, fn r -> Enum.map(r, &Float.round(&1, 4)) end) end
    Vapor.JSON.encode(%{"labels" => Enum.map(xs, label), "layers" => Enum.map([l0, l1, l2, l3, l4], round4)})
  end

  defp mat(r, c, rng), do: Enum.map_reduce(1..r, rng, fn _, rr -> Enum.map_reduce(1..c, rr, fn _, q -> {v, q} = :rand.normal_s(q); {v / :math.sqrt(c), q} end) end)
  defp dot(a, b), do: Enum.zip_with(a, b, &(&1 * &2)) |> Enum.sum()
end
