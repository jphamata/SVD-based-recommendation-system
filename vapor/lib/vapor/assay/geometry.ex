defmodule Vapor.Assay.Geometry do
  @moduledoc """
  `assay geometry` — how far apart several models' *beliefs* are, on the same
  items, in the Fisher–Rao metric (`Vapor.InfoGeom`). The pain: accuracy
  compares answers, not beliefs; KL compares beliefs but is no distance (it
  is asymmetric, can be infinite, and violates the triangle inequality), so
  "A is closer to B than to C" means nothing with it. Fisher–Rao is the
  geodesic distance of the one metric invariant under re-parametrisation
  (Čencov), bounded by π, and a metric: distances can be averaged, compared
  and clustered.

  Input: one row per (model, item) with the predicted probabilities over the
  same classes — CSV `model,item,<class>,<class>,…` or JSON lines `{"model",
  "item", "p": [...]}`. Returns the mean distance of every pair with a
  bootstrap interval over items, each model's distance to the geometric
  consensus (the Fréchet mean, item by item) — the outlier is the model the
  others do not believe with — and two checks: the triangle inequality on
  every triple (sanity of the metric as computed), and a **control**: the
  same models with their items shuffled. Two models are only "close" if they
  are closer than models that answer unrelated questions.
  """
  alias Vapor.Assay.Stats
  alias Vapor.InfoGeom, as: G

  def run(text, opts) do
    with {:ok, data} <- read(text) do
      models = data |> Map.keys() |> Enum.sort()
      items = models |> Enum.map(&MapSet.new(Map.keys(data[&1]))) |> Enum.reduce(&MapSet.intersection/2) |> Enum.sort()

      cond do
        length(models) < 2 -> {:error, "at least two models"}
        length(items) < 5 -> {:error, "at least five items common to every model (have #{length(items)})"}
        true -> analyse(data, models, items, opts)
      end
    end
  end

  defp analyse(data, models, items, opts) do
    reps = Keyword.get(opts, :reps, 400)
    rng = Stats.rng(Keyword.get(opts, :seed, 1))
    dist = fn a, b, it -> G.fisher_rao(data[a][it], data[b][it]) end

    {pairs, rng} =
      for(a <- models, b <- models, a < b, do: {a, b})
      |> Enum.map_reduce(rng, fn {a, b}, r ->
        ds = Enum.map(items, &dist.(a, b, &1))
        {ci, r} = boot(ds, reps, r)
        {%{a: a, b: b, mean: mean(ds), ci: ci, kl_asymmetry: kl_gap(data, a, b, items)}, r}
      end)

    consensus = Map.new(items, fn it -> {it, elem(G.frechet_mean(Enum.map(models, &data[&1][it])), 0)} end)
    to_consensus = for m <- models, do: %{model: m, mean: mean(Enum.map(items, &G.fisher_rao(data[m][&1], consensus[&1])))}

    # the control: the same predictions, paired with other items
    {shuffled, _} = Vapor.Entropy.shuffle(items, rng)
    pairing = Map.new(Enum.zip(items, shuffled))
    control = mean(for {a, b} <- Enum.map(pairs, &{&1.a, &1.b}), it <- items, do: G.fisher_rao(data[a][it], data[b][pairing[it]]))

    triangle =
      for a <- models, b <- models, c <- models, a < b, b < c, it <- items, reduce: {0, 0} do
        {n, bad} ->
          {x, y, z} = {dist.(a, b, it), dist.(b, c, it), dist.(a, c, it)}
          ok = x <= y + z + 1.0e-12 and y <= x + z + 1.0e-12 and z <= x + y + 1.0e-12
          {n + 1, if(ok, do: bad, else: bad + 1)}
      end

    closer = Enum.count(pairs, &(Enum.at(&1.ci, 1) < control))

    {:ok,
     %{models: models, items: length(items), classes: length(data[hd(models)][hd(items)]),
       pairs: Enum.sort_by(pairs, & &1.mean), consensus: Enum.sort_by(to_consensus, & &1.mean, :desc), control: control,
       summary: "#{closer} of #{length(pairs)} pairs are closer than the shuffled-item control (#{Stats.f(control)} rad); the farthest from the consensus is #{hd(Enum.sort_by(to_consensus, & &1.mean, :desc)).model}",
       evidence: [
         %{check: "metric", ok: elem(triangle, 1) == 0, detail: "triangle inequality on #{elem(triangle, 0)} (triple, item) cases: #{elem(triangle, 1)} violations"},
         %{check: "control", ok: closer > 0, detail: "#{closer}/#{length(pairs)} pairs' upper bound below the shuffled baseline #{Stats.f(control)}"}
       ]}}
  end

  defp kl_gap(data, a, b, items) do
    gaps = for it <- items, ab = G.kl(data[a][it], data[b][it]), ba = G.kl(data[b][it], data[a][it]), is_float(ab) and is_float(ba), do: abs(ab - ba)
    if gaps == [], do: nil, else: Enum.max(gaps)
  end

  defp boot(xs, reps, rng) do
    t = List.to_tuple(xs)
    n = tuple_size(t)

    {ms, rng} =
      Enum.map_reduce(1..reps, rng, fn _, r ->
        {idx, r} = Enum.map_reduce(1..n, r, fn _, rr -> :rand.uniform_s(n, rr) end)
        {Enum.reduce(idx, 0.0, &(&2 + elem(t, &1 - 1))) / n, r}
      end)

    s = Enum.sort(ms)
    {[Enum.at(s, round(0.025 * (reps - 1))), Enum.at(s, round(0.975 * (reps - 1)))], rng}
  end

  defp mean(xs), do: Enum.sum(xs) / max(length(xs), 1)

  defp read(text) do
    with {:ok, header, rows} <- Stats.table(text) do
      mi = Enum.find_index(header, &(&1 == "model"))
      ii = Enum.find_index(header, &(&1 == "item"))
      pi = Enum.find_index(header, &(&1 == "p"))

      cond do
        mi == nil or ii == nil -> {:error, "columns model and item are required"}
        true ->
          probs =
            Enum.map(rows, fn r ->
              ps = if pi, do: Enum.at(r, pi), else: for({_, i} <- Enum.with_index(header), i not in [mi, ii], do: Stats.num(Enum.at(r, i)))
              ps = Enum.map(List.wrap(ps), &Stats.num/1)
              if Enum.any?(ps, &(&1 == nil or &1 < 0)), do: throw({:bad, "row #{inspect(r) |> String.slice(0, 60)}: probabilities must be numbers ≥ 0"})
              {to_string(Enum.at(r, mi)), to_string(Enum.at(r, ii)), G.normalize(ps)}
            end)

          widths = probs |> Enum.map(&length(elem(&1, 2))) |> Enum.uniq()
          if length(widths) > 1, do: throw({:bad, "every row needs the same number of classes"})
          {:ok, Enum.reduce(probs, %{}, fn {m, it, p}, acc -> Map.update(acc, m, %{it => p}, &Map.put(&1, it, p)) end)}
      end
    end
  catch
    {:bad, why} -> {:error, why}
  end
end
