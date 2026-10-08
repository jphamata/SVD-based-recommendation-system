defmodule Vapor.Assay.Detect do
  @moduledoc """
  The Assay's `detect` tool: **cgF1**, the classification-gated F1 of
  SAM 3 (*Segment Anything with Concepts*, Meta, 2025, §E.3), for detectors
  asked "find every X" — including images that contain no X at all.

  A detector is judged on two separate questions, then on their product:

  * **localisation, pmF1** — on the datapoints that contain the concept,
    predictions are matched to the truth by an **optimal bipartite
    matching** that maximises the total IoU. At each τ ∈ {0.50, 0.55, …,
    0.95}, a match with IoU ≥ τ is a true positive, every other prediction
    a false positive, every other truth a false negative. Counts are summed
    over datapoints (micro), F1 is taken per τ, and pmF1 is their mean;
  * **presence, IL_MCC** — per datapoint, did it predict anything?
    Matthews' correlation between that and "the concept is there", over
    every datapoint, the negatives included;
  * **cgF1 = 100 · pmF1 · IL_MCC**, in [−100, 100].

  Only predictions with a confidence above 0.5 count, as in the paper: a
  fixed threshold replaces average precision and rewards calibration.

  **The matching is exact and certified.** Boxes are read as rationals, so
  their IoUs are exact. The matching is the balanced-assignment linear
  program over the pairs with positive IoU, solved by the rational simplex
  (`Vapor.Logic.LP`). Its constraint matrix is a bipartite incidence
  matrix, totally unimodular, so the optimum is a 0/1 matching, and its dual
  certificate is checked on every datapoint.

  **Two choices the paper leaves open, made here and reported:** a truth
  matched below τ counts as a false negative (`TP + FN = truths`; otherwise
  F1 is inflated), and IL_MCC is 0 when its denominator is 0.

  Input: JSON, one datapoint per line or an array — `{"truth": [[x0, y0,
  x1, y1], …], "pred": [[x0, y0, x1, y1, score], …]}` (`gt` is accepted
  for `truth`). The output adds a bootstrap interval of cgF1 over
  datapoints; the evidence check is that its lower end is above 0.
  """
  alias Vapor.Assay.Stats
  alias Vapor.Logic.LP

  @taus for i <- 0..9, do: {50 + 5 * i, 100}

  @doc "Run on JSON text. Options: `seed`, `reps` (bootstrap, 1000)."
  def run(text, opts \\ []) do
    with {:ok, points} <- parse(text),
         {:ok, scored} <- score_all(points) do
      {:ok, summarise(scored, opts)}
    end
  end

  # ---------------------------------------------------------------- input --

  defp parse(text) do
    t = String.trim(text)

    docs =
      case Vapor.JSON.decode(t) do
        {:ok, l} when is_list(l) -> {:ok, l}
        _ -> t |> String.split("\n", trim: true) |> Enum.reduce_while({:ok, []}, fn line, {:ok, acc} ->
               case Vapor.JSON.decode(line) do
                 {:ok, %{} = m} -> {:cont, {:ok, acc ++ [m]}}
                 _ -> {:halt, {:error, "each line is a JSON object {truth, pred}: #{String.slice(line, 0, 60)}"}}
               end
             end)
      end

    with {:ok, ds} <- docs do
      ds
      |> Enum.with_index(1)
      |> Enum.reduce_while({:ok, []}, fn {d, i}, {:ok, acc} ->
        case point(d) do
          {:ok, p} -> {:cont, {:ok, acc ++ [p]}}
          {:error, why} -> {:halt, {:error, "datapoint #{i}: #{why}"}}
        end
      end)
      |> then(fn
        {:ok, []} -> {:error, "no datapoints"}
        other -> other
      end)
    end
  end

  defp point(%{} = d) do
    truth = d["truth"] || d["gt"] || []
    pred = d["pred"] || []

    with {:ok, ts} <- boxes(truth, 4),
         {:ok, ps} <- boxes(pred, 5) do
      kept = for [x0, y0, x1, y1, s] <- ps, LP.qcmp(s, {1, 2}) > 0, do: [x0, y0, x1, y1]
      {:ok, %{truth: ts, pred: kept}}
    end
  end

  defp point(_), do: {:error, "an object {truth, pred}"}

  defp boxes(list, n) when is_list(list) do
    Enum.reduce_while(list, {:ok, []}, fn b, {:ok, acc} ->
      if is_list(b) and length(b) == n and Enum.all?(b, &is_number/1) do
        [x0, y0, x1, y1 | _] = r = Enum.map(b, &LP.rat/1)
        if LP.qcmp(x1, x0) > 0 and LP.qcmp(y1, y0) > 0,
          do: {:cont, {:ok, acc ++ [r]}},
          else: {:halt, {:error, "a box with x1 > x0 and y1 > y0: #{inspect(b)}"}}
      else
        {:halt, {:error, "boxes of #{n} numbers ([x0, y0, x1, y1#{if n == 5, do: ", score"}]): #{inspect(b)}"}}
      end
    end)
  end

  defp boxes(_, _), do: {:error, "a list of boxes"}

  # -------------------------------------------------------------- matching --

  @doc "The exact IoU of two boxes `[x0, y0, x1, y1]` of rationals."
  def iou([a0, b0, a1, b1], [c0, d0, c1, d1]) do
    w = LP.qsub(qmin(a1, c1), qmax(a0, c0))
    h = LP.qsub(qmin(b1, d1), qmax(b0, d0))

    if LP.qsign(w) <= 0 or LP.qsign(h) <= 0 do
      {0, 1}
    else
      inter = LP.qmul(w, h)
      area = fn x0, y0, x1, y1 -> LP.qmul(LP.qsub(x1, x0), LP.qsub(y1, y0)) end
      LP.qdiv(inter, LP.qsub(LP.qadd(area.(a0, b0, a1, b1), area.(c0, d0, c1, d1)), inter))
    end
  end

  defp qmin(x, y), do: if(LP.qcmp(x, y) <= 0, do: x, else: y)
  defp qmax(x, y), do: if(LP.qcmp(x, y) >= 0, do: x, else: y)

  @doc """
  The maximum-total-IoU matching of predictions to truths, exactly:
  `{:ok, [{pred, truth, iou}], certified?}`.
  """
  def match(preds, truths) do
    pairs = for {p, i} <- Enum.with_index(preds), {t, j} <- Enum.with_index(truths), v = iou(p, t), LP.qsign(v) > 0, do: {i, j, v}

    cond do
      pairs == [] ->
        {:ok, [], true}

      length(pairs) > 600 ->
        {:error, "at most 600 overlapping prediction–truth pairs per datapoint (#{length(pairs)})"}

      true ->
        var = fn i, j -> "m#{i}_#{j}" end
        vars = for {i, j, _} <- pairs, do: var.(i, j)
        rows =
          (for i <- pairs |> Enum.map(&elem(&1, 0)) |> Enum.uniq(), do: {for({^i, j, _} <- pairs, into: %{}, do: {var.(i, j), {1, 1}}), :le, {1, 1}}) ++
            (for j <- pairs |> Enum.map(&elem(&1, 1)) |> Enum.uniq(), do: {for({i, ^j, _} <- pairs, into: %{}, do: {var.(i, j), {1, 1}}), :le, {1, 1}})

        {:ok, r} = LP.solve(%{sense: :max, vars: vars, c: Map.new(pairs, fn {i, j, v} -> {var.(i, j), v} end), c0: {0, 1}, rows: rows, free: []})

        if Enum.all?(r.x, fn {_, x} -> x in [{0, 1}, {1, 1}] end),
          do: {:ok, for({i, j, v} <- pairs, r.x[var.(i, j)] == {1, 1}, do: {i, j, v}), r.check.accepted},
          else: {:error, "a fractional vertex (impossible for a bipartite matching): the solver is wrong"}
    end
  end

  # ---------------------------------------------------------------- scores --

  defp score_all(points) do
    Enum.reduce_while(points, {:ok, []}, fn p, {:ok, acc} ->
      case match(p.pred, p.truth) do
        {:ok, m, cert} ->
          n = length(p.pred)
          k = length(p.truth)
          per_tau = for tau <- @taus, do: (tp = Enum.count(m, fn {_, _, v} -> LP.qcmp(v, tau) >= 0 end); {tp, n - tp, k - tp})
          {:cont, {:ok, acc ++ [%{positive: k > 0, predicted: n > 0, counts: per_tau, certified: cert}]}}

        {:error, why} ->
          {:halt, {:error, why}}
      end
    end)
  end

  defp metrics(scored) do
    pos = Enum.filter(scored, & &1.positive)

    f1s =
      for t <- 0..9 do
        {tp, fp, fn_} = Enum.reduce(pos, {0, 0, 0}, fn s, {a, b, c} -> {x, y, z} = Enum.at(s.counts, t); {a + x, b + y, c + z} end)
        if 2 * tp + fp + fn_ == 0, do: 0.0, else: 2 * tp / (2 * tp + fp + fn_)
      end

    pm = if pos == [], do: nil, else: Enum.sum(f1s) / 10
    c = fn f -> Enum.count(scored, f) end
    {tp, fn_, fp, tn} = {c.(&(&1.positive and &1.predicted)), c.(&(&1.positive and not &1.predicted)), c.(&(not &1.positive and &1.predicted)), c.(&(not &1.positive and not &1.predicted))}
    den = :math.sqrt((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_))
    mcc = if den == 0, do: 0.0, else: (tp * tn - fp * fn_) / den
    %{pm_f1: pm, f1_by_tau: f1s, il_mcc: mcc, cg_f1: if(pm, do: 100 * pm * mcc), presence: %{tp: tp, fn: fn_, fp: fp, tn: tn}}
  end

  defp summarise(scored, opts) do
    m = metrics(scored)
    reps = Keyword.get(opts, :reps, 1000)
    pts = List.to_tuple(scored)
    boot = Stats.bootstrap(length(scored), reps, Keyword.get(opts, :seed, 1), fn idx -> metrics(Enum.map(idx, &elem(pts, &1))).cg_f1 end)
    ci = if tuple_size(boot) > 0, do: [Stats.pct(boot, 0.025), Stats.pct(boot, 0.975)], else: [nil, nil]
    certified = Enum.all?(scored, & &1.certified)

    Map.merge(m, %{
      datapoints: length(scored), positives: Enum.count(scored, & &1.positive), cg_f1_ci95: ci,
      matching: "optimal (maximum total IoU), exact rationals; dual certificate checked on #{if certified, do: "every", else: "NOT every"} datapoint",
      choices: ["a truth matched below τ is a false negative", "IL_MCC = 0 when its denominator is 0", "predictions with confidence ≤ 0.5 are dropped"],
      says: "cgF1 #{Stats.f(m.cg_f1)} (95% #{Stats.f(Enum.at(ci, 0))} … #{Stats.f(Enum.at(ci, 1))}) = 100 · pmF1 #{Stats.f(m.pm_f1)} · IL_MCC #{Stats.f(m.il_mcc)}",
      evidence: [
        %{check: "the matching", ok: certified, detail: "optimal by the simplex's dual certificate on every datapoint"},
        %{check: "better than chance", ok: is_number(Enum.at(ci, 0)) and Enum.at(ci, 0) > 0,
          detail: "the bootstrap interval of cgF1 over datapoints lies above 0"}
      ]
    })
  end

  @doc "A deterministic example: 40 datapoints, 24 with the concept; a detector that is mostly right."
  def example do
    h = fn k -> Vapor.Alembic.Builtins.hash01([:det | k]) end

    for i <- 1..40 do
      n = if i <= 24, do: 1 + trunc(3 * h.([i, :n])), else: 0
      truth = for k <- 1..n//1, do: (x = round(80 * h.([i, k, :x])); y = round(80 * h.([i, k, :y])); [x, y, x + 10 + round(10 * h.([i, k, :w])), y + 10 + round(10 * h.([i, k, :h]))])
      hits = for {[x0, y0, x1, y1], k} <- Enum.with_index(truth), h.([i, k, :miss]) > 0.15, do: (j = round(4 * h.([i, k, :j])) - 2; [x0 + j, y0 - j, x1 + j, y1, Float.round(0.55 + 0.4 * h.([i, k, :s]), 2)])
      false_alarm = if h.([i, :fa]) < 0.2, do: [[5, 5, 15, 15, Float.round(0.5 + 0.3 * h.([i, :fs]), 2)]], else: []
      low = [[50, 50, 60, 60, 0.3]]
      Vapor.JSON.encode(%{"truth" => truth, "pred" => hits ++ false_alarm ++ low})
    end
    |> Enum.join("\n")
  end
end
