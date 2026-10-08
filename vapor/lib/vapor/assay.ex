defmodule Vapor.Assay do
  @moduledoc """
  **Assay** — the test of purity: an AI research suite whose every answer
  separates signal from noise (docs/ASSAY.md). It is what vapor always
  did for its own outputs — calibrated gates, controls — handed to the
  person evaluating models.

  | tool | the pain | what it returns |
  |---|---|---|
  | `compare` | "B beats A by 1.2 points" with no error bar | paired bootstrap CI, sign-flip permutation p, McNemar for 0/1 scores, effect size, the **minimum detectable effect** at this n and the n needed for the observed one |
  | `leaderboard` | rankings that are mostly noise | per-system CI, the bootstrap distribution of each rank (P(#1), rank interval), Holm-adjusted pairwise tests, tie groups |
  | `contamination` | test items seen in training | 13-gram overlap per item (the GPT-3 method), the clean subset, and — with scores — whether contamination inflated them |
  | `dedup` | near-duplicates in a corpus | MinHash + LSH candidates verified by exact Jaccard; clusters; a filter for pipelines |
  | `scaling` | extrapolated scaling laws without uncertainty | L = E + A/N^α (+ B/D^β) by Huber loss on log-loss with many starts, bootstrap CIs, compute-optimal exponents, and a **leave-the-largest-out** check |
  | `calibration` | ECE that is biased at small n | equal-mass ECE with its CI, the ECE a perfectly calibrated model would show at the same confidences (the bias floor), Brier, NLL, temperature/Platt scaling fitted on one half and judged on the other |
  | `agreement` | annotations nobody checked | Cohen's κ, Fleiss' κ, Krippendorff's α with bootstrap CIs |
  | `geometry` | "model A is closer to B than to C" measured with KL, which is not a distance | Fisher–Rao distances between models' predictive distributions with bootstrap CIs, the geometric consensus (Fréchet mean) and each model's distance to it, the triangle inequality checked, a shuffled-item control |
  | `judge` | LLM-as-a-judge position bias | consistency under order swap, first-slot preference with an exact binomial test |
  | `layers` | probing a network's output when the best features are inside it (the Perception Encoder's finding) | a linear probe per layer, the layer chosen on validation and reported on test against the output and chance, a shuffled-label control (`Vapor.Assay.Layers`) |
  | `detect` | detectors scored by AP, which hides both calibration and false alarms on images without the object | SAM 3's cgF1 = pmF1 (optimal matching, exact and certified) × IL_MCC (presence), with a bootstrap interval (`Vapor.Assay.Detect`) |

  Inputs are CSV/TSV with a header or JSON lines — from a file or a pipe.
  """
  alias Vapor.Assay.{Data, Scaling, Stats}
  import Stats, only: [f: 1]

  defp table do
  [
    {"geometry", "several models' predicted distributions on the same items (columns model, item, then one per class)",
     "model,item,p_yes,p_no,p_maybe\n" <> Enum.map_join(for(m <- ~w(base tuned other), i <- 1..30, do: {m, i}), "\n", fn {m, i} ->
       h = fn k -> Vapor.Alembic.Builtins.hash01([:geo, m, i, k]) end
       base = [0.2 + h.(1), 0.2 + h.(2), 0.1]
       ps = case m do "base" -> base; "tuned" -> Enum.map(base, &(&1 + 0.05 * h.(3))); "other" -> [0.1 + h.(4), 0.5, 0.2 + h.(5)] end
       "#{m},q#{i}," <> Enum.map_join(ps, ",", &Float.round(&1, 4))
     end)},
    {"compare", "two systems on the same items (columns a, b)",
     "id,a,b\n" <> Enum.map_join(1..60, "\n", fn i -> "q#{i},#{if rem(i * 7, 10) < 6, do: 1, else: 0},#{if rem(i * 7 + 3, 10) < 7, do: 1, else: 0}" end)},
    {"leaderboard", "many systems on the same items (one column per system)",
     "item,sys_a,sys_b,sys_c,sys_d\n" <> Enum.map_join(1..300, "\n", fn i -> "#{i},#{b(i, 1, 0.60)},#{b(i, 2, 0.63)},#{b(i, 3, 0.74)},#{b(i, 4, 0.52)}" end)},
    {"contamination", "test items against a training corpus (JSON: {\"train\": [...], \"test\": [...], \"scores\": [...]})",
     ~s({"train": ["the quick brown fox jumps over the lazy dog near the river bank on a sunny day in may", "an unrelated document about the history of glassblowing in venice and its guilds"], "test": ["the quick brown fox jumps over the lazy dog near the river bank on a sunny day", "what is the capital of portugal and why did it move", "how many legs does a spider have"], "scores": [1, 0, 1]})},
    {"dedup", "near-duplicate lines (or a JSON lines field `text`)",
     "the cat sat on the mat and looked out of the window\nthe cat sat on the mat and looked out of the window!\na completely different sentence about prime numbers and their gaps\nthe cat sat on a mat and looked out of the window\nprime numbers and their gaps: a completely different sentence"},
    {"scaling", "runs with params N (and tokens D) and final loss L",
     "N,D,L\n" <> Enum.map_join(for(n <- [1.0e7, 3.0e7, 1.0e8, 3.0e8, 1.0e9], d <- [2.0e9, 1.0e10, 5.0e10], do: {n, d}), "\n", fn {n, d} -> "#{:erlang.float_to_binary(n, [:short])},#{:erlang.float_to_binary(d, [:short])},#{Float.round((1.69 + 406.4 / :math.pow(n, 0.34) + 410.7 / :math.pow(d, 0.28)) * (1 + 0.004 * (Vapor.Alembic.Builtins.hash01([:sc, n, d]) - 0.5)), 4)}" end)},
    {"calibration", "confidence and correctness per item (columns p, correct)",
     "p,correct\n" <> Enum.map_join(1..300, "\n", fn i -> (p = 0.5 + 0.49 * Vapor.Alembic.Builtins.hash01([:p, i]); "#{Float.round(p, 3)},#{if Vapor.Alembic.Builtins.hash01([:y, i]) < p - 0.1, do: 1, else: 0}") end)},
    {"agreement", "labels by several annotators (one column per annotator)",
     "item,ann1,ann2,ann3\n" <> Enum.map_join(1..50, "\n", fn i -> (l = Enum.at(~w(pos neg neu), rem(i, 3)); flip = fn k -> if Vapor.Alembic.Builtins.hash01([:ag, i, k]) < 0.15, do: "neu", else: l end; "#{i},#{l},#{flip.(1)},#{flip.(2)}") end)},
    {"layers", "features per layer and labels (JSON: {\"labels\": [...], \"layers\": [[[...] per item] per layer]})", Vapor.Assay.Layers.example()},
    {"detect", "detections against the truth, one JSON object per image: {\"truth\": [[x0,y0,x1,y1],…], \"pred\": [[x0,y0,x1,y1,score],…]}",
     Vapor.Assay.Detect.example()},
    {"judge", "an LLM judge's verdicts with both orders (columns ab, ba: A, B or tie)",
     "ab,ba\n" <> Enum.map_join(1..60, "\n", fn i -> (h = Vapor.Alembic.Builtins.hash01([:j, i]); if(h < 0.5, do: "A,B", else: if(h < 0.75, do: "A,A", else: "B,A"))) end)}
  ]
  end

  defp b(i, k, p), do: if(Vapor.Alembic.Builtins.hash01([:lb, i, k]) < p, do: 1, else: 0)

  @doc "The tools: `[%{tool, about, example}]`."
  def tools, do: Enum.map(table(), fn {t, a, e} -> %{tool: t, about: a, example: e} end)
  def example(t), do: Enum.find_value(table(), fn {k, _, e} -> if k == t, do: e end)

  @doc "Run a tool on text input (sandboxed). Options: `seed:`, `reps:`."
  def run(tool, text, opts \\ []) do
    if example(tool) == nil do
      {:error, "unknown tool #{tool}: #{Enum.map_join(table(), ", ", &elem(&1, 0))}"}
    else
      t0 = System.monotonic_time(:millisecond)
      res =
        Vapor.Hermetic.seal(fn ->
          try do
            dispatch(tool, text, opts)
          rescue
            e -> {:error, Exception.message(e)}
          end
        end, heap_mb: 2048, timeout: 300_000)

      case res do
        {:ok, {:ok, r}} -> {:ok, r |> Map.put(:tool, tool) |> Map.put(:ms, System.monotonic_time(:millisecond) - t0)}
        {:ok, {:error, e}} -> {:error, e}
        {:error, :memory} -> {:error, "the data needed more memory than allowed"}
        {:error, :timeout} -> {:error, "took longer than 5 minutes"}
        {:error, {:crash, w}} -> {:error, "failed: #{inspect(w) |> String.slice(0, 200)}"}
      end
    end
  end

  defp dispatch("geometry", text, o), do: Vapor.Assay.Geometry.run(text, o)
  defp dispatch("compare", text, o), do: with({:ok, h, rows} <- Stats.table(text), do: compare(h, rows, o))
  defp dispatch("leaderboard", text, o), do: with({:ok, h, rows} <- Stats.table(text), do: leaderboard(h, rows, o))
  defp dispatch("calibration", text, o), do: with({:ok, h, rows} <- Stats.table(text), do: calibration(h, rows, o))
  defp dispatch("agreement", text, o), do: with({:ok, h, rows} <- Stats.table(text), do: agreement(h, rows, o))
  defp dispatch("detect", text, o), do: Vapor.Assay.Detect.run(text, o)
  defp dispatch("layers", text, o), do: Vapor.Assay.Layers.run(text, o)
  defp dispatch("judge", text, o), do: with({:ok, h, rows} <- Stats.table(text), do: judge(h, rows, o))
  defp dispatch("contamination", text, o), do: Data.contamination(text, o)
  defp dispatch("dedup", text, o), do: Data.dedup(text, o)
  defp dispatch("scaling", text, o), do: with({:ok, h, rows} <- Stats.table(text), do: Scaling.fit(h, rows, o))

  defp column(h, rows, names) do
    case Enum.find(names, &(&1 in h)) do
      nil -> nil
      n -> (i = Enum.find_index(h, &(&1 == n)); {n, Enum.map(rows, &Stats.num(Enum.at(&1, i)))})
    end
  end

  # ================================================================ compare

  @doc false
  def compare(h, rows, o) do
    {na, a} = column(h, rows, ["a", "A", "baseline", "model_a", "system_a"]) || numeric_cols(h, rows, 0)
    {nb, b} = column(h, rows, ["b", "B", "candidate", "model_b", "system_b"]) || numeric_cols(h, rows, 1)
    pairs = Enum.zip(a, b) |> Enum.reject(fn {x, y} -> x == nil or y == nil end)
    n = length(pairs)

    if n < 5 do
      {:error, "need at least 5 items with both scores (columns a and b, or two numeric columns)"}
    else
      seed = Keyword.get(o, :seed, 1)
      reps = Keyword.get(o, :reps, 4000)
      ds = Enum.map(pairs, fn {x, y} -> y - x end)
      dt = List.to_tuple(ds)
      md = Stats.mean(ds)
      boot = Stats.bootstrap(n, reps, seed, fn idx -> Enum.reduce(idx, 0.0, &(&2 + elem(dt, &1))) / n end)
      ci = [Stats.pct(boot, 0.025), Stats.pct(boot, 0.975)]
      p_perm = Stats.sign_flip(ds, reps, seed + 1)
      binary = Enum.all?(pairs, fn {x, y} -> x in [0.0, 1.0] and y in [0.0, 1.0] end)
      mcn = if binary do
        b10 = Enum.count(pairs, &(&1 == {1.0, 0.0}))
        b01 = Enum.count(pairs, &(&1 == {0.0, 1.0}))
        %{a_only: b10, b_only: b01, p: Stats.binom_two_sided(min(b10, b01), b10 + b01)}
      end
      sdd = Stats.sd(ds)
      z = Stats.qnorm(0.975) + Stats.qnorm(0.8)
      mde = if sdd > 0, do: z * sdd / :math.sqrt(n), else: 0.0
      need = if md != 0 and sdd > 0, do: ceil((z * sdd / abs(md)) ** 2), else: nil
      sig = p_perm < 0.05 and (ci |> Enum.at(0)) * (ci |> Enum.at(1)) > 0

      {:ok, %{n: n, a: na, b: nb, mean_a: Stats.mean(Enum.map(pairs, &elem(&1, 0))), mean_b: Stats.mean(Enum.map(pairs, &elem(&1, 1))), difference: md, ci95: ci,
              p_permutation: p_perm, mcnemar: mcn, effect_size_dz: if(sdd > 0, do: md / sdd, else: nil), mde80: mde, items_needed: need, significant: sig,
              says:
                (if sig, do: "#{nb} − #{na} = #{f(md)} [#{f(Enum.at(ci, 0))}, #{f(Enum.at(ci, 1))}], p = #{f(p_perm)}: a real difference at this n",
                 else: "#{nb} − #{na} = #{f(md)} [#{f(Enum.at(ci, 0))}, #{f(Enum.at(ci, 1))}], p = #{f(p_perm)}: not distinguishable from noise") <>
                  "; with #{n} items only differences above #{f(mde)} are detectable (80 % power)" <>
                  if(need && not sig, do: "; the observed one would need ≈ #{need} items", else: ""),
              evidence: [%{check: "signal", ok: sig, detail: "bootstrap CI #{if Enum.at(ci, 0) * Enum.at(ci, 1) > 0, do: "excludes", else: "contains"} 0; permutation p = #{f(p_perm)}" <> if(mcn, do: "; McNemar p = #{f(mcn.p)} (#{mcn.a_only} vs #{mcn.b_only} discordant)", else: "")},
                         %{check: "power", ok: abs(md) >= mde, detail: "this n detects differences above #{f(mde)} with 80 % power; the observed is #{f(abs(md))}"}]}}
    end
  end

  defp numeric_cols(h, rows, k) do
    idx = Enum.filter(0..(length(h) - 1), fn i -> Enum.all?(Enum.take(rows, 20), &(Stats.num(Enum.at(&1, i)) != nil)) end)
    case Enum.at(idx, k) do
      nil -> {"?", []}
      i -> {Enum.at(h, i), Enum.map(rows, &Stats.num(Enum.at(&1, i)))}
    end
  end

  # ================================================================ leaderboard

  @doc false
  def leaderboard(h, rows, o) do
    cols = Enum.filter(0..(length(h) - 1), fn i -> rows != [] and Enum.all?(rows, &(Stats.num(Enum.at(&1, i)) != nil)) end)
    cols = Enum.reject(cols, fn i -> Enum.at(h, i) in ~w(id item idx index) end)
    if length(cols) < 2 do
      {:error, "need at least two numeric columns (one per system), every row filled"}
    else
      seed = Keyword.get(o, :seed, 1)
      reps = Keyword.get(o, :reps, 2000)
      names = Enum.map(cols, &Enum.at(h, &1))
      m = for i <- cols, do: Enum.map(rows, &Stats.num(Enum.at(&1, i)))
      n = length(rows)
      mt = Enum.map(m, &List.to_tuple/1)
      means = Enum.map(m, &Stats.mean/1)
      k = length(names)

      {rank_counts, _} =
        Enum.reduce(1..reps, {%{}, Stats.rng(seed)}, fn _, {acc, r} ->
          {idx, r} = Enum.map_reduce(1..n, r, fn _, rr -> :rand.uniform_s(n, rr) end)
          ms = Enum.map(mt, fn t -> Enum.reduce(idx, 0.0, &(&2 + elem(t, &1 - 1))) / n end)
          order = ms |> Enum.with_index() |> Enum.sort_by(&(-elem(&1, 0))) |> Enum.with_index() |> Map.new(fn {{_, sys}, rank} -> {sys, rank + 1} end)
          {Enum.reduce(0..(k - 1), acc, fn s, a -> Map.update(a, {s, order[s]}, 1, &(&1 + 1)) end), r}
        end)

      pairs = for i <- 0..(k - 1), j <- 0..(k - 1), i < j, do: {i, j}
      raw = Enum.map(pairs, fn {i, j} -> Stats.sign_flip(Enum.zip_with(Enum.at(m, j), Enum.at(m, i), &-/2), 2000, seed + i * 31 + j) end)
      adj = Stats.holm(raw)
      leader = Enum.max_by(0..(k - 1), &Enum.at(means, &1))
      tied = for {{i, j}, p} <- Enum.zip(pairs, adj), leader in [i, j], p >= 0.05, do: if(i == leader, do: j, else: i)

      systems =
        for s <- 0..(k - 1) do
          dist = for r <- 1..k, do: Map.get(rank_counts, {s, r}, 0) / reps
          cum = Enum.scan(dist, &+/2)
          lo = Enum.find_index(cum, &(&1 >= 0.025)) + 1
          hi = Enum.find_index(cum, &(&1 >= 0.975)) + 1
          col = Enum.at(mt, s)
          boot = Stats.bootstrap(n, 1000, seed + 100 + s, fn idx -> Enum.reduce(idx, 0.0, &(&2 + elem(col, &1))) / n end)
          %{system: Enum.at(names, s), mean: Enum.at(means, s), ci95: [Stats.pct(boot, 0.025), Stats.pct(boot, 0.975)], p_first: hd(dist), rank_interval: [lo, hi]}
        end
        |> Enum.sort_by(&(-&1.mean))

      {:ok, %{systems: systems, items: n, pairwise: Enum.zip_with([pairs, raw, adj], fn [{i, j}, p, pa] -> %{a: Enum.at(names, i), b: Enum.at(names, j), p: p, p_holm: pa} end),
              tied_with_leader: Enum.map(tied, &Enum.at(names, &1)),
              says: "#{Enum.at(names, leader)} leads (P(#1) = #{f(hd(systems).p_first)}); " <>
                if(tied == [], do: "it differs significantly from every other system (Holm-adjusted)", else: "not distinguishable from #{Enum.map_join(tied, ", ", &Enum.at(names, &1))} at this n (Holm-adjusted)"),
              evidence: [%{check: "rank stability", ok: hd(systems).p_first >= 0.95, detail: "the leader is first in #{round(hd(systems).p_first * 100)} % of #{reps} bootstrap leaderboards"}]}}
    end
  end

  # ================================================================ calibration

  @doc false
  def calibration(h, rows, o) do
    {_, ps} = column(h, rows, ["p", "confidence", "prob", "conf"]) || {nil, nil}
    {_, ys} = column(h, rows, ["correct", "y", "label", "hit"]) || {nil, nil}
    if ps == nil or ys == nil do
      {:error, "columns p (confidence) and correct (0/1)"}
    else
      pairs = Enum.zip(ps, ys) |> Enum.reject(fn {p, y} -> p == nil or y == nil or p < 0 or p > 1 end)
      n = length(pairs)
      seed = Keyword.get(o, :seed, 1)
      bins = Keyword.get(o, :bins, 15) |> min(max(div(n, 10), 2))
      ece = ece(pairs, bins)
      pt = List.to_tuple(pairs)
      boot = Stats.bootstrap(n, 1000, seed, fn idx -> ece(Enum.map(idx, &elem(pt, &1)), bins) end)
      {null, _} =
        Enum.map_reduce(1..500, Stats.rng(seed + 7), fn _, r ->
          {sim, r} = Enum.map_reduce(pairs, r, fn {p, _}, rr -> {u, rr} = :rand.uniform_s(rr); {{p, if(u < p, do: 1.0, else: 0.0)}, rr} end)
          {ece(sim, bins), r}
        end)
      floor = Stats.mean(null)
      p_val = (Enum.count(null, &(&1 >= ece)) + 1) / 501
      brier = Enum.reduce(pairs, 0.0, fn {p, y}, s -> s + (p - y) * (p - y) end) / n
      nll = Enum.reduce(pairs, 0.0, fn {p, y}, s -> pp = min(max(p, 1.0e-12), 1 - 1.0e-12); s - (y * :math.log(pp) + (1 - y) * :math.log(1 - pp)) end) / n
      {fitset, evalset} = pairs |> Enum.with_index() |> Enum.split_with(fn {_, i} -> rem(i, 2) == 0 end)
      fitset = Enum.map(fitset, &elem(&1, 0))
      evalset = Enum.map(evalset, &elem(&1, 0))
      {a, b} = platt(fitset)
      scaled = Enum.map(evalset, fn {p, y} -> {sigmoid(a * logit(p) + b), y} end)

      {:ok, %{n: n, bins: bins, ece: ece, ece_ci95: [Stats.pct(boot, 0.025), Stats.pct(boot, 0.975)], ece_if_calibrated: floor, ece_debiased: ece - floor, p_miscalibrated: p_val,
              brier: brier, nll: nll, reliability: reliability(pairs, bins),
              recalibration: %{temperature: if(a != 0, do: 1 / a, else: nil), bias: b, ece_before: ece(evalset, bins), ece_after: ece(scaled, bins), fitted_on: length(fitset), judged_on: length(evalset)},
              says: "ECE #{f(ece)} [#{f(Stats.pct(boot, 0.025))}, #{f(Stats.pct(boot, 0.975))}]; a perfectly calibrated model with these confidences would still show #{f(floor)} — " <>
                if(p_val < 0.05, do: "the miscalibration is real (p = #{f(p_val)})", else: "the measured ECE is within sampling noise (p = #{f(p_val)})") <>
                "; Platt scaling fitted on one half, judged on the other: #{f(ece(evalset, bins))} → #{f(ece(scaled, bins))}",
              evidence: [%{check: "bias floor", ok: true, detail: "ECE at n = #{n} is biased upward by ≈ #{f(floor)}: report #{f(ece - floor)} above the floor"}]}}
    end
  end

  defp ece(pairs, bins) do
    n = length(pairs)
    if n == 0 do
      nil
    else
      sorted = Enum.sort_by(pairs, &elem(&1, 0))
      size = max(div(n, bins), 1)
      sorted |> Enum.chunk_every(size) |> Enum.reduce(0.0, fn chunk, s ->
        conf = Enum.reduce(chunk, 0.0, &(&2 + elem(&1, 0))) / length(chunk)
        acc = Enum.reduce(chunk, 0.0, &(&2 + elem(&1, 1))) / length(chunk)
        s + length(chunk) / n * abs(conf - acc)
      end)
    end
  end

  defp reliability(pairs, bins) do
    n = length(pairs)
    pairs |> Enum.sort_by(&elem(&1, 0)) |> Enum.chunk_every(max(div(n, bins), 1)) |> Enum.map(fn c ->
      %{confidence: Enum.reduce(c, 0.0, &(&2 + elem(&1, 0))) / length(c), accuracy: Enum.reduce(c, 0.0, &(&2 + elem(&1, 1))) / length(c), count: length(c)}
    end)
  end

  defp logit(p), do: (pp = min(max(p, 1.0e-6), 1 - 1.0e-6); :math.log(pp / (1 - pp)))
  defp sigmoid(z), do: 1 / (1 + :math.exp(-z))

  # Platt scaling on the logit: maximise the likelihood by Newton's method
  defp platt(pairs) do
    xs = Enum.map(pairs, fn {p, y} -> {logit(p), y} end)
    Enum.reduce(1..50, {1.0, 0.0}, fn _, {a, b} ->
      {ga, gb, haa, hab, hbb} =
        Enum.reduce(xs, {0.0, 0.0, 0.0, 0.0, 0.0}, fn {x, y}, {ga, gb, haa, hab, hbb} ->
          p = sigmoid(a * x + b)
          w = p * (1 - p)
          {ga + (p - y) * x, gb + (p - y), haa + w * x * x, hab + w * x, hbb + w}
        end)
      det = (haa + 1.0e-9) * (hbb + 1.0e-9) - hab * hab
      if abs(det) < 1.0e-18, do: {a, b}, else: {a - ((hbb + 1.0e-9) * ga - hab * gb) / det, b - ((haa + 1.0e-9) * gb - hab * ga) / det}
    end)
  end

  # ================================================================ agreement

  @doc false
  def agreement(h, rows, o) do
    cols = Enum.reject(0..(length(h) - 1), fn i -> Enum.at(h, i) in ~w(id item idx index text) end)
    if length(cols) < 2 do
      {:error, "need at least two annotator columns"}
    else
      seed = Keyword.get(o, :seed, 1)
      data = Enum.map(rows, fn r -> Enum.map(cols, fn i -> v = Enum.at(r, i); if v in [nil, "", "NA", "na", "-"], do: nil, else: to_string(v) end) end)
      alpha = kripp(data)
      dt = List.to_tuple(data)
      n = length(data)
      boot = Stats.bootstrap(n, 1000, seed, fn idx -> kripp(Enum.map(idx, &elem(dt, &1))) end)
      complete = Enum.filter(data, fn r -> Enum.all?(r, &(&1 != nil)) end)
      fleiss = if length(complete) >= 2, do: fleiss(complete)
      cohen = if length(cols) == 2, do: cohen(complete)
      verdict = cond do alpha == nil -> "—"; alpha >= 0.8 -> "reliable (α ≥ 0.8)"; alpha >= 0.667 -> "tentative (0.667 ≤ α < 0.8)"; true -> "unreliable (α < 0.667)" end

      {:ok, %{items: n, annotators: Enum.map(cols, &Enum.at(h, &1)), krippendorff_alpha: alpha, alpha_ci95: [Stats.pct(boot, 0.025), Stats.pct(boot, 0.975)],
              fleiss_kappa: fleiss, cohen_kappa: cohen,
              says: "Krippendorff's α = #{f(alpha)} [#{f(Stats.pct(boot, 0.025))}, #{f(Stats.pct(boot, 0.975))}] — #{verdict} by Krippendorff's thresholds",
              evidence: [%{check: "lower bound", ok: Stats.pct(boot, 0.025) >= 0.667, detail: "the interval's lower end #{f(Stats.pct(boot, 0.025))} against 0.667"}]}}
    end
  end

  # nominal Krippendorff's alpha from the coincidence matrix
  defp kripp(data) do
    units = Enum.map(data, fn r -> Enum.reject(r, &is_nil/1) end) |> Enum.filter(&(length(&1) >= 2))
    coinc =
      Enum.reduce(units, %{}, fn vs, acc ->
        m = length(vs)
        for {a, i} <- Enum.with_index(vs), {b, j} <- Enum.with_index(vs), i != j, reduce: acc do
          ac -> Map.update(ac, {a, b}, 1 / (m - 1), &(&1 + 1 / (m - 1)))
        end
      end)
    nc = Enum.reduce(coinc, %{}, fn {{a, _}, v}, acc -> Map.update(acc, a, v, &(&1 + v)) end)
    total = Enum.sum(Map.values(nc))
    if total <= 1 do
      nil
    else
      dobs = Enum.reduce(coinc, 0.0, fn {{a, b}, v}, s -> if a != b, do: s + v, else: s end)
      dexp = (total * total - Enum.reduce(nc, 0.0, fn {_, v}, s -> s + v * v end)) / (total - 1)
      if dexp == 0, do: 1.0, else: 1 - dobs / dexp
    end
  end

  defp fleiss(rows) do
    cats = rows |> List.flatten() |> Enum.uniq()
    n = length(rows)
    r = length(hd(rows))
    counts = Enum.map(rows, fn row -> Enum.map(cats, fn c -> Enum.count(row, &(&1 == c)) end) end)
    pj = Enum.map(0..(length(cats) - 1), fn j -> Enum.reduce(counts, 0, &(&2 + Enum.at(&1, j))) / (n * r) end)
    pi = Enum.map(counts, fn cs -> (Enum.reduce(cs, 0, &(&2 + &1 * &1)) - r) / (r * (r - 1)) end)
    pbar = Stats.mean(pi)
    pe = Enum.reduce(pj, 0.0, &(&2 + &1 * &1))
    if pe == 1, do: 1.0, else: (pbar - pe) / (1 - pe)
  end

  defp cohen(rows) do
    n = length(rows)
    if n == 0 do
      nil
    else
      po = Enum.count(rows, fn [a, b] -> a == b end) / n
      cats = rows |> List.flatten() |> Enum.uniq()
      pe = Enum.reduce(cats, 0.0, fn c, s -> s + Enum.count(rows, &(hd(&1) == c)) / n * (Enum.count(rows, &(List.last(&1) == c)) / n) end)
      if pe == 1, do: 1.0, else: (po - pe) / (1 - pe)
    end
  end

  # ================================================================ judge

  @doc false
  def judge(h, rows, _o) do
    ia = Enum.find_index(h, &(&1 in ["ab", "AB", "order_ab", "first"]))
    ib = Enum.find_index(h, &(&1 in ["ba", "BA", "order_ba", "second"]))
    if ia == nil or ib == nil do
      {:error, "columns ab and ba: the judge's verdict (A, B or tie) when A is shown first, and when B is shown first — verdicts name the systems, not the slots"}
    else
      norm = fn v -> case String.downcase(String.trim(to_string(v))) do "a" -> :a; "b" -> :b; _ -> :tie end end
      pairs = Enum.map(rows, fn r -> {norm.(Enum.at(r, ia)), norm.(Enum.at(r, ib))} end)
      n = length(pairs)
      consistent = Enum.count(pairs, fn {x, y} -> x == y end)
      # first-slot wins: A when A is first, B when B is first
      first = Enum.count(pairs, fn {x, _} -> x == :a end) + Enum.count(pairs, fn {_, y} -> y == :b end)
      decided = Enum.count(pairs, fn {x, _} -> x != :tie end) + Enum.count(pairs, fn {_, y} -> y != :tie end)
      p = Stats.binom_two_sided(first, decided)
      flips = Enum.count(pairs, fn {x, y} -> x == :a and y == :b end)
      {:ok, %{items: n, consistent: consistent / max(n, 1), first_slot_rate: first / max(decided, 1), p_position_bias: p, order_flips: flips,
              says: "the verdict survives swapping the order in #{round(consistent / max(n, 1) * 100)} % of items; the first slot wins #{round(first / max(decided, 1) * 100)} % of decided verdicts (" <>
                if(p < 0.05, do: "position bias, p = #{f(p)}", else: "no detectable position bias, p = #{f(p)}") <> ")",
              evidence: [%{check: "position bias", ok: p >= 0.05, detail: "exact binomial test of first-slot wins = 50 %: p = #{f(p)}"}]}}
    end
  end
end
