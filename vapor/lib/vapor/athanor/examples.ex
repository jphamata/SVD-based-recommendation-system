defmodule Vapor.Athanor.Examples do
  @moduledoc """
  Starting points for the Athanor, one per kind of question, grouped by
  field (docs/ATHANOR.md §7). They are **text to edit**, not
  demonstrations: change a number, a constraint or the objective and the
  same engine answers the new question.
  """

  @examples [
    %{id: "golomb", field: "math", title: "Golomb ruler", about: "marks whose pairwise distances are all different — shortest ruler",
      text: """
      # a Golomb ruler with 7 marks (the optimum is 25)
      marks = 7
      space = subset(1..30, marks - 1)          # the first mark is 0
      ruler(r) = [0] ++ r
      dist(r) = [b - a for (a, b) in pairs(ruler(r))]
      violation(r) = len(dist(r)) - len(distinct(dist(r)))
      minimize(r) = max(r)
      target = 25
      budget = 30000
      show(r) = join([str(m) for m in ruler(r)], " ")
      """},
    %{id: "schur", field: "math", title: "Schur colouring", about: "colour 1…n so that no x + y = z is monochromatic",
      text: """
      # three colours suffice up to n = 13 (Schur's number S(3) = 13); try 14 and watch it fail
      n = 13
      space = seq(n, [0, 1, 2])
      violation(c) = count([1 for x in 1..n for y in x..n if x + y <= n and c[x-1] == c[y-1] and c[x-1] == c[x+y-1]])
      budget = 60000
      show(c) = join([str(i + 1) ++ "→" ++ ["red", "green", "blue"][c[i]] for i in 0..n-1], " ")
      """},
    %{id: "ramsey", field: "math", title: "A Ramsey claim, proved by exhaustion", about: "every graph on 6 vertices has a triangle or an independent triple",
      text: """
      # R(3,3) = 6: the claim holds for all 2^15 graphs on 6 vertices (exhaustive proof);
      # set n = 5 and the search finds the counterexample (the pentagon)
      n = 6
      space = graph(n)
      adj(g, a, b) = (min(a, b), max(a, b)) in g
      triples = [(a, b, c) for a in 0..n-1 for b in a+1..n-1 for c in b+1..n-1]
      claim(g) = any([adj(g, a, b) == adj(g, b, c) and adj(g, b, c) == adj(g, a, c) for (a, b, c) in triples])
      budget = 40000
      """},
    %{id: "euler", field: "math", title: "Refute a conjecture", about: "Euler's n² + n + 41 is always prime?",
      text: """
      space = ints(1, 0, 1000)
      claim(x) = is_prime(x[0]^2 + x[0] + 41)
      """},
    %{id: "cubes", field: "math", title: "Sums of three cubes", about: "integers x, y, z with x³ + y³ + z³ = k",
      text: """
      k = 29
      space = ints(3, -60, 60)
      violation(v) = abs(v[0]^3 + v[1]^3 + v[2]^3 - k)
      minimize(v) = sum([abs(t) for t in v])         # among solutions, the smallest
      budget = 80000
      """},
    %{id: "sorting", field: "cs", title: "Sorting network", about: "a comparator network that sorts every input (0-1 principle)",
      text: """
      # 5 wires; 9 comparators is the known minimum — try 8 and the search cannot find one
      wires = 5
      size = 9
      space = seq(size, [(a, b) for a in 0..wires-1 for b in a+1..wires-1])
      apply(net, v) = fold(net, v, (x, c) => if x[c[0]] > x[c[1]] then swap(x, c[0], c[1]) else x)
      inputs = [bits(k, wires) for k in 0..2^wires - 1]
      violation(net) = count([1 for v in inputs if not is_sorted(apply(net, v))])
      budget = 20000
      """},
    %{id: "tictactoe", field: "cs", title: "A game: tic-tac-toe", about: "any two-player game: solved exactly, searched, learned, played",
      text: """
      # tic-tac-toe: solve it exactly, search it, learn it, or play against vapor
      init = [0, 0, 0, 0, 0, 0, 0, 0, 0]
      lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
      player(s) = if count(s, 1) == count(s, 2) then 1 else 2
      moves(s) = if winner(s) != nil then [] else [i for i in 0..8 if s[i] == 0]
      play(s, m) = set_at(s, m, player(s))
      won(s, p) = any([all([s[i] == p for i in l]) for l in lines])
      winner(s) = if won(s, 1) then 1 else if won(s, 2) then 2 else if count(s, 0) == 0 then 0 else nil
      show(s) = join([join([[".", "X", "O"][s[3*r + c]] for c in 0..2]) for r in 0..2], "\n")
      features(s) = let p = player(s), q = 3 - p in [count([1 for l in lines if count([s[i] for i in l], p) == 2 and count([s[i] for i in l], q) == 0]), count([1 for l in lines if count([s[i] for i in l], q) == 2 and count([s[i] for i in l], p) == 0]), if s[4] == p then 1 else 0, if s[4] == q then 1 else 0]
      """},
    %{id: "nim", field: "math", title: "A game: subtraction", about: "take 1, 3 or 4 — who takes the last wins; solved exactly",
      text: """
      # which pile sizes lose for the player to move? (solve from different states)
      init = (30, 1)
      player(s) = s[1]
      moves(s) = [k for k in [1, 3, 4] if k <= s[0]]
      play(s, m) = (s[0] - m, 3 - s[1])
      winner(s) = if s[0] == 0 then 3 - s[1] else nil
      show(s) = str(s[0]) ++ " stones, player " ++ str(s[1]) ++ " to move"
      """},
    %{id: "tsp", field: "cs", title: "Travelling salesman", about: "the shortest tour through your cities",
      text: """
      cities = [(0.1, 0.2), (0.8, 0.9), (0.4, 0.7), (0.9, 0.1), (0.3, 0.3), (0.6, 0.5), (0.2, 0.8), (0.7, 0.3), (0.5, 0.1), (0.95, 0.6), (0.05, 0.55), (0.45, 0.95)]
      n = len(cities)
      space = perm(n)
      d(a, b) = hypot(a[0] - b[0], a[1] - b[1])
      minimize(t) = sum([d(cities[t[i]], cities[t[(i + 1) % n]]) for i in 0..n-1])
      budget = 20000
      """},
    %{id: "maxcut", field: "cs", title: "Max-cut", about: "split a graph's vertices to cut the most edges",
      text: """
      edges = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4),(3,5),(4,6),(5,6),(5,7),(6,8),(7,8),(7,9),(8,9),(0,9),(2,6),(1,8)]
      space = bits(10)
      maximize(s) = count([1 for (a, b) in edges if s[a] != s[b]])
      budget = 3000
      """},
    %{id: "binpack", field: "cs", title: "Bin packing", about: "items into as few bins as possible",
      text: """
      items = [42, 35, 33, 31, 28, 27, 24, 21, 19, 17, 15, 12, 9, 7]
      cap = 100
      space = partition(len(items), 6)
      loads(p) = [sum([items[i] for i in 0..len(items)-1 if p[i] == g]) for g in 0..max(p)]
      violation(p) = sum([max(0, l - cap) for l in loads(p)])
      minimize(p) = len(loads(p))
      budget = 30000
      show(p) = join([str([items[i] for i in 0..len(items)-1 if p[i] == g]) for g in 0..max(p)], " | ")
      """},
    %{id: "thomson", field: "science", title: "Thomson problem", about: "electrons on a sphere at minimum energy",
      text: """
      # 6 charges: the optimum is the octahedron, E = 9.985281374
      n = 6
      space = reals(2*n, 0, 1)
      pt(v, i) = let th = acos(1 - 2*v[2*i]), ph = 2*pi*v[2*i+1] in (sin(th)*cos(ph), sin(th)*sin(ph), cos(th))
      r(a, b) = sqrt((a[0]-b[0])^2 + (a[1]-b[1])^2 + (a[2]-b[2])^2)
      minimize(v) = let ps = [pt(v, i) for i in 0..n-1] in sum([1 / max(r(a, b), 1e-9) for (a, b) in pairs(ps)])
      budget = 12000
      """},
    %{id: "lab", field: "science", title: "An experiment you run", about: "vapor proposes settings, you measure (Bayesian optimisation)",
      text: """
      # temperature (°C) and time (min) for a yield you measure in the lab:
      # run `vapor athanor ask lab.nbq` and type each measurement
      space = reals(2, 0, 1)
      measured = true
      show(x) = "temperature " ++ str(round(150 + 100*x[0])) ++ " °C, time " ++ str(round(5 + 55*x[1])) ++ " min"
      budget = 25
      """},
    %{id: "hyper", field: "ai", title: "Hyperparameters with a holdout", about: "tune a classifier — and measure the search's own overfitting",
      text: """
      # logistic regression trained by gradient descent, inside the verifier;
      # holdout(x) is computed only for the finalists — the gap is the selection bias
      data = [(x1, x2, if x1 + 0.5*x2 + 0.3*noise(i, 9) > 0.6 then 1 else 0) for i in 0..59 for x1 in [noise(i, 1)] for x2 in [noise(i, 2)]]
      train = take(data, 30)
      valid_set = drop(take(data, 45), 30)
      test_set = drop(data, 45)
      sig(z) = 1 / (1 + exp(-z))
      fit(lr, l2, steps) = fold(range(steps), (0.0, 0.0, 0.0), (w, k) =>
        let g = [(sig(w[0] + w[1]*a + w[2]*b) - y) for (a, b, y) in train] in
        (w[0] - lr*mean(g), w[1] - lr*(mean([e*a for (e, (a, b, y)) in zip(g, train)]) + l2*w[1]), w[2] - lr*(mean([e*b for (e, (a, b, y)) in zip(g, train)]) + l2*w[2])))
      acc(w, rows) = mean([if (sig(w[0] + w[1]*a + w[2]*b) > 0.5) == (y == 1) then 1 else 0 for (a, b, y) in rows])
      model(x) = fit(10^(-2 + 3*x[0]), 10^(-4 + 3*x[1]), 20 + round(80*x[2]))
      space = reals(3, 0, 1)
      maximize(x) = acc(model(x), valid_set)
      holdout(x) = acc(model(x), test_set)
      budget = 120
      """},
    %{id: "adversarial", field: "ai", title: "Adversarial example", about: "the smallest change that flips a classifier",
      text: """
      w = [0.8, -0.5, 0.3, 0.9]
      b = -0.2
      x0 = [0.5, 0.1, 0.4, 0.2]                 # classified positive: w·x0 + b > 0
      eps = 0.3
      space = reals(4, -eps, eps)
      margin(d) = dot(w, [a + e for (a, e) in zip(x0, d)]) + b
      violation(d) = max(0, margin(d))           # valid when the label flips
      minimize(d) = max([abs(e) for e in d])     # the L∞ size of the change
      budget = 6000
      """},
    %{id: "portfolio", field: "finance", title: "Portfolio with an out-of-sample check", about: "weights that maximise Sharpe — and how much of it is selection bias",
      text: """
      # 5 assets, 240 days of returns (synthetic, deterministic); the first 160 days are in-sample
      assets = 5
      mu = [0.0004, 0.0003, 0.0005, 0.0002, 0.0004]
      ret(t, a) = mu[a] + 0.01*(noise(t, a) - 0.5) + 0.006*(noise(t, 99) - 0.5)
      days = [[ret(t, a) for a in 0..assets-1] for t in 0..239]
      insample = take(days, 160)
      outsample = drop(days, 160)
      weights(x) = let s = sum(x) in [v / s for v in x]
      port(x, rows) = [dot(weights(x), r) for r in rows]
      sharpe(xs) = mean(xs) / std(xs) * sqrt(252)
      space = reals(assets, 0.01, 1)
      maximize(x) = sharpe(port(x, insample))
      holdout(x) = sharpe(port(x, outsample))
      budget = 3000
      show(x) = join([str(round(100*w)) ++ "%" for w in weights(x)], " ")
      """},
    %{id: "crossover", field: "finance", title: "Trading-rule mining, honestly", about: "moving-average crossover: the in-sample winner versus the out-of-sample truth",
      text: """
      # a random walk has no edge: every in-sample "winner" is selection bias, and the holdout shows it
      n = 600
      price = scan(range(n), 100.0, (p, t) => p * (1 + 0.01*(noise(t, 7) - 0.5)))
      ma(t, k) = mean(price[t-k+1:t+1])
      pnl(f, s, lo, hi) = [if ma(t, f) > ma(t, s) then price[t+1]/price[t] - 1 else 0 for t in lo..hi-1]
      sharpe(xs) = if std(xs) == 0 then 0 else mean(xs) / std(xs) * sqrt(252)
      space = ints(2, 2, 60)
      valid(x) = x[0] < x[1]
      maximize(x) = sharpe(pnl(x[0], x[1], 60, 400))
      holdout(x) = sharpe(pnl(x[0], x[1], 400, n - 1))
      budget = 600
      show(x) = "fast " ++ str(x[0]) ++ ", slow " ++ str(x[1])
      """},
    %{id: "fitlaw", field: "science", title: "Discover a formula", about: "symbolic regression over a program space",
      text: """
      # Kepler's third law from the planets (a in AU, T in years): T = a^1.5
      planets = [(0.387, 0.241), (0.723, 0.615), (1.0, 1.0), (1.524, 1.881), (5.203, 11.86), (9.537, 29.46)]
      space = program(["a"], ["*", "/", "sqrt", "+"], [1], 7)
      minimize(f) = mean([(f(a) / t - 1)^2 for (a, t) in planets])
      target = 1e-4
      budget = 20000
      """}
  ]

  @doc "Every example: `[%{id, field, title, about, text}]`."
  def all, do: @examples

  def get(id), do: Enum.find(@examples, &(&1.id == id))
end
