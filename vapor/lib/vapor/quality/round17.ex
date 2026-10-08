defmodule Vapor.Quality.Round17 do
  @moduledoc """
  Quality checks for the 0.17 round — the hermetic seal, integer programs,
  causes, the mould, the rebirth of a model plank by plank, and
  recommendations — in the suite's discipline: a value, a **control** that
  a naive implementation would produce, and a threshold that separates them.

  | check | value | control (must fail, or be caught) |
  |---|---|---|
  | seal | 512 MB of off-heap binaries under a 64 MB seal: stopped | the 0.16 heap-only cap: survives with 8× the cap |
  | integer programs | branch and bound = brute force on 40 random programs, every certificate checked | the relaxation rounded down: wrong on some |
  | forged optimum | a better objective with the honest tree: refused | the naive check (incumbent feasible and integral) accepts a worse point |
  | identification | front door and napkin estimands = the true intervention, exactly, on random models | the naive P(y \\| x) |
  | hedge | the bow: not identifiable, the hedge checked | two models with one P(x, y) and two effects, computed |
  | trojan | a 20-bit trigger in a 21-input adder: found by SAT, DRUP-checked | 4 096 random patterns: not found |
  | mould | a mapped adder read back: proved equal (cells and NAND styles) | one pin moved: told apart |
  | target gate | a plank that restores the ship: admitted | a sham of the same norm: refused |
  | brake | the same plank under ε = 0.5: refused | under ε = π: admitted |
  | symmetry | a block with its hidden units permuted: drift at rounding level | a random change of the same norm |
  | recommendations | planted interactions: signal | unstructured ratings: not signal |
  | SVD | one-sided Jacobi: σ_min of a matrix with κ = 10⁸ to 10⁻⁶ relative | √eig(AᵀA): the small value lost |
  """
  alias Vapor.{Dense, Hermetic, Lock, Palingenesis, Qalib, Rebis, Recommend, Tensor}
  alias Vapor.Logic.{Causal, LP, MIP}

  def run(_opts \\ []) do
    %{checks: List.flatten([seal(), integer_programs(), causes(), mould(), palingenesis(), recommendations(), svd()])}
  end

  defp check(name, value, control, threshold, pass), do: %{name: name, value: value, control: control, threshold: threshold, pass: pass}

  # ------------------------------------------------------------------- seal

  defp bomb, do: for(_ <- 1..512, do: :binary.copy(<<0>>, 1_048_576)) |> length()

  defp seal do
    sealed = Hermetic.seal(&bomb/0, heap_mb: 64, timeout: 60_000)
    words = div(64 * 1_048_576, :erlang.system_info(:wordsize))
    {pid, mon} = spawn_monitor(fn -> Process.flag(:max_heap_size, %{size: words, kill: true, error_logger: false}); exit({:survived, bomb()}) end)
    old = receive do {:DOWN, ^mon, :process, ^pid, why} -> why end

    check("seal: 512 MB of off-heap binaries under a 64 MB seal", inspect(sealed), "the 0.16 heap-only cap: #{inspect(old)}",
          "stopped by the seal; the old cap lets them through", sealed == {:error, :memory} and old == {:survived, 512})
  end

  # --------------------------------------------------------- integer programs

  defp integer_programs do
    :rand.seed(:exsss, {17, 29, 31})

    progs =
      for _ <- 1..40 do
        coef = fn -> :rand.uniform(9) - 3 end
        sense = Enum.random(["maximize", "minimize"])
        rows = for _ <- 1..3, do: "#{coef.()}x + #{coef.()}y + #{coef.()}z #{Enum.random(["<=", ">="])} #{:rand.uniform(12) - 4}"
        Enum.join(["#{sense} #{coef.()}x + #{coef.()}y + #{coef.()}z"] ++ rows ++ ["x <= 4", "y <= 4", "z <= 4", "int x, y, z"], "\n")
      end

    results =
      for t <- progs do
        {:ok, p} = MIP.parse(t)
        truth = brute(p)
        {:ok, r} = MIP.solve(p)
        ok = r.check.accepted and if(truth == nil, do: r.status == :infeasible, else: r.status == :optimal and r.objective == truth)
        {:ok, lp} = LP.solve(p.lp)
        rounded = lp.status == :optimal and truth != nil and floor_obj(p, lp.x) == truth
        {ok, rounded}
      end

    agree = Enum.count(results, &elem(&1, 0))
    naive = Enum.count(results, &elem(&1, 1))

    knap = "maximize 8a + 11b + 6c + 4d\n5a + 7b + 4c + 3d <= 14\nbin a, b, c, d"
    {:ok, kp} = MIP.parse(knap)
    {:ok, kr} = MIP.solve(kp)
    forged = MIP.check(kp, put_in(kr.certificate, [:incumbent, :objective], LP.qadd(kr.objective, {1, 1})))
    worse = %{x: Map.new(kp.lp.vars, &{&1, {0, 1}}), objective: {0, 1}}
    naive_ok = feasible?(kp, worse.x)
    honest_tree_worse = MIP.check(kp, %{kr.certificate | incumbent: worse})

    [check("integer programs: branch and bound against brute force, certificates checked", "#{agree}/40", "the relaxation rounded down: #{naive}/40 right",
           "40/40, the control below", agree == 40 and naive < 40),
     check("integer programs: a forged optimum, and a worse incumbent with the honest tree", "forged: #{forged.accepted}; worse: #{honest_tree_worse.accepted}",
           "the naive check (feasible and integral) accepts the worse point: #{naive_ok}", "both refused; the naive check fooled",
           forged.accepted == false and honest_tree_worse.accepted == false and naive_ok)]
  end

  defp brute(p) do
    pts = Enum.reduce(p.lp.vars, [%{}], fn v, acc -> for m <- acc, k <- 0..4, do: Map.put(m, v, {k, 1}) end)
    vals = for x <- pts, feasible?(p, x), do: value(p, x)
    cond do
      vals == [] -> nil
      p.lp.sense == :max -> Enum.max_by(vals, &LP.to_float/1)
      true -> Enum.min_by(vals, &LP.to_float/1)
    end
  end

  defp value(p, x), do: Enum.reduce(p.lp.vars, p.lp.c0, fn v, s -> LP.qadd(s, LP.qmul(Map.get(p.lp.c, v, {0, 1}), x[v])) end)

  defp feasible?(p, x) do
    Enum.all?(p.lp.rows, fn {co, op, rhs} ->
      s = LP.qcmp(Enum.reduce(co, {0, 1}, fn {v, a}, acc -> LP.qadd(acc, LP.qmul(a, Map.get(x, v, {0, 1}))) end), rhs)
      case op do :le -> s <= 0; :ge -> s >= 0; :eq -> s == 0 end
    end)
  end

  defp floor_obj(p, x), do: value(p, Map.new(x, fn {v, {n, d}} -> {v, {Integer.floor_div(n, d), 1}} end))

  # ------------------------------------------------------------------- causes

  defp causes do
    {:ok, fd} = Causal.graph([{:dir, "x", "m"}, {:dir, "m", "y"}, {:bi, "x", "y"}])
    {:ok, nap} = Causal.graph([{:dir, "w", "z"}, {:dir, "z", "x"}, {:dir, "x", "y"}, {:bi, "w", "x"}, {:bi, "w", "y"}])
    {:ok, efd} = Causal.identify(fd, ["y"], ["x"])
    {:ok, enap} = Causal.identify(nap, ["y"], ["x"])
    runs = for {g, e} <- [{fd, efd}, {nap, enap}], seed <- 1..5, do: {agrees?(g, e, seed), agrees?(g, {:p, ["y"], ["x"]}, seed)}
    exact = Enum.count(runs, &elem(&1, 0))
    naive = Enum.count(runs, &elem(&1, 1))

    {:ok, bow} = Causal.graph([{:dir, "x", "y"}, {:bi, "x", "y"}])
    {:fail, h} = Causal.identify(bow, ["y"], ["x"])
    hedge = Causal.hedge?(bow, h.ys, h.xs, h.f, h.f_prime)
    cpt = fn ypar -> %{"x" => {["u_x_y"], %{%{"u_x_y" => 0} => {0, 1}, %{"u_x_y" => 1} => {1, 1}}},
                       "y" => {["x", "u_x_y"], Map.new(Causal.assignments(["x", "u_x_y"]), fn a -> {a, {a[ypar], 1}} end)}} end
    a = %{nodes: ["x", "y"], hidden: %{"u_x_y" => {1, 2}}, cpt: cpt.("u_x_y")}
    b = %{a | cpt: cpt.("x")}
    same_obs = joint(a, %{}) == joint(b, %{})
    effects = {prob(joint(a, %{"x" => 1}), %{"y" => 1}), prob(joint(b, %{"x" => 1}), %{"y" => 1})}

    [check("causes: front-door and napkin estimands against the true intervention (exact rationals, 10 models)", "#{exact}/10",
           "the naive P(y | x): #{naive}/10", "10/10; the naive estimate wrong somewhere", exact == 10 and naive < 10),
     check("causes: the bow is not identifiable (a checked hedge)", "hedge checked: #{hedge}",
           "two models, one P(x, y): #{same_obs}; P(y=1 | do(x=1)) = #{LP.show(elem(effects, 0))} and #{LP.show(elem(effects, 1))}",
           "the hedge holds; the two effects differ", hedge and same_obs and elem(effects, 0) != elem(effects, 1))]
  end

  defp agrees?(g, e, seed) do
    m = scm(g, seed)
    obs = joint(m, %{})
    free = Causal.free_vars(e) -- ["x", "y"]

    Enum.all?([0, 1], fn xv ->
      cut = joint(m, %{"x" => xv})
      Enum.all?([0, 1], fn yv -> Enum.all?(Causal.assignments(free), fn fa -> Causal.eval(e, obs, Map.merge(fa, %{"x" => xv, "y" => yv})) == prob(cut, %{"y" => yv}) end) end)
    end)
  end

  defp scm(g, seed) do
    st = :rand.seed_s(:exsss, {seed, 7, 11})
    hidden = for {a, b} <- Enum.sort(g.bi), do: "u_#{a}_#{b}"
    hpar = fn v -> for {a, b} <- Enum.sort(g.bi), v in [a, b], do: "u_#{a}_#{b}" end
    {hp, st} = Enum.map_reduce(hidden, st, fn h, s -> {k, s} = :rand.uniform_s(9, s); {{h, LP.q(k, 10)}, s} end)

    {cpt, _} =
      Enum.map_reduce(g.nodes, st, fn v, s ->
        ps = Causal.parents(g, v) ++ hpar.(v)
        {t, s} = Enum.map_reduce(Causal.assignments(ps), s, fn a, s -> {k, s} = :rand.uniform_s(9, s); {{a, LP.q(k, 10)}, s} end)
        {{v, {ps, Map.new(t)}}, s}
      end)

    %{nodes: g.nodes, hidden: Map.new(hp), cpt: Map.new(cpt)}
  end

  defp bern(p, 1), do: p
  defp bern(p, 0), do: LP.qsub({1, 1}, p)

  defp joint(m, fixed) do
    hs = Map.keys(m.hidden)

    for obs <- Causal.assignments(m.nodes), Enum.all?(fixed, fn {k, v} -> obs[k] == v end), into: %{} do
      p =
        Enum.reduce(Causal.assignments(hs), {0, 1}, fn ha, acc ->
          a = Map.merge(obs, ha)
          ph = Enum.reduce(hs, {1, 1}, fn h, s -> LP.qmul(s, bern(m.hidden[h], ha[h])) end)
          po = Enum.reduce(m.nodes, {1, 1}, fn v, s -> if Map.has_key?(fixed, v), do: s, else: (fn {ps, t} -> LP.qmul(s, bern(t[Map.take(a, ps)], a[v])) end).(m.cpt[v]) end)
          LP.qadd(acc, LP.qmul(ph, po))
        end)

      {obs, p}
    end
  end

  defp prob(j, fixed), do: Enum.reduce(j, {0, 1}, fn {a, p}, s -> if Enum.all?(fixed, fn {k, v} -> a[k] == v end), do: LP.qadd(s, p), else: s end)

  # -------------------------------------------------------------------- mould

  defp adder(n) do
    ins = for i <- 0..(n - 1), x <- ["a", "b"], do: "#{x}#{i}"
    body = for i <- 0..(n - 1), do: (c = if(i == 0, do: "cin", else: "c#{i}"); "s#{i} = a#{i} ^ b#{i} ^ #{c}\nc#{i + 1} = maj(a#{i}, b#{i}, #{c})")
    "input #{Enum.join(ins, " ")} cin\noutput #{Enum.map_join(0..(n - 1), " ", &"s#{&1}")} c#{n}\n" <> Enum.join(body, "\n") <> "\n"
  end

  defp mould do
    {:ok, spec} = Rebis.parse(adder(10))
    trig = Enum.map_join(0..9, " & ", fn i -> if(rem(div(0x2B5, Integer.pow(2, i)), 2) == 1, do: "a#{i}", else: "~a#{i}") end) <> " & " <>
             Enum.map_join(0..9, " & ", fn i -> if(rem(div(0x14A, Integer.pow(2, i)), 2) == 1, do: "b#{i}", else: "~b#{i}") end)
    {:ok, trojan} = Rebis.parse(String.replace(adder(10), "s0 = a0 ^ b0 ^ cin", "t = #{trig}\ns0 = a0 ^ b0 ^ cin ^ t"))
    found = Qalib.certify(spec, trojan)
    # the control: 4096 random patterns, simulated on both
    st = :rand.seed_s(:exsss, {4096, 1, 1})
    {words, _} = Enum.map_reduce(spec.inputs, st, fn n, s -> {w, s} = Enum.map_reduce(1..4096, s, fn _, s -> :rand.uniform_s(2, s) end); {{n, w |> Enum.with_index() |> Enum.reduce(0, fn {b, j}, acc -> acc + (b - 1) * Integer.pow(2, j) end)}, s} end)
    mask = Integer.pow(2, 4096) - 1
    oa = Rebis.simulate(spec, Map.new(words), mask)
    ob = Rebis.simulate(trojan, Map.new(words), mask)
    random_found = Enum.any?(oa, fn {k, v} -> v != ob[k] end)
    found_ok = match?({:different, %{method: :sat}}, found)

    {:ok, small} = Rebis.parse(adder(4))
    styles = for s <- [:cells, :nand], do: (fn {:ok, v, _} -> {s, Qalib.certify(small, v)} end).(Qalib.to_verilog(small, style: s))
    {:ok, v, _} = Qalib.to_verilog(small)
    moved = Qalib.certify(small, String.replace(v, ".A(a0), .B(b0)", ".A(a0), .B(a0)", global: false))

    [check("mould: a 20-bit trojan trigger in a 21-input adder", if(found_ok, do: "found by SAT, re-simulated on both circuits", else: inspect(found)),
           "4,096 random patterns: #{if random_found, do: "found", else: "not found"}", "found exactly; random simulation blind", found_ok and not random_found),
     check("mould: a mapped adder read back (cells, NAND)", Enum.map_join(styles, ", ", fn {s, r} -> "#{s}: #{elem(r, 0)}" end), "one pin moved: #{elem(moved, 0)}",
           "equivalent both ways; the moved pin told apart", Enum.all?(styles, &match?({_, {:equivalent, _}}, &1)) and match?({:different, _}, moved))]
  end

  # ------------------------------------------------------------- palingenesis

  @plank "model.layers.1.mlp"

  defp palingenesis do
    map = Vapor.Quality.Round17.Tiny.config()
    {:ok, c} = Vapor.Model.Config.from_map(map)
    {:ok, spec, ws} = Lock.from_map(map, Vapor.Quality.Round17.Tiny.weights(c))
    :rand.seed(:exsss, {1, 2, 3})
    anchors = for _ <- 1..3, do: for(_ <- 1..6, do: :rand.uniform(95))
    targets = for i <- 1..16, do: (pr = for(_ <- 1..3, do: :rand.uniform(95)); pr ++ Vapor.Modal.Text.generate(spec, ws, pr, 12, temperature: 1.0, seed: i))
    names = Palingenesis.planks(ws)[@plank]
    noisy = fn t, seed, s -> Tensor.from_list(:f32, t.shape, Enum.zip_with(Tensor.to_floats(t), Tensor.to_floats(Tensor.random(:f32, t.shape, seed, scale: s)), &(&1 + &2))) end
    old = Map.new(ws, fn {k, v} -> if k in names, do: {k, noisy.(v, 900 + :erlang.phash2(k, 1000), 0.2)}, else: {k, v} end)
    clean = Map.take(ws, names)

    sham =
      Map.new(names, fn k ->
        d = Enum.zip_with(Tensor.to_floats(clean[k]), Tensor.to_floats(old[k]), &(&1 - &2))
        r = Tensor.to_floats(Tensor.random(:f32, old[k].shape, 4242 + :erlang.phash2(k, 1000)))
        sc = :math.sqrt(Enum.sum(Enum.map(d, &(&1 * &1)))) / :math.sqrt(Enum.sum(Enum.map(r, &(&1 * &1))))
        {k, Tensor.from_list(:f32, old[k].shape, Enum.zip_with(Tensor.to_floats(old[k]), r, &(&1 + &2 * sc)))}
      end)

    hull = fn w -> name = {:round17, System.unique_integer()}; {:ok, _} = Palingenesis.launch(name, %{spec: spec, weights: w}); name end
    try_ = fn w, plank_ts, opts -> n = hull.(w); r = Palingenesis.propose(n, @plank, plank_ts, opts); Palingenesis.retire(n); r end
    wide = [anchors: anchors, targets: targets, epsilon: :math.pi()]

    sham_r = try_.(old, sham, wide)
    real_r = try_.(old, clean, wide)
    brake_r = try_.(old, clean, Keyword.put(wide, :epsilon, 0.5))
    perm_r = try_.(ws, permuted(ws, names), anchors: anchors, epsilon: :math.pi())
    rand_r = try_.(ws, Map.new(names, fn k -> {k, noisy.(ws[k], 31 + :erlang.phash2(k, 100), 0.05)} end), anchors: anchors, epsilon: :math.pi())

    gate = fn {:ok, _, _} -> "admitted"; {:error, r} -> "refused (#{r.gate})" end
    drift = fn {:ok, _, r} -> r.drift.max; {:error, r} -> r[:drift] && r.drift.max end

    [check("palingenesis: a plank that restores the ship (target gate)", gate.(real_r), "a sham of the same norm: #{gate.(sham_r)}",
           "admitted; the sham refused at the target gate", match?({:ok, _, _}, real_r) and match?({:error, %{gate: "target"}}, sham_r)),
     check("palingenesis: the brake at ε = 0.5 for the same plank", gate.(brake_r) <> ", drift max #{fmt(drift.(brake_r))}", "at ε = π: #{gate.(real_r)}",
           "refused by the brake; admitted without it", match?({:error, %{gate: "drift"}}, brake_r) and match?({:ok, _, _}, real_r)),
     check("palingenesis: a block with its hidden units permuted", "drift max #{fmt(drift.(perm_r))}", "a random change (σ 0.05): drift max #{fmt(drift.(rand_r))}",
           "< 10⁻³ (rounding); the random change well above", drift.(perm_r) < 1.0e-3 and drift.(rand_r) > 100 * drift.(perm_r))]
  end

  defp permuted(ws, _names) do
    g = ws[@plank <> ".gate_proj.weight"]
    [n, k] = g.shape
    {perm, _} = Enum.map_reduce(1..n, :rand.seed_s(:exsss, {3, 5, 9}), fn _, s -> :rand.uniform_s(s) end) |> then(fn {us, s} -> {us |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1)), s} end)
    rows = fn t -> rs = t |> Tensor.to_floats() |> Enum.chunk_every(k) |> List.to_tuple(); Tensor.from_list(:f32, t.shape, Enum.flat_map(perm, &elem(rs, &1))) end
    d = ws[@plank <> ".down_proj.weight"]
    cols = d |> Tensor.to_floats() |> Enum.chunk_every(n) |> Enum.flat_map(fn r -> rt = List.to_tuple(r); Enum.map(perm, &elem(rt, &1)) end)
    %{@plank <> ".gate_proj.weight" => rows.(g), @plank <> ".up_proj.weight" => rows.(ws[@plank <> ".up_proj.weight"]),
      @plank <> ".down_proj.weight" => Tensor.from_list(:f32, d.shape, cols)}
  end

  defmodule Tiny do
    @moduledoc false
    # the tiny Qwen2 of the tests, without depending on test/support
    def config do
      %{"model_type" => "qwen2", "vocab_size" => 96, "hidden_size" => 64, "intermediate_size" => 96,
        "num_hidden_layers" => 2, "num_attention_heads" => 4, "num_key_value_heads" => 2,
        "max_position_embeddings" => 32, "rms_norm_eps" => 1.0e-5, "rope_theta" => 10_000.0,
        "tie_word_embeddings" => true, "hidden_act" => "silu"}
    end

    def weights(c) do
      c
      |> Vapor.Model.Decoder.expected_weights()
      |> Enum.with_index(1000)
      |> Map.new(fn
        {{name, shape, :norm}, i} ->
          r = Vapor.Tensor.random(:f32, shape, i, scale: 0.2) |> Vapor.Tensor.to_floats()
          {name, Vapor.Tensor.from_list(:f32, shape, Enum.map(r, &(&1 + 1.0)))}

        {{name, shape, kind}, i} ->
          scale = cond do kind == :bias or kind == :vector -> 0.1; name == "model.embed_tokens.weight" -> 1.0; true -> 0.2 end
          {name, Vapor.Tensor.random(:f32, shape, i, scale: scale)}
      end)
    end
  end

  # --------------------------------------------------------- recommendations

  defp recommendations do
    planted = table(fn u, i -> 3.0 + :math.cos(u * 0.7) * :math.cos(i * 0.9) * 2 + :math.sin(u * 1.3) * :math.sin(i * 0.4) * 2 end)
    flat = table(fn u, i -> 3.0 + rem(u, 3) * 0.2 + :erlang.phash2({u, i, 7}, 100) / 100 end)
    {:ok, a} = Recommend.evaluate(planted)
    {:ok, b} = Recommend.evaluate(flat)

    check("recommendations: planted interactions (40 × 24, 60 % observed)", "#{a.verdict}, RMSE #{fmt(a.rmse.model)} vs biases #{fmt(a.rmse.biases)}",
          "unstructured ratings: #{b.verdict}", "signal; the unstructured table not", a.verdict == "signal" and b.verdict != "signal")
  end

  defp table(f) do
    ratings = for u <- 0..39, i <- 0..23, :erlang.phash2({u, i}, 10) < 6, do: {u, i, f.(u, i)}
    %{users: Enum.map(0..39, &"u#{&1}"), items: Enum.map(0..23, &"i#{&1}"), ratings: ratings}
  end

  # ---------------------------------------------------------------------- SVD

  defp svd do
    # A = Q₁ diag(1, 10⁻⁴, 10⁻⁸) Q₂ᵀ with orthogonal Q from a seeded Gram–Schmidt
    q1 = orth(5, 3, 1)
    q2 = orth(3, 3, 2)
    sig = [1.0, 1.0e-4, 1.0e-8]
    a = Dense.matmul(Enum.map(q1, fn r -> Enum.zip_with(r, sig, &(&1 * &2)) end), Dense.transpose(q2))
    jac = Dense.svd(a).s |> List.last()
    {vals, _} = Dense.eigh(Dense.matmul(Dense.transpose(a), a))
    gram = vals |> Enum.map(&:math.sqrt(max(&1, 0.0))) |> Enum.min()
    rel = fn x -> abs(x - 1.0e-8) / 1.0e-8 end

    check("SVD: the smallest singular value of a matrix with condition 10⁸", "one-sided Jacobi: relative error #{fmt(rel.(jac))}",
          "√eig(AᵀA): relative error #{fmt(rel.(gram))}", "< 10⁻⁶; the Gram route loses it", rel.(jac) < 1.0e-6 and rel.(gram) > 1.0e-2)
  end

  defp orth(m, n, seed) do
    {cols, _} = Enum.map_reduce(1..n, :rand.seed_s(:exsss, {seed, 2, 3}), fn _, s -> Enum.map_reduce(1..m, s, fn _, s -> :rand.normal_s(s) end) end)

    cols
    |> Enum.reduce([], fn c, acc ->
      v = Enum.reduce(acc, c, fn q, v -> Dense.axpy(-Dense.dot(q, v), q, v) end)
      nv = :math.sqrt(Dense.dot(v, v))
      acc ++ [Enum.map(v, &(&1 / nv))]
    end)
    |> Dense.transpose()
  end

  defp fmt(nil), do: "—"
  defp fmt(x) when is_float(x), do: :erlang.float_to_binary(x, [{:scientific, 2}])
  defp fmt(x), do: to_string(x)
end
