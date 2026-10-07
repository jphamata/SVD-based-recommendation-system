defmodule Vapor.Solve.Ensemble do
  @moduledoc """
  Ensembles of a typed-in ODE system on the native worker
  (docs/BANCADA.md §3): the HPC path of the workbench.

  Uncertainty quantification and parameter sweeps integrate the same
  system thousands of times with different parameters. Here the system's
  right-hand side is **compiled into a vapor program** — the expression
  tree becomes terms of the algebra (`+ − × ÷`, `exp`, `log`, `tanh`,
  `√`, `min`/`max`, comparisons and `if` as selections), four RK4 stages
  per step, several steps unrolled per call — and run over a batch of
  members at once on the native worker, the state resident in a session
  between calls. It is binary32, and it is **canonical**: the same bits
  on every substrate, checked here against the exact oracle on the first
  call.

  Uncertain quantities are written with `~`:

      x' = v
      v' = -k/m*x - c/m*v
      k ~ normal(4, 0.2); c ~ uniform(0.05, 0.2); m = 1
      x(0) ~ normal(1, 0.05); v(0) = 0
      t = 0 .. 10
      members = 4096; h = 0.01

  The result is the band (mean, 5 %, 50 %, 95 %) of every state over
  time, the agreement with the same members integrated in binary64 on
  the BEAM, and the speed of each.

  The algebra has no sine or cosine: a system using them is refused with
  that reason, never approximated in silence (the binary64 solvers of
  `Vapor.Solve` take it).
  """
  alias Vapor.{Expr, Solve, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Runtime.{Native, Session, Substrates}

  @max_members 65_536

  @doc "Run an ensemble described by text. Options: `worker` (default: a native worker if present), `reference` (members integrated in binary64 for the comparison, default 64)."
  def run(text, opts \\ []) do
    with {:ok, spec} <- parse(text),
         {:ok, sys} <- Solve.parse_ode(spec.ode_text),
         {:ok, plan} <- plan(sys, spec) do
      simulate(plan, sys, spec, opts)
    end
  end

  # ---------------------------------------------------------------- parse --

  defp parse(text) do
    {dists, rest, members, h} =
      Solve.statements(text)
      |> Enum.reduce({%{}, [], 1024, nil}, fn {s, _}, {d, r, m, h} ->
        cond do
          x = Regex.run(~r/^([\p{L}_][\p{L}\p{N}_]*)(\(\s*0\s*\))?\s*~\s*(normal|uniform|lognormal)\s*\(\s*([^,]+)\s*,\s*([^)]+)\)$/u, s) ->
            [_, n, init, kind, a, b] = x
            {Map.put(d, {n, init != ""}, {kind, a, b}), r, m, h}
          x = Regex.run(~r/^(?:members|membros)\s*=\s*(\d+)$/u, s) -> {d, r, String.to_integer(Enum.at(x, 1)), h}
          x = Regex.run(~r/^h\s*=\s*(.+)$/u, s) -> {d, r, m, Enum.at(x, 1)}
          true -> {d, r ++ [s], m, h}
        end
      end)

    # uncertain quantities enter the deterministic parse at their centre
    centre = for {{n, init?}, {kind, a, b}} <- dists, do: (if init?, do: "#{n}(0) = #{centre(kind, a, b)}", else: "#{n} = #{centre(kind, a, b)}")
    members = members |> max(1) |> min(@max_members)
    {:ok, %{dists: dists, ode_text: Enum.join(centre ++ rest, "\n"), members: members, h: h}}
  end

  defp centre("uniform", a, b), do: "((#{a}) + (#{b}))/2"
  defp centre("lognormal", a, _), do: "exp(#{a})"
  defp centre(_, a, _), do: a

  # ----------------------------------------------------------------- plan --

  defp plan(sys, spec) do
    {t0, t1} = sys.span
    h = case spec.h do nil -> (t1 - t0) / 1000; s -> (case Float.parse(s) do {v, _} -> v; _ -> (t1 - t0) / 1000 end) end
    steps = max(1, round((t1 - t0) / h))
    h = (t1 - t0) / steps
    unroll = Enum.find([10, 8, 5, 4, 2, 1], &(rem(steps, &1) == 0))
    uncertain_params = for {{n, false}, _} <- spec.dists, do: n
    # parameters that vary per member stay symbolic; the rest are folded in
    fixed = Map.drop(sys.params, uncertain_params) |> Map.new(fn {k, v} -> {k, {:n, v}} end)
    raw = for s <- sys.states, do: sys.rhs[s] |> Expr.subst(fixed) |> Expr.simplify()

    case Enum.flat_map(raw, &unsupported/1) |> Enum.uniq() do
      [] -> {:ok, %{h: h, steps: steps, unroll: unroll, calls: div(steps, unroll), params: Enum.sort(uncertain_params), rhs: raw}}
      fs -> {:error, "the native path has no #{Enum.join(fs, ", ")} (the algebra's functions are + − × ÷ exp log tanh √ min max and comparisons); solve this system in binary64 with method = rk45"}
    end
  end

  @native ~w(exp log ln tanh sqrt abs min max sq if step sinh cosh)
  defp unsupported({:f, f, as}), do: (if f in @native, do: [], else: [f]) ++ Enum.flat_map(as, &unsupported/1)
  defp unsupported({:^, a, {:n, _}}), do: unsupported(a)
  defp unsupported({:^, a, b}), do: unsupported(a) ++ unsupported(b)
  defp unsupported({op, a, b}) when is_atom(op), do: unsupported(a) ++ unsupported(b)
  defp unsupported({:neg, a}), do: unsupported(a)
  defp unsupported(_), do: []

  # ------------------------------------------------------ expression → term

  @doc false
  def term({:n, x}, _), do: T.splat(x)
  def term({:q, x, _, _}, _), do: T.splat(x)
  def term({:v, n}, env), do: Map.fetch!(env, n)
  def term({:neg, a}, env), do: T.neg(term(a, env))
  def term({:+, a, b}, env), do: T.add(term(a, env), term(b, env))
  def term({:-, a, b}, env), do: T.sub(term(a, env), term(b, env))
  def term({:*, a, b}, env), do: T.mul(term(a, env), term(b, env))
  def term({:/, a, b}, env), do: T.divide(term(a, env), term(b, env))

  def term({:^, a, {:n, k}}, env) when k == trunc(k) and abs(k) <= 16 do
    ta = term(a, env)
    p = Enum.reduce(1..max(trunc(abs(k)), 1), nil, fn _, acc -> if acc, do: T.mul(acc, ta), else: ta end)
    cond do
      k == 0 -> T.splat(1.0)
      k < 0 -> T.rcp(p)
      true -> p
    end
  end

  def term({:^, a, {:n, 0.5}}, env), do: sqrt(term(a, env))
  def term({:^, a, b}, env), do: T.exp(T.mul(term(b, env), T.log(term(a, env))))
  def term({:<, a, b}, env), do: T.sel(term(a, env), term(b, env), T.splat(1.0), T.splat(0.0))
  def term({:>, a, b}, env), do: T.sel(term(b, env), term(a, env), T.splat(1.0), T.splat(0.0))
  def term({:<=, a, b}, env), do: T.sel(term(b, env), term(a, env), T.splat(0.0), T.splat(1.0))
  def term({:>=, a, b}, env), do: T.sel(term(a, env), term(b, env), T.splat(0.0), T.splat(1.0))
  def term({:f, "if", [c, a, b]}, env), do: branch(c, term(a, env), term(b, env), env)
  def term({:f, "step", [a]}, env), do: T.sel(term(a, env), T.splat(0.0), T.splat(0.0), T.splat(1.0))
  def term({:f, "exp", [a]}, env), do: T.exp(term(a, env))
  def term({:f, l, [a]}, env) when l in ["log", "ln"], do: T.log(term(a, env))
  def term({:f, "tanh", [a]}, env), do: T.tanh(term(a, env))
  def term({:f, "sqrt", [a]}, env), do: sqrt(term(a, env))
  def term({:f, "sq", [a]}, env), do: (t = term(a, env); T.mul(t, t))
  def term({:f, "abs", [a]}, env), do: (t = term(a, env); T.max(t, T.neg(t)))
  def term({:f, "sinh", [a]}, env), do: (t = term(a, env); T.mul(T.splat(0.5), T.sub(T.exp(t), T.exp(T.neg(t)))))
  def term({:f, "cosh", [a]}, env), do: (t = term(a, env); T.mul(T.splat(0.5), T.add(T.exp(t), T.exp(T.neg(t)))))
  def term({:f, "min", as}, env), do: as |> Enum.map(&term(&1, env)) |> Enum.reduce(&T.min(&2, &1))
  def term({:f, "max", as}, env), do: as |> Enum.map(&term(&1, env)) |> Enum.reduce(&T.max(&2, &1))

  # √x as 1/rsqrt(x): exact 0 at 0 (x·rsqrt(x) would be 0·∞)
  defp sqrt(t), do: T.rcp(T.rsqrt(t))

  defp branch({:<, a, b}, x, y, env), do: T.sel(term(a, env), term(b, env), x, y)
  defp branch({:>, a, b}, x, y, env), do: T.sel(term(b, env), term(a, env), x, y)
  defp branch({:<=, a, b}, x, y, env), do: T.sel(term(b, env), term(a, env), y, x)
  defp branch({:>=, a, b}, x, y, env), do: T.sel(term(a, env), term(b, env), y, x)
  defp branch(c, x, y, env), do: T.sel(T.splat(0.0), T.mul(term(c, env), term(c, env)), x, y)

  @doc false
  # one call of the program: `unroll` RK4 steps of size h over a batch of b members
  def program(plan, sys, b) do
    sidx = Enum.with_index(sys.states)
    states = Map.new(sidx, fn {s, i} -> {s, T.input(:"s#{i}", :f32, [1, b])} end)
    params = Map.new(Enum.with_index(plan.params), fn {p, i} -> {p, T.input(:"p#{i}", :f32, [1, b])} end)
    t_in = T.input(:t, :f32, [1, b])
    h = plan.h
    f = fn t, st -> for r <- plan.rhs, do: term(r, Map.merge(params, Map.put(st, "t", t))) end
    bind = fn {lets, name}, term -> {T.ref(name, term), [{name, term} | lets]} end

    {lets, st, t} =
      Enum.reduce(1..plan.unroll, {[], states, t_in}, fn k, {lets, st, t} ->
        add = fn st, ks, c -> Map.new(Enum.zip(sys.states, ks), fn {s, kk} -> {s, T.add(st[s], T.mul(T.splat(c), kk))} end) end
        bindall = fn lets, ks, tag -> Enum.map_reduce(Enum.with_index(ks), lets, fn {kk, i}, lets -> (({r, l} = bind.({lets, :"#{tag}#{i}"}, kk)); {r, l}) end) end
        th = T.add(t, T.splat(h / 2))
        {k1, lets} = bindall.(lets, f.(t, st), "k#{k}a")
        {k2, lets} = bindall.(lets, f.(th, add.(st, k1, h / 2)), "k#{k}b")
        {k3, lets} = bindall.(lets, f.(th, add.(st, k2, h / 2)), "k#{k}c")
        tn = T.add(t, T.splat(h))
        {k4, lets} = bindall.(lets, f.(tn, add.(st, k3, h)), "k#{k}d")
        {next, lets} =
          Enum.zip([sys.states, k1, k2, k3, k4]) |> Enum.map_reduce(lets, fn {s, a, bb, c, d}, lets ->
            incr = T.mul(T.splat(h / 6), T.add(T.add(a, d), T.mul(T.splat(2.0), T.add(bb, c))))
            {r, l} = bind.({lets, :"y#{k}_#{s |> :erlang.phash2()}"}, T.add(st[s], incr))
            {{s, r}, l}
          end)
        {tr, lets} = bind.({lets, :"t#{k}"}, tn)
        {lets, Map.new(next), tr}
      end)

    outs = (for {s, i} <- sidx, do: {:"s#{i}_next", st[s]}) ++ [t_next: t]
    Vapor.Program.new(outs, lets: Enum.reverse(lets), state: (for {_, i} <- sidx, do: {:"s#{i}", :"s#{i}_next"}) ++ [t: :t_next])
  end

  # ------------------------------------------------------------- simulate --

  defp draw({"normal", a, b}, seed, k), do: num(a) + num(b) * gauss(seed, k)
  defp draw({"lognormal", a, b}, seed, k), do: :math.exp(num(a) + num(b) * gauss(seed, k))
  defp draw({"uniform", a, b}, seed, k), do: num(a) + (num(b) - num(a)) * Vapor.Sampler.uniform(seed, k)

  defp num(s), do: Expr.eval(Expr.parse!(s))

  defp gauss(seed, k) do
    u1 = max(Vapor.Sampler.uniform(seed, 2 * k), 1.0e-12)
    u2 = Vapor.Sampler.uniform(seed, 2 * k + 1)
    :math.sqrt(-2 * :math.log(u1)) * :math.cos(2 * :math.pi() * u2)
  end

  defp simulate(plan, sys, spec, opts) do
    b = spec.members
    seed = Keyword.get(opts, :seed, 1)
    sample = fn {n, init?}, def_v, salt ->
      case Map.fetch(spec.dists, {n, init?}) do
        {:ok, d} -> for k <- 0..(b - 1), do: draw(d, seed * 7919 + salt, k)
        :error -> List.duplicate(def_v, b)
      end
    end

    y0 = for {s, i} <- Enum.with_index(sys.states), do: sample.({s, true}, sys.init[s], 100 + i)
    pv = for {p, i} <- Enum.with_index(plan.params), do: sample.({p, false}, sys.params[p], 200 + i)
    t0 = elem(sys.span, 0)

    prog = program(plan, sys, b)
    {:ok, comp} = Vapor.Compile.Lower.lower(prog)
    tens = fn rows -> Tensor.from_list(:f32, [1, b], rows) end
    env = Map.new(Enum.with_index(y0), fn {r, i} -> {:"s#{i}", tens.(r)} end)
    |> Map.merge(Map.new(Enum.with_index(pv), fn {r, i} -> {:"p#{i}", tens.(r)} end))
    |> Map.put(:t, tens.(List.duplicate(t0, b)))
    want = for({_, i} <- Enum.with_index(sys.states), do: :"s#{i}_next") ++ [:t_next]
    worker = Keyword.get_lazy(opts, :worker, &Vapor.Modal.Runner.worker/0)

    {native_us, {traj, substrate, parity}} =
      :timer.tc(fn ->
        case worker do
          nil -> {oracle_run(comp, env, plan, sys, want), "oracle", nil}
          w ->
            {:ok, sess} = Session.open(w, comp, isa: Substrates.host_isa())
            try do
              {traj, first} =
                Enum.map_reduce(1..plan.calls, nil, fn k, first ->
                  {:ok, o, _} = Session.step(sess, if(k == 1, do: env, else: %{}), want)
                  {Enum.map(Enum.with_index(sys.states), fn {_, i} -> Tensor.to_floats(o[:"s#{i}_next"]) end), first || o}
                end)
              {traj, Atom.to_string(Substrates.host_isa()), first}
            after
              Session.close(sess)
            end
        end
      end)

    # rung: the first call on the oracle, bit for bit (outside the timing)
    parity =
      case parity do
        nil -> nil
        first ->
          {:ok, ro} = Native.run_oracle(comp, env)
          Enum.all?(Enum.with_index(sys.states), fn {_, i} -> ro.outputs[:"s#{i}_next"].data == first[:"s#{i}_next"].data end)
      end

    # the binary64 reference on the first members, same RK4, same h
    nref = min(Keyword.get(opts, :reference, 64), b)
    {ref_us, ref_final} = :timer.tc(fn -> reference(plan, sys, y0, pv, nref, t0) end)
    final = List.last(traj)
    rel = for {s_ref, s_nat} <- Enum.zip(ref_final, final), {a, nat} <- Enum.zip(s_ref, Enum.take(s_nat, nref)), do: abs(a - nat) / max(abs(a), 1.0e-6)
    per_member_beam = ref_us / max(nref, 1)
    every = max(1, div(plan.calls, 120))
    ts = for k <- 1..plan.calls, do: t0 + k * plan.unroll * plan.h

    bands =
      Map.new(Enum.with_index(sys.states), fn {s, i} ->
        pts = for {row, k} <- Enum.with_index(traj), rem(k + 1, every) == 0 or k == plan.calls - 1, do: stats(Enum.at(row, i))
        {s, pts}
      end)

    {:ok, %{members: b, steps: plan.steps, h: plan.h, unroll: plan.unroll, calls: plan.calls, substrate: substrate, oracle_parity: parity,
            t: for({t, k} <- Enum.with_index(ts), rem(k + 1, every) == 0 or k == plan.calls - 1, do: t), bands: bands, states: sys.states, uncertain: Enum.map(Map.keys(spec.dists), fn {n, i} -> if i, do: "#{n}(0)", else: n end) |> Enum.sort(),
            f64_agreement: %{members: nref, max_relative: Enum.max(rel, fn -> 0.0 end), median_relative: median(rel)},
            native_ms: native_us / 1000, beam_f64_ms_estimate: per_member_beam * b / 1000, speedup: per_member_beam * b / max(native_us, 1),
            final: Map.new(Enum.with_index(sys.states), fn {s, i} -> {s, stats(Enum.at(final, i))} end)}}
  end

  defp oracle_run(comp, env, plan, sys, want) do
    {traj, _} =
      Enum.map_reduce(1..plan.calls, env, fn _, cur ->
        {:ok, r} = Native.run_oracle(comp, cur)
        o = Map.take(r.outputs, want)
        next = Map.merge(cur, Map.new(Enum.with_index(sys.states), fn {_, i} -> {:"s#{i}", o[:"s#{i}_next"]} end)) |> Map.put(:t, o[:t_next])
        {Enum.map(Enum.with_index(sys.states), fn {_, i} -> Tensor.to_floats(o[:"s#{i}_next"]) end), next}
      end)
    traj
  end

  defp reference(plan, sys, y0, pv, nref, t0) do
    f = Expr.compile(plan.rhs, ["t" | sys.states] ++ plan.params)
    per =
      for m <- 0..(nref - 1) do
        y = Enum.map(y0, &Enum.at(&1, m))
        p = Enum.map(pv, &Enum.at(&1, m))
        rhs = fn t, y -> f.(List.to_tuple([t | y] ++ p)) end
        h = plan.h
        Enum.reduce(0..(plan.steps - 1), y, fn k, y ->
          t = t0 + k * h
          ax = fn y, kk, c -> Enum.zip_with(y, kk, &(&1 + c * &2)) end
          k1 = rhs.(t, y); k2 = rhs.(t + h / 2, ax.(y, k1, h / 2)); k3 = rhs.(t + h / 2, ax.(y, k2, h / 2)); k4 = rhs.(t + h, ax.(y, k3, h))
          Enum.zip_with([y, k1, k2, k3, k4], fn [a, b, c, d, e] -> a + h / 6 * (b + 2 * c + 2 * d + e) end)
        end)
      end
    for i <- 0..(length(sys.states) - 1), do: Enum.map(per, &Enum.at(&1, i))
  end

  defp stats(xs) do
    s = Enum.sort(xs)
    n = length(s)
    q = fn p -> Enum.at(s, min(n - 1, trunc(p * (n - 1) + 0.5))) end
    %{mean: Enum.sum(s) / n, p05: q.(0.05), p50: q.(0.5), p95: q.(0.95), min: hd(s), max: List.last(s)}
  end

  defp median([]), do: 0.0
  defp median(xs), do: (s = Enum.sort(xs); Enum.at(s, div(length(s), 2)))
end
