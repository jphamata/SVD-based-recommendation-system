defmodule Vapor.Finance.MonteCarlo do
  @moduledoc """
  Monte Carlo under geometric Brownian motion, **compiled into a vapor
  program** and run on the native worker (docs/FINANCAS.md §5).

  The pain this addresses is not speed alone: a risk number that changes
  when the job moves to another machine, another thread count or a GPU is
  a model-risk finding waiting to happen (parallel reductions in a
  different order, a fused multiply-add here and not there, a different
  `exp`). Here the path step is a term of vapor's algebra — the normal
  deviate by Wichura's AS241 (PPND7) from a uniform, `exp` and `log` the
  canonical microprograms — so the paths are **the same bits on every
  substrate and every thread count**, and the first call is compared bit
  for bit with the exact oracle.

  Per step, for each path: L ← L + μΔ + σ√Δ·Φ⁻¹(u); A ← A + e^L;
  G ← G + L; alive ← alive · [L > ln(B/S₀)]. One call advances `unroll`
  steps of a batch of paths, the state resident in the worker's session.
  Payoffs (European, arithmetic and geometric Asian, down-and-out barrier)
  are formed afterwards in binary64, in a fixed order.

  The answer carries its own judges:

  * the 99 % confidence interval and whether it covers the closed form
    (Black–Scholes for the European; Kemna–Vorst for the discretely
    monitored geometric Asian);
  * the arithmetic Asian with the geometric one as **control variate**
    (the variance reduction is reported);
  * the same uniforms integrated in binary64 on the BEAM (agreement);
  * parity with the oracle, and with another thread count when asked.

  The control (`drift: :no_ito`) forgets Itô's −σ²/2: its interval must
  exclude the closed form — a biased estimator is caught, not averaged.
  """
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Finance.{Num, Options}
  alias Vapor.Runtime.{Native, Session, Substrates}
  alias Vapor.Tensor
  import Bitwise

  @max_paths 65_536
  @max_steps 1024

  # AS241 PPND7 (Wichura 1988), 7 significant digits: enough for binary32 paths
  @a [3.3871327179, 50.434271938, 159.29113202, 59.109374720]
  @b [1.0, 17.895169469, 78.757757664, 67.187563600]
  @c [1.4234372777, 2.7568153900, 1.3067284816, 0.17023821103]
  @d [1.0, 0.73700164250, 0.12021132975]
  @e [6.6579051150, 3.0812263860, 0.42868294337, 0.017337203997]
  @f [1.0, 0.24197894225, 0.012258202635]

  @doc "The 24-bit uniform (k + ½)/2²⁴ of stream `seed` at index `i` — exactly representable in binary32, never 0 or 1."
  def u24(seed, i) do
    x = band(seed * 0x9E3779B97F4A7C15 + (i + 1) * 0xBF58476D1CE4E5B9, 0xFFFF_FFFF_FFFF_FFFF)
    x = band(bxor(x, x >>> 30) * 0xBF58476D1CE4E5B9, 0xFFFF_FFFF_FFFF_FFFF)
    x = band(bxor(x, x >>> 27) * 0x94D049BB133111EB, 0xFFFF_FFFF_FFFF_FFFF)
    x = bxor(x, x >>> 31)
    ((x >>> 40) + 0.5) / 16_777_216
  end

  defp poly(cs, r), do: cs |> Enum.reverse() |> Enum.reduce(nil, fn c, acc -> if acc, do: T.add(T.mul(acc, r), T.splat(c)), else: T.splat(c) end)

  @doc false
  # Φ⁻¹(u) as a term: the central rational for |u − ½| ≤ 0.425, the tail rationals in r = √(−ln min(u, 1−u))
  def ppnd7(u) do
    q = T.sub(u, T.splat(0.5))
    r = T.sub(T.splat(0.180625), T.mul(q, q))
    central = T.divide(T.mul(q, poly(@a, r)), poly(@b, r))
    m = T.min(u, T.sub(T.splat(1.0), u))
    rr = T.rcp(T.rsqrt(T.neg(T.log(m))))           # √(−ln m), exact 0 at 0
    r1 = T.sub(rr, T.splat(1.6))
    r2 = T.sub(rr, T.splat(5.0))
    tail_mid = T.divide(poly(@c, r1), poly(@d, r1))
    tail_far = T.divide(poly(@e, r2), poly(@f, r2))
    tail = T.sel(T.splat(5.0), rr, tail_far, tail_mid)          # rr > 5 → far
    signed = T.sel(q, T.splat(0.0), T.neg(tail), tail)           # q < 0 → −tail
    aq = T.max(q, T.neg(q))
    T.sel(T.splat(0.425), aq, signed, central)                   # |q| > 0.425 → tail
  end

  @doc "Φ⁻¹ by PPND7 in binary64 (the same rational; for the reference)."
  def ppnd7_f64(p) do
    h = fn cs, x -> cs |> Enum.reverse() |> Enum.reduce(0.0, &(&2 * x + &1)) end
    q = p - 0.5
    if abs(q) <= 0.425 do
      r = 0.180625 - q * q
      q * h.(@a, r) / h.(@b, r)
    else
      r = :math.sqrt(-:math.log(min(p, 1 - p)))
      x = if r <= 5, do: (r = r - 1.6; h.(@c, r) / h.(@d, r)), else: (r = r - 5; h.(@e, r) / h.(@f, r))
      if q < 0, do: -x, else: x
    end
  end

  # floor of a non-negative binary32 x < 2²³, exactly: round-to-nearest by the 2²³ trick, then one step down if it rounded up
  defp floor_t(x) do
    r = T.sub(T.add(x, T.splat(8_388_608.0)), T.splat(8_388_608.0))
    T.sub(r, T.sel(x, r, T.splat(1.0), T.splat(0.0)))
  end

  # (a·s) mod m, exactly in binary32 (a·s < 2²³; the quotient is never within half an ulp of an integer, the guards are belts)
  defp lcg(s, a, m) do
    p = T.mul(T.splat(a), s)
    x = T.sub(p, T.mul(T.splat(m), floor_t(T.divide(p, T.splat(m)))))
    x = T.sel(x, T.splat(0.0), T.add(x, T.splat(m)), x)
    T.sel(x, T.splat(m), x, T.sub(x, T.splat(m)))
  end

  @wh [{171.0, 30269.0}, {172.0, 30307.0}, {170.0, 30323.0}]

  @doc false
  # rng :host — uniforms u1…u_unroll are inputs; rng :device — Wichmann–Hill (1982) in the program, three
  # Lehmer generators whose products stay below 2²³, so every state update is exact in binary32
  def program(b, unroll, mu_dt, vol_dt, log_barrier, rng \\ :host) do
    vec = fn n -> T.input(n, :f32, [1, b]) end
    st = %{l: vec.(:l), a: vec.(:a), g: vec.(:g), alive: vec.(:alive), w1: vec.(:w1), w2: vec.(:w2), w3: vec.(:w3)}
    # every intermediate is named once (a let) and referred to by name
    bind = fn lets, name, term -> {T.ref(name, term), [{name, term} | lets]} end
    {lets, st} =
      Enum.reduce(1..unroll, {[], st}, fn j, {lets, st} ->
        {u, lets, st} =
          case rng do
            :host -> {vec.(:"u#{j}"), lets, st}
            :device ->
              {w, lets} = Enum.zip([st.w1, st.w2, st.w3], @wh) |> Enum.with_index(1) |> Enum.map_reduce(lets, fn {{sv, {a, m}}, c}, lets -> bind.(lets, :"w#{c}_#{j}", lcg(sv, a, m)) end)
              [w1, w2, w3] = w
              [{_, m1}, {_, m2}, {_, m3}] = @wh
              sum = T.add(T.add(T.divide(w1, T.splat(m1)), T.divide(w2, T.splat(m2))), T.divide(w3, T.splat(m3)))
              {ss, lets} = bind.(lets, :"us#{j}", sum)
              frac = T.sub(ss, floor_t(ss))
              {uu, lets} = bind.(lets, :"u#{j}", T.min(T.max(frac, T.splat(2.9802322e-8)), T.splat(0.99999994)))
              {uu, lets, %{st | w1: w1, w2: w2, w3: w3}}
          end
        {z, lets} = bind.(lets, :"z#{j}", ppnd7(u))
        {l, lets} = bind.(lets, :"l#{j}", T.add(st.l, T.add(T.splat(mu_dt), T.mul(T.splat(vol_dt), z))))
        {a, lets} = bind.(lets, :"a#{j}", T.add(st.a, T.exp(l)))
        {g, lets} = bind.(lets, :"g#{j}", T.add(st.g, l))
        {alive, lets} =
          if log_barrier, do: bind.(lets, :"v#{j}", T.mul(st.alive, T.sel(T.splat(log_barrier), l, T.splat(1.0), T.splat(0.0)))), else: {st.alive, lets}
        {lets, %{st | l: l, a: a, g: g, alive: alive}}
      end)
    outs = [l_next: st.l, a_next: st.a, g_next: st.g, alive_next: st.alive]
    state = [l: :l_next, a: :a_next, g: :g_next, alive: :alive_next]
    {outs, state} = if rng == :device, do: {outs ++ [w1_next: st.w1, w2_next: st.w2, w3_next: st.w3], state ++ [w1: :w1_next, w2: :w2_next, w3: :w3_next]}, else: {outs, state}
    Vapor.Program.new(outs, lets: Enum.reverse(lets), state: state)
  end

  @doc "Wichmann–Hill seeds for path p: three integers in [1, mᵢ − 1] drawn from the splitmix stream."
  def wh_seeds(seed, p) do
    for {{_, m}, c} <- Enum.with_index(@wh) do
      1.0 + Float.floor(u24(seed * 31 + c + 7, p) * (m - 1))
    end
  end

  @doc "The same Wichmann–Hill stream in binary64 on the BEAM (exact integers; the uniform rounded to binary32 as on the device)."
  def wh_uniforms(seed, p, n) do
    [s1, s2, s3] = wh_seeds(seed, p) |> Enum.map(&trunc/1)
    {us, _} = Enum.map_reduce(1..n, {s1, s2, s3}, fn _, {a, b, c} ->
      a = rem(171 * a, 30269); b = rem(172 * b, 30307); c = rem(170 * c, 30323)
      # the device sums the three quotients in binary32: reproduce that rounding exactly
      f = fn x -> Vapor.F32.to_float(Vapor.F32.from_float(x)) end
      sum = f.(f.(f.(a / 30269) + f.(b / 30307)) + f.(c / 30323))
      u = f.(sum - Float.floor(sum)) |> max(2.9802322e-8) |> min(0.99999994)
      {u, {a, b, c}}
    end)
    us
  end

  @doc """
  Price by simulation. Options (all numbers): `s0 k t r q sigma`, `type`
  (`:call`/`:put`), `paths` (default 4096), `steps` (monitoring dates,
  default 64), `barrier` (down-and-out level, optional), `seed`, `threads`
  (also run with this many worker threads and compare the bits),
  `drift: :no_ito` (the control), `reference` (paths re-simulated in
  binary64, default 512).
  """
  def price(o) do
    s0 = o[:s0] * 1.0; k = o[:k] * 1.0; t = o[:t] * 1.0; r = (o[:r] || 0.0) * 1.0; q = (o[:q] || 0.0) * 1.0; sg = o[:sigma] * 1.0
    type = o[:type] || :call
    b = (o[:paths] || 4096) |> max(64) |> min(@max_paths)
    n = (o[:steps] || 64) |> max(1) |> min(@max_steps)
    seed = o[:seed] || 1
    dt = t / n
    ito = if o[:drift] == :no_ito, do: 0.0, else: -0.5 * sg * sg
    mu_dt = (r - q + ito) * dt
    vol_dt = sg * :math.sqrt(dt)
    log_b = if o[:barrier], do: :math.log(o[:barrier] / s0), else: nil
    # lowering costs ~0.6 s per unrolled step and target (the Lean-extracted allocation checker is quadratic),
    # a call ~1 ms: one step per call, for the host's ISA (all four ISAs with `all_targets: true`)
    unroll = o[:unroll] || 1
    n = div(n, unroll) * unroll
    calls = div(n, unroll)
    rng = o[:rng] || :device

    prog = program(b, unroll, mu_dt, vol_dt, log_b, rng)
    lower_opts = if o[:all_targets], do: [], else: [targets: host_targets()]
    {lower_us, {:ok, comp}} = :timer.tc(fn -> Vapor.Compile.Lower.lower(prog, lower_opts) end)
    tens = fn rows -> Tensor.from_list(:f32, [1, b], rows) end
    uniforms =
      case rng do
        :host -> fn call -> Map.new(1..unroll, fn j -> step = (call - 1) * unroll + (j - 1); {:"u#{j}", tens.(for p <- 0..(b - 1), do: u24(seed, step * @max_paths + p))} end) end
        :device -> fn _ -> %{} end
      end
    zero = tens.(List.duplicate(0.0, b))
    init = %{l: zero, a: zero, g: zero, alive: tens.(List.duplicate(1.0, b))}
    init =
      if rng == :device do
        seeds = for p <- 0..(b - 1), do: wh_seeds(seed, p)
        Map.merge(init, %{w1: tens.(Enum.map(seeds, &Enum.at(&1, 0))), w2: tens.(Enum.map(seeds, &Enum.at(&1, 1))), w3: tens.(Enum.map(seeds, &Enum.at(&1, 2)))})
      else
        init
      end
    want = [:l_next, :a_next, :g_next, :alive_next]

    {us, run} = :timer.tc(fn -> simulate(comp, init, uniforms, calls, want, o[:worker], 1) end)
    final = run.final

    # the oracle rung: the same step compiled for 64 lanes, run by the exact oracle on the first 64 paths'
    # uniforms, must give the first 64 lanes of the worker's first call bit for bit (lanes are independent)
    parity =
      case run.first do
        nil -> nil
        first ->
          {:ok, small} = Vapor.Compile.Lower.lower(program(64, unroll, mu_dt, vol_dt, log_b, rng), lower_opts)
          sub = fn %Tensor{} = x -> Tensor.from_list(:f32, [1, 64], Enum.take(Tensor.to_floats(x), 64)) end
          {:ok, ro} = Native.run_oracle(small, Map.new(run.first_env, fn {kk, v} -> {kk, sub.(v)} end))
          Enum.all?(want, fn w -> ro.outputs[w].data == binary_part(first[w].data, 0, 64 * 4) end)
      end

    # payoffs in binary64, fixed order
    pay = fn x -> if type == :call, do: max(x - k, 0.0), else: max(k - x, 0.0) end
    disc = :math.exp(-r * t)
    [l, a, g, alive] = Enum.map(want, &Tensor.to_floats(final[&1]))
    euro = Enum.map(l, &(disc * pay.(s0 * :math.exp(&1))))
    arith = Enum.map(a, &(disc * pay.(s0 * &1 / n)))
    geo = Enum.map(g, &(disc * pay.(s0 * :math.exp(&1 / n))))
    barrier = if log_b, do: Enum.zip_with(euro, alive, &(&1 * &2))

    closed_euro = Options.bsm(type, s0, k, t, r, q, sg)
    closed_geo = kemna_vorst(type, s0, k, t, r, q, sg, n)
    euro_s = summary(euro, closed_euro)
    geo_s = summary(geo, closed_geo)
    # arithmetic Asian with the geometric as control variate: Y − β(G − E[G])
    beta = cov(arith, geo) / max(Num.var(geo), 1.0e-300)
    cv = Enum.zip_with(arith, geo, fn y, gg -> y - beta * (gg - closed_geo) end)
    arith_s = summary(arith, nil)
    cv_s = summary(cv, nil)

    # binary64 reference: the same uniforms, Acklam's Φ⁻¹ (full precision), on the BEAM
    nref = min(o[:reference] || 512, b)
    {ref_us, ref_euro} = :timer.tc(fn -> reference(s0, k, type, disc, mu_dt, vol_dt, n, seed, nref, rng) end)
    agree = Enum.zip(Enum.take(euro, nref), ref_euro) |> Enum.map(fn {x, y} -> abs(x - y) end)
    threads_check =
      case o[:threads] do
        nt when is_integer(nt) and nt > 1 and run.substrate != "oracle" ->
          {:ok, w} = Vapor.Runtime.Worker.start_link(exec: [Substrates.binary("vapor-worker", "native")], threads: nt)
          try do
            other = simulate(comp, init, uniforms, calls, want, w, nt)
            %{threads: nt, identical_bits: Enum.all?(want, &(other.final[&1].data == final[&1].data))}
          after
            GenServer.stop(w)
          end
        _ -> nil
      end

    {:ok, %{type: type, paths: b, steps: n, unroll: unroll, calls: calls, substrate: run.substrate, seed: seed,
            european: euro_s, geometric_asian: geo_s, arithmetic_asian: arith_s, arithmetic_asian_cv: Map.put(cv_s, :variance_reduction, Num.var(arith) / max(Num.var(cv), 1.0e-300)),
            barrier: barrier && Map.merge(summary(barrier, nil), %{level: o[:barrier], knocked_out: Enum.count(alive, &(&1 == 0.0)) / b}),
            certificate: %{oracle_parity: parity, threads: threads_check,
                           binary64_agreement: %{paths: nref, max_abs: Enum.max(agree), mean_abs: Num.mean(agree)},
                           european_covered: euro_s.covers, geometric_covered: geo_s.covers},
            rng: rng, lowering_ms: lower_us / 1000, native_ms: us / 1000, beam_f64_ms_estimate: ref_us / nref * b / 1000, speedup: ref_us / nref * b / max(us, 1),
            paths_preview: preview(mu_dt, vol_dt, s0, n, seed, rng)}}
  end

  defp simulate(comp, init, uniforms, calls, want, worker, _threads) do
    worker = worker || Vapor.Modal.Runner.worker()
    case worker do
      nil ->
        # without a native worker the oracle carries the state itself — including the device
        # generator's (w1, w2, w3), which the worker's session keeps between calls
        rng_state = Map.has_key?(init, :w1)
        {final, _} = Enum.reduce(1..calls, {nil, init}, fn c, {_, cur} ->
          {:ok, r} = Native.run_oracle(comp, Map.merge(cur, uniforms.(c)))
          o = Map.take(r.outputs, want)
          next = %{l: o.l_next, a: o.a_next, g: o.g_next, alive: o.alive_next}
          next = if rng_state, do: Map.merge(next, %{w1: r.outputs.w1_next, w2: r.outputs.w2_next, w3: r.outputs.w3_next}), else: next
          {o, next}
        end)
        %{final: final, substrate: "oracle", first: nil, first_env: nil}
      w ->
        {:ok, sess} = Session.open(w, comp, isa: Substrates.host_isa())
        try do
          {final, first} =
            Enum.reduce(1..calls, {nil, nil}, fn c, {_, first} ->
              env = if c == 1, do: Map.merge(init, uniforms.(c)), else: uniforms.(c)
              {:ok, o, _} = Session.step(sess, env, want)
              {o, first || {o, env}}
            end)
          {o1, env1} = first
          %{final: final, substrate: Atom.to_string(Substrates.host_isa()), first: o1, first_env: env1}
        after
          Session.close(sess)
        end
    end
  end

  defp path_uniforms(seed, p, n, :host), do: for(step <- 0..(n - 1), do: u24(seed, step * @max_paths + p))
  defp path_uniforms(seed, p, n, :device), do: wh_uniforms(seed, p, n)

  defp reference(s0, k, type, disc, mu_dt, vol_dt, n, seed, nref, rng) do
    for p <- 0..(nref - 1) do
      l = path_uniforms(seed, p, n, rng) |> Enum.reduce(0.0, fn u, l -> l + mu_dt + vol_dt * Num.ninv(u) end)
      x = s0 * :math.exp(l)
      disc * if(type == :call, do: max(x - k, 0.0), else: max(k - x, 0.0))
    end
  end

  # a few paths for the picture (binary64, the same uniforms)
  defp preview(mu_dt, vol_dt, s0, n, seed, rng) do
    for p <- 0..11 do
      {pts, _} = path_uniforms(seed, p, n, rng) |> Enum.map_reduce(0.0, fn u, l -> l2 = l + mu_dt + vol_dt * ppnd7_f64(u); {s0 * :math.exp(l2), l2} end)
      [s0 | pts]
    end
  end

  @doc false
  def host_targets do
    case Substrates.host_isa() do
      :aarch64 -> [Vapor.Emit.ARM]
      :riscv64 -> [Vapor.Emit.RVV]
      _ -> [Vapor.Emit.X86]
    end
  end

  defp summary(xs, closed) do
    m = Num.mean(xs); se = Num.std(xs) / :math.sqrt(length(xs))
    lo = m - 2.5758293035489 * se; hi = m + 2.5758293035489 * se
    base = %{price: m, stderr: se, ci99: [lo, hi]}
    if closed, do: Map.merge(base, %{closed_form: closed, covers: lo <= closed and closed <= hi, z: (m - closed) / max(se, 1.0e-300)}), else: base
  end

  defp cov(xs, ys) do
    mx = Num.mean(xs); my = Num.mean(ys)
    Enum.zip(xs, ys) |> Enum.reduce(0.0, fn {x, y}, a -> a + (x - mx) * (y - my) end) |> Kernel./(length(xs) - 1)
  end

  @doc """
  Kemna–Vorst: the geometric average of S at the n monitoring dates
  t_i = iT/n (excluding t₀) is lognormal; its price in closed form.
  """
  def kemna_vorst(type, s0, k, t, r, q, sg, n) do
    dt = t / n
    # ln G = ln S0 + (1/n) Σ_i L(t_i);  mean and variance of (1/n) Σ_i W(t_i)
    mean = (r - q - 0.5 * sg * sg) * dt * (n + 1) / 2
    var = sg * sg * dt * (n + 1) * (2 * n + 1) / (6 * n)
    fwd = s0 * :math.exp(mean + var / 2)
    sd = :math.sqrt(var)
    d1 = (:math.log(fwd / k) + var / 2) / sd
    d2 = d1 - sd
    disc = :math.exp(-r * t)
    case type do
      :call -> disc * (fwd * Num.ncdf(d1) - k * Num.ncdf(d2))
      :put -> disc * (k * Num.ncdf(-d2) - fwd * Num.ncdf(-d1))
    end
  end

  # ------------------------------------------------- Longstaff–Schwartz (BEAM, binary64)

  @doc """
  American option by least-squares Monte Carlo (Longstaff & Schwartz
  2001): regression of the discounted continuation on {1, x, x²} over the
  in-the-money paths (x = S/K), exercise when the intrinsic beats it.
  Antithetic paths. Returns the price, its standard error and the
  European price from the same paths.
  """
  def lsm(type, s0, k, t, r, q, sg, opts \\ []) do
    paths = Keyword.get(opts, :paths, 20_000) |> min(200_000)
    n = Keyword.get(opts, :steps, 50)
    seed = Keyword.get(opts, :seed, 7)
    dt = t / n; mu = (r - q - 0.5 * sg * sg) * dt; vs = sg * :math.sqrt(dt); disc = :math.exp(-r * dt)
    half = div(paths, 2)
    z = for p <- 0..(half - 1), do: (for st <- 0..(n - 1), do: Num.ninv(u24(seed, st * 400_009 + p)))
    sims = Enum.flat_map(z, fn zs -> [zs, Enum.map(zs, &(-&1))] end)
    spaths = Enum.map(sims, fn zs -> {xs, _} = Enum.map_reduce(zs, s0, fn zz, s -> s2 = s * :math.exp(mu + vs * zz); {s2, s2} end); List.to_tuple(xs) end)
    pay = fn x -> if type == :call, do: max(x - k, 0.0), else: max(k - x, 0.0) end
    cash0 = Enum.map(spaths, &pay.(elem(&1, n - 1)))
    cash =
      Enum.reduce((n - 2)..0//-1, cash0, fn st, cash ->
        cash = Enum.map(cash, &(&1 * disc))
        itm = for {sp, i} <- Enum.with_index(spaths), pay.(elem(sp, st)) > 0, do: i
        if length(itm) < 3 do
          cash
        else
          ct = List.to_tuple(cash)
          spt = List.to_tuple(spaths)
          xs = Enum.map(itm, fn i -> x = elem(elem(spt, i), st) / k; [1.0, x, x * x] end)
          ys = Enum.map(itm, &elem(ct, &1))
          case Vapor.Finance.Num.normal_lstsq(xs, ys) do
            {:ok, beta} ->
              ex = Map.new(Enum.zip(itm, xs), fn {i, row} -> {i, Vapor.Dense.dot(row, beta)} end)
              Enum.with_index(cash) |> Enum.map(fn {c, i} ->
                case ex do
                  %{^i => cont} -> (iv = pay.(elem(elem(spt, i), st)); if iv > cont, do: iv, else: c)
                  _ -> c
                end
              end)
            _ -> cash
          end
        end
      end)
    vals = Enum.map(cash, &(&1 * disc))
    euro = Enum.map(cash0, &(&1 * :math.exp(-r * t)))
    # antithetic pairs averaged before the standard error
    pairs = vals |> Enum.chunk_every(2) |> Enum.map(&Num.mean/1)
    %{price: Num.mean(vals), stderr: Num.std(pairs) / :math.sqrt(length(pairs)), european: Num.mean(euro), paths: length(vals), steps: n}
  end
end
