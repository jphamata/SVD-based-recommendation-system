defmodule Vapor.Train.LM do
  @moduledoc """
  **Pre-training a language model from scratch, reproducibly to the last
  bit, on any number of workers.**

  The model is a Llama: byte embeddings, pre-norm decoder layers (RMSNorm,
  multi-head causal attention with rotary positions, SwiGLU MLP), a final
  norm and an output head — written with the canonical operators only, so
  its whole training step (forward, the cross-entropy gradient, the
  backward pass by `Vapor.Autodiff`, gradient clipping, AdamW) is a set of
  ordinary vapor programs: compiled to machine code, run in the native
  workers, the same bits on every substrate. A trained model is exported as
  a Hugging Face Llama checkpoint that `Vapor.Model`, the engine and
  `transformers` load.

  Lateral choices, each making the step expressible without a new operator:

  * **heads as separate maps** — `q_h = x·W_qhᵀ`, …, and the output
    projection as `Σ_h o_h·W_ohᵀ`: the same function as one wide matrix
    split in heads, without slicing columns (the export concatenates them);
  * **rotation by a signed permutation** — `rope(q) = q⊙cos + (q·Rᵀ)⊙sin`
    with `R` the constant `±1` matrix of "rotate half": a dot product with a
    single nonzero term is exact, so this *is* the rotation;
  * **sequences packed with a block-causal mask** — several sequences of a
    micro-batch are one `[R, R]` score matrix whose forbidden entries are
    `−10³⁰` (finite: `exp` gives an exact `+0`, no `inf` reaches a gradient);
  * **embedding as a one-hot contraction** — `x·Eᵀ` with `x` one-hot: the
    exact row, differentiable by the existing `linear` rule.

  **Data parallelism whose bits do not depend on the number of workers.**
  A step averages the gradients of `M` micro-batches (`M` a power of two),
  summed in a *fixed binary tree* over the micro-batch index. Addition is
  commutative bit for bit but not associative, so the tree shape — not who
  computes which node — determines the bits: one worker, two, eight, or a
  worker lost mid-step and its micro-batches recomputed elsewhere, all give
  the same parameters. Dividing by `M = 2ᵏ` is exact. (The usual ring
  all-reduce adds in an order that depends on the world size, which is why
  distributed runs are not reproducible across cluster shapes.)

  Micro-batch `i` of step `k` is drawn from the corpus by SplitMix64 on
  `(seed, k, i, j)` — independent of the workers — so a run is a pure
  function of the corpus, the configuration and the seed: `checkpoint/3`
  and `resume/2` continue it with the bits of an uninterrupted run.

  **Or no tree at all** (`reduce: :exact`, 0.15): every micro-batch's
  gradient enters a `Vapor.Amalgam` as it arrives — on whichever worker
  finished first, in whatever order — and the step's gradient is the
  **correctly rounded mean** of the exact sum. The bits are then a property
  of the set of micro-batches alone: `micro` need not be a power of two,
  workers are a work queue rather than a fixed assignment, and a micro-batch
  recomputed after a crash may arrive last. The price, stated: one gradient
  per micro-batch crosses back to the BEAM (no folding inside a session),
  and the exact sum costs a bignum add per parameter per micro-batch.
  """
  import Bitwise
  alias Vapor.{Amalgam, Autodiff, CR, F32, Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Runtime.{Native, Substrates}

  defstruct vocab: 256, d: 128, layers: 2, heads: 4, ff: 384, seq: 64, seqs: 2, eps: 1.0e-5, theta: 10_000.0

  @neg -1.0e30

  @doc "A configuration (all extents multiples of 16; `seqs·seq` rows per micro-batch)."
  def new(opts \\ []) do
    c = struct(__MODULE__, opts)
    dh = div(c.d, c.heads)

    for {n, v} <- [d: c.d, ff: c.ff, vocab: c.vocab, rows: c.seqs * c.seq, head_dim: dh],
        rem(v, 16) != 0,
        do: raise(ArgumentError, "#{n} = #{v} must be a multiple of 16")

    c
  end

  def head_dim(c), do: div(c.d, c.heads)
  def rows(c), do: c.seqs * c.seq

  @doc "Parameters, in a fixed order: `[{name, shape}]`."
  def shapes(%__MODULE__{} = c) do
    dh = head_dim(c)

    [{"emb", [c.vocab, c.d]}] ++
      Enum.flat_map(0..(c.layers - 1), fn l ->
        [{"l#{l}.ln1", [1, c.d]}] ++
          Enum.flat_map(0..(c.heads - 1), fn h ->
            [{"l#{l}.q#{h}", [dh, c.d]}, {"l#{l}.k#{h}", [dh, c.d]}, {"l#{l}.v#{h}", [dh, c.d]}, {"l#{l}.o#{h}", [c.d, dh]}]
          end) ++
          [{"l#{l}.ln2", [1, c.d]}, {"l#{l}.gate", [c.ff, c.d]}, {"l#{l}.up", [c.ff, c.d]}, {"l#{l}.down", [c.d, c.ff]}]
      end) ++ [{"norm", [1, c.d]}, {"head", [c.vocab, c.d]}]
  end

  def norm?(name), do: String.ends_with?(name, "ln1") or String.ends_with?(name, "ln2") or name == "norm"

  @doc "Initial parameters from `seed`: norms at 1; matrices uniform ±√(3/fan_in), residual outputs scaled by 1/√(2·layers)."
  def init(%__MODULE__{} = c, seed) do
    shapes(c)
    |> Enum.with_index()
    |> Map.new(fn {{name, [r, k] = shape}, i} ->
      cond do
        norm?(name) ->
          {name, Tensor.from_list(:f32, shape, List.duplicate(1.0, r * k))}

        true ->
          resid = String.ends_with?(name, "down") or String.match?(name, ~r/\.o\d+$/)
          scale = :math.sqrt(3.0 / k) * if(resid, do: 1 / :math.sqrt(2 * c.layers), else: 1.0)
          {name, Tensor.random(:f32, shape, seed * 1_000_003 + i, scale: scale)}
      end
    end)
  end

  # ----------------------------------------------------------- the model --

  defp p(name), do: {:p, name}

  defp param_inputs(c), do: Map.new(shapes(c), fn {n, s} -> {p(n), T.input(:"p.#{n}", :f32, s)} end)

  defp rmsnorm(x, g, c) do
    ms = T.mul(T.reduce(:sum, T.mul(x, x)), T.splat(1.0 / c.d))
    T.mul(g, T.mul(x, T.rsqrt(T.add(ms, T.splat(c.eps)))))
  end

  @doc false
  # the constants of a micro-batch: block-causal mask, rotary tables, the rotation
  def constants(%__MODULE__{} = c) do
    r = rows(c)
    dh = head_dim(c)
    half = div(dh, 2)
    mask = for i <- 0..(r - 1), j <- 0..(r - 1), do: if(div(i, c.seq) == div(j, c.seq) and j <= i, do: 0.0, else: @neg)
    # inv_freq_j = θ^(−2j/dh), angles in binary64 rounded once (correctly rounded cos/sin)
    ang = fn pos, j -> pos * :math.pow(c.theta, -2 * j / dh) end
    cs = for i <- 0..(r - 1), jj <- 0..(dh - 1), do: CR.cos_f64(ang.(rem(i, c.seq), rem(jj, half)))
    sn = for i <- 0..(r - 1), jj <- 0..(dh - 1), do: CR.sin_f64(ang.(rem(i, c.seq), rem(jj, half)))
    rot = for i <- 0..(dh - 1), j <- 0..(dh - 1), do: (cond do
            i < half and j == i + half -> -1.0
            i >= half and j == i - half -> 1.0
            true -> 0.0
          end)

    %{mask: Tensor.from_list(:f32, [r, r], mask), cos: Tensor.from_list(:f32, [r, dh], cs),
      sin: Tensor.from_list(:f32, [r, dh], sn), rot: Tensor.from_list(:f32, [dh, dh], rot)}
  end

  # name a value: a let-binding and its ref (terms stay small; see
  # Vapor.Autodiff.grad_lets/5 for why a deep network needs this)
  defp bind({lets, det}, name, term), do: {T.ref(name, term), {[{name, term} | lets], det}}

  # logits [R, V] as a ref, the bindings (reversed) and the nodes to detach
  defp forward(c, ps, x) do
    k = constants(c)
    {mask, cos, sin, rot} = {T.const(k.mask), T.const(k.cos), T.const(k.sin), T.const(k.rot)}
    rope = fn t -> T.add(T.mul(t, cos), T.mul(T.linear(t, rot), sin)) end
    scale = 1.0 / :math.sqrt(head_dim(c))
    {h, b} = bind({[], []}, :"f.embed", T.linear(x, T.transpose(ps[p("emb")])))

    {h, b} =
      Enum.reduce(0..(c.layers - 1), {h, b}, fn l, {h, b} ->
        nm = fn s -> :"f.l#{l}.#{s}" end
        {a, b} = bind(b, nm.(:attn_in), rmsnorm(h, ps[p("l#{l}.ln1")], c))

        {att, b} =
          Enum.reduce(0..(c.heads - 1), {nil, b}, fn hh, {acc, b} ->
            hn = fn s -> nm.("h#{hh}.#{s}") end
            {q0, b} = bind(b, hn.(:q0), T.linear(a, ps[p("l#{l}.q#{hh}")]))
            {q, b} = bind(b, hn.(:q), rope.(q0))
            {k0, b} = bind(b, hn.(:k0), T.linear(a, ps[p("l#{l}.k#{hh}")]))
            {kk, b} = bind(b, hn.(:k), rope.(k0))
            {v, b} = bind(b, hn.(:v), T.linear(a, ps[p("l#{l}.v#{hh}")]))
            {sc, b} = bind(b, hn.(:s), T.add(T.mul(T.linear(q, kk), T.splat(scale)), mask))
            m = T.reduce(:max, sc)
            {e, {lets, det}} = bind(b, hn.(:e), T.exp(T.sub(sc, m)))
            b = {lets, [m | det]}
            {pr, b} = bind(b, hn.(:p), T.mul(e, T.rcp(T.reduce(:sum, e))))
            {ctx, b} = bind(b, hn.(:ctx), T.linear(pr, T.transpose(v)))
            o = T.linear(ctx, ps[p("l#{l}.o#{hh}")])
            {o, b} = bind(b, hn.(:o), if(acc, do: T.add(acc, o), else: o))
            {o, b}
          end)

        {h, b} = bind(b, nm.(:resid1), T.add(h, att))
        {bn, b} = bind(b, nm.(:mlp_in), rmsnorm(h, ps[p("l#{l}.ln2")], c))
        {gt, b} = bind(b, nm.(:gate), T.linear(bn, ps[p("l#{l}.gate")]))
        {up, b} = bind(b, nm.(:up), T.linear(bn, ps[p("l#{l}.up")]))
        {act, b} = bind(b, nm.(:act), T.mul(T.silu(gt), up))
        bind(b, nm.(:resid2), T.add(h, T.linear(act, ps[p("l#{l}.down")])))
      end)

    {hf, b} = bind(b, :"f.final", rmsnorm(h, ps[p("norm")], c))
    {z, {lets, det}} = bind(b, :"f.logits", T.linear(hf, ps[p("head")]))
    {z, Enum.reverse(lets), det}
  end

  @doc """
  The gradient program of one micro-batch: inputs `p.<name>` (every
  parameter), `x`, `y` (one-hot inputs and next-byte targets, `f32[R, V]`);
  outputs `loss` (mean cross-entropy in nats, `f32[1,1]`) and `g.<name>`.
  """
  def grad_program(%__MODULE__{} = c, opts \\ []) do
    ps = param_inputs(c)
    r = rows(c)
    x = T.input(:x, :f32, [r, c.vocab])
    y = T.input(:y, :f32, [r, c.vocab])
    {z, lets, det} = forward(c, ps, x)

    mz = T.reduce(:max, z)
    ez = T.exp(T.sub(z, mz))
    lets = lets ++ [{:"l.ez", ez}]
    ez = T.ref(:"l.ez", ez)
    zs = T.reduce(:sum, ez)
    # ∂(mean CE)/∂z = (softmax z − y)/R, exact for this loss
    seed = T.mul(T.sub(T.mul(ez, T.rcp(zs)), y), T.splat(1.0 / r))
    lets = lets ++ [{:"l.seed", seed}]
    seed = T.ref(:"l.seed", seed)
    ls = T.sub(T.sub(z, mz), T.log(zs))
    loss = T.mul(T.reduce(:sum, T.transpose(T.reduce(:sum, T.mul(y, ls)))), T.splat(-1.0 / r))

    names = Enum.map(shapes(c), &elem(&1, 0))
    {:ok, grads, all} = Autodiff.grad_lets(lets, z, seed, Enum.map(names, &ps[p(&1)]), detach: det)

    if Keyword.get(opts, :accumulate, false) do
      # a resident accumulator per parameter: acc ← acc + g after every
      # micro-batch (the session feeds acc_next back), read once per chunk
      accs = for {n, s} <- shapes(c), do: {n, T.input(:"acc.#{n}", :f32, s)}
      outs = Enum.zip_with(accs, grads, fn {n, a}, g -> {:"acc_next.#{n}", T.add(a, g)} end)
      Program.new([loss: loss] ++ outs, lets: all, state: for({n, _} <- accs, do: {:"acc.#{n}", :"acc_next.#{n}"}))
    else
      Program.new([loss: loss] ++ Enum.zip_with(names, grads, fn n, g -> {:"g.#{n}", g} end), lets: all)
    end
  end

  @doc "The forward program for evaluation: `x` one-hot → `logits` (`f32[R, V]`)."
  def eval_program(%__MODULE__{} = c) do
    ps = param_inputs(c)
    {z, lets, _} = forward(c, ps, T.input(:x, :f32, [rows(c), c.vocab]))
    Program.new([logits: z], lets: lets)
  end

  @doc "One node of the reduction tree: `s.<name> = a.<name> + b.<name>` for every parameter."
  def add_program(%__MODULE__{} = c) do
    Program.new(for {n, s} <- shapes(c), do: {:"s.#{n}", T.add(T.input(:"a.#{n}", :f32, s), T.input(:"b.#{n}", :f32, s))})
  end

  @doc """
  The optimizer step: inputs `p.*`, `m.*`, `v.*` and the summed gradients
  `g.*`, scalars `gscale` (1/M, exact), `lr`, `c1`, `c2` (bias corrections)
  as `f32[1,1]`; outputs `p_next.*`, `m_next.*`, `v_next.*` and `gnorm`
  (the global norm before clipping). Options: `clip` (1.0), `beta1`
  (0.9), `beta2` (0.95), `eps` (1e-8), `weight_decay` (0.1, matrices only).
  """
  def opt_program(%__MODULE__{} = c, opts \\ []) do
    {b1, b2} = {Keyword.get(opts, :beta1, 0.9), Keyword.get(opts, :beta2, 0.95)}
    {eps, wd, clip} = {Keyword.get(opts, :eps, 1.0e-8), Keyword.get(opts, :weight_decay, 0.1), Keyword.get(opts, :clip, 1.0)}
    sc = fn n -> T.input(n, :f32, [1, 1]) end
    {gscale, lr, c1, c2} = {sc.(:gscale), sc.(:lr), sc.(:c1), sc.(:c2)}

    gs = Map.new(shapes(c), fn {n, s} -> {n, T.mul(T.input(:"g.#{n}", :f32, s), gscale)} end)

    # global norm: Σ over parameters of Σ g², every sum canonical
    n2 =
      shapes(c)
      |> Enum.map(fn {n, [r, _]} ->
        sq = T.reduce(:sum, T.mul(gs[n], gs[n]))
        if r == 1, do: sq, else: T.reduce(:sum, T.transpose(sq))
      end)
      |> Enum.reduce(&T.add(&2, &1))

    factor = T.min(T.splat(1.0), T.mul(T.rsqrt(n2), T.splat(clip)))

    updates =
      Enum.flat_map(shapes(c), fn {n, s} ->
        pv = T.input(:"p.#{n}", :f32, s)
        m = T.input(:"m.#{n}", :f32, s)
        v = T.input(:"v.#{n}", :f32, s)
        g = T.mul(gs[n], factor)
        m2 = T.add(T.mul(m, T.splat(b1)), T.mul(g, T.splat(1 - b1)))
        v2 = T.add(T.mul(v, T.splat(b2)), T.mul(T.mul(g, g), T.splat(1 - b2)))
        step = T.mul(T.mul(m2, c1), T.rsqrt(T.add(T.mul(v2, c2), T.splat(eps * eps))))
        step = if wd > 0 and not norm?(n), do: T.add(step, T.mul(pv, T.splat(wd))), else: step
        [{:"p_next.#{n}", T.sub(pv, T.mul(step, lr))}, {:"m_next.#{n}", m2}, {:"v_next.#{n}", v2}]
      end)

    Program.new([gnorm: T.mul(n2, T.splat(1.0))] ++ updates)
  end

  # --------------------------------------------------------------- data --

  @doc """
  The micro-batch `i` of step `k`: `seqs` windows of `seq + 1` bytes at
  offsets drawn by SplitMix64 from `(seed, k, i, j)` — the same whoever
  computes it. Returns `%{x, y}` one-hot `f32[R, V]` and the offsets.
  """
  def batch(%__MODULE__{} = c, corpus, seed, k, i) do
    n = byte_size(corpus) - c.seq - 1
    offs = for j <- 0..(c.seqs - 1), do: rem(mix(mix(mix(seed, k), i), j) >>> 11, n)
    windows = Enum.map(offs, &binary_part(corpus, &1, c.seq + 1))
    xs = Enum.flat_map(windows, &(:binary.bin_to_list(&1) |> Enum.take(c.seq)))
    ys = Enum.flat_map(windows, &(:binary.bin_to_list(&1) |> Enum.drop(1)))
    %{x: onehot(xs, c.vocab), y: onehot(ys, c.vocab), offsets: offs}
  end

  defp mix(a, b), do: elem(Tensor.splitmix(bxor(a * 0x9E37_79B9_7F4A_7C15 &&& 0xFFFF_FFFF_FFFF_FFFF, b + 1)), 0)

  @one <<0x3F80_0000::32-little>>
  @doc false
  def onehot(ids, v) do
    data = for b <- ids, into: <<>>, do: <<0::size(b * 32), @one::binary, 0::size((v - b - 1) * 32)>>
    Tensor.new(:f32, [length(ids), v], data)
  end

  # ------------------------------------------------------------ training --

  defmodule Run do
    @moduledoc "A training run: configuration, compiled programs, state, schedule."
    defstruct [:cfg, :grad, :acc, :add, :opt, :params, :m, :v, :step, :seed, :micro, :chunk, :steps, :lr, :warmup, :lr_end,
               :beta1, :beta2, :corpus_digest, reduce: :tree, sessions: %{}, losses: [], gnorms: []]
  end

  @doc """
  A run ready to train: compiles the gradient, reduction and optimizer
  programs once. Options: `seed` (1), `micro` (micro-batches per step, a
  power of two; 8), `chunk` (micro-batches folded in a worker before the
  tree, a power of two dividing `micro`; min(4, micro)), `steps` (total, for the
  schedule; 1000), `lr` (3e-3), `warmup` (steps; 50), `lr_end` (fraction at
  the end of the cosine; 0.1), and the optimizer options of `opt_program/2`.

  The step's gradient is defined as a balanced binary tree over the
  `micro/chunk` chunks, each chunk the left fold `((0 + g₀) + g₁) + …` of
  its micro-batches. Both are fixed by the run, not by the workers: any
  number of workers up to `micro/chunk` gives the same bits. A chunk is
  folded *inside* one worker, in a resident session whose accumulator
  never leaves it until the chunk ends (the parameters are written once
  per step) — what turns the per-micro-batch traffic into two one-hot
  matrices in and one number out.

  With `reduce: :exact` there is no tree and no chunk: `micro` is any
  positive integer and the gradient is the correctly rounded mean of the
  exact sum (`Vapor.Amalgam`), whatever the arrival order.
  """
  def start(%__MODULE__{} = c, corpus, opts \\ []) do
    reduce = Keyword.get(opts, :reduce, :tree)
    micro = Keyword.get(opts, :micro, 8)
    chunk = if reduce == :exact, do: 1, else: Keyword.get(opts, :chunk, min(4, micro))
    if reduce not in [:tree, :exact], do: raise(ArgumentError, "reduce must be :tree or :exact")
    if not (is_integer(micro) and micro >= 1), do: raise(ArgumentError, "micro-batches per step must be a positive integer")
    if reduce == :tree and (micro &&& (micro - 1)) != 0, do: raise(ArgumentError, "micro-batches per step must be a power of two (or reduce: :exact)")
    if chunk < 1 or (chunk &&& (chunk - 1)) != 0 or rem(micro, chunk) != 0, do: raise(ArgumentError, "chunk must be a power of two dividing micro")
    seed = Keyword.get(opts, :seed, 1)
    {:ok, grad} = Vapor.Compile.Lower.lower(grad_program(c))
    {:ok, acc} = Vapor.Compile.Lower.lower(grad_program(c, accumulate: true))
    {:ok, add} = Vapor.Compile.Lower.lower(add_program(c))
    {:ok, opt} = Vapor.Compile.Lower.lower(opt_program(c, opts))
    params = init(c, seed)
    zeros = Map.new(params, fn {n, t} -> {n, Tensor.new(:f32, t.shape, :binary.copy(<<0::32>>, Enum.product(t.shape)))} end)

    %Run{cfg: c, grad: grad, acc: acc, add: add, opt: opt, params: params, m: zeros, v: zeros, step: 0, seed: seed, micro: micro, chunk: chunk, reduce: reduce,
         steps: Keyword.get(opts, :steps, 1000), lr: Keyword.get(opts, :lr, 3.0e-3), warmup: Keyword.get(opts, :warmup, 50),
         lr_end: Keyword.get(opts, :lr_end, 0.1), beta1: Keyword.get(opts, :beta1, 0.9), beta2: Keyword.get(opts, :beta2, 0.95),
         corpus_digest: Base.encode16(:crypto.hash(:sha256, corpus), case: :lower)}
  end

  @doc "Learning rate of step `k` (1-based): linear warm-up, then cosine to `lr·lr_end`."
  def lr_at(%Run{} = r, k) do
    if k <= r.warmup do
      r.lr * k / max(r.warmup, 1)
    else
      t = (k - r.warmup) / max(r.steps - r.warmup, 1)
      r.lr * (r.lr_end + (1 - r.lr_end) * 0.5 * (1 + CR.cos_f64(:math.pi() * min(t, 1.0))))
    end
  end

  @doc """
  Run `n` steps on `workers` (a list of native `Vapor.Runtime.Worker`s, or
  `[]` for the exact oracle). Option `assign` (a function `(step, i) →
  worker index`, default round robin) exists to show that the assignment
  does not matter; `log` receives `%{step, loss, gnorm, tokens_per_s}`.
  """
  def train(%Run{} = r, corpus, workers, n, opts \\ []) do
    log = Keyword.get(opts, :log, fn _ -> :ok end)

    Enum.reduce(1..n//1, r, fn _, r ->
      t0 = System.monotonic_time(:microsecond)
      r = step(r, corpus, workers, opts)
      dt = (System.monotonic_time(:microsecond) - t0) / 1.0e6
      {_, loss} = hd(r.losses)
      {_, gn} = hd(r.gnorms)
      log.(%{step: r.step, loss: loss, gnorm: gn, seconds: dt, tokens_per_s: r.micro * rows(r.cfg) / dt})
      r
    end)
  end

  @doc "One step: the chunk sums (folded in the workers), their fixed-tree sum, the optimizer."
  def step(run, corpus, workers, opts \\ [])
  def step(%Run{reduce: :exact} = r, corpus, workers, opts), do: exact_step(r, corpus, workers, opts)

  def step(%Run{cfg: c} = r, corpus, workers, opts) do
    k = r.step + 1
    nchunks = div(r.micro, r.chunk)
    # chunk j on worker assign(k, j) mod W; a worker runs its chunks in order
    assign = Keyword.get(opts, :assign, fn _k, j -> j end)
    by_worker = Enum.group_by(0..(nchunks - 1), &(if workers == [], do: 0, else: rem(assign.(k, &1), length(workers))))

    {results, sessions} =
      by_worker
      |> Task.async_stream(fn {wi, js} ->
        w = Enum.at(workers, wi)
        {res, sess} = Enum.map_reduce(js, Map.get(r.sessions, w), fn j, sess -> chunk(r, corpus, k, j, w, sess, workers) end)
        {w, sess, res}
      end, timeout: :infinity, max_concurrency: max(length(workers), 1))
      |> Enum.reduce({[], r.sessions}, fn {:ok, {w, sess, res}}, {acc, ss} -> {res ++ acc, if(w, do: Map.put(ss, w, sess), else: ss)} end)

    results = Enum.sort_by(results, &elem(&1, 0))
    loss = results |> Enum.flat_map(fn {_, _, ls} -> ls end) |> Enum.sum() |> Kernel./(r.micro)
    grads = results |> Enum.map(fn {_, g, _} -> g end) |> tree(workers, r.add, c)

    optimize(r, k, grads, 1.0 / r.micro, loss, sessions, workers)
  end

  # the exact step: micro-batches as a work queue, gradients amalgamated in
  # arrival order, the mean rounded once (gscale = 1, exact)
  defp exact_step(%Run{cfg: c} = r, corpus, workers, opts) do
    k = r.step + 1
    assign = Keyword.get(opts, :assign, fn _k, i -> i end)
    pin = Map.new(r.params, fn {n, t} -> {:"p.#{n}", t} end)
    empty = Map.new(shapes(c), fn {n, s} -> {n, Amalgam.new(:f32, Enum.product(s))} end)

    {grads, losses} =
      0..(r.micro - 1)
      |> Task.async_stream(fn i ->
        b = batch(c, corpus, r.seed, k, i)
        out = run_on(workers, assign.(k, i), r.grad, Map.merge(pin, %{x: b.x, y: b.y}))
        {Map.new(shapes(c), fn {n, _} -> {n, Amalgam.add(empty[n], out[:"g.#{n}"])} end), Amalgam.add(Amalgam.new(:f32, 1), out.loss)}
      end, ordered: false, timeout: :infinity, max_concurrency: max(length(workers), 1))
      |> Enum.reduce({empty, Amalgam.new(:f32, 1)}, fn {:ok, {g, l}}, {acc, la} ->
        {Map.new(acc, fn {n, a} -> {n, Amalgam.merge(a, g[n])} end), Amalgam.merge(la, l)}
      end)

    grads = Map.new(shapes(c), fn {n, s} -> {n, Amalgam.mean(grads[n], shape: s)} end)
    loss = losses |> Amalgam.mean() |> Tensor.to_floats() |> hd()
    optimize(r, k, grads, 1.0, loss, r.sessions, workers)
  end

  defp optimize(%Run{cfg: c} = r, k, grads, gscale, loss, sessions, workers) do
    {b1, b2} = {r.beta1, r.beta2}
    one = fn x -> Tensor.from_list(:f32, [1, 1], [x]) end

    env =
      Map.merge(
        %{gscale: one.(gscale), lr: one.(lr_at(r, k)), c1: one.(1 / (1 - CR.pow_f64(b1, k * 1.0))), c2: one.(1 / (1 - CR.pow_f64(b2, k * 1.0)))},
        Enum.reduce(shapes(c), %{}, fn {n, _}, acc ->
          Map.merge(acc, %{:"p.#{n}" => r.params[n], :"m.#{n}" => r.m[n], :"v.#{n}" => r.v[n], :"g.#{n}" => grads[n]})
        end))

    out = run_on(workers, k, r.opt, env)
    pick = fn pre -> Map.new(shapes(c), fn {n, _} -> {n, out[:"#{pre}.#{n}"]} end) end

    %{r | params: pick.("p_next"), m: pick.("m_next"), v: pick.("v_next"), step: k, sessions: sessions,
          losses: [{k, loss} | r.losses], gnorms: [{k, :math.sqrt(out.gnorm |> Tensor.to_floats() |> hd())} | r.gnorms]}
  end

  # one chunk: the fold ((0 + g₀) + g₁) + … of its micro-batches. On a live
  # worker, inside a resident session (parameters written once per step);
  # otherwise — no worker, or a worker lost — recomputed with the same
  # additions in the same order. Returns {{j, sum, losses}, session}.
  defp chunk(r, corpus, k, j, w, sess, workers) do
    idx = (j * r.chunk)..(j * r.chunk + r.chunk - 1)

    case w && in_session(r, corpus, k, idx, w, sess) do
      {:ok, sum, losses, sess} ->
        {{j, sum, losses}, sess}

      _ ->
        zero = Map.new(shapes(r.cfg), fn {n, s} -> {n, Tensor.new(:f32, s, :binary.copy(<<0::32>>, Enum.product(s)))} end)
        pin = Map.new(r.params, fn {n, t} -> {:"p.#{n}", t} end)

        {sum, losses} =
          Enum.reduce(idx, {zero, []}, fn i, {acc, ls} ->
            b = batch(r.cfg, corpus, r.seed, k, i)
            out = run_on(workers, i, r.grad, Map.merge(pin, %{x: b.x, y: b.y}))
            env = Enum.reduce(shapes(r.cfg), %{}, fn {n, _}, e -> e |> Map.put(:"a.#{n}", acc[n]) |> Map.put(:"b.#{n}", out[:"g.#{n}"]) end)
            s = run_on(workers, i, r.add, env)
            {Map.new(shapes(r.cfg), fn {n, _} -> {n, s[:"s.#{n}"]} end), [out.loss |> Tensor.to_floats() |> hd() | ls]}
          end)

        {{j, sum, Enum.reverse(losses)}, nil}
    end
  end

  # a worker may live on another node (`Vapor.Cluster.workers/1`)
  defp alive?(w) when node(w) == node(), do: Process.alive?(w)

  defp alive?(w) do
    :erpc.call(node(w), Process, :alive?, [w], 5_000)
  catch
    _, _ -> false
  end

  defp in_session(r, corpus, k, idx, w, sess) do
    alias Vapor.Runtime.Session
    c = r.cfg

    with true <- alive?(w),
         {:ok, s} <- (if sess, do: {:ok, sess}, else: Session.open(w, r.acc, isa: Substrates.host_isa())) do
      zero = Map.new(shapes(c), fn {n, sh} -> {:"acc.#{n}", Tensor.new(:f32, sh, :binary.copy(<<0::32>>, Enum.product(sh)))} end)
      params = Map.new(r.params, fn {n, t} -> {:"p.#{n}", t} end)
      last = Enum.max(idx)

      Enum.reduce_while(idx, {:ok, [], nil}, fn i, {:ok, ls, _} ->
        b = batch(c, corpus, r.seed, k, i)
        # the first micro-batch of a chunk resets the accumulator and writes the step's parameters
        ins = if i == Enum.min(idx), do: Map.merge(Map.merge(zero, params), %{x: b.x, y: b.y}), else: %{x: b.x, y: b.y}
        want = if i == last, do: [:loss | for({n, _} <- shapes(c), do: :"acc_next.#{n}")], else: [:loss]

        case Session.step(s, ins, want) do
          {:ok, out, _} -> {:cont, {:ok, [out.loss |> Tensor.to_floats() |> hd() | ls], out}}
          err -> {:halt, err}
        end
      end)
      |> case do
        {:ok, ls, out} -> {:ok, Map.new(shapes(c), fn {n, _} -> {n, out[:"acc_next.#{n}"]} end), Enum.reverse(ls), s}
        err -> err
      end
    else
      _ -> :error
    end
  catch
    :exit, _ -> :error
  end

  # the fixed binary tree over micro-batch indices: level by level, pairs
  # (2j, 2j+1); its shape alone determines the bits
  defp tree([g], _workers, _add, _c), do: g

  defp tree(level, workers, add, c) do
    level
    |> Enum.chunk_every(2)
    |> Enum.with_index()
    |> Task.async_stream(fn {[a, b], j} ->
      env = Enum.reduce(shapes(c), %{}, fn {n, _}, acc -> acc |> Map.put(:"a.#{n}", a[n]) |> Map.put(:"b.#{n}", b[n]) end)
      out = run_on(workers, j, add, env)
      Map.new(shapes(c), fn {n, _} -> {n, out[:"s.#{n}"]} end)
    end, timeout: :infinity, max_concurrency: max(length(workers), 1))
    |> Enum.map(fn {:ok, g} -> g end)
    |> tree(workers, add, c)
  end

  # on worker `i mod W` — and if it fails, on the next ones, then the oracle:
  # every substrate gives the same bits, so a retry changes nothing
  defp run_on([], _i, comp, env), do: oracle(comp, env)

  defp run_on(workers, i, comp, env) do
    n = length(workers)

    Enum.find_value(0..(n - 1), fn d ->
      w = Enum.at(workers, rem(i + d, n))

      case safe_run(w, comp, env) do
        {:ok, res} -> res.outputs
        _ -> nil
      end
    end) || oracle(comp, env)
  end

  defp safe_run(w, comp, env) do
    if alive?(w), do: Native.run(w, comp, env, isa: Substrates.host_isa(), mode: :native), else: {:error, :dead}
  catch
    :exit, r -> {:error, r}
  end

  defp oracle(comp, env) do
    {:ok, res} = Native.run_oracle(comp, env)
    res.outputs
  end

  # ------------------------------------------------------------ evaluation --

  @doc """
  Bits per byte of `text` under the run's parameters: non-overlapping
  windows of `seq` bytes, each predicted from its own prefix (the first
  byte of a window from nothing, as in training). Returns `{bits_per_byte, bytes}`.
  """
  def bits_per_byte(%Run{cfg: c, params: ps}, text, workers \\ []) do
    {:ok, comp} = compiled_eval(c)
    pin = Map.new(ps, fn {n, t} -> {:"p.#{n}", t} end)
    bytes = :binary.bin_to_list(text)
    # windows of seq+1 bytes stepping by seq: every byte after the first is predicted once
    wins = for st <- 0..(length(bytes) - c.seq - 1)//c.seq, do: Enum.slice(bytes, st, c.seq + 1)
    groups = Enum.chunk_every(wins, c.seqs)

    {nats, count} =
      groups
      |> Enum.with_index()
      |> Task.async_stream(fn {g, i} ->
        padded = g ++ List.duplicate(List.duplicate(32, c.seq + 1), c.seqs - length(g))
        x = onehot(Enum.flat_map(padded, &Enum.take(&1, c.seq)), c.vocab)
        out = run_on(workers, i, comp, Map.put(pin, :x, x))
        logits = out.logits |> Tensor.to_floats() |> Enum.chunk_every(c.vocab)
        targets = Enum.flat_map(g, &Enum.drop(&1, 1))
        rows = Enum.take(logits, length(targets))
        nats = Enum.zip_with(rows, targets, fn row, t -> -Enum.at(Vapor.Quality.Text.log_softmax(row), t) end) |> Enum.sum()
        {nats, length(targets)}
      end, timeout: :infinity, max_concurrency: max(length(workers), 1))
      |> Enum.reduce({0.0, 0}, fn {:ok, {a, b}}, {x, y} -> {x + a, y + b} end)

    {nats / count / :math.log(2), count}
  end

  defp compiled_eval(c) do
    key = {__MODULE__, :eval, c}

    case :persistent_term.get(key, nil) do
      nil -> with {:ok, comp} <- Vapor.Compile.Lower.lower(eval_program(c)), do: (:persistent_term.put(key, comp); {:ok, comp})
      comp -> {:ok, comp}
    end
  end

  @doc "Greedy (or sampled, with `temperature`) continuation of `prompt` (bytes) for `n` bytes."
  def generate(%Run{cfg: c, params: ps}, prompt, n, workers \\ []) do
    {:ok, comp} = compiled_eval(c)
    pin = Map.new(ps, fn {k, t} -> {:"p.#{k}", t} end)

    Enum.reduce(1..n//1, :binary.bin_to_list(prompt), fn _, acc ->
      ctx = Enum.take(acc, -c.seq)
      x = onehot(ctx ++ List.duplicate(32, c.seqs * c.seq - length(ctx)), c.vocab)
      out = run_on(workers, 0, comp, Map.put(pin, :x, x))
      row = out.logits |> Tensor.to_floats() |> Enum.chunk_every(c.vocab) |> Enum.at(length(ctx) - 1)
      {_, best} = row |> Enum.with_index() |> Enum.max_by(&elem(&1, 0))
      acc ++ [best]
    end)
    |> :binary.list_to_bin()
  end

  @doc """
  Baselines on the same held-out bytes, from the same training bytes: the
  byte frequencies (`unigram`) and interpolated Witten–Bell n-grams of
  orders 3 and 5 (`Vapor.Vision.CharLM`, over the 256 byte values).
  """
  def baselines(corpus, holdout) do
    as_syms = fn bin -> bin |> :binary.bin_to_list() |> Enum.map(&<<&1::utf8>>) |> Enum.join() end
    alpha = Enum.map(0..255, &<<&1::utf8>>)
    {c, h} = {as_syms.(corpus), as_syms.(holdout)}
    wb = fn o -> Vapor.Vision.CharLM.bits_per_char(Vapor.Vision.CharLM.build(c, alpha, o), h) end
    %{unigram: Vapor.Quality.Text.unigram_bits(holdout, Vapor.Quality.Text.profile(corpus)), witten_bell_3: wb.(3), witten_bell_5: wb.(5)}
  end

  # ----------------------------------------------------------- checkpoints --

  @doc """
  Save the run (parameters and moments, with the step, seed, schedule and
  the corpus digest in the safetensors metadata). `resume/2` continues it
  with the bits of an uninterrupted run.
  """
  def checkpoint(%Run{} = r, path) do
    tensors =
      Enum.flat_map(shapes(r.cfg), fn {n, _} -> [{"p.#{n}", r.params[n]}, {"m.#{n}", r.m[n]}, {"v.#{n}", r.v[n]}] end)
      |> Map.new()

    meta = %{"vapor.lm" => Vapor.JSON.encode(%{step: r.step, seed: r.seed, micro: r.micro, steps: r.steps, lr: r.lr, warmup: r.warmup,
                                                lr_end: r.lr_end, beta1: r.beta1, beta2: r.beta2, corpus: r.corpus_digest, reduce: r.reduce,
                                                config: Map.from_struct(r.cfg)})}
    Vapor.Ingest.Safetensors.write(path, tensors, meta)
  end

  @doc "Restore a run saved by `checkpoint/2` (the compiled programs come from `start/3` on the same configuration)."
  def resume(%Run{} = fresh, path) do
    {:ok, ts} = Vapor.Ingest.Safetensors.read(path)
    {:ok, %{metadata: md}} = Vapor.Ingest.Safetensors.index(path)
    {:ok, m} = Vapor.JSON.decode(md["vapor.lm"])
    pick = fn pre -> Map.new(shapes(fresh.cfg), fn {n, _} -> {n, Map.fetch!(ts, "#{pre}.#{n}")} end) end
    %{fresh | params: pick.("p"), m: pick.("m"), v: pick.("v"), step: m["step"], seed: m["seed"]}
  end

  @doc "SHA-256 of the parameters (what reproducibility claims are about)."
  def digest(%Run{cfg: c, params: ps}),
    do: Base.encode16(:crypto.hash(:sha256, for({n, _} <- shapes(c), do: [n, ps[n].data])), case: :lower)

  # -------------------------------------------------------------- export --

  @doc """
  Write the trained model as a Hugging Face Llama checkpoint (`config.json`,
  `model.safetensors`; the per-head maps concatenated into `q_proj`, …,
  `o_proj`), loadable by `Vapor.Model`, the engine and `transformers`.
  """
  def export(%Run{cfg: c, params: ps}, dir) do
    File.mkdir_p!(dir)
    dh = head_dim(c)
    cat_rows = fn names -> Tensor.new(:f32, [length(names) * dh, c.d], IO.iodata_to_binary(Enum.map(names, &ps[&1].data))) end
    # o_proj = [o_0 | o_1 | …] side by side: row r is the concatenation of each head's row r
    cat_cols = fn names ->
      rows = for rr <- 0..(c.d - 1), n <- names, do: binary_part(ps[n].data, rr * dh * 4, dh * 4)
      Tensor.new(:f32, [c.d, length(names) * dh], IO.iodata_to_binary(rows))
    end
    vec = fn n -> Tensor.new(:f32, [c.d], ps[n].data) end

    layers =
      Enum.flat_map(0..(c.layers - 1), fn l ->
        hs = 0..(c.heads - 1)
        pre = "model.layers.#{l}."
        [{pre <> "input_layernorm.weight", vec.("l#{l}.ln1")},
         {pre <> "self_attn.q_proj.weight", cat_rows.(for h <- hs, do: "l#{l}.q#{h}")},
         {pre <> "self_attn.k_proj.weight", cat_rows.(for h <- hs, do: "l#{l}.k#{h}")},
         {pre <> "self_attn.v_proj.weight", cat_rows.(for h <- hs, do: "l#{l}.v#{h}")},
         {pre <> "self_attn.o_proj.weight", cat_cols.(for h <- hs, do: "l#{l}.o#{h}")},
         {pre <> "post_attention_layernorm.weight", vec.("l#{l}.ln2")},
         {pre <> "mlp.gate_proj.weight", ps["l#{l}.gate"]},
         {pre <> "mlp.up_proj.weight", ps["l#{l}.up"]},
         {pre <> "mlp.down_proj.weight", ps["l#{l}.down"]}]
      end)

    tensors = Map.new([{"model.embed_tokens.weight", ps["emb"]}, {"model.norm.weight", vec.("norm")}, {"lm_head.weight", ps["head"]}] ++ layers)

    config = %{
      "architectures" => ["LlamaForCausalLM"], "model_type" => "llama", "vocab_size" => c.vocab, "hidden_size" => c.d,
      "intermediate_size" => c.ff, "num_hidden_layers" => c.layers, "num_attention_heads" => c.heads,
      "num_key_value_heads" => c.heads, "max_position_embeddings" => c.seq, "rms_norm_eps" => c.eps,
      "rope_theta" => c.theta, "hidden_act" => "silu", "tie_word_embeddings" => false, "attention_bias" => false,
      "torch_dtype" => "float32", "bos_token_id" => nil, "eos_token_id" => nil
    }

    File.write!(Path.join(dir, "config.json"), Vapor.JSON.encode(config))
    :ok = Vapor.Ingest.Safetensors.write(Path.join(dir, "model.safetensors"), tensors)
    {:ok, dir}
  end

  @doc false
  def f32_bits(x), do: F32.from_float(x)
end
