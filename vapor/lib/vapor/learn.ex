defmodule Vapor.Learn do
  @moduledoc """
  **Small networks trained by vapor itself, reproducibly to the last bit.**

  A multilayer perceptron (`x → linear → silu → … → linear`) and its whole
  training step — forward pass, mean-squared error, the gradient by
  `Vapor.Autodiff`, an AdamW update — are one recurrent program
  (`Vapor.Program` with the parameters and moments as state, the batches
  as sequence inputs). A run of `k` steps is one crossing into the native
  worker; like every program it is bit-identical on every substrate and
  for any thread count, so a trained model is a reproducible artefact: the
  data, the seed and the schedule determine its weights exactly
  (`receipt/1` records them).

  Used by the consistent upscaler (`Vapor.Vision.Upscale`), the policy
  gradient and behaviour-cloning learners (`Vapor.RL`). Extents are padded
  to the 16-lane contraction: padded inputs are zero columns, padded
  outputs are masked out of the loss (their weights start at zero and get
  no gradient, so they stay zero).
  """
  alias Vapor.{Autodiff, CR, Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Runtime.{Native, Substrates}

  defstruct [:sizes, :padded, :params, :act]

  @doc """
  A network with layer sizes `[d_in, h1, …, d_out]`, initialized from
  `seed` (uniform ±√(6/fan_in), zero biases). Option `act` (`:silu`, `:relu`).
  """
  def new(sizes, seed, opts \\ []) do
    padded = Enum.map(sizes, &Vapor.Spatial.pad16/1)
    layers = Enum.zip([Enum.drop(sizes, -1), tl(sizes), Enum.drop(padded, -1), tl(padded)])

    params =
      layers
      |> Enum.with_index()
      |> Enum.flat_map(fn {{fi, fo, pi, po}, l} ->
        a = :math.sqrt(6.0 / fi)
        w = Tensor.random(:f32, [fo, fi], seed * 1000 + l, scale: a)
        [{:"w#{l}", pad2(w, po, pi)}, {:"b#{l}", Tensor.new(:f32, [1, po], :binary.copy(<<0::32>>, po))}]
      end)

    %__MODULE__{sizes: sizes, padded: padded, params: params, act: Keyword.get(opts, :act, :silu)}
  end

  defp pad2(%Tensor{shape: [r, c]} = t, rp, cp) do
    rows = for i <- 0..(r - 1), into: <<>>, do: binary_part(t.data, i * c * 4, c * 4) <> :binary.copy(<<0::32>>, cp - c)
    Tensor.new(:f32, [rp, cp], rows <> :binary.copy(<<0::32>>, (rp - r) * cp))
  end

  defp forward(%__MODULE__{} = net, x, p) do
    n = length(net.sizes) - 1

    Enum.reduce(0..(n - 1), x, fn l, h ->
      z = T.add(T.linear(h, p[:"w#{l}"]), p[:"b#{l}"])
      cond do
        l == n - 1 -> z
        net.act == :relu -> T.relu(z)
        true -> T.silu(z)
      end
    end)
  end

  defp param_inputs(net), do: Map.new(net.params, fn {k, t} -> {k, T.input(k, :f32, t.shape)} end)

  @doc "The inference program: `x : f32[rows, d_in_padded] → y : f32[rows, d_out_padded]`."
  def predict_program(%__MODULE__{} = net, rows) do
    p = Map.new(net.params, fn {k, t} -> {k, T.const(t)} end)
    Program.new(y: forward(net, T.input(:x, :f32, [rows, hd(net.padded)]), p))
  end

  @doc """
  The training-step program for batches of `batch` rows (a multiple of 16).
  Options: `beta1` (0.9), `beta2` (0.999), `eps` (1e-8), `weight_decay` (0);
  the learning rate arrives per step (`schedule/3`: `lr`, `lr_end`).
  """
  def train_program(%__MODULE__{} = net, batch, opts \\ []) do
    {b1, b2} = {Keyword.get(opts, :beta1, 0.9), Keyword.get(opts, :beta2, 0.999)}
    {eps, wd} = {Keyword.get(opts, :eps, 1.0e-8), Keyword.get(opts, :weight_decay, 0.0)}
    {din, dout, dreal} = {hd(net.padded), List.last(net.padded), List.last(net.sizes)}
    x = T.input(:x, :f32, [batch, din])
    y = T.input(:y, :f32, [batch, dout])
    c1 = T.input(:c1, :f32, [1, 1])
    c2 = T.input(:c2, :f32, [1, 1])
    # the learning rate is a per-step input (a schedule), the same for every parameter
    lr_in = T.input(:lr, :f32, [1, 1])
    p = param_inputs(net)
    pred = forward(net, x, p)
    mask = T.const(Tensor.from_list(:f32, [1, dout], for(i <- 0..(dout - 1), do: if(i < dreal, do: 1.0, else: 0.0))))
    diff = T.mul(T.sub(pred, y), mask)
    k = 1.0 / (batch * dreal)
    loss = T.mul(T.reduce(:sum, T.transpose(T.reduce(:sum, T.mul(diff, diff)))), T.splat(k))
    seed = T.mul(diff, T.splat(2.0 * k))
    keys = Enum.map(net.params, &elem(&1, 0))
    {:ok, grads} = Autodiff.grad(pred, seed, Enum.map(keys, &p[&1]))

    updates =
      Enum.zip(keys, grads)
      |> Enum.flat_map(fn {name, g} ->
        pv = p[name]
        {:input, _, _, shape} = pv
        m = T.input(:"#{name}_m", :f32, shape)
        v = T.input(:"#{name}_v", :f32, shape)
        m2 = T.add(T.mul(m, T.splat(b1)), T.mul(g, T.splat(1 - b1)))
        v2 = T.add(T.mul(v, T.splat(b2)), T.mul(T.mul(g, g), T.splat(1 - b2)))
        step = T.mul(T.mul(m2, c1), T.rsqrt(T.add(T.mul(v2, c2), T.splat(eps * eps))))
        step = if wd > 0, do: T.add(step, T.mul(pv, T.splat(wd))), else: step
        [{:"#{name}_next", T.sub(pv, T.mul(step, lr_in))}, {:"#{name}_m_next", m2}, {:"#{name}_v_next", v2}]
      end)

    state = Enum.flat_map(keys, fn n -> [{n, :"#{n}_next"}, {:"#{n}_m", :"#{n}_m_next"}, {:"#{n}_v", :"#{n}_v_next"}] end)
    Program.new([loss: loss] ++ updates, state: state)
  end

  @doc """
  Per-step inputs for steps `from+1 … from+n` of `total`: AdamW's bias
  corrections and the learning rate — cosine from `lr` down to
  `lr·lr_end` (`lr_end`: 1.0 = constant) — from correctly rounded `pow` and
  `cos`, so the same on every machine.
  """
  def schedule(from, n, opts \\ []) do
    {b1, b2} = {Keyword.get(opts, :beta1, 0.9), Keyword.get(opts, :beta2, 0.999)}
    {lr, lr_end, total} = {Keyword.get(opts, :lr, 1.0e-3), Keyword.get(opts, :lr_end, 1.0), Keyword.get(opts, :total, from + n)}
    lrs = for k <- (from + 1)..(from + n), do: lr * (lr_end + (1 - lr_end) * 0.5 * (1 + CR.cos_f64(:math.pi() * (k - 1) / max(total, 1))))
    # beyond k_lim, bᵏ < 2⁻⁸⁰ and 1/(1 − bᵏ) rounds to 1 in binary32 whatever the libm: exactly 1
    c = fn b ->
      lim = 80 / -:math.log2(b)
      Tensor.from_list(:f32, [n, 1, 1], for(k <- (from + 1)..(from + n), do: if(k > lim, do: 1.0, else: 1 / (1 - CR.pow_f64(b, k * 1.0)))))
    end
    %{c1: c.(b1), c2: c.(b2), lr: Tensor.from_list(:f32, [n, 1, 1], lrs)}
  end

  @doc """
  Train on rows (`xs`, `ys`: lists of float lists) for `steps` steps of
  `batch` rows, batches drawn by a seeded permutation (epochs without
  replacement). Runs `chunk` steps per crossing. Options as
  `train_program/3`, plus `worker` (nil: the oracle, exact but slow),
  `seed`, `chunk` (100), `log` (a function of `{step, loss}`).
  Returns `{net, %{losses: [{step, loss}], steps, data_digest}}`.
  """
  def train(%__MODULE__{} = net, xs, ys, steps, opts \\ []) do
    batch = Keyword.get(opts, :batch, 64)
    chunk = min(Keyword.get(opts, :chunk, 100), steps)
    w = Keyword.get(opts, :worker)
    {:ok, comp} = Vapor.Compile.Lower.lower(train_program(net, batch, opts))
    {din, dout} = {hd(net.padded), List.last(net.padded)}
    n = length(xs)
    xt = xs |> Enum.map(&row(&1, din)) |> List.to_tuple()
    yt = ys |> Enum.map(&row(&1, dout)) |> List.to_tuple()
    order = epochs(n, steps * batch, Keyword.get(opts, :seed, 1))
    zeros = fn t -> Tensor.new(:f32, t.shape, :binary.copy(<<0::32>>, Enum.product(t.shape))) end
    state0 = Map.new(Enum.flat_map(net.params, fn {k, t} -> [{k, t}, {:"#{k}_m", zeros.(t)}, {:"#{k}_v", zeros.(t)}] end))
    log = Keyword.get(opts, :log, fn _ -> :ok end)

    {state, losses, _} =
      Enum.reduce(Stream.iterate(0, &(&1 + chunk)) |> Enum.take_while(&(&1 < steps)), {state0, [], order}, fn from, {state, losses, order} ->
        k = min(chunk, steps - from)
        {idx, rest} = Enum.split(order, k * batch)
        xb = Tensor.new(:f32, [k, batch, din], IO.iodata_to_binary(Enum.map(idx, &elem(xt, &1))))
        yb = Tensor.new(:f32, [k, batch, dout], IO.iodata_to_binary(Enum.map(idx, &elem(yt, &1))))
        env = state |> Map.merge(%{x: xb, y: yb}) |> Map.merge(schedule(from, k, Keyword.put(opts, :total, steps)))
        run = [iterations: k, sequence: [:x, :y, :c1, :c2, :lr], stream: false]
        {:ok, r} = if w, do: Native.run(w, comp, env, [isa: Substrates.host_isa(), mode: :native] ++ run), else: Native.run_oracle(comp, env, run)
        state = Map.new(state, fn {name, _} -> {name, r.outputs[:"#{name}_next"]} end)
        loss = r.outputs.loss |> Tensor.to_floats() |> hd()
        log.({from + k, loss})
        {state, [{from + k, loss} | losses], rest}
      end)

    net = %{net | params: Enum.map(net.params, fn {k, _} -> {k, state[k]} end)}
    {net, %{losses: Enum.reverse(losses), steps: steps, batch: batch, data_digest: data_digest(xt, yt)}}
  end

  defp row(vals, d), do: IO.iodata_to_binary([for(v <- vals, do: <<Vapor.F32.from_float(v * 1.0)::32-little>>), :binary.copy(<<0::32>>, d - length(vals))])

  defp data_digest(xt, yt), do: Base.encode16(:crypto.hash(:sha256, [Tuple.to_list(xt), Tuple.to_list(yt)]), case: :lower)

  # a seeded permutation per epoch, concatenated (SplitMix64-driven Fisher–Yates)
  defp epochs(n, total, seed) do
    Stream.iterate(0, &(&1 + 1))
    |> Stream.flat_map(fn e -> Vapor.Modal.Rng.permute(Enum.to_list(0..(n - 1)), seed * 7919 + e) end)
    |> Enum.take(total)
  end

  @doc "Predict rows (lists of floats) in chunks of `rows` (default 4096); options `worker`."
  def predict(%__MODULE__{} = net, xs, opts \\ []) do
    rows = Keyword.get(opts, :rows, 4096)
    {din, dout, dreal} = {hd(net.padded), List.last(net.padded), List.last(net.sizes)}
    comp = compiled(net, rows)
    w = Keyword.get(opts, :worker)

    xs
    |> Enum.chunk_every(rows)
    |> Enum.flat_map(fn ch ->
      data = IO.iodata_to_binary([Enum.map(ch, &row(&1, din)), :binary.copy(<<0::32>>, (rows - length(ch)) * din)])
      env = %{x: Tensor.new(:f32, [rows, din], data)}
      {:ok, r} = if w, do: Native.run(w, comp, env, isa: Substrates.host_isa(), mode: :native), else: Native.run_oracle(comp, env)
      r.outputs.y.data
      |> Vapor.F32.decode()
      |> Enum.map(&Vapor.F32.to_float/1)
      |> Enum.chunk_every(dout)
      |> Enum.take(length(ch))
      |> Enum.map(&Enum.take(&1, dreal))
    end)
  end

  defp compiled(net, rows) do
    key = {__MODULE__, :erlang.phash2({net.params, rows})}
    case :persistent_term.get(key, nil) do
      nil ->
        {:ok, c} = Vapor.Compile.Lower.lower(predict_program(net, rows))
        :persistent_term.put(key, c)
        c
      c -> c
    end
  end

  @doc "Weights as named tensors (padding removed is not attempted: the padded shapes are part of the model)."
  def tensors(%__MODULE__{params: ps}), do: Map.new(ps, fn {k, t} -> {Atom.to_string(k), t} end)

  @doc "A network from saved tensors and its sizes."
  def from_tensors(sizes, tensors, opts \\ []) do
    net = new(sizes, 0, opts)
    %{net | params: Enum.map(net.params, fn {k, _} -> {k, Map.fetch!(tensors, Atom.to_string(k))} end)}
  end

  @doc "SHA-256 of the weights."
  def digest(%__MODULE__{params: ps}), do: Base.encode16(:crypto.hash(:sha256, Enum.map(ps, fn {k, t} -> [Atom.to_string(k), t.data] end)), case: :lower)
end
