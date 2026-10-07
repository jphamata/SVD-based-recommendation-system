defmodule Vapor.Train do
  @moduledoc """
  LoRA fine-tuning by distillation, as one recurrent program (phase P7).

  The trainable part is the last decoder layer's MLP and everything after
  it: low-rank adapters `W + s·B·A` (`s = α/r`) on its gate, up and down
  projections, then the final norm and the output head. The frozen layers
  before it run once, in inference, to produce the block's input features
  `h` (`features/4`). One *training step* is then a pure function of terms:

      forward   z = head(norm(h + down_LoRA(silu(gate_LoRA(h₂)) ⊙ up_LoRA(h₂))))
      loss      KL(p_teacher ‖ softmax z), mean over the T rows
      backward  ∂loss/∂z = (softmax z − p_teacher)/T   (exact for this loss)
                then `Vapor.Autodiff` down to the six adapter matrices
      update    AdamW: m ← β₁m + (1−β₁)g, v ← β₂v + (1−β₂)g²,
                p ← p − lr·(m·ĉ₁·rsqrt(v·ĉ₂ + ε²) + λp)

  with the bias corrections `ĉ₁ = 1/(1−β₁ᵗ)`, `ĉ₂ = 1/(1−β₂ᵗ)` fed per step.
  (`rsqrt(v̂ + ε²)` puts ε inside the root — the one deviation from the
  textbook form, so that only canonical operators appear.)

  Parameters and moments are *state*: the whole training run is
  `Vapor.Runtime.Native.run(…, iterations: steps)` — every step inside the
  worker, one crossing in total, the logits of each step streamed back for
  the loss curve. Like every program it is bit-identical on every
  substrate and for any thread count, so a training run is reproducible
  to the last bit.

  The loss itself (`kl/2`) is evaluated in the BEAM from the streamed
  logits, in binary64.
  """
  alias Vapor.{Autodiff, Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Model.Config

  @adapters [:gate, :up, :down]

  @doc """
  The training-step program. `w` holds the frozen tensors:
  `ln2` (f32[d]), `wg`, `wu` (f32[ff, d]), `wd` (f32[d, ff]), `norm`
  (f32[d]), `head` (f32[V, d]). Options: `rank` (16), `alpha` (32),
  `tokens` (rows per step, a multiple of 16; default 16), `lr` (1e-3),
  `beta1` (0.9), `beta2` (0.999), `eps` (1e-8), `weight_decay` (0).
  """
  def program(%Config{} = c, w, opts \\ []) do
    r = Keyword.get(opts, :rank, 16)
    s = Keyword.get(opts, :alpha, 32) / r
    t = Keyword.get(opts, :tokens, 16)
    {d, ff, v} = {c.hidden, c.intermediate, c.vocab}
    {b1, b2} = {Keyword.get(opts, :beta1, 0.9), Keyword.get(opts, :beta2, 0.999)}
    {lr, eps, wd} = {Keyword.get(opts, :lr, 1.0e-3), Keyword.get(opts, :eps, 1.0e-8), Keyword.get(opts, :weight_decay, 0.0)}

    row = fn x -> T.const(Tensor.new(:f32, [1 | x.shape], x.data)) end
    {ln2, norm} = {row.(w.ln2), row.(w.norm)}
    {wg, wu, wdn, head} = {T.const(w.wg), T.const(w.wu), T.const(w.wd), T.const(w.head)}

    h = T.input(:h, :f32, [t, d])
    pt = T.input(:p_teacher, :f32, [t, v])
    c1 = T.input(:c1, :f32, [1, 1])
    c2 = T.input(:c2, :f32, [1, 1])

    shapes = %{gate: {[r, d], [ff, r]}, up: {[r, d], [ff, r]}, down: {[r, ff], [d, r]}}
    params = for a <- @adapters, {name, shape} <- [{:"#{a}_a", elem(shapes[a], 0)}, {:"#{a}_b", elem(shapes[a], 1)}], do: {name, T.input(name, :f32, shape)}
    p = Map.new(params)
    lora = fn x, wfull, a -> T.add(T.linear(x, wfull), T.mul(T.linear(T.linear(x, p[:"#{a}_a"]), p[:"#{a}_b"]), T.splat(s))) end

    h2 = rmsnorm(h, ln2, c.eps, d)
    act = T.mul(T.silu(lora.(h2, wg, :gate)), lora.(h2, wu, :up))
    xo = T.add(h, lora.(act, wdn, :down))
    z = T.linear(rmsnorm(xo, norm, c.eps, d), head)

    # ∂KL/∂z = (softmax z − p)/T
    e = T.exp(T.sub(z, T.reduce(:max, z)))
    q = T.mul(e, T.rcp(T.reduce(:sum, e)))
    gz = T.mul(T.sub(q, pt), T.splat(1.0 / t))
    {:ok, grads} = Autodiff.grad(z, gz, Enum.map(params, &elem(&1, 1)))

    updates =
      Enum.zip(params, grads)
      |> Enum.flat_map(fn {{name, pv}, g} ->
        {:input, _, _, shape} = pv
        m = T.input(:"#{name}_m", :f32, shape)
        vv = T.input(:"#{name}_v", :f32, shape)
        m2 = T.add(T.mul(m, T.splat(b1)), T.mul(g, T.splat(1 - b1)))
        v2 = T.add(T.mul(vv, T.splat(b2)), T.mul(T.mul(g, g), T.splat(1 - b2)))
        step = T.mul(T.mul(m2, c1), T.rsqrt(T.add(T.mul(v2, c2), T.splat(eps * eps))))
        step = if wd > 0, do: T.add(step, T.mul(pv, T.splat(wd))), else: step
        p2 = T.sub(pv, T.mul(step, T.splat(lr)))
        [{:"#{name}_next", p2}, {:"#{name}_m_next", m2}, {:"#{name}_v_next", v2}]
      end)

    state =
      Enum.flat_map(params, fn {name, _} ->
        [{name, :"#{name}_next"}, {:"#{name}_m", :"#{name}_m_next"}, {:"#{name}_v", :"#{name}_v_next"}]
      end)

    Program.new([logits: z] ++ updates, state: state)
  end

  @doc "Initial state: A small and random, B zero (the student starts as the base model), moments zero."
  def init(%Config{} = c, opts \\ []) do
    r = Keyword.get(opts, :rank, 16)
    seed = Keyword.get(opts, :seed, 1)
    {d, ff} = {c.hidden, c.intermediate}
    zeros = fn shape -> Tensor.new(:f32, shape, :binary.copy(<<0::32>>, Enum.product(shape))) end
    shapes = %{gate: {[r, d], [ff, r]}, up: {[r, d], [ff, r]}, down: {[r, ff], [d, r]}}

    for {a, i} <- Enum.with_index(@adapters), {suffix, shape, init} <- [{"a", elem(shapes[a], 0), :rand}, {"b", elem(shapes[a], 1), :zero}],
        name = :"#{a}_#{suffix}", {key, val} <- [{name, if(init == :rand, do: Tensor.random(:f32, shape, seed + i, scale: 1 / :math.sqrt(List.last(shape))), else: zeros.(shape))},
                                                  {:"#{name}_m", zeros.(shape)}, {:"#{name}_v", zeros.(shape)}],
        into: %{},
        do: {key, val}
  end

  @doc "Bias corrections for steps 1…n: sequence inputs `c1, c2 : f32[n, 1, 1]`."
  def schedule(n, opts \\ []) do
    {b1, b2} = {Keyword.get(opts, :beta1, 0.9), Keyword.get(opts, :beta2, 0.999)}
    c = fn b -> Tensor.from_list(:f32, [n, 1, 1], for(k <- 1..n, do: 1 / (1 - :math.pow(b, k)))) end
    %{c1: c.(b1), c2: c.(b2)}
  end

  @doc "Mean KL(p ‖ softmax z) over rows, in binary64 (`p`, `z` : f32[T, V])."
  def kl(%Tensor{shape: [_, v]} = p, %Tensor{} = z) do
    ps = p |> Tensor.to_floats() |> Enum.chunk_every(v)
    zs = z |> Tensor.to_floats() |> Enum.chunk_every(v)

    rows =
      Enum.zip_with(ps, zs, fn pr, zr ->
        m = Enum.max(zr)
        lse = m + :math.log(Enum.sum(Enum.map(zr, &:math.exp(&1 - m))))
        Enum.zip_with(pr, zr, fn pi, zi -> if pi > 0, do: pi * (:math.log(pi) - (zi - lse)), else: 0.0 end) |> Enum.sum()
      end)

    Enum.sum(rows) / length(rows)
  end

  defp rmsnorm(x, w, eps, d) do
    ms = T.mul(T.reduce(:sum, T.mul(x, x)), T.splat(1.0 / d))
    T.mul(w, T.mul(x, T.rsqrt(T.add(ms, T.splat(eps)))))
  end
end
