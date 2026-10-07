defmodule Vapor.AutodiffTest do
  @moduledoc """
  Phase P7 — gradients as programs. The vector–Jacobian products built by
  `Vapor.Autodiff` are checked against central finite differences of an
  independent binary64 evaluation, then compiled and run: oracle = host =
  RVV interpreter, bit for bit, for any thread count. The transposes the
  backward pass relies on satisfy the adjoint identity exactly.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Autodiff, Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Runtime.{Native, Oracle, Worker}
  import Vapor.TestHelpers

  # y = x + W₂·silu(W₁·(g ⊙ rmsnorm x)), the shape of a transformer MLP block
  defp block do
    x = T.input(:x, :f32, [16, 16])
    g = T.input(:g, :f32, [1, 16])
    w1 = T.input(:w1, :f32, [32, 16])
    w2 = T.input(:w2, :f32, [16, 32])
    ms = T.mul(T.reduce(:sum, T.mul(x, x)), T.splat(1 / 16))
    h = T.mul(g, T.mul(x, T.rsqrt(T.add(ms, T.splat(1.0e-5)))))
    y = T.add(x, T.linear(T.silu(T.linear(h, w1)), w2))
    {y, [x, g, w1, w2]}
  end

  defp env do
    %{x: Tensor.random(:f32, [16, 16], 1), g: Tensor.random(:f32, [1, 16], 2, scale: 0.5) |> shift(1.0),
      w1: Tensor.random(:f32, [32, 16], 3, scale: 0.3), w2: Tensor.random(:f32, [16, 32], 4, scale: 0.3),
      seed: Tensor.random(:f32, [16, 16], 5)}
  end

  defp shift(t, c), do: Tensor.from_list(:f32, t.shape, Enum.map(Tensor.to_floats(t), &(&1 + c)))
  defp f64(t), do: {t.shape, t |> Tensor.to_floats() |> List.to_tuple()}

  test "vector–Jacobian products agree with binary64 central differences" do
    {y, wrt} = block()
    seed = T.input(:seed, :f32, [16, 16])
    {:ok, grads} = Autodiff.grad(y, seed, wrt)
    e = env()
    got = Oracle.eval_many(grads, e)

    # L(θ) = ⟨seed, y(θ)⟩ in binary64
    loss = fn env64 ->
      {_, yv} = Vapor.TestF64.eval(y, env64)
      {_, sv} = env64.seed
      Enum.sum(Enum.zip_with(Tuple.to_list(yv), Tuple.to_list(sv), &(&1 * &2)))
    end

    base = Map.new(e, fn {k, t} -> {k, f64(t)} end)
    :rand.seed(:exsss, {9, 9, 9})

    for {{:input, name, _, _}, g} <- Enum.zip(wrt, got), _ <- 1..6 do
      {shape, v} = base[name]
      i = :rand.uniform(tuple_size(v)) - 1
      h = 1.0e-4 * max(1.0, abs(elem(v, i)))
      at = fn d -> Map.put(base, name, {shape, put_elem(v, i, elem(v, i) + d)}) end
      fd = (loss.(at.(h)) - loss.(at.(-h))) / (2 * h)
      an = g |> Tensor.to_floats() |> Enum.at(i)
      scale = g |> Tensor.to_floats() |> Enum.map(&abs/1) |> Enum.max()
      assert abs(an - fd) <= 2.0e-3 * max(scale, 1.0), "∂/∂#{name}[#{i}]: autodiff #{an}, finite differences #{fd}"
    end
  end

  test "transpose is a permutation of the values: applied twice it is the identity" do
    a = Tensor.random(:f32, [16, 48], 1)
    c = elem(Vapor.Compile.Lower.lower(Program.new(t: T.transpose(T.input(:a, :f32, [16, 48])))), 1)
    {:ok, %{outputs: %{t: at}}} = Native.run_oracle(c, %{a: a})
    assert at.shape == [48, 16]
    assert Oracle.eval(T.transpose(T.const(at)), %{}).data == a.data
  end

  @tag :native
  test "gradient programs run bit-identically on the oracle, host (1–3 threads) and RVV interpreter" do
    {y, wrt} = block()
    {:ok, grads} = Autodiff.grad(y, T.input(:seed, :f32, [16, 16]), wrt)
    p = Program.new(Enum.zip([:dx, :dg, :dw1, :dw2], grads))
    assert :ok = Program.check(p)
    {:ok, c} = Vapor.Compile.Lower.lower(p)
    {:ok, ref} = Native.run_oracle(c, env())

    for threads <- [1, 3] do
      {:ok, w} = Worker.start_link(exec: worker_exec(:host), threads: threads)
      for isa <- Vapor.Runtime.Substrates.host_isas() do
        {:ok, got} = Native.run(w, c, env(), isa: isa, mode: :native)
        assert got.outputs == ref.outputs, inspect(isa)
      end
      {:ok, emu} = Native.run(w, c, env(), isa: :riscv64, mode: :emulate, vlen: 256, poison: true)
      assert emu.outputs == ref.outputs
    end
  end

  @tag :vulkan
  test "gradient programs (transposes included) run bit-identically on the Vulkan fabric" do
    {y, wrt} = block()
    {:ok, grads} = Autodiff.grad(y, T.input(:seed, :f32, [16, 16]), wrt)
    {:ok, c} = Vapor.Compile.Lower.lower(Program.new(Enum.zip([:dx, :dg, :dw1, :dw2], grads)))
    {:ok, ref} = Native.run_oracle(c, env())
    fabric = Enum.find(Vapor.Runtime.Substrates.list(), &(&1.kind == :fabric))
    assert {:ok, %{outputs: got}} = Vapor.Runtime.Dispatch.run_on(fabric, c, env(), [])
    assert got == ref.outputs
  end

  test "operators off the differentiable subset are refused by name" do
    x = T.input(:x, :f32, [16, 16])
    assert {:error, %Vapor.Rejection{}} = Autodiff.grad(T.reduce(:max, x), T.input(:s, :f32, [16, 1]), [x])
  end
end
