defmodule Vapor.MambaTest do
  @moduledoc """
  State-space models through the airlock (`Vapor.Lock.Adapters.Mamba`) and
  the two canonical functions they needed (`log`; `softplus`, a composition
  of canonical nodes).

  The model against transformers itself: `Vapor.MambaHFTest`.

  The reference for `log` is the correctly rounded binary64 logarithm
  (`Vapor.CR.log_f64`) rounded to binary32; for softplus, log1p(eˣ) in
  binary64 (by its series where 1 + eˣ would round).
  """
  use ExUnit.Case, async: false
  import Bitwise
  alias Vapor.{F32, Lock, Program, Recurrent, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Runtime.{Dispatch, Native, Oracle, Substrates, Worker}
  import Vapor.TestHelpers

  defp ulps(got, want) do
    if (bxor(got, want) &&& 0x8000_0000) != 0 and ((got ||| want) &&& 0x7FFF_FFFF) != 0,
      do: :infinity,
      else: abs((got &&& 0x7FFF_FFFF) - (want &&& 0x7FFF_FFFF))
  end

  defp normal_operands(n, seed) do
    :rand.seed(:exsss, {seed, 7, 7})
    for _ <- 1..n, do: 0x0080_0000 + :rand.uniform(0x7F7F_FFFF - 0x0080_0000)
  end

  defp softplus_ref(x) do
    u = :math.exp(x)

    cond do
      x > 20 -> x
      u < 1.0e-3 -> u - u * u / 2 + u * u * u / 3 - u * u * u * u / 4 + u * u * u * u * u / 5
      true -> :math.log(1 + u)
    end
  end

  defp eval1(term_fun, bits) do
    n = length(bits)
    p = Program.new(y: term_fun.(T.input(:x, :f32, [n])))
    Oracle.eval_program(p, %{x: Tensor.new(:f32, [n], F32.encode(bits))}).y |> Tensor.to_list()
  end

  test "log: within an ulp of the correctly rounded logarithm; IEEE specials" do
    xs = normal_operands(60_000, 1) ++ Enum.map(-3000..3000, &F32.from_float(1.0 + &1 * 1.0e-6))
    got = eval1(&T.log/1, xs)

    worst =
      Enum.zip(xs, got)
      |> Enum.map(fn {x, g} -> ulps(g, F32.from_float(Vapor.CR.log_f64(F32.to_float(x)))) end)
      |> Enum.max()

    assert worst <= 1

    #        +0           −0           +∞           −∞           NaN          −1           subnormal    1
    specials = [0, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0001, 0xBF80_0000, 0x0000_0001, 0x3F80_0000]
    assert eval1(&T.log/1, specials) == [0xFF80_0000, 0xFF80_0000, 0x7F80_0000, 0x7FC0_0000, 0x7FC0_0000, 0x7FC0_0000, 0xFF80_0000, 0]
  end

  test "softplus: within a few ulps of log1p(eˣ) over [−60, 60]; x itself above 20 (PyTorch's threshold)" do
    xs = for i <- 0..12_000, do: F32.from_float(-60.0 + i * 0.01)
    got = eval1(&T.softplus/1, xs)

    worst =
      Enum.zip(xs, got)
      |> Enum.reject(fn {x, _} -> softplus_ref(F32.to_float(x)) < 1.2e-38 end)
      |> Enum.map(fn {x, g} -> ulps(g, F32.from_float(softplus_ref(F32.to_float(x)))) end)
      |> Enum.max()

    assert worst <= 4
    assert eval1(&T.softplus/1, [F32.from_float(25.5)]) == [F32.from_float(25.5)]
  end

  @tag :native
  test "log and softplus: host ISAs, the RVV interpreter and the fabric = the oracle" do
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    xs = normal_operands(4096, 2) ++ for(i <- 0..4095, do: F32.from_float(-40.0 + i * 0.02))
    n = length(xs)
    e = %{x: Tensor.new(:f32, [n], F32.encode(xs))}

    for f <- [&T.log/1, &T.softplus/1] do
      p = Program.new(y: f.(T.input(:x, :f32, [n])))
      want = Oracle.eval_program(p, e).y
      {:ok, c} = Lower.lower(p)
      for isa <- Substrates.host_isas(), do: assert(elem(Native.run(w, c, e, isa: isa, mode: :native), 1).outputs.y == want, "#{isa}")
      assert elem(Native.run(w, c, e, isa: :riscv64, mode: :emulate, vlen: 256, poison: true), 1).outputs.y == want

      case Enum.find(Substrates.list(), &(&1.kind == :fabric)) do
        nil -> :ok
        fabric -> assert elem(Dispatch.run_on(fabric, c, e, []), 1).outputs.y == want
      end
    end
  end

  test "contracts: the engine refuses a recurrent model (no paged cache to serve); Recurrent refuses an attention model" do
    cfg = %{"model_type" => "mamba", "vocab_size" => 32, "hidden_size" => 32, "intermediate_size" => 64, "num_hidden_layers" => 1,
            "state_size" => 16, "time_step_rank" => 16, "conv_kernel" => 4}
    ws = random_weights(Vapor.Lock.Adapters.Mamba.spec(struct(Vapor.Lock.Adapters.Mamba.Config, config_fields(cfg))))
    {:ok, spec, _} = Vapor.Lock.Adapters.Mamba.admit(%{config: cfg}, ws, [])
    {:ok, p} = Lock.build(spec, ws, [])
    assert :ok == Vapor.Lock.Contract.check(spec, p)
    assert {:error, %Vapor.Rejection{}} = Vapor.Engine.prepare(config: spec, weights: ws)
    # a missing tensor is refused at the lock, by name
    assert {:error, %Vapor.Rejection{node: {:weight, "backbone.norm_f.weight"}}} =
             Vapor.Lock.Adapters.Mamba.admit(%{config: cfg}, Map.delete(ws, "backbone.norm_f.weight"), [])

    {:ok, c} = Vapor.Model.Config.from_map(tiny_config("llama"))
    assert {:error, %Vapor.Rejection{}} = Recurrent.open(Lock.spec(c), tiny_weights(c))
  end

  # weights of the right names and shapes for a spec
  defp random_weights(spec) do
    for {name, shape, kind} <- Lock.expected(spec), into: %{} do
      t = Tensor.random(:f32, shape, :erlang.phash2(name), scale: 0.2)
      {name, if(kind == :norm, do: Tensor.from_list(:f32, shape, Enum.map(Tensor.to_floats(t), &(&1 + 1.0))), else: t)}
    end
  end

  defp config_fields(c) do
    [vocab: c["vocab_size"], hidden: c["hidden_size"], inner: c["intermediate_size"], layers: c["num_hidden_layers"],
     state: c["state_size"], rank: c["time_step_rank"], conv: c["conv_kernel"], eps: 1.0e-5, bias: false, conv_bias: true,
     tie: true, bos: nil, eos: nil, raw: c]
  end
end
