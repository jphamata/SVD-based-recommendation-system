defmodule Vapor.CupelTest do
  use ExUnit.Case, async: true
  import Bitwise
  alias Vapor.{Cupel, F32, Tensor}
  alias Vapor.Cupel.Sentinel
  alias Vapor.Runtime.Worker
  import Vapor.TestHelpers

  defp rand(shape, seed, scale \\ 1.0) do
    t = Tensor.random(:f32, shape, seed)
    Tensor.from_list(:f32, shape, Enum.map(Tensor.to_floats(t), &(&1 * scale)))
  end

  # a conforming substrate that sums in its own order (any order is allowed by the envelope)
  defp linear_in_order(w, x, order) do
    [n, k] = w.shape
    [b, ^k] = x.shape
    wr = w |> Tensor.to_list() |> Enum.chunk_every(k)
    xr = x |> Tensor.to_list() |> Enum.chunk_every(k)

    out =
      for xi <- xr, wj <- wr do
        prods = Enum.zip_with(xi, wj, &F32.mul/2)

        case order do
          :forward -> Enum.reduce(prods, 0, &F32.add(&2, &1))
          :reverse -> prods |> Enum.reverse() |> Enum.reduce(0, &F32.add(&2, &1))
          :pairwise -> pairwise(prods)
          :fma -> Enum.zip(xi, wj) |> Enum.reduce(0, fn {a, c}, acc -> F32.fma(a, c, acc) end)
        end
      end

    Tensor.new(:f32, [b, n], F32.encode(out))
  end

  defp pairwise([v]), do: v
  defp pairwise(xs), do: xs |> Enum.chunk_every(2) |> Enum.map(fn [a, b] -> F32.add(a, b); [a] -> a end) |> pairwise()

  describe "never accuses a correct substrate" do
    test "the oracle and four summation orders (forward, reverse, pairwise, fused) all pass, over scales from 1e-30 to 1e15" do
      for {seed, scale} <- [{1, 1.0}, {2, 1.0e-30}, {3, 1.0e15}, {4, 3.0e-3}] do
        w = rand([24, 48], seed, scale)
        p = Cupel.probe(w, seed: seed)
        x = rand([5, 48], seed + 100, 1.0 / scale)
        assert {:ok, _} = Cupel.assay(p, x, Cupel.oracle_linear(w, x))

        for order <- [:forward, :reverse, :pairwise, :fma] do
          assert {:ok, rep} = Cupel.assay(p, x, linear_in_order(w, x, order)), "#{order} at scale #{scale}"
          assert rep.worst < 1.0
        end
      end
    end

    test "subnormal weights and activations (a DAZ/FTZ substrate reads them as zero) stay inside the envelope" do
      tiny = for i <- 1..64, do: if(rem(i, 3) == 0, do: i, else: F32.from_float(1.0e-3 * i))
      w = Tensor.new(:f32, [4, 16], F32.encode(tiny))
      x = Tensor.new(:f32, [2, 16], F32.encode(for i <- 1..32, do: if(rem(i, 5) == 0, do: 7, else: F32.from_float(i * 1.0))))
      p = Cupel.probe(w)
      # a substrate that flushes every subnormal operand to zero
      daz = fn t -> Tensor.new(:f32, t.shape, F32.encode(Enum.map(Tensor.to_list(t), &if(F32.subnormal?(&1), do: 0, else: &1)))) end
      assert {:ok, _} = Cupel.assay(p, x, Cupel.oracle_linear(daz.(w), daz.(x)))
      assert {:ok, _} = Cupel.assay(p, x, Cupel.oracle_linear(w, x))
    end

    test "a subnormal weight times a huge activation: the DAZ substrate's zero is far outside γₖ·Σ|x||W| and still correct" do
      # w₀ = 2⁻¹³⁰ (subnormal), x₀ = 2¹⁰⁰: the exact product 2⁻³⁰; a DAZ substrate reads w₀ as 0
      w = Tensor.new(:f32, [1, 16], F32.encode([1 <<< 19 | List.duplicate(0, 15)]))
      x = Tensor.from_list(:f32, [1, 16], [:math.pow(2, 100) | List.duplicate(0.0, 15)])
      p = Cupel.probe(w)
      assert {:ok, _} = Cupel.assay(p, x, Tensor.from_list(:f32, [1, 1], [:math.pow(2, -30)]))
      assert {:ok, _} = Cupel.assay(p, x, Tensor.from_list(:f32, [1, 1], [0.0]))
      # …but not anything: twice the product is no substrate's answer
      assert {:corrupt, _} = Cupel.assay(p, x, Tensor.from_list(:f32, [1, 1], [:math.pow(2, -28)]))
    end

    test "bf16 weights: the probe reads them as their exact f32 values" do
      w = rand([16, 32], 9) |> Tensor.to_bf16()
      x = rand([3, 32], 10)
      assert {:ok, _} = Cupel.check(w, x, Cupel.oracle_linear(Tensor.widen(w), x))
    end
  end

  describe "sees what correct arithmetic cannot produce" do
    test "every sign and exponent bit flip is caught; the profile over all 32 bits is reported" do
      w = rand([32, 64], 21)
      xs = for s <- 1..12, do: rand([4, 64], 300 + s)
      profile = Cupel.sensitivity(w, xs, seed: 5)
      by_bit = Map.new(profile, fn {b, hit, n} -> {b, hit / n} end)
      for b <- 23..31, do: assert(by_bit[b] == 1.0, "bit #{b}")
      # high mantissa bits of an output are far above the rounding envelope
      for b <- 19..22, do: assert(by_bit[b] == 1.0, "bit #{b}")
      # the lowest bit is below it: the cupel says so instead of pretending
      assert by_bit[0] < 1.0
    end

    test "int8: exact, every one of the 32 bits of an s32 result is caught" do
      w = Tensor.from_list(:s8, [8, 16], for(i <- 1..128, do: rem(i * 37, 255) - 127))
      x = Tensor.from_list(:s8, [3, 16], for(i <- 1..48, do: rem(i * 91, 255) - 127))
      y = Cupel.int_linear(w, x)
      p = Cupel.probe(w, seed: 3)
      assert {:ok, _} = Cupel.assay(p, x, y)

      for b <- 0..31, pos <- [0, 7, 23] do
        assert {:corrupt, %{corrupt: [_ | _]}} = Cupel.assay(p, x, Cupel.flip(y, pos, b))
      end
    end

    test "a non-finite output from finite inputs is corruption; non-finite inputs are 'unchecked', never 'ok'" do
      w = rand([4, 16], 31)
      x = rand([2, 16], 32)
      y = Cupel.oracle_linear(w, x)
      p = Cupel.probe(w)
      <<_::binary-size(4), rest::binary>> = y.data
      nan_y = %{y | data: <<0x7FC0_0000::32-little, rest::binary>>}
      assert {:corrupt, %{corrupt: [0]}} = Cupel.assay(p, x, nan_y)

      <<_::binary-size(4), xrest::binary>> = x.data
      inf_x = %{x | data: <<0x7F80_0000::32-little, xrest::binary>>}
      # (the oracle is defined over finite values only; a substrate returns ∞/NaN here)
      <<_::binary-size(16), yrest::binary>> = y.data
      inf_y = %{y | data: <<0x7F80_0000::32-little, 0xFF80_0000::32-little, 0x7FC0_0000::32-little, 0x7F80_0000::32-little, yrest::binary>>}
      assert {:ok, %{unchecked: [0]}} = Cupel.assay(p, inf_x, inf_y)
    end

    test "outputs that may legitimately overflow are unchecked, not accused" do
      w = Tensor.from_list(:f32, [1, 16], List.duplicate(3.0e38, 16))
      x = Tensor.from_list(:f32, [1, 16], List.duplicate(1.0, 16))
      # 16 · 3e38 overflows: a correct substrate returns +∞
      y = Tensor.new(:f32, [1, 1], <<0x7F80_0000::32-little>>)
      assert {:ok, %{unchecked: [0]}} = Cupel.check(w, x, y)
    end

    test "cancellation needs the seed: a forger who knows r slips through, the same forgery under another seed does not" do
      w = rand([16, 16], 41)
      x = rand([1, 16], 42)
      y = Cupel.oracle_linear(w, x)
      p = Cupel.probe(w, seed: 99)
      [r0, r1 | _] = p.r
      # δ₀·r₀ + δ₁·r₁ = 0 with large δ: add r₁·2^k to y₀ and subtract r₀·2^k from y₁
      [y0, y1 | rest] = Tensor.to_floats(y)
      forged = Tensor.from_list(:f32, [1, 16], [y0 + r1 * 0.25, y1 - r0 * 0.25 | rest])
      assert {:ok, _} = Cupel.assay(p, x, forged)
      assert {:corrupt, _} = Cupel.assay(Cupel.probe(w, seed: 100), x, forged)
    end

    test "shapes and dtypes that do not belong to the probe are refused" do
      p = Cupel.probe(rand([4, 8], 1))
      assert_raise ArgumentError, fn -> Cupel.assay(p, rand([2, 7], 2), rand([2, 4], 3)) end
      assert_raise ArgumentError, fn -> Cupel.probe(Tensor.from_list(:f32, [1, 2], [1.0, 0.0]) |> then(&%{&1 | data: <<0x7F80_0000::32-little, 0::32>>})) end
    end
  end

  describe "the sentinel" do
    defp idle, do: spawn(fn -> receive do: (:stop -> :ok) end)

    test "a corrupting substrate is quarantined on the evidence; callers still get correct bits; the journal says so" do
      w = rand([16, 32], 51)
      x = rand([3, 32], 52)
      good = Cupel.oracle_linear(w, x)
      [a, b, c] = for _ <- 1..3, do: idle()
      runner = fn wk, x -> y = Cupel.oracle_linear(w, x); if wk == a, do: {:ok, Cupel.flip(y, 5, 30)}, else: {:ok, y} end
      {:ok, s} = Sentinel.start_link(w: w, workers: [a, b, c], runner: runner, seed: 7)

      assert {:ok, y, %{by: "w1", attempts: [{"w0", :corrupt}]}} = Sentinel.linear(s, x)
      assert y.data == good.data
      assert Sentinel.healthy(s) == ["w1", "w2"]
      assert Sentinel.removed(s) == %{"w0" => "quarantined"}
      assert {:ok, _, %{by: "w1", attempts: []}} = Sentinel.linear(s, x)

      kinds = s |> Sentinel.journal() |> Enum.map(& &1["kind"])
      assert kinds == ["probe", "quarantined", "served", "served"]
      assert byte_size(Sentinel.merkle_root(s)) == 32
    end

    test "a worker that dies is a journal entry, not an outage; with no healthy worker the oracle serves" do
      w = rand([8, 16], 61)
      x = rand([2, 16], 62)
      [a, b] = for _ <- 1..2, do: idle()
      runner = fn _wk, _x -> {:error, :fault} end
      {:ok, s} = Sentinel.start_link(w: w, workers: [a, b], runner: runner)
      send(a, :stop)
      Process.sleep(50)
      assert Sentinel.removed(s)["w0"] == "worker_crashed"
      assert {:ok, y, %{by: "oracle", attempts: [{"w1", :failed}]}} = Sentinel.linear(s, x)
      assert y.data == Cupel.oracle_linear(w, x).data
      assert Sentinel.healthy(s) == []
    end

    test "concurrent callers against a pool with one bad core: every answer is correct" do
      w = rand([16, 16], 71)
      xs = for i <- 1..20, do: rand([2, 16], 700 + i)
      [a, b] = for _ <- 1..2, do: idle()
      runner = fn wk, x -> y = Cupel.oracle_linear(w, x); if wk == b, do: {:ok, Cupel.flip(y, 1, 27)}, else: {:ok, y} end
      {:ok, s} = Sentinel.start_link(w: w, workers: [b, a], runner: runner)

      results = xs |> Task.async_stream(&Sentinel.linear(s, &1), max_concurrency: 8) |> Enum.map(fn {:ok, {:ok, y, _}} -> y end)
      for {x, y} <- Enum.zip(xs, results), do: assert(y.data == Cupel.oracle_linear(w, x).data)
      assert Sentinel.removed(s) == %{"w0" => "quarantined"}
    end

    @tag :native
    test "native workers: the real AVX path passes, a worker made to flip a bit is quarantined" do
      w = rand([64, 128], 81)
      x = rand([4, 128], 82)
      ws = for _ <- 1..2, do: elem(Worker.start_link(exec: worker_exec(:host)), 1)
      bad = hd(ws)
      runner = fn wk, x ->
        with {:ok, y} <- Sentinel.native_linear(w, wk, x), do: {:ok, if(wk == bad, do: Cupel.flip(y, 17, 24), else: y)}
      end

      {:ok, s} = Sentinel.start_link(w: w, workers: ws, runner: runner, seed: 11)
      assert {:ok, y, %{by: "w1"}} = Sentinel.linear(s, x)
      assert {:ok, _} = Cupel.check(w, x, y, seed: 12)
      assert Sentinel.removed(s) == %{"w0" => "quarantined"}
    end
  end

  test "the oracle refuses a contraction it cannot evaluate canonically instead of dropping its tail" do
    assert_raise ArgumentError, ~r/mod 16/, fn -> Cupel.oracle_linear(rand([2, 40], 1), rand([1, 40], 2)) end
  end

  test "the probe is a pure function of the weights and the seed" do
    w = rand([8, 8], 91)
    assert Cupel.probe(w, seed: 4) == Cupel.probe(w, seed: 4)
    refute Cupel.probe(w, seed: 4).r == Cupel.probe(w, seed: 5).r
    assert Enum.all?(Cupel.draws(4, 1000), &(&1 != 0 and abs(&1) <= 1 <<< 20))
  end
end
