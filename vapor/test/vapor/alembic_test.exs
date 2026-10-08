defmodule Vapor.AlembicTest do
  # not async: two tests count atoms VM-wide, which concurrent tests would disturb
  use ExUnit.Case, async: false
  alias Vapor.Alembic
  alias Vapor.Alembic.Tree

  defp ev(text, prog \\ nil) do
    case Alembic.eval(text, program: prog) do
      {:ok, v} -> v
      {:error, e} -> {:error, e.message}
    end
  end

  describe "the language" do
    test "precedence, power, unary minus and chained comparisons" do
      assert ev("1 + 2 * 3 ^ 2") == 19
      assert ev("-2 ^ 2") == -4
      assert ev("2 ^ 3 ^ 2") == 512
      assert ev("7 // 2") == 3 and ev("-7 // 2") == -4 and ev("-7 % 3") == 2
      assert ev("1 / 4") == 0.25
      assert ev("0 <= 3 < 5") == true and ev("0 <= 7 < 5") == false
      assert ev("2 in [1, 2, 3] and not (4 in [1, 2])")
      assert ev("6 & 3 | 8") == 10 and ev("5 xor 1") == 4 and ev("1 << 70") == Integer.pow(2, 70)
    end

    test "comprehensions, let, lambdas, pipes, slices, fields" do
      assert ev("[x * y for x in 1..3 for y in 1..x if (x + y) % 2 == 0]") == [1, 4, 3, 9]
      assert ev("let a = 2, b = a + 1 in a * b") == 6
      assert ev("((a, b) => a - b)(10, 3)") == 7
      assert ev("[1, 2, 3, 4] |> map(x => x * x) |> filter(x => x > 2) |> sum") == 29
      assert ev("[0, 1, 2, 3, 4, 5][1:4]") == [1, 2, 3] and ev("[0, 1, 2][-1]") == 2
      assert ev("{name: \"ana\", n: 3}.n") == 3
      assert ev("[(a, b) for (a, b) in zip([1, 2], [3, 4])]") == [{1, 3}, {2, 4}]
      assert ev("fold([1, 2, 3], 10, (acc, x) => acc + x)") == 16
    end

    test "programs: definitions in any order, recursion, constants evaluated once" do
      {:ok, p} = Alembic.load("""
      total = f(10)
      f(n) = if n == 0 then 0 else n + f(n - 1)
      primes = [p for p in 2..50 if is_prime(p)]
      """)
      assert Alembic.const(p, "total") == 55
      assert length(Alembic.const(p, "primes")) == 15
      assert Alembic.call(p, "f", [100]) == {:ok, 5050}
    end

    test "newlines continue an expression after an operator or before |>, and end it otherwise" do
      {:ok, q} = Alembic.load("a = 1 +\n  2\nb = [1, 2]\n  |> sum")
      assert Alembic.const(q, "a") == 3 and Alembic.const(q, "b") == 3
    end

    test "errors carry line and column, and suggest names" do
      assert {:error, %{line: 3, message: m}} = Alembic.load("a = 1\nb = (2 +\n")
      assert m =~ "unexpected"
      assert {:error, %{message: "x is defined twice", line: 2}} = Alembic.load("x = 1\nx = 2")
      assert {:error, msg} = ev("lenn([1])")
      assert msg =~ "did you mean len"
    end
  end

  describe "sanitised: open input cannot hurt the host" do
    test "every loop and call pays fuel; recursion and sizes are bounded" do
      {:ok, p} = Alembic.load("spin(x) = spin(x + 1)\nloop(n) = fold(range(n), 0, (a, k) => a + k)")
      assert {:error, m} = Alembic.call(p, "spin", [0])
      assert m =~ "recursion"
      assert {:error, m2} = Alembic.call(p, "loop", [1_900_000], fuel: 10_000)
      assert m2 =~ "fuel"
      assert {:error, m3} = ev("2 ^ 100000000")
      assert m3 =~ "bits"
      assert {:error, m4} = ev("range(10^9)")
      assert m4 =~ "range"
    end

    test "a memory bomb is killed by the VM and reported; the caller lives" do
      assert {:error, :memory} = Vapor.Hermetic.seal(fn -> Alembic.eval("[repeat(1, 1000000) for k in 1..200]", fuel: 10_000_000_000) end, heap_mb: 16)
      assert {:error, :timeout} = Vapor.Hermetic.seal(fn -> Process.sleep(5_000) end, timeout: 100)
    end

    test "identifiers never become atoms" do
      names = for i <- 1..2_000, do: "fresh_identifier_#{i}_#{System.unique_integer([:positive])}"
      src = Enum.map_join(names, "\n", &"#{&1} = 1")
      before = :erlang.system_info(:atom_count)
      {:ok, _} = Alembic.load(src)
      assert :erlang.system_info(:atom_count) - before < 50
    end

    test "compiled numeric expressions stop minting modules past the cap, with the same values" do
      cap = Vapor.Expr.jit_cap()
      try do
        Application.put_env(:vapor, :expr_jit_cap, 0)
        before = :erlang.system_info(:atom_count)
        vals = for k <- 1..100, do: (({:ok, t} = Vapor.Expr.parse("x * #{k} + sin(y) + (x > 1)")); Vapor.Expr.compile(t, ["x", "y"]).({2.0, 0.5}))
        assert :erlang.system_info(:atom_count) - before < 50
        assert_in_delta hd(vals), 2.0 + :math.sin(0.5) + 1.0, 1.0e-12
      after
        Application.put_env(:vapor, :expr_jit_cap, cap)
      end
    end

    test "literals accept data and refuse code" do
      assert {:ok, [1, 2.5, "a", {1, 2}, %{"k" => [3]}]} = Alembic.literal(~S|[1, 2.5, "a", (1, 2), {"k": [3]}]|)
      assert {:error, _} = Alembic.literal("f(1)")
      assert {:error, _} = Alembic.literal("[x for x in 1..3]")
      assert {:ok, 7} = Alembic.literal("3 + 4")
    end

    test "show and literal are inverse on random values" do
      :rand.seed(:exsss, {1, 2, 3})
      gen = fn gen, d ->
        case if(d > 3, do: :rand.uniform(4), else: :rand.uniform(7)) do
          1 -> :rand.uniform(10_000) - 5000
          2 -> Float.round(:rand.uniform() * 100 - 50, 6)
          3 -> Enum.random(["", "a\"b", "linha\nnova", "π"])
          4 -> Enum.random([true, false, nil])
          5 -> for _ <- 1..:rand.uniform(4), do: gen.(gen, d + 1)
          6 -> List.to_tuple(for _ <- 1..(1 + :rand.uniform(3)), do: gen.(gen, d + 1))
          7 -> Map.new(for k <- 1..:rand.uniform(3), do: {"k#{k}", gen.(gen, d + 1)})
        end
      end
      for _ <- 1..300 do
        v = gen.(gen, 0)
        assert {:ok, v2} = Alembic.literal(Alembic.show(v))
        assert Alembic.show(v2) == Alembic.show(v)
      end
    end
  end

  describe "the portable numeric tree" do
    test "parses the numeric subset and refuses the rest" do
      assert {:ok, ["+", 0.5, ["*", 0.2, ["f", "sin", [["v", "t"]]]]]} = Tree.parse("0.5 + 0.2*sin(t)", ["t"])
      assert {:error, m} = Tree.parse("len([1])", ["t"])
      assert m =~ "not available"
      assert {:error, _} = Tree.parse("q + 1", ["t"])
    end

    test "evaluation is total and matches the browser's (golden values of noise)" do
      {:ok, tr} = Tree.parse("if t > 1 then 1/0 else sqrt(-1) + fract(2.75)", ["t"])
      assert Tree.eval(tr, %{"t" => 2.0}) == 0.0
      assert Tree.eval(tr, %{"t" => 0.0}) == 0.75
      # the same numbers are asserted in test/js/scene_noise.mjs against the engine's JavaScript
      assert Tree.noise([1.0, 2.0]) == 0.16636425908654928
      assert Tree.noise([-1.25, 7.0]) == 0.03921110928058624
      assert Tree.noise([123.4567, -0.0015]) == 0.10643366817384958
    end
  end
end
