defmodule Vapor.AmalgamTest do
  use ExUnit.Case, async: true
  import Bitwise
  alias Vapor.{Amalgam, F32, Tensor}
  alias Vapor.Verify.Dyadic, as: D

  # random f32 bit patterns spread over the whole exponent range, finite
  defp rand_f32(seed, n, opts \\ []) do
    :rand.seed(:exsss, {seed, 7, 11})
    emax = Keyword.get(opts, :emax, 254)

    for _ <- 1..n do
      e = Enum.random(0..emax)
      f = :rand.uniform(1 <<< 23) - 1
      s = :rand.uniform(2) - 1
      s <<< 31 ||| e <<< 23 ||| f
    end
  end

  defp rand_f64(seed, n, emin, emax) do
    :rand.seed(:exsss, {seed, 3, 5})

    for _ <- 1..n do
      e = Enum.random(emin..emax)
      f = :rand.uniform(1 <<< 52) - 1
      s = :rand.uniform(2) - 1
      s <<< 63 ||| e <<< 52 ||| f
    end
  end

  defp f64(bits) do
    <<x::float-64>> = <<bits::64>>
    x
  end

  defp bits64(x) do
    <<b::64>> = <<x::float-64>>
    b
  end

  describe "the one rounding" do
    test "f32: round_rational(m, s, 1) is the oracle's round_dyadic over the whole range" do
      :rand.seed(:exsss, {1, 2, 3})

      for _ <- 1..4000 do
        m = :rand.uniform(1 <<< 60) * Enum.random([1, -1]) >>> Enum.random(0..50)
        e = Enum.random(-220..150)
        s = -e
        assert Amalgam.round_rational(m, s, 1, :f32) == F32.round_dyadic(m, e), "m=#{m} e=#{e}"
      end
    end

    test "f64 and f32: the sum of two values is IEEE addition (exact sum, rounded once)" do
      for {a, b} <- Enum.zip(rand_f64(1, 3000, 0, 2000), rand_f64(2, 3000, 0, 2000)) do
        x = f64(a)
        y = f64(b)

        expected =
          try do
            bits64(x + y)
          rescue
            ArithmeticError -> :overflow
          end

        got = Amalgam.sum(:f64, [[a], [b]]) |> Amalgam.round() |> then(fn <<v::64-little>> -> v end)

        case expected do
          :overflow -> assert (got &&& 0x7FF0_0000_0000_0000) == 0x7FF0_0000_0000_0000
          e -> assert got == e, "#{x} + #{y}"
        end
      end

      for {a, b} <- Enum.zip(rand_f32(3, 3000), rand_f32(4, 3000)) do
        got = Amalgam.sum(:f32, [[a], [b]]) |> Amalgam.round_bits() |> hd()
        assert got == F32.add(a, b) or (not F32.finite?(got) and not F32.finite?(F32.add(a, b)))
      end
    end

    test "mean: the correctly rounded quotient, checked against both neighbours exactly" do
      :rand.seed(:exsss, {9, 9, 9})

      for _ <- 1..1500 do
        m = (:rand.uniform(1 <<< 70) - (1 <<< 69)) >>> Enum.random(0..60)
        d = Enum.random(1..1000)
        s = 149
        r = Amalgam.round_rational(m, s, d, :f32)
        if m != 0, do: assert_nearest(r, {m, -s}, d)
      end
    end

    test "every format rounds its own extreme values exactly and overflows to infinity" do
      for fmt <- Amalgam.formats() do
        a = Amalgam.new(fmt, 1)
        # the largest finite value of the format, twice: the true sum overflows
        maxbits = maxfinite(fmt)
        r = a |> Amalgam.add([maxbits]) |> Amalgam.add([maxbits]) |> Amalgam.round_bits() |> hd()
        assert r == inf(fmt), "#{fmt}"
        # max + (−max) is +0 (no intermediate overflow can happen)
        z = a |> Amalgam.add([maxbits]) |> Amalgam.add([sign(fmt) ||| maxbits]) |> Amalgam.round_bits() |> hd()
        assert z == 0
        # the smallest subnormal survives alone
        assert a |> Amalgam.add([1]) |> Amalgam.round_bits() |> hd() == 1
      end
    end
  end

  describe "order independence" do
    test "every permutation and every grouping of the same vectors gives the same cells and bits" do
      n = 64
      vecs = for s <- 1..24, do: rand_f32(100 + s, n, emax: 200)
      ref = Amalgam.sum(:f32, vecs)
      bits = Amalgam.round_bits(ref)
      :rand.seed(:exsss, {5, 5, 5})

      for trial <- 1..30 do
        shuffled = Enum.shuffle(vecs)
        # a random partition into groups, each summed on its own, merged in a random order
        groups = Enum.chunk_every(shuffled, Enum.random(1..7))
        parts = groups |> Enum.map(&Amalgam.sum(:f32, &1)) |> Enum.shuffle()
        merged = if rem(trial, 2) == 0, do: Amalgam.merge_all(parts), else: tree(parts)
        assert merged.cells == ref.cells
        assert merged.count == ref.count
        assert Amalgam.round_bits(merged) == bits
      end
    end

    test "the control: left-to-right f32 addition does depend on the order (the problem exists)" do
      vecs = for s <- 1..24, do: rand_f32(100 + s, 64, emax: 160)
      naive = fn vs -> Enum.reduce(vs, List.duplicate(0, 64), fn v, acc -> Enum.zip_with(acc, v, &F32.add/2) end) end
      :rand.seed(:exsss, {6, 6, 6})
      outcomes = for _ <- 1..10, uniq: true, do: naive.(Enum.shuffle(vecs))
      assert length(outcomes) > 1
    end

    test "f64 sums equal the exact sum rounded, on cancellation-heavy data" do
      # a + b − a with b tiny: naive left-to-right loses b
      big = bits64(1.0e300)
      tiny = bits64(1.0e-300)
      negbig = bits64(-1.0e300)
      got = Amalgam.sum(:f64, [[big], [tiny], [negbig]]) |> Amalgam.round() |> then(fn <<v::64-little>> -> v end)
      assert got == tiny
      assert bits64(1.0e300 + 1.0e-300 - 1.0e300) == 0
    end
  end

  describe "IEEE special values, order-free" do
    test "NaN, infinities and signed zeros follow IEEE 754 §6.3 in every order" do
      nz = 0x8000_0000
      pinf = 0x7F80_0000
      ninf = 0xFF80_0000
      nan = 0x7FC0_0000
      one = F32.from_float(1.0)

      cases = [
        {[[nz], [nz], [nz]], nz},
        {[[nz], [0], [nz]], 0},
        {[[one], [F32.neg(one)]], 0},
        {[[pinf], [one]], pinf},
        {[[ninf], [nz]], ninf},
        {[[pinf], [ninf]], :nan},
        {[[nan], [one]], :nan},
        {[[pinf], [pinf]], pinf}
      ]

      for {vs, want} <- cases, perm <- perms(vs) do
        got = Amalgam.sum(:f32, perm) |> Amalgam.round_bits() |> hd()
        if want == :nan, do: assert(F32.nan?(got)), else: assert(got == want, inspect(perm))
      end
    end

    test "the empty sum is +0 and an empty amalgam is the identity of merge" do
      e = Amalgam.new(:f32, 3)
      v = Amalgam.sum(:f32, [[0x8000_0000, 1, 0x7F80_0000]])
      assert Amalgam.merge(e, v).cells == v.cells
      assert Amalgam.merge(v, e).cells == v.cells
      assert Amalgam.round_bits(e) == [0, 0, 0]
    end
  end

  describe "dot products" do
    test "dot/3 equals the exact dyadic dot product rounded by the oracle" do
      for seed <- 1..40 do
        k = 1 + rem(seed * 37, 300)
        xs = rand_f32(seed, k, emax: 200)
        ys = rand_f32(seed + 1000, k, emax: 200)
        exact = Enum.zip(xs, ys) |> Enum.map(fn {x, y} -> D.mul(D.of_f32(x), D.of_f32(y)) end) |> D.sum()
        {m, e} = exact
        assert Amalgam.dot(:f32, xs, ys) == F32.round_dyadic(m, e)
      end
    end

    test "shards of a contraction, merged in any order, round to the unsharded dot (row-parallel made exact)" do
      k = 512
      xs = rand_f32(77, k, emax: 180)
      ys = rand_f32(78, k, emax: 180)
      whole = Amalgam.dot(:f32, xs, ys)

      for shards <- [2, 3, 5, 8, 16] do
        size = div(k + shards - 1, shards)
        parts = Enum.zip(Enum.chunk_every(xs, size), Enum.chunk_every(ys, size)) |> Enum.map(fn {a, b} -> Amalgam.partial_dot(:f32, a, b) end)
        assert parts |> Enum.reverse() |> Amalgam.merge_all() |> Amalgam.round_bits() |> hd() == whole
      end
    end

    test "products of specials: 0·∞ is NaN, signs of zero multiply" do
      a = Amalgam.new(:f32, 4, products: true)
      a = Amalgam.add_products(a, [0, 0x8000_0000, 0x8000_0000, 0x7F80_0000], [0x7F80_0000, F32.from_float(2.0), F32.from_float(-2.0), F32.from_float(-1.0)])
      [r0, r1, r2, r3] = Amalgam.round_bits(a)
      assert F32.nan?(r0)
      assert r1 == 0x8000_0000
      assert r2 == 0
      assert r3 == 0xFF80_0000
    end
  end

  describe "the wire" do
    test "canonical bytes round-trip and do not depend on how the cells were produced" do
      vecs = for s <- 1..6, do: rand_f32(s, 10)
      a = Amalgam.sum(:f32, vecs)
      b = Amalgam.merge(Amalgam.sum(:f32, Enum.take(vecs, 2)), Amalgam.sum(:f32, Enum.drop(vecs, 2)))
      assert Amalgam.to_wire(a) == Amalgam.to_wire(b)
      assert {:ok, a2} = Amalgam.from_wire(Amalgam.to_wire(a))
      assert a2.cells == a.cells and a2.count == 6
    end

    test "a hostile peer cannot inject a sum its count could not produce, nor garbage" do
      a = Amalgam.sum(:f32, [[F32.from_float(1.0)]])
      {:ok, m} = Vapor.Canonical.decode(Amalgam.to_wire(a))
      forged = Vapor.Canonical.encode(%{m | "cells" => [1 <<< 400]})
      assert {:error, msg} = Amalgam.from_wire(forged)
      assert msg =~ "exceeds"
      assert {:error, _} = Amalgam.from_wire(Vapor.Canonical.encode(%{m | "format" => "f128"}))
      assert {:error, _} = Amalgam.from_wire(Vapor.Canonical.encode(%{m | "cells" => [%{"x" => 1}]}))
      assert {:error, _} = Amalgam.from_wire(<<0xFF, 0x00, 0x13>>)
      assert {:error, _} = Amalgam.from_wire(Vapor.Canonical.encode(%{m | "scale" => 7}))
    end
  end

  describe "edges" do
    test "mismatched shapes and formats are refused, not silently broadcast" do
      a = Amalgam.new(:f32, 3)
      assert_raise ArgumentError, fn -> Amalgam.add(a, [1, 2]) end
      assert_raise ArgumentError, fn -> Amalgam.merge(a, Amalgam.new(:f32, 4)) end
      assert_raise ArgumentError, fn -> Amalgam.merge(a, Amalgam.new(:bf16, 3)) end
      assert_raise ArgumentError, fn -> Amalgam.add(a, Tensor.new(:bf16, [1, 3], <<0::48>>)) end
      assert_raise ArgumentError, fn -> Amalgam.new(:f8, 3) end
      assert_raise ArgumentError, fn -> Amalgam.mean(a, by: 0) end
    end

    test "bf16 and f16 tensors amalgamate and round in their own format" do
      t = Tensor.from_list(:f32, [1, 4], [1.0, -2.5, 3.0e-5, 65_504.0])
      h = Tensor.to_bf16(t)
      a = Amalgam.new(:bf16, 4) |> Amalgam.add(h) |> Amalgam.add(h)
      assert %Tensor{dtype: :bf16} = r = Amalgam.mean(a)
      assert r.data == h.data
    end
  end

  @tag timeout: 120_000
  test "stress: 2 000 vectors of 256 values, reduced by 4 concurrent tasks in random shards, agree bit for bit" do
    n = 256
    vecs = for s <- 1..2000, do: rand_f32(s, n, emax: 170)
    ref = Amalgam.sum(:f32, vecs) |> Amalgam.round_bits()

    for seed <- 1..3 do
      :rand.seed(:exsss, {seed, 0, 1})
      shards = vecs |> Enum.shuffle() |> Enum.chunk_every(Enum.random(37..700))

      got =
        shards
        |> Task.async_stream(&Amalgam.sum(:f32, &1), ordered: false, max_concurrency: 4)
        |> Enum.map(fn {:ok, a} -> Amalgam.from_wire(Amalgam.to_wire(a)) |> elem(1) end)
        |> Amalgam.merge_all()
        |> Amalgam.round_bits()

      assert got == ref
    end
  end

  @tag :python
  test "f64: equal to Python's math.fsum (an independent correctly rounded sum) on adversarial data" do
    vecs = for s <- 1..30, do: rand_f64(s, 1, 900, 1150) ++ rand_f64(s + 50, 1, 0, 2046) ++ rand_f64(s + 99, 1, 1000, 1100)
    columns = Enum.zip_with(vecs, & &1)

    script = """
    import sys, struct, math
    for line in sys.stdin:
        xs = [struct.unpack('<d', struct.pack('<Q', int(t)))[0] for t in line.split()]
        try:
            s = math.fsum(xs)
            print(struct.unpack('<Q', struct.pack('<d', s))[0])
        except OverflowError:
            print('overflow')
    """

    input = Enum.map_join(columns, "\n", &Enum.join(&1, " ")) <> "\n"
    want = Vapor.TestHelpers.py!(script, [], input) |> String.split()
    got = Amalgam.sum(:f64, vecs) |> Amalgam.round() |> then(fn bin -> for <<b::64-little <- bin>>, do: b end)

    for {w, g} <- Enum.zip(want, got) do
      if w == "overflow", do: assert((g &&& 0x7FF0_0000_0000_0000) == 0x7FF0_0000_0000_0000), else: assert(String.to_integer(w) == g)
    end
  end

  # ------------------------------------------------------------ helpers --

  defp tree([a]), do: a
  defp tree(xs), do: xs |> Enum.chunk_every(2) |> Enum.map(fn [a, b] -> Amalgam.merge(b, a); [a] -> a end) |> tree()

  defp perms([]), do: [[]]
  defp perms(xs), do: for(x <- xs, rest <- perms(xs -- [x]), do: [x | rest]) |> Enum.uniq()

  defp maxfinite(:f32), do: 0x7F7F_FFFF
  defp maxfinite(:bf16), do: 0x7F7F
  defp maxfinite(:f16), do: 0x7BFF
  defp maxfinite(:f64), do: 0x7FEF_FFFF_FFFF_FFFF
  defp inf(:f32), do: 0x7F80_0000
  defp inf(:bf16), do: 0x7F80
  defp inf(:f16), do: 0x7C00
  defp inf(:f64), do: 0x7FF0_0000_0000_0000
  defp sign(:f32), do: 0x8000_0000
  defp sign(:bf16), do: 0x8000
  defp sign(:f16), do: 0x8000
  defp sign(:f64), do: 0x8000_0000_0000_0000

  # r is the f32 nearest to x/d: no neighbour is strictly closer (ties: even)
  defp assert_nearest(r, x, d) do
    xr = D.mul(x, D.of_int(1))
    err = fn bits -> D.abs(D.sub(D.mul(D.of_f32(bits), D.of_int(d)), xr)) end
    up = if F32.negative?(r), do: r - 1, else: r + 1
    down = if F32.negative?(r), do: r + 1, else: max(r - 1, 0)

    for nb <- [up, down], F32.finite?(nb), nb != r do
      assert D.le?(err.(r), err.(nb)), "#{r} vs neighbour #{nb}"
    end
  end
end
