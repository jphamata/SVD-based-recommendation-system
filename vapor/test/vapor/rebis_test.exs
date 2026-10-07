defmodule Vapor.RebisTest do
  use ExUnit.Case, async: true
  import Bitwise
  alias Vapor.Rebis
  alias Vapor.Rebis.{Field, GCM, Gen, Stabilizer}

  @full_adder """
  # a full adder
  input a b cin
  output s cout
  s = a ^ b ^ cin
  cout = maj(a, b, cin)
  """

  defp word(asg, prefix, n), do: Enum.reduce(0..(n - 1), 0, fn i, acc -> acc ||| Map.get(asg, "#{prefix}#{i}", 0) <<< i end)

  describe "circuits and their algebra" do
    test "a full adder: the truth table is arithmetic, the ANF is a+b+c and ab+ac+bc" do
      c = Rebis.parse!(@full_adder)

      for a <- 0..1, b <- 0..1, cin <- 0..1 do
        out = Rebis.eval(c, %{"a" => a, "b" => b, "cin" => cin})
        assert out["s"] + 2 * out["cout"] == a + b + cin
      end

      anf = Rebis.anf(c)
      assert anf["s"].degree == 1 and anf["s"].terms == 3
      assert anf["cout"].degree == 2 and anf["cout"].text == "a·b + a·cin + b·cin"
    end

    test "every output bit of the AES S-box has algebraic degree 7 (a known property, recomputed by Möbius)" do
      for k <- 0..7 do
        table = Enum.reduce(0..255, 0, fn j, acc -> acc ||| (Field.sbox(j) >>> k &&& 1) <<< j end)
        monos = table |> Rebis.mobius(8) |> Rebis.set_bits()
        assert monos |> Enum.map(&Rebis.popcount/1) |> Enum.max() == 7, "bit #{k}"
      end
    end

    test "the multiplier multiplies, for every pair of 4-bit numbers" do
      c = Rebis.parse!(Gen.multiplier(4))
      tables = Rebis.truth_tables(c)

      for a <- 0..15, b <- 0..15 do
        j = a ||| b <<< 4
        got = Enum.reduce(0..7, 0, fn k, acc -> acc ||| (tables["m#{k}"] >>> j &&& 1) <<< k end)
        assert got == a * b
      end
    end

    test "hash-consing: the same subexpression is one node" do
      c = Rebis.parse!("input a b\noutput y z\ny = a & b\nz = (a & b) ^ 1\n")
      assert Rebis.stats(c).by_kind[:and] == 1
    end
  end

  describe "equivalence" do
    test "two adders of different structure are equivalent: by truth table at 8 bits, by a checked DRUP proof at 16" do
      assert {:equivalent, %{method: :truth_table, patterns: 65_536}} = Rebis.equivalent(Rebis.parse!(Gen.ripple(8)), Rebis.parse!(Gen.kogge_stone(8)))
      assert {:equivalent, ev} = Rebis.equivalent(Rebis.parse!(Gen.ripple(16)), Rebis.parse!(Gen.kogge_stone(16)))
      assert ev.method == :sat and ev.checked_lemmas > 0
    end

    test "a 32-bit trojan with a one-in-four-billion trigger: random simulation cannot see it, the miter finds the trigger itself" do
      n = 32
      trigger = 0xDEAD_BEEF
      clean = Rebis.parse!(Gen.ripple(n))
      bad = Rebis.parse!(Gen.ripple(n, trojan: trigger))
      # method :sat means the 4096 random patterns simulated first did not hit the trigger
      assert {:different, %{method: :sat, counterexample: cex}} = Rebis.equivalent(clean, bad, patterns: 4096)
      # shrinking leaves exactly the trigger on a and nothing on b
      assert word(cex, "a", n) == trigger
      assert word(cex, "b", n) == 0
    end

    test "at 8 bits the truth table gives the counterexample with the fewest ones, exactly" do
      assert {:different, %{counterexample: cex, method: :truth_table}} = Rebis.equivalent(Rebis.parse!(Gen.ripple(8)), Rebis.parse!(Gen.ripple(8, trojan: 0x81)))
      assert word(cex, "a", 8) == 0x81 and word(cex, "b", 8) == 0
    end

    test "the two procedures agree on 120 random pairs of circuits (half rewritten by De Morgan, half mutated)" do
      :rand.seed(:exsss, {7, 7, 7})

      for t <- 1..120 do
        {text, n} = random_circuit(t)
        a = Rebis.parse!(text)
        b = if rem(t, 2) == 0, do: Rebis.parse!(de_morgan(text)), else: Rebis.parse!(mutate(text))
        by_table = Rebis.equivalent(a, b) |> elem(0)
        by_sat = Rebis.equivalent(a, b, truth_table: 0, patterns: 1) |> elem(0)
        assert by_table == by_sat, "pair #{t} (#{n} inputs): table #{by_table}, sat #{by_sat}"
        if rem(t, 2) == 0, do: assert(by_table == :equivalent)
      end
    end

    test "interfaces must match; outputs pair by name, or by position when the names differ" do
      a = Rebis.parse!("input a b\noutput y\ny = a & b\n")
      b = Rebis.parse!("input a c\noutput y\ny = a & c\n")
      assert {:error, msg} = Rebis.equivalent(a, b)
      assert msg =~ "inputs differ"
      c = Rebis.parse!("input b a\noutput q\nq = ~(~a | ~b)\n")
      assert {:equivalent, _} = Rebis.equivalent(a, c)
    end

    test "a SAT budget that runs out is 'unknown', never a guess" do
      a = Rebis.parse!(Gen.multiplier(7))
      b = Rebis.parse!(Gen.multiplier(7) |> swap_operands())
      assert {:unknown, _} = Rebis.equivalent(a, b, truth_table: 0, patterns: 64, conflicts: 5)
    end
  end

  describe "word-level identities by algebra over ℤ (the circuit's Gröbner basis)" do
    alias Vapor.Rebis.Ideal

    test "a 16-bit multiplier multiplies — proved by backward rewriting, where CDCL is exponential" do
      c = Rebis.parse!(Gen.multiplier(16))
      spec = Ideal.sub(Ideal.word("m", 32), Ideal.mul(Ideal.word("a", 16), Ideal.word("b", 16)))
      assert {:proved, %{peak_terms: peak}} = Ideal.prove(c, spec)
      assert peak < 5_000
    end

    test "a wrong partial product: refuted with a point where the identity fails, re-evaluated on the circuit" do
      c = Rebis.parse!(String.replace(Gen.multiplier(8), "pp3_4 = a4 & b3", "pp3_4 = a4 | b3"))
      spec = Ideal.sub(Ideal.word("m", 16), Ideal.mul(Ideal.word("a", 8), Ideal.word("b", 8)))
      assert {:refuted, %{counterexample: cex, value: v}} = Ideal.prove(c, spec)
      assert v != 0 and Ideal.evaluate(c, spec, cex) == v
      m = Rebis.eval(c, cex) |> then(fn o -> Enum.reduce(0..15, 0, &(&2 ||| o["m#{&1}"] <<< &1)) end)
      refute m == word(cex, "a", 8) * word(cex, "b", 8)
    end

    test "ripple adders are linear for the algebra; parallel-prefix adders blow up — and the bound says so instead of hanging" do
      spec = fn n -> Ideal.sub(Ideal.word("s", n) ++ [{1 <<< n, ["cout"]}], Ideal.word("a", n) ++ Ideal.word("b", n)) end
      assert {:proved, %{peak_terms: p}} = Ideal.prove(Rebis.parse!(Gen.ripple(64)), spec.(64))
      assert p < 1_000
      assert {:unknown, msg} = Ideal.prove(Rebis.parse!(Gen.kogge_stone(32)), spec.(32), max_terms: 50_000)
      assert msg =~ "terms"
    end

    test "names that are not ports are refused" do
      c = Rebis.parse!(@full_adder)
      assert {:error, _} = Ideal.prove(c, [{1, ["nope"]}])
    end
  end

  describe "AIGER" do
    test "a circuit written as AIGER and read back is the same function" do
      for text <- [@full_adder, Gen.ripple(6), Gen.kogge_stone(5), Gen.multiplier(3), "input s a b\noutput y\ny = mux(s, a, b) ^ 1\n"] do
        c = Rebis.parse!(text)
        assert {:ok, back} = Rebis.from_aiger(Rebis.to_aiger(c))
        assert back.inputs == c.inputs
        assert {:equivalent, _} = Rebis.equivalent(c, back)
      end
    end

    test "a hand-written half adder, symbols included; latches and garbage refused" do
      aag = "aag 7 2 0 2 3\n2\n4\n6\n12\n6 13 15\n12 2 4\n14 3 5\ni0 x\ni1 y\no0 s\no1 c\n"
      assert {:ok, c} = Rebis.from_aiger(aag)
      assert c.inputs == ["x", "y"]
      for x <- 0..1, y <- 0..1, do: assert(Rebis.eval(c, %{"x" => x, "y" => y}) == %{"s" => bxor(x, y), "c" => x &&& y})
      assert {:error, msg} = Rebis.from_aiger("aag 3 1 1 1 0\n2\n4 2\n4\n")
      assert msg =~ "latch"
      assert {:error, _} = Rebis.from_aiger("hello")
      assert {:error, _} = Rebis.from_aiger("aag 3 1 0 1 1\n2\n6\n6 2 9\n")
    end
  end

  describe "the netlist language refuses" do
    test "undefined wires, redefinitions, bad names, stray characters, outputs never defined" do
      assert {:error, m1} = Rebis.parse("input a\noutput y\ny = a & b\n")
      assert m1 =~ "before it is defined"
      assert {:error, m2} = Rebis.parse("input a\noutput y\ny = a\ny = ~a\n")
      assert m2 =~ "twice"
      assert {:error, _} = Rebis.parse("input 9a\noutput y\ny = 1\n")
      assert {:error, _} = Rebis.parse("input a\noutput y\ny = a $ a\n")
      assert {:error, m3} = Rebis.parse("input a\noutput y z\ny = a\n")
      assert m3 =~ "never defined"
      assert {:error, _} = Rebis.parse("input a\noutput y\ny = mux(a, a)\n")
      assert {:error, _} = Rebis.parse("input a\noutput y\ny = (a & a\n")
    end

    test "names never become atoms" do
      before = :erlang.system_info(:atom_count)
      for i <- 1..200, do: Rebis.parse("input zz_unique_#{i}_q\noutput y_#{i}_q\ny_#{i}_q = ~zz_unique_#{i}_q\n")
      assert :erlang.system_info(:atom_count) - before < 5
    end
  end

  describe "GF(2ⁿ), AES and GCM" do
    test "the field polynomials are irreducible; a reducible one is not" do
      assert Field.irreducible?(Field.aes_poly())
      assert Field.irreducible?(Field.gcm_poly())
      refute Field.irreducible?(0b10101)
    end

    test "the S-box from the field: FIPS-197's examples, and its inverse inverts it" do
      assert Field.sbox(0x00) == 0x63
      assert Field.sbox(0x53) == 0xED
      for x <- 0..255, do: assert(Field.inv_sbox(Field.sbox(x)) == x)
      assert 0..255 |> Enum.map(&Field.sbox/1) |> Enum.uniq() |> length() == 256
    end

    test "GCM multiplication two ways — the NIST algorithm and reflected carry-less product — agree" do
      :rand.seed(:exsss, {1, 2, 3})

      for _ <- 1..200 do
        <<x::128>> = :rand.bytes(16)
        <<y::128>> = :rand.bytes(16)
        assert Field.gcm_mul(x, y) == Field.gcm_mul_spec(x, y)
      end
    end

    test "AES-128/192/256 equal OpenSSL (through :crypto) on random keys and blocks" do
      :rand.seed(:exsss, {4, 5, 6})

      for size <- [16, 24, 32], _ <- 1..20 do
        key = :rand.bytes(size)
        block = :rand.bytes(16)
        cipher = %{16 => :aes_128_ecb, 24 => :aes_192_ecb, 32 => :aes_256_ecb}[size]
        assert GCM.encrypt_block(GCM.expand(key), block) == :crypto.crypto_one_time(cipher, key, block, true)
      end
    end

    test "AES-GCM equals OpenSSL: every length of message and AAD, 96-bit and other IVs; tampering is refused" do
      :rand.seed(:exsss, {8, 9, 10})

      for {pl, al, ivl} <- [{0, 0, 12}, {1, 0, 12}, {16, 20, 12}, {33, 7, 12}, {64, 64, 12}, {17, 3, 8}, {40, 0, 60}] do
        key = :rand.bytes(16)
        iv = :rand.bytes(ivl)
        pt = :rand.bytes(pl)
        aad = :rand.bytes(al)
        {ct, tag} = GCM.encrypt(key, iv, pt, aad)
        assert {ct, tag} == :crypto.crypto_one_time_aead(:aes_128_gcm, key, iv, pt, aad, true)
        assert {:ok, ^pt} = GCM.decrypt(key, iv, ct, aad, tag)
        <<t0, trest::binary>> = tag
        assert :error = GCM.decrypt(key, iv, ct, aad, <<bxor(t0, 1), trest::binary>>)
      end
    end

    test "NIST GCM test case 2" do
      {ct, tag} = GCM.encrypt(<<0::128>>, <<0::96>>, <<0::128>>, "")
      assert Base.encode16(ct, case: :lower) == "0388dace60b6a392f328c2b971b2fe78"
      assert Base.encode16(tag, case: :lower) == "ab6e47d42cec13bdf53a67b21257bddf"
    end
  end

  describe "stabilizer circuits" do
    test "a Bell pair: stabilized by +XX and +ZZ; the first measurement is random, the second follows it" do
      {:ok, %{state: st}} = Stabilizer.run("h 0\ncx 0 1\n", 2)
      assert Enum.sort(Stabilizer.stabilizers(st)) == ["+XX", "+ZZ"]

      for seed <- 1..20 do
        {:ok, r} = Stabilizer.run("h 0\ncx 0 1\nm 0\nm 1\n", 2, seed: seed)
        assert [a, a] = r.outcomes
        assert r.kinds == [:random, :deterministic]
      end

      outcomes = for seed <- 1..40, uniq: true, do: hd(elem(Stabilizer.run("h 0\nm 0\n", 1, seed: seed), 1).outcomes)
      assert Enum.sort(outcomes) == [0, 1]
    end

    test "random Clifford circuits on up to 5 qubits agree with a dense state-vector simulation" do
      :rand.seed(:exsss, {11, 12, 13})

      for _ <- 1..60 do
        n = Enum.random(1..5)
        gates = for _ <- 1..Enum.random(1..25), do: random_gate(n)
        text = Enum.join(gates, "\n")
        {:ok, %{state: st}} = Stabilizer.run(text, n)
        psi = Enum.reduce(gates, dense_zero(n), &dense_gate(&2, &1, n))

        for p <- Stabilizer.stabilizers(st) do
          assert_in_delta expectation(psi, p, n), 1.0, 1.0e-9, "#{text}\n#{p}"
        end

        # measuring each qubit: deterministic exactly when the dense probability is 0 or 1
        for q <- 0..(n - 1) do
          {out, _, kind} = Stabilizer.measure(st, q)
          p1 = prob_one(psi, q)

          case kind do
            :deterministic -> assert_in_delta p1, out * 1.0, 1.0e-9
            :random -> assert_in_delta p1, 0.5, 1.0e-9
          end
        end
      end
    end

    @tag timeout: 120_000
    test "a 400-qubit GHZ state: one random outcome, 399 that follow it — polynomial, not 2⁴⁰⁰" do
      n = 400
      text = "h 0\n" <> Enum.map_join(1..(n - 1), "\n", &"cx 0 #{&1}") <> "\n" <> Enum.map_join(0..(n - 1), "\n", &"m #{&1}")
      {t, {:ok, r}} = :timer.tc(fn -> Stabilizer.run(text, n, seed: 3) end)
      assert length(Enum.uniq(r.outcomes)) == 1
      assert hd(r.kinds) == :random and Enum.all?(tl(r.kinds), &(&1 == :deterministic))
      assert t < 60_000_000
    end

    test "non-Clifford gates are refused by name; qubits out of range too" do
      assert {:error, msg} = Stabilizer.run("h 0\nt 0\n", 1)
      assert msg =~ "not a Clifford gate"
      assert {:error, _} = Stabilizer.run("h 3\n", 2)
      assert {:error, _} = Stabilizer.run("cx 1 1\n", 2)
    end
  end

  # ================================================================ helpers

  defp random_circuit(t) do
    n = Enum.random(2..7)
    ins = for i <- 0..(n - 1), do: "i#{i}"
    {lines, wires} =
      Enum.reduce(1..Enum.random(3..14), {[], ins}, fn k, {ls, ws} ->
        a = Enum.random(ws)
        b = Enum.random(ws)
        op = Enum.random(["&", "|", "^"])
        neg = if :rand.uniform(3) == 1, do: "~", else: ""
        {ls ++ ["w#{k} = #{neg}(#{a} #{op} #{b})"], ws ++ ["w#{k}"]}
      end)

    outs = wires |> Enum.take(-2)
    _ = t
    {"input #{Enum.join(ins, " ")}\noutput #{Enum.join(outs, " ")}\n" <> Enum.join(lines, "\n") <> "\n", n}
  end

  # a | b → ~(~a & ~b), a & b → ~(~a | ~b): the same function, other gates
  defp de_morgan(text) do
    Regex.replace(~r/^(w\d+) = (~?)\((\w+) ([&|]) (\w+)\)$/m, text, fn _, w, neg, a, op, b ->
      dual = if op == "&", do: "|", else: "&"
      "#{w} = #{neg}~(~#{a} #{dual} ~#{b})"
    end)
  end

  # flip one operator: usually another function (sometimes not — the procedures must agree either way)
  defp mutate(text) do
    lines = String.split(text, "\n")
    idx = Enum.find_index(lines, &String.starts_with?(&1, "w"))
    line = Enum.at(lines, idx)
    swapped = cond do
      String.contains?(line, " & ") -> String.replace(line, " & ", " ^ ", global: false)
      String.contains?(line, " ^ ") -> String.replace(line, " ^ ", " | ", global: false)
      true -> String.replace(line, " | ", " & ", global: false)
    end
    lines |> List.replace_at(idx, swapped) |> Enum.join("\n")
  end

  defp swap_operands(text) do
    text |> then(&Regex.replace(~r/\ba(\d+)/, &1, "TMP\\1")) |> then(&Regex.replace(~r/\bb(\d+)/, &1, "a\\1")) |> then(&Regex.replace(~r/\bTMP(\d+)/, &1, "b\\1"))
  end

  defp random_gate(n) do
    case Enum.random(1..8) do
      k when k <= 2 -> "h #{Enum.random(0..(n - 1))}"
      3 -> "s #{Enum.random(0..(n - 1))}"
      4 -> Enum.random(["x", "y", "z"]) <> " #{Enum.random(0..(n - 1))}"
      _ when n == 1 -> "h 0"
      _ -> [a, b] = Enum.take_random(0..(n - 1), 2); Enum.random(["cx", "cz"]) <> " #{a} #{b}"
    end
  end

  # dense simulation: a map basis index → {re, im}; qubit q is bit q of the index
  defp dense_zero(n), do: Map.new(0..((1 <<< n) - 1), fn k -> {k, if(k == 0, do: {1.0, 0.0}, else: {0.0, 0.0})} end)

  defp dense_gate(psi, line, _n) do
    case String.split(line) do
      ["h", q] -> q = String.to_integer(q); s = :math.sqrt(0.5)
        Map.new(psi, fn {k, _} ->
          k0 = k &&& bnot(1 <<< q)
          k1 = k ||| 1 <<< q
          {a0, a1} = {psi[k0], psi[k1]}
          if (k >>> q &&& 1) == 0, do: {k, cadd(cmul(a0, {s, 0.0}), cmul(a1, {s, 0.0}))}, else: {k, cadd(cmul(a0, {s, 0.0}), cmul(a1, {-s, 0.0}))}
        end)
      ["s", q] -> q = String.to_integer(q); Map.new(psi, fn {k, a} -> {k, if((k >>> q &&& 1) == 1, do: cmul(a, {0.0, 1.0}), else: a)} end)
      ["z", q] -> q = String.to_integer(q); Map.new(psi, fn {k, a} -> {k, if((k >>> q &&& 1) == 1, do: cmul(a, {-1.0, 0.0}), else: a)} end)
      ["x", q] -> q = String.to_integer(q); Map.new(psi, fn {k, _} -> {k, psi[bxor(k, 1 <<< q)]} end)
      ["y", q] -> q = String.to_integer(q)
        # Y|0⟩ = i|1⟩, Y|1⟩ = −i|0⟩
        Map.new(psi, fn {k, _} -> src = psi[bxor(k, 1 <<< q)]; {k, if((k >>> q &&& 1) == 1, do: cmul(src, {0.0, 1.0}), else: cmul(src, {0.0, -1.0}))} end)
      ["cx", c, t] -> {c, t} = {String.to_integer(c), String.to_integer(t)}
        Map.new(psi, fn {k, _} -> {k, if((k >>> c &&& 1) == 1, do: psi[bxor(k, 1 <<< t)], else: psi[k])} end)
      ["cz", a, b] -> {a, b} = {String.to_integer(a), String.to_integer(b)}
        Map.new(psi, fn {k, v} -> {k, if((k >>> a &&& 1) == 1 and (k >>> b &&& 1) == 1, do: cmul(v, {-1.0, 0.0}), else: v)} end)
    end
  end

  # ⟨ψ|P|ψ⟩ for a signed Pauli string, qubit 0 first
  defp expectation(psi, "" <> p, n) do
    {sign, ops} = {if(String.starts_with?(p, "-"), do: -1.0, else: 1.0), String.slice(p, 1..-1) |> String.graphemes()}
    ppsi =
      Map.new(psi, fn {k, _} ->
        # (P ψ)(k) = Σ over the source basis state that P maps to k
        {src, phase} =
          ops |> Enum.with_index() |> Enum.reduce({k, {1.0, 0.0}}, fn {o, q}, {src, ph} ->
            bitk = k >>> q &&& 1
            case o do
              "I" -> {src, ph}
              "X" -> {bxor(src, 1 <<< q), ph}
              "Z" -> {src, cmul(ph, {if(bitk == 1, do: -1.0, else: 1.0), 0.0})}
              # Y = iXZ: (Y ψ)(k) = i·(−1)^(1−k_q)… : Y|0⟩ = i|1⟩, Y|1⟩ = −i|0⟩, so (Yψ)(k) = (k_q ? i : −i)·ψ(k ⊕ e_q)
              "Y" -> {bxor(src, 1 <<< q), cmul(ph, if(bitk == 1, do: {0.0, 1.0}, else: {0.0, -1.0}))}
            end
          end)
        {k, cmul(phase, psi[src])}
      end)

    {re, _} = Enum.reduce(0..((1 <<< n) - 1), {0.0, 0.0}, fn k, acc -> cadd(acc, cmul(conj(psi[k]), ppsi[k])) end)
    sign * re
  end

  defp prob_one(psi, q), do: Enum.reduce(psi, 0.0, fn {k, {re, im}}, acc -> if (k >>> q &&& 1) == 1, do: acc + re * re + im * im, else: acc end)
  defp cmul({a, b}, {c, d}), do: {a * c - b * d, a * d + b * c}
  defp cadd({a, b}, {c, d}), do: {a + c, b + d}
  defp conj({a, b}), do: {a, -b}
end
