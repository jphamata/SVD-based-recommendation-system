defmodule Vapor.Quality.Round15 do
  @moduledoc """
  Quality checks for the 0.15 round (the Opus: Amalgam, Cupel, Rebis,
  Aludel, Tabula, and the merge's permutation alignment), in the suite's
  discipline: a value, a **control** that a broken, naive or lucky
  implementation would produce, and a threshold that separates them. Every
  piece of the round decides something, so each control asks the question
  that matters for a decision procedure: *would it have said the same
  thing if the claim were false?*

  | check | value | control (must fail, or be caught) |
  |---|---|---|
  | order | 2 048 f32 values, 30 orders and groupings: 1 result, = the exact sum rounded once | left-to-right f32 addition: many results |
  | detection | a bit flip in a sign, exponent or high mantissa bit: caught 100 % | the lowest mantissa bit: below the envelope, said so (< 100 %) |
  | no false alarm | 4 conforming summation orders × 6 scales: 0 accused | a forger who knows the seed passes; another seed catches the same forgery |
  | quarantine | a pool with one corrupting core: every answer correct, the core removed | — |
  | trojan | a 32-bit adder with a 2⁻³² trigger: found by the miter, the trigger itself | 4 096 random patterns: missed |
  | algebra | a 16-bit multiplier proved by backward rewriting | one wrong partial product: refuted at a re-simulated point |
  | AES-GCM | 64 random messages = OpenSSL | a flipped tag bit: refused 64/64 |
  | stabilizers | a 400-qubit GHZ state: 1 random outcome, 399 that follow it | 400 Hadamards: 400 random outcomes |
  | positivity | Motzkin + 1/1000 > 0 certified, witness replayed | Motzkin ≥ 0 (touches 0): exhausted, not faked; a tampered witness: refused |
  | barrier | damped oscillator: three conditions certified | the unstable field: refuted at an exact point |
  | antinomy | the sale contract: C1 × C6 with its scenario; C1 × C3 proved apart (DRUP) | with the overrides: no antinomy left |
  | alignment | a network fused with its shuffled copy, aligned: itself (logits equal) | fused without alignment: damaged |
  """
  import Bitwise
  alias Vapor.{Amalgam, Aludel, Cupel, F32, Lock, Merge, Rebis, Tabula, Tensor}
  alias Vapor.Cupel.Sentinel
  alias Vapor.Logic.LP
  alias Vapor.Rebis.{GCM, Gen, Ideal, Stabilizer}

  def run(_opts \\ []) do
    %{checks: List.flatten([amalgam(), cupel(), rebis(), aludel(), tabula(), align()])}
  end

  defp check(name, value, control, threshold, pass), do: %{name: name, value: value, control: control, threshold: threshold, pass: pass}

  # ============================================================ Amalgam

  defp amalgam do
    :rand.seed(:exsss, {15, 1, 1})
    # cancellation-heavy: big values and their near-negatives among small ones
    base = for _ <- 1..1024, do: (:rand.uniform(2) - 1) <<< 31 ||| Enum.random(60..200) <<< 23 ||| (:rand.uniform(1 <<< 23) - 1)
    xs = Enum.shuffle(base ++ Enum.map(base, &(bxor(&1, 0x8000_0000) + Enum.random(0..3))))

    exact = xs |> Enum.map(&F32.to_dyadic/1) |> Enum.reduce(&F32.dyadic_add/2) |> then(fn {m, e} -> F32.round_dyadic(m, e) end)

    amalgams =
      for t <- 1..30, uniq: true do
        :rand.seed(:exsss, {15, t, 2})
        parts = xs |> Enum.shuffle() |> Enum.chunk_every(Enum.random(1..300)) |> Enum.map(&Amalgam.sum(:f32, Enum.map(&1, fn x -> [x] end)))
        parts |> Enum.shuffle() |> Amalgam.merge_all() |> Amalgam.round_bits() |> hd()
      end

    naive =
      for t <- 1..30, uniq: true do
        :rand.seed(:exsss, {15, t, 3})
        xs |> Enum.shuffle() |> Enum.reduce(0, &F32.add(&2, &1))
      end

    [check("order: 2 048 f32 values summed in 30 orders and groupings", "#{length(amalgams)} result(s), #{if amalgams == [exact], do: "= exact sum rounded once", else: "≠ exact"}",
           "left-to-right f32: #{length(naive)} different results", "one result, equal to the exact rounding; the naive sum varies",
           amalgams == [exact] and length(naive) > 1)]
  end

  # ============================================================ Cupel

  defp rand(shape, seed, scale \\ 1.0) do
    t = Tensor.random(:f32, shape, seed)
    Tensor.from_list(:f32, shape, Enum.map(Tensor.to_floats(t), &(&1 * scale)))
  end

  defp cupel do
    w = rand([32, 64], 21)
    xs = for s <- 1..12, do: rand([4, 64], 300 + s)
    by_bit = Cupel.sensitivity(w, xs, seed: 5) |> Map.new(fn {b, hit, n} -> {b, hit / n} end)
    high = Enum.all?(19..31, &(by_bit[&1] == 1.0))

    # conforming substrates: any summation order inside the envelope must pass
    accused =
      for scale <- [1.0e-30, 1.0e-10, 1.0, 1.0e5, 1.0e10, 1.0e15], order <- [:forward, :reverse, :pairwise, :fma], reduce: 0 do
        acc ->
          w = rand([8, 48], 41, scale)
          x = rand([3, 48], 42, scale)
          case Cupel.assay(Cupel.probe(w, seed: 9), x, in_order(w, x, order)) do
            {:ok, _} -> acc
            _ -> acc + 1
          end
      end

    # a forger who knows r: change two outputs so that y·r is unchanged
    w = rand([16, 16], 41)
    x = rand([1, 16], 42)
    y = Cupel.oracle_linear(w, x)
    p = Cupel.probe(w, seed: 99)
    forged = forge(y, p)
    knows = match?({:ok, _}, Cupel.assay(p, x, forged))
    other = match?({:corrupt, _}, Cupel.assay(Cupel.probe(w, seed: 100), x, forged))

    [check("detection: a bit flip in one output of a 32×64 product (12 batches)", "bits 19–31: #{if high, do: "100 %", else: "missed"}",
           "bit 0: #{Float.round(by_bit[0] * 100, 1)} % (below the rounding envelope)", "sign, exponent and high mantissa all caught; the lowest bit honestly not",
           high and by_bit[0] < 1.0),
     check("no false alarm: 4 summation orders × 6 scales (10⁻³⁰ … 10¹⁵)", "#{accused} accused of 24", "forger with the seed: #{if knows, do: "passes", else: "caught"}; another seed: #{if other, do: "caught", else: "passes"}",
           "0 accused; the seed is the secret", accused == 0 and knows and other),
     quarantine()]
  end

  defp in_order(w, x, order) do
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

  # y₀ += δ·r₁, y₁ −= δ·r₀ leaves y·r unchanged (up to the rounding the envelope allows)
  defp forge(y, %Cupel.Probe{r: r}) do
    [r0, r1 | _] = r
    [y0, y1 | rest] = Tensor.to_floats(y)
    Tensor.from_list(:f32, y.shape, [y0 + r1 * 0.25, y1 - r0 * 0.25 | rest])
  end

  defp quarantine do
    w = rand([16, 16], 71)
    xs = for i <- 1..20, do: rand([2, 16], 700 + i)
    [a, b] = for _ <- 1..2, do: spawn(fn -> receive do: (:stop -> :ok) end)
    runner = fn wk, x -> y = Cupel.oracle_linear(w, x); if wk == b, do: {:ok, Cupel.flip(y, 1, 27)}, else: {:ok, y} end
    {:ok, s} = Sentinel.start_link(w: w, workers: [b, a], runner: runner)
    results = xs |> Task.async_stream(&Sentinel.linear(s, &1), max_concurrency: 8) |> Enum.map(fn {:ok, {:ok, y, _}} -> y end)
    right = Enum.zip(xs, results) |> Enum.count(fn {x, y} -> y.data == Cupel.oracle_linear(w, x).data end)
    removed = Sentinel.removed(s)
    GenServer.stop(s)
    Enum.each([a, b], &send(&1, :stop))

    check("quarantine: 20 concurrent callers, a pool with one core that flips an exponent bit", "#{right}/20 answers correct", "removed: #{inspect(removed)}",
          "every answer correct; the bad core quarantined", right == 20 and removed == %{"w0" => "quarantined"})
  end

  # ============================================================ Rebis

  defp word(asg, prefix, n), do: Enum.reduce(0..(n - 1), 0, fn i, acc -> acc ||| Map.get(asg, "#{prefix}#{i}", 0) <<< i end)

  defp rebis do
    trigger = 0xDEAD_BEEF
    clean = Rebis.parse!(Gen.ripple(32))
    bad = Rebis.parse!(Gen.ripple(32, trojan: trigger))
    # method :sat means the 4 096 random patterns simulated first did not hit the trigger
    {kind, ev} = Rebis.equivalent(clean, bad, patterns: 4096)
    found = kind == :different and ev.method == :sat and word(ev.counterexample, "a", 32) == trigger and word(ev.counterexample, "b", 32) == 0

    mul = Rebis.parse!(Gen.multiplier(16))
    spec = Ideal.sub(Ideal.word("m", 32), Ideal.mul(Ideal.word("a", 16), Ideal.word("b", 16)))
    proved = Ideal.prove(mul, spec)
    broken = Rebis.parse!(String.replace(Gen.multiplier(8), "pp3_4 = a4 & b3", "pp3_4 = a4 | b3"))
    spec8 = Ideal.sub(Ideal.word("m", 16), Ideal.mul(Ideal.word("a", 8), Ideal.word("b", 8)))
    refuted = Ideal.prove(broken, spec8)

    refuted_ok =
      case refuted do
        {:refuted, %{counterexample: cex, value: v}} -> v != 0 and Ideal.evaluate(broken, spec8, cex) == v
        _ -> false
      end

    :rand.seed(:exsss, {15, 4, 4})

    {same, refused} =
      for i <- 1..64, reduce: {0, 0} do
        {s, r} ->
          key = :crypto.strong_rand_bytes(Enum.random([16, 24, 32]))
          iv = :crypto.strong_rand_bytes(12)
          pt = :crypto.strong_rand_bytes(rem(i * 7, 97))
          aad = :crypto.strong_rand_bytes(rem(i * 5, 41))
          {ct, tag} = GCM.encrypt(key, iv, pt, aad)
          cipher = %{16 => :aes_128_gcm, 24 => :aes_192_gcm, 32 => :aes_256_gcm}[byte_size(key)]
          {ct2, tag2} = :crypto.crypto_one_time_aead(cipher, key, iv, pt, aad, true)
          <<t0, trest::binary>> = tag
          {s + if({ct, tag} == {ct2, tag2}, do: 1, else: 0), r + if(GCM.decrypt(key, iv, ct, aad, <<bxor(t0, 1), trest::binary>>) == :error, do: 1, else: 0)}
      end

    n = 400
    ghz = "h 0\n" <> Enum.map_join(1..(n - 1), "\n", &"cx 0 #{&1}") <> "\n" <> Enum.map_join(0..(n - 1), "\n", &"m #{&1}")
    {:ok, g} = Stabilizer.run(ghz, n, seed: 3)
    {:ok, h} = Stabilizer.run(Enum.map_join(0..(n - 1), "\n", &"h #{&1}\nm #{&1}"), n, seed: 3)
    g_random = Enum.count(g.kinds, &(&1 == :random))
    h_random = Enum.count(h.kinds, &(&1 == :random))
    ones = Enum.sum(h.outcomes)

    [check("trojan: a 32-bit adder that misbehaves only when a = 0xDEADBEEF, b = 0", if(found, do: "trigger found exactly (SAT, shrunk)", else: inspect(kind)),
           "4 096 random patterns: missed (the miter was needed)", "the trigger itself, after random simulation missed it", found),
     check("algebra: a 16-bit array multiplier, m = a·b, by backward rewriting over ℤ", proof_text(proved),
           "a wrong partial product: #{if refuted_ok, do: "refuted, the point re-simulated", else: inspect(elem(refuted, 0))}", "proved; the broken one refuted at a real point",
           match?({:proved, %{peak_terms: p}} when p < 5_000, proved) and refuted_ok),
     check("AES-GCM from GF(2⁸) and GF(2¹²⁸) arithmetic, 64 random keys, IVs, messages, AAD", "#{same}/64 = OpenSSL", "a flipped tag bit: #{refused}/64 refused",
           "all equal; all tampering refused", same == 64 and refused == 64),
     check("stabilizers: a 400-qubit GHZ state measured qubit by qubit", "#{g_random} random, #{n - g_random} determined, #{length(Enum.uniq(g.outcomes))} distinct value(s)",
           "400 Hadamards: #{h_random} random, #{ones} ones", "GHZ: 1 random and all equal; Hadamards: 400 random, ones in 160–240",
           g_random == 1 and length(Enum.uniq(g.outcomes)) == 1 and h_random == n and ones in 160..240)]
  end

  defp proof_text({:proved, s}), do: "proved: #{s.substitutions} substitutions, peak #{s.peak_terms} terms"
  defp proof_text(other), do: inspect(elem(other, 0))

  # ============================================================ Aludel

  defp aludel do
    vars = ["x", "y"]
    p! = fn text -> {:ok, p} = Aludel.parse(text, vars); p end
    {:ok, b} = Aludel.box([{-2, 2}, {-2, 2}])
    m = p!.("x^4*y^2 + x^2*y^4 - 3*x^2*y^2 + 1")
    eps = Aludel.add(m, Aludel.const(2, "1/1000"))
    cert = Aludel.decide(eps, b, sense: :pos)
    replayed = match?({:certified, %{witness: w}} when is_map(w), cert) and Aludel.check(eps, b, elem(cert, 1).witness, sense: :pos) == :ok
    exhausted = Aludel.decide(m, b, depth: 16)
    # the witness of one polynomial does not certify another
    tampered = if replayed, do: Aludel.check(Aludel.sub(eps, Aludel.const(2, "1/500")), b, elem(cert, 1).witness, sense: :pos), else: :ok

    sys = %{vars: vars, field: [p!.("y"), p!.("-x - y")], domain: [{-2, 2}, {-2, 2}], init: [{"-1/2", "1/2"}, {"-1/2", "1/2"}], unsafe: [{"3/2", 2}, {"3/2", 2}]}
    stable = Aludel.barrier(sys, p!.("x^2 + y^2 - 1"))
    unstable = Aludel.barrier(%{sys | field: [p!.("x"), p!.("y")]}, p!.("x^2 + y^2 - 1"))
    flow = Enum.find(unstable.conditions, &(&1.name == "flow")).result

    flow_ok =
      case flow do
        {:refuted, %{point: pt, value: v}} -> LP.qsign(LP.rat(v)) != 0 and length(pt) == 2
        _ -> false
      end

    [check("positivity: Motzkin's polynomial + 1/1000 > 0 on [−2, 2]² (non-negative, no sum of squares)",
           if(replayed, do: "certified (#{elem(cert, 1).cells} cells), witness replayed", else: inspect(elem(cert, 0))),
           "Motzkin ≥ 0: #{elem(exhausted, 0)}; the witness on another polynomial: #{if tampered == :ok, do: "ACCEPTED", else: "refused"}",
           "certified and replayed; the touching case not faked; witnesses do not transfer",
           replayed and elem(exhausted, 0) == :exhausted and tampered != :ok),
     check("barrier: ẋ = y, ẏ = −x − y with B = x² + y² − 1", "#{stable.verdict}: #{Enum.map_join(stable.conditions, ", ", & &1.name)}",
           "ẋ = x, ẏ = y: #{unstable.verdict} (flow condition #{elem(flow, 0)})", "three conditions certified; the unstable field refuted at an exact point",
           stable.verdict == :proved and length(stable.conditions) == 3 and unstable.verdict == :refuted and flow_ok)]
  end

  # ============================================================ Tabula

  @sale """
  parties buyer seller
  facts delivered late defective force_majeure
  exclusive pay withhold
  assume not (late and not delivered)
  C1: if delivered and not defective then buyer must pay seller
  C2: if late then buyer may withhold
  C3: if defective then buyer must not pay
  C4: if force_majeure then seller is exempt from deliver
  C5: seller must deliver buyer
  C6: if late and delivered then buyer must withhold
  """

  defp tabula do
    {:ok, t} = Tabula.parse(@sale)
    r = Tabula.analyze(t)
    anti = Enum.find(r.findings, &(&1.clauses == ["C1", "C6"] and &1.kind == :antinomy))
    scen = anti && anti.scenario["late"] and anti.scenario["delivered"] and not anti.scenario["defective"]
    apart = match?([%{clauses: ["C1", "C3"], drup: true}], r.checked_pairs)
    {:ok, t2} = Tabula.parse(@sale <> "C4 overrides C5\nC6 overrides C1\n")
    r2 = Tabula.analyze(t2)
    left = Enum.count(r2.findings, &(&1.kind == :antinomy))

    [check("antinomy: a sale contract of six clauses over four facts", "#{r.verdict}: C1 × C6 #{if scen, do: "when delivered ∧ late ∧ ¬defective", else: "MISSING"}; C1 × C3 #{if apart, do: "proved apart (DRUP)", else: "unproved"}",
           "with C4 > C5 and C6 > C1: #{left} antinomies, #{length(r2.resolved)} resolved", "the clash found with its scenario, the safe pair proved; the overrides settle it",
           r.verdict == :antinomies and scen == true and apart and left == 0 and length(r2.resolved) >= 2)]
  end

  # ============================================================ alignment

  defp align do
    map = %{"model_type" => "llama", "vocab_size" => 96, "hidden_size" => 64, "intermediate_size" => 96, "num_hidden_layers" => 2,
            "num_attention_heads" => 4, "num_key_value_heads" => 2, "max_position_embeddings" => 32, "rms_norm_eps" => 1.0e-5,
            "rope_theta" => 10_000.0, "hidden_act" => "silu"}
    {c, ws} = Vapor.Bench.Round08.tiny("llama", map)
    {:ok, spec, ws} = Lock.from_map(map, ws)
    a = %{spec: spec, weights: ws}
    b = %{a | weights: shuffled(ws)}
    ref = logits(c, a.weights)
    {:ok, naive} = Merge.merge([a, b], method: :linear)
    {:ok, fixed} = Merge.merge([a, b], method: :linear, align: true)
    d_fixed = max_diff(ref, logits(c, fixed.weights))
    d_naive = max_diff(ref, logits(c, naive.weights))

    [check("alignment: a network averaged with a copy whose MLP units are shuffled", "aligned: largest logit change #{d_fixed}",
           "unaligned: #{Float.round(d_naive, 4)}", "aligned = the network itself (0.0); unaligned > 10⁻³", d_fixed == 0.0 and d_naive > 1.0e-3)]
  end

  defp shuffled(ws) do
    :rand.seed(:exsss, {15, 5, 5})

    ws
    |> Enum.filter(fn {k, _} -> is_binary(k) and String.ends_with?(k, ".mlp.gate_proj.weight") end)
    |> Enum.reduce(ws, fn {k, g}, acc ->
      blk = String.replace_suffix(k, ".mlp.gate_proj.weight", "")
      [n, k] = g.shape
      perm = Enum.shuffle(0..(n - 1))
      rows = fn t -> rs = t |> Tensor.to_floats() |> Enum.chunk_every(k) |> List.to_tuple(); Tensor.from_list(:f32, t.shape, Enum.flat_map(perm, &elem(rs, &1))) end
      cols = fn t -> t |> Tensor.to_floats() |> Enum.chunk_every(n) |> Enum.flat_map(fn r -> rt = List.to_tuple(r); Enum.map(perm, &elem(rt, &1)) end) |> then(&Tensor.from_list(:f32, t.shape, &1)) end
      acc
      |> Map.put(blk <> ".mlp.gate_proj.weight", rows.(g))
      |> Map.put(blk <> ".mlp.up_proj.weight", rows.(acc[blk <> ".mlp.up_proj.weight"]))
      |> Map.put(blk <> ".mlp.down_proj.weight", cols.(acc[blk <> ".mlp.down_proj.weight"]))
    end)
  end

  defp logits(c, ws) do
    toks = [3, 50, 7, 81, 12]
    n = length(toks)
    {:ok, p} = Vapor.Model.Llama.program(c, ws, max_seq: 16)
    env = Map.merge(Vapor.Model.Llama.empty_caches(c, 16), %{tok: Tensor.from_list(:s32, [n], toks), pos: Tensor.from_list(:s32, [n], Enum.to_list(0..(n - 1)))})
    Vapor.Runtime.Oracle.eval_program(p, env).logits |> Tensor.to_floats()
  end

  defp max_diff(a, b), do: Enum.zip_reduce(a, b, 0.0, fn x, y, m -> max(m, abs(x - y)) end)
end
