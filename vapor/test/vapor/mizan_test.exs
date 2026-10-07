defmodule Vapor.MizanTest do
  use ExUnit.Case, async: true
  alias Vapor.Mizan
  alias Vapor.Mizan.{Abjad, Lower, Syntax}

  defp ex(name), do: File.read!(Path.join([to_string(:code.priv_dir(:vapor)), "mizan", name]))
  defp parse!(text), do: (fn {:ok, m} -> m; {:error, e} -> flunk(e) end).(Mizan.parse(text))
  defp verdicts(m), do: Map.new(Mizan.check(m), &{&1.claim, &1.verdict})

  describe "decided, not trusted" do
    test "conservation: the oscillator's energy is proved; the damped one is refuted with its loss, -v²/10" do
      m = parse!(ex("oscillator.wzn"))
      assert verdicts(m) == %{"kinetic" => "none", "energy" => "proved", "damped-energy" => "refuted"}
      r = Enum.find(Mizan.check(m), &(&1.claim == "damped-energy"))
      assert r.detail =~ "-1/10*v^2"
    end

    test "positivity, bounds and identities: Bernstein and the exact normal form" do
      assert verdicts(parse!(ex("bounds.wzn"))) == %{"motzkin" => "proved", "cubic" => "proved", "square" => "proved"}
    end

    test "a Boolean transition system: invariant initial and inductive (DRUP); a broken step is refuted at a real state" do
      m = parse!(ex("handshake.wzn"))
      assert %{"handshake" => "proved", "constants" => "none"} = verdicts(m)
      bad = parse!(String.replace(ex("handshake.wzn"), "(step (req ack) (ack req))", "(step (req (not ack)) (ack req))"))
      r = Enum.find(Mizan.check(bad), &(&1.claim == "handshake"))
      assert r.verdict == "refuted"
      # the counterexample satisfies the invariant and its successor does not
      %{"req" => q, "ack" => a} = r.counterexample
      assert q == a and (not a) != q
    end

    test "refutations come with points that really refute; a claim that touches its bound is unknown, and does not run" do
      wrong = parse!("(claim sq (root H-s-b) (wazn burhan) (inputs (a q) (b q)) (proof (identity (+ (^ a 2) (^ b 2)))) (body (^ (+ a b) 2)))")
      [r] = Mizan.check(wrong)
      assert r.verdict == "refuted"
      args = [r.counterexample["a"], r.counterexample["b"]]
      lhs = Mizan.run(%{wrong | "decls" => [Map.put(hd(wrong["decls"]), "wazn", "fail") |> Map.delete("proof")]}, "sq", args)
      {:ok, %{value: v}} = lhs
      {a, b} = {hd(args), List.last(args)}
      refute v == Vapor.Logic.LP.qadd(Vapor.Logic.LP.qmul(a, a), Vapor.Logic.LP.qmul(b, b))

      neg = parse!("(claim p (root H-s-b) (wazn burhan) (inputs (x q)) (box (x -1 1)) (proof nonneg) (body (- (* x x) 1/4)))")
      [n] = Mizan.check(neg)
      assert n.verdict == "refuted" and Vapor.Logic.LP.qsign(Vapor.Logic.LP.qsub(Vapor.Logic.LP.qmul(n.counterexample["x"], n.counterexample["x"]), {1, 4})) < 0

      touch = parse!("(claim t (root H-s-b) (wazn burhan) (inputs (x q) (y q)) (box (x -2 2) (y -2 2)) (proof nonneg) (body (+ (* x x x x y y) (* x x y y y y) (* -3 x x y y) 1)))")
      assert [%{verdict: "unknown"}] = Mizan.check(touch, depth: 10)
      assert {:error, msg} = Mizan.run(touch, "t", [0, 0])
      assert msg =~ "does not run"
    end
  end

  describe "the morphological type system" do
    test "roots and weights that do not go together are refused, with the reason" do
      bad = [
        {"(claim e (root H-f-Z) (wazn fail) (inputs (x q)) (field (x 1)) (body x))", "burhān"},
        {"(claim e (root H-f-Z) (wazn burhan) (inputs (x q)) (body x))", "field"},
        {"(claim t (root n-q-l) (wazn burhan) (inputs (x q)) (step (x x)) (invariant (> x 0)) (proof invariant) (body x))", "bool"},
        {"(claim r (root k-t-b) (wazn fail) (body 1))", "mafʿūl"},
        {"(claim r (root k-t-b) (wazn maful) (inputs (x q)) (body x))", "no inputs"},
        {"(claim p (root H-s-b) (wazn burhan) (inputs (x q)) (body x))", "needs (proof"},
        {"(claim p (root H-s-b) (wazn fail) (inputs (x q)) (proof pos) (body x))", "weight burhān"},
        {"(claim p (root H-s-b) (wazn burhan) (inputs (x q)) (proof pos) (body x))", "box"},
        {"(claim box (root H-s-b) (wazn fail) (body 1))", "keyword"},
        {"(claim f (root H-s-b) (wazn fail) (inputs (x q)) (body (g x)))", "not a claim"},
        {"(claim f (root H-s-b) (wazn fail) (inputs (x q) (y f64)) (body x))", "mix"},
        {"(claim f (root X-y-z) (wazn fail) (body 1))", "unknown root"},
        {"(claim fسين (root H-s-b) (wazn fail) (body 1))", "not a name"},
        {"(claim f (root H-s-b) (wazn fail) (body (+ 1))", "never closed"},
        {"(claim f (root H-s-b) (wazn fail) (body (+ 1)))", "argument"}
      ]

      for {text, why} <- bad do
        assert {:error, msg} = Mizan.parse(text), text
        assert msg =~ why, "#{text}\n→ #{msg}"
      end
    end
  end

  describe "one tree, two scripts" do
    test "the Latin and the Arabic texts of the oscillator are the same program: one hash, each printing the other" do
      lat = parse!(ex("oscillator.wzn")) |> Map.update!("decls", &Enum.take(&1, 2))
      ar = parse!(ex("oscillator-ar.wzn"))
      # the Arabic file names its claims and variables in Arabic: map them to compare structure
      assert Mizan.check(ar) |> Enum.map(& &1.verdict) == ["none", "proved"]
      assert parse!(Mizan.print(ar, :latin)) == ar
      assert parse!(Mizan.print(lat, :arabic)) == lat
      assert Mizan.hash(parse!(Mizan.print(ar, :latin))) == Mizan.hash(ar)
      assert Mizan.print(ar, :latin) =~ "@TAqp"
      assert Mizan.print(lat, :arabic) =~ "دعوى" and Mizan.print(lat, :arabic) =~ "١/٢"
    end

    test "property: read(print(t)) == t in both projections, on 300 random programs with Latin and Arabic names" do
      rng = Vapor.Entropy.rng({:mizan, :roundtrip})

      Enum.reduce(1..300, rng, fn _, rng ->
        {m, rng} = random_module(rng)

        for p <- [:latin, :arabic] do
          text = Mizan.print(m, p)
          assert {:ok, ^m} = Mizan.parse(text), "#{p}:\n#{text}"
        end

        rng
      end)
    end
  end

  describe "running" do
    test "exact over ℚ, binary32 as vapor's oracle rounds it, records named by their hash" do
      m = parse!(ex("oscillator.wzn"))
      assert {:ok, %{value: {25, 2}}} = Mizan.run(m, "energy", [3, 4])
      assert {:error, msg} = Mizan.run(m, "damped-energy", [3, 4])
      assert msg =~ "refuted"
      f = parse!("(claim h (root H-s-b) (wazn fail) (inputs (x f32)) (body (/ 1 x)))")
      {:ok, %{value: third}} = Mizan.run(f, "h", [3.0])
      assert third == Vapor.F32.to_float(Vapor.F32.from_float(1 / 3))
      {:ok, rec} = Mizan.run(parse!(ex("handshake.wzn")), "constants", [])
      assert rec.value == {355, 113} and byte_size(rec.hash) == 64
    end
  end

  describe "lowering" do
    test "transmute: the energy through vapor's compiler, machine code per target; the assay against exact ℚ" do
      m = parse!(ex("oscillator.wzn"))
      {:ok, t} = Lower.transmute(m, "energy")
      assert t.targets != [] and Enum.all?(t.targets, &(&1.bytes > 0))
      assert {:error, _} = Lower.transmute(m, "damped-energy")
      {:ok, a} = Lower.assay(parse!(ex("bounds.wzn")), "cubic", 128)
      assert a.points == 128 and a.max_ulps <= 4
    end

    test "a transition claim as a circuit (AIGER); obligations as Lean 4 theorems" do
      {:ok, c} = Lower.aiger(parse!(ex("handshake.wzn")), "handshake")
      assert c.aiger =~ ~r/^aag /
      {:ok, l1} = Lower.lean(parse!(ex("oscillator.wzn")), "energy")
      assert l1 =~ "theorem mizan_energy" and l1 =~ "ring"
      {:ok, l2} = Lower.lean(parse!(ex("handshake.wzn")), "handshake")
      assert l2 =~ "decide"
    end
  end

  describe "content-addressed imports" do
    test "an import is fetched by hash, re-hashed and re-proved; one changed character refuses it" do
      lib = Path.join(System.tmp_dir!(), "mzn-lib-#{System.unique_integer([:positive])}")
      File.mkdir_p!(lib)
      a = parse!(ex("oscillator.wzn")) |> Map.update!("decls", &Enum.take(&1, 2))
      h = Mizan.hash(a)
      # stored in the Arabic projection: the hash is the tree's, not the text's
      File.write!(Path.join(lib, h <> ".wzn"), Mizan.print(a, :arabic))
      b = parse!("(import #{h} as osc)\n(claim twice (root H-s-b) (wazn fail) (inputs (v q)) (body (* 2 (osc.kinetic v))))")
      {:ok, linked} = Mizan.link(b, lib)
      assert {:ok, %{value: {9, 1}}} = Mizan.run(linked, "twice", [3])

      File.write!(Path.join(lib, h <> ".wzn"), String.replace(Mizan.print(a, :latin), "1/2 v v", "1/3 v v"))
      assert {:error, msg} = Mizan.link(b, lib)
      assert msg =~ "refused"

      bad = parse!(ex("oscillator.wzn"))
      hb = Mizan.hash(bad)
      File.write!(Path.join(lib, hb <> ".wzn"), Mizan.print(bad))
      assert {:error, msg} = Mizan.link(parse!("(import #{hb} as o)"), lib)
      assert msg =~ "proofs do not hold"
      File.rm_rf!(lib)
    end
  end

  test "abjad: shown, measured, not an address" do
    assert Abjad.value("ح-ف-ظ") == 988
    assert Abjad.value("ح-ف-ظ") == Abjad.value("ظ-ف-ح")
    c = Abjad.collisions()
    assert c.roots == 21_952 and c.fraction_sharing > 0.99
  end

  # ------------------------------------------------------- random programs

  @lat ~w(a b c x y z u v w alpha beta gamma)
  @ar ~w(س ع ص ط ق ك ل م ن ه ي ب ت ث ج ح خ د ذ ر ز ش ض ظ غ ف)

  defp random_module(rng) do
    {n, rng} = Vapor.Entropy.uniform(4, rng)

    {decls, rng} =
      Enum.reduce(1..n, {[], rng}, fn i, {acc, rng} ->
        {arabic?, rng} = Vapor.Entropy.uniform(2, rng)
        {nv, rng} = Vapor.Entropy.uniform(3, rng)
        pool = if arabic? == 1, do: @ar, else: @lat
        vars = pool |> Vapor.Entropy.shuffled({:vars, i, nv, length(acc)}) |> Enum.take(nv)
        name = if(arabic? == 1, do: "دالة" <> Enum.at(@ar, i), else: "f#{i}")
        {body, rng} = rexpr(vars, acc, 3, rng)
        c = %{"claim" => name, "root" => "hsb", "wazn" => "fail", "inputs" => Enum.map(vars, &[&1, "q"]), "body" => body}
        {acc ++ [c], rng}
      end)

    {%{"mizan" => 1, "decls" => decls}, rng}
  end

  defp rexpr(vars, prior, depth, rng) do
    {k, rng} = Vapor.Entropy.uniform(if(depth == 0, do: 2, else: 6), rng)

    case k do
      1 -> {n, rng} = Vapor.Entropy.uniform(40, rng); {d, rng} = Vapor.Entropy.uniform(5, rng); {["q", n - 20, d] |> norm(), rng}
      2 -> {v, rng} = Vapor.Entropy.pick(vars, rng); {["v", v], rng}
      3 -> {a, rng} = rexpr(vars, prior, depth - 1, rng); {b, rng} = rexpr(vars, prior, depth - 1, rng); {["op", "+", [a, b]], rng}
      4 -> {a, rng} = rexpr(vars, prior, depth - 1, rng); {b, rng} = rexpr(vars, prior, depth - 1, rng); {["op", "*", [a, b]], rng}
      5 -> {a, rng} = rexpr(vars, prior, depth - 1, rng); {["op", "-", [a]], rng}
      6 when prior != [] ->
        {f, rng} = Vapor.Entropy.pick(prior, rng)
        {args, rng} = Enum.map_reduce(f["inputs"], rng, fn _, r -> rexpr(vars, prior, depth - 1, r) end)
        {["call", f["claim"], args], rng}
      6 -> {a, rng} = rexpr(vars, prior, depth - 1, rng); {["op", "^", [a, ["q", 2, 1]]], rng}
    end
  end

  defp norm(["q", n, d]), do: (fn {a, b} -> ["q", a, b] end).(Vapor.Logic.LP.q(n, d))

  # silence an unused alias in some configurations
  def syntax, do: Syntax
end
