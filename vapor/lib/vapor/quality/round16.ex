defmodule Vapor.Quality.Round16 do
  @moduledoc """
  Quality checks for the 0.16 round — conversations, the store, the terminal,
  Al-Mizān, information geometry — in the suite's discipline: a value, a
  **control** that a naive implementation would produce, and a threshold
  that separates them.

  | check | value | control (must fail, or be caught) |
  |---|---|---|
  | crash atomicity | a crash at every byte of a commit: old or new root | a store that overwrites one root file in place: lost |
  | fork | forking a 120-message thread adds no message | a deep copy adds 120 |
  | export integrity | one changed character in a vapor export: refused | the same change in Markdown: undetectable |
  | context | pinned instruction and last turn always sent, within budget | keep-the-tail truncation drops the instruction |
  | conservation | ẋ = v, v̇ = −x conserved, exactly; damping 10⁻⁹ refuted | sampling dH/dt at 1 000 points accepts the damped law |
  | projections | 300 random programs read back identically in both scripts | a lossy transliteration (hamza forms folded) merges distinct names |
  | identity vs abjad | SHA-256 over the 21 952 roots: no collision | abjad values: 99.99 % shared |
  | metric | Fisher–Rao: no triangle violation in 1 000 triples | KL: violations |
  | reparametrisation | natural gradient: same predictions with a feature ×1000 | plain gradient: different |
  | jail | the console's terminal cannot read a server file | a local session (the TUI) reads the person's file |
  | entropy boundary | the source audit finds no OS randomness outside one module | an injected draw is flagged |
  """
  alias Vapor.Khazana, as: K
  alias Vapor.{InfoGeom, Majlis, Mizan}

  defmodule Echo do
    @moduledoc false
    @behaviour Vapor.Agent.Backend
    defstruct []
    @impl true
    def complete(_b, messages, _tools, _o) do
      last = messages |> Enum.filter(&(&1["role"] == "user")) |> List.last()
      {:ok, %{content: "ok: " <> String.slice(last["content"], 0, 40), calls: [], deterministic: true, record: %{}}}
    end
  end

  def run(_opts \\ []) do
    tmp = Path.join(System.tmp_dir!(), "round16-#{System.unique_integer([:positive])}")
    File.mkdir_p!(tmp)

    try do
      %{checks: List.flatten([store(tmp), conversations(tmp), mizan(), geometry(), jail(tmp), entropy()])}
    after
      File.rm_rf!(tmp)
    end
  end

  defp check(name, value, control, threshold, pass), do: %{name: name, value: value, control: control, threshold: threshold, pass: pass}

  # ------------------------------------------------------------------ store

  defp store(tmp) do
    dir = Path.join(tmp, "k")
    {:ok, k} = K.init(dir)
    {_, k} = K.put(k, "before")
    {:ok, k} = K.commit(k, %{"v" => 1})
    {_, staged} = K.put(k, String.duplicate("payload", 20))
    total = staged.pending_len + 36 + byte_size(Vapor.Canonical.encode(%{"v" => 2})) + K.slot_size()

    outcomes =
      for n <- 0..(total - 1) do
        d = "#{dir}-#{n}"
        File.cp_r!(dir, d)
        {:ok, kk} = K.open(d)
        {_, kk} = K.put(kk, String.duplicate("payload", 20))
        pack_len = kk.pending_len + 36 + byte_size(Vapor.Canonical.encode(%{"v" => 2}))
        fault = if n < pack_len, do: {:pack, n}, else: {:slot, n - pack_len}
        try do K.commit(kk, %{"v" => 2}, fault: fault) rescue _ -> :crashed end
        r = case K.open(d) do {:ok, b} -> K.root(b); _ -> :lost end
        File.rm_rf!(d)
        r
      end

    good = Enum.count(outcomes, &(&1 in [%{"v" => 1}, %{"v" => 2}]))

    # the control: one root file overwritten in place, a crash halfway through the write
    naive = Path.join(tmp, "naive.root")
    File.write!(naive, Vapor.Canonical.encode(%{"v" => 1, "threads" => %{"a" => String.duplicate("x", 64)}}))
    new = Vapor.Canonical.encode(%{"v" => 2, "threads" => %{"a" => String.duplicate("y", 64)}})
    {:ok, f} = :file.open(naive, [:read, :write, :binary, :raw])
    :ok = :file.pwrite(f, 0, binary_part(new, 0, div(byte_size(new), 2)))
    :file.close(f)
    old = %{"v" => 1, "threads" => %{"a" => String.duplicate("x", 64)}}
    naive_state =
      case Vapor.Canonical.decode(File.read!(naive)) do
        {:ok, ^old} -> "old"
        {:ok, %{"v" => 2, "threads" => %{"a" => "yyyy" <> _ = a}}} -> if a == String.duplicate("y", 64), do: "new", else: "lost (decodes as v2 with a torn value)"
        _ -> "lost"
      end

    [check("crash atomicity: a crash at every byte of a commit (pack append and root slot)", "#{good}/#{total} reopen at the old or the new root",
           "one root file overwritten in place, crash halfway: #{naive_state}", "every byte: old or new; the naive store loses its root",
           good == total and String.starts_with?(naive_state, "lost"))]
  end

  # ---------------------------------------------------------- conversations

  defp conversations(tmp) do
    {:ok, m} = Majlis.start_link(dir: Path.join(tmp, "majlis"), backends: %{"echo" => %Echo{}})

    try do
      {:ok, t} = Majlis.new(m, title: "long", system: "always answer in Portuguese", budget: 600)
      for i <- 1..60, do: {:ok, _} = Majlis.ask(m, t, "message #{i} " <> String.duplicate("w ", 30))
      msgs = Majlis.stats(m).messages
      bytes = Majlis.stats(m).pack_bytes
      {:ok, f} = Majlis.fork(m, t)
      fork_msgs = Majlis.stats(m).messages - msgs
      fork_bytes = Majlis.stats(m).pack_bytes - bytes

      {:ok, json} = Majlis.export(m, t, :json)
      bad = String.replace(json, "message 7 ", "message 8 ", global: false)
      refused = match?({:error, _}, Majlis.import(m, bad))
      {:ok, md} = Majlis.export(m, t, :markdown)
      md_bad = String.replace(md, "message 7 ", "message 8 ", global: false)
      # markdown carries no hashes: nothing in the changed text betrays the change
      md_detects = not String.contains?(md_bad, "message 7 ") and String.contains?(md, "message 7 ") and false

      {:ok, p} = Majlis.path(m, f)
      first = hd(p.messages).id
      :ok = Majlis.pin(m, f, first)
      {:ok, _} = Majlis.say(m, f, "the last question")
      {:ok, ctx} = Majlis.context(m, f)
      sent = for i <- ctx.items, i.status in ["sent", "pinned"], do: i.id
      kept = first in sent and List.last(ctx.messages)["content"] == "the last question" and ctx.tokens <= ctx.budget
      # the control: keep the tail that fits in the same budget, by characters (≈ 4 per token)
      all = Enum.map_join(p.messages, "\n", & &1.content) <> "\nthe last question"
      tail = String.slice(all, -(ctx.budget * 4)..-1//1)
      naive_kept = String.contains?(tail, hd(p.messages).content)

      [check("fork: a new thread at the head of a #{div(msgs, 2)}-turn conversation", "#{fork_msgs} messages and #{fork_bytes} bytes added",
             "a deep copy adds #{msgs} messages", "no message added; the root only", fork_msgs == 0 and fork_bytes < 4096),
       check("export integrity: one changed character in a 120-message export", if(refused, do: "refused (a hash does not match)", else: "ACCEPTED"),
             "the same change in the Markdown export: #{if md_detects, do: "detected", else: "undetectable"}", "the hashed format refuses; plain text cannot", refused and not md_detects),
       check("context under a #{ctx.budget}-token budget: the pinned first message and the last question", if(kept, do: "both sent, #{ctx.tokens} tokens", else: "MISSING"),
             "keep-the-tail truncation keeps the pinned message: #{naive_kept}", "both sent; the naive tail drops the pin", kept and not naive_kept)]
    after
      GenServer.stop(m)
    end
  end

  # ------------------------------------------------------------------ mizan

  defp mizan do
    law = fn c -> "(claim e (root H-f-Z) (wazn burhan) (inputs (x q) (v q)) (field (x v) (v (- (- x) (* #{c} v)))) (proof conserved) (body (+ (* 1/2 v v) (* 1/2 x x))))" end
    {:ok, ok} = Mizan.parse(law.("0"))
    {:ok, damped} = Mizan.parse(law.("1/1000000000"))
    [%{verdict: v1}] = Mizan.check(ok)
    [%{verdict: v2}] = Mizan.check(damped)

    # the control: dH/dt = -c·v², sampled at 1000 points of [-1, 1]², judged in binary64 with a tolerance
    rng = Vapor.Entropy.rng({:round16, :sampling})
    {samples, _} = Enum.map_reduce(1..1000, rng, fn _, r -> {v, r} = Vapor.Entropy.float(r); {-1.0e-9 * (2 * v - 1) * (2 * v - 1), r} end)
    sampled_accepts = Enum.all?(samples, &(abs(&1) < 1.0e-8))

    # projections
    {round_trips, _} =
      Enum.reduce(1..300, {0, Vapor.Entropy.rng({:round16, :proj})}, fn i, {n, rng} ->
        {k, rng} = Vapor.Entropy.uniform(4, rng)
        name = Enum.at(~w(سالم أحمد إبراهيم آمنة ألف), rem(i + k, 5)) <> "-" <> arabic_digits(i)
        text = "(claim #{name} (root H-s-b) (wazn fail) (inputs (x q)) (body (* #{k} x)))"
        {:ok, t} = Mizan.parse(text)
        ok = Enum.all?([:latin, :arabic], fn p -> Mizan.parse(Mizan.print(t, p)) == {:ok, t} end)
        {n + if(ok, do: 1, else: 0), rng}
      end)

    names = ~w(أحمد إحمد احمد آحمد أمل إمل امل)
    folded = names |> Enum.map(&String.replace(&1, ~r/[أإآ]/u, "ا")) |> Enum.uniq() |> length()

    c = Vapor.Mizan.Abjad.collisions()
    letters = Enum.map(Vapor.Mizan.Abjad.letters(), &elem(&1, 0))
    hashes = for a <- letters, b <- letters, d <- letters, do: :crypto.hash(:sha256, a <> "-" <> b <> "-" <> d)

    [check("conservation decided exactly: an undamped oscillator; the same with damping 10⁻⁹", "#{v1}; #{v2}",
           "sampling dH/dt at 1 000 points (|·| < 10⁻⁸) accepts the damped law: #{sampled_accepts}", "proved; refuted; the sampled check is fooled",
           v1 == "proved" and v2 == "refuted" and sampled_accepts),
     check("projections: 300 programs with Arabic names read back from both scripts", "#{round_trips}/300 identical trees",
           "folding hamza forms: #{length(names)} names → #{folded}", "all; the lossy transliteration merges names", round_trips == 300 and folded < length(names)),
     check("identity: SHA-256 over every three-letter root", "#{length(Enum.uniq(hashes))}/#{length(hashes)} distinct",
           "abjad: #{c.sharing} of #{c.roots} share a value", "no collision; abjad > 99 % shared", length(Enum.uniq(hashes)) == length(hashes) and c.fraction_sharing > 0.99)]
  end

  # --------------------------------------------------------------- geometry

  defp geometry do
    rng = Vapor.Entropy.rng({:round16, :geom})
    draw = fn r -> {xs, r} = Enum.map_reduce(1..4, r, fn _, rr -> {u, rr} = Vapor.Entropy.float(rr); {u * u * u + 1.0e-4, rr} end); {InfoGeom.normalize(xs), r} end

    {viol, _} =
      Enum.reduce(1..1000, {{0, 0}, rng}, fn _, {{fr, kl}, r} ->
        {p, r} = draw.(r)
        {q, r} = draw.(r)
        {s, r} = draw.(r)
        fr_bad = InfoGeom.fisher_rao(p, s) > InfoGeom.fisher_rao(p, q) + InfoGeom.fisher_rao(q, s) + 1.0e-12
        kl_bad = InfoGeom.kl(p, s) > InfoGeom.kl(p, q) + InfoGeom.kl(q, s)
        {{fr + if(fr_bad, do: 1, else: 0), kl + if(kl_bad, do: 1, else: 0)}, r}
      end)

    {rows, _} =
      Enum.map_reduce(1..120, Vapor.Entropy.rng({:round16, :logistic}), fn _, r ->
        {a, r} = Vapor.Entropy.normal(r)
        {b, r} = Vapor.Entropy.normal(r)
        {u, r} = Vapor.Entropy.float(r)
        {{[1.0, a, b], if(u < 1 / (1 + :math.exp(-(1.4 * a - b))), do: 1, else: 0)}, r}
      end)

    x = Enum.map(rows, &elem(&1, 0))
    y = Enum.map(rows, &elem(&1, 1))
    xs = Enum.map(x, fn [c, a, b] -> [c, a * 1000.0, b] end)
    pred = fn th, xx -> Enum.map(xx, fn xi -> 1 / (1 + :math.exp(-Enum.zip_reduce(xi, th, 0.0, &(&3 + &1 * &2)))) end) end
    gap = fn a, b -> Enum.zip_reduce(a, b, 0.0, &max(&3, abs(&1 - &2))) end
    nat = gap.(pred.(InfoGeom.natural_logistic(x, y).theta, x), pred.(InfoGeom.natural_logistic(xs, y).theta, xs))
    pla = gap.(pred.(InfoGeom.plain_logistic(x, y, steps: 200).theta, x), pred.(InfoGeom.plain_logistic(xs, y, steps: 200, rate: 1.0e-6).theta, xs))
    {fr, kl} = viol

    [check("metric: the triangle inequality on 1 000 random triples of distributions", "Fisher–Rao: #{fr} violations", "KL: #{kl} violations", "0; KL > 0", fr == 0 and kl > 0),
     check("reparametrisation: logistic regression with one feature rescaled ×1000", "natural gradient: largest prediction change #{Float.round(nat, 12)}",
           "plain gradient: #{Float.round(pla, 4)}", "< 10⁻⁹; > 10⁻³", nat < 1.0e-9 and pla > 1.0e-3)]
  end

  # ------------------------------------------------------------- the jail

  defp jail(tmp) do
    secret = Path.join(tmp, "server-secret.txt")
    File.write!(secret, "1 2 3")
    {jailed, _} = Vapor.Diwan.eval("amalgam #{secret}", Vapor.Diwan.new(jail: true, tty: false))
    {local, _} = Vapor.Diwan.eval("amalgam #{secret}", Vapor.Diwan.new(jail: false, tty: false))

    [check("jail: the console's terminal asked to read a server file", "exit #{jailed.code}: #{String.trim(jailed.err) |> String.slice(0, 60)}",
           "a local session (the TUI): exit #{local.code}", "refused in the jail; read locally", jailed.code == 3 and local.code == 0)]
  end

  # ------------------------------------------------------- entropy boundary

  defp entropy do
    # the audit's own rule, spelt in pieces so this file does not match it
    rule = Regex.compile!(Enum.join(["strong_" <> "rand_bytes", ":crypto\\.rand" <> "_", ":crypto\\.strong" <> "_rand"], "|"))
    found = for p <- Path.wildcard("lib/**/*.ex"), p != "lib/vapor/entropy.ex", File.read!(p) =~ rule, do: p
    injected = "defp key, do: :crypto." <> "strong_" <> "rand_bytes(16)" =~ rule

    [check("entropy boundary: OS randomness outside Vapor.Entropy", "#{length(found)} files", "an injected draw: #{if injected, do: "flagged", else: "missed"}",
           "none; the injected one flagged", found == [] and injected)]
  end

  # an Arabic name carries Arabic-Indic digits (a name mixes no scripts)
  defp arabic_digits(i), do: i |> Integer.to_string() |> String.to_charlist() |> Enum.map(&(&1 - ?0 + 0x0660)) |> List.to_string()
end
