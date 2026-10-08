defmodule Vapor.Main.AthanorCli do
  @moduledoc false
  # vapor athanor / vapor search / vapor verify / vapor game
  import Vapor.Main
  alias Vapor.Alembic
  alias Vapor.Athanor
  alias Vapor.Athanor.{Game, Spec, Touchstone}

  @run_opts [budget: :integer, seed: :integer, seconds: :integer, control: :boolean, only: :string, set: :keep,
             interactive: :boolean, measure: :string, mind: :string, every: :integer]

  def run(["run" | argv]), do: search(argv)
  def run(["ask" | argv]), do: ask(argv)
  def run(["verify" | argv]), do: verify(argv)
  def run(["game" | argv]), do: game(argv)
  def run(["card" | _]), do: (out(Alembic.card()); 0)
  def run(_), do: (err("usage: vapor athanor run|ask|verify|game FILE …  (vapor help)"); 2)

  # ================================================================ run

  defp search(argv) do
    with {:ok, o, args} <- opts(argv, @run_opts),
         true <- not (o[:measure] != nil and Vapor.Main.jailed?()) || {:error, "--measure runs a program on the server, which a console session may not do (run it from your own terminal)"},
         {:ok, text} <- read_input(List.first(args)),
         {:ok, consts} <- consts(o),
         {:ok, mind} <- mind(o) do
      only = o[:only] && o[:only] |> String.split(",") |> Enum.map(&String.to_existing_atom(String.trim(&1)))
      base = [consts: consts, budget: o[:budget], seed: o[:seed], seconds: o[:seconds] || 300, control: Keyword.get(o, :control, true)] ++
               if(only, do: [only: only], else: []) ++ if(mind, do: [mind: mind], else: []) ++
               if(o[:measure], do: [measure: shell_measure(o[:measure]), measured: true], else: [])
      base = Enum.reject(base, fn {_, v} -> v == nil end)

      result =
        if o[:interactive] do
          interactive(text, base, o)
        else
          base = if tty?() and not json?(o) and not Vapor.Main.jailed?(), do: [{:on_round, progress()} | base], else: base
          Athanor.run(text, base)
        end

      case result do
        {:ok, cert} ->
          if tty?() and not Vapor.Main.jailed?(), do: IO.write(:stderr, "\r\e[K")
          if json?(o), do: emit_json(cert), else: out(render(cert))
          exit_code(cert)
        {:error, m} -> err("athanor: " <> m); 3
      end
    else
      :usage -> 2
      {:error, m} -> err("athanor: " <> m); 3
    end
  rescue
    ArgumentError -> err("athanor: --only takes strategy names: exhaustive, random, anneal, evolve, cmaes, bayes, mind"); 2
  end

  defp mind(o) do
    case o[:mind] do
      nil -> {:ok, nil}
      "env" -> {:ok, Vapor.Mind.from_env()}
      spec -> Vapor.Mind.parse(spec)
    end
  end

  defp exit_code(cert) do
    cond do
      cert.reason == "counterexample" -> 1
      cert.best == nil and cert.reason != "exhausted" -> 1
      cert.best == nil and cert.sense == "find" -> 1
      true -> 0
    end
  end

  defp progress do
    t = :counters.new(1, [])
    fn r ->
      now = System.monotonic_time(:millisecond)
      if now - :counters.get(t, 1) > 250 do
        :counters.put(t, 1, now)
        best = if r.best, do: "best #{Alembic.show(r.best.value)} (#{r.best.by})", else: "nothing valid yet"
        IO.write(:stderr, "\r\e[K" <> dim("athanor · #{r.evals}/#{r.spec.budget} evaluations · #{best}"))
      end
    end
  end

  @doc false
  def shell_measure(cmd) do
    fn x ->
      show = Alembic.show(x)
      json = Vapor.JSON.encode(Alembic.to_data(x))
      case Vapor.Main.Measure.run(cmd, [{"VAPOR_CANDIDATE", show}, {"VAPOR_CANDIDATE_JSON", json}]) do
        {:ok, out} ->
          case Regex.scan(~r/-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?/, out) |> List.last() do
            [n] -> {:ok, (case Integer.parse(n) do {i, ""} -> i; _ -> String.to_float(normalize(n)) end)}
            nil -> {:error, "the command printed no number"}
          end
        {:error, _} = e -> e
      end
    end
  end

  defp normalize(n), do: if(String.contains?(n, "."), do: n, else: String.replace(n, ~r/([eE])/, ".0\\1"))

  # ---------------------------------------------------------------- interactive

  defp interactive(text, base, o) do
    with {:ok, spec} <- Spec.parse(text, Keyword.take(base, [:consts, :budget, :seed, :measured])) do
      Vapor.Hermetic.cap_self(1024)
      r = Athanor.init(spec, base)
      err(dim("interactive: Enter continues · p LITERAL proposes · pin N · ban N · more N · mind · stop"))
      loop(r, o[:every] || 20)
    end
  end

  defp loop(r, every) do
    r = Athanor.step(r, every)
    show_top(r)
    if r.status != :running and r.queue == [] do
      err(dim("#{r.status}: #{r.reason}"))
      cmd(r, every, IO.gets(:stdio, "athanor (done; Enter ends)> "))
    else
      cmd(r, every, IO.gets(:stdio, "athanor> "))
    end
  end

  defp cmd(r, _every, :eof), do: {:ok, Athanor.certificate(r)}
  defp cmd(r, every, line) do
    line = String.trim(to_string(line))
    top = r.archive |> Enum.filter(&(&1.status == :ok)) |> Enum.take(9)
    case String.split(line, " ", parts: 2) do
      [""] -> if r.status == :running, do: loop(r, every), else: {:ok, Athanor.certificate(r)}
      ["stop"] -> {:ok, Athanor.certificate(r)}
      ["p", lit] ->
        case Alembic.literal(lit) do
          {:ok, v} -> loop(Athanor.propose(r, [v], :human), every)
          {:error, m} -> err(bad(m)); loop(r, 0)
        end
      ["pin", n] -> with_index(top, n, r, every, &Athanor.pin(r, &1.key))
      ["ban", n] -> with_index(top, n, r, every, &Athanor.ban(r, &1.key))
      ["more", n] -> loop(Athanor.extend(r, String.to_integer(n)), every)
      ["mind"] ->
        case r.opts[:mind] && Vapor.Mind.propose(r.opts[:mind], %{spec: r.spec, archive: r.archive, best: r.best, observed: r.observed, elites: r.elites}, 8) do
          {:ok, xs} -> loop(Athanor.propose(r, xs, :mind), every)
          _ -> err(bad("no model (use --mind anthropic:MODEL, openai:MODEL or script:FILE)")); loop(r, 0)
        end
      _ -> err(bad("?")); loop(r, 0)
    end
  rescue
    _ -> err(bad("?")); loop(r, 0)
  end

  defp with_index(top, n, r, every, f) do
    case Integer.parse(n) do
      {i, ""} when i >= 1 and i <= length(top) -> loop(f.(Enum.at(top, i - 1)), every)
      _ -> err(bad("no candidate #{n}")); loop(r, 0)
    end
  end

  defp show_top(r) do
    top = r.archive |> Enum.filter(&(&1.status == :ok)) |> Enum.take(9)
    err(bold("#{r.evals}/#{r.spec.budget} evaluations") <> dim(" · round #{r.round}"))
    top |> Enum.with_index(1) |> Enum.each(fn {e, i} ->
      pin = if MapSet.member?(r.pins, e.key), do: "★", else: " "
      err("  #{i}#{pin} #{Alembic.show(e.value) |> String.pad_leading(12)}  #{String.slice(e.key, 0, 90)}  #{dim(to_string(e.by))}")
    end)
  end

  # ================================================================ ask (measured)

  defp ask(argv) do
    with {:ok, o, args} <- opts(argv, [budget: :integer, seed: :integer, set: :keep, maximize: :boolean]),
         {:ok, text} <- read_input(List.first(args)),
         {:ok, consts} <- consts(o),
         {:ok, spec} <- Spec.parse(text, consts: consts, budget: o[:budget] || 30, seed: o[:seed], measured: true, sense: if(o[:maximize], do: :max, else: :min)) do
      err(dim("vapor proposes; you measure. Type the measured value, `skip` to drop a candidate, `stop` to finish."))
      r = Athanor.init(spec, control: false)
      ask_loop(Athanor.step(r, 1), o)
    else
      :usage -> 2
      {:error, m} -> err("athanor: " <> m); 3
    end
  end

  defp ask_loop(%{pending: []} = r, o) do
    if r.status == :running, do: ask_loop(Athanor.step(r, 1), o), else: finish_ask(r, o)
  end

  defp ask_loop(%{pending: [p | _]} = r, o) do
    shown = Alembic.show(p.x)
    case IO.gets(:stdio, "#{bold("measure")} #{shown} = ") do
      :eof -> finish_ask(r, o)
      line ->
        case String.trim(to_string(line)) do
          "stop" -> finish_ask(r, o)
          "skip" -> ask_loop(%{r | pending: tl(r.pending)}, o)
          v ->
            case Float.parse(v) do
              {x, _} -> {:ok, r} = Athanor.measure(r, p.key, x); ask_loop(r, o)
              :error -> err(bad("a number, skip or stop")); ask_loop(r, o)
            end
        end
    end
  end

  defp finish_ask(r, o) do
    cert = Athanor.certificate(r)
    if json?(o), do: emit_json(cert), else: out(render(cert))
    0
  end

  # ================================================================ verify

  def verify(argv) do
    with {:ok, o, args} <- opts(argv, [full: :boolean, replay: :boolean]),
         [prob, cert_path] <- pad(args),
         {:ok, text} <- read_input(prob),
         {:ok, cert_json} <- read_input(cert_path),
         {:ok, cert} <- decode(cert_json) do
      case Touchstone.verify(text, cert, full: o[:full], replay: o[:replay]) do
        {:ok, v} ->
          if json?(o) do
            emit_json(v)
          else
            out(if v.verified, do: good("✓ verified"), else: bad("✗ not verified"))
            Enum.each(v.checks, fn ch -> out("  #{if ch.ok, do: good("✓"), else: bad("✗")} #{bold(ch.check)}: #{ch.detail}") end)
          end
          if v.verified, do: 0, else: 1
        {:error, m} -> err("verify: " <> m); 3
      end
    else
      :usage -> 2
      :bad_args -> err("usage: vapor verify PROBLEM.nbq CERTIFICATE.json [--full] [--replay]   (either may be -)"); 2
      {:error, m} -> err("verify: " <> to_string(m)); 3
    end
  end

  defp pad([a, b]), do: [a, b]
  defp pad([a]), do: [a, "-"]
  defp pad(_), do: :bad_args

  defp decode(text) do
    case Vapor.JSON.decode(String.trim(text)) do
      {:ok, m} when is_map(m) -> {:ok, m}
      _ -> {:error, "the certificate is not a JSON object"}
    end
  end

  # ================================================================ render

  @doc false
  def render(cert) do
    lines = [
      bold("athanor") <> dim(" · #{cert.problem}"),
      verdict_line(cert),
      cert.best && best_block(cert.best),
      cert.control && dim("control  ") <> cert.control.says,
      cert.holdout && dim("holdout  ") <> cert.holdout.says,
      dim("effort   ") <> "#{cert.evaluations} evaluations (#{cert.invalid} invalid, #{cert.errors} errors, #{cert.duplicates} repeats) in #{cert.ms} ms; found by #{cert.found_by || "—"}",
      dim("strategy ") <> Enum.map_join(cert.strategies, " · ", fn {k, v} -> "#{k} #{v.evals}#{if v.improved > 0, do: "↑#{v.improved}", else: ""}" end),
      cert.error_samples != [] && dim("errors   ") <> warn(Enum.join(Enum.take(cert.error_samples, 3), " | ")),
      cert.notes != [] && dim("notes    ") <> Enum.join(cert.notes, "; "),
      dim("journal  #{cert.journal_root}"),
      dim("check    vapor verify PROBLEM.nbq CERT.json   (replay: #{cert.replay})")
    ]

    lines |> Enum.filter(&is_binary/1) |> Enum.join("\n")
  end

  defp verdict_line(cert) do
    v = cert.verdict
    cond do
      cert.reason in ["exhausted", "target", "found"] -> good("● " <> v)
      cert.reason == "counterexample" -> warn("● " <> v)
      cert.best == nil -> bad("● " <> v)
      true -> "● " <> v
    end
  end

  defp best_block(b) do
    shown = if b.shown, do: "\n" <> indent(b.shown), else: ""
    dim("best     ") <> bold(b.candidate) <> dim("  = #{Alembic.show(b.value)}") <> shown
  end

  defp indent(t), do: t |> String.split("\n") |> Enum.map_join("\n", &("         " <> &1))

  # ================================================================ games

  def game(argv) do
    with {:ok, o, args} <- opts(argv, [sims: :integer, games: :integer, seed: :integer, state: :string, as: :integer, max_positions: :integer]),
         [path | rest] <- (if args == [], do: [nil], else: args),
         {:ok, text} <- read_input(path),
         {:ok, g} <- Game.load(text),
         {:ok, state} <- state(o, g) do
      Game.safely(fn -> game_cmd(List.first(rest) || "solve", g, state, o) end)
      |> case do
        {:error, m} -> err("game: " <> m); 3
        code -> code
      end
    else
      :usage -> 2
      {:error, m} -> err("game: " <> m); 3
    end
  end

  defp state(o, g) do
    case o[:state] do
      nil -> {:ok, g.init}
      s -> Alembic.literal(s)
    end
  end

  defp game_cmd("solve", g, s, o) do
    case Game.solve(g, state: s, max_positions: o[:max_positions] || 400_000) do
      {:ok, r} ->
        if json?(o), do: emit_json(r), else: out(Game.render(g, s) <> "\n" <> bold(r.says) <> dim("\nbest moves: #{Enum.join(r.best, ", ")} · #{r.positions} positions"))
        0
      {:error, m} -> err("game: " <> m); 1
    end
  end

  defp game_cmd("search", g, s, o) do
    r = Game.mcts(g, s, sims: o[:sims] || 800, seed: o[:seed] || 1)
    if json?(o), do: emit_json(%{move: Alembic.show(r.move), stats: r.stats}), else: out(bold("move #{Alembic.show(r.move)}") <> "\n" <> Enum.map_join(Enum.take(r.stats, 6), "\n", &"  #{&1.move}: #{&1.visits} visits, value #{Float.round(&1.q, 3)}"))
    0
  end

  defp game_cmd("learn", g, _s, o) do
    sims = o[:sims] || 32
    case Game.learn(g, games: o[:games] || 30, sims: sims, seed: o[:seed] || 1) do
      {:ok, l} ->
        m = Game.match(g, {:learned, l.weights, sims}, {:mcts, sims}, 40, (o[:seed] || 1) + 99)
        res = %{weights: l.weights, games: l.games, loss: l.loss, versus_plain_search: m}
        if json?(o) do
          emit_json(res)
        else
          [lo, hi] = m.interval
          out(bold("learned value v(s) = tanh(w·features(s) + b)") <> " from #{l.games} self-play games")
          out("against plain search at #{sims} simulations (the control): score #{Float.round(m.score, 3)} [#{Float.round(lo, 3)}, #{Float.round(hi, 3)}] over #{m.games} games — " <>
              cond do lo > 0.5 -> good("better than the control"); hi < 0.5 -> bad("worse than the control"); true -> warn("not distinguishable from the control") end)
        end
        0
      {:error, m} -> err("game: " <> m); 3
    end
  end

  defp game_cmd("play", g, s, o) do
    human = o[:as] || 1
    play_loop(g, s, human, o)
  end

  defp game_cmd(other, _g, _s, _o), do: (err("game: unknown action #{other} — solve, search, learn or play"); 2)

  defp play_loop(g, s, human, o) do
    out(Game.render(g, s))
    case Game.winner(g, s) do
      nil ->
        ms = Game.moves(g, s)
        if ms == [] do
          out(bold("draw")); 0
        else
          p = Game.player(g, s)
          if p == human or (is_integer(human) and Vapor.Alembic.Builtins.eq?(p, human)) do
            out(dim("moves: " <> (ms |> Enum.with_index(1) |> Enum.map_join("  ", fn {m, i} -> "#{i}:#{Alembic.show(m)}" end))))
            case IO.gets(:stdio, "your move (number)> ") do
              :eof -> 0
              line ->
                case Integer.parse(String.trim(to_string(line))) do
                  {i, ""} when i >= 1 and i <= length(ms) -> play_loop(g, Game.play(g, s, Enum.at(ms, i - 1)), human, o)
                  _ -> err(bad("pick a number from the list")); play_loop(g, s, human, o)
                end
            end
          else
            m = case Game.solve(g, state: s, max_positions: 200_000) do
              {:ok, r} -> {:ok, v} = Alembic.literal(hd(r.best)); v
              _ -> Game.mcts(g, s, sims: o[:sims] || 600).move
            end
            out(dim("vapor plays #{Alembic.show(m)}"))
            play_loop(g, Game.play(g, s, m), human, o)
          end
        end
      0 -> out(bold("draw")); 0
      w -> out(bold("winner: #{Alembic.show(w)}")); 0
    end
  end
end
