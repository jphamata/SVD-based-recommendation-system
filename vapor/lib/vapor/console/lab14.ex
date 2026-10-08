defmodule Vapor.Console.Lab14 do
  @moduledoc """
  The console's open workspace (docs/CONSOLE.md §0.14): Alembic, the
  Athanor's live sessions (steered by the person: propose, pin, ban, ask
  the model, measure), games, the Crucible, the Assay, scene operations
  and formalisation by a model. Every request is bounded; every
  computation runs sandboxed.
  """
  alias Vapor.{Alembic, Assay, Crucible, Mind}
  alias Vapor.Athanor.{Examples, Game, Session, Space, Touchstone}

  @max_text 400_000

  @doc "What the workspace offers: examples, domains, tools, whether a model is configured, the reference cards."
  def info(ctx) do
    %{athanor: Enum.map(Examples.all(), &Map.take(&1, [:id, :field, :title, :about, :text])),
      crucible: Crucible.kinds(), assay: Assay.tools(),
      mind: case mind(ctx) do nil -> nil; m -> m.name end,
      card: Alembic.card(), scene_card: Vapor.Scene.Ops.card(), scene_kinds: Vapor.Scene.Ops.kinds()}
  end

  @doc "The model the console uses: VAPOR_MIND, else the served model when there is one."
  def mind(ctx) do
    case Mind.from_env() do
      nil ->
        if ctx[:engine] && ctx[:tk] do
          Mind.wrap(Vapor.Agent.Backend.Local.new(engine: ctx.engine, tokenizer: ctx.tk, model_id: ctx[:model]), "served:" <> to_string(ctx[:model]))
        end
      m -> m
    end
  rescue
    _ -> nil
  end

  defp text(req, key \\ "text") do
    case req[key] do
      t when is_binary(t) and byte_size(t) <= @max_text -> {:ok, t}
      t when is_binary(t) -> {:error, "#{key}: at most #{div(@max_text, 1000)} kB"}
      _ -> {:error, "#{key}: a string"}
    end
  end

  @doc "What a text is: an Athanor problem, a game, a Crucible domain, a table for the Assay, or a plain program."
  def detect(t) do
    cond do
      Game.game?(t) -> %{kind: "game"}
      t =~ ~r/^\s*space\s*=/m -> %{kind: "athanor"}
      t =~ ~r/^\s*>\S+/m -> %{kind: "crucible", domain: "phylogeny"}
      t =~ ~r/^\s*H\s*=/m -> %{kind: "crucible", domain: "hamiltonian"}
      t =~ ~r/^\s*V\(\s*x\s*\)\s*=/m -> %{kind: "crucible", domain: "quantum"}
      t =~ ~r/(->|<->)/ and t =~ ~r/;\s*k/ -> %{kind: "crucible", domain: "reactions"}
      t =~ ~r/^\s*[\p{L}_]\w*'\s*=/mu -> %{kind: "crucible", domain: "laws"}
      t =~ ~r/^\s*(E[xyz]|B[xyz])\s*=/m -> %{kind: "crucible", domain: "fields"}
      t =~ ~r/^\s*(H|He)\s+-?[\d.]+\s+-?[\d.]+\s+-?[\d.]+\s*$/m -> %{kind: "crucible", domain: "molecule"}
      t =~ ~r/sequence\s*=\s*[HP]+/i -> %{kind: "crucible", domain: "fold"}
      t =~ ~r/^\s*N\s*=/m and t =~ ~r/^\s*s\s*=/m -> %{kind: "crucible", domain: "evolution"}
      t =~ ~r/^\s*target\s*=/m and t =~ ~r/,/ -> %{kind: "crucible", domain: "regress"}
      true -> %{kind: "alembic"}
    end
  end

  # ---------------------------------------------------------------- alembic

  def alembic(req) do
    with {:ok, src} <- text(req) do
      expr = req["expr"]
      run = fn ->
        case Alembic.load(src) do
          {:error, e} -> {:error, Map.put(e, :message, e.message)}
          {:ok, prog} ->
            if is_binary(expr) and String.trim(expr) != "" do
              case Alembic.eval(expr, program: prog) do
                {:ok, v} -> {:ok, %{value: Alembic.show(v), data: Alembic.to_data(v)}}
                {:error, e} -> {:error, e}
              end
            else
              consts = for d <- prog.defs, d.params == nil, do: %{name: d.name, value: Alembic.show(Alembic.const(prog, d.name)) |> String.slice(0, 2000)}
              fns = for d <- prog.defs, d.params != nil, do: %{name: d.name, params: d.params, line: d.line}
              {:ok, %{constants: consts, functions: fns}}
            end
        end
      end
      case Vapor.Hermetic.seal(run, heap_mb: 256, timeout: 20_000) do
        {:ok, r} -> r
        {:error, w} -> {:error, %{message: "stopped: #{inspect(w)}", line: 0, col: 0}}
      end
    end
  end

  # ---------------------------------------------------------------- athanor

  def athanor_start(req, ctx) do
    with {:ok, src} <- text(req) do
      opts =
        [budget: int(req["budget"], 1, 2_000_000), seed: int(req["seed"], 1, 1_000_000_000)]
        |> Enum.reject(fn {_, v} -> v == nil end)
        |> Kernel.++(if(m = mind(ctx), do: [mind: m], else: []))
      Session.start(src, opts)
    end
  end

  def athanor_action(id, req) do
    case req["action"] do
      "snapshot" -> Session.call(id, {:snapshot, int(req["since"], 0, 1_000_000_000) || 0})
      "propose" ->
        with {:ok, items} <- candidates(id, req["candidates"]), do: Session.call(id, {:propose, items})
      "pin" -> with({:ok, k} <- key(req), do: Session.call(id, {:pin, k}))
      "ban" -> with({:ok, k} <- key(req), do: Session.call(id, {:ban, k}))
      "stop" -> Session.call(id, :stop)
      "resume" -> Session.call(id, :resume)
      "extend" -> Session.call(id, {:extend, int(req["n"], 1, 1_000_000) || 1000})
      "measure" ->
        case {key(req), req["value"]} do
          {{:ok, k}, v} when is_number(v) -> Session.call(id, {:measure, k, v})
          _ -> {:error, "measure: candidate and a numeric value"}
        end
      "mind" -> Session.call(id, {:mind, int(req["n"], 1, 32) || 8}, 180_000)
      "certificate" -> Session.call(id, :certificate, 300_000)
      "close" -> Session.close(id); {:ok, %{closed: true}}
      other -> {:error, "action #{inspect(other)}: snapshot, propose, pin, ban, stop, resume, extend, measure, mind, certificate, close"}
    end
    |> case do
      :ok -> {:ok, %{ok: true}}
      other -> other
    end
  end

  defp key(%{"candidate" => k}) when is_binary(k) and byte_size(k) <= 100_000, do: {:ok, k}
  defp key(_), do: {:error, "candidate: the candidate's text"}

  # candidates typed by a person: literals (or expressions, for program spaces) — never code
  defp candidates(_id, items) when is_list(items) and length(items) <= 64 do
    parsed = Enum.map(items, fn
      t when is_binary(t) -> (case Alembic.literal(t) do {:ok, v} -> v; _ -> {:program_text, t} end)
      v -> Alembic.from_data(v)
    end)
    {:ok, Enum.map(parsed, fn {:program_text, t} -> t; v -> v end)}
  end

  defp candidates(_, _), do: {:error, "candidates: a list of up to 64 values (literal text or JSON)"}

  def verify(req) do
    with {:ok, src} <- text(req),
         %{} = cert <- req["certificate"] || {:error, "certificate: the JSON certificate"} do
      case Vapor.Hermetic.seal(fn -> Touchstone.verify(src, cert, full: req["full"] == true) end, heap_mb: 1024, timeout: 120_000) do
        {:ok, r} -> r
        {:error, w} -> {:error, "verification stopped: #{inspect(w)}"}
      end
    end
  end

  # ---------------------------------------------------------------- games

  def game(req) do
    with {:ok, src} <- text(req) do
      run = fn ->
        with {:ok, g} <- Game.load(src) do
          state = case req["state"] do nil -> g.init; s when is_binary(s) -> (case Alembic.literal(s) do {:ok, v} -> v; _ -> g.init end); v -> Alembic.from_data(v) end
          Game.safely(fn -> game_action(g, state, req) end)
        end
      end
      case Vapor.Hermetic.seal(run, heap_mb: 1024, timeout: 120_000) do
        {:ok, r} -> r
        {:error, w} -> {:error, "stopped: #{inspect(w)}"}
      end
    end
  end

  defp game_action(g, s, req) do
    view = fn st ->
      win = Game.winner(g, st)
      ms = if win == nil, do: Game.moves(g, st), else: []
      %{state: Alembic.show(st), board: Game.render(g, st), player: Alembic.show(Game.player(g, st)), winner: win && Alembic.show(win),
        over: win != nil or ms == [], moves: Enum.map(ms, &Alembic.show/1)}
    end

    case req["action"] do
      "view" -> {:ok, view.(s)}
      "play" ->
        with {:ok, m} <- Alembic.literal(to_string(req["move"])) do
          if Enum.any?(Game.moves(g, s), &Vapor.Alembic.Builtins.eq?(&1, m)), do: {:ok, view.(Game.play(g, s, m))}, else: {:error, "not a legal move"}
        end
      "reply" ->
        m =
          case Game.solve(g, state: s, max_positions: 150_000) do
            {:ok, r} -> (with {:ok, v} <- Alembic.literal(hd(r.best)), do: v)
            _ -> Game.mcts(g, s, sims: int(req["sims"], 10, 5000) || 600).move
          end
        {:ok, view.(Game.play(g, s, m)) |> Map.put(:reply, Alembic.show(m))}
      "solve" ->
        case Game.solve(g, state: s, max_positions: 400_000) do
          {:ok, r} -> {:ok, Map.merge(view.(s), %{solution: r})}
          e -> e
        end
      "search" -> {:ok, Map.put(view.(s), :search, Game.mcts(g, s, sims: int(req["sims"], 10, 20_000) || 800) |> Map.update!(:move, &Alembic.show/1))}
      "learn" ->
        with {:ok, l} <- Game.learn(g, games: int(req["games"], 2, 200) || 20, sims: 32) do
          {:ok, %{weights: l.weights, loss: l.loss, versus_plain_search: Game.match(g, {:learned, l.weights, 32}, {:mcts, 32}, 30)}}
        end
      _ -> {:error, "action: view, play, reply, solve, search, learn"}
    end
  end

  # ---------------------------------------------------------------- crucible, assay

  def crucible(req), do: with({:ok, t} <- text(req), do: Crucible.run(to_string(req["kind"]), t))
  def assay(req), do: with({:ok, t} <- text(req), do: Assay.run(to_string(req["tool"]), t, seed: int(req["seed"], 1, 1_000_000) || 1))

  # ---------------------------------------------------------------- mind, scenes

  def formalize(req, ctx) do
    with {:ok, words} <- text(req, "words") do
      case mind(ctx) do
        nil -> {:error, "no language model: start the server with VAPOR_MIND=anthropic:MODEL (or openai:MODEL[@URL]) or with --model — or write the problem in Alembic (the examples are starting points)"}
        m ->
          case Mind.formalize(m, words, String.to_existing_atom(req["kind"] || "auto")) do
            {:ok, r} -> {:ok, Map.merge(r, %{detected: detect(r.program), transcript: Mind.transcript(m)})}
            {:error, why, log} -> {:error, %{message: why, attempts: log}}
          end
      end
    end
  rescue
    ArgumentError -> {:error, "kind: auto, search or game"}
  end

  def scene_ops(req) do
    with {:ok, t} <- text(req) do
      {:ok, ops, probs} = Vapor.Scene.Ops.parse(t)
      {:ok, %{ops: ops, problems: probs}}
    end
  end

  def scene_mind(req, ctx) do
    with {:ok, words} <- text(req, "words") do
      scene = if is_map(req["scene"]), do: req["scene"], else: Vapor.Scene.Ops.blank()
      case mind(ctx) do
        nil ->
          r = Vapor.Scene.direct(words)
          {:ok, %{ops: r.ops, problems: Enum.map(r.unknown, &"not understood: #{&1}"), via: "vocabulary"}}
        m ->
          case Mind.direct(m, scene, words) do
            {:ok, r} -> {:ok, Map.put(r, :via, m.name)}
            e -> e
          end
      end
    end
  end

  def program_text(space, t), do: Space.parse_program(space, t)

  defp int(v, lo, hi) when is_integer(v), do: v |> max(lo) |> min(hi)
  defp int(_, _, _), do: nil
end
