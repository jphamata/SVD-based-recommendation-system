defmodule Vapor.Console.Lab12 do
  @moduledoc """
  The console's laboratories for the 0.12 round (docs/CONSOLE.md): the
  workbench (any text problem: formulas with units, ODEs, PDEs,
  algebra; ensembles on the native worker), engineering (circuits, power
  flow, frames, plane FEM, pipe networks, reaction networks, flash,
  distillation), the logic desk, the boards (chess, shogi, Go, m,n,k,
  poker), proteins, the reference renderer and the living scene's exact
  frames as a GIF. Each a function from a small JSON request to a
  JSON-ready result, each bounded before it runs (a request is data from
  outside).
  """
  alias Vapor.Bio.{Align, Coevolution, Structure}
  alias Vapor.Engineering.{Circuit, FEM, Pipes, Power, Process}
  alias Vapor.Play.{Chess, Go, MNK, Poker, Shogi}
  alias Vapor.{Logic, Render, Solve}

  @max_text 200_000

  defp text(nil), do: {:error, "text: the problem, as text"}
  defp text(t) when is_binary(t) and byte_size(t) <= @max_text, do: {:ok, t}
  defp text(_), do: {:error, "text: at most 200 kB"}

  # every solver runs under a deadline: a request cannot hold the server
  defp timed(fun, ms \\ 120_000) do
    task = Task.async(fn -> try do fun.() rescue e -> {:error, Exception.message(e)} catch kind, why -> {:error, "#{kind}: #{inspect(why)}"} end end)
    case Task.yield(task, ms) || Task.shutdown(task, :brutal_kill) do
      {:ok, r} -> r
      nil -> {:error, "the computation took longer than #{div(ms, 1000)} s and was stopped (make the problem smaller)"}
    end
  end

  # ============================================================ workbench

  def solve(req) do
    with {:ok, t} <- text(req["text"]) do
      t0 = System.monotonic_time(:millisecond)
      case timed(fn -> if req["ensemble"], do: Solve.Ensemble.run(t), else: Solve.run(t) end) do
        {:ok, r} -> {:ok, r |> Map.drop([:pieces]) |> Map.put(:ms, System.monotonic_time(:millisecond) - t0) |> slim()}
        {:error, w} -> {:error, w}
      end
    end
  end

  # results are JSON: tuples become lists, maps with tuple keys become lists of pairs
  @doc false
  def slim(v) when is_map(v) and not is_struct(v) do
    if Enum.any?(Map.keys(v), &is_tuple/1), do: Enum.map(v, fn {k, x} -> [slim(k), slim(x)] end), else: Map.new(v, fn {k, x} -> {k, slim(x)} end)
  end
  def slim(v) when is_tuple(v), do: v |> Tuple.to_list() |> Enum.map(&slim/1)
  def slim(v) when is_list(v), do: Enum.map(v, &slim/1)
  def slim(v) when is_function(v), do: nil
  def slim(v), do: v

  # =========================================================== engineering

  def engineering(req) do
    with {:ok, t} <- text(req["text"]) do
      r = timed(fn ->
        case req["kind"] do
          "circuit" -> Circuit.run(t) |> ok_map(&Map.drop(&1, []))
          "power" -> Power.run(t, method: if(req["method"] == "gauss_seidel", do: :gauss_seidel, else: :newton))
          "structure" -> Vapor.Engineering.Structure.run(t)
          "fem" -> FEM.run(t)
          "pipes" -> Pipes.run(t)
          "reactions" -> Process.reactions(t)
          "flash" -> Process.flash(t)
          "distill" -> Process.distill(t)
          k -> {:error, "kind: circuit, power, structure, fem, pipes, reactions, flash or distill (got #{inspect(k)})"}
        end
      end)
      case r do
        {:ok, v} -> {:ok, v |> strip() |> slim() |> Map.put(:kind, req["kind"])}
        e -> e
      end
    end
  end

  defp ok_map({:ok, v}, f), do: {:ok, f.(v)}
  defp ok_map(e, _), do: e

  # drop solver internals that are not data (x vectors of the op point stay; matrices none)
  defp strip(%{op: op} = r), do: %{r | op: Map.drop(op, [:x])}
  defp strip(r), do: r

  # ================================================================= logic

  def logic(req), do: with({:ok, t} <- text(req["text"]), do: (case timed(fn -> Logic.run(t) end) do {:ok, r} -> {:ok, slim(r)}; e -> e end))

  # ================================================================ boards

  def chess(req) do
    with {:ok, p} <- Chess.from_fen(req["fen"] || "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") do
      timed(fn ->
        case req["action"] || "state" do
          "state" -> {:ok, chess_state(p)}
          "move" ->
            with {:ok, m} <- Chess.parse_move(p, req["move"] || "") do
              q = Chess.make(p, m)
              {:ok, Map.merge(chess_state(q), %{played: Chess.san(p, m), uci: Chess.uci(m)})}
            end
          "engine" ->
            depth = clamp(req["depth"], 1, 6, 4)
            r = Chess.search(p, depth: depth, nodes: clamp(req["nodes"], 1000, 400_000, 120_000))
            if r && r.best, do: {:ok, Map.merge(chess_state(Chess.make(p, r.best)), %{played: r.san, uci: Chess.uci(r.best), score: r.score, pv: r.pv, nodes: r.nodes, depth: r.depth})},
              else: {:ok, Map.put(chess_state(p), :note, "no legal move")}
          "analyse" ->
            r = Chess.search(p, depth: clamp(req["depth"], 1, 6, 4), nodes: 200_000)
            {:ok, %{score: r && r.score, pv: r && r.pv, best: r && r.san, nodes: r && r.nodes, depth: r && r.depth}}
          "mate" ->
            n = clamp(req["n"], 1, 3, 2)
            case Chess.prove_mate(p, n) do
              {:mate, proof} -> {:ok, %{mate: true, proof: proof, check: inspect(Chess.verify_mate(p, proof, n))}}
              :no_mate -> {:ok, %{mate: false, n: n}}
            end
          "perft" ->
            d = clamp(req["depth"], 1, 4, 3)
            {:ok, %{depth: d, nodes: Chess.perft_parallel(p, d), divide: if(d <= 3, do: Chess.divide(p, d))}}
          a -> {:error, "action: state, move, engine, analyse, mate or perft (got #{a})"}
        end
      end)
    end
  end

  defp chess_state(p) do
    board = for r <- 7..0//-1, do: (for f <- 0..7, do: elem(p.board, r * 8 + f))
    moves = Chess.moves(p)
    %{fen: Chess.to_fen(p), board: board, turn: if(p.turn == 1, do: "w", else: "b"), status: Chess.status(p), check: Chess.in_check?(p),
      legal: Enum.map(moves, &%{uci: Chess.uci(&1), san: Chess.san(p, &1)}), eval: Chess.evaluate(p) * p.turn}
  end

  def shogi(req) do
    with {:ok, p} <- Shogi.from_sfen(req["sfen"] || "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1") do
      timed(fn ->
        case req["action"] || "state" do
          "state" -> {:ok, shogi_state(p)}
          "move" -> with({:ok, m} <- Shogi.parse_move(p, req["move"] || ""), do: {:ok, Map.put(shogi_state(Shogi.make(p, m)), :played, Shogi.usi(m))})
          "engine" ->
            r = Shogi.search(p, depth: clamp(req["depth"], 1, 3, 2), nodes: 40_000)
            case r.best do
              nil -> {:ok, Map.put(shogi_state(p), :note, "no legal move")}
              b -> (({:ok, m} = Shogi.parse_move(p, b)); {:ok, Map.merge(shogi_state(Shogi.make(p, m)), %{played: b, score: r.score, nodes: r.nodes})})
            end
          "perft" -> (d = clamp(req["depth"], 1, 3, 2); {:ok, %{depth: d, nodes: Shogi.perft_parallel(p, d)}})
          a -> {:error, "action: state, move, engine or perft (got #{a})"}
        end
      end)
    end
  end

  defp shogi_state(p) do
    %{sfen: Shogi.to_sfen(p), board: for(y <- 0..8, do: for(x <- 0..8, do: elem(p.board, y * 9 + x))), turn: if(p.turn == 1, do: "b", else: "w"),
      hands: %{"b" => p.hands[1], "w" => p.hands[-1]} |> Map.new(fn {k, h} -> {k, Map.new(h, fn {kk, n} -> {to_string(kk), n} end)} end),
      legal: p |> Shogi.moves() |> Enum.map(&Shogi.usi/1), status: Shogi.status(p), check: Shogi.in_check?(p)}
  end

  def go(req) do
    n = clamp(req["size"], 2, 13, 9)
    timed(fn ->
      s0 = Go.new(n, komi: (req["komi"] || 6.5) * 1.0)
      s = Enum.reduce_while(req["moves"] || [], {:ok, s0}, fn m, {:ok, s} ->
        mv = if m == "pass", do: :pass, else: m
        if mv in Go.legal(s), do: {:cont, {:ok, Go.play(s, mv)}}, else: {:halt, {:error, "illegal move #{inspect(m)}"}}
      end)
      with {:ok, s} <- s do
        s =
          if req["action"] == "engine" and Go.outcome(s) == nil do
            r = Vapor.Play.mcts(Go.Playouts, s, sims: clamp(req["sims"], 50, 3000, 600), seed: length(req["moves"] || []) + 1)
            Map.put(Go.play(s, r.best), :engine, %{move: if(r.best == :pass, do: "pass", else: r.best), value: r.value, sims: r.simulations})
          else
            s
          end
        {:ok, %{size: n, board: Tuple.to_list(s.board), turn: s.turn, passes: s.passes, legal: Enum.map(Go.legal(s), &(if &1 == :pass, do: "pass", else: &1)),
                score: Go.score(s), over: Go.outcome(s) != nil, engine: Map.get(s, :engine)}}
      end
    end)
  end

  def mnk(req) do
    m = clamp(req["m"], 2, 7, 3); n = clamp(req["n"], 2, 7, 3); k = clamp(req["k"], 2, 5, 3)
    timed(fn ->
      s0 = MNK.new(m, n, k, gravity: req["gravity"] == true)
      s = Enum.reduce(req["moves"] || [], s0, fn mv, s -> if mv in MNK.legal(s), do: MNK.play(s, mv), else: s end)
      cond do
        MNK.outcome(s) != nil -> {:ok, %{board: Tuple.to_list(s.board), outcome: MNK.outcome(s), turn: s.turn}}
        m * n <= 16 ->
          {v, best} = Vapor.Play.solve(MNK, s)
          {:ok, %{board: Tuple.to_list(s.board), value: v, best: best, turn: s.turn, solved: true}}
        true ->
          r = Vapor.Play.mcts(MNK, s, sims: 800)
          {:ok, %{board: Tuple.to_list(s.board), best: [r.best], value: r.value, turn: s.turn, solved: false}}
      end
    end, 90_000)
  end

  def poker(req) do
    game = if req["game"] == "leduc", do: Poker.Leduc, else: Poker.Kuhn
    its = clamp(req["iterations"], 10, if(game == Poker.Leduc, do: 200, else: 3000), if(game == Poker.Leduc, do: 60, else: 800))
    key = {:poker, game, its}
    r = case :persistent_term.get({__MODULE__, key}, nil) do
      nil -> (v = timed(fn -> Poker.solve(game, its, every: max(div(its, 20), 1)) end, 300_000); if is_map(v), do: :persistent_term.put({__MODULE__, key}, v); v)
      v -> v
    end
    case r do
      %{} -> {:ok, %{game: req["game"] || "kuhn", iterations: r.iterations, exploitability: r.exploitability, value: r.value, curve: r.curve, infosets: r.infosets,
                    uniform_exploitability: Poker.exploitability(game, Poker.uniform()),
                    strategy: r.strategy |> Enum.sort() |> Enum.take(if(game == Poker.Kuhn, do: 50, else: 80)) |> Map.new()}}
      e -> e
    end
  end

  # ============================================================== proteins

  def protein(req) do
    timed(fn ->
      case req["action"] || "analyse" do
        "analyse" ->
          with {:ok, s} <- pdb(req) do
            {:ok, %{sequence: s.sequence, length: length(s.ca), secondary: Structure.secondary(s.ca), ca: Enum.map(s.ca, &Tuple.to_list/1),
                    contacts: Structure.contacts(s.ca, 8.0, 6) |> Enum.map(&Tuple.to_list/1), chain: s.chain}}
          end
        "compare" ->
          with {:ok, a} <- pdb(%{"pdb" => req["model"], "sample" => req["model_sample"]}), {:ok, b} <- pdb(%{"pdb" => req["native"], "sample" => req["native_sample"]}) do
            if length(a.ca) != length(b.ca), do: {:error, "the two chains differ in length (#{length(a.ca)} vs #{length(b.ca)}): residue correspondence needed"}, else: (
              r = Structure.tm_score(a.ca, b.ca)
              moved = Structure.transform(a.ca, r.superposition)
              {:ok, %{tm: r.tm, rmsd: r.rmsd, gdt_ts: r.gdt_ts, gdt_ha: r.gdt_ha, lddt: Structure.lddt(a.ca, b.ca), d0: r.d0, model: Enum.map(moved, &Tuple.to_list/1), native: Enum.map(b.ca, &Tuple.to_list/1)}})
          end
        "pipeline" ->
          with {:ok, nat} <- pdb(req) do
            l = length(nat.ca)
            if l > 120, do: throw({:protein, "at most 120 residues for the in-console pipeline"})
            ss = Structure.secondary(nat.ca)
            allc = Structure.contacts(nat.ca, 8.0, 3)
            truth = Structure.contacts(nat.ca, 8.0, 6)
            k = length(truth)
            msa = Coevolution.sample(l, allc, n: clamp(req["sequences"], 200, 4000, 2000), coupling: 0.6, sweeps: 3, seed: clamp(req["seed"], 1, 1_000_000, 1))
            dca = Coevolution.dca(msa)
            mi = Coevolution.mi(msa)
            local = Enum.filter(allc, fn {i, j} -> j - i < 6 end)
            fold = Structure.fold(l, local ++ Enum.take(dca, k), helices: ss, restarts: 2)
            r = Structure.tm_score(fold.ca, nat.ca)
            moved = Structure.transform(fold.ca, r.superposition)
            {:ok, %{length: l, sequence: nat.sequence, secondary: ss, true_contacts: Enum.map(truth, &Tuple.to_list/1),
                    dca_top: dca |> Enum.take(k) |> Enum.map(&Tuple.to_list/1), mi_top: mi |> Enum.take(k) |> Enum.map(&Tuple.to_list/1),
                    precision: %{dca: Structure.precision(dca, truth, k), mi: Structure.precision(mi, truth, k), chance: k / max(div((l - 6) * (l - 5), 2), 1)},
                    tm: r.tm, rmsd: r.rmsd, gdt_ts: r.gdt_ts, lddt: Structure.lddt(fold.ca, nat.ca), mirrored: fold.mirrored,
                    model: Enum.map(moved, &Tuple.to_list/1), native: Enum.map(nat.ca, &Tuple.to_list/1), model_pdb: Structure.to_pdb(moved, nat.sequence)}}
          end
        "align" ->
          a = req["a"] || ""; b = req["b"] || ""
          if byte_size(a) > 3000 or byte_size(b) > 3000 or a == "" or b == "", do: {:error, "a, b: sequences of 1–3000 letters"},
            else: {:ok, Align.align(a, b, mode: if(req["mode"] == "local", do: :local, else: :global), open: clamp(req["open"], 1, 30, 11), extend: clamp(req["extend"], 1, 10, 1))}
        a -> {:error, "action: analyse, compare, pipeline or align (got #{a})"}
      end
    end, 240_000)
  catch
    {:protein, w} -> {:error, w}
  end

  @samples ~w(1A8O 1LCD)
  # an NMR entry's other models: "1LCD#2" is the second model of 1LCD
  defp pdb(%{"sample" => <<id::binary-size(4), "#", k::binary>>}) when id in @samples do
    with {n, ""} <- Integer.parse(k), models = Structure.read_models(File.read!(Path.join([to_string(:code.priv_dir(:vapor)), "quality", "protein", id <> ".pdb"]))),
         m when m != nil <- Enum.at(models, n - 1) do
      {:ok, m}
    else
      _ -> {:error, "sample #{id}##{k}: no such model"}
    end
  end
  defp pdb(%{"sample" => s}) when s in @samples, do: Structure.read_pdb(File.read!(Path.join([to_string(:code.priv_dir(:vapor)), "quality", "protein", s <> ".pdb"])))
  defp pdb(%{"pdb" => t}) when is_binary(t) and byte_size(t) < 5_000_000, do: Structure.read_pdb(t)
  defp pdb(_), do: {:error, "pdb: the text of a PDB file, or sample: 1A8O | 1LCD"}

  # ================================================================ render

  def render(req) do
    with {:ok, t} <- text(req["text"]), {:ok, scene} <- Render.parse(t) do
      w = clamp(req["width"], 16, 480, 240); h = clamp(req["height"], 16, 320, 150); spp = clamp(req["spp"], 1, 256, 16)
      if w * h * spp > 6_000_000, do: {:error, "width × height × spp at most 6 million on the server (the GPU tracer in the page has no such limit)"},
        else: (case timed(fn -> Render.render(scene, width: w, height: h, spp: spp, seed: clamp(req["seed"], 1, 1_000_000, 1)) end, 240_000) do
          %{} = r -> {:ok, %{png: "data:image/png;base64," <> Base.encode64(r.png), ms: r.ms, rays: r.rays, w: w, h: h, spp: spp,
                            mean: r.linear |> List.flatten() |> Enum.map(fn {a, b, c} -> 0.2126 * a + 0.7152 * b + 0.0722 * c end) |> then(&(Enum.sum(&1) / length(&1)))}}
          e -> e
        end)
    end
  end

  def furnace do
    case :persistent_term.get({__MODULE__, :furnace}, nil) do
      nil ->
        v = %{uniform: Render.furnace(0.8, spp: 16), gradient: Render.furnace_gradient(0.8), control: Render.furnace_gradient(0.8, biased: true)}
        :persistent_term.put({__MODULE__, :furnace}, v)
        v
      v -> v
    end
  end

  # ========================================================== scene frames

  def scene_gif(req) do
    frames = req["frames"] || []
    cond do
      not is_list(frames) or frames == [] or length(frames) > 240 -> {:error, "frames: 1–240 PNG data URLs"}
      true ->
        imgs = Enum.map(frames, fn f ->
          bin = f |> String.replace(~r/^data:image\/png;base64,/, "") |> Base.decode64!(ignore: :whitespace)
          {:ok, %{image: img}} = Vapor.Docs.Pictures.read(:png, bin)
          img
        end)
        gif = Vapor.Media.GIF.encode(imgs, fps: clamp(req["fps"], 1, 50, 10))
        {:ok, %{gif: "data:image/gif;base64," <> Base.encode64(gif), frames: length(imgs), bytes: byte_size(gif)}}
    end
  rescue
    _ -> {:error, "frames: PNG data URLs of one size"}
  end

  defp clamp(n, lo, hi, _d) when is_integer(n), do: n |> max(lo) |> min(hi)
  defp clamp(n, lo, hi, d) when is_float(n), do: clamp(round(n), lo, hi, d)
  defp clamp(_, _lo, _hi, d), do: d
end
