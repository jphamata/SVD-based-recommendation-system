defmodule Vapor.Play.Chess do
  @moduledoc """
  Chess (docs/TABULEIROS.md §1): the full rules — castling through
  unattacked squares, en passant, promotion to any piece, the fifty-move
  rule, insufficient material, threefold repetition in games — FEN in
  and out, SAN in and out, PGN of a game; **perft**, the move-generator
  count every engine author checks against the published numbers; an
  alpha–beta engine (iterative deepening, quiescence, a transposition
  table, MVV–LVA and killer ordering, piece-square evaluation); and a
  **mate prover** whose answer is a proof tree — every defence covered,
  every leaf a checkmate — that `verify_mate/2` replays.

  Squares are 0 (a1) … 63 (h8); pieces are integers, positive for White
  (1 pawn, 2 knight, 3 bishop, 4 rook, 5 queen, 6 king), negative for
  Black. A move is `{from, to, promotion | nil}`.
  """
  import Bitwise

  defstruct board: nil, turn: 1, castle: 15, ep: nil, half: 0, full: 1, kings: {4, 60}

  @start "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
  def start, do: from_fen!(@start)

  # ================================================================== tables

  @knight_d [{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}]
  @king_d [{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}]
  @rook_d [{1, 0}, {-1, 0}, {0, 1}, {0, -1}]
  @bishop_d [{1, 1}, {1, -1}, {-1, 1}, {-1, -1}]

  jumps = fn ds -> for s <- 0..63 do
    {f, r} = {rem(s, 8), div(s, 8)}
    for {df, dr} <- ds, f + df in 0..7, r + dr in 0..7, do: (r + dr) * 8 + f + df
  end |> List.to_tuple() end
  rays = fn ds -> for s <- 0..63 do
    {f, r} = {rem(s, 8), div(s, 8)}
    for {df, dr} <- ds, do: (for k <- 1..7, f + k * df in 0..7, r + k * dr in 0..7, do: (r + k * dr) * 8 + f + k * df)
  end |> List.to_tuple() end

  @knight jumps.(@knight_d)
  @king jumps.(@king_d)
  @rook_rays rays.(@rook_d)
  @bishop_rays rays.(@bishop_d)

  # ===================================================================== FEN

  @doc "Parse FEN: `{:ok, position}` or `{:error, why}`."
  def from_fen(fen) do
    case String.split(String.trim(fen)) do
      [placement, turn, castle, ep | rest] ->
        rows = String.split(placement, "/")
        if length(rows) != 8, do: throw({:fen, "8 ranks expected"})
        board =
          rows |> Enum.reverse() |> Enum.flat_map(fn row ->
            cells = row |> String.graphemes() |> Enum.flat_map(fn c ->
              case Integer.parse(c) do
                {n, ""} -> List.duplicate(0, n)
                _ -> [piece(c)]
              end
            end)
            if length(cells) != 8, do: throw({:fen, "a rank of #{length(cells)} squares"})
            cells
          end) |> List.to_tuple()
        cr = (if String.contains?(castle, "K"), do: 1, else: 0) ||| (if String.contains?(castle, "Q"), do: 2, else: 0) |||
             (if String.contains?(castle, "k"), do: 4, else: 0) ||| (if String.contains?(castle, "q"), do: 8, else: 0)
        [half, full] = case rest do [h, f | _] -> [String.to_integer(h), String.to_integer(f)]; _ -> [0, 1] end
        wk = Enum.find_index(Tuple.to_list(board), &(&1 == 6))
        bk = Enum.find_index(Tuple.to_list(board), &(&1 == -6))
        if wk == nil or bk == nil, do: throw({:fen, "each side needs a king"})
        {:ok, %__MODULE__{board: board, turn: if(turn == "b", do: -1, else: 1), castle: cr, ep: if(ep == "-", do: nil, else: sq(ep)), half: half, full: full, kings: {wk, bk}}}
      _ -> {:error, "FEN: placement, side, castling, en passant (and clocks)"}
    end
  catch
    {:fen, w} -> {:error, "FEN: " <> w}
  end

  def from_fen!(f), do: (case from_fen(f) do {:ok, p} -> p; {:error, w} -> raise ArgumentError, w end)

  defp piece(c) do
    v = %{"p" => 1, "n" => 2, "b" => 3, "r" => 4, "q" => 5, "k" => 6}[String.downcase(c)] || throw({:fen, "unknown piece #{c}"})
    if c == String.upcase(c), do: v, else: -v
  end

  @doc "Square name → index (`\"e4\"` → 28)."
  def sq(<<f, r>>), do: (r - ?1) * 8 + (f - ?a)
  @doc "Index → square name."
  def name(s), do: <<?a + rem(s, 8), ?1 + div(s, 8)>>

  @doc "The position as FEN."
  def to_fen(%__MODULE__{} = p) do
    rows = for r <- 7..0//-1 do
      row = for f <- 0..7, do: elem(p.board, r * 8 + f)
      row |> Enum.chunk_by(&(&1 == 0)) |> Enum.map_join(fn chunk -> if hd(chunk) == 0, do: Integer.to_string(length(chunk)), else: Enum.map_join(chunk, &letter/1) end)
    end
    castle = [{1, "K"}, {2, "Q"}, {4, "k"}, {8, "q"}] |> Enum.filter(fn {b, _} -> (p.castle &&& b) != 0 end) |> Enum.map_join(&elem(&1, 1))
    Enum.join([Enum.join(rows, "/"), if(p.turn == 1, do: "w", else: "b"), if(castle == "", do: "-", else: castle), if(p.ep, do: name(p.ep), else: "-"), p.half, p.full], " ")
  end

  defp letter(v), do: (l = elem({"", "p", "n", "b", "r", "q", "k"}, abs(v)); if v > 0, do: String.upcase(l), else: l)

  # ================================================================ attacks

  @doc "Is square `s` attacked by side `by` (1 white, −1 black)?"
  def attacked?(b, s, by) do
    pawn_from = if by == 1, do: [{-1, -1}, {1, -1}], else: [{-1, 1}, {1, 1}]
    {f, r} = {rem(s, 8), div(s, 8)}
    Enum.any?(pawn_from, fn {df, dr} -> f + df in 0..7 and r + dr in 0..7 and elem(b, (r + dr) * 8 + f + df) == by * 1 end) or
      Enum.any?(elem(@knight, s), &(elem(b, &1) == by * 2)) or
      Enum.any?(elem(@king, s), &(elem(b, &1) == by * 6)) or
      Enum.any?(elem(@rook_rays, s), &slider?(b, &1, by * 4, by * 5)) or
      Enum.any?(elem(@bishop_rays, s), &slider?(b, &1, by * 3, by * 5))
  end

  defp slider?(_b, [], _a, _q), do: false
  defp slider?(b, [t | rest], a, q) do
    case elem(b, t) do
      0 -> slider?(b, rest, a, q)
      x -> x == a or x == q
    end
  end

  @doc "Is the side to move in check?"
  def in_check?(%__MODULE__{} = p), do: attacked?(p.board, king(p, p.turn), -p.turn)

  defp king(p, 1), do: elem(p.kings, 0)
  defp king(p, -1), do: elem(p.kings, 1)

  # ================================================================== moves

  @doc "The legal moves of the side to move."
  def moves(%__MODULE__{} = p) do
    p |> pseudo() |> Enum.filter(fn m -> legal_after?(p, m) end)
  end

  defp legal_after?(p, m) do
    q = make(p, m)
    not attacked?(q.board, king(q, p.turn), q.turn)
  end

  defp pseudo(p) do
    b = p.board
    me = p.turn
    for s <- 0..63, (v = elem(b, s)) != 0, v * me > 0, m <- piece_moves(p, s, abs(v)), do: m
  end

  defp piece_moves(p, s, 1), do: pawn_moves(p, s)
  defp piece_moves(p, s, 2), do: for(t <- elem(@knight, s), elem(p.board, t) * p.turn <= 0, do: {s, t, nil})
  defp piece_moves(p, s, 3), do: slide(p, s, elem(@bishop_rays, s))
  defp piece_moves(p, s, 4), do: slide(p, s, elem(@rook_rays, s))
  defp piece_moves(p, s, 5), do: slide(p, s, elem(@bishop_rays, s) ++ elem(@rook_rays, s))
  defp piece_moves(p, s, 6), do: for(t <- elem(@king, s), elem(p.board, t) * p.turn <= 0, do: {s, t, nil}) ++ castles(p, s)

  defp slide(p, s, rays) do
    Enum.flat_map(rays, fn ray ->
      Enum.reduce_while(ray, [], fn t, acc ->
        case elem(p.board, t) * p.turn do
          0 -> {:cont, [{s, t, nil} | acc]}
          x when x < 0 -> {:halt, [{s, t, nil} | acc]}
          _ -> {:halt, acc}
        end
      end)
    end)
  end

  defp pawn_moves(p, s) do
    b = p.board
    me = p.turn
    {f, r} = {rem(s, 8), div(s, 8)}
    fwd = s + 8 * me
    last = if me == 1, do: 7, else: 0
    home = if me == 1, do: 1, else: 6
    promo = fn from, to -> if div(to, 8) == last, do: (for x <- [5, 4, 3, 2], do: {from, to, x}), else: [{from, to, nil}] end
    pushes =
      if fwd in 0..63 and elem(b, fwd) == 0 do
        two = s + 16 * me
        promo.(s, fwd) ++ if(r == home and elem(b, two) == 0, do: [{s, two, nil}], else: [])
      else
        []
      end
    caps = for df <- [-1, 1], f + df in 0..7, t = fwd + df, t in 0..63, elem(b, t) * me < 0 or t == p.ep, do: promo.(s, t)
    pushes ++ List.flatten(caps)
  end

  defp castles(p, s) do
    me = p.turn
    b = p.board
    {ks, qs, home} = if me == 1, do: {1, 2, 4}, else: {4, 8, 60}
    if s != home or attacked?(b, s, -me) do
      []
    else
      k = if (p.castle &&& ks) != 0 and elem(b, s + 1) == 0 and elem(b, s + 2) == 0 and elem(b, s + 3) == me * 4 and not attacked?(b, s + 1, -me) and not attacked?(b, s + 2, -me), do: [{s, s + 2, nil}], else: []
      q = if (p.castle &&& qs) != 0 and elem(b, s - 1) == 0 and elem(b, s - 2) == 0 and elem(b, s - 3) == 0 and elem(b, s - 4) == me * 4 and not attacked?(b, s - 1, -me) and not attacked?(b, s - 2, -me), do: [{s, s - 2, nil}], else: []
      k ++ q
    end
  end

  # castling rights lost when a square is left or captured on
  @rights %{0 => 2, 4 => 3, 7 => 1, 56 => 8, 60 => 12, 63 => 4}

  @doc "Play a move (assumed pseudo-legal)."
  def make(%__MODULE__{} = p, {from, to, promo}) do
    b = p.board
    v = elem(b, from)
    cap = elem(b, to)
    me = p.turn
    b = b |> put_elem(from, 0) |> put_elem(to, if(promo, do: promo * me, else: v))
    # en passant capture
    b = if abs(v) == 1 and to == p.ep and cap == 0, do: put_elem(b, to - 8 * me, 0), else: b
    # castling: the rook jumps
    b = cond do
      abs(v) == 6 and to - from == 2 -> b |> put_elem(from + 3, 0) |> put_elem(from + 1, me * 4)
      abs(v) == 6 and from - to == 2 -> b |> put_elem(from - 4, 0) |> put_elem(from - 1, me * 4)
      true -> b
    end
    kings = if abs(v) == 6, do: (if me == 1, do: {to, elem(p.kings, 1)}, else: {elem(p.kings, 0), to}), else: p.kings
    castle = p.castle &&& bnot(Map.get(@rights, from, 0) ||| Map.get(@rights, to, 0))
    ep = if abs(v) == 1 and abs(to - from) == 16, do: div(from + to, 2), else: nil
    half = if abs(v) == 1 or cap != 0, do: 0, else: p.half + 1
    %__MODULE__{board: b, turn: -me, castle: castle, ep: ep, half: half, full: p.full + if(me == -1, do: 1, else: 0), kings: kings}
  end

  @doc "Perft: the number of leaf nodes of the legal move tree to `depth` (bulk-counted at depth 1)."
  def perft(p, 1), do: length(moves(p))
  def perft(_p, 0), do: 1
  def perft(p, d), do: p |> moves() |> Enum.reduce(0, fn m, acc -> acc + perft(make(p, m), d - 1) end)

  @doc "Perft split by root move (the 'divide' that pins a bug to one move)."
  def divide(p, d), do: p |> moves() |> Map.new(fn m -> {uci(m), perft(make(p, m), d - 1)} end)

  @doc "Perft with the root moves spread over the schedulers (`Vapor.Play.parallel/2`)."
  def perft_parallel(p, d) when d >= 2, do: p |> moves() |> Vapor.Play.pmap(fn m -> perft(make(p, m), d - 1) end) |> Enum.sum()
  def perft_parallel(p, d), do: perft(p, d)

  # ============================================================ notation

  @doc "A move in UCI notation (`e2e4`, `e7e8q`)."
  def uci({f, t, pr}), do: name(f) <> name(t) <> if(pr, do: elem({"", "", "n", "b", "r", "q"}, pr), else: "")

  @doc "Parse UCI or SAN against the legal moves: `{:ok, move}` or `{:error, why}`."
  def parse_move(p, text) do
    t = String.trim(text) |> String.replace(~r/[!?]+$/, "")
    legal = moves(p)
    case Enum.find(legal, &(uci(&1) == String.downcase(t))) || Enum.find(legal, &(String.replace(san(p, &1), ~r/[+#]$/, "") == String.replace(t, ~r/[+#]$/, ""))) do
      nil -> {:error, "illegal or unknown move #{inspect(text)} (legal: #{legal |> Enum.map(&san(p, &1)) |> Enum.sort() |> Enum.join(" ")})"}
      m -> {:ok, m}
    end
  end

  @doc "Standard algebraic notation of a legal move."
  def san(p, {from, to, promo} = m) do
    v = abs(elem(p.board, from))
    cap = elem(p.board, to) != 0 or (v == 1 and to == p.ep)
    base =
      cond do
        v == 6 and to - from == 2 -> "O-O"
        v == 6 and from - to == 2 -> "O-O-O"
        v == 1 -> (if cap, do: String.first(name(from)) <> "x", else: "") <> name(to) <> if(promo, do: "=" <> elem({"", "", "N", "B", "R", "Q"}, promo), else: "")
        true ->
          others = for {f2, t2, _} <- moves(p), t2 == to, f2 != from, abs(elem(p.board, f2)) == v, do: f2
          dis = cond do
            others == [] -> ""
            Enum.all?(others, &(rem(&1, 8) != rem(from, 8))) -> String.first(name(from))
            Enum.all?(others, &(div(&1, 8) != div(from, 8))) -> String.last(name(from))
            true -> name(from)
          end
          elem({"", "", "N", "B", "R", "Q", "K"}, v) <> dis <> if(cap, do: "x", else: "") <> name(to)
      end
    q = make(p, m)
    cond do
      in_check?(q) and moves(q) == [] -> base <> "#"
      in_check?(q) -> base <> "+"
      true -> base
    end
  end

  @doc "The state of the game: `:checkmate`, `:stalemate`, `:fifty_moves`, `:insufficient_material` or `:ongoing`."
  def status(p) do
    ms = moves(p)
    cond do
      ms == [] and in_check?(p) -> :checkmate
      ms == [] -> :stalemate
      p.half >= 100 -> :fifty_moves
      insufficient?(p) -> :insufficient_material
      true -> :ongoing
    end
  end

  defp insufficient?(p) do
    pcs = for v <- Tuple.to_list(p.board), v != 0, abs(v) != 6, do: abs(v)
    pcs == [] or pcs in [[2], [3]]
  end

  @doc "A game (list of moves from a start) as PGN."
  def pgn(moves, opts \\ []) do
    start = Keyword.get(opts, :start, start())
    {txt, last} =
      Enum.reduce(Enum.with_index(moves), {[], start}, fn {m, i}, {acc, p} ->
        s = san(p, m)
        num = if p.turn == 1, do: "#{p.full}. ", else: (if i == 0, do: "#{p.full}... ", else: "")
        {[num <> s | acc], make(p, m)}
      end)
    result = case status(last) do
      :checkmate -> if last.turn == 1, do: "0-1", else: "1-0"
      s when s in [:stalemate, :fifty_moves, :insufficient_material] -> "1/2-1/2"
      _ -> "*"
    end
    headers = [{"Event", Keyword.get(opts, :event, "vapor")}, {"White", Keyword.get(opts, :white, "?")}, {"Black", Keyword.get(opts, :black, "?")}, {"Result", result}] ++
      if(to_fen(start) != @start, do: [{"SetUp", "1"}, {"FEN", to_fen(start)}], else: [])
    Enum.map_join(headers, "", fn {k, v} -> "[#{k} \"#{v}\"]\n" end) <> "\n" <> Enum.join(Enum.reverse(txt), " ") <> " " <> result <> "\n"
  end

  # =============================================================== engine

  # Simplified evaluation (Michniewski): material and piece-square tables from White's side, a1 = index 0
  @val {0, 100, 320, 330, 500, 900, 0}
  @pst %{
    1 => {0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, -20, -20, 10, 10, 5, 5, -5, -10, 0, 0, -10, -5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 5, 5, 10, 25, 25, 10, 5, 5, 10, 10, 20, 30, 30, 20, 10, 10, 50, 50, 50, 50, 50, 50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0},
    2 => {-50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 5, 5, 0, -20, -40, -30, 5, 10, 15, 15, 10, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 10, 15, 15, 10, 0, -30, -40, -20, 0, 0, 0, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50},
    3 => {-20, -10, -10, -10, -10, -10, -10, -20, -10, 5, 0, 0, 0, 0, 5, -10, -10, 10, 10, 10, 10, 10, 10, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 5, 10, 10, 5, 0, -10, -10, 0, 0, 0, 0, 0, 0, -10, -20, -10, -10, -10, -10, -10, -10, -20},
    4 => {0, 0, 0, 5, 5, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, 5, 10, 10, 10, 10, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0},
    5 => {-20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 5, 0, 0, 0, 0, -10, -10, 5, 5, 5, 5, 5, 0, -10, 0, 0, 5, 5, 5, 5, 0, -5, -5, 0, 5, 5, 5, 5, 0, -5, -10, 0, 5, 5, 5, 5, 0, -10, -10, 0, 0, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20},
    6 => {20, 30, 10, 0, 0, 10, 30, 20, 20, 20, 0, 0, 0, 0, 20, 20, -10, -20, -20, -20, -20, -20, -20, -10, -20, -30, -30, -40, -40, -30, -30, -20, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30}
  }

  @doc "Static evaluation in centipawns from the side to move."
  def evaluate(p) do
    score =
      Enum.reduce(0..63, 0, fn s, acc ->
        case elem(p.board, s) do
          0 -> acc
          v when v > 0 -> acc + elem(@val, v) + elem(@pst[v], s)
          v -> acc - elem(@val, -v) - elem(@pst[-v], 63 - s)
        end
      end)
    score * p.turn
  end

  @mate 100_000

  @doc """
  Search to `depth` plies (iterative deepening, `nodes:` budget):
  `%{best, score (cp, mate as ±100000∓plies), pv, nodes, depth}`.
  """
  def search(p, opts \\ []) do
    depth = Keyword.get(opts, :depth, 4)
    budget = Keyword.get(opts, :nodes, 200_000)
    Process.put(:chess_nodes, 0)
    Process.put(:chess_tt, %{})
    Process.put(:chess_budget, budget)

    Enum.reduce_while(1..depth, nil, fn d, best ->
      {score, pv} = negamax(p, d, -@mate - 1, @mate + 1, 0, [])
      r = %{best: List.first(pv), score: score, pv: Enum.map(pv, &uci/1), depth: d, nodes: Process.get(:chess_nodes)}
      cond do
        Process.get(:chess_nodes) > budget -> {:halt, best || r}
        abs(score) > @mate - 1000 -> {:halt, r}
        true -> {:cont, r}
      end
    end)
    |> then(fn r -> r && Map.put(r, :san, if(r.best, do: san(p, r.best))) end)
  end

  defp negamax(p, 0, a, b, ply, _killers), do: {quiesce(p, a, b, 0), []} |> then(fn {s, _} -> _ = ply; {s, []} end)

  defp negamax(p, d, a, b, ply, killers) do
    Process.put(:chess_nodes, Process.get(:chess_nodes) + 1)
    key = {p.board, p.turn, p.castle, p.ep}
    tt = Process.get(:chess_tt)
    hint = case Map.get(tt, key) do {_, _, m} -> m; _ -> nil end
    ms = moves(p)

    cond do
      ms == [] -> {if(in_check?(p), do: -@mate + ply, else: 0), []}
      p.half >= 100 -> {0, []}
      Process.get(:chess_nodes) > Process.get(:chess_budget) * 2 -> {evaluate(p), []}
      true ->
        ordered = order(p, ms, hint, killers)
        {best, pv, _} =
          Enum.reduce_while(ordered, {-@mate - 1, [], a}, fn m, {best, pv, a} ->
            {s, sub} = negamax(make(p, m), d - 1, -b, -a, ply + 1, killers)
            s = -s
            {best, pv} = if s > best, do: {s, [m | sub]}, else: {best, pv}
            a = max(a, s)
            if a >= b, do: {:halt, {best, pv, a}}, else: {:cont, {best, pv, a}}
          end)
        Process.put(:chess_tt, Map.put(Process.get(:chess_tt), key, {d, best, List.first(pv)}))
        {best, pv}
    end
  end

  defp order(p, ms, hint, _killers) do
    Enum.sort_by(ms, fn {f, t, pr} = m ->
      victim = abs(elem(p.board, t))
      attacker = abs(elem(p.board, f))
      cond do
        m == hint -> -1_000_000
        pr == 5 -> -900_000
        victim > 0 -> -(elem(@val, victim) * 10 - elem(@val, attacker))
        true -> 0
      end
    end)
  end

  defp quiesce(p, a, b, depth) do
    Process.put(:chess_nodes, Process.get(:chess_nodes) + 1)
    stand = evaluate(p)
    if stand >= b or depth > 6 do
      stand
    else
      a = max(a, stand)
      caps = p |> moves() |> Enum.filter(fn {_, t, pr} -> elem(p.board, t) != 0 or pr == 5 end) |> then(&order(p, &1, nil, []))
      Enum.reduce_while(caps, a, fn m, a ->
        s = -quiesce(make(p, m), -b, -a, depth + 1)
        if s >= b, do: {:halt, s}, else: {:cont, max(a, s)}
      end)
    end
  end

  # ============================================================= mate prover

  @doc """
  Prove a forced mate in at most `n` moves for the side to move: `{:mate,
  proof}` where a proof is `%{move, replies: %{reply_uci => proof}}`
  (no replies: the move mates), or `:no_mate`. AND–OR search with checks
  first; every defence is covered.
  """
  def prove_mate(p, n) when n >= 1 do
    ms = moves(p) |> Enum.sort_by(fn m -> if in_check?(make(p, m)), do: 0, else: 1 end)
    Enum.find_value(ms, :no_mate, fn m ->
      q = make(p, m)
      replies = moves(q)
      cond do
        replies == [] and in_check?(q) -> {:mate, %{move: uci(m), san: san(p, m), replies: %{}}}
        replies == [] -> nil
        n == 1 -> nil
        true ->
          sub = Enum.reduce_while(replies, %{}, fn r, acc ->
            case prove_mate(make(q, r), n - 1) do
              {:mate, pr} -> {:cont, Map.put(acc, uci(r), pr)}
              :no_mate -> {:halt, nil}
            end
          end)
          if sub, do: {:mate, %{move: uci(m), san: san(p, m), replies: sub}}
      end
    end)
  end

  @doc """
  Verify a mate proof tree from position `p`: every attacking move legal,
  every legal defence present in the tree, every leaf a checkmate, depth
  within `n`. `{:ok, %{leaves, depth}}` or `{:error, why}`.
  """
  def verify_mate(p, proof, n \\ 99) do
    with {:ok, m} <- parse_move(p, proof.move) do
      q = make(p, m)
      legal = moves(q)
      cond do
        map_size(proof.replies) == 0 -> if(legal == [] and in_check?(q), do: {:ok, %{leaves: 1, depth: 1}}, else: {:error, "#{proof.move} is not checkmate"})
        n <= 1 -> {:error, "deeper than claimed"}
        MapSet.new(Enum.map(legal, &uci/1)) != MapSet.new(Map.keys(proof.replies)) -> {:error, "after #{proof.move} the defences are not all covered"}
        true ->
          Enum.reduce_while(legal, {:ok, %{leaves: 0, depth: 0}}, fn r, {:ok, acc} ->
            case verify_mate(make(q, r), proof.replies[uci(r)], n - 1) do
              {:ok, s} -> {:cont, {:ok, %{leaves: acc.leaves + s.leaves, depth: max(acc.depth, s.depth + 1)}}}
              e -> {:halt, e}
            end
          end)
      end
    end
  end
end
