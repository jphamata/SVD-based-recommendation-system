defmodule Vapor.Play.Shogi do
  @moduledoc """
  Shogi (docs/BOARDS.md §2): the full rules — eight piece kinds and
  their six promotions, captured pieces changing sides and **dropped**
  back (no two unpromoted pawns on a file, no piece dropped where it
  could never move, no pawn drop that mates), compulsory promotion, the
  promotion zone — SFEN in and out, USI moves (`7g7f`, `8h2b+`, `P*5e`),
  and **perft** against the published counts from the initial position
  (30, 900, 25 470, 719 731) and against `python-shogi` on random
  positions in the tests.

  Squares: x = 9 − file (0 … 8 left to right as SFEN writes them), y =
  rank a … i (0 … 8, top to bottom); index y·9 + x. Pieces are integers:
  1 pawn, 2 lance, 3 knight, 4 silver, 5 gold, 6 bishop, 7 rook, 8 king,
  +8 when promoted; positive for Sente (Black, moving up), negative for Gote.
  """

  defstruct board: nil, turn: 1, hands: %{1 => %{}, -1 => %{}}, ply: 1

  @start "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
  def start, do: from_sfen!(@start)

  @letters %{"p" => 1, "l" => 2, "n" => 3, "s" => 4, "g" => 5, "b" => 6, "r" => 7, "k" => 8}
  @names {nil, "P", "L", "N", "S", "G", "B", "R", "K"}

  # ================================================================= SFEN

  def from_sfen(sfen) do
    [placement, side, hands | rest] = String.split(String.trim(sfen)) ++ ["1"]
    rows = String.split(placement, "/")
    if length(rows) != 9, do: throw({:sfen, "9 ranks expected"})
    board =
      rows |> Enum.flat_map(fn row ->
        {cells, _} = row |> String.graphemes() |> Enum.reduce({[], false}, fn c, {acc, plus} ->
          cond do
            c == "+" -> {acc, true}
            c =~ ~r/^\d$/ -> {acc ++ List.duplicate(0, String.to_integer(c)), false}
            true -> {acc ++ [piece(c, plus)], false}
          end
        end)
        if length(cells) != 9, do: throw({:sfen, "a rank of #{length(cells)} squares"})
        cells
      end) |> List.to_tuple()
    hands = parse_hands(hands)
    {:ok, %__MODULE__{board: board, turn: if(side == "w", do: -1, else: 1), hands: hands, ply: (case Integer.parse(hd(rest)) do {n, _} -> n; _ -> 1 end)}}
  catch
    {:sfen, w} -> {:error, "SFEN: " <> w}
  end

  def from_sfen!(s), do: (case from_sfen(s) do {:ok, p} -> p; {:error, w} -> raise ArgumentError, w end)

  defp piece(c, plus) do
    k = @letters[String.downcase(c)] || throw({:sfen, "unknown piece #{c}"})
    k = if plus, do: k + 8, else: k
    if c == String.upcase(c), do: k, else: -k
  end

  defp parse_hands("-"), do: %{1 => %{}, -1 => %{}}
  defp parse_hands(s) do
    Regex.scan(~r/(\d*)([PLNSGBRplnsgbr])/, s)
    |> Enum.reduce(%{1 => %{}, -1 => %{}}, fn [_, n, c], h ->
      side = if c == String.upcase(c), do: 1, else: -1
      k = @letters[String.downcase(c)]
      put_in(h, [side, k], Map.get(h[side], k, 0) + if(n == "", do: 1, else: String.to_integer(n)))
    end)
  end

  def to_sfen(p) do
    rows = for y <- 0..8 do
      for(x <- 0..8, do: elem(p.board, y * 9 + x)) |> Enum.chunk_by(&(&1 == 0))
      |> Enum.map_join(fn ch -> if hd(ch) == 0, do: Integer.to_string(length(ch)), else: Enum.map_join(ch, &pname/1) end)
    end
    hand = for side <- [1, -1], k <- [7, 6, 5, 4, 3, 2, 1], (n = Map.get(p.hands[side], k, 0)) > 0, do: (if n > 1, do: "#{n}", else: "") <> (if side == 1, do: elem(@names, k), else: String.downcase(elem(@names, k)))
    Enum.join([Enum.join(rows, "/"), if(p.turn == 1, do: "b", else: "w"), if(hand == [], do: "-", else: Enum.join(hand)), p.ply], " ")
  end

  defp pname(v) do
    k = abs(v)
    s = (if k > 8, do: "+", else: "") <> elem(@names, if(k > 8, do: k - 8, else: k))
    if v > 0, do: s, else: String.downcase(s)
  end

  # ================================================================ moves

  @gold [{0, -1}, {-1, -1}, {1, -1}, {-1, 0}, {1, 0}, {0, 1}]
  @king [{0, -1}, {-1, -1}, {1, -1}, {-1, 0}, {1, 0}, {0, 1}, {-1, 1}, {1, 1}]

  # steps and slides for Sente; Gote mirrors dy
  defp steps(1), do: [{0, -1}]
  defp steps(3), do: [{-1, -2}, {1, -2}]
  defp steps(4), do: [{0, -1}, {-1, -1}, {1, -1}, {-1, 1}, {1, 1}]
  defp steps(5), do: @gold
  defp steps(8), do: @king
  defp steps(k) when k in [9, 10, 11, 12], do: @gold
  defp steps(14), do: [{0, -1}, {-1, 0}, {1, 0}, {0, 1}]
  defp steps(15), do: [{-1, -1}, {1, -1}, {-1, 1}, {1, 1}]
  defp steps(_), do: []
  defp slides(2), do: [{0, -1}]
  defp slides(k) when k in [6, 14], do: [{-1, -1}, {1, -1}, {-1, 1}, {1, 1}]
  defp slides(k) when k in [7, 15], do: [{0, -1}, {-1, 0}, {1, 0}, {0, 1}]
  defp slides(_), do: []

  defp targets(b, s, v) do
    side = if v > 0, do: 1, else: -1
    k = abs(v)
    {x, y} = {rem(s, 9), div(s, 9)}
    st = for {dx, dy} <- steps(k), (tx = x + dx) in 0..8, (ty = y + dy * side) in 0..8, elem(b, ty * 9 + tx) * side <= 0, do: ty * 9 + tx
    sl = Enum.flat_map(slides(k), fn {dx, dy} ->
      Enum.reduce_while(1..8, [], fn t, acc ->
        {tx, ty} = {x + t * dx, y + t * dy * side}
        if tx in 0..8 and ty in 0..8 do
          case elem(b, ty * 9 + tx) * side do
            0 -> {:cont, [ty * 9 + tx | acc]}
            q when q < 0 -> {:halt, [ty * 9 + tx | acc]}
            _ -> {:halt, acc}
          end
        else
          {:halt, acc}
        end
      end)
    end)
    st ++ sl
  end

  defp king_sq(b, side), do: Enum.find(0..80, &(elem(b, &1) == 8 * side))

  @doc "Is square `s` attacked by `side`?"
  def attacked?(b, s, side), do: Enum.any?(0..80, fn f -> (v = elem(b, f)) * side > 0 and s in targets(b, f, v) end)

  def in_check?(p), do: (k = king_sq(p.board, p.turn); k != nil and attacked?(p.board, k, -p.turn))

  # last ranks for a side: rank y "rows to go"
  defp rows_left(y, 1), do: y
  defp rows_left(y, -1), do: 8 - y
  defp zone?(y, side), do: rows_left(y, side) <= 2

  @doc "Legal moves: `{:move, from, to, promote?}` and `{:drop, kind, to}`."
  def moves(p) do
    p |> pseudo() |> Enum.filter(&legal?(p, &1))
  end

  defp pseudo(p) do
    b = p.board
    side = p.turn
    board_moves =
      for f <- 0..80, (v = elem(b, f)) * side > 0, t <- targets(b, f, v), m <- promos(abs(v), f, t, side), do: m
    pawn_files = for x <- 0..8, Enum.any?(0..8, &(elem(b, &1 * 9 + x) == side)), into: MapSet.new(), do: x
    drops =
      for {k, n} <- p.hands[side], n > 0, t <- 0..80, elem(b, t) == 0, drop_ok?(k, t, side, pawn_files), do: {:drop, k, t}
    board_moves ++ drops
  end

  defp promos(k, f, t, side) do
    {fy, ty} = {div(f, 9), div(t, 9)}
    can = k in [1, 2, 3, 4, 6, 7] and (zone?(fy, side) or zone?(ty, side))
    must = (k in [1, 2] and rows_left(ty, side) == 0) or (k == 3 and rows_left(ty, side) <= 1)
    cond do
      must -> [{:move, f, t, true}]
      can -> [{:move, f, t, true}, {:move, f, t, false}]
      true -> [{:move, f, t, false}]
    end
  end

  defp drop_ok?(k, t, side, pawn_files) do
    left = rows_left(div(t, 9), side)
    cond do
      k in [1, 2] and left == 0 -> false
      k == 3 and left <= 1 -> false
      k == 1 and MapSet.member?(pawn_files, rem(t, 9)) -> false
      true -> true
    end
  end

  defp legal?(p, m) do
    q = make(p, m)
    k = king_sq(q.board, p.turn)
    ok = k == nil or not attacked?(q.board, k, q.turn)
    # uchifuzume: a pawn drop may not deliver checkmate
    ok and not (match?({:drop, 1, _}, m) and in_check?(q) and pseudo(q) |> Enum.all?(fn r -> not legal_simple?(q, r) end))
  end

  defp legal_simple?(p, m) do
    q = make(p, m)
    k = king_sq(q.board, p.turn)
    k == nil or not attacked?(q.board, k, q.turn)
  end

  @doc "Play a move."
  def make(p, {:move, f, t, promote}) do
    b = p.board
    v = elem(b, f)
    cap = elem(b, t)
    side = p.turn
    hands = if cap != 0, do: (k = abs(cap); k = if k > 8, do: k - 8, else: k; update_in(p.hands, [side], &Map.update(&1, k, 1, fn n -> n + 1 end))), else: p.hands
    nv = if promote, do: v + 8 * side, else: v
    %__MODULE__{board: b |> put_elem(f, 0) |> put_elem(t, nv), turn: -side, hands: hands, ply: p.ply + 1}
  end

  def make(p, {:drop, k, t}) do
    side = p.turn
    hands = update_in(p.hands, [side], &Map.update!(&1, k, fn n -> n - 1 end))
    %__MODULE__{board: put_elem(p.board, t, k * side), turn: -side, hands: hands, ply: p.ply + 1}
  end

  def perft(p, 1), do: length(moves(p))
  def perft(p, d), do: p |> moves() |> Enum.reduce(0, fn m, acc -> acc + perft(make(p, m), d - 1) end)
  def perft_parallel(p, d) when d >= 2, do: p |> moves() |> Vapor.Play.pmap(fn m -> perft(make(p, m), d - 1) end) |> Enum.sum()
  def perft_parallel(p, d), do: perft(p, d)

  # ================================================================ USI

  def sq_name(s), do: "#{9 - rem(s, 9)}#{<<?a + div(s, 9)>>}"
  def sq(<<f, r>>), do: (r - ?a) * 9 + (9 - (f - ?0))

  def usi({:move, f, t, pr}), do: sq_name(f) <> sq_name(t) <> if(pr, do: "+", else: "")
  def usi({:drop, k, t}), do: elem(@names, k) <> "*" <> sq_name(t)

  def parse_move(p, text) do
    t = String.trim(text)
    case Enum.find(moves(p), &(usi(&1) == t)) do
      nil -> {:error, "illegal or unknown move #{inspect(text)}"}
      m -> {:ok, m}
    end
  end

  def status(p) do
    if moves(p) == [], do: (if in_check?(p), do: :checkmate, else: :no_moves), else: :ongoing
  end

  # a material evaluation for a simple searcher (pieces, hand pieces a little more)
  @value {0, 100, 300, 350, 500, 550, 800, 1000, 0, 550, 550, 550, 550, 0, 1100, 1300}
  def evaluate(p) do
    board = Enum.reduce(Tuple.to_list(p.board), 0, fn v, acc -> if v == 0, do: acc, else: acc + (if v > 0, do: 1, else: -1) * elem(@value, abs(v)) end)
    hand = Enum.reduce([1, -1], 0, fn side, acc -> acc + side * Enum.reduce(p.hands[side], 0, fn {k, n}, a -> a + n * round(elem(@value, k) * 1.1) end) end)
    (board + hand) * p.turn
  end

  @doc "Alpha–beta to `depth` with a node budget: `%{best, score, nodes}`."
  def search(p, opts \\ []) do
    depth = Keyword.get(opts, :depth, 3)
    Process.put(:shogi_nodes, 0)
    {score, best} = ab(p, depth, -1_000_000, 1_000_000, Keyword.get(opts, :nodes, 60_000))
    %{best: best && usi(best), score: score, nodes: Process.get(:shogi_nodes)}
  end

  defp ab(p, 0, _a, _b, _cap), do: {evaluate(p), nil}
  defp ab(p, d, a, b, cap) do
    Process.put(:shogi_nodes, Process.get(:shogi_nodes) + 1)
    ms = moves(p)
    cond do
      ms == [] -> {if(in_check?(p), do: -100_000, else: 0), nil}
      Process.get(:shogi_nodes) > cap -> {evaluate(p), hd(ms)}
      true ->
        ordered = Enum.sort_by(ms, fn
          {:move, _f, t, pr} -> -(abs(elem(p.board, t)) * 10 + if(pr, do: 5, else: 0))
          _ -> 0
        end)
        Enum.reduce_while(ordered, {-1_000_001, nil, a}, fn m, {best, bm, a} ->
          {s, _} = ab(make(p, m), d - 1, -b, -a, cap)
          s = -s
          {best, bm} = if s > best, do: {s, m}, else: {best, bm}
          a = max(a, s)
          if a >= b, do: {:halt, {best, bm, a}}, else: {:cont, {best, bm, a}}
        end)
        |> then(fn {s, m, _} -> {s, m} end)
    end
  end
end
