defmodule Vapor.Play.MNK do
  @moduledoc """
  The m,n,k family (docs/BOARDS.md §4): k in a row on an m×n board,
  with or without gravity — tic-tac-toe is (3, 3, 3), Connect Four is
  (7, 6, 4) with gravity, gomoku (15, 15, 5). Any member is one line:
  `new(4, 4, 3, gravity: true)`. Small members are **solved** exactly by
  `Vapor.Play.solve/2`; the known values pin the implementation (3,3,3 a
  draw; 4,3,3 a first-player win; 4,4,4 a draw).

  The board is a tuple, column-major from the bottom-left: cell = col·n + row.
  """

  def new(m, n, k, opts \\ []) when m in 1..19 and n in 1..19 and k >= 1 do
    %{m: m, n: n, k: k, gravity: Keyword.get(opts, :gravity, false), board: Tuple.duplicate(0, m * n), turn: 1, last: nil, filled: 0}
  end

  def tictactoe, do: new(3, 3, 3)
  def connect4, do: new(7, 6, 4, gravity: true)

  def legal(s) do
    if s.gravity do
      for c <- 0..(s.m - 1), r = Enum.find(0..(s.n - 1), &(elem(s.board, c * s.n + &1) == 0)), r != nil, do: c * s.n + r
    else
      for i <- 0..(s.m * s.n - 1), elem(s.board, i) == 0, do: i
    end
  end

  def play(s, i), do: %{s | board: put_elem(s.board, i, s.turn), turn: -s.turn, last: i, filled: s.filled + 1}

  @doc "Value for the player to move: −1 if the previous move completed k in a row, 0 on a full board, else nil."
  def outcome(%{last: nil}), do: nil
  def outcome(s) do
    cond do
      line?(s, s.last) -> -1.0
      s.filled == s.m * s.n -> 0.0
      true -> nil
    end
  end

  def key(s), do: {s.board, s.turn}

  # centre-first ordering helps alpha–beta
  def order(s, moves) do
    {cm, cn} = {(s.m - 1) / 2, (s.n - 1) / 2}
    Enum.sort_by(moves, fn i -> abs(div(i, s.n) - cm) + abs(rem(i, s.n) - cn) end)
  end

  defp line?(s, i) do
    c = elem(s.board, i)
    {x, y} = {div(i, s.n), rem(i, s.n)}
    Enum.any?([{1, 0}, {0, 1}, {1, 1}, {1, -1}], fn {dx, dy} -> 1 + run(s, x, y, dx, dy, c) + run(s, x, y, -dx, -dy, c) >= s.k end)
  end

  defp run(s, x, y, dx, dy, c) do
    Enum.reduce_while(1..s.k, 0, fn t, acc ->
      {a, b} = {x + t * dx, y + t * dy}
      if a in 0..(s.m - 1) and b in 0..(s.n - 1) and elem(s.board, a * s.n + b) == c, do: {:cont, acc + 1}, else: {:halt, acc}
    end)
  end

  # ---------------------------------------------------- for self-play nets

  @doc "Features from the mover's side: own stones, then the opponent's."
  def features(s), do: (l = Tuple.to_list(s.board); Enum.map(l, &if(&1 == s.turn, do: 1.0, else: 0.0)) ++ Enum.map(l, &if(&1 == -s.turn, do: 1.0, else: 0.0)))
  def actions(s), do: s.m * s.n
  def index(_s, i), do: i

  @doc """
  The board's symmetries as cell permutations (new cell i takes old cell
  perm[i]): the eight of the square without gravity on a square board,
  the four of the rectangle without gravity, the mirror with gravity.
  """
  def symmetries(s) do
    {m, n} = {s.m, s.n}
    at = fn c, r -> c * n + r end
    id = for c <- 0..(m - 1), r <- 0..(n - 1), do: {c, r}
    maps =
      [fn {c, r} -> {c, r} end, fn {c, r} -> {m - 1 - c, r} end] ++
      if(s.gravity, do: [], else: [fn {c, r} -> {c, n - 1 - r} end, fn {c, r} -> {m - 1 - c, n - 1 - r} end] ++
        if(m == n, do: [fn {c, r} -> {r, c} end, fn {c, r} -> {n - 1 - r, c} end, fn {c, r} -> {r, m - 1 - c} end, fn {c, r} -> {n - 1 - r, m - 1 - c} end], else: []))
    for f <- maps, do: Enum.map(id, fn cr -> (({c2, r2} = f.(cr)); at.(c2, r2)) end)
  end

  @doc "A state and a move distribution under a symmetry."
  def transform(s, pi, perm) do
    inv = perm |> Enum.with_index() |> Map.new()
    b = perm |> Enum.map(&elem(s.board, &1)) |> List.to_tuple()
    {%{s | board: b, last: s.last && inv[s.last]}, Map.new(pi, fn {mv, p} -> {inv[mv], p} end)}
  end

  @doc "The board as text."
  def show(s) do
    for r <- (s.n - 1)..0//-1 do
      for(c <- 0..(s.m - 1), do: case elem(s.board, c * s.n + r) do 1 -> "X"; -1 -> "O"; _ -> "." end) |> Enum.join(" ")
    end |> Enum.join("\n")
  end
end
