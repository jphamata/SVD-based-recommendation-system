defmodule Vapor.Play.Go do
  @moduledoc """
  Go (docs/BOARDS.md §3) on any board up to 19×19: captures, no
  suicide (or Tromp–Taylor's suicide allowed, by option), **positional
  superko**, passes, area scoring with komi — Tromp & Taylor's
  formalisation of the rules, the one the computer-Go literature counts
  with. The rules are pinned by Tromp & Farnebäck's count of legal
  positions (every chain with a liberty): 1 on 1×1, 57 on 2×2, 12 675 on
  3×3 — `legal_positions/1` enumerates them.

  A state: `%{n, board (tuple, 1 black, −1 white, 0 empty), turn, passes,
  history (positions seen), komi, suicide, moves}`; a move is a point
  index or `:pass`.
  """

  def new(n, opts \\ []) when n in 1..19 do
    b = Tuple.duplicate(0, n * n)
    %{n: n, board: b, turn: 1, passes: 0, history: MapSet.new([b]), komi: Keyword.get(opts, :komi, 0.0), suicide: Keyword.get(opts, :suicide, false), moves: 0,
      max_moves: Keyword.get(opts, :max_moves, 3 * n * n)}
  end

  defp nbrs(n, i) do
    {x, y} = {rem(i, n), div(i, n)}
    for {dx, dy} <- [{1, 0}, {-1, 0}, {0, 1}, {0, -1}], x + dx in 0..(n - 1), y + dy in 0..(n - 1), do: (y + dy) * n + x + dx
  end

  # the chain through i and its liberties
  defp chain(b, n, i) do
    c = elem(b, i)
    grow(b, n, c, [i], MapSet.new([i]), MapSet.new())
  end

  defp grow(_b, _n, _c, [], stones, libs), do: {stones, libs}
  defp grow(b, n, c, [i | rest], stones, libs) do
    {stack, stones, libs} =
      Enum.reduce(nbrs(n, i), {rest, stones, libs}, fn j, {st, ss, ls} ->
        v = elem(b, j)
        cond do
          v == 0 -> {st, ss, MapSet.put(ls, j)}
          v == c and not MapSet.member?(ss, j) -> {[j | st], MapSet.put(ss, j), ls}
          true -> {st, ss, ls}
        end
      end)
    grow(b, n, c, stack, stones, libs)
  end

  @doc "Place a stone (no legality beyond occupancy): `{board, captured}` or `:suicide`."
  def place(b, n, i, color, suicide_ok \\ false) do
    b = put_elem(b, i, color)
    {b, cap} =
      Enum.reduce(nbrs(n, i), {b, 0}, fn j, {b, cap} ->
        if elem(b, j) == -color do
          {stones, libs} = chain(b, n, j)
          if MapSet.size(libs) == 0, do: {Enum.reduce(stones, b, &put_elem(&2, &1, 0)), cap + MapSet.size(stones)}, else: {b, cap}
        else
          {b, cap}
        end
      end)
    {own, libs} = chain(b, n, i)
    cond do
      MapSet.size(libs) > 0 -> {b, cap}
      suicide_ok -> {Enum.reduce(own, b, &put_elem(&2, &1, 0)), cap}
      true -> :suicide
    end
  end

  # ------------------------------------------------------- game interface

  def legal(%{} = s) do
    pts = for i <- 0..(s.n * s.n - 1), elem(s.board, i) == 0, ok?(s, i), do: i
    pts ++ [:pass]
  end

  defp ok?(s, i) do
    case place(s.board, s.n, i, s.turn, s.suicide) do
      :suicide -> false
      {b, _} -> not MapSet.member?(s.history, b)
    end
  end

  def play(s, :pass), do: %{s | turn: -s.turn, passes: s.passes + 1, moves: s.moves + 1}
  def play(s, i) do
    {b, _} = place(s.board, s.n, i, s.turn, s.suicide)
    %{s | board: b, turn: -s.turn, passes: 0, history: MapSet.put(s.history, b), moves: s.moves + 1}
  end

  def outcome(s) do
    if s.passes >= 2 or s.moves >= s.max_moves do
      d = score(s).margin
      cond do d > 0 -> 1.0 * s.turn; d < 0 -> -1.0 * s.turn; true -> 0.0 end
    end
  end

  def key(s), do: {s.board, s.turn, s.passes}

  @doc "Area score (Tromp–Taylor): stones plus empty regions that reach only one colour; `margin` = black − white − komi."
  def score(s) do
    n = s.n
    {terr, _} =
      Enum.reduce(0..(n * n - 1), {%{1 => 0, -1 => 0}, MapSet.new()}, fn i, {t, seen} ->
        if elem(s.board, i) == 0 and not MapSet.member?(seen, i) do
          {region, borders} = region(s.board, n, i)
          seen = MapSet.union(seen, region)
          case MapSet.to_list(borders) do
            [c] -> {Map.update!(t, c, &(&1 + MapSet.size(region))), seen}
            _ -> {t, seen}
          end
        else
          {t, seen}
        end
      end)
    stones = Tuple.to_list(s.board)
    black = Enum.count(stones, &(&1 == 1)) + terr[1]
    white = Enum.count(stones, &(&1 == -1)) + terr[-1]
    %{black: black, white: white, komi: s.komi, margin: black - white - s.komi}
  end

  defp region(b, n, i) do
    walk(b, n, [i], MapSet.new([i]), MapSet.new())
  end

  defp walk(_b, _n, [], reg, bord), do: {reg, bord}
  defp walk(b, n, [i | rest], reg, bord) do
    {st, reg, bord} = Enum.reduce(nbrs(n, i), {rest, reg, bord}, fn j, {st, r, bd} ->
      case elem(b, j) do
        0 -> if MapSet.member?(r, j), do: {st, r, bd}, else: {[j | st], MapSet.put(r, j), bd}
        c -> {st, r, MapSet.put(bd, c)}
      end
    end)
    walk(b, n, st, reg, bord)
  end

  # random playouts must not fill their own eyes, or no game ends: a point all of whose neighbours are own stones
  @doc "A playout-friendly move list: the legal moves minus own one-point eyes (pass only when nothing else)."
  def sensible(s) do
    moves = legal(s) -- [:pass]
    good = Enum.reject(moves, fn i -> Enum.all?(nbrs(s.n, i), &(elem(s.board, &1) == s.turn)) end)
    if good == [], do: [:pass], else: good
  end

  @doc "Positions of an n×n board where every chain has a liberty (Tromp & Farnebäck: 1, 57, 12 675 for n = 1, 2, 3)."
  def legal_positions(n) when n <= 3 do
    cells = n * n
    Enum.count(0..(Integer.pow(3, cells) - 1), fn code ->
      b = digits(code, cells) |> List.to_tuple()
      Enum.all?(0..(cells - 1), fn i -> elem(b, i) == 0 or MapSet.size(elem(chain(b, n, i), 1)) > 0 end)
    end)
  end

  defp digits(code, k), do: Enum.map(0..(k - 1), fn j -> case rem(div(code, Integer.pow(3, j)), 3) do 0 -> 0; 1 -> 1; 2 -> -1 end end)

  @doc "The board as text (X black, O white)."
  def show(s) do
    for y <- (s.n - 1)..0//-1 do
      for(x <- 0..(s.n - 1), do: case elem(s.board, y * s.n + x) do 1 -> "X"; -1 -> "O"; 0 -> "." end) |> Enum.join(" ")
    end |> Enum.join("\n")
  end
end

defmodule Vapor.Play.Go.Playouts do
  @moduledoc false
  # Go with the eye-avoiding playout move list, for the generic MCTS
  alias Vapor.Play.Go
  def legal(s), do: if(s.moves < 2 * s.n * s.n, do: Go.sensible(s), else: [:pass])
  def play(s, m), do: Go.play(s, m)
  def outcome(s), do: Go.outcome(s)
  def key(s), do: Go.key(s)
end
