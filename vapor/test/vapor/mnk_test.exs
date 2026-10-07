defmodule Vapor.MNKTest do
  use ExUnit.Case, async: true

  test "Connect Four: pieces fall to the lowest free cell, a full column leaves the legal moves, four stacked win" do
    alias Vapor.Play.MNK
    s = MNK.connect4()
    assert length(MNK.legal(s)) == 7
    col = fn s, c -> Enum.find(MNK.legal(s), &(div(&1, s.n) == c)) end
    # first piece in column 3 lands on row 0, the next on row 1
    assert col.(s, 3) == 3 * 6
    s1 = MNK.play(s, col.(s, 3))
    assert col.(s1, 3) == 3 * 6 + 1

    # player 1 stacks column 0, player 2 column 1: the fourth stacked piece wins
    won =
      Enum.reduce(1..7, s, fn k, st -> MNK.play(st, col.(st, if(rem(k, 2) == 1, do: 0, else: 1))) end)

    assert MNK.outcome(won) == -1.0

    full = Enum.reduce(1..6, s, fn _, st -> MNK.play(st, col.(st, 5)) end)
    refute Enum.any?(MNK.legal(full), &(div(&1, 6) == 5))
  end
end
