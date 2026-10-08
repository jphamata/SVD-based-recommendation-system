# Boards and cards — rules pinned by published counts, play judged by perfect play

> Request (0.12), translated: "similar or superior to […] alpha zero (with comparisons)
> […] maybe some special care for chess, shogi and go and card
> games". Scrutiny: [DIRECTIVE.md §15](DIRECTIVE.md).

The real pain of whoever researches or teaches games is not the lack of
strong engines — they exist, and are free — it is **knowing that the rules
are right** and **measuring** an agent without depending on another program
as judge. That is why every game here is pinned by the published counts,
and every agent is judged against perfect play whenever it exists.

Console *Simulate → Boards & cards* · MCP `board_query`.

## 1. Chess (`Vapor.Play.Chess`)

Complete rules (castling through unattacked squares, en passant, promotion
to any piece, 50 moves, insufficient material, repetition), FEN and SAN
round trip, PGN. **Perft** equal to the published numbers at the initial
position and at the four standard positions (Kiwipete etc.);
**python-chess** agrees on legal moves and on perft(2) along random
games. Alpha–beta engine (iterative deepening, quiescence, transposition
table, MVV–LVA, *killers*, piece-square tables), ~76 thousand nodes/s on
one vCPU. **Mate prover**: returns a tree — every defence answered, every
leaf a checkmate — that `verify_mate/3` reproduces independently; a wrong
proof is rejected.

## 2. Shogi (`Vapor.Play.Shogi`)

The eight pieces and their six promotions, captured pieces that **change
sides and come back as drops** (no two unpromoted pawns on the same file,
no piece dropped where it could never move, no pawn drop that gives
mate), compulsory promotion, promotion zone; SFEN and USI. Perft
**30 · 900 · 25,470** from the initial position; **python-shogi** agrees
on legal moves along random games with drops and promotions.

## 3. Go (`Vapor.Play.Go`)

Tromp–Taylor rules (area scoring, komi), suicide rejected, **positional
superko**. Legal-position counts **1 · 57 · 12,675** on 1×1, 2×2, 3×3
(Tromp & Farnebäck). An MCTS player with *playouts* that avoid filling its
own eyes beats the random player on 5×5 (random against itself, the
control, draws on average).

## 4. k in a row (`Vapor.Play.MNK`)

Any m×n with k in a row, with or without gravity (Connect Four is 7×6,
k = 4, gravity), board symmetries. Solved **exactly** by negamax with a
transposition table up to 16 squares: 3×3×3 a draw, 4×3×3 and 4×4×3 with
gravity a first-player win.

## 5. Poker (`Vapor.Play.Poker`)

**CFR+** with the strategy frozen per iteration, and **exact
exploitability** by best response. Kuhn: game value −1/18 (−0.0556) and
exploitability < 10⁻³ chips/hand; uniform play (control) is exploitable by
0.458. Leduc hold'em (with the community card and two rounds):
exploitability < 0.02 in 100 iterations, value near −0.0856.

## 6. Generic self-play (`Vapor.Play.SelfPlay`)

The 0.11 learner (policy + value + PUCT, trained only on its own games)
generalised to **any game** that provides `features/1`, `actions/1` and
`index/2`, with data augmentation by the symmetries and Dirichlet noise at
the root. Judged against **all** the optimal lines of the perfect player,
on both sides: on 3×3×3, 300 training games, with 8 simulations the
trained network loses **24 of 114** lines; the same search without
training (control) loses **165 of 205**.

Honestly: the specialised 0.11 learner (`Vapor.Games`, retired in 0.16 in
favour of this one) lost 17 of 129 with 8 simulations — the generic one is a
little worse on the same game (21% against 13%), in exchange for serving
any game.

## 7. Comparison with what exists

| | here | Stockfish / YaneuraOu / KataGo / Lc0 |
|---|---|---|
| rules | pinned by perft and by another program (python-chess, python-shogi) | likewise (the community uses the same perft) |
| strength | teaching engine, ~76 k nodes/s, depth 4–6 | millions of nodes/s, networks trained on billions of positions; orders of magnitude stronger |
| proof | **mate tree checked** by an independent verifier | a move and an evaluation |
| judging agents | against **perfect play**, exhaustive, where it exists | against other engines (Elo) |
| poker | **exact** exploitability on Kuhn and Leduc | commercial/research solvers for full hold'em (much larger scale) |

What is equal or superior here: **verifiability** (perft, checked mate
proofs, exact exploitability, an exhaustive perfect judge). What is not:
playing strength at chess, shogi and full-size Go.

## 8. Honest limits

- No engine here is competitive with the leading open engines.
- Go: MCTS without a network plays well only on small boards; 19×19 is
  outside the console's budget (13×13 at most).
- Shogi: the engine is material + shallow search (depth ≤ 3).
- Poker: Kuhn and Leduc; full hold'em requires abstraction and another scale.
