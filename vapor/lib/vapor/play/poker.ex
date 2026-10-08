defmodule Vapor.Play.Poker do
  @moduledoc """
  Poker with imperfect information (docs/BOARDS.md §5): **Kuhn poker**
  (three cards) and **Leduc hold'em** (six cards, two betting rounds, a
  public card), solved by counterfactual regret minimisation (CFR+:
  regrets floored at zero, linearly weighted averages, alternating
  updates) — the method behind the programs that beat professionals at
  heads-up hold'em.

  The certificate of an equilibrium is its **exploitability**: the best
  response of each player to the other's average strategy, computed
  exactly over the whole tree. At zero, nobody can gain by deviating;
  `exploitability/2` measures it, in chips per hand, and the solver
  reports it as it falls. Kuhn's game value for the first player is
  −1/18 (Kuhn 1950); Leduc's is about −0.0856.

  A game module gives `deals/0` (`[{deal, probability}]`), `player/1`,
  `actions/1`, `terminal?/1`, `utility/2` (to player 0) and `infoset/3`;
  histories are strings of actions (`c` check/call, `b` bet/raise, `f`
  fold, `/` the end of a round).
  """

  # ================================================================== Kuhn

  defmodule Kuhn do
    @moduledoc "Kuhn poker: J < Q < K, ante 1, one bet of 1."
    def deals, do: for(a <- 0..2, b <- 0..2, a != b, do: {{a, b}, 1 / 6})
    def player(h), do: rem(String.length(h), 2)
    def actions(_h), do: ["c", "b"]
    def terminal?(h), do: h in ["cc", "bc", "bb", "cbc", "cbb"]
    def utility({a, b}, h) do
      win = if a > b, do: 1, else: -1
      case h do
        "cc" -> win
        "bc" -> 1
        "cbc" -> -1
        x when x in ["bb", "cbb"] -> 2 * win
      end * 1.0
    end
    def infoset({a, b}, h, p), do: "#{elem({"J", "Q", "K"}, if(p == 0, do: a, else: b))}#{h}"
  end

  # ================================================================= Leduc

  defmodule Leduc do
    @moduledoc """
    Leduc hold'em: J J Q Q K K; ante 1; round one raises of 2, round two
    of 4, at most two raises a round; a public card between the rounds;
    a pair with the board wins, else the higher card, ties split.
    """
    # deal: {p0 card, p1 card, public}, cards 0..5, rank = div(card, 2)
    def deals do
      all = for a <- 0..5, b <- 0..5, c <- 0..5, a != b, b != c, a != c, do: {a, b, c}
      Enum.map(all, &{&1, 1 / length(all)})
    end

    defp rounds(h), do: String.split(h, "/")
    def player(h), do: rem(String.length(List.last(rounds(h))), 2)
    def actions(h) do
      r = List.last(rounds(h))
      raises = r |> String.graphemes() |> Enum.count(&(&1 == "b"))
      facing = String.ends_with?(r, "b")
      cond do
        facing and raises >= 2 -> ["f", "c"]
        facing -> ["f", "c", "b"]
        true -> ["c", "b"]
      end
    end

    # a round is over after cc, or a call of a bet
    defp closed?(r), do: r == "cc" or (String.ends_with?(r, "c") and String.contains?(r, "b"))

    def terminal?(h) do
      rs = rounds(h)
      String.ends_with?(h, "f") or (length(rs) == 2 and closed?(List.last(rs)))
    end

    @doc false
    # after a closed first round the history continues with "/"
    def next(h, a) do
      h2 = h <> a
      if not String.contains?(h2, "/") and closed?(h2) and a != "f", do: h2 <> "/", else: h2
    end

    # chips each player has put in
    defp pot(h) do
      {c0, c1, _} =
        h |> rounds() |> Enum.with_index() |> Enum.reduce({1, 1, nil}, fn {r, k}, {c0, c1, _} ->
          size = if k == 0, do: 2, else: 4
          r |> String.graphemes() |> Enum.with_index() |> Enum.reduce({c0, c1, nil}, fn {a, i}, {c0, c1, _} ->
            p = rem(i, 2)
            {me, other} = if p == 0, do: {c0, c1}, else: {c1, c0}
            me = case a do "c" -> other; "b" -> other + size; "f" -> me end
            if p == 0, do: {me, c1, nil}, else: {c0, me, nil}
          end)
        end)
      {c0, c1}
    end

    def utility({a, b, c}, h) do
      {c0, c1} = pot(h)
      cond do
        String.ends_with?(h, "f") ->
          folder = player(String.slice(h, 0..-2//1))
          if folder == 0, do: -c0 * 1.0, else: c1 * 1.0
        true ->
          {ra, rb, rc} = {div(a, 2), div(b, 2), div(c, 2)}
          sa = if ra == rc, do: 10 + ra, else: ra
          sb = if rb == rc, do: 10 + rb, else: rb
          cond do sa > sb -> c1 * 1.0; sa < sb -> -c0 * 1.0; true -> 0.0 end
      end
    end

    def infoset({a, b, c}, h, p) do
      mine = elem({"J", "Q", "K"}, div(if(p == 0, do: a, else: b), 2))
      pub = if String.contains?(h, "/"), do: ":" <> elem({"J", "Q", "K"}, div(c, 2)), else: ""
      mine <> pub <> "|" <> h
    end
  end

  # ============================================================ the solver

  defp next(game, h, a), do: if(function_exported?(game, :next, 2), do: game.next(h, a), else: h <> a)

  @doc """
  Solve by CFR+ for `iterations`: `%{strategy, exploitability, value, curve}`
  (`curve`: exploitability at checkpoints, in chips per hand).
  """
  def solve(game, iterations \\ 1000, opts \\ []) do
    every = Keyword.get(opts, :every, max(div(iterations, 20), 1))
    deals = game.deals()

    {st, curve} =
      Enum.reduce(1..iterations, {%{regret: %{}, avg: %{}}, []}, fn t, {st, curve} ->
        # σᵗ stays fixed through the iteration: the regret changes are gathered, then applied (floored: CFR+)
        st = Enum.reduce([0, 1], st, fn p, st ->
          st = Enum.reduce(deals, Map.put(st, :delta, %{}), fn {d, pr}, st -> elem(cfr(game, d, "", p, 1.0, pr, st, t), 1) end)
          regret = Enum.reduce(st.delta, st.regret, fn {i, dm}, r ->
            old = Map.get(r, i, %{})
            Map.put(r, i, Map.new(dm, fn {a, x} -> {a, max(Map.get(old, a, 0.0) + x, 0.0)} end))
          end)
          %{st | regret: regret} |> Map.delete(:delta)
        end)
        curve = if rem(t, every) == 0 or t == iterations, do: [%{iteration: t, exploitability: exploitability(game, average(st.avg))} | curve], else: curve
        {st, curve}
      end)

    sigma = average(st.avg)
    %{strategy: sigma, exploitability: exploitability(game, sigma), value: value(game, sigma), curve: Enum.reverse(curve), iterations: iterations, infosets: map_size(sigma)}
  end

  defp current(st, i, acts) do
    r = Map.get(st.regret, i, %{})
    pos = Enum.map(acts, &max(Map.get(r, &1, 0.0), 0.0))
    s = Enum.sum(pos)
    if s > 0, do: Map.new(Enum.zip(acts, pos), fn {a, x} -> {a, x / s} end), else: Map.new(acts, &{&1, 1.0 / length(acts)})
  end

  # returns {value to player p, state}; pi_o: opponents' (and chance's) reach
  defp cfr(game, d, h, p, pi_p, pi_o, st, t) do
    if game.terminal?(h) do
      u = game.utility(d, h)
      {if(p == 0, do: u, else: -u), st}
    else
      who = game.player(h)
      acts = game.actions(h)
      i = game.infoset(d, h, who)
      sig = current(st, i, acts)
      if who == p do
        {vals, st} = Enum.map_reduce(acts, st, fn a, st -> cfr(game, d, next(game, h, a), p, pi_p * sig[a], pi_o, st, t) end)
        v = Enum.zip(acts, vals) |> Enum.reduce(0.0, fn {a, x}, acc -> acc + sig[a] * x end)
        reg = Map.get(st.delta, i, %{})
        reg = Enum.zip(acts, vals) |> Enum.reduce(reg, fn {a, x}, r -> Map.put(r, a, Map.get(r, a, 0.0) + pi_o * (x - v)) end)
        avg = Map.get(st.avg, i, %{})
        avg = Enum.reduce(acts, avg, fn a, m -> Map.put(m, a, Map.get(m, a, 0.0) + t * pi_p * sig[a]) end)
        {v, %{st | delta: Map.put(st.delta, i, reg), avg: Map.put(st.avg, i, avg)}}
      else
        {vals, st} = Enum.map_reduce(acts, st, fn a, st -> cfr(game, d, next(game, h, a), p, pi_p, pi_o * sig[a], st, t) end)
        {Enum.zip(acts, vals) |> Enum.reduce(0.0, fn {a, x}, acc -> acc + sig[a] * x end), st}
      end
    end
  end

  defp average(avg), do: Map.new(avg, fn {i, m} -> (s = Enum.sum(Map.values(m)); {i, Map.new(m, fn {a, x} -> {a, if(s > 0, do: x / s, else: 1.0 / map_size(m))} end)}) end)

  defp prob(sigma, i, a, acts), do: (case Map.get(sigma, i) do nil -> 1.0 / length(acts); m -> Map.get(m, a, 0.0) end)

  @doc "The expected value of a strategy profile to player 0."
  def value(game, sigma), do: Enum.reduce(game.deals(), 0.0, fn {d, pr}, acc -> acc + pr * ev(game, sigma, d, "") end)

  defp ev(game, sigma, d, h) do
    if game.terminal?(h), do: game.utility(d, h), else: (
      acts = game.actions(h)
      i = game.infoset(d, h, game.player(h))
      Enum.reduce(acts, 0.0, fn a, acc -> acc + prob(sigma, i, a, acts) * ev(game, sigma, d, next(game, h, a)) end))
  end

  @doc """
  Exploitability of `sigma` (chips per hand): the mean of the two best
  responses' gains, computed exactly — infosets of the responder settled
  deepest first, each choosing the action of highest counterfactual value.
  """
  def exploitability(game, sigma) do
    v0 = br_value(game, sigma, 0)
    v1 = br_value(game, sigma, 1)
    (v0 + v1) / 2
  end

  @doc false
  def br_value(game, sigma, i) do
    deals = game.deals()
    # every (deal, history) of player i's decisions, with the opponent's and chance's reach
    nodes = Enum.flat_map(deals, fn {d, pr} -> collect(game, sigma, d, "", pr, i) end)
    by_set = Enum.group_by(nodes, fn {d, h, _} -> game.infoset(d, h, i) end)
    order = by_set |> Map.keys() |> Enum.sort_by(fn set -> -(by_set[set] |> hd() |> elem(1) |> String.length()) end)

    br =
      Enum.reduce(order, %{}, fn set, br ->
        [{_, h0, _} | _] = by_set[set]
        acts = game.actions(h0)
        best = Enum.max_by(acts, fn a -> Enum.reduce(by_set[set], 0.0, fn {d, h, w}, acc -> acc + w * resp(game, sigma, br, d, next(game, h, a), i) end) end)
        Map.put(br, set, best)
      end)

    Enum.reduce(deals, 0.0, fn {d, pr}, acc -> acc + pr * resp(game, sigma, br, d, "", i) end)
  end

  defp collect(game, sigma, d, h, w, i) do
    if game.terminal?(h) do
      []
    else
      who = game.player(h)
      acts = game.actions(h)
      if who == i do
        [{d, h, w} | Enum.flat_map(acts, &collect(game, sigma, d, next(game, h, &1), w, i))]
      else
        set = game.infoset(d, h, who)
        Enum.flat_map(acts, fn a -> collect(game, sigma, d, next(game, h, a), w * prob(sigma, set, a, acts), i) end)
      end
    end
  end

  # value to player i of the best response `br` against sigma (unset infosets: none reached here by construction)
  defp resp(game, sigma, br, d, h, i) do
    if game.terminal?(h) do
      u = game.utility(d, h)
      if i == 0, do: u, else: -u
    else
      who = game.player(h)
      acts = game.actions(h)
      set = game.infoset(d, h, who)
      if who == i do
        case Map.fetch(br, set) do
          {:ok, a} -> resp(game, sigma, br, d, next(game, h, a), i)
          :error -> acts |> Enum.map(&resp(game, sigma, br, d, next(game, h, &1), i)) |> Enum.max()
        end
      else
        Enum.reduce(acts, 0.0, fn a, acc -> acc + prob(sigma, set, a, acts) * resp(game, sigma, br, d, next(game, h, a), i) end)
      end
    end
  end

  @doc "A uniformly random strategy (the control: highly exploitable)."
  def uniform, do: %{}
end
