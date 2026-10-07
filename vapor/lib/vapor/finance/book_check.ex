defmodule Vapor.Finance.Book.Check do
  @moduledoc """
  The independent judge of an order-book session (docs/FINANCAS.md §9).

  It shares nothing with `Vapor.Finance.Book` but the specification: the
  book is a plain list of resting orders, priority is a sort by (price,
  time), every match scans the list. From the journal alone it

  1. recomputes the SHA-256 chain (`hᵢ = H(hᵢ₋₁ ‖ canonical(entryᵢ))`);
  2. replays every event in this naive engine and demands the **same
     reports**, in the same order (the first divergence is named);
  3. checks the invariants on the journal itself: each fill at the
     maker's price and within the taker's limit, the maker the first in
     price–time priority at that moment, quantities conserved (filled +
     rested + cancelled = ordered), no crossed book after any event, FOK
     all-or-none, IOC and market orders never resting, post-only never
     taking, no fill between orders of the same owner under self-trade
     prevention.

  Differential fuzzing (`financas_test.exs`) runs both on thousands of
  random event streams.
  """
  alias Vapor.Canonical

  @doc "Check a journal (entries with `seq, event, reports, hash`). `%{ok, checked, failures}`."
  def check(journal, opts \\ []) do
    stp = Keyword.get(opts, :stp, :cancel_taker)
    {_, chain_bad} =
      Enum.reduce(journal, {<<0::256>>, nil}, fn e, {h, bad} ->
        h2 = :crypto.hash(:sha256, h <> Canonical.encode(Map.delete(e, :hash)))
        {h2, bad || if(h2 != e.hash, do: e.seq)}
      end)
    {_, failures} =
      Enum.reduce(journal, {[], []}, fn e, {state, fails} ->
        {state2, reps} = naive(state, e.event, e.seq, stp)
        f1 = if reps != e.reports, do: [{e.seq, :reports_differ, %{journal: e.reports, reference: reps}}], else: []
        f2 = invariants(state, state2, e, stp)
        {state2, fails ++ f1 ++ f2}
      end)
    failures = if chain_bad, do: [{chain_bad, :hash_chain_broken, nil} | failures], else: failures
    %{ok: failures == [], checked: length(journal), failures: Enum.take(failures, 20),
      verdict: if(failures == [], do: "the chain holds; a naive engine reproduces every report; every invariant holds", else: "#{length(failures)} failure(s); the first at event #{elem(hd(failures), 0)}: #{elem(hd(failures), 1)}")}
  end

  # ------------------------------------------------------- the naive engine

  defp opp(:buy), do: :sell
  defp opp(:sell), do: :buy

  defp queue(state, side) do
    state |> Enum.filter(&(&1.side == side)) |> Enum.sort_by(fn o -> {if(side == :buy, do: -o.price, else: o.price), o.time} end)
  end

  defp ok_price(nil, _, _), do: true
  defp ok_price(lim, :buy, p), do: p <= lim
  defp ok_price(lim, :sell, p), do: p >= lim

  @doc false
  def naive(state, %{type: :new} = ev, seq, stp) do
    price = Map.get(ev, :price)
    cond do
      Enum.any?(state, &(&1.id == ev.id)) -> {state, [{:rejected, ev.id, :duplicate_id}]}
      not (is_integer(ev.qty) and ev.qty > 0) -> {state, [{:rejected, ev.id, :bad_quantity}]}
      ev.side not in [:buy, :sell] -> {state, [{:rejected, ev.id, :bad_side}]}
      price != nil and not (is_integer(price) and price > 0) -> {state, [{:rejected, ev.id, :bad_price}]}
      true -> naive_new(state, %{id: ev.id, owner: Map.get(ev, :owner, 0), side: ev.side, price: price, qty: ev.qty,
                                 tif: (case {price, Map.get(ev, :tif, :gtc)} do {nil, :gtc} -> :ioc; {_, t} -> t end), post_only: Map.get(ev, :post_only, false)}, seq, stp)
    end
  end

  def naive(state, %{type: :cancel, id: id}, _seq, _stp) do
    case Enum.find(state, &(&1.id == id)) do
      nil -> {state, [{:rejected, id, :unknown_order}]}
      o -> {List.delete(state, o), [{:cancelled, id, o.qty, :user}]}
    end
  end

  def naive(state, %{type: :modify, id: id} = ev, seq, stp) do
    case Enum.find(state, &(&1.id == id)) do
      nil -> {state, [{:rejected, id, :unknown_order}]}
      o ->
        price = Map.get(ev, :price) || o.price
        cond do
          not (is_integer(ev.qty) and ev.qty > 0) -> {state, [{:rejected, id, :bad_quantity}]}
          price == o.price and ev.qty <= o.qty -> {Enum.map(state, &(if &1.id == id, do: %{&1 | qty: ev.qty}, else: &1)), [{:modified, id, ev.qty}]}
          true ->
            {s2, reps} = naive_new(List.delete(state, o), %{id: id, owner: o.owner, side: o.side, price: price, qty: ev.qty, tif: :gtc, post_only: false}, seq, stp)
            {s2, [{:cancelled, id, o.qty, :replaced} | reps]}
        end
    end
  end

  def naive(state, %{type: :risk_reject, id: id, reason: why}, _seq, _stp), do: {state, [{:risk_rejected, id, why}]}

  def naive(state, %{type: :kill, owner: ow}, _seq, _stp) do
    mine = state |> Enum.filter(&(&1.owner == ow)) |> Enum.sort_by(& &1.time)
    {state -- mine, Enum.map(mine, &{:cancelled, &1.id, &1.qty, :kill})}
  end

  def naive(state, ev, _seq, _stp), do: {state, [{:rejected, Map.get(ev, :id), :unknown_event}]}

  defp naive_new(state, o, seq, stp) do
    book = queue(state, opp(o.side))
    crossing = Enum.filter(book, &ok_price(o.price, o.side, &1.price))
    # what the taker could get: in priority order, until its limit or (cancel_taker) its own order
    avail = Enum.reduce_while(crossing, 0, fn m, acc ->
      cond do
        acc >= o.qty -> {:halt, acc}
        m.owner == o.owner and stp == :cancel_taker -> {:halt, acc}
        m.owner == o.owner and stp == :cancel_resting -> {:cont, acc}
        true -> {:cont, acc + m.qty}
      end
    end)
    cond do
      o.post_only and crossing != [] -> {state, [{:cancelled, o.id, o.qty, :post_only}]}
      o.tif == :fok and avail < o.qty -> {state, [{:cancelled, o.id, o.qty, :fok}]}
      true ->
        {state, left, reps, stp_hit} = naive_match(state, o, o.qty, [], stp)
        reps = [{:accepted, o.id} | Enum.reverse(reps)]
        cond do
          left == 0 -> {state, reps}
          stp_hit and stp == :cancel_taker -> {state, reps ++ [{:cancelled, o.id, left, :stp}]}
          o.tif in [:ioc, :fok] -> {state, reps ++ [{:cancelled, o.id, left, :ioc_remainder}]}
          true -> {state ++ [%{id: o.id, owner: o.owner, side: o.side, price: o.price, qty: left, time: seq}], reps ++ [{:rested, o.id, o.price, left}]}
        end
    end
  end

  defp naive_match(state, _o, 0, reps, _stp), do: {state, 0, reps, false}
  defp naive_match(state, o, left, reps, stp) do
    case queue(state, opp(o.side)) |> Enum.filter(&ok_price(o.price, o.side, &1.price)) do
      [] -> {state, left, reps, false}
      [m | _] ->
        cond do
          m.owner == o.owner and stp == :cancel_taker -> {state, left, reps, true}
          m.owner == o.owner and stp == :cancel_resting -> naive_match(List.delete(state, m), o, left, [{:cancelled, m.id, m.qty, :stp} | reps], stp)
          true ->
            x = min(left, m.qty)
            fill = {:fill, %{taker: o.id, maker: m.id, price: m.price, qty: x, taker_owner: o.owner, maker_owner: m.owner, taker_side: o.side}}
            state = if x == m.qty, do: List.delete(state, m), else: Enum.map(state, &(if &1.id == m.id, do: %{&1 | qty: &1.qty - x}, else: &1))
            naive_match(state, o, left - x, [fill | reps], stp)
        end
    end
  end

  # ------------------------------------------------------------ invariants

  defp invariants(before, after_state, %{seq: seq, event: ev, reports: reps}, stp) do
    fills = for {:fill, f} <- reps, do: f
    bids = for o <- after_state, o.side == :buy, do: o.price
    asks = for o <- after_state, o.side == :sell, do: o.price
    crossed = bids != [] and asks != [] and Enum.max(bids) >= Enum.min(asks)
    limit_ok = Enum.all?(fills, fn f -> lim = Map.get(ev, :price); lim == nil or (if f.taker_side == :buy, do: f.price <= lim, else: f.price >= lim) end)
    makers_ok =
      fills |> Enum.reduce_while({before, true}, fn f, {st, _} ->
        q = queue(st, opp(f.taker_side)) |> Enum.reject(&(&1.owner == f.taker_owner and stp == :cancel_resting))
        case q do
          [m | _] when m.id == f.maker and m.price == f.price ->
            st = if f.qty == m.qty, do: List.delete(st, m), else: Enum.map(st, &(if &1.id == m.id, do: %{&1 | qty: &1.qty - f.qty}, else: &1))
            {:cont, {st, true}}
          _ -> {:halt, {st, false}}
        end
      end) |> elem(1)
    taker_id = Map.get(ev, :id)
    ordered = if ev.type in [:new, :modify], do: ev.qty, else: nil
    filled = fills |> Enum.filter(&(&1.taker == taker_id)) |> Enum.map(& &1.qty) |> Enum.sum()
    rested = Enum.find_value(reps, 0, fn {:rested, ^taker_id, _, q} -> q; _ -> nil end)
    cancelled_rem = Enum.find_value(reps, 0, fn {:cancelled, ^taker_id, q, r} when r != :replaced and r != :user -> q; _ -> nil end)
    accepted = Enum.any?(reps, &match?({:accepted, ^taker_id}, &1))
    conserve = ordered == nil or not accepted or filled + rested + cancelled_rem == ordered
    fok_ok = Map.get(ev, :tif) != :fok or filled == 0 or filled == ordered
    ioc_ok = not (Map.get(ev, :tif) in [:ioc, :fok] or (ev.type == :new and Map.get(ev, :price) == nil)) or rested == 0
    post_ok = not Map.get(ev, :post_only, false) or fills == []
    self_ok = stp == :off or Enum.all?(fills, &(&1.taker_owner != &1.maker_owner))
    pos_ok = Enum.all?(fills, &(&1.qty > 0))
    [{crossed, :crossed_book}, {not limit_ok, :limit_violated}, {not makers_ok, :priority_violated}, {not conserve, :quantity_not_conserved},
     {not fok_ok, :fok_partial}, {not ioc_ok, :ioc_rested}, {not post_ok, :post_only_took}, {not self_ok, :self_trade}, {not pos_ok, :empty_fill}]
    |> Enum.filter(&elem(&1, 0)) |> Enum.map(&{seq, elem(&1, 1), nil})
  end
end
