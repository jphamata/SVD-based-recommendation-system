defmodule Vapor.Finance.Book do
  @moduledoc """
  A limit order book with price–time priority, whose every session is a
  **verifiable object** (docs/FINANCAS.md §9).

  The pain: when an exchange, a broker's internaliser or a backtester
  says "your order was filled at 101.25 behind three others", nobody
  outside can check it — the matching logic is a black box and the
  backtest's simulator is not the production engine. Here:

  * the engine is a pure function `(book, event) → (book, reports)`;
    prices are integer ticks and quantities integers, so nothing rounds;
  * every event and its reports enter a **journal chained by SHA-256**
    over the canonical bytes (`Vapor.Canonical`), and the session closes
    with a **Merkle root** (RFC 6962 leaves) — any single fill can be
    proved to belong to the session without revealing the others;
  * `Vapor.Finance.Book.Check` re-runs the journal with a deliberately
    naive engine written separately (lists, sorting) and checks the
    invariants — conservation, no crossed book, limit prices respected,
    price–time priority, all-or-none for FOK, IOC never rests, post-only
    never takes, no self-trade — so the session is judged by code that
    shares nothing with the engine but the specification.

  Orders: limit or market; time in force GTC, IOC, FOK; post-only;
  cancel; modify (a quantity reduction at the same price keeps priority,
  anything else is cancel-and-replace and loses it). Self-trade
  prevention: `:cancel_taker` (default), `:cancel_resting` or `:off`.
  """
  alias Vapor.Canonical

  defstruct bids: :gb_trees.empty(), asks: :gb_trees.empty(), orders: %{}, seq: 0, ts: 0, stp: :cancel_taker,
            journal: [], head: <<0::256>>, last_trade: nil, volume: 0, trades: 0

  @type side :: :buy | :sell

  def new(opts \\ []), do: %__MODULE__{stp: Keyword.get(opts, :stp, :cancel_taker)}

  # ------------------------------------------------------------- top of book

  def best_bid(%{bids: b}), do: if(:gb_trees.is_empty(b), do: nil, else: elem(:gb_trees.largest(b), 0))
  def best_ask(%{asks: a}), do: if(:gb_trees.is_empty(a), do: nil, else: elem(:gb_trees.smallest(a), 0))

  @doc "Aggregated depth: `%{bids: [{price, qty, orders}…], asks: …}` best first, `levels` deep."
  def depth(book, levels \\ 10) do
    lv = fn tree, dir ->
      it = if dir == :desc, do: :gb_trees.to_list(tree) |> Enum.reverse(), else: :gb_trees.to_list(tree)
      it |> Enum.take(levels) |> Enum.map(fn {p, {q, ids}} -> {p, q, :queue.len(ids)} end)
    end
    %{bids: lv.(book.bids, :desc), asks: lv.(book.asks, :asc)}
  end

  @doc "The resting orders in priority order per side: `[{id, owner, price, qty}]`."
  def resting(book, side) do
    {tree, dir} = if side == :buy, do: {book.bids, :desc}, else: {book.asks, :asc}
    levels = if dir == :desc, do: Enum.reverse(:gb_trees.to_list(tree)), else: :gb_trees.to_list(tree)
    for {_p, {_, q}} <- levels, id <- :queue.to_list(q), o = book.orders[id], do: {id, o.owner, o.price, o.qty}
  end

  # ------------------------------------------------------------- the engine

  @doc """
  Apply one event; `{book, reports}`. Events are maps:
  `%{type: :new, id, owner, side, price (ticks, nil for market), qty, tif (:gtc | :ioc | :fok), post_only, ts}`,
  `%{type: :cancel, id}`, `%{type: :modify, id, price, qty}`.
  """
  def apply_event(book, ev) do
    book = %{book | seq: book.seq + 1, ts: Map.get(ev, :ts, book.ts)}
    {book, reps} = step(book, ev)
    entry = %{seq: book.seq, event: ev, reports: reps}
    h = :crypto.hash(:sha256, book.head <> Canonical.encode(entry))
    {%{book | journal: [Map.put(entry, :hash, h) | book.journal], head: h}, reps}
  end

  @doc "Apply a list of events; the book after them."
  def run(book, events), do: Enum.reduce(events, book, fn e, b -> elem(apply_event(b, e), 0) end)

  @doc "The journal in order, and its Merkle root over the entries' canonical bytes."
  def journal(book), do: Enum.reverse(book.journal)

  def merkle_root(book), do: book |> journal() |> Enum.map(&Vapor.Merkle.leaf(Canonical.encode(Map.delete(&1, :hash)))) |> Vapor.Merkle.root()

  defp step(book, %{type: :new} = ev) do
    cond do
      Map.has_key?(book.orders, ev.id) -> {book, [{:rejected, ev.id, :duplicate_id}]}
      not (is_integer(ev.qty) and ev.qty > 0) -> {book, [{:rejected, ev.id, :bad_quantity}]}
      ev.side not in [:buy, :sell] -> {book, [{:rejected, ev.id, :bad_side}]}
      Map.get(ev, :price) != nil and not (is_integer(ev.price) and ev.price > 0) -> {book, [{:rejected, ev.id, :bad_price}]}
      true -> new_order(book, ev)
    end
  end

  defp step(book, %{type: :cancel, id: id}) do
    case Map.fetch(book.orders, id) do
      {:ok, o} -> {remove(book, o), [{:cancelled, id, o.qty, :user}]}
      :error -> {book, [{:rejected, id, :unknown_order}]}
    end
  end

  defp step(book, %{type: :modify, id: id} = ev) do
    case Map.fetch(book.orders, id) do
      :error -> {book, [{:rejected, id, :unknown_order}]}
      {:ok, o} ->
        price = Map.get(ev, :price) || o.price
        qty = ev.qty
        cond do
          not (is_integer(qty) and qty > 0) -> {book, [{:rejected, id, :bad_quantity}]}
          price == o.price and qty <= o.qty ->
            # reduction in place: priority kept
            book = adjust_level(book, o.side, o.price, qty - o.qty)
            {%{book | orders: Map.put(book.orders, id, %{o | qty: qty})}, [{:modified, id, qty}]}
          true ->
            book = remove(book, o)
            {book, reps} = new_order(book, %{type: :new, id: id, owner: o.owner, side: o.side, price: price, qty: qty, tif: :gtc, post_only: false})
            {book, [{:cancelled, id, o.qty, :replaced} | reps]}
        end
    end
  end

  # written by the pre-trade gate: the refusal is part of the record, the book is untouched
  defp step(book, %{type: :risk_reject, id: id, reason: why}), do: {book, [{:risk_rejected, id, why}]}

  # a kill switch: every open order of the owner leaves, oldest first
  defp step(book, %{type: :kill, owner: ow}) do
    mine = book.orders |> Map.values() |> Enum.filter(&(&1.owner == ow)) |> Enum.sort_by(& &1.seq)
    Enum.reduce(mine, {book, []}, fn o, {b, reps} -> {remove(b, o), reps ++ [{:cancelled, o.id, o.qty, :kill}]} end)
  end

  defp step(book, ev), do: {book, [{:rejected, Map.get(ev, :id), :unknown_event}]}

  defp new_order(book, ev) do
    o = %{id: ev.id, owner: Map.get(ev, :owner, 0), side: ev.side, price: Map.get(ev, :price), qty: ev.qty, seq: book.seq,
          tif: Map.get(ev, :tif, :gtc), post_only: Map.get(ev, :post_only, false)}
    market = o.price == nil
    # a market order never rests: GTC becomes IOC; FOK stays all-or-none (the checker caught a market FOK degraded to IOC)
    tif = if market and o.tif == :gtc, do: :ioc, else: o.tif
    crosses = crosses?(book, o)
    cond do
      o.post_only and crosses -> {book, [{:cancelled, o.id, o.qty, :post_only}]}
      tif == :fok and fillable(book, o) < o.qty -> {book, [{:cancelled, o.id, o.qty, :fok}]}
      true ->
        {book, left, fills, stp_hit} = match(book, o, o.qty, [])
        reps = [{:accepted, o.id} | Enum.reverse(fills)]
        cond do
          left == 0 -> {book, reps}
          stp_hit and book.stp == :cancel_taker -> {book, reps ++ [{:cancelled, o.id, left, :stp}]}
          tif in [:ioc, :fok] -> {book, reps ++ [{:cancelled, o.id, left, :ioc_remainder}]}
          true -> {rest(book, %{o | qty: left}), reps ++ [{:rested, o.id, o.price, left}]}
        end
    end
  end

  defp crosses?(book, %{side: :buy, price: p}), do: (a = best_ask(book); a != nil and (p == nil or a <= p))
  defp crosses?(book, %{side: :sell, price: p}), do: (b = best_bid(book); b != nil and (p == nil or b >= p))

  # what a taker could get before hitting its limit or (with STP that cancels the taker) its own order
  defp fillable(book, o) do
    opp = if o.side == :buy, do: :sell, else: :buy
    resting(book, opp)
    |> Enum.reduce_while(0, fn {_id, owner, price, qty}, acc ->
      cond do
        acc >= o.qty -> {:halt, acc}
        not price_ok(o, price) -> {:halt, acc}
        owner == o.owner and book.stp == :cancel_taker -> {:halt, acc}
        owner == o.owner and book.stp == :cancel_resting -> {:cont, acc}
        true -> {:cont, acc + qty}
      end
    end)
  end

  defp price_ok(%{price: nil}, _), do: true
  defp price_ok(%{side: :buy, price: lim}, p), do: p <= lim
  defp price_ok(%{side: :sell, price: lim}, p), do: p >= lim

  defp match(book, _o, 0, fills), do: {book, 0, fills, false}

  defp match(book, o, left, fills) do
    {tree, best} = if o.side == :buy, do: {book.asks, best_ask(book)}, else: {book.bids, best_bid(book)}
    if best == nil or not price_ok(o, best) do
      {book, left, fills, false}
    else
      {_tot, q} = :gb_trees.get(best, tree)
      {:value, mid} = :queue.peek(q)
      m = book.orders[mid]
      cond do
        m.owner == o.owner and book.stp == :cancel_taker -> {book, left, fills, true}
        m.owner == o.owner and book.stp == :cancel_resting ->
          book = remove(book, m)
          match(book, o, left, [{:cancelled, m.id, m.qty, :stp} | fills])
        true ->
          x = min(left, m.qty)
          fill = {:fill, %{taker: o.id, maker: m.id, price: m.price, qty: x, taker_owner: o.owner, maker_owner: m.owner, taker_side: o.side}}
          book = if x == m.qty, do: remove(book, m), else: (b2 = adjust_level(book, m.side, m.price, -x); %{b2 | orders: Map.put(b2.orders, m.id, %{m | qty: m.qty - x})})
          book = %{book | last_trade: m.price, volume: book.volume + x, trades: book.trades + 1}
          match(book, o, left - x, [fill | fills])
      end
    end
  end

  defp rest(book, o) do
    key = if o.side == :buy, do: :bids, else: :asks
    tree = Map.fetch!(book, key)
    tree =
      case :gb_trees.lookup(o.price, tree) do
        {:value, {tot, q}} -> :gb_trees.update(o.price, {tot + o.qty, :queue.in(o.id, q)}, tree)
        :none -> :gb_trees.insert(o.price, {o.qty, :queue.from_list([o.id])}, tree)
      end
    book |> Map.put(key, tree) |> Map.put(:orders, Map.put(book.orders, o.id, %{o | seq: book.seq}))
  end

  defp remove(book, o) do
    key = if o.side == :buy, do: :bids, else: :asks
    tree = Map.fetch!(book, key)
    {tot, q} = :gb_trees.get(o.price, tree)
    q2 = :queue.filter(&(&1 != o.id), q)
    tree = if :queue.is_empty(q2), do: :gb_trees.delete(o.price, tree), else: :gb_trees.update(o.price, {tot - o.qty, q2}, tree)
    book |> Map.put(key, tree) |> Map.put(:orders, Map.delete(book.orders, o.id))
  end

  defp adjust_level(book, side, price, dq) do
    key = if side == :buy, do: :bids, else: :asks
    tree = Map.fetch!(book, key)
    {tot, q} = :gb_trees.get(price, tree)
    Map.put(book, key, :gb_trees.update(price, {tot + dq, q}, tree))
  end

  # ------------------------------------------------------------ a session

  @doc """
  Run a list of events and close the session: the journal's head hash,
  its Merkle root, the final depth, the trades and per-event latencies
  (µs, wall clock of this BEAM — reported as measured, not promised).
  """
  def session(events, opts \\ []) do
    book0 = new(opts)
    {book, lat} =
      Enum.reduce(events, {book0, []}, fn e, {b, l} ->
        t0 = System.monotonic_time(:nanosecond)
        {b2, _} = apply_event(b, e)
        {b2, [System.monotonic_time(:nanosecond) - t0 | l]}
      end)
    j = journal(book)
    trades = for %{reports: rs, seq: s} <- j, {:fill, f} <- rs, do: Map.put(f, :seq, s)
    %{book: book, journal: j, head: Base.encode16(book.head, case: :lower), merkle_root: Base.encode16(merkle_root(book), case: :lower),
      trades: trades, depth: depth(book), events: length(events), latency_ns: Enum.reverse(lat)}
  end

  @doc "Inclusion proof of journal entry `seq` (1-based) under the session's Merkle root."
  def prove(book, seq) do
    leaves = book |> journal() |> Enum.map(&Vapor.Merkle.leaf(Canonical.encode(Map.delete(&1, :hash))))
    %{seq: seq, leaf: Enum.at(leaves, seq - 1), proof: Vapor.Merkle.proof(leaves, seq - 1), root: Vapor.Merkle.root(leaves)}
  end
end
