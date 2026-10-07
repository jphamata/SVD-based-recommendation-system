defmodule Vapor.Finance.PreTrade do
  @moduledoc """
  Pre-trade risk controls in front of the book (docs/FINANCAS.md §11) —
  what SEC Rule 15c3-5 and MiFID II RTS 6 ask a firm with market access
  to have, and to be able to show it had.

  Limits per owner (a map; `:default` applies to owners without one):

  | limit | rejects when |
  |---|---|
  | `max_qty` | the order's quantity exceeds it (fat finger) |
  | `max_notional` | quantity × price (ticks) exceeds it |
  | `collar` | the limit price is further than this fraction from the reference (last trade, else mid) |
  | `max_position` | the owner's worst-case position — filled + open orders on that side + this order — would exceed it |
  | `max_rate` | more than this many messages in any `window` (ns of event time) |
  | `kill` | the owner is switched off (`%{type: :kill, owner}` also cancels every open order) |

  A rejected order never reaches the book; it is written to the journal
  as `%{type: :risk_reject, …}` with its reason, so the audit trail shows
  the refusals too. `check/2` re-derives positions, open orders and rates
  from the journal and confirms that no accepted order broke a limit.
  """
  alias Vapor.Finance.Book

  @doc "A gate in front of an empty book (`opts` go to `Book.new/1`)."
  def init(opts \\ []), do: %{book: Book.new(opts), pos: %{}, open: %{}, rate: %{}, killed: MapSet.new()}

  @doc "Pass one event through the gate and, if admitted, the book: `{state, reports}`."
  def step(st, ev, limits) do
    st2 =
      case ev do
        %{type: :kill, owner: ow} -> %{apply_book(st, ev) | killed: MapSet.put(st.killed, ow)}
        %{type: :new} ->
          case gate(st, ev, limits) do
            :ok -> apply_book(note_rate(st, ev), ev)
            {:reject, why} -> reject(note_rate(st, ev), ev, why)
          end
        _ -> apply_book(st, ev)
      end
    {st2, hd(st2.book.journal).reports}
  end

  def session(events, limits, opts \\ []) do
    st = Enum.reduce(events, init(opts), fn ev, st -> elem(step(st, ev, limits), 0) end)
    j = Book.journal(st.book)
    %{book: st.book, journal: j, positions: st.pos, rejected: Enum.count(j, &(&1.event.type == :risk_reject)), head: Base.encode16(st.book.head, case: :lower)}
  end

  defp lim(limits, owner), do: Map.get(limits, owner) || Map.get(limits, :default, %{})

  defp note_rate(st, ev), do: %{st | rate: Map.update(st.rate, Map.get(ev, :owner), [Map.get(ev, :ts, 0)], &[Map.get(ev, :ts, 0) | Enum.take(&1, 1000)])}

  defp reference(book) do
    case {book.last_trade, Book.best_bid(book), Book.best_ask(book)} do
      {p, _, _} when p != nil -> p
      {nil, b, a} when b != nil and a != nil -> (b + a) / 2
      _ -> nil
    end
  end

  @doc false
  def gate(st, ev, limits) do
    l = lim(limits, ev.owner)
    open_side = st.open |> Map.get({ev.owner, ev.side}, 0)
    pos = Map.get(st.pos, ev.owner, 0)
    worst = if ev.side == :buy, do: pos + open_side + ev.qty, else: -pos + open_side + ev.qty
    ref = reference(st.book)
    win = Map.get(l, :window, 1_000_000_000)
    recent = Map.get(st.rate, ev.owner, []) |> Enum.count(&(Map.get(ev, :ts, 0) - &1 < win))
    cond do
      MapSet.member?(st.killed, ev.owner) -> {:reject, :kill_switch}
      l[:max_qty] && ev.qty > l.max_qty -> {:reject, :max_qty}
      l[:max_notional] && ev.price && ev.qty * ev.price > l.max_notional -> {:reject, :max_notional}
      l[:collar] && ev.price && ref && abs(ev.price - ref) > l.collar * ref -> {:reject, :price_collar}
      l[:collar] && ev.price == nil && ref == nil -> {:reject, :no_reference_for_market_order}
      l[:max_position] && worst > l.max_position -> {:reject, :max_position}
      l[:max_rate] && recent >= l.max_rate -> {:reject, :message_rate}
      true -> :ok
    end
  end

  defp reject(st, ev, why) do
    {book, _} = Book.apply_event(st.book, %{type: :risk_reject, id: ev.id, owner: ev.owner, reason: why, order: Map.delete(ev, :type)})
    %{st | book: book}
  end

  defp apply_book(st, ev) do
    {book, reps} = Book.apply_event(st.book, ev)
    pos = Enum.reduce(reps, st.pos, fn
      {:fill, f}, p ->
        s = if f.taker_side == :buy, do: 1, else: -1
        p |> Map.update(f.taker_owner, s * f.qty, &(&1 + s * f.qty)) |> Map.update(f.maker_owner, -s * f.qty, &(&1 - s * f.qty))
      _, p -> p
    end)
    %{st | book: book, pos: pos, open: open_qty(book.orders)}
  end

  defp open_qty(orders), do: Enum.reduce(orders, %{}, fn {_, o}, m -> Map.update(m, {o.owner, o.side}, o.qty, &(&1 + o.qty)) end)

  @doc """
  The auditor's check: replay the journal, recompute positions, open
  orders, references and message counts, and confirm every **accepted**
  new order respected its owner's limits and every refusal had a reason
  that held. `%{ok, accepted, refused, violations}`.
  """
  def check(journal, limits, opts \\ []) do
    st0 = init(opts)
    {_, acc, ref, viol} =
      Enum.reduce(journal, {st0, 0, 0, []}, fn %{event: ev, seq: seq}, {st, a, r, v} ->
        case ev do
          %{type: :risk_reject, order: o, reason: why} ->
            o = Map.put(o, :type, :new)
            held = gate(st, o, limits) == {:reject, why}
            st = note_rate(st, o)
            {%{st | book: elem(Book.apply_event(st.book, ev), 0)}, a, r + 1, if(held, do: v, else: [{seq, :refusal_without_cause, why} | v])}
          %{type: :new} ->
            ok = gate(st, ev, limits) == :ok
            st = note_rate(st, ev)
            {apply_book(st, ev), a + 1, r, if(ok, do: v, else: [{seq, :accepted_over_limit, elem(gate(st, ev, limits), 1)} | v])}
          %{type: :kill, owner: ow} -> {%{apply_book(st, ev) | killed: MapSet.put(st.killed, ow)}, a, r, v}
          _ -> {apply_book(st, ev), a, r, v}
        end
      end)
    %{ok: viol == [], accepted: acc, refused: ref, violations: Enum.reverse(viol)}
  end
end
