defmodule Vapor.Finance.Exchange do
  @moduledoc """
  An exchange session simulated by agents through the **same** gate and
  engine a production deployment would use (docs/FINANCE.md §13) — the
  backtest is not a model of the venue, it is the venue's code.

  Agents: market makers quoting post-only around an inventory-skewed
  reservation price (Avellaneda–Stoikov's rule), noise takers whose
  arrivals follow a Hawkes process, and an informed trader who knows the
  fundamental and trades when the book strays from it. Every order passes
  `Vapor.Finance.PreTrade` and `Vapor.Finance.Book`.

  The session returns its own audit: the journal checked by the naive
  engine (`Book.Check`), the pre-trade limits checked from the journal
  (`PreTrade.check`), the ITCH feed rebuilt into the same book
  (`Itch.consistent?`), the same seed replayed to the same head hash; and
  the microstructure measured on the tape — the Hawkes fit of the taker
  arrivals (it must find the self-excitation that was planted), Roll's
  spread against the quoted spread, Kyle's λ, the signature plot.
  """
  alias Vapor.Finance.{Book, Itch, Micro, Num, PreTrade}

  @limits %{:default => %{max_qty: 500, collar: 0.05, max_position: 400, max_rate: 400, window: 1_000_000_000},
            "info" => %{max_qty: 50, collar: 0.05, max_position: 150, max_rate: 20, window: 1_000_000_000}}

  def limits, do: @limits

  @doc """
  Options: `steps` (1500), `dt` (time units per step, 0.2), `seed`,
  `makers` (2), `half_spread` (ticks, 3), `size` (lots, 20), `gamma`
  (inventory aversion, 0.02), `sigma` (fundamental ticks per √time, 2.0),
  `hawkes` ({μ, α, β} of taker arrivals, {1.0, 0.8, 2.0}), `informed`
  (threshold in ticks or nil, 6).
  """
  def simulate(opts \\ []) do
    steps = Keyword.get(opts, :steps, 1500) |> min(20_000)
    dt = Keyword.get(opts, :dt, 0.2)
    seed = Keyword.get(opts, :seed, 1)
    nm = Keyword.get(opts, :makers, 2) |> max(1) |> min(6)
    hs = Keyword.get(opts, :half_spread, 3)
    size = Keyword.get(opts, :size, 20)
    gam = Keyword.get(opts, :gamma, 0.02)
    sig = Keyword.get(opts, :sigma, 2.0)
    {mu, al, be} = Keyword.get(opts, :hawkes, {1.0, 0.8, 2.0})
    thr = Keyword.get(opts, :informed, 4)
    p0 = 10_000
    u = fn k -> Num.u01(seed, k) end

    # taker arrivals: an exact Hawkes path in continuous time, simulated up front
    arrivals = Micro.hawkes_simulate(mu, al, be, steps * dt, seed + 101)
    by_step = Enum.group_by(arrivals, &trunc(&1 / dt))

    init = %{gate: PreTrade.init(), next_id: 1, fund: p0 * 1.0, prev_fund: p0 * 1.0, quotes: %{}, series: [], signed: 0}
    final =
      Enum.reduce(0..(steps - 1), init, fn step, st ->
        ts = round(step * dt * 1.0e9)
        k0 = step * 64
        # the fundamental moves; the makers quote off the public price of the step before
        fund = st.fund + sig * :math.sqrt(dt) * Num.ninv(u.(k0))
        st = %{st | fund: fund, prev_fund: st.fund}
        ref = st.prev_fund
        # market makers: cancel, then requote around the inventory-skewed reservation price
        st =
          Enum.reduce(1..nm, st, fn m, st ->
            ow = "mm#{m}"
            q = Map.get(st.gate.pos, ow, 0)
            st = Enum.reduce(Map.get(st.quotes, ow, []), st, fn id, st -> if Map.has_key?(st.gate.book.orders, id), do: send_ev(st, %{type: :cancel, id: id, ts: ts}), else: st end)
            r = ref - q * gam * sig * sig
            bid = round(r - hs - (m - 1)); ask = round(r + hs + (m - 1))
            {st, b_id} = new_id(st); st = send_ev(st, %{type: :new, id: b_id, owner: ow, side: :buy, price: bid, qty: size, tif: :gtc, post_only: true, ts: ts})
            {st, a_id} = new_id(st); st = send_ev(st, %{type: :new, id: a_id, owner: ow, side: :sell, price: ask, qty: size, tif: :gtc, post_only: true, ts: ts})
            %{st | quotes: Map.put(st.quotes, ow, [b_id, a_id])}
          end)
        # noise takers at their exact arrival times within the step
        st =
          by_step |> Map.get(step, []) |> Enum.with_index() |> Enum.reduce(st, fn {at, j}, st ->
            side = if u.(k0 + 10 + j) < 0.5, do: :buy, else: :sell
            qty = 1 + trunc(u.(k0 + 30 + j) * 8)
            {st, id} = new_id(st)
            st = send_ev(st, %{type: :new, id: id, owner: "noise#{rem(id, 7)}", side: side, price: nil, qty: qty, tif: :ioc, ts: round(at * 1.0e9)})
            %{st | signed: st.signed + if(side == :buy, do: qty, else: -qty)}
          end)
        # the informed trader
        st =
          if thr do
            {b, a} = {Book.best_bid(st.gate.book), Book.best_ask(st.gate.book)}
            cond do
              a != nil and fund - a > thr -> ({st, id} = new_id(st); %{send_ev(st, %{type: :new, id: id, owner: "info", side: :buy, price: a, qty: 10, tif: :ioc, ts: ts}) | signed: st.signed + 10})
              b != nil and b - fund > thr -> ({st, id} = new_id(st); %{send_ev(st, %{type: :new, id: id, owner: "info", side: :sell, price: b, qty: 10, tif: :ioc, ts: ts}) | signed: st.signed - 10})
              true -> st
            end
          else
            st
          end
        book = st.gate.book
        {b, a} = {Book.best_bid(book), Book.best_ask(book)}
        d = Book.depth(book, 1)
        row = %{t: step * dt, fund: fund, bid: b, ask: a, mid: if(b && a, do: (b + a) / 2, else: nil), last: book.last_trade,
                qb: (case d.bids do [{_, q, _} | _] -> q; _ -> 0 end), qa: (case d.asks do [{_, q, _} | _] -> q; _ -> 0 end), signed: st.signed,
                inv: Map.new(1..nm, &{"mm#{&1}", Map.get(st.gate.pos, "mm#{&1}", 0)})}
        %{st | series: [row | st.series], signed: 0}
      end)

    series = Enum.reverse(final.series)
    book = final.gate.book
    journal = Book.journal(book)
    session = %{book: book, journal: journal}
    trades = for %{reports: rs, seq: s, event: e} <- journal, {:fill, f} <- rs, do: Map.merge(f, %{seq: s, ts: Map.get(e, :ts, 0)})
    {check_us, check} = :timer.tc(fn -> Book.Check.check(journal) end)
    risk = PreTrade.check(journal, @limits)
    feed = Itch.from_session(session)
    {:ok, back} = Itch.unframe(Itch.frame(feed))
    itch = Itch.consistent?(session, back)
    t_end = steps * dt
    hawkes = if length(arrivals) > 50, do: Micro.hawkes_fit(arrivals, t_end), else: nil
    mids = series |> Enum.map(& &1.mid) |> Enum.reject(&is_nil/1)
    dmid = Enum.zip(series, tl(series)) |> Enum.filter(fn {x, y} -> x.mid && y.mid end) |> Enum.map(fn {x, y} -> y.mid - x.mid end)
    sv = Enum.zip(series, tl(series)) |> Enum.filter(fn {x, y} -> x.mid && y.mid end) |> Enum.map(fn {_, y} -> y.signed * 1.0 end)
    quoted = series |> Enum.filter(&(&1.bid && &1.ask)) |> Enum.map(&(&1.ask - &1.bid))
    prices = Enum.map(trades, &(&1.price * 1.0))
    pnl = for m <- 1..nm do
      ow = "mm#{m}"
      cash = Enum.reduce(trades, 0.0, fn f, c ->
        cond do
          f.maker_owner == ow -> c + if(f.taker_side == :buy, do: f.price * f.qty, else: -f.price * f.qty)
          f.taker_owner == ow -> c + if(f.taker_side == :buy, do: -f.price * f.qty, else: f.price * f.qty)
          true -> c
        end
      end)
      q = Map.get(final.gate.pos, ow, 0)
      %{maker: ow, cash: cash, inventory: q, pnl_marked: cash + q * List.last(series).fund}
    end
    %{steps: steps, events: length(journal), trades: length(trades), volume: book.volume, head: Base.encode16(book.head, case: :lower),
      merkle_root: Base.encode16(Book.merkle_root(book), case: :lower), rejected: Enum.count(journal, &(&1.event.type == :risk_reject)),
      series: thin(series, 600), tape: Enum.take(trades, -40), depth: Book.depth(book, 12), makers: pnl,
      measures: %{quoted_spread: Num.mean(quoted), roll_spread: Micro.roll_spread(prices), kyle: Micro.kyle_lambda(dmid, sv), hawkes: hawkes,
                  signature: Micro.signature(mids, 12)},
      certificate: %{book_check: Map.take(check, [:ok, :checked, :verdict]), book_check_ms: check_us / 1000, pre_trade: risk, itch: itch,
                     itch_messages: length(feed)},
      final_journal: journal}
  end

  defp new_id(st), do: {%{st | next_id: st.next_id + 1}, st.next_id}

  defp send_ev(st, ev) do
    {g, _} = PreTrade.step(st.gate, ev, @limits)
    %{st | gate: g}
  end

  defp thin(xs, m) do
    n = length(xs)
    if n <= m, do: xs, else: (step = n / m; for(i <- 0..(m - 1), do: Enum.at(xs, trunc(i * step))))
  end
end
