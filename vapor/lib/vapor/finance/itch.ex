defmodule Vapor.Finance.Itch do
  @moduledoc """
  NASDAQ TotalView-ITCH 5.0, the binary market-by-order feed
  (docs/FINANCE.md §10): an encoder, a decoder, and a book rebuilt from
  the feed alone.

  Messages handled (lengths as in the specification, big-endian, prices
  with four implied decimals, timestamps in nanoseconds since midnight in
  six bytes): `S` system event (12), `R` stock directory (39), `A` add
  order (36), `F` add with attribution (40), `E` executed (31), `C`
  executed with price (36), `X` cancel (23), `D` delete (19), `U` replace
  (35), `P` non-cross trade (44). Files are the BinaryFILE framing: each
  message preceded by its two-byte length.

  `from_session/2` turns an engine session (`Vapor.Finance.Book`) into the
  feed an exchange would publish — `A` when an order rests, `E` when it is
  hit, `X`/`D` when it shrinks or leaves — and `rebuild/1` reconstructs the
  book from the feed: equality with the engine's own book is a second,
  independent consistency check (the feed shows only what rests; the
  engine knows everything).
  """
  @lengths %{?S => 12, ?R => 39, ?A => 36, ?F => 40, ?E => 31, ?C => 36, ?X => 23, ?D => 19, ?U => 35, ?P => 44}
  def lengths, do: @lengths

  defp stock8(s), do: s |> String.slice(0, 8) |> String.pad_trailing(8)

  @doc "Encode one message (a map with `:type` and its fields) to its bytes (no framing)."
  def encode(%{type: :system} = m), do: <<?S, lt(m)::binary, m.event::8>>
  def encode(%{type: :directory} = m), do: <<?R, lt(m)::binary, stock8(m.stock)::binary, Map.get(m, :market_category, ?Q)::8, ?N, Map.get(m, :round_lot, 100)::32, ?N, ?C, "  ", ?P, ?N, ?N, ?1, ?N, 0::32, ?N>>
  def encode(%{type: :add} = m), do: <<?A, lt(m)::binary, m.ref::64, side(m.side)::binary, m.shares::32, stock8(m.stock)::binary, m.price::32>>
  def encode(%{type: :add_mpid} = m), do: <<?F, lt(m)::binary, m.ref::64, side(m.side)::binary, m.shares::32, stock8(m.stock)::binary, m.price::32, String.pad_trailing(m.mpid, 4)::binary>>
  def encode(%{type: :executed} = m), do: <<?E, lt(m)::binary, m.ref::64, m.shares::32, m.match::64>>
  def encode(%{type: :executed_price} = m), do: <<?C, lt(m)::binary, m.ref::64, m.shares::32, m.match::64, (if m.printable, do: ?Y, else: ?N), m.price::32>>
  def encode(%{type: :cancel} = m), do: <<?X, lt(m)::binary, m.ref::64, m.shares::32>>
  def encode(%{type: :delete} = m), do: <<?D, lt(m)::binary, m.ref::64>>
  def encode(%{type: :replace} = m), do: <<?U, lt(m)::binary, m.ref::64, m.new_ref::64, m.shares::32, m.price::32>>
  def encode(%{type: :trade} = m), do: <<?P, lt(m)::binary, m.ref::64, side(m.side)::binary, m.shares::32, stock8(m.stock)::binary, m.price::32, m.match::64>>

  # stock locate, tracking number, timestamp (48 bits)
  defp lt(m), do: <<Map.get(m, :locate, 1)::16, Map.get(m, :tracking, 0)::16, Map.get(m, :ts, 0)::48>>
  defp side(:buy), do: <<?B>>
  defp side(:sell), do: <<?S>>

  @doc "Decode one message's bytes. `{:ok, map}` or `{:error, why}` (length checked against the type)."
  def decode(<<t, _::binary>> = b) do
    case @lengths do
      %{^t => len} when byte_size(b) == len -> {:ok, dec(b)}
      %{^t => len} -> {:error, "message #{<<t>>} must be #{len} bytes, got #{byte_size(b)}"}
      _ -> {:error, "unknown message type #{inspect(<<t>>)}"}
    end
  end

  defp hdr(loc, tr, ts), do: %{locate: loc, tracking: tr, ts: ts}
  defp sd(?B), do: :buy
  defp sd(?S), do: :sell
  defp st(s), do: String.trim_trailing(s)

  defp dec(<<?S, l::16, tr::16, ts::48, ev::8>>), do: Map.merge(hdr(l, tr, ts), %{type: :system, event: ev})
  defp dec(<<?R, l::16, tr::16, ts::48, stock::binary-8, mc::8, _fs::8, lot::32, _::binary>>), do: Map.merge(hdr(l, tr, ts), %{type: :directory, stock: st(stock), market_category: mc, round_lot: lot})
  defp dec(<<?A, l::16, tr::16, ts::48, ref::64, s::8, sh::32, stock::binary-8, p::32>>), do: Map.merge(hdr(l, tr, ts), %{type: :add, ref: ref, side: sd(s), shares: sh, stock: st(stock), price: p})
  defp dec(<<?F, l::16, tr::16, ts::48, ref::64, s::8, sh::32, stock::binary-8, p::32, mpid::binary-4>>), do: Map.merge(hdr(l, tr, ts), %{type: :add_mpid, ref: ref, side: sd(s), shares: sh, stock: st(stock), price: p, mpid: st(mpid)})
  defp dec(<<?E, l::16, tr::16, ts::48, ref::64, sh::32, mt::64>>), do: Map.merge(hdr(l, tr, ts), %{type: :executed, ref: ref, shares: sh, match: mt})
  defp dec(<<?C, l::16, tr::16, ts::48, ref::64, sh::32, mt::64, pr::8, p::32>>), do: Map.merge(hdr(l, tr, ts), %{type: :executed_price, ref: ref, shares: sh, match: mt, printable: pr == ?Y, price: p})
  defp dec(<<?X, l::16, tr::16, ts::48, ref::64, sh::32>>), do: Map.merge(hdr(l, tr, ts), %{type: :cancel, ref: ref, shares: sh})
  defp dec(<<?D, l::16, tr::16, ts::48, ref::64>>), do: Map.merge(hdr(l, tr, ts), %{type: :delete, ref: ref})
  defp dec(<<?U, l::16, tr::16, ts::48, ref::64, nref::64, sh::32, p::32>>), do: Map.merge(hdr(l, tr, ts), %{type: :replace, ref: ref, new_ref: nref, shares: sh, price: p})
  defp dec(<<?P, l::16, tr::16, ts::48, ref::64, s::8, sh::32, stock::binary-8, p::32, mt::64>>), do: Map.merge(hdr(l, tr, ts), %{type: :trade, ref: ref, side: sd(s), shares: sh, stock: st(stock), price: p, match: mt})

  @doc "Frame messages as a BinaryFILE (two-byte length before each)."
  def frame(msgs), do: for(m <- msgs, into: <<>>, do: (b = encode(m); <<byte_size(b)::16, b::binary>>))

  @doc "Read a BinaryFILE: `{:ok, messages}` or the offset where it breaks."
  def unframe(bin), do: unframe(bin, 0, [])
  defp unframe(<<>>, _, acc), do: {:ok, Enum.reverse(acc)}
  defp unframe(<<len::16, rest::binary>>, off, acc) when byte_size(rest) >= len do
    <<b::binary-size(len), rest2::binary>> = rest
    case decode(b) do
      {:ok, m} -> unframe(rest2, off + 2 + len, [m | acc])
      {:error, w} -> {:error, "at byte #{off}: #{w}"}
    end
  end
  defp unframe(_, off, _), do: {:error, "truncated message at byte #{off}"}

  # --------------------------------------------------- engine → feed → book

  @doc """
  The ITCH feed of an engine session (`Vapor.Finance.Book.session/2`):
  prices in ticks become four-decimal prices with `tick` (default 0.01 →
  ×100). Order references are the engine's ids; match numbers count up.
  """
  def from_session(session, opts \\ []) do
    stock = Keyword.get(opts, :stock, "VAPR")
    mult = Keyword.get(opts, :price_multiplier, 100)
    {msgs, _} =
      Enum.flat_map_reduce(session.journal, %{resting: %{}, sides: %{}, match: 0}, fn %{seq: seq, event: ev, reports: reps}, st ->
        ts = Map.get(ev, :ts, seq * 1000)
        Enum.flat_map_reduce(reps, st, fn rep, st ->
          case rep do
            {:rested, id, price, qty} ->
              side = Map.get(ev, :side) || st.sides[id]
              st = %{st | sides: Map.put(st.sides, id, side)}
              {[%{type: :add, ts: ts, ref: id, side: side, shares: qty, stock: stock, price: price * mult}], put_in(st, [:resting, id], %{side: side, qty: qty})}
            {:fill, f} ->
              st = %{st | match: st.match + 1}
              m = st.resting[f.maker]
              st = if m.qty == f.qty, do: %{st | resting: Map.delete(st.resting, f.maker)}, else: put_in(st, [:resting, f.maker, :qty], m.qty - f.qty)
              {[%{type: :executed, ts: ts, ref: f.maker, shares: f.qty, match: st.match}], st}
            {:cancelled, id, _q, _why} ->
              case st.resting[id] do
                nil -> {[], st}
                _ -> {[%{type: :delete, ts: ts, ref: id}], %{st | resting: Map.delete(st.resting, id)}}
              end
            {:modified, id, qty} ->
              m = st.resting[id]
              {[%{type: :cancel, ts: ts, ref: id, shares: m.qty - qty}], put_in(st, [:resting, id, :qty], qty)}
            _ -> {[], st}
          end
        end)
      end)
    [%{type: :system, ts: 0, event: ?O}, %{type: :directory, ts: 0, stock: stock}] ++ msgs ++ [%{type: :system, ts: 86_399_000_000_000, event: ?C}]
  end

  @doc """
  Rebuild the aggregated book from a feed: `%{bids: [{price, shares, orders}], asks: …}`
  best first, prices in the feed's units; plus the volume executed.
  """
  def rebuild(msgs) do
    {orders, vol} =
      Enum.reduce(msgs, {%{}, 0}, fn m, {o, v} ->
        case m.type do
          t when t in [:add, :add_mpid] -> {Map.put(o, m.ref, %{side: m.side, price: m.price, shares: m.shares}), v}
          t when t in [:executed, :executed_price] -> {dec_shares(o, m.ref, m.shares), v + m.shares}
          :cancel -> {dec_shares(o, m.ref, m.shares), v}
          :delete -> {Map.delete(o, m.ref), v}
          :replace -> (old = o[m.ref]; {o |> Map.delete(m.ref) |> Map.put(m.new_ref, %{side: old.side, price: m.price, shares: m.shares}), v})
          _ -> {o, v}
        end
      end)
    levels = fn side, dir ->
      orders |> Map.values() |> Enum.filter(&(&1.side == side)) |> Enum.group_by(& &1.price)
      |> Enum.map(fn {p, os} -> {p, Enum.sum(Enum.map(os, & &1.shares)), length(os)} end)
      |> Enum.sort_by(&elem(&1, 0), dir)
    end
    %{bids: levels.(:buy, :desc), asks: levels.(:sell, :asc), executed: vol, orders: map_size(orders)}
  end

  defp dec_shares(o, ref, sh) do
    case o[ref] do
      nil -> o
      %{shares: s} when s <= sh -> Map.delete(o, ref)
      x -> Map.put(o, ref, %{x | shares: x.shares - sh})
    end
  end

  @doc "Does the feed rebuild exactly the engine's final book (all levels) and its traded volume?"
  def consistent?(session, msgs, mult \\ 100) do
    fb = rebuild(msgs)
    eb = Vapor.Finance.Book.depth(session.book, 1_000_000)
    scale = fn ls -> Enum.map(ls, fn {p, q, n} -> {p * mult, q, n} end) end
    %{book_equal: fb.bids == scale.(eb.bids) and fb.asks == scale.(eb.asks), volume_equal: fb.executed == session.book.volume,
      levels: length(fb.bids) + length(fb.asks), executed: fb.executed}
  end
end
