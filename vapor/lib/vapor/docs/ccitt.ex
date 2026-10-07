defmodule Vapor.Docs.CCITT do
  @moduledoc """
  **CCITT fax decoding** (ITU-T T.4 and T.6) — the compression of almost
  every black-and-white page an office scanner ever put into a PDF
  (`/CCITTFaxDecode`), with no dependency.

  Three codings, all decoded here:

    * **Group 3, 1-D** (`K = 0`, Modified Huffman): each row is alternating
      white/black run lengths, Huffman-coded (terminating codes 0–63,
      make-up codes in multiples of 64, the shared extended make-ups to
      2560), rows usually separated by `EOL` (`000000000001`);
    * **Group 3, 2-D** (`K > 0`, Modified READ): after each `EOL` a tag bit
      says whether the next row is 1-D or coded against the row above;
    * **Group 4** (`K < 0`, Modified Modified READ): every row is coded
      against the row above (the first against an all-white row), no `EOL`s,
      ended by `EOFB`.

  Two-dimensional coding describes the *changing elements* of the coding
  row by where they fall relative to those of the reference row: **pass**
  (`b2` is passed by), **vertical** (`a1 = b1 + d`, `|d| ≤ 3`) or
  **horizontal** (two Huffman runs). The decoder keeps a row as its *run
  ends* (white first) and follows the same state machine as the reference
  decoders (libtiff, pdf.js, Xpdf): the same handling of `EOL`s, of
  byte-aligned rows (`EncodedByteAlign`), of `EndOfBlock` and of a code that
  is not in the tables (the row is finished white and, when the stream has
  `EOL`s, decoding resynchronises on the next one).

  Output follows the PDF filter: rows packed to whole bytes, **0 = black**
  unless `BlackIs1`. Verified bit for bit against libtiff's encoder through
  Pillow (`test/vapor/ccitt_test.exs`): random noise, text pages, every run
  length up to and past 2560, the three codings, with and without fill bits.
  """
  import Bitwise

  @eol 1

  # -------------------------------------------------------------- tables --

  @white_term ~w(00110101 000111 0111 1000 1011 1100 1110 1111 10011 10100 00111 01000 001000 000011 110100 110101
                 101010 101011 0100111 0001100 0001000 0010111 0000011 0000100 0101000 0101011 0010011 0100100 0011000
                 00000010 00000011 00011010 00011011 00010010 00010011 00010100 00010101 00010110 00010111 00101000
                 00101001 00101010 00101011 00101100 00101101 00000100 00000101 00001010 00001011 01010010 01010011
                 01010100 01010101 00100100 00100101 01011000 01011001 01011010 01011011 01001010 01001011 00110010
                 00110011 00110100)
  @white_makeup ~w(11011 10010 010111 0110111 00110110 00110111 01100100 01100101 01101000 01100111 011001100 011001101
                   011010010 011010011 011010100 011010101 011010110 011010111 011011000 011011001 011011010 011011011
                   010011000 010011001 010011010 011000 010011011)
  @black_term ~w(0000110111 010 11 10 011 0011 0010 00011 000101 000100 0000100 0000101 0000111 00000100 00000111
                 000011000 0000010111 0000011000 0000001000 00001100111 00001101000 00001101100 00000110111
                 00000101000 00000010111 00000011000 000011001010 000011001011 000011001100 000011001101 000001101000
                 000001101001 000001101010 000001101011 000011010010 000011010011 000011010100 000011010101
                 000011010110 000011010111 000001101100 000001101101 000011011010 000011011011 000001010100
                 000001010101 000001010110 000001010111 000001100100 000001100101 000001010010 000001010011
                 000000100100 000000110111 000000111000 000000100111 000000101000 000001011000 000001011001
                 000000101011 000000101100 000001011010 000001100110 000001100111)
  @black_makeup ~w(0000001111 000011001000 000011001001 000001011011 000000110011 000000110100 000000110101
                   0000001101100 0000001101101 0000001001010 0000001001011 0000001001100 0000001001101 0000001110010
                   0000001110011 0000001110100 0000001110101 0000001110110 0000001110111 0000001010010 0000001010011
                   0000001010100 0000001010101 0000001011010 0000001011011 0000001100100 0000001100101)
  @ext_makeup ~w(00000001000 00000001100 00000001101 000000010010 000000010011 000000010100 000000010101 000000010110
                 000000010111 000000011100 000000011101 000000011110 000000011111)

  # {code string, run}; make-ups are 64·(i + 1), extended 1792 + 64·i
  codes = fn term, makeup ->
    Enum.with_index(term, fn c, i -> {c, i} end) ++
      Enum.with_index(makeup, fn c, i -> {c, 64 * (i + 1)} end) ++
      Enum.with_index(@ext_makeup, fn c, i -> {c, 1792 + 64 * i} end)
  end

  # a direct table over `bits`-bit peeks: index → {length, run} (or {0, nil})
  table = fn list, bits ->
    entries =
      for {c, run} <- list, len = byte_size(c), v = String.to_integer(c, 2), s = bits - len,
          i <- (v <<< s)..((v <<< s) + (1 <<< s) - 1), into: %{}, do: {i, {len, run}}

    for(i <- 0..((1 <<< bits) - 1), do: Map.get(entries, i, {0, nil})) |> List.to_tuple()
  end

  @white table.(codes.(@white_term, @white_makeup), 12)
  @black table.(codes.(@black_term, @black_makeup), 13)

  # two-dimensional mode codes (≤ 7 bits)
  @modes table.([{"0001", :pass}, {"001", :horiz}, {"1", {:v, 0}}, {"011", {:v, 1}}, {"000011", {:v, 2}},
                 {"0000011", {:v, 3}}, {"010", {:v, -1}}, {"000010", {:v, -2}}, {"0000010", {:v, -3}}], 7)

  @doc false
  def tables, do: %{white: @white, black: @black, modes: @modes}

  # ------------------------------------------------------------- decoding --

  @doc """
  Decode a CCITT stream. Options (the PDF `DecodeParms` keys, as atoms):
  `k` (0), `columns` (1728), `rows` (0 = until the data ends), `black_is_1`
  (false), `end_of_line` (false), `byte_align` (false), `end_of_block`
  (true). Returns `{:ok, %{columns, rows, data, warnings}}` — `data` the
  rows packed to whole bytes — or `{:error, reason}`.
  """
  def decode(data, opts \\ []) when is_binary(data) do
    k = Keyword.get(opts, :k, 0)
    cols = Keyword.get(opts, :columns, 1728)
    rows = Keyword.get(opts, :rows, 0)

    cond do
      not (is_integer(cols) and cols > 0 and cols <= 65_536) -> {:error, "Columns #{inspect(cols)}"}
      not (is_integer(rows) and rows >= 0) -> {:error, "Rows #{inspect(rows)}"}
      rows > 0 and rows * cols > 256_000_000 -> {:error, "an image of #{cols}×#{rows} (too large)"}
      true ->
        st = %{
          bin: data, pos: 0, k: k, cols: cols, rows: rows, eoline: !!opts[:end_of_line],
          align: !!opts[:byte_align], eob: Keyword.get(opts, :end_of_block, true) != false,
          white: if(opts[:black_is_1], do: 0, else: 1), eof: false, done: false, err: false,
          next2d: k < 0, warns: []
        }

        st = start(st)
        {out, st} = rows_loop(st, {cols}, [], 0)
        {:ok, %{columns: cols, rows: length(out), data: out |> Enum.reverse() |> IO.iodata_to_binary(), warnings: Enum.uniq(Enum.reverse(st.warns)),
                bits_used: st.pos}}
    end
  end

  # leading fill and EOL, and the first tag bit of mixed Group 3
  defp start(st) do
    st = skip_zeros(st)
    # an EOL before the first row: the stream has EOLs (TIFF's Group 3 does)
    st = if look(st, 12) == @eol, do: %{eat(st, 12) | eoline: true}, else: st

    if st.k > 0 do
      %{eat(st, 1) | next2d: look(st, 1) == 0}
    else
      st
    end
  end

  defp skip_zeros(st), do: if(look(st, 12) == 0, do: skip_zeros(eat(st, 1)), else: st)

  defp rows_loop(st, _ref, out, _n) when st.eof or st.done, do: {out, st}
  defp rows_loop(st, _ref, out, n) when st.rows > 0 and n >= st.rows, do: {out, st}

  defp rows_loop(st, ref, out, n) do
    cond do
      st.pos >= bit_size(st.bin) -> {out, st}
      n * st.cols > 256_000_000 -> {out, warn(st, "image cut at #{n} rows (too large)")}
      true ->
        {ends, st} = if st.next2d, do: row_2d(%{st | err: false}, ref), else: row_1d(%{st | err: false})
        st = after_row(st, n)
        rows_loop(st, ref_of(ends, st.cols), [pack(ends, st) | out], n + 1)
    end
  end

  # the reference row: run ends below `columns`, then `columns` twice
  defp ref_of(ends, cols) do
    (ends |> Enum.reverse() |> Enum.take_while(&(&1 < cols))) ++ [cols, cols] |> List.to_tuple()
  end

  # ---- a row as run ends: `ends` reversed (head = current end), `n` = count

  defp row_1d(st) do
    one_d(st, [0], 1, 0)
  end

  defp one_d(st, [cur | _] = ends, n, black) do
    if cur >= st.cols do
      {ends, st}
    else
      {run, st} = run_length(st, black)
      {ends, n} = add(ends, n, cur + run, black, st.cols)
      one_d(st, ends, n, bxor(black, 1))
    end
  end

  defp run_length(st, black, acc \\ 0) do
    {r, st} = if black == 1, do: code(st, @black, 13), else: code(st, @white, 12)
    if r >= 64, do: run_length(st, black, acc + r), else: {acc + r, st}
  end

  # a Huffman run: unknown codes eat one bit and count 1 (as pdf.js/Xpdf)
  defp code(st, tab, bits) do
    case look(st, bits) do
      :eof -> {1, %{st | eof: true}}
      v ->
        case elem(tab, v) do
          {0, nil} -> {1, warn(%{eat(st, 1) | err: true}, "a code not in the CCITT tables")}
          {len, run} -> {run, eat(st, len)}
        end
    end
  end

  defp row_2d(st, ref) do
    two_d(st, ref, [0], 1, 0, 0)
  end

  defp two_d(st, ref, [cur | _] = ends, n, rp, black) do
    cols = st.cols

    if cur >= cols do
      {ends, st}
    else
      {mode, st} = mode(st)

      case mode do
        :pass ->
          b2 = at(ref, rp + 1, cols)
          {ends, n} = add(ends, n, b2, black, cols)
          two_d(st, ref, ends, n, if(b2 < cols, do: rp + 2, else: rp), black)

        :horiz ->
          {r1, st} = run_length(st, black)
          {r2, st} = run_length(st, bxor(black, 1))
          {ends, n} = add(ends, n, cur + r1, black, cols)
          [c1 | _] = ends
          {ends, n} = if c1 < cols, do: add(ends, n, c1 + r2, bxor(black, 1), cols), else: {ends, n}
          two_d(st, ref, ends, n, skip(ref, rp, hd(ends), cols), black)

        {:v, d} when d >= 0 ->
          {ends, n} = add(ends, n, at(ref, rp, cols) + d, black, cols)
          [c | _] = ends
          rp = if c < cols, do: skip(ref, rp + 1, c, cols), else: rp
          two_d(st, ref, ends, n, rp, bxor(black, 1))

        {:v, d} ->
          {ends, n, st} = add_neg(ends, n, at(ref, rp, cols) + d, black, st)
          [c | _] = ends
          rp = if c < cols, do: skip(ref, if(rp > 0, do: rp - 1, else: rp + 1), c, cols), else: rp
          two_d(st, ref, ends, n, rp, bxor(black, 1))

        :eof ->
          {ends, _} = add(ends, n, cols, 0, cols)
          {ends, %{st | eof: true}}

        :bad ->
          {ends, _} = add(ends, n, cols, 0, cols)
          {ends, warn(%{st | err: true}, "a two-dimensional code not in the tables")}
      end
    end
  end

  defp mode(st) do
    case look(st, 7) do
      :eof -> {:eof, st}
      v ->
        case elem(@modes, v) do
          {0, nil} -> {:bad, st}
          {len, m} -> {m, eat(st, len)}
        end
    end
  end

  defp at(ref, i, _cols) when i < tuple_size(ref), do: elem(ref, i)
  defp at(_ref, _i, cols), do: cols

  defp skip(ref, rp, cur, cols) do
    v = at(ref, rp, cols)
    if v <= cur and v < cols, do: skip(ref, rp + 2, cur, cols), else: rp
  end

  # paint up to a1 in colour `black`: the current run grows, or a new one starts
  defp add([e | rest] = ends, n, a1, black, cols) do
    if a1 > e do
      a1 = min(a1, cols)
      if bxor(band(n - 1, 1), black) == 1, do: {[a1 | ends], n + 1}, else: {[a1 | rest], n}
    else
      {ends, n}
    end
  end

  # vertical-left: a1 may fall before the current end (only in damaged data)
  defp add_neg([e | _] = ends, n, a1, black, st) do
    cond do
      a1 > e -> {ends2, n2} = add(ends, n, a1, black, st.cols); {ends2, n2, st}
      a1 < e ->
        {a1, st} = if a1 < 0, do: {0, warn(%{st | err: true}, "a vertical code left of the row")}, else: {a1, st}
        {ends, n} = back(ends, n, a1)
        {[a1 | tl(ends)], n, st}
      true -> {ends, n, st}
    end
  end

  defp back([_, prev | rest] = ends, n, a1) when n > 1 do
    if a1 < prev, do: back([prev | rest], n - 1, a1), else: {ends, n}
  end

  defp back(ends, n, _a1), do: {ends, n}

  # ---- after a row: alignment, EOL, tag bit, RTC/EOFB, resynchronisation

  # (Xpdf's order: EOL search only where EOLs can be told from data; byte
  # alignment only when no EOL was found; RTC/EOFB after an EOL; after a
  # damaged row, resynchronise on the next EOL when the stream has them)
  defp after_row(st, n) do
    {st, got_eol} =
      cond do
        not st.eob and st.rows > 0 and n == st.rows - 1 -> {%{st | done: true}, false}
        st.eoline or not st.align ->
          st = if st.eoline, do: to_eol(st), else: skip_zeros(st)
          if look(st, 12) == @eol, do: {eat(st, 12), true}, else: {st, false}
        true -> {st, false}
      end

    st = if st.align and not got_eol, do: %{st | pos: st.pos + rem(8 - rem(st.pos, 8), 8)}, else: st
    st = if look(st, 1) == :eof, do: %{st | eof: true}, else: st
    st = if not st.eof and not st.done and st.k > 0, do: %{eat(st, 1) | next2d: look(st, 1) == 0}, else: st

    {st, got_eol} =
      if st.eob and not st.eoline and st.align and look(st, 24) == 0x001001, do: {eat(st, 12), true}, else: {st, got_eol}

    cond do
      st.eob and got_eol and look(st, 12) == @eol -> %{st | eof: true}
      st.err and st.eoline -> resync(st)
      # a damaged row in a stream without EOLs: nothing to resynchronise on
      st.err and st.next2d -> warn(%{st | eof: true}, "decoding stopped at a damaged row")
      true -> st
    end
  end

  defp to_eol(st) do
    case look(st, 12) do
      v when v in [@eol, :eof] -> st
      _ -> to_eol(eat(st, 1))
    end
  end

  defp resync(st) do
    case look(st, 13) do
      :eof -> %{st | eof: true}
      v when v >>> 1 == @eol ->
        st = eat(st, 12)
        if st.k > 0, do: %{eat(st, 1) | next2d: band(v, 1) == 0}, else: st
      _ -> resync(eat(st, 1))
    end
  end

  # ---------------------------------------------------------------- bits --

  defp look(%{bin: bin, pos: pos}, n) do
    case bin do
      <<_::bitstring-size(pos), v::size(n), _::bitstring>> -> v
      _ ->
        avail = bit_size(bin) - pos

        if avail <= 0 do
          :eof
        else
          <<_::bitstring-size(pos), v::size(avail)>> = bin
          v <<< (n - avail)
        end
    end
  end

  defp eat(st, n), do: %{st | pos: st.pos + n}

  defp warn(st, w), do: %{st | warns: [w | st.warns]}

  # run ends (reversed) → one packed row, white = st.white
  defp pack(ends, %{cols: cols, white: wbit}) do
    ends = Enum.reverse(ends)
    bwbit = 1 - wbit

    {bits, _, _} =
      Enum.reduce(ends, {[], 0, 0}, fn e, {acc, from, colour} ->
        e = min(e, cols)
        len = max(e - from, 0)
        v = if colour == 0, do: wbit, else: bwbit
        {[<<-v::size(len)>> | acc], max(e, from), bxor(colour, 1)}
      end)

    row = bits |> Enum.reverse() |> :erlang.list_to_bitstring()
    row = if bit_size(row) < cols, do: <<row::bitstring, -wbit::size(cols - bit_size(row))>>, else: row
    pad = rem(8 - rem(cols, 8), 8)
    <<row::bitstring, 0::size(pad)>>
  end
end
