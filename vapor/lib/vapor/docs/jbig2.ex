defmodule Vapor.Docs.JBIG2 do
  @moduledoc """
  JBIG2 (ITU-T T.88 | ISO/IEC 14492) — the other format of scanned PDFs,
  the one `pdfimages` meets in archives, court records and every document
  made smaller by `jbig2enc`/`ocrmypdf`. Without dependencies; bit-exact
  against the reference decoder (`jbig2dec`, the one Ghostscript and
  MuPDF ship) on streams from the reference encoder (`jbig2enc`) and on
  streams written in the tests for the parts no encoder exercises.

  Decoded here:

    * the **MQ arithmetic decoder** (Annex E) and the integer decoders
      `IAx` and `IAID` (Annex A);
    * **generic regions** (6.2): templates 0–3 with adaptive pixels,
      typical prediction (`TPGDON`), and MMR coding (6.2.6, through
      `Vapor.Docs.CCITT`'s Group 4 decoder);
    * **generic refinement regions** (6.3): templates 0–1, `TPGRON`;
    * **symbol dictionaries** (6.5), arithmetic: height classes, symbols
      by generic decoding, refinement and aggregation (`SDREFAGG`), the
      export flags; symbols from referred-to dictionaries (and the PDF's
      `JBIG2Globals`);
    * **text regions** (6.4), arithmetic: strips, the four reference
      corners, transposition, refinement of symbol instances, the four
      combination operators;
    * **pages** (7.4.8): page information, regions composed with their
      external combination operator, end of stripe, striped pages of
      unknown height.

  Refused, with a reason (a warning, the page keeps what was decoded):
  Huffman-coded symbol dictionaries and text regions (`SDHUFF`/`SBHUFF`,
  Annex B tables), pattern dictionaries and halftone regions (7.4.4–7.4.5),
  and the retained-context flags of symbol dictionaries.

  Bitmaps are `%{w, h, rows}` — `rows` a tuple of row tuples of 0/1, where
  **1 is black** (JBIG2's convention); `packed/1` gives the rows packed
  MSB first, as a PDF filter's output after inversion (`pdf/2`).
  """
  import Bitwise

  # --------------------------------------------------------------- the MQ decoder --

  # Qe, NMPS, NLPS, SWITCH (Table E.1)
  @qe {{0x5601, 1, 1, 1}, {0x3401, 2, 6, 0}, {0x1801, 3, 9, 0}, {0x0AC1, 4, 12, 0}, {0x0521, 5, 29, 0}, {0x0221, 38, 33, 0},
       {0x5601, 7, 6, 1}, {0x5401, 8, 14, 0}, {0x4801, 9, 14, 0}, {0x3801, 10, 14, 0}, {0x3001, 11, 17, 0}, {0x2401, 12, 18, 0},
       {0x1C01, 13, 20, 0}, {0x1601, 29, 21, 0}, {0x5601, 15, 14, 1}, {0x5401, 16, 14, 0}, {0x5101, 17, 15, 0}, {0x4801, 18, 16, 0},
       {0x3801, 19, 17, 0}, {0x3401, 20, 18, 0}, {0x3001, 21, 19, 0}, {0x2801, 22, 19, 0}, {0x2401, 23, 20, 0}, {0x2201, 24, 21, 0},
       {0x1C01, 25, 22, 0}, {0x1801, 26, 23, 0}, {0x1601, 27, 24, 0}, {0x1401, 28, 25, 0}, {0x1201, 29, 26, 0}, {0x1101, 30, 27, 0},
       {0x0AC1, 31, 28, 0}, {0x09C1, 32, 29, 0}, {0x08A1, 33, 30, 0}, {0x0521, 34, 31, 0}, {0x0441, 35, 32, 0}, {0x02A1, 36, 33, 0},
       {0x0221, 37, 34, 0}, {0x0141, 38, 35, 0}, {0x0111, 39, 36, 0}, {0x0085, 40, 37, 0}, {0x0049, 41, 38, 0}, {0x0025, 42, 39, 0},
       {0x0015, 43, 40, 0}, {0x0009, 44, 41, 0}, {0x0005, 45, 42, 0}, {0x0001, 45, 43, 0}, {0x5601, 46, 46, 0}}

  # the decoder's registers: data, position, A, C, CT (the software
  # convention of T.88 E.3: C holds the complement of the code)
  defmodule MQ do
    @moduledoc false
    defstruct [:data, :size, pos: 0, a: 0, c: 0, ct: 0]
  end

  @doc false
  def mq_new(data) do
    b = byte(data, 0)
    st = %MQ{data: data, size: byte_size(data), pos: 0, c: bnot(b) <<< 16 &&& 0xFF0000}
    st = bytein(st)
    %{st | c: st.c <<< 7 &&& 0xFFFFFFFF, ct: st.ct - 7, a: 0x8000}
  end

  defp byte(data, i) when i < byte_size(data), do: :binary.at(data, i)
  defp byte(_data, _i), do: 0xFF

  # BYTEIN (Figure E.20, software convention): past the end the stream is
  # an endless marker — 1 bits
  defp bytein(%MQ{data: d, pos: p} = st) do
    if byte(d, p) == 0xFF do
      b1 = byte(d, p + 1)
      if b1 > 0x8F, do: %{st | ct: 8}, else: %{st | pos: p + 1, c: st.c + 0xFE00 - (b1 <<< 9), ct: 7}
    else
      b1 = byte(d, p + 1)
      %{st | pos: p + 1, c: st.c + 0xFF00 - (b1 <<< 8), ct: 8}
    end
  end

  # decode one bit in context `cx` of the stats array `stats`
  # (each entry index·2 + MPS); returns {bit, st}
  @doc false
  def decode_bit(%MQ{} = st, stats, cx) do
    v = :atomics.get(stats, cx + 1)
    i = v >>> 1
    mps = v &&& 1
    {qe, nmps, nlps, sw} = elem(@qe, i)
    a = st.a - qe

    if st.c >>> 16 < a do
      if (a &&& 0x8000) == 0 do
        # MPS_EXCHANGE
        d =
          if a < qe do
            :atomics.put(stats, cx + 1, nlps <<< 1 ||| if(sw == 1, do: 1 - mps, else: mps))
            1 - mps
          else
            :atomics.put(stats, cx + 1, nmps <<< 1 ||| mps)
            mps
          end

        {d, renorm(%{st | a: a})}
      else
        {mps, %{st | a: a}}
      end
    else
      c = st.c - (a <<< 16)

      d =
        if a < qe do
          :atomics.put(stats, cx + 1, nmps <<< 1 ||| mps)
          mps
        else
          :atomics.put(stats, cx + 1, nlps <<< 1 ||| if(sw == 1, do: 1 - mps, else: mps))
          1 - mps
        end

      {d, renorm(%{st | a: qe, c: c})}
    end
  end

  defp renorm(st) do
    st = if st.ct == 0, do: bytein(st), else: st
    st = %{st | a: st.a <<< 1 &&& 0xFFFF, c: st.c <<< 1 &&& 0xFFFFFFFF, ct: st.ct - 1}
    if (st.a &&& 0x8000) == 0, do: renorm(st), else: st
  end

  @doc false
  def stats(n), do: :atomics.new(n, signed: false)

  # IAx (A.2): an integer, or :oob
  @doc false
  def int(st, stats) do
    {s, st} = decode_bit(st, stats, 1)
    prev = 2 ||| s
    {b0, st} = decode_bit(st, stats, prev)
    prev = prev <<< 1 ||| b0

    {n, off, prev, st} =
      if b0 == 0 do
        {2, 0, prev, st}
      else
        {b1, st} = decode_bit(st, stats, prev)
        prev = prev <<< 1 ||| b1

        if b1 == 0 do
          {4, 4, prev, st}
        else
          {b2, st} = decode_bit(st, stats, prev)
          prev = prev <<< 1 ||| b2

          if b2 == 0 do
            {6, 20, prev, st}
          else
            {b3, st} = decode_bit(st, stats, prev)
            prev = prev <<< 1 ||| b3

            if b3 == 0 do
              {8, 84, prev, st}
            else
              {b4, st} = decode_bit(st, stats, prev)
              prev = prev <<< 1 ||| b4
              if b4 == 0, do: {12, 340, prev, st}, else: {32, 4436, prev, st}
            end
          end
        end
      end

    {v, _prev, st} =
      Enum.reduce(1..n, {0, prev, st}, fn _, {v, prev, st} ->
        {b, st} = decode_bit(st, stats, prev)
        {v <<< 1 ||| b, (prev <<< 1 &&& 511) ||| (prev &&& 256) ||| b, st}
      end)

    v = v + off

    cond do
      s == 1 and v == 0 -> {:oob, st}
      s == 1 -> {-v, st}
      true -> {v, st}
    end
  end

  # IAID (A.3): `len` bits, one context per prefix
  @doc false
  def iaid(st, stats, len) do
    {prev, st} =
      Enum.reduce(1..len//1, {1, st}, fn _, {prev, st} ->
        {b, st} = decode_bit(st, stats, prev)
        {prev <<< 1 ||| b, st}
      end)

    {prev - (1 <<< len), st}
  end

  # -------------------------------------------------------------- bitmaps --

  @doc "A blank bitmap of `w × h` filled with `v`."
  def blank(w, h, v \\ 0), do: %{w: w, h: h, rows: Tuple.duplicate(Tuple.duplicate(v, w), h)}

  defp px(%{w: w, h: h, rows: rows}, x, y) when x >= 0 and y >= 0 and x < w and y < h, do: elem(elem(rows, y), x)
  defp px(_bm, _x, _y), do: 0

  # a row tuple read with zero outside
  defp rpx(row, x, w) when x >= 0 and x < w, do: elem(row, x)
  defp rpx(_row, _x, _w), do: 0

  @doc "Rows packed MSB first, padded to whole bytes (1 = black)."
  def packed(%{w: w, rows: rows}) do
    pad = rem(8 - rem(w, 8), 8)

    for row <- Tuple.to_list(rows), into: <<>> do
      bits = for b <- Tuple.to_list(row), into: <<>>, do: <<b::1>>
      <<bits::bitstring, 0::size(pad)>>
    end
  end

  # compose `src` onto a mutable page (atomics, w × h) at (x, y) with `op`
  # (0 OR, 1 AND, 2 XOR, 3 XNOR, 4 REPLACE)
  defp compose(page, pw, ph, %{w: sw, h: sh, rows: rows}, x, y, op) do
    for j <- max(0, -y)..(min(sh, ph - y) - 1)//1, row = elem(rows, j), i <- max(0, -x)..(min(sw, pw - x) - 1)//1 do
      k = (y + j) * pw + x + i + 1
      s = elem(row, i)

      case op do
        0 -> if s == 1, do: :atomics.put(page, k, 1)
        1 -> if s == 0, do: :atomics.put(page, k, 0)
        2 -> if s == 1, do: :atomics.put(page, k, 1 - :atomics.get(page, k))
        3 -> if s == 0, do: :atomics.put(page, k, 1 - :atomics.get(page, k))
        _ -> :atomics.put(page, k, s)
      end
    end

    :ok
  end

  defp canvas(w, h, v) do
    a = :atomics.new(max(w * h, 1), signed: false)
    if v == 1, do: for(k <- 1..(w * h)//1, do: :atomics.put(a, k, 1))
    a
  end

  defp freeze(a, w, h),
    do: %{w: w, h: h, rows: List.to_tuple(for(y <- 0..(h - 1)//1, do: List.to_tuple(for(x <- 1..w//1, do: :atomics.get(a, y * w + x)))))}

  # ------------------------------------------------------- generic region --

  @nominal_at %{0 => [{3, -1}, {-3, -1}, {2, -2}, {-2, -2}], 1 => [{3, -1}], 2 => [{2, -1}], 3 => [{2, -1}]}
  @sltp %{0 => 0x9B25, 1 => 0x0795, 2 => 0x00E5, 3 => 0x0195}

  @doc false
  # generic region decoding (6.2.5), arithmetic: `{bitmap, st}`
  def generic(st, stats, w, h, template, at, tpgdon) do
    nominal? = at == @nominal_at[template]
    empty = Tuple.duplicate(0, w)

    {rows, st, _ltp} =
      Enum.reduce(0..(h - 1)//1, {[], st, 0}, fn y, {rows, st, ltp} ->
        {ltp, st} =
          if tpgdon do
            {b, st} = decode_bit(st, stats, @sltp[template])
            {bxor(ltp, b), st}
          else
            {0, st}
          end

        p1 = case rows do [r | _] -> r; _ -> empty end
        p2 = case rows do [_, r | _] -> r; _ -> empty end

        if ltp == 1 do
          {[p1 | rows], st, ltp}
        else
          {row, st} = generic_row(st, stats, w, template, at, nominal?, p1, p2, rows, y)
          {[row | rows], st, ltp}
        end
      end)

    {%{w: w, h: h, rows: rows |> Enum.reverse() |> List.to_tuple()}, st}
  end

  # one row: shift registers over the two rows above and the current row,
  # adaptive pixels looked up (the current row's from the register)
  defp generic_row(st, stats, w, template, at, nominal?, p1, p2, rows, y) do
    at_px = fn cur, x, {dx, dy} ->
      cond do
        dy == 0 -> if x + dx >= 0, do: cur >>> (-dx - 1) &&& 1, else: 0
        dy == -1 -> rpx(p1, x + dx, w)
        dy == -2 -> rpx(p2, x + dx, w)
        true ->
          r = Enum.at(rows, -dy - 1)
          if r, do: rpx(r, x + dx, w), else: 0
      end
    end

    _ = y

    {bits, st, _cur} =
      Enum.reduce(0..(w - 1)//1, {[], st, 0}, fn x, {bits, st, cur} ->
        cx =
          case template do
            0 ->
              c = (cur &&& 0xF) ||| rpx(p1, x + 2, w) <<< 5 ||| rpx(p1, x + 1, w) <<< 6 ||| rpx(p1, x, w) <<< 7 |||
                    rpx(p1, x - 1, w) <<< 8 ||| rpx(p1, x - 2, w) <<< 9 ||| rpx(p2, x + 1, w) <<< 12 ||| rpx(p2, x, w) <<< 13 |||
                    rpx(p2, x - 1, w) <<< 14

              if nominal? do
                c ||| rpx(p1, x + 3, w) <<< 4 ||| rpx(p1, x - 3, w) <<< 10 ||| rpx(p2, x + 2, w) <<< 11 ||| rpx(p2, x - 2, w) <<< 15
              else
                [a1, a2, a3, a4] = at
                c ||| at_px.(cur, x, a1) <<< 4 ||| at_px.(cur, x, a2) <<< 10 ||| at_px.(cur, x, a3) <<< 11 ||| at_px.(cur, x, a4) <<< 15
              end

            1 ->
              (cur &&& 0x7) ||| at_px.(cur, x, hd(at)) <<< 3 ||| rpx(p1, x + 2, w) <<< 4 ||| rpx(p1, x + 1, w) <<< 5 |||
                rpx(p1, x, w) <<< 6 ||| rpx(p1, x - 1, w) <<< 7 ||| rpx(p1, x - 2, w) <<< 8 ||| rpx(p2, x + 2, w) <<< 9 |||
                rpx(p2, x + 1, w) <<< 10 ||| rpx(p2, x, w) <<< 11 ||| rpx(p2, x - 1, w) <<< 12

            2 ->
              (cur &&& 0x3) ||| at_px.(cur, x, hd(at)) <<< 2 ||| rpx(p1, x + 1, w) <<< 3 ||| rpx(p1, x, w) <<< 4 |||
                rpx(p1, x - 1, w) <<< 5 ||| rpx(p1, x - 2, w) <<< 6 ||| rpx(p2, x + 1, w) <<< 7 ||| rpx(p2, x, w) <<< 8 |||
                rpx(p2, x - 1, w) <<< 9

            3 ->
              (cur &&& 0xF) ||| at_px.(cur, x, hd(at)) <<< 4 ||| rpx(p1, x + 1, w) <<< 5 ||| rpx(p1, x, w) <<< 6 |||
                rpx(p1, x - 1, w) <<< 7 ||| rpx(p1, x - 2, w) <<< 8 ||| rpx(p1, x - 3, w) <<< 9
          end

        {b, st} = decode_bit(st, stats, cx)
        {[b | bits], st, (cur <<< 1 ||| b) &&& 0xFFFFFFFF}
      end)

    {bits |> Enum.reverse() |> List.to_tuple(), st}
  end

  # ------------------------------------------------- refinement region --

  @doc false
  # generic refinement region decoding (6.3.5): `{bitmap, st}`
  def refine(st, stats, w, h, template, ref, dx, dy, grat, tpgron) do
    empty = Tuple.duplicate(0, w)

    {rows, st, _} =
      Enum.reduce(0..(h - 1)//1, {[], st, 0}, fn y, {rows, st, ltp} ->
        {ltp, st} =
          if tpgron do
            # the SLTP context: jbig2dec's (template 0: the reference pixel
            # under X; template 1: its right neighbour — pdf.js reads the
            # latter as the pixel under X: docs/OCR.md §3f)
            {b, st} = decode_bit(st, stats, if(template == 0, do: 0x100, else: 0x40))
            {bxor(ltp, b), st}
          else
            {0, st}
          end

        p1 = case rows do [r | _] -> r; _ -> empty end

        {bits, st, _} =
          Enum.reduce(0..(w - 1)//1, {[], st, 0}, fn x, {bits, st, cur} ->
            rx = x - dx
            ry = y - dy

            # typical prediction: a uniform 3×3 reference neighbourhood is copied
            tp =
              if ltp == 1 do
                vals = for j <- -1..1, i <- -1..1, do: px(ref, rx + i, ry + j)
                if Enum.all?(vals, &(&1 == hd(vals))), do: hd(vals), else: nil
              end

            if tp != nil do
              {[tp | bits], st, (cur <<< 1 ||| tp) &&& 0xFF}
            else
              cx =
                if template == 0 do
                  [ax, ay, bx, by] = grat
                  (cur &&& 1) ||| rpx(p1, x + 1, w) <<< 1 ||| rpx(p1, x, w) <<< 2 |||
                    cur_or_above(cur, p1, x, ax, ay, w) <<< 3 |||
                    px(ref, rx + 1, ry + 1) <<< 4 ||| px(ref, rx, ry + 1) <<< 5 ||| px(ref, rx - 1, ry + 1) <<< 6 |||
                    px(ref, rx + 1, ry) <<< 7 ||| px(ref, rx, ry) <<< 8 ||| px(ref, rx - 1, ry) <<< 9 |||
                    px(ref, rx + 1, ry - 1) <<< 10 ||| px(ref, rx, ry - 1) <<< 11 ||| px(ref, rx + bx, ry + by) <<< 12
                else
                  (cur &&& 1) ||| rpx(p1, x + 1, w) <<< 1 ||| rpx(p1, x, w) <<< 2 ||| rpx(p1, x - 1, w) <<< 3 |||
                    px(ref, rx + 1, ry + 1) <<< 4 ||| px(ref, rx, ry + 1) <<< 5 ||| px(ref, rx + 1, ry) <<< 6 |||
                    px(ref, rx, ry) <<< 7 ||| px(ref, rx - 1, ry) <<< 8 ||| px(ref, rx, ry - 1) <<< 9
                end

              {b, st} = decode_bit(st, stats, cx)
              {[b | bits], st, (cur <<< 1 ||| b) &&& 0xFF}
            end
          end)

        {[bits |> Enum.reverse() |> List.to_tuple() | rows], st, ltp}
      end)

    {%{w: w, h: h, rows: rows |> Enum.reverse() |> List.to_tuple()}, st}
  end

  # an adaptive pixel of the region being refined: the current row (from
  # the register) or the row above
  defp cur_or_above(cur, _p1, x, ax, 0, _w), do: if(ax < 0 and x + ax >= 0, do: cur >>> (-ax - 1) &&& 1, else: 0)
  defp cur_or_above(_cur, p1, x, ax, -1, w), do: rpx(p1, x + ax, w)
  defp cur_or_above(_cur, _p1, _x, _ax, _ay, _w), do: 0

  # ------------------------------------------------------- symbol dictionary --

  defp symbol_dict(data, insyms, ws) do
    <<flags::16, rest::binary>> = data
    sdhuff = flags &&& 1
    refagg = flags >>> 1 &&& 1
    sdtemplate = flags >>> 10 &&& 3
    sdrtemplate = flags >>> 12 &&& 1
    retained = flags >>> 9 &&& 1

    cond do
      sdhuff == 1 ->
        {:error, "a Huffman-coded symbol dictionary (SDHUFF; Annex B tables are not implemented)"}

      true ->
        {at, rest} = at_pairs(rest, if(sdtemplate == 0, do: 4, else: 1))
        {rat, rest} = if refagg == 1 and sdrtemplate == 0, do: at_pairs(rest, 2), else: {[], rest}
        <<nex::32, nnew::32, body::binary>> = rest
        ws = if retained == 1, do: ["a symbol dictionary marks its contexts retained (not reused by later dictionaries here)" | ws], else: ws
        nin = length(insyms)
        st = mq_new(body)
        gb = stats(65_536)
        gr = stats(8192)
        ctx = %{iadh: stats(512), iadw: stats(512), iaex: stats(512), iaai: stats(512), iardx: stats(512), iardy: stats(512),
                iaid_len: codelen(nin + nnew)}
        ctx = Map.put(ctx, :iaid, stats(1 <<< (ctx.iaid_len + 1)))
        text_ctx = %{iadt: stats(512), iafs: stats(512), iads: stats(512), iait: stats(512), iari: stats(512), iardw: stats(512),
                     iardh: stats(512), iardx: ctx.iardx, iardy: ctx.iardy, iaid: ctx.iaid}
        insyms_t = List.to_tuple(insyms)

        {news, st} = height_classes(st, nnew, 0, [], gb, gr, ctx, text_ctx, %{at: at, template: sdtemplate, rtemplate: sdrtemplate,
                                                                                 rat: rat, refagg: refagg, insyms: insyms_t, nin: nin})

        # the export flags (6.5.10): runs alternating not-exported / exported
        all = insyms ++ news
        {flags, _st} = export_runs(st, ctx.iaex, length(all), 0, 0, [])
        exported = for {s, 1} <- Enum.zip(all, flags), do: s
        {:ok, Enum.take(exported, nex), ws}
    end
  end

  defp codelen(n), do: Enum.find(0..32, fn l -> 1 <<< l >= n end)

  defp at_pairs(bin, n) do
    <<raw::binary-size(2 * n), rest::binary>> = bin
    {for(<<dx::signed-8, dy::signed-8 <- raw>>, do: {dx, dy}), rest}
  end

  defp height_classes(st, nnew, hc, acc, gb, gr, ctx, tctx, p) do
    if length(acc) >= nnew do
      {Enum.reverse(acc), st}
    else
      {dh, st} = int(st, ctx.iadh)
      hc = hc + dh
      {acc, st} = symbols_in_class(st, nnew, hc, 0, acc, gb, gr, ctx, tctx, p)
      height_classes(st, nnew, hc, acc, gb, gr, ctx, tctx, p)
    end
  end

  defp symbols_in_class(st, nnew, hc, symw, acc, gb, gr, ctx, tctx, p) do
    {dw, st} = int(st, ctx.iadw)

    if dw == :oob or length(acc) >= nnew do
      {acc, st}
    else
      symw = symw + dw

      {bm, st} =
        if p.refagg == 0 do
          generic(st, gb, symw, hc, p.template, p.at, false)
        else
          {n, st} = int(st, ctx.iaai)
          known = List.to_tuple(Tuple.to_list(p.insyms) ++ Enum.reverse(acc))

          if n == 1 do
            # one instance: a refinement of a known symbol
            {id, st} = iaid(st, ctx.iaid, ctx.iaid_len)
            {rdx, st} = int(st, ctx.iardx)
            {rdy, st} = int(st, ctx.iardy)
            refine(st, gr, symw, hc, p.rtemplate, elem(known, id), rdx, rdy, flat(p.rat), false)
          else
            # an aggregate: a small text region of known symbols
            tp = %{strips: 1, logstrips: 0, refcorner: 1, transposed: false, combop: 0, defpixel: 0, dsoffset: 0, refine: true,
                   rtemplate: p.rtemplate, rat: flat(p.rat), ninst: n}
            text_decode(st, tctx, gr, symw, hc, tp, known, ctx.iaid_len)
          end
        end

      symbols_in_class(st, nnew, hc, symw, [bm | acc], gb, gr, ctx, tctx, p)
    end
  end

  defp flat(pairs), do: Enum.flat_map(pairs, fn {a, b} -> [a, b] end)

  defp export_runs(st, _stats, total, i, _flag, acc) when i >= total, do: {Enum.reverse(acc), st}

  defp export_runs(st, stats, total, i, flag, acc) do
    case int(st, stats) do
      {:oob, st} -> {Enum.reverse(acc), st}
      {n, st} ->
        n = min(max(n, 0), total - i)
        export_runs(st, stats, total, i + n, 1 - flag, List.duplicate(flag, n) ++ acc)
    end
  end

  # ------------------------------------------------------------ text region --

  defp text_region(data, syms, ws) do
    <<w::32, h::32, x::32, y::32, eflags, flags::16, rest::binary>> = data
    sbhuff = flags &&& 1

    if sbhuff == 1 do
      {:error, "a Huffman-coded text region (SBHUFF; Annex B tables are not implemented)"}
    else
      refine = (flags >>> 1 &&& 1) == 1
      logstrips = flags >>> 2 &&& 3
      rtemplate = flags >>> 15 &&& 1
      ds = flags >>> 10 &&& 31
      ds = if ds >= 16, do: ds - 32, else: ds
      {rat, rest} = if refine and rtemplate == 0, do: at_pairs(rest, 2), else: {[], rest}
      <<ninst::32, body::binary>> = rest

      tp = %{strips: 1 <<< logstrips, logstrips: logstrips, refcorner: flags >>> 4 &&& 3, transposed: (flags >>> 6 &&& 1) == 1,
             combop: flags >>> 7 &&& 3, defpixel: flags >>> 9 &&& 1, dsoffset: ds, refine: refine, rtemplate: rtemplate,
             rat: flat(rat), ninst: ninst}

      n = length(syms)
      len = codelen(n)
      st = mq_new(body)
      tctx = %{iadt: stats(512), iafs: stats(512), iads: stats(512), iait: stats(512), iari: stats(512), iardw: stats(512),
               iardh: stats(512), iardx: stats(512), iardy: stats(512), iaid: stats(1 <<< (len + 1))}
      {bm, _st} = text_decode(st, tctx, stats(8192), w, h, tp, List.to_tuple(syms), len)
      {:ok, {bm, x, y, eflags &&& 7}, ws}
    end
  end

  # text region decoding (6.4.5), arithmetic: `{bitmap, st}`
  defp text_decode(st, c, gr, w, h, tp, syms, len) do
    page = canvas(w, h, tp.defpixel)
    {dt, st} = int(st, c.iadt)
    stript = -dt * tp.strips
    st = strips(st, c, gr, tp, syms, len, page, w, h, stript, 0, 0)
    {freeze(page, w, h), st}
  end

  defp strips(st, _c, _gr, %{ninst: n}, _syms, _len, _page, _w, _h, _stript, _firsts, done) when done >= n, do: st

  defp strips(st, c, gr, tp, syms, len, page, w, h, stript, firsts, done) do
    {dt, st} = int(st, c.iadt)
    stript = stript + dt * tp.strips
    {dfs, st} = int(st, c.iafs)
    firsts = firsts + dfs
    {st, done} = instances(st, c, gr, tp, syms, len, page, w, h, stript, firsts, done, true)
    strips(st, c, gr, tp, syms, len, page, w, h, stript, firsts, done)
  end

  defp instances(st, c, gr, tp, syms, len, page, w, h, stript, curs, done, first) do
    {curs, st, go} =
      if first do
        {curs, st, true}
      else
        case int(st, c.iads) do
          {:oob, st} -> {curs, st, false}
          {ids, st} -> {curs + ids + tp.dsoffset, st, true}
        end
      end

    if not go or done >= tp.ninst do
      {st, done}
    else
      {curt, st} = if tp.strips == 1, do: {0, st}, else: int(st, c.iait)
      t = stript + curt
      {id, st} = iaid(st, c.iaid, len)
      ib = if id < tuple_size(syms), do: elem(syms, id)

      {ri, st} = if tp.refine, do: int(st, c.iari), else: {0, st}

      {ib, st} =
        if ri == 1 and ib do
          {rdw, st} = int(st, c.iardw)
          {rdh, st} = int(st, c.iardh)
          {rdx, st} = int(st, c.iardx)
          {rdy, st} = int(st, c.iardy)
          refine(st, gr, ib.w + rdw, ib.h + rdh, tp.rtemplate, ib, (rdw >>> 1) + rdx, (rdh >>> 1) + rdy, tp.rat, false)
        else
          {ib, st}
        end

      {iw, ih} = if ib, do: {ib.w, ib.h}, else: {1, 1}

      curs =
        cond do
          not tp.transposed and tp.refcorner > 1 -> curs + iw - 1
          tp.transposed and (tp.refcorner &&& 1) == 0 -> curs + ih - 1
          true -> curs
        end

      s = curs

      {x, y} =
        case {tp.transposed, tp.refcorner} do
          {false, 1} -> {s, t}
          {false, 3} -> {s - iw + 1, t}
          {false, 0} -> {s, t - ih + 1}
          {false, 2} -> {s - iw + 1, t - ih + 1}
          {true, 1} -> {t, s}
          {true, 3} -> {t - iw + 1, s}
          {true, 0} -> {t, s - ih + 1}
          {true, 2} -> {t - iw + 1, s - ih + 1}
        end

      if ib, do: compose(page, w, h, ib, x, y, tp.combop)

      curs =
        cond do
          not tp.transposed and tp.refcorner < 2 -> curs + iw - 1
          tp.transposed and (tp.refcorner &&& 1) == 1 -> curs + ih - 1
          true -> curs
        end

      instances(st, c, gr, tp, syms, len, page, w, h, stript, curs, done + 1, false)
    end
  end

  # ------------------------------------------------------------- segments --

  @doc """
  Parse the segments of a JBIG2 stream: the embedded organisation of a PDF
  stream (segment after segment) or a whole `.jb2` file (with its header,
  sequential or random-access). `[%{number, type, page, refs, data}]`.
  """
  def segments(<<0x97, "JB2", 0x0D, 0x0A, 0x1A, 0x0A, flags, rest::binary>>) do
    rest = if (flags &&& 2) == 0, do: binary_part(rest, 4, byte_size(rest) - 4), else: rest

    if (flags &&& 1) == 1 do
      parse(rest, [])
    else
      # random access: every header first, then every segment's data in order
      {headers, data} = headers(rest, [])
      {segs, _} = Enum.map_reduce(headers, data, fn h, d ->
        n = min(h.length, byte_size(d))
        {Map.put(h, :data, binary_part(d, 0, n)), binary_part(d, n, byte_size(d) - n)}
      end)
      segs
    end
  end

  def segments(bin), do: parse(bin, [])

  defp parse(bin, acc) do
    case header(bin) do
      {:ok, h, rest} ->
        n = if h.length == 0xFFFFFFFF, do: unknown_length(rest, h), else: min(h.length, byte_size(rest))
        seg = Map.put(h, :data, binary_part(rest, 0, n))
        rest = binary_part(rest, n, byte_size(rest) - n)
        if h.type == 51, do: Enum.reverse([seg | acc]), else: parse(rest, [seg | acc])

      :done ->
        Enum.reverse(acc)
    end
  end

  defp headers(bin, acc) do
    case header(bin) do
      {:ok, %{type: 51} = h, rest} -> {Enum.reverse([h | acc]), rest}
      {:ok, h, rest} -> headers(rest, [h | acc])
      :done -> {Enum.reverse(acc), <<>>}
    end
  end

  defp header(<<num::32, flags, rest::binary>>) do
    type = flags &&& 63
    big_page = (flags &&& 64) != 0

    {nref, rest} =
      case rest do
        <<r, more::binary>> when r >>> 5 < 7 -> {r >>> 5, more}
        <<long::32, more::binary>> ->
          n = long &&& 0x1FFFFFFF
          skip = div(n + 8, 8)
          if byte_size(more) >= skip, do: {n, binary_part(more, skip, byte_size(more) - skip)}, else: {0, <<>>}
        _ -> {0, <<>>}
      end

    size = cond do
      num <= 256 -> 1
      num <= 65_536 -> 2
      true -> 4
    end

    with <<refs_raw::binary-size(nref * size), rest::binary>> <- rest,
         {page, rest} <- (if big_page, do: (case rest do <<p::32, r::binary>> -> {p, r}; _ -> nil end),
                                   else: (case rest do <<p, r::binary>> -> {p, r}; _ -> nil end)),
         <<len::32, rest::binary>> <- rest do
      refs = for <<r::size(size * 8) <- refs_raw>>, do: r
      {:ok, %{number: num, type: type, page: page, refs: refs, length: len}, rest}
    else
      _ -> :done
    end
  end

  defp header(_), do: :done

  # an immediate generic region of unknown length (7.2.7): its data ends at
  # the marker (FF AC arithmetic, 00 00 MMR) followed by the row count
  defp unknown_length(rest, _h) do
    mmr = byte_size(rest) > 17 and (:binary.at(rest, 17) &&& 1) == 1
    marker = if mmr, do: <<0, 0>>, else: <<0xFF, 0xAC>>

    case :binary.match(rest, marker, scope: {min(18, byte_size(rest)), max(byte_size(rest) - min(18, byte_size(rest)), 0)}) do
      {at, 2} -> min(at + 6, byte_size(rest))
      :nomatch -> byte_size(rest)
    end
  end

  # ------------------------------------------------------------------ pages --

  @doc """
  Decode the first page of a JBIG2 stream. `globals`: the segments of a
  PDF's `JBIG2Globals` stream (or `nil`). Returns `{:ok, %{w, h, rows,
  warnings}}` or `{:error, reason}`.
  """
  def decode(bin, globals \\ nil) do
    segs = segments(globals || <<>>) ++ segments(bin)

    case Enum.find(segs, &(&1.type == 48)) do
      nil ->
        {:error, "no page information segment"}

      pinfo ->
        <<pw::32, ph::32, _xr::32, _yr::32, pflags, striping::16, _::binary>> = pinfo.data <> <<0::size(19 * 8)>>
        default = pflags >>> 2 &&& 1
        striped = (striping &&& 0x8000) != 0
        # a page of unknown height (striped) grows to its end-of-stripe rows
        ph = if ph == 0xFFFFFFFF, do: stripe_height(segs, pinfo.page), else: ph

        cond do
          pw == 0 or ph == 0 or pw * ph > 256_000_000 -> {:error, "a page of #{pw}×#{ph}"}
          true ->
            page = canvas(pw, ph, default)
            _ = striped
            {dicts, ws} = run(segs, pinfo.page, page, pw, ph, %{}, [])
            _ = dicts
            {:ok, freeze(page, pw, ph) |> Map.put(:warnings, Enum.reverse(ws) |> Enum.uniq())}
        end
    end
  rescue
    e in [MatchError, ArgumentError, FunctionClauseError, ArithmeticError, CaseClauseError] ->
      {:error, "a damaged JBIG2 stream (#{Exception.message(e) |> String.slice(0, 80)})"}
  end

  defp stripe_height(segs, page) do
    segs
    |> Enum.filter(&(&1.type == 50 and &1.page == page))
    |> Enum.map(fn %{data: <<row::32, _::binary>>} -> row + 1; _ -> 0 end)
    |> Enum.max(fn -> 0 end)
  end

  # run every segment in order; symbol dictionaries are kept by number
  defp run([], _page_no, _page, _pw, _ph, dicts, ws), do: {dicts, ws}

  defp run([s | rest], page_no, page, pw, ph, dicts, ws) do
    {dicts, ws} =
      cond do
        s.page not in [0, page_no] ->
          {dicts, ws}

        s.type == 0 ->
          insyms = Enum.flat_map(s.refs, &Map.get(dicts, &1, []))

          case symbol_dict(s.data, insyms, []) do
            {:ok, syms, w2} -> {Map.put(dicts, s.number, syms), w2 ++ ws}
            {:error, why} -> {dicts, [why | ws]}
          end

        s.type in [4, 6, 7] ->
          syms = Enum.flat_map(s.refs, &Map.get(dicts, &1, []))

          case text_region(s.data, syms, []) do
            {:ok, {bm, x, y, op}, w2} -> place(page, pw, ph, bm, x, y, op, s.type); {dicts, w2 ++ ws}
            {:error, why} -> {dicts, [why | ws]}
          end

        s.type in [36, 38, 39] ->
          case generic_region(s.data) do
            {:ok, {bm, x, y, op}} -> place(page, pw, ph, bm, x, y, op, s.type); {dicts, ws}
            {:error, why} -> {dicts, [why | ws]}
          end

        s.type in [40, 42, 43] ->
          case refinement_region(s.data, page, pw, ph) do
            {:ok, {bm, x, y, op}} -> place(page, pw, ph, bm, x, y, op, s.type); {dicts, ws}
            {:error, why} -> {dicts, [why | ws]}
          end

        s.type in [16, 20, 22, 23] ->
          {dicts, ["halftone regions and pattern dictionaries (7.4.4–7.4.5) are not decoded here" | ws]}

        s.type == 53 ->
          {dicts, ["custom Huffman tables (7.4.13) are not used here" | ws]}

        true ->
          {dicts, ws}
      end

    run(rest, page_no, page, pw, ph, dicts, ws)
  end

  # intermediate regions (types 4, 36, 40) are not drawn on the page
  defp place(_page, _pw, _ph, _bm, _x, _y, _op, type) when type in [4, 36, 40], do: :ok
  defp place(page, pw, ph, bm, x, y, op, _type), do: compose(page, pw, ph, bm, x, y, op)

  defp generic_region(<<w::32, h::32, x::32, y::32, eflags, flags, rest::binary>>) do
    mmr = flags &&& 1
    template = flags >>> 1 &&& 3
    tpgdon = (flags >>> 3 &&& 1) == 1

    cond do
      w == 0 or h == 0 or w * h > 256_000_000 -> {:error, "a generic region of #{w}×#{h}"}

      mmr == 1 ->
        case Vapor.Docs.CCITT.decode(rest, k: -1, columns: w, rows: h, black_is_1: true) do
          {:ok, %{data: data}} -> {:ok, {unpack(data, w, h), x, y, eflags &&& 7}}
          {:error, why} -> {:error, "an MMR generic region (#{inspect(why)})"}
        end

      true ->
        {at, body} = at_pairs(rest, if(template == 0, do: 4, else: 1))
        {bm, _st} = generic(mq_new(body), stats(65_536), w, h, template, at, tpgdon)
        {:ok, {bm, x, y, eflags &&& 7}}
    end
  end

  defp generic_region(_), do: {:error, "a truncated generic region"}

  # a refinement region refines the page's own pixels under it (7.4.7)
  defp refinement_region(<<w::32, h::32, x::32, y::32, eflags, flags, rest::binary>>, page, pw, ph) do
    template = flags &&& 1
    tpgron = (flags >>> 1 &&& 1) == 1
    {rat, body} = if template == 0, do: at_pairs(rest, 2), else: {[], rest}

    ref = %{w: w, h: h, rows: List.to_tuple(for(j <- 0..(h - 1)//1, do: List.to_tuple(for(i <- 0..(w - 1)//1, do:
      if(x + i < pw and y + j < ph, do: :atomics.get(page, (y + j) * pw + x + i + 1), else: 0)))))}

    {bm, _} = refine(mq_new(body), stats(8192), w, h, template, ref, 0, 0, flat(rat), tpgron)
    {:ok, {bm, x, y, eflags &&& 7}}
  end

  defp refinement_region(_, _, _, _), do: {:error, "a truncated refinement region"}

  defp unpack(data, w, h) do
    stride = div(w + 7, 8)
    data = if byte_size(data) < stride * h, do: data <> :binary.copy(<<0>>, stride * h - byte_size(data)), else: data

    rows =
      for y <- 0..(h - 1)//1 do
        <<row::bitstring-size(w), _::bitstring>> = binary_part(data, y * stride, stride)
        List.to_tuple(for <<b::1 <- row>>, do: b)
      end

    %{w: w, h: h, rows: List.to_tuple(rows)}
  end

  # ------------------------------------------------------------------- PDF --

  @doc """
  The `JBIG2Decode` filter of a PDF: the page decoded with the stream's
  `JBIG2Globals` (if any), returned as PDF image data — 1 bit per pixel,
  rows padded to bytes, **0 = black** (the filter inverts JBIG2's 1 =
  black, as pdf.js and poppler do). `{:ok, data, w, h, warnings}`.
  """
  def pdf(stream, globals \\ nil) do
    with {:ok, %{w: w, h: h} = bm} <- decode(stream, globals) do
      inv = for <<b <- packed(bm)>>, into: <<>>, do: <<bxor(b, 0xFF)>>
      {:ok, inv, w, h, bm.warnings}
    end
  end
end
