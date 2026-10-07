defmodule Vapor.Docs.JPEG do
  @moduledoc """
  A JPEG decoder with no dependency: baseline and extended sequential
  (SOF0/SOF1, 8-bit) and **progressive** (SOF2) Huffman JPEG, any
  sampling factors, restart intervals, grayscale, YCbCr and Adobe RGB.

  It reproduces libjpeg(-turbo)'s default decompression **bit for bit**
  (what Pillow, browsers and most tools show), because every step is the
  integer arithmetic libjpeg specifies, not a floating-point
  approximation of it:

    * the accurate integer IDCT (`jidctint.c`, `JDCT_ISLOW`: 13-bit
      constants, two passes, its range-limit table with the 10-bit wrap);
    * "fancy" upsampling (`jdsample.c`: the triangle filter for h2v1,
      h1v2 and h2v2 with libjpeg's rounding biases and edge rules, context
      rows replicated at the top and bottom), plain replication otherwise;
    * YCbCr → RGB by libjpeg's 16-bit fixed-point tables (`jdcolor.c`).

  Tested against Pillow (libjpeg-turbo) on real photographs and on files
  covering every sampling layout, progressive scans with successive
  approximation, restart markers, odd sizes and optimised Huffman tables:
  identical pixels.

  Refused, with a reason: arithmetic coding, lossless and hierarchical
  JPEG, 12-bit samples, CMYK/YCCK (four components). Coefficients live in
  `:atomics` (progressive refinement updates them in place).

  `decode(bytes)` → `{:ok, %{width, height, channels, mode, pixels}}`
  (`pixels`: interleaved 8-bit samples, row-major) or `{:error, reason}`.
  """
  import Bitwise

  @zigzag {0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
           28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54,
           47, 55, 62, 63}

  @max_pixels 64_000_000

  @doc "Decode a JPEG file's bytes (see the moduledoc)."
  def decode(<<0xFF, 0xD8, rest::binary>>) do
    try do
      st = %{qt: %{}, dc: %{}, ac: %{}, frame: nil, dri: 0, adobe: nil, jfif: false, coefs: nil, scans: 0}
      st = segments(rest, st)
      finish(st)
    catch
      {:jpeg, why} -> {:error, why}
    end
  end

  def decode(_), do: {:error, "not a JPEG (no SOI marker)"}

  defp fail(why), do: throw({:jpeg, why})

  # ------------------------------------------------------------- markers --

  defp segments(<<0xFF, 0xFF, rest::binary>>, st), do: segments(<<0xFF, rest::binary>>, st)
  defp segments(<<0xFF, 0xD9, _::binary>>, st), do: st
  defp segments(<<0xFF, m, _::binary>> = b, st) when m in 0xD0..0xD7, do: segments(binary_part(b, 2, byte_size(b) - 2), st)

  defp segments(<<0xFF, m, len::16, rest::binary>>, st) when len >= 2 do
    cond do
      byte_size(rest) >= len - 2 ->
        <<body::binary-size(len - 2), more::binary>> = rest
        segment(m, body, more, st)

      # a file cut short after its scans began: show what was decoded (as libjpeg)
      st.scans > 0 ->
        st

      true ->
        fail("truncated segment")
    end
  end

  # garbage between segments (some writers pad): skip to the next marker
  defp segments(<<_, rest::binary>>, st), do: segments(rest, st)
  defp segments(<<>>, st), do: st

  defp segment(m, body, more, st) do
    case m do
      0xDB -> segments(more, %{st | qt: dqt(body, st.qt)})
      0xC4 -> segments(more, dht(body, st))
      0xDD -> <<n::16, _::binary>> = body; segments(more, %{st | dri: n})
      0xE0 -> segments(more, %{st | jfif: st.jfif or match?(<<"JFIF", 0, _::binary>>, body)})
      0xEE -> segments(more, %{st | adobe: adobe(body)})
      c when c in [0xC0, 0xC1, 0xC2] -> segments(more, frame(body, c, st))
      c when c in [0xC3, 0xC5, 0xC6, 0xC7, 0xCB, 0xCD, 0xCE, 0xCF] -> fail("lossless or hierarchical JPEG (SOF#{c - 0xC0}) is not decoded")
      c when c in [0xC9, 0xCA] -> fail("arithmetic-coded JPEG is not decoded")
      0xDA ->
        {st, rest2} = scan(body, more, st)
        segments(rest2, %{st | scans: st.scans + 1})
      _ -> segments(more, st)
    end
  end

  defp dqt(<<pq::4, tq::4, rest::binary>>, qt) do
    {vals, more} =
      case pq do
        0 -> <<v::binary-size(64), m::binary>> = rest; {for(<<x <- v>>, do: x), m}
        1 -> <<v::binary-size(128), m::binary>> = rest; {for(<<x::16 <- v>>, do: x), m}
      end

    # stored in zigzag order → natural order
    natural = vals |> Enum.with_index() |> Enum.reduce(Tuple.duplicate(0, 64), fn {q, k}, t -> put_elem(t, elem(@zigzag, k), q) end)
    dqt(more, Map.put(qt, tq, natural))
  end

  defp dqt(<<>>, qt), do: qt

  defp dht(<<tc::4, th::4, counts::binary-size(16), rest::binary>>, st) do
    cs = for <<c <- counts>>, do: c
    n = Enum.sum(cs)
    <<vals::binary-size(n), more::binary>> = rest
    table = huff_table(cs, :binary.bin_to_list(vals))
    st = if tc == 0, do: %{st | dc: Map.put(st.dc, th, table)}, else: %{st | ac: Map.put(st.ac, th, table)}
    dht(more, st)
  end

  defp dht(<<>>, st), do: st

  defp adobe(<<"Adobe", _v::16, _f0::16, _f1::16, t, _::binary>>), do: t
  defp adobe(_), do: nil

  defp frame(<<p, h::16, w::16, n, rest::binary>>, sof, st) do
    if p != 8, do: fail("#{p}-bit samples (only 8-bit JPEG is decoded)")
    if w == 0 or h == 0, do: fail("a frame with no lines or no columns (DNL is not supported)")
    if w * h > @max_pixels, do: fail("#{w}×#{h} exceeds the #{@max_pixels}-pixel limit")
    if n not in [1, 3], do: fail("#{n} components (1 or 3 are decoded; CMYK/YCCK are not)")

    comps =
      for <<id, hs::4, vs::4, tq <- binary_part(rest, 0, n * 3)>> do
        if hs not in 1..4 or vs not in 1..4, do: fail("sampling factors #{hs}×#{vs}")
        %{id: id, h: hs, v: vs, tq: tq}
      end

    hmax = comps |> Enum.map(& &1.h) |> Enum.max()
    vmax = comps |> Enum.map(& &1.v) |> Enum.max()
    mcux = div(w + 8 * hmax - 1, 8 * hmax)
    mcuy = div(h + 8 * vmax - 1, 8 * vmax)

    comps =
      Enum.map(comps, fn c ->
        bw = mcux * c.h
        bh = mcuy * c.v
        # the component's true size in samples and in blocks (non-interleaved scans cover only these)
        dw = div(w * c.h + hmax - 1, hmax)
        dh = div(h * c.v + vmax - 1, vmax)
        Map.merge(c, %{bw: bw, bh: bh, dw: dw, dh: dh, nbw: div(dw + 7, 8), nbh: div(dh + 7, 8), coefs: :atomics.new(bw * bh * 64, signed: true)})
      end)

    %{st | frame: %{w: w, h: h, comps: comps, hmax: hmax, vmax: vmax, mcux: mcux, mcuy: mcuy, progressive: sof == 0xC2}}
  end

  # -------------------------------------------------------------- huffman --

  # canonical codes; a 9-bit lookahead table for the common short codes and
  # the (maxcode, valptr) walk of the standard for the rest
  defp huff_table(counts, vals) do
    {codes, _, _} =
      counts
      |> Enum.with_index(1)
      |> Enum.reduce({[], 0, vals}, fn {cnt, len}, {acc, code, vs} ->
        {mine, rest} = Enum.split(vs, cnt)
        entries = mine |> Enum.with_index() |> Enum.map(fn {sym, i} -> {len, code + i, sym} end)
        {acc ++ entries, (code + cnt) <<< 1, rest}
      end)

    look =
      Enum.reduce(codes, Tuple.duplicate(nil, 512), fn {len, code, sym}, t ->
        if len <= 9 do
          base = code <<< (9 - len)
          Enum.reduce(0..((1 <<< (9 - len)) - 1), t, fn j, t -> put_elem(t, base + j, {len, sym}) end)
        else
          t
        end
      end)

    by_len = Enum.group_by(codes, &elem(&1, 0))

    slow =
      for len <- 10..16, Map.has_key?(by_len, len) do
        cs = by_len[len]
        {len, elem(hd(cs), 1), elem(List.last(cs), 1), cs |> Enum.map(&elem(&1, 2)) |> List.to_tuple()}
      end

    %{look: look, slow: slow}
  end

  # ------------------------------------------------------------- bit reader --
  #
  # A segment of entropy-coded data (between restart markers), unstuffed:
  # {binary, bit position}. Reading past the end yields zero bits, as libjpeg.

  defp peek(bin, pos, n) do
    size = bit_size(bin)

    cond do
      pos + n <= size ->
        <<_::size(pos), v::size(n), _::bitstring>> = bin
        v

      pos >= size ->
        0

      true ->
        avail = size - pos
        <<_::size(pos), v::size(avail)>> = bin
        v <<< (n - avail)
    end
  end

  defp huff(%{look: look, slow: slow}, {bin, pos}) do
    case elem(look, peek(bin, pos, 9)) do
      {len, sym} -> {sym, {bin, pos + len}}
      nil -> huff_slow(slow, bin, pos)
    end
  end

  defp huff_slow(slow, bin, pos) do
    Enum.find_value(slow, fn {len, lo, hi, syms} ->
      code = peek(bin, pos, len)
      if code >= lo and code <= hi, do: {elem(syms, code - lo), {bin, pos + len}}
    end) || fail("a Huffman code not in the table")
  end

  defp receive_extend(0, br), do: {0, br}

  defp receive_extend(s, {bin, pos}) do
    v = peek(bin, pos, s)
    {if(v < 1 <<< (s - 1), do: v - (1 <<< s) + 1, else: v), {bin, pos + s}}
  end

  defp bits(n, {bin, pos}), do: {peek(bin, pos, n), {bin, pos + n}}

  # -------------------------------------------------------------- the scan --

  defp scan(<<n, rest::binary>>, after_header, %{frame: f} = st) do
    if f == nil, do: fail("a scan before the frame header")
    <<specs::binary-size(n * 2), ss, se, ah::4, al::4, _::binary>> = rest

    comps =
      for <<id, td::4, ta::4 <- specs>> do
        ci = Enum.find_index(f.comps, &(&1.id == id)) || fail("a scan names component #{id}, not in the frame")
        {ci, td, ta}
      end

    {segs, rest2} = entropy(after_header)
    sp = %{ss: ss, se: se, ah: ah, al: al}
    kind = scan_kind(f.progressive, sp, length(comps))
    decode_scan(kind, comps, segs, sp, st)
    {st, rest2}
  end

  defp scan_kind(false, _, _), do: :baseline
  defp scan_kind(true, %{ss: 0, ah: 0}, _), do: :dc_first
  defp scan_kind(true, %{ss: 0}, _), do: :dc_refine
  defp scan_kind(true, %{ah: 0}, 1), do: :ac_first
  defp scan_kind(true, _, 1), do: :ac_refine
  defp scan_kind(true, _, _), do: fail("a progressive AC scan with several components")

  # entropy-coded data up to the next marker that is not RSTn, unstuffed
  # (FF 00 → FF), cut at restart markers
  defp entropy(bin), do: entropy(bin, [], [])

  defp entropy(bin, cur, segs) do
    case :binary.match(bin, <<0xFF>>) do
      :nomatch ->
        {Enum.reverse([IO.iodata_to_binary(Enum.reverse([bin | cur])) | segs]), <<>>}

      {i, 1} ->
        <<before::binary-size(i), 0xFF, rest::binary>> = bin

        case rest do
          <<0x00, more::binary>> -> entropy(more, [<<0xFF>>, before | cur], segs)
          <<m, more::binary>> when m in 0xD0..0xD7 -> entropy(more, [], [IO.iodata_to_binary(Enum.reverse([before | cur])) | segs])
          <<0xFF, _::binary>> -> entropy(rest, [before | cur], segs)
          _ -> {Enum.reverse([IO.iodata_to_binary(Enum.reverse([before | cur])) | segs]), <<0xFF, rest::binary>>}
        end
    end
  end

  # The units of a scan: MCUs of an interleaved scan (h×v blocks per
  # component, padding included) or the component's own blocks for a
  # single-component scan (only those inside its true size).
  defp units(f, [{ci, _, _}]) do
    c = Enum.at(f.comps, ci)
    for by <- 0..(c.nbh - 1), bx <- 0..(c.nbw - 1), do: [{ci, by * c.bw + bx}]
  end

  defp units(f, comps) do
    for my <- 0..(f.mcuy - 1), mx <- 0..(f.mcux - 1) do
      for {ci, _, _} <- comps, c = Enum.at(f.comps, ci), y <- 0..(c.v - 1), x <- 0..(c.h - 1),
          do: {ci, (my * c.v + y) * c.bw + mx * c.h + x}
    end
  end

  defp decode_scan(kind, comps, segs, sp, %{frame: f, dri: dri} = st) do
    tables = Map.new(comps, fn {ci, td, ta} -> {ci, {st.dc[td], st.ac[ta]}} end)
    us = units(f, comps)
    groups = if dri > 0, do: Enum.chunk_every(us, dri), else: [us]
    groups = Enum.zip(groups, segs ++ List.duplicate(<<>>, max(length(groups) - length(segs), 0)))

    for {group, seg} <- groups do
      # predictors and the EOB run restart with every interval
      Enum.reduce(group, {{seg, 0}, %{}, 0}, fn unit, {br, preds, eobrun} ->
        Enum.reduce(unit, {br, preds, eobrun}, fn {ci, blk}, {br, preds, eobrun} ->
          c = Enum.at(f.comps, ci)
          {dc, ac} = tables[ci]
          block(kind, c.coefs, blk * 64, dc, ac, br, preds, ci, eobrun, sp)
        end)
      end)
    end

    :ok
  end

  defp put(a, i, v), do: :atomics.put(a, i + 1, v)
  defp get(a, i), do: :atomics.get(a, i + 1)

  defp block(:baseline, a, base, dc, ac, br, preds, ci, eob, _sp) do
    {s, br} = huff(dc || fail("a missing DC table"), br)
    {d, br} = receive_extend(s, br)
    pred = Map.get(preds, ci, 0) + d
    put(a, base, pred)
    br = ac_seq(a, base, ac || fail("a missing AC table"), br, 1, 63, 0)
    {br, Map.put(preds, ci, pred), eob}
  end

  defp block(:dc_first, a, base, dc, _ac, br, preds, ci, eob, %{al: al}) do
    {s, br} = huff(dc || fail("a missing DC table"), br)
    {d, br} = receive_extend(s, br)
    pred = Map.get(preds, ci, 0) + d
    put(a, base, pred <<< al)
    {br, Map.put(preds, ci, pred), eob}
  end

  defp block(:dc_refine, a, base, _dc, _ac, br, preds, _ci, eob, %{al: al}) do
    {b, br} = bits(1, br)
    if b == 1, do: put(a, base, get(a, base) ||| (1 <<< al))
    {br, preds, eob}
  end

  defp block(:ac_first, _a, _base, _dc, _ac, br, preds, _ci, eob, _sp) when eob > 0, do: {br, preds, eob - 1}

  defp block(:ac_first, a, base, _dc, ac, br, preds, _ci, 0, %{ss: ss, se: se, al: al}) do
    {br, eob} = ac_first(a, base, ac || fail("a missing AC table"), br, ss, se, al)
    {br, preds, eob}
  end

  defp block(:ac_refine, a, base, _dc, ac, br, preds, _ci, eob, %{ss: ss, se: se, al: al}) do
    {br, eob} = ac_refine(a, base, ac || fail("a missing AC table"), br, ss, se, 1 <<< al, -1 <<< al, eob)
    {br, preds, eob}
  end

  # sequential AC: run/size pairs until EOB
  defp ac_seq(_a, _base, _ac, br, k, se, _al) when k > se, do: br

  defp ac_seq(a, base, ac, br, k, se, al) do
    {rs, br} = huff(ac, br)
    {r, s} = {rs >>> 4, rs &&& 15}

    cond do
      s != 0 ->
        k = k + r
        if k > 63, do: fail("an AC run past the end of the block")
        {v, br} = receive_extend(s, br)
        put(a, base + elem(@zigzag, k), v <<< al)
        ac_seq(a, base, ac, br, k + 1, se, al)

      r == 15 -> ac_seq(a, base, ac, br, k + 16, se, al)
      true -> br
    end
  end

  # progressive AC, first pass: returns the EOB run left for the next blocks
  defp ac_first(_a, _base, _ac, br, k, se, _al) when k > se, do: {br, 0}

  defp ac_first(a, base, ac, br, k, se, al) do
    {rs, br} = huff(ac, br)
    {r, s} = {rs >>> 4, rs &&& 15}

    cond do
      s != 0 ->
        k = k + r
        if k > 63, do: fail("an AC run past the end of the block")
        {v, br} = receive_extend(s, br)
        put(a, base + elem(@zigzag, k), v <<< al)
        ac_first(a, base, ac, br, k + 1, se, al)

      r == 15 ->
        ac_first(a, base, ac, br, k + 16, se, al)

      true ->
        {extra, br} = if r > 0, do: bits(r, br), else: {0, br}
        # this block ends the run: EOBRUN = 2^r + extra, minus this block
        {br, (1 <<< r) + extra - 1}
    end
  end

  # progressive AC, refinement (libjpeg's decode_mcu_AC_refine)
  defp ac_refine(a, base, _ac, br, ss, se, p1, m1, eob) when eob > 0 do
    br = refine_rest(a, base, br, ss, se, p1, m1)
    {br, eob - 1}
  end

  defp ac_refine(a, base, ac, br, k, se, p1, m1, 0) when k <= se do
    {rs, br} = huff(ac, br)
    {r, s} = {rs >>> 4, rs &&& 15}

    {s_val, r, eob, br} =
      cond do
        s != 0 ->
          {b, br} = bits(1, br)
          {if(b == 1, do: p1, else: m1), r, 0, br}

        r != 15 ->
          {extra, br} = if r > 0, do: bits(r, br), else: {0, br}
          {0, nil, (1 <<< r) + extra, br}

        true ->
          {0, 15, 0, br}
      end

    if r == nil do
      # an EOB run starts here: refine the rest of this block, count it
      br = refine_rest(a, base, br, k, se, p1, m1)
      {br, eob - 1}
    else
      # skip r zero-history coefficients, refining the nonzero ones passed
      {k, br} = skip(a, base, br, k, se, r, p1, m1)

      if s_val != 0 do
        if k > se, do: fail("a refinement coefficient past the band")
        put(a, base + elem(@zigzag, k), s_val)
      end

      ac_refine(a, base, ac, br, k + 1, se, p1, m1, 0)
    end
  end

  defp ac_refine(_a, _base, _ac, br, _k, _se, _p1, _m1, eob), do: {br, eob}

  # advance over coefficients: nonzero ones get a correction bit; stop at
  # the (r+1)-th zero one, returning its index
  defp skip(_a, _base, br, k, se, _r, _p1, _m1) when k > se, do: {k, br}

  defp skip(a, base, br, k, se, r, p1, m1) do
    i = base + elem(@zigzag, k)
    v = get(a, i)

    if v != 0 do
      br = correct(a, i, v, br, p1, m1)
      skip(a, base, br, k + 1, se, r, p1, m1)
    else
      if r == 0, do: {k, br}, else: skip(a, base, br, k + 1, se, r - 1, p1, m1)
    end
  end

  defp refine_rest(_a, _base, br, k, se, _p1, _m1) when k > se, do: br

  defp refine_rest(a, base, br, k, se, p1, m1) do
    i = base + elem(@zigzag, k)
    v = get(a, i)
    br = if v != 0, do: correct(a, i, v, br, p1, m1), else: br
    refine_rest(a, base, br, k + 1, se, p1, m1)
  end

  defp correct(a, i, v, br, p1, m1) do
    {b, br} = bits(1, br)
    if b == 1 and (v &&& p1) == 0, do: put(a, i, if(v >= 0, do: v + p1, else: v + m1))
    br
  end

  # ------------------------------------------------------------- output --

  defp finish(%{frame: nil}), do: {:error, "no frame header"}
  defp finish(%{scans: 0}), do: {:error, "no scan"}

  defp finish(%{frame: f} = st) do
    planes =
      Enum.map(f.comps, fn c ->
        q = st.qt[c.tq] || fail("a missing quantisation table #{c.tq}")
        {c, idct_plane(c, q)}
      end)

    full = Enum.map(planes, fn {c, plane} -> upsample(c, plane, f) end)

    {mode, pixels} =
      case {full, colour(st)} do
        {[y], _} -> {"L", y |> Enum.map(&row_bin/1) |> IO.iodata_to_binary()}
        {[a, b, c], :rgb} -> {"RGB", interleave(a, b, c)}
        {[y, cb, cr], :ycc} -> {"RGB", ycc_rgb(y, cb, cr)}
      end

    {:ok, %{width: f.w, height: f.h, channels: if(mode == "L", do: 1, else: 3), mode: mode, pixels: pixels,
            progressive: f.progressive, sampling: Enum.map(f.comps, &{&1.h, &1.v})}}
  end

  # libjpeg's default_decompress_parms: JFIF → YCbCr; Adobe transform 0 →
  # RGB; component ids 'R','G','B' → RGB; otherwise YCbCr
  defp colour(%{jfif: true}), do: :ycc
  defp colour(%{adobe: 0}), do: :rgb
  defp colour(%{adobe: t}) when is_integer(t), do: :ycc
  defp colour(%{frame: %{comps: [%{id: ?R}, %{id: ?G}, %{id: ?B}]}}), do: :rgb
  defp colour(_), do: :ycc

  defp row_bin(row), do: :erlang.list_to_binary(Tuple.to_list(row))

  # every block of a component through the integer IDCT: rows of samples
  # (bh·8 rows of bw·8 samples), as tuples
  defp idct_plane(c, q) do
    for by <- 0..(c.bh - 1) do
      blocks = for bx <- 0..(c.bw - 1), do: idct(c.coefs, (by * c.bw + bx) * 64, q)
      # 8 rows across the blocks of this block row
      for r <- 0..7, do: blocks |> Enum.flat_map(&elem(&1, r)) |> List.to_tuple()
    end
    |> Enum.concat()
    |> List.to_tuple()
  end

  @c0_298 2446
  @c0_390 3196
  @c0_541 4433
  @c0_765 6270
  @c0_899 7373
  @c1_175 9633
  @c1_501 12299
  @c1_847 15137
  @c1_961 16069
  @c2_053 16819
  @c2_562 20995
  @c3_072 25172

  # jidctint.c, JDCT_ISLOW: returns 8 rows, each a list of 8 samples
  defp idct(a, base, q) do
    co = for k <- 0..63, do: get(a, base + k) * elem(q, k)
    co = List.to_tuple(co)

    # pass 1: columns → workspace (descaled by CONST_BITS − PASS1_BITS = 11)
    cols =
      for x <- 0..7 do
        c = fn r -> elem(co, r * 8 + x) end
        {c0, c1, c2, c3, c4, c5, c6, c7} = {c.(0), c.(1), c.(2), c.(3), c.(4), c.(5), c.(6), c.(7)}

        if c1 == 0 and c2 == 0 and c3 == 0 and c4 == 0 and c5 == 0 and c6 == 0 and c7 == 0 do
          v = c0 <<< 2
          {v, v, v, v, v, v, v, v}
        else
          butterfly({c0, c1, c2, c3, c4, c5, c6, c7}, 11)
        end
      end
      |> List.to_tuple()

    # pass 2: rows → samples (descaled by CONST_BITS + PASS1_BITS + 3 = 18)
    for y <- 0..7 do
      w = fn x -> elem(elem(cols, x), y) end
      {w0, w1, w2, w3, w4, w5, w6, w7} = {w.(0), w.(1), w.(2), w.(3), w.(4), w.(5), w.(6), w.(7)}

      if w1 == 0 and w2 == 0 and w3 == 0 and w4 == 0 and w5 == 0 and w6 == 0 and w7 == 0 do
        v = limit((w0 + 16) >>> 5)
        [v, v, v, v, v, v, v, v]
      else
        {o0, o1, o2, o3, o4, o5, o6, o7} = butterfly({w0, w1, w2, w3, w4, w5, w6, w7}, 18)
        [limit(o0), limit(o1), limit(o2), limit(o3), limit(o4), limit(o5), limit(o6), limit(o7)]
      end
    end
    |> List.to_tuple()
  end

  # the even/odd butterfly of jidctint.c, outputs descaled by `n` bits
  defp butterfly({i0, i1, i2, i3, i4, i5, i6, i7}, n) do
    z1 = (i2 + i6) * @c0_541
    tmp2 = z1 - i6 * @c1_847
    tmp3 = z1 + i2 * @c0_765
    tmp0 = (i0 + i4) <<< 13
    tmp1 = (i0 - i4) <<< 13
    tmp10 = tmp0 + tmp3
    tmp13 = tmp0 - tmp3
    tmp11 = tmp1 + tmp2
    tmp12 = tmp1 - tmp2

    {t0, t1, t2, t3} = {i7, i5, i3, i1}
    z1 = t0 + t3
    z2 = t1 + t2
    z3 = t0 + t2
    z4 = t1 + t3
    z5 = (z3 + z4) * @c1_175
    t0 = t0 * @c0_298
    t1 = t1 * @c2_053
    t2 = t2 * @c3_072
    t3 = t3 * @c1_501
    z1 = -z1 * @c0_899
    z2 = -z2 * @c2_562
    z3 = -z3 * @c1_961 + z5
    z4 = -z4 * @c0_390 + z5
    t0 = t0 + z1 + z3
    t1 = t1 + z2 + z4
    t2 = t2 + z2 + z3
    t3 = t3 + z1 + z4

    half = 1 <<< (n - 1)
    d = fn x -> (x + half) >>> n end
    {d.(tmp10 + t3), d.(tmp11 + t2), d.(tmp12 + t1), d.(tmp13 + t0), d.(tmp13 - t0), d.(tmp12 - t1), d.(tmp11 - t2), d.(tmp10 - t3)}
  end

  # libjpeg's post-IDCT range-limit table, indexed by x & 1023
  defp limit(x) do
    v = x &&& 1023

    cond do
      v < 128 -> v + 128
      v < 512 -> 255
      v < 896 -> 0
      true -> v - 896
    end
  end

  # -------------------------------------------------------------- upsampling --

  # rows (tuples) of the component at full resolution, cropped to w × h
  defp upsample(c, plane, f) do
    {hx, vx} = {div(f.hmax, c.h), div(f.vmax, c.v)}
    if rem(f.hmax, c.h) != 0 or rem(f.vmax, c.v) != 0, do: fail("non-integral sampling ratios")
    row = fn y -> elem(plane, y |> max(0) |> min(c.dh - 1)) end
    crop = fn list -> list |> Enum.take(f.w) |> List.to_tuple() end

    case {hx, vx} do
      {1, 1} ->
        for y <- 0..(f.h - 1), do: elem(plane, y) |> Tuple.to_list() |> crop.()

      {2, 1} when c.dw > 2 ->
        for y <- 0..(f.h - 1), do: h2v1(row.(y), c.dw) |> crop.()

      {1, 2} ->
        for y <- 0..(f.h - 1) do
          {inr, odd} = {div(y, 2), rem(y, 2)}
          {near, far, bias} = if odd == 0, do: {row.(inr), row.(inr - 1), 1}, else: {row.(inr), row.(inr + 1), 2}
          for(x <- 0..(c.dw - 1), do: (elem(near, x) * 3 + elem(far, x) + bias) >>> 2) |> crop.()
        end

      {2, 2} when c.dw > 2 ->
        for y <- 0..(f.h - 1) do
          {inr, odd} = {div(y, 2), rem(y, 2)}
          {near, far} = if odd == 0, do: {row.(inr), row.(inr - 1)}, else: {row.(inr), row.(inr + 1)}
          h2v2(near, far, c.dw) |> crop.()
        end

      {hx, vx} ->
        # plain replication (int_upsample, and h2v1/h2v2 of tiny components)
        for y <- 0..(f.h - 1) do
          r = elem(plane, div(y, vx))
          for(x <- 0..(f.w - 1), do: elem(r, div(x, hx))) |> List.to_tuple()
        end
    end
  end

  defp h2v1(r, dw) do
    first = elem(r, 0)
    last = elem(r, dw - 1)

    mid =
      for x <- 1..(dw - 2)//1 do
        v = elem(r, x) * 3
        [(v + elem(r, x - 1) + 1) >>> 2, (v + elem(r, x + 1) + 2) >>> 2]
      end

    [first, (first * 3 + elem(r, 1) + 2) >>> 2] ++ List.flatten(mid) ++ [(last * 3 + elem(r, dw - 2) + 1) >>> 2, last]
  end

  defp h2v2(near, far, dw) do
    sum = fn x -> elem(near, x) * 3 + elem(far, x) end
    sums = for(x <- 0..(dw - 1), do: sum.(x)) |> List.to_tuple()
    s = fn x -> elem(sums, x) end

    mid =
      for x <- 1..(dw - 2)//1 do
        [(s.(x) * 3 + s.(x - 1) + 8) >>> 4, (s.(x) * 3 + s.(x + 1) + 7) >>> 4]
      end

    [(s.(0) * 4 + 8) >>> 4, (s.(0) * 3 + s.(1) + 7) >>> 4] ++ List.flatten(mid) ++
      [(s.(dw - 1) * 3 + s.(dw - 2) + 8) >>> 4, (s.(dw - 1) * 4 + 7) >>> 4]
  end

  # ----------------------------------------------------------------- colour --

  @scale 65_536
  @half 32_768
  @fix_1_402 91_881
  @fix_1_772 116_130
  @fix_0_714 46_802
  @fix_0_344 22_554

  defp ycc_rgb(ys, cbs, crs) do
    cr_r = for(i <- 0..255, do: (@fix_1_402 * (i - 128) + @half) >>> 16) |> List.to_tuple()
    cb_b = for(i <- 0..255, do: (@fix_1_772 * (i - 128) + @half) >>> 16) |> List.to_tuple()
    cr_g = for(i <- 0..255, do: -@fix_0_714 * (i - 128)) |> List.to_tuple()
    cb_g = for(i <- 0..255, do: -@fix_0_344 * (i - 128) + @half) |> List.to_tuple()
    _ = @scale
    clamp = fn v -> if v < 0, do: 0, else: (if v > 255, do: 255, else: v) end

    Enum.zip_with([ys, cbs, crs], fn [y, cb, cr] ->
      for x <- 0..(tuple_size(y) - 1), into: <<>> do
        {yy, b, r} = {elem(y, x), elem(cb, x), elem(cr, x)}
        <<clamp.(yy + elem(cr_r, r)), clamp.(yy + ((elem(cb_g, b) + elem(cr_g, r)) >>> 16)), clamp.(yy + elem(cb_b, b))>>
      end
    end)
    |> IO.iodata_to_binary()
  end

  defp interleave(as, bs, cs) do
    Enum.zip_with([as, bs, cs], fn [a, b, c] ->
      for x <- 0..(tuple_size(a) - 1), into: <<>>, do: <<elem(a, x), elem(b, x), elem(c, x)>>
    end)
    |> IO.iodata_to_binary()
  end
end
