defmodule Vapor.Media.GIF do
  @moduledoc """
  GIF (89a), both ways, without a dependency: the format every browser
  animates, so it is how the studio shows a video.

  **Encoding** (`encode/2`): one global palette for all frames, chosen by
  median cut over the pixels of every frame (ties broken by order, so the
  palette is a function of the pixels), each pixel mapped to its nearest
  entry (squared distance, lowest index on ties) — optionally with
  Floyd–Steinberg error diffusion in binary64 — then LZW with the codes
  growing to 12 bits and a clear code when the table fills; NETSCAPE2.0
  looping. A grey image with at most 256 levels gets those levels exactly.

  **Decoding** (`decode/1`): global and local palettes, interlacing,
  transparency, the four disposal methods; frames are composed on the
  logical screen as a browser does, and returned as RGB images with each
  frame's delay. Malformed input is refused by name, never a crash.
  """
  import Bitwise
  alias Vapor.Modal.Image
  alias Vapor.Rejection

  # ------------------------------------------------------------- encoding --

  @doc """
  Encode frames (`Vapor.Modal.Image`, all one size). Options: `fps` (10),
  `loop` (true), `dither` (false), `colors` (256).
  """
  def encode([%Image{w: w, h: h} | _] = frames, opts \\ []) do
    fps = Keyword.get(opts, :fps, 10)
    delay = max(1, round(100 / fps))
    rgb = Enum.map(frames, &to_rgb8/1)
    palette = palette(rgb, Keyword.get(opts, :colors, 256))
    bits = palette_bits(length(palette))
    table = palette ++ List.duplicate({0, 0, 0}, (1 <<< bits) - length(palette))
    lookup = :ets.new(:gif_lookup, [:set, :private])

    body =
      for px <- rgb, into: <<>> do
        idx = if Keyword.get(opts, :dither, false), do: dither(px, w, palette, lookup), else: Enum.map(px, &nearest(&1, palette, lookup))
        min_code = max(2, bits)
        gce = <<0x21, 0xF9, 4, 0b0000_0100, delay::16-little, 0, 0>>
        desc = <<0x2C, 0::16, 0::16, w::16-little, h::16-little, 0>>
        gce <> desc <> <<min_code>> <> sub_blocks(lzw(idx, min_code)) <> <<0>>
      end

    :ets.delete(lookup)
    loop = if Keyword.get(opts, :loop, true), do: <<0x21, 0xFF, 11, "NETSCAPE2.0", 3, 1, 0::16, 0>>, else: <<>>
    gct = for {r, g, b} <- table, into: <<>>, do: <<r, g, b>>
    "GIF89a" <> <<w::16-little, h::16-little, 0x80 ||| 0x70 ||| (bits - 1), 0, 0>> <> gct <> loop <> body <> <<0x3B>>
  end

  defp to_rgb8(%Image{c: c, px: px}) do
    vals = px |> Tuple.to_list() |> Enum.map(&round(min(1.0, max(0.0, &1)) * 255))
    if c == 3, do: vals |> Enum.chunk_every(3) |> Enum.map(&List.to_tuple/1), else: Enum.map(vals, &{&1, &1, &1})
  end

  defp palette_bits(n), do: Enum.find(1..8, &((1 <<< &1) >= n))

  # median cut over the distinct colours (weighted by count)
  defp palette(frames, k) do
    counts = frames |> List.flatten() |> Enum.frequencies()

    if map_size(counts) <= k do
      counts |> Map.keys() |> Enum.sort()
    else
      boxes = split([Enum.sort(Map.to_list(counts))], k)
      Enum.map(boxes, fn box ->
        n = box |> Enum.map(&elem(&1, 1)) |> Enum.sum()
        avg = fn i -> div(Enum.reduce(box, 0, fn {c, m}, s -> s + elem(c, i) * m end) * 2 + n, 2 * n) end
        {avg.(0), avg.(1), avg.(2)}
      end)
      |> Enum.uniq()
      |> Enum.sort()
    end
  end

  defp split(boxes, k) when length(boxes) >= k, do: boxes

  defp split(boxes, k) do
    {box, i} = boxes |> Enum.with_index() |> Enum.filter(fn {b, _} -> length(b) > 1 end) |> Enum.max_by(fn {b, _} -> elem(range(b), 0) end, fn -> {nil, nil} end)

    if box == nil do
      boxes
    else
      {_, ch} = range(box)
      sorted = Enum.sort_by(box, fn {c, _} -> {elem(c, ch), c} end)
      total = sorted |> Enum.map(&elem(&1, 1)) |> Enum.sum()
      {lo, hi} = cut(sorted, div(total, 2))
      split(List.replace_at(boxes, i, lo) ++ [hi], k)
    end
  end

  defp range(box) do
    for ch <- 0..2 do
      vs = Enum.map(box, fn {c, _} -> elem(c, ch) end)
      {Enum.max(vs) - Enum.min(vs), ch}
    end
    |> Enum.max_by(fn {r, ch} -> {r, -ch} end)
  end

  defp cut(sorted, half) do
    {lo, hi, _} =
      Enum.reduce(sorted, {[], [], 0}, fn {c, m}, {lo, hi, acc} ->
        if acc < half or lo == [], do: {[{c, m} | lo], hi, acc + m}, else: {lo, [{c, m} | hi], acc + m}
      end)

    if hi == [], do: {Enum.reverse(tl(lo)), [hd(lo)]}, else: {Enum.reverse(lo), Enum.reverse(hi)}
  end

  defp nearest({r, g, b} = c, palette, lookup) do
    case :ets.lookup(lookup, c) do
      [{_, i}] -> i
      [] ->
        {_, i} =
          palette
          |> Enum.with_index()
          |> Enum.reduce({nil, 0}, fn {{pr, pg, pb}, i}, {best, bi} ->
            d = (r - pr) * (r - pr) + (g - pg) * (g - pg) + (b - pb) * (b - pb)
            if best == nil or d < best, do: {d, i}, else: {best, bi}
          end)

        :ets.insert(lookup, {c, i})
        i
    end
  end

  # Floyd–Steinberg in binary64, row by row, left to right
  defp dither(px, w, palette, lookup) do
    pt = List.to_tuple(palette)
    rows = Enum.chunk_every(px, w)

    {out, _} =
      Enum.map_reduce(rows, List.duplicate({0.0, 0.0, 0.0}, w + 2), fn row, carry ->
        {idx, _err, next} =
          row
          |> Enum.with_index()
          |> Enum.reduce({[], {0.0, 0.0, 0.0}, List.duplicate({0.0, 0.0, 0.0}, w + 2) |> List.to_tuple()}, fn {{r, g, b}, x}, {acc, {er, eg, eb}, nxt} ->
            {cr, cg, cb} = Enum.at(carry, x + 1)
            want = {clamp(r + er + cr), clamp(g + eg + cg), clamp(b + eb + cb)}
            i = nearest(round3(want), palette, lookup)
            {pr, pg, pb} = elem(pt, i)
            {dr, dg, db} = {elem(want, 0) - pr, elem(want, 1) - pg, elem(want, 2) - pb}
            nxt = add_at(nxt, x, {dr * 3 / 16, dg * 3 / 16, db * 3 / 16})
            nxt = add_at(nxt, x + 1, {dr * 5 / 16, dg * 5 / 16, db * 5 / 16})
            nxt = add_at(nxt, x + 2, {dr / 16, dg / 16, db / 16})
            {[i | acc], {dr * 7 / 16, dg * 7 / 16, db * 7 / 16}, nxt}
          end)

        {Enum.reverse(idx), Tuple.to_list(next)}
      end)

    List.flatten(out)
  end

  defp add_at(t, i, {a, b, c}), do: (fn {x, y, z} -> put_elem(t, i, {x + a, y + b, z + c}) end).(elem(t, i))
  defp clamp(v), do: min(255.0, max(0.0, v))
  defp round3({a, b, c}), do: {round(a), round(b), round(c)}

  # LZW, variable code width (LSB first), the code widths and the clear on a
  # full table exactly as Go's compress/lzw (which every GIF reader accepts)
  defp lzw([first | rest], min_code) do
    clear = 1 <<< min_code
    eoi = clear + 1
    init = %{dict: %{}, hi: eoi, width: min_code + 1, overflow: 1 <<< (min_code + 1)}

    {st, saved, out} =
      Enum.reduce(rest, {init, first, [{clear, min_code + 1}]}, fn k, {st, saved, out} ->
        key = {saved, k}

        case st.dict do
          %{^key => code} -> {st, code, out}
          _ ->
            out = [{saved, st.width} | out]
            {st, out} = inc_hi(st, out, clear, min_code)
            st = if st.hi == eoi, do: st, else: %{st | dict: Map.put(st.dict, key, st.hi)}
            {st, k, out}
        end
      end)

    out = [{saved, st.width} | out]
    {st, out} = inc_hi(st, out, clear, min_code)
    pack(Enum.reverse([{eoi, st.width} | out]))
  end

  defp inc_hi(st, out, clear, min_code) do
    hi = st.hi + 1
    st = if hi == st.overflow, do: %{st | hi: hi, width: st.width + 1, overflow: st.overflow <<< 1}, else: %{st | hi: hi}

    if st.hi == 4095 do
      {%{dict: %{}, hi: clear + 1, width: min_code + 1, overflow: 1 <<< (min_code + 1)}, [{clear, st.width} | out]}
    else
      {st, out}
    end
  end

  defp pack(codes) do
    {bytes, buf, n} =
      Enum.reduce(codes, {[], 0, 0}, fn {c, w}, {out, buf, n} ->
        buf = buf ||| (c <<< n)
        n = n + w
        drain(out, buf, n)
      end)

    tail = if n > 0, do: [buf &&& 0xFF], else: []
    :binary.list_to_bin(Enum.reverse(tail ++ bytes))
  end

  defp drain(out, buf, n) when n >= 8, do: drain([buf &&& 0xFF | out], buf >>> 8, n - 8)
  defp drain(out, buf, n), do: {out, buf, n}

  defp sub_blocks(<<>>), do: <<>>
  defp sub_blocks(<<b::binary-size(255), rest::binary>>), do: <<255>> <> b <> sub_blocks(rest)
  defp sub_blocks(b), do: <<byte_size(b)>> <> b

  # ------------------------------------------------------------- decoding --

  @doc "Decode a GIF: `{:ok, %{width, height, frames: [Image], delays_cs: [int], loop: bool}}` or a rejection."
  def decode(<<"GIF8", v, "a", w::16-little, h::16-little, flags, bg, _aspect, rest::binary>>) when v in [?7, ?9] do
    {gct, rest} = if (flags &&& 0x80) != 0, do: table(rest, 1 <<< ((flags &&& 7) + 1)), else: {nil, rest}
    canvas = :binary.copy(<<0, 0, 0>>, w * h)
    bgc = if gct && bg < tuple_size(gct), do: elem(gct, bg), else: {0, 0, 0}

    case blocks(rest, %{w: w, h: h, gct: gct, canvas: canvas, gce: nil, frames: [], delays: [], loop: false, bg: bgc}) do
      {:ok, st} when st.frames != [] ->
        frames = st.frames |> Enum.reverse() |> Enum.map(&to_image(&1, w, h))
        {:ok, %{width: w, height: h, frames: frames, delays_cs: Enum.reverse(st.delays), loop: st.loop}}

      {:ok, _} -> bad("at least one image")
      err -> err
    end
  rescue
    _ -> bad("a well-formed GIF")
  end

  def decode(_), do: bad("a GIF87a/GIF89a header")

  defp bad(b), do: {:error, Rejection.new(:gif, b, "re-encode the animation")}

  defp to_image(canvas, w, h), do: Image.new(w, h, 3, for(<<b <- canvas>>, do: b / 255))

  defp table(bin, n) do
    <<t::binary-size(n * 3), rest::binary>> = bin
    {List.to_tuple(for(<<r, g, b <- t>>, do: {r, g, b})), rest}
  end

  defp blocks(<<0x3B, _::binary>>, st), do: {:ok, st}
  defp blocks(<<>>, st), do: {:ok, st}

  defp blocks(<<0x21, 0xF9, 4, packed, delay::16-little, ti, 0, rest::binary>>, st),
    do: blocks(rest, %{st | gce: %{disposal: (packed >>> 2) &&& 7, transparent: if((packed &&& 1) == 1, do: ti), delay: delay}})

  defp blocks(<<0x21, 0xFF, 11, "NETSCAPE2.0", rest::binary>>, st) do
    {_data, rest} = read_sub(rest)
    blocks(rest, %{st | loop: true})
  end

  defp blocks(<<0x21, _label, rest::binary>>, st) do
    {_data, rest} = read_sub(rest)
    blocks(rest, st)
  end

  defp blocks(<<0x2C, x::16-little, y::16-little, fw::16-little, fh::16-little, flags, rest::binary>>, st) do
    {lct, rest} = if (flags &&& 0x80) != 0, do: table(rest, 1 <<< ((flags &&& 7) + 1)), else: {nil, rest}
    <<min_code, rest::binary>> = rest
    {data, rest} = read_sub(rest)
    pal = lct || st.gct || {{0, 0, 0}}
    idx = lzw_decode(data, min_code, fw * fh)
    idx = if (flags &&& 0x40) != 0, do: deinterlace(idx, fw, fh), else: idx
    gce = st.gce || %{disposal: 0, transparent: nil, delay: 0}
    before = st.canvas
    canvas = paint(st.canvas, st.w, st.h, x, y, fw, fh, idx, pal, gce.transparent)
    frames = [canvas | st.frames]

    after_dispose =
      case gce.disposal do
        2 -> paint_rect(canvas, st.w, st.h, x, y, fw, fh, st.bg)
        3 -> before
        _ -> canvas
      end

    blocks(rest, %{st | canvas: after_dispose, frames: frames, delays: [gce.delay | st.delays], gce: nil})
  end

  defp read_sub(bin, acc \\ [])
  defp read_sub(<<0, rest::binary>>, acc), do: {IO.iodata_to_binary(Enum.reverse(acc)), rest}
  defp read_sub(<<n, d::binary-size(n), rest::binary>>, acc), do: read_sub(rest, [d | acc])

  defp paint(canvas, w, h, x0, y0, fw, _fh, idx, pal, transparent) do
    rows = Enum.chunk_every(idx, fw)
    ct = canvas

    Enum.reduce(Enum.with_index(rows), ct, fn {row, dy}, acc ->
      y = y0 + dy
      if y >= h do
        acc
      else
        Enum.reduce(Enum.with_index(row), acc, fn {i, dx}, a ->
          x = x0 + dx
          if x >= w or i == transparent or i >= tuple_size(pal) do
            a
          else
            {r, g, b} = elem(pal, i)
            p = (y * w + x) * 3
            <<pre::binary-size(p), _::binary-size(3), post::binary>> = a
            <<pre::binary, r, g, b, post::binary>>
          end
        end)
      end
    end)
  end

  defp paint_rect(canvas, w, h, x0, y0, fw, fh, {r, g, b}) do
    Enum.reduce(y0..min(h - 1, y0 + fh - 1)//1, canvas, fn y, acc ->
      xs = x0..min(w - 1, x0 + fw - 1)//1
      Enum.reduce(xs, acc, fn x, a ->
        p = (y * w + x) * 3
        <<pre::binary-size(p), _::binary-size(3), post::binary>> = a
        <<pre::binary, r, g, b, post::binary>>
      end)
    end)
  end

  defp deinterlace(idx, w, h) do
    rows = idx |> Enum.chunk_every(w) |> List.to_tuple()
    order = Enum.to_list(0..(h - 1)//8) ++ Enum.to_list(4..(h - 1)//8) ++ Enum.to_list(2..(h - 1)//4) ++ Enum.to_list(1..(h - 1)//2)
    placed = order |> Enum.with_index() |> Map.new(fn {y, k} -> {y, elem(rows, k)} end)
    for(y <- 0..(h - 1), do: placed[y]) |> List.flatten()
  end

  defp lzw_decode(data, min_code, n) do
    clear = 1 <<< min_code
    eoi = clear + 1
    st = %{dict: %{}, hi: eoi, width: min_code + 1, overflow: 1 <<< (min_code + 1), last: nil}
    out = lzw_codes(data, 0, 0, st, clear, eoi, min_code, [], 0, n)
    flat = out |> Enum.reverse() |> List.flatten()
    len = length(flat)
    if len >= n, do: Enum.take(flat, n), else: flat ++ List.duplicate(0, n - len)
  end

  defp lzw_codes(_data, _buf, _nb, _st, _c, _e, _m, out, count, n) when count >= n, do: out

  defp lzw_codes(data, buf, nb, st, clear, eoi, min_code, out, count, n) when nb < st.width do
    case data do
      <<b, rest::binary>> -> lzw_codes(rest, buf ||| (b <<< nb), nb + 8, st, clear, eoi, min_code, out, count, n)
      <<>> -> out
    end
  end

  defp lzw_codes(data, buf, nb, st, clear, eoi, min_code, out, count, n) do
    code = buf &&& ((1 <<< st.width) - 1)
    {buf, nb} = {buf >>> st.width, nb - st.width}

    cond do
      code == clear ->
        lzw_codes(data, buf, nb, %{st | dict: %{}, hi: eoi, width: min_code + 1, overflow: 1 <<< (min_code + 1), last: nil}, clear, eoi, min_code, out, count, n)

      code == eoi -> out

      true ->
        expand = fn c -> if c < clear, do: [c], else: Map.fetch!(st.dict, c) end

        entry =
          cond do
            code < clear or Map.has_key?(st.dict, code) -> expand.(code)
            code == st.hi and st.last != nil -> (fn l -> l ++ [hd(l)] end).(expand.(st.last))
            true -> throw(:bad_code)
          end

        dict = if st.last != nil and st.hi < 4096, do: Map.put(st.dict, st.hi, expand.(st.last) ++ [hd(entry)]), else: st.dict
        hi = st.hi + 1
        st = %{st | dict: dict, hi: hi, last: code}

        st =
          if hi >= st.overflow do
            if st.width == 12, do: %{st | last: nil, hi: hi - 1}, else: %{st | width: st.width + 1, overflow: st.overflow <<< 1}
          else
            st
          end

        lzw_codes(data, buf, nb, st, clear, eoi, min_code, [entry | out], count + length(entry), n)
    end
  end
end
