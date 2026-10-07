defmodule Vapor.Docs.Pictures do
  @moduledoc """
  Pictures for the document airlock.

    * **PNG** is decoded here completely: colour types 0, 2, 3, 4, 6, bit
      depths 1–16, the five scanline filters, Adam7 interlacing, `tRNS` is
      ignored (alpha is dropped, not composited); text chunks `tEXt`,
      `zTXt` and `iTXt` become searchable text. CRCs are checked.
    * **PPM/PGM** through `Vapor.Modal.Image.parse/1`.
    * **JPEG** is decoded by `Vapor.Docs.JPEG` (baseline and progressive,
      libjpeg's integer arithmetic — Pillow's pixels bit for bit), and its
      embedded text (comments, EXIF `ImageDescription`, `Artist`,
      `Copyright`, `UserComment`) is indexed; what the decoder refuses
      (arithmetic coding, CMYK, 12-bit) keeps size and text, with a warning.
    * **GIF, WebP**: dimensions and embedded text (GIF comment extensions)
      — no pixel decoding, said in a warning.

  A decoded picture is returned as a `Vapor.Modal.Image` (RGB or grey, in
  [0, 1]) for the library's visual index.
  """
  import Bitwise
  alias Vapor.Modal.Image
  alias Vapor.Rejection

  @max_pixels 50_000_000
  # beyond this, the BEAM decoder would take minutes: metadata and text only
  @max_decode 4_000_000

  @doc "`{:ok, %{format, width, height, mode, texts, image?, warnings}}`."
  def read(:png, bytes), do: png(bytes)
  def read(:pnm, bytes), do: with({:ok, img} <- Image.parse(bytes), do: {:ok, %{format: "PPM/PGM", width: img.w, height: img.h, mode: if(img.c == 3, do: "RGB", else: "L"), texts: [], image: img}})
  def read(:jpeg, bytes), do: jpeg(bytes)
  def read(:gif, <<"GIF8", _v::binary-size(2), w::16-little, h::16-little, _::binary>> = b), do: {:ok, %{format: "GIF", width: w, height: h, mode: "P", texts: gif_comments(b), warnings: ["GIF pixels are not decoded"]}}
  def read(:webp, b), do: {:ok, Map.merge(%{format: "WebP", mode: "?", texts: [], warnings: ["WebP pixels are not decoded"]}, webp_size(b))}
  def read(_, _), do: bad("a known picture format")

  @doc "The searchable text of a picture: its name, size and embedded texts."
  def describe(name, pic) do
    texts = Enum.map_join(pic.texts, "\n", fn {k, v} -> "#{k}: #{v}" end)
    "imagem #{Path.basename(name)} (#{pic.format}, #{pic[:width]}×#{pic[:height]}, #{pic.mode})" <> if(texts == "", do: "", else: "\n" <> texts)
  end

  @doc false
  def utf8_prefix(b) do
    case String.chunk(b, :valid) do
      [] -> ""
      cs -> if String.valid?(List.last(cs)) and byte_size(List.last(cs)) == byte_size(b), do: b, else: trim_tail(b)
    end
  end

  # a sample cut in the middle of a character is still text
  defp trim_tail(b) do
    Enum.find_value(0..3, b, fn k ->
      p = binary_part(b, 0, max(byte_size(b) - k, 0))
      if String.valid?(p), do: p
    end)
  end

  # ------------------------------------------------------------------ PNG --

  defp png(<<0x89, "PNG\r\n", 0x1A, "\n", rest::binary>>) do
    with {:ok, chunks} <- png_chunks(rest, []),
         {_, ihdr} <- List.keyfind(chunks, "IHDR", 0) || bad("an IHDR chunk"),
         <<w::32, h::32, depth, ctype, 0, 0, interlace>> <- ihdr,
         true <- (ctype in [0, 2, 3, 4, 6] and depth in valid_depths(ctype)) || bad("a valid colour type / bit depth (#{ctype}/#{depth})"),
         true <- (w > 0 and h > 0 and w * h <= @max_pixels) || bad("at most #{@max_pixels} pixels"),
         :decode <- (if w * h > @max_decode, do: {:large, w, h, depth, ctype}, else: :decode),
         idat = for({"IDAT", d} <- chunks, into: <<>>, do: d),
         {:ok, raw} <- inflate(idat, expected_size(w, h, depth, ctype, interlace)),
         {:ok, samples} <- unfilter_all(raw, w, h, depth, ctype, interlace),
         {:ok, img} <- to_image(samples, w, h, depth, ctype, chunks) do
      {:ok, %{format: "PNG", width: w, height: h, mode: mode(ctype) <> if(depth != 8, do: "/#{depth}", else: ""),
              interlaced: interlace == 1, texts: png_texts(chunks), image: img}}
    else
      {:error, _} = e -> e
      {:large, w, h, depth, ctype} ->
        {:ok, chunks} = png_chunks(rest, [])
        {:ok, %{format: "PNG", width: w, height: h, mode: mode(ctype) <> if(depth != 8, do: "/#{depth}", else: ""),
                texts: png_texts(chunks), warnings: ["#{w}×#{h} pixels: larger than #{@max_decode}, not decoded (text and size only)"]}}
      _ -> bad("a well-formed PNG")
    end
  end

  defp valid_depths(0), do: [1, 2, 4, 8, 16]
  defp valid_depths(3), do: [1, 2, 4, 8]
  defp valid_depths(_), do: [8, 16]

  defp mode(0), do: "L"
  defp mode(2), do: "RGB"
  defp mode(3), do: "P"
  defp mode(4), do: "LA"
  defp mode(6), do: "RGBA"

  defp channels(0), do: 1
  defp channels(2), do: 3
  defp channels(3), do: 1
  defp channels(4), do: 2
  defp channels(6), do: 4

  defp png_chunks(<<len::32, type::binary-size(4), data::binary-size(len), crc::32, rest::binary>>, acc) do
    if :erlang.crc32(type <> data) != crc do
      bad("chunk #{type} with a correct CRC")
    else
      acc = [{type, data} | acc]
      if type == "IEND", do: {:ok, Enum.reverse(acc)}, else: png_chunks(rest, acc)
    end
  end

  defp png_chunks(_, _), do: bad("complete chunks up to IEND")

  defp row_bytes(w, depth, ctype), do: div(w * channels(ctype) * depth + 7, 8)

  defp expected_size(w, h, depth, ctype, 0), do: h * (1 + row_bytes(w, depth, ctype))
  defp expected_size(w, h, depth, ctype, 1), do: adam7(w, h) |> Enum.map(fn {pw, ph, _, _, _, _} -> if pw == 0 or ph == 0, do: 0, else: ph * (1 + row_bytes(pw, depth, ctype)) end) |> Enum.sum()

  # the seven passes: {width, height, x0, y0, dx, dy}
  defp adam7(w, h) do
    for {x0, y0, dx, dy} <- [{0, 0, 8, 8}, {4, 0, 8, 8}, {0, 4, 4, 8}, {2, 0, 4, 4}, {0, 2, 2, 4}, {1, 0, 2, 2}, {0, 1, 1, 2}] do
      {div(max(w - x0, 0) + dx - 1, dx), div(max(h - y0, 0) + dy - 1, dy), x0, y0, dx, dy}
    end
  end

  # inflate with the decoded size as a hard cap (a PNG bomb is refused)
  defp inflate(data, cap) do
    z = :zlib.open()
    :ok = :zlib.inflateInit(z)

    try do
      loop = fn loop, {st, out}, acc, n ->
        n = n + IO.iodata_length(out)
        cond do
          n > cap -> bad("image data no larger than its header says")
          st == :finished -> {:ok, IO.iodata_to_binary(Enum.reverse([out | acc]))}
          true -> loop.(loop, :zlib.safeInflate(z, []), [out | acc], n)
        end
      end

      loop.(loop, :zlib.safeInflate(z, data), [], 0)
    rescue
      _ -> bad("valid zlib image data")
    after
      :zlib.close(z)
    end
  end

  # every pass unfiltered into a w×h grid of sample tuples
  defp unfilter_all(raw, w, h, depth, ctype, 0) do
    with {:ok, rows, _} <- unfilter(raw, w, h, depth, ctype), do: {:ok, rows |> Enum.map(&samples(&1, w, depth, ctype))}
  end

  defp unfilter_all(raw, w, h, depth, ctype, 1) do
    {grid, _} =
      Enum.reduce(adam7(w, h), {%{}, raw}, fn {pw, ph, x0, y0, dx, dy}, {grid, rest} ->
        if pw == 0 or ph == 0 do
          {grid, rest}
        else
          {:ok, rows, rest} = unfilter(rest, pw, ph, depth, ctype)

          grid =
            rows
            |> Enum.with_index()
            |> Enum.reduce(grid, fn {row, j}, g ->
              row |> samples(pw, depth, ctype) |> Enum.with_index() |> Enum.reduce(g, fn {px, i}, g -> Map.put(g, {x0 + i * dx, y0 + j * dy}, px) end)
            end)

          {grid, rest}
        end
      end)

    {:ok, for(y <- 0..(h - 1), do: for(x <- 0..(w - 1), do: Map.fetch!(grid, {x, y})))}
  rescue
    _ -> bad("complete interlaced passes")
  end

  defp unfilter(raw, w, h, depth, ctype) do
    rb = row_bytes(w, depth, ctype)
    bpp = max(1, div(channels(ctype) * depth, 8))

    if byte_size(raw) < h * (rb + 1) do
      bad("#{h} scanlines of #{rb + 1} bytes")
    else
      {rows, _prev} =
        Enum.map_reduce(0..(h - 1), :binary.copy(<<0>>, rb), fn j, prev ->
          <<f, line::binary-size(rb)>> = binary_part(raw, j * (rb + 1), rb + 1)
          row = recon(f, line, prev, bpp)
          {row, row}
        end)

      {:ok, rows, binary_part(raw, h * (rb + 1), byte_size(raw) - h * (rb + 1))}
    end
  end

  @doc """
  Undo PNG row predictors (the PDF `/Predictor ≥ 10` of Flate streams):
  `h` rows of `rb` bytes, each prefixed by its filter byte, `bpp` bytes per
  pixel. Returns `{:ok, bytes}` (rows concatenated) or an error.
  """
  def unpredict(raw, rb, bpp, h) do
    if byte_size(raw) < h * (rb + 1) do
      {:error, "#{h} predicted rows of #{rb + 1} bytes"}
    else
      {rows, _} =
        Enum.map_reduce(0..(h - 1), :binary.copy(<<0>>, rb), fn j, prev ->
          <<f, line::binary-size(rb)>> = binary_part(raw, j * (rb + 1), rb + 1)
          row = recon(f, line, prev, max(bpp, 1))
          {row, row}
        end)

      {:ok, IO.iodata_to_binary(rows)}
    end
  rescue
    e in ArgumentError -> {:error, Exception.message(e)}
  end

  defp recon(0, line, _prev, _bpp), do: line
  defp recon(2, line, prev, _bpp), do: for({a, b} <- Enum.zip(:binary.bin_to_list(line), :binary.bin_to_list(prev)), into: <<>>, do: <<a + b &&& 255>>)

  defp recon(f, line, prev, bpp) when f in [1, 3, 4] do
    up = List.to_tuple(:binary.bin_to_list(prev))
    xs = :binary.bin_to_list(line)

    # `win` holds the last bpp reconstructed bytes, newest first
    {out, _} =
      xs
      |> Enum.with_index()
      |> Enum.reduce({[], []}, fn {x, i}, {acc, win} ->
        a = if i >= bpp, do: List.last(win), else: 0
        b = elem(up, i)
        c = if i >= bpp, do: elem(up, i - bpp), else: 0

        v =
          case f do
            1 -> x + a
            3 -> x + div(a + b, 2)
            4 -> x + paeth(a, b, c)
          end
          |> band(255)

        {[v | acc], Enum.take([v | win], bpp)}
      end)

    out |> Enum.reverse() |> :binary.list_to_bin()
  end

  defp recon(f, _, _, _), do: raise(ArgumentError, "PNG filter #{f}")

  defp paeth(a, b, c) do
    p = a + b - c
    {pa, pb, pc} = {abs(p - a), abs(p - b), abs(p - c)}
    cond do
      pa <= pb and pa <= pc -> a
      pb <= pc -> b
      true -> c
    end
  end

  # a row as pixel tuples of integer samples
  defp samples(row, w, depth, ctype) do
    n = channels(ctype)

    vals =
      case depth do
        8 -> :binary.bin_to_list(row)
        16 -> for <<v::16 <- row>>, do: v
        d -> for(<<v::size(d) <- row>>, do: v)
      end

    vals |> Enum.take(w * n) |> Enum.chunk_every(n) |> Enum.map(&List.to_tuple/1)
  end

  defp to_image(rows, w, h, depth, ctype, chunks) do
    max = (1 <<< depth) - 1

    case ctype do
      3 ->
        {_, plte} = List.keyfind(chunks, "PLTE", 0) || {nil, nil}

        if plte == nil do
          bad("a PLTE chunk for a palette image")
        else
          pal = for(<<r, g, b <- plte>>, do: {r, g, b}) |> List.to_tuple()
          vals = for row <- rows, {i} <- row, {r, g, b} = elem(pal, min(i, tuple_size(pal) - 1)), v <- [r, g, b], do: v / 255
          {:ok, Image.new(w, h, 3, vals)}
        end

      c when c in [0, 4] -> {:ok, Image.new(w, h, 1, for(row <- rows, px <- row, do: elem(px, 0) / max))}
      _ -> {:ok, Image.new(w, h, 3, for(row <- rows, px <- row, i <- 0..2, do: elem(px, i) / max))}
    end
  end

  defp png_texts(chunks) do
    Enum.flat_map(chunks, fn
      {"tEXt", d} -> case :binary.split(d, <<0>>) do
        [k, v] -> [{k, latin1(v)}]
        _ -> []
      end

      {"zTXt", d} -> case :binary.split(d, <<0>>) do
        [k, <<0, z::binary>>] -> [{k, latin1(safe_unzip(z))}]
        _ -> []
      end

      {"iTXt", d} ->
        with [k, <<cflag, _m, rest::binary>>] <- :binary.split(d, <<0>>),
             [_lang, rest] <- :binary.split(rest, <<0>>),
             [_tk, text] <- :binary.split(rest, <<0>>) do
          [{k, if(cflag == 1, do: safe_unzip(text), else: text)}]
        else
          _ -> []
        end

      _ -> []
    end)
  end

  defp safe_unzip(z) do
    case inflate(z, 1_000_000) do
      {:ok, t} -> t
      _ -> ""
    end
  end

  defp latin1(b), do: :unicode.characters_to_binary(b, :latin1)

  # ----------------------------------------------------------------- JPEG --

  # pixels by Vapor.Docs.JPEG (libjpeg's integer arithmetic, bit for bit);
  # what it refuses (arithmetic coding, CMYK, 12-bit) keeps size and text
  defp jpeg(<<0xFF, 0xD8, rest::binary>> = bytes) do
    {dims, texts} = jpeg_segments(rest, nil, [])
    {w, h, comps} = dims || {nil, nil, nil}
    base = %{format: "JPEG", width: w, height: h, mode: case(comps, do: (1 -> "L"; 3 -> "YCbCr"; 4 -> "CMYK"; _ -> "?")), texts: Enum.reverse(texts)}

    case Vapor.Docs.JPEG.decode(bytes) do
      {:ok, d} ->
        img = %Image{w: d.width, h: d.height, c: d.channels, px: d.pixels |> :binary.bin_to_list() |> Enum.map(&(&1 / 255)) |> List.to_tuple()}
        {:ok, Map.merge(base, %{mode: d.mode, image: img, progressive: d.progressive})}

      {:error, why} ->
        {:ok, Map.put(base, :warnings, ["JPEG pixels not decoded: #{why}; indexed by its text"])}
    end
  end

  defp jpeg_segments(<<0xFF, m, len::16, rest::binary>>, dims, texts) when m not in [0xD8, 0xD9, 0xDA, 0x01] and m not in 0xD0..0xD7 and len >= 2 and byte_size(rest) >= len - 2 do
    <<seg::binary-size(len - 2), more::binary>> = rest

    {dims, texts} =
      cond do
        m in [0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF] ->
          <<_p, h::16, w::16, n, _::binary>> = seg
          {{w, h, n}, texts}

        m == 0xFE -> {dims, [{"Comment", Vapor.Docs.utf8(seg)} | texts]}
        m == 0xE1 and match?(<<"Exif", 0, 0, _::binary>>, seg) -> {dims, exif(binary_part(seg, 6, byte_size(seg) - 6)) ++ texts}
        true -> {dims, texts}
      end

    jpeg_segments(more, dims, texts)
  end

  defp jpeg_segments(_, dims, texts), do: {dims, texts}

  @exif_tags %{0x010E => "ImageDescription", 0x013B => "Artist", 0x8298 => "Copyright", 0x9286 => "UserComment"}

  # IFD0 (and the Exif sub-IFD) ASCII/undefined text tags of a TIFF block
  defp exif(<<order::binary-size(2), _::binary>> = tiff) when order in ["II", "MM"] do
    u16 = fn b, o -> if order == "II", do: :binary.decode_unsigned(binary_part(b, o, 2), :little), else: :binary.decode_unsigned(binary_part(b, o, 2)) end
    u32 = fn b, o -> if order == "II", do: :binary.decode_unsigned(binary_part(b, o, 4), :little), else: :binary.decode_unsigned(binary_part(b, o, 4)) end

    ifd = fn ifd, off, depth ->
      if off + 2 > byte_size(tiff) or depth > 2 do
        []
      else
        n = u16.(tiff, off)

        Enum.flat_map(0..(n - 1)//1, fn i ->
          e = off + 2 + i * 12
          if e + 12 > byte_size(tiff), do: [], else: (
            tag = u16.(tiff, e)
            count = u32.(tiff, e + 4)
            val_off = if count > 4, do: u32.(tiff, e + 8), else: e + 8
            cond do
              tag == 0x8769 -> ifd.(ifd, u32.(tiff, e + 8), depth + 1)
              Map.has_key?(@exif_tags, tag) and val_off + count <= byte_size(tiff) ->
                v = tiff |> binary_part(val_off, count) |> String.trim_trailing(<<0>>)
                v = if tag == 0x9286 and byte_size(v) > 8, do: binary_part(v, 8, byte_size(v) - 8), else: v
                if String.trim(v) == "", do: [], else: [{@exif_tags[tag], Vapor.Docs.utf8(v)}]
              true -> []
            end)
        end)
      end
    end

    ifd.(ifd, u32.(tiff, 4), 0)
  rescue
    _ -> []
  end

  defp exif(_), do: []

  defp gif_comments(b) do
    for [_, body] <- Regex.scan(~r/\x21\xFE((?:[\x01-\xFF][\s\S]*?)*?)\x00/, b), do: {"Comment", Vapor.Docs.utf8(gif_blocks(body))}
  end

  defp gif_blocks(<<n, data::binary-size(n), rest::binary>>) when n > 0, do: data <> gif_blocks(rest)
  defp gif_blocks(_), do: ""

  defp webp_size(<<"RIFF", _::32, "WEBP", "VP8X", _::32, _flags::32, w::24-little, h::24-little, _::binary>>), do: %{width: w + 1, height: h + 1}
  defp webp_size(<<"RIFF", _::32, "WEBP", "VP8 ", _::32, _::binary-size(6), w::16-little, h::16-little, _::binary>>), do: %{width: band(w, 0x3FFF), height: band(h, 0x3FFF)}
  defp webp_size(_), do: %{width: nil, height: nil}

  defp bad(b), do: {:error, Rejection.new(:picture, b, "convert the picture to PNG")}
end
