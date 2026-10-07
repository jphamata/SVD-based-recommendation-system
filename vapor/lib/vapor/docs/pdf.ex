defmodule Vapor.Docs.PDF do
  @moduledoc """
  The text of a PDF, page by page, with no dependency.

  Reading strategy, chosen for robustness over speed:

    * **objects by scanning**, not by trusting the cross-reference table:
      every `n g obj … endobj` is parsed (the last definition of a number
      wins, as incremental updates intend), so damaged or rewritten xrefs
      do not lose text; **object streams** (`/Type /ObjStm`, PDF 1.5+) are
      unpacked and their objects added; the catalog comes from the trailer
      or the xref-stream dictionary;
    * **filters**: FlateDecode, LZWDecode (`EarlyChange`), RunLengthDecode,
      ASCIIHexDecode, ASCII85Decode, chained, each with its `DecodeParms`;
      image codecs: DCTDecode (`Vapor.Docs.JPEG`), CCITTFaxDecode
      (`Vapor.Docs.CCITT`) and JBIG2Decode with its `JBIG2Globals`
      (`Vapor.Docs.JBIG2`); JPX is named in a warning;
    * **pages** in document order through the page tree, with inherited
      resources; contents streams concatenated;
    * **text** from the content stream operators `Tj`, `TJ`, `'`, `"`, with
      lines from `Td`/`TD`/`T*`/`Tm` moves and spaces from large `TJ`
      kerning; inline images (`BI … ID … EI`) skipped;
    * **decoding** through each font's `ToUnicode` CMap (`bfchar`, `bfrange`,
      code lengths from the code-space ranges); without one, simple fonts
      are read as WinAnsi (with `/Differences` glyph names mapped), and a
      composite font without a `ToUnicode` is a warning — its text cannot be
      recovered honestly.

  Encrypted PDFs are refused: their strings are ciphertext.
  """
  import Bitwise
  alias Vapor.Rejection

  @doc "`{:ok, [page_text], warnings}` or a rejection."
  def pages(bytes) do
    with :ok <- not_encrypted(bytes),
         objs = objects(bytes),
         {:ok, root} <- root(bytes, objs),
         %{} = catalog <- deref(objs, root),
         %{} = tree <- deref(objs, catalog["Pages"]),
         pages when is_list(pages) <- page_list(objs, tree, nil, MapSet.new()) |> nonempty() do
      {texts, warns} =
        pages
        |> Enum.map(fn {page, res} -> page_text(objs, page, res) end)
        |> Enum.unzip()

      {:ok, texts, warns |> List.flatten() |> Enum.uniq()}
    else
      {:error, _} = e -> e
      _ -> bad("a page tree")
    end
  rescue
    e -> bad("a readable PDF (#{Exception.message(e) |> String.slice(0, 80)})")
  end

  @doc """
  The raster images of the given pages (1-based), for pages that have no
  text layer: `{:ok, %{page => [image]}, warnings}` where each image is a
  `Vapor.Modal.Image`. Decoded: DCTDecode (JPEG, `Vapor.Docs.JPEG`),
  CCITTFaxDecode (Group 3/4 fax, `Vapor.Docs.CCITT` — what office scanners
  write), and FlateDecode/LZWDecode/RunLengthDecode (8-bit gray or RGB,
  and 1-bit, with PNG and TIFF predictors), with the `Decode` inversion of
  1-bit images and stencil masks (`ImageMask`: a 0 sample is ink),
  and JBIG2Decode (`Vapor.Docs.JBIG2`, arithmetic coding). JPX images and
  the JBIG2 parts not decoded (Huffman tables, halftone regions) are named
  in a warning.
  """
  def images(bytes, wanted) do
    with :ok <- not_encrypted(bytes),
         objs = objects(bytes),
         {:ok, root} <- root(bytes, objs),
         %{} = catalog <- deref(objs, root),
         %{} = tree <- deref(objs, catalog["Pages"]),
         pages when is_list(pages) <- page_list(objs, tree, nil, MapSet.new()) |> nonempty() do
      wanted = MapSet.new(wanted)

      {found, warns} =
        pages
        |> Enum.with_index(1)
        |> Enum.filter(fn {_, i} -> MapSet.member?(wanted, i) end)
        |> Enum.map_reduce([], fn {{_page, res}, i}, warns ->
          xo = deref(objs, (res || %{})["XObject"]) || %{}

          {imgs, ws} =
            xo
            |> Enum.map(fn {_k, v} -> deref(objs, v) end)
            |> Enum.filter(&match?(%{"Subtype" => {:name, "Image"}}, &1))
            |> Enum.map(&raster(objs, &1))
            |> Enum.split_with(&(elem(&1, 0) == :ok))

          damaged = for {:ok, _, why} <- imgs, do: "page #{i}: #{why}"
          {{i, Enum.map(imgs, &elem(&1, 1))}, warns ++ damaged ++ Enum.map(ws, fn {:skip, why} -> "page #{i}: #{why}" end)}
        end)

      {:ok, Map.new(found), Enum.uniq(warns)}
    else
      {:error, _} = e -> e
      _ -> bad("a page tree")
    end
  rescue
    e -> bad("a readable PDF (#{Exception.message(e) |> String.slice(0, 80)})")
  end

  defp raster(objs, %{stream: _} = d) do
    filters = d["Filter"] |> deref_all(objs) |> List.wrap() |> Enum.map(fn {:name, f} -> f; _ -> "?" end)
    d = Map.put(d, "DecodeParms", parms_list(objs, d, length(filters)))
    {w, h} = {deref(objs, d["Width"]), deref(objs, d["Height"])}
    bpc = deref(objs, d["BitsPerComponent"]) || 8
    cs = case deref(objs, d["ColorSpace"]) do
      {:name, n} -> n
      [{:name, "ICCBased"}, ref] -> (case deref(objs, ref) do %{"N" => 1} -> "DeviceGray"; %{"N" => 3} -> "DeviceRGB"; _ -> "ICC" end)
      [{:name, "Indexed"} | _] -> "Indexed"
      nil -> if d["ImageMask"] == true, do: "Mask", else: "DeviceGray"
      _ -> "?"
    end
    cs = if d["ImageMask"] == true, do: "Mask", else: cs
    # a 1-bit sample is ink when 0 (DeviceGray, a stencil mask), unless Decode is [1 0]
    inv = deref(objs, d["Decode"]) == [1, 0]
    sane = is_integer(w) and is_integer(h) and w > 0 and h > 0 and w * h <= 64_000_000

    cond do
      List.last(filters) == "DCTDecode" ->
        with {:ok, pre} <- decode(drop_last(d, filters)),
             {:ok, j} <- Vapor.Docs.JPEG.decode(pre) do
          {:ok, Vapor.Modal.Image.new(j.width, j.height, j.channels, j.pixels |> :binary.bin_to_list() |> Enum.map(&(&1 / 255)))}
        else
          {:error, why} -> {:skip, "a JPEG image not decoded (#{inspect(why)})"}
        end

      List.last(filters) == "CCITTFaxDecode" and sane ->
        p = List.last(d["DecodeParms"]) || %{}
        opts = [k: p["K"] || 0, columns: p["Columns"] || 1728, rows: p["Rows"] || h, black_is_1: p["BlackIs1"] == true,
                end_of_line: p["EndOfLine"] == true, byte_align: p["EncodedByteAlign"] == true, end_of_block: p["EndOfBlock"] != false]

        with {:ok, pre} <- decode(drop_last(d, filters)),
             {:ok, %{columns: cols, rows: rows, data: data, warnings: ws}} when rows > 0 <- Vapor.Docs.CCITT.decode(pre, opts) do
          if ws == [], do: {:ok, one_bit(data, cols, rows, inv)}, else: {:ok, one_bit(data, cols, rows, inv), "a CCITT image: " <> Enum.join(ws, "; ")}
        else
          {:ok, _} -> {:skip, "a CCITT image with no rows"}
          {:error, why} -> {:skip, "a CCITT image not decoded (#{inspect(why)})"}
        end

      List.last(filters) == "JBIG2Decode" and sane ->
        p = List.last(d["DecodeParms"]) || %{}

        globals =
          case deref(objs, p["JBIG2Globals"]) do
            %{stream: _} = g -> (case decode(g) do {:ok, b} -> b; _ -> nil end)
            _ -> nil
          end

        with {:ok, pre} <- decode(drop_last(d, filters)),
             {:ok, data, jw, jh, ws} <- Vapor.Docs.JBIG2.pdf(pre, globals) do
          img = one_bit(data, jw, jh, inv)
          if ws == [], do: {:ok, img}, else: {:ok, img, "a JBIG2 image: " <> Enum.join(ws, "; ")}
        else
          {:error, why} -> {:skip, "a JBIG2 image not decoded (#{inspect(why)})"}
        end

      Enum.any?(filters, &(&1 in ["JBIG2Decode", "JPXDecode"])) ->
        {:skip, "an image in #{Enum.join(filters, "+")} (not decoded here)"}

      not sane ->
        {:skip, "an image without a sane size"}

      true ->
        with {:ok, data} <- decode(d),
             {:ok, data} <- predictor(objs, d, data, w, h, cs, bpc) do
          case {cs, bpc} do
            {"DeviceGray", 8} -> {:ok, Vapor.Modal.Image.new(w, h, 1, for(<<v <- binary_part(data, 0, min(byte_size(data), w * h))>>, do: v / 255))}
            {"DeviceRGB", 8} -> {:ok, Vapor.Modal.Image.new(w, h, 3, for(<<v <- binary_part(data, 0, min(byte_size(data), w * h * 3))>>, do: v / 255))}
            {c, 1} when c in ["DeviceGray", "Mask"] -> {:ok, one_bit(data, w, h, inv)}
            {"Mask", _} -> {:ok, one_bit(data, w, h, inv)}
            other -> {:skip, "an image with colour space/depth #{inspect(other)} (not decoded here)"}
          end
        else
          {:error, why} -> {:skip, "an image stream not decoded (#{inspect(why)})"}
        end
    end
  end

  defp raster(_objs, _), do: {:skip, "an image without data"}

  # 1-bit rows padded to whole bytes → gray (0 = ink, 1 = paper; `inv` flips)
  defp one_bit(data, w, h, inv) do
    stride = div(w + 7, 8)
    data = if byte_size(data) < stride * h, do: data <> :binary.copy(<<255>>, stride * h - byte_size(data)), else: data

    vals =
      for y <- 0..(h - 1), <<row::bitstring-size(w), _::bitstring>> = binary_part(data, y * stride, stride), <<b::1 <- row>>,
          do: if(inv, do: 1 - b, else: b) * 1.0

    Vapor.Modal.Image.new(w, h, 1, vals)
  end

  defp drop_last(d, filters) do
    d |> Map.put("Filter", Enum.drop(filters, -1) |> Enum.map(&{:name, &1})) |> Map.put("DecodeParms", Enum.drop(d["DecodeParms"], -1))
  end

  defp deref_all(v, objs), do: deref(objs, v)

  # DecodeParms aligned with the filters (a dict for one filter, an array for a chain), references resolved
  defp parms_list(objs, d, n) do
    ps = case deref(objs, d["DecodeParms"]) do
      l when is_list(l) -> l
      nil -> []
      p -> [p]
    end

    ps = Enum.map(ps, fn p -> case deref(objs, p) do %{} = m -> Map.new(m, fn {k, v} -> {k, deref(objs, v)} end); _ -> nil end end)
    ps ++ List.duplicate(nil, max(n - length(ps), 0))
  end

  # Flate's PNG predictors (Predictor ≥ 10): rows carry a filter byte
  defp predictor(objs, d, data, w, h, cs, bpc) do
    parms = case d["DecodeParms"] do
      l when is_list(l) -> Enum.find(l, &is_map/1)
      p -> p
    end

    case parms && deref(objs, parms["Predictor"]) do
      2 when bpc == 8 ->
        colors = deref(objs, parms["Colors"]) || (if cs == "DeviceRGB", do: 3, else: 1)
        {:ok, tiff_unpredict(data, (deref(objs, parms["Columns"]) || w) * colors, colors)}

      p when is_integer(p) and p >= 10 ->
        colors = deref(objs, parms["Colors"]) || (if cs == "DeviceRGB", do: 3, else: 1)
        cols = deref(objs, parms["Columns"]) || w
        Vapor.Docs.Pictures.unpredict(data, div(cols * colors * bpc + 7, 8), div(colors * bpc, 8), h)

      p when p in [nil, 1] -> {:ok, data}
      p -> {:error, "predictor #{inspect(p)}"}
    end
  end

  # TIFF predictor 2, 8 bits: each sample is a difference from the one `colors` to its left
  defp tiff_unpredict(data, stride, colors) do
    for <<row::binary-size(stride) <- data>>, into: <<>> do
      {out, _} =
        for <<v <- row>>, reduce: {[], :queue.new()} do
          {acc, q} ->
            {left, q} = if :queue.len(q) < colors, do: {0, q}, else: (with {{:value, l}, q2} <- :queue.out(q), do: {l, q2})
            x = rem(v + left, 256)
            {[x | acc], :queue.in(x, q)}
        end

      out |> Enum.reverse() |> :erlang.list_to_binary()
    end
  end

  defp nonempty([]), do: :none
  defp nonempty(l), do: l

  defp not_encrypted(bytes) do
    if Regex.match?(~r/\/Encrypt\s+(\d+\s+\d+\s+R|<<)/, bytes),
      do: {:error, Rejection.new(:pdf, "an unencrypted PDF (this one is encrypted: its strings are ciphertext)", "decrypt it first (qpdf --decrypt)")},
      else: :ok
  end

  # ------------------------------------------------------------- objects --

  defp objects(bytes) do
    direct =
      Regex.scan(~r/(?<![0-9])(\d+)\s+(\d+)\s+obj\b/, bytes, return: :index)
      |> Enum.reduce(%{}, fn [{at, len}, {n0, nl}, _], acc ->
        n = String.to_integer(binary_part(bytes, n0, nl))

        case safe_value(bytes, at + len) do
          {v, pos} -> Map.put(acc, n, {v, stream_at(bytes, pos)})
          nil -> acc
        end
      end)

    # streams whose /Length is indirect are resolved now that every object is known
    direct = Map.new(direct, fn {n, {v, s}} -> {n, finish_stream(bytes, direct, v, s)} end)

    # unpack object streams
    Enum.reduce(direct, direct, fn
      {_, %{"Type" => {:name, "ObjStm"}} = d}, acc -> Map.merge(objstm(acc, d), acc)
      _, acc -> acc
    end)
  end

  defp safe_value(bin, pos) do
    value(bin, skip_ws(bin, pos))
  rescue
    _ -> nil
  end

  defp stream_at(bytes, pos) do
    pos = skip_ws(bytes, pos)

    case bytes do
      <<_::binary-size(pos), "stream\r\n", _::binary>> -> pos + 8
      <<_::binary-size(pos), "stream\n", _::binary>> -> pos + 7
      <<_::binary-size(pos), "stream\r", _::binary>> -> pos + 7
      _ -> nil
    end
  end

  defp finish_stream(_bytes, _objs, v, nil), do: v

  defp finish_stream(bytes, objs, %{} = d, start) do
    len =
      case d["Length"] do
        n when is_integer(n) -> n
        {:ref, r} -> case objs[r] do
          n when is_integer(n) -> n
          _ -> nil
        end
        _ -> nil
      end

    len =
      if len && start + len <= byte_size(bytes) do
        len
      else
        case :binary.match(bytes, "endstream", scope: {start, byte_size(bytes) - start}) do
          {e, _} -> e - start
          :nomatch -> byte_size(bytes) - start
        end
      end

    Map.put(d, :stream, binary_part(bytes, start, len))
  end

  defp finish_stream(_b, _o, v, _), do: v

  defp objstm(_objs, d) do
    with {:ok, data} <- decode(d),
         n when is_integer(n) <- d["N"],
         first when is_integer(first) <- d["First"] do
      header = binary_part(data, 0, min(first, byte_size(data)))
      nums = Regex.scan(~r/\d+/, header) |> List.flatten() |> Enum.map(&String.to_integer/1) |> Enum.chunk_every(2)

      for [num, off] <- Enum.take(nums, n), into: %{} do
        {v, _} = value(data, skip_ws(data, first + off))
        {num, v}
      end
    else
      _ -> %{}
    end
  rescue
    _ -> %{}
  end

  defp root(bytes, objs) do
    trailer =
      case Regex.scan(~r/trailer\s*<</, bytes, return: :index) do
        [] -> nil
        ms -> [{at, len}] = List.last(ms); elem(value(bytes, at + len - 2), 0)
      end

    cond do
      is_map(trailer) and trailer["Root"] -> {:ok, trailer["Root"]}
      xr = Enum.find_value(objs, fn {_, %{"Type" => {:name, "XRef"}, "Root" => r}} -> r; _ -> nil end) -> {:ok, xr}
      cat = Enum.find_value(objs, fn {n, %{"Type" => {:name, "Catalog"}}} -> {:ref, n}; _ -> nil end) -> {:ok, cat}
      true -> bad("a document catalog")
    end
  end

  defp deref(objs, {:ref, n}), do: Map.get(objs, n)
  defp deref(_objs, v), do: v

  # pages in order, each with its (inherited) resources
  defp page_list(objs, %{} = node, res, seen) do
    res = deref(objs, node["Resources"]) || res

    case node["Type"] do
      {:name, "Page"} -> [{node, res}]
      _ ->
        kids = deref(objs, node["Kids"]) || []
        Enum.flat_map(kids, fn
          {:ref, n} = r -> if MapSet.member?(seen, n), do: [], else: page_list(objs, deref(objs, r), res, MapSet.put(seen, n))
          %{} = k -> page_list(objs, k, res, seen)
          _ -> []
        end)
    end
  end

  defp page_list(_objs, _, _res, _seen), do: []

  # -------------------------------------------------------------- filters --

  @doc false
  def decode(%{stream: data} = d) do
    filters = d["Filter"] |> List.wrap() |> Enum.map(fn {:name, f} -> f; _ -> "?" end)
    parms = case d["DecodeParms"] do
      l when is_list(l) -> l
      nil -> []
      p -> [p]
    end

    filters
    |> Enum.with_index()
    |> Enum.reduce_while({:ok, data}, fn {f, i}, {:ok, acc} ->
      p = case Enum.at(parms, i) do %{} = m -> m; _ -> %{} end

      case filter(f, acc, p) do
        {:ok, out} -> {:cont, {:ok, out}}
        err -> {:halt, err}
      end
    end)
  end

  def decode(_), do: {:error, :no_stream}

  defp filter("LZWDecode", data, p), do: {:ok, lzw(data, if(p["EarlyChange"] == 0, do: 0, else: 1))}
  defp filter("RunLengthDecode", data, _p), do: {:ok, run_length(data, [], 0)}
  defp filter(f, data, _p), do: filter(f, data)

  defp filter("FlateDecode", data) do
    z = :zlib.open()
    :zlib.inflateInit(z)

    try do
      {:ok, inflate_all(z, :zlib.safeInflate(z, data), [], 0)}
    rescue
      _ -> {:error, "corrupt Flate data"}
    after
      :zlib.close(z)
    end
  end

  defp filter("ASCIIHexDecode", data) do
    hex = data |> String.split(">") |> hd() |> String.replace(~r/[^0-9A-Fa-f]/, "")
    hex = if rem(byte_size(hex), 2) == 1, do: hex <> "0", else: hex
    {:ok, Base.decode16!(hex, case: :mixed)}
  end

  defp filter("ASCII85Decode", data), do: {:ok, a85(data |> String.split("~>") |> hd() |> String.replace(~r/\s/, "") |> String.trim_leading("<~"))}
  defp filter(f, _), do: {:error, "filter #{f}"}

  @doc """
  LZW as PDF and TIFF use it: 9- to 12-bit codes, MSB first, 256 = clear,
  257 = end; with `early` = 1 (the default) the code width grows one code
  early. A code that cannot be decoded ends the stream (what came before
  is kept); output is capped at 256 MB.
  """
  def lzw(data, early \\ 1), do: lzw_loop(data, 0, 9, 258, nil, %{}, early, [], 0)

  defp lzw_loop(data, pos, width, next, prev, tab, early, out, n) do
    case data do
      <<_::bitstring-size(pos), code::size(width), _::bitstring>> when n <= 256_000_000 ->
        pos = pos + width

        cond do
          code == 256 -> lzw_loop(data, pos, 9, 258, nil, %{}, early, out, n)
          code == 257 -> out |> Enum.reverse() |> IO.iodata_to_binary()
          prev == nil and code < 256 -> lzw_loop(data, pos, width, next, <<code>>, tab, early, [<<code>> | out], n + 1)
          prev == nil -> out |> Enum.reverse() |> IO.iodata_to_binary()
          true ->
            entry =
              cond do
                code < 256 -> <<code>>
                code < next -> Map.get(tab, code)
                code == next -> prev <> binary_part(prev, 0, 1)
                true -> nil
              end

            if entry == nil do
              out |> Enum.reverse() |> IO.iodata_to_binary()
            else
              {tab, next} = if next < 4096, do: {Map.put(tab, next, prev <> binary_part(entry, 0, 1)), next + 1}, else: {tab, next}
              width = if next + early >= 1 <<< width and width < 12, do: width + 1, else: width
              lzw_loop(data, pos, width, next, entry, tab, early, [entry | out], n + byte_size(entry))
            end
        end

      _ ->
        out |> Enum.reverse() |> IO.iodata_to_binary()
    end
  end

  # PackBits: L < 128 copies L + 1 bytes, L > 128 repeats the next 257 − L times, 128 ends
  defp run_length(<<128, _::binary>>, out, _n), do: out |> Enum.reverse() |> IO.iodata_to_binary()
  defp run_length(<<l, rest::binary>>, out, n) when l < 128 and n <= 256_000_000 do
    take = min(l + 1, byte_size(rest))
    <<lit::binary-size(take), rest::binary>> = rest
    run_length(rest, [lit | out], n + take)
  end
  defp run_length(<<l, b, rest::binary>>, out, n) when l > 128 and n <= 256_000_000, do: run_length(rest, [:binary.copy(<<b>>, 257 - l) | out], n + 257 - l)
  defp run_length(_, out, _n), do: out |> Enum.reverse() |> IO.iodata_to_binary()

  defp inflate_all(z, {st, out}, acc, n) do
    acc = [out | acc]
    n = n + IO.iodata_length(out)
    if st == :finished or n > 256_000_000, do: IO.iodata_to_binary(Enum.reverse(acc)), else: inflate_all(z, :zlib.safeInflate(z, []), acc, n)
  end

  defp a85(s) do
    s
    |> String.replace("z", "!!!!!")
    |> :binary.bin_to_list()
    |> Enum.chunk_every(5)
    |> Enum.map(fn g ->
      pad = 5 - length(g)
      v = (g ++ List.duplicate(?u, pad)) |> Enum.reduce(0, fn c, a -> a * 85 + (c - 33) end)
      binary_part(<<v::32>>, 0, 4 - pad)
    end)
    |> IO.iodata_to_binary()
  end

  # ---------------------------------------------------------------- pages --

  defp page_text(objs, page, res) do
    data =
      page["Contents"]
      |> then(&deref(objs, &1))
      |> List.wrap()
      |> Enum.map(&deref(objs, &1))
      |> Enum.map(fn s -> case decode(s) do
        {:ok, d} -> d
        _ -> ""
      end end)
      |> Enum.join("\n")

    fonts = (deref(objs, (res || %{})["Font"]) || %{}) |> Map.new(fn {k, v} -> {k, font(objs, deref(objs, v))} end)
    {text, warns} = run(data, fonts)
    {Vapor.Docs.Markup.tidy(text), warns}
  end

  # a font as a decoder: %{codes: [lengths], map: %{code_binary => text}, fallback: :winansi | :none}
  defp font(objs, %{} = f) do
    cmap = with %{} = s <- deref(objs, f["ToUnicode"]), {:ok, d} <- decode(s), do: cmap(d), else: (_ -> nil)

    cond do
      cmap -> Map.put(cmap, :fallback, :none)
      f["Subtype"] == {:name, "Type0"} -> %{lens: [2], map: %{}, fallback: :none, warn: "a composite font without ToUnicode: its text cannot be recovered"}
      true -> %{lens: [1], map: differences(objs, f["Encoding"]), fallback: :winansi}
    end
  end

  defp font(_objs, _), do: %{lens: [1], map: %{}, fallback: :winansi}

  @doc false
  def cmap(d) do
    lens =
      for [_, body] <- Regex.scan(~r/begincodespacerange(.*?)endcodespacerange/s, d),
          [_, lo, _hi] <- Regex.scan(~r/<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>/, body), uniq: true, do: div(byte_size(lo), 2)

    chars =
      for [_, body] <- Regex.scan(~r/beginbfchar(.*?)endbfchar/s, d),
          [_, src, dst] <- Regex.scan(~r/<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]*)>/, body),
          into: %{}, do: {hex(src), utf16(hex(dst))}

    ranges =
      for [_, body] <- Regex.scan(~r/beginbfrange(.*?)endbfrange/s, d),
          [_, lo, hi, dst] <- Regex.scan(~r/<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(<[0-9A-Fa-f]*>|\[[^\]]*\])/, body),
          reduce: %{} do
        acc ->
          {l, h, w} = {:binary.decode_unsigned(hex(lo)), :binary.decode_unsigned(hex(hi)), byte_size(hex(lo))}

          if h - l > 65_535 do
            acc
          else
            case dst do
              "[" <> list ->
                outs = Regex.scan(~r/<([0-9A-Fa-f]*)>/, list) |> Enum.map(fn [_, x] -> utf16(hex(x)) end)
                Enum.zip(l..h, outs) |> Enum.reduce(acc, fn {c, o}, a -> Map.put(a, <<c::size(w * 8)>>, o) end)

              _ ->
                base = hex(String.slice(dst, 1..-2//1))
                bv = :binary.decode_unsigned(base)
                bw = byte_size(base)
                Enum.reduce(l..h, acc, fn c, a -> Map.put(a, <<c::size(w * 8)>>, utf16(<<bv + (c - l)::size(bw * 8)>>)) end)
            end
          end
      end

    lens = if lens == [], do: [2], else: Enum.sort(lens)
    %{lens: lens, map: Map.merge(ranges, chars)}
  end

  defp hex(h), do: Base.decode16!(if(rem(byte_size(h), 2) == 1, do: h <> "0", else: h), case: :mixed)

  defp utf16(b) do
    case :unicode.characters_to_binary(b, {:utf16, :big}) do
      s when is_binary(s) -> s
      _ -> ""
    end
  end

  defp differences(objs, enc) do
    case deref(objs, enc) do
      %{"Differences" => diffs} ->
        {m, _} = Enum.reduce(diffs, {%{}, 0}, fn
          n, {m, _} when is_integer(n) -> {m, n}
          {:name, g}, {m, c} -> {Map.put(m, <<c>>, glyph(g)), c + 1}
          _, acc -> acc
        end)
        Map.reject(m, fn {_, v} -> v == nil end)

      _ ->
        %{}
    end
  end

  @glyphs %{"space" => " ", "quotesingle" => "'", "quoteright" => "’", "quoteleft" => "‘", "quotedblleft" => "“", "quotedblright" => "”",
            "endash" => "–", "emdash" => "—", "bullet" => "•", "ellipsis" => "…", "fi" => "fi", "fl" => "fl", "hyphen" => "-",
            "period" => ".", "comma" => ",", "colon" => ":", "semicolon" => ";", "exclam" => "!", "question" => "?",
            "parenleft" => "(", "parenright" => ")", "Euro" => "€", "degree" => "°", "ordfeminine" => "ª", "ordmasculine" => "º"}
  @accents %{"acute" => 0x301, "grave" => 0x300, "circumflex" => 0x302, "tilde" => 0x303, "dieresis" => 0x308, "cedilla" => 0x327, "ring" => 0x30A}

  defp glyph(g) do
    cond do
      Map.has_key?(@glyphs, g) -> @glyphs[g]
      String.match?(g, ~r/^[A-Za-z]$/) -> g
      String.match?(g, ~r/^uni[0-9A-F]{4}$/) -> <<String.to_integer(String.slice(g, 3, 4), 16)::utf8>>
      m = Regex.run(~r/^([A-Za-z])(acute|grave|circumflex|tilde|dieresis|cedilla|ring)$/, g) ->
        [_, base, acc] = m
        :unicode.characters_to_nfc_binary(base <> <<@accents[acc]::utf8>>)
      true -> nil
    end
  end

  # ------------------------------------------------------- content streams --

  defp run(data, fonts) do
    toks = content_tokens(data, 0, [])

    {out, _font, _y, warns} =
      Enum.reduce(toks, {[], nil, nil, MapSet.new(), []}, fn
        {:op, "Tf"}, {out, _f, y, w, [_size, {:name, n} | _]} -> {out, fonts[n], y, w, []}
        {:op, op}, {out, f, y, w, ops} when op in ["Tj", "'", "\""] ->
          s = case ops do [{:str, s} | _] -> s; _ -> "" end
          pre = if op in ["'", "\""], do: "\n", else: ""
          {[pre <> show(s, f) | out], f, y, warn(w, f), []}
        {:op, "TJ"}, {out, f, y, w, [arr | _]} when is_list(arr) ->
          txt = Enum.map_join(arr, fn {:str, s} -> show(s, f); n when is_number(n) and n < -250 -> " "; _ -> "" end)
          {[txt | out], f, y, warn(w, f), []}
        {:op, op}, {out, f, _y, w, [ty, _tx | _]} when op in ["Td", "TD"] and is_number(ty) ->
          {if(ty != 0, do: ["\n" | out], else: [" " | out]), f, nil, w, []}
        {:op, "Tm"}, {out, f, y, w, [ny, _x | _]} when is_number(ny) ->
          {if(y != nil and abs(ny - y) > 0.5, do: ["\n" | out], else: out), f, ny, w, []}
        {:op, "T*"}, {out, f, y, w, _} -> {["\n" | out], f, y, w, []}
        {:op, "ET"}, {out, f, y, w, _} -> {[" " | out], f, y, w, []}
        {:op, _}, {out, f, y, w, _} -> {out, f, y, w, []}
        operand, {out, f, y, w, ops} -> {out, f, y, w, [operand | ops]}
      end)
      |> then(fn {out, f, y, w, _} -> {out, f, y, w} end)

    {out |> Enum.reverse() |> IO.iodata_to_binary(), MapSet.to_list(warns)}
  end

  defp warn(w, %{warn: msg}), do: MapSet.put(w, msg)
  defp warn(w, _), do: w

  defp show(s, nil), do: winansi(s)
  defp show(s, %{lens: lens, map: map, fallback: fb}), do: decode_codes(s, lens, map, fb, [])

  defp decode_codes(<<>>, _lens, _map, _fb, acc), do: acc |> Enum.reverse() |> IO.iodata_to_binary()

  defp decode_codes(s, lens, map, fb, acc) do
    hit =
      Enum.find_value(lens, fn n ->
        if byte_size(s) >= n do
          case Map.fetch(map, binary_part(s, 0, n)) do
            {:ok, t} -> {n, t}
            :error -> nil
          end
        end
      end)

    case hit do
      {n, t} -> decode_codes(binary_part(s, n, byte_size(s) - n), lens, map, fb, [t | acc])
      nil ->
        n = min(hd(lens), byte_size(s))
        <<code::binary-size(n), rest::binary>> = s
        decode_codes(rest, lens, map, fb, [if(fb == :winansi, do: winansi(code), else: "") | acc])
    end
  end

  # cp1252 (WinAnsi) bytes → UTF-8
  @cp1252 %{0x80 => "€", 0x82 => "‚", 0x83 => "ƒ", 0x84 => "„", 0x85 => "…", 0x86 => "†", 0x87 => "‡", 0x88 => "ˆ", 0x89 => "‰",
            0x8A => "Š", 0x8B => "‹", 0x8C => "Œ", 0x8E => "Ž", 0x91 => "‘", 0x92 => "’", 0x93 => "“", 0x94 => "”", 0x95 => "•",
            0x96 => "–", 0x97 => "—", 0x98 => "˜", 0x99 => "™", 0x9A => "š", 0x9B => "›", 0x9C => "œ", 0x9E => "ž", 0x9F => "Ÿ"}

  defp winansi(s), do: for(<<b <- s>>, into: "", do: Map.get(@cp1252, b) || <<b::utf8>>)

  # operands and operators of a content stream (strings, numbers, names, arrays)
  defp content_tokens(bin, pos, acc) when pos >= byte_size(bin), do: Enum.reverse(acc)

  defp content_tokens(bin, pos, acc) do
    pos = skip_ws(bin, pos)

    if pos >= byte_size(bin) do
      Enum.reverse(acc)
    else
      case binary_part(bin, pos, 1) do
        c when c in ["(", "<", "[", "/"] or (c >= "0" and c <= "9") or c in ["-", "+", "."] ->
          if c == "<" and binary_part(bin, pos, min(2, byte_size(bin) - pos)) == "<<" do
            {v, p} = value(bin, pos)
            content_tokens(bin, p, [v | acc])
          else
            {v, p} = value(bin, pos)
            content_tokens(bin, p, [v | acc])
          end

        _ ->
          {word, p} = keyword(bin, pos)

          if word == "BI" do
            # inline image: skip to "EI" after the "ID" data
            case :binary.match(bin, "ID", scope: {p, byte_size(bin) - p}) do
              {i, _} ->
                case :binary.match(bin, "EI", scope: {i + 2, byte_size(bin) - i - 2}) do
                  {e, _} -> content_tokens(bin, e + 2, acc)
                  :nomatch -> Enum.reverse(acc)
                end
              :nomatch -> Enum.reverse(acc)
            end
          else
            content_tokens(bin, max(p, pos + 1), [{:op, word} | acc])
          end
      end
    end
  end

  # ------------------------------------------------------------- the parser --

  @delims ~c"()<>[]{}/%"

  defp skip_ws(bin, pos) do
    case bin do
      <<_::binary-size(pos), c, _::binary>> when c in [0, 9, 10, 12, 13, 32] -> skip_ws(bin, pos + 1)
      <<_::binary-size(pos), ?%, _::binary>> ->
        case :binary.match(bin, ["\n", "\r"], scope: {pos, byte_size(bin) - pos}) do
          {i, _} -> skip_ws(bin, i + 1)
          :nomatch -> byte_size(bin)
        end
      _ -> pos
    end
  end

  defp keyword(bin, pos) do
    len = Enum.reduce_while(pos..(byte_size(bin) - 1)//1, 0, fn i, n ->
      c = :binary.at(bin, i)
      if c in [0, 9, 10, 12, 13, 32] or c in @delims, do: {:halt, n}, else: {:cont, n + 1}
    end)

    {binary_part(bin, pos, len), pos + len}
  end

  @doc false
  def value(bin, pos) do
    pos = skip_ws(bin, pos)

    case bin do
      <<_::binary-size(pos), "<<", _::binary>> -> dict(bin, pos + 2, %{})
      <<_::binary-size(pos), "<", _::binary>> -> hexstr(bin, pos + 1)
      <<_::binary-size(pos), "(", _::binary>> -> litstr(bin, pos + 1, 1, [])
      <<_::binary-size(pos), "[", _::binary>> -> array(bin, pos + 1, [])
      <<_::binary-size(pos), "/", _::binary>> -> {n, p} = keyword(bin, pos + 1); {{:name, name(n)}, p}
      _ ->
        {w, p} = keyword(bin, pos)

        cond do
          w == "true" -> {true, p}
          w == "false" -> {false, p}
          w == "null" -> {nil, p}
          String.match?(w, ~r/^[+-]?\d+$/) ->
            n = String.to_integer(w)
            # `n g R`: a reference
            case Regex.run(~r/^\s+(\d+)\s+R(?![A-Za-z])/, binary_part(bin, p, min(24, byte_size(bin) - p))) do
              [whole, _g] when n >= 0 -> {{:ref, n}, p + byte_size(whole)}
              _ -> {n, p}
            end
          String.match?(w, ~r/^[+-]?(\d*\.\d+|\d+\.\d*)$/) -> {parse_float(w), p}
          w == "" -> raise ArgumentError, "unexpected byte at #{pos}"
          true -> {{:kw, w}, p}
        end
    end
  end

  defp parse_float(w) do
    w = if String.starts_with?(w, ".") or String.starts_with?(w, "-.") or String.starts_with?(w, "+."), do: String.replace(w, ".", "0.", global: false), else: w
    w = if String.ends_with?(w, "."), do: w <> "0", else: w
    String.to_float(w)
  end

  defp name(n), do: Regex.replace(~r/#([0-9A-Fa-f]{2})/, n, fn _, h -> <<String.to_integer(h, 16)>> end)

  defp dict(bin, pos, acc) do
    pos = skip_ws(bin, pos)

    case bin do
      <<_::binary-size(pos), ">>", _::binary>> -> {acc, pos + 2}
      _ ->
        {{:name, k}, p} = value(bin, pos)
        {v, p} = value(bin, p)
        dict(bin, p, Map.put(acc, k, v))
    end
  end

  defp array(bin, pos, acc) do
    pos = skip_ws(bin, pos)

    case bin do
      <<_::binary-size(pos), "]", _::binary>> -> {Enum.reverse(acc), pos + 1}
      _ -> {v, p} = value(bin, pos); array(bin, p, [v | acc])
    end
  end

  defp hexstr(bin, pos) do
    {i, _} = :binary.match(bin, ">", scope: {pos, byte_size(bin) - pos})
    {{:str, hex(binary_part(bin, pos, i - pos) |> String.replace(~r/[^0-9A-Fa-f]/, ""))}, i + 1}
  end

  defp litstr(bin, pos, depth, acc) do
    case bin do
      <<_::binary-size(pos), "\\", c, _::binary>> ->
        case c do
          ?n -> litstr(bin, pos + 2, depth, ["\n" | acc])
          ?r -> litstr(bin, pos + 2, depth, ["\r" | acc])
          ?t -> litstr(bin, pos + 2, depth, ["\t" | acc])
          ?b -> litstr(bin, pos + 2, depth, ["\b" | acc])
          ?f -> litstr(bin, pos + 2, depth, ["\f" | acc])
          c when c in ?0..?7 ->
            [oct] = Regex.run(~r/^[0-7]{1,3}/, binary_part(bin, pos + 1, min(3, byte_size(bin) - pos - 1)))
            litstr(bin, pos + 1 + byte_size(oct), depth, [<<String.to_integer(oct, 8) &&& 255>> | acc])
          ?\r -> litstr(bin, if(:binary.at(bin, pos + 2) == ?\n, do: pos + 3, else: pos + 2), depth, acc)
          ?\n -> litstr(bin, pos + 2, depth, acc)
          c -> litstr(bin, pos + 2, depth, [<<c>> | acc])
        end

      <<_::binary-size(pos), "(", _::binary>> -> litstr(bin, pos + 1, depth + 1, ["(" | acc])
      <<_::binary-size(pos), ")", _::binary>> when depth == 1 -> {{:str, acc |> Enum.reverse() |> IO.iodata_to_binary()}, pos + 1}
      <<_::binary-size(pos), ")", _::binary>> -> litstr(bin, pos + 1, depth - 1, [")" | acc])
      <<_::binary-size(pos), c, _::binary>> -> litstr(bin, pos + 1, depth, [<<c>> | acc])
    end
  end

  defp bad(b), do: {:error, Rejection.new(:pdf, b, "check the PDF (qpdf --check)")}
end
