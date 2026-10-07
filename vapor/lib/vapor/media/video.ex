defmodule Vapor.Media.Video do
  @moduledoc """
  Video containers the studio reads and writes without a dependency.

    * **Y4M** (YUV4MPEG2, `read_y4m/1`, `y4m/2`): raw frames, the format
      every tool pipes (`ffmpeg -i x.mp4 -f yuv4mpegpipe -`). Colour spaces
      `444`, `420jpeg`/`420mpeg2`/`420paldv` (read alike; chroma upsampled by
      replication) and `mono`. RGB ↔ Y′CbCr is BT.601 limited range, the
      default of ffmpeg's converter, in binary64 with one rounding.
    * **MJPEG in AVI** (`read_avi/1`): the format of webcams and many
      cameras — the RIFF chunks of the `movi` list, each frame a JPEG decoded
      by `Vapor.Docs.JPEG` (libjpeg's output, bit for bit). Frames without
      Huffman tables (the "AVI1" convention) get the standard tables of
      JPEG Annex K.
    * **GIF** (`Vapor.Media.GIF`): animated, both ways.

  Compressed codecs (H.264/HEVC/AV1/VP9) are not decoded: each is a large
  standard whose decoder would have to be conferred bit for bit against its
  reference; `ffmpeg -f yuv4mpegpipe` converts any of them to Y4M.
  """
  alias Vapor.Modal.Image
  alias Vapor.Rejection
  alias Vapor.Studio.Video

  @doc "Read any supported container from its bytes (sniffed): `{:ok, %Video{}}` or a rejection."
  def read("YUV4MPEG2" <> _ = b), do: read_y4m(b)
  def read(<<"RIFF", _::32, "AVI ", _::binary>> = b), do: read_avi(b)

  def read(<<"GIF8", _::binary>> = b) do
    with {:ok, d} <- Vapor.Media.GIF.decode(b) do
      cs = d.delays_cs |> Enum.reject(&(&1 == 0))
      fps = if cs == [], do: 10.0, else: 100 / (Enum.sum(cs) / length(cs))
      {:ok, %Video{fps: Float.round(fps * 1.0, 3), frames: d.frames}}
    end
  end

  def read(_), do: {:error, Rejection.new(:video, "Y4M, MJPEG-AVI or GIF", "convert with ffmpeg -f yuv4mpegpipe")}

  # ------------------------------------------------------------------ Y4M --

  @doc "Read a Y4M stream."
  def read_y4m(bin) do
    with [header, body] <- :binary.split(bin, "\n"),
         "YUV4MPEG2" <> params <- header do
      p = params |> String.split(" ", trim: true) |> Map.new(fn <<k, v::binary>> -> {<<k>>, v} end)
      w = String.to_integer(p["W"])
      h = String.to_integer(p["H"])
      fps = case String.split(p["F"] || "25:1", ":") do
        [n, d] -> String.to_integer(n) / max(String.to_integer(d), 1)
        _ -> 25.0
      end
      cs = p["C"] || "420jpeg"
      {cw, ch, planes} = chroma(cs, w, h)
      size = w * h + planes * cw * ch
      frames = frames(body, size, [])
      {:ok, %Video{fps: fps * 1.0, frames: Enum.map(frames, &yuv_to_image(&1, w, h, cw, ch, planes))}}
    else
      _ -> {:error, Rejection.new(:y4m, "a YUV4MPEG2 header", "write it with ffmpeg -f yuv4mpegpipe")}
    end
  rescue
    _ -> {:error, Rejection.new(:y4m, "a well-formed Y4M stream", "re-export it")}
  end

  defp chroma("444" <> _, w, h), do: {w, h, 2}
  defp chroma("420" <> _, w, h), do: {div(w + 1, 2), div(h + 1, 2), 2}
  defp chroma("mono" <> _, w, h), do: {w, h, 0}
  defp chroma(cs, _, _), do: throw({:unsupported, cs})

  defp frames(<<"FRAME", rest::binary>>, size, acc) do
    [_params, data] = :binary.split(rest, "\n")
    <<f::binary-size(size), more::binary>> = data
    frames(more, size, [f | acc])
  end

  defp frames(_, _, acc), do: Enum.reverse(acc)

  # BT.601, limited range: Y in 16..235, Cb/Cr in 16..240
  defp yuv_to_image(f, w, h, cw, ch, planes) do
    <<y::binary-size(w * h), rest::binary>> = f

    if planes == 0 do
      Image.new(w, h, 1, for(<<v <- y>>, do: clamp01((v - 16) / 219)))
    else
      <<u::binary-size(cw * ch), v::binary-size(cw * ch)>> = rest
      {ut, vt} = {u, v}
      sx = div(w + cw - 1, cw)
      sy = div(h + ch - 1, ch)

      vals =
        for row <- 0..(h - 1), col <- 0..(w - 1) do
          yy = (:binary.at(y, row * w + col) - 16) / 219
          ci = div(row, sy) * cw + div(col, sx)
          cb = (:binary.at(ut, ci) - 128) / 224
          cr = (:binary.at(vt, ci) - 128) / 224
          [clamp01(yy + 1.402 * cr), clamp01(yy - 0.344136286201022 * cb - 0.714136286201022 * cr), clamp01(yy + 1.772 * cb)]
        end

      Image.new(w, h, 3, List.flatten(vals))
    end
  end

  defp clamp01(x), do: min(1.0, max(0.0, x))

  @doc "Write a Y4M stream (`420jpeg`, or `mono` for one-channel frames). Option `fps`."
  def y4m(%Video{frames: [%Image{w: w, h: h, c: c} | _] = frames, fps: fps}, opts \\ []) do
    {num, den} = rational(Keyword.get(opts, :fps, fps))
    cs = if c == 1, do: "mono", else: "420jpeg"
    header = "YUV4MPEG2 W#{w} H#{h} F#{num}:#{den} Ip A1:1 C#{cs}\n"
    IO.iodata_to_binary([header | Enum.map(frames, &["FRAME\n", planes(&1)])])
  end

  defp rational(fps) do
    if fps == trunc(fps), do: {trunc(fps), 1}, else: {round(fps * 1001), 1001}
  end

  defp planes(%Image{w: w, h: h, c: 1, px: px}), do: for(p <- 0..(w * h - 1), into: <<>>, do: <<round(16 + 219 * clamp01(elem(px, p)))>>)

  defp planes(%Image{w: w, h: h, px: px}) do
    rgb = fn x, y -> i = (y * w + x) * 3; {clamp01(elem(px, i)), clamp01(elem(px, i + 1)), clamp01(elem(px, i + 2))} end
    luma = fn {r, g, b} -> 0.299 * r + 0.587 * g + 0.114 * b end
    y = for yy <- 0..(h - 1), xx <- 0..(w - 1), into: <<>>, do: <<round(16 + 219 * luma.(rgb.(xx, yy)))>>
    {cw, ch} = {div(w + 1, 2), div(h + 1, 2)}

    # 2×2 means (the 420jpeg siting), edge pixels repeated
    avg = fn cx, cy, f ->
      pts = for dy <- 0..1, dx <- 0..1, do: rgb.(min(2 * cx + dx, w - 1), min(2 * cy + dy, h - 1))
      Enum.reduce(pts, 0.0, fn p, s -> s + f.(p, luma.(p)) end) / 4
    end

    cb = for cy <- 0..(ch - 1), cx <- 0..(cw - 1), into: <<>>, do: <<round(128 + 224 * avg.(cx, cy, fn {_, _, b}, l -> (b - l) / 1.772 end))>>
    cr = for cy <- 0..(ch - 1), cx <- 0..(cw - 1), into: <<>>, do: <<round(128 + 224 * avg.(cx, cy, fn {r, _, _}, l -> (r - l) / 1.402 end))>>
    y <> cb <> cr
  end

  # ------------------------------------------------------------------ AVI --

  @doc "Read an AVI whose video stream is Motion-JPEG."
  def read_avi(<<"RIFF", _::32-little, "AVI ", rest::binary>>) do
    {usec, jpegs} = riff(rest, nil, [])
    jpegs = Enum.reverse(jpegs)
    fps = if usec && usec > 0, do: 1.0e6 / usec, else: 25.0

    frames =
      Enum.map(jpegs, fn j ->
        case Vapor.Docs.JPEG.decode(with_huffman(j)) do
          {:ok, d} -> {:ok, %Image{w: d.width, h: d.height, c: d.channels, px: d.pixels |> :binary.bin_to_list() |> Enum.map(&(&1 / 255)) |> List.to_tuple()}}
          {:error, why} -> {:error, why}
        end
      end)

    case Enum.find(frames, &match?({:error, _}, &1)) do
      nil when frames != [] -> {:ok, %Video{fps: Float.round(fps, 3), frames: Enum.map(frames, &elem(&1, 1))}}
      nil -> {:error, Rejection.new(:avi, "MJPEG frames in the movi list", "re-encode with ffmpeg -c:v mjpeg")}
      {:error, why} -> {:error, Rejection.new(:avi, "decodable JPEG frames", inspect(why))}
    end
  rescue
    _ -> {:error, Rejection.new(:avi, "a well-formed AVI", "re-encode with ffmpeg -c:v mjpeg")}
  end

  def read_avi(_), do: {:error, Rejection.new(:avi, "a RIFF AVI file", "re-encode with ffmpeg -c:v mjpeg")}

  defp riff(<<"LIST", size::32-little, type::binary-size(4), rest::binary>>, usec, acc) do
    body_size = size - 4
    <<body::binary-size(body_size), more::binary>> = rest
    more = skip_pad(more, size)
    {usec, acc} = if type in ["hdrl", "movi", "strl", "rec "], do: riff(body, usec, acc), else: {usec, acc}
    riff(more, usec, acc)
  end

  defp riff(<<"avih", size::32-little, rest::binary>>, _usec, acc) do
    <<body::binary-size(size), more::binary>> = rest
    <<us::32-little, _::binary>> = body
    riff(skip_pad(more, size), us, acc)
  end

  defp riff(<<id::binary-size(4), size::32-little, rest::binary>>, usec, acc) do
    <<body::binary-size(size), more::binary>> = rest
    acc = if binary_part(id, 2, 2) in ["dc", "db"] and size > 0, do: [body | acc], else: acc
    riff(skip_pad(more, size), usec, acc)
  end

  defp riff(_, usec, acc), do: {usec, acc}

  defp skip_pad(bin, size) when rem(size, 2) == 1 and byte_size(bin) > 0, do: binary_part(bin, 1, byte_size(bin) - 1)
  defp skip_pad(bin, _), do: bin

  # JPEG Annex K tables (K.3) for frames that omit DHT
  @dc_l <<0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0>>
  @dc_v <<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11>>
  @dcc_l <<0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>>
  @ac_l <<0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D>>
  @acc_l <<0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77>>
  @ac_v <<0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
          0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
          0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
          0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
          0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
          0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
          0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA>>
  @acc_v <<0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
           0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
           0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
           0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
           0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
           0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
           0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA>>

  defp with_huffman(j) do
    if :binary.match(j, <<0xFF, 0xC4>>) != :nomatch do
      j
    else
      case :binary.match(j, <<0xFF, 0xDA>>) do
        {pos, _} ->
          dht = fn class_id, l, v -> body = <<class_id>> <> l <> v; <<0xFF, 0xC4, byte_size(body) + 2::16>> <> body end
          tables = dht.(0x00, @dc_l, @dc_v) <> dht.(0x10, @ac_l, @ac_v) <> dht.(0x01, @dcc_l, @dc_v) <> dht.(0x11, @acc_l, @acc_v)
          binary_part(j, 0, pos) <> tables <> binary_part(j, pos, byte_size(j) - pos)

        :nomatch -> j
      end
    end
  end

end
