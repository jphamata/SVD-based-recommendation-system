defmodule Vapor.CCITTTest do
  @moduledoc """
  Scanned office pages, decoded: CCITT Group 3/4 fax (`Vapor.Docs.CCITT`),
  LZW and RunLength — against libtiff's encoder through Pillow, bit for bit
  (committed fixtures with digests, always; every coding of
  `test/python/ccitt_fixtures.py` when Python with Pillow is present) — and
  inside a PDF, with the polarity rules (`BlackIs1`, `Decode`, stencil
  masks) that decide whether ink comes out black.
  """
  use ExUnit.Case, async: true
  alias Vapor.Docs.{CCITT, PDF}
  import Vapor.TestHelpers

  defp fx(name), do: File.read!(Path.expand("../fixtures/docs/ccitt/#{name}", __DIR__))
  defp digest(bin), do: Base.encode16(:crypto.hash(:sha256, bin), case: :lower)

  # {fixture, params, columns, rows, sha256 of libtiff's bitmap (1 = black)}
  @committed [
    {"page_g4.bin", [k: -1], 1240, 420, "5fb0679aa971c843695ce37ef1fb49bb6c91ec1918eb0918ca3c9623b56c0b87"},
    {"runs_g3_2dfill.bin", [k: 1], 2700, 71, "12f8edc0e6f92557ac4613cc958cad2f051154a7bfc947f7aeda148b40496ddc"},
    {"noise30_mh_aligned.bin", [k: 0, byte_align: true], 64, 64, "1f6fc5bc9ab41b9273b8b96470a80dffbfd37dc2de7b7e568bf85637e4ae4dcb"},
    {"noise5_g3.bin", [k: 0], 101, 37, "318cd021cc515c06095230083aa6abed94f408768f305b9bcd213d7c32c26ade"}
  ]

  test "committed streams: Group 4, Group 3 2-D with fill, byte-aligned MH, Group 3 1-D = libtiff (SHA-256)" do
    for {f, p, cols, rows, sha} <- @committed do
      # rows: 0 — the decoder finds the end itself (EOFB, RTC, end of data)
      assert {:ok, r} = CCITT.decode(fx(f), [columns: cols, rows: 0, black_is_1: true] ++ p), f
      assert {r.columns, r.rows, r.warnings} == {cols, rows, []}, f
      assert digest(r.data) == sha, f
    end
  end

  test "polarity: without BlackIs1, ink is 0 (the PDF default) — the exact complement" do
    {:ok, a} = CCITT.decode(fx("page_g4.bin"), k: -1, columns: 1240, black_is_1: true)
    {:ok, b} = CCITT.decode(fx("page_g4.bin"), k: -1, columns: 1240)
    assert b.data == for(<<x <- a.data>>, into: <<>>, do: <<Bitwise.bxor(x, 255)>>)
  end

  test "damaged data: a truncated stream keeps its rows, garbage never crashes nor runs away" do
    full = fx("page_g4.bin")
    assert {:ok, r} = CCITT.decode(binary_part(full, 0, div(byte_size(full), 2)), k: -1, columns: 1240, rows: 420)
    assert r.rows in 100..420

    for seed <- 1..20 do
      :rand.seed(:exsss, {seed, 2, 3})
      junk = for _ <- 1..400, into: <<>>, do: <<:rand.uniform(256) - 1>>

      for k <- [-1, 0, 1] do
        assert {:ok, %{rows: n}} = CCITT.decode(junk, k: k, columns: 300, rows: 500)
        assert n <= 500
      end
    end

    assert {:error, _} = CCITT.decode(<<0>>, columns: 0)
    assert {:error, _} = CCITT.decode(<<0>>, columns: 60_000, rows: 60_000)
  end

  test "RunLength (PackBits) and LZW, both EarlyChange values, end markers honoured" do
    rle = fn bin -> elem(PDF.decode(%{"Filter" => {:name, "RunLengthDecode"}, :stream => bin}), 1) end
    assert rle.(<<2, ?a, ?b, ?c, 254, ?x, 128, ?z>>) == "abcxxx"
    # LZW: "ABABABA" as 9-bit codes 256 65 66 258 260 257
    codes = [256, 65, 66, 258, 260, 257]
    bits = for c <- codes, into: <<>>, do: <<c::9>>
    pad = rem(8 - rem(bit_size(bits), 8), 8)
    assert PDF.lzw(<<bits::bitstring, 0::size(pad)>>) == "ABABABA"
    lzw = %{"Filter" => {:name, "LZWDecode"}, "DecodeParms" => %{"EarlyChange" => 0}, :stream => <<bits::bitstring, 0::size(pad)>>}
    assert {:ok, "ABABABA"} = PDF.decode(lzw)
  end

  # a one-page PDF around an image XObject
  defp pdf(img_dict, stream) do
    objs = [
      "<< /Type /Catalog /Pages 2 0 R >>",
      "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
      "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 200] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>",
      "<< /Type /XObject /Subtype /Image #{img_dict} /Length #{byte_size(stream)} >>\nstream\n" <> stream <> "\nendstream",
      "<< /Length 25 >>\nstream\nq 600 0 0 200 0 0 cm /Im0 Do Q\nendstream"
    ]

    body = objs |> Enum.with_index(1) |> Enum.map_join(fn {o, i} -> "#{i} 0 obj\n#{o}\nendobj\n" end)
    "%PDF-1.4\n" <> body <> "trailer\n<< /Root 1 0 R >>\n%%EOF\n"
  end

  defp ink(%Vapor.Modal.Image{px: px}), do: px |> Tuple.to_list() |> Enum.map(&if(&1 < 0.5, do: 1, else: 0))

  test "in a PDF: Group 4 as DeviceGray, as a stencil mask, and BlackIs1 + Decode [1 0] — ink is black in all three" do
    {:ok, ref} = CCITT.decode(fx("page_g4.bin"), k: -1, columns: 1240, black_is_1: true)
    want = for <<row::binary-size(155) <- ref.data>>, <<b::1 <- row>>, do: b
    want = want |> Enum.chunk_every(1240) |> Enum.flat_map(& &1)
    parms = "/DecodeParms << /K -1 /Columns 1240 /Rows 420 >>"

    variants = [
      "/Width 1240 /Height 420 /ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /CCITTFaxDecode #{parms}",
      "/Width 1240 /Height 420 /ImageMask true /Filter /CCITTFaxDecode #{parms}",
      "/Width 1240 /Height 420 /ColorSpace /DeviceGray /BitsPerComponent 1 /Decode [1 0] /Filter [/CCITTFaxDecode] /DecodeParms [<< /K -1 /Columns 1240 /Rows 420 /BlackIs1 true >>]"
    ]

    for v <- variants do
      assert {:ok, %{1 => [img]}, []} = PDF.images(pdf(v, fx("page_g4.bin")), [1]), v
      assert {img.w, img.h} == {1240, 420}
      assert ink(img) == want, v
    end

    # a broken JBIG2 stream (decoded since 0.8) is named with the reason, not guessed
    assert {:ok, %{1 => []}, [w]} = PDF.images(pdf("/Width 8 /Height 8 /Filter /JBIG2Decode", "xx"), [1])
    assert w =~ "JBIG2"
  end

  test "a stencil mask in Flate: a 0 sample is ink (it was read inverted before 0.7)" do
    # 16×2: row 0 all ink (0 bits), row 1 all paper (1 bits)
    raw = <<0, 0, 255, 255>>
    z = :zlib.compress(raw)
    assert {:ok, %{1 => [img]}, []} = PDF.images(pdf("/Width 16 /Height 2 /ImageMask true /Filter /FlateDecode", z), [1])
    assert ink(img) == List.duplicate(1, 16) ++ List.duplicate(0, 16)
  end

  @tag :python
  @tag timeout: 600_000
  test "every coding = libtiff, bit for bit; LZW and PackBits = libtiff; LZW EarlyChange 0 = an independent encoder" do
    if python?(["PIL", "numpy"]) do
      dir = Path.join(System.tmp_dir!(), "vapor-ccitt-#{System.unique_integer([:positive])}")
      py!(File.read!(Path.expand("../python/ccitt_fixtures.py", __DIR__)), [dir])
      {:ok, idx} = Vapor.JSON.decode(File.read!(Path.join(dir, "index.json")))
      assert map_size(idx) >= 51
      read = fn n, ext -> File.read!(Path.join(dir, n <> ext)) end

      for {name, p} <- idx do
        got =
          case p do
            %{"filter" => "lzw", "early" => e} -> PDF.lzw(read.(name, ".bin"), e)
            %{"filter" => "rle"} -> elem(PDF.decode(%{"Filter" => {:name, "RunLengthDecode"}, :stream => read.(name, ".bin")}), 1)
            %{"k" => k, "columns" => c, "rows" => r, "byte_align" => a} ->
              {:ok, d} = CCITT.decode(read.(name, ".bin"), k: k, columns: c, black_is_1: true, byte_align: a)
              assert d.rows == r, name
              d.data
          end

        assert got == read.(name, ".raw"), "#{name}: differs from libtiff"
      end

      File.rm_rf!(dir)
    end
  end
end
