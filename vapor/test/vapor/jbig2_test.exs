defmodule Vapor.JBIG2Test do
  @moduledoc """
  JBIG2 (`Vapor.Docs.JBIG2`, docs/OCR.md §3f). Every fixture was kept only
  because the reference decoder, jbig2dec, decoded it to the bitmap its
  writer composed (`test/python/jbig2_streams.py`); vapor must produce the
  same bitmap — pinned by SHA-256 — for all of them: jbig2enc's streams of a
  real scanned page (generic, TPGD, symbol mode, PDF mode with globals) and
  the streams written for everything jbig2enc does not exercise (generic
  templates 1–3 and moved adaptive pixels, MMR, refinement regions with and
  without TPGRON, text regions in all eight corner/transposition
  combinations with strips, DSOFFSET, XOR, a black default pixel and refined
  instances, symbol dictionaries with refinement and aggregation, a striped
  page of unknown height, and — 0.15 — Huffman-coded dictionaries and text
  regions with standard and custom code tables).
  """
  use ExUnit.Case, async: true
  import Bitwise
  alias Vapor.Docs.JBIG2

  @dir Path.expand("../fixtures/docs/jbig2", __DIR__)

  defp manifest, do: @dir |> Path.join("manifest.json") |> File.read!() |> Vapor.JSON.decode() |> elem(1)

  defp pbm_sha(%{w: w, h: h} = bm),
    do: :crypto.hash(:sha256, "P4\n#{w} #{h}\n" <> JBIG2.packed(bm)) |> Base.encode16(case: :lower)

  test "the MQ decoder reproduces the T.88 H.2 test sequence" do
    enc = Base.decode16!("84C73BFCE1A1430402200000410DBB86F4317FFF88FF37471ADB6ADFFFAC")
    cx = JBIG2.stats(1)
    {bits, _} = Enum.map_reduce(1..256, JBIG2.mq_new(enc), fn _, st -> JBIG2.decode_bit(st, cx, 0) end)
    out = for c <- Enum.chunk_every(bits, 8), into: <<>>, do: <<Enum.reduce(c, 0, &(&2 * 2 + &1))>>
    assert Base.encode16(out) == "00020051000000C00352872AAAAAAAAA82C02000FCD79EF6BF7FED904F46A3BF"
  end

  test "every fixture decodes to jbig2dec's bitmap, bit for bit" do
    m = manifest()
    assert map_size(m) >= 56
    assert Enum.count(m, fn {k, _} -> String.starts_with?(k, "huff_") end) == 13

    for {name, %{"files" => files, "sha256" => sha, "w" => w, "h" => h}} <- m do
      bins = Enum.map(files, &File.read!(Path.join(@dir, &1)))
      r = case bins do
        [one] -> JBIG2.decode(one)
        [globals, page] -> JBIG2.decode(page, globals)
      end

      assert {:ok, bm} = r, name
      assert {bm.w, bm.h} == {w, h}, name
      assert pbm_sha(bm) == sha, name
      assert bm.warnings == [], name
    end
  end

  test "a PDF with JBIG2Decode and JBIG2Globals: the page image is the decoded bitmap (0 = ink after the filter)" do
    pdf = File.read!(Path.join(@dir, "enc_pdf.pdf"))
    {:ok, %{1 => [img]}, warns} = Vapor.Docs.PDF.images(pdf, [1])
    refute Enum.any?(warns, &(&1 =~ "JBIG2"))
    {:ok, bm} = JBIG2.decode(File.read!(Path.join(@dir, "enc_pdf.0000")), File.read!(Path.join(@dir, "enc_pdf.sym")))
    assert {img.w, img.h} == {bm.w, bm.h}
    ink = img.px |> Tuple.to_list() |> Enum.map(&if(&1 < 0.5, do: 1, else: 0))
    assert ink == bm.rows |> Tuple.to_list() |> Enum.flat_map(&Tuple.to_list/1)
  end

  test "damaged streams never crash: truncations and garbage end in a page or an error" do
    whole = File.read!(Path.join(@dir, "text_rc0_strips_ref.jb2"))

    for cut <- [10, 30, 60, byte_size(whole) - 40, byte_size(whole) - 3] do
      assert match?({:ok, _}, JBIG2.decode(binary_part(whole, 0, cut))) or match?({:error, _}, JBIG2.decode(binary_part(whole, 0, cut)))
    end

    :rand.seed(:exsss, {1, 2, 3})
    for _ <- 1..30 do
      junk = :crypto.strong_rand_bytes(200)
      assert match?({:ok, _}, JBIG2.decode(junk)) or match?({:error, _}, JBIG2.decode(junk))
    end
  end

  test "Huffman coding with refinement, and halftones, are refused with a reason, not misread" do
    seg = fn num, type, data, refs ->
      <<num::32, type, length(refs) <<< 5>> <> for(r <- refs, into: <<>>, do: <<r>>) <> <<1, byte_size(data)::32>> <> data
    end

    page = seg.(1, 48, <<20::32, 10::32, 0::32, 0::32, 0, 0::16>>, [])
    # SDHUFF with SDREFAGG (and SDRTEMPLATE 1: no refinement AT bytes)
    huff_dict = seg.(2, 0, <<(1 ||| 2 ||| 1 <<< 12)::16, 0::32, 0::32>>, [])
    halftone = seg.(3, 22, <<0::size(17 * 8)>>, [])
    {:ok, bm} = JBIG2.decode(page <> huff_dict <> halftone)
    assert {bm.w, bm.h} == {20, 10}
    assert Enum.any?(bm.warnings, &(&1 =~ "Huffman"))
    assert Enum.any?(bm.warnings, &(&1 =~ "halftone"))
  end

  test "an empty Huffman-coded dictionary is a dictionary, not a warning" do
    seg = fn num, type, data -> <<num::32, type, 0, 1, byte_size(data)::32>> <> data end
    page = seg.(1, 48, <<8::32, 8::32, 0::32, 0::32, 0, 0::16>>)
    {:ok, bm} = JBIG2.decode(page <> seg.(2, 0, <<1::16, 0::32, 0::32>>))
    assert bm.warnings == []
  end
end
