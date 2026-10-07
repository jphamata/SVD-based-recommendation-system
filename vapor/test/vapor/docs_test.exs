defmodule Vapor.DocsTest do
  @moduledoc """
  The document airlock against files made by other producers (fpdf2,
  Ghostscript, qpdf, python-docx, openpyxl, python-pptx, Pillow, pypng):
  text with provenance, refusals that say why, archives that cannot be
  bombed — and two external oracles when present (poppler's pdftotext,
  Pillow's PNG decoder).
  """
  use ExUnit.Case, async: true
  alias Vapor.Docs
  alias Vapor.Docs.{Markup, PDF, Pictures, Zip}
  alias Vapor.Rejection
  import Vapor.TestHelpers

  @dir Path.expand("../fixtures/docs", __DIR__)
  defp fx(name), do: Path.join(@dir, name)
  defp ingest!(name, opts \\ []), do: ({:ok, r} = Docs.ingest(fx(name), opts); r)
  defp texts(r), do: Map.new(r.passages, &{&1.doc, &1.text})

  test "kinds come from bytes, not names" do
    for {f, k} <- [{"simple.pdf", :pdf}, {"scene.png", :png}, {"scene.jpg", :jpeg}, {"doc.docx", :docx}, {"sheet.xlsx", :xlsx},
                   {"deck.pptx", :pptx}, {"note.odt", :odf}, {"book.epub", :epub}, {"bundle.zip", :zip}, {"page.html", :html}] do
      assert Docs.sniff("renamed.bin", File.read!(fx(f))) == (if k == :html, do: :html, else: k), f
    end

    assert Docs.sniff("x.txt", <<0, 1, 2, 255>>) == :binary
    assert Docs.sniff("x.md", "# título") == :markdown
  end

  test "PDF: per-page text from three producers, WinAnsi, Type 1 re-encoded, Identity-H through ToUnicode, object streams, linearized" do
    assert texts(ingest!("simple.pdf")) == %{"simple.pdf#p1" => "Eclusa de documentos: página um. Olá, ação, coração.",
                                             "simple.pdf#p2" => "Second page: the quick brown fox jumps over the lazy dog."}
    uni = "Unicode: αβγ → ∑ — São Paulo, Ñandú, Ελληνικά.\nMerkle roots name the corpus; every chunk carries a proof."
    assert texts(ingest!("unicode.pdf")) == %{"unicode.pdf#p1" => uni}
    assert texts(ingest!("objstm.pdf")) == %{"objstm.pdf#p1" => uni}
    assert texts(ingest!("gs.pdf"))["gs.pdf#p1"] == "Ghostscript: produtor tipo 1, com acentuação e élève.\nSegunda linha, mesma página."
    assert texts(ingest!("linearized.pdf")) |> map_size() == 2
  end

  test "PDF: refusals and gaps are said, not guessed" do
    assert {:error, %Rejection{bound: b}} = PDF.pages(File.read!(fx("encrypted.pdf")))
    assert b =~ "encrypted"
    r = ingest!("scanned.pdf")
    assert r.passages == []
    assert [w] = r.warnings
    assert w =~ "no text layer"
    assert {:error, %Rejection{}} = PDF.pages("%PDF-1.4 garbage")
  end

  @tag :native
  test "OCR: a picture of text is found by what it says; pictures and scans without text add nothing" do
    line = Path.expand("../../priv/quality/ocr/00002.png", __DIR__)
    {:ok, r} = Docs.ingest(line)
    assert [%{text: text, kind: :image_text, meta: %{ocr: %{confidence: c}}}] = Enum.filter(r.passages, &String.ends_with?(&1.doc, "#ocr"))
    assert Vapor.Vision.OCR.cer(text, "extents, which the monotonicity lemmas") < 0.1 and c >= 0.8

    {:ok, lib, _} = Docs.Library.add(Docs.Library.new(), line)
    {:ok, lib, _} = Docs.Library.add(lib, fx("scene.png"))
    assert [%{doc: doc} | _] = Docs.Library.search(lib, "monotonicity lemmas", k: 3).hits
    assert doc =~ "00002.png#ocr"

    # a synthetic scene, a gradient, a scanned page that is a black bar: no invented text
    for f <- ["scene.png", "gray.png", "palette.png", "interlaced.png"] do
      r = ingest!(f)
      assert Enum.all?(r.passages, &(&1.kind == :image)) and r.warnings == [], f
    end

    assert %{passages: [], warnings: [w]} = ingest!("scanned.pdf")
    assert w =~ "no text layer, and OCR found no text"
  end

  test "Office and OpenDocument: paragraphs, tables, sheets by name, slides in order" do
    assert texts(ingest!("doc.docx"))["doc.docx"] ==
             "Relatório da eclusa\nPrimeiro parágrafo com acentuação: ação & reação <ok>.\nmodelo\tfamília\nphi3\tllama\nÚltimo parágrafo."

    assert texts(ingest!("sheet.xlsx")) == %{"sheet.xlsx#sheet:Modelos" => "família\tcamadas\tativação\nllama\t32\tsilu\ngemma3_text\t26\tgelu_tanh",
                                             "sheet.xlsx#sheet:Notas" => "a soma\t2.5\n=B1*2"}
    assert texts(ingest!("deck.pptx")) == %{"deck.pptx#slide1" => "Any-to-any\nHub, não matriz", "deck.pptx#slide2" => "Qualidade\nPortões calibrados"}
    assert texts(ingest!("note.odt"))["note.odt"] == "Título ODT\nTexto em OpenDocument com & e <tags>."
    assert texts(ingest!("book.epub")) == %{"book.epub!/OEBPS/cap1.xhtml" => "Capítulo 1\nEra uma vez uma eclusa."}
  end

  test "HTML: title, blocks, entities; scripts and styles dropped" do
    assert texts(ingest!("page.html"))["page.html"] == "Página\nCabeçalho\nUm parágrafo & outro espaço.\nitem um\nitem dois"
    assert Markup.entities("&#x263A; &#65; &amp;amp; &bogus;") == "☺ A &amp; &bogus;"
  end

  test "archives: recursive, with a provenance path per passage and a manifest of every file" do
    r = ingest!("bundle.zip")
    docs = Enum.map(r.passages, & &1.doc)
    assert "bundle.zip!/pasta/simple.pdf#p2" in docs
    assert "bundle.zip!/inner.zip!/deep/notes.md" in docs
    assert "bundle.zip!/pasta/doc.docx" in docs
    assert [%{doc: "bundle.zip!/pasta/scene.png"}] = r.images
    assert Enum.any?(r.warnings, &(&1 =~ "binario.bin: binary"))
    manifest = Map.new(r.files, &{&1.path, &1})
    assert manifest["bundle.zip"].sha256 == :crypto.hash(:sha256, File.read!(fx("bundle.zip"))) |> Base.encode16(case: :lower)
    assert manifest["bundle.zip!/pasta/simple.pdf"].sha256 == :crypto.hash(:sha256, File.read!(fx("simple.pdf"))) |> Base.encode16(case: :lower)
  end

  test "zip bombs are refused before inflation; lying headers are refused during it; limits are options" do
    assert {:error, %Rejection{bound: b}} = Docs.ingest(fx("bomb.zip"))
    assert b =~ "compression ratio"
    assert {:error, %Rejection{}} = Docs.ingest(fx("bundle.zip"), max_depth: 1)
    assert {:error, %Rejection{}} = Docs.ingest(fx("bundle.zip"), max_total: 10_000)
    assert {:error, %Rejection{}} = Docs.ingest(fx("bundle.zip"), max_entries: 2)

    # a member that declares 10 bytes and inflates to 100 000
    data = :binary.copy("a", 100_000)
    z = :zlib.open(); :ok = :zlib.deflateInit(z, :default, :deflated, -15, 8, :default)
    comp = IO.iodata_to_binary(:zlib.deflate(z, data, :finish)); :zlib.close(z)
    lie = zip_one("x.txt", comp, 10, :erlang.crc32(data))
    assert {:ok, [e]} = Zip.entries(lie)
    assert {:error, %Rejection{bound: b2}} = Zip.member(lie, e, 1_000_000)
    assert b2 =~ "inflates beyond"
  end

  # a one-member zip with an arbitrary declared size
  defp zip_one(name, comp, usize, crc) do
    local = <<"PK", 3, 4, 20::16-little, 0::16, 8::16-little, 0::32, crc::32-little, byte_size(comp)::32-little, usize::32-little,
              byte_size(name)::16-little, 0::16>> <> name <> comp
    cd = <<"PK", 1, 2, 20::16-little, 20::16-little, 0::16, 8::16-little, 0::32, crc::32-little, byte_size(comp)::32-little, usize::32-little,
           byte_size(name)::16-little, 0::16, 0::16, 0::16, 0::16, 0::32, 0::32-little>> <> name
    local <> cd <> <<"PK", 5, 6, 0::16, 0::16, 1::16-little, 1::16-little, byte_size(cd)::32-little, byte_size(local)::32-little, 0::16>>
  end

  test "pictures: PNG decoded (all colour types and depths, Adam7), text chunks; JPEG by its EXIF text" do
    r = ingest!("scene.png")
    assert [%{text: t}] = r.passages
    assert t =~ "Description: dois discos: vermelho e azul" and t =~ "Title: Cena de teste"
    assert [%{image: img}] = r.images
    assert {img.w, img.h, img.c} == {64, 48, 3}

    for f <- ~w(palette rgba gray gray16 interlaced gray4 bits1 rgb16) do
      assert {:ok, %{image: %Vapor.Modal.Image{}}} = Pictures.read(:png, File.read!(fx("#{f}.png"))), f
    end

    # JPEG: decoded since 0.5.0 (Vapor.Docs.JPEG), and still indexed by its EXIF text
    j = ingest!("scene.jpg")
    assert [%{text: jt}] = j.passages
    assert jt =~ "ImageDescription: foto de dois discos" and jt =~ "64×48"
    assert j.warnings == []
    assert [%{image: %Vapor.Modal.Image{w: 64, h: 48, c: 3}}] = j.images
  end

  @tag :python
  test "PNG pixels = Pillow's, for every fixture (16-bit within Pillow's 8-bit truncation)" do
    if python?(["PIL"]) do
      for f <- ~w(scene palette rgba gray gray16 interlaced gray4 bits1 rgb16) do
        {:ok, %{image: img}} = Pictures.read(:png, File.read!(fx("#{f}.png")))
        script = """
        import json, sys, warnings
        warnings.simplefilter("ignore")
        from PIL import Image
        im = Image.open(sys.argv[1])
        im = im.convert("RGB") if #{img.c} == 3 else (im.point(lambda v: v / 257).convert("L") if im.mode.startswith("I") else im.convert("L"))
        print(json.dumps([v / 255 for px in im.getdata() for v in (px if isinstance(px, tuple) else (px,))]))
        """
        {:ok, ref} = Vapor.JSON.decode(py!(script, [fx("#{f}.png")]))
        tol = if f in ~w(gray16 rgb16), do: 1 / 255 + 1.0e-9, else: 1.0e-9
        worst = Enum.zip_with(Vapor.Modal.Image.values(img), ref, &abs(&1 - &2)) |> Enum.max()
        assert worst <= tol, "#{f}: #{worst}"
      end
    end
  end

  @tag :python
  test "PDF text = poppler's pdftotext (whitespace-normalised), when present" do
    if System.find_executable("pdftotext") do
      norm = fn t -> t |> String.split() |> Enum.join(" ") end

      for f <- ~w(simple unicode objstm gs linearized) do
        {:ok, pages, _} = PDF.pages(File.read!(fx("#{f}.pdf")))
        {out, 0} = System.cmd("pdftotext", ["-layout", fx("#{f}.pdf"), "-"])
        assert norm.(Enum.join(pages, " ")) == norm.(String.replace(out, "\f", " ")), f
      end
    end
  end

  test "the library: one root over passages and files, receipts that recompute, duplicates ignored, picture search" do
    alias Vapor.Docs.Library
    lib = Enum.reduce(~w(bundle.zip gs.pdf sheet.xlsx palette.png gray.png), Library.new(), fn f, l -> {:ok, l, _} = Library.add(l, fx(f)); l end)
    {:ok, same, %{duplicate: "gs.pdf"}} = Library.add(lib, fx("gs.pdf"))
    assert Library.root(same) == Library.root(lib)

    r = Library.search(lib, "camadas ativação", k: 2)
    assert hd(r.hits).doc == "sheet.xlsx#sheet:Modelos"
    assert :ok = Library.verify(lib, r)
    assert Enum.all?(r.hits, &Vapor.RAG.member?(&1, lib.rag.root))

    # another file changes the root, and the old result no longer verifies against it
    {:ok, more, _} = Library.add(lib, fx("deck.pptx"))
    refute Library.root(more) == Library.root(lib)
    assert {:error, :different_library} = Library.verify(more, r)

    {:ok, %{image: q}} = Pictures.read(:png, File.read!(fx("rgba.png")))
    assert [%{doc: top} | _] = Library.search_image(lib, q, k: 3)
    assert top in ["bundle.zip!/pasta/scene.png", "palette.png"]
  end

  test "command-line arguments read without a UTF-8 locale are repaired, nothing else is touched" do
    mangled = "família" |> :binary.bin_to_list() |> List.to_string()
    assert Vapor.CLI.utf8_arg(mangled) == "família"
    assert Vapor.CLI.utf8_arg("família") == "família"
    assert Vapor.CLI.utf8_arg("plain ascii") == "plain ascii"
    assert Vapor.CLI.utf8_arg("ÿ é") == "ÿ é"
  end
end
