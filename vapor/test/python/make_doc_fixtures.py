"""Regenerate test/fixtures/docs — the document airlock's fixtures.

    python3 test/python/make_doc_fixtures.py   (needs fpdf2, python-docx,
    openpyxl, python-pptx, Pillow, pypng, and qpdf + ghostscript on the PATH)

Every fixture is made by a mainstream producer (fpdf2, Word's python-docx,
openpyxl, python-pptx, Pillow, qpdf), so the extractors are tested against
files they did not write. The texts are fixed; tests assert on them.
"""
import io, json, os, subprocess, zipfile, zlib

from fpdf import FPDF
import docx
import openpyxl
from pptx import Presentation
from pptx.util import Inches
from PIL import Image, ImageDraw, PngImagePlugin

OUT = os.path.join(os.path.dirname(__file__), "..", "fixtures", "docs")
os.makedirs(OUT, exist_ok=True)
P = lambda *a: os.path.join(OUT, *a)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# ---- PDF, core font (WinAnsi), two pages, Flate-compressed content
pdf = FPDF()
pdf.set_compression(True)
pdf.add_page(); pdf.set_font("Helvetica", size=14)
pdf.multi_cell(0, 8, new_x="LMARGIN", new_y="NEXT", text="Eclusa de documentos: página um. Olá, ação, coração.")
pdf.add_page()
pdf.multi_cell(0, 8, new_x="LMARGIN", new_y="NEXT", text="Second page: the quick brown fox jumps over the lazy dog.")
pdf.output(P("simple.pdf"))

# ---- PDF, embedded TrueType (Type0 / Identity-H with a ToUnicode CMap)
pdf = FPDF()
pdf.add_font("DejaVu", "", FONT)
pdf.add_page(); pdf.set_font("DejaVu", size=12)
pdf.multi_cell(0, 7, new_x="LMARGIN", new_y="NEXT", text="Unicode: αβγ → ∑ — São Paulo, Ñandú, Ελληνικά.")
pdf.multi_cell(0, 7, new_x="LMARGIN", new_y="NEXT", text="Merkle roots name the corpus; every chunk carries a proof.")
pdf.output(P("unicode.pdf"))

# ---- PDF 1.5 object streams and an xref stream (qpdf)
subprocess.run(["qpdf", "--object-streams=generate", P("unicode.pdf"), P("objstm.pdf")], check=True)
# ---- encrypted (must be refused, not misread)
subprocess.run(["qpdf", "--encrypt", "u", "o", "256", "--", P("simple.pdf"), P("encrypted.pdf")], check=True)

# ---- a "scanned" PDF: one page that is only an image (no text layer)
img = Image.new("RGB", (200, 80), (240, 240, 240)); ImageDraw.Draw(img).rectangle([20, 20, 180, 60], fill=(30, 30, 30))
buf = io.BytesIO(); img.save(buf, "PNG")
pdf = FPDF(); pdf.add_page(); pdf.image(buf, x=10, y=10, w=100); pdf.output(P("scanned.pdf"))

# ---- Office Open XML
d = docx.Document()
d.add_heading("Relatório da eclusa", 1)
d.add_paragraph("Primeiro parágrafo com acentuação: ação & reação <ok>.")
t = d.add_table(rows=2, cols=2)
t.cell(0, 0).text, t.cell(0, 1).text, t.cell(1, 0).text, t.cell(1, 1).text = "modelo", "família", "phi3", "llama"
d.add_paragraph("Último parágrafo.")
d.save(P("doc.docx"))

wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Modelos"
for row in [["família", "camadas", "ativação"], ["llama", 32, "silu"], ["gemma3_text", 26, "gelu_tanh"]]:
    ws.append(row)
ws2 = wb.create_sheet("Notas"); ws2["A1"] = "a soma"; ws2["B1"] = 2.5; ws2["B2"] = "=B1*2"
wb.save(P("sheet.xlsx"))

pr = Presentation()
s = pr.slides.add_slide(pr.slide_layouts[1]); s.shapes.title.text = "Any-to-any"; s.placeholders[1].text = "Hub, não matriz"
s = pr.slides.add_slide(pr.slide_layouts[1]); s.shapes.title.text = "Qualidade"; s.placeholders[1].text = "Portões calibrados"
pr.save(P("deck.pptx"))

# ---- OpenDocument text (written by hand: a zip with content.xml)
odt = io.BytesIO()
with zipfile.ZipFile(odt, "w") as z:
    z.writestr("mimetype", "application/vnd.oasis.opendocument.text", compress_type=zipfile.ZIP_STORED)
    z.writestr("content.xml", '<?xml version="1.0" encoding="UTF-8"?><office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text><text:h>Título ODT</text:h><text:p>Texto em <text:span>OpenDocument</text:span> com &amp; e &lt;tags&gt;.</text:p></office:text></office:body></office:document-content>')
open(P("note.odt"), "wb").write(odt.getvalue())

# ---- web and plain formats
open(P("page.html"), "w", encoding="utf-8").write("<!doctype html><html><head><title>Página</title><style>p{color:red}</style><script>var x = '<p>não</p>';</script></head><body><h1>Cabeçalho</h1><p>Um parágrafo &amp; outro&nbsp;espaço.</p><ul><li>item um</li><li>item dois</li></ul></body></html>")
open(P("notes.md"), "w", encoding="utf-8").write("# Notas\n\nA eclusa recusa o que não entende.\n")
open(P("data.csv"), "w", encoding="utf-8").write("modelo,bits\nbigrama,3.69\nunigrama,4.30\n")
json.dump({"modelo": "bigrama", "bits": 3.69, "nota": "texto em JSON"}, open(P("data.json"), "w", encoding="utf-8"), ensure_ascii=False)

# ---- images
def scene(mode="RGB"):
    im = Image.new("RGB", (64, 48), (120, 130, 150)); dr = ImageDraw.Draw(im)
    dr.ellipse([6, 10, 30, 34], fill=(220, 40, 30)); dr.ellipse([36, 10, 60, 34], fill=(30, 60, 220))
    return im
meta = PngImagePlugin.PngInfo(); meta.add_text("Description", "dois discos: vermelho e azul"); meta.add_itxt("Title", "Cena de teste", lang="pt")
scene().save(P("scene.png"), pnginfo=meta)
scene().convert("P", palette=Image.ADAPTIVE, colors=16).save(P("palette.png"))
scene().convert("RGBA").save(P("rgba.png"))
g = scene().convert("L"); g.save(P("gray.png"))
scene().convert("L").point(lambda v: v).convert("I;16").save(P("gray16.png"))
ex = Image.Exif(); ex[0x010E] = "foto de dois discos"  # ImageDescription
scene().save(P("scene.jpg"), quality=90, exif=ex.tobytes())

# ---- PNG edge cases (pypng): Adam7 interlacing at odd sizes, sub-byte depths, 16-bit RGB
import png
w, h = 37, 23
rgb = [[(x * 7 + y * 3) % 256 if c == 0 else (x * y) % 256 if c == 1 else (255 - x * 5) % 256 for x in range(w) for c in range(3)] for y in range(h)]
png.Writer(w, h, greyscale=False, bitdepth=8, interlace=True).write(open(P("interlaced.png"), "wb"), rgb)
png.Writer(w, h, greyscale=True, bitdepth=4).write(open(P("gray4.png"), "wb"), [[(x + y) % 16 for x in range(w)] for y in range(h)])
png.Writer(w, h, greyscale=True, bitdepth=1, interlace=True).write(open(P("bits1.png"), "wb"), [[(x ^ y) & 1 for x in range(w)] for y in range(h)])
png.Writer(w, h, greyscale=False, bitdepth=16).write(open(P("rgb16.png"), "wb"), [[(x * 1777 + c * 9000 + y * 13) % 65536 for x in range(w) for c in range(3)] for y in range(h)])

# ---- a third PDF producer: Ghostscript (Type 1 font, ISOLatin1 re-encoding), and a linearized PDF (qpdf)
ps = b"""%!PS
/Helvetica findfont dup length dict begin {1 index /FID ne {def} {pop pop} ifelse} forall /Encoding ISOLatin1Encoding def currentdict end /HelvL1 exch definefont pop
/HelvL1 findfont 14 scalefont setfont
72 720 moveto (Ghostscript: produtor tipo 1, com acentua\347\343o e \351l\350ve.) show
72 700 moveto (Segunda linha, mesma p\341gina.) show
showpage
72 720 moveto /HelvL1 findfont 14 scalefont setfont (P\341gina dois do PostScript.) show
showpage
"""
open(P("gs.ps"), "wb").write(ps)
subprocess.run(["ps2pdf", P("gs.ps"), P("gs.pdf")], check=True); os.remove(P("gs.ps"))
subprocess.run(["qpdf", "--linearize", P("simple.pdf"), P("linearized.pdf")], check=True)

# ---- EPUB (zip of XHTML)
ep = io.BytesIO()
with zipfile.ZipFile(ep, "w") as z:
    z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
    z.writestr("OEBPS/cap1.xhtml", "<html><body><h1>Capítulo 1</h1><p>Era uma vez uma eclusa.</p></body></html>")
open(P("book.epub"), "wb").write(ep.getvalue())

# ---- a bundle: nested zip, a directory, mixed formats
inner = io.BytesIO()
with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(P("notes.md"), "deep/notes.md")
with zipfile.ZipFile(P("bundle.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["simple.pdf", "doc.docx", "scene.png", "data.csv"]:
        z.write(P(f), "pasta/" + f)
    z.writestr("inner.zip", inner.getvalue())
    z.writestr("binario.bin", bytes(range(256)) * 4)

# ---- a zip bomb: 64 MiB of zeros compressed to a few KiB (must be refused before inflation)
with zipfile.ZipFile(P("bomb.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    with z.open("zeros.txt", "w") as f:
        chunk = b"\0" * (1 << 20)
        for _ in range(64):
            f.write(chunk)
print("fixtures in", os.path.abspath(OUT))
