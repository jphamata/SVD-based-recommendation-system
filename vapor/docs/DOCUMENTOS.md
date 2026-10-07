# Documentos: a eclusa de arquivos e a biblioteca verificável

> O RAG falha antes da busca: no PDF que vira lixo, no zip que derruba o
> servidor, na resposta que não sabe dizer de que arquivo veio.

## 1. As dores

- **Extração silenciosamente errada.** Leitores de PDF que devolvem lixo
  quando a fonte é composta (Identity-H sem tratar o `ToUnicode`), que
  perdem páginas inteiras em *object streams* (PDF 1.5+), que leem texto
  cifrado de um PDF criptografado como se fosse texto.
- **Arquivos hostis.** Uma *zip bomb* de poucos KB que infla para GB; um
  cabeçalho que mente o tamanho; nomes de membro com `../`.
- **Proveniência rasa.** O trecho recuperado diz "doc 17", não
  "`relatorio.zip ▸ anexos/contrato.pdf`, página 3, arquivo cujo SHA-256 é …".
- **Formato pelo nome.** `.pdf` que é HTML, `.docx` que é zip qualquer.
- **Dependências pesadas** (Tika, poppler, LibreOffice) só para tirar texto.

## 2. A eclusa de documentos (`Vapor.Docs`)

O mesmo princípio da eclusa de modelos: um leitor **reivindica o arquivo
pelos bytes** (números mágicos e estrutura do contêiner), nunca pelo nome, e o
transforma em passagens com proveniência:

```
%{doc: "relatorio.zip!/anexos/contrato.pdf#p3", text: "…", kind: :pdf, meta: %{page: 3}}
```

`!/` entra num arquivo compactado, `#p3` é uma página, `#slide2` um slide,
`#sheet:Nome` uma planilha. Tudo sem dependência:

| leitor | formatos | como |
|---|---|---|
| `Docs.Zip` | zip recursivo; e por ele EPUB, DOCX/XLSX/PPTX, ODT/ODS/ODP | diretório central lido aqui; cada membro inflado em blocos com teto; CRC e tamanho conferidos; sabor pelo conteúdo |
| `Docs.PDF` | PDF 1.0–2.0, texto por página | objetos por varredura (resiste a xref quebrada), *object streams*, Flate/ASCIIHex/ASCII85, árvore de páginas com recursos herdados, `ToUnicode` (bfchar/bfrange), WinAnsi com `/Differences`, `Tj`/`TJ`/`'`/`"` com linhas e espaços por posicionamento, imagens *inline* puladas |
| `Docs.Office` | Word, Excel, PowerPoint, OpenDocument | XML tokenizado sem expansão de entidade; tabelas com tabulações; planilhas pelo nome, em ordem; slides na ordem da apresentação; fórmula sem valor guardado aparece como fórmula (não é avaliada) |
| `Docs.Markup` | HTML/XHTML, XML | título primeiro, blocos em linhas, `script`/`style` descartados, entidades |
| `Docs.Pictures` | PNG (decodificado inteiro: todos os tipos de cor, profundidades 1–16, Adam7; `tEXt`/`zTXt`/`iTXt`), PPM/PGM; JPEG/GIF/WebP só tamanho e texto embutido (EXIF, comentários) | inflação com teto pelo tamanho declarado; CRCs conferidos |
| texto | UTF-8, Markdown, CSV, JSON; Latin-1 transcodificado | — |

**O que não é texto é dito, não adivinhado.** Página sem camada de texto
("escaneada? não há OCR aqui"), PDF criptografado (recusado: as strings são
cifradas), pixels de JPEG (não há decodificador), binário — tudo vira aviso,
nunca lixo no índice.

**Limites** (opções): 64 MiB por arquivo expandido, 256 MiB por ingestão,
10 000 membros, 4 níveis de aninhamento, **razão de compressão ≤ 200×** para
membros acima de 1 MiB. Tamanhos e razões declarados são checados *antes* de
inflar; a inflação para no tamanho declarado (cabeçalho mentiroso = recusa).
Nomes de membro nunca viram caminho no disco.

## 3. Verificado contra produtores e oráculos que não são meus

Os fixtures (`test/fixtures/docs/`, regenerados por
`test/python/make_doc_fixtures.py`) são escritos por **fpdf2, Ghostscript,
qpdf, python-docx, openpyxl, python-pptx, Pillow e pypng**:

- PDF com fonte padrão (WinAnsi), com TrueType embutida (Type0/Identity-H +
  `ToUnicode`, incluindo grego e setas), o mesmo com *object streams* e
  xref stream (qpdf), linearizado, gerado pelo Ghostscript (Type 1
  re-codificada em ISOLatin1), criptografado (AES-256) e "escaneado";
- **texto idêntico ao `pdftotext` do poppler** (normalizado em espaços) em
  todos os PDFs com texto;
- **pixels idênticos aos do Pillow** em nove PNGs (RGB, RGBA, paleta, cinza
  1/4/8/16 bits, 16 bits RGB, entrelaçado Adam7 em tamanho ímpar) — os de 16
  bits dentro do truncamento de 8 bits que o próprio Pillow faz;
- zip-bomb de 64 MiB recusada pela razão antes de inflar; um membro que
  declara 10 bytes e infla 100 000 recusado durante a inflação.

## 4. A biblioteca (`Vapor.Docs.Library`)

As passagens alimentam o `Vapor.RAG` existente (raiz Merkle, BM25 ou
híbrido, provas de inclusão, recibos), e a biblioteca acrescenta:

- **uma raiz** = digest canônico (raiz do corpus de texto, manifesto de todo
  arquivo encontrado com seu SHA-256). Um recibo de busca prende consulta,
  resultados e essa raiz; `verify/2` recalcula tudo. Uma resposta prova de
  que **arquivos** veio, não só de que trechos;
- **por resultado**: o arquivo, o contêiner mais externo e os SHA-256 dos dois;
- **índice visual** para imagens decodificadas: miniatura 8×8 por canal +
  histograma de cor, centrado e normalizado, comparado por uma contração
  certificada — acha imagens *visualmente* parecidas (cor e disposição).
  **Não é busca semântica**: não há CLIP aqui; o lugar dele é o adaptador de
  encoder + um projetor ([ANY_TO_ANY.md](ANY_TO_ANY.md));
- o mesmo arquivo (por SHA-256) adicionado duas vezes não muda nada.

```sh
mix vapor.rag index minha.vlib ./pasta ./arquivos.zip    # recursivo
mix vapor.rag search minha.vlib "o que diz o contrato sobre multa" --k 5
mix vapor.rag search minha.vlib "…" --json               # com provas e recibo
mix vapor.rag image minha.vlib foto.png
mix vapor.rag show minha.vlib
```

A biblioteca é gravada no formato externo de termos do BEAM e lida de volta
com `:safe`.

## 5. Limites honestos

- **OCR** (desde 0.5.0, [OCR.md](OCR.md)): texto impresso horizontal em
  páginas escaneadas e imagens, **em colunas, na ordem de leitura**, com
  modelo de língua (0.7.0), e **tabelas** com réguas ou filetes célula a
  célula (0.8.0, cada tabela uma passagem com HTML e CSV); manuscrito e
  perspectiva não. Imagens **CCITT** (Group 3/4, o formato dos *scanners*
  P&B) são decodificadas desde 0.7.0, bit a bit iguais ao libtiff, e
  **JBIG2** aritmético desde 0.8.0, bit a bit igual ao jbig2dec (Huffman e
  meio-tom: aviso).
- **JPEG** é decodificado (= libjpeg bit a bit); **GIF/WebP** não: indexados
  pelo texto embutido.
- **PDF**: o texto segue a ordem do fluxo de conteúdo, não a leitura visual
  (colunas podem se intercalar); fontes compostas sem `ToUnicode` são
  avisadas, não decifradas. Filtros: Flate, LZW (`EarlyChange` 0/1),
  RunLength, ASCIIHex, ASCII85, com os preditores PNG e TIFF; DCT e CCITT
  e JBIG2 (aritmético) para imagens; JPX e JBIG2 Huffman recusados com aviso.
- **Excel**: fórmulas sem valor guardado não são avaliadas.
- **Escala**: tudo roda na BEAM; o PNG para de decodificar pixels acima de
  4 milhões (só texto e tamanho), e o índice é reconstruído a cada arquivo —
  bom para milhares de páginas, não para milhões.
- **Busca de imagem**: por similaridade visual, e por significado quando a
  imagem carrega texto (OCR); fotos sem texto exigem o CLIP — as duas torres
  estão conferidas desde 0.6.0, mas ligá-las à biblioteca só tem sentido com
  pesos treinados, que não são embarcados (TODO).
