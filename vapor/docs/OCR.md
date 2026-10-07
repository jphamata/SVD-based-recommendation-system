# Visão: OCR por um modelo admitido na eclusa, JPEG bit a bit, e o escaneado de escritório

A 0.4.0 declarou: *não há OCR nem decodificador de JPEG; a busca por imagem é
por semelhança visual, não por significado.* A 0.6.0 ainda declarava: *sem
ordem de colunas, CTC guloso sem modelo de língua, PDFs CCITT/JBIG2 recusados.*
Esta página diz o que foi feito, como foi medido e o que continua de fora;
§3b–§3d são da 0.7.0.

## 1. JPEG: o decodificador do libjpeg, bit a bit, sem dependência

`Vapor.Docs.JPEG` decodifica JPEG *baseline*, sequencial estendido e
**progressivo** (com aproximação sucessiva), qualquer amostragem (4:4:4, 4:2:2,
4:2:0, 4:4:0, 4:1:1), intervalos de *restart*, tons de cinza, YCbCr e RGB da
Adobe. Não aproxima o libjpeg: **é** a aritmética que ele especifica —

- a IDCT inteira exata (`jidctint.c`, `JDCT_ISLOW`: constantes de 13 bits, duas
  passadas, a tabela de limite com a dobra de 10 bits);
- o *upsampling* "fancy" (`jdsample.c`: o filtro triangular de h2v1, h1v2 e h2v2
  com os vieses de arredondamento e as regras de borda do libjpeg, linhas de
  contexto replicadas no topo e no fundo);
- YCbCr → RGB pelas tabelas de ponto fixo de 16 bits (`jdcolor.c`).

Resultado: **os mesmos pixels que o Pillow (libjpeg-turbo) em 51 de 51
arquivos** — as duas fotos reais do scikit-learn como vêm e recodificadas, uma
imagem sintética de bordas duras, todas as amostragens, progressivo,
*restart*, tabelas otimizadas, qualidade 30 a 100, tamanhos ímpares
(`test/vapor/jpeg_test.exs`, `test/python/jpeg_fixtures.py`). Uma foto de
640×427 decodifica em ~0,5 s na BEAM. Recusados com motivo: codificação
aritmética, JPEG sem perdas e hierárquico, 12 bits, CMYK/YCCK.

Consequências: imagens JPEG entram no índice visual da biblioteca, ganham
miniatura no console, e **páginas escaneadas de PDF** (que quase sempre são
JPEG, `DCTDecode`) ficam legíveis pelo OCR.

## 2. OCR: uma linha impressa é uma sequência de colunas

O OCR clássico corta a linha em caracteres e classifica cada um. Isso falha
exatamente onde o texto real é difícil: letras que se tocam (serifas, *kerning*,
baixa resolução, borrão), ligaduras. Medimos isso aqui: com segmentação por
componentes conexos, **54 %** das linhas das fontes de teste não tinham o número
certo de caracteres — "ex", "ti", "rn" viravam um bloco só.

Pensamento lateral: **ler uma linha é o mesmo problema que ouvir uma fala.** A
linha vira uma sequência de quadros (janelas de 8 colunas a cada 2), um encoder
bidirecional classifica cada quadro, e o **CTC** (classificação temporal
conexionista) colapsa os quadros em caracteres. Não há segmentação de
caracteres: letras coladas e ligaduras são problema do modelo, não da
geometria.

```
imagem ─ Segment: tinta (Sauvola), componentes, linhas ─▶ bitmap da linha (32 px, linha de base na linha 22)
       ─ quadros 32×8 a cada 2 colunas ─▶ vapor_encoder (head: rows) no substrato ─▶ CTC guloso ─▶ texto
```

**Nenhum operador novo.** O leitor é um `vapor_encoder` — a mesma topologia que
lê patches de imagem — com uma cabeça por linha (`head: "rows"`, acrescentada
nesta rodada: um `linear` sobre todas as linhas em vez da linha 0). Janelas
sobrepostas são só uma cópia de linhas (uma convolução com passo menor que o
núcleo é um *gather* de janelas). O checkpoint (`priv/ocr`: `config.json` com o
alfabeto em `labels`, `model.safetensors`) entra pela eclusa como qualquer
modelo; outro leitor do mesmo contrato — outro alfabeto, outro idioma — entra
igual.

**A geometria** (`Vapor.Vision.Segment`, sem modelo):

1. **Tinta** pelo limiar adaptativo de Sauvola sobre imagens integrais — uma
   página fotografada com luz desigual binariza como um *scan* limpo.
2. **Componentes** 8-conexos (rótulos em `:atomics`).
3. **Linhas** pela projeção do **miolo** vertical de cada componente (a metade
   do meio): ascendentes e descendentes de linhas vizinhas não as emendam.
4. **Bitmap da linha** só com a tinta dela (o descendente da linha de cima não
   entra), escalado para que a mediana das alturas dos glifos fique em 12 px e
   a linha de base na linha 22 de 32.

**Treino** (`test/python/ocr_render.py`, `test/python/train_ocr.py`): 31 434 linhas
renderizadas pelo FreeType em 18 fontes (DejaVu, Free, Liberation, Nimbus,
Inter, Caladea), 14–44 px, degradadas como *scan* ou foto (borrão, ruído de
sensor, gradiente de luz, JPEG, leve rotação), texto dos corpora de referência
e cadeias aleatórias sobre todo o alfabeto (ASCII imprimível e os acentos do
português). Os bitmaps de treino são **os que o vapor calcula**
(`Vapor.Vision.OCR.dataset/2`): treino e inferência veem a mesma coisa.

## 3. Medido

Fontes **nunca vistas no treino** (C059, P052, Carlito, URW Gothic, URW Bookman)
e texto dos corpora **retidos**; uma **foto real** de página impressa
(`skimage.data.page()`, luz desigual); o Tesseract 5.3.4 (LSTM, inglês) nas mesmas
imagens, para referência — leituras congeladas em `priv/quality/ocr/tesseract.json`
por `test/python/ocr_tesseract.py`: o produto não executa ferramenta externa
(`audit_test.exs`), e a comparação se reproduz sem o Tesseract instalado. Números regeneráveis por `mix vapor.quality`
([bench/QUALITY.md §4c](bench/QUALITY.md)) e por `mix vapor.ocr eval`.

| conjunto | vapor CER | vapor WER | Tesseract CER | Tesseract WER |
|---|---|---|---|---|
| 5 fontes fora do treino, 40 linhas | **6,8 %** | 25,6 % | 5,2 % | 20,6 % |
| — C059 | 2,3 % | 11,9 % | 2,3 % | 7,1 % |
| — P052 | 1,9 % | 8,3 % | 3,8 % | 16,7 % |
| — Carlito | 1,7 % | 9,1 % | 9,9 % | 33,3 % |
| — URW Bookman | 6,2 % | 31,1 % | 4,1 % | 20,0 % |
| — URW Gothic | 17,8 % | 58,8 % | 7,4 % | 27,5 % |
| foto real de página (6 linhas) | **11,7 %** | 32,6 % | 36,4 % | 39,5 % |

**Desde 0.7.0** (o mesmo leitor; feixe com modelo de língua, §3b, e o alcance
dos acentos corrigido, §3c): **4,4 %** de CER nas 40 linhas (Tesseract 5,2 %) —
C059 1,5 %, Carlito 1,2 %, P052 1,9 %, URW Bookman 2,4 %, URW Gothic 12,5 % —
e **9,5 %** na foto real (Tesseract 36,4 %). A tabela acima é a da 0.5.0, para
comparação.

Lido com honestidade: um leitor de 2,6 MB, treinado em 24 minutos numa CPU de
2 núcleos, fica **perto do Tesseract** em fontes que nunca viu (melhor em três
das cinco, pior na geométrica URW Gothic, cujo `a` de um andar e `t` sem
curva ele não conhece) e **lê melhor a foto de luz desigual**, onde o Tesseract
perde o início das linhas na sombra. A WER é alta para os dois porque um único
caractere errado derruba a palavra inteira; um feixe com modelo de língua
(TODO) é o próximo ganho. Validação durante o treino (592 outras linhas nas mesmas cinco fontes,
só acompanhada — o checkpoint é o do último passo, não o melhor na validação):
CER 7,2 %.

O controle "texto fluente errado" (cada linha lida como a verdade da linha
anterior) mostra que a medida separa *parecer texto* de *ser o texto certo*.

## 3b. Modelo de língua: escolher entre o que a página pode dizer, nunca contra ela (0.7.0)

O CTC guloso decide cada quadro sozinho: "c1áusula", "rão" por "não",
"agerdar". Um **modelo de língua de caracteres** (`Vapor.Vision.CharLM`:
5-gramas com suavização de Witten–Bell interpolada — nenhuma probabilidade
zero, nenhum desconto a ajustar) entra numa **busca em feixe CTC por
prefixos** (`OCR.ctc_beam/3`; Graves 2006, Hannun et al. 2014): estender um
prefixo por `c` soma `log P_quadros + 0,8·log P_LM(c | prefixo) + 3,0`.

O modelo *é* o seu corpus: contado no carregamento a partir dos corpora de
**referência** (o texto em que o leitor foi treinado), nunca do texto retido
em que é medido; `priv/ocr/lm.json` diz quais, a ordem e os pesos.

**O escrutínio achou três maneiras de o modelo de língua virar ruído — cada
uma virou uma regra testada:**

1. **Reescrever o que não é língua.** Sem guarda, cadeias aleatórias pioraram
   (CER 16,0 % → 22,5 %): o modelo empurrava a leitura para o que é comum.
   Regra: **o modelo se abstém** quando a leitura gulosa da linha custa mais de
   8 bits/caractere sob ele (prosa retida ≈ 2,6; cadeias aleatórias ≈ 12) — o
   limiar foi escolhido num conjunto aleatório de **validação** separado.
   Controle permanente na suíte: com a guarda, 0 de 30 linhas aleatórias
   mudam; sem ela, 27.
2. **Apagar uma letra lida com certeza.** Um teste sintético pegou: "á" lida
   a 0,97 era apagada porque "clá" é raro no corpus. Regra: **a cada quadro
   só se oferece ao modelo o que os quadros acham plausível** (probabilidade ≥
   10⁻³, o *blank* inclusive) — uma letra certa não pode ser apagada, um
   espaço certo não vira letra.
3. **Trazer o domínio do corpus.** Os corpora são Markdown; um modelo contado
   com crases e asteriscos **escreveu crases em páginas escaneadas que não
   tinham nenhuma**. Regra: o modelo é contado sobre o texto **como impresso**
   (`strip` em `lm.json`). Custo, dito: no conjunto de linhas retidas — que
   foi renderizado a partir do Markdown e contém crases — o ganho cai de
   −51 % para −32 % de CER; nas páginas escaneadas, que não têm marcação, a
   leitura melhora e nenhuma crase é inventada.

Pesos (0,8 e 3,0), feixe (8) e ordem (5) foram escolhidos num conjunto de
**validação** (160 linhas, outra semente, mesmas fontes de teste), nunca no
de teste. Medido ([bench/QUALITY.md §5c](bench/QUALITY.md)):

| conjunto | guloso | feixe + LM | controle |
|---|---|---|---|
| 40 linhas retidas (fontes fora do treino), CER | 6,5 % | **4,4 %** | LM de corpus embaralhado: 7,0 % (nenhum ganho: o ganho é da língua) |
| 30 linhas com valores, datas, códigos (R$ 85.691,34, AZ-69511, #28129), CER | 8,9 % | **7,4 %** | — (o modelo não pode piorar) |
| 30 linhas de cadeias aleatórias: linhas mudadas | — | **0** | sem a guarda: 27 |

Na interface, cada caractere que o modelo escolheu fica destacado, e um botão
mostra a leitura só dos quadros — inclusive quando o modelo erra (há um
"retomam" → "retoman" numa das páginas de teste: está lá para ser visto).

## 3c. Ordem de leitura: colunas (0.7.0)

Antes, as linhas eram faixas horizontais da página inteira: numa página de
duas colunas cada "linha" juntava a linha da esquerda com a da direita —
**CER 73 %** numa página que, lida na ordem, dá 1,3 %. `Segment.blocks/2` corta
a página por um **XY-cut recursivo** sobre os componentes, com uma regra de
primeiros princípios: **uma calha de coluna (uma faixa vertical vazia que
atravessa a região inteira) corta antes de um vão horizontal**. Assim um
título ou um rodapé que atravessa as colunas é separado primeiro (ele
bloqueia a calha), e depois as colunas de cada faixa são lidas uma após a
outra. Os limiares são relativos à altura do texto **da própria região** (um
título maior tem espaços entre palavras maiores — o primeiro corte da
interface separava o título ao meio, e foi assim que achamos isso), e um vão
horizontal só corta se for maior que 1,5× o vão típico da região (um texto
em espaço duplo não vira um bloco por linha).

**Páginas escaneadas** (`priv/quality/scans`, `test/python/scan_pages.py`): 8
páginas de 1, 2 e 3 colunas, uma com título e rodapé atravessando as colunas,
em fontes nunca vistas pelo leitor, a 200 dpi, binarizadas como um *scan* P&B,
com poeira, **comprimidas em CCITT dentro de um PDF** (Group 4, Group 3 2-D,
máscara de estêncil, `BlackIs1`); texto retido em português e inglês e, fora
do domínio dos corpora, prosa jurídica (as licenças Apache-2.0 e MPL-2.0).

| | vapor | sem ordem de leitura (controle) | Tesseract 5.3.4 (eng, psm 3) |
|---|---|---|---|
| CER médio das 8 páginas, PDF → CCITT → blocos → leitor → LM | **1,3 %** | 53 % | 1,6 % |

Lido com honestidade: o Tesseract é melhor nas páginas em inglês com fontes
"comuns" (0,0–0,3 %), o vapor nas páginas em português (o Tesseract daqui só
tem o modelo `eng`) e a pior página do vapor continua sendo a URW Gothic
(7,5 %), a fonte geométrica que o leitor não conhece.

**Um bug real achado olhando a interface**: uma linha sem ascendentes nem
descendentes ("mesma execução não anunciam a mesma ação") era lida "mesmã
eeeução rão anuneiãm ã mesnã ãção": o achador de linhas aceitava um
componente a no máximo 3 px do miolo da linha, e o til e a cedilha ficam
mais longe que isso quando não há letras altas para alargar o miolo — os
acentos eram **descartados**, e o leitor via outra palavra. Agora o alcance é
0,6× a altura do texto; a linha lê certo (teste), e o CER guloso das linhas
retidas caiu de 6,8 % para 6,5 % e o das cadeias aleatórias de 16,0 % para
10,7 % só por isso.

## 3d. CCITT: o formato dos escaneados de escritório (0.7.0)

Quase todo PDF de *scanner* em preto e branco guarda a página em
`/CCITTFaxDecode`. `Vapor.Docs.CCITT` decodifica, sem dependência, as três
codificações (Group 3 1-D, Group 3 2-D, Group 4; T.4/T.6), com `EndOfLine`,
`EncodedByteAlign`, `EndOfBlock`, `BlackIs1`, ressincronização por EOL após
dano — a mesma máquina de estados do Xpdf/pdf.js. **Conferido bit a bit contra
o codificador do libtiff** (pelo Pillow): 42 fluxos — ruído em três densidades,
uma página de texto, corridas de todo comprimento até 2 600 (códigos de
*make-up* estendidos), as três codificações com e sem *fill bits*, o MH
alinhado do TIFF — e, no mesmo arquivo de testes, LZW (`EarlyChange` 0 e 1) e
PackBits (`RunLengthDecode`). Uma página de 1 700 × 1 000 decodifica em ~1 s
na BEAM. Dados lixo e truncados: nunca travam, devolvem as linhas que havia.

E outro bug real: **máscaras de estêncil (`ImageMask`) eram lidas
invertidas** (amostra 0 é tinta; o código invertia). Não havia teste; agora
há, para Flate e CCITT.

JBIG2 (o outro formato de escaneados, com dicionários de símbolos e
codificação aritmética): decodificado desde 0.8.0 (§3f).

## 3e. Tabelas: estrutura, células, tipos de coluna (0.8.0)

Até 0.7 uma tabela era lida pelo XY-cut como colunas de texto: "Código
Descrição T1 T2 DQ-20430 lado…", sem linhas nem células. `Vapor.Vision.Table`
lê a **estrutura** antes do texto, e cada célula depois:

1. **Segmentos de régua** saem dos componentes com forma de régua (longos e
   finos, ou trechos que a régua quebrou). Uma **grade com réguas** junta os
   segmentos horizontais e verticais em linhas e colunas (bordas virtuais onde
   a régua externa falta) e decide cada **célula mesclada** pela cobertura das
   réguas: uma divisória ausente no meio de duas células as funde (o
   "Trimestre" sobre T1 e T2, um rótulo que ocupa duas linhas).
2. **Tabelas só de filetes** (*booktabs*: régua no topo, sob o cabeçalho e no
   fim, nenhuma vertical) têm as colunas achadas pelas **calhas** — vãos
   verticais em todas as linhas do corpo, mais largos que um espaço entre
   palavras; ruído de digitalização não abre calha.
3. **As células são lidas** pelo mesmo leitor, com três passos tipados por
   coluna: (a) uma coluna é numérica quando ≥ 70 % das células podem ser lidas
   só com dígitos e os símbolos que a própria coluna compartilha (`R`, `$`,
   `.`, `,`, `%`…) a um custo de até 2 nats por caractere sobre o melhor
   caminho livre — e então "5" lido "õ" volta a ser o melhor dígito; (b) o
   espaço entre dígitos é a convenção da coluna ("1 234,56") ou ruído, e
   decide a hipótese cujas células mais concordam numa forma; (c) quando a
   maioria das células tem uma **forma** (`AA-99999`, `99/99/9999`,
   `R$ 99.999,99` — `Vapor.Vision.Template`), cada célula é decodificada de
   novo *dentro* da forma (Viterbi CTC sobre o autômato da forma) e o
   resultado vale se custar até 1,5 nat a mais. Os limiares foram escolhidos
   num **conjunto de validação separado** (semente 2028) e medidos no de teste
   (semente 2027).

A saída é estrutura (`rows`, `cols`, `header_rows`, células com `row`, `col`,
`rowspan`, `colspan`, caixa, texto e confiança) e três renderizações: Markdown
(GFM; cabeçalho de várias linhas achatado "Grupo / Sub"), HTML (com `rowspan`,
`colspan`, `<thead>`) e CSV. No texto do OCR a tabela entra no seu lugar na
ordem de leitura, como Markdown; na biblioteca de documentos cada tabela é uma
passagem própria (`#table1`, com HTML e CSV em `meta.table`); no console, a
tabela aparece desenhada a partir das células — mescladas, cabeçalho,
números alinhados à direita, cada célula com a sua confiança e ligada à sua
caixa na página.

**Medido** (`mix vapor.quality` §5d; 12 tabelas escaneadas em 4 estilos —
grade, grade interna, *booktabs*, filetes —, fontes fora do treino, ruído de
*scanner*; `test/python/table_render.py`):

| medida | vapor 0.8 | controle |
|---|---:|---|
| estrutura (F1 de adjacência ICDAR 2013) | **1,000** | 0,343 (a leitura da 0.7: linhas em ordem) |
| grades com células mescladas | **1,000** | 0,905 (detecção de mesclas desligada) |
| CER por célula | **5,5 %** | 11,7 % (leitura livre das mesmas células) |
| CER por célula, validação | 6,0 % | 12,5 % (livre) |
| Tesseract, células **recortadas à mão** (caixa perfeita) | 2,3 % | — |

O Tesseract, mesmo recebendo as caixas perfeitas que nenhum leitor real tem,
lê melhor as células: o leitor do vapor é pequeno e treinado em linhas de
prosa. A estrutura — o que o Tesseract não dá — é exata nas 12. **Um erro
visível**: tokens curtos isolados de cabeçalho ("T1") às vezes saem errados
("TP1") e nem sempre com confiança baixa — os passos tipados só valem no
corpo.

Fora: tabelas sem nenhuma régua (só alinhamento), tabelas que atravessam
páginas, células com várias linhas de texto em tabelas só de filetes.

## 3f. JBIG2: o último formato de escaneados (0.8.0; Huffman e meio-tom na 0.15)

`Vapor.Docs.JBIG2` decodifica JBIG2 (ITU-T T.88) com **codificação
aritmética**, sem dependência: o decodificador MQ, os contextos IAx/IAID,
regiões genéricas (modelos 0–3, pixels adaptativos, TPGDON), MMR (pelo CCITT
da 0.7), refinamento (TPGRON), dicionários de símbolos (inclusive agregados por
refinamento), regiões de texto (os 8 cantos/transposições, faixas,
refinamento por instância), páginas com e sem faixas, o arquivo solto e o
modo embutido do PDF (`/JBIG2Decode` com `/JBIG2Globals`).

**Conferido bit a bit contra o jbig2dec** em 43 fluxos: os do jbig2enc
(genérico, símbolos, PDF com globais) e os de um codificador próprio em Python
(`test/python/jbig2_streams.py`: um codificador MQ portado do jbig2enc e
conferido contra o exemplo H.2 do padrão; cada fluxo é validado pelo jbig2dec
antes de virar fixture). Controle: os mesmos fluxos com o modelo genérico
declarado errado — 19 de 43 ainda "batem" (os de símbolo/MMR não usam o
modelo), os outros 24 não; um decodificador que ignorasse o modelo passaria
no teste ingênuo. Página inteira de 1 700 × 870 em ~0,5–1 s na BEAM. Uma
tabela escaneada num PDF JBIG2 é lida com a mesma estrutura e o mesmo texto
que a mesma página em PNG.

Duas divergências achadas entre implementações de referência, documentadas
no código: o jbig2dec usa a página inteira como referência de um refinamento
sem deslocamento (T.88 7.4.7.4 diz a região), e os contextos SLTP do TPGRON
do modelo 1 diferem entre jbig2dec e pdf.js (seguimos o jbig2dec, que é o
que se confere aqui).

**Huffman e meio-tom (0.15).** O que a 0.8 recusava por não haver codificador
aberto que os emitisse para conferir foi fechado do outro lado: um codificador
**independente** em Python (`test/python/jbig2_streams.py`, sem código em comum
com o decodificador) emite os fluxos, e cada um é julgado pelo jbig2dec antes de
virar fixture. Decodificados agora:

- **codificação Huffman** (`Vapor.Docs.JBIG2Huffman`): as quinze tabelas
  padrão B.1–B.15 (com as linhas de faixa inferior/superior e OOB), tabelas do
  usuário (segmento tipo 53, construídas pelo algoritmo B.3), dicionários de
  símbolos SDHUFF (alturas, larguras, tamanhos de agregação, e o bitmap coletivo
  de cada classe de altura cru ou MMR) e regiões de texto SBHUFF (a tabela de
  IDs de símbolo por comprimentos de código em *run-length*, faixas e
  coordenadas, com as tabelas do usuário na ordem FS, DS, DT, …);
- **dicionários de padrões e regiões de meio-tom** (tipos 16, 20, 22, 23): os
  planos de cinza em código de Gray (genéricos ou MMR), a grade rotacionada,
  `HSKIP` (células fora da região puladas), as quatro combinações e
  `HDEFPIXEL`.

21 fixtures novas (13 Huffman, 8 meio-tom), **64 no total**, todas bit a bit; as
39 antigas regeneram byte a byte. Um defeito do **jbig2dec** apareceu no caminho:
com `HDEFPIXEL = 1` ele preenche a região com o byte `0x01` (um pixel preto em
oito, listras verticais) em vez de preto. Essa fixture é julgada pelo T.88
6.6.5.2 passo 1, e a diferença (em pixels) fica no manifesto. Outras duas
armadilhas da especificação, respeitadas: a tabela B.2 tem larguras negativas e
a B.11 não tem `DT = 0`.

**Ainda recusados com aviso**, pelo nome: Huffman **com refinamento** (SDREFAGG
ou SBREFINE sob SDHUFF/SBHUFF), contextos aritméticos **retidos** entre
segmentos, e um meio-tom sem o seu dicionário de padrões.


## 3g. Chinês, japonês e coreano: milhares de classes sem rede treinada (0.10)

`Vapor.Vision.CJK`. Três fatos das escritas carregam o desenho:

1. **A escrita é uma grade.** Cada hanzi, kanji, kana ou sílaba hangul
   ocupa a mesma célula quadrada, da altura da linha. Um caractere de
   várias partes (好 = 女 + 子) não é segmentado pelos seus componentes,
   mas por **células**: os cortes candidatos ficam nos vãos de tinta, e um
   segmento é julgado numa célula da altura da linha centrada nele (só as
   colunas dele: um dígito estreito não vê o vizinho).
2. **Uma fonte é o próprio conjunto de treino.** Os modelos de classe são
   renderizados das normas nacionais (GB 2312 nível 1: 3 755 hanzi; JIS X
   0208 nível 1 + kana; KS X 1001: 2 350 sílabas) em várias fontes e
   resumidos por **features de elemento direcional** — o gradiente das
   bordas dos traços em 8 direções numa grade 8×8 (512 valores), a
   representação em que o OCR impresso de CJK se apoia desde os anos 1990,
   porque a direção de um traço sobrevive à troca de fonte muito melhor
   que os seus pixels. Ler é o modelo mais próximo pelo cosseno: um
   `linear` no worker contra a matriz inteira de classes.
3. **O reconhecimento decide a segmentação.** Cada segmentação da linha em
   células é pontuada pelo quão bem as células são reconhecidas, menos um
   preço fixo por célula; vence o melhor caminho (programação dinâmica
   sobre os vãos; corridas que se tocam são cortadas pelo passo da linha).
   Um modelo de língua por caractere (Witten–Bell) reordena só entre
   candidatos visualmente plausíveis, e **se abstém** quando a leitura
   gulosa já é improvável para ele (> 10 bits/caractere).

Pacotes embarcados (`priv/ocr-cjk-{zh,ja,ko}`, modelos em 8 bits: 1,2–1,9
MB cada; `CJK.default(:zh | :ja | :ko)`). Medido em linhas **renderizadas
com uma semente nova, em fontes que nunca entraram nos modelos**, com o
ruído do restante do OCR (borrão, ruído, iluminação, JPEG, rotação):

| língua (fontes do teste) | CER guloso | com o modelo de língua | caracteres aleatórios (controle) |
|---|---|---|---|
| chinês (Noto Serif SC, AR PL UKai) | **8,9 %** (Noto Serif 0,0 %; UKai 16,9 %) | 9,2 % | 9,6 % → 9,6 % |
| japonês (Noto Serif JP, Sawarabi Mincho) | **5,0 %** | **2,6 %** | 0,5 % → 0,5 % |
| coreano (Noto Serif KR, NanumBarunGothic) | **16,0 %** | **11,3 %** | 9,9 % → 9,9 % |

**O que o escrutínio achou:** no conjunto de desenvolvimento (a semente
com que as escolhas foram feitas), o modelo de língua levava o chinês de
7,8 % a 2,5 %; numa semente nova, **não ajuda** (8,9 % → 9,2 %). A melhora
de antes era em parte ajuste ao conjunto — por isso os números acima são
os da semente nova. O corpus do modelo de língua vem das mesmas listas de
palavras do Faker que as linhas de teste (outra semente): o ganho em
japonês e coreano é um número **do mesmo domínio**. O controle
(caracteres aleatórios) prova a outra metade da promessa: onde não há o
que explorar, o modelo de língua não muda nada.

Kai (pincel) é a fonte difícil: sem um Kai entre os modelos, o UKai ficava
em 46,6 %; com o AR PL KaitiM (mesma linhagem da UKai — dito aqui), 16,9 %.

## 3h. Figuras: onde estão, o que dizem as legendas, quais são os números (0.10)

`Vapor.Vision.Figure`, e no `OCR.read/2` (as marcas de uma figura não
poluem mais o texto; a legenda é texto).

- **Detecção.** Texto é feito de marcas de no máximo ~1,5 altura de texto;
  a moldura de um gráfico, uma curva, uma barra ou as manchas de uma
  fotografia, não. Uma marca de 6 alturas (nos dois sentidos) semeia uma
  figura; a figura cresce sobre o que a toca (rótulos de eixo, títulos,
  legenda), nunca sobre uma **linha de texto da página** que passa dos
  lados dela; uma fotografia é estendida até onde o papel recomeça. Uma
  caixa que contém linhas de texto é um parágrafo emoldurado, não figura.
- **Legenda**: a linha mais próxima abaixo (ou acima) que **diz** ser uma
  ("Figure 3", "Fig. 2.", "Figura 1 —", "Gráfico", "图 2", "図", "그림",
  "شكل"), lida de um recorte ampliado (legendas vêm em corpo menor).
- **Digitalização de gráficos**: a moldura (duas espinhas num canto), as
  marcas dos *ticks*, os rótulos lidos — e **a escala só é aceita se os
  rótulos concordam com uma**: um mapa linear ou logarítmico de pixel para
  valor ajustado a *todos* os rótulos; mais de um em desacordo e o eixo é
  **recusado**. Outra regra de primeiro princípio: *ticks* lineares caem
  em múltiplos do passo (0, 20, 40 — nunca 1, 21, 41); uma leitura que
  ajusta uma reta mas não essa regra é um dígito lido errado em todos os
  rótulos, e é recusada. Um eixo log com só dois rótulos (10¹, 10²) é
  conferido pelos *ticks* menores, que têm de cair em log₁₀(2…9).
  Rótulos pequenos são lidos de recortes ampliados, cada eixo também como
  **uma frase** (todos os rótulos lado a lado: um "4" sozinho é uma linha
  que o leitor nunca viu), e por **consenso dos glifos**: os rótulos de um
  eixo usam uma fonte só, então o mesmo dígito é o mesmo desenho — os
  glifos são agrupados por forma e cada grupo recebe o caractere que o
  leitor mais lhe dá; ponto e sinal de menos são decididos pela forma e
  pelo lugar. As séries são separadas pela **direção da cor a partir do
  branco** (a borda suavizada de uma linha é a sua cor misturada com o
  papel, `W − p = (1 − t)(W − c)`: a direção de `W − p` é a da cor), e
  classificadas pela forma das marcas: retângulos com base comum são
  barras (lacunas de grade fechadas), manchas compactas são pontos (manchas
  sobrepostas divididas por k-médias), o resto é uma linha lida coluna a
  coluna pelo centro ponderado pela tinta.

Medido em gráficos do matplotlib que nunca serviram para ajustar o
digitalizador:

| conjunto | dentro da tolerância | recusados | lidos grosseiramente errados |
|---|---|---|---|
| estilo padrão (30) | **27** (mediana do erro por coluna ≤ 1 % do intervalo do eixo; barras ≤ 2 %; pontos: revocação e precisão ≥ 0,9) | 2 | 0 |
| difícil: serifa, grade, eixo log, JPEG (30) | 18 | 12 | **0** |
| **controle: rótulos permutados** (12) | — | **12 de 12** | 0 |
| páginas (10): figura achada, IoU ≥ 0,9; legenda | 10/10, tipo certo (gráfico × foto) em todas; legendas 10/10, CER < 10 % | | |

**Escrutínio das legendas:** a primeira execução completa da suíte achou
2 legendas perdidas nas páginas de teste — uma **acima** da figura com
texto do corpo logo abaixo (só as duas linhas mais próximas *abaixo* eram
olhadas), outra depois de um título de eixo que ficou fora da caixa; e um
"8" lido "B". Agora são olhadas as quatro linhas mais próximas dos dois
lados, numa faixa de 8 alturas de texto, e uma legenda pode ser numerada
por letra ("Figure A:", de um apêndice; maiúscula seguida de pontuação). As
10/10 acima vêm **depois** dessa correção: para as legendas, este conjunto
deixou de ser cego (dito aqui).

O conjunto difícil mostra o limite — dígitos serifados de 8 px que o
leitor não separa —, e mostra a propriedade que importa: **o que não é
lido é recusado, nunca inventado**.

## 3i. Fórmulas tipografadas → LaTeX (0.10)

`Vapor.Vision.Math`. O que cada marca é, e onde ela está, separados:
símbolos pelo modelo mais próximo (features de borda em 4 direções e
cobertura numa célula quadrada, mais o log da proporção comparado à
parte, para `-`, `=`, `)` e `∫` não se confundirem); **estrutura pela
geometria** — a barra mais larga primeiro (numerador acima dela, no seu
vão; denominador abaixo), radicais (a marca oca que envolve outras), os
limites de um `∑` (a linha acima e a abaixo, que pode passar dos lados
dele), e o resto da esquerda para a direita, cada marca na linha da base
anterior ou elevada (expoente), rebaixada (índice) ou as duas, pelo
centro em relação ao **corpo** da base (sem a descendente de um y, μ, β,
nem a ascendente de um d, λ ou dígito). Grafia canônica (`x^{2}`,
`a_{i}^{n}`, `\frac{a}{b}`, `\sqrt{x}`, `\sum_{i=1}^{n}`).

Modelos de 6 famílias de tipos (DejaVu Sans/Serif, STIX Sans e conjuntos
com Liberation Serif, FreeSerif, Noto Serif); **medido em Computer Modern
e STIX, que nunca entraram** (60 fórmulas de uma semente nova):
**4,9 % de erro por *token*, 36/60 idênticas**; o controle — os mesmos
símbolos lidos sem a geometria — erra mais de 3×. Confusões restantes:
n/π e a/α do Computer Modern, 5/6. Fora da gramática (matrizes,
acentos, `\left…\right`, várias linhas): fora do leitor.

## 3j. Árabe, cirílico — e o que "manuscrito" pode e não pode significar aqui (0.10)

**Árabe** (`OCR.default(:arabic)`, `priv/ocr-arabic`): o mesmo leitor CTC,
treinado com 59 fontes de texto (todas as do sistema, menos as quatro de
teste, os Nastaliq e as decorativas), rótulos na **ordem visual** — a dos
quadros que um leitor de colunas vê — e devolvidos à ordem lógica por
`Vapor.Vision.Bidi.logical/1` (inverte a linha, re-inverte as corridas
da esquerda para a direita — números com seus separadores, palavras
latinas — e desfaz o espelhamento de parênteses): igual ao `python-bidi`
em **todas as 600 linhas** de teste. As colunas de uma página RTL são lidas
da direita para a esquerda. Os pontos das letras (ب ت ث ن ي) formavam
"linhas" próprias e uma em cada quatro linhas árabes saía partida; o
localizador de linhas agora junta marcas baixas sobre as colunas de uma
vizinha alta (o latim não muda: a suíte da 0.7 continua passando).

| | CER |
|---|---|
| 4 fontes nunca vistas (Scheherazade 7,0 %, KacstNaskh 9,2 %, ae_Cortoba 16,1 %, ae_Granada 39,4 %) | **19,2 %** |
| primeira versão, 17 fontes de treino | 25,7 % |
| controle: o leitor latino nas mesmas linhas | 91 % |
| Nastaliq (caligrafia persa/urdu, nunca no treino) | 57 % |

As fontes Arabeyes de teste têm linhagem comum com as de treino (dito
aqui). Sem modelo de língua para o árabe ainda.

**Cirílico** (`priv/ocr-cyrillic`): ver §3k, abaixo.

**Manuscrito.** Nenhum conjunto de letra de mão real cabe nesta máquina
(sem rede para IAM, KHATT, CASIA-HWDB, os conjuntos cirílicos); o
substituto foram **fontes manuscritas**. Treinado com 17 "mãos", o leitor
de cursiva latina leu as 4 mãos de teste com **64 % de CER** (o leitor de
impressos: 76 %). Pela regra da suíte, isso é ruído — e **não é
embarcado**: `OCR.default(:cursive)` recusa, dizendo a medida e o
caminho (`test/python/train_ocr.py` aceita qualquer conjunto de linhas
reais; o contrato do leitor não muda). O mesmo vale para o "manuscrito"
árabe (Nastaliq: 57 %), cirílico (78 %) e japonês (§3k): medidos e
reportados, não prometidos.

## 3k. Cirílico, e os "manuscritos" medidos (0.10)

**Cirílico** (`OCR.default(:cyrillic)`, `priv/ocr-cyrillic`): o mesmo
leitor CTC (91 classes: o alfabeto russo com Ё, dígitos, pontuação com « »,
— e №), treinado 5 000 passos em 11 989 linhas de 30 fontes, com as
palavras de **uma metade** do vocabulário (Faker `ru_RU`); o teste usa a
outra metade e quatro fontes que nunca entraram (Lora, Carlito, URW
Bookman Light, Noto Sans Display):

| | CER |
|---|---|
| 4 fontes nunca vistas, 60 linhas (Noto Sans Display 1,5 %, URW Bookman 2,2 %, Carlito 3,6 %, Lora 4,1 %) | **2,8 %** |
| controle: o leitor latino nas mesmas linhas | 98 % |

O cirílico é a escrita mais fácil das novas: alfabética, da esquerda para a
direita, sem contextuais — o mesmo problema do latim com outras classes. O
treino de 5 000 passos levou ~2 h nesta máquina de 2 núcleos (dividida com
a suíte). Na página do console, a primeira letra de uma linha colada à
borda do recorte às vezes some (o recorte de teste tem margem): dito aqui.

**Os "manuscritos" pedidos, medidos com o único substituto possível aqui**
(fontes caligráficas; nenhuma letra de mão real cabe nesta máquina) — e
nenhum prometido:

| escrita | substituto (nunca no treino) | CER | o que isso diz |
|---|---|---|---|
| latim cursivo | 4 fontes manuscritas, leitor treinado em 17 outras | 64 % (impresso: 76 %) | ruído → `OCR.default(:cursive)` recusa |
| árabe | Nastaliq (IranNastaliq, Noto Nastaliq Urdu) | 57 % | outra caligrafia: a linha de base desce em diagonal |
| cirílico | SteveHand 60 %; Klee One e SetoFont (fontes japonesas com cirílico) 94 % | 78 % | ruído: nenhum leitor de mão |
| japonês | Klee One, SetoFont (pincel/caneta) | **9,8 % → 6,4 %** com o modelo de língua | um quadrado de traços sobrevive à mão; ainda é fonte, não mão |

O japonês "manuscrito" é o único número útil — e ainda é de **fontes**:
letra de mão real (ETL, CASIA-HWDB, KHATT, IAM) varia de pessoa para
pessoa, o que nenhuma fonte reproduz. O caminho é o mesmo para todas: um
conjunto de linhas reais, `test/python/train_ocr.py`, e o leitor resultante
entra pela mesma eclusa e é medido pela mesma suíte.

**LaTeX** (pedido junto): §3i — fórmulas tipografadas de uma linha.

## 4. Onde o OCR entra

- **Documentos**: páginas de PDF sem camada de texto são lidas (imagens
  `DCTDecode`, `FlateDecode` com preditores PNG, 1 bit); a passagem diz
  `meta.ocr` com a confiança. Imagens soltas (PNG, JPEG) ganham uma passagem
  `#ocr` quando o leitor tem confiança — **é assim que a busca encontra uma
  imagem pelo que está escrito nela**: busca por significado para a grande
  classe de imagens que carregam texto (escaneados, prints, slides, fotos de
  quadro).
- **Console**, painel *Visão*: a imagem (ou a página do PDF escaneado) com as
  linhas marcadas, os **blocos numerados na ordem de leitura** e o **fio de
  leitura** que vai do fim de cada linha ao começo da seguinte; o texto ao
  lado, agrupado por bloco, com cada caractere tingido pela confiança
  (pontilhado abaixo de 80 %, ondulado abaixo de 50 %) e os caracteres que o
  **modelo de língua escolheu** destacados — com um botão para ver a leitura
  só dos quadros.
- **CLI**: `mix vapor.ocr read arquivo.{png,jpg,pdf}`, `mix vapor.ocr eval DIR
  --tesseract`, `mix vapor.ocr page imagem.png verdade.json`.

## 5. Busca de imagem por significado: o que existe e o que falta

- Texto dentro da imagem → OCR → índice de texto: **feito**.
- Torre de visão do **CLIP** (`clip_vision_model`, com `visual_projection`) pela
  eclusa, **conferida contra o `transformers`** (pesos aleatórios, ≤ 3·10⁻⁷).
  Com um checkpoint CLIP do usuário, o vapor calcula o embedding de imagem
  certificado.
- Torre de **texto** do CLIP e o tokenizador BPE com `</w>`: **não feitos** — sem
  eles não há consulta em texto contra embeddings de imagem. É o próximo passo
  declarado no TODO. Sem pesos pré-treinados no ambiente desta rodada, nenhum
  número de busca semântica de fotos sem texto foi medido.

## 6. Limites

- Texto impresso horizontal. Manuscrito e texto em perspectiva ou curvo
  (`skimage.data.text()`): fora. Colunas, título e rodapé (§3c) e tabelas com
  réguas ou filetes (§3e): feitos; tabelas sem nenhuma régua, não.
- Alfabeto latino com acentos do português; outros alfabetos exigem outro
  leitor (o contrato é o mesmo) — e outro corpus para o modelo de língua.
- O modelo de língua é pequeno (≈ 27 mil palavras de documentação técnica):
  ajuda também em prosa jurídica fora do domínio (medido), mas um corpus do
  domínio do usuário ajudaria mais; `lm.json` aceita outros corpora.
- O leitor roda no worker nativo (uma linha por execução, ~50–150 ms); no
  oráculo é exato porém lento.
- PDF escaneado: CCITT, LZW, RunLength (§3d) e JBIG2 aritmético, Huffman e
  meio-tom (§3f) decodificados; JBIG2 Huffman com refinamento e JPX: aviso.
- **Árabe, CJK, cursivo, fórmulas** (pedidos na rodada 0.8): recusados — o
  pipeline (geometria, CTC, feixe com modelo de língua) não é o limite; o
  limite é um leitor treinado nesses sistemas de escrita, com dados que este
  ambiente não tem. Mudar a altura da faixa para 64 px ou o cabeçalho para
  10 mil classes sem treino só produziria ruído com cara de saída
  ([DIRETRIZ.md §11](DIRETRIZ.md)).
