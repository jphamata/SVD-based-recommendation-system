# Dossiês de auditoria — evidência que se confere sozinha

> `Vapor.Audit`, `mix vapor.audit`, painel *Dossiê* do console. Desde 0.8.0.

## A dor

Quem precisa mostrar a um auditor (ou a um regulador, ou a um cliente) como
um sistema de IA foi construído e se comportou junta, hoje, capturas de tela,
planilhas e PDFs exportados à mão. Nada disso prova nada: um PDF não diz se
foi editado, uma planilha não diz de onde vieram os números, e o auditor não
tem como refazer a conta. O vapor já produz evidência que **se confere**: o
certificado de cada compilação (a escada de seis degraus, assinada), o diário
de cada execução de agente (cadeia de Merkle, atestado Ed25519), os recibos do
log de transparência, o relatório de qualidade com controles, o contrato de
cada modelo admitido pela eclusa. Faltava um envelope.

## O que é um dossiê

Um arquivo `.vdossier` (CBOR canônico, sem relógio dentro: a mesma evidência e
a mesma data dão os mesmos bytes) com:

- um **manifesto**: para cada item, o tipo, o nome, o tamanho, o SHA-256, o
  resumo da sua verificação e **as cláusulas a que é pertinente**;
- os **itens**, byte a byte como o sistema os escreveu;
- uma **raiz de Merkle** (RFC 6962) sobre os itens;
- **assinaturas Ed25519** sobre o manifesto canônico — quantas se queira
  (quem montou, uma testemunha, o responsável técnico), com quórum na
  verificação;
- **âncoras** opcionais num log de transparência (`Vapor.Tlog`): o dossiê
  passa a existir publicamente numa data, e não pode ser trocado depois sem
  que a troca apareça.

```sh
mix vapor.audit keygen chave
mix vapor.audit export --out sistema.vdossier --key chave --cosign testemunha \
    --system "assistente-juridico" --version 1.4.0 --provider ACME \
    --certificate build/cert.bin --journal runs/42.journal --attestation runs/42.att.json \
    --quality docs/bench/quality.json --model /modelos/qwen2-7b \
    --document avaliacao-de-risco.md:AIA-11,ISO42001-A.6.2.3 --html --pdf
mix vapor.audit verify sistema.vdossier --trusted chave.pub --quorum 1   # sai 1 se algo falhar, dizendo o quê
mix vapor.audit demo                                                     # um dossiê completo deste checkout
```

## Três formas de conferir, nenhuma exige confiar no vapor

1. **`mix vapor.audit verify`** (ou `Vapor.Audit.verify/2`): refaz cada hash,
   a raiz, as assinaturas (só as chaves `--trusted` contam), o quórum, as
   âncoras (com `--log-key`) e **cada item pelas suas próprias regras** — o
   certificado pelas assinaturas dele, o diário pela cadeia e pelo atestado, o
   recibo pela prova de inclusão e pelo checkpoint assinado, o relatório de
   qualidade por todas as verificações aprovadas.
2. **A página HTML** (`--html`): o dossiê inteiro dentro de uma página que se
   confere **no navegador, offline** — um leitor CBOR em JavaScript, SHA-256 e
   Ed25519 do WebCrypto. Abrir o arquivo basta; nada é buscado na rede. Um byte
   alterado deixa a página vermelha e diz qual item.
3. **O relatório PDF** (`--pdf`): para imprimir e arquivar; o texto lista as
   evidências e a raiz, e o `.vdossier` vai **anexado** (`/EmbeddedFiles`) —
   `pdfdetach` o extrai, e `verify` aceita o próprio PDF.

No console, o painel **Dossiê** confere um arquivo solto ali e desenha a
**trama**: dispositivos nas linhas, evidências nas colunas, célula cheia onde
o tipo de evidência é pertinente, vermelha onde o item falhou, e à direita
quantos itens verificados sustentam cada dispositivo — **uma lacuna aparece
como lacuna** ("nenhum"), não some. O botão "Montar um dossiê de
demonstração" faz o que `mix vapor.audit demo` faz e oferece o `.vdossier` e a
página que se verifica sozinha para baixar.

## O mapeamento de cláusulas é dado, não opinião

| tipo de evidência | AI Act (UE) | ISO/IEC 42001 |
|---|---|---|
| certificado de compilação | Art. 15; Anexo IV §2(g) | A.6.2.4 |
| diário de agente atestado | Art. 12, Art. 19 | A.6.2.8 |
| recibo do log de transparência | Art. 12, Art. 19 | A.6.2.8 |
| relatório de qualidade (com controles) | Art. 15; Anexo IV §2(g), §4 | A.6.2.4 |
| contrato do modelo (eclusa) | Art. 11, Art. 13; Anexo IV §2(b) | A.6.2.7 |
| recibo de fusão | Art. 11 | A.6.2.3 |
| documento (avaliação de risco, etc.) | Art. 11 (ou o que se declarar) | A.6.2.7 (ou o que se declarar) |

`Vapor.Audit.clauses/0` e `mapping/0` são a tabela; um item pode declarar as
suas (`--document caminho:CLÁUSULA,…`). A trama mostra, por exemplo, que a
evidência gerada pelo sistema **não cobre** a ISO/IEC 42001 A.6.2.3
(documentação de projeto e desenvolvimento) sem um documento humano — é o tipo
de lacuna que um auditor procura, e é melhor que o dossiê a mostre primeiro.

## O que um dossiê não é

**Não é avaliação de conformidade, certificação nem parecer jurídico.** Um
dossiê autentica evidências: prova que estes bytes saíram deste sistema,
nesta forma, assinados por estas chaves, e que cada item passa nas suas
próprias regras. Se a evidência é *suficiente* para um artigo do AI Act é uma
pergunta jurídica que o dossiê não responde — ele a torna respondível. O
aviso vai dentro de cada dossiê, no PDF e na página.

## Como é testado

- `test/vapor/audit_dossier_test.exs`: montar, assinar, co-assinar, ancorar;
  conferir offline; **qualquer byte alterado é recusado, e o relatório diz
  onde** (um item: o hash; o manifesto: as assinaturas; um item trocado com o
  hash atualizado no manifesto: as assinaturas); signatário não confiável não
  conta; o PDF devolve o dossiê (e o `pdftotext`/`pdfdetach` do poppler
  concordam); a página HTML confere no motor de um navegador (WebCrypto do
  Node) e fica vermelha com um item alterado.
- `mix vapor.quality` §5d: 0 de ~200 alterações de um bit espalhadas pelo
  arquivo aceitas (controle: os próprios arquivos alterados).
- `test/vapor/console_test.exs`: o dossiê de demonstração confere pelo
  console; o mesmo com um byte trocado, não.
