# Transparência: o que foi assinado não pode ser reescrito em silêncio

Um certificado Ed25519 prova que **alguém assinou**. Não prova que essa
pessoa não assinou, para outra plateia, outra coisa: um operador pode
mostrar a um auditor um histórico e aos usuários outro (*split view*), ou
apagar ontem o recibo que hoje o incomoda. A resposta da indústria a isso —
Certificate Transparency, o *checksum database* do Go, o Rekor do Sigstore —
não é uma blockchain: é um **log só-acréscimo com testemunhas**.

`Vapor.Tlog` é esse log, nos formatos que essas ferramentas já falam, de
modo que um log mantido pelo vapor pode ser testemunhado e auditado por
programas que nunca ouviram falar do vapor.

## 1. As peças

| peça | o que é | padrão |
|---|---|---|
| árvore | Merkle com `SHA-256(0x00‖entrada)` nas folhas e `SHA-256(0x01‖e‖d)` nos nós | RFC 9162 (o mesmo hashing de `Vapor.Merkle`) |
| prova de inclusão | a entrada `i` está na árvore de tamanho `n` com raiz `r` | RFC 9162 §2.1.3 |
| prova de consistência | a árvore de `n` entradas **estende** a de `m` — nada foi removido ou reescrito | RFC 9162 §2.1.4 |
| *checkpoint* | origem, tamanho, raiz em base64, assinados | C2SP `tlog-checkpoint` em `signed-note` (Ed25519) |
| co-assinatura | uma testemunha atesta que viu esse *checkpoint* **e** a consistência com o anterior | C2SP `tlog-cosignature/v1` |
| arquivo | entradas com prefixo de tamanho, cada acréscimo com `fsync`; `open/1` reconstrói **e reverifica** a árvore | — |

Os verificadores são conferidos contra as 196 sondas do transparency-dev
(`test/fixtures/tlog/probes.json`), casos negativos incluídos — provas
truncadas, hashes de tamanho errado, `size1 = 0`, raízes trocadas. Um
verificador "ingênuo" (que confia na contagem de passos da própria prova)
acerta 182 de 196: é o controle da verificação de qualidade
(`mix vapor.quality`, §5b).

## 2. A testemunha

`Vapor.Tlog.Witness.cosign/4` só co-assina um *checkpoint* depois de
conferir a prova de consistência desde o último que viu daquele log; recusa
**retrocesso** (tamanho menor) e **bifurcação** (mesmo tamanho, outra raiz,
ou prova que não fecha). Para mostrar duas histórias a duas plateias, o
operador precisaria que as testemunhas conspirassem também — e qualquer um
pode ser testemunha.

## 3. No servidor e no console

Com `mix vapor.serve --tlog PATH` (origem: `--tlog-origin`, padrão
`vapor.local/console`), o servidor guarda o log e a sua chave (`PATH.key`,
modo 0600) e:

| chamada | resposta |
|---|---|
| `GET /v1/vapor/tlog` | origem, tamanho, raiz, *checkpoint* assinado, chave de verificação, últimas entradas |
| `POST /v1/vapor/tlog` | `{"text"}` ancorado; o seu recibo (índice, prova, *checkpoint*) |
| `GET /v1/vapor/tlog/proof?index=i` | a entrada `i` com o recibo contra a árvore atual |
| `GET /v1/vapor/tlog/consistency?from=m` | a prova de que a árvore atual estende a de `m` entradas |
| `POST /v1/vapor/search` | os recibos de busca saem **ancorados** (`tlog`) |

No console, a aba **Registro / Ledger** (grupo *Confiar*) não pede que se
confie no servidor: **o navegador verifica sozinho**. O verificador em
JavaScript (WebCrypto Ed25519, SHA-256) confere a assinatura do
*checkpoint*, a prova de inclusão de cada entrada e — guardando o último
*checkpoint* visto — a consistência com ele; a chave do log é fixada no
primeiro uso (TOFU, como o SSH) e uma troca de chave é denunciada, não
aceita. A prova de inclusão é desenhada: a folha, os irmãos que sobem, a
raiz. O mesmo verificador roda no Node contra as sondas
(`test/js/tlog_verify.mjs`).

## 4. Por que não uma blockchain

O que a ancoragem precisa é (1) um compromisso só-acréscimo e (2)
observadores independentes que detectem bifurcação. Um log com testemunhas
dá as duas coisas com uma assinatura por *checkpoint*. Uma blockchain dá as
mesmas duas por consenso pago, com latência de minutos e um token no meio —
custo sem propriedade nova. Se um dia for útil publicar a raiz numa cadeia
pública, ela é uma linha de texto: o *checkpoint* já é o objeto a ancorar.

## 5. Limites

- Uma testemunha embutida (`Vapor.Tlog.Witness`) está pronta; a rede de
  testemunhas é social — é preciso que outros a rodem.
- O log guarda as entradas inteiras em memória (o arquivo é a persistência);
  para centenas de milhões de entradas, a forma de *tiles* do C2SP
  (`tlog-tiles`) é o próximo passo.
- O console fixa a chave por navegador; um auditor deve fixá-la fora de
  banda.
