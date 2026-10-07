# A Khazāna — um depósito por conteúdo com raiz *crash-atomic*

> خزانة, raiz خ-ز-ن *kh-z-n*, "guardar". `Vapor.Khazana`. Testes: `khazana_test.exs`; §5l da
> qualidade. A ideia é a do ASAS §8.3–8.4 (*Atomic Stream Application Substrate*), levada da NVRAM
> para um diretório POSIX — [DIRETRIZ §19](DIRETRIZ.md).

## A dor

Todo programa que guarda estado em arquivos reinventa "escreve num temporário, `fsync`, `rename`,
`fsync` no diretório" — e a OTP **não consegue** dar `fsync` num diretório, então na BEAM um
`rename` não é durável. Um arquivo de estado sobrescrito no lugar, por sua vez, pode ficar
*rasgado* por uma queda no meio da escrita. A medição de §5l mostra que o rasgo é pior que a
perda: metade de uma raiz CBOR do mesmo tamanho decodifica — como v2, com um valor que nunca
existiu.

## O protocolo

**Depois de `init/1` nenhum arquivo é criado, renomeado ou apagado.**

```
DIR/pack.0, DIR/pack.1   blobs anexados:  [u32 tamanho][SHA-256 de 32 bytes][bytes]
DIR/root.a, DIR/root.b   um registro fixo de 128 bytes cada, reescrito no lugar
DIR/key                  um segredo de 32 bytes (capacidades, mac/2)
```

Um registro de raiz é `KHZ1 · seq (u64) · geração do pacote (u8) · tamanho confirmado (u64) ·
hash do blob-raiz (32) · etiqueta (32)`, a etiqueta sendo o SHA-256 de tudo antes dela.
`commit/2`:

1. anexa os blobs novos (e o termo-raiz codificado) ao pacote ativo e dá `datasync` — o conteúdo
   é durável antes que algo o nomeie;
2. escreve o slot **inativo** com `seq + 1` e dá `datasync` — o slot ativo nunca é tocado;
3. a raiz corrente é o slot válido de maior sequência.

`open/1` lê os dois slots, descarta o que não confere a etiqueta, fica com a maior sequência
sobrevivente e lê o pacote só até o tamanho que essa raiz confirmou: um anexo rasgado depois dele é
ignorado. Cada blob no prefixo confirmado é re-hasheado na abertura — a corrupção é achada, não
servida. A compactação (`gc/2`) escreve os blobs vivos no *outro* pacote e confirma uma raiz que o
nomeia: o mesmo protocolo, então uma queda no meio deixa o pacote e a raiz anteriores em vigor.

## Medido

`commit(k, termo, fault: {:pack, n} | {:slot, n})` interrompe a escrita no byte `n`. O teste e a
qualidade (§5l) derrubam **cada byte** de um *commit* — 344 pontos, do primeiro byte do anexo ao
último do slot — e reabrem: **sempre** a raiz velha ou a nova. O controle (um arquivo de raiz
sobrescrito no lugar, queda na metade) é lido como um valor rasgado.

## Capacidades

`mac(k, partes)` é HMAC-SHA256 sob a chave do depósito; `mac_ok?/3` compara em tempo constante. As
conversas usam isso para links compartilhados: `mac(["share", conversa, geração])` — infalsificável
sem a chave, e revogar é incrementar a geração (ASAS §6.2: revogação em massa em tempo constante,
sem tabela de capacidades).

## O que não faz

- Supõe que `datasync` só volta quando o dado é durável (verdade para sistemas de arquivos locais
  com discos honestos; não para alguns sistemas de rede ou discos que mentem sobre o cache).
- Supõe que uma escrita de 128 bytes não seja corrompida *em silêncio* de modo a ainda bater com a
  etiqueta (uma colisão de SHA-256).
- Concorrência é do chamador: um processo é dono de um depósito (o Majlis é um `GenServer`).
- P2P (a "Al-Khazāna" do manifesto) fica para depois: sem um modelo de confiança, buscar
  dependências de pares é um vetor de cadeia de suprimentos.

## API

`init/1` · `open/2` · `put/2` · `put_term/2` · `get/2` · `get_term/2` · `has?/2` · `hashes/1` ·
`root/1` · `commit/3` · `gc/3` · `mac/2` · `mac_ok?/3` · `hex/1` · `unhex/1`.
