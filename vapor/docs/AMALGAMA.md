# Amálgama — somas que não lembram a ordem

> Desde 0.15.0. Código: `lib/vapor/amalgam.ex`, `lib/vapor/train/lm.ex` (`reduce: :exact`).
> Testes: `test/vapor/amalgam_test.exs`, `test/vapor/train_exact_test.exs`. Console: *Opus → Amálgama*.
> Terminal: `echo "1e16 1 -1e16" | vapor amalgam -` (`--f32` para binary32). MCP: `amalgam_sum`.

## A dor

A adição em ponto flutuante é comutativa e **não associativa**. Uma soma distribuída — o
*all-reduce* dos gradientes, a junção de resultados parciais de nós que entram e saem — tem bits
que dependem de *quem somou o quê primeiro*. A política canônica até a 0.14 respondia
**fixando a forma** da redução (16 pistas, uma árvore binária fixa sobre os índices dos
micro-lotes, contagem em potência de dois). Correto, mas uma restrição sobre tudo em volta: o
número de micro-lotes, o escalonamento, a troca de nó numa falha, e a divisão por linhas que o
`Vapor.Shard` precisava recusar.

## O primeiro princípio

Todo valor de um formato binário é um múltiplo inteiro do seu menor subnormal, `2^qmin`. Logo a
soma de `n` valores é um inteiro vezes `2^qmin`, e a adição de inteiros **é** associativa. Os
inteiros de precisão arbitrária da BEAM fazem desse inteiro um acumulador de Kulisch sem limbos a
administrar:

- uma **célula** é `Σ mᵢ·2^(eᵢ − qmin)`, exata;
- `merge/2` soma células — um monoide comutativo com `:empty` como identidade;
- `round/1` arredonda **uma vez**, ao par mais próximo, com *underflow* gradual e estouro para ±∞.

O resultado é o valor corretamente arredondado da soma real verdadeira — uma definição que não
nomeia ordem nenhuma, então é também a canônica. Os valores especiais seguem a IEEE 754 §6.3 em
qualquer ordem: NaN, `+∞` com `−∞` dá NaN; um infinito domina os finitos; a soma só de `−0` é
`−0`, qualquer cancelamento dá `+0`. Formatos: f16, bf16, f32, f64.

`dot/3` e `partial_dot/3` fazem o mesmo com produtos (o produto de dois binários é exato num
inteiro maior): fatias de uma contração, juntadas em qualquer ordem, arredondam para o produto
escalar sem fatias — a divisão por linhas feita exata. `mean/2` é o quociente corretamente
arredondado (conferido contra os dois vizinhos, em aritmética exata). `to_wire/1` e `from_wire/1`
levam uma amálgama pela rede; a leitura recusa células além de `contagem × maior finito`.

## No treino

`LM.start(…, reduce: :exact)` troca a árvore fixa pela média exata: o gradiente do passo é a
média corretamente arredondada da soma exata dos gradientes dos micro-lotes. Os bits passam a
depender **só do conjunto** de micro-lotes — não de quantos trabalhadores, de quem calcula o quê,
da ordem de chegada, de uma queda no meio, nem de a contagem ser potência de dois. Medido:
três micro-lotes com 1, 2 ou 3 trabalhadores, atribuições arbitrárias, um trabalhador morto no
meio e o oráculo — um só *digest*; o *checkpoint* e a retomada continuam com os bits da corrida
ininterrupta. É **outra definição** que a árvore (os *digests* diferem — o teste pode falhar), e
as duas aprendem (parâmetros a < 10⁻³ um do outro).

## Medido

| | |
|---|---|
| 64 vetores × 4 096 f32 | 42 ms na BEAM (≈ 6 M adições exatas/s); arredondar 4 096 células: 6 ms |
| 2 048 valores com cancelamento, 30 ordens e agrupamentos | 1 resultado, igual à soma exata arredondada uma vez; a soma da esquerda para a direita deu 30 resultados diferentes |
| f64, `[1e16, 1, −1e16]` | `1.0` (a soma ingênua dá `0.0`) |

## O que não é

Não é de graça: uma adição de *bignum* por elemento (≈ 0,1–0,2 µs). É uma redução para o plano
de controle — gradientes entre micro-lotes e nós, resultados parciais num *cluster* — e não para
o laço interno de um *kernel*, cuja forma exata pede operações inteiras nos cinco emissores
([TODO](TODO.md)). Termos finitos cuja soma verdadeira é finita nunca estouram (a IEEE da
esquerda para a direita pode estourar no caminho) — é uma diferença de semântica, dita.
