# Copela — corrupção silenciosa, flagrada

> Desde 0.15.0. Código: `lib/vapor/cupel.ex`, `lib/vapor/cupel/sentinel.ex`.
> Testes: `test/vapor/cupel_test.exs`. Console: *Opus → Copela*. Terminal: `vapor cupel`.
> MCP: `cupel_drill`.

A copela é o prato poroso do ensaiador: o metal vil é absorvido, o nobre fica.

## A dor

Em escala de frota, um núcleo defeituoso devolve **números errados sem erro nenhum** (os relatos
de *silent data corruption at scale* da Meta e do Google: cerca de uma máquina em mil). O treino
absorve isso como um pico de perda dias depois; a inferência serve. Replicar cada produto e
comparar bit a bit (o que o `Vapor.Cluster` já faz) custa 2×. Uma verificação precisa ser **mais
barata que o produto** e **nunca acusar um substrato correto**.

## O primeiro princípio

Para `y = x·Wᵀ` e qualquer vetor `r`, `y·r = x·(Wᵀr)` — a identidade adjunta
⟨Wx, r⟩ = ⟨x, Wᵀr⟩. `Wᵀr` depende só dos pesos: calculado uma vez por matriz (a **sonda**), torna
cada verificação `O(b·(n + k))` contra `O(b·n·k)` do produto (Freivalds, 1977). Duas escolhas
fazem disso um veredito e não uma heurística:

- **comparação exata** — `ŷ·r` e `x·(Wᵀr)` em racionais diádicos (as células da
  [Amálgama](AMALGAMA.md)): a única folga é o arredondamento do próprio substrato;
- **tolerância provada, não ajustada** — o lema 3.1 de Higham (em `proofs/Vapor/Higham.lean`):
  qualquer substrato conforme, em qualquer ordem de soma, com ou sem FMA, satisfaz
  `|ŷᵢⱼ − yᵢⱼ| ≤ γₖ·Σₗ|xᵢₗ||Wⱼₗ| + 2k·η` (η = 2⁻¹²⁶ cobre *flush-to-zero*), mais os operandos
  que um substrato DAZ pode ler como zero. Projetada em `|r|`, a cota custa o mesmo
  `O(b·(n + k))`. Uma linha cuja discrepância passa da cota **não pode** ter vindo de um
  substrato correto.

`r` tem entradas inteiras `±[1, 2²⁰]` sorteadas de uma semente: um elemento corrompido é pego
sempre que `|δ|·|rⱼ| > 2·tol`; vários elementos conspirando se cancelam com probabilidade ≈ 2⁻²⁰
por linha, e não contra quem não conhece a semente. Para pesos e ativações `s8` (o GEMM int8,
resultados `s32`) a verificação é exata: tolerância zero, os 32 bits pegos.

Três vereditos por linha: `:ok`, `:corrupt`, `:unchecked` (entradas não finitas, ou saídas que
podem estourar legitimamente — nunca acusadas, nunca chamadas de "ok"). Uma saída não finita de
entradas finitas é corrupção.

## A sentinela

`Vapor.Cupel.Sentinel` guarda um conjunto de substratos que calculam `x·Wᵀ` para uma matriz:
cada resultado passa pela copela antes de ser devolvido; um substrato cujo resultado não pode ter
vindo de aritmética correta é posto em **quarentena** com a evidência, o produto é refeito no
próximo saudável (ou pelo oráculo exato), e o evento entra num diário encadeado por SHA-256 e
fechado por uma raiz de Merkle — o registro que um operador leva ao fabricante. O trabalho roda
no processo de quem chama (chamadas concorrentes); o servidor só guarda a sonda, a saúde e o
diário. Um trabalhador que morre é uma entrada do diário, não uma queda.

## Medido

| | |
|---|---|
| 256 × 256, lote 8 | sonda 82 ms (uma vez); produto pelo oráculo 142 ms; verificação 3,2 ms |
| um bit trocado numa saída (32 × 64, 12 lotes) | bits 19–31: 100 %; bit 0: 0 % — abaixo do envelope, e dito |
| 4 ordens de soma conformes × 6 escalas (10⁻³⁰…10¹⁵) | 0 acusações em 24 |
| o falsificador que conhece `r` | passa; a mesma falsificação sob outra semente: pega |
| 20 chamadas concorrentes, um núcleo que troca um bit do expoente | 20/20 respostas corretas, o núcleo em quarentena |

O perfil completo por bit (`sensitivity/3`, `vapor cupel`): sinal, expoente e mantissa alta
sempre pegos; os bits baixos da mantissa de saídas grandes são indistinguíveis do arredondamento.

## Achado no caminho

O oráculo `dot16` **truncava em silêncio** contrações com `k` fora de múltiplo de 16 (a cauda era
ignorada). Hoje recusa com erro; o teste que achou está em `cupel_test.exs`.

## O que não é

Silício não se torna defeituoso sob pedido: as falhas são injetadas na fronteira onde um núcleo
defeituoso as emitiria (o `runner` dos testes e do exercício). A verificação cobre a camada
linear; as não linearidades entre camadas são verificadas pela própria escada do vapor
(envelopes por operador), não por esta identidade.
