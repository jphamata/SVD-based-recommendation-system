# Rebis — dois circuitos, uma função?

> Desde 0.15.0. Código: `lib/vapor/rebis.ex`, `lib/vapor/rebis/{gen,ideal,field,gcm,stabilizer}.ex`.
> Testes: `test/vapor/rebis_test.exs`. Console: *Opus → Rebis*. Terminal: `vapor rebis equiv|anf|identity|stabilizer|aiger`.
> MCP: `rebis_check`.

O *rebis* é a "coisa dupla" dos alquimistas: duas naturezas mostradas como uma.

## A dor

Uma *netlist* depois da síntese, um bloco de IP de terceiros, um chip de volta da fundição — é a
especificação? A simulação responde pelos padrões que tentou; um cavalo de Troia de hardware
cujo gatilho é uma coincidência de 64 bits sobrevive a toda bancada de teste já rodada.
Equivalência é um teorema ou não é nada.

## A álgebra, dita com exatidão — e a correção da proposta

Sobre GF(2), XOR é `+`, AND é `·`, NOT é `+1`; toda função booleana de `n` entradas tem **um**
polinômio multilinear — a forma normal algébrica (Zhegalkin) — em `GF(2)[x₁…xₙ]/⟨xᵢ² − xᵢ⟩`.
A proposta da rodada ("provar `P_A − P_B ≡ 0` com bases de Gröbner") está certa e é a
ferramenta errada: a ANF **já é** a forma normal módulo esse ideal, calculada pela transformada
de Möbius em `O(n·2ⁿ)` operações de palavra — sem Buchberger — e além de ~20 entradas nenhuma
forma normal é barata (equivalência é coNP-completa). Então, dois procedimentos, cada um com
saída conferível:

- **`n ≤ 16`**: a tabela-verdade de cada saída como **um** inteiro da BEAM de `2ⁿ` bits (todos os
  padrões de uma vez), comparada; a ANF por Möbius em `n` passos de deslocamento e XOR.
- **além**: 4 096 padrões aleatórios de uma vez, depois um **miter** — `OR(outᵃᵢ ⊕ outᵇᵢ)` — por
  Tseitin em CNF para o `Vapor.Logic.SAT`. UNSAT vem com uma prova DRUP que o
  `Vapor.Logic.DRUP` (código que não compartilha nada com o resolvedor) confere antes de a
  resposta ser "equivalente"; SAT vem com um modelo, re-simulado nos dois circuitos antes de ser
  chamado de contraexemplo.

Um contraexemplo é **encolhido** (o menor número de entradas em 1, gulosamente), então o
gatilho de um cavalo de Troia se lê como o gatilho.

## Aritmética de palavras: Gröbner onde Gröbner é a ferramenta certa

Para identidades de palavra (64 fios são o produto de duas palavras de 32 bits), GF(2) é o anel
errado. `Vapor.Rebis.Ideal` trabalha **sobre ℤ** com `x² = x`: cada porta é um polinômio
(`¬a = 1 − a`, `a∧b = ab`, `a⊕b = a + b − 2ab`, …) e, numa ordem lexicográfica que põe cada
porta acima das suas entradas, os polinômios das portas **já são** uma base de Gröbner do ideal
do circuito (os termos líderes são variáveis distintas). Reduzir a especificação por eles é
substituir portas de trás para frente: o resto é `0` sse a identidade vale para toda entrada
(Lv, Kalla & Enescu 2013; Ritirc, Biere & Kauers 2017). Um resto não nulo dá um ponto onde a
identidade falha — reavaliado no circuito antes de ser relatado. Especificações em texto:
`m[16] = a[8] * b[8]`, `s[8] + 2^8*cout = a[8] + b[8]`.

Os dois procedimentos são **complementares**, medido:

| | CDCL + DRUP | álgebra sobre ℤ |
|---|---|---|
| multiplicador, comutatividade | 2 963 conflitos a 5 bits; não termina em minutos a 6 | 16 bits: 2 748 substituições, pico de 522 termos; **32 bits: 1,4 s, pico de 2 058** |
| somador *ripple* 64 bits | — | linear: pico < 1 000 termos |
| *ripple* × Kogge–Stone | 16 bits: prova DRUP conferida; 64 bits: 88 s (7 541 conflitos, 5 380 lemas conferidos) | Kogge–Stone 32 bits: passa de 50 000 termos → `:unknown`, nunca um palpite |

## Corpos binários, AES e GCM

`Vapor.Rebis.Field`: GF(2ⁿ) com o produto **sem vai-um** (o que `PCLMULQDQ`, `PMULL` e `vclmul`
calculam), redução, inverso por Euclides estendido, teste de irredutibilidade de Rabin. A S-box
do AES é **derivada** (o inverso em GF(2⁸) seguido da afim), não tabelada — e cada bit de saída
tem grau algébrico 7, recalculado por Möbius. O GHASH é feito de dois jeitos (o algoritmo do
NIST e o produto refletido) que concordam. `Vapor.Rebis.GCM`: AES-GCM inteiro sobre isso, igual
ao OpenSSL (`:crypto`) em toda combinação testada de comprimento de mensagem, AAD e IV; uma
etiqueta adulterada é recusada. É um **conferidor**, não uma biblioteca para cifrar dados de
produção (sem tempo constante).

## Estabilizadores

`Vapor.Rebis.Stabilizer`: circuitos de Clifford em milhares de qubits, exatos, numa máquina
clássica (Gottesman–Knill, quadro CHP de Aaronson–Gottesman). Cada linha é dois inteiros da BEAM
(`x`, `z`) e um sinal; a fase de um produto de linhas é calculada para todos os qubits de uma vez
(as posições que ganham `+i` e `−i` são duas máscaras, a fase é a diferença dos *popcounts*
mod 4). Conferido contra um simulador denso de vetor de estado em 60 circuitos aleatórios de até
5 qubits; um estado GHZ de 400 qubits dá 1 medida aleatória e 399 determinadas. O `T` sai do
formalismo e é recusado pelo nome.

## Entrada

Uma pequena linguagem de *netlist* (`input`, `output`, `w = a & ~b ^ c`, `mux(s, a, b)`,
`maj(a, b, c)`, com *hash-consing*) ou AIGER ASCII (`aag`, o formato das competições de
verificação de hardware), lido em ordem topológica de Kahn (portas fora de ordem aceitas, ciclos
recusados). Nomes nunca viram átomos; tamanhos têm teto.

## Achado no caminho

- Um miter com **literais repetidos** numa cláusula travava o resolvedor; as cláusulas são
  normalizadas (repetições removidas, tautologias descartadas) antes do SAT.
- AIGER com portas fora de ordem era recusado; arquivos reais não garantem a ordem.
- A fase das linhas desestabilizadoras do CHP pode ser ímpar (só as estabilizadoras são
  hermitianas); a soma de linhas aceita isso nelas.

## O que não é

Não é síntese nem *place-and-route*; circuitos sequenciais (com registradores) entram só como a
sua parte combinacional (o *miter* de um passo). Um emulador de chips antigos como redes sobre
GF(2⁸) não foi feito ([DIRETRIZ §18](DIRETRIZ.md)).
