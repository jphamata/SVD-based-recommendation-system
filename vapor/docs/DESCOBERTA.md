# Descoberta de algoritmos, com certificado (0.11)

> Pedido: "algo similar para síntese e descobrimento de algoritmos com ou
> sem métodos formais e análise de complexidade". Escrutínio:
> [DIRETRIZ.md §14](DIRETRIZ.md). Testes: `discover_test.exs`. Console:
> *Descobrir → Algoritmos*.

## 1. A ideia

AlphaDev (ordenação), AlphaTensor (multiplicação de matrizes) e FunSearch
mostraram que uma **busca** acha algoritmos melhores que os humanos — e
que o que torna o resultado utilizável é a **verificação**: uma busca
pode ser enganada, um certificado não. O vapor faz a mesma divisão, em
escala de uma máquina de 2 núcleos: a busca é heurística e semeada; o
certificado é exato e não confia nela.

## 2. Redes de ordenação (`Vapor.Discover.network/2`)

- **Busca**: em feixe sobre sequências de comparadores, pontuadas pelo
  número de vetores 0-1 que o prefixo ainda pode produzir (uma rede que
  ordena deixa n + 1); a primeira camada fixa no emparelhamento máximo
  (lema de Parberry); comparadores redundantes podados no fim.
- **Certificado**: o **princípio 0-1** (Knuth, TAOCP 5.3.4) — uma rede
  ordena tudo se ordena as 2ⁿ entradas de zeros e uns —, conferido em
  paralelo de bits (cada fio é um inteiro de 2ⁿ bits; um comparador é
  `(a ∧ b, a ∨ b)`).
- **Medido**: tamanhos **ótimos conhecidos para n = 3…8** (3, 5, 9, 12,
  16, 19), achados em < 0,2 s; n = 9 dá 26 (ótimo 25), n = 10 dá 31
  (ótimo 29) — dito. A profundidade não é otimizada (a busca mira o
  tamanho): ótima para n = 3, 4, 7, 8. **Controle**: comparadores sorteados
  até ordenar, cada redundante removido — n = 8: tamanho 25, profundidade
  14. E o certificado não é vazio: tirar **qualquer** comparador da rede
  achada a faz falhar.
- Uma rede achada vira um **programa vapor** (min/max por fio), os mesmos
  bits em todo substrato; conferida contra `Enum.sort` em floats
  sorteados.

## 3. Multiplicação de matrizes (`Vapor.Discover.matmul/3`)

- **Busca**: um algoritmo bilinear de posto *r* é uma decomposição do
  tensor da multiplicação n²×n²×n² em *r* produtos. Mínimos quadrados
  alternados (cada fator resolvido exatamente com os outros fixos, por
  equações normais) a partir de inícios sorteados, com um puxão crescente
  de cada coeficiente para −1, 0 ou 1 na segunda metade, depois
  arredondamento.
- **Certificado**: os 64 coeficientes do tensor 2×2 conferidos **nos
  inteiros** (bignums), e o algoritmo multiplicando matrizes inteiras
  sorteadas exatamente.
- **Medido**: um algoritmo 2×2 com **7 multiplicações** (classe de
  Strassen; 22 somas, contra as 18 de Strassen e as 15 de Winograd) na
  11ª tentativa, ~2 s. **Controles**: as 10 tentativas anteriores também
  foram arredondadas — e estavam erradas: só a conferência exata separa;
  e o **posto 6, impossível** (Winograd 1971), nunca é achado.
- **Complexidade**: as contagens exatas da recursão (7ᵏ multiplicações
  em matrizes 2ᵏ) ajustam **n^2,807** (= n^log₂7) entre n, n log n, n²,
  n^2,807, n³ e 2ⁿ; as do algoritmo escolar, n³; um custo n log n, n log
  n (`complexity/1`, mínimos quadrados com erro relativo). Dito com
  clareza: aqui a classe já está nas contagens (7ᵏ é n^log₂7 por
  definição) — isto confere o classificador, não descobre a taxa; o uso
  real dele é sobre custos **medidos** (tempo, operações contadas) de um
  programa cuja classe não se conhece.
- **Limite**: 3×3 (posto 23, Laderman) não foi tentado; a busca é a de
  livro-texto, não o *flip graph* de Kauers & Moosbauer.

## 4. Síntese de truques de bits (`Vapor.Discover.synthesize/2`)

- **Busca**: enumeração de baixo para cima por número de operações sobre
  `add sub and or xor not neg >>1 sinal` e a constante 1, guardando **um
  termo por função** — a função inteira, nos valores de **todas** as
  entradas do domínio (palavras de 4 bits para duas entradas, 5 para uma).
  Num domínio finito, isso é **sólido e completo**: dois termos iguais em
  todas as entradas são intercambiáveis, então nenhum programa mais curto
  existe sobre este conjunto de operações — uma prova de minimalidade por
  exaustão, não uma heurística.
- **Certificado**: o programa achado é conferido em **todas** as entradas
  de 8 bits e em 10⁵ entradas sorteadas de 16 e 32 bits.
- **Medido**: ⌊(x+y)/2⌋ sem estouro → `(x & y) + ((x ^ y) >> 1)`, **4
  operações**, nada com 3 (85 388 funções distintas exploradas); ⌈(x+y)/2⌉
  → `(x | y) − ((x ^ y) >> 1)`; apagar o bit 1 mais baixo → `x & (x − 1)`;
  isolá-lo → `x & −x`; |x| em 4 (contado como árvore: o sinal aparece
  duas vezes). Os truques do *Hacker's Delight*, redescobertos.
  **Controle**: `(x + y) >> 1`, a fórmula ingênua, falha na conferência
  (estoura).
- **Limite**: a minimalidade vale para palavras de 4–5 bits; a validade
  em 32 bits é amostrada, não provada (uma prova simbólica por bit-vetores
  seria o próximo passo).

## 5. O que liga isto ao resto do vapor

Os três certificados são o mesmo princípio do resto do projeto: o
resultado vale pelo que se confere, não pelo que se buscou. Toda
descoberta é função da semente e se salva como **arquivo recalculável**
([CENA.md §7](CENA.md)).
