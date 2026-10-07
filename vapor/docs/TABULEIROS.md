# Tabuleiros e cartas — regras fixadas por contagens publicadas, jogo julgado por jogo perfeito

> Pedido (0.12): "similar ou superior a […] alpha zero (com comparações)
> […] quem sabe um carinho especial a xadrez, shogi e go e jogos de
> cartas". Escrutínio: [DIRETRIZ.md §15](DIRETRIZ.md).

A dor real de quem pesquisa ou ensina jogos não é a falta de motores
fortes — eles existem, livres — é **saber que as regras estão certas** e
**medir** um agente sem depender de outro programa como juiz. Por isso
cada jogo aqui é fixado pelas contagens publicadas, e cada agente é
julgado contra o jogo perfeito sempre que ele existe.

Console *Simular → Tabuleiros e cartas* · MCP `board_query`.

## 1. Xadrez (`Vapor.Play.Chess`)

Regras completas (roque por casas não atacadas, en passant, promoção a
qualquer peça, 50 lances, material insuficiente, repetição), FEN e SAN
de ida e volta, PGN. **Perft** igual aos números publicados na posição
inicial e nas quatro posições-padrão (Kiwipete etc.); o
**python-chess** concorda nos lances legais e no perft(2) ao longo de
partidas aleatórias. Motor alfa–beta (aprofundamento iterativo,
quiescência, tabela de transposição, MVV–LVA, *killers*, tabelas
peça-casa), ~76 mil nós/s numa vCPU. **Provador de mate**: devolve uma
árvore — toda defesa respondida, toda folha um xeque-mate — que
`verify_mate/3` reproduz independentemente; uma prova errada é recusada.

## 2. Shogi (`Vapor.Play.Shogi`)

As oito peças e suas seis promoções, peças capturadas que **mudam de
lado e voltam como lançamento** (sem dois peões não promovidos na mesma
coluna, sem peça lançada onde nunca poderia mover, sem lançamento de peão
que dê mate), promoção compulsória, zona de promoção; SFEN e USI. Perft
**30 · 900 · 25 470** da posição inicial; o **python-shogi** concorda nos
lances legais ao longo de partidas aleatórias com lançamentos e
promoções.

## 3. Go (`Vapor.Play.Go`)

Regras de Tromp–Taylor (contagem por área, komi), suicídio recusado,
**superko posicional**. Contagem de posições legais **1 · 57 · 12 675**
em 1×1, 2×2, 3×3 (Tromp & Farnebäck). Um jogador MCTS com *playouts* que
evitam encher os próprios olhos vence o aleatório em 5×5 (o aleatório
contra si mesmo, o controle, empata em média).

## 4. k em linha (`Vapor.Play.MNK`)

Qualquer m×n com k em linha, com ou sem gravidade (Connect Four é 7×6,
k = 4, gravidade), simetrias do tabuleiro. Resolvido **exatamente** por
negamax com tabela de transposição até 16 casas: 3×3×3 empate, 4×3×3 e
4×4×3 com gravidade vitória do primeiro.

## 5. Pôquer (`Vapor.Play.Poker`)

**CFR+** com estratégia congelada por iteração, e a
**explorabilidade exata** por melhor resposta. Kuhn: valor do jogo
−1/18 (−0,0556) e explorabilidade < 10⁻³ fichas/mão; o jogo uniforme
(controle) é explorável em 0,458. Leduc hold'em (com a carta comunitária
e duas rodadas): explorabilidade < 0,02 em 100 iterações, valor perto de
−0,0856.

## 6. Autojogo genérico (`Vapor.Play.SelfPlay`)

O aprendiz de 0.11 (política + valor + PUCT, treinado só pelas próprias
partidas) generalizado para **qualquer jogo** que dê `features/1`,
`actions/1` e `index/2`, com aumento de dados pelas simetrias e ruído de
Dirichlet na raiz. Julgado contra **todas** as linhas ótimas do jogador
perfeito, dos dois lados: em 3×3×3, 300 partidas de treino, com 8
simulações a rede treinada perde **24 de 114** linhas; a mesma busca sem
treino (controle) perde **165 de 205**.

Honestamente: o aprendiz especializado de 0.11 (`Vapor.Games`, retirado em
0.16 em favor deste) perdia 17 de 129 com 8 simulações — o genérico é um
pouco pior no mesmo jogo (21 % contra 13 %), em troca de servir a qualquer
jogo.

## 7. Comparação com o que existe

| | aqui | Stockfish / YaneuraOu / KataGo / Lc0 |
|---|---|---|
| regras | fixadas por perft e por outro programa (python-chess, python-shogi) | idem (a comunidade usa os mesmos perft) |
| força | motor didático, ~76 k nós/s, profundidade 4–6 | milhões de nós/s, redes treinadas em bilhões de posições; ordens de grandeza mais fortes |
| prova | **árvore de mate conferida** por um verificador independente | um lance e uma avaliação |
| julgamento de agentes | contra o **jogo perfeito**, exaustivo, onde ele existe | contra outros motores (Elo) |
| pôquer | explorabilidade **exata** em Kuhn e Leduc | solvers comerciais/de pesquisa para hold'em completo (escala muito maior) |

O que é igual ou superior aqui: a **verificabilidade** (perft, provas de
mate conferidas, explorabilidade exata, juiz perfeito exaustivo). O que
não é: força de jogo em xadrez, shogi e Go de tamanho real.

## 8. Limites honestos

- Nenhum motor daqui é competitivo com os motores abertos de ponta.
- Go: o MCTS sem rede joga bem só em tabuleiros pequenos; 19×19 está
  fora do orçamento do console (13×13 no máximo).
- Shogi: o motor é material + busca rasa (profundidade ≤ 3).
- Pôquer: Kuhn e Leduc; hold'em completo exige abstração e outra escala.
