# Autojogo e mundos manipulados (0.11)

> Pedido: "reinforcement learning (com algo similar a manipulação de
> ambientes e similar a AlphaZero e além)". Escrutínio:
> [DIRETRIZ.md §14](DIRETRIZ.md). Testes: `games_test.exs`. Console:
> *Simular → Jogos*. O RL da 0.9 (CartPole, FrozenLake, Reacher) e da 0.10
> (carro-pêndulo como programa, gêmeos) continua: [ESTUDIO.md](ESTUDIO.md),
> [FISICA.md](FISICA.md).

## 1. Autojogo com política, valor e PUCT, pequeno (`Vapor.Games`; o método de Silver et al. 2018)

O algoritmo de Silver et al. (2018), inteiro, em escala de jogo da velha:

- uma rede com cabeça de **política** e de **valor** (18 entradas — as
  pedras de quem joga e as do adversário — → 64 tanh → 9 + 1);
- **busca em árvore Monte Carlo guiada por ela** (PUCT, ruído na raiz
  durante o autojogo, os dois primeiros lances sorteados pelas visitas);
- **treino só com as próprias partidas**: cada posição guardada com a
  distribuição de visitas da busca e o resultado final, aumentada pelas
  8 simetrias do tabuleiro; SGD com retropropagação escrita à mão.

Nada de partidas humanas, nada de heurística além da legalidade e do fim.
O treino embarcado (`priv/games/tictactoe.json`, 400 partidas, 32
simulações por lance, semente 1) leva ~77 s e é **função da semente**: o
mesmo digest em qualquer máquina (testado).

**Medido contra um jogador perfeito** (negamax sobre a árvore inteira).
A busca do agente é determinística, então a árvore de partidas que ele
enfrenta só se ramifica onde o jogador perfeito tem mais de um lance
ótimo — e **todos** são seguidos, dos dois lados
(`versus_every_optimal_line/2`): uma contagem exaustiva, não uma amostra.

| simulações por lance | linhas perdidas, rede treinada | linhas perdidas, a mesma busca sem treino (controle) |
|---|---|---|
| 8 | **17 de 129** (13 %) | 169 de 175 (97 %) |
| 16 | 13 de 133 | 66 de 110 |
| 32 | 8 de 147 | 10 de 113 |
| 64 | 4 de 131 | 4 de 111 |
| 128 | **0 de 135** | 0 de 87 |

A rede **compra força por simulação**: com 8 simulações, a busca
treinada perde 13 % das linhas onde a crua perde quase todas; a partir de
32–64 simulações a busca sozinha alcança a rede neste jogo pequeno, e com
128 nenhuma das duas perde. Contra um jogador aleatório, 39 vitórias e 1
empate em 40.

**O que a revisão achou.** A primeira versão desta medida sorteava 60
partidas (30 de cada lado) entre as linhas ótimas e relatava "0 derrotas
com 64 simulações". A contagem exaustiva mostra 4 linhas perdidas de 131
com 64 — a amostra não passou por elas. A afirmação foi corrigida para o
que vale (nenhuma derrota com 128, em todas as linhas), e o teste agora
confere as duas coisas: a amostra dá 0, a exaustão dá 4. Dois retreinos
maiores (1 200 partidas com 32 simulações; 1 200 com 64 e duas épocas)
foram tentados: o primeiro melhora o meio da curva (2 linhas perdidas de
16 a 128 simulações) mas não zera em 128; o segundo é pior em tudo. A
rede embarcada continua a de 400 partidas.

## 2. Manipulação de ambientes: aleatorização de domínio

O carro-pêndulo do gymnasium com as constantes físicas expostas
(`cart_step/3`). Políticas lineares por busca aleatória (estilo ARS),
treinadas:

- no carro-pêndulo **nominal**, ou
- em carros-pêndulo **sorteados** a cada episódio (haste ×0,5–3, massa
  da haste ×0,5–4, massa do carro ×0,5–2, força do motor ×0,3–1) — Tobin
  et al. (2017);

e testadas em **quatro carros-pêndulo que nenhuma viu**: três **fora** da
faixa de treino (haste longa e pesada com motor fraco; carro pesado,
2,4×; haste curta e pesada) e um **no canto** dela (haste longa, carro
pesado e motor fraco ao mesmo tempo — dentro da caixa, mas uma combinação
que o sorteio quase nunca produz).

| semente | treinada em um | treinada em muitos |
|---|---|---|
| 1 | 171 | 500 |
| 2 | 500 | 496 |
| 3 | 153 | 457 |
| **média** | **275** | **484** |

(passos em pé, de 500). As duas equilibram o nominal (500). A semente 2
mostra a honestidade da medida: às vezes a política de um mundo
generaliza por sorte — por isso a média de três sementes e o limiar
"≥ 400 e > nominal + 100".

## 3. "E além": o que foi ponderado

- **MuZero** (aprender o modelo do ambiente) e **jogos maiores** (Go,
  xadrez) exigem ordens de grandeza mais computação; o caminho no vapor é
  o mesmo código com a rede como programa vapor (o autojogo determinístico
  é o que torna uma curva de treino reprodutível — o argumento da 0.10).
- **Ambientes manipulados por um adversário** (PAIRED, currículos
  gerados) — o passo seguinte natural da aleatorização: está no TODO.
- A rede aqui roda em binary64 na BEAM, não como programa compilado.
