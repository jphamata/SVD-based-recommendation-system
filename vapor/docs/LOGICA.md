# Lógica — um procedimento propõe, um verificador decide

> Pedido (0.12): "controle fino […] para CS, matemática, física (nível
> Fields, Turing, Nobel e fronteira), usando IA com maestria, com ou sem
> humano no loop, para problemas arbitrários". Escrutínio:
> [DIRETRIZ.md §15](DIRETRIZ.md).

O que torna confiável um sistema que produz matemática — seja um
provador clássico, seja um modelo de linguagem treinado por reforço sobre
um assistente de provas — é sempre a mesma separação: **quem propõe não
decide**. A mesa de lógica implementa essa separação para quatro lógicas
decidíveis e a abre a qualquer proponente: uma pessoa, uma busca, ou um
modelo de linguagem pelo MCP. A aceitação depende só do verificador.

`Vapor.Logic.run/1` (a mesa decide) · `Vapor.Logic.check/2` (a mesa
confere a proposta de outro) · console *Resolver → Lógica* · MCP
`logic_check`.

## 1. As quatro lógicas

| entrada | lógica | procedimento | certificado e verificador |
|---|---|---|---|
| `p cnf …` (DIMACS) | proposicional | **CDCL** (2 literais vigiados, 1-UIP, minimização, reinícios de Luby, atividade) | modelo (avaliado cláusula a cláusula) ou refutação **DRUP**, conferida por `Vapor.Logic.DRUP` (para trás, núcleo primeiro) — um programa separado que não compartilha código com o solver |
| `valid: φ`, `sat: φ`, `equiv: a ; b` | fórmulas | Tseitin → CDCL | o mesmo; um contraexemplo avaliado na fórmula original |
| `schur k`, `vdw k r`, `ramsey s t`, `pigeonhole p h`, `queens n` | combinatória finita | codificação + CDCL, busca do limiar | o número: uma testemunha **abaixo** (conferida pela definição) e uma refutação DRUP **no** limiar |
| equações + `decide s = t` | teorias equacionais | **Knuth–Bendix** com ordem lexicográfica de caminhos | sistema de reescrita convergente; formas normais com as derivações |
| `vars` / `hyp` / `claim` | geometria polinomial | **Buchberger** (critérios, base reduzida), truque de **Rabinowitsch** | base {1} para a implicação; o resto, quando não segue; condições de não degenerescência |

Resultados conferidos (`logica_test.exs`, §5h): o solver concorda com a
enumeração exaustiva em 150 3-CNFs aleatórias no limiar; **S(3) = 13**,
**W(3; 2) = 9**, **R(3, 3) = 6**, cada um com testemunha e refutação
conferidas; uma prova adulterada é rejeitada (o controle); Knuth–Bendix
completa os três axiomas de grupo nas **dez regras clássicas** e decide
i(x·y) = i(y)·i(x) — os axiomas só orientados (controle) não decidem; a
base lex do exemplo de Cox–Little–O'Shea; o **teorema de Tales** provado e
uma variante falsa "não implicada".

## 2. Humano ou IA no loop: `logic_check` com proposta

Um proponente externo manda a afirmação **e** um candidato:

| afirmação | proposta | aceita quando |
|---|---|---|
| DIMACS | `{"model": [1, -2, 3]}` | toda cláusula tem um literal verdadeiro |
| DIMACS | `{"drup": [[…], …, []]}` | o verificador DRUP deriva a cláusula vazia por propagação de unidades |
| `schur k` | `{"witness": [cores de 1..n]}` | cores em 1..k e nenhum x + y = z monocromático → S(k) ≥ n |
| `vdw k r` | `{"witness": [cores de 1..n]}` | nenhuma progressão de k termos monocromática → W(k; r) > n |
| `ramsey s t` | `{"n": n, "red": [[a, b], …]}` | nem K_s vermelho nem K_t azul → R(s, t) > n |
| `sat: φ` / `valid: φ` | `{"assignment": {…}}` | φ verdadeira (modelo) / φ falsa (contraexemplo à validade) |

É assim que um modelo de linguagem participa sem precisar ser confiável:
ele pode propor uma coloração de Schur maior, um contraexemplo, uma
refutação que outro solver produziu — e a mesa responde ACEITO ou
REJEITADO com o motivo. Os testes (`mcp_server_test.exs`, `logica_test.exs`)
mostram a coloração certa aceita e a mesma com **uma** cor trocada
rejeitada; a refutação inteira aceita e a truncada rejeitada.

## 3. Comparação com o que existe

- Provadores por RL sobre o Lean (o nível olímpico do estado da arte):
  propõem passos num assistente de provas geral. Aqui não há Lean nem
  modelo treinado; há quatro lógicas **decidíveis** em que tanto propor
  quanto conferir é completo. A separação proponente/verificador é a
  mesma; o alcance é menor e o veredito é sempre definitivo.
- Solvers SAT industriais (Kissat, CaDiCaL): ordens de grandeza mais
  rápidos. O formato da refutação (DRUP) é o mesmo que as competições de
  SAT exigem; o verificador daqui confere provas desses solvers também.

## 4. Limites honestos

- A conferência DRUP é em Elixir: R(3, 4) = 9 é refutado em segundos mas
  sua refutação leva ~2 minutos para conferir — fora da suíte rápida.
- Sem lógica de primeira ordem geral, sem aritmética (o Lean não está
  nesta máquina); Gröbner sobre ℚ, sem desigualdades (geometria real
  ordenada não é decidida aqui).
- Knuth–Bendix pode não terminar para uma teoria sem sistema finito
  convergente: há um limite de 4000 passos e a resposta diz que a completação não terminou (em vez de girar para sempre).

## 5. Aritmética linear exata: simplex racional com certificados (0.13)

```
maximize 3x + 2y
subject to
x + y <= 4
x + 3y <= 6
x <= 3
free z
```

`Vapor.Logic.LP` resolve programas lineares em **racionais exatos** (duas
fases, regra de Bland — não cicla), de modo que "ótimo", "inviável" e
"ilimitado" são decididos, não estimados. Cada veredito vem com o objeto
que o prova, achado resolvendo o **sistema alternativo** e conferido por
`LP.check/2`, que só multiplica e compara:

| veredito | certificado | a conferência |
|---|---|---|
| ótimo | o primal x e um dual y | x viável; y dual-viável (Aᵀy ≥ c com os sinais das linhas); cᵀx = bᵀy |
| inviável | o vetor de Farkas y | Aᵀy ≥ 0, y ≥ 0 nas linhas ≤, ≤ 0 nas ≥, bᵀy < 0 |
| ilimitado | um x viável e um raio d ≥ 0 | A·d com o sinal de cada linha e cᵀd > 0 |

A mesa de lógica decide (`maximize …`/`minimize …` na primeira linha) e
**confere propostas** pelo `logic_check`: `{"x": {…}, "y": […]}` para
otimalidade, `{"farkas": […]}` para inviabilidade, `{"x": …, "ray": …}`
para ilimitação — números como `"p/q"`. Seis LPs aleatórios conferem com o
HiGHS do SciPy a 10⁻⁹; uma proposta errada (um x que viola a segunda
restrição) é rejeitada com a linha.

O mesmo simplex decide a **arbitragem** (o teorema fundamental da
precificação é o lema de Farkas): [FINANCAS.md §8](FINANCAS.md). Isto
fecha o item "aritmética linear (Simplex com certificado de Farkas)" do
TODO da 0.12.
