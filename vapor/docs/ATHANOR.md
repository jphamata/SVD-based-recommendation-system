# Athanor — a fornalha de busca, e a Touchstone que a confere

> Desde 0.14.0. Código: `lib/vapor/athanor.ex`, `lib/vapor/athanor/` (`space.ex`, `spec.ex`,
> `strategy.ex`, `gauss.ex`, `touchstone.ex`, `game.ex`, `session.ex`, `examples.ex`) e
> `lib/vapor/mind.ex`. Testes: `test/vapor/athanor_test.exs`, `mind_test.exs`,
> `workspace_test.exs`. Qualidade: `mix vapor.quality --only round14`.

O athanor é o forno do alquimista que mantém o fogo constante por muito tempo. Aqui ele é um
**buscador geral**: recebe um problema escrito em [Alembic](ALEMBIC.md) — um espaço e um
objetivo, ou uma afirmação — e devolve um **certificado** que qualquer um pode reconferir sem
confiar na busca. Não há categorias: Golomb, Ramsey, redes de ordenação, caixeiro-viajante,
portfólios, hiperparâmetros, exemplos adversariais, regras de *trading* e fórmulas são apenas
exemplos de partida (`Examples.all/0`), escritos na mesma linguagem que a pessoa usa.

## 1. Um problema

```
# a régua de Golomb com 7 marcas (o ótimo é 25)
space = subset(1..30, 6)
ruler(r) = [0] ++ r
violation(r) = let d = [b - a for (a, b) in pairs(ruler(r))] in len(d) - len(distinct(d))
minimize(r) = max(r)
target = 25
budget = 30000
```

Nomes reservados: `space`, `minimize`/`maximize`/`claim`, `valid`, `violation`, `margin`,
`target`, `budget`, `seed`, `start`, `show`, `describe`, `holdout`, `neighbor`, `measured`.
Espaços: `bits`, `ints`, `reals`, `perm`, `subset`, `subsets`, `seq`, `graph`, `partition` e
`program` (árvores de expressão — o candidato chega às suas funções como função).

## 2. A fornalha

- **Portfólio de estratégias** — exaustiva (retomável, quando o espaço cabe), aleatória,
  recozimento, evolução com arquivo MAP-Elites (`describe`), CMA-ES (espaços reais, com
  autovalores de Jacobi), bayesiana (processo gaussiano Matérn-5/2 + melhoria esperada, lote
  pelo *kriging believer*), **mente** (um modelo de linguagem propõe) e **humano** (a pessoa
  propõe). Um bandido UCB descontado (γ = 0,97) reparte o orçamento pelo que cada uma rende.
- **Inviáveis ordenados**: com `violation`, o candidato inválido ainda tem posto (−violação),
  e a busca sobe até a validade (Schur, Golomb 8).
- **Controle**: o mesmo orçamento gasto em amostras uniformes. Se o aleatório nunca alcança o
  melhor, o certificado dá o limite superior da chance por amostra (regra de três, 95 %).
- **Holdout**: com `holdout(x)`, os finalistas são reavaliados num objetivo que a busca não viu;
  a correlação de postos (Spearman) e o máximo-z do ruído (√(2 ln N)) dizem se o vencedor é
  sinal ou viés de seleção. Num passeio aleatório, regras de média móvel: ρ ≈ −0,33; com
  momento AR(1) plantado: ρ ≈ 0,73.
- **Diário**: cada avaliação entra numa cadeia SHA-256 sobre a codificação canônica; a raiz
  depende só do texto e da semente.

## 3. O certificado e a Touchstone

O certificado diz o melhor (candidato, valor, quem o achou), o motivo da parada (`exhausted`,
`target`, `counterexample`, `found`, `budget`, `time`), o veredito em palavras ("ótimo provado
pela enumeração", "afirmação provada sobre o espaço inteiro"), o controle, o holdout e as
propostas de fora. `Touchstone.verify/3` reconfere **sem confiar na busca**: reavalia o
candidato, confere o valor e a pertença ao espaço; com `full: true` refaz a enumeração; com
`replay: true` refaz a corrida e compara a raiz do diário. Um valor forjado é recusado com a
verificação que falhou.

```
vapor athanor run golomb.alb > cert.json      # código de saída 0/1 conforme o resultado
vapor verify golomb.alb cert.json --replay     # a pedra de toque
```

## 4. Humano e modelo no laço

- **Sessões** (`Athanor.Session`, sob `DynamicSupervisor`): a busca corre em segundo plano; a
  pessoa observa as faíscas, **propõe** candidatos (conferidos e avaliados como os outros),
  **fixa** e **bane** finalistas, estende o orçamento, para e retoma. No console, isso é a
  fornalha ao vivo; no terminal, `vapor athanor run --interactive`.
- **Medido** (`measured = true`): o objetivo está fora da máquina — um experimento de
  laboratório, um treino, uma pessoa. `vapor athanor ask` propõe, a pessoa mede e digita;
  `--measure 'comando'` mede por um programa externo. A estratégia bayesiana trabalha em lotes.
- **Mente** (`Vapor.Mind`, `VAPOR_MIND=anthropic:MODELO | openai:MODELO[@URL] | script:ARQ`):
  `formalize` traduz palavras em Alembic, compila, repara até três vezes com o erro do
  compilador e devolve uma **retrotradução** para a pessoa conferir o que foi entendido;
  `propose` sugere candidatos — que entram pela mesma porta que os de qualquer um. O modelo
  nunca decide: a Touchstone decide.

## 5. Jogos

Com `init`, `player`, `moves`, `play` e `winner`, qualquer jogo de dois jogadores:
`solve` (negamax com tabela de transposição — o jogo da velha é empate sobre 5 478 posições),
`search` (UCT/MCTS), `learn` (valor linear tanh(w·features) por autojogo) e `play`
(partida contra o humano), e `match` com intervalo de Wilson.

## 6. O que não é

Não é um provador de teoremas: "provado" aqui significa **enumeração completa de um espaço
finito**, e o certificado diz o tamanho. Espaços infinitos dão evidência, não prova. Não é
paralelo entre nós (ainda). O orçamento padrão é modesto; a fornalha é honesta sobre o que
não achou.
