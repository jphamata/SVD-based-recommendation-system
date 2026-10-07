# Redes complexas (0.10)

> Pedido: "pondere também acerca de aplicações de redes complexas (teoria da
> complexidade etc.)". Escrutínio: [DIRETRIZ.md §13](DIRETRIZ.md). Testes:
> `graph_test.exs` (com o networkx, quando presente, como segunda opinião).
> Console: *Simular → Redes*.

## 1. As duas dores

**Numérica.** Detecção de comunidades, amostragem, *bootstraps* e epidemias
são aleatórios e dependem da ordem: uma modularidade ou um veredito de
"livre de escala" muda de uma execução para outra e de uma máquina para
outra.

**Estatística.** Por anos, uma reta aproximada num gráfico log–log de
graus foi tomada por lei de potência. Quando os testes de Clauset, Shalizi
& Newman (2009) foram aplicados a quase mil redes reais, as fortemente
livres de escala se mostraram raras (Broido & Clauset 2019). Uma métrica
estrutural sem modelo nulo não diz nada: o agrupamento de uma rede só
interessa pelo quanto excede o de uma aleatória **com os mesmos graus**.

## 2. O que `Vapor.Graph` faz

Toda escolha aleatória é um sorteio por contador `(semente, contador)`
(`Vapor.Sampler`) e os logaritmos que transformam sorteios em saltos são
corretamente arredondados (`Vapor.CR`): um gerador, um nulo, uma amostra
de *bootstrap* ou uma epidemia é função da semente, em qualquer máquina.
E toda afirmação vem com seu controle:

| ferramenta | o que mede | o controle |
|---|---|---|
| `power_law/2` | lei de potência discreta por máxima verossimilhança, `x_min` pela distância de Kolmogorov–Smirnov, p-valor por *bootstrap* semi-paramétrico, razão de verossimilhança contra a exponencial (teste de Vuong) e um **veredito** | Erdős–Rényi (Poisson): não pode ser chamado de livre de escala |
| `rewire/3`, `zscore/4` | o nulo de configuração (trocas duplas de arestas que preservam cada grau) e o escore-z de qualquer estatística contra ele | um grafo aleatório tem z ≈ 0 |
| `communities/2` | Louvain determinístico (ordem dos nós fixa, empates pelo menor índice), modularidade, NMI | o nulo da partição plantada |
| `sir/3`, `threshold/1` | SIR em tempo discreto; o limiar de campo médio heterogêneo `T_c = ⟨k⟩/(⟨k²⟩ − ⟨k⟩)` | abaixo do limiar, surtos morrem |
| `percolation/4`, `giant/2`, `er_giant/1` | componente gigante após falhas aleatórias ou ataque aos de maior grau; `S = 1 − e^{−cS}` | — |
| `pagerank/2`, `pagerank_program/3` | PageRank em binary64 em ordem fixa; e como **programa vapor** (denso, f32) — os mesmos bits em todo substrato | o *ranking* do programa = o do hospedeiro |
| `clustering/1`, `assortativity/1`, `betweenness/1`, `avg_path_length/1` | — | = networkx (10⁻⁹) |

## 3. Medido

- **Barabási–Albert** (n = 1000, m = 3): α = 2,66, *bootstrap* p = 0,78,
  razão contra a exponencial muito favorável → **lei de potência**.
  **Erdős–Rényi** com a mesma densidade: a exponencial ajusta
  significativamente melhor → **não** é livre de escala. (Sem o teste da
  razão, o KS sozinho aceitaria uma "lei de potência" com α = 8 na cauda
  da Poisson: por isso o veredito exige as duas coisas.)
- **Mundo pequeno** (Watts–Strogatz, n = 500): agrupamento 0,56 contra
  0,017 do nulo, **z ≈ 400** (405 com a semente do teste); um grafo aleatório da mesma densidade: |z| < 3.
- **Comunidades plantadas** (4 × 50): NMI = **1,0**, modularidade 0,57
  contra a do nulo bem menor.
- **Componente gigante** de G(n, c/n): o simulado bate `S = 1 − e^{−cS}` a
  0,03 para c = 1,5, 2, 3; nada para c = 0,5.
- **Epidemias**: com transmissibilidade 0,5·T_c os surtos morrem (< 5 %);
  com 2,5·T_c atingem mais de 30 %.
- **Robustez** (Albert, Jeong & Barabási 2000): uma rede livre de escala
  perde quase nada com 15 % de falhas aleatórias e se parte sob ataque
  aos de maior grau.
- **Achado (0.10):** a troca dupla de arestas que sempre liga as
  extremidades "menores" às "maiores" explora só parte do nulo — um
  Erdős–Rényi tinha z = 20 contra o próprio nulo. A orientação da segunda
  aresta agora é sorteada. E a propagação de rótulos (tentada primeiro)
  inunda uma partição plantada numa única comunidade quando a ordem de
  visita não é aleatória — daí o Louvain determinístico.

## 4. Onde isto encontra o resto do vapor

- **RAG por grafo**: o PageRank pessoal sobre o grafo de citações de uma
  biblioteca (`Vapor.Docs.Library`) é a ideia do HippoRAG; com o
  PageRank como programa, o *ranking* é reprodutível em qualquer
  substrato. (Ligar os dois está no [TODO](TODO.md).)
- **Gêmeos digitais de redes**: o SIR é a dinâmica; `Vapor.Physics.Twin`
  é o padrão de vigilância (resíduos, CUSUM, livro verificável).
- **O próprio cluster**: a robustez a falhas contra ataques é a pergunta
  de onde colocar réplicas e auditorias (`Vapor.Cluster`).

## 5. Limites

- Pensado para redes de até alguns milhares de nós: caminhos mínimos de
  todos os pares, intermediação e o PageRank denso são quadráticos ou
  piores. Redes de milhões de arestas pedem operadores esparsos no
  compilador.
- As estatísticas do teste de lei de potência usam `pow`/`exp` da
  plataforma (idênticos numa plataforma; a um ulp entre plataformas — uma
  comparação do *bootstrap* só mudaria num empate exato).
- Sem redes reais embarcadas (os dados públicos clássicos não estão nesta
  máquina); os geradores e os nulos são os controles.
