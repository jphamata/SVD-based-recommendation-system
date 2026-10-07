# Geometria da informação — onde há uma variedade para medir

> `Vapor.InfoGeom`, `Vapor.Assay.Geometry` (`vapor assay geometry`). Testes: `info_geom_test.exs`;
> §5l. As críticas e o escopo: [DIRETRIZ §19](DIRETRIZ.md), crítica 1.

## Onde entra, onde não entra

A geometria da informação mede **distribuições**: saídas de modelos, estimadores, conjuntos. Ali há
uma variedade estatística, e a métrica de Fisher é a única invariante por reparametrização
(Čencov). Onde **não** há distribuição — o custo de um compilador sobre alocações de registradores,
a superfície de um *kernel* — o espaço é um reticulado discreto, e geodésicas não significam nada:
o vapor não as usa ali.

E não se roda uma variedade no silício: roda-se a **fórmula fechada** que a geometria entrega.

## O que há

No simplex, `p ↦ 2√p` leva a métrica de Fisher à métrica redonda de uma esfera. Tudo segue daí:

| função | o quê | custo |
|---|---|---|
| `fisher_rao(p, q)` | `2·arccos Σ√(pᵢqᵢ)`: a distância geodésica — **uma métrica** (simétrica, desigualdade triangular), limitada por π | uma soma e um arco-cosseno |
| `geodesic(p, q, t)` | a interpolação pelo grande círculo (*slerp* nas raízes) | O(k) |
| `frechet_mean(ps)` | a média de Karcher: o *ensemble* geométrico, que continua uma distribuição e comuta com renomear as classes | iterações de ponto fixo |
| `gaussian_fisher_rao(μ₁, σ₁, μ₂, σ₂)` | entre normais a métrica é hiperbólica (semiplano de Poincaré em (μ/√2, σ)): forma fechada | O(1) |
| `natural_logistic(X, y)` | regressão logística por passos de gradiente natural (a matriz de Fisher `Xᵀ diag(p(1−p)) X`) | Newton |
| `kl`, `js`, `hellinger` | os controles e os vizinhos | O(k) |

## Medido, com controles (§5l)

- **Métrica**: em 1 000 triplas aleatórias de distribuições, Fisher–Rao viola a desigualdade
  triangular **0** vezes; KL, **217**. Com KL, "A está mais perto de B que de C" não quer dizer
  nada.
- **Invariância**: uma variável reescalada ×1000 — o gradiente natural dá as mesmas previsões
  (maior diferença 7·10⁻¹²); o gradiente simples (o controle), não (0,54).
- **Normais**: com médias iguais a distância é `√2·|ln(σ₂/σ₁)|`; a mesma diferença de médias pesa
  mais quando σ é pequeno — o que a distância euclidiana nos parâmetros não vê.

## No Assay

`vapor assay geometry` recebe uma linha por (modelo, item) com as probabilidades previstas (CSV
`model,item,<classe>,…` ou JSON lines) e devolve a distância média de cada par com intervalo
*bootstrap* sobre os itens, a distância de cada modelo ao consenso geométrico (a média de Fréchet,
item a item) — o *outlier* é o modelo em que os outros não acreditam — e duas conferências: a
desigualdade triangular em toda tripla e um **controle**: os mesmos modelos com os itens
embaralhados. Dois modelos só estão "perto" se estiverem mais perto que modelos que respondem a
perguntas não relacionadas. [ASSAY.md](ASSAY.md).
