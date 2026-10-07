# Aludel — afirmações sobre polinômios, decididas em inteiros

> Desde 0.15.0. Código: `lib/vapor/aludel.ex`. Testes: `test/vapor/aludel_test.exs`.
> Console: *Opus → Aludel*. Terminal: `vapor aludel decide 'x^2 - x + 1/4' --vars x --box '0,1'`.
> MCP: `aludel_decide`. Absorvido do núcleo do PALADIN (anexo da rodada), reescrito sobre os inteiros da BEAM.

O aludel é o vaso em que uma substância volátil é *fixada*.

## A dor

Toda afirmação "controlador comprovadamente seguro" esbarra no mesmo ponto: uma verificação por
amostras não diz nada entre as amostras, e o ponto flutuante não diz nada perto de zero. A
pergunta "`p(x) ≥ 0` para todo `x` desta caixa?" precisa de uma resposta exata e conferível.

## O procedimento

- **Envoltória de Bernstein.** Numa caixa, `p` é combinação dos polinômios de Bernstein, que são
  não negativos e somam um; logo `p` fica entre o menor e o maior coeficiente de Bernstein, e os
  coeficientes nos vértices **são** `p` ali. Uma conversão exata da base de potências (racionais,
  depois inteiros sobre um denominador comum; por eixo, um deslocamento de Taylor e a
  transformada de Bernstein, sobre um arranjo denso).
- **Três vereditos, nunca dois.** Todos os coeficientes `≥ 0` (ou `> 0` na afirmação estrita):
  **certificado** na célula. Um coeficiente de vértice `< 0`: **refutado**, com o vértice — um
  ponto exato e o valor. Senão a célula é dividida no meio do eixo mais largo pelo algoritmo de de
  Casteljau (só somas e deslocamentos inteiros). Uma célula que chega ao orçamento é
  **esgotada**: um veredito com nome e a célula, nunca uma espera maior e nunca um palpite.
- **A testemunha é a árvore de subdivisão** (um bit por célula, em profundidade). `check/4` a
  reproduz sem buscar e, em cada folha, recalcula os coeficientes de Bernstein **diretamente** do
  polinômio na caixa daquela folha — uma conta diferente das subdivisões da busca, com o mesmo
  teorema por trás. Testemunhas não se transferem: bits adulterados, outro polinômio, truncamento
  — recusados.

`enclose/3` dá um intervalo rigoroso de `p` na caixa (contém os valores exatos, conferido em
pontos racionais aleatórios).

## Certificados de barreira

Para um campo polinomial `ẋ = f(x)`, uma função `B` com `B ≤ 0` no conjunto inicial, `B > 0` no
inseguro e `λB − ∇B·f ≥ 0` no domínio prova que nenhuma trajetória do inicial chega ao inseguro
sem antes sair do domínio (Prajna & Jadbabaie, 2004). Cada condição é uma afirmação de
positividade numa caixa — o mesmo procedimento. O candidato vem da pessoa, de um modelo ou de
`synthesize/2`: um LP exato (`Vapor.Logic.LP`) sobre os coeficientes de `B` cujas linhas são
coeficientes de Bernstein, crescido por **geração de restrições** (só as linhas violadas entram).
Em todos os casos o candidato só é aceito pela mesma decisão. Quando o inseguro encontra o
inicial, não existe barreira, e o LP diz isso com um certificado de inviabilidade.

## Medido

| | |
|---|---|
| Motzkin + 1/1000 > 0 em [−2, 2]² (não negativo, não é soma de quadrados) | certificado em 159 células; testemunha reproduzida |
| Motzkin ≥ 0 (toca o zero em (±1, ±1)) | **esgotado** — dito, não fingido |
| 80 polinômios aleatórios | todo "certificado" vale em 60 pontos exatos aleatórios; todo ponto "refutado" é negativo |
| oscilador amortecido, `B = x² + y² − 1` | três condições certificadas; o campo instável `ẋ = x, ẏ = y`: refutado num ponto exato |
| síntese, campo não linear `ẋ = −x + y², ẏ = −y` | barreira achada pelo LP e aceita pela decisão |

## O que não é

Não é um modelo de uma aeronave, de um plasma ou de um córtex. Prova propriedades **do sistema
polinomial que recebe**; se esse sistema descreve o mundo é outra afirmação, e é dita como tal.
O custo da conversão é exponencial no número de variáveis (o grau por eixo entra como produto),
como no PALADIN: o lugar dele são sistemas de poucas variáveis.
