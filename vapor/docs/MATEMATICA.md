# Matemática por máquina: provas com certificado (0.11)

> Pedido: "pondere sobre algo similar a AlphaProof para matemática
> (geometria, topologia, e além)". Escrutínio:
> [DIRETRIZ.md §14](DIRETRIZ.md). Testes: `prove_test.exs`. Console:
> *Descobrir → Matemática*.

## 1. O que AlphaProof e AlphaGeometry são, e o que se pode fazer aqui

AlphaProof é aprendizado por reforço sobre o Lean: um modelo de linguagem
propõe passos, o núcleo do Lean os confere. AlphaGeometry combina um
motor de dedução simbólica com um modelo de linguagem que propõe
**construções auxiliares**. O que os dois têm em comum, e o que importa,
é a divisão: **a busca propõe, um verificador formal decide**. Sem GPU e
sem pesos aqui, o vapor faz a parte que não precisa de rede treinada — o
verificador e uma busca simbólica completa para uma classe grande de
teoremas — e é honesto sobre o que falta (provas legíveis, construções
auxiliares aprendidas).

## 2. Geometria pelo método algébrico (`Vapor.Prove.prove/1`)

Uma figura é uma **construção**: parâmetros livres, e pontos definidos a
partir dos anteriores (ponto médio, interseção de duas retas, pé de uma
perpendicular, circuncentro, ponto racional do círculo unitário
`((1−t²)/(1+t²), 2t/(1+t²))`, ponto numa razão). Cada coordenada é então
uma **função racional exata** dos parâmetros, com coeficientes inteiros, e
cada afirmação (colineares, paralelas, perpendiculares, comprimentos
iguais, concíclicos, o mesmo ponto) é um polinômio nas coordenadas. O
teorema vale para toda figura em posição geral **se e só se o numerador
da afirmação é o polinômio nulo** — a família dos métodos de Wu (1978) e
de bases de Gröbner, sem precisar triangularizar porque as construções
são explícitas.

- **Certificado**: o numerador expandido (zero) e as **condições de não
  degenerescência** — os denominadores que apareceram (por exemplo, para a
  reta de Euler: `u²v² + u²w² − 2uv³ − 2uvw² + v⁴ + 2v²w² + w⁴ ≠ 0`, o
  triângulo não degenerado).
- **Conferência em outra aritmética**: a mesma construção em **racionais
  exatos** em pontos inteiros sorteados até 10⁹ (Schwartz–Zippel: um
  polinômio não nulo de grau *D* se anula num ponto sorteado de um
  conjunto de tamanho *N* com probabilidade ≤ D/N). É independente da
  álgebra de polinômios, **não das fórmulas das construções** (o mesmo
  código as monta nas duas aritméticas): um erro numa fórmula — o pé da
  perpendicular, digamos — passaria pelas duas. O que o pega são os
  teoremas conhecidos (uma fórmula errada derruba a reta de Euler) e os
  controles falsos.
- **Construções degeneradas**: uma figura que é 0/0 para todo valor dos
  parâmetros (o pé de uma perpendicular sobre a "reta" de um ponto só) é
  relatada como `{:degenerate, :construction}`, nunca "provada" (testado).
- **Medido**: medianas, alturas e mediatrizes concorrentes, **reta de
  Euler**, a razão 1 : 2 do baricentro em OH, **círculo dos nove pontos**
  (duas formas), base média, Varignon, Tales, **Pappus**, **Simson** —
  provados; os **controles**, enunciados falsos de mesma forma (baricentro
  no circuncírculo, ortocentro = circuncentro, base média perpendicular,
  Simson fora do círculo, o círculo dos nove pontos por um vértice) —
  refutados pela prova simbólica **e** pela conferência. Tudo em < 0,1 s,
  exceto Simson simbólico (~100 s: o crescimento dos graus sem MDC de
  polinômios — conferido só em racionais exatos nos testes).

## 3. Conjecturar e provar (`Vapor.Prove.discover/2`)

O que AlphaGeometry chama de "dedução": sem que nada seja perguntado,
toda trinca de pontos da figura de um triângulo (vértices, pontos médios,
pés das alturas, baricentro, ortocentro, circuncentro, centro dos nove
pontos) é testada para uma reta e toda quadra para um círculo, em ponto
flutuante em duas figuras sorteadas; as trincas **triviais** (três pontos
de uma reta de definição) são descartadas; o resto é **provado
simbolicamente** (numerador nulo, como em §2) e conferido em racionais
exatos. Medido: de 1 001 candidatos, 29 sobrevivem e 29 são provados
(~7 s) — entre eles a **reta de Euler** (O, G, H e o centro N), a
**terceira mediana** e a **terceira altura**, e os **15 quadriláteros** do
círculo dos nove pontos, além de círculos menos famosos (B, C e os pés
das alturas de B e C; B, os pontos médios de AB e BC, e O).

## 4. Topologia

**Homologia** (`betti/2`): números de Betti de complexos simpliciais
sobre GF(2) (eliminação com bitsets) e sobre ℚ (eliminação de Bareiss,
inteiros exatos), e a característica de Euler. Medido:

| complexo | χ | GF(2) | ℚ |
|---|---|---|---|
| esfera | 2 | 1 0 1 | 1 0 1 |
| toro | 0 | 1 2 1 | 1 2 1 |
| garrafa de Klein | 0 | 1 2 1 | **1 1 0** |
| RP² | 1 | 1 1 1 | **1 0 0** |
| faixa de Möbius | 0 | 1 1 0 | 1 1 0 |

Sobre GF(2) o toro e a garrafa de Klein parecem iguais; sobre ℚ não — a
diferença é a **torção** ℤ/2, e é assim que a máquina distingue as
superfícies não orientáveis.

**Homologia persistente** (`persistence/2`): Vietoris–Rips até
triângulos, redução de colunas sobre GF(2). Um laço com ruído tem **uma**
barra H₁ longa (1,42); uma mancha de pontos sorteados (o controle), barras
de no máximo 0,13 — a ferramenta de análise topológica de dados, com o
controle que separa estrutura de ruído.

## 5. Limites

- Geometria de **igualdades** (incidência, perpendicularidade,
  comprimentos, círculos); desigualdades e ângulos orientados fora. As
  construções precisam ser explícitas (um ponto definido por duas
  condições quadráticas — dois círculos — exigiria a triangularização de
  Wu completa).
- Nenhuma prova **legível** passo a passo (o que AlphaGeometry produz); o
  certificado é algébrico.
- Sem Lean nesta rodada (o `lake` não está nesta máquina): as provas
  geométricas não são exportadas para um assistente de provas.
- Teoria dos números, análise, combinatória: fora. "E além" foi
  ponderado (DIRETRIZ §14) e ficou no TODO com o caminho — exportar
  certificados para o Lean e uma busca guiada por modelo quando houver
  pesos.
