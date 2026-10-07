# Física para aprendizado por reforço e gêmeos digitais (0.10)

> Pedido: "pondere sobre motor de física no vapor" — no sentido de
> *reinforcement learning* e *digital twins*. Escrutínio:
> [DIRETRIZ.md §13](DIRETRIZ.md). Testes: `physics_test.exs`. Console:
> *Simular → Física*.

## 1. Por que um motor de física dentro de um compilador de tensores

Simuladores discordam de si mesmos: o mesmo modelo, semente e ações dão
outra trajetória noutra GPU, com outro número de *threads* ou outra versão
da biblioteca, porque somas em ponto flutuante são reordenadas. Para um
sistema suave é um incômodo; para um **caótico** — um pêndulo duplo, um
robô que anda, um escoamento — é total: duas execuções se separam em
segundos. Daí as duas dores:

- **RL não se reproduz.** Uma curva de treino, um episódio "bom", uma
  falha rara: nenhum se refaz noutra máquina.
- **Um gêmeo digital não se audita.** "O gêmeo previu X no instante t" só
  vale se outra pessoa, com o modelo e as entradas, recalcula X.

No vapor, o passo de um mundo é um **programa** como qualquer outro: roda
sob a semântica canônica e tem **os mesmos bits em todo substrato** — o
oráculo exato, os workers nativos, as GPUs admitidas —, para qualquer
lote de mundos.

## 2. O método (`Vapor.Physics`)

Dinâmica por posições estendida (XPBD; Macklin, Müller & Chentanez 2016)
com muitos subpassos e uma passada de restrições cada (Müller et al.
2020): partículas com massa inversa, **hastes** (restrições de distância
com complacência), **trilhos** (uma coordenada fixa — o carro no trilho),
**chão** e **atuadores** (forças vindas da ação). O grafo de restrições é
uma matriz de incidência com sinal `D` (haste c: +1 em i, −1 em j), e uma
passada de Jacobi sobre todas as hastes de todos os mundos são dois
produtos de matrizes:

    d = p·Dᵀ                         vetores das hastes, todos os mundos
    λ = −(|d| − ℓ) / (wᵢ + wⱼ + α/h²)
    Δp = ((λ/|d|)·d)·D ∘ w ∘ 1/grau  a parte de cada partícula

— operações `linear` que a eclusa já certifica. Mundos são linhas
(`f32[B, N]` por eixo): mil carros-pêndulo são uma execução.

**Achado (0.10):** calcular a velocidade como `(p − p_antes)/h` perde três
dígitos em precisão simples quando `h` é pequeno (cancelamento): o período
do pêndulo **piorava** acima de 8 subpassos. A velocidade agora é
`v + (correções)/h` — sem cancelamento —, e o erro cai pela metade a cada
subpasso dividido por dois, como deve num método de primeira ordem.

## 3. O que está medido

| medida | valor | controle |
|---|---|---|
| período do pêndulo (θ₀ = 0,5) contra o exato `4√(L/g)·K(sen θ₀/2)` | erro 6,5·10⁻⁴ → 3,4·10⁻⁴ → 1,7·10⁻⁴ → 8,5·10⁻⁵ (4 → 32 subpassos) | primeira ordem: cada metade de `h`, metade do erro |
| pêndulo duplo, oráculo × nativo | **bit a bit** | o mesmo mundo a **um ulp** de distância: separado (> 10⁻²) em 20 s |
| gradiente da trajetória (`Vapor.Autodiff`) | = diferenças finitas a 3 % | — |
| identificação do gêmeo: haste e amortecimento a partir de medidas com ruído de 2 mm | **1,0002 m (verdade 1) e 0,300 (0,3)** | as mesmas medidas embaralhadas no tempo: 0,72 m e 1,55 — nada recuperado |
| carro-pêndulo por busca aleatória (ARS, política linear, direções ±1) | **200/200** em 8 inícios nunca vistos | política nula: ~50 |
| gêmeo vigiando a planta (CUSUM dos resíduos) | alarme **18 passos** depois de a haste esticar 0,5 % (5 mm) | sem falha: nenhum alarme em 150 passos |
| livro do gêmeo (cadeia de *hashes*) | refeito do modelo e das ações: cada previsão recalculada | uma previsão forjada: pega na entrada exata (medidas e resíduos: só com uma âncora externa, §4) |

O carro-pêndulo do vapor é um sistema de partículas (carro num trilho,
massa na ponta de uma haste), não as equações do gymnasium; o CartPole
analítico do gymnasium continua em `Vapor.RL` (igual a ele a 10⁻¹²).

## 4. O gêmeo digital (`Physics.twin/2`, `observe/3`, `replay/1`)

O gêmeo roda o modelo ao lado da planta, passo a passo, com as ações da
planta; compara posições previstas e medidas (resíduo em unidades do
ruído do sensor) e acumula o excesso num **CUSUM** (Page 1954): um desvio
lento — uma haste que alonga, um rolamento que gasta — soma até cruzar o
limiar, onde um limiar por passo deixaria passar. Cada passo entra numa
**cadeia de hashes** `(t, ações, digest da previsão, digest da medida,
resíduo)`. Como a simulação é exata, quem tem o modelo e as ações refaz
as previsões e confere a cadeia (`replay/1`): o registro de "o que
esperávamos em t" não pode ser reescrito depois. **O que a cadeia sozinha
não garante:** as medidas e os resíduos (logo, a história dos alarmes)
entram só como *digests*; quem reescreve uma medida e recalcula a cadeia
inteira produz outra cabeça, igualmente válida. Para isso a cabeça precisa
de uma âncora externa — um *checkpoint* do log de transparência
(`Vapor.Tlog`) ou uma assinatura no instante —, ainda não ligada aqui.

E se o modelo do gêmeo estiver errado? `sysid/3` ajusta seus parâmetros
(comprimentos, amortecimento) às medidas da planta por **descida de
gradiente através do simulador** — o gradiente da trajetória inteira,
calculado como programa (`sysid_program/2`), também bit a bit em qualquer
substrato.

## 5. Limites (e o que fica para o [TODO](TODO.md))

- Partículas e hastes, 2-D e 3-D; **sem corpos rígidos com rotação**
  (inércia, juntas angulares), sem colisão entre corpos (só o chão), sem
  atrito. São o próximo passo natural (XPBD trata juntas e contatos do
  mesmo modo), e cada um precisa da sua medida contra uma referência.
- Jacobi com relaxação 1/grau: estável, mas converge mais devagar que
  Gauss-Seidel em cadeias longas; subpassos compensam.
- O ARS roda a política no BEAM entre passos (uma ida e volta por passo,
  pequena); embutir a política no programa tornaria o episódio inteiro uma
  só execução.
- Planta e gêmeo aqui são o mesmo simulador (com a falha injetada): o
  caso real — o gêmeo de uma máquina física — tem erro de modelo, e o
  limiar do CUSUM tem de ser calibrado nele.
