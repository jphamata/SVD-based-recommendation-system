# Cena viva, esboço → desenho, salvar e exportar (0.11)

> Pedido: "a partir de um esboço passar para modelos para desenhar algo com
> IA (fotorrealista, de arquitetura ou mesmo engenharia) […] animações
> (offline) ou loops infinitos com entropia e interação de cenas a partir
> de desenhos […] a partir de uma imagem complexa transformá-la num loop
> infinito interativo com NPC, 3D, profundidade, navegação livre pela cena,
> esqueletos e ações para entidades, efeitos (luz, gravidade e além) […]
> com direção e ajustes interativos via prompt e salvamento e exportação
> dos resultados (isso vale para tudo!)". Escrutínio:
> [DIRETRIZ.md §14](DIRETRIZ.md). Testes: `scene_test.exs`,
> `sketch_test.exs`, `archive_test.exs`, `console_test.exs`. Console:
> *Fazer → Cena viva*, *Fazer → Esboço*, *Confiar → Arquivos*.

## 1. O que é possível aqui, e o que não é

O pedido, levado ao pé da letra, é um gerador de mundos por IA: imagem →
profundidade aprendida → malha 3D → personagens com esqueleto aprendido →
síntese de vídeo. Cada uma dessas peças, no estado da arte, é uma **rede
treinada** (estimação monocular de profundidade, segmentação, *pose
estimation*, difusão de vídeo) — e esta máquina não tem GPU nem pode
baixar pesos. Fingir essas peças seria entregar ruído com cara de mágica.

O que se faz a partir de primeiros princípios — e é o que foi feito — é
uma cena **2,5D**: a imagem separada em camadas a profundidades
plausíveis, o que fica atrás de cada uma reconstruído, um chão onde se
pode andar, e um motor que dá vida a tudo isso com física simples, luz e
clima, **dirigido por palavras** e **reproduzível pela semente**. Onde uma
rede treinada entraria (profundidade, segmentação, geração), a interface
é a mesma: as camadas e suas profundidades são dados, editáveis à mão, e
substituíveis por um modelo quando houver um.

## 2. A análise (`Vapor.Scene.analyze/2`)

| etapa | método | o que garante |
|---|---|---|
| regiões | superpixels SLIC (Achanta et al. 2012) em CIELAB, fundidos sobre o grafo de adjacência por cor (as regiões pequenas primeiro) | regiões que respeitam bordas |
| céu | região lisa e clara (ou azulada) que toca o alto da imagem, **crescida para baixo** por regiões lisas de cor vizinha (um degradê de céu é várias regiões) | horizonte onde o céu acaba |
| profundidade | o **plano do chão**: um objeto está onde estão seus pixels mais baixos, e num chão visto da altura dos olhos a profundidade de um ponto na linha *y* abaixo do horizonte é ∝ 1/(y − y_h); o céu no infinito | ordem de profundidade coerente com a perspectiva — **heurística, dita como tal**, editável camada a camada |
| camadas | regiões agrupadas por log-profundidade (no máximo 6) | poucas camadas, bem separadas |
| o que está atrás | cada camada guarda seus pixels e preenche a faixa escondida atrás das camadas mais próximas por **push-pull** (Gortler et al. 1996) com puxada bilinear; a camada do fundo preenche tudo | mover a câmera revela continuação plausível, não buracos |
| chão caminhável | células das regiões de chão (as que tocam a borda de baixo) abaixo do horizonte | caminhos para A* |
| luz | centroide e cor dos 1 % de pixels mais claros | a lareira de uma sala, o sol de uma paisagem |

Medido em imagens desenhadas por um script com a geometria conhecida
(`priv/quality/scene`): horizonte da paisagem a 0,44 (verdade 0,417), céu
no infinito, chão por último e caminhável; na sala de guilda, a luz na
lareira (x = 0,20, quente). Uma paisagem de 640×420 analisa em ~5 s na
BEAM.

## 3. O motor (`SceneEngine`, no console e no HTML exportado)

Canvas 2D, sem bibliotecas. Cada camada é um **plano** a uma profundidade
*D*, vista por uma câmera (x, y, z) com projeção em perspectiva: mover a
câmera desloca cada camada por 1/(D − z) — paralaxe correta para planos.
O chão é desenhado em **faixas**, cada uma na profundidade da sua linha,
para que recue como um chão de verdade.

- **Habitantes** andam pelo chão por **A\*** na grade caminhável; a altura
  de quem está na linha *y* sai da geometria da foto tirada na altura dos
  olhos: **a cabeça fica no horizonte** (câmera a 1,6 m, pessoa a 1,7 m).
  Destinos por palavra ("até a porta" = o fundo do chão, "esquerda",
  "frente", "a luz"…).
- **Clima e efeitos**: chuva (gotas numa fatia de profundidades, com
  respingo onde a gota encontra o chão da sua profundidade), neve,
  neblina por profundidade (1 − e^{−D/6}), tempestade com relâmpagos,
  vento com rajadas que balança a vegetação (camadas verdes desenhadas em
  faixas deslocadas), tochas e velas com cintilação, brasas com empuxo e
  turbulência, fumaça, vaga-lumes, pássaros, borboletas, folhas.
- **Hora do dia**: cada camada é "graduada" uma vez por mudança —
  multiplicada pela cor da luz (amanhecer, dia, entardecer, noite) e
  enevoada pela profundidade, com o alfa preservado —, e o ciclo do dia
  regradua a cada meio segundo.
- **Entropia**: um único gerador (mulberry32) com a semente da cena e um
  passo fixo de 30 Hz; a entropia controla rajadas, hesitação dos
  habitantes, ruído de voo, cintilação. **Mesma cena, mesmas operações,
  mesma semente → o mesmo laço**, em qualquer máquina.

## 4. Desenhos que se mexem (`Vapor.Scene.rig/2`)

Traços escuros em papel claro → afinamento (Zhang & Suen 1984) → grafo do
esqueleto (nós pelo *crossing number*, para que uma escada de pixels não
vire junção; laços fechados cortados num pixel) → cada cadeia vira ossos
(RDP), orientados para fora da junção mais próxima do centro → uma malha
cobre a tinta e cada vértice é ligado aos dois ossos mais próximos. O
motor anima por **cinemática direta e skinning linear**, desenhando cada
triângulo da malha com a sua transformação afim: *acenar* (a cadeia cuja
ponta está mais alta), *andar* (cadeias para baixo balançam alternadas,
as outras em oposição), *dançar*, *respirar*, *balançar*.

Medido: o boneco de palitos de teste tem as **quatro** extremidades onde
foram desenhadas (a menos de 30 px). **O movimento vem da topologia do
esqueleto, não de saber o que o desenho mostra** — um "braço" é uma cadeia
que termina num ponto livre.

## 5. Esboço → desenho técnico e planta → 3D (`Vapor.Sketch`)

**Vetorização com restrições.** Cada cadeia do esqueleto é ajustada por
uma reta (mínimos quadrados totais), um círculo ou arco (ajuste algébrico
de Kåsa) ou quebrada em retas; depois **embelezada**: orientação a 0°,
45°, 90° quando a menos de 4°; quase-paralelas tornadas paralelas;
horizontais e verticais colineares alinhadas; cantos soldados no ponto
de mínimos quadrados das suas retas. As restrições achadas são listadas
— o que o desenho *queria dizer*, declarado. Saídas: **SVG** e **DXF R12**.

Medido (`priv/quality/sketch`, desenhado por script com tremor de mão):
um retângulo desenhado **2,2° torto** volta reto e fechado (o controle —
o mesmo ajuste sem restrições — continua torto); o círculo volta com
centro e raio a menos de 1 px; a hipotenusa quase a 45° vai a 45°; uma
reta livre a 29,5° fica onde está.

**Planta → 3D.** As retas são paredes; um vão entre paredes colineares é
uma **porta** (a largura corrigida pela espessura do traço, que o
afinamento encurta pela metade em cada ponta); os **cômodos** são as faces
do grafo planar das paredes (portas fechadas), achadas pela caminhada nas
meias-arestas; a escala vem da parede mais longa (8 m por padrão,
ajustável) ou de `scale:`. As paredes são extrudadas (2,7 m, 15 cm de
espessura, vergas sobre as portas a 2,1 m) num `Vapor.Geom.Mesh`
exportado em **glTF (GLB)**; o console tem um visualizador 3D próprio.
Medido: cômodos de **11,98 e 19,80 m²** (verdade 12 e 20), portas de
**0,90 e 1,00 m** (verdade 0,9 e 1,0), o GLB aberto pelo trimesh com 2,7 m
de altura e a planta como base.

**O fotorrealista** (esboço → render): o pipeline de Stable Diffusion do
vapor (img2img, inpainting) recebe o esboço como imagem inicial — com um
checkpoint que o usuário carrega. Nenhum vem junto; nada disso foi medido
aqui.

## 6. Direção por prompt (`Vapor.Scene.direct/1`)

Um vocabulário em português e inglês vira **operações** — clima e
intensidade ("chuva forte", "neve leve"), hora, vento, luzes, habitantes
e quantidades ("três pessoas", "muitos pássaros"), câmera ("orbite",
"aproxime"), entropia ("calmo", "caótico"), animação do desenho, destinos
("até a porta"), negações ("sem chuva"). **Toda palavra não entendida é
relatada, nunca adivinhada.** As operações formam o **roteiro** da cena,
salvo com ela. Um modelo de linguagem carregado poderia emitir o mesmo
esquema de operações sob a decodificação restrita por JSON Schema que o
vapor já tem — não ligado nesta rodada (TODO).

### 6.1 Habitantes com nome, no tempo (0.12)

Desde 0.12 a direção alcança **um habitante de cada vez e um instante**.
O prompt é dividido em orações (`,` `;` `then` `depois` `e então`); uma
oração que nomeia um habitante — criado no próprio prompt ("um cavaleiro
chamado Artur", "a guard named Ana") ou já na cena — vira operação sobre
ele: ir até um lugar, dizer uma fala entre aspas (balão), acenar,
dançar, sentar, pular, correr, parar, patrulhar, fugir, seguir outro;
um **pronome** ("ele", "she") refere-se ao último nomeado; um papel sem
nome com um verbo ("a guard patrols") cria um habitante com o nome do
papel; uma oração que começa por um tempo ("aos 3 s", "after 5 seconds",
ou um tempo sozinho seguido de vírgula) vira um **quadro-chave** que o
motor toca naquele instante.

```
a knight named Arthur walks to the door, then at 3s he says "hello" and waves; a guard patrols
→ spawn Arthur · Arthur goto door · (3 s) Arthur say "hello" · (3 s) Arthur wave · Guard patrol
```

No console, o inspetor de **habitantes** lista cada um (nome, cor,
comportamento), permite renomear, mudar velocidade, tamanho e cor, fazer
falar, mandar ir a um lugar e **traçar uma rota** clicando no chão; um
clique na figura a seleciona; a **linha do tempo** mostra os
quadros-chave. **GIF de quadros exatos**: o motor tem passo fixo e
semente, então os quadros são calculados fora do tempo real (não
gravados da tela) e codificados no servidor. Conferido em
`scene_test.exs` (alvo, tempo, pronome; o controle: uma frase sem nomes
não mira ninguém) e no Chromium sem cabeça (`console_desks.mjs`).

## 7. Salvar e exportar — tudo (`Vapor.Archive`)

Todo resultado do console tem **Salvar**: um zip com `manifest.json`
(tipo, versão, semântica, **receita**, SHA-256 de cada arquivo) e os
arquivos; a **identidade** é o SHA-256 do manifesto canônico. **Confiar →
Arquivos** confere um arquivo byte a byte e, se o tipo é determinístico
(redes de ordenação, multiplicação de matrizes, geometria, homologia,
ciência, a direção de uma cena — palavras → operações…), **recalcula a
receita e compara**; os demais (uma cena viva salva inteira, com as
camadas) são conferidos byte a byte contra o manifesto — o que pega
corrupção e edição descuidada, não uma mentira coerente (o manifesto não
é assinado; assinar arquivos com a chave de operador da 0.10 está no
TODO). Um arquivo é bytes não confiáveis: no máximo 512 entradas e
256 MB descomprimidos, contados **enquanto** se descomprime (uma bomba de
zip de 1 MB que diz 300 MB é recusada sem ser expandida — testado), e
todo parâmetro de receita é limitado antes de rodar (uma rede de
ordenação de 32 fios, 10⁶ partidas de autojogo: recusadas). Um arquivo
nomeia um *tipo*, nunca uma função: abrir um arquivo não executa nada que
ele escolha. Medido: o arquivo íntegro refaz `{:ok, :same}`; o mesmo com
um byte do resultado trocado e re-zipado é recusado como
`{:tampered, ["result.json"]}`; uma mentira coerente (manifesto refeito)
passa na conferência de integridade e é pega pelo recálculo.

A cena viva sai também como **um único HTML** que toca offline (motor,
camadas, roteiro e semente embutidos; ~130 kB para a sala de guilda) e
como **vídeo** gravado do canvas (WebM, o codificador do próprio
navegador). O esboço sai em SVG, DXF e GLB.

## 8. Limites

- Profundidade **heurística**: ótima para cenas com chão (paisagens,
  salas, ruas); uma foto de rosto, um teto, uma vista aérea não têm chão
  — as camadas existem, a ordem pode estar errada (os controles deslizantes
  existem para isso).
- Sem segmentação semântica: uma pessoa na foto não vira um habitante
  animável a menos que o usuário a traga como desenho; os habitantes são
  figuras do motor, coloridas com a paleta da cena.
- 2,5D, não 3D: a câmera se move numa janela pequena (o que está atrás é
  plausível, não verdadeiro); navegação livre "dentro" da cena pediria
  geometria que uma imagem só não tem.
- O esqueleto de um desenho vem dos traços: um desenho preenchido (uma
  silhueta) dá um esqueleto pelo eixo medial, nem sempre o anatômico.
- Vídeo: o navegador grava em tempo real (WebM); desde 0.12, GIF de
  quadros exatos (até 240). MP4 de quadros exatos segue no TODO.
- A direção entende uma gramática de orações, não linguagem livre: "Ana
  e Bento dançam" dirige o primeiro nome achado; frases subordinadas não
  são analisadas.

## 9. Cenas livres (0.14): um documento editado por operações

As cenas deixaram de depender de ações pré-determinadas. Uma cena é um documento JSON; ela muda
por **operações em texto** (`Vapor.Scene.Ops`), que qualquer um escreve — a pessoa, um programa
ou um modelo:

```
add circle sol { x: 0.7, y: 0.22, r: 0.06, color: "#E8B04A" }
add particles chuva { count: 200, x: fract(u + 0.05*t), y: fract(0.3*t + u*7) }
set sol.r = 0.06 + 0.01*sin(2*t)
at 4: remove chuva
set world.weather = rain
```

Tipos: `circle`, `ring`, `rect`, `line`, `text`, `glow`, `particles`, `trail`; `at SEGUNDOS:` agenda uma operação e `set world.…` muda o clima, a hora, o vento, a câmera. Qualquer campo
aceita uma expressão do subconjunto numérico de Alembic em `t` (tempo), `i`/`n`/`u` (índice,
total e fração nas partículas) e `aspect`; a expressão vira uma árvore interpretada no navegador
([ALEMBIC.md §3](ALEMBIC.md)) — nunca código. `direct "palavras"` pede ao modelo as operações e
elas passam pelo mesmo leitor; o que não for operação válida volta como problema, com a linha.

```
vapor scene new --w 960 --h 600 > s.json
vapor scene edit s.json "add glow sol { x: 0.7, y: 0.2 }" > s2.json
vapor scene direct s2.json "uma chuva fina caindo na diagonal" > s3.json   # com VAPOR_MIND
vapor scene export s3.json > cena.html                                    # HTML autocontido
```

