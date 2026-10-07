# Any-to-any sem kernels novos

> Desde 0.9: as rotas de mídia se compõem num **estúdio** de nós tipados (imagem, som, vídeo, 3D, difusão, RL), com cache exato e raiz de Merkle por execução, e o Stable Diffusion entra igual ao diffusers. Ver [ESTUDIO.md](ESTUDIO.md).

> Toda modalidade é uma sequência de linhas. Um modelo só precisa saber
> combinar linhas.

## 1. A pergunta certa

"Any-to-any" costuma ser respondido com mais um runtime por modalidade:
ComfyUI para imagem, whisper.cpp para áudio, llama.cpp para texto — cada um
com seus operadores, seus formatos, sua numérica e nenhuma garantia comum. A
pergunta de primeiros princípios é outra: **o que, minimamente, um modelo de
outra modalidade exige da álgebra?** Para o vapor isso importa em dobro,
porque cada operador novo custa cinco emissores (x86, AVX-512, NEON, RVV,
SPIR-V), uma semântica no oráculo e um degrau na escada de verificação.

A resposta desta rodada: **nada novo**. Cada peça abaixo é um termo da álgebra
existente, então herda paridade bit a bit em todo substrato e o certificado.

| peça | o que parece exigir | o que de fato é |
|---|---|---|
| *patch embedding* (ViT, CLIP, SigLIP) | `Conv2d` | uma convolução com *stride* = *kernel* é `linear` sobre as linhas que o codec corta; o peso `[d, C, p, p]` vira `[d, C·p·p]` (`Vapor.Modal.Image.patches/3` produz a ordem `(c, i, j)` do produto escalar da convolução) |
| atenção bidirecional (encoders) | uma máscara nova | a atenção causal do vapor deixa a linha `t` ver `0 … pos[t]`; com `horizon[t] = n − 1` para toda linha, cada uma vê as `n` reais e nenhuma de *padding* — **bidirecional é causal com horizonte no fim** |
| [CLS] | concatenação | `sel(máscara, ½, x, cls)` com a linha 0 do codec zerada |
| atenção cruzada | um operador novo | atenção sobre as K/V de outro fluxo como "cache", com horizonte no fim (expressável; não exercitada aqui) |
| espectrograma (Whisper) | FFT | `P = (X·Cᵀ)² + (X·Sᵀ)²`, tabelas de Hann com `cos`/`sin` corretamente arredondados (`Vapor.CR`) — duas contrações e três operações elemento a elemento; banco mel = mais um `linear` |
| quantização vetorial (tokenizadores de imagem/áudio) | argmin | `argmin‖x − c‖² = argmax(2x·c − ‖c‖²)`: `linear`, uma linha constante e o operador `sample` no modo guloso (empates para o menor índice) |
| decodificar códigos | — | `gather_row(codebook, codes)` — exato |
| injetar outra modalidade num decoder (LLaVA) | concatenar embeddings | opção `inject: true` do decoder: `x = sel(soft_mask, ½, embed[tok], soft)` — **seleção**, então uma linha substituída não carrega nada do token, nem um NaN |
| síntese de áudio | vocoder | síntese aditiva: amplitudes por quadro → quadros por um `linear` contra uma base de senos |
| GELU exata (ViT, BERT, Whisper) | `erf` | microprograma canônico novo `:gelu` sobre `+ − ×` (Chebyshev do `erfcc`, ordem do PyTorch), erro absoluto ≤ 4,4·10⁻⁷; como todo microprograma, é expandido por todos os emissores sem código novo — e fecha o item 4.1 do TODO |

## 2. Hub, não matriz

Com `N` modalidades, conversores par a par são `N·(N − 1)`. Com um **pivô**,
são `N` codecs: cada modalidade diz como chegar ao pivô e como voltar
(`Vapor.Modal.Hub.register/3`), e a rota `a → b` é
`from_pivot_b ∘ traduzir ∘ to_pivot_a`. Rotas diretas (a ida e volta de um codec,
a injeção de *soft tokens*) são registradas ao lado e preferidas. Acrescentar
uma modalidade é **um** codec, e ela passa a conversar com todas as outras.

O pivô aqui é **texto** — palavras que um decoder lê e escreve — porque é onde
estão os modelos mais capazes. Cada passo é um programa construído pela eclusa
e executado num substrato (`Vapor.Modal.Runner`).

## 3. Medido, não demonstrado

Uma demonstração escolhe o exemplo que funciona. Uma medição tem verdade de
referência, controle e dados retidos. `Vapor.Modal.World` é um mundo pequeno
e **totalmente especificado** em três modalidades: quatro cores ↔ quatro notas
(bijeção), cenas 16×16 com um disco à esquerda e um à direita, tons de 1024
amostras a 8 kHz. Quatro dos dezesseis pares de cores **nunca entram em nenhum
ajuste**; toda imagem de treino é uma variante com *jitter* de posição e raio,
iluminação ±8 % e ruído de sensor, de modo que nenhum *patch* se repete e
nenhum ajuste acerta por memorização (a primeira versão do mundo não tinha
isso e dava PSNR de 155 dB — medindo recordação, não generalização; foi
endurecida antes de qualquer número ser publicado).

Todos os codecs são ajustados **em forma fechada**: pontes lineares por ridge
(equações normais, forma primal ou dual, Cholesky em binary64), codebooks por
k-means determinístico (inicialização por travessia do ponto mais distante),
tradutores por contagem (`Vapor.Quality.Planted`).

Resultados em entradas retidas (ver [bench/QUALITY.md](bench/QUALITY.md),
regenerado por `mix vapor.quality`; a galeria PNG/WAV está em `bench/modal/`):

| rota | medida | valor | controle |
|---|---|---|---|
| imagem → texto | acurácia da legenda (pares inéditos, variantes, ruído σ = 0,05) | 1,00 | acaso 0,25 |
| texto → imagem | PSNR contra a cena canônica | 23,7 dB | 12,8 dB (legenda trocada) |
| imagem → imagem | ida e volta VQ | 23,0 dB | 11,3 dB (codebook aleatório) |
| áudio → texto | acurácia da nota (fase/amplitude inéditas, 10 dB SNR) | 1,00 | 0,25 |
| texto → áudio | erro de altura | ≤ 1,5 % | — |
| áudio → áudio | ida e volta VQ | 17,3 dB SNR | 0 dB |
| imagem → áudio, áudio → imagem | via pivô (com tradução cor ↔ nota) | 1,00 / 23,8 dB | — |
| imagem → *soft token* → decoder | projetor injetado como linha do decoder | 1,00 | 0,25 |

Todos os programas modais (codificar/decodificar VQ, ponte, espectro,
síntese) dão **os mesmos bits** no worker nativo e no oráculo.

## 4. O que isto é, e o que não é

**É**: a prova, executável e medida, de que a álgebra certificada do vapor
carrega imagem e áudio de ponta a ponta sem operador novo; de que a eclusa
recebe topologias novas (encoder, codec, projetor, perceptron) sem tocar o
núcleo; de que ViT e CLIP-vision do Hugging Face são admitidos e computados
como o `transformers` os computa (desde 0.5.0, contra ele mesmo); e de um
*harness* que mede qualquer rota contra verdade e controle — agora também em
**dados reais** (§6).

**Não é**:

- **difusão fotográfica**. A difusão existe desde 0.5.0 (§6) e é verificada;
  desde 0.6.0 o VAE (`AutoencoderKL`) e o DiT do diffusers também
  ([ESPACIAL.md](ESPACIAL.md)) — mas conferidos com pesos aleatórios:
  qualidade fotográfica exige pesos treinados, que não estão aqui;
- **todos os modelos multimodais pré-treinados**. ViT, as duas torres do
  CLIP e o Whisper são conferidos contra o `transformers` (0.6.0); SigLIP e
  LLaVA têm o caminho desenhado mas não adaptadores;
- ~~convoluções sobrepostas no meio da rede~~ — desde 0.6.0,
  `Vapor.Spatial` (gather + sel + reshape + GEMV, sem kernel de convolução);
  o *stem* do Whisper é feito assim;
- **jogos, renderização, upscaling**: fora do escopo, pelas razões do §5.

## 5. Fora do escopo, por princípio

- **Renderização gráfica de jogos**: o `vapor-fabric` é Vulkan *compute*
  headless; rasterização não é um problema de tensores certificados.
- **ComfyUI como produto**: um editor de grafos de difusão é uma interface; o
  que o vapor oferece é o *motor* — programas verificados — que uma interface
  dessas poderia chamar.
- **Upscaling** (Real-ESRGAN, SwinIR): convoluções densas sobrepostas em
  resolução cheia; expressáveis, mas sem um caso que justifique o custo antes
  de existirem kernels de convolução com memória de *workgroup* (TODO 2.3).

## 6. Em dados reais (0.5.0)

O mundo de teste prova que a pilha carrega sinal; não diz nada sobre sinais
de verdade. Esta seção mede rotas sobre dados reais **retidos**, com modelos
pequenos treinados pelo PyTorch e admitidos pela eclusa como qualquer
checkpoint (`priv/digits`, `priv/speech`, `priv/ocr`; scripts em
`test/python/train_*.py`). Números regeneráveis por `mix vapor.quality`
([bench/QUALITY.md §4c](bench/QUALITY.md)).

| rota | dados | modelo | medida | controle |
|---|---|---|---|---|
| caligrafia → dígito | UCI/scikit-learn, 497 dígitos retidos | `vapor_mlp` 64→128→128→10 | acerto **0,980** | acaso 0,10 |
| dígito → caligrafia | — | `vapor_mlp` denoiser + DDIM (25 passos, guia 2) | 50 gerados, lidos de volta: **1,000** | acaso 0,10 |
| novidade do gerado | distância à imagem de treino mais próxima (mediana) | — | **18,6** níveis de cinza | reais retidos 16,7; memorização 0,0 |
| fala → dígito | FSDD, 100 gravações de uma voz nunca ouvida | `vapor_encoder` sobre o espectro mel certificado | acerto **0,900** | acaso 0,10; invertida no tempo 0,84¹ |
| voz → texto → desenho → leitura | 10 gravações retidas | os três acima em cadeia | **0,900** | acaso 0,10 |
| imagem de texto → texto | 5 fontes fora do treino; foto real de página | `vapor_encoder` + CTC ([OCR.md](OCR.md)) | CER **0,068** (Tesseract 0,052); página **0,117** (Tesseract 0,364) | texto fluente errado 1,06 |

A distância mediana do gerado ao treino (18,6) é a de um dígito real retido
(16,7), não a de uma cópia (0): o gerador produz dígitos novos e legíveis.
¹ A fala invertida no tempo ainda é lida em 84 % dos casos: um dígito falado
se reconhece sobretudo pelo timbre das suas vogais, não pela ordem dos sons.
Fica registrado como achado sobre o problema, não como controle — o controle
é o acaso.

**Difusão, verificável.** O amostrador DDIM é testado antes de qualquer rede
contra o denoiser **ótimo em forma fechada** de uma mistura gaussiana: os modos
caem nos seus pesos e nas suas variâncias. Para um conjunto finito de pontos,
esse denoiser é **atenção** (consulta = imagem ruidosa, chaves e valores = as
imagens de treino) — e reproduz cópias do treino: a memorização dos livros,
que vira o controle da métrica de novidade. Foi assim que um bug real do
DDIM com *clamp* apareceu: sem recalcular ε̂ a partir de x̂₀ já limitado, o par
fica inconsistente nos primeiros passos (ᾱ ≈ 0) e só 17 % dos dígitos
gerados eram lidos certo; com a correção, 99 %.

**Fala, certificada.** O front-end (`Vapor.Modal.Speech`: quadros de 32 ms,
espectro de Hann e banco mel como **um programa** da álgebra, log e média por
banda em binary64 correto) dá os mesmos bits em todo substrato; o leitor foi
treinado em cinco vozes do Free Spoken Digit Dataset e medido na sexta,
escolhida antes do treino. Essa voz é a mais fácil das seis: com cada voz
retida por vez (`train_speech.py --loso`), o acerto médio é **0,72** (de 0,59 a
0,90: george 0,59, nicolas 0,62, lucas 0,73, yweweler 0,73, jackson 0,79,
theo 0,90). Cinco vozes de treino são pouco para generalizar a qualquer voz;
o número a citar é o médio, e o 0,90 é o da voz embarcada.
