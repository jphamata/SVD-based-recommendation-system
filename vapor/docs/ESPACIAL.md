# Espacial, latente e áudio: convolução, VAE, DiT, atenção cruzada, Whisper

O segundo anexo da rodada perguntou o que faltaria ao vapor para chegar a
geradores de imagem e vídeo (Sora, Midjourney, modelos de mundo). A resposta
desta página é a parte que cabe a um runtime: os operadores e as topologias
que esses modelos usam, **sem um único kernel novo**, conferidos contra as
implementações de referência (torch, diffusers, transformers). A parte que
não cabe — dados, treino, computação — está dita no fim.

## 1. Convolução sem kernel de convolução

Uma imagem de `H×W` pixels e `C` canais é a tabela `f32[H·W, Cp]` (pixels em
linhas, canais nas colunas, `Cp` = `C` arredondado a 16, colunas extras
exatamente zero). Então

    conv(x) = reshape(sel(pad, ½, 0, gather_row(x, idx)), [n, k·C])·Wᵀ + b

- `idx` lista, para cada pixel de saída e cada *tap* do kernel, a linha de
  entrada que ele lê (o im2col da convolução, uma constante);
- `pad` vale 1 nos *taps* que caem no *padding*: a **seleção** devolve `+0`
  ali — seleção, não multiplicação, então exatamente `+0` mesmo que a linha
  lida contenha `NaN`;
- `reshape` (uma cópia de bytes) junta os *taps* de um pixel numa linha;
- um `linear` faz a contração: cada saída é um produto canônico sobre `k·C`,
  **os mesmos bits em todo substrato**.

`Vapor.Spatial.conv2d/7` e `conv3d/7` (vídeo, volumes) aceitam *stride*,
*padding* e dilatação. A normalização por grupos (`group_norm/9`) usa
contrações por matrizes seletoras exatas (0/1); `upsample_nearest/5` é um
`gather_row`; `pixel_attention/9` é a atenção do encoder (horizonte no fim).
Tudo conferido contra o torch a ≤ 4·10⁻⁷.

O que **não** está aqui: convoluções agrupadas/*depthwise* (o layout do
im2col intercala *taps* e canais) e transpostas (o *upsampling* é *nearest*
+ convolução, como fazem os decoders do diffusers). O Mamba resolve a sua
convolução *depthwise* causal de outro jeito — como estado (FRONTEIRA §4).

## 2. VAE contínuo (`Vapor.Lock.Adapters.VAE`)

O decoder do `AutoencoderKL` do diffusers — o VAE do Stable Diffusion 1.x/2.x,
SDXL e, com 16 canais latentes, a forma do Flux e do SD3:

    z → post_quant_conv → conv_in → meio: resnet, atenção sobre pixels, resnet
      → blocos de subida (resnets, nearest ×2 + conv) → GroupNorm → SiLU → conv_out

Contrato `:map`: latentes em linhas → pixels em linhas. Um bug instrutivo
apareceu aqui: o programa como **árvore** (sem nomes) crescia
exponencialmente com a profundidade; todo intermediário passou a ser uma
*let-binding* nomeada (`Spatial.name/3`) e o programa é um DAG. Conferido
contra o diffusers: 8,7·10⁻⁷ e 7,2·10⁻⁷ (dois tamanhos).

## 3. DiT (`Vapor.Lock.Adapters.DiT`)

O `DiTTransformer2DModel` (Peebles & Xie): *patchify* (convolução de passo =
`linear`), tabela de posições 2-D senoidal (**bit a bit** a do diffusers),
*timestep embedding* senoidal com exponenciais e senos corretamente
arredondados, embedding de classe, blocos com **adaLN-Zero** (escala,
deslocamento e portão por bloco calculados do condicionamento), e a camada
final com *unpatchify*. 3,1·10⁻⁷ contra o diffusers.

## 4. Atenção cruzada: fusão no espaço latente

A fusão multimodal "sem pivô de texto" — áudio atendendo a *patches* de
vídeo, um DiT atendendo a um prompt — é a **atenção existente**: as
projeções da outra corrente são a tabela de chaves e valores, e o horizonte
de toda consulta é a última linha dessa tabela. Nenhum operador novo;
≤ 10⁻⁶ contra `torch.nn.MultiheadAttention` (`cross_attention_test.exs`).

## 5. Whisper (`Vapor.Lock.Adapters.Whisper`)

Reconhecimento de fala como **dois programas** do mesmo modelo — o contrato
da eclusa ganhou *partes* declaradas (`spec.parts`):

- **encoder** (`part: :encoder`, contrato `:encoder`): quadros log-mel em
  linhas → `conv1d → gelu → conv1d(stride 2) → gelu` (convoluções de §1, com
  altura 1) → + posições guardadas → camadas pré-LN bidirecionais →
  LayerNorm; e, uma vez por áudio, as **chaves e valores cruzados** de cada
  camada do decoder — um passo de decodificação nunca os recalcula;
- **decoder** (padrão, contrato `:causal_lm`): tokens + posições
  aprendidas → autoatenção causal (cache KV) → **atenção cruzada** (§4) →
  MLP GELU → LayerNorm → logits (cabeça ligada aos embeddings).

A query é escalada por `dh^−½` antes do produto, na ordem do `transformers`.
Conferido contra o `WhisperForConditionalGeneration`: estados do encoder
4,8·10⁻⁷, logits 4,2·10⁻⁷, decodificação gulosa idêntica — no oráculo e no
worker nativo (bits iguais) — com dois controles que precisam falhar: o
decoder alimentado com o encoder de **outro** áudio, e os quadros
**invertidos** no tempo. O front-end log-mel do `transformers` não é
reproduzido (o adaptador recebe as *features*); o do vapor é
`Vapor.Modal.Speech`.

Nas larguras do whisper-tiny (d 384, 4 + 4 camadas), 30 s de áudio
(3 000 quadros → 1 500 posições) passam pelo encoder em ≈ 1,6 s nesta VM de
2 núcleos; um passo do decoder custa ≈ 8 ms
([bench/FRONTIER.md](bench/FRONTIER.md)).

## 6. O que separa isto de Sora e Midjourney

Os operadores e as topologias acima são o que um runtime precisa oferecer;
estão aqui, conferidos. Um gerador de imagem ou de vídeo é, além disso:

- **pesos treinados** em centenas de milhões de pares — não estão neste
  ambiente, e sem eles nenhuma afirmação de qualidade visual é possível (por
  isso não há nenhuma);
- uma **U-Net** com ControlNet (cópia do encoder + convoluções-zero) para
  condicionamento espacial — os blocos existem, o adaptador não;
- **memória longa** de vídeo: das três formas medidas (janela com anel,
  cache latente, estado de SSM), nenhuma foi treinada para vídeo;
- **computação**: um DiT de vídeo é uma GPU (ou muitas) por minutos — e as
  sessões residentes no Vulkan existem desde 0.8, mas só foram medidas no
  lavapipe.

O vapor não chega a Sora; chega ao ponto em que um Sora treinado poderia
rodar **com os mesmos bits em toda máquina** e com certificado.
