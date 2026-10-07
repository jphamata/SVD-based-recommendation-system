# O estúdio — grafos de mídia que se conferem

O `Vapor.Studio` é um editor de grafos de nós tipados para imagem, som, vídeo,
3D, difusão e aprendizado por reforço, no espírito do ComfyUI. Ele tem três
coisas que o ComfyUI não tem:

1. **Cache por conteúdo, exato.** A chave de um nó é o SHA-256 do seu tipo,
   versão, parâmetros e das chaves de quem o alimenta (um DAG de Merkle). Se
   você muda um parâmetro, roda de novo só o que depende dele. A chave não é
   uma heurística de "parece igual": nós determinísticos dão os mesmos bits.
2. **Recibos.** Cada saída leva o digest dos seus bits exatos, e cada
   execução termina numa **raiz de Merkle**. `Studio.verify/3` reexecuta o
   grafo sem cache e compara as raízes. Qualquer pessoa com o grafo confere
   que um resultado veio dele.
3. **Recusa inteira.** Um grafo mal tipado, ou um workflow do ComfyUI com um
   nó sem tradução, é recusado antes de rodar. Todos os problemas aparecem
   juntos, cada um com o reparo. Nada roda pela metade.

| onde | o quê |
|---|---|
| `Vapor.Studio` | registro, validação (tipos, ciclos), execução, cache, recibos, `verify` |
| `Vapor.Studio.Nodes.*` | os nós: Core, Image, Audio, Video, Vision, Diffusion, RL, Geom |
| `Vapor.Studio.Comfy` | importação de workflows do ComfyUI (formato de API) |
| `Vapor.Studio.Templates` | seis grafos de partida, que também servem de exemplo para agentes |
| `Vapor.MCP.Server` / `mix vapor.mcp` | o estúdio e a busca com provas como ferramentas MCP |
| console → *Estúdio* | a tela de nós ([CONSOLE.md](CONSOLE.md)) |

![O estúdio no console](img/console-estudio.png)

## 1. Os nós

Ao todo são 64 tipos; a lista viva sai de `Studio.catalogue/0` ou da
ferramenta MCP `studio_catalogue`. Os parâmetros são tipados e têm faixa e
padrão. Os valores que passam pelos fios são `image`, `mask`, `audio`,
`video`, `mesh`, `text`, `number`, `tensor`, `latent` e `json`, e
`Studio.Value` define o digest de cada um.

| categoria | nós | fidelidade conferida |
|---|---|---|
| imagem | load (PNG, JPEG, GIF, PPM), scene, solid, resize, scale_by, crop, pad, flip, rotate, blur, sharpen, invert, grayscale, levels, blend, composite; máscaras (threshold, rect, invert, feather) | redimensionamento = `torch.nn.functional.interpolate` (nearest-exact, bilinear, bicubic a = −0,75, area) e o Lanczos do Pillow, a 2,2·10⁻⁶; programa nativo = avaliador esparso da BEAM, bit a bit (`studio_media_test.exs`) |
| som | load (WAV), tone, noise, gain, mix, concat, trim, fade, normalize, reverse, resample, spectrogram | reamostragem 48 → 16 kHz: 69,6 dB de SNR contra o tom ideal (decimação desalinhada: 17,7 dB); 7 kHz → 8 kHz não dobra (< −40 dB) |
| vídeo | load (Y4M, MJPEG-AVI, GIF), still, camera (Ken Burns), frame, map (subgrafo por quadro), resize, trim, reverse, concat, crossfade, fps, upscale | AVI = libjpeg quadro a quadro; planos Y4M = ffmpeg; GIF lido pelo Pillow com os nossos pixels |
| visão | upscale (×2/×4), OCR, dígito manuscrito | o ampliador consistente (§3) |
| 3D | shape (esfera, toro, caixa, giroide, bolhas por SDF), heightmap, render, turntable | malha fechada, volume e área a 0,5 % da esfera analítica; GLB/OBJ/PLY lidos pelo trimesh |
| RL | episode (CartPole, FrozenLake, braço de duas juntas; política treinada, aleatória ou especialista) | dinâmica = gymnasium a 10⁻¹²; tabela do FrozenLake = a do gymnasium (`rl_test.exs`) |
| difusão | checkpoint, text, empty_latent, sample, decode, encode, generate | = pipelines do diffusers a ~10⁻⁶ (§2) |

Todos os nós seguem o mesmo **contrato de determinismo**: as mesmas entradas
e parâmetros dão os mesmos bits em qualquer máquina. Na BEAM entram só as
operações IEEE (+ − × ÷ √) e as funções corretamente arredondadas de
`Vapor.CR` (sin, cos, exp, log, pow). O resto roda como programa certificado
no worker.

Os codecs foram escritos aqui:

- **GIF**: paleta por *median cut*, Floyd–Steinberg opcional, LZW idêntico
  ao `compress/lzw` do Go.
- **Y4M**: BT.601 em faixa limitada.
- **MJPEG-AVI**: leitura dos blocos `movi`, com as tabelas Huffman do Anexo
  K inseridas quando falta o DHT.

H.264, MP4 e WebM ficaram de fora (§6).

## 2. Stable Diffusion

`Vapor.Diffusion.Pipeline` abre um diretório no layout do diffusers e
admite cada rede pela eclusa:

| parte | adaptador | conferência |
|---|---|---|
| `text_encoder/` (CLIPTextModel) | `Encoder` (clip_text) | já conferido contra o `transformers` (0.6) |
| `unet/` (UNet2DConditionModel) | `UNet` (novo) | SD 1.x: 1,17·10⁻⁶; SD 2.x (projeções lineares, cabeças de 64): 1,03·10⁻⁶ |
| `vae/` (AutoencoderKL) | `VAE`, encoder **e** decoder | encoder 1,14·10⁻⁶ (os *pads* assimétricos {0,1,0,1} do diffusers) |
| `scheduler/` | `Vapor.Diffusion.Scheduler` | DDIM, Euler, DPM-Solver++ 2M × espaçamento leading/linspace/trailing × 10/25 passos: os mesmos *timesteps* e os mesmos latentes finais (≤ 6·10⁻⁷, o float32 do diffusers) |
| `tokenizer/` | `Vapor.Tokenizer` | `tokenizer.json` ou os arquivos lentos (vocab.json + merges.txt), montados no tokenizador que o `transformers` escreveria; mesmos ids |

As redes rodam como programas float32 compilados uma vez por forma. O
amostrador roda em binary64 na BEAM: o *schedule*, a orientação sem
classificador (CFG), o passo e a mistura de latentes do inpainting. O ruído
inicial vem de `Vapor.Modal.Rng`. **Uma imagem é função de (pesos, prompt,
semente, passos, orientação, amostrador).**

O pipeline inteiro foi comparado com o próprio diffusers num checkpoint
minúsculo aleatório, incluído em `priv/quality/sd_tiny`, com os mesmos ids e
o mesmo ruído:

| modo | diffusers | maior diferença por pixel |
|---|---|---|
| texto → imagem, DDIM | `StableDiffusionPipeline` | 9,5·10⁻⁷ |
| texto → imagem, Euler | idem | 9,2·10⁻⁷ |
| texto → imagem, DPM++ 2M | idem | 9,5·10⁻⁷ |
| imagem → imagem (força 0,6) | `StableDiffusionImg2ImgPipeline` | 1,1·10⁻⁶ |
| inpainting (U-Net de 4 canais) | `StableDiffusionInpaintPipeline` | 1,4·10⁻⁶ |
| **controle**: DPM++ contra a referência DDIM | — | 0,096 |

Há duas diferenças deliberadas em relação ao diffusers, ambas declaradas:

- O latente do encoder do VAE é a **média** da distribuição. O diffusers
  amostra, o que é uma segunda fonte de aleatoriedade, escondida; na
  comparação, o diffusers foi forçado à média.
- O inpainting é feito por mistura de latentes com uma U-Net de 4 canais. A
  U-Net de inpainting de 9 canais é recusada pela eclusa.

No estúdio, os nós `diffusion.*` têm a forma do ComfyUI: checkpoint → texto
→ latente vazio → amostrador → decodificação. O workflow txt2img do ComfyUI
é importado, roda e dá **os mesmos bits** que `Pipeline.generate`
(`diffusion_pipeline_test.exs`).

O checkpoint precisa ser um **diretório diffusers** dentro do diretório do
estúdio ou de `VAPOR_MODELS`. Um `.safetensors` de arquivo único é recusado
com a conversão a fazer:
`StableDiffusionPipeline.from_single_file(f).save_pretrained(dir)`.

## 3. O ampliador consistente

O `Vapor.Vision.Upscale` (`image.upscale`, `video.upscale`,
`mix vapor.upscale`) amplia ×2 ou ×4 com uma garantia: **reduzir o
resultado devolve a entrada**. Em símbolos, D(y) = x, onde D é a média de
blocos 2×2, e isso vale por construção, até o arredondamento de binary64.

O modelo funciona em três passos:

1. Uma MLP de *patches* (49 → 64 → 64 → 4) prevê uma correção sobre o
   Lanczos ×2.
2. Uma projeção exata impõe D(y) = x, redistribuindo a diferença quando há
   saturação.
3. Em imagens coloridas, a luma passa pela rede e a croma pelo Lanczos; tudo
   é projetado no fim.

O treino foi feito aqui: 30 mil passos sobre imagens públicas do scikit-image
e texto renderizado em fontes livres. Ele é reprodutível bit a bit, e o
recibo do treino está em `priv/upscale/config.json`.

PSNR em imagens retidas (nunca vistas no treino):

| imagem | vapor | Lanczos + a mesma projeção |
|---|---:|---:|
| texto 0 | **24,58** | 21,17 |
| texto 1 | **22,91** | 20,16 |
| fantoma de Shepp–Logan | **34,04** | 28,23 |
| fotografia (camera) | 30,49 | 30,42 |
| fotografia (chelsea) | 34,46 | **34,69** |
| texto 2 | **21,36** | 19,81 |
| grama | 23,94 | **24,03** |
| roda de cores (gradiente suave) | 53,73 | **55,15** |

- **Inconsistência** |D(y) − x|: ~10⁻¹⁶ no vapor, contra 0,05–0,17 no
  Lanczos e no bicúbico.
- **OCR a jusante**: o CER cai à metade em relação à imagem pequena.
- **Achado**: a projeção sozinha já melhora o Lanczos em 0,2–1 dB.

A afirmação honesta é esta:

- em texto e gráficos de traço, o ampliador ganha de +1,5 a +5,8 dB da
  melhor linha de base com a mesma garantia;
- em fotografias, empata (−0,2 a +0,1 dB);
- num gradiente suave, perde 1,4 dB, com os dois acima de 53 dB.

Ele **não inventa detalhe**, e isso é de propósito: um resultado que
contradiz a entrada não é consistente. Um GAN que alucina textura é outra
ferramenta, e não está aqui.

## 4. Aprendizado por reforço, 3D, jogos

Aprendizado por reforço (`Vapor.RL`):

- **Ambientes**: CartPole (a dinâmica do gymnasium), FrozenLake (a tabela P
  do gymnasium) e um braço de duas juntas que alcança alvos.
- **Algoritmos**: iteração de valor, Q-learning, REINFORCE (gradiente de
  política como programa, via `Vapor.Autodiff`) e clonagem de
  comportamento.
- **Resultados das políticas incluídas**:
  - CartPole: retorno 472,9/500 em partidas nunca vistas; sem treino, 18.
  - Braço: 77 % dos alvos retidos; o especialista acerta 100 %, a política
    aleatória 1 %.
  - FrozenLake: o Q-learning acha a política ótima, com 74,7 % de sucesso;
    "sempre à esquerda" dá 0 %.
- **Replay**: um episódio é a semente mais as ações. O nó `rl.episode`
  devolve o vídeo, o retorno e as ações, e o replay redesenha os mesmos
  quadros.

3D (`Vapor.Geom`):

- **Malhas**: tetraedros marchantes sobre SDFs, com solda de vértices e
  orientação para fora; relevo a partir de imagens.
- **Exportação**: OBJ, PLY e GLB.
- **Visualização**: rasterizador com z-buffer determinístico e vídeo
  girando.

## 5. Para agentes: o servidor MCP

```sh
mix vapor.mcp --dir /caminho/do/trabalho
```

Isso sobe um servidor Model Context Protocol (stdio, JSON-RPC 2.0) com seis
ferramentas. Ele foi testado com o **cliente oficial do SDK Python do MCP**
(`mcp_server_test.exs`). Cada ferramenta ataca uma dor de quem usa agentes:

| ferramenta | dor |
|---|---|
| `studio_catalogue` | o agente adivinha nomes de nós e parâmetros; aqui ele os lê, com tipos e faixas |
| `studio_validate` | um fio errado descoberto depois de minutos de cálculo; aqui antes, com o nó e o reparo |
| `studio_run` | a cada nova tentativa, tudo é recalculado; aqui o cache **vive entre chamadas**, e o agente que edita um nó recalcula só o que depende dele (medido: 2 de 5 nós) |
| `studio_verify` | "a ferramenta produziu mesmo isso?"; a raiz de Merkle é verificável por qualquer um |
| `comfy_import` | workflows compartilhados como JSON do ComfyUI |
| `context_search` | citações que o agente pode inventar; aqui cada trecho vem com a prova de Merkle contra a raiz do corpus |

Detalhes do comportamento:

- As saídas são gravadas como arquivos com o digest no nome, e as imagens
  pequenas também voltam embutidas na resposta.
- Uma falha de ferramenta volta como resultado com `isError` e a rejeição
  (o que se esperava, como reparar), para que o modelo leia e corrija.
- Um nó que lança exceção vira erro da ferramenta; o servidor continua de
  pé.
- Caminhos fora do diretório de trabalho são recusados.

## 6. Os cursos da Hugging Face: mapa de cobertura

O pedido foi "cobrir tudo dos cursos". Cobrir tudo não é uma meta
verificável. O que se pode verificar é este mapa: o que cada curso ensina, o
que existe aqui conferido e o que falta, com o motivo.

| curso | conferido aqui | falta, e por quê |
|---|---|---|
| LLM | decodificadores pela eclusa (Llama, Qwen2/3, Mistral, Mixtral, Phi-3, Granite, DeepSeek-V3…), tokenizadores BPE/SentencePiece, motor com lote contínuo, especulação, saída estruturada | Unigram/WordPiece (T5/BERT) |
| smol (ajuste fino de modelos pequenos) | LoRA por destilação como programa recorrente (`Vapor.Train`), fusão de modelos (`Vapor.Merge`), juiz de checkpoints | SFT com dados de instrução, DPO (o treino só destila hoje) |
| MCP / contexto | servidor MCP (este documento), cliente MCP para agentes, RAG com provas de Merkle | manifestos de contexto assinados |
| Agentes | runtime de agentes com diário, *replay* e classes de efeito; ferramentas MCP | — |
| Deep RL | Q-learning, iteração de valor, REINFORCE, clonagem; CartPole/FrozenLake = gymnasium | PPO/DQN, Atari, MuJoCo (sem os simuladores aqui) |
| Robótica (LeRobot) | clonagem de comportamento num braço simulado | formato de dataset do LeRobot, políticas ACT/Diffusion Policy, hardware |
| Visão computacional | OCR com tabelas, ViT/CLIP conferidos, ampliador consistente, classificador de dígitos | detecção/segmentação com pesos treinados |
| Áudio | Whisper conferido (encoder + decoder), fala → dígito, nós de som, espectrograma | front-end log-mel igual ao do `transformers`, TTS |
| Cookbook | RAG, saída estruturada, agentes, fusão — receitas com teste | — |
| ML para jogos | episódios de RL como vídeo, cenas procedurais com semente | ambientes de jogo (Unity ML-Agents, Godot) |
| Difusão | DDPM/DDIM próprios (dígitos), **SD 1.x/2.x = diffusers**, DDIM/Euler/DPM++ 2M, img2img, inpainting | ControlNet, LoRA de difusão, SDXL (duas torres de texto), DiTs com texto (SD3, Flux) |
| ML para 3D | malhas por SDF, relevo, GLB/OBJ/PLY, renderização | NeRF, Gaussian splatting, imagem → 3D com pesos treinados |

## 7. Limites

Este documento não afirma nada além do que está abaixo:

- **Stable Diffusion com pesos reais não rodou aqui.** Esta máquina não
  baixa pesos; a paridade é com pesos aleatórios minúsculos. A velocidade de
  um SD 512×512 na CPU também não foi medida. Cada passo é um programa
  compilado, e a CPU será lenta.
- **Pixels do ComfyUI**: um KSampler traduzido usa os amostradores do
  diffusers sobre o *schedule* do checkpoint, não as sigmas do ComfyUI; cada
  tradução avisa isso. O código do ComfyUI não foi reproduzido: as traduções
  foram escritas pela semântica documentada dos nós.
- **Vídeo**: não há H.264, MP4 nem WebM. São formatos enormes, e um
  ffmpeg por *shell* viola a regra do produto, que não executa processos
  alheios. A leitura cobre Y4M, MJPEG-AVI e GIF; a escrita, GIF e Y4M.
- **Desempenho do estúdio**: a câmera sobre uma imagem 320×192 leva cerca de
  0,25 s por quadro em 2 núcleos (os pesos de Lanczos passam pelo seno
  corretamente arredondado). Os quadros são calculados em paralelo, e uma
  execução totalmente em cache volta em cerca de 1 s com as prévias.
- **Cache**: o checkpoint entra na chave pelo caminho. Se os pesos forem
  trocados no mesmo caminho, o cache não percebe; use um diretório novo.

## Como contestar

```sh
mix test test/vapor/diffusion_pipeline_test.exs   # schedulers, U-Net, VAE, pipelines e ComfyUI × diffusers
mix test test/vapor/studio_test.exs test/vapor/studio_media_test.exs test/vapor/geom_test.exs \
         test/vapor/rl_test.exs test/vapor/upscale_test.exs test/vapor/mcp_server_test.exs test/vapor/console_test.exs
python3 test/python/diffusers_pipeline.py /tmp/sd  # o checkpoint minúsculo e as referências do diffusers
mix vapor.mcp --dir /tmp/trabalho                  # e um cliente MCP qualquer
mix vapor.quality                                  # §5e de docs/bench/QUALITY.md
```
