# Ecossistema vapor — escrutínio e plano

> **0.4.0:** os modelos agora entram pela **eclusa de modelos** (`Vapor.Lock`,
> [ECLUSA.md](ECLUSA.md)) — o motor, o embedder e o servidor leem um contrato,
> não uma família. Imagem e áudio: [ANY_TO_ANY.md](ANY_TO_ANY.md). Fusão:
> [FUSAO.md](FUSAO.md). Qualidade das saídas: [QUALIDADE.md](QUALIDADE.md).

Este documento examina a proposta de expansão (tokenizador, ingestão Hugging
Face, LoRA/QLoRA, destilação, serviço, aparato de benchmark) contra as
garantias que o núcleo já entrega, e fixa a ordem de construção. Ele é
revisado a cada fase: o que for aprendido na implementação volta para cá.

## Critério

Uma camada nova entra se preservar os invariantes do núcleo, ou se disser
explicitamente onde deixa de preservá-los:

1. `deps: []` no plano de controle; nenhum código gerado dentro da BEAM.
2. Semântica exata no oráculo; política `:canonical` bit-idêntica entre
   substratos; `:fast` dentro de envelope.
3. Todo artefato aceito tem certificado; toda rejeição tem contraexemplo.
4. Números publicados são medidos aqui, com a fonte citada.

## Veredito por item

| Proposta | Veredito | Motivo técnico |
|---|---|---|
| API OpenAI | **sim**, HTTP/1.1 + SSE sobre `:gen_tcp` | é o que a API OpenAI usa para streaming; cabe em OTP puro |
| HTTP/3 | **não no núcleo** | exige QUIC + TLS 1.3; terminar HTTP/3 num proxy (Caddy, nginx, h2o) na frente é mais seguro e não custa nada ao modelo |
| Phoenix WebSockets | **não como dependência** | o vapor é biblioteca: uma app Phoenix chama `Vapor.Serve` na mesma BEAM |
| Ponte Python zero-copy | **depois** | cliente HTTP basta; tensores já vivem em `/dev/shm` endereçados por conteúdo |
| Tokenizador BPE puro | **sim** | BPE byte-level (GPT-2/Llama 3/Qwen) e SentencePiece-BPE com byte fallback (Llama 2/Mistral), lendo `tokenizer.json`; oráculo diferencial: a biblioteca `tokenizers` da HF |
| Trie em Zig SIMD | **não** | seria NIF (quebra a contenção) ou uma travessia de processo mais cara que a própria tokenização |
| "20 M tokens/s por núcleo" | **rejeitado a priori** | nenhuma medição sustenta; será medido e publicado |
| Ingestão safetensors | **sim** | cabeçalho JSON + offsets validados como eclusa; pesos f32 mapeados pelo worker sem cópia |
| `config.json` → termos | **sim para Llama, Mistral, Qwen2** → depois Qwen3, Qwen3-MoE, Mixtral, Gemma 3, DeepSeek-V3 (2026-10-01) | mesma estrutura (RMSNorm, RoPE, GQA, SwiGLU); MLA e MoE entraram como contrações exatas; Mamba na 0.6.0 |
| GGUF | **depois** → feito em P8 | formato secundário; safetensors cobre a HF. Leitura, desquantização e exportação (f32/q8_0) conferidas contra gguf-py e o próprio llama.cpp |
| 70 B: 140 GB → 41 GB | **aritmética correta** | 70·10⁹ × 4,6875 / 8 = 41,0 GB |
| "perda de perplexidade matematicamente delimitada" | **falso** | o erro por peso é limitado e o erro de arredondamento dos kernels tem envelope; perplexidade só se mede |
| QLoRA "somando B·A antes de emitir" | **corrigido** | nunca materializar B·A: `y = W₀x + s·B(Ax)`, dois GEMV finos |
| Fundir B·A no `:sb4` | **com ressalva** | requantizar altera o erro de quantização; não é sem perdas |
| Backprop "pela Rung 3" | **sim, com a leitura certa** | a identidade adjunta ⟨Wx,v⟩ = ⟨x,Wᵀv⟩ certifica os kernels transpostos que o reverse-mode usa |
| Destilação "estabilizada abaixo do ruído 2γ_K S_ij" | **reformulado** | estabilidade vem de log-softmax com subtração do máximo; o envelope certifica o erro de arredondamento da perda, não "estabiliza gradientes" |
| PagedKV com processo OTP dedicado | **estrutura sim, processo não** | a tabela de páginas é um dado puro no estado do motor; um processo a mais só serializa mensagens |
| Batelada contínua sem padding | **sim** | e com uma propriedade forte: a ordem canônica não depende do tamanho do lote ⇒ **invariância a lote** (mesmos tokens sozinho ou em lote) |
| Speculative decoding "aceito dentro do envelope de Wilkinson" | **conceito errado** | aceitação é por comparação de tokens (greedy) ou amostragem por rejeição; com invariância a lote, o resultado é **idêntico** ao decode sem especulação — essa é a garantia real |
| "até 3,5×" | **rejeitado a priori** | depende da taxa de aceitação; será medido |
| Contadores `perf_event_open` | **sim, quando o kernel permite** | abertos antes do seccomp; relata `:unavailable` caso contrário |
| Joules/token via RAPL | **sim, quando existe** | nesta máquina `/sys/class/powercap` não existe ⇒ não validável aqui |
| "interfaces energéticas Vulkan" | **não existe** | o Vulkan core não expõe energia |
| Deriva ULP vs Float128 | **substituído** | o oráculo é exato (racional diádico), mais forte que binary128 |
| Roofline SVG | **sim** | dados já vêm do árbitro |
| Saturação por e-graphs | **depois, baixo retorno** | quase nenhuma regra de ponto flutuante é exata |
| AVX-512 | **depois** → feito (HPC) | esta CPU tem AVX-512 (x86-64-v4); ganho de desempenho, não de garantia — os bits são os mesmos |
| Treino de 8B em LoRA "em dados corporativos" | **escala errada para 1 núcleo** | a maquinaria é demonstrada em modelos pequenos; escala real pede multi-núcleo/GPU |

## O pré-requisito que a proposta omite

Um Llama precisa de `exp`, divisão, raiz quadrada inversa, máximo, reduções
por linha, RoPE, softmax, atenção com cache KV de comprimento variável e
lookup de embedding. Nada disso existe no núcleo, e a forma de introduzir
decide se a garantia central sobrevive.

**Decisão:** na política `:canonical`, toda função é um programa sobre
{+, −, × corretamente arredondados, operações inteiras, comparação, seleção}.

- `+ − ×` são corretamente arredondados em x86, AArch64, RVV **e** Vulkan
  (onde divisão e raiz *não* são: 2,5 ULP no Vulkan). Construir `div`,
  `rsqrt` e `exp` com Newton–Raphson e polinômios em ordem fixa dá bits
  idênticos em todo substrato, inclusive GPU, e o oráculo os computa por
  composição, sem caso especial.
- `max`/`min` como `select(a < b, b, a)`: a semântica de NaN de `maxps`,
  `fmax` e `vfmax` difere entre ISAs; comparação + seleção não.
- `exp` zera por definição abaixo de 2⁻¹²⁶: a saída nunca é subnormal, então
  GPUs que descartam subnormais continuam bit-idênticas.
- RoPE: tabelas cos/sin calculadas uma vez na BEAM e embarcadas como
  constantes (bits no certificado).
- Atenção sobre o cache: a redução em 16 lanes atribui o elemento *i* à lane
  *i mod 16*; posições além de `pos` contribuem zeros exatos. Logo parar o
  laço em `pos + 1` produz **os mesmos bits** que a definição mascarada sobre
  o comprimento máximo: o comprimento em tempo de execução é otimização, não
  semântica.
- `:fast` pode usar `vdivps`/`vrsqrtps`/FMA do hardware e é certificado por
  envelope.

**Certificação composicional.** O oráculo exato é lento demais para um
modelo inteiro. Mas cada kernel é o mesmo código de máquina qualquer que seja
a dimensão (dimensões são argumentos). Um certificado de modelo liga: o hash
de cada kernel, a sua verificação diferencial em instâncias de sonda, a boa
formação do cronograma e a admissão nas extensões máximas; a igualdade
completa com o oráculo é verificada numa configuração pequena da mesma
arquitetura.

## Fases

Cada fase termina com testes verdes em todos os níveis, commit e registro aqui.

| Fase | Entrega | Pronto quando |
|---|---|---|
| P1 | Numérica e operadores: primitivas inteiras/comparação/seleção nos 3 ISAs de CPU; `recip`, `div`, `rsqrt`, `exp`, `sigmoid`, `silu`, `max`; reduções por linha; broadcast; GEMV f32; gather e escrita de linha em tempo de execução; ambiente de FP do worker verificado | oráculo = x86 = AArch64 = RVV (QEMU, VLEN 128–512) bit a bit; objdump decodifica tudo; erro ULP de cada função medido e publicado |
| P2 | JSON, safetensors (eclusa), `config.json` → programa Llama/Mistral/Qwen2, quantização na ingestão | rejeição tipada para cabeçalho malformado; programas certificados |
| P3 | Oráculo de modelo: `transformers` + `torch` (PyPI) em modelos pequenos com pesos aleatórios | logits dentro de tolerância declarada; geração greedy idêntica fora de empates |
| P4 | Tokenizador | idêntico à `tokenizers` em corpora de teste e em tokenizadores reais obtidos offline |
| P5 | Geração, amostragem determinística, motor com lote contínuo e PagedKV, servidor OpenAI + SSE | invariância a lote testada; cliente OpenAI conversa com o servidor |
| P6 | Aparato de medição: contadores, histograma ULP vs exato, roofline SVG | números reproduzíveis com a fonte registrada |
| P7 | Autodiff, LoRA, AdamW, perda de destilação | gradientes = diferenças finitas no oráculo; adjunta certificada; perda cai num modelo pequeno |
| P8 | Speculative decoding, SPIR-V para os novos operadores, GGUF, AVX-512, Mamba | cada item com o seu critério |

## Registro

- 2026-09-30 — plano inicial.
- P2/P3 (f32) — `Vapor.JSON` (RFC 8259, chaves duplicadas recusadas, profundidade
  limitada; diferencial contra o `json` do Python), eclusa safetensors
  (ladrilhamento exato do segmento de dados, `bf16`/`f16` alargados exatamente —
  os 65 536 padrões `f16` iguais aos do numpy; lê bit a bit o que a biblioteca
  `safetensors` escreve), `config.json` de Llama/Mistral/Qwen2 (formatos de
  RoPE `rope_scaling` e `rope_parameters` do transformers ≥ 5), carregador de
  diretório com shards.
  **Descoberta:** termos BEAM são árvores; o fluxo residual usado ~3× por camada
  fazia a árvore não compartilhada crescer como 3^L — e hashing/cópia são
  lineares nela. Um Qwen2 de 2 camadas levava 23,5 s para baixar e 6,5 s no
  oráculo. **Correção:** *let-bindings* no `Program` (substituição semântica;
  pesos como ligações constantes, logo nenhuma chave contém bytes de peso):
  0,19 s e 0,05 s, e o maior termo não cresce com a profundidade (testado com
  12 camadas). O worker passou de tabelas fixas (64 buffers/chamadas) para
  tabelas dimensionadas pelo quadro, com a contagem limitada pelos bytes
  recebidos.
  **Paridade com transformers 5.18 / torch 2.14 (float32):** cinco variantes
  (Llama com viés e cabeça solta, RoPE Llama 3, RoPE linear com MHA, Mistral,
  Qwen2 com cabeça atada e 1 cabeça KV): |Δlogit| máximo relativo
  2,7·10⁻⁷ … 5,6·10⁻⁷ (tolerância declarada 10⁻⁵); decodificação gulosa de 16
  tokens idêntica nas cinco, sem empates. Modelo inteiro bit a bit igual ao
  oráculo em x86, interpretador RVV, AArch64 e RVV (QEMU); invariância a lote
  (prefill = passos de decode) verificada.
  **Reordenado:** a quantização na ingestão sai de P2 para depois de P3 — o
  erro que ela introduz só tem significado medido contra a referência f32.
- Quantização na ingestão — `Llama.program(..., quantize: :sb4)` guarda toda
  matriz de projeção e a cabeça em superblocos de 4 bits (o embedding fica
  f32: é consultado, não multiplicado). O `qgemv` passou a aceitar
  `x : f32[b, k]`: cada linha de ativação é um GEMV independente, com as
  mesmas instruções na mesma ordem que `b = 1` — invariância a lote no
  kernel, nos três ISAs, no interpretador e no SPIR-V (despacho `rows·b`).
  Para caber nos 14 GPRs do x86 sem perder a variante de 2 linhas por
  iteração, o registrador-base de W foi eliminado (a linha seguinte começa
  onde terminou o último ponteiro de linha) e a contagem de sub-blocos é
  relida dos argumentos num registrador novo (intervalos de vida são
  envoltórias). Bench GEMV 4096² sem regressão (66–80 ms observados).
  **Qualidade medida** (Llama largura 256, pesos gaussianos, 24 posições,
  contra os logits float32 do transformers): `sb4` (4,6875 bit/peso) erro
  RMS relativo 0,161 e KL média 0,0129; Q4_1 de referência (5 bit/peso)
  0,158 / 0,0129; Q4_0 (4,5 bit/peso) 0,190 / 0,0178. O teste exige
  `sb4` ≤ 1,25× Q4_1 nas duas medidas. O erro absoluto alto é próprio de um
  modelo minúsculo de pesos aleatórios (sem redundância); não é uma
  previsão de perplexidade em modelos treinados.
- P4 (tokenizador) — `Vapor.Tokenizer`: BPE byte-level (GPT-2, Llama 3,
  Qwen2) e BPE SentencePiece com *byte fallback* (Llama 2), um único
  procedimento de fusão por prioridade em `O(n log n)` (heap + lista
  duplamente ligada; por posto de merge ou por score do vocabulário), tokens
  guardados pela sua superfície em bytes (decodificar é concatenar).
  Fontes: GGUF (`Vapor.Ingest.GGUF`, antecipado de P8: o hub HF é bloqueado
  por política aqui; os vocabulários reais chegam como os GGUF só-vocabulário
  do llama.cpp via raw.githubusercontent, `make fixtures`, SHA-256 fixado) e
  `tokenizer.json`.
  **Resultados:** os 4 vocabulários reais (Llama 3, Qwen2, GPT-2, Llama 2)
  reproduzem os 184 vetores que o Hugging Face gerou para eles; contra a
  biblioteca `tokenizers` em 7 configurações (as convertidas pelo
  transformers e as opções originais de cada lançamento: `ignore_merges` do
  Llama 3, NFC do Qwen2, normalizador Prepend+Replace do Llama 2) × 415
  textos adversariais: ids e bytes decodificados idênticos.
  **Descoberta:** o NFC do OTP (`:unicode.characters_to_nfc_binary`) compõe
  através de uma marca de classe 0 (`и ๎ ̈` → `ӥ ๎`), violando UAX #15 — e
  muda ids de token no Qwen2. `Vapor.Unicode` implementa NFC/NFKC pela
  definição (tabela de 941 compostos primários derivada em compilação);
  igual ao `unicodedata` do Python em todo ponto de código até U+2FFFF e em
  20 000 sequências aleatórias de marcas.
  Velocidade (1 núcleo, BEAM): ~330 mil tokens/s codificando texto em inglês
  (Llama 3); carga do `tokenizer.json` de 11,6 MB em ~4,5 s.

- 2026-09-30 — **diretriz nova: "HPC máximo", pipeline ponta a ponta, zip final.**
  Escrutínio das formas de paralelismo pertinentes a este sistema e onde
  cada uma entra (a garantia a preservar: resultados bit-idênticos
  independentemente de quantas threads, de quantas sequências no lote e de
  onde o KV mora):

  | forma | onde | invariante verificado |
  |---|---|---|
  | SIMD | já: AVX2, NEON, RVV (VLEN 128–512); **AVX-512** novo backend | mesmos bits que o oráculo |
  | threads (intra-op) | worker: pool de threads com barreira; cada chamada carrega um *descritor de partição* (qual argumento conta linhas, quais ponteiros avançam quantos bytes por linha) | igual para 1…N threads (linhas independentes) |
  | lote contínuo | motor: prefill em blocos + decode no mesmo passo | tokens de uma sequência iguais sozinha ou em lote |
  | PagedKV | kernels `kv_write`/`attention` com tabela de blocos | paginado = contíguo bit a bit |
  | réplicas (dados) | pool de sessões, uma por worker, requisições distribuídas | — |
  | concorrência BEAM | um processo por conexão HTTP, motor como GenServer, streaming SSE | — |
  | memória | pesos mmap compartilhados entre workers, sessões residentes | — |
  | especulação | decodificação especulativa com rascunho; aceitação por comparação greedy | saída idêntica à sem especulação |
  | GPU | SPIR-V dos operadores de modelo (lavapipe aqui) | fabric = oráculo |

  Todas as linhas da tabela estão implementadas e testadas; o estado de
  cada uma está nos itens abaixo e em [ARCHITECTURE.md §4.6](ARCHITECTURE.md).

- P5 (geração e serviço) — `Vapor.Sampler` (binary64 + SplitMix64, truncamento
  exato; amostragem gulosa e por temperatura também como operador `sample`
  no substrato), `Vapor.Engine` (lote contínuo sobre KV paginado, páginas
  reservadas na admissão — sem impasse nem preempção), `Vapor.Serve`
  (HTTP/1.1 + SSE em `:gen_tcp`, `/v1/completions`, `/v1/chat/completions`,
  `/v1/models`, modelos de chat ChatML, Llama 3 e `[INST]`), CLI
  (`mix vapor.generate | serve | demo_model | export | bench | shm`).
  **Resultados:** invariância a lote testada (9 requisições concorrentes =
  cada uma sozinha, sob qualquer fatiamento de prompt e número de threads);
  o cliente `openai` oficial conversa com o servidor, com e sem streaming;
  `make e2e` percorre eclusas → tokenizador → compilador → motor → HTTP →
  cliente. Amostragem no substrato: 1,8 → 316 tokens/s num vocabulário de
  151 936 (a ordenação na BEAM era o gargalo).
  **Robustez:** um worker que morre sob um passo encerra as sequências que
  ele guardava com `{:done, :error, …}` (HTTP 503, ou evento de erro no
  SSE), o worker renasce, a sessão é reaberta e o motor continua.

- P6 (medição) — `mix vapor.bench` gera [bench/BENCH.md](bench/BENCH.md),
  `roofline.svg` e `engine.svg`, tudo medido na hora: kernels por ISA e
  threads, motor por concorrência/threads/ISA/armazenamento/réplicas,
  prefill, ULP das funções canônicas contra binary64, tokenizador. Esta VM
  (Xeon 2,1 GHz, 2 vCPUs) não expõe PMU: o tempo de CPU vem do task-clock.
  Números de 2026-10-01: teto de memória 53 GB/s; GEMV f32 2048² 0,28 ms
  (60 GB/s); GEMV bf16 2048² 0,17 ms; linear 512² × 64 linhas 84 GFLOP/s
  (AVX-512, 2 threads); motor Llama 256-largura/4 camadas/vocab 32 000:
  ~1 500–1 800 tokens/s com 8 sequências, prefill ~5 500 tokens/s;
  tokenizador Llama 3 ~535 mil tokens/s num núcleo da BEAM. Variam com a
  carga do host (VM compartilhada): a fonte é sempre o BENCH.md da execução.

- P7 (autodiff e destilação) — `Vapor.Autodiff`: modo reverso de termos em
  termos (gradientes são programas como quaisquer outros: baixados,
  executados em todo substrato, bit a bit iguais e certificados); operador
  `transpose`. `Vapor.Train`: LoRA na última MLP + cabeça, perda KL contra o
  professor, AdamW — um passo de treino é **um programa recorrente**, a
  corrida inteira roda no worker atrás de uma travessia. **Resultados:**
  gradientes = diferenças finitas (oráculo); a adjunta dos kernels
  transpostos é certificada pela Rung 3; KL 0,051 → 0,00093 em 120 passos
  num modelo pequeno (~20 ms de worker); bits idênticos em 1–3 threads e no
  fabric.

- P8 — **especulação** (`Vapor.Speculative`): rascunho propõe `k`, alvo
  verifica `k+1` linhas num passo; pela invariância a lote a saída é
  *idêntica* à do alvo sozinho (testado), a aceitação só muda a velocidade
  (rascunho igual 1,0; parecido 0,66; sem relação 0,0).
  **GGUF** (`Vapor.Ingest.GGUF`, `Vapor.Ingest.GGML`, `Vapor.Model.GGUF`):
  desquantização de F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/Q4_K/Q5_K/Q6_K
  bit a bit igual à do `gguf-py`; a permutação q/k do conversor do llama.cpp
  desfeita; `rope_freqs` do Llama 3 como fatores. Um GGUF produzido pelo
  próprio `convert_hf_to_gguf.py` (do sdist do PyPI) reproduz os logits do
  transformers: f32 com erro relativo 5,6·10⁻⁷, q8_0 dentro da sua
  quantização. **Exportação** (`Vapor.Model.GGUF.write/4`, `mix vapor.export
  --out x.gguf`): f32 e q8_0 (quantização bit a bit igual à do `gguf-py`,
  empates incluídos); `load(write(m)) = m`; o **llama.cpp** (libllama
  compilada do mesmo sdist) carrega o arquivo, tokeniza com o nosso
  vocabulário exatamente como o vapor e reproduz os logits (f32 ≤ 10⁻⁴,
  q8_0 ≤ 5 % do maior logit — o llama.cpp quantiza também as ativações) para
  Llama e Qwen2.
  **SPIR-V dos operadores de modelo**: gather, RoPE, escrita de KV (cópia, no
  lugar, paginada, última escrita vence), atenção contígua e paginada (três
  passadas que recalculam os escores — sem memória de rascunho), amostragem
  e transposição; o daemon passou a tabelas dimensionadas pelo quadro. Um
  modelo inteiro, o passo paginado com amostragem do motor e o decode
  recorrente rodam no lavapipe **bit a bit iguais ao oráculo**; um modelo
  `storage: :bf16` é certificado com paridade em fabric, AVX2, AVX-512,
  RVV e oráculo.
  **AVX-512**: backend EVEX completo (ver ARCHITECTURE §3.2), bit a bit igual
  ao oráculo em todo programa canônico, modelo inteiro, atenção paginada,
  amostragem, `sb4` e gradientes; o binutils decodifica toda instrução de
  todo kernel. Ganho medido: `x·silu(x)` 2,3×, linear em lote 1,4×, atenção
  1,2×; kernels limitados por banda ficam iguais, como o roofline prevê.
  Mamba ficou fora nesta fase (nenhuma arquitetura pedida o usava); entrou
  na 0.6.0 como adaptador de topologia ([FRONTEIRA.md §4](FRONTEIRA.md)).

- HPC — réplicas (`Vapor.Engine.Pool`, `--replicas N`): uma compilação,
  pesos nas mesmas páginas (`/dev/shm` endereçado por conteúdo), roteamento
  para a menor fila, réplica morta substituída sem afetar as outras; mesma
  resposta por qualquer réplica (testado). **bf16 residente**
  (`storage: :bf16`): instrução portátil `vld_bf16` nos quatro backends e
  no SPIR-V, `gemv_bf16` e `gather_row_bf16`; o programa é *bit a bit* o
  programa f32 sobre os pesos arredondados (testado em todo substrato, no
  motor e pela escada); GEMV com metade dos bytes: 2× mais rápido nesta
  máquina (0,59 vs 1,22 ms em 4096², 2 threads — limitado por banda nos dois
  casos). Um checkpoint bf16 do Hugging Face entra sem ser alargado.

- Safetensors completo — todos os dtypes do formato reconhecidos (tamanhos
  conferidos, sub-byte em bits): F64 arredondado como torch/numpy;
  F8_E4M3, F8_E4M3FNUZ, F8_E5M2, F8_E5M2FNUZ e F8_E8M0 alargados exatamente
  (os 256 padrões de cada um iguais ao `.float()` do torch, NaN incluídos);
  inteiros de 16/64 bits e sem sinal levados a s32 quando cabem (senão
  recusa nomeando o tensor); BOOL; F4/F6/C64 reconhecidos e recusados pelo
  nome. Escrita em F32 ou estreitada para BF16/F16 (RNE; igual ao torch em
  20 000 padrões com empates, subnormais e overflow), forma com shards e
  `model.safetensors.index.json`. `Config.to_map/1` (inverso testado de
  `from_map/1`) e `Vapor.Model.write/4` (`mix vapor.export --out DIR`):
  o transformers carrega o diretório exportado pelo vapor e reproduz **bit a
  bit** os logits do checkpoint original nas cinco variantes; em BF16, os
  logits do vapor sobre os mesmos pesos ficam dentro de 10⁻⁵. Um modelo
  entra por GGUF e sai como diretório Hugging Face com shards bf16 —
  percorrido no `make e2e`.

- 2026-10-01 — **diretriz nova: modelos de fronteira, agentes, RAG, "agentes
  imutáveis"; depois, o ecossistema Elixir.** O escrutínio completo e o
  desenho estão em [AGENTES.md](AGENTES.md) e
  [ECOSSISTEMA_ELIXIR.md](ECOSSISTEMA_ELIXIR.md). Os vereditos:

  | proposta | veredito | motivo técnico |
  |---|---|---|
  | modelos de fronteira (arquiteturas abertas) | **sim**: Qwen3, Qwen3-MoE, Mixtral, Gemma 3, DeepSeek-V3 | cada peça nova reduzida a contrações exatas (seleção 0/1, posto por contagem, `sel`), sem operador "aproximado"; paridade com `transformers` ≤ 8·10⁻⁷ |
  | modelos de fronteira (hospedados) | **sim, como observação** | OpenAI-compatível e API de Mensagens; a decisão é registrada, não reproduzida — dizê-lo é parte da garantia |
  | "mais um framework de agentes" | **não** | o laço é commodity; o que só o vapor tem é determinismo ⇒ execução como objeto de prova (repetir = verificar) |
  | agentes imutáveis | **sim, com cinco leituras e seus limites** | spec como valor; diário só-acréscimo; capacidades no digest; repetição como prova; apagamento por destruição de chave |
  | "exatamente uma vez" | **reformulado** | no máximo uma vez com destinatário idempotente (chave estável por passo); é o que existe |
  | RAG | **sim, verificável** | corpus = raiz Merkle; recuperação = função (BM25 com log CR, densos certificados, RRF exato); citações literais por construção |
  | saída estruturada / tool calls | **sim, por construção** | gramática de bytes sobre a trie do vocabulário; nenhum *retry* |
  | templates de chat | **sim, hermético** | Jinja próprio com a semântica do ambiente do HF; byte a byte igual ao `jinja2` em 240 renderizações reais |
  | funções elementares | **corretamente arredondadas** para tabelas (RoPE/YaRN) | Ziv: binary64 + inteiro de precisão crescente; validado contra mpmath; digests fixados |
  | codificação canônica | **CBOR determinístico** (RFC 8949 §4.2) | `term_to_binary` não é contrato entre versões do OTP nem entre linguagens; certificados verificáveis em Python |
  | Phoenix/Plug, Nx, Livebook | **sim, fora do núcleo** | `integrations/` com dependências próprias; núcleo `deps: []` |
  | Ecto, Oban, LiveView, Broadway | ***behaviours*/ganchos + receitas** | `Store`, `on_event:`, cancelamento por monitor |
  | AtomVM, Membrane, Riak | **não** | sem *ports*/MMU; sem modelos de mídia; armazenamento endereçado por conteúdo serve melhor |

  **Descobertas** (cada uma com teste):
  - Funções elementares: o caminho rápido de Ziv não decidia nada para
    resultados negativos (pontos médios trocados): 27 % caíam no caminho
    lento. `log(1)` ficava em laço infinito (o teste de exatidão diádica
    estava errado).
  - Um laço `let` morto sem saídas quebrava o *lowering*. Agora há
    eliminação de código morto.
  - O `transformers` 5 escreve `num_local_experts` no Qwen3-MoE. As normas
    latentes do DeepSeek usam ε = 10⁻⁶, não `rms_norm_eps`.
  - Servidor: uma requisição restrita derrubava o motor e ficava pendurada
    para sempre. Agora o motor é monitorado e exceções viram 500. Clientes
    desconectados ocupavam o lote até `max_tokens`; agora são cancelados.
  - Gramática: três becos sem saída (escape no limite de `maxLength`,
    surrogates soltos, UTF-8 malformado), achados por teste de propriedade
    e por um modelo aleatório no Livebook.
  - Mix ≥ 1.15 poda o *code path*: `:inets`/`:ssl` precisavam ser declarados.
  - Uma corrida antiga, exposta pelo *timing* do Elixir 1.18: escrever num
    worker recém-morto (EPIPE) virava sinal de saída pelo *link* do *port*
    e derrubava o motor. Os donos de *port* agora interceptam saídas.
  - **ZK/FHE** (proposta de 2026-10-01, escrutínio em [ZK_FHE.md](ZK_FHE.md)):
    as zkVMs são RV32IM, então “RVV para zkVM” cai. Ponto flutuante não é
    aritmética de corpo, e o fragmento inteiro sim, com a ponte provada em
    Lean. Com pesos públicos, os recibos já verificam por reexecução, e o
    ZK só acrescenta privacidade e sucintez. Construídos: corpos/NTT,
    R1CS → Groth16 → EVM (209 922 gás de execução medidos, contra os
    “< 200k” prometidos) e polinômios com erro provado. Recusado: CKKS/BFV
    sem auditoria. A revisão independente achou dois furos de solidez no
    circuito, ambos corrigidos e testados: a entrada privada não tinha
    checagem de faixa (dava para alcançar qualquer saída), e um ReLU em
    corpo pequeno podia ter duas decomposições.
  - A revisão independente achou mais defeitos, todos corrigidos e
    testados. A retomada podia seguir de um prefixo que não verificava e
    agir. Um histórico apagado deixava a retomada em laço. Uma execução
    parada por `max_steps` não verificava. Dois retomadores podiam agir os
    dois; agora a ação é anunciada antes. Um titular apagado ganhava chave
    nova em silêncio. O comprimento de string do caminho rápido era contado
    em grafemas.
  - A divisão canônica não é a IEEE: até 1 ulp de diferença (era sabido,
    nunca medido contra uma referência externa; o avaliador do Nx mediu).
