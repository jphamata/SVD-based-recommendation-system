# Pendências — estado em 2026-10-07 (0.15.0)

Só o que está **aberto**. O que foi fechado está no [CHANGELOG](../CHANGELOG.md),
com o teste que o prova. Cada item diz por que importa e o que o fecha;
◐ = feito em parte (o que falta está escrito). Os itens dos anexos das
rodadas 0.6, 0.8, 0.9, 0.10, 0.11, 0.12, 0.13, 0.14 e 0.15 que ficaram de fora estão aqui com o motivo
([DIRETRIZ.md §9, §11–§18](DIRETRIZ.md)).

## Rodada 0.15 — o que ficou aberto

- ☐ **Amálgama no worker**: a soma exata como programa nativo (um acumulador de Kulisch de ~4 300 bits para f32 em registradores, ou a soma em duas passadas por expoente) — hoje ≈ 6 M adições/s na BEAM, o que basta para gradientes entre micro-lotes e nós, não para dentro de um *kernel*. Também: o *all-reduce* exato entre nós do `Vapor.Cluster` (as células já atravessam a rede por `to_wire/1`), e a divisão **por linhas** no `Vapor.Shard` (hoje recusada) refeita com `partial_dot/3` — a peça existe e está testada; falta ligá-la ao particionador.
- ☐ **Copela no caminho de treino**: a mesma identidade no *backward* (`∂x = ∂y·W` confere com `r`) e na atualização do otimizador; a sentinela ligada ao `Vapor.Cluster` (quarentena de um nó inteiro, não só de um trabalhador); medir a sobrecarga no worker nativo.
- ☐ **Rebis**: circuitos sequenciais (indução k sobre o *miter* de vários passos, com IC3/PDR como horizonte); ler Verilog estrutural/BLIF; `PCLMULQDQ`/`vclmul` como operação do compilador (multiplicação sem vai-um nos emissores) para o GHASH e para torres binárias; reescrita algébrica com regras para somadores de prefixo paralelo (hoje `:unknown` acima de 50 000 termos).
- ☐ **Aludel**: dividir também pelo grau (elevação de grau quando a envoltória é larga), mais variáveis com subdivisão adaptativa por eixo, e barreiras com termos racionais; exportar a testemunha para um verificador em Lean quando o `lake` estiver presente.
- ☐ **Tábua**: prazos (lógica temporal linear limitada sobre os fatos), quantificação sobre partes, e o rascunho por modelo (`Mind`) de cláusulas a partir de texto, com retrotradução, como o Alembic tem.
- ☐ **Mecanismos** (VCG, Gale–Shapley) com estabilidade e veracidade conferidas — pequenos, adiados por falta de caso (§18).
- ☐ **Compilação dupla diversificada** (Wheeler) do worker com dois Zig independentes — a resposta certa ao *trusting trust* (§18).
- ☐ **JBIG2**: Huffman com refinamento (SDREFAGG/SBREFINE sob SDHUFF/SBHUFF) e contextos aritméticos retidos entre segmentos; JPX.

## Rodada 0.14 — o que ficou aberto

- ☐ **Athanor distribuído**: as avaliações são puras e o diário é canônico — repartir o orçamento entre nós BEAM (o `Vapor.Cluster` já existe) e juntar os diários por raiz de Merkle. Hoje: um nó, avaliações sequenciais por estratégia.
- ☐ **Alembic compilado**: o compilador de fechos é ~20–50× mais lento que Elixir nativo; os objetivos numéricos poderiam descer para a álgebra de tensores (o caminho de `Vapor.Expr.compile`) com o mesmo combustível contado por bloco.
- ☐ **Provas além da enumeração**: exportar a afirmação e o espaço para SAT/SMT (a mesa de lógica já tem DRUP) quando o espaço não cabe na enumeração — "provado" passaria a valer para espaços que hoje só têm evidência.
- ☐ **Crucible**: orbitais p (bases 6-31G) e UHF para quebrar ligações; EDPs em 2D com a mesma evidência de ordem; leis de conservação para sistemas com parâmetros simbólicos.
- ☐ **Assay**: avaliação de modelos generativos com IRT (dificuldade por item), testes sequenciais (parar cedo com controle de erro), e o *dedup* em escala de corpus (LSH em disco).
- ☐ **Mente**: um laço autônomo opcional (o modelo propõe, a fornalha mede, o modelo lê o certificado e propõe de novo) com orçamento de chamadas e registro no diário; hoje o laço é guiado pela pessoa.
- ◐ **Cenas livres**: operações em texto, expressões por quadro e direção por modelo existem; falta física simples entre entidades (colisão, molas) escrita em Alembic, e a linha do tempo editável no console.

## Rodada 0.13 — o que ficou aberto

- ☐ **Motor de ofertas de baixa latência**: o casamento como programa do worker (Zig, sem alocação, livro em arranjos por nível) com o mesmo diário e o mesmo juiz; latência medida com PMU em *bare-metal*. *Kernel bypass* e FPGA ficam fora. Hoje: ~8,5 µs por evento na BEAM, com o hash.
- ☐ **Dados de mercado reais**: um dia de ITCH de amostra da Nasdaq e as curvas e preços indicativos da ANBIMA (precisam de rede): reconstruir o livro e conferir contra os instantâneos publicados; reprecificar títulos públicos com as taxas do dia.
- ☐ **Gerador inteiro no compilador**: Philox/Threefry exigem multiplicação inteira de 32/64 bits na álgebra (a mesma falta do kernel NTT); substituiriam o Wichmann–Hill do worker. Quase-Monte Carlo (Sobol, com *scrambling*) e gregas por diferenciação automática *pathwise* (o autodiff de termos já existe).
- ☐ **Modelos de taxa e de volatilidade**: Hull–White e LMM; volatilidade local de Dupire e calibração do Heston a uma superfície inteira (SVI por fatia → SSVI); XVA e crédito.
- ☐ **Backtests**: dados intradiários com o livro (o motor já existe), custo com impacto de Almgren–Chriss dentro do backtest, *walk-forward*; o CSCV custa O(N·12 870) — amostrar as metades quando N > 100.
- ☐ **Arbitragem na superfície inteira**: calendário × strike num só LP (hoje: uma maturidade por vez, mais o calendário do SVI à parte).
- ☐ **Juiz em outra linguagem**: o motor ingênuo também em Python, para que a independência seja de linguagem e não só de algoritmo; FIX de sessão (logon, *heartbeat*, *resend*), não só mensagens de aplicação.
- ☐ **Pré-negociação**: limites de crédito por contraparte e taxa em tempo de parede (hoje: tempo do evento).

## Fora do alcance desta máquina (precisam de hardware)

- ☐ RVV 1.0 em silício (BPI-F3 / Milk-V) e `cooperative_matrix` numa GPU real. *Aqui: QEMU, interpretador RVV próprio, lavapipe.*
- ☐ PMU e RAPL (J/token) em bare-metal. *O código já lê os dois; esta VM não tem nenhum.*
- ☐ Medir as sessões residentes (0.8) numa GPU real, discreta (caminho de *staging*) e integrada.
- ◐ **Apple Silicon** (0.10): worker de CPU (`MAP_JIT`, `__ulock`) e daemon Metal **compilados**; o MSL executado por um *shim* com clang e admitido pela eclusa. Falta **executar num Mac** (o daemon `mtl.zig` nunca rodou) e o isolamento do worker no macOS (sem seccomp; processo sem direitos + `posix_spawn` restrito); e Windows. *Também decide o Tauri ([INTERFACES.md](INTERFACES.md)).*
- ☐ **Tenstorrent de verdade** (0.10): o kit StableHLO + PJRT foi julgado no XLA de CPU; falta rodá-lo numa placa (tt-xla) e admitir o que ela é.
- ☐ **FreeBSD executado** (0.10): o worker compila e o binário é conferido; falta rodar a suíte num FreeBSD (Capsicum, `_umtx_op`) e portar o fabric Vulkan (`fabric.zig` usa chamadas Linux cruas) para `sys.zig`.

## Rodada 0.12 — o que ficou aberto

- ☐ **Proteínas com alinhamentos reais**: ler um MSA (Stockholm/A3M do Pfam), pesos de sequência (80 % de identidade), pseudocontagens e DCA por pseudoverossimilhança (plmDCA), que é bem mais preciso que o de campo médio em alinhamentos reais; estrutura secundária prevista da sequência (não tirada da nativa); cadeias laterais. Medir no conjunto de Jones et al. (PSICOV) contra o que se publica. *A predição de estrutura com redes treinadas segue fora: pesos e bases.*
- ☐ **Render**: amostragem por importância múltipla (BSDF × luz) para fechar as cáusticas ruidosas; BVH e malhas de triângulos (GLB do esboço → cena), texturas de imagem, materiais de microfacetas (GGX) e subsuperfície; denoiser conferido contra a referência de muitas amostras.
- ◐ **Bancada**: unidades afins (°C, °F) como leituras ✅ 0.13; faltam EDPs 2-D no tempo e sistemas acoplados (reação–difusão de várias espécies), malhas não estruturadas; DAEs (índice 1) e EDOs com atraso; otimização global (multipartida com certificado de intervalo).
- ◐ **Engenharia**: transistores (Ebers–Moll, MOSFET nível 1) ✅ 0.13, iguais ao ngspice; faltam `.subckt`, capacitâncias dos dispositivos no transitório; curto-circuito e limites de reativos no fluxo de potência; flambagem (autovalor geométrico) e não linearidade geométrica nos pórticos; bombas e válvulas nas redes; equilíbrio líquido–vapor não ideal (NRTL/UNIQUAC).
- ◐ **Lógica**: aritmética linear (simplex racional com certificado de Farkas) ✅ 0.13 ([LOGICA.md §5](LOGICA.md)); faltam o verificador DRUP no worker nativo (R(3, 4) em segundos); LRAT (verificação linear); exportar certificados de Gröbner e de Knuth–Bendix para o Lean quando o `lake` estiver presente; programação inteira (*branch and bound* com certificados).
- ☐ **Tabuleiros**: avaliação por rede treinada pelo autojogo genérico (o laço já existe) para xadrez/shogi pequenos (minishogi 5×5, Los Alamos 6×6); NNUE como programa do compilador; Go 9×9 com rede; hold'em com abstração de cartas.
- ☐ **Autojogo genérico** perde 21 % das linhas ótimas com 8 simulações no jogo da velha, contra 13 % do especializado — falta igualar (rede residual, *temperature schedule*, mais partidas) antes de ir a jogos maiores.
- ☐ **Cena**: MP4 de quadros exatos (o GIF de quadros exatos existe); a direção por gramática de orações ainda não analisa coordenação ("Ana e Bento dançam") nem subordinadas.

## Rodada 0.11 — o que ficou aberto

- ☐ **Profundidade aprendida para a cena viva**: a heurística do plano do chão é a reserva; um modelo de profundidade monocular pela eclusa (as camadas e as profundidades já são dados) abriria fotos sem chão (retratos, vistas aéreas) e uma navegação maior. Com segmentação, as pessoas da própria foto viram habitantes.
- ☐ **Direção por um modelo de linguagem**: o esquema de operações da cena sob a decodificação restrita por JSON Schema que o vapor já tem, quando um modelo útil estiver carregado; o vocabulário continua a reserva que relata o que não entende.
- ◐ **Quadros exatos offline**: ✅ 0.12 — GIF de quadros exatos pelo codificador do vapor (até 240 quadros, passo fixo). Falta um laço que fecha (o último quadro = o primeiro) e MP4.
- ☐ **Esboço → fotorrealista** com um checkpoint do usuário (img2img já existe) e a medida que diz se a imagem gerada respeita o esboço (as retas vetorizadas da saída contra as do esboço).
- ☐ **Geometria**: triangularização de Wu (pontos por duas condições quadráticas), desigualdades, provas legíveis; exportar certificados para o Lean (quando o `lake` estiver presente).
- ☐ **Descoberta**: 3×3 (posto 23), o *flip graph* de Kauers & Moosbauer; profundidade mínima das redes; síntese com prova simbólica em 32/64 bits (bit-vetores) em vez de amostra.
- ☐ **Ciência como programas vapor**: o *split-step* (DFT como `linear`), o Boris e o Lennard-Jones como programas — os mesmos bits em todo substrato; DFT de sólidos, eletrólitos, corpos rígidos.
- ✅ **Arquivos assinados** (0.13): Ed25519 sobre o manifesto com a chave do operador; `verify(zip, trusted: …)`; `mix vapor.archive`. Falta: assinatura por KMS/HSM (o mesmo item do `Keys`).
- ☐ **Controles mais fortes na ciência**: o estado coerente e o tunelamento comparam com referência, mas o "controle" do primeiro é uma conservação e o do segundo uma previsão calculada; E×B e HeH⁺ não têm. Um esquema errado rodado (Lie em vez de Strang com passo grande; potencial com sinal trocado) seria o controle de verdade.
- ☐ **Autojogo maior**: um jogo com estado que não cabe na memória (Connect-Four 6×7) e a rede como programa do compilador; ambientes gerados por um adversário (PAIRED) além da aleatorização.

## Rodada 0.10 — o que ficou aberto

- ☐ **Contexto sem fim no motor paginado**: `Vapor.Streaming` usa a sessão densa; falta fixar as páginas das âncoras no anel do `Vapor.Engine` e re-rotacionar por *slot* (muitas sequências), e medir o efeito das âncoras num modelo grande (pesos que esta máquina não baixa) e com *needle-in-a-haystack*.
- ☐ **Letra de mão real**: nenhum leitor de manuscrito é embarcado (fontes manuscritas: 64 % de CER em mãos nunca vistas). O caminho é treinar `test/python/train_ocr.py` em IAM (latim), KHATT (árabe), CASIA-HWDB (chinês) e um conjunto cirílico, numa máquina com rede — o contrato do leitor não muda.
- ☐ **Rótulos de eixo pequenos e serifados**: o digitalizador recusa 12 de 30 gráficos do conjunto difícil (nunca lê errado). Um leitor de dígitos por modelos de forma (como o do CJK) e a segmentação de dígitos que se tocam fechariam boa parte.
- ☐ **Modelos de língua de outro domínio** para o CJK (o atual vem das listas de palavras do Faker: em chinês, numa semente nova, não ajuda) e um para o árabe (em ordem visual).
- ☐ **Física de corpos rígidos** (a ciência da 0.11 cobre quântica, relatividade, tokamak, química e biologia, não isto): rotação e inércia, juntas angulares, contato entre corpos e atrito (XPBD trata todos do mesmo modo); a política do RL dentro do programa (um episódio = uma execução).
- ☐ **Redes grandes**: operadores esparsos no compilador (PageRank e SIR em milhões de arestas); PageRank pessoal na biblioteca (`Vapor.Docs.Library`) como ordenação de RAG.
- ☐ **Fórmulas além da gramática**: matrizes, acentos, `\left…\right`, várias linhas; e fórmulas manuscritas.

## Desempenho

- ◐ GPU: sessões residentes no fabric ✅ (0.8: `OPEN/STEP/CLOSE`, memória direta ou *staging*, gravações reaproveitadas, o motor serve na GPU); faltam a atenção com memória de *workgroup* (fim do `@max_dh = 512`) e o laço de difusão inteiro numa sessão (o protocolo serve; não está medido).
- ☐ FlashAttention como política `:fast` declarada (o softmax online em blocos é outra ordem canônica: não pode ser a canônica).
- ☐ *Prefill* do Mamba num único quadro `RUN` com iterações (hoje um `STEP` por token); servir modelos recorrentes no `Vapor.Engine` (estado por *slot*).
- ☐ Especulação em árvore sem recomputar a página parcial (cópia-na-escrita da página) e integrada ao `Vapor.Engine`; árvores gerais (hoje: ramos raiz-folha).
- ☐ GEMV `sb4` com α/β vetorizados e `k` não múltiplo de 256; AVX-512 VNNI/AMX. (GEMV predicado para `sb4`: ✅ 0.8.)
- ◐ Fusão: 20× mais rápida ✅; *streaming* disco→disco ✅ (0.7, métodos elemento a elemento, mesmos bytes); falta o RegMean como programa no worker (Cholesky e as matrizes de calibração; hoje `O(d³)` na BEAM e com os modelos em memória) — pedido no anexo da 0.8, adiado sem dor medida.
- ◐ Paralelismo de tensor: entre nós BEAM ✅ (0.8: fragmentos residentes com SHA-256, *failover* sem deriva, réplicas comparadas bit a bit); falta a atenção fragmentada por cabeças e servir um modelo inteiro pelo `Vapor.Engine` em cluster.

## Modelos e operadores

- ◐ SSMs: Mamba ✅, **Mamba-2 ✅** (0.8); faltam Jamba/Zamba/Bamba (híbridos: cache KV **e** estado por sequência no motor) e Falcon-Mamba (RMS de B, C, Δ dentro do misturador); *soft-capping* de atenção (Gemma 2); `relu²`; NTK dinâmico e LongRoPE (recusados pelo nome).
- ☐ RoPE 2D/3D (M-RoPE do Qwen2-VL, DiTs de vídeo) — quando um modelo admitido precisar, conferido contra o `transformers`.
- ☐ Tokenizadores Unigram e WordPiece (famílias T5/BERT).
- ◐ Busca de imagem por significado: texto nas imagens via OCR ✅, torres de visão **e de texto** do CLIP conferidas contra o `transformers` ✅; falta ligar as duas torres à biblioteca (`Vapor.Docs.Library`) — só tem sentido com pesos CLIP treinados, que não são embarcados.
- ◐ Whisper: encoder + decoder conferidos ✅; faltam o front-end log-mel idêntico ao do `transformers`, os tokens de tarefa/idioma/tempo do `generate` e uma medida de WER com pesos reais.
- ☐ LLaVA (encoder + projetor + `inject`) como adaptador conferido contra o `transformers`.
- ◐ Difusão: U-Net do diffusers ✅, encoder do VAE ✅, DDIM/Euler/DPM++ 2M ✅, pipelines txt2img/img2img/inpainting = diffusers ✅ (0.9); faltam **ControlNet** (cópia do encoder + convoluções-zero), LoRA de difusão, **SDXL** (duas torres de texto, `add_embeds`), a U-Net de inpainting de 9 canais, sigmas de Karras, DiTs com atenção cruzada a texto (PixArt, SD3, Flux: recusados com quase-acerto) e **medir um SD real** (qualidade e tempo por passo) numa máquina que baixe pesos.
- ☐ Convoluções agrupadas/*depthwise* e transpostas em `Vapor.Spatial`.

## Estúdio (0.9)

- ☐ Vídeo: H.264/MP4/WebM. Um codificador próprio é um projeto inteiro, e o ffmpeg por *shell* viola a regra do produto; o caminho é um **extrator confinado** (abaixo) que fale o protocolo de quadros.
- ☐ Nós do ComfyUI além do subconjunto: `LoraLoader`, `ControlNetApply`, `KSamplerAdvanced`, `UpscaleModelLoader`/`ImageUpscaleWithModel` (traduzível para `image.upscale` quando o modelo for o nosso), `SetLatentNoiseMask`. Cada um traduzido pela semântica documentada, com teste.
- ☐ A chave do checkpoint no cache: hoje é o caminho; deve passar a incluir o digest dos pesos sem relê-los a cada execução (por exemplo, o índice do safetensors e o `mtime`).
- ☐ Lanczos da câmera mais barato: hoje os pesos passam pelo seno corretamente arredondado, ~0,25 s por quadro 320×192 num núcleo. A identidade sin(π(d+k)) = (−1)ᵏ sin(πd) reduziria a um seno por pixel de saída, mas muda os bits; tem de entrar com recalibragem dos testes de paridade.
- ☐ Destilação robusta a perturbações adversariais (o sentido defensivo, oferecido na DIRETRIZ §12): a perda do aluno sob perturbação limitada, contra o aluno comum.
- ☐ Medida do **excesso de recusa** de um modelo (recusas em pedidos benignos), como um portão calibrado com controles.
- ☐ RL: PPO e DQN como programas; formato de dataset do LeRobot para as demonstrações; um ambiente de jogo.
- ☐ 3D: NeRF ou *Gaussian splatting* conferidos contra uma referência; imagem → 3D só com pesos treinados.
- ☐ Manifestos de contexto assinados para o servidor MCP (o que um agente recebeu, com raiz) e SFT/DPO no `Vapor.Train`.

## Leitura (documentos e visão)

- ☐ **Extratores plugáveis confinados**: um lançador que aplica seccomp a um binário alheio e fala o protocolo de quadros do worker — só então formatos externos (CAD, RAW de câmera…) entram sem executar código hostil na BEAM.
- ◐ OCR: texto impresso horizontal ✅; ordem de leitura em colunas ✅ e feixe CTC com modelo de língua ✅ (0.7); **tabelas** com réguas ou filetes ✅ (0.8: estrutura exata em 12/12, CER por célula 5,5 %); faltam tabelas sem régua, tabelas entre páginas, tokens curtos de cabeçalho ("T1"), fontes geométricas (URW Gothic: 7,5 %), manuscrito, um corpus do domínio do usuário.
- ◐ Outros sistemas de escrita: árabe com RTL ✅, CJK ✅, cirílico ✅, fórmulas → LaTeX ✅ (0.10, [OCR.md §3g–§3k](OCR.md)); faltam modelo de língua para árabe e cirílico, Nastaliq, CJK vertical, fórmulas de várias linhas e matrizes, e **letra de mão real** (abaixo).
- ◐ Fala: dígitos falados ✅; faltam mais vozes, aumento de dados e vocabulário além de dígitos (ou o Whisper com pesos reais).
- ◐ Imagens de escaneados em PDF: CCITT Group 3/4 ✅, LZW e RunLength ✅ (0.7, = libtiff); **JBIG2 aritmético** ✅ (0.8, = jbig2dec); **JBIG2 Huffman e meio-tom** ✅ (0.15, conferidos por um codificador independente e pelo jbig2dec); faltam Huffman com refinamento e JPX.
- ☐ Índice incremental da biblioteca e sua persistência no servidor.

## Numérica e verificação

- ☐ **Kernel NTT** no worker: exige multiplicação inteira alta (`mulhi` 64 bits) nos cinco codificadores (hoje a NTT exata roda na BEAM).
- ☐ Envelope analítico para `log`, `softplus` e `div` (hoje `:na` — a escada mede, não limita).
- ◐ Lean: regras de reescrita ✅ (0.8, `Binary32.lean`, finitos; NaN fora por princípio); falta a dobra de constantes (hoje: a semântica do oráculo, conferida por teste) e a correspondência formal do emulador RVV com a ISA (exigiria a especificação da RVV 1.0 em Lean).
- ☐ Kits DO-178C / IEC 62304.
- ☐ Autodiff de atenção, RoPE, gather e RMSNorm; treino multinúcleo/GPU; `Vapor.Train` como *callback* da eclusa.

## Agentes, ecossistema, ZK

- ◐ Atestados ancorados num log de transparência ✅ (`Vapor.Tlog`, testemunha, verificador no navegador); faltam *tiles* C2SP (`tlog-tiles`) para logs de centenas de milhões de entradas e testemunhas de terceiros de fato rodando.
- ☐ *Lease* sobre o `intent`; KMS/HSM para `Keys`; diário com acréscimo O(1); `fsync` do diretório no `Store.File`.
- ☐ Ecto/Postgres, Oban, LiveView testados com serviços reais (*store* Ecto oficial, `UNIQUE(run_id, seq)`); *streaming* zero-cópia no transporte Plug/Bandit; imagem Nerves e AOT para microcontroladores (um *backend* MVE/Helium novo). *Sem esses serviços nem dispositivos neste ambiente.*
- ◐ Dossiês de auditoria ✅ (0.8); faltam perfis de mapeamento para outras normas (NIST AI RMF, ISO/IEC 23894) e assinatura por KMS/HSM.
- ◐ `pattern`/`format` de JSON Schema ✅ (0.7: ECMA-262 → autômato de bytes; `date`, `time`, `date-time`, `uuid`, `ipv4`, `email`, `hostname`); faltam `ipv6`/`uri`, a interseção de `pattern` com `minLength`/`maxLength` (hoje recusada) e `\b`/olhar adiante (não regulares por bytes, ou caros).
- ☐ Requantização `s32 → s8`; PLONK/STARK sem *setup* por circuito; cotas de ruído BFV propagadas. *Recusados por ora: CKKS/BFV próprios, zkVM sem caso, "zkFHE de 70 B".*

## Menores (cada um, horas)

- ☐ Log-softmax do juiz de checkpoints (`Vapor.Quality.Model`) no substrato — agora possível com o `log` canônico.
- ◐ Reivindicar diretórios HF pelo índice de tensores antes de ler os pesos: feito para a fusão em *streaming* (`Lock.select/1` sobre o catálogo do safetensors); falta usá-lo no `Lock.open/2` para recusar antes de ler gigabytes.
- ☐ Exportar para o HF fatores de RoPE que chegaram como tensor GGUF.
- ☐ Portão de imagem recalibrado com fotos retidas (hoje: cenas sintéticas).
- ☐ Selar os nomes de ferramentas nos diários quando a política exigir.
- ☐ Nx: termo `view`, reduções em outros eixos, `dot` em lote; ponte zero-cópia com Python.
- ☐ Testar `send_body/5` e o token do console no transporte Plug (sem Plug neste ambiente).

## Prioridade proposta

| | item | por quê |
|---|---|---|
| P0 | GPU real (sessões medidas, atenção com *workgroup*) | a sessão residente existe; falta o número que importa |
| P1 | silício real; Apple Silicon | tira o asterisco da emulação; o maior parque de máquinas de desenvolvedor |
| P1 | Engine com modelos recorrentes e híbridos (Jamba); árvore integrada; atenção por cabeças entre nós | servir SSMs e híbridos no motor; modelos maiores que um nó |
| P2 | ControlNet, SDXL, um SD real medido; DiTs com texto; CLIP ligado à biblioteca | o estúdio com geração condicionada; busca semântica de fotos (com pesos) |
| P2 | extrator confinado para vídeo (H.264/MP4) | o estúdio lendo e escrevendo o vídeo que as pessoas têm |
| P2 | extratores confinados; tabelas sem régua; JPX | o resto dos escaneados de escritório, formatos externos sem risco |
| P2 | motor de ofertas no worker; dados de mercado reais; gerador inteiro (Philox) no compilador | a mesa de operações com latência de máquina e dados do mundo; o Monte Carlo com um gerador moderno |
| P3 | kernel NTT; Unigram/WordPiece; treino; homologação | escopo maior, retorno mais tardio |
