# vapor — arquitetura

Este documento descreve o sistema como ele está implementado e testado neste
repositório: fluxo de dados, protocolos de IPC, modelo de memória, modelo
numérico, escada de verificação e o que está provado, o que está testado e o
que é confiado. A última seção registra onde a diretiva original foi
refinada por escrutínio técnico, e as limitações que permanecem.

## 1. Visão geral

```
                    ┌──────────────────────── BEAM (plano de controle, deps: []) ────────────────────────┐
  Program ──▶ Rung 1 (sorts) ──▶ Rewrite (ε=0) ──▶ Lower + cut sweep ──▶ KIR ──▶ seleção por ISA
   (termos       │                                   │                          │
   simbólicos)   │                                   ▼                          ▼
                 │                          SPIR-V (assembler)      liveness ─▶ linear scan (grupos g)
                 │                                   │                          │
                 │                                   │              checker extraído do Lean (aceita?)
                 │                                   │                          │
                 │                                   ▼                          ▼
                 │                            módulos .spv     x86-64 AVX2 · AVX-512 · AArch64 · RV64GCV (bits)
                 ▼                                   │                          │
   Rungs 2–5: admissão · adjunta · oráculo · paridade + envelope (execução real nos substratos)
                 │
   Rung 6: certificado Ed25519 (determinístico, co-assinável) ──▶ Bundle ──▶ nó de borda (só verifica)
                 │
   Vapor.run ──▶ árbitro (3 tetos, perfis declarados) ──▶ Dispatch com failover
                 └───────────┬───────────────────────────┬───────────────────────────┬───────────┘
                     {packet,4} stdio            {packet,4} stdio                   (puro)
                             ▼                           ▼                             ▼
                 vapor-worker (processo)        vapor-fabric (processo)           Oracle (BEAM)
                 seccomp · W^X · watchdog       Vulkan compute completo           binary32 exato
                 pool de threads · sessões      lavapipe / GPU real
                 nativo | interpretador RVV
```

Nada gerado executa dentro da BEAM. Não há NIF (o teste de auditoria proíbe
`load_nif`/`@on_load`); o único código nativo que a BEAM carrega é o da
própria VM.

| Componente | Onde | Linhas (0.13) |
|---|---|---|
| Núcleo: álgebra, compilação, emissores, verificação, runtime, certificado, eclusa de substratos | `lib/vapor/{algebra,compile,kir,emit,verify,runtime}` e módulos de topo | 12 853 |
| Código extraído do Lean (gerado) | `lib/vapor/extracted.ex` | 166 |
| Worker, pool de threads, contadores, interpretador RVV, sandbox, daemons Vulkan e Metal | `native/src/` | 4 717 |
| Provas Lean 4 + extrator | `proofs/` | 1 515 |
| Todas as camadas (modelos, documentos, estúdio, mesas, finanças…) | `lib/` | ~78 000 |
| Testes (Elixir, scripts Python e Node dos níveis diferenciais) | `test/` | ~21 000 |

O ecossistema construído sobre o núcleo (modelos Llama/Mistral/Qwen2,
tokenizador, motor de geração, servidor OpenAI, autodiff/LoRA, ingestão
safetensors/GGUF) e o seu escrutínio estão em
[ECOSSISTEMA.md](ECOSSISTEMA.md); o §4.6 abaixo descreve as formas de
paralelismo e concorrência e o invariante que cada uma preserva.

## 2. Álgebra e programas

`Vapor.Algebra.Term` é uma álgebra livre *simbólica*: todo operador é um nome
com semântica fixa (`Vapor.Runtime.Oracle`). O antecessor guardava closures
Elixir dentro dos termos, o que nenhum emissor consegue baixar para código de
máquina.

- Geradores: `input`, `const`, `ew` (ι: `add sub mul fma neg relu`, com
  `splat`), `qgemv` (contração com matriz `:sb4`), `gemm_i8` (contração em
  ℤ/2³²ℤ).
- **Dimensões semi-dinâmicas**: `{:dyn, :S, max}`. Todas as cotas são
  estabelecidas no máximo; os lemas de monotonicidade
  (`admissible_mono`, `withinEnvelope_mono`) garantem que o certificado vale
  para todo `S ≤ max`. `Compiled.dims/2` rejeita extensões acima do máximo.
- **Programas recorrentes** (`Vapor.Program`, `state: [h: :h_next]`)
  realizam o gerador `scan` sobre o monoide afim σ = 2 sem materializar a
  sequência: o loop inteiro roda no worker atrás de **uma** travessia da
  BEAM, com saída por token em streaming.
- Dados numéricos vivem em binários contíguos (`Vapor.Tensor`). Não há
  `Enum.at` no código; onde há acesso aleatório usa-se tupla (O(1)); as
  listas aparecem só em travessias sequenciais transitórias do oráculo.

## 3. Compilação

### 3.1 Reescrita (ε = 0)
`Vapor.Compile.Rewrite` só admite identidades válidas bit a bit para todo
IEEE-754, zeros com sinal incluídos: `neg(neg x) → x`, `x·1 → x`,
`x + (−0) → x`, `x − (+0) → x`, mas **não** `x + (+0) → x` (pois
`(−0) + (+0) = +0`). O dobramento de constantes avalia a própria semântica
declarada.

### 3.2 KIR, seleção e o fator de grupo
Os kernels (`Vapor.KIR.Kernels`: `ew` fundido, reduções, `gemv_f32` e
`gemv_bf16`, `sb_sums`, `gemv_sb4`, `gemm_i8`, e os operadores de modelo —
`gather_row`, `rope`, `kv_write` contíguo e paginado, `attention` contígua e
paginada, `sample`, `transpose`) são escritos uma vez em IR portátil sobre registradores virtuais
com *tipos* (`:strip`, `:f16l`, `:i32acc`, …). Cada backend dimensiona os
tipos. O **fator de grupo `g` generaliza o LMUL do RVV para todas as ISAs**:
um strip AVX2 com `g = 4` são quatro ymm em lockstep; um strip RVV com
`g = 8` é um grupo m8.

| tipo | RVV | AVX2 | AVX-512 | NEON |
|---|---|---|---|---|
| `:strip` | m`g` | `g` ymm | `g` zmm | `g` q |
| `:f16l` (16 lanes f32) | m4, vl=16 | 2 ymm | 1 zmm | 4 q |
| `:i32acc` | m`4g` | `g` ymm | `g` zmm | 2 q |

O backend AVX-512 (`Vapor.Emit.X86.AVX512`, nível x86-64-v4) reaproveita o
lado inteiro do AVX2 e codifica todo o resto em EVEX: 32 registradores zmm
(zmm31 de rascunho), a largura canônica de 16 lanes num registrador só,
caudas de strip feitas por uma passada mascarada (`bzhi` → `k2…k5`, só
loads/stores mascarados: lanes inativas computam e são descartadas) em vez
de laço escalar, e constantes de 4 bytes lidas com broadcast embutido
`{1to16}`. Seleções viram `vcmpps → k1` + `vblendmps`; a árvore de redução é
a mesma do AVX2 lane a lane, logo os bits são os mesmos.

A seleção (`select/2` em cada backend) acontece **antes** da alocação, como
em compiladores de produção; instruções "lanewise" são emitidas como *bundles*
(uma unidade de alocação), o que permite destino e fonte compartilharem
registradores quando a fonte morre ali.

### 3.3 Alocação linear scan sem spill, verificada
`Vapor.KIR.Liveness` calcula liveness por ponto fixo sobre o CFG (valores
vivos no back-edge cobrem o loop inteiro); `early clobber` modela as regras
de sobreposição das instruções de alargamento do RVV. `Vapor.KIR.RegAlloc`
faz linear scan com blocos alinhados de tamanho 1/2/4/8 (preferência:
caller-saved, depois "buddy" ocupado, depois ordem da ISA).

Não existe caminho de spill. Se falta registrador, o alocador devolve o ponto
de pressão e: (a) o compilador tenta `g` menor; (b) o `gemv_sb4` tenta menos
linhas por iteração (R ∈ {4, 2, 1}); (c) o **cut sweep** fecha a região
fundida ali. Toda alocação aceita é revalidada por `check_alloc/3`,
**extraído do Lean** e provado correto (`checkAlloc_sound`): grupos
alinhados, dentro do arquivo, fora dos reservados, e nenhum par de valores
simultaneamente vivos compartilha registrador. Logo, cada região emitida é
livre de spill e de clobber por construção, não por suposição sobre a
heurística.

### 3.4 Codificação binária pura
- x86-64: REX, VEX de 3 bytes, EVEX (AVX-512, sempre disp32 — nunca o
  disp8·N comprimido), ModR/M, SIB; ABI SysV (`rdi = args`, callee-saved
  salvos só se usados, `vzeroupper; ret`).
- AArch64: palavras A64/NEON; AAPCS64 (`x0 = args`, `x19–x28` e `d8–d15`
  salvos se usados; `x16`/`v31` scratch; `x18` nunca alocado).
- RV64GCV: `vtype = vlmul[2:0] | vsew[5:3] | vta | vma`; loads/stores
  vetoriais em LOAD-FP/STORE-FP com campo de largura; FP escalar com modo
  estático RNE; ABI psABI completa com `ret`; desvios condicionais como
  `b<inverso> +8; jal` (alcance de ±1 MiB independente do tamanho).
- SPIR-V: assembler simbólico que interna tipos/constantes, ordena as seções
  lógicas e emite palavras; `NoContraction` em todo `FMul/FAdd/FSub` na
  política canônica.

Os testes validam cada instrução emitida — todo construtor de kernel, nos
quatro backends, em todo fator de grupo e política — contra o GNU binutils
(x86 incl. EVEX, AArch64 e RISC-V com `rv64gcv`) e cada módulo SPIR-V contra
o `spirv-val`; o produto nunca invoca essas ferramentas.

## 4. Substratos, isolamento e protocolos

### 4.1 `vapor-worker` (Substrato I)
Executável Zig estático, sem libc (240–340 KB). O HELLO fixa o número de
threads: o pool (`pool.zig`) e os contadores de eventos (`perf_event_open`:
ciclos/instruções/cache quando há PMU; task-clock, faltas de página e trocas
de contexto sempre) são criados **antes** do filtro seccomp, instalado com
`TSYNC` em todas as threads. Cada chamada carrega um *descritor de
partição* (argumento de contagem, grão, ponteiros que avançam por unidade,
scratch por thread, guarda opcional) e o worker a divide entre as threads;
como as unidades são linhas independentes, o resultado é o mesmo para 1…N
threads. A passagem de bastão é espera ativa limitada (2 ms) e depois futex.
Programas ficam **residentes** em sessões (OPEN/STEP/CLOSE): pesos
mapeados uma vez, estado (caches KV) mantido entre passos, cada passo
escreve só as entradas dadas e devolve só as saídas pedidas. Por RUN ou STEP:

- **nativo**: o blob é copiado para uma página RW, a página vira R+X (W^X),
  cache de instruções sincronizado (AArch64/RISC-V), chamada como
  `void k(const uint64_t *args)`;
- **emulado**: o blob RV64GCV é interpretado por `rvemu.zig`: todo acesso à
  memória é checado contra os buffers vinculados, o fetch contra o blob, e
  instrução desconhecida vira `IllegalInstruction` com pc e palavra.
  Semântica RVV 1.0 fiel (RNE, `vfmacc` fundido, NaN canônico, NaN-boxing,
  alinhamento de grupos, `vill`), com modo *poison* que escreve 1s em
  elementos agnósticos de tail/máscara — prova que os kernels não dependem
  deles. VLEN configurável (128–512).

Contenção: seccomp-BPF (allowlist: read, write, openat, close, lseek, mmap,
munmap, mprotect, clock_gettime, setitimer, sinais e saída; arquitetura
auditada), `PR_SET_NO_NEW_PRIVS`, e um watchdog `SIGALRM` para código
nativo que não termina. Os testes provocam as quatro classes de falha
(SIGILL, SIGSEGV, SIGALRM, SIGSYS): em todas, só o worker morre, o
`GenServer` dono da porta reporta `{:worker_crashed, {:signal, …}}`,
respawna e a unidade seguinte roda.

### 4.2 `vapor-fabric` (Substrato II)
Daemon Zig (libc só para o `dlopen` do loader), bindings Vulkan escritos à
mão. Pipeline headless completa: instância → dispositivo físico com fila de
compute → dispositivo lógico → buffers → `vkCreateShaderModule` com o
SPIR-V emitido pela BEAM → descriptor set layouts → pipeline layouts com
push constants → `vkCreateComputePipelines` → descriptor pool/sets → um
command buffer por RUN (dispatches, barreiras, janelas por iteração, cópias
de estado, staging das emissões) → submit → fence com deadline → leitura.
Queda do driver ou `VK_ERROR_DEVICE_LOST` matam só o daemon; timeout de
fence é tratado como device lost.

### 4.3 Protocolo (ambos os processos)
Frames `{packet, 4}` no stdio (comprimento big-endian); campos internos
little-endian. Um processo que morre no meio de um frame não pode
dessincronizar a BEAM.

```
HELLO   1 | flags:u32 | threads:u32      → 1 | arch | sandbox | version:u32 | threads:u32
RUN     2 | mode:u8 | vlen:u32 | flags:u32 | fuel:u64 | deadline_ms:u32
          | code_len:u32 | code
          | nbuf:u32 | (kind:u8 writable:u8 len:u64 [data | path_len:u16 path offset:u64])*
          | ncall:u32 | (entry:u32 nargs:u32 (0 imm:u64 | 1 buf:u32 off:u64
          |                                    | 2 buf:u32 base:u64 stride:u64)*)*
          | iters:u32 | nemit:u32 (buf base stride len)* | ncopy:u32 (src soff dst doff len)*
          | nret:u32 (buf)*
EMIT    3 | t:u32 | bytes                  (por iteração, em streaming)
DONE    4 | elapsed_ns:u64 | retired:u64 | n:u8 (id:u8 value:u64)* | (len:u64 bytes)*
ERR     5 | code:u32 | pc:u64 | word:u32 | msg
OPEN    6 | (como RUN, sem calls)         → sessão residente: código, buffers, constantes
STEP    7 | deadline | writes (buf off bytes)* | calls | returns (buf off len)*
          | [ncopy:u32 (src soff dst doff len)*]   (opcional, desde 0.6)
CLOSE   8
```

As cópias opcionais do STEP são a realimentação de estado que não é
atualizado no lugar — o `s ← s_next` de um modelo recorrente (Mamba) —,
feitas dentro do worker depois da passada, como as do RUN entre iterações:
o estado nunca atravessa para a BEAM. Um STEP sem elas é byte a byte o de
antes.

Cada chamada pode trazer até dois descritores de partição (`count, grain,
ptrs (arg, stride)*, scratch (arg, bytes)*, guard`); o primeiro aplicável é
usado. As tabelas do worker e do daemon são dimensionadas pelo próprio
quadro (uma contagem hostil não aloca mais do que o quadro descreve).

O frame RUN do fabric troca `code/calls` por módulos SPIR-V e dispatches
(`module, gx, gy, gz, push*, binds*`, com binds do tipo buffer ou *janela*
`buf, base, stride, len`). Planos são autocontidos (sem estado no worker):
um processo reiniciado não precisa de replay.

### 4.4 Memória
- Pesos ≥ 64 KiB vão uma única vez para `/dev/shm/vapor-<sha256>`
  (endereçamento por conteúdo, escrita atômica) e são mapeados
  copy-on-write pelo worker e **importados sem cópia** pelo fabric via
  `VK_EXT_external_memory_host` quando o dispositivo o oferece.
- Entradas por token são *janelas*: o daemon copia a fatia `t` antes da
  iteração `t`; o worker passa `base + t·stride`.
- Realimentação de estado (`h ← h_next`) é uma cópia declarada no plano.
- Réplicas e reinícios mapeiam os mesmos arquivos: `n` workers com o mesmo
  modelo ocupam uma cópia dos pesos no cache de páginas.
  `Vapor.Runtime.Shm.prune/0` (`mix vapor.shm`, e ao fim da suíte de testes)
  remove os arquivos que nenhum processo mapeia; quem precisa de um nome
  removido o reescreve (`put/1` verifica a cada vez).
- Pesos podem ficar em **bfloat16** (`storage: :bf16`): metade dos bytes;
  `vld_bf16` alarga 16 pesos exatamente ao carregar (`vpmovzxwd` + shift,
  `SHLL #16`, `vle16`+`vzext.vf2`+`vsll`, palavra/meia-palavra no SPIR-V),
  então o resultado é bit a bit o do programa f32 sobre os mesmos valores.

### 4.5 Failover
`Vapor.Runtime.Dispatch` tenta a escolha do árbitro e desce a cadeia
`fabric → host AVX-512 → host base → interpretador RVV → oráculo`,
registrando o motivo de cada salto. O fabric também recusa por limite
estático (cabeça de atenção acima de 512), e a unidade desce a cadeia. Sob a política canônica todos os elos computam os mesmos bits,
então o reroteamento nunca muda a resposta; o oráculo não falha para um
programa certificado.

### 4.6 Paralelismo e concorrência

A garantia a preservar: os bits de cada saída não dependem de quantas
threads, de quantas sequências dividem o passo, de onde o KV mora, de qual
réplica atende nem de qual substrato executa. Toda forma abaixo tira
paralelismo de **saídas independentes**, nunca de reassociar uma soma.

| forma | onde | invariante | teste |
|---|---|---|---|
| SIMD | AVX2, AVX-512, NEON, RVV (VLEN 128–512), SPIR-V | mesmos bits que o oráculo | `canon_test`, `native_test`, `model_test` (QEMU, lavapipe) |
| ILP | R linhas por iteração no GEMV (4/2/1, escolhido pelo alocador) | idem | idem |
| threads intra-operação | pool no worker, descritor de partição por chamada; atenção de decode dividida por cabeça KV | 1 = 2 = 3 threads | `threads_test`, `model_ops_test` |
| lote contínuo | `Vapor.Engine`: decode + pedaços de prefill no mesmo passo | tokens de uma sequência iguais sozinha ou em lote, qualquer fatiamento | `engine_test` |
| KV paginado | `kv_write_paged` / `attention_paged` com tabela de blocos | paginado = contíguo | `model_ops_test` |
| réplicas (dados) | `Vapor.Engine.Pool`: uma compilação, páginas de pesos compartilhadas, menor fila | mesma resposta por qualquer réplica; réplica morta só leva as suas requisições | `engine_test` |
| concorrência BEAM | processo por conexão HTTP, motor como `GenServer`, SSE | — | `serve_test` (cliente `openai`) |
| banda de memória | pesos f32 compartilhados (mmap), `bf16` residente, GEMV *weight-stationary* (bloco de linhas de W encontra todo o lote) | bf16 = f32 sobre pesos arredondados | `model_test`, `engine_test` |
| especulação | rascunho propõe, alvo verifica `k+1` linhas num passo | saída idêntica ao alvo sozinho | `speculative_test` |
| GPU | SPIR-V de todos os operadores de modelo | fabric = oráculo (modelo inteiro, passo paginado com amostragem, decode recorrente) | `model_test`, `autodiff_test` |

## 5. Modelo numérico

`Vapor.F32` é aritmética binary32 **exata** na BEAM: valores como padrões de
bits; `add/sub/mul` via binary64 + um arredondamento (correto, pois
53 ≥ 2·24 + 2); `fma` em racionais diádicos com um único arredondamento
(RNE, underflow gradual, overflow para ∞). Validado contra `fmaf`/`+`/`*` do
hardware em 800 000 operações, incluindo subnormais e zeros com sinal.

| Política | Semântica | Garantia |
|---|---|---|
| `:canonical` | mul e add arredondados separadamente; redução em árvore fixa de 16 lanes (`r8 → r4 → r2 → r`); `÷` corretamente arredondada (semântica v2) | **bits idênticos** em x86, ARM, RVV, interpretador e GPU (digest FNV-1a) |
| `:fast` | FMA fundido onde o operador permite | dentro do envelope rigoroso em todos os substratos |

A ideia central: reprodutibilidade não exige lentidão se o paralelismo vier
de **saídas independentes** (linhas, elementos) e não de reassociar uma
soma. Por isso cada linha do GEMV tem exatamente 16 acumuladores, e o
kernel ganha vazão processando R linhas por iteração — escolhido pelo
alocador por ISA — sem mudar um bit.

`Vapor.Verify.Envelope` calcula, para cada saída, o valor real exato e uma
cota rigorosa válida para *qualquer* substrato conforme (arredondamento
correto, FMA fundido ou não, qualquer ordem de redução, underflow gradual
ou flush-to-zero), em racionais diádicos exatos: forma de Wilkinson
`γₙ·S + a` para contrações sobre entradas exatas (decidida pelo
`within_envelope/4` extraído do Lean) e análise de erro corrente para o
resto. O verificador nunca arredonda; só cotas são encurtadas, e só para
cima. Desde 0.6 a forma de Higham (`γₙ` como cota de produtos de
`(1 + δᵢ)`) também está provada em Lean (`proofs/Vapor/Higham.lean`).

**Versão da semântica.** Os bits de um programa são função dos seus termos
**e** da definição das funções canônicas; os certificados registram
`Vapor.Canon.version/0`. Versão 1 (até 0.5): `a/b = a·rcp(b)`, a ≤ 1 ulp.
Versão 2 (desde 0.6): `÷` **corretamente arredondada** (IEEE, com a
convenção DAZ/FTZ das GPUs) — Markstein com resíduo exato de Dekker, sem
FMA — e `log` (≤ 1 ulp do corretamente arredondado). Com isso
`+ − × ÷` são IEEE em todo substrato.

**Eclusa de substratos (0.10).** Um substrato só recebe programas
canônicos depois de medido: as sondas de `Vapor.Substrate` dão o veredito
(`:canonical`, `:envelope` com a impressão numérica, `:refused`) e o
despachante o respeita. O envelope cobre também DAZ (subnormais de
entrada lidos como zero), achado pelo kit no XLA. Metal, StableHLO/PJRT
(Tenstorrent), o cluster e o FreeBSD: [SUBSTRATOS.md](SUBSTRATOS.md).

## 6. Escada de verificação e certificado

| Rung | Estabelece | Como |
|---|---|---|
| 1 | sorts, formas, realimentação | `Program.check/1` |
| — | reescrita exata | `Rewrite` |
| 2 | admissão | alocação aceita pelo checker extraído em **toda** ISA; no-wrap int8 via `admissible/3` no máximo das extensões; limites de SPIR-V |
| 3 | identidade adjunta ⟨Wx, v⟩ = ⟨x, Wᵀv⟩ | kernel compilado vs. transposta exata (erros de layout/transposição) |
| 4 | oráculo diferencial | cada substrato de CPU bit a bit igual ao oráculo exato |
| 5 | paridade + envelope | digests FNV-1a iguais entre substratos; toda saída dentro do envelope |
| 6 | proof-carrying code | certificado Ed25519 com quórum por co-assinatura |

O payload contém SHA-256 do programa, de cada blob de máquina e de cada
módulo SPIR-V, o digest das fontes Lean dos checkers, as evidências de cada
rung e o **trabalho contado** (FLOPs, bytes, instruções por ISA). Não contém
nada dependente do host (nem tempos, nem decisão de despacho), então nós
independentes que refazem a escada produzem payloads **idênticos byte a
byte** e podem co-assinar (`Certificate.cosign/3`). `verify/3` exige
`quorum` assinaturas válidas de chaves distintas confiáveis. `Bundle`
empacota artefatos + certificado; o nó de borda verifica assinaturas e
hashes em ~2 ms e executa sem refazer a escada.

## 7. Árbitro: roofline de três tetos

`t = Σ max(F/π, Q/β, I/ι) + overheads`, com `F`, `Q` contados do schedule e
`I` = contagem estática das instruções do corpo de cada loop quente, *como
emitido*, vezes o número de iterações. O terceiro teto é o que torna o
modelo honesto: o GEMV de 4 bits é limitado por emissão de instruções
(~1,1 instr/peso), não por banda, e um modelo de dois tetos erra por ~5×.
Com os três, o GEMV 4096×4096 é previsto em 71,7 ms e observado em
72–75 ms. Os perfis são **declarados** (tabela por fornecedor; o Mesa
lavapipe tem perfil de classe CPU). A decisão é o argmin do tempo previsto,
tomada em cada nó a partir do trabalho certificado e do seu próprio perfil.

## 8. Modelo de garantias

**Provado em Lean 4 (núcleo apenas, sem `axiom`/`sorry`, sem aviso):**
- ℤ/2³²ℤ: associatividade/comutatividade de soma e produto; `wrap32` é
  homomorfismo; **toda árvore de redução com somador que dá a volta computa
  o valor exato** sob `K·|A|max·|B|max < 2³¹` (`integer_parity`), para
  qualquer permutação das folhas; monotonicidade da admissibilidade.
- Bancos: lema de Euclides a partir de `gcd`; stride coprimo ⇒ injetor nas
  lanes (`bank_injective`); pad em forma fechada para 2ᵐ bancos, suficiente
  e **mínimo**.
- Monoide afim: associativo, neutro (1, 0), e composição = aplicação em
  sequência; levantamento segmentado associativo para qualquer monoide.
- Checker de alocação: correção (`checkAlloc_sound`).
- Envelope: a decisão exata, monotônica na extensão; forma bilateral.
- Profundidade de Estrin = 3 vs. Horner = 7, calculada sobre a árvore.

**Extraído para Elixir** (`lake exe vapor-extract`): `check_alloc`,
`admissible`, `wrap_s32`, `pad_stride`, `bank_of`, `affine_op`,
`seg_affine`, `within_envelope`. O extrator lê os termos *elaborados* — os
mesmos sobre os quais os teoremas falam — e falha em qualquer construção
fora do fragmento. O Lean calcula 480+ vetores de conformidade que o Elixir
extraído precisa reproduzir; o módulo gerado carrega o FNV-1a das fontes
Lean, conferido pelo teste de auditoria sem precisar do toolchain.

**Testado (não provado):** encoders (binutils, QEMU, lavapipe),
interpretador RVV vs. QEMU em VLEN 128/256/512, igualdade bit a bit entre
todos os substratos, contenção de falhas.

**Base confiável:** a BEAM e o kernel Linux; o printer do extrator (~200
linhas) e seu prelúdio de 6 linhas; o *modelo padrão* da IEEE-754
(`fl(x op y) = (x op y)(1 + δ)`, `|δ| ≤ 2⁻²⁴`, longe do underflow — uma
propriedade da especificação do hardware; o lema de Higham que parte dele
está provado desde 0.6); o driver Vulkan (contido em processo).

## 9. Escrutínio da diretiva

1. **`VK_KHR_external_memory_fd` não importa um memfd arbitrário.** Um
   handle OPAQUE_FD só aceita memória exportada pelo mesmo driver. O
   mecanismo que entrega cópia zero a partir de arquivos escritos pela BEAM
   é `VK_EXT_external_memory_host` sobre um mapeamento — o implementado.
2. **Extração Lean→Elixir é de tempo de build, não de execução.** Rodar o
   toolchain Lean em produção reintroduziria o contêiner gigante que a
   diretiva quer eliminar. A frescura é garantida por digest.
3. **A condição de crista da Eq. 3 foi rebaixada a informação.** Com todos
   os tetos na previsão, ela só consegue escolher o substrato mais lento.
4. **A decisão de despacho saiu do certificado**, que agora carrega o
   trabalho contado (portátil e co-assinável); a decisão é por nó.
5. **Uma NIF "segura" foi recusada.** Um interpretador em NIF compartilha o
   espaço de endereçamento da BEAM; um bug nele derruba a VM. O
   interpretador roda no worker.
6. **Nomes:** módulos `Vapor.Emit.*` (não `Aether.Emit.*`), coerente com o
   nome do sistema.
7. **Bugs do antecessor corrigidos:** `vtype` com SEW/LMUL trocados;
   `vle8.v` no opcode OP-V; opcodes de cooperative matrix errados (4448 é
   uma *capability*); `decode_i8` em floats; `axiom` e um "teorema" com
   conclusão `True`.

## 10. Limitações

- Sem hardware físico RVV, ARM ou GPU discreta aqui: RVV/NEON validados sob
  QEMU (e o interpretador contra o QEMU), Vulkan sob lavapipe.
- O caminho cooperative matrix é emitido, validado pelo `spirv-val` e
  selecionado quando o dispositivo anuncia (s8,s8,s32,16×16×16,subgrupo),
  mas o lavapipe não o suporta: **não foi executado**.
- O worker é Linux (syscalls diretas). O código NEON segue o AAPCS64, que o
  arm64 da Apple respeita (x18 reservado, d8–d15 callee-saved), mas um worker
  para macOS/Apple Silicon (`MAP_JIT`, `pthread_jit_write_protect_np`,
  isolamento sem seccomp) não foi escrito — sem máquina para executá-lo.
- Esta VM tem 2 vCPUs e nenhuma PMU: o escalonamento a muitos núcleos e os
  contadores de hardware não foram medidos aqui (o código os lê quando
  existem). O GEMV de 4 bits segue limitado por emissão (o cálculo escalar
  de α/β por sub-bloco e as conversões u8→f32 dominam): o AVX-512 quase não
  o acelera; vetorizar esse cálculo é o próximo passo.
- O fabric executa unidades autocontidas (sem sessões residentes): serve a
  modelos inteiros e à certificação, mas o motor de geração usa o worker.
- A atenção no SPIR-V recalcula os escores em três passadas (sem memória
  compartilhada): correta e bit a bit, não otimizada para GPU real.
- Fatores RoPE que chegam como tensor de GGUF não têm grafia em
  `config.json`; exportar um modelo assim para Hugging Face é recusado.
- Buffers do fabric são host-visible; GPUs discretas pedem device-local +
  staging.
- O oráculo e o envelope são exatos e portanto lentos: limitam o tamanho das
  sondas das rungs 3–5 (as cotas valem no máximo das extensões pela
  monotonicidade).
- seccomp não está disponível sob `qemu-user` (reportado como
  `:unsupported`); o isolamento por processo continua valendo.
- Famílias de fronteira (desde 0.6: MoE esparso, MLA latente, janela e anel,
  Mamba — [FRONTEIRA.md](FRONTEIRA.md)): *soft-capping* de atenção (Gemma 2),
  NTK dinâmico, LongRoPE e os híbridos SSM + atenção são recusados pelo
  nome; o Mamba-2 entra desde 0.8 e o MoE em 4 bits é esparso desde 0.8
  (`qgemv_masked`).
- As funções transcendentais canônicas (`exp`, `log`, `tanh`, `gelu`…) são
  idênticas em todo substrato, mas não são corretamente arredondadas (a ≤ 1–2
  ulps, medido); `+ − × ÷` coincidem com a IEEE bit a bit (÷ desde 0.6, com
  DAZ/FTZ).
- O texto normalizado depende da versão do Unicode do OTP. Ela é registrada
  no digest de agentes e testada separadamente, mas não é eliminada.

## 11. Camadas sobre o núcleo

Desde 0.10: pré-treino e contexto sem fim ([TREINO.md](TREINO.md)), física
para RL e gêmeos digitais ([FISICA.md](FISICA.md)), redes complexas
([REDES.md](REDES.md)), CJK, árabe, figuras e fórmulas ([OCR.md §3g–§3k](OCR.md)).
Desde 0.11: cena viva, esboço e arquivos ([CENA.md](CENA.md)), descoberta de
algoritmos ([DESCOBERTA.md](DESCOBERTA.md)), matemática ([MATEMATICA.md](MATEMATICA.md)),
ciência ([CIENCIA.md](CIENCIA.md)), autojogo ([JOGOS.md](JOGOS.md)).
Desde 0.12: bancada, engenharia, lógica, tabuleiros, proteínas, render.
Desde 0.13: finanças e a mesa de operações ([FINANCAS.md](FINANCAS.md)) —
a primeira camada desde a bancada a **compilar para o núcleo** de novo: o
Monte Carlo escreve o passo da trajetória, o gerador inclusive, como termos
da álgebra (o gerador escolhido por caber **exato** em binary32: produtos
de Lehmer abaixo de 2²³, módulo pelo truque de arredondamento 2²³), e herda
assim a garantia central — os mesmos bits em todo substrato e em toda
contagem de threads, conferidos contra o oráculo na própria resposta. As
outras peças (o livro de ofertas e o seu juiz, o simplex racional, os
portões de ruído) herdam do núcleo o CBOR canônico e as árvores de Merkle
dos diários.


As camadas desta seção usam o núcleo sem alterá-lo: tudo que produzem é
programa certificado ou dado canônico.

| camada | módulos | garantia que herda | documento |
|---|---|---|---|
| saída restrita | `Vapor.Grammar`, `Grammar.{JSONSchema, Vocab, Constraint}`, `Vapor.Tools` | o motor só escolhe tokens que a gramática admite; nenhum prefixo aceito é beco sem saída | [AGENTES.md §6](AGENTES.md) |
| templates | `Vapor.Template`, `Vapor.Chat` | renderização = `jinja2` do HF, byte a byte; relógio explícito | [AGENTES.md §2](AGENTES.md) |
| agentes | `Vapor.Agent`, `Agent.{Spec, Journal, Keys, Store, Backend.*, MCP}` | decisões locais re-deriváveis em qualquer substrato ⇒ repetir é verificar | [AGENTES.md §3](AGENTES.md) |
| RAG | `Vapor.RAG`, `Vapor.Merkle`, `Vapor.Embed` | escores densos são `linear` canônico ⇒ índice igual em qualquer nó | [AGENTES.md §4](AGENTES.md) |
| canônico | `Vapor.Canonical` (CBOR RFC 8949 §4.2), `Vapor.CR` | certificados e diários verificáveis fora da BEAM | [ECOSSISTEMA.md](ECOSSISTEMA.md) |
| transporte | `Vapor.Serve.dispatch/3` + responders (`:gen_tcp`, `Vapor.Plug`) | o mesmo handler, os mesmos recibos | [ECOSSISTEMA_ELIXIR.md](ECOSSISTEMA_ELIXIR.md) |

O motor monitora o destinatário de cada requisição. Um destinatário que
morre retira as suas sequências do lote na fronteira de passo seguinte, e
`cancel/2` faz o mesmo de propósito. Esses dois caminhos são a única forma
de uma sequência deixar o lote antes do fim, e nenhum dos dois altera os
bits das outras (invariância a lote).
