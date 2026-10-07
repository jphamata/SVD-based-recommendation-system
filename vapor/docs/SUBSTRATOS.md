# Substratos: o hardware admitido por medida (0.10)

> Rodada 0.10. Pedido: "suporte Metal (Apple) + Tenstorrent + orquestração de
> cluster vapor". Escrutínio do pedido e o que ficou de fora:
> [DIRETRIZ.md §13](DIRETRIZ.md). Testes: `metal_test.exs`, `substrate_test.exs`,
> `stablehlo_test.exs`, `cluster_test.exs`, `freebsd_test.exs`.

## 1. A dor

Um acelerador novo (uma GPU da Apple, um chip da Tenstorrent, a próxima NPU)
chega com um compilador que **não promete os mesmos números**: soma em
outra ordem, funde `a·b + c` numa instrução, zera subnormais, arredonda
operandos para bf16 sem avisar. O resultado é que "o modelo roda no
dispositivo X" não diz se ele calcula **a mesma coisa** — e a diferença
só aparece em produção, num caso de borda.

O vapor já tinha uma semântica canônica (somas em árvore fixa de 16
lanes, divisão corretamente arredondada; os mesmos bits em todo
substrato) e um **envelope de erro provado** (Wilkinson/Higham diádico)
para os substratos que não a reproduzem. Faltava a porta de entrada.

## 2. A eclusa de substratos (`Vapor.Substrate`)

Um substrato só calcula depois de **medido**. A admissão roda sondas de
resposta conhecida — cada uma isola um comportamento:

| sonda | o que revela |
|---|---|
| `contraction` | `1 + 2⁻¹²` num empate: FMA muda o último bit |
| `subnormal_out`, `subnormal_in` | FTZ (saída) e DAZ (entrada) |
| `signed_zero`, `nan_select` | IEEE ou não |
| `reduction` | a ordem da soma (árvore canônica ou outra) |
| `input_precision` | linhas `1 + 2⁻ᵏ`: quantos bits de mantissa o operando realmente tem |
| `division`, `functions` | divisão e transcendentes canônicas |
| `linear_f32/bf16`, `qgemv_sb4`, `gemm_i8`, `attention`, `sample` | os núcleos de verdade, em formas reais |

A resposta exata do oráculo decide o veredito:

- **`:canonical`** — todos os bits iguais;
- **`:envelope`** — diferente, mas dentro do envelope provado, com a
  **impressão numérica** (`contraction`, `flush_to_zero`,
  `denormals_are_zero`, `reduction_order`, `significand_bits`, …)
  registrada; programas `:canonical` não são despachados para ele,
  programas `:fast` podem ser;
- **`:refused`** — fora do envelope, inteiros que não dão a volta, ou
  operandos com menos de 24 bits de mantissa (medidos: "8 significand
  bits" num motor bf16), ou um dispositivo que trava, responde fora do
  protocolo ou deixa uma saída de fora; e uma diferença **onde nenhum
  limite existe** (divisão, funções, atenção) só passa se as sondas
  mediram uma causa para ela (contração, subnormais zerados, outra ordem
  de soma) — sem causa, é um dispositivo calculando outra coisa.

Uma sonda que não rodou (admissão com `only:`, um kit sem ela) aparece na
impressão como `:unmeasured`, nunca como "canônica".

**Achado (0.10, revisão independente):** a primeira versão olhava o
envelope só das sondas analíticas — um dispositivo errado só na divisão
saía `:envelope` — e quebrava (em vez de recusar) com uma resposta sem uma
das saídas. Corrigido, com os casos na suíte de testes.

A admissão é um registro CBOR canônico **assinado (Ed25519)**; um byte
alterado é pego. O despachante consulta a admissão; um acelerador que
chega é admitido na chegada.

**Achado (0.10):** o kit portátil (§4) rodado no XLA de CPU mostrou
DAZ — subnormais de *entrada* lidos como zero — que o envelope não
cobria: `min(subnormal, 0,25)` saía fora do limite. O envelope agora cobra
DAZ (`max(e, |v|)` por operando suspeito), e continua apertado: o controle
com o dobro do valor exato é recusado (`substrate_test.exs`).

Medido: worker nativo, AVX-512, RVV emulado, lavapipe (Vulkan) e oráculo —
**canônicos**; shim Metal "contraindo" — envelope com FMA; shim com FTZ —
envelope com FTZ e DAZ; motor bf16 simulado — recusado; GEMM int8 que
satura — recusado; XLA de CPU (via kit) — envelope (FMA, FTZ/DAZ, outra
ordem de redução).

## 3. Metal (Apple)

- **Tradutor MSL** (`Vapor.Emit.MSL`): a biblioteca simbólica de núcleos
  SPIR-V (a mesma que gera x86, NEON, RVV e SPIR-V) traduzida para Metal
  Shading Language, com o fluxo de controle **reconstruído** (MSL não tem
  `goto`: laços e seleções estruturados).
- **Daemon `vapor-metal`** (Zig, macOS): o protocolo do fabric
  (HELLO/RUN/OPEN/STEP/CLOSE), o runtime Objective-C aberto em tempo de
  execução (`dlopen` + `objc_msgSend`), `MTLMathModeSafe`, *buffers*
  compartilhados (memória unificada, `newBufferWithBytesNoCopy`).
  Compila de qualquer host (`make metal`), sem SDK.
- **Worker de CPU para macOS**: `MAP_JIT` + `pthread_jit_write_protect_np`
  + invalidação de cache de instruções; `__ulock_wait/wake` no lugar do
  futex.
- **Como se testa sem um Mac:** o mesmo texto MSL compila com `clang` por um
  *shim* de cabeçalhos; o daemon roda em Linux sobre esse *shim*
  (`vapor-metal-sim`). Três "dispositivos" pelo *flag* de compilação:
  conforme (`-ffp-contract=off`) — canônico, bit a bit com o oráculo em
  programas canônicos, SSM com iterações, GEMM, atenção, uma sessão
  Llama e o motor; contraindo (`-ffp-contract=fast -mfma`) — envelope; FTZ
  (MXCSR) — envelope.

**Limite:** o daemon real (`mtl.zig`) é **compilado e verificado em tipos,
nunca executado** aqui. Admitido na chegada pelo mesmo processo, ele dirá
o que é.

## 4. Tenstorrent (e qualquer PJRT)

O caminho honesto não é fingir um *backend* sem o hardware: é **exportar**
para o formato que o compilador da Tenstorrent (tt-xla / tt-mlir) aceita e
**julgar o resultado com a mesma eclusa**.

- **Exportação StableHLO** (`Vapor.Export.StableHLO`, `mix vapor.export`):
  elementwise, reduções, `linear` (e mascarado, agrupado), `gemm_i8`,
  `gather_row`, RoPE, escrita de cache KV, atenção (com janela e
  agrupamento de cabeças), `transpose`, `reshape`; aritmética de índices
  sempre com sinal; constantes densas em hexadecimal. O que não tem
  equivalente exato é **recusado pelo nome** (amostragem, núcleos de 4
  bits, operações paginadas, `gelu` exato).
  Conferido: Llama/Qwen2/Mistral minúsculos exportados e executados no XLA
  de CPU pelo JAX — *logits* a ~10⁻⁶, mesmo argmax.
- **Kit de admissão portátil** (`mix vapor.substrate kit DIR`): as sondas
  como StableHLO, um `run_kit.py` (`--platform tt`, `tpu`, `cpu`, …) que
  roda no dispositivo pela PJRT, e `mix vapor.substrate judge DIR [--sign CHAVE]`
  que traz as respostas de volta e emite a admissão assinada com a chave do operador (a de `mix vapor.audit keygen`: a assinatura diz *quem* admitiu). O
  dispositivo nunca precisa da BEAM.

**Limite:** sem placa Tenstorrent aqui; o kit foi rodado no XLA de CPU.

## 5. Orquestração de cluster (`Vapor.Cluster`)

O determinismo muda o que um orquestrador pode fazer:

- **Cache por conteúdo em todo o cluster**: a chave de um trabalho é o
  digest do programa e das entradas; repetir é um acerto de cache, e o
  resultado é o mesmo em qualquer nó.
- **Auditoria por execução redundante**: uma amostra dos trabalhos roda em
  dois nós e os bits são comparados; uma divergência é decidida pelo
  oráculo, o nó que errou vai para **quarentena** com a evidência e só
  volta **por medida** (a eclusa no próprio nó). A amostra é um *hash*
  com chave (`H(sal ‖ chave) < taxa·2²⁵⁶`): quem tem o sal recalcula quais
  trabalhos foram conferidos — a amostra não pode ser escolhida depois.
- **Falha e *hedging* sem mudar um bit**: um nó perdido tem seus trabalhos
  refeitos em outro; um retardatário é ultrapassado por uma cópia.
- **Treino distribuído com os bits de uma máquina** (§ [TREINO.md](TREINO.md)).

Testado com nós `:peer` reais (`cluster_test.exs`): um nó que vira um bit
é pego e posto em quarentena — e, sem a auditoria (o controle), as
respostas erradas passam; readmissão recusada enquanto mente; nó perdido;
*hedging* (e o controle sem *hedging* espera o retardatário).

## 6. FreeBSD

O worker de CPU compila para `x86_64-freebsd` e `aarch64-freebsd`
(`make freebsd`; o zig traz os cabeçalhos da libc do FreeBSD): libc, futex
por `_umtx_op`, páginas de código W^X por `mprotect`, e o isolamento por
**Capsicum** — depois de `cap_enter` o processo não tem espaço de nomes
global (nada de `open` por caminho, sockets, `execve`, novos processos);
guarda só o que já tem: stdio, memória e **um descritor de diretório
limitado a busca, leitura, `seek`, `fstat` e mapeamento só-leitura**, pelo
qual os pesos são abertos (`openat`). A mesma política do filtro seccomp
do Linux, por capacidades em vez de lista de chamadas.

**Limite:** compilado e conferido (o binário é FreeBSD, importa
`cap_enter`, `cap_rights_limit`, `_umtx_op`, `openat`, `mprotect`, e não
importa nada do JIT da Apple — `freebsd_test.exs`; o seccomp do Linux é feito por chamadas cruas, invisíveis na tabela de importação: fica fora por compilação condicional, não por conferência); **não
executado** — não há FreeBSD nesta máquina. O fabric Vulkan não foi
portado (usa chamadas Linux cruas).
