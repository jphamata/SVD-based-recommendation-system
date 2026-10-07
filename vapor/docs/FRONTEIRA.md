# Servir modelos de fronteira sem perder os bits

Os modelos que a indústria serve em 2026 não são mais um Llama denso: são
misturas de especialistas (Mixtral, Qwen3-MoE, DeepSeek-V3), atenção latente
(MLA), janelas deslizantes (Mistral, Gemma 3), modelos de estado (Mamba) — e
são servidos com decodificação especulativa e espalhados por várias
máquinas. Cada um desses truques, do jeito como costuma ser implementado,
**troca a reprodutibilidade por velocidade**: o resultado passa a depender
do tamanho do lote, da ordem dos tokens, do número de GPUs.

Esta página mostra como o vapor implementa cada um **sem** essa troca — a
saída continua sendo uma função dos termos, igual em todo substrato — e o
que cada um custa. Números: [bench/FRONTIER.md](bench/FRONTIER.md).

## 1. Mistura de especialistas: predicação, não permutação

**A dor.** Um MoE top-2 de 8 especialistas calculado denso custa 4× o
necessário. A solução habitual em GPU — ordenar os tokens por especialista
e despachar em blocos — introduz gather/scatter e formas que dependem dos
dados, e (com *capacity factor*) descarta tokens conforme o lote.

**Primeiros princípios.** Num GEMV *weight-stationary* em CPU, o custo é
**ler os pesos**. Não é preciso mover tokens: basta não ler as linhas de
peso que nenhum token escolheu. `Term.linear_masked(x, W, m)` é o GEMV com
uma máscara por linha: linhas com `m = 0` saem `+0` sem tocar em `W`, e as
outras executam **as mesmas instruções** do GEMV denso. Como a seleção
(`sel`) descarta as linhas não escolhidas de qualquer forma, `moe: :sparse`
e `moe: :dense` dão os mesmos bits — conferido em AVX2, AVX-512, RVV
(interpretador envenenado), Vulkan e QEMU.

| Mixtral reduzido, decode | denso | esparso |
|---|---|---|
| T = 1 | ≈ 13 ms | ≈ 5,1 ms (2,6×) |

Com mais tokens por passo, mais especialistas são escolhidos por alguém e a
vantagem cai (≈ 1,6× em T = 8 e 32) — a física do método, medida.

**Em 4 bits (0.8).** Os pesos `sb4` (4,75 bits/peso) rodavam densos: faltava
o GEMV quantizado com máscara. `Term.qgemv_masked/3` e o kernel KIR
`gemv_sb4_masked` (emitido para x86, AVX-512, NEON, RVV e SPIR-V — no Vulkan
a máscara zera o número de sub-blocos da linha) pulam as linhas não
escolhidas sem ler os seus nibbles nem as suas escalas; as escolhidas
executam as instruções do GEMV denso. Bits = denso em todo substrato; um
especialista não escolhido pode ter escalas NaN sem mudar um bit (teste).

| Mixtral reduzido em sb4 (largura 512, top-2 de 8) | denso | esparso | instruções retiradas (RVV, exatas) |
|---|---:|---:|---|
| T = 1 | 2,94 ms | 1,84 ms (1,6×) | 15,0 M → 5,3 M |
| T = 8 | 14,3 ms | 5,6 ms (2,6×) | 119 M → 41 M |

([bench/ROUND08.md](bench/ROUND08.md) §2.) O ganho total é menor que 4×
porque atenção, roteador e cabeça não mudam.

## 2. Atenção latente: dois programas, a escolha é do operador

O MLA do DeepSeek guarda por token um latente `c` (512) e uma chave
rotativa (64) em vez de K e V completos por cabeça (128 × 384): 85× menos
memória. Para atender sem expandir, a projeção de chave é **absorvida** na
query e a de valor na saída (`Term.linear_grouped`, bloco-diagonal por
cabeça), e a atenção vira MQA sobre o latente.

O escrutínio: isso **não é a mesma conta** em outra ordem — é outra conta
(outros bits, ~3× os FLOPs de atenção). Então são dois programas: a forma
latente (padrão dos modelos MLA) e a expandida (`mla: :expanded`), ambas
conferidas contra o `transformers`, cada uma com bits próprios e estáveis.

## 3. Janela deslizante e o cache circular

Uma linha na posição `p` com janela `w` atende `max(0, p − w + 1) … p` **na
ordem canônica**: só a faixa muda, então uma linha com janela é, bit a bit,
a atenção sem janela sobre as mesmas chaves movidas para o início (teste).
O kernel paginado começa a caminhar a tabela de blocos na primeira página da
janela.

**O cache circular.** Se a janela liga em *todas* as camadas, nenhuma
consulta lê uma posição mais velha que `w`. Um anel numa memória contígua
quebraria a ordem de leitura; um anel **na tabela de blocos** não: a página
lógica `j` aponta para `mine[j mod R]` e a ordem lógica fica intacta. A
posição `x` é sobrescrita pela posição `x + R·página`; um passo escreve
`p0 … p0+n−1` e depois lê até `p0 − w + 1`, logo nada legível é sobrescrito
se `R·página ≥ w + n − 1`:

    R = ⌈(w + step_tokens − 1) / página⌉

O teste confere os bits contra o cache inteiro **e** que `R − 1` páginas os
mudam (o limite é justo). Um modelo híbrido (camadas globais entre as
locais, Gemma 2/3) guarda tudo — o motor decide por `Config.ring_window/2`.

| contexto | janela | páginas por sequência (anel / inteiro) |
|---|---|---|
| 32 768 | 4 096 | 260 / 2 048 → 7,9× mais sequências |
| 131 072 | 4 096 | 260 / 8 192 → 31,5× |

## 4. Modelos de estado (Mamba): a recorrência é a definição

Um SSM troca o cache KV por um estado de tamanho fixo: o custo por token é
o mesmo no token 10 e no token 10⁶. O *scan* paralelo (associativo) que as
GPUs usam para o *prefill* reordena somas — os bits dependeriam do grau de
paralelismo. Aqui o programa é **um passo** (`t = 1`): *prefill* e decode
são as mesmas instruções e dão os mesmos bits.

O estado `s : f32[di, N]` e a janela da convolução causal são `state` do
programa; a sessão do worker os realimenta (`s ← s_next`) **dentro do
worker** depois de cada passo — o quadro `STEP` ganhou cópias de estado
para isso —, então um token custa um id para dentro e uma linha de logits
para fora. Dois operadores novos, ambos canônicos: `log` (≤ 1 ulp do
logaritmo corretamente arredondado) e `softplus` (o Δ do Mamba).

| d 256, 4 camadas | passo | memória por sequência |
|---|---|---|
| Mamba, qualquer contexto | ≈ 0,5 ms | 38 912 floats, fixo |
| atenção, contexto 8 192 | ≈ 1,9 ms | 4,2 M floats |

Conferido contra o `MambaForCausalLM` do `transformers` (3,4·10⁻⁷, gulosa
idêntica), com o controle de um Mamba sem memória (estado zerado a cada
token), que falha. `Vapor.Recurrent` gera; o `Vapor.Engine` recusa modelos
recorrentes por contrato (o seu modelo de memória é o cache paginado).

**Mamba-2 (0.8).** `Vapor.Lock.Adapters.Mamba2` (`Mamba2ForCausalLM`):
multi-cabeça com `A` escalar por cabeça, `B`/`C` compartilhados por grupos de
cabeças e passados pela mesma convolução causal que `x`, Δ por cabeça com
`time_step_limit`, e a RMSNorm com porta antes da projeção de saída. A
expansão cabeça → canais é exata: Δ chega aos canais de cada cabeça por um
produto com uma matriz *one-hot* (`Δ·1 + Σ 0`: um termo não nulo, nenhum
arredondamento), `A` e `D` expandidos na construção, e a linha do grupo de
`B`/`C` copiada por canal com `gather_row`. Conferido contra o
`transformers` (chunked scan, sem cache): 7,8·10⁻⁷ de erro relativo nos
logits, gulosa idêntica, worker nativo = oráculo, GPU = CPU bit a bit.

Duas divergências achadas no caminho, e o que se fez com elas:

- **A norma com porta.** O código de treino (`mamba_ssm`, `RMSNormGated` com
  `group_size = d_inner / n_groups`) normaliza **cada grupo**; o
  `MambaRMSNormGated` do `transformers` normaliza a largura inteira, qualquer
  que seja `n_groups`. Só concordam com um grupo. O padrão aqui é o do
  treino (`gated_norm: :group`); `gated_norm: :whole` reproduz o
  `transformers`. Os dois são conferidos contra a sua referência, e cada um
  **falha** contra a do outro (erro 0,74 e 0,66) — a comparação discrimina.
- **O limite de Δ.** O passo de decodificação com cache do `transformers`
  não aplica o `time_step_limit` que o seu *chunked scan* aplica; o passo
  aqui sempre aplica (a semântica treinada). A gulosa de referência é por
  isso gerada pelo passe completo, não pelo `generate()`.

## 5. Especulação em árvore sobre páginas compartilhadas

A especulação linear já era exata (a verificação de `k + 1` linhas dá os
bits de `k + 1` passos). A árvore verifica **vários** rascunhos num passo:

- o ramo 0 escreve direto nas páginas do contexto;
- o ramo `b ≥ 1` é um *slot* cuja tabela lista as páginas do contexto até a
  última página cheia (compartilhadas, ninguém escreve nelas) e depois
  páginas próprias; ele recomputa os tokens da página parcial (menos que
  uma página) antes dos seus;
- o ramo que concorda por mais tempo vence; se for um ramo bifurcado, as
  suas páginas **passam a ser** as do contexto — a posse muda, nada é copiado.

Como os logits de uma linha só dependem do seu token, da sua posição e do
KV que ela lê, a saída é a gulosa do alvo, token a token, para qualquer
rascunho (testado em cinco formas de árvore e com um rascunho sempre
errado). O rascunho padrão não é um modelo: é **busca no prompt** — o que
seguiu as ocorrências anteriores do último n-grama, copiado com sobreposição
(um laço em que o texto entrou é proposto até a profundidade toda). Ele é
forte exatamente onde a geração copia a entrada: respostas com recuperação,
edição de código, resumos que citam.

| alvo | sem rascunho | busca, linear | busca, árvore de 4 |
|---|---|---|---|
| bigrama plantado (a saída segue o contexto) | 1,0 token/passo | 6,0 | 6,0 |
| pesos aleatórios (a saída não segue) | 1,0 | 1,1 | 1,2 |

A segunda linha está aqui de propósito: num alvo cuja saída não copia o
contexto, a busca quase não acerta, e as linhas recomputadas custam mais do
que economizam. Medido, não escondido.

## 6. Paralelismo de tensor que não muda os bits

O produto canônico acumula 16 faixas em sequência e as soma numa árvore
fixa; cada elemento de saída é um desses produtos. Daí:

- **coluna-paralelo** (cada fragmento calcula linhas inteiras de `W`) é
  exato: cada elemento é calculado inteiro, num fragmento, pelas mesmas
  instruções;
- **linha-paralelo** (dividir `k` e somar parciais — a metade de Megatron com
  all-reduce) **não** é: a acumulação recomeça em cada fragmento e as
  parciais se encontram em outra ordem (medido: 81 % dos elementos mudam).

O MLP exato (`Vapor.Shard.mlp/4`) troca o all-reduce por um **all-gather**
da ativação intermediária e volta a ser coluna-paralelo: cada camada fica
exata; o preço é o volume de comunicação (`b·inter` em vez de `b·d`, ≈ 2,7–4×
nos MLPs SwiGLU). É a troca que um runtime distribuído reprodutível precisa
fazer — dita com o seu custo.

**Entre nós BEAM (0.8).** `Vapor.Shard.Cluster` leva a forma exata para nós
de um cluster Erlang (Distributed Erlang: autenticado pelo *cookie*, e por
TLS com `-proto_dist inet_tls`). Cada nó recebe as suas linhas de `W` uma vez,
com o SHA-256 — o `Vapor.Shard.Host` recalcula e **recusa** um fragmento que
não confere —, e as guarda **residentes** (compiladas com o fragmento como
constante); uma chamada leva só a ativação e traz as colunas. Duas
consequências que um runtime inexato não pode ter, ambas testadas com nós
`:peer` reais:

- **Failover sem deriva**: um nó perdido no meio da execução tem os seus
  fragmentos recolocados nos sobreviventes, a chamada é refeita, e a
  resposta tem **os mesmos bits** — nada a jusante percebe.
- **Réplica como verificação**: com `replicas: 2` cada fragmento é calculado
  em dois nós e os resultados comparados **bit a bit**; um nó que corrompe
  um bit é pego, não promediado. Ponto flutuante não reprodutível só
  poderia comparar com tolerância — e um adversário cuidadoso fica dentro
  dela.

## 7. Sessões residentes na GPU (0.8)

Até 0.7 cada execução no Vulkan (`vapor-fabric`) recriava *pipelines*,
*buffers* e *command buffers* e levava o cache KV de ida e volta: o motor não
servia na GPU. Agora o fabric tem as sessões que o worker de CPU já tinha:

- `OPEN` cria os *pipelines* uma vez, aloca os *buffers* no tamanho máximo
  do programa — **memória direta** (`DEVICE_LOCAL | HOST_VISIBLE`, o caso de
  GPUs integradas e do lavapipe) ou ***staging*** (um *buffer* de cópia, o
  caminho de GPU discreta; forçável com `staging: true`, e escolhido sozinho
  se faltar memória direta) — e sobe as constantes;
- `STEP` escreve só as entradas, despacha, copia o estado (`s ← s_next`)
  **dentro da GPU** e lê só as saídas pedidas. Os *command buffers* gravados
  ficam num cache cuja chave são **os bytes exatos** do passo (escritas,
  despachos, cópias, constantes de *push*): um passo igual a um anterior é
  só reenviado — 31 de 32 passos de decode reaproveitam a gravação;
- `CLOSE` libera; um fabric que morre invalida as sessões (geração) e o
  motor reabre — nunca a BEAM.

| Llama reduzido (largura 256, 4 camadas), um token | ms/token | bytes host↔GPU por token |
|---|---:|---:|
| GPU, um `RUN` por token (0.7) | 72,4 | 1 056 776 |
| GPU, sessão, memória direta | 12,4 | 8 200 |
| GPU, sessão, *staging* | 10,5 | 8 200 |
| CPU, sessão no worker | 1,3 | — |

Os mesmos bits nos quatro caminhos. O `Vapor.Engine` serve inteiro na GPU
(`mix vapor.serve --gpu`), com os tokens da CPU. Aqui a "GPU" é o lavapipe —
a própria CPU emulando Vulkan —, então a vazão (83 contra 1 384 tokens/s
na CPU) não diz nada sobre GPU real; o que se prova é o protocolo, a
residência e a igualdade dos bits.

## 8. O que não está feito

- Atenção com memória de *workgroup* no SPIR-V e o fim do `@max_dh = 512`;
  medir numa GPU real (só lavapipe aqui).
- FlashAttention: o softmax online em blocos é outra ordem canônica; só como
  política `:fast` declarada.
- *Prefill* do Mamba num único quadro `RUN` com iterações (hoje: um `STEP`
  por token); Jamba/Zamba/Bamba (híbridos: cache KV **e** estado por
  sequência no motor) e Falcon-Mamba.
- Atenção fragmentada por cabeças entre nós (o MLP e as projeções exatas
  estão feitos).
- Especulação em árvore sem recomputar a página parcial (cópia-na-escrita da
  página), e servida pelo `Vapor.Engine` em vez de uma sessão dedicada.
