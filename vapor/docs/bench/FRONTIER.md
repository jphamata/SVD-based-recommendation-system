# Medições da rodada 0.6 — vapor

Gerado por `mix vapor.bench --frontier` em 2026-10-03, nesta máquina
(2 núcleos). Pesos aleatórios: medem a maquinaria, não a qualidade.
Tempos são medianas de 7 execuções dentro do worker; dependem da carga do host.

## Experts esparsos (Mixtral reduzido: d 512, 8 experts, top-2, 2 camadas)

| tokens no passo | denso ms | esparso ms | aceleração | mesmos bits |
|---|---|---|---|---|
| 1 | 13.16 | 5.09 | 2.59× | true |
| 8 | 25.03 | 15.44 | 1.62× | true |
| 32 | 51.45 | 32.82 | 1.57× | true |

## Atenção latente (MLA, 16 cabeças, decode com o cache cheio de 256 posições)

| forma | floats/token/camada | cache (MB, 256 pos.) | passo ms |
|---|---|---|---|
| expanded | 1536 | 3.146 | 1.73 |
| latent | 144 | 0.295 | 1.58 |

Nas formas do DeepSeek-V3 (128 cabeças, 512 + 64 latentes): 576 contra 49 152 floats — 85×.

## Janela deslizante (8 cabeças, dh 64, cache de 4 096 posições, uma consulta na última)

| janela | ms |
|---|---|
| nenhuma | 1.099 |
| 1024 | 0.293 |
| 256 | 0.121 |

## Divisão corretamente arredondada (1048576 quocientes)

| microprograma | primitivas | ms |
|---|---|---|
| `a·rcp(b)` (até 0.5, erra o arredondamento em ~20 %) | 23 | 2.48 |
| correto (IEEE, DAZ/FTZ) | 123 | 5.42 |

## Espacial (convolução sem kernel de convolução)

| bloco | ms | GFLOP/s |
|---|---|---|
| GroupNorm + SiLU + conv 3×3 (16×16×64) + upsample ×2 + conv 3×3 (32×32×64) | 8.55 | 11.04 |

## Log de transparência (4096 entradas)

| operação | por segundo |
|---|---|
| acréscimo (com a árvore densa) | 227948 |
| prova de inclusão (12 hashes) | 180663 |
| verificação de inclusão | 85906 |

Prova de consistência 1 000 → 4096 verificada em 0.027 ms; todas as verificações: true.

## Paralelismo de tensor (GEMV 2048×1024, 8 linhas)

| | ms (de ponta a ponta, com compilação) |
|---|---|
| 1 worker | 44.12 |
| 2 workers, colunas | 35.85 |

Bits iguais entre 1 e 2 workers: true. A forma por linhas (k dividido, soma das parciais)
mudaria 13284 de 16384 elementos — por isso o MLP exato usa all-gather.

## Decodificação por espaço de estados (Mamba: d 256, interno 512, estado 16, 4 camadas)

| | passo ms | memória por sequência (floats) |
|---|---|---|
| Mamba, contexto de ~10 tokens | 0.587 | 38912 (estado fixo) |
| Mamba, após 1 000 tokens | 0.463 | 38912 |
| atenção (Llama, mesmas larguras), contexto 64 | 0.968 | 32768 (cache KV) |
| atenção (Llama, mesmas larguras), contexto 1024 | 1.126 | 524288 (cache KV) |
| atenção (Llama, mesmas larguras), contexto 8192 | 1.924 | 4194304 (cache KV) |

O passo de um SSM não depende do contexto; o da atenção cresce com ele (e o cache, linearmente).

## Cache KV circular (janela em todas as camadas)

Motor real (Mistral reduzido, janela 32, contexto 512, páginas de 16, 32 tokens por passo):
4 páginas por sequência no anel contra 32 sem ele.

| contexto | janela | páginas no anel (página 16, 64 tokens/passo) | sem anel | sequências a mais na mesma memória |
|---|---|---|---|---|
| 32768 | 4096 | 260 | 2048 | 7.9× |
| 131072 | 4096 | 260 | 8192 | 31.5× |
| 8192 | 2047 | 132 | 512 | 3.9× |

## Whisper (larguras do whisper-tiny: d 384, 4 + 4 camadas, 6 cabeças; vocabulário reduzido a 4 096)

| | ms |
|---|---|
| compilar o encoder (uma vez) | 1279 |
| encoder sobre 30 s de áudio (3 000 quadros mel → 1 500 posições) | 1637.6 |
| um passo do decoder (atenção cruzada sobre 1 500 posições) | 8.308 |

## Especulação em árvore (96 tokens gulosos; páginas de 8)

**bigrama plantado**

| rascunho | passos do alvo | tokens/passo | linhas computadas | ms | saída = gulosa |
|---|---|---|---|---|---|
| sem rascunho | 96 | 1.00 | 163 | 69 | true |
| busca no prompt, linear | 16 | 6.00 | 168 | 16 | true |
| busca no prompt, árvore de 4 | 16 | 6.00 | 543 | 19 | true |
| rascunho sempre errado, árvore de 4 | 96 | 1.00 | 3634 | 117 | true |

**pesos aleatórios**

| rascunho | passos do alvo | tokens/passo | linhas computadas | ms | saída = gulosa |
|---|---|---|---|---|---|
| sem rascunho | 96 | 1.00 | 163 | 119 | true |
| busca no prompt, linear | 86 | 1.12 | 447 | 108 | true |
| busca no prompt, árvore de 4 | 81 | 1.19 | 831 | 128 | true |
| rascunho sempre errado, árvore de 4 | 93 | 1.03 | 3523 | 247 | true |

A saída é a decodificação gulosa do alvo em todas as linhas; o rascunho muda só o número de passos.
Com pesos aleatórios a saída não copia o contexto e a busca acerta pouco — medido, não escondido.
