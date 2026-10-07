# Medições — vapor

Gerado por `mix vapor.bench` em 2026-10-01. Tudo abaixo foi medido nesta
máquina, na hora; tempos de parede dependem da carga do host.

## Máquina

| | |
|---|---|
| CPU | Intel(R) Xeon(R) Processor @ 2.10GHz |
| núcleos (BEAM schedulers) | 2 |
| extensões relevantes | avx2, fma, avx512f, avx512bw, avx512vl, avx512_bf16, avx512_fp16, amx_tile |
| contadores de eventos disponíveis | context_switches, page_faults, task_clock_ns |
| OTP / Elixir | 25 / 1.14.0 |

Contadores de hardware (ciclos, instruções, cache) só aparecem quando o
kernel expõe uma PMU; aqui não há PMU (VM) — o tempo de CPU vem do contador de software task-clock.

## Kernels

Tempo dentro do worker (mínimo de 15 passos de uma sessão residente,
entradas já residentes),
trabalho contado pelo árbitro, previsão do perfil nativo declarado.
Resultados bit-idênticos para qualquer número de threads e entre AVX2 e
AVX-512 (testado); a aceleração é relativa à primeira linha (AVX2, 1 thread).

| kernel | ISA | threads | ms | CPU ms | GFLOP/s | GB/s | aceleração | previsto ms |
|---|---|---|---|---|---|---|---|---|
| stream y = x + 0.5 (16 M) | AVX2 | 1 | 9.85 | 9.86 | 1.70 | 13.6 | 1.00× | 6.74 |
| stream y = x + 0.5 (16 M) | AVX2 | 2 | 3.17 | 6.45 | 5.29 | 42.3 | 3.11× | 6.74 |
| stream y = x + 0.5 (16 M) | AVX-512 | 1 | 11.3 | 11.3 | 1.49 | 11.9 | 0.875× | 6.74 |
| stream y = x + 0.5 (16 M) | AVX-512 | 2 | 2.52 | 5.17 | 6.66 | 53.3 | 3.91× | 6.74 |
| GEMV f32 2048² | AVX2 | 1 | 0.661 | 0.664 | 12.7 | 25.4 | 1.00× | 0.87 |
| GEMV f32 2048² | AVX2 | 2 | 0.296 | 0.733 | 28.4 | 56.8 | 2.24× | 0.87 |
| GEMV f32 2048² | AVX-512 | 1 | 0.582 | 0.585 | 14.4 | 28.9 | 1.14× | 0.87 |
| GEMV f32 2048² | AVX-512 | 2 | 0.282 | 0.688 | 29.8 | 59.7 | 2.35× | 0.87 |
| GEMV bf16 2048² | AVX2 | 1 | 0.428 | 0.431 | 19.6 | 19.6 | 1.00× | 0.45 |
| GEMV bf16 2048² | AVX2 | 2 | 0.201 | 0.579 | 41.8 | 41.9 | 2.13× | 0.45 |
| GEMV bf16 2048² | AVX-512 | 1 | 0.3 | 0.303 | 28.0 | 28.0 | 1.43× | 0.45 |
| GEMV bf16 2048² | AVX-512 | 2 | 0.174 | 0.458 | 48.3 | 48.4 | 2.46× | 0.45 |
| linear f32 512² × 64 rows | AVX2 | 1 | 1.12 | 1.12 | 30.0 | 1.17 | 1.00× | 1.02 |
| linear f32 512² × 64 rows | AVX2 | 2 | 0.559 | 1.25 | 60.1 | 2.35 | 2.00× | 1.02 |
| linear f32 512² × 64 rows | AVX-512 | 1 | 0.807 | 0.811 | 41.6 | 1.62 | 1.39× | 1.02 |
| linear f32 512² × 64 rows | AVX-512 | 2 | 0.398 | 0.924 | 84.3 | 3.29 | 2.82× | 1.02 |
| GEMV sb4 1024×4096 | AVX2 | 1 | 0.567 | 0.569 | 23.6 | 4.40 | 1.00× | 0.591 |
| GEMV sb4 1024×4096 | AVX2 | 2 | 0.287 | 0.74 | 46.6 | 8.69 | 1.97× | 0.591 |
| GEMV sb4 1024×4096 | AVX-512 | 1 | 0.537 | 0.539 | 24.9 | 4.64 | 1.06× | 0.591 |
| GEMV sb4 1024×4096 | AVX-512 | 2 | 0.272 | 0.634 | 49.1 | 9.16 | 2.08× | 0.591 |
| x·silu(x) fused (4 M) | AVX2 | 1 | 9.27 | 9.27 | 19.5 | 3.62 | 1.00× | 4.34 |
| x·silu(x) fused (4 M) | AVX2 | 2 | 4.65 | 9.39 | 38.8 | 7.22 | 1.99× | 4.34 |
| x·silu(x) fused (4 M) | AVX-512 | 1 | 4.10 | 4.10 | 44.0 | 8.19 | 2.26× | 4.34 |
| x·silu(x) fused (4 M) | AVX-512 | 2 | 2.12 | 4.34 | 85.0 | 15.8 | 4.37× | 4.34 |
| attention S=1024 (8 h, 4 kv, dh 64) | AVX2 | 1 | 0.166 | 0.167 | 14.6 | 12.7 | 1.00× | 0.146 |
| attention S=1024 (8 h, 4 kv, dh 64) | AVX2 | 2 | 0.082 | 0.255 | 29.7 | 25.8 | 2.03× | 0.146 |
| attention S=1024 (8 h, 4 kv, dh 64) | AVX-512 | 1 | 0.136 | 0.137 | 17.9 | 15.5 | 1.22× | 0.146 |
| attention S=1024 (8 h, 4 kv, dh 64) | AVX-512 | 2 | 0.074 | 0.244 | 32.7 | 28.4 | 2.23× | 0.146 |
| attention paged S=1024, page 16 | AVX2 | 1 | 0.159 | 0.161 | 15.2 | 13.2 | 1.00× | 0.251 |
| attention paged S=1024, page 16 | AVX2 | 2 | 0.088 | 0.27 | 27.5 | 23.9 | 1.81× | 0.251 |
| attention paged S=1024, page 16 | AVX-512 | 1 | 0.134 | 0.136 | 18.1 | 15.7 | 1.19× | 0.251 |
| attention paged S=1024, page 16 | AVX-512 | 2 | 0.079 | 0.251 | 30.8 | 26.7 | 2.02× | 0.251 |
| sample V=32000 (T=1) | AVX2 | 1 | 0.069 | 0.07 | 13.9 | 3.71 | 1.00× | 0.043 |
| sample V=32000 (T=1) | AVX2 | 2 | 0.069 | 0.071 | 13.9 | 3.71 | 1.00× | 0.043 |
| sample V=32000 (T=1) | AVX-512 | 1 | 0.053 | 0.054 | 18.2 | 4.86 | 1.31× | 0.043 |
| sample V=32000 (T=1) | AVX-512 | 2 | 0.054 | 0.056 | 17.8 | 4.75 | 1.28× | 0.043 |

![roofline](roofline.svg)

Teto de memória medido: **53.3 GB/s** (kernel de streaming, todos os
núcleos); teto de cômputo teórico do código AVX-512 emitido: 64 FLOP/ciclo ×
2.10 GHz × 2 núcleos = 269 GFLOP/s (duas unidades FMA; a política
canônica arredonda produto e soma separadamente, então seu teto é metade).

## Motor (lote contínuo, KV paginado)

Modelo Llama de pesos aleatórios: vocabulário 32000, largura 256,
4 camadas, 8 cabeças (4 KV), f32; prompts de 16 tokens,
48 tokens gerados por requisição, amostragem no substrato (T = 0,8).

| ISA | threads | requisições simultâneas | tokens | s | tokens/s |
|---|---|---|---|---|---|
| AVX2 | 1 | 1 | 48 | 0.143 | 337 |
| AVX2 | 1 | 2 | 96 | 0.166 | 579 |
| AVX2 | 1 | 4 | 192 | 0.231 | 830 |
| AVX2 | 1 | 8 | 384 | 0.429 | 895 |
| AVX2 | 2 | 1 | 48 | 0.111 | 433 |
| AVX2 | 2 | 2 | 96 | 0.111 | 867 |
| AVX2 | 2 | 4 | 192 | 0.187 | 1028 |
| AVX2 | 2 | 8 | 384 | 0.26 | 1477 |
| AVX-512 | 1 | 1 | 48 | 0.123 | 390 |
| AVX-512 | 1 | 2 | 96 | 0.167 | 574 |
| AVX-512 | 1 | 4 | 192 | 0.245 | 783 |
| AVX-512 | 1 | 8 | 384 | 0.337 | 1139 |
| AVX-512 | 2 | 1 | 48 | 0.109 | 440 |
| AVX-512 | 2 | 2 | 96 | 0.121 | 794 |
| AVX-512 | 2 | 4 | 192 | 0.142 | 1349 |
| AVX-512 | 2 | 8 | 384 | 0.261 | 1474 |

![engine](engine.svg)

Armazenamento dos pesos em bfloat16 (`storage: :bf16`: metade dos bytes
lidos por passo; mesmos bits que o programa f32 sobre os pesos arredondados — testado):

| pesos | ISA | threads | requisições simultâneas | tokens | s | tokens/s |
|---|---|---|---|---|---|---|
| f32 | AVX-512 | 2 | 1 | 48 | 0.119 | 403 |
| f32 | AVX-512 | 2 | 8 | 384 | 0.212 | 1812 |
| bf16 | AVX-512 | 2 | 1 | 48 | 0.082 | 588 |
| bf16 | AVX-512 | 2 | 8 | 384 | 0.239 | 1606 |

Paralelismo de dados em vez de threads intra-operação (`replicas:`, um
motor por núcleo com uma thread cada, uma compilação e as mesmas páginas
de pesos; mesmos tokens por requisição — testado):

| réplicas × threads | requisições simultâneas | tokens | s | tokens/s |
|---|---|---|---|---|
| 2 × 1 | 2 | 96 | 0.125 | 765 |
| 2 × 1 | 4 | 192 | 0.182 | 1054 |
| 2 × 1 | 8 | 384 | 0.247 | 1555 |

Prefill (um prompt de 256 tokens, primeiro token):

| ISA | threads | tokens | ms | tokens/s |
|---|---|---|---|---|
| AVX2 | 1 | 256 | 67.7 | 3782 |
| AVX2 | 2 | 256 | 38.3 | 6691 |
| AVX-512 | 1 | 256 | 56.7 | 4512 |
| AVX-512 | 2 | 256 | 46.8 | 5475 |

## Exatidão das funções canônicas (ULP contra binary64)

| função | amostras | 0 ULP | 1 ULP | 2 ULP | ≥ 3 ULP |
|---|---|---|---|---|---|
| e^x, x ∈ [−87, 88] | 20000 | 66.11 % | 33.87 % | 0.03 % | 0.0 % |
| 1/x, |x| ∈ [2⁻¹⁰⁰, 2¹⁰⁰] | 20000 | 85.22 % | 14.78 % | 0.0 % | 0.0 % |
| 1/√x, x ∈ [2⁻¹²⁰, 2¹²⁰] | 20000 | 84.56 % | 15.44 % | 0.0 % | 0.0 % |
| σ(x), x ∈ [−40, 40] | 20000 | 66.17 % | 31.6 % | 2.15 % | 0.08 % |
| x·σ(x), x ∈ [−40, 40] | 20000 | 67.8 % | 29.89 % | 2.25 % | 0.05 % |

## Tokenizador

Vocabulário do Llama 3 (128 256 tokens), README × 8: 65688 bytes → 20753 tokens, **535 mil tokens/s** num núcleo da BEAM.
