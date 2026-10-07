# Medições da rodada 0.8

Regeneráveis com `mix vapor.bench --round08`. Máquina: Intel(R) Xeon(R) Processor @ 2.80GHz, 2 vCPUs, OTP 25.
A GPU é o **lavapipe** (Vulkan da Mesa nos mesmos núcleos da CPU): os
tempos de GPU medem a maquinaria — o que a sessão residente tira de
cada passo —, não o que uma GPU discreta entregaria. Pesos aleatórios:
medidas da máquina, não da qualidade (essa é `mix vapor.quality`).

## 1. Sessões residentes na GPU (P0 da 0.7)

Llama, largura 256, 4 camadas, 8 cabeças (2 KV), vocabulário 2048, contexto 256; um *prompt* de 32 tokens, depois 32 passos de um token.
Tempo de parede por passo, medido na BEAM (inclui o protocolo), mediana:

| caminho | ms/token | bytes host↔dispositivo por token | gravações reaproveitadas |
|---|---:|---:|---:|
| GPU, um `RUN` por token (sem sessão: pipelines e cache KV a cada passo) | 72.36 | 1056776 | — |
| GPU, sessão residente, memória direta | 12.36 | 8200 | 31/32 |
| GPU, sessão residente, *staging* (caminho de GPU discreta) | 10.54 | 8200 | 31/32 |
| CPU, sessão no worker nativo | 1.29 | — | — |

Mesmos bits nos quatro caminhos (logits de cada passo): **sim**.
A sessão corta 5.86× o tempo por token e
129× o tráfego: o passo move os ids e uma linha de logits,
não o cache.

**Motor** (4 pedidos simultâneos, 24 tokens de *prompt*, 32 gerados, guloso):

| substrato | tokens | tokens/s |
|---|---:|---:|
| CPU (worker nativo) | 128 | 1384.04 |
| GPU (sessão residente) | 128 | 83.38 |

Mesmos tokens nos dois: **sim**. No lavapipe a GPU
é a própria CPU, então a comparação de vazão diz pouco sobre GPU real; o que
ela prova é que o motor serve inteiro no Vulkan, com os bits da CPU.

## 2. Especialistas esparsos em 4 bits (`qgemv_masked`)

Mixtral, largura 512, 8 especialistas (top-2), 2 camadas, pesos sb4 (4,75 bits/peso); ISA `x86_64_avx512`. Tempo no worker (mediana de 5) e instruções
retiradas contadas exatamente pelo interpretador RVV (VLEN 256):

| tokens | denso ms | esparso ms | × | instruções denso | instruções esparso | mesmos bits |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 2.94 | 1.84 | 1.60 | 15033088 | 5273888 | sim |
| 8 | 14.26 | 5.59 | 2.55 | 119430325 | 41309909 | sim |

Com top-2 de 8, cada token lê 1/4 dos especialistas; o resto do modelo
(atenção, roteador, cabeça) não muda — o ganho total é menor que 4×.
