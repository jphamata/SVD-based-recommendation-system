# Qualidade das saídas — vapor

Gerado por `mix vapor.quality` (substrato: **native**, 1997099 ms no total).
**147 de 147 verificações passaram.** Cada verificação compara uma saída
com a verdade conhecida **e** com um controle que um gerador de ruído alcançaria — passar
significa "sinal", e uma falha diz a distância. Metodologia: [docs/QUALIDADE.md](../QUALIDADE.md).

## 1. Os portões de ruído, calibrados antes de julgar

Cada portão é ajustado a controles negativos (ruído, sinal embaralhado, repetição) e
positivos (sinal real retido) e **recusa existir** se eles se sobrepõem. A margem é a
folga entre o pior negativo e o pior positivo.

| portão | escore | pior negativo | pior positivo | margem | controles |
|---|---|---|---|---|---|
| audio_noise | ↓ mais estrutura | 0.534 | 0.000 | 0.534 | chords: 0.000…0.000; noise: 0.546…0.605; shuffled: 0.534…0.583; tones: 0.000…0.000 |
| image_noise | ↑ mais estrutura | 0.081 | 0.844 | 0.763 | noise: -0.092…0.062; scenes: 0.844…0.982; shuffled: -0.048…0.081 |
| text_collapse | ↑ mais estrutura | 0.095 | 0.665 | 0.570 | holdout: 0.665…0.770; loop: 0.090…0.095 |
| text_noise | ↑ mais estrutura | 0.636 | 0.864 | 0.227 | holdout: 0.864…1.000; uniform: 0.000…0.000; unigram: 0.414…0.636 |

## 2. Texto: um modelo plantado pela pilha inteira

Um bigrama contado na prosa em português de `docs/` embutido **exatamente** nos pesos de um
decoder Llama (`Vapor.Quality.Planted`), admitido pela eclusa, compilado e executado.

| medida | valor |
|---|---|
| max \|logit − log P\| contra a tabela analítica | 1.2e-6 |
| bits/caractere do modelo (retido) | 3.690 |
| bits/caractere da tabela | 3.690 |
| linha de base unigrama | 4.303 |
| linha de base uniforme | 5.555 |
| veredito do portão, gerações do modelo plantado | [pass: :natural, pass: :structured, pass: :structured, pass: :natural] |
| veredito do portão, mesmo modelo com pesos aleatórios | [fail: :noise, fail: :noise, fail: :noise, fail: :noise] |

Amostra (temperatura 1): `que s__  bf32orilo. k-loas___ __s⏎⏎aingr_  rrino va m iso ___ coriorva.robscy__de er, brueciderass:u rvt___copo ___m e⏎ ____-adotosundadeirde.⏎ catoda m ba _o _`

É texto com estatística de português, não português: o portão diz `:structured` (mais
estrutura que todo controle de ruído, menos que todo texto real), e é isso que deve dizer.

## 3. Any-to-any: todas as rotas do hub, em entradas retidas

Pares de cores nunca vistos em nenhum ajuste, variantes novas (jitter, iluminação, ruído de
sensor), fases e amplitudes novas, ruído a 10 dB.

| rota | o que mede | verificação | valor | controle | limiar | ok |
|---|---|---|---|---|---|---|
| image → text | caption accuracy (unseen pairs, variants, σ=0.05 noise) | image→text accuracy | 1.000 | 0.250 | ≥ 0.95 (chance 0.25 per slot) | ✅ |
| text → image | PSNR vs the canonical scene; control: the swapped caption | text→image PSNR (dB) | 23.690 | 12.762 | ≥ 18 and ≥ control + 3 | ✅ |
| text → image | PSNR vs the canonical scene; control: the swapped caption | text→image outputs pass the image gate | 4× natural | — | signal | ✅ |
| image → image | VQ round trip PSNR on unseen variants; control: random codebook | image→image PSNR (dB) | 23.038 | 11.286 | ≥ 20 and ≥ control + 6 | ✅ |
| image → image | VQ round trip PSNR on unseen variants; control: random codebook | image→image outputs pass the image gate | 8× natural | — | signal | ✅ |
| audio → text | note accuracy (unseen phase/amplitude, 10 dB SNR) | audio→text accuracy | 1.000 | 0.250 | ≥ 0.95 (chance 0.25) | ✅ |
| text → audio | pitch error, audio gate, SNR vs the reference tone | text→audio max pitch error | 0.015 | — | ≤ 3 % | ✅ |
| text → audio | pitch error, audio gate, SNR vs the reference tone | text→audio outputs pass the audio gate | 3× natural, 1× structured | — | signal | ✅ |
| text → audio | pitch error, audio gate, SNR vs the reference tone | text→audio SNR vs reference (dB) | 140.790 | 0.000 | ≥ 20 | ✅ |
| audio → audio | VQ round trip SNR on unseen phases | audio→audio SNR (dB) | 17.278 | 0.000 | ≥ 10 | ✅ |
| audio → audio | VQ round trip SNR on unseen phases | audio→audio outputs pass the audio gate | 8× structured | — | signal | ✅ |
| text → text | translation exact match (colour ↔ note, counted bigram decoders) | colour→note exact match | 1.000 | 0.250 | = 1 | ✅ |
| text → text | translation exact match (colour ↔ note, counted bigram decoders) | note→colour exact match | 1.000 | 0.250 | = 1 | ✅ |
| image → audio | via the pivot: the note of the left colour, pitch within 3 % | image→audio correct pitch | 1.000 | 0.250 | = 1 | ✅ |
| audio → image | via the pivot: PSNR vs the scene of the note's colour | audio→image PSNR (dB) | 23.774 | — | ≥ 18 | ✅ |
| audio → image | via the pivot: PSNR vs the scene of the note's colour | audio→image outputs pass the image gate | 4× natural | — | signal | ✅ |
| image → soft token → decoder | projector output injected as a row of the decoder (inject: true) | image→soft-token→note accuracy | 1.000 | 0.250 | = 1 | ✅ |

## 4. Fusão de modelos

Um especialista em português (A) e um em inglês (B), bigramas plantados; bits/caractere em
texto retido de cada domínio (menor é melhor).
Especialistas: A@pt 3.910, A@en 4.038, B@pt 4.432, B@en 3.780.

| método | pt | en | média | ms | recibo |
|---|---|---|---|---|---|
| linear | 4.047 | 3.793 | 3.920 | 8 | ✅ |
| slerp | 4.050 | 3.794 | 3.922 | 4 | ✅ |
| task_arithmetic | 4.047 | 3.793 | 3.920 | 3 | ✅ |
| ties | 4.543 | 4.439 | 4.491 | 68 | ✅ |
| dare_linear | 4.325 | 4.218 | 4.272 | 9 | ✅ |

- ✅ merge: linear beats the wrong specialist on each domain
- ✅ merge: linear's mean bits beat both specialists'
- ✅ merge: merge(A, A) = A bit for bit
- ✅ merge: slerp(t = 0) = A bit for bit
- ✅ merge: every receipt verifies

TIES e DARE pressupõem vetores de tarefa esparsos (diferenças pequenas de um fine-tune);
as diferenças entre tabelas de log-probabilidade de dois domínios são densas, e a poda as
distorce — medido aqui, não presumido.

### 4b. Fusão de transformers treinados (não plantados)

Decoders Llama de caracteres treinados pelo PyTorch (`priv/quality/merge`,
`test/python/train_merge_models.py`). Bits/caractere, teste disjunto da validação.

| modelo | pt | en | média |
|---|---|---|---|
| base | 3.316 | 3.282 | 3.299 |
| ft_en | 3.499 | 3.134 | 3.316 |
| ft_pt | 3.236 | 3.399 | 3.317 |
| solo_en | 5.052 | 3.049 | 4.050 |
| solo_pt | 3.182 | 4.491 | 3.837 |

**Fine-tunes de uma base (ft_pt + ft_en)** — regime diagnosticado: `small_deltas`; escolhido na validação: **linear**.

> task vectors at 3 % of the weights' norm; the top 20 % of entries carry 62 % of their energy (TIES at density 0.2 discards 38 %)  
> signs disagree on 39 % of the deltas' shared magnitude; delta cosine 0.186  
> DARE at p = 0.2 perturbs each delta by 200 % of its own norm — safe only if the fine-tune is redundant  
> admissible: every method; :task_arithmetic at λ = 1 applies each delta at full strength — also where it hurts the other models' domains (λ = 1/n is :linear) — measure (select/4)  

| método | validação | pt (teste) | en (teste) | média (teste) | ms |
|---|---|---|---|---|---|
| linear | 3.303 | 3.289 | 3.199 | 3.244 | 6 |
| slerp | 3.304 | 3.289 | 3.200 | 3.244 | 30 |
| regmean | 3.306 | 3.295 | 3.190 | 3.243 | 428 |
| task_arithmetic | 3.454 | 3.475 | 3.269 | 3.372 | 9 |
| task_arithmetic λ=0.7 | 3.334 | 3.336 | 3.205 | 3.271 | 9 |
| ties 0.2 | 3.306 | 3.288 | 3.218 | 3.253 | 116 |
| ties 0.5 | 3.325 | 3.323 | 3.209 | 3.266 | 118 |
| dare_linear 0.5 | 3.479 | 3.487 | 3.296 | 3.391 | 57 |
| dare_ties 0.5 | 3.380 | 3.388 | 3.232 | 3.310 | 71 |


**Treinados separadamente (solo_pt + solo_en, sem base comum)** — regime diagnosticado: `unrelated`; escolhido na validação: **linear**.

> the weights are nearly orthogonal (cosine 0.343): these networks do not share an ancestor, so their units are not aligned (a network is the same function under any permutation of its hidden units)  
> a weight average of misaligned networks destroys both; do not fuse them in weight space (align permutations first — not implemented here)  

| método | validação | pt (teste) | en (teste) | média (teste) | ms |
|---|---|---|---|---|---|
| linear | 5.088 | 5.081 | 5.074 | 5.078 | 6 |
| slerp | 6.218 | 6.185 | 6.086 | 6.136 | 37 |
| regmean | 5.970 | 5.766 | 6.032 | 5.899 | 402 |


- ✅ merge (trained): linear fusion of the fine-tunes beats both specialists and the base (mean bits/char, test)
- ✅ merge (trained): diagnose calls fine-tunes of one base related, not :unrelated
- ✅ merge (trained): diagnose calls independently trained models :unrelated
- ✅ merge (trained): …and it was right — their linear average is worse than either specialist
- ✅ merge (trained): selection on validation lands within 0.02 bits of the best test score


## 4c. Dados reais — leitura, fala, caligrafia

Modelos admitidos pela eclusa (`priv/ocr`, `priv/speech`, `priv/digits`), dados nunca vistos no treino.

| rota | medida | vapor | controle | Tesseract |
|---|---|---|---|---|
| OCR, 5 fontes fora do treino (40 linhas) | CER | 0.044 | 1.061 (texto fluente errado) | 0.052 |
| OCR, foto real de página | CER | 0.095 | — | 0.364 |
| fala, voz nunca ouvida (100 gravações) | acerto | 0.900 | 0.100 (acaso); invertida no tempo: 0.840 | — |
| caligrafia → dígito (497 retidos) | acerto | 0.980 | 0.100 | — |
| dígito → caligrafia (50 gerados), lidos de volta | acerto | 1.000 | 0.100 | — |
| distância ao treino mais próximo (mediana) | níveis de cinza | 18.647 | reais retidos 16.733; memorização 0.000 | — |
| voz → texto → desenho → leitura | acerto | 0.900 | 0.100 | — |

CER por fonte: C059-Roman.otf 0.015, Carlito-Regular.ttf 0.012, P052-Roman.otf 0.019, URWBookman-Light.otf 0.024, URWGothic-Book.otf 0.125.

Fala: a voz retida (escolhida antes do treino) é a mais fácil das seis do conjunto; retendo cada voz por vez (`test/python/train_speech.py --loso`) o acerto médio é 0,72 ([ANY_TO_ANY.md §6](../ANY_TO_ANY.md)). Inverter a gravação no tempo quase não muda a leitura: um dígito falado se reconhece pelo timbre das vogais — por isso o controle é o acaso.

Página real lida pelo vapor:

```
Region-based segmentation
let us first determine markers of the coins and the
background. These markens are piels that we can label
remotiguousiy as either object or background. Hert
FP nearkers are found at the two entreme parts of the
ferem of gre values:
```

- ✅ OCR: CER on typefaces never seen in training
- ✅ OCR: CER on a real photographed page (uneven light)
- ✅ speech: spoken-digit accuracy on a voice never heard
- ✅ handwriting: image → digit accuracy on 497 held-out real digits
- ✅ handwriting: digit → image, generated digits read back by the real-data classifier
- ✅ handwriting: generated digits are not copies (median distance to the nearest training image)
- ✅ chain: held-out voice → text → drawn digit → read back = what was said
- ✅ JPEG: baseline and progressive pixels = libjpeg-turbo (Pillow), SHA-256


## 5. Substratos

- ✅ vq encode: nativo = oráculo bit a bit
- ✅ vq decode: nativo = oráculo bit a bit
- ✅ linear bridge: nativo = oráculo bit a bit
- ✅ spectrum (Hann STFT): nativo = oráculo bit a bit
- ✅ additive synthesis: nativo = oráculo bit a bit

## 5b. Rodada 0.6 — especialistas, cache latente, janelas, ÷, convolução, registro, estado, árvore

Cada recurso da rodada 0.6 contra a sua verdade **e** contra um controle que mostra que a
verificação discrimina (a forma ingênua, a forma errada, a entrada adulterada).

| verificação | valor | controle | limiar | ok |
|---|---|---|---|---|
| experts: elements differing, sparse dispatch vs dense | 0 | 480 | 0, control > 0 | ✅ |
| latent MLA: argmax agreement with the expanded form | 1.000 | 0.000 | 1.0, control < 1.0 | ✅ |
| sliding window: bits differing from attention over the window moved to the front | 0 | 64 | 0, control > 0 | ✅ |
| ÷: IEEE mismatches over 14775 random normal quotients | 0 | 0.312 | 0, old a·rcp(b) > 0.1 | ✅ |
| convolution: relative error against a direct binary64 convolution | 6.4e-8 | 1.669 | ≤ 1e-6, control ≥ 0.1 | ✅ |
| transparency log: probes answered correctly | 196/196 | 182/196 | all; control fewer | ✅ |
| state-space: native session vs the oracle's recurrence (relative distance of logits) | 0.000 | 0.275 | 0, control > 0.05 | ✅ |
| tree speculation: tokens differing from plain greedy (4.4 tokens/step) | 0 | 32 | 0, control > 0 | ✅ |


## 5c. Rodada 0.7 — o escaneado de escritório, padrões na saída, fusão em disco

O caminho inteiro de uma página escaneada (PDF → CCITT → ordem de leitura → leitor → modelo de
língua), os formatos e padrões da saída restrita e a fusão em *streaming*, cada um contra o
controle que a forma ingênua produziria.

| verificação | valor | controle | limiar | ok |
|---|---|---|---|---|
| CCITT fax: streams decoded to libtiff's bitmap (G4, G3 2-D, G3 1-D) | 3/3 | 0/3 (K misdeclared) | 3/3, control 0/3 | ✅ |
| formats: generated date / date-time / ipv4 / uuid that OTP's parsers refuse (of 600) | 0 | 145 of 150 (naive \d{4}-\d{2}-\d{2}) | 0, control > 0 | ✅ |
| fusion from disk: output root = in-memory fusion; peak memory / models' size | true · 0.075 | false (another t) | same root, memory < 0.25 of the models, control differs | ✅ |
| reading order: mean CER of 8 scanned pages (PDF → CCITT → layout → reader; Tesseract 1.6 %) | 1.31 % | 53.1 % (no layout: lines across columns) | < 3 %, control > 20 % | ✅ |
| language model: CER of the held-out lines, beam + model (greedy 6.54 %) | 4.43 % | 6.97 % (shuffled corpus) | < 0.75 × greedy, control ≥ 0.95 × greedy | ✅ |
| the model abstains: random-string lines whose reading it changed (of 30) | 0 | 27 (no gate) | 0, control > 0 | ✅ |
| codes on a page (amounts, dates, IDs): CER with the model | 7.19 % | 8.69 % (greedy) | ≤ greedy | ✅ |


## 5d. Rodada 0.8 — tabelas, JBIG2, GPU residente, especialistas esparsos, Mamba-2, dossiê, cluster

Estrutura e texto de tabelas escaneadas (contra a leitura da 0.7 e a leitura livre), o decodificador
JBIG2 contra o jbig2dec, a sessão residente na GPU contra a CPU, os especialistas de 4 bits
predicados contra os densos, o Mamba-2 contra os logits do próprio transformers, o dossiê de
auditoria contra adulteração e os *shards* entre nós BEAM contra um nó só.

| verificação | valor | controle | limiar | ok |
|---|---|---|---|---|
| tables: structure, ICDAR-2013 adjacency F1 of 12 scanned tables (grid, inner, booktabs, rules under rows) | 1.000 | 0.343 (the 0.7 reading: lines in order, one column) | ≥ 0.95, control ≤ 0.6 | ✅ |
| tables: merged cells, adjacency F1 on the grids (header cells spanning rows and columns) | 1.000 | 0.905 (span detection off) | ≥ 0.95, control < value | ✅ |
| tables: mean cell CER, column types and shapes (Tesseract on perfectly cropped cells: 2.3 %) | 5.5 % | 11.7 % (free reading of the same cells) | < 8 %, control > value | ✅ |
| JBIG2: streams decoded to jbig2dec's bitmap (generic T0–T3, MMR, refinement, text in 8 corner modes, refinement/aggregate dictionaries, jbig2enc PDF mode) | 43/43 | 19/43 (generic template misdeclared) | all, control misses every generic stream | ✅ |
| JBIG2 → OCR: a scanned table in a JBIG2 PDF (globals + page), structure and text against the same page as PNG | [{7, 3}] · CER 0.0 % | [{7, 3}] (the PNG) | same table, CER < 1 % | ✅ |
| audit dossier: verifies offline (hashes, Merkle root, Ed25519, each item); single-bit alterations accepted (of 209 spread over the file) | true · 0 | the altered files themselves | verifies, 0 accepted | ✅ |
| Mamba-2 (2 groups, Δ clamped) = transformers' chunked scan: per-group norm vs mamba_ssm's formula, whole-width vs transformers (max relative logit error); native bits = oracle | 7.8e-7 / 6.7e-7, native true | 7.4e-1 / 6.6e-1 (each against the other's reference) | < 1e-5, control > 0.05 | ✅ |
| sparse 4-bit experts: predicated qgemv_masked = dense sb4 MoE, bit for bit (4 tokens, top-2 of 4) | true | false (experts' down projections × 1.5) | identical, control differs | ✅ |

## 5e. Rodada 0.9 — estúdio: difusão, determinismo e cache, ampliação, RL, 3D, áudio, MCP

O pipeline de Stable Diffusion contra o do próprio diffusers (checkpoint minúsculo incluído), o
estúdio contra si mesmo (raiz de Merkle estável, cache que só recalcula o que mudou), o ampliador
consistente contra Lanczos com a mesma projeção, as políticas contra controles, a malha contra a
esfera analítica, a reamostragem contra a decimação e o servidor MCP contra uma raiz falsa.

| verificação | valor | controle | limiar | ok |
|---|---|---|---|---|
| Stable Diffusion = diffusers (tiny checkpoint: txt2img DDIM/Euler/DPM++ 2M, img2img, inpainting): largest pixel difference | 1.4e-6 | 0.096 (DPM++ against the DDIM reference) | < 1e-5, control > 1e-3 | ✅ |
| studio: two cache-free runs of a template give one Merkle root | true | true (another seed changes the root) | equal, control differs | ✅ |
| studio: nodes recomputed after editing one parameter (feather σ) | 3 | 6 (no cache) | only the edited node and what depends on it (3 of 6) | ✅ |
| consistent upscaler: dB over Lanczos + the same projection, held-out text (worst of 2) | 2.760 | 0 (Lanczos + projection itself) | ≥ +1 dB | ✅ |
| consistent upscaler: inconsistency max(D(y) − x) on a held-out photograph | 1.1e-16 | 0.0771 (Lanczos) | ≤ 1e-12, control ≥ 0.01 | ✅ |
| RL: tabular Q-learning on FrozenLake finds value iteration's policy; its success rate | 0.747 | 0.0 (always left) | policy = optimum, 0.7–0.8, control 0 | ✅ |
| RL: the shipped CartPole policy (REINFORCE), mean return on 20 unseen starts | 472.900 | 18.1 (untrained) | ≥ 450 of 500, control < 50 | ✅ |
| 3D: marching-tetrahedra sphere — watertight, relative volume error | 0.004 | false (one face removed: watertight?) | watertight, < 0.5 %, control not watertight | ✅ |
| audio: resampling a 1 kHz tone 48 → 16 kHz, SNR against the ideal tone (dB) | 69.600 | 17.7 (decimation one source sample late) | > 50 dB, control < 30 dB | ✅ |
| MCP server: the same run again is all cache; verify accepts the root | 0 recomputed, verified true | false (a wrong root accepted?) | 0 recomputed, verified, control refused | ✅ |

## 5f. Rodada 0.10 — eclusa de substratos, treino, contexto sem fim, física, redes, figuras, fórmulas, CJK, árabe

A eclusa contra um dispositivo que arredonda operandos a bf16; o modelo que o vapor treinou contra
Witten–Bell e contra o próprio texto embaralhado; o fluxo sem fim contra posições crescentes; o
simulador contra o período exato do pêndulo e contra si mesmo em outro substrato; a identificação
do gêmeo contra medidas embaralhadas no tempo; a política contra a política nula; o alarme contra
a ausência de falha; as redes contra seus nulos; os gráficos contra rótulos permutados (que devem
ser recusados); as fórmulas contra a leitura plana; o CJK contra caracteres aleatórios; o árabe
contra o leitor latino.

| verificação | valor | controle | limiar | ok |
|---|---|---|---|---|
| airlock: the exact oracle admitted canonical; a bf16-operand device refused | :canonical | :refused | canonical vs refused (8 bits measured) | ✅ |
| envelope: a DAZ result inside the bound; twice the exact value outside | :ok | :error | ok vs error | ✅ |
| trained LM: held-out bits/byte (inference stack) vs Witten–Bell order 5 | 2.782 | 2.904 | model < WB-5 | ✅ |
| trained LM: byte-shuffled held-out text (the model reads order, not frequencies) | 2.782 | 8.477 | shuffled > model + 1.5 | ✅ |
| stream: bits/byte 14× past the training length in 64 rows vs growing positions | 3.034 | 5.808 | dense > stream + 1.5 | ✅ |
| physics: pendulum period, relative error at 16 substeps (first order: 4 → 8 → 16) | 1.7e-4 | 6.5e-4 | e16 < 2e-4, each halving ≥ 1.6× better | ✅ |
| chaos: oracle = native, bit for bit, for 200 steps; a one-ulp start parts by 20 s | true | 0.358 | identical and gap > 1e-2 | ✅ |
| twin identification: rod length recovered from noisy measurements (error) | 7.0e-4 | 0.104 | < 0.01 vs > 0.05 (time-shuffled) | ✅ |
| reinforcement: cart-pole returns on unseen starts (random search) | 200 | 79 | all 200 vs zero policy < 100 | ✅ |
| digital twin: steps from a 0.5 % fault to the alarm | 18 | — | ≤ 30 vs no alarm without the fault | ✅ |
| networks: Barabási–Albert degrees, power-law verdict (Clauset–Shalizi–Newman) | :power_law | :exponential | power_law vs not (Erdős–Rényi) | ✅ |
| networks: small-world clustering, z against the configuration null | 404.868 | 0.918 | > 50 vs |z| < 3 (random graph) | ✅ |
| networks: Louvain on a planted partition (NMI; modularity vs its null) | 1.000 | 0.205 | NMI > 0.95, Q > Q_null + 0.25 | ✅ |
| networks: scale-free robustness — giant component after 15 % random failure | 0.981 | 0.105 | > 0.9 vs attack < 0.6× failure | ✅ |
| figures: charts read within 1 % of the axis span (held-out, default style) | 0.900 | 1.000 | ≥ 0.8; permuted tick labels: all refused | ✅ |
| formulas → LaTeX: token error on Computer Modern and STIX (never in the templates) | 0.048 | 0.612 | < 0.08 vs flat reading > 3× | ✅ |
| CJK zh: CER on unseen typefaces (greedy → with the language model) | 0.0892 → 0.0919 | 0.000 | ≤ 0.15; ja, ko: the model helps (> 0.01); on random characters it changes nothing (≤ +0.005) | ✅ |
| CJK ja: CER on unseen typefaces (greedy → with the language model) | 0.0499 → 0.0262 | 0.000 | ≤ 0.08; ja, ko: the model helps (> 0.01); on random characters it changes nothing (≤ +0.005) | ✅ |
| CJK ko: CER on unseen typefaces (greedy → with the language model) | 0.1601 → 0.1129 | 0.000 | ≤ 0.2; ja, ko: the model helps (> 0.01); on random characters it changes nothing (≤ +0.005) | ✅ |
| Arabic: CER on unseen typefaces, read in visual order and returned to logical order | 0.192 | 0.911 | < 0.25 vs the Latin reader > 0.8; inverse bidi = python-bidi on every line | ✅ |
| Cyrillic: CER on unseen typefaces (Russian, a vocabulary half the training never saw) | 0.028 | 0.999 | < 0.08 vs the Latin reader > 0.6 | ✅ |

## 5g. Rodada 0.11 — descoberta, matemática, ciência, autojogo, cena viva, esboço, arquivos

Redes de ordenação contra redes sorteadas e podadas; o truque de bits contra a fórmula ingênua
que estoura; o posto 7 contra o posto 6 impossível; teoremas contra seus gêmeos falsos;
conjecturas contra as trincas triviais; a homologia sobre ℚ contra a de GF(2); o laço contra a
mancha; cada experimento de ciência contra a sua forma fechada ou o valor publicado e o seu
controle; o agente treinado contra a mesma busca sem treino; a política de muitos mundos contra a
de um; a cena, o esqueleto, a direção, o esboço e a planta contra a verdade com que foram
desenhados; e o arquivo contra um byte trocado.

| verificação | valor | controle | limiar | ok |
|---|---|---|---|---|
| discovery: sorting networks — sizes equal to the known optima, n = 3…8 (certified by the 0-1 principle) | 6/6 | 25 | 6/6; removing any comparator breaks each; random-and-pruned n = 8 larger than 19 | ✅ |
| discovery: ⌊(x+y)/2⌋ without overflow — shortest program, verified on 8/16/32 bits | 4 ops: ((x & y) + ((x ^ y) >> 1)) | false | 4 ops, none in 3, verified; the naive (x+y)>>1 fails | ✅ |
| discovery: 2×2 matrix product in 7 multiplications, exact over the integers | rank 7, 22 additions | {:error, {:not_found, 10}} | exact; rank 6 never found (Winograd 1971) | ✅ |
| geometry: classical theorems proved (numerator ≡ 0) and checked in exact rationals; their false twins refuted by both | 15/15 | Simson: {:holds, :fails} | all; Simson holds on the circle, fails off it | ✅ |
| conjecture and prove: Euler line and the nine-point circle found among 1001 candidates | true | false | found; no trivially collinear triple reported | ✅ |
| homology: Betti numbers over GF(2) and ℚ — torsion tells the Klein bottle and RP² from the torus | [{:torus, [1, 2, 1], [1, 2, 1]}, {:klein, [1, 2, 1], [1, 1, 0]}, {:rp2, [1, 1, 1], [1, 0, 0]}] | GF(2): torus = Klein | torus (1,2,1)/(1,2,1); Klein (1,2,1)/(1,1,0); RP² (1,1,1)/(1,0,0) | ✅ |
| persistent homology: a noisy loop's longest H₁ bar | 1.422 | 0.132 | loop > 1; blob (the control) < 0.3 | ✅ |
| science: quantum: a coherent state follows the classical orbit, ⟨x⟩ = x₀ cos t (split-step Fourier) | 4.0e-5 | 3.2e-14 | max |⟨x⟩ − x₀cos t| < 10⁻³ over a period; norm drift (the unitarity check) < 10⁻¹⁰ | ✅ |
| science: quantum: a packet below the barrier tunnels with the exact probability ∫T(k)|φ(k)|²dk | 0.543 | 0.001 | |simulated − exact| < 0.01; the classical particle (the control): < 0.01 crosses | ✅ |
| science: relativity: gyration at v = 0.9c takes 2πγ/B (γ = 2.29), |u| kept by the Boris pusher | 14.415 | 0.067 | period within 10⁻⁴ (relative); |u| drift < 10⁻¹²; explicit Euler (the control) gains > 1 % | ✅ |
| science: relativity: the E×B drift velocity is E/B | 0.299 | — | within 1 % | ✅ |
| science: tokamak: Grad–Shafranov solved against Solov'ev's exact equilibrium; the axis (parabola through the grid maximum) converges as h² | 2.2e-12 | 0.008 | flux error < 10⁻⁸; axis error falls ≥ 3× when h halves; the Cartesian Laplacian (the control) misses by > 10⁻³ | ✅ |
| science: chemistry: H₂ (STO-3G, R = 1.4 bohr) by restricted Hartree–Fock vs Szabo & Ostlund's −1.1167 hartree | -1.117 | 0.337 | within 10⁻⁴ hartree; pulled apart, RHF stays > 0.2 hartree above two atoms (the method's known failure, shown) | ✅ |
| science: chemistry: HeH⁺ (STO-3G, R = 1.4632 bohr) vs Szabo & Ostlund's −2.860662 hartree | -2.861 | — | within 10⁻⁵ hartree | ✅ |
| science: matter: a Lennard-Jones liquid (64 atoms) — velocity Verlet keeps the energy; its g(r) peaks near 2^{1/6}σ | 5.2e-4 | 1.0e+34 | energy fluctuation < 10⁻³ (relative) over 300 steps; explicit Euler (the control) > 10 %; first g(r) peak in [1.0, 1.25]σ | ✅ |
| science: evolution: a beneficial mutant (N = 50, s = 0.05) fixes with the Wright–Fisher chain's exact probability | 0.096 | 0.019 | within 3 standard errors; the neutral control fixes at ≈ 1/N | ✅ |
| science: genomics: a 6-taxon tree rebuilt from evolved sequences (Jukes–Cantor + neighbour joining) | 0 | 6 | Robinson–Foulds 0; distances within 10 %; shuffled sites (the control): RF > 0 | ✅ |
| science: folding: the HP 20-mer reaches the published optimum −9; on a 12-mer the search equals exact enumeration | -9 | -1.320 | E = −9 (Unger & Moult 1993); 12-mer: search = enumeration; random conformations (the control) > −3 on average | ✅ |
| self-play: the tic-tac-toe network against every optimal line of perfect play (lines lost, 8 simulations; 128) | 17/129; 0/135 | 169/175 | < 20 % at 8 (the untrained search, the control: > 80 %); none at 128 | ✅ |
| domain randomisation: steps up on four unseen cart-poles (mean of 3 seeds) | 484.258 | 274.667 | ≥ 400 and > the nominal policy + 100 | ✅ |
| living scene: sky at infinity, horizon (truth 0.417), ground walkable; the guild's light warm at the hearth | horizon 0.44, 6 layers | — | sky first, |horizon − truth| < 0.06, ground last, walkable cells; light x ∈ (0.12, 0.28), red − blue > 60 | ✅ |
| drawing rig: limb ends of a stick figure | 4 | — | 4 | ✅ |
| direction: a Portuguese prompt as operations; a nonsense word reported | 7 | ["xyzzy"] | night, heavy rain, wind, 3 people, fireflies, orbit, slow; unknown = [xyzzy] | ✅ |
| sketch: a rectangle drawn 2.2° askew comes back square and closed; the circle a circle | true | true | square; without constraints (the control) still askew | ✅ |
| floor plan: rooms (m²) and doors (m) at the scale of the longest wall | [11.98, 19.8] · [0.9, 1.0] | — | rooms 12 and 20 within 3 %; doors 0.9 and 1.0 within 0.08 m | ✅ |
| archives: an archive verifies and replays to the same result; one changed byte, re-zipped, is caught | {:ok, :same} | {:error, {:tampered, ...}} | {:ok, :same}; the forged one {:tampered, [result.json]} | ✅ |

## 5h. Rodada 0.12 — bancada, engenharia, lógica, tabuleiros, proteínas, render, direção de cena

Cada solucionador contra a forma fechada, o valor publicado ou o oráculo, e contra o seu controle:
Dormand–Prince contra RK4 de passo fixo; Crank–Nicolson (ordem 2) contra Euler implícito (ordem 1);
o sistema com unidades coerentes contra o mesmo com uma força no lugar de uma velocidade; a lata
ótima com limites contra a divergência sem eles; trapézios contra Euler no RC; Newton contra
Gauss–Seidel; dez elementos contra um; QM6 contra o Q4 que trava; invariantes contra uma espécie
que não se conserva; Fenske contra o refluxo abaixo do mínimo; a prova DRUP inteira contra a
truncada; Knuth–Bendix contra os axiomas só orientados; Tales contra a variante falsa; a proposta
certa contra a trocada; perft contra os números publicados; o CFR+ contra o jogo uniforme; o
alinhamento planejado contra o embaralhado; a fornalha contra o estimador viciado; e a direção
com nomes contra a frase sem ninguém.

| verificação | valor | controle | limiar | ok |
|---|---|---|---|---|
| workbench: Dormand–Prince 5(4) with dense output on x'' = −4x | 2.3e-10 | 2.3e-4 | < 10⁻⁸; RK4 at h = 0.1 > 10⁻⁵ | ✅ |
| workbench: Robertson's stiff kinetics at t = 40 (Hairer & Wanner), switch to Rosenbrock | 0.716 | the explicit method exhausted 20 000 steps: the system is stiff; solved with Rosenbrock 2(3) | a = 0.7158271 ± 2·10⁻⁶; stiffness detected | ✅ |
| workbench: heat equation, observed order by manufactured solution | 2.000 | 1.052 | Crank–Nicolson 2 ± 0.1; backward Euler 1 ± 0.15 | ✅ |
| workbench: dimensions checked before integrating | :ok | {:error, "v' has force (N) but v/t is m/s²"} | consistent system runs; v' with a force refused | ✅ |
| workbench: the can of least surface (equality + box bounds, projected BFGS) | 0.542 | diverged: the objective decreases withou | r = (1/2π)^⅓ to 10⁻⁶, KKT holds; unbounded: reported diverged | ✅ |
| engineering: RC transient, observed order | 1.974 | 0.994 | trapezoidal > 1.6; backward Euler < 1.3 | ✅ |
| engineering: Stagg & El-Abiad five-bus power flow | |V5| 1.017937 in 4 it. | Gauss–Seidel 119 it. | |V5| = 1.018 ± 6·10⁻⁴, mismatch < 10⁻¹⁰; GS > 10× iterations | ✅ |
| engineering: cantilever first natural frequency (consistent mass) | 9.3e-7 | 0.005 | 10 elements < 10⁻⁴; 1 element > 10⁻³ | ✅ |
| engineering: slender plate in bending, tip deflection vs Timoshenko beam | 0.009 | 0.294 | QM6 < 2 %; Q4 (locking) > 20 % | ✅ |
| engineering: conserved moieties of a reaction network (from stoichiometry alone) | 4.4e-16 | 0.932 | 2 invariants, drift < 10⁻⁹; [A] alone changes > 0.1 | ✅ |
| engineering: McCabe–Thiele at total reflux = ⌈Fenske⌉ | 7 | {:error, "R = 1.0 is below the minimum reflux 1.2559: the co | equal; R below the minimum refused | ✅ |
| logic: Schur S(3) — witness at 13 checked, DRUP refutation at 14 checked | 13 | {:error, :no_empty_clause} | 13, both certificates valid; the proof without its last lemma rejected | ✅ |
| logic: Knuth–Bendix completes the group axioms; i(x·y) = i(y)·i(x) decided | 10 | false | 10 rules, equal; oriented axioms alone: not decided | ✅ |
| logic: Thales' theorem by Gröbner basis (Rabinowitsch) | proved | not implied | proved; the altered claim not implied | ✅ |
| logic: an outside proposal is checked, not trusted | true | false | the solver's colouring accepted; one colour changed rejected | ✅ |
| boards: chess perft (start depth 3, Kiwipete depth 2) | 8902 · 2039 | — | 8 902 · 2 039 (published) | ✅ |
| boards: mate in 2 proved and replayed by the independent checker | {:ok, %{depth: 2, leaves: 1}} | :no_mate | {:ok, …}; no mate in 1 claimed | ✅ |
| boards: shogi perft from the initial position | 30 · 900 · 25470 | — | 30 · 900 · 25 470 (published) | ✅ |
| boards: Go, legal positions on 2×2 (Tromp–Taylor) | 57 | — | 57 (Tromp & Farnebäck) | ✅ |
| boards: Kuhn poker by CFR+ — exploitability and game value | 8.7e-5 | 0.458 | < 0.01 and value −1/18 ± 0.01; uniform 0.458 | ✅ |
| boards: tic-tac-toe solved exactly | -0.000 | — | a draw (0) | ✅ |
| proteins: TM-score between NMR models 1 and 2 of 1LCD | 0.909 | — | > 0.8 (the same fold); equals TM-align in proteinas_test | ✅ |
| proteins: contacts from a planted co-evolution alignment (DCA, top-k) | 0.962 | 0.019 | > 0.9; shuffled alignment < 3× chance (0.025) | ✅ |
| proteins: the pipeline alignment → DCA → distance geometry on 1A8O | 0.690 | 0.199 | TM > 0.6; from the shuffled alignment's contacts < 0.3 | ✅ |
| render: the white furnace (energy conservation) | 1.1e-16 | — | < 10⁻⁹ on every sphere pixel | ✅ |
| render: the gradient furnace a(½ + n_y/3) (the estimator's distribution) | 6.1e-4 | -0.040 | |error| < 0.005; the biased estimator < −0.03 | ✅ |
| scene: a named inhabitant directed in place and in time (pronoun resolved) | 4 | false | Arthur goes to the door and speaks at 3 s; a sentence without names aims at no one | ✅ |


## 6. Custo da eclusa

| adaptador | família | contrato | admitir (µs) | construir (µs) | reduzir (µs) | nós |
|---|---|---|---|---|---|---|
| decoder (llama) | llama | causal_lm | 182 | 1416 | 510890 | 112 |
| decoder (qwen2) | qwen2 | causal_lm | 190 | 1596 | 525142 | 124 |
| blueprint (granite) | granite | causal_lm | 187 | 1438 | 724480 | 117 |
| planted bigram (llama) | llama | causal_lm | 170 | 3386 | 394942 | 65 |
| codec (vapor_vq) | vapor_vq | codec | 137 | 6472 | 77297 | 7 |
| projector (vapor_linear) | vapor_linear | map | 110 | 77 | 32799 | 5 |

## Tempos (ms)

any_to_any 7590 · gates 3046 · hub_fit 12944 · lock 2568 · merge 3799 · merge_real 23972 · real 132172 · round06 20396 · round07 185771 · round08 216227 · round09 72958 · round10 1129375 · round11 170627 · round12 5840 · substrates 5124 · text 4234 · total 1997099
