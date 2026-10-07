# A eclusa de modelos (`Vapor.Lock`)

> O núcleo não sabe o que é um Llama. Ele sabe o que é um **contrato**.

## 1. O problema

Até a rodada anterior, o motor, o embedder e o carregador chamavam
`Vapor.Model.Llama.program/3` diretamente, e `Vapor.Model.Config` enumerava
oito famílias com seus detalhes. Cada modelo novo exigia editar o núcleo. Esse é
o padrão da indústria — o `modeling_*.py` por família do transformers, os
`llm_build_*` do llama.cpp, os `*_model.py` do vLLM — e é a dor real:

- **acoplamento**: o servidor, o agendador e o cache conhecem detalhes de
  arquitetura (nomes de tensores, *layouts* de cabeça, roteamento de MoE);
- **custo marginal alto**: uma família que é "Llama com outros nomes"
  (Phi-3, EXAONE, …) custa um arquivo de código inteiro, revisão e release;
- **falha silenciosa**: o que a camada não entende é frequentemente ignorado
  em vez de recusado (ver o bug do exportador GGUF no §7).

## 2. A ideia, por primeiros princípios

Um programa da álgebra do vapor já é **autodescritivo**: suas entradas, saídas
e estado são termos tipados. Então o que o núcleo precisa saber de um modelo
não é a família, é a **forma do programa**. A eclusa é a única fronteira onde
uma família é conhecida; do lado de dentro só existem:

- um `Vapor.Lock.Spec` — o contrato (`interface`), larguras, vocabulário,
  EOS, modalidades de entrada e saída, *features* que o construtor aceita, e um
  `digest` da configuração admitida. A configuração específica da família vai
  em `config`, **opaca** para o núcleo;
- um `Vapor.Program` construído pelo adaptador e **conferido contra o contrato
  na própria eclusa** (`Vapor.Lock.Contract`) antes de o núcleo vê-lo.

```
checkpoint ─► manifesto ─► claim (todos os adaptadores) ─► admit ─► Spec + pesos
                                                                       │
                       núcleo ◄── Contract.check ◄── build ◄───────────┘
```

| contrato | entradas | saídas | estado |
|---|---|---|---|
| `:causal_lm` | `tok, pos : s32[t]` (+ `last`, `sampling`, `table`, `slot`, `soft`, `soft_mask` por opção) | `logits : f32[·, vocab]` e/ou `hidden : f32[·, width]` | pares `{x, x_next}` com sortes iguais |
| `:encoder` | `rows : f32[T, k]`, `horizon : s32[T]` | `hidden : f32[T, width]` (+ `pooled`, `logits`) | — |
| `:codec` | `rows : f32[T, k]` ↔ `codes : s32[T]` | idem | — |
| `:map` | `rows : f32[T, k]` | `out : f32[T, width]` | — |

## 3. Três propriedades impostas, não esperadas

1. **Um dono por checkpoint.** Todo adaptador responde `claim/1` com
   `{:claim, score}`, `{:near, por_quê}` ou `:no`. Vence o maior escore;
   adaptadores registrados em tempo de execução vêm antes dos embutidos, então
   uma sobreposição é explícita. Ninguém reivindica → recusa tipada que lista
   **todos os quase-acertos com o reparo** (ex.: "os tensores seguem o layout do
   decoder, mas `model_type "phi9"` não é uma família conhecida — registre um
   alias cujo `like` é a família mais próxima").
2. **Contrato na eclusa.** Um adaptador com defeito é parado na fronteira
   (`Rejection` com `{:contract, nome}`), não dentro do motor.
3. **Nenhuma família no núcleo — testado.** `lock_test.exs` lê a tabela de
   átomos do BEAM compilado de 27 módulos do núcleo (motor, embedder, servidor,
   especulação, RAG, fusão, hub, qualidade, compilador, runtime, escada,
   emissores) e falha se qualquer um referenciar `Vapor.Model.Llama`,
   `Vapor.Model.Config`, `Vapor.Model.GGUF` ou qualquer adaptador. A regra é
   um teste, não uma convenção.

## 4. Adicionar um modelo: três níveis, do mais barato ao mais caro

### Nível 1 — alias (só dados, sem compilar nada)

Para famílias que *são* uma topologia existente sob outros nomes. Um mapa (ou
um arquivo JSON) declara renomeações de chaves de configuração, valores
forçados, valores exigidos, chaves proibidas, renomeações de tensores por
*template* (`{l}` = índice de camada) e **divisões de tensores fundidos** com
tamanhos expressos como produtos de campos da configuração admitida:

```json
{"id": "phi3", "model_type": "phi3", "like": "llama",
 "config": {"set": {"attention_bias": false},
            "require": {"partial_rotary_factor": [null, 1, 1.0]}},
 "tensors": {"split": [
   {"from": "model.layers.{l}.self_attn.qkv_proj.weight",
    "into": [["model.layers.{l}.self_attn.q_proj.weight", "heads*head_dim"],
             ["model.layers.{l}.self_attn.k_proj.weight", "kv_heads*head_dim"],
             ["model.layers.{l}.self_attn.v_proj.weight", "kv_heads*head_dim"]]},
   {"from": "model.layers.{l}.mlp.gate_up_proj.weight",
    "into": [["model.layers.{l}.mlp.gate_proj.weight", "intermediate"],
             ["model.layers.{l}.mlp.up_proj.weight", "intermediate"]]}]}}
```

Esse é o alias embutido do **Phi-3/Phi-4**. Registro:
`Vapor.Lock.register(mapa)`, `Vapor.Lock.register_json("arquivo.json")` ou
`VAPOR_LOCK_ALIASES=a.json:b.json` no ambiente. A admissão roda as checagens da
família-base sobre a configuração reescrita, então um alias pode **estreitar**
o que a base aceita, nunca **alargar** (LongRoPE e `partial_rotary_factor ≠ 1`
são recusados pelo nome). Divisões que não cobrem as linhas exatamente são
recusadas. Cadeias de `like` são seguidas (até 8; laços são recusados).

Verificação: um checkpoint Phi-3 montado fundindo pesos Llama é admitido, os
tensores voltam **idênticos** aos originais e o programa dá **os mesmos bits**
do Llama; e uma referência NumPy escrita independentemente (fatiamento do
`qkv_proj` e `chunk(2)` do `gate_up_proj` como no HF) concorda com erro
relativo < 2·10⁻⁶ (`test/python/np_reference.py`).

### Nível 2 — blueprint (poucas linhas de código)

Para famílias que são uma topologia existente **com botões a mais**. O
adaptador só traduz a configuração para os botões. O exemplo embutido é o
**IBM Granite 3.x** (`Vapor.Lock.Adapters.Granite`): `embedding_multiplier`,
`attention_multiplier`, `residual_multiplier` e `logits_scaling` viram
`embed_scale`, `attn_scale`, `residual_scale` e `logit_divisor` do decoder (os
dois últimos são botões novos nesta rodada, em `Vapor.Model.Config`). A
configuração original é guardada, então `config.json` volta como Granite.
Verificado contra referência NumPy independente (< 2·10⁻⁶).

### Nível 3 — topologia (um programa novo, nenhum kernel novo)

Para formas de programa genuinamente novas. Embutidas nesta rodada:

- `Vapor.Lock.Adapters.Encoder` — encoder bidirecional sobre linhas (patches,
  quadros): ViT do Hugging Face (`model_type: "vit"`, com ou sem prefixo
  `vit.`, cabeça de classificação) e `vapor_encoder` (a mesma topologia
  descrita diretamente, LayerNorm ou RMSNorm). Verificado contra referência
  NumPy que computa a **convolução como convolução** (< 2·10⁻⁶) e bit a bit
  entre nativo e oráculo;
- `Vapor.Lock.Adapters.Codec` — codec VQ (`vapor_vq`): codificar = argmax via
  `sample` guloso; decodificar = `gather_row`;
- `Vapor.Lock.Adapters.Linear` — projetor afim (`vapor_linear`), o elo entre
  modalidades.

Como cada um é construído só com operadores existentes, ver
[ANY_TO_ANY.md](ANY_TO_ANY.md).

## 5. Diagnóstico

```sh
mix vapor.lock ./MeuModelo            # quem reivindica, por quê, spec, tensores faltando/sobrando
mix vapor.lock ./MeuModelo --alias meu_alias.json
mix vapor.lock --list
```

`Vapor.Lock.explain/2` devolve a resposta de cada adaptador, o vencedor e —
quando admitido — os tensores que o construtor espera e não encontrou (ou
encontrou com outra forma) e os que ele não vai ler. É o roteiro para escrever
um alias: o que sobrou de um lado e faltou do outro é exatamente a tabela de
renomeações.

## 6. Custo

Medido por `mix vapor.quality` ([QUALITY.md §6](bench/QUALITY.md)): admitir
custa 0,1–0,25 ms e construir 0,05–14 ms em modelos minúsculos; o *lowering*
(10–650 ms) domina. A abstração não tem custo mensurável no caminho quente —
o programa produzido pelo adaptador do decoder é **o mesmo termo** que o
construtor direto produzia (testado com `==`).

## 7. O que esta rodada consertou no caminho

- **Exportação GGUF silenciosamente errada.** `Vapor.Model.GGUF.write/4`
  gravava toda família que não fosse `qwen2` como `llama`, descartando em
  silêncio q/k-norm (Qwen3), especialistas (Mixtral, Qwen3-MoE), MLA
  (DeepSeek-V3), normas sanduíche e GeGLU (Gemma 3). Agora é uma recusa tipada
  que nomeia o que se perderia e sugere safetensors.
- **Phi-3 exportado para o HF** perderia a janela deslizante (o `to_map` de
  `llama` não a escreve); agora um Llama com janela e sem vieses é escrito com
  a grafia do Mistral, que a carrega.

## 8. Rodada 0.5.0: o oráculo passou a ser o próprio `transformers`

- **Paridade executada.** `test/vapor/lock_hf_test.exs` gera os checkpoints
  com o `transformers` 5.18 (o próprio `save_pretrained`: os nomes, as fusões
  de tensores e as chaves de configuração que ele escreve) e compara com o
  forward dele: Phi-3, Phi-3 com rotary parcial, Granite, ViT (classificação
  e `ViTModel` com pooler), CLIP-vision com projeção — ≤ 4,5·10⁻⁷ relativo,
  gulosa idêntica, nenhum tensor sem uso.
- **O bug que a referência NumPy não podia achar.** O `transformers` ≥ 5 move
  `partial_rotary_factor` para dentro de `rope_parameters`. A 0.4.0 checava a
  chave no topo — e admitia um checkpoint estilo Phi-4-mini que então
  calculava errado (erro relativo 0,5). Correção em duas partes: o rotary
  parcial agora é **construído** (os pares girantes vão para o layout
  *rotate-half* por uma permutação dobrada nas linhas de q/k — os escores de
  atenção não mudam sob a mesma permutação dos dois — e a tabela tem pares de
  passagem com cos = 1, sin = 0, o mesmo mecanismo do MLA); e **toda chave
  desconhecida** de `rope_parameters` é recusada pelo nome: chaves que mudam
  a matemática são lidas ou recusadas, nunca puladas.
- **Topologias novas:** `vapor_mlp` (o perceptron: classificador, denoiser de
  difusão, projetor de duas camadas); o encoder ganhou a torre de visão do
  CLIP, o pooler do ViT, a cabeça por linha (`head: "rows"`, para OCR e fala
  com CTC) e programas mais curtos (`rows:`).
- **O que medir, dito pela eclusa:** o adaptador declara `taps/1` — as
  ativações que alimentam cada matriz — e a fusão por mínimos quadrados lê
  isso, sem saber o que é uma camada.

## 9. Rodada 0.6: estado, partes, imagens

Cinco topologias novas, todas de nível 3 (um programa novo, nenhum kernel
novo), todas conferidas contra a implementação de referência executando:

| adaptador | família | contrato | conferido contra |
|---|---|---|---|
| `Mamba` | `MambaForCausalLM` | `:causal_lm` com a *feature* `:recurrent` (sem `pos`: o estado é a história) | `transformers` (3,4·10⁻⁷, gulosa idêntica) |
| `Whisper` | `WhisperForConditionalGeneration` | `:causal_lm` (decoder) + a **parte** `encoder: :encoder` | `transformers` (4,8·10⁻⁷ / 4,2·10⁻⁷, gulosa idêntica) |
| `VAE` | `AutoencoderKL` (decoder) | `:map` | diffusers (8,7·10⁻⁷) |
| `DiT` | `DiTTransformer2DModel` | `:map` | diffusers (3,1·10⁻⁷) |
| `Encoder` (CLIP texto) | `CLIPTextModel(WithProjection)`, a metade de texto de um `CLIPModel` | `:encoder` (com `tok` e `pick`) | `transformers` (3,7·10⁻⁷) |

Duas extensões do contrato, ambas pequenas e conferidas na fronteira:

- **Partes.** Um modelo de vários programas (o Whisper: encoder e decoder)
  declara em `spec.parts` o contrato de cada parte extra;
  `Lock.build(spec, ws, part: :encoder)` confere o programa contra **esse**
  contrato, e uma parte não declarada é recusada pelo nome.
- **Janela em todas as camadas.** O *callback* opcional `ring_window/2`
  diz ao `Vapor.Engine` quando pode guardar as páginas em anel
  (`Vapor.Lock.ring_window/2`); o motor não lê a configuração da família.
- **Recorrente.** Um `:causal_lm` com a *feature* `:recurrent` dispensa `pos`;
  o `Vapor.Engine` (cujo modelo de memória é o cache paginado) o recusa por
  contrato — falta `:paged` — e `Vapor.Recurrent` o serve.

As recusas com quase-acerto acompanham: `falcon_mamba` (normaliza B, C e Δ
dentro do misturador), `jamba`/`zamba`/`bamba`/`nemotron_h`/`falcon_h1`
(híbridos com atenção; o `mamba2` tem adaptador próprio desde 0.8), `PixArt`/`SD3`/`Flux` (DiTs com atenção a texto ou
posições rotativas 2-D) — cada uma diz o que falta.

## 10. O que ainda não é garantido

- **EXAONE, InternLM2, Baichuan, OLMo2** e outros não são embutidos: alguns
  são alias puros (EXAONE parece ser só renomeações), outros não (Baichuan2 tem
  `NormHead`; OLMo2 normaliza q/k na projeção inteira e põe a norma depois).
  Não embuti o que não pude verificar.
- LLaVA, U-Net/ControlNet, Jamba: o caminho está desenhado e as peças
  testadas; os adaptadores não (o Mamba-2 entrou na 0.8).
- Paridade com pesos aleatórios não é qualidade: VAE, DiT, Whisper e Mamba
  não foram medidos com pesos treinados (não estão neste ambiente).
- `Vapor.Train` ainda fala a língua do decoder (LoRA sobre `%Config{}`): é
  uma capacidade de adaptador que deveria virar *callback* da eclusa.
- O manifesto de um diretório HF é montado depois de ler os pesos; para
  checkpoints enormes, reivindicar pelo índice antes de ler seria melhor.
