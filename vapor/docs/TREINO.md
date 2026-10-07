# Treino e contexto sem fim (0.10)

> Pedido: "pipeline de treinamento completo também, com HPC"; e, no anexo da
> rodada, "inverter o RoPE" para contexto infinito. Escrutínio:
> [DIRETRIZ.md §13](DIRETRIZ.md). Testes: `train_lm_test.exs`,
> `streaming_test.exs`, `cluster_test.exs` (treino entre nós).

## 1. Pré-treino com os bits de uma máquina (`Vapor.Train.LM`)

Um Llama de bytes (vocabulário 256: nenhum tokenizador a confiar),
pré-treinado de ponta a ponta como **programas vapor**:

- **Gradientes por diferenciação reversa com *let-bindings***
  (`Vapor.Autodiff.grad_lets/5`). Sem eles, a árvore de termos de um
  transformer cresce exponencialmente com a profundidade (um modelo
  minúsculo levava mais de 10 minutos para compilar); com eles, 0,8 s.
  Conferido contra o *autograd* do PyTorch em binary64, parâmetro por
  parâmetro: erro relativo máximo 7,7·10⁻⁷.
- **Paralelismo de dados determinístico.** Cada passo soma os gradientes
  dos micro-lotes numa **árvore binária fixa sobre blocos**, cada bloco a
  dobra à esquerda `((0 + g₀) + g₁) + …`, acumulada **dentro da sessão
  residente do worker** (o estado `acc ← acc_next` nunca volta à BEAM).
  Os bits do passo não dependem do número de workers, de quem fez qual
  bloco, nem de um worker morrer no meio (testado, e com o oráculo
  exato). A forma da árvore é parte da definição: outro tamanho de bloco
  é outra execução (o teste que pode falhar falha).
- AdamW com *clipping* pela norma global, *warmup* e cosseno; RoPE por uma
  permutação com sinal (`linear`), sequências empacotadas com máscara
  causal por bloco, entropia cruzada com semente em forma fechada.
- **Checkpoint e retomada** com os bits de uma execução ininterrupta;
  **exportação** para o formato Llama do Hugging Face (o `transformers`
  carrega e calcula os mesmos *logits*; a pilha de inferência do vapor
  também).

**O modelo embarcado** (`priv/lm`, `mix vapor.train`): 492 160 parâmetros
(d = 128, 2 camadas, 4 cabeças), 1000 passos × 1024 tokens sobre a própria
documentação do vapor. Em texto retido:

| | bits/byte |
|---|---|
| sem treino | 8,558 |
| frequência de bytes | 5,032 |
| Witten–Bell, ordem 3 | 3,913 |
| Witten–Bell, ordem 5 | 3,415 |
| **o modelo** | **2,919** |

O recibo (`priv/lm/receipt.json`) guarda a curva, os digests de cada
checkpoint, os SHA-256 dos corpora e o *schedule*: refazer dá os mesmos
digests. As amostras gulosas ainda repetem ("de a prova de a prova…"): é
um modelo de meio milhão de parâmetros, honesto sobre isso.

**HPC.** d128/L2, 1024 tokens por passo: ~2 700 tokens/s com 1 worker e
blocos de 4 micro-lotes, ~3 600 com 2 workers (máquina de 2 núcleos),
mesmos digests. A compilação de cada programa leva ~27 s (uma vez).

## 2. Contexto sem fim: o anexo, escrutinado

O anexo propõe "inverter o RoPE" — medir as posições a partir do token
atual, comprimir frequências (YaRN), ou combinar âncoras de atenção
(*attention sinks*) com uma janela deslizante (StreamingLLM) — e afirma que
o vapor "já tem 90 %". Conferido:

- **Certo:** o score de RoPE depende só da distância
  (`⟨R_m q, R_n k⟩ = qᵀ R_{n−m} k`), e duas coisas quebram um modelo além
  do comprimento de treino — ângulos nunca vistos e memória que cresce. O
  vapor tem YaRN (`rope_tables` com `{:yarn, …}`, conferido contra o
  `transformers`), tabelas de RoPE corretamente arredondadas (`Vapor.CR`),
  o anel de páginas do `Vapor.Engine` e Mamba/Mamba-2 (estado fixo).
- **Impreciso:** o anel do motor só vale para modelos cuja **janela** cobre
  todas as camadas (Mistral), e as posições continuam crescendo — a
  tabela de RoPE tem `max_seq` linhas, então o "infinito" do anel era
  limitado pelo contexto declarado. Tabelas exatas evitam deriva *do
  ângulo*, não resolvem ângulos fora da distribuição.

**O que foi feito** (`Vapor.Streaming`, `kv: {:stream, âncoras, janela}`):
o cache guarda chaves **sem rotação**, `s = âncoras + janela` linhas (as
primeiras fixas, as outras um anel); a cada passo as linhas são reunidas
na ordem lógica (`gather_row`) e giradas nas **posições do próprio cache**
`0 … s−1`, a consulta em `s−1`. Como `R_a R_b = R_{a+b}`, re-basear as
posições não muda nenhum score dentro da janela — e as âncoras ficam logo
antes dela, a uma distância que o modelo conhece. Nenhuma posição passa de
`s`, nenhuma tabela cresce, a memória é constante: o fluxo não tem fim. É
a re-rotação do StreamingLLM, feita só com operadores que a eclusa já
certifica (nenhum núcleo novo).

Medido no modelo embarcado (comprimento de treino 64), 900 bytes de texto
retido — 14× o comprimento de treino — num cache de 64 linhas:

| | bits/byte depois do byte 64 |
|---|---|
| fluxo (4 âncoras + 60 de janela) | **3,03** |
| janela sem âncoras | 3,05 |
| controle: posições crescendo até 900 | 5,81 |
| até o byte 64 (referência) | 2,53 |

E até o cache encher, o fluxo calcula **exatamente** os *logits* do modelo
causal (mesmos bits). As âncoras pouco ajudam neste modelo (treinado com
sequências empacotadas em posições aleatórias, ele não concentra atenção no
primeiro token); em modelos grandes o efeito do StreamingLLM é o
documentado — medi-lo exige pesos que esta máquina não baixa.

**Limites:** a decodificação além do cache é um token por passo (como
qualquer geração); o caminho é a sessão densa — integrar ao motor paginado
(fixar as páginas das âncoras e re-rotacionar por *slot*) está no
[TODO](TODO.md). O treino de modelos com contexto longo (e a avaliação
*needle-in-a-haystack*) não foi feito.
