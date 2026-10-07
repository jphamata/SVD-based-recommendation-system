# Fusão de modelos (`Vapor.Merge`, `mix vapor.merge`)

Fecha o item 6.3 do TODO e, nesta rodada (0.5.0), as duas limitações que a
0.4.0 declarou: **vazão** (≈ 0,6 M parâmetros/s) e **"TIES e DARE pioraram"**.

## Métodos

| método | resultado por tensor |
|---|---|
| `linear` | `Σ wᵢθᵢ / Σ wᵢ` |
| `task_arithmetic` | `base + λ Σ wᵢ(θᵢ − base)` |
| `slerp` | interpolação esférica de dois modelos em `t` (linear se `|cos| > 0,9995`) |
| `ties` | vetores de tarefa podados ao `density` de maior magnitude, sinal eleito por `sign(Σ wᵢτᵢ)`, média disjunta dos que concordam |
| `dare_linear`, `dare_ties` | cada entrada do vetor de tarefa mantida com probabilidade `density`, reescalada por `1/density`; depois linear ou TIES |
| **`regmean`** (novo) | para cada matriz cujas entradas foram medidas: **mínimos quadrados** `W = (Σ Wᵢ G̃ᵢ + λW̄)(Σ G̃ᵢ + λI)⁻¹`, `G = XᵀX` das ativações de entrada de cada modelo nos seus próprios dados, `G̃ = α·G + (1 − α)·diag G`; o resto `linear` |

E duas ferramentas que mudam a pergunta de "qual receita usar?" para "o que os
pesos dizem, e o que a medição mostra?":

- **`Merge.diagnose/2`** (`mix vapor.merge --diagnose`) — só pelos pesos, antes
  de fundir: quanto cada modelo se afastou da base (`relative_delta`), quanta
  energia do delta está nas entradas que o TIES mantém (`concentration`), quanto
  os sinais brigam (`sign_conflict`), o cosseno entre os deltas, e — o que
  faltava — **se os modelos têm um ancestral comum** (`weight_cosine`: ≈ 1 para
  ajustes de uma mesma base, ≈ 0 para redes treinadas de inicializações
  diferentes, cujas unidades não estão alinhadas).
- **`Merge.select/4`** (`mix vapor.merge --try "linear;ties:density=0.2;…" --eval retido.txt`) —
  funde cada candidato, mede bits por byte em texto retido pelo substrato
  certificado e fica com o melhor; o recibo `vapor.merge.select/1` guarda todas
  as notas e o digest do texto de avaliação. Os pesos fundidos e as notas são
  determinísticos: qualquer um refaz a tabela.

## Vazão: 0,6 → ≈ 11 M parâmetros/s de ponta a ponta

O 0.4.0 fundia com listas do Elixir. Agora cada tensor é cortado em blocos de
262 144 entradas fundidos em todos os escalonadores, cada bloco por um kernel
que casa os binários diretamente (`<<x::float-32-little, …>>`), sem lista
intermediária. O gerador do DARE (splitmix64) é um **contador**: o estado
depois de `n` sorteios é `chave + n·γ`, então qualquer bloco começa no seu
deslocamento — o paralelismo não muda um bit.

| 26 M parâmetros (Qwen2 aleatório, 2 vCPUs) | 0.4.0 | 0.5.0 |
|---|---|---|
| linear (só a fusão) | ≈ 13 s | 1,2 s |
| SLERP (só a fusão) | ≈ 20 s | 1,9 s |
| abrir A + B, fundir, raiz Merkle, gravar | ≈ 45 s | 2,4 s |

Saída **idêntica bit a bit** à da 0.4.0 em todos os métodos (conferido contra o
módulo antigo, inclusive recibos). Um modelo de 7 B passa de horas a cerca de
10 minutos — mas precisa de memória para três cópias em f32; fundir em
*streaming* direto do disco continua no TODO.

## Por que TIES e DARE pioraram — e quando não pioram

A 0.4.0 mediu em dois bigramas plantados (português × inglês) cujos "vetores de
tarefa" eram a diferença de duas tabelas inteiras: deltas do tamanho dos
pesos. Isso não é o regime para o qual TIES e DARE foram propostos (ajustes
finos de uma mesma base, deltas pequenos). Faltava medir no regime certo, com
modelos **treinados de verdade**.

`test/python/train_merge_models.py` treina, com PyTorch, decoders Llama de
caracteres (largura 64, 2 camadas) nos corpora congelados de `priv/quality`: uma
**base** em português + inglês, dois **ajustes finos** dela (`ft_pt`, `ft_en`) e dois
modelos **treinados separadamente** (`solo_pt`, `solo_en`, outras sementes). Ficam
em `priv/quality/merge` com SHA-256. Bits por caractere, candidatos escolhidos
numa fatia de validação e reportados numa fatia de teste disjunta
([bench/QUALITY.md §4b](bench/QUALITY.md)):

**Ajustes finos de uma base** — diagnóstico `:small_deltas` (deltas de 3 % dos
pesos, cosseno dos pesos 0,999, 62 % da energia nos 20 % maiores, sinais em
conflito em 39 % da magnitude compartilhada).

| | pt | en | média |
|---|---|---|---|
| base | 3,316 | 3,282 | 3,299 |
| ft_pt / ft_en sozinhos | 3,236 / 3,499 | 3,399 / 3,134 | 3,317 / 3,316 |
| **linear** (escolhido na validação) | 3,289 | 3,199 | **3,244** |
| SLERP | 3,289 | 3,200 | 3,244 |
| RegMean | 3,295 | 3,190 | 3,243 |
| TIES 0,2 / 0,5 | 3,288 / 3,323 | 3,218 / 3,209 | 3,253 / 3,266 |
| task arithmetic λ = 1 / 0,7 | 3,475 / 3,336 | 3,269 / 3,205 | 3,372 / 3,271 |
| DARE linear / DARE-TIES (0,5) | 3,487 / 3,388 | 3,296 / 3,232 | 3,391 / 3,310 |

A fusão linear de dois ajustes finos **é melhor que a base nos dois idiomas e,
na média, melhor que cada especialista** — cada um continua o melhor no seu
próprio idioma, mas paga caro no outro; a fusão é um modelo só que serve aos
dois. É a promessa da fusão, medida, e do tamanho que ela tem. O TIES a
densidade baixa chega perto; o DARE e a aritmética de tarefas com λ = 1 pioram.

**Treinados separadamente** — diagnóstico `:unrelated` (cosseno dos pesos 0,34).
A média linear dá 5,08 bits/caractere contra 3,3 de cada especialista: pior
que qualquer um, como o diagnóstico avisou antes de fundir. Uma rede é a mesma
função sob qualquer permutação das suas unidades ocultas; duas redes treinadas
separadamente não estão alinhadas, e média de pesos desalinhados é ruído.

O que os números dizem, por primeiros princípios:

- **DARE** perturba cada delta em `√((1 − p)/p)` *da norma do próprio delta*
  (100 % a p = 0,5). É inofensivo só se o ajuste for redundante — o que o
  artigo observou em modelos de bilhões com deltas de 0,1 %. Num modelo
  pequeno, sem redundância, a perturbação é dano. Isso não se lê nos pesos; se
  mede.
- **Aritmética de tarefas com λ = 1** aplica cada delta com força total —
  inclusive a parte de cada um que piora o domínio do outro (ajustar para o
  português empurra para longe do inglês). `λ = 1/n` é a média linear.
- **TIES** só descarta energia (38 % a densidade 0,2) e elege sinais; quando os
  deltas não são esparsos, isso é perda, não limpeza.
- **RegMean** (mínimos quadrados sobre as ativações de cada modelo) empata com a
  linear aqui. Num bigrama plantado as ativações são *one-hot*: o Gram é
  diagonal e o RegMean vira, **exatamente**, a média ponderada pela contagem de
  cada contexto (testado em forma fechada). Seu ganho conhecido aparece com
  ativações correlacionadas e especialistas em tarefas diferentes; aqui foi
  medido e não ganhou — está registrado assim.

Por isso a recomendação do `diagnose` termina em "meça": o vapor não promete um
método vencedor; entrega a medição reprodutível que escolhe.

## Do disco para o disco (0.7.0)

Fundir dois modelos de 7 B pela API em memória pede três cópias de 28 GB.
`Merge.stream/3` (`mix vapor.merge --stream`) lê **um tensor por vez** de cada
entrada (pelo índice do safetensors, sem carregar o resto), funde-o pelos
**mesmos kernels na mesma ordem de blocos**, e escreve-o já no arquivo de
saída — o cabeçalho do safetensors é calculado antes de qualquer dado, a
partir das formas declaradas. A memória é a do maior tensor vezes o número
de entradas.

- **Os mesmos bits**: os arquivos são byte a byte os que `write_sharded/4`
  escreve a partir de `merge/2`, para linear, *task arithmetic*, SLERP, TIES,
  DARE e saída bf16, com fragmentação; o recibo traz as mesmas raízes Merkle
  de entradas e saída (`merge_stream_test.exs`).
- **Medido**: três modelos de 20 MB, pico de memória binária **4 MB** em
  *streaming* contra **83 MB** em memória; na suíte de qualidade, pico/tamanho
  dos modelos ≈ 0,1 com a mesma raiz e um controle (outro `t`) que muda a raiz.
- **Achado medindo**: na suíte de qualidade, chamado de um processo com um
  *heap* grande, o pico subiu para 0,9× o tamanho dos modelos — os tensores
  já escritos ficavam como lixo até a próxima coleta daquele processo. Agora
  cada tensor escrito é coletado antes do próximo ser lido (0,08×, com o
  mesmo chamador).
- **O que o *streaming* não faz**: a admissão completa da eclusa (ela lê os
  pesos). A compatibilidade é conferida no que os arquivos declaram — o
  *digest* canônico do `config.json`, o adaptador que a eclusa reivindica a
  partir da configuração e do índice de tensores, nome, forma e tipo de
  cada tensor — e o campo `config` do recibo é esse *digest*. RegMean fica
  fora (mede ativações dos modelos rodando).

## O que é diferente aqui

- **Compatibilidade checada na eclusa.** Mesmo adaptador, mesmo contrato, mesmos
  nomes e formas, e — salvo `allow_config_mismatch: true` — o mesmo *digest* de
  configuração. Fundir duas topologias é uma recusa; um NaN num peso é uma
  recusa que nomeia o tensor.
- **Os mesmos bits em todo host.** binary64 `+ − × ÷ √` (corretamente
  arredondados), `acos` sem libm, um arredondamento para binary32 por entrada;
  os Grams do RegMean vêm do substrato (programas: as ativações e `XᵀX`) e o
  sistema é resolvido por Cholesky em binary64 (`Vapor.Linalg`).
- **Recibos.** `vapor.merge/1` nomeia método, parâmetros, *digests* de
  configuração, raízes Merkle dos pesos de cada entrada, da base e da saída e,
  no RegMean, o *digest* dos Grams; `vapor.merge.select/1` acrescenta todas as
  notas. Co-assináveis por nós independentes.
- **A eclusa diz o que medir.** O núcleo não sabe o que é uma camada: o
  adaptador declara `taps/1` — as ativações do programa que alimentam cada
  matriz (no decoder: a entrada normalizada da atenção para q, k, v; a do MLP
  para gate e up; a norma final para a cabeça).

## Limites

- RegMean resolve na BEAM: `O(d³)` por matriz e `O(saída·d²)` para as linhas —
  segundos para larguras de algumas centenas, impraticável para 4 096. O
  caminho é o produto e a inversa como programas no worker. Também não mede
  `o_proj` e `down_proj` (suas entradas não são ativações nomeadas no programa).
- *Streaming* do disco para o disco (0.7.0): ver a seção abaixo; o RegMean
  ainda precisa dos modelos em memória (mede ativações).
- Alinhamento de permutações (Git Re-Basin) para fundir redes sem ancestral
  comum: o diagnóstico detecta o caso; o alinhamento não foi implementado.
