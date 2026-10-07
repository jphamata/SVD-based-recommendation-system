# A diretriz desta rodada, e o seu escrutínio

> Pedido (2026-10-02): "Forneça um zip refinado ao final → esta própria
> diretriz está sujeita a refinamentos e escrutínio → resultado deve ser um
> artefato que resolva as dores reais da indústria e academia com inovação
> real, pensamento lateral e primeiros princípios + any to any, fusão, resolva
> o todo + demais questões pertinentes + sem dependências de conhecimentos
> prévios de modelos no núcleo (fazer com uma eclusa de modelos, para que as
> especificidades de cada modelo e suas topologias sejam abstraídas na eclusa e
> que seja fácil adicionar suporte na eclusa a novos modelos diversos e
> inovadores) + teste de benchmark + de qualidade das saídas dos modelos texto
> e any to any para garantir que não está sendo gerado apenas ruído".

## 1. Escrutínio

Cada cláusula foi lida como requisito testável. Onde não era, foi
reformulada — e a reformulação está aqui, para ser contestada.

| cláusula | leitura literal | problema | refinamento adotado |
|---|---|---|---|
| "resolva o todo" | implementar tudo | infalsificável; leva a fachada (o que o próprio projeto proíbe) | resolver **as dores verificáveis** e declarar, com motivo, o que fica fora (§4) |
| "inovação real" | novidade | novidade não é critério; o critério é uma propriedade que antes não existia e agora é testada | cada entrega tem um teste que falharia sem ela |
| "any to any" | todo par de modalidades | `N²` conversores é o anti-padrão; e "imagem" sem modelo treinado é ruído | **hub com pivô** (`N` codecs), **sem operador novo** na álgebra, **medido** em dados retidos com controle |
| "fusão" | ambíguo | três sentidos: fundir pesos (*model merging*), fundir modalidades, fundir *kernels* | os dois primeiros (fusão de *kernels* o compilador já faz: *cut sweep*) |
| "sem conhecimento prévio de modelos no núcleo" | intenção | "núcleo" e "conhecimento" precisam de definição operacional | núcleo = 27 módulos listados; conhecimento = referência a módulo de família; **teste** sobre a tabela de átomos do BEAM |
| "fácil adicionar modelos" | intenção | "fácil" sem medida | três níveis de custo: dados (JSON), *blueprint* (≈ 40 linhas), topologia; o mais barato que serve |
| "teste de qualidade … não é só ruído" | um teste | um teste com limiar à mão aprova ruído sem que ninguém saiba | portões **calibrados contra controles**, que se recusam a existir se não separam; modelos **plantados** com verdade em forma fechada |
| "sem dependências" (implícito, axioma do projeto) | — | — | mantido: `deps: []`; PNG, PPM, WAV, FFT, k-means, Cholesky, ridge, CBOR — tudo aqui |

## 2. Dores reais atacadas

**Indústria**

1. *Acoplamento de família no servidor* → eclusa com contratos; o motor
   serve qualquer `:causal_lm` que declare `:paged`, `:sample`, `:last`.
2. *Custo marginal de cada arquitetura nova* → alias por dados (Phi-3 em 20
   linhas de JSON), *blueprint* (Granite).
3. *Falha silenciosa* → recusa tipada com quase-acertos e reparo; o
   exportador GGUF que gravava errado em silêncio agora recusa (bug real
   encontrado nesta rodada).
4. *Regressões que deixam a saída "plausível"* → portão de qualidade em CI
   (`mix vapor.quality` sai 1), `--model` para checkpoints reais.
5. *Fusão sem procedência* → recibo assinável e co-assinável, bits
   independentes do host.
6. *Um runtime por modalidade* → o mesmo compilador certificado carrega
   imagem e áudio.

**Academia**

1. *Reprodutibilidade de avaliações* → todo número do relatório é
   determinístico (substrato nativo = oráculo, bits iguais) e regenerável.
2. *Métricas sem controle* → cada verificação traz controle e limiar
   declarados.
3. *Generalização vs. memorização* → pares retidos, variantes; a primeira
   versão do mundo media recordação (PSNR 155 dB) e foi endurecida antes de
   publicar qualquer número.
4. *Hipóteses de métodos* → TIES/DARE medidos onde a hipótese de
   esparsidade falha.

## 3. Pensamento lateral, concretamente

- **Bidirecional = causal com horizonte no fim.** Nenhuma máscara nova.
- **Convolução de *stride* = *kernel* = `linear` sobre linhas.** Nenhum `Conv2d`.
- **VQ = `sample` guloso sobre `2x·c − ‖c‖²`.** O operador de amostragem do
  LLM vira o tokenizador de imagem.
- **Injeção de modalidade = `sel`.** O mesmo operador que torna o MoE
  imune a NaN de especialista não escolhido.
- **Treinar = contar.** Um bigrama embutido exatamente num transformer é
  verdade de referência para toda a pilha.
- **O teste se testa.** Um portão que não separa seus controles não existe.
- **Contrato em vez de família.** O programa já é autodescritivo; o núcleo lê
  sortes, não nomes.

## 4. O que ficou de fora, e por quê

| item | motivo |
|---|---|
| geração de imagem por difusão, vídeo | exige U-Nets/DiTs treinados; os blocos se expressam, nada foi construído nem verificado — fazer agora seria fachada |
| ComfyUI, *upscaling*, renderização de jogos | interface de produto / convolução densa sem caso / rasterização: ver [ANY_TO_ANY.md §5](ANY_TO_ANY.md) |
| Whisper, CLIP, LLaVA pré-treinados | caminho desenhado e peças testadas (encoder, espectro, projetor, injeção); adaptadores não embutidos sem verificação contra o `transformers` |
| paridade dos níveis novos com o `transformers` | sem PyTorch neste ambiente; feita contra referências NumPy independentes; o nível `:torch` fecha |
| fusão rápida de modelos de bilhões | BEAM ≈ 0,6 M parâmetros/s; fusão como programa no worker é o caminho |
| TIES/DARE "melhores" | medidos piores para modelos densos; não foram "ajustados" para parecer bons |

## 5. Resposta ao documento anexo ("Hoje: NÃO")

O documento que acompanhou o pedido descrevia o vapor como estritamente
texto→tensores→texto. Item a item, depois desta rodada:

| afirmação do anexo | agora |
|---|---|
| "O vapor não tem modelos de áudio nem de visão" | **parcial → sim para a infraestrutura**: encoder bidirecional (ViT do HF admitido e verificado contra NumPy), espectro certificado, síntese, codecs VQ, projetores, injeção; mundo de teste medido em 10 rotas. Modelos de áudio pré-treinados: ainda não. |
| "faltam `Conv2d`/`Conv3d`" | **não são necessárias** para *patch embedding* (= `linear`); convoluções sobrepostas são somas de `gather_row` deslocados — expressáveis, não empacotadas |
| "faltaria atenção cruzada" | expressável sem operador novo (atenção sobre K/V de outro fluxo com horizonte no fim); a rota usada aqui é a injeção (LLaVA), testada |
| "faltam blocos de patchificação e VAEs" | patchificação: `Vapor.Modal.Image.patches/3`, exata; VAE: não (o codec VQ ocupa o lugar de tokenizador discreto) |
| "Fusão de modelos — NÃO, roadmap P3" | **sim**: seis métodos, determinísticos, com recibo, CLI, *streaming* |
| "ComfyUI / difusão / upscaling / vídeo — NÃO" | **continua não**, pelos motivos do §4 |
| "Games: renderização — NÃO; NPCs — SIM" | inalterado |
| "GUI: Livebook, LiveView, Open WebUI, CLI" | **console web próprio** em `/` (conversa com evidência, documentos, contrato, medidor de ruído), além de `mix vapor.lock`, `vapor.merge`, `vapor.quality`, `vapor.rag` e a galeria PNG/WAV; Open WebUI e afins continuam funcionando pela API |

## 6. Segundo pedido: "UI/UX e RAG de zip, PDF, imagens etc."

| cláusula | leitura | refinamento |
|---|---|---|
| UI/UX | "uma interface" | a interface que falta nas existentes não é outro chat: é a **evidência ao lado da resposta** (fontes com caminho até a página, citações conferidas, recibos) e um **medidor de ruído** a um clique; offline, sem CDN, servida pelo próprio servidor, utilizável sem modelo |
| RAG de zip | descompactar e indexar | sem *zip bomb* (razão e teto checados antes de inflar, cabeçalho mentiroso recusado), recursivo, e com o caminho dentro do arquivo preservado até o recibo |
| RAG de PDF | extrair texto | correto onde extratores falham (fontes compostas via `ToUnicode`, *object streams*), recusa explícita de criptografados, aviso em páginas sem texto; conferido contra o `pdftotext` |
| RAG de imagens | "entender imagens" | sem OCR nem CLIP aqui, o honesto é: texto embutido (PNG/EXIF) no índice de texto e **similaridade visual** num índice próprio — dito com essas palavras na interface |
| "etc." | todo formato | Office, OpenDocument, EPUB, HTML, CSV/JSON, Markdown; e o que não foi feito (OCR, JPEG, ordem de leitura de colunas) está no TODO, não escondido |

Detalhes: [DOCUMENTOS.md](DOCUMENTOS.md), [CONSOLE.md](CONSOLE.md).

## 7. Como contestar este documento

Toda afirmação acima aponta para um teste ou para um número regenerável:

```sh
mix test test/vapor/lock_test.exs test/vapor/modal_test.exs test/vapor/merge_test.exs \
         test/vapor/quality_test.exs test/vapor/docs_test.exs test/vapor/console_test.exs
mix vapor.quality                       # docs/bench/QUALITY.md, quality.json, modal/
mix vapor.quality --model ./SeuModelo --text retido.txt --reference corpus.txt
```

## 8. Rodada 0.5.0: "ataque as limitações"

> Pedido (2026-10-02, tarde): "Forneça um zip refinado ao final → esta própria
> diretriz está sujeita a refinamentos e escrutínio → resultado deve ser um
> artefato que resolva as dores reais da indústria e academia com inovação
> real, pensamento lateral e primeiros princípios (ataque as limitações, todo
> e complete o que for pertinente … + testes de qualidade para garantir que
> as respostas não sejam apenas ruído + UI/UX original e elegante)", seguido
> das seis limitações que a 0.4.0 declarou; depois: "sem pontas soltas +
> minimize o todo + na UI/UX inglês padrão e alternativa português (além de
> light e dark) + crie um logo SVG + um favicon + considere ou não Tauri +
> GUI/TUI + garanta que não esteja gerando ruído de saída".

### Escrutínio

| cláusula | leitura literal | problema | refinamento adotado |
|---|---|---|---|
| "ataque as limitações" | fazer sumir as seis frases | apagar uma frase de limitação é fácil; fechar a limitação exige um teste que falharia antes | cada limitação vira um teste novo **ou** uma recusa medida com motivo (tabela abaixo) |
| "conferido contra NumPy que eu mesmo escrevi" | trocar NumPy por PyTorch | a lição é epistemológica: um oráculo escrito a partir da mesma leitura compartilha o erro | o oráculo agora é o **código de referência executado** (`transformers` gravando e calculando); achou um bug real que o NumPy não podia achar |
| "mundo sintético" | usar dados reais | dados reais sem controle só trocam o tipo de ruído | dados reais **retidos** (fontes, voz, dígitos nunca vistos) **com** controles (acaso, texto fluente errado, memorização) |
| "não há difusão" | implementar difusão | difusão sem verificação é gerador de imagens bonitas | o amostrador é testado contra o denoiser ótimo em forma fechada antes de qualquer rede; a rede é medida por um juiz independente e contra cópia |
| "não há OCR" | adicionar OCR | OCR por dependência (Tesseract) contradiz `deps: []` e a eclusa | OCR é um **modelo admitido** pela eclusa; o Tesseract fica como oráculo de comparação nos testes |
| "busca por cor, não por significado" | busca semântica | sem pesos de CLIP no ambiente, prometer "semântica" seria fachada | o significado que **existe** nas imagens de documentos é o texto: OCR → índice; torre de visão do CLIP conferida; torre de texto declarada no TODO |
| "fusão a 0,6 M/s" | mais rápido | rapidez que muda bits destrói recibos | 20×, e **bit a bit igual** à 0.4.0 (testado contra o módulo antigo) |
| "TIES e DARE pioraram" | fazê-los melhorar | ajustar o benchmark até o método ganhar é a dor acadêmica clássica | medir onde a hipótese vale (ajustes finos) e onde não vale (sem ancestral comum), e dar ao usuário o **diagnóstico e a seleção por medição** em vez de uma receita |
| "minimize o TODO" | apagar itens | itens apagados sem entrega são pontas soltas escondidas | fechar o que coube, mover o fechado para o CHANGELOG, manter o aberto com o motivo |
| "inglês padrão, português alternativa" | traduzir | traduzir a página não traduz o servidor; prosa gerada no servidor em uma língua só vaza | todo texto da página num dicionário; os diagnósticos são **montados no cliente a partir dos números**, nas duas línguas; mensagens do servidor ficam em inglês (como a API) |
| "considere ou não Tauri" | decidir | — | decidido **não**, por critérios (toolchains, `deps: []`, sidecar da BEAM, worker só Linux), com a alternativa sem dependência adotada: o console como app instalável; revisitar quando houver worker fora do Linux |
| "GUI/TUI" | duas interfaces | duas interfaces com lógicas diferentes divergem | a TUI é um intérprete puro sobre os mesmos módulos que a GUI chama (`Vapor.TUI.eval/2`, testado sem terminal) |
| "garanta que não gere ruído de saída" | sem ruído | duas leituras: ruído no *conteúdo* e ruído no *canal* | conteúdo: toda resposta carrega sua medida (confiança, certeza, leitura de volta, distância ao treino) e a suíte reprova saídas sem sinal; canal: sem cor fora de terminal, `NO_COLOR` respeitado, CLIs sem logs de depuração |

### As seis limitações, depois desta rodada

| limitação (0.4.0) | agora | evidência |
|---|---|---|
| Phi-3, Granite, ViT conferidos contra NumPy próprio | conferidos contra o `transformers` 5.18 executando; + rotary parcial (bug corrigido), pooler do ViT, CLIP-vision | `lock_hf_test.exs` |
| mundo any-to-any sintético | rotas em dados reais retidos: caligrafia (dois sentidos), fala de voz nunca ouvida, cadeia voz → desenho | `mix vapor.quality` §4c |
| sem difusão, OCR, JPEG | difusão verificada em forma fechada; OCR por modelo admitido; JPEG = libjpeg bit a bit | `diffusion_test.exs`, `vision_test.exs`, `jpeg_test.exs`, §4c |
| busca de imagem só visual | texto nas imagens indexado por OCR; CLIP-vision conferido; torre de texto aberta | `docs_test.exs`, `lock_hf_test.exs`, TODO |
| fusão a 0,6 M/s | ≈ 11 M/s ponta a ponta, mesmos bits | `merge_test.exs`, FUSAO.md |
| TIES/DARE pioraram | explicado e medido nos dois regimes com modelos treinados; diagnóstico e seleção por medição | §4b, `merge_test.exs` |

## 9. Rodada 0.6: os dois anexos — rumo à 1.0 e "Sora / Midjourney / modelos de mundo"

> Pedido (2026-10-03): o mesmo da §8 ("zip refinado … sujeito a refinamentos
> e escrutínio … dores reais … testes de qualidade para que as respostas não
> sejam apenas ruído … UI/UX original e elegante"), com um anexo de
> **quinze itens para a 1.0** (MoE esparso por permutação determinística de
> tokens; cache latente do MLA por absorção da query; worker para Apple
> Silicon; eclusas de documentos plugáveis com extratores em subprocesso
> confinado; sessões residentes no Vulkan; cache KV circular; kernel NTT
> verificado; divisão canônica corretamente arredondada; lema de Higham em
> Lean; torre de texto do CLIP + BPE com `</w>`; Whisper; Mamba/SSM;
> *streaming* zero-cópia Plug/Bandit; *store* Ecto oficial; atestados
> ancorados em logs de transparência). Depois: "continue de onde parou, sem
> pontas soltas e sem abandonar as diretrizes anteriores + pondere sobre as
> novas diretrizes (sujeitas a refinamento e escrutínio, testadas para evitar
> ruído como resposta e benchmark)", com um segundo anexo sobre o que faltaria
> para chegar a Sora, Midjourney e modelos de mundo: `Conv2d`/`Conv3d`, VAE
> contínuo, DiT, FlashAttention em Vulkan/CUDA, fusão multimodal latente por
> atenção cruzada (sem pivô de texto), condicionamento espacial (ControlNet),
> KV de longo prazo para vídeo, paralelismo de tensor entre nós, árvores de
> decodificação especulativa.

### Escrutínio da rodada

| cláusula | leitura literal | problema | refinamento adotado |
|---|---|---|---|
| "MoE esparso via permutação determinística de tokens" | ordenar os tokens por especialista e despachar em blocos, como fazem os kernels de GPU | em CPU o GEMV é *weight-stationary*: o custo é **ler os pesos**, não agrupar tokens; a permutação acrescenta gather/scatter e uma forma dependente dos dados, que quebra os extents certificados | **predicação por linha** (`linear_masked`): o mesmo kernel pula as linhas não escolhidas e lê só os especialistas que algum token escolheu; bits **= denso** em todo substrato; 2,6× no decode (T = 1) |
| "cache latente do MLA por absorção da query" | absorver `W_UK` na query e guardar só o latente | a absorção muda a ordem de contração (outros bits) e triplica os FLOPs da atenção em troca de 85× menos memória — não é otimização grátis | a forma latente é **outro programa**, com bits próprios, conferido contra o `transformers` (argmax igual, 3,5·10⁻⁷); a expandida continua lá (`mla: :expanded`); a escolha é do operador, com o custo medido |
| "worker para Apple Silicon" | portar | sem macOS aqui; no Darwin não há seccomp (`sandbox_init` é obsoleto), a geração de código exige `MAP_JIT` + `pthread_jit_write_protect_np` e a espera usa `os_sync_wait_on_address`; um worker não executado é pior que nenhum | **não feito, por falta de máquina** — plano no TODO; o protocolo já não depende de SO |
| "eclusas de documentos plugáveis (extratores confinados)" | aceitar extratores externos | um extrator externo é **código arbitrário sobre arquivos hostis** — exatamente o que a eclusa existe para evitar; confinar binários alheios exige o filtro seccomp *fora* do binário (lançador) | **não feito** — os leitores continuam Elixir puro, sem execução; o desenho (lançador com seccomp + protocolo de quadros) está no TODO |
| "sessões residentes no Vulkan" | `OPEN/STEP/CLOSE` no fabric | trabalho grande e só mensurável com GPU real (aqui: lavapipe, uma CPU fingindo GPU) | **não feito** (continua P0); o `STEP` ganhou a realimentação de estado que o fabric também vai precisar |
| "cache KV circular" | buffer circular | numa memória contígua, o anel quebra a ordem canônica (a janela tem de ser lida em ordem de posição) | **anel de páginas**: a tabela de blocos mapeia a página lógica `j` em `mine[j mod R]`, a ordem lógica fica intacta; `R = ⌈(w + T − 1)/página⌉` **provado justo** (um a menos muda os bits — teste de controle); 7,9× mais sequências a 32 k / janela 4 k |
| "kernel NTT verificado" | NTT no worker | a NTT exata já existe (BabyBear/Goldilocks/BN254 na BEAM); levá-la ao worker exige multiplicação inteira alta (`mulhi` 64 bits) nos cinco backends — um primitivo novo em cada codificador | **não feito**; motivo e caminho no TODO |
| "divisão corretamente arredondada (0 ULP)" | IEEE | muda bits de programas existentes: exige versão nova da semântica (os certificados registram qual) | **feito**: semântica versão 2; Markstein + resíduo exato de Dekker, sem FMA; 0 erros em 400 mil pares e em 50 M exaustivos; custo ≈ 2,2× o `a·rcp(b)` |
| "lema de Higham em Lean" | provar | — | **feito**, núcleo só (`propext`, `Quot.sound`) |
| "torre de texto do CLIP + BPE `</w>`" | — | — | **feita**: tokenizador = o do CLIP em 20/20 linhas; torre 3,7·10⁻⁷ contra o `transformers`. A busca **semântica** de fotos pede pesos CLIP treinados (não embarcados): fica no TODO com esse motivo |
| "Whisper" | adaptador | sem pesos treinados aqui, paridade não é qualidade de transcrição; o front-end log-mel é outra peça | **feito**: encoder (convoluções por `Vapor.Spatial`) + decoder com atenção cruzada, K/V cruzados calculados **uma vez** por áudio; 4,8·10⁻⁷ / 4,2·10⁻⁷ e decodificação gulosa idêntica ao `transformers`; controles (outro áudio, quadros invertidos) |
| "Mamba/SSM" | adaptador | o *scan* paralelo (associativo) reordena somas: os bits dependeriam do paralelismo; e o motor paginado não serve estado | **feito**: um passo recorrente (`t = 1`) é a definição — *prefill* e decode são as mesmas instruções; o estado fica no worker (o `STEP` realimenta `s ← s_next`); `log` e `softplus` canônicos novos; 3,4·10⁻⁷ e gulosa idêntica ao `transformers`; custo por token constante (medido) |
| "*streaming* zero-cópia Plug/Bandit" | — | sem Plug/Bandit no ambiente (`deps: []`); o Bandit não tem HTTP/3 | **não feito** (não testável aqui); o servidor próprio já transmite SSE |
| "*store* Ecto oficial" | — | sem Ecto/Postgres aqui; um adaptador não testado é ponta solta disfarçada | **não feito** |
| "atestados ancorados em logs de transparência" | blockchain? | uma blockchain acrescenta consenso pago e não muda a garantia: o que importa é **detectar bifurcação** (visões divergentes do log), e isso testemunhas fazem (RFC 9162, C2SP) | **feito**: `Vapor.Tlog` (Merkle RFC 9162, provas de inclusão e consistência, *checkpoints* C2SP *signed-note*, co-assinaturas de testemunha), recibos de busca ancorados, verificador **no navegador** (WebCrypto Ed25519, chave fixada no primeiro uso); 196 sondas do transparency-dev |

| cláusula do 2º anexo | leitura literal | problema | refinamento adotado |
|---|---|---|---|
| "`Conv2d`/`Conv3d`" | kernels novos | cinco backends × verificação × política por kernel | **nenhum kernel novo**: im2col = `gather_row` + `sel` (padding exatamente `+0`) + `reshape` + `linear`; mesmos bits em todo substrato; ≤ 4·10⁻⁷ contra o torch |
| "VAE contínuo" | adaptador | sem pesos treinados, nenhuma afirmação de qualidade de imagem é possível | `AutoencoderKL` (decoder) do diffusers, 8,7·10⁻⁷ contra o diffusers |
| "DiT" | adaptador | idem | `DiTTransformer2DModel` (adaLN-Zero), 3,1·10⁻⁷; tabela de posições **bit a bit**; *timestep embedding* 8,3·10⁻⁷ |
| "FlashAttention em Vulkan/CUDA" | kernel com softmax online em blocos | o softmax online reduz máximo e soma em blocos — **outra ordem canônica**, outros bits que os demais substratos; CUDA: sem GPU NVIDIA aqui | **não feito**; só faria sentido como política `:fast` declarada; a janela e o paginado já cortam a leitura onde importa |
| "fusão multimodal latente por atenção cruzada" | operador novo | — | é **a atenção existente** (as K/V de outra corrente, todo horizonte na última linha): ≤ 10⁻⁶ contra `nn.MultiheadAttention`; é o que o Whisper usa |
| "condicionamento espacial (ControlNet)" | adaptador | ControlNet = cópia do encoder da U-Net + convoluções-zero; ainda não há adaptador de U-Net | **não feito**; os blocos (conv, GroupNorm, atenção por pixel, upsampling) existem e estão conferidos |
| "KV de longo prazo para vídeo" | memória longa | "longo prazo" sem critério vira cache infinito | três respostas medidas, cada uma com a sua troca: **janela + anel** (memória `O(w)`), **latente** (85×), **estado fixo** (SSM). Um modelo de mundo de vídeo: fora do alcance sem dados e treino |
| "paralelismo de tensor entre nós" | Megatron | linha-paralelo + all-reduce **muda os bits** (medido: 13 284 de 16 384) | coluna-paralelo + **all-gather**: exato; cada fragmento é um worker (porta), na mesma máquina ou em outra |
| "árvores de decodificação especulativa" | árvore de rascunhos | árvore sem garantia vira aproximação | ramos em *slots* sobre as **páginas compartilhadas** do contexto (sem cópia; a página parcial é recomputada por cada ramo); a saída **é** a gulosa do alvo (testado em cinco formas de árvore); rascunho por **busca no prompt** com cópia sobreposta: 6 tokens/passo onde a saída segue o contexto, 1,2 com pesos aleatórios — medido |

### O que o escrutínio achou no caminho (sem ser pedido)

- **DeepSeek com `n_group = 1`** (o DeepSeek-V2-Lite) quebrava a construção
  do programa: a limitação por grupos não tinha termos. Agora é a identidade
  (mesmos bits que "todos os grupos mantidos"), com teste.
- **Opções ignoradas nos testes do motor**: o *helper* juntava os padrões
  *antes* das opções (`++`), então `sequences: 1`, `step_tokens: 5` etc.
  nunca valeram — os testes de invariância passavam sem testar o que diziam.
  Corrigido (`Keyword.merge`) e re-verificado.
- **Um pedido maior que o pool inteiro** esperava na fila para sempre; agora
  é recusado (`{:kv_pages, need, pool}`).
- **`softplus` como um microprograma só não cabia nos registradores** de um
  kernel: virou composição de nós canônicos (mesmos bits que o microprograma).
- **`layer_types` ignorado fora do Gemma 3**: um Mistral híbrido (camadas
  globais entre as deslizantes) seria lido como todo deslizante — calculado
  errado e com páginas recicladas que as camadas globais ainda leriam. Agora
  a chave é lida ou recusada, nunca pulada.
- **O motor lia a configuração da família** para decidir o anel — o teste que
  audita a tabela de átomos do núcleo pegou. Agora pergunta ao adaptador
  (`Vapor.Lock.ring_window/2`, *callback* opcional); o núcleo continua sem
  conhecer família nenhuma.

### O que este documento não afirma

- **Nenhuma qualidade de imagem, vídeo ou transcrição.** VAE, DiT, Whisper e
  Mamba estão conferidos **contra as implementações de referência** com pesos
  aleatórios; qualidade exige pesos treinados, que não estão neste ambiente.
- **Sora e Midjourney não foram alcançados.** O que faltava *no runtime*
  (convolução, VAE, DiT, atenção cruzada, memória longa, paralelismo exato)
  agora existe e é verificável; o que separa daí um modelo de vídeo são
  dados, treino e computação — fora do escopo de um compilador.
- **As medições são desta VM** (2 vCPUs, sem GPU): [bench/FRONTIER.md](bench/FRONTIER.md).

### Como contestar

```sh
mix test test/vapor/sparse_experts_test.exs test/vapor/latent_attention_test.exs \
         test/vapor/sliding_window_test.exs test/vapor/engine_test.exs test/vapor/division_test.exs \
         test/vapor/tlog_test.exs test/vapor/spatial_test.exs test/vapor/mamba_test.exs \
         test/vapor/speculative_tree_test.exs test/vapor/shard_test.exs
mix test --include torch test/vapor/mamba_hf_test.exs test/vapor/whisper_hf_test.exs test/vapor/lock_hf_test.exs
mix vapor.bench --frontier             # docs/bench/FRONTIER.md
mix vapor.quality                      # §5b de docs/bench/QUALITY.md: cada recurso contra um controle
```

## 10. Rodada 0.7: o mesmo pedido, pela terceira vez — e o escaneado de escritório

> Pedido (2026-10-03): "Forneça um zip refinado ao final → esta própria
> diretriz está sujeita a refinamentos e escrutínio → resultado deve ser um
> artefato que resolva as dores reais da indústria e academia com inovação
> real, pensamento lateral e primeiros princípios (ataque as limitações, todo
> e complete o que for pertinente e que está sujeito a refinamento + testes de
> qualidade para garantir que as respostas não sejam apenas ruído + UI/UX
> original e elegante)".

### Escrutínio

| cláusula | leitura literal | problema | refinamento adotado |
|---|---|---|---|
| o pedido repete o da 0.5 | refazer a 0.5 | repetir as mesmas entregas é ruído; o que mudou foram as limitações declaradas pela 0.6 | atacar **as limitações declaradas no README e no TODO da 0.6** que esta máquina consegue verificar; dizer quais ficaram e por quê |
| "ataque as limitações" | todas | metade exige hardware que não está aqui (GPU real, Apple Silicon, RVV em silício) ou pesos treinados (Whisper/CLIP/VAE com qualidade) — "fazer" sem poder executar é fachada | escolhidas por **verificabilidade aqui** e por **dor**: o PDF escaneado de escritório (CCITT, colunas, CTC sem língua), a saída estruturada com campos inválidos, a fusão que não cabe na memória |
| "inovação real" | algo novo | novidade de vitrine não é critério | uma propriedade que antes falhava e agora é testada **com controle** (a forma ingênua reprovada no mesmo teste) |
| "testes de qualidade … não ruído" | medir a saída | um modelo de língua é o exemplo clássico de **ruído plausível**: melhora a métrica média e inventa texto onde não há língua | toda melhora do leitor é medida junto do seu modo de falha: cadeias aleatórias (o modelo tem de se abster), códigos e valores (não pode piorar), corpus embaralhado (o ganho tem de ser da língua) |
| "UI/UX original e elegante" | redesenhar | o console já tem identidade (a eclusa: cada resultado num nível de água); trocar a pele seria trabalho sem dor resolvida | a interface ganha **o que a evidência precisa mostrar agora**: a ordem de leitura (blocos numerados e o fio de leitura) e *quem decidiu cada letra* (quadros ou modelo de língua), com a leitura alternativa a um clique |
| "todo e complete o que for pertinente" | tudo | — | fechar o que coube, com teste; o resto fica no TODO com o motivo (§ abaixo) |

### Dores atacadas e o que as prova

| dor | antes (0.6) | agora (0.7) | prova (teste / suíte §5c) |
|---|---|---|---|
| PDF de *scanner* P&B (CCITT) | "an image in CCITTFaxDecode (not decoded here)" | Group 3 1-D/2-D e Group 4 decodificados, LZW e RunLength | 42 fluxos = libtiff bit a bit, LZW/PackBits = libtiff; `ccitt_test.exs` |
| documento de 2–3 colunas | linhas atravessando colunas: CER 73 % numa página que, em ordem, dá 1,3 % | XY-cut com calha antes de vão, limiares da própria região | 8 páginas escaneadas: CER 1,3 % (sem ordem: 53 %; Tesseract: 1,6 %); `reading_test.exs` |
| "c1áusula", "rão", "agerdar" | CTC guloso | feixe CTC + modelo de língua de caracteres, que **se abstém** onde não há língua e só escolhe entre o que os quadros acham plausível | linhas retidas CER 6,5 % → 4,4 %; cadeias aleatórias: 0 linhas mudadas (sem a guarda: 27); corpus embaralhado: nenhum ganho |
| saída JSON válida com campo inválido | `pattern`/`format` recusados | regex ECMA-262 e formatos → autômato de bytes | mesmo veredito que o `re` do Python e a biblioteca padrão em 7 240 cadeias; `regex_test.exs` |
| fundir modelos maiores que a RAM | três cópias em memória | `Merge.stream/3`: um tensor por vez | arquivos byte a byte iguais à fusão em memória; 4 MB × 83 MB de pico; `merge_stream_test.exs` |

### O que o escrutínio achou no caminho (sem ser pedido)

- **Máscaras de estêncil (`ImageMask`) lidas invertidas** no leitor de PDF —
  sem teste até aqui. Corrigido e testado (Flate e CCITT).
- **Acentos descartados** em linhas sem letras altas: o achador de linhas
  aceitava componentes a até 3 px do miolo da linha; o til e a cedilha de
  "mesma execução não anunciam a mesma ação" ficavam de fora e o leitor via
  "mesmã eeeução rão anuneiãm". Achado **olhando a nova interface**. Agora o
  alcance é relativo à altura do texto; o CER guloso das linhas retidas caiu
  de 6,8 % para 6,5 %, e o das cadeias aleatórias de 16,0 % para 10,7 %.
- **Três modos de falha do modelo de língua**, cada um virou regra testada:
  reescrever cadeias aleatórias (guarda em bits/caractere), apagar letras
  lidas com certeza (só candidatos plausíveis, o *blank* inclusive), e trazer
  o domínio do corpus — crases do Markdown inventadas em páginas impressas
  (o modelo é contado sobre o texto como impresso; o custo no conjunto
  renderizado a partir do Markdown está dito em [OCR.md §3b](OCR.md)).
- **O título cortado ao meio** pela primeira versão do XY-cut (o espaço entre
  palavras de um título grande parecia uma calha): os limiares passaram a ser
  da própria região.

### O que este documento não afirma

- **Nenhum OCR de tabelas, manuscrito ou JBIG2.** O XY-cut lê uma tabela
  coluna a coluna; JBIG2 continua recusado com aviso.
- **O modelo de língua não é "inteligência"**: 5-gramas de 27 mil palavras.
  Ele erra (há um "retomam" → "retoman" numa página de teste, visível na
  interface). O que se afirma é a medida, com os controles.
- **O Tesseract continua melhor** nas páginas em inglês com fontes comuns; o
  vapor é melhor onde o Tesseract daqui não tem o idioma (português) e na
  foto de luz desigual.
- Os itens de hardware (GPU, Apple Silicon, RVV em silício) e de pesos
  treinados continuam no [TODO](TODO.md), com o motivo.

### Como contestar

```sh
mix test test/vapor/ccitt_test.exs test/vapor/reading_test.exs test/vapor/regex_test.exs test/vapor/merge_stream_test.exs
python3 test/python/scan_pages.py /tmp/scans --tesseract     # regenera as páginas escaneadas e as leituras do Tesseract
python3 test/python/ocr_render.py /tmp/sets val 160 101      # o conjunto em que os pesos do modelo de língua foram escolhidos
mix vapor.quality                                            # §5c de docs/bench/QUALITY.md
```

## 11. Rodada 0.8: o roteiro "Vapor 1.0" — o que entrou, o que foi recusado, e por quê

> Pedido (2026-10-03): o mesmo texto da 0.7 ("Forneça um zip refinado ao
> final → esta própria diretriz está sujeita a refinamentos e escrutínio →
> … ataque as limitações, todo e complete o que for pertinente (quais
> features interessantes estão faltantes?) … + testes de qualidade para
> garantir que as respostas não sejam apenas ruído + UI/UX original e
> elegante"), com um anexo: um roteiro para "um Vapor 1.0 definitivo" em
> cinco áreas e dezessete itens.

### Escrutínio do pedido

| cláusula | leitura literal | problema | refinamento adotado |
|---|---|---|---|
| o anexo como especificação | fazer os 17 itens | metade exige o que esta máquina não tem (Apple Silicon, GPU real, Postgres, um dispositivo Nerves) ou pesos treinados que não existem aqui (OCR árabe/CJK, fórmulas, texto→vídeo); "fazer" sem poder executar nem medir é **fachada**, que o projeto proíbe | cada item foi lido como **dor + critério verificável aqui**; entrou o que pode ser conferido com controle; o resto foi recusado **pelo nome**, com o que o fecharia (tabela abaixo) |
| "à prova de balas", "100 % dos PDFs", "plataforma definitiva" | metas absolutas | infalsificáveis; "100 % dos PDFs escaneados" é falso mesmo com JBIG2 (sobram JBIG2 Huffman, meio-tom, JPX) | trocadas por medidas: fluxos conferidos bit a bit contra a referência, e a lista do que continua recusado |
| "quais features interessantes estão faltantes?" | sugerir mais | uma lista de desejos é ruído | só entram faltas que **o escrutínio achou** ao executar (§ "achados") — cada uma virou teste |
| "testes de qualidade … não ruído" | medir | um decodificador de imagem, um adaptador de modelo, um dossiê: cada um tem um jeito de "passar" estando errado | cada entrega nova tem o controle que a forma errada produziria (§5d da suíte): o modelo genérico declarado errado no JBIG2, a norma do outro lado no Mamba-2, o estado zerado na GPU, a fatia trocada no cluster, o byte trocado no dossiê |
| "UI/UX original e elegante" | redesenhar | a identidade (a eclusa: cada resultado no seu nível de água) já existe | o console ganha o que a evidência nova precisa mostrar: a tabela desenhada das suas células e ligada à página; a **trama** de evidência × dispositivo, que mostra a lacuna em vez de escondê-la |

### Os dezessete itens

| item do anexo | decisão | o que prova (teste / suíte §5d / bench) | escrutínio |
|---|---|---|---|
| Sessões residentes na GPU (P0) | **feito** | `OPEN/STEP/CLOSE` no fabric; memória direta ou *staging*; *command buffers* gravados reaproveitados pela chave dos bytes exatos; 72,4 → 12,4 ms/token, 1 MB → 8 kB por token; motor inteiro na GPU com os tokens da CPU; queda do driver → `:session_lost`, nunca a BEAM. `gpu_session_test.exs`, bench §1 | o anexo pedia *timeline semaphores*: desnecessário — um *fence* por passo e o cache de gravações bastam; o que importa é não mover o cache KV. Medido no lavapipe (CPU): a vazão não diz nada sobre GPU real |
| Apple Silicon / Metal | **recusado** | — | sem Mac aqui; MoltenVK resolveria o Vulkan mas não o *sandbox* (seccomp é Linux). Fica no TODO com o desenho (MAP_JIT, `posix_spawn` sem direitos) |
| GEMV 4 bits esparso (`qgemv_masked`) | **feito** | kernel `gemv_sb4_masked` em x86/AVX-512/NEON/RVV/SPIR-V; bits = denso; 1,6–2,6×; 15,0 M → 5,3 M instruções. `sparse_sb4_test.exs`, bench §2 | "destravar DeepSeek-V3" é afirmação de escala que um modelo reduzido não prova; o que se afirma é a razão medida |
| Cholesky do RegMean no worker | **adiado** | — | sem dor medida nesta rodada (os modelos de teste são pequenos); TODO |
| OCR universal (árabe, CJK, cursivo, cirílico) | **recusado** | — | o limite não é a geometria nem o CTC: é um leitor **treinado** nesses sistemas de escrita, com dados que não existem aqui. Faixa de 64 px e 10 mil classes sem treino = ruído com cara de saída — o contrário do que a suíte existe para impedir |
| Tabelas (TSR) | **feito** | grades com réguas e células mescladas, tabelas só de filetes, colunas tipadas e formas; F1 de estrutura 1,000 (0,7: 0,343), CER por célula 5,5 % (livre: 11,7 %); Markdown, HTML, CSV, JSON pela API. `table_test.exs`, [OCR.md §3e](OCR.md) | "esquemas Ecto" recusado (o núcleo não depende do Ecto); o Tesseract lê melhor as células mesmo assim (2,3 % com caixas perfeitas) — dito |
| JBIG2 | **feito, aritmético** | MQ, genérico, MMR, refinamento, símbolos, texto, PDF com globais; 43 fluxos = jbig2dec bit a bit (controle: 19/43). `jbig2_test.exs`, [OCR.md §3f](OCR.md) | Huffman e meio-tom recusados pelo nome: nenhum codificador daqui os emite, decodificar sem conferir seria fachada |
| OCR de fórmulas → LaTeX | **recusado** | — | a parte "gramática restrita" existe (a decodificação dentro de uma forma, §3e, é um autômato CTC); falta um leitor de símbolos matemáticos treinado |
| Texto → vídeo | **recusado** | — | os blocos (conv3d, VAE causal, DiT) estão conferidos; sem pesos treinados de vídeo a saída é ruído, e a diretriz pede o contrário |
| Mamba-2 | **feito** | adaptador pela eclusa; = `transformers` (7,8·10⁻⁷), gulosa idêntica, nativo = oráculo, GPU = CPU. `mamba2_hf_test.exs`, `gpu_session_test.exs`, suíte §5d | achou duas divergências do `transformers` com o código de treino (abaixo) |
| Híbridos (Jamba, Zamba, Bamba) | **adiado** | — | exigem cache KV **e** estado por sequência no motor; recusados com quase-acerto pela eclusa |
| RoPE 2D/3D | **adiado** | — | nenhum modelo admitido precisa ainda; quando um precisar (Qwen2-VL M-RoPE), é conferido contra o `transformers` como os outros |
| Lean: regras de reescrita | **feito** | modelo IEEE-754 a partir dos padrões de bits (núcleo do Lean, sem Mathlib): `x·1`, `1·x`, `x+(−0)`, `(−0)+x`, `x−(+0)` — **o único resultado corretamente arredondado é `x`**; `x+(+0)→x` **refutado**; `units` extraído e conferido contra `Vapor.F32`. `proofs/Vapor/Binary32.lean`, `rewrite_soundness_test.exs` | o anexo pedia "incluindo NaNs": **impossível** bit a bit entre substratos (x86 aquieta o sNaN, RISC-V devolve o NaN canônico) — e a documentação da 0.7 afirmava isso; corrigido e testado |
| Lean: emulador RVV | **recusado** | — | exigiria a especificação formal da RVV 1.0 em Lean (o modelo Sail não está em Lean); o emulador continua conferido contra o QEMU |
| Lean: Wilkinson com janela | **já coberto** | `withinEnvelope_mono` (Lean, 0.5): o envelope de uma soma de `w` termos é dominado pelo de `n ≥ w`; a janela exata é uma soma mais curta na mesma ordem canônica (`sliding_window_test.exs`) | não há teorema novo a provar — dizer o contrário seria inflar |
| Ecto/Postgres para o `Agent.Store` | **recusado aqui** | — | sem Postgres neste ambiente; um adaptador não testado contra o banco é exatamente o que o projeto não publica |
| Paralelismo de tensor entre nós BEAM | **feito** | `Vapor.Shard.Cluster` com nós `:peer` reais: bits de um worker em 1, 2, 3 nós; nó perdido → fragmentos recolocados, mesmos bits; réplicas comparadas bit a bit pegam um nó que corrompe um bit; fragmento com SHA-256 errado recusado. `shard_cluster_test.exs`, suíte §5d | a atenção fragmentada por cabeças entre nós fica no TODO |
| Kit EU AI Act / ISO 42001 | **feito** | `mix vapor.audit export/verify/demo`: dossiê assinado e co-assinado, raiz de Merkle, âncora no log, cada item conferido pelas suas regras; PDF com o dossiê anexado; HTML que se confere offline no navegador; painel *Dossiê* com a trama. `audit_dossier_test.exs`, [AUDITORIA.md](AUDITORIA.md) | o comando é `mix vapor.audit export` (não `vapor.audit.export`); **não é avaliação de conformidade** — o aviso vai em cada dossiê |
| AOT para Nerves | **recusado** | — | sem dispositivo; Cortex-M não tem as ISAs vetoriais que os emissores cobrem (seria um *backend* novo, MVE/Helium), e sem conferência no alvo não há certificado |

### O que o escrutínio achou no caminho (sem ser pedido)

- **O `transformers` e o código de treino do Mamba-2 discordam na norma com
  porta** quando há mais de um grupo (por grupo × largura inteira): o padrão
  aqui é o do treino, a outra é opção, e cada uma falha contra a referência
  da outra (erro 0,74/0,66) — a comparação discrimina.
- **O passo com cache do `transformers` (Mamba-2) pula o `time_step_limit`**
  que o seu *scan* aplica: o `generate()` dele mistura duas semânticas.
- **"x · 1 → x exato, inclusive NaN"** estava escrito em `Rewrite` desde a
  0.5: falso para sNaN em x86 e para qualquer NaN com payload em RISC-V. O
  modelo em Lean cobre os finitos e diz por que o NaN fica fora; um teste
  impede a frase de voltar.
- **A suíte de qualidade chamava `System.cmd("epmd")`** — o teste de auditoria
  de fonte (nada de *shell* no produto) pegou. Removido: sem epmd, a
  verificação do cluster não roda e o relatório diz isso.
- **Configurações antigas do `transformers` escrevem `Infinity`** (JSON
  inválido) no `time_step_limit`: o leitor de `config.json` aceita agora os
  não-finitos do Python **só ali** (`nonfinite: true`); o JSON estrito
  continua estrito.
- **jbig2dec e pdf.js discordam** nos contextos do TPGRON (modelo 1), e o
  jbig2dec usa a página inteira como referência de um refinamento sem
  deslocamento: seguimos o que se pode conferir (jbig2dec), documentado.
- **Olhando a interface nova**: o cabeçalho "T1" de uma tabela lido "TP1"
  sem confiança baixa — os passos tipados só valem no corpo; dito em
  [OCR.md §3e](OCR.md) em vez de escondido.

### O que este documento não afirma

- **GPU real**: tudo o que é GPU aqui roda no lavapipe; a igualdade dos bits
  e o protocolo estão provados, a velocidade numa GPU de verdade não.
- **Conformidade regulatória**: um dossiê autentica evidência; a
  suficiência jurídica não é dele.
- **OCR melhor que o Tesseract** nas células: não é (5,5 % × 2,3 %); a
  estrutura, que o Tesseract não dá, é que é exata nas 12 tabelas.
- **Mamba-2 com pesos reais**: conferido com pesos aleatórios em torno da
  inicialização do `transformers`, como os outros adaptadores; nenhum número
  de perplexidade de um checkpoint publicado foi medido aqui.

### Como contestar

```sh
mix test test/vapor/gpu_session_test.exs test/vapor/sparse_sb4_test.exs test/vapor/table_test.exs \
         test/vapor/jbig2_test.exs test/vapor/mamba2_hf_test.exs test/vapor/rewrite_soundness_test.exs \
         test/vapor/shard_cluster_test.exs test/vapor/audit_dossier_test.exs
cd proofs && lake build                                       # Binary32.lean entre as provas, sem sorry nem axiom
python3 test/python/hf_mamba2.py /tmp/m2 mamba2-g2 11         # a referência do Mamba-2, com e sem a norma por grupo
python3 test/python/table_render.py /tmp/val --seed 2028      # o conjunto de validação dos limiares das tabelas
mix vapor.audit demo && mix vapor.audit verify _build/audit-demo/dossier.vdossier
mix vapor.bench --round08                                     # docs/bench/ROUND08.md
mix vapor.quality                                             # §5d de docs/bench/QUALITY.md
```

## 12. Rodada 0.9: "algo similar ao ComfyUI, any-to-any, e tudo dos cursos da Hugging Face"

> Pedido 1 (2026-10-03, no meio da rodada 0.8): "pondere suporte para
> destilação adversarial e remoção de marca d'água e censura de modelos".
>
> Pedido 2 (2026-10-04): "minha intenção era completude e uma filosofia de
> neutralidade da tecnologia; nesse sentido, vamos atrás de viabilizar algo
> similar ao Comfy com any-to-any, áudio, vídeo e imagens e edição de tais +
> features inovadoras pertinentes que resolvam dores e utilidades na parte
> agêntica e que cubram tudo dos cursos Hugging Face (games, 3D, deep RL
> etc.) + quem sabe upscale com IA de qualquer fonte com inovação e
> qualidade" — com os links de doze cursos.

### O registro da decisão sobre o pedido 1

Três coisas foram recusadas, pelo nome:

- uma ferramenta para **remover marcas d'água** de saídas de modelos;
- uma ferramenta para **remover as recusas** ("censura") de modelos;
- a **clonagem de modelos fechados pelas suas APIs** (destilação adversarial
  no sentido de extrair um modelo de terceiros).

O motivo não é o tema. O vapor treina, destila, funde e roda pesos do
usuário sem filtro de conteúdo próprio, e isso continua assim. O motivo é a
função: o uso principal dessas três ferramentas é desfazer o controle que um
terceiro pôs no que é dele, seja a proveniência de um conteúdo, a política
de um modelo ou os termos de uma API. O projeto inteiro existe para o
contrário. Recibos, o log de transparência, dossiês de auditoria e agora a
raiz de Merkle de cada execução do estúdio são máquinas de **proveniência**.
Uma ferramenta de apagar proveniência dentro delas contradiria o que elas
afirmam.

"Neutralidade da tecnologia" foi levada a sério como princípio de projeto e
lida assim:

- as **ferramentas gerais** não têm opinião: o estúdio roda qualquer peso,
  qualquer prompt, qualquer grafo do usuário;
- as **ferramentas de propósito único**, desenhadas para derrotar uma
  salvaguarda alheia, não são neutras: são essa função.

O que foi oferecido no lugar, e onde está:

| alternativa | estado |
|---|---|
| destilação a partir de **pesos locais** | já existe: `Vapor.Train` (LoRA por destilação KL como programa recorrente) |
| destilação **robusta a perturbações adversariais** (o sentido defensivo do termo) | TODO, com o critério: a perda do aluno sob perturbação limitada, contra o aluno comum |
| **medir o excesso de recusa** de um modelo (recusas em pedidos benignos) | TODO: um portão calibrado como os de `Vapor.Quality`, com controles |

### Escrutínio do pedido 2

| cláusula | leitura literal | problema | refinamento adotado |
|---|---|---|---|
| "algo similar ao Comfy" | clonar a interface | o valor do ComfyUI é o ecossistema: milhares de nós e modelos. Uma cópia da tela sem isso seria fachada. Reproduzir o código dele também não é o caminho | um grafo de nós tipados com o que o ComfyUI **não** tem: cache exato por conteúdo (chaves de Merkle), recibos por saída, raiz por execução, `verify` sem cache, recusa inteira de grafos mal tipados. A **importação** de workflows do ComfyUI (formato de API) traduz o subconjunto cuja semântica está fixada e avisa cada tradução; um nó sem tradução recusa o workflow inteiro. As traduções foram escritas pela semântica documentada dos nós, sem copiar código |
| "any-to-any, áudio, vídeo e imagens e edição" | todo formato, toda edição | "todo" é infalsificável; H.264/MP4 são enormes, e ffmpeg por *shell* viola a regra do produto, que não executa processos alheios | 64 nós em imagem, som, vídeo, visão, 3D, RL e difusão, cada família conferida contra uma referência (torch, Pillow, libjpeg, ffmpeg, gymnasium, trimesh, diffusers); codecs próprios (GIF, Y4M, MJPEG-AVI); H.264/MP4/WebM recusados pelo nome |
| "features inovadoras … dores na parte agêntica" | mais funções | uma lista de desejos é ruído | as dores foram nomeadas uma a uma e cada uma ganhou uma ferramenta MCP com teste: o agente que adivinha parâmetros (catálogo tipado), o erro descoberto tarde (validar antes), a nova tentativa que recalcula tudo (o cache vive entre chamadas: 2 de 5 nós), o resultado que não se pode conferir (raiz + verify), a citação inventada (busca com prova de Merkle) |
| "cubram tudo dos cursos" | doze cursos completos | infalsificável, e parte exige simuladores, hardware ou pesos treinados que esta máquina não tem | um **mapa de cobertura** ([ESTUDIO.md §6](ESTUDIO.md)): por curso, o que existe conferido e o que falta, com o motivo |
| "upscale com IA de qualquer fonte com inovação e qualidade" | um super-resolvedor que melhore tudo | "qualidade" sem medida é ruído, e um GAN que inventa textura passa nos olhos e falha na fidelidade | um ampliador com **garantia**: D(y) = x por construção, ou seja, reduzir o resultado devolve a entrada. Treinado aqui e reprodutível. Medido em imagens retidas contra a linha de base com a mesma garantia: +1,5 a +5,8 dB em texto e no fantoma, −0,2 a +0,1 dB em fotografias, e −1,4 dB na roda de cores, um gradiente suave em que os dois passam de 53 dB (dito, não escondido). "Qualquer fonte" = imagem ou vídeo de qualquer codec lido aqui, colorido ou cinza, ×2 ou ×4 |

### Entregas e o que as prova

| entrega | prova (teste / suíte §5e) | controle |
|---|---|---|
| Stable Diffusion: U-Net, encoder do VAE, três *schedulers*, pipelines txt2img/img2img/inpainting | = diffusers a ~10⁻⁶ em todos os modos, no checkpoint minúsculo incluído. `diffusion_pipeline_test.exs` | DPM++ contra a referência DDIM: 0,096 |
| O estúdio | duas execuções sem cache = uma raiz; editar um parâmetro recalcula 3 de 6 nós | outra semente muda a raiz; sem cache, 6 de 6 |
| A importação do ComfyUI | o workflow txt2img roda e dá **os mesmos bits** que o pipeline | um nó sem tradução recusa o workflow, nomeado |
| O ampliador consistente | +2,76 dB (pior caso em texto) sobre Lanczos + projeção; \|D(y) − x\| = 1,1·10⁻¹⁶ | Lanczos: 0,077 |
| RL | Q-learning = iteração de valor (74,7 %); CartPole 472,9/500 | sempre à esquerda: 0 %; sem treino: 18 |
| 3D | esfera fechada, volume a 0,4 % | uma face a menos: não fechada |
| Som | 69,6 dB de SNR na reamostragem | decimação desalinhada: 17,7 dB |
| Servidor MCP | cliente oficial do SDK; a mesma execução de novo é toda cache; verify aceita a raiz | raiz errada recusada |
| Console: *Estúdio* | tela de nós; seis modelos de partida; prévias; selo e verificação. `console_test.exs` | — |

### O que o escrutínio achou no caminho (sem ser pedido)

- **Cada *scheduler* do diffusers espaça os *timesteps* do seu jeito.** No
  `leading`, o DPM-Solver divide por `passos + 1`; no `linspace`, o Euler
  mantém *timesteps* fracionários e interpola σ; o DDIM usa `passos` pontos,
  não `passos + 1`. Uma implementação única erra os latentes em até 0,19.
  Agora cada um reproduz o seu a 6·10⁻⁷.
- **O img2img e o inpainting do diffusers amostram o latente do VAE**, uma
  aleatoriedade escondida que a semente do usuário não controla. O vapor usa
  a média; para comparar, o diffusers foi forçado à média.
- **Checkpoints de SD trazem só os arquivos lentos do tokenizador**
  (vocab.json + merges.txt, sem tokenizer.json). O vapor monta o tokenizador
  que o `transformers` escreveria; o teste confere os ids.
- **O digest de um vídeo longo enchia o *heap* de lixo.** Um episódio de
  CartPole de 160 passos levava 36 s, quase todo em coleta de lixo. O hash
  agora sai do formato externo de termos, fora do *heap*: 2,8 s. A definição
  do digest de imagem mudou e está documentada em `Studio.Value`.
- **Uma execução toda em cache levava 19,6 s no console.** As prévias eram
  refeitas e os digests recalculados a cada vez. As prévias agora são
  memorizadas pelo digest, e o cache guarda os digests: 1 s.
- **O KSampler do ComfyUI não é o amostrador do diffusers.** As sigmas são
  outras, e os pixels diferem. Cada tradução diz isso; nada afirma igualdade
  com o ComfyUI.
- **Focar um nó rolava a tela do grafo** (`overflow: hidden` rola mesmo
  assim) e descolava os fios. A rolagem agora vira deslocamento da tela, e o
  nó focado continua à vista.
- **A projeção de consistência sozinha melhora o Lanczos** em 0,2–1 dB. Ela
  é a linha de base justa, e é contra ela que o ampliador é medido.

### O que este documento não afirma

- **SD com pesos reais**: a paridade é com pesos aleatórios. Nem a
  qualidade de imagem de um checkpoint publicado nem a velocidade de um SD
  512×512 na CPU foram medidas aqui.
- **Igualdade com o ComfyUI**: nem nos pixels nem na cobertura de nós; a
  importação é de um subconjunto e diz qual.
- **O ampliador melhor em fotografias**: ele empata, e perde 1,4 dB num
  gradiente suave. Também não é um GAN, e não inventa detalhe.
- **Cobertura completa dos cursos**: o mapa diz o que falta (PPO/DQN,
  LeRobot de verdade, NeRF/splatting, ControlNet, SDXL, TTS…).

### Como contestar

```sh
mix test test/vapor/diffusion_pipeline_test.exs test/vapor/studio_test.exs test/vapor/studio_media_test.exs \
         test/vapor/upscale_test.exs test/vapor/rl_test.exs test/vapor/geom_test.exs \
         test/vapor/mcp_server_test.exs test/vapor/console_test.exs
python3 test/python/diffusers_pipeline.py /tmp/sd        # o checkpoint minúsculo e as referências do diffusers
python3 test/python/diffusers_scheduler.py dpmpp_2m leading 10
mix vapor.upscale eval DIR                               # DIR de test/python/upscale_data.py
mix vapor.rl eval
mix vapor.quality                                        # §5e de docs/bench/QUALITY.md
```

## 13. Rodada 0.10: substratos, treino, OCR de outras escritas — e, no meio, física, redes e contexto sem fim

> Pedido 1 (2026-10-04): "resultado deve ser um artefato que resolva as
> dores reais da indústria e da academia com inovação real, pensamento
> lateral e primeiros princípios (ataque as limitações, TODO, e complete o
> que for pertinente) + testes de qualidade para garantir que as respostas
> não sejam apenas ruído + UI/UX original e elegante + considere: suporte
> Metal (Apple) + Tenstorrent + orquestração de cluster vapor + OCR cursivo,
> árabe, CJK e de figuras + pipeline de treinamento completo, com HPC". E:
> "esta própria diretriz está sujeita a refinamentos e escrutínio".
>
> Pedido 2 (no meio da rodada, com um anexo sobre "inverter o RoPE" para
> contexto infinito): "incorpore a ideia + cirílico + manuscrito cirílico,
> árabe e CJK e LaTeX + suporte FreeBSD + pondere sobre motor de física no
> vapor (no sentido de *reinforcement learning* e *digital twins*) + pondere
> também acerca de aplicações de redes complexas".

### Escrutínio do pedido

| cláusula | leitura literal | problema | refinamento adotado |
|---|---|---|---|
| "suporte Metal" | um *backend* Metal | sem um Mac, um *backend* não executado seria fachada | tradutor MSL da mesma biblioteca de núcleos; daemon Metal compilado; **o mesmo texto MSL executado** por um *shim* de cabeçalhos com clang, em três "dispositivos" (conforme, que contrai, que zera subnormais); e uma porta que admite qualquer dispositivo **por medida** ([SUBSTRATOS.md](SUBSTRATOS.md)) |
| "Tenstorrent" | um *backend* Tenstorrent | sem a placa; o compilador deles fala StableHLO | **exportação StableHLO** (conferida no XLA: Llama/Qwen2/Mistral a 10⁻⁶) e um **kit de admissão portátil** que roda as sondas no dispositivo pela PJRT e volta para ser julgado e assinado |
| "orquestração de cluster" | um escalonador | escalonadores existem; o que o determinismo permite e eles não têm é o ponto | cache por conteúdo em todo o cluster, **auditoria por execução redundante** com amostra que não pode ser escolhida depois, quarentena com evidência e readmissão por medida, *failover* e *hedging* sem mudar um bit |
| "pipeline de treinamento completo, com HPC" | treinar um LLM grande | sem GPU e sem dias de máquina | um **pré-treino de verdade, pequeno e completo**: gradientes conferidos contra o PyTorch, paralelismo de dados cujos bits não dependem do número de workers, checkpoint/retomada exatos, exportação para o Hugging Face, um modelo embarcado com recibo que vence Witten–Bell em texto retido ([TREINO.md](TREINO.md)) |
| "OCR cursivo" | ler letra de mão | nenhum conjunto de letra de mão real cabe nesta máquina | substituto **dito como tal** (fontes manuscritas), **medido** — 64 % de CER em mãos nunca vistas — e, pela regra da suíte, **não embarcado**: o leitor recusa com a medida ([OCR.md §3j–§3k](OCR.md)) |
| "árabe, CJK" | leitores | — | árabe por CTC com bidi inverso exato; CJK sem rede treinada, por features direcionais e segmentação decidida pelo reconhecimento ([OCR.md §3g, §3j](OCR.md)) |
| "OCR de figuras" | achar imagens na página | achar é pouco; a dor é **os números presos em gráficos** | detecção de figuras com legenda, e **digitalização de gráficos que recusa** quando os rótulos não confirmam uma escala ([OCR.md §3h](OCR.md)) |
| "LaTeX" | OCR de fórmulas | — | fórmulas tipografadas → LaTeX, símbolos pela forma e estrutura pela geometria, medidas em tipos nunca vistos ([OCR.md §3i](OCR.md)) |
| "cirílico + manuscrito cirílico" | dois leitores | o manuscrito, como acima | leitor de cirílico impresso (2,8 % de CER em fontes nunca vistas); o "manuscrito" cirílico, árabe e japonês medido em fontes caligráficas e reportado, não prometido ([OCR.md §3k](OCR.md)) |
| "FreeBSD" | portar | sem FreeBSD aqui | o worker compila para FreeBSD (x86-64 e AArch64) com isolamento **Capsicum**; o teste confere o binário; **não executado** |
| "inverter o RoPE" (anexo) | contexto infinito | o anexo exagera o que já existia (§2 de [TREINO.md](TREINO.md)) | âncoras + janela com o RoPE aplicado no referencial do cache: memória constante, nenhuma distância fora do treino, **sem núcleo novo** |
| "motor de física" | um simulador | simuladores existem; a dor é que não se reproduzem | física cujo passo é um programa vapor: **os mesmos bits em todo substrato**, diferenciável, em lote; um gêmeo digital com livro verificável ([FISICA.md](FISICA.md)) |
| "redes complexas" | funções de grafo | bibliotecas existem; a dor é reprodutibilidade e afirmações sem nulo | geradores e estatísticas reprodutíveis, **cada afirmação contra um modelo nulo** ([REDES.md](REDES.md)) |
| "UI/UX original e elegante" | telas novas | — | painéis *Substratos*, *Treino*, *Física*, *Redes*; *Visão* com a escolha da escrita, linhas RTL, figuras com os dados do gráfico, LaTeX; em inglês e português, claro e escuro, na mesma linguagem visual (a eclusa e o nível d'água) |

### Entregas e o que as prova

| entrega | prova | controle |
|---|---|---|
| Eclusa de substratos | veredito e impressão numérica por sondas; registro assinado. `substrate_test.exs` | motor bf16 simulado: recusado, "8 significand bits" |
| Metal (via *shim*) | programas canônicos, SSM, GEMM, atenção, sessão Llama, o motor: = oráculo bit a bit. `metal_test.exs` | *shim* que contrai: envelope; FTZ: envelope |
| StableHLO + kit | Llama/Qwen2/Mistral no XLA a 10⁻⁶; XLA de CPU julgado: envelope. `stablehlo_test.exs` | operações sem equivalente exato: recusadas pelo nome |
| Cluster | um nó que vira um bit é pego e posto em quarentena. `cluster_test.exs` | sem auditoria, as respostas erradas passam |
| Pré-treino | gradientes = PyTorch (7,7·10⁻⁷); os mesmos bits com 1 ou 2 workers e com um worker morto. `train_lm_test.exs` | outro tamanho de bloco: outros bits |
| O modelo embarcado | 2,919 bits/byte em texto retido | Witten–Bell ordem 5: 3,415; texto embaralhado: muito pior |
| Contexto sem fim | 3,03 bits/byte 14× além do comprimento de treino, em 64 linhas; até encher: os bits do modelo causal. `streaming_test.exs` | posições crescendo: 5,81 |
| Física | período do pêndulo de primeira ordem; oráculo = nativo no caos; parâmetros do gêmeo recuperados; carro-pêndulo 200/200; alarme 18 passos após a falha. `physics_test.exs` | um ulp separa os mundos; medidas embaralhadas não recuperam nada; política nula ~50; sem falha, sem alarme |
| Redes | BA livre de escala, WS z ≈ 400, Louvain NMI 1, gigante = teoria, limiar epidêmico; = networkx. `graph_test.exs` | ER não é livre de escala e tem z ≈ 0; o nulo da partição |
| Figuras | 27/30 gráficos dentro da tolerância; 10/10 figuras com legenda (depois da correção abaixo). `figure_test.exs` | rótulos permutados: 12/12 recusados; nenhum gráfico lido grosseiramente errado |
| Fórmulas | 4,9 % de erro por *token* em Computer Modern e STIX. `math_test.exs` | leitura plana: > 3× |
| CJK | 8,9 % / 2,6 % / 11,3 % de CER (zh / ja / ko) em fontes nunca vistas | caracteres aleatórios: o modelo de língua não muda nada |
| Árabe | 19,2 % de CER em fontes nunca vistas; bidi inverso = python-bidi em 600/600 | leitor latino: 91 % |
| Cirílico | 2,8 % de CER em fontes nunca vistas, metade do vocabulário nunca vista. `scripts_test.exs` | leitor latino: 98 % |
| FreeBSD | o binário é FreeBSD e importa Capsicum e `_umtx_op`. `freebsd_test.exs` | nada do JIT da Apple (o seccomp, por chamadas cruas, não é visível no binário) |

### O que o escrutínio achou no caminho (sem ser pedido)

- **O envelope não cobria DAZ** (subnormais de entrada lidos como zero),
  achado pelo kit no XLA de CPU. Agora cobre, e continua apertado.
- **A árvore de termos de um transformer cresce exponencialmente** sem
  *let-bindings*: a diferenciação foi refeita sobre elas (10 min → 0,8 s).
- **O modelo de língua do CJK não ajuda o chinês numa semente nova** (no
  conjunto de desenvolvimento parecia levar 7,8 % a 2,5 %). Os números
  publicados são os da semente nova.
- **A velocidade como `(p − p_antes)/h` perde três dígitos em f32**: o
  período do pêndulo piorava com mais subpassos. Corrigido; agora converge
  em primeira ordem.
- **A troca dupla de arestas enviesada** fazia um grafo aleatório parecer
  estruturado contra o próprio nulo (z = 20). Corrigido.
- **Os pontos das letras árabes formavam linhas próprias** (uma linha em
  quatro saía partida). O localizador de linhas junta marcas baixas às
  vizinhas; o latim não muda.
- **Um dígito lido errado em todos os rótulos de um eixo** dá uma escala
  consistente e errada (1, 21, 41…). A regra "*ticks* são múltiplos do
  passo" pega isso; sem ela, o digitalizador mentiria com confiança.
- **Legendas perdidas no conjunto de teste** (uma acima da figura com
  texto logo abaixo; outra depois de um título de eixo fora da caixa): a
  primeira execução completa da suíte pegou. Corrigido — e dito que, para
  as legendas, o conjunto deixou de ser cego.
- **O console atribuía toda leitura ao leitor latino** e mostrava o modelo
  de língua latino ao lado de linhas árabes e cirílicas. O rodapé agora diz
  qual leitor leu e se houve modelo de língua.
- **Duas sessões no mesmo worker**: abrir a segunda derruba a primeira. Os
  laboratórios do console usam um worker cada.

### O que este documento não afirma

- **Metal, Tenstorrent e FreeBSD em hardware**: nenhum foi executado no
  hardware de verdade; cada um diz o que foi conferido no lugar.
- **Letra de mão**: o que se mediu foram fontes manuscritas; o leitor de
  cursiva não é embarcado.
- **Um LLM útil**: o modelo treinado é pequeno; o que ele prova é o
  *pipeline* (gradientes, determinismo, recibo, exportação), não a
  qualidade de um modelo grande.
- **O efeito das âncoras em modelos grandes**: neste modelo pequeno, quase
  nenhum; o que se provou é o mecanismo (memória constante, distâncias em
  faixa, os bits do modelo causal até o cache encher).
- **Física de corpos rígidos**: partículas e hastes, sem rotação, contato
  entre corpos ou atrito.

### Como contestar

```sh
mix test test/vapor/substrate_test.exs test/vapor/metal_test.exs test/vapor/stablehlo_test.exs \
         test/vapor/cluster_test.exs test/vapor/train_lm_test.exs test/vapor/streaming_test.exs \
         test/vapor/physics_test.exs test/vapor/graph_test.exs test/vapor/figure_test.exs \
         test/vapor/math_test.exs test/vapor/freebsd_test.exs test/vapor/scripts_test.exs
mix vapor.substrate kit /tmp/kit && python3 /tmp/kit/run_kit.py /tmp/kit --platform cpu && mix vapor.substrate judge /tmp/kit
mix vapor.train --steps 1000                     # refaz priv/lm: os digests do recibo
python3 test/python/chart_render.py /tmp/f charts 30 104      # os gráficos do teste
python3 test/python/math_render.py /tmp/m test 60 5           # as fórmulas do teste
mix vapor.quality                                 # §5f de docs/bench/QUALITY.md
```

## 14. Rodada 0.11: descoberta, matemática, ciência, autojogo — e dar vida a uma imagem

> Pedido 3 (2026-10-04, à noite): "esta diretriz está sujeita a refinamento
> e escrutínio: pondere sobre algo similar a AlphaProof para matemática
> (geometria, topologia, e além), algo similar para síntese e
> descobrimento de algoritmos com ou sem métodos formais e análise de
> complexidade, simulação física (clássica, relativística, quântica,
> tokamak, química — baterias, moléculas, materiais —, biologia —
> mutações, genômica, algo similar a AlphaFold) + reinforcement learning
> (manipulação de ambientes, algo similar a AlphaZero) + um pedido para
> puxar os limites: a partir de um esboço, modelos para desenhar com IA
> (fotorrealista, arquitetura, engenharia); animações offline ou laços
> infinitos com entropia a partir de desenhos; a partir de uma imagem
> complexa, um laço infinito interativo com NPC, 3D, profundidade,
> navegação livre, esqueletos e ações para entidades, efeitos (luz,
> gravidade e além) — dar vida a uma imagem como numa demo scene, com
> direção e ajustes por prompt, e salvamento e exportação dos resultados
> (isso vale para tudo!)".

### Escrutínio do pedido

O fio comum de AlphaProof, AlphaGeometry, AlphaDev, AlphaTensor,
AlphaFold e AlphaZero não é "uma IA grande": é **uma busca que propõe e
um verificador que decide**. Os modelos treinados dessas obras não cabem
nesta máquina (sem GPU, sem rede para pesos); os verificadores, e buscas
honestas guiadas por eles, cabem. Esse é o refinamento adotado em quase
toda cláusula.

| cláusula | leitura literal | problema | refinamento adotado |
|---|---|---|---|
| "AlphaProof para matemática" | RL sobre o Lean com um modelo de linguagem | sem pesos, sem GPU; o `lake` (Lean) não está nesta máquina nesta rodada | o **verificador** e uma busca completa para uma classe grande: geometria pelo método algébrico (numerador ≡ 0, com as condições de não degenerescência como certificado) e conferência em outra aritmética (racionais exatos); **conjecturas achadas e provadas sem serem pedidas**; topologia por homologia exata (torção) e persistente ([MATEMATICA.md](MATEMATICA.md)) |
| "síntese e descoberta de algoritmos" | AlphaDev/AlphaTensor | — | três buscas, três certificados que não confiam nelas: redes de ordenação (princípio 0-1), multiplicação de matrizes (tensor exato nos inteiros), truques de bits (minimalidade por exaustão sólida num domínio finito); complexidade ajustada às contagens ([DESCOBERTA.md](DESCOBERTA.md)) |
| "física em suas diversas frentes" | um motor universal | não existe; cada frente tem o seu método | um experimento por frente, cada um contra forma fechada ou valor publicado, com controle: quântica, relatividade, plasma, **tokamak** (Solov'ev), química (Hartree–Fock contra Szabo & Ostlund, e o fracasso conhecido do RHF mostrado), materiais (Lennard-Jones) ([CIENCIA.md](CIENCIA.md)) |
| "biologia, algo similar a AlphaFold" | prever estruturas de proteínas | exige o modelo e as bases | **recusado pelo nome**; no lugar, o que se confere: fixação de mutantes contra a cadeia exata, filogenia por neighbour joining (RF 0), dobramento no modelo HP até o ótimo publicado |
| "baterias, materiais" | DFT, eletroquímica | fora do que uma rodada mede honestamente | o primeiro degrau conferido (Hartree–Fock de moléculas, um líquido de LJ); o resto no TODO |
| "AlphaZero e além" | autojogo | — | AlphaZero inteiro em escala de jogo da velha (rede de política e valor, PUCT, treino só por autojogo), julgado contra o **jogador perfeito**, com a busca sem treino como controle; **aleatorização de domínio** como "manipulação de ambientes" ([JOGOS.md](JOGOS.md)) |
| "esboço → fotorrealista" | geração por difusão | sem pesos aqui | **não medido**: o img2img do vapor aceita o esboço com um checkpoint do usuário. No lugar, o que se mede: **esboço → desenho técnico** (retas, círculos, restrições, SVG/DXF) e **planta → 3D** (cômodos, portas, GLB) ([CENA.md §5](CENA.md)) |
| "dar vida a uma imagem: NPC, 3D, profundidade, navegação livre, esqueletos, luz, gravidade" | um gerador de mundos | profundidade monocular, segmentação e pose são redes treinadas | uma **cena 2,5D** de primeiros princípios: camadas por SLIC e o plano do chão (a profundidade como heurística, dita e editável), o fundo reconstruído por push-pull, habitantes por A* com a cabeça no horizonte, clima, luz, vento, partículas, desenhos animados por esqueleto e skinning ([CENA.md](CENA.md)) |
| "navegação livre" | andar dentro da cena | uma imagem não tem a geometria de trás | paralaxe numa janela de câmera pequena, dito; "livre" fica para quando houver profundidade aprendida |
| "loop infinito com entropia" | — | — | um gerador semeado e passo fixo: o laço é **reproduzível** (a mesma semente, as mesmas operações, o mesmo laço); a entropia é um controle |
| "direção por prompt" | um LLM interpreta | sem LLM útil embarcado | um vocabulário PT/EN → operações; **toda palavra não entendida é relatada**; o esquema de operações é o ponto de encaixe de um modelo carregado (TODO) |
| "salvamento e exportação — vale para tudo" | botões de download | um download não diz nada sobre o que contém | **`Vapor.Archive`**: zip com manifesto e hashes; a identidade é o hash do manifesto; os tipos determinísticos **se recalculam**; a cena sai como HTML autônomo e vídeo; o esboço em SVG/DXF/GLB |

### Entregas e o que as prova

| entrega | prova | controle |
|---|---|---|
| Redes de ordenação | ótimos conhecidos para n = 3…8, certificados pelo princípio 0-1. `discover_test.exs` | sorteadas e podadas: maiores; tirar um comparador quebra a rede |
| Multiplicação de matrizes | 7 produtos, exato nos inteiros; contagens da recursão ajustam n^2,807 | posto 6 nunca achado; as tentativas arredondadas anteriores, erradas |
| Truques de bits | ⌊(x+y)/2⌋ em 4 operações, nada em 3; conferido em 8/16/32 bits | `(x+y)>>1` falha |
| Geometria | 12 teoremas (Euler, nove pontos, Pappus, Simson…): 11 provados simbolicamente e conferidos; Simson conferido em racionais exatos. `prove_test.exs` | 5 gêmeos falsos refutados pelas duas vias |
| Conjecturas | reta de Euler e círculo dos nove pontos achados entre 1 001 candidatos, cada sobrevivente provado simbolicamente | nenhuma trinca trivial relatada |
| Topologia | Betti sobre GF(2) e ℚ; a persistência de um laço | GF(2) sozinho confunde toro e Klein; a mancha não tem barra longa |
| Ciência (11) | cada experimento contra a referência. `science_test.exs` | onde há um bom: implementações erradas rodadas (Euler, Laplaciano cartesiano, sítios embaralhados); dois sem controle, e dois mais fracos — ditos em [CIENCIA.md §1](CIENCIA.md) |
| AlphaZero | contra **todas** as linhas ótimas do jogo perfeito: nenhuma perdida com 128 simulações; com 8, 17 de 129. `games_test.exs` (a amostra de 60 partidas da primeira versão dizia "0 com 64" — a contagem exaustiva acha 4 de 131; corrigido) | a mesma busca sem treino: 169 de 175 com 8 |
| Aleatorização de domínio | 484 passos em média em carros-pêndulo não vistos | treinada em um: 275 |
| Cena viva | céu, horizonte, chão, luz; esqueleto de um desenho; prompts → operações. `scene_test.exs` | uma palavra sem sentido: relatada |
| Esboço e planta | retângulo endireitado; cômodos 12/20 m²; portas 0,9/1,0 m; GLB no trimesh. `sketch_test.exs` | sem restrições, o retângulo continua torto |
| Arquivos | verificação, recálculo `{:ok, :same}`. `archive_test.exs` | um byte trocado e re-zipado: `{:tampered, ["result.json"]}`; uma mentira coerente: pega pelo recálculo |

### O que o escrutínio achou no caminho (sem ser pedido)

- **Precedência de operadores** em Elixir (`-(x)**2` e `a + b |> min(2)`)
  produziu dois resultados físicos errados — o pacote quântico e as
  mutações. As referências pegaram os dois na primeira execução: a regra
  "toda medida tem a sua referência" fez o seu trabalho.
- **Uma barreira desalinhada da grade** mudava a transmissão em 18 %.
- **O tamanho dos habitantes** foi primeiro uma constante (0,42 da altura
  da imagem na linha de baixo) e saía desproporcional; a regra de
  primeiros princípios — numa foto na altura dos olhos, **as cabeças ficam
  no horizonte** — resolveu.
- **A análise de cena não achava o céu inteiro**: um degradê de céu vira
  várias regiões; o céu agora cresce para baixo por regiões lisas de cor
  vizinha.
- **Laços fechados** (um círculo, um retângulo) não tinham nós no grafo
  do esqueleto e sumiam da vetorização.
- **Um módulo existente foi sobrescrito** durante a rodada (o novo
  `Vapor.Bundle` por cima do `Vapor.Bundle` de implantação); restaurado do
  zip original, e o novo virou `Vapor.Archive`.

**A revisão independente da 0.11** (um revisor que não escreveu o código,
lendo-o contra estes documentos) achou, e foi corrigido:

- **Arquivos como vetor de ataque**: uma receita é dado vindo de fora, e
  nenhuma tinha limite — uma rede de ordenação de 32 fios (2³² vetores),
  `beam: 0` (laço sem fim), 10⁶ partidas de autojogo; e um zip era
  descomprimido sem limite (bomba). Agora todo parâmetro é limitado antes
  de rodar e a descompressão é contada enquanto acontece (testados).
- **0/0 "provado"**: o provador simbólico não conferia denominadores
  identicamente nulos; uma construção degenerada para todo valor (o pé de
  uma perpendicular sobre a "reta" de um ponto só) saía `:proved`. Agora
  é `{:degenerate, :construction}` (testado).
- **"Conferência independente"** era independente da álgebra de
  polinômios, não das fórmulas das construções — reescrito; e as
  conjecturas eram "provadas" só por amostra em racionais: agora cada
  sobrevivente é provado simbolicamente.
- **"Nunca perde"** vinha de 60 partidas sorteadas; a contagem exaustiva
  de todas as linhas ótimas acha 4 perdidas de 131 com 64 simulações. A
  afirmação passou a ser a que vale (nenhuma com 128) — [JOGOS.md](JOGOS.md).
- Controles fracos ditos como tais (o tunelamento clássico é calculado; a
  norma é uma conservação); a taxa h² do tokamak é do localizador do
  eixo, não do esquema (exato nesses polinômios); a classe n^2,807 está
  nas contagens por construção; um carro-pêndulo "não visto" está no canto
  da faixa de treino; uma imagem não decodificada derrubava o laboratório.

### O que este documento não afirma

- Nenhuma afirmação de **qualidade generativa** (fotorrealismo, vídeo
  gerado): nada disso roda aqui.
- **AlphaFold, DFT de baterias, relatividade geral**: recusados pelo nome.
- **Profundidade "de verdade"**: a da cena viva é heurística.
- **Provas legíveis**: os certificados geométricos são algébricos.
- Os números da ciência são de **modelos de livro-texto** (STO-3G, HP,
  Solov'ev): corretos contra as referências, não preditivos para o mundo.

### Como contestar

```sh
mix test test/vapor/discover_test.exs test/vapor/prove_test.exs test/vapor/science_test.exs \
         test/vapor/games_test.exs test/vapor/scene_test.exs test/vapor/sketch_test.exs \
         test/vapor/archive_test.exs test/vapor/console_test.exs
mix run -e 'IO.inspect Vapor.Games.train(games: 400, sims: 32, lr: 0.02, seed: 1).net |> Vapor.Games.digest()'   # = o digest de priv/games
mix vapor.quality                                 # §5g de docs/bench/QUALITY.md
```

## 15. Rodada 0.12: resolver problemas arbitrários — bancada, engenharia, lógica, tabuleiros, proteínas, render

> Pedido 4 (2026-10-05): "Forneça um zip refinado ao final → esta própria
> diretriz está sujeita a refinamentos e escrutínio → resultado deve ser um
> artefato que resolva as dores reais da indústria e academia com inovação
> real, pensamento lateral e primeiros princípios (ataque as limitações,
> TODO e complete o que for pertinente e que está sujeito a refinamento +
> testes de qualidade para garantir que as respostas não sejam apenas
> ruído + UI/UX original e elegante). Tarefa primária: questão de
> acabamento, completude de features, maleabilidade e flexibilidade de
> operação, resolver dores reais de engenharia elétrica, mecânica, química
> e civil, bem como física, química, matemática, CS e biologia, HPC
> máximo, remoção de nomes proprietários como alpha fold etc., e além
> disso expandir capacidades e habilidades como similar ou superior (em
> capacidades e features e flexibilidade) a alpha fold e alpha zero (com
> comparações vs o próprio alpha fold ou seu equivalente open source) +
> direção interativa da cena 'animada' + controle fino de NPCs + amplitude
> das features (… aumentar a flexibilidade, casos de uso reais, dores
> reais e controle fino como alpha proof ou equivalente para CS,
> matemática, física (nível Fields, Turing, Nobel e fronteira, usando IA
> com maestria com ou sem humano no loop) etc. para problemas arbitrários
> e não apenas categorias pré-definidas e limitadas, e a cena e criação e
> edição no estúdio muito mais flexível e personalizável e mirando a
> possibilidade do foto-realismo) + mais acabamento em UI/UX e interface
> profissional e estonteante → foco em flexibilidade e controle fino e
> arbitrário do usuário em vez de limitação a opções pré-determinadas +
> quem sabe um carinho especial a xadrez, shogi e go e jogos de cartas, e
> para tudo: interface profissional, ordem alfabética etc."

### Escrutínio do pedido (e desta diretriz)

O pedido tem uma tese que vale mais que qualquer item: **"problemas
arbitrários e não apenas categorias pré-definidas"**. As rodadas 0.10 e
0.11 entregaram laboratórios — cada um uma demonstração bem conferida de
*um* problema escolhido por nós. Isso é exatamente a "limitação a opções
pré-determinadas" que o pedido rejeita. O refinamento adotado:

1. **A entrada é texto do domínio, não um formulário.** A equação como no
   papel, a netlist como no SPICE, a lista de barras, os nós e barras de
   um pórtico, as reações como na lousa, o DIMACS, o FEN. Os exemplos são
   pontos de partida; o usuário escreve o próprio problema.
2. **Cada resposta traz o que permite julgá-la** — e esse "o quê" é
   calculado **fora do solver**: Kirchhoff reavaliado, equilíbrio de
   cargas e reações, desbalanço recomputado pelas admitâncias, invariantes
   pela estequiometria, KKT, a ordem observada por solução manufaturada, a
   refutação DRUP conferida por outro programa, a árvore de mate refeita.
   É a regra de 0.11 ("uma busca propõe, um verificador decide") estendida
   a tudo — e é ela que responde a "garantir que as respostas não sejam
   apenas ruído".
3. **"Similar ou superior a" é decomposto no que é verificável.** Um
   preditor de estrutura de ponta e um agente de autojogo de ponta são,
   cada um, uma cadeia — métricas, sinal, busca, juiz. Aqui cada elo
   existe, com teste e controle, e a comparação diz onde é igual (as
   métricas, a verificabilidade) e onde é muito inferior (predição a
   partir de sequência real; força de jogo em tabuleiro real).
4. **"IA com ou sem humano no loop" é implementada como protocolo, não
   como promessa**: a mesa de lógica aceita a *proposta* de qualquer um —
   pessoa, busca ou modelo de linguagem pelo MCP — e só o verificador
   decide (`Vapor.Logic.check/2`, `logic_check`).
5. **Nomes proprietários** saem da superfície do produto (identificadores,
   painéis, API, tipos de arquivo; os nomes de 0.11 seguem aceitos para
   não quebrar arquivos salvos) e ficam só onde o pedido os quer: nas
   **comparações**, como referência.
6. **Sobre esta própria diretriz**: a seção 14 dizia "AlphaFold: recusado
   pelo nome". A recusa era honesta mas preguiçosa — tratava o pedido como
   tudo-ou-nada. A 0.12 a refina: a predição a partir da sequência
   continua fora de alcance (e é dito), mas tudo o que a cerca e que se
   confere — métricas iguais ao TM-align, dobramento por contatos,
   contatos pela coevolução, o pipeline com controle — foi feito.

| pedido | leitura literal | o que impede | o que foi entregue (e medido) |
|---|---|---|---|
| "problemas arbitrários" | um solver universal | não existe | **Bancada**: EDOs (rígidas inclusive), EDPs (parabólicas, hiperbólicas, Poisson), sistemas, ajustes, otimização, planilhas com **unidades conferidas antes de rodar** — [BANCADA.md](BANCADA.md) |
| "dores reais de engenharia elétrica, mecânica, química, civil" | pacotes comerciais | escala, licenças | oito ferramentas com **certificado independente**: SPICE (MNA, .op/.dc/.ac/.tran), fluxo de potência (Newton), pórticos e treliças (modos), MEF plano (QM6), redes de tubos, cinética com invariantes, flash, destilação — [ENGENHARIA.md](ENGENHARIA.md) |
| "HPC máximo" | — | sem GPU nesta máquina | o conjunto de incerteza **compilado para o worker nativo**: 4096 osciladores × 1000 passos em 236 ms, **53×** a BEAM, paridade bit a bit com o oráculo; a GPU do navegador para a luz (render) |
| "similar ou superior a AlphaProof … nível Fields/Turing" | RL sobre o Lean | sem pesos, sem GPU, sem Lean | quatro lógicas decidíveis com **proponente/verificador**: CDCL + DRUP (S(3) = 13, W(3; 2) = 9, R(3, 3) = 6 certificados), Knuth–Bendix, Gröbner; **propostas externas conferidas** pelo MCP — [LOGICA.md](LOGICA.md). Nível Fields não é afirmado. |
| "similar ou superior a AlphaFold (com comparações)" | prever estruturas | pesos e bases | métricas **iguais ao TM-align**, dobramento por contatos (TM > 0,75 de contatos verdadeiros), DCA (precisão 0,96), pipeline (TM 0,69; controle 0,20), tabela de comparação com AF2/OpenFold/ESMFold — [PROTEINAS.md](PROTEINAS.md) |
| "similar ou superior a AlphaZero … xadrez, shogi, go, cartas" | um motor de ponta | escala de treino | regras **fixadas por perft** e por python-chess/python-shogi, provas de mate conferidas, Go com superko, k em linha resolvido, pôquer por CFR+ com explorabilidade exata, autojogo **genérico** julgado contra o jogo perfeito — [TABULEIROS.md](TABULEIROS.md) |
| "foto-realismo" no estúdio | um gerador de imagens | sem pesos | **traçado de caminhos** fisicamente baseado na GPU do navegador, com referência no servidor; fornalhas e N^−½ como conferência — [RENDER.md](RENDER.md) |
| "direção interativa da cena; controle fino de NPCs" | — | — | habitantes com nome, fala, ações, comportamentos, rotas por clique, linha do tempo; direção por orações com tempo e pronomes; GIF de quadros exatos — [CENA.md §6.1](CENA.md) |
| "interface profissional, estonteante; ordem alfabética" | — | — | navegação regrupada e **alfabética por idioma**, paleta de comandos (Ctrl/⌘ K), seis painéis novos no mesmo desenho, diagramas próprios (unifilar, pórtico, MEF, rede, McCabe–Thiele, Bode, tabuleiros, visor 3-D) — [CONSOLE.md](CONSOLE.md) |
| "testes de qualidade" | — | — | §5h: 27 verificações com controle; testes de cada ferramenta contra forma fechada, valor publicado ou oráculo externo (SciPy, python-chess, python-shogi, TM-align, Biopython); **todo exemplo de todo painel** rodado no Chromium |

### O que a revisão desta rodada achou (e corrigiu)

- **Otimização**: sem limites, a lata ótima descia para r < 0 e o
  resultado devolvia −6·10⁶² como "ótimo". Agora: limites de caixa por
  projeção (BFGS projetado), divergência detectada e dita, e um veredito
  KKT explícito em todo resultado.
- **Render**: o leitor de cenas aceitava `r=oops` e quebrava depois;
  agora recusa com a linha. A primeira conferência N^−½ usava vidro sob o
  sol e falhava — não por defeito, mas porque essas cáusticas têm
  variância de cauda pesada sem MIS; a conferência passou a uma cena
  difusa e o limite foi escrito ([RENDER.md §3](RENDER.md)).
- **Direção de cena**: em inglês, "a knight named Arthur walks to the
  door, then at 3s he says…" virava três pessoas sem nome e quatro
  palavras desconhecidas; o tempo sozinho numa oração ("depois de 2
  segundos, …") se perdia. Agora: criar e dirigir na mesma oração,
  pronomes, papéis sem nome, tempo levado à oração seguinte — com teste e
  controle.
- **Console**: uma classe CSS do pôquer encolhia as peças pretas do
  xadrez; a página referia scripts externos (agora embutidos, para seguir
  um documento único); o histórico do navegador era sombreado por uma
  variável. Achados pelas capturas de tela do Chromium, não por leitura.

### O que este documento não afirma

- **Superioridade** sobre os preditores de estrutura ou os motores de
  jogo de ponta: as comparações dizem o contrário, em força.
- Nível **Fields/Turing/Nobel**: as lógicas são decidíveis e os números
  provados são clássicos; o que é novo é o protocolo verificável, não o
  teorema.
- **Foto-realismo gerado**: há transporte de luz correto para cenas de
  primitivas; não há geração de imagens, malhas, texturas nem MIS.
- **Engenharia de projeto**: as ferramentas são lineares/estacionárias
  (exceto os circuitos e a cinética) e não substituem normas de cálculo.

### Como contestar

```sh
mix test test/vapor/bancada_test.exs test/vapor/engenharia_test.exs test/vapor/logica_test.exs \
         test/vapor/tabuleiros_test.exs test/vapor/proteinas_test.exs test/vapor/render_test.exs \
         test/vapor/scene_test.exs test/vapor/console_desks_test.exs test/vapor/mcp_server_test.exs \
         test/vapor/rodada12_test.exs
mix vapor.quality                                 # §5h de docs/bench/QUALITY.md
node test/js/console_desks.mjs http://127.0.0.1:8000/ /tmp/capturas   # com `mix vapor.serve --docs .`
```

## 16. Rodada 0.13: finanças e HFT — e a própria diretriz como objeto de defesa

> Pedido 5 (2026-10-05): "Forneça um zip refinado ao final → esta própria
> diretriz está sujeita a refinamentos e escrutínio → resultado deve ser um
> artefato que resolva as dores reais da indústria e academia com inovação
> real, pensamento lateral e primeiros princípios (ataque as limitações,
> TODO e complete o que for pertinente e que está sujeito a refinamento +
> testes de qualidade para garantir que as respostas não sejam apenas
> ruído + UI/UX original e elegante) + considere: suporte a finanças, HFT,
> ataque as limitações atuais e TODO + atualização dos slides + documento
> LaTeX no formato de monografia em português (com figuras e didática desde
> bacharel CS a PhD CS, e com ênfase na arquitetura e inovações: o que,
> para quem, para que, por que, como, porém num formato sóbrio e
> acadêmico) acerca do vapor + script para defesa oral de tal monografia
> (formato md)."

### Escrutínio do pedido (e desta diretriz)

**A tese.** A dor de finanças não é velocidade, é **verificabilidade**. Um
preço que muda quando o cálculo muda de máquina, um backtest que olhou o
amanhã, a melhor de cinquenta estratégias apresentada como a única, um
livro de ofertas que ninguém de fora audita — são falhas de prova, não de
desempenho. O vapor tem, desde a 0.1, a resposta de princípio (os mesmos
bits em todo substrato; uma busca propõe, um verificador decide); a rodada
a leva ao mercado em vez de inventar um produto paralelo.

**"HFT", lido literalmente**, é um motor de bolsa em nanossegundos (FPGA,
*kernel bypass*). A BEAM é tempo real brando; afirmar isso seria mentir.
O refinamento: entregar o que de HFT se confere — um motor preço-tempo
cujo diário é um objeto verificável, os protocolos reais (ITCH 5.0, FIX
4.4), o risco pré-negociação exigido por regra, a microestrutura com a
medida que diz se o modelo serve, e uma sessão de bolsa em que **o
backtest é o código da bolsa** — e medir a latência honestamente
(microssegundos por evento, com o hash).

**"Inovação real, pensamento lateral, primeiros princípios"** foi
decomposto em cinco movimentos, cada um com teste e controle:

1. **Antecipação como propriedade de prefixo.** Em vez de auditar
   operadores (o que deixaria passar uma normalização pela amostra
   inteira, causal passo a passo e não causal no todo), o sinal é
   recalculado sobre históricos truncados e tem de ser igual bit a bit — um
   teste de caixa-preta do pipeline inteiro, com o dia da espiada.
2. **O teorema fundamental como lema de Farkas.** Arbitragem não é
   estimada: o simplex racional decide, e a resposta é **ou** o portfólio
   **ou** os preços de estado — objetos que qualquer um confere
   multiplicando. E a mesa aceita a *proposta* de qualquer um (pessoa,
   busca, modelo de linguagem via MCP), como a lógica da 0.12.
3. **Monte Carlo como programa canônico.** O passo da trajetória — o
   gerador inclusive, Wichmann–Hill escolhido porque cabe **exato** em
   binary32 — é álgebra do vapor: bits iguais ao oráculo e entre contagens
   de threads, conferidos na própria resposta.
4. **O livro julgado por um motor ingênuo.** Um segundo motor, escrito à
   parte com listas e ordenação, refaz o diário e exige os mesmos
   relatórios; os invariantes são conferidos sem executar nada. O
   *fuzzing* diferencial achou um bug que os dois motores compartilhavam
   (abaixo).
5. **A simulação é a produção.** Formadores, agressores Hawkes e um
   informado passam pelo mesmo portão de risco e pelo mesmo motor; a
   sessão devolve a própria auditoria e se repete na mesma cabeça de hash.

**"Ataque as limitações e o TODO"** foi refinado em vez de espalhado: as
pendências fechadas são as que o domínio novo **tornou baratas ou
necessárias** — o simplex com certificado de Farkas (pendência da lógica
na 0.12) é o motor da arbitragem; arquivos **assinados** (pendência da
0.11) são o que uma sessão de bolsa ou um backtest exigem para valer como
registro; as escalas afins (°C/°F, pendência da bancada) e os transistores
(MOSFET nível 1 e Ebers–Moll, pendência da engenharia, agora iguais ao
ngspice) vieram junto porque custavam horas e removiam desculpas.

**Sobre esta própria diretriz.** O pedido se repete, rodada após rodada,
com "resolva as dores reais… inovação real… testes… UI/UX…". O risco de
um pedido-fórmula é a resposta-fórmula: mais um painel, mais uma tabela de
"pedido × entrega". O refinamento desta rodada é duplo: (i) **um domínio
por rodada, fundo** — finanças inteira, do dinheiro exato ao livro de
ofertas — em vez de dez superficiais; (ii) a diretriz agora pede que o
artefato seja **defensável por escrito e oralmente**: a monografia e o
roteiro de defesa obrigam o projeto a explicar *o que, para quem, para
que, por que e como* num registro sóbrio, e qualquer afirmação que não
sobreviva a essa explicação sai. (Duas saíram: "HFT" virou "verificabilidade
de mesa e de bolsa"; "superior a" não aparece.)

| pedido | leitura literal | o que impede | o que foi entregue (e medido) |
|---|---|---|---|
| "suporte a finanças" | uma biblioteca quantitativa completa | escala (o QuantLib tem 20 anos) | dinheiro exato; calendários ANBIMA/NYSE/TARGET **iguais ao QuantLib dia a dia (1990–2078)**; curvas DI1/LTN/NTN-F/swaps reprecificadas; BSM/Heston/árvores **= QuantLib** (10⁻⁹–10⁻¹²); SVI com arbitragem; VaR com Kupiec/Christoffersen/Basileia; carteiras com KKT — [FINANCAS.md](FINANCAS.md) |
| "HFT" | motor de bolsa em nanossegundos | a BEAM; sem FPGA | motor preço-tempo com **diário SHA-256 + Merkle**, juiz ingênuo independente (fuzzing diferencial), ITCH 5.0, FIX 4.4 (= simplefix), portão pré-negociação, Hawkes, Avellaneda–Stoikov (o artigo reproduzido), Almgren–Chriss, sessão de bolsa auditada; latência medida (~8,5 µs/evento com hash) |
| "HPC" (herdado) | — | sem GPU | Monte Carlo **no worker** com o gerador dentro do programa: 6–23× a BEAM, bits = oráculo = 2 threads |
| "respostas não apenas ruído" | — | — | **quatro portões de ruído** em backtests; §5i: **23 verificações com controle**; o tamanho e o poder do teste de Kupiec medidos |
| "inovação, pensamento lateral" | — | — | invariância de prefixo; arbitragem por Farkas com propostas de qualquer um; risco com bits canônicos; o livro julgado por um motor ingênuo; backtest = código da bolsa |
| "TODO" | tudo | hardware, pesos | LP racional com Farkas; arquivos assinados; °C/°F; MOSFET/BJT = ngspice |
| "UI/UX original e elegante" | — | — | grupo *Mercados*: selo, portões, semáforo de Basileia, escada do livro, cadeia do diário; 87 exemplos rodados no Chromium em EN e PT — [CONSOLE.md](CONSOLE.md) |
| "slides" | — | — | `slides/vapor.tex` atualizado (rodada 0.13), PDF recompilado |
| "monografia LaTeX, PT, bacharel → PhD" | — | — | `monografia/` (abnTeX2): o que, para quem, para que, por que, como; arquitetura e inovações; figuras TikZ/pgfplots e capturas; PDF compilado |
| "roteiro de defesa oral" | — | — | `monografia/DEFESA.md`: falas por slide, tempos, perguntas prováveis da banca e respostas |

### O que o escrutínio achou no caminho (sem ser pedido)

- **FOK a mercado virava IOC** nos dois motores — o rápido e o ingênuo,
  porque a mesma leitura errada da especificação foi escrita duas vezes.
  Só o invariante "FOK é tudo-ou-nada", que não executa nada, viu. Lição
  registrada no documento: dois motores não bastam; os invariantes são o
  terceiro juiz.
- **Calendários**: a primeira comparação com o QuantLib divergiu num
  único dia do NYSE em 89 anos (27/04/1994, o luto por Nixon) e nos anos
  do TARGET anteriores a 2000. Viraram dados e regras; o teste exige
  igualdade dia a dia.
- **Monte Carlo**: a primeira versão levava 95 s — 30 s na compilação de
  16 passos desenrolados para quatro ISAs (o verificador de alocação
  extraído do Lean é quadrático) e 105 s no oráculo exato sobre 8 192
  pistas. Agora: um passo por chamada, só a ISA da máquina, e a paridade
  com o oráculo feita num programa de 64 pistas comparado às 64 primeiras
  do worker (as pistas são independentes — o que a própria comparação
  confirma). E o primeiro gerador vinha da BEAM: o "nativo" não era mais
  rápido que a BEAM. Com Wichmann–Hill dentro do programa, 6–23×.
- **A sessão de bolsa** "reprovou" o próprio Hawkes: o simulador
  distribuía as chegadas uniformemente dentro de cada passo, e o teste de
  reescala do tempo — corretamente — recusou o modelo. As chegadas agora
  vêm de um Hawkes exato em tempo contínuo. E os formadores perdiam muito
  para o informado porque cotavam em torno do meio do próprio livro (que
  só acompanhava o fundamental através das agressões); passaram a cotar
  em torno do preço público do passo anterior. Era o simulador, não o
  motor.
- **Avellaneda–Stoikov**: o spread médio daqui (1,49) difere do da
  tabela do artigo (1,29): o artigo parece reportar só o termo
  (2/γ)ln(1 + γ/k). Dito no documento; as dispersões, que são o resultado
  do artigo, batem.
- **Kupiec**: o controle (VaR normal em t(3)) rejeita com p = 0,038 —
  perto da borda. Registrado como está, sem trocar de semente.
- **O compilador do OTP 25** falhou com um erro interno (`beam_ssa_type`)
  numa compreensão com padrões complexos no texto das opções; a função foi
  dividida em funções por tarefa — o código ficou melhor do que estava.

### O que este documento não afirma

- Latência de bolsa, nem que a BEAM sirva a HFT de nanossegundos.
- Dados de mercado reais: os exemplos são ilustrativos; os oráculos são
  bibliotecas (QuantLib, simplefix, SciPy, ngspice), não o mercado.
- Que passar nos portões torne uma estratégia lucrativa.
- Cobertura de XVA, crédito, taxas de vários fatores, volatilidade local
  ou estocástica calibrada a uma superfície.
- Que Wichmann–Hill seja um gerador moderno (não é; serve a precificação
  com o caminho `rng: :host` como alternativa).

### Como contestar

```sh
mix test test/vapor/financas_test.exs test/vapor/console_markets_test.exs test/vapor/engenharia_test.exs \
         test/vapor/bancada_test.exs test/vapor/archive_test.exs test/vapor/mcp_server_test.exs
mix vapor.quality --only round13                      # §5i, em ~25 s
node test/js/console_markets.mjs http://127.0.0.1:8000/ /tmp/capturas
cd monografia && latexmk -pdf monografia.tex          # a monografia
```

## 17. Rodada 0.14: de vitrine a ferramenta — a bancada aberta

> Pedido 6 (2026-10-05): "muitas das features estão jogadas e sem liberdade
> (sobretudo no módulo ciência que parecem apenas problemas já resolvidos
> sendo apenas apresentados ao usuário) → mudança total de filosofia e
> funcionamento do vapor: em vez de ser algo expositivo deve ser uma
> ferramenta real que resolva dores reais e problemas reais com input
> flexível e aberto (porém sanitizado) do usuário, sem se limitar a
> categorias pré definidas […] human in the loop com modelos de AI […]
> retorno às raízes do vapor com uma suíte mais completa em termos de
> atividades de pesquisa em AI […] filosofia unix com tudo sendo possível de
> ser executado via terminal […] cenas […] livre para criação edição
> pesquisa". E, no meio: "rebatize a linguagem interna para algo na linha de
> alquimia […] em inglês + não use nomes consagrados […] foco em ciência, cs,
> matemática, AI e finanças + melhore a aparência da interface […] mais
> marcante e ui/ux inovadora".

### O pedido, escrutinado

- **"Uma coletânea de" buscadores famosos** — tomado ao pé da letra, seriam
  N ferramentas estreitas com N formatos de entrada: exatamente a falta de
  liberdade criticada. O primeiro princípio comum a todos é *proponha,
  avalie com um verificador, guarde o certificado*. Então: **uma** linguagem
  para escrever o problema (Alembic), **uma** fornalha que busca qualquer
  coisa escrita nela (Athanor) e **uma** pedra de toque que confere sem
  confiar (Touchstone). Os casos famosos viram exemplos de partida, não
  produtos. Nenhum nome de terceiros é usado.
- **"Uso intensivo de AI em tudo"** — com um risco: um modelo que decide é
  um modelo que alucina com autoridade. A regra adotada: o modelo
  **propõe e rascunha** (formaliza palavras em Alembic, com retrotradução para
  a pessoa conferir; sugere candidatos), e a máquina **decide** pelo mesmo
  verificador que julga a pessoa e o acaso. Sem modelo, tudo funciona.
- **"Entrada aberta porém sanitizada"** — a sanitização certa não é uma
  lista de palavras proibidas, é uma linguagem sem efeitos e com custos:
  sem E/S, combustível em todo passo, tetos de tamanho, processo isolado com
  teto de memória, identificadores que nunca viram átomos, dados lidos por um
  leitor que recusa código, e no navegador uma árvore interpretada (nunca
  `eval`). Cada limite tem um teste que tenta quebrá-lo.
- **"Ciência que só apresenta problemas resolvidos"** — a crítica é justa. O
  Crucible inverte: a pessoa traz o sistema, e a evidência é a que **não
  precisa de gabarito** (ordem observada, teoremas para qualquer entrada,
  dois métodos que concordam, controles que um método errado reprovaria).
  Os experimentos fixos viram **Calibração**, que é o que eram.
- **"Raízes do vapor em pesquisa de IA"** — a dor mais comum não é treinar,
  é **saber se uma diferença é real**. O Assay responde a isso: comparação
  pareada com poder, *leaderboard* com empates, calibração contra o seu
  piso, concordância, viés de juiz, contaminação, deduplicação verificada,
  leis de escala que precisam prever as maiores corridas sem tê-las visto.
- **"Interface mais marcante"** — o risco é decorar. A identidade (fuligem,
  pergaminho, latão, verdete, cinábrio; serifa antiga nos títulos) carrega
  uma metáfora que é também a função: a **fornalha** é o gráfico ao vivo da
  busca (faíscas, a linha do melhor, a linha tracejada do controle, a
  partilha do portfólio) e a **pedra de toque** é o veredito (um risco de
  ouro por verificação que passou, de chumbo pela que falhou). O ornamento
  está onde está a informação.

### O que foi feito

[ALEMBIC.md](ALEMBIC.md), [ATHANOR.md](ATHANOR.md), [CRUCIBLE.md](CRUCIBLE.md),
[ASSAY.md](ASSAY.md), [CLI.md](CLI.md); console: Espaço de trabalho, Crisol,
Ensaio; cenas livres (operações em texto, expressões por quadro, `direct`
por modelo); MCP com 20 ferramentas; TUI com os mesmos verbos.

### O que foi achado no caminho

- **A lei de escala estourava `exp`** com perdas embaralhadas (o controle da
  rodada 14 achou): parâmetros em log sem estrutura cresciam sem limite. O
  ajuste agora é finito e o *holdout* diz que a lei não prevê nada.
- **O controle do Kepler** no primeiro desenho (dt = 0,05, 30 órbitas) não
  separava nada: o RK4 também conservava a energia a 10⁻⁶. Um controle que
  não pode falhar não é controle; a verificação passou a usar dt = 0,1 e
  ~250 órbitas, onde o RK4 deriva 40× mais que o erro limitado do simplético.
- **`Vapor.Expr.compile`** criava um módulo (e um átomo) por expressão nova:
  com entrada aberta, isso é um vazamento sem limite. Agora há um teto, e o
  excedente é interpretado sem criar átomo nenhum.
- **A fornalha escolhia aleatório demais** com UCB simples; sem o braço
  aleatório, perdia nos problemas rugosos. O UCB descontado com o braço
  mantido resolve os dois.
- **O holdout de regras de média móvel**: num passeio aleatório o vencedor
  dentro da amostra tem ρ negativo fora dela — o viés de seleção, medido.
  Para garantir que o teste não acusa tudo, um momento AR(1) plantado
  (φ = 0,5) sustenta ρ ≈ 0,73; φ = 0,2 já não (ρ ≈ 0,21), e isso também é
  informação.

### O que este documento não afirma

- Que "provado" signifique mais que **enumeração completa de um espaço
  finito**, com o tamanho dito no certificado.
- Que a fornalha seja competitiva com resolvedores especializados (SAT, MIP)
  nos seus próprios domínios; a lógica e o LP exatos continuam nas suas
  mesas.
- Que o rascunho do modelo esteja certo: a retrotradução existe para a
  pessoa conferir.
- Química além de STO-3G de camada fechada com H e He.

### Como contestar

```sh
mix test test/vapor/alembic_test.exs test/vapor/athanor_test.exs test/vapor/crucible_test.exs \
         test/vapor/assay_test.exs test/vapor/mind_test.exs test/vapor/scene_ops_test.exs \
         test/vapor/workspace_test.exs test/vapor/mcp_server_test.exs
node test/js/scene_noise.mjs
mix vapor.quality --only round14                      # §5j, em ~60 s
bin/vapor alembic --card                              # e então escreva o seu problema
```
