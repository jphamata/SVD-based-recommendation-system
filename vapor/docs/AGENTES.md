# Agentes imutáveis, RAG verificável e modelos de fronteira

Este documento registra o escrutínio da diretriz desta rodada, o desenho
que saiu dele e — com o mesmo peso — o que **não** é garantido. Tudo que
aparece como fato aqui tem teste no repositório. O nome do teste vai entre
parênteses.

## 1. A diretriz, examinada

> *“zip refinado + suporte a modelos fronteira + agentes + rag + o que mais
> for pertinente + reflita sobre agentes imutáveis → esta própria diretriz
> está sujeita a refinamentos e escrutínio → resultado deve resolver as
> dores reais da indústria e academia com inovação real, pensamento lateral
> e primeiros princípios”*

Cada termo da diretriz admite mais de uma leitura. A leitura escolhida
decide o que vale a pena construir.

**“Modelos de fronteira” tem duas leituras, e as duas valem.** A primeira
é rodar localmente, com a garantia do vapor, as *arquiteturas* das famílias
de fronteira abertas: Qwen3, Qwen3-MoE, Mixtral, Gemma 3 e DeepSeek-V3
(MLA + MoE). A segunda é usar os *modelos hospedados* (Claude, GPT, ou
qualquer servidor compatível com OpenAI) como cérebro de um agente. As duas
estão implementadas, mas são de naturezas diferentes, e o desenho torna
essa diferença explícita. Uma decisão de um modelo local é **re-derivável**:
é função pura de pesos, tokens, parâmetros e semente, com os mesmos bits em
qualquer substrato. Uma decisão de modelo remoto é **observação**: pode ser
registrada e protegida contra adulteração, mas não reproduzida. A semente
da OpenAI é declaradamente de “melhor esforço”, e a API de Mensagens da
Anthropic não oferece semente. Nenhum texto deste projeto promete o
contrário.

**“Agentes” não pedem mais um laço de orquestração.** Frameworks de
agentes existem às dezenas, e mais um laço `pensar → chamar ferramenta →
observar` não resolve dor nenhuma. O que o vapor tem de único é o
**determinismo**: invariância a lote, a threads e a substrato, com
certificado. A pergunta de primeiros princípios passa a ser outra: *o que
um agente pode ser quando o seu modelo é uma função?* A resposta orienta
todo o resto. Uma execução deixa de ser um log e vira um **objeto de
prova**, que qualquer um com os mesmos pesos pode repetir e conferir.

**“RAG” pela mesma lente.** Ninguém precisa de outro índice vetorial. As
dores reais são três: o resultado não se reproduz, ninguém prova de qual
corpus veio a resposta, e o modelo inventa citações. O RAG daqui ataca
essas três.

**“Agentes imutáveis” tem pelo menos cinco leituras**, e cada uma tem um
limite:

| leitura | o que entrega | o que *não* entrega |
|---|---|---|
| a definição do agente é um valor (endereçado por conteúdo, com linhagem) | saber exatamente *qual* agente agiu; versões comparáveis | correção: um agente imutável pode estar imutavelmente errado |
| a execução é só-acréscimo e evidente a adulteração | auditoria, atribuição, prova de inclusão de um evento | verdade dos fatos observados: registra-se o que a ferramenta disse, não o que o mundo era |
| o agente não muda as próprias capacidades no meio da execução | injeção de prompt não escala privilégio | que o modelo não *tente*; a tentativa fica registrada e é negada |
| repetir é provar | verificação independente das decisões locais e das ferramentas puras | re-execução do mundo: `observe`/`act` nunca são re-executadas |
| imutável *versus* direito ao esquecimento | apagamento por destruição de chave (*crypto-shredding*) | apagamento de metadados estruturais: nomes de ferramentas e a forma da execução permanecem |

A última linha é o conflito que a palavra “imutável” costuma esconder. Um
log imutável de dados pessoais colide com o art. 17 do RGPD e o art. 18 da
LGPD. A resolução não exige escolher entre auditoria e privacidade: o
conteúdo é cifrado por titular, e só a cifra é encadeada. Destruir a chave
apaga o conteúdo em todo lugar, inclusive nos backups, e a cadeia, a raiz
Merkle e as provas continuam válidas.

**O que a diretriz não pediu e as dores pedem.** Uma execução sobrevive à
queda do processo? Uma ação externa (e-mail, pagamento) acontece no
máximo uma vez quando o agente cai no meio dela? Quem garante que o
diário guardado não foi reescrito *por quem o guarda*? O §3 responde as
três perguntas.

## 2. Dores reais → peças

| dor | contexto | o que o vapor faz | evidência |
|---|---|---|---|
| chamadas de ferramenta malformadas, nomes inventados, argumentos fora do esquema (laços de *retry*) | comum até em modelos grandes; pior nos pequenos | a chamada é decodificada sob gramática derivada dos esquemas declarados: nome declarado, argumentos válidos **por construção** | `agentic_serve_test` (`tool_choice: "required"` com modelo de pesos aleatórios: 100 % válido) |
| saída estruturada que às vezes não é JSON válido | `response_format` com `json_schema` | gramática de JSON Schema (subconjunto declarado; o resto é recusado pelo nome, ou aceito com `lenient: true`) | `grammar_test`, `agentic_serve_test` |
| JSON válido com campos inválidos (CEP sem hífen, 30 de fevereiro, e-mail sem domínio) | `pattern` e `format` eram recusados | desde 0.7.0, `pattern` (ECMA-262) e `format` (`date`, `time`, `date-time` com o calendário real, `uuid`, `ipv4`, `email`, `hostname`) viram autômatos de bytes: só saem valores que casam | `regex_test` (mesmo veredito que o `re` do Python e os *parsers* da biblioteca padrão em 7 240 cadeias) |
| resultados de LLM não reproduzíveis, nem com semente | o lote muda os bits dos *kernels* usuais (He et al., Thinking Machines, 2025: *Defeating Nondeterminism in LLM Inference*) | invariância a lote e a substrato **já era** propriedade do vapor; cada resposta leva um recibo (`x-vapor-receipt`) = digest canônico de modelo, ids do prompt, parâmetros e ids gerados | `serve_test`, `engine_test` |
| reprodutibilidade acadêmica | quem tem os pesos deve conseguir refazer o número do artigo | recibos + execução certificada; um Livebook que roda como teste | `notebook_test` |
| citações alucinadas em RAG | o modelo “cita” o que não está na fonte | dentro de `<quote src="i">…</quote>` só podem sair bytes que continuam uma substring da fonte *i* (autômato de sufixos); `check_citations` reconfere respostas de qualquer origem | `rag_test` (modelo aleatório: só cita literalmente) |
| “de qual corpus veio isso?” | versões de índice, reindexação em outro hardware | corpus = raiz Merkle (RFC 6962); BM25 com log corretamente arredondado; escores densos como programa certificado; RRF em racionais exatos; recibo de recuperação re-verificável | `rag_test` |
| obrigação de registro de eventos | AI Act da UE, art. 12 (registro automático em sistemas de alto risco) e art. 19 (guarda) | diário encadeado por hash + raiz Merkle + **atestado Ed25519** do nó que executou; prova de inclusão de um evento sem revelar os outros | `agent_test` |
| apagamento de dados pessoais em logs imutáveis | RGPD art. 17, LGPD art. 18 | *crypto-shredding* por titular: entrada, textos, tokens, argumentos, resultados e erros selados; estrutura legível para auditoria | `agent_test` (“erasure”) |
| injeção de prompt levando a ações | texto lido por ferramenta pode instruir o modelo | capacidades (`grants`) fazem parte do digest do agente; ferramenta `act` sem concessão é negada, e a negação é registrada; `confirm:` registra a aprovação humana | `agent_test` |
| agente cai no meio do fluxo; ação duplicada | e-mail enviado duas vezes, cobrança dupla | diário gravado antes do passo seguinte (*write-ahead*); toda ação é anunciada (`intent`) e gravada **antes** de acontecer; retomar é repetir (nada do que foi gravado se repete); chave de idempotência estável por (execução, passo, chamada); de dois retomadores, só um anuncia cada ação | `agent_store_test` |
| template de chat errado degrada o modelo em silêncio | cada modelo traz o seu Jinja | interpretador Jinja hermético, byte a byte igual ao `jinja2` do Hugging Face | `template_test` (240 renderizações de 20 templates reais + 40 trechos) |
| cliente desconecta, GPU segue gerando para ninguém | desperdício de lote | o motor monitora o processo destinatário: se ele morre (LiveView fechada, job morto), as suas sequências saem do lote no passo seguinte, também através das réplicas; no HTTP, um cliente de *streaming* que desconecta é cancelado na escrita seguinte (TCP e Plug). Uma requisição sem *streaming* roda até o fim (limitada por `max_tokens`) | `engine_test`, `serve_test`, `vapor_plug_test` |
| dependência de fornecedor | trocar de provedor reescreve o agente | o mesmo agente roda com modelo local, qualquer servidor OpenAI-compatível (inclusive o próprio vapor) ou a API de Mensagens; ferramentas MCP | `agent_test` (AgentBackendsTest) |

## 3. O desenho

```
Spec (valor, digest, linhagem) ──run──▶ Journal (eventos encadeados, raiz Merkle) ──attest──▶ assinatura Ed25519
   │ tools: pure | observe | act          │  start · model · tool · intent · tool · … · final|halt
   │ grants, policy (seed, max_steps)     │
   ▼                                      ▼
Backend: Local (re-derivável) | OpenAI | Anthropic (observação)      Store (write-ahead, fencing) ──resume──▶ continua
```

**Spec** (`Vapor.Agent.Spec`). Nome, instruções, modelo, ferramentas com
classe de efeito e esquema, concessões e política (semente, passos máximos,
temperatura). O digest é o hash canônico (CBOR determinístico, RFC 8949
§4.2) do valor inteiro, e o ambiente entra nele: versão do Unicode do OTP
e versão da codificação canônica. Duas BEAMs com tabelas Unicode diferentes
normalizam texto de forma diferente, e isso não pode passar despercebido.
`evolve/2` cria a versão seguinte com `parent` apontando para a anterior.
Uma execução de v1 não é aceita como execução de v2.

**Classes de efeito.** Três classes cobrem o comportamento de qualquer
ferramenta diante da repetição:
- `pure`: função dos argumentos. Na repetição é **re-executada** e precisa
  concordar com o registro.
- `observe`: lê o mundo. Na repetição o resultado é **lido** do registro.
- `act`: muda o mundo. Só roda com concessão; na repetição **nunca** roda;
  ao vivo recebe a chave de idempotência.

Ferramentas MCP chegam como `act` quando o operador não declara a classe.
O MCP não diz o que uma ferramenta faz, e a classe mais restritiva é o
padrão seguro.

**Journal.** `hashᵢ = SHA-256(canonical({hashᵢ₋₁, i, kind, data}))`, com
raiz RFC 6962 sobre os hashes. Qualquer linguagem recomputa, porque a
codificação é CBOR canônico, verificado contra o `cbor2` do Python. Uma
cadeia de hashes só prova integridade **relativa a uma cabeça em que já se
confia**: quem guarda o diário pode reescrevê-lo inteiro e reencadeá-lo.
Por isso existe `Journal.attest/2`. O nó que executou assina (id, número de
eventos, cabeça, raiz), e um diário reencadeado deixa de ser atestado
(`agent_test`). Publicar o atestado, ou ancorá-lo num log de transparência,
fecha a porta.

**Repetir é provar** (`Vapor.Agent.replay/3`). A repetição reconstrói a
execução a partir do diário. Cada decisão local é **recomputada** e precisa
dar os mesmos tokens. A invariância entre substratos é o que torna isso uma
verificação feita em *outra* máquina, e não apenas na mesma. Ferramentas
`pure` são re-executadas; `observe`, `act` e decisões remotas são lidas;
os anúncios de ação (`intent`) são re-derivados e conferidos. No fim, o
diário reconstruído precisa terminar na mesma cabeça. A repetição não toca
o mundo, e nenhum gancho `on_event:` dispara durante ela.

**Retomar é repetir; durabilidade sem motor de workflow**
(`Vapor.Agent.Store`). O gancho `on_event:` roda depois de cada evento novo
e antes do passo seguinte. É o ponto *write-ahead*: se o armazenamento
recusa, a execução para antes de agir. Toda ferramenta `act` é precedida de
um evento `intent` (chamada, chave de idempotência, aprovação), gravado
**antes** de a ação acontecer. `Store.resume/4` repete o prefixo gravado
sem efeitos e só segue ao vivo se o prefixo **verificou**: outro spec, uma
decisão recomputada diferente ou um evento fora de ordem fazem a retomada
parar (`{:error, {:diverged, …}}`) antes de qualquer evento novo. Um
histórico apagado pode ser verificado como registrado, mas nunca
continuado. Há duas janelas de queda:
1. *antes* do anúncio: a ação não aconteceu e acontece uma vez na retomada;
2. *depois* do anúncio e antes do resultado: a ação pode ter acontecido.
   A retomada grava um evento `retry` e a repete com a **mesma** chave de
   idempotência; um destinatário que respeita a chave aplica uma vez só.

A garantia é de **no máximo uma vez com destinatário idempotente**, não
“exatamente uma vez” em geral. Nenhum sistema distribuído oferece mais do
que isso (`agent_store_test` cobre as duas janelas). O `Store.File` grava
cada evento num temporário exclusivo, faz `fsync` e cria um *hard link* com
o nome final. `link(2)` é atômico e falha se o nome existe, e isso dá duas
propriedades de uma vez: um evento meio escrito nunca é lido, e dois
retomadores não gravam o mesmo evento. Como o anúncio (ou o `retry`) é
gravado antes da ação, **de dois retomadores concorrentes só um anuncia
cada ação**; o outro recebe `:conflict` antes de tocar o mundo. O limite
honesto: uma ação já anunciada e ainda em curso não se distingue de uma
interrompida pela queda. Um segundo retomador que chegue nesse intervalo
grava o seu `retry` e a tenta de novo, com a mesma chave de idempotência.
Fechar esse caso exigiria um arrendamento (*lease*) com relógio, que fica
como trabalho futuro. Ferramentas `pure` e `observe` não são anunciadas, e
o perdedor pode tê-las executado antes do conflito, o que é inofensivo por
definição. Um adaptador relacional obtém o mesmo com
`UNIQUE (run_id, seq)`.

Esse é o contrato de Temporal, Step Functions ou workflows do Oban,
obtido de dois fatos que o vapor já tinha (determinismo e diário
encadeado) em vez de um agendador. Uma diferença importa: o Temporal exige
que o *código* do workflow seja determinístico, e o vapor estende o
determinismo à *decisão do modelo*.

**Capacidades.** Uma ferramenta `act` sem concessão é negada, e a negação
entra no diário. Texto lido por uma ferramenta pode convencer o modelo a
*tentar*, mas não concede nada: as concessões estão no digest, fixado antes
da execução. `confirm:` coloca um humano no caminho, e a decisão dele
também vai para o diário: a recusa no resultado (`"not approved"`), a
aprovação no anúncio (`"approval" => "approved"`).

**Apagamento** (`Journal.seal/4`, `Keys`). Os valores são cifrados com
AES-256-GCM por titular, com o id da execução como dado associado. O nonce
é derivado de chave, execução e valor, de modo que selar é determinístico e
a repetição reproduz os mesmos hashes. São selados: entrada, textos do
modelo, tokens do modelo local, argumentos de chamadas, resultados e erros.
Ficam legíveis: tipos de evento, nomes de ferramentas, classes de efeito,
ids. Uma cifra que não abre com a chave existente é relatada como
divergência (`:sealed`), não como apagamento. Os recibos e o digest do prompt do modelo local também são selados,
porque um hash de texto curto confirma um palpite. Depois de
`Keys.shred/2`, a repetição relata os eventos como `redacted` em vez de
falhar, e o titular fica marcado como apagado: selar para ele de novo é
recusado, em vez de criar uma chave nova que faria a cifra antiga parecer
adulterada. Em produção as chaves moram num KMS/HSM; o
contrato tem as mesmas três operações.

**Backends.** `Local` renderiza com o template do próprio modelo e restringe
as chamadas aos esquemas. `OpenAI` atende qualquer servidor compatível
(OpenAI, vLLM, llama.cpp, Ollama ou o `Vapor.Serve`; neste caso o recibo
torna o modelo remoto verificável por quem tem os pesos). `Anthropic` fala
a API de Mensagens (`tool_use`/`tool_result`). O MCP entra por cliente
stdio próprio (`2025-06-18`), testado contra o SDK oficial.

## 4. RAG verificável

`Vapor.RAG.corpus/2` normaliza o texto (NFC), parte em trechos com offsets
e faz de cada trecho uma folha Merkle. Com isso:

- o **corpus é um valor**: a raiz nomeia exatamente o que foi indexado;
- a **recuperação é uma função**: BM25 em binary64 com ordem fixa e
  logaritmo corretamente arredondado; escores densos como `linear`
  certificado no worker (mesmos bits em todo substrato, assim como os
  embeddings de `Vapor.Embed`); fusão RRF em racionais exatos; desempate
  por id;
- o recibo `(raiz, método, k, consulta, embedder, ids)` é re-verificável
  com `verify/2`, e cada trecho leva a sua prova de inclusão.

As citações são literais **por construção**. O modelo escreve livremente,
mas dentro de `<quote src="i">` só pode continuar uma substring da fonte
*i*. Isso não garante que a citação *sustente* a afirmação; garante que
ela *existe* na fonte. Julgar a sustentação continua sendo trabalho de
quem lê (ou de outro modelo), agora com a certeza de que o texto citado é
real.

## 5. Modelos de fronteira, localmente

| família | o que exige | como entrou |
|---|---|---|
| Qwen3 | normas RMS por cabeça em q/k | `linear` com matrizes 0/1 de seleção (contração exata) |
| Qwen3-MoE, Mixtral | roteador top-k | posto por contagem + `sel`; especialistas em ordem fixa |
| Gemma 3 | GeGLU (tanh), normas `(1+w)`, normas sanduíche, √d no embedding, RoPE local/global por camada, `query_pre_attn_scalar` | novas funções canônicas `tanh` e `gelu_tanh`; escala de atenção explícita no termo |
| DeepSeek-V3 | MLA, MoE com grupos e sigmoide, especialistas compartilhados, YaRN com `mscale_all_dim`, `rope_interleave` | MLA como layout de cabeças (pares RoPE passantes, intercalação como permutação); grupos por seleção exata |

Os resultados vêm de `frontier_test`. Os logits ficam a ≤ 8·10⁻⁷ (relativo)
do `transformers` 5.18 nas 8 variantes. Os bits são idênticos em AVX2,
AVX-512, no interpretador RVV, em QEMU aarch64/riscv64 e no Vulkan. Prefill
é igual a decode passo a passo. Envenenar especialistas *não escolhidos* com
NaN/∞ não muda um bit, e todo tensor do checkpoint é lido. As tabelas RoPE
(inclusive YaRN e Llama 3) são **corretamente arredondadas** (`Vapor.CR`,
estratégia de Ziv) e coincidem bit a bit com as do `transformers` em 95–97 %
das entradas. As demais diferem porque o `transformers` não arredonda
corretamente, e os digests estão fixados.

**Limitações honestas.**
- O MoE é **denso**: todos os especialistas são computados e a seleção é
  exata. A semântica é a do modelo, mas o custo é `E/k` vezes o necessário.
  O despacho esparso com a mesma garantia é trabalho futuro.
- O MLA guarda K/V completos no cache, sem a compressão latente, que é o
  ganho de memória do MLA.
- Janela deslizante menor que o cache e *soft-capping* de logits de
  atenção (Gemma 2) são recusados pelo nome.
- RoPE dinâmico (NTK) e LongRoPE são recusados.

## 6. Decodificação restrita

`Vapor.Grammar` é uma IR em nível de byte (literais, sequências,
alternativas, repetições, classes, strings JSON, números, referências,
substrings). A execução é um conjunto de configurações ao estilo Thompson.
`Vocab` guarda o vocabulário numa trie e separa os tokens “seguros dentro
de string”, que levam toda configuração de string ilimitada a uma
equivalente. É a mesma ideia dos tokens independentes de contexto do
XGrammar. O custo fica entre 0 e 50 ms por token no vocabulário de 151 936
do Qwen2. O modo `{:lazy, gatilho}` deixa o texto livre até a chamada abrir;
o modo `:strict` restringe tudo.

**O que os testes desta rodada acharam e corrigiram.** O critério que
importa é **nenhum beco sem saída**: todo prefixo aceito tem de poder ser
completado. Um teste de propriedade (passeios aleatórios enviesados para
os bytes delicados) revelou três violações:
1. um escape `\` aceito quando a string já estava no `maxLength`, morto
   quatro bytes depois;
2. surrogates soltos em `\uD800`, que nenhum parser estrito aceita;
3. sequências UTF-8 malformadas: surrogates codificados (`ED A0–BF`) e
   formas *overlong*.

As três estão corrigidas. O escape só começa se couber, um surrogate alto
exige o baixo e é recusado no segundo dígito hexadecimal, e o segundo byte
de `E0/ED/F0/F4` é estreitado pela tabela da RFC 3629. A primeira violação
apareceu ao executar o Livebook: um modelo aleatório produziu `\u35E` e
travou.

## 7. O que não é garantido (resumo)

- **Correção do modelo.** Determinismo e certificados dizem *o que* foi
  computado, não se está certo.
- **Repetição de decisões remotas.** São observações.
- **Exatamente uma vez.** É no máximo uma vez com destinatário idempotente.
- **Ancoragem externa.** O atestado é assinado, mas publicá-lo ou ancorá-lo
  num log de transparência fica com o operador.
- **Divisão e transcendentais corretamente arredondadas.** Na política
  canônica são microprogramas idênticos em todo substrato, a ≤ 1 ulp (÷)
  do valor IEEE. Só `+ − ×` coincidem com IEEE bit a bit (`vapor_nx_test`).
- **Dependência da versão do Unicode do OTP.** É registrada no digest do
  agente, não eliminada.
