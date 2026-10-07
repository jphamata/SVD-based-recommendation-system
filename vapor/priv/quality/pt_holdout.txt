# vapor no ecossistema Elixir/Erlang — escrutínio e integrações

A diretriz pediu para *ponderar* a integração com Phoenix, LiveView, Ecto,
Plug, Nerves, AtomVM, Nx, Livebook, Bumblebee, Axon, Broadway, Membrane,
Oban, EMQX, RabbitMQ e Riak. Ponderar é decidir, para cada item, se ele
resolve uma dor que o vapor tem (ou cria), e não integrar tudo.

## Princípio

O núcleo continua com `deps: []`. Ele é a base confiável: um certificado
vale tanto quanto o código que o produziu, e cada dependência entraria
nessa base. Por isso o núcleo expõe **pontos de extensão sem dependências**
(*behaviours*, ganchos, um despacho independente de transporte), e as
integrações que precisam de pacotes vivem em `integrations/`, cada uma um
projeto Mix próprio. Elas foram compiladas e testadas aqui contra os
**fontes** dos pacotes (clonados do GitHub, porque o Hex não é alcançável
desta máquina) com `VAPOR_ECO=<dir>`. Sem a variável, as dependências vêm
do Hex normalmente.

| integração | versões testadas | testes |
|---|---|---|
| `integrations/vapor_plug` | Plug 1.21.0-dev (master), Bandit 1.12.5, Elixir 1.18.4 | 3: corpos e recibos iguais ao `Vapor.Serve`, streaming, servidor real com chamadas de ferramenta e desconexão |
| `integrations/vapor_nx` | Nx 1.0.0 (master), Elixir 1.18.4 | 5: conversão, `+ − ×` bit a bit iguais ao avaliador do Nx, cotas de ulp, paridade entre substratos, recusas |
| `notebooks/vapor_tour.livemd` | Livebook (formato), Elixir 1.14 e 1.18 | `notebook_test`: as células rodam em ordem e as afirmações do texto são reconferidas |

## Veredito por item

| item | veredito | por quê / o que foi feito |
|---|---|---|
| **Plug** | **feito** (`Vapor.Plug`) | É o ponto de entrada de todo servidor HTTP Elixir. O `Vapor.Serve` foi refeito sobre um despacho independente de transporte (`context/1`, `dispatch/3` com um *responder*), e o `Vapor.Plug` é um responder sobre `Plug.Conn`. TLS, HTTP/2, autenticação e telemetria passam a ser os do endpoint da aplicação. |
| **Phoenix** | **feito, via Plug** | `forward "/llm", Vapor.Plug, name: MyApp.LLM`. O contexto (índice do vocabulário para a decodificação restrita, que é caro) é montado uma vez, na árvore de supervisão. O Phoenix não precisa de mais nada. |
| **LiveView** | **gancho no núcleo + receita** | A dor real é mostrar um agente trabalhando ao vivo. `Vapor.Agent.run(…, on_event: fn e, _ -> Phoenix.PubSub.broadcast(…) end)` entrega cada evento do diário no momento em que é gravado. Os tokens de geração já chegam como mensagens ao processo que pediu (`{:vapor, ref, {:token, …}}`), e uma LiveView é um processo. A receita está abaixo. |
| **Ecto** | ***behaviour* no núcleo + esboço** | `Vapor.Agent.Store` define o contrato (gravar o evento *seq* com exclusividade; carregar só o prefixo que verifica). `Store.File` o implementa com OTP puro e está testado. Um adaptador Ecto/Postgres obtém a exclusividade de `UNIQUE (run_id, seq)`. O esboço está abaixo; não foi executado aqui por falta de Postgres. |
| **Oban** | **o núcleo já entrega o contrato; Oban agenda** | Os workflows duráveis do Oban resolvem “retomar depois de cair”. Com `Store` + `resume/4`, a retomada vem do diário: repete-se o que foi gravado, sem efeitos, e segue-se ao vivo. Um job Oban que chama `Store.resume/4` é a forma de agendar e repetir. A unicidade do job evita trabalho duplicado. Toda ação é anunciada no `Store` antes de acontecer, então dois nós que retomam a mesma execução não anunciam a mesma ação duas vezes. Uma ação já anunciada e ainda em curso não se distingue de uma interrompida, e pode ser tentada de novo com a mesma chave de idempotência. |
| **Broadway** | **receita** | Ingestão em escala com contrapressão: embeddings em lote (`Vapor.Embed.embed/2` aceita listas) e construção de corpus RAG. O ganho específico do vapor é que embeddings e escores densos são **iguais bit a bit** em qualquer nó: um índice construído em paralelo por N máquinas é o mesmo índice. |
| **Nx** | **feito** (`Vapor.Nx.Compiler`) | Um compilador `Nx.Defn` que transforma um `defn` em programa vapor certificado: os mesmos bits em x86 (AVX2, AVX-512), RVV, Vulkan e no oráculo. Para `+ − ×` também são os mesmos bits do avaliador de referência do Nx. Cobre só um **fragmento** (aritmética f32 com *broadcasting*, `exp`, `sigmoid`, `tanh`, `rsqrt`, `max`/`min`, `select` sobre `less`, `dot` contra posto 2, somas e máximos no último eixo, com `keep_axes: true` quando o resultado é reutilizado). O resto é recusado pelo nome da operação, nunca aproximado. É uma ferramenta para quando a reprodutibilidade entre hardwares é o requisito; o EXLA continua sendo a escolha para desempenho geral. |
| **Livebook** | **feito** | `notebooks/vapor_tour.livemd`: programa certificado, MoE de fronteira com invariância a lote, JSON por construção, RAG com recibo, agente que cai e retoma do disco, repetição como prova. É testado como código. |
| **Bumblebee** | **não integrar agora** | Sobrepõe-se ao que o vapor já faz (carregar checkpoints do HF, tokenizar, servir) e chega aos pesos pelos nomes de camada do Axon. O vapor lê o checkpoint HF diretamente, e a ponte de tensores (`Vapor.Nx.from_nx/1`) basta para quem já tem pesos no Nx. Uma ponte Bumblebee→vapor só se justificaria por modelos que o vapor ainda não constrói. |
| **Axon** | **não integrar** | Treino e definição de redes no Axon têm outro objetivo. O treino do vapor (LoRA, destilação) é um programa recorrente certificado. O que se compartilha são tensores (ponte Nx). |
| **Nerves** | **viável; não validado em hardware** | Os workers são binários Zig **estáticos** para aarch64 e riscv64, já construídos e testados sob QEMU. O plano de controle é Elixir sem dependências. Uma imagem Nerves leva os dois sem mudança de código. A atualização OTA de firmware ganha um certificado por modelo, e o dispositivo pode recusar um modelo cujo certificado não tenha o quórum de assinaturas. Falta executar numa placa. |
| **AtomVM** | **não** | O AtomVM não tem *ports* para processos do sistema operacional (o vapor nunca executa código gerado dentro da VM; ele roda em workers isolados) e roda num microcontrolador sem MMU para W^X. O que caberia ali é *verificar* (um atestado, um recibo), e isso depende do suporte a Ed25519/SHA-256 da plataforma. Fica fora. |
| **Membrane** | **não pertinente agora** | O vapor não tem modelos de áudio nem de visão. Encaixar um ASR num pipeline Membrane sem o modelo seria integração de fachada. |
| **EMQX / RabbitMQ** | **transporte, não fonte de verdade** | O gancho `on_event:` publica eventos de diário e recibos num tópico MQTT/AMQP (auditoria em tempo real, dispositivos Nerves reportando execuções). A fonte de verdade continua no `Store`. A fila entrega pelo menos uma vez, e os eventos são idempotentes por `(run_id, seq, hash)`, de modo que um consumidor deduplica sem estado extra. |
| **Riak** | **não recomendado** | O Riak KV é mantido pela comunidade desde o fim da Basho (2017). Para o que o vapor guarda (pesos e unidades endereçados por conteúdo, diários só-acréscimo), um armazenamento de objetos endereçado por hash (S3/MinIO) ou o próprio Postgres servem melhor e são mais fáceis de operar. A escolha de um banco novo não deve vir de uma lista de marcas. |

## O que a integração revelou no núcleo

Integrar de verdade, e não só descrever, expôs problemas que estavam
escondidos:

1. **Um cliente que desconecta deixava a sequência gerando até
   `max_tokens`.** O motor passou a monitorar o processo destinatário de
   cada requisição. Se ele morre (LiveView, job), as suas sequências saem
   do lote no passo seguinte, e as páginas de KV voltam para o *pool*. Um
   cliente HTTP de *streaming* que desconecta é cancelado na escrita
   seguinte. Uma resposta sem *streaming* não escreve nada até o fim,
   então roda até `max_tokens`. Também existe
   `Vapor.Engine.cancel/2`, e a `Pool` repassa o cancelamento à réplica.
   O teste revelou um defeito latente: um `:step` já agendado podia chegar
   com o lote vazio, e o motor quebrava. Está corrigido e testado (`engine_test`,
   `serve_test`, `vapor_plug_test`).
2. **O Mix ≥ 1.15 poda o *code path*.** `:httpc` sumia em projetos que
   usam o vapor como dependência (`module :http_util is not available`). O
   núcleo agora declara `:inets`, `:ssl` e `:public_key`, que são
   aplicações do próprio OTP, em `extra_applications`. A suíte em Elixir
   1.18 também acusava o mesmo problema.
3. **O monitor do motor era por PID.** Motores registrados por nome (o
   normal numa árvore de supervisão) não eram monitorados. Agora o monitor
   usa `GenServer.whereis/1` e casa a referência exata. Num processo de
   transporte de vida longa (Plug), mensagens `:DOWN` de outros monitores
   não são confundidas com a queda do motor.
4. **Uma escrita num worker recém-morto derrubava o dono do *port*.**
   Rodar a suíte em Elixir 1.18, com outro *timing*, revelou uma corrida
   antiga. Se o worker morre entre duas requisições e a requisição seguinte
   chega antes da mensagem `exit_status`, a escrita falha com EPIPE. Como o
   *port* é ligado ao processo dono, isso virava um sinal de saída que
   derrubava o dono, o motor e quem estivesse ligado a eles. Worker, fabric
   e cliente MCP agora interceptam saídas (`trap_exit`), e a falha volta a
   ser uma resposta (`:worker_crashed`), como o projeto promete. O cliente
   MCP também deixou de terminar quando o seu servidor morre: as chamadas
   passam a falhar com o motivo.
5. **Ordem de propriedades em JSON Schema.** Um `Plug.Parsers` antes do
   `Vapor.Plug` entrega um mapa, que perdeu a ordem das chaves. A gramática
   segue a ordem do esquema (`required` primeiro), então a saída continua
   válida, mas a ordem dos campos opcionais vira alfabética. A
   documentação manda montar antes do *parser*, e o adaptador recodifica
   quando o corpo já foi lido.

## Receitas (esboços — não executados aqui)

Estes trechos usam pacotes que não estão nesta máquina (Postgres, Phoenix
PubSub, Oban). Eles mostram o encaixe com os pontos de extensão testados
acima, sem afirmar que foram executados.

**Store em Postgres via Ecto**: a exclusividade vem do índice único.

```elixir
# migração
create table(:vapor_events, primary_key: false) do
  add :run_id, :string, null: false
  add :seq, :integer, null: false
  add :event, :binary, null: false        # Vapor.Canonical.encode(event)
end
create unique_index(:vapor_events, [:run_id, :seq])

defmodule MyApp.EctoStore do
  @behaviour Vapor.Agent.Store
  defstruct [:repo]
  import Ecto.Query

  @impl true
  def append(%{repo: repo}, run_id, %{"seq" => seq} = e) do
    case repo.insert_all("vapor_events", [%{run_id: run_id, seq: seq, event: Vapor.Canonical.encode(e)}], on_conflict: :nothing) do
      {1, _} -> :ok
      {0, _} -> {:error, :conflict}
    end
  end

  @impl true
  def load(%{repo: repo}, run_id) do
    rows = repo.all(from e in "vapor_events", where: e.run_id == ^run_id, order_by: e.seq, select: e.event)
    if rows == [], do: {:error, :not_found}, else: {:ok, Vapor.Agent.Store.rebuild(run_id, rows)}
  end

  @impl true
  def runs(%{repo: repo}), do: repo.all(from e in "vapor_events", distinct: true, select: e.run_id)
end
```

**LiveView acompanhando um agente**

```elixir
def handle_event("ask", %{"q" => q}, socket) do
  topic = "run:" <> Base.encode16(:crypto.strong_rand_bytes(8))
  Phoenix.PubSub.subscribe(MyApp.PubSub, topic)
  hook = fn e, _j -> Phoenix.PubSub.broadcast(MyApp.PubSub, topic, {:vapor_event, e}) end
  Task.start(fn -> Vapor.Agent.Store.run(store(), spec(), q, backend: backend(), impls: impls(), on_event: hook) end)
  {:noreply, assign(socket, events: [])}
end

def handle_info({:vapor_event, e}, socket), do: {:noreply, update(socket, :events, &(&1 ++ [e]))}
```

**Oban retomando o que caiu**

```elixir
defmodule MyApp.ResumeRuns do
  use Oban.Worker, queue: :agents, unique: [keys: [:run_id]]

  @impl true
  def perform(%Oban.Job{args: %{"run_id" => id}}) do
    case Vapor.Agent.Store.resume(store(), spec(), id, backend: backend(), impls: impls()) do
      {:ok, _} -> :ok
      {:error, {:on_event, :conflict}, _} -> {:cancel, "another node is running it"}
      {:error, {:diverged, _} = why, _} -> {:cancel, inspect(why)}   # não repetir às cegas
      {:error, why, _} when why in [:history_redacted, :empty_journal] -> {:cancel, inspect(why)}
      {:error, why, _} -> {:error, why}
      {:error, :not_found} -> {:cancel, "no such run"}
    end
  end
end

# na partida: for id <- Vapor.Agent.Store.unfinished(store()), do: Oban.insert(MyApp.ResumeRuns.new(%{run_id: id}))
```

**Broadway construindo um corpus**

```elixir
def handle_batch(:default, messages, _info, _ctx) do
  texts = Enum.map(messages, & &1.data.text)
  {:ok, vectors} = Vapor.Embed.embed(embedder(), texts)   # mesmos bits em qualquer nó
  # gravar (doc_id, vetor) e a raiz Merkle do lote
  messages
end
```

## Como rodar as integrações

```sh
cd integrations/vapor_plug && mix deps.get && mix test     # com Hex
cd integrations/vapor_nx   && mix deps.get && mix test

# sem Hex: fontes lado a lado (plug, mime, plug_crypto, telemetry, bandit,
# thousand_island, hpax, websock, nx, complex) e o archive do Hex compilado
# do GitHub só para o SCM
VAPOR_ECO=/caminho/dos/fontes HEX_OFFLINE=1 mix test
```

Os dois projetos precisam do worker nativo construído no núcleo
(`make native`).
