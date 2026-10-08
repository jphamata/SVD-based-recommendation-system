# vapor in the Elixir/Erlang ecosystem — scrutiny and integrations

The directive asked us to *weigh* integration with Phoenix, LiveView, Ecto,
Plug, Nerves, AtomVM, Nx, Livebook, Bumblebee, Axon, Broadway, Membrane,
Oban, EMQX, RabbitMQ and Riak. Weighing means deciding, for each item, whether
it solves a pain that vapor has (or creates), not integrating everything.

## Principle

The core stays at `deps: []`. It is the trusted base: a certificate
is worth as much as the code that produced it, and every dependency would enter
that base. So the core exposes **dependency-free extension points**
(*behaviours*, hooks, a transport-independent dispatch), and the
integrations that need packages live in `integrations/`, each its own
Mix project. They were compiled and tested here against the packages'
**sources** (cloned from GitHub, because Hex is not reachable
from this machine) with `VAPOR_ECO=<dir>`. Without the variable, the dependencies come
from Hex as usual.

| integration | versions tested | tests |
|---|---|---|
| `integrations/vapor_plug` | Plug 1.21.0-dev (master), Bandit 1.12.5, Elixir 1.18.4 | 3: bodies and receipts identical to `Vapor.Serve`, streaming, real server with tool calls and disconnection |
| `integrations/vapor_nx` | Nx 1.0.0 (master), Elixir 1.18.4 | 5: conversion, `+ − ×` bit-for-bit identical to the Nx evaluator, ulp bounds, parity across substrates, refusals |
| `notebooks/vapor_tour.livemd` | Livebook (format), Elixir 1.14 and 1.18 | `notebook_test`: the cells run in order and the claims in the text are re-checked |

## Verdict per item

| item | verdict | why / what was done |
|---|---|---|
| **Plug** | **done** (`Vapor.Plug`) | It is the entry point of every Elixir HTTP server. `Vapor.Serve` was rebuilt on a transport-independent dispatch (`context/1`, `dispatch/3` with a *responder*), and `Vapor.Plug` is a responder over `Plug.Conn`. TLS, HTTP/2, authentication and telemetry become those of the application's endpoint. |
| **Phoenix** | **done, via Plug** | `forward "/llm", Vapor.Plug, name: MyApp.LLM`. The context (the vocabulary index for constrained decoding, which is expensive) is built once, in the supervision tree. Phoenix needs nothing else. |
| **LiveView** | **hook in the core + recipe** | The real pain is showing an agent working live. `Vapor.Agent.run(…, on_event: fn e, _ -> Phoenix.PubSub.broadcast(…) end)` delivers each journal event at the moment it is written. Generation tokens already arrive as messages to the process that asked (`{:vapor, ref, {:token, …}}`), and a LiveView is a process. The recipe is below. |
| **Ecto** | ***behaviour* in the core + sketch** | `Vapor.Agent.Store` defines the contract (write event *seq* exclusively; load only the prefix that verifies). `Store.File` implements it with pure OTP and is tested. An Ecto/Postgres adapter gets exclusivity from `UNIQUE (run_id, seq)`. The sketch is below; it was not run here for lack of Postgres. |
| **Oban** | **the core already delivers the contract; Oban schedules** | Oban's durable workflows solve “resume after a crash”. With `Store` + `resume/4`, resumption comes from the journal: what was recorded is replayed, without effects, and execution continues live. An Oban job that calls `Store.resume/4` is the way to schedule and retry. Job uniqueness avoids duplicate work. Every action is announced in the `Store` before it happens, so two nodes that resume the same run do not announce the same action twice. An action already announced and still in progress cannot be told apart from an interrupted one, and may be tried again with the same idempotency key. |
| **Broadway** | **recipe** | Ingestion at scale with backpressure: batched embeddings (`Vapor.Embed.embed/2` accepts lists) and RAG corpus construction. vapor's specific gain is that embeddings and dense scores are **bit-for-bit identical** on any node: an index built in parallel by N machines is the same index. |
| **Nx** | **done** (`Vapor.Nx.Compiler`) | An `Nx.Defn` compiler that turns a `defn` into a certified vapor program: the same bits on x86 (AVX2, AVX-512), RVV, Vulkan and in the oracle. For `+ − ×` they are also the same bits as Nx's reference evaluator. It covers only a **fragment** (f32 arithmetic with *broadcasting*, `exp`, `sigmoid`, `tanh`, `rsqrt`, `max`/`min`, `select` over `less`, `dot` against rank 2, sums and maxima over the last axis, with `keep_axes: true` when the result is reused). The rest is refused by operation name, never approximated. It is a tool for when reproducibility across hardware is the requirement; EXLA remains the choice for general performance. |
| **Livebook** | **done** | `notebooks/vapor_tour.livemd`: certified program, frontier MoE with batch invariance, JSON by construction, RAG with a receipt, an agent that crashes and resumes from disk, replay as proof. It is tested as code. |
| **Bumblebee** | **do not integrate now** | It overlaps with what vapor already does (loading HF checkpoints, tokenizing, serving) and reaches the weights through Axon's layer names. vapor reads the HF checkpoint directly, and the tensor bridge (`Vapor.Nx.from_nx/1`) is enough for anyone who already has weights in Nx. A Bumblebee→vapor bridge would only be justified by models that vapor does not build yet. |
| **Axon** | **do not integrate** | Training and network definition in Axon have a different goal. vapor's training (LoRA, distillation) is a certified recurrent program. What is shared is tensors (the Nx bridge). |
| **Nerves** | **viable; not validated on hardware** | The workers are **static** Zig binaries for aarch64 and riscv64, already built and tested under QEMU. The control plane is dependency-free Elixir. A Nerves image carries both without code changes. OTA firmware updates gain a certificate per model, and the device can refuse a model whose certificate lacks the signature quorum. Running it on a board is still missing. |
| **AtomVM** | **no** | AtomVM has no *ports* to operating-system processes (vapor never executes generated code inside the VM; it runs in isolated workers) and runs on a microcontroller without an MMU for W^X. What would fit there is *verifying* (an attestation, a receipt), and that depends on the platform's Ed25519/SHA-256 support. It stays out. |
| **Membrane** | **not pertinent now** | vapor has no audio or vision models. Fitting an ASR into a Membrane pipeline without the model would be a façade integration. |
| **EMQX / RabbitMQ** | **transport, not source of truth** | The `on_event:` hook publishes journal events and receipts to an MQTT/AMQP topic (real-time auditing, Nerves devices reporting runs). The source of truth stays in the `Store`. The queue delivers at least once, and events are idempotent by `(run_id, seq, hash)`, so a consumer deduplicates without extra state. |
| **Riak** | **not recommended** | Riak KV has been community-maintained since the end of Basho (2017). For what vapor stores (content-addressed weights and units, append-only journals), a hash-addressed object store (S3/MinIO) or Postgres itself serves better and is easier to operate. The choice of a new database should not come from a list of brands. |

## What the integration revealed in the core

Integrating for real, and not just describing, exposed problems that were
hidden:

1. **A client that disconnected left its sequence generating until
   `max_tokens`.** The engine now monitors the recipient process of
   each request. If it dies (LiveView, job), its sequences leave
   the batch at the next step, and the KV pages return to the *pool*. A
   *streaming* HTTP client that disconnects is cancelled at the next
   write. A non-*streaming* response writes nothing until the end,
   so it runs until `max_tokens`. There is also
   `Vapor.Engine.cancel/2`, and the `Pool` forwards the cancellation to the replica.
   The test revealed a latent defect: an already scheduled `:step` could arrive
   with an empty batch, and the engine crashed. It is fixed and tested (`engine_test`,
   `serve_test`, `vapor_plug_test`).
2. **Mix ≥ 1.15 prunes the *code path*.** `:httpc` disappeared in projects that
   use vapor as a dependency (`module :http_util is not available`). The
   core now declares `:inets`, `:ssl` and `:public_key`, which are
   OTP's own applications, in `extra_applications`. The suite on Elixir
   1.18 also flagged the same problem.
3. **The engine monitor was by PID.** Engines registered by name (the
   normal case in a supervision tree) were not monitored. Now the monitor
   uses `GenServer.whereis/1` and matches the exact reference. In a long-lived
   transport process (Plug), `:DOWN` messages from other monitors
   are not mistaken for the engine going down.
4. **A write to a worker that had just died took down the *port* owner.**
   Running the suite on Elixir 1.18, with different *timing*, revealed an
   old race. If the worker dies between two requests and the next request
   arrives before the `exit_status` message, the write fails with EPIPE. Since the
   *port* is linked to the owner process, this became an exit signal that
   took down the owner, the engine and whatever was linked to them. Worker, fabric
   and MCP client now trap exits (`trap_exit`), and the failure is once again
   a response (`:worker_crashed`), as the project promises. The MCP client
   also no longer terminates when its server dies: calls
   now fail with the reason.
5. **Property order in JSON Schema.** A `Plug.Parsers` before
   `Vapor.Plug` delivers a map, which has lost the key order. The grammar
   follows the schema's order (`required` first), so the output remains
   valid, but the order of the optional fields becomes alphabetical. The
   documentation says to mount before the *parser*, and the adapter re-encodes
   when the body has already been read.

## Recipes (sketches — not run here)

These snippets use packages that are not on this machine (Postgres, Phoenix
PubSub, Oban). They show how things fit with the extension points tested
above, without claiming that they were run.

**Store in Postgres via Ecto**: exclusivity comes from the unique index.

```elixir
# migration
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

**LiveView following an agent**

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

**Oban resuming what crashed**

```elixir
defmodule MyApp.ResumeRuns do
  use Oban.Worker, queue: :agents, unique: [keys: [:run_id]]

  @impl true
  def perform(%Oban.Job{args: %{"run_id" => id}}) do
    case Vapor.Agent.Store.resume(store(), spec(), id, backend: backend(), impls: impls()) do
      {:ok, _} -> :ok
      {:error, {:on_event, :conflict}, _} -> {:cancel, "another node is running it"}
      {:error, {:diverged, _} = why, _} -> {:cancel, inspect(why)}   # do not retry blindly
      {:error, why, _} when why in [:history_redacted, :empty_journal] -> {:cancel, inspect(why)}
      {:error, why, _} -> {:error, why}
      {:error, :not_found} -> {:cancel, "no such run"}
    end
  end
end

# at startup: for id <- Vapor.Agent.Store.unfinished(store()), do: Oban.insert(MyApp.ResumeRuns.new(%{run_id: id}))
```

**Broadway building a corpus**

```elixir
def handle_batch(:default, messages, _info, _ctx) do
  texts = Enum.map(messages, & &1.data.text)
  {:ok, vectors} = Vapor.Embed.embed(embedder(), texts)   # same bits on any node
  # store (doc_id, vector) and the batch's Merkle root
  messages
end
```

## How to run the integrations

```sh
cd integrations/vapor_plug && mix deps.get && mix test     # with Hex
cd integrations/vapor_nx   && mix deps.get && mix test

# without Hex: sources side by side (plug, mime, plug_crypto, telemetry, bandit,
# thousand_island, hpax, websock, nx, complex) and the Hex archive built
# from GitHub only for the SCM
VAPOR_ECO=/path/to/sources HEX_OFFLINE=1 mix test
```

Both projects need the native worker built in the core
(`make native`).
