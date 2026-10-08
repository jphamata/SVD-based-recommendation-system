# Immutable agents, verifiable RAG and frontier models

This document records the scrutiny of this round's directive, the design
that came out of it and — with the same weight — what is **not** guaranteed.
Everything that appears as fact here has a test in the repository. The
test's name goes in parentheses.

## 1. The directive, examined

> *“refined zip + support for frontier models + agents + rag + whatever else
> is pertinent + reflect on immutable agents → this directive itself
> is subject to refinement and scrutiny → the result must solve the
> real pains of industry and academia with real innovation, lateral thinking
> and first principles”* (translated)

Each term of the directive admits more than one reading. The reading chosen
decides what is worth building.

**“Frontier models” has two readings, and both hold.** The first
is to run locally, with vapor's guarantee, the *architectures* of the open
frontier families: Qwen3, Qwen3-MoE, Mixtral, Gemma 3 and DeepSeek-V3
(MLA + MoE). The second is to use *hosted models* (Claude, GPT, or
any OpenAI-compatible server) as an agent's brain. Both
are implemented, but they are of different natures, and the design makes
that difference explicit. A decision by a local model is **re-derivable**:
it is a pure function of weights, tokens, parameters and seed, with the same bits on
any substrate. A decision by a remote model is an **observation**: it can be
recorded and protected against tampering, but not reproduced. OpenAI's
seed is declared to be “best effort”, and Anthropic's Messages API
offers no seed. No text in this project promises
otherwise.

**“Agents” does not call for one more orchestration loop.** Agent
frameworks exist by the dozen, and one more `think → call tool →
observe` loop solves no pain at all. What vapor has that is unique is
**determinism**: invariance to batch, to threads and to substrate, with a
certificate. The first-principles question becomes a different one: *what
can an agent be when its model is a function?* The answer steers
everything else. A run stops being a log and becomes an **object of
proof**, which anyone with the same weights can replay and check.

**“RAG” through the same lens.** Nobody needs another vector index. The
real pains are three: the result does not reproduce, nobody proves which
corpus the answer came from, and the model invents citations. The RAG here attacks
those three.

**“Immutable agents” has at least five readings**, and each one has a
limit:

| reading | what it delivers | what it does *not* deliver |
|---|---|---|
| the agent's definition is a value (content-addressed, with lineage) | knowing exactly *which* agent acted; comparable versions | correctness: an immutable agent can be immutably wrong |
| the run is append-only and tamper-evident | audit, attribution, proof of inclusion of an event | truth of the observed facts: what the tool said is recorded, not what the world was |
| the agent does not change its own capabilities mid-run | prompt injection does not escalate privilege | that the model does not *try*; the attempt is recorded and denied |
| replaying is proving | independent verification of local decisions and of pure tools | re-execution of the world: `observe`/`act` are never re-executed |
| immutable *versus* the right to be forgotten | erasure by key destruction (*crypto-shredding*) | erasure of structural metadata: tool names and the shape of the run remain |

The last row is the conflict that the word “immutable” usually hides. An
immutable log of personal data collides with Art. 17 of the GDPR and Art. 18 of the
LGPD. The resolution does not require choosing between audit and privacy: the
content is encrypted per data subject, and only the ciphertext is chained. Destroying the key
erases the content everywhere, backups included, and the chain, the Merkle
root and the proofs remain valid.

**What the directive did not ask for and the pains do.** Does a run survive
the process crashing? Does an external action (e-mail, payment) happen at
most once when the agent crashes in the middle of it? Who guarantees that the
stored journal was not rewritten *by whoever stores it*? §3 answers the
three questions.

## 2. Real pains → pieces

| pain | context | what vapor does | evidence |
|---|---|---|---|
| malformed tool calls, invented names, arguments outside the schema (*retry* loops) | common even in large models; worse in small ones | the call is decoded under a grammar derived from the declared schemas: declared name, valid arguments **by construction** | `agentic_serve_test` (`tool_choice: "required"` with a random-weights model: 100% valid) |
| structured output that is sometimes not valid JSON | `response_format` with `json_schema` | JSON Schema grammar (declared subset; the rest is refused by name, or accepted with `lenient: true`) | `grammar_test`, `agentic_serve_test` |
| valid JSON with invalid fields (a CEP postal code without its hyphen, 30 February, an e-mail without a domain) | `pattern` and `format` were refused | since 0.7.0, `pattern` (ECMA-262) and `format` (`date`, `time`, `date-time` with the real calendar, `uuid`, `ipv4`, `email`, `hostname`) become byte automata: only matching values come out | `regex_test` (same verdict as Python's `re` and the standard library's *parsers* on 7,240 strings) |
| LLM results not reproducible, not even with a seed | the batch changes the bits of the usual *kernels* (He et al., Thinking Machines, 2025: *Defeating Nondeterminism in LLM Inference*) | invariance to batch and to substrate **already was** a property of vapor; each response carries a receipt (`x-vapor-receipt`) = canonical digest of model, prompt ids, parameters and generated ids | `serve_test`, `engine_test` |
| academic reproducibility | whoever has the weights should be able to redo the paper's number | receipts + certified execution; a Livebook that runs as a test | `notebook_test` |
| hallucinated citations in RAG | the model “cites” what is not in the source | inside `<quote src="i">…</quote>` only bytes that continue a substring of source *i* can come out (suffix automaton); `check_citations` re-checks answers from any origin | `rag_test` (random model: only cites literally) |
| “which corpus did this come from?” | index versions, reindexing on other hardware | corpus = Merkle root (RFC 6962); BM25 with a correctly rounded log; dense scores as a certified program; RRF in exact rationals; re-verifiable retrieval receipt | `rag_test` |
| event-logging obligation | EU AI Act, Art. 12 (automatic logging in high-risk systems) and Art. 19 (retention) | hash-chained journal + Merkle root + **Ed25519 attestation** from the node that executed; proof of inclusion of one event without revealing the others | `agent_test` |
| erasure of personal data in immutable logs | GDPR Art. 17, LGPD Art. 18 | *crypto-shredding* per data subject: input, texts, tokens, arguments, results and errors sealed; structure readable for audit | `agent_test` (“erasure”) |
| prompt injection leading to actions | text read by a tool can instruct the model | capabilities (`grants`) are part of the agent's digest; an `act` tool without a grant is denied, and the denial is recorded; `confirm:` records the human approval | `agent_test` |
| agent crashes mid-flow; duplicated action | e-mail sent twice, double charge | journal written before the next step (*write-ahead*); every action is announced (`intent`) and written **before** it happens; resuming is replaying (nothing that was written is repeated); idempotency key stable per (run, step, call); of two resumers, only one announces each action | `agent_store_test` |
| the wrong chat template silently degrades the model | each model brings its own Jinja | hermetic Jinja interpreter, byte for byte equal to Hugging Face's `jinja2` | `template_test` (240 renderings of 20 real templates + 40 snippets) |
| client disconnects, GPU keeps generating for nobody | wasted batch | the engine monitors the recipient process: if it dies (LiveView closed, job dead), its sequences leave the batch on the next step, across replicas too; over HTTP, a *streaming* client that disconnects is cancelled on the next write (TCP and Plug). A non-*streaming* request runs to the end (bounded by `max_tokens`) | `engine_test`, `serve_test`, `vapor_plug_test` |
| vendor lock-in | switching provider rewrites the agent | the same agent runs with a local model, any OpenAI-compatible server (including vapor itself) or the Messages API; MCP tools | `agent_test` (AgentBackendsTest) |

## 3. The design

```
Spec (value, digest, lineage)  ──run──▶ Journal (chained events, Merkle root) ──attest──▶ Ed25519 signature
   │ tools: pure | observe | act          │  start · model · tool · intent · tool · … · final|halt
   │ grants, policy (seed, max_steps)     │
   ▼                                      ▼
Backend: Local (re-derivable) | OpenAI | Anthropic (observation)     Store (write-ahead, fencing) ──resume──▶ continues
```

**Spec** (`Vapor.Agent.Spec`). Name, instructions, model, tools with
effect class and schema, grants and policy (seed, maximum steps,
temperature). The digest is the canonical hash (deterministic CBOR, RFC 8949
§4.2) of the whole value, and the environment enters it: OTP's Unicode version
and the canonical encoding's version. Two BEAMs with different Unicode tables
normalize text differently, and that cannot go unnoticed.
`evolve/2` creates the next version with `parent` pointing to the previous one.
A run of v1 is not accepted as a run of v2.

**Effect classes.** Three classes cover the behavior of any
tool under replay:
- `pure`: a function of the arguments. On replay it is **re-executed** and must
  agree with the record.
- `observe`: reads the world. On replay the result is **read** from the record.
- `act`: changes the world. Only runs with a grant; on replay it **never** runs;
  live, it receives the idempotency key.

MCP tools arrive as `act` when the operator does not declare the class.
MCP does not say what a tool does, and the most restrictive class is the
safe default.

**Journal.** `hashᵢ = SHA-256(canonical({hashᵢ₋₁, i, kind, data}))`, with an
RFC 6962 root over the hashes. Any language can recompute it, because the
encoding is canonical CBOR, checked against Python's `cbor2`. A
hash chain only proves integrity **relative to a head that is already
trusted**: whoever stores the journal can rewrite it entirely and re-chain it.
That is why `Journal.attest/2` exists. The node that executed signs (id, number of
events, head, root), and a re-chained journal stops being attested
(`agent_test`). Publishing the attestation, or anchoring it in a transparency log,
closes the door.

**Replaying is proving** (`Vapor.Agent.replay/3`). Replay rebuilds the
run from the journal. Each local decision is **recomputed** and must
give the same tokens. Invariance across substrates is what makes this a
verification done on *another* machine, and not just on the same one. `pure`
tools are re-executed; `observe`, `act` and remote decisions are read;
action announcements (`intent`) are re-derived and checked. At the end, the
rebuilt journal must end at the same head. Replay does not touch
the world, and no `on_event:` hook fires during it.

**Resuming is replaying; durability without a workflow engine**
(`Vapor.Agent.Store`). The `on_event:` hook runs after each new event
and before the next step. It is the *write-ahead* point: if the storage
refuses, the run stops before acting. Every `act` tool is preceded by
an `intent` event (call, idempotency key, approval), written
**before** the action happens. `Store.resume/4` replays the written prefix
without effects and only continues live if the prefix **verified**: another spec, a
different recomputed decision or an out-of-order event make the resumption
stop (`{:error, {:diverged, …}}`) before any new event. An
erased history can be verified as recorded, but never
continued. There are two crash windows:
1. *before* the announcement: the action did not happen and happens once on resumption;
2. *after* the announcement and before the result: the action may have happened.
   The resumption writes a `retry` event and repeats it with the **same**
   idempotency key; a recipient that honors the key applies it only once.

The guarantee is **at most once with an idempotent recipient**, not
“exactly once” in general. No distributed system offers more
than that (`agent_store_test` covers both windows). `Store.File` writes
each event to an exclusive temporary file, does `fsync` and creates a *hard link* with
the final name. `link(2)` is atomic and fails if the name exists, and that gives two
properties at once: a half-written event is never read, and two
resumers do not write the same event. Since the announcement (or the `retry`) is
written before the action, **of two concurrent resumers only one announces
each action**; the other receives `:conflict` before touching the world. The
honest limit: an action already announced and still in progress cannot be told apart from one
interrupted by the crash. A second resumer that arrives in that interval
writes its `retry` and tries it again, with the same idempotency key.
Closing that case would require a *lease* with a clock, which remains
future work. `pure` and `observe` tools are not announced, and
the loser may have executed them before the conflict, which is harmless by
definition. A relational adapter gets the same with
`UNIQUE (run_id, seq)`.

This is the contract of Temporal, Step Functions or Oban workflows,
obtained from two facts vapor already had (determinism and a chained
journal) instead of a scheduler. One difference matters: Temporal requires
the workflow *code* to be deterministic, and vapor extends
determinism to the *model's decision*.

**Capabilities.** An `act` tool without a grant is denied, and the denial
goes into the journal. Text read by a tool can convince the model to
*try*, but grants nothing: the grants are in the digest, fixed before
the run. `confirm:` puts a human in the path, and their decision
also goes into the journal: a refusal in the result (`"not approved"`), an
approval in the announcement (`"approval" => "approved"`).

**Erasure** (`Journal.seal/4`, `Keys`). Values are encrypted with
AES-256-GCM per data subject, with the run id as associated data. The nonce
is derived from key, run and value, so sealing is deterministic and
replay reproduces the same hashes. Sealed: input, model
texts, local model tokens, call arguments, results and errors.
Readable: event types, tool names, effect classes,
ids. A ciphertext that does not open with the existing key is reported as a
divergence (`:sealed`), not as erasure. The receipts and the local model's prompt digest are also sealed,
because a hash of a short text confirms a guess. After
`Keys.shred/2`, replay reports the events as `redacted` instead of
failing, and the data subject is marked as erased: sealing for them again is
refused, instead of creating a new key that would make the old ciphertext look
tampered with. In production the keys live in a KMS/HSM; the
contract has the same three operations.

**Backends.** `Local` renders with the model's own template and constrains
calls to the schemas. `OpenAI` serves any compatible server
(OpenAI, vLLM, llama.cpp, Ollama or `Vapor.Serve`; in that case the receipt
makes the remote model verifiable by whoever has the weights). `Anthropic` speaks
the Messages API (`tool_use`/`tool_result`). MCP comes in through its own stdio
client (`2025-06-18`), tested against the official SDK.

## 4. Verifiable RAG

`Vapor.RAG.corpus/2` normalizes the text (NFC), splits it into chunks with offsets
and makes each chunk a Merkle leaf. With that:

- the **corpus is a value**: the root names exactly what was indexed;
- **retrieval is a function**: BM25 in binary64 with a fixed order and a
  correctly rounded logarithm; dense scores as a certified `linear`
  in the worker (same bits on every substrate, as are the
  embeddings of `Vapor.Embed`); RRF fusion in exact rationals; tie-break
  by id;
- the receipt `(root, method, k, query, embedder, ids)` is re-verifiable
  with `verify/2`, and each chunk carries its proof of inclusion.

Citations are literal **by construction**. The model writes freely,
but inside `<quote src="i">` it can only continue a substring of source
*i*. This does not guarantee that the citation *supports* the claim; it guarantees that
it *exists* in the source. Judging support remains the work of
the reader (or of another model), now with the certainty that the quoted text is
real.

## 5. Frontier models, locally

| family | what it requires | how it came in |
|---|---|---|
| Qwen3 | per-head RMS norms on q/k | `linear` with 0/1 selection matrices (exact contraction) |
| Qwen3-MoE, Mixtral | top-k router | rank by counting + `sel`; experts in fixed order |
| Gemma 3 | GeGLU (tanh), `(1+w)` norms, sandwich norms, √d in the embedding, local/global RoPE per layer, `query_pre_attn_scalar` | new canonical functions `tanh` and `gelu_tanh`; explicit attention scale in the term |
| DeepSeek-V3 | MLA, MoE with groups and sigmoid, shared experts, YaRN with `mscale_all_dim`, `rope_interleave` | MLA as a head layout (pass-through RoPE pairs, interleaving as a permutation); groups by exact selection |

The results come from `frontier_test`. The logits are within ≤ 8·10⁻⁷ (relative)
of `transformers` 5.18 in the 8 variants. The bits are identical on AVX2,
AVX-512, the RVV interpreter, QEMU aarch64/riscv64 and Vulkan. Prefill
equals step-by-step decode. Poisoning *unchosen* experts with
NaN/∞ does not change a bit, and every tensor in the checkpoint is read. The RoPE tables
(YaRN and Llama 3 included) are **correctly rounded** (`Vapor.CR`,
Ziv's strategy) and match `transformers`'s bit for bit in 95–97%
of the entries. The rest differ because `transformers` does not round
correctly, and the digests are pinned.

**Honest limitations.**
- The MoE is **dense**: all experts are computed and the selection is
  exact. The semantics are the model's, but the cost is `E/k` times what is necessary.
  Sparse dispatch with the same guarantee is future work.
- The MLA stores full K/V in the cache, without the latent compression, which is
  MLA's memory gain.
- A sliding window smaller than the cache and attention logit *soft-capping*
  (Gemma 2) are refused by name.
- Dynamic RoPE (NTK) and LongRoPE are refused.

## 6. Constrained decoding

`Vapor.Grammar` is a byte-level IR (literals, sequences,
alternatives, repetitions, classes, JSON strings, numbers, references,
substrings). Execution is a set of Thompson-style configurations.
`Vocab` stores the vocabulary in a trie and separates the “safe inside a
string” tokens, which take every unbounded string configuration to an
equivalent one. It is the same idea as XGrammar's context-independent tokens.
The cost is between 0 and 50 ms per token on Qwen2's 151,936-token
vocabulary. The `{:lazy, trigger}` mode leaves text free until the call opens;
the `:strict` mode constrains everything.

**What this round's tests found and fixed.** The criterion that
matters is **no dead end**: every accepted prefix must be
completable. A property test (random walks biased towards
the delicate bytes) revealed three violations:
1. a `\` escape accepted when the string was already at `maxLength`, dead
   four bytes later;
2. lone surrogates in `\uD800`, which no strict parser accepts;
3. malformed UTF-8 sequences: encoded surrogates (`ED A0–BF`) and
   *overlong* forms.

All three are fixed. The escape only starts if it fits, a high surrogate
requires the low one and is refused at the second hex digit, and the second byte
of `E0/ED/F0/F4` is narrowed by the RFC 3629 table. The first violation
showed up when running the Livebook: a random model produced `\u35E` and
got stuck.

## 7. What is not guaranteed (summary)

- **Model correctness.** Determinism and certificates say *what* was
  computed, not whether it is right.
- **Replay of remote decisions.** They are observations.
- **Exactly once.** It is at most once with an idempotent recipient.
- **External anchoring.** The attestation is signed, but publishing it or anchoring it
  in a transparency log is left to the operator.
- **Correctly rounded division and transcendentals.** In the canonical policy
  they are microprograms identical on every substrate, within ≤ 1 ulp (÷)
  of the IEEE value. Only `+ − ×` match IEEE bit for bit (`vapor_nx_test`).
- **Dependence on OTP's Unicode version.** It is recorded in the
  agent's digest, not eliminated.
