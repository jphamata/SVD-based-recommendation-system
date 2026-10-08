# Transparency: what was signed cannot be silently rewritten

An Ed25519 certificate proves that **someone signed**. It does not prove
that this person did not sign something else for another audience: an
operator can show an auditor one history and the users another (*split
view*), or erase yesterday's receipt that is inconvenient today. The
industry's answer to this — Certificate Transparency, Go's *checksum
database*, Sigstore's Rekor — is not a blockchain: it is an **append-only
log with witnesses**.

`Vapor.Tlog` is that log, in the formats those tools already speak, so
that a log kept by vapor can be witnessed and audited by programs that
have never heard of vapor.

## 1. The pieces

| piece | what it is | standard |
|---|---|---|
| tree | Merkle with `SHA-256(0x00‖entry)` at the leaves and `SHA-256(0x01‖l‖r)` at the nodes | RFC 9162 (the same hashing as `Vapor.Merkle`) |
| inclusion proof | entry `i` is in the tree of size `n` with root `r` | RFC 9162 §2.1.3 |
| consistency proof | the tree of `n` entries **extends** the one of `m` — nothing was removed or rewritten | RFC 9162 §2.1.4 |
| *checkpoint* | origin, size, root in base64, signed | C2SP `tlog-checkpoint` in `signed-note` (Ed25519) |
| co-signature | a witness attests that it saw this *checkpoint* **and** the consistency with the previous one | C2SP `tlog-cosignature/v1` |
| file | size-prefixed entries, each append with `fsync`; `open/1` rebuilds **and re-verifies** the tree | — |

The verifiers are checked against transparency-dev's 196 probes
(`test/fixtures/tlog/probes.json`), negative cases included — truncated
proofs, hashes of the wrong size, `size1 = 0`, swapped roots. A "naive"
verifier (one that trusts the step count of the proof itself) gets 182 of
196 right: it is the control of the quality check
(`mix vapor.quality`, §5b).

## 2. The witness

`Vapor.Tlog.Witness.cosign/4` only co-signs a *checkpoint* after checking
the consistency proof from the last one it saw of that log; it rejects
**rollback** (smaller size) and **fork** (same size, another root, or a
proof that does not close). To show two histories to two audiences, the
operator would need the witnesses to conspire too — and anyone can be a
witness.

## 3. On the server and in the console

With `mix vapor.serve --tlog PATH` (origin: `--tlog-origin`, default
`vapor.local/console`), the server keeps the log and its key (`PATH.key`,
mode 0600) and:

| call | response |
|---|---|
| `GET /v1/vapor/tlog` | origin, size, root, signed *checkpoint*, verification key, latest entries |
| `POST /v1/vapor/tlog` | `{"text"}` anchored; its receipt (index, proof, *checkpoint*) |
| `GET /v1/vapor/tlog/proof?index=i` | entry `i` with the receipt against the current tree |
| `GET /v1/vapor/tlog/consistency?from=m` | the proof that the current tree extends the one of `m` entries |
| `POST /v1/vapor/search` | search receipts come out **anchored** (`tlog`) |

In the console, the **Ledger** tab (*Trust* group) does not ask you to
trust the server: **the browser verifies on its own**. The verifier in
JavaScript (WebCrypto Ed25519, SHA-256) checks the signature of the
*checkpoint*, the inclusion proof of each entry and — keeping the last
*checkpoint* seen — the consistency with it; the log's key is pinned on
first use (TOFU, like SSH) and a key change is flagged, not
accepted. The inclusion proof is drawn: the leaf, the siblings going up, the
root. The same verifier runs in Node against the probes
(`test/js/tlog_verify.mjs`).

## 4. Why not a blockchain

What anchoring needs is (1) an append-only commitment and (2)
independent observers who detect a fork. A log with witnesses gives both
with one signature per *checkpoint*. A blockchain gives the same two
through paid consensus, with minutes of latency and a token in the middle —
cost with no new property. If one day it is useful to publish the root on a
public chain, it is one line of text: the *checkpoint* already is the object
to anchor.

## 5. Limits

- A built-in witness (`Vapor.Tlog.Witness`) is ready; the network of
  witnesses is social — others have to run it.
- The log keeps the whole entries in memory (the file is the persistence);
  for hundreds of millions of entries, C2SP's *tiles* form
  (`tlog-tiles`) is the next step.
- The console pins the key per browser; an auditor should pin it out of
  band.
