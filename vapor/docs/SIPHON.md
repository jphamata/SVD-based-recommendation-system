# The siphon — the network airlock

> `Vapor.Siphon` (`lib/vapor/siphon.ex`), `vapor siphon …`, the MCP tool `siphon_propose`,
> `Vapor.Lock.preflight/2`, `Vapor.Ingest.Safetensors.parse_header/1`. Tests: `siphon_test.exs`.

A siphon draws liquid from outside through one tube, in one direction, and only when someone opens
the tap.

## The rule

**vapor's core opens no connection to fetch anything, and an agent never opens one.** Bytes from
outside come in through a **fetcher** the person declares, run when the person says so. The
fetcher can be a Python script using the official `huggingface_hub`, `aws s3 cp`, `rsync` from
your own cluster, `curl`, a peer of your private network: anything. A test holds the first half
of the rule. Outside the siphon, the only module of `lib/` that opens a connection is the agent
backends (`lib/vapor/agent/backends.ex`), which speak to a model API the person configured.

## Declaring fetchers

`$VAPOR_HOME/siphons.json` (default `~/.vapor/siphons.json`):

```json
{"fetchers": [
  {"name": "hf", "argv": ["python3", "/home/me/fetch_hf.py", "{ref}", "{out}"], "env": ["HF_TOKEN", "HTTPS_PROXY"]},
  {"name": "hf-file", "argv": ["curl", "-sfL", "-o", "{out}/file", "https://huggingface.co/{ref}"]},
  {"name": "hf-range", "argv": ["curl", "-sfL", "-r", "{range}", "-o", "{out}/part", "https://huggingface.co/{ref}"],
   "max_bytes": 200000000},
  {"name": "garage", "argv": ["rsync", "-a", "node1:/models/{ref}", "{out}/"], "timeout_s": 7200}
]}
```

`{ref}` is what to fetch, `{out}` the empty directory the fetcher writes into, and `{range}` a byte
range `a-b` (inclusive, as HTTP's `Range`) for fetchers that can read one. `env` lists the
variables passed through. A token goes only to the fetcher that names it.

## Who opens the tap

| surface | can |
|---|---|
| the person, at a terminal | `vapor siphon run NAME REF`, `vapor siphon approve ID`, `vapor siphon headers`, `vapor siphon preflight` |
| an agent over MCP | `siphon_propose`: append a request (fetcher, ref, why) to the queue. Nothing more. |
| a Majlis conversation | nothing. Its allowlist keeps conversations off the filesystem, and a queued request is a file |
| a console page | nothing: `siphon` is not one of the console terminal's verbs |

`vapor siphon queue` lists the requests, and `approve ID` runs one. The receipt names who proposed
and who approved. The cost to the person is one command. That is deliberate: the act of
connecting stays with someone who can be asked why.

## What a fetch can do, and what comes in

- It runs as its own OS process, never inside the BEAM, from an empty directory. Its environment
  is reduced to `PATH`, `HOME`, `LANG`, `LC_ALL`, `TMPDIR` and the names its declaration lists.
- It is killed at its deadline (`timeout_s`, 3600 by default), or as soon as it has written more
  than `max_bytes` (64 GiB by default).
- A ref is one argument, never a shell word. A ref that starts with `-` is refused, because a
  program could read it as an option.
- Every file it wrote passes its **format airlock**: `.safetensors` headers and tiling, `.gguf`
  magic, `.json` syntax. Each file is hashed (SHA-256) and checked against a pinned digest when one
  is given (`--sha256`, or `pins` in the declaration). Only then does it land, with `RECEIPT.json`:
  the fetcher, the ref, the exact argv, the digests and formats, the times, the proposer, the
  approver.
- A refused fetch leaves nothing behind.

## Headers before data

A safetensors file begins with its table of tensors. `vapor siphon headers NAME REF` reads only
that table, by two ranged fetches: the eight-byte length, then the header.
`vapor siphon preflight NAME CONFIG SHARD…` fetches the configuration and every shard's header,
then asks the airlock (`Vapor.Lock.preflight/2`) whether vapor can run the checkpoint. It names
any tensor that is missing, misshapen or unread, with no weight fetched. For Kimi K3, that is
about 1.5 TB that need not move to learn the answer.

## Streaming weights: the arithmetic

The proposal that came with this request was to stream a model from Hugging Face block by block, so
that a machine with little RAM runs a model larger than it, at "1 token every 3 to 5 seconds".
Decoding reads every active weight once per token, so the time per token is bounded below by
active bytes ÷ bandwidth:

| model | active bytes per token | RAM, 50 GB/s | NVMe, 3 GB/s | network, 20 MB/s |
|---|---|---|---|---|
| a 15 GB dense model | 15 GB | 0.3 s | 5 s | **12.5 min** |
| Kimi K3 (MXFP4 experts) | ≈ 146 GB | ≈ 3 s | ≈ 50 s | **≈ 2 h** |

Streaming over a network is therefore not a way to chat with a model, and neither is a cache of
"only the experts in use". Across a generation, the router touches most experts. What streaming
*can* serve is work that reads the weights **once** for a whole input: one pass of prefill to
score, classify or embed a long document. The siphon is how those bytes come in. vapor does not
yet have a layer-streaming executor, and docs/TODO.md says so.
