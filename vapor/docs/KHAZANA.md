# The Khazāna — a content-addressed store with a *crash-atomic* root

> خزانة, root خ-ز-ن *kh-z-n*, "to keep". `Vapor.Khazana`. Tests: `khazana_test.exs`; §5l of the
> quality suite. The idea is that of ASAS §8.3–8.4 (*Atomic Stream Application Substrate*), carried from NVRAM
> to a POSIX directory — [DIRECTIVE §19](DIRECTIVE.md).

## The pain

Every program that keeps state in files reinvents "write to a temporary, `fsync`, `rename`,
`fsync` on the directory" — and OTP **cannot** `fsync` a directory, so on the BEAM a
`rename` is not durable. A state file overwritten in place, on the other hand, can be left
*torn* by a crash in the middle of the write. The measurement in §5l shows that tearing is worse than
loss: half of a CBOR root of the same size decodes — as v2, with a value that never
existed.

## The protocol

**After `init/1` no file is created, renamed or deleted.**

```
DIR/pack.0, DIR/pack.1   appended blobs:  [u32 size][32-byte SHA-256][bytes]
DIR/root.a, DIR/root.b   one fixed 128-byte record each, rewritten in place
DIR/key                  a 32-byte secret (capabilities, mac/2)
```

A root record is `KHZ1 · seq (u64) · pack generation (u8) · committed size (u64) ·
root-blob hash (32) · tag (32)`, the tag being the SHA-256 of everything before it.
`commit/2`:

1. appends the new blobs (and the encoded root term) to the active pack and calls `datasync` — the content
   is durable before anything names it;
2. writes the **inactive** slot with `seq + 1` and calls `datasync` — the active slot is never touched;
3. the current root is the valid slot with the highest sequence.

`open/1` reads both slots, discards any whose tag does not check, keeps the highest surviving
sequence and reads the pack only up to the size that root committed: a torn append after it is
ignored. Each blob in the committed prefix is re-hashed on opening — corruption is found, not
served. Compaction (`gc/2`) writes the live blobs into the *other* pack and commits a root that
names it: the same protocol, so a crash in the middle leaves the previous pack and root in force.

## Measured

`commit(k, term, fault: {:pack, n} | {:slot, n})` interrupts the write at byte `n`. The test and the
quality suite (§5l) crash **every byte** of a *commit* — 344 points, from the first byte of the append to the
last of the slot — and reopen: **always** the old root or the new one. The control (a root file
overwritten in place, crash halfway) is read as a torn value.

## Capabilities

`mac(k, parts)` is HMAC-SHA256 under the store's key; `mac_ok?/3` compares in constant time. The
conversations use this for shared links: `mac(["share", conversation, generation])` — unforgeable
without the key, and revoking is incrementing the generation (ASAS §6.2: mass revocation in constant time,
with no capability table).

## What it does not do

- It assumes that `datasync` only returns when the data is durable (true for local file systems
  with honest disks; not for some network file systems or disks that lie about their cache).
- It assumes that a 128-byte write is not corrupted *silently* in such a way that it still matches the
  tag (a SHA-256 collision).
- Concurrency is the caller's: one process owns a store (the Majlis is a `GenServer`).
- P2P (the manifesto's "Al-Khazāna") is left for later: without a trust model, fetching
  dependencies from peers is a supply-chain vector.

## API

`init/1` · `open/2` · `put/2` · `put_term/2` · `get/2` · `get_term/2` · `has?/2` · `hashes/1` ·
`root/1` · `commit/3` · `gc/3` · `mac/2` · `mac_ok?/3` · `hex/1` · `unhex/1`.
