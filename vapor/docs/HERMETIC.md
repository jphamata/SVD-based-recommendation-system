# The hermetic seal — containing work on untrusted input

> `Vapor.Hermetic` (`lib/vapor/hermetic.ex`). Tests: `hermetic_test.exs`; §5m. Scrutiny:
> [DIRECTIVE §20](DIRECTIVE.md).

*Hermetic* comes from Hermes' seal, the alchemists' vessel closed against the world.

## What it is

The single place where vapor contains work on untrusted input: a stranger's file (PDF, JBIG2,
JPEG, zip, GGUF), a program in Alembic written by a person or a model, a verb typed in the
console's terminal. Each job runs in a fresh BEAM process that:

- has its memory capped by the VM and is killed when it crosses the cap. The cap counts the
  process heap **and the off-heap binaries it holds** (`include_shared_binaries: true`);
- is killed at a wall-clock deadline;
- cannot take its caller down: a crash, an exhausted cap or a missed deadline comes back as a
  value (`{:error, :memory | :timeout | {:crash, reason}}`).

`Hermetic.seal/2` runs one job; `Hermetic.cap_self/1` caps a long-lived process (an Athanor
session). The admission boundaries (`Vapor.Lock` for models, `Vapor.Docs` for files,
`Vapor.Substrate` for accelerators) decide *what* gets in; the seal bounds *what it can cost*.

## Why it was needed

Until 0.16, four places carried their own copy of "spawn, set `max_heap_size`, wait": the Alembic
sandbox, the Dīwān's jail for the console terminal, the Athanor session, and the interactive
furnace. All four capped the **heap only**. On the BEAM, every binary larger than 64 bytes lives
outside the process heap, so a decompression bomb, or a verb that built a large binary, could
allocate past any cap and take the whole node down. Measured on OTP 28: under a 64 MB cap, 512 MB of
binaries survived the old flag and are killed by the seal. The four copies are now one, and document
ingestion, which ran unsealed, runs sealed (`heap_mb` 2048, `timeout` 600 s by default).

## Why not move the parsers into the Zig worker

The request proposed confining the PDF and JBIG2 parsers in the Zig worker under seccomp. That
would trade a memory-safe language for a memory-unsafe one, to gain an isolation the BEAM already
gives every process. An Elixir parser cannot overflow a buffer. Its real exposures are memory, time
and crashes, and the seal contains those three. Machine code still runs only in the isolated
workers, as before.

## What is not claimed

That the seal resists an attacker with access to the BEAM itself, or code that calls native
functions. The seal bounds memory and time; the BEAM's process isolation bounds the rest.
