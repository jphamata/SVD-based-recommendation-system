# The Majlis — conversations as a content-addressed tree

> مجلس, root ج-ل-س *j-l-s*, "to sit": the council where people converse. `Vapor.Majlis`. Doors: the
> console (*Converse → Conversations*), `vapor chat` (terminal), the API `/v1/vapor/threads…`
> ([CONSOLE.md](CONSOLE.md)). Tests: `majlis_test.exs`, `hall_test.exs`,
> `console_majlis_test.exs` (Chromium). Scrutiny: [DIRECTIVE §19](DIRECTIVE.md).

## The idea

Chat products treat the conversation as a mutable list with patches (edit branches,
"branch into a new conversation", invisible compaction). Here it is what it actually is: a **tree**.

- A **message** is an immutable node `M1 ‖ CBOR{role, content, parent, t, meta, o}`, addressed
  by the SHA-256 of its bytes. Since the parent is inside the hash, **a message commits to
  its whole history** — like a git *commit*.
- A **conversation** is a pointer (`head`) plus settings: instructions, model, tools,
  context budget, pinned messages, summary. Each conversation has its own anchor node and each
  message its owner (`o`), so that garbage collection of one never erases another's.
- **Edit** writes a sibling; **another answer** writes a sibling of the answer; **‹ ›** switches
  sibling and follows the most recent branch below it; **continue from here** moves the pointer back;
  **fork** creates a new conversation pointing at an existing message — **O(1)**, no
  message copied (§5l: 0 messages, 964 bytes of root against 120 messages for a copy).
- Everything lives in a [Khazāna](KHAZANA.md): each action is an atomic *commit*.

## The context, computed and shown

`context/2` returns **exactly** what the model will read: the instructions (with the summary, if there is one),
then the pinned messages and the last turn — always — then the most recent ones that fit in the
budget. Each message on the path comes out marked `sent`, `pinned`, `summarized` or `dropped`, with
its tokens (exact when the server serves the model's tokenizer; estimated, and said to be
estimated, when not). The control in §5l: truncation from the tail, with the same budget,
drops the pinned instruction.

**Compact** asks the model (or accepts from the user) for a summary of the path up to a message; the summary
**names the hash it covers**, and therefore commits to everything before it. Undoing returns the
whole history to the context; the summary never erases anything.

## Agent

A conversation can enable vapor tools by allow-list (`Vapor.Majlis.Tools`, the
same as the MCP server's): pure ones (`alembic_eval`, `athanor_verify`, `rebis_check`, `aludel_decide`,
`tabula_analyze`, `amalgam_sum`, `cupel_drill`, `logic_check`, `workbench_solve`, …) and
observation ones (`athanor_run`, `crucible_run`, `assay_run`, `finance_run`, …). An answer with
tools runs vapor's agent loop and keeps the run's **journal** (Merkle,
verifiable: `GET /v1/vapor/journal/:id`); the message carries the journal in `meta`.

## Exchange, share

- **Export**: JSON `vapor-majlis/1` (each message with its hash; on import, **all are
  recomputed**, and a changed character is rejected — §5l) or Markdown (for reading; the control: the
  same change in the Markdown is undetectable).
- **Import**: vapor's JSON, the ChatGPT export (`conversations.json`, the `mapping`
  tree preserved with its branches) or Claude's (`chat_messages`).
- **Search**: BM25 over all messages, from all branches.
- **Share**: a read-only link `/shared/ID?cap=…`, where `cap` = HMAC(conversation,
  generation). It does not ask for the console token — the capability is the authority. The page has no *script* and
  escapes all text. **Revoke** increments the generation and kills every link already given out.

## Models

The Majlis has no model of its own: it uses the server's *backends* — the locally served model (the
answers are re-derivable: seed, receipt) and the one from `VAPOR_MIND` (`anthropic:…`, `openai:…@URL`,
`script:FILE`). With neither, the messages are kept and the answer says why.

## What it does not do (yet)

Token-by-token *streaming*; image attachments in the conversation; memory across conversations. They are in the
[TODO](TODO.md).

## Terminal

```sh
T=$(vapor chat new --title "draft" --system "answer in Portuguese")
vapor chat say $T "what is a Gröbner basis?"
vapor chat show $T --tree          # all branches
vapor chat edit $T 3fa2c1 "and an example?"   # 6+ hex digits are enough
vapor chat context $T              # what goes, what is left out
vapor chat fork $T 3fa2c1 --title "another line"
vapor chat export $T --md > rascunho.md
```
