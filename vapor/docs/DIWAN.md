# The Dīwān — one interpreter for every terminal

> ديوان, root د-و-ن *d-w-n*, "to record": the register, and the hall where business is dispatched. `Vapor.Diwan`.
> Tests: `diwan_test.exs`, `hall_test.exs`, `tui_test.exs`, `console_majlis_test.exs`; §5l.

## One function, four doors

`eval(line, session) → {result, session}`. The same line gives the same answer at every door,
because they are the same function:

| door | how | session |
|---|---|---|
| command line | `bin/vapor VERB …` (each verb is `Vapor.Main.run/1`) | the user's shell |
| TUI | `mix vapor.tui` (no curses: works over SSH and in a CI log) | **local**: reads and writes the files of whoever is at the keyboard |
| console terminal | *Converse → Terminal* | **jailed** |
| API | `POST /v1/vapor/diwan` `{session, line}` → `{out, err, code, codes, files}` | jailed |

A new verb appears in all four without one extra line.

## The language of the line

A small shell — **no shell is executed**: vapor's verbs (the word `vapor` is
optional), `|`, `>`, `>>`, `<`, `;`, quotes and `\`, and the builtins `help ls cat echo rm cp mv head
wc history clear`.

```sh
athanor run golomb.nbq | verify golomb.nbq -
rebis equiv a.net b.net > veredito.json
echo "(claim q (root H-s-b) (wazn fail) (inputs (x q)) (body (* x x)))" > q.wzn ; wzn check q.wzn
chat search kulisch | head 5
```

Every stage but the last writes JSON (the verb sees a *pipe*); the last one writes for people —
exactly the Unix behaviour of `bin/vapor`. A leading or trailing `|`, or `a | | b`, is refused
(an empty command), not ignored.

## The jail

The console session is a door for whoever arrives over the network, so:

- the files are **the session's own** (`ls`, `cat`, `rm`, `>` and the panel's editor and upload): 128
  files, 8 MB each, 64 MB in total; a FILE argument names one of them — `/etc/passwd` does not
  exist there;
- `--measure` (which runs a program) is refused; no verb opens an external process;
- each command runs in **its own process** with a heap ceiling and a deadline: a runaway command is
  killed, the session stays;
- a session runs one command at a time (a second one, arriving in the middle, gets 409 instead of
  contending for the files);
- at most 256 live sessions; the oldest one goes.

The control in §5l: the same read in a local session (the TUI) reads the file; in the jail, exit 3 and "no
such file in this session". The ledger says what this rests on: the BEAM's process
isolation and every verb reading through the door `Vapor.Main.read_input`.

## The panel

A screen with the commands' ANSI colours (the page's palette), history (↑ ↓), Tab completes verbs and
session files (`POST /v1/vapor/diwan/complete`), Ctrl+L clears; on the right, the session's
files with an editor (Ctrl+S saves), upload and download. Ready-made examples write the files they
need and run.
