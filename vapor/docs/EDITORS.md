# Editors — one language server, three thin clients

> `vapor lsp` (`Vapor.LSP`), `editors/`. Tests: `lsp_test.exs` (the protocol over real stdio, from a
> Node client), `editors_test.exs` (VS Code manifest and grammars validated; the Vim syntax files
> load without error in `vim -Es`). Scrutiny: [DIRECTIVE §19](DIRECTIVE.md).

## The principle

An ecosystem "à la VS Code, Neovim and Emacs" is not three *plugins* that reimplement the language —
that would give three truths. It is **one** server (LSP 3.17, JSON-RPC over stdio, `Content-Length`
*framing*, columns in UTF-16) and clients that only connect to it. Every editor that speaks LSP — Helix,
Zed, Kakoune, Sublime — gets the same, with nothing from here.

| | Almizan (`.wzn`) | Alembic (`.nbq`) |
|---|---|---|
| diagnostics | syntax and morphological rules while typing; on open and save, **every obligation decided** — a refuted claim is an error on its line, with the point that refutes it | parse errors with line and column |
| *hover* | the verdict and the decider of a claim; the meaning and the abjad value of a root; a keyword in both scripts | the signature of a built-in |
| completion | keywords in the file's script, roots, claims defined above | built-ins and constants |
| symbols, go to definition | claims | definitions |
| formatting | the canonical printing, in the file's script | — |
| inlay hints (0.17) | at the end of each proved claim's first line, its verdict and decider: `✓ proved · SAT + DRUP`, `✗ refuted · polynomial normal form over ℚ` (the detail as tooltip) | — |
| commands | `vapor.almizan.toArabic` / `toLatin`: the same program in the other script (the same tree, the same hash), as an edit the person chooses | — |

## Installing

| editor | what | tested here |
|---|---|---|
| VS Code | `editors/vscode` (TextMate grammars for both scripts and for Alembic, language configuration, the client; `vapor.path` points to the executable) | manifest and grammars validated; the extension itself needs VS Code |
| Neovim 0.10+ | `editors/nvim` on the *runtimepath*; `require("vapor").setup()`; commands `:AlmizanArabic`, `:AlmizanLatin` | not installed on this machine |
| Vim | `editors/nvim/{ftdetect,syntax}` (highlighting; LSP through any client *plugin*) | loads without error in `vim -Es` |
| Emacs 29+ | `editors/emacs/vapor-mode.el` (`almizan-mode`, `alembic-mode`, registration with `eglot`; `almizan-to-arabic`, `almizan-to-latin`) | not installed on this machine |
| JetBrains IDEs | the IDE's LSP support (2023.2+, paid editions) or the LSP4IJ plugin (any edition): command `vapor lsp`, file types `.wzn`, `.nbq` | not installed on this machine |
| Helix, Zed, Kakoune, Sublime | any LSP client: command `vapor lsp`, types `.wzn`, `.nbq` | the protocol is tested end to end |

Highlighting treats both scripts equally: Latin and Arabic keywords, roots in Buckwalter and
in Arabic, Western and Arabic-Indic digits. Arabic text is displayed right to
left by the editor itself (Unicode bidi); the S-expression structure needs no direction marks.

## Should vapor have its own editor? (0.17)

Asked in round 0.17, from someone who uses Emacs and Neovim and likes both. The answer is no.
What the question is really after is done instead, inside the editors people already have.

**What an editor is.** Emacs has accumulated since 1976, Vim since 1991, and the VS Code editor
(Monaco) is the work of a large team. What makes them good is everything around the text:
keymaps in muscle memory, extension ecosystems, input methods and bidirectional text,
accessibility, very large files, remote editing, debuggers, version control. None of that is
vapor's competence, and none of it is something vapor could claim to do better. A new editor
would start every user at zero in exactly the places where they are fastest today.

**What the three traditions teach, and where vapor already applies it:**

| tradition | the lesson | in vapor |
|---|---|---|
| Microsoft (LSP, 2016) | separate the language from the editor: N languages × M editors becomes N + M | one server, `vapor lsp`, and clients that only connect to it; an editor of its own would undo this |
| JetBrains (PSI) | the value is a semantic model of the code (error-tolerant trees, indexes, refactorings with preview), and that model lives behind the editor | the server holds it: the decided obligations, the roots, the claims; it grows there (inlay hints in 0.17; rename and code actions are owed) |
| GNU Emacs | an environment that documents and programs itself | the Dīwān: every verb has `help`, pipes and files, and the same interpreter runs in the terminal, the TUI and the console |

**What no general editor shows, and vapor should:** the *evidence* next to the code, meaning the
verdict, the decider, the counterexample point and the hash. That needs a surface, not an editor:

- **in every editor**, through the protocol: diagnostics carry the refuting point, *hover*
  carries the decider, and since 0.17 **inlay hints** put each claim's verdict at the end of its
  line. Neovim 0.10 shows them with `vim.lsp.inlay_hint.enable()`, Emacs with
  `eglot-inlay-hints-mode`, and VS Code and JetBrains by default;
- **in the console**, which already has an editor pane beside live results (the workbench, the
  furnace, the Opus desks) and the Dīwān terminal. That is where a notebook-like surface belongs,
  served offline from one page, with no editor to install.

What stays owed: rename and code actions in the server (quick fixes such as "add the missing
`(box …)`"), semantic tokens, and actually running the clients in their editors (below).

## What is not claimed

That the VS Code extension, the Neovim *plugin* and the Emacs mode have been opened in their
respective editors on this machine — they are not installed. What is tested is the server (over
real stdio) and the files each editor reads (validated by format).
