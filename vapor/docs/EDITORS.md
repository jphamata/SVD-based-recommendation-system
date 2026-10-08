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
| commands | `vapor.almizan.toArabic` / `toLatin`: the same program in the other script (the same tree, the same hash), as an edit the person chooses | — |

## Installing

| editor | what | tested here |
|---|---|---|
| VS Code | `editors/vscode` (TextMate grammars for both scripts and for Alembic, language configuration, the client; `vapor.path` points to the executable) | manifest and grammars validated; the extension itself needs VS Code |
| Neovim 0.10+ | `editors/nvim` on the *runtimepath*; `require("vapor").setup()`; commands `:AlmizanArabic`, `:AlmizanLatin` | not installed on this machine |
| Vim | `editors/nvim/{ftdetect,syntax}` (highlighting; LSP through any client *plugin*) | loads without error in `vim -Es` |
| Emacs 29+ | `editors/emacs/vapor-mode.el` (`almizan-mode`, `alembic-mode`, registration with `eglot`; `almizan-to-arabic`, `almizan-to-latin`) | not installed on this machine |
| Helix, Zed, Kakoune, Sublime | any LSP client: command `vapor lsp`, types `.wzn`, `.nbq` | the protocol is tested end to end |

Highlighting treats both scripts equally: Latin and Arabic keywords, roots in Buckwalter and
in Arabic, Western and Arabic-Indic digits. Arabic text is displayed right to
left by the editor itself (Unicode bidi); the S-expression structure needs no direction marks.

## What is not claimed

That the VS Code extension, the Neovim *plugin* and the Emacs mode have been opened in their
respective editors on this machine — they are not installed. What is tested is the server (over
real stdio) and the files each editor reads (validated by format).
