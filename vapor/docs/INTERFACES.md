# Interfaces: GUI, TUI, CLI — and why not Tauri

vapor has four faces over a single core. In all of them, every answer arrives with
its measure (confidence, certainty, receipt, distance to the training data): the interface is
where the question "is this signal or noise?" is answered for a person.

| face | for | how |
|---|---|---|
| **Web console** (GUI) | daily use, demonstrations, reviewing evidence | `mix vapor.serve [--model DIR] [--docs PATH]` → `http://127.0.0.1:8000/` |
| **Installed app** | a window of its own, without a browser tab | the console is an installable web app (`manifest.webmanifest`, icons): in Chromium/Edge, *Install vapor* opens it in its own window |
| **Terminal console** (TUI) | SSH, servers without a browser, CI logs | `mix vapor.tui [--lang pt] [--docs PATH]` |
| **Command-line tasks** | scripts and pipelines | `mix vapor.ocr`, `vapor.merge`, `vapor.quality`, `vapor.rag`, `vapor.lock`, … |
| **HTTP API** | other programs | OpenAI-compatible `/v1/*` and the console's `/v1/vapor/*` ([CONSOLE.md](CONSOLE.md)) |

## Since 0.14: the terminal first

Everything the console does, `bin/vapor` does — with JSON in *pipes*, exit codes with
meaning and reading from standard input ([CLI.md](CLI.md)). The console, the TUI and the
MCP tools call the same functions; no capability exists only in the graphical interface.

## The web console

A self-contained HTML file (`priv/console/index.html`): no CDN, no downloaded
fonts, works without internet, served by the same BEAM that runs the models.

- **Language**: English by default, Portuguese one click away (remembered in the
  browser). All of the page's text lives in a dictionary — a third language
  is a table, not a rewrite. The diagnostics that depend on numbers
  (the regime of a merge) are assembled on the client from the numbers, in
  both languages; messages that come from the server (refusals, errors) stay in
  English, like the API.
- **Theme**: light and dark, following the system until the person chooses.
- **Identity**: a canal lock. The logo (`priv/console/logo.svg`,
  `docs/img/logo.svg`) is the basin between the gates, with the water at a level and the
  vapour rising; the page's bold element is the same idea — each result
  in a tank whose level is its measure, with the calibrated line marked. The rest
  is quiet: a sans for the text, a condensed signage face for the
  headings, monospace only for *digests*.
- **Panels** grouped by what one does: *Ask* (conversation with evidence),
  *Read* (documents, vision/OCR, speech), *Create* (drawing by diffusion), *Measure*
  (merging, quality gate, the model airlock).
- **Accessibility**: keyboard navigation (arrows between sections, skip link,
  visible focus), reduced motion respected, readable at 390 px. Checked
  in headless Chromium in light, dark, in both languages and at phone
  width, with no console error.

## The terminal console

`Vapor.TUI` — a line-based session (no curses, no dependency): `read`
(OCR), `listen`, `draw` (the digit drawn in the terminal on the 24-grey scale
and read back by the classifier), `add`/`search`, `quality`, `merge`,
`lang en|pt`. Colour only in a terminal and never with `NO_COLOR`; in a pipe,
plain text. The interpreter (`eval/2`) is pure and tested without a terminal.

## Tauri: considered, not adopted

A Tauri wrapper would give a native window around the same page. Weighed
against what vapor is:

| | Tauri | installable web app (adopted) |
|---|---|---|
| new toolchains | Rust, Cargo, a WebView per OS, Node for the bundler | none |
| the models' runtime | the BEAM and the native worker as a *sidecar* per OS, with signing | already running: it is what serves the page |
| dependency policy (`deps: []`) | breaks (crates, npm) | kept |
| no internet | yes | yes (one file, no CDN) |
| window, icon, taskbar entry | yes | yes ("Install" in Chromium/Edge) |
| file system, tray, automatic updates | yes | no — and it is not missed: files come in through the picker and by drag and drop, the server reads `--docs` |
| what it adds to the *evidence* | nothing | — |

The worker is Linux-only today ([TODO](TODO.md)), so a cross-platform desktop
package would wrap a runtime that does not run on two of the three targets.
When there are workers for macOS/Windows, it is worth revisiting — and the page does not
change: a Tauri would point to the same `http://127.0.0.1:PORT/`.
