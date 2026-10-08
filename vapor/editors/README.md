# Editors

One language server, `vapor lsp`, serves every editor (docs/EDITORS.md):

| editor | what to install | tested here |
|---|---|---|
| VS Code | `editors/vscode` (extension: grammars, language configuration, the client) | grammars and manifest validated by a test; the extension itself needs VS Code |
| Neovim 0.10+ | `editors/nvim` on the runtimepath; `require("vapor").setup()` | not installed on this machine |
| Vim | `editors/nvim/{ftdetect,syntax}` (highlighting; LSP through any client plugin) | syntax files load without errors in `vim -Es` (test) |
| Emacs 29+ | `editors/emacs/vapor-mode.el` (modes, eglot registration) | not installed on this machine |
| Helix, Zed, Kakoune, Sublime | any LSP client: command `vapor lsp`, file types `.wzn`, `.nbq` | the protocol is tested end to end by a Node client over stdio |
