# vapor for VS Code

Install `vapor` (a checkout's `bin/vapor` works), then in this folder:

```sh
npm install                 # vscode-languageclient
npx @vscode/vsce package    # → vapor-0.16.0.vsix ; Extensions → Install from VSIX…
```

Settings: `vapor.path` (default `vapor`). Commands: *Almizan: show in Arabic/Latin
script*. Everything else — diagnostics that decide every obligation, hover, completion,
symbols, go to definition, formatting — comes from `vapor lsp`.
