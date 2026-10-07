// The VS Code side is a thin client: every feature is `vapor lsp`'s.
const vscode = require("vscode");
const { LanguageClient } = require("vscode-languageclient/node");

let client;

function activate(context) {
  const command = vscode.workspace.getConfiguration("vapor").get("path") || "vapor";
  const server = { command, args: ["lsp"] };
  client = new LanguageClient("vapor", "vapor", { run: server, debug: server }, {
    documentSelector: [{ language: "mizan" }, { language: "alembic" }],
  });
  for (const cmd of ["vapor.mizan.toArabic", "vapor.mizan.toLatin"]) {
    context.subscriptions.push(vscode.commands.registerCommand(cmd, () => {
      const ed = vscode.window.activeTextEditor;
      if (ed) return client.sendRequest("workspace/executeCommand", { command: cmd, arguments: [ed.document.uri.toString()] });
    }));
  }
  client.start();
  context.subscriptions.push({ dispose: () => client && client.stop() });
}

function deactivate() { return client && client.stop(); }

module.exports = { activate, deactivate };
