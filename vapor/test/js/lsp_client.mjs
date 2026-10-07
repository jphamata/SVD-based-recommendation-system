// A minimal LSP client: spawns `bin/vapor lsp`, speaks Content-Length-framed
// JSON-RPC over its stdio, opens an Al-Mizān file with a refuted claim, and
// prints what came back as one JSON object. usage: node lsp_client.mjs VAPOR_BIN
import { spawn } from "node:child_process";

const bin = process.argv[2];
const srv = spawn(bin, ["lsp"], { stdio: ["pipe", "pipe", "inherit"], env: process.env });
let buf = Buffer.alloc(0);
const pending = new Map();
const notes = [];

srv.stdout.on("data", (d) => {
  buf = Buffer.concat([buf, d]);
  for (;;) {
    const sep = buf.indexOf("\r\n\r\n");
    if (sep < 0) return;
    const len = Number(/Content-Length: (\d+)/i.exec(buf.slice(0, sep).toString())[1]);
    if (buf.length < sep + 4 + len) return;
    const msg = JSON.parse(buf.slice(sep + 4, sep + 4 + len).toString("utf8"));
    buf = buf.slice(sep + 4 + len);
    if (msg.id !== undefined && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
    else notes.push(msg);
  }
});

let next = 1;
const send = (m) => { const b = Buffer.from(JSON.stringify(m), "utf8"); srv.stdin.write(`Content-Length: ${b.length}\r\n\r\n`); srv.stdin.write(b); };
const call = (method, params) => new Promise((res) => { const id = next++; pending.set(id, res); send({ jsonrpc: "2.0", id, method, params }); });
const until = async (pred, ms = 20000) => { const t0 = Date.now(); while (!notes.some(pred)) { if (Date.now() - t0 > ms) return null; await new Promise((r) => setTimeout(r, 20)); } return notes.find(pred); };

const uri = "file:///tmp/oscillator.wzn";
const text = `(claim energy (root H-f-Z) (wazn burhan)
  (inputs (x q) (v q))
  (field (x v) (v (- (- x) (* 1/10 v))))
  (proof conserved)
  (body (+ (* 1/2 v v) (* 1/2 x x))))
`;

const out = {};
const init = await call("initialize", { processId: null, rootUri: null, capabilities: {} });
out.capabilities = Object.keys(init.result.capabilities).sort();
send({ jsonrpc: "2.0", method: "initialized", params: {} });
send({ jsonrpc: "2.0", method: "textDocument/didOpen", params: { textDocument: { uri, languageId: "mizan", version: 1, text } } });
const diag = await until((m) => m.method === "textDocument/publishDiagnostics");
out.diagnostics = diag ? diag.params.diagnostics.map((d) => ({ line: d.range.start.line, severity: d.severity, message: d.message })) : null;
const hover = await call("textDocument/hover", { textDocument: { uri }, position: { line: 0, character: 22 } });
out.hover = hover.result && hover.result.contents.value;
const fmt = await call("textDocument/formatting", { textDocument: { uri }, options: { tabSize: 2, insertSpaces: true } });
out.formatted = fmt.result.length;
const conv = await call("workspace/executeCommand", { command: "vapor.mizan.toArabic", arguments: [uri] });
const edit = await until((m) => m.method === "workspace/applyEdit");
out.arabic = edit ? edit.params.edit.changes[uri][0].newText : null;
await call("shutdown", null);
send({ jsonrpc: "2.0", method: "exit" });
console.log(JSON.stringify(out));
setTimeout(() => process.exit(0), 200);
