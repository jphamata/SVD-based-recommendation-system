// The console's in-browser ledger verifier, run under Node (WebCrypto is the
// same API): the code between the tlog-verify markers of
// priv/console/index.html against (1) transparency-dev's probes and (2) the
// receipts, consistency proofs and signed checkpoint the Elixir log produced.
// usage: node tlog_verify.mjs INDEX_HTML PROBES_JSON CASES_JSON  → prints "ok N" or the failures
import { readFileSync } from "node:fs";
const [html, probesPath, casesPath] = process.argv.slice(2);
const src = readFileSync(html, "utf8");
const code = src.slice(src.indexOf("/* tlog-verify:begin */"), src.indexOf("/* tlog-verify:end */"));
const TLOG = new Function(code + "\nreturn TLOG;")();
const b64 = (s) => (s ? Uint8Array.from(Buffer.from(s, "base64")) : new Uint8Array(0));
const fails = [];
let n = 0;
const probes = JSON.parse(readFileSync(probesPath, "utf8"));
for (const p of probes.consistency) {
  const ok = await TLOG.consistency(p.size1, p.size2, (p.proof || []).map(b64), b64(p.root1), b64(p.root2));
  n++; if (ok !== !p.wantErr) fails.push(p.file);
}
for (const p of probes.inclusion) {
  const r = await TLOG.inclusion(b64(p.leafHash), p.leafIdx, p.treeSize, (p.proof || []).map(b64), b64(p.root));
  n++; if (r.ok !== !p.wantErr) fails.push(p.file);
}
const cases = JSON.parse(readFileSync(casesPath, "utf8"));
const v = await TLOG.verifyNote(cases.checkpoint, cases.verifier);
n++; if (!v.ok) fails.push("checkpoint signature");
const forged = cases.checkpoint.replace(/\n(\d+)\n/, (_, s) => `\n${Number(s) + 1}\n`);
n++; if ((await TLOG.verifyNote(forged, cases.verifier)).ok) fails.push("forged checkpoint accepted");
const cp = TLOG.parseCheckpoint(v.text);
for (const r of cases.receipts) {
  const lh = await TLOG.leaf(b64(r.entry));
  const res = await TLOG.inclusion(lh, r.index, r.size, r.proof.map(TLOG.unhex), cp.root);
  n++; if (!res.ok) fails.push(`receipt ${r.index}`);
  const bad = await TLOG.inclusion(lh, (r.index + 1) % r.size, r.size, r.proof.map(TLOG.unhex), cp.root);
  n++; if (bad.ok && r.size > 1) fails.push(`receipt ${r.index} accepted at another index`);
}
for (const c of cases.consistency) {
  const ok = await TLOG.consistency(c.from, c.to, c.proof.map(TLOG.unhex), TLOG.unhex(c.root1), TLOG.unhex(c.root2));
  n++; if (!ok) fails.push(`consistency ${c.from}→${c.to}`);
  if (c.from < c.to) {
    const swapped = await TLOG.consistency(c.from, c.to, c.proof.map(TLOG.unhex), TLOG.unhex(c.root2), TLOG.unhex(c.root1));
    n++; if (swapped) fails.push(`consistency ${c.from}→${c.to} with swapped roots`);
  }
}
console.log(fails.length ? "FAIL " + fails.join(", ") : `ok ${n}`);
