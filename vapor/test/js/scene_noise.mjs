// The scene engine's noise() must equal Vapor.Alembic.Tree.noise/1 bit for bit
// (golden values asserted in test/vapor/alembic_test.exs). Usage: node test/js/scene_noise.mjs priv/console/index.html
import { readFileSync } from "node:fs";
const html = readFileSync(process.argv[2] || "priv/console/index.html", "utf8");
const m = /function noise\(args\) \{[\s\S]*?\n    \}/.exec(html);
if (!m) { console.error("noise() not found in the engine"); process.exit(2); }
const noise = new Function(m[0] + "; return noise;")();
const golden = [[[1, 2], 0.16636425908654928], [[-1.25, 7], 0.03921110928058624], [[123.4567, -0.0015], 0.10643366817384958]];
let bad = 0;
for (const [args, want] of golden) { const got = noise(args); if (got !== want) { console.error(`noise(${args}) = ${got}, want ${want}`); bad++; } }
console.log(bad ? "FAIL" : "ok: the engine's noise equals the server's");
process.exit(bad ? 1 : 0);
