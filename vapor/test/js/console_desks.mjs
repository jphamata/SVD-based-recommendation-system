// The console's 0.12 desks driven end to end in headless Chromium: every
// example of the workbench, engineering, logic and render desks is run
// through the page (the same clicks a person makes), the boards are played
// a few moves, the protein desk runs its four actions, and the command
// palette is used to jump. Any page error, any example answered with an
// error notice, any empty result is reported.
// usage: node console_desks.mjs URL [SHOTS_DIR]  → JSON {ok, failures, checks, errors}
import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
const require = createRequire(import.meta.url);
let playwright;
try { playwright = require("playwright"); } catch { playwright = require("/opt/node22/lib/node_modules/playwright"); }
const [url, shots] = process.argv.slice(2);
if (shots) mkdirSync(shots, { recursive: true });

const browser = await playwright.chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [], failures = [], checks = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => { if (m.type() === "error" && !/favicon|manifest|404/.test(m.text())) errors.push(m.text()); });
await page.goto(url, { waitUntil: "load" });
await page.waitForTimeout(500);

const settle = async (root, ms = 120000) => {
  await page.waitForFunction((r) => { const n = document.getElementById(r); return n && !n.querySelector(".pulse") && !/running|rodando|…$/.test(n.querySelector(".wb-state")?.textContent || ""); }, root, { timeout: ms }).catch(() => failures.push(`${root}: timed out`));
};
const errText = (root) => page.evaluate((r) => [...document.getElementById(r).querySelectorAll(".wb-out .notice.err, .wb-out > .notice")].map((n) => n.textContent).join(" | "), root);
const shot = async (name) => { if (shots) await page.screenshot({ path: `${shots}/${name}.png`, fullPage: true }); };
const tab = async (id) => { await page.click(`#t-${id}`); await page.waitForTimeout(150); };

// examples of a desk, one by one through its select
async function examples(root, label, opts = {}) {
  const keys = await page.evaluate((r) => [...document.querySelectorAll(`#${r} .wb-bar select`)][0] ? [...document.querySelector(`#${r} .wb-bar select`).options].map((o) => o.value).filter(Boolean) : [], root);
  for (const k of keys) {
    await page.selectOption(`#${root} .wb-bar select >> nth=0`, k);
    await page.waitForTimeout(80);
    await settle(root, opts.timeout);
    const e = await errText(root);
    const filled = await page.evaluate(([r, sel]) => document.querySelector(`#${r} ${sel}`).children.length, [root, opts.filled || ".wb-out"]);
    if (e && !(opts.expectError || []).includes(k)) failures.push(`${label}/${k}: ${e}`);
    else if (!filled) failures.push(`${label}/${k}: empty result`);
    else checks.push(`${label}/${k}`);
  }
}

// workbench
await tab("bench"); await settle("wb-bench"); await examples("wb-bench", "bench", { timeout: 180000 }); await page.selectOption("#wb-bench .wb-bar select >> nth=0", "lorenz"); await settle("wb-bench"); await shot("bench");
// engineering: every kind, every example
await tab("eng");
const kinds = await page.evaluate(() => [...document.querySelectorAll("#wb-eng > .seg button")].map((b) => b.textContent));
for (let i = 0; i < kinds.length; i++) {
  await page.click(`#wb-eng > .seg button >> nth=${i}`); await page.waitForTimeout(100); await settle("wb-eng");
  await examples("wb-eng", `eng:${kinds[i]}`);
  await shot(`eng-${i}`);
}
// logic
await tab("logic"); await settle("wb-logic"); await examples("wb-logic", "logic"); await page.selectOption("#wb-logic .wb-bar select >> nth=0", "ramsey"); await settle("wb-logic"); await shot("logic");
// boards
await tab("boards");
const games = await page.evaluate(() => [...document.querySelectorAll("#wb-boards > .seg button")].map((b) => b.textContent));
for (let i = 0; i < games.length; i++) {
  await page.click(`#wb-boards > .seg button >> nth=${i}`); await page.waitForTimeout(1500);
  await page.waitForFunction(() => !/…$/.test(document.querySelector("#wb-boards .wb-state")?.textContent || ""), null, { timeout: 120000 }).catch(() => failures.push(`boards ${games[i]}: timed out`));
  const kind = await page.evaluate(() => (document.querySelector("#wb-boards svg.board")?.classList[1]) || (document.querySelector("#wb-boards .probs") ? "poker" : ""));
  if (kind === "chess") {
    await page.click('#wb-boards [data-sq="e2"]'); await page.click('#wb-boards [data-sq="e4"]');
    await page.waitForFunction(() => document.querySelectorAll("#wb-boards .movelist span").length >= 2 && !/…$/.test(document.querySelector("#wb-boards .wb-state").textContent), null, { timeout: 60000 }).catch(() => failures.push("chess: no engine reply"));
    const n = await page.evaluate(() => [...document.querySelectorAll("#wb-boards .movelist span")].map((s) => s.textContent).filter(Boolean));
    n.length >= 2 && n[0] === "e4" ? checks.push(`chess: 1. ${n[0]} ${n[1]}`) : failures.push(`chess moves ${n}`);
  } else if (kind === "go") {
    const before = await page.evaluate(() => document.querySelectorAll("#wb-boards .stone").length);
    await page.click("#wb-boards .hit >> nth=40");
    await page.waitForFunction((b) => document.querySelectorAll("#wb-boards .stone").length >= b + 2, before, { timeout: 120000 }).catch(() => failures.push("go: no engine reply"));
    checks.push("go: a move and a reply");
  } else if (kind === "mnk") {
    await page.click("#wb-boards .cell-m >> nth=4"); await page.waitForTimeout(2500);
    const marks = await page.evaluate(() => document.querySelectorAll("#wb-boards .mk").length);
    marks === 2 ? checks.push("mnk: centre and a reply") : failures.push(`mnk: ${marks} marks`);
  } else if (kind === "shogi") {
    await page.waitForSelector("#wb-boards svg.shogi");
    checks.push("shogi: board drawn");
  } else if (kind === "poker") {
    await page.waitForSelector("#wb-boards .probs", { timeout: 300000 }).catch(() => failures.push("poker: no strategy"));
    checks.push("poker: solved");
  }
  await shot(`boards-${i}`);
}
// proteins: analyse (automatic), pipeline, compare, align
await tab("protein"); await settle("wb-protein", 60000);
const acts = await page.evaluate(() => [...document.querySelectorAll("#wb-protein > .seg button")].map((b) => b.textContent));
for (let i = 0; i < acts.length; i++) {
  await page.click(`#wb-protein > .seg button >> nth=${i}`); await page.waitForTimeout(150);
  const run = await page.$("#wb-protein .wb-bar .primary");
  if (run) { await run.click(); await page.waitForTimeout(100); }
  await settle("wb-protein", 300000);
  const e = await errText("wb-protein"), n = await page.evaluate(() => document.querySelector("#wb-protein .wb-out").children.length);
  e ? failures.push(`protein/${acts[i]}: ${e}`) : n ? checks.push(`protein/${acts[i]}`) : failures.push(`protein/${acts[i]}: empty`);
  await shot(`protein-${i}`);
}
// render: the GPU tracer converges a few frames; the reference and the furnace come back
await tab("render"); await page.waitForTimeout(500);
const gpu = await page.evaluate(() => !document.querySelector("#wb-render .notice"));
if (gpu) { await page.click("#wb-render .wb-bar .primary"); await page.waitForTimeout(3000); await page.click("#wb-render .wb-bar .primary"); }
const spp = await page.evaluate(() => document.querySelector("#wb-render figcaption span").textContent);
checks.push(`render: gpu ${gpu} · ${spp}`);
await examples("wb-render", "render", { filled: ".gpu-fig" });
await page.click("#wb-render .wb-bar button >> text=/Furnace|Teste da fornalha/");
await page.waitForSelector("#wb-render .wb-out table", { timeout: 120000 }).catch(() => failures.push("render: no furnace"));
await page.click("#wb-render .wb-bar button >> text=/reference|referência/");
await page.waitForSelector("#wb-render .refimg", { timeout: 240000 }).catch(() => failures.push("render: no reference"));
await shot("render");
// the palette: Ctrl+K, a word, Enter
await page.keyboard.press("Control+k"); await page.keyboard.type("Warren"); await page.keyboard.press("Enter");
await settle("wb-eng");
const t = await page.evaluate(() => document.querySelector("#wb-eng textarea").value.includes("Warren"));
t ? checks.push("palette: Warren truss opened") : failures.push("palette: did not open the example");
// alphabetical order of the rail, in both languages
const order = async () => page.evaluate(() => { const out = []; let cur = [], skip = false; /* a data-fixed group keeps its working order (write → test → measure) */ for (const n of document.querySelector('nav [role="tablist"]').children) { if (n.classList.contains("group")) { if (cur.length) out.push(cur); cur = []; skip = n.hasAttribute("data-fixed"); } else if (!skip) cur.push(n.querySelector("b").textContent); } out.push(cur); return out; });
const sorted = (groups, loc) => groups.every((g) => g.every((x, i) => i === 0 || x.localeCompare(g[i - 1], loc, { sensitivity: "base" }) >= 0));
const en = await order(); await page.click("#lang"); await page.waitForTimeout(300); const pt = await order();
sorted(en, "en") && sorted(pt, "pt-BR") ? checks.push("rail: alphabetical in en and pt") : failures.push(`rail order ${JSON.stringify(en)} ${JSON.stringify(pt)}`);
await tab("bench"); await shot("bench-pt");
await browser.close();
console.log(JSON.stringify({ ok: failures.length === 0 && errors.length === 0, failures, errors, checks }));
