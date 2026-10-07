// The 0.15 Opus desks driven end to end in headless Chromium: every shelf
// example of Rebis, Aludel, Tabula, Cupel and Amalgam is clicked as a
// person would, the answer must carry its touchstone and its drawing (the
// seal, the subdivided box, the phase portrait, the tablet, the float's 32
// bits, the line of sums); a fact is toggled on the tablet; the page is
// switched to Portuguese and to the dark theme. Any page error, error
// notice or empty result is reported.
// usage: node console_opus.mjs URL [SHOTS_DIR]  → JSON {failures, checks, errors}
import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
const require = createRequire(import.meta.url);
let playwright;
try { playwright = require("playwright"); } catch { playwright = require("/opt/node22/lib/node_modules/playwright"); }
const [url, shots] = process.argv.slice(2);
if (shots) mkdirSync(shots, { recursive: true });

const browser = await playwright.chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [], failures = [], checks = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => { if (m.type() === "error" && !/favicon|manifest|404/.test(m.text())) errors.push(m.text()); });
page.on("response", async (r) => { if (r.status() >= 400 && !/favicon|manifest/.test(r.url())) { let b = ""; try { b = (await r.text()).slice(0, 200); } catch (_) {} errors.push(`${r.status()} ${r.request().method()} ${r.url().replace(/^https?:\/\/[^/]+/, "")} ${b}`); } });
await page.goto(url, { waitUntil: "load" });
await page.waitForTimeout(400);

const shot = async (name) => { if (shots) await page.screenshot({ path: `${shots}/${name}.png`, fullPage: true }); };
const settle = async (root, ms = 120000) => {
  await page.waitForFunction((r) => { const n = document.querySelector(`#${r} .results`); return n && n.children.length > 0 && !n.querySelector(".pulse"); }, root, { timeout: ms })
    .catch(() => failures.push(`${root}: timed out`));
};
const errText = (root) => page.evaluate((r) => [...document.querySelectorAll(`#${r} .results .notice.err`)].map((n) => n.textContent).join(" | "), root);

async function desk(id, expect) {
  await page.click(`#t-${id}`);
  await page.waitForSelector(`#op-${id} .lede`);
  const vials = await page.$$eval(`#op-${id} .vial`, (vs) => vs.length);
  if (vials === 0 && id !== "cupel") failures.push(`${id}: no shelf`);
  for (let i = 0; i < vials; i++) {
    await page.click(`#op-${id} .vial >> nth=${i}`);
    await page.waitForTimeout(60);
    await settle(`op-${id}`);
    const e = await errText(`op-${id}`);
    const ok = await page.evaluate(([r, sel]) => !!document.querySelector(`#${r} ${sel}`), [`op-${id}`, expect[i] || ".touch"]);
    const title = await page.$eval(`#op-${id} .vial >> nth=${i}`, (b) => b.textContent);
    if (e) failures.push(`${id}/${title}: ${e}`);
    else if (!ok) failures.push(`${id}/${title}: no ${expect[i] || ".touch"}`);
    else checks.push(`${id}/${title}`);
    await shot(`${id}-${i}`);
  }
  if (id === "cupel") { await settle("op-cupel"); const ok = await page.$$eval("#op-cupel .bitcell", (c) => c.length); if (ok === 32) checks.push("cupel/drill"); else failures.push(`cupel: ${ok} bit cells`); await shot("cupel"); }
}

await desk("rebis", ["svg.seal.ok", "svg.seal.ok", ".bitwords", "svg.seal.ok", ".anf", ".qstrip"]);
await desk("aludel", ["svg.aludel-box .leaf", ".aludel-box .pending", ".aludel-box .refute", "svg.phase .bzero", "svg.phase .bzero"]);
await desk("tabula", [".tablet li"]);
await desk("cupel", []);
await desk("amalgam", ["svg.amalgam-line .mark"]);

// the trojan's trigger on the page: a = 0xDEADBEEF
await page.click("#t-rebis"); await page.click("#op-rebis .vial >> nth=2"); await settle("op-rebis");
const trig = await page.$$eval("#op-rebis .bitword code", (cs) => cs.map((c) => c.textContent));
if (trig.includes("0xDEADBEEF")) checks.push("rebis/trigger"); else failures.push(`rebis: trigger read as ${trig}`);

// a fact toggled: positions recomputed, a clause struck through by its override
await page.click("#t-tabula"); await page.click("#op-tabula .vial >> nth=0"); await settle("op-tabula");
await page.click("#op-tabula .fact:has-text('force_majeure')"); await page.waitForTimeout(100); await settle("op-tabula");
const struck = await page.$$eval("#op-tabula .tablet li.over b", (bs) => bs.map((b) => b.textContent));
if (struck.includes("C5")) checks.push("tabula/override"); else failures.push(`tabula: struck ${struck}`);
// a finding's scenario set with one click
await page.click("#op-tabula .clash.antinomy button"); await page.waitForTimeout(100); await settle("op-tabula");
const on = await page.$$eval("#op-tabula .fact.on", (fs) => fs.map((f) => f.textContent).sort().join(","));
if (on === "delivered,late") checks.push("tabula/scenario"); else failures.push(`tabula: scenario facts ${on}`);
await shot("tabula-scenario");

// Portuguese and the dark theme
await page.click("#lang"); await page.waitForTimeout(150);
const pt = await page.$eval("#t-tabula b", (b) => b.textContent);
if (pt === "Tábua") checks.push("lang/pt"); else failures.push(`lang: ${pt}`);
await page.click("#t-amalgam"); await page.waitForTimeout(200); await settle("op-amalgam");
const lede = await page.$eval("#op-amalgam .lede", (p) => p.textContent);
if (/ordem/.test(lede)) checks.push("lang/pt-desk"); else failures.push("lang: desk not redrawn in Portuguese");
await page.evaluate(() => document.documentElement.setAttribute("data-theme", "dark")); await page.waitForTimeout(100);
await page.click("#t-aludel"); await page.waitForTimeout(150); await page.click("#op-aludel .vial >> nth=3"); await settle("op-aludel");
await shot("aludel-dark-pt");
await page.click("#t-rebis"); await page.waitForTimeout(150);
// what the person wrote survives the language switch, and the result is computed again
const kept = await page.$eval("#op-rebis-b", (t) => /trig/.test(t.value));
if (kept) checks.push("lang/kept-text"); else failures.push("lang: the circuit text was lost");
await settle("op-rebis");
const ptVerdict = await page.$eval("#op-rebis .touch .verdict", (p) => p.textContent).catch(() => "");
if (/não é a mesma/.test(ptVerdict)) checks.push("lang/result-again"); else failures.push(`lang: result after switch: ${ptVerdict}`);
await page.click("#op-rebis .vial >> nth=2"); await settle("op-rebis");
await shot("rebis-dark-pt");

await browser.close();
console.log(JSON.stringify({ failures, checks, errors }));
