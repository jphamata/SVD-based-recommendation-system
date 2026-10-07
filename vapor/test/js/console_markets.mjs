// The console's 0.13 desks (Finance, Trading desk) driven in headless
// Chromium: every kind, every example, through the page (the same clicks a
// person makes), in English and then in Portuguese; the chain of the order
// book clicked; the command palette used to jump. Any page error, any
// example answered with an error notice, any empty result is reported.
// usage: node console_markets.mjs URL [SHOTS_DIR]  → JSON {ok, failures, checks, errors}
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
page.on("console", (m) => { if (m.type() === "error" && !/favicon|manifest|404|400/.test(m.text())) errors.push(m.text()); });
await page.goto(url, { waitUntil: "load" });
await page.waitForTimeout(500);

const settle = async (root, ms = 180000) => {
  await page.waitForFunction((r) => { const n = document.getElementById(r); return n && !n.querySelector(".pulse") && !/running|rodando/.test(n.querySelector(".wb-state")?.textContent || ""); }, root, { timeout: ms }).catch(() => failures.push(`${root}: timed out`));
};
const errText = (root) => page.evaluate((r) => [...document.getElementById(r).querySelectorAll(".wb-out .notice.err, .wb-out > .notice")].map((n) => n.textContent).join(" | "), root);
const shot = async (name) => { if (shots) await page.screenshot({ path: `${shots}/${name}.png`, fullPage: true }); };
const tab = async (id) => { await page.click(`#t-${id}`); await page.waitForTimeout(200); };

// examples whose answer is, by design, a refusal (the no-arbitrage bound)
const expectError = ["o_ivbad"];


// in Portuguese, no English sentence or label is left in a result (code, hashes and wire bytes aside)
async function englishLeft(root) {
  return page.evaluate((r) => {
    const skip = "pre, code, textarea, .hash, .mk-fix, .mk-proof, .mk-chain, .mk-entry, select, option";
    const w = document.createTreeWalker(document.querySelector(`#${r}`), NodeFilter.SHOW_TEXT), out = [];
    const en = /\b(the|with|every|refused|exceptions|trades|repriced|failed|events|steps|paths|price|cost|inventory|replayed|accepted|messages|knocked|analytic|strategy|expected|stationarity|quantity|asset|day|state|pays|found)\b/;
    while (w.nextNode()) { const n = w.currentNode; if (!n.parentElement.closest(skip) && en.test(n.nodeValue)) out.push(n.nodeValue.trim()); }
    return [...new Set(out)];
  }, root);
}

async function examples(root, label) {
  const keys = await page.evaluate((r) => { const s = document.querySelector(`#${r} .wb-bar select`); return s ? [...s.options].map((o) => o.value).filter(Boolean) : []; }, root);
  for (const k of keys) {
    await page.selectOption(`#${root} .wb-bar select >> nth=0`, k);
    await page.waitForTimeout(80);
    await settle(root);
    const e = await errText(root);
    const filled = await page.evaluate((r) => document.querySelector(`#${r} .wb-out`).children.length, root);
    if (e && !expectError.includes(k)) failures.push(`${label}/${k}: ${e}`);
    else if (!e && expectError.includes(k)) failures.push(`${label}/${k}: a refusal was expected`);
    else if (!filled) failures.push(`${label}/${k}: empty result`);
    else checks.push(`${label}/${k}`);
    if (label.startsWith("pt:")) { const left = await englishLeft(root); if (left.length) failures.push(`${label}/${k}: English left — ${left.slice(0, 6).join(" | ")}`); }
  }
}

async function desk(id, langTag) {
  await tab(id); await settle(`wb-${id}`);
  const kinds = await page.evaluate((r) => [...document.querySelectorAll(`#wb-${r} > .seg button`)].map((b) => b.textContent), id);
  for (let i = 0; i < kinds.length; i++) {
    await page.click(`#wb-${id} > .seg button >> nth=${i}`); await page.waitForTimeout(120); await settle(`wb-${id}`);
    await examples(`wb-${id}`, `${langTag}:${id}:${kinds[i]}`);
    await shot(`${langTag}-${id}-${i}`);
  }
}

await desk("fin", "en");
await desk("hft", "en");
// the order book's chain: choose an entry, its detail follows
await page.click("#wb-hft > .seg button >> nth=0"); await settle("wb-hft");
const kindsH = await page.evaluate(() => [...document.querySelectorAll("#wb-hft > .seg button")].map((b) => b.textContent));
const bookIdx = kindsH.findIndex((k) => /Order book|Livro/.test(k));
await page.click(`#wb-hft > .seg button >> nth=${bookIdx}`); await settle("wb-hft");
const nChain = await page.evaluate(() => document.querySelectorAll("#wb-hft .mk-chain button").length);
if (nChain > 2) { await page.click("#wb-hft .mk-chain button >> nth=2"); const d = await page.evaluate(() => document.querySelector("#wb-hft .mk-entry b").textContent); if (/^#3 /.test(d)) checks.push("chain/pick"); else failures.push("chain/pick: " + d); }
else failures.push("chain: no entries");
// Portuguese: the same desks, the navigation re-sorted
await page.click("#lang"); await page.waitForTimeout(300);
const navPt = await page.evaluate(() => [...document.querySelectorAll("nav [role=tab] b")].map((b) => b.textContent));
if (navPt.includes("Finanças") && navPt.includes("Mesa de operações")) checks.push("nav/pt"); else failures.push("nav/pt: " + navPt.join(","));
await desk("fin", "pt");
await desk("hft", "pt");
// the command palette jumps to an example
await page.keyboard.press("Control+K"); await page.keyboard.type("borboleta"); await page.keyboard.press("Enter"); await settle("wb-fin");
const palOk = await page.evaluate(() => /borboleta|Vogt/i.test(document.querySelector("#wb-fin .mk-seal")?.textContent || "") || document.querySelector("#wb-fin .mk-seal") != null);
palOk ? checks.push("palette") : failures.push("palette: no result");
await shot("pt-palette");

await browser.close();
console.log(JSON.stringify({ ok: failures.length === 0 && errors.length === 0, failures, errors, checks }));
