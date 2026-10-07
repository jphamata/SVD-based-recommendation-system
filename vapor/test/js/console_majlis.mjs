// The 0.16 surfaces driven end to end in headless Chromium, as a person
// would: a conversation is started, a message is edited (a second branch),
// the branches are walked with ‹ ›, an answer is regenerated, a message is
// pinned, the conversation is forked and compacted, searched, shared — the
// link read without the console — and revoked. Then the terminal: a sample
// file checked by the Mīzān, a pipe, Tab completion, the file editor, a read
// outside the jail refused. Portuguese and dark at the end. Any page error,
// error notice or missing element is reported.
// usage: node console_majlis.mjs URL [SHOTS_DIR]  → JSON {failures, checks, errors}
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
page.on("console", (m) => { if (m.type() === "error" && !/favicon|manifest|40[34]/.test(m.text())) errors.push(m.text()); });
// the revoked link's 403 is the point of the check, not an error
page.on("response", async (r) => { if (r.status() >= 400 && !/favicon|manifest|\/shared\//.test(r.url())) { let b = ""; try { b = (await r.text()).slice(0, 200); } catch (_) {} errors.push(`${r.status()} ${r.request().method()} ${r.url().replace(/^https?:\/\/[^/]+/, "")} ${b}`); } });
await page.goto(url, { waitUntil: "load" });
await page.waitForTimeout(400);

const shot = async (name) => { if (shots) await page.screenshot({ path: `${shots}/${name}.png`, fullPage: false }); };
const check = async (name, f) => {
  try { const ok = await f(); if (ok === false) failures.push(name); else checks.push(name); }
  catch (e) { failures.push(`${name}: ${String(e).split("\n")[0]}`); }
};
const wait = (fn, arg, ms = 20000) => page.waitForFunction(fn, arg, { timeout: ms });
const notices = () => page.evaluate(() => [...document.querySelectorAll("#mj .notice.err, #dw .notice.err")].map((n) => n.textContent).join(" | "));
const msgs = () => page.$$eval("#mj-log .mj-msg:not(.pending)", (ns) => ns.map((n) => ({ role: [...n.classList].find((c) => c === "user" || c === "assistant"), text: n.querySelector(".mj-text")?.textContent || "", branch: n.querySelector(".mj-branch span")?.textContent || "", pinned: n.classList.contains("pinned") })));
const idle = () => wait(() => !document.querySelector("#mj-log .pending"));
const action = async (i, label) => { await page.click(`#mj-log .mj-msg:not(.pending) >> nth=${i} >> button:has-text("${label}")`); await idle(); };

/* ---------------------------------------------------------- conversations */
await page.click("#t-majlis");
await page.waitForSelector("#mj-new");
await check("a new conversation", async () => { await page.click("#mj-new"); await page.waitForSelector("#mj-input"); return true; });
await check("Enter sends; the model answers", async () => {
  await page.fill("#mj-input", "hello there");
  await page.press("#mj-input", "Enter");
  await wait(() => [...document.querySelectorAll("#mj-log .mj-msg.assistant .mj-text")].some((n) => /ok: hello there/.test(n.textContent)));
  const m = await msgs();
  return m.length === 2 && m[0].role === "user";
});
await check("a second turn", async () => {
  await page.fill("#mj-input", "and a second question");
  await page.click("#mj-send");
  await wait(() => document.querySelectorAll("#mj-log .mj-msg.assistant").length === 2);
  return true;
});
await check("editing a message makes a branch ‹ 2/2 ›, and the old one is kept", async () => {
  await page.click('#mj-log .mj-msg >> nth=0 >> button:has-text("Edit")');
  await page.fill("#mj-log .mj-msg >> nth=0 >> textarea", "hello, edited");
  await page.click('#mj-log .mj-msg >> nth=0 >> button:has-text("Save and send")');
  await wait(() => document.querySelector("#mj-log .mj-msg .mj-branch span")?.textContent === "2/2");
  const m = await msgs();
  return m[0].text.includes("hello, edited") && m.length === 2 && m[1].text.includes("ok: hello, edited");
});
await check("‹ walks back to the first branch, with its whole continuation", async () => {
  await page.click('#mj-log .mj-msg >> nth=0 >> .mj-branch button >> nth=0');
  await wait(() => document.querySelector("#mj-log .mj-msg .mj-branch span")?.textContent === "1/2");
  const m = await msgs();
  return m[0].text.includes("hello there") && m.length === 4;
});
await check("another answer: a sibling of the assistant message", async () => {
  await action(3, "Another answer");
  await wait(() => document.querySelectorAll("#mj-log .mj-msg")[3]?.querySelector(".mj-branch span")?.textContent === "2/2");
  return true;
});
await check("pinning marks the message and the context counts it as pinned", async () => {
  await action(0, "Pin");
  await wait(() => document.querySelector("#mj-log .mj-msg")?.classList.contains("pinned"));
  return page.$eval("#mj-ctx", (n) => n.querySelectorAll(".mj-ctxbar i").length >= 3 && n.querySelector(".mj-ctxbar i.pin") !== null);
});
await check("the tree shows every message, the path in gold", async () => page.$eval(".mj-tree", (s) => s.querySelectorAll("circle").length >= 7 && s.querySelectorAll("circle.on").length === 4));
await page.evaluate(() => window.scrollTo(0, 0));
await shot("majlis-thread");
await check("compacting installs a summary that names what it covers", async () => {
  await page.click("#mj-compact");
  await page.waitForSelector(".mj-summary", { timeout: 20000 });
  return page.$eval("#mj-log", (l) => l.querySelectorAll(".mj-tag.dim").length >= 1);
});
await check("forking from a message: a second conversation, the list shows both", async () => {
  await action(1, "Fork from here");
  await wait(() => document.querySelectorAll(".mj-thread").length === 2);
  const m = await msgs();
  return m.length === 2;
});
await check("search finds a message across conversations", async () => {
  await page.fill(".mj-search", "edited");
  await page.press(".mj-search", "Enter");
  await page.waitForSelector(".mj-hit");
  const n = await page.$$eval(".mj-hit", (h) => h.length);
  await page.fill(".mj-search", ""); await page.press(".mj-search", "Enter");
  await page.waitForSelector(".mj-thread");
  return n >= 1;
});
await check("a share link is read without the console; revoking kills it", async () => {
  await page.click(".mj-thread >> nth=1");
  await page.waitForSelector("#mj-share-btn");
  await page.click("#mj-share-btn");
  await page.waitForSelector("#mj-share a");
  const href = await page.$eval("#mj-share a", (a) => a.getAttribute("href"));
  const before = await page.evaluate(async (h) => { const r = await fetch(h, { credentials: "omit" }); return [r.status, await r.text()]; }, href);
  await page.click("#mj-revoke");
  await page.waitForFunction(() => !document.querySelector("#mj-share a"));
  const after = await page.evaluate(async (h) => (await fetch(h, { credentials: "omit" })).status, href);
  return before[0] === 200 && /hello/.test(before[1]) && !/<script/i.test(before[1]) && after === 403;
});
await check("no error notice in the conversations", async () => (await notices()) === "");

/* --------------------------------------------------------------- terminal */
await page.click("#t-diwan");
await page.waitForSelector("#dw-input");
await check("a sample: the Mīzān proves the conservation law and refutes the damped one, in colour", async () => {
  await page.click('#dw .dw-samples button:has-text("wzn check oscillator.wzn")');
  await wait(() => /refuted/.test(document.querySelector("#dw-screen")?.textContent || ""));
  return page.$eval("#dw-screen", (s) => /proved/.test(s.textContent) && s.querySelector(".a-c6, .a-c1") !== null);
});
await check("a pipe and the session's files", async () => {
  await page.fill("#dw-input", "ls | wc");
  await page.press("#dw-input", "Enter");
  await wait(() => document.querySelectorAll("#dw-screen .dw-entry").length === 2 && !document.querySelector("#dw-screen .pulse"));
  return page.$eval("#dw-side", (s) => /oscillator\.wzn/.test(s.textContent));
});
await check("Tab completes a verb", async () => {
  await page.fill("#dw-input", "wz");
  await page.press("#dw-input", "Tab");
  await page.waitForTimeout(300);
  return (await page.inputValue("#dw-input")) === "wzn ";
});
await check("the editor opens a session file and saves it", async () => {
  await page.click('#dw-side button:has-text("oscillator.wzn")');
  await page.waitForSelector("#dw-editor");
  const text = await page.inputValue("#dw-editor");
  await page.fill("#dw-editor", text.replace("(claim kinetic", "(claim kinetic-energy").replace("(kinetic v)", "(kinetic-energy v)").replace("(kinetic v)", "(kinetic-energy v)"));
  await page.click("#dw-save");
  await page.fill("#dw-input", "wzn check oscillator.wzn");
  await page.press("#dw-input", "Enter");
  await wait(() => /kinetic-energy/.test(document.querySelector("#dw-screen")?.textContent || ""));
  return /claim/.test(text);
});
await check("a read outside the jail is refused, with an exit code", async () => {
  await page.fill("#dw-input", "cat /etc/passwd");
  await page.press("#dw-input", "Enter");
  await wait(() => /\/etc\/passwd: no such file/.test(document.querySelector("#dw-screen")?.textContent || ""));
  return page.$eval("#dw-screen", (s) => !/root:/.test(s.textContent) && s.querySelectorAll(".dw-code").length >= 1);
});
await page.evaluate(() => window.scrollTo(0, 0));
await shot("diwan");

/* --------------------------------------------------- Portuguese and dark */
await check("Portuguese relabels both panels and keeps their state", async () => {
  await page.click("#lang");
  await page.waitForTimeout(200);
  const term = await page.$eval("#dw h2", (h) => h.textContent);
  await page.click("#t-majlis");
  const tab = await page.$eval("#t-majlis b", (b) => b.textContent);
  const n = await page.$$eval(".mj-thread", (x) => x.length);
  return tab === "Conversas" && term === "Terminal" && n === 2;
});
await check("dark theme", async () => { await page.click("#theme"); await page.waitForTimeout(200); return (await page.$eval("html", (h) => h.dataset.theme)) === "dark"; });
await page.click(".mj-thread >> nth=1");
await wait(() => document.querySelectorAll(".mj-thread")[1]?.classList.contains("on") && document.querySelector("#mj-ctx .mj-ctxbar") && document.querySelectorAll("#mj-log .mj-msg").length === 4);
await page.evaluate(() => window.scrollTo(0, 0));
await shot("majlis-escuro");

await browser.close();
console.log(JSON.stringify({ failures, checks, errors }));
