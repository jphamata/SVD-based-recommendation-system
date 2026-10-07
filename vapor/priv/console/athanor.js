"use strict";
/* athanor.js — the 0.14 open bench (docs/CONSOLE.md §0.14): the Workspace
   (any problem in Alembic or in words: searched by the Athanor and steered
   by hand, a game played, a Crucible system run, a scene composed), the
   Crucible and the Assay. Every result ends on the touchstone: a gold
   streak for each check that holds, a lead streak for each that does not.
   Runs after the page's scripts and uses their helpers ($, el, t, api,
   svgEl, lineChart, legend, editor, table, stats, stat, wbNum, busy, shown,
   openPanel, PALETTE_ITEMS, rerender, download, SceneEngine). */

/* ================================================================== words */
Object.assign(I18N.en, {
  g_open: "Open bench", work: "Workspace", work_s: "any problem, searched and checked", crucible: "Crucible", crucible_s: "your own system, with its evidence",
  assay: "Assay", assay_s: "model evaluations: signal or noise",
  work_lede: "Write a problem in Alembic — a space, an objective or a claim, a game, a system — or describe it in words and let a model draft it for you to read. The Athanor searches with a portfolio of strategies, a random search spends the same budget as the control, and the result is rubbed on the touchstone.",
  crucible_lede: "Each domain takes your own system and answers with evidence that needs no reference solution: observed orders of convergence, conservation laws proved over ℚ, theorems that hold for any input, two methods that must agree, and controls a wrong method would fail.",
  assay_lede: "Paste or drop the per-item results of your evaluations. Each tool says whether a difference, a ranking, a score or an agreement is signal or noise — with the interval, the test and the size of effect this much data can see.",
  ask_ph: "Describe a problem in words — “the shortest Golomb ruler with 8 marks”, “a strategy that…” — or write it below",
  formalize: "Draft it", formalizing: "drafting…", no_mind: "No language model is configured on this server (VAPOR_MIND or --model). Write the problem below — the shelf has a starting point for each kind.",
  readback: "The model read its own program back as:", attempts: (n) => `loaded after ${n} attempt${n === 1 ? "" : "s"}`,
  run: "Run", stop: "Stop", resume: "Resume", more: "+ budget", budget: "budget", seed: "seed", verify_again: "Check again", save_cert: "Save certificate",
  detecting: "reading…", k_athanor: "Athanor", k_athanor_d: "a search", k_game: "Game", k_game_d: "two players", k_crucible: "Crucible", k_alembic: "Alembic", k_alembic_d: "a program", k_scene: "Scene", k_scene_d: "operations",
  furnace_empty: "Nothing in the furnace yet. Choose something from the shelf, or write a problem, and run it.",
  evals: (a, b) => `${a} of ${b} evaluations`, best: "best", control: "random search, same budget",
  strategies: "share of the budget · improvements", running_s: "the furnace is lit", done_s: "settled",
  finalists: "Finalists", candidate: "candidate", value: "value", by: "found by", hand: "your hand",
  pin: "Keep this candidate in the archive", ban: "Never evaluate this candidate again", edit: "Copy into the proposal box",
  propose_ph: "Propose a candidate — a literal like [0, 1, 4, 9, 11], or an expression for a program space", propose: "Propose", ask_mind: "Ask the model for ideas",
  measure_title: "Measured outside: run these and type what you measured", measure: "Record",
  touch_title: "Touchstone", stone_note: "A gold streak: the check holds. Lead: it does not. Hover a streak for what was checked.",
  c_problem: "Problem", c_membership: "In the space", c_value: "Value re-computed", c_counterexample: "Counterexample", c_control: "Beats chance", c_holdout: "Survives the holdout", c_journal: "Replayable", c_proof: "Proof",
  control_beat: (q) => `random search never matched it (chance per sample ≤ ${q})`, control_easy: (k, n) => `random search matched it ${k} times in ${n}`,
  journal_d: "the journal's root hash: re-run from the seed and the outside proposals to get the same",
  game_solve: "Solve exactly", game_search: "Search (MCTS)", game_learn: "Learn from self-play", game_new: "New game", your_move: "Your move", vapor_plays: (m) => `vapor plays ${m}`,
  scene_words_ph: "Direct in words: “a slow comet, falling snow, dusk”", scene_direct: "Direct", scene_hint: "Scene operations run in the canvas; each line is kept in the document's log.",
  eval_ph: "Evaluate an expression against this program, e.g. fib(20)", evaluate: "Evaluate",
  domains: "Domains", tools: "Tools", data: "Data", drop_data: "Drop a CSV, TSV or JSON-lines file", says: "", laws_found: "Conserved", evidence: "Evidence",
  fields: { math: "Mathematics", cs: "Computing", science: "Science", ai: "AI", finance: "Finance", scene: "Scenes" }
});
Object.assign(I18N.pt, {
  g_open: "Bancada aberta", work: "Espaço de trabalho", work_s: "qualquer problema, buscado e conferido", crucible: "Crisol", crucible_s: "o seu próprio sistema, com a evidência",
  assay: "Ensaio", assay_s: "avaliações de modelos: sinal ou ruído",
  work_lede: "Escreva um problema em Alembic — um espaço, um objetivo ou uma afirmação, um jogo, um sistema — ou descreva-o em palavras e deixe um modelo rascunhá-lo para você ler. O Athanor busca com um portfólio de estratégias, uma busca aleatória gasta o mesmo orçamento como controle, e o resultado é esfregado na pedra de toque.",
  crucible_lede: "Cada domínio aceita o seu próprio sistema e responde com evidência que não precisa de solução de referência: ordens de convergência observadas, leis de conservação provadas sobre ℚ, teoremas válidos para qualquer entrada, dois métodos que precisam concordar e controles que um método errado reprovaria.",
  assay_lede: "Cole ou solte os resultados item a item das suas avaliações. Cada ferramenta diz se uma diferença, um ranking, um escore ou uma concordância é sinal ou ruído — com o intervalo, o teste e o tamanho de efeito que esse volume de dados enxerga.",
  ask_ph: "Descreva um problema em palavras — “a régua de Golomb mais curta com 8 marcas”, “uma estratégia que…” — ou escreva-o abaixo",
  formalize: "Rascunhar", formalizing: "rascunhando…", no_mind: "Nenhum modelo de linguagem está configurado neste servidor (VAPOR_MIND ou --model). Escreva o problema abaixo — a prateleira tem um ponto de partida para cada tipo.",
  readback: "O modelo leu o próprio programa de volta como:", attempts: (n) => `carregado após ${n} tentativa${n === 1 ? "" : "s"}`,
  run: "Rodar", stop: "Parar", resume: "Retomar", more: "+ orçamento", budget: "orçamento", seed: "semente", verify_again: "Conferir de novo", save_cert: "Salvar certificado",
  detecting: "lendo…", k_athanor: "Athanor", k_athanor_d: "uma busca", k_game: "Jogo", k_game_d: "dois jogadores", k_crucible: "Crisol", k_alembic: "Alembic", k_alembic_d: "um programa", k_scene: "Cena", k_scene_d: "operações",
  furnace_empty: "Nada na fornalha ainda. Escolha algo na prateleira, ou escreva um problema, e rode.",
  evals: (a, b) => `${a} de ${b} avaliações`, best: "melhor", control: "busca aleatória, mesmo orçamento",
  strategies: "fatia do orçamento · melhorias", running_s: "a fornalha está acesa", done_s: "assentado",
  finalists: "Finalistas", candidate: "candidato", value: "valor", by: "achado por", hand: "a sua mão",
  pin: "Manter este candidato no arquivo", ban: "Nunca mais avaliar este candidato", edit: "Copiar para a caixa de proposta",
  propose_ph: "Proponha um candidato — um literal como [0, 1, 4, 9, 11], ou uma expressão num espaço de programas", propose: "Propor", ask_mind: "Pedir ideias ao modelo",
  measure_title: "Medido fora: rode estes e digite o que mediu", measure: "Registrar",
  touch_title: "Pedra de toque", stone_note: "Risco de ouro: a verificação vale. Chumbo: não vale. Passe o mouse num risco para ver o que foi conferido.",
  c_problem: "Problema", c_membership: "No espaço", c_value: "Valor recalculado", c_counterexample: "Contraexemplo", c_control: "Vence o acaso", c_holdout: "Resiste ao holdout", c_journal: "Reexecutável", c_proof: "Prova",
  control_beat: (q) => `a busca aleatória nunca chegou lá (chance por amostra ≤ ${q})`, control_easy: (k, n) => `a busca aleatória chegou lá ${k} vezes em ${n}`,
  journal_d: "a raiz do diário: reexecutar da semente e das propostas externas dá a mesma",
  game_solve: "Resolver exatamente", game_search: "Buscar (MCTS)", game_learn: "Aprender jogando consigo", game_new: "Novo jogo", your_move: "Sua vez", vapor_plays: (m) => `o vapor joga ${m}`,
  scene_words_ph: "Dirija em palavras: “um cometa lento, neve caindo, entardecer”", scene_direct: "Dirigir", scene_hint: "As operações de cena rodam na tela; cada linha fica no registro do documento.",
  eval_ph: "Avalie uma expressão contra este programa, p. ex. fib(20)", evaluate: "Avaliar",
  domains: "Domínios", tools: "Ferramentas", data: "Dados", drop_data: "Solte um arquivo CSV, TSV ou JSON por linha", says: "", laws_found: "Conservado", evidence: "Evidência",
  fields: { math: "Matemática", cs: "Computação", science: "Ciência", ai: "IA", finance: "Finanças", scene: "Cenas" }
});

/* ================================================================ shared */
const W = { info: null };
async function workInfo() { if (!W.info) W.info = await api("/v1/vapor/workspace"); return W.info; }
const fmtv = (v) => (v == null ? "—" : typeof v === "number" ? wbNum(v, 5) : String(v));
const SCENE_STARTER = `# a free scene: every number may be an expression of t (seconds), i, n, u
add glow sun { x: 0.78, y: 0.24 + 0.03*sin(t/2), r: 0.13, color: "#ffcc66" }
add particles stars { count: 160, x: noise(i, 1), y: 0.62*noise(i, 2), r: 0.0012 + 0.002*noise(i, 3), alpha: 0.35 + 0.65*tri(t/3 + noise(i, 4)), color: "#ffffff" }
add trail comet { x: 0.5 + 0.34*cos(0.7*t), y: 0.42 + 0.18*sin(0.7*t), length: 2.5, width: 0.004, color: hsl(190 + 60*sin(t), 85, 70) }
add text title { x: 0.5, y: 0.9, text: "vapor", size: 0.06, color: "#ece4d3" }
set world.weather = "snow"`;
const isScene = (txt) => { const l = txt.split("\n").map((s) => s.replace(/#.*$/, "").trim()).find((s) => s); return !!l && /^(add|set|remove|clear|at|direct)\b/.test(l); };

/* a streak on the touchstone: a rough, tapered smear, the same for the same check (seeded by its name) */
function streak(name, ok, detail) {
  let h = 2166136261; for (const c of name) h = Math.imul(h ^ c.charCodeAt(0), 16777619) >>> 0;
  const rnd = () => { h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0; return (h & 0xffff) / 65536; };
  const pts = [], n = 18;
  for (let k = 0; k <= n; k++) { const x = 6 + (138 * k) / n, w = Math.sin((Math.PI * k) / n) ** 0.6 * (6 + 3 * rnd()); pts.push([x, 11 - w / 2 - rnd() * 1.5, 11 + w / 2 + rnd() * 1.5]); }
  const d = "M" + pts.map((p) => `${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" L") + " L" + pts.reverse().map((p) => `${p[0].toFixed(1)} ${p[2].toFixed(1)}`).join(" L") + " Z";
  const svg = svgEl("svg", { viewBox: "0 0 150 22", "aria-hidden": "true" });
  svg.append(svgEl("path", { class: "mk", d, opacity: ok ? "0.95" : "0.85" }));
  if (ok) svg.append(svgEl("path", { d, fill: "none", stroke: "rgba(255,240,190,.35)", "stroke-width": "0.6" }));
  const box = el("div", { class: "streak" + (ok ? "" : " fail"), title: detail || "", tabindex: "0", role: "img", "aria-label": `${name}: ${ok ? "✓" : "✗"} ${detail || ""}` }, svg, el("b", { text: name }), el("span", { text: detail || "" }));
  return box;
}
function touchstone(verdict, checks, negative = false) {
  const wrap = el("section", { class: "touch", "aria-label": t("touch_title") });
  if (verdict) wrap.append(el("p", { class: "verdict" + (negative ? " neg" : ""), text: verdict }));
  if (checks && checks.length) {
    wrap.append(el("div", { class: "stone" }, ...checks.map((c) => streak(c.name, c.ok, c.detail))));
    wrap.append(el("p", { class: "stone-note", text: t("stone_note") }));
  }
  return wrap;
}
const evidenceChecks = (ev) => (ev || []).map((e) => ({ name: e.check, ok: e.ok, detail: e.detail }));

/* ================================================================ furnace */
const STRAT_COLOR = { exhaustive: "#D9AE3F", random: "#8D877C", anneal: "#E0694C", evolve: "#6FB8A6", cmaes: "#7FA7D9", bayes: "#C79BE8", mind: "#F2D27A", human: "#FFFFFF", start: "#B8B2A8" };
function furnaceView(F) {
  const box = el("div", { class: "furnace" });
  if (!F.sparks.length && !F.trace.length) { box.append(el("div", { class: "empty-furnace", text: F.status === "running" ? t("running_s") + "…" : t("furnace_empty") })); return box; }
  const w = 560, h = 300, pad = { l: 46, r: 12, t: 12, b: 26 };
  const pts = F.sparks.filter((s) => typeof s.value === "number" && isFinite(s.value));
  const all = pts.map((s) => s.value).concat(F.trace.map((r) => r[1])).concat(F.control != null ? [F.control] : []);
  let y0 = Math.min(...all), y1 = Math.max(...all); if (y0 === y1) { y0 -= 1; y1 += 1; }
  const x1 = Math.max(F.evals, 1);
  const better_up = F.sense !== "min";
  const X = (i) => pad.l + (i / x1) * (w - pad.l - pad.r);
  const Y = (v) => { const f = (v - y0) / (y1 - y0); return better_up ? h - pad.b - f * (h - pad.t - pad.b) : pad.t + f * (h - pad.t - pad.b); };
  const svg = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": t("evals", F.evals, F.budget) });
  for (let k = 0; k <= 4; k++) { const v = y0 + ((y1 - y0) * k) / 4, yy = Y(v); svg.append(svgEl("line", { class: "ax", x1: pad.l, x2: w - pad.r, y1: yy, y2: yy })); const tx = svgEl("text", { class: "axt", x: pad.l - 6, y: yy + 4, "text-anchor": "end" }); tx.textContent = +v.toPrecision(3); svg.append(tx); }
  const tx = svgEl("text", { class: "axt", x: w - pad.r, y: h - 6, "text-anchor": "end" }); tx.textContent = t("evals", F.evals, F.budget); svg.append(tx);
  // sparks: newer ones hotter
  const n = pts.length;
  pts.forEach((s, k) => { const age = (n - k) / Math.max(n, 1); svg.append(svgEl("circle", { cx: X(s.i).toFixed(1), cy: Y(s.value).toFixed(1), r: (1.6 + 1.4 * (1 - age)).toFixed(2), fill: STRAT_COLOR[s.by] || "#ccc", opacity: (0.25 + 0.6 * (1 - age)).toFixed(2) })); });
  if (F.control != null) { const yy = Y(F.control); svg.append(svgEl("line", { class: "ctl", x1: pad.l, x2: w - pad.r, y1: yy, y2: yy })); const ct = svgEl("text", { class: "ctlt", x: pad.l + 6, y: yy - 5 }); ct.textContent = t("control"); svg.append(ct); }
  if (F.trace.length) {
    let d = "", prevY = null;
    F.trace.forEach(([i, v], k) => { const xx = X(i), yy = Y(v); d += k === 0 ? `M${xx.toFixed(1)} ${yy.toFixed(1)}` : ` L${xx.toFixed(1)} ${prevY.toFixed(1)} L${xx.toFixed(1)} ${yy.toFixed(1)}`; prevY = yy; });
    d += ` L${X(F.evals).toFixed(1)} ${prevY.toFixed(1)}`;
    svg.append(svgEl("path", { class: "cond", d }));
    const [li, lv] = F.trace[F.trace.length - 1]; svg.append(svgEl("circle", { cx: X(li), cy: Y(lv), r: 4.5, fill: "#D9AE3F" }));
  }
  box.append(svg);
  const best = F.trace.length ? F.trace[F.trace.length - 1][1] : null;
  box.append(el("div", { class: "furnace-cap" }, el("span", {}, t("best") + " ", el("b", { text: fmtv(best) })), el("span", { text: F.status === "running" ? t("running_s") : t("done_s") + (F.reason ? ` · ${F.reason}` : "") })));
  const strat = Object.entries(F.strategies || {}).filter(([, s]) => s.evals > 0);
  const tot = strat.reduce((a, [, s]) => a + s.evals, 0) || 1;
  if (strat.length) {
    box.append(el("div", { class: "meters", "aria-label": t("strategies") }, ...strat.sort((a, b) => b[1].evals - a[1].evals).map(([k, s]) => {
      const bar = el("i", {}, el("s")); bar.firstChild.style.width = `${(100 * s.evals) / tot}%`; bar.firstChild.style.background = STRAT_COLOR[k] || "#ccc";
      return el("div", { class: "meter" }, el("span", { text: k }), bar, el("span", { text: `${Math.round((100 * s.evals) / tot)} %${s.improved ? " · ↑" + s.improved : ""}` }));
    })));
  }
  return box;
}

/* ================================================================ the bench */
function buildBench(root, mode) {
  root.replaceChildren();
  root.append(el("h2", { text: t(mode === "work" ? "work" : mode) }), el("p", { class: "lede", text: t(mode + "_lede") }));
  const S = { mode, run: 0, session: null, F: null, cert: null, kind: null, scene: null };
  const ta = editor(`ed-${mode}`, 18);
  const kindEl = el("span", { class: "kind", "aria-live": "polite" });
  const errEl = el("p", { class: "err-line", hidden: "" });
  const runB = el("button", { class: "primary", type: "button", text: t("run") });
  const stopB = el("button", { class: "quiet", type: "button", text: t("stop"), hidden: "" });
  const moreB = el("button", { class: "quiet", type: "button", text: t("more"), hidden: "" });
  const budget = el("input", { type: "number", min: "1", step: "1", "aria-label": t("budget") });
  const seed = el("input", { type: "number", min: "1", step: "1", value: "1", "aria-label": t("seed") });
  const right = el("div", { class: "vessel" });
  const below = el("div", { class: "results" });
  let domain = mode === "crucible" ? "laws" : mode === "assay" ? "compare" : null;

  // the ask bar (workspace only)
  if (mode === "work") {
    const ask = el("input", { type: "text", placeholder: t("ask_ph"), "aria-label": t("ask_ph") });
    const go = el("button", { class: "quiet", type: "button", text: t("formalize") });
    const readback = el("div", { class: "readback", hidden: "" });
    const doAsk = async () => {
      const words = ask.value.trim(); if (!words) return;
      const info = await workInfo();
      readback.hidden = false;
      if (!info.mind) { readback.replaceChildren(el("span", { text: t("no_mind") })); return; }
      go.disabled = true; go.textContent = t("formalizing");
      try {
        const r = await api("/v1/vapor/formalize", { words });
        ta.value = r.program; onEdit();
        readback.replaceChildren(el("b", { text: t("readback") + " " }), el("span", { text: r.back_translation || "—" }), el("div", { class: "muted", text: `${t("attempts", r.attempts)} · ${info.mind}` }));
      } catch (e) { readback.replaceChildren(el("span", { class: "bad-t", text: e.message })); }
      go.disabled = false; go.textContent = t("formalize");
    };
    go.onclick = doAsk; ask.onkeydown = (e) => { if (e.key === "Enter") doAsk(); };
    root.append(el("div", { class: "work-ask" }, ask, go), readback);
  }

  // the shelf: starting points
  const shelf = el("div", { class: "shelf" });
  root.append(shelf);
  workInfo().then((info) => {
    const vial = (label, about, field, onPick) => { const b = el("button", { class: "vial", type: "button", "data-field": field, title: about, text: label }); b.onclick = onPick; return b; };
    if (mode === "work") {
      const byField = {};
      info.athanor.forEach((e) => (byField[e.field] = byField[e.field] || []).push(e));
      ["math", "cs", "science", "ai", "finance"].forEach((f) => {
        if (!byField[f]) return;
        shelf.append(el("span", { class: "field", text: t("fields")[f] }), el("span", { class: "vials" }, ...byField[f].map((e) => vial(e.title, e.about, f, () => load(e.text)))));
      });
      shelf.append(el("span", { class: "field", text: t("fields").scene }), el("span", { class: "vials" }, vial(t("k_scene"), t("scene_hint"), "science", () => load(SCENE_STARTER))));
      if (!ta.value) load(info.athanor[0].text);
    } else if (mode === "crucible") {
      shelf.append(el("span", { class: "field", text: t("domains") }), el("span", { class: "vials" }, ...info.crucible.map((k) => vial(k.kind, k.about, "science", () => { domain = k.kind; load(k.example); }))));
      if (!ta.value) { domain = "laws"; load(info.crucible.find((k) => k.kind === "laws").example); }
    } else {
      shelf.append(el("span", { class: "field", text: t("tools") }), el("span", { class: "vials" }, ...info.assay.map((k) => vial(k.tool, k.about, "ai", () => { domain = k.tool; load(k.example); }))));
      if (!ta.value) { domain = "compare"; load(info.assay[0].example); }
    }
  }).catch((e) => shelf.append(el("p", { class: "notice err", text: e.message })));

  const left = el("div", { class: "vessel" },
    el("div", { class: "vessel-head" }, kindEl),
    ta, errEl,
    el("div", { class: "vessel-foot" }, runB, stopB, moreB,
      mode === "work" ? el("label", {}, t("budget"), budget) : "", mode !== "crucible" ? el("label", {}, t("seed"), seed) : ""));
  if (mode === "assay") {
    const file = el("input", { type: "file", accept: ".csv,.tsv,.txt,.json,.jsonl", hidden: "" });
    const drop = el("button", { class: "quiet", type: "button", text: t("drop_data") });
    drop.onclick = () => file.click();
    file.onchange = async () => { if (file.files[0]) { ta.value = await file.files[0].text(); onEdit(); } };
    ta.addEventListener("dragover", (e) => e.preventDefault());
    ta.addEventListener("drop", async (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) { ta.value = await f.text(); onEdit(); } });
    left.lastChild.append(drop, file);
  }
  root.append(el("div", { class: "bench" }, left, right), below);
  right.replaceChildren(furnaceView({ sparks: [], trace: [], status: "idle" }));

  function load(text) { ta.value = text.trim() + "\n"; budget.value = ""; onEdit(); ta.focus({ preventScroll: true }); ta.setSelectionRange(0, 0); ta.scrollTop = 0; }

  // what the text is (the workspace decides by itself)
  let detT = 0;
  async function onEdit() {
    errEl.hidden = true;
    if (mode === "crucible") { S.kind = { kind: "crucible", domain }; kindEl.replaceChildren(el("b", { text: t("k_crucible") }), " · " + domain); return; }
    if (mode === "assay") { S.kind = { kind: "assay", tool: domain }; kindEl.replaceChildren(el("b", { text: t("assay") }), " · " + domain); return; }
    clearTimeout(detT);
    detT = setTimeout(async () => {
      const txt = ta.value;
      if (isScene(txt)) { S.kind = { kind: "scene" }; kindEl.replaceChildren(el("b", { text: t("k_scene") }), " · " + t("k_scene_d")); return; }
      kindEl.textContent = t("detecting");
      try {
        S.kind = await api("/v1/vapor/detect", { text: txt });
        const k = S.kind.kind;
        kindEl.replaceChildren(el("b", { text: t("k_" + k) }), " · " + (k === "crucible" ? S.kind.domain : t("k_" + k + "_d")));
        const m = /^\s*budget\s*=\s*(\d+)/m.exec(txt); if (m && !budget.value) budget.placeholder = m[1];
      } catch (e) { kindEl.textContent = ""; }
    }, 280);
  }
  ta.addEventListener("input", onEdit);
  ta.addEventListener("run", () => runB.click());

  /* ---------------------------------------------------------------- run */
  runB.onclick = async () => {
    const txt = ta.value, my = ++S.run;
    if (S.session) api(`/v1/vapor/athanor/${S.session}`, { action: "close" }).catch(() => {});
    S.session = null; S.cert = null; below.replaceChildren(); errEl.hidden = true; stopB.hidden = true; moreB.hidden = true;
    if (S.scene) { S.scene.destroy(); S.scene = null; }
    const kind = (S.kind && S.kind.kind) || (isScene(txt) ? "scene" : "athanor");
    try {
      if (kind === "athanor") await runAthanor(txt, my);
      else if (kind === "game") await runGame(txt, null);
      else if (kind === "crucible") await runCrucible(txt, S.kind.domain || domain);
      else if (kind === "assay") await runAssay(txt, domain);
      else if (kind === "scene") await runScene(txt);
      else await runAlembic(txt);
    } catch (e) { errEl.hidden = false; errEl.textContent = e.message; right.replaceChildren(furnaceView({ sparks: [], trace: [], status: "idle" })); }
  };

  /* ------------------------------------------------------------- athanor */
  async function runAthanor(txt, my) {
    const body = { text: txt, seed: +seed.value || 1 }; if (budget.value) body.budget = +budget.value;
    const snap = await api("/v1/vapor/athanor", body);
    S.session = snap.id;
    S.F = { sparks: [], trace: [], status: "running", evals: 0, budget: snap.budget, sense: snap.sense, strategies: {}, seen: new Set(), control: null };
    stopB.hidden = false; stopB.textContent = t("stop");
    stopB.onclick = async () => { const st = S.F.status === "running"; await api(`/v1/vapor/athanor/${S.session}`, { action: st ? "stop" : "resume" }); if (!st) poll(my); };
    moreB.onclick = async () => { await api(`/v1/vapor/athanor/${S.session}`, { action: "extend", n: Math.max(500, Math.round(S.F.budget / 2)) }); moreB.hidden = true; below.replaceChildren(); poll(my); };
    absorb(snap); poll(my);
  }
  function absorb(snap) {
    const F = S.F;
    (snap.sparks || []).forEach(([i, value, by]) => { if (!F.seen.has(i)) { F.seen.add(i); F.sparks.push({ i, value, by }); F.since = Math.max(F.since || 0, i); } });
    F.sparks.sort((a, b) => a.i - b.i); if (F.sparks.length > 2500) F.sparks = F.sparks.filter((_, k) => k % 2 === 0 || k > F.sparks.length - 500);
    Object.assign(F, { trace: snap.trace || [], status: snap.status, reason: snap.reason, evals: snap.evaluations, budget: snap.budget, strategies: snap.strategies, sense: snap.sense });
    right.replaceChildren(furnaceView(F));
    stopB.textContent = snap.status === "running" ? t("stop") : t("resume");
    stopB.hidden = !(snap.status === "running" || (snap.status === "stopped" && snap.reason === "stopped by the user"));
    renderLive(snap);
  }
  async function poll(my) {
    while (S.run === my && S.session) {
      let snap;
      try { snap = await api(`/v1/vapor/athanor/${S.session}?since=${S.F.since || 0}`); } catch (e) { errEl.hidden = false; errEl.textContent = e.message; return; }
      if (S.run !== my) return;
      absorb(snap);
      if (snap.pending && snap.pending.length) { renderPending(snap.pending); return; }
      if (snap.status !== "running" && !snap.queue) { await settle(my); return; }
      await new Promise((r) => setTimeout(r, 380));
    }
  }
  const finalsBox = el("div", { class: "finals" }), handBox = el("div"), pendBox = el("div"), touchBox = el("div");
  function renderLive(snap) {
    if (!below.contains(touchBox)) below.append(touchBox, finalsBox, handBox, pendBox);
    const rows = (snap.top || []).map((c) => {
      const tr = el("tr", { class: c.pinned ? "pinned" : "" },
        el("td", { class: "c" }, c.candidate, c.shown ? el("div", { class: "muted", text: c.shown }) : ""), el("td", { class: "v", text: c.valid === false ? `✗ ${fmtv(c.violation)}` : fmtv(c.value) }),
        el("td", { text: c.by || "" }));
      const hand = el("span", { class: "hand" });
      const b = (sym, tip, f) => { const x = el("button", { type: "button", title: tip, "aria-label": tip, text: sym }); x.onclick = f; return x; };
      hand.append(b("★", t("pin"), () => api(`/v1/vapor/athanor/${S.session}`, { action: "pin", candidate: c.candidate }).then(refresh)),
        b("✕", t("ban"), () => api(`/v1/vapor/athanor/${S.session}`, { action: "ban", candidate: c.candidate }).then(refresh)),
        b("✎", t("edit"), () => { prop.value = c.candidate; prop.focus(); }));
      tr.append(el("td", {}, hand)); return tr;
    });
    const tb = el("table", {}, el("thead", {}, el("tr", {}, ...[t("candidate"), t("value"), t("by"), t("hand")].map((h) => el("th", { text: h })))), el("tbody", {}, ...rows));
    finalsBox.replaceChildren(el("h3", { text: t("finalists") }), el("div", { class: "tw" }, tb));
  }
  const prop = el("input", { type: "text", placeholder: t("propose_ph"), "aria-label": t("propose_ph") });
  const propB = el("button", { class: "quiet", type: "button", text: t("propose") });
  const mindB = el("button", { class: "quiet", type: "button", text: t("ask_mind") });
  handBox.append(el("div", { class: "propose" }, prop, propB, mindB));
  propB.onclick = async () => { if (!S.session || !prop.value.trim()) return; try { const r = await api(`/v1/vapor/athanor/${S.session}`, { action: "propose", candidates: [prop.value.trim()] }); prop.value = ""; if (r.rejected && r.rejected.length) { errEl.hidden = false; errEl.textContent = r.rejected[0]; } below.replaceChildren(); poll(S.run); } catch (e) { errEl.hidden = false; errEl.textContent = e.message; } };
  prop.onkeydown = (e) => { if (e.key === "Enter") propB.click(); };
  mindB.onclick = async () => { if (!S.session) return; mindB.disabled = true; try { await api(`/v1/vapor/athanor/${S.session}`, { action: "mind", n: 8 }); below.replaceChildren(); poll(S.run); } catch (e) { errEl.hidden = false; errEl.textContent = e.message; } mindB.disabled = false; };
  async function refresh() { const snap = await api(`/v1/vapor/athanor/${S.session}?since=${S.F.since || 0}`); absorb(snap); }
  function renderPending(keys) {
    pendBox.replaceChildren(el("div", { class: "pending" }, el("h3", { text: t("measure_title") }), ...keys.map((k) => {
      const v = el("input", { type: "number", step: "any", "aria-label": k }), b = el("button", { class: "quiet", type: "button", text: t("measure") });
      b.onclick = async () => { if (v.value === "") return; await api(`/v1/vapor/athanor/${S.session}`, { action: "measure", candidate: k, value: +v.value }); b.closest(".row").remove(); if (!pendBox.querySelector(".row")) { pendBox.replaceChildren(); poll(S.run); } };
      return el("div", { class: "row" }, el("code", { text: k }), v, b);
    })));
  }
  async function settle(my) {
    const cert = await api(`/v1/vapor/athanor/${S.session}`, { action: "certificate" });
    if (S.run !== my) return;
    S.cert = cert;
    if (cert.control && cert.control.best != null) { S.F.control = cert.control.best; right.replaceChildren(furnaceView(S.F)); }
    let ver = null;
    try { ver = await api("/v1/vapor/athanor/verify", { text: ta.value, certificate: cert }); } catch (_) {}
    const checks = [];
    (ver ? ver.checks : []).forEach((c) => checks.push({ name: t("c_" + c.check) || c.check, ok: c.ok, detail: c.detail }));
    if (cert.reason === "exhausted") checks.push({ name: t("c_proof"), ok: true, detail: cert.verdict });
    if (cert.control && cert.best) checks.push({ name: t("c_control"), ok: cert.control.reached_best === 0, detail: cert.control.reached_best === 0 ? t("control_beat", fmtv(cert.control.p_chance_upper95)) : t("control_easy", cert.control.reached_best, cert.control.evaluations) });
    if (cert.holdout) checks.push({ name: t("c_holdout"), ok: cert.holdout.rank_correlation != null && cert.holdout.rank_correlation >= 0.2, detail: cert.holdout.says });
    checks.push({ name: t("c_journal"), ok: true, detail: `${cert.journal_root.slice(0, 16)}… — ${t("journal_d")}` });
    const neg = cert.reason === "counterexample" || !cert.best;
    touchBox.replaceChildren(touchstone(cert.verdict, checks, neg && cert.reason !== "exhausted"));
    const save = el("button", { class: "quiet", type: "button", text: t("save_cert") });
    save.onclick = () => download("certificate.json", new Blob([JSON.stringify(cert, null, 2)], { type: "application/json" }));
    touchBox.append(el("div", { class: "vessel-foot" }, save));
    moreB.hidden = cert.reason !== "budget" && cert.reason !== "time";
  }

  /* ---------------------------------------------------------------- games */
  async function runGame(txt, state, action = "view", extra = {}) {
    const r = await api("/v1/vapor/game", { text: txt, state, action, ...extra });
    const box = el("div", { class: "furnace" });
    box.append(el("div", { class: "board-text", text: r.board }));
    if (r.reply) box.append(el("p", { class: "furnace-cap", text: t("vapor_plays", r.reply) }));
    if (!r.over) {
      box.append(el("p", { class: "furnace-cap", text: `${t("your_move")} (${r.player})` }));
      box.append(el("div", { class: "moves" }, ...r.moves.map((m) => { const b = el("button", { type: "button", text: m }); b.onclick = async () => { const a = await api("/v1/vapor/game", { text: txt, state: r.state, action: "play", move: m }); if (a.over) return runGame(txt, a.state); runGame(txt, a.state, "reply"); }; return b; })));
    } else box.append(el("p", { class: "furnace-cap", text: r.winner ? `winner: ${r.winner}` : "draw" }));
    const acts = el("div", { class: "vessel-foot" });
    const act = (label, a, ex = {}) => { const b = el("button", { class: "quiet", type: "button", text: t(label) }); b.onclick = async () => { busy(below); try { const res = await api("/v1/vapor/game", { text: txt, state: r.state, action: a, ...ex }); gameResult(a, res); } catch (e) { below.replaceChildren(el("p", { class: "notice err", text: e.message })); } }; return b; };
    acts.append(act("game_solve", "solve"), act("game_search", "search"), act("game_learn", "learn", { games: 20 }), Object.assign(el("button", { class: "quiet", type: "button", text: t("game_new") }), { onclick: () => runGame(txt, null) }));
    right.replaceChildren(box, acts);
  }
  function gameResult(a, res) {
    if (a === "solve") { const s = res.solution; below.replaceChildren(touchstone(s.says, [{ name: t("c_proof"), ok: true, detail: `${s.positions} positions, best: ${s.best.join(", ")}` }]), table(["move", "value", "distance"], s.moves.map((m) => [m.move, m.value, m.distance]))); }
    else if (a === "search") { below.replaceChildren(table(["move", "visits", "value"], res.search.stats.slice(0, 12).map((m) => [m.move, m.visits, wbNum(m.q, 3)]))); }
    else if (a === "learn") { const m = res.versus_plain_search; below.replaceChildren(touchstone(`score ${wbNum(m.score, 3)} [${wbNum(m.interval[0], 3)}, ${wbNum(m.interval[1], 3)}] against plain search at the same simulations`, [{ name: t("c_control"), ok: m.interval[0] > 0.5, detail: `${m.a_wins} wins, ${m.draws} draws, ${m.b_wins} losses over ${m.games} games` }]), lineChart({ series: [{ points: res.loss.map((y, i) => [i + 1, y]) }], xlab: "game", ylab: "loss" })); }
  }

  /* ------------------------------------------------------------- crucible */
  async function runCrucible(txt, kind) {
    right.replaceChildren(furnaceView({ sparks: [], trace: [], status: "running" }));
    const r = await api("/v1/vapor/crucible", { kind, text: txt });
    right.replaceChildren(crucibleView(r));
    below.replaceChildren(touchstone(r.says, evidenceChecks(r.evidence), (r.evidence || []).some((e) => !e.ok)), ...crucibleDetails(r));
  }
  async function runAssay(txt, tool) {
    right.replaceChildren(furnaceView({ sparks: [], trace: [], status: "running" }));
    const r = await api("/v1/vapor/assay", { tool, text: txt, seed: +seed.value || 1 });
    right.replaceChildren(assayView(r));
    below.replaceChildren(touchstone(r.says, evidenceChecks(r.evidence)), ...assayDetails(r));
  }

  /* ---------------------------------------------------------------- scene */
  async function runScene(txt) {
    const r = await api("/v1/vapor/scene/ops", { text: txt });
    const cv = el("canvas", { class: "scene-cv" });
    right.replaceChildren(cv);
    const scene = { w: 960, h: 600, horizon: 0.62, layers: [], walk: { cols: 0, rows: 0, cells: [] }, bg: ["#0b1424", "#2a3550"], seed: +seed.value || 1, ops: r.ops };
    S.scene = SceneEngine.create(cv, scene); S.scene.play();
    const words = el("input", { type: "text", placeholder: t("scene_words_ph") }), go = el("button", { class: "quiet", type: "button", text: t("scene_direct") });
    go.onclick = async () => { if (!words.value.trim()) return; const d = await api("/v1/vapor/scene/mind", { words: words.value, scene }); d.ops.forEach((op) => S.scene.apply(op)); ta.value += (d.text ? "\n" + d.text : "\n" + `direct "${words.value}"`) + "\n"; words.value = ""; if (d.problems && d.problems.length) { errEl.hidden = false; errEl.textContent = d.problems.join(" · "); } };
    const exp = el("button", { class: "quiet", type: "button", text: "HTML" });
    exp.onclick = async () => { const h = await fetch("/v1/vapor/scene/export", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ scene: S.scene.snapshot(), title: "vapor — scene" }) }); download("scene.html", new Blob([await h.text()], { type: "text/html" })); };
    below.replaceChildren(el("div", { class: "propose" }, words, go, exp), el("p", { class: "muted", text: t("scene_hint") }));
    if (r.problems.length) { errEl.hidden = false; errEl.textContent = r.problems.join(" · "); }
  }

  /* -------------------------------------------------------------- alembic */
  async function runAlembic(txt) {
    const r = await api("/v1/vapor/alembic", { text: txt });
    const ex = el("input", { type: "text", placeholder: t("eval_ph") }), go = el("button", { class: "quiet", type: "button", text: t("evaluate") }), out = el("pre", { class: "code-out" });
    go.onclick = async () => { try { const v = await api("/v1/vapor/alembic", { text: txt, expr: ex.value }); out.textContent = v.value; } catch (e) { out.textContent = e.message; } };
    ex.onkeydown = (e) => { if (e.key === "Enter") go.click(); };
    right.replaceChildren(el("div", { class: "furnace" }, el("div", { class: "board-text", style: "font-size:14px;letter-spacing:0", text: r.constants.map((c) => `${c.name} = ${c.value}`).join("\n") || "—" }), el("p", { class: "furnace-cap", text: r.functions.map((f) => `${f.name}(${f.params.join(", ")})`).join("  ") })));
    below.replaceChildren(el("div", { class: "propose" }, ex, go), out);
  }

  return S;
}

/* ================================================================ domain views */
function darkBox(...kids) { return el("div", { class: "furnace" }, ...kids); }
function chartIn(spec) { const c = lineChart(spec); return c; }
function crucibleView(r) {
  const pts = (xs, ys) => xs.map((x, i) => [x, ys[i]]).filter(([x, y]) => isFinite(x) && isFinite(y));
  switch (r.kind) {
    case "quantum": {
      const s = [{ points: pts(r.plot.x, r.plot.v), cls: 3 }];
      r.plot.psi.forEach((p, k) => { const E = r.states[k].energy, sc = 0.4 * Math.max(...r.states.map((q) => Math.abs(q.energy)), 1) / Math.max(...p.map(Math.abs), 1e-9); s.push({ points: r.plot.x.map((x, i) => [x, E + sc * p[i]]), cls: k % 3 }); });
      return darkBox(chartIn({ series: s, xlab: "x", ylab: "E, ψ" }));
    }
    case "hamiltonian": { const e0 = r.trajectory[0].h; return darkBox(chartIn({ series: [{ points: r.trajectory.map((p) => [p.t, Math.abs(p.h - e0) + 1e-18]) }, ...(r.control ? [{ points: r.control.map((p) => [p.t, Math.abs(p.h - e0) + 1e-18]), cls: 1 }] : [])], logy: true, xlab: "t", ylab: "|H − H₀|" }), legend([[0, r.method], [1, "RK4 (control)"]]), chartIn({ series: [{ points: r.trajectory.map((p) => [p.z[0], p.z[r.coordinates.length]]), dots: true }], xlab: r.coordinates[0], ylab: r.momenta[0] })); }
    case "laws": return darkBox(el("ul", { class: "laws" }, ...(r.laws.length ? r.laws.map((l) => el("li", {}, l.law, el("small", { text: l.status }))) : [el("li", { text: "—" })])), ...(r.laws.filter((l) => l.series && l.series.length).slice(0, 2).map((l) => { const y0 = l.series[0], sc = Math.abs(y0) || 1;
      /* the law's relative drift along the trajectory: the visible thing is how small it stays */
      return chartIn({ series: [{ points: l.series.map((y, i) => [i, Math.abs(y - y0) / sc + 1e-18]) }], logy: true, ylab: "|Δ| / |" + (l.law.length > 28 ? l.law.slice(0, 27) + "…" : l.law) + "|", h: 140 }); })));
    case "reactions": return darkBox(chartIn({ series: r.species.map((sp, k) => ({ points: pts(r.t, r.series[sp]), cls: k % 4 })), xlab: "t", ylab: "concentration" }), legend(r.species.map((sp, k) => [k % 4, sp])));
    case "fields": return darkBox(chartIn({ series: [{ points: r.trajectory.map((p) => [p[1], p[2]]) }, { points: r.control.map((p) => [p[1], p[2]]), cls: 1 }], xlab: "x", ylab: "y" }), legend([[0, "Boris"], [1, "Euler (control)"]]));
    case "phylogeny": return darkBox(el("div", { class: "board-text", style: "font-size:12.5px;letter-spacing:0;white-space:pre-wrap", text: r.newick }));
    case "molecule": return darkBox(stats(stat("E (hartree)", wbNum(r.energy, 7)), stat("−V/T", wbNum(r.virial, 4)), stat("iterations", String(r.iterations))), r.scan ? chartIn({ series: [{ points: r.scan.points.map((p) => [p.r, p.energy]), dots: true }], xlab: "R (bohr)", ylab: "E" }) : "");
    case "fold": { const c = r.coords, xs = c.map((p) => p[0]), ys = c.map((p) => p[1]), mx = Math.min(...xs), my = Math.min(...ys), sz = 22;
      const svg = svgEl("svg", { viewBox: `-10 -10 ${(Math.max(...xs) - mx) * sz + 20} ${(Math.max(...ys) - my) * sz + 20}`, style: "max-height:300px" });
      svg.append(svgEl("path", { d: "M" + c.map((p) => `${(p[0] - mx) * sz} ${(p[1] - my) * sz}`).join(" L"), fill: "none", stroke: "rgba(236,228,211,.5)", "stroke-width": 2 }));
      c.forEach((p, i) => svg.append(svgEl("circle", { cx: (p[0] - mx) * sz, cy: (p[1] - my) * sz, r: 6, fill: r.sequence[i] === "H" ? "#D9AE3F" : "#6FB8A6" })));
      return darkBox(svg, el("p", { class: "furnace-cap", text: `E = ${r.energy} · bound ${r.bound}` })); }
    case "regress": return darkBox(el("div", { class: "board-text", style: "font-size:16px;letter-spacing:0;white-space:pre-wrap", text: `${r.target} ≈ ${r.best.expression}` }), chartIn({ series: [{ points: r.front.map((p) => [p.size, Math.max(p.test_mse, 1e-18)]), dots: true }], logy: true, xlab: "size", ylab: "held-out MSE", h: 180 }));
    case "evolution": return darkBox(stats(stat("exact", wbNum(r.exact, 4)), stat("simulated", wbNum(r.simulated, 4)), stat("Kimura", wbNum(r.kimura, 4)), stat("neutral", wbNum(r.neutral, 4))));
    default: return darkBox(el("p", { text: r.says }));
  }
}
function crucibleDetails(r) {
  if (r.kind === "quantum") return [table(["n", "E", "± error", "order", "virial"], r.states.map((s) => [s.index, wbNum(s.energy, 8), wbNum(s.error_estimate, 2), wbNum(s.observed_order, 3), wbNum(s.virial_residual, 2)]))];
  if (r.kind === "phylogeny") return [table(["split", "support"], r.splits.map((s) => [s.taxa.join(" "), `${Math.round(s.support * 100)} %`]))];
  if (r.kind === "regress") return [table(["size", "expression", "held-out R²"], r.front.map((p) => [p.size, p.expression, wbNum(p.test_r2, 4)]))];
  if (r.kind === "hamiltonian") return [el("pre", { class: "code-out", text: r.equations.join("\n") })];
  if (r.kind === "laws") return [el("pre", { class: "code-out", text: (r.vector_field || []).join("\n") })];
  return [];
}
/* a forest plot: each row an interval with its estimate — the honest picture of a ranking or a difference */
function forest(rows, { zero = null, xlab = "" } = {}) {
  const w = 560, rowH = 30, pad = { l: 120, r: 70, t: 10, b: 40 }, h = pad.t + pad.b + rowH * rows.length;
  const xs = rows.flatMap((r) => [r.lo, r.hi, r.mean]).concat(zero != null ? [zero] : []).filter(isFinite);
  let x0 = Math.min(...xs), x1 = Math.max(...xs); const m = (x1 - x0) * 0.08 || 1; x0 -= m; x1 += m;
  const X = (v) => pad.l + ((v - x0) / (x1 - x0)) * (w - pad.l - pad.r);
  const svg = svgEl("svg", { class: "chart", viewBox: `0 0 ${w} ${h}`, role: "img" });
  for (let k = 0; k <= 4; k++) { const v = x0 + ((x1 - x0) * k) / 4; svg.append(svgEl("line", { class: "grid", x1: X(v), x2: X(v), y1: pad.t, y2: h - pad.b })); const tx = svgEl("text", { class: "axis", x: X(v), y: h - 24, "text-anchor": "middle" }); tx.textContent = +v.toPrecision(3); svg.append(tx); }
  if (zero != null) svg.append(svgEl("line", { x1: X(zero), x2: X(zero), y1: pad.t, y2: h - pad.b, stroke: "#ECE4D3", "stroke-dasharray": "3 4", opacity: ".55" }));
  if (xlab) { const tx = svgEl("text", { class: "axis", x: (pad.l + w - pad.r) / 2, y: h - 4, "text-anchor": "middle" }); tx.textContent = xlab; svg.append(tx); }
  rows.forEach((r, i) => {
    const y = pad.t + rowH * i + rowH / 2, col = r.strong ? "#D9AE3F" : "#6FB8A6";
    const lab = svgEl("text", { class: "axis", x: pad.l - 10, y: y + 4, "text-anchor": "end" }); lab.textContent = r.label; svg.append(lab);
    svg.append(svgEl("line", { x1: X(r.lo), x2: X(r.hi), y1: y, y2: y, stroke: col, "stroke-width": 3, "stroke-linecap": "round", opacity: ".8" }));
    svg.append(svgEl("circle", { cx: X(r.mean), cy: y, r: 5.5, fill: col }));
    if (r.note) { const nt = svgEl("text", { class: "axis", x: w - pad.r + 8, y: y + 4 }); nt.textContent = r.note; svg.append(nt); }
  });
  return svg;
}

function assayView(r) {
  switch (r.tool) {
    case "leaderboard": { const tied = new Set([r.systems[0].system, ...(r.tied_with_leader || [])]);
      return darkBox(forest(r.systems.map((s) => ({ label: s.system, mean: s.mean, lo: s.ci95[0], hi: s.ci95[1], strong: tied.has(s.system), note: `P(#1) ${Math.round(s.p_first * 100)} %` })), { xlab: "mean, 95 % CI" })); }
    case "calibration": return darkBox(forest([{ label: "ECE", mean: r.ece, lo: r.ece_ci95[0], hi: r.ece_ci95[1], strong: r.p_miscalibrated < 0.05, note: `floor ${wbNum(r.ece_if_calibrated, 2)}` }], { zero: r.ece_if_calibrated, xlab: "ECE (dashed: what a calibrated model would show)" }), chartIn({ series: [{ points: r.reliability.map((b) => [b.confidence, b.accuracy]), dots: true }, { points: [[0, 0], [1, 1]], cls: 3 }], xlab: "confidence", ylab: "accuracy", ymin: 0, ymax: 1 }));
    case "scaling": return darkBox(stats(stat("α", wbNum(r.params.alpha, 4)), stat("β", r.params.beta == null ? "—" : wbNum(r.params.beta, 4)), stat("E", wbNum(r.params.e, 4)), stat("a = β/(α+β)", r.params.a_opt == null ? "—" : wbNum(r.params.a_opt, 4))), table(["held out", "loss", "predicted", "error"], r.holdout.predictions.map((p) => [wbNum(p.x, 3), wbNum(p.loss, 4), wbNum(p.predicted, 4), `${wbNum(p.error * 100, 3)} %`])));
    case "compare": return darkBox(forest([{ label: `${r.b} − ${r.a}`, mean: r.difference, lo: r.ci95[0], hi: r.ci95[1], strong: r.significant, note: `p ${wbNum(r.p_permutation, 2)}` }, { label: "detectable ±", mean: 0, lo: -r.mde80, hi: r.mde80, note: "80 % power" }], { zero: 0, xlab: "difference" }), stats(stat("Δ", wbNum(r.difference, 4), `[${wbNum(r.ci95[0], 3)}, ${wbNum(r.ci95[1], 3)}]`), stat("p", wbNum(r.p_permutation, 3)), stat("MDE", wbNum(r.mde80, 3)), stat("n", String(r.n))));
    case "agreement": return darkBox(forest([{ label: "Krippendorff α", mean: r.krippendorff_alpha, lo: r.alpha_ci95[0], hi: r.alpha_ci95[1], strong: r.alpha_ci95[0] >= 0.667 }], { zero: 0.667, xlab: "α (0.667: tentative, 0.8: reliable)" }), stats(stat("α", wbNum(r.krippendorff_alpha, 3), `[${wbNum(r.alpha_ci95[0], 3)}, ${wbNum(r.alpha_ci95[1], 3)}]`), stat("Fleiss κ", wbNum(r.fleiss_kappa, 3)), stat("items", String(r.items))));
    case "judge": return darkBox(stats(stat("consistent", `${Math.round(r.consistent * 100)} %`), stat("first slot", `${Math.round(r.first_slot_rate * 100)} %`), stat("p", wbNum(r.p_position_bias, 3))));
    case "contamination": return darkBox(stats(stat("contaminated", String(r.contaminated)), stat("clean", String(r.clean))));
    case "dedup": return darkBox(stats(stat("documents", String(r.documents)), stat("removed", String(r.removed)), stat("clusters", String(r.clusters.length))));
    default: return darkBox(el("p", { text: r.says }));
  }
}
function assayDetails(r) {
  if (r.tool === "leaderboard") return [table(["system", "mean", "95 % CI", "P(#1)", "rank interval"], r.systems.map((s) => [s.system, wbNum(s.mean, 4), `[${wbNum(s.ci95[0], 3)}, ${wbNum(s.ci95[1], 3)}]`, `${Math.round(s.p_first * 100)} %`, `${s.rank_interval[0]}–${s.rank_interval[1]}`]))];
  if (r.tool === "dedup") return [table(["cluster"], r.clusters.map((c) => [c.join(", ")]))];
  if (r.tool === "contamination") return [table(["item", "overlap", "contaminated"], r.items.map((i) => [i.index, `${Math.round(i.overlap * 100)} %`, i.contaminated ? "yes" : "no"]))];
  return [];
}

/* ================================================================ wiring */
const BENCHES = {};
function benchOnce(mode) { if (!BENCHES[mode]) BENCHES[mode] = buildBench($("wb-" + mode), mode); }
benchOnce("work");
shown.set("p-crucible", () => benchOnce("crucible"));
shown.set("p-assay", () => benchOnce("assay"));
rerender.push(() => { Object.keys(BENCHES).forEach((m) => { const ed = $("ed-" + m), v = ed && ed.value; delete BENCHES[m]; benchOnce(m); if (v) { $("ed-" + m).value = v; } }); });
PALETTE_ITEMS.push(...[["work", "work_s"], ["crucible", "crucible_s"], ["assay", "assay_s"]].map(([k, s]) => () => ({ label: t(k), sub: t(s), run: () => openPanel(k) })));
applyLang();
if (!location.hash || location.hash === "#work") { const tb = $("t-work"); if (tb) { select(tb); } }
