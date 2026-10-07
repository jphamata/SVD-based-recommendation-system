"use strict";
/* opus.js — the 0.15 desks (docs/CONSOLE.md §0.15): Rebis (two circuits,
   one function?), Aludel (polynomial claims and barrier certificates),
   Tabula (contracts without antinomies), Cupel (silent corruption,
   caught) and Amalgam (sums without order). One bold element per desk
   carries the answer: the seal between the two vessels, the subdivided
   box, the tablet's clashing lines, the 32 bits of a float, the one exact
   mark among the scattered sums. Runs after the page's scripts and uses
   their helpers ($, el, t, api, svgEl, editor, busy, table, stat, stats,
   dl, chipV, touchstone, streak, shown, rerender, PALETTE_ITEMS, wbNum). */

(() => {  // everything here is local: the page's scripts share one global scope
/* ================================================================== words */
Object.assign(I18N.en, {
  op_f64: "f64", op_f32: "f32",
  g_opus: "Opus",
  rebis: "Rebis", rebis_s: "two circuits, one function?", aludel: "Aludel", aludel_s: "polynomial claims, barriers",
  tabula: "Tabula", tabula_s: "contracts without antinomies", cupel: "Cupel", cupel_s: "silent corruption, caught",
  amalgam: "Amalgam", amalgam_s: "sums without order",
  rebis_lede: "Two netlists (or AIGER files) — are they the same function? Up to 16 inputs every pattern is compared at once; beyond, random simulation looks first and a miter goes to the SAT solver, whose proof is checked by code that shares nothing with it. A difference comes back as the smallest input that shows it. Word-level identities (a multiplier multiplies) are proved by algebra over ℤ.",
  aludel_lede: "A claim about a polynomial on a box — p ≥ 0, or p > 0 — decided in exact integers: certified with a subdivision anyone can replay, refuted at an exact point, or exhausted with the cell where the budget ran out. Never a guess. For a polynomial system ẋ = f(x), a barrier certificate proves that no trajectory from the initial set reaches the unsafe one.",
  tabula_lede: "Write the clauses of a contract or a rule as norms over facts. Every pair that could clash — a duty and a prohibition, a prohibition and a privilege, a duty and an exemption, two duties that cannot both be done — is settled: the scenario that triggers it, or a proof that none does. Toggle facts to see the positions in force.",
  cupel_lede: "A product y = x·Wᵀ checked through the adjoint identity y·r = x·(Wᵀr), in exact arithmetic, against a tolerance proved for any correct substrate and any summation order. One bit of a correct result is flipped at every position: what can a check that costs a fraction of the product see?",
  amalgam_lede: "Floating-point addition depends on the order. Sum the same numbers in different orders and the results scatter; the amalgam adds them exactly — as integers — and rounds once. Its result depends on the numbers alone: not on the order, the number of workers, or a crash in the middle.",
  m_equiv: "Compare", m_anf: "ANF", m_identity: "Identity", m_stab: "Stabilizer", m_decide: "Decide", m_barrier: "Barrier",
  vessel_a: "A", vessel_b: "B", spec: "specification", spec_ph: "m[16] = a[8] * b[8]", qubits: "qubits", seed: "seed",
  same_fn: "the same function", diff_fn: "not the same function", unknown_fn: "not settled", inputs_that_show: "the inputs that show it",
  outputs_differ: "outputs that differ", by_method: (m) => `by ${m}`, patterns: "patterns", op_lemmas: "proof lemmas checked", conflicts: "conflicts",
  proved: "proved", refuted: "refuted", substitutions: "substitutions", peak: "peak terms",
  degree: "degree", terms: "terms", outcomes: "outcomes", random_k: "random", determ_k: "deterministic", stabilizers: "stabilizers",
  vars: "variables", poly: "polynomial", box: "box", sense_nonneg: "≥ 0", sense_pos: "> 0", certified: "certified", exhausted: "exhausted",
  replayed: "witness replayed", cells: "cells", depth: "depth", at_point: "at", value_is: "the value is", cell_left: "the budget ran out on",
  field: "field", domain: "domain", init: "initial set", unsafe: "unsafe set", barrier_b: "barrier B", find_one: "find one", degree_b: "degree",
  b_zero: "B = 0", conditions: "conditions", lp_note: (r, n, tot) => `found by an exact LP in ${r} rounds (${n} of ${tot} rows)`, not_found: "no barrier found",
  contract: "contract", facts: "facts", positions: "positions in force", claims: "claims (Hohfeld)", overridden: "overridden",
  antinomy: "antinomy", conflict: "conflict", resolved: "resolved", consistent: "consistent", antinomies: "antinomies",
  prevails: (c) => `${c} prevails`, when: "when", no_fact: "no fact holds", show_scenario: "Set these facts", silence: "nothing governs",
  never_clash: (n) => `${n} pair${n === 1 ? "" : "s"} proved never to clash`, holder: (h, a, x) => `${h} may claim from ${a}: ${x}`,
  rows_n: "n (rows of W)", cols_k: "k (columns)", trials: "trials", bit_ex: "example bit", sign: "sign", exponent: "exponent", mantissa: "mantissa",
  op_caught: "caught", missed: "below the rounding envelope", int8_exact: (n) => `int8: ${n} of 32 bits caught — the check is exact`,
  flip_ex: (b, a, c, v) => `bit ${b}: ${a} → ${c} · ${v}`, cost: "cost",
  numbers: "numbers", format: "format", orders: "left to right, in different orders", amalgam_mark: "the amalgam", exact_sum: "exact sum",
  distinct: (k, n) => `${k} different results from ${n} orders`, run_it: "Run",
  cond_initial: "B ≤ 0 on the initial set", cond_unsafe: "B > 0 on the unsafe set", cond_flow: "λB − ∇B·f ≥ 0 on the domain"
});
Object.assign(I18N.pt, {
  op_f64: "f64", op_f32: "f32",
  g_opus: "Opus",
  rebis: "Rebis", rebis_s: "dois circuitos, uma função?", aludel: "Aludel", aludel_s: "afirmações polinomiais, barreiras",
  tabula: "Tábua", tabula_s: "contratos sem antinomias", cupel: "Copela", cupel_s: "corrupção silenciosa, flagrada",
  amalgam: "Amálgama", amalgam_s: "somas sem ordem",
  rebis_lede: "Duas netlists (ou arquivos AIGER) — são a mesma função? Até 16 entradas todos os padrões são comparados de uma vez; além disso, a simulação aleatória olha primeiro e um miter vai ao resolvedor SAT, cuja prova é conferida por código que não compartilha nada com ele. Uma diferença volta como a menor entrada que a mostra. Identidades de palavra (um multiplicador multiplica) são provadas por álgebra sobre ℤ.",
  aludel_lede: "Uma afirmação sobre um polinômio numa caixa — p ≥ 0, ou p > 0 — decidida em inteiros exatos: certificada com uma subdivisão que qualquer um reconfere, refutada num ponto exato, ou esgotada com a célula onde o orçamento acabou. Nunca um palpite. Para um sistema polinomial ẋ = f(x), um certificado de barreira prova que nenhuma trajetória do conjunto inicial chega ao inseguro.",
  tabula_lede: "Escreva as cláusulas de um contrato ou de uma norma como deveres sobre fatos. Todo par que pode colidir — um dever e uma proibição, uma proibição e uma permissão, um dever e uma isenção, dois deveres que não se cumprem juntos — é decidido: o cenário que o dispara, ou a prova de que nenhum dispara. Alterne os fatos para ver as posições em vigor.",
  cupel_lede: "Um produto y = x·Wᵀ conferido pela identidade adjunta y·r = x·(Wᵀr), em aritmética exata, contra uma tolerância provada para qualquer substrato correto e qualquer ordem de soma. Um bit de um resultado correto é invertido em cada posição: o que vê uma conferência que custa uma fração do produto?",
  amalgam_lede: "A soma em ponto flutuante depende da ordem. Some os mesmos números em ordens diferentes e os resultados se espalham; a amálgama os soma exatamente — como inteiros — e arredonda uma vez. O resultado depende só dos números: não da ordem, do número de workers, nem de uma queda no meio.",
  m_equiv: "Comparar", m_anf: "FNA", m_identity: "Identidade", m_stab: "Estabilizador", m_decide: "Decidir", m_barrier: "Barreira",
  vessel_a: "A", vessel_b: "B", spec: "especificação", spec_ph: "m[16] = a[8] * b[8]", qubits: "qubits", seed: "semente",
  same_fn: "a mesma função", diff_fn: "não é a mesma função", unknown_fn: "não decidido", inputs_that_show: "as entradas que mostram",
  outputs_differ: "saídas que diferem", by_method: (m) => `por ${m}`, patterns: "padrões", op_lemmas: "lemas da prova conferidos", conflicts: "conflitos",
  proved: "provado", refuted: "refutado", substitutions: "substituições", peak: "pico de termos",
  degree: "grau", terms: "termos", outcomes: "resultados", random_k: "aleatórios", determ_k: "determinados", stabilizers: "estabilizadores",
  vars: "variáveis", poly: "polinômio", box: "caixa", sense_nonneg: "≥ 0", sense_pos: "> 0", certified: "certificado", exhausted: "esgotado",
  replayed: "testemunha reconferida", cells: "células", depth: "profundidade", at_point: "em", value_is: "o valor é", cell_left: "o orçamento acabou em",
  field: "campo", domain: "domínio", init: "conjunto inicial", unsafe: "conjunto inseguro", barrier_b: "barreira B", find_one: "achar uma", degree_b: "grau",
  b_zero: "B = 0", conditions: "condições", lp_note: (r, n, tot) => `achada por um LP exato em ${r} rodadas (${n} de ${tot} linhas)`, not_found: "nenhuma barreira achada",
  contract: "contrato", facts: "fatos", positions: "posições em vigor", claims: "pretensões (Hohfeld)", overridden: "afastadas",
  antinomy: "antinomia", conflict: "conflito", resolved: "resolvido", consistent: "consistente", antinomies: "antinomias",
  prevails: (c) => `${c} prevalece`, when: "quando", no_fact: "nenhum fato vale", show_scenario: "Marcar esses fatos", silence: "nada rege",
  never_clash: (n) => `${n} par${n === 1 ? "" : "es"} provado${n === 1 ? "" : "s"} sem colisão`, holder: (h, a, x) => `${h} pode exigir de ${a}: ${x}`,
  rows_n: "n (linhas de W)", cols_k: "k (colunas)", trials: "tentativas", bit_ex: "bit de exemplo", sign: "sinal", exponent: "expoente", mantissa: "mantissa",
  op_caught: "flagrado", missed: "abaixo do envelope de arredondamento", int8_exact: (n) => `int8: ${n} de 32 bits flagrados — a conferência é exata`,
  flip_ex: (b, a, c, v) => `bit ${b}: ${a} → ${c} · ${v}`, cost: "custo",
  numbers: "números", format: "formato", orders: "da esquerda para a direita, em ordens diferentes", amalgam_mark: "a amálgama", exact_sum: "soma exata",
  distinct: (k, n) => `${k} resultados diferentes em ${n} ordens`, run_it: "Rodar",
  cond_initial: "B ≤ 0 no conjunto inicial", cond_unsafe: "B > 0 no conjunto inseguro", cond_flow: "λB − ∇B·f ≥ 0 no domínio"
});

/* ================================================================ shared */
const OPUS = { info: null };
async function opusInfo() { if (!OPUS.info) OPUS.info = await api("/v1/vapor/opus"); return OPUS.info; }

function opusHead(root, key) {
  root.replaceChildren(el("h2", { text: t(key) }), el("p", { class: "lede", text: t(key + "_lede") }));
}
function opusShelf(root, items, onPick) {
  const shelf = el("div", { class: "shelf" });
  const pt = lang === "pt";
  shelf.append(el("span", { class: "vials" }, ...items.map((e) => { const b = el("button", { class: "vial", type: "button", "data-field": "cs", title: (pt && e.about_pt) || e.about, text: (pt && e.title_pt) || e.title }); b.onclick = () => onPick(e); return b; })));
  root.append(shelf);
  return shelf;
}
function segmented(options, value, onChange) {
  const s = el("div", { class: "seg small", role: "radiogroup" });
  const draw = (v) => s.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.v === v)));
  options.forEach(([v, label]) => { const b = el("button", { type: "button", role: "radio", "data-v": v, text: t(label) }); b.onclick = () => { draw(v); onChange(v); }; s.append(b); });
  draw(value);
  return s;
}
const runButton = (onRun) => { const b = el("button", { class: "primary", type: "button", text: t("run_it") }); b.onclick = onRun; return b; };
const errLine = (e) => el("p", { class: "notice err", text: e.message || String(e) });
const numIn = (v, label, attrs = {}) => el("input", { type: "number", value: String(v), "aria-label": label, class: "op-num", ...attrs });
// a desk's state outlives its DOM: a language switch rebuilds the desk in the mode it was in;
// inputs carry a stable data-k so what was typed is put back in the same field
const OPUS_STATE = {};
const textIn = (v, label, attrs = {}) => el("input", { type: "text", value: v, "aria-label": label, class: "op-text", spellcheck: "false", ...attrs });
const labeled = (label, input) => el("label", { class: "op-field" }, el("span", { text: label }), input);

/* the seal between two vessels: ≡ in gold when they are one, ≢ in cinnabar when not, ? when unsettled */
function seal(state) {
  const sym = state === "equivalent" || state === "proved" ? "≡" : state === "different" || state === "refuted" ? "≢" : "?";
  const cls = state === "equivalent" || state === "proved" ? "ok" : state === "different" || state === "refuted" ? "no" : "un";
  const svg = svgEl("svg", { viewBox: "0 0 64 64", class: "seal " + cls, role: "img", "aria-label": state });
  for (let k = 0; k < 14; k++) { const a = (k / 14) * 2 * Math.PI, r = 27 + (k % 2) * 3; svg.append(svgEl("circle", { cx: (32 + r * Math.cos(a)).toFixed(1), cy: (32 + r * Math.sin(a)).toFixed(1), r: "4.2", class: "lobe" })); }
  svg.append(svgEl("circle", { cx: "32", cy: "32", r: "26", class: "disc" }));
  const tx = svgEl("text", { x: "32", y: "42", "text-anchor": "middle", class: "glyph" }); tx.textContent = sym; svg.append(tx);
  return svg;
}

/* a row of bits as squares, grouped into words (prefix + index), most significant first */
function bitWords(assign) {
  const words = {}, singles = [];
  Object.entries(assign).forEach(([k, v]) => { const m = /^([A-Za-z_]+)(\d+)$/.exec(k); if (m) (words[m[1]] = words[m[1]] || []).push([+m[2], v]); else singles.push([k, v]); });
  const box = el("div", { class: "bitwords" });
  Object.keys(words).sort().forEach((w) => {
    const bits = words[w].sort((a, b) => b[0] - a[0]);
    const val = bits.reduce((acc, [i, v]) => acc + (v ? 2n ** BigInt(i) : 0n), 0n);
    const strip = el("span", { class: "bits", role: "img", "aria-label": `${w} = 0x${val.toString(16).toUpperCase()}` }, ...bits.map(([i, v]) => el("i", { class: v ? "on" : "", title: `${w}${i} = ${v}` })));
    box.append(el("div", { class: "bitword" }, el("b", { text: w }), strip, el("code", { text: `0x${val.toString(16).toUpperCase()}` })));
  });
  singles.sort().forEach(([k, v]) => box.append(el("div", { class: "bitword" }, el("b", { text: k }), el("span", { class: "bits" }, el("i", { class: v ? "on" : "" })), el("code", { text: String(v) }))));
  return box;
}

/* ================================================================ Rebis */
function buildRebis(root) {
  opusHead(root, "rebis");
  const S = (OPUS_STATE.rebis ||= { mode: "equivalent" });
  const a = editor("op-rebis-a", 16), b = editor("op-rebis-b", 16);
  const spec = textIn("", t("spec"), { placeholder: t("spec_ph"), "data-k": "spec" });
  const qubits = numIn(64, t("qubits"), { min: "1", max: "2000", "data-k": "qubits" }), seedI = numIn(1, t("seed"), { min: "1", "data-k": "seed" });
  const out = el("div", { class: "results" });
  const sealBox = el("div", { class: "seal-col", "aria-live": "polite" }, seal("unknown"));
  const vesselB = el("div", { class: "op-vessel" }, el("div", { class: "vessel-head" }, el("b", { class: "vtag", text: t("vessel_b") })), b);
  const extra = el("div", { class: "op-row" });
  const modes = segmented([["equivalent", "m_equiv"], ["anf", "m_anf"], ["identity", "m_identity"], ["stabilizer", "m_stab"]], S.mode, (m) => setMode(m));
  const setMode = (m) => {
    S.mode = m;
    vesselB.hidden = m !== "equivalent";
    sealBox.hidden = m !== "equivalent" && m !== "identity";
    extra.replaceChildren(...(m === "identity" ? [labeled(t("spec"), spec)] : m === "stabilizer" ? [labeled(t("qubits"), qubits), labeled(t("seed"), seedI)] : []));
    modes.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x.dataset.v === m)));
  };
  opusInfo().then((info) => opusShelf(root, info.rebis, (e) => { a.value = e.a || ""; b.value = e.b || ""; spec.value = e.spec || ""; if (e.n) qubits.value = e.n; setMode(e.op); run(); }))
    .then(() => root.append(modes, el("div", { class: "op-pair" }, el("div", { class: "op-vessel" }, el("div", { class: "vessel-head" }, el("b", { class: "vtag", text: t("vessel_a") })), a), sealBox, vesselB), extra, el("div", { class: "op-row" }, runButton(() => run())), out))
    .then(() => { if (!a.value) { const e = OPUS.info.rebis[0]; a.value = e.a; b.value = e.b; } setMode(S.mode); });
  const run = async () => {
    busy(out);
    sealBox.replaceChildren(seal("unknown"));
    try {
      const req = S.mode === "equivalent" ? { op: "equivalent", a: a.value, b: b.value } : S.mode === "identity" ? { op: "identity", a: a.value, spec: spec.value }
        : S.mode === "stabilizer" ? { op: "stabilizer", a: a.value, n: +qubits.value, seed: +seedI.value } : { op: "anf", a: a.value };
      const r = await api("/v1/vapor/rebis", req);
      out.replaceChildren(...rebisView(S.mode, r));
      if (r.verdict) sealBox.replaceChildren(seal(r.verdict));
    } catch (e) { out.replaceChildren(errLine(e)); }
  };
  [a, b].forEach((x) => x.addEventListener("run", run));
}
function rebisView(mode, r) {
  if (mode === "anf") return r.outputs.map((o) => el("div", { class: "anf" }, el("h3", { text: o.output }), el("p", { class: "muted", text: `${t("degree")} ${o.degree} · ${o.terms} ${t("terms")}` }), el("code", { class: "anf-text", text: o.text })));
  if (mode === "stabilizer") {
    const strip = el("div", { class: "qstrip", role: "img", "aria-label": r.outcomes.join("") }, ...r.outcomes.map((v, i) => el("i", { class: (v ? "on" : "") + (r.kinds[i] === "random" ? " rnd" : ""), title: `q${i}: ${v} (${r.kinds[i] === "random" ? t("random_k") : t("determ_k")})` })));
    const k = r.kinds.filter((x) => x === "random").length;
    return [el("h3", { text: `${r.measured} ${t("outcomes")}` }), strip, stats(stat(t("random_k"), String(k)), stat(t("determ_k"), String(r.kinds.length - k))),
      ...(r.stabilizers ? [el("p", { class: "muted", text: t("stabilizers") }), el("code", { class: "anf-text", text: r.stabilizers.join("  ") })] : [])];
  }
  if (mode === "identity") {
    if (r.verdict === "proved") return [touchstone(`${t("proved")}: ${r.spec}`, [{ name: t("substitutions"), ok: true, detail: String(r.stats.substitutions) }, { name: t("peak"), ok: true, detail: String(r.stats.peak_terms) }])];
    if (r.verdict === "refuted") return [touchstone(`${t("refuted")}: ${r.spec}`, [{ name: r.spec, ok: false, detail: `${r.value}` }], true), el("p", { class: "muted", text: t("inputs_that_show") }), bitWords(r.counterexample)];
    return [el("p", { class: "notice", text: r.why || t("unknown_fn") })];
  }
  if (r.verdict === "equivalent") {
    const ev = r.evidence, checks = [{ name: t("by_method", ev.method), ok: true, detail: ev.method === "sat" ? `${ev.checked_lemmas} ${t("op_lemmas")}` : `${ev.patterns} ${t("patterns")}` }];
    if (ev.cnf_sha256) checks.push({ name: "CNF", ok: true, detail: ev.cnf_sha256.slice(0, 16) + "…" });
    return [touchstone(t("same_fn"), checks), stats(stat("A", `${r.a.gates}`, `${r.a.inputs} → ${r.a.outputs}`), stat("B", `${r.b.gates}`, `${r.b.inputs} → ${r.b.outputs}`), stat("ms", String(r.ms)))];
  }
  if (r.verdict === "different") {
    const diffs = Object.keys(r.a_out).filter((k) => r.a_out[k] !== r.b_out[k]);
    return [touchstone(t("diff_fn"), [{ name: t("by_method", r.method), ok: false, detail: `${r.ms} ms` }], true), el("h3", { text: t("inputs_that_show") }), bitWords(r.counterexample),
      el("p", { class: "muted", text: `${t("outputs_differ")}: ${diffs.map((k) => `${k} ${r.a_out[k]}≠${r.b_out[k]}`).join(", ")}` })];
  }
  return [el("p", { class: "notice", text: r.why || t("unknown_fn") })];
}

/* ================================================================ Aludel */
function buildAludel(root) {
  opusHead(root, "aludel");
  const S = (OPUS_STATE.aludel ||= { mode: "decide", sense: "nonneg",
    boxes: { box: [["-2", "2"], ["-2", "2"]], domain: [["-2", "2"], ["-2", "2"]], init: [["-1/2", "1/2"], ["-1/2", "1/2"]], unsafe: [["3/2", "2"], ["3/2", "2"]] } });
  const vars = textIn("x, y", t("vars"), { "data-k": "vars" }), poly = editor("op-aludel-p", 3), boxes = S.boxes;
  const boxInputs = {};
  const boxField = (name) => { const ins = boxes[name].map((p, i) => [textIn(p[0], `${name} lo ${i}`, { class: "op-text op-small", "data-k": `${name}-lo-${i}` }), textIn(p[1], `${name} hi ${i}`, { class: "op-text op-small", "data-k": `${name}-hi-${i}` })]); boxInputs[name] = ins; return labeled(t(name), el("span", { class: "op-box" }, ...ins.map(([lo, hi]) => el("span", { class: "op-iv" }, "[", lo, ", ", hi, "]")))); };
  const readBox = (name) => boxInputs[name].map(([lo, hi]) => [lo.value.trim(), hi.value.trim()]);
  // a mode switch keeps the intervals typed in the mode it leaves
  const keepBoxes = () => Object.keys(boxInputs).forEach((n) => (boxes[n] = readBox(n)));
  const fx = textIn("y", "ẋ", { "data-k": "fx" }), fy = textIn("-x - y", "ẏ", { "data-k": "fy" }), barrierIn = textIn("x^2 + y^2 - 1", t("barrier_b"), { "data-k": "barrier" }), deg = numIn(2, t("degree_b"), { min: "1", max: "4", "data-k": "degree" });
  const find = el("input", { type: "checkbox", "aria-label": t("find_one"), "data-k": "find" });
  const form = el("div", { class: "op-form" }), out = el("div", { class: "results" });
  const senseSeg = segmented([["nonneg", "sense_nonneg"], ["pos", "sense_pos"]], S.sense, (v) => (S.sense = v));
  const draw = () => {
    if (S.mode === "decide") form.replaceChildren(labeled(t("vars"), vars), labeled(t("poly"), poly), boxField("box"), labeled("", senseSeg));
    else form.replaceChildren(labeled(t("vars"), vars), labeled("ẋ", fx), labeled("ẏ", fy), boxField("domain"), boxField("init"), boxField("unsafe"), labeled(t("barrier_b"), barrierIn), el("div", { class: "op-check" }, el("label", {}, find, " ", t("find_one")), labeled(t("degree_b"), deg)));
  };
  const modes = segmented([["decide", "m_decide"], ["barrier", "m_barrier"]], S.mode, (m) => { keepBoxes(); S.mode = m; draw(); });
  opusInfo().then((info) => opusShelf(root, info.aludel, (e) => {
    S.mode = e.op; vars.value = e.vars;
    if (e.op === "decide") { poly.value = e.poly; boxes.box = e.box; S.sense = e.sense || "nonneg"; senseSeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.v === S.sense))); }
    else { fx.value = e.field[0]; fy.value = e.field[1]; boxes.domain = e.domain; boxes.init = e.init; boxes.unsafe = e.unsafe; barrierIn.value = e.barrier || ""; find.checked = !!e.synthesize; deg.value = e.degree || 2; }
    modes.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.v === S.mode)));
    draw(); run();
  })).then(() => { root.append(modes, form, el("div", { class: "op-row" }, runButton(() => run())), out); if (!poly.value) poly.value = OPUS.info.aludel[0].poly; draw(); });
  const run = async () => {
    busy(out);
    try {
      const req = S.mode === "decide" ? { op: "decide", vars: vars.value, poly: poly.value, box: readBox("box"), sense: S.sense }
        : { op: "barrier", vars: vars.value, field: [fx.value, fy.value], domain: readBox("domain"), init: readBox("init"), unsafe: readBox("unsafe"),
            barrier: barrierIn.value, synthesize: find.checked, degree: +deg.value };
      const r = await api("/v1/vapor/aludel", req);
      out.replaceChildren(...(S.mode === "decide" ? decideView(r, req) : barrierView(r, req)));
    } catch (e) { out.replaceChildren(errLine(e)); }
  };
  poly.addEventListener("run", run);
}
const qf = (s) => { const m = /^\s*(-?\d+(?:\.\d+)?)\s*\/\s*(\d+)\s*$/.exec(s); return m ? +m[1] / +m[2] : +s; };
function decideView(r, req) {
  const out = [];
  if (r.verdict === "certified") out.push(touchstone(`${t("certified")}: ${r.polynomial} ${req.sense === "pos" ? "> 0" : "≥ 0"}`, [{ name: t("replayed"), ok: r.replayed, detail: `${r.witness.bits} bits` }, { name: t("cells"), ok: true, detail: `${r.cells} · ${t("depth")} ${r.depth}` }]));
  else if (r.verdict === "refuted") out.push(touchstone(`${t("refuted")}: ${t("at_point")} ${Object.entries(r.point).map(([k, v]) => `${k} = ${v}`).join(", ")} ${t("value_is")} ${r.value}`, [{ name: t("refuted"), ok: false, detail: r.value }], true));
  else if (r.verdict === "exhausted") out.push(touchstone(`${t("exhausted")}: ${t("cell_left")} ${r.cell.map(([a, b]) => `[${a}, ${b}]`).join(" × ")}`, [{ name: t("exhausted"), ok: false, detail: `${r.cells} ${t("cells")}` }], true));
  else out.push(el("p", { class: "notice err", text: r.why || r.verdict }));
  if (req.box.length === 2) out.push(boxPicture(r, req.box.map(([a, b]) => [qf(a), qf(b)])));
  return out;
}
/* the box, subdivided: certified leaves in verdigris with gold edges, the refuting vertex in cinnabar, the exhausted cell in lead */
function boxPicture(r, [[x0, x1], [y0, y1]]) {
  const W = 420, H = 420, P = 26, X = (x) => P + ((x - x0) / (x1 - x0)) * (W - 2 * P), Y = (y) => H - P - ((y - y0) / (y1 - y0)) * (H - 2 * P);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "aludel-box", role: "img", "aria-label": r.verdict });
  svg.append(svgEl("rect", { x: P, y: P, width: W - 2 * P, height: H - 2 * P, class: "frame" }));
  (r.leaves || []).forEach(([a, b, c, d]) => svg.append(svgEl("rect", { x: X(a).toFixed(1), y: Y(d).toFixed(1), width: (X(b) - X(a)).toFixed(1), height: (Y(c) - Y(d)).toFixed(1), class: "leaf" })));
  if (r.cell_f) { const [[a, b], [c, d]] = r.cell_f; svg.append(svgEl("rect", { x: X(a) - 3, y: Y(d) - 3, width: Math.max(X(b) - X(a), 2) + 6, height: Math.max(Y(c) - Y(d), 2) + 6, class: "pending" })); }
  if (r.point_f) svg.append(svgEl("circle", { cx: X(r.point_f[0]), cy: Y(r.point_f[1]), r: 7, class: "refute" }));
  [[x0, X(x0), H - P + 16, "middle"], [x1, X(x1), H - P + 16, "middle"]].forEach(([v, x, y, a]) => { const tx = svgEl("text", { x, y, "text-anchor": a, class: "axt" }); tx.textContent = +v.toPrecision(3); svg.append(tx); });
  [[y0, Y(y0)], [y1, Y(y1)]].forEach(([v, y]) => { const tx = svgEl("text", { x: P - 6, y: y + 4, "text-anchor": "end", class: "axt" }); tx.textContent = +v.toPrecision(3); svg.append(tx); });
  return el("figure", { class: "op-fig" }, svg);
}
function barrierView(r, req) {
  if (r.verdict === "not found") return [el("p", { class: "notice err", text: `${t("not_found")}: ${r.why}` })];
  const checks = r.conditions.map((c) => ({ name: t("cond_" + c.name), ok: c.result.verdict === "certified", detail: t(c.result.verdict) }));
  const out = [touchstone(`${r.verdict === "proved" ? t("proved") : r.verdict}: B = ${r.barrier}`, checks, r.verdict !== "proved")];
  if (r.lp && r.lp.lp_rounds) out.push(el("p", { class: "muted", text: t("lp_note", r.lp.lp_rounds, r.lp.lp_rows, r.lp.lp_rows_total) }));
  if (r.plot) out.push(phasePortrait(r.plot, req));
  return out;
}
/* the phase portrait: the flow (arrows), the initial set, the unsafe set, and B = 0 by marching squares */
function phasePortrait(pl, req) {
  const [x0, x1, y0, y1] = pl.box, W = 440, H = 440, P = 24;
  const X = (x) => P + ((x - x0) / (x1 - x0)) * (W - 2 * P), Y = (y) => H - P - ((y - y0) / (y1 - y0)) * (H - 2 * P);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "phase", role: "img", "aria-label": t("b_zero") });
  svg.append(svgEl("rect", { x: P, y: P, width: W - 2 * P, height: H - 2 * P, class: "frame" }));
  const rectOf = (bx, cls) => { const [[a, b], [c, d]] = bx.map(([lo, hi]) => [qf(lo), qf(hi)]); svg.append(svgEl("rect", { x: X(a), y: Y(d), width: X(b) - X(a), height: Y(c) - Y(d), class: cls })); };
  rectOf(req.init, "init"); rectOf(req.unsafe, "unsafe");
  // arrows, normalised
  const n = 15, mags = pl.arrows.map(([u, v]) => Math.hypot(u, v)), mx = Math.max(...mags, 1e-12);
  pl.arrows.forEach(([u, v], k) => {
    const i = k % n, j = Math.floor(k / n), x = x0 + ((x1 - x0) * i) / (n - 1), y = y0 + ((y1 - y0) * j) / (n - 1);
    const L = 9 * Math.sqrt(Math.hypot(u, v) / mx), m = Math.hypot(u, v) || 1, dx = (u / m) * L, dy = (-v / m) * L;
    svg.append(svgEl("line", { x1: X(x).toFixed(1), y1: Y(y).toFixed(1), x2: (X(x) + dx).toFixed(1), y2: (Y(y) + dy).toFixed(1), class: "flow" }));
    svg.append(svgEl("circle", { cx: (X(x) + dx).toFixed(1), cy: (Y(y) + dy).toFixed(1), r: "1.4", class: "flowtip" }));
  });
  // B = 0: marching squares on the 41×41 samples
  const N = pl.b.n, B = (i, j) => pl.b.values[j * N + i], gx = (i) => x0 + ((x1 - x0) * i) / (N - 1), gy = (j) => y0 + ((y1 - y0) * j) / (N - 1);
  let d = "";
  const lerp = (a, b, va, vb) => a + ((b - a) * va) / (va - vb);
  for (let j = 0; j < N - 1; j++) for (let i = 0; i < N - 1; i++) {
    const v = [B(i, j), B(i + 1, j), B(i + 1, j + 1), B(i, j + 1)], c = [[gx(i), gy(j)], [gx(i + 1), gy(j)], [gx(i + 1), gy(j + 1)], [gx(i), gy(j + 1)]];
    const pts = [];
    for (let e = 0; e < 4; e++) { const a = e, b = (e + 1) % 4; if ((v[a] < 0) !== (v[b] < 0)) pts.push([lerp(c[a][0], c[b][0], v[a], v[b]), lerp(c[a][1], c[b][1], v[a], v[b])]); }
    for (let p = 0; p + 1 < pts.length; p += 2) d += `M${X(pts[p][0]).toFixed(1)} ${Y(pts[p][1]).toFixed(1)}L${X(pts[p + 1][0]).toFixed(1)} ${Y(pts[p + 1][1]).toFixed(1)}`;
  }
  if (d) svg.append(svgEl("path", { d, class: "bzero" }));
  return el("figure", { class: "op-fig" }, svg, legend([["init", t("init")], ["unsafe", t("unsafe")], ["bzero", t("b_zero")]]));
}

/* ================================================================ Tabula */
function buildTabula(root) {
  opusHead(root, "tabula");
  const ta = editor("op-tabula", 14), out = el("div", { class: "results" });
  const S = (OPUS_STATE.tabula ||= { facts: {}, last: null });
  opusInfo().then((info) => opusShelf(root, info.tabula, (e) => { ta.value = e.text; S.facts = {}; run(); }))
    .then(() => { root.append(ta, el("div", { class: "op-row" }, runButton(() => run())), out); if (!ta.value) ta.value = OPUS.info.tabula[0].text; });
  const run = async () => {
    busy(out);
    try { S.last = await api("/v1/vapor/tabula", { text: ta.value, facts: S.facts }); out.replaceChildren(...tabulaView(S, run)); }
    catch (e) { out.replaceChildren(errLine(e)); }
  };
  ta.addEventListener("run", run);
}
function tabulaView(S, rerun) {
  const r = S.last;
  const factBar = el("div", { class: "facts-bar", role: "group", "aria-label": t("facts") }, el("span", { class: "muted", text: t("facts") }),
    ...r.facts.map((f) => { const b = el("button", { type: "button", class: "fact" + (S.facts[f] ? " on" : ""), "aria-pressed": String(!!S.facts[f]), text: f }); b.onclick = () => { S.facts[f] = !S.facts[f]; rerun(); }; return b; }));
  const pos = r.positions || { active: [], overridden: [], claims: [], clashes: [] };
  const active = new Set((pos.active || []).map((c) => c.id)), over = new Set(pos.overridden || []);
  const clashing = new Set((pos.clashes || []).flatMap((c) => c.clauses));
  const tablet = el("ol", { class: "tablet" }, ...r.clauses.map((c) => el("li", { "data-id": c.id, class: [active.has(c.id) ? "active" : "", over.has(c.id) ? "over" : "", clashing.has(c.id) ? "clash" : ""].join(" ") }, el("b", { text: c.id }), el("span", { text: c.text.replace(/^[^:]+:\s*/, "") }))));
  const mark = (ids, on) => ids.forEach((id) => tablet.querySelectorAll(`[data-id="${id}"]`).forEach((n) => n.classList.toggle("hl", on)));
  const scen = (sc) => { const on = Object.entries(sc).filter(([, v]) => v).map(([k]) => k); return on.length ? on.join(" ∧ ") : t("no_fact"); };
  const card = (f, cls, title) => {
    const set = el("button", { type: "button", class: "quiet", text: t("show_scenario") });
    set.onclick = () => { S.facts = { ...f.scenario }; rerun(); };
    const c = el("div", { class: "clash " + cls, tabindex: "0" }, el("b", { text: `${f.clauses.join(" × ")} · ${title}` }), el("span", { text: f.why }), el("small", { text: `${t("when")} ${scen(f.scenario)}` }), set);
    c.onmouseenter = c.onfocus = () => mark(f.clauses, true); c.onmouseleave = c.onblur = () => mark(f.clauses, false);
    return c;
  };
  const verdict = r.verdict === "consistent" ? t("consistent") : t("antinomies");
  const checks = [...r.findings.map((f) => ({ name: f.clauses.join(" × "), ok: false, detail: t(f.kind) })), ...r.resolved.map((f) => ({ name: f.clauses.join(" × "), ok: true, detail: t("prevails", f.prevails) })),
    ...r.checked_pairs.map((p) => ({ name: p.clauses.join(" × "), ok: p.drup, detail: "DRUP" }))];
  const claims = (pos.claims || []).map((c) => el("li", { text: t("holder", c.holder, c.against, c.action) }));
  return [touchstone(`${verdict} · ${t("never_clash", r.checked_pairs.length)}`, checks, r.verdict !== "consistent"), factBar,
    el("div", { class: "tabula-grid" }, el("div", {}, el("h3", { text: t("contract") }), tablet),
      el("div", {}, ...r.findings.map((f) => card(f, f.kind, t(f.kind))), ...r.resolved.map((f) => card(f, "resolved", t("prevails", f.prevails))),
        ...r.silences.map((s) => el("div", { class: "clash silence" }, el("b", { text: `${t("silence")} ${s.party} · ${s.action}` }), el("small", { text: `${t("when")} ${scen(s.scenario)}` }))),
        claims.length ? el("div", {}, el("h3", { text: t("claims") }), el("ul", { class: "claims" }, ...claims)) : ""))];
}

/* ================================================================ Cupel */
function buildCupel(root) {
  opusHead(root, "cupel");
  const n = numIn(32, t("rows_n"), { min: "4", max: "128", "data-k": "n" }), k = numIn(64, t("cols_k"), { min: "16", max: "256", step: "16", "data-k": "k" }), seed = numIn(5, t("seed"), { min: "1", "data-k": "seed" });
  const trials = numIn(8, t("trials"), { min: "2", max: "24", "data-k": "trials" }), bit = el("input", { type: "range", min: "0", max: "31", value: "26", "aria-label": t("bit_ex"), "data-k": "bit" });
  const out = el("div", { class: "results" });
  root.append(el("div", { class: "op-row" }, labeled(t("rows_n"), n), labeled(t("cols_k"), k), labeled(t("seed"), seed), labeled(t("trials"), trials), labeled(t("bit_ex"), bit), runButton(() => run())), out);
  const run = async () => {
    busy(out);
    try { const r = await api("/v1/vapor/cupel", { n: +n.value, k: +k.value, seed: +seed.value, trials: +trials.value, bit: +bit.value }); out.replaceChildren(...cupelView(r)); }
    catch (e) { out.replaceChildren(errLine(e)); }
  };
  run();
}
/* the 32 bits of a float, most significant first: each cell's gold is the share of flips caught there */
function cupelView(r) {
  const prof = [...r.profile].sort((a, b) => b.bit - a.bit);
  const cell = (p) => { const f = p.detected / p.trials; const c = el("div", { class: "bitcell", title: `bit ${p.bit}: ${p.detected}/${p.trials}`, role: "img", "aria-label": `bit ${p.bit} ${Math.round(f * 100)} %` }, el("i")); c.firstChild.style.height = `${Math.round(f * 100)}%`; return c; };
  const strip = el("div", { class: "float32" }, el("div", { class: "fgroup sign" }, el("small", { text: t("sign") }), el("div", { class: "cells" }, ...prof.slice(0, 1).map(cell))),
    el("div", { class: "fgroup exp" }, el("small", { text: t("exponent") }), el("div", { class: "cells" }, ...prof.slice(1, 9).map(cell))),
    el("div", { class: "fgroup man" }, el("small", { text: t("mantissa") }), el("div", { class: "cells" }, ...prof.slice(9).map(cell))));
  const ex = r.example;
  const caught = prof.filter((p) => p.detected === p.trials).length;
  return [touchstone(`${caught}/32 ${t("op_caught")} · ${t("int8_exact", r.int8.bits_detected)}`, [{ name: `bit ${ex.bit}`, ok: ex.verdict === "corrupt", detail: ex.verdict }, { name: "int8", ok: r.int8.bits_detected === 32, detail: `${r.int8.bits_detected}/32` }]),
    strip, el("p", { class: "muted", text: t("flip_ex", ex.bit, wbNum(ex.before, 6), wbNum(ex.after, 6), ex.verdict) }), el("p", { class: "muted", text: `${t("cost")}: ${r.check_cost}` })];
}

/* ================================================================ Amalgam */
function buildAmalgam(root) {
  opusHead(root, "amalgam");
  const ta = editor("op-amalgam", 4), out = el("div", { class: "results" }), S = { fmt: "f64" };
  const fmt = segmented([["f64", "op_f64"], ["f32", "op_f32"]], S.fmt, (v) => { S.fmt = v; run(); });
  opusInfo().then((info) => opusShelf(root, info.amalgam, (e) => { ta.value = e.numbers; run(); }))
    .then(() => { root.append(ta, el("div", { class: "op-row" }, labeled(t("format"), fmt), runButton(() => run())), out); if (!ta.value) { ta.value = OPUS.info.amalgam[0].numbers; run(); } });
  const run = async () => {
    busy(out);
    try { const r = await api("/v1/vapor/amalgam", { numbers: ta.value, format: S.fmt }); out.replaceChildren(...amalgamView(r)); }
    catch (e) { out.replaceChildren(errLine(e)); }
  };
  ta.addEventListener("run", run);
}
/* the scattered sums on one line, the amalgam as the one gold mark */
function amalgamView(r) {
  const vals = r.naive.map((x) => +x.value).filter(isFinite), A = +r.amalgam.value;
  const lo = Math.min(...vals, A), hi = Math.max(...vals, A), span = hi - lo || Math.abs(A) || 1;
  const W = 620, H = 120, P = 30, X = (v) => P + ((v - lo + span * 0.05) / (span * 1.1)) * (W - 2 * P);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "amalgam-line", role: "img", "aria-label": t("distinct", r.distinct_naive, r.naive.length) });
  svg.append(svgEl("line", { x1: P, x2: W - P, y1: 70, y2: 70, class: "axis" }));
  const seen = {};
  r.naive.forEach((x) => { const v = +x.value; if (!isFinite(v)) return; const k = x.bits; seen[k] = (seen[k] || 0) + 1; svg.append(svgEl("circle", { cx: X(v).toFixed(1), cy: (70 - 9 * (seen[k] - 1)).toFixed(1), r: 4.5, class: "naive" })); });
  svg.append(svgEl("line", { x1: X(A), x2: X(A), y1: 30, y2: 92, class: "mark" }));
  const tx = svgEl("text", { x: X(A), y: 24, "text-anchor": "middle", class: "marktext" }); tx.textContent = `${t("amalgam_mark")} ${r.amalgam.value}`; svg.append(tx);
  return [touchstone(t("distinct", r.distinct_naive, r.naive.length), [{ name: t("amalgam_mark"), ok: true, detail: r.amalgam.value }]),
    el("figure", { class: "op-fig" }, svg), el("p", { class: "muted" }, el("b", { text: `${t("exact_sum")}: ` }), el("code", { text: r.exact })),
    table([t("orders"), r.format, "bits"], r.naive.map((x) => [x.order, x.value, x.bits]).concat([[t("amalgam_mark"), r.amalgam.value, r.amalgam.bits]]))];
}

/* ================================================================ wiring */
const OPUS_DESKS = { rebis: buildRebis, aludel: buildAludel, tabula: buildTabula, cupel: buildCupel, amalgam: buildAmalgam };
const OPUS_BUILT = {};
function opusOnce(k) { if (!OPUS_BUILT[k]) { OPUS_BUILT[k] = true; OPUS_DESKS[k]($("op-" + k)); } }
Object.keys(OPUS_DESKS).forEach((k) => shown.set("p-" + k, () => opusOnce(k)));
// a language switch redraws the desks; what the person wrote stays, and a result that was shown is computed again
rerender.push(() => Object.keys(OPUS_BUILT).forEach((k) => {
  const root = $("op-" + k), kept = {};
  const key = (n) => n.id || n.dataset.k;
  root.querySelectorAll("textarea[id], input[data-k]").forEach((n) => (kept[key(n)] = n.type === "checkbox" ? n.checked : n.value));
  const ran = !!root.querySelector(".results > *");
  delete OPUS_BUILT[k]; opusOnce(k);
  const restore = (tries) => {
    if (!root.querySelector(".op-row .primary") && tries > 0) return setTimeout(() => restore(tries - 1), 50);
    root.querySelectorAll("textarea[id], input[data-k]").forEach((n) => { const v = kept[key(n)]; if (v === undefined) return; if (n.type === "checkbox") n.checked = v; else n.value = v; });
    if (ran) root.querySelector(".op-row .primary")?.click();
  };
  restore(100);
}));
PALETTE_ITEMS.push(...Object.keys(OPUS_DESKS).map((k) => () => ({ label: t(k), sub: t(k + "_s"), run: () => openPanel(k) })));
applyLang();
})();
