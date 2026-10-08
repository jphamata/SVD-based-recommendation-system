"use strict";
/* workbench.js — the console's 0.12 panels (docs/CONSOLE.md §0.12): the
   workbench, engineering, logic, boards & cards, proteins and the renderer;
   the navigation in alphabetical order (per language) and a command palette.
   It runs after the page's main script and uses its helpers ($, el, t, api,
   lineChart, legend, svgEl, download, b64blob, saveArchive, chipV, fmt, dl). */

/* ================================================================== words */
Object.assign(I18N.en, {
  g_solve: "Solve", bench: "Workbench", bench_s: "any equation, with units and checks", eng: "Engineering", eng_s: "circuits, grids, structures, pipes, plants",
  logic: "Logic", logic_s: "claims settled with a checkable proof", boards: "Boards & cards", boards_s: "chess, shogi, Go, poker — with proofs",
  protein: "Proteins", protein_s: "structures compared, folded, read from evolution", render: "Render", render_s: "physically based light, on your GPU",
  pal_open: "Go to…", pal_hint: "↑ ↓ to move · Enter to open · Esc to close", pal_ph: "Search panels and examples…",
  example: "Example", run: "Run", running: "running…", solve_btn: "Solve", ensemble_btn: "Ensemble on the native worker", copy: "Copy", certificate: "Certificate",
  verified: "verified", refused: "refused", not_verified: "not verified", ms: (n) => `${n} ms`, frames: "frames", remove: "remove",
  bench_lede: "Type the problem as you would write it: formulas with units, ODE systems (x' = …), PDEs (u_t = …, poisson …), systems of equations, fits, minimisations. The workbench recognises the kind, checks the units before anything runs, and shows with each answer the numbers that let you judge it — steps, residuals, the observed order of accuracy.",
  eng_lede: "Each tool reads the text an engineer would write — a netlist, a bus list, nodes and members, a mesh, pipes, reactions — and answers with a certificate computed independently of the solver: Kirchhoff's law re-evaluated, the power mismatch, equilibrium of loads and reactions, continuity and loop energy, conserved moieties.",
  logic_lede: "Claims of four logics — propositional (DIMACS or formulas), finite combinatorics, equational theories, polynomial geometry — settled by a procedure that proposes and a checker that decides. Anyone can propose here, a person or a language model (the MCP tool logic_check): acceptance never depends on who proposed.",
  boards_lede: "The rules of chess, shogi and Go pinned by the published counts (perft, legal positions), an engine to play against, mate proofs checked move by move, any k-in-a-row game solved exactly, and poker solved to an equilibrium whose exploitability is measured.",
  protein_lede: "Protein structure from first principles: the metrics of structure prediction (TM-score = TM-align, GDT, lDDT), folding from contacts by distance geometry, and contacts read from co-evolution — on alignments sampled from a model planted on a real protein, so the truth is known.",
  render_lede: "Physically based rendering: light transport by Monte Carlo path tracing — diffuse, metal, glass, area lights, sky and sun — progressively on your GPU, with the server's tracer as the reference (the white-furnace test and the N^−½ convergence are its checks).",
  kinds: { worksheet: "Worksheet", ode: "Ordinary differential equations", pde: "Partial differential equation", nonlinear: "Nonlinear system", fit: "Fit", minimize: "Optimisation" },
  ex_bench: { osc: "Damped oscillator with units", lorenz: "Lorenz attractor", robertson: "Robertson's stiff kinetics", ball: "Projectile until it lands (event)", sir: "SIR epidemic",
    heat: "Heat equation, verified (MMS)", fisher: "Fisher–KPP reaction–diffusion", consol: "Terzaghi consolidation of a clay layer", wave: "A plucked string (wave)", poisson: "Poisson in 2-D, verified",
    sheet: "Worksheet: cantilever deflection", circle: "Circle ∩ hyperbola: every root", fit: "Fit an exponential decay", rosen: "Rosenbrock's valley", cons: "Minimise with a constraint", ens: "Uncertainty: 4096 oscillators (native)" },
  e_kinds: { circuit: "Circuit (SPICE)", power: "Power flow", structure: "Frames & trusses", fem: "Plane stress (FEM)", pipes: "Pipe network", reactions: "Reactions", flash: "Flash (VLE)", distill: "Distillation" },
  ex_logic: { schur: "Schur number S(3)", vdw: "van der Waerden W(3; 2)", ramsey: "Ramsey R(3, 3)", taut: "A tautology (syllogism)", equiv: "De Morgan, as an equivalence", nontaut: "Converse (not valid)",
    dimacs: "A CNF in DIMACS", php: "Pigeonhole 6 → 5", queens: "8 queens", group: "Groups: Knuth–Bendix", thales: "Thales' theorem (Gröbner)", lexsys: "A polynomial system (lex basis)" },
  b_kinds: { chess: "Chess", shogi: "Shogi", go: "Go", mnk: "k in a row", poker: "Poker" },
  verdict: "Verdict", model: "Model", witness: "Witness", proof: "Proof", lemmas: "lemmas", checked: "checked", core: "core",
  engine: "Engine", engine_reply: "engine replies", new_game: "New game", undo: "Undo", flip: "Flip", pass: "Pass", analyse: "Analyse", prove_mate: "Prove mate in", perft: "Perft",
  depth: "depth", sims: "simulations", score: "score", to_move: (w) => `${w} to move`, white: "White", black: "Black", checkmate: "Checkmate", stalemate: "Stalemate",
  solve_exactly: "Solve exactly", value: "value", best_moves: "best moves", win: "win", draw: "draw", loss: "loss",
  p_solve: "Solve by CFR+", exploitability: "exploitability (chips/hand)", game_value: "game value", iterations: "iterations", strategy: "strategy",
  samples_pdb: "Samples", pipeline: "Run the pipeline: alignment → DCA → fold", compare: "Compare two structures", align: "Align two sequences",
  precision: "Precision of the top-k predictions (k = number of true contacts)", caught: "bias detected ✓", not_caught: "bias missed ✗", bus_legend: "bus colour: |V − 1| below 3 % green, below 5 % amber, otherwise red · line width ∝ |P|", tm: "TM-score", contacts: "contacts", truth: "true", predicted: "predicted", native: "native",
  gpu: "GPU (progressive)", reference: "Reference on the server", reset: "Reset", exposure: "exposure", spp: "samples / pixel", furnace: "Furnace test", control: "control",
  render_reference: "Render the reference", unavailable_gpu: "The GPU tracer needs WebGL2 with float targets; the reference renderer on the server still works.",
  sc_npcs: "Inhabitants", sc_npc_add: "+ inhabitant", sc_labels: "names", sc_timeline: "Timeline", sc_gif: "GIF (exact frames)", sc_frames: "drawing exact frames…",
  sc_name: "name", sc_behavior: "behaviour", sc_speed: "speed", sc_scale: "size", sc_color: "colour", sc_say: "Say", sc_say_ph: "a line to say…", sc_goto: "go to…",
  sc_wp: "Set a route (click the ground)", sc_wp_done: "Route done", sc_wp_hint: "click on the ground to add points to the route", sc_wp_n: (n) => `${n} route point(s)`,
  beh: { wander: "wanders", idle: "stays", patrol: "patrols", follow: "follows", flee: "flees", goto: "goes" },
  act: { wave: "wave", dance: "dance", sit: "sit", jump: "jump", run: "run", stand: "stand" },
  place: { door: "the door", left: "the left", right: "the right", front: "the front", back: "the back", light: "the light", center: "the centre" },
});
Object.assign(I18N.pt, {
  g_solve: "Resolver", bench: "Bancada", bench_s: "qualquer equação, com unidades e conferência", eng: "Engenharia", eng_s: "circuitos, redes, estruturas, tubulações, plantas",
  logic: "Lógica", logic_s: "afirmações decididas com prova conferível", boards: "Tabuleiros e cartas", boards_s: "xadrez, shogi, Go, pôquer — com provas",
  protein: "Proteínas", protein_s: "estruturas comparadas, dobradas, lidas da evolução", render: "Render", render_s: "luz fisicamente baseada, na sua GPU",
  pal_open: "Ir para…", pal_hint: "↑ ↓ para mover · Enter para abrir · Esc para fechar", pal_ph: "Buscar painéis e exemplos…",
  example: "Exemplo", run: "Rodar", running: "rodando…", solve_btn: "Resolver", ensemble_btn: "Conjunto no worker nativo", copy: "Copiar", certificate: "Certificado",
  verified: "conferido", refused: "recusado", not_verified: "não conferido", frames: "quadros", remove: "remover",
  bench_lede: "Escreva o problema como no papel: fórmulas com unidades, sistemas de EDOs (x' = …), EDPs (u_t = …, poisson …), sistemas de equações, ajustes, otimizações. A bancada reconhece o tipo, confere as unidades antes de rodar e mostra junto de cada resposta os números que permitem julgá-la — passos, resíduos, a ordem de precisão observada.",
  eng_lede: "Cada ferramenta lê o texto que um engenheiro escreveria — netlist, lista de barras, nós e barras, malha, tubos, reações — e responde com um certificado calculado independentemente do solver: a lei de Kirchhoff reavaliada, o desbalanço de potência, o equilíbrio entre cargas e reações, a continuidade e a energia nas malhas, os invariantes conservados.",
  logic_lede: "Afirmações de quatro lógicas — proposicional (DIMACS ou fórmulas), combinatória finita, teorias equacionais, geometria polinomial — decididas por um procedimento que propõe e um verificador que decide. Qualquer um pode propor aqui, uma pessoa ou um modelo de linguagem (a ferramenta MCP logic_check): a aceitação nunca depende de quem propôs.",
  boards_lede: "As regras do xadrez, do shogi e do Go fixadas pelas contagens publicadas (perft, posições legais), um motor para jogar contra, provas de mate conferidas lance a lance, qualquer jogo de k em linha resolvido exatamente, e o pôquer resolvido até um equilíbrio cuja explorabilidade é medida.",
  protein_lede: "Estrutura de proteínas por primeiros princípios: as métricas da predição de estrutura (TM-score = TM-align, GDT, lDDT), o dobramento a partir de contatos por geometria de distâncias, e os contatos lidos da coevolução — em alinhamentos amostrados de um modelo plantado numa proteína real, de modo que a verdade é conhecida.",
  render_lede: "Renderização fisicamente baseada: o transporte de luz por traçado de caminhos de Monte Carlo — difuso, metal, vidro, luzes de área, céu e sol — progressivo na sua GPU, com o traçador do servidor como referência (o teste da fornalha branca e a convergência N^−½ são a conferência).",
  kinds: { worksheet: "Planilha", ode: "Equações diferenciais ordinárias", pde: "Equação diferencial parcial", nonlinear: "Sistema não linear", fit: "Ajuste", minimize: "Otimização" },
  ex_bench: { osc: "Oscilador amortecido com unidades", lorenz: "Atrator de Lorenz", robertson: "Cinética rígida de Robertson", ball: "Projétil até tocar o chão (evento)", sir: "Epidemia SIR",
    heat: "Equação do calor, verificada (MMS)", fisher: "Reação–difusão de Fisher–KPP", consol: "Adensamento de Terzaghi de uma argila", wave: "Uma corda dedilhada (onda)", poisson: "Poisson em 2-D, verificada",
    sheet: "Planilha: flecha de uma viga em balanço", circle: "Círculo ∩ hipérbole: todas as raízes", fit: "Ajuste de um decaimento exponencial", rosen: "O vale de Rosenbrock", cons: "Minimizar com restrição", ens: "Incerteza: 4096 osciladores (nativo)" },
  e_kinds: { circuit: "Circuito (SPICE)", power: "Fluxo de potência", structure: "Pórticos e treliças", fem: "Estado plano (MEF)", pipes: "Rede de tubulações", reactions: "Reações", flash: "Flash (ELV)", distill: "Destilação" },
  ex_logic: { schur: "Número de Schur S(3)", vdw: "van der Waerden W(3; 2)", ramsey: "Ramsey R(3, 3)", taut: "Uma tautologia (silogismo)", equiv: "De Morgan, como equivalência", nontaut: "A recíproca (não válida)",
    dimacs: "Uma CNF em DIMACS", php: "Casa dos pombos 6 → 5", queens: "8 rainhas", group: "Grupos: Knuth–Bendix", thales: "Teorema de Tales (Gröbner)", lexsys: "Um sistema polinomial (base lex)" },
  b_kinds: { chess: "Xadrez", shogi: "Shogi", go: "Go", mnk: "k em linha", poker: "Pôquer" },
  verdict: "Veredito", model: "Modelo", witness: "Testemunha", proof: "Prova", lemmas: "lemas", checked: "conferidos", core: "núcleo",
  engine: "Motor", engine_reply: "o motor responde", new_game: "Nova partida", undo: "Desfazer", flip: "Girar", pass: "Passar", analyse: "Analisar", prove_mate: "Provar mate em", perft: "Perft",
  depth: "profundidade", sims: "simulações", score: "placar", to_move: (w) => `${w} joga`, white: "Brancas", black: "Pretas", checkmate: "Xeque-mate", stalemate: "Afogamento",
  solve_exactly: "Resolver exatamente", value: "valor", best_moves: "melhores lances", win: "vitória", draw: "empate", loss: "derrota",
  p_solve: "Resolver por CFR+", exploitability: "explorabilidade (fichas/mão)", game_value: "valor do jogo", iterations: "iterações", strategy: "estratégia",
  samples_pdb: "Amostras", pipeline: "Rodar o pipeline: alinhamento → DCA → dobra", compare: "Comparar duas estruturas", align: "Alinhar duas sequências",
  precision: "Precisão das k primeiras predições (k = número de contatos verdadeiros)", caught: "viés detectado ✓", not_caught: "viés não detectado ✗", bus_legend: "cor da barra: |V − 1| abaixo de 3 % verde, abaixo de 5 % âmbar, senão vermelho · espessura ∝ |P|", tm: "TM-score", contacts: "contatos", truth: "verdadeiros", predicted: "previstos", native: "nativa",
  gpu: "GPU (progressivo)", reference: "Referência no servidor", reset: "Recomeçar", exposure: "exposição", spp: "amostras / pixel", furnace: "Teste da fornalha", control: "controle",
  render_reference: "Renderizar a referência", unavailable_gpu: "O traçador na GPU precisa de WebGL2 com alvos float; o renderizador de referência do servidor continua disponível.",
  sc_npcs: "Habitantes", sc_npc_add: "+ habitante", sc_labels: "nomes", sc_timeline: "Linha do tempo", sc_gif: "GIF (quadros exatos)", sc_frames: "desenhando quadros exatos…",
  sc_name: "nome", sc_behavior: "comportamento", sc_speed: "velocidade", sc_scale: "tamanho", sc_color: "cor", sc_say: "Dizer", sc_say_ph: "uma fala…", sc_goto: "ir até…",
  sc_wp: "Traçar uma rota (clique no chão)", sc_wp_done: "Rota pronta", sc_wp_hint: "clique no chão para acrescentar pontos à rota", sc_wp_n: (n) => `${n} ponto(s) de rota`,
  beh: { wander: "passeia", idle: "fica", patrol: "patrulha", follow: "segue", flee: "foge", goto: "vai" },
  act: { wave: "acenar", dance: "dançar", sit: "sentar", jump: "pular", run: "correr", stand: "levantar" },
  place: { door: "a porta", left: "a esquerda", right: "a direita", front: "a frente", back: "o fundo", light: "a luz", center: "o centro" },
});

/* ============================================================= navigation */
// items sorted alphabetically within their group, in the language shown (Intl.Collator); arrow keys follow the visible order
function sortNav() {
  const list = document.querySelector('nav [role="tablist"]');
  const kids = [...list.children];
  const col = new Intl.Collator(lang === "pt" ? "pt-BR" : "en", { sensitivity: "base" });
  let i = 0;
  const out = [];
  while (i < kids.length) {
    if (kids[i].classList.contains("group")) {
      out.push(kids[i]); i++;
      const run = [];
      while (i < kids.length && !kids[i].classList.contains("group")) run.push(kids[i++]);
      if (!out[out.length - 1].dataset.fixed) run.sort((a, b) => col.compare(a.querySelector("b").textContent, b.querySelector("b").textContent));
      out.push(...run);
    } else out.push(kids[i++]);
  }
  out.forEach((n) => list.append(n));
}
function tabOrder() { return [...document.querySelectorAll('nav [role="tab"]')]; }
tabOrder().forEach((tb) => {
  tb.onkeydown = (e) => {
    const d = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[e.key];
    if (!d) return;
    e.preventDefault(); const ts = tabOrder(), j = (ts.indexOf(tb) + d + ts.length) % ts.length; ts[j].focus(); select(ts[j]); onShow(ts[j]);
  };
  tb.addEventListener("click", () => onShow(tb));
});
const shown = new Map(); // panel id → init function, run once when first shown
function onShow(tb) { const id = tb.getAttribute("aria-controls"); const f = shown.get(id); if (f) { shown.delete(id); f(); } window.scrollTo(0, 0); window.history.replaceState(null, "", "#" + id.slice(2)); }
function openPanel(id) { const tb = document.querySelector(`[aria-controls="p-${id}"]`); if (tb) { select(tb); onShow(tb); tb.focus(); } }

/* the command palette: every panel, every example, alphabetical */
const PALETTE_ITEMS = [];
function paletteItems() {
  const col = new Intl.Collator(lang === "pt" ? "pt-BR" : "en", { sensitivity: "base" });
  const panels = tabOrder().map((tb) => ({ label: tb.querySelector("b").textContent, sub: tb.querySelector("span").textContent, run: () => openPanel(tb.getAttribute("aria-controls").slice(2)) }));
  return [...panels, ...PALETTE_ITEMS.map((f) => f())].sort((a, b) => col.compare(a.label, b.label));
}
function openPalette() {
  const dlg = $("palette"), q = $("palette-q"), ul = $("palette-list");
  q.placeholder = t("pal_ph"); q.value = ""; let items = paletteItems(), sel = 0;
  const draw = () => {
    const words = q.value.toLowerCase().normalize("NFD").replace(/\p{M}/gu, "").split(/\s+/).filter(Boolean);
    const norm = (s) => s.toLowerCase().normalize("NFD").replace(/\p{M}/gu, "");
    const hits = items.filter((it) => words.every((w) => norm(it.label + " " + it.sub).includes(w)));
    sel = Math.min(sel, Math.max(hits.length - 1, 0));
    ul.replaceChildren(...hits.slice(0, 60).map((it, i) => { const li = el("li", { role: "option", "aria-selected": String(i === sel) }, el("b", { text: it.label }), el("span", { text: it.sub })); li.onclick = () => { dlg.close(); it.run(); }; return li; }));
    ul.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: "nearest" });
    return hits;
  };
  let hits = draw();
  q.oninput = () => { sel = 0; hits = draw(); };
  q.onkeydown = (e) => {
    if (e.key === "ArrowDown") { sel = Math.min(sel + 1, hits.length - 1); hits = draw(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { sel = Math.max(sel - 1, 0); hits = draw(); e.preventDefault(); }
    else if (e.key === "Enter" && hits[sel]) { dlg.close(); hits[sel].run(); }
  };
  dlg.showModal(); q.focus();
}
$("palette-open").onclick = openPalette;
document.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openPalette(); } });
$("palette").addEventListener("click", (e) => { if (e.target === $("palette")) $("palette").close(); });

/* ================================================================ helpers */
const wbNum = (x, d = 4) => (x == null || Number.isNaN(x) ? "—" : typeof x !== "number" ? String(x) : x === 0 ? "0" : Math.abs(x) >= 1e5 || Math.abs(x) < 1e-3 ? x.toExponential(d - 1) : (+x.toPrecision(d + 1)).toString());
const okChip = (ok, yes, no) => chipV(!!ok, ok ? yes || t("verified") : no || t("not_verified"));
function head(root, title, lede) { root.append(el("h2", { text: t(title) }), el("p", { class: "lede", text: t(lede) })); }
function exampleSelect(examples, labels, onPick) {
  const col = new Intl.Collator(lang === "pt" ? "pt-BR" : "en", { sensitivity: "base" });
  const keys = Object.keys(examples).sort((a, b) => col.compare(labels()[a] || a, labels()[b] || b));
  const s = el("select", { "aria-label": t("example") }, el("option", { value: "", text: t("example") + "…" }), ...keys.map((k) => el("option", { value: k, text: labels()[k] || k })));
  s.onchange = () => { if (s.value) onPick(s.value, examples[s.value]); };
  return s;
}
function editor(id, rows = 14) {
  const ta = el("textarea", { id, class: "code", spellcheck: "false", rows: String(rows), autocomplete: "off" });
  ta.onkeydown = (e) => { if (e.key === "Tab") { e.preventDefault(); const s = ta.selectionStart; ta.setRangeText("  ", s, ta.selectionEnd, "end"); } if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); ta.dispatchEvent(new CustomEvent("run")); } };
  return ta;
}
function busy(out) { out.replaceChildren(el("p", { class: "muted pulse", text: t("running") })); }
function table(headers, rows, cls = "cands") { const tb = el("table", { class: cls }); tb.append(el("thead", {}, el("tr", {}, ...headers.map((h) => el("th", { text: h }))))); tb.append(el("tbody", {}, ...rows.map((r) => el("tr", {}, ...r.map((c) => (c instanceof Node ? el("td", {}, c) : el("td", { text: String(c) }))))))); return el("div", { class: "tw" }, tb); }
function facts(pairs) { return dl(pairs.map(([k, v]) => [k, typeof v === "number" ? wbNum(v) : String(v)])); }
function stat(label, value, sub) { return el("div", { class: "stat" }, el("span", { text: label }), el("b", { text: value }), sub ? el("small", { text: sub }) : ""); }
function stats(...s) { return el("div", { class: "stats" }, ...s); }
// a heat map of a grid (rows of numbers), viridis-like ramp from the page's own colours
function heatmap(grid, { w = 520, h = 220, flipY = false } = {}) {
  const cv = el("canvas", { width: String(grid[0].length), height: String(grid.length), class: "heat" });
  cv.style.width = w + "px"; cv.style.maxWidth = "100%"; cv.style.height = "auto"; cv.style.aspectRatio = `${grid[0].length} / ${grid.length}`;
  const g = cv.getContext("2d"), img = g.createImageData(grid[0].length, grid.length);
  let lo = Infinity, hi = -Infinity; grid.forEach((r) => r.forEach((v) => { if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
  const ramp = [[13, 8, 135], [84, 2, 163], [139, 10, 165], [185, 50, 137], [219, 92, 104], [244, 136, 73], [254, 188, 43], [240, 249, 33]];
  grid.forEach((r, y) => r.forEach((v, x) => {
    const tt = hi > lo ? (v - lo) / (hi - lo) : 0.5, k = Math.min(ramp.length - 2, Math.floor(tt * (ramp.length - 1))), f = tt * (ramp.length - 1) - k;
    const c = ramp[k].map((a, i) => a + (ramp[k + 1][i] - a) * f), yy = flipY ? grid.length - 1 - y : y, o = (yy * grid[0].length + x) * 4;
    img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
  }));
  g.putImageData(img, 0, 0);
  return el("figure", { class: "heatfig" }, cv, el("figcaption", { class: "muted", text: `${wbNum(lo)} … ${wbNum(hi)}` }));
}
const seg = (keys, labels, cur, on) => { const d = el("div", { class: "seg", role: "radiogroup" }); keys.forEach((k) => { const b = el("button", { type: "button", role: "radio", "aria-checked": String(k === cur), text: labels[k] || k }); b.onclick = () => { d.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x === b))); on(k); }; d.append(b); }); return d; };
// a simple orbit viewer for 3-D polylines and points (Cα traces): drag to rotate, wheel to zoom
function orbit3d(cv, sets) {
  const g = cv.getContext("2d"); const all = sets.flatMap((s) => s.pts);
  const c = [0, 1, 2].map((i) => all.reduce((a, p) => a + p[i], 0) / all.length);
  const r = Math.max(...all.map((p) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]))) || 1;
  const st = { yaw: 0.6, pitch: 0.3, zoom: 1, auto: true }; let drag = null, raf = 0;
  cv.onpointerdown = (e) => { drag = [e.clientX, e.clientY, st.yaw, st.pitch]; st.auto = false; cv.setPointerCapture(e.pointerId); };
  cv.onpointermove = (e) => { if (drag) { st.yaw = drag[2] + (e.clientX - drag[0]) * 0.01; st.pitch = Math.max(-1.5, Math.min(1.5, drag[3] + (e.clientY - drag[1]) * 0.01)); } };
  cv.onpointerup = () => (drag = null);
  cv.onwheel = (e) => { e.preventDefault(); st.zoom = Math.max(0.4, Math.min(4, st.zoom * (1 - e.deltaY * 0.001))); };
  function draw() {
    if (!cv.isConnected) return;
    if (st.auto) st.yaw += 0.004;
    const W = cv.width, H = cv.height, s = Math.min(W, H) * 0.42 / r * st.zoom;
    const [cy, sy, cp, sp] = [Math.cos(st.yaw), Math.sin(st.yaw), Math.cos(st.pitch), Math.sin(st.pitch)];
    const P = (p) => { const x = p[0] - c[0], y = p[1] - c[1], z = p[2] - c[2]; const x1 = cy * x - sy * z, z1 = sy * x + cy * z; const y2 = cp * y - sp * z1, z2 = sp * y + cp * z1; return [W / 2 + x1 * s, H / 2 - y2 * s, z2]; };
    g.clearRect(0, 0, W, H);
    for (const set of sets) {
      const q = set.pts.map(P);
      g.lineWidth = set.width || 2.5; g.lineCap = "round"; g.lineJoin = "round";
      for (let i = 1; i < q.length; i++) { g.strokeStyle = set.colors ? set.colors[i] : set.color; g.globalAlpha = set.alpha ?? (0.55 + 0.45 * (1 - (q[i][2] / r + 1) / 2)); g.beginPath(); g.moveTo(q[i - 1][0], q[i - 1][1]); g.lineTo(q[i][0], q[i][1]); g.stroke(); }
    }
    g.globalAlpha = 1; raf = requestAnimationFrame(draw);
  }
  draw();
  return { stop: () => cancelAnimationFrame(raf) };
}

/* ======================================================= more words (0.12) */
Object.assign(I18N.en, {
  cheat_h: "What you can write", kind_found: "Recognised as", series: "Series", phase: "Phase portrait", vs: "against", final_values: "Final values",
  steps_n: "steps", rejected_n: "rejected", evals_n: "evaluations", stiff_switch: "Stiffness detected", event_at: "Event at", method: "Method",
  snapshot: "time", profile: "profile", field: "field", verification: "Verification (manufactured solution)", level: "grid", error_max: "max error", order: "observed order",
  residual: "residual", iterations_n: "iterations", roots: "Roots", unknown_s: "unknowns", params: "Parameters", data_fit: "Data and fit", residuals: "Residuals",
  r2: "R²", rmse: "RMSE", aic: "AIC", optimum: "Optimum", kkt: "KKT conditions", stationarity: "stationarity", infeasibility: "infeasibility", complementarity: "complementarity",
  multipliers: "multipliers", bounds: "bounds", history_h: "Progress", bands: "Percentile bands (5 %, 50 %, 95 %)", members: "members", native_ms: "native", speedup: "speed-up vs one f64 run each",
  parity: "oracle parity (bit for bit)", f64: "agreement with binary64", uncertain: "uncertain", worksheet_h: "Worksheet", quantity: "quantity", value_h: "value", dimension: "dimension",
  name: "name", expr: "expression", cert_ok: "certificate holds", cert_bad: "certificate fails",
  nodes_h: "Node voltages", branches_h: "Branch currents and power", kcl: "KCL residual (max)", power_bal: "power balance", warnings: "Warnings", bode: "Frequency response", magnitude: "magnitude (dB)", phase_deg: "phase (°)",
  transient: "Transient", time_s: "time (s)", node: "node", buses_h: "Buses", lines_h: "Lines", mismatch: "max mismatch (pu)", balance: "balance (MW)", losses: "losses",
  oneline: "One-line diagram", frame_h: "Frame (deformed ×", reactions_h: "Reactions", members_h: "Members", diagram: "Diagram", axial: "axial", shear: "shear", moment: "moment",
  modes_h: "Natural modes", animate: "animate", equilibrium: "equilibrium residual (relative)", mesh_h: "Mesh, coloured by von Mises stress (deformed ×", max_disp: "max displacement", max_vm: "max von Mises",
  dofs: "degrees of freedom", bandwidth: "bandwidth (after RCM)", pipes_h: "Pipes", junctions_h: "Junctions", network: "Network", continuity: "continuity (max, m³/s)", loop_energy: "loop energy (max, m)",
  species: "Species", invariants: "Conserved quantities (from the stoichiometry alone)", drift: "drift", combination: "combination", initial: "initial", odes_h: "Rate equations",
  phase_h: "Phase", vapour_fraction: "vapour fraction V/F", bubble: "bubble pressure", dew: "dew pressure", component: "component", stages: "theoretical stages", feed_stage: "feed stage",
  rmin: "minimum reflux", reflux: "reflux", fenske: "Fenske (total reflux)", gilliland: "Gilliland estimate", mccabe: "McCabe–Thiele", equilibrium_curve: "equilibrium", operating: "operating lines",
  logic_verdict: "Verdict", model_checked: "model checked by evaluation", drup_ok: (n, c) => `DRUP refutation checked: ${n} lemmas, ${c} in the core`, drup_bad: "the refutation did NOT check",
  counterexample: "Counterexample", rules: "Convergent rewriting system", normal_forms: "Normal forms", derivation: "Derivation", basis: "Gröbner basis", remainder: "remainder",
  below: (n) => `a witness at n = ${n}, checked`, refuted_at: (n) => `refuted at n = ${n}`, proof_head: "First lemmas of the refutation", mcp_note: "A language model can propose to this desk through the MCP tool logic_check; the checker, not the proposer, decides.",
  history_moves: "Moves", fen: "FEN", load: "Load", status: "Status", eval: "evaluation", engine_depth: "engine depth", engine_on: "the engine replies", promote: "Promote to", mate_none: (n) => `no forced mate in ${n}`,
  mate_proof: "Proof tree (every defence answered)", proof_checked: "replayed by the independent checker", nodes_n: "nodes", hands: "In hand", drop_hint: "click a piece in hand, then a square",
  size: "size", komi: "komi", black_s: "Black", white_s: "White", margin: "margin", game_over: "game over (two passes)", sims_n: "simulations", engine_value: "engine's value",
  gravity: "gravity", m_cols: "columns", n_rows: "rows", k_line: "in a row", exact: "solved exactly", by_mcts: "by Monte Carlo tree search", to_play: "to play",
  kuhn: "Kuhn", leduc: "Leduc", curve: "Exploitability by iteration", uniform: "uniform random play", infoset: "information set",
  upload_pdb: "Open a PDB file…", analyse_h: "Structure", sequence: "Sequence", secondary_h: "Secondary structure", contact_map: "Contact map", upper_true: "upper: true contacts · lower: predicted (green right, red wrong)",
  seqs: "sequences", seed: "seed", model_pdb: "Download the model (PDB)", superposition: "Superposition (model red, native green)", identity: "identity", mode: "mode", global: "global", local: "local",
  model_s: "model", native_s: "native", mi: "mutual information", dca: "direct coupling (DCA)", chance: "chance", mirrored: "mirror image chosen by helix handedness",
  play: "Play", pause: "Pause", resolution: "resolution", download_png: "Download PNG", gpu_mean: "GPU mean radiance", ref_mean: "reference mean radiance", agree: "agreement",
  furnace_uniform: "uniform furnace (exact a·L)", furnace_gradient: "gradient furnace a(½ + n_y/3)", furnace_control: "biased estimator (the control)", mean_err: "mean error", orbit_hint: "drag to orbit the camera · wheel to dolly · the text follows",
  errors_h: "Errors", scene_h: "Scene", spp_n: (n) => `${n} samples/pixel`,
});
Object.assign(I18N.pt, {
  cheat_h: "O que se pode escrever", kind_found: "Reconhecido como", series: "Séries", phase: "Retrato de fase", vs: "contra", final_values: "Valores finais",
  steps_n: "passos", rejected_n: "rejeitados", evals_n: "avaliações", stiff_switch: "Rigidez detectada", event_at: "Evento em", method: "Método",
  snapshot: "tempo", profile: "perfil", field: "campo", verification: "Verificação (solução manufaturada)", level: "malha", error_max: "erro máximo", order: "ordem observada",
  residual: "resíduo", iterations_n: "iterações", roots: "Raízes", unknown_s: "incógnitas", params: "Parâmetros", data_fit: "Dados e ajuste", residuals: "Resíduos",
  r2: "R²", rmse: "RMSE", aic: "AIC", optimum: "Ótimo", kkt: "Condições KKT", stationarity: "estacionariedade", infeasibility: "inviabilidade", complementarity: "complementaridade",
  multipliers: "multiplicadores", bounds: "limites", history_h: "Progresso", bands: "Faixas de percentis (5 %, 50 %, 95 %)", members: "membros", native_ms: "nativo", speedup: "aceleração vs uma execução f64 cada",
  parity: "paridade com o oráculo (bit a bit)", f64: "concordância com binary64", uncertain: "incertos", worksheet_h: "Planilha", quantity: "grandeza", value_h: "valor", dimension: "dimensão",
  name: "nome", expr: "expressão", cert_ok: "certificado confere", cert_bad: "certificado falha",
  nodes_h: "Tensões nodais", branches_h: "Correntes e potências dos ramos", kcl: "resíduo de Kirchhoff (máx.)", power_bal: "balanço de potência", warnings: "Avisos", bode: "Resposta em frequência", magnitude: "módulo (dB)", phase_deg: "fase (°)",
  transient: "Transitório", time_s: "tempo (s)", node: "nó", buses_h: "Barras", lines_h: "Linhas", mismatch: "desbalanço máx. (pu)", balance: "balanço (MW)", losses: "perdas",
  oneline: "Diagrama unifilar", frame_h: "Estrutura (deformada ×", reactions_h: "Reações", members_h: "Barras", diagram: "Diagrama", axial: "normal", shear: "cortante", moment: "momento",
  modes_h: "Modos naturais", animate: "animar", equilibrium: "resíduo de equilíbrio (relativo)", mesh_h: "Malha, colorida pela tensão de von Mises (deformada ×", max_disp: "deslocamento máx.", max_vm: "von Mises máx.",
  dofs: "graus de liberdade", bandwidth: "largura de banda (após RCM)", pipes_h: "Tubos", junctions_h: "Nós", network: "Rede", continuity: "continuidade (máx., m³/s)", loop_energy: "energia nas malhas (máx., m)",
  species: "Espécies", invariants: "Grandezas conservadas (só pela estequiometria)", drift: "deriva", combination: "combinação", initial: "inicial", odes_h: "Equações de taxa",
  phase_h: "Fase", vapour_fraction: "fração vaporizada V/F", bubble: "pressão de bolha", dew: "pressão de orvalho", component: "componente", stages: "estágios teóricos", feed_stage: "estágio de alimentação",
  rmin: "refluxo mínimo", reflux: "refluxo", fenske: "Fenske (refluxo total)", gilliland: "estimativa de Gilliland", mccabe: "McCabe–Thiele", equilibrium_curve: "equilíbrio", operating: "retas de operação",
  logic_verdict: "Veredito", model_checked: "modelo conferido por avaliação", drup_ok: (n, c) => `refutação DRUP conferida: ${n} lemas, ${c} no núcleo`, drup_bad: "a refutação NÃO conferiu",
  counterexample: "Contraexemplo", rules: "Sistema de reescrita convergente", normal_forms: "Formas normais", derivation: "Derivação", basis: "Base de Gröbner", remainder: "resto",
  below: (n) => `testemunha em n = ${n}, conferida`, refuted_at: (n) => `refutado em n = ${n}`, proof_head: "Primeiros lemas da refutação", mcp_note: "Um modelo de linguagem pode propor a esta mesa pela ferramenta MCP logic_check; quem decide é o verificador, não o proponente.",
  history_moves: "Lances", fen: "FEN", load: "Carregar", status: "Situação", eval: "avaliação", engine_depth: "profundidade do motor", engine_on: "o motor responde", promote: "Promover a", mate_none: (n) => `sem mate forçado em ${n}`,
  mate_proof: "Árvore de prova (toda defesa respondida)", proof_checked: "reproduzida pelo verificador independente", nodes_n: "nós", hands: "Na mão", drop_hint: "clique numa peça da mão e depois numa casa",
  size: "tamanho", komi: "komi", black_s: "Pretas", white_s: "Brancas", margin: "margem", game_over: "fim de jogo (dois passes)", sims_n: "simulações", engine_value: "valor do motor",
  gravity: "gravidade", m_cols: "colunas", n_rows: "linhas", k_line: "em linha", exact: "resolvido exatamente", by_mcts: "por busca em árvore Monte Carlo", to_play: "a jogar",
  kuhn: "Kuhn", leduc: "Leduc", curve: "Explorabilidade por iteração", uniform: "jogo uniforme aleatório", infoset: "conjunto de informação",
  upload_pdb: "Abrir um arquivo PDB…", analyse_h: "Estrutura", sequence: "Sequência", secondary_h: "Estrutura secundária", contact_map: "Mapa de contatos", upper_true: "acima: contatos verdadeiros · abaixo: previstos (verde certo, vermelho errado)",
  seqs: "sequências", seed: "semente", model_pdb: "Baixar o modelo (PDB)", superposition: "Superposição (modelo vermelho, nativa verde)", identity: "identidade", mode: "modo", global: "global", local: "local",
  model_s: "modelo", native_s: "nativa", mi: "informação mútua", dca: "acoplamento direto (DCA)", chance: "acaso", mirrored: "imagem especular escolhida pela quiralidade das hélices",
  play: "Rodar", pause: "Pausar", resolution: "resolução", download_png: "Baixar PNG", gpu_mean: "radiância média na GPU", ref_mean: "radiância média da referência", agree: "concordância",
  furnace_uniform: "fornalha uniforme (exato a·L)", furnace_gradient: "fornalha em gradiente a(½ + n_y/3)", furnace_control: "estimador viciado (o controle)", mean_err: "erro médio", orbit_hint: "arraste para orbitar a câmera · roda para aproximar · o texto acompanha",
  errors_h: "Erros", scene_h: "Cena", spp_n: (n) => `${n} amostras/pixel`,
});

/* ================================================================ panels */
// each panel is built from its state, so a change of language rebuilds it in place (the text, the game, the result survive)
function mount(id, build) {
  const root = $("wb-" + id), state = {};
  const draw = () => { root.replaceChildren(); build(root, state, draw); };
  shown.set("p-" + id, () => { draw(); rerender.push(draw); });
}
const fmtU = (v, u) => (u ? `${wbNum(v)} ${u}` : wbNum(v));
const ctrl = (label, input) => el("label", { class: "ctl" }, el("span", { text: label }), input);
const numIn = (v, { min, max, step = 1, w = 5 } = {}) => { const i = el("input", { type: "number", value: String(v), min: String(min ?? ""), max: String(max ?? ""), step: String(step) }); i.style.width = w + "em"; return i; };
const sel = (opts, v) => { const s = el("select", {}, ...opts.map(([k, lab]) => el("option", { value: String(k), text: lab }))); s.value = String(v); return s; };
const btn = (text, cls = "quiet", on) => { const b = el("button", { type: "button", class: cls, text }); if (on) b.onclick = on; return b; };
const sect = (title, ...kids) => el("section", { class: "wb-sec" }, el("h3", { text: title }), ...kids);
function errBox(out, e) { out.replaceChildren(notice(e.message || String(e))); }
// a pair of numbers that keeps the order of magnitude honest: a chip "verified" when below the bound
function certRow(label, v, bound) { return el("div", { class: "cert-row" }, chipV(v != null && Math.abs(v) <= bound, v != null && Math.abs(v) <= bound ? "✓" : "✗"), el("span", { text: label }), el("b", { text: wbNum(v, 2) })); }
// a toolbar: the example list, the main action, a status line
function deskBar(examples, labels, onPick, actions) {
  const state = el("span", { class: "muted wb-state", "aria-live": "polite" });
  return { bar: el("div", { class: "wb-bar" }, exampleSelect(examples, labels, onPick), ...actions, state), state };
}
const palColors = ["var(--lock)", "var(--ember)", "var(--sand)", "var(--silt)", "#7A5BA6", "#3B7FC4", "#C25B91", "#5E8C31"];
function multiChart(t, series, names, { logy = false, xlab = "", ylab = "", w = 760, h = 260 } = {}) {
  return lineChart({ series: names.map((n, i) => ({ points: t.map((x, k) => [x, series[n][k]]), cls: i < 4 ? i : 3, color: i < 4 ? null : palColors[i % palColors.length] })), logy, xlab, ylab, w, h });
}
function legendOf(names) { const d = el("div", { class: "legend" }); names.forEach((n, i) => { const sw = el("i", { class: i < 4 ? `l${i}` : "" }); if (i >= 4) sw.style.background = palColors[i % palColors.length]; d.append(el("span", {}, sw, n)); }); return d; }
// checkboxes to choose which series to draw
function seriesPicker(names, chosen, on) {
  const d = el("div", { class: "picker" });
  names.forEach((n) => { const c = el("input", { type: "checkbox" }); c.checked = chosen.has(n); c.onchange = () => { c.checked ? chosen.add(n) : chosen.delete(n); on(); }; d.append(el("label", {}, c, " " + n)); });
  return d;
}

/* ------------------------------------------------------------ workbench */
const EX_BENCH = {
  osc: { text: "# a damped oscillator: units are checked before anything runs\nx' = v\nv' = -k/m*x - c/m*v\nk = 4[N/m]; m = 1[kg]; c = 0.4[N*s/m]\nx(0) = 1[m]; v(0) = 0[m/s]\nt = 0 .. 20[s]\nE := 0.5*m*v^2 + 0.5*k*x^2" },
  lorenz: { text: "x' = s*(y - x)\ny' = x*(r - z) - y\nz' = x*y - b*z\ns = 10; r = 28; b = 8/3\nx(0) = 1; y(0) = 1; z(0) = 1\nt = 0 .. 40\nrtol = 1e-9\natol = 1e-12\nsamples = 4000", phase: ["x", "z"] },
  robertson: { text: "# stiff: the solver notices and switches to Rosenbrock\na' = -0.04*a + 1e4*b*c\nb' = 0.04*a - 1e4*b*c - 3e7*b^2\nc' = 3e7*b^2\na(0) = 1; b(0) = 0; c(0) = 0\nt = 0 .. 40\nrtol = 1e-6\natol = 1e-10", logy: true },
  ball: { text: "# quadratic drag; the event is located on the dense output\ny' = v\nv' = -g - c*v*abs(v)\ng = 9.81[m/s^2]; c = 0.02[1/m]\ny(0) = 0[m]; v(0) = 20[m/s]\nt = 0 .. 10[s]\nstop when y < 0" },
  sir: { text: "S' = -b*S*I/N\nI' = b*S*I/N - g*I\nR' = g*I\nb = 0.3; g = 0.1; N = 1000\nS(0) = 999; I(0) = 1; R(0) = 0\nt = 0 .. 160" },
  heat: { text: "u_t = D*u_xx\nD = 0.1\nx = 0 .. 1\nt = 0 .. 0.5\nu(0, t) = 0\nu(1, t) = 0\nnx = 21; nt = 20\nexact u = exp(-D*pi^2*t)*sin(pi*x)\nverify" },
  fisher: { text: "# a travelling front: diffusion + logistic growth\nu_t = D*u_xx + r*u*(1 - u)\nD = 0.01; r = 1\nx = 0 .. 10\nt = 0 .. 20\nu(x, 0) = exp(-10*x^2)\nu_x(0, t) = 0\nu_x(10, t) = 0\nnx = 201; nt = 400" },
  consol: { text: "# excess pore pressure (kPa) in a clay layer drained at the top, z in m, t in years\nu_t = cv*u_xx\ncv = 1\nx = 0 .. 5\nt = 0 .. 10\nu(x, 0) = 100\nu(0, t) = 0\nu_x(5, t) = 0\nnx = 51; nt = 200" },
  wave: { text: "u_tt = c^2*u_xx\nc = 1\nx = 0 .. 1\nt = 0 .. 2\nu(x, 0) = min(x/0.3, (1 - x)/0.7)\nu_t(x, 0) = 0\nu(0, t) = 0\nu(1, t) = 0\nnx = 101; nt = 400" },
  poisson: { text: "poisson -(u_xx + u_yy) = f\nf = 1\nx = 0 .. 1; y = 0 .. 1\nnx = 21; ny = 21\nexact u = sin(pi*x)*sinh(pi*y)/sinh(pi) + x*y*(1-x)\nverify" },
  sheet: { text: "F = 3[kN]\nL = 2.5[m]\nE = 200[GPa]\nI = 8.33e-6[m^4]\nM = F*L in [kN*m]\nd = F*L^3/(3*E*I) in [mm]\nk = 3*E*I/L^3 in [kN/mm]\nf = sqrt(k/120[kg])/(2*pi) in [Hz]\np = 101325[Pa] in [psi]" },
  circle: { text: "unknowns x = 2, y = 0.5\nx^2 + y^2 = 4\nx*y = 1\nsearch = [-3, 3]" },
  fit: { text: "fit y = a*exp(-b*x) + c\na = 1; b = 0.5\ndata\nx, y\n" + Array.from({ length: 31 }, (_, i) => { const x = i * 0.15; return `${x.toFixed(2)}, ${(3 * Math.exp(-1.3 * x) + 0.5 + 0.03 * Math.sin(17 * i)).toFixed(5)}`; }).join("\n") },
  rosen: { text: "minimize (1-x)^2 + 100*(y-x^2)^2\nfrom x = -1.2, y = 1" },
  cons: { text: "# the can of least surface for one litre (dm): r and h bounded below\nminimize 2*pi*r^2 + 2*pi*r*h\nfrom r = 1, h = 1\nsubject to pi*r^2*h = 1, r >= 0.01, h >= 0.01" },
  ens: { text: "x' = v\nv' = -k/m*x - c/m*v\nk ~ normal(4, 0.2); c ~ uniform(0.05, 0.2); m = 1\nx(0) ~ normal(1, 0.05); v(0) = 0\nt = 0 .. 10\nmembers = 4096; h = 0.01", ens: true },
};
const CHEAT = [["x' = …  ·  x(0) = …  ·  t = 0 .. 10[s]", "ode"], ["stop when y < 0  ·  E := …", "ode+"], ["u_t = …  ·  u_tt = …  ·  poisson -(u_xx + u_yy) = f", "pde"],
  ["verify  ·  exact u = …", "mms"], ["unknowns x = 1, y = 2  ·  search = [-3, 3]", "nl"], ["fit y = a*exp(-b*x)  ·  data", "fit"], ["minimize …  ·  subject to …  ·  r >= 0", "min"],
  ["3[kN] · 2.5[m] · … in [mm]", "units"], ["k ~ normal(4, 0.2)  ·  members = 4096", "ens"]];
Object.assign(I18N.en, { cheat: { ode: "a system of ODEs, with units", "ode+": "events and derived outputs", pde: "parabolic, hyperbolic, elliptic", mms: "verification by a manufactured solution", nl: "every root in a box", fit: "nonlinear least squares", min: "constraints and bounds", units: "quantities and conversions", ens: "uncertainty, on the native worker" } });
Object.assign(I18N.pt, { cheat: { ode: "um sistema de EDOs, com unidades", "ode+": "eventos e saídas derivadas", pde: "parabólica, hiperbólica, elíptica", mms: "verificação por solução manufaturada", nl: "todas as raízes numa caixa", fit: "mínimos quadrados não lineares", min: "restrições e limites", units: "grandezas e conversões", ens: "incerteza, no worker nativo" } });

mount("bench", (root, S) => {
  head(root, "bench", "bench_lede");
  const ed = editor("wb-bench-ed", 13); ed.value = S.text ?? EX_BENCH.osc.text; ed.oninput = () => (S.text = ed.value);
  const out = el("div", { class: "wb-out", "aria-live": "polite" }), side = el("aside", { class: "wb-side" });
  const go = async (ensemble) => {
    S.text = ed.value; S.ens = ensemble; busy(out); bar.state.textContent = t("running");
    try { S.r = await api("/v1/vapor/solve", { text: ed.value, ensemble }); S.err = null; } catch (e) { S.r = null; S.err = e.message; }
    show();
  };
  ed.addEventListener("run", () => go(!!S.ens));
  const bar = deskBar(EX_BENCH, () => t("ex_bench"), (k, ex) => { S.ex = k; ed.value = S.text = ex.text; go(!!ex.ens); },
    [btn(t("solve_btn"), "primary", () => go(false)), btn(t("ensemble_btn"), "quiet", () => go(true))]);
  root.append(bar.bar, el("div", { class: "wb-main" }, ed, side), out);
  const cheat = () => sect(t("cheat_h"), el("dl", { class: "cheat" }, ...CHEAT.flatMap(([c, k]) => [el("dt", {}, el("code", { text: c })), el("dd", { text: t("cheat")[k] })])));
  function show() {
    side.replaceChildren(); out.replaceChildren();
    if (S.err) { bar.state.textContent = ""; side.append(cheat()); out.append(notice(S.err)); return; }
    if (!S.r) { side.append(cheat()); return; }
    const r = S.r; bar.state.textContent = t("ms", r.ms);
    side.append(el("p", { class: "muted", text: t("kind_found") }), el("p", { class: "big-kind", text: r.uncertain ? t("ex_bench").ens : t("kinds")[r.kind] || r.kind }));
    (BENCH_VIEW[r.uncertain ? "ensemble" : r.kind] || (() => {}))(r, side, out, S);
  }
  show();
  if (!S.r && !S.err && !S.ran) { S.ran = true; go(false); }
});

const BENCH_VIEW = {
  worksheet(r, side, out) {
    const ok = r.lines.filter((l) => !l.error).length;
    side.append(stats(stat(t("quantity"), `${ok}/${r.lines.length}`)));
    out.append(sect(t("worksheet_h"), table([t("name"), t("expr"), t("value_h"), t("dimension")], r.lines.map((l) => l.error
      ? [l.name || "", l.text, el("span", { class: "err-t", text: l.error }), ""]
      : [l.name || "", l.text, el("b", { text: fmtU(l.shown, l.unit) }), l.dimension || "—"]))));
  },
  ode(r, side, out, S) {
    const names = [...r.states, ...(r.outputs || [])];
    side.append(stats(stat(t("steps_n"), String(r.steps)), stat(t("rejected_n"), String(r.rejected)), stat(t("evals_n"), String(r.evals))),
      facts([[t("method"), r.method], ["rtol / atol", `${wbNum(r.rtol)} / ${wbNum(r.atol)}`], ["t", fmtU(r.t_end, r.time_unit)]]));
    if (r.switched) side.append(el("p", { class: "notice" }, el("b", { text: t("stiff_switch") + ": " }), r.switched));
    if (r.event) side.append(el("p", { class: "notice ok" }, el("b", { text: t("event_at") + " t = " }), fmtU(r.event.t, r.time_unit)));
    const ex = EX_BENCH[S.ex] || {};
    const chosen = S.chosen && names.some((n) => S.chosen.has(n)) ? S.chosen : (S.chosen = new Set(names.slice(0, 6)));
    const chartBox = el("div");
    let logy = !!ex.logy && S.ex && EX_BENCH[S.ex].text === S.text;
    const draw = () => { const ns = names.filter((n) => chosen.has(n)); const pos = ns.every((n) => r.series[n].every((v) => v > 0)); chartBox.replaceChildren(legendOf(ns), multiChart(r.t, r.series, ns, { logy: logy && pos, xlab: `t${r.time_unit ? " (" + r.time_unit + ")" : ""}` })); };
    const lg = el("input", { type: "checkbox" }); lg.checked = logy; lg.onchange = () => { logy = lg.checked; draw(); };
    out.append(sect(t("series"), el("div", { class: "row" }, seriesPicker(names, chosen, draw), el("label", { class: "muted" }, lg, " log y")), chartBox));
    draw();
    if (names.length >= 2) {
      const [a0, b0] = ex.phase && S.text === ex.text ? ex.phase : names;
      const sa = sel(names.map((n) => [n, n]), a0), sb = sel(names.map((n) => [n, n]), b0), box = el("div");
      const dr = () => box.replaceChildren(lineChart({ series: [{ points: r.series[sa.value].map((v, k) => [v, r.series[sb.value][k]]), cls: 0 }], xlab: sa.value, ylab: sb.value, w: 520, h: 360 }));
      sa.onchange = sb.onchange = dr; dr();
      out.append(sect(t("phase"), el("div", { class: "row" }, sb, el("span", { class: "muted", text: t("vs") }), sa), box));
    }
    side.append(sect(t("final_values"), facts(Object.entries(r.final).map(([k, v]) => [k, fmtU(v, r.units?.[k])]))));
  },
  pde(r, side, out) {
    side.append(stats(stat("nx", String(r.nx)), r.nt ? stat("nt", String(r.nt)) : stat("ny", String(r.ny))), facts([[t("method"), r.scheme], ...(r.cfl != null ? [["CFL", r.cfl]] : []),
      ...(r.residual != null ? [[t("residual"), r.residual], [t("iterations_n"), r.iterations]] : []), ...(r.error != null ? [[t("error_max"), r.error]] : [])]));
    if (r.verification) {
      const v = r.verification, good = /verified/.test(v.verdict);
      side.append(sect(t("verification"), el("p", {}, chipV(good, good ? t("verified") : t("not_verified")), " ", el("span", { class: "muted", text: v.verdict })),
        table([t("level"), t("error_max"), t("order")], v.levels.map((l, i) => [l.nt ? `${l.nx} × ${l.nt}` : `${l.nx} × ${l.ny}`, wbNum(l.error, 3), i ? wbNum(v.orders[i - 1], 3) : "—"]))));
    }
    if (r.family === "elliptic") { out.append(sect(t("field") + " u(x, y)", heatmap(r.grid.map((row) => row.slice()), { w: 420, h: 420, flipY: true }))); return; }
    const k = el("input", { type: "range", min: "0", max: String(r.u.length - 1), value: String(r.u.length - 1) }), lab = el("span", { class: "muted" }), box = el("div");
    const lo = Math.min(...r.u.flat()), hi = Math.max(...r.u.flat());
    const dr = () => { const i = +k.value; lab.textContent = `t = ${wbNum(r.t[i])}`; box.replaceChildren(lineChart({ series: [{ points: r.x.map((x, j) => [x, r.u[i][j]]), cls: 0 }, { points: r.x.map((x, j) => [x, r.u[0][j]]), cls: 3 }], ymin: lo, ymax: hi, xlab: "x", w: 760, h: 240 })); };
    k.oninput = dr; dr();
    out.append(sect(t("profile"), el("div", { class: "row" }, el("span", { class: "muted", text: t("snapshot") }), k, lab), box),
      sect(t("field") + " u(x, t)", heatmap(r.u, { w: 760, h: 260, flipY: true }), el("p", { class: "muted", text: "x →  ·  t ↑" })));
  },
  nonlinear(r, side, out) {
    side.append(stats(stat(t("roots"), String(r.roots.length)), stat("starts", String(r.starts))), facts(r.equations.map((e, i) => [`#${i + 1}`, e])));
    out.append(sect(t("roots"), table([...r.unknowns, t("residual"), t("iterations_n")], r.roots.map((x) => [...r.unknowns.map((u) => wbNum(x.values[u], 10)), wbNum(x.residual, 2), x.iterations]))));
    if (r.unknowns.length === 2) { const [a, b] = r.unknowns; out.append(lineChart({ series: [{ points: r.roots.map((x) => [x.values[a], x.values[b]]), dots: true, cls: 1 }], xlab: a, ylab: b, w: 420, h: 360 })); }
  },
  fit(r, side, out) {
    side.append(stats(stat(t("r2"), wbNum(r.r2, 6)), stat(t("rmse"), wbNum(r.rmse, 3)), stat(t("aic"), wbNum(r.aic, 4))), el("p", { class: "muted" }, chipV(r.converged, r.converged ? "converged" : "not converged"), ` · ${r.iterations} ${t("iterations_n")} · dof ${r.dof}`));
    out.append(sect(t("params"), table([t("name"), t("value_h"), "± σ", "σ / |value|"], r.order.map((k) => [k, wbNum(r.params[k].value, 8), wbNum(r.params[k].stderr, 3), wbNum(Math.abs(r.params[k].stderr / r.params[k].value), 2)]))));
    if (r.inputs.length === 1) {
      const xs = r.data.x.map((v) => (Array.isArray(v) ? v[0] : v)), idx = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
      out.append(sect(t("data_fit"), el("p", { class: "muted", text: r.model }), legend([[1, "data"], [0, "fit"]]),
        lineChart({ series: [{ points: idx.map((i) => [xs[i], r.fitted[i]]), cls: 0 }, { points: xs.map((x, i) => [x, r.data.y[i]]), dots: true, cls: 1 }], xlab: r.inputs[0], ylab: r.response, w: 760, h: 260 }),
        el("h3", { text: t("residuals") }), lineChart({ series: [{ points: idx.map((i) => [xs[i], r.residuals[i]]), bars: true, cls: 2 }], levels: [{ y: 0 }], w: 760, h: 140 })));
    }
  },
  minimize(r, side, out) {
    side.append(el("p", {}, chipV(r.converged, r.converged ? t("verified") : t("not_verified")), " ", el("span", { class: "muted", text: r.verdict })),
      stats(stat(t("optimum"), wbNum(r.value, 8)), stat(t("iterations_n"), String(r.iterations))),
      sect(t("kkt"), certRow(t("stationarity"), r.kkt.stationarity, 1e-5), certRow(t("infeasibility"), r.kkt.infeasibility, 1e-6), certRow(t("complementarity"), r.kkt.complementarity, 1e-6)));
    out.append(sect(t("optimum"), table([t("name"), t("value_h")], r.vars.map((v) => [v, wbNum(r.x[v], 10)]))));
    if (r.constraints.length || r.bounds.length) out.append(sect(t("multipliers"), table(["", t("multipliers")], [...r.constraints.map((c, i) => [c, wbNum(r.multipliers[i], 6)]), ...r.bounds.map((b) => [b, t("bounds")])])));
    if (r.history.length > 1) out.append(sect(t("history_h"), legend([[0, t("infeasibility")]]), lineChart({ series: [{ points: r.history.map((h) => [h.outer, Math.max(h.infeasibility, 1e-16)]), dots: true, cls: 0 }], logy: true, xlab: "outer", w: 520, h: 180 })));
  },
  ensemble(r, side, out) {
    side.append(el("p", {}, okChip(r.oracle_parity, t("parity"))),
      stats(stat(t("members"), String(r.members)), stat(t("native_ms"), t("ms", wbNum(r.native_ms, 3))), stat(t("speedup"), `${wbNum(r.speedup, 3)}×`)),
      facts([[t("f64"), `max ${wbNum(r.f64_agreement.max_relative, 2)} · median ${wbNum(r.f64_agreement.median_relative, 2)}`], ["h", r.h], [t("steps_n"), r.steps], [t("uncertain"), r.uncertain.join(", ")], ["substrate", r.substrate]]));
    r.states.forEach((s) => {
      const b = r.bands[s];
      out.append(sect(`${s}(t) — ${t("bands")}`, legend([[0, "p50"], [2, "p05 / p95"], [3, "min / max"]]), lineChart({ series: [
        { points: r.t.map((x, i) => [x, b[i].p50]), cls: 0 }, { points: r.t.map((x, i) => [x, b[i].p05]), cls: 2 }, { points: r.t.map((x, i) => [x, b[i].p95]), cls: 2 },
        { points: r.t.map((x, i) => [x, b[i].min]), cls: 3 }, { points: r.t.map((x, i) => [x, b[i].max]), cls: 3 }], xlab: "t", w: 760, h: 220 })));
    });
  },
};
Object.keys(EX_BENCH).forEach((k) => PALETTE_ITEMS.push(() => ({ label: t("ex_bench")[k], sub: t("bench"), run: () => { openPanel("bench"); const s = $("wb-bench").querySelector(".wb-bar select"); if (s) { s.value = k; s.dispatchEvent(new Event("change")); } } })));

/* ---------------------------------------------------------- engineering */
const EX_ENG = {
  circuit: {
    c_diode: "* a diode biased through a divider: the operating point and its certificate\nV1 in 0 DC 5\nR1 in a 1k\nR2 a 0 2k\nR3 a d 1k\nD1 d 0 IS=1e-14\n.op",
    c_rlc: "* series RLC: resonance at 1/2π√LC\nV1 in 0 AC 1\nR1 in a 10\nL1 a b 1m\nC1 b 0 1u\n.ac dec 40 100 100k",
    c_rc: "* RC charging and discharging: a pulse through 1 kΩ into 1 µF\nV1 in 0 PULSE(0 1 0 1u 1u 2m 4m)\nR1 in out 1k\nC1 out 0 1u\n.tran 10u 8m",
    c_amp: "* an inverting amplifier with an ideal op-amp: gain −R2/R1\nV1 in 0 DC 1\nR1 in m 1k\nR2 m out 10k\nO1 out 0 m\n.op",
  },
  power: {
    p_stagg: "# Stagg & El-Abiad's five-bus system (base 100 MVA)\nbase 100\nbus 1 slack V=1.06\nbus 2 pq P=20 Q=20\nbus 3 pq P=-45 Q=-15\nbus 4 pq P=-40 Q=-5\nbus 5 pq P=-60 Q=-10\nline 1 2 r=0.02 x=0.06 b=0.06\nline 1 3 r=0.08 x=0.24 b=0.05\nline 2 3 r=0.06 x=0.18 b=0.04\nline 2 4 r=0.06 x=0.18 b=0.04\nline 2 5 r=0.04 x=0.12 b=0.03\nline 3 4 r=0.01 x=0.03 b=0.02\nline 4 5 r=0.08 x=0.24 b=0.05",
    p_pv: "# a generator holding its voltage (PV) and a capacitor bank\nbase 100\nbus 1 slack V=1.06\nbus 2 pv P=40 V=1.045\nbus 3 pq P=-45 Q=-15\nbus 4 pq P=-60 Q=-20\nline 1 2 r=0.02 x=0.06 b=0.06\nline 2 3 r=0.06 x=0.18 b=0.04\nline 1 4 r=0.05 x=0.2 b=0.04\nline 3 4 r=0.01 x=0.03 b=0.02\nshunt 4 b=0.15",
  },
  structure: {
    s_portal: "# a portal frame: wind on the column top, a uniform load on the beam\nnode 1 0 0\nnode 2 0 4[m]\nnode 3 6[m] 4[m]\nnode 4 6[m] 0\nsupport 1 fixed\nsupport 4 fixed\nbeam 1 2 E=210[GPa] A=5.38e-3 I=8.36e-5 n=6\nbeam 2 3 E=210[GPa] A=5.38e-3 I=8.36e-5 n=8\nbeam 3 4 E=210[GPa] A=5.38e-3 I=8.36e-5 n=6\nload 2 fx=15[kN]\nudl 2 3 w=-12[kN/m]",
    s_cant: "# a steel cantilever: tip load, and its first three natural modes\nnode 1 0 0\nnode 2 2 0\nsupport 1 fixed\nbeam 1 2 E=200e9 A=1e-3 I=1e-7 rho=7850 n=12\nload 2 fy=-1[kN]\nmodes 3",
    s_ss: "# simply supported, uniform load: 5wL⁴/384EI at mid-span\nnode 1 0 0\nnode 2 10 0\nsupport 1 pinned\nsupport 2 roller-x\nbeam 1 2 E=200e9 A=0.01 I=1e-4 n=10\nudl 1 2 w=-5e3",
    s_truss: "# a Warren truss bridge: member forces by the stiffness method\nnode 1 0 0\nnode 2 4 0\nnode 3 8 0\nnode 4 12 0\nnode 5 2 3\nnode 6 6 3\nnode 7 10 3\nsupport 1 pinned\nsupport 4 roller-x\ntruss 1 2 E=200e9 A=2e-3\ntruss 2 3 E=200e9 A=2e-3\ntruss 3 4 E=200e9 A=2e-3\ntruss 5 6 E=200e9 A=2e-3\ntruss 6 7 E=200e9 A=2e-3\ntruss 1 5 E=200e9 A=2e-3\ntruss 5 2 E=200e9 A=2e-3\ntruss 2 6 E=200e9 A=2e-3\ntruss 6 3 E=200e9 A=2e-3\ntruss 3 7 E=200e9 A=2e-3\ntruss 7 4 E=200e9 A=2e-3\nload 2 fy=-50e3\nload 3 fy=-50e3",
  },
  fem: {
    f_cant: "# a cantilever slab in plane stress: QM6 (incompatible modes) does not lock\nplate x=0..10 y=-0.5..0.5 nx=20 ny=4\nmaterial E=1000 nu=0.3 t=1\nelement qm6\nfix x=0\ntraction x=10 ty=-1",
    f_q4: "# the same slab with plain Q4 (the control): shear locking, too stiff\nplate x=0..10 y=-0.5..0.5 nx=20 ny=4\nmaterial E=1000 nu=0.3 t=1\nelement q4\nfix x=0\ntraction x=10 ty=-1",
    f_wall: "# a shear wall: concrete, 3 m × 6 m, lateral load at the top\nplate x=0..3 y=0..6 nx=8 ny=16\nmaterial E=30e9 nu=0.2 t=0.2\nfix y=0\ntraction y=6 tx=50e3",
  },
  pipes: {
    p_loop: "# a looped network fed by one reservoir\nreservoir R head=60[m]\njunction A elev=10[m] demand=15[L/s]\njunction B elev=12[m] demand=20[L/s]\njunction C elev=8[m] demand=10[L/s]\npipe 1 R A L=800[m] D=250[mm] eps=0.1[mm]\npipe 2 A B L=600[m] D=200[mm] eps=0.1[mm] K=2\npipe 3 R B L=1200[m] D=200[mm] eps=0.1[mm]\npipe 4 A C L=500[m] D=150[mm] eps=0.1[mm]\npipe 5 C B L=400[m] D=150[mm] eps=0.1[mm]",
    p_two: "# three reservoirs: which way does the middle one flow?\nreservoir A head=100[m]\nreservoir B head=80[m]\nreservoir C head=60[m]\njunction J elev=50[m] demand=0\npipe 1 A J L=1000[m] D=300[mm] eps=0.26[mm]\npipe 2 B J L=800[m] D=250[mm] eps=0.26[mm]\npipe 3 J C L=1200[m] D=300[mm] eps=0.26[mm]",
  },
  reactions: {
    r_series: "# A → B → C: Bateman's closed form is the check\nA -> B ; k = 1\nB -> C ; k = 0.5\nA0 = 1\nt = 0 .. 10",
    r_net: "# a network: its conserved moieties come from the stoichiometry alone\nA + B -> C ; k = 2\nC <-> D ; kf = 1, kb = 0.2\n2 A -> E ; k = 0.1\nA0 = 1; B0 = 0.8\nt = 0 .. 20",
    r_cstr: "# a CSTR approaching its steady state C₀/(1 + kτ)\nA -> B ; k = 2\nreactor cstr tau=5\nfeed A=1\nt = 0 .. 60",
  },
  flash: { x_bt: "# benzene–toluene at 100 °C and 1 atm (Antoine in mmHg, °C)\ncomponent benzene z=0.4 A=6.90565 B=1211.033 C=220.79\ncomponent toluene z=0.6 A=6.95464 B=1344.8 C=219.482\nT = 100\nP = 760" },
  distill: { d_bt: "# a benzene–toluene column: relative volatility 2.5, saturated liquid feed\nalpha = 2.5; xF = 0.45; xD = 0.95; xB = 0.05; q = 1; Rfactor = 1.3" },
};
Object.assign(I18N.en, { ex_eng: { c_diode: "Diode bias network (operating point)", c_rlc: "Series RLC resonance (AC sweep)", c_rc: "RC pulse response (transient)", c_amp: "Inverting amplifier (ideal op-amp)",
  p_stagg: "Stagg & El-Abiad five-bus system", p_pv: "PV generator and capacitor bank", s_portal: "Portal frame: wind and gravity", s_cant: "Cantilever: deflection and modes", s_ss: "Simply supported beam, uniform load", s_truss: "Warren truss bridge",
  f_cant: "Cantilever slab (QM6)", f_q4: "Cantilever slab with Q4 (control: locking)", f_wall: "Concrete shear wall", p_loop: "Looped water network", p_two: "Three-reservoir problem",
  r_series: "A → B → C (Bateman)", r_net: "Reaction network: invariants", r_cstr: "CSTR start-up", x_bt: "Benzene–toluene flash", d_bt: "Benzene–toluene column" } });
Object.assign(I18N.pt, { ex_eng: { c_diode: "Polarização de diodo (ponto de operação)", c_rlc: "Ressonância RLC série (varredura CA)", c_rc: "Resposta RC a pulso (transitório)", c_amp: "Amplificador inversor (amp-op ideal)",
  p_stagg: "Sistema de cinco barras de Stagg & El-Abiad", p_pv: "Gerador PV e banco de capacitores", s_portal: "Pórtico: vento e gravidade", s_cant: "Balanço: flecha e modos", s_ss: "Viga biapoiada, carga uniforme", s_truss: "Treliça Warren de ponte",
  f_cant: "Placa em balanço (QM6)", f_q4: "Placa em balanço com Q4 (controle: travamento)", f_wall: "Parede de concreto (pilar-parede)", p_loop: "Rede de água em malha", p_two: "Problema dos três reservatórios",
  r_series: "A → B → C (Bateman)", r_net: "Rede de reações: invariantes", r_cstr: "Partida de um CSTR", x_bt: "Flash benzeno–tolueno", d_bt: "Coluna benzeno–tolueno" } });

mount("eng", (root, S, redraw) => {
  head(root, "eng", "eng_lede");
  S.kind ??= "circuit"; S.res ??= {}; S.errs ??= {};
  const firstOf = (k) => Object.keys(EX_ENG[k])[0];
  S.texts ??= Object.fromEntries(Object.keys(EX_ENG).map((k) => [k, EX_ENG[k][firstOf(k)]]));
  const col = new Intl.Collator(lang === "pt" ? "pt-BR" : "en");
  const kinds = Object.keys(EX_ENG).sort((a, b) => col.compare(t("e_kinds")[a], t("e_kinds")[b]));
  const ed = editor("wb-eng-ed", 14); ed.value = S.texts[S.kind]; ed.oninput = () => (S.texts[S.kind] = ed.value);
  const out = el("div", { class: "wb-out", "aria-live": "polite" }), side = el("aside", { class: "wb-side" });
  const method = sel([["newton", "Newton–Raphson"], ["gauss_seidel", "Gauss–Seidel (control)"]], S.method || "newton"); method.onchange = () => (S.method = method.value);
  const kind = S.kind;
  const go = async () => {
    busy(out); bar.state.textContent = t("running");
    try { S.res[kind] = await api("/v1/vapor/engineering", { kind, text: ed.value, method: method.value }); S.errs[kind] = null; } catch (e) { S.res[kind] = null; S.errs[kind] = e.message; }
    if (S.kind === kind && root.contains(out)) show();
  };
  ed.addEventListener("run", go);
  const bar = deskBar(EX_ENG[kind], () => t("ex_eng"), (k, text) => { ed.value = S.texts[kind] = text; go(); }, [btn(t("run"), "primary", go)]);
  if (kind === "power") bar.bar.insertBefore(method, bar.state);
  root.append(seg(kinds, t("e_kinds"), kind, (k) => { S.kind = k; redraw(); }), bar.bar, el("div", { class: "wb-main" }, ed, side), out);
  function show() {
    side.replaceChildren(); out.replaceChildren();
    const err = S.errs[kind], r = S.res[kind];
    if (err) { bar.state.textContent = ""; out.append(notice(err)); return; }
    if (!r) return;
    bar.state.textContent = "";
    ENG_VIEW[kind](r, side, out, ed.value, S);
  }
  show();
  if (S.res[kind] === undefined && S.errs[kind] === undefined) go();
});

// a deterministic spring layout for graphs without coordinates (power buses, pipe nodes)
function springLayout(ids, edges, w, h) {
  const n = ids.length, P = ids.map((_, i) => [w / 2 + Math.cos((2 * Math.PI * i) / n) * w * 0.35, h / 2 + Math.sin((2 * Math.PI * i) / n) * h * 0.35]);
  const ix = new Map(ids.map((d, i) => [d, i])), k = Math.sqrt((w * h) / Math.max(n, 1)) * 0.75;
  for (let it = 0; it < 300; it++) {
    const D = P.map(() => [0, 0]), temp = (w / 10) * (1 - it / 300);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) { const dx = P[i][0] - P[j][0], dy = P[i][1] - P[j][1], d = Math.max(Math.hypot(dx, dy), 0.01); D[i][0] += (dx / d) * (k * k) / d; D[i][1] += (dy / d) * (k * k) / d; }
    edges.forEach(([a, b]) => { const i = ix.get(a), j = ix.get(b); if (i == null || j == null) return; const dx = P[i][0] - P[j][0], dy = P[i][1] - P[j][1], d = Math.max(Math.hypot(dx, dy), 0.01), f = (d * d) / k; D[i][0] -= (dx / d) * f; D[i][1] -= (dy / d) * f; D[j][0] += (dx / d) * f; D[j][1] += (dy / d) * f; });
    P.forEach((p, i) => { const d = Math.max(Math.hypot(...D[i]), 0.01); p[0] = Math.min(w - 40, Math.max(40, p[0] + (D[i][0] / d) * Math.min(d, temp))); p[1] = Math.min(h - 30, Math.max(30, p[1] + (D[i][1] / d) * Math.min(d, temp))); });
  }
  return new Map(ids.map((d, i) => [d, P[i]]));
}
const svgText = (x, y, text, cls = "lbl", anchor = "middle") => { const n = svgEl("text", { x, y, class: cls, "text-anchor": anchor }); n.textContent = text; return n; };
function arrowDefs(svg) { const d = svgEl("defs", {}); const m = svgEl("marker", { id: "arr", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "6", markerHeight: "6", orient: "auto-start-reverse" }); m.append(svgEl("path", { d: "M0 0 L10 5 L0 10 z", class: "arrowhead" })); d.append(m); svg.append(d); }
const ramp = (v) => { const c = [[13, 8, 135], [126, 3, 168], [204, 71, 120], [248, 149, 64], [240, 249, 33]]; const tt = Math.max(0, Math.min(1, v)) * (c.length - 1), k = Math.min(c.length - 2, Math.floor(tt)), f = tt - k; return `rgb(${c[k].map((a, i) => Math.round(a + (c[k + 1][i] - a) * f)).join(",")})`; };

const ENG_VIEW = {
  circuit(r, side, out) {
    side.append(stats(stat("elements", String(r.elements)), stat(t("node"), String(r.nodes.length))));
    if (r.op) {
      const c = r.op.certificate;
      side.append(sect(t("certificate"), certRow(t("kcl"), c.kcl_max, 1e-9), certRow(t("power_bal"), c.power_balance, 1e-8)), facts([[t("iterations_n"), r.op.iterations]]));
      (r.op.warnings || []).forEach((w) => out.append(el("p", { class: "notice", text: w })));
      out.append(sect(t("nodes_h"), table([t("node"), "V"], Object.entries(r.op.nodes).sort().map(([k, v]) => [k, wbNum(v, 8)]))),
        sect(t("branches_h"), table(["", "I (A)", "P (W)"], Object.keys(r.op.currents).sort().map((k) => [k, wbNum(r.op.currents[k], 6), wbNum(r.op.power[k], 6)]))));
    }
    if (r.ac) {
      const pts = r.ac.points, ns = Object.keys(pts[0].nodes).filter((n) => n !== "0").sort(), pick = sel(ns.map((n) => [n, `V(${n})`]), ns[ns.length - 1]), box = el("div");
      const lf = (f) => Math.log10(f), lo = Math.floor(lf(pts[0].f)), hi = Math.ceil(lf(pts[pts.length - 1].f));
      const ticks = Array.from({ length: hi - lo + 1 }, (_, i) => [lo + i, wbNum(10 ** (lo + i))]);
      const dr = () => { const n = pick.value; const peak = pts.reduce((a, p) => (p.nodes[n].db > a.nodes[n].db ? p : a));
        box.replaceChildren(el("p", { class: "muted", text: `peak ${wbNum(peak.nodes[n].db, 4)} dB @ ${wbNum(peak.f, 5)} Hz` }),
          lineChart({ series: [{ points: pts.map((p) => [lf(p.f), p.nodes[n].db]), cls: 0 }], xticks: ticks, ylab: t("magnitude"), w: 760, h: 220, marks: [{ x: lf(peak.f), label: wbNum(peak.f, 4) + " Hz" }] }),
          lineChart({ series: [{ points: pts.map((p) => [lf(p.f), p.nodes[n].phase]), cls: 1 }], xticks: ticks, ylab: t("phase_deg"), w: 760, h: 180 })); };
      pick.onchange = dr; dr(); out.append(sect(t("bode"), pick, box));
    }
    if (r.tran) {
      const names = Object.keys(r.tran.nodes).sort(), chosen = new Set(names), box = el("div");
      const dr = () => { const ns = names.filter((n) => chosen.has(n)); box.replaceChildren(legendOf(ns.map((n) => `V(${n})`)), multiChart(r.tran.t.map((x) => x * 1e3), r.tran.nodes, ns, { xlab: "t (ms)" })); };
      side.append(facts([[t("method"), r.tran.method], ["h", r.tran.h], [t("steps_n"), r.tran.steps]]));
      out.append(sect(t("transient"), seriesPicker(names, chosen, dr), box)); dr();
    }
  },
  power(r, side, out) {
    side.append(sect(t("certificate"), certRow(t("mismatch"), r.certificate.max_mismatch_pu, 1e-8), certRow(t("balance"), r.certificate.balance_mw, 1e-6)),
      stats(stat(t("iterations_n"), String(r.iterations)), stat(t("losses"), `${wbNum(r.losses_mw, 4)} MW`)), facts([[t("method"), r.method], ["base", `${r.base_mva} MVA`]]));
    const W = 760, H = 360, svg = svgEl("svg", { class: "diag", viewBox: `0 0 ${W} ${H}` }); arrowDefs(svg);
    const pos = springLayout(r.buses.map((b) => b.id), r.lines.map((l) => [l.from, l.to]), W, H), pmax = Math.max(...r.lines.map((l) => Math.abs(l.p_from)), 1);
    r.lines.forEach((l) => { const [a, b] = l.p_from >= 0 ? [pos.get(l.from), pos.get(l.to)] : [pos.get(l.to), pos.get(l.from)];
      const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, d = Math.hypot(b[0] - a[0], b[1] - a[1]), ux = (b[0] - a[0]) / d, uy = (b[1] - a[1]) / d;
      svg.append(svgEl("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: "wire", "stroke-width": 1.5 + 6 * Math.abs(l.p_from) / pmax }),
        svgEl("line", { x1: mx - ux * 12, y1: my - uy * 12, x2: mx + ux * 12, y2: my + uy * 12, class: "flow", "marker-end": "url(#arr)" }),
        svgText(mx - uy * 14, my + ux * 14 + 4, `${wbNum(Math.abs(l.p_from), 3)} MW`, "lbl small")); });
    r.buses.forEach((b) => { const [x, y] = pos.get(b.id), dev = Math.abs(b.v - 1);
      svg.append(svgEl("rect", { x: x - 22, y: y - 5, width: 44, height: 10, rx: 2, class: "bus", style: `fill:${dev > 0.05 ? "var(--ember)" : dev > 0.03 ? "var(--sand)" : "var(--lock)"}` }),
        svgText(x, y - 10, `${b.id} · ${b.kind}`, "lbl"), svgText(x, y + 22, `${wbNum(b.v, 4)} ∠${wbNum(b.angle_deg, 3)}°`, "lbl small")); });
    out.append(sect(t("oneline"), svg, el("p", { class: "muted", text: t("bus_legend") })),
      sect(t("buses_h"), table(["bus", "kind", "|V| (pu)", "∠ (°)", "P (MW)", "Q (Mvar)"], r.buses.map((b) => [b.id, b.kind, wbNum(b.v, 6), wbNum(b.angle_deg, 5), wbNum(b.p_mw, 5), wbNum(b.q_mvar, 5)]))),
      sect(t("lines_h"), table(["from", "to", "P→ (MW)", "Q→ (Mvar)", "P← (MW)", t("losses") + " (MW)"], r.lines.map((l) => [l.from, l.to, wbNum(l.p_from, 5), wbNum(l.q_from, 5), wbNum(l.p_to, 5), wbNum(l.loss_mw, 4)]))));
  },
  structure(r, side, out, _text, S) {
    side.append(sect(t("certificate"), certRow(t("equilibrium"), r.certificate.relative, 1e-9)), stats(stat(t("max_disp"), wbNum(r.max_displacement, 4)), stat(t("dofs"), String(r.dofs)), stat(t("bandwidth"), String(r.bandwidth))));
    if (r.auto_restrained_rotations) side.append(el("p", { class: "muted", text: `${r.auto_restrained_rotations} truss-node rotations restrained automatically` }));
    const ids = Object.keys(r.nodes), xs = ids.map((i) => r.nodes[i].x), ys = ids.map((i) => r.nodes[i].y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), span = Math.max(x1 - x0, y1 - y0, 1e-9);
    const W = 760, H = 380, pad = 60, sc = Math.min((W - 2 * pad) / Math.max(x1 - x0, span * 0.3), (H - 2 * pad) / Math.max(y1 - y0, span * 0.3));
    const X = (x) => W / 2 + (x - (x0 + x1) / 2) * sc, Y = (y) => H / 2 - (y - (y0 + y1) / 2) * sc;
    const amp = r.max_displacement > 0 ? (0.08 * span) / r.max_displacement : 1;
    const svg = svgEl("svg", { class: "diag", viewBox: `0 0 ${W} ${H}` }), gDef = svgEl("g", {});
    r.elements.forEach((e) => svg.append(svgEl("line", { x1: X(r.nodes[e.a].x), y1: Y(r.nodes[e.a].y), x2: X(r.nodes[e.b].x), y2: Y(r.nodes[e.b].y), class: e.truss ? "undeformed truss" : "undeformed" })));
    svg.append(gDef);
    const beamNodes = new Set(r.elements.filter((e) => !e.truss).flatMap((e) => [e.a, e.b]));
    const supports = Object.entries(r.reactions).filter(([id, v]) => v.fx != null || v.fy != null || (v.mz != null && beamNodes.has(id))).map(([id]) => id);
    supports.forEach((id) => { const n = r.nodes[id], x = X(n.x), y = Y(n.y), v = r.reactions[id];
      svg.append(v.mz != null && beamNodes.has(id) ? svgEl("path", { d: `M${x - 12} ${y + 2} h24 M${x - 12} ${y + 2} l-5 7 M${x - 4} ${y + 2} l-5 7 M${x + 4} ${y + 2} l-5 7 M${x + 12} ${y + 2} l-5 7`, class: "fixed" }) : svgEl("path", { d: `M${x} ${y} l-10 16 h20 z`, class: "support" })); });
    ids.filter((i) => !i.includes("~")).forEach((i) => svg.append(svgText(X(r.nodes[i].x) + 8, Y(r.nodes[i].y) - 8, i, "lbl small", "start")));
    const drawShape = (get, k) => { gDef.replaceChildren(...r.elements.map((e) => { const a = r.nodes[e.a], b = r.nodes[e.b], da = get(e.a), db = get(e.b);
      return svgEl("line", { x1: X(a.x + k * da[0]), y1: Y(a.y + k * da[1]), x2: X(b.x + k * db[0]), y2: Y(b.y + k * db[1]), class: "deformed" }); })); };
    let raf = 0; const stopAnim = () => cancelAnimationFrame(raf);
    const statics = () => { stopAnim(); drawShape((id) => [r.nodes[id].ux, r.nodes[id].uy], amp); };
    statics();
    out.append(sect(`${t("frame_h")} ${wbNum(amp, 3)})`, svg));
    if (r.modes && r.modes.length) {
      const rows = r.modes.map((m, i) => { const b = btn(`${t("animate")} ${i + 1}`, "quiet", () => {
        stopAnim(); const mx = Math.max(...Object.values(m.shape).map((v) => Math.hypot(v[0], v[1]))) || 1, k = (0.1 * span) / mx, t0 = performance.now();
        const loop = (now) => { if (!svg.isConnected) return; drawShape((id) => m.shape[id], k * Math.sin(((now - t0) / 1000) * 2 * Math.PI * 0.8)); raf = requestAnimationFrame(loop); }; raf = requestAnimationFrame(loop); });
        return [String(i + 1), wbNum(m.omega, 6), wbNum(m.hz, 6), b]; });
      out.append(sect(t("modes_h"), table(["#", "ω (rad/s)", "f (Hz)", ""], rows), btn("static", "quiet", statics)));
    }
    // member diagrams: the sub-elements of one member, end to end
    const groups = new Map(); r.members.forEach((m) => { if (!groups.has(m.member)) groups.set(m.member, []); groups.get(m.member).push(m); });
    const names = [...groups.keys()], pick = sel(names.map((n) => [n, n]), S.member && groups.has(S.member) ? S.member : names[0]), box = el("div");
    const dr = () => { S.member = pick.value; let off = 0; const pts = { axial: [], shear: [], moment: [] };
      groups.get(pick.value).forEach((m) => { m.x.forEach((x, i) => ["axial", "shear", "moment"].forEach((q) => pts[q].push([off + x, m[q][i]]))); off += m.length; });
      box.replaceChildren(...(groups.get(pick.value)[0].truss ? ["axial"] : ["axial", "shear", "moment"]).map((q, i) => el("div", {}, el("p", { class: "muted", text: `${t(q)} — max |·| ${wbNum(Math.max(...pts[q].map((p) => Math.abs(p[1]))), 5)}` }),
        lineChart({ series: [{ points: pts[q], cls: i }], levels: [{ y: 0 }], xlab: "x (m)", w: 760, h: 150 })))); };
    pick.onchange = dr; dr();
    out.append(sect(t("diagram"), pick, box), sect(t("reactions_h"), table(["node", "Fx", "Fy", "Mz"], Object.entries(r.reactions).filter(([id]) => supports.includes(id)).map(([k, v]) => [k, wbNum(v.fx, 6), wbNum(v.fy, 6), wbNum(v.mz, 6)]))));
  },
  fem(r, side, out) {
    side.append(sect(t("certificate"), certRow(t("residual"), r.certificate.residual / Math.max(1, Math.abs(r.max_von_mises)), 1e-8), certRow("ΣFx", r.certificate.sum_fx, 1e-6), certRow("ΣFy", r.certificate.sum_fy, 1e-6)),
      stats(stat(t("max_disp"), wbNum(r.max_displacement, 4)), stat(t("max_vm"), wbNum(r.max_von_mises, 4)), stat(t("dofs"), String(r.dofs))), facts([[t("bandwidth"), r.bandwidth]]));
    const N = r.nodes, ids = Object.keys(N), xs = ids.map((i) => N[i].x), ys = ids.map((i) => N[i].y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), span = Math.max(x1 - x0, y1 - y0);
    const amp = r.max_displacement > 0 ? (0.12 * span) / r.max_displacement : 1, W = 760, pad = 40;
    const sc = Math.min((W - 2 * pad) / (x1 - x0 || 1), 420 / (y1 - y0 || 1)), H = Math.max(160, (y1 - y0) * sc + 2 * pad + 0.25 * span * sc);
    const X = (x) => W / 2 + (x - (x0 + x1) / 2) * sc, Y = (y) => H / 2 - (y - (y0 + y1) / 2) * sc;
    const svg = svgEl("svg", { class: "diag", viewBox: `0 0 ${W} ${H}` }), vm = (id) => r.stress[id]?.von_mises ?? 0, vmax = r.max_von_mises || 1;
    r.quads.forEach((q) => svg.append(svgEl("polygon", { points: q.map((id) => `${X(N[id].x)},${Y(N[id].y)}`).join(" "), class: "ghost" })));
    r.quads.forEach((q) => { const v = q.reduce((a, id) => a + vm(id), 0) / 4;
      svg.append(svgEl("polygon", { points: q.map((id) => `${X(N[id].x + amp * N[id].ux).toFixed(1)},${Y(N[id].y + amp * N[id].uy).toFixed(1)}`).join(" "), class: "cell", style: `fill:${ramp(v / vmax)}` })); });
    const lg = el("div", { class: "ramp" }, el("span", { text: "0" }), el("i"), el("span", { text: wbNum(vmax, 4) }));
    out.append(sect(`${t("mesh_h")} ${wbNum(amp, 3)})`, svg, lg));
  },
  pipes(r, side, out) {
    const c = r.certificate;
    side.append(sect(t("certificate"), certRow(t("continuity"), c.continuity_max, 1e-9), certRow(t("loop_energy"), c.loop_energy_max, 1e-6), certRow("reservoir paths (m)", c.reservoir_paths, 1e-6)),
      stats(stat(t("iterations_n"), String(r.iterations)), stat("loops", String(r.loops))));
    const ids = [...new Set(r.pipes.flatMap((p) => [p.from, p.to]))], W = 760, H = 360, svg = svgEl("svg", { class: "diag", viewBox: `0 0 ${W} ${H}` }); arrowDefs(svg);
    const pos = springLayout(ids, r.pipes.map((p) => [p.from, p.to]), W, H), qmax = Math.max(...r.pipes.map((p) => Math.abs(p.flow)), 1e-9), J = new Map(r.junctions.map((j) => [j.id, j]));
    r.pipes.forEach((p) => { const [a, b] = p.flow >= 0 ? [pos.get(p.from), pos.get(p.to)] : [pos.get(p.to), pos.get(p.from)];
      const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, d = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1, ux = (b[0] - a[0]) / d, uy = (b[1] - a[1]) / d;
      svg.append(svgEl("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: "pipe", "stroke-width": 2 + 8 * Math.abs(p.flow) / qmax }),
        svgEl("line", { x1: mx - ux * 12, y1: my - uy * 12, x2: mx + ux * 12, y2: my + uy * 12, class: "flow", "marker-end": "url(#arr)" }),
        svgText(mx - uy * 16, my + ux * 16 + 4, `${p.id}: ${wbNum(Math.abs(p.flow) * 1000, 3)} L/s`, "lbl small")); });
    ids.forEach((id) => { const [x, y] = pos.get(id), j = J.get(id);
      svg.append(j ? svgEl("circle", { cx: x, cy: y, r: 8, class: "junction" }) : svgEl("rect", { x: x - 14, y: y - 10, width: 28, height: 20, rx: 3, class: "reservoir" }),
        svgText(x, y - 16, id, "lbl"), ...(j ? [svgText(x, y + 24, `H ${wbNum(j.head, 4)} m`, "lbl small")] : [])); });
    out.append(sect(t("network"), svg),
      sect(t("pipes_h"), table(["", "from → to", "Q (L/s)", "v (m/s)", "Re", "f", "regime", "h_f (m)"], r.pipes.map((p) => [p.id, `${p.from} → ${p.to}`, wbNum(p.flow * 1000, 5), wbNum(p.velocity, 4), wbNum(p.reynolds, 4), wbNum(p.friction, 5), p.regime, wbNum(p.headloss, 5)]))),
      sect(t("junctions_h"), table(["", "H (m)", "p/γ (m)", "p (kPa)"], r.junctions.map((j) => [j.id, wbNum(j.head, 6), wbNum(j.pressure_head, 6), wbNum(j.pressure / 1000, 5)]))));
  },
  reactions(r, side, out) {
    side.append(stats(stat(t("species"), String(r.species.length)), stat(t("steps_n"), String(r.steps))), facts([[t("method"), r.method + (r.switched ? " → Rosenbrock" : "")], ["reactor", r.reactor]]),
      sect(t("final_values"), facts(Object.entries(r.final).map(([k, v]) => [k, wbNum(v, 6)]))));
    const chosen = new Set(r.species), box = el("div"), dr = () => { const ns = r.species.filter((s) => chosen.has(s)); box.replaceChildren(legendOf(ns), multiChart(r.t, r.series, ns, { xlab: "t" })); };
    out.append(sect(t("species"), seriesPicker(r.species, chosen, dr), box)); dr();
    out.append(sect(t("invariants"), r.invariants.length ? table([t("combination"), t("initial"), t("drift"), ""], r.invariants.map((v) => [el("code", { text: v.combination }), wbNum(v.initial, 6), wbNum(v.drift, 2), okChip(v.drift < 1e-8)])) : el("p", { class: "muted", text: "—" })),
      sect(t("odes_h"), el("pre", { class: "code-out", text: r.odes.join("\n") })),
      sect("ν", table(["", ...r.reactions.map((_, j) => `R${j + 1}`)], r.species.map((s, i) => [s, ...r.stoichiometry[i].map(String)]))));
  },
  flash(r, side, out) {
    const c = r.certificate, comps = Object.keys(r.x);
    side.append(el("p", { class: "big-kind", text: r.phase }), sect(t("certificate"), certRow("Σ balance", c.balance, 1e-12), certRow("Rachford–Rice", c.rachford_rice, 1e-10), certRow("Σx − 1", c.sum_x - 1, 1e-10), certRow("Σy − 1", c.sum_y - 1, 1e-10)),
      stats(stat(t("vapour_fraction"), wbNum(r.vapour_fraction, 5)), stat(t("bubble"), wbNum(r.bubble_p, 5)), stat(t("dew"), wbNum(r.dew_p, 5))), facts([["T", r.t], ["P", r.p]]));
    out.append(sect(t("phase_h"), table([t("component"), "x (liquid)", "y (vapour)", "K"], comps.map((k) => [k, wbNum(r.x[k], 6), wbNum(r.y[k], 6), wbNum(r.k[k], 6)]))),
      legend([[0, "x"], [1, "y"]]), lineChart({ series: [{ points: comps.map((k, i) => [i, r.x[k]]), bars: true, cls: 0, bw: 0.12 }, { points: comps.map((k, i) => [i + 0.3, r.y[k]]), bars: true, cls: 1, bw: 0.12 }], xticks: comps.map((k, i) => [i + 0.15, k]), w: 420, h: 200, ymin: 0 }));
  },
  distill(r, side, out, text) {
    const g = (k, d) => { const m = text.match(new RegExp(`\\b${k}\\s*=\\s*([0-9.eE+-]+)`)); return m ? +m[1] : d; };
    const a = g("alpha", 2.5), xF = g("xF", 0.5), xD = g("xD", 0.95), xB = g("xB", 0.05), q = g("q", 1), R = r.reflux;
    side.append(stats(stat(t("stages"), String(r.stages)), stat(t("feed_stage"), String(r.feed_stage))),
      facts([[t("rmin"), r.rmin], [t("reflux"), R], [t("fenske"), r.fenske], [t("gilliland"), r.gilliland]]),
      sect(t("control"), el("p", {}, okChip(r.control.total_reflux_stages === Math.ceil(r.control.fenske)), " ", el("span", { class: "muted", text: r.control.note }))));
    const S = 420, P = 36, X = (x) => P + x * (S - 2 * P), Y = (y) => S - P - y * (S - 2 * P), svg = svgEl("svg", { class: "diag square", viewBox: `0 0 ${S} ${S}` });
    const path = (pts, cls) => svgEl("path", { d: pts.map(([x, y], i) => `${i ? "L" : "M"}${X(x).toFixed(1)} ${Y(y).toFixed(1)}`).join(" "), class: cls });
    for (let k = 0; k <= 10; k++) svg.append(svgEl("line", { x1: X(k / 10), x2: X(k / 10), y1: Y(0), y2: Y(1), class: "gridl" }), svgEl("line", { x1: X(0), x2: X(1), y1: Y(k / 10), y2: Y(k / 10), class: "gridl" }));
    const eq = Array.from({ length: 101 }, (_, i) => { const x = i / 100; return [x, (a * x) / (1 + (a - 1) * x)]; });
    // the intersection of the rectifying line with the q-line
    const xi = Math.abs(q - 1) < 1e-9 ? xF : (xD / (R + 1) + xF / (q - 1)) / (q / (q - 1) - R / (R + 1)), yi = (R / (R + 1)) * xi + xD / (R + 1);
    svg.append(path([[0, 0], [1, 1]], "diag-l"), path(eq, "eq-l"), path([[xD, xD], [xi, yi]], "op-l"), path([[xB, xB], [xi, yi]], "op-l"), path([[xF, xF], [xi, yi]], "q-l"), path(r.steps, "steps-l"),
      svgText(X(0.5), S - 6, "x", "lbl"), svgText(10, Y(0.5), "y", "lbl"), svgText(X(xD), Y(xD) + 16, "xD", "lbl small"), svgText(X(xB), Y(xB) - 8, "xB", "lbl small"), svgText(X(xF), Y(xF) + 16, "xF", "lbl small"));
    out.append(sect(t("mccabe"), legend([[0, t("equilibrium_curve")], [1, t("operating")], [2, "q"]]), svg));
  },
};

/* ---------------------------------------------------------------- logic */
const EX_LOGIC = {
  schur: "schur 3", vdw: "vdw 3 2", ramsey: "ramsey 3 3",
  taut: "valid: ((p -> q) & (q -> r)) -> (p -> r)", equiv: "equiv: !(a & b) ; !a | !b", nontaut: "valid: (p -> q) -> (q -> p)",
  dimacs: "c a small CNF: satisfiable or refuted, with a certificate either way\np cnf 4 6\n1 2 0\n-1 3 0\n-2 3 0\n-3 4 0\n-4 -1 0\n-4 -2 0",
  php: "pigeonhole 6 5", queens: "queens 8",
  group: "vars x y z\nprecedence i > * > e\ne * x = x\ni(x) * x = e\n(x * y) * z = x * (y * z)\ndecide i(x * y) = i(y) * i(x)\ndecide i(i(x)) * e = x\ndecide x * y = y * x",
  thales: "vars x y a b\nhyp x^2 + y^2 - 1\nhyp a + 1\nhyp b - 1\nclaim (x - a)*(x - b) + y*y",
  lexsys: "vars x y z\nhyp x^2 + y + z - 1\nhyp x + y^2 + z - 1\nhyp x + y + z^2 - 1\norder lex",
};
mount("logic", (root, S) => {
  head(root, "logic", "logic_lede");
  const ed = editor("wb-logic-ed", 10); ed.value = S.text ?? EX_LOGIC.schur; ed.oninput = () => (S.text = ed.value);
  const out = el("div", { class: "wb-out", "aria-live": "polite" }), side = el("aside", { class: "wb-side" });
  const go = async () => { S.text = ed.value; busy(out); bar.state.textContent = t("running");
    try { S.r = await api("/v1/vapor/logic", { text: ed.value }); S.err = null; } catch (e) { S.r = null; S.err = e.message; } show(); };
  ed.addEventListener("run", go);
  const bar = deskBar(EX_LOGIC, () => t("ex_logic"), (k, text) => { ed.value = S.text = text; go(); }, [btn(t("run"), "primary", go)]);
  root.append(bar.bar, el("div", { class: "wb-main" }, ed, side), out);
  function show() {
    side.replaceChildren(el("p", { class: "muted", text: t("mcp_note") })); out.replaceChildren();
    if (S.err) { bar.state.textContent = ""; out.append(notice(S.err)); return; }
    if (!S.r) return;
    const r = S.r; bar.state.textContent = r.ms != null ? t("ms", r.ms) : "";
    const verdict = r.verdict || (r.basis ? (r.consistent ? "consistent" : "inconsistent") : r.kind);
    const head = [el("p", { class: "muted", text: t("logic_verdict") }), el("p", { class: "big-kind", text: verdict })];
    const cert = r.certificate || r.refutation;
    if (r.below) head.push(el("p", {}, okChip(r.below.checked, t("below", r.below.n))), el("p", {}, okChip(r.refutation?.drup?.valid, t("refuted_at", r.refuted_at))));
    if (cert && cert.drup) head.push(el("p", {}, okChip(cert.drup.valid, cert.drup.valid ? t("drup_ok", cert.drup.lemmas, cert.drup.core_lemmas) : t("drup_bad"))));
    if (cert && cert.model_checked != null) head.push(el("p", {}, okChip(cert.model_checked, t("model_checked"))));
    side.prepend(...head);
    if (r.stats) side.append(facts(Object.entries(r.stats).map(([k, v]) => [k, v])));
    const W = r.below ? r.below.witness : r.witness, prob = (r.problem || "").toLowerCase();
    if (W && /schur|waerden|w\(/.test(prob)) out.append(sect(t("witness") + ` (n = ${W.length})`, strip(W)));
    else if (W && /ramsey|r\(/.test(prob)) out.append(sect(t("witness") + ` (K${r.below.n})`, kGraph(r.below.n, W)));
    else if (W && /queens|rainhas/.test(prob)) out.append(sect(t("witness"), queensBoard(W)));
    else if (W) out.append(sect(t("witness"), el("pre", { class: "code-out", text: JSON.stringify(W) })));
    if (r.model && !W) out.append(sect(t("model"), el("p", { class: "mono", text: r.model.length ? r.model.map((v) => "x" + v).join(" ∧ ") + " (others false)" : "all false" })));
    if (r.counterexample) out.append(sect(t("counterexample"), table(["", ""], Object.entries(r.counterexample).map(([k, v]) => [k, v ? "true" : "false"]))));
    if (r.claim) out.prepend(el("p", { class: "claim", text: r.claim }));
    if (r.proof_head) out.append(sect(t("proof_head"), el("pre", { class: "code-out", text: r.proof_head.map((c) => (c.length ? c.join(" ") : "⊥") + " 0").join("\n") })));
    if (r.rules) out.append(sect(t("rules") + ` (${r.rules.length})`, el("ol", { class: "rules" }, ...r.rules.map((x) => el("li", {}, el("code", { text: x })))), el("p", { class: "muted", text: `${r.critical_pairs} critical pairs · ${r.steps} steps` })));
    (r.decisions || []).forEach((d) => out.append(sect(d.question, el("p", {}, chipV(d.equal, d.equal ? "equal" : "not equal (distinct normal forms)")),
      table([t("normal_forms"), t("derivation")], d.normal_forms.map((nf, i) => [el("code", { text: nf }), el("span", { class: "mono small", text: (d.derivations[i] || []).join("  →  ") })])))));
    if (r.basis) out.append(sect(t("basis") + (r.order ? ` (${r.order})` : ""), el("ol", { class: "rules" }, ...r.basis.map((x) => el("li", {}, el("code", { text: x }))))));
    if (r.basis_of_hypotheses) out.append(sect(t("basis"), el("ol", { class: "rules" }, ...r.basis_of_hypotheses.map((x) => el("li", {}, el("code", { text: x })))), el("p", { class: "muted", text: `${t("remainder")}: ${r.remainder}` }), r.certificate && typeof r.certificate === "string" ? el("p", { text: r.certificate }) : ""));
    if (r.conditions && r.conditions.length) out.append(sect("non-degeneracy", el("ul", {}, ...r.conditions.map((c) => el("li", { text: String(c) })))));
  }
  show();
  if (!S.r && !S.err) go();
});
function strip(w) { const d = el("div", { class: "strip" }); w.forEach((c, i) => { const s = el("span", { text: String(i + 1), title: `${i + 1}: colour ${c}` }); s.style.background = palColors[(c - 1) % palColors.length]; d.append(s); }); return d; }
function kGraph(n, red) {
  const S = 300, R = 120, svg = svgEl("svg", { class: "diag square small", viewBox: `0 0 ${S} ${S}` }), P = (i) => [S / 2 + R * Math.cos((2 * Math.PI * (i - 1)) / n - Math.PI / 2), S / 2 + R * Math.sin((2 * Math.PI * (i - 1)) / n - Math.PI / 2)];
  const isRed = new Set(red.map(([a, b]) => `${a}-${b}`));
  for (let a = 1; a <= n; a++) for (let b = a + 1; b <= n; b++) { const [x1, y1] = P(a), [x2, y2] = P(b); svg.append(svgEl("line", { x1, y1, x2, y2, class: isRed.has(`${a}-${b}`) ? "edge-a" : "edge-b" })); }
  for (let a = 1; a <= n; a++) { const [x, y] = P(a); svg.append(svgEl("circle", { cx: x, cy: y, r: 12, class: "vertex" }), svgText(x, y + 4, String(a), "lbl")); }
  return el("div", {}, svg, el("p", { class: "muted", text: "no monochromatic triangle: every triangle has both colours" }));
}
function queensBoard(cols) {
  const n = cols.length, s = 30, svg = svgEl("svg", { class: "diag board-q", viewBox: `0 0 ${n * s} ${n * s}`, width: String(n * s) });
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) svg.append(svgEl("rect", { x: c * s, y: r * s, width: s, height: s, class: (r + c) % 2 ? "sq-d" : "sq-l" }));
  cols.forEach((c, r) => svg.append(svgText((c - 1) * s + s / 2, r * s + s * 0.75, "♛", "piece-q")));
  return svg;
}

/* -------------------------------------------------------- boards & cards */
const CHESS_POS = { start: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", ladder: "7k/8/8/8/8/8/1R6/R5K1 w - - 0 1",
  kiwipete: "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", italian: "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
  kq: "8/8/8/4k3/8/8/8/4K2Q w - - 0 1" };
Object.assign(I18N.en, { chess_pos: { start: "Initial position", ladder: "Mate in 2: the rook ladder", kiwipete: "Kiwipete (perft test position)", italian: "Italian Game", kq: "King and queen against king" },
  infoset_note: "An information set is the player's card followed by the actions so far (c check/call, b bet, f fold, r raise; / separates Leduc's rounds)." });
Object.assign(I18N.pt, { chess_pos: { start: "Posição inicial", ladder: "Mate em 2: a escada de torres", kiwipete: "Kiwipete (posição de teste de perft)", italian: "Partida Italiana", kq: "Rei e dama contra rei" },
  infoset_note: "Um conjunto de informação é a carta do jogador seguida das ações até ali (c passa/paga, b aposta, f desiste, r aumenta; / separa as rodadas do Leduc)." });

mount("boards", (root, S, redraw) => {
  head(root, "boards", "boards_lede");
  S.game ??= "chess";
  const col = new Intl.Collator(lang === "pt" ? "pt-BR" : "en");
  const games = Object.keys(t("b_kinds")).sort((a, b) => col.compare(t("b_kinds")[a], t("b_kinds")[b]));
  root.append(seg(games, t("b_kinds"), S.game, (g) => { S.game = g; redraw(); }));
  const area = el("div", { class: "board-area" }); root.append(area);
  ({ chess: chessDesk, shogi: shogiDesk, go: goDesk, mnk: mnkDesk, poker: pokerDesk })[S.game](area, (S[S.game] ??= {}), redraw);
});

function boardShell(area) {
  const board = el("div", { class: "board-wrap" }), side = el("aside", { class: "board-side" }), msg = el("p", { class: "muted wb-state", "aria-live": "polite" });
  area.append(el("div", { class: "board-grid" }, board, side), msg);
  return { board, side, msg };
}

/* chess */
const CH_GLYPH = { 1: "♟", 2: "♞", 3: "♝", 4: "♜", 5: "♛", 6: "♚" };
const sqName = (r, f) => "abcdefgh"[f] + (r + 1);
async function chessCall(C, body) { const r = await api("/v1/vapor/chess", { fen: C.fen, ...body }); return r; }
function chessDesk(area, C, redraw) {
  C.fen ??= CHESS_POS.start; C.hist ??= []; C.sans ??= []; C.depth ??= 4; C.engineOn ??= true; C.flip ??= false;
  const { board, side, msg } = boardShell(area);
  const load = async (fen, keepHist) => { try { const r = await api("/v1/vapor/chess", { fen }); C.fen = r.fen; C.st = r; if (!keepHist) { C.hist = []; C.sans = []; C.last = null; } C.selected = null; C.extra = null; draw(); } catch (e) { msg.textContent = e.message; } };
  const play = async (uci) => {
    msg.textContent = t("running");
    try {
      const r = await chessCall(C, { action: "move", move: uci });
      C.hist.push(C.fen); C.sans.push(r.played); C.fen = r.fen; C.st = r; C.last = [uci.slice(0, 2), uci.slice(2, 4)]; C.selected = null; C.promo = null; draw();
      if (C.engineOn && r.status === "ongoing") await engine();
      else msg.textContent = "";
    } catch (e) { msg.textContent = e.message; }
  };
  const engine = async () => {
    msg.textContent = `${t("engine")} · ${t("depth")} ${C.depth}…`;
    try { const r = await chessCall(C, { action: "engine", depth: C.depth });
      if (r.played) { C.hist.push(C.fen); C.sans.push(r.played); C.last = [r.uci.slice(0, 2), r.uci.slice(2, 4)]; }
      C.fen = r.fen; C.st = r; C.extra = { kind: "engine", r }; msg.textContent = r.played ? `${t("engine_reply")}: ${r.played} · ${t("score")} ${wbNum(r.score / 100, 3)} · ${r.nodes} ${t("nodes_n")}` : r.note || ""; draw();
    } catch (e) { msg.textContent = e.message; }
  };
  function draw() {
    const st = C.st; if (!st) return;
    const s = 56, svg = svgEl("svg", { class: "board chess", viewBox: `0 0 ${8 * s + 18} ${8 * s + 18}`, role: "grid" });
    const targets = C.selected ? st.legal.filter((m) => m.uci.startsWith(C.selected)).map((m) => m.uci.slice(2, 4)) : [];
    for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) {
      const r = C.flip ? i : 7 - i, f = C.flip ? 7 - j : j, name = sqName(r, f), p = st.board[7 - r][f];
      const g = svgEl("g", { class: "sq", "data-sq": name });
      const lightSq = (r + f) % 2 === 1;
      g.append(svgEl("rect", { x: 18 + j * s, y: i * s, width: s, height: s, class: (lightSq ? "sq-l" : "sq-d") + (C.last && C.last.includes(name) ? " sq-last" : "") + (C.selected === name ? " sq-sel" : "") }));
      if (st.check && Math.abs(p) === 6 && Math.sign(p) === (st.turn === "w" ? 1 : -1)) g.append(svgEl("circle", { cx: 18 + j * s + s / 2, cy: i * s + s / 2, r: s * 0.42, class: "sq-check" }));
      if (p) g.append(svgText(18 + j * s + s / 2, i * s + s * 0.78, CH_GLYPH[Math.abs(p)], p > 0 ? "pc pw" : "pc pb"));
      if (targets.includes(name)) g.append(svgEl("circle", { cx: 18 + j * s + s / 2, cy: i * s + s / 2, r: p ? s * 0.46 : s * 0.14, class: p ? "tgt cap" : "tgt" }));
      g.onclick = () => click(name, p);
      svg.append(g);
    }
    for (let k = 0; k < 8; k++) { svg.append(svgText(9, k * s + s / 2 + 4, String(C.flip ? k + 1 : 8 - k), "coord"), svgText(18 + k * s + s / 2, 8 * s + 14, "abcdefgh"[C.flip ? 7 - k : k], "coord")); }
    board.replaceChildren(svg);
    if (C.promo) board.append(el("div", { class: "promo" }, el("span", { text: t("promote") }), ...["q", "r", "b", "n"].map((x) => btn({ q: "♛", r: "♜", b: "♝", n: "♞" }[x], "quiet", () => play(C.promo + x)))));
    renderSide();
  }
  function click(name, p) {
    const st = C.st, mine = p && Math.sign(p) === (st.turn === "w" ? 1 : -1);
    if (C.selected) {
      const ms = st.legal.filter((m) => m.uci.startsWith(C.selected + name));
      if (ms.length === 1) return play(ms[0].uci);
      if (ms.length > 1) { C.promo = C.selected + name; return draw(); }
    }
    C.selected = mine ? name : null; C.promo = null; draw();
  }
  function renderSide() {
    const st = C.st, who = st.turn === "w" ? t("white") : t("black");
    const status = st.status === "checkmate" ? t("checkmate") : st.status === "stalemate" ? t("stalemate") : st.status !== "ongoing" ? String(st.status) : t("to_move", who) + (st.check ? " · +" : "");
    const pos = el("select", { "aria-label": t("example") }, el("option", { value: "", text: t("example") + "…" }), ...Object.keys(CHESS_POS).map((k) => el("option", { value: k, text: t("chess_pos")[k] })));
    pos.onchange = () => pos.value && load(CHESS_POS[pos.value]);
    const fen = el("input", { type: "text", value: C.fen, class: "mono wide", "aria-label": "FEN" });
    const depth = sel([1, 2, 3, 4, 5, 6].map((d) => [d, `${t("depth")} ${d}`]), C.depth); depth.onchange = () => (C.depth = +depth.value);
    const eng = el("input", { type: "checkbox" }); eng.checked = C.engineOn; eng.onchange = () => (C.engineOn = eng.checked);
    const mateN = sel([1, 2, 3].map((d) => [d, String(d)]), C.mateN || 2); mateN.onchange = () => (C.mateN = +mateN.value);
    const pd = sel([1, 2, 3, 4].map((d) => [d, String(d)]), C.perftD || 3); pd.onchange = () => (C.perftD = +pd.value);
    const moves = el("ol", { class: "movelist" }); for (let i = 0; i < C.sans.length; i += 2) moves.append(el("li", {}, el("span", { text: C.sans[i] }), el("span", { text: C.sans[i + 1] || "" })));
    side.replaceChildren(
      el("p", { class: "big-kind", text: status }), el("p", { class: "muted", text: `${t("eval")} ${wbNum(st.eval / 100, 3)} · ${st.legal.length} legal` }),
      el("div", { class: "row" }, pos), el("div", { class: "row" }, fen, btn(t("load"), "quiet", () => load(fen.value.trim()))),
      el("div", { class: "row" }, btn(t("new_game"), "quiet", () => load(CHESS_POS.start)), btn(t("undo"), "quiet", () => { if (C.hist.length) { let f = C.hist.pop(); C.sans.pop(); if (C.engineOn && C.hist.length && C.sans.length % 2 === 1) { f = C.hist.pop(); C.sans.pop(); } C.last = null; load(f, true); } }), btn(t("flip"), "quiet", () => { C.flip = !C.flip; draw(); })),
      el("div", { class: "row" }, depth, el("label", { class: "muted" }, eng, " " + t("engine_on")), btn(t("engine"), "quiet", engine)),
      el("div", { class: "row" }, btn(t("analyse"), "quiet", async () => { msg.textContent = t("running"); try { const r = await chessCall(C, { action: "analyse", depth: C.depth }); C.extra = { kind: "analyse", r }; msg.textContent = ""; renderSide(); } catch (e) { msg.textContent = e.message; } }),
        btn(t("prove_mate"), "quiet", async () => { msg.textContent = t("running"); try { const r = await chessCall(C, { action: "mate", n: C.mateN || 2 }); C.extra = { kind: "mate", r }; msg.textContent = ""; renderSide(); } catch (e) { msg.textContent = e.message; } }), mateN),
      el("div", { class: "row" }, btn(t("perft"), "quiet", async () => { msg.textContent = t("running"); try { const r = await chessCall(C, { action: "perft", depth: C.perftD || 3 }); C.extra = { kind: "perft", r }; msg.textContent = ""; renderSide(); } catch (e) { msg.textContent = e.message; } }), pd),
      C.sans.length ? sect(t("history_moves"), moves) : "", extraView());
  }
  function extraView() {
    const x = C.extra; if (!x) return "";
    if (x.kind === "analyse") return sect(t("analyse"), facts([[t("best_moves"), x.r.best || "—"], [t("score"), wbNum((x.r.score ?? 0) / 100, 3)], ["PV", (x.r.pv || []).join(" ")], [t("nodes_n"), x.r.nodes], [t("depth"), x.r.depth]]));
    if (x.kind === "perft") return sect(`${t("perft")}(${x.r.depth}) = ${x.r.nodes.toLocaleString()}`, x.r.divide ? table(["", t("nodes_n")], Object.entries(x.r.divide).sort().map(([k, v]) => [k, v])) : "");
    if (x.kind === "mate") {
      if (!x.r.mate) return sect(t("prove_mate"), el("p", { text: t("mate_none", x.r.n) }));
      const tree = (n) => el("li", {}, el("b", { text: n.san }), Object.keys(n.replies || {}).length ? el("ul", {}, ...Object.entries(n.replies).map(([d, sub]) => el("li", {}, el("span", { class: "muted", text: `… ${d}  ` }), el("ul", {}, tree(sub))))) : el("span", { class: "muted", text: "  #" }));
      return sect(t("mate_proof"), el("p", {}, okChip(/:ok/.test(x.r.check), t("proof_checked")), " ", el("code", { class: "small", text: x.r.check })), el("ul", { class: "tree" }, tree(x.r.proof)));
    }
    return "";
  }
  if (C.st) draw(); else load(C.fen, true);
}

/* shogi */
const SH_KANJI = { 1: "歩", 2: "香", 3: "桂", 4: "銀", 5: "金", 6: "角", 7: "飛", 8: "玉", 9: "と", 10: "杏", 11: "圭", 12: "全", 14: "馬", 15: "龍" };
const SH_LETTER = { 1: "P", 2: "L", 3: "N", 4: "S", 5: "G", 6: "B", 7: "R" };
const shName = (x, y) => `${9 - x}${"abcdefghi"[y]}`;
function shogiDesk(area, G) {
  G.sfen ??= "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"; G.moves ??= []; G.hist ??= []; G.engineOn ??= true; G.depth ??= 2;
  const { board, side, msg } = boardShell(area);
  const call = (body) => api("/v1/vapor/shogi", { sfen: G.sfen, ...body });
  const set = (r) => { G.sfen = r.sfen; G.st = r; G.sel = null; G.drop = null; G.promo = null; draw(); };
  const play = async (usi) => {
    msg.textContent = t("running");
    try { const r = await call({ action: "move", move: usi }); G.hist.push(G.sfen); G.moves.push(r.played); G.last = usi; set(r);
      if (G.engineOn && r.status === "ongoing") { msg.textContent = t("engine") + "…"; const e = await call({ action: "engine", depth: G.depth }); if (e.played) { G.hist.push(G.sfen); G.moves.push(e.played); G.last = e.played; } set(e); msg.textContent = e.played ? `${t("engine_reply")}: ${e.played} · ${t("score")} ${e.score}` : e.note || ""; }
      else msg.textContent = r.status !== "ongoing" ? String(r.status) : "";
    } catch (e) { msg.textContent = e.message; }
  };
  function draw() {
    const st = G.st; if (!st) return;
    const s = 46, svg = svgEl("svg", { class: "board shogi", viewBox: `0 0 ${9 * s + 4} ${9 * s + 20}` });
    const legal = st.legal, targets = G.drop ? legal.filter((m) => m.startsWith(G.drop + "*")).map((m) => m.slice(2, 4)) : G.sel ? legal.filter((m) => m.startsWith(G.sel)).map((m) => m.slice(2, 4)) : [];
    for (let k = 0; k < 9; k++) svg.append(svgText(2 + k * s + s / 2, 14, String(9 - k), "coord"));
    for (let y = 0; y < 9; y++) for (let x = 0; x < 9; x++) {
      const p = st.board[y][x], name = shName(x, y), g = svgEl("g", {}), X0 = 2 + x * s, Y0 = 18 + y * s;
      g.append(svgEl("rect", { x: X0, y: Y0, width: s, height: s, class: "sq-s" + (G.last && G.last.includes(name) ? " sq-last" : "") + (G.sel === name ? " sq-sel" : "") }));
      if (p) { const k = Math.abs(p), txt = svgText(X0 + s / 2, Y0 + s * 0.7, k === 8 && p > 0 ? "王" : SH_KANJI[k] || "?", "spc" + (k > 8 ? " promoted" : ""));
        if (p < 0) txt.setAttribute("transform", `rotate(180 ${X0 + s / 2} ${Y0 + s / 2})`); g.append(txt); }
      if (targets.includes(name)) g.append(svgEl("circle", { cx: X0 + s / 2, cy: Y0 + s / 2, r: p ? s * 0.44 : s * 0.12, class: p ? "tgt cap" : "tgt" }));
      g.onclick = () => click(name, p);
      svg.append(g);
    }
    [3, 6].forEach((k) => [3, 6].forEach((m) => svg.append(svgEl("circle", { cx: 2 + k * s, cy: 18 + m * s, r: 2.5, class: "star" }))));
    for (let k = 0; k < 9; k++) svg.append(svgText(9 * s + 2 - 2, 18 + k * s + s / 2 + 4, "", "coord"));
    const hand = (side_) => { const h = st.hands[side_] || {}, mine = side_ === st.turn, d = el("div", { class: "hand" + (mine ? " mine" : "") }, el("span", { class: "muted", text: (side_ === "b" ? "☗ " : "☖ ") + t("hands") }));
      Object.entries(h).filter(([, n]) => n > 0).sort((a, b) => b[0] - a[0]).forEach(([k, n]) => { const b = btn(`${SH_KANJI[k]}${n > 1 ? " ×" + n : ""}`, "quiet" + (G.drop === SH_LETTER[k] && mine ? " on" : ""), mine ? () => { G.drop = G.drop === SH_LETTER[k] ? null : SH_LETTER[k]; G.sel = null; draw(); } : null); if (!mine) b.disabled = true; d.append(b); });
      return d; };
    board.replaceChildren(hand("w"), svg, hand("b"));
    if (G.promo) board.append(el("div", { class: "promo" }, el("span", { text: t("promote") + "?" }), btn("成 (+)", "quiet", () => play(G.promo + "+")), btn("不成", "quiet", () => play(G.promo))));
    const pd = sel([1, 2, 3].map((d) => [d, String(d)]), G.perftD || 2); pd.onchange = () => (G.perftD = +pd.value);
    const depth = sel([1, 2, 3].map((d) => [d, `${t("depth")} ${d}`]), G.depth); depth.onchange = () => (G.depth = +depth.value);
    const eng = el("input", { type: "checkbox" }); eng.checked = G.engineOn; eng.onchange = () => (G.engineOn = eng.checked);
    const sfen = el("input", { type: "text", value: G.sfen, class: "mono wide", "aria-label": "SFEN" });
    side.replaceChildren(el("p", { class: "big-kind", text: st.status !== "ongoing" ? String(st.status) : (st.turn === "b" ? "☗ Sente" : "☖ Gote") + (st.check ? " · 王手" : "") }),
      el("p", { class: "muted", text: `${st.legal.length} legal · ${t("drop_hint")}` }),
      el("div", { class: "row" }, sfen, btn(t("load"), "quiet", async () => { try { G.sfen = sfen.value.trim(); G.moves = []; G.hist = []; set(await call({})); } catch (e) { msg.textContent = e.message; } })),
      el("div", { class: "row" }, btn(t("new_game"), "quiet", async () => { G.sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"; G.moves = []; G.hist = []; G.last = null; set(await call({})); }),
        btn(t("undo"), "quiet", async () => { if (!G.hist.length) return; let f = G.hist.pop(); G.moves.pop(); if (G.engineOn && G.hist.length && G.moves.length % 2 === 1) { f = G.hist.pop(); G.moves.pop(); } G.sfen = f; G.last = null; set(await call({})); })),
      el("div", { class: "row" }, depth, el("label", { class: "muted" }, eng, " " + t("engine_on"))),
      el("div", { class: "row" }, btn(t("perft"), "quiet", async () => { msg.textContent = t("running"); try { const r = await call({ action: "perft", depth: G.perftD || 2 }); msg.textContent = `${t("perft")}(${r.depth}) = ${r.nodes.toLocaleString()}  (published from the start: 30 · 900 · 25 470)`; } catch (e) { msg.textContent = e.message; } }), pd),
      G.moves.length ? sect(t("history_moves"), el("p", { class: "mono small", text: G.moves.map((m, i) => `${i + 1}. ${m}`).join("  ") })) : "");
  }
  function click(name, p) {
    const st = G.st, mine = p && Math.sign(p) === (st.turn === "b" ? 1 : -1);
    if (G.drop) { const m = `${G.drop}*${name}`; if (st.legal.includes(m)) return play(m); }
    if (G.sel) { const plain = G.sel + name, can = st.legal.includes(plain), canP = st.legal.includes(plain + "+");
      if (can && canP) { G.promo = plain; return draw(); } if (canP) return play(plain + "+"); if (can) return play(plain); }
    G.sel = mine ? name : null; G.drop = null; G.promo = null; draw();
  }
  if (G.st) draw(); else call({}).then(set, (e) => (msg.textContent = e.message));
}

/* Go */
function goDesk(area, G) {
  G.size ??= 9; G.komi ??= 6.5; G.moves ??= []; G.sims ??= 600; G.engineOn ??= true;
  const { board, side, msg } = boardShell(area);
  const call = (extra = {}) => api("/v1/vapor/go", { size: G.size, komi: G.komi, moves: G.moves, ...extra });
  const refresh = async (engine) => {
    msg.textContent = engine ? `${t("engine")} · ${G.sims} ${t("sims_n")}…` : "";
    try { const r = await call(engine ? { action: "engine", sims: G.sims } : {}); if (engine && r.engine) { G.moves.push(r.engine.move); G.engineMsg = `${t("engine_reply")}: ${r.engine.move === "pass" ? t("pass") : coord(r.engine.move)} · ${t("engine_value")} ${wbNum(r.engine.value, 3)}`; } G.st = r; msg.textContent = G.engineMsg || ""; draw(); }
    catch (e) { msg.textContent = e.message; }
  };
  const coord = (i) => "ABCDEFGHJKLMNOPQRST"[i % G.size] + (G.size - Math.floor(i / G.size));
  const play = async (mv) => { G.moves.push(mv); G.engineMsg = ""; await refresh(false); if (G.engineOn && !G.st.over) await refresh(true); };
  function draw() {
    const st = G.st; if (!st) return;
    const n = st.size, s = Math.min(46, Math.floor(440 / n)), m = s, W = (n - 1) * s + 2 * m, svg = svgEl("svg", { class: "board go", viewBox: `0 0 ${W} ${W}` });
    svg.append(svgEl("rect", { x: 0, y: 0, width: W, height: W, class: "goban", rx: 6 }));
    for (let k = 0; k < n; k++) { svg.append(svgEl("line", { x1: m, x2: m + (n - 1) * s, y1: m + k * s, y2: m + k * s, class: "gl" }), svgEl("line", { y1: m, y2: m + (n - 1) * s, x1: m + k * s, x2: m + k * s, class: "gl" }));
      svg.append(svgText(m + k * s, m - s * 0.55, "ABCDEFGHJKLMNOPQRST"[k], "coord"), svgText(m - s * 0.6, m + k * s + 4, String(n - k), "coord")); }
    const stars = n >= 13 ? [3, 6, 9] : n >= 9 ? [2, 4, 6] : n >= 7 ? [2, 4] : [];
    stars.forEach((a) => stars.forEach((b) => svg.append(svgEl("circle", { cx: m + a * s, cy: m + b * s, r: 3, class: "star" }))));
    const last = G.moves.length ? G.moves[G.moves.length - 1] : null, legal = new Set(st.legal);
    st.board.forEach((v, i) => { const x = m + (i % n) * s, y = m + Math.floor(i / n) * s;
      if (v) { svg.append(svgEl("circle", { cx: x, cy: y, r: s * 0.46, class: v > 0 ? "stone sb" : "stone sw" })); if (i === last) svg.append(svgEl("circle", { cx: x, cy: y, r: s * 0.16, class: v > 0 ? "lastm w" : "lastm b" })); }
      else { const hit = svgEl("circle", { cx: x, cy: y, r: s * 0.48, class: "hit" + (legal.has(i) ? "" : " no") }); if (legal.has(i) && !st.over) hit.onclick = () => play(i); svg.append(hit); } });
    board.replaceChildren(svg);
    const size = sel([5, 7, 9, 13].map((k) => [k, `${k}×${k}`]), G.size); size.onchange = () => { G.size = +size.value; G.moves = []; G.engineMsg = ""; refresh(false); };
    const komi = numIn(G.komi, { min: -20, max: 20, step: 0.5, w: 4 }); komi.onchange = () => { G.komi = +komi.value; refresh(false); };
    const sims = sel([100, 300, 600, 1200, 3000].map((k) => [k, `${k} ${t("sims_n")}`]), G.sims); sims.onchange = () => (G.sims = +sims.value);
    const eng = el("input", { type: "checkbox" }); eng.checked = G.engineOn; eng.onchange = () => (G.engineOn = eng.checked);
    const sc = st.score;
    side.replaceChildren(el("p", { class: "big-kind", text: st.over ? t("game_over") : t("to_move", st.turn > 0 ? t("black_s") : t("white_s")) }),
      stats(stat(t("black_s"), String(sc.black)), stat(t("white_s") + " + komi", wbNum(sc.white + sc.komi, 4)), stat(t("margin"), (sc.margin > 0 ? "B+" : "W+") + wbNum(Math.abs(sc.margin), 4))),
      el("p", { class: "muted", text: "area scoring (Tromp–Taylor), positional superko" }),
      el("div", { class: "row" }, ctrl(t("size"), size), ctrl(t("komi"), komi)),
      el("div", { class: "row" }, sims, el("label", { class: "muted" }, eng, " " + t("engine_on"))),
      el("div", { class: "row" }, btn(t("pass"), "quiet", () => play("pass")), btn(t("engine"), "quiet", () => refresh(true)), btn(t("undo"), "quiet", () => { G.moves.pop(); if (G.engineOn && G.moves.length % 2 === 1) G.moves.pop(); G.engineMsg = ""; refresh(false); }), btn(t("new_game"), "quiet", () => { G.moves = []; G.engineMsg = ""; refresh(false); })),
      G.moves.length ? sect(t("history_moves"), el("p", { class: "mono small", text: G.moves.map((v, i) => `${i + 1}.${v === "pass" ? t("pass") : coord(v)}`).join(" ") })) : "");
  }
  if (G.st) draw(); else refresh(false);
}

/* m,n,k */
function mnkDesk(area, G) {
  G.m ??= 3; G.n ??= 3; G.k ??= 3; G.gravity ??= false; G.moves ??= []; G.engineOn ??= true;
  const { board, side, msg } = boardShell(area);
  const call = () => api("/v1/vapor/mnk", { m: G.m, n: G.n, k: G.k, gravity: G.gravity, moves: G.moves });
  const refresh = async () => { msg.textContent = t("running"); try { G.st = await call(); msg.textContent = ""; draw(); } catch (e) { msg.textContent = e.message; } };
  const play = async (i) => { G.moves.push(i); await refresh(); if (G.engineOn && G.st.outcome == null && G.st.best && G.st.best.length) { G.moves.push(G.st.best[0]); await refresh(); } };
  function draw() {
    const st = G.st; if (!st) return;
    const s = Math.min(70, Math.floor(420 / Math.max(G.m, G.n))), svg = svgEl("svg", { class: "board mnk", viewBox: `0 0 ${G.m * s} ${G.n * s}` }), best = new Set(st.outcome == null ? st.best || [] : []);
    for (let c = 0; c < G.m; c++) for (let r = 0; r < G.n; r++) {
      const i = c * G.n + r, v = st.board[i], x = c * s, y = (G.n - 1 - r) * s, g = svgEl("g", {});
      g.append(svgEl("rect", { x, y, width: s, height: s, class: "cell-m" + (best.has(i) ? " best" : "") }));
      if (v) g.append(svgText(x + s / 2, y + s * 0.68, v > 0 ? "✕" : "◯", v > 0 ? "mk x" : "mk o"));
      if (!v && st.outcome == null) g.onclick = () => { if (G.gravity) { const rr = [...Array(G.n).keys()].find((q) => !st.board[c * G.n + q]); if (rr != null) play(c * G.n + rr); } else play(i); };
      svg.append(g);
    }
    board.replaceChildren(svg);
    const mk = (key, lo, hi) => { const i = numIn(G[key], { min: lo, max: hi, w: 3.5 }); i.onchange = () => { G[key] = Math.max(lo, Math.min(hi, +i.value || lo)); G.moves = []; refresh(); }; return i; };
    const grav = el("input", { type: "checkbox" }); grav.checked = G.gravity; grav.onchange = () => { G.gravity = grav.checked; G.moves = []; refresh(); };
    const eng = el("input", { type: "checkbox" }); eng.checked = G.engineOn; eng.onchange = () => (G.engineOn = eng.checked);
    const who = st.turn > 0 ? "✕" : "◯", v = st.outcome != null ? st.outcome : st.value;
    const verdict = st.outcome != null ? (st.outcome === 0 ? t("draw") : `${st.turn > 0 ? "◯" : "✕"} ${t("win")}`) : `${who} ${t("to_play")}: ${v > 0 ? t("win") : v < 0 ? t("loss") : t("draw")}`;
    side.replaceChildren(el("p", { class: "big-kind", text: verdict }), el("p", { class: "muted", text: st.outcome != null ? "" : st.solved ? t("exact") + " (negamax + transposition table)" : t("by_mcts") }),
      el("div", { class: "row" }, ctrl(t("m_cols"), mk("m", 2, 7)), ctrl(t("n_rows"), mk("n", 2, 7)), ctrl(t("k_line"), mk("k", 2, 5))),
      el("div", { class: "row" }, el("label", { class: "muted" }, grav, " " + t("gravity")), el("label", { class: "muted" }, eng, " " + t("engine_on"))),
      el("div", { class: "row" }, btn(t("undo"), "quiet", () => { G.moves.pop(); if (G.engineOn && G.moves.length % 2 === 1) G.moves.pop(); refresh(); }), btn(t("new_game"), "quiet", () => { G.moves = []; refresh(); })),
      el("p", { class: "muted", text: "3×3, k = 3: draw · 4×4, k = 3: the first player wins · 4×4, k = 4: draw · 7×6, k = 4 with gravity is Connect Four" }));
  }
  if (G.st) draw(); else refresh();
}

/* poker */
function pokerDesk(area, G) {
  G.game ??= "kuhn"; G.its ??= 800;
  const { board, side, msg } = boardShell(area);
  const solve = async () => { msg.textContent = t("running"); try { G.r = await api("/v1/vapor/poker", { game: G.game, iterations: G.its }); msg.textContent = ""; draw(); } catch (e) { msg.textContent = e.message; } };
  function draw() {
    const gs = seg(["kuhn", "leduc"], { kuhn: t("kuhn"), leduc: t("leduc") }, G.game, (g) => { G.game = g; G.its = g === "leduc" ? 60 : 800; G.r = null; draw(); solve(); });
    const its = numIn(G.its, { min: 10, max: G.game === "leduc" ? 200 : 3000, step: 10, w: 5 }); its.onchange = () => (G.its = +its.value);
    side.replaceChildren(gs, el("div", { class: "row" }, ctrl(t("iterations"), its), btn(t("p_solve"), "primary", solve)));
    board.replaceChildren();
    const r = G.r; if (!r) return;
    side.append(stats(stat(t("exploitability"), wbNum(r.exploitability, 3)), stat(t("game_value"), wbNum(r.value, 5)), stat("infosets", String(r.infosets))),
      facts([[t("uniform"), wbNum(r.uniform_exploitability, 4)], ...(r.game === "kuhn" ? [["Kuhn, exact value", "−1/18 = −0.05556"]] : [])]));
    board.append(sect(t("curve"), lineChart({ series: [{ points: r.curve.map((p) => [p.iteration, p.exploitability]), cls: 0, dots: true }], logy: true, levels: [{ y: r.uniform_exploitability, label: t("uniform"), cls: "mark" }], xlab: t("iterations"), w: 640, h: 220 })),
      sect(t("strategy"), el("p", { class: "muted", text: t("infoset_note") }), table([t("infoset"), "σ"], Object.entries(r.strategy).sort((a, b) => a[0].length - b[0].length || a[0].localeCompare(b[0])).map(([k, acts]) => [el("code", { text: k }),
        el("div", { class: "probs" }, ...Object.entries(acts).sort().map(([a, p]) => { const b = el("span", { class: "pbar", title: `${a}: ${wbNum(p, 4)}` }, el("i"), `${a} ${(p * 100).toFixed(1)}%`); b.querySelector("i").style.width = `${p * 100}%`; return b; }))]))));
  }
  draw(); if (!G.r) solve();
}

/* -------------------------------------------------------------- proteins */
const PDB_SAMPLES = { "1A8O": "1A8O · HIV-1 capsid, C-terminal domain (70 res, X-ray)", "1LCD": "1LCD · lac repressor headpiece (51 res, NMR model 1)", "1LCD#2": "1LCD · NMR model 2", "1LCD#3": "1LCD · NMR model 3" };
const SS_COLOR = { H: "var(--ember)", E: "var(--sand)", "-": "var(--silt)" };
function contactMap(L, upper, lower, truth) {
  const k = Math.max(3, Math.floor(360 / L)), cv = el("canvas", { width: String(L * k), height: String(L * k), class: "cmap" }), g = cv.getContext("2d");
  g.fillStyle = css("--slab") || "#ddd"; g.fillRect(0, 0, L * k, L * k);
  g.strokeStyle = css("--rule"); g.beginPath(); g.moveTo(0, 0); g.lineTo(L * k, L * k); g.stroke();
  const T = new Set((truth || []).map(([i, j]) => `${i}-${j}`));
  g.fillStyle = css("--ink"); upper.forEach(([i, j]) => g.fillRect(j * k, i * k, k, k));
  (lower || []).forEach(([i, j]) => { g.fillStyle = T.has(`${i}-${j}`) ? css("--lock") : css("--ember"); g.fillRect(i * k, j * k, k, k); });
  return cv;
}
function seqView(seq, ss) {
  const d = el("div", { class: "seqv" });
  for (let i = 0; i < seq.length; i += 10) { const b = el("span", { class: "blk" }, el("small", { text: String(i + 1) })); for (let j = i; j < Math.min(i + 10, seq.length); j++) { const c = el("span", { text: seq[j], title: `${j + 1} ${seq[j]} · ${ss[j]}` }); c.style.borderBottomColor = SS_COLOR[ss[j]] || "transparent"; b.append(c); } d.append(b); }
  return d;
}
function viewer3d(sets) { const cv = el("canvas", { width: "520", height: "420", class: "mol", title: "drag to rotate · wheel to zoom" }); requestAnimationFrame(() => orbit3d(cv, sets)); return cv; }
const ssColors = (ss) => [...ss].map((c) => ({ H: css("--ember"), E: css("--sand") }[c] || css("--silt")));
const metric = (label, v, good) => el("div", { class: "stat" + (good ? " good" : "") }, el("span", { text: label }), el("b", { text: wbNum(v, 3) }));

mount("protein", (root, P, redraw) => {
  head(root, "protein", "protein_lede");
  P.action ??= "analyse"; P.src ??= { sample: "1A8O" }; P.res ??= {};
  const col = new Intl.Collator(lang === "pt" ? "pt-BR" : "en");
  const acts = { analyse: t("analyse_h"), pipeline: t("pipeline"), compare: t("compare"), align: t("align") };
  const keys = Object.keys(acts).sort((a, b) => col.compare(acts[a], acts[b]));
  const out = el("div", { class: "wb-out", "aria-live": "polite" }), state = el("span", { class: "muted wb-state" });
  const srcSel = sel([...Object.entries(PDB_SAMPLES), ...(P.upload ? [["upload", P.upload.name]] : [])], P.src.sample || "upload");
  const file = el("input", { type: "file", accept: ".pdb,.ent,.txt", hidden: "" });
  file.onchange = async () => { const f = file.files[0]; if (!f) return; P.upload = { name: f.name, pdb: await f.text() }; P.src = { pdb: P.upload.pdb }; P.res = {}; redraw(); };
  srcSel.onchange = () => { P.src = srcSel.value === "upload" ? { pdb: P.upload.pdb } : { sample: srcSel.value }; P.res = {}; redraw(); };
  const call = async (body, key) => { busy(out); state.textContent = t("running"); try { P.res[key] = await api("/v1/vapor/protein", body); P.err = null; } catch (e) { P.err = e.message; } state.textContent = ""; show(); };
  const srcBody = () => (P.src.sample ? { sample: P.src.sample } : { pdb: P.src.pdb });
  const bar = el("div", { class: "wb-bar" }, ctrl(t("samples_pdb"), srcSel), btn(t("upload_pdb"), "quiet", () => file.click()), file);
  const actRow = el("div", { class: "wb-bar" });
  if (P.action === "pipeline") {
    const n = numIn(P.seqs || 2000, { min: 200, max: 4000, step: 100, w: 6.5 }), sd = numIn(P.seed || 1, { min: 1, max: 1e6, w: 5.5 });
    n.onchange = () => (P.seqs = +n.value); sd.onchange = () => (P.seed = +sd.value);
    actRow.append(ctrl(t("seqs"), n), ctrl(t("seed"), sd), btn(t("run"), "primary", () => call({ action: "pipeline", ...srcBody(), sequences: P.seqs || 2000, seed: P.seed || 1 }, "pipeline")));
  } else if (P.action === "compare") {
    P.cmp ??= { model: "1LCD#2", native: "1LCD" };
    const opts = Object.entries(PDB_SAMPLES), a = sel(opts, P.cmp.model), b = sel(opts, P.cmp.native);
    a.onchange = () => (P.cmp.model = a.value); b.onchange = () => (P.cmp.native = b.value);
    actRow.append(ctrl(t("model_s"), a), ctrl(t("native_s"), b), btn(t("run"), "primary", () => call({ action: "compare", model_sample: P.cmp.model, native_sample: P.cmp.native }, "compare")));
  } else if (P.action === "align") {
    P.al ??= { a: "MDIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNWMTETLLVQNANPDCKTILKALGPGATLEEMMTACQGVGGPGHKARVL", b: "PIVQNLQGQMVHQAISPRTLNAWVKVVEEKAFSPEVIPMFSALSEGATPQDLNTMLNTVGGHQAAMQMLKETINEEAAEWDRLHPVHAGPIAPGQMREPRGSDIAGTTSTLQEQIGWMTHNPPIPVGEIYKRWIILGLNKIVRMYSPTSILDIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNWMTETLLVQNANPDCKTILKALGPGATLEEMMTACQGVGGPGHKARVL", mode: "local" };
    const ta = el("textarea", { class: "code", rows: "3", "aria-label": "A" }), tb = el("textarea", { class: "code", rows: "3", "aria-label": "B" }); ta.value = P.al.a; tb.value = P.al.b;
    ta.oninput = () => (P.al.a = ta.value.replace(/\s/g, "")); tb.oninput = () => (P.al.b = tb.value.replace(/\s/g, ""));
    const md = sel([["global", t("global")], ["local", t("local")]], P.al.mode); md.onchange = () => (P.al.mode = md.value);
    actRow.append(el("div", { class: "al-in" }, ta, tb), ctrl(t("mode"), md), btn(t("run"), "primary", () => call({ action: "align", a: P.al.a, b: P.al.b, mode: P.al.mode }, "align")));
  }
  actRow.append(state);
  root.append(seg(keys, acts, P.action, (k) => { P.action = k; redraw(); }), P.action === "align" || P.action === "compare" ? "" : bar, actRow, out);
  function show() {
    out.replaceChildren(); if (P.err) { out.append(notice(P.err)); return; }
    const r = P.res[P.action]; if (!r) return;
    if (P.action === "analyse") {
      const ss = r.secondary, frac = (c) => [...ss].filter((x) => x === c).length / ss.length;
      out.append(el("div", { class: "split2" },
        sect(`${t("analyse_h")} · ${r.length} res · chain ${r.chain}`, viewer3d([{ pts: r.ca, colors: ssColors(ss), width: 3.5 }]), legend([[1, "α-helix"], [2, "β-strand"], [3, "coil"]])),
        sect(t("contact_map"), contactMap(r.length, r.contacts), el("p", { class: "muted", text: `${r.contacts.length} ${t("contacts")} (Cα < 8 Å, |i − j| ≥ 6)` }))),
        stats(stat("α", `${(frac("H") * 100).toFixed(0)} %`), stat("β", `${(frac("E") * 100).toFixed(0)} %`), stat(t("contacts"), String(r.contacts.length))),
        sect(`${t("sequence")} · ${t("secondary_h")}`, seqView(r.sequence, ss)));
    } else if (P.action === "pipeline") {
      const p = r.precision;
      out.append(stats(metric("TM-score", r.tm, r.tm > 0.5), metric("RMSD (Å)", r.rmsd), metric("GDT-TS", r.gdt_ts), metric("lDDT", r.lddt)),
        el("div", { class: "split2" }, sect(t("superposition"), viewer3d([{ pts: r.native, color: css("--lock"), width: 3 }, { pts: r.model, color: css("--ember"), width: 2.2, alpha: 0.9 }]), r.mirrored ? el("p", { class: "muted", text: t("mirrored") }) : ""),
          sect(t("contact_map"), contactMap(r.length, r.true_contacts, r.dca_top, r.true_contacts), el("p", { class: "muted", text: t("upper_true") }))),
        sect(t("precision"), table(["", "precision"], [[t("dca"), `${(p.dca * 100).toFixed(1)} %`], [t("mi"), `${(p.mi * 100).toFixed(1)} %`], [t("chance"), `${(p.chance * 100).toFixed(1)} %`]])),
        el("div", { class: "row" }, btn(t("model_pdb"), "quiet", () => download("vapor-model.pdb", new Blob([r.model_pdb], { type: "chemical/x-pdb" })))));
    } else if (P.action === "compare") {
      out.append(stats(metric("TM-score", r.tm, r.tm > 0.5), metric("RMSD (Å)", r.rmsd), metric("GDT-TS", r.gdt_ts), metric("GDT-HA", r.gdt_ha), metric("lDDT", r.lddt), metric("d₀ (Å)", r.d0)),
        sect(t("superposition"), viewer3d([{ pts: r.native, color: css("--lock"), width: 3 }, { pts: r.model, color: css("--ember"), width: 2.2, alpha: 0.9 }])));
    } else if (P.action === "align") {
      const rows = []; for (let i = 0; i < r.a.length; i += 60) { const a = r.a.slice(i, i + 60), b = r.b.slice(i, i + 60); rows.push(a, [...a].map((c, j) => (c === b[j] && c !== "-" ? "|" : c !== "-" && b[j] !== "-" ? "·" : " ")).join(""), b, ""); }
      out.append(stats(stat("score", String(r.score)), stat(t("identity"), `${(r.identity * 100).toFixed(1)} %`), stat(t("mode"), String(r.mode))), el("pre", { class: "code-out align", text: rows.join("\n") }),
        el("p", { class: "muted", text: "BLOSUM62 · Gotoh affine gaps (open 11, extend 1)" }));
    }
  }
  show();
  if (P.action === "analyse" && !P.res.analyse && !P.err) call({ action: "analyse", ...srcBody() }, "analyse");
});

/* ---------------------------------------------------------------- render */
const EX_RENDER = {
  studio: "# a studio: glass, gold, a red ball, a blue box, a lamp, sky and sun\ncamera pos=0,1.2,4.5 look=0,0.8,0 fov=45\nsky top=0.55,0.7,1.0 bottom=1,1,1\nsun dir=0.4,1,0.3 color=1,0.95,0.85 power=2.5\nplane y=0 mat=diffuse albedo=0.75,0.75,0.75 checker=0.5\nsphere c=0,0.8,0 r=0.8 mat=glass ior=1.5\nsphere c=-1.7,0.6,-0.5 r=0.6 mat=metal albedo=0.95,0.75,0.4 rough=0.08\nsphere c=1.6,0.5,0.3 r=0.5 mat=diffuse albedo=0.8,0.2,0.15\nbox min=-0.4,0,-2 max=0.4,1.2,-1.4 mat=diffuse albedo=0.3,0.5,0.8\nsphere c=0,4,0 r=0.5 mat=emit color=1,0.9,0.8 power=12",
  cornell: "# a closed box lit only by an area light: every bounce matters\ncamera pos=0,1,3.4 look=0,1,0 fov=40\nsky color=0,0,0\nbox min=-1.05,-0.05,-1.05 max=1.05,0,1.05 mat=diffuse albedo=0.75,0.75,0.75\nbox min=-1.05,2,-1.05 max=1.05,2.05,1.05 mat=diffuse albedo=0.75,0.75,0.75\nbox min=-1.05,0,-1.1 max=1.05,2,-1.05 mat=diffuse albedo=0.75,0.75,0.75\nbox min=-1.1,0,-1.05 max=-1.05,2,1.05 mat=diffuse albedo=0.65,0.08,0.06\nbox min=1.05,0,-1.05 max=1.1,2,1.05 mat=diffuse albedo=0.12,0.45,0.15\nbox min=-0.3,1.98,-0.3 max=0.3,1.995,0.3 mat=emit color=1,0.85,0.6 power=18\nbox min=-0.65,0,-0.6 max=-0.05,1.2,0 mat=diffuse albedo=0.75,0.75,0.75\nsphere c=0.45,0.35,0.3 r=0.35 mat=glass ior=1.5",
  metals: "# roughness from mirror to satin, under a soft sky\ncamera pos=0,1.1,5 look=0,0.5,0 fov=38\nsky top=0.35,0.45,0.6 bottom=1,0.95,0.9\nsun dir=-0.5,0.8,0.4 color=1,0.9,0.75 power=3\nplane y=0 mat=diffuse albedo=0.2,0.2,0.22 checker=1\nsphere c=-2.2,0.5,0 r=0.5 mat=metal albedo=0.95,0.95,0.95 rough=0\nsphere c=-1.1,0.5,0 r=0.5 mat=metal albedo=0.95,0.8,0.5 rough=0.05\nsphere c=0,0.5,0 r=0.5 mat=metal albedo=0.95,0.64,0.54 rough=0.15\nsphere c=1.1,0.5,0 r=0.5 mat=metal albedo=0.6,0.7,0.9 rough=0.3\nsphere c=2.2,0.5,0 r=0.5 mat=metal albedo=0.8,0.8,0.8 rough=0.6",
  sunset: "# low sun, long shadows, a glass lens on the ground\ncamera pos=0,0.6,4 look=0,0.4,0 fov=42\nsky top=0.25,0.3,0.55 bottom=1,0.55,0.3\nsun dir=1,0.18,-0.4 color=1,0.6,0.3 power=4\nplane y=0 mat=diffuse albedo=0.8,0.7,0.6\nsphere c=-0.8,0.45,0 r=0.45 mat=glass ior=1.5\nbox min=0.3,0,-0.6 max=1.1,0.9,0.2 mat=diffuse albedo=0.85,0.85,0.8\nsphere c=1.6,0.25,0.8 r=0.25 mat=metal albedo=0.9,0.9,0.9 rough=0.02",
  furnace: "# the white furnace: a sphere of albedo 0.8 in a uniform sky of radiance 1 must read exactly 0.8\ncamera pos=0,0,3 look=0,0,0 fov=40\nsky color=1,1,1\nsphere c=0,0,0 r=1 mat=diffuse albedo=0.8,0.8,0.8",
};
Object.assign(I18N.en, { ex_render: { studio: "Studio: glass, gold, lamp, sun", cornell: "Closed box with an area light", metals: "Metals: roughness series", sunset: "Low sun and a glass lens", furnace: "White furnace (a test you can see)" } });
Object.assign(I18N.pt, { ex_render: { studio: "Estúdio: vidro, ouro, lâmpada, sol", cornell: "Caixa fechada com luz de área", metals: "Metais: série de rugosidade", sunset: "Sol baixo e uma lente de vidro", furnace: "Fornalha branca (um teste que se vê)" } });

mount("render", (root, R, redraw) => {
  head(root, "render", "render_lede");
  R.text ??= EX_RENDER.studio; R.res ??= "480x300";
  const ed = editor("wb-render-ed", 16); ed.value = R.text;
  const [w, h] = R.res.split("x").map(Number);
  const cv = R.canvas && R.canvas.width === w ? R.canvas : el("canvas", { width: String(w), height: String(h), class: "gpu" });
  const counter = el("span", { class: "muted mono" }), errs = el("div"), out = el("div", { class: "wb-out" }), state = el("span", { class: "muted wb-state" });
  let tr = R.canvas === cv ? R.tracer : null;
  if (!tr) { try { R.tracer?.destroy(); tr = GPUTracer.create(cv); R.canvas = cv; R.tracer = tr; } catch (e) { tr = null; R.tracer = null; errs.append(el("p", { class: "notice", text: `${t("unavailable_gpu")} (${e.message})` })); } }
  const apply = () => {
    R.text = ed.value; const p = GPUTracer.parseScene(ed.value);
    errs.replaceChildren(...(p.errors.length ? [el("p", { class: "notice err", text: p.errors.join(" · ") })] : []));
    if (!tr || p.errors.length) return; R.scene = p.scene; tr.setScene(p.scene); if (!tr.playing) tr.render(1); counter.textContent = t("spp_n", tr.frames);
  };
  let deb = 0; ed.oninput = () => { clearTimeout(deb); deb = setTimeout(apply, 250); };
  const onFrame = (n) => { counter.textContent = t("spp_n", n); if (!cv.isConnected) tr.pause(); if (n >= 4096) { tr.pause(); playB.textContent = t("play"); } };
  const playB = btn(tr && tr.playing ? t("pause") : t("play"), "primary", () => { if (!tr) return; if (tr.playing) { tr.pause(); playB.textContent = t("play"); } else { tr.play(onFrame); playB.textContent = t("pause"); } });
  const resSel = sel(["320x200", "480x300", "640x400", "960x600"].map((k) => [k, k.replace("x", " × ")]), R.res); resSel.onchange = () => { R.res = resSel.value; R.tracer?.destroy(); R.tracer = null; R.canvas = null; redraw(); };
  const ex = exampleSelect(EX_RENDER, () => t("ex_render"), (k, text) => { ed.value = R.text = text; apply(); if (tr && !tr.playing) playB.click(); });
  // orbit the camera by dragging the picture: the camera line of the text is rewritten, so the text stays the truth
  let drag = null;
  cv.onpointerdown = (e) => { const c = R.scene?.camera; if (!c) return; const d = c.pos.map((v, i) => v - c.look[i]); drag = { x: e.clientX, y: e.clientY, r: Math.hypot(...d), th: Math.atan2(d[0], d[2]), ph: Math.asin(d[1] / Math.hypot(...d)), c }; cv.setPointerCapture(e.pointerId); };
  const setCam = (th, ph, r, c) => { const pos = [c.look[0] + r * Math.cos(ph) * Math.sin(th), c.look[1] + r * Math.sin(ph), c.look[2] + r * Math.cos(ph) * Math.cos(th)].map((v) => +v.toFixed(3));
    const line = `camera pos=${pos.join(",")} look=${c.look.map((v) => +v.toFixed(3)).join(",")} fov=${c.fov}${c.aperture ? ` aperture=${c.aperture} focus=${c.focus}` : ""}`;
    ed.value = /^\s*camera\b.*$/m.test(ed.value) ? ed.value.replace(/^\s*camera\b.*$/m, line) : line + "\n" + ed.value; apply(); };
  cv.onpointermove = (e) => { if (!drag) return; setCam(drag.th - (e.clientX - drag.x) * 0.008, Math.max(-1.4, Math.min(1.4, drag.ph + (e.clientY - drag.y) * 0.006)), drag.r, drag.c); };
  cv.onpointerup = () => (drag = null);
  cv.onwheel = (e) => { const c = R.scene?.camera; if (!c) return; e.preventDefault(); const d = c.pos.map((v, i) => v - c.look[i]), r = Math.hypot(...d); setCam(Math.atan2(d[0], d[2]), Math.asin(d[1] / r), Math.max(0.3, r * (1 + e.deltaY * 0.001)), c); };
  const refB = btn(t("render_reference"), "quiet", async () => {
    const ww = 240, hh = Math.round((240 * h) / w), spp = Math.max(1, Math.min(256, Math.floor(6e6 / (ww * hh))));
    state.textContent = t("running");
    try { const r = await api("/v1/vapor/render", { text: ed.value, width: ww, height: hh, spp: Math.min(spp, 96) }); R.ref = r; state.textContent = ""; showRef(); } catch (e) { state.textContent = e.message; }
  });
  const furB = btn(t("furnace"), "quiet", async () => { state.textContent = t("running"); try { R.furn = await api("/v1/vapor/render/furnace"); state.textContent = ""; showRef(); } catch (e) { state.textContent = e.message; } });
  const pngB = btn(t("download_png"), "quiet", () => { if (tr) download("vapor-render.png", b64blob(tr.png().split(",")[1], "image/png")); });
  root.append(el("div", { class: "wb-bar" }, ex, playB, btn(t("reset"), "quiet", () => { if (tr) { tr.reset(); tr.render(1); counter.textContent = t("spp_n", tr.frames); } }), ctrl(t("resolution"), resSel), pngB, refB, furB, state),
    el("div", { class: "render-grid" }, el("figure", { class: "gpu-fig" }, cv, el("figcaption", {}, counter, el("span", { class: "muted", text: " · " + t("orbit_hint") }))), el("div", {}, ed, errs)), out);
  function gpuMean() { if (!tr || !tr.frames) return null; const L = tr.readLinear(); let s = 0; for (let i = 0; i < L.w * L.h; i++) s += 0.2126 * L.data[i * 4] + 0.7152 * L.data[i * 4 + 1] + 0.0722 * L.data[i * 4 + 2]; return s / (L.w * L.h); }
  function showRef() {
    out.replaceChildren();
    if (R.ref) { const gm = gpuMean(), rel = gm != null ? Math.abs(gm - R.ref.mean) / Math.max(R.ref.mean, 1e-9) : null;
      out.append(sect(t("reference"), el("div", { class: "split2" }, el("img", { src: R.ref.png, class: "refimg", alt: t("reference"), width: String(R.ref.w * 2) }),
        el("div", {}, stats(stat(t("ref_mean"), wbNum(R.ref.mean, 4)), stat(t("gpu_mean"), wbNum(gm, 4)), stat(t("agree"), rel == null ? "—" : `${(rel * 100).toFixed(2)} %`)),
          facts([["w × h", `${R.ref.w} × ${R.ref.h}`], [t("spp"), R.ref.spp], ["rays", R.ref.rays.toLocaleString()], ["ms", R.ref.ms]]),
          el("p", { class: "muted", text: "Both estimate the same mean radiance; their difference shrinks as N^−½ with the samples of each." })))));
    }
    if (R.furn) { const f = R.furn;
      out.append(sect(t("furnace"), table(["", t("mean_err"), "", ""], [[t("furnace_uniform"), wbNum(f.uniform.max_error, 2), okChip(f.uniform.max_error < 1e-9), `${f.uniform.pixels} px`],
        [t("furnace_gradient"), wbNum(f.gradient.mean_error, 2), okChip(Math.abs(f.gradient.mean_error) < 0.01), `${f.gradient.spp} spp`],
        [t("furnace_control"), wbNum(f.control.mean_error, 2), okChip(Math.abs(f.control.mean_error) >= 0.01, t("caught"), t("not_caught")), t("control")]])));
    }
  }
  apply(); if (tr && !tr.frames) tr.render(1); showRef();
});

/* ================================================================== start */
for (const k of Object.keys(EX_ENG)) for (const ex of Object.keys(EX_ENG[k])) PALETTE_ITEMS.push(() => ({ label: t("ex_eng")[ex], sub: `${t("eng")} · ${t("e_kinds")[k]}`, run: () => { openPanel("eng"); const root = $("wb-eng"); const b = [...root.querySelectorAll(".seg button")].find((x) => x.textContent === t("e_kinds")[k]); if (b) b.click(); const s = $("wb-eng").querySelector(".wb-bar select"); if (s) { s.value = ex; s.dispatchEvent(new Event("change")); } } }));
for (const k of Object.keys(EX_LOGIC)) PALETTE_ITEMS.push(() => ({ label: t("ex_logic")[k], sub: t("logic"), run: () => { openPanel("logic"); const s = $("wb-logic").querySelector(".wb-bar select"); if (s) { s.value = k; s.dispatchEvent(new Event("change")); } } }));
for (const k of ["chess", "go", "mnk", "poker", "shogi"]) PALETTE_ITEMS.push(() => ({ label: t("b_kinds")[k], sub: t("boards"), run: () => { openPanel("boards"); const b = [...$("wb-boards").querySelectorAll(".seg button")].find((x) => x.textContent === t("b_kinds")[k]); if (b) b.click(); } }));
for (const k of Object.keys(EX_RENDER)) PALETTE_ITEMS.push(() => ({ label: t("ex_render")[k], sub: t("render"), run: () => { openPanel("render"); const s = $("wb-render").querySelector(".wb-bar select"); if (s) { s.value = k; s.dispatchEvent(new Event("change")); } } }));
sortNav();
rerender.push(sortNav);
applyLang();
{ const h = location.hash.slice(1); if (h && $("p-" + h)) openPanel(h); }
