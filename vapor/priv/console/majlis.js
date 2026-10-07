"use strict";
/* majlis.js — the 0.16 surfaces (docs/MAJLIS.md, docs/DIWAN.md): the
   conversations (a content-addressed tree — edit, regenerate, the branches
   ‹ i/n ›, fork, rewind, pins, the context the model will see token by
   token, compaction, search, import, export, a share link that dies on
   revocation) and the terminal (the same verbs as `bin/vapor`, jailed, with
   its own files). Every action is one call to the endpoints of
   Vapor.Console.Hall, the same ones the CLI and the TUI reach. Runs after
   the page's scripts and uses their helpers ($, el, t, api, svgEl, shown,
   rerender, PALETTE_ITEMS, applyLang, notice, download). */

(() => {
/* ================================================================== words */
Object.assign(I18N.en, {
  g_talk: "Converse",
  majlis: "Conversations", majlis_s: "branch, edit, fork, share — every word kept",
  diwan: "Terminal", diwan_s: "every vapor command, here",
  mj_lede: "A conversation is a tree, not a list: editing a message or asking for another answer adds a branch beside the old one, and ‹ › walks between them. Forking copies nothing. The panel on the right shows exactly what the model will read — and what it will not — before you send.",
  mj_new: "New conversation", mj_search_ph: "Search every conversation", mj_import: "Import…", mj_import_hint: "a vapor, ChatGPT or Claude export",
  mj_none: "No conversations yet.", mj_pick: "Start a conversation, or pick one on the left.", mj_untitled: "Untitled",
  mj_msg_ph: "Write a message — Enter sends, Shift+Enter breaks the line", mj_send: "Send", mj_reply: "Answer", mj_thinking: "thinking…",
  mj_edit: "Edit", mj_save_send: "Save and send", mj_save: "Save", mj_cancel: "Cancel", mj_regen: "Another answer", mj_pin: "Pin", mj_unpin: "Unpin",
  mj_fork: "Fork from here", mj_rewind: "Continue from here", mj_copy: "Copy", mj_copied: "copied", mj_branch: (i, n) => `branch ${i} of ${n}`,
  mj_prev: "previous branch", mj_next: "next branch", mj_edited: "edited", mj_pinned: "pinned", mj_summarised: "summarised",
  mj_role_user: "you", mj_role_assistant: "model", mj_role_tool: "tool", mj_role_system: "system",
  mj_ctx: "Context", mj_ctx_of: (n, b) => `${n} of ${b} tokens`, mj_ctx_sys: "instructions", mj_ctx_sum: "summary", mj_ctx_sent: "sent", mj_ctx_dropped: "left out",
  mj_ctx_over: "the pinned messages and the last turn alone exceed the budget", mj_ctx_est: "estimated (no tokenizer served)", mj_ctx_exact: "counted by the model's tokenizer",
  mj_dropped_n: (n) => `${n} message${n === 1 ? "" : "s"} left out — pin one to keep it, or compact`, mj_compact: "Compact", mj_uncompact: "Restore the full history",
  mj_compact_hint: "The model summarises everything before the last turn; the summary names the message it covers.", mj_summary: "Summary",
  mj_tree: "Branches", mj_tree_hint: "every message; the path in gold; click one to go there",
  mj_settings: "Settings", mj_title: "title", mj_system: "instructions (system)", mj_model: "model", mj_tools: "tools the model may call", mj_budget: "context budget (tokens)",
  mj_temp: "temperature", mj_maxtok: "longest answer (tokens)", mj_apply: "Apply", mj_applied: "applied", mj_nomodel: "no model configured",
  mj_share: "Share", mj_share_hint: "A read-only link: no console token needed, and revoking kills every link given so far.", mj_shared: "link", mj_revoke: "Revoke every link",
  mj_revoked: "revoked", mj_export: "Export", mj_delete: "Delete conversation", mj_delete_q: "Delete this conversation? Its messages stay only in its forks.",
  mj_hits: (n) => `${n} found`, mj_in: "in", mj_off_path: "on another branch", mj_disabled: "Conversations are not enabled on this server.",
  mj_no_backend: "No model answers here: serve a model, or set VAPOR_MIND. Your messages are kept.",
  dw_lede: "The same commands as bin/vapor, with the same answers, in a session of its own: pipes, redirection and files that live in the session. Nothing runs on the server but vapor itself — no shell, no reading the server's files.",
  dw_ph: "help · wzn check oscillator.wzn · rebis equiv a.net b.net | …", dw_files: "Session files", dw_nofiles: "No files yet — write one with >, the editor or an upload.",
  dw_newfile: "New file", dw_upload: "Upload…", dw_download: "Download", dw_name_q: "File name", dw_samples: "Try", dw_editing: "editing",
  dw_busy: "a command is still running", dw_exit: (c) => `exit ${c}`, dw_jailed: "jailed session"
});
Object.assign(I18N.pt, {
  g_talk: "Conversar",
  majlis: "Conversas", majlis_s: "ramifique, edite, bifurque, compartilhe — cada palavra guardada",
  diwan: "Terminal", diwan_s: "cada comando do vapor, aqui",
  mj_lede: "Uma conversa é uma árvore, não uma lista: editar uma mensagem ou pedir outra resposta acrescenta um ramo ao lado do antigo, e ‹ › anda entre eles. Bifurcar não copia nada. O painel à direita mostra exatamente o que o modelo vai ler — e o que não vai — antes de enviar.",
  mj_new: "Nova conversa", mj_search_ph: "Buscar em todas as conversas", mj_import: "Importar…", mj_import_hint: "uma exportação do vapor, do ChatGPT ou do Claude",
  mj_none: "Nenhuma conversa ainda.", mj_pick: "Comece uma conversa, ou escolha uma à esquerda.", mj_untitled: "Sem título",
  mj_msg_ph: "Escreva uma mensagem — Enter envia, Shift+Enter quebra a linha", mj_send: "Enviar", mj_reply: "Responder", mj_thinking: "pensando…",
  mj_edit: "Editar", mj_save_send: "Salvar e enviar", mj_save: "Salvar", mj_cancel: "Cancelar", mj_regen: "Outra resposta", mj_pin: "Fixar", mj_unpin: "Desafixar",
  mj_fork: "Bifurcar daqui", mj_rewind: "Continuar daqui", mj_copy: "Copiar", mj_copied: "copiado", mj_branch: (i, n) => `ramo ${i} de ${n}`,
  mj_prev: "ramo anterior", mj_next: "próximo ramo", mj_edited: "editada", mj_pinned: "fixada", mj_summarised: "resumida",
  mj_role_user: "você", mj_role_assistant: "modelo", mj_role_tool: "ferramenta", mj_role_system: "sistema",
  mj_ctx: "Contexto", mj_ctx_of: (n, b) => `${n} de ${b} tokens`, mj_ctx_sys: "instruções", mj_ctx_sum: "resumo", mj_ctx_sent: "enviada", mj_ctx_dropped: "fora",
  mj_ctx_over: "as mensagens fixadas e o último turno sozinhos excedem o orçamento", mj_ctx_est: "estimado (nenhum tokenizador servido)", mj_ctx_exact: "contado pelo tokenizador do modelo",
  mj_dropped_n: (n) => `${n} mensage${n === 1 ? "m" : "ns"} fora — fixe uma para mantê-la, ou compacte`, mj_compact: "Compactar", mj_uncompact: "Restaurar o histórico inteiro",
  mj_compact_hint: "O modelo resume tudo antes do último turno; o resumo nomeia a mensagem que cobre.", mj_summary: "Resumo",
  mj_tree: "Ramos", mj_tree_hint: "cada mensagem; o caminho em ouro; clique numa para ir até ela",
  mj_settings: "Ajustes", mj_title: "título", mj_system: "instruções (sistema)", mj_model: "modelo", mj_tools: "ferramentas que o modelo pode chamar", mj_budget: "orçamento de contexto (tokens)",
  mj_temp: "temperatura", mj_maxtok: "resposta mais longa (tokens)", mj_apply: "Aplicar", mj_applied: "aplicado", mj_nomodel: "nenhum modelo configurado",
  mj_share: "Compartilhar", mj_share_hint: "Um link só de leitura: não pede o token do console, e revogar mata todos os links já dados.", mj_shared: "link", mj_revoke: "Revogar todos os links",
  mj_revoked: "revogado", mj_export: "Exportar", mj_delete: "Apagar conversa", mj_delete_q: "Apagar esta conversa? As mensagens ficam só nas bifurcações.",
  mj_hits: (n) => `${n} achada${n === 1 ? "" : "s"}`, mj_in: "em", mj_off_path: "noutro ramo", mj_disabled: "As conversas não estão habilitadas neste servidor.",
  mj_no_backend: "Nenhum modelo responde aqui: sirva um modelo, ou defina VAPOR_MIND. Suas mensagens ficam guardadas.",
  dw_lede: "Os mesmos comandos do bin/vapor, com as mesmas respostas, numa sessão própria: pipes, redirecionamento e arquivos que vivem na sessão. Nada roda no servidor além do próprio vapor — nem shell, nem leitura dos arquivos do servidor.",
  dw_ph: "help · wzn check oscillator.wzn · rebis equiv a.net b.net | …", dw_files: "Arquivos da sessão", dw_nofiles: "Nenhum arquivo ainda — escreva um com >, o editor ou um envio.",
  dw_newfile: "Novo arquivo", dw_upload: "Enviar…", dw_download: "Baixar", dw_name_q: "Nome do arquivo", dw_samples: "Experimente", dw_editing: "editando",
  dw_busy: "um comando ainda está rodando", dw_exit: (c) => `saída ${c}`, dw_jailed: "sessão enjaulada"
});

// the server names a new conversation "untitled" (and a fork "… (fork)"): shown in the page's language
const titleOf = (title) => String(title || "untitled").replace(/^untitled\b/, t("mj_untitled"));
const mjOpt = (v, label, sel) => { const o = el("option", { value: v, text: label }); if (sel) o.selected = true; return o; };
const btn = (label, cls, onclick, extra = {}) => { const b = el("button", Object.assign({ type: "button", class: cls || "quiet", text: label }, extra)); b.onclick = onclick; return b; };
const saveBlob = (name, text, type) => { const a = el("a", { href: URL.createObjectURL(new Blob([text], { type })), download: name }); document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 4000); };
const readText = (f) => new Promise((ok, no) => { const r = new FileReader(); r.onload = () => ok(String(r.result)); r.onerror = no; r.readAsText(f); });
const readB64 = (f) => new Promise((ok, no) => { const r = new FileReader(); r.onload = () => ok(String(r.result).split(",")[1] || ""); r.onerror = no; r.readAsDataURL(f); });
async function getText(path) { const r = await fetch(path); const text = await r.text(); if (!r.ok) { let m = r.statusText; try { m = JSON.parse(text).error.message; } catch (_) {} throw new Error(m); } return text; }

/* a message's text: paragraphs as typed, fenced code as code — never HTML from the model */
function richText(s) {
  const wrap = el("div", { class: "mj-text" });
  String(s || "").split(/(```[^\n]*\n[\s\S]*?```)/).forEach((part) => {
    const m = /^```([^\n]*)\n([\s\S]*?)```$/.exec(part);
    if (m) wrap.append(el("pre", { class: "mj-code", "data-lang": m[1].trim() }, el("code", { text: m[2] })));
    else if (part) wrap.append(el("div", { class: "mj-para", text: part }));
  });
  return wrap;
}

/* ========================================================= conversations */
const MJ = { threads: [], models: { names: [] }, tools: [], cur: null, thread: null, path: null, ctx: null, tree: null, busy: false, disabled: null, hits: null, shareUrl: null, editing: null, err: null };

function buildMajlis(root) {
  root.replaceChildren(
    el("h2", { text: t("majlis") }), el("p", { class: "lede", text: t("mj_lede") }),
    el("div", { class: "mj" },
      el("aside", { class: "mj-list", id: "mj-list", "aria-label": t("majlis") }),
      el("div", { class: "mj-talk", id: "mj-talk" }),
      el("aside", { class: "mj-side", id: "mj-side", "aria-label": t("mj_ctx") })));
  drawList(); drawTalk(); drawSide();
  loadThreads();
}

async function loadThreads(open) {
  try {
    const j = await api("/v1/vapor/threads");
    Object.assign(MJ, { threads: j.threads, models: j.models, tools: j.tools, disabled: null });
    if (open) await openThread(open); else drawList();
  } catch (e) { MJ.disabled = e.message; drawList(); drawTalk(); drawSide(); }
}

async function openThread(id, keepHits) {
  MJ.cur = id; MJ.shareUrl = null; MJ.editing = null; MJ.err = null; if (!keepHits) MJ.hits = null;
  await refresh();
}

async function refresh() {
  if (!MJ.cur) { drawList(); drawTalk(); drawSide(); return; }
  const id = encodeURIComponent(MJ.cur);
  try {
    const [th, ctx, tree] = await Promise.all([api(`/v1/vapor/threads/${id}`), api(`/v1/vapor/threads/${id}/context`), api(`/v1/vapor/threads/${id}/tree`)]);
    Object.assign(MJ, { thread: th.thread, path: th.path, ctx, tree });
  } catch (e) { MJ.err = e.message; }
  drawList(); drawTalk(); drawSide();
}

async function op(name, body = {}) {
  return api(`/v1/vapor/threads/${encodeURIComponent(MJ.cur)}/${name}`, body);
}

// an action that may run a model: the page shows it thinking, then the new state (or why not)
async function act(f) {
  if (MJ.busy) return;
  MJ.busy = true; MJ.err = null; drawTalk();
  try { await f(); } catch (e) { MJ.err = e.message; }
  MJ.busy = false;
  await loadThreads(); await refresh();
}

function drawList() {
  const box = $("mj-list"); if (!box) return;
  const search = el("input", { type: "search", class: "mj-search", placeholder: t("mj_search_ph"), "aria-label": t("mj_search_ph") });
  if (MJ.hits) search.value = MJ.hits.q;
  search.onkeydown = async (e) => {
    if (e.key !== "Enter") return;
    const q = search.value.trim();
    if (!q) { MJ.hits = null; drawList(); return; }
    try { const j = await api("/v1/vapor/chat/search", { query: q }); MJ.hits = { q, hits: j.hits }; } catch (err) { MJ.hits = { q, hits: [], err: err.message }; }
    drawList(); setTimeout(() => $("mj-list")?.querySelector(".mj-search")?.focus(), 0);
  };
  const file = el("input", { type: "file", accept: ".json,.zip,application/json", hidden: "" });
  file.onchange = async () => {
    const f = file.files[0]; if (!f) return;
    try { const j = await api("/v1/vapor/threads/import", { data: await readText(f) }); await loadThreads(j.threads[0]); }
    catch (e) { MJ.err = e.message; drawTalk(); }
  };
  const head = el("div", { class: "mj-list-head" },
    btn(t("mj_new"), "primary", async () => { try { const j = await api("/v1/vapor/threads", {}); await loadThreads(j.id); $("mj-input")?.focus(); } catch (e) { MJ.err = e.message; drawTalk(); } }, { id: "mj-new" }),
    btn(t("mj_import"), "quiet", () => file.click(), { title: t("mj_import_hint") }), file);
  const items = [];
  if (MJ.disabled) items.push(el("p", { class: "muted", text: t("mj_disabled") }));
  else if (MJ.hits) {
    items.push(el("p", { class: "muted small", text: MJ.hits.err || t("mj_hits", MJ.hits.hits.length) }));
    for (const h of MJ.hits.hits) {
      const owner = h.threads[0];
      const b = el("button", { type: "button", class: "mj-hit" }, el("span", { class: "mj-role", text: t("mj_role_" + h.role) }), el("span", { text: h.snippet }),
        el("small", { class: "muted", text: owner ? `${t("mj_in")} ${titleOf((MJ.threads.find((x) => x.id === owner) || {}).title)}` : t("mj_off_path") }));
      b.disabled = !owner;
      b.onclick = async () => { await openThread(owner, true); };
      items.push(b);
    }
  } else if (!MJ.threads.length) items.push(el("p", { class: "muted", text: t("mj_none") }));
  else for (const th of MJ.threads) {
    const b = el("button", { type: "button", class: "mj-thread" + (th.id === MJ.cur ? " on" : ""), "aria-current": th.id === MJ.cur ? "true" : "false" },
      el("b", { text: titleOf(th.title) }), el("span", { class: "muted", text: th.preview || "" }),
      el("small", { class: "muted", text: `${th.messages} · ${new Date(th.updated).toLocaleString(lang === "pt" ? "pt-BR" : "en")}${th.forked_from ? " · ⑂" : ""}` }));
    b.onclick = () => openThread(th.id);
    items.push(b);
  }
  box.replaceChildren(head, search, el("div", { class: "mj-threads" }, ...items));
}

function drawTalk() {
  const box = $("mj-talk"); if (!box) return;
  if (MJ.disabled) { box.replaceChildren(notice(`${t("mj_disabled")} ${MJ.disabled}`)); return; }
  if (!MJ.cur || !MJ.path) { box.replaceChildren(el("p", { class: "muted mj-empty", text: t("mj_pick") }), ...(MJ.err ? [notice(MJ.err)] : [])); return; }
  const msgs = MJ.path.messages.map(drawMessage);
  const last = MJ.path.messages[MJ.path.messages.length - 1];
  const log = el("div", { class: "mj-log", id: "mj-log", "aria-live": "polite" }, ...msgs);
  if (MJ.busy) log.append(el("div", { class: "mj-msg assistant pending" }, el("span", { class: "muted pulse", text: t("mj_thinking") })));
  if (MJ.err) log.append(notice(/no model|no backend|not configured/i.test(MJ.err) ? t("mj_no_backend") : MJ.err));
  const ta = el("textarea", { id: "mj-input", class: "mj-input", rows: "3", placeholder: t("mj_msg_ph"), "aria-label": t("mj_msg_ph") });
  const send = async () => { const text = ta.value.trim(); if (!text || MJ.busy) return; ta.value = ""; await act(() => op("say", { text })); $("mj-input")?.focus(); };
  ta.onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } };
  const row = el("div", { class: "mj-compose-row" }, btn(t("mj_send"), "primary", send, { id: "mj-send" }));
  if (last && last.role === "user" && !MJ.busy) row.prepend(btn(t("mj_reply"), "quiet", () => act(() => op("reply")), { id: "mj-reply" }));
  const comp = el("form", { class: "mj-compose" }, ta, row);
  comp.onsubmit = (e) => { e.preventDefault(); send(); };
  box.replaceChildren(el("h3", { class: "mj-title", text: titleOf(MJ.thread && MJ.thread.title) }), log, comp);
  log.scrollTop = log.scrollHeight;
}

function drawMessage(m) {
  const pinned = MJ.thread && (MJ.thread.settings.pins || []).includes(m.id);
  const card = el("article", { class: `mj-msg ${m.role}${pinned ? " pinned" : ""}`, "data-id": m.id });
  const meta = el("div", { class: "mj-meta" }, el("span", { class: "mj-role", text: t("mj_role_" + m.role) }));
  if (m.siblings.length > 1) {
    const i = m.siblings.indexOf(m.id);
    const go = (d) => () => act(() => op("switch", { node: m.siblings[i + d] }));
    const prev = btn("‹", "mj-nav", go(-1), { "aria-label": t("mj_prev") }); prev.disabled = i === 0 || MJ.busy;
    const next = btn("›", "mj-nav", go(1), { "aria-label": t("mj_next") }); next.disabled = i === m.siblings.length - 1 || MJ.busy;
    meta.append(el("span", { class: "mj-branch", title: t("mj_branch", i + 1, m.siblings.length) }, prev, el("span", { text: `${i + 1}/${m.siblings.length}` }), next));
  }
  if (m.meta && m.meta.edited_from) meta.append(el("span", { class: "muted", text: t("mj_edited") }));
  if (pinned) meta.append(el("span", { class: "mj-tag", text: t("mj_pinned") }));
  if (MJ.thread && MJ.thread.settings.summary && covered(m.id)) meta.append(el("span", { class: "mj-tag dim", text: t("mj_summarised") }));
  if (m.meta && m.meta.backend) meta.append(el("small", { class: "muted", text: `${m.meta.backend}${m.meta.context ? ` · ${m.meta.context.tokens} tok` : ""}${m.meta.ms != null ? ` · ${m.meta.ms} ms` : ""}` }));
  card.append(meta);
  if (MJ.editing === m.id) {
    const ta = el("textarea", { class: "mj-input", rows: String(Math.min(14, Math.max(3, m.content.split("\n").length + 1))) }); ta.value = m.content;
    const save = (reply) => act(() => op("edit", { node: m.id, text: ta.value, reply })).then(() => { MJ.editing = null; drawTalk(); });
    card.append(ta, el("div", { class: "mj-actions" },
      ...(m.role === "user" ? [btn(t("mj_save_send"), "primary", () => save(true))] : []),
      btn(t("mj_save"), m.role === "user" ? "quiet" : "primary", () => save(false)), btn(t("mj_cancel"), "quiet", () => { MJ.editing = null; drawTalk(); })));
    setTimeout(() => ta.focus(), 0);
    return card;
  }
  card.append(richText(m.content));
  const acts = el("div", { class: "mj-actions" },
    btn(t("mj_edit"), "link", () => { MJ.editing = m.id; drawTalk(); }),
    ...(m.role === "assistant" ? [btn(t("mj_regen"), "link", () => act(() => op("regenerate", { node: m.id })))] : []),
    btn(pinned ? t("mj_unpin") : t("mj_pin"), "link", () => act(() => op("pin", { node: m.id, on: !pinned }))),
    btn(t("mj_rewind"), "link", () => act(() => op("rewind", { node: m.id }))),
    btn(t("mj_fork"), "link", async () => { try { const j = await op("fork", { node: m.id }); await loadThreads(j.id); } catch (e) { MJ.err = e.message; drawTalk(); } }),
    btn(t("mj_copy"), "link", async (e) => { const b = e.currentTarget; try { await navigator.clipboard.writeText(m.content); b.textContent = t("mj_copied"); } catch (_) {} }));
  acts.querySelectorAll("button").forEach((b) => (b.disabled = MJ.busy));
  card.append(acts);
  return card;
}

function covered(id) {
  const s = MJ.thread.settings.summary; if (!s) return false;
  const msgs = MJ.path.messages, k = msgs.findIndex((x) => x.id === s.upto);
  return k >= 0 && msgs.findIndex((x) => x.id === id) <= k;
}

function drawSide() {
  const box = $("mj-side"); if (!box) return;
  if (!MJ.cur || !MJ.thread || MJ.disabled) { box.replaceChildren(); return; }
  box.replaceChildren(sideContext(), sideTree(), sideSettings(), sideShare());
}

function sideContext() {
  const c = MJ.ctx, sec = el("section", { class: "mj-sec", id: "mj-ctx" }, el("h3", { text: t("mj_ctx") }));
  if (!c) return sec;
  const budget = c.budget || 1, scale = (n) => `${Math.max(0.4, (100 * n) / Math.max(budget, c.tokens))}%`;
  const bar = el("div", { class: "mj-ctxbar", role: "img", "aria-label": t("mj_ctx_of", c.tokens, c.budget) });
  const seg = (cls, n, title) => { const s = el("i", { class: cls, title: `${title} · ${n}` }); s.style.width = scale(n); bar.append(s); };
  if (c.system_tokens) seg("sys", c.system_tokens, c.summary ? `${t("mj_ctx_sys")} + ${t("mj_ctx_sum")}` : t("mj_ctx_sys"));
  for (const it of c.items) if (it.status === "sent" || it.status === "pinned") seg(`${it.role}${it.status === "pinned" ? " pin" : ""}`, it.tokens, `${t("mj_role_" + it.role)} · ${it.status === "pinned" ? t("mj_pinned") : t("mj_ctx_sent")}`);
  const budgetMark = el("b", { class: "mj-budget" }); budgetMark.style.left = `${Math.min(100, (100 * budget) / Math.max(budget, c.tokens))}%`; bar.append(budgetMark);
  sec.append(el("p", { class: "mj-ctxnum" }, el("b", { text: t("mj_ctx_of", c.tokens, c.budget) }), el("small", { class: "muted", text: ` · ${c.exact ? t("mj_ctx_exact") : t("mj_ctx_est")}` })), bar);
  if (c.over) sec.append(el("p", { class: "warn-t small", text: t("mj_ctx_over") }));
  const dropped = c.items.filter((x) => x.status === "dropped");
  if (dropped.length) sec.append(el("p", { class: "small muted", text: t("mj_dropped_n", dropped.length) }));
  const s = MJ.thread.settings.summary;
  if (s) sec.append(el("details", { class: "mj-summary" }, el("summary", { text: `${t("mj_summary")} (${s.covers})` }), el("div", { class: "mj-para small", text: s.text })),
                    btn(t("mj_uncompact"), "quiet", () => act(() => op("uncompact"))));
  else if (MJ.path.messages.length >= 3) sec.append(btn(t("mj_compact"), "quiet", () => act(() => op("compact", {})), { title: t("mj_compact_hint"), id: "mj-compact" }));
  return sec;
}

/* the tree, layered by depth: the path in gold; a click goes to the newest answer under a message */
function sideTree() {
  const sec = el("section", { class: "mj-sec" }, el("h3", { text: t("mj_tree") }), el("p", { class: "small muted", text: t("mj_tree_hint") }));
  const nodes = (MJ.tree && MJ.tree.nodes) || [];
  if (nodes.length < 2) return sec;
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n])), depth = {}, col = {};
  let next = 0;
  const roots = nodes.filter((n) => !n.parent || !byId[n.parent]);
  const place = (n, d) => { depth[n.id] = d; const kids = n.children.filter((c) => byId[c]); if (!kids.length) col[n.id] = next++; else { kids.forEach((k) => place(byId[k], d + 1)); col[n.id] = kids.map((k) => col[k]).reduce((a, b) => a + b, 0) / kids.length; } };
  roots.forEach((r) => place(r, 0));
  const D = Math.max(...Object.values(depth)) + 1, W = 230, step = Math.min(22, W / Math.max(1, next)), H = 14 + D * 18;
  const X = (id) => 10 + col[id] * step, Y = (id) => 8 + depth[id] * 18;
  const svg = svgEl("svg", { viewBox: `0 0 ${Math.max(W, 20 + next * step)} ${H}`, class: "mj-tree", role: "img", "aria-label": t("mj_tree") });
  for (const n of nodes) for (const c of n.children) if (byId[c]) svg.append(svgEl("line", { x1: X(n.id), y1: Y(n.id), x2: X(c), y2: Y(c), class: byId[c].on_path && n.on_path ? "on" : "" }));
  for (const n of nodes) {
    const dot = svgEl("circle", { cx: X(n.id), cy: Y(n.id), r: n.id === MJ.tree.head ? 5 : 3.6, class: `${n.role}${n.on_path ? " on" : ""}${n.id === MJ.tree.head ? " head" : ""}`, tabindex: "0" });
    const tt = svgEl("title", {}); tt.textContent = `${t("mj_role_" + n.role)}: ${String(n.content).slice(0, 80)}`; dot.append(tt);
    const go = () => act(() => op("switch", { node: n.id }));
    dot.addEventListener("click", go); dot.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    svg.append(dot);
  }
  sec.append(svg);
  return sec;
}

function sideSettings() {
  const s = MJ.thread.settings, g = s.gen || {};
  const title = el("input", { class: "op-text", value: MJ.thread.title || "" });
  const sys = el("textarea", { class: "mj-input", rows: "3" }); sys.value = s.system || "";
  const model = el("select", {}, ...(MJ.models.names.length ? [mjOpt("", MJ.models.default ? `${MJ.models.default} (default)` : "—", !s.model), ...MJ.models.names.map((n) => mjOpt(n, n, s.model === n))] : [mjOpt("", t("mj_nomodel"), true)]));
  const budget = el("input", { type: "number", class: "op-num", min: "64", max: "1000000", value: String(s.budget || "") });
  const temp = el("input", { type: "number", class: "op-num", min: "0", max: "2", step: "0.1", value: String(g.temperature ?? 0.7) });
  const maxtok = el("input", { type: "number", class: "op-num", min: "16", max: "32768", value: String(g.max_tokens ?? 1024) });
  const tools = el("div", { class: "mj-tools" }, ...MJ.tools.map((n) => { const c = el("input", { type: "checkbox", value: n }); c.checked = (s.tools || []).includes(n); return el("label", {}, c, el("code", { text: n })); }));
  const state = el("small", { class: "muted" });
  const apply = btn(t("mj_apply"), "primary", async () => {
    try {
      await op("settings", { title: title.value, system: sys.value, model: model.value || null, budget: +budget.value || undefined,
        tools: [...tools.querySelectorAll("input:checked")].map((c) => c.value), gen: Object.assign({}, g, { temperature: +temp.value, max_tokens: +maxtok.value }) });
      state.textContent = t("mj_applied"); await loadThreads(); await refresh();
    } catch (e) { state.textContent = e.message; }
  }, { id: "mj-apply" });
  const f = (label, input) => el("label", { class: "op-field" }, el("span", { text: label }), input);
  return el("details", { class: "mj-sec" }, el("summary", {}, el("h3", { text: t("mj_settings") })),
    el("div", { class: "op-form" }, f(t("mj_title"), title), f(t("mj_system"), sys), f(t("mj_model"), model), el("div", { class: "op-row" }, f(t("mj_budget"), budget), f(t("mj_temp"), temp), f(t("mj_maxtok"), maxtok)),
      f(t("mj_tools"), tools), el("div", {}, apply, " ", state)));
}

function sideShare() {
  const sec = el("section", { class: "mj-sec" }, el("h3", { text: `${t("mj_share")} · ${t("mj_export")}` }), el("p", { class: "small muted", text: t("mj_share_hint") }));
  const out = el("div", { class: "mj-share", id: "mj-share" });
  if (MJ.shareUrl) out.append(el("a", { href: MJ.shareUrl, target: "_blank", rel: "noopener noreferrer", text: location.origin + MJ.shareUrl }));
  const id = encodeURIComponent(MJ.cur), name = (MJ.thread.title || "conversation").replace(/[^\p{L}\p{N}._-]+/gu, "-").slice(0, 60);
  sec.append(el("div", { class: "mj-actions" },
    btn(t("mj_share"), "quiet", async () => { try { const j = await op("share"); MJ.shareUrl = j.url; drawSide(); } catch (e) { out.replaceChildren(notice(e.message)); } }, { id: "mj-share-btn" }),
    btn(t("mj_revoke"), "quiet", async () => { try { await op("revoke"); MJ.shareUrl = null; drawSide(); $("mj-share").replaceChildren(el("span", { class: "muted", text: t("mj_revoked") })); } catch (e) { out.replaceChildren(notice(e.message)); } }, { id: "mj-revoke" }),
    btn("JSON", "quiet", async () => { try { saveBlob(`${name}.vapor.json`, await getText(`/v1/vapor/threads/${id}/export?format=json`), "application/json"); } catch (e) { out.replaceChildren(notice(e.message)); } }),
    btn("Markdown", "quiet", async () => { try { saveBlob(`${name}.md`, await getText(`/v1/vapor/threads/${id}/export?format=markdown`), "text/markdown"); } catch (e) { out.replaceChildren(notice(e.message)); } })),
    out,
    btn(t("mj_delete"), "danger quiet", async () => { if (!confirm(t("mj_delete_q"))) return; try { await op("delete"); MJ.cur = null; MJ.thread = null; MJ.path = null; await loadThreads(); refresh(); } catch (e) { out.replaceChildren(notice(e.message)); } }));
  return sec;
}

/* ============================================================== terminal */
const DW = { session: null, files: [], hist: [], hi: -1, open: null, busy: false };
const SAMPLES = {
  "oscillator.wzn": `; A harmonic oscillator: its energy is conserved along the flow — proved over ℚ.
(claim kinetic (root H-s-b) (wazn fail)
  (inputs (v q))
  (body (* 1/2 v v)))

(claim energy (root H-f-Z) (wazn burhan)
  (inputs (x q) (v q))
  (field (x v) (v (- x)))
  (proof conserved)
  (body (+ (kinetic v) (* 1/2 x x))))

; the same law, damped: the decider refuses it and says where
(claim damped-energy (root H-f-Z) (wazn burhan)
  (inputs (x q) (v q))
  (field (x v) (v (- (- x) (* 1/10 v))))
  (proof conserved)
  (body (+ (kinetic v) (* 1/2 x x))))
`,
  "and.net": "# a AND b, two ways\ninput a b\noutput y\ny = a & b\n",
  "and2.net": "input a b\noutput y\nn = ~(a & b)\ny = ~n\n"
};
const SAMPLE_LINES = [["help", null], ["wzn check oscillator.wzn", "oscillator.wzn"], ["wzn show oscillator.wzn --arabic", "oscillator.wzn"], ["wzn abjad كتب", null], ["rebis equiv and.net and2.net", "and.net,and2.net"], ["ls", null]];

/* ANSI SGR → spans (the commands colour for a terminal; the page draws the same colours) */
function ansi(text) {
  const frag = document.createDocumentFragment(), re = /\x1b\[([0-9;]*)m/g;
  let cls = new Set(), last = 0, m;
  const push = (s) => { if (!s) return; frag.append(cls.size ? el("span", { class: [...cls].map((c) => "a-" + c).join(" "), text: s }) : document.createTextNode(s)); };
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index)); last = re.lastIndex;
    for (const c of (m[1] || "0").split(";").map(Number)) {
      if (c === 0) cls = new Set(); else if (c === 1) cls.add("b"); else if (c === 2) cls.add("dim"); else if (c === 3) cls.add("i"); else if (c === 4) cls.add("u");
      else if ((c >= 30 && c <= 37) || (c >= 90 && c <= 97)) { [...cls].filter((x) => /^c\d/.test(x)).forEach((x) => cls.delete(x)); cls.add("c" + (c % 10)); }
      else if (c === 39) [...cls].filter((x) => /^c\d/.test(x)).forEach((x) => cls.delete(x));
    }
  }
  push(text.slice(last));
  return frag;
}

function buildDiwan(root) {
  const screen = el("div", { class: "dw-screen", id: "dw-screen", role: "log", "aria-live": "polite", tabindex: "0" });
  const input = el("input", { class: "dw-input", id: "dw-input", autocomplete: "off", autocapitalize: "off", spellcheck: "false", placeholder: t("dw_ph"), "aria-label": t("diwan") });
  const status = el("span", { class: "dw-status muted", id: "dw-status", text: t("dw_jailed") });
  input.onkeydown = async (e) => {
    if (e.key === "Enter") { e.preventDefault(); const line = input.value; input.value = ""; await run(line); }
    else if (e.key === "ArrowUp" && DW.hist.length) { e.preventDefault(); DW.hi = DW.hi < 0 ? DW.hist.length - 1 : Math.max(0, DW.hi - 1); input.value = DW.hist[DW.hi]; }
    else if (e.key === "ArrowDown" && DW.hi >= 0) { e.preventDefault(); DW.hi = DW.hi + 1 >= DW.hist.length ? -1 : DW.hi + 1; input.value = DW.hi < 0 ? "" : DW.hist[DW.hi]; }
    else if (e.key === "Tab") { e.preventDefault(); await complete(input); }
    else if (e.key === "l" && e.ctrlKey) { e.preventDefault(); screen.replaceChildren(); }
  };
  screen.onclick = () => { if (!getSelection().toString()) input.focus(); };
  const samples = el("div", { class: "dw-samples" }, el("span", { class: "muted", text: t("dw_samples") }),
    ...SAMPLE_LINES.map(([line, files]) => btn(line, "vial", async () => { if (files) for (const f of files.split(",")) await writeFile(f, SAMPLES[f]); input.value = ""; await run(line); })));
  root.replaceChildren(el("h2", { text: t("diwan") }), el("p", { class: "lede", text: t("dw_lede") }), samples,
    el("div", { class: "dw" },
      el("div", { class: "dw-term" }, screen, el("div", { class: "dw-prompt" }, el("span", { class: "dw-ps", text: "vapor ›" }), input), status),
      el("aside", { class: "dw-side", id: "dw-side" })));
  drawFiles();
}

async function run(line) {
  const screen = $("dw-screen");
  if (!line.trim()) return;
  if (DW.busy) { $("dw-status").textContent = t("dw_busy"); return; }
  if (line.trim() === "clear") { screen.replaceChildren(); DW.hist.push(line); DW.hi = -1; return; }
  DW.hist.push(line); DW.hi = -1; DW.busy = true;
  const entry = el("div", { class: "dw-entry" }, el("div", { class: "dw-cmd" }, el("span", { class: "dw-ps", text: "vapor › " }), line));
  const pending = el("div", { class: "muted pulse", text: t("running") });
  entry.append(pending); screen.append(entry); screen.scrollTop = screen.scrollHeight;
  try {
    const j = await api("/v1/vapor/diwan", { session: DW.session, line });
    DW.session = j.session; DW.files = j.files || [];
    pending.remove();
    if (j.out) entry.append(el("pre", { class: "dw-out" }, ansi(j.out)));
    if (j.err) entry.append(el("pre", { class: "dw-err" }, ansi(j.err)));
    if (j.code !== 0) entry.append(el("div", { class: "dw-code", text: t("dw_exit", j.code) }));
    $("dw-status").textContent = `${t("dw_jailed")} · ${t("dw_exit", j.code)}`;
  } catch (e) { pending.remove(); entry.append(el("pre", { class: "dw-err", text: e.message })); }
  DW.busy = false; screen.scrollTop = screen.scrollHeight;
  drawFiles();
}

async function complete(input) {
  try {
    const j = await api("/v1/vapor/diwan/complete", { session: DW.session, line: input.value });
    const c = j.completions;
    if (!c.length) return;
    const words = input.value.split(/\s+/), lastw = words.pop() || "";
    const common = c.reduce((a, b) => { let i = 0; while (i < a.length && a[i] === b[i]) i++; return a.slice(0, i); });
    if (c.length === 1) input.value = [...words, c[0]].join(" ") + " ";
    else if (common.length > lastw.length) input.value = [...words, common].join(" ");
    else { const s = $("dw-screen"); s.append(el("div", { class: "dw-out muted", text: c.join("   ") })); s.scrollTop = s.scrollHeight; }
  } catch (_) {}
}

async function writeFile(name, text, base64) {
  const j = await api("/v1/vapor/diwan/file", Object.assign({ session: DW.session, name }, base64 ? { base64 } : { text }));
  DW.session = j.session; DW.files = j.files; drawFiles();
}

function drawFiles() {
  const box = $("dw-side"); if (!box) return;
  const up = el("input", { type: "file", hidden: "", multiple: "" });
  up.onchange = async () => { for (const f of up.files) { try { await writeFile(f.name, null, await readB64(f)); } catch (e) { box.append(notice(e.message)); } } };
  const list = DW.files.length
    ? el("ul", { class: "dw-files" }, ...DW.files.map((f) => { const li = el("li", {}, btn(f, "link" + (DW.open && DW.open.name === f ? " on" : ""), () => openFile(f))); return li; }))
    : el("p", { class: "small muted", text: t("dw_nofiles") });
  const parts = [el("h3", { text: t("dw_files") }), list,
    el("div", { class: "mj-actions" },
      btn(t("dw_newfile"), "quiet", () => { const n = prompt(t("dw_name_q")); if (n) { DW.open = { name: n, text: "" }; drawFiles(); } }),
      btn(t("dw_upload"), "quiet", () => up.click()), up)];
  if (DW.open) {
    const ta = el("textarea", { class: "code dw-editor", id: "dw-editor", spellcheck: "false", rows: "16" }); ta.value = DW.open.text;
    const state = el("small", { class: "muted" });
    const save = async () => { try { await writeFile(DW.open.name, ta.value); DW.open.text = ta.value; state.textContent = "✓"; } catch (e) { state.textContent = e.message; } };
    ta.onkeydown = (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "s") { e.preventDefault(); save(); } if (e.key === "Tab") { e.preventDefault(); const s = ta.selectionStart; ta.setRangeText("  ", s, ta.selectionEnd, "end"); } };
    parts.push(el("div", { class: "dw-edit" }, el("p", { class: "small" }, el("span", { class: "muted", text: `${t("dw_editing")} ` }), el("code", { text: DW.open.name })), ta,
      el("div", { class: "mj-actions" }, btn(t("mj_save"), "primary", save, { id: "dw-save" }), btn(t("dw_download"), "quiet", () => saveBlob(DW.open.name, ta.value, "text/plain")),
        btn(t("mj_cancel"), "quiet", () => { DW.open = null; drawFiles(); }), state)));
  }
  box.replaceChildren(...parts);
}

async function openFile(name) {
  try {
    const j = await api(`/v1/vapor/diwan/file?session=${encodeURIComponent(DW.session)}&name=${encodeURIComponent(name)}`);
    DW.open = { name, text: j.text != null ? j.text : atob(j.base64 || "") }; drawFiles();
  } catch (e) { $("dw-side").append(notice(e.message)); }
}

/* ================================================================ wiring */
const BUILT = {};
const DESKS = { majlis: ["mj", buildMajlis], diwan: ["dw", buildDiwan] };
Object.entries(DESKS).forEach(([k, [mount, f]]) => shown.set("p-" + k, () => { BUILT[k] = true; f($(mount)); }));
// a language switch redraws the words; the state (threads, the terminal's screen and files) stays
rerender.push(() => {
  if (BUILT.majlis) { const root = $("mj"); const h = root.querySelector("h2"), p = root.querySelector(".lede"); if (h) h.textContent = t("majlis"); if (p) p.textContent = t("mj_lede"); drawList(); drawTalk(); drawSide(); }
  if (BUILT.diwan) { const root = $("dw"); root.querySelector("h2").textContent = t("diwan"); root.querySelector(".lede").textContent = t("dw_lede"); $("dw-input").placeholder = t("dw_ph"); drawFiles(); }
});
PALETTE_ITEMS.push(() => ({ label: t("mj_new"), sub: t("majlis"), run: async () => { openPanel("majlis"); try { const j = await api("/v1/vapor/threads", {}); await loadThreads(j.id); } catch (_) {} } }));
applyLang();
})();
