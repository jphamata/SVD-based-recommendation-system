"use strict";
/* mercado.js — the console's 0.13 desks (docs/CONSOLE.md §0.13, docs/FINANCAS.md):
   Finance (calendars and money, curves, options, Monte Carlo on the native
   worker, risk, portfolios, backtests with noise gates, arbitrage) and the
   Trading desk (an order book with its hash-chained journal, an exchange
   session audited by a naive engine, microstructure). It runs after
   bancada.js and uses its helpers (mount, head, editor, deskBar, seg, sect,
   stats, stat, table, facts, okChip, wbNum, lineChart, heatmap, legendOf). */
{
/* ================================================================ style */
const css = `
.mk-seal { display: inline-grid; gap: 2px; border: 2px solid var(--lock); color: var(--lock); border-radius: 12px; padding: 8px 14px; font: 600 14px var(--head); letter-spacing: .02em; max-width: 100%; }
.mk-seal small { font: 400 12px var(--body); color: var(--silt); letter-spacing: 0; }
.mk-seal.bad { border-color: var(--ember); color: var(--ember); }
.mk-seal.warn { border-color: var(--sand); color: var(--sand); }
.mk-gates { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
.mk-gates li { display: grid; grid-template-columns: 22px 1fr; gap: 8px; align-items: start; font-size: 13.5px; }
.mk-gates li i { width: 18px; height: 18px; border-radius: 50%; display: grid; place-items: center; font-style: normal; font-size: 12px; color: var(--mist); background: var(--lock); margin-top: 1px; }
.mk-gates li.no i { background: var(--ember); }
.mk-gates small { display: block; color: var(--silt); font-size: 12px; }
.mk-light { display: inline-flex; gap: 6px; padding: 6px 8px; background: var(--ink); border-radius: 999px; }
.mk-light span { width: 16px; height: 16px; border-radius: 50%; background: #3a4a4c; }
.mk-light span.on.g { background: #3fbf7f; box-shadow: 0 0 10px #3fbf7f; } .mk-light span.on.y { background: #e8c547; box-shadow: 0 0 10px #e8c547; } .mk-light span.on.r { background: #e5533d; box-shadow: 0 0 10px #e5533d; }
.mk-ladder { width: 100%; max-width: 520px; font: 12.5px/1 var(--mono); }
.mk-ladder .px { fill: var(--ink); } .mk-ladder .qty { fill: var(--silt); font-size: 11px; }
.mk-ladder .bid { fill: var(--lock); opacity: .78; } .mk-ladder .ask { fill: var(--ember); opacity: .78; }
.mk-ladder .spread { fill: var(--sand); opacity: .16; } .mk-ladder .axisl { stroke: var(--rule); }
.mk-chain { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; max-width: 100%; }
.mk-chain button { font: 11px var(--mono); border: 1px solid var(--rule); background: var(--mist); color: var(--silt); border-radius: 6px; padding: 3px 6px; cursor: pointer; }
.mk-chain button.fill { border-color: var(--lock); color: var(--lock); }
.mk-chain button.rej { border-color: var(--ember); color: var(--ember); }
.mk-chain button[aria-pressed="true"] { background: var(--lock); color: var(--mist); border-color: var(--lock); }
.mk-chain .link { color: var(--rule); font-size: 11px; }
.mk-entry { background: var(--slab); border-radius: 10px; padding: 10px 12px; box-shadow: inset 0 0 0 1px var(--rule); font-size: 13px; display: grid; gap: 4px; }
.mk-entry code { font-size: 12px; }
.mk-proof { display: grid; gap: 3px; font: 11.5px/1.35 var(--mono); }
.mk-proof span { color: var(--silt); } .mk-proof b { color: var(--lock); font-weight: 600; }
.mk-fix { font: 11.5px/1.45 var(--mono); color: var(--silt); overflow-wrap: anywhere; margin: 0; padding: 8px 10px; background: var(--slab); border-radius: 8px; max-height: 220px; overflow: auto; }
.mk-wbars { display: grid; gap: 5px; max-width: 760px; }
.mk-wbars .row { display: grid; grid-template-columns: 54px 1fr; gap: 10px; align-items: center; font-size: 12.5px; }
.mk-wbars .bars { display: grid; gap: 2px; }
.mk-wbars .bars i { display: block; height: 6px; border-radius: 3px; min-width: 1px; }
.mk-split { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; align-items: start; }
.mk-tape td.b { color: var(--lock); } .mk-tape td.s { color: var(--ember); }
#wb-fin .verdict-chip, #wb-hft .verdict-chip { white-space: nowrap; }
`;
document.head.append(el("style", { text: css }));

/* ================================================================ words */
Object.assign(I18N.en, {
  g_markets: "Markets", fin: "Finance", fin_s: "curves, options, risk, backtests — certified", hft: "Trading desk", hft_s: "order book, exchange, microstructure",
  fin_lede: "Write the problem the way a desk writes it — DI1 futures and NTN-Fs, an option quote, a smile, a P&L series, a strategy and its sweep, bid/ask quotes. Each answer brings what lets you judge it: every instrument repriced, parity and Greeks against finite differences, the no-arbitrage bounds checked before solving, the paths' bits against the exact oracle, Kupiec and Christoffersen on the VaR, and four noise gates on a backtest — look-ahead, deflated Sharpe, overfitting probability, Reality Check. Arbitrage is decided exactly: a portfolio, or state prices.",
  hft_lede: "A limit order book with price–time priority whose every session is a verifiable object: each event and its reports are chained by SHA-256 and closed by a Merkle root; a naive engine written separately replays the journal and must produce the same reports; the ITCH feed rebuilds the same book; a fill can be proved to belong to the session. The exchange session runs market makers, Hawkes-driven takers and an informed trader through the same pre-trade gate and engine — the backtest is the venue's code.",
  f_kinds: { arbitrage: "Arbitrage", backtest: "Backtest", calendar: "Calendar & money", curve: "Curve", mc: "Monte Carlo (native)", options: "Options", portfolio: "Portfolio", risk: "Risk (VaR)" },
  h_kinds: { book: "Order book", exchange: "Exchange session", micro: "Microstructure" },
  ex_fin: { cal_br: "B3/ANBIMA: business days, holidays, DI1", money: "Exact money: sums, rounding, allocation, 252 factor", nyse: "NYSE and TARGET calendars",
    c_di: "DI curve: DI1 + LTN + NTN-F (flat-forward, 252)", c_us: "USD: deposits + par swaps", c_bad: "An inconsistent quote (negative forward)",
    o_bsm: "European call: price, Greeks, parity", o_iv: "Implied volatility (well-conditioned)", o_ivbad: "A price below its no-arbitrage bound", o_am: "American put (Longstaff–Schwartz table 1)", o_heston: "Heston: price and its smile",
    o_smile: "A smile fitted by SVI, arbitrage-free", o_vogt: "A smile with butterfly arbitrage (Vogt)",
    m_euro: "Call, Asian and barrier on the worker", m_control: "Control: Itô's term forgotten", m_put: "Put with 16 384 paths",
    r_normal: "Normal VaR on fat tails (rejected)", r_hist: "Historical VaR on fat tails", r_ewma: "EWMA (RiskMetrics) VaR",
    p_factor: "Eight assets, two factors", p_cap: "Twelve assets, 20 % cap",
    b_noise: "Noise: 30 moving-average crossovers", b_planted: "A planted signal (AR(1) momentum)", b_peek: "A peek at tomorrow (lead)", b_leak: "Full-sample centring (a leak)",
    a_ok: "One period, two states: state prices", a_arb: "A mispriced call: the arbitrage", a_calls: "Call quotes: no static arbitrage", a_bfly: "Call quotes: a butterfly that pays", a_fx: "FX triangle: a profitable cycle", a_fxok: "FX triangle: consistent quotes" },
  ex_hft: { k_basic: "Price–time priority, IOC, market order", k_stp: "Self-trade prevention", k_fok: "FOK, post-only, modify, kill switch", k_rand: "Two hundred random orders",
    x_calm: "Two makers, Hawkes takers, an informed trader", x_wide: "Wider, more inventory-averse makers", x_noinf: "No informed trader", x_hot: "A hot Hawkes process (branching 0.8)",
    u_hawkes: "Hawkes: fit and the time-rescaling test", u_mm: "Avellaneda–Stoikov against the symmetric quote", u_ac: "Almgren–Chriss: closed form vs numeric" },
  mk_cert: "Certificate", mk_verdict: "Verdict", mk_nodes: "Nodes", mk_reprice: "Every instrument repriced", mk_forwards: "Forwards", mk_zero: "zero rate", mk_fwd: "instantaneous forward",
  mk_nss: "Nelson–Siegel–Svensson", mk_greeks: "Greeks: analytic and finite differences", mk_payoff: "Price against the underlying", mk_conv: "Lattice convergence (American)", mk_smile: "Smile",
  mk_density: "Risk-neutral density of ln(K/F)", mk_g: "Durrleman's g(k) — must stay ≥ 0", mk_paths: "Paths (binary64 preview of the same uniforms)", mk_speed: "Speed", mk_bits: "Bits",
  mk_var: "One-step VaR forecasts against the P&L", mk_full: "VaR and ES over the whole sample", mk_weights: "Weights", mk_corr: "Correlation (shrunk)",
  mk_gates: "Noise gates", mk_equity: "Equity: the best trial against buy-and-hold", mk_dd: "Drawdown", mk_pbo: "Out-of-sample logits of the in-sample winner (CSCV)", mk_trials: "Trials",
  mk_portfolio: "The arbitrage portfolio", mk_states: "State prices", mk_cycle: "The cycle", mk_ladder: "Depth", mk_chain: "Journal (SHA-256 chain) — choose an entry",
  mk_check: "Naive replay and invariants", mk_itch: "ITCH 5.0 feed", mk_fix: "FIX 4.4 execution reports", mk_proof: "Merkle proof of the first fill", mk_session: "Prices", mk_inv: "Maker inventory",
  mk_measures: "Measured on the tape", mk_tape: "Last trades", mk_makers: "Makers", mk_trajectory: "Holdings: closed form (line) and numeric optimum (dots)", mk_frontier: "Efficient frontier",
  mk_counts: "Arrivals per bin", mk_compare: "Inventory strategy against the symmetric quote (paper: γ = 0.1)", covers: "covers", excludes: "excludes", identical: "identical", differ: "differ",
  free_arb: "arbitrage-free", has_arb: "butterfly arbitrage", signal: "signal", noise: "noise or flawed", arbitrage: "arbitrage", no_arbitrage: "no arbitrage",
  save: "Save", bound: "bound", planted: "planted", fitted: "fitted", early_ex: "early exercise",
});
Object.assign(I18N.pt, {
  g_markets: "Mercados", fin: "Finanças", fin_s: "curvas, opções, risco, backtests — certificados", hft: "Mesa de operações", hft_s: "livro de ofertas, bolsa, microestrutura",
  fin_lede: "Escreva o problema como uma mesa escreve — DI1 e NTN-F, a cotação de uma opção, um sorriso, uma série de P&L, uma estratégia e a sua varredura, ofertas de compra e venda. Cada resposta traz o que permite julgá-la: todo instrumento reprecificado, paridade e gregas contra diferenças finitas, os limites de não arbitragem conferidos antes de resolver, os bits das trajetórias contra o oráculo exato, Kupiec e Christoffersen no VaR, e quatro portões de ruído num backtest — antecipação, Sharpe deflacionado, probabilidade de sobreajuste, Reality Check. A arbitragem é decidida exatamente: um portfólio, ou preços de estado.",
  hft_lede: "Um livro de ofertas com prioridade preço–tempo em que cada sessão é um objeto verificável: cada evento e os seus relatórios são encadeados por SHA-256 e fechados por uma raiz de Merkle; um motor ingênuo, escrito à parte, refaz o diário e tem de produzir os mesmos relatórios; o feed ITCH reconstrói o mesmo livro; um negócio pode ser provado como parte da sessão. A sessão de bolsa roda formadores de mercado, agressores guiados por Hawkes e um operador informado pelo mesmo portão de risco e pelo mesmo motor — o backtest é o código da bolsa.",
  f_kinds: { arbitrage: "Arbitragem", backtest: "Backtest", calendar: "Calendário e dinheiro", curve: "Curva", mc: "Monte Carlo (nativo)", options: "Opções", portfolio: "Carteira", risk: "Risco (VaR)" },
  h_kinds: { book: "Livro de ofertas", exchange: "Sessão de bolsa", micro: "Microestrutura" },
  ex_fin: { cal_br: "B3/ANBIMA: dias úteis, feriados, DI1", money: "Dinheiro exato: somas, arredondamento, rateio, fator 252", nyse: "Calendários NYSE e TARGET",
    c_di: "Curva DI: DI1 + LTN + NTN-F (flat-forward, 252)", c_us: "USD: depósitos + swaps par", c_bad: "Uma cotação inconsistente (forward negativo)",
    o_bsm: "Call europeia: preço, gregas, paridade", o_iv: "Volatilidade implícita (bem condicionada)", o_ivbad: "Um preço abaixo do limite de não arbitragem", o_am: "Put americana (tabela 1 de Longstaff–Schwartz)", o_heston: "Heston: preço e o seu sorriso",
    o_smile: "Um sorriso ajustado por SVI, sem arbitragem", o_vogt: "Um sorriso com arbitragem de borboleta (Vogt)",
    m_euro: "Call, asiática e barreira no worker", m_control: "Controle: o termo de Itô esquecido", m_put: "Put com 16 384 trajetórias",
    r_normal: "VaR normal em caudas grossas (rejeitado)", r_hist: "VaR histórico em caudas grossas", r_ewma: "VaR EWMA (RiskMetrics)",
    p_factor: "Oito ativos, dois fatores", p_cap: "Doze ativos, teto de 20 %",
    b_noise: "Ruído: 30 cruzamentos de médias", b_planted: "Um sinal plantado (momento AR(1))", b_peek: "Uma espiada no amanhã (lead)", b_leak: "Centragem pela amostra inteira (um vazamento)",
    a_ok: "Um período, dois estados: preços de estado", a_arb: "Uma call mal precificada: a arbitragem", a_calls: "Cotações de calls: sem arbitragem estática", a_bfly: "Cotações de calls: uma borboleta que paga", a_fx: "Triângulo de câmbio: um ciclo lucrativo", a_fxok: "Triângulo de câmbio: cotações coerentes" },
  ex_hft: { k_basic: "Prioridade preço–tempo, IOC, ordem a mercado", k_stp: "Prevenção de autonegociação", k_fok: "FOK, post-only, alteração, kill switch", k_rand: "Duzentas ordens aleatórias",
    x_calm: "Dois formadores, agressores Hawkes, um informado", x_wide: "Formadores mais largos e avessos a estoque", x_noinf: "Sem operador informado", x_hot: "Um Hawkes quente (ramificação 0,8)",
    u_hawkes: "Hawkes: ajuste e o teste de reescala do tempo", u_mm: "Avellaneda–Stoikov contra a cotação simétrica", u_ac: "Almgren–Chriss: forma fechada contra numérico" },
  mk_cert: "Certificado", mk_verdict: "Veredito", mk_nodes: "Nós", mk_reprice: "Todo instrumento reprecificado", mk_forwards: "Forwards", mk_zero: "taxa zero", mk_fwd: "forward instantâneo",
  mk_nss: "Nelson–Siegel–Svensson", mk_greeks: "Gregas: analíticas e por diferenças finitas", mk_payoff: "Preço contra o ativo", mk_conv: "Convergência da árvore (americana)", mk_smile: "Sorriso",
  mk_density: "Densidade neutra ao risco de ln(K/F)", mk_g: "g(k) de Durrleman — tem de ficar ≥ 0", mk_paths: "Trajetórias (prévia em binary64 dos mesmos uniformes)", mk_speed: "Velocidade", mk_bits: "Bits",
  mk_var: "Previsões de VaR de um passo contra o P&L", mk_full: "VaR e ES na amostra inteira", mk_weights: "Pesos", mk_corr: "Correlação (encolhida)",
  mk_gates: "Portões de ruído", mk_equity: "Patrimônio: a melhor tentativa contra o comprar-e-segurar", mk_dd: "Rebaixamento", mk_pbo: "Logits fora da amostra do vencedor dentro da amostra (CSCV)", mk_trials: "Tentativas",
  mk_portfolio: "O portfólio de arbitragem", mk_states: "Preços de estado", mk_cycle: "O ciclo", mk_ladder: "Profundidade", mk_chain: "Diário (cadeia SHA-256) — escolha uma entrada",
  mk_check: "Refazer ingênuo e invariantes", mk_itch: "Feed ITCH 5.0", mk_fix: "Relatórios de execução FIX 4.4", mk_proof: "Prova de Merkle do primeiro negócio", mk_session: "Preços", mk_inv: "Estoque dos formadores",
  mk_measures: "Medido na fita", mk_tape: "Últimos negócios", mk_makers: "Formadores", mk_trajectory: "Posição: forma fechada (linha) e ótimo numérico (pontos)", mk_frontier: "Fronteira eficiente",
  mk_counts: "Chegadas por intervalo", mk_compare: "Estratégia de estoque contra a cotação simétrica (artigo: γ = 0,1)", covers: "cobre", excludes: "exclui", identical: "idênticos", differ: "diferem",
  free_arb: "sem arbitragem", has_arb: "arbitragem de borboleta", signal: "sinal", noise: "ruído ou falha", arbitrage: "arbitragem", no_arbitrage: "sem arbitragem",
  save: "Salvar", bound: "limite", planted: "plantado", fitted: "ajustado", early_ex: "exercício antecipado",
});

/* ============================================================= examples */
const EX_FIN = {
  calendar: {
    cal_br: "du 2025-01-02 2026-01-02 anbima\nholidays anbima 2026\nadjust 2026-04-03 modified_following anbima\nadd 2026-12-30 2 anbima\ndi1 F27\nyf du252 2025-01-15 2025-07-15 anbima\nyf act/360 2025-01-15 2025-07-15",
    money: "sum 0.1 0.2\nround 2.675 2 half_even\nround 2.665 2 half_up\nround -2.665 2 half_down\nallocate 100.00 1 1 1\nallocate 1000.01 0.1 0.3 0.6\nfactor 13.65% 21\nfactor 13.65% 252",
    nyse: "holidays nyse 2026\ndu 2026-01-02 2026-12-31 nyse\nholidays target 2026\nyf act/act 2024-12-31 2027-07-15\nyf 30/360 2025-01-31 2025-03-31",
  },
  curve: {
    c_di: "# DI pre curve, ANBIMA conventions (prices illustrative)\ndate = 2025-01-02\ncalendar = anbima\nbasis = du252\ninterpolation = flat_forward\ndi1 G25 = 12.33%\ndi1 J25 = 13.05%\ndi1 N25 = 14.10%\ndi1 F26 = 15.02%\ndi1 F27 = 15.35%\nltn 2028-01-01 price = 652.30\nntnf 2031-01-01 price = 800.00\nfit nss",
    c_us: "# deposits and annual par swaps on one curve\ndate = 2025-03-03\ncalendar = nyse\nbasis = act/360\ncompounding = annual\ndeposit 2025-04-03 = 4.32% simple\ndeposit 2025-06-03 = 4.30% simple\ndeposit 2025-09-03 = 4.25% simple\nswap 1y = 4.05%\nswap 2y = 3.90%\nswap 3y = 3.85%\nswap 5y = 3.88%\nswap 7y = 3.97%\nswap 10y = 4.10%\nfit nss",
    c_bad: "# January at 15 %, July at 9 %: the forward between them is negative\ndate = 2025-01-02\ndi1 F26 = 15%\ndi1 N26 = 9%\ndi1 F27 = 14%",
  },
  options: {
    o_bsm: "price call S=100 K=105 T=0.5 r=5% q=2% vol=25%",
    o_iv: "iv put S=100 K=95 T=0.25 r=5% price=1.80",
    o_ivbad: "iv call S=100 K=50 T=1 r=5% price=40",
    o_am: "american put S=36 K=40 T=1 r=6% vol=20% steps=801",
    o_heston: "heston call S=100 K=100 T=1 r=2% v0=0.04 kappa=1.5 theta=0.04 xi=0.5 rho=-0.7",
    o_smile: "smile T=0.5 F=100\n70 32%\n80 28%\n90 25%\n100 23%\n110 22%\n120 22.5%\n130 23.5%",
    o_vogt: "# quotes drawn from the slice of Gatheral & Jacquier (2014) attributed to Vogt\nsmile T=1 F=100\n60.65 22.578%\n74.08 18.923%\n90.48 15.057%\n110.52 11.655%\n134.99 11.152%\n164.87 15.216%\n201.38 21.086%\n245.96 26.809%\n300.42 31.985%\n366.93 36.646%",
  },
  mc: {
    m_euro: "call S=100 K=100 T=1 r=5% vol=20% paths=8192 steps=64 barrier=85 threads=2",
    m_control: "call S=100 K=100 T=1 r=5% vol=20% paths=8192 steps=64 drift=no_ito",
    m_put: "put S=100 K=110 T=0.5 r=3% vol=30% paths=16384 steps=32 threads=2",
  },
  risk: {
    r_normal: "# Student t (ν = 4) returns: the normal VaR misses the tails\ndata = t n=1500 sigma=0.01 seed=4 nu=4\nalpha = 0.99\nwindow = 250\nmethod = normal",
    r_hist: "data = t n=1500 sigma=0.01 seed=4 nu=4\nalpha = 0.99\nwindow = 250\nmethod = historical",
    r_ewma: "data = t n=1500 sigma=0.01 seed=4 nu=4\nalpha = 0.99\nwindow = 250\nmethod = ewma",
  },
  portfolio: { p_factor: "assets = 8 n = 750 seed = 2", p_cap: "assets = 12 n = 500 seed = 5 cap = 0.2" },
  backtest: {
    b_noise: "# a random walk: no strategy can work; the best of 30 still looks good\ndata = gbm n=2520 sigma=0.01 seed=5\nsweep fast = 5..30 step 5\nsweep slow = 40..120 step 20\nsignal = sign(ema(close, fast) - ema(close, slow))\ncost = 2bp",
    b_planted: "# returns with autocorrelation 0.15: momentum is real here\ndata = ar1 n=5040 phi=0.15 sigma=0.01 seed=3\nsweep k = 1..3\nsignal = sign(sma(ret(close), k))\ncost = 1bp",
    b_peek: "data = gbm n=1500 sigma=0.01 seed=2\nsignal = sign(lead(close) - close)",
    b_leak: "# z-scored with the whole sample's mean: tomorrow's prices leak into today\ndata = gbm n=1500 sigma=0.01 seed=2\nsignal = sign(center(close))",
  },
  arbitrage: {
    a_ok: "states = up, down\nbond  bid=0.95 ask=0.96 payoff = 1, 1\nstock bid=100 ask=100.5 payoff = 120, 90\ncall  bid=10 ask=10.5 payoff = 20, 0",
    a_arb: "states = up, down\nbond  bid=0.95 ask=0.96 payoff = 1, 1\nstock bid=100 ask=100.5 payoff = 120, 90\ncall  bid=14 ask=14.5 payoff = 20, 0",
    a_calls: "calls T=1 r=5% S=100\n90  bid=14.1 ask=14.4\n100 bid=7.9 ask=8.2\n110 bid=3.6 ask=3.9",
    a_bfly: "calls T=1 r=5%\n90  bid=14.1 ask=14.4\n100 bid=9.4 ask=9.6\n110 bid=3.6 ask=3.9",
    a_fx: "fx\nUSD/BRL bid=5.40 ask=5.41\nEUR/USD bid=1.08 ask=1.081\nEUR/BRL bid=5.90 ask=5.91",
    a_fxok: "fx\nUSD/BRL bid=5.40 ask=5.41\nEUR/USD bid=1.08 ask=1.081\nEUR/BRL bid=5.833 ask=5.847",
  },
};
const rnd = (() => { let s = 7; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();
const randomOrders = Array.from({ length: 200 }, (_, i) => {
  const u = rnd(), side = rnd() < 0.5 ? "buy" : "sell", q = 1 + Math.floor(rnd() * 40), own = ["ana", "bia", "caio", "davi"][Math.floor(rnd() * 4)];
  if (u < 0.12 && i > 5) return `cancel ${1 + Math.floor(rnd() * i)}`;
  if (u < 0.17) return `${side} ${q} mkt owner=${own}`;
  // most orders rest on their own side of 100.00; one in six crosses
  const cross = rnd() < 0.17, step = Math.floor(rnd() * 8) * 0.05;
  const px = (side === "buy" ? (cross ? 100.05 + step / 2 : 99.95 - step) : (cross ? 99.95 - step / 2 : 100.05 + step)).toFixed(2);
  return `${side} ${q} @ ${px}${u > 0.9 ? " ioc" : ""} owner=${own}`;
}).join("\n");
const EX_HFT = {
  book: {
    k_basic: "# prices in currency, tick = 0.01; ids are the line order of new orders\nsell 50 @ 101.30 owner=bia\nsell 30 @ 101.25 owner=caio\nsell 40 @ 101.25 owner=davi\nbuy 100 @ 101.10 owner=ana\nbuy 60 @ 101.25 owner=eva\nbuy 20 @ 101.40 ioc owner=fabio\nsell 70 mkt owner=gil",
    k_stp: "stp = cancel_taker\nsell 10 @ 50.00 owner=ana\nsell 10 @ 50.01 owner=bia\nbuy 15 @ 50.01 owner=ana\nbuy 15 @ 50.01 owner=caio",
    k_fok: "sell 10 @ 20.00 owner=ana\nsell 10 @ 20.02 owner=bia\nbuy 25 @ 20.02 fok owner=caio\nbuy 15 @ 20.02 fok owner=caio\nbuy 5 @ 20.02 post owner=davi\nbuy 5 @ 19.98 post owner=davi\nmodify 6 qty=3\nbuy 8 @ 19.97 owner=davi\nkill davi",
    k_rand: "tick = 0.01\n" + randomOrders,
  },
  exchange: { x_calm: "steps=1500 seed=3", x_wide: "steps=1500 seed=7 half_spread=6 gamma=0.06", x_noinf: "steps=1500 seed=3 informed=off", x_hot: "steps=1500 seed=3 mu=0.4 alpha=1.6 beta=2.0" },
  micro: { u_hawkes: "hawkes mu=1 alpha=0.6 beta=1.5 T=2000 seed=3", u_mm: "mm gamma=0.1 runs=600", u_ac: "execution X=1e6 N=20 T=5 sigma=0.95 eta=2.5e-6 gamma=2.5e-7 epsilon=0.0625 lambda=1e-6" },
};

/* ============================================================== helpers */
const pct = (x, d = 2) => (x == null ? "—" : (100 * x).toFixed(d) + " %");
const short = (h, n = 10) => (h ? String(h).slice(0, n) : "—");
function seal(ok, title, sub, warn = false) { return el("div", { class: "mk-seal" + (ok ? "" : warn ? " warn" : " bad") }, el("span", { text: title }), sub ? el("small", { text: sub }) : ""); }
function gates(list) { return el("ul", { class: "mk-gates" }, ...list.map((g) => el("li", { class: g.pass ? "" : "no" }, el("i", { text: g.pass ? "✓" : "✗" }), el("span", {}, el("b", { text: g.gate }), el("small", { text: g.detail || "" }))))); }
function light(zone) { return el("span", { class: "mk-light", role: "img", "aria-label": zone }, ...["g", "y", "r"].map((c) => el("span", { class: c + ({ green: "g", yellow: "y", red: "r" }[zone] === c ? " on" : "") }))); }
function pts(xs, ys) { return xs.map((x, i) => [x, ys[i]]).filter(([x, y]) => x != null && y != null && isFinite(x) && isFinite(y)); }
function chart(series, opts = {}) { return lineChart({ w: 760, h: 250, ...opts, series }); }
// a legend whose swatches follow the series' own classes (0 lock, 1 ember, 2 sand, 3 silt)
function legendC(pairs) { const d = el("div", { class: "legend" }); pairs.forEach(([n, c]) => d.append(el("span", {}, el("i", { class: `l${c}` }), n))); return d; }
// grouped horizontal bars: one row per asset, one bar per method
function wbars(names, groups) {
  const max = Math.max(1e-12, ...groups.flatMap((g) => g.values.map((v) => Math.abs(v))));
  return el("div", { class: "mk-wbars" }, ...names.map((n, i) => el("div", { class: "row" }, el("span", { text: n }),
    el("div", { class: "bars" }, ...groups.map((g, k) => { const b = el("i", { title: `${g.name}: ${pct(g.values[i], 1)}` }); b.style.width = `${(100 * Math.abs(g.values[i])) / max}%`; b.style.background = palColors[k % palColors.length]; return b; })))),
    legendOf(groups.map((g) => g.name)));
}
// a price ladder: asks above, bids below; bars by quantity, the spread shaded
function ladder(depth, { rows = 8, fmt = (p) => (typeof p === "number" ? p.toFixed(2) : String(p)) } = {}) {
  const asks = (depth.asks || []).slice(0, rows).map((a) => (Array.isArray(a) ? { price: a[0], qty: a[1], orders: a[2] } : a)).reverse();
  const bids = (depth.bids || []).slice(0, rows).map((b) => (Array.isArray(b) ? { price: b[0], qty: b[1], orders: b[2] } : b));
  const all = [...asks, ...bids]; if (!all.length) return el("p", { class: "muted", text: "∅" });
  const W = 520, rh = 22, H = (all.length + 1) * rh + 8, mid = W / 2, qmax = Math.max(...all.map((r) => r.qty)), bw = mid - 120;
  const svg = svgEl("svg", { class: "mk-ladder", viewBox: `0 0 ${W} ${H}`, role: "img" });
  let y = 4;
  const row = (r, side) => {
    const len = (bw * r.qty) / qmax;
    svg.append(svgEl("rect", { class: side, x: side === "bid" ? mid - 44 - len : mid + 44, y: y + 3, width: len, height: rh - 6, rx: 3 }));
    const p = svgEl("text", { class: "px", x: mid, y: y + rh / 2 + 4, "text-anchor": "middle" }); p.textContent = fmt(r.price); svg.append(p);
    const q = svgEl("text", { class: "qty", x: side === "bid" ? mid - 50 - len : mid + 50 + len, y: y + rh / 2 + 4, "text-anchor": side === "bid" ? "end" : "start" }); q.textContent = `${r.qty}${r.orders ? " · " + r.orders : ""}`; svg.append(q);
    y += rh;
  };
  asks.forEach((r) => row(r, "ask"));
  if (asks.length && bids.length) {
    svg.append(svgEl("rect", { class: "spread", x: 0, y, width: W, height: rh }));
    const s = svgEl("text", { class: "qty", x: mid, y: y + rh / 2 + 4, "text-anchor": "middle" }); s.textContent = `spread ${fmt(asks[asks.length - 1].price - bids[0].price)}`; svg.append(s);
  }
  y += rh;
  bids.forEach((r) => row(r, "bid"));
  svg.append(svgEl("line", { class: "axisl", x1: mid, x2: mid, y1: 0, y2: H }));
  return svg;
}
function saveBtn(kind, text, result, state) { return btn(t("save"), "quiet", () => saveArchive("finance." + kind, { text }, result, state)); }

/* ============================================================== finance */
/* ============================================================ português
   The views are written once, with the English the server speaks. In
   Portuguese a single pass over the rendered result translates every label
   and every server sentence — after the view ran, so the logic that reads
   the English (a verdict's regex) is untouched. Code, hashes, FIX and ITCH
   bytes are never translated. A phrase not in the table stays in English
   (the test lists the strings it sees, so a gap is found, not hidden). */
const PT_WORDS = {
  price: "preço", analytic: "analítica", "finite differences": "diferenças finitas", repriced: "reprecificado", quote: "cotação", quotes: "cotações", model: "modelo", rel: "rel.",
  date: "data", calendar: "calendário", basis: "base", interpolation: "interpolação", compounding: "capitalização", "t (years)": "t (anos)",
  American: "Americana", European: "Europeia", steps: "passos", "BSM at √v₀": "BSM em √v₀", paths: "trajetórias", seed: "semente", "binary64 Δ max": "Δ máx. binary64",
  "Geometric Asian": "Asiática geométrica", "Arithmetic Asian (CV)": "Asiática aritmética (VC)", lowering: "compilação", step: "passo", barrier: "barreira",
  exceptions: "exceções", rate: "taxa", method: "método", day: "dia", "cond. coverage p": "p de cobertura cond.", "Basel (last 250 days)": "Basileia (últimos 250 dias)",
  assets: "ativos", observations: "observações", "min variance": "variância mínima", "risk parity": "paridade de risco", "min var": "var. mínima", "KKT · risk budgets": "KKT · orçamentos de risco",
  "Sharpe (best)": "Sharpe (melhor)", "max DD": "rebaix. máx.", "hit rate": "acerto", strategy: "estratégia", "buy & hold": "comprar e segurar", "look-ahead": "antecipação", params: "parâmetros",
  cost: "custo", asset: "ativo", quantity: "quantidade", payoff: "pagamento", state: "estado", gross: "bruto", from: "de", to: "para",
  events: "eventos", trades: "negócios", volume: "volume", refused: "recusadas", "quoted spread": "spread cotado", "Roll spread": "spread de Roll", fundamental: "fundamental",
  cash: "caixa", inventory: "estoque", "P&L (marked)": "P&L (marcado)", "RV signature": "assinatura da VR", "sampling step": "passo de amostragem",
  seq: "seq", side: "lado", qty: "qtd", taker: "agressor", maker: "formador", buy: "compra", sell: "venda",
  "mean P&L": "P&L médio", "mean q": "q médio", "mean spread": "spread médio", symmetric: "simétrica", "paper · inventory": "artigo · estoque", "paper · symmetric": "artigo · simétrica",
  period: "período", verified: "verificada", "feed = book": "feed = livro", messages: "mensagens", bytes: "bytes", levels: "níveis", executed: "executado", "checked exactly": "conferido exatamente", "= total": "= total",
  "naive engine = engine": "motor ingênuo = motor", "pre-trade limits": "limites pré-negociação", "ITCH feed = book": "feed ITCH = livro",
  "Prices (ticks from the start)": "Preços (ticks desde o início)", "Preços (ticks from the start)": "Preços (ticks desde o início)",
  "buy at ask": "compra no ask", "sell at bid": "venda no bid", "bond (pays 1)": "título (paga 1)", underlying: "ativo-objeto", oracle: "oráculo",
  "business days": "dias úteis", holidays: "feriados", "adjusted date": "data ajustada", "business days added": "dias úteis somados", "year fraction": "fração de ano",
  "DI1 maturity": "vencimento do DI1", "allocation (largest remainder)": "rateio (maior resto)", rounding: "arredondamento",
  "factor (1 + r)^(du/252), truncated at 8 places": "fator (1 + r)^(du/252), truncado na 8ª casa", "exact sum (binary64 for contrast)": "soma exata (binary64 para contraste)",
  monotonicity: "monotonicidade", "convexity (butterfly)": "convexidade (borboleta)", "spread above the discounted strike gap": "spread acima da diferença descontada dos strikes",
  "no look-ahead (prefix invariance)": "sem antecipação (invariância de prefixo)", "deflated Sharpe > 0.95": "Sharpe deflacionado > 0.95",
  "one trial: not applicable": "uma tentativa: não se aplica", "every gate passed": "todos os portões aprovados", "E[cost]": "E[custo]", "σ[cost]": "σ[custo]", "signal: every gate passed": "sinal: todos os portões aprovados",
  "the chain holds; a naive engine reproduces every report; every invariant holds": "a cadeia confere; um motor ingênuo reproduz todo relatório; todo invariante vale",
  "too many exceptions: the model underestimates risk": "exceções demais: o modelo subestima o risco",
  "too few exceptions: the model overestimates risk (capital wasted)": "exceções de menos: o modelo superestima o risco (capital desperdiçado)",
  "exceptions cluster in time: the model reacts too slowly": "as exceções se agrupam no tempo: o modelo reage devagar demais",
  "coverage and independence not rejected at 5 %": "cobertura e independência não rejeitadas a 5 %",
  "an instrument is not repriced: the curve does not hold": "um instrumento não é reprecificado: a curva não se sustenta",
  "self-exciting: the Hawkes model passes the time-rescaling test, Poisson fails it": "autoexcitação: o Hawkes passa no teste de reescala do tempo, o Poisson não",
  "both models pass: no evidence of self-excitation": "os dois modelos passam: sem evidência de autoexcitação",
  "the Hawkes model fails the time-rescaling test: another kernel is needed": "o Hawkes falha no teste de reescala do tempo: outro núcleo é necessário",
  "strictly positive state prices reproduce every quote inside its bid–ask (fundamental theorem)": "preços de estado estritamente positivos reproduzem toda cotação dentro do seu bid–ask (teorema fundamental)",
  "neither an arbitrage nor positive state prices were found (weak arbitrage at the boundary: a quote equal to its bound)": "nem arbitragem nem preços de estado positivos (arbitragem fraca na fronteira: uma cotação igual ao seu limite)",
  "a cycle of conversions returns more than it started with": "um ciclo de conversões devolve mais do que começou",
  "every cycle of up to five conversions returns at most what it started with": "todo ciclo de até cinco conversões devolve no máximo o que começou",
  "payoff ≥ 0 in every state, and cost < 0 (or cost ≤ 0 with a positive payoff somewhere)": "pagamento ≥ 0 em todo estado, e custo < 0 (ou custo ≤ 0 com pagamento positivo em algum)",
  "ψ > 0 and bid ≤ Σ Xψ ≤ ask for every asset": "ψ > 0 e bid ≤ Σ Xψ ≤ ask para todo ativo",
  "the product of the cycle's rates, in rationals, exceeds 1": "o produto das taxas do ciclo, em racionais, excede 1",
  "all simple cycles of length ≤ 5 enumerated; each product ≤ 1 in rationals": "todos os ciclos simples de comprimento ≤ 5 enumerados; cada produto ≤ 1 em racionais",
  "e^(−rT) rounded to 15 decimals": "e^(−rT) arredondado a 15 casas",
};
const PT_RULES = [
  [/^parity (\S+) · Greeks (\S+)$/, "paridade $1 · gregas $2"], [/^parity (\S+) · Feller (.)$/, "paridade $1 · Feller $2"],
  [/^price\(σ\): (.*) and the solution$/, "preço(σ): $1 e a solução"], [/^(.*) · (\d+) runs$/, "$1 · $2 trajetórias"],
  [/^variance ÷ (.*)$/, "variância ÷ $1"], [/^Down-and-out @ (.*)$/, "Barreira de saída @ $1"], [/^(.*) knocked out$/, "$1 desativadas"],
  [/^(.*) \(native\)$/, "$1 (nativo)"], [/^expected (.*)$/, "esperado $1"], [/^stationarity (\S+) · RC error (\S+)$/, "estacionaridade $1 · erro de CR $2"],
  [/^day (\d+) · cut (\d+)$/, "dia $1 · corte $2"], [/^(\d+) events replayed in (.*) ms$/, "$1 eventos refeitos em $2 ms"],
  [/^(\d+) accepted · (\d+) refused$/, "$1 aceitas · $2 recusadas"], [/^(\d+) messages · (\d+) executed$/, "$1 mensagens · $2 executados"],
  [/^half-life (\S+) · \|closed − numeric\| (\S+)$/, "meia-vida $1 · |fechada − numérica| $2"], [/^ … (\d+) more \(the Merkle root covers all\)$/, " … mais $1 (a raiz de Merkle cobre todas)"],
  [/^DSR = (\S+) with N = (\d+) trials$/, "DSR = $1 com N = $2 tentativas"], [/^PBO = (\S+) over (\d+) splits$/, "PBO = $1 em $2 partições"],
  [/^p = (\S+) \((\d+) stationary bootstraps\)$/, "p = $1 ($2 bootstraps estacionários)"], [/^identical on (\d+) truncated histories$/, "idêntico em $1 históricos truncados"],
  [/^day (\d+): (.*) with the full history, (.*) when the data stops at day (\d+) — the signal uses the future$/, "dia $1: $2 com o histórico inteiro, $3 quando os dados param no dia $4 — o sinal usa o futuro"],
  [/^(\d+) failure\(s\); the first at event (\d+): (.*)$/, "$1 falha(s); a primeira no evento $2: $3"],
  [/^every instrument repriced to (\S+); all forwards positive$/, "todo instrumento reprecificado a $1; todos os forwards positivos"],
  [/^every instrument repriced, but (\d+) forward\(s\) are negative — check the quotes$/, "todo instrumento reprecificado, mas $1 forward(s) negativo(s) — confira as cotações"],
  [/^a portfolio that pays at least zero in every state and is paid (\S+) to enter$/, "um portfólio que paga ao menos zero em todo estado e recebe $1 para entrar"],
  [/^a portfolio that costs nothing, never loses, and pays in some state$/, "um portfólio que não custa nada, nunca perde e paga em algum estado"],
  [/^price (\S+) is at or below the lower no-arbitrage bound (\S+) \(intrinsic value discounted\): no volatility reproduces it$/, "o preço $1 está no limite inferior de não arbitragem $2 (valor intrínseco descontado) ou abaixo dele: nenhuma volatilidade o reproduz"],
  [/^price (\S+) is at or above the upper bound (\S+): no volatility reproduces it$/, "o preço $1 está no limite superior $2 ou acima dele: nenhuma volatilidade o reproduz"],
  [/^(.*) failed$/, (_, s) => "reprovado: " + s.split("; ").map(ptOf).join("; ")],
];
function ptOf(s) {
  const k = s.trim(); if (!k) return s;
  if (Object.hasOwn(PT_WORDS, k)) return s.replace(k, PT_WORDS[k]);
  for (const [re, to] of PT_RULES) if (re.test(s)) return s.replace(re, to);
  return s;
}
const PT_SKIP = "pre, code, textarea, .hash, .mk-fix, .mk-proof, .mk-chain, .mk-entry";
function localize(...roots) {
  if (lang !== "pt") return;
  for (const root of roots) {
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT), nodes = [];
    while (w.nextNode()) nodes.push(w.currentNode);
    for (const n of nodes) if (!n.parentElement?.closest(PT_SKIP)) { const v = ptOf(n.nodeValue); if (v !== n.nodeValue) n.nodeValue = v; }
  }
}
// the first line of each example is a comment; in Portuguese, so is it
const PT_COMMENTS = {
  "# DI pre curve, ANBIMA conventions (prices illustrative)": "# curva DI pré, convenções ANBIMA (preços ilustrativos)",
  "# deposits and annual par swaps on one curve": "# depósitos e swaps par anuais numa curva só",
  "# January at 15 %, July at 9 %: the forward between them is negative": "# janeiro a 15 %, julho a 9 %: o forward entre eles é negativo",
  "# quotes drawn from the slice of Gatheral & Jacquier (2014) attributed to Vogt": "# cotações tiradas da fatia de Gatheral & Jacquier (2014) atribuída a Vogt",
  "# Student t (ν = 4) returns: the normal VaR misses the tails": "# retornos t de Student (ν = 4): o VaR normal erra as caudas",
  "# a random walk: no strategy can work; the best of 30 still looks good": "# um passeio aleatório: nenhuma estratégia funciona; a melhor de 30 ainda parece boa",
  "# returns with autocorrelation 0.15: momentum is real here": "# retornos com autocorrelação 0,15: aqui o momento é real",
  "# z-scored with the whole sample's mean: tomorrow's prices leak into today": "# padronizado pela média da amostra inteira: os preços de amanhã vazam para hoje",
  "# prices in currency, tick = 0.01; ids are the line order of new orders": "# preços em moeda, tick = 0,01; os ids são a ordem das linhas das ordens novas",
};
const exText = (text) => (lang === "pt" ? text.split("\n").map((l) => PT_COMMENTS[l] || l).join("\n") : text);

const FIN_VIEW = {
  calendar(r, side, out) {
    side.append(el("p", { class: "muted", text: "du · holidays · adjust · add · di1 · yf · allocate · round · factor · sum" }));
    const val = (l) => {
      if (Array.isArray(l.value) && l.value.length && l.value[0].date) return el("div", { class: "chips" }, ...l.value.map((h) => el("span", { class: "pill", title: h.name }, el("b", { text: h.date }), " " + h.name)));
      if (Array.isArray(l.value)) return el("span", {}, l.value.join(" + "), " ", l.certificate ? okChip(l.certificate.sum_equals_total, "= total") : "");
      if (l.binary64) return el("span", {}, el("b", { text: l.value }), el("span", { class: "muted", text: `  (binary64: ${l.binary64})` }));
      return el("b", { text: typeof l.value === "number" ? wbNum(l.value, 10) : String(l.value) });
    };
    out.append(sect(t("mk_cert"), table(["", "", ""], r.lines.map((l) => [el("code", { text: l.input }), l.kind, val(l)]))));
  },
  curve(r, side, out) {
    const c = r.certificate;
    side.append(seal(c.repriced && !c.negative_forwards, c.repriced ? t("mk_reprice") : "✗", c.verdict, c.repriced), facts([["calendar", r.calendar], ["basis", r.basis], ["interpolation", r.interpolation], ["compounding", r.compounding]]));
    const z = r.dense.map((d) => [d.t, 100 * d.zero]), f = r.forward_curve.map((d) => [d.t, 100 * d.forward]);
    const nodes = r.nodes.map((n) => [n.t, 100 * n.zero]);
    const series = [{ points: z, cls: 0 }, { points: f, cls: 1 }, { points: nodes, cls: 2, dots: true }];
    const neg = f.some((p) => p[1] < 0.5);
    out.append(sect(t("mk_zero") + " / " + t("mk_fwd"), chart(series, { xlab: "t (years)", ylab: "%", levels: neg ? [{ y: 0, cls: "mark" }] : [] }),
      legendC([[t("mk_zero"), 0], [t("mk_fwd"), 1], [t("mk_nodes"), 2]])));
    out.append(el("div", { class: "mk-split" },
      sect(t("mk_nodes"), table(["date", "t", "DF", t("mk_zero")], r.nodes.map((n) => [n.date, wbNum(n.t, 4), n.df.toFixed(10), pct(n.zero, 4)]))),
      sect(t("mk_cert"), table(["", "quote", "model", "rel"], c.reprice.map((p) => [el("code", { text: p.instrument }), wbNum(p.quote_value, 8), wbNum(p.model_value, 8), okChip(p.rel_error < 1e-10, wbNum(p.rel_error, 2), wbNum(p.rel_error, 2))])))));
    if (r.nss && r.nss.beta) out.append(sect(t("mk_nss"), facts([["β", r.nss.beta.map((b) => wbNum(b, 4)).join(", ")], ["τ", r.nss.tau.map((x) => wbNum(x, 4)).join(", ")], ["RMSE", pct(r.nss.rmse, 3)]])));
  },
  options(r, side, out) {
    if (r.task === "price") {
      side.append(seal(r.certificate.ok, t("mk_cert"), `parity ${wbNum(r.certificate.parity_residual, 2)} · Greeks ${wbNum(r.certificate.greeks_max_rel_diff, 2)}`));
      out.append(stats(stat("price", wbNum(r.price, 6)), stat("Δ", wbNum(r.greeks.delta, 5)), stat("Γ", wbNum(r.greeks.gamma, 5)), stat("vega", wbNum(r.greeks.vega, 5)), stat("θ", wbNum(r.greeks.theta, 5)), stat("ρ", wbNum(r.greeks.rho, 5)), stat("LR 1001", wbNum(r.lattice_lr_1001, 6))));
      out.append(sect(t("mk_greeks"), table(["", "analytic", "finite differences"], ["delta", "gamma", "vega", "theta", "rho"].map((k) => [k, wbNum(r.greeks[k], 8), wbNum(r.greeks_fd[k], 8)]))));
      out.append(sect(t("mk_payoff"), chart([{ points: r.payoff_curve.map((p) => [p.s, p.price]), cls: 0 }, { points: r.payoff_curve.map((p) => [p.s, p.intrinsic]), cls: 3 }], { xlab: "S" })));
    } else if (r.task === "implied volatility") {
      const c = r.certificate;
      side.append(seal(c.well_conditioned, "σ = " + pct(r.sigma, 4), `|error| ${wbNum(c.abs_error, 2)} · ${t("bound")}s [${wbNum(c.bounds[0], 5)}, ${wbNum(c.bounds[1], 5)}]`, true));
      out.append(stats(stat("σ", pct(r.sigma, 4)), stat("vega", wbNum(c.vega, 5)), stat("σ / tick", pct(c.vol_per_tick, 4)), stat("repriced", wbNum(c.repriced, 8))));
      if (r.curve) out.append(sect("price(σ): " + t("bound") + "s and the solution", chart([{ points: r.curve.map((p) => [100 * p.sigma, p.price]), cls: 0 }], { xlab: "σ (%)",
        levels: [{ y: r.quote, cls: "mark", label: "quote" }, { y: c.bounds[0], cls: "mark", label: t("bound") }, { y: c.bounds[1], cls: "mark", label: t("bound") }], marks: [{ x: 100 * r.sigma, label: "σ*" }] })));
    } else if (r.task === "american") {
      side.append(seal(r.certificate.premium_nonnegative, wbNum(r.price, 6), `${t("early_ex")} ${wbNum(r.early_exercise_premium, 5)} · |LR − BSM| ${wbNum(r.certificate.european_lattice_vs_bsm, 2)}`));
      out.append(stats(stat("American", wbNum(r.price, 6)), stat("European", wbNum(r.european, 6)), stat("BSM", wbNum(r.bsm, 6)), stat("steps", String(r.steps))));
      out.append(sect(t("mk_conv"), chart([{ points: r.convergence.map((c) => [c.steps, c.american]), cls: 0, dots: true }, { points: r.convergence.map((c) => [c.steps, c.crr]), cls: 1, dots: true }], { xlab: "steps", w: 560 }), legendOf(["Leisen–Reimer", "Cox–Ross–Rubinstein"])));
    } else if (r.task === "heston") {
      side.append(seal(Math.abs(r.certificate.parity_residual) < 1e-8, wbNum(r.price, 6), `parity ${wbNum(r.certificate.parity_residual, 2)} · Feller ${r.feller ? "✓" : "✗"}`));
      out.append(stats(stat("Heston", wbNum(r.price, 6)), stat("BSM at √v₀", wbNum(r.certificate.bsm_at_sqrt_v0, 6))));
      out.append(sect(t("mk_smile"), chart([{ points: r.smile.map((p) => [p.k, 100 * p.iv]), cls: 0 }], { xlab: "K", ylab: "σ (%)", w: 560 })));
    } else if (r.task === "smile") {
      const c = r.certificate;
      side.append(seal(c.free, c.free ? t("free_arb") : t("has_arb"), c.free ? `min g = ${wbNum(c.g_min, 3)}` : `g < 0 on k ∈ [${wbNum(c.negative_interval[0], 3)}, ${wbNum(c.negative_interval[1], 3)}]`),
        facts(Object.entries(r.params).map(([k, v]) => [k, wbNum(v, 5)])));
      out.append(sect(t("mk_smile"), chart([{ points: r.curve.map((p) => [p.strike, 100 * p.iv]), cls: 0 }, { points: r.quotes.map((q) => [q.strike, 100 * q.iv]), cls: 1, dots: true }], { xlab: "K", ylab: "σ (%)" }), legendOf(["SVI", "quotes"])));
      out.append(el("div", { class: "mk-split" },
        sect(t("mk_density"), chart([{ points: r.curve.map((p) => [p.k, p.density]), cls: 0 }, { points: r.curve.filter((p) => p.density < 0).map((p) => [p.k, p.density]), cls: 1, dots: true }], { w: 420, h: 220, levels: [{ y: 0, cls: "mark" }], xlab: "k" })),
        sect(t("mk_g"), chart([{ points: r.curve.map((p) => [p.k, p.g]), cls: c.free ? 0 : 1 }], { w: 420, h: 220, levels: [{ y: 0, cls: "mark" }], xlab: "k" }))));
    }
  },
  mc(r, side, out) {
    const c = r.certificate;
    side.append(seal(c.oracle_parity !== false && (!c.threads || c.threads.identical_bits) && c.european_covered, t("mk_bits") + ": " + (c.oracle_parity ? t("identical") : c.oracle_parity === false ? t("differ") : "oracle"),
      `${r.substrate} · ${r.rng} RNG · ${c.threads ? c.threads.threads + " threads " + (c.threads.identical_bits ? "=" : "≠") : ""}`),
      facts([["paths", r.paths], ["steps", r.steps], ["seed", r.seed], ["binary64 Δ max", wbNum(c.binary64_agreement.max_abs, 2)]]));
    const e = r.european, g = r.geometric_asian, a = r.arithmetic_asian_cv;
    out.append(stats(stat("European", `${wbNum(e.price, 5)} ± ${wbNum(e.stderr, 2)}`, `BSM ${wbNum(e.closed_form, 6)} · z ${wbNum(e.z, 2)} · ${e.covers ? t("covers") : t("excludes")}`),
      stat("Geometric Asian", `${wbNum(g.price, 5)} ± ${wbNum(g.stderr, 2)}`, `Kemna–Vorst ${wbNum(g.closed_form, 6)}`),
      stat("Arithmetic Asian (CV)", `${wbNum(a.price, 5)} ± ${wbNum(a.stderr, 2)}`, `variance ÷ ${wbNum(a.variance_reduction, 3)}`),
      ...(r.barrier ? [stat(`Down-and-out @ ${r.barrier.level}`, `${wbNum(r.barrier.price, 5)} ± ${wbNum(r.barrier.stderr, 2)}`, `${pct(r.barrier.knocked_out, 1)} knocked out`)] : [])));
    out.append(stats(stat(t("mk_speed") + " (native)", `${wbNum(r.native_ms, 4)} ms`), stat("BEAM binary64", `${wbNum(r.beam_f64_ms_estimate, 4)} ms`), stat("×", wbNum(r.speedup, 3)), stat("lowering", `${wbNum(r.lowering_ms, 4)} ms`)));
    const n = r.paths_preview[0].length, xs = Array.from({ length: n }, (_, i) => i);
    const series = r.paths_preview.map((p, i) => ({ points: xs.map((x) => [x, p[x]]), cls: 0, color: palColors[i % palColors.length] }));
    out.append(sect(t("mk_paths"), chart(series, { xlab: "step", levels: r.barrier ? [{ y: r.barrier.level, cls: "mark", label: "barrier" }] : [] })));
  },
  risk(r, side, out) {
    const b = r.backtest, l = r.basel_last_250;
    side.append(el("div", {}, light(l ? l.zone : b.zone), el("span", { class: "muted", text: "  Basel (last 250 days)" })), seal(!/too|cluster/.test(b.verdict), b.verdict, `Kupiec p = ${wbNum(b.kupiec.p_value, 3)} · Christoffersen p = ${wbNum(b.independence.p_value, 3)}`));
    out.append(stats(stat("exceptions", `${b.exceptions}`, `expected ${wbNum(b.expected, 3)}`), stat("rate", pct(b.rate, 2)), stat("Kupiec LR", wbNum(b.kupiec.lr, 3)), stat("cond. coverage p", wbNum(b.conditional_coverage.p_value, 3)), stat("method", String(r.method))));
    const f = r.forecasts;
    out.append(sect(t("mk_var"), chart([{ points: f.map((p) => [p.i, p.pnl]), cls: 3, bars: true, bw: 0.0012 }, { points: f.map((p) => [p.i, -p.var]), cls: 0 }, { points: (r.exceptions || []).map((p) => [p.i, p.pnl]), cls: 1, dots: true }], { xlab: "day", h: 260 }), legendC([["P&L", 3], ["−VaR", 0], ["exceptions", 1]])));
    out.append(sect(t("mk_full"), table(["method", "VaR", "ES"], r.full_sample.map((m) => [m.method, pct(m.var, 3), m.es == null ? "—" : pct(m.es, 3)]))));
  },
  portfolio(r, side, out) {
    side.append(seal(r.min_variance.certificate.stationarity < 1e-10 && r.risk_parity.certificate.max_budget_error < 1e-10, "KKT · risk budgets", `stationarity ${wbNum(r.min_variance.certificate.stationarity, 2)} · RC error ${wbNum(r.risk_parity.certificate.max_budget_error, 2)}`),
      facts([["assets", r.assets], ["observations", r.observations], ["Ledoit–Wolf δ", wbNum(r.shrinkage, 3)], ["κ(S) → κ(Σ*)", `${wbNum(r.condition_sample, 3)} → ${wbNum(r.condition_shrunk, 3)}`]]));
    out.append(stats(stat("min variance", pct(r.min_variance.vol_annual, 2)), stat("risk parity", pct(r.risk_parity.vol_annual, 2)), stat("HRP", pct(r.hrp.vol_annual, 2)), stat("1/N", pct(r.equal_weight.vol_annual, 2))));
    const names = r.min_variance.weights.map((_, i) => `A${i + 1}`);
    out.append(el("div", { class: "mk-split" }, sect(t("mk_weights"), wbars(names, [{ name: "min var", values: r.min_variance.weights }, { name: "risk parity", values: r.risk_parity.weights }, { name: "HRP", values: r.hrp.weights }])),
      sect(t("mk_corr"), heatmap(r.correlation.map((row) => row.map((x) => (x + 1) / 2)), { w: 300, h: 300 }))));
  },
  backtest(r, side, out) {
    const ok = r.gates.every((g) => g.pass);
    side.append(seal(ok, ok ? t("signal") : t("noise"), r.verdict.replace(/^[^:]*:\s*/, "")), gates(r.gates));
    const s = r.best.stats;
    out.append(stats(stat("Sharpe (best)", wbNum(s.sharpe_annual, 3), Object.entries(r.best.params).map(([k, v]) => `${k}=${v}`).join(" ")), stat("DSR", wbNum(r.deflated_sharpe.dsr, 3), `N = ${r.deflated_sharpe.trials}`),
      stat("PSR", wbNum(r.deflated_sharpe.psr, 3)), stat("PBO", r.pbo ? wbNum(r.pbo.pbo, 3) : "—"), stat("RC p", wbNum(r.reality_check.p_value, 3)), stat("max DD", pct(s.max_drawdown, 1)), stat("hit rate", pct(s.hit_rate, 1))));
    const xs = r.equity.map((_, i) => i);
    out.append(sect(t("mk_equity"), chart([{ points: pts(xs, r.equity), cls: 0 }, { points: pts(xs, r.buy_and_hold_equity), cls: 3 }], { xlab: "", logy: false }), legendC([["strategy", 0], ["buy & hold", 3]])));
    out.append(el("div", { class: "mk-split" }, sect(t("mk_dd"), chart([{ points: pts(xs, r.drawdown.map((d) => 100 * d)), cls: 1 }], { w: 420, h: 200, ylab: "%" })),
      r.pbo ? sect(t("mk_pbo"), chart([{ points: r.pbo.logit_histogram.map((b) => [(b.from + b.to) / 2, b.count]), cls: 2, bars: true }], { w: 420, h: 200, marks: [{ x: 0, label: "λ = 0" }] })) : el("span")));
    if (!r.lookahead.clean) out.append(sect("look-ahead", seal(false, `day ${r.lookahead.day} · cut ${r.lookahead.cut}`, r.lookahead.detail)));
    const top = [...r.trial_table].sort((a, b) => b.sharpe_annual - a.sharpe_annual).slice(0, 10);
    out.append(sect(t("mk_trials") + ` (${r.trials})`, table(["params", "Sharpe"], top.map((x) => [Object.entries(x.params).map(([k, v]) => `${k}=${v}`).join(" ") || "—", wbNum(x.sharpe_annual, 3)]))));
  },
  arbitrage(r, side, out) {
    const c = r.certificate || {};
    side.append(seal(!r.arbitrage, r.arbitrage ? t("arbitrage") : t("no_arbitrage"), r.why, false), okChip(c.checked_exactly, "checked exactly"), el("p", { class: "muted", text: c.rule || "" }));
    if (r.portfolio) {
      out.append(stats(stat("cost", r.cost.exact, wbNum(r.cost.value, 6))));
      out.append(el("div", { class: "mk-split" }, sect(t("mk_portfolio"), table(["asset", "quantity", ""], r.portfolio.map((p) => [p.asset, p.quantity.exact, p.side]))),
        sect("payoff", table(["state", "payoff"], r.payoffs.map((p) => [p.state, p.payoff.exact])))));
    }
    if (r.state_prices) {
      out.append(sect(t("mk_states"), chart([{ points: r.state_prices.map((s, i) => [i, s.price.value]), cls: 0, bars: true }], { w: 560, h: 200, xticks: r.state_prices.map((s, i) => [i, s.state]) })),
        table(["asset", "Σ Xψ", "bid", "ask"], (r.model_values || []).map((m) => [m.asset, wbNum(m.value, 8), wbNum(m.bid, 6), wbNum(m.ask, 6)])));
    }
    if (r.cycle || r.best_cycle) {
      const cy = r.cycle || r.best_cycle;
      out.append(sect(t("mk_cycle"), stats(stat("gross", r.gross.exact, wbNum(r.gross.value, 8))), table(["from", "to", "rate", ""], cy.map((x) => [x.from, x.to, wbNum(x.rate, 8), x.how]))));
    }
  },
};

mount("fin", (root, S, redraw) => {
  head(root, "fin", "fin_lede");
  S.kind ??= "curve"; S.res ??= {}; S.errs ??= {};
  S.texts ??= Object.fromEntries(Object.keys(EX_FIN).map((k) => [k, exText(Object.values(EX_FIN[k])[0])]));
  const col = new Intl.Collator(lang === "pt" ? "pt-BR" : "en");
  const kinds = Object.keys(EX_FIN).sort((a, b) => col.compare(t("f_kinds")[a], t("f_kinds")[b]));
  const kind = S.kind;
  const ed = editor("wb-fin-ed", 13); ed.value = S.texts[kind]; ed.oninput = () => (S.texts[kind] = ed.value);
  const out = el("div", { class: "wb-out", "aria-live": "polite" }), side = el("aside", { class: "wb-side" });
  const go = async () => {
    busy(out); bar.state.textContent = t("running");
    try { S.res[kind] = await api("/v1/vapor/finance", { kind, text: ed.value }); S.errs[kind] = null; } catch (e) { S.res[kind] = null; S.errs[kind] = e.message; }
    if (S.kind === kind && root.contains(out)) show();
  };
  ed.addEventListener("run", go);
  const save = btn(t("save"), "quiet", () => { if (S.res[kind] && kind !== "mc") saveArchive("finance." + kind, { text: ed.value }, S.res[kind], bar.state); });
  const bar = deskBar(EX_FIN[kind], () => t("ex_fin"), (k, text) => { ed.value = S.texts[kind] = exText(text); go(); }, [btn(t("run"), "primary", go), save]);
  root.append(seg(kinds, t("f_kinds"), kind, (k) => { S.kind = k; redraw(); }), bar.bar, el("div", { class: "wb-main" }, ed, side), out);
  function show() {
    side.replaceChildren(); out.replaceChildren();
    const err = S.errs[kind], r = S.res[kind];
    if (err) { bar.state.textContent = ""; out.append(notice(err)); localize(out); return; }
    if (!r) return;
    bar.state.textContent = r.ms != null ? t("ms", r.ms) : "";
    try { FIN_VIEW[kind](r, side, out); } catch (e) { out.append(notice(String(e))); }
    localize(side, out);
  }
  show();
  if (S.res[kind] === undefined && S.errs[kind] === undefined) go();
});

/* =========================================================== trading desk */
const HFT_VIEW = {
  book(r, side, out, S, redraw) {
    side.append(seal(r.check.ok, t("mk_check"), r.check.verdict), facts([["events", r.events], ["trades", r.trades], ["volume", r.volume], ["STP", r.stp], ["p50", `${wbNum(r.latency_us.p50, 3)} µs`]]),
      el("div", { class: "hash", text: "head " + r.head }), el("div", { class: "hash", text: "merkle " + r.merkle_root }));
    out.append(sect(t("mk_ladder"), ladder(r.depth, { rows: 10 })));
    S.pick = Math.min(S.pick ?? 0, r.journal.length - 1);
    const detail = el("div", { class: "mk-entry" });
    const drawDetail = () => { const e = r.journal[S.pick]; if (!e) return; detail.replaceChildren(el("b", { text: `#${e.seq} · ${e.event}` }), ...e.reports.map((x) => el("code", { text: x })), el("span", { class: "hash", text: "h = " + e.hash + "…" })); };
    const chainEl = el("div", { class: "mk-chain" });
    r.journal.slice(0, 240).forEach((e, i) => {
      const cls = e.reports.some((x) => x.startsWith("fill")) ? "fill" : e.reports.some((x) => /reject/.test(x)) ? "rej" : "";
      const b = el("button", { type: "button", class: cls, "aria-pressed": String(i === S.pick), title: e.event, text: short(e.hash, 6) });
      b.onclick = () => { S.pick = i; chainEl.querySelectorAll("button").forEach((x, j) => x.setAttribute("aria-pressed", String(j === i))); drawDetail(); };
      chainEl.append(b); if (i < Math.min(r.journal.length, 240) - 1) chainEl.append(el("span", { class: "link", text: "→" }));
    });
    if (r.journal.length > 240) chainEl.append(el("span", { class: "muted", text: ` … ${r.journal.length - 240} more (the Merkle root covers all)` }));
    drawDetail();
    out.append(sect(t("mk_chain"), chainEl, detail));
    const blocks = [];
    if (r.proof) blocks.push(sect(t("mk_proof"), okChip(r.proof.verified, "verified"), el("div", { class: "mk-proof" }, el("span", {}, "leaf ", el("b", { text: short(r.proof.leaf, 24) + "…" })), ...r.proof.path.map((p) => el("span", { text: `${p.side === "left" ? "← " : "→ "}${short(p.hash, 24)}…` })), el("span", {}, "root ", el("b", { text: short(r.proof.root, 24) + "…" })))));
    blocks.push(sect(t("mk_itch"), okChip(r.itch.book_equal && r.itch.volume_equal, "feed = book"), facts([["messages", r.itch.messages], ["bytes", r.itch.bytes], ["levels", r.itch.levels], ["executed", r.itch.executed]]), el("pre", { class: "mk-fix", text: r.itch.sample.join("\n") })));
    out.append(el("div", { class: "mk-split" }, ...blocks));
    out.append(sect(t("mk_fix"), el("pre", { class: "mk-fix", text: r.fix.join("\n") })));
  },
  exchange(r, side, out) {
    const c = r.certificate;
    const allOk = c.book_check.ok && c.pre_trade.ok && c.itch.book_equal && c.itch.volume_equal;
    side.append(seal(allOk, t("mk_cert"), `${c.book_check.checked} events replayed in ${wbNum(c.book_check_ms, 3)} ms`),
      gates([{ gate: "naive engine = engine", pass: c.book_check.ok, detail: c.book_check.verdict }, { gate: "pre-trade limits", pass: c.pre_trade.ok, detail: `${c.pre_trade.accepted} accepted · ${c.pre_trade.refused} refused` },
        { gate: "ITCH feed = book", pass: c.itch.book_equal && c.itch.volume_equal, detail: `${c.itch_messages} messages · ${c.itch.executed} executed` }]),
      el("div", { class: "hash", text: "head " + r.head }));
    out.append(stats(stat("events", String(r.events)), stat("trades", String(r.trades)), stat("volume", String(r.volume)), stat("refused", String(r.rejected)),
      stat("quoted spread", wbNum(r.measures.quoted_spread, 3)), stat("Roll spread", wbNum(r.measures.roll_spread, 3)), stat("Kyle λ", wbNum(r.measures.kyle.lambda, 3), `t = ${wbNum(r.measures.kyle.t, 2)}`)));
    const s = r.series;
    const p0 = s[0].fund;
    out.append(sect(t("mk_session") + " (ticks from the start)", chart([{ points: s.map((p) => [p.t, p.fund - p0]), cls: 3 }, { points: s.filter((p) => p.bid).map((p) => [p.t, p.bid - p0]), cls: 0 }, { points: s.filter((p) => p.ask).map((p) => [p.t, p.ask - p0]), cls: 1 }], { xlab: "t", h: 280 }), legendC([["fundamental", 3], ["bid", 0], ["ask", 1]])));
    const makers = Object.keys(s[0].inv || {});
    out.append(el("div", { class: "mk-split" },
      sect(t("mk_inv"), chart(makers.map((m, i) => ({ points: s.map((p) => [p.t, p.inv[m]]), cls: i % 4 })), { w: 420, h: 220 }), legendOf(makers)),
      sect(t("mk_ladder"), ladder(r.depth, { rows: 6, fmt: (p) => String(p) }))));
    const h = r.measures.hawkes;
    if (h) out.append(sect("Hawkes", seal(h.time_rescaling.p_value > 0.05, h.verdict, `KS p: Hawkes ${wbNum(h.time_rescaling.p_value, 3)} · Poisson ${wbNum(h.poisson_time_rescaling.p_value, 2)}`, true),
      table(["", "μ", "α", "β", "α/β"], [[t("fitted"), wbNum(h.mu, 4), wbNum(h.alpha, 4), wbNum(h.beta, 4), wbNum(h.branching_ratio, 3)]])));
    out.append(el("div", { class: "mk-split" },
      sect(t("mk_makers"), table(["", "cash", "inventory", "P&L (marked)"], r.makers.map((m) => [m.maker, wbNum(m.cash, 6), m.inventory, wbNum(m.pnl_marked, 6)]))),
      sect("RV signature", chart([{ points: r.measures.signature.map((x) => [x.step, x.rv_per_step]), cls: 2, dots: true }], { w: 420, h: 200, xlab: "sampling step" }))));
    const tape = el("table", { class: "cands mk-tape" }, el("thead", {}, el("tr", {}, ...["seq", "side", "qty", "price", "taker", "maker"].map((x) => el("th", { text: x })))),
      el("tbody", {}, ...r.tape.slice().reverse().slice(0, 20).map((f) => el("tr", {}, el("td", { text: String(f.seq) }), el("td", { class: f.taker_side === "buy" ? "b" : "s", text: f.taker_side }), el("td", { text: String(f.qty) }), el("td", { text: String(f.price) }), el("td", { text: String(f.taker_owner) }), el("td", { text: String(f.maker_owner) })))));
    out.append(sect(t("mk_tape"), el("div", { class: "tw" }, tape)));
  },
  micro(r, side, out) {
    if (r.task === "hawkes") {
      const f = r.fit;
      side.append(seal(f.time_rescaling.p_value > 0.05 && f.poisson_time_rescaling.p_value < 0.05, f.verdict, `KS p: Hawkes ${wbNum(f.time_rescaling.p_value, 3)} · Poisson ${wbNum(f.poisson_time_rescaling.p_value, 2)}`, true));
      out.append(table(["", "μ", "α", "β", "α/β", "log L"], [[t("planted"), wbNum(r.planted.mu, 4), wbNum(r.planted.alpha, 4), wbNum(r.planted.beta, 4), wbNum(r.planted.branching_ratio, 3), ""], [t("fitted"), wbNum(f.mu, 4), wbNum(f.alpha, 4), wbNum(f.beta, 4), wbNum(f.branching_ratio, 3), wbNum(f.loglik, 6)], ["Poisson", wbNum(f.events / (r.bin * r.counts.length), 4), "0", "—", "0", wbNum(f.poisson_loglik, 6)]]));
      out.append(sect(t("mk_counts"), chart([{ points: r.counts.map((c, i) => [i * r.bin, c]), cls: 0, bars: true }], { xlab: "t" })));
    } else if (r.task === "market making") {
      const c = r.certificate;
      side.append(seal(c.pnl_dispersion_ratio < 0.6, `σ(P&L) × ${wbNum(c.pnl_dispersion_ratio, 3)}`, `σ(q) × ${wbNum(c.inventory_dispersion_ratio, 3)} · ${r.runs} runs`));
      const I = r.inventory, Y = r.symmetric, P = r.paper_gamma_0_1;
      out.append(sect(t("mk_compare"), table(["", "mean P&L", "σ(P&L)", "mean q", "σ(q)", "mean spread"], [
        ["inventory", wbNum(I.mean_pnl, 4), wbNum(I.std_pnl, 4), wbNum(I.mean_q, 3), wbNum(I.std_q, 4), wbNum(I.mean_spread, 4)],
        ["symmetric", wbNum(Y.mean_pnl, 4), wbNum(Y.std_pnl, 4), wbNum(Y.mean_q, 3), wbNum(Y.std_q, 4), wbNum(Y.mean_spread, 4)],
        ["paper · inventory", P.inventory.mean_pnl, P.inventory.std_pnl, "", P.inventory.std_q, ""], ["paper · symmetric", P.symmetric.mean_pnl, P.symmetric.std_pnl, "", P.symmetric.std_q, ""]])));
    } else if (r.task === "execution") {
      side.append(seal(r.certificate.relative < 1e-10, `κ = ${wbNum(r.kappa, 4)}`, `half-life ${wbNum(r.half_life, 4)} · |closed − numeric| ${wbNum(r.certificate.max_trajectory_gap, 2)}`));
      out.append(stats(stat("E[cost]", wbNum(r.expected_cost, 6)), stat("σ[cost]", wbNum(Math.sqrt(r.variance), 6)), stat("TWAP E", wbNum(r.twap.expected_cost, 6)), stat("TWAP σ", wbNum(Math.sqrt(r.twap.variance), 6))));
      const xs = r.trajectory.map((_, i) => i);
      out.append(el("div", { class: "mk-split" },
        sect(t("mk_trajectory"), chart([{ points: pts(xs, r.trajectory), cls: 0 }, { points: pts(xs, r.numeric), cls: 1, dots: true }], { w: 420, h: 220, xlab: "period" })),
        sect(t("mk_frontier"), chart([{ points: r.frontier.map((f) => [f.std, f.expected_cost]), cls: 2, dots: true }], { w: 420, h: 220, xlab: "σ", ylab: "E" }))));
    }
  },
};

mount("hft", (root, S, redraw) => {
  head(root, "hft", "hft_lede");
  S.kind ??= "book"; S.res ??= {}; S.errs ??= {};
  S.texts ??= Object.fromEntries(Object.keys(EX_HFT).map((k) => [k, exText(Object.values(EX_HFT[k])[0])]));
  const col = new Intl.Collator(lang === "pt" ? "pt-BR" : "en");
  const kinds = Object.keys(EX_HFT).sort((a, b) => col.compare(t("h_kinds")[a], t("h_kinds")[b]));
  const kind = S.kind;
  const ed = editor("wb-hft-ed", 12); ed.value = S.texts[kind]; ed.oninput = () => (S.texts[kind] = ed.value);
  const out = el("div", { class: "wb-out", "aria-live": "polite" }), side = el("aside", { class: "wb-side" });
  const go = async () => {
    busy(out); bar.state.textContent = t("running");
    try { S.res[kind] = await api("/v1/vapor/finance", { kind, text: ed.value }); S.errs[kind] = null; } catch (e) { S.res[kind] = null; S.errs[kind] = e.message; }
    if (S.kind === kind && root.contains(out)) show();
  };
  ed.addEventListener("run", go);
  const save = btn(t("save"), "quiet", () => { if (S.res[kind]) saveArchive("finance." + kind, { text: ed.value }, S.res[kind], bar.state); });
  const bar = deskBar(EX_HFT[kind], () => t("ex_hft"), (k, text) => { ed.value = S.texts[kind] = exText(text); S.pick = 0; go(); }, [btn(t("run"), "primary", go), save]);
  root.append(seg(kinds, t("h_kinds"), kind, (k) => { S.kind = k; redraw(); }), bar.bar, el("div", { class: "wb-main" }, ed, side), out);
  function show() {
    side.replaceChildren(); out.replaceChildren();
    const err = S.errs[kind], r = S.res[kind];
    if (err) { bar.state.textContent = ""; out.append(notice(err)); localize(out); return; }
    if (!r) return;
    bar.state.textContent = r.ms != null ? t("ms", r.ms) : "";
    try { HFT_VIEW[kind](r, side, out, S, redraw); } catch (e) { out.append(notice(String(e))); }
    localize(side, out);
  }
  show();
  if (S.res[kind] === undefined && S.errs[kind] === undefined) go();
});

/* ================================================================== start */
for (const k of Object.keys(EX_FIN)) for (const ex of Object.keys(EX_FIN[k])) PALETTE_ITEMS.push(() => ({ label: t("ex_fin")[ex], sub: `${t("fin")} · ${t("f_kinds")[k]}`, run: () => { openPanel("fin"); const b = [...$("wb-fin").querySelectorAll(".seg button")].find((x) => x.textContent === t("f_kinds")[k]); if (b) b.click(); const s = $("wb-fin").querySelector(".wb-bar select"); if (s) { s.value = ex; s.dispatchEvent(new Event("change")); } } }));
for (const k of Object.keys(EX_HFT)) for (const ex of Object.keys(EX_HFT[k])) PALETTE_ITEMS.push(() => ({ label: t("ex_hft")[ex], sub: `${t("hft")} · ${t("h_kinds")[k]}`, run: () => { openPanel("hft"); const b = [...$("wb-hft").querySelectorAll(".seg button")].find((x) => x.textContent === t("h_kinds")[k]); if (b) b.click(); const s = $("wb-hft").querySelector(".wb-bar select"); if (s) { s.value = ex; s.dispatchEvent(new Event("change")); } } }));
applyLang();
sortNav();
{ const h = location.hash.slice(1); if (h === "fin" || h === "hft") openPanel(h); }
}
