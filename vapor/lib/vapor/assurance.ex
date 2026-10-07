defmodule Vapor.Assurance do
  @moduledoc """
  The **assurance ledger** (ASAS §11, adopted): vapor's correctness claims,
  each with its basis and an honest status — what is *proved*, what is
  *checked* by code on every build or on every answer, what is *tested*,
  what is *argued*, and what is still *owed*. The ledger is data; the
  document (`docs/GARANTIAS.md`) is rendered from it (`mix vapor.assurance`),
  and a test fails when a cited file is missing or the document drifts from
  the data — so the ledger cannot quietly overstate.

  Statuses:

    * `proved` — a machine-checked proof (Lean 4, no axiom, no `sorry`);
    * `checked` — an independent checker runs on every answer or every
      build (a DRUP checker, a witness replay, a source audit);
    * `tested` — differential or property tests with controls, on the
      substrates and inputs present;
    * `argued` — a rigorous argument, not mechanised;
    * `owed` — stated and not done.
  """

  @claims [
    # ---- the core
    {"core", "Every substrate produces the same bits under the canonical policy", "differential suites against the exact oracle on x86 AVX2/AVX-512, AArch64 and RVV (emulated), SPIR-V (lavapipe)", :tested,
     ["test/vapor/substrate_test.exs", "test/vapor/fabric_test.exs", "test/vapor/canonical_test.exs"], "RVV and NEON run under emulation here; no discrete GPU"},
    {"core", "Higham's bound on any summation order (γₖ envelope)", "Lean 4, no axiom", :proved, ["proofs/Vapor/Higham.lean"], nil},
    {"core", "Wilkinson's bound for the canonical reductions", "Lean 4, no axiom", :proved, ["proofs/Vapor/Wilkinson.lean"], nil},
    {"core", "Rewrite rules preserve IEEE-754 binary32 semantics (finite values)", "Lean 4 model of binary32 bits; NaN excluded by principle", :proved, ["proofs/Vapor/Binary32.lean", "test/vapor/rewrite_soundness_test.exs"], "constant folding is tested against the oracle, not proved"},
    {"core", "Register allocation is checked, not trusted", "a checker proved in Lean, extracted to Elixir; extraction freshness audited", :proved, ["proofs/Vapor/RegAlloc.lean", "test/vapor/regalloc_test.exs", "test/vapor/extracted_conformance_test.exs"], nil},
    {"core", "Generated machine code never runs inside the BEAM (only isolated worker processes); no shell; no dependencies", "source audit on every build", :checked, ["test/vapor/audit_test.exs"], nil},
    {"core", "One entropy boundary: OS randomness only through Vapor.Entropy, process generators seeded", "source audit on every build", :checked, ["lib/vapor/entropy.ex", "test/vapor/audit_test.exs"], nil},
    # ---- 0.15
    {"0.15", "Amalgam: the correctly rounded exact sum, whatever the order or grouping", "exact integer cells; rounding compared with exact rationals", :tested, ["test/vapor/amalgam_test.exs", "test/vapor/train_exact_test.exs"], nil},
    {"0.15", "Cupel never accuses a conforming substrate", "Higham's bound (proved) projected on |r|; four summation orders at six scales", :proved, ["proofs/Vapor/Higham.lean", "test/vapor/cupel_test.exs"], "detection below the rounding envelope is impossible and measured, not claimed"},
    {"0.15", "Cupel catches a corrupted element above the envelope", "probability ≥ 1 − 2⁻²⁰ per row for a forger without the seed", :argued, ["lib/vapor/cupel.ex", "test/vapor/cupel_test.exs"], nil},
    {"0.15", "Rebis 'equivalent' by SAT carries a DRUP proof checked by separate code", "Vapor.Logic.DRUP on every UNSAT answer", :checked, ["lib/vapor/logic/drup.ex", "test/vapor/rebis_test.exs"], nil},
    {"0.15", "Rebis word identities: remainder 0 ⇔ the identity holds", "the gate polynomials form a Gröbner basis under reverse-topological lex order (Lv–Kalla–Enescu)", :argued, ["lib/vapor/rebis/ideal.ex", "test/vapor/rebis_test.exs"], "the substitution trace is not independently re-checked"},
    {"0.15", "Aludel 'certified' is replayed by direct per-leaf conversion", "Aludel.check/4 on every answer; Bernstein enclosure is a classical theorem", :checked, ["lib/vapor/aludel.ex", "test/vapor/aludel_test.exs"], nil},
    {"0.15", "Tabula consistency proofs are DRUP-checked", "Vapor.Logic.DRUP on every pair proved apart", :checked, ["lib/vapor/tabula.ex", "test/vapor/tabula_test.exs"], nil},
    {"0.15", "JBIG2 decoding equals jbig2dec bit for bit (or T.88 where jbig2dec departs)", "64 fixtures from an independent encoder", :tested, ["test/vapor/jbig2_test.exs", "test/python/jbig2_streams.py"], nil},
    # ---- 0.16
    {"0.16", "The khazāna opens at the old or the new root after a crash at any byte", "fault injection at every byte of the pack append and of the root slot", :tested, ["lib/vapor/khazana.ex", "test/vapor/khazana_test.exs"], "assumes datasync is honest and a SHA-256 tag cannot collide (the platform's half of the contract)"},
    {"0.16", "A conversation message commits to its whole history", "its hash covers its parent's hash (SHA-256)", :argued, ["lib/vapor/majlis.ex", "test/vapor/majlis_test.exs"], nil},
    {"0.16", "An altered vapor export is refused on import", "every message hash recomputed", :checked, ["lib/vapor/majlis/exchange.ex", "test/vapor/majlis_test.exs"], nil},
    {"0.16", "Shared links cannot be forged and die on revocation", "HMAC-SHA256 under the store's key; generation counter", :argued, ["lib/vapor/khazana.ex", "test/vapor/hall_test.exs"], "reduces to the PRF security of HMAC-SHA256"},
    {"0.16", "The console terminal cannot read or write the server's files or run programs", "jailed reads and writes, --measure refused, heap ceiling and deadline per command", :tested, ["lib/vapor/diwan.ex", "test/vapor/diwan_test.exs", "test/vapor/hall_test.exs"], "rests on the BEAM's process isolation and on every verb reading through Vapor.Main.read_input"},
    {"0.16", "Al-Mizān: the Latin and Arabic projections are bijective with the tree", "read(print(t)) = t on 300 random programs in both scripts", :tested, ["lib/vapor/mizan/syntax.ex", "test/vapor/mizan_test.exs"], nil},
    {"0.16", "Al-Mizān burhān: identities and conservation laws decided exactly", "exact polynomial normal form over ℚ; positivity by Aludel; invariants by SAT + DRUP", :checked, ["lib/vapor/mizan.ex", "test/vapor/mizan_test.exs"], nil},
    {"0.16", "Al-Mizān obligations exported to Lean 4 close by ring / decide", "Lower.lean/2 emits the theorems", :owed, ["lib/vapor/mizan/lower.ex"], "Lean is not installed on this machine: the export is generated, not checked here"},
    {"0.16", "Fisher–Rao distances are a metric; natural gradient is invariant to feature scaling", "property tests against controls (KL; plain gradient)", :tested, ["lib/vapor/info_geom.ex", "test/vapor/info_geom_test.exs"], nil},
    {"0.16", "The language server answers editors over real stdio framing", "a Node client against bin/vapor lsp", :tested, ["lib/vapor/lsp.ex", "test/vapor/lsp_test.exs", "test/js/lsp_client.mjs"], "VS Code, Neovim and Emacs themselves are not run here"},
    # ---- what is owed
    {"owed", "Weak-memory correctness of the worker's shared-memory protocol", "litmus tests against a RVWMO/TSO model", :owed, ["native/src"], "the worker uses pipes and /dev/shm with explicit synchronisation; no axiomatic-model check exists yet"},
    {"owed", "Constant folding proved, not only tested", "extend Binary32.lean", :owed, ["proofs/Vapor/Binary32.lean"], nil}
  ]

  @doc "Every claim as a map."
  def claims do
    for {area, claim, basis, status, evidence, limits} <- @claims,
        do: %{area: area, claim: claim, basis: basis, status: status, evidence: evidence, limits: limits}
  end

  @doc "Claims whose cited evidence is missing from the tree (relative to `root`)."
  def missing(root \\ ".") do
    for c <- claims(), path <- c.evidence, not File.exists?(Path.join(root, path)), do: {c.claim, path}
  end

  @labels %{proved: "**provado**", checked: "**conferido**", tested: "testado", argued: "argumentado", owed: "*devido*"}

  @doc "The ledger as Markdown (docs/GARANTIAS.md)."
  def markdown do
    counts = claims() |> Enum.frequencies_by(& &1.status)

    rows =
      claims()
      |> Enum.group_by(& &1.area)
      |> Enum.sort_by(fn {a, _} -> Enum.find_index(["core", "0.15", "0.16", "owed"], &(&1 == a)) end)
      |> Enum.map_join("\n\n", fn {area, cs} ->
        "## #{%{"core" => "O núcleo", "0.15" => "Rodada 0.15", "0.16" => "Rodada 0.16", "owed" => "O que é devido"}[area]}\n\n" <>
          "| afirmação | base | estado | evidência | limites |\n|---|---|---|---|---|\n" <>
          Enum.map_join(cs, "\n", fn c ->
            "| #{c.claim} | #{c.basis} | #{@labels[c.status]} | #{Enum.map_join(c.evidence, " · ", &"`#{&1}`")} | #{c.limits || "—"} |"
          end)
      end)

    """
    # Garantias — o livro-razão

    > Gerado de `lib/vapor/assurance.ex` por `mix vapor.assurance`; um teste falha se a evidência citada sumir ou se
    > este arquivo divergir dos dados. A disciplina é a do ASAS (§11): cada afirmação diz se é **provada** (Lean 4,
    > sem axioma), **conferida** (um verificador independente roda a cada resposta ou a cada build), testada,
    > argumentada ou *devida*.

    #{Enum.map_join([:proved, :checked, :tested, :argued, :owed], " · ", &"#{@labels[&1]}: #{Map.get(counts, &1, 0)}")}

    #{rows}
    """
  end
end
