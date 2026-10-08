defmodule Vapor.Assurance do
  @moduledoc """
  The **assurance ledger** (ASAS §11, adopted): vapor's correctness claims,
  each with its basis and an honest status — what is *proved*, what is
  *checked* by code on every build or on every answer, what is *tested*,
  what is *argued*, and what is still *owed*. The ledger is data; the
  document (`docs/ASSURANCE.md`) is rendered from it (`mix vapor.assurance`),
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
    {"0.16", "Almizan: the Latin and Arabic projections are bijective with the tree", "read(print(t)) = t on 300 random programs in both scripts", :tested, ["lib/vapor/almizan/syntax.ex", "test/vapor/almizan_test.exs"], nil},
    {"0.16", "Almizan burhān: identities and conservation laws decided exactly", "exact polynomial normal form over ℚ; positivity by Aludel; invariants by SAT + DRUP", :checked, ["lib/vapor/almizan.ex", "test/vapor/almizan_test.exs"], nil},
    {"0.16", "Almizan obligations exported to core Lean 4 close by grind / decide; a refuted claim's statement does not", "Lower.lean/2 emits theorems over Rat and Bool with no Mathlib; the :lean tier runs Lean on the example modules' exports and on a refuted one (0.17)", :tested, ["lib/vapor/almizan/lower.ex", "test/vapor/almizan_test.exs"], "positivity on a box has no core-Lean proof: it is exported as a statement, its certificate is Aludel's witness"},
    {"0.16", "Fisher–Rao distances are a metric; natural gradient is invariant to feature scaling", "property tests against controls (KL; plain gradient)", :tested, ["lib/vapor/info_geom.ex", "test/vapor/info_geom_test.exs"], nil},
    {"0.16", "The language server answers editors over real stdio framing", "a Node client against bin/vapor lsp", :tested, ["lib/vapor/lsp.ex", "test/vapor/lsp_test.exs", "test/js/lsp_client.mjs"], "VS Code, Neovim and Emacs themselves are not run here"},
    # ---- 0.17
    {"0.17", "The Lean development builds warning-free on Lean 4.34.1, and re-extraction is byte-identical", "lake build; the extracted module compared with the sources' digest", :checked, ["proofs/lean-toolchain", "test/vapor/audit_test.exs"], nil},
    {"0.17", "The hermetic seal stops a job whose heap and off-heap binaries together exceed its cap, or that misses its deadline; the caller survives", "max_heap_size with include_shared_binaries; a 512 MB binary bomb under 64 MB against the 0.16 heap-only cap", :tested, ["lib/vapor/hermetic.ex", "test/vapor/hermetic_test.exs"], "bounds memory and time; does not resist code with access to the BEAM itself"},
    {"0.17", "An integer program's 'optimal' or 'infeasible' carries a branch-and-bound tree checked by code that shares nothing with the search", "MIP.check/2 on every answer: the splits cover the integer points, every leaf's Farkas or dual certificate, the incumbent", :checked, ["lib/vapor/logic/mip.ex", "test/vapor/mip_test.exs"], "branching only (no cutting planes); a relaxation unbounded at the root is refused"},
    {"0.17", "Causal estimands equal the true intervention", "exact rationals on random structural causal models with explicit hidden parents; the naive P(y | x) as control", :tested, ["lib/vapor/logic/causal.ex", "test/vapor/causal_test.exs"], "every verdict is conditional on the stated diagram; ID's completeness is a published theorem (Shpitser & Pearl, 2006), not mechanised"},
    {"0.17", "A causal identification that fails carries a hedge, checked", "Causal.hedge?/5 on every failure", :checked, ["lib/vapor/logic/causal.ex", "test/vapor/causal_test.exs"], nil},
    {"0.17", "Qālib: a netlist and its specification are called equivalent only with a truth table or a DRUP-checked SAT proof", "Vapor.Rebis.equivalent/3; mapped netlists are read back and proved before they are printed", :checked, ["lib/vapor/qalib.ex", "test/vapor/qalib_test.exs"], "combinational only; each cell's function is tested against its formula, not read from the liberty files"},
    {"0.17", "Palingenesis: a published generation passed every gate, and its lineage re-derives from launch", "the gates measured before publication; verify/2 recomputes the record chain and the root from the weights", :tested, ["lib/vapor/palingenesis.ex", "test/vapor/palingenesis_test.exs"], "drift is measured on the anchors only; measured on a tiny model here, not on production checkpoints"},
    {"0.17", "Palingenesis: a reader keeps its generation whole while new ones are published", "immutable generations published by one persistent_term put; concurrent readers re-hash what they hold", :tested, ["lib/vapor/palingenesis.ex", "test/vapor/palingenesis_test.exs"], "at the BEAM level; a worker's shared-memory session sees the new generation at its next session"},
    {"0.17", "Recommendations are called signal only with a significant, minimum-size gain over the biases and a shuffled control at the mean", "paired sign-flip test, a 1 % minimum gain, a shuffled-ratings control, nested selection", :tested, ["lib/vapor/recommend.ex", "test/vapor/recommend_test.exs"], nil},
    {"0.17", "The one-sided Jacobi SVD comes with its residual and orthogonality, and keeps small singular values to relative precision", "Dense.svd_residual/2; σ_min of a κ = 10⁸ matrix against the Gram-matrix route", :tested, ["lib/vapor/dense.ex", "test/vapor/recommend_test.exs"], nil},
    {"0.17", "The delta-rule hybrid (Kimi K3's topology) computes what its report's equations define", "an independent float64 reference written from the report alone: logits within 1.35·10⁻⁵, greedy identical, native = oracle; forgetting either memory fails", :tested, ["lib/vapor/lock/adapters/delta_hybrid.ex", "test/vapor/kimi_test.exs", "test/python/kimi_k3_reference.py"], "tiny models; the release's own spelling (config keys, tensor names) is not verified"},
    {"0.17", "MXFP4 decodes exactly to binary32; a value that cannot be represented is refused", "all sixteen E2M1 values under every finite scale; NaN scales and overflowing elements refused by name", :tested, ["lib/vapor/quant/mxfp4.ex", "test/vapor/kimi_test.exs"], "the layout read is gpt-oss's"},
    {"0.17", "An agent cannot fetch from the network: it can only queue a request the person approves", "one MCP tool, which writes a queue entry; outside the siphon only the agent backends open connections", :tested, ["lib/vapor/siphon.ex", "test/vapor/siphon_test.exs"], "the fetchers themselves are the person's programs, bounded in time, bytes and environment, not audited"},
    {"0.17", "A balanced top-k assignment is certified optimal", "the b-matching LP solved by the rational simplex with its dual certificate; Algorithm 1's ties shown, midpoints converge", :checked, ["lib/vapor/train/balance.ex", "test/vapor/balance_test.exs"], nil},
    {"0.17", "A cached test file passed on the same bytes of everything it can reach", "keys over the file, support, fixtures, priv/, the native binaries, the toolchain, the transitive bytecode closure and the repository trees the file names; records only full passes", :tested, ["lib/vapor/test_cache.ex", "test/vapor/test_cache_test.exs"], "a module built from a computed name, or a file read through a path the test's text does not name, would escape the key; mix test ignores the cache"},
    {"0.17", "The documents name no module, function or repository path that does not exist", "every Vapor.… reference and every lib/, test/, priv/ path in docs/ and the README, resolved", :tested, ["test/vapor/docs_references_test.exs"], "DIRECTIVE.md keeps the paths of past rounds"},
    # ---- what is owed
    {"owed", "Palingenesis measured on production checkpoints (Qwen 2.5, DeepSeek-V3)", "vapor palingenesis try on real models", :owed, ["lib/vapor/palingenesis.ex"], "this machine has neither the weights nor the memory"},
    {"owed", "Sequential equivalence (latches, flip-flops) for Qālib", "k-induction on the miter, IC3/PDR", :owed, ["lib/vapor/qalib.ex"], nil},
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

  @labels %{proved: "**proved**", checked: "**checked**", tested: "tested", argued: "argued", owed: "*owed*"}

  @doc "The ledger as Markdown (docs/ASSURANCE.md)."
  def markdown do
    counts = claims() |> Enum.frequencies_by(& &1.status)

    rows =
      claims()
      |> Enum.group_by(& &1.area)
      |> Enum.sort_by(fn {a, _} -> Enum.find_index(["core", "0.15", "0.16", "0.17", "owed"], &(&1 == a)) end)
      |> Enum.map_join("\n\n", fn {area, cs} ->
        "## #{%{"core" => "The core", "0.15" => "Round 0.15", "0.16" => "Round 0.16", "0.17" => "Round 0.17", "owed" => "What is owed"}[area]}\n\n" <>
          "| claim | basis | status | evidence | limits |\n|---|---|---|---|---|\n" <>
          Enum.map_join(cs, "\n", fn c ->
            "| #{c.claim} | #{c.basis} | #{@labels[c.status]} | #{Enum.map_join(c.evidence, " · ", &"`#{&1}`")} | #{c.limits || "—"} |"
          end)
      end)

    """
    # Assurance — the ledger

    > Generated from `lib/vapor/assurance.ex` by `mix vapor.assurance`; a test fails if cited evidence disappears or if
    > this file drifts from the data. The discipline is the ASAS's (§11): each claim says whether it is **proved** (Lean 4,
    > no axiom), **checked** (an independent checker runs on every answer or every build), tested,
    > argued or *owed*.

    #{Enum.map_join([:proved, :checked, :tested, :argued, :owed], " · ", &"#{@labels[&1]}: #{Map.get(counts, &1, 0)}")}

    #{rows}
    """
  end
end
