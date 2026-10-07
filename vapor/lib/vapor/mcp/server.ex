defmodule Vapor.MCP.Server do
  @moduledoc """
  vapor as a **Model Context Protocol server** (stdio, JSON-RPC 2.0, one
  message per line; `mix vapor.mcp`): any MCP client — an agent, an IDE, a
  chat app — gets the studio and verifiable retrieval as tools.

  The tools, and the agent pains they answer:

  | tool | what it does | the pain |
  |---|---|---|
  | `studio_catalogue` | the node types, their typed ports and parameters | agents guess node names and parameters; here they read them |
  | `studio_validate` | type-check a graph without running it | a wrong wire found after minutes of compute |
  | `studio_run` | run a graph; outputs written as files named by their digest, small images returned inline | recomputation on every retry: the server keeps the content-addressed cache **between calls**, so an agent that edits one node re-runs only what depends on it |
  | `studio_verify` | re-run without the cache and compare the Merkle root | "did the tool really produce this?" — a result is checkable by anyone with the graph |
  | `comfy_import` | a ComfyUI API workflow → a studio graph, every translation stated | workflows shared as ComfyUI JSON |
  | `context_search` | BM25 over files of the studio directory; each hit with its Merkle proof against the corpus root | citations an agent cannot fake: a quote is checkable against the root |
  | `workbench_solve` | any problem as text — units, ODEs, PDEs, systems, fits, optimisation — with its evidence | a model's arithmetic and integration are not to be trusted; the workbench's come with residuals, step statistics, KKT, observed order |
  | `engineering_run` | circuits, power flow, frames, plane FEM, pipes, kinetics, flash, distillation | engineering answers with an independent certificate (KCL, mismatch, equilibrium, continuity) |
  | `logic_check` | settle a claim, or **check a proposal** (model, DRUP proof, colouring, counterexample) | AI-in-the-loop mathematics: the model proposes, the checker decides |
  | `board_query` | chess and shogi: legal moves, engine, analysis, mate proofs, perft | game positions answered by the rules, not recalled |
  | `render_scene` | path-trace a scene; PNG written and returned inline | physically based images on request, reproducible from the text |
  | `finance_run` | the finance and trading desk (0.13): calendars and money, curves, options, Monte Carlo, risk, portfolios, backtests with noise gates, arbitrage, the order book, an exchange session, microstructure | a model's finance is not to be trusted; each answer here carries its repricing, parity, bounds, oracle parity, backtest gates or exact LP certificate |
  | `arbitrage_check` | check a **proposed** arbitrage portfolio or state-price vector against quotes, exactly | an agent claims "this is free money": exact rational arithmetic decides |

  Tool failures are tool results with `isError: true` and the rejection
  (what was expected, how to repair) — the model reads them and corrects
  itself; protocol errors are JSON-RPC errors. Files are read and written
  only inside the studio directory (`--dir`), outputs under `--out`.
  """
  alias Vapor.{RAG, Rejection, Studio}
  alias Vapor.Studio.{Export, Value}

  @protocol "2025-06-18"

  @doc "A server state: `dir` (the studio directory), `out` (outputs), `worker`, a cache shared by all calls."
  def new(opts \\ []) do
    dir = Path.expand(Keyword.get(opts, :dir, "."))
    out = Path.expand(Keyword.get(opts, :out) || Path.join(dir, "vapor-out"))
    {:ok, cache} = Studio.Cache.start_link(max: Keyword.get(opts, :cache_size, 512))
    %{dir: dir, out: out, worker: Keyword.get(opts, :worker), cache: cache, inline_max: Keyword.get(opts, :inline_max, 1_000_000)}
  end

  @doc "Serve on stdio until end of input."
  def serve(state) do
    case IO.read(:stdio, :line) do
      :eof -> :ok
      {:error, _} -> :ok
      line ->
        state =
          case Vapor.JSON.decode(String.trim(line)) do
            {:ok, msg} ->
              {reply, state} = handle(msg, state)
              if reply, do: IO.write(:stdio, Vapor.JSON.encode(reply) <> "\n")
              state

            _ ->
              IO.write(:stdio, Vapor.JSON.encode(%{"jsonrpc" => "2.0", "id" => nil, "error" => %{"code" => -32700, "message" => "parse error"}}) <> "\n")
              state
          end

        serve(state)
    end
  end

  @doc "One JSON-RPC message → `{reply | nil, state}` (notifications get no reply)."
  def handle(%{"method" => m} = msg, state) do
    id = msg["id"]
    params = msg["params"] || %{}

    result =
      case m do
        "initialize" ->
          {:ok, %{"protocolVersion" => params["protocolVersion"] || @protocol, "capabilities" => %{"tools" => %{"listChanged" => false}},
                  "serverInfo" => %{"name" => "vapor", "version" => to_string(Application.spec(:vapor, :vsn) || "dev")},
                  "instructions" => "vapor's studio (typed media graphs: image, audio, video, 3D, diffusion, RL) and verifiable retrieval. " <>
                                      "Read studio_catalogue before writing a graph; validate, then run; results carry a Merkle root anyone can verify."}}

        "ping" -> {:ok, %{}}
        "tools/list" -> {:ok, %{"tools" => tools()}}
        "tools/call" -> {:ok, safe_call(params["name"], params["arguments"] || %{}, state)}
        "notifications/" <> _ -> :notification
        _ -> {:error, -32601, "method not found: #{m}"}
      end

    case {id, result} do
      {_, :notification} -> {nil, state}
      {nil, _} -> {nil, state}
      {id, {:ok, r}} -> {%{"jsonrpc" => "2.0", "id" => id, "result" => r}, state}
      {id, {:error, code, text}} -> {%{"jsonrpc" => "2.0", "id" => id, "error" => %{"code" => code, "message" => text}}, state}
    end
  end

  def handle(%{"id" => id}, state), do: {%{"jsonrpc" => "2.0", "id" => id, "error" => %{"code" => -32600, "message" => "invalid request"}}, state}
  def handle(_, state), do: {nil, state}

  @doc "The tool declarations (name, description, JSON Schema of the arguments)."
  def tools do
    graph = %{"type" => "object", "description" => "a studio graph: {\"nodes\": {id: {type, params, inputs: {port: [source_id, source_port]}}}}"}

    [
      %{"name" => "studio_catalogue", "description" => "List the studio's node types: typed input/output ports, parameters with ranges and defaults, docs. Filter by category (image, audio, video, vision, diffusion, rl, geom, core).",
        "inputSchema" => %{"type" => "object", "properties" => %{"category" => %{"type" => "string"}}}},
      %{"name" => "studio_validate", "description" => "Type-check a studio graph without running it: unknown nodes, bad parameters, incompatible wires and cycles are reported with the node id and the repair.",
        "inputSchema" => %{"type" => "object", "properties" => %{"graph" => graph}, "required" => ["graph"]}},
      %{"name" => "studio_run", "description" => "Run a studio graph. Every studio.output is written to a file named by its digest (PNG, GIF, WAV, GLB, …) and returned with its description; small images are also returned inline. Unchanged nodes come from the cache shared by all calls. Returns the Merkle root of the run.",
        "inputSchema" => %{"type" => "object", "properties" => %{"graph" => graph, "formats" => %{"type" => "object", "description" => "output name → format (png, ppm, gif, y4m, wav, glb, obj, ply)"}}, "required" => ["graph"]}},
      %{"name" => "studio_verify", "description" => "Re-run a graph without the cache and compare its Merkle root with the one given: proves a result came from this graph.",
        "inputSchema" => %{"type" => "object", "properties" => %{"graph" => graph, "root" => %{"type" => "string"}}, "required" => ["graph", "root"]}},
      %{"name" => "comfy_import", "description" => "Translate a ComfyUI API-format workflow (\"Save (API)\") into a studio graph; each translation is stated and untranslatable nodes are all named.",
        "inputSchema" => %{"type" => "object", "properties" => %{"workflow" => %{"type" => "object"}}, "required" => ["workflow"]}},
      %{"name" => "context_search", "description" => "Search text files of the studio directory (BM25). Each hit carries its document, offsets, and a Merkle proof against the corpus root, so a citation can be checked.",
        "inputSchema" => %{"type" => "object", "properties" => %{"query" => %{"type" => "string"}, "paths" => %{"type" => "array", "items" => %{"type" => "string"}},
                                                                 "k" => %{"type" => "integer", "minimum" => 1, "maximum" => 50}}, "required" => ["query", "paths"]}},
      %{"name" => "workbench_solve", "description" => "Solve a problem written as text (docs/BANCADA.md): formulas with units (3[kN]*2[m] in [kN*m]), ODE systems (x' = …, x(0) = …, t = 0 .. 10[s], stop when …), PDEs (u_t = …, u_tt = …, poisson …, with `verify` for a manufactured-solution order check), systems of equations (unknowns …, search = [a, b]), fits (fit y = …, data), minimisation (minimize …, subject to …, bounds like r >= 0). Units are checked before anything runs; each answer carries its own evidence (steps, residuals, KKT, observed order). ensemble: true runs `k ~ normal(…)` uncertainty on the native worker.",
        "inputSchema" => %{"type" => "object", "properties" => %{"text" => %{"type" => "string"}, "ensemble" => %{"type" => "boolean"}}, "required" => ["text"]}},
      %{"name" => "engineering_run", "description" => "Engineering solvers that read an engineer's text and return a certificate computed independently of the solver (docs/ENGENHARIA.md). kind: circuit (SPICE netlist: R C L V I D E G F H O; .op .dc .ac .tran), power (bus/line list: Newton–Raphson power flow), structure (node/support/beam/truss/load/udl/modes: 2-D frames), fem (plate/material/fix/traction: plane stress Q4/QM6), pipes (reservoir/junction/pipe: Colebrook networks), reactions (A + B -> C ; k = …: kinetics and conserved moieties), flash (Rachford–Rice VLE), distill (McCabe–Thiele/Fenske/Underwood/Gilliland).",
        "inputSchema" => %{"type" => "object", "properties" => %{"kind" => %{"type" => "string", "enum" => ~w(circuit power structure fem pipes reactions flash distill)}, "text" => %{"type" => "string"},
                                                                 "method" => %{"type" => "string", "enum" => ~w(newton gauss_seidel)}}, "required" => ["kind", "text"]}},
      %{"name" => "logic_check", "description" => "The proposer/checker desk (docs/LOGICA.md). Without `proposal`, the desk settles the claim itself with a certificate: DIMACS CNF (model or DRUP refutation), `valid: φ` / `sat: φ` / `equiv: a ; b`, `schur k`, `vdw k r`, `ramsey s t`, `pigeonhole p h`, `queens n`, equations (Knuth–Bendix; `decide s = t`), polynomials (vars/hyp/claim: Gröbner). With `proposal`, YOUR candidate is checked and never trusted: {model: [lits]} or {drup: [[lits]…]} for a CNF, {witness: [colours]} for schur/vdw, {n, red: [[a,b]…]} for ramsey, {assignment: {var: bool}} for sat:/valid:. Acceptance depends only on the checker.",
        "inputSchema" => %{"type" => "object", "properties" => %{"text" => %{"type" => "string"}, "proposal" => %{"type" => "object"}}, "required" => ["text"]}},
      %{"name" => "board_query", "description" => "Chess and shogi (docs/TABULEIROS.md): game chess with fen, or shogi with sfen; action state (legal moves), move (move: UCI/USI), engine (best reply, depth), analyse (score, principal variation), mate (chess: a proof tree of mate in n, replayed by an independent checker), perft (move-generator count).",
        "inputSchema" => %{"type" => "object", "properties" => %{"game" => %{"type" => "string", "enum" => ~w(chess shogi)}, "fen" => %{"type" => "string"}, "sfen" => %{"type" => "string"},
                                                                 "action" => %{"type" => "string", "enum" => ~w(state move engine analyse mate perft)}, "move" => %{"type" => "string"},
                                                                 "depth" => %{"type" => "integer"}, "n" => %{"type" => "integer"}}, "required" => ["game"]}},
      %{"name" => "finance_run", "description" => "The finance and trading desk (docs/FINANCAS.md). kind: calendar (du/holidays/adjust/add/yf/di1/allocate/round/factor/sum lines), curve (date, calendar, basis; di1 F26 = 15%, ltn/ntnf DATE price = …, deposit/zero DATE = r%, bond DATE coupon = … price = …, swap 5y = r%; fit nss), options (price/iv/american/heston call|put S= K= T= r= q= vol= …; smile T= F= then 'K vol' lines), mc (S= K= T= r= vol= paths= steps= barrier= threads=: GBM on the native worker, bits checked against the oracle), risk (data = t|gbm|ar1 n= sigma= seed= | csv; alpha=, window=, method=historical|normal|cornish_fisher|ewma: VaR backtest with Kupiec/Christoffersen/Basel), portfolio (assets= n= seed= cap=), backtest (data = …; sweep p = a..b step s; signal = causal expression of close; cost = 5bp: prefix-invariance look-ahead certificate, deflated Sharpe, PBO, Reality Check), arbitrage (states = …/calls T= r=/fx: an exact LP decides — a portfolio or state prices), book (buy/sell QTY @ PRICE [ioc|fok|post] owner=…, buy QTY mkt, cancel ID, modify ID qty= price=, kill OWNER: journal, naive replay, ITCH, FIX, Merkle proof), exchange (steps= seed= makers= half_spread= gamma= sigma= mu= alpha= beta= informed=), micro (hawkes mu= alpha= beta= T= | mm gamma= | execution X= N= T= sigma= eta= gamma= lambda=).",
        "inputSchema" => %{"type" => "object", "properties" => %{"kind" => %{"type" => "string", "enum" => Vapor.Finance.kinds()}, "text" => %{"type" => "string"}}, "required" => ["kind", "text"]}},
      %{"name" => "arbitrage_check", "description" => "Check YOUR proposal against quotes (the `states = …` or `calls T=… r=…` formats of finance_run's arbitrage kind): {portfolio: {asset: quantity}} — positive bought at the ask, negative sold at the bid — is accepted as an arbitrage only if its payoff is ≥ 0 in every state and it costs < 0 (or ≤ 0 with a positive payoff); {state_prices: [ψ per state]} is accepted as a no-arbitrage certificate only if every ψ > 0 and every asset's Σ Xψ lies inside its bid–ask. Numbers may be \"p/q\". Exact rational arithmetic decides, never the proposer.",
        "inputSchema" => %{"type" => "object", "properties" => %{"text" => %{"type" => "string"}, "proposal" => %{"type" => "object"}}, "required" => ["text", "proposal"]}},
      %{"name" => "alembic_eval", "description" => "Evaluate an Alembic expression (docs/ALEMBIC.md), optionally against a program of definitions. Alembic is vapor's small, sandboxed problem language: integers of any size, floats, lists, tuples, maps, comprehensions, lambdas; fuel-limited, no I/O. Returns the value as Alembic text and as JSON. The reference card is in the description of athanor_run.",
        "inputSchema" => %{"type" => "object", "properties" => %{"program" => %{"type" => "string"}, "expr" => %{"type" => "string"}}, "required" => ["expr"]}},
      %{"name" => "athanor_run", "description" => "Search any problem written in Alembic with vapor's Athanor (docs/ATHANOR.md): a portfolio of strategies (exhaustive with proof, annealing, evolution, CMA-ES, Bayesian), a random-search control at the same budget, and a certificate (verdict, best candidate, control, holdout, journal root). YOU may pass `proposals` (candidates as Alembic literals, or infix expressions for program spaces): they are checked against the space and scored by the same verifier as everything else — never trusted. Reference:\n" <> Vapor.Alembic.card(),
        "inputSchema" => %{"type" => "object", "properties" => %{"text" => %{"type" => "string"}, "budget" => %{"type" => "integer"}, "seed" => %{"type" => "integer"}, "proposals" => %{"type" => "array", "items" => %{"type" => "string"}}}, "required" => ["text"]}},
      %{"name" => "athanor_verify", "description" => "Check an Athanor certificate against its problem without trusting the search (the Touchstone): the candidate's membership in the space, its value re-computed by the verifier, the problem's hash; with full: true, the exhaustive enumeration behind an optimality or proof claim is re-run.",
        "inputSchema" => %{"type" => "object", "properties" => %{"text" => %{"type" => "string"}, "certificate" => %{"type" => "object"}, "full" => %{"type" => "boolean"}}, "required" => ["text", "certificate"]}},
      %{"name" => "game_query", "description" => "Any two-player game written in Alembic (init, player(s), moves(s), play(s, m), winner(s)): action view | play (move) | reply (vapor answers) | solve (exact value by negamax, a proof) | search (MCTS) | learn (self-play value, measured against plain search). state is an Alembic literal.",
        "inputSchema" => %{"type" => "object", "properties" => %{"text" => %{"type" => "string"}, "action" => %{"type" => "string"}, "state" => %{"type" => "string"}, "move" => %{"type" => "string"}}, "required" => ["text", "action"]}},
      %{"name" => "crucible_run", "description" => "Open science (docs/CRUCIBLE.md) on YOUR system, with evidence that needs no reference: kind quantum (V(x) = …), hamiltonian (H = …), laws (an ODE system: conservation laws proved over ℚ), reactions, evolution (N, s, i0), phylogeny (FASTA), molecule (H/He atoms), fold (HP sequence), fields (E, B), regress (CSV + target).",
        "inputSchema" => %{"type" => "object", "properties" => %{"kind" => %{"type" => "string"}, "text" => %{"type" => "string"}}, "required" => ["kind", "text"]}},
      %{"name" => "rebis_check", "description" => "Circuits over GF(2) (docs/REBIS.md). op equivalent: two netlists (`input a b`, `output s`, `s = a ^ b & ~c`, mux(s,a,b), maj(a,b,c)) or AIGER ASCII (aag) a and b — proved equal (truth table up to 16 inputs, else random simulation then a miter whose UNSAT answer carries a DRUP proof, checked) or told apart by a shrunk, re-simulated counterexample. op anf: the algebraic normal form (Zhegalkin polynomial) of each output. op identity: a word-level spec over ports (`m[16] = a[8] * b[8]`, `s[8] + 2^8*cout = a[8] + b[8]`) proved by algebra over ℤ (the circuit's Gröbner basis; polynomial for multipliers). op stabilizer: a Clifford circuit (h, s, x, y, z, cx, cz, swap, m, reset on n qubits) by the stabilizer tableau. op aiger: a netlist as AIGER.",
        "inputSchema" => %{"type" => "object", "properties" => %{"op" => %{"type" => "string", "enum" => ~w(equivalent anf identity stabilizer aiger)}, "a" => %{"type" => "string"}, "b" => %{"type" => "string"},
                                                                 "spec" => %{"type" => "string"}, "n" => %{"type" => "integer"}, "seed" => %{"type" => "integer"}}, "required" => ["op", "a"]}},
      %{"name" => "aludel_decide", "description" => "Claims about polynomials on boxes, decided in exact arithmetic (docs/ALUDEL.md). op decide: `poly` ≥ 0 (sense nonneg) or > 0 (sense pos) on `box` ([[lo, hi] per variable], numbers or fractions) over `vars` (\"x, y\"): certified (with a subdivision witness, replayed), refuted (an exact point and value) or exhausted (the cell) — never a guess. op enclose: a rigorous range. op barrier: a barrier certificate for ẋ = field (one polynomial per variable) with domain, init and unsafe boxes — check `barrier` or synthesize: true (an exact LP proposes, the decision accepts).",
        "inputSchema" => %{"type" => "object", "properties" => %{"op" => %{"type" => "string", "enum" => ~w(decide enclose barrier)}, "vars" => %{"type" => "string"}, "poly" => %{"type" => "string"},
                                                                 "box" => %{"type" => "array"}, "sense" => %{"type" => "string", "enum" => ~w(nonneg pos)}, "field" => %{"type" => "array", "items" => %{"type" => "string"}},
                                                                 "domain" => %{"type" => "array"}, "init" => %{"type" => "array"}, "unsafe" => %{"type" => "array"}, "barrier" => %{"type" => "string"},
                                                                 "synthesize" => %{"type" => "boolean"}, "degree" => %{"type" => "integer"}}, "required" => ["op"]}},
      %{"name" => "tabula_analyze", "description" => "Contracts and regulations as norms over facts (docs/TABULA.md): `parties …`, `facts …`, `exclusive a b`, `assume <formula>`, clauses `ID: [if COND then] PARTY must|must not|may|is exempt from ACTION [COUNTERPARTY]` (Portuguese: deve, não deve, pode, está isento de; se … então), `A overrides B`. Every clash (duty/prohibition, prohibition/privilege, duty/exemption, two exclusive duties) is decided by SAT: a scenario that triggers it, or a DRUP-checked proof that it never arises; overrides resolve; gaps reported. With facts ({fact: true}), the positions in force and the Hohfeld claims.",
        "inputSchema" => %{"type" => "object", "properties" => %{"text" => %{"type" => "string"}, "facts" => %{"type" => "object"}}, "required" => ["text"]}},
      %{"name" => "cupel_drill", "description" => "Silent data corruption caught by the adjoint identity y·r = x·(Wᵀr) in exact arithmetic, with a tolerance from Higham's lemma (docs/COPELA.md): a drill flips each bit position of correct products and reports what a check costing O(b·(n+k)) can and cannot see (f32), and that the int8 check is exact.",
        "inputSchema" => %{"type" => "object", "properties" => %{"n" => %{"type" => "integer"}, "k" => %{"type" => "integer"}, "seed" => %{"type" => "integer"}, "trials" => %{"type" => "integer"}, "bit" => %{"type" => "integer"}}}},
      %{"name" => "amalgam_sum", "description" => "Sum numbers (text, whitespace-separated) in several orders left to right and once exactly (docs/AMALGAMA.md): the exact sum rounded once does not depend on the order, the topology or the number of workers. format f32 or f64.",
        "inputSchema" => %{"type" => "object", "properties" => %{"numbers" => %{"type" => "string"}, "format" => %{"type" => "string", "enum" => ~w(f32 f64)}}, "required" => ["numbers"]}},
      %{"name" => "assay_run", "description" => "AI-research statistics (docs/ASSAY.md): tool compare (columns a, b), leaderboard (one column per system), contamination (JSON train/test/scores), dedup (one document per line), scaling (N, D, L), calibration (p, correct), agreement (one column per annotator), judge (ab, ba). Every answer says whether it is signal or noise.",
        "inputSchema" => %{"type" => "object", "properties" => %{"tool" => %{"type" => "string"}, "text" => %{"type" => "string"}}, "required" => ["tool", "text"]}},
      %{"name" => "scene_ops", "description" => "Parse scene operations (docs/CENA.md §9) into the operations a living scene applies; problems name each line not understood.\n" <> Vapor.Scene.Ops.card(),
        "inputSchema" => %{"type" => "object", "properties" => %{"text" => %{"type" => "string"}}, "required" => ["text"]}},
      %{"name" => "render_scene", "description" => "Path-trace a scene (docs/RENDER.md): camera pos= look= fov=; sky top= bottom=; sun dir= color= power=; sphere c= r=; plane y=; box min= max=; mat=diffuse|metal|glass|emit with albedo= rough= ior= color= power= checker=. The PNG is written under the output directory (named by its digest) and returned inline; mean linear radiance included.",
        "inputSchema" => %{"type" => "object", "properties" => %{"text" => %{"type" => "string"}, "width" => %{"type" => "integer"}, "height" => %{"type" => "integer"}, "spp" => %{"type" => "integer"}}, "required" => ["text"]}}
    ]
  end

  # ---------------------------------------------------------------- tools --

  # a crash inside a tool is the tool's error, not the server's death
  defp safe_call(name, args, st) do
    call(name, args, st)
  rescue
    e -> err("#{name} failed: " <> Exception.message(e))
  end

  defp call("studio_catalogue", args, _st) do
    cat = Studio.catalogue()
    cat = case args["category"] do
      nil -> cat
      c -> Enum.filter(cat, &(to_string(&1.category) == c))
    end

    ok(%{"nodes" => json(cat)}, "#{length(cat)} node types")
  end

  defp call("studio_validate", %{"graph" => g}, _st) do
    case Studio.validate(g) do
      {:ok, plan} -> ok(%{"valid" => true, "order" => plan.order}, "valid: #{length(plan.order)} nodes")
      {:error, why} -> err(why)
    end
  end

  defp call("studio_run", %{"graph" => g} = args, st) do
    formats = args["formats"] || %{}

    case Studio.run(g, worker: st.worker, dir: st.dir, cache: st.cache) do
      {:ok, r} ->
        File.mkdir_p!(st.out)

        files =
          for {name, v} <- Enum.sort(r.results) do
            {mime, ext, bytes} = Export.encode(v, Map.get(formats, name, :auto))
            hex = Value.hex(v)
            path = Path.join(st.out, "#{safe(name)}-#{String.slice(hex, 0, 16)}.#{ext}")
            File.write!(path, bytes)
            {name, %{"path" => path, "mime" => mime, "digest" => hex, "bytes" => byte_size(bytes), "value" => json(Value.describe(v))}, {mime, bytes}}
          end

        inline = for {_, _, {mime, b}} <- files, mime == "image/png" and byte_size(b) <= st.inline_max, do: %{"type" => "image", "data" => Base.encode64(b), "mimeType" => mime}
        data = %{"root" => r.root, "outputs" => Map.new(files, fn {n, f, _} -> {n, f} end), "executed" => r.executed, "cached" => r.cached,
                 "ms" => r.ms |> Map.values() |> Enum.sum()}
        text = "root #{r.root}; #{length(r.executed)} nodes run, #{length(r.cached)} from cache; " <>
                 Enum.map_join(files, "; ", fn {n, f, _} -> "#{n} → #{f["path"]}" end)
        %{ok(data, text) | "content" => [%{"type" => "text", "text" => text} | inline]}

      {:error, why} ->
        err(why)
    end
  end

  defp call("studio_verify", %{"graph" => g, "root" => root}, st) do
    case Studio.verify(g, root, worker: st.worker, dir: st.dir) do
      :ok -> ok(%{"verified" => true, "root" => root}, "verified: a cache-free re-run gives root #{root}")
      {:error, {:root, want, got}} -> Map.put(err("root mismatch: expected #{want}, re-run gives #{got}"), "structuredContent", %{"verified" => false, "root" => got})
      {:error, why} -> err(why)
    end
  end

  defp call("comfy_import", %{"workflow" => wf}, _st) do
    case Studio.Comfy.import(wf) do
      {:ok, g, notes} -> ok(%{"graph" => g, "notes" => notes}, "translated #{map_size(g["nodes"])} nodes" <> Enum.map_join(notes, "", &("\n- " <> &1)))
      {:error, why} -> err(why)
    end
  end

  defp call("context_search", %{"query" => q, "paths" => paths} = args, st) do
    read =
      Enum.reduce_while(paths, {:ok, []}, fn p, {:ok, acc} ->
        full = Path.expand(p, st.dir)

        cond do
          not (String.starts_with?(full, st.dir <> "/")) -> {:halt, {:error, Rejection.new({:path, p}, "a path inside the studio directory", "copy the file there")}}
          not File.regular?(full) -> {:halt, {:error, Rejection.new({:path, p}, "a readable file", "check the name")}}
          true -> {:cont, {:ok, [{p, File.read!(full)} | acc]}}
        end
      end)

    case read do
      {:ok, docs} ->
        rag = RAG.corpus(Enum.reverse(docs))
        hits = for {i, score} <- RAG.bm25(rag, q, args["k"] || 5) do
          c = RAG.chunk_with_proof(rag, i)
          %{"doc" => c.doc, "start" => c.start, "stop" => c.stop, "score" => score, "text" => c.text, "index" => i, "proof" => json(c.proof)}
        end

        ok(%{"root" => RAG.root_hex(rag), "hits" => hits},
           "corpus root #{RAG.root_hex(rag)}\n" <> Enum.map_join(hits, "\n", &"[#{&1["doc"]}:#{&1["start"]}-#{&1["stop"]}] #{String.slice(&1["text"], 0, 300)}"))

      {:error, why} ->
        err(why)
    end
  end

  defp call("alembic_eval", %{"expr" => e} = args, _st) when is_binary(e) do
    prog = case args["program"] do p when is_binary(p) -> Vapor.Alembic.load(p); _ -> {:ok, nil} end
    case prog do
      {:ok, p} ->
        case Vapor.Alembic.sandbox(fn -> Vapor.Alembic.eval(e, program: p) end, heap_mb: 256, timeout: 20_000) do
          {:ok, {:ok, v}} -> lab({:ok, %{value: Vapor.Alembic.show(v), data: Vapor.Alembic.to_data(v)}})
          {:ok, {:error, er}} -> err(Vapor.Alembic.format_error(er))
          {:error, w} -> err("stopped: #{inspect(w)}")
        end
      {:error, er} -> err(Vapor.Alembic.format_error(er))
    end
  end

  defp call("athanor_run", %{"text" => t} = args, _st) when is_binary(t) do
    proposals =
      (args["proposals"] || [])
      |> Enum.take(64)
      |> Enum.flat_map(fn p when is_binary(p) -> (case Vapor.Alembic.literal(p) do {:ok, v} -> [v]; _ -> [p] end); _ -> [] end)
    opts = [seconds: 120, proposals: proposals] ++ Enum.flat_map(["budget", "seed"], fn k -> if is_integer(args[k]) and args[k] > 0, do: [{String.to_existing_atom(k), min(args[k], 500_000)}], else: [] end)
    lab(Vapor.Athanor.run(t, opts) |> then(fn {:ok, c} -> {:ok, Vapor.Main.jsonable(c)}; e -> e end))
  end

  defp call("athanor_verify", %{"text" => t, "certificate" => c} = args, _st) when is_binary(t) and is_map(c),
    do: lab(Vapor.Console.Lab14.verify(%{"text" => t, "certificate" => c, "full" => args["full"] == true}) |> then(fn {:ok, v} -> {:ok, Vapor.Main.jsonable(v)}; e -> e end))

  defp call("game_query", %{"text" => _} = args, _st), do: lab(Vapor.Console.Lab14.game(args) |> then(fn {:ok, v} -> {:ok, Vapor.Main.jsonable(v)}; e -> e end))
  defp call("crucible_run", %{"kind" => _, "text" => _} = args, _st), do: lab(Vapor.Console.Lab14.crucible(args) |> then(fn {:ok, v} -> {:ok, Vapor.Main.jsonable(v)}; e -> e end))
  defp call(name, args, _st) when name in ~w(rebis_check aludel_decide tabula_analyze cupel_drill amalgam_sum) and is_map(args) do
    alias Vapor.Console.Lab15, as: L
    f = %{"rebis_check" => &L.rebis/1, "aludel_decide" => &L.aludel/1, "tabula_analyze" => &L.tabula/1, "cupel_drill" => &L.cupel/1, "amalgam_sum" => &L.amalgam/1}[name]
    lab(f.(args) |> then(fn {:ok, v} -> {:ok, Vapor.Main.jsonable(v)}; e -> e end))
  end

  defp call("assay_run", %{"tool" => _, "text" => _} = args, _st), do: lab(Vapor.Console.Lab14.assay(args) |> then(fn {:ok, v} -> {:ok, Vapor.Main.jsonable(v)}; e -> e end))
  defp call("scene_ops", %{"text" => _} = args, _st), do: lab(Vapor.Console.Lab14.scene_ops(args) |> then(fn {:ok, v} -> {:ok, Vapor.Main.jsonable(v)}; e -> e end))

  defp call("workbench_solve", %{"text" => _} = args, _st), do: lab(Vapor.Console.Lab12.solve(args))
  defp call("engineering_run", %{"kind" => _, "text" => _} = args, _st), do: lab(Vapor.Console.Lab12.engineering(args))

  defp call("logic_check", %{"text" => t, "proposal" => p}, _st) when is_binary(t) and is_map(p) do
    case Vapor.Logic.check(t, p) do
      {:ok, r} -> ok(json(r), "#{if r.accepted, do: "ACCEPTED", else: "REJECTED"}: #{r.claim} — #{r.reason}")
      {:error, why} -> err(why)
    end
  end

  defp call("logic_check", %{"text" => _} = args, _st), do: lab(Vapor.Console.Lab12.logic(args))
  defp call("board_query", %{"game" => "shogi"} = args, _st), do: lab(Vapor.Console.Lab12.shogi(args))
  defp call("board_query", %{"game" => "chess"} = args, _st), do: lab(Vapor.Console.Lab12.chess(args))

  defp call("render_scene", %{"text" => _} = args, st) do
    case Vapor.Console.Lab12.render(args) do
      {:ok, r} ->
        png = r.png |> String.replace_prefix("data:image/png;base64,", "") |> Base.decode64!()
        File.mkdir_p!(st.out)
        file = Path.join(st.out, (:crypto.hash(:sha256, png) |> Base.encode16(case: :lower) |> binary_part(0, 16)) <> ".png")
        File.write!(file, png)
        data = %{"file" => file, "width" => r.w, "height" => r.h, "spp" => r.spp, "mean_radiance" => r.mean, "ms" => r.ms}
        %{"content" => [%{"type" => "text", "text" => "#{file} · #{r.w}×#{r.h} · #{r.spp} spp · mean radiance #{Float.round(r.mean, 5)}"},
                        %{"type" => "image", "data" => Base.encode64(png), "mimeType" => "image/png"}],
          "structuredContent" => data, "isError" => false}
      {:error, why} -> err(why)
    end
  end

  defp call("finance_run", %{"kind" => _, "text" => _} = args, _st), do: lab(Vapor.Console.Lab13.finance(args))

  defp call("arbitrage_check", %{"text" => t, "proposal" => p}, _st) when is_binary(t) and is_map(p) do
    case Vapor.Finance.Arbitrage.check(t, p) do
      {:ok, r} -> ok(json(r), "#{if r.accepted, do: "ACCEPTED", else: "REJECTED"}: #{r.claim} — #{r.reason}")
      {:error, why} -> err(why)
    end
  end

  defp call(name, _args, _st) when name in ["studio_validate", "studio_run", "studio_verify", "comfy_import", "context_search", "workbench_solve", "engineering_run", "logic_check", "board_query", "render_scene", "finance_run", "arbitrage_check"],
    do: err("missing required arguments for #{name}")

  defp call(name, _, _), do: err("unknown tool #{inspect(name)}")

  # a laboratory's answer: the whole result as structured content, a bounded text rendering for the model
  defp lab({:ok, r}), do: (j = json(r); text = Vapor.JSON.encode(j); ok(j, if(byte_size(text) > 6000, do: binary_part(text, 0, 6000) <> " …(truncated; see structuredContent)", else: text)))
  defp lab({:error, why}) when is_binary(why), do: err(why)
  defp lab(other), do: err(inspect(other))

  defp ok(data, text), do: %{"content" => [%{"type" => "text", "text" => text}], "structuredContent" => data, "isError" => false}

  defp err(%Rejection{} = r), do: err("expected #{r.bound} at #{inspect(r.node)}; repair: #{r.repair}")
  defp err(text) when is_binary(text), do: %{"content" => [%{"type" => "text", "text" => text}], "isError" => true}
  defp err(other), do: err(inspect(other))

  defp safe(name), do: String.replace(to_string(name), ~r/[^A-Za-z0-9_.-]/, "_")

  # Elixir terms → JSON-shaped data (atoms and tuples become strings and lists)
  defp json(m) when is_map(m) and not is_struct(m), do: Map.new(m, fn {k, v} -> {to_string(k), json(v)} end)
  defp json(l) when is_list(l), do: Enum.map(l, &json/1)
  defp json(t) when is_tuple(t), do: t |> Tuple.to_list() |> json()
  defp json(b) when is_binary(b), do: (if String.valid?(b), do: b, else: Base.encode16(b, case: :lower))
  defp json(a) when is_atom(a) and a not in [nil, true, false], do: to_string(a)
  defp json(x), do: x
end
