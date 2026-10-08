# The command line — the whole workbench from the terminal

> Since 0.14.0. Code: `lib/vapor/main.ex`, `lib/vapor/main/*`, `bin/vapor`, `mix vapor`.

Everything the console does, the terminal does, in the Unix philosophy: each command reads a file or
standard input (`-`), writes text for people at a terminal and **JSON when the output is a
*pipe*** (or with `--json`), and reports the result through the exit code.

```
vapor alembic FILE | -e EXPR | --card
vapor athanor run FILE [--budget N --seed N --seconds N --only s1,s2 --set k=v --mind SPEC
                        --measure CMD --interactive --no-control --json]
vapor athanor ask FILE                  # measured: vapor proposes, you measure
vapor verify FILE CERT [--full --replay]
vapor game FILE solve|search|learn|play
vapor crucible KIND FILE                # `vapor crucible` lists; --example shows one
vapor assay TOOL FILE                   # `vapor assay` lists; --example shows one
vapor mind ask|formalize|transcript     # --model or VAPOR_MIND
vapor render FILE [--ink] [--out F.png]  # a scene: path-traced, or in ink (0.17)
vapor qalam FILE                         # Al-Qalam, the editor: vi keys, verdicts in the gutter (0.17)
vapor solve FILE                        # the equation workbench
vapor rebis equiv|anf|identity|stabilizer|aiger FILE…   # circuits over GF(2) (0.15)
vapor aludel decide P --vars x,y --box '0,1;0,1' | REQUEST.json   # positivity, barriers
vapor tabula FILE [--facts a,b]          # a contract: antinomies, proofs, gaps
vapor cupel [--n 32 --k 64 --bit 26]     # the silent-corruption drill
vapor amalgam FILE|- [--f32]             # a sum that does not depend on the order
vapor chat new|say|show|edit|regen|switch|rewind|fork|pin|context|compact|search|export|import|share …  (0.16)
vapor wzn check|show|hash|run|transmute|assay|abjad|fmt FILE …   # Almizan (0.16): decided claims; fmt (0.17)
vapor logic FILE | check FILE PROPOSAL.json   # the logic desk (0.17): SAT, LP, integer LP, causal, Gröbner…
vapor qalib map|check …                  # sky130 netlists, mapped and proved equal (0.17)
vapor recommend RATINGS.csv              # factorisation with baselines and a control (0.17)
vapor palingenesis planks|try MODEL …    # renew a model plank by plank (0.17)
vapor siphon fetchers|queue|approve|reject|run|headers|preflight …   # the network airlock (0.17)
vapor lsp                                # the language server (VS Code, Neovim, Emacs, …)
vapor serve | tui | ocr | merge | quality …   # the earlier mix tasks
```

| code | meaning |
|---|---|
| 0 | positive: found, proved, verified, signal |
| 1 | negative: refuted, not found, verification failed, noise |
| 2 | wrong usage |
| 3 | invalid input (with line and column) |
| 4 | failure |

`NO_COLOR` turns colour off; `VAPOR_TTY=0|1` forces the mode. Composition examples:

```sh
# Euler's conjecture refuted, the certificate checked by another process
vapor athanor run euler.nbq > c.json || vapor verify euler.nbq c.json && echo "real counterexample"

# the model drafts, the person reads the back-translation, the furnace searches
echo "the shortest Golomb ruler with 8 marks" | vapor mind formalize - > g.nbq && vapor athanor run g.nbq

# an objective measured by an external program (a training run, a simulation)
vapor athanor run hp.nbq --measure './train.sh'   # reads $VAPOR_CANDIDATE_JSON, prints the number (the last one in the output)

# a scene drawn in ink in milliseconds, then with physical light
vapor render studio.txt --ink --out sketch.png && vapor render studio.txt --spp 64 --out final.png
```

## Opus from the terminal (0.15)

The same exit codes: `0` is equivalent, proved, certified or consistent; `1` is
different, refuted, exhausted or with antinomies — so a *script* can require the proof.

```sh
# the netlist after synthesis against the specification; the counterexample grouped into words
vapor rebis equiv spec.net synth.aag || echo "not the same function"

# a multiplier proved by algebra over ℤ (where SAT is exponential)
vapor rebis identity mul16.net --spec 'm[32] = a[16] * b[16]'

# a strict claim about a polynomial, with the reproducible witness in the JSON
vapor aludel decide 'x^4*y^2 + x^2*y^4 - 3*x^2*y^2 + 1 + 1/1000' --vars x,y --box '-2,2;-2,2' --strict --json > w.json

# a contract: antinomies with the scenario; the positions when delivery was late
vapor tabula venda.txt --facts delivered,late
```

## Conversations, Almizan and the single terminal (0.16)

```sh
# a conversation in a file of your own (~/.vapor/majlis, or $VAPOR_HOME), with the model from VAPOR_MIND
T=$(vapor chat new --title draft --system "answer in Portuguese")
vapor chat say $T "what is a Gröbner basis?"
vapor chat edit $T 3fa2c1 "and a small example?"     # a branch; the old one stays
vapor chat context $T                                # what the model will read, and what is left out
vapor chat export $T --md > rascunho.md

# a conservation law proved over ℚ; the damped version refuted at the point
vapor wzn check priv/almizan/oscillator.wzn
vapor wzn show priv/almizan/oscillator.wzn --arabic | vapor wzn hash -   # the same hash
```

The command line, the TUI (`mix vapor.tui`), the console terminal and `POST /v1/vapor/diwan` are the
same interpreter — the Dīwān ([DIWAN.md](DIWAN.md)): pipes, redirection, `;`, quotes and files;
in the console, a jailed session. Conversations: [MAJLIS.md](MAJLIS.md); Almizan: [ALMIZAN.md](ALMIZAN.md);
editors: [EDITORS.md](EDITORS.md).

## Decisions, netlists, recommendations and planks (0.17)

```sh
# an integer program: the optimum and the tree that proves it (exit 0); anyone's proposal, checked
vapor logic knapsack.lp | jq .objective
vapor logic check knapsack.lp my_proposal.json          # exit 0 accepted, 1 refused

# a causal question on a stated diagram: the estimand, or the hedge (exit 1)
printf 'causal\nx -> m\nm -> y\nx <-> y\nidentify y | do(x)\n' | vapor logic -

# a netlist written in sky130 cells, read back and proved equal before it is printed
vapor qalib map adder.net --style nand > adder.v
vapor qalib check adder.net yosys_out.v                 # exit 1 with the input that tells them apart

# recommendations, with the evidence: exit 0 only for signal
vapor recommend ratings.csv --top 5

# renew one plank from a donor checkpoint, through the brake and the target test
vapor palingenesis planks ./Model
vapor palingenesis try ./Model --plank model.layers.12.mlp --from ./Finetuned \
      --anchors keep.txt --targets improve.txt --epsilon 0.05 --out ./Model-gen1

# Almizan in canonical form, comments kept (exit 1 under --check when it is not)
vapor wzn fmt claims.wzn --check || vapor wzn fmt claims.wzn --write

# the network airlock: you declare the fetchers, an agent may only propose, you approve
vapor siphon queue                                       # what agents asked for, with their reasons
vapor siphon approve 4be1c09a7f3e --sha256 9f2c…         # runs it now, pinned; anything else stays queued
vapor siphon preflight hub config.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors
                                                         # the airlock's verdict from the headers alone
```

Logic: [LOGIC.md §6–§7](LOGIC.md); netlists: [QALIB.md](QALIB.md); recommendations:
[RECOMMEND.md](RECOMMEND.md); planks: [PALINGENESIS.md](PALINGENESIS.md); the siphon:
[SIPHON.md](SIPHON.md); render: [RENDER.md](RENDER.md). In the console's terminal (the jail)
`palingenesis` is refused, because it reads models from the server's disk, `render` because it
writes a file on the server (the console has its render desk), `qalam` because it needs a terminal
of one's own, and `siphon` is not offered at all: fetching is the person's act at their own terminal. The other verbs read only the session's
files.

## Agents and terminal console

The same verbs are MCP tools (`mix vapor.mcp`: `alembic_eval`, `athanor_run` with
`proposals`, `athanor_verify`, `game_query`, `crucible_run`, `assay_run`; in 0.15
`rebis_check`, `aludel_decide`, `tabula_analyze`, `cupel_drill`, `amalgam_sum`; in 0.17
`qalib_check`, `recommend_run` and `siphon_propose` (which only queues a fetch for the person to
approve), with `logic_check` now covering LP, integer programs and causal diagrams, `render_scene`
taking `style: "ink"`, and `scene_ops` gone with the living scene — 27 in all) and TUI commands (`mix vapor.tui`: `alembic -e "…"`, `athanor run file`, …). An
agent proposes; the Touchstone checks — the same door for people, models and programs.

## Mind

`VAPOR_MIND` chooses the model: `anthropic:MODEL`, `openai:MODEL[@URL]` (any compatible
server, including a local one) or `script:FILE` (recorded answers — the tests use this one, with no
network). Without a model, everything works except drafting from words, which says how to
configure one.
