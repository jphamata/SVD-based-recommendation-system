// The editor packages, checked where they can be without the editors:
// every JSON file parses, every TextMate pattern compiles as a JS regular
// expression (lookbehind, Unicode), and the grammars tokenize both scripts.
import { readFileSync } from "node:fs";
const root = process.argv[2];
const read = (p) => JSON.parse(readFileSync(`${root}/${p}`, "utf8"));
const out = { files: 0, patterns: 0, hits: {} };
for (const p of ["editors/vscode/package.json", "editors/vscode/language-configuration.json", "editors/vscode/alembic-configuration.json"]) { read(p); out.files++; }
const grammar = (p) => { const g = read(p); out.files++; return g.patterns.filter((x) => x.match).map((x) => { out.patterns++; return { name: x.name, re: new RegExp(x.match, "gmu") }; }); };
const tag = (g, text) => { const found = new Set(); for (const { name, re } of g) { re.lastIndex = 0; if (re.test(text)) found.add(name); } return [...found].sort(); };
const mizan = grammar("editors/vscode/syntaxes/mizan.tmLanguage.json");
const alembic = grammar("editors/vscode/syntaxes/alembic.tmLanguage.json");
out.hits.latin = tag(mizan, readFileSync(`${root}/priv/mizan/oscillator.wzn`, "utf8"));
out.hits.arabic = tag(mizan, readFileSync(`${root}/priv/mizan/oscillator-ar.wzn`, "utf8"));
out.hits.alembic = tag(alembic, "# golomb\nspace = ints(7, 0, 30)\nf(x) = if x > 2 then x * 3 else 1\nminimize(s) = max(s)\n");
console.log(JSON.stringify(out));
