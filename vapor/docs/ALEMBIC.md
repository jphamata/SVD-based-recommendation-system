# Alembic — the language of problems

> Since 0.14.0. Code: `lib/vapor/alembic/` (`lexer.ex`, `parser.ex`, `compiler.ex`,
> `builtins.ex`) and `lib/vapor/alembic.ex`. Tests: `test/vapor/alembic_test.exs`.
> Reference card: `vapor alembic --card`.

Alembic is the language in which **any** problem is written for the workbench: a space and an
objective (Athanor searches), a claim (Athanor looks for the counterexample or proves by
enumeration), a game (Athanor solves, searches or learns). It replaces the predefined categories of the earlier rounds: instead of
choosing among ready-made problems, the person — or a language model — writes their own.

The name is the apparatus's: the alembic receives what is put into it and gives back what is distilled.

## 1. The language on one page

```
# a program is a list of definitions, one per line, in any order
n = 7
ruler(r) = [0] ++ r
dist(r) = [b - a for (a, b) in pairs(ruler(r))]
violation(r) = len(dist(r)) - len(distinct(dist(r)))
primes = [p for p in 2..50 if is_prime(p)]
total = fold(primes, 0, (acc, p) => acc + p)
```

- **Values**: arbitrary-size integers, floats, strings, `true`/`false`/`nil`, lists,
  tuples, maps. Everything immutable.
- **Expressions**: `if … then … else …`, `let a = …, b = … in …`, lambdas `x => …` and
  `(a, b) => …`, comprehensions with several `for` and `if`, *pipes* `xs |> map(f) |> sum`,
  slices `xs[a:b]`, negative indices, fields `m.key`, chained comparisons `0 <= i < n`.
- **Line breaks** continue the expression after an operator or before `|>`; otherwise
  they end the definition.
- **Errors** state line, column and, for names, suggest the nearest one (`lenn` → *did you mean len*).
- ~120 built-in functions (`vapor alembic --card` lists them all); `noise(...)` and `hash(...)` are
  deterministic — the same program gives the same number on any machine.

## 2. Sanitised: open input does not hurt the host

The input is free, so the limits belong to the machine and not to the person:

| risk | defence | test |
|---|---|---|
| infinite loop | **fuel** charged on every call and iteration (`fuel:`) | `spin(x) = spin(x + 1)` → *recursion*; large `fold` → *fuel* |
| deep recursion | maximum depth 5,000 | same |
| huge numbers | integers up to 65,536 bits | `2 ^ 100000000` → *bits* |
| huge lists | 2 million elements; `range` bounded | `range(10^9)` → *range* |
| memory | `Vapor.Hermetic.seal/2`: its own process with `max_heap_size`; the VM kills it | memory bomb → `{:error, :memory}`, the caller lives |
| time | `seal(…, timeout:)` | `{:error, :timeout}` |
| atom table | identifiers stay binaries, never atoms | 2,000 new names → < 50 atoms |
| code in place of data | `Alembic.literal/1` reads only data (and constant arithmetic) | `f(1)` and comprehensions rejected |

There is no I/O, no clock, no randomness outside `noise`/`hash`: a program is a pure
function of its text. That is what makes Athanor's journal reproducible.

The same discipline holds for the numeric expressions compiled by the equation workbench
(`Vapor.Expr.compile/2`): each new expression becomes a BEAM module, so, beyond a ceiling
(`config :vapor, expr_jit_cap: 4096`), new ones are **interpreted** with the same values and
no atom is created (test in `alembic_test.exs`).

## 3. API

```elixir
{:ok, prog} = Vapor.Alembic.load(text, consts: %{"n" => 8}, skip: ["space"], fuel: 10_000_000)
Vapor.Alembic.call(prog, "f", [10])               # {:ok, value} | {:error, message}
Vapor.Alembic.eval("sum([x^2 for x in 1..10])")    # {:ok, 385}
Vapor.Alembic.literal(~S|[1, (2, 3), {"k": 4}]|)   # data, never code
Vapor.Alembic.show(value)                           # text that literal/1 reads back
Vapor.Hermetic.seal(fn -> … end, heap_mb: 64, timeout: 5_000)   # the one seal on untrusted work (docs/HERMETIC.md)
```

`show` and `literal` are inverses (tested with 300 random nested values).

## 4. Terminal

```
vapor alembic programa.nbq            # evaluates the constants and shows them
vapor alembic -e "factorial(30)"      # one expression
echo "f(n) = n*n" | vapor alembic - --json
vapor alembic --card                  # the reference card (also in the console and in MCP)
```
