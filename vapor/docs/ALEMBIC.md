# Alembic — a linguagem dos problemas

> Desde 0.14.0. Código: `lib/vapor/alembic/` (`lexer.ex`, `parser.ex`, `compiler.ex`,
> `builtins.ex`, `tree.ex`) e `lib/vapor/alembic.ex`. Testes: `test/vapor/alembic_test.exs`.
> Cartão de referência: `vapor alembic --card`.

Alembic é a linguagem em que se escreve **qualquer** problema para a bancada: um espaço e um
objetivo (o Athanor busca), uma afirmação (o Athanor procura o contraexemplo ou prova pela
enumeração), um jogo (o Athanor resolve, busca ou aprende), uma cena (o subconjunto numérico
roda no navegador). Ela substitui as categorias pré-definidas das rodadas anteriores: em vez de
escolher entre problemas prontos, a pessoa — ou um modelo de linguagem — escreve o seu.

O nome é o do aparelho: a alambique recebe o que se põe nela e devolve o que se destila.

## 1. A linguagem em uma página

```
# um programa é uma lista de definições, uma por linha, em qualquer ordem
n = 7
ruler(r) = [0] ++ r
dist(r) = [b - a for (a, b) in pairs(ruler(r))]
violation(r) = len(dist(r)) - len(distinct(dist(r)))
primes = [p for p in 2..50 if is_prime(p)]
total = fold(primes, 0, (acc, p) => acc + p)
```

- **Valores**: inteiros de tamanho arbitrário, floats, strings, `true`/`false`/`nil`, listas,
  tuplas, mapas. Tudo imutável.
- **Expressões**: `if … then … else …`, `let a = …, b = … in …`, lambdas `x => …` e
  `(a, b) => …`, compreensões com vários `for` e `if`, *pipes* `xs |> map(f) |> sum`,
  fatias `xs[a:b]`, índices negativos, campos `m.chave`, comparações encadeadas `0 <= i < n`.
- **Quebras de linha** continuam a expressão depois de um operador ou antes de `|>`; fora disso
  terminam a definição.
- **Erros** dizem linha, coluna e, para nomes, sugerem o mais próximo (`lenn` → *did you mean len*).
- ~120 funções embutidas (`vapor alembic --card` lista todas); `noise(...)` e `hash(...)` são
  determinísticos — o mesmo programa dá o mesmo número em qualquer máquina.

## 2. Saneada: entrada aberta não machuca o hospedeiro

A entrada é livre, então os limites são da máquina e não da pessoa:

| risco | defesa | teste |
|---|---|---|
| laço infinito | **combustível** cobrado em toda chamada e iteração (`fuel:`) | `spin(x) = spin(x + 1)` → *recursion*; `fold` grande → *fuel* |
| recursão profunda | profundidade máxima 5 000 | idem |
| números enormes | inteiros até 65 536 bits | `2 ^ 100000000` → *bits* |
| listas enormes | 2 milhões de elementos; `range` limitado | `range(10^9)` → *range* |
| memória | `Alembic.sandbox/2`: processo próprio com `max_heap_size`; a VM o mata | bomba de memória → `{:error, :memory}`, o chamador vive |
| tempo | `sandbox(…, timeout:)` | `{:error, :timeout}` |
| tabela de átomos | identificadores ficam binários, nunca átomos | 2 000 nomes novos → < 50 átomos |
| código no lugar de dados | `Alembic.literal/1` lê só dados (e aritmética constante) | `f(1)` e compreensões recusadas |

Não há E/S, nem relógio, nem aleatoriedade fora de `noise`/`hash`: um programa é uma função
pura do seu texto. Isso é o que torna o diário do Athanor reprodutível.

A mesma disciplina vale para as expressões numéricas compiladas pela bancada de equações
(`Vapor.Expr.compile/2`): cada expressão nova vira um módulo BEAM, então, além de um teto
(`config :vapor, expr_jit_cap: 4096`), as novas são **interpretadas** com os mesmos valores e
nenhum átomo é criado (teste em `alembic_test.exs`).

## 3. A árvore portátil (cenas)

`Vapor.Alembic.Tree` é o subconjunto numérico (aritmética, comparações, `if`, funções
matemáticas, `noise`) convertido em uma árvore JSON que o navegador **interpreta** — nunca
compila nem avalia como JavaScript. É assim que uma cena aceita `x: 0.5 + 0.2*sin(t)` digitado
pela pessoa sem abrir uma porta de injeção. `noise()` é bit a bit idêntico em Elixir e em JS
(valores de ouro em `test/js/scene_noise.mjs` e `alembic_test.exs`).

## 4. API

```elixir
{:ok, prog} = Vapor.Alembic.load(text, consts: %{"n" => 8}, skip: ["space"], fuel: 10_000_000)
Vapor.Alembic.call(prog, "f", [10])               # {:ok, valor} | {:error, mensagem}
Vapor.Alembic.eval("sum([x^2 for x in 1..10])")    # {:ok, 385}
Vapor.Alembic.literal(~S|[1, (2, 3), {"k": 4}]|)   # dados, nunca código
Vapor.Alembic.show(valor)                           # texto que literal/1 lê de volta
Vapor.Alembic.sandbox(fn -> … end, heap_mb: 64, timeout: 5_000)
```

`show` e `literal` são inversos (teste com 300 valores aleatórios aninhados).

## 5. Terminal

```
vapor alembic programa.alb            # avalia as constantes e mostra
vapor alembic -e "factorial(30)"      # uma expressão
echo "f(n) = n*n" | vapor alembic - --json
vapor alembic --card                  # o cartão de referência (também no console e no MCP)
```
