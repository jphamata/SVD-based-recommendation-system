# Al-Mīzān — um dialeto para afirmações que se decidem

> الميزان, "a balança"; raiz و-ز-ن *w-z-n*, "pesar". Arquivos **`.wzn`** (não `.mzn`: essa é a
> extensão do MiniZinc). `Vapor.Mizan`, `Vapor.Mizan.Syntax`, `Vapor.Mizan.Lower`,
> `Vapor.Mizan.Abjad`. Testes: `mizan_test.exs`; §5l. A pesagem do manifesto que o propôs:
> [DIRETRIZ §19](DIRETRIZ.md).

## Em uma frase

Uma linguagem pequena em que cada afirmação diz **de que tipo é** (a raiz) e **como pode ser
usada** (o *wazn*), e em que uma afirmação marcada como provada **não compila** até um procedimento
de decisão do vapor prová-la — ou refutá-la com o ponto que a quebra.

## A árvore e as duas escritas

O programa é uma árvore. Latim e árabe são **impressões bijetivas** dela — a mesma árvore, o mesmo
hash:

```lisp
(claim energy (root H-f-Z) (wazn burhan)
  (inputs (x q) (v q))
  (field (x v) (v (- x)))
  (proof conserved)
  (body (+ (kinetic v) (* 1/2 x x))))
```

```lisp
(دعوى energy (جذر ح-ف-ظ) (وزن برهان)
  (مدخلات (x نسبي) (v نسبي))
  (حقل (x v) (v (- x)))
  (برهان محفوظ)
  (تنفيذ (+ (kinetic v) (* ١/٢ x x))))
```

- Palavras-chave têm forma latina e árabe (`claim` دعوى, `root` جذر, `wazn` وزن, `inputs` مدخلات,
  `field` حقل, `box` صندوق, `step` خطوة, `init` بداية, `invariant` ثابت, `proof` برهان, `body`
  تنفيذ, `import` استيراد; tipos `q` نسبي, `int` صحيح, `f64` عائم٦٤, `f32` عائم٣٢, `bool` منطقي;
  `and` و, `or` أو, `not` ليس, `if` إذا; e os operadores por extenso جمع طرح ضرب قسمة أس).
- Números: algarismos ocidentais numa escrita, arábico-índicos (٠–٩) na outra; frações exatas.
- Nomes: latinos (letras, dígitos, `-`, `_`) **ou** árabes (letras, `-`, algarismos arábico-índicos)
  — **nunca misturados**. Raízes na forma latina usam Buckwalter (`H-f-Z` = ح-ف-ظ): sem perdas,
  ao contrário de uma romanização "bonita" que funde formas do hamza (7 nomes → 2, medido).
- `ler(imprimir(t)) = t` nas duas escritas: testado em 300 programas aleatórios.
- O hash (`vapor wzn hash`) é SHA-256 da árvore canônica, igual nas duas escritas.

## Raízes e *awzān*: um sistema de tipos morfológico

| raiz | domínio | o que se pode afirmar |
|---|---|---|
| ح-س-ب H-s-b | cálculo | um valor; identidades; limites e positividade numa caixa |
| ح-ف-ظ H-f-Z | conservação | uma grandeza constante ao longo de um campo `ẋ = f(x)` |
| ن-ق-ل n-q-l | transição | um sistema de passos booleanos e o seu invariante |
| ك-ت-ب k-t-b | registro | um valor guardado, nomeado pelo seu hash |

| *wazn* | regime | consequência |
|---|---|---|
| فاعل *fāʿil* | transitório | uma função pura, baixada para o compilador do vapor; não enuncia teorema |
| مفعول *mafʿūl* | persistente | o valor é guardado e nomeado pelo hash |
| برهان *burhān* | provado | nada roda até a obrigação ser cumprida |

Pares sem sentido são recusados na verificação: uma conservação (ح-ف-ظ) sem `burhān` não é uma lei;
um registro (ك-ت-ب) com `proof` não tem o que provar.

## Provas por decisão, não por pessoa num assistente

| `proof` | decisor | certificado |
|---|---|---|
| `(identity E)` | forma normal polinomial exata sobre ℚ | os dois lados normalizam ao mesmo polinômio |
| `conserved` | dH/dt = ∇H·f, normalizado sobre ℚ | o polinômio nulo — ou o resto e um ponto onde ≠ 0 |
| `nonneg`, `pos`, `(bounded a b)` | Aludel (Bernstein em inteiros exatos) | uma subdivisão replayável — ou o ponto exato que refuta |
| `invariant` | SAT (indução: `init ⇒ I`, `I ∧ step ⇒ I'`) | prova DRUP conferida por código separado — ou o contraexemplo |

O veredito é *provado*, *refutado* (com o ponto) ou *desconhecido*; **desconhecido não compila**.
A linguagem é recortada para caber nos decisores. O Lean 4 é uma segunda opinião:
`transmute --to lean` gera os teoremas (`ring`/`decide`); fechá-los está no livro-razão como
**devido**, porque o Lean não está instalado nesta máquina.

## Os verbos (o `ikseer` do manifesto)

```sh
vapor wzn check FILE           # destila e decide: ✓ provado · ✗ refutado (com o ponto) · ? desconhecido
vapor wzn show FILE --arabic   # a outra escrita (a mesma árvore)
vapor wzn hash FILE            # a identidade
vapor wzn run FILE CLAIM ARGS  # avalia (exato em ℚ); um burhān só depois de provado
vapor wzn transmute FILE CLAIM --to vapor|aiger|lean [--out DIR]
                               # o compilador do vapor (x86-64, AVX-512, AArch64, RISC-V, SPIR-V) · um circuito · Lean 4
vapor wzn assay FILE CLAIM     # o fāʿil em binary32 no oráculo do vapor contra ℚ, em ULPs
vapor wzn abjad كتب            # o valor abjad, e por que não é um endereço
```

Saída: 0 tudo provado · 1 algo refutado ou desconhecido · 3 entrada ruim. Exemplos:
`priv/mizan/oscillator.wzn` (e a sua versão árabe), `bounds.wzn` (Motzkin + 1/1000, um limite
cúbico, um quadrado), `handshake.wzn` (um invariante por SAT).

## *Abjad*, medido

O manifesto propunha endereçar por *abjad* (o valor numérico das letras). Das 21 952 raízes de três
letras, **21 950 compartilham o valor com outra** (99,99 %); a maior classe tem 82 raízes; todo
anagrama colide. O valor é mostrado, nunca usado como endereço. A identidade é o hash: zero
colisões nas mesmas raízes.

## Editores

O servidor de linguagem (`vapor lsp`) diagnostica (cada obrigação decidida ao salvar), explica
(*hover*: o veredito e o decisor de uma afirmação; o sentido e o valor abjad de uma raiz; uma
palavra-chave nas duas escritas), completa, vai à definição, formata e troca de escrita.
[EDITORES.md](EDITORES.md).

## O que não é

Não é uma linguagem de propósito geral, nem substitui a Alembic (os *kernels*) — o Mīzān enuncia
afirmações e as decide; a execução é do compilador do vapor. Não prova o que os seus decisores não
decidem: recusa.
