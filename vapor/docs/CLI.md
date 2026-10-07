# A linha de comando — a bancada inteira pelo terminal

> Desde 0.14.0. Código: `lib/vapor/main.ex`, `lib/vapor/main/*`, `bin/vapor`, `mix vapor`.

Tudo o que o console faz, o terminal faz, na filosofia Unix: cada comando lê um arquivo ou a
entrada padrão (`-`), escreve texto para pessoas num terminal e **JSON quando a saída é um
*pipe*** (ou com `--json`), e diz o resultado pelo código de saída.

```
vapor alembic FILE | -e EXPR | --card
vapor athanor run FILE [--budget N --seed N --seconds N --only s1,s2 --set k=v --mind SPEC
                        --measure CMD --interactive --no-control --json]
vapor athanor ask FILE                  # medido: vapor propõe, você mede
vapor verify FILE CERT [--full --replay]
vapor game FILE solve|search|learn|play
vapor crucible KIND FILE                # `vapor crucible` lista; --example mostra um
vapor assay TOOL FILE                   # `vapor assay` lista; --example mostra um
vapor mind ask|formalize|transcript     # --model ou VAPOR_MIND
vapor scene new|edit|direct|export|card
vapor solve FILE                        # a bancada de equações
vapor rebis equiv|anf|identity|stabilizer|aiger FILE…   # circuitos sobre GF(2) (0.15)
vapor aludel decide P --vars x,y --box '0,1;0,1' | REQUEST.json   # positividade, barreiras
vapor tabula FILE [--facts a,b]          # um contrato: antinomias, provas, lacunas
vapor cupel [--n 32 --k 64 --bit 26]     # o exercício de corrupção silenciosa
vapor amalgam FILE|- [--f32]             # uma soma que não depende da ordem
vapor chat new|say|show|edit|regen|switch|rewind|fork|pin|context|compact|search|export|import|share …  (0.16)
vapor wzn check|show|hash|run|transmute|assay|abjad FILE …   # o Mīzān (0.16): afirmações decididas
vapor lsp                                # o servidor de linguagem (VS Code, Neovim, Emacs, …)
vapor serve | tui | ocr | merge | quality …   # as tarefas mix anteriores
```

| código | significa |
|---|---|
| 0 | positivo: achado, provado, verificado, sinal |
| 1 | negativo: refutado, não achado, verificação falhou, ruído |
| 2 | uso errado |
| 3 | entrada inválida (com linha e coluna) |
| 4 | falha |

`NO_COLOR` desliga a cor; `VAPOR_TTY=0|1` força o modo. Exemplos de composição:

```sh
# a conjectura de Euler refutada, o certificado conferido por outro processo
vapor athanor run euler.alb > c.json || vapor verify euler.alb c.json && echo "contraexemplo real"

# o modelo rascunha, a pessoa lê a retrotradução, a fornalha busca
echo "a régua de Golomb mais curta com 8 marcas" | vapor mind formalize - > g.alb && vapor athanor run g.alb

# um objetivo medido por um programa externo (um treino, uma simulação)
vapor athanor run hp.alb --measure './treina.sh'   # lê $VAPOR_CANDIDATE_JSON, imprime o número (o último da saída)

# uma cena editada por operações, como um documento
vapor scene new > s.json && vapor scene edit s.json "add glow sol { x: 0.7, y: 0.2 }" > s2.json
```

## O Opus pelo terminal (0.15)

Os mesmos códigos de saída: `0` é equivalente, provado, certificado ou consistente; `1` é
diferente, refutado, esgotado ou com antinomias — então um *script* pode exigir a prova.

```sh
# a netlist depois da síntese contra a especificação; o contraexemplo agrupado em palavras
vapor rebis equiv spec.net synth.aag || echo "não é a mesma função"

# um multiplicador provado por álgebra sobre ℤ (onde o SAT é exponencial)
vapor rebis identity mul16.net --spec 'm[32] = a[16] * b[16]'

# uma afirmação estrita sobre um polinômio, com a testemunha reproduzível no JSON
vapor aludel decide 'x^4*y^2 + x^2*y^4 - 3*x^2*y^2 + 1 + 1/1000' --vars x,y --box '-2,2;-2,2' --strict --json > w.json

# um contrato: antinomias com o cenário; as posições quando a entrega atrasou
vapor tabula venda.txt --facts delivered,late
```

## Conversas, o Mīzān e o terminal único (0.16)

```sh
# uma conversa num arquivo seu (~/.vapor/majlis, ou $VAPOR_HOME), com o modelo de VAPOR_MIND
T=$(vapor chat new --title rascunho --system "responda em português")
vapor chat say $T "o que é uma base de Gröbner?"
vapor chat edit $T 3fa2c1 "e um exemplo pequeno?"    # um ramo; o antigo fica
vapor chat context $T                                # o que o modelo vai ler, e o que fica de fora
vapor chat export $T --md > rascunho.md

# uma lei de conservação provada sobre ℚ; a versão amortecida refutada no ponto
vapor wzn check priv/mizan/oscillator.wzn
vapor wzn show priv/mizan/oscillator.wzn --arabic | vapor wzn hash -   # o mesmo hash
```

A linha de comando, o TUI (`mix vapor.tui`), o terminal do console e `POST /v1/vapor/diwan` são o
mesmo interpretador — o Dīwān ([DIWAN.md](DIWAN.md)): pipes, redireção, `;`, aspas e arquivos;
no console, uma sessão enjaulada. Conversas: [MAJLIS.md](MAJLIS.md); o Mīzān: [MIZAN.md](MIZAN.md);
editores: [EDITORES.md](EDITORES.md).

## Agentes e console de terminal

Os mesmos verbos são ferramentas MCP (`mix vapor.mcp`: `alembic_eval`, `athanor_run` com
`proposals`, `athanor_verify`, `game_query`, `crucible_run`, `assay_run`, `scene_ops`; na 0.15
`rebis_check`, `aludel_decide`, `tabula_analyze`, `cupel_drill`, `amalgam_sum` — 25 ao todo) e comandos da TUI (`mix vapor.tui`: `alembic -e "…"`, `athanor run arquivo`, …). Um
agente propõe; a Touchstone confere — a mesma porta para pessoas, modelos e programas.

## Mente

`VAPOR_MIND` escolhe o modelo: `anthropic:MODELO`, `openai:MODELO[@URL]` (qualquer servidor
compatível, inclusive local) ou `script:ARQUIVO` (respostas gravadas — os testes usam este, sem
rede). Sem modelo, tudo funciona, menos o rascunho a partir de palavras, que diz como
configurar um.
