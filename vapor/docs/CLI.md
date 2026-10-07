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

## Agentes e console de terminal

Os mesmos verbos são ferramentas MCP (`mix vapor.mcp`: `alembic_eval`, `athanor_run` com
`proposals`, `athanor_verify`, `game_query`, `crucible_run`, `assay_run`, `scene_ops` — 20 ao
todo) e comandos da TUI (`mix vapor.tui`: `alembic -e "…"`, `athanor run arquivo`, …). Um
agente propõe; a Touchstone confere — a mesma porta para pessoas, modelos e programas.

## Mente

`VAPOR_MIND` escolhe o modelo: `anthropic:MODELO`, `openai:MODELO[@URL]` (qualquer servidor
compatível, inclusive local) ou `script:ARQUIVO` (respostas gravadas — os testes usam este, sem
rede). Sem modelo, tudo funciona, menos o rascunho a partir de palavras, que diz como
configurar um.
