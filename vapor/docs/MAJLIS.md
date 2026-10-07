# O Majlis — conversas como árvore endereçada por conteúdo

> مجلس, raiz ج-ل-س *j-l-s*, "sentar": o conselho onde se conversa. `Vapor.Majlis`. Portas: o
> console (*Conversar → Conversas*), `vapor chat` (terminal), a API `/v1/vapor/threads…`
> ([CONSOLE.md](CONSOLE.md)). Testes: `majlis_test.exs`, `hall_test.exs`,
> `console_majlis_test.exs` (Chromium). Escrutínio: [DIRETRIZ §19](DIRETRIZ.md).

## A ideia

Os produtos de conversa tratam a conversa como uma lista mutável com remendos (ramos de edição,
"ramificar em nova conversa", compactação invisível). Aqui ela é o que de fato é: uma **árvore**.

- Uma **mensagem** é um nó imutável `M1 ‖ CBOR{role, content, parent, t, meta, o}`, endereçado
  pelo SHA-256 dos seus bytes. Como o pai está dentro do hash, **uma mensagem compromete-se com
  toda a sua história** — como um *commit* do git.
- Uma **conversa** é um ponteiro (`head`) mais ajustes: instruções, modelo, ferramentas,
  orçamento de contexto, mensagens fixadas, resumo. Cada conversa tem um nó-âncora próprio e cada
  mensagem o seu dono (`o`), para que a coleta de lixo de uma nunca apague as de outra.
- **Editar** escreve um irmão; **outra resposta** escreve um irmão da resposta; **‹ ›** troca de
  irmão e segue o ramo mais recente abaixo dele; **continuar daqui** move o ponteiro para trás;
  **bifurcar** cria uma conversa nova apontando para uma mensagem existente — **O(1)**, nenhuma
  mensagem copiada (§5l: 0 mensagens, 964 bytes de raiz contra 120 mensagens de uma cópia).
- Tudo mora numa [Khazāna](KHAZANA.md): cada ação é um *commit* atômico.

## O contexto, calculado e mostrado

`context/2` devolve **exatamente** o que o modelo vai ler: as instruções (com o resumo, se houver),
depois as mensagens fixadas e o último turno — sempre —, depois as mais recentes que couberem no
orçamento. Cada mensagem do caminho sai marcada `sent`, `pinned`, `summarized` ou `dropped`, com
os seus tokens (exatos quando o servidor serve o tokenizador do modelo; estimados, e ditos
estimados, quando não). O controle de §5l: o truncamento pela cauda, com o mesmo orçamento,
derruba a instrução fixada.

**Compactar** pede ao modelo (ou aceita do usuário) um resumo do caminho até uma mensagem; o resumo
**nomeia o hash que cobre**, e por isso compromete-se com tudo antes dele. Desfazer devolve o
histórico inteiro ao contexto; o resumo nunca apaga nada.

## Agente

Uma conversa pode habilitar ferramentas do vapor por lista de permissão (`Vapor.Majlis.Tools`, as
mesmas do servidor MCP): puras (`alembic_eval`, `athanor_verify`, `rebis_check`, `aludel_decide`,
`tabula_analyze`, `amalgam_sum`, `cupel_drill`, `logic_check`, `workbench_solve`, …) e de
observação (`athanor_run`, `crucible_run`, `assay_run`, `finance_run`, …). Uma resposta com
ferramentas roda o laço de agente do vapor e guarda o **diário** da execução (Merkle,
verificável: `GET /v1/vapor/journal/:id`); a mensagem leva o diário no `meta`.

## Trocar, compartilhar

- **Exportar**: JSON `vapor-majlis/1` (cada mensagem com o seu hash; na importação, **todos são
  recalculados**, e um caractere trocado é recusado — §5l) ou Markdown (para ler; o controle: a
  mesma troca no Markdown é indetectável).
- **Importar**: o JSON do vapor, a exportação do ChatGPT (`conversations.json`, a árvore
  `mapping` preservada com os ramos) ou a do Claude (`chat_messages`).
- **Buscar**: BM25 sobre todas as mensagens, de todos os ramos.
- **Compartilhar**: um link só de leitura `/shared/ID?cap=…`, onde `cap` = HMAC(conversa,
  geração). Não pede o token do console — a capacidade é a autoridade. A página não tem *script* e
  escapa todo texto. **Revogar** incrementa a geração e mata todos os links já dados.

## Modelos

O Majlis não tem modelo próprio: usa os *backends* do servidor — o modelo servido localmente (as
respostas são re-deriváveis: semente, recibo) e o de `VAPOR_MIND` (`anthropic:…`, `openai:…@URL`,
`script:ARQUIVO`). Sem nenhum, as mensagens ficam guardadas e a resposta diz por quê.

## O que não faz (ainda)

*Streaming* token a token; anexos de imagem na conversa; memória entre conversas. Estão no
[TODO](TODO.md).

## Terminal

```sh
T=$(vapor chat new --title "rascunho" --system "responda em português")
vapor chat say $T "o que é uma base de Gröbner?"
vapor chat show $T --tree          # todos os ramos
vapor chat edit $T 3fa2c1 "e um exemplo?"   # 6+ dígitos hex bastam
vapor chat context $T              # o que vai, o que fica de fora
vapor chat fork $T 3fa2c1 --title "outra linha"
vapor chat export $T --md > rascunho.md
```
