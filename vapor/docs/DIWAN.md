# O Dīwān — um interpretador para todos os terminais

> ديوان, raiz د-و-ن *d-w-n*, "registrar": o registro, e a sala onde se despacha. `Vapor.Diwan`.
> Testes: `diwan_test.exs`, `hall_test.exs`, `tui_test.exs`, `console_majlis_test.exs`; §5l.

## Uma função, quatro portas

`eval(linha, sessão) → {resultado, sessão}`. A mesma linha dá a mesma resposta em todas as portas,
porque elas são a mesma função:

| porta | como | sessão |
|---|---|---|
| linha de comando | `bin/vapor VERBO …` (cada verbo é `Vapor.Main.run/1`) | o shell do usuário |
| TUI | `mix vapor.tui` (sem curses: funciona por SSH e num log de CI) | **local**: lê e escreve os arquivos de quem está ao teclado |
| terminal do console | *Conversar → Terminal* | **enjaulada** |
| API | `POST /v1/vapor/diwan` `{session, line}` → `{out, err, code, codes, files}` | enjaulada |

Um verbo novo aparece nas quatro sem uma linha a mais.

## A linguagem da linha

Um shell pequeno — **nenhum shell é executado**: os verbos do vapor (a palavra `vapor` é
opcional), `|`, `>`, `>>`, `<`, `;`, aspas e `\`, e os embutidos `help ls cat echo rm cp mv head
wc history clear`.

```sh
athanor run golomb.alb | verify golomb.alb -
rebis equiv a.net b.net > veredito.json
echo "(claim q (root H-s-b) (wazn fail) (inputs (x q)) (body (* x x)))" > q.wzn ; wzn check q.wzn
chat search kulisch | head 5
```

Todo estágio menos o último escreve JSON (o verbo vê um *pipe*); o último escreve para pessoas —
exatamente o comportamento Unix de `bin/vapor`. Um `|` inicial ou final, ou `a | | b`, é recusado
(um comando vazio), não ignorado.

## A jaula

A sessão do console é uma porta para quem chega pela rede, então:

- os arquivos são **da sessão** (`ls`, `cat`, `rm`, `>` e o editor e o envio do painel): 128
  arquivos, 8 MB cada, 64 MB no total; um argumento ARQUIVO nomeia um deles — `/etc/passwd` não
  existe ali;
- `--measure` (que roda um programa) é recusado; nenhum verbo abre um processo externo;
- cada comando roda no **seu próprio processo** com teto de heap e prazo: um comando que dispara é
  morto, a sessão fica;
- uma sessão faz um comando por vez (um segundo, no meio, recebe 409 em vez de disputar os
  arquivos);
- no máximo 256 sessões vivas; a mais antiga sai.

O controle de §5l: a mesma leitura numa sessão local (o TUI) lê o arquivo; na jaula, saída 3 e "no
such file in this session". O livro-razão diz no que isso repousa: no isolamento de processos da
BEAM e em todo verbo ler pela porta `Vapor.Main.read_input`.

## O painel

Tela com as cores ANSI dos comandos (a paleta da página), histórico (↑ ↓), Tab completa verbos e
arquivos da sessão (`POST /v1/vapor/diwan/complete`), Ctrl+L limpa; à direita, os arquivos da
sessão com um editor (Ctrl+S salva), envio e download. Exemplos prontos escrevem os arquivos de que
precisam e rodam.
