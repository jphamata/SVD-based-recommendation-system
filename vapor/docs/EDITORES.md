# Editores — um servidor de linguagem, três clientes finos

> `vapor lsp` (`Vapor.LSP`), `editors/`. Testes: `lsp_test.exs` (o protocolo por stdio real, de um
> cliente Node), `editors_test.exs` (manifesto e gramáticas do VS Code validados; os arquivos de
> sintaxe do Vim carregam sem erro em `vim -Es`). Escrutínio: [DIRETRIZ §19](DIRETRIZ.md).

## O princípio

Um ecossistema "à la VS Code, Neovim e Emacs" não é três *plugins* que reimplementam a linguagem —
isso daria três verdades. É **um** servidor (LSP 3.17, JSON-RPC por stdio, *framing*
`Content-Length`, colunas em UTF-16) e clientes que só o ligam. Todo editor que fala LSP — Helix,
Zed, Kakoune, Sublime — ganha o mesmo, sem nada daqui.

| | Al-Mīzān (`.wzn`) | Alembic (`.alb`) |
|---|---|---|
| diagnósticos | sintaxe e regras morfológicas ao digitar; ao abrir e salvar, **cada obrigação decidida** — uma afirmação refutada é um erro na sua linha, com o ponto que a refuta | erros de análise com linha e coluna |
| *hover* | o veredito e o decisor de uma afirmação; o sentido e o valor abjad de uma raiz; uma palavra-chave nas duas escritas | a assinatura de um embutido |
| completar | palavras-chave na escrita do arquivo, raízes, afirmações definidas acima | embutidos e constantes |
| símbolos, ir à definição | afirmações | definições |
| formatar | a impressão canônica, na escrita do arquivo | — |
| comandos | `vapor.mizan.toArabic` / `toLatin`: o mesmo programa na outra escrita (a mesma árvore, o mesmo hash), como uma edição que a pessoa escolhe | — |

## Instalar

| editor | o quê | testado aqui |
|---|---|---|
| VS Code | `editors/vscode` (gramáticas TextMate das duas escritas e da Alembic, configuração de linguagem, o cliente; `vapor.path` aponta o executável) | manifesto e gramáticas validados; a extensão em si pede o VS Code |
| Neovim 0.10+ | `editors/nvim` no *runtimepath*; `require("vapor").setup()`; comandos `:MizanArabic`, `:MizanLatin` | não instalado nesta máquina |
| Vim | `editors/nvim/{ftdetect,syntax}` (realce; LSP por qualquer *plugin* cliente) | carrega sem erro em `vim -Es` |
| Emacs 29+ | `editors/emacs/vapor-mode.el` (`mizan-mode`, `alembic-mode`, registro no `eglot`; `mizan-to-arabic`, `mizan-to-latin`) | não instalado nesta máquina |
| Helix, Zed, Kakoune, Sublime | qualquer cliente LSP: comando `vapor lsp`, tipos `.wzn`, `.alb` | o protocolo é testado de ponta a ponta |

O realce trata as duas escritas igualmente: palavras-chave latinas e árabes, raízes em Buckwalter e
em árabe, algarismos ocidentais e arábico-índicos. O texto árabe é exibido da direita para a
esquerda pelo próprio editor (Unicode bidi); a estrutura em S-expressão dispensa marcas de direção.

## O que não se afirma

Que a extensão do VS Code, o *plugin* do Neovim e o modo do Emacs tenham sido abertos nos
respectivos editores nesta máquina — não estão instalados. O que está testado é o servidor (por
stdio real) e os arquivos que cada editor lê (validados por formato).
