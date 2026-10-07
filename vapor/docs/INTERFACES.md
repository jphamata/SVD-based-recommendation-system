# Interfaces: GUI, TUI, CLI — e por que não Tauri

O vapor tem quatro faces sobre um núcleo só. Em todas, cada resposta chega com
a sua medida (confiança, certeza, recibo, distância ao treino): a interface é
onde a pergunta "isto é sinal ou ruído?" é respondida para uma pessoa.

| face | para | como |
|---|---|---|
| **Console web** (GUI) | uso diário, demonstrações, revisar evidência | `mix vapor.serve [--model DIR] [--docs CAMINHO]` → `http://127.0.0.1:8000/` |
| **App instalado** | uma janela própria, sem aba de navegador | o console é um app web instalável (`manifest.webmanifest`, ícones): no Chromium/Edge, *Instalar vapor* o abre em janela própria |
| **Console no terminal** (TUI) | SSH, servidores sem navegador, logs de CI | `mix vapor.tui [--lang pt] [--docs CAMINHO]` |
| **Tarefas de linha de comando** | scripts e pipelines | `mix vapor.ocr`, `vapor.merge`, `vapor.quality`, `vapor.rag`, `vapor.lock`, … |
| **API HTTP** | outros programas | `/v1/*` compatível com OpenAI e `/v1/vapor/*` do console ([CONSOLE.md](CONSOLE.md)) |

## Desde 0.14: o terminal primeiro

Tudo o que o console faz, `bin/vapor` faz — com JSON em *pipes*, códigos de saída com
significado e leitura da entrada padrão ([CLI.md](CLI.md)). O console, a TUI e as
ferramentas MCP chamam as mesmas funções; nenhuma capacidade existe só na interface gráfica.

## O console web

Um arquivo HTML autocontido (`priv/console/index.html`): sem CDN, sem baixar
fontes, funciona sem internet, servido pela mesma BEAM que roda os modelos.

- **Língua**: inglês por padrão, português a um clique (lembrado no
  navegador). Todo texto da página vive num dicionário — uma terceira língua
  é uma tabela, não uma reescrita. Os diagnósticos que dependem de números
  (o regime de uma fusão) são montados no cliente a partir dos números, nas
  duas línguas; mensagens que vêm do servidor (recusas, erros) ficam em
  inglês, como a API.
- **Tema**: claro e escuro, seguindo o sistema até a pessoa escolher.
- **Identidade**: uma eclusa. O logo (`priv/console/logo.svg`,
  `docs/img/logo.svg`) é a bacia entre as comportas, com a água num nível e o
  vapor subindo; o elemento ousado da página é a mesma ideia — cada resultado
  num tanque cujo nível é a sua medida, com a linha calibrada marcada. O resto
  é quieto: uma sans para o texto, uma condensada de sinalização para os
  títulos, monoespaçada só para *digests*.
- **Painéis** agrupados pelo que se faz: *Perguntar* (conversa com evidência),
  *Ler* (documentos, visão/OCR, fala), *Criar* (desenho por difusão), *Medir*
  (fusão, portão de qualidade, a eclusa de modelos).
- **Acessibilidade**: navegação por teclado (setas entre seções, link de pular,
  foco visível), movimento reduzido respeitado, legível a 390 px. Conferido
  no Chromium headless em claro, escuro, nas duas línguas e na largura de
  celular, sem erro de console.

## O console no terminal

`Vapor.TUI` — uma sessão por linhas (sem curses, sem dependência): `read`
(OCR), `listen`, `draw` (o dígito desenhado no terminal na escala de 24 cinzas
e lido de volta pelo classificador), `add`/`search`, `quality`, `merge`,
`lang en|pt`. Cor só em terminal e nunca com `NO_COLOR`; numa tubulação,
texto puro. O intérprete (`eval/2`) é puro e testado sem terminal.

## Tauri: considerado, não adotado

Um invólucro Tauri daria uma janela nativa em volta da mesma página. Pesado
contra o que o vapor é:

| | Tauri | app web instalável (adotado) |
|---|---|---|
| toolchains novas | Rust, Cargo, um WebView por SO, Node para o empacotador | nenhuma |
| o runtime dos modelos | a BEAM e o worker nativo como *sidecar* por SO, com assinatura | já está rodando: é ele que serve a página |
| política de dependências (`deps: []`) | quebra (crates, npm) | mantida |
| sem internet | sim | sim (um arquivo, sem CDN) |
| janela, ícone, entrada na barra de tarefas | sim | sim ("Instalar" no Chromium/Edge) |
| sistema de arquivos, bandeja, atualização automática | sim | não — e não faz falta: arquivos entram pelo seletor e por arrastar e soltar, o servidor lê `--docs` |
| o que acrescenta à *evidência* | nada | — |

O worker hoje é só Linux ([TODO](TODO.md)), então um pacote de desktop
multiplataforma embrulharia um runtime que não roda em dois dos três alvos.
Quando houver workers para macOS/Windows, vale revisitar — e a página não
muda: um Tauri apontaria para o mesmo `http://127.0.0.1:PORTA/`.
