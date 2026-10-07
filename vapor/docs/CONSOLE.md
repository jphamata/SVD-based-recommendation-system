# O console (`/` no `Vapor.Serve`)

```sh
mix vapor.serve --model ./Qwen2-0.5B --docs ./pasta       # API em /v1, console em /
mix vapor.serve --docs ./pasta                            # sem modelo de texto: documentos, visão, fala, desenho, estúdio, fusão, qualidade
mix vapor.serve --ip 0.0.0.0 --token SEGREDO --docs ./p   # exposto: o token é obrigatório
mix vapor.tui                                             # o mesmo, no terminal
```

Uma página servida pelo próprio servidor (`priv/console/index.html`): sem CDN,
sem fonte externa, funciona sem internet. **Inglês por padrão, português a um
clique** (lembrado no navegador); claro e escuro pelo sistema ou pelo botão;
navegável por teclado; legível a 390 px. Instalável como app (manifesto e
ícones), com logo e favicon próprios. Decisões de interface — e por que não
Tauri — em [INTERFACES.md](INTERFACES.md).

## 0.14 — a bancada aberta

O console abre no grupo **Bancada aberta**: *Espaço de trabalho* (escreva ou descreva qualquer
problema; o tipo é detectado — busca, afirmação, jogo, sistema, cena, Alembic), *Crisol* (o
seu sistema, com evidência) e *Ensaio* (avaliações de modelos: sinal ou ruído). A busca
aparece na **fornalha** — faíscas por avaliação, a linha do melhor, a linha tracejada do
controle aleatório, a partilha do portfólio — e o veredito na **pedra de toque**: um risco de
ouro por verificação que passou, de chumbo pela que falhou. A pessoa propõe candidatos, fixa e
bane finalistas, mede objetivos externos e pede ao modelo um rascunho ou propostas. Chamadas:
`GET /v1/vapor/workspace`, `POST /v1/vapor/{detect,alembic,athanor,athanor/verify,game,crucible,assay,formalize,scene/ops,scene/mind}`,
`GET|POST /v1/vapor/athanor/:id` (`?since=N` para as faíscas novas). Identidade: fuligem
`#14110D`, pergaminho, latão, verdete e cinábrio; títulos em serifa antiga das fontes do
sistema (nenhuma fonte baixada). Capturas: `docs/img/bancada-*.png`.

## A ideia

O que falta nas interfaces de modelos locais não é outro chat: é a
**evidência ao lado da saída**. A identidade visual é a de uma eclusa: cada
resultado fica num tanque cujo **nível da água é a sua medida** — a confiança
de uma linha lida pelo OCR, a certeza do leitor de fala, a probabilidade que o
classificador dá ao dígito desenhado, o escore relativo de uma passagem — com
a linha calibrada marcada. O resto é silencioso.

## Os painéis

**Perguntar — Conversa.** Com "responder com os documentos", a pergunta é
buscada na biblioteca; as passagens entram como fontes numeradas e a resposta
chega em *streaming*. A coluna *Evidência* mostra o caminho até a página
(`bundle.zip ▸ pasta/simple.pdf página 1`) e os recibos; citações
`<quote src="N">…</quote>` são conferidas no servidor.

**Ler — Documentos.** Zip (aninhado), PDF (também escaneado: o OCR lê),
Office, EPUB, HTML, PNG, JPEG: o que não vira texto é dito. A busca mostra
proveniência, escore, prova Merkle e o hash do arquivo, e recalcula o recibo.
Imagens com texto são achadas pelo que está escrito nelas.

**Ler — Visão.** Uma foto, um print ou um PDF escaneado (CCITT, JPEG,
Flate…): a página com as linhas marcadas, **os blocos numerados na ordem em
que são lidos** e o **fio de leitura** — uma linha fina do fim de cada linha
ao começo da seguinte, que mostra de relance se a máquina leu a coluna da
esquerda inteira antes da direita. Ao lado, o texto agrupado por bloco, cada
caractere tingido pela confiança (pontilhado abaixo de 80 %, ondulado abaixo
de 50 %) e **os caracteres que o modelo de língua escolheu** destacados, com
um botão que mostra a leitura só dos quadros (o que o modelo mudou aparece
riscado em cor de alerta). A evidência da leitura não é só a confiança: é
também *quem* decidiu cada letra.

![Visão](img/console-visao.png)

Uma **tabela** na página (0.8) aparece emoldurada na imagem e, abaixo,
desenhada a partir das suas células — cabeçalho, células mescladas, números
alinhados à direita, cada célula com a sua confiança (o mesmo pontilhado e
ondulado) e ligada à sua caixa na página: passar o mouse numa acende a
outra. Na lista de leitura, o bloco da tabela diz onde ela está. Um seletor
troca o desenho por Markdown, CSV ou HTML, e *Copiar* copia o formato
mostrado.

![Visão com tabela](img/console-tabelas.png)

**Ler — Ouvir.** Grave 1,5 s pelo microfone (o navegador codifica um WAV; o
servidor reamostra para 8 kHz) ou solte um WAV: o dígito, todas as
probabilidades e o espectro mel que o leitor ouviu; um botão desenha o dígito
ouvido — voz → texto → imagem.

**Criar — Desenhar.** Um dígito por difusão, a trajetória de remoção de ruído
passo a passo, o que o classificador de dados reais lê na imagem gerada e a
distância à imagem de treino mais próxima. A mesma semente dá a mesma imagem
em qualquer máquina.

![Desenhar, escuro, português](img/console-desenhar-escuro.png)

**Medir — Fusão.** O laboratório sobre os decoders treinados de
`priv/quality/merge`: o regime diagnosticado pelos pesos (em palavras montadas
dos números, nas duas línguas), cada método medido em validação e teste, o
escolhido.

![Fusão](img/console-fusao.png)

**Medir — Qualidade.** Cole um texto ou meça a última resposta: onde ele cai
entre as comportas calibradas de ruído e de texto real.

**Medir — Eclusa.** O contrato do modelo servido e os adaptadores registrados.

**Confiar — Registro.** O log de transparência do servidor (`--tlog`):
tamanho, raiz e *checkpoint* assinado, as últimas entradas (notas e recibos
de busca ancorados) e um campo para ancorar uma nota. O botão *verificar*
não pergunta ao servidor se está tudo certo: **o navegador confere sozinho**
— a assinatura Ed25519 do *checkpoint* (WebCrypto), a prova de inclusão de
cada entrada e a consistência com o último *checkpoint* que este navegador
viu; a chave do log é fixada no primeiro uso e uma troca é denunciada. A
prova escolhida é desenhada: a folha, os irmãos que sobem, a raiz.
Detalhes: [TRANSPARENCIA.md](TRANSPARENCIA.md).

![Registro, escuro](img/console-ledger.png)

**Confiar — Dossiê** (0.8). Solte um dossiê de auditoria (`.vdossier`, ou o
PDF que o carrega) ou monte o de demonstração: o veredito no nível da água
(a fração de itens verificados; vermelho se algo falhou), a raiz de Merkle
recalculada, e a **trama** — dispositivos do AI Act europeu e da ISO/IEC
42001 nas linhas, as evidências nas colunas, à direita quantos itens
verificados sustentam cada dispositivo, e "nenhum" onde não há evidência.
Embaixo, cada item com o hash e a conferência pelas suas próprias regras, as
assinaturas e as âncoras no log. O dossiê de demonstração pode ser baixado,
com a página que se verifica sozinha. Detalhes: [AUDITORIA.md](AUDITORIA.md).

![Dossiê, escuro](img/console-dossie-escuro.png)

**Criar — Estúdio** (0.9). Uma tela de nós para imagem, som, vídeo, 3D,
difusão e RL ([ESTUDIO.md](ESTUDIO.md)).

- **Montar o grafo**: a paleta à esquerda, por categoria e com busca,
  acrescenta nós por clique ou por arrasto. Os fios vão de um ponto de saída
  a um de entrada, e durante o arrasto só as entradas de tipo compatível
  acendem. **Pelo teclado**, o inspetor lista as fontes possíveis de cada
  entrada. Clicar num fio o remove; `Delete` remove o nó selecionado; `Alt`
  + setas o move.
- **Navegar**: arrastar o fundo desloca a tela, a roda dá zoom, e um modelo
  de partida ou um workflow importado chega enquadrado.
- **Executar**: depois de *Executar*, cada nó mostra a sua prévia (imagem,
  GIF, som tocável, malha renderizada, valor) e se foi **calculado** ou veio
  do **cache**, com o tempo e o digest. É a mesma metáfora da eclusa: o
  tanque de um nó fica vazio antes de rodar, enche em fluxo quando é
  calculado e mostra água parada quando vem do cache.
- **Conferir**: a execução termina no **selo**, a raiz de Merkle, e
  *Verificar* reexecuta sem cache e diz se a raiz bate.
- **Importar e exportar**: *Importar ComfyUI* aceita o JSON do formato de
  API e mostra as notas de tradução; *Exportar JSON* baixa o grafo.

São seis modelos de partida, e todos rodam sem arquivo nem download, exceto
o de texto → imagem, que pede um checkpoint diffusers. O grafo em edição
fica guardado no navegador. O cache e as prévias vivem enquanto o servidor
vive: mudar um parâmetro recalcula só o que depende dele, e uma execução
toda em cache volta em cerca de 1 s.

![Estúdio](img/console-estudio.png)

![Estúdio, escuro, em português: o mesmo grafo, já calculado, volta inteiro do cache (0 calculados, 5 do cache)](img/console-estudio-escuro.png)

### Rodada 0.10: substratos, treino, física, redes, outras escritas

**Medir — Substratos.** Cada substrato presente roda as sondas da eclusa;
a tabela mostra o veredito (canônico, dentro do envelope, recusado) e a
**impressão numérica** — FMA, FTZ, DAZ, ordem de redução, bits de
mantissa, zero com sinal, NaN, divisão, funções —, com cada campo que se
afasta do oráculo exato marcado.

**Medir — Treino.** O recibo do modelo que o vapor treinou (`priv/lm`): a
curva em texto retido contra as linhas de base que ele precisa vencer
(frequência de bytes, Witten–Bell 3 e 5), os digests, a amostra; e o
**fluxo** além do comprimento de treino, janela a janela, contra o
controle de posições crescendo.

**Simular — Física.** *O caos, rodado duas vezes*: o pêndulo duplo e a
sua cópia a um ulp, animados; o oráculo confere o worker nativo bit a bit
nos primeiros passos; a distância entre os mundos em escala log. *Um gêmeo
digital*: resíduos e CUSUM com o limiar, a falha e o alarme marcados, e o
livro refeito do modelo e das ações.

**Simular — Redes.** Um modelo (Barabási–Albert, Erdős–Rényi,
Watts–Strogatz, comunidades plantadas), o leiaute por forças colorido pelas
comunidades do Louvain, o agrupamento contra o nulo de configuração, o
veredito da lei de potência com o nível d'água no p do *bootstrap*, a
robustez a falhas e a ataques e o topo do PageRank.

**Ler — Visão** ganhou a escolha da **escrita** (latim, árabe, cirílico,
cursiva, 中文, 日本語, 한국어, fórmula), com uma nota que diz o que cada
leitor promete — a cursiva responde com a recusa medida e o caminho. Linhas
árabes aparecem da direita para a esquerda (a ordem visual dos quadros no
*hover*); figuras são marcadas na página com a legenda, e os dados de um
gráfico lido aparecem redesenhados ao lado — linhas, pontos ou barras, nas
cores da série, com as categorias lidas (ou o motivo da recusa); uma
fórmula volta como LaTeX. O rodapé diz **qual leitor** leu (e se houve
modelo de língua): o árabe e o cirílico têm os seus, sem modelo de língua.

![Física: o gêmeo digital](img/console-fisica.png)

![Redes, escuro, em português](img/console-redes-escuro.png)

![Treino](img/console-treino.png)

![Substratos](img/console-substratos.png)

![Visão: árabe, da direita para a esquerda](img/console-visao-arabe.png)

![Visão: cirílico](img/console-visao-cirilico.png)

![Visão: uma figura com legenda, e o gráfico lido de volta](img/console-figura.png)

![Visão: fórmula → LaTeX, escuro, em português](img/console-formula-escuro.png)

![Visão: a cursiva recusada, com a medida e o caminho](img/console-cursiva-recusa.png)

A identidade no topo diz também **onde o modelo roda**: `substrato CPU` ou
`substrato GPU · <dispositivo>` (com `mix vapor.serve --gpu`).

### Rodada 0.11: dar vida a imagens, esboços, descobrir, ciência, jogos, arquivos

**Fazer — Cena viva**: solte uma foto, uma pintura, uma imagem gerada ou
escolha um exemplo; em segundos ela se move — câmera em perspectiva sobre
as camadas, habitantes que andam pelo chão, clima, luz, vento. A barra de
direção aceita frases ("uma noite de tempestade, três aldeões andando até
a porta, vaga-lumes; orbite devagar") e os chips aplicam uma operação
cada; o roteiro aparece ao lado, as profundidades se ajustam camada a
camada, "mostrar profundidade" e "mostrar o chão caminhável" revelam a
análise. Um desenho solto ganha esqueleto e acena, anda, dança. Saídas:
vídeo de 8 s, **um arquivo HTML que toca offline**, e o arquivo
verificável.

**Fazer — Esboço**: desenho técnico (o esboço ao lado do que ele queria
dizer, as restrições listadas, SVG e DXF; "mostrar sem as restrições" é o
controle) ou planta → 3D (cômodos com área, portas com largura, um
visualizador 3D que gira e aproxima, GLB).

**Descobrir — Matemática**: escolha um teorema (ou um falso, marcado ✗) e
prove — a figura com a afirmação desenhada, o certificado e a conferência
independente; *conjecturar e provar* desenha as retas e círculos achados;
a tabela de Betti realça a torção; a persistência mostra a nuvem e as
barras. **Algoritmos**: o diagrama da rede de ordenação, os 7 produtos e
o gráfico de contagens, o programa mínimo com as conferências de 8/16/32
bits.

**Simular — Ciência**: onze cartões, cada um com o veredito, valor,
referência e controle, e um gráfico quando há (a órbita do estado
coerente, o g(r) do líquido, a dobra do 20-mero). **Jogos**: jogue contra
o agente de autojogo vendo onde a busca olhou; a curva de derrotas por simulação;
as barras da aleatorização de domínio.

**Confiar — Arquivos**: solte um zip salvo de qualquer painel: íntegro ou
alterado, e — se determinístico — recalculado e comparado.

![Cena viva: uma sala de guilda numa noite de tempestade, com tochas, brasas e três aldeões indo até a porta](img/console-cena-guilda.png)

![Cena viva: a paisagem ao entardecer, com pássaros e borboletas](img/console-cena-paisagem.png)

![Um boneco de palitos desenhado à mão, com esqueleto, dançando na paisagem](img/console-cena-desenho.png)

![A cena exportada, sozinha, offline](img/console-cena-exportada.png)

![Esboço → desenho técnico: retas, círculo e arco com as restrições achadas](img/console-esboco-tecnico.png)

![Esboço → planta 3D](img/console-esboco-planta.png)

![Matemática](img/console-matematica.png)

![Algoritmos](img/console-algoritmos.png)

![Ciência](img/console-ciencia.png)

![Jogos](img/console-jogos.png)

### Rodada 0.12: bancada, engenharia, lógica, tabuleiros, proteínas, render

A navegação foi regrupada pelo que se faz — *Perguntar*, **Resolver**
(Bancada, Engenharia, Lógica), *Descobrir*, *Simular* (com Tabuleiros e
cartas, Proteínas), *Fazer* (com Render), *Ler*, *Medir*, *Confiar* — e
cada grupo está **em ordem alfabética no idioma mostrado** (o
`Intl.Collator` reordena ao trocar de idioma; as setas seguem a ordem
visível). **Ctrl/⌘ K** abre a **paleta de comandos**: todo painel e todo
exemplo, em ordem alfabética, busca sem acentos, Enter abre. O endereço
guarda o painel (`/#bench`).

Os painéis novos (`priv/console/bancada.js`, servido **embutido** na
página — ela continua um documento único que funciona offline) seguem o
mesmo desenho: uma barra (exemplos em ordem alfabética, a ação, o
estado), o **texto** à esquerda (Tab indenta, Ctrl+Enter roda) e um
**resumo** à direita (o que foi reconhecido, o certificado com ✓/✗ e os
números que o justificam), e os resultados embaixo:

- **Bancada**: 16 exemplos (oscilador com unidades, Lorenz, Robertson,
  projétil com arrasto e evento, SIR, calor verificado, Fisher–KPP,
  adensamento de Terzaghi, corda dedilhada, Poisson verificado, planilha
  de viga, raízes, ajuste, Rosenbrock, lata ótima, 4096 osciladores
  nativos); séries escolhíveis, retrato de fase, perfil com controle de
  tempo e mapa de calor u(x, t), tabela de verificação com a ordem, raízes,
  dados e ajuste com resíduos, KKT, faixas de percentis.
- **Engenharia**: oito ferramentas em abas (ordem alfabética), 20
  exemplos; tabelas de tensões e correntes, Bode (módulo e fase),
  transitório; **diagrama unifilar** com fluxos e cor por tensão;
  **pórtico deformado**, apoios, diagramas de normal/cortante/momento por
  barra e **modos animados**; **malha MEF colorida por von Mises**,
  deformada; **rede de tubos** com setas de vazão; espécies, invariantes
  e matriz estequiométrica; flash; **McCabe–Thiele** desenhado.
- **Lógica**: 12 exemplos; veredito, certificados (testemunha, DRUP),
  a testemunha desenhada (faixa colorida de Schur/van der Waerden, K₅ de
  Ramsey, tabuleiro das rainhas), regras de Knuth–Bendix, formas normais e
  derivações, bases de Gröbner.
- **Tabuleiros e cartas**: xadrez (clique para mover, promoção, motor que
  responde, desfazer, girar, FEN, análise, prova de mate com a árvore,
  perft com *divide*), shogi (peças em kanji, as do adversário giradas,
  **mão** clicável para lançamentos, promoção opcional perguntada), Go
  (5–13, komi, MCTS que responde, passe, placar por área), k em linha
  (m, n, k, gravidade; as melhores jogadas destacadas pelo solver exato),
  pôquer (Kuhn/Leduc, curva de explorabilidade, estratégia por conjunto
  de informação).
- **Proteínas**: amostras (1A8O, 1LCD e seus modelos de RMN) ou um PDB
  aberto; **visor 3-D** do traço Cα colorido pela estrutura secundária
  (arrastar gira, roda aproxima), mapa de contatos, sequência; o
  **pipeline** com superposição modelo × nativa, mapa verdade/predição,
  precisões e o PDB do modelo; comparar; alinhar.
- **Render**: o traçador **na GPU** convergindo ao vivo, cena editável
  como texto com recompilação a cada tecla, **arrastar orbita a câmera e
  reescreve a linha `camera`**, cinco exemplos, resolução, PNG, a
  referência do servidor com a **concordância** das radiâncias médias, o
  teste da fornalha com o controle.
- **Cena viva**: inspetor de habitantes, rota por cliques, linha do
  tempo, GIF de quadros exatos ([CENA.md §6.1](CENA.md)).

Conferido por `test/js/console_desks.mjs` no Chromium sem cabeça (com
WebGL2 sobre SwiftShader): **todo exemplo de todo painel** rodado pela
página, xadrez e Go jogados, a paleta usada, a ordem alfabética conferida
nos dois idiomas, nenhum erro de página — 65 conferências
(`console_desks_test.exs`).

![Bancada: o atrator de Lorenz, séries e retrato de fase](img/console-bancada.png)

![Engenharia: pórtico deformado, apoios e diagramas](img/console-engenharia-portico.png)

![Engenharia: fluxo de potência de Stagg & El-Abiad, diagrama unifilar](img/console-engenharia-potencia.png)

![Engenharia: placa em balanço (QM6) colorida por von Mises](img/console-engenharia-mef.png)

![Lógica: R(3, 3) = 6 com a testemunha em K₅ e a refutação DRUP](img/console-logica.png)

![Tabuleiros: xadrez com o motor respondendo e os lances legais do cavalo](img/console-xadrez.png)

![Tabuleiros: shogi](img/console-shogi.png)

![Proteínas: o pipeline em 1A8O — superposição e mapa de contatos](img/console-proteinas.png)

![Render: o traçador na GPU, cena como texto](img/console-render.png)

![A paleta de comandos (Ctrl/⌘ K)](img/console-paleta.png)

### Mercados (0.13): Finanças e Mesa de operações

Um grupo novo na navegação, em ordem alfabética como os outros. Os dois
painéis seguem o desenho dos painéis de *Resolver*: o texto do domínio à
esquerda, à direita um **selo** com o veredito e os números que o
sustentam, e embaixo os gráficos e as tabelas. O que é novo no desenho:

- **o selo** — uma moldura que muda de cor (eclusa, brasa, areia) e diz em
  uma linha se o resultado se sustenta e por quê ("todo instrumento
  reprecificado a 4,5·10⁻¹⁶", "arbitragem de borboleta em k ∈ [0,645;
  1,255]", "bits idênticos — x86_64 · 2 threads");
- **os portões** — uma lista de verificações com ✓/✗ e o detalhe de cada
  uma (os quatro portões de ruído do backtest; os três certificados da
  sessão de bolsa);
- **o semáforo de Basileia** — verde, amarelo ou vermelho para as últimas
  250 previsões de VaR;
- **a escada do livro** — preços no centro, profundidade de compra à
  esquerda e de venda à direita, o spread sombreado;
- **a cadeia** — cada entrada do diário como um bloco com o começo do seu
  hash, ligado ao anterior; um clique mostra o evento e os relatórios;
  negócios em verde-eclusa, recusas em brasa; abaixo, a prova de Merkle do
  primeiro negócio, o feed ITCH em hexadecimal e os relatórios FIX.

Finanças tem oito tarefas (Arbitragem, Backtest, Calendário e dinheiro,
Curva, Monte Carlo nativo, Opções, Carteira, Risco) e a Mesa três (Livro
de ofertas, Sessão de bolsa, Microestrutura), cada uma com exemplos —
inclusive os que **devem** dar errado (um preço abaixo do limite de não
arbitragem, uma espiada no amanhã, uma cotação que deixa um forward
negativo). *Salvar* grava um arquivo do vapor que se recalcula
(`finance.*` é replayable; o Monte Carlo não, porque exige o worker).
`test/js/console_markets.mjs` roda **todo exemplo de toda tarefa** pela
página no Chromium, em inglês e depois em português, clica na cadeia e usa
a paleta (87 conferências, nenhum erro).

**Em português, tudo em português.** As visualizações são escritas uma vez,
com o inglês que o servidor fala; em português, uma única passada sobre o
resultado já desenhado traduz cada rótulo e cada frase do servidor
(vereditos, portões, recusas, a primeira linha de comentário dos
exemplos) por uma tabela de frases e regras com lacunas numéricas. Ela roda
**depois** da visualização, então a lógica que lê o inglês (um veredito por
expressão regular) não muda; código, *hashes* e os bytes de FIX e ITCH
nunca são traduzidos. O teste do navegador lista, exemplo por exemplo, toda
palavra inglesa que sobrar na versão em português — uma lacuna da tabela
aparece como falha, não como um painel meio traduzido.

![Finanças: curva DI (DI1 + LTN + NTN-F), cada instrumento reprecificado](img/console-financas-curva.png)

![Finanças: a fatia SVI de Vogt — o ajuste recupera os parâmetros, a densidade fica negativa e g(k) < 0 é apontado](img/console-financas-sorriso.png)

![Finanças: Monte Carlo no worker — bits do oráculo e de duas threads, asiática com variável de controle](img/console-financas-montecarlo.png)

![Finanças: o melhor de 30 cruzamentos de médias sobre ruído — reprovado pelo Sharpe deflacionado](img/console-financas-backtest.png)

![Finanças: VaR normal em caudas grossas — Kupiec rejeita, semáforo amarelo](img/console-financas-var.png)

![Finanças: uma borboleta que paga, achada e conferida em racionais](img/console-financas-arbitragem.png)

![Mesa: o livro de ofertas, a cadeia do diário, a prova de Merkle, ITCH e FIX](img/console-mesa-livro.png)

![Mesa: uma sessão de bolsa auditada — motor ingênuo, limites, feed, Hawkes](img/console-mesa-sessao.png)

![Mesa: Hawkes plantado e ajustado, com o teste de reescala do tempo](img/console-mesa-hawkes.png)

## As chamadas, para qualquer cliente

| | |
|---|---|
| `GET /v1/vapor/info` | contrato do modelo, contexto do servidor, adaptadores, estado da biblioteca |
| `GET /v1/vapor/library` · `POST /v1/vapor/library` `{name, data}` | arquivos e avisos · ingerir (base64, até ~12 MB por chamada) |
| `POST /v1/vapor/search` `{query, k}` | passagens com proveniência, provas, raiz e recibo |
| `POST /v1/vapor/verify` | recalcular um resultado de busca |
| `POST /v1/vapor/citations` `{answer, query, k}` | conferir citações literais |
| `POST /v1/vapor/search_image` `{name, data}` | imagens parecidas |
| `POST /v1/vapor/quality` `{text}` | veredito do portão de texto, limiares e medidas |
| `POST /v1/vapor/ocr` `{name, data, script?}` | texto de uma imagem ou das páginas escaneadas de um PDF: linhas, caixas, confianças por caractere; tabelas com estrutura, células e Markdown/HTML/CSV; figuras com legenda e os dados dos gráficos; `script`: `latin`, `arabic`, `cyrillic`, `cursive` (recusada), `zh`, `ja`, `ko`, `math` |
| `GET /v1/vapor/substrates` | cada substrato presente admitido por medida: veredito, impressão numérica, sondas |
| `POST /v1/vapor/physics` `{demo}` | `chaos`: o pêndulo duplo e a sua cópia a um ulp, oráculo × nativo; `twin`: resíduos, CUSUM, alarme e o livro refeito |
| `POST /v1/vapor/graph` `{model, n, seed}` | uma rede, comunidades, lei de potência, agrupamento contra o nulo, robustez, PageRank |
| `GET /v1/vapor/lm` | o recibo de `priv/lm` e o fluxo além do comprimento de treino |
| `POST /v1/vapor/scene/analyze` `{name, data}` (ou `name: "sample:outdoor"`) | a cena: camadas (PNG com alfa) e profundidades, horizonte, chão caminhável, luz, paleta |
| `POST /v1/vapor/scene/rig` `{name, data}` | o esqueleto de um desenho: ossos, malha e pesos, a imagem com alfa |
| `POST /v1/vapor/scene/direct` `{prompt}` | o prompt como operações e as palavras não entendidas |
| `POST /v1/vapor/scene/export` `{scene, title?}` | **uma página HTML** autônoma que toca a cena offline |
| `POST /v1/vapor/sketch` `{name, data, mode, snap?, longest?}` | `vector`: retas, círculos, arcos, restrições, SVG, DXF; `plan`: paredes, portas, cômodos com área, malha e GLB |
| `POST /v1/vapor/prove` `{theorem}` · `{discover: true}` · `{homology: true}` · `{persistence, seed}` · `{}` | uma prova com certificado e conferência; as conjecturas provadas; os números de Betti; as barras de persistência; a lista de teoremas |
| `POST /v1/vapor/discover` `{task: network, n}` · `{task: matmul}` · `{task: synth, spec}` | a rede de ordenação e o controle; o algoritmo de 7 produtos, as contagens e a classe; o programa mínimo e a conferência |
| `POST /v1/vapor/science` `{experiment}` | valor, referência, controle, limiar, veredito e dados para gráfico |
| `POST /v1/vapor/games` `{task: selfplay \| randomization}` (`alphazero`, o nome de 0.11, ainda aceito) · `POST /v1/vapor/games/move` `{board, sims}` | a curva de derrotas por simulação; a robustez em mundos não vistos; o lance do agente com as visitas da busca |
| `POST /v1/vapor/archive` `{kind, recipe, result}` · `POST /v1/vapor/archive/check` `{data}` | o arquivo (zip, base64; os tipos determinísticos calculados pelo servidor a partir da receita); a conferência e o recálculo |
| `POST /v1/vapor/audit` `{data, log_key?}` | conferir um dossiê (ou o seu PDF): itens, cláusulas, assinaturas, âncoras, raiz |
| `POST /v1/vapor/audit/demo` | montar e conferir o dossiê de demonstração; devolve o `.vdossier` e a página HTML |
| `POST /v1/vapor/draw` `{digit, seed, steps, guidance}` | dígito gerado, trajetória, leitura de volta, distância ao treino |
| `POST /v1/vapor/listen` `{name, data}` | dígito falado: rótulo, probabilidades, espectro mel |
| `POST /v1/vapor/merge` `{pair}` | laboratório de fusão: diagnóstico, candidatos medidos, escolha |
| `GET /v1/vapor/studio/nodes` | o catálogo de nós do estúdio (portas tipadas, parâmetros com faixa e padrão) e os modelos de partida |
| `POST /v1/vapor/studio/run` `{graph}` | executar um grafo: por nó, calculado ou em cache, tempo, digest e prévia de cada saída; a raiz de Merkle |
| `POST /v1/vapor/studio/verify` `{graph, root}` | reexecutar sem cache e comparar a raiz |
| `POST /v1/vapor/studio/comfy` `{workflow}` | traduzir um workflow do ComfyUI (formato de API): o grafo e as notas, ou a recusa com os nós sem tradução |
| `POST /v1/vapor/solve` `{text, ensemble?}` | a bancada: tipo reconhecido, solução e evidência (passos, ordem observada, KKT, faixas) |
| `POST /v1/vapor/engineering` `{kind, text, method?}` | circuit, power, structure, fem, pipes, reactions, flash, distill — resultado e certificado |
| `POST /v1/vapor/logic` `{text}` | veredito com certificado (modelo, DRUP, testemunha, regras, base) |
| `POST /v1/vapor/chess` `{fen, action, move?, depth?, n?}` · `/shogi` `{sfen, …}` | estado e lances legais, lance, motor, análise, prova de mate, perft |
| `POST /v1/vapor/go` `{size, komi, moves, action?, sims?}` · `/mnk` `{m, n, k, gravity, moves}` · `/poker` `{game, iterations}` | Go por lista de lances; k em linha resolvido; CFR+ com curva e estratégia |
| `POST /v1/vapor/protein` `{action: analyse \| compare \| pipeline \| align, …}` | estrutura, métricas, pipeline, alinhamento |
| `POST /v1/vapor/render` `{text, width, height, spp}` · `GET /v1/vapor/render/furnace` | PNG de referência e radiância média; as fornalhas e o controle |
| `POST /v1/vapor/scene/gif` `{frames, fps}` | quadros PNG exatos → GIF |
| `POST /v1/vapor/finance` `{kind, text}` | a mesa de finanças (0.13): `arbitrage`, `backtest`, `book`, `calendar`, `curve`, `exchange`, `mc`, `micro`, `options`, `portfolio`, `risk` — resultado e certificado ([FINANCAS.md](FINANCAS.md)) |
| `GET /v1/vapor/thumb?doc=…` | miniatura PNG de uma imagem indexada |
| `GET /favicon.ico`, `/logo.svg`, `/manifest.webmanifest`, `/icon-192.png`, `/icon-512.png` | identidade e instalação |

Com `--token` (ou `VAPOR_TOKEN`), toda chamada exceto `/health` exige
`Authorization: Bearer …` ou o cookie *HttpOnly* que `/?token=…` define uma vez;
o servidor se recusa a escutar fora de 127.0.0.1 sem token.

Verificado por `test/vapor/console_test.exs` (HTTP de verdade, sem modelo de
texto) e num navegador headless (Chromium) em claro, escuro, inglês, português
e 390 px de largura, sem erro de console.

## Limites

- Uma biblioteca por servidor, em memória (persistência: `mix vapor.rag`).
- Upload pela página limitado a ~12 MB por arquivo; para mais, `--docs` ou `mix vapor.rag`.
- O token protege o acesso; a confidencialidade na rede exige TLS na frente (um proxy).
- Mensagens vindas do servidor (recusas, erros) ficam em inglês, como a API; os nomes e descrições dos nós do estúdio também.
- O estúdio lê arquivos (`image.load` por caminho, checkpoints) só dentro de `VAPOR_STUDIO_DIR` (padrão: o diretório onde o servidor foi iniciado) ou de `VAPOR_MODELS`; pela página, os arquivos entram como dados.
