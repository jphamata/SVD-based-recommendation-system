<p><img src="docs/img/logo.svg" alt="vapor" height="40"></p>

# vapor

Compilador e runtime de tensores certificados, escrito em Elixir com
`deps: []`, e um ecossistema de LLM inteiro em cima dele. Programas são
termos de uma álgebra simbólica; o plano de controle na BEAM os reescreve
(só regras exatas), aloca registradores vetoriais com um linear scan cujo
resultado é conferido por um checker provado em Lean, e emite **bits**
diretamente — x86-64 AVX2 e AVX-512, AArch64 NEON, RV64GCV (RVV 1.0) e
SPIR-V — sem C, sem assembler externo, sem LLVM. O código gerado nunca roda
dentro da BEAM: roda em processos isolados (`vapor-worker`, `vapor-fabric`)
que podem morrer sem derrubar nada. Cada compilação passa por uma escada de
verificação de seis degraus e sai com um certificado Ed25519 determinístico,
co-assinável por nós independentes.

A garantia central: na política canônica, **todo substrato produz os mesmos
bits** — x86, ARM, RISC-V (VLEN 128–512), GPU via Vulkan e o oráculo exato —
e esses bits não dependem de quantas threads, de quantas sequências dividem
o passo, de onde mora o cache KV, de qual réplica atende nem de o peso estar
em f32 ou bf16 (sobre os mesmos valores).

Desde 0.4.0 o núcleo **não conhece nenhuma família de modelo**: todo checkpoint
entra por uma **eclusa de modelos** (`Vapor.Lock`) que o transforma num
contrato e num programa conferido; adicionar uma família custa de um arquivo
JSON a um adaptador de topologia. Sobre a mesma álgebra — sem nenhum operador
novo — o vapor carrega **imagem e áudio** (encoder bidirecional, ViT do
Hugging Face, espectro certificado, codecs VQ, projetores, injeção de *soft
tokens*, hub any-to-any), **funde modelos** com recibo assinável, **mede se o
que sai é sinal ou ruído** com portões calibrados contra controles, **lê
arquivos** — zip, PDF, Office, EPUB, HTML, imagens — por uma eclusa de
documentos que dá a cada trecho o caminho até a página e o hash do arquivo, e
traz um **console web** embutido onde cada resposta aparece ao lado da sua
evidência.

Desde 0.5.0, as limitações que a 0.4.0 declarou foram atacadas uma a uma: os
adaptadores novos são conferidos contra o **próprio `transformers`** (o que
achou um bug real de rotary parcial, admitido e calculado errado em
silêncio), as rotas any-to-any rodam em **dados reais retidos** (caligrafia
nos dois sentidos por **difusão**, fala de uma voz nunca ouvida), há **OCR**
como modelo admitido pela eclusa e um **decodificador JPEG** idêntico ao
libjpeg bit a bit, a fusão ficou **20× mais rápida com os mesmos bits** e ganhou
**diagnóstico e seleção por medição**, e o console fala **inglês e português**
(e tem uma face no terminal, `mix vapor.tui`).

Desde 0.6.0, o vapor serve **modelos de fronteira sem trocar bits por
velocidade**: MoE **esparso** por predicação de linha (bits = denso, 2,6× no
decode), cache **latente** do MLA (85× menos memória), janela deslizante
exata com **cache circular** na tabela de blocos (7,9× mais sequências a
32 k), **Mamba** com estado mantido no worker (custo por token constante),
**especulação em árvore** sobre páginas compartilhadas (saída = gulosa do
alvo) e **paralelismo de tensor exato**. Ganhou **convolução sem kernel de
convolução**, **VAE** e **DiT** do diffusers, **atenção cruzada** e o
**Whisper**, todos conferidos contra as referências; `÷` virou **IEEE** em
todo substrato (semântica v2), o lema de **Higham** está provado em Lean, e
os atestados podem ser **ancorados num log de transparência** (RFC 9162 +
C2SP) que o próprio navegador verifica.

Desde 0.7.0, o vapor lê **o documento escaneado de escritório** de ponta a
ponta: PDFs de *scanner* em **CCITT Group 3/4** (bit a bit iguais ao libtiff),
LZW e RunLength; páginas de **duas e três colunas na ordem de leitura**; e um
**modelo de língua** na decodificação que corta os erros do leitor e **se
abstém onde não há língua** — com os controles que provam as duas coisas. A
saída estruturada passou a obedecer **`pattern` e `format`** do JSON Schema
(datas com o calendário real, UUID, IPv4, e-mail…), a fusão de modelos roda
**do disco para o disco** com os mesmos bytes, e o console mostra a ordem de
leitura e *quem decidiu cada letra*.

Desde 0.8.0, o motor **serve na GPU** com sessões residentes no Vulkan
(10–12 ms/token contra 72 sem sessão, 8 kB por token em vez de 1 MB, os bits
da CPU), os especialistas em **4 bits** pulam as linhas não escolhidas com os
mesmos bits, o **Mamba-2** entra pela eclusa conferido contra o
`transformers` (e com uma divergência dele com o código de treino achada e
medida), e o paralelismo de tensor exato atravessa **nós de um cluster
Erlang** — com *failover* que não muda um bit e réplicas comparadas bit a
bit. A leitura de escaneados fecha com **tabelas** (estrutura, células
mescladas, colunas tipadas; Markdown, HTML, CSV) e **JBIG2** (bit a bit
igual ao jbig2dec). As regras de reescrita estão **provadas em Lean** num
modelo IEEE-754 de bits — e a prova corrigiu uma afirmação falsa sobre NaN. E
toda essa evidência vira um **dossiê de auditoria** assinado, que se confere
offline (no terminal, no PDF ou no navegador) e mostra a quais dispositivos
do AI Act europeu e da ISO/IEC 42001 cada peça se refere — e onde não há
nenhuma.

Desde 0.9.0, há um **estúdio**: grafos de nós tipados para imagem, som,
vídeo, 3D, difusão e aprendizado por reforço, no espírito do ComfyUI, mas
com **cache exato por conteúdo** (mude um parâmetro e só o que depende dele
roda de novo) e uma **raiz de Merkle por execução** que qualquer um
reexecuta para conferir. Workflows do ComfyUI são importados no subconjunto
traduzível, com cada tradução declarada. O **Stable Diffusion** entra pela
eclusa (U-Net, VAE, CLIP, DDIM/Euler/DPM++ 2M; texto → imagem, imagem →
imagem, inpainting) e é **igual ao diffusers a ~10⁻⁶**. Um **ampliador
consistente** garante que reduzir o resultado devolve a entrada e ganha de
+1,5 a +5,8 dB em texto. Há políticas de RL treinadas como programas (o
CartPole equilibra 473 de 500 passos) e malhas 3D fechadas exportadas em
GLB. Para agentes, um **servidor MCP** cujo cache vive entre chamadas e cujos
resultados se verificam. E, no console, a tela de nós com o selo de cada
execução.

Desde 0.10.0, um acelerador novo só calcula depois de **medido**: a
**eclusa de substratos** roda sondas de resposta conhecida e devolve uma
admissão assinada — canônico, dentro do envelope de erro provado (com a
impressão numérica: FMA, FTZ/DAZ, ordem de soma, bits de mantissa) ou
recusado. Por ela entram o **Metal** (tradutor MSL e daemon, testados por um
*shim* de cabeçalhos, não num Mac), a **Tenstorrent** e qualquer PJRT (exportação
StableHLO e um kit de admissão portátil, rodado no XLA de CPU) e o
**FreeBSD** (worker com Capsicum; compilado, não executado). O **pré-treino**
de um Llama de bytes roda como programas vapor com paralelismo de dados que
não muda um bit com o número de workers, e o **contexto sem fim** (âncoras
+ janela, RoPE re-baseado) lê 14× o comprimento de treino a 3,03 bits/byte
contra 5,81 do controle. Há um **motor de física** (XPBD) para RL e
**gêmeos digitais** — o pêndulo duplo caótico com os mesmos bits em todo
substrato, identificação de parâmetros por gradiente através do simulador,
um gêmeo que alarma 18 passos depois de uma falha de 0,5 % e cujo livro se
refaz —, **redes complexas** em que toda afirmação vem com o seu modelo
nulo, e o OCR passou a ler **árabe** (bidi conferido contra o python-bidi),
**cirílico**, **chinês, japonês e coreano**, **gráficos** (os dados de volta,
ou uma recusa — nunca um número inventado) e **fórmulas** (LaTeX). A letra de
mão continua recusada, com a medida que a recusa.

Desde 0.11.0, o vapor **descobre e prova**, sempre com uma busca que propõe
e um verificador que decide: redes de ordenação ótimas certificadas pelo
princípio 0-1, um produto 2×2 com 7 multiplicações conferido nos inteiros,
truques de bits mínimos por exaustão; teoremas de geometria pelo método
algébrico (reta de Euler, nove pontos, Pappus, Simson) — e **conjecturas
achadas sem serem pedidas** —, homologia com torção. A **ciência** vai do
estado coerente quântico ao equilíbrio de um **tokamak** (Solov'ev), ao
Hartree–Fock do H₂ (−1,1167 hartree, o valor do livro), à fixação de
mutantes, à filogenia e ao dobramento HP, cada um contra a sua referência.
Um agente de **autojogo** pequeno (política + valor + PUCT) aprendeu o jogo da velha só jogando consigo e não
perde nenhuma das linhas ótimas do jogo perfeito com 128 simulações por lance (com 8, perde 13 % delas, contra 97 % da busca sem treino); políticas treinadas em mundos variados
aguentam mundos que não viram. E **qualquer imagem ganha vida**: camadas
com o fundo reconstruído, câmera, habitantes que andam pelo chão, clima,
luz e vento, desenhos animados por esqueleto, tudo **dirigido em
palavras**, reproduzível pela semente e exportado como uma página que toca
offline; um esboço vira desenho técnico (SVG/DXF) ou planta 3D (GLB).
Tudo se **salva** como arquivo verificável — e, quando determinístico,
recalculável.

Desde 0.12.0, o vapor **resolve o problema que o usuário escreve**, não
uma lista de demonstrações: a **bancada** lê equações com unidades
(conferidas antes de rodar), EDOs rígidas, EDPs verificadas por solução
manufaturada, sistemas, ajustes e otimização com KKT, e espalha a
incerteza por 4096 cópias compiladas para o worker nativo (53× a BEAM);
a **engenharia** lê netlists SPICE, listas de barras, pórticos, malhas de
MEF, redes de tubos e reações, e devolve com cada resposta um
**certificado calculado fora do solver** (Kirchhoff, desbalanço,
equilíbrio, continuidade, invariantes estequiométricos); a **lógica**
decide afirmações com prova conferível (SAT com refutação DRUP, Schur,
van der Waerden, Ramsey, Knuth–Bendix, Gröbner) e **confere a proposta
de qualquer um**, inclusive de um modelo de linguagem pelo MCP; os
**tabuleiros** têm xadrez, shogi e Go fixados por perft e contagens
publicadas, provas de mate conferidas, k em linha resolvido e pôquer por
CFR+ com explorabilidade exata; as **proteínas** têm as métricas do
campo (iguais ao TM-align), dobramento por contatos e contatos pela
coevolução, com a comparação honesta com os preditores de ponta; e o
**render** traça a luz fisicamente na GPU de quem olha, com uma
referência que passa no teste da fornalha. Habitantes da cena viva têm
nome, fala, rota e tempo. Tudo no console — navegação alfabética, paleta
de comandos — e pelo MCP.

Desde 0.13.0, o vapor leva o seu princípio — **cada número com o que
permite julgá-lo** — ao mercado, onde a dor é verificabilidade, não
velocidade. **Finanças**: dinheiro decimal exato; calendários ANBIMA/B3,
NYSE e TARGET por regras **iguais ao QuantLib dia a dia de 1990 a 2078**;
curvas DI1/LTN/NTN-F e swaps com todo instrumento reprecificado; opções
(BSM, Heston, árvores) iguais ao QuantLib, com os limites de não
arbitragem conferidos antes de resolver e o sorriso SVI com a arbitragem
de borboleta apontada; **Monte Carlo compilado para o worker com o
gerador dentro do programa** — os bits do oráculo e de duas threads, 6–23×
a BEAM; VaR com Kupiec e Christoffersen; e **backtests com quatro portões
de ruído**, entre eles um certificado de **ausência de antecipação** por
invariância de prefixo. A **arbitragem é decidida exatamente** pelo lema
de Farkas: um portfólio ou preços de estado, em racionais, conferidos por
quem quiser — e a mesa aceita a proposta de qualquer um, inclusive de um
modelo pelo MCP. Na **mesa de operações**, um livro de ofertas cujo
diário é encadeado por SHA-256 e fechado por Merkle, julgado por um
**motor ingênuo independente** (que achou um bug real), ITCH 5.0, FIX 4.4,
risco pré-negociação, Hawkes, Avellaneda–Stoikov e Almgren–Chriss, e uma
sessão de bolsa em que **o backtest é o código da bolsa**. Da lista de
pendências: simplex racional com certificado de Farkas, arquivos
**assinados**, °C/°F como leituras e MOSFET/bipolar **iguais ao ngspice**.
E o projeto passou a se **defender**: uma monografia em abnTeX2 e o
roteiro da defesa oral.

Desde 0.14.0, o vapor deixa de ser vitrine e vira **bancada aberta**: em
vez de escolher entre problemas prontos, a pessoa (ou um modelo) **escreve
o problema** em Alembic, uma linguagem pura e saneada — combustível em todo
passo, tetos de tamanho, processo com teto de memória, nenhum átomo criado.
O **Athanor** busca qualquer coisa escrita nela com um portfólio de
estratégias (exaustiva, recozimento, MAP-Elites, CMA-ES, bayesiana, o
modelo e a pessoa) contra **busca aleatória com o mesmo orçamento**, e
devolve um certificado que a **Touchstone** reconfere sem confiar na busca
(R(3,3) = 6 provado sobre 32 768 grafos; a conjectura de Euler refutada em
40; regras de *trading* desmascaradas pelo *holdout*). O **Crucible** recebe
o sistema da pessoa e responde com evidência que não precisa de gabarito
(leis de conservação **provadas sobre ℚ**, ordem observada, teoremas que
valem para qualquer entrada). O **Assay** responde à pergunta mais comum da
pesquisa em IA — *essa diferença é real?* — com poder, empates, calibração
contra o seu piso e leis de escala que precisam prever as maiores corridas.
Tudo também pelo terminal, em estilo Unix (JSON em *pipes*, códigos de
saída com significado), por MCP e pela TUI; o console ganhou uma identidade
própria em que a fornalha é o gráfico da busca e a pedra de toque é o
veredito.

Desde 0.16.0, o vapor é mais **puro** e **conversa**. Saiu o que só reencenava um exemplo fixo
(redes complexas, descoberta de algoritmos, jogo da velha, dez painéis de demonstração — quase
3 000 linhas). Entrou o **Majlis**: conversas como uma árvore endereçada por conteúdo, em que
editar e pedir outra resposta criam ramos navegáveis (‹ i/n ›), bifurcar não copia nada, o
contexto que o modelo vai ler aparece mensagem a mensagem antes de enviar (com as fixadas, que
nunca saem, e uma compactação que nomeia o que resume), e um link compartilhado morre quando é
revogado — tudo num arquivo seu, a **Khazāna**, cuja raiz sobrevive a uma queda em qualquer byte
(o protocolo de dois slots do ASAS). O **Dīwān** é um interpretador só para a linha de comando, o
TUI, o terminal do console (enjaulado) e a API. O **Mīzān** é um dialeto formal para afirmações que
se *decidem*: uma árvore neutra impressa em latim ou em árabe com o mesmo hash, raízes trilaterais
como domínios e *awzān* como regimes, e cada obrigação provada por um procedimento de decisão do
vapor — ou refutada no ponto que a quebra. Um servidor de linguagem (`vapor lsp`) serve VS Code,
Neovim e Emacs; a geometria da informação entra onde há distribuições para medir; e um
**livro-razão de garantias** diz, afirmação por afirmação, o que é provado, conferido, testado,
argumentado ou devido.

- Rodada 0.16 — purificar: conversas **[docs/MAJLIS.md](docs/MAJLIS.md)** · o depósito **[docs/KHAZANA.md](docs/KHAZANA.md)** · o terminal único **[docs/DIWAN.md](docs/DIWAN.md)** · o Mīzān **[docs/MIZAN.md](docs/MIZAN.md)** · editores **[docs/EDITORES.md](docs/EDITORES.md)** · geometria da informação **[docs/GEOMETRIA.md](docs/GEOMETRIA.md)** · garantias **[docs/GARANTIAS.md](docs/GARANTIAS.md)** · o escrutínio do pedido, do ASAS e do manifesto Al-Mīzān: **[docs/DIRETRIZ.md §19](docs/DIRETRIZ.md)**

<p><img src="docs/img/conversas.png" alt="Conversas: uma conversa com um ramo editado (1/2), uma mensagem fixada, o contexto como barra e a árvore de ramos com o caminho em ouro" width="760"></p>

Desde 0.15.0, o vapor **decide** onde antes exibia ou replicava — o grupo
**Opus**. A **Amálgama** soma qualquer coisa em qualquer ordem, com qualquer
agrupamento e qualquer número de nós, e dá **um** resultado: a soma real
exata arredondada uma vez (um acumulador de Kulisch nos inteiros da BEAM);
o treino com `reduce: :exact` dá os mesmos bits com 1, 2 ou 3 trabalhadores
e uma queda no meio, para qualquer número de micro-lotes. A **Copela** pega
corrupção silenciosa de silício conferindo `y·r = x·(Wᵀr)` em aritmética
exata, com a tolerância de Higham **provada**: a conferência custa
`O(b·(n + k))` contra `O(b·n·k)` do produto (256 × 256, lote 8, na BEAM:
3 ms contra 142 ms do produto pelo oráculo exato), nenhuma acusação contra
quatro ordens de soma conformes, quarentena com diário Merkle. O **Rebis** decide se dois circuitos são a mesma
função (tabela-verdade, ou *miter* + SAT com prova DRUP conferida) e prova
identidades de palavra por álgebra sobre ℤ — um multiplicador de 32 bits em
1,4 s, onde o SAT é exponencial; acha o gatilho de 32 bits de um cavalo de
Troia que 4 096 padrões aleatórios não viram; AES-GCM derivado de GF(2⁸) e
GF(2¹²⁸) igual ao OpenSSL; estabilizadores em 400 qubits. O **Aludel**
(absorvido do PALADIN) decide afirmações polinomiais numa caixa em inteiros
exatos — certificado com testemunha reproduzível, refutado num ponto exato
ou esgotado, nunca um palpite — e prova certificados de barreira. A
**Tábua** acha antinomias em contratos com o cenário que as dispara e prova
(DRUP) os pares que nunca colidem. E dois caminhos antes recusados foram
fechados: **JBIG2 Huffman e meio-tom** (conferidos por um codificador
independente e pelo jbig2dec, cujo defeito em `HDEFPIXEL` ficou registrado)
e o **alinhamento de permutações** antes da fusão (Git Re-Basin com húngaro
exato: a rede fundida com a sua cópia embaralhada volta a ser ela mesma).

- Rodada 0.15 — o Opus: Amálgama **[docs/AMALGAMA.md](docs/AMALGAMA.md)** · Copela **[docs/COPELA.md](docs/COPELA.md)** · Rebis **[docs/REBIS.md](docs/REBIS.md)** · Aludel **[docs/ALUDEL.md](docs/ALUDEL.md)** · Tábua **[docs/TABULA.md](docs/TABULA.md)** · fusão alinhada **[docs/FUSAO.md §8](docs/FUSAO.md)** · JBIG2 **[docs/OCR.md §3f](docs/OCR.md)** · o escrutínio do pedido e dos anexos: **[docs/DIRETRIZ.md §18](docs/DIRETRIZ.md)**

<p><img src="docs/img/opus-tabula.png" alt="a Tábua: um contrato de venda com a antinomia C1 × C6 e o cenário que a dispara, as cláusulas em vigor e a sobreposta riscada" width="760"></p>

- Rodada 0.14 — a bancada aberta: Alembic **[docs/ALEMBIC.md](docs/ALEMBIC.md)** · Athanor e Touchstone **[docs/ATHANOR.md](docs/ATHANOR.md)** · Crucible **[docs/CRUCIBLE.md](docs/CRUCIBLE.md)** · Assay **[docs/ASSAY.md](docs/ASSAY.md)** · terminal **[docs/CLI.md](docs/CLI.md)** · o escrutínio: **[docs/DIRETRIZ.md §17](docs/DIRETRIZ.md)**

<p><img src="docs/img/bancada-athanor-escuro.png" alt="a fornalha ao vivo: a régua de Golomb de 7 marcas achada em 25, a linha tracejada da busca aleatória, a partilha do portfólio" width="760"></p>

- Rodada 0.13 — finanças e mesa de operações: **[docs/FINANCAS.md](docs/FINANCAS.md)** · aritmética linear exata: **[docs/LOGICA.md §5](docs/LOGICA.md)** · transistores: **[docs/ENGENHARIA.md §1](docs/ENGENHARIA.md)** · °C/°F: **[docs/BANCADA.md §1](docs/BANCADA.md)** · o escrutínio: **[docs/DIRETRIZ.md §16](docs/DIRETRIZ.md)** · monografia: **[monografia/monografia.pdf](monografia/monografia.pdf)** · defesa: **[monografia/DEFESA.md](monografia/DEFESA.md)**
- Rodada 0.12 — bancada: **[docs/BANCADA.md](docs/BANCADA.md)** · engenharia: **[docs/ENGENHARIA.md](docs/ENGENHARIA.md)** · lógica: **[docs/LOGICA.md](docs/LOGICA.md)** · tabuleiros e cartas: **[docs/TABULEIROS.md](docs/TABULEIROS.md)** · proteínas: **[docs/PROTEINAS.md](docs/PROTEINAS.md)** · render: **[docs/RENDER.md](docs/RENDER.md)** · habitantes da cena: **[docs/CENA.md §6.1](docs/CENA.md)** · o escrutínio: **[docs/DIRETRIZ.md §15](docs/DIRETRIZ.md)**
- Rodada 0.11 — cena viva, esboço, salvar/exportar: **[docs/CENA.md](docs/CENA.md)** · matemática: **[docs/MATEMATICA.md](docs/MATEMATICA.md)** · ciência: **[docs/CIENCIA.md](docs/CIENCIA.md)** · o escrutínio: **[docs/DIRETRIZ.md §14](docs/DIRETRIZ.md)**
- Rodada 0.10 — substratos (Metal, Tenstorrent, FreeBSD, cluster): **[docs/SUBSTRATOS.md](docs/SUBSTRATOS.md)** · treino e contexto sem fim: **[docs/TREINO.md](docs/TREINO.md)** · física, RL e gêmeos: **[docs/FISICA.md](docs/FISICA.md)** · escritas, gráficos e fórmulas: **[docs/OCR.md §3g–§3k](docs/OCR.md)** · o escrutínio do pedido: **[docs/DIRETRIZ.md §13](docs/DIRETRIZ.md)**
- Rodada 0.9 — o estúdio, Stable Diffusion, ampliação consistente, RL, 3D, MCP e o mapa dos cursos da Hugging Face: **[docs/ESTUDIO.md](docs/ESTUDIO.md)**
- Rodada 0.8 — GPU residente, 4 bits esparso, Mamba-2, cluster: **[docs/FRONTEIRA.md §1, §4, §6, §7](docs/FRONTEIRA.md)** · medições: **[docs/bench/ROUND08.md](docs/bench/ROUND08.md)**
- Tabelas e JBIG2: **[docs/OCR.md §3e–§3f](docs/OCR.md)** · dossiês de auditoria: **[docs/AUDITORIA.md](docs/AUDITORIA.md)**
- O escaneado de escritório — CCITT, colunas, modelo de língua que se abstém: **[docs/OCR.md §3b–§3d](docs/OCR.md)**
- Modelos de fronteira sem perder os bits — MoE, MLA, janela e anel, Mamba, árvore, fragmentos: **[docs/FRONTEIRA.md](docs/FRONTEIRA.md)** · medições: **[docs/bench/FRONTIER.md](docs/bench/FRONTIER.md)**
- Espacial, latente e áudio — convolução, VAE, DiT, atenção cruzada, Whisper: **[docs/ESPACIAL.md](docs/ESPACIAL.md)**
- Transparência — log Merkle, testemunhas, verificação no navegador: **[docs/TRANSPARENCIA.md](docs/TRANSPARENCIA.md)**
- Eclusa de modelos — contratos, três níveis de adaptador, diagnóstico: **[docs/ECLUSA.md](docs/ECLUSA.md)**
- Visão: OCR por um modelo admitido e JPEG bit a bit: **[docs/OCR.md](docs/OCR.md)**
- Interfaces — GUI, TUI, CLI, e por que não Tauri: **[docs/INTERFACES.md](docs/INTERFACES.md)**
- Any-to-any por primeiros princípios, medido: **[docs/ANY_TO_ANY.md](docs/ANY_TO_ANY.md)**
- Qualidade das saídas — sinal ou ruído: **[docs/QUALIDADE.md](docs/QUALIDADE.md)** · resultados: **[docs/bench/QUALITY.md](docs/bench/QUALITY.md)**
- Fusão de modelos: **[docs/FUSAO.md](docs/FUSAO.md)**
- Documentos (zip, PDF, Office, imagens) e a biblioteca verificável: **[docs/DOCUMENTOS.md](docs/DOCUMENTOS.md)**
- O console web: **[docs/CONSOLE.md](docs/CONSOLE.md)**
- A diretriz desta rodada e o seu escrutínio: **[docs/DIRETRIZ.md](docs/DIRETRIZ.md)**
- Arquitetura, protocolos, modelo de garantias: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**
- Ecossistema (modelos, tokenizador, motor, servidor, treino, formatos) e seu
  escrutínio: **[docs/ECOSSISTEMA.md](docs/ECOSSISTEMA.md)**
- Agentes imutáveis, RAG verificável, modelos de fronteira — reflexão,
  desenho e o que *não* é garantido: **[docs/AGENTES.md](docs/AGENTES.md)**
- Phoenix/Plug, Nx, Livebook, Ecto, Oban… com veredito por item:
  **[docs/ECOSSISTEMA_ELIXIR.md](docs/ECOSSISTEMA_ELIXIR.md)**
- ZK e FHE — o que é fato, o que é analogia, o que foi construído:
  **[docs/ZK_FHE.md](docs/ZK_FHE.md)**
- Pendências (os 8 eixos, atualizados): **[docs/TODO.md](docs/TODO.md)**
- Um passeio executável: **[notebooks/vapor_tour.livemd](notebooks/vapor_tour.livemd)**
- Medições desta máquina: **[docs/bench/BENCH.md](docs/bench/BENCH.md)**
- Apresentação (42 slides): [slides/vapor.pdf](slides/vapor.pdf) (fonte `slides/vapor.tex`)

## O que há aqui

| Camada | Conteúdo |
|---|---|
| Núcleo | álgebra simbólica com dimensões semi-dinâmicas e programas recorrentes; funções canônicas (`exp`, `log`, `rcp`, `rsqrt`, `div` **corretamente arredondada**, `sigmoid`, `silu`, `max`, `tanh`, `gelu_tanh`, `gelu` exata; `softplus` composta) como microprogramas sobre `+ − ×` corretamente arredondados; GEMV f32/bf16/4-bit, **GEMV predicado por linha** e **bloco-diagonal**, GEMM int8, RoPE, atenção GQA contígua e paginada **com janela**, amostragem, transposição, `reshape` |
| Backends | AVX2, **AVX-512** (EVEX, 32 zmm, caudas mascaradas), NEON, RVV, SPIR-V — codificação binária própria, conferida contra binutils e `spirv-val` |
| Substratos | worker nativo (seccomp, W^X, watchdog, **pool de threads**, sessões residentes, contadores `perf_event`), interpretador RVV, QEMU, **Vulkan** (lavapipe) **com sessões residentes** (memória direta ou *staging*, gravações reaproveitadas), oráculo exato |
| Verificação | escada 1–6, envelope diádico exato, Lean 4 (sem `axiom`/`sorry`; Wilkinson, Higham **e as regras de reescrita num modelo IEEE-754 de bits**) com extração para Elixir, certificados com quórum e versão da semântica |
| Auditoria | `Vapor.Audit`: dossiês CBOR canônicos com raiz de Merkle, assinaturas Ed25519 com quórum e âncora no log; cada item conferido pelas suas regras; mapeamento AI Act / ISO 42001 como dado; PDF com o dossiê anexado e HTML que se confere offline; `mix vapor.audit` |
| Transparência | `Vapor.Tlog`: log Merkle RFC 9162 em arquivo só-acréscimo, provas de inclusão e consistência, *checkpoints* C2SP *signed-note*, co-assinaturas de testemunha (recusa retrocesso e bifurcação); recibos de busca ancorados; verificador no navegador |
| Eclusa de modelos | `Vapor.Lock`: o único lugar onde uma família é conhecida; contratos `:causal_lm`, `:encoder`, `:codec`, `:map` (e **partes** declaradas: encoder + decoder) conferidos na fronteira; adaptadores por **dados** (alias JSON: renomear, dividir tensores fundidos), **blueprint** e **topologia**; recusa com quase-acertos e reparo; `mix vapor.lock` |
| Modelos | Llama / Mistral / Qwen2 e as famílias de fronteira **Qwen3, Qwen3-MoE, Mixtral, Gemma 3, DeepSeek-V3 (MLA)**, mais **Phi-3/Phi-4** (alias, com rotary parcial), **Granite 3.x** (blueprint), **ViT**, as torres de visão **e de texto** do **CLIP**, **Mamba** e **Mamba-2** (SSM) e **Whisper** (encoder-decoder) — conferidos contra o `transformers` —, **VAE** (`AutoencoderKL`, encoder e decoder), **U-Net** (`UNet2DConditionModel`) e **DiT** do diffusers, a partir de `config.json` + safetensors **ou** GGUF; YaRN; pesos f32, **bf16 residente** ou 4-bit (`sb4`) |
| Espacial | convolução 2-D/3-D (qualquer *stride*, *padding*, dilatação) **sem kernel de convolução** (gather + sel + reshape + GEMV), GroupNorm exata, *upsampling*, atenção sobre pixels e **atenção cruzada** — mesmos bits em todo substrato |
| Any-to-any | imagem (PPM/PNG, patches exatos), áudio (WAV, espectro de Hann certificado, síntese aditiva), codec VQ (codificar por `sample` guloso), projetores ajustados em forma fechada, injeção de *soft tokens* no decoder, hub com pivô — **nenhum operador novo**; em dados reais: caligrafia → dígito e dígito → caligrafia por **difusão** (DDIM verificado contra o denoiser ótimo em forma fechada), fala → dígito (espectro mel certificado), cadeia voz → desenho |
| Fusão | linear, *task arithmetic*, SLERP, TIES, DARE, **RegMean**: determinística, ≈ 11 M parâmetros/s, compatibilidade checada pela eclusa, recibo co-assinável; **diagnóstico** do regime pelos pesos e **seleção por medição**; **do disco para o disco** (um tensor por vez, mesmos bytes); `mix vapor.merge` |
| Documentos | eclusa de arquivos sem dependência: zip recursivo à prova de *zip bomb*, PDF (object streams, `ToUnicode`, criptografado recusado, **CCITT G3/G4, LZW, RunLength, JBIG2**), Word/Excel/PowerPoint/OpenDocument/EPUB, HTML, PNG decodificado (Adam7, 1–16 bits), **JPEG decodificado** (baseline e progressivo, = libjpeg), **OCR** de páginas escaneadas e imagens **em colunas, na ordem de leitura, com modelo de língua**, e **tabelas** célula a célula; proveniência até a página; biblioteca com raiz sobre texto **e** arquivos, busca por imagem parecida; `mix vapor.rag` |
| Estúdio | `Vapor.Studio`: grafos de 64 nós tipados (imagem, som, vídeo, 3D, RL, visão, difusão) com **cache exato por conteúdo**, recibos, **raiz de Merkle por execução** e `verify`; importação de workflows do ComfyUI; codecs GIF/Y4M/MJPEG-AVI próprios; **Stable Diffusion = diffusers** (`Vapor.Diffusion`); **ampliador consistente** (D(y) = x); `Vapor.RL` (CartPole/FrozenLake = gymnasium, REINFORCE como programa); `Vapor.Geom` (SDF → malha fechada, GLB/OBJ/PLY); **servidor MCP** (`mix vapor.mcp`) |
| Console | página única servida em `/` (sem CDN, offline), **inglês/português**, claro/escuro, instalável: conversa com evidência, documentos, **estúdio** (tela de nós com selo e verificação), **visão (OCR, com a ordem de leitura, o que o modelo de língua decidiu e as tabelas desenhadas das células)**, **dossiê** (a trama evidência × dispositivo), **ouvir** (microfone), **desenhar** (difusão), **fusão**, qualidade, eclusa; token para expor; **`mix vapor.tui`** no terminal |
| Qualidade | portões calibrados contra controles (recusam existir sem separação), modelos plantados com verdade em forma fechada, bits/byte contra linhas de base; `mix vapor.quality` (suíte com controles, inclusive em dados reais; sai 1 em falha) e `--model` para checkpoints reais |
| Formatos | **safetensors** (todos os dtypes do formato, shards, escrita BF16/F16 = torch), **GGUF** (leitura F32…Q6_K = gguf-py; escrita f32/q8_0 que o llama.cpp executa), `config.json` ida e volta |
| Texto | tokenizador BPE byte-level e SentencePiece com byte fallback; NFC/NFKC próprio (corrige um bug do OTP); **Jinja hermético** para os chat templates de qualquer modelo |
| Geração | amostragem determinística, motor com **lote contínuo** e **KV paginado** (**anel** quando a janela liga em todas as camadas), **réplicas**, **MoE esparso** e **MLA latente**, geração **recorrente** (SSM, estado no worker), decodificação **especulativa** linear e **em árvore** (busca no prompt) com saída idêntica, **paralelismo de tensor** exato (coluna + all-gather), **entre nós BEAM** com *failover* e réplicas comparadas bit a bit; motor **na GPU** |
| Serviço | API compatível com OpenAI (HTTP/1.1 + SSE, chat templates, **tools**, **`response_format`**, **embeddings**, **recibos** `x-vapor-receipt`), CLI; o mesmo despacho como **Plug** para Phoenix (`integrations/vapor_plug`) |
| Saída restrita | gramáticas de bytes sobre a trie do vocabulário: JSON Schema (com **`pattern` ECMA-262 e `format`**), chamadas de ferramenta, citações literais — válidas **por construção** |
| Agentes | spec como valor; diário encadeado + Merkle + atestado Ed25519; repetir = provar; retomar do disco sem repetir ações; capacidades; apagamento por titular; backends local / OpenAI / Anthropic; cliente **e servidor** MCP |
| RAG | corpus = raiz Merkle; BM25 e escores densos determinísticos; RRF exato; recibos re-verificáveis |
| Numérica | funções elementares **corretamente arredondadas** (Ziv) para tabelas; CBOR canônico (RFC 8949) para tudo que é assinado |
| ZK e FHE | redes int8 como **R1CS** (testemunha pelos kernels certificados, formatos iden3, Groth16 de ponta a ponta, verificador Solidity com gás medido); corpos BabyBear/Goldilocks/BN254 e **NTT** exatos; **polinômios com erro provado** para CKKS; lema em Lean ligando o limite inteiro ao corpo |
| Ecossistema | `integrations/vapor_plug` (Plug/Phoenix/Bandit), `integrations/vapor_nx` (compilador `Nx.Defn` certificado), Livebook |
| Finanças | `Vapor.Finance`: dinheiro exato, calendários (= QuantLib), curvas, opções (= QuantLib), Monte Carlo no worker com bits canônicos, VaR com backtests estatísticos, carteiras com KKT, **backtests com portões de ruído**, arbitragem por LP exato |
| Mesa de operações | livro preço–tempo com diário SHA-256 + Merkle, **juiz ingênuo independente**, ITCH 5.0, FIX 4.4, risco pré-negociação, Hawkes, Avellaneda–Stoikov, Almgren–Chriss, sessão de bolsa auditada |
| Treino | autodiff reverso em termos, LoRA + AdamW + destilação KL como um programa recorrente |
| Opus (0.15) | **Amálgama** (soma exata sem ordem, f16/bf16/f32/f64; treino `reduce: :exact`), **Copela** (corrupção silenciosa por identidade adjunta com tolerância provada; sentinela com quarentena e diário), **Rebis** (equivalência de circuitos com prova DRUP, Gröbner sobre ℤ, GF(2ⁿ), AES-GCM, estabilizadores, AIGER), **Aludel** (positividade de Bernstein em inteiros, barreiras, síntese por LP), **Tábua** (antinomias deônticas, precedências, silêncios, Hohfeld); console, terminal e MCP |

## Início rápido

```sh
nix develop                # Elixir, Zig 0.16, QEMU, Vulkan+lavapipe, spirv-tools, binutils cruzado, elan
make native cross          # vapor-worker + vapor-fabric (host) e workers aarch64/riscv64
make fixtures              # vocabulários reais (SHA-256 fixado) para os níveis de tokenizador
make test                  # todos os níveis cujo tooling está presente
make e2e                   # pipeline inteiro, ver abaixo
mix vapor.bench            # regenera docs/bench (kernels, motor, ULP, tokenizador)
mix vapor.quality          # qualidade das saídas: docs/bench/QUALITY.md, quality.json, galeria PNG/WAV
make slides                # PDF da apresentação
```

Com um modelo (diretório Hugging Face ou arquivo `.gguf`):

```sh
mix vapor.generate --model ./Qwen2-0.5B --prompt "Olá" --max-tokens 64 --temperature 0.7 --storage bf16
mix vapor.serve --model ./model-q8_0.gguf --port 8000 --threads 8 --replicas 2
curl -s localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"Oi!"}],"stream":true}'
mix vapor.export --model ./Qwen2-0.5B --out qwen2.q8_0.gguf --type q8_0        # para o llama.cpp
mix vapor.export --model qwen2.q8_0.gguf --out ./qwen2-bf16 --dtype bf16      # de volta para o HF
mix vapor.lock ./Phi-3-mini-4k-instruct                                       # quem reivindica, o que falta
mix vapor.lock ./MeuModelo --alias meu_alias.json                             # uma família nova, só com dados
mix vapor.merge --method slerp --t 0.3 --out ./fundido ./ModeloA ./ModeloB    # com merge.receipt assinado
mix vapor.quality --model ./Qwen2-0.5B --text retido.txt --reference corpus.txt   # sinal ou ruído?
mix vapor.serve --model ./Qwen2-0.5B --docs ./meus-arquivos                    # console em http://127.0.0.1:8000/
mix vapor.rag index minha.vlib ./meus-arquivos ./pacote.zip && mix vapor.rag search minha.vlib "multa contratual"
mix vapor.ocr read escaneado.pdf foto.jpg                                     # OCR, com a confiança de cada linha
mix vapor.merge --diagnose --base ./Base ./AjusteA ./AjusteB                  # o regime, antes de fundir
mix vapor.merge --out ./f --try "linear;ties:density=0.2" --eval retido.txt ./A ./B   # fundir por medição
mix vapor.merge --stream --method slerp --t 0.3 --out ./f ./ModeloA ./ModeloB  # do disco para o disco, um tensor por vez
mix vapor.tui                                                                 # o console no terminal
mix vapor.finance curve curva_di.txt                                          # curva com o certificado de reprecificação
mix vapor.finance backtest estrategia.txt                                     # os quatro portões de ruído
mix vapor.finance book ordens.txt                                             # diário, juiz ingênuo, ITCH, FIX
mix vapor.archive verify resultado.zip --trusted operador.key.pub             # arquivo assinado
mix vapor.quality --only round13                                              # §5i em ~25 s
mix vapor.quality --only round14                                              # §5j em ~60 s
bin/vapor alembic --card                                                      # a linguagem; depois: bin/vapor athanor run problema.alb
```

`make e2e` faz isso tudo com um checkpoint de demonstração (pesos
aleatórios, vocabulário real do Qwen2): gera texto do diretório safetensors,
exporta para GGUF q8_0, gera do GGUF, exporta o GGUF de volta para um
diretório bf16 com shards, gera dele com pesos bf16 residentes e conversa
com o servidor pelo `curl` (completions, chat em streaming) a partir do
diretório e do GGUF.

Na API Elixir:

```elixir
{:ok, %{config: c, program: p}} = Vapor.Model.load("./Qwen2-0.5B", max_seq: 512, storage: :bf16)
{:ok, e} = Vapor.Engine.start_link(model: "./Qwen2-0.5B", threads: 8, replicas: 2)
{:ok, ids, :length, usage} = Vapor.Engine.complete(e, prompt_ids, max_tokens: 32, temperature: 0.8, seed: 1)
```

Um agente cuja execução é um objeto de prova:

```elixir
spec = Vapor.Agent.Spec.new(name: "ops", instructions: "…", model: %{"kind" => "local", "id" => digest},
                            tools: [%{name: "lookup", effect: "observe", parameters: schema},
                                    %{name: "notify", effect: "act", parameters: schema2}],
                            grants: ["notify"])
store = Vapor.Agent.Store.File.new("/var/lib/vapor/runs")
{:ok, run} = Vapor.Agent.Store.run(store, spec, "…", backend: backend, impls: impls)   # gravado antes de cada passo
{:ok, report} = Vapor.Agent.replay(spec, run.journal, backend: backend, impls: impls) # recomputa, não age
attestation = Vapor.Agent.Journal.attest(run.journal, node_key)
# depois de uma queda: Vapor.Agent.Store.resume(store, spec, run_id, backend: backend, impls: impls)
```

Num app Phoenix: `forward "/llm", Vapor.Plug, name: MyApp.LLM` (ver
[ECOSSISTEMA_ELIXIR.md](docs/ECOSSISTEMA_ELIXIR.md)).

## Resultados (esta máquina: Xeon 2,1 GHz, 2 vCPUs, AVX-512, sem PMU, sem GPU)

- **Rodada 0.14.0** (Elixir 1.14/OTP 25, sem Zig nesta sessão — sem o
  processo nativo): **87 testes, 0 falhas** nos oito arquivos da bancada
  aberta, mais o SDK oficial de MCP, a paridade do ruído em JS e o teste de
  navegador de todas as mesas antigas. **Qualidade: 20/20** (§5j,
  `--only round14`, 60 s): Golomb 25 contra 0 acertos do acaso em 3 923;
  R(3,3) provado em 32 768 grafos; *holdout* ρ = 0,73 com momento plantado
  contra −0,33 num passeio aleatório; 6/6 hamiltonianos aleatórios
  redescobertos e 0 leis inventadas em dissipativos; Kepler simplético 40×
  abaixo da deriva do RK4; erro tipo I 0,00; Krippendorff 0,743. Os
  controles acharam três defeitos (lei de escala que estourava `exp`, um
  átomo por expressão compilada, estado do gerador perdido no Monte Carlo
  sem processo nativo), todos corrigidos. Sem o processo nativo, os três
  exemplos de Monte Carlo do console passam de 120 s pelo oráculo exato —
  limitação do ambiente, não do código.
- **Rodada 0.13.0** (Xeon, 2 vCPUs, sem GPU física, Elixir 1.14/OTP 25, Zig 0.16,
  QuantLib 1.43, simplefix, ngspice 42): **108 testes, 0 falhas** nos oito
  arquivos que a rodada tocou (finanças, console de mercados, rodada 13,
  arquivos, MCP, engenharia, bancada, lógica), em 125 s. A execução completa
  de `mix test` foi interrompida por um reinício da VM e **não** foi repetida
  — os outros módulos não mudaram desde a 0.12. **Qualidade: 23/23**
  verificações novas com controle (§5i, `--only round13`, 23 s): calendários =
  QuantLib por 89 anos; curva reprecificada a 4,5·10⁻¹⁶ (dias corridos:
  > 10⁻⁴); Monte Carlo com os bits do oráculo e de 2 threads, z = 1,17 (Itô
  esquecido: 8,70); DSR 1,0 no sinal plantado e 0,10 no ruído; espiada pega
  no dia 89; 6 000 eventos refeitos pelo motor ingênuo; Hawkes 0,358 (Poisson
  p = 1,7·10⁻²⁰). No console, **todo exemplo** das duas mesas em EN e PT,
  sem nenhuma palavra inglesa sobrando no PT (87 conferências).
- **Rodada 0.12.0** (Xeon, 2 vCPUs, sem GPU física, Elixir 1.14/OTP 25, Zig 0.16;
  Chromium sem cabeça com WebGL2 sobre SwiftShader): **702 testes, 0 falhas**
  por `mix test` em 64 min (115 excluídos por falta de ferramenta nesta
  sessão: QEMU, Vulkan, Lean, binutils, spirv-tools, torch/diffusers,
  llama.cpp, vocabulários, snarkjs, trimesh, jax). **Qualidade: 147/147**
  verificações com controle em 33 min (`mix vapor.quality`, nativo; §5h
  com 27 novas, em 5,8 s): Dormand–Prince a < 10⁻⁸ (RK4 de passo fixo:
  2·10⁻⁴); Robertson 0,7158271 com a troca para Rosenbrock; Crank–Nicolson
  ordem 2,00 contra Euler implícito 1,05; a lata ótima a 10⁻⁶ e a
  divergência relatada sem limites; RC ordem 1,97 contra 0,99; Stagg em 4
  iterações (Gauss–Seidel: 119); QM6 a 0,9 % da viga, Q4 travando a 29 %;
  S(3) = 13 com DRUP (a prova truncada rejeitada); Knuth–Bendix nas dez
  regras; Tales provado; perft 8 902 · 2 039 · 30/900/25 470; 57 posições
  de Go 2×2; Kuhn a 9·10⁻⁵ de explorabilidade; DCA 0,96 contra 0,02
  embaralhado; pipeline TM 0,69 contra 0,20; a fornalha em gradiente a
  6·10⁻⁴ e o estimador viciado pego. No console, **todo exemplo de todo
  painel** rodado pela página no Chromium (65 conferências, nenhum erro).
- **Rodada 0.11.0** (Xeon, 2 vCPUs, sem GPU física, Elixir 1.14/OTP 25, Zig 0.16):
  **611 testes, 0 falhas** — 610 por `mix test` e o 611º (o benchmark
  inteiro) pelo próprio `mix vapor.quality`; 21 excluídos por falta de
  ferramenta nesta sessão (llama.cpp, cliente `openai`, vocabulários
  GGUF/CLIP/HF, snarkjs e **Lean**: o `lake` não estava presente — as
  provas da 0.8 não foram re-executadas). **Qualidade: 122/122**
  verificações com controle (§5f da 0.10, §5g da 0.11): redes de ordenação
  ótimas para n = 3…8 com todo comparador necessário; ⌊(x+y)/2⌋ em 4
  operações, nenhuma em 3; Strassen em 7 produtos, exato nos inteiros (o
  posto 6 nunca); 15/15 enunciados de geometria certos (verdadeiros
  provados, falsos refutados) e 29 conjecturas provadas sem serem pedidas;
  torção de Klein e RP²; 11 experimentos de ciência contra as suas
  referências (H₂ −1,11671 contra −1,1167; tunelamento 0,5431 contra
  0,5445; Solov'ev a 2·10⁻¹²); o agente de autojogo contra **todas** as linhas
  ótimas do jogo perfeito (17/129 perdidas com 8 simulações, 0/135 com
  128; sem treino: 169/175); aleatorização de domínio 484 contra 275
  passos; cômodos de 11,98/19,80 m² e portas de 0,90/1,00 m num esboço; o
  arquivo forjado pego. Uma revisão independente desta rodada achou nove
  problemas — dois de segurança nos arquivos (receitas sem limite, bomba
  de zip), um "provado" por 0/0, um "nunca perde" que vinha de amostra —,
  todos corrigidos e testados ([DIRETRIZ.md §14](docs/DIRETRIZ.md)).
- **Rodada 0.9.0** (Xeon, 2 vCPUs, sem GPU física, Elixir 1.14/OTP 25, Zig 0.16):
  **501 testes, 0 falhas** com todos os níveis presentes aqui — nativo,
  QEMU aarch64/riscv64, Vulkan (lavapipe), Lean, binutils, spirv-val,
  Python, `torch` + `transformers`, **diffusers**, **gymnasium**,
  **trimesh**, **ffmpeg**, Pillow, o **SDK oficial do MCP**, Node, nós
  `:peer` —; 18 excluídos por falta de ferramenta (llama.cpp, cliente
  `openai`, vocabulários GGUF e CLIP, snarkjs). **Qualidade: 75/75**
  verificações com controle (§5e para as novas): pipelines de Stable
  Diffusion = diffusers (maior diferença por pixel 1,4·10⁻⁶; outro
  amostrador: 0,096); o estúdio com uma raiz por grafo e 3 de 6 nós
  recalculados depois de uma edição; o ampliador +2,76 dB sobre Lanczos com
  a mesma projeção, em texto retido, e |D(y) − x| = 10⁻¹⁶ (Lanczos: 0,077);
  CartPole 472,9/500 (sem treino: 18); reamostragem com 69,6 dB de SNR
  (decimação desalinhada: 17,7); o servidor MCP com a segunda execução
  inteira em cache e a raiz falsa recusada.
- **Rodada 0.8.0** (Xeon, 2 vCPUs, sem GPU física, Elixir 1.14/OTP 25, Zig 0.16):
  **469 testes** com todos os níveis presentes aqui — nativo, QEMU
  aarch64/riscv64, **Vulkan (lavapipe)**, **Lean**, binutils, spirv-val,
  Python, `torch` + `transformers` 5.18, diffusers, Node, nós `:peer` —;
  18 excluídos por falta de ferramenta (llama.cpp, cliente `openai`,
  vocabulários GGUF, snarkjs). A execução completa achou **1 falha**: um
  teste da 0.7 que exigia que o JBIG2 fosse recusado — agora ele é
  decodificado; o teste foi atualizado e re-executado em verde.
  **Qualidade: 65/65** verificações com controle (§5d para as novas):
  tabelas com estrutura exata (F1 1,000; a leitura da 0.7: 0,343) e CER por
  célula 5,5 % (livre: 11,7 %); 43/43 fluxos JBIG2 = jbig2dec; sessão na GPU
  = CPU bit a bit; MoE 4 bits esparso = denso; Mamba-2 = `transformers`
  (7,8·10⁻⁷; a norma do outro lado: 0,74); 0 de 209 alterações de um bit
  aceitas num dossiê; fragmentos entre nós BEAM = um worker, também depois
  de perder um nó. Medições: [ROUND08.md](docs/bench/ROUND08.md).
- **Rodada 0.7.0** (Xeon, 2 vCPUs, sem GPU, Elixir 1.14/OTP 25, Zig 0.16):
  **425 testes, 0 falhas** com os níveis nativo, Python (NumPy, Pillow,
  jinja2, mpmath, cbor2) e Node; 90 excluídos por falta de ferramenta aqui
  (QEMU, Vulkan, Lean, binutils, `torch`, vocabulários…) — nenhum dos módulos
  desses níveis mudou nesta rodada. **Qualidade: 55/55** verificações com
  controle (§5c para as novas): páginas escaneadas de 1–3 colunas em CCITT,
  CER **1,3 %** (sem ordem de leitura: 53 %; Tesseract 1,6 %); linhas retidas
  CER **4,4 %** (guloso 6,5 %, Tesseract 5,2 %), foto de página **9,5 %**
  (Tesseract 36,4 %); o modelo de língua muda 0 de 30 linhas de cadeias
  aleatórias (sem a guarda: 27).
- **Rodada 0.6.0** (outra VM: Xeon 2,8 GHz, 2 vCPUs, sem GPU): **400 testes**
  executados em Elixir 1.14 com todos os níveis presentes — nativo, QEMU,
  **Vulkan (lavapipe)**, Lean, binutils, spirv-val, Python, `torch` +
  `transformers` 5.18, diffusers, Node —, 3 excluídos (snarkjs, llama.cpp);
  a suíte completa achou 2 falhas (uma violação real de arquitetura — o motor
  lia a configuração da família — e um `layer_types` ignorado), ambas
  corrigidas e os módulos afetados re-executados em verde. **Qualidade:
  48/48** verificações com controle ([QUALITY.md](docs/bench/QUALITY.md), §5b
  para os recursos novos). Medições da rodada: [FRONTIER.md](docs/bench/FRONTIER.md).
- **Testes:** 375 no núcleo. **Na rodada 0.5.0**, em Elixir 1.14 com os
  níveis de controle, **nativo** (worker com Zig 0.16), **QEMU**
  aarch64/riscv64, **Python** (NumPy, Pillow, mpmath, jinja2, MCP) e
  **`torch` + `transformers` 5.18**: 343 executados, 0 falhas; os 32 de
  níveis sem ferramenta aqui (Vulkan, binutils, spirv-val, Lean, llama.cpp,
  gguf-py, snarkjs, cbor2, cliente `openai`, vocabulários) foram excluídos e
  não reexecutados. Na 0.3.0 todos os níveis — inclusive esses — estavam
  verdes em Elixir 1.14/OTP 25 e 1.18, mais 8 nas integrações (Plug/Bandit,
  Nx).
- **Qualidade das saídas** ([QUALITY.md](docs/bench/QUALITY.md)): 48/48
  verificações (40 até a 0.5.0), cada uma contra um controle — um modelo plantado reproduz sua
  tabela analítica a 1,2·10⁻⁶ pela pilha inteira e, com pesos aleatórios, é
  reprovado como ruído; **em dados reais retidos**: OCR com CER 6,8 % em
  fontes nunca vistas (Tesseract 5,2 %) e 11,7 % numa foto de página com luz
  desigual (Tesseract 36,4 %); fala de uma voz nunca ouvida 90 % (72 % na
  média das vozes); caligrafia → dígito 98 %; dígito → caligrafia por difusão
  lido de volta 100 %, a uma distância do treino de dígito real, não de cópia;
  voz → texto → desenho 90 %; a fusão de dois ajustes finos treinados bate a
  base nos dois idiomas; todo programa modal nativo = oráculo bit a bit.
- **Paridade bit a bit** com o oráculo exato, em todo substrato: programas
  canônicos, modelos inteiros (Llama com viés, Mistral, Qwen2 e as famílias
  de fronteira Qwen3, Qwen3-MoE, Mixtral, Gemma 3, DeepSeek-V3), atenção
  paginada, amostragem, 4-bit, bf16, gradientes — também no GPU (modelo
  inteiro, passo paginado do motor, decode recorrente).
- **Invariâncias** testadas: 1 = 2 = 3 threads; lote = sozinho; qualquer
  fatiamento do prompt; paginado = contíguo; prefill = decode passo a passo;
  qualquer réplica; especulação = alvo sozinho; bf16 = f32 sobre pesos
  arredondados.
- **Oráculos externos** (só nos testes): `transformers`/`torch` (logits
  ≤ 5,6·10⁻⁷ relativo em f32, greedy idêntico; o transformers carrega o
  diretório exportado pelo vapor e reproduz o checkpoint **bit a bit**);
  `tokenizers` (7 configurações × 415 textos idênticos; 184 vetores reais);
  `gguf-py` (desquantização e Q8_0 bit a bit); **llama.cpp** (converte →
  vapor reproduz o transformers; vapor exporta → llama.cpp tokeniza igual e
  reproduz os logits); cliente `openai`; `numpy`/`torch` para todos os
  dtypes do safetensors; `transformers` 5.18 nas 8 variantes de fronteira
  (≤ 8·10⁻⁷); `jinja2` (240 renderizações de templates reais, byte a byte);
  `mpmath` (funções corretamente arredondadas); `cbor2` + `cryptography`
  (certificados verificados fora da BEAM); SDK oficial do **MCP**; avaliador
  de referência do **Nx**; **snarkjs** (Groth16) e uma EVM (gás medido).
- **Desempenho** (detalhes e variação em [BENCH.md](docs/bench/BENCH.md)):
  teto de memória ~53 GB/s; GEMV bf16 2× o f32 (limitado por banda nos dois);
  AVX-512 até 2,3× em kernels de cômputo, igual em kernels de banda, como o
  roofline prevê; motor ~1 500–1 800 tokens/s com 8 sequências num Llama de
  largura 256 e vocabulário 32 000; tokenizador ~535 mil tokens/s num núcleo.
- **Contenção:** SIGILL, SIGSEGV, SIGSYS (seccomp) e SIGALRM no worker,
  SIGSEGV no driver Vulkan e um worker morto sob o motor viram erros
  tipados; o processo renasce, o dispatch faz failover, o motor reabre a
  sessão e segue servindo.
- **Certificados:** compilações independentes produzem payloads idênticos;
  quórum 2-de-3 verificado na borda sem refazer a escada.

## Como as três limitações do antecessor foram fechadas

| Limitação | Resolução | Evidência |
|---|---|---|
| Hardware RVV físico raro | O mesmo worker estático roda nativo numa placa RVV ou sob `qemu-riscv64`; em x86 há três caminhos que precisam concordar bit a bit — worker riscv64 sob QEMU (VLEN 128/256/512), interpretador RVV próprio com *poison* e o oráculo. Nada depende de reduções de ordem não especificada. | `native_test`, `canon_test`, `model_test` |
| Pipeline Vulkan simplificado | Compute completo com bindings à mão (instância → pipelines → barreiras → fence), importação sem cópia via `VK_EXT_external_memory_host`, tabelas do tamanho do quadro, falha do driver contida e roteada. Agora roda modelos inteiros. | `fabric_test`, `model_test` |
| Lean só como oráculo offline | `lake exe vapor-extract` gera `lib/vapor/extracted.ex` a partir dos termos elaborados; o runtime chama o checker de alocação, a admissão sem wrap, o monoide afim e a decisão do envelope provados; digest das fontes conferido. | `extracted_conformance_test`, `audit_test` |

## Estrutura

| Onde | O quê |
|---|---|
| `lib/vapor/f32.ex`, `tensor.ex`, `quant/sb4.ex` | binary32 exato na BEAM, tensores (f32, bf16, inteiros, 4-bit) |
| `lib/vapor/algebra/`, `program.ex`, `canon.ex` | álgebra, let-bindings, funções canônicas |
| `lib/vapor/compile/`, `kir/` | reescrita exata, lowering e cut sweep, IR portátil, kernels, liveness, linear scan |
| `lib/vapor/emit/` | x86 (AVX2), `x86_avx512.ex`, ARM, RVV, SPIR-V, link |
| `lib/vapor/runtime/` | oráculo, protocolo, worker, sessões, fabric, substratos, dispatch, `/dev/shm` |
| `lib/vapor/verify/`, `certificate.ex`, `bundle.ex`, `arbiter.ex` | escada, envelope, Ed25519, árbitro de três tetos |
| `lib/vapor/ingest/`, `model/`, `model.ex` | JSON, safetensors, GGUF/ggml, `config.json`, programa do decoder, exportação HF/GGUF |
| `lib/vapor/lock.ex`, `lock/` | a eclusa de modelos: spec, contratos, alias por dados, adaptadores (decoder, Granite, encoder/ViT, codec VQ, projetor) |
| `lib/vapor/modal/` | any-to-any: imagem, áudio, VQ, pontes, runner, texto, mundo de teste, hub |
| `lib/vapor/merge.ex`, `linalg.ex` | fusão de modelos com recibo, diagnóstico, seleção, mínimos quadrados |
| `lib/vapor/vision/`, `docs/jpeg.ex`, `docs/ccitt.ex` | OCR (geometria, ordem de leitura, leitor CTC, feixe com modelo de língua de caracteres), decodificadores JPEG e CCITT |
| `lib/vapor/modal/diffusion.ex`, `digits.ex`, `speech.ex` | difusão (DDIM, denoiser analítico), caligrafia real, fala real |
| `lib/vapor/studio.ex`, `studio/` | o estúdio: valores, nós, cache, recibos, reamostragem, exportação, ComfyUI, modelos de partida |
| `lib/vapor/diffusion/`, `lock/adapters/unet.ex` | Stable Diffusion: *schedulers*, pipeline, U-Net |
| `lib/vapor/vision/upscale.ex`, `learn.ex`, `rl.ex`, `geom.ex`, `media/` | ampliador consistente, MLPs treinadas como programa, RL, 3D, GIF/vídeo |
| `lib/vapor/mcp/server.ex` | o servidor MCP |
| `lib/vapor/substrate.ex`, `substrate/kit.ex`, `emit/msl.ex`, `export/stablehlo.ex` | a eclusa de substratos (sondas, vereditos, admissões assinadas), o kit portátil, Metal (MSL), exportação StableHLO |
| `lib/vapor/cluster.ex`, `train/lm.ex`, `streaming.ex` | orquestração (cache por conteúdo, auditoria redundante, quarentena, *hedging*), pré-treino determinístico, contexto sem fim |
| `lib/vapor/physics.ex`, `graph.ex` | física XPBD como programas, RL e gêmeos digitais; redes complexas com modelos nulos |
| `lib/vapor/vision/bidi.ex`, `cjk.ex`, `figure.ex`, `math.ex` | bidi (árabe), CJK com modelo de língua, figuras e gráficos, fórmulas → LaTeX |
| `lib/vapor/scene.ex`, `sketch.ex`, `archive.ex` | cena viva (análise, esqueleto, direção, exportação), esboço → desenho e planta 3D, arquivos verificáveis e recalculáveis |
| `lib/vapor/discover.ex`, `prove.ex` | descoberta de algoritmos com certificado; geometria algébrica, conjecturas, homologia |
| `lib/vapor/science.ex`, `science/`, `games.ex` | quântica, relatividade, tokamak, química, matéria, biologia; autojogo (política + valor + PUCT) e aleatorização de domínio |
| `lib/vapor/units.ex`, `expr.ex`, `dense.ex`, `solve.ex`, `solve/` | a bancada: unidades, expressões compiladas e derivadas, álgebra linear densa e esparsa (RCM + Cholesky em banda), EDOs (Dormand–Prince, Rosenbrock), EDPs com verificação manufaturada, sistemas, ajustes, otimização, conjunto nativo |
| `lib/vapor/engineering/` | circuitos (MNA), fluxo de potência, pórticos e treliças, MEF plano (Q4/QM6), tubulações, cinética, flash, destilação — cada um com certificado |
| `lib/vapor/logic.ex`, `logic/` | CDCL e o verificador DRUP, problemas de Ramsey, fórmulas (Tseitin), Knuth–Bendix, Gröbner; a conferência de propostas externas |
| `lib/vapor/play.ex`, `play/` | MCTS genérico, solver exato, xadrez, shogi, Go, k em linha, autojogo genérico, pôquer (CFR+) |
| `lib/vapor/bio/`, `priv/quality/protein/` | estrutura de proteínas (métricas, dobramento por contatos), coevolução (DCA), alinhamento (BLOSUM62); 1A8O, 1LCD |
| `lib/vapor/render.ex`, `priv/console/gpu_tracer.js` | traçado de caminhos: a referência em Elixir e o progressivo na GPU (WebGL2) |
| `lib/vapor/console/lab12.ex`, `priv/console/bancada.js` | os painéis da 0.12 e as suas chamadas |
| `lib/vapor/finance.ex`, `finance/` | finanças e a mesa de operações: dinheiro, calendários, curvas, opções, Monte Carlo, risco, backtests, arbitragem, livro e juiz, ITCH, FIX, pré-negociação, microestrutura, sessão de bolsa |
| `lib/vapor/logic/lp.ex` | simplex racional exato com certificados (dual, Farkas, raio) |
| `lib/vapor/console/lab13.ex`, `priv/console/mercado.js` | os painéis da 0.13 (*Mercados*) e a sua chamada |
| `lib/vapor/amalgam.ex`, `cupel.ex`, `cupel/`, `rebis.ex`, `rebis/`, `aludel.ex`, `tabula.ex` | o Opus (0.15): soma exata, corrupção silenciosa, circuitos, polinômios, contratos |
| `lib/vapor/merge/align.ex`, `docs/jbig2_huffman.ex`, `main/measure.ex` | alinhamento antes da fusão; JBIG2 Huffman; o comando externo com grupo de processos e prazo |
| `lib/vapor/console/lab15.ex`, `main/opus_cli.ex`, `priv/console/opus.js` | o Opus no console, no terminal e as suas chamadas |
| `monografia/` | a monografia (abnTeX2) com as figuras, e o roteiro da defesa oral |
| `lib/vapor/tui.ex` | o console no terminal |
| `priv/ocr`, `priv/ocr-arabic`, `priv/ocr-cyrillic`, `priv/ocr-cjk-*`, `priv/math`, `priv/lm`, `priv/games`, `priv/speech`, `priv/digits`, `priv/upscale`, `priv/rl`, `priv/quality/*` | os leitores e políticas treinados, o checkpoint SD minúsculo e os dados retidos da suíte |
| `lib/vapor/docs.ex`, `docs/` | eclusa de documentos (zip, PDF, Office, marcação, imagens) e a biblioteca |
| `lib/vapor/console.ex`, `priv/console/` | o console web (bilíngue), suas chamadas, logo, favicon, manifesto |
| `lib/vapor/quality/` | portões calibrados, métricas de texto/imagem/áudio, modelos plantados, suíte, relatório, juiz de checkpoints |
| `lib/vapor/tokenizer.ex`, `unicode.ex`, `chat.ex`, `template.ex` | tokenizador, normalização Unicode, chat templates (Jinja hermético) |
| `lib/vapor/grammar.ex`, `grammar/`, `tools.ex` | decodificação restrita: IR de bytes, JSON Schema, regex ECMA-262 e formatos, vocabulário, restrição; dialetos de chamada de ferramenta |
| `lib/vapor/agent.ex`, `agent/` | agentes: spec, diário, chaves, store, backends, MCP |
| `lib/vapor/rag.ex`, `merkle.ex`, `embed.ex` | RAG verificável, árvores RFC 6962, embeddings |
| `lib/vapor/cr.ex`, `canonical.ex` | funções elementares corretamente arredondadas, CBOR canônico |
| `lib/vapor/field.ex`, `zk.ex`, `poly.ex` | corpos primos e NTT, R1CS de inferência inteira, aproximações polinomiais certificadas |
| `integrations/`, `notebooks/` | Plug/Phoenix, Nx; Livebook (testado como código) |
| `lib/vapor/sampler.ex`, `engine.ex`, `engine/pool.ex`, `speculative.ex`, `serve.ex` | geração, motor, réplicas, especulação, servidor |
| `lib/vapor/autodiff.ex`, `train.ex` | gradientes e LoRA/destilação |
| `lib/vapor/bench.ex`, `lib/mix/tasks/` | aparato de medição, CLI |
| `native/src/` | worker (seccomp no Linux, Capsicum no FreeBSD, `MAP_JIT` no macOS; pool, contadores, interpretador RVV), daemons Vulkan e Metal (e o simulador sobre o *shim*), em Zig |
| `proofs/` | Lean 4 + extrator |
| `test/`, `test/python/` | testes por nível e scripts dos oráculos diferenciais |
| `slides/`, `flake.nix`, `scripts/` | apresentação, Nix, e2e; `scripts/pack.py` empacota os três `.zip` (código, qualidade, modelos) de forma reprodutível, com `SHA256SUMS` |

## Limitações

Não há aqui hardware RVV, ARM ou GPU discreta: RVV/NEON foram validados sob
QEMU e Vulkan sob lavapipe; o cooperative matrix é emitido e validado, não
executado. A VM tem 2 vCPUs e nenhuma PMU, então escala a muitos núcleos e
contadores de hardware não foram medidos. O worker é Linux-only (até 0.9; a 0.10 o compila para macOS e FreeBSD). O GEMV de
4 bits segue limitado por emissão (o MoE em 4 bits já é esparso, desde 0.8.0). Decisões de modelos
hospedados são registradas, não reproduzidas. Das novidades da 0.5.0:
os leitores treinados (OCR, fala, caligrafia) são **pequenos e honestos sobre
o seu domínio** — o OCR lê texto impresso horizontal (em colunas e com modelo
de língua desde 0.7.0; sem tabelas célula a célula, sem manuscrito), a fala reconhece dígitos
falados, a difusão desenha dígitos 8×8; cada um vem com a sua medida em dados
retidos e com o controle que ele precisa bater. A busca de imagem acha
imagens pelo texto que está nelas; as duas torres do CLIP estão conferidas,
mas a busca pelo que a imagem mostra pede pesos CLIP treinados (não
embarcados). Da 0.6.0: VAE, DiT, Whisper e Mamba estão conferidos contra as
referências **com pesos aleatórios** — nenhuma afirmação de qualidade de
imagem, vídeo ou transcrição é feita (o Mamba-2 da 0.8.0 também); não há
worker para macOS. Da 0.8.0: o motor serve na GPU com sessões residentes,
mas **só no lavapipe** — protocolo, residência e bits provados, velocidade
numa GPU real não medida; as tabelas precisam de réguas ou filetes, e o
Tesseract lê melhor as células (a estrutura, que ele não dá, é exata nas 12
do teste); JBIG2 Huffman e meio-tom e JPX são recusados com aviso; o
paralelismo entre nós BEAM é exato para as projeções e o MLP, não ainda para
a atenção por cabeças; um dossiê de auditoria autentica evidência e **não é
avaliação de conformidade**. Árabe, CJK, cursivo, fórmulas, texto → vídeo,
Apple Silicon, Postgres e Nerves foram pedidos e recusados com o motivo
([DIRETRIZ.md §11](docs/DIRETRIZ.md)) — árabe, CJK, fórmulas e Apple
entraram na 0.10, com as medidas abaixo. Da 0.9.0: o Stable Diffusion está conferido contra o diffusers **com pesos aleatórios minúsculos** — qualidade e velocidade de um SD real não foram medidas aqui; um KSampler importado do ComfyUI dá a imagem do diffusers, não os pixels do ComfyUI; o ampliador ganha em texto e empata em fotografias, e não inventa detalhe; não há H.264/MP4/WebM; remoção de marca d'água e de recusas foram pedidas e recusadas pelo nome ([DIRETRIZ.md §12](docs/DIRETRIZ.md)). O RegMean resolve `O(d³)` na BEAM e
com os modelos em memória (os outros métodos fundem do disco para o disco). Da 0.10.0: o daemon Metal real e
o worker FreeBSD são **compilados e conferidos, nunca executados** (não há
Mac nem FreeBSD aqui), e a Tenstorrent é alcançada por exportação e pelo kit
de admissão, rodado só no XLA de CPU; os leitores de árabe, cirílico e CJK
foram treinados em fontes **sintéticas** e medidos em fontes que não viram —
manuscrito (o nastaliq do árabe, a letra de mão do japonês e do russo) só por
fontes caligráficas como substituto, com erros bem maiores, e o cursivo
latino segue recusado (64 % de CER em mãos não vistas); o digitalizador
de gráficos lê linhas, barras e dispersão com eixos em L e recusa o resto
(no conjunto difícil, 12 de 30 recusados, nenhum erro grosseiro); as
fórmulas são impressas e de uma linha (sem matrizes, sem manuscrito); a física é de
partículas e hastes (sem corpos rígidos, sem atrito, sem colisões entre
corpos); as redes são para milhares de nós, não milhões; o contexto sem fim
foi medido num modelo de meio milhão de parâmetros. Da 0.11.0: nenhuma afirmação de
qualidade generativa (o esboço → fotorrealista precisa de pesos que não
vêm junto); a profundidade da cena viva é **heurística** (o plano do chão)
e a navegação é 2,5D numa janela pequena; os habitantes são figuras do
motor, e o movimento de um desenho vem da topologia do esqueleto, não do
que ele mostra; DFT de baterias e relatividade geral foram
recusados pelo nome; a geometria cobre igualdades com construções
explícitas e dá certificados algébricos, não provas legíveis; a ciência roda em binary64 na BEAM (determinística),
ainda não como programas do compilador. Da 0.12.0: a predição de
estrutura a partir só da sequência continua fora de alcance (o pipeline
de proteínas usa um alinhamento amostrado de um modelo plantado e a
estrutura secundária nativa); os motores de xadrez, shogi e Go são
didáticos, ordens de grandeza abaixo dos abertos de ponta; o autojogo
genérico perde 21 % das linhas ótimas com 8 simulações, contra 13 % do
especializado de 0.11; o render não tem MIS (cáusticas de luzes pequenas
são ruidosas) nem malhas; as ferramentas de engenharia são lineares ou
estacionárias, salvo circuitos e cinética, e não substituem normas; a
conferência DRUP de R(3, 4) leva ~2 minutos. Da 0.13.0: o motor de
ofertas roda na BEAM — microssegundos por evento com o hash, não os
nanossegundos de uma bolsa (o que se afirma é a verificabilidade e a
identidade entre backtest e motor); nenhum dado de mercado real foi
baixado (os oráculos são o QuantLib, o simplefix, o SciPy e o ngspice, e os
preços dos exemplos são ilustrativos); o gerador do Monte Carlo no worker
é o Wichmann–Hill, antigo (há o caminho splitmix na BEAM); não há XVA,
crédito, modelos de taxa de vários fatores nem volatilidade calibrada a uma
superfície; passar nos portões de ruído não torna uma estratégia
lucrativa. A lista completa está em
[docs/ARCHITECTURE.md §10](docs/ARCHITECTURE.md), [docs/AGENTES.md §7](docs/AGENTES.md)
e [docs/TODO.md](docs/TODO.md).

Licença ISC.
