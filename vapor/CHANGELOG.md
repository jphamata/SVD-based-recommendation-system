# Mudanças

## 0.14.0 — 2026-10-05

Escrutínio do sexto pedido ("de expositivo a ferramenta real, entrada aberta e saneada, sem categorias pré-definidas, humano e modelo no laço, pesquisa em IA, tudo pelo terminal, cenas livres"; e "nomes de alquimia em inglês, nada de nomes consagrados, foco em ciência, computação, matemática, IA e finanças, uma interface mais marcante"): [docs/DIRETRIZ.md §17](docs/DIRETRIZ.md). A tese: **uma linguagem, uma fornalha, uma pedra de toque** — os buscadores famosos são casos de *proponha, avalie, certifique*, então o produto é o caso geral, e os casos viram exemplos. Verificações com controle em `mix vapor.quality` (§5j, 20; `--only round14`).

**Alembic** ([ALEMBIC.md](docs/ALEMBIC.md))
- Linguagem pura de problemas: inteiros arbitrários, listas, tuplas, mapas, compreensões, lambdas, *pipes*, `let`, ~120 funções; erros com linha, coluna e sugestão.
- Saneada: combustível, recursão ≤ 5 000, inteiros ≤ 65 536 bits, listas ≤ 2 M, `sandbox/2` com teto de memória e tempo, identificadores nunca viram átomos, `literal/1` lê só dados.
- `Alembic.Tree`: o subconjunto numérico como árvore JSON interpretada no navegador; `noise()` bit a bit igual em Elixir e JS.

**Athanor e Touchstone** ([ATHANOR.md](docs/ATHANOR.md))
- Dez espaços (bits, ints, reals, perm, subset, subsets, seq, graph, partition, program); minimizar, maximizar ou refutar uma afirmação; `violation`, `holdout`, `describe`, `neighbor`, `measured`.
- Portfólio sob UCB descontado: exaustiva retomável, aleatória, recozimento, evolução/MAP-Elites, CMA-ES, bayesiana (GP Matérn-5/2, EI, lotes), mente, humano.
- Controle aleatório com o mesmo orçamento (regra de três), *holdout* com Spearman e máximo-z do ruído, diário SHA-256; `Touchstone.verify/3` com `full:` e `replay:`.
- Sessões supervisionadas (propor, fixar, banir, estender, medir); jogos de dois jogadores (negamax, UCT, aprendizado por autojogo, partida com Wilson).
- `Vapor.Mind`: `formalize` (até três reparos com o erro do compilador e retrotradução), `propose`, `ask`; Anthropic, compatível com OpenAI ou roteiro gravado.

**Crucible** ([CRUCIBLE.md](docs/CRUCIBLE.md)) — dez domínios abertos com evidência sem gabarito: leis de conservação provadas sobre ℚ (com `ln`), quântica (Sturm, split-step), hamiltonianos simpléticos, reações, Wright–Fisher, filogenia com *bootstrap*, RHF/STO-3G, dobramento HP, Boris, regressão simbólica.

**Assay** ([ASSAY.md](docs/ASSAY.md)) — compare, leaderboard, calibration, agreement, judge, contamination, dedup, scaling, cada um com o seu controle.

**Interfaces**
- `bin/vapor` / `mix vapor` ([CLI.md](docs/CLI.md)): JSON em *pipes*, `NO_COLOR`, códigos de saída 0–4; `--measure` liga um programa externo como objetivo.
- Console: grupo "Bancada aberta" (Espaço de trabalho, Crisol, Ensaio) como abertura; a fornalha ao vivo e a pedra de toque; nova identidade (fuligem, pergaminho, latão, verdete, cinábrio), marca da alambique, ícones e manifesto novos; o painel Ciência renomeado Calibração.
- Cenas como documentos editados por operações de texto (`Vapor.Scene.Ops`), com expressões por quadro e direção por modelo; `vapor scene new|edit|direct|export`.
- MCP: `alembic_eval`, `athanor_run` (com `proposals`), `athanor_verify`, `game_query`, `crucible_run`, `assay_run`, `scene_ops` (20 ferramentas); TUI com os mesmos verbos.

**Corrigido no caminho**
- `Assay.Scaling` estourava `exp` em dados sem estrutura (achado pelo controle de perdas embaralhadas).
- `Vapor.Expr.compile/2` criava um módulo e um átomo por expressão nova sem limite; agora há teto e o excedente é interpretado.
- `mix vapor.serve` não exige mais `--model`/`--docs` (a bancada funciona sem modelo).
- `Finance.MonteCarlo` sem o processo nativo: o oráculo não carregava o estado do gerador (w1, w2, w3) entre chamadas.
- `vapor assay` sai 1 quando uma verificação falha (ruído, líder instável, juiz enviesado), como os outros comandos.

**Documentos**: slides (42, com quatro novos sobre a bancada aberta), monografia (capítulo 6 novo, tabela das 20 verificações, resumo e conclusão), roteiro da defesa, DIRETRIZ §17, README, TODO, CENA §9, CONSOLE, INTERFACES, CIENCIA.

## 0.13.0 — 2026-10-05

Escrutínio do quinto pedido ("suporte a finanças, HFT, ataque as limitações e o TODO, slides, monografia em LaTeX e roteiro de defesa"): [docs/DIRETRIZ.md §16](docs/DIRETRIZ.md). A tese da rodada: **a dor de finanças é verificabilidade, não velocidade** — cada número do mercado sai com o objeto que permite julgá-lo. Verificações com controle em `mix vapor.quality` (§5i, 23; `--only round13` roda só a rodada). Documento: [FINANCAS.md](docs/FINANCAS.md).

**Finanças** ([FINANCAS.md](docs/FINANCAS.md))
- `Vapor.Finance.Money`: decimal exato (inteiro + escala), sete modos de arredondamento por uma única primitiva, rateio pelo maior resto que soma exatamente, fator (1 + r)^(du/252) por raiz inteira e truncado como a ANBIMA.
- `Vapor.Finance.Calendar`: ANBIMA/B3, NYSE (com fechamentos especiais como dados) e TARGET por regras e pelo cômputo; **iguais ao QuantLib dia a dia de 1990 a 2078**; DU/252, ACT/360, ACT/365F, 30/360, 30E/360, ACT/ACT ISDA = QuantLib a 10⁻¹⁴; ajustes e vencimentos de DI1.
- `Vapor.Finance.Curve`: *bootstrap* de DI1, LTN, NTN-F, depósitos, títulos (preço limpo) e swaps par; flat-forward ou zero linear; certificado de reprecificação e forwards negativos apontados; Nelson–Siegel–Svensson. Depósitos = `PiecewiseLogLinearDiscount` do QuantLib a 10⁻¹³.
- `Vapor.Finance.Options`: BSM e gregas (contra diferenças finitas), Black-76, Bachelier, volatilidade implícita com os **limites de não arbitragem conferidos antes**, CRR e Leisen–Reimer (americana = QuantLib a 10⁻¹⁰), Heston por Lewis (= QuantLib a 10⁻⁹), SVI com g(k) de Durrleman e calendário, arbitragem estática sem modelo com o portfólio que a explora.
- `Vapor.Finance.MonteCarlo`: GBM **compilado para o worker** com o gerador (Wichmann–Hill, exato em binary32) e Φ⁻¹ (AS241) dentro do programa; europeia, asiáticas (geométrica de Kemna–Vorst como variável de controle), barreira; bits = oráculo (programa de 64 pistas) e = 2 threads; 6–23× a BEAM; o controle sem o termo de Itô pego (z = 8,7). Longstaff–Schwartz na BEAM.
- `Vapor.Finance.Risk`: VaR/ES histórico, normal, Cornish–Fisher, EWMA; Kupiec, Christoffersen, Basileia; o tamanho do teste medido; Ledoit–Wolf, variância mínima com KKT, paridade de risco (Spinu), HRP.
- `Vapor.Finance.Backtest`: linguagem de sinais causal e **quatro portões de ruído** — invariância de prefixo (o certificado de ausência de antecipação, com o dia), Sharpe deflacionado, PBO por CSCV, Reality Check com bootstrap estacionário.
- `Vapor.Finance.Arbitrage`: o teorema fundamental como lema de Farkas — o portfólio de arbitragem **ou** os preços de estado, em racionais; calls entre strikes (decisão completa para portfólios estáticos), câmbio; `check/2` confere a proposta de qualquer um.

**Mesa de operações**
- `Vapor.Finance.Book`: preço–tempo; limitada e a mercado; GTC/IOC/FOK; post-only; alteração com e sem perda de prioridade; prevenção de autonegociação; *kill switch*; **diário SHA-256 + raiz de Merkle** e provas de inclusão.
- `Vapor.Finance.Book.Check`: o juiz ingênuo independente — refaz a cadeia, reexecuta e exige os mesmos relatórios, confere nove invariantes; *fuzzing* diferencial com 6 000 eventos por política.
- `Vapor.Finance.Itch` (ITCH 5.0: dez tipos de mensagem, BinaryFILE, feed da sessão, livro reconstruído = livro do motor) e `Vapor.Finance.Fix` (FIX 4.4 com BodyLength e CheckSum = simplefix; D/F/G → eventos; relatórios → ExecutionReports).
- `Vapor.Finance.PreTrade`: quantidade, nocional, colar, posição no pior caso, taxa de mensagens, *kill switch* (15c3-5 / RTS 6); recusas no diário com a causa; `check/2` refaz tudo do diário.
- `Vapor.Finance.Micro`: Hawkes (Ogata, máxima verossimilhança, teste de reescala do tempo; o Poisson como controle), Avellaneda–Stoikov (o §4 do artigo reproduzido), Almgren–Chriss (forma fechada = ótimo numérico a 10⁻¹⁶), Roll, Kyle, assinatura da variância, *microprice*.
- `Vapor.Finance.Exchange`: formadores, agressores Hawkes em tempo contínuo e um informado pelo mesmo portão e pelo mesmo motor; a sessão devolve os três certificados e se repete na mesma cabeça de hash.

**Pendências fechadas** (do [TODO](docs/TODO.md))
- `Vapor.Logic.LP`: simplex racional exato (duas fases, Bland) com certificados de otimalidade (dual, gap zero), inviabilidade (Farkas) e ilimitação (raio), conferidos por `LP.check/2`; na mesa de lógica (`maximize …`) e no `logic_check` com proposta. = HiGHS do SciPy a 10⁻⁹.
- `Vapor.Archive`: **arquivos assinados** (Ed25519 sobre o manifesto, a chave do operador de `mix vapor.audit keygen`; a identidade não muda); `verify(zip, trusted: …)` recusa o não assinado, a chave desconhecida e a assinatura que não confere; o console assina com `VAPOR_ARCHIVE_KEY`; `mix vapor.archive sign|verify|replay`; os tipos `finance.*` são recalculáveis.
- `Vapor.Units`/`Vapor.Expr`: **escalas afins** °C/°F como leituras (98,6 °F = 37 °C; pV = nRT a 20 °C), recusadas dentro de unidades compostas com a sugestão de `degC`.
- `Vapor.Engineering.Circuit`: **MOSFET nível 1** e **bipolar de Ebers–Moll** (dois ramos; KCL certificado sem mudança), GMIN; **iguais ao ngspice 42** na 6ª casa (ponto de operação e ganho CA).

**Console**: grupo *Mercados* com *Finanças* (oito tarefas) e *Mesa de operações* (três), selo, portões, semáforo de Basileia, escada do livro, cadeia do diário; `test/js/console_markets.mjs` roda os 40 exemplos em EN e PT (87 conferências) e **exige que nenhum rótulo ou frase em inglês sobre na versão em português** — um passo único de tradução depois de cada visualização cobre os rótulos e as frases do servidor (vereditos, portões, recusas), sem tocar no código, nos *hashes* nem nos bytes de FIX/ITCH. Endpoint `POST /v1/vapor/finance`.

**MCP**: `finance_run` e `arbitrage_check` (treze ferramentas). **CLI**: `mix vapor.finance KIND ARQUIVO`, `mix vapor.archive`, `mix vapor.quality --only roundNN`.

**Defesa**: `slides/vapor.tex` atualizado (seção 7 com oito slides da rodada; tamanhos e linhas do núcleo remedidos — 1,8 MB e 19 085 linhas no núcleo, 8,7 MB com as treze rodadas —, 82 teoremas, sobreposições de rótulos corrigidas); **monografia** em abnTeX2 (`monografia/`) e **roteiro da defesa oral** (`monografia/DEFESA.md`).

**Medido**: 108 testes, 0 falhas nos oito arquivos tocados (a execução completa foi interrompida por um reinício da VM e não repetida); `mix vapor.quality --only round13` 23/23 em 23 s; console 87 conferências em EN e PT.

**Correções desta rodada** ([DIRETRIZ.md §16](docs/DIRETRIZ.md)): FOK a mercado virava IOC (achado pelo invariante, nos dois motores); fechamentos e regras do NYSE/TARGET; Monte Carlo 95 s → 0,1 s (um passo por chamada, ISA da máquina, paridade em 64 pistas, gerador no worker); o simulador de bolsa distribuía chegadas por passo (o Hawkes reprovado corretamente) e cotava em torno do próprio meio; erro interno do compilador do OTP 25 contornado.

## 0.12.0 — 2026-10-05

Escrutínio do quarto pedido ("problemas arbitrários e não apenas categorias pré-definidas", as quatro engenharias e as ciências, HPC, "similar ou superior" com comparações, humano ou IA no loop, NPCs, foto-realismo, xadrez/shogi/Go/cartas, interface profissional em ordem alfabética): [docs/DIRETRIZ.md §15](docs/DIRETRIZ.md). A regra que atravessa a rodada: **a entrada é o texto do domínio, e cada resposta traz um certificado calculado fora do solver**. Verificações com controle em `mix vapor.quality` (§5h, 27). Documentos: [BANCADA.md](docs/BANCADA.md), [ENGENHARIA.md](docs/ENGENHARIA.md), [LOGICA.md](docs/LOGICA.md), [TABULEIROS.md](docs/TABULEIROS.md), [PROTEINAS.md](docs/PROTEINAS.md), [RENDER.md](docs/RENDER.md), [CENA.md §6.1](docs/CENA.md).

**Bancada** ([BANCADA.md](docs/BANCADA.md))
- `Vapor.Units`, `Vapor.Expr`: grandezas com unidade (7 dimensões do SI, unidades com nome, prefixos, `in [unidade]`), parser com coluna do erro, derivada simbólica, simplificação, LaTeX, **compilação para módulos BEAM** (cache por sha-256); planilhas linha a linha.
- `Vapor.Solve`: EDOs por Dormand–Prince 5(4) com saída densa, troca automática para Rosenbrock (Jacobiano simbólico) na rigidez, eventos por bisseção, saídas derivadas, unidades conferidas antes de integrar; EDPs parabólicas (Crank–Nicolson + AB2), hiperbólicas (leapfrog, CFL recusado) e Poisson 2-D (CG), com **verificação por solução manufaturada** (ordem observada); sistemas com todas as raízes numa caixa; ajustes (Levenberg–Marquardt, erros-padrão, AIC); otimização (Lagrangiano aumentado + BFGS **projetado**: limites de caixa exatos, divergência relatada, veredito KKT).
- `Vapor.Solve.Ensemble`: incerteza (`k ~ normal(…)`) compilada para o worker nativo — 4096 membros × 1000 passos em 236 ms, 53× a BEAM, paridade bit a bit com o oráculo.
- `Vapor.Dense`: pivotamento parcial, Jacobi, Cholesky, autovalores generalizados, CG, Householder, Cuthill–McKee reverso + Cholesky em banda. `bancada_test.exs` (20; SciPy como oráculo).

**Engenharia** ([ENGENHARIA.md](docs/ENGENHARIA.md))
- `Vapor.Engineering.Circuit` (MNA; .op/.dc/.ac/.tran; diodo, fontes controladas, amp-op ideal; certificado de Kirchhoff e de potência; nós sem caminho DC nomeados), `Power` (Newton polar; Gauss–Seidel de controle; Stagg & El-Abiad), `Structure` (pórticos e treliças, massa consistente, modos, mecanismo recusado), `FEM` (Q4 e QM6, patch test), `Pipes` (gradiente global, Colebrook, malhas), `Process` (cinética com invariantes por espaço nulo racional, CSTR, Rachford–Rice, McCabe–Thiele/Fenske/Underwood/Gilliland). `engenharia_test.exs` (21).

**Lógica** ([LOGICA.md](docs/LOGICA.md))
- `Vapor.Logic.SAT` (CDCL) e `Vapor.Logic.DRUP` (verificador independente), problemas de Schur/van der Waerden/Ramsey/casa dos pombos/rainhas com testemunha e refutação, fórmulas por Tseitin, `Rewrite` (Knuth–Bendix com LPO), `Groebner` (Buchberger, Rabinowitsch). **`Vapor.Logic.check/2`**: a proposta de qualquer um (modelo, prova DRUP, coloração, contraexemplo) conferida, nunca confiada. `logica_test.exs` (8).

**Tabuleiros e cartas** ([TABULEIROS.md](docs/TABULEIROS.md))
- `Vapor.Play`: MCTS genérico (UCT/PUCT, ruído na raiz), negamax exato com tabela; `Chess` (perft publicado, python-chess, motor, **provas de mate conferidas**), `Shogi` (lançamentos, nifu, uchifuzume; perft 30/900/25 470; python-shogi), `Go` (Tromp–Taylor, superko; 1/57/12 675 posições legais), `MNK`, `Poker` (CFR+, explorabilidade exata em Kuhn e Leduc), `SelfPlay` (autojogo genérico julgado contra o jogo perfeito). `tabuleiros_test.exs` (16).

**Proteínas** ([PROTEINAS.md](docs/PROTEINAS.md))
- `Vapor.Bio.Structure` (PDB, Horn, TM-score = TM-align, GDT, lDDT, estrutura secundária, **dobramento por geometria de distâncias** com quiralidade pelas hélices), `Coevolution` (Potts, MI/APC, DCA), `Align` (Gotoh, BLOSUM62 = Biopython). Pipeline em 1A8O: TM 0,69 (controle 0,20). `proteinas_test.exs` (7).

**Render** ([RENDER.md](docs/RENDER.md))
- `Vapor.Render`: traçado de caminhos (difuso, metal, vidro, emissivo, céu, sol por evento seguinte, roleta russa), determinístico, paralelo; fornalha branca, fornalha em gradiente (o estimador viciado pego), N^−½. `priv/console/gpu_tracer.js`: o mesmo na GPU (WebGL2), progressivo, conferido no Chromium contra a referência. `render_test.exs` (6).

**Cena viva** ([CENA.md §6.1](docs/CENA.md))
- Habitantes com id, nome, cor, comportamento (passear, ficar, patrulhar, seguir, fugir, ir), ações (acenar, dançar, sentar, pular, correr), falas em balão, rotas por clique, linha do tempo (`at`), **GIF de quadros exatos**; direção por orações com tempo, pronomes e papéis (PT/EN).

**Console** ([CONSOLE.md](docs/CONSOLE.md))
- Navegação regrupada (*Resolver* novo) e **em ordem alfabética no idioma mostrado**; **paleta de comandos** (Ctrl/⌘ K); seis painéis novos — Bancada, Engenharia, Lógica, Tabuleiros e cartas, Proteínas, Render — com diagramas próprios; inspetor de habitantes; os scripts embutidos na página (um documento único). `console_desks_test.exs` + `test/js/console_desks.mjs`: todo exemplo de todo painel pelo Chromium.

**MCP**: `workbench_solve`, `engineering_run`, `logic_check` (com proposta), `board_query`, `render_scene` (onze ferramentas).

**Nomes**: identificadores, painéis e API sem nomes proprietários (`games.selfplay`, tarefa `selfplay`; os nomes de 0.11 seguem aceitos para os arquivos salvos); as comparações ficam nos documentos.

**Medido**: `mix test` 702 testes, 0 falhas (115 excluídos por ferramenta ausente); `mix vapor.quality` 147/147 (nativo).

**Correções desta rodada** ([DIRETRIZ.md §15](docs/DIRETRIZ.md)): otimização sem limites devolvia −6·10⁶² como ótimo (agora: limites projetados, divergência relatada); o leitor de cenas aceitava valores não numéricos; a direção em inglês perdia nomes, pronomes e tempos; colisão de CSS encolhia peças de xadrez; página com scripts externos.

## 0.11.0 — 2026-10-04

Escrutínio do terceiro pedido do dia (AlphaProof/AlphaDev/AlphaFold/AlphaZero "e além", física em várias frentes, dar vida a uma imagem, salvar e exportar tudo): [docs/DIRETRIZ.md §14](docs/DIRETRIZ.md). A regra que atravessa a rodada: **uma busca propõe, um verificador decide**. Cada entrega tem uma verificação com controle em `mix vapor.quality` (§5g, 26 verificações). Documentos: [CENA.md](docs/CENA.md), [DESCOBERTA.md](docs/DESCOBERTA.md), [MATEMATICA.md](docs/MATEMATICA.md), [CIENCIA.md](docs/CIENCIA.md), [JOGOS.md](docs/JOGOS.md).

**Dar vida a uma imagem** ([docs/CENA.md](docs/CENA.md))
- `Vapor.Scene`: SLIC + grafo de regiões, céu crescido, profundidade pelo plano do chão (heurística, editável), camadas com o fundo reconstruído por push-pull, chão caminhável, luz; esqueleto de desenhos (Zhang–Suen, grafo por *crossing number*, laços), malha com skinning; direção PT/EN → operações, palavras desconhecidas relatadas; página HTML autônoma. `scene_test.exs`.
- O motor `SceneEngine` (no console e no HTML exportado): camadas como planos em perspectiva, chão em faixas, habitantes por A* com a cabeça no horizonte, chuva/neve/neblina/tempestade, tochas, velas, brasas, fumaça, vaga-lumes, pássaros, borboletas, folhas, vento na vegetação, hora do dia e ciclo, desenhos animados (acenar, andar, dançar, respirar), entropia, semente e passo fixo (o laço é reproduzível), gravação em vídeo.
- `Vapor.Sketch`: esboço → desenho técnico com restrições (SVG, DXF R12); planta → cômodos, portas e modelo 3D (GLB). `sketch_test.exs`.

**Descobrir** ([DESCOBERTA.md](docs/DESCOBERTA.md), [MATEMATICA.md](docs/MATEMATICA.md))
- `Vapor.Discover`: redes de ordenação ótimas para n ≤ 8 (princípio 0-1), multiplicação 2×2 com 7 produtos (exata nos inteiros; posto 6 nunca), truques de bits mínimos por exaustão (⌊(x+y)/2⌋ = `(x&y)+((x^y)>>1)`), classe de complexidade de contagens. `discover_test.exs`.
- `Vapor.Prove`: geometria pelo método algébrico (numerador ≡ 0 + não degenerescência + conferência em racionais exatos; falsos refutados), conjecturas achadas e provadas (reta de Euler, círculo dos nove pontos), homologia sobre GF(2) e ℚ (torção de Klein e RP²), homologia persistente. `prove_test.exs`.

**Ciência e jogos** ([CIENCIA.md](docs/CIENCIA.md), [JOGOS.md](docs/JOGOS.md))
- `Vapor.Science`: Schrödinger por split-step (estado coerente, tunelamento exato), Boris relativístico (γ, E×B), Grad–Shafranov contra Solov'ev, Hartree–Fock STO-3G (H₂ −1,1167; HeH⁺ −2,860662; o fracasso do RHF mostrado), Lennard-Jones, Wright–Fisher contra a cadeia exata, filogenia (RF 0), dobramento HP até −9. `science_test.exs`.
- `Vapor.Games`: AlphaZero de jogo da velha (política + valor + PUCT + autojogo; `priv/games/tictactoe.json`) — contra **todas** as linhas ótimas do jogo perfeito, nenhuma perdida com 128 simulações e 13 % com 8 (a busca sem treino: 97 %); aleatorização de domínio no carro-pêndulo (484 contra 275 passos em mundos não vistos). `games_test.exs`.

**Salvar e exportar, para tudo**
- `Vapor.Archive`: zip com manifesto (receita, hashes), identidade pelo hash do manifesto, conferência byte a byte, **recálculo** dos tipos determinísticos; um arquivo nomeia um tipo, nunca uma função. `archive_test.exs`. Todo painel novo do console tem *Salvar*; *Confiar → Arquivos* confere e recalcula.

**Console**: painéis *Cena viva*, *Esboço*, *Ciência*, *Jogos*, *Matemática*, *Algoritmos*, *Arquivos* (inglês e português, claro e escuro). Endpoints `/v1/vapor/{scene/*,sketch,prove,discover,science,games,games/move,archive,archive/check}`.

**Correções da revisão independente da 0.11** ([DIRETRIZ.md §14](docs/DIRETRIZ.md))
- `Vapor.Archive`: descompressão contada enquanto acontece (no máximo 512 entradas, 256 MB; uma bomba de 1 MB que expande a 300 MB é recusada sem ser expandida), manifesto malformado, sem `result.json` ou com entradas duplicadas recusado (antes, quebrava); todo parâmetro de receita limitado antes de rodar (`n` 2–10, `beam` 1–256, autojogo ≤ 400 partidas e 64 simulações, teoremas, complexos e experimentos só pelo nome); a busca em feixe que morre é relatada, não repetida para sempre.
- `Vapor.Prove`: uma construção degenerada para todo valor é `{:degenerate, :construction}`, nunca "provada" por 0/0; os sobreviventes das conjecturas são provados simbolicamente (antes, só conferidos em racionais); a "conferência independente" é chamada pelo que é.
- `Vapor.Games.versus_every_optimal_line/2`: todas as linhas ótimas do jogo perfeito, dos dois lados. Achou 4 linhas perdidas de 131 com 64 simulações que a amostra de 60 partidas não via; a afirmação, o teste, o console e a §5g passaram a esta medida.
- O laboratório não cai com uma imagem lida mas não decodificada (mais de 4 megapixels, GIF); o tabuleiro do jogo é validado.
- Documentos: controles fracos ditos como tais (CIENCIA §1), a taxa h² do tokamak atribuída ao localizador do eixo, a classe n^2,807 como conferência do classificador, o carro-pêndulo do canto, a massa do carro ×0,5–2, a §5g das redes de ordenação exige que todo comparador seja necessário.

**Correções da revisão independente da 0.10**
- A eclusa de substratos: uma diferença sem limite e sem causa medida é recusada; resposta fora do protocolo ou sem uma saída é recusada (antes, quebrava); sondas não rodadas aparecem como `:unmeasured`; `judge --sign` usa a chave do operador.
- `/v1/vapor/ocr` não dá mais 500 em `{:error, _}` genérico; o cluster tira do cache as respostas de um nó posto em quarentena; os laboratórios do console serializam a primeira execução (uma sessão por worker) e fecham a sessão do gêmeo.
- Documentos: z ≈ 400 (não 300) para Watts–Strogatz; o comando do kit; referências a testes inexistentes; o livro do gêmeo não protege medidas sem âncora externa (dito); a ausência de seccomp no FreeBSD não é visível no binário (dito); a linha de base Witten–Bell da §5f agora nos mesmos bytes; o CJK exige que o modelo de língua ajude em japonês e coreano.

## 0.10.0 — 2026-10-04

Escrutínio dos dois pedidos desta rodada (substratos, treino, OCR de outras escritas; e, no meio, contexto sem fim, física, redes, cirílico, LaTeX, FreeBSD): [docs/DIRETRIZ.md §13](docs/DIRETRIZ.md). Cada entrega tem uma verificação com controle em `mix vapor.quality` (§5f). Documentos da rodada: [SUBSTRATOS.md](docs/SUBSTRATOS.md), [TREINO.md](docs/TREINO.md), [FISICA.md](docs/FISICA.md), [REDES.md](docs/REDES.md), [OCR.md §3g–§3k](docs/OCR.md).

**Substratos** ([docs/SUBSTRATOS.md](docs/SUBSTRATOS.md))
- **Eclusa de substratos** (`Vapor.Substrate`, `mix vapor.substrate list|kit|judge`): sondas de resposta conhecida (FMA, FTZ, DAZ, zero com sinal, NaN, ordem de redução, bits reais de mantissa, divisão, funções, os núcleos de verdade) → veredito `:canonical | :envelope | :refused` com a **impressão numérica**; registro CBOR assinado (Ed25519); o despachante só manda programas canônicos para substratos canônicos; admissão na chegada. `substrate_test.exs`.
- **Envelope com DAZ**: achado pelo kit no XLA de CPU (subnormais de entrada lidos como zero); coberto, e continua apertado.
- **Metal**: tradutor MSL (`Vapor.Emit.MSL`, controle de fluxo estruturado), daemon `vapor-metal` (runtime Objective-C aberto em tempo de execução, `MTLMathModeSafe`, memória unificada), worker de CPU para macOS (`MAP_JIT`, `__ulock`); `make metal` de qualquer host. O mesmo MSL executado por um *shim* com clang (`vapor-metal-sim`): = oráculo bit a bit em programas canônicos, SSM, GEMM, atenção, sessão Llama e o motor; *shim* que contrai e *shim* FTZ admitidos no envelope. O daemon real é compilado, não executado. `metal_test.exs`.
- **Tenstorrent e qualquer PJRT**: exportação **StableHLO** (`Vapor.Export.StableHLO`, `mix vapor.export`) — Llama/Qwen2/Mistral no XLA a ~10⁻⁶ —, e um **kit de admissão portátil** (StableHLO + `run_kit.py --platform tt|tpu|cpu`) julgado e assinado de volta. `stablehlo_test.exs`.
- **Cluster** (`Vapor.Cluster`): cache por conteúdo, auditoria por execução redundante com amostra por *hash* com chave, quarentena e readmissão por medida, *failover* e *hedging* sem mudar um bit, treino entre nós com os bits de uma máquina. Nós `:peer` reais. `cluster_test.exs`.
- **FreeBSD**: o worker compila para x86-64 e AArch64 (`make freebsd`), libc, `_umtx_op`, W^X por `mprotect`, isolamento **Capsicum** (um descritor de diretório com direitos de leitura e mapeamento; `openat`). Não executado. `freebsd_test.exs`.

**Treino e contexto** ([docs/TREINO.md](docs/TREINO.md))
- **Pré-treino** (`Vapor.Train.LM`, `mix vapor.train`): Llama de bytes, gradientes por `Autodiff.grad_lets/5` (= PyTorch a 7,7·10⁻⁷), paralelismo de dados determinístico (árvore fixa sobre blocos acumulados na sessão residente: bits independentes do número de workers e de um worker morrer), AdamW com *clipping*, checkpoint/retomada exatos, exportação Hugging Face. **`priv/lm`**: 492 160 parâmetros, 2,919 bits/byte em texto retido (Witten–Bell ordem 5: 3,415), com recibo. `train_lm_test.exs`.
- **Contexto sem fim** (`Vapor.Streaming`, `kv: {:stream, âncoras, janela}`): chaves sem rotação num anel com âncoras fixas, giradas a cada passo no referencial do cache — memória constante, nenhuma distância fora do treino, os bits do modelo causal até o cache encher. 3,03 bits/byte 14× além do comprimento de treino; posições crescendo: 5,81. `streaming_test.exs`.
- **Autodiff**: regras para `max`, `min`, `relu`, `sel` (por partes), `tanh`, `fma`.

**Física** ([docs/FISICA.md](docs/FISICA.md))
- `Vapor.Physics`: XPBD com subpassos, como programa vapor — em lote, **bit a bit em todo substrato**, diferenciável. Pêndulo de primeira ordem contra o período elíptico exato; caos (pêndulo duplo) oráculo = nativo enquanto um ulp separa os mundos; `sysid/3` recupera haste e amortecimento de medidas com ruído pelo gradiente da trajetória (embaralhadas no tempo: nada); carro-pêndulo por busca aleatória 200/200; **gêmeo digital** com CUSUM e livro em cadeia de *hashes* que se refaz do modelo e das ações. `physics_test.exs`.

**Redes** ([docs/REDES.md](docs/REDES.md))
- `Vapor.Graph`: geradores reprodutíveis (ER, BA, WS, plantado), lei de potência de Clauset–Shalizi–Newman com *bootstrap* e razão de verossimilhança, nulo de configuração e escores-z, Louvain determinístico, SIR e limiar de campo médio, percolação e robustez, PageRank no hospedeiro e **como programa vapor**; = networkx. `graph_test.exs`.

**Visão** ([docs/OCR.md §3g–§3k](docs/OCR.md))
- **CJK** (`Vapor.Vision.CJK`, `priv/ocr-cjk-{zh,ja,ko}`): features de elemento direcional, segmentação decidida pelo reconhecimento, modelo de língua que se abstém. CER em fontes nunca vistas (semente nova): zh 8,9 %, ja 2,6 %, ko 11,3 %; caracteres aleatórios: o modelo de língua não muda nada.
- **Árabe** (`OCR.default(:arabic)`, `priv/ocr-arabic`): leitura em ordem visual, devolvida à lógica (`Vapor.Vision.Bidi` = python-bidi em 600/600); 19,2 % de CER em fontes nunca vistas (leitor latino: 91 %); colunas RTL; o localizador de linhas não parte mais as linhas árabes pelos pontos das letras.
- **Cirílico** (`OCR.default(:cyrillic)`, `priv/ocr-cyrillic`): 2,8 % de CER em quatro fontes nunca vistas, com a metade do vocabulário que o treino não viu (leitor latino: 98 %). `scripts_test.exs`.
- **Legendas**: as quatro linhas mais próximas dos dois lados da figura (antes: duas abaixo), numeração por letra ("Figure A:"); achado pela suíte no conjunto de teste, dito em [OCR.md §3h](docs/OCR.md).
- **Figuras** (`Vapor.Vision.Figure`, e no `OCR.read`): detecção com legenda; **digitalização de gráficos** que recusa quando os rótulos não confirmam uma escala (rótulos permutados: 12/12 recusados; estilo padrão: 27/30 dentro da tolerância, nenhum grosseiramente errado). `figure_test.exs`.
- **Fórmulas → LaTeX** (`Vapor.Vision.Math`, `priv/math`): 4,9 % de erro por *token* em Computer Modern e STIX, nunca vistos; a leitura plana erra mais de 3×. `math_test.exs`.
- **Manuscrito**: medido em fontes manuscritas e **não embarcado** (cursiva latina: 64 % de CER em mãos nunca vistas; Nastaliq 57 %; cirílico 78 %; o japonês em fontes de pincel, 6,4 % com o modelo de língua — ainda fontes, não mãos); `OCR.default(:cursive)` recusa com a medida e o caminho ([OCR.md §3k](docs/OCR.md)).

**Console** ([docs/CONSOLE.md](docs/CONSOLE.md))
- Painéis novos: **Substratos** (impressões numéricas e vereditos), **Treino** (curva, linhas de base, o fluxo além do comprimento de treino), **Física** (o caos rodado duas vezes, animado; o gêmeo com CUSUM e livro), **Redes** (leiaute por forças colorido por comunidade, lei de potência, robustez). *Visão* com a escolha da escrita (latim, árabe, cirílico, cursiva, 中文, 日本語, 한국어, fórmula), linhas RTL, figuras com os dados do gráfico (linhas, pontos, barras nas cores da série) e LaTeX; o rodapé diz qual leitor leu; a cursiva mostra a recusa e o caminho. Endpoints `/v1/vapor/{substrates,physics,graph,lm}`; `/v1/vapor/ocr` com `script`.

**Menores**
- `Vapor.Runtime.Substrates` reconhece os nomes de ISA do FreeBSD (`amd64`, `arm64`).
- `mix.exs` 0.10.0.
- O teste da suíte de qualidade inteira tem 60 min de limite (a rodada 0.10 soma ~11 min aos ~12 de antes).
- O comentário do tradutor MSL não nomeia mais um compilador (a auditoria do código-fonte proíbe o nome no produto; os testes é que compilam o *shim*).

## 0.9.0 — 2026-10-04

Escrutínio dos dois pedidos desta rodada ("um ComfyUI any-to-any … e tudo dos cursos da Hugging Face"; e, antes, remoção de marca d'água e de recusas, destilação adversarial — o registro da decisão): [docs/DIRETRIZ.md §12](docs/DIRETRIZ.md). Cada entrega tem uma verificação com controle em `mix vapor.quality` (§5e). O documento da rodada: [docs/ESTUDIO.md](docs/ESTUDIO.md).

**O estúdio** (`Vapor.Studio`, [docs/ESTUDIO.md](docs/ESTUDIO.md))
- Grafo de nós tipados (`image`, `mask`, `audio`, `video`, `mesh`, `text`, `number`, `tensor`, `latent`, `json`). **Cache exato por conteúdo**: a chave de um nó é o SHA-256 do tipo, versão, parâmetros e chaves de quem o alimenta (DAG de Merkle). Recibo por saída, **raiz de Merkle por execução**, `verify/3` reexecuta sem cache. Grafos mal tipados recusados antes de rodar, cada problema com o nó e o reparo; subgrafos (`studio.input`/`studio.output`, `video.map` por quadro). `studio_test.exs`.
- **64 nós**: imagem (redimensionamento = `torch` e Pillow a 2,2·10⁻⁶; programa nativo = avaliador esparso da BEAM, bit a bit), máscaras e composição, som (reamostragem com 69,6 dB de SNR, espectrograma), vídeo (câmera, mapa por quadro, concatenação, *crossfade*), 3D, RL, visão, difusão; `image.scene`, cenas determinísticas para começar sem arquivo. Codecs próprios: **GIF** (LZW = Go/Pillow), **Y4M**, leitura de **MJPEG-AVI** (= libjpeg quadro a quadro). `studio_media_test.exs`.
- **Importação do ComfyUI** (formato de API): `LoadImage`, `SaveImage`/`PreviewImage`, `EmptyImage`, `ImageScale(By)`, `ImageInvert`, `ImageCrop`, `ImageBlur` (σ em unidades do raio, convertido), `ImageCompositeMasked`, `CheckpointLoaderSimple`, `CLIPTextEncode`, `EmptyLatentImage`, `KSampler`, `VAEDecode`, `VAEEncode`. Cada tradução é declarada, e um nó sem tradução recusa o workflow inteiro. Escritas pela semântica documentada dos nós, sem reproduzir código.
- `Vapor.Studio.Templates`: seis grafos de partida (imagem → plano animado, som e espectrograma, relevo 3D, política treinada × controle, edição com máscara, texto → imagem).

**Stable Diffusion** (`Vapor.Diffusion`, [docs/ESTUDIO.md §2](docs/ESTUDIO.md))
- **U-Net** (`Vapor.Lock.Adapters.UNet`, `UNet2DConditionModel`): blocos com atenção cruzada, cabeças preenchidas a 16 com a escala 1/√dₕ original, GEGLU, concatenação exata de canais, *features* de *timestep* corretamente arredondadas. = diffusers: SD 1.x 1,17·10⁻⁶, SD 2.x 1,03·10⁻⁶.
- **Encoder do VAE** (parte `encoder: :map` do adaptador, com os *pads* assimétricos do diffusers): 1,14·10⁻⁶. `Vapor.Spatial.conv2d` ganha `pads: {topo, base, esq, dir}`.
- **Schedulers** (`Vapor.Diffusion.Scheduler`): DDIM, Euler, DPM-Solver++ 2M, com espaçamento leading/linspace/trailing; os *timesteps* e os latentes finais do diffusers a ≤ 6·10⁻⁷ nas 18 combinações. **Achado**: cada *scheduler* do diffusers espaça os passos do seu jeito, e uma implementação única erra em até 0,19.
- **Pipeline** (`Vapor.Diffusion.Pipeline`): texto → imagem, imagem → imagem e inpainting sobre um diretório diffusers, com tokenizador CLIP a partir de `tokenizer.json` **ou** dos arquivos lentos (vocab.json + merges.txt, = `transformers`). = `StableDiffusion{,Img2Img,Inpaint}Pipeline` a ~10⁻⁶ num checkpoint minúsculo incluído (`priv/quality/sd_tiny`); controle (outro amostrador): 0,096. O latente do VAE é a média (o diffusers amostra, escondido). Nós `diffusion.*` no estúdio; o workflow txt2img do ComfyUI dá os mesmos bits do pipeline. `diffusion_pipeline_test.exs`.

**Ampliação consistente** (`Vapor.Vision.Upscale`, `mix vapor.upscale`)
- ×2/×4 com **D(y) = x por construção**: reduzir o resultado devolve a entrada (10⁻¹⁶; Lanczos 0,05–0,17). MLP de *patches* sobre o Lanczos e projeção exata com redistribuição na saturação; luma pela rede e croma pelo Lanczos. Treinado aqui, reprodutível bit a bit, com recibo. Em imagens retidas, contra Lanczos com a mesma projeção: **+1,5 a +5,8 dB em texto e gráficos de traço**, empate em fotografias, −1,4 dB num gradiente suave (ambos > 53 dB). O CER do OCR a jusante cai à metade. `upscale_test.exs`.

**Aprendizado por reforço e 3D** (`Vapor.RL`, `Vapor.Geom`, `Vapor.Learn`)
- `Vapor.Learn`: MLPs treinadas como um programa recorrente (AdamW, *schedule* cosseno pelas funções CR), oráculo = nativo bit a bit.
- CartPole (= gymnasium a 10⁻¹²), FrozenLake (= a tabela do gymnasium), braço de duas juntas. Iteração de valor, Q-learning (74,7 % = o ótimo; sempre à esquerda 0 %), **REINFORCE** como programa via `Vapor.Autodiff` (472,9/500; sem treino 18), clonagem de comportamento (77 %; aleatório 1 %). Replay = semente + ações. `mix vapor.rl`. `rl_test.exs`.
- Malhas por SDF (tetraedros marchantes, fechadas, volume a 0,4 %), relevo de imagens, OBJ/PLY/GLB (lidos pelo trimesh), rasterizador determinístico, vídeo girando. `geom_test.exs`.

**Agentes** (`Vapor.MCP.Server`, `mix vapor.mcp`)
- **Servidor MCP** (stdio): `studio_catalogue`, `studio_validate`, `studio_run` (o cache vive entre chamadas: o agente que edita um nó recalcula só o que depende dele), `studio_verify`, `comfy_import`, `context_search` (BM25 com prova de Merkle por trecho). Falhas viram resultados `isError` com o reparo; uma exceção num nó não derruba o servidor; caminhos fora do diretório são recusados. Testado com o **cliente oficial do SDK Python do MCP**. `mcp_server_test.exs`.

**Console** ([docs/CONSOLE.md](docs/CONSOLE.md))
- *Estúdio* (novo): tela de nós com paleta por categoria e busca, fios por arrasto **ou pelo teclado** (as fontes de cada entrada no inspetor), arrastar/zoom/enquadrar, prévias (imagem, GIF, som, malha renderizada), o nível de água de cada nó (vazio, em cache, calculado), o **selo** com a raiz de Merkle e *Verificar*, importação do ComfyUI, exportação do JSON. Endpoints `/v1/vapor/studio/{nodes,run,verify,comfy}`; cache e prévias vivem enquanto o servidor vive (uma execução toda em cache: ~1 s). `console_test.exs`.

**Menores**
- O digest de imagem em `Studio.Value` passou a usar o formato externo de termos (os mesmos bits binary64, com etiqueta), fora do *heap*: um episódio de 160 passos foi de 36 s para 2,8 s.
- O cache do estúdio guarda os digests junto com os valores: um nó em cache não é hasheado de novo.
- `video.camera` calcula os quadros em paralelo, com os mesmos bits.
- `mix.exs` 0.9.0.

## 0.8.0 — 2026-10-03

Escrutínio do pedido e do anexo desta rodada (um roteiro "Vapor 1.0" de dezessete itens: o que entrou, o que foi recusado e por quê): [docs/DIRETRIZ.md §11](docs/DIRETRIZ.md). Cada entrega tem uma verificação com controle em `mix vapor.quality` (§5d). Medições: [docs/bench/ROUND08.md](docs/bench/ROUND08.md).

**GPU e esparsidade** ([docs/FRONTEIRA.md §1, §7](docs/FRONTEIRA.md))
- **Sessões residentes no Vulkan** (`OPEN/STEP/CLOSE` no `vapor-fabric`): *pipelines* uma vez, *buffers* na memória direta ou por *staging* (GPU discreta; escolhido sozinho se faltar memória direta), estado realimentado dentro da GPU, *command buffers* gravados reaproveitados pela chave dos bytes exatos do passo (31/32). 72,4 → **12,4 ms/token** (10,5 por *staging*) e 1 MB → **8 kB por token** num Llama reduzido; mesmos bits que a CPU. **O `Vapor.Engine` serve na GPU** (`mix vapor.serve --gpu`, `mix vapor.generate --gpu`); queda do driver → `:session_lost` e reabertura, nunca a BEAM; quadros hostis recusados. `gpu_session_test.exs`.
- **MoE esparso em 4 bits** (`Term.qgemv_masked/3`, kernel `gemv_sb4_masked` em x86/AVX-512/NEON/RVV/SPIR-V): linhas não escolhidas não leem nibbles nem escalas; bits = denso; 1,6× (T = 1) e 2,6× (T = 8), 15,0 M → 5,3 M instruções retiradas. Escalas NaN num especialista não escolhido não mudam um bit. `sparse_sb4_test.exs`.

**Documentos e visão** ([docs/OCR.md §3e–§3f](docs/OCR.md))
- **Tabelas** (`Vapor.Vision.Table`): grades com réguas (bordas virtuais, células mescladas pela cobertura das réguas) e tabelas só de filetes (colunas pelas calhas); células lidas com colunas tipadas (numéricas pela leitura restrita, convenção de espaço, **formas** decodificadas por Viterbi CTC sobre o autômato da forma — `Vapor.Vision.Template`), limiares escolhidos em validação separada. F1 de estrutura **1,000** (0,7: 0,343), mesclas 1,000 (sem detecção: 0,905), CER por célula **5,5 %** (livre: 11,7 %; Tesseract com caixas perfeitas: 2,3 %). Markdown, HTML, CSV; no texto do OCR na ordem de leitura; uma passagem por tabela na biblioteca; `mix vapor.ocr --table-format`, `mix vapor.ocr tables DIR`. `table_test.exs`.
- **JBIG2 aritmético** (`Vapor.Docs.JBIG2`): MQ, genérico (modelos 0–3, AT, TPGDON), MMR, refinamento, dicionários de símbolos (com agregação), regiões de texto (8 cantos, faixas, refinamento), páginas, `/JBIG2Decode` com `/JBIG2Globals`. **43 fluxos = jbig2dec bit a bit** (do jbig2enc e de um codificador MQ próprio conferido contra o H.2 do padrão); controle com o modelo declarado errado: 19/43. Huffman e meio-tom: aviso pelo nome. `jbig2_test.exs`.

**Modelos** ([docs/FRONTEIRA.md §4](docs/FRONTEIRA.md))
- **Mamba-2** (`Vapor.Lock.Adapters.Mamba2`, `Mamba2ForCausalLM`): A escalar por cabeça, B/C por grupo, Δ com `time_step_limit`, RMSNorm com porta; expansão cabeça → canal exata (produto *one-hot*, `gather_row`). = `transformers` (7,8·10⁻⁷, gulosa idêntica), nativo = oráculo, GPU = CPU. **Achado**: o `transformers` normaliza a largura inteira onde o `mamba_ssm` normaliza por grupo — padrão do treino, `gated_norm: :whole` para o outro, cada um falha contra a referência do outro; e o seu passo com cache pula o `time_step_limit`. `mamba2_hf_test.exs`.

**Distribuição** ([docs/FRONTEIRA.md §6](docs/FRONTEIRA.md))
- **Paralelismo de tensor entre nós BEAM** (`Vapor.Shard.Cluster`, `Vapor.Shard.Host`): fragmentos residentes por nó com SHA-256 conferido na chegada; bits de um worker em 1–3 nós; **nó perdido → fragmentos recolocados, mesmos bits**; `replicas: 2` compara réplicas bit a bit e pega um nó que corrompe um bit. Testado com nós `:peer` reais. `shard_cluster_test.exs`.

**Conformidade** ([docs/AUDITORIA.md](docs/AUDITORIA.md))
- **Dossiês de auditoria** (`Vapor.Audit`, `mix vapor.audit keygen|export|verify|demo`): certificados, diários atestados, recibos do log, relatório de qualidade, contratos de modelo e documentos num arquivo CBOR canônico com raiz de Merkle, assinaturas Ed25519 com quórum e âncora no log de transparência; cada item conferido pelas suas regras; mapeamento para o AI Act (Art. 11, 12, 13, 15, 19; Anexo IV) e a ISO/IEC 42001 (A.6.2.3/4/7/8) como dado. **Relatório PDF** com o dossiê anexado (`verify` aceita o PDF) e **página HTML que se confere offline** no navegador (WebCrypto). Um byte alterado: recusado, dizendo onde. Não é avaliação de conformidade (o aviso vai dentro). `audit_dossier_test.exs`.

**Verificação** (`proofs/Vapor/Binary32.lean`)
- **As regras de reescrita provadas** num modelo IEEE-754 a partir dos padrões de bits (núcleo do Lean): para `x` finito, o único resultado corretamente arredondado de `x·1`, `1·x`, `x+(−0)`, `(−0)+x`, `x−(+0)` é `x`; `x+(+0) → x` refutado (`(−0)+(+0) = +0`); `neg(neg x) = x`. `units` extraído e conferido contra `Vapor.F32` em todo expoente. **Correção**: a doc de `Rewrite` dizia "exato inclusive NaN" desde a 0.5 — falso (x86 aquieta sNaN; RISC-V devolve o NaN canônico); corrigida, com teste. `rewrite_soundness_test.exs`.

**Console** ([docs/CONSOLE.md](docs/CONSOLE.md))
- *Visão*: tabelas emolduradas na página e desenhadas das células (mescladas, cabeçalho, números à direita, confiança por célula, ligação célula ↔ caixa), com Markdown/CSV/HTML a um clique; o bloco da tabela marcado na lista de leitura.
- *Dossiê* (novo): conferir um `.vdossier` ou o seu PDF, ou montar o de demonstração; veredito no nível da água; a **trama** evidência × dispositivo, com as lacunas à vista; baixar o dossiê e a página que se verifica sozinha.
- A identidade mostra o substrato (CPU ou `GPU · dispositivo`).

**Menores**
- `Vapor.JSON.decode(…, nonfinite: true)` para os `Infinity`/`NaN` que o Python escreve (usado só no `config.json`); o padrão continua estrito.
- A verificação de cluster da suíte não chama mais `epmd` por *shell* (o teste de auditoria de fonte pegou); sem epmd, ela não roda e o relatório diz.
- `Session.info/1` diz o dispositivo; `/v1/vapor/info` diz o substrato.

## 0.7.0 — 2026-10-03

Escrutínio do pedido desta rodada (o mesmo da 0.5, pela terceira vez): [docs/DIRETRIZ.md §10](docs/DIRETRIZ.md). Tema: **o documento escaneado de escritório**, a saída estruturada com campos inválidos e a fusão que não cabe na memória. Cada entrega tem uma verificação com controle em `mix vapor.quality` (§5c).

**Escaneados de escritório** ([docs/OCR.md §3b–§3d](docs/OCR.md))
- **CCITT Group 3 (1-D e 2-D) e Group 4** (`Vapor.Docs.CCITT`), sem dependência: 42 fluxos iguais **bit a bit** ao codificador do libtiff (ruído, texto, corridas até 2 600, *fill bits*, MH alinhado); dados truncados ou lixo nunca travam. **LZW** (`EarlyChange` 0 e 1) e **RunLength** no PDF, conferidos contra o libtiff e um codificador independente; preditor TIFF 2. `ccitt_test.exs`.
- **Ordem de leitura** (`Segment.blocks/2`): XY-cut recursivo em que uma calha de coluna corta antes de um vão horizontal, com limiares da própria região. Oito páginas escaneadas (1–3 colunas, título e rodapé atravessando, fontes nunca vistas, CCITT em PDF): **CER 1,3 %**; sem a ordem, 53 %; Tesseract 1,6 %. `reading_test.exs`, `priv/quality/scans`, `test/python/scan_pages.py`.
- **Feixe CTC + modelo de língua de caracteres** (`Vapor.Vision.CharLM`, Witten–Bell, contado do corpus de treino *como impresso*): linhas retidas CER 6,5 % → **4,4 %**, linhas com valores e códigos 8,9 % → 7,4 %; o modelo **se abstém** em linhas sem língua (0 de 30 cadeias aleatórias mudadas; sem a guarda, 27), só escolhe entre o que os quadros acham plausível, e um modelo de corpus embaralhado não ganha nada (controle). Pesos escolhidos em validação separada (`priv/ocr/lm.json`).
- **Bugs reais**: máscaras de estêncil (`ImageMask`) eram lidas invertidas; acentos de linhas sem letras altas eram descartados pelo achador de linhas ("não" lido "rão") — achado olhando a interface nova; o título de uma página era cortado ao meio pela primeira versão do XY-cut.

**Saída estruturada** ([docs/AGENTES.md](docs/AGENTES.md))
- JSON Schema **`pattern`** (ECMA-262: classes, escapes, grupos, quantificadores, âncoras nas pontas; o que não é regular por bytes é recusado pelo nome) e **`format`** (`date`, `time`, `date-time` com o calendário real, `uuid`, `ipv4`; `email` e `hostname` como subconjuntos que nunca emitem um valor inválido) viram autômatos de bytes UTF-8 com os escapes do JSON (`Vapor.Grammar.Regex`). Mesmo veredito que o `re` do Python e os *parsers* da biblioteca padrão em 7 240 cadeias. `regex_test.exs`.

**Fusão** ([docs/FUSAO.md](docs/FUSAO.md))
- **Do disco para o disco** (`Merge.stream/3`, `mix vapor.merge --stream`): um tensor por vez, os mesmos kernels na mesma ordem — arquivos **byte a byte** iguais à fusão em memória, mesmas raízes no recibo; pico de memória 4 MB contra 83 MB (a suíte pegou um pico de 0,9× chamado de um processo de *heap* grande — lixo não coletado entre tensores; agora 0,08×). `Safetensors.catalog/1`, `read_entry/3`, `header/2` (escrita de um tensor por vez). `merge_stream_test.exs`.

**Console** ([docs/CONSOLE.md](docs/CONSOLE.md))
- Painel *Visão*: a página escaneada aparece mesmo vinda de PDF; **blocos numerados na ordem de leitura** e o **fio de leitura**; texto agrupado por bloco; **os caracteres escolhidos pelo modelo de língua** destacados, com a leitura só dos quadros a um clique — inclusive onde o modelo erra. Inglês e português, claro e escuro.

## 0.6.0 — 2026-10-03

Escrutínio dos dois anexos desta rodada (quinze itens para a 1.0; o que faltaria para Sora, Midjourney e modelos de mundo): [docs/DIRETRIZ.md §9](docs/DIRETRIZ.md). Medições: [docs/bench/FRONTIER.md](docs/bench/FRONTIER.md). Cada recurso novo tem uma verificação com controle em `mix vapor.quality` (§5b, 48/48).

**Servir modelos de fronteira** ([docs/FRONTEIRA.md](docs/FRONTEIRA.md))
- **MoE esparso por predicação de linha** (`Term.linear_masked/3`): o GEMV pula as linhas que não escolheram o especialista e lê só os pesos escolhidos; bits = denso em AVX2/AVX-512/RVV/Vulkan/QEMU; 2,6× no decode de um Mixtral reduzido. A permutação de tokens do anexo foi recusada com motivo (custo de CPU é ler pesos). `sparse_experts_test.exs`.
- **Cache latente do MLA** (`mla: :latent`, padrão dos modelos MLA): cache `[c | k_rope]` compartilhado pelas cabeças, query e valor absorvidos (`Term.linear_grouped/3`, bloco-diagonal); 85× menos memória nas formas do DeepSeek-V3; conferido contra o `transformers`. `latent_attention_test.exs`.
- **Janela deslizante executada exatamente** (era recusada quando ligava): só a faixa muda, na ordem canônica; Mistral e Gemma 3 contra o `transformers` (controle: sem janela falha). E o **cache circular**: quando a janela liga em todas as camadas, cada sequência guarda `⌈(w + T − 1)/página⌉` páginas num anel da tabela de blocos (7,9× mais sequências a 32 k / janela 4 k), bits = cache inteiro; o limite é justo (um a menos muda os bits — teste). `sliding_window_test.exs`, `engine_test.exs`.
- **Mamba** (`Vapor.Lock.Adapters.Mamba`, `MambaForCausalLM`): passo recorrente — prefill e decode são as mesmas instruções —, estado mantido no worker (o quadro `STEP` agora realimenta estado que não é atualizado no lugar), `Vapor.Recurrent` para gerar; 3,4·10⁻⁷ e gulosa idêntica ao `transformers`; custo por token constante. `mamba_test.exs`, `mamba_hf_test.exs`.
- **Especulação em árvore** (`Vapor.Speculative.Tree`): ramos em *slots* sobre as páginas compartilhadas do contexto (o ramo vencedor herda suas páginas — sem cópia), saída = gulosa do alvo em toda forma de árvore; rascunho por **busca no prompt** com cópia sobreposta (6 tokens/passo onde a saída segue o contexto). `speculative_tree_test.exs`.
- **Paralelismo de tensor exato** (`Vapor.Shard`): coluna-paralelo + all-gather, bits = um worker; a forma linha-paralela é medida (muda 81 % dos elementos). `shard_test.exs`.
- **Bug real**: DeepSeek com `n_group = 1` (V2-Lite) não construía; agora a limitação por grupos é a identidade, com teste.

**Numérica e verificação**
- **Divisão corretamente arredondada** (semântica versão 2, registrada nos certificados): Markstein + resíduo exato de Dekker sem FMA; `+ − × ÷` = IEEE (DAZ/FTZ) em todo substrato, GPUs incluídas; 0 erros em 50 M quocientes exaustivos. `division_test.exs`.
- **`log`** canônico (≤ 1 ulp do logaritmo corretamente arredondado, especiais IEEE) e **`softplus`** (composição de nós canônicos, ≤ 4 ulps), em todo substrato = oráculo.
- **Lema de Higham em Lean** (`proofs/Vapor/Higham.lean`, núcleo só); extração regenerada.

**Espacial, latente e áudio** ([docs/ESPACIAL.md](docs/ESPACIAL.md))
- `Vapor.Spatial`: convolução 2-D e 3-D (qualquer *stride*, *padding*, dilatação) **sem kernel de convolução** (gather + sel + reshape + GEMV), GroupNorm por contrações seletoras exatas, *upsampling*, atenção sobre pixels; ≤ 4·10⁻⁷ contra o torch. `spatial_test.exs`.
- Adaptadores **VAE** (`AutoencoderKL`, decoder; 8,7·10⁻⁷ contra o diffusers) e **DiT** (`DiTTransformer2DModel`, adaLN-Zero; posições bit a bit).
- **Atenção cruzada** = a atenção existente (horizonte na última linha da outra corrente), ≤ 10⁻⁶ contra `nn.MultiheadAttention`. `cross_attention_test.exs`.
- **Whisper** (`Vapor.Lock.Adapters.Whisper`): encoder e decoder como dois programas (o contrato ganhou **partes** declaradas), K/V cruzados uma vez por áudio; 4,8·10⁻⁷ / 4,2·10⁻⁷ e gulosa idêntica ao `transformers`. `whisper_hf_test.exs`.
- **Torre de texto do CLIP** + tokenizador BPE com `</w>` (= o do CLIP em 20/20 linhas). `clip_tokenizer_test.exs`, `lock_hf_test.exs`.

**Transparência** ([docs/TRANSPARENCIA.md](docs/TRANSPARENCIA.md))
- `Vapor.Tlog`: log Merkle RFC 9162 em arquivo só-acréscimo com `fsync`, provas de inclusão e consistência, *checkpoints* C2SP *signed-note*, co-assinaturas de testemunha (`Vapor.Tlog.Witness`: recusa retrocesso e bifurcação); 196 sondas do transparency-dev. `tlog_test.exs`.
- Servidor: `GET/POST /v1/vapor/tlog`, provas por HTTP, recibos de busca ancorados; `mix vapor.serve --tlog`.
- Console: aba **Registro/Ledger** — o navegador verifica sozinho (WebCrypto Ed25519, chave fixada no primeiro uso, consistência com o último *checkpoint* visto) e desenha a prova de inclusão. `test/js/tlog_verify.mjs`.

**Testes e medições**
- `mix vapor.bench --frontier` → `docs/bench/FRONTIER.md`; `mix vapor.quality` ganhou a seção 0.6 (oito verificações com controle).
- Corrigidos: opções ignoradas no *helper* dos testes do motor; um pedido maior que o pool inteiro esperava para sempre (agora `{:kv_pages, …}`); `layer_types` era ignorado fora do Gemma 3 (um híbrido Mistral seria lido como todo deslizante) — agora é lido ou recusado; o motor lia a configuração da família para o anel (o teste de arquitetura pegou) — agora pergunta ao adaptador (`Vapor.Lock.ring_window/2`, *callback* opcional).

## 0.5.0 — 2026-10-02

Escrutínio do pedido desta rodada: [docs/DIRETRIZ.md §8](docs/DIRETRIZ.md). As seis limitações que a 0.4.0 declarou, uma a uma:

**1. Paridade com o próprio `transformers`** (não mais contra NumPy escrito a partir dele)
- `test/vapor/lock_hf_test.exs` + `test/python/hf_lock.py`: checkpoints **escritos pelo `transformers` 5.18** (Phi-3, Phi-3 com rotary parcial, Granite, ViT, ViT com pooler, CLIP-vision) admitidos pela eclusa sem tensor esquecido e comparados ao forward dele: ≤ 4,5·10⁻⁷ relativo, decodificação gulosa idêntica.
- **Bug real encontrado:** o `transformers` ≥ 5 grava `partial_rotary_factor` *dentro* de `rope_parameters`, onde a 0.4.0 não olhava — um checkpoint estilo Phi-4-mini era admitido e calculado errado (erro relativo 0,5) em silêncio. Agora o rotary parcial é **construído** (permutação dobrada nos pesos de q/k, tabela com pares de passagem) e **toda chave desconhecida** de `rope_parameters` é recusada pelo nome.
- Encoder: pooler do `ViTModel` (`tanh(dense)`), torre de visão do **CLIP** (`pre_layrnorm`, norma final só na linha agrupada, `quick_gelu`, `visual_projection`).

**2. Any-to-any em dados reais** ([docs/ANY_TO_ANY.md §6](docs/ANY_TO_ANY.md))
- Caligrafia real (UCI/scikit-learn, 1 797 dígitos, 497 retidos): imagem → dígito (`vapor_mlp`), dígito → imagem por **difusão** (`Vapor.Modal.Diffusion`, DDIM determinístico no substrato), lida de volta pelo classificador e comparada com a imagem de treino mais próxima.
- Fala real (Free Spoken Digit Dataset): front-end certificado (`Vapor.Modal.Speech`: espectro de Hann + banco mel como programa) e um `vapor_encoder` treinado em 5 vozes, medido numa 6ª nunca ouvida.
- Cadeia voz real → texto → desenho → leitura, medida.
- Novo adaptador de topologia `vapor_mlp`; o encoder ganhou `head: "rows"` (classificação por linha) e `rows:` (programas mais curtos).

**3. Difusão, OCR e JPEG** ([docs/OCR.md](docs/OCR.md))
- `Vapor.Docs.JPEG`: baseline, sequencial estendido e progressivo; **os mesmos pixels do Pillow (libjpeg-turbo) em 51 de 51 arquivos** (IDCT inteira, *upsampling* fancy e tabelas de cor do libjpeg reproduzidos exatamente).
- `Vapor.Vision.OCR`: OCR como **modelo admitido pela eclusa** — geometria por primeiros princípios (Sauvola, componentes, linhas pelo miolo vertical) e um `vapor_encoder` que lê as colunas da linha com **CTC** (sem segmentar caracteres). Páginas escaneadas de PDF (`DCTDecode`, Flate com preditores PNG, 1 bit) e imagens entram no índice com a confiança. `mix vapor.ocr`. `test/vapor/ocr_test.exs`: o leitor embarcado medido em fontes fora do treino e numa foto real, e os logits por coluna iguais aos de uma implementação PyTorch independente.
- Difusão verificável: o denoiser ótimo de uma mistura gaussiana em forma fechada (`analytic/2`) — e, para um conjunto finito, **atenção** sobre os pontos de treino — testa o amostrador; um bug de consistência do DDIM com *clamp* foi achado assim (17 % → 99 % dos dígitos gerados lidos certo).

**4. Busca de imagem por significado**
- Imagens com texto (prints, escaneados, slides) são achadas pelo que está escrito nelas (OCR → índice). Torre de visão do CLIP conferida; a torre de texto fica no TODO.

**5. Fusão rápida: ≈ 0,6 → ≈ 11 M parâmetros/s** ([docs/FUSAO.md](docs/FUSAO.md))
- Kernels por casamento de binários, blocos em todos os escalonadores, DARE com gerador-contador (qualquer bloco começa no seu deslocamento): **saída bit a bit igual à 0.4.0** em todos os métodos, inclusive recibos.

**6. "TIES e DARE pioraram" → diagnóstico, medição e seleção**
- `Merge.diagnose/2` (`--diagnose`): regime pelos pesos — tamanho dos deltas, concentração de energia, conflito de sinais, e **se há ancestral comum** (cosseno dos pesos).
- `Merge.select/4` (`--try … --eval`): cada candidato medido em texto retido, recibo `vapor.merge.select/1`.
- `:regmean` (mínimos quadrados sobre as ativações; Grams no substrato; `Vapor.Lock.taps/1` diz o que medir) — medido, empata com a linear; registrado assim.
- Benchmark com **transformers treinados de verdade** (`priv/quality/merge`): a fusão linear de dois ajustes finos bate a base nos dois idiomas e, na média, os dois especialistas; a de redes sem ancestral comum é pior que ambas — como o diagnóstico avisa antes.

**Interfaces** ([docs/INTERFACES.md](docs/INTERFACES.md))
- Console redesenhado (identidade de eclusa: cada resultado num tanque cujo nível é a sua medida), **inglês por padrão e português**, claro e escuro, painéis *Visão*, *Ouvir* (gravação pelo microfone), *Desenhar*, *Fusão*; logo SVG, favicon, manifesto de app instalável; **token** obrigatório para expor além de 127.0.0.1.
- `mix vapor.tui`: o console no terminal, sem dependência.
- Tauri considerado e não adotado (motivos e quando revisitar no documento).

**Qualidade** — novas verificações com controle em `mix vapor.quality`: fusão de modelos treinados (5), dados reais (OCR em fontes fora do treino e em foto real, fala de voz nunca ouvida, caligrafia nos dois sentidos, novidade contra memorização, cadeia), JPEG.

## 0.4.0 — 2026-10-02

Escrutínio do pedido e do que ficou de fora: [docs/DIRETRIZ.md](docs/DIRETRIZ.md).

**Eclusa de modelos** ([docs/ECLUSA.md](docs/ECLUSA.md))
- `Vapor.Lock`: o único lugar onde uma família de modelo é conhecida; contratos (`:causal_lm`, `:encoder`, `:codec`, `:map`) conferidos na fronteira; recusa com quase-acertos e reparo; `mix vapor.lock`.
- Três níveis de adaptador: alias por dados/JSON (Phi-3/Phi-4 embutido), *blueprint* (Granite 3.x), topologia (encoder/ViT do HF, codec VQ, projetor linear).
- Motor, embedder, servidor e RAG leem só o contrato; teste de arquitetura sobre a tabela de átomos do BEAM.

**Any-to-any sem operador novo** ([docs/ANY_TO_ANY.md](docs/ANY_TO_ANY.md))
- GELU exata como microprograma canônico (fecha TODO 4.1).
- Imagem (PPM/PNG, patches exatos), áudio (WAV, espectro de Hann certificado, síntese aditiva), VQ por `sample`, injeção de *soft tokens* (`inject: true`), hub com pivô; dez rotas medidas em dados retidos.

**Fusão de modelos** ([docs/FUSAO.md](docs/FUSAO.md))
- linear, *task arithmetic*, SLERP, TIES, DARE; determinística, em *streaming*, recibo co-assinável; `mix vapor.merge` (fecha TODO 6.3).

**Qualidade das saídas** ([docs/QUALIDADE.md](docs/QUALIDADE.md), [docs/bench/QUALITY.md](docs/bench/QUALITY.md))
- Portões calibrados contra controles que recusam existir sem separação; modelos plantados; `mix vapor.quality` (27 verificações, sai 1 em falha) e `--model` para checkpoints reais.

**Documentos e RAG de arquivos** ([docs/DOCUMENTOS.md](docs/DOCUMENTOS.md))
- `Vapor.Docs`: zip recursivo à prova de *zip bomb*, PDF (object streams, ToUnicode, Type 1/TrueType; criptografado recusado), Office/OpenDocument/EPUB, HTML, PNG completo, metadados de JPEG; conferido contra `pdftotext` e Pillow.
- `Vapor.Docs.Library`: raiz sobre texto e arquivos, proveniência até a página, busca por imagem parecida; `mix vapor.rag`.

**Console web** ([docs/CONSOLE.md](docs/CONSOLE.md))
- Página em `/` do `Vapor.Serve`, offline: conversa com evidência e citações conferidas, documentos, contrato do modelo, medidor de ruído; `mix vapor.serve --docs`, funciona sem modelo.

**Consertos**
- Exportação GGUF gravava Qwen3/Gemma 3/Mixtral/DeepSeek como `llama` em silêncio: agora recusa.
- Phi-3 exportado para o HF perderia a janela deslizante: escrito com a grafia do Mistral.
- `Vapor.RAG` lia um campo do decoder (`cfg.hidden`): agora o contrato.
- CLIs sem locale UTF-8 recebiam argumentos em mojibake: reparados.
