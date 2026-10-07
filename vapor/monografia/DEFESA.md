# Roteiro da defesa oral — *vapor: computação certificada, do termo algébrico ao registrador*

> Material de apoio da defesa da monografia ([monografia.pdf](monografia.pdf)), com
> a apresentação [slides/vapor.pdf](../slides/vapor.pdf) (42 slides). Todo número
> dito em voz alta está na monografia e é reproduzível pelo Apêndice A; se a banca
> perguntar "de onde vem isso?", a resposta é sempre um comando.

## 0. Formato e tempo

| bloco | slides | tempo | capítulos da monografia |
|---|---|---|---|
| Abertura | 1 | 1 min 30 s | 1 |
| 1 · Motivação: as dores | 2–4 | 4 min | 1 |
| 2 · Por quê: axiomas e garantias | 5–7 | 4 min | 1, 2, 3.8 |
| 3 · Como: a arquitetura | 8–17 | 10 min | 3 |
| 4 · Ecossistema e HPC | 18–21 | 3 min | 4.1–4.2 |
| 5 · Agentes, RAG e ZK | 22–25 | 2 min | 4.6–4.8 |
| 6 · Para quem | 26–29 | 3 min | 4, 8 |
| 7 · Rodadas, finanças (0.13) e a bancada aberta (0.14) | 30–41 | 13 min | 5, 6, 7 |
| 8 · Contribuição | 42 | 2 min 30 s | 9 |
| **total** | | **≈ 43 min** | |

Regra prática: **um minuto por slide**, com dois lugares para gastar mais (a
arquitetura e a rodada 0.13) e dois para economizar (as rodadas 0.6–0.7, slides
28–29, e a tabela de ZK, slide 25). Se a banca impuser 20 minutos, use o
**corte curto** da §3.

Postura: sóbria. Não vender; mostrar. Cada afirmação forte vem acompanhada de
"e o controle foi…" ou "e onde perdemos é…". A banca confia mais em quem diz
os próprios limites antes de ser perguntado.

---

## 1. Fala por slide

### Slide 1 — Título (1 min 30 s)
**Objetivo:** dizer a tese em uma frase e o roteiro em outra.

> "Esta monografia defende que **verificabilidade pode ser uma propriedade de
> arquitetura**: obtida por construção, a custo baixo, se três decisões forem
> tomadas desde o início — semântica simbólica fixa, compilador não confiado com
> código isolado, e toda resposta acompanhada do seu juiz. O vapor é a
> evidência: um compilador de tensores em Elixir, sem dependências, que emite
> bits para cinco alvos, nunca executa código gerado dentro da máquina virtual e
> sai de cada compilação com um certificado. Sobre ele, treze rodadas; a última
> leva o princípio a finanças.
> Vou seguir as cinco perguntas da monografia — o quê, para quem, para que, por
> que e como — nesta ordem: motivação, por quê, como, ecossistema, para quem,
> as rodadas com ênfase em finanças, e a contribuição."

**Transição:** "Começo pelas dores, porque a arquitetura só faz sentido diante delas."

### Slide 2 — Dor 1: gigabytes que ninguém audita (1 min 20 s)
> "Uma instalação de PyTorch com CUDA são cerca de 3 GB de *wheels*. O núcleo do
> vapor — plano de controle compilado mais os dois binários nativos — é 1,8 MB;
> o sistema inteiro, com as treze rodadas, 8,7 MB. O núcleo tem cerca de 19 mil
> linhas: 12,9 mil de Elixir, 4,7 mil de Zig, 1,5 mil de Lean.
> O ponto não é o tamanho: é que **cada componente é um ponto de confiança, e
> confiança não é transitiva**. Escopo menor, sim — e digo isso já: o vapor
> cobre muito menos operadores que o PyTorch. Mas a base inteira cabe numa
> revisão de código."

### Slide 3 — Dor 2: o driver cai, o serviço cai junto (1 min)
> "Na arquitetura usual, o *runtime* do fornecedor vive no mesmo espaço de
> endereçamento do serviço; um sinal derruba tudo. No vapor, o código gerado
> roda em processos separados; uma falha vira uma mensagem tipada, o supervisor
> reinicia o processo e a próxima requisição roda. Contenção não é try/catch: é
> outro espaço de endereçamento."

### Slide 4 — Dor 3: a mesma soma, bits diferentes (1 min 40 s)
> "Ponto flutuante não é associativo: (a+b)+c pode diferir de a+(b+c). Quem
> paraleliza uma redução muda a ordem com o número de *threads*, com a largura
> do vetor, com o *driver* — e o resultado muda com a máquina. A resposta usual
> é tolerância. A do vapor é **fixar a ordem**: toda contração usa 16
> acumuladores reduzidos numa árvore fixa, a mesma em x86, ARM, RISC-V e GPU. O
> paralelismo vem de processar linhas independentes, nunca de reassociar uma soma.
> Isso transforma 'deu diferente na outra máquina' num teste de igualdade."

**Transição:** "Por que essas três dores pedem uma arquitetura nova, e não mais uma camada?"

### Slide 5 — Cinco axiomas (1 min 20 s)
> "Em vez de empilhar bibliotecas, cinco axiomas, cada um eliminando uma classe
> de falha: certificar ou rejeitar; residência limitada — sem *spill*;
> contenção de falhas; síntese binária direta — sem assembler, sem LLVM; e uma
> porta de entrada tipada. Cada axioma é conferido por máquina a cada compilação."

### Slide 6 — Provado, testado, confiado (1 min 20 s)
> "Segurança começa por dizer o que **não** está provado. Provado em Lean 4,
> sem axiomas nem *sorry*: 82 teoremas — o verificador de alocação, o envelope
> de erro, a aritmética modular, as regras de reescrita num modelo IEEE de bits.
> Testado contra quem não compartilha código conosco: binutils, spirv-val, QEMU,
> lavapipe, PyTorch, QuantLib. Confiado, e nomeado: o kernel, a BEAM, o
> compilador Zig, o kernel do Lean, o *driver* Vulkan — este último contido num
> processo."

### Slide 7 — Precisão: exata por construção, envelope quando rápida (1 min 20 s)
> "Dois contratos. `:canonical` — os mesmos bits do oráculo exato em toda
> máquina. `:fast` — qualquer resultado dentro de um envelope rigoroso, calculado
> em racionais diádicos com a cota de Wilkinson e decidido por código extraído
> do Lean. **Reprodutível não é o mesmo que exato**: o canônico responde 'por que
> mudou?'; o envelope responde 'quão longe da verdade?'."

**Transição:** "Agora, como isso é construído."

### Slide 8 — Arquitetura tripartite (1 min 30 s) · Figura 2 da monografia
> "Três partes. O plano de controle, na BEAM, decide: tipos, reescrita exata,
> baixar para a KIR, alocar, emitir bits, verificar, assinar. Dois substratos
> executam: o worker, para CPU — seccomp, W^X, *watchdog*, *pool* de *threads*
> —, e o fabric, para GPU via Vulkan. Entre eles, uma barreira de processo e
> quadros com comprimento prefixado. **Nenhum byte gerado roda dentro da VM** —
> um teste de auditoria proíbe NIFs."

### Slide 9 — Ponta a ponta (1 min)
> "Do arquivo de pesos ao token: os pesos atravessam uma vez, por mapeamento de
> memória endereçado pelo SHA-256; o código atravessa como bits; os tokens voltam
> em fluxo. Sem C, sem LLVM, sem *runtime* do fornecedor na CPU."

### Slide 10 — Bits sem toolchain (1 min)
> "A mesma multiplicação vetorial em três ISAs, campo a campo. A especificação
> da ISA é o único contrato — e uma ferramenta que não compartilha código
> conosco, o objdump, é o árbitro de **cada** instrução emitida nos testes. O
> produto nunca a invoca."

### Slide 11 — Alocação sem spill, verificada (1 min 20 s)
> "O alocador é um *linear scan* com grupos alinhados — o LMUL do RISC-V
> generalizado a todas as ISAs. Não há *spill*: se faltam registradores, o
> compilador muda a forma do código. E o ponto principal: **a heurística não é
> provada; o verificador é**. Toda alocação é conferida por um verificador
> provado em Lean e extraído para Elixir. Isso vale para qualquer heurística
> futura — é a ideia da validação de tradução aplicada a uma fase."

### Slide 12 — Isolamento em camadas (50 s)
> "As quatro classes de falha — instrução ilegal, acesso inválido, laço
> infinito, chamada de sistema proibida — são provocadas nos testes. Em todas,
> só o worker morre."

### Slide 13 — Auto-recuperação (50 s)
> "A GPU falha no meio do serviço; o despachante desce a cadeia de *failover* e
> roda o mesmo plano em código nativo. Como os bits são os mesmos em todo
> substrato, **recuperar é rotear**: a resposta não muda, só o tempo."

### Slide 14 — Escada de verificação (1 min)
> "Seis degraus: tipos; admissão — alocação aceita pelo verificador em toda
> ISA; identidade adjunta, que pega erros de *layout*; oráculo diferencial;
> paridade entre substratos mais envelope; e certificado. Toda compilação
> termina em certificado ou em contraexemplo — nunca em 'talvez'."

### Slide 15 — Do teorema ao runtime (50 s)
> "O extrator lê os termos elaborados do Lean — os mesmos sobre os quais os
> teoremas falam — e gera Elixir. Mais de 480 vetores de conformidade calculados
> no Lean precisam ser reproduzidos, e o módulo gerado carrega o *digest* das
> fontes. A extração é de *build*, não de execução."

### Slide 16 — Árbitro de três tetos (1 min)
> "O despacho escolhe o substrato pelo tempo previsto com três tetos: cálculo,
> banda e emissão de instruções. O terceiro é o que torna o modelo honesto: o
> GEMV de 4 bits é limitado por instruções, não por banda; o *roofline* clássico
> erra por cerca de 5×. Com os três: previsto 71,7 ms, observado 72 a 75."

### Slide 17 — Certificado, quórum e borda (1 min)
> "O certificado não contém nada que dependa do *host* — nem tempos, nem a
> decisão de despacho, só o trabalho contado. Por isso dois nós que refazem a
> escada produzem os mesmos bytes e podem co-assinar. A borda confere em
> milissegundos e executa sem refazer nada."

**Transição:** "Esse núcleo nunca precisou mudar para sustentar o resto."

### Slide 18 — Do núcleo ao LLM (40 s)
> "Modelos de linguagem, visão, áudio e difusão entram por uma eclusa que os
> transforma num contrato conferido. Nenhuma família exigiu exceção às garantias."

### Slide 19 — HPC sem trocar bits (1 min)
> "Cada técnica de desempenho — SIMD, *threads*, lote contínuo, KV paginado,
> réplicas, especulação, GPU — tira paralelismo de saídas independentes. A
> tabela tem uma coluna que importa: o invariante testado, sempre uma igualdade de bits."

### Slide 20 — Medido (40 s)
> "Os ganhos aparecem onde o *roofline* diz que estão; nenhum deles muda um bit."

### Slide 21 — Interoperabilidade conferida (40 s)
> "Um formato só é suportado quando a ferramenta de referência concorda:
> `transformers`, `gguf-py`, `llama.cpp`. O caso mais útil: a conferência contra
> o próprio `transformers` achou um defeito real de *rotary* parcial."

### Slides 22–25 — Fronteira, agentes, Elixir, ZK (2 min no total)
> (22) "As famílias de fronteira entraram como contrações exatas da mesma álgebra."
> (23) "Um agente cuja execução é um diário encadeado: **repetir é verificar**."
> (24) "O ecossistema Elixir entra por pontos de extensão, cada um com veredito."
> (25) "Em ZK e FHE separei fato de analogia; o que foi construído está testado.
> Não reivindico mais que isso."

### Slide 26 — Pilha tradicional × vapor (1 min)
> "Esta é a comparação honesta, **inclusive onde o vapor perde**: cobertura de
> operadores, desempenho de pico, treino e ecossistema. O vapor não é um
> substituto do PyTorch; é outro ponto do espaço de projeto."

### Slide 27 — Para quem (1 min)
> "Missão crítica e regulados, quem aposta em RISC-V, quem já vive na BEAM. Para
> cada um, o que ganha hoje e **o que ainda falta** — certificação DO-178C, silício
> RVV, *buffers* de GPU discreta."

### Slides 28–29 — Rodadas 0.6 e 0.7 (30 s no total; passar rápido)
> "Duas rodadas de exemplo: modelos de fronteira sem perder os bits; o
> escaneado de escritório, com um modelo de língua que **se abstém** onde não há
> língua — e um controle que mostra isso."

### Slide 30 — De 0.8 a 0.14 (1 min)
> "A regra que atravessa as rodadas: uma busca **propõe**, um verificador
> **decide**, e cada resposta traz o número que poderia ter saído se o juiz
> fosse frouxo — o controle. A suíte de qualidade cresceu de 122 para 147 e
> agora 170 verificações, todas com controle."

**Transição:** "A rodada 0.13 testa a tese onde ela tem preço: o mercado."

### Slide 31 — 0.13: verificabilidade, não velocidade (1 min 20 s) · Cap. 5
> "As falhas mais caras em finanças não são de latência; são de prova: um
> número de risco que muda de servidor, um *backtest* que olhou o amanhã, a
> melhor de cinquenta estratégias vendida como a única, um livro de ofertas que
> ninguém audita. A mesa escreve o problema como texto, e cada resposta sai com
> o que permite julgá-la: calendários iguais ao QuantLib dia a dia por 89 anos,
> curvas com todo instrumento reprecificado, opções com os limites de não
> arbitragem conferidos **antes** de resolver."

### Slide 32 — Monte Carlo canônico (1 min 30 s) · §5.5, Proposição 5.1
> "Aqui a camada volta a compilar para o núcleo. O passo da trajetória —
> **inclusive o gerador** — é escrito como termos da álgebra. Escolhi o
> Wichmann–Hill porque seus produtos ficam abaixo de 2²³: cada atualização é
> exata em binary32, e o piso sai do truque de arredondamento por 2²³; conferi
> exaustivamente os cerca de 91 mil estados. O resultado: os bits do oráculo,
> os bits de outra contagem de *threads*, e 6 a 23 vezes a BEAM. O controle:
> esquecer o termo de Itô dá z = 8,7 — o estimador viciado é pego.
> O limite, dito: Wichmann–Hill é antigo; serve à precificação, não a estudos
> de cauda."

### Slide 33 — Backtests: antecipação é propriedade de prefixo (1 min 30 s) · §5.7
> "Este é o resultado que considero mais útil para a indústria. O sinal é
> recalculado sobre históricos truncados e tem de ser **igual, bit a bit**, ao
> calculado com a história inteira. É caixa-preta — não confia nos operadores —
> e só é exato porque a aritmética é determinística. Pega a espiada no amanhã
> no dia 89, e pega a normalização pela amostra inteira, que é causal a cada
> passo e não no todo. Os outros três portões tratam a seleção: o melhor de 30
> cruzamentos sobre ruído tem Sharpe positivo e DSR de 0,10 — reprovado."

### Slide 34 — Arbitragem pelo lema de Farkas (1 min) · §5.8
> "O teorema fundamental é um teorema da alternativa: **ou** um portfólio de
> arbitragem, **ou** preços de estado positivos. Um simplex racional exato acha
> o lado verdadeiro e entrega o objeto que o prova, conferido só por
> multiplicação. E aceita a proposta de qualquer um — inclusive de um modelo de
> linguagem pelo MCP."

### Slide 35 — O livro de ofertas como objeto verificável (1 min 20 s) · §5.9
> "O motor é uma função pura; o diário é encadeado por SHA-256 e fechado por
> Merkle; um motor ingênuo, escrito à parte, refaz cada evento. O episódio
> que mais ensinou: o *fuzzing* achou que uma ordem **FOK a mercado** virava IOC
> nos **dois** motores — a mesma leitura errada da especificação escrita duas
> vezes. A reexecução não viu, porque os dois concordavam; só o invariante, que
> não executa nada, viu. Por isso o juiz tem duas camadas."

### Slide 36 — Microestrutura (40 s)
> "Cada modelo com a medida que diz se serve: Hawkes com o teste de reescala do
> tempo, o Poisson como controle; Avellaneda–Stoikov reproduzindo o artigo;
> Almgren–Chriss com a forma fechada igual ao ótimo numérico a 10⁻¹⁶."

### Slide 37 — Pendências fechadas e abertas (1 min)
> "Quatro itens da lista de pendências fechados nesta rodada — LP com Farkas,
> arquivos assinados, °C/°F como leituras, transistores iguais ao ngspice — e o
> que fica aberto, a começar pelo motor de baixa latência. A verificação desta
> rodada: 23 de 23 verificações com controle, e o console conferido em inglês
> e português, exemplo por exemplo."

### Slide 38 — 0.14: de vitrine a ferramenta (1 min 20 s) · Cap. 6

"A crítica que abriu esta rodada foi que o sistema era expositivo: a ciência
mostrava problemas já resolvidos; as cenas só aceitavam ações prontas. Os
buscadores famosos têm um denominador comum — propor, avaliar, certificar.
Então o produto passou a ser o caso geral: uma linguagem pura em que qualquer
problema se escreve, uma fornalha que o busca e uma pedra de toque que confere.
Os casos famosos viraram exemplos." Apontar as três entradas (pessoa, modelo,
programa) chegando à mesma porta. **Frase-chave:** o modelo propõe; quem decide
é o verificador.

### Slide 39 — Athanor: a fornalha contra o acaso (1 min 20 s) · §6.3

Ler a tabela pela coluna do controle: Golomb 25 contra zero acertos do acaso;
a força bruta independente e o certificado forjado recusado; R(3,3) provado e o
pentágono refutando K5; e o holdout — o mesmo minerador de regras dá ρ = 0,73
quando há momento plantado e −0,33 num passeio aleatório. "Provado quer dizer
enumeração completa de um espaço finito, e o certificado diz o tamanho."

### Slide 40 — Crucible e Assay (1 min 20 s) · §6.4–6.5

Crucible: evidência sem gabarito — leis de conservação provadas sobre ℚ por um
sistema linear exato; seis hamiltonianos aleatórios redescobertos, nenhuma lei
inventada em seis dissipativos. Assay: "a pergunta da pesquisa em IA não é como
treinar, é se a diferença é real". Contar o defeito: o controle de perdas
embaralhadas fez a lei de escala estourar — corrigido, e agora o holdout diz
33 % de erro, que é a resposta certa.

### Slide 41 — Tudo pelo terminal (40 s) · §6.6

Um comando que refuta (sai 1) seguido de um que confere (sai 0). Nenhuma
capacidade existe só na interface gráfica; o console é um cliente do terminal.
A identidade visual: a fornalha é o gráfico da busca, a pedra de toque é o
veredito.

### Slide 42 — Legado (2 min 30 s) · Cap. 9
> "O que fica, com a evidência de cada item: tensor para bits em cinco alvos
> sem *toolchain*; contenção total; numérica canônica portátil; provas que
> executam; desempenho previsível; o ecossistema sobre o mesmo núcleo; e
> domínios inteiros com certificado.
> As limitações, nas mesmas palavras da monografia: uma máquina modesta, sem
> silício RVV ou GPU discreta; transcendentais não corretamente arredondadas;
> finanças em microssegundos, não nanossegundos, e sem dados reais.
> A lição de método que fica: **todo número deve sair com o que permite
> julgá-lo, e todo juiz deve mostrar que poderia ter recusado.** Obrigado."

---

## 2. Perguntas prováveis da banca e respostas

As respostas são curtas de propósito: responder, citar a seção, parar.

### Sobre a bancada aberta (0.14)

**"Isso não é só um otimizador genérico?"** É, de propósito — com três coisas que
otimizadores genéricos não entregam juntas: o controle aleatório com o mesmo
orçamento (a busca precisa mostrar que venceu o acaso), o holdout (precisa
mostrar que o vencedor não é viés de seleção) e o certificado reconferível por
quem não confia nela. Onde há resolvedores especializados (SAT, LP), as mesas
exatas continuam; a fornalha não compete com eles no terreno deles.

**"Entrada aberta não é perigosa?"** É por isso que a linguagem não tem efeitos:
sem E/S, combustível em todo passo, tetos de tamanho, processo com teto de
memória, nenhum átomo criado. Cada limite tem um teste que tenta quebrá-lo; a
análise achou um vazamento antigo (um módulo por expressão compilada) e o fechou.

**"E se o modelo de linguagem alucinar?"** Ele não decide nada: o rascunho é
compilado e devolvido com uma retrotradução para a pessoa ler, e as propostas
dele passam pelo mesmo verificador que as da pessoa e as do acaso.

### Graduação (fundamentos)

**P1. Por que não usar `double` em tudo e acabar com o problema de precisão?**
Precisão e reprodutibilidade são problemas diferentes. `double` reduz o erro,
mas a ordem das somas continua mudando o último bit entre máquinas. O vapor
fixa a ordem (reprodutibilidade) e calcula um envelope rigoroso (precisão). Em
dinheiro, nem `double` basta: o rateio por piso perde um centavo; por isso
inteiros com escala decimal (§5.2).

**P2. O que é exatamente um "certificado"?**
Um objeto que acompanha a resposta e é conferido por um procedimento mais
simples que o que a produziu: a alocação (conferida pelo verificador extraído),
os preços de estado (conferidos por multiplicação), o diário (refeito pelo motor
ingênuo). Na compilação, é o documento assinado com os *hashes* e as evidências
de cada degrau (§3.6).

**P3. Por que Elixir e não Rust ou C++?**
Porque o plano de controle precisa de supervisão e isolamento de falhas, e a
BEAM os dá de graça; o código quente não roda em Elixir — roda em bits emitidos,
em processos separados. O custo é que o compilador é mais lento do que seria em
Rust, o que aparece na compilação do Monte Carlo (0,6 s por passo e ISA).

### Mestrado (mecanismos)

**P4. Os 16 acumuladores fixos não desperdiçam desempenho em máquinas largas?**
Em uma linha, sim; mas o paralelismo vem de processar R linhas por iteração,
escolhido pelo alocador por ISA. O GEMV f32 2048² escala 2,24× com 2 *threads*
sem mudar um bit (Tabela 13). Onde há perda real é no GEMV de 4 bits, limitado
por instruções — e isso está medido e explicado pelo terceiro teto (§3.7).

**P5. Se o verificador de alocação é provado, o que garante que o código emitido faz o que a KIR diz?**
Nada provado: a codificação é **testada** contra o binutils instrução a
instrução, e a execução contra o oráculo (degraus 4 e 5). A monografia classifica
isso como "testado", não "provado" (Figura 5), e coloca a ligação com uma
semântica formal de ISA como questão aberta (§3.8).

**P6. A invariância de prefixo é completa?**
Não. É correta (nunca acusa um *pipeline* causal) e completa só se todos os
cortes forem testados; com oito cortes, uma antecipação que só se manifeste entre
dois cortes passa. O custo de cortes adicionais é linear (Proposição 5.2 e o box
de mestrado da §5.7). O exercício 20 pede exatamente esse contraexemplo.

**P7. Por que aritmética racional no LP? Não é lento?**
É mais lento, mas a decisão de arbitragem é instável em ponto flutuante perto da
fronteira (um custo de −10⁻¹⁷ é arbitragem ou arredondamento?). Em racionais, a
resposta é exata e conferível por terceiros com inteiros. Coeficientes
irracionais, como e^(−rT), entram arredondados a 15 casas — e a resposta diz isso.

**P8. Como você sabe que o juiz do livro é independente se você escreveu os dois?**
Independência de **algoritmo e estrutura de dados**, não de autor: lista e
ordenação contra mapas ordenados e filas. O defeito do FOK mostra o limite — os
dois compartilharam uma leitura errada da especificação — e por isso há
invariantes que não executam nada. O próximo passo, na lista de pendências, é o
juiz em outra linguagem (Python).

**P9. Os controles não são escolhidos para passar?**
São escolhidos como o erro mais plausível (Itô esquecido, dias corridos,
normalização pela amostra inteira) e publicados ao lado de cada verificação, para
serem contestados. É uma ameaça de validade de construto declarada (§7.6). Três
verificações não têm controle numérico, e a tabela diz quais e por quê.

**P10. O Wichmann–Hill é aceitável em 2026?**
Para precificação com milhares de trajetórias, sim; para criptografia ou cauda
extrema, não — falha na BigCrush e tem período ~7·10¹². A escolha foi forçada
pela exatidão em binary32. O substituto (Philox) exige multiplicação inteira de
32/64 bits na álgebra — trabalho futuro nº 2.

### Doutorado (questões abertas e posicionamento)

**P11. Qual é a contribuição científica, e não de engenharia?**
Três afirmações testáveis: (i) reprodutibilidade bit a bit entre ISAs e GPU é
obtível a custo baixo fixando a ordem e tirando paralelismo de saídas
independentes; (ii) verificar uma fase do compilador com um verificador provado
é suficiente para dar garantias úteis sem provar o compilador; (iii) com
aritmética determinística, propriedades como ausência de antecipação viram
**certificados exatos** em vez de testes estatísticos. A terceira é a mais nova.

**P12. Como isso se compara ao CompCert?**
O CompCert prova o compilador inteiro, com garantias muito mais fortes; o vapor
usa a mesma ideia de validar a alocação *a posteriori* (Rideau & Leroy 2010) num
compilador de tensores pequeno, com cinco ISAs, execução isolada e
co-assinatura. Não compete em garantias; ocupa outro ponto do espaço (§8.3).

**P13. O que impede essas garantias de valerem num cluster heterogêneo de verdade?**
A eclusa de substratos: um acelerador só recebe programas canônicos depois de
medido. O que não foi feito é medir em silício real — RVV, ARM e GPU discreta
foram validados sob QEMU e lavapipe. É a principal ameaça à validade externa.

**P14. Existe uma teoria para os "controles"?**
Não ainda; a monografia a propõe como questão aberta (box de doutorado da
§4.9): uma noção de controle adversarial mínimo por classe de erro, com cotas
sobre a probabilidade de um defeito passar. O corpus de 170 pares
(verificação, controle) é um ponto de partida empírico.

**P15. Por que não fazer o motor de ofertas em nanossegundos?**
Porque o objetivo desta rodada era verificabilidade, e ela foi atingida a
~8,5 µs por evento com o SHA-256 incluído. O caminho de baixa latência — o
motor como programa do worker, com diário e juiz fora do caminho crítico — é o
trabalho futuro nº 1 e o exercício 24.

**P16. O que acontece com as garantias se o Unicode do OTP mudar?**
O texto normalizado muda; a versão é registrada no *digest* dos agentes e
testada, mas não eliminada. Está nas limitações (§3.5 e ARCHITECTURE §10).

### Perguntas difíceis (e como não se perder)

**"Isso não é só um produto de engenharia muito bem feito?"** — "A engenharia é
a evidência; a tese é que essas propriedades são baratas quando decididas cedo.
A prova disso é que treze rodadas, de LLMs a finanças, não exigiram mudar o
núcleo — e que o custo medido foi pequeno ou nulo."

**"Os números vêm de uma máquina de 2 vCPUs; por que eu deveria acreditar?"** —
"As afirmações centrais não são de desempenho absoluto, são de igualdade de bits
e de certificados, que não dependem da máquina. As de desempenho são relativas e
reproduzíveis pelo Apêndice A. Concordo que escala a muitos núcleos não foi
medida."

**Se não souber:** "Não sei; o que eu sei é X, e o experimento que responderia é
Y." Nunca improvisar um número.

---

## 3. Corte curto (20 minutos)

Slides **1, 2, 4, 6, 8, 11, 14, 17, 26, 31, 32, 33, 35, 38** — um minuto e
meio cada, mais a abertura e o fechamento. Os outros ficam como apêndice para
perguntas.

## 4. Demonstração ao vivo (opcional, 3 minutos, só se a banca pedir)

1. `mix vapor.serve` e abrir `http://127.0.0.1:8000/#fin` (console em português).
2. *Backtest* → "Uma espiada no amanhã (lead)": mostrar o selo com o **dia 89**.
3. *Mesa de operações* → *Livro de ofertas*: clicar numa entrada do diário e
   mostrar a prova de Merkle e o feed ITCH.
4. Plano B sem rede ou sem servidor: as figuras 9–14 e 16–18 da monografia são capturas
   desse mesmo console.

## 5. Checklist da véspera

- [ ] `mix vapor.quality --only round13` verde (≈ 25 s) — rodar na manhã da defesa.
- [ ] PDF da monografia e dos slides abertos, com cópia local fora da rede.
- [ ] Este roteiro impresso, com os tempos marcados.
- [ ] Saber de cor: 3 010 MB / 1,8 MB; 82 teoremas; 71,7 ms vs 72–75 ms;
      6–23×; DSR 0,10; dia 89; o defeito do FOK a mercado; 170 verificações com controle.
- [ ] Saber dizer em uma frase cada limitação do Cap. 9.
