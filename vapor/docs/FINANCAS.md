# Finanças e a mesa de operações — cada número com o que permite julgá-lo

> Pedido (0.13): "suporte a finanças, HFT, ataque as limitações atuais e
> TODO […] testes de qualidade para garantir que as respostas não sejam
> apenas ruído". Escrutínio: [DIRETRIZ.md §16](DIRETRIZ.md).

Finanças é o domínio em que "o número saiu do computador" vale menos. Um
preço que muda quando o cálculo muda de máquina, um backtest que olhou o
amanhã sem querer, a melhor de cinquenta estratégias apresentada como se
fosse a única, um livro de ofertas que ninguém de fora consegue auditar —
são as dores reais da indústria, e todas são dores de **verificabilidade**,
não de velocidade. O vapor já tinha a resposta de princípio (uma busca
propõe, um verificador decide; os mesmos bits em todo substrato); esta
rodada a aplica ao mercado.

`Vapor.Finance.run/2` (a mesa) · console *Mercados → Finanças* e
*Mercados → Mesa de operações* · MCP `finance_run` e `arbitrage_check` ·
`mix vapor.finance KIND ARQUIVO` · testes `financas_test.exs` (com
QuantLib, SciPy e simplefix como oráculos externos) · §5i de
[bench/QUALITY.md](bench/QUALITY.md).

| tarefa | o que entra (texto) | o certificado que sai |
|---|---|---|
| calendário e dinheiro | `du`, `holidays`, `adjust`, `yf`, `allocate`, `factor`… | feriados por regra (cômputo), conferidos contra o QuantLib; rateio que soma exatamente |
| curva | DI1, LTN, NTN-F, depósitos, títulos, swaps | todo instrumento reprecificado; forwards negativos apontados |
| opções | `price`, `iv`, `american`, `heston`, `smile` | paridade; gregas contra diferenças finitas; limites de não arbitragem **antes** de resolver; g(k) de Durrleman |
| Monte Carlo | S, K, T, r, σ, trajetórias, barreira | bits = oráculo exato e = outra contagem de threads; intervalo de 99 % contra a forma fechada |
| risco | série de P&L, método de VaR | Kupiec, Christoffersen, zona de Basileia |
| carteira | retornos de ativos | KKT da variância mínima; contribuições de risco |
| backtest | dados, varredura, sinal, custo | **quatro portões**: invariância de prefixo, Sharpe deflacionado, PBO, Reality Check |
| arbitragem | estados, calls, câmbio | **ou** o portfólio de arbitragem **ou** os preços de estado — em racionais exatos |
| livro de ofertas | ordens como texto | diário SHA-256 + raiz de Merkle; motor ingênuo independente; feed ITCH; FIX |
| sessão de bolsa | agentes e semente | os três certificados + a mesma cabeça de hash ao repetir |
| microestrutura | Hawkes, Avellaneda–Stoikov, Almgren–Chriss | teste de reescala do tempo; o artigo reproduzido; forma fechada = ótimo numérico |

## 1. Dinheiro exato

`Vapor.Finance.Money`: um valor é um inteiro e uma escala decimal
(`%{c: 12345, e: 2}` = 123,45). Soma, subtração e produto são exatos; todo
arredondamento é um ato explícito com um modo nomeado (`half_even`,
`half_up`, `half_down`, `down`/`truncate`, `up`, `floor`, `ceiling`) —
uma única primitiva, `div_round/3`, testada nos empates dos dois lados do
zero.

- `allocate/3` rateia pelo **maior resto** (Hamilton): R$ 100,00 em três
  partes dá 33,34 + 33,33 + 33,33, e a soma é o total **por construção**;
  o certificado diz também que cada parte está a menos de uma unidade da
  sua fração exata. O controle da §5i: o piso em binary64 perde um
  centavo (99,99).
- `factor_252/3` calcula (1 + r)^(du/252) — o fator da convenção
  brasileira — com uma raiz inteira em aritmética de inteiros grandes e
  **trunca** na 8ª casa como a ANBIMA. (1,1365)^(1/252) = 1,00050788…:
  o Python `decimal` com 60 dígitos concorda; a decisão na 8ª casa não é
  um acidente de ponto flutuante.

## 2. Calendários e contagem de dias

`Vapor.Finance.Calendar`: cada feriado é uma **regra** — data fixa,
n-ésimo dia da semana, ou festa móvel pelo cômputo gregoriano (Páscoa
pelo algoritmo anônimo de 1876; Carnaval = Páscoa − 48 e − 47; Sexta-feira
Santa − 2; Corpus Christi + 60). Os fechamentos que nenhuma regra produz
(11 de setembro de 2001, furacão Sandy, dias de luto presidenciais) são
dados, cada um com o motivo.

| calendário | regras | conferência |
|---|---|---|
| `:anbima` (= B3, DU/252) | nacionais + móveis; 20 de novembro desde 2024 (Lei 14.759/2023) | **903 feriados em dia útil, 1990–2078, idênticos ao `ql.Brazil(Settlement)`** |
| `:nyse` | datas observadas (sábado → sexta, domingo → segunda; 1º de janeiro de sábado **não** recua), MLK (1998–), Juneteenth (2022–), fechamentos especiais | **848 idênticos ao `ql.UnitedStates(NYSE)`** |
| `:target` | TARGET2: Sexta Santa, Segunda de Páscoa, 1º de maio e 26 de dezembro desde 2000 | **401 idênticos ao `ql.TARGET()`** |

Na primeira comparação o NYSE divergiu num único dia — 27/04/1994, o luto
pelo presidente Nixon — e o TARGET em datas anteriores a 2000 (quando o
sistema não existia e o QuantLib só fecha o 1º de janeiro): as duas
diferenças viraram dados e regras, e o teste agora exige igualdade dia a
dia por 89 anos.

Contagens (`year_fraction/4`): `:bus252` (DU/252), `:act360`, `:act365f`,
`:thirty360` (base de títulos dos EUA, ISDA 4.16(f)), `:thirty_e360`,
`:act_act_isda` — todas iguais ao QuantLib a 10⁻¹⁴ nos casos de borda
(fim de fevereiro, dia 31, anos bissextos). `adjust/3` (following,
modified following, preceding…), `add_business_days/3`, `di1_maturity/1`
(F26 → primeiro dia útil de janeiro de 2026).

## 3. Curvas

`Vapor.Finance.Curve` lê os instrumentos como uma mesa os escreve:

```
date = 2025-01-02
calendar = anbima
basis = du252
interpolation = flat_forward
di1 F26 = 15.02%
ltn 2028-01-01 price = 652.30
ntnf 2031-01-01 price = 800.00
swap 5y = 4.10%
fit nss
```

O *bootstrap* põe um nó no último fluxo de cada instrumento e acha (Brent)
o fator de desconto que o reprecifica, com os nós anteriores fixos. A
interpolação é `flat_forward` (log-linear no desconto — a convenção da
curva DI) ou `linear_zero`. Os fluxos seguem as convenções: DI1 com PU
100 000/(1 + r)^(DU/252); LTN de face 1 000 pagando no dia útil seguinte;
NTN-F com cupom semestral 1 000·(√1,1 − 1) em 1º de janeiro e 1º de julho;
títulos com juros acumulados para preço limpo; swaps par de curva única.

**O certificado** reprecifica cada instrumento a partir da curva pronta e
dá o pior erro relativo (4,5·10⁻¹⁶ na curva DI do exemplo); lista os
forwards entre nós e **aponta os negativos** — o exemplo "cotação
inconsistente" (janeiro a 15 %, julho a 9 %) é reprecificado mas sai com o
forward negativo nomeado. O controle da §5i: o mesmo DI1 contado em dias
corridos erra o PU em 2,5·10⁻³. `fit nss` ajusta Nelson–Siegel–Svensson
(β exatos por mínimos quadrados numa grade de τ, depois Nelder–Mead).
Depósitos simples conferem com o `PiecewiseLogLinearDiscount` do QuantLib
a 10⁻¹³ (`financas_test.exs`).

## 4. Opções

`Vapor.Finance.Options`:

| função | modelo | certificado |
|---|---|---|
| `bsm/7`, `greeks/7` | Black–Scholes–Merton com dividendos | paridade (7·10⁻¹⁵); cada grega contra a diferença central (< 2·10⁻⁸ relativo) |
| `black76/6`, `bachelier/6` | futuros (lognormal / normal: taxas negativas, spreads) | paridade |
| `implied_vol/8` | inversão por Brent em σ ∈ [10⁻⁶, 10] | os **limites de não arbitragem conferidos antes**: um preço abaixo do intrínseco descontado é recusado com o limite; o erro de reprecificação; a vega e quanto um *tick* de preço move σ |
| `binomial/10` | Cox–Ross–Rubinstein e Leisen–Reimer (Peizer–Pratt 2), europeia e americana | a europeia da árvore contra a BSM |
| `heston/7` | Heston pela integral única de Lewis e a função característica "little trap" de Albrecher | paridade; o limite ξ → 0 = BSM em √v₀ |
| `svi_fit/2`, `svi_arbitrage/2`, `svi_calendar/3` | SVI bruto | g(k) ≥ 0 de Durrleman numa grade fina (borboleta); variância total crescente em T (calendário); a densidade neutra ao risco |
| `static_arbitrage/3` | sem modelo: calls entre strikes | monotonicidade, inclinação em [−e^(−rT), 0], convexidade — cada violação devolvida como o portfólio que a explora, com o *payoff* conferido em todo vértice |

**Contra o QuantLib** (`financas_test.exs`): BSM e Δ, Γ a 10⁻¹², vega a
10⁻¹⁰; americana de Leisen–Reimer com 801 passos a 10⁻¹⁰ (4,486076);
Heston a 10⁻⁹ (o integrador de Gauss–Kronrod adaptativo do QuantLib e a
Gauss–Legendre composta daqui concordam até a 9ª casa). O CRR daqui é o
de livro-texto (p = (e^{rΔ} − d)/(u − d)); o do QuantLib usa a
aproximação aditiva em log — diferem em 1,6·10⁻⁵ e o documento diz qual é
qual.

A put americana da tabela 1 de Longstaff & Schwartz (S = 36, K = 40,
σ = 0,2, T = 1, r = 6 %): árvore 4,4861, LSM daqui 4,4788 ± 0,0143 (20 000
trajetórias antitéticas, regressão em {1, x, x²}); o artigo dá 4,472 para o
LSM. A fatia SVI atribuída a Vogt em Gatheral & Jacquier (2014) — o
exemplo clássico de arbitragem de borboleta — é detectada (g mínimo
−0,033 em k ∈ [0,645; 1,255]); um sorriso calmo ajustado sai com g ≥ 0.

## 5. Monte Carlo compilado para o worker — os mesmos bits em toda máquina

A dor: um número de risco que muda quando o cálculo muda de máquina, de
contagem de threads ou para uma GPU é um achado de risco de modelo
esperando para acontecer (reduções paralelas em outra ordem, uma FMA aqui
e não ali, outro `exp`). `Vapor.Finance.MonteCarlo` escreve o passo da
trajetória como **termos da álgebra do vapor** e o compila para o worker
nativo:

- o **gerador** é Wichmann–Hill (1982) dentro do programa: três geradores
  de Lehmer cujos produtos ficam abaixo de 2²³, de modo que cada
  atualização de estado é **exata em binary32**; o módulo é feito com o
  truque de arredondamento 2²³ e um passo de correção — toda a geração
  roda no worker, a BEAM só semeia;
- o desvio normal é a AS241 de Wichura (PPND7, 7 dígitos: o que binary32
  comporta), com `log` e `rsqrt` canônicos;
- por passo: L ← L + μΔ + σ√Δ·Φ⁻¹(u); A ← A + e^L; G ← G + L; viva ← viva·[L > ln(B/S₀)].

**Certificados**, todos na mesma resposta: (i) a primeira chamada,
compilada para 64 pistas e rodada pelo **oráculo exato**, igual bit a bit
às 64 primeiras pistas do worker; (ii) a simulação inteira repetida num
worker com **2 threads**: os mesmos bits; (iii) as mesmas uniformes em
binary64 na BEAM (Φ⁻¹ de precisão total): diferença máxima 2·10⁻⁵ no
*payoff*; (iv) o intervalo de 99 % contra a forma fechada — BSM para a
europeia, Kemna–Vorst para a asiática geométrica discreta; (v) a asiática
aritmética com a geométrica como **variável de controle** (variância ÷
1 300). O **controle**: esquecer o −σ²/2 de Itô dá z = 8,7 — o estimador
viciado é pego, não promediado.

Medido (Xeon 2 vCPUs, AVX-512, um núcleo do worker): 8 192 trajetórias ×
64 passos em 125 ms contra 750 ms estimados para a BEAM em binary64 (6×);
16 384 × 32 em 86 ms contra 1 950 ms (23×). A compilação custa ~0,6 s por
passo desenrolado e por ISA (o verificador de alocação extraído do Lean é
quadrático); por isso um passo por chamada e só a ISA da máquina
(`all_targets: true` compila as quatro).

Limites ditos: Wichmann–Hill é um gerador antigo (período ~7·10¹², falha
baterias modernas como a BigCrush); serve a precificação, não a
criptografia nem estudos de cauda extrema — `rng: :host` usa splitmix64 na
BEAM e compara.

## 6. Risco de mercado

`Vapor.Finance.Risk`: VaR e ES histórico, normal, Cornish–Fisher e EWMA
(λ = 0,94); previsões de um passo em janela móvel; **backtest** com
Kupiec (proporção de falhas, χ²₁), Christoffersen (independência, χ²₁; e
cobertura condicional, χ²₂) e a zona de Basileia (verde 0–4, amarela 5–9,
vermelha ≥ 10 exceções em 250 dias a 99 %). Um teste de risco que nunca
rejeita é tão suspeito quanto um que sempre rejeita: o **tamanho** do
Kupiec é medido (exceções sorteadas a 1 % exato: rejeita em 1,5–10 % das
400 séries) e o seu **poder** também — num t de Student com ν = 3, o VaR
normal é rejeitado (p = 0,038) e o histórico não (p = 0,43).

Carteiras: covariância de Ledoit–Wolf (alvo μI), **variância mínima só
compra** por conjunto ativo com as condições KKT como certificado
(estacionaridade < 10⁻¹², multiplicadores com o sinal certo), **paridade
de risco** por Newton na formulação convexa de Spinu (contribuições iguais
aos orçamentos a 10⁻¹²), HRP de López de Prado.

## 7. Backtests com portões de ruído

Um backtest é uma máquina de produzir índices de Sharpe; tente variantes
bastantes e uma parecerá boa sobre puro ruído. As duas falhas silenciosas
da indústria são **antecipação** (o sinal usou o que não teria) e
**seleção** (a melhor de muitas tentativas apresentada como a única).
`Vapor.Finance.Backtest`:

```
data = ar1 n=5040 phi=0.15 sigma=0.01 seed=3
sweep k = 1..3
signal = sign(sma(ret(close), k))
cost = 1bp
```

A linguagem do sinal tem operadores causais (`lag`, `diff`, `ret`, `sma`,
`ema`, `std`, `zscore`, `rmax`, `rmin`, `rsi`, `sign`, `clip`, `if`…) e
aceita, porque as pessoas os escrevem, três que espiam (`lead`, `center`,
`normalize` com a amostra inteira). A posição em (t, t+1] é o sinal em t;
o custo incide sobre |Δposição|.

1. **Invariância de prefixo — o certificado de ausência de antecipação.**
   O sinal é recalculado sobre históricos truncados `close[0..c]` em oito
   pontos de corte e tem de ser **igual, bit a bit**, ao calculado com a
   história inteira até c. É um teste de caixa-preta do pipeline todo: não
   precisa confiar nos operadores. Uma espiada (`lead`), uma normalização
   pela amostra inteira (`center`), um erro de digitação: pegos, com o
   **dia** em que o sinal muda. (Um primeiro rascunho da ideia — conferir
   só os operadores — teria deixado passar o `center`, que é causal em
   cada passo e não causal no todo; por isso o teste é sobre o resultado,
   não sobre a gramática.)
2. **Sharpe deflacionado** (Bailey & López de Prado 2014): a
   probabilidade de o Sharpe verdadeiro exceder o máximo esperado de N
   tentativas sem habilidade, corrigida por assimetria e curtose; N inclui
   as tentativas declaradas antes do arquivo (`trials =`).
3. **PBO por CSCV** (Bailey, Borwein, López de Prado & Zhu 2017): as
   12 870 metades de 16 blocos; com que frequência o campeão dentro da
   amostra cai abaixo da mediana fora dela.
4. **Reality Check de White** com o bootstrap estacionário de
   Politis–Romano.

Medido (§5i): o melhor de 30 cruzamentos de médias sobre um passeio
aleatório tem Sharpe positivo e **DSR 0,10** — reprovado; o sinal plantado
(momento sobre AR(1) com φ = 0,15) passa os quatro portões (DSR 1,0; PBO <
0,5; RC p < 0,05); `sign(lead(close) − close)` tem Sharpe anual 20 e é
pego no dia 89.

## 8. Arbitragem decidida exatamente

O teorema fundamental da precificação é um teorema da alternativa — o
lema de Farkas: **ou** um portfólio custa nada (ou menos) e paga algo
sem nunca perder, **ou** existem preços de estado estritamente positivos
que reproduzem toda cotação dentro do seu bid–ask. Nunca os dois, nunca
nenhum. `Vapor.Finance.Arbitrage` acha os dois lados com o **simplex
racional exato** de `Vapor.Logic.LP` (duas fases, regra de Bland) e cada
resposta vem com o objeto que a prova, conferido só por multiplicação:

- uma arbitragem: o portfólio (compra no ask, venda no bid), o custo e o
  *payoff* em cada estado — no exemplo da call mal precificada, vender um
  título, comprar 1/90 da ação e vender 1/60 da call recebe 1/15 hoje e
  paga 0 nos dois estados;
- sem arbitragem: o vetor ψ > 0 com bid ≤ Σ Xψ ≤ ask — (1/2, 9/20) no
  binomial do exemplo.

Para calls entre strikes, os estados são 0, cada strike, 2·K_max e a
**inclinação** além dele: como o *payoff* de qualquer combinação é linear
por partes com vértices nos strikes, conferir os vértices e a inclinação
confere todo preço terminal — a decisão é completa para portfólios
estáticos. O fator e^(−rT) é irracional e entra arredondado a 15 casas (a
resposta diz). Câmbio: todos os ciclos simples de até cinco conversões,
o produto das taxas em racionais.

**Humano ou IA no loop**, como na lógica: `Arbitrage.check/2` e a
ferramenta MCP `arbitrage_check` recebem a *proposta* de qualquer um — um
portfólio que alguém diz ser arbitragem, ou preços de estado que alguém diz
provarem que não há — e só a aritmética exata decide. O controle do teste:
"compre a ação e venda a call" paga nos dois estados mas custa 86,5 — não
é arbitragem, e a mesa diz por quê.

## 9. O livro de ofertas como objeto verificável

`Vapor.Finance.Book`: prioridade preço–tempo; ordens limitadas e a
mercado; GTC, IOC, FOK; post-only; cancelamento; alteração (reduzir a
quantidade no mesmo preço mantém a prioridade; qualquer outra coisa é
cancelar-e-substituir); prevenção de autonegociação (`:cancel_taker`,
`:cancel_resting`, `:off`); *kill switch* por dono. Preços em *ticks*
inteiros, quantidades inteiras: nada arredonda. O motor é uma função
pura (livro, evento) → (livro, relatórios).

- **Diário encadeado**: hᵢ = SHA-256(hᵢ₋₁ ‖ canônico(evento, relatórios));
  a sessão fecha com uma **raiz de Merkle** (folhas RFC 6962), e
  `prove/2` dá a prova de inclusão de um negócio sem revelar os outros —
  o que um regulador ou um cliente pede sem acesso ao livro inteiro.
- **O juiz independente** (`Vapor.Finance.Book.Check`) não compartilha
  nada com o motor além da especificação: o livro é uma lista, a
  prioridade é uma ordenação por (preço, tempo), cada casamento varre a
  lista. Do diário sozinho ele refaz a cadeia de hashes, **reexecuta cada
  evento no motor ingênuo e exige os mesmos relatórios**, e confere os
  invariantes: preço do negócio = preço do *maker* e dentro do limite do
  agressor; o *maker* é o primeiro na prioridade naquele instante;
  quantidades conservadas (executado + repousado + cancelado = ordenado);
  livro nunca cruzado; FOK tudo-ou-nada; IOC e a mercado nunca repousam;
  post-only nunca agride; nenhum negócio entre ordens do mesmo dono sob
  prevenção.
- **Fuzzing diferencial**: 6 000 eventos aleatórios por política,
  motores idênticos. Na primeira rodada o juiz apontou `:fok_partial`:
  uma ordem **FOK a mercado** virava IOC nos dois motores (a mesma
  leitura errada da especificação escrita duas vezes) — só o invariante,
  que não executa nada, viu. Corrigido nos dois; o invariante ficou.
- Um diário adulterado é pego: um preço trocado quebra a cadeia; o mesmo
  negócio forjado **com o hash recalculado** (uma mentira coerente) passa
  na cadeia e cai na reexecução.

Medido: ~8,5 µs por evento na mediana (p99 ~100 µs) **incluindo** o
SHA-256 e a codificação canônica de cada entrada; o juiz reexecuta
12 500 eventos em ~150 ms. A BEAM é tempo real brando: estes números
servem a um *matching* de mesa, a um internalizador, a um simulador de
backtest — não a uma bolsa de nanossegundos (ver §14).

## 10. Os protocolos do mercado: ITCH 5.0 e FIX 4.4

`Vapor.Finance.Itch`: codificador e decodificador das mensagens S, R, A,
F, E, C, X, D, U, P com os comprimentos da especificação (12, 39, 36, 40,
31, 36, 23, 19, 35, 44 bytes; *big-endian*; preço com 4 casas implícitas;
*timestamp* de 6 bytes em ns), enquadramento BinaryFILE, e
`from_session/2`: o feed que uma bolsa publicaria da sessão (A quando uma
ordem repousa, E quando é atingida, X/D quando encolhe ou sai).
`rebuild/1` reconstrói o livro **só a partir do feed** — e ele tem de ser
igual ao livro do motor, nível a nível, e o volume executado também: um
segundo certificado independente (o feed só vê o que repousa; o motor vê
tudo).

`Vapor.Finance.Fix`: tag=valor com BodyLength (9) e CheckSum (10)
conferidos na leitura e calculados na escrita — **iguais ao `simplefix`**
byte a byte; NewOrderSingle (D), OrderCancelRequest (F) e
OrderCancelReplaceRequest (G) viram eventos do livro; os relatórios viram
ExecutionReports (8) com MsgSeqNum e ExecID próprios.

## 11. Risco pré-negociação

`Vapor.Finance.PreTrade` — o que a Regra 15c3-5 da SEC e o RTS 6 da
MiFID II pedem a quem tem acesso ao mercado, *e que possa mostrar que
tinha*: quantidade máxima (dedo gordo), nocional máximo, colar de preço
contra a referência (último negócio, senão o meio), posição máxima no pior
caso (executado + ordens abertas daquele lado + esta), taxa de mensagens
numa janela, *kill switch*. Uma recusa nunca chega ao livro e **entra no
diário** com a causa. `check/2` refaz posições, ordens abertas,
referências e contagens a partir do diário e confere que nenhuma ordem
aceita violou um limite **e que toda recusa teve a causa dita** — o
controle da §5i atribui a uma recusa por quantidade a causa "colar de
preço", e é pego.

## 12. Microestrutura, cada modelo com a medida que diz se serve

`Vapor.Finance.Micro`:

- **Hawkes** (chegadas que se excitam, núcleo exponencial): simulação por
  *thinning* de Ogata, máxima verossimilhança pela recursão O(n) e o
  **teste de reescala do tempo** — sob o modelo certo, os incrementos do
  compensador são Exp(1), julgados por Kolmogorov–Smirnov. Plantado
  (μ, α, β) = (1; 0,6; 1,5), ajustado (1,08; 0,61; 1,69), razão de
  ramificação 0,36 (plantada 0,40), KS p = 0,73; o **controle** — um
  Poisson ajustado às mesmas chegadas — tem p = 1,7·10⁻²⁰.
- **Avellaneda–Stoikov**: preço de reserva r = s − qγσ²(T − t), spread
  γσ²(T − t) + (2/γ)ln(1 + γ/k), simulado como no §4 do artigo (s₀ = 100,
  T = 1, σ = 2, dt = 0,005, k = 1,5, A = 140, 1 000 trajetórias) contra a
  cotação simétrica de mesmo spread. Daqui / artigo (γ = 0,1): σ(P&L)
  6,44 / 5,89 e 13,57 / 13,43; σ(q) 2,97 / 2,80 e 8,37 / 8,66; P&L médio
  65,0 / 62,9 e 68,0 / 67,2. O spread médio daqui (1,49) inclui o termo
  γσ²(T − t) que a tabela do artigo parece omitir (1,29 = só o segundo
  termo) — dito, não escondido.
- **Almgren–Chriss**: a trajetória fechada xⱼ = X·sinh(κ(T − tⱼ))/sinh(κT)
  com κ da relação discreta cosh(κτ) = 1 + κ̃²τ²/2, **certificada**
  resolvendo o mesmo problema de média-variância numericamente (um sistema
  tridiagonal): diferença relativa 10⁻¹⁶. Meia-vida 1,65 dia nos
  parâmetros do exemplo do artigo; TWAP exato com λ = 0; a fronteira
  eficiente.
- Na fita: spread de Roll, λ de Kyle (com estatística t), o gráfico de
  assinatura da variância realizada, o *microprice* de Stoikov.

## 13. Uma sessão de bolsa que se audita

`Vapor.Finance.Exchange`: formadores de mercado cotando *post-only* em
torno de um preço de reserva enviesado pelo estoque (a regra de
Avellaneda–Stoikov) sobre o preço público do passo anterior; agressores
de ruído cujas chegadas seguem um **Hawkes exato em tempo contínuo**; um
operador informado que conhece o fundamental e agride quando o livro se
afasta dele. Toda ordem passa pelo mesmo `PreTrade` e pelo mesmo `Book`
que uma implantação usaria — **o backtest é o código da bolsa**, não um
modelo dela.

A sessão devolve a própria auditoria: o diário conferido pelo motor
ingênuo, os limites conferidos a partir do diário, o feed ITCH
reconstruindo o mesmo livro, a mesma semente reproduzindo **a mesma cabeça
de hash** (outra semente, outra cabeça); e a microestrutura medida na
fita — o ajuste de Hawkes das chegadas **acha a autoexcitação plantada**
(ramificação 0,37 contra 0,40; o Poisson rejeitado), o spread de Roll
(5,66) contra o cotado (5,85), o λ de Kyle, a assinatura da variância.

## 14. O que este documento não afirma

- **Latência de HFT de bolsa.** O motor roda na BEAM: microssegundos por
  evento com hash, não nanossegundos. O que se afirma é a
  **verificabilidade** (diário, juiz independente, feed, prova de
  inclusão) e a identidade entre backtest e motor. O caminho de baixa
  latência — o motor como programa compilado do vapor ou em Zig no worker
  — está no [TODO](TODO.md).
- **Dados de mercado reais.** Nenhum arquivo ITCH da Nasdaq, nenhuma
  curva da ANBIMA foi baixada (sem rede para isso nesta máquina); os
  preços dos exemplos são ilustrativos. O formato é o da especificação e
  as convenções as publicadas; os oráculos são o QuantLib, o simplefix, o
  SciPy e o ngspice, não os dados.
- **Conselho de investimento.** Os portões dizem quando um backtest é
  indistinguível de ruído; passar neles não torna uma estratégia
  lucrativa (custos reais, capacidade, regime).
- **Modelos completos de mercado.** Sem XVA, sem modelos de taxa de juros
  de vários fatores (Hull–White, LMM), sem volatilidade local/estocástica
  calibrada a uma superfície inteira, sem crédito.

## 15. Como contestar

```sh
mix test test/vapor/financas_test.exs test/vapor/console_markets_test.exs   # QuantLib, SciPy, simplefix quando presentes
mix vapor.quality --only round13                                            # §5i: 23 verificações com controle
mix vapor.finance backtest minha_estrategia.txt                             # os quatro portões na sua estratégia
mix vapor.finance book minhas_ordens.txt                                    # diário, juiz ingênuo, ITCH, FIX
node test/js/console_markets.mjs http://127.0.0.1:8000/ /tmp/capturas       # com `mix vapor.serve`
```
