# Qualidade das saídas: sinal ou ruído?

> Um teste de qualidade que nunca foi visto reprovando ruído não testa nada.

## 1. A dor

Em quase todo repositório de inferência, "funciona" significa "gerou alguma
coisa". O modelo de demonstração do próprio vapor (`mix vapor.demo_model`,
pesos aleatórios) gera `" scen scen authorised authorised scen …"` e todos os
testes passavam. Paridade bit a bit prova que dois substratos concordam — não
que o que eles concordam em produzir seja sinal. Na indústria, o mesmo
problema aparece como regressões de tokenização, de *chat template*, de
quantização ou de *kernel* que deixam a saída plausível à primeira vista e
degradada de fato; na academia, como métricas sem linha de base nem controle.

## 2. O método

### 2.1 Portões calibrados que se recusam a existir

`Vapor.Quality.Gate` não tem limiar escolhido à mão. Cada portão é **ajustado a
controles**:

- **negativos** (devem reprovar): ruído branco; o próprio sinal **embaralhado**
  (mesma distribuição marginal, nenhuma estrutura — o controle mais difícil);
  repetição degenerada;
- **positivos** (devem passar): sinal real **retido**.

`t_ruído` = o pior negativo; `t_natural` = o pior positivo. Se `t_ruído ≥
t_natural`, a calibração devolve `{:error, {:inseparable, …}}` e **não há
portão** — por exemplo, amostras curtas demais para separar texto de letras
embaralhadas. Entre os dois limiares fica `:structured`: mais estrutura do que
qualquer ruído, menos do que todo sinal real. "Não é ruído" = `:structured` ou
`:natural`.

| modalidade | escore principal | por quê |
|---|---|---|
| texto | fração de trigramas de bytes presentes num corpus de referência | bytes aleatórios quase nunca acertam; letras embaralhadas acertam algo; texto real acerta a maioria |
| texto (colapso) | razão de compressão zlib da própria amostra | laços degenerados comprimem demais |
| imagem | correlação de vizinhos (lag 1, luminância) | imagens naturais ≈ 0,8–0,99; ruído e pixels embaralhados ≈ 0 |
| áudio | planura espectral (entropia de Wiener) | ruído branco ≈ 1; tons → 0 |

Mais medidas estão disponíveis (inclinação espectral 1/f, PSNR, SSIM, SNR,
frequência dominante, compressão condicional ao corpus, entropia, UTF-8).

### 2.2 Modelos plantados: verdade em forma fechada

`Vapor.Quality.Planted.bigram/3` escreve — não treina — os pesos de um decoder
Llama de modo que ele **seja exatamente** o bigrama contado de um corpus:
embedding one-hot, todos os ramos de atenção e MLP nulos (somam +0 ao fluxo
residual), cabeça = `log P(j|i)/s` com `s` o que a RMSNorm faz de `e_i`. O
checkpoint é um `config.json` + pesos comuns; passa pela eclusa, compilador,
escada e substrato como um modelo real, e **tem de reproduzir a tabela
analítica** — medido: |Δ| ≤ 1,2·10⁻⁶ nos logits, bits/caractere idênticos aos
da tabela até 10⁻⁹. Qualquer defeito na pilha vira um desvio mensurável.

### 2.3 Para um checkpoint real

`mix vapor.quality --model CAMINHO` (`Vapor.Quality.Model`) exige **dois**
testes independentes:

1. **bits por byte** num texto retido (*teacher forcing* em janelas), contra o
   unigrama de bytes de um corpus de referência e contra o uniforme (8);
2. **gerações pelo portão de texto** calibrado.

Medido no modelo de demonstração (Qwen2, vocabulário real de 151 936 tokens,
pesos aleatórios): 5,59 bits/byte contra 5,10 do unigrama → **ruído**; 2 de 3
gerações reprovadas. Num bigrama de bytes plantado com o tokenizador BPE de
bytes: 4,01 bits/byte < 5,05 → **sinal**, gerações aprovadas. Os dois casos
estão nos testes.

## 3. Achados honestos (medidos, não presumidos)

- **O portão de trigramas sozinho é enganável por vocabulários BPE grandes.**
  Tokens aleatórios de um vocabulário real já são subpalavras reais
  ("actualizar_standard bart…"), e uma das três gerações do modelo aleatório
  passou como `:structured`. Por isso o veredito sobre checkpoints exige bits
  por byte também; foi esse critério que reprovou o modelo.
- **UTF-8 estrito é rígido demais para modelos de bytes**: um bigrama de bytes
  corta um caractere multibyte de vez em quando. O portão exige ≥ 95 % dos
  bytes dentro de caracteres válidos — ainda pega *detokenização* quebrada.
- **Vocabulário acolchoado**: modelos reais têm matriz de embedding maior que o
  tokenizador (Qwen2: 151 936 vs. 151 646). IDs além do tokenizador contam
  contra a amostra em vez de derrubar o teste.
- **O portão de texto mede estatística da língua, não significado.** O texto
  do bigrama plantado é `:structured` — e é isso que deve ser. Para significado
  é preciso tarefa com resposta exata (a tradução cor ↔ nota, com acerto 1,0).
- **TIES e DARE pioram a fusão de modelos densos** (ver [FUSAO.md](FUSAO.md)).

## 4. A suíte

`mix vapor.quality` (alguns minutos no worker nativo, a maior parte nos dados reais) roda tudo e escreve
[bench/QUALITY.md](bench/QUALITY.md), `bench/quality.json` e a galeria
`bench/modal/` (PNG ×8, WAV). Sai com status 1 se qualquer verificação falhar
— o CI pode condicionar o *merge* a "as saídas são sinal". As verificações
(55 na 0.7.0):

- portões calibrados com margem positiva (texto, colapso, imagem, áudio);
- modelo plantado = tabela; bits abaixo do unigrama; gerações aprovadas; **as
  mesmas gerações com pesos aleatórios reprovadas** (o controle);
- dez rotas any-to-any em entradas retidas, cada uma com controle
  ([ANY_TO_ANY.md](ANY_TO_ANY.md));
- fusão: a linear vence o especialista errado em cada domínio, `merge(A, A) =
  A` e `slerp(t = 0) = A` bit a bit, recibos verificáveis;
- substratos: todo programa modal nativo = oráculo bit a bit;
- dados reais (0.5.0): OCR, fala, caligrafia nos dois sentidos, fusão de
  transformers treinados — cada um com o seu controle;
- **rodada 0.6** (`Vapor.Quality.Round06`, §5b do relatório): cada recurso
  novo contra a sua verdade **e** contra um controle que mostra que a
  verificação discrimina — experts esparsos = denso (controle: um
  especialista escolhido perturbado muda os bits); MLA latente = expandido
  (controle: projeção permutada); janela = atenção sobre a janela movida
  (controle: atenção completa); `÷` = IEEE (controle: o antigo `a·rcp(b)`
  erra ~30 %); convolução = binary64 direta (controle: kernel espelhado,
  correlação no lugar de convolução); log de transparência nas sondas do
  transparency-dev (controle: verificador ingênuo); Mamba nativo = oráculo
  (controle: sem estado); especulação em árvore = gulosa (controle: rascunho
  aceito sem conferir).
- **rodada 0.7** (`Vapor.Quality.Round07`, §5c): CCITT = libtiff (controle:
  a codificação declarada errada); páginas escaneadas lidas em ordem
  (controle: sem a ordem, 53 % de CER); modelo de língua (controle: o mesmo
  corpus embaralhado não ganha nada); **a abstenção do modelo** (controle: sem
  a guarda ele reescreve 27 de 30 linhas aleatórias); códigos e valores não
  pioram; formatos gerados válidos pelos *parsers* do OTP (controle: o padrão
  ingênuo de data gera 145 inválidas em 150); fusão do disco com a mesma
  raiz e um décimo da memória (controle: outro `t` muda a raiz).
- **rodada 0.10** (`Vapor.Quality.Round10`, §5f): a eclusa de substratos
  (controle: um motor bf16 simulado, recusado com 8 bits medidos); o
  envelope com DAZ (controle: o dobro do valor exato, fora); o modelo
  treinado contra Witten–Bell e contra o próprio texto embaralhado; o fluxo
  sem fim contra posições crescendo; o pêndulo de primeira ordem; o caos
  bit a bit entre substratos (controle: um ulp); a identificação do gêmeo
  (controle: medidas embaralhadas no tempo); o carro-pêndulo (controle:
  política nula); o alarme do gêmeo (controle: sem falha); as redes contra
  os seus nulos; os gráficos (controle: rótulos permutados, **todos
  recusados**); as fórmulas (controle: leitura plana); CJK (controle:
  caracteres aleatórios — o modelo de língua não pode ajudar, e em japonês e
  coreano tem de ajudar); árabe e cirílico (controle: o leitor latino).
- **rodada 0.11** (`Vapor.Quality.Round11`, §5g, 26 verificações): redes de
  ordenação nos ótimos conhecidos (controle: sorteadas e podadas); o truque
  de bits mínimo (controle: a fórmula ingênua estoura); o posto 7 exato
  (controle: o posto 6 nunca); 15 teoremas e gêmeos falsos pelas duas vias;
  conjecturas (controle: nenhuma trinca trivial); homologia sobre ℚ contra
  GF(2); a persistência de um laço contra uma mancha; 11 experimentos de
  ciência, cada um contra a sua referência e o seu controle; o agente de
  autojogo contra o jogo perfeito (controle: a busca sem treino); mundos
  variados contra um mundo; a cena, o esqueleto, a direção (controle: uma
  palavra sem sentido é relatada), o esboço (controle: sem restrições,
  continua torto), a planta; e o arquivo (controle: um byte trocado e
  re-zipado).
- **rodada 0.12** (`Vapor.Quality.Round12`, §5h, 27 verificações, ~6 s):
  Dormand–Prince contra RK4 de passo fixo; Robertson nos valores de
  Hairer & Wanner; ordem 2 de Crank–Nicolson contra a ordem 1 de Euler
  implícito por solução manufaturada; unidades incoerentes recusadas antes
  de rodar; a lata ótima com limites (controle: sem limites, "diverged");
  trapézios contra Euler no RC; Newton contra Gauss–Seidel no sistema de
  Stagg; 10 elementos contra 1 nos modos; QM6 contra o Q4 que trava;
  invariantes estequiométricos contra uma espécie que não se conserva;
  Fenske contra o refluxo abaixo do mínimo; S(3) com testemunha e DRUP
  (controle: a prova truncada); Knuth–Bendix contra os axiomas só
  orientados; Tales contra a variante falsa; a proposta certa contra a
  trocada; perft do xadrez e do shogi, posições legais do Go; CFR+ contra
  o jogo uniforme; o alinhamento planejado contra o embaralhado (precisão e
  dobra); a fornalha contra o estimador viciado; a direção com nomes contra
  a frase sem ninguém. Também em `rodada12_test.exs`.

## 5. Limites

- Os mundos e modelos plantados são **instrumentos**: provam que a pilha
  carrega sinal, não que um modelo de produção é bom. Para isso: o
  `--model` com texto retido na língua do modelo, e avaliações de tarefa.
- O portão de imagem é calibrado em cenas sintéticas 16×16; imagens reais
  pedem recalibração com fotos retidas (a API é a mesma).
- A avaliação de bits por byte de vocabulários grandes roda o log-softmax na
  BEAM (≈ 6 min para 1500 bytes com 151 936 tokens); levá-lo ao substrato é
  um item aberto.
