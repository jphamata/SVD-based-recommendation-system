# ZK e FHE — escrutínio da proposta e o que foi construído

A proposta (anexo de 2026-10-01) pede para transformar o vapor “no
compilador de ponta para zkML e fheML”, num catálogo de dez itens e uma
tabela de prioridades. Como a própria diretriz está sujeita a escrutínio,
este documento separa três coisas: o que é **fato**, conferido com fonte;
o que é **analogia**, às vezes útil e às vezes enganosa; e o que foi
**construído e testado** aqui. As fontes estão no fim.

## 1. A pergunta de primeiros princípios

**O que uma prova de conhecimento zero acrescenta ao que o vapor já dá?**

O vapor já torna qualquer saída **re-derivável**. O recibo (`x-vapor-receipt`)
liga modelo, prompt, parâmetros e saída, e quem tem os pesos refaz a conta
e obtém os mesmos bits em qualquer substrato. Isso é verificabilidade por
reexecução. Uma prova ZK acrescenta exatamente duas coisas, e só elas
justificam o custo:

1. **Privacidade de uma das partes**: a entrada (ou os pesos) não é revelada.
2. **Verificação sucinta**: verificar custa muito menos do que reexecutar,
   por exemplo num contrato inteligente.

Quando nenhuma das duas é necessária, uma prova ZK é custo sem benefício.
O recibo já basta.

**O que FHE acrescenta?** Uma propriedade que nenhuma outra peça do vapor
dá: o servidor computa sobre dados que **nunca vê**. A avaliação
homomórfica, porém, é aritmética **inteira modular exata**. Ela já é
determinística por natureza, e por isso o diferencial central do vapor
(bits idênticos em todo substrato apesar do ponto flutuante) pesa pouco
ali. O que o vapor pode oferecer ao FHE é outra coisa: kernels certificados
(NTT), aproximações polinomiais com erro **provado** e a disciplina de
recusar o que não sabe garantir.

## 2. Veredito por item

| item da proposta | veredito | por quê |
|---|---|---|
| 1.1 “otimizar a emissão RVV para SP1/RISC Zero” | **premissa falsa** | As zkVMs executam **RV32IM**: inteiro de 32 bits, sem vetores nem ponto flutuante (especificação da RISC Zero). O vapor emite RV64GCV com RVV. Seria um backend **novo** (RV32IM, sem `V`, com f32 emulado em software, o que é caríssimo dentro de uma prova). A ideia que sobra é boa e está na direção certa: **separar testemunha e prova**. Ela foi feita, mas por outro caminho (§3.2). |
| 1.2 aritmetização R1CS/Plonkish de `add`, `mul`, `fma`, `linear` | **feito para o fragmento inteiro** | Ponto flutuante **não** é aritmética de corpo. Cada arredondamento f32 exige decomposição em bits e prova de intervalo, centenas de restrições por operação. Já o `gemm_i8` do vapor é aritmética inteira, e a ausência de *overflow* dele está **provada em Lean**. Esse é o fragmento que vira restrições baratas. |
| 1.3 corpos BabyBear, Goldilocks, BN254 | **feito (referência exata)** | `Vapor.Field`, com primalidade, 2-adicidade e geradores conferidos. Kernels SIMD com Montgomery/Barrett são trabalho futuro e terão esta referência como oráculo. |
| 1.4 provar em Lean “Circuit(x,y)=1 ⟺ y=Oracle(x)”, “eliminaria 100% dos riscos” | **parcialmente feito; a promessa é exagerada** | Provado: o lema que liga o limite inteiro à igualdade no corpo (§3.3). A correção do *circuito* em relação à especificação é uma parte do risco. O resto continua de fora: a segurança do sistema de prova, a cerimônia de *setup*, os bugs do provador e do verificador, e a pergunta “estes são os pesos certos?”. Nenhuma prova de circuito elimina 100 % disso. |
| 1.5 verificador Solidity “< 200k gas” | **feito e medido: a promessa não se sustenta com o verificador padrão** | O verificador Groth16 exportado pelo snarkjs, implantado numa EVM em memória, gastou **209 922 gás de execução**, cerca de 237 k com a transação, para 3 entradas públicas. O piso dos *precompiles* é ≈ 181 k + 6 k por entrada pública. Abaixo de 200 k só com verificador enxuto e poucas entradas. |
| 2.1 CKKS/BFV como termos da álgebra | **não agora** | Implementar um esquema de criptografia sem auditoria e sem a escolha de parâmetros do *Homomorphic Encryption Standard* seria um brinquedo perigoso. Bibliotecas maduras (OpenFHE, Lattigo, TFHE-rs) existem. O lugar do vapor é **abaixo** delas (kernels, aproximações), não no lugar delas. |
| 2.2 NTT em AVX-512 e RVV | **referência feita; kernel é o próximo passo** | `Vapor.NTT` (cíclica e negacíclica, `Z_q[X]/(Xᴺ+1)`) bate com o produto ingênuo em quatro corpos. É o **denominador comum** de FHE e de provadores STARK, logo o investimento de kernel com maior alcance. A frase “90 % do tempo é NTT” varia com o esquema e a operação; não foi medida aqui. |
| 2.3 envelope de Wilkinson = orçamento de ruído | **analogia, não reuso** | O envelope limita erro de arredondamento de forma **determinística**. O ruído CKKS nasce **aleatório**, da cifragem, e se analisa por cotas de pior caso (muito pessimistas) ou de caso médio (heurísticas). O que se reaproveita é a *infraestrutura* de propagar cotas pelo DAG de termos, com funções de transferência novas. |
| 2.4 ativações por Chebyshev; “Canon já decompõe em polinômios” | **feito, com certificado; a premissa sobre o `Canon` é falsa** | As funções canônicas usam comparação e seleção (`sel`), redução de faixa e Newton. **Não** são polinômios puros. Aproximar por Chebyshev é o passo certo. O que faltava, e foi feito, é **provar** o erro (§3.4). |
| 2.5 *bootstrapping* inserido pelo *cut sweep* | **analogia** | O *cut sweep* corta regiões por pressão de registradores. Posicionar *bootstrapping* é gestão de níveis (o problema de compiladores como EVA, HECATE ou Fhelipe). O formalismo de cortes no grafo pode inspirar; não é o mesmo algoritmo. |
| 3 zkFHE para 70 B | **pesquisa, não roteiro** | Ver §4 para a ordem de grandeza: só provar a inferência em claro de um modelo de 13 B já leva minutos de GPU. |

## 3. O que foi construído

### 3.1 Corpos e NTT (`Vapor.Field`, `Vapor.NTT`)

BabyBear (2³¹ − 2²⁷ + 1), Goldilocks (2⁶⁴ − 2³² + 1), o escalar do BN254 e
998 244 353 = 119·2²³ + 1. Os testes conferem:
- primalidade (Miller–Rabin), 2-adicidade e que o gerador é não-resíduo, o
  que implica que a raiz 2ˢ-ésima tem ordem exata;
- que a NTT inverte e que produtos NTT = produtos ingênuos, cíclicos e
  **negacíclicos** (o anel do BFV, do CKKS e do ML-KEM);
- que o mergulho com sinal é injetor abaixo de p/2 e colide logo acima.

### 3.2 Inferência inteira como R1CS (`Vapor.ZK`)

Uma rede int8 (`linear → ReLU → linear`) é compilada para R1CS sobre o
BN254. A **testemunha da primeira camada é calculada pelo kernel
`gemm_i8` certificado** do vapor, em qualquer substrato e com os mesmos
bits; o circuito só a confere. Essa é a separação testemunha/prova que o
item 1.1 buscava, sem precisar de uma zkVM. Os arquivos saem nos formatos
binários do iden3 (`.r1cs` e `.wtns`), aceitos pelo ecossistema
circom/snarkjs.

Três escolhas de desenho:
- **Pesos públicos são constantes do circuito.** Multiplicar por constante
  é grátis em R1CS (faz parte da combinação linear). Uma camada densa
  oculta **não custa restrição nenhuma**, e a última custa uma por saída.
  O custo da prova está nas não-linearidades (o ReLU custa uma decomposição
  em bits, `B + 3` restrições por neurônio) e na checagem de faixa das
  entradas (9 por entrada int8 privada). É o **inverso** da estrutura de
  custo da CPU.
- **A entrada privada é int8 porque o circuito diz.** Cada `x + 128` é a
  soma de 8 bits booleanos. A primeira versão não tinha essa checagem, e a
  revisão independente mostrou o efeito: o provador podia escolher
  qualquer valor de corpo para `x`, alcançar qualquer ativação oculta e,
  com ela, qualquer saída. O enunciado provado agora é exatamente
  **∃ x ∈ int8ᵏ : modelo(x) = y**.
- **Pesos privados não são oferecidos.** Com pesos como testemunha
  privada, a prova diz apenas que *existem* pesos que produzem a saída.
  Isso é inútil sem um compromisso, dentro do circuito, com um modelo
  publicado. E mesmo um compromisso diz *quais* pesos, não que eles são o
  modelo anunciado: o ataque *Hollow-LLM* (2026) mostra pesos “ocos” que
  passam na verificação de um modelo maior. Constantes no circuito
  amarram o modelo: a chave de verificação deriva do circuito, e
  `ZK.digest/1` nomeia os pesos.
- **Recusa do que o corpo não representa sem ambiguidade.** O compilador
  calcula as cotas exatas e recusa em dois casos: quando um valor
  intermediário pode chegar a p/2, e quando a decomposição de um ReLU
  precisaria de mais bits do que o corpo distingue (`2^(B+1) > p`, caso em
  que um segundo representante do mesmo valor também caberia). No BN254,
  contrações int8 ficam muito longe dos dois limites. No BabyBear, uma
  contração int8 com K ≈ 70 000, ou um ReLU sobre valores perto de 10⁹,
  já são recusados (testado).

Números desta máquina (`zk_test`, nível `:snarkjs`):
- 16 → 8 → 3, int8 com entrada privada: **311 restrições, 304 fios**
  (144 delas são a checagem de faixa das 16 entradas);
- `snarkjs wtns check` aceita a testemunha;
- prova Groth16 gerada e verificada, e a mesma prova com uma saída
  forjada é rejeitada;
- o verificador Solidity compila (1 721 bytes de EVM) e gasta **209 922 gás
  de execução** (236 582 com a transação), medido numa EVM do ethereumjs;
- mudar o valor de qualquer fio isolado quebra alguma restrição, e uma
  entrada fora de int8 não tem testemunha. Isso é teste empírico, não
  prova. A garantia formal de não-sub-restrição é a do enunciado acima:
  fixadas as entradas, cada fio interno é determinado (bits únicos abaixo
  de p, ReLU e camadas lineares como funções dos fios anteriores). Entradas
  diferentes com a mesma saída continuam possíveis, e isso é próprio do
  enunciado “existe x”.

### 3.3 O lema em Lean (`proofs/Vapor/FieldEmbedding.lean`)

```
embed_injective   : p ∣ a − b, |2a| < p, |2b| < p  ⟹  a = b
field_parity      : |dot| ≤ K·A·B, 2·K·A·B < p, p ∣ z − dot, |2z| < p  ⟹  z = dot
machine_field_agree : integer_parity ∧ field_parity  ⟹  máquina de 32 bits = inteiros = corpo
```

Os enunciados acima estão em notação matemática. Em Lean, cada
`|2x| < p` é a conjunção `-p < 2 * x ∧ 2 * x < p`.

Junto com o teorema de paridade inteira já existente, isso fecha a cadeia
*kernel certificado → inteiros → corpo*. As hipóteses do lema (entradas
int8, cotas abaixo de p/2) são **fatos do circuito**: a checagem de faixa
garante a primeira, e o compilador recusa o que violaria a segunda. Por
isso, para uma dada entrada, o valor que o worker calcula é o que o
circuito impõe. O núcleo do Lean continua sem
Mathlib, sem `axiom`, sem `sorry`, agora com 45 teoremas, e o módulo
extraído foi regenerado com o novo digest.

### 3.4 Polinômios com erro provado (`Vapor.Poly`)

Sob CKKS, toda não-linearidade vira polinômio. A prática comum é aproximar
e **medir** o erro em amostras. `Vapor.Poly` aproxima por Chebyshev e
**prova** o erro em aritmética racional exata:
- avalia `p` exatamente numa grade diádica;
- cerca `f` com a exponencial corretamente arredondada do `Vapor.CR`;
- entre pontos da grade, usa o resto de interpolação `h²/8 · sup|e''|`,
  com `sup|p''|` limitado pela desigualdade de Markov.

| função | intervalo | grau | erro **provado** | erro observado | profundidade |
|---|---|---|---|---|---|
| sigmoide | [−8, 8] | 7 | 2,9642·10⁻² | 2,9640·10⁻² | 3 |
| sigmoide | [−8, 8] | 15 | 1,388·10⁻³ | 1,382·10⁻³ | 4 |
| tanh | [−4, 4] | 15 | 2,776·10⁻³ | 2,765·10⁻³ | 4 |
| exp | [−1, 1] | 8 | 1,47·10⁻⁸ | 1,22·10⁻⁸ | 4 |

A cota é válida em **todo** o intervalo, não só nas amostras, e fica a no
máximo 1,21× do erro observado. A grade é de 4 096 pontos, exceto na
linha da exp, que usa 32 768 porque uma aproximação muito precisa precisa
de grade mais fina para o termo de resto não dominar. A “profundidade” é
⌈log₂(d+1)⌉, o mínimo de níveis para multiplicar cifras entre si. Uma
avaliação CKKS real costuma gastar mais um nível com as constantes.

## 4. Ordem de grandeza (para calibrar a ambição)

- **zkLLM** (CCS 2024) prova a inferência do LLaMA-2 13B em **803 s** numa
  A100. A prova tem 188 kB e a verificação leva 3,95 s. O compromisso dos
  pesos leva 986 s e ocupa 11 MB.
- **Hollow-LLM** (2026) mostra que uma prova ZK de inferência certifica
  coerência com pesos comprometidos, não que o modelo comprometido é o
  anunciado.
- **Groth16 na Ethereum**: o piso imposto pelo custo dos *precompiles*
  (EIP-1108) é cerca de 181 k + 6 k·ℓ gás, com ℓ entradas públicas. O
  verificador do snarkjs mediu 210 k de execução aqui.

“Plataforma definitiva de IA confidencial e verificável do mundo” não é uma
afirmação que este repositório possa sustentar. O que ele sustenta, com
testes, é menor e mais útil: um caminho exato, provado e de ponta a ponta,
de um kernel int8 certificado até uma prova Groth16 verificável na EVM, e
as peças de base (corpos, NTT, polinômios com erro provado) sobre as quais
o resto pode ser construído sem perder a garantia.

## 5. Próximos passos, em ordem

1. **Kernel NTT certificado** (AVX-512, NEON, RVV, SPIR-V), com
   `Vapor.NTT` como oráculo bit a bit. É o maior alcance: FHE e STARK.
2. **Requantização** (`s32 → s8`) como gadget e como termo da álgebra. Ela
   permite redes inteiras mais profundas com testemunha certificada em
   todas as camadas.
3. **Backend PLONK/STARK** (em vez de Groth16, que exige *setup* por
   circuito) e um compromisso Poseidon que ligue o circuito ao digest do
   certificado do modelo.
4. **Cotas de ruído BFV de pior caso** propagadas pelo DAG, a parte honesta
   do item 2.3.
5. RV32IM como alvo de zkVM só se um caso concreto pedir. As restrições
   diretas (§3.2) são mais baratas que emular uma CPU.

## Fontes

- RISC Zero, *zkVM Technical Specification* — “The zkVM implements the RV32IM instruction set”: https://dev.risczero.com/api/zkvm/zkvm-specification
- Sun et al., *zkLLM: Zero Knowledge Proofs for Large Language Models* (CCS 2024): https://arxiv.org/pdf/2404.16109
- Gong, Liu, Li, *Hollow-LLM Attack: Computationally Trivial Weights in Zero-Knowledge Verification of LLM Inference* (2026): https://arxiv.org/html/2607.28884v1
- Nebra, *Groth16 Verification Gas cost*: https://hackmd.io/@nebra-one/ByoMB8Zf6
- iden3, formatos binários `r1cs`/`wtns` (circom/snarkjs), conferidos pelo próprio `snarkjs r1cs info` e `wtns check` nos testes.
