# Proteínas — as métricas da predição de estrutura, dobramento por contatos, contatos pela evolução

> Pedido (0.12): "expandir capacidades […] similar ou superior a alpha
> fold (com comparações vs o próprio alpha fold ou seu equivalente open
> source)". Escrutínio: [DIRETRIZ.md §15](DIRETRIZ.md).

## 1. O que os preditores modernos fazem, e o que se pode fazer aqui

Os preditores de estrutura de ponta (o AlphaFold 2 e seus equivalentes
abertos OpenFold e ESMFold) leem uma sequência e um alinhamento múltiplo
(ou um modelo de linguagem de proteínas, no ESMFold) com redes treinadas
no PDB inteiro, e chegam a GDT-TS ~90 em alvos do CASP14. Sem esses
pesos, sem as bases e sem GPU, **prever uma estrutura nova a partir só
da sequência não é possível aqui** — e não é fingido.

O que é possível, e foi feito, é a cadeia de raciocínio que esses
sistemas automatizam, cada elo com seu teste e seu controle:

1. **as métricas** com que o campo inteiro julga uma predição — TM-score
   (igual ao TM-align), RMSD após superposição ótima, GDT-TS/HA, lDDT;
2. **dobrar** a partir de contatos — geometria de distâncias, a ideia que
   precede as redes (e que o AlphaFold 1 usava, com potenciais
   aprendidos);
3. **ler contatos da evolução** — acoplamento direto (DCA) num
   alinhamento, a fonte de sinal que as redes aprenderam a explorar;
4. o **pipeline** inteiro, sobre um alinhamento amostrado de um modelo
   plantado numa proteína real, de modo que a verdade é conhecida.

`Vapor.Bio.{Structure, Coevolution, Align}` · console *Simular →
Proteínas*.

## 2. Métricas (`Vapor.Bio.Structure`)

PDB lido (Cα, cadeias, modelos de RMN), superposição ótima pelo método
dos quatérnions de **Horn**, TM-score com d₀ = 1,24·∛(L − 15) − 1,8 e
busca por fragmentos-semente como no TM-align, GDT-TS e GDT-HA, lDDT
(Cα, raio 15 Å, quatro limiares). Verificado: TM-score e RMSD **iguais
aos do TM-align** (pacote `tmtools`) a 10⁻⁶ entre os modelos de RMN de
1LCD; cópia girada e transladada dá RMSD 0, TM 1, lDDT 1.

## 3. Dobrar por contatos

Limites de distância (contato < 8 Å, ligações Cα–Cα 3,8 Å, cotas
inferiores suaves para não contatos, restrições de hélice a partir da
estrutura secundária), caminhos mínimos para completar as cotas,
imersão pela decomposição espectral da matriz de Gram, refinamento por
gradiente, várias partidas; a **quiralidade** é decidida pela mão das
hélices (α-hélices são destras) — a imagem especular é rejeitada por
isso. Verificado em 1A8O (domínio C-terminal do capsídeo do HIV-1, 70
resíduos, raios X): do mapa de contatos verdadeiro, **TM > 0,75**; a
imagem especular (controle) < 0,4.

## 4. Contatos pela coevolução (`Vapor.Bio.Coevolution`)

Um modelo de Potts com acoplamentos plantados nos contatos de 1A8O é
amostrado por Gibbs (2000 sequências); os contatos são lidos por
informação mútua (MI, com correção APC) e por **DCA de campo médio**
(norma de Frobenius + APC). Precisão dos k primeiros (k = número de
contatos verdadeiros com |i − j| ≥ 6): **DCA 0,96**, MI 0,89; o
alinhamento embaralhado por coluna (controle, que preserva a
conservação e destrói a coevolução) cai ao acaso (0,02).

## 5. O pipeline

alinhamento → DCA → contatos → geometria de distâncias, com a estrutura
secundária dada: **TM 0,69, GDT-TS 0,69, lDDT 0,68** em 1A8O; dos contatos
do alinhamento embaralhado (controle), TM 0,20.

## 6. Comparação

| | aqui | AlphaFold 2 / OpenFold | ESMFold |
|---|---|---|---|
| entrada | alinhamento (aqui: **amostrado de um modelo plantado**) + estrutura secundária | sequência + MSA real + moldes | só a sequência |
| sinal de contato | DCA de campo médio | Evoformer (atenção sobre MSA e pares), aprendido | modelo de linguagem de proteínas |
| geometria | geometria de distâncias + refinamento | módulo de estrutura (IPA), átomos completos | idem |
| saída | traço Cα | todos os átomos + confiança (pLDDT, PAE) | idem |
| qualidade típica | TM 0,69 em 1A8O com MSA planejado | GDT-TS ~90 (CASP14, mediana) | um pouco abaixo do AF2, muito mais rápido |
| métricas | **as mesmas**, conferidas contra o TM-align | — | — |

Igual: as métricas (conferidas), a cadeia lógica e a verificabilidade de
cada elo. Inferior, e muito: predição a partir de sequência real, átomos
completos, confiança calibrada.

## 7. Alinhamento (`Vapor.Bio.Align`)

Needleman–Wunsch e Smith–Waterman com lacunas afins (Gotoh), BLOSUM62;
pontuações **iguais às do `PairwiseAligner` do Biopython** (lacunas 11/1)
e a recuperação do alinhamento atinge a pontuação.

## 8. Limites honestos

- O MSA do pipeline é **sintético**, amostrado de um modelo cujo grafo de
  acoplamentos é o mapa de contatos verdadeiro; a estrutura secundária é
  tirada da estrutura nativa. Num alinhamento real (Pfam), DCA de campo
  médio tem precisão bem menor e precisaria de pesos de sequência e
  pseudocontagens ajustadas.
- Só Cα; sem cadeias laterais, sem energia física, sem predição de
  estrutura secundária a partir da sequência.
- Até 120 resíduos no console (o tempo do dobramento cresce como L³).
