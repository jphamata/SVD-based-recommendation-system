# Ciência: da mecânica quântica ao genoma, contra referências (0.11)

> Pedido: "simulação física (física em suas diversas frentes, desde
> clássica, relativística, quântica, tokamak, química (baterias, moléculas,
> materiais e além), biologia (mutações, genômica, algo similar a
> AlphaFold e além))". Escrutínio: [DIRETRIZ.md §14](DIRETRIZ.md).
> Testes: `science_test.exs`. Console: *Simular → Ciência*. A física
> clássica (pêndulos, caos, gêmeos digitais) é a da 0.10:
> [FISICA.md](FISICA.md).

> **Desde 0.14.0** estes experimentos fixos aparecem no console como **Calibração**: são a
> calibração dos instrumentos, com respostas conhecidas. Para pôr o *seu* sistema na bancada —
> com evidência que não precisa de gabarito — use o [Crucible](CRUCIBLE.md).

## 1. A regra

Cada experimento tem uma **referência** — forma fechada ou valor
publicado — e, onde existe um bom, um **controle**: o que uma
implementação errada ou ingênua produziria, e que tem de falhar. Nem todo
"controle" da tabela tem a mesma força, e isso é dito: o integrador de
Euler, o Laplaciano cartesiano, os sítios embaralhados e as conformações
aleatórias são **implementações erradas rodadas** e reprovadas; a
partícula clássica do tunelamento é uma **previsão calculada** (a parte
do pacote acima da barreira), não uma simulação; a deriva da norma do
estado coerente é uma **conservação**, que um esquema unitário com a
física errada também manteria; E×B e HeH⁺ não têm controle (—). Tudo em binary64 na BEAM, determinístico:
os mesmos números em qualquer máquina, cada um uma função dos seus
parâmetros (`Vapor.Science.run/1`), salvável e recalculável como arquivo.

## 2. Os experimentos

| frente | experimento | método | referência | controle | medido |
|---|---|---|---|---|---|
| quântica | estado coerente do oscilador | Schrödinger 1-D por *split-step* de Fourier (Feit, Fleck & Steiger 1982), FFT radix-2 própria | ⟨x⟩(t) = x₀ cos t | a deriva da norma (unitariedade) | erro máx. **4,0·10⁻⁵**; norma a 3·10⁻¹⁴ |
| quântica | tunelamento | pacote abaixo da barreira | ∫T(k)\|φ(k)\|²dk, T exato da onda plana | a partícula clássica: só passa a parte do pacote acima da barreira | **0,5431** contra **0,5445**; clássico 0,0013 |
| relatividade | giro a 0,9 c | empurrador de Boris (o integrador dos códigos PIC) sobre u = γv | período 2πγ/B (γ = 2,294) | Euler explícito ganha energia a cada volta | 14,41464 contra 14,41462; \|u\| a 2·10⁻¹⁵; Euler +6,7 % |
| plasma | deriva E×B | Boris com E ⊥ B | v = E/B | — | 0,2992 contra 0,3 |
| fusão | equilíbrio de um tokamak | Grad–Shafranov (Δ*ψ = −μ₀R²p′ − FF′) por diferenças finitas conservativas e SOR | a solução exata de **Solov'ev** (p′, FF′ constantes) | o Laplaciano cartesiano (sem o termo 1/R da geometria toroidal) | fluxo a 2·10⁻¹² (o esquema não tem erro de truncamento nesses polinômios); o **eixo magnético** (vértice de uma parábola pelo máximo da grade) converge como h² (1,25·10⁻³ → 3,1·10⁻⁴) — a taxa é do localizador, não do esquema; o controle erra 8·10⁻³ e não converge |
| química | H₂ | Hartree–Fock restrito, STO-3G, integrais em forma fechada (função de Boys por `erf`) — Szabo & Ostlund §3.5 | **−1,1167** hartree (R = 1,4 bohr) | — | **−1,11671**; energias orbitais −0,578 e 0,670 (as do livro) |
| química | HeH⁺ | o mesmo | **−2,860662** hartree | — | **−2,860659** |
| química | o fracasso conhecido | H₂ separado (R = 10) | dois átomos de H: −0,9332 | — | RHF fica **0,34 hartree acima** — um determinante só não descreve dois elétrons separados: por isso existe interação de configurações |
| materiais | líquido de Lennard-Jones | 64 átomos, caixa periódica, velocity Verlet | conservação da energia; o primeiro pico de g(r) perto de 2^{1/6}σ | Euler explícito | flutuação **5·10⁻⁴** em 300 passos; pico de g(r) em 1,11σ; Euler explode (10³³) |
| evolução | fixação de um mutante | Wright–Fisher haploide, 4 000 réplicas com sorteios por contador | a probabilidade **exata** da cadeia de Markov (resolvida) | o mutante neutro fixa a 1/N | 0,0955 contra 0,0941 (Kimura: 0,0958); neutro 0,019 contra 0,02 |
| genômica | árvore refeita de genomas | Jukes–Cantor ao longo de uma árvore conhecida; distâncias corrigidas; *neighbour joining* | Robinson–Foulds 0 | os sítios de cada sequência embaralhados | RF **0**; distâncias a 3,3 %; controle RF 6 (o máximo) |
| proteínas | dobramento HP (Dill 1985) | Monte Carlo com pivôs, cantos e pontas, recozimento | o ótimo publicado do 20-mero clássico: **−9** (Unger & Moult 1993); num 12-mero, a enumeração exata | conformações aleatórias | **−9**; 12-mero igual à enumeração (−5); aleatórias −1,3 |

## 3. O que foi achado no caminho

- **Precedência**: em Elixir, `-(x - x0) ** 2` é `(x0 - x)²`, não
  `−(x − x0)²` — o pacote gaussiano nasceu como uma exponencial crescente.
  A referência (o estado coerente) pegou na primeira execução.
- **A barreira desalinhada da grade** (5,1 células para uma largura de 1)
  dava 0,447 contra 0,544: a barreira agora tem exatamente a/dx células.
- **A mutação** de Jukes–Cantor com `|> min(2)` no lugar errado sorteava
  bases que às vezes eram a mesma: as distâncias saíam 40 % curtas. A
  conferência das distâncias contra os comprimentos verdadeiros pegou.

## 4. O que não é, e por quê

- **AlphaFold**: prever estruturas reais de proteínas exige o modelo
  treinado e as suas bases (PDB, alinhamentos múltiplos) — nada disso
  existe aqui. O modelo HP é o brinquedo físico que torna o problema de
  busca exato; dito como tal.
- **Baterias e materiais reais**: DFT de sólidos (ondas planas,
  pseudopotenciais) e química de eletrólitos estão muito além do que uma
  rodada honesta mede; o Hartree–Fock de moléculas de dois elétrons e o
  líquido de Lennard-Jones são os primeiros degraus, conferidos.
- **Tokamak**: o equilíbrio, não a estabilidade nem o transporte; o
  Solov'ev é o caso de teste padrão dos códigos de equilíbrio.
- **Relatividade geral**: fora.
- Nada aqui é um programa vapor ainda (binary64 na BEAM, determinístico,
  mas não "os mesmos bits em todo substrato" do compilador): o *split-step*
  é linear (DFT como `linear`) e é o candidato natural para virar programa.
