# Engenharia — o texto do engenheiro, um certificado independente do solver

> Pedido (0.12): "resolver dores reais de engenharia elétrica, mecânica,
> química e civil". Escrutínio: [DIRETRIZ.md §15](DIRETRIZ.md).

A dor comum às quatro engenharias não é a falta de solver — é **confiar**
no número que ele devolve. Cada ferramenta aqui lê o texto que o
engenheiro já escreveria (uma netlist, uma lista de barras, nós e barras,
uma malha, tubos, reações) e devolve, ao lado da resposta, um
**certificado calculado de forma independente**: a lei física
re-avaliada sobre a solução, não o resíduo interno do método.

`Vapor.Engineering.*` · console *Resolver → Engenharia* · MCP `engineering_run`.

## 1. Elétrica

### Circuitos (`Vapor.Engineering.Circuit`) — um subconjunto do SPICE

```
* um diodo polarizado por um divisor
V1 in 0 DC 5
R1 in a 1k
R2 a 0 2k
R3 a d 1k
D1 d 0 IS=1e-14
.op
```

Elementos R C L V I D E G F H O (amp-op ideal); valores SPICE (`1k`,
`2.2u`, `10meg`); fontes `DC`, `AC`, `SIN(…)`, `PULSE(…)`, `PWL(…)`;
análises `.op` (Newton com *pnjlim* e *source stepping*), `.dc`, `.ac`
(fasores complexos), `.tran` (trapézios; `.method be`). MNA.

**Certificado**: a lei de Kirchhoff das correntes reavaliada em cada nó
a partir das tensões e dos modelos (não do sistema linear), o balanço de
potência (Σ fornecida = Σ dissipada), saídas de amp-op dentro do trilho.
**Avisos** nomeiam nós sem caminho DC para a terra; um laço de fontes de
tensão é recusado como singular.

Verificado: a equação de Shockley satisfeita no ponto resolvido; RC por
trapézios **ordem 1,97**, Euler implícito (controle) **0,99**; RLC série —
corrente de pico em 1/2π√LC, tensão no capacitor de pico em
ω₀√(1 − 1/2Q²) com o valor exato Q/√(1 − 1/4Q²); amplificador inversor
−R2/R1 a 10⁻¹².

### Transistores (0.13)

```
* um estágio fonte comum e um inversor CMOS
VDD vdd 0 DC 5
VG g 0 DC 1.5 AC 1
RD vdd d 10k
M1 d g 0 0 NMOS KP=50u VTO=0.7 LAMBDA=0.02 W=10u L=1u
Q1 c b e NPN IS=1e-15 BF=150 BR=2
```

`M d g s [b] NMOS|PMOS` é o **MOSFET nível 1** (Shichman–Hodges: corte,
triodo e saturação com modulação de canal λ; dispositivo simétrico — o
menor potencial entre dreno e fonte age como fonte); `Q c b e NPN|PNP` é o
**bipolar de Ebers–Moll** em forma de transporte (`IS`, `BF`, `BR`), as
junções limitadas como as do diodo. Cada transistor entra no MNA como um
elemento de três terminais cuja corrente depende das três tensões; as
condutâncias de Newton saem de diferenças centrais com h = 10⁻⁷ V (erro
~(h/V_T)² ≈ 10⁻¹¹), e um GMIN de 10⁻¹² S atravessa o canal ou a junção,
como no SPICE, para que um dispositivo cortado não deixe nó flutuando. O
bipolar são **dois ramos** (coletor → emissor e base → emissor): a
corrente de emissor é a soma, e o certificado de Kirchhoff — que reavalia
cada corrente pela equação do próprio dispositivo — continua valendo sem
mudar uma linha.

Verificado **contra o ngspice 42** (`engenharia_test.exs`, quando o
binário está presente): o fonte comum (3,2945736 V no dreno), o estágio
emissor comum com divisor e resistor de emissor (V_C, V_B, V_E iguais na
6ª casa — com `.temp`/`TNOM` iguais aos 300,00 K daqui; com o TNOM padrão
de 27 °C o ngspice reescala IS e a diferença de 2,4 mV no coletor é a
física do modelo, não um erro) e o inversor CMOS (3,1436422 V); o ganho de
pequenos sinais do fonte comum em `.ac` igual ao do ngspice a 10⁻⁶. À mão:
a lei quadrática Id = β/2·(V_GS − V_t)²·(1 + λV_DS) fecha a 3·10⁻¹² A — a
corrente do GMIN.

### Fluxo de potência (`Vapor.Engineering.Power`)

```
base 100
bus 1 slack V=1.06
bus 2 pq P=20 Q=20
line 1 2 r=0.02 x=0.06 b=0.06
shunt 4 b=0.15
```

Newton–Raphson em coordenadas polares (barras slack, PV, PQ, taps,
shunts); Gauss–Seidel como controle. **Certificado**: o desbalanço máximo
em cada barra recomputado pelas admitâncias e o balanço de potência
ativa (geração = carga + perdas).

Verificado no sistema de cinco barras de **Stagg & El-Abiad** (1968): as
tensões e ângulos publicados a 6·10⁻⁴ pu e 6·10⁻³ °, a potência da barra
slack 129,59 MW, desbalanço < 10⁻¹⁰ em 4 iterações; Gauss–Seidel chega
às mesmas tensões em 119.

## 2. Civil e mecânica

### Pórticos e treliças 2-D (`Vapor.Engineering.Structure`)

```
node 1 0 0
node 2 3[m] 0
support 1 fixed            # fixed | pinned | roller-x | roller-y | combinações x y rz
beam 1 2 E=200[GPa] A=0.01 I=1e-4 rho=7850 n=8
truss 2 3 E=200e9 A=1e-3
load 2 fy=-10[kN]          # fx fy mz
udl 1 2 w=-5[kN/m]
modes 3
```

Rigidez direta com subdivisão de barras (`n=`), cargas distribuídas por
forças nodais equivalentes, **massa consistente** e o problema de
autovalores generalizado para os modos, rotações de nós só de treliça
restringidas automaticamente (e relatadas), **mecanismo recusado**,
numeração por **Cuthill–McKee reverso** e Cholesky em banda.
**Certificado**: equilíbrio global — Σ cargas + Σ reações = 0 em força e
momento, relativo à maior carga.

Verificado: balanço PL³/3EI e momento de engaste PL **exatos**; viga
biapoiada 5wL⁴/384EI e wL²/8; treliça pelo método dos nós; frequências do
balanço contra Euler–Bernoulli (1,8751² e 4,6941² √(EI/ρAL⁴)) a 10⁻⁶ com
10 elementos — um elemento só (controle) erra 4,8·10⁻³.

### Estado plano de tensão (`Vapor.Engineering.FEM`)

```
plate x=0..10 y=-0.5..0.5 nx=20 ny=4
material E=1000 nu=0.3 t=1        # strain para estado plano de deformação
element qm6                       # q4 | qm6 (modos incompatíveis, padrão)
fix x=0
traction x=10 ty=-1
```

Q4 e **QM6** (Wilson–Taylor, modos incompatíveis condensados), tensões
nodais e von Mises, banda reduzida por RCM. **Certificado**: resíduo
K·u − f e equilíbrio das reações.

Verificado: **patch test** passado por Q4 e QM6 em elementos distorcidos
(deslocamento 10⁻¹⁴, tensão 10⁻¹²); na placa esbelta em flexão o QM6 fica a
**0,9 %** da viga de Timoshenko, o Q4 (controle) **trava** e erra 29 %.

### Redes de tubulação (`Vapor.Engineering.Pipes`)

```
reservoir R head=60[m]
junction A elev=10[m] demand=15[L/s]
pipe 1 R A L=800[m] D=250[mm] eps=0.1[mm] K=2
```

Newton global em vazões e cargas (método do gradiente de Todini–Pilati),
Colebrook–White resolvido por Newton (laminar 64/Re), perdas localizadas.
**Certificado**: continuidade em cada nó, energia em cada **malha**
(achadas por árvore geradora), caminhos entre reservatórios.

Verificado: um tubo entre dois reservatórios igual ao `brentq` do
**SciPy** a 10⁻¹⁰; rede com duas malhas fechando continuidade a 10⁻¹⁸ e
energia a 10⁻¹².

## 3. Química

### Cinética e reatores (`Vapor.Engineering.Process.reactions/1`)

```
A + B -> C ; k = 2
C <-> D ; kf = 1, kb = 0.2
2 A -> E ; k = 0.1
A0 = 1; B0 = 0.8
t = 0 .. 20
reactor cstr tau=5     # batch (padrão) ou CSTR com feed
```

Ação das massas → EDOs (com a troca automática para Rosenbrock da
bancada). **Invariantes**: o espaço nulo à esquerda da matriz
estequiométrica, em aritmética racional exata — as quantidades
conservadas saem **só da estequiometria** e são conferidas ao longo da
solução.

Verificado: A → B → C contra a forma fechada de Bateman a 10⁻⁸; a rede
acima tem exatamente 2 invariantes com deriva < 10⁻⁹ ([A] sozinho, o
controle, muda 0,93); CSTR no estado estacionário C₀/(1 + kτ).

### Flash e destilação (`Vapor.Engineering.Process.flash/1`, `distill/1`)

Flash isotérmico por **Rachford–Rice** com Antoine (pressões de bolha e
orvalho, uma fase relatada quando é o caso); certificado: balanços de
massa, Σx = Σy = 1, resíduo de Rachford–Rice. Coluna binária por
**McCabe–Thiele** (estágios, prato de alimentação), **Fenske**,
**Underwood** e **Gilliland**; controle: com refluxo 10⁴ o degrau de
McCabe–Thiele dá exatamente ⌈Fenske⌉, e um refluxo abaixo do mínimo é
recusado.

## 4. Limites honestos

- Circuitos: sem modelos de transistor (BJT/MOSFET), sem ruído,
  sem `.subckt`. O diodo é Shockley com resistência série zero.
- Fluxo de potência: equilibrado, monofásico equivalente; sem limites de
  reativos nas barras PV, sem curto-circuito.
- Estruturas: 2-D, linear, pequenos deslocamentos; sem flambagem, sem
  não linearidade geométrica ou material; MEF só em quadriláteros e
  retângulos estruturados.
- Tubos: regime permanente; sem bombas nem válvulas controladas.
- Processos: idealidade (Raoult), destilação binária a α constante.
