# Bancada — qualquer equação, com unidades e conferência

> Pedido (0.12): "resolver dores reais de engenharia elétrica, mecânica,
> química e civil, bem como física, química, matemática, CS e biologia,
> HPC máximo […] para problemas arbitrários e não apenas categorias
> pré-definidas". Escrutínio: [DIRETRIZ.md §15](DIRETRIZ.md).

A dor que a bancada ataca é a de quem precisa de um número confiável e
hoje tem de escolher entre uma planilha (rápida, sem unidades, sem
conferência), um script (flexível, sem garantia nenhuma) e um pacote
pesado (licença, instalação, um formulário por tipo de problema). Aqui o
problema é **escrito como no papel**, a bancada reconhece o tipo, confere
as **unidades antes de rodar** e devolve, junto de cada resposta, os
números que permitem julgá-la.

`Vapor.Solve.run/1` · console *Resolver → Bancada* · MCP `workbench_solve`.

## 1. A linguagem

Uma expressão é texto comum com multiplicação implícita (`2x`, `3(x+1)`),
funções (`sin cos tan atan sinh cosh tanh exp ln sqrt cbrt erf asinh abs
min max …`) e **grandezas com unidade** entre colchetes:

```
F = 3[kN]
L = 2.5[m]
E = 200[GPa]
I = 8.33e-6[m^4]
M = F*L in [kN*m]                 → 7.5 kN·m
d = F*L^3/(3*E*I) in [mm]         → 9.0036 mm
k = 3*E*I/L^3 in [kN/mm]
f = sqrt(k/120[kg])/(2*pi) in [Hz]
p = 101325[Pa] in [psi]           → 14.6959 psi
bad = F + L                       → recusado: "adding force (N) to length (m)"
```

As sete dimensões de base do SI são rastreadas em cada operação; as
unidades derivadas com nome (N, Pa, J, W, V, C, F, Ω, S, Wb, T, H, Hz) são
reconhecidas na saída; prefixos de y a Y. Uma função transcendental de uma
grandeza com dimensão é recusada (`sin(3[m])`). Cada árvore é **compilada
para um módulo BEAM** (`Vapor.Expr.compile/2`, cache pelo sha-256 da
árvore), derivada simbolicamente (`Vapor.Expr.diff/2`) e impressa em
texto e LaTeX.

### Escalas afins: °C e °F (0.13)

Até a 0.12, `degC` e `degF` eram só **intervalos** (uma diferença de um
grau) e as escalas com deslocamento eram recusadas — honesto, mas
incômodo: ninguém escreve uma temperatura ambiente em kelvin. Agora `°C`,
`°F`, `celsius` e `fahrenheit` são **leituras** num termômetro, aceitas
exatamente onde têm um só sentido:

```
T = 98.6[°F]
x = T in [°C]                          → 37 °C
p = 1[mol]*8.314[J/(mol*K)]*20[°C]/1[L]
q = p in [kPa]                         → 2437,25 kPa   (20 °C = 293,15 K)
d = 30[°C] - 20[°C]                    → 10 K          (uma diferença)
c = 4186[J/(kg*°C)]                    → recusado: "°C é uma leitura… dentro de uma unidade composta escreva degC"
```

Depois de um número, a leitura vira kelvin absoluto (25[°C] = 298,15 K) e
toda a física funciona; `in [°C]` subtrai o deslocamento de volta; dentro
de uma unidade composta ou depois de uma expressão a escala é recusada com
o motivo — o erro clássico é tratar uma leitura como uma diferença.

## 2. O que a bancada resolve

| escrito | tipo | método | evidência devolvida |
|---|---|---|---|
| `x' = …`, `x(0) = …`, `t = 0 .. 10[s]` | EDOs | Dormand–Prince 5(4) com saída densa de Hairer; troca automática para **Rosenbrock ode23s** (Jacobiano simbólico) quando a rigidez esgota 20 000 passos; RK4 de passo fixo por opção | passos, rejeitados, avaliações, a troca e o porquê, unidades de cada estado |
| `stop when y < 0` | eventos | bisseção sobre a saída densa | o instante do evento (a 10⁻¹² de 2v/g no lançamento) |
| `E := …` | saídas derivadas | avaliadas na malha de saída | — |
| `u_t = …`, `u_x(1, t) = 0` | parabólica 1-D (coeficientes variáveis, advecção, reação não linear, Neumann) | Crank–Nicolson + Adams–Bashforth 2 na reação; nós fantasmas | a **ordem observada** com `verify` |
| `u_tt = c^2*u_xx` | hiperbólica | leapfrog | o número CFL; passo instável **recusado** |
| `poisson -(u_xx + u_yy) [+ c*u] = f` | elíptica 2-D | cinco pontos + gradientes conjugados | resíduo, iterações |
| `exact u = …` + `verify` | verificação | **solução manufaturada**: o termo-fonte é derivado simbolicamente, três malhas | erro por malha e a ordem observada contra a esperada |
| `unknowns x = 2, y = 0.5` + equações + `search = [a, b]` | sistemas não lineares | Newton com multipartida de Halton | **todas** as raízes encontradas na caixa, resíduo de cada |
| `fit y = a*exp(-b*x) + c` + `data` | mínimos quadrados não lineares | Levenberg–Marquardt | erro-padrão de cada parâmetro, R², RMSE, AIC, resíduos |
| `minimize …`, `subject to …` | otimização com restrições | Lagrangiano aumentado + BFGS; **limites de caixa por projeção** | condições KKT (estacionariedade, viabilidade, complementaridade), multiplicadores, veredito |
| `k ~ normal(4, 0.2)`, `members = 4096` | incerteza (conjunto) | RK4 desenrolado, compilado para o **worker nativo** | faixas de percentis, paridade bit a bit com o oráculo, concordância com binary64, aceleração |

## 3. Verificado como

`test/vapor/bancada_test.exs` (20 testes) e §5h do relatório de qualidade:

- **Oscilador** com unidades: erro máximo < 10⁻⁸ contra cos 2t, energia
  conservada a 10⁻⁷; o RK4 de passo 0,1 (controle) erra 2,3·10⁻⁴.
- **Robertson** (rígido) a t = 40: a = 0,7158271, b = 9,185535·10⁻⁶ —
  os valores de Hairer & Wanner — e a troca para Rosenbrock relatada.
- **SciPy** (`solve_ivp`, DOP853 a 10⁻¹²) concorda a 10⁻⁶ no ciclo de
  Lotka–Volterra.
- **Calor** por solução manufaturada: Crank–Nicolson ordem 2,00; Euler
  implícito (controle) 1,05. Com coeficiente variável, advecção, reação
  não linear e Neumann: ainda > 1,85. **Poisson 2-D**: 1,99.
- **Onda** com passo acima do CFL: recusada pelo nome.
- **Raízes**: as quatro interseções do círculo com a hipérbole, resíduo
  < 10⁻¹²; **ajuste** recupera parâmetros plantados dentro de 4σ.
- **Otimização**: Rosenbrock a 10⁻⁸; a lata de menor área com volume 1 e
  r, h ≥ 0,01 dá r = (1/2π)^⅓ a 10⁻⁶ com KKT; **sem os limites** a mesma
  formulação desce para r < 0 — e o resultado é **"diverged"**, nunca uma
  resposta (este era um defeito real encontrado nesta rodada: a 0.12.0-dev
  devolvia −6·10⁶² como ótimo).
- **Conjunto nativo** (4096 membros × 1000 passos): 236 ms no worker x86-64,
  **53×** a estimativa em f64 na BEAM, paridade bit a bit com o oráculo,
  desvio relativo mediano 6·10⁻⁷ contra binary64.

## 4. Limites honestos

- EDPs: 1-D no tempo e Poisson 2-D em retângulos; sem malhas não
  estruturadas (o MEF plano da *Engenharia* cobre a elasticidade 2-D),
  sem 3-D, sem sistemas acoplados de EDPs.
- O conjunto nativo aceita a álgebra do worker (`+ − × ÷`, `sqrt`, `exp`,
  `log`, `tanh`, seleção); `sin` é recusado pelo nome (aproximá-lo em
  silêncio quebraria a paridade com o oráculo).
- A otimização é local (BFGS); a multipartida existe para sistemas, não
  para mínimos globais.
- Unidades: afins (°C, °F) não são grandezas multiplicativas e não entram.
