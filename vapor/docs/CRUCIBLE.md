# Crucible — ciência aberta, com evidência que não precisa de gabarito

> Desde 0.14.0. Código: `lib/vapor/crucible.ex` e `lib/vapor/crucible/` (`sheet`, `poly`,
> `laws`, `quantum`, `hamiltonian`, `reactions`, `evolution`, `phylo`, `molecule`, `fold`,
> `fields`, `regress`). Testes: `test/vapor/crucible_test.exs`.

O painel "Ciência" das rodadas anteriores mostrava experimentos fixos com respostas conhecidas —
problemas resolvidos apresentados. Eles continuam (renomeados **Calibração** no console: são a
calibração dos instrumentos). O Crucible é o contrário: **a pessoa traz o sistema** — a sua
equação, a sua molécula, as suas sequências, os seus dados — e a resposta vem com evidência que
vale **sem solução de referência**:

- **ordem observada** (Richardson): o erro cai na taxa que o método promete, ou não;
- **teoremas que valem para qualquer entrada**: virial, Ehrenfest, trabalho–energia, a
  unitariedade, o comutador do SCF;
- **dois métodos que precisam concordar** (cadeia exata × simulação × Kimura);
- **controles que um método errado reprovaria**: RK4 contra o simplético, Euler contra Boris,
  colunas embaralhadas, alvo embaralhado, termos genéricos adicionados.

## Os domínios

| domínio | entrada | o que devolve | a evidência |
|---|---|---|---|
| `laws` | um sistema de EDOs | **leis de conservação provadas sobre ℚ** (inclusive com `ln`) | espaço nulo exato do coeficiente da derivada de Lie; sobrevivem a empurrar cada coeficiente 1 %; o controle genérico não tem nenhuma |
| `quantum` | V(x), caixa, malha | autovalores (bissecção de Sturm), dinâmica (split-step Fourier) | ordem 2 observada, virial, Ehrenfest, norma |
| `hamiltonian` | H(q, p) | integração simplética (Yoshida 4, Verlet, ponto médio implícito) | ordem, reversibilidade, deriva contra RK4, leis |
| `reactions` | reações químicas | ODE e Gillespie | invariantes de massa |
| `evolution` | N, s, i₀ | fixação de Wright–Fisher | cadeia exata = simulação = Kimura |
| `phylogeny` | FASTA | árvore NJ com *bootstrap* | o controle de colunas embaralhadas sem suporte |
| `molecule` | átomos H/He | RHF/STO-3G de N centros | comutador convergido, virial; H₂ = −1,1167 |
| `fold` | sequência HP | dobramento na rede | cota de paridade; ótimo provado por enumeração quando cabe |
| `fields` | E, B, carga | Boris | \|u\| conservado, trabalho–energia; Euler como controle |
| `regress` | tabela | regressão simbólica (escala linear de Keijzer + Nelder–Mead) | R² em teste; alvo embaralhado como controle |

Exemplo — uma epidemia SIR:

```
S' = -0.3*S*I
I' = 0.3*S*I - 0.1*I
R' = 0.1*I
S(0) = 0.99; I(0) = 0.01; R(0) = 0
t = 0 .. 100
degree = 2
```

→ `S + I + R` e `3·S + 3·I − ln(S)`, ambas **provadas** por cancelamento exato, ambas
estruturais (sobrevivem a coeficientes perturbados), deriva relativa < 10⁻⁴ ao longo da
trajetória. Um oscilador amortecido: nenhuma lei — e isso também é provado.

## Limites

Tudo roda em `Alembic.sandbox` (1 GB, 4 min): uma malha absurda volta como erro de tempo ou
memória, nunca derruba o servidor. A química é STO-3G de camada fechada com H e He; orbitais
p são recusados com o motivo. As leis com funções não polinomiais (`basis = [cos(q)]`) são
verificadas numericamente e **rotuladas** "não é prova".
