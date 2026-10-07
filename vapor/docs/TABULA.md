# Tábua — contratos sem antinomias

> Desde 0.15.0. Código: `lib/vapor/tabula.ex`. Testes: `test/vapor/tabula_test.exs`.
> Console: *Opus → Tábua*. Terminal: `vapor tabula FILE [--facts a,b]`. MCP: `tabula_analyze`.

A tábua de esmeralda é o texto de lei dos alquimistas.

## A dor

Um contrato cujas cláusulas, sob alguma combinação de eventos, obrigam uma parte a fazer o que
outra cláusula proíbe — ou a fazer duas coisas que não podem ser feitas juntas — é descoberto por
um tribunal, anos depois. Se essa combinação **existe** é uma pergunta de lógica proposicional, e
pode ser respondida antes da assinatura.

```
parties buyer seller
facts delivered late defective force_majeure
exclusive pay withhold
assume not (late and not delivered)
C1: if delivered and not defective then buyer must pay seller
C2: if late then buyer may withhold
C3: if defective then buyer must not pay
C4: if force_majeure then seller is exempt from deliver
C5: seller must deliver buyer
C6: if late and delivered then buyer must withhold
C4 overrides C5
```

Modalidades deônticas (as posições de Hohfeld entre parênteses): `must` (dever — a pretensão da
contraparte), `must not` (proibição), `may` (privilégio), `is exempt from` / `need not` (sem
dever). Em português também: `deve`, `não deve`, `pode`, `está isento de`; `se … então`;
`prevalece sobre`.

## A decisão

Para cada par de cláusulas sobre a mesma parte e a mesma ação cujas modalidades colidem — dever ×
proibição, proibição × privilégio, dever × isenção, ou dois deveres sobre ações declaradas
`exclusive` — a pergunta "as duas condições podem valer juntas, dadas as premissas `assume`?" vai
ao resolvedor SAT (`Vapor.Logic.Formula`, Tseitin):

- **satisfazível** é uma antinomia **com o cenário que a dispara** (conferido avaliando as
  cláusulas);
- **insatisfazível** é uma prova, conferida pelo `Vapor.Logic.DRUP`, de que a colisão nunca
  acontece.

Também relatados: **silêncios** (um cenário em que nenhuma cláusula diz nada sobre uma ação que
outras cláusulas regem — uma lacuna, não um erro). Uma colisão entre cláusulas uma das quais
`overrides` a outra (*lex specialis*, *lex posterior*: a pessoa diz qual) é **resolvida**,
relatada com o seu cenário, não contada como antinomia; um ciclo de precedências é recusado.
`positions/2` dá, para um conjunto de fatos, as posições em vigor, as sobrepostas, as pretensões
(todo dever devido a alguém é a pretensão desse alguém) e as colisões.

No exemplo: C1 × C6 é uma antinomia quando `delivered ∧ late ∧ ¬defective`; C1 × C3 nunca colide
(prova DRUP); C4 × C5 é resolvida por C4. Com `C6 overrides C1` também, nenhuma antinomia sobra.
Estresse: 120 cláusulas sobre 30 fatos — cada achado reavaliado, cada prova conferida.

## No console

As cláusulas como uma tábua (as sobrepostas riscadas), os fatos como botões que ligam e
desligam e recalculam as posições, e cada achado com um botão "fixar estes fatos" que põe a
tábua no cenário que dispara a colisão.

## O que não é

Não interpreta linguagem natural nem direito. A pessoa (ou um modelo, como rascunho que a pessoa
revisa) escreve as cláusulas nesta forma; a tábua decide só o que segue delas. Prazos, valores e
quantificação sobre partes ficam fora (lógica proposicional, por escolha: decidível, com prova).
