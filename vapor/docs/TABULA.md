# Tabula — contracts without antinomies

> Since 0.15.0. Code: `lib/vapor/tabula.ex`. Tests: `test/vapor/tabula_test.exs`.
> Console: *Opus → Tabula*. Terminal: `vapor tabula FILE [--facts a,b]`. MCP: `tabula_analyze`.

The emerald tablet (*tabula smaragdina*) is the alchemists' text of law.

## The pain

A contract whose clauses, under some combination of events, oblige one party to do what
another clause forbids — or to do two things that cannot be done together — is found out by
a court, years later. Whether that combination **exists** is a question of propositional logic, and
it can be answered before signing.

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

Deontic modalities (Hohfeld's positions in parentheses): `must` (duty — the counterparty's
claim), `must not` (prohibition), `may` (privilege), `is exempt from` / `need not` (no
duty). In Portuguese as well: `deve`, `não deve`, `pode`, `está isento de`; `se … então`;
`prevalece sobre`.

## The decision

For each pair of clauses about the same party and the same action whose modalities collide — duty ×
prohibition, prohibition × privilege, duty × exemption, or two duties over actions declared
`exclusive` — the question "can both conditions hold together, given the `assume` premises?" goes
to the SAT solver (`Vapor.Logic.Formula`, Tseitin):

- **satisfiable** is an antinomy **with the scenario that triggers it** (checked by evaluating the
  clauses);
- **unsatisfiable** is a proof, checked by `Vapor.Logic.DRUP`, that the collision never
  happens.

Also reported: **silences** (a scenario in which no clause says anything about an action that
other clauses govern — a gap, not an error). A collision between clauses one of which
`overrides` the other (*lex specialis*, *lex posterior*: the person says which) is **resolved**,
reported with its scenario, not counted as an antinomy; a cycle of precedences is refused.
`positions/2` gives, for a set of facts, the positions in force, the overridden ones, the claims
(every duty owed to someone is that someone's claim) and the collisions.

In the example: C1 × C6 is an antinomy when `delivered ∧ late ∧ ¬defective`; C1 × C3 never collides
(DRUP proof); C4 × C5 is resolved by C4. With `C6 overrides C1` as well, no antinomy remains.
Stress: 120 clauses over 30 facts — each finding re-evaluated, each proof checked.

## In the console

The clauses as a tablet (the overridden ones struck through), the facts as buttons that switch on and
off and recompute the positions, and each finding with a "pin these facts" button that puts the
tablet into the scenario that triggers the collision.

## What it is not

It does not interpret natural language or law. The person (or a model, as a draft the person
reviews) writes the clauses in this form; Tabula decides only what follows from them. Deadlines, amounts and
quantification over parties stay out (propositional logic, by choice: decidable, with a proof).
