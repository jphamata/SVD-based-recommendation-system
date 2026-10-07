# round16 — checks against their controls

Written by `mix vapor.quality --only round16 --md …` (11/11 passed in 3932 ms,
substrate: the exact oracle). Each check has a value, a control that a broken, naive or lucky
implementation would produce, and the threshold that separates them; see `Vapor.Quality.Round16`.

| check | value | control | threshold | ok |
|---|---|---|---|---|
| crash atomicity: a crash at every byte of a commit (pack append and root slot) | 344/344 reopen at the old or the new root | one root file overwritten in place, crash halfway: lost (decodes as v2 with a torn value) | every byte: old or new; the naive store loses its root | ✓ |
| fork: a new thread at the head of a 60-turn conversation | 0 messages and 964 bytes added | a deep copy adds 120 messages | no message added; the root only | ✓ |
| export integrity: one changed character in a 120-message export | refused (a hash does not match) | the same change in the Markdown export: undetectable | the hashed format refuses; plain text cannot | ✓ |
| context under a 600-token budget: the pinned first message and the last question | both sent, 597 tokens | keep-the-tail truncation keeps the pinned message: false | both sent; the naive tail drops the pin | ✓ |
| conservation decided exactly: an undamped oscillator; the same with damping 10⁻⁹ | proved; refuted | sampling dH/dt at 1 000 points (\|·\| < 10⁻⁸) accepts the damped law: true | proved; refuted; the sampled check is fooled | ✓ |
| projections: 300 programs with Arabic names read back from both scripts | 300/300 identical trees | folding hamza forms: 7 names → 2 | all; the lossy transliteration merges names | ✓ |
| identity: SHA-256 over every three-letter root | 21952/21952 distinct | abjad: 21950 of 21952 share a value | no collision; abjad > 99 % shared | ✓ |
| metric: the triangle inequality on 1 000 random triples of distributions | Fisher–Rao: 0 violations | KL: 217 violations | 0; KL > 0 | ✓ |
| reparametrisation: logistic regression with one feature rescaled ×1000 | natural gradient: largest prediction change 7.0e-12 | plain gradient: 0.5418 | < 10⁻⁹; > 10⁻³ | ✓ |
| jail: the console's terminal asked to read a server file | exit 3: /tmp/round16-4/server-secret.txt: no such file in this sessi | a local session (the TUI): exit 0 | refused in the jail; read locally | ✓ |
| entropy boundary: OS randomness outside Vapor.Entropy | 0 files | an injected draw: flagged | none; the injected one flagged | ✓ |
