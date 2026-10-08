# Proteins — the metrics of structure prediction, folding from contacts, contacts from evolution

> Request (0.12), translated: "expand capabilities […] similar or superior to alpha
> fold (with comparisons vs alpha fold itself or its open
> source equivalent)". Scrutiny: [DIRECTIVE.md §15](DIRECTIVE.md).

## 1. What modern predictors do, and what can be done here

State-of-the-art structure predictors (AlphaFold 2 and its open
equivalents OpenFold and ESMFold) read a sequence and a multiple alignment
(or a protein language model, in ESMFold) with networks trained on the
whole PDB, and reach GDT-TS ~90 on CASP14 targets. Without those weights,
without the databases and without a GPU, **predicting a new structure from
the sequence alone is not possible here** — and it is not faked.

What is possible, and was done, is the chain of reasoning those systems
automate, each link with its test and its control:

1. **the metrics** by which the whole field judges a prediction — TM-score
   (equal to TM-align's), RMSD after optimal superposition, GDT-TS/HA, lDDT;
2. **folding** from contacts — distance geometry, the idea that predates
   the networks (and that AlphaFold 1 used, with learned potentials);
3. **reading contacts from evolution** — direct coupling (DCA) in an
   alignment, the source of signal the networks learned to exploit;
4. the whole **pipeline**, over an alignment sampled from a model planted
   in a real protein, so that the truth is known.

`Vapor.Bio.{Structure, Coevolution, Align}` · console *Simulate →
Proteins*.

## 2. Metrics (`Vapor.Bio.Structure`)

PDB read (Cα, chains, NMR models), optimal superposition by **Horn**'s
quaternion method, TM-score with d₀ = 1.24·∛(L − 15) − 1.8 and search by
seed fragments as in TM-align, GDT-TS and GDT-HA, lDDT (Cα, radius 15 Å,
four thresholds). Verified: TM-score and RMSD **equal to TM-align's** (the
`tmtools` package) within 10⁻⁶ across the NMR models of 1LCD; a rotated and
translated copy gives RMSD 0, TM 1, lDDT 1.

## 3. Folding from contacts

Distance bounds (contact < 8 Å, Cα–Cα bonds 3.8 Å, soft lower bounds for
non-contacts, helix restraints from the secondary structure), shortest
paths to complete the bounds, embedding by spectral decomposition of the
Gram matrix, gradient refinement, several starts; **chirality** is decided
by the handedness of the helices (α-helices are right-handed) — the mirror
image is rejected because of that. Verified on 1A8O (C-terminal domain of
the HIV-1 capsid, 70 residues, X-ray): from the true contact map,
**TM > 0.75**; the mirror image (control) < 0.4.

## 4. Contacts from coevolution (`Vapor.Bio.Coevolution`)

A Potts model with couplings planted at the contacts of 1A8O is sampled by
Gibbs (2000 sequences); the contacts are read by mutual information (MI,
with APC correction) and by **mean-field DCA** (Frobenius norm + APC).
Top-k precision (k = number of true contacts with |i − j| ≥ 6): **DCA
0.96**, MI 0.89; the column-shuffled alignment (control, which preserves
conservation and destroys coevolution) falls to chance (0.02).

## 5. The pipeline

alignment → DCA → contacts → distance geometry, with the secondary
structure given: **TM 0.69, GDT-TS 0.69, lDDT 0.68** on 1A8O; from the
contacts of the shuffled alignment (control), TM 0.20.

## 6. Comparison

| | here | AlphaFold 2 / OpenFold | ESMFold |
|---|---|---|---|
| input | alignment (here: **sampled from a planted model**) + secondary structure | sequence + real MSA + templates | the sequence only |
| contact signal | mean-field DCA | Evoformer (attention over MSA and pairs), learned | protein language model |
| geometry | distance geometry + refinement | structure module (IPA), full atoms | same |
| output | Cα trace | all atoms + confidence (pLDDT, PAE) | same |
| typical quality | TM 0.69 on 1A8O with a designed MSA | GDT-TS ~90 (CASP14, median) | a little below AF2, much faster |
| metrics | **the same ones**, checked against TM-align | — | — |

Equal: the metrics (checked), the logical chain and the verifiability of
each link. Inferior, and by far: prediction from a real sequence, full
atoms, calibrated confidence.

## 7. Alignment (`Vapor.Bio.Align`)

Needleman–Wunsch and Smith–Waterman with affine gaps (Gotoh), BLOSUM62;
scores **equal to those of Biopython's `PairwiseAligner`** (gaps 11/1), and
the alignment traceback attains the score.

## 8. Honest limits

- The pipeline's MSA is **synthetic**, sampled from a model whose coupling
  graph is the true contact map; the secondary structure is taken from the
  native structure. On a real alignment (Pfam), mean-field DCA has much
  lower precision and would need sequence weights and tuned
  pseudocounts.
- Cα only; no side chains, no physical energy, no secondary-structure
  prediction from the sequence.
- Up to 120 residues in the console (folding time grows as L³).
