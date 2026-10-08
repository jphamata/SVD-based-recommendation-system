# Recommendations — factorisation with the evidence that decides

> `Vapor.Recommend` (`lib/vapor/recommend.ex`), `Vapor.Dense.svd/1`, `vapor recommend`, the MCP tool
> `recommend_run`. Tests: `recommend_test.exs`; §5m. It absorbs `SVD_Recommendation_System.py`,
> the script of the repository that now hosts vapor.

## The model

A rating is `μ + b_u + b_i + p_u · q_i`: a global mean, a user bias, an item bias and a rank-k
interaction. This is the "SVD" of Funk's Netflix-prize model, which is a factorisation fitted to the
observed entries, not a singular value decomposition. It is fitted by **alternating least
squares**: with the items fixed, each user's `(b_u, p_u)` is the exact solution of a ridge
regression, and the other way round. There is no learning rate, and the same data, rank, λ and
seed give the same model bit for bit.

## The evidence

An RMSE on its own says nothing, so `Recommend.evaluate/2` reports it with what it is measured
against:

- a **test split** that no choice ever sees. Rank and λ are chosen on a validation split carved
  from the training ratings (nested);
- two **baselines** on the same test ratings: the global mean, and the biases without interaction
  (k = 0);
- a **paired test**: the factor model's squared errors against the biases', rating by rating, by
  the sign-flip test, *and* a minimum gain (1% of the biases' mean squared error by default);
- a **control**: the same pipeline on the training ratings shuffled across cells. It must not beat
  the global mean, or the pipeline finds structure where there is none.

The verdict is `signal` only when the factor model beats the biases significantly *and* by a margin
that matters, *and* the control stays at the mean. The minimum gain was added because the control
caught its absence. On unstructured ratings, a heavily regularised model differed from the biases by
10⁻¹⁰ in RMSE, and every difference had the same sign, so the paired test called it significant.
Statistically significant and worthless; the test file keeps that case.

## The script it replaces

| defect in `SVD_Recommendation_System.py` | here |
|---|---|
| unseeded (`np.random.rand`): every run differs | seeded splits and initialisation; the same model bit for bit |
| 60% of the ratings **overwritten with 50.0** and then trained on as observed | missing stays missing. Measured: the midpoint imputation more than 1.5× the honest test RMSE on planted data |
| the last row of the table is two values short, so pandas reads NaN ratings | two missing cells |
| grid search on **all** the ratings, the test set included | nested: chosen on validation, reported on test |
| a 1–100 scale declared for ratings in {25, 50, 75, 100} | predictions clamped to the observed range |
| an RMSE with nothing to compare it with | two baselines, a paired test with a minimum gain, a shuffled control |

On the script's own table (6 genre profiles × 70 films, 418 ratings; 20% held out) the verdict is
**signal**: rank 6, test RMSE 22.7 against 29.3 for the biases and 28.5 for the mean, with the
shuffled control at 28.7. With 40% of the cells removed as missing, it is "no evidence the
interactions help": 212 training ratings are too few for the gain (25.2 against 28.9) to be told
from luck.

## The exact SVD

For a complete matrix the classical truncated SVD is the best rank-k approximation (Eckart–Young).
`Dense.svd/1` computes it by **one-sided Jacobi** (Hestenes): plane rotations orthogonalise the
columns, and the column norms are the singular values. It is deterministic, and accurate to the
*relative* precision of small singular values. `Dense.svd_residual/2` is its certificate
(`‖A − UΣVᵀ‖_F / ‖A‖_F`, `max|UᵀU − I|`, `max|VᵀV − I|`), and `Recommend.truncated_svd/2` checks
`‖A − A_k‖_F² = Σ_{i>k} σᵢ²` from the factors.

Measured (§5m): for a matrix with singular values 1, 10⁻⁴, 10⁻⁸, Jacobi recovers 10⁻⁸ with a relative
error of 2·10⁻⁹. The usual shortcut, the square roots of the eigenvalues of AᵀA, loses it (a relative
error of 3·10⁻²) because AᵀA squares the condition number.

## Limits

Explicit ratings only: implicit feedback (clicks, plays) needs weighted ALS and is not here. Cold
start (a user with no ratings) gets the biases. One machine, BEAM floats, and no worker kernel:
fine for tables of tens of thousands of ratings, not for MovieLens-25M.
