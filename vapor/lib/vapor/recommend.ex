defmodule Vapor.Recommend do
  @moduledoc """
  Recommendation by matrix factorisation, with the evidence that decides
  whether it works (docs/RECOMMEND.md). It absorbs the
  `SVD_Recommendation_System.py` of the repository that hosts vapor, and
  answers the questions that script left open.

  **The model.** A rating is `μ + b_u + b_i + p_u · q_i`: a global mean, a
  user bias, an item bias and a rank-`k` interaction (the "SVD" of
  Funk's Netflix-prize model, which is a factorisation fitted to the
  observed entries, not a singular value decomposition). It is fitted by
  **alternating least squares**: with the items fixed, every user's
  `(b_u, p_u)` is the exact solution of a ridge regression, and the other
  way round. There is no learning rate to tune, and the same data, rank,
  `λ` and seed give the same model bit for bit.

  **The evidence**, because an RMSE alone says nothing:

    * a held-out test split that no choice ever sees: rank and `λ` are
      chosen on a validation split carved from the training ratings
      (nested);
    * two baselines on the same test ratings: the global mean, and the
      biases without interaction (`k = 0`). The factor model's squared
      errors are compared with the biases' rating by rating, by the paired
      sign-flip test;
    * a control: the same pipeline on the training ratings shuffled across
      cells. It must not beat the global mean by much, or the pipeline is
      finding structure where there is none.

  **The script it replaces** had six defects: it was unseeded; its table's
  last row ("Cinema Arte/Cult") is two values short, which pandas reads as
  NaN ratings and passes on (here: two missing cells); it
  *overwrote* 60 % of the ratings with 50.0 and then trained on them as if
  they had been observed (missing is not 50); it ran its grid search on
  all the ratings, test set included; it declared a 1–100 scale for
  ratings in {25, 50, 75, 100}; and it reported an RMSE with nothing to
  compare it with. Each one is fixed here, and the test file shows the
  second one's damage measured.

  For a **complete** matrix the classical truncated SVD is exact
  (Eckart–Young): `truncated_svd/2` gives `A_k` and the certificate
  `‖A − A_k‖_F² = Σ_{i>k} σᵢ²`, checked from the factors.
  """
  alias Vapor.Dense
  alias Vapor.Assay.Stats

  defstruct [:mu, :bu, :bi, :p, :q, :k, :lambda, :users, :items, :scale]

  # ------------------------------------------------------------------ data

  @doc """
  A ratings table from CSV: the first row names the items, the first
  column the users; an empty cell is **missing** (never a number).
  `{:ok, %{users, items, ratings: [{u, i, r}]}}` with indices.
  """
  def parse_csv(text) do
    rows = text |> String.split(~r/\r?\n/) |> Enum.reject(&(String.trim(&1) == "")) |> Enum.map(&csv_row/1)

    case rows do
      [[_ | items] | body] when items != [] ->
        {users, ratings} =
          body
          |> Enum.with_index()
          |> Enum.map_reduce([], fn {[name | cells], u}, acc ->
            rs = for {c, i} <- Enum.with_index(cells), String.trim(c) != "", do: {u, i, number!(c)}
            {name, acc ++ rs}
          end)

        {:ok, %{users: users, items: items, ratings: ratings}}

      _ ->
        {:error, "a ratings table: the first row names the items, the first column the users"}
    end
  catch
    {:recommend, why} -> {:error, why}
  end

  defp csv_row(line) do
    Regex.scan(~r/(?:^|,)(?:"((?:[^"]|"")*)"|([^,]*))/, line)
    |> Enum.map(fn
      [_, q] -> String.replace(q, "\"\"", "\"")
      [_, "", plain] -> plain
      [_, q, _] -> String.replace(q, "\"\"", "\"")
    end)
  end

  defp number!(c) do
    case Float.parse(String.trim(c)) do
      {x, ""} -> x
      _ -> throw({:recommend, "#{inspect(c)} is not a rating (leave the cell empty when it is missing)"})
    end
  end

  @doc "A deterministic split: each rating goes to the held-out side with probability `frac`, by a hash of (seed, user, item)."
  def split(ratings, frac, seed) do
    Enum.split_with(ratings, fn {u, i, _} -> :erlang.phash2({seed, u, i}, 1_000_000) >= trunc(frac * 1_000_000) end)
  end

  # ------------------------------------------------------------------ model

  @doc """
  Fit the biased factorisation to `ratings` (`[{u, i, r}]`) by alternating
  least squares. Options: `k` (rank, 0 = biases only), `lambda` (the ridge
  penalty on biases and factors, default 1.0), `iters` (15), `seed`,
  `users`, `items` (counts; default from the data).
  """
  def fit(ratings, opts \\ []) do
    k = Keyword.get(opts, :k, 4)
    lambda = Keyword.get(opts, :lambda, 1.0)
    iters = Keyword.get(opts, :iters, 15)
    nu = Keyword.get(opts, :users, ratings |> Enum.map(&elem(&1, 0)) |> Enum.max(fn -> -1 end) |> Kernel.+(1))
    ni = Keyword.get(opts, :items, ratings |> Enum.map(&elem(&1, 1)) |> Enum.max(fn -> -1 end) |> Kernel.+(1))
    rs = Enum.map(ratings, &elem(&1, 2))
    mu = Stats.mean(rs) || 0.0
    by_u = Enum.group_by(ratings, &elem(&1, 0))
    by_i = Enum.group_by(ratings, &elem(&1, 1))
    rng = :rand.seed_s(:exsss, {Keyword.get(opts, :seed, 1), 17, 29})
    {q, _} = Enum.map_reduce(0..(ni - 1)//1, rng, fn _, r -> Enum.map_reduce(1..k//1, r, fn _, r -> {x, r} = :rand.normal_s(r); {0.1 * x, r} end) end)
    q = List.to_tuple(q)
    bi = Tuple.duplicate(0.0, ni)

    {p, bu, q, bi} =
      Enum.reduce(1..iters, {nil, nil, q, bi}, fn _, {_p, _bu, q, bi} ->
        {p, bu} = side(nu, by_u, fn {_u, i, r} -> {r - mu - elem(bi, i), elem(q, i)} end, k, lambda)
        {q, bi} = side(ni, by_i, fn {u, _i, r} -> {r - mu - elem(bu, u), elem(p, u)} end, k, lambda)
        {p, bu, q, bi}
      end)

    lo = Enum.min(rs, fn -> 0.0 end)
    hi = Enum.max(rs, fn -> 0.0 end)
    %__MODULE__{mu: mu, bu: bu, bi: bi, p: p, q: q, k: k, lambda: lambda, users: nu, items: ni, scale: {lo, hi}}
  end

  # one half-step of ALS: for every row of one side, the ridge solution for [bias | factors]
  defp side(n, groups, target, k, lambda) do
    rows =
      for j <- 0..(n - 1)//1 do
        obs = Map.get(groups, j, [])

        if obs == [] do
          {List.duplicate(0.0, k), 0.0}
        else
          xs = Enum.map(obs, fn o -> {y, f} = target.(o); {[1.0 | f], y} end)
          d = k + 1
          a = for r <- 0..(d - 1), do: for(c <- 0..(d - 1), do: Enum.reduce(xs, if(r == c, do: lambda, else: 0.0), fn {x, _}, s -> s + Enum.at(x, r) * Enum.at(x, c) end))
          b = for r <- 0..(d - 1), do: Enum.reduce(xs, 0.0, fn {x, y}, s -> s + Enum.at(x, r) * y end)
          {:ok, [bias | f]} = Dense.solve(a, b)
          {f, bias}
        end
      end

    {rows |> Enum.map(&elem(&1, 0)) |> List.to_tuple(), rows |> Enum.map(&elem(&1, 1)) |> List.to_tuple()}
  end

  @doc "The predicted rating, clamped to the observed scale."
  def predict(%__MODULE__{} = m, u, i) do
    {lo, hi} = m.scale
    raw = m.mu + safe(m.bu, u) + safe(m.bi, i) + Dense.dot(safe_vec(m.p, u, m.k), safe_vec(m.q, i, m.k))
    raw |> max(lo) |> min(hi)
  end

  defp safe(t, j), do: if(j < tuple_size(t), do: elem(t, j), else: 0.0)
  defp safe_vec(t, j, k), do: if(j < tuple_size(t), do: elem(t, j), else: List.duplicate(0.0, k))

  @doc "Squared errors on `ratings`, in order."
  def sq_errors(m, ratings), do: Enum.map(ratings, fn {u, i, r} -> (predict(m, u, i) - r) * (predict(m, u, i) - r) end)

  @doc "Root mean squared error on `ratings`."
  def rmse(m, ratings), do: :math.sqrt(Stats.mean(sq_errors(m, ratings)))

  @doc "The `n` best unrated items for user `u`: `[{item, predicted}]`."
  def recommend(m, u, rated, n) do
    seen = MapSet.new(rated)
    0..(m.items - 1)//1 |> Enum.reject(&MapSet.member?(seen, &1)) |> Enum.map(&{&1, predict(m, u, &1)}) |> Enum.sort_by(&elem(&1, 1), :desc) |> Enum.take(n)
  end

  # ------------------------------------------------------------- the pipeline

  @doc """
  The whole evaluation on a ratings table (`parse_csv/1`'s map). Options:
  `test` (fraction held out, 0.2), `valid` (fraction of training held for
  choosing, 0.2), `ranks` ([0, 1, 2, 3, 4, 6, 8]), `lambdas` ([0.1, 1, 5,
  20]), `seed`, `top` (recommendations per user, 5), `alpha` (0.05),
  `min_gain` (0.01: the factor model must cut the biases' mean squared
  error by at least this fraction).

  Returns `{:ok, report}` with the chosen rank and `λ`, test RMSE of the
  model and of both baselines, the paired test against the biases, the
  shuffled control and the recommendations. The verdict is `"signal"` only
  if the factor model beats the biases significantly, by a margin that
  matters (`min_gain`: a gain of 10⁻¹⁰ on every rating is "significant"
  and worthless, and the control found exactly that), *and* the shuffled
  control does not beat the global mean.
  """
  def evaluate(%{users: users, items: items, ratings: ratings}, opts \\ []) do
    seed = Keyword.get(opts, :seed, 1)
    {train, test} = split(ratings, Keyword.get(opts, :test, 0.2), seed)
    {fit_set, valid} = split(train, Keyword.get(opts, :valid, 0.2), seed + 1)
    dims = [users: length(users), items: length(items), seed: seed]

    if test == [] or valid == [] or fit_set == [] do
      {:error, "too few ratings to hold out a test and a validation split"}
    else
      grid = for k <- Keyword.get(opts, :ranks, [0, 1, 2, 3, 4, 6, 8]), l <- Keyword.get(opts, :lambdas, [0.1, 1.0, 5.0, 20.0]), do: {k, l}
      {k, lambda} = Enum.min_by(grid, fn {k, l} -> rmse(fit(fit_set, [k: k, lambda: l] ++ dims), valid) end)

      model = fit(train, [k: k, lambda: lambda] ++ dims)
      biases = fit(train, [k: 0, lambda: lambda] ++ dims)
      mean = Stats.mean(Enum.map(train, &elem(&1, 2)))
      mean_rmse = :math.sqrt(Stats.mean(Enum.map(test, fn {_, _, r} -> (r - mean) * (r - mean) end)))
      d = Enum.zip_with(sq_errors(biases, test), sq_errors(model, test), &(&1 - &2))
      p = Stats.sign_flip(d, 20_000, seed)

      # the control: the same choice of rank and λ, on ratings shuffled across their cells
      shuffled = shuffle(train, seed)
      control = fit(shuffled, [k: k, lambda: lambda] ++ dims)
      control_rmse = rmse(control, test)
      alpha = Keyword.get(opts, :alpha, 0.05)
      bias_mse = Stats.mean(sq_errors(biases, test))
      gain = Stats.mean(d) / max(bias_mse, 1.0e-300)
      beats = gain >= Keyword.get(opts, :min_gain, 0.01) and p <= alpha
      control_ok = control_rmse >= 0.95 * mean_rmse

      rated = Enum.group_by(ratings, &elem(&1, 0), &elem(&1, 1))
      top = Keyword.get(opts, :top, 5)
      recs = for {name, u} <- Enum.with_index(users), into: %{}, do: {name, Enum.map(recommend(model, u, Map.get(rated, u, []), top), fn {i, r} -> {Enum.at(items, i), r} end)}

      {:ok,
       %{ratings: length(ratings), train: length(train), test: length(test), rank: k, lambda: lambda,
         rmse: %{model: rmse(model, test), biases: rmse(biases, test), mean: mean_rmse, shuffled_control: control_rmse},
         paired: %{better: Enum.count(d, &(&1 > 0)), of: length(d), mean_gain: Stats.mean(d), relative_gain: gain, p_value: p},
         verdict: cond do
           beats and control_ok -> "signal"
           not control_ok -> "flawed: the shuffled control beats the mean"
           true -> "no evidence the interactions help (the biases explain what is predictable)"
         end,
         recommendations: recs}}
    end
  end

  defp shuffle(ratings, seed) do
    {vals, _} = Enum.map_reduce(ratings, :rand.seed_s(:exsss, {seed, 101, 103}), fn {_, _, r}, s -> {x, s} = :rand.uniform_s(s); {{x, r}, s} end)
    shuffled = vals |> Enum.sort() |> Enum.map(&elem(&1, 1))
    Enum.zip_with(ratings, shuffled, fn {u, i, _}, r -> {u, i, r} end)
  end

  # ------------------------------------------------------------ exact SVD

  @doc """
  The rank-`k` truncation of a complete matrix by its SVD (Eckart–Young:
  the best rank-`k` approximation in the Frobenius norm), with the
  certificate: the factorisation's own residual and orthogonality, and
  `‖A − A_k‖_F²` against `Σ_{i>k} σᵢ²`.
  """
  def truncated_svd(a, k) do
    f = Dense.svd(a)
    cert = Dense.svd_residual(a, f)
    uk = Enum.map(f.u, &Enum.take(&1, k))
    vk = Enum.map(f.v, &Enum.take(&1, k))
    sk = Enum.take(f.s, k)
    ak = Dense.matmul(Enum.map(uk, fn r -> Enum.zip_with(r, sk, &(&1 * &2)) end), Dense.transpose(vk))
    err2 = Enum.zip_with(a, ak, fn x, y -> Enum.zip_with(x, y, &((&1 - &2) * (&1 - &2))) |> Enum.sum() end) |> Enum.sum()
    tail = f.s |> Enum.drop(k) |> Enum.map(&(&1 * &1)) |> Enum.sum()
    {:ok, %{approx: ak, singular_values: f.s, error2: err2, tail2: tail, certificate: cert}}
  end
end
