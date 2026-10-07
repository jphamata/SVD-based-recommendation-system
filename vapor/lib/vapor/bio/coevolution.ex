defmodule Vapor.Bio.Coevolution do
  @moduledoc """
  Contacts read from evolution (docs/PROTEINAS.md §3): residues in
  contact mutate together, so their columns in a multiple sequence
  alignment co-vary. Two estimators:

    * **mutual information** with the average-product correction (Dunn et
      al. 2008) — it also scores pairs correlated only through a third
      residue (transitive chains);
    * **mean-field direct coupling analysis** (Morcos et al. 2011): the
      couplings of a Potts model, −C⁻¹ of the pseudocount-regularised
      covariance, scored by the Frobenius norm with APC — direct pairs
      only. This is the step that turned contact prediction from noise
      into the input of the structure predictors that followed.

  To measure without trusting anyone's alignment, the alignment is
  **sampled from a Potts model planted on a real protein's contact map**
  (Gibbs sampling; couplings on the true contacts, nothing elsewhere):
  the truth is known exactly, and a shuffled alignment (the control) must
  score at the level of chance.
  """
  alias Vapor.{Dense, Sampler}

  @doc """
  Sample `n` sequences of length `l` over `q` states from a Potts model
  with coupling `j` on `contacts` (`[{i, j}]`) and small random fields:
  E(s) = −Σ fields − Σ_contacts J·δ(sᵢ, sⱼ). `sweeps` Gibbs sweeps between samples.
  """
  def sample(l, contacts, opts \\ []) do
    q = Keyword.get(opts, :q, 2)
    jc = Keyword.get(opts, :coupling, 0.5)
    n = Keyword.get(opts, :n, 2000)
    seed = Keyword.get(opts, :seed, 1)
    sweeps = Keyword.get(opts, :sweeps, 4)
    nbr = Enum.reduce(contacts, %{}, fn {i, j}, m -> m |> Map.update(i, [j], &[j | &1]) |> Map.update(j, [i], &[i | &1]) end)
    fields = for i <- 0..(l - 1), do: (for a <- 0..(q - 1), do: 0.3 * (Sampler.uniform(seed + 7, i * q + a) - 0.5))
    ft = List.to_tuple(Enum.map(fields, &List.to_tuple/1))
    s0 = for i <- 0..(l - 1), do: trunc(Sampler.uniform(seed + 3, i) * q)

    {seqs, _} =
      Enum.map_reduce(1..n, {List.to_tuple(s0), 0}, fn _k, {s, ctr} ->
        {s, ctr} = Enum.reduce(1..sweeps, {s, ctr}, fn _, {s, ctr} ->
          Enum.reduce(0..(l - 1), {s, ctr}, fn i, {s, ctr} ->
            e = for a <- 0..(q - 1), do: elem(elem(ft, i), a) + jc * Enum.count(Map.get(nbr, i, []), &(elem(s, &1) == a))
            mx = Enum.max(e)
            w = Enum.map(e, &:math.exp(&1 - mx))
            z = Enum.sum(w)
            u = Sampler.uniform(seed, ctr) * z
            a = Enum.reduce_while(Enum.with_index(w), 0.0, fn {wi, a}, acc -> if acc + wi >= u, do: {:halt, {:pick, a}}, else: {:cont, acc + wi} end)
            a = case a do {:pick, x} -> x; _ -> q - 1 end
            {put_elem(s, i, a), ctr + 1}
          end)
        end)
        {Tuple.to_list(s), {s, ctr}}
      end)

    seqs
  end

  @doc "The alignment with every column permuted independently (the control: co-variation destroyed, conservation kept)."
  def shuffle(seqs, seed \\ 1) do
    cols = Enum.zip_with(seqs, & &1)
    cols = cols |> Enum.with_index() |> Enum.map(fn {c, i} -> Vapor.Modal.Rng.permute(c, seed * 1000 + i) end)
    Enum.zip_with(cols, & &1)
  end

  defp freqs(seqs, q, lambda) do
    n = length(seqs)
    l = length(hd(seqs))
    t = Enum.map(seqs, &List.to_tuple/1)
    fi = for i <- 0..(l - 1), into: %{}, do: {i, (for a <- 0..(q - 1), do: (Enum.count(t, &(elem(&1, i) == a)) * (1 - lambda) / n + lambda / q))}
    fij = for i <- 0..(l - 1), j <- (i + 1)..(l - 1)//1, into: %{} do
      counts = Enum.frequencies(Enum.map(t, &{elem(&1, i), elem(&1, j)}))
      {{i, j}, for(a <- 0..(q - 1), do: for(b <- 0..(q - 1), do: Map.get(counts, {a, b}, 0) * (1 - lambda) / n + lambda / (q * q)))}
    end
    {fi, fij, l}
  end

  @doc "Mutual information with APC: pairs (|i − j| ≥ `min_sep`) ranked by score."
  def mi(seqs, opts \\ []) do
    q = Keyword.get(opts, :q, 2)
    {fi, fij, l} = freqs(seqs, q, 0.0001)
    raw = for {{i, j}, m} <- fij, into: %{} do
      v = for a <- 0..(q - 1), b <- 0..(q - 1), (p = Enum.at(Enum.at(m, a), b)) > 0, reduce: 0.0 do
        acc -> acc + p * :math.log(p / (Enum.at(fi[i], a) * Enum.at(fi[j], b)))
      end
      {{i, j}, v}
    end
    rank(apc(raw, l), Keyword.get(opts, :min_sep, 6))
  end

  @doc "Mean-field DCA (Frobenius norm of −C⁻¹ blocks in the zero-sum gauge, with APC): pairs ranked by score."
  def dca(seqs, opts \\ []) do
    q = Keyword.get(opts, :q, 2)
    lambda = Keyword.get(opts, :pseudocount, 0.5)
    {fi, fij, l} = freqs(seqs, q, lambda)
    k = q - 1
    idx = fn i, a -> i * k + a end
    dim = l * k
    c = for r <- 0..(dim - 1) do
      {i, a} = {div(r, k), rem(r, k)}
      for cc <- 0..(dim - 1) do
        {j, b} = {div(cc, k), rem(cc, k)}
        cond do
          i == j -> (if a == b, do: Enum.at(fi[i], a) - Enum.at(fi[i], a) ** 2, else: -Enum.at(fi[i], a) * Enum.at(fi[i], b))
          i < j -> Enum.at(Enum.at(fij[{i, j}], a), b) - Enum.at(fi[i], a) * Enum.at(fi[j], b)
          true -> Enum.at(Enum.at(fij[{j, i}], b), a) - Enum.at(fi[i], a) * Enum.at(fi[j], b)
        end
      end
    end
    {:ok, inv} = Dense.inverse(c)
    it = inv |> Enum.map(&List.to_tuple/1) |> List.to_tuple()
    raw = for i <- 0..(l - 1), j <- (i + 1)..(l - 1)//1, into: %{} do
      # J_ij(a,b) = −(C⁻¹), the last state at 0; zero-sum gauge before the norm
      jm = for a <- 0..(q - 1), do: (for b <- 0..(q - 1), do: (if a < k and b < k, do: -elem(elem(it, idx.(i, a)), idx.(j, b)), else: 0.0))
      rowm = Enum.map(jm, &(Enum.sum(&1) / q))
      colm = Enum.zip_with(jm, & &1) |> Enum.map(&(Enum.sum(&1) / q))
      tot = Enum.sum(rowm) / q
      fro = for {r, a} <- Enum.with_index(jm), {x, b} <- Enum.with_index(r), reduce: 0.0, do: (acc -> acc + (x - Enum.at(rowm, a) - Enum.at(colm, b) + tot) ** 2)
      {{i, j}, :math.sqrt(fro)}
    end
    rank(apc(raw, l), Keyword.get(opts, :min_sep, 6))
  end

  defp apc(raw, l) do
    mean_i = for i <- 0..(l - 1), into: %{}, do: {i, (for {{a, b}, v} <- raw, a == i or b == i, reduce: 0.0, do: (acc -> acc + v)) / max(l - 1, 1)}
    all = Enum.sum(Map.values(raw)) / max(map_size(raw), 1)
    Map.new(raw, fn {{i, j}, v} -> {{i, j}, v - mean_i[i] * mean_i[j] / max(all, 1.0e-300)} end)
  end

  defp rank(scores, min_sep), do: scores |> Enum.filter(fn {{i, j}, _} -> j - i >= min_sep end) |> Enum.sort_by(&(-elem(&1, 1))) |> Enum.map(&elem(&1, 0))
end
