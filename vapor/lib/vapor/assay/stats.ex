defmodule Vapor.Assay.Stats do
  @moduledoc """
  The statistics under the Assay (docs/ASSAY.md §8), in-tree and
  deterministic: every resampling takes a seed, so a report is a function
  of its data.
  """

  def mean([]), do: nil
  def mean(xs), do: Enum.sum(xs) / length(xs)

  def var(xs) when length(xs) < 2, do: 0.0
  def var(xs), do: (m = mean(xs); Enum.reduce(xs, 0.0, &(&2 + (&1 - m) * (&1 - m))) / (length(xs) - 1))

  def sd(xs), do: :math.sqrt(var(xs))

  @doc "Standard normal CDF."
  def phi(z), do: 0.5 * :math.erfc(-z / :math.sqrt(2))

  @doc "Standard normal quantile (Acklam's rational approximation, refined by one Halley step)."
  def qnorm(p) when p <= 0, do: -1.0e300
  def qnorm(p) when p >= 1, do: 1.0e300

  def qnorm(p) do
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    poly = fn cs, x -> Enum.reduce(cs, 0.0, &(&2 * x + &1)) end

    x =
      cond do
        p < pl -> (q = :math.sqrt(-2 * :math.log(p)); poly.(c, q) / (poly.(d, q) * q + 1))
        p > 1 - pl -> (q = :math.sqrt(-2 * :math.log(1 - p)); -poly.(c, q) / (poly.(d, q) * q + 1))
        true -> (q = p - 0.5; r = q * q; poly.(a, r) * q / (poly.(b, r) * r + 1))
      end

    e = phi(x) - p
    u = e * :math.sqrt(2 * :math.pi()) * :math.exp(x * x / 2)
    x - u / (1 + x * u / 2)
  end

  @doc "A seeded generator."
  def rng(seed), do: :rand.seed_s(:exsss, {seed, 1_000_003, 77})

  @doc "Percentile of a sorted tuple (linear interpolation)."
  def pct(sorted, q) when is_tuple(sorted) do
    n = tuple_size(sorted)
    h = (n - 1) * q
    lo = trunc(Float.floor(h))
    hi = min(lo + 1, n - 1)
    elem(sorted, lo) + (h - lo) * (elem(sorted, hi) - elem(sorted, lo))
  end

  @doc "Bootstrap distribution of `stat` (a function of a list of indices) over n items: a sorted tuple."
  def bootstrap(n, reps, seed, stat) do
    {vals, _} =
      Enum.map_reduce(1..reps, rng(seed), fn _, r ->
        {idx, r} = Enum.map_reduce(1..n, r, fn _, rr -> :rand.uniform_s(n, rr) end)
        {stat.(Enum.map(idx, &(&1 - 1))), r}
      end)
    vals |> Enum.reject(&is_nil/1) |> Enum.sort() |> List.to_tuple()
  end

  @doc "Paired sign-flip permutation test of mean(d) = 0, two-sided (exact for n ≤ 16)."
  def sign_flip(ds, reps, seed) do
    n = length(ds)
    obs = abs(Enum.sum(ds))
    t = List.to_tuple(ds)
    if n <= 16 do
      total = Integer.pow(2, n)
      hits = Enum.count(0..(total - 1), fn mask ->
        s = Enum.reduce(0..(n - 1), 0.0, fn i, acc -> if Bitwise.band(mask, Bitwise.bsl(1, i)) != 0, do: acc - elem(t, i), else: acc + elem(t, i) end)
        abs(s) >= obs - 1.0e-12
      end)
      hits / total
    else
      {hits, _} =
        Enum.reduce(1..reps, {0, rng(seed)}, fn _, {h, r} ->
          {s, r} = Enum.reduce(0..(n - 1), {0.0, r}, fn i, {acc, rr} -> {u, rr} = :rand.uniform_s(rr); {if(u < 0.5, do: acc - elem(t, i), else: acc + elem(t, i)), rr} end)
          {if(abs(s) >= obs - 1.0e-12, do: h + 1, else: h), r}
        end)
      (hits + 1) / (reps + 1)
    end
  end

  @doc "Exact two-sided binomial test p-value for k successes in n at p = 0.5 (McNemar)."
  def binom_two_sided(k, n) when n == 0, do: (_ = k; 1.0)

  def binom_two_sided(k, n) do
    lf = Enum.scan(1..n, 0.0, fn i, a -> a + :math.log(i) end) |> then(&List.to_tuple([0.0 | &1]))
    pmf = fn i -> :math.exp(elem(lf, n) - elem(lf, i) - elem(lf, n - i) - n * :math.log(2)) end
    pk = pmf.(k)
    Enum.reduce(0..n, 0.0, fn i, s -> p = pmf.(i); if p <= pk * (1 + 1.0e-9), do: s + p, else: s end) |> min(1.0)
  end

  @doc "Holm's step-down adjustment of p-values (same order as given)."
  def holm(ps) do
    m = length(ps)
    {adj, _} =
      ps |> Enum.with_index() |> Enum.sort_by(&elem(&1, 0))
      |> Enum.with_index()
      |> Enum.map_reduce(0.0, fn {{p, i}, k}, running -> v = min(max(running, (m - k) * p), 1.0); {{i, v}, v} end)
    adj |> Enum.sort() |> Enum.map(&elem(&1, 1))
  end

  @doc "Benjamini–Hochberg adjusted p-values (same order)."
  def bh(ps) do
    m = length(ps)
    sorted = ps |> Enum.with_index() |> Enum.sort_by(&elem(&1, 0), :desc)
    {adj, _} =
      sorted |> Enum.with_index() |> Enum.map_reduce(1.0, fn {{p, i}, k}, running ->
        rank = m - k
        v = min(running, p * m / rank)
        {{i, v}, v}
      end)
    adj |> Enum.sort() |> Enum.map(&elem(&1, 1))
  end

  @doc "Ranks with ties averaged (1 = smallest)."
  def ranks(xs) do
    xs
    |> Enum.with_index()
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.chunk_by(&elem(&1, 0))
    |> Enum.flat_map_reduce(1, fn g, start -> {Enum.map(g, fn {_, i} -> {i, start + (length(g) - 1) / 2} end), start + length(g)} end)
    |> elem(0)
    |> Enum.sort()
    |> Enum.map(&elem(&1, 1))
  end

  @doc "Read a table: CSV/TSV with a header, or JSON lines. `{:ok, header, rows (lists of strings or values)}`."
  def table(text) do
    lines = text |> String.split("\n") |> Enum.map(&String.trim_trailing/1) |> Enum.reject(&(String.trim(&1) == "" or String.starts_with?(String.trim(&1), "#")))
    case lines do
      [] -> {:error, "no data"}
      ["{" <> _ | _] ->
        objs = Enum.map(lines, &Vapor.JSON.decode/1)
        if Enum.all?(objs, &match?({:ok, %{}}, &1)) do
          maps = Enum.map(objs, &elem(&1, 1))
          header = maps |> Enum.flat_map(&Map.keys/1) |> Enum.uniq()
          {:ok, header, Enum.map(maps, fn m -> Enum.map(header, &Map.get(m, &1)) end)}
        else
          {:error, "a line is not a JSON object"}
        end
      [h | rows] ->
        sep = cond do String.contains?(h, "\t") -> "\t"; String.contains?(h, ";") and not String.contains?(h, ",") -> ";"; true -> "," end
        split = fn l -> l |> String.split(sep) |> Enum.map(&(&1 |> String.trim() |> String.trim("\""))) end
        header = split.(h)
        rows = Enum.map(rows, split)
        bad = Enum.find_index(rows, &(length(&1) != length(header)))
        if bad, do: {:error, "row #{bad + 2} has #{length(Enum.at(rows, bad))} fields, the header #{length(header)}"}, else: {:ok, header, rows}
    end
  end

  @doc "A field as a number (strings parsed), or nil."
  def num(nil), do: nil
  def num(x) when is_number(x), do: x * 1.0
  def num(true), do: 1.0
  def num(false), do: 0.0
  def num(s) when is_binary(s) do
    case Float.parse(String.trim(s)) do
      {x, ""} -> x
      {x, "%"} -> x / 100
      _ -> case String.downcase(String.trim(s)) do "true" -> 1.0; "false" -> 0.0; "yes" -> 1.0; "no" -> 0.0; _ -> nil end
    end
  end
  def num(_), do: nil

  @doc "Format a number for reports."
  def f(nil), do: "—"
  def f(x) when is_integer(x), do: Integer.to_string(x)
  def f(x) when is_float(x) and abs(x) >= 1.0e-3 and abs(x) < 1.0e6, do: :erlang.float_to_binary(x, [{:decimals, 4}, :compact])
  def f(x) when is_float(x), do: :erlang.float_to_binary(x, [{:scientific, 2}])
  def f(x), do: to_string(x)
end
