defmodule Vapor.Sampler do
  @moduledoc """
  Next-token choice, reproducible bit for bit on any machine.

  The logits are binary32 values from a certified program; everything after
  them uses only IEEE-754 binary64 `+ − × ÷` (identical on every
  conforming machine — the BEAM does no fused or extended arithmetic) and a
  counter-based generator, so a sequence's tokens depend only on its logits,
  its `seed` and its step — never on the batch it ran in, the thread count
  or the host.

    * `temperature: 0` → greedy: the first maximum (lowest id).
    * otherwise `z_i = logit_i / T`; candidates sorted by `(z desc, id asc)`;
      `top_k` keeps the first `k`; `p_i = exp(z_i − z_max)` with
      `exp/1` below; `top_p` keeps the shortest prefix whose running sum
      reaches `top_p · Σp`; a uniform `u ∈ [0, 1)` from SplitMix64 of
      `(seed, step)` selects the first candidate whose running sum exceeds
      `u · Σp_kept`.

  Candidates with `z < z_max − 38` are skipped. This is not an
  approximation: their `p = e^(z − z_max) < 2⁻⁵⁴`, while every running sum
  is `≥ 1` from its first (largest) term on, so under round-to-nearest
  adding such a term leaves the sum unchanged — no sum and no selection
  can differ.
  """
  import Bitwise

  @type params :: %{temperature: float, top_k: non_neg_integer, top_p: float, seed: non_neg_integer}

  @doc """
  Whether a request samples on the substrate (`Term.sample/2`): greedy, or
  plain temperature sampling (no top-k, no top-p). There the draw is made
  in binary32 with the canonical `exp`, in index order, from `(1/T, u)` =
  `native/2`; top-k/top-p requests use `sample/3` here, in binary64.
  """
  def native?(%{top_k: k, top_p: p}), do: k == 0 and p >= 1.0

  @doc "The substrate's per-row parameters `(1/T rounded to binary32, u)`, u with 24 random bits."
  def native(%{temperature: t}, _step) when t <= 0.0, do: {0.0, 0.0}

  def native(%{temperature: t, seed: seed}, step) do
    inv = Vapor.F32.to_float(Vapor.F32.from_float(1.0 / t))
    {inv, Float.floor(uniform(seed, step) * 16_777_216) / 16_777_216}
  end

  @doc "Default parameters (greedy)."
  def params(opts \\ []) do
    %{temperature: (Keyword.get(opts, :temperature) || 0.0) * 1.0, top_k: Keyword.get(opts, :top_k) || 0,
      top_p: (Keyword.get(opts, :top_p) || 1.0) * 1.0, seed: Keyword.get(opts, :seed) || 0}
  end

  @doc "Choose a token from one row of logits (a list of floats)."
  @spec sample([float] | binary, params, non_neg_integer) :: non_neg_integer
  def sample(logits, %{temperature: t} = _p, _step) when t <= 0.0, do: argmax(logits)

  def sample(logits, p, step) when is_binary(logits), do: sample(floats(logits), p, step)

  def sample(logits, %{temperature: t, top_k: k, top_p: top_p, seed: seed}, step) do
    z = Enum.map(logits, &(&1 / t))
    zmax = Enum.max(z)

    cands =
      z
      |> Enum.with_index()
      |> Enum.filter(fn {zi, _} -> zi >= zmax - 38.0 end)
      |> Enum.sort(fn {a, i}, {b, j} -> a > b or (a == b and i < j) end)

    cands = if k > 0, do: Enum.take(cands, k), else: cands
    ps = Enum.map(cands, fn {zi, i} -> {exp(zi - zmax), i} end)
    kept = nucleus(ps, top_p)
    total = Enum.reduce(kept, 0.0, fn {p, _}, s -> s + p end)
    target = uniform(seed, step) * total

    Enum.reduce_while(kept, 0.0, fn {p, i}, acc ->
      acc = acc + p
      if acc > target, do: {:halt, {:id, i}}, else: {:cont, acc}
    end)
    |> case do
      {:id, i} -> i
      # u·Σ rounds up to Σ only if u is within an ulp of 1: take the last
      _ -> kept |> List.last() |> elem(1)
    end
  end

  @doc """
  Choose a token among `allowed` ids only (`:all`, or a MapSet), with the
  rules of `sample/3` applied to that subset: greedy takes the first
  maximum among the allowed ids, sampling draws from the allowed
  candidates only. The draw uses the same `(seed, step)` uniform, so a
  constrained sequence is as reproducible as a free one. Allowed ids
  outside the row are ignored; with nothing allowed, `nil`.
  """
  def sample_masked(logits, :all, p, step), do: sample(logits, p, step)

  def sample_masked(logits, allowed, %{temperature: t}, _step) when is_binary(logits) and t <= 0.0 do
    v = div(byte_size(logits), 4)

    allowed
    |> Enum.filter(&(&1 < v))
    |> Enum.reduce(nil, fn i, best ->
      <<x::float-32-little>> = binary_part(logits, i * 4, 4)
      case best do
        nil -> {x, i}
        {bx, bi} -> if x > bx or (x == bx and i < bi), do: {x, i}, else: best
      end
    end)
    |> case do
      nil -> nil
      {_, i} -> i
    end
  end

  def sample_masked(logits, allowed, p, step) when is_binary(logits) do
    v = div(byte_size(logits), 4)
    ids = allowed |> Enum.filter(&(&1 < v)) |> Enum.sort()

    if ids == [] do
      nil
    else
      sub = Enum.map(ids, fn i -> <<x::float-32-little>> = binary_part(logits, i * 4, 4); x end)
      Enum.at(ids, sample(sub, p, step))
    end
  end

  @doc "Index of the first maximum (of floats, or of a binary of binary32 values)."
  def argmax(bin) when is_binary(bin), do: argmax_bin(bin, 0, nil, 0)

  def argmax([x | rest]) do
    {_, i, _} = Enum.reduce(rest, {x, 0, 1}, fn y, {m, mi, j} -> if y > m, do: {y, j, j + 1}, else: {m, mi, j + 1} end)
    i
  end

  # a certified program's logits are finite; a non-finite one would be a
  # substrate fault, and is refused loudly rather than ordered arbitrarily
  defp argmax_bin(<<x::float-32-little, rest::binary>>, i, best, bi) do
    if best == nil or x > best, do: argmax_bin(rest, i + 1, x, i), else: argmax_bin(rest, i + 1, best, bi)
  end

  defp argmax_bin(<<>>, _i, _best, bi), do: bi
  defp argmax_bin(<<b::32-little, _::binary>>, _, _, _), do: raise(ArithmeticError, "non-finite logit 0x#{Integer.to_string(b, 16)}")

  @doc "binary32 values (little-endian) as floats."
  def floats(bin), do: for(<<b::32-little <- bin>>, do: Vapor.F32.to_float(b))

  defp nucleus(ps, top_p) when top_p >= 1.0, do: ps

  defp nucleus(ps, top_p) do
    total = Enum.reduce(ps, 0.0, fn {p, _}, s -> s + p end)
    goal = top_p * total

    {kept, _} =
      Enum.reduce_while(ps, {[], 0.0}, fn {p, _} = c, {acc, s} ->
        s = s + p
        if s >= goal, do: {:halt, {[c | acc], s}}, else: {:cont, {[c | acc], s}}
      end)

    Enum.reverse(kept)
  end

  @doc "SplitMix64 of (seed, step) → a uniform binary64 in [0, 1) (53 random bits)."
  def uniform(seed, step) do
    x = band(seed * 0x9E3779B97F4A7C15 + (step + 1) * 0xBF58476D1CE4E5B9, 0xFFFF_FFFF_FFFF_FFFF)
    x = band(bxor(x, x >>> 30) * 0xBF58476D1CE4E5B9, 0xFFFF_FFFF_FFFF_FFFF)
    x = band(bxor(x, x >>> 27) * 0x94D049BB133111EB, 0xFFFF_FFFF_FFFF_FFFF)
    x = bxor(x, x >>> 31)
    (x >>> 11) / 9_007_199_254_740_992
  end

  @ln2_hi 0.6931471803691238
  @ln2_lo 1.9082149292705877e-10
  @inv_ln2 1.4426950408889634

  @doc """
  e^x for x ≤ 0 in binary64 from `+ − × ÷` only (the same bits everywhere):
  x = n·ln2 + r with |r| ≤ ln2/2 (Cody–Waite, two-part ln2), e^r by its
  Taylor polynomial to degree 13 (error < 2⁻⁵³ on that range), times 2ⁿ
  exactly. Results below 2⁻¹⁰²² are 0.
  """
  def exp(x) when x < -708.0, do: 0.0

  def exp(x) do
    n = Float.round(x * @inv_ln2)
    r = x - n * @ln2_hi - n * @ln2_lo
    poly = Enum.reduce(13..1//-1, 1.0, fn k, acc -> 1.0 + r * acc / k end)
    poly * pow2(trunc(n))
  end

  defp pow2(n) when n >= -1022 do
    <<f::float-64>> = <<(n + 1023) <<< 52::64>>
    f
  end

  defp pow2(_n), do: 0.0
end
