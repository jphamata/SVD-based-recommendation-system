defmodule Vapor.Modal.Rng do
  @moduledoc """
  Deterministic pseudo-randomness for the modal layer, the merger and the
  quality gate: splitmix64 (`Vapor.Tensor.splitmix/1`), the same numbers on
  every host. Uniform values are 53-bit dyadic rationals in `[0, 1)`.
  """
  import Bitwise

  @doc "`n` uniforms in [0, 1)."
  def uniform(seed, n) do
    {vals, _} = Enum.map_reduce(1..n//1, seed, fn _, s -> Vapor.Tensor.splitmix(s) end)
    Enum.map(vals, &((&1 >>> 11) / 9_007_199_254_740_992))
  end

  @doc "`n` standard normals (Box–Muller over `uniform/2`; libm-free: `Vapor.CR`)."
  def normal(seed, n) do
    us = uniform(seed, 2 * div(n + 1, 2))

    us
    |> Enum.chunk_every(2)
    |> Enum.flat_map(fn [u1, u2] ->
      r = :math.sqrt(-2.0 * Vapor.CR.log_f64(max(u1, 1.0e-300)))
      a = 2.0 * :math.pi() * u2
      [r * Vapor.CR.cos_f64(a), r * Vapor.CR.sin_f64(a)]
    end)
    |> Enum.take(n)
  end

  @doc "A seeded permutation of a list (Fisher–Yates order by sort keys)."
  def permute(list, seed) do
    keys = uniform(seed, length(list))
    list |> Enum.zip(keys) |> Enum.sort_by(&elem(&1, 1)) |> Enum.map(&elem(&1, 0))
  end

  @doc "A 64-bit key for `(seed, parts…)`: SHA-256 of the canonical encoding, first 8 bytes."
  def key(parts) do
    <<k::64, _::binary>> = :crypto.hash(:sha256, Vapor.Canonical.encode(parts))
    k
  end
end
