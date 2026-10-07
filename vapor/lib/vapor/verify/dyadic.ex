defmodule Vapor.Verify.Dyadic do
  @moduledoc """
  Exact dyadic rationals `m·2^e` (arbitrary-precision `m`) — the number system
  of every error bound in the ladder. Every finite binary32 is one; sums and
  products stay dyadic, so the verifier never rounds. Only *bounds* are ever
  shortened, and only upward (`round_up/2`), which keeps them rigorous.
  """
  import Bitwise

  @type t :: {integer, integer}

  def zero, do: {0, 0}
  def one, do: {1, 0}
  def pow2(e), do: {1, e}
  def of_int(n), do: {n, 0}
  def of_f32(bits), do: Vapor.F32.to_dyadic(bits)

  def add({0, _}, y), do: y
  def add(x, {0, _}), do: x
  def add(x, y), do: Vapor.F32.dyadic_add(x, y)
  def neg({m, e}), do: {-m, e}
  def sub(x, y), do: add(x, neg(y))
  def mul({m1, e1}, {m2, e2}), do: {m1 * m2, e1 + e2}
  def abs({m, e}), do: {Kernel.abs(m), e}
  def sum(xs), do: Enum.reduce(xs, zero(), &add/2)

  def compare(x, y) do
    {m, _} = sub(x, y)

    cond do
      m < 0 -> :lt
      m > 0 -> :gt
      true -> :eq
    end
  end

  def le?(x, y), do: compare(x, y) != :gt
  def max(x, y), do: if(le?(x, y), do: y, else: x)

  @doc "Upper bound of a non-negative dyadic with at most `bits` significant bits."
  def round_up({m, e}, bits \\ 80) when m >= 0 do
    l = bitlen(m)

    if l <= bits do
      {m, e}
    else
      sh = l - bits
      q = m >>> sh
      q = if q <<< sh == m, do: q, else: q + 1
      {q, e + sh}
    end
  end

  @doc "Scale non-negative dyadics to integers over a common exponent."
  def to_common_ints(xs) do
    emin = xs |> Enum.map(&elem(&1, 1)) |> Enum.min()
    Enum.map(xs, fn {m, e} -> m <<< (e - emin) end)
  end

  @doc "Upward dyadic bound of γₙ = n·u/(1 − n·u), u = 2⁻²⁴."
  def gamma(n) when n >= 0 and n < 16_777_216 do
    # γₙ = n / (2²⁴ − n); round up with 64 fractional bits
    num = n <<< 64
    den = 16_777_216 - n
    q = div(num, den)
    q = if q * den == num, do: q, else: q + 1
    {q, -64}
  end

  def to_float({m, e}), do: m * :math.pow(2, e)

  defp bitlen(0), do: 0
  defp bitlen(m), do: bitlen(m, 0)
  defp bitlen(0, n), do: n
  defp bitlen(m, n) when m >= 1 <<< 64, do: bitlen(m >>> 64, n + 64)
  defp bitlen(m, n), do: bitlen(m >>> 1, n + 1)
end
