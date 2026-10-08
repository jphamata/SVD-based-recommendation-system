defmodule Vapor.Finance.Money do
  @moduledoc """
  Exact decimal money (docs/FINANCE.md §1).

  A ledger cannot round in binary: `0.1 + 0.2` is not `0.3` in IEEE-754,
  and a sum of a million cents drifts. Here an amount is an integer
  coefficient and a decimal scale — `%{c: 12345, e: 2}` is 123.45 — so
  addition, subtraction and multiplication are exact, and every rounding
  is an explicit act with a named mode:

  | mode | rule | used by |
  |---|---|---|
  | `:half_even` | ties to the even neighbour | IEEE-754 default, banking |
  | `:half_up` | ties away from zero | most tax and invoice rules |
  | `:half_down` | ties toward zero | — |
  | `:down` / `:truncate` | toward zero | ANBIMA's PU and rate truncations |
  | `:up` | away from zero | — |
  | `:floor`, `:ceiling` | toward −∞ / +∞ | — |

  `allocate/3` splits an amount in proportion to weights by the largest
  remainder method, so the parts **sum to the whole exactly** — the
  invariant the certificate checks.
  """
  import Bitwise, only: [<<<: 2]

  @type t :: %{c: integer, e: non_neg_integer}

  @doc "Parse a decimal string (`\"-1234.5678\"`, `\"1e-3\"`, `\"1_000.00\"`) or an integer."
  def parse(n) when is_integer(n), do: {:ok, %{c: n, e: 0}}

  def parse(s) when is_binary(s) do
    s = s |> String.trim() |> String.replace("_", "")
    case Regex.run(~r/^([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/, s) do
      [_ | parts] when s not in ["", "+", "-", ".", "-.", "+."] ->
        [sign, int, frac, exp] = parts ++ List.duplicate("", 4 - length(parts))
        if int == "" and frac == "", do: {:error, "not a number: #{inspect(s)}"}, else: (
          digits = String.to_integer(if(int == "", do: "0", else: int) <> frac)
          scale = String.length(frac) - if(exp == "", do: 0, else: String.to_integer(exp))
          c = if sign == "-", do: -digits, else: digits
          {:ok, if(scale >= 0, do: %{c: c, e: scale}, else: %{c: c * pow10(-scale), e: 0})})
      _ -> {:error, "not a number: #{inspect(s)}"}
    end
  end

  def parse!(s), do: (case parse(s) do {:ok, m} -> m; {:error, w} -> raise ArgumentError, w end)

  @doc "An exact decimal from a float's shortest round-trip representation (never its binary expansion)."
  def from_float(x) when is_float(x), do: parse!(:erlang.float_to_binary(x, [:short]))

  def to_float(%{c: c, e: e}), do: c / pow10(e)

  @doc "The decimal string, with exactly `e` places."
  def to_string(%{c: c, e: 0}), do: Integer.to_string(c)

  def to_string(%{c: c, e: e}) do
    s = c |> abs() |> Integer.to_string() |> String.pad_leading(e + 1, "0")
    {i, f} = String.split_at(s, String.length(s) - e)
    if(c < 0, do: "-", else: "") <> i <> "." <> f
  end

  @doc "A rational {num, den} with den > 0 (for the exact checks of other modules)."
  def to_rational(%{c: c, e: e}), do: {c, pow10(e)}

  def pow10(k) when k >= 0, do: Integer.pow(10, k)

  defp align(%{c: a, e: ea}, %{c: b, e: eb}) do
    e = max(ea, eb)
    {a * pow10(e - ea), b * pow10(e - eb), e}
  end

  def add(x, y), do: ({a, b, e} = align(x, y); %{c: a + b, e: e})
  def sub(x, y), do: ({a, b, e} = align(x, y); %{c: a - b, e: e})
  def mul(%{c: a, e: ea}, %{c: b, e: eb}), do: %{c: a * b, e: ea + eb}
  def neg(%{c: a} = x), do: %{x | c: -a}
  def compare(x, y), do: ({a, b, _} = align(x, y); cond do a < b -> :lt; a > b -> :gt; true -> :eq end)
  def zero?(%{c: c}), do: c == 0
  def sum(xs, scale \\ 0), do: Enum.reduce(xs, %{c: 0, e: scale}, &add(&2, &1))

  @doc "Round to `scale` places with `mode` (exact: no binary intermediate)."
  def round(%{c: c, e: e}, scale, mode \\ :half_even) do
    cond do
      scale >= e -> %{c: c * pow10(scale - e), e: scale}
      true -> %{c: div_round(c, pow10(e - scale), mode), e: scale}
    end
  end

  @doc "x / y rounded to `scale` places (`mode`)."
  def divide(%{c: a, e: ea}, %{c: b, e: eb}, scale, mode \\ :half_even) when b != 0 do
    # a·10^-ea / (b·10^-eb) = (a·10^(eb+scale-ea)) / b · 10^-scale
    k = eb + scale - ea
    {num, den} = if k >= 0, do: {a * pow10(k), b}, else: {a, b * pow10(-k)}
    %{c: div_round(num, den, mode), e: scale}
  end

  @doc "Integer division n/d (d ≠ 0) rounded by `mode` — the single rounding primitive."
  def div_round(n, d, mode) do
    {n, d} = if d < 0, do: {-n, -d}, else: {n, d}
    q = Integer.floor_div(n, d)
    r = n - q * d            # 0 ≤ r < d
    cond do
      r == 0 -> q
      true ->
        twice = 2 * r
        tie = twice == d
        above = twice > d     # the fraction is > 1/2
        neg = n < 0
        case mode do
          :floor -> q
          :ceiling -> q + 1
          m when m in [:down, :truncate] -> if neg, do: q + 1, else: q
          :up -> if neg, do: q, else: q + 1
          :half_up -> cond do above -> q + 1; tie -> (if neg, do: q, else: q + 1); true -> q end
          :half_down -> cond do above -> q + 1; tie -> (if neg, do: q + 1, else: q); true -> q end
          :half_even -> cond do above -> q + 1; tie -> (if rem(q, 2) == 0, do: q, else: q + 1); true -> q end
        end
    end
  end

  @doc """
  Split `total` (a Money) into parts proportional to `weights` (positive
  numbers or Money), each with `scale` places, by the largest remainder
  method: every part gets the floor of its exact share and the leftover
  units go to the largest remainders (ties to the lower index). The parts
  sum to `total` exactly; `{:ok, parts, certificate}`.
  """
  def allocate(total, weights, scale \\ 2) do
    ws = Enum.map(weights, fn w -> if is_map(w), do: to_rational(w), else: rational_of(w) end)
    cond do
      ws == [] -> {:error, "no weights"}
      Enum.any?(ws, fn {n, _} -> n < 0 end) -> {:error, "weights must be ≥ 0"}
      true ->
        t = __MODULE__.round(total, max(scale, total.e), :half_even)
        units = if t.e > scale, do: nil, else: t.c * pow10(scale - t.e)
        if units == nil or __MODULE__.round(total, scale, :down) |> compare(total) != :eq do
          {:error, "the total has more places than #{scale}: round it first"}
        else
          # common denominator of the weights: integer weights W_i, exact shares units·W_i/ΣW
          den = Enum.reduce(ws, 1, fn {_, d}, acc -> lcm(acc, d) end)
          wi = Enum.map(ws, fn {n, d} -> n * div(den, d) end)
          sw = Enum.sum(wi)
          if sw == 0 do
            {:error, "the weights sum to zero"}
          else
            shares = Enum.map(wi, fn w -> {Integer.floor_div(units * w, sw), Integer.mod(units * w, sw)} end)
            left = units - Enum.sum(Enum.map(shares, &elem(&1, 0)))
            bonus = shares |> Enum.with_index() |> Enum.sort_by(fn {{_, r}, i} -> {-r, i} end) |> Enum.take(left) |> MapSet.new(fn {_, i} -> i end)
            parts = shares |> Enum.with_index() |> Enum.map(fn {{q, _}, i} -> %{c: q + if(MapSet.member?(bonus, i), do: 1, else: 0), e: scale} end)
            # certificate: the parts sum to the whole, and each is within one unit of its exact share
            exact_ok = Enum.zip(parts, wi) |> Enum.all?(fn {p, w} -> abs(p.c * sw - units * w) < sw end)
            {:ok, parts, %{sum_equals_total: Enum.sum(Enum.map(parts, & &1.c)) == units, within_one_unit: exact_ok, leftover_units: left}}
          end
        end
    end
  end

  defp rational_of(w) when is_integer(w), do: {w, 1}
  defp rational_of(w) when is_float(w), do: to_rational(from_float(w))
  defp rational_of(w) when is_binary(w), do: to_rational(parse!(w))

  defp lcm(a, b), do: div(a * b, Integer.gcd(a, b))

  @doc """
  Compound a principal at an annual rate over `du` business days on the
  252 basis, the way the Brazilian market writes it — `P·(1 + r)^(du/252)`
  — with the factor truncated at `places` (ANBIMA truncates the factor of
  the DI at 8 places and PUs at 6). The power is computed by exact
  rational arithmetic to 40 digits before the truncation, so the decision
  at the 8th place is not a binary accident.
  """
  def factor_252(rate, du, places \\ 8) do
    r = to_rational(rate)
    # (1 + r)^(du/252): exact when du is a multiple of 252; otherwise a
    # 40-digit fixed-point power (ln/exp in big integers)
    {n, d} = r
    base = {d + n, d}
    {:ok, fixed_power(base, du, 252, 40) |> then(fn f -> __MODULE__.round(f, places, :down) end)}
  end

  # x^(p/q) for a positive rational x, to `digits` decimal places, by
  # Newton on y^q = x^p in big integers (exact up to the last digit, rounded down)
  defp fixed_power({xn, xd}, p, q, digits) do
    g = Integer.gcd(p, q); {p, q} = {div(p, g), div(q, g)}
    # target T = (xn/xd)^p, find y = T^(1/q) with y = Y/10^digits
    s = pow10(digits)
    tn = Integer.pow(xn, p); td = Integer.pow(xd, p)
    # Y = floor((tn/td)^(1/q) · s) = floor((tn·s^q / td)^(1/q))
    big = div(tn * Integer.pow(s, q), td)
    %{c: iroot(big, q), e: digits}
  end

  # floor(n^(1/k)) for n ≥ 0 by Newton's method on integers
  @doc false
  def iroot(0, _), do: 0
  def iroot(n, 1), do: n
  def iroot(n, k) do
    bits = bit_length(n)
    x0 = 1 <<< (div(bits, k) + 1)
    newton_root(n, k, x0)
  end

  defp newton_root(n, k, x) do
    y = div((k - 1) * x + div(n, Integer.pow(x, k - 1)), k)
    if y >= x, do: fix_root(n, k, x), else: newton_root(n, k, y)
  end

  defp fix_root(n, k, x) do
    cond do
      Integer.pow(x, k) > n -> fix_root(n, k, x - 1)
      Integer.pow(x + 1, k) <= n -> fix_root(n, k, x + 1)
      true -> x
    end
  end

  defp bit_length(n), do: length(Integer.digits(n, 2))
end
