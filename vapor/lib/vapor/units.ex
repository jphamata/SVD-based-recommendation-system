defmodule Vapor.Units do
  @moduledoc """
  Physical units with dimensional analysis (docs/WORKBENCH.md §2).

  A dimension is the vector of exponents over the seven SI base
  quantities `{L, M, T, I, Θ, N, J}` (metre, kilogram, second, ampere,
  kelvin, mole, candela); a unit is a factor to SI and a dimension. The
  whole of an expression is checked before anything is computed: adding
  metres to seconds, taking the sine of a length, an ODE whose right side
  is not the left side per second — each is an error that names the
  quantities involved, the way the Mars Climate Orbiter's
  pound-force-seconds against newton-seconds should have been.

  Exponents are kept as floats rounded to 10⁻⁹ (√(m²) is m; √m is m^0.5,
  said as such). Temperatures are absolute (K, °R); `degC`/`degF` are
  **intervals** (a difference of one degree).

  **Affine scales** (0.13): `°C` and `°F` (also `celsius`, `fahrenheit`)
  are readings on a thermometer, not multiples of a unit — 25[°C] is
  298.15 K, and `T in [°C]` subtracts the offset back. They are accepted
  exactly where they have one meaning: after a number, alone, and as the
  target of `in`. Inside a compound unit (J/(kg·°C)) or after an
  expression they are refused with the reason, and `degC` (the interval)
  is suggested: the classic error is to treat a reading as a difference.
  """

  @base ~w(m kg s A K mol cd)
  @zero {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}

  @doc "The dimensionless dimension."
  def none, do: @zero

  defp d(l, m, t, i \\ 0, th \\ 0, n \\ 0, j \\ 0), do: {l * 1.0, m * 1.0, t * 1.0, i * 1.0, th * 1.0, n * 1.0, j * 1.0}

  # name → {factor to SI, dimension}; exact names win over prefix + name
  defp table do
    pi = :math.pi()

    %{
      "m" => {1.0, d(1, 0, 0)}, "g" => {1.0e-3, d(0, 1, 0)}, "s" => {1.0, d(0, 0, 1)}, "A" => {1.0, d(0, 0, 0, 1)},
      "K" => {1.0, d(0, 0, 0, 0, 1)}, "mol" => {1.0, d(0, 0, 0, 0, 0, 1)}, "cd" => {1.0, d(0, 0, 0, 0, 0, 0, 1)},
      "Hz" => {1.0, d(0, 0, -1)}, "N" => {1.0, d(1, 1, -2)}, "Pa" => {1.0, d(-1, 1, -2)}, "J" => {1.0, d(2, 1, -2)},
      "W" => {1.0, d(2, 1, -3)}, "C" => {1.0, d(0, 0, 1, 1)}, "V" => {1.0, d(2, 1, -3, -1)}, "F" => {1.0, d(-2, -1, 4, 2)},
      "ohm" => {1.0, d(2, 1, -3, -2)}, "Ω" => {1.0, d(2, 1, -3, -2)}, "S" => {1.0, d(-2, -1, 3, 2)}, "Wb" => {1.0, d(2, 1, -2, -1)},
      "T" => {1.0, d(0, 1, -2, -1)}, "H" => {1.0, d(2, 1, -2, -2)}, "L" => {1.0e-3, d(3, 0, 0)}, "l" => {1.0e-3, d(3, 0, 0)},
      "min" => {60.0, d(0, 0, 1)}, "h" => {3600.0, d(0, 0, 1)}, "hr" => {3600.0, d(0, 0, 1)}, "day" => {86_400.0, d(0, 0, 1)},
      "yr" => {3.15576e7, d(0, 0, 1)}, "bar" => {1.0e5, d(-1, 1, -2)}, "atm" => {101_325.0, d(-1, 1, -2)},
      "psi" => {6894.757293168361, d(-1, 1, -2)}, "mmHg" => {133.322387415, d(-1, 1, -2)}, "in" => {0.0254, d(1, 0, 0)},
      "ft" => {0.3048, d(1, 0, 0)}, "yd" => {0.9144, d(1, 0, 0)}, "mi" => {1609.344, d(1, 0, 0)}, "nmi" => {1852.0, d(1, 0, 0)},
      "lb" => {0.45359237, d(0, 1, 0)}, "lbm" => {0.45359237, d(0, 1, 0)}, "lbf" => {4.4482216152605, d(1, 1, -2)},
      "kip" => {4448.2216152605, d(1, 1, -2)}, "t" => {1000.0, d(0, 1, 0)}, "tonne" => {1000.0, d(0, 1, 0)},
      "eV" => {1.602176634e-19, d(2, 1, -2)}, "cal" => {4.184, d(2, 1, -2)}, "Wh" => {3600.0, d(2, 1, -2)},
      "BTU" => {1055.05585262, d(2, 1, -2)}, "hp" => {745.69987158227, d(2, 1, -3)}, "rad" => {1.0, @zero}, "sr" => {1.0, @zero},
      "deg" => {pi / 180, @zero}, "°" => {pi / 180, @zero}, "rpm" => {2 * pi / 60, d(0, 0, -1)}, "%" => {0.01, @zero},
      "ppm" => {1.0e-6, @zero}, "degC" => {1.0, d(0, 0, 0, 0, 1)}, "degF" => {5 / 9, d(0, 0, 0, 0, 1)}, "R" => {5 / 9, d(0, 0, 0, 0, 1)},
      "M" => {1000.0, d(-3, 0, 0, 0, 0, 1)}, "Da" => {1.66053906660e-27, d(0, 1, 0)}, "Å" => {1.0e-10, d(1, 0, 0)},
      "angstrom" => {1.0e-10, d(1, 0, 0)}, "gal" => {3.785411784e-3, d(3, 0, 0)}, "knot" => {1852 / 3600, d(1, 0, -1)},
      "VA" => {1.0, d(2, 1, -3)}, "var" => {1.0, d(2, 1, -3)}, "Gy" => {1.0, d(2, 0, -2)}, "Sv" => {1.0, d(2, 0, -2)},
      "Bq" => {1.0, d(0, 0, -1)}, "kat" => {1.0, d(0, 0, -1, 0, 0, 1)}, "lm" => {1.0, d(0, 0, 0, 0, 0, 0, 1)}, "lx" => {1.0, d(-2, 0, 0, 0, 0, 0, 1)}
    }
  end

  @prefixes [{"da", 1.0e1}, {"Y", 1.0e24}, {"Z", 1.0e21}, {"E", 1.0e18}, {"P", 1.0e15}, {"T", 1.0e12}, {"G", 1.0e9}, {"M", 1.0e6},
             {"k", 1.0e3}, {"h", 1.0e2}, {"d", 1.0e-1}, {"c", 1.0e-2}, {"m", 1.0e-3}, {"u", 1.0e-6}, {"µ", 1.0e-6}, {"μ", 1.0e-6},
             {"n", 1.0e-9}, {"p", 1.0e-12}, {"f", 1.0e-15}, {"a", 1.0e-18}]

  # units that take no prefix (a "ft" is not femto-tonnes)
  @no_prefix ~w(min h hr day yr in ft yd mi nmi lb lbm lbf kip psi atm mmHg BTU hp rpm deg ° % ppm degC degF R gal knot Å angstrom t tonne)

  @affine %{"°C" => {1.0, 273.15}, "celsius" => {1.0, 273.15}, "°F" => {5 / 9, 459.67 * 5 / 9}, "fahrenheit" => {5 / 9, 459.67 * 5 / 9}}

  @doc "An affine temperature scale by name: `{:ok, {factor, offset}}` (SI = x·factor + offset) or `:none`."
  def affine(u), do: (case Map.fetch(@affine, String.trim(u)) do {:ok, v} -> {:ok, v}; :error -> :none end)

  @doc "The unit names known (without prefixes), sorted."
  def names, do: table() |> Map.keys() |> Enum.sort()

  @doc """
  Parse a unit expression (`"kg*m/s^2"`, `"kN·m"`, `"m/s²"`, `"1/s"`,
  `"(W/m^2)/K"`): `{:ok, {factor, dimension}}` or `{:error, why}`.
  """
  def parse(str) when is_binary(str) do
    s = str |> String.replace(["·", "⋅", "×"], "*") |> String.replace("²", "^2") |> String.replace("³", "^3") |> String.replace("⁻¹", "^-1") |> String.trim()

    cond do
      s == "" or s == "1" -> {:ok, {1.0, @zero}}
      match?({:ok, _}, affine(s)) -> (({:ok, {f, _}} = affine(s)); {:ok, {f, d(0, 0, 0, 0, 1)}})
      true ->
      case uexpr(tokens(s)) do
        {:ok, v, []} -> {:ok, v}
        {:ok, _, rest} -> {:error, "unit: unexpected #{inspect(Enum.join(Enum.map(rest, &tok_s/1)))} in #{inspect(str)}"}
        {:error, _} = e -> e
      end
    end
  catch
    {:unit_error, why} -> {:error, why}
  end

  defp tok_s({:w, w}), do: w
  defp tok_s({:n, n}), do: to_string(n)
  defp tok_s(c), do: to_string([c])

  defp tokens(s), do: tokens(String.to_charlist(s), [])
  defp tokens([], acc), do: Enum.reverse(acc)
  defp tokens([c | r], acc) when c in [?\s, ?\t], do: tokens(r, acc)
  defp tokens([c | r], acc) when c in [?*, ?/, ?^, ?(, ?)], do: tokens(r, [c | acc])

  defp tokens([c | _] = l, acc) when c in ?0..?9 or c == ?- or c == ?. do
    {num, rest} = Enum.split_while(l, &(&1 in ?0..?9 or &1 in [?-, ?.]))
    case Float.parse(to_string(num)) do
      {f, ""} -> tokens(rest, [{:n, f} | acc])
      _ -> throw({:unit_error, "unit: bad number #{to_string(num)}"})
    end
  end

  defp tokens(l, acc) do
    {w, rest} = Enum.split_while(l, &(&1 not in [?\s, ?*, ?/, ?^, ?(, ?)]))
    tokens(rest, [{:w, to_string(w)} | acc])
  end

  defp uexpr(toks) do
    with {:ok, a, rest} <- upow(toks), do: umore(a, rest)
  end

  defp umore(a, [?* | r]), do: (with {:ok, b, r2} <- upow(r), do: umore(mul(a, b), r2))
  defp umore(a, [?/ | r]), do: (with {:ok, b, r2} <- upow(r), do: umore(mul(a, pow(b, -1.0)), r2))
  defp umore(a, [{:w, _} | _] = r), do: (with {:ok, b, r2} <- upow(r), do: umore(mul(a, b), r2))
  defp umore(a, r), do: {:ok, a, r}

  defp upow(toks) do
    with {:ok, a, rest} <- uatom(toks) do
      case rest do
        [?^, ?( , {:n, e}, ?/, {:n, q}, ?) | r] -> {:ok, pow(a, e / q), r}
        [?^, {:n, e} | r] -> {:ok, pow(a, e), r}
        [?^ | _] -> {:error, "unit: ^ needs a number"}
        _ -> {:ok, a, rest}
      end
    end
  end

  defp uatom([?( | r]) do
    case uexpr(r) do
      {:ok, v, [?) | r2]} -> {:ok, v, r2}
      {:ok, _, _} -> {:error, "unit: missing )"}
      e -> e
    end
  end

  defp uatom([{:n, n} | r]), do: {:ok, {n, @zero}, r}
  defp uatom([{:w, w} | r]), do: (with {:ok, u} <- unit(w), do: {:ok, u, r})
  defp uatom(_), do: {:error, "unit: expected a unit name"}

  @doc "One unit name, possibly prefixed: `{:ok, {factor, dim}}`."
  def unit(w) do
    t = table()
    if Map.has_key?(@affine, w), do: throw({:unit_error, "unit: #{w} is a reading on a scale with an offset, not a unit — inside a compound unit write #{if w in ["°F", "fahrenheit"], do: "degF", else: "degC"} (a difference of one degree)"})

    case Map.fetch(t, w) do
      {:ok, u} ->
        {:ok, u}

      :error ->
        Enum.find_value(@prefixes, {:error, "unit: unknown #{inspect(w)}"}, fn {p, f} ->
          with true <- String.starts_with?(w, p),
               base = String.replace_prefix(w, p, ""),
               true <- base != "" and base not in @no_prefix,
               {:ok, {g, dim}} <- Map.fetch(t, base) do
            # a prefixed gram is a fraction of the kilogram; anything else scales as said
            {:ok, {f * g, dim}}
          else
            _ -> nil
          end
        end)
    end
  end

  @doc "Product of two units."
  def mul({f, a}, {g, b}), do: {f * g, add(a, b)}
  @doc "A unit to a power."
  def pow({f, a}, e), do: {:math.pow(f, e), scale(a, e)}

  @doc "Sum of dimension vectors (the dimension of a product)."
  def add(a, b), do: zip(a, b, &(&1 + &2))
  @doc "Difference of dimension vectors (the dimension of a quotient)."
  def sub(a, b), do: zip(a, b, &(&1 - &2))
  @doc "A dimension vector scaled (the dimension of a power)."
  def scale(a, e), do: a |> Tuple.to_list() |> Enum.map(&r(&1 * e)) |> List.to_tuple()

  defp zip(a, b, f), do: Enum.zip_with(Tuple.to_list(a), Tuple.to_list(b), &r(f.(&1, &2))) |> List.to_tuple()
  defp r(x), do: Float.round(x * 1.0, 9) + 0.0

  @doc "Whether a dimension is dimensionless."
  def none?(dim), do: same?(dim, @zero)
  @doc "Whether two dimensions are equal."
  def same?(a, b), do: Enum.zip(Tuple.to_list(a), Tuple.to_list(b)) |> Enum.all?(fn {x, y} -> abs(x - y) < 1.0e-9 end)

  # named derived units, tried in this order when formatting a result
  @named [{"N", "N"}, {"Pa", "Pa"}, {"J", "J"}, {"W", "W"}, {"V", "V"}, {"C", "C"}, {"F", "F"}, {"ohm", "Ω"}, {"S", "S"},
          {"Wb", "Wb"}, {"T", "T"}, {"H", "H"}, {"Hz", "Hz"}]

  @doc """
  A dimension written in SI: a named unit when one matches exactly
  (`N`, `Pa`, `J`, `W`, `V`, `Ω`…), else the base units (`kg·m²/s³`),
  `""` when dimensionless.
  """
  def format(dim) do
    t = table()
    named = Enum.find_value(@named, fn {k, show} -> if same?(elem(t[k], 1), dim) and not none?(dim), do: show end)

    named || base_string(dim)
  end

  defp base_string(dim) do
    parts = Enum.zip(@base, Tuple.to_list(dim)) |> Enum.reject(fn {_, e} -> e == 0 end)
    {num, den} = Enum.split_with(parts, fn {_, e} -> e > 0 end)
    f = fn {u, e} -> if e == 1, do: u, else: u <> sup(e) end
    n = if num == [], do: (if den == [], do: "", else: "1"), else: Enum.map_join(num, "·", f)
    case den do
      [] -> n
      [one] -> n <> "/" <> f.({elem(one, 0), -elem(one, 1)})
      many -> n <> "/(" <> Enum.map_join(many, "·", fn {u, e} -> f.({u, -e}) end) <> ")"
    end
  end

  defp sup(e) do
    s = if e == trunc(e), do: Integer.to_string(trunc(e)), else: :erlang.float_to_binary(e * 1.0, [:compact, decimals: 4])
    if s =~ ~r/^-?\d+$/, do: String.replace(s, ["-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], &Map.fetch!(%{"-" => "⁻", "0" => "⁰", "1" => "¹", "2" => "²", "3" => "³", "4" => "⁴", "5" => "⁵", "6" => "⁶", "7" => "⁷", "8" => "⁸", "9" => "⁹"}, &1)), else: "^" <> s
  end

  @doc "The dimension's name in words, for error messages (`length`, `force`, `L·T⁻¹`…)."
  def describe(dim) do
    t = table()
    words = [{"m", "length"}, {"s", "time"}, {"g", "mass"}, {"N", "force"}, {"Pa", "pressure"}, {"J", "energy"}, {"W", "power"},
             {"V", "voltage"}, {"A", "current"}, {"K", "temperature"}, {"ohm", "resistance"}, {"Hz", "frequency"}, {"C", "charge"}]
    case Enum.find(words, fn {k, _} -> same?(elem(t[k], 1), dim) end) do
      {_, w} -> w
      nil -> if none?(dim), do: "a pure number", else: format(dim)
    end
  end

  @doc "A dimension for messages: `force (N)`, or just `kg/m³` when it has no name."
  def label(dim) do
    {w, f} = {describe(dim), format(dim)}
    if w == f or f == "", do: w, else: "#{w} (#{f})"
  end

  @doc """
  Convert an SI value of dimension `dim` to the unit `target`:
  `{:ok, value}` or `{:error, why}` when the dimensions differ.
  """
  def convert(si_value, dim, target) do
    case affine(target) do
      {:ok, {f, off}} -> if same?(dim, d(0, 0, 0, 0, 1)), do: {:ok, (si_value - off) / f}, else: {:error, "cannot express #{label(dim)} in #{target} (a temperature reading)"}
      :none -> convert_linear(si_value, dim, target)
    end
  end

  defp convert_linear(si_value, dim, target) do
    with {:ok, {f, tdim}} <- parse(target) do
      if same?(dim, tdim), do: {:ok, si_value / f}, else: {:error, "cannot express #{label(dim)} in #{target} (#{label(tdim)})"}
    end
  end
end
