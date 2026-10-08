defmodule Vapor.Alembic.Builtins do
  @moduledoc """
  Alembic's standard library (docs/ALEMBIC.md §1, §3): arithmetic on
  arbitrary-precision integers and floats, lists, tuples, maps, strings,
  bits, combinatorics and statistics — all pure, all charged fuel in
  proportion to the work they do, none able to touch the host.
  """
  alias Vapor.Alembic.Compiler, as: C
  import Bitwise

  # ============================================================== values

  @doc "A value's type name, as the error messages say it."
  def type(v) when is_integer(v), do: "an integer"
  def type(v) when is_float(v), do: "a float"
  def type(v) when is_boolean(v), do: "a boolean"
  def type(nil), do: "nil"
  def type(v) when is_binary(v), do: "a string"
  def type(v) when is_list(v), do: "a list"
  def type(v) when is_tuple(v) and tuple_size(v) == 4 and elem(v, 0) == :fn, do: "a function"
  def type(v) when is_tuple(v), do: "a tuple"
  def type(v) when is_map(v), do: "a map"
  def type(_), do: "a value"

  @doc "A value as Alembic source (`literal/1` reads it back)."
  def show(v) when is_integer(v), do: Integer.to_string(v)
  def show(v) when is_float(v), do: fshow(v)
  def show(true), do: "true"
  def show(false), do: "false"
  def show(nil), do: "nil"
  def show(v) when is_binary(v), do: "\"" <> (v |> String.replace("\\", "\\\\") |> String.replace("\"", "\\\"") |> String.replace("\n", "\\n") |> String.replace("\t", "\\t")) <> "\""
  def show(v) when is_list(v), do: "[" <> Enum.map_join(v, ", ", &show/1) <> "]"
  def show({:fn, name, _, _}), do: "<function #{name}>"
  def show({x}), do: "(" <> show(x) <> ",)"
  def show(v) when is_tuple(v), do: "(" <> (v |> Tuple.to_list() |> Enum.map_join(", ", &show/1)) <> ")"
  def show(v) when is_map(v), do: "{" <> (v |> Enum.sort() |> Enum.map_join(", ", fn {k, x} -> show(k) <> ": " <> show(x) end)) <> "}"

  @doc "A short rendering for error messages."
  def show_short(v) do
    s = show(v)
    if String.length(s) > 60, do: String.slice(s, 0, 57) <> "…", else: s
  end

  defp fshow(x) do
    s = :erlang.float_to_binary(x, [:short])
    if String.contains?(s, ".") or String.contains?(s, "e"), do: s, else: s <> ".0"
  end

  @doc "A value as JSON-ready data: tuples become lists, maps with non-string keys become pair lists."
  def to_data(v) when is_list(v), do: Enum.map(v, &to_data/1)
  def to_data({:fn, name, _, _}), do: "<function #{name}>"
  def to_data(v) when is_tuple(v), do: v |> Tuple.to_list() |> Enum.map(&to_data/1)
  def to_data(v) when is_map(v) do
    if Enum.all?(Map.keys(v), &is_binary/1), do: Map.new(v, fn {k, x} -> {k, to_data(x)} end), else: v |> Enum.sort() |> Enum.map(fn {k, x} -> [to_data(k), to_data(x)] end)
  end
  def to_data(v), do: v

  # ============================================================== operators

  @doc "The function for a binary operator."
  def binop("+"), do: &add/2
  def binop("-"), do: &sub/2
  def binop("*"), do: &mul/2
  def binop("/"), do: &fdiv/2
  def binop("//"), do: &idiv/2
  def binop("%"), do: &imod/2
  def binop("^"), do: &pow/2
  def binop("++"), do: &concat/2
  def binop("&"), do: &int2("&", &1, &2, fn a, b -> band(a, b) end)
  def binop("|"), do: &int2("|", &1, &2, fn a, b -> bor(a, b) end)
  def binop("xor"), do: &int2("xor", &1, &2, fn a, b -> bxor(a, b) end)
  def binop("<<"), do: &shl/2
  def binop(">>"), do: &int2(">>", &1, &2, fn a, b -> if b < 0, do: C.fail(">> by a negative amount"), else: a >>> b end)

  @doc "The function for a comparison operator."
  def cmpop("=="), do: &eq?/2
  def cmpop("!="), do: &(not eq?(&1, &2))
  def cmpop("<"), do: &(ord(&1, &2) == :lt)
  def cmpop("<="), do: &(ord(&1, &2) != :gt)
  def cmpop(">"), do: &(ord(&1, &2) == :gt)
  def cmpop(">="), do: &(ord(&1, &2) != :lt)
  def cmpop("in"), do: &member?(&2, &1)
  def cmpop("not in"), do: &(not member?(&2, &1))

  def eq?(a, b) when is_number(a) and is_number(b), do: a == b
  def eq?(a, b), do: a === b or (is_list(a) and is_list(b) and length(a) == length(b) and Enum.all?(Enum.zip(a, b), fn {x, y} -> eq?(x, y) end)) or
                       (is_tuple(a) and is_tuple(b) and tuple_size(a) == tuple_size(b) and eq?(Tuple.to_list(a), Tuple.to_list(b)))

  defp ord(a, b) when is_number(a) and is_number(b), do: (cond do a < b -> :lt; a > b -> :gt; true -> :eq end)
  defp ord(a, b) when is_binary(a) and is_binary(b), do: (cond do a < b -> :lt; a > b -> :gt; true -> :eq end)
  defp ord(a, b) when is_list(a) and is_list(b), do: lex_ord(a, b)
  defp ord(a, b) when is_tuple(a) and is_tuple(b), do: lex_ord(Tuple.to_list(a), Tuple.to_list(b))
  defp ord(a, b), do: C.fail("cannot order #{type(a)} against #{type(b)}")

  defp lex_ord([], []), do: :eq
  defp lex_ord([], _), do: :lt
  defp lex_ord(_, []), do: :gt
  defp lex_ord([x | xs], [y | ys]), do: (case ord(x, y) do :eq -> lex_ord(xs, ys); o -> o end)

  @doc "A total order on values for sorting (numbers by value, then by type)."
  def compare(a, b) do
    case {rank(a), rank(b)} do
      {r, r} when r in [1, 2, 3] -> ord(a, b)
      {r, r} when r == 4 -> lex_ord_total(Tuple.to_list(a), Tuple.to_list(b))
      {r, r} when r == 5 -> lex_ord_total(a, b)
      {r, r} -> (cond do a < b -> :lt; a > b -> :gt; true -> :eq end)
      {r1, r2} -> if r1 < r2, do: :lt, else: :gt
    end
  end

  defp rank(v) when is_number(v), do: 1
  defp rank(v) when is_binary(v), do: 2
  defp rank(v) when is_boolean(v), do: 3
  defp rank(v) when is_tuple(v), do: 4
  defp rank(v) when is_list(v), do: 5
  defp rank(_), do: 6

  defp lex_ord_total([], []), do: :eq
  defp lex_ord_total([], _), do: :lt
  defp lex_ord_total(_, []), do: :gt
  defp lex_ord_total([x | xs], [y | ys]), do: (case compare(x, y) do :eq -> lex_ord_total(xs, ys); o -> o end)

  defp sort_values(xs), do: Enum.sort(xs, fn a, b -> compare(a, b) != :gt end)

  defp add(a, b) when is_integer(a) and is_integer(b), do: cap(a + b)
  defp add(a, b) when is_number(a) and is_number(b), do: fl(a + b)
  defp add(a, b), do: C.fail("+ needs numbers, got #{type(a)} and #{type(b)} (use ++ to join lists or strings)")
  defp sub(a, b) when is_integer(a) and is_integer(b), do: cap(a - b)
  defp sub(a, b) when is_number(a) and is_number(b), do: fl(a - b)
  defp sub(a, b), do: C.fail("- needs numbers, got #{type(a)} and #{type(b)}")

  defp mul(a, b) when is_integer(a) and is_integer(b) do
    if bits(a) + bits(b) > C.max_bits(), do: C.fail("integer larger than #{C.max_bits()} bits")
    a * b
  end

  defp mul(a, b) when is_number(a) and is_number(b), do: fl(a * b)
  defp mul(a, b) when is_list(a) and is_integer(b) and b >= 0, do: repeat(a, b)
  defp mul(a, b), do: C.fail("* needs numbers, got #{type(a)} and #{type(b)}")

  defp fdiv(_, b) when b == 0, do: C.fail("division by zero")
  defp fdiv(a, b) when is_number(a) and is_number(b), do: fl(a / b)
  defp fdiv(a, b), do: C.fail("/ needs numbers, got #{type(a)} and #{type(b)}")

  defp idiv(_, b) when b == 0, do: C.fail("division by zero")
  defp idiv(a, b) when is_integer(a) and is_integer(b), do: Integer.floor_div(a, b)
  defp idiv(a, b) when is_number(a) and is_number(b), do: fl(Float.floor(a / b))
  defp idiv(a, b), do: C.fail("// needs numbers, got #{type(a)} and #{type(b)}")

  defp imod(_, b) when b == 0, do: C.fail("modulo by zero")
  defp imod(a, b) when is_integer(a) and is_integer(b), do: Integer.mod(a, b)
  defp imod(a, b) when is_number(a) and is_number(b), do: fl(a - b * Float.floor(a / b))
  defp imod(a, b), do: C.fail("% needs numbers, got #{type(a)} and #{type(b)}")

  defp pow(a, b) when is_integer(a) and is_integer(b) and b >= 0 do
    cond do
      a in [0, 1] -> a
      a == -1 -> if rem(b, 2) == 0, do: 1, else: -1
      bits(a) * b > C.max_bits() -> C.fail("integer larger than #{C.max_bits()} bits")
      true -> Integer.pow(a, b)
    end
  end

  defp pow(a, b) when is_number(a) and is_number(b) do
    cond do
      a == 0 and b < 0 -> C.fail("0 to a negative power")
      a < 0 and b != Float.round(b * 1.0) -> C.fail("a negative number to a fractional power")
      true -> fl(:math.pow(a * 1.0, b * 1.0))
    end
  rescue
    ArithmeticError -> C.fail("power out of range")
  end

  defp pow(a, b), do: C.fail("^ needs numbers, got #{type(a)} and #{type(b)}")

  defp concat(a, b) when is_list(a) and is_list(b) do
    C.tick(div(length(a), 64))
    r = a ++ b
    if length(r) > C.max_list(), do: C.fail("a list longer than #{C.max_list()} elements")
    r
  end

  defp concat(a, b) when is_binary(a) and is_binary(b), do: a <> b
  defp concat(a, b) when is_binary(a), do: a <> show_text(b)
  defp concat(a, b) when is_binary(b), do: show_text(a) <> b
  defp concat(a, b), do: C.fail("++ joins two lists or two strings, got #{type(a)} and #{type(b)}")

  defp show_text(v) when is_binary(v), do: v
  defp show_text(v), do: show(v)

  defp shl(a, b) when is_integer(a) and is_integer(b) and b >= 0 do
    if bits(a) + b > C.max_bits(), do: C.fail("integer larger than #{C.max_bits()} bits")
    a <<< b
  end

  defp shl(a, b), do: int2("<<", a, b, fn _, _ -> C.fail("<< by a negative amount") end)

  defp int2(_op, a, b, f) when is_integer(a) and is_integer(b), do: f.(a, b)
  defp int2(op, a, b, _), do: C.fail("#{op} needs integers, got #{type(a)} and #{type(b)}")

  defp bits(0), do: 1
  defp bits(n), do: n |> abs() |> :binary.encode_unsigned() |> byte_size() |> Kernel.*(8)

  defp cap(n) do
    if bits(n) > C.max_bits(), do: C.fail("integer larger than #{C.max_bits()} bits")
    n
  end

  defp fl(x) when is_float(x), do: x
  defp fl(x), do: x * 1.0

  # ============================================================== access

  def index(xs, i) when is_list(xs) and is_integer(i) do
    n = length(xs)
    j = if i < 0, do: n + i, else: i
    if j < 0 or j >= n, do: C.fail("index #{i} outside a list of #{n}")
    C.tick(div(j, 256))
    Enum.at(xs, j)
  end

  def index(t, i) when is_tuple(t) and is_integer(i) do
    n = tuple_size(t)
    j = if i < 0, do: n + i, else: i
    if j < 0 or j >= n, do: C.fail("index #{i} outside a tuple of #{n}")
    elem(t, j)
  end

  def index(s, i) when is_binary(s) and is_integer(i) do
    case String.at(s, i) do
      nil -> C.fail("index #{i} outside a string of #{String.length(s)}")
      g -> g
    end
  end

  def index(m, k) when is_map(m) do
    case Map.fetch(m, k) do
      {:ok, v} -> v
      :error -> C.fail("key #{show_short(k)} not in the map")
    end
  end

  def index(xs, i), do: C.fail("cannot index #{type(xs)} with #{type(i)}")

  def slice(xs, i, j) when is_list(xs) do
    n = length(xs)
    {a, b} = bounds(n, i, j)
    C.tick(div(b - a, 64))
    Enum.slice(xs, a, max(b - a, 0))
  end

  def slice(s, i, j) when is_binary(s) do
    n = String.length(s)
    {a, b} = bounds(n, i, j)
    String.slice(s, a, max(b - a, 0))
  end

  def slice(t, i, j) when is_tuple(t), do: t |> Tuple.to_list() |> slice(i, j) |> List.to_tuple()
  def slice(v, _, _), do: C.fail("cannot slice #{type(v)}")

  defp bounds(n, i, j) do
    norm = fn nil, d -> d; k, _ when is_integer(k) and k < 0 -> max(n + k, 0); k, _ when is_integer(k) -> min(k, n); k, _ -> C.fail("slice bound #{show_short(k)} is not an integer") end
    {norm.(i, 0), norm.(j, n)}
  end

  def field(m, f) when is_map(m) do
    case Map.fetch(m, f) do
      {:ok, v} -> v
      :error -> C.fail("no field #{f} (keys: #{m |> Map.keys() |> Enum.map_join(", ", &show_short/1)})")
    end
  end

  def field(v, f), do: C.fail("#{type(v)} has no field #{f}")

  def member?(xs, x) when is_list(xs), do: (C.tick(div(length(xs), 64)); Enum.any?(xs, &eq?(&1, x)))
  def member?(t, x) when is_tuple(t), do: member?(Tuple.to_list(t), x)
  def member?(m, k) when is_map(m), do: Map.has_key?(m, k)
  def member?(s, x) when is_binary(s) and is_binary(x), do: String.contains?(s, x)
  def member?(v, _), do: C.fail("`in` needs a list, tuple, map or string, got #{type(v)}")

  @doc "A value as a list to iterate over."
  def iterable(xs) when is_list(xs), do: xs
  def iterable(t) when is_tuple(t), do: Tuple.to_list(t)
  def iterable(m) when is_map(m), do: m |> Enum.sort() |> Enum.map(fn {k, v} -> {k, v} end)
  def iterable(s) when is_binary(s), do: String.graphemes(s)
  def iterable(n) when is_integer(n) and n >= 0, do: range(0, n, false)
  def iterable(v), do: C.fail("cannot iterate over #{type(v)}")

  @doc "The integers from a to b (inclusive or not)."
  def range(a, b, inclusive) when is_integer(a) and is_integer(b) do
    hi = if inclusive, do: b, else: b - 1
    n = hi - a + 1
    cond do
      n <= 0 -> []
      n > C.max_list() -> C.fail("a range of #{n} elements (the limit is #{C.max_list()})")
      true -> C.tick(div(n, 64)); Enum.to_list(a..hi)
    end
  end

  def range(a, b, _), do: C.fail("a range needs integers, got #{type(a)} and #{type(b)}")

  defp repeat(xs, n) do
    if length(xs) * n > C.max_list(), do: C.fail("a list longer than #{C.max_list()} elements")
    C.tick(div(length(xs) * n, 64))
    xs |> List.duplicate(n) |> Enum.concat()
  end

  # ============================================================== library

  defp call(f, args), do: C.apply_fn(f, args, "function argument", {0, 0})

  defp num!(x, _name) when is_number(x), do: x * 1.0
  defp num!(x, name), do: C.fail("#{name} needs a number, got #{type(x)}")

  defp list!(xs, _name) when is_list(xs), do: (C.tick(div(length(xs), 64)); xs)
  defp list!(t, name) when is_tuple(t) and elem(t, 0) != :fn, do: list!(Tuple.to_list(t), name)
  defp list!(s, _name) when is_binary(s), do: String.graphemes(s)
  defp list!(m, _name) when is_map(m), do: iterable(m)
  defp list!(n, _name) when is_integer(n) and n >= 0, do: range(0, n, false)
  defp list!(v, name), do: C.fail("#{name} needs a list, got #{type(v)}")

  defp int!(x, _name) when is_integer(x), do: x
  defp int!(x, name) when is_float(x) and x == trunc(x) * 1.0, do: (_ = name; trunc(x))
  defp int!(x, name), do: C.fail("#{name} needs an integer, got #{type(x)}")

  defp math1(name, f), do: {name, 1, fn [x] -> mathcall(name, f, [num!(x, name)]) end}

  defp mathcall(name, f, args) do
    apply(f, args)
  rescue
    ArithmeticError -> C.fail("#{name}(#{Enum.map_join(args, ", ", &show/1)}) is undefined")
  end

  defp agg(xs, name) do
    case list!(xs, name) do
      [] -> C.fail("#{name} of an empty list")
      l -> l
    end
  end

  defp nums(xs, name), do: xs |> agg(name) |> Enum.map(&num!(&1, name))

  defp minmax(name, args, pick) do
    l = case args do [xs] -> agg(xs, name); xs -> xs end
    Enum.reduce(l, fn x, acc -> if pick.(compare(x, acc)), do: x, else: acc end)
  end

  defp by(name, xs, f, pick) do
    l = agg(xs, name)
    {best, _} = l |> Enum.map(fn x -> {x, call(f, [x])} end) |> Enum.reduce(fn {x, k}, {bx, bk} -> if pick.(compare(k, bk)), do: {x, k}, else: {bx, bk} end)
    best
  end

  defp argby(name, xs, pick) do
    l = agg(xs, name)
    {_, i} = l |> Enum.with_index() |> Enum.reduce(fn {x, i}, {bx, bi} -> if pick.(compare(x, bx)), do: {x, i}, else: {bx, bi} end)
    i
  end

  defp combinations(_, 0), do: [[]]
  defp combinations([], _), do: []
  defp combinations([x | xs], k), do: Enum.map(combinations(xs, k - 1), &[x | &1]) ++ combinations(xs, k)

  defp permutations([]), do: [[]]
  defp permutations(xs), do: for(x <- xs, rest <- permutations(xs -- [x]), do: [x | rest])

  defp binom(n, k) when k < 0 or k > n, do: 0
  defp binom(n, k), do: Enum.reduce(1..min(k, n - k)//1, 1, fn i, acc -> div(acc * (n - min(k, n - k) + i), i) end)

  defp prime?(n) when n < 2, do: false
  defp prime?(n) when n < 4, do: true
  defp prime?(n) when rem(n, 2) == 0, do: false
  defp prime?(n) when n < 1_000_000_000_000 do
    lim = isqrt(n)
    C.tick(div(lim, 8))
    Enum.all?(Stream.iterate(3, &(&1 + 2)) |> Enum.take_while(&(&1 <= lim)), &(rem(n, &1) != 0))
  end
  defp prime?(n), do: miller_rabin(n)

  defp miller_rabin(n) do
    {d, s} = split2(n - 1, 0)
    Enum.all?([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37], fn a ->
      x = :crypto.mod_pow(a, d, n) |> :binary.decode_unsigned()
      x == 1 or x == n - 1 or Enum.any?(1..(s - 1)//1, fn r -> :crypto.mod_pow(a, d * Integer.pow(2, r), n) |> :binary.decode_unsigned() == n - 1 end)
    end)
  end

  defp split2(d, s) when rem(d, 2) == 0, do: split2(div(d, 2), s + 1)
  defp split2(d, s), do: {d, s}

  @doc "Integer square root (floor)."
  def isqrt(n) when n < 0, do: C.fail("isqrt of a negative number")
  def isqrt(n) when n < 2, do: n
  def isqrt(n), do: isqrt_iter(n, Integer.pow(2, div(bits(n), 2) + 1))
  defp isqrt_iter(n, x) do
    y = div(x + div(n, x), 2)
    if y >= x, do: x, else: isqrt_iter(n, y)
  end

  defp mean(xs), do: Enum.sum(xs) / length(xs)

  defp variance(xs) do
    n = length(xs)
    if n < 2, do: C.fail("var needs at least 2 values")
    m = mean(xs)
    Enum.reduce(xs, 0.0, fn x, s -> s + (x - m) * (x - m) end) / (n - 1)
  end

  defp median(xs) do
    s = Enum.sort(xs)
    n = length(s)
    if rem(n, 2) == 1, do: Enum.at(s, div(n, 2)), else: (Enum.at(s, div(n, 2) - 1) + Enum.at(s, div(n, 2))) / 2
  end

  @doc "A deterministic float in [0, 1) from any values (for noise, jitter and tie-breaking)."
  def hash01(args) do
    <<x::unsigned-53, _::bits>> = :crypto.hash(:sha256, :erlang.term_to_binary(Enum.map(args, &to_data/1)))
    x / 9_007_199_254_740_992
  end

  defp set_at(xs, i, v) do
    n = length(xs)
    j = if i < 0, do: n + i, else: i
    if j < 0 or j >= n, do: C.fail("index #{i} outside a list of #{n}")
    List.replace_at(xs, j, v)
  end

  defp lst(xs, name), do: list!(xs, name)

  defp table do
    [
      math1("sqrt", &:math.sqrt/1), math1("exp", &:math.exp/1), math1("ln", &:math.log/1), math1("log2", &:math.log2/1),
      math1("log10", &:math.log10/1), math1("sin", &:math.sin/1), math1("cos", &:math.cos/1), math1("tan", &:math.tan/1),
      math1("asin", &:math.asin/1), math1("acos", &:math.acos/1), math1("atan", &:math.atan/1), math1("sinh", &:math.sinh/1),
      math1("cosh", &:math.cosh/1), math1("tanh", &:math.tanh/1), math1("erf", &:math.erf/1), math1("erfc", &:math.erfc/1),
      {"log", {1, 2}, fn [x] -> mathcall("log", &:math.log/1, [num!(x, "log")]); [x, b] -> mathcall("log", &(:math.log(&1) / :math.log(&2)), [num!(x, "log"), num!(b, "log")]) end},
      {"atan2", 2, fn [y, x] -> mathcall("atan2", &:math.atan2/2, [num!(y, "atan2"), num!(x, "atan2")]) end},
      {"hypot", 2, fn [x, y] -> :math.sqrt(num!(x, "hypot") ** 2 + num!(y, "hypot") ** 2) end},
      {"pow", 2, fn [a, b] -> pow(a, b) end},
      {"abs", 1, fn [x] when is_number(x) -> abs(x); [x] -> C.fail("abs needs a number, got #{type(x)}") end},
      {"sign", 1, fn [x] when is_number(x) -> (cond do x > 0 -> 1; x < 0 -> -1; true -> 0 end); [x] -> C.fail("sign needs a number, got #{type(x)}") end},
      {"floor", 1, fn [x] when is_integer(x) -> x; [x] -> trunc(Float.floor(num!(x, "floor"))) end},
      {"ceil", 1, fn [x] when is_integer(x) -> x; [x] -> trunc(Float.ceil(num!(x, "ceil"))) end},
      {"round", {1, 2}, fn [x] when is_integer(x) -> x; [x] -> round(num!(x, "round")); [x, d] -> Float.round(num!(x, "round"), int!(d, "round")) end},
      {"trunc", 1, fn [x] when is_integer(x) -> x; [x] -> trunc(num!(x, "trunc")) end},
      {"int", 1, fn [x] when is_integer(x) -> x; [x] when is_float(x) -> trunc(x); [true] -> 1; [false] -> 0
                    [s] when is_binary(s) -> (case Integer.parse(String.trim(s)) do {n, ""} -> n; _ -> C.fail("int(#{show_short(s)}): not an integer") end)
                    [x] -> C.fail("int of #{type(x)}") end},
      {"float", 1, fn [x] when is_number(x) -> x * 1.0
                      [s] when is_binary(s) -> (case Float.parse(String.trim(s)) do {f, ""} -> f; _ -> C.fail("float(#{show_short(s)}): not a number") end)
                      [x] -> C.fail("float of #{type(x)}") end},
      {"min", :any, fn [] -> C.fail("min of nothing"); args -> minmax("min", args, &(&1 == :lt)) end},
      {"max", :any, fn [] -> C.fail("max of nothing"); args -> minmax("max", args, &(&1 == :gt)) end},
      {"clamp", 3, fn [x, lo, hi] -> x |> max(lo) |> min(hi) end},
      {"gcd", 2, fn [a, b] -> Integer.gcd(int!(a, "gcd"), int!(b, "gcd")) end},
      {"lcm", 2, fn [a, b] -> (a = int!(a, "lcm"); b = int!(b, "lcm"); if a == 0 or b == 0, do: 0, else: cap(div(abs(a * b), Integer.gcd(a, b)))) end},
      {"isqrt", 1, fn [n] -> isqrt(int!(n, "isqrt")) end},
      {"is_prime", 1, fn [n] -> prime?(int!(n, "is_prime")) end},
      {"factorial", 1, fn [n] -> (n = int!(n, "factorial"); if n < 0 or n > 2000, do: C.fail("factorial defined here for 0..2000"), else: (C.tick(div(n, 16)); Enum.reduce(1..max(n, 1)//1, 1, &*/2))) end},
      {"binom", 2, fn [n, k] -> (n = int!(n, "binom"); if n > 100_000, do: C.fail("binom: n too large"), else: (C.tick(div(n, 64)); binom(n, int!(k, "binom")))) end},
      {"len", 1, fn [xs] when is_list(xs) -> length(xs); [s] when is_binary(s) -> String.length(s); [t] when is_tuple(t) -> tuple_size(t); [m] when is_map(m) -> map_size(m); [v] -> C.fail("len of #{type(v)}") end},
      {"sum", {1, 2}, fn [xs] -> Enum.reduce(lst(xs, "sum"), 0, &add/2); [xs, f] -> Enum.reduce(lst(xs, "sum"), 0, &add(call(f, [&1]), &2)) end},
      {"prod", 1, fn [xs] -> Enum.reduce(lst(xs, "prod"), 1, &mul/2) end},
      {"mean", 1, fn [xs] -> mean(nums(xs, "mean")) end},
      {"var", 1, fn [xs] -> variance(nums(xs, "var")) end},
      {"std", 1, fn [xs] -> :math.sqrt(variance(nums(xs, "std"))) end},
      {"median", 1, fn [xs] -> median(nums(xs, "median")) end},
      {"sort", {1, 2}, fn [xs] -> (l = lst(xs, "sort"); C.tick(length(l)); sort_values(l))
                         [xs, f] -> (l = lst(xs, "sort"); C.tick(length(l)); l |> Enum.map(&{call(f, [&1]), &1}) |> Enum.sort(fn {a, _}, {b, _} -> compare(a, b) != :gt end) |> Enum.map(&elem(&1, 1))) end},
      {"reverse", 1, fn [xs] when is_binary(xs) -> String.reverse(xs); [xs] -> Enum.reverse(lst(xs, "reverse")) end},
      {"range", {1, 3}, fn [b] -> range(0, b, false); [a, b] -> range(a, b, false)
                          [a, b, s] -> (s = int!(s, "range"); a = int!(a, "range"); b = int!(b, "range")
                                        if s == 0, do: C.fail("range with step 0")
                                        n = max(div(b - a + (if s > 0, do: s - 1, else: s + 1), s), 0)
                                        if n > C.max_list(), do: C.fail("range too long"), else: (C.tick(div(n, 64)); for i <- 0..(n - 1)//1, do: a + i * s)) end},
      {"map", 2, fn [xs, f] -> Enum.map(lst(xs, "map"), &call(f, [&1])) end},
      {"filter", 2, fn [xs, f] -> Enum.filter(lst(xs, "filter"), &C.truthy(call(f, [&1]))) end},
      {"fold", 3, fn [xs, init, f] -> Enum.reduce(lst(xs, "fold"), init, &call(f, [&2, &1])) end},
      {"scan", 3, fn [xs, init, f] -> lst(xs, "scan") |> Enum.scan(init, &call(f, [&2, &1])) end},
      {"all", {1, 2}, fn [xs] -> Enum.all?(lst(xs, "all"), &C.truthy/1); [xs, f] -> Enum.all?(lst(xs, "all"), &C.truthy(call(f, [&1]))) end},
      {"any", {1, 2}, fn [xs] -> Enum.any?(lst(xs, "any"), &C.truthy/1); [xs, f] -> Enum.any?(lst(xs, "any"), &C.truthy(call(f, [&1]))) end},
      {"count", {1, 2}, fn [xs] -> Enum.count(lst(xs, "count"), &C.truthy/1)
                          [xs, {:fn, _, _, _} = f] -> Enum.count(lst(xs, "count"), &C.truthy(call(f, [&1])))
                          [xs, v] -> Enum.count(lst(xs, "count"), &eq?(&1, v)) end},
      {"zip", :any, fn [] -> []; ls -> ls |> Enum.map(&lst(&1, "zip")) |> Enum.zip() end},
      {"enumerate", 1, fn [xs] -> lst(xs, "enumerate") |> Enum.with_index() |> Enum.map(fn {x, i} -> {i, x} end) end},
      {"distinct", 1, fn [xs] -> (l = lst(xs, "distinct"); C.tick(length(l)); Enum.uniq_by(l, &norm_key/1)) end},
      {"is_distinct", 1, fn [xs] -> (l = lst(xs, "is_distinct"); C.tick(length(l)); length(Enum.uniq_by(l, &norm_key/1)) == length(l)) end},
      {"set", 1, fn [xs] -> (l = lst(xs, "set"); C.tick(length(l)); l |> Enum.uniq_by(&norm_key/1) |> sort_values()) end},
      {"union", 2, fn [a, b] -> (lst(a, "union") ++ lst(b, "union")) |> Enum.uniq_by(&norm_key/1) |> sort_values() end},
      {"intersect", 2, fn [a, b] -> (kb = MapSet.new(lst(b, "intersect"), &norm_key/1); lst(a, "intersect") |> Enum.filter(&MapSet.member?(kb, norm_key(&1))) |> Enum.uniq_by(&norm_key/1) |> sort_values()) end},
      {"difference", 2, fn [a, b] -> (kb = MapSet.new(lst(b, "difference"), &norm_key/1); lst(a, "difference") |> Enum.reject(&MapSet.member?(kb, norm_key(&1))) |> Enum.uniq_by(&norm_key/1) |> sort_values()) end},
      {"flatten", 1, fn [xs] -> Enum.flat_map(lst(xs, "flatten"), fn l when is_list(l) -> l; x -> [x] end) end},
      {"concat", :any, fn ls -> Enum.reduce(ls, [], &concat(&2, &1)) end},
      {"take", 2, fn [xs, n] -> Enum.take(lst(xs, "take"), max(int!(n, "take"), 0)) end},
      {"drop", 2, fn [xs, n] -> Enum.drop(lst(xs, "drop"), max(int!(n, "drop"), 0)) end},
      {"first", 1, fn [xs] -> (case lst(xs, "first") do [] -> C.fail("first of an empty list"); [x | _] -> x end) end},
      {"last", 1, fn [xs] -> (case lst(xs, "last") do [] -> C.fail("last of an empty list"); l -> List.last(l) end) end},
      {"append", 2, fn [xs, x] -> lst(xs, "append") ++ [x] end},
      {"prepend", 2, fn [xs, x] -> [x | lst(xs, "prepend")] end},
      {"set_at", 3, fn [xs, i, v] -> set_at(lst(xs, "set_at"), int!(i, "set_at"), v) end},
      {"insert", 3, fn [xs, i, v] -> List.insert_at(lst(xs, "insert"), int!(i, "insert"), v) end},
      {"remove_at", 2, fn [xs, i] -> List.delete_at(lst(xs, "remove_at"), int!(i, "remove_at")) end},
      {"swap", 3, fn [xs, i, j] -> (l = lst(xs, "swap"); a = index(l, i); b = index(l, j); l |> set_at(i, b) |> set_at(j, a)) end},
      {"index_of", 2, fn [xs, v] -> Enum.find_index(lst(xs, "index_of"), &eq?(&1, v)) end},
      {"find", 2, fn [xs, f] -> Enum.find(lst(xs, "find"), &C.truthy(call(f, [&1]))) end},
      {"repeat", 2, fn [x, n] -> (n = int!(n, "repeat"); if n > C.max_list(), do: C.fail("repeat too long"), else: (C.tick(div(n, 64)); List.duplicate(x, max(n, 0)))) end},
      {"argmin", 1, fn [xs] -> argby("argmin", xs, &(&1 == :lt)) end},
      {"argmax", 1, fn [xs] -> argby("argmax", xs, &(&1 == :gt)) end},
      {"min_by", 2, fn [xs, f] -> by("min_by", xs, f, &(&1 == :lt)) end},
      {"max_by", 2, fn [xs, f] -> by("max_by", xs, f, &(&1 == :gt)) end},
      {"chunks", 2, fn [xs, n] -> Enum.chunk_every(lst(xs, "chunks"), max(int!(n, "chunks"), 1)) end},
      {"windows", 2, fn [xs, n] -> Enum.chunk_every(lst(xs, "windows"), max(int!(n, "windows"), 1), 1, :discard) end},
      {"pairs", 1, fn [xs] -> (l = lst(xs, "pairs"); n = length(l); if n * n > 2 * C.max_list(), do: C.fail("pairs: list too long"), else: (C.tick(div(n * n, 128)); for({a, i} <- Enum.with_index(l), {b, j} <- Enum.with_index(l), i < j, do: {a, b}))) end},
      {"combinations", 2, fn [xs, k] -> (l = lst(xs, "combinations"); k = int!(k, "combinations"); c = binom(length(l), k)
                                          if c > 200_000, do: C.fail("combinations: #{c} is too many"), else: (C.tick(c); combinations(l, k))) end},
      {"permutations", 1, fn [xs] -> (l = lst(xs, "permutations"); if length(l) > 8, do: C.fail("permutations of more than 8 elements"), else: (C.tick(Enum.reduce(1..max(length(l), 1), 1, &*/2)); permutations(l))) end},
      {"product", :any, fn ls -> (ls = Enum.map(ls, &lst(&1, "product")); total = Enum.reduce(ls, 1, &(length(&1) * &2))
                                   if total > 200_000, do: C.fail("product: #{total} tuples is too many"), else: (C.tick(total); Enum.reduce(Enum.reverse(ls), [[]], fn l, acc -> for x <- l, rest <- acc, do: [x | rest] end) |> Enum.map(&List.to_tuple/1))) end},
      {"transpose", 1, fn [rows] -> (rs = Enum.map(lst(rows, "transpose"), &lst(&1, "transpose")); if rs == [], do: [], else: rs |> Enum.zip() |> Enum.map(&Tuple.to_list/1)) end},
      {"dot", 2, fn [a, b] -> (a = lst(a, "dot"); b = lst(b, "dot"); if length(a) != length(b), do: C.fail("dot of lists of different lengths"), else: Enum.zip(a, b) |> Enum.reduce(0, fn {x, y}, s -> add(s, mul(x, y)) end)) end},
      {"norm", 1, fn [a] -> :math.sqrt(Enum.reduce(nums(a, "norm"), 0.0, &(&1 * &1 + &2))) end},
      {"cumsum", 1, fn [xs] -> Enum.scan(lst(xs, "cumsum"), &add/2) end},
      {"diffs", 1, fn [xs] -> lst(xs, "diffs") |> Enum.chunk_every(2, 1, :discard) |> Enum.map(fn [a, b] -> sub(b, a) end) end},
      {"popcount", 1, fn [n] -> (n = int!(n, "popcount"); if n < 0, do: C.fail("popcount of a negative number"), else: (for <<b::1 <- :binary.encode_unsigned(n)>>, reduce: 0, do: (acc -> acc + b))) end},
      {"bit", 2, fn [n, i] -> (int!(n, "bit") >>> int!(i, "bit")) &&& 1 end},
      {"bits", 2, fn [n, w] -> (n = int!(n, "bits"); w = int!(w, "bits"); if w > 4096, do: C.fail("bits: width over 4096"), else: for(i <- 0..(w - 1)//1, do: (n >>> i) &&& 1)) end},
      {"from_bits", 1, fn [bs] -> lst(bs, "from_bits") |> Enum.with_index() |> Enum.reduce(0, fn {b, i}, acc -> if C.truthy(b), do: acc ||| (1 <<< i), else: acc end) end},
      {"band", 2, fn [a, b] -> int!(a, "band") &&& int!(b, "band") end},
      {"bor", 2, fn [a, b] -> int!(a, "bor") ||| int!(b, "bor") end},
      {"bxor", 2, fn [a, b] -> bxor(int!(a, "bxor"), int!(b, "bxor")) end},
      {"str", 1, fn [v] -> show_text(v) end},
      {"chars", 1, fn [s] when is_binary(s) -> String.graphemes(s); [v] -> C.fail("chars of #{type(v)}") end},
      {"join", {1, 2}, fn [xs] -> lst(xs, "join") |> Enum.map_join("", &show_text/1); [xs, sep] when is_binary(sep) -> lst(xs, "join") |> Enum.map_join(sep, &show_text/1) end},
      {"split", {1, 2}, fn [s] when is_binary(s) -> String.split(s); [s, sep] when is_binary(s) and is_binary(sep) and sep != "" -> String.split(s, sep); _ -> C.fail("split needs strings") end},
      {"upper", 1, fn [s] when is_binary(s) -> String.upcase(s) end},
      {"lower", 1, fn [s] when is_binary(s) -> String.downcase(s) end},
      {"keys", 1, fn [m] when is_map(m) -> m |> Map.keys() |> sort_values(); [v] -> C.fail("keys of #{type(v)}") end},
      {"values", 1, fn [m] when is_map(m) -> m |> Enum.sort() |> Enum.map(&elem(&1, 1)); [v] -> C.fail("values of #{type(v)}") end},
      {"items", 1, fn [m] when is_map(m) -> iterable(m); [v] -> C.fail("items of #{type(v)}") end},
      {"get", {2, 3}, fn [m, k] when is_map(m) -> Map.get(m, k); [m, k, d] when is_map(m) -> Map.get(m, k, d)
                        [xs, i] when is_list(xs) -> Enum.at(xs, int!(i, "get")); [xs, i, d] when is_list(xs) -> Enum.at(xs, int!(i, "get"), d)
                        [v | _] -> C.fail("get from #{type(v)}") end},
      {"put", 3, fn [m, k, v] when is_map(m) -> Map.put(m, k, v); [xs, i, v] when is_list(xs) -> set_at(xs, int!(i, "put"), v); [v, _, _] -> C.fail("put into #{type(v)}") end},
      {"has", 2, fn [m, k] when is_map(m) -> Map.has_key?(m, k); [c, x] -> member?(c, x) end},
      {"dict", 1, fn [pairs] -> lst(pairs, "dict") |> Map.new(fn {k, v} -> {k, v}; [k, v] -> {k, v}; x -> C.fail("dict needs (key, value) pairs, got #{show_short(x)}") end) end},
      {"tuple", :any, fn [xs] when is_list(xs) -> List.to_tuple(xs); args -> List.to_tuple(args) end},
      {"list", 1, fn [v] -> list!(v, "list") end},
      {"type", 1, fn [v] -> type(v) |> String.replace(~r/^an? /, "") end},
      {"hash", :any, fn args -> :erlang.phash2(Enum.map(args, &to_data/1), 4_294_967_296) end},
      {"noise", :any, fn args -> hash01(args) end},
      {"error", 1, fn [msg] -> C.fail(show_text(msg)) end},
      {"assert", {1, 2}, fn [c] -> (if C.truthy(c), do: true, else: C.fail("assertion failed")); [c, m] -> if C.truthy(c), do: true, else: C.fail("assertion failed: " <> show_text(m)) end},
      {"inversions", 1, fn [xs] -> (l = lst(xs, "inversions"); n = length(l); C.tick(div(n * n, 64)); t = List.to_tuple(l)
                                      Enum.reduce(0..(n - 1)//1, 0, fn i, acc -> acc + Enum.count((i + 1)..(n - 1)//1, &(compare(elem(t, i), elem(t, &1)) == :gt)) end)) end},
      {"is_sorted", 1, fn [xs] -> lst(xs, "is_sorted") |> Enum.chunk_every(2, 1, :discard) |> Enum.all?(fn [a, b] -> compare(a, b) != :gt end) end}
    ]
  end

  defp norm_key(v) when is_float(v) and v == trunc(v) * 1.0 and abs(v) < 9.0e15, do: trunc(v)
  defp norm_key(v) when is_list(v), do: Enum.map(v, &norm_key/1)
  defp norm_key(v) when is_tuple(v), do: v |> Tuple.to_list() |> Enum.map(&norm_key/1) |> List.to_tuple()
  defp norm_key(v), do: v

  @constants %{"pi" => :math.pi(), "π" => :math.pi(), "e" => :math.exp(1.0), "tau" => 2 * :math.pi()}

  @doc "Every builtin name."
  def names, do: Map.keys(lib()) ++ Map.keys(@constants)

  @doc "A builtin by name, or nil."
  def get(n) do
    case Map.fetch(@constants, n) do
      {:ok, v} -> v
      :error -> Map.get(lib(), n)
    end
  end

  defp lib do
    case :persistent_term.get({__MODULE__, :lib}, nil) do
      nil ->
        m = Map.new(table(), fn {name, arity, f} -> {name, {:fn, name, arity, guard(name, f)}} end)
        :persistent_term.put({__MODULE__, :lib}, m)
        m
      m -> m
    end
  end

  # a builtin given arguments it has no clause for says so instead of crashing
  defp guard(name, f) do
    fn args ->
      try do
        f.(args)
      rescue
        e in [FunctionClauseError, CaseClauseError, MatchError, ArgumentError, ArithmeticError, BadMapError] ->
          C.fail("#{name}(#{Enum.map_join(args, ", ", &show_short/1)}): #{short_reason(e)}")
      end
    end
  end

  defp short_reason(%FunctionClauseError{}), do: "wrong kind of arguments"
  defp short_reason(%ArithmeticError{}), do: "arithmetic error"
  defp short_reason(e), do: Exception.message(e) |> String.slice(0, 80)
end
