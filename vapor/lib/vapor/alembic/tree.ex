defmodule Vapor.Alembic.Tree do
  @moduledoc """
  The numeric subset of Alembic as a portable tree (docs/ALEMBIC.md §6):
  what a scene's motion is written in, evaluated **by an interpreter** in
  the browser (and here), never compiled to code — a scene file carries
  data, so a hostile one can only draw.

      0.5 + 0.2*sin(t/3)   →   ["+", 0.5, ["*", 0.2, ["f", "sin", [["/", ["v", "t"], 3]]]]]

  Nodes: numbers; `["v", name]`; `[op, a, b]` for `+ - * / ^ % < <= > >= == !=`;
  `["neg", a]`; `["and"|"or", a, b]`; `["not", a]`; `["if", c, a, b]`;
  `["f", name, [args]]` for the functions in `functions/0`.
  """
  alias Vapor.Alembic.Parser

  @functions ~w(sin cos tan asin acos atan atan2 exp ln log sqrt abs floor ceil round min max clamp hypot pow sign noise fract mix smooth tri sq)

  @doc "The functions a tree may call."
  def functions, do: @functions

  @doc "Parse text into a tree over the allowed `vars`: `{:ok, tree}` or `{:error, why}`."
  def parse(text, vars) when is_binary(text) do
    with {:ok, ast} <- Parser.expression(text) do
      try do
        {:ok, conv(ast, MapSet.new(vars))}
      catch
        {:tree, m} -> {:error, m}
      end
    else
      {:error, e} -> {:error, Vapor.Alembic.format_error(e)}
    end
  end

  defp conv({:lit, n}, _) when is_number(n), do: n
  defp conv({:lit, true}, _), do: 1
  defp conv({:lit, false}, _), do: 0
  defp conv({:var, n, _}, vars) do
    cond do
      MapSet.member?(vars, n) -> ["v", n]
      n in ["pi", "π"] -> :math.pi()
      n == "tau" -> 2 * :math.pi()
      n == "e" -> :math.exp(1)
      true -> throw({:tree, "unknown name #{n} (known: #{vars |> Enum.sort() |> Enum.join(", ")})"})
    end
  end
  defp conv({:neg, a, _}, v), do: ["neg", conv(a, v)]
  defp conv({:not, a}, v), do: ["not", conv(a, v)]
  defp conv({:and, a, b}, v), do: ["and", conv(a, v), conv(b, v)]
  defp conv({:or, a, b}, v), do: ["or", conv(a, v), conv(b, v)]
  defp conv({:if, c, a, b, _}, v), do: ["if", conv(c, v), conv(a, v), conv(b, v)]
  defp conv({:bin, op, a, b, _}, v) when op in ["+", "-", "*", "/", "^", "%"], do: [op, conv(a, v), conv(b, v)]
  defp conv({:bin, "//", a, b, _}, v), do: ["f", "floor", [["/", conv(a, v), conv(b, v)]]]
  defp conv({:cmp, a, [{op, b}], _}, v) when op in ["<", "<=", ">", ">=", "==", "!="], do: [op, conv(a, v), conv(b, v)]
  defp conv({:call, {:var, f, _}, args, _}, v) when f in @functions, do: ["f", f, Enum.map(args, &conv(&1, v))]
  defp conv({:call, {:var, f, _}, _, _}, _), do: throw({:tree, "#{f} is not available in motion expressions (#{Enum.join(@functions, " ")})"})
  defp conv(_, _), do: throw({:tree, "only numeric expressions here: numbers, variables, + − * / ^ %, comparisons, if, and the functions #{Enum.join(@functions, " ")}"})

  @doc "Evaluate a tree with variable values (the same semantics as the browser's interpreter)."
  def eval(n, _env) when is_number(n), do: n * 1.0
  def eval(["v", name], env), do: Map.get(env, name, 0.0) * 1.0
  def eval(["neg", a], env), do: -eval(a, env)
  def eval(["not", a], env), do: b(eval(a, env) == 0)
  def eval(["and", a, c], env), do: b(eval(a, env) != 0 and eval(c, env) != 0)
  def eval(["or", a, c], env), do: b(eval(a, env) != 0 or eval(c, env) != 0)
  def eval(["if", c, a, d], env), do: if(eval(c, env) != 0, do: eval(a, env), else: eval(d, env))
  def eval(["f", f, args], env), do: fun(f, Enum.map(args, &eval(&1, env)))
  def eval([op, a, c], env), do: binop(op, eval(a, env), eval(c, env))

  defp b(true), do: 1.0
  defp b(false), do: 0.0

  defp binop("+", x, y), do: x + y
  defp binop("-", x, y), do: x - y
  defp binop("*", x, y), do: x * y
  defp binop("/", x, y), do: if(y == 0, do: 0.0, else: x / y)
  defp binop("%", x, y), do: if(y == 0, do: 0.0, else: x - y * Float.floor(x / y))
  defp binop("^", x, y), do: (try do :math.pow(x, y) rescue _ -> 0.0 end)
  defp binop("<", x, y), do: b(x < y)
  defp binop("<=", x, y), do: b(x <= y)
  defp binop(">", x, y), do: b(x > y)
  defp binop(">=", x, y), do: b(x >= y)
  defp binop("==", x, y), do: b(x == y)
  defp binop("!=", x, y), do: b(x != y)

  # total functions: out-of-domain inputs give 0, as in the browser (a drawing never throws)
  defp fun(f, args) do
    try do
      fun!(f, args)
    rescue
      _ -> 0.0
    end
  end

  defp fun!("sin", [x]), do: :math.sin(x)
  defp fun!("cos", [x]), do: :math.cos(x)
  defp fun!("tan", [x]), do: :math.tan(x)
  defp fun!("asin", [x]), do: :math.asin(x)
  defp fun!("acos", [x]), do: :math.acos(x)
  defp fun!("atan", [x]), do: :math.atan(x)
  defp fun!("atan2", [y, x]), do: :math.atan2(y, x)
  defp fun!("exp", [x]), do: :math.exp(min(x, 700.0))
  defp fun!(l, [x]) when l in ["ln", "log"], do: :math.log(x)
  defp fun!("sqrt", [x]), do: :math.sqrt(x)
  defp fun!("abs", [x]), do: abs(x)
  defp fun!("floor", [x]), do: Float.floor(x)
  defp fun!("ceil", [x]), do: Float.ceil(x)
  defp fun!("round", [x]), do: Float.round(x)
  defp fun!("min", xs), do: Enum.min(xs)
  defp fun!("max", xs), do: Enum.max(xs)
  defp fun!("clamp", [x, a, c]), do: x |> max(a) |> min(c)
  defp fun!("hypot", [x, y]), do: :math.sqrt(x * x + y * y)
  defp fun!("pow", [x, y]), do: :math.pow(x, y)
  defp fun!("sign", [x]), do: (cond do x > 0 -> 1.0; x < 0 -> -1.0; true -> 0.0 end)
  defp fun!("fract", [x]), do: x - Float.floor(x)
  defp fun!("mix", [a, c, t]), do: a + (c - a) * t
  defp fun!("smooth", [a, c, x]), do: (t = ((x - a) / (c - a)) |> max(0.0) |> min(1.0); t * t * (3 - 2 * t))
  defp fun!("tri", [x]), do: (f = x - Float.floor(x); 1 - abs(2 * f - 1))
  defp fun!("sq", [x]), do: x * x
  defp fun!("noise", args), do: noise(args)

  @doc "Deterministic noise in [0, 1) from numbers — the browser computes the same (integer hash of the rounded arguments)."
  def noise(args) do
    Enum.reduce(args, 2_166_136_261, fn x, h ->
      k = trunc(Float.floor(x * 1000 + 0.5)) |> Bitwise.band(0xFFFFFFFF)
      h = Bitwise.bxor(h, k) |> Kernel.*(16_777_619) |> Bitwise.band(0xFFFFFFFF)
      h = Bitwise.bxor(h, Bitwise.bsr(h, 13)) |> Kernel.*(1_274_126_177) |> Bitwise.band(0xFFFFFFFF)
      Bitwise.bxor(h, Bitwise.bsr(h, 16))
    end)
    |> Kernel./(4_294_967_296)
  end
end
