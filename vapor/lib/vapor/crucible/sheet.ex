defmodule Vapor.Crucible.Sheet do
  @moduledoc """
  The input form shared by the Crucible's domains (docs/CRUCIBLE.md §1):
  statements one per line (or separated by `;`), `#` comments,

      V(x) = 0.5*x^2 + 0.1*x^4          # a function of named arguments
      k = 2.5                            # a constant (any expression of earlier constants)
      x = -8 .. 8                        # a range
      q(0) = 1.0                         # an initial value
      method = yoshida4                  # a word

  Expressions are the workbench's (`Vapor.Expr`): infix arithmetic,
  functions, `pi`, `e`, units in brackets. Nothing in them can reach the
  host.
  """
  alias Vapor.Expr

  defstruct funs: %{}, consts: %{}, ranges: %{}, inits: %{}, words: %{}, order: [], raw: [], errors: []

  @doc "Parse a sheet. Constants are evaluated in order; errors are collected with their line."
  def parse(text) when is_binary(text) do
    stmts =
      text
      |> String.split("\n")
      |> Enum.with_index(1)
      |> Enum.flat_map(fn {line, n} ->
        line |> String.replace(~r/#.*$/, "") |> String.split(";") |> Enum.map(&{String.trim(&1), n})
      end)
      |> Enum.reject(fn {s, _} -> s == "" end)

    Enum.reduce(stmts, %__MODULE__{}, &statement/2)
    |> then(fn s -> %{s | order: Enum.reverse(s.order), errors: Enum.reverse(s.errors)} end)
  end

  defp statement({s, n}, acc) do
    cond do
      m = Regex.run(~r/^([\p{L}_][\w]*)\(\s*([+-]?[\d.eE+-]+)\s*\)\s*=\s*(.+)$/u, s) ->
        [_, name, at, rhs] = m
        with {:ok, v} <- num(rhs, acc) do
          %{acc | inits: Map.put(acc.inits, name, {num!(at), v}), order: [{:init, name} | acc.order]}
        else
          {:error, e} -> err(acc, n, e)
        end

      m = Regex.run(~r/^([\p{L}_][\w]*)\(([^)]*)\)\s*=\s*(.+)$/u, s) ->
        [_, name, args, rhs] = m
        args = args |> String.split(",") |> Enum.map(&String.trim/1) |> Enum.reject(&(&1 == ""))
        case Expr.parse(rhs) do
          {:ok, t} -> %{acc | funs: Map.put(acc.funs, name, {args, Expr.subst(t, Map.new(acc.consts, fn {k, v} -> {k, {:n, v}} end)), rhs}), order: [{:fun, name} | acc.order]}
          {:error, e} -> err(acc, n, "#{name}: #{e}")
        end

      m = Regex.run(~r/^([\p{L}_][\w]*)\s*=\s*(.+?)\s*\.\.\s*(.+)$/u, s) ->
        [_, name, a, b] = m
        with {:ok, x} <- num(a, acc), {:ok, y} <- num(b, acc) do
          %{acc | ranges: Map.put(acc.ranges, name, {x, y}), order: [{:range, name} | acc.order]}
        else
          {:error, e} -> err(acc, n, "#{name}: #{e}")
        end

      m = Regex.run(~r/^([\p{L}_][\w]*)\s*=\s*([\p{L}_][\w\-]*)$/u, s) ->
        [_, name, word] = m
        case num(word, acc) do
          {:ok, v} -> %{acc | consts: Map.put(acc.consts, name, v), order: [{:const, name} | acc.order]}
          _ -> %{acc | words: Map.put(acc.words, name, word), order: [{:word, name} | acc.order]}
        end

      m = Regex.run(~r/^([\p{L}_][\w]*)\s*=\s*(.+)$/u, s) ->
        [_, name, rhs] = m
        case num(rhs, acc) do
          {:ok, v} -> %{acc | consts: Map.put(acc.consts, name, v), order: [{:const, name} | acc.order]}
          {:error, _} ->
            # a definition of a function of no arguments that uses variables (H = p^2/2 + …): kept as a function of its free variables
            case Expr.parse(rhs) do
              {:ok, t} ->
                t = Expr.subst(t, Map.new(acc.consts, fn {k, v} -> {k, {:n, v}} end))
                %{acc | funs: Map.put(acc.funs, name, {Expr.vars(t), t, rhs}), order: [{:fun, name} | acc.order]}
              {:error, e} -> err(acc, n, "#{name}: #{e}")
            end
        end

      true ->
        %{acc | raw: acc.raw ++ [{n, s}]}
    end
  end

  defp err(acc, n, e), do: %{acc | errors: ["line #{n}: #{e}" | acc.errors]}

  @doc "Evaluate an expression text to a number with the sheet's constants."
  def num(text, %__MODULE__{consts: c}) do
    with {:ok, t} <- Expr.parse(text) do
      try do
        {:ok, Expr.eval(t, c) * 1.0}
      rescue
        e -> {:error, Exception.message(e)}
      end
    end
  end

  defp num!(s), do: (case Float.parse(s) do {f, _} -> f; :error -> 0.0 end)

  @doc "A constant, or a default."
  def const(%__MODULE__{consts: c}, name, default \\ nil), do: Map.get(c, name, default)
  def word(%__MODULE__{words: w}, name, default \\ nil), do: Map.get(w, name, default)

  @doc "A one-argument function f(x) as a BEAM function (raising on domain errors)."
  def fun1(%__MODULE__{funs: f}, name) do
    case Map.fetch(f, name) do
      {:ok, {[a], t, _}} -> {:ok, fn x -> Expr.eval(t, %{a => x}) end, t, a}
      {:ok, {args, _, _}} -> {:error, "#{name} should take one argument, it takes #{length(args)}"}
      :error -> {:error, "#{name} is not defined"}
    end
  end
end
