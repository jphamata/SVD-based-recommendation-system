defmodule Vapor.Athanor.Spec do
  @moduledoc """
  A problem for the Athanor, read from an Alembic program
  (docs/ATHANOR.md §1). Reserved names:

  | name | meaning |
  |---|---|
  | `space = kind(…)` | where candidates live (`Vapor.Athanor.Space`) — required |
  | `minimize(x)` · `maximize(x)` | the objective, a number |
  | `valid(x)` | a hard constraint; invalid candidates get no score |
  | `violation(x)` | how far from valid (0 = valid): ranks invalid candidates below every valid one so the search can climb toward validity; alone it also defines `valid` |
  | `claim(x)` | a statement to refute: the search looks for an `x` where it is false |
  | `margin(x)` | with `claim`: how far `x` is from refuting it (≤ 0 refutes) — guides the search |
  | `target` | stop when the objective reaches it (a known bound, a goal) |
  | `budget` · `seed` | evaluations and the seed (both overridable from outside) |
  | `show(x)` | how a candidate is displayed |
  | `describe(x)` | a list of numbers — keeps a diverse archive (MAP-Elites) |
  | `holdout(x)` | a second, unseen objective, computed only for the finalists — measures the search's own selection bias |
  | `neighbor(x, s)` | a custom local move (s: an integer seed for `noise`) |
  | `start` | a list of candidates to begin from |
  | `measured = true` | the objective is measured outside (a lab, a person, a command): `minimize(x)` may be absent |

  Without an objective and without a claim, the search looks for any
  valid candidate (`find`).
  """
  alias Vapor.Alembic
  alias Vapor.Athanor.Space

  defstruct [:prog, :space, :sense, :objective, :source, :hash, valid: false, violation: false, target: nil, budget: 2000, seed: 1,
             show: false, describe: false, holdout: false, neighbor: false, margin: false, start: [], measured: false, notes: []]

  @doc "`{:ok, %Spec{}}` or `{:error, message}`. Options: `consts:` overrides, `budget:`, `seed:`."
  def parse(text, opts \\ []) do
    consts = Keyword.get(opts, :consts, %{})

    with {:ok, prog} <- load(text, consts),
         {:ok, space_ast} <- fetch_space(prog),
         {:ok, space} <- Space.from_ast(prog, space_ast) do
      has = &Alembic.defined?(prog, &1, 1)
      measured = Alembic.const(prog, "measured") == true or Keyword.get(opts, :measured, false)

      {sense, objective} =
        cond do
          has.("minimize") -> {:min, "minimize"}
          has.("maximize") -> {:max, "maximize"}
          has.("claim") -> {:claim, "claim"}
          measured -> {Keyword.get(opts, :sense, :min), nil}
          true -> {:find, nil}
        end

      notes =
        [has.("minimize") and has.("maximize") && "both minimize and maximize are defined: minimize is used",
         has.("claim") and objective != "claim" && "claim is ignored when an objective is defined",
         (sense == :find and not has.("valid")) && "no objective, no claim and no valid(x): every candidate is a solution"]
        |> Enum.filter(&is_binary/1)

      start =
        case Alembic.const(prog, "start") do
          xs when is_list(xs) -> xs
          _ -> []
        end

      {:ok,
       %__MODULE__{prog: prog, space: space, sense: sense, objective: objective, source: text, hash: prog.hash,
                   valid: has.("valid") or has.("violation"), violation: has.("violation"), target: num_or_nil(Alembic.const(prog, "target")),
                   budget: Keyword.get(opts, :budget) || int_or(Alembic.const(prog, "budget"), 2000),
                   seed: Keyword.get(opts, :seed) || int_or(Alembic.const(prog, "seed"), 1),
                   show: has.("show"), describe: has.("describe"), holdout: has.("holdout"),
                   neighbor: Alembic.defined?(prog, "neighbor", 2), margin: has.("margin"), start: start, measured: measured, notes: notes}}
    end
  end

  defp load(text, consts) do
    case Alembic.load(text, skip: ["space"], consts: consts) do
      {:ok, p} -> {:ok, p}
      {:error, e} -> {:error, Alembic.format_error(e)}
    end
  end

  defp fetch_space(prog) do
    case Map.fetch(prog.reserved, "space") do
      {:ok, {ast, _line}} -> {:ok, ast}
      :error -> {:error, "no space: write a line like `space = perm(8)` or `space = reals(3, -5, 5)` (see docs/ATHANOR.md)"}
    end
  end

  defp num_or_nil(v) when is_number(v), do: v
  defp num_or_nil(_), do: nil
  defp int_or(v, _d) when is_integer(v) and v > 0, do: v
  defp int_or(_, d), do: d

  @doc "What the problem asks, in words."
  def describe(%__MODULE__{} = s) do
    what =
      case s.sense do
        :min -> "minimise " <> if(s.objective, do: "minimize(x)", else: "a measured objective")
        :max -> "maximise " <> if(s.objective, do: "maximize(x)", else: "a measured objective")
        :claim -> "refute claim(x)"
        :find -> "find a valid candidate"
      end

    "#{what} over #{s.space.text}" <> if(s.valid, do: ", subject to valid(x)", else: "") <> size_text(s.space.size)
  end

  defp size_text(:infinite), do: ""
  defp size_text(n) when n < 1_000_000, do: " (#{n} candidates)"
  defp size_text(n), do: " (≈ 10^#{n |> Integer.to_string() |> String.length() |> Kernel.-(1)} candidates)"
end
