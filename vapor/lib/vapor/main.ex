defmodule Vapor.Main do
  @moduledoc """
  `vapor` — every capability from the terminal, in the Unix manner
  (docs/CLI.md):

    * one verb per tool: `vapor alembic`, `vapor athanor`, `vapor game`,
      `vapor crucible`, `vapor assay`, `vapor mind`, `vapor scene`,
      `vapor solve`, `vapor verify`, and the older tasks (`serve`, `tui`,
      `ocr`, `merge`, `quality`, …) through `bin/vapor`;
    * input from a file or from standard input (`-`, or a pipe);
    * output for people on a terminal, **JSON when piped** (or with
      `--json`), so `vapor athanor run p.alb | vapor verify p.alb -` works;
    * exit status: 0 — done, the answer is positive; 1 — done, the answer
      is negative (refuted, not found, verification failed); 2 — usage;
      3 — the input is not valid (a parse error, a bad file); 4 — failure.
    * colour only on a terminal, never with `NO_COLOR`; diagnostics on
      standard error.
  """
  alias Vapor.Alembic

  @verbs ~w(alembic athanor search game crucible assay mind scene solve verify rebis aludel tabula cupel amalgam help version)

  @doc "Entry point: run and halt with the exit status."
  def main(argv) do
    code =
      try do
        run(Enum.map(argv, &Vapor.CLI.utf8_arg/1))
      rescue
        e ->
          err("vapor: internal error: " <> Exception.message(e))
          if System.get_env("VAPOR_DEBUG"), do: err(Exception.format(:error, e, __STACKTRACE__))
          4
      end

    System.halt(code)
  end

  @doc "Run a command line; returns the exit status (no halt) — what the tests call."
  def run([]), do: run(["help"])
  def run([v | _]) when v in ["-h", "--help", "help"], do: (out(help()); 0)
  def run([v | _]) when v in ["--version", "version"], do: (out("vapor " <> to_string(Application.spec(:vapor, :vsn) || "dev")); 0)
  def run(["alembic" | rest]), do: Vapor.Main.AlembicCli.run(rest)
  def run(["athanor" | rest]), do: Vapor.Main.AthanorCli.run(rest)
  def run(["search" | rest]), do: Vapor.Main.AthanorCli.run(["run" | rest])
  def run(["game" | rest]), do: Vapor.Main.AthanorCli.game(rest)
  def run(["verify" | rest]), do: Vapor.Main.AthanorCli.verify(rest)
  def run(["crucible" | rest]), do: Vapor.Main.ScienceCli.run(rest)
  def run(["assay" | rest]), do: Vapor.Main.AssayCli.run(rest)
  def run(["mind" | rest]), do: Vapor.Main.MindCli.run(rest)
  def run(["scene" | rest]), do: Vapor.Main.SceneCli.run(rest)
  def run(["solve" | rest]), do: solve(rest)
  def run(["rebis" | rest]), do: Vapor.Main.OpusCli.rebis(rest)
  def run(["aludel" | rest]), do: Vapor.Main.OpusCli.aludel(rest)
  def run(["tabula" | rest]), do: Vapor.Main.OpusCli.tabula(rest)
  def run(["cupel" | rest]), do: Vapor.Main.OpusCli.cupel(rest)
  def run(["amalgam" | rest]), do: Vapor.Main.OpusCli.amalgam(rest)
  def run([verb | _]) do
    err("vapor: unknown command #{verb}. Commands: #{Enum.join(@verbs, ", ")} (and serve, tui, ocr, merge, quality, rag, lock, train… through bin/vapor)")
    2
  end

  defp help do
    """
    vapor — a laboratory you drive from the terminal. Every result carries what lets you judge it.

      vapor alembic FILE | -e EXPR | --card       run Alembic, the problem language (reference card: --card)
      vapor athanor run FILE [opts]               search any problem written in Alembic; prints a certificate
      vapor athanor ask FILE                      a measured problem: vapor proposes, you measure (Bayesian)
      vapor verify FILE CERT [--full --replay]    re-check a certificate (Touchstone) without trusting the search
      vapor game FILE solve|search|learn|play     any two-player game written in Alembic
      vapor crucible KIND FILE                    open science: quantum, hamiltonian, laws, reactions, evolution,
                                                  phylogeny, molecule, fold, fields, regress (`vapor crucible` lists)
      vapor assay TOOL FILE                       AI research: compare, leaderboard, contamination, dedup, scaling,
                                                  calibration, agreement (`vapor assay` lists)
      vapor mind ask|formalize|propose …          a language model, always checked (VAPOR_MIND=anthropic:MODEL …)
      vapor scene new|edit|direct|export …        scenes as documents edited by operations
      vapor solve FILE                            the workbench: equations with units, ODEs, PDEs, fits
      vapor rebis equiv|anf|identity|stabilizer … circuits over GF(2): proved equal or told apart
      vapor aludel decide POLY --box … | REQ.json polynomial claims and barrier certificates, decided exactly
      vapor tabula FILE [--facts a,b]             a contract: antinomies, proofs of consistency, positions
      vapor cupel | vapor amalgam [FILE]          silent-corruption drill · sums that do not depend on order
      vapor serve | tui | ocr | merge | quality … the console and the older tasks (via bin/vapor)

    FILE may be - (standard input). Output is JSON when piped or with --json.
    Exit: 0 positive · 1 negative (refuted, not found, failed check) · 2 usage · 3 bad input · 4 failure.
    """
  end

  # ================================================================ shared IO

  @doc "Write to standard output."
  def out(text), do: IO.puts(text)

  @doc "Write to standard error."
  def err(text), do: IO.puts(:stderr, text)

  @doc "Is standard output a terminal?"
  def tty? do
    case System.get_env("VAPOR_TTY") do
      "1" -> true
      "0" -> false
      _ -> match?({:ok, _}, :io.columns())
    end
  end

  @doc "Should output be JSON (`--json`, or not a terminal)?"
  def json?(opts), do: Keyword.get(opts, :json, false) or not tty?()

  @doc "Colour (only on a terminal, never with NO_COLOR)."
  def c(text, code) do
    if tty?() and System.get_env("NO_COLOR") in [nil, ""], do: "\e[#{code}m#{text}\e[0m", else: text
  end

  def bold(t), do: c(t, "1")
  def dim(t), do: c(t, "2")
  def good(t), do: c(t, "36")
  def warn(t), do: c(t, "33")
  def bad(t), do: c(t, "31")

  @doc "Read a FILE argument (`-` or nil = standard input)."
  def read_input(nil) do
    if tty?() and System.get_env("VAPOR_TTY") != "0", do: {:error, "no input: give a FILE or pipe it in"}, else: read_input("-")
  end

  def read_input("-") do
    case IO.read(:stdio, :eof) do
      {:error, e} -> {:error, "standard input: #{inspect(e)}"}
      :eof -> {:ok, ""}
      data -> {:ok, data}
    end
  end

  def read_input(path) do
    case File.read(path) do
      {:ok, d} -> {:ok, d}
      {:error, e} -> {:error, "#{path}: #{:file.format_error(e)}"}
    end
  end

  @doc "Print a value as JSON."
  def emit_json(v), do: out(Vapor.JSON.encode(jsonable(v)))

  @doc "A value as JSON-ready terms."
  def jsonable(%{__struct__: _} = s), do: s |> Map.from_struct() |> jsonable()
  def jsonable(m) when is_map(m), do: Map.new(m, fn {k, v} -> {to_string(k), jsonable(v)} end)
  def jsonable(l) when is_list(l), do: Enum.map(l, &jsonable/1)
  def jsonable({:fn, name, _, _}), do: "<function #{name}>"
  def jsonable(t) when is_tuple(t), do: t |> Tuple.to_list() |> jsonable()
  def jsonable(a) when is_atom(a) and a not in [nil, true, false], do: Atom.to_string(a)
  def jsonable(b) when is_binary(b), do: if(String.valid?(b), do: b, else: Base.encode64(b))
  def jsonable(x) when is_float(x), do: x
  def jsonable(x), do: x

  @doc "Parse options; on an unknown option print usage and return :usage."
  def opts(argv, strict, aliases \\ []) do
    case OptionParser.parse(argv, strict: [json: :boolean] ++ strict, aliases: aliases) do
      {o, args, []} -> {:ok, o, args}
      {_, _, bad} -> err("vapor: unknown or malformed option(s): " <> Enum.map_join(bad, " ", fn {k, _} -> k end)); :usage
    end
  end

  @doc "`--set name=value` pairs as Alembic constants."
  def consts(o) do
    o
    |> Keyword.get_values(:set)
    |> Enum.reduce_while({:ok, %{}}, fn kv, {:ok, acc} ->
      case String.split(kv, "=", parts: 2) do
        [k, v] ->
          case Alembic.literal(v) do
            {:ok, val} -> {:cont, {:ok, Map.put(acc, String.trim(k), val)}}
            {:error, m} -> {:halt, {:error, "--set #{kv}: #{m}"}}
          end
        _ -> {:halt, {:error, "--set expects name=value, got #{kv}"}}
      end
    end)
  end

  # ================================================================ solve

  defp solve(argv) do
    with {:ok, o, args} <- opts(argv, []),
         {:ok, text} <- read_input(List.first(args)) do
      case Vapor.Solve.run(text) do
        {:ok, r} ->
          if json?(o), do: emit_json(slim(r)), else: out(solve_text(r))
          0
        {:error, why} -> err("vapor solve: " <> to_string(why)); 3
      end
    else
      :usage -> 2
      {:error, m} -> err("vapor solve: " <> m); 3
    end
  end

  defp slim(r), do: Vapor.Console.Lab12.slim(r)

  defp solve_text(%{kind: "worksheet", lines: lines}), do: Enum.map_join(lines, "\n", &inspect/1)
  defp solve_text(r), do: r |> slim() |> jsonable() |> Vapor.JSON.encode()
end
