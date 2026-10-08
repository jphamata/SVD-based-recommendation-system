defmodule Vapor.Main do
  @moduledoc """
  `vapor` — every capability from the terminal, in the Unix manner
  (docs/CLI.md):

    * one verb per tool: `vapor alembic`, `vapor athanor`, `vapor game`,
      `vapor crucible`, `vapor assay`, `vapor mind`, `vapor render`, `vapor qalam`,
      `vapor solve`, `vapor verify`, and the older tasks (`serve`, `tui`,
      `ocr`, `merge`, `quality`, …) through `bin/vapor`;
    * input from a file or from standard input (`-`, or a pipe);
    * output for people on a terminal, **JSON when piped** (or with
      `--json`), so `vapor athanor run p.nbq | vapor verify p.nbq -` works;
    * exit status: 0 — done, the answer is positive; 1 — done, the answer
      is negative (refuted, not found, verification failed); 2 — usage;
      3 — the input is not valid (a parse error, a bad file); 4 — failure.
    * colour only on a terminal, never with `NO_COLOR`; diagnostics on
      standard error.
  """
  alias Vapor.Alembic

  @verbs ~w(alembic athanor search game crucible assay mind render qalam solve verify logic rebis aludel tabula cupel amalgam qalib recommend palingenesis siphon chat wzn lsp help version)

  @doc "The verbs (what `help` lists and the console's terminal completes)."
  def verbs, do: @verbs

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
  def run(["render" | rest]), do: render(rest)
  def run(["qalam" | rest]), do: qalam(rest)
  def run(["solve" | rest]), do: solve(rest)
  def run(["logic" | rest]), do: Vapor.Main.ForgeCli.logic(rest)
  def run(["qalib" | rest]), do: Vapor.Main.ForgeCli.qalib(rest)
  def run(["recommend" | rest]), do: Vapor.Main.ForgeCli.recommend(rest)
  def run(["palingenesis" | rest]), do: Vapor.Main.ForgeCli.palingenesis(rest)
  def run(["siphon" | rest]), do: Vapor.Main.SiphonCli.run(rest)
  def run(["rebis" | rest]), do: Vapor.Main.OpusCli.rebis(rest)
  def run(["aludel" | rest]), do: Vapor.Main.OpusCli.aludel(rest)
  def run(["tabula" | rest]), do: Vapor.Main.OpusCli.tabula(rest)
  def run(["cupel" | rest]), do: Vapor.Main.OpusCli.cupel(rest)
  def run(["amalgam" | rest]), do: Vapor.Main.OpusCli.amalgam(rest)
  def run(["chat" | rest]), do: Vapor.Main.ChatCli.run(rest)
  def run(["wzn" | rest]), do: Vapor.Main.AlmizanCli.run(rest)
  def run(["almizan" | rest]), do: Vapor.Main.AlmizanCli.run(rest)
  def run(["lsp" | _]), do: (Vapor.LSP.serve(); 0)
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
      vapor render FILE [--ink] [--out F.png]     a scene path-traced (physical light), or the same scene in ink and flat colour
      vapor qalam FILE                            the editor: vi keys, each claim's verdict in the gutter, scrubbable numbers
      vapor solve FILE                            the workbench: equations with units, ODEs, PDEs, fits
      vapor logic FILE | check FILE PROPOSAL      SAT, LP, integer LP, causal diagrams, Gröbner…: decided, with certificates
      vapor rebis equiv|anf|identity|stabilizer … circuits over GF(2): proved equal or told apart
      vapor qalib map|check …                     sky130 netlists: mapped, read back, proved equal (or a counterexample)
      vapor recommend RATINGS.csv                 matrix factorisation against baselines, a paired test, a shuffled control
      vapor palingenesis planks|try MODEL …       renew a model plank by plank, through the drift brake and the target test
      vapor siphon run|queue|approve|headers …    the network airlock: fetchers you declare, run only when you say so
      vapor aludel decide POLY --box … | REQ.json polynomial claims and barrier certificates, decided exactly
      vapor tabula FILE [--facts a,b]             a contract: antinomies, proofs of consistency, positions
      vapor cupel | vapor amalgam [FILE]          silent-corruption drill · sums that do not depend on order
      vapor chat [new|say|show|edit|regen|fork|context|search|export|import|share …]
                                                  conversations as a tree: branches, forks, context, compaction
      vapor wzn check|show|run|transmute|assay FILE  Almizan: claims decided (Latin or Arabic script), lowered by vapor's compiler
      vapor lsp                                   the language server for editors (Almizan, Alembic)
      vapor serve | tui | ocr | merge | quality … the console and the older tasks (via bin/vapor)

    FILE may be - (standard input). Output is JSON when piped or with --json.
    Exit: 0 positive · 1 negative (refuted, not found, failed check) · 2 usage · 3 bad input · 4 failure.
    """
  end

  # ================================================================ shared IO

  @doc "Write to standard output."
  def out(text), do: IO.puts(text)

  @doc "Write to standard error (inside `Vapor.Diwan`, to the session's error stream)."
  def err(text) do
    case Process.get(:vapor_stderr) do
      nil -> IO.puts(:stderr, text)
      io -> IO.puts(io, text)
    end
  end

  @doc "Is standard output a terminal? (`Vapor.Diwan` decides per pipeline stage.)"
  def tty? do
    case Process.get(:vapor_tty) do
      nil ->
        case System.get_env("VAPOR_TTY") do
          "1" -> true
          "0" -> false
          _ -> match?({:ok, _}, :io.columns())
        end

      v ->
        v
    end
  end

  @doc """
  Whether this command runs jailed (`Vapor.Diwan` in the console's
  terminal): files are the session's, never the server's, and nothing runs
  outside the BEAM.
  """
  def jailed?, do: Process.get(:vapor_jail) != nil

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
    case Process.get(:vapor_jail) do
      nil ->
        case File.read(path) do
          {:ok, d} -> {:ok, d}
          {:error, e} -> {:error, "#{path}: #{:file.format_error(e)}"}
        end

      files ->
        case Map.fetch(files, Vapor.Diwan.clean_path(path)) do
          {:ok, d} -> {:ok, d}
          :error -> {:error, "#{path}: no such file in this session (ls lists them; files come from > redirection, the editor or an upload)"}
        end
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

  # vapor qalam FILE: Al-Qalam, the editor (Vapor.Qalam), on this terminal
  defp qalam(argv) do
    with {:ok, _o, [path]} <- opts(argv, []),
         :ok <- if(jailed?(), do: {:error, "the editor needs your own terminal; in the console, the Workspace edits files"}, else: :ok),
         :ok <- Vapor.Qalam.run(path) do
      0
    else
      :usage -> 2
      {:ok, _, _} -> err("vapor qalam FILE"); 2
      {:error, m} -> err("vapor qalam: " <> m); 2
    end
  end

  # vapor render FILE: the scene language of docs/RENDER.md; the PNG goes to --out (default: FILE with .png, or render.png)
  defp render(argv) do
    with {:ok, o, args} <- opts(argv, [ink: :boolean, width: :integer, height: :integer, spp: :integer, bands: :integer, seed: :integer, out: :string]),
         :ok <- if(jailed?(), do: :jailed, else: :ok),
         {:ok, text} <- read_input(List.first(args)),
         {:ok, scene} <- Vapor.Render.parse(text) do
      {w, h} = {o[:width] || 240, o[:height] || 150}
      file = o[:out] || if(List.first(args) in [nil, "-"], do: "render.png", else: Path.rootname(hd(args)) <> ".png")

      info =
        if o[:ink] do
          r = Vapor.Render.ink(scene, width: w, height: h, bands: o[:bands] || 3)
          File.write!(file, r.png)
          %{file: file, width: r.w, height: r.h, style: "ink", outline_pixels: r.edges, ms: r.ms}
        else
          r = Vapor.Render.render(scene, width: w, height: h, spp: o[:spp] || 16, seed: o[:seed] || 1)
          File.write!(file, r.png)
          lum = r.linear |> List.flatten() |> Enum.map(fn {a, b, c} -> 0.2126 * a + 0.7152 * b + 0.0722 * c end)
          %{file: file, width: r.w, height: r.h, style: "physical", spp: r.spp, mean_radiance: Enum.sum(lum) / length(lum), ms: r.ms}
        end

      if json?(o), do: emit_json(info), else: out("#{bold(file)}  #{info.width}×#{info.height} · #{info.style}" <> dim("  #{info.ms} ms"))
      0
    else
      :usage -> 2
      :jailed -> err("vapor render: writes a file on the server; in the console, the render desk draws the same scene"); 2
      {:error, m} -> err("vapor render: " <> to_string(m)); 3
    end
  end

  defp solve_text(%{kind: "worksheet", lines: lines}), do: Enum.map_join(lines, "\n", &inspect/1)
  defp solve_text(r), do: r |> slim() |> jsonable() |> Vapor.JSON.encode()
end
