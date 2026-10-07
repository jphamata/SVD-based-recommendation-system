defmodule Vapor.Main.AlembicCli do
  @moduledoc false
  # vapor alembic — run the language from the terminal
  import Vapor.Main
  alias Vapor.Alembic

  def run(argv) do
    case opts(argv, [eval: :string, card: :boolean, call: :string, args: :string, fuel: :integer, set: :keep], e: :eval) do
      :usage -> 2
      {:ok, o, args} ->
        cond do
          o[:card] -> out(Alembic.card()); 0
          o[:eval] -> eval(o, args)
          args == [] and tty?() -> repl(o)
          true -> file(o, args)
        end
    end
  end

  defp load_prog(nil, _o), do: {:ok, nil}

  defp load_prog(path, o) do
    with {:ok, text} <- read_input(path),
         {:ok, consts} <- consts(o) do
      case Alembic.load(text, consts: consts, fuel: o[:fuel] || 10_000_000) do
        {:ok, p} -> {:ok, p}
        {:error, e} -> {:error, Alembic.format_error(e)}
      end
    end
  end

  defp eval(o, args) do
    with {:ok, prog} <- load_prog(List.first(args), o) do
      case Alembic.eval(o[:eval], program: prog, fuel: o[:fuel] || Alembic.default_fuel()) do
        {:ok, v} -> print_value(v, o); if(v in [false, nil], do: 1, else: 0)
        {:error, e} -> err("alembic: " <> Alembic.format_error(e)); 3
      end
    else
      {:error, m} -> err("alembic: " <> m); 3
    end
  end

  defp file(o, args) do
    with {:ok, prog} <- load_prog(List.first(args) || "-", o) do
      cond do
        o[:call] ->
          argl = case o[:args] do nil -> {:ok, []}; a -> Alembic.literal(a) end
          case argl do
            {:ok, l} when is_list(l) ->
              case Alembic.call(prog, o[:call], l, fuel: o[:fuel] || Alembic.default_fuel()) do
                {:ok, v} -> print_value(v, o); 0
                {:error, m} -> err("alembic: " <> m); 3
              end
            _ -> err("alembic: --args must be a list literal, like '[3, [1, 2]]'"); 2
          end

        Alembic.defined?(prog, "main") and not Alembic.defined?(prog, "main", 0) and Alembic.const(prog, "main") != nil ->
          print_value(Alembic.const(prog, "main"), o)
          0

        true ->
          consts = for d <- prog.defs, d.params == nil, do: {d.name, Alembic.const(prog, d.name)}
          if json?(o) do
            emit_json(Map.new(consts, fn {k, v} -> {k, Alembic.to_data(v)} end))
          else
            Enum.each(consts, fn {k, v} -> out("#{bold(k)} = #{Alembic.show(v)}") end)
            fns = for d <- prog.defs, d.params != nil, do: "#{d.name}(#{Enum.join(d.params, ", ")})"
            if fns != [], do: out(dim("functions: " <> Enum.join(fns, ", ")))
          end
          0
      end
    else
      {:error, m} -> err("alembic: " <> m); 3
    end
  end

  defp print_value(v, o) do
    if Keyword.get(o, :json, false) or not tty?(), do: emit_json(Alembic.to_data(v)), else: out(Alembic.show(v))
  end

  # a line-oriented session: definitions accumulate, expressions print
  defp repl(o) do
    out(dim("alembic — definitions (name = …, f(x) = …) are kept; expressions are evaluated. :card, :defs, :quit"))
    repl_loop("", o)
  end

  defp repl_loop(src, o) do
    case IO.gets("alembic> ") do
      :eof -> 0
      {:error, _} -> 0
      line ->
        line = String.trim(line)
        cond do
          line in [":quit", ":q", "quit", "exit"] -> 0
          line == ":card" -> out(Alembic.card()); repl_loop(src, o)
          line == ":defs" -> out(src); repl_loop(src, o)
          line == "" -> repl_loop(src, o)
          Regex.match?(~r/^[\p{L}_][\w']*\s*(\([^)]*\))?\s*=[^=>]/u, line) ->
            candidate = src <> line <> "\n"
            case Alembic.load(candidate) do
              {:ok, _} -> repl_loop(candidate, o)
              {:error, e} -> err(bad(Alembic.format_error(e))); repl_loop(src, o)
            end
          true ->
            prog = case Alembic.load(src) do {:ok, p} -> p; _ -> nil end
            case Alembic.eval(line, program: prog, fuel: o[:fuel] || Alembic.default_fuel()) do
              {:ok, v} -> out(good(Alembic.show(v)))
              {:error, e} -> err(bad(Alembic.format_error(e)))
            end
            repl_loop(src, o)
        end
    end
  end
end
