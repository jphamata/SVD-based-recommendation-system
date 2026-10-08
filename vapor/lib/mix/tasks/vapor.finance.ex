defmodule Mix.Tasks.Vapor.Finance do
  @shortdoc "Run a finance or trading-desk task from a text file (docs/FINANCE.md)"
  @moduledoc """
      mix vapor.finance KIND FILE [--json OUT]

  KIND is one of `arbitrage backtest book calendar curve exchange mc micro
  options portfolio risk`; FILE holds the text the desk reads (`-` for
  standard input). Prints the verdict and the certificate; `--json` writes
  the whole result. Exits 1 when the task's own certificate fails (an
  instrument not repriced, a backtest that fails a gate, a journal the
  naive engine does not reproduce), so a pipeline can gate on it.

      mix vapor.finance curve curva_di.txt
      mix vapor.finance backtest estrategia.txt --json resultado.json
      mix vapor.finance book ordens.txt
  """
  use Mix.Task

  @impl true
  def run(argv) do
    {o, args, _} = OptionParser.parse(argv, strict: [json: :string])
    Mix.Task.run("app.start")
    case args do
      [kind, file] ->
        text = if file == "-", do: IO.read(:stdio, :eof), else: File.read!(file)
        case Vapor.Finance.run(kind, text) do
          {:ok, r} ->
            if o[:json], do: File.write!(o[:json], Vapor.JSON.encode(Vapor.Console.Lab12.slim(r)))
            {ok, line} = summary(kind, r)
            Mix.shell().info(line)
            unless ok, do: exit({:shutdown, 1})
          {:error, why} -> Mix.raise(to_string(why))
        end
      _ -> Mix.raise("usage: mix vapor.finance KIND FILE [--json OUT] — KIND: #{Enum.join(Vapor.Finance.kinds(), ", ")}")
    end
  end

  defp summary("curve", r), do: {r.certificate.repriced, "curve: #{length(r.nodes)} nodes · #{r.certificate.verdict}"}
  defp summary("backtest", r), do: {Enum.all?(r.gates, & &1.pass), "backtest: #{r.verdict}\n" <> Enum.map_join(r.gates, "\n", &"  #{if &1.pass, do: "✓", else: "✗"} #{&1.gate} — #{&1.detail}")}
  defp summary("book", r), do: {r.check.ok, "book: #{r.events} events, #{r.trades} trades · #{r.check.verdict}\n  head #{r.head}\n  merkle #{r.merkle_root}"}
  defp summary("exchange", r), do: {r.certificate.book_check.ok and r.certificate.pre_trade.ok, "exchange: #{r.events} events, #{r.trades} trades · head #{r.head}"}
  defp summary("arbitrage", r), do: {get_in(r, [:certificate, :checked_exactly]) != false, "arbitrage: #{inspect(r.arbitrage)} — #{r.why}"}
  defp summary("mc", r), do: {r.certificate.oracle_parity != false, "mc: European #{r.european.price} ± #{r.european.stderr} (closed form #{r.european.closed_form}) · oracle parity #{inspect(r.certificate.oracle_parity)}"}
  defp summary("risk", r), do: {true, "risk: #{r.backtest.exceptions} exceptions (expected #{Float.round(r.backtest.expected, 1)}) · #{r.backtest.verdict} · zone #{r.backtest.zone}"}
  defp summary(kind, r), do: {true, "#{kind}: " <> (r |> Vapor.Console.Lab12.slim() |> Vapor.JSON.encode() |> String.slice(0, 400))}
end
