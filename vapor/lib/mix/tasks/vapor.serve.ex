defmodule Mix.Tasks.Vapor.Serve do
  @shortdoc "Serve a checkpoint over an OpenAI-compatible HTTP API, with the operator console at /"
  @moduledoc """
      mix vapor.serve --model PATH [--port 8000] [--ip 127.0.0.1] [--threads N] [--gpu] [--quantize sb4] [--storage bf16]
                      [--max-seq 512] [--sequences 8] [--page 16] [--step-tokens 64] [--replicas 1]
                      [--docs FILE_OR_DIR]... [--tlog PATH] [--tlog-origin NAME] [--data DIR]
      mix vapor.serve --docs ./pasta            # the console and the document library alone, no model

  Starts `Vapor.Engine` (continuous batching, paged KV; with `--replicas N`
  a pool of N engines sharing one compilation and the weight pages) and
  `Vapor.Serve`, then runs until interrupted. The operator console
  (`Vapor.Console`) is at `http://IP:PORT/`; `--docs` ingests files (every
  file of a directory, recursively) into its library at start. Search
  receipts are anchored in a transparency log (`Vapor.Tlog`): `--tlog PATH`
  keeps it in an append-only file (its signing key in `PATH.key`), otherwise
  it lives in memory for the run; the console's *Ledger* panel verifies it
  in the browser.
  """
  use Mix.Task

  @switches [model: :string, port: :integer, ip: :string, threads: :integer, gpu: :boolean, quantize: :string, storage: :string, max_seq: :integer, replicas: :integer,
             sequences: :integer, page: :integer, step_tokens: :integer, docs: :keep, token: :string, tlog: :string, data: :string,
             tlog_origin: :string]

  @impl true
  def run(argv) do
    Mix.Task.run("app.start")
    {o, _, _} = OptionParser.parse(argv, strict: @switches)
    dir = o[:model]
    docs = Keyword.get_values(o, :docs)
    # neither is required since 0.14: the open bench (Athanor, Crucible, Assay, scenes) needs no model and no library
    _ = {dir, docs}

    {e, tk} =
      if dir do
        tk =
          case Vapor.Model.tokenizer(dir) do
            {:ok, tk} -> tk
            {:error, r} -> Mix.raise(inspect(r))
          end

        {Vapor.CLI.engine(dir, tk, o), tk}
      else
        {nil, nil}
      end

    {:ok, holder} = Vapor.Console.Holder.start_link()

    for path <- Enum.flat_map(docs, &files/1) do
      Vapor.Console.Holder.update(holder, fn lib ->
        case Vapor.Docs.Library.add(lib, path) do
          {:ok, lib2, rep} -> IO.puts("vapor: #{path}: #{count(rep.added, "passage")}" <> warn(rep)); {:ok, lib2}
          {:error, r} -> IO.puts("vapor: #{path}: refused — #{r.bound}"); {:error, lib}
        end
      end)
    end

    {:ok, ip} = :inet.parse_address(String.to_charlist(o[:ip] || "127.0.0.1"))
    token = o[:token] || System.get_env("VAPOR_TOKEN")
    loopback = match?({127, _, _, _}, ip) or ip == {0, 0, 0, 0, 0, 0, 0, 1}

    if not loopback and token in [nil, ""],
      do: Mix.raise("--ip #{o[:ip]} exposes the models and the library beyond this machine: give --token SECRET (or VAPOR_TOKEN), and put TLS in front")
    name = if dir, do: Path.basename(Path.expand(dir)), else: "vapor"
    # receipts are anchored in a transparency log: a file with --tlog (and its
    # key beside it), else in memory for this run
    {:ok, tlog} = Vapor.Tlog.Holder.start_link(Enum.reject([path: o[:tlog], origin: o[:tlog_origin]], &is_nil(elem(&1, 1))))
    # conversations live here (crash-atomic store); --data DIR, $VAPOR_HOME, or ~/.vapor
    data = o[:data] || System.get_env("VAPOR_HOME") || Path.join(System.user_home!(), ".vapor")
    {:ok, srv} = Vapor.Serve.start_link(engine: e, tokenizer: tk, port: o[:port] || 8000, ip: ip, model_name: name, library: holder, data: data,
                                        token: token, tlog: tlog)
    base = "http://#{o[:ip] || "127.0.0.1"}:#{Vapor.Serve.port(srv)}"
    IO.puts("vapor: #{if dir, do: "serving #{dir}", else: "no model"}; API #{base}/v1, console #{base}/" <> if(token, do: "?token=…", else: ""))
    Process.sleep(:infinity)
  end

  defp files(path) do
    if File.dir?(path), do: Path.wildcard(Path.join(path, "**/*")) |> Enum.filter(&File.regular?/1) |> Enum.sort(), else: [path]
  end

  defp warn(%{warnings: []}), do: ""
  defp warn(%{warnings: ws}), do: " (#{count(length(ws), "warning")}: #{Enum.join(Enum.take(ws, 2), "; ")})"
  defp warn(_), do: ""

  defp count(1, word), do: "1 #{word}"
  defp count(n, word), do: "#{n} #{word}s"
end
