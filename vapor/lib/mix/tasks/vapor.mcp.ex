defmodule Mix.Tasks.Vapor.Mcp do
  @shortdoc "Serve vapor's studio and verifiable retrieval over MCP (stdio)"
  @moduledoc """
      mix vapor.mcp [--dir DIR] [--out DIR]

  A Model Context Protocol server on stdin/stdout (`Vapor.MCP.Server`).
  `--dir` is the studio directory (files are read only inside it; default
  the current directory), `--out` where outputs are written (default
  `DIR/vapor-out`). Logs go to stderr; stdout carries only the protocol.

  A client configuration (Claude Desktop, an IDE, any MCP host):

      {"mcpServers": {"vapor": {"command": "mix", "args": ["vapor.mcp", "--dir", "/path/to/work"],
                                "cwd": "/path/to/vapor"}}}
  """
  use Mix.Task

  @impl true
  def run(argv) do
    {o, _, _} = OptionParser.parse(argv, strict: [dir: :string, out: :string])
    # nothing but the protocol on stdout
    Logger.configure_backend(:console, device: :standard_error)
    Mix.shell(Mix.Shell.Quiet)
    Mix.Task.run("app.start", ["--no-compile"])
    w = if Vapor.Runtime.Substrates.binary("vapor-worker", "native"), do: Vapor.Vision.OCR.worker()
    Vapor.MCP.Server.serve(Vapor.MCP.Server.new(dir: o[:dir] || File.cwd!(), out: o[:out], worker: w))
  end
end
