defmodule Mix.Tasks.Vapor.Tui do
  @shortdoc "The console in the terminal: OCR, speech, drawing, search, quality, fusion"
  @moduledoc """
      mix vapor.tui [--lang en|pt] [--docs PATH]…

  A line-oriented session over the same readers as the web console
  (`Vapor.TUI`); `help` lists the commands. Colour only on a terminal and
  never with `NO_COLOR`. `--docs` ingests files into the session's library
  before the prompt. In a pipe (`printf 'read scan.pdf\\n' | mix vapor.tui`)
  there is no prompt and no banner: only the answers.
  """
  use Mix.Task

  @impl true
  def run(argv) do
    {o, _, _} = OptionParser.parse(argv, strict: [lang: :string, docs: :keep])
    Mix.Task.run("app.start")
    st = Vapor.TUI.new(lang: if(o[:lang] == "pt", do: :pt, else: :en))

    st =
      Enum.reduce(Keyword.get_values(o, :docs), st, fn path, st ->
        {out, st} = Vapor.TUI.eval("add " <> path, st)
        IO.puts(out)
        st
      end)

    if st.tty, do: IO.puts(Vapor.TUI.eval("help", st) |> elem(0))
    Vapor.TUI.loop(st)
  end
end
