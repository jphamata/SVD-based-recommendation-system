# Run a Livebook notebook's Elixir cells in order, as Livebook would (one
# binding and environment threaded through the cells), skipping the
# Mix.install setup cell — the notebook is checked by executing it.
#
#     mix run scripts/run_livemd.exs notebooks/vapor_tour.livemd
[path] = System.argv()

cells =
  ~r/^```elixir\n(.*?)^```$/ms
  |> Regex.scan(File.read!(path), capture: :all_but_first)
  |> Enum.map(&hd/1)
  |> Enum.reject(&String.contains?(&1, "Mix.install("))

env = %{__ENV__ | file: Path.expand(path), line: 1}

Enum.reduce(Enum.with_index(cells, 1), {[], env}, fn {code, i}, {binding, env} ->
  IO.puts("── cell #{i}")
  quoted = Code.string_to_quoted!(code, file: env.file)
  {value, binding, env} = Code.eval_quoted_with_env(quoted, binding, env)
  IO.puts(inspect(value, pretty: true, limit: 12, printable_limit: 200))
  {binding, env}
end)

IO.puts("── #{length(cells)} cells ok")
