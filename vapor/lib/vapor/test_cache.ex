defmodule Vapor.TestCache do
  @moduledoc """
  A **content-addressed test cache** (`mix vapor.test`): a test file that
  passed, with everything it can reach unchanged, is not run again.

  A test file's key is the SHA-256 of:

  * the file itself, every file under `test/support`, `test/python`,
    `test/js` and `test/fixtures`, and `test/test_helper.exs`;
  * the compiled bytecode of every vapor module the file can reach: the
    modules it names (aliases expanded), those named by the test support
    modules it uses and by `test_helper.exs`, then, transitively, every
    module named in their atom tables. A literal module reference anywhere is an
    edge, so the closure over-approximates what the file runs (a module
    built from a string at run time would escape it; `Module.concat` on
    computed names is not used in vapor);
  * every file under `priv/`, which includes the native workers, plus the
    OTP, Elixir and vapor versions and the set of excluded tiers (tooling
    that appears changes what runs).

  Change one module and only the test files that can reach it run again.
  Change a fixture, `priv/`, a support file or the toolchain and
  everything runs. The cache records a file only when **every** test in
  it passed. A failure, an invalid module or an excluded test leaves no
  record.

  What it is for: the loop of an author, a centaur or an agent, where a
  97-minute suite is the obstacle. What it is not: a release gate. The
  release runs `mix test`, which ignores the cache. A flaky test that
  passed once stays cached until something it reaches changes, and
  `--all` forgets the cache.
  """

  @doc "Where the cache lives (`_build/<env>/vapor_test_cache.json`)."
  def path, do: Path.join(Mix.Project.build_path(), "vapor_test_cache.json")

  @doc "The cache: `%{key => %{file, tests, at}}`."
  def load do
    case File.read(path()) do
      {:ok, bin} -> case Vapor.JSON.decode(bin) do {:ok, m} when is_map(m) -> m; _ -> %{} end
      _ -> %{}
    end
  end

  @doc "Record test files that passed in full: `[{file, key, tests}]`."
  def record(entries) do
    now = DateTime.utc_now() |> DateTime.truncate(:second) |> DateTime.to_iso8601()
    cache = Enum.reduce(entries, load(), fn {file, key, n}, acc -> Map.put(acc, key, %{"file" => file, "tests" => n, "at" => now}) end)
    File.mkdir_p!(Path.dirname(path()))
    tmp = path() <> ".#{System.unique_integer([:positive])}"
    File.write!(tmp, Vapor.JSON.encode(cache))
    File.rename!(tmp, path())
  end

  @doc """
  The key of every test file: `%{file => key}`. `root` is the project's
  root; `excluded` the excluded tags (they change what a file runs).
  """
  def keys(files, root, excluded) do
    graph = module_graph()
    shared = shared_digest(root, excluded)
    beams = Map.new(graph, fn {m, _} -> {m, beam_digest(m)} end)
    support = support(root)

    Map.new(files, fn f ->
      text = File.read!(Path.join(root, f))
      mods = closure(through_support(named_modules(text), support), graph)
      digest = :crypto.hash(:sha256, [shared, text | Enum.map(Enum.sort(mods), &beams[&1])])
      {f, Base.encode16(digest, case: :lower)}
    end)
  end

  @doc "The vapor modules a test text can reach (the closure its key covers), from the project at `root`."
  def reach(text, root \\ File.cwd!()), do: text |> named_modules() |> through_support(support(root)) |> closure(module_graph()) |> Enum.sort()

  # the test support modules (not part of the application): each one's name → the modules its source names;
  # `test_helper.exs` runs before every file, so what it names is reached by all
  defp support(root) do
    files = Path.wildcard(Path.join(root, "test/support/**/*.ex"))

    defined =
      for f <- files, text = File.read!(f), [_, m] <- Regex.scan(~r/defmodule\s+(Vapor(?:\.[A-Z][A-Za-z0-9_]*)+)/, text), into: %{},
        do: {Module.concat([m]), named_modules(text)}

    helper = Path.join(root, "test/test_helper.exs")
    Map.put(defined, :test_helper, if(File.regular?(helper), do: named_modules(File.read!(helper)), else: []))
  end

  # a support module a file names brings in what that module names (transitively, through support)
  defp through_support(start, support) do
    Stream.iterate({MapSet.new([:test_helper | start]), [:test_helper | start]}, fn {seen, frontier} ->
      next = frontier |> Enum.flat_map(&Map.get(support, &1, [])) |> Enum.reject(&MapSet.member?(seen, &1))
      {MapSet.union(seen, MapSet.new(next)), Enum.uniq(next)}
    end)
    |> Enum.find(fn {_, frontier} -> frontier == [] end)
    |> elem(0)
    |> MapSet.delete(:test_helper)
    |> MapSet.to_list()
  end

  # every vapor module and the vapor modules its atom table names
  defp module_graph do
    {:ok, mods} = :application.get_key(:vapor, :modules)
    set = MapSet.new(mods)

    Map.new(mods, fn m ->
      {:ok, {_, [atoms: atoms]}} = :beam_lib.chunks(:code.which(m), [:atoms])
      {m, for({_, a} <- atoms, MapSet.member?(set, a), a != m, do: a)}
    end)
  end

  defp beam_digest(m), do: :crypto.hash(:sha256, File.read!(:code.which(m)))

  defp closure(start, graph) do
    Stream.iterate({MapSet.new(start), start}, fn {seen, frontier} ->
      next = frontier |> Enum.flat_map(&Map.get(graph, &1, [])) |> Enum.reject(&MapSet.member?(seen, &1))
      {MapSet.union(seen, MapSet.new(next)), Enum.uniq(next)}
    end)
    |> Enum.find(fn {_, frontier} -> frontier == [] end)
    |> elem(0)
    |> Enum.filter(&Map.has_key?(graph, &1))
  end

  @doc "The vapor modules a source text names: `Vapor.X.Y`, and the members of `alias Vapor.{A, B}`."
  def named_modules(text) do
    full = Regex.scan(~r/\bVapor(?:\.[A-Z][A-Za-z0-9_]*)+/, text) |> Enum.map(&hd/1)

    grouped =
      Regex.scan(~r/alias\s+(Vapor(?:\.[A-Z][A-Za-z0-9_]*)*)\.\{([^}]*)\}/s, text)
      |> Enum.flat_map(fn [_, base, members] -> members |> String.split(",") |> Enum.map(&(base <> "." <> String.trim(&1))) end)

    (full ++ grouped)
    |> Enum.flat_map(&prefixes/1)
    |> Enum.uniq()
    |> Enum.map(&Module.concat([&1]))
  end

  # Vapor.A.B also names Vapor.A (a nested module is reached through its parent's alias)
  defp prefixes(name) do
    parts = String.split(name, ".")
    for n <- 2..length(parts)//1, do: Enum.join(Enum.take(parts, n), ".")
  end

  defp shared_digest(root, excluded) do
    globs = ["test/support/**", "test/python/**", "test/js/**", "test/fixtures/**", "test/test_helper.exs", "priv/**"]

    files =
      globs
      |> Enum.flat_map(&Path.wildcard(Path.join(root, &1), match_dot: false))
      |> Enum.filter(&File.regular?/1)
      |> Enum.sort()

    versions = [System.otp_release(), System.version(), to_string(Application.spec(:vapor, :vsn)), inspect(Enum.sort(excluded))]

    files
    |> Enum.reduce(:crypto.hash_init(:sha256), fn f, h ->
      h = :crypto.hash_update(h, Path.relative_to(f, root))
      File.stream!(f, 1024 * 1024) |> Enum.reduce(h, &:crypto.hash_update(&2, &1))
    end)
    |> :crypto.hash_update(Enum.join(versions, "\n"))
    |> :crypto.hash_final()
  end
end
