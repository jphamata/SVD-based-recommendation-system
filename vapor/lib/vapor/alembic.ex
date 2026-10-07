defmodule Vapor.Alembic do
  @moduledoc """
  **Alembic** — the small language in which a person (or a model) writes
  a problem for vapor: a verifier, an objective, a game, a simulation, a
  scene's motion (docs/ALEMBIC.md). The alembic is where a mixture is
  distilled; here it is where an open request becomes something a machine
  can check.

  Open, but sanitised — the properties that make arbitrary input safe to
  run on a server, by construction rather than by filtering:

    * **nothing reaches the host** — the parser builds tuples with a fixed
      set of atoms (identifiers stay strings), and the evaluator knows only
      its operators and `Vapor.Alembic.Builtins`: no files, no network, no
      processes, no `apply/3` on user names;
    * **everything halts** — calls and loop iterations pay fuel; recursion
      depth, list length and integer size are capped;
    * **memory is bounded** — `sandbox/2` runs in a process whose heap is
      capped by the VM (`max_heap_size` kills it), with a wall-clock limit;
    * **deterministic** — no clock, no randomness except `noise(…)`, a hash
      of its arguments; the same program and input give the same value on
      every machine.

      # a Golomb ruler: marks whose pairwise distances are all different
      marks = 5
      valid(r) = is_distinct([b - a for (a, b) in pairs(sort(r))])
      length(r) = max(r) - min(r)

  `load/2` compiles a program; `call/4` calls one of its functions;
  `eval/2` evaluates an expression (optionally against a program);
  `literal/1` reads a value back from its `show/1` text — the canonical,
  hashable form of every candidate vapor exchanges.
  """
  alias Vapor.Alembic.{Builtins, Compiler, Parser}

  defstruct source: "", defs: [], globals: %{}, hash: nil, names: [], reserved: %{}

  @type t :: %__MODULE__{}
  @default_fuel 2_000_000

  @doc "The default fuel of one call."
  def default_fuel, do: @default_fuel


  @doc "A compact reference card of the language (for people, and for models asked to write it)."
  def card do
    """
    ALEMBIC — reference card
    Program = definitions, one per line:  name = expr   |   name(a, b) = expr      (# comments)
    Values: integers (arbitrary size), floats, "strings", true/false/nil, [lists], (tuples,), {"maps": 1}
    Operators: + - * / (float)  // (floor div)  % (mod)  ^ or ** (power)  ++ (concat)
               == != < <= > >= (chainable: 0 <= i < n)   in, not in   and or not   & | xor << >> ~
    Ranges: a..b (inclusive)       Index: xs[i], xs[-1]   Slice: xs[a:b]   Field: m.key
    if c then a else b      let x = e1, y = e2 in body      x => expr     (a, b) => expr
    Comprehension: [f(x) for x in xs if p(x) for y in ys]   Pipe: xs |> map(f) |> sum
    Builtins: len sum prod mean var std median min max abs sign sqrt exp ln log log2 log10 sin cos tan
      atan2 hypot floor ceil round trunc int float gcd lcm isqrt is_prime factorial binom
      sort(xs[, key]) reverse range(n | a, b[, step]) map filter fold(xs, init, (acc, x) => …) scan all any
      count(xs[, f | v]) zip enumerate distinct is_distinct set union intersect difference flatten concat
      take drop first last append prepend set_at insert remove_at swap index_of find repeat
      argmin argmax min_by max_by chunks windows pairs combinations permutations product transpose
      dot norm cumsum diffs inversions is_sorted popcount bit bits(n, w) from_bits band bor bxor
      str chars join split upper lower keys values items get put has dict tuple list type
      hash noise(...) (deterministic float in [0,1)) error assert;  constants pi e tau
    Everything is pure and deterministic; loops cost fuel; there is no I/O.

    ATHANOR (search) — reserved names:
      space = bits(n) | ints(n, lo, hi) | reals(n, lo, hi) | perm(n) | perm(xs) | subset(xs, k) | subsets(xs)
            | seq(n, alphabet) | graph(n) | partition(n, k) | program(["x"], ["+","-","*","/","sin",…], [1, 2], max_size)
      minimize(x) = number   or   maximize(x) = number   or   claim(x) = bool (search for a counterexample)
      valid(x) = bool  ·  violation(x) = number ≥ 0 (0 = valid; guides the search)  ·  margin(x) (with claim)
      target = value · budget = evaluations · seed = n · start = [candidates] · show(x) = text
      describe(x) = [numbers] (diverse archive) · holdout(x) = number (unseen objective for the finalists)
      neighbor(x, s) = a custom local move · measured = true (objective measured outside)
      A program-space candidate reaches your functions as a function: f(x).
    GAMES — init = state · player(s) · moves(s) = [moves] · play(s, m) = next state
      winner(s) = nil while playing, 0 draw, else the winning player · show(s) · features(s) = [numbers]

    Example (search):
      space = subset(1..30, 6)
      ruler(r) = [0] ++ r
      violation(r) = let d = [b - a for (a, b) in pairs(ruler(r))] in len(d) - len(distinct(d))
      minimize(r) = max(r)
    """
  end

  # ============================================================ loading

  @doc """
  Compile a program and evaluate its constants. `{:ok, %Alembic{}}` or
  `{:error, %{message, line, col}}`. Options: `fuel:` for the constants
  (default 10⁷), `consts:` a map of names to values that **override** the
  program's own definitions (how a search binds `n = 8` from the outside).
  """
  def load(text, opts \\ []) when is_binary(text) do
    with {:ok, all_defs} <- Parser.program(text),
         :ok <- unique(all_defs) do
      overrides = Keyword.get(opts, :consts, %{})
      skip = Keyword.get(opts, :skip, [])
      {reserved, defs} = Enum.split_with(all_defs, fn {:def, n, ps, _, _} -> n in skip and ps == nil end)

      globals =
        Map.new(defs, fn
          {:def, name, nil, body, _pos} ->
            if Map.has_key?(overrides, name), do: {name, {:value, overrides[name]}}, else: {name, {:thunk, Compiler.compile(body)}}

          {:def, name, ps, body, _pos} ->
            cbody = Compiler.compile(body, MapSet.new(ps))
            arity = length(ps)
            {name, {:value, {:fn, name, arity, fn args -> cbody.(Map.new(Enum.zip(ps, args))) end}}}
        end)
        |> Map.merge(Map.new(Map.drop(overrides, Enum.map(defs, &elem(&1, 1))), fn {k, v} -> {k, {:value, v}} end))

      case run(globals, Keyword.get(opts, :fuel, 10_000_000), fn -> force_all(globals) end) do
        {:ok, forced, _used} ->
          {:ok, %__MODULE__{source: text, defs: Enum.map(defs, &def_info/1), globals: forced,
                            hash: Vapor.Canonical.hex_digest({:alembic, text, Enum.sort(Map.to_list(overrides)) |> Enum.map(fn {k, v} -> [k, Builtins.show(v)] end)}),
                            names: Enum.map(defs, &elem(&1, 1)),
                            reserved: Map.new(reserved, fn {:def, n, nil, body, {l, _}} -> {n, {body, l}} end)}}

        {:error, msg} ->
          {:error, %{message: msg, line: 0, col: 0}}
      end
    end
  rescue
    e in [Vapor.Alembic.Error] -> {:error, %{message: e.message, line: 0, col: 0}}
  end

  defp def_info({:def, name, ps, _body, {l, _}}), do: %{name: name, params: ps, line: l}

  defp unique(defs) do
    dup = defs |> Enum.map(&elem(&1, 1)) |> Enum.frequencies() |> Enum.find(fn {_, n} -> n > 1 end)
    case dup do
      nil -> :ok
      {n, _} ->
        {:def, _, _, _, {l, c}} = defs |> Enum.filter(&(elem(&1, 1) == n)) |> List.last()
        {:error, %{message: "#{n} is defined twice", line: l, col: c}}
    end
  end

  defp force_all(globals) do
    Enum.each(globals, fn {n, _} -> Compiler.global(n) end)
    Process.get(:alembic_globals)
  end

  @doc "Load or raise."
  def load!(text, opts \\ []) do
    case load(text, opts) do
      {:ok, m} -> m
      {:error, e} -> raise ArgumentError, "Alembic: #{format_error(e)}"
    end
  end

  @doc "An error as one line of text."
  def format_error(%{message: m, line: l, col: c}) when is_integer(l) and l > 0, do: "line #{l}, column #{c}: #{m}"
  def format_error(%{message: m}), do: m
  def format_error(m) when is_binary(m), do: m

  # ============================================================ running

  @doc "Is `name` defined by the program (as a function of `arity`, when given)?"
  def defined?(%__MODULE__{globals: g}, name, arity \\ nil) do
    case Map.get(g, name) do
      {:value, {:fn, _, a, _}} -> arity == nil or a == arity
      {:value, _} -> arity == nil
      _ -> false
    end
  end

  @doc "A constant's value, or nil."
  def const(%__MODULE__{globals: g}, name) do
    case Map.get(g, name) do
      {:value, {:fn, _, _, _}} -> nil
      {:value, v} -> v
      _ -> nil
    end
  end

  @doc """
  Call the program's function `name` with `args`. `{:ok, value}` or
  `{:error, message}`. Option `fuel:` (default #{@default_fuel}).
  """
  def call(%__MODULE__{globals: g}, name, args, opts \\ []) do
    case Map.get(g, name) do
      {:value, {:fn, _, _, _} = f} ->
        case run(g, Keyword.get(opts, :fuel, @default_fuel), fn -> Compiler.apply_fn(f, args, name, {0, 0}) end) do
          {:ok, v, _} -> {:ok, v}
          e -> e
        end

      nil -> {:error, "#{name} is not defined"}
      _ -> {:error, "#{name} is not a function"}
    end
  end

  @doc """
  Evaluate an expression, against a program when given. Options: `fuel:`,
  `bindings:` a map of names to values visible to the expression.
  """
  def eval(text, opts \\ []) do
    prog = Keyword.get(opts, :program)
    bindings = Keyword.get(opts, :bindings, %{})

    with {:ok, ast} <- Parser.expression(text) do
      c = Compiler.compile(ast, MapSet.new(Map.keys(bindings)))
      g = if prog, do: prog.globals, else: %{}
      case run(g, Keyword.get(opts, :fuel, @default_fuel), fn -> c.(bindings) end) do
        {:ok, v, _} -> {:ok, v}
        {:error, m} -> {:error, %{message: m, line: 0, col: 0}}
      end
    end
  end

  @doc "Evaluate a parsed expression tree in the program's context (used for reserved definitions)."
  def eval_ast(%__MODULE__{globals: g}, ast, opts \\ []) do
    c = Compiler.compile(ast)
    case run(g, Keyword.get(opts, :fuel, @default_fuel), fn -> c.(%{}) end) do
      {:ok, v, _} -> {:ok, v}
      e -> e
    end
  end

  defp run(globals, fuel, fun) do
    saved = {Process.get(:alembic_globals), Process.get(:alembic_fuel), Process.get(:alembic_rdepth)}
    Process.put(:alembic_globals, globals)
    Process.put(:alembic_fuel, fuel)
    Process.put(:alembic_rdepth, 0)

    try do
      v = fun.()
      {:ok, v, fuel - Process.get(:alembic_fuel)}
    catch
      {:alembic, msg} -> {:error, msg}
      :error, %ArithmeticError{} -> {:error, "arithmetic error (overflow or an undefined operation)"}
      :error, :badarith -> {:error, "arithmetic error"}
      :error, %{__exception__: true} = e -> {:error, "evaluation error: " <> String.slice(Exception.message(e), 0, 200)}
      :error, :system_limit -> {:error, "a value exceeded a system limit"}
    after
      {g, f, d} = saved
      restore(:alembic_globals, g)
      restore(:alembic_fuel, f)
      restore(:alembic_rdepth, d)
    end
  end

  defp restore(k, nil), do: Process.delete(k)
  defp restore(k, v), do: Process.put(k, v)

  # ============================================================ values

  @doc "A value as Alembic source text — the canonical form."
  defdelegate show(v), to: Builtins

  @doc "A value as JSON-ready data."
  defdelegate to_data(v), to: Builtins

  @doc """
  Read a value written as a literal (numbers, strings, booleans, nil,
  lists, tuples, maps, ranges of integers, and arithmetic on numbers).
  Nothing is called: this is how candidates typed by a person or proposed
  by a model enter a search. `{:ok, v}` or `{:error, message}`.
  """
  def literal(text) when is_binary(text) do
    with {:ok, ast} <- Parser.expression(text),
         :ok <- literal_only(ast) do
      case eval(text, fuel: 100_000) do
        {:ok, v} -> {:ok, v}
        {:error, e} -> {:error, format_error(e)}
      end
    else
      {:error, %{} = e} -> {:error, format_error(e)}
      {:error, m} -> {:error, m}
    end
  end

  def literal(_), do: {:error, "not text"}

  defp literal_only({:lit, _}), do: :ok
  defp literal_only({k, items, _}) when k in [:list, :tuple], do: all_lit(items)
  defp literal_only({:map, pairs, _}), do: all_lit(Enum.flat_map(pairs, fn {k, v} -> [k, v] end))
  defp literal_only({:neg, a, _}), do: literal_only(a)
  defp literal_only({:range, a, b, _}), do: all_lit([a, b])
  defp literal_only({:bin, op, a, b, _}) when op in ["+", "-", "*", "/", "^"], do: all_lit([a, b])
  defp literal_only({:var, n, _}) when n in ["pi", "e", "tau", "π"], do: :ok
  defp literal_only(_), do: {:error, "only literal values are accepted here (numbers, strings, lists, tuples, maps)"}

  defp all_lit(items), do: Enum.reduce_while(items, :ok, fn i, :ok -> case literal_only(i) do :ok -> {:cont, :ok}; e -> {:halt, e} end end)

  @doc "Convert JSON data (from a client or a model) to an Alembic value; arrays become lists."
  def from_data(v) when is_list(v), do: Enum.map(v, &from_data/1)
  def from_data(v) when is_map(v), do: Map.new(v, fn {k, x} -> {k, from_data(x)} end)
  def from_data(v), do: v

  # ============================================================ sandbox

  @doc """
  Run `fun` in a fresh process whose heap the VM caps (`heap_mb:`, default
  256) and that is killed after `timeout:` ms (default 30 000). Returns
  `{:ok, result}`, `{:error, :memory}`, `{:error, :timeout}` or
  `{:error, {:crash, reason}}` — the caller survives any of them.
  """
  def sandbox(fun, opts \\ []) do
    words = div(Keyword.get(opts, :heap_mb, 256) * 1_048_576, :erlang.system_info(:wordsize))
    timeout = Keyword.get(opts, :timeout, 30_000)
    parent = self()
    ref = make_ref()

    {pid, mon} =
      spawn_monitor(fn ->
        Process.flag(:max_heap_size, %{size: words, kill: true, error_logger: false})
        send(parent, {ref, fun.()})
      end)

    receive do
      {^ref, result} ->
        Process.demonitor(mon, [:flush])
        {:ok, result}

      {:DOWN, ^mon, :process, ^pid, :killed} ->
        {:error, :memory}

      {:DOWN, ^mon, :process, ^pid, reason} ->
        {:error, {:crash, reason}}
    after
      timeout ->
        Process.exit(pid, :kill)
        Process.demonitor(mon, [:flush])
        receive do
          {^ref, _} -> :ok
        after
          0 -> :ok
        end
        {:error, :timeout}
    end
  end
end

defmodule Vapor.Alembic.Error do
  defexception message: "Alembic error"
end
