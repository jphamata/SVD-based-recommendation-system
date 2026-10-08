defmodule Vapor.Mind do
  @moduledoc """
  The language-model layer (docs/CLI.md, "VAPOR_MIND"; docs/MAJLIS.md): where vapor asks a model for
  help — and never takes its word for anything.

    * `formalize/3` — a problem in words becomes an Alembic program; the
      program is parsed and loaded, and on error the error goes back to the
      model (up to three repairs). A second, independent call reads the
      program back into words (`back_translation`), so the person compares
      what they asked with what will run **before** it runs.
    * `propose/3` — inside an Athanor run, the model sees the problem and
      the best candidates with their scores and proposes new ones; each is
      parsed as a literal and checked against the space, then scored by the
      same verifier as every other strategy. A model that proposes nonsense
      just loses the bandit's budget.
    * `ask/3` — a plain answer, for the terminal.

  Backends (`from_env/0`, `parse/1`): `anthropic:MODEL` (ANTHROPIC_API_KEY),
  `openai:MODEL[@BASE_URL]` (OPENAI_API_KEY; any OpenAI-compatible server,
  vapor's own included), and `script:FILE` — a file of canned answers, one
  block per call separated by lines `---`, for offline use and tests.
  Every call is recorded (`transcript`) with the prompt's hash.
  """
  alias Vapor.Alembic
  alias Vapor.Athanor.Space

  defstruct backend: nil, name: "", transcript: nil

  # ============================================================ backends

  @doc "The model configured by `VAPOR_MIND`, or nil."
  def from_env do
    case System.get_env("VAPOR_MIND") do
      nil -> nil
      "" -> nil
      spec -> case parse(spec) do {:ok, m} -> m; _ -> nil end
    end
  end

  @doc "A model from a spec string (`anthropic:…`, `openai:…`, `script:FILE`)."
  def parse("anthropic:" <> model),
    do: {:ok, new(%Vapor.Agent.Backend.Anthropic{model: model, api_key: System.get_env("ANTHROPIC_API_KEY")}, "anthropic:" <> model)}

  def parse("openai:" <> rest) do
    {model, base} = case String.split(rest, "@", parts: 2) do [m, b] -> {m, b}; [m] -> {m, "https://api.openai.com/v1"} end
    {:ok, new(%Vapor.Agent.Backend.OpenAI{model: model, base_url: base, api_key: System.get_env("OPENAI_API_KEY")}, "openai:" <> model)}
  end

  def parse("script:" <> file) do
    case File.read(file) do
      {:ok, text} -> {:ok, script(String.split(text, ~r/^---\s*$/m) |> Enum.map(&String.trim/1))}
      {:error, e} -> {:error, "script #{file}: #{e}"}
    end
  end

  def parse(other), do: {:error, "unknown model #{inspect(other)} — use anthropic:MODEL, openai:MODEL[@URL] or script:FILE"}

  @doc "A scripted model: answers in order (then repeats the last) — or a function of the prompt."
  def script(answers) when is_list(answers), do: new(%__MODULE__.Script{answers: answers, agent: start_counter()}, "script")
  def script(fun) when is_function(fun, 1), do: new(%__MODULE__.Script{fun: fun}, "script")

  defp start_counter do
    {:ok, a} = Agent.start(fn -> 0 end)
    a
  end

  @doc "Wrap any `Vapor.Agent.Backend` (a served local model, say) as a mind."
  def wrap(backend, name), do: new(backend, name)

  defp new(backend, name) do
    {:ok, t} = Agent.start(fn -> [] end)
    %__MODULE__{backend: backend, name: name, transcript: t}
  end

  @doc "Every call made so far: `[%{purpose, prompt_sha256, response}]`."
  def transcript(%__MODULE__{transcript: t}), do: Agent.get(t, &Enum.reverse/1)

  @doc "One completion: `{:ok, text}` or `{:error, why}`."
  def complete(%__MODULE__{} = m, system, user, purpose \\ "ask", opts \\ []) do
    messages = [%{"role" => "system", "content" => system}, %{"role" => "user", "content" => user}]
    gen = %{"max_tokens" => Keyword.get(opts, :max_tokens, 2000), "temperature" => Keyword.get(opts, :temperature, 0.4), "seed" => Keyword.get(opts, :seed, 1)}

    result =
      try do
        Vapor.Agent.Backend.complete(m.backend, messages, [], gen)
      rescue
        e -> {:error, Exception.message(e)}
      end

    case result do
      {:ok, %{content: text}} ->
        Agent.update(m.transcript, &[%{purpose: purpose, prompt_sha256: Vapor.Canonical.hex_digest(messages), response: text} | &1])
        {:ok, text}
      {:error, why} -> {:error, "the model did not answer: #{inspect(why) |> String.slice(0, 300)}"}
    end
  end

  # ============================================================ formalize

  @doc """
  Turn a problem stated in words into an Alembic program that loads (and,
  for searches, parses as an Athanor spec). `{:ok, %{program, kind,
  attempts, back_translation, notes}}` or `{:error, why, attempts}`.
  `kind`: `:auto` (search, game or claim — the model decides), `:search`,
  `:game`.
  """
  def formalize(%__MODULE__{} = m, words, kind \\ :auto) do
    system = """
    You translate problems into Alembic, a small, safe language that vapor runs.
    Answer with ONE Alembic program in a ```alembic block, nothing else after it.

    #{Alembic.card()}
    """

    user = "Problem (#{kind}):\n#{words}\n\nWrite the program. Prefer exact, checkable definitions; add a `show(x)` when it helps a person read a candidate."
    attempt(m, system, user, words, 1, [])
  end

  defp attempt(_m, _system, _user, _words, n, log) when n > 3, do: {:error, "the model's program still did not load after 3 attempts", Enum.reverse(log)}

  defp attempt(m, system, user, words, n, log) do
    with {:ok, reply} <- complete(m, system, user, "formalize") do
      prog = extract_code(reply)
      case check_program(prog) do
        {:ok, kind, notes} ->
          back = back_translate(m, prog)
          {:ok, %{program: prog, kind: kind, attempts: n, back_translation: back, notes: notes, log: Enum.reverse([%{attempt: n, ok: true} | log])}}

        {:error, why} ->
          user2 = user <> "\n\nYour previous program:\n```alembic\n#{prog}\n```\nfailed: #{why}\nFix it and answer with the whole corrected program."
          attempt(m, system, user2, words, n + 1, [%{attempt: n, ok: false, error: why} | log])
      end
    else
      {:error, why} -> {:error, why, Enum.reverse(log)}
    end
  end

  @doc "What kind of program a text is, if it loads: `{:ok, :search | :game | :program, notes}`."
  def check_program(prog) do
    cond do
      Vapor.Athanor.Game.game?(prog) ->
        case Vapor.Athanor.Game.load(prog) do
          {:ok, _} -> {:ok, :game, []}
          {:error, e} -> {:error, e}
        end

      prog =~ ~r/^\s*space\s*=/m ->
        case Vapor.Athanor.Spec.parse(prog) do
          {:ok, s} -> {:ok, :search, s.notes}
          {:error, e} -> {:error, e}
        end

      true ->
        case Alembic.load(prog) do
          {:ok, _} -> {:ok, :program, []}
          {:error, e} -> {:error, Alembic.format_error(e)}
        end
    end
  end

  defp back_translate(m, prog) do
    case complete(m, "You read Alembic programs and say, in two or three plain sentences, exactly what problem the program poses: the candidates, the objective or the rule, the constraints. Do not guess intent beyond the code.", "```alembic\n#{prog}\n```", "back_translation", temperature: 0.0) do
      {:ok, t} -> String.trim(t)
      {:error, _} -> nil
    end
  end

  @doc "The code inside the first fenced block of a reply (or the whole reply)."
  def extract_code(reply) do
    case Regex.run(~r/```(?:alembic|nbq|text)?\s*\n(.*?)```/s, reply) do
      [_, code] -> String.trim(code)
      nil -> String.trim(reply)
    end
  end

  # ============================================================ propose

  @doc "Candidates proposed for an Athanor run context (each checked against the space)."
  def propose(nil, _ctx, _n), do: {:error, "no model"}

  def propose(%__MODULE__{} = m, ctx, n) do
    spec = ctx.spec
    space = spec.space
    best = ctx.archive |> Enum.filter(&(&1.status == :ok)) |> Enum.take(8)
    lines = Enum.map_join(best, "\n", fn e -> "#{e.key}    → #{Alembic.show(e.value)}" end)
    goal = Vapor.Athanor.Spec.describe(spec)
    form = if space.kind == :program, do: "an expression in infix over #{Enum.join(space.vars, ", ")} using #{Enum.join(space.ops, " ")}", else: "an Alembic literal (lists in [ ], tuples in ( ))"

    system = "You propose candidates for a search. Each candidate goes on its own line, written as #{form}. No commentary, no numbering — just the #{n} lines."
    user = """
    Problem: #{goal}
    ```alembic
    #{String.slice(spec.source, 0, 6000)}
    ```
    Best candidates so far (candidate → objective):
    #{if lines == "", do: "(none yet)", else: lines}

    Propose #{n} new candidates that might do better. Vary them: some small changes of the best, some different ideas.
    """

    with {:ok, reply} <- complete(m, system, user, "propose", temperature: 0.8) do
      xs =
        reply
        |> String.split("\n")
        |> Enum.map(&String.trim/1)
        |> Enum.map(&String.replace(&1, ~r/^[-*\d.)\s]+(?=[\[\(\{\-\d"a-z])/u, ""))
        |> Enum.reject(&(&1 == "" or String.starts_with?(&1, "```")))
        |> Enum.map(fn line -> parse_candidate(space, line) end)
        |> Enum.filter(&match?({:ok, _}, &1))
        |> Enum.map(&elem(&1, 1))
        |> Enum.take(n)
      {:ok, xs}
    end
  end

  defp parse_candidate(%{kind: :program} = s, line), do: Space.parse_program(s, line)
  defp parse_candidate(s, line), do: with({:ok, v} <- Alembic.literal(line), do: Space.check(s, v))

  # ============================================================ ask

  @doc "A plain answer."
  def ask(m, question, context \\ nil) do
    system = "You are vapor's assistant in a terminal. Answer plainly and briefly. When the user's problem can be checked by computation, say which vapor command would check it (vapor athanor, vapor crucible, vapor assay, vapor alembic)."
    user = if context, do: "Context:\n#{context}\n\nQuestion: #{question}", else: question
    complete(m, system, user, "ask")
  end

end
