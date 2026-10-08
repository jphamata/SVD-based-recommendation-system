defmodule Vapor.Diwan do
  @moduledoc """
  The **dīwān** (ديوان, root د-و-ن *d-w-n*, "to record") — one command
  interpreter behind every terminal vapor has (docs/DIWAN.md): the TUI
  (`mix vapor.tui`), the console's terminal panel and `POST
  /v1/vapor/diwan`. The same line gives the same answer in all three,
  because they are the same function: `eval(line, session) → {result,
  session}`.

  A line is a small shell — no shell is run: `vapor` verbs (the word
  `vapor` is optional), pipes, redirection, quoting, and a few builtins:

      athanor run golomb.nbq | verify golomb.nbq -
      rebis equiv a.net b.net > verdict.json
      echo "x^2 - x + 1/4" > p.txt ; cat p.txt
      chat new --title notes ; chat search kulisch | head 5

  Every stage but the last writes JSON (the verbs see a pipe), the last one
  writes for people — exactly the Unix behaviour of `bin/vapor`.

  **Jailed** sessions (the console) cannot touch the server: files are
  the session's own (`ls`, `cat`, `rm`, `>` and the panel's editor and
  upload), a FILE argument names one of them, `--measure` (which runs a
  program) is refused, and every command runs in its own process with a
  heap ceiling and a deadline — a runaway command is killed, the session
  stays. A local session (the TUI) reads and writes the real files of the
  person at the keyboard, as any terminal does.
  """

  @builtins ~w(help ls cat echo rm cp mv head wc history clear)
  @max_files 128
  @max_file 8 * 1024 * 1024
  @max_total 64 * 1024 * 1024

  defstruct jail: true, files: %{}, history: [], majlis: nil, tty: true, timeout: 120_000, heap_mb: 1024, cwd: nil

  @doc "A session. Options: `jail:` (default true), `majlis:`, `tty:`, `timeout:` (ms per command), `heap_mb:`."
  def new(opts \\ []), do: struct!(__MODULE__, opts)

  @doc """
  Run one line. Returns `{%{out, err, code, codes}, session}`: the last
  stage's standard output (unless redirected), every stage's standard
  error, the last stage's exit status and all of them.
  """
  def eval(line, %__MODULE__{} = st) when is_binary(line) do
    st = %{st | history: Enum.take([line | st.history], 500)}

    case parse_line(line) do
      {:ok, pipelines} ->
        Enum.reduce(pipelines, {%{out: "", err: "", code: 0, codes: []}, st}, fn {stages, redirect}, {acc, st} ->
          {r, st} = run_pipeline(stages, redirect, st)
          {%{out: acc.out <> r.out, err: acc.err <> r.err, code: r.code, codes: acc.codes ++ r.codes}, st}
        end)

      {:error, why} ->
        {%{out: "", err: "diwan: #{why}\n", code: 2, codes: [2]}, st}
    end
  end

  @doc "A line → `{:ok, [{stages, redirect}]}`: pipelines separated by `;`."
  def parse_line(line) do
    with {:ok, toks} <- lex(line, [], "", false) do
      toks
      |> split_on(:seq)
      |> Enum.reject(&(&1 == []))
      |> Enum.reduce_while({:ok, []}, fn seg, {:ok, acc} ->
        case pipeline(seg) do
          {:ok, stages, red} -> {:cont, {:ok, acc ++ [{stages, red}]}}
          e -> {:halt, e}
        end
      end)
    end
  end

  # ------------------------------------------------------------------ parse

  @doc """
  A line → `{:ok, [stage], redirect}`: stages are argv lists (the first may
  carry `{:stdin, file}`), redirect is nil, `{:write, file}` or `{:append,
  file}`. Single and double quotes group words; a backslash escapes the
  next character.
  """
  def parse(line) do
    with {:ok, toks} <- lex(line, [], "", false), do: pipeline(toks)
  end

  defp pipeline(toks) do
    stages = split_on(toks, :pipe)

    Enum.reduce_while(Enum.with_index(stages), {:ok, [], nil}, fn {stage, i}, {:ok, acc, red} ->
      last = i == length(stages) - 1

      case take_redirects(stage, last, i == 0) do
        {:ok, argv, _stdin, _red} when argv == [] -> {:halt, {:error, "an empty command around |"}}
        {:ok, argv, stdin, r} -> {:cont, {:ok, acc ++ [%{argv: argv, stdin: stdin}], r || red}}
        {:error, why} -> {:halt, {:error, why}}
      end
    end)
    |> case do
      {:ok, [], _} -> {:error, "nothing to run"}
      other -> other
    end
  end

  defp lex("", acc, cur, quoted), do: {:ok, Enum.reverse(push(acc, cur, quoted))}
  defp lex("\\" <> <<c::utf8, rest::binary>>, acc, cur, _q), do: lex(rest, acc, cur <> <<c::utf8>>, true)
  defp lex("'" <> rest, acc, cur, _q) do
    case String.split(rest, "'", parts: 2) do
      [inside, more] -> lex(more, acc, cur <> inside, true)
      _ -> {:error, "an unclosed '"}
    end
  end

  defp lex("\"" <> rest, acc, cur, q), do: dquote(rest, acc, cur, q)
  defp lex(">>" <> rest, acc, cur, q), do: lex(rest, [:append | push(acc, cur, q)], "", false)
  defp lex(">" <> rest, acc, cur, q), do: lex(rest, [:write | push(acc, cur, q)], "", false)
  defp lex("<" <> rest, acc, cur, q), do: lex(rest, [:read | push(acc, cur, q)], "", false)
  defp lex("|" <> rest, acc, cur, q), do: lex(rest, [:pipe | push(acc, cur, q)], "", false)
  defp lex(";" <> rest, acc, cur, q), do: lex(rest, [:seq | push(acc, cur, q)], "", false)
  defp lex(<<c::utf8, rest::binary>>, acc, cur, q) when c in [?\s, ?\t, ?\n], do: lex(rest, push(acc, cur, q), "", false)
  defp lex(<<c::utf8, rest::binary>>, acc, cur, q), do: lex(rest, acc, cur <> <<c::utf8>>, q)

  defp dquote("", _acc, _cur, _q), do: {:error, "an unclosed \""}
  defp dquote("\\" <> <<c::utf8, rest::binary>>, acc, cur, q), do: dquote(rest, acc, cur <> <<c::utf8>>, q)
  defp dquote("\"" <> rest, acc, cur, _q), do: lex(rest, acc, cur, true)
  defp dquote(<<c::utf8, rest::binary>>, acc, cur, q), do: dquote(rest, acc, cur <> <<c::utf8>>, q)

  defp push(acc, "", false), do: acc
  defp push(acc, cur, _), do: [cur | acc]

  # split a token list on a separator, keeping empty segments ("| b" has one)
  defp split_on(toks, sep) do
    {segs, cur} = Enum.reduce(toks, {[], []}, fn t, {segs, cur} -> if t == sep, do: {[Enum.reverse(cur) | segs], []}, else: {segs, [t | cur]} end)
    Enum.reverse([Enum.reverse(cur) | segs])
  end

  defp take_redirects(toks, last, first) do
    Enum.reduce_while(toks, {:ok, [], nil, nil, nil}, fn
      op, {:ok, argv, stdin, red, nil} when op in [:write, :append, :read] -> {:cont, {:ok, argv, stdin, red, op}}
      t, {:ok, argv, _stdin, red, :read} when is_binary(t) -> if first, do: {:cont, {:ok, argv, t, red, nil}}, else: {:halt, {:error, "< only on the first command of a pipeline"}}
      t, {:ok, argv, stdin, _red, op} when op in [:write, :append] and is_binary(t) ->
        if last, do: {:cont, {:ok, argv, stdin, {op, t}, nil}}, else: {:halt, {:error, "> only after the last command of a pipeline"}}
      t, {:ok, argv, stdin, red, nil} when is_binary(t) -> {:cont, {:ok, argv ++ [t], stdin, red, nil}}
      _, _ -> {:halt, {:error, "a redirection without a file name"}}
    end)
    |> case do
      {:ok, _, _, _, op} when op != nil -> {:error, "a redirection without a file name"}
      {:ok, argv, stdin, red, nil} -> {:ok, argv, stdin, red}
      e -> e
    end
  end

  # -------------------------------------------------------------- pipeline

  defp run_pipeline(stages, redirect, st) do
    n = length(stages)

    first_in =
      case hd(stages).stdin do
        nil -> {:ok, ""}
        f -> read_file(st, f)
      end

    case first_in do
      {:error, why} ->
        {%{out: "", err: "diwan: #{why}\n", code: 1, codes: [1]}, st}

      {:ok, input} ->
        {out, errs, codes, st} =
          stages
          |> Enum.with_index()
          |> Enum.reduce({input, "", [], st}, fn {stage, i}, {inp, errs, codes, st} ->
            tty = st.tty and i == n - 1 and redirect == nil
            {out, err, code, st} = run_stage(stage.argv, inp, tty, st)
            {out, errs <> err, codes ++ [code], st}
          end)

        case redirect do
          nil ->
            {%{out: out, err: errs, code: List.last(codes), codes: codes}, st}

          {mode, file} ->
            case write_file(st, file, out, mode) do
              {:ok, st} -> {%{out: "", err: errs, code: List.last(codes), codes: codes}, st}
              {:error, why} -> {%{out: "", err: errs <> "diwan: #{why}\n", code: 1, codes: codes ++ [1]}, st}
            end
        end
    end
  end

  defp run_stage(["vapor" | argv], input, tty, st), do: run_stage(argv, input, tty, st)
  defp run_stage([b | args], input, tty, st) when b in @builtins, do: builtin(b, args, input, tty, st)
  defp run_stage(argv, input, tty, st), do: verb(argv, input, tty, st)

  # a vapor verb, in its own sealed process (heap and binaries capped, deadline); the
  # streams belong to the caller, so a killed verb leaves nothing behind
  defp verb(argv, input, tty, st) do
    {:ok, io} = StringIO.open(input)
    {:ok, eio} = StringIO.open("")

    run = fn ->
      Process.group_leader(self(), io)
      Process.put(:vapor_stderr, eio)
      Process.put(:vapor_tty, tty)
      if st.jail, do: Process.put(:vapor_jail, st.files)
      if st.majlis, do: Process.put(:vapor_majlis, st.majlis)

      try do
        Vapor.Main.run(Enum.map(argv, &Vapor.CLI.utf8_arg/1))
      rescue
        e -> IO.puts(eio, "vapor: internal error: " <> Exception.message(e)); 4
      end
    end

    limits = [heap_mb: st.heap_mb, timeout: st.timeout]
    result = Vapor.Hermetic.seal(run, limits)
    {:ok, {_, out}} = StringIO.close(io)
    {:ok, {_, err}} = StringIO.close(eio)

    case result do
      {:ok, code} -> {out, err, code, st}
      {:error, f} -> {"", "diwan: #{hd(argv)} #{Vapor.Hermetic.describe(f, limits)}\n", 4, st}
    end
  end

  # ---------------------------------------------------------------- builtins

  defp builtin("help", _args, _in, _tty, st) do
    text = """
    vapor's terminal — the same commands as `bin/vapor`, the same answers.
      <verb> …                every vapor command (`vapor` is optional): alembic, athanor, verify, game, crucible,
                              assay, mind, scene, solve, rebis, aludel, tabula, cupel, amalgam, chat, wzn (`help` for each:
                              run it without arguments)
      a | b                   b reads a's output (a writes JSON into a pipe)
      a > f   a >> f   a < f  write, append, read #{if st.jail, do: "this session's files", else: "files"}
      ls · cat F… · echo … · rm F · cp A B · mv A B · head [N] · wc · history · clear
    #{if st.jail, do: "This session is jailed: files are its own, nothing runs on the server but vapor itself.", else: ""}
    """

    {text, "", 0, st}
  end

  defp builtin("ls", _args, _in, _tty, %{jail: true} = st) do
    out = st.files |> Enum.sort() |> Enum.map_join("", fn {n, d} -> String.pad_leading(Integer.to_string(byte_size(d)), 9) <> "  " <> n <> "\n" end)
    {out, "", 0, st}
  end

  defp builtin("ls", args, _in, _tty, st) do
    dir = List.first(args) || "."

    case File.ls(dir) do
      {:ok, names} -> {Enum.map_join(Enum.sort(names), "", &(&1 <> "\n")), "", 0, st}
      {:error, e} -> {"", "ls: #{dir}: #{:file.format_error(e)}\n", 1, st}
    end
  end

  defp builtin("cat", [], input, _tty, st), do: {input, "", 0, st}

  defp builtin("cat", files, _in, _tty, st) do
    Enum.reduce(files, {"", "", 0, st}, fn f, {out, err, code, st} ->
      case read_file(st, f) do
        {:ok, d} -> {out <> d, err, code, st}
        {:error, why} -> {out, err <> "cat: #{why}\n", 1, st}
      end
    end)
  end

  defp builtin("echo", args, _in, _tty, st), do: {Enum.join(args, " ") <> "\n", "", 0, st}
  defp builtin("history", _args, _in, _tty, st), do: {st.history |> Enum.reverse() |> Enum.with_index(1) |> Enum.map_join("", fn {l, i} -> "#{String.pad_leading(Integer.to_string(i), 4)}  #{l}\n" end), "", 0, st}
  defp builtin("clear", _args, _in, _tty, st), do: {"", "", 0, st}

  defp builtin("rm", files, _in, _tty, %{jail: true} = st) do
    {missing, files} = Enum.split_with(files, &(not Map.has_key?(st.files, clean_path(&1))))
    st = %{st | files: Map.drop(st.files, Enum.map(files, &clean_path/1))}
    {"", Enum.map_join(missing, "", &"rm: #{&1}: no such file\n"), if(missing == [], do: 0, else: 1), st}
  end

  defp builtin(cmd, args, _in, _tty, %{jail: true} = st) when cmd in ["cp", "mv"] do
    case args do
      [a, b] ->
        with {:ok, d} <- read_file(st, a), {:ok, st} <- write_file(st, b, d, :write) do
          st = if cmd == "mv", do: %{st | files: Map.delete(st.files, clean_path(a))}, else: st
          {"", "", 0, st}
        else
          {:error, why} -> {"", "#{cmd}: #{why}\n", 1, st}
        end

      _ ->
        {"", "#{cmd}: give a source and a destination\n", 2, st}
    end
  end

  defp builtin(cmd, _args, _in, _tty, st) when cmd in ["rm", "cp", "mv"],
    do: {"", "#{cmd}: use your own shell for files outside a session\n", 2, st}

  defp builtin("head", args, input, _tty, st) do
    n = case args do [x | _] -> (case Integer.parse(String.trim_leading(x, "-")) do {k, ""} when k > 0 -> k; _ -> 10 end); _ -> 10 end
    {input |> String.split("\n") |> Enum.take(n) |> Enum.join("\n") |> then(&if(&1 == "" or String.ends_with?(&1, "\n"), do: &1, else: &1 <> "\n")), "", 0, st}
  end

  defp builtin("wc", _args, input, _tty, st) do
    lines = input |> String.split("\n", trim: true) |> length()
    words = input |> String.split(~r/\s+/u, trim: true) |> length()
    {"#{lines} #{words} #{byte_size(input)}\n", "", 0, st}
  end

  # ------------------------------------------------------------------ files

  @doc "A session path: no directories above it, no leading `./`, at most 200 characters."
  def clean_path(p) do
    p |> to_string() |> String.trim() |> String.trim_leading("./") |> String.replace(~r{/+}, "/") |> String.slice(0, 200)
  end

  defp valid_name?(n), do: n != "" and not String.contains?(n, "..") and not String.starts_with?(n, "/") and String.valid?(n) and not String.contains?(n, <<0>>)

  @doc "Read a session file (jailed) or a real file (local)."
  def read_file(%{jail: true} = st, f) do
    case Map.fetch(st.files, clean_path(f)) do
      {:ok, d} -> {:ok, d}
      :error -> {:error, "#{f}: no such file"}
    end
  end

  def read_file(_st, f) do
    case File.read(f) do
      {:ok, d} -> {:ok, d}
      {:error, e} -> {:error, "#{f}: #{:file.format_error(e)}"}
    end
  end

  @doc "Write (or append to) a session file (jailed) or a real file (local)."
  def write_file(%{jail: true} = st, f, data, mode) do
    name = clean_path(f)
    data = if mode == :append, do: Map.get(st.files, name, "") <> data, else: data
    total = st.files |> Map.delete(name) |> Map.values() |> Enum.map(&byte_size/1) |> Enum.sum()

    cond do
      not valid_name?(name) -> {:error, "#{inspect(f)}: not a file name here (no .., no leading /)"}
      byte_size(data) > @max_file -> {:error, "#{name}: a file may hold #{div(@max_file, 1024 * 1024)} MB in a session"}
      total + byte_size(data) > @max_total -> {:error, "the session's files may hold #{div(@max_total, 1024 * 1024)} MB together"}
      not Map.has_key?(st.files, name) and map_size(st.files) >= @max_files -> {:error, "a session holds at most #{@max_files} files"}
      true -> {:ok, %{st | files: Map.put(st.files, name, data)}}
    end
  end

  def write_file(st, f, data, mode) do
    result = if mode == :append, do: File.write(f, data, [:append]), else: File.write(f, data)

    case result do
      :ok -> {:ok, st}
      {:error, e} -> {:error, "#{f}: #{:file.format_error(e)}"}
    end
  end

  @doc "Every builtin's name (for completion)."
  def builtins, do: @builtins
end
