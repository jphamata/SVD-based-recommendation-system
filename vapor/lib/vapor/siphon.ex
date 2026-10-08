defmodule Vapor.Siphon do
  @moduledoc """
  **The siphon** — the network airlock. A siphon draws from outside through
  one tube, in one direction, and only when someone opens the tap.

  vapor's core opens no connection to fetch anything. Bytes from outside
  (a checkpoint on Hugging Face, an object in S3, a file on your own
  cluster, a peer of your private network) come in through a **fetcher**:
  a program the person declares — a Python script using the official
  `huggingface_hub`, `aws s3 cp`, `rsync`, `curl`, anything — in
  `$VAPOR_HOME/siphons.json` (default `~/.vapor`):

      {"fetchers": [
        {"name": "hf", "argv": ["python3", "/home/me/fetch_hf.py", "{ref}", "{out}"], "env": ["HF_TOKEN"]},
        {"name": "hf-range", "argv": ["curl", "-sfL", "-r", "{range}", "-o", "{out}/part", "https://huggingface.co/{ref}"],
         "env": ["HTTPS_PROXY"], "max_bytes": 200000000}
      ]}

  `{ref}` is what to fetch, `{out}` the empty directory it writes into,
  `{range}` (`a-b`, inclusive) a byte range, for fetchers that can.

  **Who opens the tap.** A fetch runs only from the person's own surface:
  `vapor siphon run`, `vapor siphon approve`. An agent behind MCP has one
  tool here, `siphon_propose`: it appends a request (fetcher, ref, why) to
  a queue, and nothing else happens until the person approves it (a Majlis
  conversation has not even that: its allowlist keeps conversations off the
  filesystem). No tool that fetches is exposed to an agent, and the test
  suite holds that line.

  **What a fetch can do.** It runs as its own OS process (never inside
  the BEAM), from an empty directory, with an environment reduced to
  `PATH`, `HOME`, `LANG` and the names its declaration lists (a token is
  passed only to the fetcher that declares it). It is killed at its
  deadline (`timeout_s`, 3600) or as soon as it has written more than
  `max_bytes` (64 GiB). A ref is one argument, never a shell word, and a
  ref that starts with `-` is refused (it could be read as an option).

  **What comes in.** Every file the fetcher wrote passes its format
  airlock (`.safetensors` headers and tiling, `.gguf` magic, `.json`
  syntax), is hashed (SHA-256), is checked against a pinned digest when
  one is given (`--sha256`, or `pins` in the declaration), and lands with
  a **receipt** (`RECEIPT.json`): which fetcher, which ref, the exact
  argv, the digests, the times, who proposed and who approved. A refused
  fetch leaves nothing behind.

  **Headers before data.** `headers/3` fetches only the first bytes of a
  remote `.safetensors` file (its header), and `Vapor.Lock.preflight/2`
  admits a checkpoint from its configuration and tensor table alone: whether
  vapor can run a 1.5 TB model is known before one weight is fetched.
  """
  alias Vapor.{JSON, Rejection}
  alias Vapor.Ingest.Safetensors

  defmodule Fetcher do
    @moduledoc "A fetcher the person declared."
    defstruct [:name, :argv, env: [], max_bytes: 64 * 1024 * 1024 * 1024, timeout_s: 3600, pins: %{}]
  end

  @base_env ~w(PATH HOME LANG LC_ALL TMPDIR)

  # ------------------------------------------------------------ declarations --

  @doc "The state directory: `opts[:home]`, `$VAPOR_HOME`, or `~/.vapor`."
  def home(opts \\ []), do: opts[:home] || System.get_env("VAPOR_HOME") || Path.join(System.user_home!(), ".vapor")

  @doc "The declared fetchers (`siphons.json` under `home/1`, or `opts[:fetchers]`, a decoded document)."
  def fetchers(opts \\ []) do
    doc =
      case opts[:fetchers] do
        nil ->
          path = Path.join(home(opts), "siphons.json")

          case File.read(path) do
            {:ok, bin} -> JSON.decode(bin)
            {:error, :enoent} -> {:ok, %{"fetchers" => []}}
            {:error, why} -> {:error, Rejection.new({:siphon, path}, "a readable siphons.json (#{inspect(why)})", "check the file")}
          end

        d ->
          {:ok, d}
      end

    with {:ok, %{"fetchers" => list}} when is_list(list) <- doc do
      Enum.reduce_while(list, {:ok, []}, fn d, {:ok, acc} ->
        case fetcher(d) do
          {:ok, f} -> {:cont, {:ok, acc ++ [f]}}
          err -> {:halt, err}
        end
      end)
    else
      {:ok, _} -> {:error, Rejection.new({:siphon, :declarations}, ~s({"fetchers": [...]}), "see docs/SIPHON.md")}
      err -> err
    end
  end

  defp fetcher(%{"name" => n, "argv" => [_ | _] = argv} = d) when is_binary(n) do
    cond do
      not Regex.match?(~r/^[a-z0-9_-]{1,40}$/, n) -> no(n, "a name of lowercase letters, digits, - and _")
      not Enum.all?(argv, &is_binary/1) -> no(n, "argv: a list of strings")
      not Enum.any?(argv, &String.contains?(&1, "{out}")) -> no(n, "argv names {out}, the directory it writes into")
      not Enum.all?(Map.get(d, "env", []), &(is_binary(&1) and Regex.match?(~r/^[A-Za-z_][A-Za-z0-9_]*$/, &1))) -> no(n, "env: names of variables")
      not pos?(Map.get(d, "max_bytes", 1)) or not pos?(Map.get(d, "timeout_s", 1)) -> no(n, "max_bytes and timeout_s: positive integers")
      true ->
        {:ok, %Fetcher{name: n, argv: argv, env: Map.get(d, "env", []), max_bytes: Map.get(d, "max_bytes", 64 * 1024 * 1024 * 1024),
                       timeout_s: Map.get(d, "timeout_s", 3600), pins: Map.get(d, "pins", %{})}}
    end
  end

  defp fetcher(d), do: no(inspect(d, limit: 3), "a declaration {name, argv}")

  defp pos?(x), do: is_integer(x) and x > 0

  defp no(n, b), do: {:error, Rejection.new({:siphon, n}, b, "fix the declaration in siphons.json")}

  @doc "The fetcher named `name`."
  def find(name, opts \\ []) do
    with {:ok, fs} <- fetchers(opts) do
      case Enum.find(fs, &(&1.name == name)) do
        nil -> {:error, Rejection.new({:siphon, name}, "a declared fetcher (declared: #{fs |> Enum.map(& &1.name) |> Enum.join(", ")})", "declare it in siphons.json")}
        f -> {:ok, f}
      end
    end
  end

  # ---------------------------------------------------------------- the queue --

  @doc """
  An agent's request: appended to the queue, nothing fetched. `{:ok,
  proposal}` (its `id` is what the person approves).
  """
  def propose(name, ref, why, opts \\ []) do
    with {:ok, _} <- find(name, opts),
         :ok <- check_ref(ref) do
      p = %{"fetcher" => name, "ref" => ref, "why" => to_string(why), "by" => to_string(opts[:by] || "an agent"),
            "at" => DateTime.utc_now() |> DateTime.truncate(:second) |> DateTime.to_iso8601()}
      p = Map.put(p, "id", :crypto.hash(:sha256, :erlang.term_to_binary(p)) |> Base.encode16(case: :lower) |> binary_part(0, 12))
      :ok = write_queue(opts, queue(opts) ++ [p])
      {:ok, p}
    end
  end

  @doc "The pending requests, oldest first."
  def queue(opts \\ []) do
    case File.read(queue_path(opts)) do
      {:ok, bin} -> case JSON.decode(bin) do {:ok, l} when is_list(l) -> l; _ -> [] end
      _ -> []
    end
  end

  @doc "Drop a request without fetching."
  def reject(id, opts \\ []), do: write_queue(opts, Enum.reject(queue(opts), &(&1["id"] == id)))

  @doc """
  The person approves a request: it is fetched (`run/3`) and leaves the
  queue. The receipt names the proposer and the approver.
  """
  def approve(id, opts \\ []) do
    case Enum.find(queue(opts), &(&1["id"] == id)) do
      nil ->
        {:error, Rejection.new({:siphon, id}, "a pending request", "vapor siphon queue lists them")}

      p ->
        with {:ok, f} <- find(p["fetcher"], opts),
             {:ok, r} <- run(f, p["ref"], Keyword.merge(opts, proposal: p)) do
          :ok = write_queue(opts, Enum.reject(queue(opts), &(&1["id"] == id)))
          {:ok, r}
        end
    end
  end

  defp queue_path(opts), do: Path.join([home(opts), "siphon", "queue.json"])

  defp write_queue(opts, list) do
    path = queue_path(opts)
    File.mkdir_p!(Path.dirname(path))
    tmp = path <> ".#{System.unique_integer([:positive])}"
    File.write!(tmp, JSON.encode(list))
    File.rename!(tmp, path)
  end

  defp check_ref(ref) when is_binary(ref) do
    cond do
      ref == "" -> {:error, Rejection.new({:siphon, :ref}, "a non-empty ref", "name what to fetch")}
      String.starts_with?(ref, "-") -> {:error, Rejection.new({:siphon, :ref}, "a ref that does not start with - (it could be read as an option)", "name what to fetch")}
      String.contains?(ref, ["\n", "\r", <<0>>]) -> {:error, Rejection.new({:siphon, :ref}, "a ref of one line", "name what to fetch")}
      byte_size(ref) > 2048 -> {:error, Rejection.new({:siphon, :ref}, "a ref of at most 2048 bytes", "name what to fetch")}
      true -> :ok
    end
  end

  defp check_ref(_), do: {:error, Rejection.new({:siphon, :ref}, "a string", "name what to fetch")}

  # ---------------------------------------------------------------- fetching --

  @doc """
  Fetch `ref` with a fetcher, now (the person's act). Options: `into:`
  (the destination directory; default `home/siphon/store/<digest>`),
  `sha256:` (the expected digest of the only file, or a map `relative path
  => digest`), `range:` (`"a-b"`, for a fetcher whose argv names
  `{range}`; the result then stays temporary and is returned as bytes,
  see `headers/3`), `proposal:` (set by `approve/2`).
  `{:ok, receipt}` or the rejection that says why nothing landed.
  """
  def run(%Fetcher{} = f, ref, opts \\ []) do
    dir = Path.join(System.tmp_dir!(), "vapor-siphon-#{System.unique_integer([:positive])}")
    out = Path.join(dir, "out")

    try do
      with :ok <- check_ref(ref),
           {:ok, exe, args} <- command(f, ref, out, opts[:range]),
           :ok <- File.mkdir_p(out),
           started = now(),
           {:ok, log} <- exec(f, exe, args, dir, out),
           {:ok, files} <- inventory(f, out),
           :ok <- pinned(f, ref, files, opts[:sha256]) do
        receipt = %{"fetcher" => f.name, "ref" => ref, "argv" => [exe | args] |> Enum.map(&String.replace(&1, out, "{out}")),
                    "started" => started, "finished" => now(), "files" => files, "log_tail" => log,
                    "proposal" => opts[:proposal], "approved_by" => System.get_env("USER") || "the person at the terminal"}

        if opts[:range], do: {:ok, Map.put(receipt, "bytes", File.read!(Path.join(out, hd(files)["path"])))}, else: land(receipt, out, opts)
      end
    after
      File.rm_rf(dir)
    end
  end

  defp now, do: DateTime.utc_now() |> DateTime.truncate(:second) |> DateTime.to_iso8601()

  defp command(f, ref, out, range) do
    cond do
      range != nil and not Enum.any?(f.argv, &String.contains?(&1, "{range}")) ->
        {:error, Rejection.new({:siphon, f.name}, "a fetcher whose argv names {range}", "declare a ranged fetcher (curl -r {range}, an HTTP Range request)")}

      range == nil and Enum.any?(f.argv, &String.contains?(&1, "{range}")) ->
        {:error, Rejection.new({:siphon, f.name}, "a byte range for a ranged fetcher", "use vapor siphon headers")}

      true ->
        [cmd | args] = Enum.map(f.argv, &(&1 |> String.replace("{ref}", ref) |> String.replace("{out}", out) |> String.replace("{range}", range || "")))

        case if(String.contains?(cmd, "/"), do: (File.regular?(cmd) && cmd) || nil, else: System.find_executable(cmd)) do
          nil -> {:error, Rejection.new({:siphon, f.name}, "an executable #{cmd}", "install it or name it by its path")}
          exe -> {:ok, exe, args}
        end
    end
  end

  # the fetcher's process: an empty directory, a reduced environment, a deadline, a byte cap. A shell
  # wrapper runs it in its own session (setsid, where present) and ends its whole process group, so a
  # fetcher's children go with it: when the deadline passes (a watchdog in the wrapper), or when this
  # side closes the port (the byte cap), which ends the wrapper's standard input (kept as fd 3: an
  # asynchronous list's own standard input is /dev/null). The arguments reach
  # the fetcher as "$@", never through the shell's parser. Nothing here shells out from the BEAM.
  @wrapper ~S"""
  exec 3<&0
  if command -v setsid >/dev/null 2>&1; then setsid "$@" </dev/null & c=$!; g=-$c; else "$@" </dev/null & c=$!; g=$c; fi
  ( sleep "$VAPOR_SIPHON_SECS" & s=$!; trap "kill $s 2>/dev/null; exit 0" TERM; wait $s; kill -9 $g 2>/dev/null ) >/dev/null 2>&1 & w=$!
  ( cat <&3 >/dev/null; kill -9 $g 2>/dev/null ) >/dev/null 2>&1 & r=$!
  wait $c 2>/dev/null; s=$?
  kill $w $r 2>/dev/null
  exit $s
  """

  defp exec(f, exe, args, dir, out) do
    keep = @base_env ++ f.env
    env = for {k, v} <- System.get_env(), do: if(k in keep, do: {String.to_charlist(k), String.to_charlist(v)}, else: {String.to_charlist(k), false})
    env = [{~c"VAPOR_SIPHON_SECS", Integer.to_charlist(f.timeout_s)} | env]

    case System.find_executable("sh") do
      nil ->
        {:error, Rejection.new({:siphon, f.name}, "a POSIX shell to run the fetcher under a deadline", "install sh")}

      sh ->
        port = Port.open({:spawn_executable, sh}, [:binary, :exit_status, :stderr_to_stdout, :hide, args: ["-c", @wrapper, "sh", exe | args], cd: dir, env: env])
        watch(port, f, out, System.monotonic_time(:millisecond) + f.timeout_s * 1000, "")
    end
  end

  defp watch(port, f, out, deadline, log) do
    receive do
      {^port, {:data, d}} ->
        watch(port, f, out, deadline, tail(log <> d))

      {^port, {:exit_status, 0}} ->
        if bytes(out) > f.max_bytes, do: {:error, refused(f, "wrote more than max_bytes (#{f.max_bytes})", log)}, else: {:ok, log}

      {^port, {:exit_status, s}} ->
        if System.monotonic_time(:millisecond) >= deadline,
          do: {:error, refused(f, "ran past timeout_s (#{f.timeout_s} s) and was killed", log)},
          else: {:error, refused(f, "exited with status #{s}", log)}
    after
      200 ->
        cond do
          # the wrapper's watchdog ends a fetcher at its deadline; this is the second barrier
          System.monotonic_time(:millisecond) > deadline + 2_000 -> kill(port); {:error, refused(f, "ran past timeout_s (#{f.timeout_s} s) and was killed", log)}
          bytes(out) > f.max_bytes -> kill(port); {:error, refused(f, "wrote more than max_bytes (#{f.max_bytes}) and was killed", log)}
          true -> watch(port, f, out, deadline, log)
        end
    end
  end

  defp tail(log) when byte_size(log) > 4096, do: binary_part(log, byte_size(log) - 4096, 4096)
  defp tail(log), do: log

  # closing the port ends the wrapper's standard input, and the wrapper ends the fetcher's process group
  defp kill(port) do
    try do
      Port.close(port)
    rescue
      ArgumentError -> :ok
    end

    receive do
      {^port, {:exit_status, _}} -> :ok
    after
      0 -> :ok
    end
  end

  defp refused(f, why, log), do: Rejection.new({:siphon, f.name}, "a fetch that finishes (#{why})", "nothing landed; the fetcher said: " <> String.slice(log, -400..-1//1))

  defp bytes(out), do: out |> files() |> Enum.reduce(0, fn p, acc -> acc + File.stat!(p).size end)

  defp files(out), do: out |> Path.join("**") |> Path.wildcard(match_dot: true) |> Enum.filter(&File.regular?/1) |> Enum.sort()

  # every file: its digest, and its format airlock
  defp inventory(f, out) do
    case files(out) do
      [] ->
        {:error, Rejection.new({:siphon, f.name}, "a fetch that writes at least one file into {out}", "check the fetcher")}

      paths ->
        Enum.reduce_while(paths, {:ok, []}, fn p, {:ok, acc} ->
          rel = Path.relative_to(p, out)

          case format(p) do
            {:ok, kind} -> {:cont, {:ok, acc ++ [%{"path" => rel, "bytes" => File.stat!(p).size, "sha256" => sha256(p), "format" => kind}]}}
            {:error, why} -> {:halt, {:error, Rejection.new({:siphon, rel}, why, "nothing landed: the fetched file failed its format airlock")}}
          end
        end)
    end
  end

  defp format(p) do
    case Path.extname(p) do
      ".safetensors" ->
        case Safetensors.index(p) do
          {:ok, _} -> {:ok, "safetensors"}
          {:error, r} -> {:error, "a well-formed safetensors file (#{r.bound})"}
        end

      ".gguf" ->
        case File.open(p, [:read, :binary], &IO.binread(&1, 4)) do
          {:ok, "GGUF"} -> {:ok, "gguf"}
          _ -> {:error, "a GGUF file (magic GGUF)"}
        end

      ".json" ->
        case File.stat!(p).size <= 100 * 1024 * 1024 and match?({:ok, _}, JSON.decode(File.read!(p))) do
          true -> {:ok, "json"}
          false -> {:error, "well-formed JSON of at most 100 MiB"}
        end

      _ ->
        {:ok, "unchecked"}
    end
  end

  defp sha256(p) do
    File.stream!(p, 1024 * 1024) |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1)) |> :crypto.hash_final() |> Base.encode16(case: :lower)
  end

  defp pinned(f, ref, files, given) do
    want =
      case given || Map.get(f.pins, ref) do
        nil -> %{}
        hex when is_binary(hex) and length(files) == 1 -> %{hd(files)["path"] => hex}
        hex when is_binary(hex) -> %{:only => hex}
        m when is_map(m) -> m
      end

    case Enum.find(want, fn {path, hex} -> Enum.find(files, &(&1["path"] == path))["sha256"] != String.downcase(hex) end) do
      nil -> :ok
      {path, hex} -> {:error, Rejection.new({:siphon, path}, "SHA-256 #{hex}", "nothing landed: the fetched bytes are not the pinned ones")}
    end
  end

  defp land(receipt, out, opts) do
    id = :crypto.hash(:sha256, JSON.encode(Map.delete(receipt, "log_tail"))) |> Base.encode16(case: :lower) |> binary_part(0, 16)
    dest = opts[:into] || Path.join([home(opts), "siphon", "store", id])
    receipt = Map.merge(receipt, %{"id" => id, "into" => dest})

    for %{"path" => rel} <- receipt["files"] do
      to = Path.join(dest, rel)
      File.mkdir_p!(Path.dirname(to))
      if File.rename(Path.join(out, rel), to) != :ok, do: File.cp!(Path.join(out, rel), to)
    end

    File.write!(Path.join(dest, "RECEIPT.json"), JSON.encode(receipt))
    {:ok, receipt}
  end

  # ----------------------------------------------------------------- headers --

  @doc """
  The tensor table of a remote `.safetensors` file from its first bytes
  alone, by a ranged fetcher: `{:ok, %{name => {dtype, shape}}, data_len}`.
  Two ranged fetches: the eight-byte length, then the header.
  """
  def headers(%Fetcher{} = f, ref, opts \\ []) do
    with {:ok, %{"bytes" => <<hlen::64-little, _::binary>>}} <- run(f, ref, Keyword.put(opts, :range, "0-7")),
         :ok <- (if hlen <= 100 * 1024 * 1024, do: :ok, else: {:error, Rejection.new({:siphon, ref}, "a header of at most 100 MiB", "check the ref")}),
         {:ok, %{"bytes" => bin}} <- run(f, ref, Keyword.put(opts, :range, "0-#{8 + hlen - 1}")),
         {:ok, h} <- Safetensors.parse_header(bin) do
      {:ok, Map.new(h.entries, &{&1.name, {&1.dtype, &1.shape}}), h.data_len}
    else
      {:ok, %{"bytes" => _}} -> {:error, Rejection.new({:siphon, ref}, "the first 8 bytes of a safetensors file", "check the fetcher's range handling")}
      {:more, n} -> {:error, Rejection.new({:siphon, ref}, "#{n} bytes of header", "check the fetcher's range handling")}
      err -> err
    end
  end
end
