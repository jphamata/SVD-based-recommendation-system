defmodule Vapor.Agent.Store do
  @moduledoc """
  Durable runs: where journals live between a crash and a resume.

  A store keeps each run's events in order, write-ahead: `hook/1` gives the
  `on_event:` function for `Vapor.Agent.run/3`, so every event is persisted
  before the run takes its next step (before the next tool executes). After
  a crash — of the BEAM, the machine, the region — `resume/4` loads the
  journal and continues: the prefix is replayed (local decisions recomputed,
  nothing in the world repeated) and the run goes on live. `act` tools
  receive the same idempotency key on every attempt, so a side effect whose
  event did not reach the store before the crash is applied at most once by
  a receiver that honours the key.

  That is the contract of a durable-workflow engine (Temporal, Oban
  workflows, AWS Step Functions) obtained from two facts vapor already has —
  determinism and a hash-chained journal — rather than from a scheduler.

  ## Contract

    * `append(store, run_id, event)` persists event `seq` *exclusively*:
      if an event with that `seq` already exists, it fails with
      `{:error, :conflict}`. Two processes resuming the same run cannot both
      write event *n*; since every `act` call is announced (`intent`, or
      `retry` after a crash) before it runs, only one of them announces each
      action — the other stops at its first write. An action announced and
      still in flight cannot be told from one interrupted by a crash, so a
      resumer arriving then retries it, with the same idempotency key
      (closing that needs a lease). (A relational
      adapter gets this from `UNIQUE (run_id, seq)`; see
      docs/ECOSSISTEMA_ELIXIR.md.)
    * `load(store, run_id)` returns the journal — `{:ok, journal}`, or
      `{:error, :not_found}` — and only events that verify; a torn last
      write is not an event.
    * `runs(store)` lists run ids.

  `Vapor.Agent.Store.File` implements it on a POSIX filesystem with OTP
  alone. Other backends (Ecto/Postgres, S3, a log) implement the behaviour.
  """
  alias Vapor.Agent.Journal

  @callback append(term, binary, map) :: :ok | {:error, term}
  @callback load(term, binary) :: {:ok, Journal.t()} | {:error, term}
  @callback runs(term) :: [binary]

  def append(%mod{} = s, run_id, event), do: mod.append(s, run_id, event)
  def load(%mod{} = s, run_id), do: mod.load(s, run_id)
  def runs(%mod{} = s), do: mod.runs(s)

  @doc """
  The journal of `run_id` from stored events (maps, or their canonical
  bytes), in order: the longest prefix whose hashes chain from the run id.
  For adapters — a torn or forged event ends the prefix instead of failing.
  """
  def rebuild(run_id, events) do
    events
    |> Enum.map(fn
      bin when is_binary(bin) -> Vapor.Canonical.decode(bin)
      %{} = e -> {:ok, e}
    end)
    |> then(&chain(Journal.new(run_id), &1))
  end

  defp chain(j, [{:ok, %{"seq" => seq, "kind" => k, "data" => d, "hash" => h}} | rest]) do
    if seq == length(j.events) do
      j2 = Journal.append(j, k, d)
      if j2.head == h, do: chain(j2, rest), else: j
    else
      j
    end
  end

  defp chain(j, _), do: j

  @doc "The `on_event:` hook that persists every new event (write-ahead)."
  def hook(store), do: fn event, journal -> append(store, journal.run_id, event) end

  @doc """
  Runs that have not ended (no `final` or `halt` event): what a supervisor
  resumes at boot.
  """
  def unfinished(store) do
    # a run whose start never reached the store did not begin: nothing to resume
    for id <- runs(store), {:ok, %{events: [_ | _] = evs}} <- [load(store, id)], List.last(evs)["kind"] not in ["final", "halt"], do: id
  end

  @doc """
  Start a run whose events are persisted as they happen. Options as
  `Vapor.Agent.run/3`; an `on_event:` given is called after the store's.
  """
  def run(store, spec, input, opts) do
    Vapor.Agent.run(spec, input, Keyword.update(opts, :on_event, [hook(store)], &[hook(store) | List.wrap(&1)]))
  end

  @doc """
  Resume a run from the store: replay what was recorded, continue live,
  persist what is new. A finished run returns its result without acting.
  """
  def resume(store, spec, run_id, opts) do
    with {:ok, j} <- load(store, run_id) do
      run(store, spec, nil, Keyword.put(opts, :journal, j))
    end
  end
end

defmodule Vapor.Agent.Store.File do
  @moduledoc """
  A journal store on a POSIX filesystem: a directory per run, a file per
  event (`000042.cbor`, the event in canonical CBOR).

  Each event is written to a temporary file, flushed to disk, and then
  *hard-linked* to its final name. `link(2)` is atomic and fails if the
  name exists, which gives both properties of the contract at once: a crash
  mid-write leaves only a temporary file (never a torn event), and two
  writers of the same `seq` cannot both succeed (`{:error, :conflict}`).
  `load/2` reads the events in order and keeps the longest prefix whose
  hashes chain from the run id.

  Durability of the directory entry itself needs a directory `fsync`, which
  OTP's file API does not expose; on Linux ext4/xfs with default options,
  link creation after a data `fsync` is ordered in practice, but a
  power-loss guarantee needs a filesystem mounted with `dirsync` (or a
  database).
  """
  @behaviour Vapor.Agent.Store
  alias Vapor.Canonical

  defstruct [:dir]

  def new(dir) do
    File.mkdir_p!(dir)
    %__MODULE__{dir: dir}
  end

  @impl true
  def append(%__MODULE__{dir: dir}, run_id, %{"seq" => seq} = event) do
    rdir = Path.join(dir, safe!(run_id))
    File.mkdir_p!(rdir)
    final = Path.join(rdir, name(seq))
    # unique across nodes sharing the filesystem, and never truncating another's
    tag = Base.encode16(:crypto.strong_rand_bytes(8), case: :lower)
    tmp = Path.join(rdir, ".#{name(seq)}.#{tag}.tmp")

    try do
      {:ok, f} = :file.open(tmp, [:write, :exclusive, :binary, :raw])
      :ok = :file.write(f, Canonical.encode(event))
      :ok = :file.datasync(f)
      :ok = :file.close(f)

      case :file.make_link(tmp, final) do
        :ok -> :ok
        {:error, :eexist} -> {:error, :conflict}
        {:error, why} -> {:error, why}
      end
    after
      File.rm(tmp)
    end
  end

  @impl true
  def load(%__MODULE__{dir: dir}, run_id) do
    rdir = Path.join(dir, safe!(run_id))

    case File.ls(rdir) do
      {:ok, names} ->
        events =
          names
          |> Enum.filter(&String.ends_with?(&1, ".cbor"))
          |> Enum.sort()
          |> Enum.map(&File.read!(Path.join(rdir, &1)))

        {:ok, Vapor.Agent.Store.rebuild(run_id, events)}

      {:error, :enoent} ->
        {:error, :not_found}
    end
  end

  @impl true
  def runs(%__MODULE__{dir: dir}) do
    case File.ls(dir) do
      {:ok, names} -> Enum.filter(names, &File.dir?(Path.join(dir, &1))) |> Enum.sort()
      _ -> []
    end
  end

  defp name(seq), do: String.pad_leading(Integer.to_string(seq), 9, "0") <> ".cbor"

  # run ids are hex digests; refuse anything that could escape the directory
  defp safe!(run_id) do
    if run_id =~ ~r/\A[0-9a-f]{16,128}\z/, do: run_id, else: raise(ArgumentError, "not a run id: #{inspect(run_id)}")
  end
end
