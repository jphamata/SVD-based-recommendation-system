defmodule Vapor.Cupel.Sentinel do
  @moduledoc """
  A supervised guard over a pool of substrates computing `x·Wᵀ` for one
  weight matrix: every result passes the cupel before it is returned; a
  substrate whose result **cannot** have come from correct arithmetic is
  quarantined on the evidence, the product is recomputed on the next
  healthy one (or by the exact oracle), and the event enters a journal
  chained by SHA-256 and closed by a Merkle root — the record an operator
  takes to the hardware vendor.

  The work runs in the caller's process (callers are concurrent); the
  server only holds the probe, the pool's health and the journal. A
  worker that dies is a `{:worker_crashed, label, reason}` entry, not an
  outage. Workers are named by their position in the pool (`"w0"`, …), so
  the journal is canonical bytes, independent of pids.

  `runner` (`(worker, x) → {:ok, y} | {:error, why}`) defaults to the
  native worker; tests and drills pass one that corrupts — silicon cannot
  be made faulty on request, so faults are injected at the boundary where
  a faulty core would emit them.
  """
  use GenServer
  alias Vapor.{Canonical, Cupel, Merkle, Program, Tensor}
  alias Vapor.Algebra.Term, as: T

  # ------------------------------------------------------------------ API

  @doc """
  Start a sentinel. Options: `w:` (the weight matrix, required),
  `workers:` (pids; `[]` = oracle only), `seed:` (the probe's secret),
  `runner:`, `name:`.
  """
  def start_link(opts) do
    {gen, opts} = Keyword.split(opts, [:name])
    GenServer.start_link(__MODULE__, opts, gen)
  end

  @doc """
  `x·Wᵀ`, checked. `{:ok, y, %{by: label | "oracle", attempts}}`. Every
  quarantine or crash on the way is in the journal.
  """
  def linear(s, %Tensor{} = x) do
    %{probe: p, runner: run, pool: pool, w: w} = GenServer.call(s, :lease)
    attempt(s, p, run, w, x, pool, [])
  end

  @doc "Labels of the healthy workers."
  def healthy(s), do: GenServer.call(s, :healthy)

  @doc "`%{label => reason}` of the workers out of the pool (quarantined or crashed)."
  def removed(s), do: GenServer.call(s, :removed)

  @doc "The journal, oldest first: `%{seq, kind, worker, detail, hash}` (hash hex)."
  def journal(s), do: GenServer.call(s, :journal)

  @doc "Merkle root (RFC 6962 leaves) over the journal's canonical entries."
  def merkle_root(s), do: s |> journal() |> Enum.map(&Merkle.leaf(Canonical.encode(Map.delete(&1, "hash")))) |> Merkle.root()

  # ----------------------------------------------------------- the caller

  defp attempt(s, p, _run, w, x, [], tried) do
    y = Cupel.oracle_linear(w, x)
    # the oracle is the definition: if it failed the cupel, the cupel is wrong
    {:ok, _} = Cupel.assay(p, x, y)
    GenServer.cast(s, {:served, "oracle", Enum.reverse(tried)})
    {:ok, y, %{by: "oracle", attempts: Enum.reverse(tried)}}
  end

  defp attempt(s, p, run, w, x, [{label, wk} | rest], tried) do
    outcome =
      try do
        run.(wk, x)
      catch
        :exit, why -> {:error, {:exit, why}}
      end

    case outcome do
      {:ok, %Tensor{} = y} ->
        case Cupel.assay(p, x, y) do
          {:ok, rep} ->
            GenServer.cast(s, {:served, label, Enum.reverse(tried)})
            {:ok, y, %{by: label, attempts: Enum.reverse(tried), worst: rep.worst}}

          {:corrupt, rep} ->
            GenServer.call(s, {:quarantine, label, %{"rows" => rep.corrupt, "worst" => ratio_text(rep.worst), "seed_digest" => seed_digest(p)}})
            attempt(s, p, run, w, x, rest, [{label, :corrupt} | tried])
        end

      {:error, why} ->
        GenServer.call(s, {:crashed, label, inspect(why) |> String.slice(0, 200)})
        attempt(s, p, run, w, x, rest, [{label, :failed} | tried])
    end
  end

  defp ratio_text(:infinity), do: "infinity"
  defp ratio_text(r), do: :erlang.float_to_binary(r * 1.0, [{:scientific, 6}])

  # the journal names the probe without revealing its seed
  defp seed_digest(p), do: Base.encode16(:crypto.hash(:sha256, <<p.seed::64>> <> p.digest), case: :lower) |> binary_part(0, 16)

  # ----------------------------------------------------------- the server

  @impl true
  def init(opts) do
    w = Keyword.fetch!(opts, :w)
    workers = Keyword.get(opts, :workers, [])
    labels = workers |> Enum.with_index() |> Enum.map(fn {wk, i} -> {"w#{i}", wk} end)
    for {_, wk} <- labels, is_pid(wk), do: Process.monitor(wk)
    p = Cupel.probe(w, seed: Keyword.get(opts, :seed, 1))

    st = %{w: w, probe: p, runner: Keyword.get(opts, :runner, &native_linear(w, &1, &2)), pool: labels, removed: %{}, journal: [], head: <<0::256>>, seq: 0}
    {:ok, log(st, "probe", "-", %{"n" => p.n, "k" => p.k, "dtype" => Atom.to_string(p.dtype), "weights_sha256" => Base.encode16(p.digest, case: :lower)})}
  end

  @impl true
  def handle_call(:lease, _from, st), do: {:reply, %{probe: st.probe, runner: st.runner, pool: st.pool, w: st.w}, st}
  def handle_call(:healthy, _from, st), do: {:reply, Enum.map(st.pool, &elem(&1, 0)), st}
  def handle_call(:removed, _from, st), do: {:reply, st.removed, st}

  def handle_call(:journal, _from, st),
    do: {:reply, st.journal |> Enum.reverse() |> Enum.map(fn e -> Map.update!(e, "hash", &Base.encode16(&1, case: :lower)) end), st}

  def handle_call({:quarantine, label, evidence}, _from, st), do: {:reply, :ok, remove(st, label, "quarantined", evidence)}
  def handle_call({:crashed, label, why}, _from, st), do: {:reply, :ok, remove(st, label, "worker_failed", %{"reason" => why})}

  @impl true
  def handle_cast({:served, label, tried}, st) do
    {:noreply, log(st, "served", label, %{"after" => Enum.map(tried, fn {l, why} -> [l, Atom.to_string(why)] end)})}
  end

  @impl true
  def handle_info({:DOWN, _ref, :process, pid, reason}, st) do
    case Enum.find(st.pool, fn {_, wk} -> wk == pid end) do
      {label, _} -> {:noreply, remove(st, label, "worker_crashed", %{"reason" => inspect(reason) |> String.slice(0, 200)})}
      nil -> {:noreply, st}
    end
  end

  defp remove(st, label, kind, detail) do
    if List.keymember?(st.pool, label, 0) do
      st = %{st | pool: List.keydelete(st.pool, label, 0), removed: Map.put(st.removed, label, kind)}
      log(st, kind, label, detail)
    else
      st
    end
  end

  defp log(st, kind, label, detail) do
    entry = %{"seq" => st.seq + 1, "kind" => kind, "worker" => label, "detail" => detail}
    h = :crypto.hash(:sha256, st.head <> Canonical.encode(entry))
    %{st | journal: [Map.put(entry, "hash", h) | st.journal], head: h, seq: st.seq + 1}
  end

  # ------------------------------------------------------ the native path

  @doc false
  def native_linear(%Tensor{shape: [_n, k]} = w, wk, %Tensor{shape: [b, k]} = x) do
    p = Program.new(y: T.linear(T.input(:x, :f32, [b, k]), T.const(w)))

    with {:ok, c} <- Vapor.Compile.Lower.lower(p),
         {:ok, r} <- Vapor.Runtime.Native.run(wk, c, %{x: x}, isa: Vapor.Runtime.Substrates.host_isa(), mode: :native) do
      {:ok, r.outputs.y}
    end
  end
end
