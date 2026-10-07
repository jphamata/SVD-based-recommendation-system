defmodule Vapor.Console.Lab do
  @moduledoc """
  The console's laboratories for the 0.10 round — each a function from a
  small request to a JSON-ready map, measured on the spot (no figure is
  canned): the substrate airlock's verdicts, a chaotic simulation run on
  two substrates, a digital twin catching a fault, a network and its
  nulls, the trained language model's receipt and its unbounded stream,
  and reading in other scripts (Arabic, cursive, CJK, formulas, charts).
  """
  alias Vapor.{Graph, Physics, Tensor}
  alias Vapor.Runtime.{Substrates, Worker}

  # a worker of its own per laboratory: a worker holds one session at a time
  defp worker(name) do
    key = {__MODULE__, :worker, name}

    case :persistent_term.get(key, nil) do
      pid when is_pid(pid) ->
        if Process.alive?(pid), do: pid, else: start(key)

      _ ->
        start(key)
    end
  end

  defp start(key) do
    case Substrates.binary("vapor-worker", "native") do
      nil ->
        nil

      bin ->
        {:ok, pid} = Worker.start_link(exec: [bin])
        Process.unlink(pid)
        :persistent_term.put(key, pid)
        pid
    end
  end

  # computed once; concurrent first requests wait for the one computing (a
  # worker holds one session: two at once would drop each other's)
  defp memo(key, f) do
    case :persistent_term.get({__MODULE__, :memo, key}, nil) do
      nil ->
        :global.trans({{__MODULE__, :lab}, self()}, fn ->
          case :persistent_term.get({__MODULE__, :memo, key}, nil) do
            nil -> v = f.(); :persistent_term.put({__MODULE__, :memo, key}, v); v
            v -> v
          end
        end, [node()], :infinity)

      v ->
        v
    end
  end

  # ------------------------------------------------------------- airlock --

  @doc "Every substrate present, admitted by measurement (memoized for the server's life)."
  def substrates do
    memo(:substrates, fn ->
      for s <- Substrates.list() do
        a =
          try do
            Vapor.Substrate.admit(s)
          rescue
            e -> {:error, Exception.message(e)}
          catch
            _, why -> {:error, inspect(why)}
          end

        case a do
          %Vapor.Substrate.Admission{} = a ->
            %{id: s.id, kind: s.kind, isa: Map.get(s, :isa), mode: Map.get(s, :mode), device: a.device, verdict: a.verdict, fingerprint: a.fingerprint,
              reasons: a.reasons, probes: Enum.map(Enum.sort(a.probes), fn {name, p} -> %{name: name, equal: p.equal, envelope: p.envelope, ulps: Map.get(p, :ulps, %{})} end)}

          {:error, why} ->
            %{id: s.id, kind: s.kind, verdict: :unavailable, reasons: [to_string(why)]}
        end
      end
    end)
  end

  # ------------------------------------------------------------- physics --

  @doc """
  Chaos: a double pendulum and its copy started one ulp away, `seconds`
  of motion (positions of both bobs every 20 ms), the distance between
  them, and the first `check` steps run on the oracle too — bit for bit.
  """
  def chaos(seconds \\ 30, check \\ 100) do
    memo({:chaos, seconds}, fn ->
      wd = Physics.double_pendulum(substeps: 8, dt: 0.01)
      x2 = wd.pos0 |> Enum.at(2) |> hd()
      <<i::32>> = <<x2::float-32>>
      <<x2u::float-32>> = <<i + 1::32>>
      st = Physics.state(wd, 2, perturb: fn b, p, a -> if b == 1 and p == 2 and a == :x, do: x2u - x2, else: 0.0 end)
      w = worker(:physics)
      # one call = 2 control steps (20 ms); the oracle checks the first `check` calls
      n = Physics.start(wd, 2, worker: w, state: st, steps: 2)
      o = Physics.start(wd, 2, state: st, steps: 2)
      steps = round(seconds / 0.02)

      {frames, _, _, same} =
        Enum.reduce(1..steps, {[], n, o, true}, fn k, {acc, n, o, same} ->
          {n, sn} = Physics.advance(n)
          {o, same} = if k <= check, do: (({o, so} = Physics.advance(o)); {o, same and so == sn}), else: {o, same}
          [xa, xb] = Physics.rows(n, sn.p_x)
          [ya, yb] = Physics.rows(n, sn.p_y)
          {[[Enum.slice(xa, 1, 2), Enum.slice(ya, 1, 2), Enum.slice(xb, 1, 2), Enum.slice(yb, 1, 2)] | acc], n, o, same}
        end)

      Physics.stop(n)
      frames = Enum.reverse(frames)
      gap = for [[_, xa], [_, ya], [_, xb], [_, yb]] <- frames, do: :math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)

      %{dt: 0.02, frames: frames, gap: gap, checked: check, identical: same, perturbation: x2u - x2,
        substrates: %{fast: if(w, do: "native", else: "oracle"), check: "oracle"}}
    end)
  end

  @doc "A twin of a pendulum beside its plant; at `fault` the plant's rod stretches 0.5 %: residuals, CUSUM, alarm, ledger."
  def twin(fault \\ 60, steps \\ 150) do
    memo({:twin, fault, steps}, fn ->
      wd = Physics.pendulum(theta: 0.6, substeps: 4, dt: 0.02)
      noise = fn t, k -> (Vapor.Sampler.uniform(11, t * 10 + k) - 0.5) * 0.004 end
      meas = fn s, st, t -> %{x: Enum.with_index(hd(Physics.rows(s, st.p_x)), fn v, i -> v + noise.(t, i) end), y: Enum.with_index(hd(Physics.rows(s, st.p_y)), fn v, i -> v + noise.(t, i + 5) end)} end

      {_, tw, trace} =
        Enum.reduce(1..steps, {Physics.start(wd, 1), Physics.twin(wd, worker: worker(:twin)), []}, fn t, {plant, tw, tr} ->
          plant = if t == fault, do: Physics.start(wd, 1, params: [rest: [1.005]], state: unstate(plant)), else: plant
          {plant, st} = Physics.advance(plant)
          tw = Physics.observe(tw, nil, meas.(plant, st, t))
          {plant, tw, [%{t: t, residual: Enum.at(hd(tw.ledger), 4), cusum: tw.cusum, x: Enum.at(hd(Physics.rows(plant, st.p_x)), 1)} | tr]}
        end)

      replay = Physics.replay(tw)
      Physics.stop(tw.sim)
      %{fault: fault, threshold: tw.h, alarm: tw.alarm, trace: Enum.reverse(trace), head: tw.head, entries: length(tw.ledger), replay: replay}
    end)
  end

  defp unstate(sim), do: Map.new(sim.state, fn {k, t} -> [kind, a] = String.split(to_string(k), "_"); {{String.to_atom(kind), String.to_atom(a)}, Enum.chunk_every(Tensor.to_floats(t), sim.world.np)} end)

  # ------------------------------------------------------------- networks --

  @doc "A network (`ba`, `er`, `ws` or `planted`, n ≤ 400) and what can be said of it, each claim against its control."
  def graph(model, n, seed) do
    n = n |> max(40) |> min(400)

    {g, truth} =
      case model do
        "er" -> {Graph.erdos_renyi(n, 6 / (n - 1), seed), nil}
        "ws" -> {Graph.watts_strogatz(n, 6, 0.08, seed), nil}
        "planted" -> Graph.planted(4, div(n, 4), 0.25, 0.01, seed)
        _ -> {Graph.barabasi_albert(n, 3, seed), nil}
      end

    labels = Graph.communities(g)
    pl = Graph.power_law(Graph.degrees(g), boot: 30, seed: seed)
    cz = Graph.zscore(g, &Graph.avg_clustering/1, 6, seed)
    fs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    pr = Graph.pagerank(g)

    %{n: g.n, edges: Graph.edges(g) |> Enum.map(&Tuple.to_list/1), communities: labels, truth: truth,
      metrics: %{clustering: Graph.avg_clustering(g), clustering_null: cz.null_mean, clustering_z: cz.z, assortativity: Graph.assortativity(g),
                 modularity: Graph.modularity(g, labels), nmi: truth && Graph.nmi(labels, truth), giant: Graph.giant(g), threshold: Graph.threshold(g)},
      power_law: Map.take(pl, [:alpha, :xmin, :n_tail, :p, :lr, :lr_p, :verdict]),
      degrees: Graph.degrees(g),
      robustness: %{f: fs, failure: Enum.map(fs, &Graph.percolation(g, &1, :failure, seed)), attack: Enum.map(fs, &Graph.percolation(g, &1, :attack))},
      pagerank: pr |> Enum.with_index() |> Enum.sort_by(fn {v, i} -> {-v, i} end) |> Enum.take(8) |> Enum.map(fn {v, i} -> %{node: i, score: v} end)}
  end

  # ---------------------------------------------------- the trained model --

  @doc "The receipt of the shipped language model (`priv/lm`) and its unbounded stream against growing positions (measured once)."
  def lm do
    memo(:lm, fn ->
      dir = Path.join(to_string(:code.priv_dir(:vapor)), "lm")
      {:ok, receipt} = Vapor.JSON.decode(File.read!(Path.join(dir, "receipt.json")))
      stream = stream_bits(Path.join(dir, "model"))
      %{receipt: receipt, stream: stream}
    end)
  end

  defp stream_bits(model) do
    case worker(:lm) do
      nil -> nil
      w ->
        toks = Vapor.Quality.Suite.corpus("pt_holdout.txt") |> binary_part(0, 900) |> :binary.bin_to_list()
        {:ok, st} = Vapor.Streaming.open(model, sinks: 4, window: 60, worker: w)
        {st, rows} = Vapor.Streaming.feed(st, toks)
        Vapor.Streaming.close(st)
        {:ok, %{program: p}} = Vapor.Model.load(model, max_seq: 1024)
        {:ok, comp} = Vapor.Compile.Lower.lower(p)
        {:ok, s} = Vapor.Runtime.Session.open(w, comp, isa: Substrates.host_isa())
        ids = fn xs -> Tensor.from_list(:s32, [length(xs)], xs) end
        {:ok, o, _} = Vapor.Runtime.Session.step(s, %{tok: ids.(toks), pos: ids.(Enum.to_list(0..(length(toks) - 1)))}, [:logits])
        Vapor.Runtime.Session.close(s)
        dense = o.logits |> Tensor.to_floats() |> Enum.chunk_every(256)

        per = fn rs -> for k <- 0..(div(length(toks), 64) - 1), do: Vapor.Streaming.bits(Enum.slice(rs, k * 64, 65), Enum.slice(toks, k * 64, 65)) end
        %{bytes: length(toks), cache_rows: 64, trained_length: 64, window_bits: %{stream: per.(rows), dense: per.(dense)},
          stream: Vapor.Streaming.bits(rows, toks, 64), dense: Vapor.Streaming.bits(dense, toks, 64)}
    end
  end
end
