defmodule Vapor.Studio do
  @moduledoc """
  **The studio: a graph of media operations whose every result has a receipt.**

  A workflow is a graph of typed nodes (`Vapor.Studio.Node`) — load an image,
  resize it, upscale it, cut a clip, mix two sounds, render a mesh, run a
  model — wired output to input, the way ComfyUI and node editors do it.
  Three properties come from vapor's axioms rather than from the editor:

    * **Content-addressed cache.** A node's *key* is the SHA-256 of its type,
      version, parameters and the keys of what feeds it (a Merkle DAG, like
      a build system). Running a workflow again after changing one parameter
      recomputes exactly the nodes downstream of it; everything else is a
      cache hit — and a hit is *safe*, because every node is deterministic.
    * **A receipt per result.** Every output carries its key and the digest
      of its exact bits; the run has a Merkle root over all of them. Anyone
      with the workflow re-derives every result, on any machine, and gets the
      same root (`verify/2`) — a generated image or clip is a reproducible
      claim, not an artefact of one GPU.
    * **Typed, refused by name.** A wire between incompatible types, a
      missing input, a parameter out of range, a cycle, an unknown node: a
      `Vapor.Rejection` naming the node and the port, before anything runs.

  Graphs are data (`from_json/1`): `{"nodes": {"id": {"type", "params",
  "inputs": {"port": ["src", "port"]}}}}` — close to ComfyUI's API format,
  whose supported subset `Vapor.Studio.Comfy` translates. A graph with
  `studio.input` / `studio.output` nodes is a *subgraph*: `run/2` takes its
  inputs, and nodes such as `video.map` apply it to every frame.
  """
  alias Vapor.{Canonical, Merkle, Rejection}
  alias Vapor.Studio.{Node, Value}

  @key {__MODULE__, :registered}

  # --------------------------------------------------------------- registry --

  @doc "The built-in node modules."
  def builtin do
    [Vapor.Studio.Nodes.Core, Vapor.Studio.Nodes.Image, Vapor.Studio.Nodes.Audio, Vapor.Studio.Nodes.Video]
    |> Enum.filter(&Code.ensure_loaded?/1)
    |> Enum.flat_map(& &1.nodes())
    |> Kernel.++(optional())
  end

  defp optional do
    [Vapor.Studio.Nodes.Vision, Vapor.Studio.Nodes.Diffusion, Vapor.Studio.Nodes.RL, Vapor.Studio.Nodes.Geom]
    |> Enum.filter(&Code.ensure_loaded?/1)
    |> Enum.flat_map(& &1.nodes())
  end

  @doc "Register nodes: a `{module, spec}` pair, or a module implementing `Vapor.Studio.Node` (all its nodes); replaces nodes of the same type."
  def register(mod) when is_atom(mod), do: {:ok, Enum.map(mod.nodes(), &elem(register(&1), 1))}

  def register({mod, spec}) do
    :persistent_term.put(@key, [{mod, spec} | Enum.reject(:persistent_term.get(@key, []), fn {_, s} -> s.type == spec.type end)])
    {:ok, spec.type}
  end

  @doc "Every node: `%{type => {module, spec}}`."
  def nodes do
    (builtin() ++ :persistent_term.get(@key, []))
    |> Map.new(fn {mod, spec} -> {spec.type, {mod, spec}} end)
  end

  @doc "JSON-friendly catalogue of every node, sorted by category and type."
  def catalogue, do: nodes() |> Map.values() |> Enum.map(&Node.describe(elem(&1, 1))) |> Enum.sort_by(&{&1.category, &1.type})

  # ------------------------------------------------------------------ graph --

  @doc """
  A graph from its JSON form (a map with string keys, or the JSON text).
  Returns `%{id => %{type, params, inputs: %{port => {src, port}}}}`.
  """
  def from_json(text) when is_binary(text) do
    with {:ok, m} <- Vapor.JSON.decode(text), do: from_json(m)
  end

  def from_json(%{"nodes" => nodes}) when is_map(nodes) do
    Enum.reduce_while(nodes, {:ok, %{}}, fn {id, n}, {:ok, acc} ->
      case n do
        %{"type" => t} = n when is_binary(t) ->
          ins =
            Map.new(Map.get(n, "inputs", %{}), fn
              {p, [src, port]} -> {to_string(p), {to_string(src), to_string(port)}}
              {p, other} -> {to_string(p), {:bad, other}}
            end)

          {:cont, {:ok, Map.put(acc, to_string(id), %{type: t, params: Map.get(n, "params", %{}), inputs: ins})}}

        _ ->
          {:halt, {:error, Rejection.new({:node, id}, "an object with a \"type\"", "check the workflow JSON")}}
      end
    end)
  end

  def from_json(%{} = graph) when map_size(graph) > 0 do
    if Enum.all?(graph, fn {_, n} -> is_map(n) and Map.has_key?(n, :type) end),
      do: {:ok, graph},
      else: {:error, Rejection.new(:graph, "{\"nodes\": {…}}", "wrap the nodes in a \"nodes\" object")}
  end

  def from_json(_), do: {:error, Rejection.new(:graph, "{\"nodes\": {…}}", "wrap the nodes in a \"nodes\" object")}

  @doc "The JSON form of a graph."
  def to_json(graph) do
    %{"nodes" => Map.new(graph, fn {id, n} ->
      {id, %{"type" => n.type, "params" => json_params(n.params), "inputs" => Map.new(n.inputs, fn {p, {s, sp}} -> {p, [s, sp]} end)}}
    end)}
  end

  defp json_params(ps), do: Map.new(ps, fn {k, v} -> {to_string(k), if(is_binary(v) and not String.valid?(v), do: Base.encode64(v), else: v)} end)

  @doc """
  Check a graph before running it: every node known, every parameter in
  range, every required input wired to an output of a compatible type, no
  cycle. Returns `{:ok, plan}` (nodes in a deterministic topological order,
  with their specs and normalized parameters) or a rejection.
  """
  def validate(graph, opts \\ []) do
    registry = Keyword.get_lazy(opts, :nodes, &nodes/0)

    with {:ok, graph} <- from_json(graph),
         {:ok, typed} <- typecheck(graph, registry),
         {:ok, order} <- topo(graph) do
      {:ok, %{graph: graph, order: order, nodes: typed}}
    end
  end

  defp typecheck(graph, registry) do
    Enum.reduce_while(Enum.sort_by(graph, &sort_key(elem(&1, 0))), {:ok, %{}}, fn {id, n}, {:ok, acc} ->
      with {:ok, {mod, spec}} <- fetch_node(registry, id, n.type),
           {:ok, params} <- Node.params(spec, n.params, id),
           :ok <- check_inputs(graph, registry, id, n, spec) do
        {:cont, {:ok, Map.put(acc, id, %{mod: mod, spec: spec, params: params, inputs: n.inputs})}}
      else
        {:error, _} = e -> {:halt, e}
      end
    end)
  end

  defp fetch_node(registry, id, type) do
    case registry do
      %{^type => found} -> {:ok, found}
      _ ->
        near = registry |> Map.keys() |> Enum.min_by(&String.jaro_distance(&1, type) |> Kernel.-(), fn -> nil end)
        {:error, Rejection.new({:node, id, :type, type}, "a known node type", "did you mean #{inspect(near)}? (Vapor.Studio.catalogue/0 lists them)")}
    end
  end

  defp check_inputs(graph, registry, id, n, spec) do
    declared = Node.inputs(spec)
    names = Enum.map(declared, fn {p, _, _} -> to_string(p) end)

    with nil <- Enum.find_value(Map.keys(n.inputs), fn p -> p not in names && {:unknown, p} end),
         nil <- Enum.find_value(declared, fn {p, t, opt} -> check_wire(graph, registry, id, n.inputs[to_string(p)], to_string(p), t, opt) end) do
      :ok
    else
      {:unknown, p} -> {:error, Rejection.new({:node, id, :input, p}, "one of #{inspect(names)}", "check the node type #{spec.type}")}
      {:error, _} = e -> e
    end
  end

  defp check_wire(_graph, _reg, _id, nil, _p, _t, true), do: nil
  defp check_wire(_graph, _reg, id, nil, p, t, false), do: {:error, Rejection.new({:node, id, :input, p}, "a wire from an output of type #{t}", "connect it")}
  defp check_wire(_graph, _reg, id, {:bad, v}, p, _t, _), do: {:error, Rejection.new({:node, id, :input, p}, "[\"node id\", \"output\"]", "got #{inspect(v)}")}

  defp check_wire(graph, registry, id, {src, sp}, p, t, _) do
    with %{type: st} <- graph[src] || {:error, Rejection.new({:node, id, :input, p}, "a wire from an existing node", "no node #{inspect(src)}")},
         {_, sspec} <- registry[st] || {:error, Rejection.new({:node, src, :type, st}, "a known node type", "see Vapor.Studio.catalogue/0")} do
      case Enum.find(sspec.outputs, fn {op, _} -> to_string(op) == sp end) do
        nil -> {:error, Rejection.new({:node, id, :input, p}, "an output of #{src} (#{Enum.map_join(sspec.outputs, ", ", &to_string(elem(&1, 0)))})", "no output #{inspect(sp)}")}
        {_, ot} ->
          # a subgraph input carries the type its parameter declares
          ot = if st == "studio.input", do: input_type(graph[src].params), else: ot
          if Value.compatible?(ot, t), do: nil, else: {:error, Rejection.new({:node, id, :input, p}, "type #{t}", "#{src}.#{sp} gives #{ot}")}
      end
    else
      {:error, _} = e -> e
    end
  end

  defp input_type(params) do
    t = Map.get(params, "type") || Map.get(params, :type) || "image"
    if t in Enum.map(Value.types(), &Atom.to_string/1), do: String.to_existing_atom(t), else: :json
  end

  # Kahn's algorithm, ties broken by id (numeric ids in numeric order)
  defp topo(graph) do
    deps = Map.new(graph, fn {id, n} -> {id, n.inputs |> Map.values() |> Enum.map(&elem(&1, 0)) |> Enum.uniq()} end)
    go = fn go, done, order ->
      ready = deps |> Enum.filter(fn {id, ds} -> not MapSet.member?(done, id) and Enum.all?(ds, &MapSet.member?(done, &1)) end) |> Enum.map(&elem(&1, 0))

      case Enum.sort_by(ready, &sort_key/1) do
        [] -> if MapSet.size(done) == map_size(graph), do: {:ok, Enum.reverse(order)}, else: cycle(deps, done)
        [id | _] -> go.(go, MapSet.put(done, id), [id | order])
      end
    end
    go.(go, MapSet.new(), [])
  end

  defp cycle(deps, done) do
    left = deps |> Map.keys() |> Enum.reject(&MapSet.member?(done, &1)) |> Enum.sort_by(&sort_key/1)
    {:error, Rejection.new({:cycle, left}, "an acyclic graph", "break the loop between #{Enum.join(left, ", ")}")}
  end

  defp sort_key(id), do: (case Integer.parse(id) do {n, ""} -> {0, n, id}; _ -> {1, 0, id} end)

  # -------------------------------------------------------------------- run --

  @doc """
  Run a graph. Options: `cache` (a map from a previous run's `cache`, or a
  `Vapor.Studio.Cache` pid), `inputs` (values for `studio.input` nodes, by
  name), `worker` (a native worker for the nodes that compile programs),
  `dir` (where `image.load` and friends may read files).

  Returns `{:ok, run}`: `outputs` (`%{id => %{port => value}}`), `results`
  (the values of `studio.output` nodes by name), `receipts`, `root`,
  `executed` and `cached` (node ids), `ms` per executed node, and `cache`
  (the updated cache map when a map was given).
  """
  def run(graph, opts \\ []) do
    with {:ok, plan} <- validate(graph, opts) do
      ctx = %{worker: Keyword.get(opts, :worker), dir: Keyword.get(opts, :dir, "."), inputs: Keyword.get(opts, :inputs, %{}),
              run: opts}
      cache0 = Keyword.get(opts, :cache, %{})

      Enum.reduce_while(plan.order, {:ok, %{outputs: %{}, keys: %{}, receipts: %{}, executed: [], cached: [], ms: %{}, cache: cache0}}, fn id, {:ok, st} ->
        n = plan.nodes[id]
        key = node_key(n, st.keys, ctx)

        case cache_get(st.cache, key) do
          {:ok, {outs, digests}} ->
            {:cont, {:ok, record(st, id, n, key, outs, :cached, 0, digests)}}

          {:ok, outs} ->
            {:cont, {:ok, record(st, id, n, key, outs, :cached, 0)}}

          :miss ->
            ins = Map.new(n.inputs, fn {p, {src, sp}} -> {String.to_atom(p), st.outputs[src][sp]} end)
            t0 = System.monotonic_time(:microsecond)

            case safe_run(n, ins, ctx) do
              {:ok, outs} ->
                outs = Map.new(outs, fn {p, v} -> {to_string(p), v} end)

                case check_outputs(id, n.spec, outs) do
                  :ok ->
                    ms = (System.monotonic_time(:microsecond) - t0) / 1000
                    # digests once, kept with the values: a cached node is not hashed again
                    digests = Map.new(outs, fn {p, v} -> {p, Value.hex(v)} end)
                    {:cont, {:ok, %{record(st, id, n, key, outs, :executed, ms, digests) | cache: cache_put(st.cache, key, {outs, digests})}}}

                  err -> {:halt, err}
                end

              {:error, %Rejection{}} = e -> {:halt, e}
              {:error, other} -> {:halt, {:error, Rejection.new({:node, id, n.spec.type}, "a successful run", inspect(other))}}
            end
        end
      end)
      |> case do
        {:ok, st} ->
          leaves = for id <- plan.order, do: Merkle.leaf(Canonical.encode({id, st.receipts[id].key, Enum.sort(Map.to_list(Map.new(st.receipts[id].outputs, fn {p, o} -> {p, o.digest} end)))}))
          results = for {id, %{spec: %{type: "studio.output"}, params: p}} <- plan.nodes, into: %{}, do: {p.name, st.outputs[id]["value"]}

          {:ok, %{outputs: st.outputs, results: results, receipts: st.receipts, root: Base.encode16(Merkle.root(leaves), case: :lower),
                  executed: Enum.reverse(st.executed), cached: Enum.reverse(st.cached), ms: st.ms, order: plan.order,
                  cache: if(is_map(st.cache), do: st.cache, else: nil)}}

        err -> err
      end
    end
  end

  defp safe_run(n, ins, ctx) do
    n.mod.run(n.spec.type, ins, n.params, ctx)
  rescue
    e -> {:error, Rejection.new({:node, n.spec.type}, "a node that does not raise", Exception.message(e))}
  end

  defp check_outputs(id, spec, outs) do
    Enum.find_value(spec.outputs, :ok, fn {p, t} ->
      v = outs[to_string(p)]
      if Value.is?(t, v) or (t == :mask and Value.is?(:image, v)), do: nil,
        else: {:error, Rejection.new({:node, id, :output, to_string(p)}, "a value of type #{t}", "the node #{spec.type} returned #{inspect(Value.describe(v))}")}
    end)
  end

  defp record(st, id, n, key, outs, how, ms, digests \\ %{}) do
    receipt = %{type: n.spec.type, version: n.spec.version, key: key, params: receipt_params(n.params),
                inputs: Map.new(n.inputs, fn {p, {s, sp}} -> {p, [s, sp]} end),
                outputs: Map.new(outs, fn {p, v} -> {p, %{digest: Map.get_lazy(digests, p, fn -> Value.hex(v) end), value: Value.describe(v)}} end)}

    %{st | outputs: Map.put(st.outputs, id, outs), keys: Map.put(st.keys, id, key), receipts: Map.put(st.receipts, id, receipt),
           executed: if(how == :executed, do: [id | st.executed], else: st.executed),
           cached: if(how == :cached, do: [id | st.cached], else: st.cached),
           ms: if(how == :executed, do: Map.put(st.ms, id, ms), else: st.ms)}
  end

  defp receipt_params(ps), do: Map.new(ps, fn {k, v} -> {k, if(is_binary(v) and not String.valid?(v), do: "sha256:" <> Base.encode16(:crypto.hash(:sha256, v), case: :lower), else: v)} end)

  # the key: type, version, parameters, and what feeds each input (a Merkle DAG);
  # `studio.input` nodes also hash the value given to them
  defp node_key(n, keys, ctx) do
    ins = n.inputs |> Enum.sort() |> Enum.map(fn {p, {src, sp}} -> {p, keys[src], sp} end)
    extra = if n.spec.type == "studio.input", do: Value.hex(Map.get(ctx.inputs, n.params.name)), else: nil
    Canonical.hex_digest({:studio_node, n.spec.type, n.spec.version, n.params |> Enum.sort() |> Enum.map(fn {k, v} -> {Atom.to_string(k), v} end), ins, extra})
  end

  defp cache_get(nil, _), do: :miss
  defp cache_get(c, key) when is_map(c), do: (case c do %{^key => v} -> {:ok, v}; _ -> :miss end)
  defp cache_get(pid, key) when is_pid(pid) or is_atom(pid), do: Vapor.Studio.Cache.get(pid, key)

  defp cache_put(nil, _, _), do: nil
  defp cache_put(c, key, v) when is_map(c), do: Map.put(c, key, v)
  defp cache_put(pid, key, v), do: (Vapor.Studio.Cache.put(pid, key, v); pid)

  @doc """
  Re-derive every result of a graph from nothing (no cache) and compare the
  root with `root`: `:ok` or `{:error, {:root, expected, got}}`.
  """
  def verify(graph, root, opts \\ []) do
    with {:ok, r} <- run(graph, Keyword.put(opts, :cache, nil)) do
      if r.root == root, do: :ok, else: {:error, {:root, root, r.root}}
    end
  end

  @doc "The receipts of a run as JSON-friendly data (`root`, then one entry per node in order)."
  def receipt(run) do
    %{root: run.root, nodes: Enum.map(run.order, fn id -> Map.put(run.receipts[id], :id, id) end)}
  end
end

defmodule Vapor.Studio.Cache do
  @moduledoc "A bounded, shared cache of node outputs by key (least recently used evicted first), for long-lived studios such as the console."
  use Agent

  def start_link(opts \\ []), do: Agent.start_link(fn -> %{max: Keyword.get(opts, :max, 256), map: %{}, tick: 0} end, Keyword.take(opts, [:name]))

  def get(c, key) do
    Agent.get_and_update(c, fn s ->
      case s.map do
        %{^key => {v, _}} -> {{:ok, v}, %{s | map: Map.put(s.map, key, {v, s.tick}), tick: s.tick + 1}}
        _ -> {:miss, s}
      end
    end)
  end

  def put(c, key, v) do
    Agent.update(c, fn s ->
      map = Map.put(s.map, key, {v, s.tick})
      map = if map_size(map) > s.max, do: Map.delete(map, map |> Enum.min_by(fn {_, {_, t}} -> t end) |> elem(0)), else: map
      %{s | map: map, tick: s.tick + 1}
    end)
  end

  def size(c), do: Agent.get(c, &map_size(&1.map))
end
