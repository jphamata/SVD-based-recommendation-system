defmodule Vapor.Lock do
  @moduledoc """
  **The model airlock.**

  The format airlocks (`Vapor.JSON`, `Vapor.Ingest.Safetensors`,
  `Vapor.Ingest.GGUF`) make sure bytes are well formed. This airlock makes
  sure *models* are: it is the one boundary where a model family is known.
  On the inside, the core — engine, embedder, server, merger, quality gate
  — sees a `Vapor.Lock.Spec` (a contract) and a `Vapor.Program` (typed
  terms), never a family name, a tensor naming scheme or a topology.

      checkpoint ──► manifest ──► adapter.claim ──► adapter.admit ──► Spec + weights
                                                                       │
                                     core ◄── Contract.check ◄── adapter.build

  Three properties are enforced here, not hoped for:

    * **one owner** — every manifest is claimed by the highest-scoring
      adapter; adapters registered at run time come before the built-in
      ones, so an override is explicit; a checkpoint nobody claims is a
      rejection listing every near miss (`explain/1`);
    * **contract at the lock** — every program an adapter builds is checked
      against the contract its spec declares (`Vapor.Lock.Contract`) before
      the core sees it;
    * **no family in the core** — tested: the core's compiled modules hold
      no reference to any adapter or model module.

  Adding a model is one of (see `Vapor.Lock.Adapter`):

      Vapor.Lock.register(%{"id" => "phi3", "model_type" => "phi3", "like" => "llama", …})  # data
      Vapor.Lock.register_json("my_family.json")                                          # data, no code
      Vapor.Lock.register(MyApp.MyTopology)                                               # code

  and `VAPOR_LOCK_ALIASES=a.json:b.json` registers alias files at start.
  """
  alias Vapor.{JSON, Program, Rejection, Tensor}
  alias Vapor.Lock.{Alias, Contract, Spec}

  @key {__MODULE__, :registered}

  # ------------------------------------------------------------- registry --

  @doc "The built-in adapters, in priority order."
  def builtin do
    [Vapor.Lock.Adapters.Multipliers, Vapor.Lock.Adapters.Decoder, Vapor.Lock.Adapters.Encoder,
     Vapor.Lock.Adapters.Codec, Vapor.Lock.Adapters.Linear, Vapor.Lock.Adapters.MLP, Vapor.Lock.Adapters.VAE,
     Vapor.Lock.Adapters.DiT, Vapor.Lock.Adapters.UNet, Vapor.Lock.Adapters.Mamba, Vapor.Lock.Adapters.Mamba2,
     Vapor.Lock.Adapters.EncoderDecoder, Vapor.Lock.Adapters.DeltaHybrid] ++
      Enum.map(Alias.builtin(), &{Alias, &1})
  end

  @doc "Every adapter, registered ones first."
  def adapters do
    env_aliases()
    :persistent_term.get(@key, []) ++ Application.get_env(:vapor, :adapters, []) ++ builtin()
  end

  @doc """
  Register an adapter (a module implementing `Vapor.Lock.Adapter`) or an
  alias descriptor (a map, see `Vapor.Lock.Alias`). A registration with
  the id of an earlier one replaces it.
  """
  def register(adapter) do
    with {:ok, a} <- normalise(adapter) do
      i = id(a)
      :persistent_term.put(@key, [a | Enum.reject(:persistent_term.get(@key, []), &(id(&1) == i))])
      {:ok, i}
    end
  end

  @doc "Register every alias descriptor of a JSON file (an object or an array of objects)."
  def register_json(path) do
    with {:ok, bin} <- File.read(path) |> file_error(path),
         {:ok, doc} <- JSON.decode(bin) do
      doc
      |> List.wrap()
      |> Enum.reduce_while({:ok, []}, fn d, {:ok, acc} ->
        case register(d) do
          {:ok, i} -> {:cont, {:ok, acc ++ [i]}}
          err -> {:halt, err}
        end
      end)
    end
  end

  @doc "Remove a registered adapter by id (built-ins stay)."
  def unregister(id) do
    :persistent_term.put(@key, Enum.reject(:persistent_term.get(@key, []), &(__MODULE__.id(&1) == id)))
    :ok
  end

  defp normalise(mod) when is_atom(mod) do
    needed = [id: 0, claim: 1, admit: 3, build: 3, owns?: 1, spec: 1]

    case Code.ensure_loaded(mod) do
      {:module, _} ->
        case Enum.find(needed, fn {f, a} -> not function_exported?(mod, f, a) end) do
          nil -> {:ok, mod}
          {f, a} -> {:error, Rejection.new({:adapter, mod}, "implements #{f}/#{a} (Vapor.Lock.Adapter)", "complete the adapter")}
        end

      _ ->
        {:error, Rejection.new({:adapter, mod}, "a loaded module", "compile the adapter")}
    end
  end

  defp normalise({Alias, d}), do: normalise(d)
  defp normalise(%{} = d), do: with({:ok, d} <- Alias.validate(d), do: {:ok, {Alias, d}})
  defp normalise(other), do: {:error, Rejection.new({:adapter, other}, "a module or an alias map", "see Vapor.Lock.Adapter")}

  # alias files named by the environment, registered once per VM
  defp env_aliases do
    with paths when is_binary(paths) and paths != "" <- System.get_env("VAPOR_LOCK_ALIASES"),
         false <- :persistent_term.get({__MODULE__, :env, paths}, false) do
      :persistent_term.put({__MODULE__, :env, paths}, true)
      for p <- String.split(paths, ":", trim: true), do: register_json(p)
    end

    :ok
  end

  @doc "An adapter's id."
  def id({mod, data}), do: mod.id(data)
  def id(mod) when is_atom(mod), do: mod.id()

  @doc false
  def call({mod, data}, fun, args), do: apply(mod, fun, [data | args])
  def call(mod, fun, args), do: apply(mod, fun, args)

  # ------------------------------------------------------------ selection --

  @doc """
  The adapter that claims a manifest: `{:ok, adapter}` or a rejection whose
  bound lists every near miss.
  """
  def select(manifest) do
    claims = Enum.map(adapters(), &{&1, safe_claim(&1, manifest)})

    case for({a, {:claim, s}} <- claims, do: {a, s}) do
      [] ->
        near = for {a, {:near, why}} <- claims, do: "#{id(a)}: #{why}"
        what = describe(manifest)

        {:error,
         Rejection.new({:lock, what}, "an adapter that claims #{what}" <> if(near == [], do: "", else: " — near misses: " <> Enum.join(near, "; ")),
                       "register an alias (Vapor.Lock.register/1, VAPOR_LOCK_ALIASES) or an adapter; `mix vapor.lock` explains")}

      found ->
        top = found |> Enum.map(&elem(&1, 1)) |> Enum.max()
        {:ok, found |> Enum.find(fn {_, s} -> s == top end) |> elem(0)}
    end
  end

  defp safe_claim(a, manifest) do
    call(a, :claim, [manifest])
  rescue
    e -> {:near, "claim raised #{Exception.message(e)}"}
  end

  defp describe(%{config: %{"model_type" => t}}), do: "model_type #{inspect(t)}"
  defp describe(%{source: :gguf, meta: %{"general.architecture" => a}}), do: "GGUF architecture #{inspect(a)}"
  defp describe(%{path: p}) when is_binary(p), do: p
  defp describe(_), do: "this checkpoint"

  # ------------------------------------------------------------- admission --

  @doc "Admit a manifest and its weights: `{:ok, spec, weights}`."
  def admit(manifest, weights, opts \\ []) do
    manifest = Map.put_new(manifest, :tensors, index(weights))

    with {:ok, a} <- select(manifest) do
      call(a, :admit, [manifest, weights, opts])
    end
  end

  @doc """
  Admission from **shapes alone**: a configuration and the tensor table of
  a checkpoint (`%{name => {dtype, shape}}`, as safetensors headers give
  it), without one byte of tensor data — for a checkpoint too large to
  fetch before knowing whether vapor can run it (docs/SIPHON.md). Returns
  `{:ok, %{spec, expected, missing, unread}}` (`missing`: expected tensors
  absent or misshapen, as `{name, want, got}`) or the adapter's rejection.

  MXFP4 pairs (`X_blocks : U8[r, k/32, 16]`, `X_scales : U8[r, k/32]`)
  stand for the matrix `X : [r, k]` they decode to. An adapter that needs
  tensor data to admit (none of the built-in text families do) is a
  rejection that says so.
  """
  def preflight(config, table) when is_map(config) and is_map(table) do
    ws = shapes_only(table)
    manifest = %{source: :headers, config: config, tensors: index(ws)}

    with {:ok, a} <- select(manifest) do
      try do
        case call(a, :admit, [manifest, ws, []]) do
          {:ok, spec, ws} ->
            exp = expected(spec)
            have = index(ws)
            names = MapSet.new(exp, &elem(&1, 0))
            {:ok, %{spec: spec, expected: length(exp), missing: for({n, s, _} <- exp, have[n] != s, do: {n, s, have[n]}),
                    unread: have |> Map.keys() |> Enum.reject(&MapSet.member?(names, &1)) |> Enum.sort()}}

          err ->
            err
        end
      rescue
        e -> {:error, Rejection.new({:preflight, id(a)}, "an adapter that admits from shapes alone", "#{id(a)} reads tensor data at admission (#{Exception.message(e)}): fetch the weights")}
      end
    end
  end

  # tensors with a dtype and a shape and no data; MXFP4 pairs folded into the matrix they decode to
  defp shapes_only(table) do
    dt = %{"F32" => :f32, "BF16" => :bf16, "F16" => :f16, "U8" => :u8, "I8" => :s8, "I32" => :s32}
    t = Map.new(table, fn {n, {d, shape}} -> {n, %Tensor{dtype: Map.get(dt, d, :other), shape: shape, data: nil}} end)

    Enum.reduce(t, t, fn {n, blk}, acc ->
      base = String.replace_suffix(n, "_blocks", "")

      case {n != base, blk.shape, acc[base <> "_scales"]} do
        {true, [r, nb, 16], %Tensor{shape: [r, nb]}} -> acc |> Map.drop([n, base <> "_scales"]) |> Map.put(base, %Tensor{dtype: :f32, shape: [r, nb * 32], data: nil})
        _ -> acc
      end
    end)
  end

  @doc "Admit an in-memory checkpoint: a configuration map and weights."
  def from_map(config, weights, opts \\ []) when is_map(config),
    do: admit(%{source: :memory, config: config}, weights, opts)

  @doc """
  Open a checkpoint (a Hugging Face directory or a `.gguf` file) through
  the format airlocks and this one: `{:ok, %{spec, config, weights,
  tokenizer}}` (`config` is `spec.config`). Options: `bf16: :keep`.
  """
  def open(path, opts \\ []) do
    with {:ok, manifest, weights, tk} <- read(path, opts),
         {:ok, spec, weights} <- admit(manifest, weights, opts) do
      {:ok, %{spec: spec, config: spec.config, weights: weights, tokenizer: tk}}
    end
  end

  defp read(path, opts) do
    if File.regular?(path) and String.ends_with?(path, ".gguf") do
      with {:ok, m} <- Vapor.Model.GGUF.load(path) do
        {:ok, %{source: :gguf, path: path, admitted: m.config, tensors: index(m.weights)}, m.weights, m.tokenizer}
      end
    else
      with {:ok, bin} <- File.read(Path.join(path, "config.json")) |> file_error(Path.join(path, "config.json")),
           {:ok, map} <- JSON.decode(bin, nonfinite: true),
           {:ok, ws} <- Vapor.Model.weights(path, Keyword.take(opts, [:bf16])) do
        tk = case Vapor.Model.tokenizer(path) do
          {:ok, tk} -> tk
          _ -> nil
        end

        {:ok, %{source: :hf, path: path, config: map, tensors: index(ws)}, ws, tk}
      end
    end
  end

  defp file_error({:ok, _} = ok, _), do: ok
  defp file_error({:error, why}, path), do: {:error, Rejection.new({:file, path}, "readable (#{inspect(why)})", "check the path")}

  defp index(weights), do: for({k, %Tensor{shape: s}} <- weights, is_binary(k), into: %{}, do: {k, s})

  # --------------------------------------------------------------- specs --

  @doc """
  The spec of an admitted configuration (or the spec itself): the core's
  way to talk about a model it was handed as a configuration value.
  """
  def spec(%Spec{} = s), do: s

  def spec(config) do
    case Enum.find(adapters(), &owns?(&1, config)) do
      nil -> raise ArgumentError, "no adapter owns #{inspect(config, limit: 3)}"
      a -> call(a, :spec, [config])
    end
  end

  defp owns?(a, config) do
    call(a, :owns?, [config])
  rescue
    _ -> false
  end

  @doc """
  Build a program for a spec (or an admitted configuration) and check it
  against the spec's contract. Options are the adapter's (see its docs);
  the core passes only those the spec lists in `features`.
  """
  def build(spec_or_config, weights, opts \\ []) do
    spec = spec(spec_or_config)

    with {:ok, p} <- spec.adapter.build(spec, weights, opts),
         :ok <- Contract.check(spec, p, opts) do
      {:ok, p}
    end
  end

  @doc "Every tensor the spec's builder reads (`[]` when the adapter does not say)."
  def expected(%Spec{adapter: a} = s) do
    if function_exported?(a, :expected, 1), do: a.expected(s), else: []
  end

  @doc """
  The activations a spec's program exposes for measurement, and the
  matrices that read each: `[{let_name, [tensor, …]}]` (`[]` when the
  adapter does not say). Fusion by least squares (`Vapor.Merge.calibrate/3`)
  reads them; the core never learns what a layer is.
  """
  def taps(%Spec{adapter: a} = s) do
    if is_atom(a) and function_exported?(a, :taps, 1), do: a.taps(s), else: []
  end

  @doc """
  The window that binds on every attention layer of a `s`-position cache,
  or `nil` (the adapter says; `nil` when it does not) — `Vapor.Engine`'s
  circular cache asks here, never the model's configuration.
  """
  def ring_window(%Spec{adapter: a} = spec, s) do
    if is_atom(a) and function_exported?(a, :ring_window, 2), do: a.ring_window(spec, s), else: nil
  end

  @doc """
  Zero values for every state input of a program (caches, recurrences),
  at their maximal extents — derived from the program, not the model.
  """
  def zero_state(%Program{} = p) do
    ins = Map.new(Program.inputs(p), fn {:input, n, dt, s} -> {n, {dt, s}} end)

    for {i, _} <- p.state, {dt, s} = ins[i], into: %{} do
      shape = Vapor.Algebra.Term.shape_max(s)
      {i, Tensor.new(dt, shape, :binary.copy(<<0>>, Tensor.nbytes(dt, shape)))}
    end
  end

  # ---------------------------------------------------------- diagnostics --

  @doc """
  Why a checkpoint is (not) admitted: every adapter's answer, the winner,
  and — when admitted — the tensors the builder needs that are missing or
  misshapen, and the ones it will not read.
  """
  def explain(path_or_manifest, weights \\ nil) do
    {manifest, weights} =
      case path_or_manifest do
        p when is_binary(p) ->
          case read(p, []) do
            {:ok, m, ws, _} -> {m, ws}
            {:error, r} -> {%{source: :error, path: p, error: r}, %{}}
          end

        %{} = m ->
          {Map.put_new(m, :tensors, index(weights || %{})), weights || %{}}
      end

    claims = for a <- adapters(), do: %{adapter: id(a), answer: safe_claim(a, manifest)}

    admitted =
      case manifest do
        %{source: :error, error: r} -> {:error, r}
        _ -> admit(manifest, weights)
      end

    tensors =
      case admitted do
        {:ok, spec, ws} ->
          exp = expected(spec)
          have = index(ws)
          missing = for {n, s, _} <- exp, have[n] != s, do: {n, s, have[n]}
          names = MapSet.new(exp, &elem(&1, 0))
          %{expected: length(exp), missing: missing, unused: have |> Map.keys() |> Enum.reject(&MapSet.member?(names, &1)) |> Enum.sort()}

        _ ->
          nil
      end

    %{claims: claims, admitted: admitted |> then(fn {:ok, s, _} -> {:ok, s |> Map.from_struct() |> Map.drop([:config])}; e -> e end), tensors: tensors}
  end
end
