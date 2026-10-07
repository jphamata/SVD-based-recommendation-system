defmodule Vapor.Lock.Alias do
  @moduledoc """
  Tier 1 of the model airlock: a family described by **data**.

  Many "new" architectures are an existing topology under other names:
  tensors renamed, projections fused into one matrix, configuration keys
  spelled differently. An alias says exactly that, as a JSON-compatible
  map, and the airlock does the rest — no code, no recompilation:

      %{
        "id" => "phi3",
        "model_type" => "phi3",                 # or a list of model_types
        "like" => "llama",                      # the model_type it becomes
        "config" => %{
          "rename" => %{"num_layers" => "num_hidden_layers"},   # key → key
          "set" => %{"attention_bias" => false},                # forced values
          "require" => %{"partial_rotary_factor" => [nil, 1, 1.0]},  # allowed values (nil: absent)
          "forbid" => ["some_key"]                              # must be absent or null
        },
        "tensors" => %{
          "rename" => [["transformer.h.{l}.ln_1.weight", "model.layers.{l}.input_layernorm.weight"]],
          "split" => [%{"from" => "model.layers.{l}.self_attn.qkv_proj.weight",
                        "into" => [["model.layers.{l}.self_attn.q_proj.weight", "heads*head_dim"],
                                   ["model.layers.{l}.self_attn.k_proj.weight", "kv_heads*head_dim"],
                                   ["model.layers.{l}.self_attn.v_proj.weight", "kv_heads*head_dim"]]}]
        }
      }

  Templates use `{name}` placeholders for decimal indices (`{l}` a layer,
  `{e}` an expert). A split cuts a tensor along its first axis into parts
  whose row counts are products of integers and configuration fields —
  fields of the admitted base configuration (`heads`, `kv_heads`,
  `head_dim`, `hidden`, `intermediate`, `vocab`, …) or keys of the
  original `config.json` — and is refused unless the parts cover the rows
  exactly. Admission runs the base family's own checks on the rewritten
  configuration, so an alias can narrow what the base accepts but never
  widen it.

  The spec keeps the alias as its family and the base in its lineage;
  programs are built by the base adapter.
  """
  alias Vapor.{Rejection, Tensor}

  @doc "The aliases that ship with vapor."
  def builtin do
    [
      # Phi-3 / Phi-4: the Llama topology with q/k/v fused into qkv_proj and
      # gate/up fused into gate_up_proj ([gate; up] rows, HF `chunk(2)`).
      # LongRoPE and partial rotary embeddings are refused (by the base's
      # rope check and by `require`).
      %{"id" => "phi3", "model_type" => "phi3", "like" => "llama",
        "config" => %{"set" => %{"attention_bias" => false},
                      "require" => %{"mlp_bias" => [nil, false]}},
        "tensors" => %{"split" => [
          %{"from" => "model.layers.{l}.self_attn.qkv_proj.weight",
            "into" => [["model.layers.{l}.self_attn.q_proj.weight", "heads*head_dim"],
                       ["model.layers.{l}.self_attn.k_proj.weight", "kv_heads*head_dim"],
                       ["model.layers.{l}.self_attn.v_proj.weight", "kv_heads*head_dim"]]},
          %{"from" => "model.layers.{l}.mlp.gate_up_proj.weight",
            "into" => [["model.layers.{l}.mlp.gate_proj.weight", "intermediate"],
                       ["model.layers.{l}.mlp.up_proj.weight", "intermediate"]]}]}}
    ]
    |> Enum.map(fn d -> {:ok, d} = validate(d); d end)
  end

  # ---------------------------------------------------------- validation --

  @doc "Check a descriptor's shape (keys may be atoms or strings): `{:ok, normalised}`."
  def validate(%{} = d) do
    d = stringify(d)
    types = List.wrap(d["model_type"])

    with :ok <- need(is_binary(d["id"]) and d["id"] != "", "id", "a non-empty string"),
         :ok <- need(types != [] and Enum.all?(types, &is_binary/1), "model_type", "a string or a list of strings"),
         :ok <- need(is_binary(d["like"]), "like", "the model_type of the base family"),
         :ok <- need(d["like"] not in types, "like", "a model_type other than the alias's own"),
         cfg = Map.get(d, "config", %{}),
         :ok <- need(is_map(cfg), "config", "an object"),
         :ok <- need(is_map(Map.get(cfg, "rename", %{})) and is_map(Map.get(cfg, "set", %{})) and
                       is_map(Map.get(cfg, "require", %{})) and is_list(Map.get(cfg, "forbid", [])), "config",
                     "rename/set/require objects and a forbid list"),
         :ok <- need(Enum.all?(Map.get(cfg, "require", %{}), fn {_, v} -> is_list(v) end), "config.require", "lists of allowed values"),
         ts = Map.get(d, "tensors", %{}),
         :ok <- need(is_map(ts), "tensors", "an object"),
         :ok <- need(Enum.all?(Map.get(ts, "rename", []), &match?([a, b] when is_binary(a) and is_binary(b), &1)), "tensors.rename", "[[from, to], …] templates"),
         :ok <- need(Enum.all?(Map.get(ts, "split", []), &split?/1), "tensors.split", "[{from, into: [[name, size], …]}, …]"),
         :ok <- sizes_ok(Map.get(ts, "split", [])) do
      {:ok, Map.merge(%{"config" => %{}, "tensors" => %{}}, %{d | "model_type" => types})}
    end
  end

  def validate(other), do: need(false, inspect(other), "an alias map")

  defp split?(%{"from" => f, "into" => parts}) when is_binary(f) and is_list(parts) and parts != [],
    do: Enum.all?(parts, &match?([n, s] when is_binary(n) and (is_binary(s) or (is_integer(s) and s > 0)), &1))

  defp split?(_), do: false

  # size expressions: products of integers and identifiers, nothing else
  defp sizes_ok(splits) do
    bad = for %{"into" => parts} <- splits, [_, s] <- parts, is_binary(s), not String.match?(s, ~r/^\s*[A-Za-z0-9_]+(\s*\*\s*[A-Za-z0-9_]+)*\s*$/), do: s
    need(bad == [], "tensors.split", "sizes like \"heads*head_dim\" (got #{inspect(bad)})")
  end

  defp stringify(%{} = m), do: Map.new(m, fn {k, v} -> {to_string(k), stringify(v)} end)
  defp stringify(l) when is_list(l), do: Enum.map(l, &stringify/1)
  defp stringify(v), do: v

  # ------------------------------------------------------------ adapter --

  def id(d), do: d["id"]

  def claim(d, %{config: %{"model_type" => t}}) when is_binary(t) do
    if t in d["model_type"], do: {:claim, 100}, else: :no
  end

  def claim(_d, _m), do: :no

  def admit(d, manifest, ws, opts) do
    depth = Keyword.get(opts, :alias_depth, 0)
    opts = Keyword.put(opts, :alias_depth, depth + 1)

    with :ok <- need(depth < 8, d["id"], "an alias chain shorter than 8 (\"like\" loops?)"),
         {:ok, map} <- rewrite_config(d, manifest.config),
         base = %{manifest | config: map} |> Map.delete(:tensors),
         {:ok, a} <- Vapor.Lock.select(base),
         {:ok, spec, ws} <- Vapor.Lock.call(a, :admit, [base, ws, opts]),
         {:ok, ws} <- rewrite_tensors(d, ws, field_lookup(spec.config, manifest.config)) do
      {:ok, %{spec | family: d["id"], lineage: [d["id"] | spec.lineage]}, ws}
    end
  end

  # programs are built by the base adapter (the spec names it)
  def build(_d, spec, ws, opts), do: spec.adapter.build(spec, ws, opts)
  def owns?(_d, _config), do: false
  def spec(_d, config), do: Vapor.Lock.spec(config)

  # ------------------------------------------------------------ rewrites --

  @doc false
  def rewrite_config(d, map) do
    c = d["config"]
    renamed = Enum.reduce(Map.get(c, "rename", %{}), map, fn {from, to}, m ->
      case Map.pop(m, from) do
        {nil, m} -> m
        {v, m} -> Map.put(m, to, v)
      end
    end)

    bad_req = Enum.find(Map.get(c, "require", %{}), fn {k, allowed} -> Map.get(renamed, k) not in allowed end)
    bad_forbid = Enum.find(Map.get(c, "forbid", []), &(Map.get(renamed, &1) != nil))

    cond do
      bad_req ->
        {k, allowed} = bad_req
        {:error, Rejection.new({:config, k}, "one of #{inspect(allowed)} for #{d["id"]} (got #{inspect(renamed[k])})", "use a supported checkpoint")}

      bad_forbid ->
        {:error, Rejection.new({:config, bad_forbid}, "absent for #{d["id"]}", "use a supported checkpoint")}

      true ->
        {:ok, renamed |> Map.merge(Map.get(c, "set", %{})) |> Map.put("model_type", d["like"])}
    end
  end

  defp field_lookup(config, raw) do
    fields = if is_struct(config), do: Map.from_struct(config), else: config
    by_name = for {k, v} <- fields, into: %{}, do: {to_string(k), v}

    fn name ->
      cond do
        String.match?(name, ~r/^\d+$/) -> String.to_integer(name)
        is_integer(by_name[name]) -> by_name[name]
        is_integer(raw[name]) -> raw[name]
        true -> nil
      end
    end
  end

  @doc false
  def rewrite_tensors(d, ws, field) do
    t = d["tensors"]

    renamed =
      Enum.reduce(Map.get(t, "rename", []), ws, fn [from, to], acc ->
        re = template(from)

        Enum.reduce(Map.keys(acc), acc, fn
          k, acc when is_binary(k) ->
            case Regex.named_captures(re, k) do
              nil -> acc
              caps -> {v, acc} = Map.pop(acc, k); Map.put(acc, fill(to, caps), v)
            end

          _, acc ->
            acc
        end)
      end)

    Enum.reduce_while(Map.get(t, "split", []), {:ok, renamed}, fn %{"from" => from, "into" => parts}, {:ok, acc} ->
      re = template(from)
      hits = for k <- Map.keys(acc), is_binary(k), caps = Regex.named_captures(re, k), do: {k, caps}

      Enum.reduce_while(hits, {:ok, acc}, fn {k, caps}, {:ok, acc} ->
        case split(acc[k], k, parts, caps, field) do
          {:ok, pieces} -> {:cont, {:ok, acc |> Map.delete(k) |> Map.merge(pieces)}}
          err -> {:halt, err}
        end
      end)
      |> case do
        {:ok, acc} -> {:cont, {:ok, acc}}
        err -> {:halt, err}
      end
    end)
  end

  defp split(%Tensor{shape: [rows | rest], dtype: dt, data: data}, name, parts, caps, field) do
    sizes = Enum.map(parts, fn [_, s] -> size(s, field) end)

    cond do
      Enum.any?(sizes, &is_nil/1) ->
        {:error, Rejection.new({:weight, name}, "split sizes #{inspect(Enum.map(parts, &Enum.at(&1, 1)))} resolvable from the configuration", "check the alias")}

      Enum.sum(sizes) != rows ->
        {:error, Rejection.new({:weight, name}, "parts #{inspect(sizes)} covering its #{rows} rows exactly", "check the alias against config.json")}

      true ->
        row_bytes = div(byte_size(data), rows)

        {pieces, _} =
          Enum.zip(parts, sizes)
          |> Enum.map_reduce(0, fn {[tpl, _], n}, off ->
            {{fill(tpl, caps), Tensor.new(dt, [n | rest], binary_part(data, off * row_bytes, n * row_bytes))}, off + n}
          end)

        {:ok, Map.new(pieces)}
    end
  end

  defp size(n, _field) when is_integer(n), do: n

  defp size(expr, field) do
    vals = expr |> String.split("*") |> Enum.map(&field.(String.trim(&1)))
    if Enum.any?(vals, &is_nil/1), do: nil, else: Enum.product(vals)
  end

  # "model.layers.{l}.x" → ~r/^model\.layers\.(?<l>\d+)\.x$/
  defp template(tpl) do
    parts = Regex.split(~r/\{([a-z_]+)\}/, tpl, include_captures: true)

    src =
      Enum.map_join(parts, fn p ->
        case Regex.run(~r/^\{([a-z_]+)\}$/, p) do
          [_, name] -> "(?<#{name}>\\d+)"
          nil -> Regex.escape(p)
        end
      end)

    Regex.compile!("^" <> src <> "$")
  end

  defp fill(tpl, caps), do: Regex.replace(~r/\{([a-z_]+)\}/, tpl, fn _, n -> Map.fetch!(caps, n) end)

  defp need(true, _f, _b), do: :ok
  defp need(false, f, b), do: {:error, Rejection.new({:alias, f}, b, "fix the alias descriptor")}
end
