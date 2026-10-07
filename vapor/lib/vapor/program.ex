defmodule Vapor.Program do
  @moduledoc """
  A program is a finite DAG of terms with named outputs, optional
  *let-bindings* and an optional *state feedback* relation — the carrier of
  the `scan` generator.

      step : (x_t, h_{t-1}) ↦ (y_t, h_t)        state: [h: :h_next]

  A recurrent program realises the affine scan `h_t = a ⊙ h_{t-1} + u_t` (the
  σ = 2 streaming monoid of Section 9.1) *without* materialising the sequence:
  the runtime iterates the step inside the native worker, one BEAM crossing
  per sequence (not per token), and streams the declared outputs back.

  ## Let-bindings

  `lets: [name: term, …]` names intermediate values; later terms refer to
  them by the leaf `Term.ref(name, term)` — an `{:input, name, dtype, shape}`
  the program supplies itself. Semantically a binding is substitution
  (`let x = e in t` ≡ `t[x := e]`), so nothing downstream changes meaning.
  It exists for scale: BEAM terms are trees, and a value used `k` times
  per layer (a residual stream) makes the unshared tree of an `L`-layer
  model grow like `k^L` — hashing, comparing or copying it would be
  exponential. With the stream bound once per layer every term stays
  layer-sized. Weights are bound the same way (`name: {:const, tensor}`),
  so no term used as a map key ever contains weight bytes.

  Bindings are ordered: a body may refer only to earlier names.
  """
  alias Vapor.Algebra.Term
  alias Vapor.Rejection

  @enforce_keys [:outputs]
  defstruct outputs: [], state: [], lets: []

  @type t :: %__MODULE__{outputs: [{atom, term}], state: [{atom, atom}], lets: [{atom, term}]}

  def new(outputs, opts \\ []) do
    %__MODULE__{outputs: outputs, state: Keyword.get(opts, :state, []), lets: Keyword.get(opts, :lets, [])}
  end

  @doc "Distinct free `{:input, …}` leaves (not let-bound), sorted by name."
  def inputs(%__MODULE__{} = p) do
    bound = bound(p)

    p
    |> order()
    |> Enum.filter(&match?({:input, _, _, _}, &1))
    |> Enum.reject(&Map.has_key?(bound, elem(&1, 1)))
    |> Enum.uniq()
    |> Enum.sort()
  end

  @doc "Let-bound names → bodies."
  def bound(%__MODULE__{lets: lets}), do: Map.new(lets)

  @doc """
  The terms to enumerate, in dependency order: every non-constant let body,
  then every output. Constant bodies are left out on purpose — their `ref`
  leaf stands for them, so weight bytes never become part of a key.
  """
  def roots(%__MODULE__{lets: lets, outputs: outs}),
    do: for({_, b} <- lets, not match?({:const, _}, b), do: b) ++ Enum.map(outs, &elem(&1, 1))

  @doc "Children-first enumeration of all distinct terms (`ref` leaves included)."
  def order(%__MODULE__{} = p), do: Term.postorder(roots(p))

  @doc """
  The resolver of `ref` leaves: a ref to a node (or to another ref) resolves
  to that node; a ref to a constant stays a leaf; everything else is itself.
  """
  def resolver(%__MODULE__{} = p) do
    bound = bound(p)
    &follow(&1, bound)
  end

  defp follow({:input, name, _, _} = t, bound) do
    case Map.fetch(bound, name) do
      {:ok, {:const, _}} -> t
      {:ok, body} -> follow(body, bound)
      :error -> t
    end
  end

  defp follow(t, _bound), do: t

  @doc "`{:const, tensor}` if `t` is a constant or a ref bound to one, else `nil`."
  def constant(%__MODULE__{} = p, t) do
    bound = bound(p)

    case follow(t, bound) do
      {:const, _} = c -> c
      {:input, name, _, _} -> with({:const, _} = c <- Map.get(bound, name), do: c, else: (_ -> nil))
      _ -> nil
    end
  end

  @doc """
  Evaluate the program with a node evaluator `node.(term, memo, env)`:
  bindings in order (a constant binding enters `env` as `const.(tensor)`,
  any other as its body's value), then the outputs. Returns
  `%{output_name => value}`. Oracle and envelope both evaluate this way.
  """
  def evaluate(%__MODULE__{} = p, env, const, node) do
    step = fn root, {env, memo} ->
      memo =
        root
        |> Term.postorder()
        |> Enum.reduce(memo, fn t, m -> if Map.has_key?(m, t), do: m, else: Map.put(m, t, node.(t, m, env)) end)

      {env, memo}
    end

    {env, memo} =
      Enum.reduce(p.lets, {env, %{}}, fn
        {name, {:const, t}}, {env, memo} -> {Map.put(env, name, const.(t)), memo}
        {name, body}, acc -> {env, memo} = step.(body, acc); {Map.put(env, name, Map.fetch!(memo, body)), memo}
      end)

    {_, memo} = Enum.reduce(p.outputs, {env, memo}, fn {_, t}, acc -> step.(t, acc) end)
    Map.new(p.outputs, fn {name, t} -> {name, Map.fetch!(memo, t)} end)
  end

  @doc "Rung 1 over the whole program: bindings, outputs, state-feedback sort agreement."
  @spec check(t) :: :ok | {:error, Rejection.t()}
  def check(%__MODULE__{outputs: outs, state: st} = p) do
    with :ok <- check_lets(p),
         :ok <- all_ok(outs, fn {_n, t} -> Term.infer(t) end),
         :ok <- unique_inputs(p) do
      all_ok(st, fn {in_name, out_name} ->
        with {:input, _, dt, s} <- Enum.find(inputs(p), &match?({:input, ^in_name, _, _}, &1)),
             {_, t} <- List.keyfind(outs, out_name, 0),
             {:ok, {^dt, ^s}} <- Term.infer(t) do
          {:ok, :state}
        else
          _ ->
            {:error,
             %Rejection{node: {:state, in_name, out_name},
                        bound: "state input and fed-back output share a sort",
                        repair: "make #{out_name} have the dtype/shape of input #{in_name}"}}
        end
      end)
    end
  end

  # every body is well sorted and refers only to earlier bindings, with the
  # sort each earlier binding declared; the same holds for the outputs
  defp check_lets(%__MODULE__{lets: lets, outputs: outs}) do
    names = Enum.map(lets, &elem(&1, 0))
    all = MapSet.new(names)

    cond do
      length(names) != MapSet.size(all) ->
        {:error, Rejection.new(:lets, "binding names are unique", "rename the duplicate binding")}

      true ->
        lets
        |> Enum.reduce_while({:ok, %{}}, fn {name, body}, {:ok, sorts} ->
          with {:ok, sort} <- Term.infer(body),
               :ok <- refs_ok(body, sorts, all, name) do
            {:cont, {:ok, Map.put(sorts, name, sort)}}
          else
            err -> {:halt, err}
          end
        end)
        |> case do
          {:ok, sorts} -> all_ok(outs, fn {n, t} -> with(:ok <- refs_ok(t, sorts, all, n), do: {:ok, n}) end)
          err -> err
        end
    end
  end

  defp refs_ok({:const, _}, _sorts, _all, _at), do: :ok

  defp refs_ok(body, sorts, all, at) do
    body
    |> Term.postorder()
    |> Enum.find(fn
      {:input, n, dt, s} -> MapSet.member?(all, n) and Map.get(sorts, n) != {dt, s}
      _ -> false
    end)
    |> case do
      nil -> :ok
      {:input, n, _, _} = leaf ->
        {:error, Rejection.new({:let, at, leaf},
                               "#{inspect(n)} is bound earlier, with the sort its reference declares",
                               "bind #{inspect(n)} before #{inspect(at)} and refer to it with Term.ref/2")}
    end
  end

  defp unique_inputs(p) do
    names = p |> inputs() |> Enum.map(&elem(&1, 1))

    if length(names) == length(Enum.uniq(names)),
      do: :ok,
      else:
        {:error, %Rejection{node: :inputs, bound: "input names are unique",
                            repair: "rename conflicting inputs"}}
  end

  defp all_ok(xs, f) do
    Enum.reduce_while(xs, :ok, fn x, :ok ->
      case f.(x) do
        {:ok, _} -> {:cont, :ok}
        :ok -> {:cont, :ok}
        {:error, _} = e -> {:halt, e}
      end
    end)
  end
end
