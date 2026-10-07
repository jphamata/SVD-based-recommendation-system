defmodule Vapor.Nx do
  @moduledoc """
  Numerical Elixir with vapor's guarantee: a `defn` compiled by
  `Vapor.Nx.Compiler` runs as certified machine code whose results are the
  same bits on x86-64 (AVX2, AVX-512), AArch64, RISC-V and Vulkan — and,
  for `+ − ×`, the same bits as Nx's own reference evaluator, because both
  round every operation correctly. Division and the transcendental
  functions are vapor's canonical microprograms: identical on every
  substrate, within certified ulp bounds of the correctly rounded result
  (÷ within 1 ulp).

      defn layer(x, w, b), do: Nx.sigmoid(Nx.dot(x, [1], w, [1]) + b)

      f = Nx.Defn.jit(&layer/3, compiler: Vapor.Nx.Compiler)
      f.(x, w, b)                                   # an Nx.Tensor (BinaryBackend)
      {:ok, compiled} = Vapor.Nx.certify(&layer/3, [x, w, b])
      compiled.certificate.payload.parity           # which substrates agreed, bit for bit

  This is *not* an Nx backend for everything: vapor's canonical policy
  covers a fragment — f32 elementwise arithmetic with broadcasting,
  `exp`, `sigmoid`, `tanh`, `rsqrt`, `max`/`min`, `select` on `less`,
  `dot` against a rank-2 operand, and sums or maxima over the last axis
  (extent ≡ 0 mod 16). Anything else is refused at compile time with the
  operation named, never approximated. Where reproducibility across
  hardware is the requirement — a published result, a regulated
  computation, a test oracle — the fragment is the price of the guarantee;
  for everything else EXLA or Torchx remain the right tools.

  Also: `from_nx/1` and `to_nx/1` convert tensors (f32, bf16, s32; same
  row-major little-endian bytes, no copy beyond the binary).
  """
  alias Vapor.Tensor

  @doc "An `Nx.Tensor` as a `Vapor.Tensor` (f32, bf16, s32)."
  def from_nx(%Nx.Tensor{type: type, shape: shape} = t) do
    dtype = case type do
      {:f, 32} -> :f32
      {:bf, 16} -> :bf16
      {:s, 32} -> :s32
      other -> raise ArgumentError, "vapor tensors are f32, bf16 or s32, not #{inspect(other)}"
    end

    Tensor.new(dtype, Tuple.to_list(shape), Nx.to_binary(t))
  end

  @doc "A `Vapor.Tensor` as an `Nx.Tensor` on the binary backend."
  def to_nx(%Tensor{dtype: dtype, shape: shape, data: data}) do
    type = case dtype do
      :f32 -> {:f, 32}
      :bf16 -> {:bf, 16}
      :s32 -> {:s, 32}
      other -> raise ArgumentError, "no Nx type for #{inspect(other)}"
    end

    data |> Nx.from_binary(type, backend: Nx.BinaryBackend) |> Nx.reshape(List.to_tuple(shape))
  end

  @doc """
  Certify `fun` for the shapes of `args` (`Vapor.compile/2`: the six-rung
  ladder, every substrate of this machine, a signed certificate). Options
  are passed to `Vapor.compile/2` (e.g. `key:`).
  """
  def certify(fun, args, opts \\ []) do
    out = apply(Nx.Defn.debug_expr(fun), args)
    {program, _outs} = Vapor.Nx.Lower.program(Nx.Defn.Composite.flatten_list([out]))
    Vapor.compile(program, opts)
  end
end

defmodule Vapor.Nx.Compiler do
  @moduledoc """
  An `Nx.Defn.Compiler`: `Nx.Defn.jit(fun, compiler: Vapor.Nx.Compiler)`.
  The expression is lowered to a vapor program (`Vapor.Nx.Lower`), certified
  once per shape signature (cached), and run through `Vapor.run/3` — on the
  substrate the arbiter picks, with failover. Options: `key:` (certificate
  signing key; default a fresh one per node), `substrates:` (restrict where
  it runs).
  """
  @behaviour Nx.Defn.Compiler
  alias Nx.Defn.Composite

  @impl true
  def __partitions_options__(opts), do: [opts]

  @impl true
  def __to_backend__(_opts), do: {Nx.BinaryBackend, []}

  @impl true
  def __jit__(key, vars, fun, args_list, opts), do: __compile__(key, vars, fun, opts).(args_list)

  @impl true
  def __compile__(_key, vars, fun, opts) do
    {exprs, templates} = vars |> fun.() |> Composite.traverse([], &{&1, [Nx.to_template(&1) | &2]})
    templates = Enum.reverse(templates)
    {program, outs} = Vapor.Nx.Lower.program(Composite.flatten_list([exprs]))
    compiled = certified(program, opts)

    fn args_list ->
      for params <- args_list do
        env =
          for {name, {pos, shape}} <- Vapor.Nx.Lower.inputs(program), into: %{} do
            t = Enum.at(params, pos).() |> Nx.as_type(:f32) |> Vapor.Nx.from_nx()
            {name, %{t | shape: shape}}
          end

        {:ok, result, _trace} = Vapor.run(compiled, env, Keyword.take(opts, [:substrates]))

        values = for {name, tpl} <- Enum.zip(outs, templates), do: Vapor.Nx.to_nx(result.outputs[name]) |> Nx.reshape(tpl.shape)
        {out, []} = Composite.traverse(exprs, values, fn _expr, [v | rest] -> {v, rest} end)
        out
      end
    end
  end

  # one certification per program (the ladder is the expensive part)
  defp certified(program, opts) do
    # the program and the options that change the certificate (signing key, substrates)
    key = {__MODULE__, :crypto.hash(:sha256, :erlang.term_to_binary({program, Keyword.take(opts, [:key, :substrates])}))}

    case :persistent_term.get(key, nil) do
      nil ->
        case Vapor.compile(program, Keyword.take(opts, [:key, :substrates])) do
          {:ok, c} -> :persistent_term.put(key, c); c
          {:error, why} -> raise ArgumentError, "vapor refused the program: #{Exception.message(why)}"
        end

      c ->
        c
    end
  end
end

defmodule Vapor.Nx.Lower do
  @moduledoc """
  `Nx.Defn.Expr` → `Vapor.Algebra.Term`. Each node is lowered once (by id);
  the vapor shape of every term must equal the Nx shape of its node, so a
  layout the term algebra cannot express is refused rather than guessed.

  Reshapes are free on leaves: a parameter or constant is row-major bytes,
  so `Nx.reshape`/`Nx.new_axis` of a leaf (and the rank padding Nx's
  broadcasting implies) becomes the same bytes declared with another shape.
  """
  alias Vapor.Algebra.Term, as: T
  alias Nx.Defn.Expr

  @unary %{exp: :exp, sigmoid: :sigmoid, tanh: :tanh, rsqrt: :rsqrt, negate: :neg}
  @binary %{add: :add, subtract: :sub, multiply: :mul, divide: :div, max: :max, min: :min}

  @doc "The vapor program for a list of output expressions, and the output names."
  def program(outputs) do
    {terms, _memo} = Enum.map_reduce(outputs, %{}, fn o, memo -> lower(o, memo) end)

    outs =
      for {term, i} <- Enum.with_index(terms) do
        if match?({:splat, _}, term), do: refuse("a constant output", "return a tensor computed from the inputs")
        {:"out#{i}", term}
      end

    {Vapor.Program.new(outs), Keyword.keys(outs)}
  end

  @doc "Inputs of a lowered program: `name => {parameter position, declared shape}`."
  def inputs(program) do
    for {:input, name, :f32, shape} <- Vapor.Program.inputs(program), into: %{} do
      [pos | _] = name |> Atom.to_string() |> String.trim_leading("p") |> String.split("@")
      {name, {String.to_integer(pos), shape}}
    end
  end

  defp lower(%Nx.Tensor{data: %Expr{id: id}, type: type} = t, memo) do
    case memo do
      %{^id => term} -> {term, memo}
      _ when type != {:f, 32} -> refuse("#{inspect(t.data.op)} of type #{inspect(type)}", "compute in f32")
      _ ->
        {term, memo} = node(t, memo)
        {term, Map.put(memo, id, term)}
    end
  end

  defp node(%Nx.Tensor{data: %Expr{op: :parameter, args: [pos]}, shape: s}, memo), do: {leaf_input(pos, shape_list(s)), memo}

  defp node(%Nx.Tensor{data: %Expr{op: :tensor, args: [tensor]}}, memo), do: {T.const(Vapor.Nx.from_nx(Nx.as_type(tensor, :f32))), memo}

  defp node(%Nx.Tensor{data: %Expr{op: :constant, args: [n]}}, memo) when is_number(n), do: {T.splat(n * 1.0), memo}

  defp node(%Nx.Tensor{data: %Expr{op: :metadata, args: [inner, _]}}, memo), do: lower(inner, memo)

  defp node(%Nx.Tensor{data: %Expr{op: :as_type, args: [inner]}}, memo) do
    if inner.type == {:f, 32}, do: lower(inner, memo), else: refuse("as_type from #{inspect(inner.type)}", "pass f32 inputs")
  end

  # reshape and broadcast of a leaf: the same bytes under another shape
  defp node(%Nx.Tensor{data: %Expr{op: :reshape, args: [inner]}, shape: s}, memo) do
    {term, memo} = lower(inner, memo)
    {reshape_leaf!(term, shape_list(s), "reshape"), memo}
  end

  defp node(%Nx.Tensor{data: %Expr{op: :broadcast, args: [inner, shape, axes]}}, memo) do
    {term, memo} = lower(inner, memo)
    target = shape_list(shape)
    in_shape = shape_list(inner.shape)

    # the inner axes map to the trailing axes: pad with leading 1s and let the
    # elementwise operation that consumes it stretch them
    if axes == Enum.to_list((length(target) - length(in_shape))..(length(target) - 1)//1),
      do: {pad_rank(term, in_shape, length(target)), memo},
      else: refuse("broadcast to inner axes #{inspect(axes)}", "broadcast along leading axes only")
  end

  defp node(%Nx.Tensor{data: %Expr{op: op, args: [a]}, shape: s}, memo) when is_map_key(@unary, op) do
    {ta, memo} = lower(a, memo)
    {check!(T.ew(@unary[op], [ta]), s, op), memo}
  end

  defp node(%Nx.Tensor{data: %Expr{op: op, args: [a, b]}, shape: s}, memo) when is_map_key(@binary, op) do
    {ta, memo} = lower(a, memo)
    {tb, memo} = lower(b, memo)
    rank = tuple_size(s)
    {check!(T.ew(@binary[op], [pad_rank(ta, shape_list(a.shape), rank), pad_rank(tb, shape_list(b.shape), rank)]), s, op), memo}
  end

  # select(a < b, x, y) is the canonical `sel`
  defp node(%Nx.Tensor{data: %Expr{op: :select, args: [%Nx.Tensor{data: %Expr{op: :less, args: [a, b]}}, x, y]}, shape: s}, memo) do
    rank = tuple_size(s)
    {ts, memo} = Enum.map_reduce([a, b, x, y], memo, &lower/2)
    padded = Enum.zip_with(ts, [a, b, x, y], fn t, e -> pad_rank(t, shape_list(e.shape), rank) end)
    {check!(T.ew(:sel, padded), s, :select), memo}
  end

  defp node(%Nx.Tensor{data: %Expr{op: op, args: [x, opts]}, shape: s}, memo) when op in [:sum, :reduce_max] do
    rank = tuple_size(x.shape)

    unless opts[:axes] in [[rank - 1]],
      do: refuse("#{op} over axes #{inspect(opts[:axes])}", "reduce over the last axis only")

    {tx, memo} = lower(x, memo)
    term = T.reduce(if(op == :sum, do: :sum, else: :max), tx)
    # the term keeps the axis (extent 1); without keep_axes the Nx shape drops it
    want = if opts[:keep_axes], do: s, else: List.to_tuple(shape_list(x.shape) |> List.replace_at(-1, 1))
    {check!(term, want, op), memo}
  end

  # dot(x, [last], w, [c]) with w rank 2: `linear` against W[n, k]
  defp node(%Nx.Tensor{data: %Expr{op: :dot, args: [x, [cx], [], w, [cw], []]}, shape: s}, memo) do
    unless tuple_size(w.shape) == 2 and tuple_size(x.shape) in [1, 2] and cx == tuple_size(x.shape) - 1,
      do: refuse("dot of shapes #{inspect(x.shape)} · #{inspect(w.shape)}", "contract the last axis of a rank-1/2 tensor with a rank-2 one")

    {tx, memo} = lower(x, memo)
    {tw, memo} = lower(w, memo)

    tw =
      case {cw, tw} do
        {1, _} -> tw
        {0, {:const, c}} -> T.const(transpose_const(c))
        {0, _} -> T.transpose(tw)
      end

    {check!(T.linear(tx, tw), s, :dot), memo}
  end

  defp node(%Nx.Tensor{data: %Expr{op: op}}, _memo),
    do: refuse("operation #{inspect(op)}", "the certified fragment is: #{Enum.join(Map.keys(@unary) ++ Map.keys(@binary), ", ")}, select(less), dot, sum/reduce_max (last axis), reshape/broadcast of inputs")

  defp leaf_input(pos, shape), do: T.input(:"p#{pos}@#{Enum.join(shape, "x")}", :f32, shape)

  # leaves change shape freely (row-major bytes); anything else must already fit
  defp reshape_leaf!({:input, name, :f32, _}, shape, _why) do
    [pos | _] = name |> Atom.to_string() |> String.trim_leading("p") |> String.split("@")
    leaf_input(String.to_integer(pos), shape)
  end

  defp reshape_leaf!({:const, c}, shape, _why), do: T.const(%{c | shape: shape})
  defp reshape_leaf!({:splat, _} = s, _shape, _why), do: s

  defp reshape_leaf!(term, shape, why) do
    case T.infer(term) do
      {:ok, {:f32, ^shape}} -> term
      {:ok, {:f32, s}} -> drop_kept_axis(term, s, shape) || refuse("#{why} of a computed #{inspect(s)} to #{inspect(shape)}", "reshape inputs, not intermediates")
    end
  end

  # [.., 1] (a kept reduction axis) read as [..]: the same bytes
  defp drop_kept_axis(term, s, shape), do: if(s == shape ++ [1], do: term, else: nil)

  defp pad_rank({:splat, _} = t, _shape, _rank), do: t
  defp pad_rank(t, shape, rank) when length(shape) == rank, do: t
  defp pad_rank(t, shape, rank), do: reshape_leaf!(t, List.duplicate(1, rank - length(shape)) ++ shape, "broadcast")

  defp check!(term, nx_shape, op) do
    want = shape_list(nx_shape)

    case T.infer(term) do
      {:ok, {:f32, ^want}} -> term
      {:ok, {_, got}} -> refuse("#{op}: vapor shape #{inspect(got)} ≠ Nx shape #{inspect(want)}", "align ranks explicitly")
      {:error, r} -> refuse("#{op}: #{Exception.message(r)}", "see Vapor.Algebra.Term")
    end
  end

  defp transpose_const(%Vapor.Tensor{dtype: :f32, shape: [r, c], data: d}) do
    rows = for <<x::binary-4 <- d>>, do: x
    t = rows |> Enum.chunk_every(c) |> Enum.zip_with(& &1) |> List.flatten() |> IO.iodata_to_binary()
    Vapor.Tensor.new(:f32, [c, r], t)
  end

  defp shape_list(s), do: Tuple.to_list(s)

  defp refuse(what, hint), do: raise(ArgumentError, "Vapor.Nx.Compiler cannot certify #{what} (#{hint})")
end
