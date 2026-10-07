defmodule Vapor.ZK do
  @moduledoc """
  Zero-knowledge proofs of integer inference: a quantized network compiled
  to a rank-1 constraint system (R1CS) over a prime field, a witness that
  satisfies it, and both written in the formats of circom/snarkjs
  (`.r1cs`, `.wtns`) — so any Groth16 or PLONK prover of that ecosystem can
  prove, and any verifier (including the Solidity one snarkjs exports) can
  check, that *this* public model mapped a private input to a public output.

      model = [{:linear, w1, b1}, :relu, {:linear, w2, b2}]   # w1 int8, weights public
      c = Vapor.ZK.compile(model, k: 16)
      {:ok, wit} = Vapor.ZK.witness(c, x)                     # x : int8, kept private
      :ok = Vapor.ZK.check(c, wit)
      Vapor.ZK.write_r1cs(c, "m.r1cs"); Vapor.ZK.write_wtns(c, wit, "m.wtns")

  ## Why the integer fragment, and why public weights

  *Field arithmetic is integer arithmetic.* `+` and `×` of a field are
  exactly those of the integers as long as no value reaches `p/2` (the
  signed embedding, `Vapor.Field`). vapor's int8 contraction is already
  *proved* (Lean, `integer_parity`) never to wrap 32-bit accumulators under
  its admissibility bound; the same bound, checked against `p/2`, makes the
  kernel's result and the circuit's the same number — so the witness can be
  computed by the certified `gemm_i8` kernels on any substrate
  (`witness/3`, `certified: true`) and the circuit merely checks it.
  Floating point, by contrast, is not field arithmetic: every rounding would
  cost a range proof. That is why f32 programs are not compiled here.

  *Weights are constants of the circuit.* A product by a constant is free in
  R1CS (it is part of a linear combination): a hidden dense layer costs no
  constraint at all and the last one one per output; the cost of a proof is
  in the non-linearities (ReLU: a bit decomposition, `B + 3` constraints per
  neuron) and in the input range check (9 per private int8 input).

  *What is proved:* there exists an int8 vector `x` with `model(x) = y` for
  the public `y` — the range check makes `x ∈ int8ᵏ` a fact of the circuit,
  and then every intermediate value is the integer the model computes (each
  stays below `p/2` and below the ReLU decompositions' range, checked at
  compile time; Lean: `field_parity`). Constants also *bind the model*: the verification key is derived
  from the circuit, and the circuit's digest (`digest/1`) names the weights.
  A circuit with weights as private witnesses would prove only that *some*
  weights produce the output — useless without an in-circuit commitment to
  a published model, and even then a commitment says which weights, not that
  they are the real model's (the "hollow model" problem). Private weights
  are therefore not offered.

  The proof adds two things to what vapor's receipts already give (anyone
  with the weights can re-derive an output bit for bit): the input can stay
  private, and verification does not re-run the model.
  """
  import Bitwise
  alias Vapor.{Field, Tensor}
  alias Vapor.Algebra.Term, as: T

  defstruct [:field, :layers, :k, :n_out, :input, constraints: [], n_wires: 0, bounds: [], plan: [], in_bits: []]

  @type t :: %__MODULE__{}

  @doc """
  Compile `model` — `{:linear, w, b}` (w: rows of integers, `n × k`; b: n
  integers) and `:relu` — for inputs of `k` int8 values. Options: `field:`
  (default `:bn254`, snarkjs' curve), `input: :private | :public`.

  Raises if some intermediate value could reach `p/2` (the signed embedding
  would no longer be faithful) — the bound is computed exactly from the
  weights and the int8 input range.
  """
  def compile(model, opts) do
    f = Field.get(Keyword.get(opts, :field, :bn254))
    k = Keyword.fetch!(opts, :k)
    n_out = model |> Enum.filter(&match?({:linear, _, _}, &1)) |> List.last() |> then(fn {:linear, w, _} -> length(w) end)
    input = Keyword.get(opts, :input, :private)

    # wire layout of circom: 0 = one, outputs, public inputs, private inputs, internals
    outs = Enum.to_list(1..n_out)
    ins = Enum.to_list((n_out + 1)..(n_out + k))
    st = %{f: f, next: n_out + k + 1, cs: [], bounds: []}

    # a private input is int8 *because the circuit says so*: x + 128 is the sum
    # of 8 boolean bits. Without this a prover could choose any field values
    # for x — every hidden activation, hence every output, would be reachable.
    # (A public input is checked by whoever supplies it.)
    {in_bits, st} =
      if input == :private do
        Enum.map_reduce(ins, st, fn w, st ->
          bits = Enum.to_list(st.next..(st.next + 7))
          st = Enum.reduce(bits, %{st | next: st.next + 8}, fn bw, st -> add(st, %{bw => 1}, %{bw => 1}, %{bw => 1}) end)
          sum = bits |> Enum.with_index() |> Map.new(fn {bw, i} -> {bw, 1 <<< i} end)
          {hd(bits), add(st, sum, %{0 => 1}, %{w => 1, 0 => 128})}
        end)
      else
        {[], st}
      end

    # every value is a linear combination of wires ({wire => coeff}) with an
    # integer bound on its magnitude
    vals = for w <- ins, do: {%{w => 1}, 128}
    {vals, st, layers} = Enum.reduce(model, {vals, st, []}, &layer/2)
    last = List.last(layers)

    st =
      Enum.zip(vals, outs)
      |> Enum.reduce(st, fn {{lc, _}, o}, st -> add(st, lc, %{0 => 1}, %{o => 1}) end)

    if last != :linear, do: raise(ArgumentError, "the model must end with a linear layer (its outputs are the public signals)")

    %__MODULE__{field: f, layers: model, k: k, n_out: n_out, input: input, constraints: Enum.reverse(st.cs),
                n_wires: st.next, bounds: Enum.reverse(st.bounds), plan: layers, in_bits: in_bits}
  end

  defp layer({:linear, w, b}, {vals, st, ls}) do
    unless length(b) == length(w) and Enum.all?(w, &(length(&1) == length(vals))),
      do: raise(ArgumentError, "linear layer #{length(w)}×#{length(hd(w))} does not take #{length(vals)} inputs")

    out =
      Enum.zip_with(w, b, fn row, bias ->
        lc = Enum.zip(row, vals) |> Enum.reduce(%{0 => bias}, fn {c, {v, _}}, acc -> Enum.reduce(v, acc, fn {wire, cv}, acc -> lc_add(acc, wire, c * cv) end) end)
        bound = Enum.zip(row, vals) |> Enum.reduce(abs(bias), fn {c, {_, m}}, acc -> acc + abs(c) * m end)
        {lc, bound}
      end)

    m = out |> Enum.map(&elem(&1, 1)) |> Enum.max()
    if 2 * m >= st.f.p, do: raise(ArgumentError, "values up to #{m} exceed p/2 in #{st.f.name}: the circuit would not compute the integers")
    {out, %{st | bounds: [m | st.bounds]}, ls ++ [:linear]}
  end

  # ReLU of z, |z| < 2^B: bits of u = z + 2^B (B + 1 of them, each boolean),
  # their weighted sum equal to u, and h = b_B · z (b_B = 1 exactly when z ≥ 0)
  defp layer(:relu, {vals, st, ls}) do
    {out, st} =
      Enum.map_reduce(vals, st, fn {lc, bound}, st ->
        nb = bits_for(bound)

        # the decomposition is unique only if no second representative of
        # z + 2^nb (mod p) fits in nb + 1 bits
        if 1 <<< (nb + 1) > st.f.p,
          do: raise(ArgumentError, "a ReLU over values up to #{bound} needs #{nb + 1} bits, more than #{st.f.name} holds uniquely")
        {bits, st} = Enum.map_reduce(0..nb, st, fn _, st -> {st.next, %{st | next: st.next + 1}} end)
        st = Enum.reduce(bits, st, fn bw, st -> add(st, %{bw => 1}, %{bw => 1}, %{bw => 1}) end)
        sum = bits |> Enum.with_index() |> Map.new(fn {bw, i} -> {bw, 1 <<< i} end)
        st = add(st, sum, %{0 => 1}, lc_add(lc, 0, 1 <<< nb))
        h = st.next
        st = add(%{st | next: h + 1}, %{List.last(bits) => 1}, lc, %{h => 1})
        {{{%{h => 1}, bound}, {hd(bits), nb, h}}, st}
      end)

    {Enum.map(out, &elem(&1, 0)), st, ls ++ [{:relu, Enum.map(out, &elem(&1, 1))}]}
  end

  defp bits_for(bound), do: max(1, length(Integer.digits(bound, 2)))

  defp lc_add(lc, wire, c), do: Map.update(lc, wire, c, &(&1 + c))

  defp add(st, a, b, c) do
    norm = fn lc -> lc |> Enum.map(fn {w, v} -> {w, Field.from_int(st.f, v)} end) |> Enum.reject(&(elem(&1, 1) == 0)) |> Map.new() end
    %{st | cs: [{norm.(a), norm.(b), norm.(c)} | st.cs]}
  end

  @doc "Number of constraints."
  def size(%__MODULE__{constraints: cs}), do: length(cs)

  @doc "The canonical digest of the circuit — of the model it is made from."
  def digest(%__MODULE__{} = c), do: Vapor.Canonical.hex_digest({:vapor_r1cs, 1, c.field.p, c.n_wires, c.n_out, c.k, c.input, c.constraints})

  # ------------------------------------------------------------ witness --

  @doc """
  The witness for input `x` (k integers in −128..127): every wire's value.
  With `certified: true` (default), the first layer's contraction is
  computed by vapor's certified `gemm_i8` program (`first_layer_program/1`,
  run with `Vapor.run/3` — any substrate, the same bits); the rest is exact
  integer arithmetic. Returns `{:ok, [field elements], outputs}`.
  """
  def witness(%__MODULE__{} = c, x, opts \\ []) do
    # (`unchecked: true` builds the would-be witness of an out-of-range input,
    # for tests that the circuit rejects it)
    unless length(x) == c.k and (opts[:unchecked] || Enum.all?(x, &(&1 in -128..127))), do: raise(ArgumentError, "x must be #{c.k} int8 values")
    f = c.field

    first =
      if Keyword.get(opts, :certified, true) do
        {:ok, compiled} = Keyword.get_lazy(opts, :compiled, fn -> Vapor.compile(first_layer_program(c)) end) |> ok()
        {:ok, res, _} = Vapor.run(compiled, %{x: Tensor.from_list(:s8, [1, c.k], x)})
        {:linear, _, b} = hd(c.layers)
        Enum.zip_with(Tensor.to_list(res.outputs.z), b, &(&1 + &2))
      end

    {vals, _pre} = eval(c.layers, x, first)
    wires = %{0 => 1} |> Map.merge(Map.new(Enum.with_index(x, c.n_out + 1), fn {v, i} -> {i, v} end))
    wires = fill(c, x, wires)
    in_bits = for {xi, start} <- Enum.zip(x, c.in_bits), i <- 0..7, into: %{}, do: {start + i, (xi + 128) >>> i &&& 1}
    wires = Map.merge(wires, in_bits)
    outs = Enum.with_index(vals, 1) |> Map.new(fn {v, i} -> {i, v} end)
    wires = Map.merge(wires, outs)

    {:ok, for(i <- 0..(c.n_wires - 1), do: Field.from_int(f, Map.fetch!(wires, i))), vals}
  end

  defp ok({:ok, _} = o), do: o
  defp ok(%Vapor.Compiled{} = c), do: {:ok, c}

  @doc "The certified vapor program of the first contraction: `z = gemm_i8(x, W₁)` (bias added outside)."
  def first_layer_program(%__MODULE__{layers: [{:linear, w, _} | _], k: k}) do
    unless Enum.all?(List.flatten(w), &(&1 in -128..127)), do: raise(ArgumentError, "first-layer weights must be int8 for gemm_i8")
    wt = Tensor.from_list(:s8, [length(w), k], List.flatten(w))
    Vapor.Program.new(z: T.gemm_i8(T.input(:x, :s8, [1, k]), T.const(wt)))
  end

  # exact integer evaluation; `first` (from the certified kernel) replaces
  # the first layer's pre-activation when given — and must agree with it
  defp eval(layers, x, first) do
    Enum.reduce(Enum.with_index(layers), {x, []}, fn
      {{:linear, w, b}, 0}, {v, ints} ->
        exact = Enum.zip_with(w, b, fn row, bias -> bias + dot(row, v) end)
        if first && first != exact, do: raise("certified kernel disagrees with the exact contraction")
        {exact, ints ++ [exact]}

      {{:linear, w, b}, _}, {v, ints} ->
        z = Enum.zip_with(w, b, fn row, bias -> bias + dot(row, v) end)
        {z, ints ++ [z]}

      {:relu, _}, {v, ints} ->
        {Enum.map(v, &max(&1, 0)), ints}
    end)
  end

  defp dot(row, v), do: Enum.zip_with(row, v, &(&1 * &2)) |> Enum.sum()

  # the internal wires, from the gadget layout compile/2 recorded
  defp fill(c, x, wires) do
    {_v, wires} =
      Enum.zip(c.layers, c.plan)
      |> Enum.reduce({x, wires}, fn
        {{:linear, w, b}, :linear}, {v, wires} ->
          {Enum.zip_with(w, b, fn row, bias -> bias + dot(row, v) end), wires}

        {:relu, {:relu, gadgets}}, {v, wires} ->
          wires =
            Enum.zip(v, gadgets)
            |> Enum.reduce(wires, fn {z, {start, nb, h}}, wires ->
              u = z + (1 <<< nb)
              bits = for i <- 0..nb, into: %{}, do: {start + i, u >>> i &&& 1}
              wires |> Map.merge(bits) |> Map.put(h, max(z, 0))
            end)

          {Enum.map(v, &max(&1, 0)), wires}
      end)

    wires
  end

  # ------------------------------------------------------------- check --

  @doc "`:ok` if the witness satisfies every constraint, else `{:error, {:constraint, i}}`."
  def check(%__MODULE__{field: f, constraints: cs}, wit) do
    w = List.to_tuple(wit)
    ev = fn lc -> Enum.reduce(lc, 0, fn {i, c}, acc -> Field.add(f, acc, Field.mul(f, c, elem(w, i))) end) end

    cs
    |> Enum.with_index()
    |> Enum.find_value(:ok, fn {{a, b, cc}, i} -> if Field.mul(f, ev.(a), ev.(b)) != ev.(cc), do: {:error, {:constraint, i}} end)
  end

  # ------------------------------------------------------------ formats --

  @doc "The circuit in iden3's binary R1CS format (circom, snarkjs)."
  def r1cs(%__MODULE__{field: f} = c) do
    n8 = Field.bytes(f.p)
    {pub_in, prv_in} = if c.input == :public, do: {c.k, 0}, else: {0, c.k}

    header = <<n8::little-32>> <> Field.to_bytes(f, f.p, n8) <>
               <<c.n_wires::little-32, c.n_out::little-32, pub_in::little-32, prv_in::little-32, c.n_wires::little-64, length(c.constraints)::little-32>>

    lc = fn m ->
      m = Enum.sort(m)
      [<<length(m)::little-32>> | Enum.map(m, fn {wire, v} -> <<wire::little-32>> <> Field.to_bytes(f, v, n8) end)]
    end

    cons = c.constraints |> Enum.map(fn {a, b, cc} -> [lc.(a), lc.(b), lc.(cc)] end) |> IO.iodata_to_binary()
    labels = for(i <- 0..(c.n_wires - 1), into: <<>>, do: <<i::little-64>>)
    "r1cs" <> <<1::little-32, 3::little-32>> <> section(1, header) <> section(2, cons) <> section(3, labels)
  end

  @doc "A witness in iden3's binary format (`.wtns`)."
  def wtns(%__MODULE__{field: f}, wit) do
    n8 = Field.bytes(f.p)
    header = <<n8::little-32>> <> Field.to_bytes(f, f.p, n8) <> <<length(wit)::little-32>>
    "wtns" <> <<2::little-32, 2::little-32>> <> section(1, header) <> section(2, IO.iodata_to_binary(Enum.map(wit, &Field.to_bytes(f, &1, n8))))
  end

  def write_r1cs(c, path), do: File.write!(path, r1cs(c))
  def write_wtns(c, wit, path), do: File.write!(path, wtns(c, wit))

  defp section(type, body), do: <<type::little-32, byte_size(body)::little-64>> <> body
end
