defmodule Vapor.Emit.Machine do
  @moduledoc """
  The target-independent half of code generation:

      portable IR ──select──▶ machine IR (virtual regs) ──liveness──▶ intervals
        ──linear scan (group factor g)──▶ assignment ──verified checker──▶
        ──encode (prologue · body · epilogue, label fixups)──▶ bitstring

  A backend (`Vapor.Emit.X86`, `Vapor.Emit.ARM`, `Vapor.Emit.RVV`) supplies the
  register model, instruction selection and the bit-level encoders. The group
  factor loop tries the backend's factors from largest to smallest and keeps
  the first one whose allocation is accepted; if none is, the kernel is
  rejected with the pressure point (the cut sweep's signal to split).
  """
  alias Vapor.KIR.{Liveness, RegAlloc}

  @callback isa() :: atom
  @callback files() :: %{atom => RegAlloc.file()}
  @callback group_factors() :: [pos_integer]
  @callback size(atom, pos_integer) :: {atom, pos_integer}
  @callback arg_pin() :: {atom, non_neg_integer}
  @callback select(tuple | atom, map) :: {[tuple], map}
  @callback materialize(atom, tuple) :: [tuple]
  @callback prologue(map) :: binary
  @callback epilogue(map) :: binary
  @callback branch_size(atom) :: pos_integer
  @callback encode_branch(atom, list, integer, (term -> non_neg_integer)) :: binary
  @callback encode(atom, list, (term -> non_neg_integer)) :: binary

  defmodule Code do
    @moduledoc "A compiled kernel: machine code plus the evidence of its allocation."
    @enforce_keys [:isa, :kernel, :g, :policy, :bin]
    defstruct [:isa, :kernel, :g, :policy, :bin, :placements, :callee_used, :instructions,
               text: nil, variant: %{}, loops: %{}]
  end

  def mi(op, defs, uses, attrs \\ []), do: {op, defs, uses, Map.new(attrs)}

  @doc "Allocate a fresh virtual register of a portable kind."
  def fresh(s, kind) do
    {file, size} = s.backend.size(kind, s.g)
    {{:vr, s.next, file, size}, %{s | next: s.next + 1}}
  end

  def label(s, kind \\ :gen), do: {{:l, {kind, s.next}}, %{s | next: s.next + 1}}

  @doc """
  A strip's scalar tail re-executes its body on one element. Every value the
  body *defines* is renamed to a fresh virtual register in the tail copy, so
  the two loops' temporaries are separate live ranges (a single register
  per temporary would make each live interval span both loops).
  """
  def tail_copy(body, s) do
    defined =
      body
      |> Enum.flat_map(fn
        {:vst, _, _, _} -> []
        i when is_tuple(i) and tuple_size(i) >= 2 -> [elem(i, 1)]
        _ -> []
      end)
      |> Enum.filter(&match?({:vr, _, _, _}, &1))
      |> Enum.uniq()

    {map, s} =
      Enum.map_reduce(defined, s, fn {:vr, _, f, sz} = v, s -> {{v, {:vr, s.next, f, sz}}, %{s | next: s.next + 1}} end)

    map = Map.new(map)
    {Enum.map(body, &rename(&1, map)), s}
  end

  defp rename({:vr, _, _, _} = v, map), do: Map.get(map, v, v)
  defp rename(t, map) when is_tuple(t), do: t |> Tuple.to_list() |> Enum.map(&rename(&1, map)) |> List.to_tuple()
  defp rename(l, map) when is_list(l), do: Enum.map(l, &rename(&1, map))
  defp rename(x, _map), do: x

  def select_all(insts, s) do
    {lists, s} = Enum.map_reduce(insts, s, &s.backend.select/2)
    {Enum.concat(lists), s}
  end

  @doc """
  Compile a portable kernel for `backend`. Options: `:policy`
  (`:canonical` | `:fast`), `:g` (force one group factor).
  """
  @spec compile(struct, module, keyword) :: {:ok, %Code{}} | {:error, map}
  def compile(kernel, backend, opts \\ []) do
    policy = Keyword.get(opts, :policy, :canonical)
    gs = if g = opts[:g], do: [g], else: backend.group_factors()
    variants = [kernel | Map.get(kernel.meta, :fallbacks, [])]

    # first variant (then largest group factor) whose allocation is accepted
    for(v <- variants, g <- gs, do: {v, g})
    |> Enum.reduce_while({:error, %{reason: :no_group_factor}}, fn {v, g}, _ ->
      case compile_g(v, backend, g, policy) do
        {:ok, code} -> {:halt, {:ok, %{code | variant: v.meta}}}
        {:error, info} -> {:cont, {:error, Map.put(info, :g, g)}}
      end
    end)
  end

  defp compile_g(kernel, backend, g, policy) do
    ir = to_machine(kernel.code, backend, g)
    {pin_file, pin_reg} = backend.arg_pin()
    argp = {:vr, :argp, pin_file, 1}

    s0 = %{backend: backend, g: g, policy: policy, next: Vapor.KIR.max_id(kernel.code) + 1,
           consts: %{}, argp: argp, vcfg: nil}

    {body, s} = select_all(ir, s0)
    consts = Enum.flat_map(Enum.sort(s.consts), fn {k, r} -> backend.materialize(k, r) end)
    code = [mi(:entry, [argp], [])] ++ consts ++ body
    intervals = Liveness.intervals(code)

    with {:ok, alloc} <- RegAlloc.allocate(intervals, backend.files(), %{argp => pin_reg}),
         placements = RegAlloc.placements(intervals, alloc.assign),
         :ok <- check(placements, backend.files()) do
      {bin, text} = encode(code, alloc, backend)

      {:ok,
       %Code{isa: backend.isa(), kernel: kernel.name, g: g, policy: policy, bin: bin, text: text,
             placements: placements, callee_used: alloc.callee_used,
             instructions: div(byte_size(bin), 4), loops: loops(code)}}
    end
  end

  @doc """
  Static machine-instruction count of every loop body (label → back-edge),
  the counted input of the arbiter's issue roof. Bundles count per member.
  """
  def loops(code) do
    indexed = Enum.with_index(code)

    for {{_, _, _, %{label: l}}, i} <- indexed, l != nil,
        {{_, _, _, a}, j} <- indexed, j > i, a[:br] == l, into: %{} do
      body = code |> Enum.slice((i + 1)..j) |> Enum.map(&weight/1) |> Enum.sum()
      {l, body}
    end
  end

  defp weight({_, _, _, a}) do
    cond do
      a[:label] != nil or a[:ret] == true -> 0
      a[:bundle] -> length(a[:bundle])
      true -> 1
    end
  end

  # Portable kinds → machine files and group sizes (recursing into strips).
  defp to_machine({:vr, id, kind}, backend, g) when is_atom(kind) do
    {file, size} = backend.size(kind, g)
    {:vr, id, file, size}
  end

  defp to_machine(list, b, g) when is_list(list), do: Enum.map(list, &to_machine(&1, b, g))

  defp to_machine(t, b, g) when is_tuple(t),
    do: t |> Tuple.to_list() |> Enum.map(&to_machine(&1, b, g)) |> List.to_tuple()

  defp to_machine(x, _b, _g), do: x

  @doc """
  Independent validation of an allocation by the Lean-verified checker
  (`Vapor.Extracted.check_alloc/3`, extracted from `proofs/Vapor/RegAlloc.lean`).
  """
  def check(placements, files) do
    Enum.reduce_while(placements, :ok, fn {file, ps}, :ok ->
      %{count: count, order: order} = Map.fetch!(files, file)
      reserved = Enum.to_list(0..(count - 1)) -- order
      # Lean's (start, stop, reg, size) is the nested pair {a, {b, {p, s}}}
      quads = for {a, b, p, sz} <- ps, do: {a, {b, {p, sz}}}

      if Vapor.Extracted.check_alloc(count, reserved, quads),
        do: {:cont, :ok},
        else: {:halt, {:error, %{reason: :checker_rejected, file: file}}}
    end)
  end

  # ------------------------------------------------------------- encoding --

  defp encode(code, alloc, backend) do
    assign = alloc.assign

    r = fn
      {:vr, _, _, _} = v -> Map.fetch!(assign, v)
      {:sub, v, k} -> Map.fetch!(assign, v) + k
      {:phys, n} -> n
    end

    pro = backend.prologue(alloc.callee_used)
    epi = backend.epilogue(alloc.callee_used)

    # pass 1: encode everything but branches, record label offsets and the
    # pool displacements (x86 RIP-relative constants) to patch
    {items, labels, off, fixes} =
      Enum.reduce(code, {[], %{}, byte_size(pro), []}, fn {op, _d, _u, a}, {items, labels, off, fx} ->
        cond do
          a[:label] != nil -> {items, Map.put(labels, a[:label], off), off, fx}
          a[:br] != nil -> {[{:br, op, a[:br], off, a[:o] || []} | items], labels, off + backend.branch_size(op), fx}
          a[:ret] == true -> {[epi | items], labels, off + byte_size(epi), fx}
          op == :entry -> {items, labels, off, fx}
          a[:bundle] -> enc(items, labels, off, fx, Enum.map(a[:bundle], &backend.encode(op, &1, r)))
          true -> enc(items, labels, off, fx, [backend.encode(op, a[:o] || [], r)])
        end
      end)

    body =
      items
      |> Enum.reverse()
      |> Enum.map(fn
        {:br, op, l, off, o} -> backend.encode_branch(op, o, Map.fetch!(labels, l) - off, r)
        bin -> bin
      end)

    text = IO.iodata_to_binary([pro | body])
    {append_pool(text, off, fixes, backend), byte_size(text)}
  end

  # items carry plain binaries; fixups become absolute
  # {field_pos, insn_start, insn_end, bits, kind}
  defp enc(items, labels, off, fx, results) do
    {bin, fx} =
      Enum.reduce(results, {<<>>, fx}, fn
        {:fix, b, locals}, {acc, fx} ->
          start = off + byte_size(acc)

          fixes =
            Enum.map(locals, fn
              {pos, bits} -> {start + pos, start, start + byte_size(b), bits, :rel32}
              {pos, bits, kind} -> {start + pos, start, start + byte_size(b), bits, kind}
            end)

          {acc <> b, fixes ++ fx}

        b, {acc, fx} ->
          {acc <> b, fx}
      end)

    {[bin | items], labels, off + byte_size(bin), fx}
  end

  defp append_pool(bin, _end, [], _backend), do: bin

  defp append_pool(bin, _end, fixes, backend) do
    align = backend.pool_align()
    pad = rem(align - rem(byte_size(bin), align), align)
    consts = fixes |> Enum.map(&elem(&1, 3)) |> Enum.uniq() |> Enum.sort()
    base = byte_size(bin) + pad
    entry = byte_size(backend.pool_entry(0))
    at = consts |> Enum.with_index() |> Map.new(fn {c, i} -> {c, base + i * entry} end)

    patched =
      Enum.reduce(fixes, bin, fn {pos, insn_start, insn_end, bits, kind}, acc ->
        target = Map.fetch!(at, bits)

        case kind do
          # x86 RIP-relative disp32, relative to the end of the instruction
          :rel32 ->
            <<pre::binary-size(pos), _::32, post::binary>> = acc
            pre <> <<target - insn_end::signed-32-little>> <> post

          # A64 LDR (literal): imm19 word offset from the instruction, bits 23:5
          :lit19 ->
            <<pre::binary-size(pos), word::32-little, post::binary>> = acc
            imm19 = Bitwise.band(div(target - insn_start, 4), 0x7FFFF)
            pre <> <<Bitwise.bor(word, Bitwise.bsl(imm19, 5))::32-little>> <> post
        end
      end)

    IO.iodata_to_binary([patched, :binary.copy(<<backend.pad_byte()>>, pad) | Enum.map(consts, &backend.pool_entry/1)])
  end
end
