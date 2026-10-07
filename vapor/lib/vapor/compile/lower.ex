defmodule Vapor.Compile.Lower do
  @moduledoc """
  Section 2.2 — lowering a program to a kernel schedule via the **cut sweep**.

  Nodes are visited in topological order. Elementwise nodes accumulate into a
  fused region; a candidate region is *admitted* only if every target
  backend compiles it — i.e. the linear-scan allocator finds a register
  assignment at some group factor and the Lean-verified checker accepts it.
  When admitting the next node would fail, the sweep closes the region there
  (the value crossing the cut is materialised in memory) and starts a new
  one. Consequently every emitted region is spill-free by construction: the
  backends contain no spill code at all, and no region reaches emission
  without a checked allocation.

  Contractions are regions of their own (`qgemv` = sub-block sums + GEMV,
  `gemm_i8` = one kernel).
  """
  alias Vapor.{Compiled, Program, Rejection, Tensor}
  alias Vapor.Algebra.Term
  alias Vapor.KIR.Kernels
  alias Vapor.Emit.{Link, Machine}
  alias Vapor.Quant.Sb4

  @default_targets [Vapor.Emit.X86, Vapor.Emit.X86.AVX512, Vapor.Emit.ARM, Vapor.Emit.RVV]

  @spec lower(Program.t(), keyword) :: {:ok, Compiled.t()} | {:error, Rejection.t()}
  def lower(%Program{} = p, opts \\ []) do
    policy = Keyword.get(opts, :policy, :canonical)
    targets = Keyword.get(opts, :targets, @default_targets)
    r = Program.resolver(p)
    bound = Program.bound(p)
    roots = Enum.map(p.outputs, &r.(elem(&1, 1)))
    order = Program.order(p)

    # a ref to a constant is a constant leaf; a ref to a node shares its slot
    {refs, terms} = Enum.split_with(order, &(r.(&1) != &1))

    slots =
      terms
      |> Enum.with_index()
      |> Map.new(fn {t, i} ->
        {:ok, {dt, shape}} = Term.infer(t)
        {t, %{id: i, dtype: dt, shape: shape, role: role(t, bound)}}
      end)

    # dead-code elimination: only nodes some output depends on are lowered
    # (a let-bound value nobody reads is not computed — and an elementwise
    # region with no output would have nowhere to put its result)
    live = reachable(roots, r)
    nodes = terms |> Enum.reject(&Term.leaf?/1) |> Enum.filter(&MapSet.member?(live, &1))
    uses = uses_outside(nodes, roots, r)
    {slots, inplace} = alias_updates(nodes, slots, uses, roots, r)
    slots = Enum.reduce(refs, slots, fn ref, acc -> Map.put(acc, ref, acc[r.(ref)]) end)

    with :ok <- paged_inplace(nodes, slots),
         {:ok, regions} <- sweep(nodes, slots, uses, targets, policy, r) do
      build(p, slots, regions, targets, policy, inplace, r)
    end
  end

  defp reachable(roots, r) do
    Enum.reduce(roots, MapSet.new(), fn t, seen -> mark(r.(t), seen, r) end)
  end

  defp mark(t, seen, r) do
    if MapSet.member?(seen, t),
      do: seen,
      else: Enum.reduce(kids(t, r), MapSet.put(seen, t), &mark(&1, &2, r))
  end

  # A functional cache update whose old cache is read by nobody else (and is
  # neither a constant nor a program output) reuses the old cache's slot: the
  # kernel writes rows in place instead of copying the whole cache. Chains of
  # updates collapse onto one buffer.
  defp alias_updates(nodes, slots, uses, roots, r) do
    Enum.reduce(nodes, {slots, MapSet.new()}, fn
      n, {slots, inplace} when elem(n, 0) in [:kv_write, :kv_write_paged] ->
        cache = r.(elem(n, 1))

        dead? =
          case {slots[cache].role, Map.get(uses, cache)} do
            {{:const, _}, _} -> false
            {{:input, _}, _} -> only_reader?(cache, n, nodes, r) and cache not in roots
            {:tmp, {users, root?}} -> not root? and MapSet.equal?(users, MapSet.new([n]))
          end

        if dead?,
          do: {Map.put(slots, n, slots[cache]), MapSet.put(inplace, slots[cache].id)},
          else: {slots, inplace}

      _, acc ->
        acc
    end)
  end

  # a paged pool has no copying form: its old contents must be dead
  defp paged_inplace(nodes, slots) do
    case Enum.find(nodes, &(match?({:kv_write_paged, _, _, _, _, _, _}, &1) and slots[&1].id != slots[elem(&1, 1)].id)) do
      nil -> :ok
      n -> {:error, Rejection.new(n, "the paged pool is updated in place (no other reader of its old contents)",
                                  "read the updated pool, not the old one")}
    end
  end

  defp only_reader?(leaf, n, nodes, r),
    do: Enum.all?(nodes, fn m -> m == n or leaf not in kids(m, r) end)

  # children with refs resolved: the DAG the bindings denote
  defp kids(n, r), do: Enum.map(Term.children(n), r)

  defp role({:input, name, _, _}, bound) do
    case Map.get(bound, name) do
      {:const, _} = c -> role(c, bound)
      nil -> {:input, name}
    end
  end

  defp role({:const, %Tensor{dtype: :sb4} = t}, _), do: {:const, Sb4.to_exec(t)}
  defp role({:const, t}, _), do: {:const, t}
  defp role(_, _), do: :tmp

  # for each node: the set of consumers (other nodes) and whether it is an output
  defp uses_outside(nodes, roots, r) do
    base = Map.new(nodes, &{&1, MapSet.new()})

    nodes
    |> Enum.reduce(base, fn n, acc ->
      Enum.reduce(kids(n, r), acc, fn c, acc ->
        if Map.has_key?(acc, c), do: Map.update!(acc, c, &MapSet.put(&1, n)), else: acc
      end)
    end)
    |> Map.new(fn {n, users} -> {n, {users, n in roots}} end)
  end

  # ---------------------------------------------------------------- sweep --

  defp sweep(nodes, slots, uses, targets, policy, r) do
    nodes
    |> Enum.reduce_while({:ok, [], nil}, fn node, {:ok, done, open} ->
      case node do
        {:ew, _, _} ->
          candidate = if open, do: open ++ [node], else: [node]

          cond do
            open != nil and same_shape?(open, node, slots) and admitted?(candidate, slots, uses, targets, policy, r) ->
              {:cont, {:ok, done, candidate}}

            admitted?([node], slots, uses, targets, policy, r) ->
              {:cont, {:ok, close(done, open), [node]}}

            match?({:error, :broadcast}, ew_spec([node], slots, uses, r)) ->
              {:halt, {:error, Rejection.new(node, "broadcast classes [R,C] ⊕ {[R,C], [1,C], [R,1], scalar}",
                                             "reshape the operand to one of the supported classes")}}

            true ->
              {:halt, {:error, Rejection.new(node, "single elementwise node allocatable on every target",
                                             "split the operator; reduce operand count")}}
          end

        other ->
          {:cont, {:ok, [{:node, other} | close(done, open)], nil}}
      end
    end)
    |> case do
      {:ok, done, open} -> {:ok, Enum.reverse(close(done, open))}
      err -> err
    end
  end

  defp close(done, nil), do: done
  defp close(done, open), do: [{:ew, open} | done]

  defp same_shape?(open, node, slots),
    do: slots[hd(open)].shape == slots[node].shape

  defp admitted?(region, slots, uses, targets, policy, r) do
    case ew_spec(region, slots, uses, r) do
      {:error, _} -> false
      %{spec: spec} -> Enum.all?(targets, &match?({:ok, _}, Machine.compile(Kernels.ew(spec), &1, policy: policy)))
    end
  end

  @doc false
  def ew_spec(region, slots, uses, r) do
    idx = region |> Enum.with_index() |> Map.new()
    shape = slots[hd(region)].shape

    ext =
      region
      |> Enum.flat_map(&kids(&1, r))
      |> Enum.reject(&Map.has_key?(idx, &1))
      |> Enum.uniq()

    ext_idx = ext |> Enum.with_index() |> Map.new()
    classes = Enum.map(ext, &classify(slots[&1].shape, shape))

    # canonical functions are inlined as their Vapor.Canon microprograms;
    # temporaries are numbered densely across the whole region
    {ops, res, _k} =
      Enum.reduce(region, {[], %{}, 0}, fn {:ew, op, args} = n, {ops, res, k} ->
        operands =
          Enum.map(args, fn
            {:splat, _} = sp -> sp
            a -> Map.get(res, r.(a)) || {:in, ext_idx[r.(a)]}
          end)

        {new_ops, r, k} = Vapor.Canon.expand(op, operands, k)
        {ops ++ new_ops, Map.put(res, n, r), k}
      end)

    outs =
      Enum.filter(region, fn n ->
        {users, root?} = uses[n]
        root? or Enum.any?(users, &(not Map.has_key?(idx, &1)))
      end)

    if :error in classes do
      {:error, :broadcast}
    else
      # all-full (or scalar) regions run as one flat row: widest strips
      flat = Enum.all?(classes, &(&1 in [:full, :scalar]))
      out_id = slots[hd(region)].id

      %{spec: %{inputs: classes, outputs: Enum.map(outs, &res[&1]), ops: ops},
        in_slots: Enum.map(ext, &slots[&1].id),
        out_slots: Enum.map(outs, &slots[&1].id),
        extent: if(flat, do: [{:imm, 1}, {:numel, out_id}], else: [{:rows, out_id}, {:cols, out_id}])}
    end
  end

  # operand class relative to the region's iteration space [R, C]
  defp classify(s, s), do: :full

  defp classify(s, shape) do
    {lead, [last]} = Enum.split(s, -1)
    {rlead, [_rlast]} = Enum.split(shape, -1)

    cond do
      Enum.all?(s, &(&1 == 1)) -> :scalar
      Enum.all?(lead, &(&1 == 1)) and last == List.last(shape) -> :row
      last == 1 and lead == rlead -> :col
      true -> :error
    end
  end

  # ---------------------------------------------------------------- build --

  defp build(p, slots, regions, targets, policy, inplace, r) do
    uses = uses_outside(Enum.flat_map(regions, &region_nodes/1), Enum.map(p.outputs, &r.(elem(&1, 1))), r)

    {kernels, schedule, scratch} =
      Enum.reduce(regions, {%{}, [], []}, fn region, {ks, sched, scratch} ->
        {new_ks, calls, new_scratch} = region_calls(region, slots, uses, r, map_size(slots) + length(scratch))
        {Map.merge(ks, new_ks), sched ++ calls, scratch ++ new_scratch}
      end)

    with {:ok, code} <- emit_all(kernels, targets, policy) do
      slot_table =
        slots
        |> Map.values()
        |> Enum.concat(scratch)
        |> Map.new(&{&1.id, &1})

      {:ok,
       %Compiled{
         program: p,
         policy: policy,
         slots: slot_table,
         kernels: kernels,
         schedule: schedule,
         code: code,
         outputs: Enum.map(p.outputs, fn {name, t} -> {name, slots[t].id} end),
         state: Enum.map(p.state, fn {in_name, out_name} ->
           {:input, _, _, _} = inp = Enum.find(Map.keys(slots), &match?({:input, ^in_name, _, _}, &1))
           {_, out_t} = List.keyfind(p.outputs, out_name, 0)
           {slots[inp].id, slots[out_t].id}
         end),
         regions: Enum.map(regions, &describe/1),
         inplace: inplace,
         spirv:
           for({k, _} <- kernels, bin = Vapor.Emit.SpirvKernels.compile(k, policy), bin != nil, into: %{}, do: {k, bin})
       }}
    end
  end

  defp region_nodes({:ew, ns}), do: ns
  defp region_nodes({:node, n}), do: [n]

  defp describe({:ew, ns}), do: {:ew, length(ns)}
  defp describe({:node, {:reduce, op, _}}), do: {{:reduce, op}, 1}
  defp describe({:node, t}), do: {elem(t, 0), 1}

  defp region_calls({:ew, region}, slots, uses, r, _next) do
    %{spec: spec, in_slots: ins, out_slots: outs, extent: extent} = ew_spec(region, slots, uses, r)
    key = {:ew, spec}

    # partition: rows of a 2-D region, or elements of a flat one; each
    # operand advances by its class's stride (broadcast operands stay put)
    out = List.last(outs)
    step = case extent do
      [{:imm, 1}, _] -> %{full: {:bytes, 4}, col: nil, row: nil, scalar: nil}
      _ -> %{full: {:last_bytes, out}, col: {:bytes, 4}, row: nil, scalar: nil}
    end

    ptrs =
      (Enum.zip(spec.inputs, ins) |> Enum.map(fn {cl, _} -> step[cl] end)) ++ Enum.map(outs, fn _ -> step.full end)

    split =
      split(if(match?([{:imm, 1}, _], extent), do: 1, else: 0), if(match?([{:imm, 1}, _], extent), do: 64, else: 1),
            ptrs |> Enum.with_index(2) |> Enum.reject(&(elem(&1, 0) == nil)) |> Enum.map(fn {st, i} -> {i, st} end))

    {%{key => Kernels.ew(spec)},
     [%{kernel: key, args: extent ++ Enum.map(ins ++ outs, &{:slot, &1}), split: split}], []}
  end


  defp region_calls({:node, {:reduce, op, x} = n}, slots, _uses, _r, _next) do
    key = {:reduce, op}
    xs = slots[x].id

    {%{key => Kernels.reduce(op)},
     [%{kernel: key, args: [{:slot, slots[n].id}, {:slot, xs}, {:rows, xs}, {:cols, xs}],
        split: split(2, 1, [{0, {:bytes, 4}}, {1, {:last_bytes, xs}}])}], []}
  end

  defp region_calls({:node, {:linear, x, w} = n}, slots, _uses, _r, _next) do
    [ws, xs] = [slots[w].id, slots[x].id]
    {key, kernel} = if slots[w].dtype == :bf16, do: {:gemv_bf16, Kernels.gemv_bf16()}, else: {:gemv_f32, Kernels.gemv_f32()}

    {%{key => kernel},
     [%{kernel: key,
        args: [{:slot, slots[n].id}, {:slot, ws}, {:slot, xs}, {:dim, ws, 0}, {:dim, ws, 1}, {:rows, xs}, {:dim, ws, 0}],
        split: split(3, 4, [{0, {:bytes, 4}}, {1, {:last_bytes, ws}}])}], []}
  end

  # row-predicated: the dense kernel's instructions on active rows, +0 elsewhere
  defp region_calls({:node, {:linear_masked, x, w, mk} = n}, slots, _uses, _r, _next) do
    [ws, xs] = [slots[w].id, slots[x].id]
    wdt = if slots[w].dtype == :bf16, do: :bf16, else: :f32
    key = {:gemv_masked, wdt}

    {%{key => Kernels.gemv_masked(wdt)},
     [%{kernel: key,
        args: [{:slot, slots[n].id}, {:slot, ws}, {:slot, xs}, {:dim, ws, 0}, {:dim, ws, 1}, {:rows, xs}, {:dim, ws, 0},
               {:slot, slots[mk].id}],
        split: split(3, 4, [{0, {:bytes, 4}}, {1, {:last_bytes, ws}}])}], []}
  end

  # block-diagonal GEMV: threads split the groups (y, W, x advance by one group)
  defp region_calls({:node, {:linear_grouped, x, w, g} = n}, slots, _uses, _r, next) do
    [ws, xs] = [slots[w].id, slots[x].id]
    wdt = if slots[w].dtype == :bf16, do: :bf16, else: :f32
    [gn, k] = slots[w].shape
    nn = div(gn, g)
    key = {:gemv_grouped, wdt}
    scratch = %{id: next, dtype: :f32, shape: [4], role: :tmp}
    wb = if wdt == :bf16, do: 2, else: 4

    {%{key => Kernels.gemv_grouped(wdt)},
     [%{kernel: key,
        args: [{:slot, slots[n].id}, {:slot, ws}, {:slot, xs}, {:imm, nn}, {:imm, k}, {:rows, xs}, {:imm, g},
               {:imm, g * k}, {:imm, gn}, {:slot, scratch.id}],
        split: split(6, 1, [{0, {:bytes, 4 * nn}}, {1, {:bytes, wb * nn * k}}, {2, {:bytes, 4 * k}}], [{9, 16}])}],
     [scratch]}
  end

  defp region_calls({:node, {:gather_row, table, idx} = n}, slots, _uses, _r, _next) do
    [ts, is] = [slots[table].id, slots[idx].id]
    {key, kernel} = if slots[table].dtype == :bf16, do: {:gather_row_bf16, Kernels.gather_row_bf16()}, else: {:gather_row, Kernels.gather_row()}

    {%{key => kernel},
     [%{kernel: key,
        args: [{:slot, slots[n].id}, {:slot, ts}, {:slot, is}, {:dim, is, 0}, {:dim, ts, 0}, {:dim, ts, 1}],
        split: split(3, 1, [{0, {:last_bytes, slots[n].id}}, {2, {:bytes, 4}}])}], []}
  end

  defp region_calls({:node, {:rope, x, cos, sin, pos, h} = n}, slots, _uses, _r, next) do
    [_, w] = slots[x].shape
    scratch = %{id: next, dtype: :f32, shape: [16], role: :tmp}

    {%{:rope => Kernels.rope()},
     [%{kernel: :rope,
        args: [{:slot, slots[n].id}, {:slot, slots[x].id}, {:slot, slots[cos].id}, {:slot, slots[sin].id},
               {:slot, slots[pos].id}, {:slot, scratch.id}, {:dim, slots[x].id, 0}, {:imm, h},
               {:imm, div(w, 2 * h)}, {:dim, slots[cos].id, 0}],
        split: split(6, 1, [{0, {:last_bytes, slots[x].id}}, {1, {:last_bytes, slots[x].id}}, {4, {:bytes, 4}}], [{5, 64}])}],
     [scratch]}
  end

  defp region_calls({:node, {:kv_write, cache, pos, rows} = n}, slots, _uses, _r, _next) do
    [cs, ps, rs, os] = [slots[cache].id, slots[pos].id, slots[rows].id, slots[n].id]
    tail = [{:slot, ps}, {:slot, rs}, {:dim, rs, 0}, {:dim, cs, 0}, {:dim, cs, 1}]

    if os == cs,
      do: {%{{:kv_write, :inplace} => Kernels.kv_write(:inplace)},
           [%{kernel: {:kv_write, :inplace}, args: [{:slot, cs} | tail]}], []},
      else: {%{{:kv_write, :copy} => Kernels.kv_write(:copy)}, [%{kernel: {:kv_write, :copy}, args: [{:slot, os}, {:slot, cs} | tail]}], []}
  end

  defp region_calls({:node, {:attention, q, k, v, pos, heads} = n}, slots, _uses, _r, next) do
    {h, hkv} = Term.heads_of(heads)
    [_, w] = slots[q].shape
    [s, _] = slots[k].shape
    dh = div(w, h)
    key = {:attention, Term.attention_scale_bits(heads, dh)}
    scratch = %{id: next, dtype: :f32, shape: [32 + div(s + 15, 16) * 16], role: :tmp}

    {%{key => Kernels.attention(elem(key, 1))},
     [%{kernel: key,
        args: [{:slot, slots[n].id}, {:slot, slots[q].id}, {:slot, slots[k].id}, {:slot, slots[v].id},
               {:slot, slots[pos].id}, {:slot, scratch.id}, {:dim, slots[q].id, 0}, {:imm, hkv},
               {:imm, div(h, hkv)}, {:imm, s}, {:imm, dh}, {:imm, hkv * dh}, {:imm, Term.attention_window(heads) || 0}],
        split: attention_splits(q, slots, div(h, hkv), dh, scratch, [])}], [scratch]}
  end

  # cache writes are not partitioned: two rows naming the same cache row
  # must resolve in row order (the oracle's last-write-wins)
  defp region_calls({:node, {:kv_write_paged, pool, table, slot, pos, rows, page}}, slots, _uses, _r, _next) do
    ps = slots[pool].id
    [ts, rs] = [slots[table].id, slots[rows].id]
    sh = log2(page)
    key = {:kv_write_paged, sh}

    {%{key => Kernels.kv_write_paged(sh)},
     [%{kernel: key,
        args: [{:slot, ps}, {:slot, ts}, {:slot, slots[slot].id}, {:slot, slots[pos].id}, {:slot, rs},
               {:dim, rs, 0}, {:dim, ts, 0}, {:dim, ts, 1}, {:imm, div(Term.dim_max(hd(slots[pool].shape)), page)},
               {:dim, ps, 1}]}], []}
  end

  defp region_calls({:node, {:attention_paged, q, kp, vp, table, slot, pos, heads} = n}, slots, _uses, _r, next) do
    {h, hkv, page} = Term.heads_of(heads)
    [_, w] = slots[q].shape
    [pr, _] = slots[kp].shape
    [_, mp] = slots[table].shape
    cap = mp * page
    dh = div(w, h)
    sh = log2(page)
    key = {:attention_paged, Term.attention_scale_bits(heads, dh), {:paged, sh}}
    scratch = %{id: next, dtype: :f32, shape: [32 + div(cap + 15, 16) * 16], role: :tmp}
    qs = slots[q].id

    {%{key => Kernels.attention(elem(key, 1), {:paged, sh})},
     [%{kernel: key,
        args: [{:slot, slots[n].id}, {:slot, qs}, {:slot, slots[kp].id}, {:slot, slots[vp].id},
               {:slot, slots[pos].id}, {:slot, scratch.id}, {:dim, qs, 0}, {:imm, hkv},
               {:imm, div(h, hkv)}, {:imm, cap}, {:imm, dh},
               {:slot, slots[table].id}, {:slot, slots[slot].id}, {:dim, slots[table].id, 0}, {:imm, mp},
               {:imm, div(pr, page)}, {:dim, slots[kp].id, 1}, {:imm, Term.attention_window(heads) || 0}],
        split: attention_splits(q, slots, div(h, hkv), dh, scratch, [{12, {:bytes, 4}}])}], [scratch]}
  end

  # a reshape is a copy of the bytes: the transpose kernel over one row
  defp region_calls({:node, {:reshape, x, shape} = n}, slots, _uses, _r, _next) do
    total = Enum.product(shape)

    {%{:transpose => Kernels.transpose()},
     [%{kernel: :transpose, args: [{:slot, slots[n].id}, {:slot, slots[x].id}, {:imm, 1}, {:imm, total}, {:imm, 1}]}], []}
  end

  defp region_calls({:node, {:transpose, x} = n}, slots, _uses, _r, _next) do
    xs = slots[x].id

    {%{:transpose => Kernels.transpose()},
     [%{kernel: :transpose,
        args: [{:slot, slots[n].id}, {:slot, xs}, {:dim, xs, 0}, {:dim, xs, 1}, {:dim, xs, 0}],
        split: split(2, 1, [{0, {:bytes, 4}}, {1, {:last_bytes, xs}}])}], []}
  end

  defp region_calls({:node, {:sample, logits, params} = n}, slots, _uses, _r, next) do
    [_, v] = slots[logits].shape
    ls = slots[logits].id
    scratch = %{id: next, dtype: :f32, shape: [v + 16], role: :tmp}

    {%{:sample => Kernels.sample()},
     [%{kernel: :sample,
        args: [{:slot, slots[n].id}, {:slot, ls}, {:slot, slots[params].id}, {:slot, scratch.id}, {:dim, ls, 0}, {:imm, v}],
        split: split(4, 1, [{0, {:bytes, 4}}, {1, {:bytes, 4 * v}}, {2, {:bytes, 8}}], [{3, 4 * (v + 16)}])}],
     [scratch]}
  end

  # x : f32[k] or f32[b, k]; the sub-block sums of all b rows are one pass
  defp region_calls({:node, {:qgemv, w, x} = n}, slots, _uses, _r, next) do
    %{shape: [rows, k]} = slots[w]
    lead = Enum.drop(slots[x].shape, -1)
    sums = %{id: next, dtype: :f32, shape: lead ++ [div(k, 32)], role: :tmp}
    xs = slots[x].id

    {%{:sb_sums => Kernels.sb_sums(), :gemv_sb4 => Kernels.gemv_sb4()},
     [%{kernel: :sb_sums, args: [{:slot, sums.id}, {:slot, xs}, {:numel, sums.id}],
        split: split(2, 1, [{0, {:bytes, 4}}, {1, {:bytes, 128}}])},
      %{kernel: :gemv_sb4,
        args: [{:slot, slots[n].id}, {:slot, slots[w].id}, {:slot, xs}, {:slot, sums.id},
               {:imm, rows}, {:imm, div(k, 256)}, {:rows, xs}, {:imm, rows}],
        split: split(4, 2, [{0, {:bytes, 4}}, {1, {:bytes, div(k, 256) * Vapor.Quant.Sb4.exec_bytes()}}])}], [sums]}
  end

  # predicated 4-bit: sub-block sums of every row (cheap, activations only),
  # then the GEMV that skips the weight reads of masked rows
  defp region_calls({:node, {:qgemv_masked, w, x, mk} = n}, slots, _uses, _r, next) do
    %{shape: [rows, k]} = slots[w]
    lead = Enum.drop(slots[x].shape, -1)
    sums = %{id: next, dtype: :f32, shape: lead ++ [div(k, 32)], role: :tmp}
    xs = slots[x].id

    {%{:sb_sums => Kernels.sb_sums(), :gemv_sb4_masked => Kernels.gemv_sb4_masked()},
     [%{kernel: :sb_sums, args: [{:slot, sums.id}, {:slot, xs}, {:numel, sums.id}],
        split: split(2, 1, [{0, {:bytes, 4}}, {1, {:bytes, 128}}])},
      %{kernel: :gemv_sb4_masked,
        args: [{:slot, slots[n].id}, {:slot, slots[w].id}, {:slot, xs}, {:slot, sums.id},
               {:imm, rows}, {:imm, div(k, 256)}, {:rows, xs}, {:imm, rows}, {:slot, slots[mk].id}],
        split: split(4, 2, [{0, {:bytes, 4}}, {1, {:bytes, div(k, 256) * Vapor.Quant.Sb4.exec_bytes()}}])}], [sums]}
  end

  defp region_calls({:node, {:gemm_i8, a, w} = n}, slots, _uses, _r, _next) do
    {%{:gemm_i8 => Kernels.gemm_i8()},
     [%{kernel: :gemm_i8,
        args: [{:slot, slots[n].id}, {:slot, slots[a].id}, {:slot, slots[w].id},
               {:dim, slots[a].id, 0}, {:dim, slots[w].id, 0}, {:dim, slots[a].id, 1}],
        split: split(3, 1, [{0, {:last_bytes, slots[n].id}}, {1, {:last_bytes, slots[a].id}}])}], []}
  end

  # attention splits over rows; a single row (decoding) splits over its kv
  # heads instead — each thread takes whole groups of query heads
  defp attention_splits(q, slots, g, dh, scratch, extra) do
    qs = slots[q].id
    per = [{5, 4 * Enum.product(scratch.shape)}]

    [split(6, 1, [{0, {:last_bytes, qs}}, {1, {:last_bytes, qs}}, {4, {:bytes, 4}}] ++ extra, per),
     split(7, 1, [{0, {:bytes, 4 * g * dh}}, {1, {:bytes, 4 * g * dh}}, {2, {:bytes, 4 * dh}}, {3, {:bytes, 4 * dh}}], per)
     |> Map.put(:guard, 6)]
  end

  # A partition descriptor (see the worker): the count argument, the grain,
  # pointer arguments with their stride per unit, per-thread scratch.
  defp split(count, grain, ptrs, scratch \\ []), do: %{count: count, grain: grain, ptrs: ptrs, scratch: scratch}

  defp emit_all(kernels, targets, policy) do
    Enum.reduce_while(targets, {:ok, %{}}, fn be, {:ok, acc} ->
      compiled =
        Enum.reduce_while(Enum.sort_by(kernels, &inspect(elem(&1, 0))), {:ok, []}, fn {key, k}, {:ok, cs} ->
          case Machine.compile(k, be, policy: policy) do
            {:ok, c} -> {:cont, {:ok, [{key, c} | cs]}}
            {:error, info} -> {:halt, {:error, Rejection.new(key, {:allocation, be.isa(), info}, "cut the region")}}
          end
        end)

      case compiled do
        {:ok, cs} ->
          cs = Enum.reverse(cs)
          {blob, entries} = Link.link(cs)
          {:cont, {:ok, Map.put(acc, be.isa(), %{blob: blob, entries: entries, codes: Map.new(cs)})}}

        err ->
          {:halt, err}
      end
    end)
  end

  defp log2(n), do: n |> :math.log2() |> round()
end
