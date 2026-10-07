defmodule Vapor.Emit.SpirvKernels do
  @moduledoc """
  The kernel library for Substrate II, as SPIR-V compute modules.

  Each kernel reproduces the *canonical order* of `Vapor.Runtime.Oracle`
  exactly — one invocation per output element or row, the same 16-lane
  accumulation pattern, the same tree reduction — with every `OpFMul`,
  `OpFAdd`, `OpFSub` decorated `NoContraction`. Because Vulkan requires
  those operations to be correctly rounded, the GPU result is bit-identical
  to every CPU substrate (for denormal-free data on devices that flush).

  ABI: slot arguments become storage-buffer bindings `0..n` (in order), and
  immediate arguments become 32-bit push constants (in order) — the same
  argument list as the native kernels.
  """
  import Bitwise
  alias Vapor.Emit.SpirV

  @wg 64
  @max_dh 512

  @doc "Module description + dispatch geometry for a kernel key."
  def kernel({:ew, spec}, policy), do: ew(spec, policy)
  def kernel(:sb_sums, _policy), do: sb_sums()
  def kernel(:gemv_sb4, policy), do: gemv_sb4(policy)
  def kernel(:gemv_sb4_masked, policy), do: gemv_sb4(policy, true)
  def kernel(:gemm_i8, _policy), do: gemm_i8()
  def kernel(:gemm_i8_coop, _policy), do: gemm_i8_coop()
  def kernel({:reduce, op}, _policy), do: reduce(op)
  def kernel(:gemv_f32, policy), do: gemv_f32(policy)
  def kernel(:gemv_bf16, policy), do: gemv_f32(policy, :bf16)
  def kernel({:gemv_masked, wdt}, policy), do: gemv_f32(policy, wdt, true)
  def kernel({:gemv_grouped, wdt}, policy), do: gemv_grouped(policy, wdt)
  def kernel(:gather_row_bf16, _policy), do: gather_row(:bf16)
  def kernel(:gather_row, _policy), do: gather_row(:f32)
  def kernel(:rope, _policy), do: rope()
  def kernel({:kv_write, mode}, _policy), do: kv_write(mode)
  def kernel({:kv_write_paged, sh}, _policy), do: kv_write_paged(sh)
  def kernel({:attention, scale}, _policy), do: attention(scale, :contig)
  def kernel({:attention_paged, scale, layout}, _policy), do: attention(scale, layout)
  def kernel(:sample, _policy), do: sample()
  def kernel(:transpose, _policy), do: transpose()

  @doc "Assembled SPIR-V binary for a kernel key."
  def binary(key, policy), do: key |> kernel(policy) |> Map.fetch!(:module) |> SpirV.assemble() |> SpirV.to_binary()

  @doc """
  Compiled form stored in `Vapor.Compiled`: binary, binding and push-constant
  counts — or `nil` when the fabric has no module for the kernel yet (the
  program then simply does not run on the fabric; `Vapor.Compiled.runs_on?/2`).
  """
  def compile(key, policy) do
    if supported?(key) do
      %{module: m, bindings: nb} = kernel(key, policy)
      %{bin: m |> SpirV.assemble() |> SpirV.to_binary(), nbind: nb, npush: length(Map.get(m, :push, []))}
    end
  end

  def supported?({:ew, _}), do: true
  def supported?({:reduce, _}), do: true
  def supported?({:kv_write, _}), do: true
  def supported?({:kv_write_paged, _}), do: true
  def supported?({:attention, _}), do: true
  def supported?({:gemv_masked, _}), do: true
  def supported?({:gemv_grouped, _}), do: true
  def supported?({:attention_paged, _, _}), do: true
  def supported?(k),
    do: k in [:gemv_f32, :gemv_bf16, :sb_sums, :gemv_sb4, :gemv_sb4_masked, :gemm_i8, :gemm_i8_coop, :gather_row, :gather_row_bf16, :rope, :sample, :transpose]

  @doc "Workgroup counts for a dispatch, from its push constants."
  def groups({:ew, _}, [r, c]), do: {groups(r * c), 1, 1}
  def groups({:reduce, _}, [r, _c]), do: {groups(r), 1, 1}
  def groups(g, [n, _k, b, _ldy]) when g in [:gemv_f32, :gemv_bf16], do: {groups(n * b), 1, 1}
  def groups({:gemv_masked, _}, [n, _k, b, _ldy]), do: {groups(n * b), 1, 1}
  def groups({:gemv_grouped, _}, [n, _k, b, g, _ldx, _ldy]), do: {groups(n * b * g), 1, 1}
  def groups(:sb_sums, [nsub]), do: {groups(nsub), 1, 1}
  def groups(g, [rows, _nsb, b, _ldy]) when g in [:gemv_sb4, :gemv_sb4_masked], do: {groups(rows * b), 1, 1}
  def groups(:gemm_i8, [m, n, _k]), do: {groups(m * n), 1, 1}
  def groups(:gemm_i8_coop, [m, n, _k]), do: {div(m, 16) * div(n, 16), 1, 1}
  def groups(g, [t, _v, d]) when g in [:gather_row, :gather_row_bf16], do: {groups(t * d), 1, 1}
  def groups(:rope, [t, h, half, _s]), do: {groups(t * h * half), 1, 1}
  def groups({:kv_write, :copy}, [_t, s, w]), do: {groups(s * w), 1, 1}
  def groups({:kv_write, :inplace}, [t, _s, w]), do: {groups(t * w), 1, 1}
  def groups({:kv_write_paged, _}, [t, _ns, _mp, _np, w]), do: {groups(t * w), 1, 1}
  def groups({:attention, _}, [t, hkv, g | _]), do: {groups(t * hkv * g), 1, 1}
  def groups({:attention_paged, _, _}, [t, hkv, g | _]), do: {groups(t * hkv * g), 1, 1}
  def groups(:sample, [b, _v]), do: {groups(b), 1, 1}
  def groups(:transpose, [r, c, _ldo]), do: {groups(r * c), 1, 1}

  @doc "Whether a dispatch fits the module's static limits (attention heads up to #{@max_dh})."
  def fits?({:attention, _}, [_t, _hkv, _g, _s, dh | _]), do: dh <= @max_dh
  def fits?({:attention_paged, _, _}, [_t, _hkv, _g, _cap, dh | _]), do: dh <= @max_dh
  def fits?(_key, _push), do: true

  # ------------------------------------------------------------- helpers --

  defp ld(buf, idx, res),
    do: [{{res, :p}, :OpAccessChain, {:ptr, :StorageBuffer, :u32}, [{:buf, buf}, {:c, :u32, 0}, idx]},
         {res, :OpLoad, :u32, [{res, :p}]}]

  defp ldf(buf, idx, res), do: ld(buf, idx, {res, :u}) ++ [{res, :OpBitcast, :f32, [{res, :u}]}]

  defp st(buf, idx, val, tag),
    do: [{{tag, :p}, :OpAccessChain, {:ptr, :StorageBuffer, :u32}, [{:buf, buf}, {:c, :u32, 0}, idx]},
         {nil, :OpStore, nil, [{tag, :p}, val]}]

  defp stf(buf, idx, val, tag), do: [{{tag, :u}, :OpBitcast, :u32, [val]}] ++ st(buf, idx, {tag, :u}, tag)

  defp u(n), do: {:c, :u32, n}
  defp fzero, do: {:c, :f32, 0}

  defp guarded(i, n, body) do
    [{:in_range, :OpULessThan, :bool, [i, n]},
     {nil, :OpSelectionMerge, nil, [:done, {:lit, 0}]},
     {nil, :OpBranchConditional, nil, [:in_range, :work, :done]},
     {:work, :OpLabel, nil, []}] ++ body ++
      [{nil, :OpBranch, nil, [:done]}, {:done, :OpLabel, nil, []}]
  end

  defp gidx, do: [{:i, :OpCompositeExtract, :u32, [:gid, {:lit, 0}]}]

  defp mac(acc, a, b, res, :canonical),
    do: [{{res, :m}, :OpFMul, :f32, [a, b]}, {res, :OpFAdd, :f32, [acc, {res, :m}]}]

  defp mac(acc, a, b, res, :fast), do: [{res, :OpExtInst, :f32, [:glsl, {:lit, 50}, a, b, acc]}]

  # canonical 16-lane tree: r8[i]=l[i]+l[i+8], r4, r2, r
  defp reduce16(lanes, tag) do
    {code, [r]} =
      Enum.reduce([8, 4, 2, 1], {[], lanes}, fn h, {code, xs} ->
        t = List.to_tuple(xs)
        names = for i <- 0..(h - 1), do: {tag, h, i}
        adds = for i <- 0..(h - 1), do: {{tag, h, i}, :OpFAdd, :f32, [elem(t, i), elem(t, i + h)]}
        {code ++ adds, names}
      end)

    {code, r}
  end

  defp groups(n), do: max(div(n + @wg - 1, @wg), 1)

  # ------------------------------------------------------------- kernels --

  @doc """
  Fused elementwise region — same spec as `Vapor.KIR.Kernels.ew/1`: one
  invocation per element of the `[R, C]` space; operand classes pick the
  element index (full: i, row: column, col: row, scalar: 0). Values produced
  by integer primitives stay `u32`-typed and are bitcast only where a float
  operation consumes them, so no bit pattern ever sits in a float value
  that a driver could canonicalise.
  """
  def ew(spec, policy) do
    %{inputs: classes, outputs: outs, ops: ops} = Vapor.KIR.Kernels.normalize_ew(spec)
    p = length(classes)
    q = length(outs)
    cls = List.to_tuple(classes)
    int_ops = [:iadd, :isub, :iand, :ixor, :shl, :shr]

    used =
      (Enum.flat_map(ops, fn {_, _, a} -> a end) ++ outs)
      |> Enum.flat_map(fn
        {:in, i} -> [i]
        _ -> []
      end)
      |> Enum.uniq()

    index = fn i ->
      case elem(cls, i) do
        :full -> :i
        :row -> :col
        :col -> :row
        :scalar -> u(0)
      end
    end

    loads = Enum.flat_map(used, &ldf(&1, index.(&1), {:x, &1}))

    # types of temporaries: :f (f32) or :u (u32)
    types = Map.new(ops, fn {{:t, k}, op, _} -> {k, if(op in int_ops, do: :u, else: :f)} end)

    as = fn want, operand, tag ->
      case {want, operand} do
        {:f, {:splat, b}} -> {[], {:c, :f32, b}}
        {:u, {:splat, b}} -> {[], {:c, :u32, b}}
        {:u, {:imm, n}} -> {[], {:c, :u32, n}}
        {:f, {:in, i}} -> {[], {:x, i}}
        {:u, {:in, i}} -> {[{{:xu, i, tag}, :OpBitcast, :u32, [{:x, i}]}], {:xu, i, tag}}
        {w, {:t, k}} ->
          case {w, types[k]} do
            {same, same} -> {[], {:tv, k}}
            {:f, :u} -> {[{{:tf, k, tag}, :OpBitcast, :f32, [{:tv, k}]}], {:tf, k, tag}}
            {:u, :f} -> {[{{:tu, k, tag}, :OpBitcast, :u32, [{:tv, k}]}], {:tu, k, tag}}
          end
      end
    end

    args_as = fn want, args, k ->
      {pre, vs} = args |> Enum.with_index() |> Enum.map(fn {a, j} -> as.(want, a, {k, j}) end) |> Enum.unzip()
      {List.flatten(pre), vs}
    end

    compute =
      Enum.flat_map(ops, fn {{:t, k}, op, args} ->
        t = {:tv, k}

        cond do
          op in int_ops ->
            {pre, [x, y]} = args_as.(:u, args, k)
            spv = %{iadd: :OpIAdd, isub: :OpISub, iand: :OpBitwiseAnd, ixor: :OpBitwiseXor,
                    shl: :OpShiftLeftLogical, shr: :OpShiftRightLogical}[op]
            pre ++ [{t, spv, :u32, [x, y]}]

          op == :sel_lt ->
            {pre, [a, b, x, y]} = args_as.(:f, args, k)
            pre ++ [{{:lt, k}, :OpFOrdLessThan, :bool, [a, b]}, {t, :OpSelect, :f32, [{:lt, k}, x, y]}]

          true ->
            {pre, a} = args_as.(:f, args, k)

            pre ++
              case {op, a} do
                {:add, [x, y]} -> [{t, :OpFAdd, :f32, [x, y]}]
                {:sub, [x, y]} -> [{t, :OpFSub, :f32, [x, y]}]
                {:mul, [x, y]} -> [{t, :OpFMul, :f32, [x, y]}]
                {:neg, [x]} -> [{t, :OpFNegate, :f32, [x]}]
                {:relu, [x]} -> [{{:gt, k}, :OpFOrdGreaterThan, :bool, [x, fzero()]},
                                 {t, :OpSelect, :f32, [{:gt, k}, x, fzero()]}]
                {:fma, [x, y, z]} when policy == :fast -> [{t, :OpExtInst, :f32, [:glsl, {:lit, 50}, x, y, z]}]
                {:fma, [x, y, z]} -> [{{:m, k}, :OpFMul, :f32, [x, y]}, {t, :OpFAdd, :f32, [{:m, k}, z]}]
              end
        end
      end)

    stores =
      outs
      |> Enum.with_index()
      |> Enum.flat_map(fn {slot, j} ->
        {pre, [v]} = args_as.(:f, [slot], {:out, j})
        pre ++ stf(p + j, :i, v, {:o, j})
      end)

    module = %{
      buffers: for(k <- 0..(p + q - 1), do: %{binding: k, elem: :u32, writable: k >= p}),
      push: [:rows, :cols],
      glsl: policy == :fast,
      no_contraction: policy == :canonical,
      body:
        gidx() ++
          [{:total, :OpIMul, :u32, [{:push, :rows}, {:push, :cols}]}] ++
          guarded(:i, :total,
            [{:row, :OpUDiv, :u32, [:i, {:push, :cols}]}, {:col, :OpUMod, :u32, [:i, {:push, :cols}]}] ++
              loads ++ compute ++ stores)
    }

    %{module: module, bindings: p + q, groups: fn [r, c] -> {groups(r * c), 1, 1} end}
  end

  # a counted loop `for s in 0..n-1` over Function-storage variable `var`
  defp loop(tag, var, n, body) do
    [{nil, :OpStore, nil, [var, u(0)]},
     {nil, :OpBranch, nil, [{tag, :hdr}]},
     {{tag, :hdr}, :OpLabel, nil, []},
     {nil, :OpLoopMerge, nil, [{tag, :merge}, {tag, :cont}, {:lit, 0}]},
     {nil, :OpBranch, nil, [{tag, :chk}]},
     {{tag, :chk}, :OpLabel, nil, []},
     {{tag, :s}, :OpLoad, :u32, [var]},
     {{tag, :more}, :OpULessThan, :bool, [{tag, :s}, n]},
     {nil, :OpBranchConditional, nil, [{tag, :more}, {tag, :body}, {tag, :merge}]},
     {{tag, :body}, :OpLabel, nil, []}] ++
      body.({tag, :s}) ++
      [{nil, :OpBranch, nil, [{tag, :cont}]},
       {{tag, :cont}, :OpLabel, nil, []},
       {{tag, :s1}, :OpIAdd, :u32, [{tag, :s}, u(1)]},
       {nil, :OpStore, nil, [var, {tag, :s1}]},
       {nil, :OpBranch, nil, [{tag, :hdr}]},
       {{tag, :merge}, :OpLabel, nil, []}]
  end

  defp fvars(tag), do: for(l <- 0..15, do: {{tag, l}, :OpVariable, {:ptr, :Function, :f32}, [{:storage, :Function}]})

  # tree over 16 lanes with a combiner (canonical order)
  defp tree16(lanes, tag, comb) do
    Enum.reduce([8, 4, 2, 1], {[], lanes}, fn h, {code, xs} ->
      t = List.to_tuple(xs)
      steps = for i <- 0..(h - 1), do: comb.({tag, h, i}, elem(t, i), elem(t, i + h))
      {code ++ Enum.concat(steps), for(i <- 0..(h - 1), do: {tag, h, i})}
    end)
    |> then(fn {code, [r]} -> {code, r} end)
  end

  defp max_step(res, a, b), do: [{{res, :lt}, :OpFOrdLessThan, :bool, [a, b]}, {res, :OpSelect, :f32, [{res, :lt}, b, a]}]
  defp add_step(res, a, b), do: [{res, :OpFAdd, :f32, [a, b]}]

  @doc "Canonical row reduction (sum | max). bindings [y, x], push [rows, cols]"
  def reduce(op) do
    {id, comb} = if op == :sum, do: {fzero(), &add_step/3}, else: {{:c, :f32, 0xFF7F_FFFF}, &max_step/3}

    body =
      [{:c_var, :OpVariable, {:ptr, :Function, :u32}, [{:storage, :Function}]}] ++ fvars(:acc) ++ gidx() ++
        guarded(:i, {:push, :rows},
          for(l <- 0..15, do: {nil, :OpStore, nil, [{:acc, l}, id]}) ++
            [{:nchunk, :OpShiftRightLogical, :u32, [{:push, :cols}, u(4)]},
             {:rowbase, :OpIMul, :u32, [:i, {:push, :cols}]}] ++
            loop(:c, :c_var, :nchunk, fn s ->
              [{:cb0, :OpIMul, :u32, [s, u(16)]}, {:cb, :OpIAdd, :u32, [:rowbase, :cb0]}] ++
                Enum.flat_map(0..15, fn l ->
                  [{{:xi, l}, :OpIAdd, :u32, [:cb, u(l)]}] ++ ldf(1, {:xi, l}, {:xv, l}) ++
                    [{{:a0, l}, :OpLoad, :f32, [{:acc, l}]}] ++
                    comb.({:a1, l}, {:a0, l}, {:xv, l}) ++
                    [{nil, :OpStore, nil, [{:acc, l}, {:a1, l}]}]
                end)
            end) ++
            for(l <- 0..15, do: {{:fin, l}, :OpLoad, :f32, [{:acc, l}]}) ++
            (fn -> {code, r} = tree16(for(l <- 0..15, do: {:fin, l}), :red, comb); code ++ stf(0, :i, r, :out) end).())

    module = %{buffers: [%{binding: 0, elem: :u32}, %{binding: 1, elem: :u32, writable: false}],
               push: [:rows, :cols], no_contraction: true, body: body}

    %{module: module, bindings: 2, groups: fn [r, _c] -> {groups(r), 1, 1} end}
  end

  # element i of a bf16 buffer of u32 words, widened: the high or low half
  # of word i/2, moved to the top of a binary32 pattern
  defp ld_bf16(buf, idx, res) do
    n = &{res, &1}

    [{n.(:w), :OpShiftRightLogical, :u32, [idx, u(1)]}] ++ ld(buf, n.(:w), n.(:word)) ++
      [{n.(:odd), :OpBitwiseAnd, :u32, [idx, u(1)]}, {n.(:isodd), :OpIEqual, :bool, [n.(:odd), u(1)]},
       {n.(:lo), :OpShiftLeftLogical, :u32, [n.(:word), u(16)]},
       {n.(:hi), :OpBitwiseAnd, :u32, [n.(:word), u(0xFFFF_0000)]},
       {n.(:bits), :OpSelect, :u32, [n.(:isodd), n.(:hi), n.(:lo)]},
       {res, :OpBitcast, :f32, [n.(:bits)]}]
  end

  @doc """
  Dense y = x·Wᵀ (W f32, or bf16 widened), one invocation per output.
  bindings [y, W, x], push [n, k, b, ldy]. `masked`: binding 3 holds one
  mask word per activation row; a ±0 word stores +0 (`linear_masked`).
  """
  def gemv_f32(policy, wdt \\ :f32, masked \\ false) do
    ldw = if wdt == :bf16, do: &ld_bf16/3, else: &ldf/3
    body =
      [{:c_var, :OpVariable, {:ptr, :Function, :u32}, [{:storage, :Function}]}] ++ fvars(:acc) ++ gidx() ++
        [{:total, :OpIMul, :u32, [{:push, :n}, {:push, :b}]}] ++
        guarded(:i, :total,
          for(l <- 0..15, do: {nil, :OpStore, nil, [{:acc, l}, fzero()]}) ++
            [{:bi, :OpUDiv, :u32, [:i, {:push, :n}]},
             {:ri, :OpUMod, :u32, [:i, {:push, :n}]},
             {:wbase, :OpIMul, :u32, [:ri, {:push, :k}]},
             {:xbase, :OpIMul, :u32, [:bi, {:push, :k}]},
             {:nchunk, :OpShiftRightLogical, :u32, [{:push, :k}, u(4)]}] ++
            loop(:c, :c_var, :nchunk, fn s ->
              [{:cb, :OpIMul, :u32, [s, u(16)]}, {:wb, :OpIAdd, :u32, [:wbase, :cb]}, {:xb, :OpIAdd, :u32, [:xbase, :cb]}] ++
                Enum.flat_map(0..15, fn l ->
                  [{{:wi, l}, :OpIAdd, :u32, [:wb, u(l)]}, {{:xi, l}, :OpIAdd, :u32, [:xb, u(l)]}] ++
                    ldw.(1, {:wi, l}, {:wv, l}) ++ ldf(2, {:xi, l}, {:xv, l}) ++
                    [{{:a0, l}, :OpLoad, :f32, [{:acc, l}]}] ++
                    mac({:a0, l}, {:wv, l}, {:xv, l}, {:a1, l}, policy) ++
                    [{nil, :OpStore, nil, [{:acc, l}, {:a1, l}]}]
                end)
            end) ++
            for(l <- 0..15, do: {{:fin, l}, :OpLoad, :f32, [{:acc, l}]}) ++
            [{:yrow, :OpIMul, :u32, [:bi, {:push, :ldy}]}, {:yi, :OpIAdd, :u32, [:yrow, :ri]}] ++
            (fn ->
               {code, r} = reduce16(for(l <- 0..15, do: {:fin, l}), :red)

               if masked do
                 code ++ ld(3, :bi, :mword) ++
                   [{:mmag, :OpBitwiseAnd, :u32, [:mword, u(0x7FFF_FFFF)]},
                    {:mzero, :OpIEqual, :bool, [:mmag, u(0)]},
                    {:yval, :OpSelect, :f32, [:mzero, fzero(), r]}] ++ stf(0, :yi, :yval, :out)
               else
                 code ++ stf(0, :yi, r, :out)
               end
             end).())

    mask_buf = if masked, do: [%{binding: 3, elem: :u32, writable: false}], else: []

    module = %{buffers: [%{binding: 0, elem: :u32}, %{binding: 1, elem: :u32, writable: false},
                         %{binding: 2, elem: :u32, writable: false}] ++ mask_buf,
               push: [:n, :k, :b, :ldy], glsl: policy == :fast, no_contraction: true, body: body}

    %{module: module, bindings: 3 + length(mask_buf), groups: fn [n, _k, b, _ldy] -> {groups(n * b), 1, 1} end}
  end

  @doc """
  Grouped (block-diagonal) GEMV, one invocation per output: output column
  `col` of row `bi` is group `col / n`, weight row `col`. bindings
  [y, W, x, scratch (unused)], push [n, k, b, g, ldx, ldy]
  """
  def gemv_grouped(policy, wdt \\ :f32) do
    ldw = if wdt == :bf16, do: &ld_bf16/3, else: &ldf/3

    body =
      [{:c_var, :OpVariable, {:ptr, :Function, :u32}, [{:storage, :Function}]}] ++ fvars(:acc) ++ gidx() ++
        [{:gn, :OpIMul, :u32, [{:push, :n}, {:push, :g}]}, {:total, :OpIMul, :u32, [:gn, {:push, :b}]}] ++
        guarded(:i, :total,
          for(l <- 0..15, do: {nil, :OpStore, nil, [{:acc, l}, fzero()]}) ++
            [{:bi, :OpUDiv, :u32, [:i, :gn]},
             {:col, :OpUMod, :u32, [:i, :gn]},
             {:grp, :OpUDiv, :u32, [:col, {:push, :n}]},
             {:wbase, :OpIMul, :u32, [:col, {:push, :k}]},
             {:xrowb, :OpIMul, :u32, [:bi, {:push, :ldx}]},
             {:xgrp, :OpIMul, :u32, [:grp, {:push, :k}]},
             {:xbase, :OpIAdd, :u32, [:xrowb, :xgrp]},
             {:nchunk, :OpShiftRightLogical, :u32, [{:push, :k}, u(4)]}] ++
            loop(:c, :c_var, :nchunk, fn s ->
              [{:cb, :OpIMul, :u32, [s, u(16)]}, {:wb, :OpIAdd, :u32, [:wbase, :cb]}, {:xb, :OpIAdd, :u32, [:xbase, :cb]}] ++
                Enum.flat_map(0..15, fn l ->
                  [{{:wi, l}, :OpIAdd, :u32, [:wb, u(l)]}, {{:xi, l}, :OpIAdd, :u32, [:xb, u(l)]}] ++
                    ldw.(1, {:wi, l}, {:wv, l}) ++ ldf(2, {:xi, l}, {:xv, l}) ++
                    [{{:a0, l}, :OpLoad, :f32, [{:acc, l}]}] ++
                    mac({:a0, l}, {:wv, l}, {:xv, l}, {:a1, l}, policy) ++
                    [{nil, :OpStore, nil, [{:acc, l}, {:a1, l}]}]
                end)
            end) ++
            for(l <- 0..15, do: {{:fin, l}, :OpLoad, :f32, [{:acc, l}]}) ++
            [{:yrow, :OpIMul, :u32, [:bi, {:push, :ldy}]}, {:yi, :OpIAdd, :u32, [:yrow, :col]}] ++
            (fn -> {code, r} = reduce16(for(l <- 0..15, do: {:fin, l}), :red); code ++ stf(0, :yi, r, :out) end).())

    module = %{buffers: [%{binding: 0, elem: :u32}, %{binding: 1, elem: :u32, writable: false},
                         %{binding: 2, elem: :u32, writable: false}, %{binding: 3, elem: :u32}],
               push: [:n, :k, :b, :g, :ldx, :ldy], glsl: policy == :fast, no_contraction: true, body: body}

    %{module: module, bindings: 4, groups: fn [n, _k, b, g, _ldx, _ldy] -> {groups(n * b * g), 1, 1} end}
  end

  @doc "X_s = reduce16(x[32s+l] + x[32s+16+l]). bindings [X, x], push [nsub]"
  def sb_sums do
    body =
      gidx() ++
        guarded(:i, {:push, :nsub},
          [{:base, :OpIMul, :u32, [:i, u(32)]}] ++
            Enum.flat_map(0..15, fn l ->
              [{{:lo, l}, :OpIAdd, :u32, [:base, u(l)]}, {{:hi, l}, :OpIAdd, :u32, [:base, u(16 + l)]}] ++
                ldf(1, {:lo, l}, {:xl, l}) ++ ldf(1, {:hi, l}, {:xh, l}) ++
                [{{:s, l}, :OpFAdd, :f32, [{:xl, l}, {:xh, l}]}]
            end) ++
            (fn -> {code, r} = reduce16(for(l <- 0..15, do: {:s, l}), :red); code ++ stf(0, :i, r, :out) end).())

    module = %{buffers: [%{binding: 0, elem: :u32}, %{binding: 1, elem: :u32, writable: false}],
               push: [:nsub], no_contraction: true, body: body}

    %{module: module, bindings: 2, groups: fn [nsub] -> {groups(nsub), 1, 1} end}
  end

  @doc """
  sb4x GEMV over `b` activation rows, one invocation per output
  `i = bi·rows + row`. bindings [y, W, x, X], push [rows, nsb, b, ldy].
  `masked`: binding 4 holds one mask word per activation row; a ±0 word
  runs the superblock loop zero times (no weight is read) and stores +0.
  """
  def gemv_sb4(policy, masked \\ false) do
    fvar = fn name -> {name, :OpVariable, {:ptr, :Function, :f32}, [{:storage, :Function}]} end
    vars = [{:s_var, :OpVariable, {:ptr, :Function, :u32}, [{:storage, :Function}]}, fvar.(:yb_var)] ++
             for(l <- 0..15, do: fvar.({:acc_var, l}))

    init =
      [{nil, :OpStore, nil, [:s_var, u(0)]}, {nil, :OpStore, nil, [:yb_var, fzero()]}] ++
        for(l <- 0..15, do: {nil, :OpStore, nil, [{:acc_var, l}, fzero()]}) ++
        [{:nsub_all, :OpIMul, :u32, [{:push, :nsb}, u(8)]},
         {:row, :OpUMod, :u32, [:i, {:push, :rows}]},
         {:bi, :OpUDiv, :u32, [:i, {:push, :rows}]}] ++
        (if masked do
           ld(4, :bi, :mword) ++
             [{:mmag, :OpBitwiseAnd, :u32, [:mword, u(0x7FFF_FFFF)]},
              {:mzero, :OpIEqual, :bool, [:mmag, u(0)]},
              {:nsub, :OpSelect, :u32, [:mzero, u(0), :nsub_all]}]
         else
           [{:nsub, :OpCopyObject, :u32, [:nsub_all]}]
         end) ++
        [
         {:rowbase, :OpIMul, :u32, [:row, {:push, :nsb}]},
         {:soff, :OpIMul, :u32, [:bi, :nsub]},
         {:xoff, :OpIMul, :u32, [:soff, u(32)]},
         {nil, :OpBranch, nil, [:hdr]}]

    loop_head =
      [{:hdr, :OpLabel, nil, []},
       {nil, :OpLoopMerge, nil, [:lmerge, :cont, {:lit, 0}]},
       {nil, :OpBranch, nil, [:chk]},
       {:chk, :OpLabel, nil, []},
       {:s, :OpLoad, :u32, [:s_var]},
       {:more, :OpULessThan, :bool, [:s, :nsub]},
       {nil, :OpBranchConditional, nil, [:more, :lbody, :lmerge]},
       {:lbody, :OpLabel, nil, []}]

    block =
      [{:sb, :OpShiftRightLogical, :u32, [:s, u(3)]},
       {:j, :OpBitwiseAnd, :u32, [:s, u(7)]},
       {:blk, :OpIAdd, :u32, [:rowbase, :sb]},
       {:wbase, :OpIMul, :u32, [:blk, u(38)]},
       {:j4, :OpIMul, :u32, [:j, u(4)]},
       {:nibbase, :OpIAdd, :u32, [:wbase, :j4]}] ++
        Enum.flat_map(0..3, fn k ->
          [{{:nibi, k}, :OpIAdd, :u32, [:nibbase, u(k)]}] ++ ld(1, {:nibi, k}, {:nib, k})
        end) ++
        [{:jw, :OpShiftRightLogical, :u32, [:j, u(2)]},
         {:jb, :OpBitwiseAnd, :u32, [:j, u(3)]},
         {:jsh, :OpIMul, :u32, [:jb, u(8)]},
         {:uwi0, :OpIAdd, :u32, [:wbase, u(32)]}, {:uwi, :OpIAdd, :u32, [:uwi0, :jw]},
         {:vwi0, :OpIAdd, :u32, [:wbase, u(34)]}, {:vwi, :OpIAdd, :u32, [:vwi0, :jw]}] ++
        ld(1, :uwi, :uw) ++ ld(1, :vwi, :vw) ++
        [{:ush, :OpShiftRightLogical, :u32, [:uw, :jsh]}, {:uu, :OpBitwiseAnd, :u32, [:ush, u(255)]},
         {:vsh, :OpShiftRightLogical, :u32, [:vw, :jsh]}, {:vv, :OpBitwiseAnd, :u32, [:vsh, u(255)]},
         {:c1i, :OpIAdd, :u32, [:wbase, u(36)]}, {:c0i, :OpIAdd, :u32, [:wbase, u(37)]}] ++
        ldf(1, :c1i, :c1) ++ ldf(1, :c0i, :c0) ++
        [{:fu, :OpConvertUToF, :f32, [:uu]}, {:fv, :OpConvertUToF, :f32, [:vv]},
         {:alpha, :OpFMul, :f32, [:c1, :fu]},
         {:tav, :OpFMul, :f32, [:alpha, :fv]},
         {:beta, :OpFSub, :f32, [:c0, :tav]},
         {:xb0, :OpIMul, :u32, [:s, u(32)]},
         {:xb, :OpIAdd, :u32, [:xb0, :xoff]}]

    lanes =
      Enum.flat_map(0..15, fn l ->
        [{{:bsh, l}, :OpShiftRightLogical, :u32, [{:nib, div(l, 4)}, u(8 * rem(l, 4))]},
         {{:byte, l}, :OpBitwiseAnd, :u32, [{:bsh, l}, u(255)]},
         {{:qlo, l}, :OpBitwiseAnd, :u32, [{:byte, l}, u(15)]},
         {{:qhi, l}, :OpShiftRightLogical, :u32, [{:byte, l}, u(4)]},
         {{:fqlo, l}, :OpConvertUToF, :f32, [{:qlo, l}]},
         {{:fqhi, l}, :OpConvertUToF, :f32, [{:qhi, l}]},
         {{:wlo, l}, :OpFMul, :f32, [:alpha, {:fqlo, l}]},
         {{:whi, l}, :OpFMul, :f32, [:alpha, {:fqhi, l}]},
         {{:xli, l}, :OpIAdd, :u32, [:xb, u(l)]},
         {{:xhi, l}, :OpIAdd, :u32, [:xb, u(16 + l)]}] ++
          ldf(2, {:xli, l}, {:xlo, l}) ++ ldf(2, {:xhi, l}, {:xhv, l}) ++
          [{{:a0, l}, :OpLoad, :f32, [{:acc_var, l}]}] ++
          mac({:a0, l}, {:wlo, l}, {:xlo, l}, {:a1, l}, policy) ++
          mac({:a1, l}, {:whi, l}, {:xhv, l}, {:a2, l}, policy) ++
          [{nil, :OpStore, nil, [{:acc_var, l}, {:a2, l}]}]
      end)

    tail =
      [{:si, :OpIAdd, :u32, [:s, :soff]}] ++
        ldf(3, :si, :xs) ++
        [{:yb0, :OpLoad, :f32, [:yb_var]}] ++
        mac(:yb0, :beta, :xs, :yb1, policy) ++
        [{nil, :OpStore, nil, [:yb_var, :yb1]},
         {nil, :OpBranch, nil, [:cont]},
         {:cont, :OpLabel, nil, []},
         {:s1, :OpIAdd, :u32, [:s, u(1)]},
         {nil, :OpStore, nil, [:s_var, :s1]},
         {nil, :OpBranch, nil, [:hdr]},
         {:lmerge, :OpLabel, nil, []}] ++
        for(l <- 0..15, do: {{:fin, l}, :OpLoad, :f32, [{:acc_var, l}]})

    {red, r} = reduce16(for(l <- 0..15, do: {:fin, l}), :red)

    finish =
      red ++
        [{:ybf, :OpLoad, :f32, [:yb_var]}, {:ysum, :OpFAdd, :f32, [r, :ybf]},
         {:yrow, :OpIMul, :u32, [:bi, {:push, :ldy}]}, {:yi, :OpIAdd, :u32, [:yrow, :row]}] ++
        if(masked, do: [{:y, :OpSelect, :f32, [:mzero, fzero(), :ysum]}], else: [{:y, :OpCopyObject, :f32, [:ysum]}]) ++
        stf(0, :yi, :y, :out)

    nbuf = if masked, do: 4, else: 3

    module = %{
      buffers: [%{binding: 0, elem: :u32}] ++ for(k <- 1..nbuf, do: %{binding: k, elem: :u32, writable: false}),
      push: [:rows, :nsb, :b, :ldy],
      glsl: policy == :fast,
      no_contraction: true,
      body: vars ++ gidx() ++ [{:total, :OpIMul, :u32, [{:push, :rows}, {:push, :b}]}] ++
              guarded(:i, :total, init ++ loop_head ++ block ++ lanes ++ tail ++ finish)
    }

    %{module: module, bindings: nbuf + 1, groups: fn [rows, _nsb, b, _ldy] -> {groups(rows * b), 1, 1} end}
  end

  @doc "C = A·Wᵀ over ℤ/2³²ℤ, one invocation per output. bindings [C, A, W], push [m, n, k]"
  def gemm_i8 do
    ivar = fn name, ty -> {name, :OpVariable, {:ptr, :Function, ty}, [{:storage, :Function}]} end

    sbyte = fn buf, idx, res ->
      [{{res, :w}, :OpShiftRightLogical, :u32, [idx, u(2)]}] ++
        ld(buf, {res, :w}, {res, :word}) ++
        [{{res, :b}, :OpBitwiseAnd, :u32, [idx, u(3)]},
         {{res, :off}, :OpIMul, :u32, [{res, :b}, u(8)]},
         {{res, :si}, :OpBitcast, :i32, [{res, :word}]},
         {res, :OpBitFieldSExtract, :i32, [{res, :si}, {res, :off}, u(8)]}]
    end

    body =
      [ivar.(:k_var, :u32), ivar.(:acc_var, :i32)] ++ gidx() ++
        [{:total, :OpIMul, :u32, [{:push, :m}, {:push, :n}]}] ++
        guarded(:i, :total,
          [{:row, :OpUDiv, :u32, [:i, {:push, :n}]},
           {:col, :OpUMod, :u32, [:i, {:push, :n}]},
           {:abase, :OpIMul, :u32, [:row, {:push, :k}]},
           {:wbase, :OpIMul, :u32, [:col, {:push, :k}]},
           {nil, :OpStore, nil, [:k_var, u(0)]},
           {nil, :OpStore, nil, [:acc_var, {:c, :i32, 0}]},
           {nil, :OpBranch, nil, [:hdr]},
           {:hdr, :OpLabel, nil, []},
           {nil, :OpLoopMerge, nil, [:lmerge, :cont, {:lit, 0}]},
           {nil, :OpBranch, nil, [:chk]},
           {:chk, :OpLabel, nil, []},
           {:kk, :OpLoad, :u32, [:k_var]},
           {:more, :OpULessThan, :bool, [:kk, {:push, :k}]},
           {nil, :OpBranchConditional, nil, [:more, :lbody, :lmerge]},
           {:lbody, :OpLabel, nil, []},
           {:ai, :OpIAdd, :u32, [:abase, :kk]},
           {:wi, :OpIAdd, :u32, [:wbase, :kk]}] ++
            sbyte.(1, :ai, :a) ++ sbyte.(2, :wi, :b) ++
            [{:prod, :OpIMul, :i32, [:a, :b]},
             {:acc0, :OpLoad, :i32, [:acc_var]},
             {:acc1, :OpIAdd, :i32, [:acc0, :prod]},
             {nil, :OpStore, nil, [:acc_var, :acc1]},
             {nil, :OpBranch, nil, [:cont]},
             {:cont, :OpLabel, nil, []},
             {:k1, :OpIAdd, :u32, [:kk, u(1)]},
             {nil, :OpStore, nil, [:k_var, :k1]},
             {nil, :OpBranch, nil, [:hdr]},
             {:lmerge, :OpLabel, nil, []},
             {:accf, :OpLoad, :i32, [:acc_var]},
             {:accu, :OpBitcast, :u32, [:accf]}] ++
            st(0, :i, :accu, :out))

    module = %{
      buffers: [%{binding: 0, elem: :u32}, %{binding: 1, elem: :u32, writable: false},
                %{binding: 2, elem: :u32, writable: false}],
      push: [:m, :n, :k],
      body: body
    }

    %{module: module, bindings: 3, groups: fn [m, n, _k] -> {groups(m * n), 1, 1} end}
  end

  @doc """
  16×16×16 cooperative-matrix int8 GEMM (`SPV_KHR_cooperative_matrix`),
  C = A·Wᵀ with M, N, K multiples of 16: one subgroup per output tile,
  A row-major (stride K), Wᵀ read column-major from W (stride K), C
  row-major (stride N), signed components throughout. Integer accumulation
  in ℤ/2³²ℤ is order-free (Theorem 7.1), so this tile order is bit-identical
  to every other substrate. Dispatched only when the device reports the
  (s8, s8, s32, 16×16×16, subgroup) cooperative-matrix configuration.
  bindings [C, A, W], push [m, n, k]
  """
  def gemm_i8_coop do
    a_t = {:coopmat, :i8, 3, 16, 16, 0}
    b_t = {:coopmat, :i8, 3, 16, 16, 1}
    c_t = {:coopmat, :i32, 3, 16, 16, 2}
    signed = 0xF

    body =
      [{:k_var, :OpVariable, {:ptr, :Function, :u32}, [{:storage, :Function}]},
       {:acc_var, :OpVariable, {:ptr, :Function, c_t}, [{:storage, :Function}]},
       {:tile, :OpCompositeExtract, :u32, [:wid, {:lit, 0}]},
       {:ntiles, :OpShiftRightLogical, :u32, [{:push, :n}, u(4)]},
       {:ti, :OpUDiv, :u32, [:tile, :ntiles]},
       {:tj, :OpUMod, :u32, [:tile, :ntiles]},
       {:row0, :OpShiftLeftLogical, :u32, [:ti, u(4)]},
       {:col0, :OpShiftLeftLogical, :u32, [:tj, u(4)]},
       {nil, :OpStore, nil, [:k_var, u(0)]},
       {nil, :OpStore, nil, [:acc_var, {:c, c_t, 0}]},
       {nil, :OpBranch, nil, [:hdr]},
       {:hdr, :OpLabel, nil, []},
       {nil, :OpLoopMerge, nil, [:lmerge, :cont, {:lit, 0}]},
       {nil, :OpBranch, nil, [:chk]},
       {:chk, :OpLabel, nil, []},
       {:kk, :OpLoad, :u32, [:k_var]},
       {:more, :OpULessThan, :bool, [:kk, {:push, :k}]},
       {nil, :OpBranchConditional, nil, [:more, :lbody, :lmerge]},
       {:lbody, :OpLabel, nil, []},
       {:arow, :OpIMul, :u32, [:row0, {:push, :k}]},
       {:aoff, :OpIAdd, :u32, [:arow, :kk]},
       {:ap, :OpAccessChain, {:ptr, :StorageBuffer, :i8}, [{:buf, 1}, {:c, :u32, 0}, :aoff]},
       {:bcol, :OpIMul, :u32, [:col0, {:push, :k}]},
       {:boff, :OpIAdd, :u32, [:bcol, :kk]},
       {:bp, :OpAccessChain, {:ptr, :StorageBuffer, :i8}, [{:buf, 2}, {:c, :u32, 0}, :boff]},
       {:am, :OpCooperativeMatrixLoadKHR, a_t, [:ap, {:c, :u32, 0}, {:push, :k}]},
       {:bm, :OpCooperativeMatrixLoadKHR, b_t, [:bp, {:c, :u32, 1}, {:push, :k}]},
       {:acc0, :OpLoad, c_t, [:acc_var]},
       {:acc1, :OpCooperativeMatrixMulAddKHR, c_t, [:am, :bm, :acc0, {:lit, signed}]},
       {nil, :OpStore, nil, [:acc_var, :acc1]},
       {nil, :OpBranch, nil, [:cont]},
       {:cont, :OpLabel, nil, []},
       {:k1, :OpIAdd, :u32, [:kk, u(16)]},
       {nil, :OpStore, nil, [:k_var, :k1]},
       {nil, :OpBranch, nil, [:hdr]},
       {:lmerge, :OpLabel, nil, []},
       {:crow, :OpIMul, :u32, [:row0, {:push, :n}]},
       {:coff, :OpIAdd, :u32, [:crow, :col0]},
       {:cp, :OpAccessChain, {:ptr, :StorageBuffer, :i32}, [{:buf, 0}, {:c, :u32, 0}, :coff]},
       {:accf, :OpLoad, c_t, [:acc_var]},
       {nil, :OpCooperativeMatrixStoreKHR, nil, [:cp, :accf, {:c, :u32, 0}, {:push, :n}]}]

    module = %{
      version: {1, 6},
      caps: [1, 39, 4448, 6022],
      exts: ["SPV_KHR_cooperative_matrix"],
      local_size: {32, 1, 1},
      buffers: [%{binding: 0, elem: :i32}, %{binding: 1, elem: :i8, writable: false},
                %{binding: 2, elem: :i8, writable: false}],
      push: [:m, :n, :k],
      body: body
    }

    %{module: module, bindings: 3, groups: fn [m, n, _k] -> {div(m, 16) * div(n, 16), 1, 1} end}
  end

  # ------------------------------------------------------- model operators --
  # Same argument lists as the native kernels (slots → bindings, immediates →
  # push constants, in order); the per-thread scratch slots some native
  # kernels take are bound but unused — a GPU invocation recomputes instead.
  # Index operands are u32 bit patterns, so a negative index is huge and
  # clamps or skips exactly as `Vapor.Runtime.Oracle` does.

  defp buf(k, writable), do: %{binding: k, elem: :u32, writable: writable}
  defp umin(res, a, b), do: [{{res, :lt}, :OpULessThan, :bool, [a, b]}, {res, :OpSelect, :u32, [{res, :lt}, a, b]}]
  defp uvar(name), do: {name, :OpVariable, {:ptr, :Function, :u32}, [{:storage, :Function}]}
  defp farr(name, n), do: {name, :OpVariable, {:ptr, :Function, {:array, :f32, n}}, [{:storage, :Function}]}
  defp at(arr, idx, res), do: {res, :OpAccessChain, {:ptr, :Function, :f32}, [arr, idx]}
  defp fneg_max, do: {:c, :f32, 0xFF7F_FFFF}

  defp when_(cnd, tag, body),
    do: [{nil, :OpSelectionMerge, nil, [{tag, :end}, {:lit, 0}]},
         {nil, :OpBranchConditional, nil, [cnd, {tag, :then}, {tag, :end}]},
         {{tag, :then}, :OpLabel, nil, []}] ++ body ++ [{nil, :OpBranch, nil, [{tag, :end}]}, {{tag, :end}, :OpLabel, nil, []}]

  # a canonical function (`Vapor.Canon`) of one f32 value, every name under `tag`
  defp canon(fun, x, tag) do
    {ops, r, _} = Vapor.Canon.expand(fun, [{:in, 0}], 0)
    int_ops = [:iadd, :isub, :iand, :ixor, :shl, :shr]
    types = Map.new(ops, fn {{:t, k}, op, _} -> {k, if(op in int_ops, do: :u, else: :f)} end)
    val = fn k -> {tag, :v, k} end

    as = fn want, operand, j ->
      case {want, operand} do
        {:f, {:splat, b}} -> {[], {:c, :f32, b}}
        {:u, {:splat, b}} -> {[], {:c, :u32, b}}
        {:u, {:imm, n}} -> {[], {:c, :u32, n}}
        {:f, {:in, 0}} -> {[], x}
        {:u, {:in, 0}} -> {[{{tag, :xu, j}, :OpBitcast, :u32, [x]}], {tag, :xu, j}}
        {w, {:t, k}} ->
          case {w, types[k]} do
            {same, same} -> {[], val.(k)}
            {:f, :u} -> {[{{tag, :tf, j}, :OpBitcast, :f32, [val.(k)]}], {tag, :tf, j}}
            {:u, :f} -> {[{{tag, :tu, j}, :OpBitcast, :u32, [val.(k)]}], {tag, :tu, j}}
          end
      end
    end

    args_as = fn want, args, k ->
      {pre, vs} = args |> Enum.with_index() |> Enum.map(fn {a, j} -> as.(want, a, {k, j}) end) |> Enum.unzip()
      {List.flatten(pre), vs}
    end

    code =
      Enum.flat_map(ops, fn {{:t, k}, op, args} ->
        t = val.(k)

        cond do
          op in int_ops ->
            {pre, [a, b]} = args_as.(:u, args, k)
            spv = %{iadd: :OpIAdd, isub: :OpISub, iand: :OpBitwiseAnd, ixor: :OpBitwiseXor,
                    shl: :OpShiftLeftLogical, shr: :OpShiftRightLogical}[op]
            pre ++ [{t, spv, :u32, [a, b]}]

          op == :sel_lt ->
            {pre, [a, b, x1, y1]} = args_as.(:f, args, k)
            pre ++ [{{tag, :lt, k}, :OpFOrdLessThan, :bool, [a, b]}, {t, :OpSelect, :f32, [{tag, :lt, k}, x1, y1]}]

          true ->
            {pre, a} = args_as.(:f, args, k)

            pre ++
              case {op, a} do
                {:add, [x1, y1]} -> [{t, :OpFAdd, :f32, [x1, y1]}]
                {:sub, [x1, y1]} -> [{t, :OpFSub, :f32, [x1, y1]}]
                {:mul, [x1, y1]} -> [{t, :OpFMul, :f32, [x1, y1]}]
                {:neg, [x1]} -> [{t, :OpFNegate, :f32, [x1]}]
                {:relu, [x1]} -> [{{tag, :gt, k}, :OpFOrdGreaterThan, :bool, [x1, fzero()]}, {t, :OpSelect, :f32, [{tag, :gt, k}, x1, fzero()]}]
                {:fma, [x1, y1, z1]} -> [{{tag, :m, k}, :OpFMul, :f32, [x1, y1]}, {t, :OpFAdd, :f32, [{tag, :m, k}, z1]}]
              end
        end
      end)

    {pre, res} = as.(:f, r, :res)
    {code ++ pre, res}
  end

  @doc "Rows of a table (f32, or bf16 widened) at clamped indices. bindings [out, table, idx], push [t, v, d]"
  def gather_row(tdt \\ :f32) do
    load = if tdt == :bf16, do: &(ld_bf16(&1, &2, {&3, :f}) ++ [{&3, :OpBitcast, :u32, [{&3, :f}]}]), else: &ld/3

    body =
      gidx() ++ [{:total, :OpIMul, :u32, [{:push, :t}, {:push, :d}]}] ++
        guarded(:i, :total,
          [{:row, :OpUDiv, :u32, [:i, {:push, :d}]}, {:col, :OpUMod, :u32, [:i, {:push, :d}]}] ++
            ld(2, :row, :ix) ++ [{:vm1, :OpISub, :u32, [{:push, :v}, u(1)]}] ++ umin(:r, :ix, :vm1) ++
            [{:sb, :OpIMul, :u32, [:r, {:push, :d}]}, {:src, :OpIAdd, :u32, [:sb, :col]}] ++
            load.(1, :src, :val) ++ st(0, :i, :val, :o))

    %{module: %{buffers: [buf(0, true), buf(1, false), buf(2, false)], push: [:t, :v, :d], body: body}, bindings: 3}
  end

  @doc "Rotary embedding (rotate-half). bindings [out, x, cos, sin, pos, scratch], push [t, h, half, s]"
  def rope do
    body =
      gidx() ++
        [{:hh, :OpIMul, :u32, [{:push, :h}, {:push, :half}]}, {:total, :OpIMul, :u32, [{:push, :t}, :hh]}] ++
        guarded(:i, :total,
          [{:row, :OpUDiv, :u32, [:i, :hh]}, {:rem, :OpUMod, :u32, [:i, :hh]},
           {:head, :OpUDiv, :u32, [:rem, {:push, :half}]}, {:k, :OpUMod, :u32, [:rem, {:push, :half}]}] ++
            ld(4, :row, :p) ++ [{:sm1, :OpISub, :u32, [{:push, :s}, u(1)]}] ++ umin(:pc, :p, :sm1) ++
            [{:cb, :OpIMul, :u32, [:pc, {:push, :half}]}, {:ci, :OpIAdd, :u32, [:cb, :k]}] ++
            ldf(2, :ci, :c) ++ ldf(3, :ci, :sn) ++
            [{:w, :OpIMul, :u32, [:hh, u(2)]}, {:rb, :OpIMul, :u32, [:row, :w]},
             {:hb0, :OpIMul, :u32, [:head, {:push, :half}]}, {:hb1, :OpIMul, :u32, [:hb0, u(2)]},
             {:base, :OpIAdd, :u32, [:rb, :hb1]}, {:i1, :OpIAdd, :u32, [:base, :k]},
             {:i2a, :OpIAdd, :u32, [:base, {:push, :half}]}, {:i2, :OpIAdd, :u32, [:i2a, :k]}] ++
            ldf(1, :i1, :a) ++ ldf(1, :i2, :b) ++
            [{:ac, :OpFMul, :f32, [:a, :c]}, {:bs, :OpFMul, :f32, [:b, :sn]}, {:o1, :OpFSub, :f32, [:ac, :bs]},
             {:bc, :OpFMul, :f32, [:b, :c]}, {:as, :OpFMul, :f32, [:a, :sn]}, {:o2, :OpFAdd, :f32, [:bc, :as]}] ++
            stf(0, :i1, :o1, :s1) ++ stf(0, :i2, :o2, :s2))

    %{module: %{buffers: [buf(0, true)] ++ for(k <- 1..5, do: buf(k, k == 5)), push: [:t, :h, :half, :s],
                no_contraction: true, body: body}, bindings: 6}
  end

  @doc """
  Cache rows written at positions (last write wins, positions ≥ s skipped).
  `:copy`: bindings [out, cache, pos, rows], one invocation per cache
  element; `:inplace`: bindings [cache, pos, rows], one per written element.
  push [t, s, w]
  """
  def kv_write(:copy) do
    body =
      [uvar(:tv), uvar(:last)] ++ gidx() ++ [{:total, :OpIMul, :u32, [{:push, :s}, {:push, :w}]}] ++
        guarded(:i, :total,
          [{:r, :OpUDiv, :u32, [:i, {:push, :w}]}, {:col, :OpUMod, :u32, [:i, {:push, :w}]},
           {nil, :OpStore, nil, [:last, u(0xFFFF_FFFF)]}] ++
            loop(:scan, :tv, {:push, :t}, fn tt ->
              ld(2, tt, :p) ++
                [{:hit, :OpIEqual, :bool, [:p, :r]}, {:l0, :OpLoad, :u32, [:last]},
                 {:l1, :OpSelect, :u32, [:hit, tt, :l0]}, {nil, :OpStore, nil, [:last, :l1]}]
            end) ++
            [{:lt, :OpLoad, :u32, [:last]}, {:none, :OpIEqual, :bool, [:lt, u(0xFFFF_FFFF)]},
             {:lt0, :OpSelect, :u32, [:none, u(0), :lt]}, {:rb, :OpIMul, :u32, [:lt0, {:push, :w}]},
             {:ri, :OpIAdd, :u32, [:rb, :col]}] ++
            ld(3, :ri, :new) ++ ld(1, :i, :old) ++
            [{:val, :OpSelect, :u32, [:none, :old, :new]}] ++ st(0, :i, :val, :o))

    %{module: %{buffers: [buf(0, true), buf(1, false), buf(2, false), buf(3, false)], push: [:t, :s, :w], body: body}, bindings: 4}
  end

  def kv_write(:inplace) do
    body =
      [uvar(:tv), uvar(:shadow)] ++ gidx() ++ [{:total, :OpIMul, :u32, [{:push, :t}, {:push, :w}]}] ++
        guarded(:i, :total,
          [{:tt, :OpUDiv, :u32, [:i, {:push, :w}]}, {:col, :OpUMod, :u32, [:i, {:push, :w}]},
           {nil, :OpStore, nil, [:shadow, u(0)]}] ++
            ld(1, :tt, :p) ++
            loop(:scan, :tv, {:push, :t}, fn t2 ->
              ld(1, t2, :p2) ++
                [{:same, :OpIEqual, :bool, [:p2, :p]}, {:later, :OpUGreaterThan, :bool, [t2, :tt]},
                 {:hit, :OpLogicalAnd, :bool, [:same, :later]}, {:s0, :OpLoad, :u32, [:shadow]},
                 {:s1, :OpSelect, :u32, [:hit, u(1), :s0]}, {nil, :OpStore, nil, [:shadow, :s1]}]
            end) ++
            [{:sh, :OpLoad, :u32, [:shadow]}, {:last, :OpIEqual, :bool, [:sh, u(0)]},
             {:inr, :OpULessThan, :bool, [:p, {:push, :s}]}, {:go, :OpLogicalAnd, :bool, [:last, :inr]}] ++
            when_(:go, :wr,
              [{:rb, :OpIMul, :u32, [:tt, {:push, :w}]}, {:ri, :OpIAdd, :u32, [:rb, :col]},
               {:db, :OpIMul, :u32, [:p, {:push, :w}]}, {:di, :OpIAdd, :u32, [:db, :col]}] ++
                ld(2, :ri, :val) ++ st(0, :di, :val, :o)))

    %{module: %{buffers: [buf(0, true), buf(1, false), buf(2, false)], push: [:t, :s, :w], body: body}, bindings: 3}
  end

  # physical pool row of (slot[tt], pos[tt]) for a write (0xFFFFFFFF: skipped)
  defp paged_write_row(tag, tt, sh) do
    n = &{tag, &1}

    ld(2, tt, n.(:sl)) ++ ld(3, tt, n.(:p)) ++
      [{n.(:blk), :OpShiftRightLogical, :u32, [n.(:p), u(sh)]},
       {n.(:ok1), :OpULessThan, :bool, [n.(:sl), {:push, :ns}]},
       {n.(:ok2), :OpULessThan, :bool, [n.(:blk), {:push, :mp}]},
       {n.(:ok), :OpLogicalAnd, :bool, [n.(:ok1), n.(:ok2)]},
       {n.(:tb), :OpIMul, :u32, [n.(:sl), {:push, :mp}]}, {n.(:ti0), :OpIAdd, :u32, [n.(:tb), n.(:blk)]},
       {n.(:ti), :OpSelect, :u32, [n.(:ok), n.(:ti0), u(0)]}] ++
      ld(1, n.(:ti), n.(:pg)) ++
      [{n.(:ok3), :OpULessThan, :bool, [n.(:pg), {:push, :npages}]},
       {n.(:valid), :OpLogicalAnd, :bool, [n.(:ok), n.(:ok3)]},
       {n.(:pb), :OpShiftLeftLogical, :u32, [n.(:pg), u(sh)]},
       {n.(:inp), :OpBitwiseAnd, :u32, [n.(:p), u((1 <<< sh) - 1)]},
       {n.(:row0), :OpIAdd, :u32, [n.(:pb), n.(:inp)]},
       {n.(:row), :OpSelect, :u32, [n.(:valid), n.(:row0), u(0xFFFF_FFFF)]}]
  end

  @doc """
  Paged cache write: row t goes to page `table[slot, pos >> sh]`, row
  `pos & (page − 1)`; out-of-range slots, blocks or pages are skipped, the
  last write to a physical row wins. bindings [pool, table, slot, pos, rows],
  push [t, ns, mp, npages, w]
  """
  def kv_write_paged(sh) do
    body =
      [uvar(:tv), uvar(:shadow)] ++ gidx() ++ [{:total, :OpIMul, :u32, [{:push, :t}, {:push, :w}]}] ++
        guarded(:i, :total,
          [{:tt, :OpUDiv, :u32, [:i, {:push, :w}]}, {:col, :OpUMod, :u32, [:i, {:push, :w}]},
           {nil, :OpStore, nil, [:shadow, u(0)]}] ++
            paged_write_row(:me, :tt, sh) ++
            loop(:scan, :tv, {:push, :t}, fn t2 ->
              paged_write_row(:other, t2, sh) ++
                [{:same, :OpIEqual, :bool, [{:other, :row}, {:me, :row}]}, {:later, :OpUGreaterThan, :bool, [t2, :tt]},
                 {:hit, :OpLogicalAnd, :bool, [:same, :later]}, {:s0, :OpLoad, :u32, [:shadow]},
                 {:s1, :OpSelect, :u32, [:hit, u(1), :s0]}, {nil, :OpStore, nil, [:shadow, :s1]}]
            end) ++
            [{:sh, :OpLoad, :u32, [:shadow]}, {:last, :OpIEqual, :bool, [:sh, u(0)]},
             {:go, :OpLogicalAnd, :bool, [:last, {:me, :valid}]}] ++
            when_(:go, :wr,
              [{:rb, :OpIMul, :u32, [:tt, {:push, :w}]}, {:ri, :OpIAdd, :u32, [:rb, :col]},
               {:db, :OpIMul, :u32, [{:me, :row}, {:push, :w}]}, {:di, :OpIAdd, :u32, [:db, :col]}] ++
                ld(4, :ri, :val) ++ st(0, :di, :val, :o)))

    %{module: %{buffers: [buf(0, true)] ++ for(k <- 1..4, do: buf(k, false)), push: [:t, :ns, :mp, :npages, :w], body: body},
      bindings: 5}
  end

  @doc "out[j, i] = x[i, j]. bindings [out, x], push [r, c, ldo]"
  def transpose do
    body =
      gidx() ++ [{:total, :OpIMul, :u32, [{:push, :r}, {:push, :c}]}] ++
        guarded(:i, :total,
          [{:row, :OpUDiv, :u32, [:i, {:push, :c}]}, {:col, :OpUMod, :u32, [:i, {:push, :c}]},
           {:ob, :OpIMul, :u32, [:col, {:push, :ldo}]}, {:oi, :OpIAdd, :u32, [:ob, :row]}] ++
            ld(1, :i, :val) ++ st(0, :oi, :val, :o))

    %{module: %{buffers: [buf(0, true), buf(1, false)], push: [:r, :c, :ldo], body: body}, bindings: 2}
  end

  @doc """
  Next token per row, the sampling kernel's definition (`Vapor.Runtime.Oracle`):
  greedy (invT = +0) is the first index holding the canonical maximum;
  otherwise `e_j = exp((x_j − max)·invT)`, total and running sums
  sequential, the first `j` whose running sum exceeds `u·total` (else the
  last nonzero `e_j`). bindings [out, logits, params, scratch], push [b, v]
  """
  def sample do
    padded = fn j, tag -> [{{tag, :in}, :OpULessThan, :bool, [j, {:push, :v}]}, {{tag, :jc}, :OpSelect, :u32, [{tag, :in}, j, u(0)]},
                           {{tag, :li}, :OpIAdd, :u32, [:base, {tag, :jc}]}] ++ ldf(1, {tag, :li}, {tag, :x}) end
    e_of = fn tag ->
      [{{tag, :d}, :OpFSub, :f32, [{tag, :x}, :mx]}, {{tag, :dt}, :OpFMul, :f32, [{tag, :d}, :it]}] ++
        (fn -> {code, r} = canon(:exp, {tag, :dt}, {tag, :exp}); code ++ [{{tag, :e}, :OpCopyObject, :f32, [r]}] end).()
    end

    body =
      [farr(:acc, 16), uvar(:jv), uvar(:found), {:tot, :OpVariable, {:ptr, :Function, :f32}, [{:storage, :Function}]},
       {:run, :OpVariable, {:ptr, :Function, :f32}, [{:storage, :Function}]}, uvar(:pick), uvar(:last)] ++
        gidx() ++
        guarded(:i, {:push, :b},
          [{:base, :OpIMul, :u32, [:i, {:push, :v}]}] ++
            for(l <- 0..15, do: [at(:acc, u(l), {:ap, l}), {nil, :OpStore, nil, [{:ap, l}, fneg_max()]}]) ++
            [{:v15, :OpIAdd, :u32, [{:push, :v}, u(15)]}, {:v16, :OpBitwiseAnd, :u32, [:v15, u(0xFFFF_FFF0)]}] ++
            loop(:mx, :jv, :v16, fn j ->
              padded.(j, :m) ++
                [{:mxv, :OpSelect, :f32, [{:m, :in}, {:m, :x}, fneg_max()]}, {:lane, :OpBitwiseAnd, :u32, [j, u(15)]},
                 at(:acc, :lane, :lp), {:a0, :OpLoad, :f32, [:lp]}] ++ max_step(:a1, :a0, :mxv) ++ [{nil, :OpStore, nil, [:lp, :a1]}]
            end) ++
            for(l <- 0..15, do: [at(:acc, u(l), {:fp, l}), {{:fin, l}, :OpLoad, :f32, [{:fp, l}]}]) ++
            (fn -> {code, r} = tree16(for(l <- 0..15, do: {:fin, l}), :red, &max_step/3); code ++ [{:mx, :OpCopyObject, :f32, [r]}] end).() ++
            [{:mxu, :OpBitcast, :u32, [:mx]}, {:pi, :OpIMul, :u32, [:i, u(2)]}, {:pu, :OpIAdd, :u32, [:pi, u(1)]}] ++
            ld(2, :pi, :itu) ++ ldf(2, :pu, :u) ++ [{:it, :OpBitcast, :f32, [:itu]}] ++
            # greedy: the first index whose bits equal the maximum's
            [{nil, :OpStore, nil, [:found, u(0xFFFF_FFFF)]}] ++
            loop(:g, :jv, {:push, :v}, fn j ->
              [{:gli, :OpIAdd, :u32, [:base, j]}] ++ ld(1, :gli, :gx) ++
                [{:geq, :OpIEqual, :bool, [:gx, :mxu]}, {:f0, :OpLoad, :u32, [:found]},
                 {:fnone, :OpIEqual, :bool, [:f0, u(0xFFFF_FFFF)]}, {:ftake, :OpLogicalAnd, :bool, [:geq, :fnone]},
                 {:f1, :OpSelect, :u32, [:ftake, j, :f0]}, {nil, :OpStore, nil, [:found, :f1]}]
            end) ++
            # sampled: total, then the running sum against u·total
            [{nil, :OpStore, nil, [:tot, fzero()]}] ++
            loop(:sum, :jv, {:push, :v}, fn j ->
              padded.(j, :ts) ++ e_of.(:ts) ++
                [{:t0, :OpLoad, :f32, [:tot]}, {:t1, :OpFAdd, :f32, [:t0, {:ts, :e}]}, {nil, :OpStore, nil, [:tot, :t1]}]
            end) ++
            [{:tl, :OpLoad, :f32, [:tot]}, {:tgt, :OpFMul, :f32, [:u, :tl]}, {:tgtu, :OpBitcast, :u32, [:tgt]},
             {nil, :OpStore, nil, [:run, fzero()]}, {nil, :OpStore, nil, [:pick, u(0xFFFF_FFFF)]},
             {nil, :OpStore, nil, [:last, u(0)]}] ++
            loop(:c, :jv, {:push, :v}, fn j ->
              padded.(j, :c) ++ e_of.(:c) ++
                [{:c0, :OpLoad, :f32, [:run]}, {:c1, :OpFAdd, :f32, [:c0, {:c, :e}]}, {nil, :OpStore, nil, [:run, :c1]},
                 {:eu, :OpBitcast, :u32, [{:c, :e}]}, {:nz, :OpINotEqual, :bool, [:eu, u(0)]},
                 {:l0, :OpLoad, :u32, [:last]}, {:l1, :OpSelect, :u32, [:nz, j, :l0]}, {nil, :OpStore, nil, [:last, :l1]},
                 {:c1u, :OpBitcast, :u32, [:c1]}, {:over, :OpUGreaterThan, :bool, [:c1u, :tgtu]},
                 {:p0, :OpLoad, :u32, [:pick]}, {:pnone, :OpIEqual, :bool, [:p0, u(0xFFFF_FFFF)]},
                 {:ptake, :OpLogicalAnd, :bool, [:over, :pnone]}, {:p1, :OpSelect, :u32, [:ptake, j, :p0]},
                 {nil, :OpStore, nil, [:pick, :p1]}]
            end) ++
            [{:pk, :OpLoad, :u32, [:pick]}, {:pkn, :OpIEqual, :bool, [:pk, u(0xFFFF_FFFF)]},
             {:la, :OpLoad, :u32, [:last]}, {:sampled, :OpSelect, :u32, [:pkn, :la, :pk]},
             {:fd, :OpLoad, :u32, [:found]}, {:greedy, :OpIEqual, :bool, [:itu, u(0)]},
             {:res, :OpSelect, :u32, [:greedy, :fd, :sampled]}] ++ st(0, :i, :res, :o))

    %{module: %{buffers: [buf(0, true), buf(1, false), buf(2, false), buf(3, true)], push: [:b, :v],
                no_contraction: true, body: List.flatten(body)}, bindings: 4}
  end

  @doc """
  Causal attention, one invocation per (row, query head), in the canonical
  order of `Vapor.Runtime.Oracle.attend/3`: three passes over the keys
  (maximum, normaliser, weighted values) recompute each score — the same
  bits every time — instead of storing them. Value accumulators live in a
  private array, so `dh ≤ #{@max_dh}` (larger heads are refused at dispatch).

  Contiguous: bindings [out, q, k, v, pos, scratch], push [t, hkv, g, s, dh, w].
  Paged: bindings [out, q, kp, vp, pos, scratch, table, slot],
  push [t, hkv, g, cap, dh, ns, mp, npages, w].
  """
  def attention(scale_bits, layout) do
    paged = match?({:paged, _}, layout)
    cap = if paged, do: {:push, :cap}, else: {:push, :s}

    # physical key row of logical row j (paged reads clamp every index)
    phys = fn tag, j ->
      case layout do
        {:paged, sh} ->
          n = &{tag, &1}

          [{n.(:blk), :OpShiftRightLogical, :u32, [j, u(sh)]}] ++ umin(n.(:bc), n.(:blk), :mpm1) ++
            [{n.(:ti), :OpIAdd, :u32, [:tbase, n.(:bc)]}] ++ ld(6, n.(:ti), n.(:pg)) ++ umin(n.(:pgc), n.(:pg), :npm1) ++
            [{n.(:pb), :OpShiftLeftLogical, :u32, [n.(:pgc), u(sh)]}, {n.(:inp), :OpBitwiseAnd, :u32, [j, u((1 <<< sh) - 1)]},
             {n.(:r), :OpIAdd, :u32, [n.(:pb), n.(:inp)]}]

        :contig ->
          [{{tag, :r}, :OpCopyObject, :u32, [j]}]
      end
    end

    # s_j = dot16(k_j, q) · scale, under `tag`
    score = fn tag, j ->
      n = &{tag, &1}

      phys.({tag, :ph}, j) ++
        [{n.(:kb0), :OpIMul, :u32, [{{tag, :ph}, :r}, {:push, :w}]}, {n.(:kb), :OpIAdd, :u32, [n.(:kb0), :kvoff]}] ++
        for(l <- 0..15, do: {nil, :OpStore, nil, [{:dacc, l}, fzero()]}) ++
        loop(n.(:dot), :cv, :nch, fn c ->
          [{n.(:cb), :OpIMul, :u32, [c, u(16)]}, {n.(:kc), :OpIAdd, :u32, [n.(:kb), n.(:cb)]},
           {n.(:qc), :OpIAdd, :u32, [:qbase, n.(:cb)]}] ++
            Enum.flat_map(0..15, fn l ->
              [{n.({:ki, l}), :OpIAdd, :u32, [n.(:kc), u(l)]}, {n.({:qi, l}), :OpIAdd, :u32, [n.(:qc), u(l)]}] ++
                ldf(2, n.({:ki, l}), n.({:kv, l})) ++ ldf(1, n.({:qi, l}), n.({:qv, l})) ++
                [{n.({:a0, l}), :OpLoad, :f32, [{:dacc, l}]}] ++
                mac(n.({:a0, l}), n.({:kv, l}), n.({:qv, l}), n.({:a1, l}), :canonical) ++
                [{nil, :OpStore, nil, [{:dacc, l}, n.({:a1, l})]}]
            end)
        end) ++
        for(l <- 0..15, do: {n.({:d, l}), :OpLoad, :f32, [{:dacc, l}]}) ++
        (fn -> {code, r} = reduce16(for(l <- 0..15, do: n.({:d, l})), n.(:red)); code ++ [{n.(:s), :OpFMul, :f32, [r, {:c, :f32, scale_bits}]}] end).()
    end

    expo = fn tag, s ->
      {code, r} = canon(:exp, {tag, :arg}, {tag, :exp})
      [{{tag, :arg}, :OpFSub, :f32, [s, :m]}] ++ code ++ [{{tag, :e}, :OpCopyObject, :f32, [r]}]
    end

    lanes_init = fn arr, v -> for(l <- 0..15, do: [at(arr, u(l), {arr, :ip, l}), {nil, :OpStore, nil, [{arr, :ip, l}, v]}]) end
    lanes_load = fn arr -> for(l <- 0..15, do: [at(arr, u(l), {arr, :lp, l}), {{arr, :lv, l}, :OpLoad, :f32, [{arr, :lp, l}]}]) end

    prelude =
      [{:heads, :OpIMul, :u32, [{:push, :hkv}, {:push, :g}]}, {:total, :OpIMul, :u32, [{:push, :t}, :heads]}]

    body =
      [uvar(:jv), uvar(:cv), uvar(:dv), farr(:accm, 16), farr(:accz, 16), farr(:o, @max_dh)] ++
        for(l <- 0..15, do: {{:dacc, l}, :OpVariable, {:ptr, :Function, :f32}, [{:storage, :Function}]}) ++
        gidx() ++ prelude ++
        guarded(:i, :total,
          [{:row, :OpUDiv, :u32, [:i, :heads]}, {:head, :OpUMod, :u32, [:i, :heads]},
           {:kvh, :OpUDiv, :u32, [:head, {:push, :g}]}, {:kvoff, :OpIMul, :u32, [:kvh, {:push, :dh}]},
           {:qw, :OpIMul, :u32, [:heads, {:push, :dh}]}, {:qrow, :OpIMul, :u32, [:row, :qw]},
           {:hoff, :OpIMul, :u32, [:head, {:push, :dh}]}, {:qbase, :OpIAdd, :u32, [:qrow, :hoff]},
           {:nch, :OpShiftRightLogical, :u32, [{:push, :dh}, u(4)]}] ++
            ld(4, :row, :p) ++ [{:sm1, :OpISub, :u32, [cap, u(1)]}] ++ umin(:pl, :p, :sm1) ++
            # sliding window: len = min(p + 1, win) keys from s0 = p + 1 − len
            [{:full, :OpIAdd, :u32, [:pl, u(1)]}, {:haswin, :OpINotEqual, :bool, [{:push, :win}, u(0)]},
             {:wlt, :OpULessThan, :bool, [{:push, :win}, :full]}, {:slide, :OpLogicalAnd, :bool, [:haswin, :wlt]},
             {:len, :OpSelect, :u32, [:slide, {:push, :win}, :full]}, {:s0, :OpISub, :u32, [:full, :len]},
             {:lm1, :OpISub, :u32, [:len, u(1)]},
             {:l15, :OpIAdd, :u32, [:len, u(15)]},
             {:l16, :OpBitwiseAnd, :u32, [:l15, u(0xFFFF_FFF0)]}] ++
            (if paged,
               do: ld(7, :row, :sl) ++
                     [{:nsm1, :OpISub, :u32, [{:push, :ns}, u(1)]}, {:mpm1, :OpISub, :u32, [{:push, :mp}, u(1)]},
                      {:npm1, :OpISub, :u32, [{:push, :npages}, u(1)]}] ++ umin(:slc, :sl, :nsm1) ++
                     [{:tbase, :OpIMul, :u32, [:slc, {:push, :mp}]}],
               else: []) ++
            # pass 1: the canonical maximum over ⌈L/16⌉·16 lanes (padding −FLT_MAX)
            lanes_init.(:accm, fneg_max()) ++
            loop(:p1, :jv, :l16, fn j ->
              [{:in1, :OpULessThan, :bool, [j, :len]}, {:j1r, :OpSelect, :u32, [:in1, j, :lm1]}, {:j1, :OpIAdd, :u32, [:j1r, :s0]}] ++
                score.(:s1, :j1) ++
                [{:x1, :OpSelect, :f32, [:in1, {:s1, :s}, fneg_max()]}, {:ln1, :OpBitwiseAnd, :u32, [j, u(15)]},
                 at(:accm, :ln1, :mp1), {:ma, :OpLoad, :f32, [:mp1]}] ++ max_step(:mb, :ma, :x1) ++ [{nil, :OpStore, nil, [:mp1, :mb]}]
            end) ++
            lanes_load.(:accm) ++
            (fn -> {code, r} = tree16(for(l <- 0..15, do: {:accm, :lv, l}), :mred, &max_step/3); code ++ [{:m, :OpCopyObject, :f32, [r]}] end).() ++
            # pass 2: the canonical sum of e_j (padding +0), its canonical reciprocal
            lanes_init.(:accz, fzero()) ++
            loop(:p2, :jv, :l16, fn j ->
              [{:in2, :OpULessThan, :bool, [j, :len]}, {:j2r, :OpSelect, :u32, [:in2, j, :lm1]}, {:j2, :OpIAdd, :u32, [:j2r, :s0]}] ++
                score.(:s2, :j2) ++
                expo.(:e2, {:s2, :s}) ++
                [{:x2, :OpSelect, :f32, [:in2, {:e2, :e}, fzero()]}, {:ln2, :OpBitwiseAnd, :u32, [j, u(15)]},
                 at(:accz, :ln2, :zp2), {:za, :OpLoad, :f32, [:zp2]}, {:zb, :OpFAdd, :f32, [:za, :x2]}, {nil, :OpStore, nil, [:zp2, :zb]}]
            end) ++
            lanes_load.(:accz) ++
            (fn -> {code, r} = tree16(for(l <- 0..15, do: {:accz, :lv, l}), :zred, &add_step/3); code ++ [{:z, :OpCopyObject, :f32, [r]}] end).() ++
            (fn -> {code, r} = canon(:rcp, :z, :inv); code ++ [{:inv, :OpCopyObject, :f32, [r]}] end).() ++
            # pass 3: o[d] = Σ_j (e_j·inv)·v_j[d], sequential in j from +0
            loop(:oz, :dv, {:push, :dh}, fn d -> [at(:o, d, :oz0), {nil, :OpStore, nil, [:oz0, fzero()]}] end) ++
            loop(:p3, :jv, :len, fn j ->
              [{:j3, :OpIAdd, :u32, [j, :s0]}] ++ score.(:s3, :j3) ++ expo.(:e3, {:s3, :s}) ++
                [{:pj, :OpFMul, :f32, [{:e3, :e}, :inv]}] ++ phys.(:vr, :j3) ++
                [{:vb0, :OpIMul, :u32, [{:vr, :r}, {:push, :w}]}, {:vb, :OpIAdd, :u32, [:vb0, :kvoff]}] ++
                loop(:acc, :dv, {:push, :dh}, fn d ->
                  [{:vi, :OpIAdd, :u32, [:vb, d]}] ++ ldf(3, :vi, :vx) ++
                    [at(:o, d, :op3), {:oa, :OpLoad, :f32, [:op3]}, {:vp, :OpFMul, :f32, [:vx, :pj]},
                     {:ob, :OpFAdd, :f32, [:oa, :vp]}, {nil, :OpStore, nil, [:op3, :ob]}]
                end)
            end) ++
            loop(:out, :dv, {:push, :dh}, fn d ->
              [{:oi, :OpIAdd, :u32, [:qbase, d]}, at(:o, d, :opo), {:ov, :OpLoad, :f32, [:opo]}] ++ stf(0, :oi, :ov, :st)
            end))

    push = if paged, do: [:t, :hkv, :g, :cap, :dh, :ns, :mp, :npages, :w, :win], else: [:t, :hkv, :g, :s, :dh, :w, :win]
    nb = if paged, do: 8, else: 6

    %{module: %{buffers: [buf(0, true)] ++ for(k <- 1..(nb - 1), do: buf(k, k == 5)), push: push, no_contraction: true,
                body: List.flatten(body)},
      bindings: nb}
  end

end
