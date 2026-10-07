defmodule Vapor.Emit.SpirV do
  @moduledoc """
  Direct SPIR-V synthesis (Axiom 4): a symbolic assembler that interns
  types and constants, numbers ids, orders the module's logical sections
  and serialises 32-bit words — no glslang, no text, no external tool.

  A module description is a map:

      %{caps: [..], exts: [..], local_size: {x, y, z}, version: {1, 3},
        buffers: [%{binding: 0, elem: :u32, writable: true}, ...],
        push: [:n, :rows],            # u32 push-constant members, in order
        body: [inst]}

  Instructions are `{result | nil, op, type | nil, operands}`; operands are
  result names (any term), `{:lit, n}`, `{:str, s}`, `{:c, type, value}`
  (an interned constant) or `{:t, type}` (an interned type id). Built-in
  names: `:gid` (gl_GlobalInvocationID, a loaded uvec3), `:wid` (workgroup
  id), `{:buf, k}` (binding k's variable), `{:push, name}` (a loaded u32
  push constant), `:glsl` (the GLSL.std.450 import).

  With `no_contraction: true`, every `OpFMul`/`OpFAdd`/`OpFSub` result is
  decorated `NoContraction`, forbidding the driver from fusing them — the
  canonical float policy on the GPU.
  """
  import Bitwise

  @magic 0x0723_0203
  @generator 0x5650_4F52

  @op %{
    OpExtension: 10, OpExtInstImport: 11, OpExtInst: 12, OpMemoryModel: 14, OpEntryPoint: 15,
    OpExecutionMode: 16, OpCapability: 17, OpTypeVoid: 19, OpTypeBool: 20, OpTypeInt: 21,
    OpTypeFloat: 22, OpTypeVector: 23, OpTypeArray: 28, OpTypeRuntimeArray: 29, OpTypeStruct: 30,
    OpTypePointer: 32, OpTypeFunction: 33, OpConstantTrue: 41, OpConstantFalse: 42,
    OpConstant: 43, OpConstantComposite: 44, OpFunction: 54, OpFunctionEnd: 56, OpVariable: 59, OpLoad: 61, OpStore: 62,
    OpAccessChain: 65, OpDecorate: 71, OpMemberDecorate: 72, OpCompositeExtract: 81, OpCopyObject: 83,
    OpConvertUToF: 112, OpBitcast: 124, OpFNegate: 127, OpIAdd: 128, OpFAdd: 129, OpISub: 130,
    OpFSub: 131, OpIMul: 132, OpFMul: 133, OpUDiv: 134, OpUMod: 137, OpLogicalOr: 166, OpLogicalAnd: 167,
    OpLogicalNot: 168, OpSelect: 169, OpIEqual: 170, OpINotEqual: 171, OpUGreaterThan: 172, OpULessThan: 176, OpFOrdLessThan: 184, OpFOrdGreaterThan: 186,
    OpShiftRightLogical: 194, OpShiftLeftLogical: 196, OpBitwiseXor: 198, OpBitwiseAnd: 199,
    OpBitFieldSExtract: 202, OpLoopMerge: 246, OpSelectionMerge: 247, OpLabel: 248,
    OpBranch: 249, OpBranchConditional: 250, OpReturn: 253,
    OpTypeCooperativeMatrixKHR: 4456, OpCooperativeMatrixLoadKHR: 4457,
    OpCooperativeMatrixStoreKHR: 4458, OpCooperativeMatrixMulAddKHR: 4459
  }

  # ops whose first two operands are <result type> <result id>
  @typed_result ~w(OpExtInst OpLoad OpAccessChain OpCompositeExtract OpCopyObject OpConvertUToF OpBitcast
                   OpFNegate OpIAdd OpFAdd OpISub OpFSub OpIMul OpFMul OpUDiv OpUMod
                   OpLogicalOr OpLogicalAnd OpLogicalNot OpSelect OpIEqual OpINotEqual OpUGreaterThan
                   OpULessThan OpFOrdLessThan OpFOrdGreaterThan
                   OpShiftRightLogical OpShiftLeftLogical OpBitwiseXor OpBitwiseAnd OpBitFieldSExtract
                   OpVariable OpCooperativeMatrixLoadKHR OpCooperativeMatrixMulAddKHR)a

  @storage %{Input: 1, Function: 7, PushConstant: 9, StorageBuffer: 12}

  def opcode(name), do: Map.fetch!(@op, name)

  # --------------------------------------------------------------- assembly --

  @doc "Assemble a module description into SPIR-V words."
  @spec assemble(map) :: [non_neg_integer]
  def assemble(m) do
    st = %{ids: %{}, next: 1, globals: [], decos: [], interned: %{}}
    {main, st} = id(st, :main)
    {void, st} = intern(st, {:t, :void})
    {fnty, st} = intern(st, {:t, {:fn, :void}})
    {glsl, st} = if m[:glsl], do: id(st, :glsl), else: {nil, st}

    {gidvar, st} = builtin_var(st, :gid_var, 28)
    {widvar, st} = builtin_var(st, :wid_var, 26)
    {st, buf_decls} = declare_buffers(st, m.buffers)
    {st, pc_decl} = declare_push(st, Map.get(m, :push, []))

    nc? = Map.get(m, :no_contraction, false)

    # function body: variables first, then the built-in loads, then user code
    {vars, rest} = Enum.split_with(m.body, &match?({_, :OpVariable, _, _}, &1))

    prelude =
      [{:gid, :OpLoad, {:vec, :u32, 3}, [:gid_var]}, {:wid, :OpLoad, {:vec, :u32, 3}, [:wid_var]}] ++
        for {name, i} <- Enum.with_index(Map.get(m, :push, [])) do
          [{{:pushp, name}, :OpAccessChain, {:ptr, :PushConstant, :u32}, [:pc_var, {:c, :u32, i}]},
           {{:push, name}, :OpLoad, :u32, [{:pushp, name}]}]
        end
        |> List.flatten()

    {fbody, st} = encode_body([{:entry_label, :OpLabel, nil, []}] ++ vars ++ prelude ++ rest, st)

    nc_decos =
      if nc?,
        do: for({r, op, _, _} <- m.body, op in [:OpFMul, :OpFAdd, :OpFSub], do: {:deco, r, 42, []}),
        else: []

    {decos, st} =
      Enum.map_reduce(nc_decos ++ Enum.reverse(st.decos), st, fn
        {:deco, target, d, lits}, st ->
          {t, st} = id(st, target)
          {inst(:OpDecorate, [t, d | lits]), st}

        {:member, target, mem, d, lits}, st ->
          {t, st} = id(st, target)
          {inst(:OpMemberDecorate, [t, mem, d | lits]), st}
      end)

    {lx, ly, lz} = Map.get(m, :local_size, {64, 1, 1})
    {vmaj, vmin} = Map.get(m, :version, {1, 3})
    interface = if {vmaj, vmin} >= {1, 4}, do: [gidvar, widvar] ++ buf_decls ++ pc_decl, else: [gidvar, widvar]

    caps = Enum.map(Map.get(m, :caps, [1]), &inst(:OpCapability, [&1]))
    exts = Enum.map(Map.get(m, :exts, []), &inst(:OpExtension, str(&1)))
    imports = if glsl, do: [inst(:OpExtInstImport, [glsl | str("GLSL.std.450")])], else: []
    modes = Enum.map(Map.get(m, :exec_modes, []), fn {mode, lits} -> inst(:OpExecutionMode, [main, mode | lits]) end)

    words =
      caps ++ exts ++ imports ++
        [inst(:OpMemoryModel, [0, 1]),
         inst(:OpEntryPoint, [5, main | str("main")] ++ interface),
         inst(:OpExecutionMode, [main, 17, lx, ly, lz])] ++
        modes ++ decos ++ Enum.reverse(st.globals) ++
        [inst(:OpFunction, [void, main, 0, fnty])] ++ fbody ++
        [inst(:OpReturn, []), inst(:OpFunctionEnd, [])]

    List.flatten([@magic, vmaj <<< 16 ||| vmin <<< 8, @generator, st.next, 0 | words])
  end

  @doc "Serialise words as a little-endian binary."
  def to_binary(words), do: for(w <- words, into: <<>>, do: <<w::32-little>>)

  @doc "SPIR-V literal string: UTF-8 + NUL, padded to a word boundary."
  def str(s) do
    n = byte_size(s) + 1
    padded = s <> :binary.copy(<<0>>, n + rem(4 - rem(n, 4), 4) - byte_size(s))
    for <<w::32-little <- padded>>, do: w
  end

  defp inst(op, operands) do
    ops = List.flatten(operands)
    [(length(ops) + 1) <<< 16 ||| opcode(op) | ops]
  end

  defp id(st, name) do
    case Map.fetch(st.ids, name) do
      {:ok, i} -> {i, st}
      :error -> {st.next, %{st | ids: Map.put(st.ids, name, st.next), next: st.next + 1}}
    end
  end

  defp global(st, words), do: %{st | globals: [words | st.globals]}
  defp deco(st, d), do: %{st | decos: [d | st.decos]}

  defp builtin_var(st, name, builtin) do
    {ptr, st} = intern(st, {:t, {:ptr, :Input, {:vec, :u32, 3}}})
    {v, st} = id(st, name)
    st = global(st, inst(:OpVariable, [ptr, v, @storage[:Input]]))
    {v, deco(st, {:deco, name, 11, [builtin]})}
  end

  defp declare_buffers(st, bufs) do
    Enum.reduce(bufs, {st, []}, fn %{binding: k, elem: el} = b, {st, acc} ->
      {et, st} = intern(st, {:t, el})
      {arr, st} = id(st, {:rtarr, k})
      st = global(st, inst(:OpTypeRuntimeArray, [arr, et]))
      {str_t, st} = id(st, {:struct, k})
      st = global(st, inst(:OpTypeStruct, [str_t, arr]))
      {ptr, st} = id(st, {:bufptr, k})
      st = global(st, inst(:OpTypePointer, [ptr, @storage[:StorageBuffer], str_t]))
      {v, st} = id(st, {:buf, k})
      st = global(st, inst(:OpVariable, [ptr, v, @storage[:StorageBuffer]]))

      st =
        st
        |> deco({:deco, {:rtarr, k}, 6, [elem_bytes(el)]})
        |> deco({:deco, {:struct, k}, 2, []})
        |> deco({:member, {:struct, k}, 0, 35, [0]})
        |> deco({:deco, {:buf, k}, 34, [0]})
        |> deco({:deco, {:buf, k}, 33, [k]})
        |> then(fn st -> if Map.get(b, :writable, true), do: st, else: deco(st, {:deco, {:buf, k}, 24, []}) end)

      {st, acc ++ [v]}
    end)
  end

  defp declare_push(st, []), do: {st, []}

  defp declare_push(st, names) do
    {u32, st} = intern(st, {:t, :u32})
    {s, st} = id(st, :pc_struct)
    st = global(st, inst(:OpTypeStruct, [s | List.duplicate(u32, length(names))]))
    {ptr, st} = id(st, :pc_ptr)
    st = global(st, inst(:OpTypePointer, [ptr, @storage[:PushConstant], s]))
    {v, st} = id(st, :pc_var)
    st = global(st, inst(:OpVariable, [ptr, v, @storage[:PushConstant]]))
    st = deco(st, {:deco, :pc_struct, 2, []})

    st =
      names
      |> Enum.with_index()
      |> Enum.reduce(st, fn {_, i}, st -> deco(st, {:member, :pc_struct, i, 35, [4 * i]}) end)

    {st, [v]}
  end

  defp elem_bytes(:u32), do: 4
  defp elem_bytes(:i32), do: 4
  defp elem_bytes(:f32), do: 4
  defp elem_bytes(:i8), do: 1

  # intern types and constants (dependencies first) into the global section
  defp intern(st, key) do
    case Map.fetch(st.interned, key) do
      {:ok, i} -> {i, st}
      :error ->
        {words_fun, st} = declare(st, key)
        {i, st} = {st.next, %{st | next: st.next + 1}}
        st = %{st | interned: Map.put(st.interned, key, i)}
        {i, global(st, words_fun.(i))}
    end
  end

  defp declare(st, {:t, :void}), do: {&inst(:OpTypeVoid, [&1]), st}
  defp declare(st, {:t, :bool}), do: {&inst(:OpTypeBool, [&1]), st}
  defp declare(st, {:t, :u32}), do: {&inst(:OpTypeInt, [&1, 32, 0]), st}
  defp declare(st, {:t, :i32}), do: {&inst(:OpTypeInt, [&1, 32, 1]), st}
  defp declare(st, {:t, :i8}), do: {&inst(:OpTypeInt, [&1, 8, 1]), st}
  defp declare(st, {:t, :f32}), do: {&inst(:OpTypeFloat, [&1, 32]), st}

  defp declare(st, {:t, {:vec, e, n}}) do
    {et, st} = intern(st, {:t, e})
    {&inst(:OpTypeVector, [&1, et, n]), st}
  end

  # fixed-size arrays (Function-storage scratch indexed at run time)
  defp declare(st, {:t, {:array, e, n}}) do
    {et, st} = intern(st, {:t, e})
    {len, st} = intern(st, {:c, :u32, n})
    {&inst(:OpTypeArray, [&1, et, len]), st}
  end

  defp declare(st, {:t, {:ptr, sc, e}}) do
    {et, st} = intern(st, {:t, e})
    {&inst(:OpTypePointer, [&1, @storage[sc], et]), st}
  end

  defp declare(st, {:t, {:fn, r}}) do
    {rt, st} = intern(st, {:t, r})
    {&inst(:OpTypeFunction, [&1, rt]), st}
  end

  defp declare(st, {:t, {:coopmat, comp, scope, rows, cols, use}}) do
    {ct, st} = intern(st, {:t, comp})
    {sc, st} = intern(st, {:c, :u32, scope})
    {r, st} = intern(st, {:c, :u32, rows})
    {c, st} = intern(st, {:c, :u32, cols})
    {u, st} = intern(st, {:c, :u32, use})
    {&inst(:OpTypeCooperativeMatrixKHR, [&1, ct, sc, r, c, u]), st}
  end

  defp declare(st, {:c, {:coopmat, comp, _, _, _, _} = ty, v}) do
    {t, st} = intern(st, {:t, ty})
    {e, st} = intern(st, {:c, comp, v})
    {&inst(:OpConstantComposite, [t, &1, e]), st}
  end

  defp declare(st, {:c, :bool, true}) do
    {bt, st} = intern(st, {:t, :bool})
    {&inst(:OpConstantTrue, [bt, &1]), st}
  end

  defp declare(st, {:c, :f32, bits}) do
    {t, st} = intern(st, {:t, :f32})
    {&inst(:OpConstant, [t, &1, bits]), st}
  end

  defp declare(st, {:c, ty, v}) when ty in [:u32, :i32, :i8] do
    {t, st} = intern(st, {:t, ty})
    {&inst(:OpConstant, [t, &1, if(ty == :i8, do: v &&& 0xFF, else: v &&& 0xFFFF_FFFF)]), st}
  end

  defp encode_body(body, st) do
    Enum.map_reduce(body, st, fn {res, op, ty, operands}, st ->
      {tid, st} = if ty, do: intern(st, {:t, ty}), else: {nil, st}
      {rid, st} = if res, do: id(st, res), else: {nil, st}

      {ops, st} =
        Enum.map_reduce(operands, st, fn
          {:lit, n}, st -> {n, st}
          {:str, s}, st -> {str(s), st}
          {:c, _, _} = c, st -> intern(st, c)
          {:t, _} = t, st -> intern(st, t)
          {:storage, sc}, st -> {@storage[sc], st}
          name, st -> id(st, name)
        end)

      head =
        cond do
          op in @typed_result -> [tid, rid]
          op == :OpLabel -> [rid]
          true -> []
        end

      {inst(op, head ++ ops), st}
    end)
  end
end
