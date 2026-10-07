defmodule Vapor.Emit.MSL do
  @moduledoc """
  **Metal Shading Language from the same kernel library as the GPU path** —
  a translator, not a second kernel library.

  `Vapor.Emit.SpirvKernels` writes every GPU kernel once, as a symbolic
  SPIR-V module (`{result, op, type, operands}` instructions with structured
  control flow). This module translates such a module into one MSL compute
  function, so the Metal kernels are the Vulkan kernels by construction:
  the same canonical order (16 accumulators per row, the fixed
  `r8 → r4 → r2 → r` tree), the same bit tricks (bf16 widening, signed
  extraction), and every new kernel is Metal-ready the day it is written.

  Translation is mechanical and total on the subset the library uses:

  * every SSA result becomes a local declared at the top (`vN`), every
    `Function` variable a local (or a local array);
  * storage-buffer access chains become `bK[i]`; buffers are `device`
    (or `const device`) arrays of the element type; push constants arrive
    as one `constant uint*` argument after the buffers (Metal's `setBytes`);
  * structured control flow is rebuilt from the merge annotations:
    `OpSelectionMerge` → `if`/`else`, `OpLoopMerge` → `while (true)` with
    `break`/`continue` — MSL has no `goto`;
  * integer arithmetic on signed values goes through `uint` (wrap-around,
    never C++ signed overflow), floats keep one operation per statement.

  Floating point: a canonical module is compiled by the Metal daemon in
  `MTLMathModeSafe` (no transformation that could change a result, so no
  contraction of `a·b + c`); whether a device honours that is not assumed
  but measured on arrival by `Vapor.Substrate` (contraction, flush to zero,
  the kernels against the oracle), which admits the device as canonical,
  as envelope-bound, or refuses it.

  The emitted source also carries, under `#ifndef __METAL_VERSION__`, a
  C++ trampoline (`vapor_entry`) that runs the whole grid on a CPU: with
  a ten-line shim for the Metal types, the very same text compiles with
  a host C++ compiler (in the tests, never in the product), which is how the kernels are executed and compared bit for bit
  against the oracle on a machine without a Mac (`test/support/msl`).
  The cooperative-matrix variant has no MSL form (`compile/2` → `nil`):
  it is selected only on Vulkan devices that report it.
  """
  import Bitwise
  alias Vapor.Emit.SpirvKernels

  @local 64

  @doc "Threads per threadgroup of every MSL kernel (the SPIR-V local size)."
  def local_size, do: @local

  @doc """
  `%{src, nbind, npush}` for a kernel key, or `nil` when the key has no MSL
  form. Memoized per `{key, policy}` (translation is deterministic).
  """
  def compile(key, policy) do
    cache = {__MODULE__, key, policy}

    case :persistent_term.get(cache, :none) do
      :none ->
        r =
          if SpirvKernels.supported?(key) and key != :gemm_i8_coop do
            %{module: m, bindings: nb} = SpirvKernels.kernel(key, policy)
            %{src: source(m, name: name(key)), nbind: nb, npush: length(Map.get(m, :push, []))}
          end

        :persistent_term.put(cache, r)
        r

      r ->
        r
    end
  end

  @doc "The function name used for a kernel key (stable, C-identifier safe)."
  def name(key), do: "vapor_" <> (:crypto.hash(:sha256, :erlang.term_to_binary(key)) |> binary_part(0, 6) |> Base.encode16(case: :lower))

  @doc "Translate a symbolic SPIR-V module (as built by `Vapor.Emit.SpirvKernels`) to MSL source."
  def source(%{body: body, buffers: bufs} = m, opts \\ []) do
    if Map.get(m, :local_size, {@local, 1, 1}) != {@local, 1, 1}, do: raise(ArgumentError, "MSL kernels use a local size of #{@local}")
    fname = Keyword.get(opts, :name, "vapor_kernel")
    push = Map.get(m, :push, [])
    canonical? = Map.get(m, :no_contraction, false)

    {vars, rest} = Enum.split_with(body, &match?({_, :OpVariable, _, _}, &1))
    st = scan(vars ++ rest, bufs, push)
    blocks = blocks(rest)
    code = blocks |> emit(:__entry, %{stop: nil, loops: [], depth: 0}, st) |> indent()

    # declarations grouped by type, a dozen names a line
    decls =
      st.decls
      |> Enum.reverse()
      |> Enum.group_by(fn {_, ty} -> if match?({:array, _, _}, ty), do: ty, else: ctype(ty) end, &elem(&1, 0))
      |> Enum.sort()
      |> Enum.map_join("", fn
        {{:array, _, _} = ty, names} -> Enum.map_join(names, "", &"    #{decl(ty, &1)};\n")
        {ct, names} -> names |> Enum.chunk_every(12) |> Enum.map_join("", &"    #{ct} #{Enum.join(&1, ", ")};\n")
      end)

    params =
      (Enum.map(bufs, fn %{binding: k, elem: el} = b ->
         q = if Map.get(b, :writable, true), do: "device", else: "const device"
         "#{q} #{ctype(el)}* b#{k} [[buffer(#{k})]]"
       end) ++
         if(push == [], do: [], else: ["constant uint* pc [[buffer(#{length(bufs)})]]"]) ++
         ["uint3 gid [[thread_position_in_grid]]", "uint3 wid [[threadgroup_position_in_grid]]"])
      |> Enum.join(",\n    ")

    pushes = push |> Enum.with_index() |> Enum.map_join("", fn {_, i} -> "    const uint p#{i} = pc[#{i}];\n" end)

    """
    // vapor kernel #{fname} — translated from the SPIR-V kernel library (Vapor.Emit.MSL)
    #include <metal_stdlib>
    using namespace metal;
    #{if canonical?, do: "// canonical policy: one rounding per operation; compile with MTLMathModeSafe\n", else: ""}
    kernel void #{fname}(
        #{params})
    {
    #{pushes}#{decls}#{code}}
    #{trampoline(fname, bufs, push)}
    """
  end

  # nested blocks indented by their depth (the emitter writes flat lines)
  defp indent(code) do
    code
    |> String.split("\n", trim: true)
    |> Enum.map_reduce(1, fn line, d ->
      t = String.trim_leading(line)
      d = if String.starts_with?(t, "}"), do: d - 1, else: d
      {String.duplicate("    ", d) <> t, if(String.ends_with?(t, "{"), do: d + 1, else: d)}
    end)
    |> elem(0)
    |> Enum.map_join("", &(&1 <> "\n"))
  end

  # ------------------------------------------------------------- scanning --

  # names, declarations and the type of every value
  defp scan(insts, bufs, push) do
    st = %{names: %{}, decls: [], types: %{gid: {:vec, :u32, 3}, wid: {:vec, :u32, 3}}, chains: %{}, n: 0,
           bufs: Map.new(bufs, &{&1.binding, &1.elem}), push: push |> Enum.with_index() |> Map.new()}

    Enum.reduce(insts, st, fn
      {nil, _, _, _}, st -> st
      {res, :OpLabel, _, _}, st -> put_name(st, res, nil)
      {res, :OpVariable, {:ptr, :Function, t}, _}, st -> st |> put_name(res, t) |> declare(res, t)
      {res, :OpAccessChain, {:ptr, _, t}, chain}, st -> st |> put_name(res, {:ptr, t}) |> Map.update!(:chains, &Map.put(&1, res, chain))
      {res, _op, ty, _}, st -> st |> put_name(res, ty) |> declare(res, ty)
    end)
  end

  defp put_name(st, res, ty) do
    if Map.has_key?(st.names, res), do: raise(ArgumentError, "result #{inspect(res)} defined twice")
    %{st | names: Map.put(st.names, res, "v#{st.n}"), types: Map.put(st.types, res, ty), n: st.n + 1}
  end

  defp declare(st, res, ty), do: %{st | decls: [{st.names[res], ty} | st.decls]}

  defp decl({:array, e, n}, name), do: "#{ctype(e)} #{name}[#{n}]"
  defp decl(ty, name), do: "#{ctype(ty)} #{name}"

  defp ctype(:u32), do: "uint"
  defp ctype(:i32), do: "int"
  defp ctype(:f32), do: "float"
  defp ctype(:bool), do: "bool"
  defp ctype(:i8), do: "char"
  defp ctype({:vec, :u32, 3}), do: "uint3"
  defp ctype(t), do: raise(ArgumentError, "no MSL type for #{inspect(t)}")

  # --------------------------------------------------------------- blocks --

  # label → %{insts, merge: {:sel, m} | {:loop, m, c} | nil, term}
  defp blocks(body) do
    {done, cur} =
      Enum.reduce(body, {%{}, %{label: :__entry, insts: [], merge: nil}}, fn
        {l, :OpLabel, _, _}, {acc, cur} ->
          acc = if cur, do: close(acc, cur, {:branch, l}), else: acc
          {acc, %{label: l, insts: [], merge: nil}}

        {nil, :OpSelectionMerge, _, [m | _]}, {acc, cur} ->
          {acc, %{cur | merge: {:sel, m}}}

        {nil, :OpLoopMerge, _, [m, c | _]}, {acc, cur} ->
          {acc, %{cur | merge: {:loop, m, c}}}

        {nil, :OpBranch, _, [l]}, {acc, cur} ->
          {close(acc, cur, {:branch, l}), nil}

        {nil, :OpBranchConditional, _, [c, t, f]}, {acc, cur} ->
          {close(acc, cur, {:cond, c, t, f}), nil}

        {nil, :OpReturn, _, _}, {acc, cur} ->
          {close(acc, cur, :return), nil}

        inst, {acc, cur} ->
          if cur == nil, do: raise(ArgumentError, "instruction after a terminator: #{inspect(inst)}")
          {acc, %{cur | insts: [inst | cur.insts]}}
      end)

    if cur, do: close(done, cur, :return), else: done
  end

  defp close(acc, cur, term), do: Map.put(acc, cur.label, %{insts: Enum.reverse(cur.insts), merge: cur.merge, term: term})

  # ----------------------------------------------------- structured emit --

  defp emit(_blocks, _l, %{depth: d}, _st) when d > 4096, do: raise(ArgumentError, "unstructured control flow")

  defp emit(blocks, label, ctx, st) do
    cond do
      label == ctx.stop ->
        ""

      true ->
        b = Map.fetch!(blocks, label)
        ctx = %{ctx | depth: ctx.depth + 1}

        case b.merge do
          {:loop, merge, cont} ->
            inner = %{ctx | stop: nil, loops: [%{header: label, merge: merge, cont: cont} | ctx.loops]}
            "    while (true) {\n" <> stmts(b.insts, st) <> term(b.term, nil, blocks, inner, st) <> "    }\n" <>
              emit(blocks, merge, ctx, st)

          sel ->
            stmts(b.insts, st) <> term(b.term, sel, blocks, ctx, st)
        end
    end
  end

  defp term(:return, _sel, _blocks, _ctx, _st), do: "    return;\n"

  defp term({:branch, l}, _sel, blocks, ctx, st) do
    loop = List.first(ctx.loops)

    cond do
      l == ctx.stop -> ""
      loop && l == loop.merge -> "    break;\n"
      loop && l == loop.header -> "    continue;\n"
      loop && l == loop.cont -> emit(blocks, l, ctx, st)
      true -> emit(blocks, l, ctx, st)
    end
  end

  defp term({:cond, c, t, f}, {:sel, merge}, blocks, ctx, st) do
    cv = val(c, st)
    arm = fn l -> if l == merge, do: "", else: emit(blocks, l, %{ctx | stop: merge}, st) end
    {tc, fc} = {arm.(t), arm.(f)}

    body =
      cond do
        fc == "" -> "    if (#{cv}) {\n#{tc}    }\n"
        tc == "" -> "    if (!(#{cv})) {\n#{fc}    }\n"
        true -> "    if (#{cv}) {\n#{tc}    } else {\n#{fc}    }\n"
      end

    body <> emit(blocks, merge, ctx, st)
  end

  # a loop's exit test (the only unannotated conditional the library writes)
  defp term({:cond, c, t, f}, nil, blocks, ctx, st) do
    loop = List.first(ctx.loops) || raise(ArgumentError, "conditional branch without a merge outside a loop")
    cv = val(c, st)

    cond do
      f == loop.merge -> "    if (!(#{cv})) break;\n" <> emit(blocks, t, ctx, st)
      t == loop.merge -> "    if (#{cv}) break;\n" <> emit(blocks, f, ctx, st)
      true -> raise ArgumentError, "conditional branch that does not leave the loop"
    end
  end

  # ------------------------------------------------------- instructions --

  defp stmts(insts, st), do: Enum.map_join(insts, "", &stmt(&1, st))

  defp stmt({_res, :OpVariable, _, _}, _st), do: ""
  defp stmt({_res, :OpAccessChain, _, _}, _st), do: ""
  defp stmt({nil, :OpStore, _, [p, v]}, st), do: "    #{place(p, st)} = #{val(v, st)};\n"
  defp stmt({res, op, ty, args}, st), do: "    #{st.names[res]} = #{expr(op, ty, args, st)};\n"

  defp expr(:OpLoad, _ty, [p], st), do: place(p, st)
  defp expr(:OpCompositeExtract, _ty, [v, {:lit, i}], st), do: "#{val(v, st)}.#{elem({"x", "y", "z", "w"}, i)}"
  defp expr(:OpCopyObject, _ty, [v], st), do: val(v, st)
  defp expr(:OpBitcast, ty, [v], st), do: "as_type<#{ctype(ty)}>(#{val(v, st)})"
  defp expr(:OpConvertUToF, _ty, [v], st), do: "float(#{u(v, st)})"
  defp expr(:OpFNegate, _ty, [v], st), do: "(-#{val(v, st)})"
  defp expr(:OpFAdd, _ty, [a, b], st), do: "(#{val(a, st)} + #{val(b, st)})"
  defp expr(:OpFSub, _ty, [a, b], st), do: "(#{val(a, st)} - #{val(b, st)})"
  defp expr(:OpFMul, _ty, [a, b], st), do: "(#{val(a, st)} * #{val(b, st)})"
  defp expr(:OpIAdd, ty, [a, b], st), do: int(ty, "(#{u(a, st)} + #{u(b, st)})")
  defp expr(:OpISub, ty, [a, b], st), do: int(ty, "(#{u(a, st)} - #{u(b, st)})")
  defp expr(:OpIMul, ty, [a, b], st), do: int(ty, "(#{u(a, st)} * #{u(b, st)})")
  defp expr(:OpUDiv, ty, [a, b], st), do: int(ty, "(#{u(a, st)} / #{u(b, st)})")
  defp expr(:OpUMod, ty, [a, b], st), do: int(ty, "(#{u(a, st)} % #{u(b, st)})")
  defp expr(:OpShiftRightLogical, ty, [a, b], st), do: int(ty, "(#{u(a, st)} >> #{u(b, st)})")
  defp expr(:OpShiftLeftLogical, ty, [a, b], st), do: int(ty, "(#{u(a, st)} << #{u(b, st)})")
  defp expr(:OpBitwiseXor, ty, [a, b], st), do: int(ty, "(#{u(a, st)} ^ #{u(b, st)})")
  defp expr(:OpBitwiseAnd, ty, [a, b], st), do: int(ty, "(#{u(a, st)} & #{u(b, st)})")
  defp expr(:OpLogicalOr, _ty, [a, b], st), do: "(#{val(a, st)} || #{val(b, st)})"
  defp expr(:OpLogicalAnd, _ty, [a, b], st), do: "(#{val(a, st)} && #{val(b, st)})"
  defp expr(:OpLogicalNot, _ty, [a], st), do: "(!#{val(a, st)})"
  defp expr(:OpSelect, _ty, [c, a, b], st), do: "(#{val(c, st)} ? #{val(a, st)} : #{val(b, st)})"
  defp expr(:OpIEqual, _ty, [a, b], st), do: "(#{u(a, st)} == #{u(b, st)})"
  defp expr(:OpINotEqual, _ty, [a, b], st), do: "(#{u(a, st)} != #{u(b, st)})"
  defp expr(:OpUGreaterThan, _ty, [a, b], st), do: "(#{u(a, st)} > #{u(b, st)})"
  defp expr(:OpULessThan, _ty, [a, b], st), do: "(#{u(a, st)} < #{u(b, st)})"
  defp expr(:OpFOrdLessThan, _ty, [a, b], st), do: "(#{val(a, st)} < #{val(b, st)})"
  defp expr(:OpFOrdGreaterThan, _ty, [a, b], st), do: "(#{val(a, st)} > #{val(b, st)})"
  # signed bit-field extraction: shift the field to the top, then arithmetic shift down
  defp expr(:OpBitFieldSExtract, ty, [base, off, cnt], st),
    do: "#{ctype(ty)}(as_type<int>(#{u(base, st)} << (32u - #{u(off, st)} - #{u(cnt, st)})) >> int(32u - #{u(cnt, st)}))"
  defp expr(:OpExtInst, _ty, [:glsl, {:lit, 50}, a, b, c], st), do: "fma(#{val(a, st)}, #{val(b, st)}, #{val(c, st)})"
  defp expr(op, _ty, args, _st), do: raise(ArgumentError, "no MSL translation for #{op} #{inspect(args)}")

  # integer results are computed in uint (wrap-around) and reinterpreted
  defp int(:u32, e), do: e
  defp int(:i32, e), do: "as_type<int>(uint#{e})"
  defp int(:i8, e), do: "char(uint#{e})"
  defp int(ty, _), do: raise(ArgumentError, "integer op with result #{inspect(ty)}")

  # an operand as uint (reinterpreting a signed one)
  defp u(v, st) do
    case type_of(v, st) do
      :u32 -> val(v, st)
      :bool -> val(v, st)
      :i32 -> "as_type<uint>(#{val(v, st)})"
      :i8 -> "uint(int(#{val(v, st)}))"
      t -> raise ArgumentError, "integer operand of type #{inspect(t)}: #{inspect(v)}"
    end
  end

  defp type_of({:c, t, _}, _st), do: t
  defp type_of({:push, _}, _st), do: :u32
  defp type_of(v, st), do: Map.fetch!(st.types, v)

  defp place(p, st) do
    case Map.get(st.types, p) do
      {:ptr, _} -> chain(p, st)
      _ -> st.names[p] || raise(ArgumentError, "unknown pointer #{inspect(p)}")
    end
  end

  # the access chain that defined a pointer, as an lvalue
  defp chain(p, st) do
    case Map.fetch!(st.chains, p) do
      [{:buf, k}, {:c, :u32, 0}, i] -> "b#{k}[#{u(i, st)}]"
      [var, i] -> "#{st.names[var]}[#{u(i, st)}]"
      other -> raise ArgumentError, "unsupported access chain #{inspect(other)}"
    end
  end

  defp val({:c, :u32, n}, _st), do: "#{n &&& 0xFFFF_FFFF}u"
  defp val({:c, :i32, n}, _st), do: "as_type<int>(#{n &&& 0xFFFF_FFFF}u)"
  defp val({:c, :i8, n}, _st), do: "char(#{n})"
  defp val({:c, :f32, bits}, _st), do: "as_type<float>(0x#{bits |> Integer.to_string(16) |> String.pad_leading(8, "0")}u)"
  defp val({:c, :bool, true}, _st), do: "true"
  defp val({:c, :bool, false}, _st), do: "false"
  defp val({:push, name}, st), do: "p#{Map.fetch!(st.push, name)}"
  defp val(:gid, _st), do: "gid"
  defp val(:wid, _st), do: "wid"
  defp val(v, st), do: st.names[v] || raise(ArgumentError, "unknown value #{inspect(v)}")

  # ----------------------------------------------------------- trampoline --

  # The grid on a CPU, for the C++ shim used by the tests (never compiled by Metal).
  defp trampoline(fname, bufs, push) do
    args =
      (Enum.map(bufs, fn %{binding: k, elem: el} -> "(#{ctype(el)}*)b[#{k}]" end) ++
         if(push == [], do: [], else: ["pc"]) ++ ["uint3{gx * #{@local} + t, gy, gz}", "uint3{gx, gy, gz}"])
      |> Enum.join(", ")

    """
    #ifndef __METAL_VERSION__
    extern "C" void vapor_entry(void **b, const uint *pc, uint nx, uint ny, uint nz) {
        (void)pc;
        for (uint gz = 0; gz < nz; gz++)
            for (uint gy = 0; gy < ny; gy++)
                for (uint gx = 0; gx < nx; gx++)
                    for (uint t = 0; t < #{@local}u; t++)
                        #{fname}(#{args});
    }
    #endif
    """
  end
end
