defmodule Vapor.Emit.RVV do
  @moduledoc """
  RV64GCV (RVV 1.0) backend — every instruction a 32-bit word built from
  bit fields (Axiom 4).

  Corrections over the predecessor's encoder, each checked against the
  ratified RVV 1.0 specification and GNU objdump:

    * `vtype` is `vlmul[2:0] | vsew[5:3] | vta[6] | vma[7]` — the predecessor
      had SEW and LMUL swapped (the pre-1.0 draft layout).
    * vector loads/stores live in the LOAD-FP/STORE-FP major opcodes with a
      width field (`vle8.v` = width 000, `vle32.v` = 110), not in OP-V.
    * every kernel is a complete psABI function: `a0 = args`, callee-saved
      `s*`/`fs*` spilled to a 16-byte-aligned frame only when used, and
      `ret` (= `jalr x0, 0(ra)`).

  Scalar FP uses the static rounding mode RNE (rm = 000), never the dynamic
  `frm`, so results cannot depend on inherited state. Vector registers are
  allocated in LMUL-aligned groups; `v0` is reserved for masks, `x31` (t6) is
  backend scratch.

  Vector length agnosticism: `:f16l` values are e32/m4 with `vl = 16`
  (legal for every VLEN ≥ 128, i.e. every RVV 1.0 implementation), so the
  canonical reductions are VLEN-independent; strips use `vsetvli` on the
  remaining count.
  """
  import Bitwise
  import Vapor.Emit.Machine, only: [mi: 4]
  alias Vapor.Emit.Machine

  @behaviour Machine

  @t6 31
  @a0 10

  @vops [:vfadd, :vfsub, :vfmul, :vfma, :vfneg, :vrelu, :vsel_lt, :viadd, :visub, :viand, :vixor,
         :vshl, :vshr, :vfmacc]

  @impl true
  def isa, do: :riscv64

  @impl true
  def files do
    %{
      x: %{count: 32, order: [5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 28, 29, 30,
                              8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
           callee_saved: [8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]},
      f: %{count: 32, order: Enum.to_list(0..7) ++ Enum.to_list(10..17) ++ Enum.to_list(28..31) ++
                               [8, 9] ++ Enum.to_list(18..27),
           callee_saved: [8, 9] ++ Enum.to_list(18..27)},
      v: %{count: 32, order: Enum.to_list(1..31), callee_saved: []}
    }
  end

  @impl true
  def group_factors, do: [8, 4, 2, 1]

  @impl true
  def size(:gpr, _), do: {:x, 1}
  def size(:fpr, _), do: {:f, 1}
  def size(:v1, _), do: {:v, 1}
  def size(:strip, g), do: {:v, g}
  def size(:f16l, _), do: {:v, 4}
  def size(:i8strip, g), do: {:v, min(g, 2)}
  def size(:i16strip, g), do: {:v, 2 * min(g, 2)}
  def size(:i32acc, g), do: {:v, 4 * min(g, 2)}

  @impl true
  def arg_pin, do: {:x, @a0}

  defp gi8(s), do: min(s.g, 2)

  # ------------------------------------------------------------ selection --

  @impl true
  def select({:arg, x, i}, s), do: {[mi(:ld, [x], [s.argp], o: [x, s.argp, 8 * i])], s}
  def select({:li, x, imm}, s), do: {[mi(:li, [x], [], o: [x, imm])], s}
  def select({:mov, x, y}, s), do: {[mi(:addi, [x], [y], o: [x, y, 0])], s}

  def select({:addi, x, y, imm}, s) when imm >= -2048 and imm < 2048,
    do: {[mi(:addi, [x], [y], o: [x, y, imm])], s}

  def select({:addi, x, y, imm}, s),
    do: {[mi(:li, [], [], o: [{:phys, @t6}, imm]), mi(:add, [x], [y], o: [x, y, {:phys, @t6}])], s}

  def select({op, x, y, z}, s) when op in [:add, :sub, :mul], do: {[mi(op, [x], [y, z], o: [x, y, z])], s}

  def select({:label, l}, s), do: {[mi(:label, [], [], label: l)], %{s | vcfg: nil}}
  def select({:jmp, l}, s), do: {[mi(:j, [], [], br: l, uncond: true)], %{s | vcfg: nil}}
  def select({:bnez, x, l}, s), do: {[mi(:bnez, [], [x], br: l, o: [x])], %{s | vcfg: nil}}
  def select({:beqz, x, l}, s), do: {[mi(:beqz, [], [x], br: l, o: [x])], %{s | vcfg: nil}}

  def select({:blt_imm, x, imm, l}, s),
    do: {[mi(:li, [], [], o: [{:phys, @t6}, imm]), mi(:blt, [], [x], br: l, o: [x, {:phys, @t6}])], %{s | vcfg: nil}}
  def select({:bltu, x, y, l}, s), do: {[mi(:bltu, [], [x, y], br: l, o: [x, y])], %{s | vcfg: nil}}
  def select({:ld_u32, x, b, off}, s), do: {[mi(:lwu, [x], [b], o: [x, b, off])], s}
  def select({:ld64, x, b, off}, s), do: {[mi(:ld, [x], [b], o: [x, b, off])], s}
  def select({:st64, x, b, off}, s), do: {[mi(:sd, [], [x, b], o: [x, b, off])], s}
  def select({:shli, x, y, n}, s), do: {[mi(:slli, [x], [y], o: [x, y, n])], s}
  def select({:shri, x, y, n}, s), do: {[mi(:srli, [x], [y], o: [x, y, n])], s}
  def select({:ld_u8, x, b, off}, s), do: {[mi(:lbu, [x], [b], o: [x, b, off])], s}
  def select({:ld_s8, x, b, off}, s), do: {[mi(:lb, [x], [b], o: [x, b, off])], s}
  def select({:st_i32, x, b, off}, s), do: {[mi(:sw, [], [x, b], o: [x, b, off])], s}

  def select({:ldf, f, b, off}, s), do: {[mi(:flw, [f], [b], o: [f, b, off])], s}
  def select({:stf, f, b, off}, s), do: {[mi(:fsw, [], [f, b], o: [f, b, off])], s}

  def select({:lif, f, bits}, s),
    do: {[mi(:li, [], [], o: [{:phys, @t6}, bits]), mi(:fmv_w_x, [f], [], o: [f, {:phys, @t6}])], s}

  def select({:cvt_u8f, f, x}, s), do: {[mi(:fcvt_s_wu, [f], [x], o: [f, x])], s}

  # scalar operands feed .vf/.vx forms directly: nothing to broadcast
  def select({:bcast, _f}, s), do: {[], s}
  def select({op, f, a, b}, s) when op in [:fadd, :fsub, :fmul], do: {[mi(op, [f], [a, b], o: [f, a, b])], s}

  def select({:fmacc, acc, a, b}, %{policy: :fast} = s),
    do: {[mi(:fmadd, [acc], [a, b, acc], o: [acc, a, b, acc])], s}

  def select({:fmacc, acc, a, b}, s) do
    {t, s} = Machine.fresh(s, :fpr)
    {[mi(:fmul, [t], [a, b], o: [t, a, b]), mi(:fadd, [acc], [acc, t], o: [acc, acc, t])], s}
  end

  # ---- strips: vsetvli on the remaining count ----
  def select({:strip, n, ptrs, body}, s) do
    {vl, s} = Machine.fresh(s, :gpr)
    {lp, s} = Machine.label(s, :strip)
    {le, s} = Machine.label(s)
    cfg = {32, s.g, :ta}
    s = %{s | vcfg: nil}
    {vec, s} = Machine.select_all(body, %{s | vcfg: {cfg, :strip}})

    code =
      [mi(:beqz, [], [n], br: le, o: [n]),
       mi(:label, [], [], label: lp),
       mi(:vsetvli, [vl], [n], o: [vl, n, vtypei(cfg)])] ++
        vec ++
        [mi(:slli, [], [vl], o: [{:phys, @t6}, vl, 2])] ++
        Enum.map(ptrs, &mi(:add, [&1], [&1], o: [&1, &1, {:phys, @t6}])) ++
        [mi(:sub, [n], [n, vl], o: [n, n, vl]),
         mi(:bnez, [], [n], br: lp, o: [n]),
         mi(:label, [], [], label: le)]

    {code, %{s | vcfg: nil}}
  end

  # vector loads/stores: inside a strip the configuration is the strip's;
  # outside (f16l values) e32/m4 with vl = 16
  def select({:vld, v, b, off}, s) do
    {addr, pre} = addr(b, off)
    ctx(s, pre ++ [mi(:vle32, [v], [b], o: [v, addr])])
  end

  def select({:vst, v, b, off}, s) do
    {addr, pre} = addr(b, off)
    ctx(s, pre ++ [mi(:vse32, [], [v, b], o: [v, addr])])
  end

  def select(i, s) when is_tuple(i) and elem(i, 0) in @vops do
    {i, s} = hoist(i, s)
    {insts, s} = vop(i, s)
    ctx(s, insts)
  end

  def select({:vlane0, f, v}, s), do: with16(s, [mi(:vfmv_f_s, [f], [v], o: [f, v])])

  def select({:vconst, v, bits}, s) do
    {f, s} = cst(s, :f, bits)
    ctx(s, [mi(:vfmv_v_f, [v], [f], o: [v, f])])
  end

  def select({:vsplat, v, f}, s) do
    {:vr, _, _, size} = v
    {cfg, s} = ensure(s, {32, size, :ta}, :vlmax)
    {cfg ++ [mi(:vfmv_v_f, [v], [f], o: [v, f])], s}
  end

  # ---- f16l: e32, m4, vl = 16 ----
  def select({:vzero16, v}, s), do: with16(s, [mi(:vmv_v_i, [v], [], o: [v, 0])])

  def select({:vld16, v, b, off}, s) do
    {addr, pre} = addr(b, off)
    with16(s, pre ++ [mi(:vle32, [v], [b], o: [v, addr])])
  end

  def select({:vfadd16, d, a, b}, s), do: with16(s, [mi(:vfadd_vv, [d], [a, b], o: [d, a, b])])
  def select({:vmul_sf, d, a, f}, s), do: with16(s, [mi(:vfmul_vf, [d], [a, f], o: [d, a, f])])

  def select({:vld_nib, lo, hi, b, off}, s) do
    {t, s} = Machine.fresh(s, :v1)
    {u, s} = Machine.fresh(s, :v1)
    {w, s} = Machine.fresh(s, :v1)
    {addr, pre} = addr(b, off)
    {c8, s} = ensure(s, {8, 1, :ta}, {:imm, 16})

    part1 =
      c8 ++ pre ++
        [mi(:vle8, [t], [b], o: [t, addr]),
         mi(:vand_vi, [u], [t], o: [u, t, 15]),
         mi(:vsrl_vi, [w], [t], o: [w, t, 4])]

    {c32, s} = ensure(s, {32, 4, :ta}, {:imm, 16})

    part2 =
      c32 ++
        [mi(:vzext_vf4, [lo], [u], o: [lo, u], ec: true),
         mi(:vzext_vf4, [hi], [w], o: [hi, w], ec: true),
         mi(:vfcvt_f_xu, [lo], [lo], o: [lo, lo]),
         mi(:vfcvt_f_xu, [hi], [hi], o: [hi, hi])]

    {part1 ++ part2, s}
  end

  # 16 bfloat16 → 16 binary32: e16 load, zero-extend to e32, shift left 16
  def select({:vld_bf16, v, b, off}, s) do
    h = {:vr, s.next, :v, 2}
    s = %{s | next: s.next + 1}
    {addr, pre} = addr(b, off)
    {c16, s} = ensure(s, {16, 2, :ta}, {:imm, 16})
    {c32, s} = ensure(s, {32, 4, :ta}, {:imm, 16})

    {c16 ++ pre ++ [mi(:vle16, [h], [b], o: [h, addr])] ++ c32 ++
       [mi(:vzext_vf2, [v], [h], o: [v, h], ec: true), mi(:vsll_vi, [v], [v], o: [v, v, 16])], s}
  end

  def select({:vfmacc_mem, acc, w, b, off}, s) do
    {x, s} = Machine.fresh(s, :f16l)
    {addr, pre} = addr(b, off)

    arith =
      if s.policy == :fast,
        do: [mi(:vfmacc_vv, [acc], [acc, w, x], o: [acc, w, x])],
        else: [mi(:vfmul_vv, [x], [w, x], o: [x, w, x]), mi(:vfadd_vv, [acc], [acc, x], o: [acc, acc, x])]

    with16(s, pre ++ [mi(:vle32, [x], [b], o: [x, addr])] ++ arith)
  end

  def select({:vreduce16, f, v}, s) do
    {t, s} = Machine.fresh(s, :f16l)
    {u, s} = Machine.fresh(s, :f16l)

    steps =
      Enum.flat_map([{8, v}, {4, t}, {2, t}, {1, t}], fn {h, src} ->
        [mi(:vsetivli, [], [], o: [{:phys, 0}, h, vtypei({32, 4, :ta})]),
         mi(:vslidedown_vi, [u], [src], o: [u, src, h], ec: true),
         mi(:vfadd_vv, [t], [src, u], o: [t, src, u])]
      end)

    {steps ++ [mi(:vfmv_f_s, [f], [t], o: [f, t])], %{s | vcfg: nil}}
  end

  def select({:vreduce16_max, f, v}, s) do
    {t, s} = Machine.fresh(s, :f16l)
    {u, s} = Machine.fresh(s, :f16l)

    steps =
      Enum.flat_map([{8, v}, {4, t}, {2, t}, {1, t}], fn {h, src} ->
        [mi(:vsetivli, [], [], o: [{:phys, 0}, h, vtypei({32, 4, :ta})]),
         mi(:vslidedown_vi, [u], [src], o: [u, src, h], ec: true),
         mi(:vmflt_vv, [], [src, u], o: [{:phys, 0}, src, u]),
         mi(:vmerge_vvm, [t], [src, u], o: [t, src, u])]
      end)

    {steps ++ [mi(:vfmv_f_s, [f], [t], o: [f, t])], %{s | vcfg: nil}}
  end

  # ---- i8 strips (e8/m g → e16/m 2g → e32/m 4g, tail-undisturbed) ----
  def select({:vzero_i32, acc}, s) do
    {c, s} = ensure(s, {32, 4 * gi8(s), :tu}, :vlmax)
    {c ++ [mi(:vmv_v_i, [acc], [], o: [acc, 0])], s}
  end

  def select({:strip_i8, n, ptrs, body, _tail}, s) do
    {vl, s} = Machine.fresh(s, :gpr)
    {lp, s} = Machine.label(s, :strip)
    {le, s} = Machine.label(s)
    cfg = {8, gi8(s), :tu}
    {vec, s} = Machine.select_all(body, %{s | vcfg: {cfg, :strip}})

    code =
      [mi(:beqz, [], [n], br: le, o: [n]),
       mi(:label, [], [], label: lp),
       mi(:vsetvli, [vl], [n], o: [vl, n, vtypei(cfg)])] ++
        vec ++
        Enum.map(ptrs, &mi(:add, [&1], [&1, vl], o: [&1, &1, vl])) ++
        [mi(:sub, [n], [n, vl], o: [n, n, vl]),
         mi(:bnez, [], [n], br: lp, o: [n]),
         mi(:label, [], [], label: le)]

    {code, %{s | vcfg: nil}}
  end

  def select({:vi8mac, acc, pa, pb}, s) do
    {a, s} = Machine.fresh(s, :i8strip)
    {b, s} = Machine.fresh(s, :i8strip)
    {p, s} = Machine.fresh(s, :i16strip)
    g = gi8(s)

    {[mi(:vle8, [a], [pa], o: [a, pa]),
      mi(:vle8, [b], [pb], o: [b, pb]),
      mi(:vwmul_vv, [p], [a, b], o: [p, a, b], ec: true),
      mi(:vsetvli, [], [], o: [{:phys, 0}, {:phys, 0}, vtypei({16, 2 * g, :tu})]),
      mi(:vwadd_wv, [acc], [acc, p], o: [acc, acc, p])], %{s | vcfg: {{16, 2 * g, :tu}, :strip}}}
  end

  def select({:vred_i32, x, acc}, s) do
    {z, s} = Machine.fresh(s, :v1)
    {c, s} = ensure(s, {32, 4 * gi8(s), :tu}, :vlmax)

    {c ++
       [mi(:vmv_s_x, [z], [], o: [z, {:phys, 0}]),
        mi(:vredsum_vs, [z], [acc, z], o: [z, acc, z]),
        mi(:vmv_x_s, [x], [z], o: [x, z])], s}
  end

  def select(:ret, s), do: {[mi(:ret, [], [], ret: true)], s}

  # vtype bookkeeping: emit a vset only when the configuration changes
  defp with16(s, insts) do
    {c, s} = ensure(s, {32, 4, :ta}, {:imm, 16})
    {c ++ insts, s}
  end

  defp ensure(%{vcfg: {cfg, avl}} = s, cfg, avl), do: {[], s}

  defp ensure(s, cfg, {:imm, n} = avl),
    do: {[mi(:vsetivli, [], [], o: [{:phys, 0}, n, vtypei(cfg)])], %{s | vcfg: {cfg, avl}}}

  defp ensure(s, cfg, :vlmax),
    do: {[mi(:vsetvli, [], [], o: [{:phys, @t6}, {:phys, 0}, vtypei(cfg)])], %{s | vcfg: {cfg, :vlmax}}}

  defp addr(b, 0), do: {b, []}
  defp addr(b, off), do: {{:phys, @t6}, [mi(:addi, [], [b], o: [{:phys, @t6}, b, off])]}

  # ---------------------------------------------- vector operations (VOPs) --
  # Operands: vector registers, or scalars — f registers (float constants and
  # per-row broadcasts) feed the `.vf` forms, x registers the `.vx` forms.

  defp ctx(%{vcfg: {_, :strip}} = s, insts), do: {insts, s}
  defp ctx(s, insts), do: with16(s, insts)

  defp file({:vr, _, f, _}), do: f
  defp file({:phys, @t6}), do: :x

  defp fresh_vec(s), do: Machine.fresh(s, if(match?({_, :strip}, s.vcfg), do: :strip, else: :f16l))

  defp vop({op, d, a, b}, s) when op in [:vfadd, :vfmul] do
    name = %{vfadd: :vfadd, vfmul: :vfmul}[op]
    {a, b} = if file(a) == :f, do: {b, a}, else: {a, b}
    {pre, a, s} = vec_first(a, s)
    {pre ++ [bin(name, d, a, b)], s}
  end

  defp vop({:vfsub, d, a, b}, s) do
    case {file(a), file(b)} do
      {:f, :v} -> {[mi(:vfrsub_vf, [d], [b, a], o: [d, b, a])], s}
      _ ->
        {pre, a, s} = vec_first(a, s)
        {pre ++ [bin(:vfsub, d, a, b)], s}
    end
  end

  defp vop({op, d, a, b}, s) when op in [:viadd, :viand, :vixor] do
    name = %{viadd: :vadd, viand: :vand, vixor: :vxor}[op]
    {a, b} = if file(a) != :v, do: {b, a}, else: {a, b}
    {pa, a, s} = vec_first(a, s)
    {pb, b} = xform(b)
    {pa ++ pb ++ [bin(name, d, a, b)], s}
  end

  defp vop({:visub, d, a, b}, s) do
    case {file(a), file(b)} do
      {sa, :v} when sa != :v ->
        {pa, a} = xform(a)
        {pa ++ [mi(:vrsub_vx, [d], [b | regs([a])], o: [d, b, a])], s}

      _ ->
        {pa, a, s} = vec_first(a, s)
        {pb, b} = xform(b)
        {pa ++ pb ++ [bin(:vsub, d, a, b)], s}
    end
  end

  defp vop({:vshl, d, a, n}, s), do: {[mi(:vsll_vi, [d], [a], o: [d, a, n])], s}
  defp vop({:vshr, d, a, n}, s), do: {[mi(:vsrl_vi, [d], [a], o: [d, a, n])], s}
  defp vop({:vfneg, d, a}, s), do: {[mi(:vfsgnjn_vv, [d], [a], o: [d, a, a])], s}

  defp vop({:vrelu, d, a}, s) do
    {z, s} = cst(s, :f, 0)

    {[mi(:vmfgt_vf, [], [a, z], o: [{:phys, 0}, a, z]),
      mi(:vmnand_mm, [], [], o: [{:phys, 0}, {:phys, 0}, {:phys, 0}]),
      mi(:vmerge_vim, [d], [a], o: [d, a, 0], ec: true)], s}
  end

  defp vop({:vfma, d, a, b, c}, %{policy: :fast} = s) do
    {init, s} =
      if file(c) == :f,
        do: {[mi(:vfmv_v_f, [d], [c], o: [d, c])], s},
        else: {[mi(:vmv_v_v, [d], [c], o: [d, c])], s}

    {a, b} = if file(a) == :f, do: {a, b}, else: {b, a}
    {pre, b, s} = if file(a) == :f and file(b) == :f, do: to_vec(b, s), else: {[], b, s}

    fma =
      if file(a) == :f,
        do: mi(:vfmacc_vf, [d], [d, a, b], o: [d, b, a]),
        else: mi(:vfmacc_vv, [d], [d, a, b], o: [d, a, b])

    {pre ++ init ++ [fma], s}
  end

  defp vop({:vfma, d, a, b, c}, s) do
    {t, s} = fresh_vec(s)
    {m, s} = vop({:vfmul, t, a, b}, s)
    {m ++ [bin(:vfadd, d, t, c)], s}
  end

  defp vop({:vfmacc, acc, a, b}, s), do: vop({:vfma, acc, a, b, acc}, s)

  # d = (a < b) ? x : y — the mask lives in v0 (reserved), then a merge
  defp vop({:vsel_lt, d, a, b, x, y}, s) do
    {pre, a, s} = if file(a) == :f and file(b) == :f, do: to_vec(a, s), else: {[], a, s}
    invert = file(y) == :f and file(x) == :v
    {[cmp], s} = {[compare(a, b, invert)], s}

    merge =
      case {file(x), file(y)} do
        {:v, :v} -> [mi(:vmerge_vvm, [d], [y, x], o: [d, y, x])]
        {:f, :v} -> [mi(:vfmerge_vfm, [d], [y, x], o: [d, y, x])]
        {:v, :f} -> [mi(:vfmerge_vfm, [d], [x, y], o: [d, x, y])]
        {:f, :f} -> [mi(:vfmv_v_f, [d], [y], o: [d, y]), mi(:vfmerge_vfm, [d], [d, x], o: [d, d, x])]
      end

    {pre ++ [cmp] ++ merge, s}
  end

  # v0 ← a < b, or its complement a ≥ b (the canonical programs are finite)
  defp compare(a, b, false) do
    case {file(a), file(b)} do
      {:v, :v} -> mi(:vmflt_vv, [], [a, b], o: [{:phys, 0}, a, b])
      {:v, :f} -> mi(:vmflt_vf, [], [a, b], o: [{:phys, 0}, a, b])
      {:f, :v} -> mi(:vmfgt_vf, [], [b, a], o: [{:phys, 0}, b, a])
    end
  end

  defp compare(a, b, true) do
    case {file(a), file(b)} do
      {:v, :v} -> mi(:vmfle_vv, [], [b, a], o: [{:phys, 0}, b, a])
      {:v, :f} -> mi(:vmfge_vf, [], [a, b], o: [{:phys, 0}, a, b])
      {:f, :v} -> mi(:vmfle_vf, [], [b, a], o: [{:phys, 0}, b, a])
    end
  end

  defp to_vec(f, s) do
    {t, s} = fresh_vec(s)
    {[mi(:vfmv_v_f, [t], [f], o: [t, f])], t, s}
  end

  # .vv or .vf/.vx by the second operand's file
  defp bin(name, d, a, b) do
    form = %{v: :vv, f: :vf, x: :vx}[file(b)]
    mi(:"#{name}_#{form}", [d], regs([a, b]), o: [d, a, b])
  end

  defp regs(os), do: Enum.filter(os, &match?({:vr, _, _, _}, &1))

  # a first operand that is scalar (both operands scalar) goes to a vector
  defp vec_first(a, s) do
    case file(a) do
      :v -> {[], a, s}
      :f -> to_vec(a, s)
      :x ->
        {t, s} = fresh_vec(s)
        {[mi(:vmv_v_x, [t], regs([a]), o: [t, a])], t, s}
    end
  end

  # an f-register used by an integer op moves to the scratch x register
  defp xform({:vr, _, :f, _} = f), do: {[mi(:fmv_x_w, [], [f], o: [{:phys, @t6}, f])], {:phys, @t6}}
  defp xform(o), do: {[], o}

  # constants become hoisted scalar registers: f for float ops, x for integer ops
  defp hoist(i, s) do
    [op | rest] = Tuple.to_list(i)
    kind = if op in [:viadd, :visub, :viand, :vixor], do: :x, else: :f

    {rest, s} =
      Enum.map_reduce(rest, s, fn
        {:const, bits}, s -> cst(s, kind, bits)
        o, s -> {o, s}
      end)

    {List.to_tuple([op | rest]), s}
  end

  defp cst(s, kind, bits) do
    case Map.fetch(s.consts, {kind, bits}) do
      {:ok, r} -> {r, s}
      :error ->
        {r, s} = Machine.fresh(s, if(kind == :f, do: :fpr, else: :gpr))
        {r, %{s | consts: Map.put(s.consts, {kind, bits}, r)}}
    end
  end

  @impl true
  def materialize({:f, 0}, r), do: [mi(:fmv_w_x, [r], [], o: [r, {:phys, 0}])]
  def materialize({:f, bits}, r), do: [mi(:li, [], [], o: [{:phys, @t6}, bits]), mi(:fmv_w_x, [r], [], o: [r, {:phys, @t6}])]
  def materialize({:x, bits}, r), do: [mi(:li, [r], [], o: [r, bits])]

  @doc "vtype immediate (RVV 1.0 §3.4): vlmul[2:0] | vsew[5:3] | vta[6] | vma[7]."
  def vtypei({sew, lmul, pol}) do
    vsew = %{8 => 0, 16 => 1, 32 => 2, 64 => 3}[sew]
    vlmul = %{1 => 0, 2 => 1, 4 => 2, 8 => 3}[lmul]
    ta = if pol == :ta, do: 1, else: 0
    vlmul ||| vsew <<< 3 ||| ta <<< 6 ||| ta <<< 7
  end

  # ------------------------------------------------------------- frame --

  defp frame(%{x: xs, f: fs}), do: {xs, fs, (8 * (length(xs) + length(fs)) + 15) &&& -16}

  @impl true
  def prologue(used) do
    {xs, fs, size} = frame(Map.merge(%{x: [], f: []}, used))

    if size == 0 do
      <<>>
    else
      [itype(0b0010011, 2, 0, 2, -size)] ++
        (Enum.with_index(xs) |> Enum.map(fn {r, i} -> stype(0b0100011, 3, 2, r, 8 * i) end)) ++
        (Enum.with_index(fs)
         |> Enum.map(fn {r, i} -> stype(0b0100111, 3, 2, r, 8 * (length(xs) + i)) end))
      |> IO.iodata_to_binary()
    end
  end

  @impl true
  def epilogue(used) do
    {xs, fs, size} = frame(Map.merge(%{x: [], f: []}, used))

    restore =
      if size == 0 do
        []
      else
        (Enum.with_index(xs) |> Enum.map(fn {r, i} -> itype(0b0000011, r, 3, 2, 8 * i) end)) ++
          (Enum.with_index(fs)
           |> Enum.map(fn {r, i} -> itype(0b0000111, r, 3, 2, 8 * (length(xs) + i)) end)) ++
          [itype(0b0010011, 2, 0, 2, size)]
      end

    IO.iodata_to_binary(restore ++ [itype(0b1100111, 0, 0, 1, 0)])
  end

  # ------------------------------------------------------------- encoding --

  @impl true
  def branch_size(:j), do: 4
  def branch_size(_), do: 8

  # conditional branches are `b<inverse> x, zero, +8 ; jal x0, target`, giving
  # ±1 MiB reach independent of kernel size.
  @impl true
  def encode_branch(:j, _o, rel, _r), do: jal(0, rel)
  def encode_branch(:bnez, [x], rel, r), do: btype(0, r.(x), 0, 8) <> jal(0, rel - 4)
  def encode_branch(:beqz, [x], rel, r), do: btype(1, r.(x), 0, 8) <> jal(0, rel - 4)
  def encode_branch(:blt, [x, y], rel, r), do: btype(5, r.(x), r.(y), 8) <> jal(0, rel - 4)
  def encode_branch(:bltu, [x, y], rel, r), do: btype(7, r.(x), r.(y), 8) <> jal(0, rel - 4)

  @impl true
  def encode(:ld, [d, b, off], r), do: itype(0b0000011, r.(d), 3, r.(b), off)
  def encode(:lbu, [d, b, off], r), do: itype(0b0000011, r.(d), 4, r.(b), off)
  def encode(:lb, [d, b, off], r), do: itype(0b0000011, r.(d), 0, r.(b), off)
  def encode(:lwu, [d, b, off], r), do: itype(0b0000011, r.(d), 6, r.(b), off)
  def encode(:fmv_x_w, [d, f], r), do: rtype(0b1010011, r.(d), 0, r.(f), 0, 0b1110000)
  def encode(:sw, [x, b, off], r), do: stype(0b0100011, 2, r.(b), r.(x), off)
  def encode(:flw, [d, b, off], r), do: itype(0b0000111, r.(d), 2, r.(b), off)
  def encode(:fsw, [x, b, off], r), do: stype(0b0100111, 2, r.(b), r.(x), off)
  def encode(:addi, [d, a, imm], r), do: if(r.(d) == r.(a) and imm == 0, do: <<>>, else: itype(0b0010011, r.(d), 0, r.(a), imm))
  def encode(:slli, [d, a, sh], r), do: itype(0b0010011, r.(d), 1, r.(a), sh)
  def encode(:srli, [d, a, sh], r), do: itype(0b0010011, r.(d), 5, r.(a), sh)
  def encode(:sd, [x, b, off], r), do: stype(0b0100011, 3, r.(b), r.(x), off)
  def encode(:add, [d, a, b], r), do: rtype(0b0110011, r.(d), 0, r.(a), r.(b), 0)
  def encode(:sub, [d, a, b], r), do: rtype(0b0110011, r.(d), 0, r.(a), r.(b), 0b0100000)
  def encode(:mul, [d, a, b], r), do: rtype(0b0110011, r.(d), 0, r.(a), r.(b), 1)

  def encode(:li, [d, imm], r) do
    rd = r.(d)
    v = if imm >= 0x8000_0000, do: imm - 0x1_0000_0000, else: imm
    unless v >= -0x8000_0000 and v < 0x8000_0000, do: raise(ArgumentError, "li out of 32-bit range")
    hi = (v + 0x800) >>> 12
    lo = v - (hi <<< 12)

    if hi == 0,
      do: itype(0b0010011, rd, 0, 0, lo),
      else: utype(0b0110111, rd, hi &&& 0xFFFFF) <> itype(0b0011011, rd, 0, rd, lo)
  end

  # scalar FP, static rounding mode RNE (rm = 000)
  def encode(:fadd, [d, a, b], r), do: rtype(0b1010011, r.(d), 0, r.(a), r.(b), 0b0000000)
  def encode(:fsub, [d, a, b], r), do: rtype(0b1010011, r.(d), 0, r.(a), r.(b), 0b0000100)
  def encode(:fmul, [d, a, b], r), do: rtype(0b1010011, r.(d), 0, r.(a), r.(b), 0b0001000)
  def encode(:fcvt_s_wu, [d, x], r), do: rtype(0b1010011, r.(d), 0, r.(x), 1, 0b1101000)
  def encode(:fmv_w_x, [d, x], r), do: rtype(0b1010011, r.(d), 0, r.(x), 0, 0b1111000)

  def encode(:fmadd, [d, a, b, c], r),
    do: <<r.(c)::5, 0::2, r.(b)::5, r.(a)::5, 0::3, r.(d)::5, 0b1000011::7>> |> le32()

  # vector configuration
  def encode(:vsetvli, [d, a, vt], r), do: <<0::1, vt::11, r.(a)::5, 7::3, r.(d)::5, 0b1010111::7>> |> le32()
  def encode(:vsetivli, [d, uimm, vt], r), do: <<3::2, vt::10, uimm::5, 7::3, r.(d)::5, 0b1010111::7>> |> le32()

  # unit-stride loads/stores (LOAD-FP / STORE-FP with width)
  def encode(:vle8, [v, b], r), do: vmem(0b0000111, 0, r.(b), r.(v))
  def encode(:vle32, [v, b], r), do: vmem(0b0000111, 0b110, r.(b), r.(v))
  def encode(:vle16, [v, b], r), do: vmem(0b0000111, 0b101, r.(b), r.(v))
  def encode(:vse32, [v, b], r), do: vmem(0b0100111, 0b110, r.(b), r.(v))

  # OP-V: funct6 | vm | vs2 | vs1 | funct3 | vd
  def encode(:vfadd_vv, [d, a, b], r), do: opv(0b000000, 1, r.(a), r.(b), 1, r.(d))
  def encode(:vfsub_vv, [d, a, b], r), do: opv(0b000010, 1, r.(a), r.(b), 1, r.(d))
  def encode(:vfmul_vv, [d, a, b], r), do: opv(0b100100, 1, r.(a), r.(b), 1, r.(d))
  def encode(:vfmacc_vv, [d, a, b], r), do: opv(0b101100, 1, r.(b), r.(a), 1, r.(d))
  def encode(:vfsgnjn_vv, [d, a, b], r), do: opv(0b001001, 1, r.(a), r.(b), 1, r.(d))
  def encode(:vfmul_vf, [d, a, f], r), do: opv(0b100100, 1, r.(a), r.(f), 5, r.(d))
  def encode(:vfmv_v_f, [d, f], r), do: opv(0b010111, 1, 0, r.(f), 5, r.(d))
  def encode(:vmfgt_vf, [d, a, f], r), do: opv(0b011101, 1, r.(a), r.(f), 5, r.(d))
  def encode(:vmnand_mm, [d, a, b], r), do: opv(0b011101, 1, r.(a), r.(b), 2, r.(d))
  def encode(:vmerge_vim, [d, a, imm], r), do: opv(0b010111, 0, r.(a), imm &&& 31, 3, r.(d))
  def encode(:vmv_v_i, [d, imm], r), do: opv(0b010111, 1, 0, imm &&& 31, 3, r.(d))
  def encode(:vmv_v_v, [d, a], r), do: opv(0b010111, 1, 0, r.(a), 0, r.(d))
  def encode(:vmv_v_x, [d, x], r), do: opv(0b010111, 1, 0, r.(x), 4, r.(d))
  def encode(:vand_vi, [d, a, imm], r), do: opv(0b001001, 1, r.(a), imm &&& 31, 3, r.(d))
  def encode(:vsrl_vi, [d, a, imm], r), do: opv(0b101000, 1, r.(a), imm &&& 31, 3, r.(d))
  def encode(:vzext_vf4, [d, a], r), do: opv(0b010010, 1, r.(a), 0b00100, 2, r.(d))
  def encode(:vzext_vf2, [d, a], r), do: opv(0b010010, 1, r.(a), 0b00110, 2, r.(d))
  def encode(:vfcvt_f_xu, [d, a], r), do: opv(0b010010, 1, r.(a), 0b00010, 1, r.(d))
  def encode(:vslidedown_vi, [d, a, imm], r), do: opv(0b001111, 1, r.(a), imm, 3, r.(d))
  def encode(:vfmv_f_s, [f, v], r), do: opv(0b010000, 1, r.(v), 0, 1, r.(f))
  def encode(:vwmul_vv, [d, a, b], r), do: opv(0b111011, 1, r.(a), r.(b), 2, r.(d))
  def encode(:vwadd_wv, [d, a, b], r), do: opv(0b110101, 1, r.(a), r.(b), 2, r.(d))
  def encode(:vredsum_vs, [d, a, b], r), do: opv(0b000000, 1, r.(a), r.(b), 2, r.(d))
  def encode(:vmv_x_s, [x, v], r), do: opv(0b010000, 1, r.(v), 0, 2, r.(x))
  def encode(:vmv_s_x, [v, x], r), do: opv(0b010000, 1, 0, r.(x), 6, r.(v))

  # funct6 and funct3 of the binary forms: OPIVV 0, OPFVV 1, OPIVI 3, OPIVX 4, OPFVF 5
  for {name, f6, f3} <- [
        {:vfadd_vf, 0b000000, 5}, {:vfsub_vf, 0b000010, 5}, {:vfrsub_vf, 0b100111, 5},
        {:vfmacc_vf, 0b101100, 5}, {:vmflt_vf, 0b011011, 5}, {:vmfle_vf, 0b011001, 5},
        {:vmfge_vf, 0b011111, 5}, {:vmflt_vv, 0b011011, 1}, {:vmfle_vv, 0b011001, 1},
        {:vadd_vv, 0b000000, 0}, {:vsub_vv, 0b000010, 0}, {:vand_vv, 0b001001, 0}, {:vxor_vv, 0b001011, 0},
        {:vadd_vx, 0b000000, 4}, {:vsub_vx, 0b000010, 4}, {:vrsub_vx, 0b000011, 4},
        {:vand_vx, 0b001001, 4}, {:vxor_vx, 0b001011, 4}
      ] do
    def encode(unquote(name), [d, a, b], r), do: opv(unquote(f6), 1, r.(a), r.(b), unquote(f3), r.(d))
  end

  def encode(:vsll_vi, [d, a, imm], r), do: opv(0b100101, 1, r.(a), imm &&& 31, 3, r.(d))
  def encode(:vmerge_vvm, [d, a, b], r), do: opv(0b010111, 0, r.(a), r.(b), 0, r.(d))
  def encode(:vfmerge_vfm, [d, a, f], r), do: opv(0b010111, 0, r.(a), r.(f), 5, r.(d))

  defp le32(<<w::32>>), do: <<w::32-little>>

  defp opv(f6, vm, vs2, vs1, f3, vd),
    do: <<f6::6, vm::1, vs2::5, vs1::5, f3::3, vd::5, 0b1010111::7>> |> le32()

  defp vmem(opc, width, rs1, v),
    do: <<0::3, 0::1, 0::2, 1::1, 0::5, rs1::5, width::3, v::5, opc::7>> |> le32()

  defp rtype(opc, rd, f3, rs1, rs2, f7), do: <<f7::7, rs2::5, rs1::5, f3::3, rd::5, opc::7>> |> le32()

  defp itype(opc, rd, f3, rs1, imm) do
    unless imm >= -2048 and imm < 2048, do: raise(ArgumentError, "I-immediate #{imm} out of range")
    <<imm::signed-12, rs1::5, f3::3, rd::5, opc::7>> |> le32()
  end

  defp stype(opc, f3, rs1, rs2, imm) do
    <<hi::7, lo::5>> = <<imm::signed-12>>
    <<hi::7, rs2::5, rs1::5, f3::3, lo::5, opc::7>> |> le32()
  end

  defp utype(opc, rd, imm20), do: <<imm20::20, rd::5, opc::7>> |> le32()

  defp btype(f3, rs1, rs2, off) do
    <<b12::1, b11::1, b10_5::6, b4_1::4, _::1>> = <<off::signed-13>>
    <<b12::1, b10_5::6, rs2::5, rs1::5, f3::3, b4_1::4, b11::1, 0b1100011::7>> |> le32()
  end

  defp jal(rd, off) do
    unless off >= -(1 <<< 20) and off < 1 <<< 20, do: raise(ArgumentError, "jal out of range")
    <<b20::1, b19_12::8, b11::1, b10_1::10, _::1>> = <<off::signed-21>>
    <<b20::1, b10_1::10, b11::1, b19_12::8, rd::5, 0b1101111::7>> |> le32()
  end
end
