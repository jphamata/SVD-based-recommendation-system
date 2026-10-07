defmodule Vapor.Emit.ARM do
  @moduledoc """
  AArch64 (A64 + Advanced SIMD/NEON) backend — every instruction a
  little-endian 32-bit word built from its bit fields.

  AAPCS64: `x0 = args`; callee-saved `x19–x28` and the low halves `d8–d15`
  are saved in a 16-byte-aligned frame only when the allocator used them;
  `x16` (IP0) and `v31` are backend scratch; `x18` (platform), `x29`, `x30`
  are never allocated.

  Register groups: `:strip` = `g` q-registers (4·g lanes), `:f16l` = 4 q,
  `:i32acc` = 2 q (low/high halves of the widened product).
  """
  import Bitwise
  import Vapor.Emit.Machine, only: [mi: 4]
  alias Vapor.Emit.Machine

  @behaviour Machine

  @x16 16
  @v31 31

  @vops [:vfadd, :vfsub, :vfmul, :vfma, :vfneg, :vrelu, :vsel_lt, :viadd, :visub, :viand, :vixor,
         :vshl, :vshr, :vfmacc]

  @impl true
  def isa, do: :aarch64

  @impl true
  def files do
    %{
      x: %{count: 32, order: Enum.to_list(0..15) ++ [17] ++ Enum.to_list(19..28),
           callee_saved: Enum.to_list(19..28)},
      v: %{count: 32, order: Enum.to_list(0..7) ++ Enum.to_list(16..30) ++ Enum.to_list(8..15),
           callee_saved: Enum.to_list(8..15)}
    }
  end

  @impl true
  def group_factors, do: [8, 4, 2, 1]

  @impl true
  def size(:gpr, _), do: {:x, 1}
  def size(:fpr, _), do: {:v, 1}
  def size(:strip, g), do: {:v, g}
  def size(:f16l, _), do: {:v, 4}
  def size(:i32acc, _), do: {:v, 2}

  @impl true
  def arg_pin, do: {:x, 0}

  # ------------------------------------------------------------ selection --

  @impl true
  def select({:arg, x, i}, s), do: {[mi(:ldr_x, [x], [s.argp], o: [x, s.argp, 8 * i])], s}
  def select({:li, x, imm}, s), do: {[mi(:li, [x], [], o: [x, imm])], s}
  def select({:mov, x, y}, s), do: {[mi(:mov, [x], [y], o: [x, y])], s}
  def select({:addi, x, y, imm}, s), do: {[mi(:addi, [x], [y], o: [x, y, imm])], s}
  def select({op, x, y, z}, s) when op in [:add, :sub, :mul], do: {[mi(op, [x], [y, z], o: [x, y, z])], s}

  def select({:label, l}, s), do: {[mi(:label, [], [], label: l)], s}
  def select({:jmp, l}, s), do: {[mi(:b, [], [], br: l, uncond: true)], s}
  def select({:bnez, x, l}, s), do: {[mi(:cbnz, [], [x], br: l, o: [x])], s}
  def select({:beqz, x, l}, s), do: {[mi(:cbz, [], [x], br: l, o: [x])], s}
  def select({:blt_imm, x, imm, l}, s), do: {[mi(:cmp_imm, [], [x], o: [x, imm]), mi(:blt, [], [], br: l)], s}
  def select({:bltu, x, y, l}, s), do: {[mi(:cmp_rr, [], [x, y], o: [x, y]), mi(:blo, [], [], br: l)], s}
  def select({:ld_u32, x, b, off}, s), do: {[mi(:ldr_w, [x], [b], o: [x, b, off])], s}
  def select({:ld64, x, b, off}, s), do: {[mi(:ldr_x, [x], [b], o: [x, b, off])], s}
  def select({:st64, x, b, off}, s), do: {[mi(:str_x, [], [x, b], o: [x, b, off])], s}
  def select({op, x, y, n}, s) when op in [:shli, :shri], do: {[mi(op, [x], [y], o: [x, y, n])], s}
  def select({:ld_u8, x, b, off}, s), do: {[mi(:ldrb, [x], [b], o: [x, b, off])], s}
  def select({:ld_s8, x, b, off}, s), do: {[mi(:ldrsb, [x], [b], o: [x, b, off])], s}
  def select({:st_i32, x, b, off}, s), do: {[mi(:str_w, [], [x, b], o: [x, b, off])], s}

  def select({:ldf, f, b, off}, s), do: {[mi(:ldr_s, [f], [b], o: [f, b, off])], s}
  def select({:stf, f, b, off}, s), do: {[mi(:str_s, [], [f, b], o: [f, b, off])], s}

  def select({:lif, f, bits}, s),
    do: {[mi(:li, [], [], o: [{:phys, @x16}, bits]), mi(:fmov_s_w, [f], [], o: [f, {:phys, @x16}])], s}

  def select({:cvt_u8f, f, x}, s), do: {[mi(:ucvtf_s_w, [f], [x], o: [f, x])], s}
  def select({:bcast, f}, s), do: {[mi(:dup4s, [f], [f], o: [f, f])], s}
  def select({op, f, a, b}, s) when op in [:fadd, :fsub, :fmul], do: {[mi(op, [f], [a, b], o: [f, a, b])], s}

  def select({:fmacc, acc, a, b}, %{policy: :fast} = s),
    do: {[mi(:fmadd, [acc], [a, b, acc], o: [acc, a, b, acc])], s}

  def select({:fmacc, acc, a, b}, s) do
    {[mi(:fmul, [], [a, b], o: [{:phys, @v31}, a, b]),
      mi(:fadd, [acc], [acc], o: [acc, acc, {:phys, @v31}])], s}
  end

  # ---- strips: 4·g lanes per iteration, then a scalar tail on lane 0 ----
  def select({:strip, n, ptrs, body}, s) do
    w = 4 * s.g
    {tail, s} = Machine.tail_copy(body, s)
    strip(n, ptrs, w, 4 * w, 4, Enum.map(body, &{:lanes, &1}), Enum.map(tail, &{:scalar, &1}), s)
  end

  def select({:strip_i8, n, ptrs, body, tail}, s) do
    w = 8 * s.g
    strip(n, ptrs, w, w, 1, body, tail, s)
  end

  def select({:lanes, {:vld, v, b, off}}, s),
    do: {[lanewise(:ldr_q, [v], [b], lanes(s), &[sub(v, &1), b, off + 16 * &1])], s}

  def select({:lanes, {:vst, v, b, off}}, s),
    do: {[lanewise(:str_q, [], [v, b], lanes(s), &[sub(v, &1), b, off + 16 * &1])], s}

  def select({:lanes, i}, s) when elem(i, 0) in @vops, do: vop(i, lanes(s), s)

  def select({:scalar, {:vld, v, b, off}}, s), do: {[mi(:ldr_s, [v], [b], o: [sub(v, 0), b, off])], s}
  def select({:scalar, {:vst, v, b, off}}, s), do: {[mi(:str_s, [], [v, b], o: [sub(v, 0), b, off])], s}

  # scalar tail: the 4-lane forms on register 0 of each group (the other
  # lanes hold zeros or broadcasts and are never read back)
  def select({:scalar, i}, s) when elem(i, 0) in @vops, do: vop(i, [0], s)

  # bare vector ops on f16l values (outside strips)
  def select({:vld, v, b, off}, s),
    do: {[lanewise(:ldr_q, [v], [b], lanes_of(v), &[sub(v, &1), b, off + 16 * &1])], s}

  def select({:vst, v, b, off}, s),
    do: {[lanewise(:str_q, [], [v, b], lanes_of(v), &[sub(v, &1), b, off + 16 * &1])], s}

  def select(i, s) when is_tuple(i) and elem(i, 0) in @vops, do: vop(i, lanes_of(elem(i, 1)), s)

  def select({:vlane0, f, v}, s), do: {[mi(:mov16b, [f], [v], o: [f, sub(v, 0)])], s}

  def select({:vconst, v, bits}, s),
    do: {Enum.map(lanes_of(v), &mi(:ldr_q_lit, [v], [], o: [sub(v, &1), {:pool, bits}])), s}

  def select({:vsplat, v, f}, s),
    do: {[lanewise(:dup4s, [v], [f], lanes_of(v), &[sub(v, &1), f])], s}

  # ---- f16l = 4 q ----
  def select({:vzero16, v}, s), do: {[lanewise(:movi_zero, [v], [], 0..3, &[sub(v, &1)])], s}

  def select({:vld16, v, b, off}, s),
    do: {[lanewise(:ldr_q, [v], [b], 0..3, &[sub(v, &1), b, off + 16 * &1])], s}

  def select({:vfadd16, d, a, b}, s),
    do: {[lanewise(:fadd4s, [d], [a, b], 0..3, &[sub(d, &1), sub(a, &1), sub(b, &1)])], s}

  def select({:vmul_sf, d, a, f}, s),
    do: {[lanewise(:fmul4s_elem, [d], [a, f], 0..3, &[sub(d, &1), sub(a, &1), f])], s}

  def select({:vfmacc_mem, acc, w, b, off}, s) do
    op = if s.policy == :fast, do: :fmla_mem, else: :fmacc_mem_split
    {[lanewise(op, [acc], [acc, w, b], 0..3, &[sub(acc, &1), sub(w, &1), b, off + 16 * &1])], s}
  end

  # 16 bfloat16 → 16 binary32: SHLL{2} #16 is exactly the widening
  def select({:vld_bf16, v, b, off}, s) do
    t = {:phys, @v31}

    {[mi(:ldr_q, [], [b], o: [t, b, off]),
      mi(:shll_4s, [v], [], o: [sub(v, 0), t]), mi(:shll2_4s, [v], [v], o: [sub(v, 1), t]),
      mi(:ldr_q, [], [b], o: [t, b, off + 16]),
      mi(:shll_4s, [v], [v], o: [sub(v, 2), t]), mi(:shll2_4s, [v], [v], o: [sub(v, 3), t])], s}
  end

  def select({:vld_nib, lo, hi, b, off}, s) do
    {m, s} = Machine.fresh(s, :fpr)
    {t, s} = Machine.fresh(s, :fpr)
    v = {:phys, @v31}

    widen = fn dst, src ->
      [mi(:uxtl_8h, [dst], [src], o: [sub(dst, 0), src]),
       mi(:uxtl2_8h, [dst], [src], o: [sub(dst, 2), src]),
       mi(:uxtl2_4s, [dst], [dst], o: [sub(dst, 1), sub(dst, 0)]),
       mi(:uxtl_4s, [dst], [dst], o: [sub(dst, 0), sub(dst, 0)]),
       mi(:uxtl2_4s, [dst], [dst], o: [sub(dst, 3), sub(dst, 2)]),
       mi(:uxtl_4s, [dst], [dst], o: [sub(dst, 2), sub(dst, 2)]),
       lanewise(:ucvtf4s, [dst], [dst], 0..3, &[sub(dst, &1), sub(dst, &1)])]
    end

    {[mi(:ldr_q_lit, [m], [], o: [m, {:pool, 0x0F0F_0F0F}]),
      mi(:ldr_q, [], [b], o: [v, b, off]),
      mi(:and16b, [t], [m], o: [t, v, m])] ++
       widen.(lo, t) ++
       [mi(:ushr16b_4, [t], [], o: [t, v])] ++
       widen.(hi, t), s}
  end

  def select({:vreduce16, f, v}, s) do
    x = {:phys, @v31}

    {[mi(:fadd4s, [f], [v], o: [f, sub(v, 0), sub(v, 2)]),
      mi(:fadd4s, [], [v], o: [x, sub(v, 1), sub(v, 3)]),
      mi(:fadd4s, [f], [f], o: [f, f, x]),
      mi(:ext8, [], [f], o: [x, f, f]),
      mi(:fadd2s, [f], [f], o: [f, f, x]),
      mi(:faddp_s, [f], [f], o: [f, f])], s}
  end

  def select({:vreduce16_max, f, v}, s) do
    {t, s} = Machine.fresh(s, :fpr)
    mx = fn d, a, b -> mi(:sel4s, [d], [a, b], o: [d, a, b, b, a]) end

    {[mx.(f, sub(v, 0), sub(v, 2)) |> uses([v]),
      mx.(t, sub(v, 1), sub(v, 3)) |> uses([v]),
      mx.(f, f, t) |> uses([f, t]),
      mi(:ext8, [t], [f], o: [t, f, f]),
      mx.(f, f, t) |> uses([f, t]),
      mi(:dup_s1, [t], [f], o: [t, f]),
      mx.(f, f, t) |> uses([f, t])], s}
  end

  # ---- i8 ----
  def select({:vzero_i32, acc}, s), do: {[lanewise(:movi_zero, [acc], [], 0..1, &[sub(acc, &1)])], s}

  def select({:vi8mac, acc, pa, pb}, s) do
    {t, s} = Machine.fresh(s, :fpr)
    v = {:phys, @v31}

    {Enum.flat_map(lanes(s), fn k ->
       [mi(:ldr_d, [t], [pa], o: [t, pa, 8 * k]),
        mi(:ldr_d, [], [pb], o: [v, pb, 8 * k]),
        mi(:sxtl_8h, [t], [t], o: [t, t]),
        mi(:sxtl_8h, [], [], o: [v, v]),
        mi(:smlal, [acc], [acc, t], o: [sub(acc, 0), t, v]),
        mi(:smlal2, [acc], [acc, t], o: [sub(acc, 1), t, v])]
     end), s}
  end

  def select({:vred_i32, x, acc}, s) do
    {t, s} = Machine.fresh(s, :fpr)

    {[mi(:add4s, [t], [acc], o: [t, sub(acc, 0), sub(acc, 1)]),
      mi(:addv_s, [t], [t], o: [t, t]),
      mi(:umov_w, [x], [t], o: [x, t])], s}
  end

  def select(:ret, s), do: {[mi(:ret, [], [], ret: true)], s}

  defp strip(n, ptrs, w, bytes, tail_bytes, vec_body, sca_body, s) do
    {lv, s} = Machine.label(s, :strip)
    {lt, s} = Machine.label(s)
    {le, s} = Machine.label(s)
    {vec, s} = Machine.select_all(vec_body, s)
    {sca, s} = Machine.select_all(sca_body, s)

    code =
      [mi(:label, [], [], label: lv), mi(:cmp_imm, [], [n], o: [n, w]), mi(:blt, [], [], br: lt)] ++
        vec ++
        Enum.map(ptrs, &mi(:addi, [&1], [&1], o: [&1, &1, bytes])) ++
        [mi(:addi, [n], [n], o: [n, n, -w]), mi(:b, [], [], br: lv, uncond: true),
         mi(:label, [], [], label: lt), mi(:cbz, [], [n], br: le, o: [n])] ++
        sca ++
        Enum.map(ptrs, &mi(:addi, [&1], [&1], o: [&1, &1, tail_bytes])) ++
        [mi(:addi, [n], [n], o: [n, n, -1]), mi(:b, [], [], br: lt, uncond: true),
         mi(:label, [], [], label: le)]

    {code, s}
  end

  defp lanes(s), do: 0..(s.g - 1)
  defp sub(v, k), do: {:sub, v, k}
  # One machine instruction per register of the group, issued as a single
  # allocation unit: lane k reads only lane k, so the destination may share
  # registers with a source that dies here — except a size-1 (broadcast)
  # source, which every lane reads: then the definition is an early clobber.
  defp lanewise(op, defs, uses, ks, ops) do
    shared = Enum.count(ks) > 1 and Enum.any?(uses, &match?({:vr, _, _, 1}, &1))
    mi(op, defs, uses, bundle: Enum.map(ks, ops), ec: shared)
  end

  defp uses({op, defs, _u, a}, u), do: {op, defs, u, a}

  # ---------------------------------------------- vector operations (VOPs) --
  # Operands: registers (a size-1 register is a broadcast read by every lane)
  # or `{:const, bits}` — loaded from the kernel's literal pool (LDR literal)
  # into a fresh register right before its use, so constants never hold
  # registers across a kernel.

  @binops %{vfadd: :fadd4s, vfsub: :fsub4s, vfmul: :fmul4s, viadd: :add4s, visub: :sub4s,
            viand: :and16b, vixor: :eor16b}

  defp vop(i, ks, s) do
    [op | rest] = Tuple.to_list(i)

    {loads, rest, s} =
      Enum.reduce(rest, {[], [], s}, fn
        {:const, bits}, {ld, acc, s} ->
          {t, s} = Machine.fresh(s, :fpr)
          {ld ++ [mi(:ldr_q_lit, [t], [], o: [t, {:pool, bits}])], acc ++ [t], s}

        o, {ld, acc, s} ->
          {ld, acc ++ [o], s}
      end)

    {loads ++ vop_(List.to_tuple([op | rest]), ks, s.policy), s}
  end

  defp vop_({op, d, a, b}, ks, _p) when is_map_key(@binops, op),
    do: [lanewise(Map.fetch!(@binops, op), [d], regs([a, b]), ks, &[sub(d, &1), opk(a, &1), opk(b, &1)])]

  defp vop_({:vshl, d, a, n}, ks, _p), do: [lanewise(:shl4s, [d], regs([a]), ks, &[sub(d, &1), opk(a, &1), n])]
  defp vop_({:vshr, d, a, n}, ks, _p), do: [lanewise(:ushr4s, [d], regs([a]), ks, &[sub(d, &1), opk(a, &1), n])]
  defp vop_({:vfneg, d, a}, ks, _p), do: [lanewise(:fneg4s, [d], regs([a]), ks, &[sub(d, &1), opk(a, &1)])]
  defp vop_({:vrelu, d, a}, ks, _p), do: [lanewise(:relu4s, [d], regs([a]), ks, &[sub(d, &1), opk(a, &1)])]

  defp vop_({:vfma, d, a, b, c}, ks, p) do
    op = if p == :fast, do: :fma4s_fused, else: :fma4s_split
    [lanewise(op, [d], regs([a, b, c]), ks, &[sub(d, &1), opk(a, &1), opk(b, &1), opk(c, &1)])]
  end

  defp vop_({:vfmacc, acc, a, b}, ks, p), do: vop_({:vfma, acc, a, b, acc}, ks, p)

  defp vop_({:vsel_lt, d, a, b, x, y}, ks, _p),
    do: [lanewise(:sel4s, [d], regs([a, b, x, y]), ks, &[sub(d, &1), opk(a, &1), opk(b, &1), opk(x, &1), opk(y, &1)])]

  defp regs(os), do: Enum.filter(os, &match?({:vr, _, _, _}, &1))
  defp opk({:vr, _, _, 1} = v, _k), do: sub(v, 0)
  defp opk(v, k), do: sub(v, k)
  defp lanes_of({:vr, _, _, size}), do: 0..(size - 1)

  @impl true
  def materialize(_key, _r), do: []

  @doc "Pool entry: 16 bytes (the pattern in every lane), 16-byte aligned."
  def pool_entry(bits), do: :binary.copy(<<bits::32-little>>, 4)
  def pool_align, do: 16
  def pad_byte, do: 0

  # ------------------------------------------------------------- frame --

  defp frame(used) do
    xs = Map.get(used, :x, [])
    ds = Map.get(used, :v, [])
    {xs, ds, (8 * (length(xs) + length(ds)) + 15) &&& -16}
  end

  @impl true
  def prologue(used) do
    {xs, ds, size} = frame(used)

    if size == 0,
      do: <<>>,
      else:
        [w(0xD1000000 ||| size <<< 10 ||| 31 <<< 5 ||| 31)] ++
          (Enum.with_index(xs) |> Enum.map(fn {r, i} -> w(0xF9000000 ||| i <<< 10 ||| 31 <<< 5 ||| r) end)) ++
          (Enum.with_index(ds)
           |> Enum.map(fn {r, i} -> w(0xFD000000 ||| (length(xs) + i) <<< 10 ||| 31 <<< 5 ||| r) end))
        |> IO.iodata_to_binary()
  end

  @impl true
  def epilogue(used) do
    {xs, ds, size} = frame(used)

    restore =
      if size == 0,
        do: [],
        else:
          (Enum.with_index(xs) |> Enum.map(fn {r, i} -> w(0xF9400000 ||| i <<< 10 ||| 31 <<< 5 ||| r) end)) ++
            (Enum.with_index(ds)
             |> Enum.map(fn {r, i} -> w(0xFD400000 ||| (length(xs) + i) <<< 10 ||| 31 <<< 5 ||| r) end)) ++
            [w(0x91000000 ||| size <<< 10 ||| 31 <<< 5 ||| 31)]

    IO.iodata_to_binary(restore ++ [w(0xD65F03C0)])
  end

  # ------------------------------------------------------------- encoding --

  @impl true
  def branch_size(_), do: 4

  @impl true
  def encode_branch(:b, _o, rel, _r), do: w(0x14000000 ||| (rel >>> 2 &&& 0x3FFFFFF))
  def encode_branch(:blt, _o, rel, _r), do: w(0x54000000 ||| (rel >>> 2 &&& 0x7FFFF) <<< 5 ||| 0b1011)
  def encode_branch(:blo, _o, rel, _r), do: w(0x54000000 ||| (rel >>> 2 &&& 0x7FFFF) <<< 5 ||| 0b0011)
  def encode_branch(:cbz, [x], rel, r), do: w(0xB4000000 ||| (rel >>> 2 &&& 0x7FFFF) <<< 5 ||| r.(x))
  def encode_branch(:cbnz, [x], rel, r), do: w(0xB5000000 ||| (rel >>> 2 &&& 0x7FFFF) <<< 5 ||| r.(x))

  @impl true
  def encode(:ldr_x, [d, b, off], r), do: ldst(0xF9400000, 8, r.(d), r.(b), off)
  def encode(:ldrb, [d, b, off], r), do: ldst(0x39400000, 1, r.(d), r.(b), off)
  def encode(:ldrsb, [d, b, off], r), do: ldst(0x39800000, 1, r.(d), r.(b), off)
  def encode(:str_w, [x, b, off], r), do: ldst(0xB9000000, 4, r.(x), r.(b), off)
  def encode(:ldr_s, [d, b, off], r), do: ldst(0xBD400000, 4, r.(d), r.(b), off)
  def encode(:str_s, [x, b, off], r), do: ldst(0xBD000000, 4, r.(x), r.(b), off)
  def encode(:ldr_d, [d, b, off], r), do: ldst(0xFD400000, 8, r.(d), r.(b), off)
  def encode(:ldr_w, [d, b, off], r), do: ldst(0xB9400000, 4, r.(d), r.(b), off)
  def encode(:str_x, [x, b, off], r), do: ldst(0xF9000000, 8, r.(x), r.(b), off)

  # LDR Qt, <literal>: imm19 patched once the pool is laid out
  def encode(:ldr_q_lit, [t, {:pool, bits}], r), do: {:fix, w(0x9C000000 ||| r.(t)), [{0, bits, :lit19}]}
  def encode(:shli, [d, a, n], r), do: w(0xD3400000 ||| (64 - n &&& 63) <<< 16 ||| (63 - n) <<< 10 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:shri, [d, a, n], r), do: w(0xD340FC00 ||| n <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:cmp_rr, [x, y], r), do: w(0xEB000000 ||| r.(y) <<< 16 ||| r.(x) <<< 5 ||| 31)
  def encode(:ldr_q, [d, b, off], r), do: ldst(0x3DC00000, 16, r.(d), r.(b), off)
  def encode(:str_q, [x, b, off], r), do: ldst(0x3D800000, 16, r.(x), r.(b), off)

  def encode(:mov, [d, s], r),
    do: if(r.(d) == r.(s), do: <<>>, else: w(0xAA0003E0 ||| r.(s) <<< 16 ||| r.(d)))

  def encode(:addi, [d, s, imm], r) do
    cond do
      imm == 0 and r.(d) == r.(s) -> <<>>
      imm >= 0 and imm < 4096 -> w(0x91000000 ||| imm <<< 10 ||| r.(s) <<< 5 ||| r.(d))
      imm < 0 and imm > -4096 -> w(0xD1000000 ||| -imm <<< 10 ||| r.(s) <<< 5 ||| r.(d))
      true -> encode(:li, [{:phys, @x16}, imm], r) <> w(0x8B000000 ||| @x16 <<< 16 ||| r.(s) <<< 5 ||| r.(d))
    end
  end

  def encode(:add, [d, a, b], r), do: w(0x8B000000 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:sub, [d, a, b], r), do: w(0xCB000000 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:mul, [d, a, b], r), do: w(0x9B007C00 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:cmp_imm, [x, imm], r), do: w(0xF1000000 ||| imm <<< 10 ||| r.(x) <<< 5 ||| 31)

  def encode(:li, [d, imm], r) do
    rd = r.(d)
    u = imm &&& 0xFFFF_FFFF_FFFF_FFFF
    chunks = for hw <- 0..3, do: {hw, u >>> (16 * hw) &&& 0xFFFF}
    [{hw0, c0} | rest] = Enum.filter(chunks, fn {hw, c} -> c != 0 or hw == 0 end)

    [w(0xD2800000 ||| hw0 <<< 21 ||| c0 <<< 5 ||| rd) |
       for({hw, c} <- rest, do: w(0xF2800000 ||| hw <<< 21 ||| c <<< 5 ||| rd))]
    |> IO.iodata_to_binary()
  end

  # scalar binary32
  def encode(:fadd, [d, a, b], r), do: w(0x1E202800 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:fsub, [d, a, b], r), do: w(0x1E203800 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:fmul, [d, a, b], r), do: w(0x1E200800 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))

  def encode(:fmadd, [d, a, b, c], r),
    do: w(0x1F000000 ||| r.(b) <<< 16 ||| r.(c) <<< 10 ||| r.(a) <<< 5 ||| r.(d))

  def encode(:fneg_s, [d, a], r), do: w(0x1E214000 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:ucvtf_s_w, [d, x], r), do: w(0x1E230000 ||| r.(x) <<< 5 ||| r.(d))
  def encode(:fmov_s_w, [d, x], r), do: w(0x1E270000 ||| r.(x) <<< 5 ||| r.(d))

  def encode(:fma_s_fused, [d, a, b, c], r), do: encode(:fmadd, [d, a, b, c], r)

  def encode(:fma_s_split, [d, a, b, c], r),
    do: encode(:fmul, [{:phys, @v31}, a, b], r) <> encode(:fadd, [d, {:phys, @v31}, c], r)

  def encode(:relu_s, [d, a], r) do
    w(0x5EA0C800 ||| r.(a) <<< 5 ||| @v31) <> w(0x0E201C00 ||| @v31 <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  end

  # Advanced SIMD, 4 × binary32
  def encode(:fadd4s, [d, a, b], r), do: w(0x4E20D400 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:fsub4s, [d, a, b], r), do: w(0x4EA0D400 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:fmul4s, [d, a, b], r), do: w(0x6E20DC00 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:fmla4s, [d, a, b], r), do: w(0x4E20CC00 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:fneg4s, [d, a], r), do: w(0x6EA0F800 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:fadd2s, [d, a, b], r), do: w(0x0E20D400 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:faddp_s, [d, a], r), do: w(0x7E30D800 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:ext8, [d, a, b], r), do: w(0x6E000000 ||| r.(b) <<< 16 ||| 8 <<< 11 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:dup4s, [d, f], r), do: w(0x4E040400 ||| r.(f) <<< 5 ||| r.(d))
  def encode(:fmul4s_elem, [d, a, f], r), do: w(0x4F809000 ||| r.(f) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:movi_zero, [d], r), do: w(0x6F00E400 ||| r.(d))
  def encode(:movi16b, [d, imm], r), do: w(0x4F00E400 ||| (imm >>> 5) <<< 16 ||| (imm &&& 31) <<< 5 ||| r.(d))
  def encode(:and16b, [d, a, b], r), do: w(0x4E201C00 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:ushr16b_4, [d, a], r), do: w(0x6F0C0400 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:uxtl_8h, [d, a], r), do: w(0x2F08A400 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:uxtl2_8h, [d, a], r), do: w(0x6F08A400 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:uxtl_4s, [d, a], r), do: w(0x2F10A400 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:shll_4s, [d, a], r), do: w(0x2E613800 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:shll2_4s, [d, a], r), do: w(0x6E613800 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:uxtl2_4s, [d, a], r), do: w(0x6F10A400 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:ucvtf4s, [d, a], r), do: w(0x6E21D800 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:sxtl_8h, [d, a], r), do: w(0x0F08A400 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:smlal, [d, a, b], r), do: w(0x0E608000 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:smlal2, [d, a, b], r), do: w(0x4E608000 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:add4s, [d, a, b], r), do: w(0x4EA08400 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:addv_s, [d, a], r), do: w(0x4EB1B800 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:umov_w, [x, v], r), do: w(0x0E043C00 ||| r.(v) <<< 5 ||| r.(x))
  def encode(:sub4s, [d, a, b], r), do: w(0x6EA08400 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:eor16b, [d, a, b], r), do: w(0x6E201C00 ||| r.(b) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:mov16b, [d, a], r), do: if(r.(d) == r.(a), do: <<>>, else: w(0x4EA01C00 ||| r.(a) <<< 16 ||| r.(a) <<< 5 ||| r.(d)))
  def encode(:shl4s, [d, a, n], r), do: w(0x4F005400 ||| (32 + n) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:ushr4s, [d, a, n], r), do: w(0x6F000400 ||| (64 - n) <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  def encode(:dup4s_w, [d, x], r), do: w(0x4E040C00 ||| r.(x) <<< 5 ||| r.(d))
  def encode(:dup_s1, [d, a], r), do: w(0x4E0C0400 ||| r.(a) <<< 5 ||| r.(d))

  # d = (a < b) ? x : y — fcmgt v31, b, a; bsl v31, x, y; mov d, v31
  def encode(:sel4s, [d, a, b, x, y], r) do
    w(0x6EA0E400 ||| r.(a) <<< 16 ||| r.(b) <<< 5 ||| @v31) <>
      w(0x6E601C00 ||| r.(y) <<< 16 ||| r.(x) <<< 5 ||| @v31) <>
      w(0x4EA01C00 ||| @v31 <<< 16 ||| @v31 <<< 5 ||| r.(d))
  end

  def encode(:relu4s, [d, a], r) do
    w(0x4EA0C800 ||| r.(a) <<< 5 ||| @v31) <> w(0x4E201C00 ||| @v31 <<< 16 ||| r.(a) <<< 5 ||| r.(d))
  end

  def encode(:fma4s_fused, [d, a, b, c], r) do
    mov = if r.(d) == r.(c), do: <<>>, else: w(0x4EA01C00 ||| r.(c) <<< 16 ||| r.(c) <<< 5 ||| r.(d))
    if r.(d) != r.(c) and (r.(d) == r.(a) or r.(d) == r.(b)) do
      # d aliases a factor: form the product-sum in scratch first
      w(0x4EA01C00 ||| r.(c) <<< 16 ||| r.(c) <<< 5 ||| @v31) <>
        encode(:fmla4s, [{:phys, @v31}, a, b], r) <>
        w(0x4EA01C00 ||| @v31 <<< 16 ||| @v31 <<< 5 ||| r.(d))
    else
      mov <> encode(:fmla4s, [d, a, b], r)
    end
  end

  def encode(:fma4s_split, [d, a, b, c], r),
    do: encode(:fmul4s, [{:phys, @v31}, a, b], r) <> encode(:fadd4s, [d, {:phys, @v31}, c], r)

  def encode(:fmla_mem, [acc, wv, b, off], r),
    do: encode(:ldr_q, [{:phys, @v31}, b, off], r) <> encode(:fmla4s, [acc, wv, {:phys, @v31}], r)

  def encode(:fmacc_mem_split, [acc, wv, b, off], r) do
    encode(:ldr_q, [{:phys, @v31}, b, off], r) <>
      encode(:fmul4s, [{:phys, @v31}, wv, {:phys, @v31}], r) <>
      encode(:fadd4s, [acc, acc, {:phys, @v31}], r)
  end

  defp ldst(base, scale, rt, rn, off) do
    unless rem(off, scale) == 0 and off >= 0 and div(off, scale) < 4096,
      do: raise(ArgumentError, "unencodable offset #{off} (scale #{scale})")

    w(base ||| div(off, scale) <<< 10 ||| rn <<< 5 ||| rt)
  end

  defp w(x), do: <<x::32-little>>
end
