defmodule Vapor.Emit.X86 do
  @moduledoc """
  x86-64 AVX2/FMA backend — pure binary synthesis (REX, VEX, ModR/M, SIB,
  opcodes written as bits; no assembler anywhere).

  SysV AMD64 ABI: the kernel receives `const uint64_t *args` in `rdi`;
  callee-saved `rbx rbp r12–r15` are saved/restored only if the allocator
  used them; `vzeroupper` precedes `ret` (no AVX→SSE transition penalty for
  the caller). `r11` and `ymm15` are backend scratch and never allocated.

  Register groups: a `:strip` value of group factor `g` is `g` consecutive
  ymm registers (8·g lanes), `:f16l` is 2 ymm, `:i32acc` is `g` ymm; a
  size-1 register used as a vector operand is read by every lane (a
  broadcast scalar).

  Constants never occupy vector registers: a `{:const, bits}` operand is a
  RIP-relative 32-byte memory operand into a per-kernel pool appended after
  the code (x86 has 15 allocatable ymm registers — a canonical `exp` alone
  uses 15 distinct constants). Where an encoding cannot take memory in that
  position, the constant is loaded into a fresh allocated register.
  """
  import Bitwise
  alias Vapor.Emit.Machine

  @behaviour Machine

  @rdi 7
  @r11 11
  @ymm15 15

  # ------------------------------------------------------------ reg model --

  @impl true
  def isa, do: :x86_64

  @impl true
  def files do
    %{
      g: %{count: 16, order: [0, 1, 2, 6, 7, 8, 9, 10, 3, 5, 12, 13, 14, 15],
           callee_saved: [3, 5, 12, 13, 14, 15]},
      v: %{count: 16, order: Enum.to_list(0..14), callee_saved: []}
    }
  end

  @impl true
  def group_factors, do: [4, 2, 1]

  @impl true
  def size(:gpr, _g), do: {:g, 1}
  def size(:fpr, _g), do: {:v, 1}
  def size(:strip, g), do: {:v, g}
  def size(:f16l, _g), do: {:v, 2}
  def size(:i32acc, g), do: {:v, g}

  @impl true
  def arg_pin, do: {:g, @rdi}

  # ------------------------------------------------------------ selection --
  # Every selector returns a list of machine instructions
  # {op, defs, uses, attrs}; attrs.o holds the encoding operands.

  import Machine, only: [mi: 4]

  @vops [:vfadd, :vfsub, :vfmul, :vfma, :vfneg, :vrelu, :vsel_lt, :viadd, :visub, :viand, :vixor,
         :vshl, :vshr, :vfmacc]

  @impl true
  def select({:arg, x, i}, s), do: {[mi(:mov_load, [x], [s.argp], o: [x, {:mem, s.argp, 8 * i}])], s}
  def select({:li, x, imm}, s), do: {[mi(:mov_imm, [x], [], o: [x, imm])], s}
  def select({:mov, x, y}, s), do: {[mi(:mov_rr, [x], [y], o: [x, y])], s}
  def select({:addi, x, y, imm}, s), do: {[mi(:lea, [x], [y], o: [x, {:mem, y, imm}])], s}

  def select({op, x, y, z}, s) when op in [:add, :sub, :mul] do
    if x == z and x != y do
      {t, s} = Machine.fresh(s, :gpr)
      {[mi(:mov_rr, [t], [y], o: [t, y]), mi(op, [t], [t, z], o: [t, z]),
        mi(:mov_rr, [x], [t], o: [x, t])], s}
    else
      {[mi(:mov_rr, [x], [y], o: [x, y]), mi(op, [x], [x, z], o: [x, z])], s}
    end
  end

  def select({:label, l}, s), do: {[mi(:label, [], [], label: l)], s}
  def select({:jmp, l}, s), do: {[mi(:jmp, [], [], br: l, uncond: true)], s}
  def select({:bnez, x, l}, s), do: {[mi(:test, [], [x], o: [x]), mi(:jnz, [], [], br: l)], s}
  def select({:beqz, x, l}, s), do: {[mi(:test, [], [x], o: [x]), mi(:jz, [], [], br: l)], s}
  def select({:blt_imm, x, imm, l}, s), do: {[mi(:cmp_imm, [], [x], o: [x, imm]), mi(:jl, [], [], br: l)], s}
  def select({:bltu, x, y, l}, s), do: {[mi(:cmp_rr, [], [x, y], o: [x, y]), mi(:jb, [], [], br: l)], s}
  def select({:ld_u8, x, b, off}, s), do: {[mi(:movzx8, [x], [b], o: [x, {:mem, b, off}])], s}
  def select({:ld_s8, x, b, off}, s), do: {[mi(:movsx8, [x], [b], o: [x, {:mem, b, off}])], s}
  def select({:ld_u32, x, b, off}, s), do: {[mi(:load32, [x], [b], o: [x, {:mem, b, off}])], s}
  def select({:ld64, x, b, off}, s), do: {[mi(:mov_load, [x], [b], o: [x, {:mem, b, off}])], s}
  def select({:st64, x, b, off}, s), do: {[mi(:store64, [], [x, b], o: [x, {:mem, b, off}])], s}

  def select({op, x, y, n}, s) when op in [:shli, :shri],
    do: {[mi(:mov_rr, [x], [y], o: [x, y]), mi(op, [x], [x], o: [x, n])], s}
  def select({:st_i32, x, b, off}, s), do: {[mi(:store32, [], [x, b], o: [x, {:mem, b, off}])], s}

  # scalar binary32 (lane 0 of an xmm)
  def select({:ldf, f, b, off}, s), do: {[mi(:vmovss_load, [f], [b], o: [f, {:mem, b, off}])], s}
  def select({:stf, f, b, off}, s), do: {[mi(:vmovss_store, [], [f, b], o: [f, {:mem, b, off}])], s}

  def select({:lif, f, bits}, s) do
    {[mi(:mov32_imm, [], [], o: [{:phys, @r11}, bits]),
      mi(:vmovd_x_r, [f], [], o: [f, {:phys, @r11}])], s}
  end

  # a scalar made usable as a vector operand: broadcast to every lane
  def select({:bcast, f}, s), do: {[mi(:vbroadcastss, [f], [f], o: [f, f])], s}

  # `vcvtsi2ss` merges the destination's upper bits from its first source; a
  # zero idiom on `f` first breaks that false dependency (no scratch chains)
  def select({:cvt_u8f, f, x}, s),
    do: {[mi(:vxorps128, [f], [], o: [f, f, f]), mi(:vcvtsi2ss, [f], [x, f], o: [f, f, x])], s}

  def select({op, f, a, b}, s) when op in [:fadd, :fsub, :fmul] do
    vop = %{fadd: :vaddss, fsub: :vsubss, fmul: :vmulss}[op]
    {[mi(vop, [f], [a, b], o: [f, a, b])], s}
  end

  def select({:fmacc, acc, a, b}, %{policy: :fast} = s),
    do: {[mi(:vfmadd231ss, [acc], [acc, a, b], o: [acc, a, b])], s}

  def select({:fmacc, acc, a, b}, s) do
    {[mi(:vmulss, [], [a, b], o: [{:phys, @ymm15}, a, b]),
      mi(:vaddss, [acc], [acc], o: [acc, acc, {:phys, @ymm15}])], s}
  end

  # ---- strip kinds (g ymm per value) ----
  def select({:strip, n, ptrs, body}, s) do
    w = 8 * s.g
    {lv, s} = Machine.label(s, :strip)
    {lt, s} = Machine.label(s)
    {le, s} = Machine.label(s)
    {vec, s} = Machine.select_all(Enum.map(body, &{:lanes, &1}), s)
    {tail, s} = Machine.tail_copy(body, s)
    {sca, s} = Machine.select_all(Enum.map(tail, &{:scalar, &1}), s)

    code =
      [mi(:label, [], [], label: lv),
       mi(:cmp_imm, [], [n], o: [n, w]),
       mi(:jl, [], [], br: lt)] ++
        vec ++
        Enum.map(ptrs, &mi(:lea, [&1], [&1], o: [&1, {:mem, &1, 4 * w}])) ++
        [mi(:lea, [n], [n], o: [n, {:mem, n, -w}]),
         mi(:jmp, [], [], br: lv, uncond: true),
         mi(:label, [], [], label: lt),
         mi(:test, [], [n], o: [n]),
         mi(:jz, [], [], br: le)] ++
        sca ++
        Enum.map(ptrs, &mi(:lea, [&1], [&1], o: [&1, {:mem, &1, 4}])) ++
        [mi(:lea, [n], [n], o: [n, {:mem, n, -1}]),
         mi(:jmp, [], [], br: lt, uncond: true),
         mi(:label, [], [], label: le)]

    {code, s}
  end

  def select({:lanes, {:vld, v, b, off}}, s),
    do: {[lanewise(:vmovups_load, [v], [b], lanes(s), &[sub(v, &1), {:mem, b, off + 32 * &1}])], s}

  def select({:lanes, {:vst, v, b, off}}, s),
    do: {[lanewise(:vmovups_store, [], [v, b], lanes(s), &[sub(v, &1), {:mem, b, off + 32 * &1}])], s}

  def select({:lanes, {op, d, _, _} = i}, s) when op in @vops, do: vop(i, 1, lanes_of(d), s)
  def select({:lanes, {op, d, _} = i}, s) when op in @vops, do: vop(i, 1, lanes_of(d), s)
  def select({:lanes, {op, d, _, _, _} = i}, s) when op in @vops, do: vop(i, 1, lanes_of(d), s)
  def select({:lanes, {op, d, _, _, _, _} = i}, s) when op in @vops, do: vop(i, 1, lanes_of(d), s)

  # scalar tail: lane 0 only. Loads/stores move exactly 4 bytes; arithmetic
  # uses the 128-bit forms (the other lanes hold zeros or broadcasts, and
  # nothing ever reads them back)
  def select({:scalar, {:vld, v, b, off}}, s), do: {[mi(:vmovss_load, [v], [b], o: [sub(v, 0), {:mem, b, off}])], s}
  def select({:scalar, {:vst, v, b, off}}, s), do: {[mi(:vmovss_store, [], [v, b], o: [sub(v, 0), {:mem, b, off}])], s}
  def select({:scalar, i}, s) when elem(i, 0) in @vops, do: vop(i, 0, [0], s)

  # bare vector ops (f16l values outside strips): one per register of the group
  def select({:vld, v, b, off}, s),
    do: {[lanewise(:vmovups_load, [v], [b], lanes_of(v), &[sub(v, &1), {:mem, b, off + 32 * &1}])], s}

  def select({:vst, v, b, off}, s),
    do: {[lanewise(:vmovups_store, [], [v, b], lanes_of(v), &[sub(v, &1), {:mem, b, off + 32 * &1}])], s}

  def select(i, s) when is_tuple(i) and elem(i, 0) in @vops, do: vop(i, 1, lanes_of(elem(i, 1)), s)

  # lane 0 of a uniform vector as a broadcast-ready scalar register
  def select({:vlane0, f, v}, s), do: {[mi(:vmovaps, [f], [v], o: [f, sub(v, 0)])], s}

  def select({:vconst, v, bits}, s),
    do: {[lanewise(:vmovups_load, [v], [], lanes_of(v), &[sub(v, &1), {:pool, bits}])], s}

  def select({:vsplat, v, f}, s),
    do: {[lanewise(:vbroadcastss, [v], [f], lanes_of(v), &[sub(v, &1), f])], s}

  # ---- f16l: 16 binary32 lanes = 2 ymm ----
  def select({:vzero16, v}, s), do: {[lanewise(:vxorps, [v], [], 0..1, &[sub(v, &1), sub(v, &1), sub(v, &1)])], s}

  def select({:vld16, v, b, off}, s),
    do: {[lanewise(:vmovups_load, [v], [b], 0..1, &[sub(v, &1), {:mem, b, off + 32 * &1}])], s}

  def select({:vfadd16, d, a, b}, s),
    do: {[lanewise(:vaddps, [d], [a, b], 0..1, &[sub(d, &1), sub(a, &1), sub(b, &1)])], s}

  # 16 bfloat16 → 16 binary32: zero-extend each half-word, shift into the high half
  def select({:vld_bf16, v, b, off}, s) do
    {[lanewise(:vpmovzxwd, [v], [b], 0..1, &[sub(v, &1), {:mem, b, off + 16 * &1}]),
      lanewise({:vshift, 6, 1}, [v], [v], 0..1, &[sub(v, &1), sub(v, &1), 16])], s}
  end

  def select({:vld_nib, lo, hi, b, off}, s) do
    {Enum.flat_map(0..1, fn k ->
       [mi(:vpmovzxbd, [], [b], o: [{:phys, @ymm15}, {:mem, b, off + 8 * k}]),
        mi(:vpand, [lo], [], o: [sub(lo, k), {:phys, @ymm15}, {:pool, 0x0F}]),
        mi(:vpsrld, [hi], [], o: [sub(hi, k), {:phys, @ymm15}, 4]),
        mi(:vcvtdq2ps, [lo], [lo], o: [sub(lo, k), sub(lo, k)]),
        mi(:vcvtdq2ps, [hi], [hi], o: [sub(hi, k), sub(hi, k)])]
     end), s}
  end

  def select({:vmul_sf, d, a, f}, s) do
    {[mi(:vbroadcastss, [], [f], o: [{:phys, @ymm15}, f]),
      lanewise(:vmulps, [d], [a], 0..1, &[sub(d, &1), sub(a, &1), {:phys, @ymm15}])], s}
  end

  def select({:vfmacc_mem, acc, w, b, off}, %{policy: :fast} = s) do
    {[lanewise(:vfmadd231ps, [acc], [acc, w, b], 0..1,
               &[sub(acc, &1), sub(w, &1), {:mem, b, off + 32 * &1}])], s}
  end

  def select({:vfmacc_mem, acc, w, b, off}, s) do
    {Enum.flat_map(0..1, fn k ->
       [mi(:vmulps, [], [w, b], o: [{:phys, @ymm15}, sub(w, k), {:mem, b, off + 32 * k}]),
        mi(:vaddps, [acc], [acc], o: [sub(acc, k), sub(acc, k), {:phys, @ymm15}])]
     end), s}
  end

  def select({:vreduce16, f, v}, s) do
    x = {:phys, @ymm15}

    {[mi(:vaddps, [f], [v], o: [f, sub(v, 0), sub(v, 1)]),
      mi(:vextractf128, [], [f], o: [x, f, 1]),
      mi(:vaddps128, [f], [f], o: [f, f, x]),
      mi(:vmovhlps, [], [f], o: [x, x, f]),
      mi(:vaddps128, [f], [f], o: [f, f, x]),
      mi(:vmovshdup, [], [f], o: [x, f]),
      mi(:vaddss, [f], [f], o: [f, f, x])], s}
  end

  # the same tree with max(a, b) = (a < b) ? b : a at every node
  def select({:vreduce16_max, f, v}, s) do
    x = {:phys, @ymm15}
    {m, s} = Machine.fresh(s, :fpr)
    sel = fn l -> [mi({:vcmpps, l}, [m], [f], o: [m, f, x, 0x11]), mi({:vblendvps, l}, [f], [f, m], o: [f, f, x, m])] end

    {[mi({:vcmpps, 1}, [m], [v], o: [m, sub(v, 0), sub(v, 1), 0x11]),
      mi({:vblendvps, 1}, [f], [v, m], o: [f, sub(v, 0), sub(v, 1), m]),
      mi(:vextractf128, [], [f], o: [x, f, 1])] ++ sel.(0) ++
       [mi(:vmovhlps, [], [f], o: [x, x, f])] ++ sel.(0) ++
       [mi(:vmovshdup, [], [f], o: [x, f])] ++ sel.(0), s}
  end

  # ---- i8 strips ----
  def select({:vzero_i32, acc}, s),
    do: {[lanewise(:vpxor, [acc], [], lanes(s), &[sub(acc, &1), sub(acc, &1), sub(acc, &1)])], s}

  def select({:strip_i8, n, ptrs, body, tail}, s) do
    w = 8 * s.g
    {lv, s} = Machine.label(s, :strip)
    {lt, s} = Machine.label(s)
    {le, s} = Machine.label(s)
    {vec, s} = Machine.select_all(body, s)
    {sca, s} = Machine.select_all(tail, s)

    code =
      [mi(:label, [], [], label: lv), mi(:cmp_imm, [], [n], o: [n, w]), mi(:jl, [], [], br: lt)] ++
        vec ++
        Enum.map(ptrs, &mi(:lea, [&1], [&1], o: [&1, {:mem, &1, w}])) ++
        [mi(:lea, [n], [n], o: [n, {:mem, n, -w}]), mi(:jmp, [], [], br: lv, uncond: true),
         mi(:label, [], [], label: lt), mi(:test, [], [n], o: [n]), mi(:jz, [], [], br: le)] ++
        sca ++
        Enum.map(ptrs, &mi(:lea, [&1], [&1], o: [&1, {:mem, &1, 1}])) ++
        [mi(:lea, [n], [n], o: [n, {:mem, n, -1}]), mi(:jmp, [], [], br: lt, uncond: true),
         mi(:label, [], [], label: le)]

    {code, s}
  end

  def select({:vi8mac, acc, pa, pb}, s) do
    {t, s} = Machine.fresh(s, :fpr)

    {Enum.flat_map(lanes(s), fn k ->
       [mi(:vpmovsxbd, [], [pa], o: [{:phys, @ymm15}, {:mem, pa, 8 * k}]),
        mi(:vpmovsxbd, [t], [pb], o: [t, {:mem, pb, 8 * k}]),
        mi(:vpmulld, [t], [t], o: [t, t, {:phys, @ymm15}]),
        mi(:vpaddd, [acc], [acc, t], o: [sub(acc, k), sub(acc, k), t])]
     end), s}
  end

  def select({:vred_i32, x, acc}, s) do
    {t, s} = Machine.fresh(s, :fpr)
    y = {:phys, @ymm15}

    sum =
      case s.g do
        1 -> [mi(:vmovaps, [t], [acc], o: [t, sub(acc, 0)])]
        g -> [mi(:vpaddd, [t], [acc], o: [t, sub(acc, 0), sub(acc, 1)])] ++
               for(k <- 2..(g - 1)//1, do: mi(:vpaddd, [t], [t, acc], o: [t, t, sub(acc, k)]))
      end

    {sum ++
       [mi(:vextracti128, [], [t], o: [y, t, 1]),
        mi(:vpaddd128, [t], [t], o: [t, t, y]),
        mi(:vpshufd, [], [t], o: [y, t, 0x4E]),
        mi(:vpaddd128, [t], [t], o: [t, t, y]),
        mi(:vpshufd, [], [t], o: [y, t, 0xB1]),
        mi(:vpaddd128, [t], [t], o: [t, t, y]),
        mi(:vmovd_r_x, [x], [t], o: [x, t])], s}
  end

  def select(:ret, s), do: {[mi(:ret, [], [], ret: true)], s}

  # ---------------------------------------------- vector operations (VOPs) --
  # `l` is the VEX.L bit (1: ymm, 0: xmm for scalar tails); `ks` the group
  # registers. Operands: registers (a size-1 register is a broadcast read by
  # every lane) or `{:const, bits}` (a pool memory operand).

  @binops %{vfadd: {0x58, 1, 0, true}, vfsub: {0x5C, 1, 0, false}, vfmul: {0x59, 1, 0, true},
            viadd: {0xFE, 1, 1, true}, visub: {0xFA, 1, 1, false}, viand: {0xDB, 1, 1, true},
            vixor: {0xEF, 1, 1, true}}

  defp vop({op, d, a, b}, l, ks, s) when is_map_key(@binops, op) do
    {opc, map, pp, comm} = Map.fetch!(@binops, op)
    {a, b} = if const?(a) and comm and not const?(b), do: {b, a}, else: {a, b}
    {pre, a, s} = in_reg(a, s)
    {pre ++ [lanewise({:vbin, opc, map, pp, l}, [d], regs([a, b]), ks, &[sub(d, &1), opk(a, &1), opk(b, &1)])], s}
  end

  defp vop({op, d, a, n}, l, ks, s) when op in [:vshl, :vshr] do
    ext = if op == :vshl, do: 6, else: 2
    {pre, a, s} = in_reg(a, s)
    {pre ++ [lanewise({:vshift, ext, l}, [d], regs([a]), ks, &[sub(d, &1), opk(a, &1), n])], s}
  end

  defp vop({:vfneg, d, a}, l, ks, s), do: vop({:vixor, d, a, {:const, 0x8000_0000}}, l, ks, s)

  # relu(x) = x > 0 ? x : +0  ⇔  maxps(x, +0) (second operand on NaN and on ±0 ties)
  defp vop({:vrelu, d, a}, l, ks, s) do
    {pre, a, s} = in_reg(a, s)
    {pre ++ [lanewise({:vbin, 0x5F, 1, 0, l}, [d], regs([a]), ks, &[sub(d, &1), opk(a, &1), {:pool, 0}])], s}
  end

  defp vop({:vfma, d, a, b, c}, l, ks, %{policy: :fast} = s) do
    {pa, a, s} = in_reg(a, s)
    {pc, c, s} = in_reg(c, s)
    {pa ++ pc ++ Enum.map(ks, &mi({:vfma, l}, [d], regs([a, b, c]), o: [sub(d, &1), opk(a, &1), opk(b, &1), opk(c, &1)])), s}
  end

  defp vop({:vfma, d, a, b, c}, l, ks, s) do
    {pa, a, s} = in_reg(a, s)

    {pa ++ Enum.flat_map(ks, fn k ->
       [mi({:vbin, 0x59, 1, 0, l}, [], regs([a, b]), o: [{:phys, @ymm15}, opk(a, k), opk(b, k)]),
        mi({:vbin, 0x58, 1, 0, l}, [d], regs([c]), o: [sub(d, k), {:phys, @ymm15}, opk(c, k)])]
     end), s}
  end

  defp vop({:vfmacc, acc, a, b}, l, ks, s), do: vop({:vfma, acc, a, b, acc}, l, ks, s)

  # d = (a < b) ? x : y — vcmpps (LT_OQ) into ymm15, then vblendvps picks
  # its r/m operand where the mask is set. A constant `a` swaps the compare
  # (GT_OQ); a constant `y` inverts it (NLT_UQ, equal to ≥ on the finite
  # values the canonical programs produce) and swaps the blend operands.
  defp vop({:vsel_lt, d, a, b, x, y}, l, ks, s) do
    {ca, cb, pred} = if const?(a), do: {b, a, 0x1E}, else: {a, b, 0x11}
    {pre1, ca, s} = in_reg(ca, s)

    {src1, src2, pred} =
      if const?(y) and not const?(x), do: {x, y, bxor(pred, 0x04)}, else: {y, x, pred}

    {pre2, src1, s} = in_reg(src1, s)
    m = {:phys, @ymm15}

    {pre1 ++ pre2 ++
       Enum.flat_map(ks, fn k ->
         [mi({:vcmpps, l}, [], regs([ca, cb]), o: [m, opk(ca, k), opk(cb, k), pred]),
          mi({:vblendvps, l}, [d], regs([src1, src2]), o: [sub(d, k), opk(src1, k), opk(src2, k), m])]
       end), s}
  end

  defp const?({:const, _}), do: true
  defp const?(_), do: false

  # a constant in a position that needs a register: load it into a fresh one
  defp in_reg({:const, bits}, s) do
    {t, s} = Machine.fresh(s, :fpr)
    {[mi(:vmovups_load, [t], [], o: [t, {:pool, bits}])], t, s}
  end

  defp in_reg(o, s), do: {[], o, s}

  defp regs(os), do: Enum.filter(os, &match?({:vr, _, _, _}, &1))

  defp opk({:const, bits}, _k), do: {:pool, bits}
  defp opk({:vr, _, _, 1} = v, _k), do: sub(v, 0)
  defp opk(v, k), do: sub(v, k)

  defp lanes(s), do: 0..(s.g - 1)
  defp lanes_of({:vr, _, _, size}), do: 0..(size - 1)

  # One machine instruction per register of the group, issued as a single
  # allocation unit: lane k reads only lane k, so the destination may share
  # registers with a source that dies here — except a size-1 (broadcast)
  # source, which every lane reads: then the definition is an early clobber.
  defp lanewise(op, defs, uses, ks, ops) do
    shared = Enum.count(ks) > 1 and Enum.any?(uses, &match?({:vr, _, _, 1}, &1))
    mi(op, defs, uses, bundle: Enum.map(ks, ops), ec: shared)
  end
  defp sub(v, k), do: {:sub, v, k}

  @impl true
  def materialize(_key, _r), do: []

  # ------------------------------------------------------------- frame --

  @impl true
  def prologue(%{g: saved}) do
    Enum.map(saved, &push/1) |> IO.iodata_to_binary()
  end

  def prologue(_), do: <<>>

  @impl true
  def epilogue(%{g: saved}) do
    [Enum.reverse(saved) |> Enum.map(&pop/1), <<0xC5, 0xF8, 0x77>>, <<0xC3>>]
    |> IO.iodata_to_binary()
  end

  def epilogue(_), do: <<0xC5, 0xF8, 0x77, 0xC3>>

  defp push(r) when r < 8, do: <<0x50 + r>>
  defp push(r), do: <<0x41, 0x50 + (r - 8)>>
  defp pop(r) when r < 8, do: <<0x58 + r>>
  defp pop(r), do: <<0x41, 0x58 + (r - 8)>>

  # ------------------------------------------------------- constant pool --

  @doc "Pool entry for a constant: 32 bytes (the pattern in every lane), 32-byte aligned."
  def pool_entry(bits), do: :binary.copy(<<bits::32-little>>, 8)
  def pool_align, do: 32
  def pad_byte, do: 0xCC

  # ------------------------------------------------------------- encoding --
  # `r` resolves an operand to a physical register number. Encoders return a
  # binary, or `{:fix, binary, [{offset, bits}]}` when the instruction holds
  # RIP-relative displacements into the pool (patched by `Vapor.Emit.Machine`).

  @impl true
  def branch_size(op) when op in [:jz, :jnz, :jl, :jb], do: 6
  def branch_size(:jmp), do: 5

  @impl true
  def encode_branch(op, _o, rel, _r) do
    case op do
      :jmp -> <<0xE9, rel - 5::signed-32-little>>
      :jz -> <<0x0F, 0x84, rel - 6::signed-32-little>>
      :jnz -> <<0x0F, 0x85, rel - 6::signed-32-little>>
      :jl -> <<0x0F, 0x8C, rel - 6::signed-32-little>>
      :jb -> <<0x0F, 0x82, rel - 6::signed-32-little>>
    end
  end

  @impl true
  def encode(op, o, r), do: op |> enc(o, r) |> resolve()

  # {:rip, bits} leaves a marker `{:rip, bits}` inside iodata; flatten into
  # a binary plus the byte offsets of the 32-bit displacements to patch
  defp resolve(bin) when is_binary(bin), do: bin

  defp resolve(io) do
    {bin, fixes} =
      io
      |> List.flatten()
      |> Enum.reduce({<<>>, []}, fn
        {:rip, bits}, {acc, fx} -> {acc <> <<0::32>>, [{byte_size(acc), bits} | fx]}
        b, {acc, fx} -> {acc <> b, fx}
      end)

    if fixes == [], do: bin, else: {:fix, bin, Enum.reverse(fixes)}
  end

  defp enc(:mov_load, [d, m], r), do: rex_rm(1, 0x8B, r.(d), mem(m, r))
  defp enc(:mov_rr, [d, s], r), do: if(r.(d) == r.(s), do: <<>>, else: rex_rm(1, 0x89, r.(s), {:r, r.(d)}))

  defp enc(:mov_imm, [d, imm], r) do
    rd = r.(d)

    if imm >= -0x8000_0000 and imm < 0x8000_0000,
      do: rex_rm(1, 0xC7, 0, {:r, rd}) <> <<imm::signed-32-little>>,
      else: <<rex(1, 0, 0, rd >>> 3), 0xB8 + (rd &&& 7), imm::64-little>>
  end

  defp enc(:mov32_imm, [d, imm], r) do
    rd = r.(d)
    pre = if rd >= 8, do: <<0x41>>, else: <<>>
    pre <> <<0xB8 + (rd &&& 7), imm::32-little>>
  end

  defp enc(:lea, [d, m], r), do: rex_rm(1, 0x8D, r.(d), mem(m, r))
  defp enc(:add, [d, s], r), do: rex_rm(1, 0x01, r.(s), {:r, r.(d)})
  defp enc(:sub, [d, s], r), do: rex_rm(1, 0x29, r.(s), {:r, r.(d)})
  defp enc(:mul, [d, s], r), do: rex_rm(1, <<0x0F, 0xAF>>, r.(d), {:r, r.(s)})
  defp enc(:test, [x], r), do: rex_rm(1, 0x85, r.(x), {:r, r.(x)})
  defp enc(:cmp_imm, [x, imm], r), do: rex_rm(1, 0x81, 7, {:r, r.(x)}) <> <<imm::signed-32-little>>
  defp enc(:cmp_rr, [x, y], r), do: rex_rm(1, 0x39, r.(y), {:r, r.(x)})
  defp enc(:movzx8, [d, m], r), do: rex_rm(0, <<0x0F, 0xB6>>, r.(d), mem(m, r))
  defp enc(:movsx8, [d, m], r), do: rex_rm(1, <<0x0F, 0xBE>>, r.(d), mem(m, r))
  defp enc(:load32, [d, m], r), do: rex_rm(0, 0x8B, r.(d), mem(m, r))
  defp enc(:store32, [s, m], r), do: rex_rm(0, 0x89, r.(s), mem(m, r))
  defp enc(:store64, [s, m], r), do: rex_rm(1, 0x89, r.(s), mem(m, r))
  defp enc(:shli, [x, n], r), do: rex_rm(1, 0xC1, 4, {:r, r.(x)}) <> <<n>>
  defp enc(:shri, [x, n], r), do: rex_rm(1, 0xC1, 5, {:r, r.(x)}) <> <<n>>

  # VEX forms: vex(opcode, map, pp, L, W, reg, vvvv, rm)
  defp enc(:vmovups_load, [d, m], r), do: vex(0x10, 1, 0, 1, 0, r.(d), 0, rm(m, r))
  defp enc(:vmovups_store, [s, m], r), do: vex(0x11, 1, 0, 1, 0, r.(s), 0, mem(m, r))
  defp enc(:vmovss_load, [d, m], r), do: vex(0x10, 1, 2, 0, 0, r.(d), 0, rm(m, r))
  defp enc(:vmovss_store, [s, m], r), do: vex(0x11, 1, 2, 0, 0, r.(s), 0, mem(m, r))
  defp enc(:vmovaps, [d, s], r), do: if(r.(d) == r.(s), do: <<>>, else: vex(0x28, 1, 0, 1, 0, r.(d), 0, {:r, r.(s)}))

  for {name, opc, pp, l} <- [
        {:vaddps, 0x58, 0, 1}, {:vsubps, 0x5C, 0, 1}, {:vmulps, 0x59, 0, 1},
        {:vmaxps, 0x5F, 0, 1}, {:vxorps, 0x57, 0, 1}, {:vaddps128, 0x58, 0, 0},
        {:vxorps128, 0x57, 0, 0},
        {:vaddss, 0x58, 2, 0}, {:vsubss, 0x5C, 2, 0}, {:vmulss, 0x59, 2, 0},
        {:vmaxss, 0x5F, 2, 0}, {:vmovhlps, 0x12, 0, 0},
        {:vpand, 0xDB, 1, 1}, {:vpxor, 0xEF, 1, 1}, {:vpaddd, 0xFE, 1, 1},
        {:vpaddd128, 0xFE, 1, 0}
      ] do
    defp enc(unquote(name), [d, a, b], r),
      do: vex(unquote(opc), 1, unquote(pp), unquote(l), 0, r.(d), r.(a), rm(b, r))
  end

  # generic two-source VEX op (map 0F): d = a ∘ b, b may be memory
  defp enc({:vbin, opc, map, pp, l}, [d, a, b], r), do: vex(opc, map, pp, l, 0, r.(d), r.(a), rm(b, r))

  # vpslld / vpsrld imm: VEX.66.0F 72 /ext ib, destination in vvvv
  defp enc({:vshift, ext, l}, [d, a, n], r), do: [vex(0x72, 1, 1, l, 0, ext, r.(d), {:r, r.(a)}), <<n>>]

  # vcmpps m, a, b, pred: VEX.0F C2 /r ib
  defp enc({:vcmpps, l}, [m, a, b, pred], r), do: [vex(0xC2, 1, 0, l, 0, r.(m), r.(a), rm(b, r)), <<pred>>]

  # vblendvps d, a, b, m: VEX.66.0F3A 4A /r is4 — d = m ? b : a
  defp enc({:vblendvps, l}, [d, a, b, m], r), do: [vex(0x4A, 3, 1, l, 0, r.(d), r.(a), rm(b, r)), <<r.(m) <<< 4>>]

  # fused a·b + c into d, choosing the FMA form by register aliasing
  defp enc({:vfma, l}, [d, a, b, c], r) do
    {rd, ra, rb, rc} = {r.(d), reg_of(a, r), reg_of(b, r), reg_of(c, r)}

    cond do
      rd == rc -> vex(0xB8, 2, 1, l, 0, rd, ra, rm(b, r))
      rd == ra -> vex(0xA8, 2, 1, l, 0, rd, rb, rm(c, r))
      rd == rb -> vex(0xA8, 2, 1, l, 0, rd, ra, rm(c, r))
      true -> [enc(:vmovups_load, [d, c], r), vex(0xB8, 2, 1, l, 0, rd, ra, rm(b, r))]
    end
  end

  defp enc(:vpmulld, [d, a, b], r), do: vex(0x40, 2, 1, 1, 0, r.(d), r.(a), rm(b, r))
  defp enc(:vfmadd231ps, [d, a, b], r), do: vex(0xB8, 2, 1, 1, 0, r.(d), r.(a), rm(b, r))
  defp enc(:vfmadd231ss, [d, a, b], r), do: vex(0xB9, 2, 1, 0, 0, r.(d), r.(a), rm(b, r))
  defp enc(:vbroadcastss, [d, s], r), do: vex(0x18, 2, 1, 1, 0, r.(d), 0, {:r, r.(s)})
  defp enc(:vpbroadcastd, [d, s], r), do: vex(0x58, 2, 1, 1, 0, r.(d), 0, {:r, r.(s)})
  defp enc(:vpmovzxbd, [d, m], r), do: vex(0x31, 2, 1, 1, 0, r.(d), 0, mem(m, r))
  defp enc(:vpmovsxbd, [d, m], r), do: vex(0x21, 2, 1, 1, 0, r.(d), 0, mem(m, r))
  defp enc(:vpmovzxwd, [d, m], r), do: vex(0x33, 2, 1, 1, 0, r.(d), 0, mem(m, r))
  defp enc(:vcvtdq2ps, [d, s], r), do: vex(0x5B, 1, 0, 1, 0, r.(d), 0, {:r, r.(s)})
  defp enc(:vpsrld, [d, s, imm], r), do: [vex(0x72, 1, 1, 1, 0, 2, r.(d), {:r, r.(s)}), <<imm>>]
  defp enc(:vextractf128, [d, s, imm], r), do: [vex(0x19, 3, 1, 1, 0, r.(s), 0, {:r, r.(d)}), <<imm>>]
  defp enc(:vextracti128, [d, s, imm], r), do: [vex(0x39, 3, 1, 1, 0, r.(s), 0, {:r, r.(d)}), <<imm>>]
  defp enc(:vmovshdup, [d, s], r), do: vex(0x16, 1, 2, 0, 0, r.(d), 0, {:r, r.(s)})
  defp enc(:vpshufd, [d, s, imm], r), do: [vex(0x70, 1, 1, 0, 0, r.(d), 0, {:r, r.(s)}), <<imm>>]
  defp enc(:vmovd_x_r, [d, s], r), do: vex(0x6E, 1, 1, 0, 0, r.(d), 0, {:r, r.(s)})
  defp enc(:vmovd_r_x, [d, s], r), do: vex(0x7E, 1, 1, 0, 0, r.(s), 0, {:r, r.(d)})
  defp enc(:vcvtsi2ss, [d, a, s], r), do: vex(0x2A, 1, 2, 0, 0, r.(d), r.(a), {:r, r.(s)})

  defp reg_of({:pool, _}, _r), do: nil
  defp reg_of(x, r), do: r.(x)

  defp rm({:mem, _, _} = m, r), do: mem(m, r)
  defp rm({:pool, bits}, _r), do: {:rip, bits}
  defp rm(x, r), do: {:r, r.(x)}
  defp mem({:mem, b, off}, r), do: {:m, r.(b), off}

  defp rex(w, rr, x, b), do: 0x40 ||| w <<< 3 ||| rr <<< 2 ||| x <<< 1 ||| b

  # legacy (REX) encoding of `opcode /r` with reg field `reg` and r/m operand `rmo`
  defp rex_rm(w, opcode, reg, rmo) do
    {modrm, bsel} = modrm(reg, rmo)
    prefix = rex(w, reg >>> 3, 0, bsel)
    pre = if prefix == 0x40, do: <<>>, else: <<prefix>>
    op = if is_integer(opcode), do: <<opcode>>, else: opcode
    pre <> op <> modrm
  end

  defp vex(opcode, map, pp, l, w, reg, vvvv, rmo) do
    {modrm, bsel} = modrm(reg, rmo)
    b1 = bxor(1, reg >>> 3) <<< 7 ||| 1 <<< 6 ||| bxor(1, bsel) <<< 5 ||| map
    b2 = w <<< 7 ||| bxor(15, vvvv) <<< 3 ||| l <<< 2 ||| pp
    [<<0xC4, b1, b2, opcode>>, modrm]
    |> then(fn io -> if is_binary(modrm), do: IO.iodata_to_binary(io), else: io end)
  end

  # returns {modrm (+sib +disp) bytes, extension bit of the r/m base}
  defp modrm(reg, {:r, n}), do: {<<0b11::2, (reg &&& 7)::3, (n &&& 7)::3>>, n >>> 3}

  defp modrm(reg, {:m, base, disp}) do
    sib = if (base &&& 7) == 4, do: <<0x24>>, else: <<>>
    {<<0b10::2, (reg &&& 7)::3, (base &&& 7)::3>> <> sib <> <<disp::signed-32-little>>, base >>> 3}
  end

  # RIP-relative: mod 00, r/m 101, disp32 patched after layout
  defp modrm(reg, {:rip, bits}), do: {[<<0b00::2, (reg &&& 7)::3, 0b101::3>>, {:rip, bits}], 0}
end
