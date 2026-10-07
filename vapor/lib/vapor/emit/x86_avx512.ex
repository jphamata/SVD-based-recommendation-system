defmodule Vapor.Emit.X86.AVX512 do
  @moduledoc """
  x86-64 AVX-512 backend (the x86-64-v4 level: F, VL, BW, DQ, CD, plus BMI2)
  — pure binary synthesis like `Vapor.Emit.X86`, which it reuses for the
  integer side (general-purpose registers, branches, frame); every vector
  and scalar-float instruction here is EVEX-encoded.

  What the wider ISA buys, without changing a single result bit:

    * **16 lanes per register** — the canonical width. A `:f16l` value (the
      16 partial sums of every canonical dot product) is one zmm instead
      of two ymm, so a weight row is one FMA per 16 weights and the
      canonical reduction tree starts with one `vextractf64x4`.
    * **32 vector registers** (zmm0–30 allocatable, zmm31 backend scratch),
      so larger group factors fit without spilling.
    * **Masked strip tails**: the last `n < 16·g` elements of a strip run
      the vector body once more with opmask registers `k2…k5` on its loads
      and stores (`bzhi` builds the mask), instead of a scalar loop.
      Arithmetic on the masked-off lanes is computed and discarded: lanes
      are independent, and floating-point exceptions are masked.
    * **Embedded broadcast** `{1to16}`: a constant operand is a 4-byte pool
      entry read by every lane, so the pool stays a few cache lines.

  Selections (`vcmpps` into `k1`, then `vblendmps`) and the reduction tree
  are those of the AVX2 backend lane for lane, so the code is bit-identical
  to the oracle — tested on every canonical program and whole models.

  EVEX displacements are always disp32 (never the compressed disp8·N), so
  addressing is the same as in the VEX backend.
  """
  import Bitwise
  alias Vapor.Emit.{Machine, X86}
  import Machine, only: [mi: 4]

  @behaviour Machine

  @t 31
  @k1 1

  # ------------------------------------------------------------ reg model --

  @impl true
  def isa, do: :x86_64_avx512

  @impl true
  def files do
    %{
      g: X86.files().g,
      v: %{count: 32, order: Enum.to_list(0..30), callee_saved: []}
    }
  end

  @impl true
  def group_factors, do: [4, 2, 1]

  @impl true
  def size(:gpr, _g), do: {:g, 1}
  def size(:fpr, _g), do: {:v, 1}
  def size(:strip, g), do: {:v, g}
  def size(:f16l, _g), do: {:v, 1}
  def size(:i32acc, g), do: {:v, g}

  @impl true
  def arg_pin, do: X86.arg_pin()

  # ------------------------------------------------------------ selection --

  @gpr_ops [:arg, :li, :mov, :addi, :add, :sub, :mul, :label, :jmp, :bnez, :beqz, :blt_imm, :bltu,
            :ld_u8, :ld_s8, :ld_u32, :ld64, :st64, :shli, :shri, :st_i32]

  @vops [:vfadd, :vfsub, :vfmul, :vfma, :vfneg, :vrelu, :vsel_lt, :viadd, :visub, :viand, :vixor,
         :vshl, :vshr, :vfmacc]

  @impl true
  def select(:ret, s), do: X86.select(:ret, s)
  def select(i, s) when is_tuple(i) and elem(i, 0) in @gpr_ops, do: X86.select(i, s)

  # scalar binary32 (lane 0 of an xmm, EVEX scalar forms)
  def select({:ldf, f, b, off}, s), do: {[mi({:z, :movss_load}, [f], [b], o: [f, {:mem, b, off}])], s}
  def select({:stf, f, b, off}, s), do: {[mi({:z, :movss_store}, [], [f, b], o: [f, {:mem, b, off}])], s}

  def select({:lif, f, bits}, s),
    do: {[mi(:mov32_imm, [], [], o: [{:phys, 11}, bits]), mi({:z, :movd_x_r}, [f], [], o: [f, {:phys, 11}])], s}

  def select({:bcast, f}, s), do: {[mi({:z, :broadcastss}, [f], [f], o: [f, f])], s}

  def select({:cvt_u8f, f, x}, s),
    do: {[mi({:z, :pxord, 0}, [f], [], o: [f, f, f]), mi({:z, :cvtsi2ss}, [f], [x, f], o: [f, f, x])], s}

  def select({op, f, a, b}, s) when op in [:fadd, :fsub, :fmul] do
    {[mi({:z, :ss, %{fadd: 0x58, fsub: 0x5C, fmul: 0x59}[op]}, [f], [a, b], o: [f, a, b])], s}
  end

  def select({:fmacc, acc, a, b}, %{policy: :fast} = s),
    do: {[mi({:z, :fmadd231ss}, [acc], [acc, a, b], o: [acc, a, b])], s}

  def select({:fmacc, acc, a, b}, s) do
    {[mi({:z, :ss, 0x59}, [], [a, b], o: [{:phys, @t}, a, b]),
      mi({:z, :ss, 0x58}, [acc], [acc], o: [acc, acc, {:phys, @t}])], s}
  end

  # ---- strips: a 16·g-lane loop, then one masked pass over the tail ----
  def select({:strip, n, ptrs, body}, s) do
    w = 16 * s.g
    {lv, s} = Machine.label(s, :strip)
    {lt, s} = Machine.label(s)
    {le, s} = Machine.label(s)
    {vec, s} = Machine.select_all(Enum.map(body, &{:lanes, &1}), s)
    {tail, s} = Machine.tail_copy(body, s)
    {masked, s} = Machine.select_all(Enum.map(tail, &{:masked, &1}), s)

    code =
      [mi(:label, [], [], label: lv), mi(:cmp_imm, [], [n], o: [n, w]), mi(:jl, [], [], br: lt)] ++
        vec ++
        Enum.map(ptrs, &mi(:lea, [&1], [&1], o: [&1, {:mem, &1, 4 * w}])) ++
        [mi(:lea, [n], [n], o: [n, {:mem, n, -w}]), mi(:jmp, [], [], br: lv, uncond: true),
         mi(:label, [], [], label: lt), mi(:test, [], [n], o: [n]), mi(:jz, [], [], br: le),
         mi({:z, :tail_masks, s.g}, [], [n], o: [n])] ++
        masked ++
        Enum.map(ptrs, &mi({:z, :lea_idx4}, [&1], [&1, n], o: [&1, n])) ++
        [mi(:label, [], [], label: le)]

    {code, s}
  end

  def select({:lanes, {:vld, v, b, off}}, s),
    do: {[lanewise({:z, :movups_load}, [v], [b], lanes(s), &[sub(v, &1), {:mem, b, off + 64 * &1}])], s}

  def select({:lanes, {:vst, v, b, off}}, s),
    do: {[lanewise({:z, :movups_store}, [], [v, b], lanes(s), &[sub(v, &1), {:mem, b, off + 64 * &1}])], s}

  def select({:lanes, i}, s) when elem(i, 0) in @vops, do: vop(i, lanes_of(elem(i, 1)), s)

  def select({:masked, {:vld, v, b, off}}, s),
    do: {[lanewise({:z, :movups_kload}, [v], [b], lanes(s), &[sub(v, &1), {:mem, b, off + 64 * &1}, 2 + &1])], s}

  def select({:masked, {:vst, v, b, off}}, s),
    do: {[lanewise({:z, :movups_kstore}, [], [v, b], lanes(s), &[sub(v, &1), {:mem, b, off + 64 * &1}, 2 + &1])], s}

  def select({:masked, i}, s), do: select({:lanes, i}, s)

  # bare vector ops: one per register of the value
  def select({:vld, v, b, off}, s),
    do: {[lanewise({:z, :movups_load}, [v], [b], lanes_of(v), &[sub(v, &1), {:mem, b, off + 64 * &1}])], s}

  def select({:vst, v, b, off}, s),
    do: {[lanewise({:z, :movups_store}, [], [v, b], lanes_of(v), &[sub(v, &1), {:mem, b, off + 64 * &1}])], s}

  def select(i, s) when is_tuple(i) and elem(i, 0) in @vops, do: vop(i, lanes_of(elem(i, 1)), s)

  def select({:vlane0, f, v}, s), do: {[mi({:z, :movaps}, [f], [v], o: [f, sub(v, 0)])], s}

  def select({:vconst, v, bits}, s),
    do: {[lanewise({:z, :broadcastss}, [v], [], lanes_of(v), &[sub(v, &1), {:pool, bits}])], s}

  def select({:vsplat, v, f}, s),
    do: {[lanewise({:z, :broadcastss}, [v], [f], lanes_of(v), &[sub(v, &1), f])], s}

  # ---- f16l: 16 binary32 lanes = 1 zmm ----
  def select({:vzero16, v}, s), do: {[mi({:z, :pxord, 2}, [v], [], o: [v, v, v])], s}
  def select({:vld16, v, b, off}, s), do: {[mi({:z, :movups_load}, [v], [b], o: [v, {:mem, b, off}])], s}
  def select({:vfadd16, d, a, b}, s), do: {[mi({:z, :bin, 0x58, 1, 0, 2}, [d], [a, b], o: [d, a, b])], s}

  def select({:vld_bf16, v, b, off}, s),
    do: {[mi({:z, :pmovzxwd}, [v], [b], o: [v, {:mem, b, off}]), mi({:z, :shift, 6}, [v], [v], o: [v, v, 16])], s}

  def select({:vld_nib, lo, hi, b, off}, s) do
    t = {:phys, @t}

    {[mi({:z, :pmovzxbd}, [], [b], o: [t, {:mem, b, off}]),
      mi({:z, :bin, 0xDB, 1, 1, 2}, [lo], [], o: [lo, t, {:pool, 0x0F}]),
      mi({:z, :shift, 2}, [hi], [], o: [hi, t, 4]),
      mi({:z, :cvtdq2ps}, [lo], [lo], o: [lo, lo]),
      mi({:z, :cvtdq2ps}, [hi], [hi], o: [hi, hi])], s}
  end

  def select({:vmul_sf, d, a, f}, s) do
    {[mi({:z, :broadcastss}, [], [f], o: [{:phys, @t}, f]),
      mi({:z, :bin, 0x59, 1, 0, 2}, [d], [a], o: [d, a, {:phys, @t}])], s}
  end

  def select({:vfmacc_mem, acc, w, b, off}, %{policy: :fast} = s),
    do: {[mi({:z, :fma231, 2}, [acc], [acc, w, b], o: [acc, w, {:mem, b, off}])], s}

  def select({:vfmacc_mem, acc, w, b, off}, s) do
    {[mi({:z, :bin, 0x59, 1, 0, 2}, [], [w, b], o: [{:phys, @t}, w, {:mem, b, off}]),
      mi({:z, :bin, 0x58, 1, 0, 2}, [acc], [acc], o: [acc, acc, {:phys, @t}])], s}
  end

  # the canonical tree: lane i + lane i+8, then +4, +2, +1 (as in the AVX2 backend)
  def select({:vreduce16, f, v}, s) do
    x = {:phys, @t}

    {[mi({:z, :extract64x4, 0x1B}, [], [v], o: [x, v]),
      mi({:z, :bin, 0x58, 1, 0, 1}, [f], [v], o: [f, v, x]),
      mi({:z, :extract32x4, 0x19}, [], [f], o: [x, f]),
      mi({:z, :bin, 0x58, 1, 0, 0}, [f], [f], o: [f, f, x]),
      mi({:z, :movhlps}, [], [f], o: [x, x, f]),
      mi({:z, :bin, 0x58, 1, 0, 0}, [f], [f], o: [f, f, x]),
      mi({:z, :movshdup}, [], [f], o: [x, f]),
      mi({:z, :ss, 0x58}, [f], [f], o: [f, f, x])], s}
  end

  # the same tree with max(a, b) = (a < b) ? b : a at every node
  def select({:vreduce16_max, f, v}, s) do
    x = {:phys, @t}
    sel = fn ll, a -> [mi({:z, :cmpps, ll}, [], [a], o: [{:k, @k1}, a, x, 0x11]), mi({:z, :blendm, ll}, [f], [a], o: [f, a, x, @k1])] end

    {[mi({:z, :extract64x4, 0x1B}, [], [v], o: [x, v])] ++ sel.(1, v) ++
       [mi({:z, :extract32x4, 0x19}, [], [f], o: [x, f])] ++ sel.(0, f) ++
       [mi({:z, :movhlps}, [], [f], o: [x, x, f])] ++ sel.(0, f) ++
       [mi({:z, :movshdup}, [], [f], o: [x, f])] ++ sel.(0, f), s}
  end

  # ---- i8 strips (16·g bytes per iteration; the scalar tail is the kernel's) ----
  def select({:vzero_i32, acc}, s),
    do: {[lanewise({:z, :pxord, 2}, [acc], [], lanes(s), &[sub(acc, &1), sub(acc, &1), sub(acc, &1)])], s}

  def select({:strip_i8, n, ptrs, body, tail}, s) do
    w = 16 * s.g
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
       [mi({:z, :pmovsxbd}, [], [pa], o: [{:phys, @t}, {:mem, pa, 16 * k}]),
        mi({:z, :pmovsxbd}, [t], [pb], o: [t, {:mem, pb, 16 * k}]),
        mi({:z, :pmulld}, [t], [t], o: [t, t, {:phys, @t}]),
        mi({:z, :bin, 0xFE, 1, 1, 2}, [acc], [acc, t], o: [sub(acc, k), sub(acc, k), t])]
     end), s}
  end

  # integer sums are exact: any order gives the same s32
  def select({:vred_i32, x, acc}, s) do
    {t, s} = Machine.fresh(s, :fpr)
    y = {:phys, @t}
    add = fn ll -> {:z, :bin, 0xFE, 1, 1, ll} end

    sum =
      case s.g do
        1 -> [mi({:z, :movaps}, [t], [acc], o: [t, sub(acc, 0)])]
        g -> [mi(add.(2), [t], [acc], o: [t, sub(acc, 0), sub(acc, 1)])] ++
               for(k <- 2..(g - 1)//1, do: mi(add.(2), [t], [t, acc], o: [t, t, sub(acc, k)]))
      end

    {sum ++
       [mi({:z, :extract64x4, 0x3B}, [], [t], o: [y, t]),
        mi(add.(1), [t], [t], o: [t, t, y]),
        mi({:z, :extract32x4, 0x39}, [], [t], o: [y, t]),
        mi(add.(0), [t], [t], o: [t, t, y]),
        mi({:z, :pshufd}, [], [t], o: [y, t, 0x4E]),
        mi(add.(0), [t], [t], o: [t, t, y]),
        mi({:z, :pshufd}, [], [t], o: [y, t, 0xB1]),
        mi(add.(0), [t], [t], o: [t, t, y]),
        mi({:z, :movd_r_x}, [x], [t], o: [x, t])], s}
  end

  # ---------------------------------------------- vector operations (VOPs) --

  @binops %{vfadd: {0x58, 1, 0, true}, vfsub: {0x5C, 1, 0, false}, vfmul: {0x59, 1, 0, true},
            viadd: {0xFE, 1, 1, true}, visub: {0xFA, 1, 1, false}, viand: {0xDB, 1, 1, true},
            vixor: {0xEF, 1, 1, true}}

  defp vop({op, d, a, b}, ks, s) when is_map_key(@binops, op) do
    {opc, map, pp, comm} = Map.fetch!(@binops, op)
    {a, b} = if const?(a) and comm and not const?(b), do: {b, a}, else: {a, b}
    {pre, a, s} = in_reg(a, s)
    {pre ++ [lanewise({:z, :bin, opc, map, pp, 2}, [d], regs([a, b]), ks, &[sub(d, &1), opk(a, &1), opk(b, &1)])], s}
  end

  defp vop({op, d, a, n}, ks, s) when op in [:vshl, :vshr] do
    ext = if op == :vshl, do: 6, else: 2
    {pre, a, s} = in_reg(a, s)
    {pre ++ [lanewise({:z, :shift, ext}, [d], regs([a]), ks, &[sub(d, &1), opk(a, &1), n])], s}
  end

  defp vop({:vfneg, d, a}, ks, s), do: vop({:vixor, d, a, {:const, 0x8000_0000}}, ks, s)

  # relu(x) = maxps(x, +0): the second operand on NaN and on ±0 ties
  defp vop({:vrelu, d, a}, ks, s) do
    {pre, a, s} = in_reg(a, s)
    {pre ++ [lanewise({:z, :bin, 0x5F, 1, 0, 2}, [d], regs([a]), ks, &[sub(d, &1), opk(a, &1), {:pool, 0}])], s}
  end

  defp vop({:vfma, d, a, b, c}, ks, %{policy: :fast} = s) do
    {pa, a, s} = in_reg(a, s)
    {pc, c, s} = in_reg(c, s)
    {pa ++ pc ++ Enum.map(ks, &mi({:z, :fma}, [d], regs([a, b, c]), o: [sub(d, &1), opk(a, &1), opk(b, &1), opk(c, &1)])), s}
  end

  defp vop({:vfma, d, a, b, c}, ks, s) do
    {pa, a, s} = in_reg(a, s)

    {pa ++ Enum.flat_map(ks, fn k ->
       [mi({:z, :bin, 0x59, 1, 0, 2}, [], regs([a, b]), o: [{:phys, @t}, opk(a, k), opk(b, k)]),
        mi({:z, :bin, 0x58, 1, 0, 2}, [d], regs([c]), o: [sub(d, k), {:phys, @t}, opk(c, k)])]
     end), s}
  end

  defp vop({:vfmacc, acc, a, b}, ks, s), do: vop({:vfma, acc, a, b, acc}, ks, s)

  # d = (a < b) ? x : y — vcmpps (LT_OQ) into k1, vblendmps takes its r/m
  # operand where k1 is set; constant operands as in the AVX2 backend
  defp vop({:vsel_lt, d, a, b, x, y}, ks, s) do
    {ca, cb, pred} = if const?(a), do: {b, a, 0x1E}, else: {a, b, 0x11}
    {pre1, ca, s} = in_reg(ca, s)
    {src1, src2, pred} = if const?(y) and not const?(x), do: {x, y, bxor(pred, 0x04)}, else: {y, x, pred}
    {pre2, src1, s} = in_reg(src1, s)

    {pre1 ++ pre2 ++
       Enum.flat_map(ks, fn k ->
         [mi({:z, :cmpps, 2}, [], regs([ca, cb]), o: [{:k, @k1}, opk(ca, k), opk(cb, k), pred]),
          mi({:z, :blendm, 2}, [d], regs([src1, src2]), o: [sub(d, k), opk(src1, k), opk(src2, k), @k1])]
       end), s}
  end

  defp const?({:const, _}), do: true
  defp const?(_), do: false

  defp in_reg({:const, bits}, s) do
    {t, s} = Machine.fresh(s, :fpr)
    {[mi({:z, :broadcastss}, [t], [], o: [t, {:pool, bits}])], t, s}
  end

  defp in_reg(o, s), do: {[], o, s}

  defp regs(os), do: Enum.filter(os, &match?({:vr, _, _, _}, &1))
  defp opk({:const, bits}, _k), do: {:pool, bits}
  defp opk({:vr, _, _, 1} = v, _k), do: sub(v, 0)
  defp opk(v, k), do: sub(v, k)
  defp lanes(s), do: 0..(s.g - 1)
  defp lanes_of({:vr, _, _, size}), do: 0..(size - 1)
  defp sub(v, k), do: {:sub, v, k}

  defp lanewise(op, defs, uses, ks, ops) do
    shared = Enum.count(ks) > 1 and Enum.any?(uses, &match?({:vr, _, _, 1}, &1))
    mi(op, defs, uses, bundle: Enum.map(ks, ops), ec: shared)
  end

  @impl true
  def materialize(_key, _r), do: []

  # ------------------------------------------------------- frame, branches --

  @impl true
  def prologue(a), do: X86.prologue(a)
  @impl true
  def epilogue(a), do: X86.epilogue(a)
  @impl true
  def branch_size(op), do: X86.branch_size(op)
  @impl true
  def encode_branch(op, o, rel, r), do: X86.encode_branch(op, o, rel, r)

  @doc "Pool entry: one binary32 (operands read it with an embedded `{1to16}` broadcast)."
  def pool_entry(bits), do: <<bits::32-little>>
  def pool_align, do: 64
  def pad_byte, do: 0xCC

  # ------------------------------------------------------------- encoding --

  @impl true
  def encode({:z, _} = op, o, r), do: op |> enc(o, r) |> resolve()
  def encode({:z, _, _} = op, o, r), do: op |> enc(o, r) |> resolve()
  def encode({:z, _, _, _, _, _} = op, o, r), do: op |> enc(o, r) |> resolve()
  def encode(op, o, r), do: X86.encode(op, o, r)

  defp resolve(io) do
    {bin, fixes} =
      io
      |> List.wrap()
      |> List.flatten()
      |> Enum.reduce({<<>>, []}, fn
        {:rip, bits}, {acc, fx} -> {acc <> <<0::32>>, [{byte_size(acc), bits} | fx]}
        b, {acc, fx} -> {acc <> b, fx}
      end)

    if fixes == [], do: bin, else: {:fix, bin, Enum.reverse(fixes)}
  end

  # evex(opcode, map, pp, L'L, W, reg, vvvv, r/m, opmask, zeroing)
  defp ev(opc, map, pp, ll, w, reg, vvvv, rmo, aaa \\ 0, z \\ 0) do
    {modrm, b3, x4, bc} = modrm(reg, rmo)
    p0 = bxor(1, reg >>> 3 &&& 1) <<< 7 ||| bxor(1, x4) <<< 6 ||| bxor(1, b3) <<< 5 ||| bxor(1, reg >>> 4 &&& 1) <<< 4 ||| map
    p1 = w <<< 7 ||| bxor(15, vvvv &&& 15) <<< 3 ||| 1 <<< 2 ||| pp
    p2 = z <<< 7 ||| ll <<< 5 ||| bc <<< 4 ||| bxor(1, vvvv >>> 4 &&& 1) <<< 3 ||| aaa
    [<<0x62, p0, p1, p2, opc>>, modrm]
  end

  # {modrm (+sib, disp) iodata, r/m bit 3, r/m bit 4 (registers), broadcast bit}
  defp modrm(reg, {:r, n}), do: {<<0b11::2, (reg &&& 7)::3, (n &&& 7)::3>>, n >>> 3 &&& 1, n >>> 4 &&& 1, 0}

  defp modrm(reg, {:m, base, disp}) do
    sib = if (base &&& 7) == 4, do: <<0x24>>, else: <<>>
    {<<0b10::2, (reg &&& 7)::3, (base &&& 7)::3>> <> sib <> <<disp::signed-32-little>>, base >>> 3, 0, 0}
  end

  defp modrm(reg, {:rip, bits}), do: {[<<0b00::2, (reg &&& 7)::3, 0b101::3>>, {:rip, bits}], 0, 0, 0}
  defp modrm(reg, {:bcast, bits}), do: (({m, b, x, _} = modrm(reg, {:rip, bits})); {m, b, x, 1})

  defp rm({:mem, b, off}, r), do: {:m, r.(b), off}
  defp rm({:pool, bits}, _r), do: {:rip, bits}
  defp rm(x, r), do: {:r, r.(x)}

  # packed operands: a pool constant is an embedded broadcast
  defp prm({:pool, bits}, _r), do: {:bcast, bits}
  defp prm(x, r), do: rm(x, r)

  defp enc({:z, :movups_load}, [d, m], r), do: ev(0x10, 1, 0, 2, 0, r.(d), 0, rm(m, r))
  defp enc({:z, :movups_store}, [s, m], r), do: ev(0x11, 1, 0, 2, 0, r.(s), 0, rm(m, r))
  defp enc({:z, :movups_kload}, [d, m, k], r), do: ev(0x10, 1, 0, 2, 0, r.(d), 0, rm(m, r), k, 1)
  defp enc({:z, :movups_kstore}, [s, m, k], r), do: ev(0x11, 1, 0, 2, 0, r.(s), 0, rm(m, r), k)
  defp enc({:z, :movaps}, [d, s], r), do: if(r.(d) == r.(s), do: <<>>, else: ev(0x28, 1, 0, 2, 0, r.(d), 0, {:r, r.(s)}))
  defp enc({:z, :movss_load}, [d, m], r), do: ev(0x10, 1, 2, 0, 0, r.(d), 0, rm(m, r))
  defp enc({:z, :movss_store}, [s, m], r), do: ev(0x11, 1, 2, 0, 0, r.(s), 0, rm(m, r))
  defp enc({:z, :movd_x_r}, [d, s], r), do: ev(0x6E, 1, 1, 0, 0, r.(d), 0, {:r, r.(s)})
  defp enc({:z, :movd_r_x}, [d, s], r), do: ev(0x7E, 1, 1, 0, 0, r.(s), 0, {:r, r.(d)})
  defp enc({:z, :cvtsi2ss}, [d, a, s], r), do: ev(0x2A, 1, 2, 0, 0, r.(d), r.(a), {:r, r.(s)})
  defp enc({:z, :ss, opc}, [d, a, b], r), do: ev(opc, 1, 2, 0, 0, r.(d), r.(a), rm(b, r))
  defp enc({:z, :fmadd231ss}, [d, a, b], r), do: ev(0xB9, 2, 1, 0, 0, r.(d), r.(a), rm(b, r))
  defp enc({:z, :pxord, ll}, [d, a, b], r), do: ev(0xEF, 1, 1, ll, 0, r.(d), r.(a), rm(b, r))
  defp enc({:z, :bin, opc, map, pp, ll}, [d, a, b], r), do: ev(opc, map, pp, ll, 0, r.(d), r.(a), prm(b, r))
  defp enc({:z, :shift, ext}, [d, a, n], r), do: [ev(0x72, 1, 1, 2, 0, ext, r.(d), {:r, r.(a)}), <<n>>]

  # vbroadcastss zmm, xmm | m32 (a pool constant is read as its m32)
  defp enc({:z, :broadcastss}, [d, s], r), do: ev(0x18, 2, 1, 2, 0, r.(d), 0, rm(s, r))

  defp enc({:z, :cmpps, ll}, [{:k, k}, a, b, pred], r), do: [ev(0xC2, 1, 0, ll, 0, k, r.(a), prm(b, r)), <<pred>>]
  # vblendmps d{k}, a, b: d = k ? b : a
  defp enc({:z, :blendm, ll}, [d, a, b, k], r), do: ev(0x65, 2, 1, ll, 0, r.(d), r.(a), prm(b, r), k)

  defp enc({:z, :fma}, [d, a, b, c], r) do
    {rd, ra, rb, rc} = {r.(d), reg_of(a, r), reg_of(b, r), reg_of(c, r)}

    cond do
      rd == rc -> ev(0xB8, 2, 1, 2, 0, rd, ra, prm(b, r))
      rd == ra -> ev(0xA8, 2, 1, 2, 0, rd, rb, prm(c, r))
      rd == rb -> ev(0xA8, 2, 1, 2, 0, rd, ra, prm(c, r))
      true -> [copy(rd, c, r), ev(0xB8, 2, 1, 2, 0, rd, ra, prm(b, r))]
    end
  end

  defp enc({:z, :fma231, ll}, [d, a, b], r), do: ev(0xB8, 2, 1, ll, 0, r.(d), r.(a), prm(b, r))
  defp enc({:z, :pmovzxbd}, [d, m], r), do: ev(0x31, 2, 1, 2, 0, r.(d), 0, rm(m, r))
  defp enc({:z, :pmovzxwd}, [d, m], r), do: ev(0x33, 2, 1, 2, 0, r.(d), 0, rm(m, r))
  defp enc({:z, :pmovsxbd}, [d, m], r), do: ev(0x21, 2, 1, 2, 0, r.(d), 0, rm(m, r))
  defp enc({:z, :pmulld}, [d, a, b], r), do: ev(0x40, 2, 1, 2, 0, r.(d), r.(a), rm(b, r))
  defp enc({:z, :cvtdq2ps}, [d, s], r), do: ev(0x5B, 1, 0, 2, 0, r.(d), 0, {:r, r.(s)})
  # vextractf64x4 / vextracti64x4 ymm, zmm, 1 (W1); vextractf32x4 / vextracti32x4 xmm, ymm, 1
  defp enc({:z, :extract64x4, opc}, [d, s], r), do: [ev(opc, 3, 1, 2, 1, r.(s), 0, {:r, r.(d)}), <<1>>]
  defp enc({:z, :extract32x4, opc}, [d, s], r), do: [ev(opc, 3, 1, 1, 0, r.(s), 0, {:r, r.(d)}), <<1>>]
  defp enc({:z, :movhlps}, [d, a, b], r), do: ev(0x12, 1, 0, 0, 0, r.(d), r.(a), {:r, r.(b)})
  defp enc({:z, :movshdup}, [d, s], r), do: ev(0x16, 1, 2, 0, 0, r.(d), 0, {:r, r.(s)})
  defp enc({:z, :pshufd}, [d, s, imm], r), do: [ev(0x70, 1, 1, 0, 0, r.(d), 0, {:r, r.(s)}), <<imm>>]

  # tail masks for a strip of group factor g: k(2+j) ← bits 16j…16j+15 of
  # (2ⁿ − 1), n < 16·g ≤ 64:  mov r11, −1 · bzhi r11, r11, n · per j: kmovw, shr 16
  defp enc({:z, :tail_masks, g}, [n], r) do
    rn = r.(n)
    bzhi = <<0xC4, 0x42, 0x80 ||| bxor(15, rn) <<< 3, 0xF5, 0xDB>>
    kmovs = for j <- 0..(g - 1), into: <<>>, do: <<0xC4, 0xC1, 0x78, 0x92, 0xC3 ||| (2 + j) <<< 3>> <> if(j < g - 1, do: <<0x49, 0xC1, 0xEB, 16>>, else: <<>>)
    <<0x49, 0xC7, 0xC3, 0xFF, 0xFF, 0xFF, 0xFF>> <> bzhi <> kmovs
  end

  # lea p, [p + 4·n]  (REX.W 8D /r, SIB scale 4, disp8 0 so that rbp/r13 bases encode)
  defp enc({:z, :lea_idx4}, [p, n], r) do
    {rp, rn} = {r.(p), r.(n)}
    <<0x48 ||| (rp >>> 3) <<< 2 ||| (rn >>> 3) <<< 1 ||| rp >>> 3, 0x8D, 0b01::2, (rp &&& 7)::3, 0b100::3, 0b10::2, (rn &&& 7)::3, (rp &&& 7)::3, 0>>
  end

  defp copy(rd, {:pool, bits}, _r), do: ev(0x18, 2, 1, 2, 0, rd, 0, {:rip, bits})
  defp copy(rd, c, r), do: if(rd == r.(c), do: <<>>, else: ev(0x28, 1, 0, 2, 0, rd, 0, {:r, r.(c)}))

  defp reg_of({:pool, _}, _r), do: nil
  defp reg_of(x, r), do: r.(x)
end
