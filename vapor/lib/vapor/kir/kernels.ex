defmodule Vapor.KIR.Kernels do
  @moduledoc """
  The kernel library, written once in portable IR (`Vapor.KIR`).

  Every kernel has the same ABI on every substrate:

      void kernel(const uint64_t *args)

  so the worker needs exactly one call signature, and a kernel's argument
  count is bounded by register pressure (decided by the allocator), not by
  the ABI's argument registers.
  """
  import Vapor.KIR, only: [gpr: 1, fpr: 1, vec: 2]

  @doc """
  Fused elementwise region (Section 2.2 cut sweep output) over a 2-D
  iteration space `[R, C]` (leading extents collapsed into `R`).

  `spec = %{inputs: [class], outputs: [slot], ops: [{slot, prim, [operand]}]}`
  where each input class is `:full` (`[R,C]`), `:row` (`[1,C]`, shared by
  every row), `:col` (`[R,1]`, one scalar per row) or `:scalar`; ops are
  `Vapor.Canon` primitives over operands `{:in, i}`, `{:t, k}` (earlier op
  results), `{:splat, bits}` and `{:imm, n}`. `inputs: p` (an integer) means
  `p` full inputs.

  args: `[R, C, in_0 … in_{p-1}, out_0 … out_{q-1}]`

  Constants are never held in vector registers by the portable code: they
  stay `{:const, bits}` operands and each backend chooses the cheapest form
  (x86: 32-byte memory operands from a per-kernel pool; AArch64: hoisted
  broadcast registers; RVV: scalar `.vf`/`.vx` operands).
  """
  def ew(spec) do
    %{inputs: classes, outputs: outs, ops: ops} = spec = normalize_ew(spec)
    p = length(classes)
    q = length(outs)
    [rows, cols] = [gpr(0), gpr(1)]
    ptr = fn i -> gpr(2 + i) end
    n = gpr(2 + p + q)
    rowp = fn i -> gpr(3 + p + q + i) end
    base = 3 + 2 * (p + q)
    cls = List.to_tuple(classes)

    in_reg = fn i ->
      if elem(cls, i) in [:full, :row], do: vec(base + i, :strip), else: fpr(base + i)
    end

    t_reg = fn k -> vec(base + p + k, :strip) end

    opnd = fn
      {:in, i} -> in_reg.(i)
      {:t, k} -> t_reg.(k)
      {:splat, b} -> {:const, b}
      {:imm, v} -> v
    end

    used =
      (Enum.flat_map(ops, fn {_, _, a} -> a end) ++ outs)
      |> Enum.flat_map(fn
        {:in, i} -> [i]
        _ -> []
      end)
      |> Enum.uniq()
      |> Enum.sort()

    strip_ptr = fn i -> if elem(cls, i) == :row, do: rowp.(i), else: ptr.(i) end
    vec_ins = Enum.filter(used, &(elem(cls, &1) in [:full, :row]))

    body =
      Enum.map(vec_ins, fn i -> {:vld, in_reg.(i), strip_ptr.(i), 0} end) ++
        Enum.map(ops, fn {{:t, k}, op, args} -> ew_inst(op, t_reg.(k), Enum.map(args, opnd)) end) ++
        Enum.map(Enum.with_index(outs), fn {s, j} -> {:vst, opnd.(s), ptr.(p + j), 0} end)

    full_ins = for i <- 0..(p - 1)//1, elem(cls, i) == :full, do: i
    strip_ptrs = Enum.map(full_ins, ptr) ++ Enum.map(for(i <- 0..(p - 1)//1, elem(cls, i) == :row, do: i), rowp) ++
                   Enum.map(0..(q - 1)//1, &ptr.(p + &1))

    scalars = for i <- used, elem(cls, i) == :scalar, do: [{:ldf, in_reg.(i), ptr.(i), 0}, {:bcast, in_reg.(i)}]

    per_row =
      Enum.flat_map(0..(p - 1)//1, fn i ->
        case elem(cls, i) do
          :col -> if i in used, do: [{:ldf, in_reg.(i), ptr.(i), 0}, {:bcast, in_reg.(i)}, {:addi, ptr.(i), ptr.(i), 4}],
                               else: [{:addi, ptr.(i), ptr.(i), 4}]
          :row -> [{:mov, rowp.(i), ptr.(i)}]
          _ -> []
        end
      end)

    code =
      [{:arg, rows, 0}, {:arg, cols, 1}] ++
        Enum.map(0..(p + q - 1), fn i -> {:arg, ptr.(i), 2 + i} end) ++
        List.flatten(scalars) ++
        [{:beqz, rows, {:l, :done}}, {:label, {:l, :row}}] ++
        per_row ++
        [{:mov, n, cols}, {:strip, n, strip_ptrs, body},
         {:addi, rows, rows, -1}, {:bnez, rows, {:l, :row}},
         {:label, {:l, :done}}, :ret]

    %Vapor.KIR.Kernel{name: :ew, args: [:n, :n | List.duplicate(:ptr, p + q)], code: code, meta: %{spec: spec}}
  end

  @doc false
  def normalize_ew(%{inputs: p} = spec) when is_integer(p), do: %{spec | inputs: List.duplicate(:full, p)}
  def normalize_ew(spec), do: spec

  defp ew_inst(:add, d, [a, b]), do: {:vfadd, d, a, b}
  defp ew_inst(:sub, d, [a, b]), do: {:vfsub, d, a, b}
  defp ew_inst(:mul, d, [a, b]), do: {:vfmul, d, a, b}
  defp ew_inst(:fma, d, [a, b, c]), do: {:vfma, d, a, b, c}
  defp ew_inst(:neg, d, [a]), do: {:vfneg, d, a}
  defp ew_inst(:relu, d, [a]), do: {:vrelu, d, a}
  defp ew_inst(:sel_lt, d, [a, b, x, y]), do: {:vsel_lt, d, a, b, x, y}
  defp ew_inst(:iadd, d, [a, b]), do: {:viadd, d, a, b}
  defp ew_inst(:isub, d, [a, b]), do: {:visub, d, a, b}
  defp ew_inst(:iand, d, [a, b]), do: {:viand, d, a, b}
  defp ew_inst(:ixor, d, [a, b]), do: {:vixor, d, a, b}
  defp ew_inst(:shl, d, [a, n]), do: {:vshl, d, a, n}
  defp ew_inst(:shr, d, [a, n]), do: {:vshr, d, a, n}

  @doc """
  Canonical reduction of the rows of an `[R, C]` matrix, `C ≡ 0 (mod 16)`
  (`Vapor.Runtime.Oracle.reduce_row/2`). args: `[y, x, R, C]`
  """
  def reduce(op) when op in [:sum, :max] do
    [py, px, rows, cols, k] = Enum.map(0..4, &gpr/1)
    [acc, t] = [vec(5, :f16l), vec(6, :f16l)]
    f = fpr(7)

    {init, step, fold} =
      case op do
        :sum -> {{:vzero16, acc}, {:vfadd, acc, acc, t}, {:vreduce16, f, acc}}
        :max -> {{:vconst, acc, 0xFF7F_FFFF}, {:vsel_lt, acc, acc, t, t, acc}, {:vreduce16_max, f, acc}}
      end

    code = [
      {:arg, py, 0}, {:arg, px, 1}, {:arg, rows, 2}, {:arg, cols, 3},
      {:beqz, rows, {:l, :done}},
      {:label, {:l, :row}},
      init,
      {:mov, k, cols},
      {:label, {:l, :chunk}},
      {:vld, t, px, 0},
      step,
      {:addi, px, px, 64},
      {:addi, k, k, -16},
      {:bnez, k, {:l, :chunk}},
      fold,
      {:stf, f, py, 0},
      {:addi, py, py, 4},
      {:addi, rows, rows, -1},
      {:bnez, rows, {:l, :row}},
      {:label, {:l, :done}},
      :ret
    ]

    %Vapor.KIR.Kernel{name: {:reduce, op}, args: [:ptr, :ptr, :n, :n], code: code}
  end

  @doc """
  y[b, n] = Σ_k W[n, k]·x[b, k] in the canonical per-row order (lane l of 16
  accumulates k ≡ l, then the tree). Rows are processed `r` at a time with
  independent accumulators — every row's arithmetic is unchanged, only the
  instruction-level parallelism grows. args: `[y, W, x, N, K, B, ldy]` (output
  rows `ldy` floats apart: a thread may own a range of `N`)
  """
  def gemv_f32, do: gemv_f32(4) |> with_fallbacks([gemv_f32(2), gemv_f32(1)])

  @doc """
  The same product with `W` stored as bfloat16 (`{:vld_bf16, …}` widens 16
  weights exactly as they are loaded): bit-identical to `gemv_f32` over the
  widened matrix, reading half the bytes. args as `gemv_f32`.
  """
  def gemv_bf16, do: gemv_f32(4, :bf16) |> with_fallbacks([gemv_f32(2, :bf16), gemv_f32(1, :bf16)])

  @doc """
  Row-predicated GEMV (`Vapor.Algebra.Term.linear_masked/3`): the product of
  `gemv_f32`/`gemv_bf16`, except that an activation row whose mask word is
  ±0 stores `+0` and touches no weight — a block of weight rows is only
  read for the active rows. Active rows run the very instructions of the
  dense kernel, hence the same bits. args: `[y, W, x, n, k, b, ldy, mask]`.
  """
  def gemv_masked(wdt \\ :f32) when wdt in [:f32, :bf16],
    do: gemv_f32(4, wdt, true) |> with_fallbacks([gemv_f32(2, wdt, true), gemv_f32(1, wdt, true)])

  def gemv_f32(r, wdt \\ :f32, masked \\ false) when r in [1, 2, 4] and wdt in [:f32, :bf16] do
    # bytes per weight: 4 (f32) or 2 (bf16) — as a shift
    wsh = if wdt == :f32, do: 2, else: 1
    # Weight-stationary order: for each block of r weight rows, every
    # activation row in turn (the block stays in cache across the batch, so
    # a batch of b rows reads W once, not b times). Each output is still
    # one row of W against one row of x in the canonical 16-lane order, so
    # the bits do not depend on the loop order, the batch or the split.
    [pw, cnt, pyrow, xrow, py, bcnt] = Enum.map(1..6, &gpr/1)
    pwj = fn j -> gpr(20 + j) end
    acc = fn j -> vec(30 + j, :f16l) end
    xv = vec(40, :f16l)
    wv = fn j -> vec(41 + j, :f16l) end
    fj = fn j -> fpr(50 + j) end

    block = fn rr, tag ->
      # fresh registers per site keep every live range local
      [st, xp, kc, yb] = Enum.map(0..3, &gpr(60 + 10 * tag_n(tag) + &1))
      [pm, mk] = [gpr(64 + 10 * tag_n(tag)), gpr(65 + 10 * tag_n(tag))]

      # masked: the row's mask word, sign bit shifted out; ±0 skips the row
      {mask_init, mask_test, mask_skip, mask_step} =
        if masked do
          {[{:arg, pm, 7}],
           [{:ld_u32, mk, pm, 0}, {:shli, mk, mk, 33}, {:beqz, mk, {:l, {tag, :skip}}}],
           [{:jmp, {:l, {tag, :adv}}}, {:label, {:l, {tag, :skip}}}, {:lif, fj.(0), 0}] ++
             Enum.map(0..(rr - 1), &{:stf, fj.(0), py, 4 * &1}) ++ [{:label, {:l, {tag, :adv}}}],
           [{:addi, pm, pm, 4}]}
        else
          {[], [], [], []}
        end

      [{:arg, xrow, 2}, {:mov, py, pyrow}] ++ mask_init ++ [{:arg, bcnt, 5},
       {:beqz, bcnt, {:l, {tag, :bdone}}},
       {:label, {:l, {tag, :b}}}] ++ mask_test ++
       [{:mov, pwj.(0), pw}, {:arg, st, 4}, {:shli, st, st, wsh}] ++
        Enum.map(1..(rr - 1)//1, fn j -> {:add, pwj.(j), pwj.(j - 1), st} end) ++
        Enum.map(0..(rr - 1), &{:vzero16, acc.(&1)}) ++
        [{:mov, xp, xrow}, {:arg, kc, 4}, {:label, {:l, {tag, :k}}}, {:vld, xv, xp, 0}] ++
        Enum.flat_map(0..(rr - 1), fn j ->
          load = if wdt == :f32, do: {:vld, wv.(j), pwj.(j), 0}, else: {:vld_bf16, wv.(j), pwj.(j), 0}
          [load, {:vfmacc, acc.(j), wv.(j), xv}, {:addi, pwj.(j), pwj.(j), Bitwise.bsl(16, wsh)}]
        end) ++
        [{:addi, xp, xp, 64}, {:addi, kc, kc, -16}, {:bnez, kc, {:l, {tag, :k}}}] ++
        Enum.flat_map(0..(rr - 1), fn j -> [{:vreduce16, fj.(j), acc.(j)}, {:stf, fj.(j), py, 4 * j}] end) ++
        mask_skip ++
        [{:arg, yb, 4}, {:shli, yb, yb, 2}, {:add, xrow, xrow, yb},
         {:arg, yb, 6}, {:shli, yb, yb, 2}, {:add, py, py, yb}] ++ mask_step ++
        [{:addi, bcnt, bcnt, -1}, {:bnez, bcnt, {:l, {tag, :b}}},
         {:label, {:l, {tag, :bdone}}}] ++
        # next block of weight rows: rr rows further in W and in y
        [{:arg, st, 4}, {:shli, st, st, wsh}] ++ Enum.map(1..rr, fn _ -> {:add, pw, pw, st} end) ++
        [{:addi, pyrow, pyrow, 4 * rr}, {:addi, cnt, cnt, -rr}]
    end

    main =
      if r == 1,
        do: [],
        else: [{:label, {:l, :blk}}, {:blt_imm, cnt, r, {:l, :tail}}] ++ block.(r, :blk) ++ [{:jmp, {:l, :blk}}]

    code =
      [{:arg, pyrow, 0}, {:arg, pw, 1}, {:arg, cnt, 3}] ++
        main ++
        [{:label, {:l, :tail}}, {:beqz, cnt, {:l, :done}}] ++
        block.(1, :one) ++
        [{:jmp, {:l, :tail}},
         {:label, {:l, :done}}, :ret]

    name = if wdt == :f32, do: :gemv_f32, else: :gemv_bf16
    {name, args} = if masked, do: {{:gemv_masked, wdt}, [:ptr, :ptr, :ptr, :n, :n, :n, :n, :ptr]}, else: {name, [:ptr, :ptr, :ptr, :n, :n, :n, :n]}
    %Vapor.KIR.Kernel{name: name, args: args, code: code, meta: %{rows_per_iter: r}}
  end

  @doc """
  Grouped (block-diagonal) GEMV — `Vapor.Algebra.Term.linear_grouped/3`:
  for every group `i < g` and activation row, `y[:, i·n …] = x[:, i·k …]·Wᵢᵀ`
  with `Wᵢ` rows `i·n … i·n + n − 1` of `W`. Every output is the canonical
  16-lane dot product of `linear`, so the bits are those of `g` separate
  `linear` calls. The group loop keeps its state (groups left, the group's
  x column) in a 16-byte scratch; threads split the groups.
  args: `[y, W, x, n, k, b, g, ldx, ldy, scratch]` (`ldx`, `ldy` in floats)
  """
  def gemv_grouped(wdt \\ :f32) when wdt in [:f32, :bf16],
    do: gemv_grouped(4, wdt) |> with_fallbacks([gemv_grouped(2, wdt), gemv_grouped(1, wdt)])

  def gemv_grouped(r, wdt) when r in [1, 2, 4] do
    wsh = if wdt == :f32, do: 2, else: 1
    [pw, cnt, pyrow, xrow, py, bcnt, t] = Enum.map(1..7, &gpr/1)
    pwj = fn j -> gpr(20 + j) end
    acc = fn j -> vec(30 + j, :f16l) end
    xv = vec(40, :f16l)
    wv = fn j -> vec(41 + j, :f16l) end
    fj = fn j -> fpr(50 + j) end

    block = fn rr, tag ->
      [st, xp, kc, yb] = Enum.map(0..3, &gpr(60 + 10 * tag_n(tag) + &1))

      [{:arg, yb, 9}, {:ld64, xrow, yb, 8}, {:mov, py, pyrow}, {:arg, bcnt, 5},
       {:beqz, bcnt, {:l, {tag, :bdone}}},
       {:label, {:l, {tag, :b}}},
       {:mov, pwj.(0), pw}, {:arg, st, 4}, {:shli, st, st, wsh}] ++
        Enum.map(1..(rr - 1)//1, fn j -> {:add, pwj.(j), pwj.(j - 1), st} end) ++
        Enum.map(0..(rr - 1), &{:vzero16, acc.(&1)}) ++
        [{:mov, xp, xrow}, {:arg, kc, 4}, {:label, {:l, {tag, :k}}}, {:vld, xv, xp, 0}] ++
        Enum.flat_map(0..(rr - 1), fn j ->
          load = if wdt == :f32, do: {:vld, wv.(j), pwj.(j), 0}, else: {:vld_bf16, wv.(j), pwj.(j), 0}
          [load, {:vfmacc, acc.(j), wv.(j), xv}, {:addi, pwj.(j), pwj.(j), Bitwise.bsl(16, wsh)}]
        end) ++
        [{:addi, xp, xp, 64}, {:addi, kc, kc, -16}, {:bnez, kc, {:l, {tag, :k}}}] ++
        Enum.flat_map(0..(rr - 1), fn j -> [{:vreduce16, fj.(j), acc.(j)}, {:stf, fj.(j), py, 4 * j}] end) ++
        [{:arg, yb, 7}, {:shli, yb, yb, 2}, {:add, xrow, xrow, yb},
         {:arg, yb, 8}, {:shli, yb, yb, 2}, {:add, py, py, yb},
         {:addi, bcnt, bcnt, -1}, {:bnez, bcnt, {:l, {tag, :b}}},
         {:label, {:l, {tag, :bdone}}}] ++
        [{:arg, st, 4}, {:shli, st, st, wsh}] ++ Enum.map(1..rr, fn _ -> {:add, pw, pw, st} end) ++
        [{:addi, pyrow, pyrow, 4 * rr}, {:addi, cnt, cnt, -rr}]
    end

    main =
      if r == 1,
        do: [],
        else: [{:label, {:l, :blk}}, {:blt_imm, cnt, r, {:l, :tail}}] ++ block.(r, :blk) ++ [{:jmp, {:l, :blk}}]

    # scratch: 0 groups left, 8 the group's first x column
    code =
      [{:arg, t, 9}, {:arg, cnt, 6}, {:st64, cnt, t, 0}, {:beqz, cnt, {:l, :done}},
       {:arg, cnt, 2}, {:st64, cnt, t, 8},
       {:arg, pyrow, 0}, {:arg, pw, 1},
       # after a group, y and W already point at the next group's first column
       # and row (n columns and n·k weights further)
       {:label, {:l, :group}},
       {:arg, cnt, 3}] ++
        main ++
        [{:label, {:l, :tail}}, {:beqz, cnt, {:l, :gnext}}] ++
        block.(1, :one) ++
        [{:jmp, {:l, :tail}},
         {:label, {:l, :gnext}},
         {:arg, t, 9}, {:ld64, cnt, t, 8}, {:arg, xrow, 4}, {:shli, xrow, xrow, 2}, {:add, cnt, cnt, xrow}, {:st64, cnt, t, 8},
         {:ld64, cnt, t, 0}, {:addi, cnt, cnt, -1}, {:st64, cnt, t, 0}, {:bnez, cnt, {:l, :group}},
         {:label, {:l, :done}}, :ret]

    %Vapor.KIR.Kernel{name: {:gemv_grouped, wdt}, args: [:ptr, :ptr, :ptr, :n, :n, :n, :n, :n, :n, :ptr], code: code,
                      meta: %{rows_per_iter: r}}
  end

  defp tag_n(:blk), do: 0
  defp tag_n(:one), do: 1

  @doc """
  Inline a `Vapor.Canon` function over vector registers of `kind`: operands
  are registers (a size-1 register is read as a broadcast) or constants.
  Temporaries are numbered from `base`. Returns `{code, result, next_base}`.
  """
  def inline(op, operands, kind, base) do
    {ops, res, _} = Vapor.Canon.expand(op, Enum.with_index(operands, fn _, i -> {:in, i} end), 0)
    ins = List.to_tuple(operands)
    reg = fn k -> vec(base + k, kind) end

    opnd = fn
      {:in, i} -> elem(ins, i)
      {:t, k} -> reg.(k)
      {:splat, b} -> {:const, b}
      {:imm, n} -> n
    end

    code = Enum.map(ops, fn {{:t, k}, prim, args} -> ew_inst(prim, reg.(k), Enum.map(args, opnd)) end)
    {:t, r} = res
    {code, reg.(r), base + length(ops)}
  end

  # a strip that copies `n` binary32 words from `src` to `dst` (advancing both)
  defp copy(n, src, dst, v), do: {:strip, n, [src, dst], [{:vld, v, src, 0}, {:vst, v, dst, 0}]}

  @doc """
  Rows of a table at unsigned indices clamped to the table.
  args: `[out, table, idx, T, V, d]`
  """
  def gather_row do
    [po, tab, pi, t, v, d, i, src, n, rb] = Enum.map(0..9, &gpr/1)

    code = [
      {:arg, po, 0}, {:arg, tab, 1}, {:arg, pi, 2}, {:arg, t, 3}, {:arg, v, 4}, {:arg, d, 5},
      {:shli, rb, d, 2},
      {:beqz, t, {:l, :done}},
      {:label, {:l, :row}},
      {:ld_u32, i, pi, 0},
      {:bltu, i, v, {:l, :ok}},
      {:addi, i, v, -1},
      {:label, {:l, :ok}},
      {:mul, src, i, rb},
      {:add, src, src, tab},
      {:mov, n, d},
      copy(n, src, po, vec(20, :strip)),
      {:addi, pi, pi, 4},
      {:addi, t, t, -1},
      {:bnez, t, {:l, :row}},
      {:label, {:l, :done}},
      :ret
    ]

    %Vapor.KIR.Kernel{name: :gather_row, args: [:ptr, :ptr, :ptr, :n, :n, :n], code: code}
  end

  @doc """
  Rows of a bfloat16 table, widened exactly, at clamped indices; `d ≡ 0
  (mod 16)`. args: `[out, table, idx, T, V, d]`
  """
  def gather_row_bf16 do
    [po, tab, pi, t, v, d, i, src, n, rb] = Enum.map(0..9, &gpr/1)
    x = vec(20, :f16l)

    code = [
      {:arg, po, 0}, {:arg, tab, 1}, {:arg, pi, 2}, {:arg, t, 3}, {:arg, v, 4}, {:arg, d, 5},
      {:shli, rb, d, 1},
      {:beqz, t, {:l, :done}},
      {:label, {:l, :row}},
      {:ld_u32, i, pi, 0},
      {:bltu, i, v, {:l, :ok}},
      {:addi, i, v, -1},
      {:label, {:l, :ok}},
      {:mul, src, i, rb},
      {:add, src, src, tab},
      {:mov, n, d},
      {:label, {:l, :chunk}},
      {:vld_bf16, x, src, 0},
      {:vst, x, po, 0},
      {:addi, src, src, 32},
      {:addi, po, po, 64},
      {:addi, n, n, -16},
      {:bnez, n, {:l, :chunk}},
      {:addi, pi, pi, 4},
      {:addi, t, t, -1},
      {:bnez, t, {:l, :row}},
      {:label, {:l, :done}},
      :ret
    ]

    %Vapor.KIR.Kernel{name: :gather_row_bf16, args: [:ptr, :ptr, :ptr, :n, :n, :n], code: code}
  end

  @doc """
  Rotary embedding (rotate-half): for each row `t` and head, with
  `c = cos[p], s = sin[p]`, `p = min(pos[t], S−1)`:
  `o₁ = x₁c − x₂s`, `o₂ = x₂c + x₁s` (each product rounded, then the sum).
  Row state is kept in a 64-byte scratch (explicit spill slots).
  args: `[out, x, cos, sin, pos, scratch, T, H, half, S]`
  """
  def rope do
    [sc, x, p, s, rb, pc, pss, po, px, hn, n, x1, x2, c, sn, o1, o2] = Enum.map(0..16, &gpr/1)
    [va, vb, vc, vs, m1, m2, r1, m3, m4, r2] = for i <- 30..39, do: vec(i, :strip)
    ld = fn r, off -> {:ld64, r, sc, off} end
    st = fn r, off -> {:st64, r, sc, off} end

    body = [
      {:vld, va, x1, 0}, {:vld, vb, x2, 0}, {:vld, vc, c, 0}, {:vld, vs, sn, 0},
      {:vfmul, m1, va, vc}, {:vfmul, m2, vb, vs}, {:vfsub, r1, m1, m2}, {:vst, r1, o1, 0},
      {:vfmul, m3, vb, vc}, {:vfmul, m4, va, vs}, {:vfadd, r2, m3, m4}, {:vst, r2, o2, 0}
    ]

    code = [
      # slots: 0 rows left, 8 pos ptr, 16 x ptr, 24 out ptr, 32 cos row, 40 sin row
      {:arg, sc, 5}, {:arg, x, 6}, st.(x, 0), {:beqz, x, {:l, :done}},
      {:arg, x, 4}, st.(x, 8), {:arg, x, 1}, st.(x, 16), {:arg, x, 0}, st.(x, 24),
      {:label, {:l, :row}},
      ld.(x, 8), {:ld_u32, p, x, 0}, {:addi, x, x, 4}, st.(x, 8),
      {:arg, s, 9}, {:bltu, p, s, {:l, :ok}}, {:addi, p, s, -1}, {:label, {:l, :ok}},
      {:arg, rb, 8}, {:shli, rb, rb, 2},
      {:mul, pc, p, rb}, {:arg, x, 3}, {:add, pss, pc, x}, {:arg, x, 2}, {:add, pc, pc, x},
      st.(pc, 32), st.(pss, 40),
      ld.(px, 16), ld.(po, 24), {:arg, hn, 7},
      {:label, {:l, :head}},
      {:arg, n, 8}, {:shli, x, n, 2},
      {:mov, x1, px}, {:add, x2, px, x}, {:mov, o1, po}, {:add, o2, po, x},
      ld.(c, 32), ld.(sn, 40),
      {:strip, n, [x1, x2, c, sn, o1, o2], body},
      # x2 and o2 now point one head further
      {:mov, px, x2}, {:mov, po, o2},
      {:addi, hn, hn, -1},
      {:bnez, hn, {:l, :head}},
      st.(px, 16), st.(po, 24),
      ld.(x, 0), {:addi, x, x, -1}, st.(x, 0),
      {:bnez, x, {:l, :row}},
      {:label, {:l, :done}},
      :ret
    ]

    %Vapor.KIR.Kernel{name: :rope, args: List.duplicate(:ptr, 6) ++ [:n, :n, :n, :n], code: code}
  end

  @doc """
  Functional row update of a cache. `:inplace` writes into the cache buffer
  itself (the lowering proved the old cache dead); `:copy` first copies it.
  args (`:inplace`): `[cache, pos, rows, T, S, n]`; (`:copy`): `[out, cache, pos, rows, T, S, n]`
  """
  def kv_write(mode) when mode in [:inplace, :copy] do
    [pout, pin, pp, pr, t, s, w, p, rb, dst, n] = Enum.map(0..10, &gpr/1)

    {args, prelude} =
      case mode do
        :inplace ->
          {[{:arg, pout, 0}, {:arg, pp, 1}, {:arg, pr, 2}, {:arg, t, 3}, {:arg, s, 4}, {:arg, w, 5}], []}

        :copy ->
          {[{:arg, pout, 0}, {:arg, pin, 1}, {:arg, pp, 2}, {:arg, pr, 3}, {:arg, t, 4}, {:arg, s, 5}, {:arg, w, 6}],
           [{:mul, n, s, w}, {:mov, dst, pout}, copy(n, pin, dst, vec(20, :strip))]}
      end

    code =
      args ++ prelude ++
        [{:shli, rb, w, 2},
         {:beqz, t, {:l, :done}},
         {:label, {:l, :row}},
         {:ld_u32, p, pp, 0},
         {:bltu, p, s, {:l, :write}},
         {:jmp, {:l, :next}},
         {:label, {:l, :write}},
         {:mul, dst, p, rb},
         {:add, dst, dst, pout},
         {:mov, n, w},
         {:mov, pin, pr},
         copy(n, pin, dst, vec(21, :strip)),
         {:label, {:l, :next}},
         {:add, pr, pr, rb},
         {:addi, pp, pp, 4},
         {:addi, t, t, -1},
         {:bnez, t, {:l, :row}},
         {:label, {:l, :done}},
         :ret]

    nargs = if mode == :inplace, do: 6, else: 7
    %Vapor.KIR.Kernel{name: {:kv_write, mode}, args: List.duplicate(:ptr, nargs - 3) ++ [:n, :n, :n], code: code}
  end

  @doc """
  Paged cache write: row `t` to logical row `pos[t]` of sequence `slot[t]`,
  i.e. pool row `table[slot, pos >> sh]·2^sh + pos mod 2^sh`; skipped when the
  slot, the page index or the page id is out of range (the oracle's rule).
  args: `[pool, table, slot, pos, rows, T, NS, MP, P, w]`
  """
  def kv_write_paged(sh) do
    [pool, ptab, psl, pp, pr, t, w, rb, sl, p, blk, pg, dst, n, tmp, src, lim] = Enum.map(0..16, &gpr/1)

    code =
      [{:arg, pool, 0}, {:arg, ptab, 1}, {:arg, psl, 2}, {:arg, pp, 3}, {:arg, pr, 4}, {:arg, t, 5}, {:arg, w, 9},
       {:shli, rb, w, 2},
       {:beqz, t, {:l, :done}},
       {:label, {:l, :row}},
       {:ld_u32, sl, psl, 0}, {:arg, lim, 6}, {:bltu, sl, lim, {:l, :s_ok}}, {:jmp, {:l, :next}}, {:label, {:l, :s_ok}},
       {:ld_u32, p, pp, 0}, {:shri, blk, p, sh}, {:arg, lim, 7}, {:bltu, blk, lim, {:l, :b_ok}}, {:jmp, {:l, :next}},
       {:label, {:l, :b_ok}},
       {:mul, tmp, sl, lim}, {:add, tmp, tmp, blk}, {:shli, tmp, tmp, 2}, {:add, tmp, tmp, ptab}, {:ld_u32, pg, tmp, 0},
       {:arg, lim, 8}, {:bltu, pg, lim, {:l, :p_ok}}, {:jmp, {:l, :next}}, {:label, {:l, :p_ok}},
       {:shli, pg, pg, sh}, {:shli, tmp, blk, sh}, {:sub, tmp, p, tmp}, {:add, pg, pg, tmp},
       {:mul, dst, pg, rb}, {:add, dst, dst, pool}, {:mov, n, w}, {:mov, src, pr},
       copy(n, src, dst, vec(21, :strip)),
       {:label, {:l, :next}},
       {:add, pr, pr, rb}, {:addi, pp, pp, 4}, {:addi, psl, psl, 4}, {:addi, t, t, -1}, {:bnez, t, {:l, :row}},
       {:label, {:l, :done}},
       :ret]

    %Vapor.KIR.Kernel{name: {:kv_write_paged, sh}, args: List.duplicate(:ptr, 5) ++ List.duplicate(:n, 5), code: code}
  end

  @doc """
  Causal grouped-query attention in the canonical order of
  `Vapor.Runtime.Oracle.attend/3`, one (row, head) at a time. Outer loop
  state lives in the first 128 bytes of the scratch buffer (explicit spill
  slots — the portable code never spills implicitly); the scores follow.
  args: `[out, q, k, v, pos, scratch, T, HKV, G, S, DH, kvw]` (`kvw` the
  cache row width in floats; paged: `table, slot, NS, MP, P` before it)
  """
  def attention(scale_bits, layout \\ :contig) do
    # the cache row width (floats) is the last argument: a thread given a
    # range of kv heads still walks whole cache rows
    kvw_arg = if layout == :contig, do: 11, else: 16
    # the sliding window (0: none) is the argument after it
    win_arg = kvw_arg + 1

    [sc, x, pp, p, s, l, l16, pk, ps, j, dh, rs, qh, pq, pkk, cn, npad, n, off, po, pvb, pv, hb] =
      Enum.map(0..22, &gpr/1)

    # short-lived temporaries of the per-token prologue (window bounds)
    [w0, w1] = [gpr(90), gpr(91)]

    [f, fs, fneg, m, z, pj, fz, invf] = for i <- 30..37, do: fpr(i)
    [acc, qv, mx, t, sm, oa, vv, tv] = for i <- 40..47, do: vec(i, :f16l)
    [ts, e2] = [vec(48, :strip), vec(49, :strip)]
    neg_max = 0xFF7F_FFFF
    {exp_code, e, nb} = inline(:exp, [ts], :strip, 100)
    {rcp_code, inv, _} = inline(:rcp, [z], :f16l, nb)

    ld = fn r, off -> {:ld64, r, sc, off} end
    st = fn r, off -> {:st64, r, sc, off} end
    dec = fn off, l -> [ld.(x, off), {:addi, x, x, -1}, st.(x, off), {:bnez, x, l}] end

    chunks = fn label, inner ->
      [{:addi, ps, sc, 128}, ld.(n, 72), {:label, label}] ++ inner ++
        [{:addi, ps, ps, 64}, {:addi, n, n, -16}, {:bnez, n, label}]
    end

    # Paged layout (`{:paged, sh}`, page = 2^sh rows): the key/value pointer
    # walks a page at a time — at each page boundary the next page id is
    # read from the sequence's table row and the pointer jumps to that page
    # of the pool. Keys are visited in the same logical order, so the
    # arithmetic is the contiguous kernel's. The walk's state (keys left in
    # the page at scratch 96, table cursor at 104) lives in the scratch
    # block, so the hot loops keep the contiguous kernel's register set.
    walk = fn tag, ptr, base_code, base ->
      # a fresh register per site: live ranges are hulls, and a register
      # shared between the page switch and the step would span the loop
      [c0, t0, cnt, tb, pgr, lim, c1] = Enum.map(base..(base + 6), &gpr/1)

      case layout do
        # (contiguous: the window's first row is folded into the k/v bases)
        :contig -> {[], [], []}
        {:paged, sh} ->
          # the window's first page (table cursor) and its row within that
          # page (scratch 112, consumed by the first page switch)
          {[ld.(t0, 120), {:shri, c0, t0, sh}, {:shli, c0, c0, 2}, ld.(t0, 80), {:add, t0, t0, c0}, st.(t0, 104),
            ld.(t0, 120), {:shri, c0, t0, sh}, {:shli, c0, c0, sh}, {:sub, t0, t0, c0}, st.(t0, 112),
            {:li, c0, 0}, st.(c0, 96)],
           [ld.(cnt, 96), {:bnez, cnt, {:l, {tag, :have}}},
            ld.(tb, 104), {:ld_u32, pgr, tb, 0}, {:addi, tb, tb, 4}, st.(tb, 104),
            {:arg, lim, 15}, {:bltu, pgr, lim, {:l, {tag, :pg_ok}}}, {:addi, pgr, lim, -1}, {:label, {:l, {tag, :pg_ok}}},
            {:shli, lim, rs, sh}, {:mul, pgr, pgr, lim}] ++ base_code ++ [{:add, ptr, ptr, pgr},
            {:li, tb, Bitwise.bsl(1, sh)},
            ld.(lim, 112), {:sub, tb, tb, lim}, {:mul, lim, lim, rs}, {:add, ptr, ptr, lim}, {:li, lim, 0}, st.(lim, 112),
            st.(tb, 96), {:label, {:l, {tag, :have}}}],
           [ld.(c1, 96), {:addi, c1, c1, -1}, st.(c1, 96)]}
      end
    end

    # this row's table row: table + min(slot, NS−1)·MP·4 (slot pointer at 88)
    slot_row =
      case layout do
        :contig -> []
        {:paged, _} ->
          [sr, st2] = [gpr(80), gpr(81)]

          [ld.(x, 88), {:ld_u32, sr, x, 0}, {:addi, x, x, 4}, st.(x, 88),
           {:arg, st2, 13}, {:bltu, sr, st2, {:l, :slot_ok}}, {:addi, sr, st2, -1}, {:label, {:l, :slot_ok}},
           {:arg, st2, 14}, {:mul, sr, sr, st2}, {:shli, sr, sr, 2}, {:arg, st2, 11}, {:add, sr, sr, st2}, st.(sr, 80)]
      end

    # scores walk from the kv head's base in the pool (scratch 40); the
    # weighted sum from the current output chunk's base (pvb)
    {sj_pre, sj_page, sj_step} = walk.(:sj, pk, [ld.(pk, 40)], 60)
    {oj_pre, oj_page, oj_step} = walk.(:oj, pv, [{:mov, pv, pvb}], 72)

    code =
      [{:arg, sc, 5}, {:arg, x, 6}, st.(x, 0), {:beqz, x, {:l, :done}},
       {:arg, x, 4}, st.(x, 8), {:arg, x, 1}, st.(x, 16), {:arg, x, 0}, st.(x, 24)] ++
        (if layout == :contig, do: [], else: [{:arg, x, 12}, st.(x, 88)]) ++
      [
       {:label, {:l, :tok}},
       ld.(pp, 8), {:ld_u32, p, pp, 0}, {:addi, pp, pp, 4}, st.(pp, 8),
       {:arg, s, 9}, {:bltu, p, s, {:l, :pos_ok}}, {:addi, p, s, -1}, {:label, {:l, :pos_ok}},
       {:addi, l, p, 1},
       # sliding window: L = min(p + 1, w) keys from row p + 1 − L (scratch 120)
       {:li, w0, 0}, {:arg, w1, win_arg}, {:beqz, w1, {:l, :win_done}}, {:bltu, w1, l, {:l, :slide}}, {:jmp, {:l, :win_done}},
       {:label, {:l, :slide}}, {:sub, w0, l, w1}, {:mov, l, w1},
       {:label, {:l, :win_done}}, st.(w0, 120), st.(l, 64),
       {:addi, l16, l, 15}, {:shri, l16, l16, 4}, {:shli, l16, l16, 4}, st.(l16, 72)] ++
        slot_row ++
       [
       {:arg, x, 7}, st.(x, 32)] ++
        (case layout do
           # contiguous: k and v bases start at the window's first row
           :contig ->
             [ld.(w0, 120), {:arg, w1, kvw_arg}, {:shli, w1, w1, 2}, {:mul, w0, w0, w1},
              {:arg, w1, 2}, {:add, w1, w1, w0}, st.(w1, 40), {:arg, w1, 3}, {:add, w1, w1, w0}, st.(w1, 48)]

           {:paged, _} ->
             [{:arg, x, 2}, st.(x, 40), {:arg, x, 3}, st.(x, 48)]
         end) ++
       [
       {:label, {:l, :kv}},
       {:arg, x, 8}, st.(x, 56),
       {:label, {:l, :grp}},
       # ---- scores s_j = (q · k_j)·scale, j < L ----
       ld.(pk, 40), {:addi, ps, sc, 128}, ld.(j, 64),
       {:arg, dh, 10}, {:arg, rs, kvw_arg}, {:shli, rs, rs, 2},
       ld.(qh, 16), {:lif, fs, scale_bits}] ++ sj_pre ++
      [{:label, {:l, :sj}}] ++ sj_page ++
      [{:vzero16, acc}, {:mov, pq, qh}, {:mov, pkk, pk}, {:mov, cn, dh},
       {:label, {:l, :sc}},
       {:vld, qv, pq, 0}, {:vfmacc_mem, acc, qv, pkk, 0},
       {:addi, pq, pq, 64}, {:addi, pkk, pkk, 64}, {:addi, cn, cn, -16}, {:bnez, cn, {:l, :sc}},
       {:vreduce16, f, acc}, {:fmul, f, f, fs}, {:stf, f, ps, 0},
       {:addi, ps, ps, 4}, {:add, pk, pk, rs}] ++ sj_step ++ [{:addi, j, j, -1}, {:bnez, j, {:l, :sj}},
       # ---- pad to a multiple of 16 with −FLT_MAX ----
       ld.(l16, 72), ld.(l, 64), {:sub, npad, l16, l}, {:beqz, npad, {:l, :padded}},
       {:lif, fneg, neg_max},
       {:label, {:l, :pad}}, {:stf, fneg, ps, 0}, {:addi, ps, ps, 4}, {:addi, npad, npad, -1}, {:bnez, npad, {:l, :pad}},
       {:label, {:l, :padded}},
       # ---- m = canonical max ----
       {:vconst, mx, neg_max}] ++
        chunks.({:l, :mx}, [{:vld, t, ps, 0}, {:vsel_lt, mx, mx, t, t, mx}]) ++
        [{:vreduce16_max, m, mx}, {:bcast, m}] ++
        # ---- e_j = exp(s_j − m): a strip, so every ISA picks its own width ----
        [{:addi, ps, sc, 128}, ld.(n, 72),
         {:strip, n, [ps], [{:vld, ts, ps, 0}, {:vfsub, ts, ts, m}] ++ exp_code ++ [{:vst, e, ps, 0}]}] ++
        # ---- e = +0 on the padding ----
        [ld.(l, 64), {:shli, off, l, 2}, {:add, ps, sc, off}, {:addi, ps, ps, 128},
         ld.(l16, 72), {:sub, npad, l16, l}, {:beqz, npad, {:l, :zeroed}}, {:lif, fz, 0},
         {:label, {:l, :zp}}, {:stf, fz, ps, 0}, {:addi, ps, ps, 4}, {:addi, npad, npad, -1}, {:bnez, npad, {:l, :zp}},
         {:label, {:l, :zeroed}},
         # ---- Z = canonical sum, p_j = e_j · rcp(Z) ----
         {:vzero16, sm}] ++
        chunks.({:l, :sum}, [{:vld, t, ps, 0}, {:vfadd, sm, sm, t}]) ++
        [{:vreduce16, z, sm}, {:bcast, z}] ++ rcp_code ++
        [{:vlane0, invf, inv}, {:addi, ps, sc, 128}, ld.(n, 72),
         {:strip, n, [ps], [{:vld, e2, ps, 0}, {:vfmul, e2, e2, invf}, {:vst, e2, ps, 0}]}] ++
        # ---- o[d] = Σ_j p_j·v_j[d], sequential in j ----
        [ld.(po, 24), ld.(pvb, 48), {:mov, cn, dh},
         {:label, {:l, :oc}},
         {:vzero16, oa}, {:addi, ps, sc, 128}, {:mov, pv, pvb}, ld.(j, 64)] ++ oj_pre ++
        [{:label, {:l, :oj}}] ++ oj_page ++
        [{:ldf, pj, ps, 0}, {:vld, vv, pv, 0}, {:vmul_sf, tv, vv, pj}, {:vfadd, oa, oa, tv},
         {:addi, ps, ps, 4}, {:add, pv, pv, rs}] ++ oj_step ++ [{:addi, j, j, -1}, {:bnez, j, {:l, :oj}},
         {:vst, oa, po, 0}, {:addi, po, po, 64}, {:addi, pvb, pvb, 64}, {:addi, cn, cn, -16}, {:bnez, cn, {:l, :oc}},
         # next head of the group: q and out advance by one head
         st.(po, 24), {:shli, hb, dh, 2}, ld.(x, 16), {:add, x, x, hb}, st.(x, 16)] ++
        dec.(56, {:l, :grp}) ++
        # next kv head
        [{:arg, dh, 10}, {:shli, hb, dh, 2},
         ld.(x, 40), {:add, x, x, hb}, st.(x, 40), ld.(x, 48), {:add, x, x, hb}, st.(x, 48)] ++
        dec.(32, {:l, :kv}) ++
        dec.(0, {:l, :tok}) ++
        [{:label, {:l, :done}}, :ret]

    args =
      case layout do
        :contig -> List.duplicate(:ptr, 6) ++ List.duplicate(:n, 5) ++ [:n, :n]
        {:paged, _} -> List.duplicate(:ptr, 6) ++ List.duplicate(:n, 5) ++ [:ptr, :ptr, :n, :n, :n, :n, :n]
      end

    name = if layout == :contig, do: {:attention, scale_bits}, else: {:attention_paged, scale_bits, layout}
    %Vapor.KIR.Kernel{name: name, args: args, code: code}
  end

  @doc """
  Categorical sampling on the substrate, one row of logits per output, in
  `Vapor.Runtime.Oracle`'s order: `m` = canonical max; with `invT = +0`
  the first index whose bits equal `m` (greedy); otherwise
  `e_i = exp((l_i − m)·invT)` (canonical exp, stored in the scratch row),
  `Σe` sequential in index order, `τ = u·Σe`, and the first index whose
  running sum exceeds `τ` (else the last with `e_i > 0`). Running sums and
  `τ` are non-negative, so they compare as unsigned bit patterns.
  args: `[out, logits, params(invT, u), scratch, B, V]`
  """
  def sample do
    [po, pl, pp, sc, b, k, pe, ps, j, cand, bits, last, n, tmp, tb, cb, rb] = Enum.map(0..16, &gpr/1)
    [m, it, tot, f, uu, tgt, c] = for i <- 30..36, do: fpr(i)
    [acc, t] = [vec(40, :f16l), vec(41, :f16l)]
    ts = vec(42, :strip)
    {exp_code, e, _} = inline(:exp, [ts], :strip, 100)

    code =
      [{:arg, po, 0}, {:arg, pl, 1}, {:arg, pp, 2}, {:arg, b, 4},
       {:beqz, b, {:l, :done}},
       {:label, {:l, :row}},
       # canonical max over the row
       {:vconst, acc, 0xFF7F_FFFF}, {:mov, pe, pl}, {:arg, k, 5},
       {:label, {:l, :mx}},
       {:vld, t, pe, 0}, {:vsel_lt, acc, acc, t, t, acc}, {:addi, pe, pe, 64}, {:addi, k, k, -16}, {:bnez, k, {:l, :mx}},
       {:vreduce16_max, m, acc},
       {:arg, sc, 3}, {:ld_u32, tb, pp, 0}, {:bnez, tb, {:l, :sampled}},
       # greedy: the first index holding exactly m
       {:stf, m, sc, 0}, {:ld_u32, cand, sc, 0}, {:mov, pe, pl}, {:li, j, 0},
       {:label, {:l, :am}},
       {:ld_u32, bits, pe, 0}, {:sub, bits, bits, cand}, {:beqz, bits, {:l, :pick}},
       {:addi, pe, pe, 4}, {:addi, j, j, 1}, {:jmp, {:l, :am}},
       {:label, {:l, :sampled}},
       {:ldf, it, pp, 0}, {:bcast, m}, {:bcast, it},
       {:mov, pe, pl}, {:mov, ps, sc}, {:arg, n, 5},
       {:strip, n, [pe, ps], [{:vld, ts, pe, 0}, {:vfsub, ts, ts, m}, {:vfmul, ts, ts, it}] ++ exp_code ++ [{:vst, e, ps, 0}]},
       # Σe in index order
       {:lif, tot, 0}, {:mov, ps, sc}, {:arg, k, 5},
       {:label, {:l, :sum}},
       {:ldf, f, ps, 0}, {:fadd, tot, tot, f}, {:addi, ps, ps, 4}, {:addi, k, k, -1}, {:bnez, k, {:l, :sum}},
       {:ldf, uu, pp, 4}, {:fmul, tgt, uu, tot},
       # τ's bits, kept in the scratch slot after the row
       {:arg, rb, 5}, {:shli, rb, rb, 2}, {:add, tmp, sc, rb}, {:stf, tgt, tmp, 0}, {:ld_u32, tb, tmp, 0},
       {:lif, c, 0}, {:mov, ps, sc}, {:li, j, 0}, {:li, last, 0}, {:arg, k, 5},
       {:label, {:l, :scan}},
       {:ldf, f, ps, 0}, {:fadd, c, c, f},
       {:ld_u32, bits, ps, 0}, {:beqz, bits, {:l, :zero}}, {:mov, last, j}, {:label, {:l, :zero}},
       {:stf, c, tmp, 0}, {:ld_u32, cb, tmp, 0}, {:bltu, tb, cb, {:l, :pick}},
       {:addi, ps, ps, 4}, {:addi, j, j, 1}, {:addi, k, k, -1}, {:bnez, k, {:l, :scan}},
       {:mov, j, last},
       {:label, {:l, :pick}},
       {:st_i32, j, po, 0},
       {:addi, po, po, 4}, {:addi, pp, pp, 8}, {:arg, rb, 5}, {:shli, rb, rb, 2}, {:add, pl, pl, rb},
       {:addi, b, b, -1}, {:bnez, b, {:l, :row}},
       {:label, {:l, :done}},
       :ret]

    %Vapor.KIR.Kernel{name: :sample, args: [:ptr, :ptr, :ptr, :ptr, :n, :n], code: code}
  end

  @doc """
  out[j, i] = x[i, j] for x : f32[R, C] (output rows `ldo` floats apart, so
  a thread can own a range of input rows). args: `[out, x, R, C, ldo]`
  """
  def transpose do
    [po, px, r, c, w, ldo, pd] = Enum.map(0..6, &gpr/1)

    code =
      [{:arg, po, 0}, {:arg, px, 1}, {:arg, r, 2},
       {:beqz, r, {:l, :done}},
       {:label, {:l, :row}},
       {:arg, c, 3}, {:arg, ldo, 4}, {:shli, ldo, ldo, 2}, {:mov, pd, po},
       {:label, {:l, :col}},
       {:ld_u32, w, px, 0}, {:st_i32, w, pd, 0},
       {:addi, px, px, 4}, {:add, pd, pd, ldo}, {:addi, c, c, -1}, {:bnez, c, {:l, :col}},
       {:addi, po, po, 4}, {:addi, r, r, -1}, {:bnez, r, {:l, :row}},
       {:label, {:l, :done}},
       :ret]

    %Vapor.KIR.Kernel{name: :transpose, args: [:ptr, :ptr, :n, :n, :n], code: code}
  end

  @doc "Sub-block activation sums X_s (canonical order). args: `[X, x, nsub]`"
  def sb_sums do
    [px_out, px, n] = [gpr(0), gpr(1), gpr(2)]
    [lo, hi, t] = [vec(3, :f16l), vec(4, :f16l), vec(5, :f16l)]
    f = fpr(6)

    code = [
      {:arg, px_out, 0},
      {:arg, px, 1},
      {:arg, n, 2},
      {:beqz, n, {:l, :done}},
      {:label, {:l, :loop}},
      {:vld16, lo, px, 0},
      {:vld16, hi, px, 64},
      {:vfadd16, t, lo, hi},
      {:vreduce16, f, t},
      {:stf, f, px_out, 0},
      {:addi, px, px, 128},
      {:addi, px_out, px_out, 4},
      {:addi, n, n, -1},
      {:bnez, n, {:l, :loop}},
      {:label, {:l, :done}},
      :ret
    ]

    %Vapor.KIR.Kernel{name: :sb_sums, args: [:ptr, :ptr, :n], code: code}
  end

  @doc """
  Quantized matrix product over `:sb4x` rows in the canonical order
  (`Vapor.Runtime.Oracle`): `y[i] = W · x[i]` for `b` activation rows.
  args: `[y, W, x, X, rows, nsb, b, ldy]` (`X` holds the sub-block sums of
  all `b` rows, `sb_sums` over `b·k/32` sub-blocks; output rows are `ldy`
  floats apart, so a thread can own a column range `rows < ldy`).

  Each output is one weight row against one activation row, in the same
  instructions and order whatever `b` — so the result of a row does not
  depend on how many rows are processed with it (batch invariance). The
  loop order is weight-stationary: a block of weight rows meets every
  activation row before the next block is read.

  The canonical order fixes 16 accumulation lanes *per row*, so a single
  row is a latency-bound dependency chain. Parallelism is therefore taken
  from independent rows: `r` rows are processed per iteration (each with its
  own accumulators, so every row's arithmetic — and result — is unchanged),
  with the 8 sub-blocks of a superblock unrolled so that every weight, scale
  and activation operand is an immediate offset from one pointer per row.
  The primary kernel uses `r = 4`; `meta.fallbacks` holds `r = 2, 1`, and
  code generation keeps the first variant the allocator can place on the
  target — the same allocation-driven choice as the LMUL factor.
  """
  def gemv_sb4, do: gemv_sb4(4) |> with_fallbacks([gemv_sb4(2), gemv_sb4(1)])

  @doc """
  Row-predicated `gemv_sb4` (`Vapor.Algebra.Term.qgemv_masked/3`): an
  activation row whose mask word is ±0 stores `+0` and reads no weight
  superblock; active rows run the dense kernel's very instructions, hence
  its bits. This is what lets a 4-bit mixture of experts read only its
  top-k experts' weights. args: `[y, W, x, X, rows, nsb, b, ldy, mask]`.
  """
  def gemv_sb4_masked, do: gemv_sb4(4, true) |> with_fallbacks([gemv_sb4(2, true), gemv_sb4(1, true)])

  def gemv_sb4(r, masked \\ false) when r in [1, 2, 4] do
    # Weight-stationary, like gemv_f32: each block of r weight rows meets
    # every activation row before the next block is read.
    [pwb, rows, xrow, xsrow, py, bcnt, k, px, pxs] = Enum.map(0..8, &gpr/1)
    pw = fn j -> gpr(20 + j) end
    acc = fn j -> vec(30 + j, :f16l) end
    yb = fn j -> fpr(40 + j) end
    tmp = fn id, i, kind -> {:vr, 1000 + 40 * id + i, kind} end

    block = fn rr, tag ->
      # fresh registers per site: live ranges are hulls
      [st, a0, a1, a2, a3, a4] = Enum.map(0..5, &gpr(60 + 10 * tag_id(tag) + &1))
      [pm, mk] = [gpr(66 + 10 * tag_id(tag)), gpr(67 + 10 * tag_id(tag))]
      zf = fpr(46 + tag_id(tag))

      # masked: the row's mask word, sign bit shifted out; ±0 stores +0 for
      # the block's rr outputs and skips every weight read
      {mask_init, mask_test, mask_skip, mask_step} =
        if masked do
          {[{:arg, pm, 8}],
           [{:ld_u32, mk, pm, 0}, {:shli, mk, mk, 33}, {:beqz, mk, {:l, {tag, :skip}}}],
           [{:jmp, {:l, {tag, :adv}}}, {:label, {:l, {tag, :skip}}}, {:lif, zf, 0}] ++
             Enum.map(0..(rr - 1), &{:stf, zf, py, 4 * &1}) ++ [{:label, {:l, {tag, :adv}}}],
           [{:addi, pm, pm, 4}]}
        else
          {[], [], [], []}
        end

      # one activation row against `rr` weight rows; temporaries are fresh
      # per (sub-block, row)
      init =
        [{:mov, pw.(0), pwb}, {:arg, st, 5}, {:li, a0, 152}, {:mul, st, st, a0}] ++
          Enum.flat_map(0..(rr - 1), fn j -> [{:vzero16, acc.(j)}, {:lif, yb.(j), 0}] end) ++
          Enum.map(1..(rr - 1)//1, fn j -> {:add, pw.(j), pw.(j - 1), st} end) ++
          [{:mov, px, xrow}, {:mov, pxs, xsrow}, {:arg, k, 5}, {:label, {:l, {tag, :sb}}}]

      body =
        for sub <- 0..7, j <- 0..(rr - 1) do
          id = (tag_id(tag) * 8 + sub) * 4 + j
          [xu, xv] = [tmp.(id, 0, :gpr), tmp.(id, 1, :gpr)]
          [fu, fv, c1, c0, alpha, t, beta, fx] = for i <- 2..9, do: tmp.(id, i, :fpr)
          [qlo, qhi, wlo, whi] = for i <- 10..13, do: tmp.(id, i, :f16l)

          [{:ld_u8, xu, pw.(j), 128 + sub}, {:ld_u8, xv, pw.(j), 136 + sub},
           {:cvt_u8f, fu, xu}, {:cvt_u8f, fv, xv},
           {:ldf, c1, pw.(j), 144}, {:ldf, c0, pw.(j), 148},
           {:fmul, alpha, c1, fu}, {:fmul, t, alpha, fv}, {:fsub, beta, c0, t},
           {:vld_nib, qlo, qhi, pw.(j), 16 * sub},
           {:vmul_sf, wlo, qlo, alpha}, {:vmul_sf, whi, qhi, alpha},
           {:vfmacc_mem, acc.(j), wlo, px, 128 * sub}, {:vfmacc_mem, acc.(j), whi, px, 128 * sub + 64},
           {:ldf, fx, pxs, 4 * sub}, {:fmacc, yb.(j), beta, fx}]
        end
        |> List.flatten()

      step =
        Enum.map(0..(rr - 1), fn j -> {:addi, pw.(j), pw.(j), 152} end) ++
          [{:addi, px, px, 1024}, {:addi, pxs, pxs, 32}, {:addi, k, k, -1}, {:bnez, k, {:l, {tag, :sb}}}]

      finish =
        Enum.flat_map(0..(rr - 1), fn j ->
          f = tmp.(tag_id(tag) * 64 + 60 + j, 0, :fpr)
          [{:vreduce16, f, acc.(j)}, {:fadd, f, f, yb.(j)}, {:stf, f, py, 4 * j}]
        end)

      # every activation row (x, X and y advance), then the next block
      [{:arg, xrow, 2}, {:arg, xsrow, 3}, {:arg, bcnt, 6}] ++ mask_init ++
       [{:beqz, bcnt, {:l, {tag, :bdone}}},
       {:label, {:l, {tag, :b}}}] ++ mask_test ++
        init ++ body ++ step ++ finish ++ mask_skip ++
        [{:arg, a1, 5}, {:li, a2, 1024}, {:mul, a2, a2, a1}, {:add, xrow, xrow, a2},
         {:li, a2, 32}, {:mul, a2, a2, a1}, {:add, xsrow, xsrow, a2},
         {:arg, a2, 7}, {:shli, a2, a2, 2}, {:add, py, py, a2}] ++ mask_step ++
        [{:addi, bcnt, bcnt, -1}, {:bnez, bcnt, {:l, {tag, :b}}},
         {:label, {:l, {tag, :bdone}}},
         # back to this block's first column (b rows of ldy down), then rr on
         {:arg, a3, 6}, {:arg, a4, 7}, {:mul, a3, a3, a4}, {:shli, a3, a3, 2}, {:sub, py, py, a3},
         {:addi, py, py, 4 * rr},
         {:arg, a3, 5}, {:li, a4, 152 * rr}, {:mul, a3, a3, a4}, {:add, pwb, pwb, a3},
         {:addi, rows, rows, -rr}]
    end

    main =
      if r == 1,
        do: [],
        else: [{:label, {:l, :blk}}, {:blt_imm, rows, r, {:l, :tail}}] ++ block.(r, :blk) ++ [{:jmp, {:l, :blk}}]

    code =
      [{:arg, py, 0}, {:arg, pwb, 1}, {:arg, rows, 4}] ++
        main ++
        [{:label, {:l, :tail}}, {:beqz, rows, {:l, :done}}] ++
        block.(1, :one) ++
        [{:jmp, {:l, :tail}},
         {:label, {:l, :done}}, :ret]

    {name, args} =
      if masked, do: {:gemv_sb4_masked, [:ptr, :ptr, :ptr, :ptr, :n, :n, :n, :n, :ptr]},
                 else: {:gemv_sb4, [:ptr, :ptr, :ptr, :ptr, :n, :n, :n, :n]}

    %Vapor.KIR.Kernel{name: name, args: args, code: code, meta: %{rows_per_iter: r}}
  end

  defp tag_id(:blk), do: 0
  defp tag_id(:one), do: 1

  defp with_fallbacks(k, fbs), do: %{k | meta: Map.put(k.meta, :fallbacks, fbs)}

  @doc "C[M,N] = A[M,K] · W[N,K]ᵀ over (ℤ/2³²ℤ, +, ×). args: `[C, A, W, M, N, K]`"
  def gemm_i8 do
    [pc, pa0, w0, m, n0, kk0, pw, j, sacc, pa, pb, kk, ta, tb, r] = Enum.map(0..14, &gpr/1)
    acc = vec(15, :i32acc)

    code = [
      {:arg, pc, 0},
      {:arg, pa0, 1},
      {:arg, w0, 2},
      {:arg, m, 3},
      {:arg, n0, 4},
      {:arg, kk0, 5},
      {:beqz, m, {:l, :done}},
      {:beqz, n0, {:l, :done}},
      {:label, {:l, :i}},
      {:mov, pw, w0},
      {:mov, j, n0},
      {:label, {:l, :j}},
      {:vzero_i32, acc},
      {:li, sacc, 0},
      {:mov, pa, pa0},
      {:mov, pb, pw},
      {:mov, kk, kk0},
      {:strip_i8, kk, [pa, pb], [{:vi8mac, acc, pa, pb}],
       [
         {:ld_s8, ta, pa, 0},
         {:ld_s8, tb, pb, 0},
         {:mul, ta, ta, tb},
         {:add, sacc, sacc, ta}
       ]},
      {:vred_i32, r, acc},
      {:add, r, r, sacc},
      {:st_i32, r, pc, 0},
      {:addi, pc, pc, 4},
      {:add, pw, pw, kk0},
      {:addi, j, j, -1},
      {:bnez, j, {:l, :j}},
      {:add, pa0, pa0, kk0},
      {:addi, m, m, -1},
      {:bnez, m, {:l, :i}},
      {:label, {:l, :done}},
      :ret
    ]

    %Vapor.KIR.Kernel{name: :gemm_i8, args: [:ptr, :ptr, :ptr, :n, :n, :n], code: code}
  end
end
