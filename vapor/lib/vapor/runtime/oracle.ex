defmodule Vapor.Runtime.Oracle do
  @moduledoc """
  The declared semantics of every operator, executed in exact binary32
  (`Vapor.F32`) and exact integers. This is the reference the whole ladder
  compares against — and the last-resort substrate: it cannot crash the node
  and never needs hardware.

  ## Float policies (a certified property, recorded in the certificate)

    * `:canonical` — every multiply and add is rounded separately (no
      contraction), reductions follow the fixed 16-lane tree below. All
      substrates reproduce it **bit for bit**; parity is a digest equality.
    * `:fast` — fused multiply-add where the operator allows it. Substrates
      are certified only up to the Wilkinson envelope.

  ## The canonical 16-lane reduction (the only reduction order in the system)

      r8[i] = l[i] + l[i+8]   r4[i] = r8[i] + r8[i+4]   r2[i] = r4[i] + r4[i+2]   r = r2[0] + r2[1]
  """
  import Bitwise
  alias Vapor.{F32, Tensor}
  alias Vapor.Algebra.Term
  alias Vapor.Quant.Sb4

  @type policy :: :canonical | :fast

  @neg_max 0xFF7F_FFFF

  @doc "Evaluate a list of root terms under `env` (input name → Tensor)."
  @spec eval_many([term], map, policy) :: [Tensor.t()]
  def eval_many(roots, env, policy \\ :canonical) do
    memo =
      roots
      |> Term.postorder()
      |> Enum.reduce(%{}, fn t, memo -> Map.put(memo, t, node(t, memo, env, policy)) end)

    Enum.map(roots, &Map.fetch!(memo, &1))
  end

  def eval(root, env \\ %{}, policy \\ :canonical), do: hd(eval_many([root], env, policy))

  @doc "Evaluate a whole program (let-bindings included): `%{output => Tensor}`."
  def eval_program(%Vapor.Program{} = p, env, policy \\ :canonical),
    do: Vapor.Program.evaluate(p, env, &Tensor.widen/1, &node(&1, &2, &3, policy))

  defp node({:input, name, dt, _s}, _m, env, _p) do
    case Map.fetch(env, name) do
      {:ok, %Tensor{dtype: ^dt} = t} -> t
      # a bf16 binding is held as the f32 values it denotes
      {:ok, %Tensor{dtype: :f32} = t} when dt == :bf16 -> t
      _ -> raise ArgumentError, "unbound input #{inspect(name)} : #{dt}"
    end
  end

  defp node({:const, t}, _m, _env, _p), do: Tensor.widen(t)

  defp node({:ew, op, args} = t, m, _env, p) do
    {:ok, {:f32, shape}} = Term.infer(t)
    shape = concrete(shape, Enum.map(Term.children(t), &Map.fetch!(m, &1).shape))
    n = Enum.product(shape)
    f = scalar_fun(op, p)

    cols =
      Enum.map(args, fn
        {:splat, b} -> List.duplicate(b, n)
        a -> m |> Map.fetch!(a) |> broadcast(shape)
      end)

    Tensor.new(:f32, shape, cols |> Enum.zip_with(f) |> F32.encode())
  end

  defp node({:reduce, op, x}, m, _env, _p) do
    %Tensor{shape: shape} = xt = Map.fetch!(m, x)
    n = List.last(shape)

    out =
      xt.data
      |> F32.decode()
      |> Enum.chunk_every(n)
      |> Enum.map(&reduce_row(op, &1))

    Tensor.new(:f32, List.replace_at(shape, -1, 1), F32.encode(out))
  end

  defp node({:linear, x, w}, m, _env, p) do
    %Tensor{shape: [n, k]} = wt = Map.fetch!(m, w)
    %Tensor{shape: xs} = xt = Map.fetch!(m, x)
    wrows = wt.data |> F32.decode() |> Enum.chunk_every(k) |> Enum.map(&List.to_tuple/1)

    out =
      xt.data
      |> F32.decode()
      |> Enum.chunk_every(k)
      |> Enum.flat_map(fn xrow -> xr = List.to_tuple(xrow); Enum.map(wrows, &dot16(&1, xr, k, p)) end)

    Tensor.new(:f32, List.replace_at(xs, -1, n), F32.encode(out))
  end

  # row-predicated linear: inactive rows (mask ±0) are +0 and never computed
  defp node({:linear_masked, x, w, mk}, m, _env, p) do
    %Tensor{shape: [n, k]} = wt = Map.fetch!(m, w)
    %Tensor{shape: xs} = xt = Map.fetch!(m, x)
    wrows = wt.data |> F32.decode() |> Enum.chunk_every(k) |> Enum.map(&List.to_tuple/1)
    zero = List.duplicate(0, n)

    out =
      xt.data
      |> F32.decode()
      |> Enum.chunk_every(k)
      |> Enum.zip(F32.decode(Map.fetch!(m, mk).data))
      |> Enum.flat_map(fn
        {_xrow, mbits} when (mbits &&& 0x7FFF_FFFF) == 0 -> zero
        {xrow, _} -> xr = List.to_tuple(xrow); Enum.map(wrows, &dot16(&1, xr, k, p))
      end)

    Tensor.new(:f32, List.replace_at(xs, -1, n), F32.encode(out))
  end

  # block-diagonal: group i of a row against rows i·n … of W
  defp node({:linear_grouped, x, w, g}, m, _env, p) do
    %Tensor{shape: [gn, k]} = wt = Map.fetch!(m, w)
    %Tensor{shape: xs} = xt = Map.fetch!(m, x)
    n = div(gn, g)
    groups = wt.data |> F32.decode() |> Enum.chunk_every(k) |> Enum.map(&List.to_tuple/1) |> Enum.chunk_every(n)

    out =
      xt.data
      |> F32.decode()
      |> Enum.chunk_every(g * k)
      |> Enum.flat_map(fn xrow ->
        xrow
        |> Enum.chunk_every(k)
        |> Enum.zip(groups)
        |> Enum.flat_map(fn {xg, wrows} -> xr = List.to_tuple(xg); Enum.map(wrows, &dot16(&1, xr, k, p)) end)
      end)

    Tensor.new(:f32, List.replace_at(xs, -1, gn), F32.encode(out))
  end

  defp node({:qgemv, w, x}, m, _env, p) do
    wt = Sb4.to_exec(Map.fetch!(m, w))
    xt = Map.fetch!(m, x)
    %Tensor{shape: [rows, k]} = wt
    wrows = for r <- 0..(rows - 1), do: Tensor.row(wt, r)

    # x : f32[k] or f32[b, k]: every activation row is an independent GEMV
    y =
      xt.data
      |> F32.decode()
      |> Enum.chunk_every(k)
      |> Enum.flat_map(fn xs ->
        sums = subblock_sums(xs)
        xtup = List.to_tuple(xs)
        for wr <- wrows, do: qgemv_row(wr, xtup, sums, p)
      end)

    Tensor.new(:f32, Enum.drop(xt.shape, -1) ++ [rows], F32.encode(y))
  end

  # row-predicated 4-bit contraction: inactive rows (mask ±0) are +0 and
  # read no weight; active rows are exactly qgemv's
  defp node({:qgemv_masked, w, x, mk}, m, _env, p) do
    wt = Sb4.to_exec(Map.fetch!(m, w))
    xt = Map.fetch!(m, x)
    %Tensor{shape: [rows, k]} = wt
    wrows = for r <- 0..(rows - 1), do: Tensor.row(wt, r)
    zero = List.duplicate(0, rows)

    y =
      xt.data
      |> F32.decode()
      |> Enum.chunk_every(k)
      |> Enum.zip(F32.decode(Map.fetch!(m, mk).data))
      |> Enum.flat_map(fn
        {_xs, mbits} when (mbits &&& 0x7FFF_FFFF) == 0 -> zero
        {xs, _} ->
          sums = subblock_sums(xs)
          xtup = List.to_tuple(xs)
          for wr <- wrows, do: qgemv_row(wr, xtup, sums, p)
      end)

    Tensor.new(:f32, Enum.drop(xt.shape, -1) ++ [rows], F32.encode(y))
  end

  defp node({:gemm_i8, a, w}, m, _env, _p) do
    at = Map.fetch!(m, a)
    wt = Map.fetch!(m, w)
    [mm, _k] = at.shape
    [nn, _] = wt.shape
    wrows = for j <- 0..(nn - 1), do: s8_list(Tensor.row(wt, j))

    data =
      for i <- 0..(mm - 1), into: <<>> do
        arow = s8_list(Tensor.row(at, i))
        for wr <- wrows, into: <<>>, do: <<wrap_s32(dot(arow, wr))::signed-32-little>>
      end

    Tensor.new(:s32, [mm, nn], data)
  end

  # ---------------------------------------------------------------- scalars --

  defp node({:gather_row, table, idx}, m, _env, _p) do
    %Tensor{shape: [v, d]} = tt = Map.fetch!(m, table)
    rows = for i <- idx_list(Map.fetch!(m, idx)), do: Tensor.row(tt, min(i, v - 1))
    Tensor.new(:f32, [length(rows), d], IO.iodata_to_binary(rows))
  end

  defp node({:rope, x, cos, sin, pos, h}, m, _env, _p) do
    %Tensor{shape: [n, w]} = xt = Map.fetch!(m, x)
    %Tensor{shape: [s, half]} = ct = Map.fetch!(m, cos)
    st = Map.fetch!(m, sin)
    dh = div(w, h)

    out =
      for {row, p} <- Enum.zip(0..(n - 1), idx_list(Map.fetch!(m, pos))), into: <<>> do
        c = ct |> Tensor.row(min(p, s - 1)) |> F32.decode()
        sn = st |> Tensor.row(min(p, s - 1)) |> F32.decode()

        for head <- xt |> Tensor.row(row) |> F32.decode() |> Enum.chunk_every(dh), into: <<>> do
          {x1, x2} = Enum.split(head, half)
          o1 = [x1, x2, c, sn] |> Enum.zip_with(fn [a, b, cc, ss] -> F32.sub(F32.mul(a, cc), F32.mul(b, ss)) end)
          o2 = [x1, x2, c, sn] |> Enum.zip_with(fn [a, b, cc, ss] -> F32.add(F32.mul(b, cc), F32.mul(a, ss)) end)
          F32.encode(o1 ++ o2)
        end
      end

    Tensor.new(:f32, [n, w], out)
  end

  defp node({:kv_write, cache, pos, rows}, m, _env, _p) do
    %Tensor{shape: [s, w]} = ct = Map.fetch!(m, cache)
    rt = Map.fetch!(m, rows)
    writes = idx_list(Map.fetch!(m, pos)) |> Enum.with_index() |> Enum.filter(fn {p, _} -> p < s end)
    table = Enum.reduce(writes, Map.new(0..(s - 1), &{&1, Tensor.row(ct, &1)}), fn {p, t}, acc -> Map.put(acc, p, Tensor.row(rt, t)) end)
    Tensor.new(:f32, [s, w], IO.iodata_to_binary(for r <- 0..(s - 1), do: table[r]))
  end

  defp node({:attention, q, k, v, pos, heads}, m, _env, _p) do
    {h, hkv} = Term.heads_of(heads)
    %Tensor{shape: [s, _]} = Map.fetch!(m, k)
    %Tensor{shape: [_, w]} = qt = Map.fetch!(m, q)
    ps = idx_list(Map.fetch!(m, pos))
    win = Term.attention_window(heads)

    attention_rows(qt, Map.fetch!(m, k), Map.fetch!(m, v), h, hkv,
                   Enum.map(ps, fn p -> window(min(p, s - 1), win) end), Term.attention_scale_bits(heads, div(w, h)))
  end

  defp node({:kv_write_paged, pool, table, slot, pos, rows, page}, m, _env, _p) do
    %Tensor{shape: [pr, w]} = pt = Map.fetch!(m, pool)
    rt = Map.fetch!(m, rows)
    tab = paged_table(Map.fetch!(m, table))

    writes =
      Enum.zip([idx_list(Map.fetch!(m, slot)), idx_list(Map.fetch!(m, pos)), 0..(rt.shape |> hd() |> max(1)) - 1])
      |> Enum.flat_map(fn {sl, p, t} ->
        case paged_row(tab, sl, p, page, div(pr, page), :skip) do
          nil -> []
          r -> [{r, t}]
        end
      end)

    rows_of = Enum.reduce(writes, Map.new(0..(pr - 1), &{&1, Tensor.row(pt, &1)}), fn {r, t}, acc -> Map.put(acc, r, Tensor.row(rt, t)) end)
    Tensor.new(:f32, [pr, w], IO.iodata_to_binary(for r <- 0..(pr - 1), do: rows_of[r]))
  end

  defp node({:attention_paged, q, kp, vp, table, slot, pos, heads}, m, _env, _p) do
    {h, hkv, page} = Term.heads_of(heads)
    %Tensor{shape: [_, w]} = Map.fetch!(m, q)
    scale = Term.attention_scale_bits(heads, div(w, h))
    %Tensor{shape: [pr, _]} = Map.fetch!(m, kp)
    tab = paged_table(Map.fetch!(m, table))
    cap = elem(tab, 1) * page

    keys =
      Enum.zip_with(idx_list(Map.fetch!(m, slot)), idx_list(Map.fetch!(m, pos)), fn sl, p ->
        for j <- window(min(p, cap - 1), Term.attention_window(heads)), do: paged_row(tab, sl, j, page, div(pr, page), :clamp)
      end)

    attention_rows(Map.fetch!(m, q), Map.fetch!(m, kp), Map.fetch!(m, vp), h, hkv, keys, scale)
  end

  defp node({:reshape, x, shape}, m, _env, _p), do: %{Map.fetch!(m, x) | shape: shape}

  defp node({:transpose, x}, m, _env, _p) do
    %Tensor{shape: [r, c]} = xt = Map.fetch!(m, x)
    words = for <<w::binary-4 <- xt.data>>, do: w
    tup = List.to_tuple(words)
    Tensor.new(:f32, [c, r], IO.iodata_to_binary(for j <- 0..(c - 1), i <- 0..(r - 1), do: elem(tup, i * c + j)))
  end

  # the sampling kernel's definition, step for step
  defp node({:sample, logits, params}, m, _env, _p) do
    %Tensor{shape: [b, _v]} = lt = Map.fetch!(m, logits)
    pt = Map.fetch!(m, params)
    exp = Vapor.Canon.compile(:exp)

    out =
      for r <- 0..(b - 1)//1, into: <<>> do
        row = lt |> Tensor.row(r) |> F32.decode()
        [it, u] = pt |> Tensor.row(r) |> F32.decode()
        mx = reduce_row(:max, row)

        i =
          if it == 0 do
            Enum.find_index(row, &(&1 == mx))
          else
            es = Enum.map(row, &exp.([F32.mul(F32.sub(&1, mx), it)]))
            tot = Enum.reduce(es, 0, &F32.add(&2, &1))
            tgt = F32.mul(u, tot)

            {pick, last, _} =
              es
              |> Enum.with_index()
              |> Enum.reduce_while({nil, 0, 0}, fn {e, j}, {nil, last, c} ->
                c = F32.add(c, e)
                last = if e != 0, do: j, else: last
                if c > tgt, do: {:halt, {j, last, c}}, else: {:cont, {nil, last, c}}
              end)

            pick || last
          end

        <<i::signed-32-little>>
      end

    Tensor.new(:s32, [b], out)
  end

  # {rows as tuples of page ids, pages per sequence}
  defp paged_table(%Tensor{shape: [ns, mp]} = t) do
    rows = t |> idx_list() |> Enum.chunk_every(mp) |> Enum.map(&List.to_tuple/1) |> List.to_tuple()
    {rows, mp, ns}
  end

  @doc false
  # physical pool row of logical row j of sequence sl: writes skip anything
  # out of range, reads clamp every index (total, like the other gathers)
  def paged_row({rows, mp, ns}, sl, j, page, npages, mode) do
    blk = div(j, page)

    case mode do
      :skip ->
        if sl < ns and blk < mp do
          pg = elem(elem(rows, sl), blk)
          if pg < npages, do: pg * page + rem(j, page)
        end

      :clamp ->
        pg = elem(elem(rows, min(sl, ns - 1)), min(blk, mp - 1))
        min(pg, npages - 1) * page + rem(j, page)
    end
  end

  # causal attention of each row of q over the given physical key rows,
  # in the canonical order (`attend/3`)
  defp attention_rows(%Tensor{shape: [n, w]} = qt, kt, vt, h, hkv, keys_per_row, scale) do
    dh = div(w, h)
    g = div(h, hkv)
    krow = fn r -> kt |> Tensor.row(r) |> F32.decode() |> Enum.chunk_every(dh) |> Enum.map(&List.to_tuple/1) |> List.to_tuple() end
    vrow = fn r -> vt |> Tensor.row(r) |> F32.decode() |> Enum.chunk_every(dh) |> List.to_tuple() end
    needed = keys_per_row |> List.flatten() |> Enum.uniq()
    kc = Map.new(needed, &{&1, krow.(&1)})
    vc = Map.new(needed, &{&1, vrow.(&1)})

    out =
      for {row, keys} <- Enum.zip(0..(n - 1), keys_per_row), into: <<>> do
        for {qh, head} <- qt |> Tensor.row(row) |> F32.decode() |> Enum.chunk_every(dh) |> Enum.with_index(), into: <<>> do
          kv = div(head, g)
          qtup = List.to_tuple(qh)
          scores = for j <- keys, do: F32.mul(dot16(elem(kc[j], kv), qtup, dh, :canonical), scale)
          F32.encode(attend(scores, for(j <- keys, do: elem(vc[j], kv)), dh))
        end
      end

    Tensor.new(:f32, [n, w], out)
  end

  @doc "1/√dh as binary32 — the attention scale baked into the kernels."
  def attention_scale(dh), do: F32.from_float(1 / :math.sqrt(dh))

  @doc """
  Canonical softmax-weighted sum over `L` scores: `m` = canonical max over
  `⌈L/16⌉·16` lanes padded with −FLT_MAX; `e_j = exp(s_j − m)` (canonical
  exp) and `+0` on the padding; `Z` = canonical sum; `p_j = e_j·rcp(Z)`;
  `o[d] = Σ_j p_j·v_j[d]` accumulated sequentially in `j` from `+0`.
  """
  def attend(scores, vs, dh) do
    l = length(scores)
    l16 = div(l + 15, 16) * 16
    m = reduce_row(:max, scores ++ List.duplicate(@neg_max, l16 - l))
    exp = Vapor.Canon.compile(:exp)
    es = Enum.map(scores, &exp.([F32.sub(&1, m)]))
    inv = Vapor.Canon.compile(:rcp).([reduce_row(:sum, es ++ List.duplicate(0, l16 - l))])
    ps = Enum.map(es, &F32.mul(&1, inv))

    Enum.zip(ps, vs)
    |> Enum.reduce(List.duplicate(0, dh), fn {p, vrow}, acc -> Enum.zip_with(acc, vrow, fn a, x -> F32.add(a, F32.mul(x, p)) end) end)
  end

  defp idx_list(%Tensor{dtype: :s32} = t), do: t |> Tensor.to_list() |> Enum.map(&(&1 &&& 0xFFFF_FFFF))

  # dynamic extents resolved from the operands' concrete shapes
  defp concrete(decl, actual) do
    Enum.with_index(decl, fn
      d, _i when is_integer(d) -> d
      _d, i -> actual |> Enum.map(&Enum.at(&1, i)) |> Enum.max()
    end)
  end

  @doc "Elementwise function of a node: primitive or canonical microprogram."
  def scalar_fun(op, p) do
    if Vapor.Canon.function?(op) do
      run = Vapor.Canon.compile(op, p)
      fn xs -> run.(xs) end
    else
      fn xs -> ew_scalar(op, xs, p) end
    end
  end

  @doc "NumPy equal-rank broadcast of a tensor's elements to `shape` (row-major)."
  def broadcast(%Tensor{shape: shape, data: d}, shape), do: F32.decode(d)

  def broadcast(%Tensor{shape: s, data: d}, target) do
    xs = d |> F32.decode() |> List.to_tuple()
    strides = s |> Enum.reverse() |> Enum.map_reduce(1, fn e, acc -> {if(e == 1, do: 0, else: acc), acc * e} end) |> elem(0) |> Enum.reverse()

    for idx <- indices(target) do
      elem(xs, Enum.zip_reduce(idx, strides, 0, fn i, st, acc -> acc + i * st end))
    end
  end

  defp indices([]), do: [[]]
  defp indices([n | rest]), do: for(i <- 0..(n - 1), r <- indices(rest), do: [i | r])

  @doc """
  Canonical row reduction: lane l (of 16) folds the elements i ≡ l (mod 16)
  in increasing order from the identity (+0 for sum, −FLT_MAX for max),
  then the 16-lane tree combines the lanes.
  """
  def reduce_row(op, xs) do
    {id, f} = combiner(op)

    xs
    |> Enum.chunk_every(16)
    |> Enum.reduce(List.duplicate(id, 16), fn chunk, acc -> Enum.zip_with(acc, chunk ++ List.duplicate(id, 16 - length(chunk)), f) end)
    |> tree16(f)
  end

  defp combiner(:sum), do: {0, &F32.add/2}
  defp combiner(:max), do: {@neg_max, fn a, b -> Vapor.Canon.prim(:sel_lt, [a, b, b, a], :canonical) end}

  @doc "The canonical tree over 16 lanes with a combining function."
  def tree16(lanes, f) do
    Enum.reduce([8, 4, 2, 1], lanes, fn h, xs ->
      t = List.to_tuple(xs)
      for i <- 0..(h - 1), do: f.(elem(t, i), elem(t, i + h))
    end)
    |> hd()
  end

  # canonical dot: lane l accumulates w[16c+l]·x[16c+l] over c, then the tree
  # the positions a row at (clamped) position p attends: all of 0 … p, or
  # the last `w` of them
  defp window(p, nil), do: Enum.to_list(0..p)
  defp window(p, w), do: Enum.to_list(max(0, p - w + 1)..p)

  # rung 1 refuses k ≢ 0 (mod 16); a program evaluated here without it must
  # not lose the tail of its contraction in silence (found in 0.15)
  defp dot16(_w, _x, k, _p) when rem(k, 16) != 0,
    do: raise(ArgumentError, "canonical dot over k = #{k}: k must be ≡ 0 (mod 16) (Program.check/1 rejects this program)")

  defp dot16(w, x, k, p) do
    Enum.reduce(0..(div(k, 16) - 1), List.duplicate(0, 16), fn c, acc ->
      base = 16 * c
      acc |> Enum.with_index() |> Enum.map(fn {a, l} -> mac(elem(w, base + l), elem(x, base + l), a, p) end)
    end)
    |> reduce16()
  end

  @doc "One elementwise application under a float policy."
  def ew_scalar(:add, [a, b], _), do: F32.add(a, b)
  def ew_scalar(:sub, [a, b], _), do: F32.sub(a, b)
  def ew_scalar(:mul, [a, b], _), do: F32.mul(a, b)
  def ew_scalar(:neg, [a], _), do: F32.neg(a)
  def ew_scalar(:relu, [a], _), do: F32.relu(a)
  def ew_scalar(:fma, [a, b, c], :canonical), do: F32.add(F32.mul(a, b), c)
  def ew_scalar(:fma, [a, b, c], :fast), do: F32.fma(a, b, c)

  @doc "The canonical 16-lane tree reduction."
  def reduce16(lanes) when length(lanes) == 16 do
    t = List.to_tuple(lanes)
    r8 = for i <- 0..7, do: F32.add(elem(t, i), elem(t, i + 8))
    r4 = pairwise(r8, 4)
    r2 = pairwise(r4, 2)
    [a, b] = r2
    F32.add(a, b)
  end

  defp pairwise(xs, h) do
    t = List.to_tuple(xs)
    for i <- 0..(h - 1), do: F32.add(elem(t, i), elem(t, i + h))
  end

  @doc "X_s = reduce16(x_{s,l} + x_{s,16+l}) — the sub-block activation sums."
  def subblock_sums(xs) do
    xs
    |> Enum.chunk_every(32)
    |> Enum.map(fn sub ->
      {lo, hi} = Enum.split(sub, 16)
      reduce16(Enum.zip_with(lo, hi, &F32.add/2))
    end)
    |> List.to_tuple()
  end

  defp qgemv_row(row_bin, xtup, sums, p) do
    zero16 = List.duplicate(0, 16)

    {acc, yb, _s} =
      for <<blk::binary-152 <- row_bin>>, reduce: {zero16, 0, 0} do
        {acc, yb, s0} ->
          {q, ab} = Sb4.exec_block(blk)

          Enum.with_index(ab)
          |> Enum.reduce({acc, yb, s0}, fn {{alpha, beta}, sl}, {acc, yb, s} ->
            base = 32 * sl
            gbase = 32 * s

            acc =
              acc
              |> lane_step(alpha, q, base, xtup, gbase, p)
              |> lane_step(alpha, q, base + 16, xtup, gbase + 16, p)

            yb = mac(beta, elem(sums, s), yb, p)
            {acc, yb, s + 1}
          end)
      end

    F32.add(reduce16(acc), yb)
  end

  defp lane_step(acc, alpha, q, qbase, xtup, xbase, p) do
    acc
    |> Enum.with_index()
    |> Enum.map(fn {a, l} ->
      w = F32.mul(alpha, F32.from_float(elem(q, qbase + l)))
      mac(w, elem(xtup, xbase + l), a, p)
    end)
  end

  defp mac(a, b, c, :canonical), do: F32.add(c, F32.mul(a, b))
  defp mac(a, b, c, :fast), do: F32.fma(a, b, c)

  # --------------------------------------------------------------- integers --

  defp s8_list(bin), do: for(<<x::signed-8 <- bin>>, do: x)
  defp dot(a, b), do: Enum.zip_reduce(a, b, 0, fn x, y, acc -> acc + x * y end)

  @doc """
  Two's-complement reading of the accumulator — the Lean-extracted
  `Vapor.wrapS32` (Theorem 7.1: equal to the exact value when admissible).
  """
  def wrap_s32(x), do: Vapor.Extracted.wrap_s32(x)
end
