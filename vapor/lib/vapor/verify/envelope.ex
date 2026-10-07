defmodule Vapor.Verify.Envelope do
  @moduledoc """
  Theorem 7.2, generalised: a rigorous *a-priori* bound on the distance
  between any conforming substrate's output and the exact real result, for
  every output element of a program, computed in exact dyadic arithmetic.

  "Conforming" means: each `add`/`sub`/`mul` correctly rounded (any
  substrate), `fma` fused *or not*, reductions in *any* order, underflow
  either gradual or flushed to zero, and subnormal operands read either as
  themselves or as zero (DAZ). So one bound covers x86, ARM, RVV, every
  Vulkan driver and XLA's CPU under both float policies.

  (DAZ was added in 0.10, after the substrate airlock measured it on XLA's
  CPU: `min(x, 0.25)` with a subnormal `x` returned `+0`, outside a bound
  that treated selections as exact. A subnormal operand flushed to zero is
  an operand at distance `|v|` from its exact value; every elementwise rule
  and every reduction now charges it to operands that may be subnormal.)

  Two forms:

    * `{:wilkinson, v, s, a, n}` — a contraction over exact inputs:
      `|ŷ − v| ≤ γₙ·s + a`. Decided by `Vapor.Extracted.within_envelope/4`,
      the Lean-extracted decision (`Vapor.withinEnvelope`, monotone in `n`).
      `n` is the longest rounding path of the declared expression tree
      (Higham, Lemma 3.1): `K + nsub` products/adds plus 31 adds inside a
      sub-block sum.
    * `{:running, v, e}` — running error analysis (Wilkinson/Higham) through
      elementwise operators and contractions over inexact inputs:
      `add: e = (1+u)(e₁+e₂) + u|v| + η`,
      `mul: m = |v₁|e₂ + |v₂|e₁ + e₁e₂, e = m + u(|v₁v₂| + m) + η`,
      `fma` = `mul` then `add` (covers the fused case), `neg`/`relu` exact
      (1-Lipschitz); `η = 2⁻¹²⁶` per rounding covers flush-to-zero.

  Row reductions and dense linear maps use the same running analysis with
  `γₙ` over the reduced length (valid for *any* summation order); `max`/`min`
  are exact selections (1-Lipschitz: the error is the larger input error).
  The remaining canonical functions (`exp`, `rcp`, `rsqrt`, `div`,
  `sigmoid`, `silu`) are defined by their `Vapor.Canon` microprograms rather
  than by real analysis, so their outputs carry `:na`: under `:canonical`
  they are certified by bit parity with the oracle (Rung 5), and the ladder
  refuses `:fast` for programs that contain them.

  The analytic lemma (Higham, *Accuracy and Stability of Numerical
  Algorithms*, 2nd ed., Lemma 3.1 and (3.4), any summation order) is
  machine-checked in `proofs/Vapor/Higham.lean` from the standard model of
  rounding, together with its bridge to the decision for the Wilkinson form
  (`withinEnvelope_of_bound`); the standard model itself — IEEE-754
  round-to-nearest returns `(x op y)(1 + δ)`, `|δ| ≤ 2⁻²⁴`, away from
  underflow — is the hardware specification's, and underflow is the
  absolute term.
  """
  alias Vapor.{F32, Tensor}
  alias Vapor.Algebra.Term
  alias Vapor.Quant.Sb4
  alias Vapor.Verify.Dyadic, as: D

  @u {1, -24}
  @eta {1, -126}

  @type bound :: {:wilkinson, D.t(), D.t(), D.t(), pos_integer} | {:running, D.t(), D.t()}

  @doc """
  Bounds for every output of a program, per iteration. Options as
  `Vapor.Runtime.Native.run/4` (`:iterations`, `:sequence`).
  """
  def bounds(%Vapor.Compiled{program: p}, env, opts \\ []) do
    t_count = Keyword.get(opts, :iterations, 1)
    seq = MapSet.new(Keyword.get(opts, :sequence, []))

    exact_env =
      Map.new(env, fn {k, %Tensor{} = t} -> {k, if(MapSet.member?(seq, k), do: t, else: val(t))} end)

    {steps, _} =
      Enum.map_reduce(0..(t_count - 1), exact_env, fn t, cur ->
        env_t =
          Map.new(cur, fn {k, v} ->
            {k, if(MapSet.member?(seq, k), do: val(Vapor.Runtime.Native.slice(env[k], t)), else: v)}
          end)

        vals = Vapor.Program.evaluate(p, env_t, &node({:const, &1}, %{}, %{}), &node/3)
        outs = Map.new(vals, fn {name, v} -> {name, v.elems} end)

        next =
          Enum.reduce(p.state, cur, fn {in_n, out_n}, acc ->
            Map.put(acc, in_n, %{shape: vals[out_n].shape, elems: Enum.map(vals[out_n].elems, &as_elem/1)})
          end)

        {outs, next}
      end)

    steps
  end

  @doc "Check a substrate's tensor against its bounds: `:ok` or the first violation."
  def check(bounds, %Tensor{dtype: dt} = t) do
    got = Tensor.to_list(t)
    bounds = if is_map(bounds), do: bounds.elems, else: bounds

    Enum.zip(bounds, got)
    |> Enum.with_index()
    |> Enum.reduce_while(:ok, fn {{b, g}, i}, :ok ->
      if within?(b, g, dt), do: {:cont, :ok}, else: {:halt, {:error, %{index: i, got: g, bound: b}}}
    end)
  end

  @doc "Largest bound (absolute) over a list — the figure the certificate reports."
  def max_error(bounds) do
    bounds |> Enum.reject(&(&1 == :na)) |> Enum.map(&abs_bound/1) |> Enum.reduce(D.zero(), &D.max/2) |> D.round_up(24)
  end

  @doc "Whether every bound is an analytic one (no `:na` from canonical functions)."
  def analytic?(bounds), do: :na not in bounds

  defp within?(:na, _g, _dt), do: true
  defp within?({:running, v, e}, g, :s32), do: D.le?(D.abs(D.sub(D.of_int(g), v)), e)

  defp within?(b, bits, :f32) do
    if F32.finite?(bits), do: decide(b, D.of_f32(bits)), else: false
  end

  defp decide({:running, v, e}, y), do: D.le?(D.abs(D.sub(y, v)), e)

  defp decide({:wilkinson, v, s, a, n}, y) do
    [d, s_i, a_i] = D.to_common_ints([D.abs(D.sub(y, v)), s, a])
    Vapor.Extracted.within_envelope(d, s_i, a_i, n)
  end

  defp abs_bound({:running, _v, e}), do: e
  defp abs_bound({:wilkinson, _v, s, a, n}), do: D.add(D.mul(D.gamma(n), s), a)

  # elements carried between nodes: {value, error} or :na
  defp as_elem(:na), do: :na
  defp as_elem({:running, v, e}), do: {v, e}
  defp as_elem({{_, _} = v, {_, _} = e}), do: {v, e}
  defp as_elem({:wilkinson, _, _, _, _} = b), do: {elem(b, 1), D.round_up(abs_bound(b))}

  defp exact_elems(%Tensor{dtype: :f32} = t), do: Enum.map(Tensor.to_list(t), &{D.of_f32(&1), D.zero()})
  defp exact_elems(%Tensor{dtype: :bf16} = t), do: exact_elems(Tensor.widen(t))
  defp exact_elems(%Tensor{dtype: d} = t) when d in [:s8, :s32, :u8], do: Enum.map(Tensor.to_list(t), &{D.of_int(&1), D.zero()})
  defp exact_elems(%Tensor{} = t), do: t

  # ------------------------------------------------------------------ nodes --
  # every node yields %{shape: concrete shape, elems: bounds (or a Tensor for sb4)}

  defp node({:input, name, _, _}, _m, env), do: Map.fetch!(env, name)

  defp node({:const, %Tensor{dtype: q} = t}, _m, _env) when q in [:sb4, :sb4x],
    do: %{shape: t.shape, elems: Sb4.to_exec(t)}

  defp node({:const, t}, _m, _env), do: %{shape: t.shape, elems: exact_elems(t)}

  defp node({:ew, op, args}, m, _env) do
    shapes = for a <- args, not match?({:splat, _}, a), do: m[a].shape
    {:ok, shape} = Term.broadcast_shapes(shapes)
    n = Enum.product(shape)

    cols =
      Enum.map(args, fn
        {:splat, b} -> List.duplicate({D.of_f32(b), D.zero()}, n)
        a -> m[a].elems |> Enum.map(&as_elem/1) |> broadcast(m[a].shape, shape)
      end)

    elems =
      cols
      |> Enum.zip_with(fn xs -> if :na in xs, do: :na, else: ew(op, xs) end)
      |> Enum.map(fn
        :na -> :na
        {v, e} -> {:running, v, e}
      end)

    %{shape: shape, elems: elems}
  end

  defp node({:reduce, op, x}, m, _env) do
    %{shape: shape, elems: xs} = m[x]
    c = List.last(shape)

    elems =
      xs
      |> Enum.map(&as_elem/1)
      |> Enum.chunk_every(c)
      |> Enum.map(fn row -> if :na in row, do: :na, else: reduce_bound(op, row) end)

    %{shape: List.replace_at(shape, -1, 1), elems: elems}
  end

  defp node({:linear, x, w}, m, _env) do
    %{shape: [n, k], elems: ws} = m[w]
    %{shape: xshape, elems: xs} = m[x]
    wrows = ws |> Enum.map(&as_elem/1) |> Enum.chunk_every(k)
    g = D.gamma(k)

    elems =
      for xrow <- xs |> Enum.map(&as_elem/1) |> Enum.chunk_every(k), wrow <- wrows do
        if :na in xrow or :na in wrow do
          :na
        else
          {v, s, prop, cmax} =
            Enum.zip_reduce(wrow, xrow, {D.zero(), D.zero(), D.zero(), D.zero()}, fn {vw, ew}, {vx, ex}, {v, s, p, c} ->
              mw = D.add(D.abs(vw), ew)
              mx = D.add(D.abs(vx), ex)
              {D.add(v, D.mul(vw, vx)), D.add(s, D.mul(mw, mx)), D.add(p, D.add(D.mul(mw, ex), D.mul(D.abs(vx), ew))), D.max(c, mw)}
            end)

          a = D.mul(D.mul(@eta, D.of_int(4 * k)), D.add(D.one(), cmax)) |> D.round_up()

          if prop == D.zero(),
            do: {:wilkinson, v, D.round_up(s), a, k},
            else: {:running, v, D.add(D.add(D.mul(g, s), D.mul(D.add(D.one(), g), prop)), a) |> D.round_up()}
        end
      end

    %{shape: List.replace_at(xshape, -1, n), elems: elems}
  end

  # block-diagonal: each group is a dense linear map over its own columns
  defp node({:linear_grouped, x, w, g}, m, env) do
    %{shape: [gn, k], elems: we} = m[w]
    %{shape: xshape, elems: xe} = m[x]
    n = div(gn, g)
    rows = Enum.chunk_every(xe, g * k)
    wgroups = Enum.chunk_every(we, n * k)

    per_group =
      for {wg, i} <- Enum.with_index(wgroups) do
        xg = Enum.flat_map(rows, &Enum.slice(&1, i * k, k))
        lead = Enum.drop(xshape, -1)
        node({:linear, {:g, :x}, {:g, :w}}, %{{:g, :x} => %{shape: lead ++ [k], elems: xg}, {:g, :w} => %{shape: [n, k], elems: wg}}, env).elems
        |> Enum.chunk_every(n)
      end

    elems = per_group |> Enum.zip() |> Enum.flat_map(fn t -> t |> Tuple.to_list() |> Enum.concat() end)
    %{shape: List.replace_at(xshape, -1, gn), elems: elems}
  end

  # a row is active iff its mask is nonzero: decided exactly when the mask
  # is known exactly, else unknown (:na) for that row
  defp node({:linear_masked, x, w, mk}, m, env) do
    %{shape: shape, elems: dense} = node({:linear, x, w}, m, env)
    n = List.last(shape)

    elems =
      dense
      |> Enum.chunk_every(n)
      |> Enum.zip(Enum.map(m[mk].elems, &as_elem/1))
      |> Enum.flat_map(fn
        {_row, :na} -> List.duplicate(:na, n)
        {row, {{vm, _}, {0, _}}} -> if vm == 0, do: List.duplicate({:running, D.zero(), D.zero()}, n), else: row
        {row, {v, e}} -> if D.compare(D.abs(v), e) == :gt, do: row, else: List.duplicate(:na, n)
      end)

    %{shape: shape, elems: elems}
  end

  # rows and positions are selected exactly: bounds are carried, not changed
  defp node({:gather_row, table, idx}, m, _env) do
    %{shape: [v, d], elems: te} = m[table]
    rows = Enum.chunk_every(te, d)
    %{shape: [n], elems: ie} = m[idx]
    elems = Enum.flat_map(ie, fn i -> Enum.at(rows, min(uidx(i), v - 1)) end)
    %{shape: [n, d], elems: Enum.map(elems, &bound/1)}
  end

  defp node({:kv_write, cache, pos, rows}, m, _env) do
    %{shape: [s, w], elems: ce} = m[cache]
    rows_e = Enum.chunk_every(m[rows].elems, w)

    table =
      m[pos].elems
      |> Enum.zip(rows_e)
      |> Enum.reduce(ce |> Enum.chunk_every(w) |> Enum.with_index() |> Map.new(fn {r, i} -> {i, r} end), fn {p, r}, acc ->
        if uidx(p) < s, do: Map.put(acc, uidx(p), r), else: acc
      end)

    %{shape: [s, w], elems: Enum.flat_map(0..(s - 1), &table[&1]) |> Enum.map(&bound/1)}
  end

  # o₁ = x₁c − x₂s, o₂ = x₂c + x₁s through the running analysis
  defp node({:rope, x, cos, sin, pos, h}, m, _env) do
    %{shape: [n, w], elems: xe} = m[x]
    %{shape: [s, half], elems: ce} = m[cos]
    crow = ce |> Enum.chunk_every(half) |> List.to_tuple()
    srow = m[sin].elems |> Enum.chunk_every(half) |> List.to_tuple()
    dh = div(w, h)

    elems =
      Enum.zip(Enum.chunk_every(xe, w), m[pos].elems)
      |> Enum.flat_map(fn {row, p} ->
        r = min(uidx(p), s - 1)
        c = elem(crow, r) |> Enum.map(&as_elem/1)
        sn = elem(srow, r) |> Enum.map(&as_elem/1)

        Enum.flat_map(Enum.chunk_every(row, dh), fn head ->
          {x1, x2} = head |> Enum.map(&as_elem/1) |> Enum.split(half)
          o1 = Enum.zip_with([x1, x2, c, sn], fn [a, b, cc, ss] -> rope_elem(:sub, a, cc, b, ss) end)
          o2 = Enum.zip_with([x1, x2, c, sn], fn [a, b, cc, ss] -> rope_elem(:add, b, cc, a, ss) end)
          o1 ++ o2
        end)
      end)

    %{shape: [n, w], elems: elems}
  end

  defp node({:kv_write_paged, pool, table, slot, pos, rows, page}, m, _env) do
    %{shape: [pr, w], elems: pe} = m[pool]
    %{shape: [ns, mp], elems: te} = m[table]
    tab = {te |> Enum.map(&uidx/1) |> Enum.chunk_every(mp) |> Enum.map(&List.to_tuple/1) |> List.to_tuple(), mp, ns}
    rows_e = Enum.chunk_every(m[rows].elems, w)

    table_rows =
      Enum.zip([m[slot].elems, m[pos].elems, rows_e])
      |> Enum.reduce(pe |> Enum.chunk_every(w) |> Enum.with_index() |> Map.new(fn {r, i} -> {i, r} end), fn {sl, p, r}, acc ->
        case Vapor.Runtime.Oracle.paged_row(tab, uidx(sl), uidx(p), page, div(pr, page), :skip) do
          nil -> acc
          row -> Map.put(acc, row, r)
        end
      end)

    %{shape: [pr, w], elems: Enum.flat_map(0..(pr - 1), &table_rows[&1]) |> Enum.map(&bound/1)}
  end

  defp node({:attention_paged, q, _kp, _vp, _t, _s, _pos, _h}, m, _env),
    do: %{shape: m[q].shape, elems: List.duplicate(:na, length(m[q].elems))}

  defp node({:reshape, x, shape}, m, _env), do: %{m[x] | shape: shape}

  defp node({:transpose, x}, m, _env) do
    %{shape: [r, c], elems: xe} = m[x]
    rows = xe |> Enum.chunk_every(c) |> Enum.map(&List.to_tuple/1) |> List.to_tuple()
    %{shape: [c, r], elems: for(j <- 0..(c - 1), i <- 0..(r - 1), do: elem(elem(rows, i), j))}
  end

  # a sampled index follows canonical exp: certified by bit parity
  defp node({:sample, _l, p}, m, _env), do: %{shape: [hd(m[p].shape)], elems: List.duplicate(:na, hd(m[p].shape))}

  # softmax attention runs canonical exp and rcp: certified by bit parity
  defp node({:attention, q, _k, _v, _pos, _h}, m, _env), do: %{shape: m[q].shape, elems: List.duplicate(:na, length(m[q].elems))}

  defp node({:gemm_i8, a, w}, m, _env) do
    %{shape: [mm, k], elems: ae} = m[a]
    %{shape: [nn, ^k], elems: we} = m[w]
    at = ae |> Enum.map(&as_elem/1) |> Enum.map(&elem(&1, 0)) |> Enum.chunk_every(k)
    wt = we |> Enum.map(&as_elem/1) |> Enum.map(&elem(&1, 0)) |> Enum.chunk_every(k)

    elems =
      for arow <- at, wrow <- wt do
        {:running, Enum.zip_reduce(arow, wrow, D.zero(), fn x, y, acc -> D.add(acc, D.mul(x, y)) end), D.zero()}
      end

    %{shape: [mm, nn], elems: elems}
  end

  # row by row for x : f32[b, k] (each activation row is its own GEMV)
  defp node({:qgemv, w, x}, m, _env) do
    wt = m[w].elems
    %Tensor{shape: [rows, k]} = wt
    lead = Enum.drop(m[x].shape, -1)

    elems =
      m[x].elems
      |> Enum.map(&as_elem/1)
      |> Enum.chunk_every(k)
      |> Enum.flat_map(fn xs ->
        if :na in xs, do: List.duplicate(:na, rows), else: qgemv_bounds(wt, List.to_tuple(xs), rows, k)
      end)

    %{shape: lead ++ [rows], elems: elems}
  end

  # predicated: qgemv's bounds on active rows, an exact +0 on inactive ones
  defp node({:qgemv_masked, w, x, mk}, m, env) do
    %{shape: shape, elems: dense} = node({:qgemv, w, x}, m, env)
    n = List.last(shape)

    elems =
      dense
      |> Enum.chunk_every(n)
      |> Enum.zip(Enum.map(m[mk].elems, &as_elem/1))
      |> Enum.flat_map(fn
        {_row, :na} -> List.duplicate(:na, n)
        {row, {{vm, _}, {0, _}}} -> if vm == 0, do: List.duplicate({:running, D.zero(), D.zero()}, n), else: row
        {row, {v, e}} -> if D.compare(D.abs(v), e) == :gt, do: row, else: List.duplicate(:na, n)
      end)

    %{shape: shape, elems: elems}
  end

  defp qgemv_bounds(wt, xs, rows, k) do
    exact? = Enum.all?(Tuple.to_list(xs), fn {_, e} -> e == D.zero() end)
    nsub = div(k, 32)
    path = k + nsub + 31
    g = D.gamma(path)

    for r <- 0..(rows - 1) do
      {v, s, prop, cmax} = qrow(Tensor.row(wt, r), xs)
      ops = k + 34 * nsub
      a = D.mul(D.mul(@eta, D.of_int(2 * ops)), D.add(D.one(), cmax)) |> D.round_up()

      if exact? do
        {:wilkinson, v, D.round_up(s), a, path}
      else
        e = D.add(D.add(D.mul(g, s), D.mul(D.add(D.one(), g), prop)), a) |> D.round_up()
        {:running, v, e}
      end
    end
  end

  # sum: γₙ over any order, plus the inputs' own errors and underflow; max: exact selection
  defp reduce_bound(:sum, row) do
    row = Enum.map(row, &daz/1)
    n = length(row)
    v = D.sum(Enum.map(row, &elem(&1, 0)))
    mag = D.sum(Enum.map(row, fn {v, e} -> D.add(D.abs(v), e) end))
    ein = D.sum(Enum.map(row, &elem(&1, 1)))
    {:running, v, D.add(D.add(D.mul(D.gamma(n), mag), ein), D.mul(@eta, D.of_int(n))) |> D.round_up()}
  end

  defp reduce_bound(:max, row) do
    row = Enum.map(row, &daz/1)
    {v, _} = Enum.reduce(row, fn {v, e}, {bv, be} -> if D.le?(bv, v), do: {v, e}, else: {bv, be} end)
    {:running, v, row |> Enum.map(&elem(&1, 1)) |> Enum.reduce(&D.max/2)}
  end

  # a carried element as a bound (inputs and constants are bare {v, e} pairs)
  defp bound({{_, _} = v, {_, _} = e}), do: {:running, v, e}
  defp bound(b), do: b

  defp rope_elem(op, a, c, b, s) do
    if :na in [a, c, b, s] do
      :na
    else
      {v, e} = ew(op, [ew(:mul, [a, c]), ew(:mul, [b, s])])
      {:running, v, e}
    end
  end

  defp uidx({:running, {m, 0}, _}), do: Bitwise.band(m, 0xFFFF_FFFF)
  defp uidx({{m, 0}, _}), do: Bitwise.band(m, 0xFFFF_FFFF)

  defp broadcast(xs, shape, shape), do: xs

  defp broadcast(xs, s, target) do
    t = List.to_tuple(xs)
    strides = s |> Enum.reverse() |> Enum.map_reduce(1, fn e, acc -> {if(e == 1, do: 0, else: acc), acc * e} end) |> elem(0) |> Enum.reverse()
    for idx <- indices(target), do: elem(t, Enum.zip_reduce(idx, strides, 0, fn i, st, acc -> acc + i * st end))
  end

  defp indices([]), do: [[]]
  defp indices([n | rest]), do: for(i <- 0..(n - 1), r <- indices(rest), do: [i | r])

  defp val(%Tensor{} = t), do: %{shape: t.shape, elems: exact_elems(t)}

  # one row: exact value, magnitude sum (with input errors), propagated input error, max |coef|
  defp qrow(row_bin, xs) do
    {acc, _} =
      for <<blk::binary-152 <- row_bin>>, reduce: {{D.zero(), D.zero(), D.zero(), D.zero()}, 0} do
        {acc, s0} ->
          {q, ab} = Sb4.exec_block(blk)

          Enum.with_index(ab)
          |> Enum.reduce({acc, s0}, fn {{alpha, beta}, sl}, {{v, s, prop, cmax}, sg} ->
            bd = D.of_f32(beta)
            bmag = D.abs(bd)

            {v, s, prop, cmax, xsum_v, xsum_mag, xsum_e} =
              Enum.reduce(0..31, {v, s, prop, cmax, D.zero(), D.zero(), D.zero()}, fn i, {v, s, prop, cmax, xv, xm, xe} ->
                wd = D.of_f32(F32.mul(alpha, F32.from_float(elem(q, 32 * sl + i))))
                {xi, ei} = elem(xs, 32 * sg + i)
                mag = D.add(D.abs(xi), ei)
                wm = D.abs(wd)

                {D.add(v, D.mul(wd, xi)), D.add(s, D.mul(wm, mag)), D.add(prop, D.mul(wm, ei)), D.max(cmax, wm),
                 D.add(xv, xi), D.add(xm, mag), D.add(xe, ei)}
              end)

            {{D.add(v, D.mul(bd, xsum_v)), D.add(s, D.mul(bmag, xsum_mag)), D.add(prop, D.mul(bmag, xsum_e)),
              D.max(cmax, bmag)}, sg + 1}
          end)
      end

    acc
  end

  # ------------------------------------------------------------ elementwise --

  defp ew(op, xs) when op in [:neg, :relu, :add, :sub, :mul, :max, :min, :fma], do: ew_exact(op, Enum.map(xs, &daz/1))

  defp ew(_function, _xs), do: :na

  # a subnormal operand read as zero (DAZ) is then at distance |v| from the
  # exact value: the operand's error becomes max(e, |v|) — charged only where
  # it may be a nonzero subnormal (|v| − e < 2⁻¹²⁶)
  defp daz({v, e} = x) do
    a = D.abs(v)

    if D.compare(D.sub(a, e), @eta) == :lt and D.compare(D.add(a, e), D.zero()) == :gt,
      do: {v, D.max(e, a)},
      else: x
  end

  defp ew_exact(:neg, [{v, e}]), do: {D.neg(v), e}
  defp ew_exact(:relu, [{v, e}]), do: {if(D.le?(v, D.zero()), do: D.zero(), else: v), e}
  defp ew_exact(:add, [{v1, e1}, {v2, e2}]), do: rounded_add(D.add(v1, v2), D.add(e1, e2))
  defp ew_exact(:sub, [{v1, e1}, {v2, e2}]), do: rounded_add(D.sub(v1, v2), D.add(e1, e2))
  defp ew_exact(:mul, [a, b]), do: rounded_mul(a, b)

  defp ew_exact(op, [{v1, e1}, {v2, e2}]) when op in [:max, :min] do
    pick = if op == :max, do: D.le?(v1, v2), else: D.le?(v2, v1)
    {if(pick, do: v2, else: v1), D.max(e1, e2)}
  end

  defp ew_exact(:fma, [a, b, {v3, e3}]) do
    {vp, ep} = rounded_mul(a, b)
    rounded_add(D.add(vp, v3), D.add(ep, e3))
  end

  defp rounded_add(v, e_in) do
    e = D.add(D.add(D.mul(D.add(D.one(), @u), e_in), D.mul(@u, D.abs(v))), @eta)
    {v, D.round_up(e)}
  end

  defp rounded_mul({v1, e1}, {v2, e2}) do
    v = D.mul(v1, v2)
    m = D.add(D.add(D.mul(D.abs(v1), e2), D.mul(D.abs(v2), e1)), D.mul(e1, e2))
    e = D.add(D.add(m, D.mul(@u, D.add(D.abs(v), m))), @eta)
    {v, D.round_up(e)}
  end
end
