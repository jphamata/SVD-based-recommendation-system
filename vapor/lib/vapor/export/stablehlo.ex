defmodule Vapor.Export.StableHLO do
  @moduledoc """
  **A vapor program as a StableHLO module** — the portable front door of
  the accelerator compilers vapor does not write itself.

  Tenstorrent's compiler (`tt-mlir`, behind `tt-xla`'s PJRT plugin) takes
  StableHLO; so do XLA (CPU, GPU, TPU), IREE and the Neuron compiler. One
  exporter reaches all of them, instead of one hand-written backend per
  chip whose bindings this machine could not even compile. What vapor keeps
  is the part a vendor compiler cannot give: the airlock. The admission
  battery itself is exported (`Vapor.Substrate.Kit`), run on the target
  through PJRT, and judged here against the oracle and the rigorous
  envelope — so a Tensix result is admitted as canonical, as
  envelope-bound, or refused, by measurement.

  The translation is the oracle's semantics, operator by operator, with
  static shapes (dynamic extents at their maximum, or as given in `dims:`):

  | vapor | StableHLO |
  |---|---|
  | `add sub mul neg`, `fma` | `add subtract multiply negate`; `fma` as multiply then add (two roundings, as canonical) |
  | `relu`, `max`, `min`, `sel` | `compare` + `select` with vapor's NaN and signed-zero rules (`relu(NaN) = +0`, `max = a < b ? b : a`) |
  | `exp rcp rsqrt div sigmoid silu tanh log gelu_tanh` | `exponential`, `divide`, `rsqrt`, `logistic`, `tanh`, `log` (approximations of the vendor: within the measured tolerance, not bits) |
  | `reduce` | `reduce` over the last axis, extent kept as 1 |
  | `linear`, `linear_masked`, `linear_grouped` | `dot_general` (bf16 weights `convert`ed exactly), masks by `select` |
  | `gather_row`, `rope` | `gather` with indices clamped as vapor reads them (unsigned, so a negative one is the last row) — in signed arithmetic only, `slice`/`concatenate` |
  | `kv_write` | one `select` per written row, in order (last write wins, out-of-range rows skipped — `scatter` leaves both unspecified) |
  | `attention` | grouped `dot_general`s, the causal (and sliding-window) mask, max-subtracted softmax |
  | `transpose`, `reshape`, `gemm_i8` | `transpose`, `reshape`, `dot_general` i8 → i32 (wrap-around) |

  Refused, by name: `sample` (the decode loop stays in vapor), the 4-bit
  `qgemv` (export dequantized weights instead), the paged forms (export
  the contiguous program), and the exact `gelu` (StableHLO has no `erf`).
  """
  import Bitwise
  alias Vapor.{Program, Rejection, Tensor}
  alias Vapor.Algebra.Term

  @doc """
  `{:ok, %{mlir, inputs: [{name, dtype, shape}], outputs: [{name, dtype, shape}]}}`
  or `{:error, rejection}`. Options: `dims: %{sym => extent}`, or `env:`
  (tensors whose shapes bind the dynamic extents); otherwise every dynamic
  extent is exported at its maximum.
  """
  def export(%Program{} = p, opts \\ []) do
    dims = Map.merge(env_dims(p, Keyword.get(opts, :env, %{})), Keyword.get(opts, :dims, %{}))

    with :ok <- Program.check(p) |> ok(),
         :ok <- translatable(p) do
      inputs = for {:input, n, dt, s} <- Program.inputs(p), do: {n, dt, concrete(s, dims)}
      st = %{lines: [], n: 0, dims: dims, consts: %{}}
      env = inputs |> Enum.with_index() |> Map.new(fn {{n, dt, s}, i} -> {n, %{v: "%arg#{i}", dt: dt, shape: s}} end)

      try do
        {env, memo, st} =
          Enum.reduce(p.lets, {env, %{}, st}, fn
            {name, {:const, t}}, {env, memo, st} ->
              {v, st} = const(t, st)
              {Map.put(env, name, v), memo, st}

            {name, body}, {env, memo, st} ->
              {memo, st} = walk(body, memo, env, st)
              {Map.put(env, name, Map.fetch!(memo, body)), memo, st}
          end)

        {memo, st} = Enum.reduce(p.outputs, {memo, st}, fn {_, t}, {m, s} -> walk(t, m, env, s) end)
        outs = Enum.map(p.outputs, fn {name, t} -> {name, Map.fetch!(memo, t)} end)
        {:ok, %{mlir: module(inputs, outs, st), inputs: inputs, outputs: Enum.map(outs, fn {n, v} -> {n, v.dt, v.shape} end)}}
      catch
        {:reject, node, bound, repair} -> {:error, Rejection.new(node, bound, repair)}
      end
    end
  end

  # operators without a translation, refused before anything is emitted
  defp translatable(p) do
    case Enum.find(Program.order(p), &(elem(&1, 0) in [:sample, :qgemv, :qgemv_masked, :kv_write_paged, :attention_paged])) do
      nil ->
        :ok

      t ->
        try do
          refuse(t, elem(t, 0))
        catch
          {:reject, node, bound, repair} -> {:error, Rejection.new(node, bound, repair)}
        end
    end
  end

  defp env_dims(p, env) do
    for {:input, n, _, decl} <- Program.inputs(p), %Tensor{shape: actual} <- [env[n]], length(actual) == length(decl),
        {{:dyn, sym, _}, a} <- Enum.zip(decl, actual), into: %{}, do: {sym, a}
  end

  defp ok(:ok), do: :ok
  defp ok({:ok, _}), do: :ok
  defp ok(e), do: e

  defp concrete(shape, dims), do: Enum.map(shape, fn {:dyn, s, m} -> Map.get(dims, s, m); n -> n end)

  defp module(inputs, outs, st) do
    args = inputs |> Enum.with_index() |> Enum.map_join(", ", fn {{n, dt, s}, i} -> "%arg#{i}: #{ty(dt, s)} {vapor.name = \"#{n}\"}" end)
    rets = Enum.map_join(outs, ", ", fn {_, v} -> ty(v.dt, v.shape) end)
    body = st.lines |> Enum.reverse() |> Enum.map_join("", &("    " <> &1 <> "\n"))

    """
    // exported by vapor (Vapor.Export.StableHLO): inputs #{Enum.map_join(inputs, ", ", &elem(&1, 0))}; outputs #{Enum.map_join(outs, ", ", &elem(&1, 0))}
    module @vapor attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
      func.func public @main(#{args}) -> (#{rets}) {
    #{body}    return #{Enum.map_join(outs, ", ", fn {_, v} -> v.v end)} : #{rets}
      }
    }
    """
  end

  # ----------------------------------------------------------- traversal --

  defp walk(root, memo, env, st) do
    root
    |> Term.postorder()
    |> Enum.reduce({memo, st}, fn t, {m, s} ->
      if Map.has_key?(m, t) do
        {m, s}
      else
        {v, s} = node(t, m, env, s)
        {Map.put(m, t, v), s}
      end
    end)
  end

  defp node({:input, name, _, _}, _m, env, st), do: {Map.fetch!(env, name), st}
  defp node({:const, t}, _m, _env, st), do: const(t, st)

  defp node({:ew, op, args} = t, m, _env, st) do
    {:ok, {:f32, shape}} = Term.infer(t)
    shape = concrete(shape, st.dims)

    {vals, st} =
      Enum.map_reduce(args, st, fn
        {:splat, bits}, st -> splat(bits, shape, st)
        a, st -> bcast(Map.fetch!(m, a), shape, st)
      end)

    ew(op, vals, shape, t, st)
  end

  defp node({:reduce, op, x}, m, _env, st) do
    xv = Map.fetch!(m, x)
    r = length(xv.shape) - 1
    out_shape = List.replace_at(xv.shape, -1, 1)
    red_shape = Enum.drop(xv.shape, -1)
    {init, st} = scalar(if(op == :sum, do: 0, else: 0xFF80_0000), :f32, st)
    body = if op == :sum, do: "stablehlo.add", else: "stablehlo.maximum"
    {v, st} = emit("stablehlo.reduce(#{xv.v} init: #{init.v}) applies #{body} across dimensions = [#{r}] : (#{ty(:f32, xv.shape)}, tensor<f32>) -> #{ty(:f32, red_shape)}", :f32, red_shape, st)
    reshape(v, out_shape, st)
  end

  defp node({:linear, x, w}, m, _env, st) do
    {xv, wv} = {Map.fetch!(m, x), Map.fetch!(m, w)}
    {wv, st} = to_f32(wv, st)
    [n, _k] = wv.shape
    cdim = length(xv.shape) - 1
    out = List.replace_at(xv.shape, -1, n)
    emit("stablehlo.dot_general #{xv.v}, #{wv.v}, contracting_dims = [#{cdim}] x [1], precision = [HIGHEST, HIGHEST] : (#{ty(:f32, xv.shape)}, #{ty(:f32, wv.shape)}) -> #{ty(:f32, out)}", :f32, out, st)
  end

  defp node({:linear_masked, x, w, mk}, m, env, st) do
    {y, st} = node({:linear, x, w}, m, env, st)
    mv = Map.fetch!(m, mk)
    {z, st} = scalar(0, :f32, st)
    {mb, st} = bcast(mv, y.shape, st)
    {zb, st} = bcast(z, y.shape, st)
    {ne, st} = emit("stablehlo.compare NE, #{mb.v}, #{zb.v}, FLOAT : (#{ty(:f32, y.shape)}, #{ty(:f32, y.shape)}) -> #{ty(:i1, y.shape)}", :i1, y.shape, st)
    select(ne, y, zb, st)
  end

  defp node({:linear_grouped, x, w, g}, m, _env, st) do
    {xv, wv} = {Map.fetch!(m, x), Map.fetch!(m, w)}
    {wv, st} = to_f32(wv, st)
    [gn, k] = wv.shape
    n = div(gn, g)
    {xv, b} = if length(xv.shape) == 1, do: {xv, 1}, else: {xv, hd(xv.shape)}
    {x3, st} = reshape(xv, [b, g, k], st)
    {w3, st} = reshape(wv, [g, n, k], st)
    {y, st} = emit("stablehlo.dot_general #{x3.v}, #{w3.v}, batching_dims = [1] x [0], contracting_dims = [2] x [2], precision = [HIGHEST, HIGHEST] : (#{ty(:f32, [b, g, k])}, #{ty(:f32, [g, n, k])}) -> #{ty(:f32, [g, b, n])}", :f32, [g, b, n], st)
    {yt, st} = transpose(y, [1, 0, 2], st)
    reshape(yt, List.replace_at(Map.fetch!(m, x).shape, -1, gn), st)
  end

  defp node({:gemm_i8, a, w}, m, _env, st) do
    {av, wv} = {Map.fetch!(m, a), Map.fetch!(m, w)}
    [mm, _] = av.shape
    [n, _] = wv.shape
    emit("stablehlo.dot_general #{av.v}, #{wv.v}, contracting_dims = [1] x [1] : (#{ty(:s8, av.shape)}, #{ty(:s8, wv.shape)}) -> #{ty(:s32, [mm, n])}", :s32, [mm, n], st)
  end

  defp node({:gather_row, table, idx}, m, _env, st) do
    {tv, st} = to_f32(Map.fetch!(m, table), st)
    [v, d] = tv.shape
    {iv, st} = clamp_idx(Map.fetch!(m, idx), v, st)
    gather_rows(tv, iv, d, st)
  end

  defp node({:rope, x, cos, sin, pos, h}, m, _env, st) do
    xv = Map.fetch!(m, x)
    [n, w] = xv.shape
    dh = div(w, h)
    half = div(dh, 2)
    {cv, sv} = {Map.fetch!(m, cos), Map.fetch!(m, sin)}
    [s, ^half] = cv.shape
    {iv, st} = clamp_idx(Map.fetch!(m, pos), s, st)
    {c, st} = gather_rows(cv, iv, half, st)
    {sn, st} = gather_rows(sv, iv, half, st)
    {x3, st} = reshape(xv, [n, h, dh], st)
    {x1, st} = slice(x3, [0, 0, 0], [n, h, half], st)
    {x2, st} = slice(x3, [0, 0, half], [n, h, dh], st)
    {cb, st} = bcast_dims(c, [n, h, half], [0, 2], st)
    {sb, st} = bcast_dims(sn, [n, h, half], [0, 2], st)
    sh = [n, h, half]
    {a, st} = bin("multiply", x1, cb, sh, st)
    {b, st} = bin("multiply", x2, sb, sh, st)
    {o1, st} = bin("subtract", a, b, sh, st)
    {c2, st} = bin("multiply", x2, cb, sh, st)
    {d2, st} = bin("multiply", x1, sb, sh, st)
    {o2, st} = bin("add", c2, d2, sh, st)
    {cat, st} = emit("stablehlo.concatenate #{o1.v}, #{o2.v}, dim = 2 : (#{ty(:f32, sh)}, #{ty(:f32, sh)}) -> #{ty(:f32, [n, h, dh])}", :f32, [n, h, dh], st)
    reshape(cat, [n, w], st)
  end

  # rows in order: each row t replaces cache row pos[t] (if in range) — a
  # select per row, so a later write wins exactly as in the oracle
  defp node({:kv_write, cache, pos, rows}, m, _env, st) do
    cv = Map.fetch!(m, cache)
    [s, w] = cv.shape
    rv = Map.fetch!(m, rows)
    [n, ^w] = rv.shape
    pu = Map.fetch!(m, pos)
    {iota, st} = emit("stablehlo.iota dim = 0 : #{ty(:s32, [s, w])}", :s32, [s, w], st)

    Enum.reduce(0..(n - 1)//1, {cv, st}, fn t, {acc, st} ->
      {p1, st} = slice(pu, [t], [t + 1], st)
      {p0, st} = reshape(p1, [], st)
      {pb, st} = bcast(p0, [s, w], st)
      {r1, st} = slice(rv, [t, 0], [t + 1, w], st)
      {rb, st} = bcast_dims(r1, [s, w], [0, 1], st)
      # a negative or too large position equals no row: the write is skipped
      {eq, st} = emit("stablehlo.compare EQ, #{iota.v}, #{pb.v}, SIGNED : (#{ty(:s32, [s, w])}, #{ty(:s32, [s, w])}) -> #{ty(:i1, [s, w])}", :i1, [s, w], st)
      select(eq, rb, acc, st)
    end)
  end

  defp node({:attention, q, k, v, pos, heads}, m, _env, st) do
    {h, hkv} = Term.heads_of(heads)
    win = Term.attention_window(heads)
    qv = Map.fetch!(m, q)
    [n, w] = qv.shape
    dh = div(w, h)
    g = div(h, hkv)
    kv = Map.fetch!(m, k)
    [s, _] = kv.shape
    scale_bits = Term.attention_scale_bits(heads, dh)

    {q4, st} = reshape(qv, [n, hkv, g, dh], st)
    {qt, st} = transpose(q4, [1, 2, 0, 3], st)
    {k3, st} = reshape(kv, [s, hkv, dh], st)
    {kt, st} = transpose(k3, [1, 0, 2], st)
    {v3, st} = reshape(Map.fetch!(m, v), [s, hkv, dh], st)
    {vt, st} = transpose(v3, [1, 0, 2], st)
    sc = [hkv, g, n, s]
    {raw, st} = emit("stablehlo.dot_general #{qt.v}, #{kt.v}, batching_dims = [0] x [0], contracting_dims = [3] x [2], precision = [HIGHEST, HIGHEST] : (#{ty(:f32, [hkv, g, n, dh])}, #{ty(:f32, [hkv, s, dh])}) -> #{ty(:f32, sc)}", :f32, sc, st)
    {scv, st} = splat(scale_bits, sc, st)
    {scores, st} = bin("multiply", raw, scv, sc, st)

    # keys 0 … min(pos, S−1) (or the last `win` of them)
    {pc, st} = clamp_idx(Map.fetch!(m, pos), s, st)
    {pb, st} = bcast_dims(pc, sc, [2], st)
    {iota, st} = emit("stablehlo.iota dim = 3 : #{ty(:s32, sc)}", :s32, sc, st)
    {le, st} = emit("stablehlo.compare LE, #{iota.v}, #{pb.v}, SIGNED : (#{ty(:s32, sc)}, #{ty(:s32, sc)}) -> #{ty(:i1, sc)}", :i1, sc, st)

    {valid, st} =
      if win do
        # j ≥ p − w + 1  ⇔  j + w > p
        {wv, st} = scalar(win, :s32, st)
        {wb, st} = bcast(wv, sc, st)
        {jw, st} = emit("stablehlo.add #{iota.v}, #{wb.v} : #{ty(:s32, sc)}", :s32, sc, st)
        {gt, st} = emit("stablehlo.compare GT, #{jw.v}, #{pb.v}, SIGNED : (#{ty(:s32, sc)}, #{ty(:s32, sc)}) -> #{ty(:i1, sc)}", :i1, sc, st)
        emit("stablehlo.and #{le.v}, #{gt.v} : #{ty(:i1, sc)}", :i1, sc, st)
      else
        {le, st}
      end

    {ninf, st} = splat(0xFF80_0000, sc, st)
    {masked, st} = select(valid, scores, ninf, st)
    rs = [hkv, g, n]
    {mx0, st} = scalar(0xFF80_0000, :f32, st)
    {mx, st} = emit("stablehlo.reduce(#{masked.v} init: #{mx0.v}) applies stablehlo.maximum across dimensions = [3] : (#{ty(:f32, sc)}, tensor<f32>) -> #{ty(:f32, rs)}", :f32, rs, st)
    {mxb, st} = bcast_dims(mx, sc, [0, 1, 2], st)
    {d, st} = bin("subtract", masked, mxb, sc, st)
    {e, st} = emit("stablehlo.exponential #{d.v} : #{ty(:f32, sc)}", :f32, sc, st)
    {z0, st} = scalar(0, :f32, st)
    {zs, st} = emit("stablehlo.reduce(#{e.v} init: #{z0.v}) applies stablehlo.add across dimensions = [3] : (#{ty(:f32, sc)}, tensor<f32>) -> #{ty(:f32, rs)}", :f32, rs, st)
    {one, st} = splat(0x3F80_0000, rs, st)
    {inv, st} = bin("divide", one, zs, rs, st)
    {invb, st} = bcast_dims(inv, sc, [0, 1, 2], st)
    {pr, st} = bin("multiply", e, invb, sc, st)
    {o, st} = emit("stablehlo.dot_general #{pr.v}, #{vt.v}, batching_dims = [0] x [0], contracting_dims = [3] x [1], precision = [HIGHEST, HIGHEST] : (#{ty(:f32, sc)}, #{ty(:f32, [hkv, s, dh])}) -> #{ty(:f32, [hkv, g, n, dh])}", :f32, [hkv, g, n, dh], st)
    {ot, st} = transpose(o, [2, 0, 1, 3], st)
    reshape(ot, [n, w], st)
  end

  defp node({:transpose, x}, m, _env, st), do: transpose(Map.fetch!(m, x), [1, 0], st)
  defp node({:reshape, x, shape}, m, _env, st), do: reshape(Map.fetch!(m, x), shape, st)

  defp node(t, _m, _env, _st) when elem(t, 0) in [:sample, :qgemv, :qgemv_masked, :kv_write_paged, :attention_paged], do: refuse(t, elem(t, 0))

  defp refuse(t, op) do
    repair =
      case op do
        :sample -> "export the logits; sampling stays in vapor's decode loop"
        q when q in [:qgemv, :qgemv_masked] -> "export dequantized (f32 or bf16) weights"
        _ -> "export the contiguous program (kv_write + attention)"
      end

    throw({:reject, t, "an operator with a StableHLO translation (#{op} has none here)", repair})
  end

  # ------------------------------------------------------- elementwise ops --

  defp ew(:add, [a, b], sh, _t, st), do: bin("add", a, b, sh, st)
  defp ew(:sub, [a, b], sh, _t, st), do: bin("subtract", a, b, sh, st)
  defp ew(:mul, [a, b], sh, _t, st), do: bin("multiply", a, b, sh, st)
  defp ew(:div, [a, b], sh, _t, st), do: bin("divide", a, b, sh, st)
  defp ew(:neg, [a], sh, _t, st), do: un("negate", a, sh, st)
  defp ew(:exp, [a], sh, _t, st), do: un("exponential", a, sh, st)
  defp ew(:log, [a], sh, _t, st), do: un("log", a, sh, st)
  defp ew(:rsqrt, [a], sh, _t, st), do: un("rsqrt", a, sh, st)
  defp ew(:tanh, [a], sh, _t, st), do: un("tanh", a, sh, st)
  defp ew(:sigmoid, [a], sh, _t, st), do: un("logistic", a, sh, st)

  defp ew(:fma, [a, b, c], sh, _t, st) do
    {p, st} = bin("multiply", a, b, sh, st)
    bin("add", p, c, sh, st)
  end

  defp ew(:rcp, [a], sh, _t, st) do
    {one, st} = splat(0x3F80_0000, sh, st)
    bin("divide", one, a, sh, st)
  end

  defp ew(:silu, [a], sh, _t, st) do
    {s, st} = un("logistic", a, sh, st)
    bin("multiply", a, s, sh, st)
  end

  # relu(x) = x if x > 0 else +0 (NaN and −0 give +0)
  defp ew(:relu, [a], sh, _t, st) do
    {z, st} = splat(0, sh, st)
    {gt, st} = cmp("GT", a, z, sh, st)
    select(gt, a, z, st)
  end

  defp ew(:max, [a, b], sh, _t, st), do: sel_lt(a, b, b, a, sh, st)
  defp ew(:min, [a, b], sh, _t, st), do: sel_lt(b, a, b, a, sh, st)
  defp ew(:sel, [a, b, x, y], sh, _t, st), do: sel_lt(a, b, x, y, sh, st)
  defp ew(:sel_lt, [a, b, x, y], sh, _t, st), do: sel_lt(a, b, x, y, sh, st)

  # 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
  defp ew(:gelu_tanh, [x], sh, _t, st) do
    {c1, st} = splat(Vapor.F32.from_float(0.044715), sh, st)
    {c2, st} = splat(Vapor.F32.from_float(:math.sqrt(2 / :math.pi())), sh, st)
    {half, st} = splat(0x3F00_0000, sh, st)
    {one, st} = splat(0x3F80_0000, sh, st)
    {x2, st} = bin("multiply", x, x, sh, st)
    {x3, st} = bin("multiply", x2, x, sh, st)
    {a, st} = bin("multiply", x3, c1, sh, st)
    {b, st} = bin("add", x, a, sh, st)
    {c, st} = bin("multiply", b, c2, sh, st)
    {t, st} = un("tanh", c, sh, st)
    {u, st} = bin("add", one, t, sh, st)
    {v, st} = bin("multiply", x, u, sh, st)
    bin("multiply", v, half, sh, st)
  end

  defp ew(op, _vals, _sh, t, _st),
    do: throw({:reject, t, "an elementwise operator with a StableHLO translation (#{op} has none: StableHLO has no erf)", "use gelu_tanh, or keep this node in vapor"})

  defp sel_lt(a, b, x, y, sh, st) do
    {lt, st} = cmp("LT", a, b, sh, st)
    select(lt, x, y, st)
  end

  # ------------------------------------------------------------- builders --

  defp emit(rhs, dt, shape, st) do
    name = "%v#{st.n}"
    {%{v: name, dt: dt, shape: shape}, %{st | lines: ["#{name} = #{rhs}" | st.lines], n: st.n + 1}}
  end

  defp bin(op, a, b, sh, st), do: emit("stablehlo.#{op} #{a.v}, #{b.v} : #{ty(:f32, sh)}", :f32, sh, st)
  defp un(op, a, sh, st), do: emit("stablehlo.#{op} #{a.v} : #{ty(:f32, sh)}", :f32, sh, st)
  defp cmp(dir, a, b, sh, st), do: emit("stablehlo.compare #{dir}, #{a.v}, #{b.v}, FLOAT : (#{ty(:f32, sh)}, #{ty(:f32, sh)}) -> #{ty(:i1, sh)}", :i1, sh, st)

  defp select(c, a, b, st),
    do: emit("stablehlo.select #{c.v}, #{a.v}, #{b.v} : #{ty(:i1, a.shape)}, #{ty(a.dt, a.shape)}", a.dt, a.shape, st)

  defp reshape(%{shape: s} = v, s, st), do: {v, st}
  defp reshape(v, shape, st), do: emit("stablehlo.reshape #{v.v} : (#{ty(v.dt, v.shape)}) -> #{ty(v.dt, shape)}", v.dt, shape, st)

  defp transpose(v, perm, st) do
    out = Enum.map(perm, &Enum.at(v.shape, &1))
    emit("stablehlo.transpose #{v.v}, dims = [#{Enum.join(perm, ", ")}] : (#{ty(v.dt, v.shape)}) -> #{ty(v.dt, out)}", v.dt, out, st)
  end

  defp slice(v, start, limit, st) do
    out = Enum.zip_with(start, limit, &(&2 - &1))
    emit("stablehlo.slice #{v.v} [#{Enum.zip_with(start, limit, &"#{&1}:#{&2}") |> Enum.join(", ")}] : (#{ty(v.dt, v.shape)}) -> #{ty(v.dt, out)}", v.dt, out, st)
  end

  # equal-rank broadcast (extent-1 axes stretch) or a scalar to a shape
  defp bcast(%{shape: s} = v, s, st), do: {v, st}
  defp bcast(%{shape: []} = v, shape, st), do: bcast_dims(v, shape, [], st)
  defp bcast(v, shape, st), do: bcast_dims(v, shape, Enum.to_list(0..(length(shape) - 1)), st)

  defp bcast_dims(v, shape, dims, st),
    do: emit("stablehlo.broadcast_in_dim #{v.v}, dims = [#{Enum.join(dims, ", ")}] : (#{ty(v.dt, v.shape)}) -> #{ty(v.dt, shape)}", v.dt, shape, st)

  defp splat(bits, shape, st) do
    {s, st} = scalar(bits, :f32, st)
    bcast(s, shape, st)
  end

  # a scalar constant (f32 as exact bits, integers as themselves), interned
  defp scalar(val, dt, st) do
    key = {dt, val}

    case st.consts do
      %{^key => v} ->
        {v, st}

      _ ->
        lit = if dt == :f32, do: "0x" <> (val |> Integer.to_string(16) |> String.pad_leading(8, "0")), else: Integer.to_string(val)
        {v, st} = emit("stablehlo.constant dense<#{lit}> : tensor<#{elem_ty(dt)}>", dt, [], st)
        {v, %{st | consts: Map.put(st.consts, key, v)}}
    end
  end

  defp const(%Tensor{dtype: dt, shape: s, data: d}, st) when dt in [:f32, :bf16, :s32, :s8, :u8, :f16] do
    emit("stablehlo.constant dense<\"0x#{Base.encode16(d)}\"> : #{ty(dt, s)}", dt, s, st)
  end

  defp const(%Tensor{dtype: dt} = t, _st), do: throw({:reject, {:const, t}, "a constant of a plain dtype (#{dt} is vapor's own)", "dequantize before exporting"})

  # bf16 (and f16) operands widen exactly to f32
  defp to_f32(%{dt: :f32} = v, st), do: {v, st}
  defp to_f32(v, st), do: emit("stablehlo.convert #{v.v} : (#{ty(v.dt, v.shape)}) -> #{ty(:f32, v.shape)}", :f32, v.shape, st)

  # vapor reads an index as unsigned and clamps it to the table: a negative
  # one is huge, hence the last row. Here in signed arithmetic only (no
  # unsigned types, which not every StableHLO consumer lowers):
  # i < 0 ? n − 1 : min(i, n − 1)
  defp clamp_idx(iv, rows, st) do
    {mx, st} = scalar(rows - 1, :s32, st)
    {mb, st} = bcast(mx, iv.shape, st)
    {z, st} = scalar(0, :s32, st)
    {zb, st} = bcast(z, iv.shape, st)
    {lo, st} = emit("stablehlo.minimum #{iv.v}, #{mb.v} : #{ty(:s32, iv.shape)}", :s32, iv.shape, st)
    {neg, st} = emit("stablehlo.compare LT, #{iv.v}, #{zb.v}, SIGNED : (#{ty(:s32, iv.shape)}, #{ty(:s32, iv.shape)}) -> #{ty(:i1, iv.shape)}", :i1, iv.shape, st)
    select(neg, mb, lo, st)
  end

  defp gather_rows(tv, iv, d, st) do
    [n] = iv.shape
    {i2, st} = reshape(iv, [n, 1], st)

    emit(~s|"stablehlo.gather"(#{tv.v}, #{i2.v}) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, #{d}>} : (#{ty(tv.dt, tv.shape)}, #{ty(:s32, [n, 1])}) -> #{ty(tv.dt, [n, d])}|,
         tv.dt, [n, d], st)
  end

  defp ty(dt, []), do: "tensor<#{elem_ty(dt)}>"
  defp ty(dt, shape), do: "tensor<#{Enum.join(shape, "x")}x#{elem_ty(dt)}>"

  defp elem_ty(:f32), do: "f32"
  defp elem_ty(:bf16), do: "bf16"
  defp elem_ty(:f16), do: "f16"
  defp elem_ty(:s32), do: "i32"
  defp elem_ty(:u32), do: "ui32"
  defp elem_ty(:s8), do: "i8"
  defp elem_ty(:u8), do: "ui8"
  defp elem_ty(:i1), do: "i1"

  @doc false
  def mask32(x), do: x &&& 0xFFFF_FFFF
end
