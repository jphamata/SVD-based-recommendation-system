defmodule Vapor.Merge do
  @moduledoc """
  **Model fusion** in weight space, deterministic and with a receipt.

  Methods (the vocabulary of mergekit, with its definitions, plus one
  closed form):

  | method | result per tensor |
  |---|---|
  | `:linear` | `Σ wᵢ θᵢ / Σ wᵢ` (`normalize: false` keeps the raw sum) |
  | `:task_arithmetic` | `base + λ Σ wᵢ (θᵢ − base)` |
  | `:slerp` | spherical interpolation of two models at `t` (linear when the tensors are almost parallel, `|cos| > 0.9995`) |
  | `:ties` | task vectors trimmed to the top `density` by magnitude, sign elected by `sign(Σ wᵢτᵢ)`, disjoint weighted mean of the agreeing entries, `base + λ·τ` |
  | `:dare_linear` / `:dare_ties` | task vectors with each entry dropped with probability `1 − density` and the rest rescaled by `1/density`, then linear / TIES |
  | `:regmean` | for every matrix whose inputs were measured (`calibrate/3`): the **least-squares** fusion `W = (Σ Wᵢ G̃ᵢ)(Σ G̃ᵢ)⁻¹`, `G̃ = α·G + (1 − α)·diag G`, `G = XᵀX` of the matrix's input activations on each model's own data; every other tensor `:linear` |

  `:regmean` is the answer to "TIES and DARE made it worse" (measured,
  `docs/FUSAO.md`): they assume small, sparse fine-tune deltas; fusing
  models that differ densely needs to know *which inputs each model is
  responsible for*, and the Gram matrix of its activations is exactly
  that. `diagnose/2` measures which regime a set of models is in before
  anything is fused.

  What makes it fit the rest of vapor:

    * **airlock-checked compatibility** — every input is a `{spec, weights}`
      admitted by `Vapor.Lock`; they must share the adapter, the contract
      and every tensor's name and shape, and (unless `allow_config_mismatch:
      true`) the configuration digest;
    * **the same bits on every host** — arithmetic in binary64 `+ − × ÷ √`
      (correctly rounded), angles through `Vapor.CR` (no libm), DARE's
      drops from splitmix64 keyed by `(seed, model, tensor)` (a counter, so
      any block can start at its offset), one rounding to binary32 per
      output entry; the parallel schedule never changes a bit (every entry
      is computed from the same operands in the same order);
    * **a receipt** — a `Vapor.Certificate` whose payload names the method,
      its parameters, the inputs' configuration digests and Merkle roots of
      their weights (and, for `:regmean`, the digest of the calibration
      Grams), and the output's root.

  Throughput: tensors are cut into blocks fused on every scheduler, each
  block by a binary-matching kernel (no lists): ~10 M entries/s per core
  for linear, task arithmetic and SLERP (`docs/bench`), against 0.6 M/s in
  0.4.0.
  """
  alias Vapor.{Certificate, CR, Linalg, Merkle, Rejection, Tensor}
  alias Vapor.Lock.Spec

  @methods [:linear, :task_arithmetic, :slerp, :ties, :dare_linear, :dare_ties, :regmean]
  def methods, do: @methods

  @golden 0x9E3779B97F4A7C15
  @mask64 0xFFFFFFFFFFFFFFFF

  @doc """
  Fuse `models` (a list of `%{spec, weights}` or `{spec, weights}`).
  Options: `method` (`:linear`), `weights` (per model, default equal),
  `base` (a `%{spec, weights}`, required by task-vector methods), `lambda`
  (1.0), `t` (SLERP, 0.5), `density` (TIES/DARE, 0.5), `seed` (DARE, 0),
  `normalize` (linear, true), `grams` (`:regmean`: one `calibrate/3`
  result per model), `alpha` (`:regmean`, 0.9), `ridge` (`:regmean`,
  1.0e-4 of the mean Gram diagonal, toward the weighted mean), `key` (an Ed25519 key to
  sign the receipt), `allow_config_mismatch`, `concurrency` (schedulers).

  Returns `{:ok, %{spec, weights, receipt}}`.
  """
  def merge(models, opts \\ []) do
    models = Enum.map(models, &pair/1)
    method = Keyword.get(opts, :method, :linear)
    base = opts[:base] && pair(opts[:base])
    all = if base, do: [base | models], else: models

    with :ok <- need(method in @methods, :method, "one of #{inspect(@methods)}"),
         :ok <- need(models != [], :models, "at least one model"),
         :ok <- need(method != :slerp or length(models) == 2, :models, "exactly two models for :slerp"),
         :ok <- need(method not in [:task_arithmetic, :ties, :dare_linear, :dare_ties] or base != nil, :base, "a base model for #{method}"),
         :ok <- need(method != :regmean or (is_list(opts[:grams]) and length(opts[:grams]) == length(models)), :grams,
                     "one calibration (Vapor.Merge.calibrate/3) per model for :regmean"),
         :ok <- compatible(all, opts),
         ws = Keyword.get(opts, :weights, List.duplicate(1.0, length(models))),
         :ok <- need(length(ws) == length(models) and Enum.all?(ws, &is_number/1), :weights, "one number per model") do
      {spec, first} = hd(models)
      names = first |> Map.keys() |> Enum.filter(&is_binary/1) |> Enum.sort()
      params = params(method, ws, opts)
      conc = Keyword.get(opts, :concurrency, System.schedulers_online())

      try do
        merged = fuse_all(method, names, models, base, params, opts, conc)
        merged = Map.merge(Map.reject(first, fn {k, _} -> is_binary(k) end), merged)
        {:ok, %{spec: spec, weights: merged, receipt: receipt(method, params, models, base, merged, opts)}}
      catch
        {:nonfinite, name} -> reject({:weight, name}, "finite weights (#{name} holds a NaN or an infinity)")
        {:singular, name} -> reject({:grams, name}, "positive definite Grams (#{name}: raise alpha's diagonal share or calibrate on more tokens)")
      end
    end
  end

  defp pair(%{spec: s, weights: w}), do: {s, w}
  defp pair({%Spec{}, _} = p), do: p

  defp params(method, ws, opts) do
    %{method: method, weights: Enum.map(ws, &(&1 * 1.0)), lambda: Keyword.get(opts, :lambda, 1.0) * 1.0,
      t: Keyword.get(opts, :t, 0.5) * 1.0, density: Keyword.get(opts, :density, 0.5) * 1.0,
      seed: Keyword.get(opts, :seed, 0), normalize: Keyword.get(opts, :normalize, true),
      alpha: Keyword.get(opts, :alpha, 0.9) * 1.0, ridge: Keyword.get(opts, :ridge, 1.0e-4) * 1.0}
  end

  # ------------------------------------------------------------ compatibility --

  defp compatible([{s0, w0} | rest], opts) do
    shapes = fn w -> for {k, %Tensor{shape: s}} <- w, is_binary(k), into: %{}, do: {k, s} end
    sh0 = shapes.(w0)

    Enum.reduce_while(rest, :ok, fn {s, w}, :ok ->
      cond do
        s.adapter != s0.adapter or s.interface != s0.interface ->
          {:halt, reject(:spec, "the same adapter and contract (#{inspect(s0.adapter)} #{s0.interface} vs #{inspect(s.adapter)} #{s.interface})")}

        s.digest != s0.digest and not Keyword.get(opts, :allow_config_mismatch, false) ->
          {:halt, reject(:config, "the same configuration digest (#{s0.family} vs #{s.family}); allow_config_mismatch: true to fuse anyway")}

        shapes.(w) != sh0 ->
          diff = Map.keys(sh0) |> Kernel.++(Map.keys(shapes.(w))) |> Enum.uniq() |> Enum.find(&(sh0[&1] != shapes.(w)[&1]))
          {:halt, reject({:weight, diff}, "the same tensor names and shapes in every model")}

        true ->
          {:cont, :ok}
      end
    end)
  end

  # ------------------------------------------------------------------ fusion --
  #
  # Every float tensor is cut into blocks of @block entries; whole-tensor
  # statistics (SLERP's angle, TIES's magnitude cut) are computed first, per
  # tensor, in parallel; then every (tensor, block) is fused in parallel by a
  # binary kernel. Blocks are independent by construction (DARE's generator
  # is a counter), so the schedule cannot change a bit.

  @block 262_144

  defp fuse_all(method, names, models, base, p, opts, conc) do
    grams = opts[:grams]

    {float, other} =
      Enum.split_with(names, fn name -> match?(%Tensor{dtype: dt} when dt in [:f32, :bf16, :f16], elem(hd(models), 1)[name]) end)

    copied =
      Map.new(other, fn name ->
        ts = Enum.map(models, fn {_, w} -> w[name] end)
        if Enum.all?(ts, &(&1 == hd(ts))), do: {name, hd(ts)}, else: raise(ArgumentError, "#{name}: non-float tensors differ")
      end)

    # tensors fused by least squares (when every model measured their inputs)
    {lsq, elementwise} =
      if method == :regmean,
        do: Enum.split_with(float, fn name -> Enum.all?(grams, &Map.has_key?(&1.grams, name)) end),
        else: {[], float}

    ew_method = if method == :regmean, do: :linear, else: method

    jobs = pmap(elementwise, conc, &prepare_tensor(ew_method, &1, models, base, p))

    blocks =
      for {job, ji} <- Enum.with_index(jobs), k <- 0..max(div(job.n - 1, @block), 0)//1, do: {ji, k}

    jobs_t = List.to_tuple(jobs)

    out =
      blocks
      |> pmap(conc, fn {ji, k} ->
        job = elem(jobs_t, ji)
        off = k * @block
        len = min(@block, job.n - off)
        {ji, kernel(ew_method, job, off, len, p)}
      end)
      |> Enum.reduce(%{}, fn {ji, bin}, acc -> Map.update(acc, ji, [bin], &[bin | &1]) end)

    fused =
      jobs
      |> Enum.with_index()
      |> Map.new(fn {job, ji} -> {job.name, Tensor.new(:f32, job.shape, out |> Map.get(ji, []) |> Enum.reverse() |> IO.iodata_to_binary())} end)

    least = lsq |> pmap(conc, &{&1, regmean_tensor(&1, models, grams, p)}) |> Map.new()

    copied |> Map.merge(fused) |> Map.merge(least)
  end

  # Task.async_stream whose throws come back to the caller (a refusal raised
  # deep in a kernel must become the merge's {:error, rejection})
  defp pmap(items, conc, f) do
    items
    |> Task.async_stream(fn x -> try do {:ok, f.(x)} catch v -> {:thrown, v} end end,
         max_concurrency: conc, timeout: :infinity, ordered: true)
    |> Enum.map(fn
      {:ok, {:ok, v}} -> v
      {:ok, {:thrown, v}} -> throw(v)
    end)
  end

  defp prepare_tensor(method, name, models, base, p) do
    ts = Enum.map(models, fn {_, w} -> w[name] end)
    xs = Enum.map(ts, &f32_data/1)
    b = base && f32_data(elem(base, 1)[name])
    n = div(byte_size(hd(xs)), 4)
    stats = stats(method, name, xs, b, n, p)
    %{name: name, shape: hd(ts).shape, xs: xs, b: b, n: n, stats: stats}
  end

  # f32 storage for every float dtype (bf16 → f32 is exact; f16 too)
  defp f32_data(%Tensor{dtype: :f32, data: d}), do: d
  defp f32_data(%Tensor{dtype: :bf16} = t), do: Tensor.widen(t).data
  defp f32_data(%Tensor{dtype: :f16, data: d}), do: for(<<x::float-16-little <- d>>, into: <<>>, do: <<x::float-32-little>>)

  defp part(bin, off, len), do: binary_part(bin, off * 4, len * 4)

  # whole-tensor statistics
  defp stats(:slerp, name, [a, b], _base, _n, %{t: t}) do
    {aa, bb, ab} = dots(a, b, 0.0, 0.0, 0.0, name)
    %{coeffs: slerp_coeffs(:math.sqrt(aa), :math.sqrt(bb), ab, t)}
  end

  defp stats(:ties, name, xs, b, n, %{density: d}),
    do: %{cuts: Enum.map(xs, fn x -> if d >= 1.0, do: :all, else: cut(x, b, n, round(d * n), name) end)}

  defp stats(m, name, xs, _b, _n, %{seed: seed}) when m in [:dare_linear, :dare_ties],
    do: %{keys: xs |> Enum.with_index() |> Enum.map(fn {_, i} -> Vapor.Modal.Rng.key({:dare, seed, i, name}) end)}

  defp stats(_m, _name, _xs, _b, _n, _p), do: %{}

  defp dots(<<x::float-32-little, ra::binary>>, <<y::float-32-little, rb::binary>>, aa, bb, ab, name),
    do: dots(ra, rb, aa + x * x, bb + y * y, ab + x * y, name)

  defp dots(<<>>, <<>>, aa, bb, ab, _name), do: {aa, bb, ab}
  defp dots(_, _, _, _, _, name), do: throw({:nonfinite, name})

  # The exact top-k magnitude cut of a task vector: a histogram of |τ| over
  # the top 16 bits of its binary64 pattern (for non-negative values the
  # patterns are ordered as integers), then only the boundary bucket's
  # entries sorted by (|τ| desc, index asc). Keeps every entry above the
  # boundary bucket and the first r inside it — the same set as sorting the
  # whole tensor. Bucket tests become float comparisons against the bucket's
  # edges (bucket(v) > bd ⇔ v ≥ edge(bd + 1)).
  defp cut(x, b, n, k, name) do
    hist = :counters.new(65_536, [])
    hist_pass(x, b, hist, name)

    {boundary, above} =
      Enum.reduce_while(65_535..0//-1, 0, fn bk, acc ->
        c = :counters.get(hist, bk + 1)
        if acc + c >= k, do: {:halt, {bk, acc}}, else: {:cont, acc + c}
      end)
      |> case do
        {bk, acc} -> {bk, acc}
        _ -> {-1, n}
      end

    if boundary < 0 do
      :all
    else
      edge = fn bk -> if bk <= 0x7FEF, do: (<<v::float-64>> = <<Bitwise.bsl(bk, 48)::64>>; v), else: nil end
      {lo, hi} = {edge.(boundary), edge.(boundary + 1)}
      inside = collect(x, b, lo, hi, 0, [])
      keep = inside |> Enum.sort_by(fn {v, i} -> {-v, i} end) |> Enum.take(max(k - above, 0)) |> MapSet.new(&elem(&1, 1))
      %{lo: lo, hi: hi, keep: keep}
    end
  end

  defp hist_pass(<<x::float-32-little, ra::binary>>, <<y::float-32-little, rb::binary>>, hist, name) do
    <<bk::16, _::48>> = <<abs(x - y)::float-64>>
    :counters.add(hist, bk + 1, 1)
    hist_pass(ra, rb, hist, name)
  end

  defp hist_pass(<<>>, <<>>, _hist, _name), do: :ok
  defp hist_pass(_, _, _, name), do: throw({:nonfinite, name})

  defp collect(<<x::float-32-little, ra::binary>>, <<y::float-32-little, rb::binary>>, lo, hi, i, acc) do
    v = abs(x - y)
    acc = if lo != nil and v >= lo and (hi == nil or v < hi), do: [{v, i} | acc], else: acc
    collect(ra, rb, lo, hi, i + 1, acc)
  end

  defp collect(<<>>, <<>>, _lo, _hi, _i, acc), do: acc

  defp keep?(:all, _v, _i), do: true

  defp keep?(%{lo: lo, hi: hi, keep: keep}, v, i) do
    a = abs(v)

    cond do
      hi != nil and a >= hi -> true
      lo != nil and a >= lo -> MapSet.member?(keep, i)
      true -> false
    end
  end

  # ----------------------------------------------------------------- kernels --
  #
  # One block of one tensor. Every kernel computes each entry from the same
  # operands in the same order as the definition (and as 0.4.0's list code:
  # the sums start at 0.0, terms added in model order), rounding once to
  # binary32 — so the outputs are the 0.4.0 bits (tested).

  defp kernel(:linear, %{xs: xs, stats: _, name: name}, off, len, %{weights: ws, normalize: norm}) do
    tot = if norm, do: Enum.sum(ws), else: 1.0

    case {Enum.map(xs, &part(&1, off, len)), ws} do
      {[a, b], [w0, w1]} -> lin2(a, b, w0, w1, tot, name, <<>>)
      {[a], [w0]} -> lin1(a, w0, tot, name, <<>>)
      {cols, ws} -> linn(cols, ws, tot, name, <<>>)
    end
  end

  defp kernel(:task_arithmetic, %{xs: xs, b: b, name: name}, off, len, %{weights: ws, lambda: l}) do
    case {Enum.map(xs, &part(&1, off, len)), ws} do
      {[x0, x1], [w0, w1]} -> ta2(x0, x1, part(b, off, len), w0, w1, l, name, <<>>)
      {cols, ws} -> tan(cols, part(b, off, len), ws, l, name, <<>>)
    end
  end

  defp kernel(:slerp, %{xs: [a, b], stats: %{coeffs: {s0, s1}}, name: name}, off, len, _p),
    do: lin2(part(a, off, len), part(b, off, len), s0, s1, 1.0, name, <<>>)

  defp kernel(:ties, %{xs: xs, b: b, stats: %{cuts: cuts}, name: name}, off, len, %{weights: ws, lambda: l}),
    do: ties_k(Enum.map(xs, &part(&1, off, len)), part(b, off, len), cuts, ws, l, off, name, <<>>)

  defp kernel(m, %{xs: xs, b: b, stats: %{keys: keys}, name: name}, off, len, %{weights: ws, lambda: l, density: d})
       when m in [:dare_linear, :dare_ties] do
    # splitmix64's state after `off` draws is key + off·γ: any block starts at its offset
    states = Enum.map(keys, &Bitwise.band(&1 + off * @golden, @mask64))
    dare_k(m, Enum.map(xs, &part(&1, off, len)), part(b, off, len), states, ws, l, d, name, <<>>)
  end

  defp lin1(<<a::float-32-little, ra::binary>>, w0, tot, name, acc), do: lin1(ra, w0, tot, name, <<acc::binary, (0.0 + w0 * a) / tot::float-32-little>>)
  defp lin1(<<>>, _w0, _tot, _name, acc), do: acc
  defp lin1(_, _, _, name, _), do: throw({:nonfinite, name})

  defp lin2(<<a::float-32-little, ra::binary>>, <<b::float-32-little, rb::binary>>, w0, w1, tot, name, acc),
    do: lin2(ra, rb, w0, w1, tot, name, <<acc::binary, (0.0 + w0 * a + w1 * b) / tot::float-32-little>>)

  defp lin2(<<>>, <<>>, _w0, _w1, _tot, _name, acc), do: acc
  defp lin2(_, _, _, _, _, name, _), do: throw({:nonfinite, name})

  defp linn(cols, ws, tot, name, acc) do
    case heads(cols, [], []) do
      :done -> acc
      {vals, rest} -> linn(rest, ws, tot, name, <<acc::binary, wsum(vals, ws, 0.0) / tot::float-32-little>>)
      :bad -> throw({:nonfinite, name})
    end
  end

  # the first entry of every column, and the rests
  defp heads([<<x::float-32-little, r::binary>> | cs], vs, rs), do: heads(cs, [x | vs], [r | rs])
  defp heads([], vs, rs), do: {:lists.reverse(vs), :lists.reverse(rs)}
  defp heads([<<>> | _], [], _), do: :done
  defp heads(_, _, _), do: :bad

  defp wsum([x | xs], [w | ws], s), do: wsum(xs, ws, s + w * x)
  defp wsum([], [], s), do: s

  defp ta2(<<x0::float-32-little, r0::binary>>, <<x1::float-32-little, r1::binary>>, <<y::float-32-little, rb::binary>>, w0, w1, l, name, acc),
    do: ta2(r0, r1, rb, w0, w1, l, name, <<acc::binary, y + l * (0.0 + w0 * (x0 - y) + w1 * (x1 - y))::float-32-little>>)

  defp ta2(<<>>, <<>>, <<>>, _w0, _w1, _l, _name, acc), do: acc
  defp ta2(_, _, _, _, _, _, name, _), do: throw({:nonfinite, name})

  defp tan(cols, <<y::float-32-little, rb::binary>>, ws, l, name, acc) do
    case heads(cols, [], []) do
      {vals, rest} -> tan(rest, rb, ws, l, name, <<acc::binary, y + l * wsum(Enum.map(vals, &(&1 - y)), ws, 0.0)::float-32-little>>)
      _ -> throw({:nonfinite, name})
    end
  end

  defp tan(_cols, <<>>, _ws, _l, _name, acc), do: acc
  defp tan(_, _, _, _, name, _), do: throw({:nonfinite, name})

  defp ties_k(cols, <<y::float-32-little, rb::binary>>, cuts, ws, l, i, name, acc) do
    case heads(cols, [], []) do
      {vals, rest} ->
        taus = Enum.zip_with(vals, cuts, fn x, cut -> t = x - y; if keep?(cut, t, i), do: t, else: 0.0 end)
        ties_k(rest, rb, cuts, ws, l, i + 1, name, <<acc::binary, y + l * ties(taus, ws)::float-32-little>>)

      _ ->
        throw({:nonfinite, name})
    end
  end

  defp ties_k(_cols, <<>>, _cuts, _ws, _l, _i, _name, acc), do: acc
  defp ties_k(_, _, _, _, _, _, name, _), do: throw({:nonfinite, name})

  # sign election by the weighted sum, disjoint weighted mean of agreeing entries
  defp ties(taus, ws) do
    elected = sign(wsum(taus, ws, 0.0))

    {num, den} =
      Enum.zip_reduce(taus, ws, {0.0, 0.0}, fn x, w, {n, dd} ->
        if x != 0.0 and sign(x) == elected, do: {n + w * x, dd + w}, else: {n, dd}
      end)

    if den == 0.0, do: 0.0, else: num / den
  end

  defp sign(x) when x > 0, do: 1
  defp sign(x) when x < 0, do: -1
  defp sign(_), do: 0

  # each entry kept with probability `d` and rescaled by 1/d: one draw per
  # entry and model (none when d ≥ 1, where every entry is kept)
  defp dare_k(m, cols, <<y::float-32-little, rb::binary>>, states, ws, l, d, name, acc) do
    case heads(cols, [], []) do
      {vals, rest} ->
        {taus, states} =
          Enum.zip(vals, states)
          |> Enum.map(fn {x, s} ->
            t = x - y

            if d >= 1.0 do
              {t, s}
            else
              {v, s2} = Tensor.splitmix(s)
              u = Bitwise.bsr(v, 11) / 9_007_199_254_740_992
              {if(u < d, do: t / d, else: 0.0), s2}
            end
          end)
          |> Enum.unzip()

        merged = if m == :dare_linear, do: wsum(taus, ws, 0.0), else: ties(taus, ws)
        dare_k(m, rest, rb, states, ws, l, d, name, <<acc::binary, y + l * merged::float-32-little>>)

      _ ->
        throw({:nonfinite, name})
    end
  end

  defp dare_k(_m, _cols, <<>>, _st, _ws, _l, _d, _name, acc), do: acc
  defp dare_k(_, _, _, _, _, _, _, name, _), do: throw({:nonfinite, name})

  defp slerp_coeffs(na, nb, ab, t) do
    dot = if na == 0 or nb == 0, do: 1.0, else: ab / (na * nb)

    if abs(dot) > 0.9995 do
      {1 - t, t}
    else
      th = acos(dot)
      st = CR.sin_f64(th)
      {CR.sin_f64(th - th * t) / st, CR.sin_f64(th * t) / st}
    end
  end

  @doc false
  # mergekit's slerp: the angle between the normalised tensors, applied to the originals
  def slerp(a, b, t) do
    na = :math.sqrt(Enum.reduce(a, 0.0, &(&1 * &1 + &2)))
    nb = :math.sqrt(Enum.reduce(b, 0.0, &(&1 * &1 + &2)))
    {s0, s1} = slerp_coeffs(na, nb, Enum.zip_reduce(a, b, 0.0, fn x, y, s -> s + x * y end), t)
    Enum.zip_with(a, b, fn x, y -> s0 * x + s1 * y end)
  end

  @doc false
  # arccos without libm: acos(c) = 2·asin(√((1 − c)/2)) for c ≥ 0, where the
  # half angle lies in [0, π/4] and sin is well conditioned, found by
  # bisection of the correctly rounded sine; acos(c) = π − acos(−c) below 0
  def acos(c) when c < 0, do: :math.pi() - acos(-c)

  def acos(c) do
    y = :math.sqrt((1.0 - min(1.0, c)) / 2)

    Enum.reduce(1..64, {0.0, :math.pi() / 4}, fn _, {lo, hi} ->
      mid = (lo + hi) / 2
      if CR.sin_f64(mid) < y, do: {mid, hi}, else: {lo, mid}
    end)
    |> then(fn {lo, hi} -> lo + hi end)
  end

  # ----------------------------------------------------- least squares (RegMean) --

  @doc """
  Measure what a model's matrices see: for every matrix the model's
  adapter can *tap* (`Vapor.Lock.taps/1`), the Gram `G = XᵀX` of its input
  rows over `batches` (lists of token ids — the model's own data).

  Returns `%{grams: %{tensor => %{n, g}}, tokens, digest}` (`g` a binary64
  tuple matrix; tensors that read the same activation share it). The Grams
  are computed on the substrate when `worker:` is given — the activations
  and `XᵀX` are programs, so they are the oracle's bits — and accumulated
  across batches in binary64. Options: `worker`, `max_seq` (default the
  longest batch).
  """
  def calibrate(model, batches, opts \\ []) do
    {spec, ws} = pair(model)
    taps = Vapor.Lock.taps(spec)
    s = Keyword.get(opts, :max_seq, batches |> Enum.map(&length/1) |> Enum.max())

    with :ok <- need(taps != [], :taps, "an adapter that declares taps (#{inspect(spec.adapter)} declares none)"),
         {:ok, p} <- Vapor.Lock.build(spec, ws, max_seq: s, logits: :all) do
      bound = Vapor.Program.bound(p)
      outs = for {tap, _} <- taps, do: {tap, Vapor.Algebra.Term.ref(tap, Map.fetch!(bound, tap))}
      probe = %{p | outputs: p.outputs ++ outs}
      w = opts[:worker]

      grams =
        Enum.reduce(batches, %{}, fn ids, acc ->
          env = Map.merge(Vapor.Lock.zero_state(probe), %{tok: Tensor.from_list(:s32, [length(ids)], ids),
                                                          pos: Tensor.from_list(:s32, [length(ids)], Enum.to_list(0..(length(ids) - 1)))})
          out = Vapor.Modal.Runner.run(probe, env, worker: w)

          Enum.reduce(taps, acc, fn {tap, _}, acc ->
            g = gram_of(out[tap], w)
            Map.update(acc, tap, g, &Linalg.add(&1, g))
          end)
        end)

      tokens = batches |> Enum.map(&length/1) |> Enum.sum()

      per_tensor =
        for {tap, tensors} <- taps, t <- tensors, into: %{}, do: {t, %{n: tokens, g: grams[tap], tap: tap}}

      digest = Vapor.Canonical.hex_digest(taps |> Enum.map(fn {tap, _} -> {Atom.to_string(tap), :crypto.hash(:sha256, Linalg.to_f32(grams[tap]))} end))
      {:ok, %{grams: per_tensor, tokens: tokens, digest: digest}}
    end
  end

  # XᵀX of rows [T, d]: on the substrate as linear(Xᵀ, Xᵀ) (the contraction
  # over T, zero rows padded to a multiple of 16 contribute nothing), else
  # in binary64 on the BEAM
  defp gram_of(%Tensor{shape: [t, d]} = x, nil) do
    rows = x |> Tensor.to_floats() |> Enum.chunk_every(d)
    _ = t
    Linalg.gram_add(nil, rows)
  end

  defp gram_of(%Tensor{shape: [t, d]} = x, w) do
    tp = div(t + 15, 16) * 16
    xp = Tensor.new(:f32, [tp, d], Tensor.widen(x).data <> :binary.copy(<<0::32>>, (tp - t) * d))
    alias Vapor.Algebra.Term, as: T
    xt = T.transpose(T.input(:x, :f32, [tp, d]))
    p = Vapor.Program.new(gram: T.linear(xt, xt))
    out = Vapor.Modal.Runner.run(p, %{x: xp}, worker: w)
    Linalg.from_f32(Tensor.widen(out.gram).data, d, d)
  end

  # W = (Σ Wᵢ G̃ᵢ + λ W̄)(Σ G̃ᵢ + λ I)⁻¹ — least squares on every model's
  # inputs, pulled by a vanishing ridge toward the weighted mean W̄: an input
  # direction no model ever activated (a padding token's embedding, a dead
  # unit) has no evidence, and there the answer is the plain average instead
  # of an arbitrary solution. λ = ridge · mean(diag Σ G̃). Solved row by row:
  # one Cholesky per tensor, one pair of triangular solves per output row.
  defp regmean_tensor(name, models, grams, %{alpha: alpha, weights: ws, ridge: ridge}) do
    %Tensor{shape: [out, inw] = shape} = elem(hd(models), 1)[name]
    gs = Enum.map(grams, fn c -> reduce_offdiag(c.grams[name].g, alpha) end)
    gs = Enum.zip_with(gs, ws, fn g, w -> Linalg.scale(g, w) end)
    total = Enum.reduce(tl(gs), hd(gs), &Linalg.add(&2, &1))
    lam = ridge * (Enum.reduce(0..(inw - 1), 0.0, &(&2 + Linalg.at(total, &1, &1))) / inw)
    lam = if lam > 0.0, do: lam, else: ridge
    total = for(i <- 0..(inw - 1), do: elem(total, i) |> put_elem(i, Linalg.at(total, i, i) + lam)) |> List.to_tuple()

    l =
      case Linalg.cholesky(total) do
        {:ok, l} -> l
        _ -> throw({:singular, name})
      end

    mats = Enum.map(models, fn {_, w} -> Linalg.from_f32(f32_data(w[name]), out, inw) end)
    tot_w = Enum.sum(ws)
    mean = Enum.zip_with(mats, ws, &Linalg.scale(&1, &2 / tot_w)) |> then(fn [h | t] -> Enum.reduce(t, h, &Linalg.add(&2, &1)) end)

    # Σ Wᵢ G̃ᵢ + λ W̄, as rows (out × in)
    rhs =
      Enum.zip_with(mats, gs, &Linalg.mul/2)
      |> then(fn [h | t] -> Enum.reduce(t, h, &Linalg.add(&2, &1)) end)
      |> Linalg.add(Linalg.scale(mean, lam))

    rows = for r <- 0..(out - 1), do: Linalg.chol_solve(l, elem(rhs, r))
    Tensor.new(:f32, shape, for(row <- rows, x <- Tuple.to_list(row), into: <<>>, do: <<x::float-32-little>>))
  end

  defp reduce_offdiag(g, alpha) when alpha == 1.0, do: g

  defp reduce_offdiag(g, alpha) do
    n = tuple_size(g)
    for(i <- 0..(n - 1), do: for(j <- 0..(n - 1), do: (if i == j, do: Linalg.at(g, i, j), else: alpha * Linalg.at(g, i, j))) |> List.to_tuple())
    |> List.to_tuple()
  end

  # -------------------------------------------------------------- diagnosis --

  @doc """
  **Which regime are these models in?** — measured before fusing, from the
  weights alone (and a base, when there is one). Per model (energy-weighted
  over float tensors):

    * `relative_delta` = ‖θᵢ − base‖ / ‖base‖ — how far the model moved;
    * `concentration` = share of ‖τᵢ‖² in the top `density` entries by
      magnitude — what TIES's trim keeps (1 − it is the energy it discards);
    * `dare_noise` = √((1 − p)/p) at `p = density` — DARE's drop-and-rescale
      perturbs each task vector by this fraction *of its own norm*; it is
      harmless only when the delta is redundant, which the weights alone
      cannot show (measure it: `select/4`).

  Per pair: `weight_cosine` between the weights themselves (≈ 1 for models
  that share an ancestor, ≈ 0 for networks trained from different
  initialisations — whose units are not aligned, so no weight average is
  meaningful), and with a base, `delta_cosine` and `sign_conflict` = Σ
  min(|τᵢ|, |τⱼ|) over entries of opposite sign / Σ min(|τᵢ|, |τⱼ|).

  Returns `%{models, pairs, regime, advice}` with `regime` one of
  `:unrelated`, `:dense`, `:small_deltas`, `:single`. The advice states
  what the measurements imply and stops there: which method wins among
  the admissible ones is measured, not predicted (`select/4`). Options:
  `base`, `density` (0.2), `sample` (entries read per tensor, at most).
  """
  def diagnose(models, opts \\ []) do
    models = Enum.map(models, &pair/1)
    base = opts[:base] && pair(opts[:base])
    d = Keyword.get(opts, :density, 0.2) * 1.0
    {_, first} = hd(models)
    names = first |> Enum.filter(fn {k, t} -> is_binary(k) and match?(%Tensor{dtype: dt} when dt in [:f32, :bf16, :f16], t) end) |> Enum.map(&elem(&1, 0)) |> Enum.sort()
    cap = Keyword.get(opts, :sample)
    floats = fn t -> sample(Tensor.to_floats(%{t | data: f32_data(t), dtype: :f32}), cap) end
    k = length(models)

    per =
      pmap(names, System.schedulers_online(), fn name ->
        xs = Enum.map(models, fn {_, w} -> floats.(w[name]) end)
        b = base && floats.(elem(base, 1)[name])
        taus = if b, do: Enum.map(xs, fn x -> Enum.zip_with(x, b, &(&1 - &2)) end)

        %{bn: b && sq(b), xn: Enum.map(xs, &sq/1), tn: taus && Enum.map(taus, &sq/1), conc: taus && Enum.map(taus, &concentration(&1, d)),
          pairs: for(i <- 0..(k - 1), j <- (i + 1)..(k - 1)//1, into: %{}, do: {{i, j}, pair_sums(Enum.at(xs, i), Enum.at(xs, j), taus && Enum.at(taus, i), taus && Enum.at(taus, j))})}
      end)

    total = fn f -> per |> Enum.map(f) |> Enum.sum() end

    model_stats =
      for i <- 0..(k - 1) do
        if base do
          tn = total.(&Enum.at(&1.tn, i))
          bn = total.(& &1.bn)
          conc = if tn == 0, do: 1.0, else: total.(&(Enum.at(&1.tn, i) * Enum.at(&1.conc, i))) / tn
          %{relative_delta: if(bn == 0, do: 0.0, else: :math.sqrt(tn / bn)), concentration: conc,
            dare_noise: :math.sqrt((1 - min(d, 0.999_999)) / max(d, 1.0e-9))}
        else
          %{norm: :math.sqrt(total.(&Enum.at(&1.xn, i)))}
        end
      end

    pair_stats =
      for i <- 0..(k - 1), j <- (i + 1)..(k - 1)//1 do
        sums = Enum.reduce(per, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, fn r, acc -> add8(acc, r.pairs[{i, j}]) end)
        {xy, xx, yy, cn, cd, tt, ta, tb} = sums
        cosw = if xx == 0 or yy == 0, do: 0.0, else: xy / :math.sqrt(xx * yy)
        rel = :math.sqrt(max(xx + yy - 2 * xy, 0.0) / max(min(xx, yy), 1.0e-300))

        %{models: {i, j}, weight_cosine: cosw, relative_distance: rel} |>
          Map.merge(if base, do: %{sign_conflict: if(cd == 0, do: 0.0, else: cn / cd), delta_cosine: if(ta == 0 or tb == 0, do: 0.0, else: tt / :math.sqrt(ta * tb))}, else: %{})
      end

    {regime, advice} = advise(model_stats, pair_stats, base != nil, d)
    %{models: model_stats, pairs: pair_stats, regime: regime, advice: advice, density: d, tensors: length(names)}
  end

  defp add8({a1, a2, a3, a4, a5, a6, a7, a8}, {b1, b2, b3, b4, b5, b6, b7, b8}),
    do: {a1 + b1, a2 + b2, a3 + b3, a4 + b4, a5 + b5, a6 + b6, a7 + b7, a8 + b8}

  # weights x, y and (with a base) task vectors s, t:
  # {x·y, x·x, y·y, Σ min(|s|,|t|) over opposite signs, Σ min(|s|,|t|), s·t, s·s, t·t}
  defp pair_sums(x, y, s, t) do
    {xy, xx, yy} = Enum.zip_reduce(x, y, {0.0, 0.0, 0.0}, fn a, b, {p, q, r} -> {p + a * b, q + a * a, r + b * b} end)

    {cn, cd, st, ss, tt} =
      if s,
        do: Enum.zip_reduce(s, t, {0.0, 0.0, 0.0, 0.0, 0.0}, fn a, b, {n, dd, p, q, r} ->
          m = min(abs(a), abs(b))
          {if(a * b < 0, do: n + m, else: n), dd + m, p + a * b, q + a * a, r + b * b}
        end),
        else: {0.0, 0.0, 0.0, 0.0, 0.0}

    {xy, xx, yy, cn, cd, st, ss, tt}
  end

  defp sample(xs, nil), do: xs
  defp sample(xs, cap) when length(xs) <= cap, do: xs
  defp sample(xs, cap), do: Enum.take_every(xs, div(length(xs), cap) + 1)

  defp sq(xs), do: Enum.reduce(xs, 0.0, &(&1 * &1 + &2))

  defp concentration(tau, d) do
    total = sq(tau)
    k = max(round(d * length(tau)), 1)
    if total == 0, do: 1.0, else: (tau |> Enum.map(&(&1 * &1)) |> Enum.sort(:desc) |> Enum.take(k) |> Enum.sum()) / total
  end

  defp advise(_ms, [], _based?, _d), do: {:single, ["one model: nothing to fuse"]}

  defp advise(ms, ps, based?, d) do
    pct = fn x -> "#{round(x * 100)} %" end
    cosw = ps |> Enum.map(& &1.weight_cosine) |> Enum.min()

    cond do
      cosw < 0.5 ->
        {:unrelated,
         ["the weights are nearly orthogonal (cosine #{Float.round(cosw, 3)}): these networks do not share an ancestor, " <>
            "so their units are not aligned (a network is the same function under any permutation of its hidden units)",
          "a weight average of misaligned networks destroys both; do not fuse them in weight space (align permutations first — not implemented here)"]}

      not based? ->
        rel = ps |> Enum.map(& &1.relative_distance) |> Enum.max()
        {:small_deltas,
         ["related models (weight cosine ≥ #{Float.round(cosw, 3)}), #{pct.(rel)} apart; without a base only :linear, :slerp and :regmean apply",
          "measure them on held-out data (Vapor.Merge.select/4)"]}

      true ->
        rel = ms |> Enum.map(& &1.relative_delta) |> Enum.max()
        conc = ms |> Enum.map(& &1.concentration) |> Enum.min()
        conflict = ps |> Enum.map(& &1.sign_conflict) |> Enum.max()
        dc = ps |> Enum.map(& &1.delta_cosine) |> Enum.min()

        facts =
          ["task vectors at #{pct.(rel)} of the weights' norm; the top #{round(d * 100)} % of entries carry #{pct.(conc)} of their energy " <>
             "(TIES at density #{d} discards #{pct.(1 - conc)})",
           "signs disagree on #{pct.(conflict)} of the deltas' shared magnitude; delta cosine #{Float.round(dc, 3)}",
           "DARE at p = #{d} perturbs each delta by #{pct.(hd(ms).dare_noise)} of its own norm — safe only if the fine-tune is redundant"]

        if rel > 0.25 do
          {:dense, facts ++ ["deltas this large are not the small fine-tune deltas TIES and DARE assume; prefer :linear, :slerp, :regmean — and measure (select/4)"]}
        else
          {:small_deltas, facts ++ ["admissible: every method; :task_arithmetic at λ = 1 applies each delta at full strength — also where it hurts the other models' domains (λ = 1/n is :linear) — measure (select/4)"]}
        end
    end
  end

  # -------------------------------------------------------------- selection --

  @doc """
  **Fuse by measurement**: build every candidate (`[{label, merge_opts}]`),
  score each with `score.(merged)` (lower is better — e.g. held-out bits
  per byte through the certified substrate, `Vapor.Quality.Model.bits_per_byte/4`),
  and keep the best. The fused weights are deterministic and so are the
  scores, so the table is reproducible by anyone with the inputs.

  Returns `{:ok, %{best, merged, table, receipt}}`: `table` lists
  `%{label, score, ms}` in candidate order, `receipt` a certificate of kind
  `vapor.merge.select/1` holding the chosen merge's receipt payload, every
  candidate's score and `eval` (an identifier of the scoring data — pass
  its digest as `eval:`). Options: `key`, `eval`.
  """
  def select(models, candidates, score, opts \\ []) when is_function(score, 1) do
    rows =
      Enum.map(candidates, fn {label, mo} ->
        {t, res} = :timer.tc(fn -> merge(models, mo) end)

        case res do
          {:ok, m} -> %{label: to_string(label), merged: m, score: score.(m) * 1.0, ms: div(t, 1000)}
          {:error, r} -> %{label: to_string(label), merged: nil, score: nil, ms: div(t, 1000), refused: r.bound}
        end
      end)

    case rows |> Enum.filter(& &1.score) |> Enum.min_by(& &1.score, fn -> nil end) do
      nil ->
        reject(:candidates, "at least one candidate that fuses")

      best ->
        table = Enum.map(rows, &Map.drop(&1, [:merged]))
        payload = %{kind: "vapor.merge.select/1", chosen: best.label, merge: best.merged.receipt.payload, eval: opts[:eval],
                    scores: Enum.map(table, fn r -> %{label: r.label, score: r.score} end)}
        cert = %Certificate{payload: payload}
        {:ok, %{best: best.label, merged: best.merged, table: table, receipt: if(opts[:key], do: Certificate.sign(cert, opts[:key]), else: cert)}}
    end
  end

  # ---------------------------------------------------------------- streaming --

  @doc """
  Fuse checkpoints **from disk to disk**, one tensor at a time: memory is
  that of the largest tensor times the number of inputs, not of the
  models — fusing two 7 B checkpoints needs a few GB, not three copies of
  28 GB.

  `dirs` are Hugging Face directories (`model.safetensors`, single or
  sharded); `out` receives the fused weights (sharded as
  `Vapor.Ingest.Safetensors.write_sharded/4` would, `max_shard` bytes,
  default 4 GiB), the first input's `config.json` and tokenizer files are
  the caller's to copy. Options as `merge/2` (`method` among the
  element-wise ones — `:regmean` measures activations and runs on admitted
  models), plus `base` (a directory), `dtype` (`"F32"`, `"BF16"`,
  `"F16"`), `max_shard`.

  **The same bits as `merge/2`**: every tensor goes through the same
  kernels in the same block order, so the files are byte for byte those
  that `write_sharded/4` writes from the in-memory fusion, and the receipt
  carries the same Merkle roots of inputs and output
  (`test/vapor/merge_stream_test.exs`). What the stream cannot do without
  loading the models is the airlock's full admission: compatibility is
  checked on what the files declare — the configuration's canonical
  digest, the adapter the airlock's claim picks from the configuration
  and the tensor index, every tensor's name, shape and kind.

  Returns `{:ok, %{files, receipt, tensors, bytes}}`.
  """
  def stream(dirs, out, opts \\ []) do
    method = Keyword.get(opts, :method, :linear)
    base_dir = opts[:base]
    as = Keyword.get(opts, :dtype, "F32")
    max_shard = Keyword.get(opts, :max_shard, 4 * 1024 * 1024 * 1024)
    conc = Keyword.get(opts, :concurrency, System.schedulers_online())

    with :ok <- need(method in @methods -- [:regmean], :method, "an element-wise method (#{inspect(@methods -- [:regmean])}); :regmean runs on admitted models"),
         :ok <- need(dirs != [], :models, "at least one model"),
         :ok <- need(method != :slerp or length(dirs) == 2, :models, "exactly two models for :slerp"),
         :ok <- need(method not in [:task_arithmetic, :ties, :dare_linear, :dare_ties] or base_dir != nil, :base, "a base model for #{method}"),
         :ok <- need(as in ["F32", "BF16", "F16"], :dtype, "F32, BF16 or F16"),
         {:ok, ins} <- lazy_all(dirs),
         {:ok, base} <- (if base_dir, do: lazy_one(base_dir), else: {:ok, nil}),
         :ok <- lazy_compatible(Enum.reject([base | ins], &is_nil/1), opts),
         ws = Keyword.get(opts, :weights, List.duplicate(1.0, length(dirs))),
         :ok <- need(length(ws) == length(dirs) and Enum.all?(ws, &is_number/1), :weights, "one number per model") do
      p = params(method, ws, opts)
      first = hd(ins)
      names = first.catalog |> Map.keys() |> Enum.sort()

      # the plan: every tensor's stored dtype, shape and size, before any data is read
      plan =
        for name <- names do
          {_f, _ds, e} = first.catalog[name]
          shape = if e.shape == [], do: [1], else: e.shape
          {dt, el} = stored_kind(e.dtype, as)
          {name, dt, shape, Enum.product(shape) * el}
        end

      shards = Vapor.Ingest.Safetensors.shard_plan(plan, &elem(&1, 3), max_shard)
      n = length(shards)
      files = if n <= 1, do: ["model.safetensors"], else: for(k <- 1..n, do: "model-#{pad5(k)}-of-#{pad5(n)}.safetensors")
      File.mkdir_p!(out)

      try do
        leaves =
          Enum.zip(files, shards)
          |> Enum.flat_map(fn {file, shard} ->
            {:ok, io} = File.open(Path.join(out, file), [:write, :binary, :raw])

            try do
              :ok = :file.write(io, Vapor.Ingest.Safetensors.header(Enum.map(shard, fn {nm, dt, sh, b} -> {nm, dt, sh, b} end), %{"format" => "pt"}))

              for {name, dt, _shape, bytes} <- shard do
                {inputs, base_t, fused} = stream_tensor(method, name, ins, base, p, conc)
                {^dt, data} = Vapor.Ingest.Safetensors.stored_as(fused, as_for(fused, as))
                ^bytes = byte_size(data)
                :ok = :file.write(io, data)
                leaves = {Enum.map(inputs, &leaf(name, &1)), base_t && leaf(name, base_t), leaf(name, fused)}
                # the tensors just written are garbage now: collect them before reading the next
                # (a caller with a large heap would otherwise keep them all — measured in the suite)
                :erlang.garbage_collect()
                leaves
              end
            after
              File.close(io)
            end
          end)

        if n > 1 do
          index = %{"metadata" => %{"total_size" => plan |> Enum.map(&elem(&1, 3)) |> Enum.sum()},
                    "weight_map" => for({file, shard} <- Enum.zip(files, shards), {name, _, _, _} <- shard, into: %{}, do: {name, file})}
          :ok = File.write(Path.join(out, "model.safetensors.index.json"), Vapor.JSON.encode(index))
        end

        roots = fn sel -> leaves |> Enum.map(sel) |> Merkle.root() |> Base.encode16(case: :lower) end
        inputs = for {m, i} <- Enum.with_index(ins), do: %{family: m.family, config: m.digest, weights: roots.(&Enum.at(elem(&1, 0), i))}

        payload = %{
          kind: "vapor.merge/1", method: Atom.to_string(method),
          params: p |> Map.delete(:method) |> Map.update!(:normalize, &to_string/1) |> Map.drop([:alpha, :ridge]),
          inputs: inputs, base: base && %{config: base.digest, weights: roots.(&elem(&1, 1))}, output: roots.(&elem(&1, 2))
        }

        cert = %Certificate{payload: payload}
        receipt = if opts[:key], do: Certificate.sign(cert, opts[:key]), else: cert
        {:ok, %{files: files ++ if(n > 1, do: ["model.safetensors.index.json"], else: []), receipt: receipt,
                tensors: length(plan), bytes: plan |> Enum.map(&elem(&1, 3)) |> Enum.sum()}}
      catch
        {:nonfinite, name} -> reject({:weight, name}, "finite weights (#{name} holds a NaN or an infinity)")
        {:read, rej} -> {:error, rej}
        {:differ, name} -> reject({:weight, name}, "equal non-float tensors in every model (#{name} differs)")
      end
    end
  end

  defp pad5(k), do: k |> Integer.to_string() |> String.pad_leading(5, "0")

  # float dtypes are read as f32 and fused; the rest is copied as read
  @float_st ~w(F32 BF16 F16 F64 F8_E4M3 F8_E4M3FNUZ F8_E5M2 F8_E5M2FNUZ F8_E8M0)
  defp stored_kind(dt, as) when dt in @float_st, do: {as, if(as == "F32", do: 4, else: 2)}
  defp stored_kind(dt, _as) when dt in ["I8"], do: {"I8", 1}
  defp stored_kind(dt, _as) when dt in ["U8"], do: {"U8", 1}
  defp stored_kind(_dt, _as), do: {"I32", 4}

  defp as_for(%Tensor{dtype: :f32}, as), do: as
  defp as_for(_t, _as), do: "F32"

  defp stream_tensor(method, name, ins, base, p, conc) do
    read = fn m ->
      {f, ds, e} = m.catalog[name]
      case Vapor.Ingest.Safetensors.read_entry(f, ds, e) do
        {:ok, t} -> t
        {:error, rej} -> throw({:read, rej})
      end
    end

    ts = Enum.map(ins, read)
    bt = base && read.(base)

    fused =
      if match?(%Tensor{dtype: :f32}, hd(ts)) do
        models = Enum.map(ts, &{nil, %{name => &1}})
        job = prepare_tensor(method, name, models, bt && {nil, %{name => bt}}, p)

        data =
          0..max(div(job.n - 1, @block), 0)//1
          |> pmap(conc, fn k -> kernel(method, job, k * @block, min(@block, job.n - k * @block), p) end)
          |> IO.iodata_to_binary()

        Tensor.new(:f32, job.shape, data)
      else
        if Enum.all?(ts, &(&1 == hd(ts))), do: hd(ts), else: throw({:differ, name})
      end

    {ts, bt, fused}
  end

  defp lazy_all(dirs) do
    Enum.reduce_while(dirs, {:ok, []}, fn d, {:ok, acc} ->
      case lazy_one(d) do
        {:ok, m} -> {:cont, {:ok, acc ++ [m]}}
        e -> {:halt, e}
      end
    end)
  end

  # a checkpoint by its declarations: config, the airlock's claim, the tensor catalogue
  defp lazy_one(dir) do
    with {:ok, bin} <- File.read(Path.join(dir, "config.json")) |> (fn {:ok, _} = ok -> ok; {:error, w} -> {:error, Rejection.new({:file, dir}, "a config.json (#{inspect(w)})", "check the path")} end).(),
         {:ok, cfg} <- Vapor.JSON.decode(bin),
         {:ok, cat} <- Vapor.Ingest.Safetensors.catalog(dir),
         shapes = Map.new(cat, fn {k, {_, _, e}} -> {k, if(e.shape == [], do: [1], else: e.shape)} end),
         {:ok, {adapter, _}} <- claim(%{source: :hf, path: dir, config: cfg, tensors: shapes}) do
      {:ok, %{dir: dir, catalog: cat, adapter: adapter, family: cfg["model_type"] || cfg["architectures"] |> List.wrap() |> List.first(),
              digest: Vapor.Canonical.hex_digest({:config, cfg}), kinds: Map.new(cat, fn {k, {_, _, e}} -> {k, {e.shape, e.dtype in @float_st}} end)}}
    end
  end

  defp claim(manifest) do
    case Vapor.Lock.select(manifest) do
      {:ok, a} -> {:ok, {a, nil}}
      {:error, _} = e -> e
    end
  end

  defp lazy_compatible([m0 | rest], opts) do
    Enum.reduce_while(rest, :ok, fn m, :ok ->
      cond do
        m.adapter != m0.adapter -> {:halt, reject(:spec, "the same adapter (#{inspect(m0.adapter)} vs #{inspect(m.adapter)})")}
        m.digest != m0.digest and not Keyword.get(opts, :allow_config_mismatch, false) ->
          {:halt, reject(:config, "the same configuration (#{m0.dir} vs #{m.dir}); allow_config_mismatch: true to fuse anyway")}
        m.kinds != m0.kinds ->
          diff = Map.keys(m0.kinds) |> Kernel.++(Map.keys(m.kinds)) |> Enum.uniq() |> Enum.find(&(m0.kinds[&1] != m.kinds[&1]))
          {:halt, reject({:weight, diff}, "the same tensor names, shapes and kinds in every model")}
        true -> {:cont, :ok}
      end
    end)
  end

  # ------------------------------------------------------------------ receipt --

  @doc "Merkle root of a weight map: leaves `name ‖ dtype ‖ shape ‖ SHA-256(data)`, sorted by name."
  def weights_root(weights) do
    weights
    |> Enum.filter(fn {k, _} -> is_binary(k) end)
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.map(fn {k, t} -> leaf(k, t) end)
    |> Merkle.root()
  end

  defp leaf(k, %Tensor{dtype: dt, shape: s, data: d}), do: Merkle.leaf(Vapor.Canonical.encode({k, Atom.to_string(dt), s, :crypto.hash(:sha256, d)}))

  defp receipt(method, params, models, base, merged, opts) do
    inputs = for {s, w} <- models, do: %{family: s.family, config: s.digest, weights: Base.encode16(weights_root(w), case: :lower)}

    params = params |> Map.delete(:method) |> Map.update!(:normalize, &to_string/1)
    params = if method == :regmean, do: Map.put(params, :grams, Enum.map(opts[:grams], & &1.digest)), else: Map.drop(params, [:alpha, :ridge])

    payload = %{
      kind: "vapor.merge/1", method: Atom.to_string(method), params: params, inputs: inputs,
      base: base && %{config: elem(base, 0).digest, weights: Base.encode16(weights_root(elem(base, 1)), case: :lower)},
      output: Base.encode16(weights_root(merged), case: :lower)
    }

    cert = %Certificate{payload: payload}
    if opts[:key], do: Certificate.sign(cert, opts[:key]), else: cert
  end

  @doc "Recompute a receipt's output root from weights and compare: `:ok` or `{:error, :mismatch}`."
  def verify_receipt(%Certificate{payload: %{output: out}}, weights) do
    if Base.encode16(weights_root(weights), case: :lower) == out, do: :ok, else: {:error, :mismatch}
  end

  defp need(true, _f, _b), do: :ok
  defp need(false, f, b), do: reject(f, b)
  defp reject(f, b), do: {:error, Rejection.new({:merge, f}, b, "fuse models admitted from the same topology")}
end
