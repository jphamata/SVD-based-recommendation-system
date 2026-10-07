defmodule Vapor.Spatial do
  @moduledoc """
  Spatial operators — convolution (2-D and 3-D, any stride, padding and
  dilation), group normalisation, nearest upsampling, attention over pixels
  — built from the existing operators of the algebra: the image is a table
  of pixel rows, and

      conv(x) = reshape(sel(pad, ½, 0, gather_row(x, idx)), [n, k·C])·Wᵀ + b

  `idx` lists, for every output pixel and kernel tap, the input row it reads
  (the im2col of the convolution, a constant); `pad` (1 on taps that fall
  in the padding) selects `+0` there — selection, so exactly `+0`;
  `reshape` (a byte copy) turns the gathered taps into one row per output
  pixel; one `linear` does the contraction. Every output is then a
  canonical dot product over `k·C` — bit-identical on every substrate —
  and no convolution kernel exists anywhere: the same certified
  `gather_row`, `sel`, `transpose` (the copy) and GEMV.

  Layout: an activation of `H×W` pixels and `C` channels is `f32[H·W, Cp]`
  (row-major pixels, channels last), with `Cp` = `C` rounded up to 16 — the
  contraction width the GEMV needs; the padding channels are exactly zero
  and every operator here keeps them so (zero weight rows, zero affine
  parameters). PyTorch's `NCHW` converts at the boundary (`from_nchw/1`,
  `to_nchw/2`).

  What this does not do: grouped and depthwise convolutions (the im2col
  layout interleaves taps and channels; a depthwise filter would need a
  per-pixel transpose) and transposed convolutions (upsampling here is
  nearest + convolution, as diffusers' VAE and U-Net decoders do).

  Functions take and return a list of let-bindings (`lets`, reversed, as
  `Vapor.Program` wants them): constants are bound once by name.
  """
  alias Vapor.Tensor
  alias Vapor.Algebra.Term, as: T

  @doc "`n` rounded up to a multiple of 16."
  def pad16(n), do: div(n + 15, 16) * 16

  # ----------------------------------------------------------- constants --

  @doc false
  def bind(lets, name, %Tensor{} = t) do
    atom = if is_atom(name), do: name, else: String.to_atom(name)
    {[{atom, T.const(t)} | lets], T.ref(atom, T.const(t))}
  end

  @doc """
  Name an intermediate value (a let-binding): later terms refer to it, so a
  deep network stays a DAG of named values rather than a tree that repeats
  its own inputs.
  """
  def name(lets, name, term) do
    atom = if is_atom(name), do: name, else: String.to_atom(name)
    {[{atom, term} | lets], T.ref(atom, term)}
  end

  defp f32(shape, list), do: Tensor.from_list(:f32, shape, list)

  # a [m] or [1, m] parameter as a [1, mp] row, zero beyond m
  defp row(%Tensor{} = t, mp) do
    v = Tensor.to_floats(Tensor.widen(t))
    f32([1, mp], v ++ List.duplicate(0.0, mp - length(v)))
  end

  # ---------------------------------------------------------- convolution --

  @doc """
  2-D convolution of `x : f32[h·w, pad16(cin)]` by a PyTorch weight
  `[cout, cin, kh, kw]` (and bias `[cout]` or nil). Options: `stride`,
  `padding`, `dilation` (integers or `{y, x}` pairs), `pads` (`{top,
  bottom, left, right}`, overriding `padding`). Returns
  `{lets, y, {h', w', cout}}`, `y : f32[h'·w', pad16(cout)]`.
  """
  def conv2d(lets, x, {h, w, cin}, name, %Tensor{shape: [cout, cin, kh, kw]} = wt, bias, opts \\ []) do
    {sy, sx} = pair(Keyword.get(opts, :stride, 1))
    {py, px} = pair(Keyword.get(opts, :padding, 0))
    # asymmetric padding {top, bottom, left, right} (diffusers' encoder downsampler pads right and bottom only)
    {pt, pb, pl, pr} = Keyword.get(opts, :pads, {py, py, px, px})
    {py, px} = {pt, pl}
    {dy, dx} = pair(Keyword.get(opts, :dilation, 1))
    ho = div(h + pt + pb - dy * (kh - 1) - 1, sy) + 1
    wo = div(w + pl + pr - dx * (kw - 1) - 1, sx) + 1
    taps = for ky <- 0..(kh - 1), kx <- 0..(kw - 1), do: {ky * dy, kx * dx}

    pos =
      for oy <- 0..(ho - 1), ox <- 0..(wo - 1), {oky, okx} <- taps do
        {iy, ix} = {oy * sy - py + oky, ox * sx - px + okx}
        if iy >= 0 and iy < h and ix >= 0 and ix < w, do: iy * w + ix, else: nil
      end

    contract(lets, x, cin, name, wt, bias, pos, length(taps), ho * wo, fn co, ci, tap ->
      {ky, kx} = {div(tap, kw), rem(tap, kw)}
      {co, ci, ky, kx}
    end)
    |> then(fn {lets, y} -> {lets, y, {ho, wo, cout}} end)
  end

  @doc """
  3-D convolution (time, height, width) of `x : f32[t·h·w, pad16(cin)]` by
  a weight `[cout, cin, kt, kh, kw]` — the video and volume case, by the
  same construction. Options as `conv2d/7`, with triples.
  """
  def conv3d(lets, x, {t, h, w, cin}, name, %Tensor{shape: [cout, cin, kt, kh, kw]} = wt, bias, opts \\ []) do
    {st, sy, sx} = triple(Keyword.get(opts, :stride, 1))
    {pt, py, px} = triple(Keyword.get(opts, :padding, 0))
    {dt, dy, dx} = triple(Keyword.get(opts, :dilation, 1))
    to = div(t + 2 * pt - dt * (kt - 1) - 1, st) + 1
    ho = div(h + 2 * py - dy * (kh - 1) - 1, sy) + 1
    wo = div(w + 2 * px - dx * (kw - 1) - 1, sx) + 1
    taps = for kz <- 0..(kt - 1), ky <- 0..(kh - 1), kx <- 0..(kw - 1), do: {kz * dt, ky * dy, kx * dx}

    pos =
      for oz <- 0..(to - 1), oy <- 0..(ho - 1), ox <- 0..(wo - 1), {okz, oky, okx} <- taps do
        {iz, iy, ix} = {oz * st - pt + okz, oy * sy - py + oky, ox * sx - px + okx}
        if iz >= 0 and iz < t and iy >= 0 and iy < h and ix >= 0 and ix < w, do: (iz * h + iy) * w + ix, else: nil
      end

    contract(lets, x, cin, name, wt, bias, pos, length(taps), to * ho * wo, fn co, ci, tap ->
      {kz, r} = {div(tap, kh * kw), rem(tap, kh * kw)}
      {co, ci, kz, div(r, kw), rem(r, kw)}
    end)
    |> then(fn {lets, y} -> {lets, y, {to, ho, wo, cout}} end)
  end

  # im2col by gather + mask + reshape, then one GEMV; `at.(co, ci, tap)`
  # names the weight element of output co, input ci, tap
  defp contract(lets, x, cin, name, wt, bias, pos, k, n, at) do
    cout = hd(wt.shape)
    {cpi, cpo} = {pad16(cin), pad16(cout)}
    vals = wt |> Tensor.widen() |> Tensor.to_floats() |> List.to_tuple()
    strides = wt.shape |> tl() |> Enum.reverse() |> Enum.scan(1, &(&1 * &2)) |> Enum.reverse() |> tl() |> Kernel.++([1])
    stride0 = Enum.product(tl(wt.shape))
    get = fn idx -> elem(vals, idx) end

    wrows =
      for co <- 0..(cpo - 1), tap <- 0..(k - 1), ci <- 0..(cpi - 1) do
        if co < cout and ci < cin do
          [c0 | rest] = Tuple.to_list(at.(co, ci, tap))
          get.(c0 * stride0 + Enum.sum(Enum.zip_with(rest, strides, &(&1 * &2))))
        else
          0.0
        end
      end

    {lets, wref} = bind(lets, name <> ".weight", f32([cpo, k * cpi], wrows))

    {lets, cols} =
      if k == 1 and Enum.all?(Enum.with_index(pos), fn {p, i} -> p == i end) do
        # a 1×1 convolution with stride 1: the rows themselves
        {lets, x}
      else
        {lets, idx} = bind(lets, name <> ".im2col", Tensor.from_list(:s32, [length(pos)], Enum.map(pos, &(&1 || 0))))
        if Enum.all?(pos, & &1) do
          {lets, T.reshape(T.gather_row(x, idx), [n, k * cpi])}
        else
          {lets, m} = bind(lets, name <> ".pad", f32([length(pos), 1], Enum.map(pos, &if(&1, do: 1.0, else: 0.0))))
          # sel(m, ½, a, b) = m < ½ ? a : b — a tap in the padding (m = 0) is +0
          {lets, T.reshape(T.sel(m, T.splat(0.5), T.splat(0.0), T.gather_row(x, idx)), [n, k * cpi])}
        end
      end

    y = T.linear(cols, wref)

    case bias do
      nil -> name(lets, name <> ".out", y)
      b -> {lets, bref} = bind(lets, name <> ".bias", row(b, cpo)); name(lets, name <> ".out", T.add(y, bref))
    end
  end

  defp pair({a, b}), do: {a, b}
  defp pair(n) when is_integer(n), do: {n, n}
  defp triple({a, b, c}), do: {a, b, c}
  defp triple(n) when is_integer(n), do: {n, n, n}

  # ------------------------------------------------------- normalisation --

  @doc """
  GroupNorm of `x : f32[n, pad16(c)]` with `groups` groups over the `c`
  real channels: per group, the mean and the biased variance over every
  pixel and channel of the group (two passes), then weight and bias.
  Column sums are a transpose and a row reduction, group sums and their
  broadcast back are exact 0/1 contractions; `n` must be a multiple of 16
  (the canonical reduction's width).
  """
  def group_norm(lets, x, n, c, groups, name, w, b, eps) do
    unless rem(n, 16) == 0, do: raise(ArgumentError, "group_norm over #{n} pixels: a multiple of 16 is needed")
    cp = pad16(c)
    gp = pad16(groups)
    cg = div(c, groups)
    {lets, bsel} = bind(lets, "$gn.#{c}.#{groups}.sum", f32([gp, cp], for(g <- 0..(gp - 1), ch <- 0..(cp - 1), do: if(ch < c and div(ch, cg) == g, do: 1.0, else: 0.0))))
    {lets, esel} = bind(lets, "$gn.#{c}.#{groups}.expand", f32([cp, gp], for(ch <- 0..(cp - 1), g <- 0..(gp - 1), do: if(ch < c and div(ch, cg) == g, do: 1.0, else: 0.0))))
    inv = T.splat(1.0 / (n * cg))
    # per-channel column sums [1, cp] → per-group means → per-channel means
    colsum = fn v -> T.transpose(T.reduce(:sum, T.transpose(v))) end
    mean = T.linear(T.mul(T.linear(colsum.(x), bsel), inv), esel)
    {lets, xc} = name(lets, name <> ".centred", T.sub(x, mean))
    var = T.linear(T.mul(T.linear(colsum.(T.mul(xc, xc)), bsel), inv), esel)
    {lets, wr} = bind(lets, name <> ".weight", row(w, cp))
    {lets, br} = bind(lets, name <> ".bias", row(b, cp))
    name(lets, name <> ".out", T.add(T.mul(wr, T.mul(xc, T.rsqrt(T.add(var, T.splat(eps))))), br))
  end

  # ----------------------------------------------------------- resampling --

  @doc "Nearest-neighbour upsampling by an integer factor: `f32[h·w, C] → f32[(f·h)·(f·w), C]`."
  def upsample_nearest(lets, x, {h, w, c}, name, factor \\ 2) do
    idx = for oy <- 0..(h * factor - 1), ox <- 0..(w * factor - 1), do: div(oy, factor) * w + div(ox, factor)
    {lets, r} = bind(lets, "$up.#{h}x#{w}x#{factor}", Tensor.from_list(:s32, [length(idx)], idx))
    {lets, y} = name(lets, name, T.gather_row(x, r))
    {lets, y, {h * factor, w * factor, c}}
  end

  # ---------------------------------------------------------- attention --

  @doc """
  Single-head self-attention over the `n` pixels (a VAE's or U-Net's
  mid-block): bidirectional attention is causal attention whose horizon is
  the last row. `wq, wk, wv, wo : [c, c]` and biases; scale 1/√c.
  """
  def pixel_attention(lets, x, n, c, name, {wq, bq}, {wk, bk}, {wv, bv}, {wo, bo}) do
    cp = pad16(c)
    lin = fn lets, tag, wt, bt ->
      {lets, wr} = bind(lets, "#{name}.#{tag}.weight", pad_matrix(wt, cp, cp))
      {lets, br} = bind(lets, "#{name}.#{tag}.bias", row(bt, cp))
      {lets, fn v -> T.add(T.linear(v, wr), br) end}
    end

    {lets, fq} = lin.(lets, "q", wq, bq)
    {lets, fk} = lin.(lets, "k", wk, bk)
    {lets, fv} = lin.(lets, "v", wv, bv)
    {lets, fo} = lin.(lets, "o", wo, bo)
    {lets, hz} = bind(lets, "$attn.horizon.#{n}", Tensor.from_list(:s32, [n], List.duplicate(n - 1, n)))
    {lets, k} = name(lets, name <> ".k", fk.(x))
    {lets, v} = name(lets, name <> ".v", fv.(x))
    a = T.attention(fq.(x), k, v, hz, 1, 1, 1.0 / :math.sqrt(c))
    name(lets, name <> ".out", fo.(a))
  end

  defp pad_matrix(%Tensor{shape: [m, k]} = t, mp, kp) do
    v = t |> Tensor.widen() |> Tensor.to_floats() |> Enum.chunk_every(k)
    rows = for r <- 0..(mp - 1), do: (if r < m, do: Enum.at(v, r) ++ List.duplicate(0.0, kp - k), else: List.duplicate(0.0, kp))
    f32([mp, kp], List.flatten(rows))
  end

  # -------------------------------------------------------------- layout --

  @doc "A PyTorch `[C, H, W]` (or `[1, C, H, W]`) tensor as rows `f32[H·W, pad16(C)]`."
  def from_nchw(%Tensor{shape: shape} = t) do
    [c, h, w] = Enum.take(shape, -3)
    v = t |> Tensor.widen() |> Tensor.to_floats() |> List.to_tuple()
    cp = pad16(c)
    data = for y <- 0..(h - 1), x <- 0..(w - 1), ch <- 0..(cp - 1), do: (if ch < c, do: elem(v, (ch * h + y) * w + x), else: 0.0)
    {f32([h * w, cp], data), {h, w, c}}
  end

  @doc "Rows `f32[H·W, Cp]` back to a PyTorch `[C, H, W]` of the `c` real channels."
  def to_nchw(%Tensor{shape: [n, cp]} = t, {h, w, c}) when n == h * w do
    v = t |> Tensor.to_floats() |> List.to_tuple()
    f32([c, h, w], for(ch <- 0..(c - 1), y <- 0..(h - 1), x <- 0..(w - 1), do: elem(v, (y * w + x) * cp + ch)))
  end
end
