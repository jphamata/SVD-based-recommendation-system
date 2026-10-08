defmodule Vapor.Cupel do
  @moduledoc """
  The **cupel** — the porous dish of the assayer, where base metal soaks
  away and noble metal stays — for silent data corruption (docs/CUPEL.md).

  The pain: at fleet scale, a defective core returns *wrong numbers without
  an error* (Meta's and Google's "silent data corruption at scale"
  reports: about one machine in a thousand). Training absorbs it as a loss
  spike days later; inference serves it. Replicating every product
  (compare two runs bit for bit — what `Vapor.Cluster` already does) costs
  2×. A check must be **cheaper than the product** and must **never accuse
  a correct substrate**.

  First principles. For `y = x·Wᵀ` (`x : [b, k]`, `W : [n, k]`) and any
  vector `r`, `y·r = x·(Wᵀr)` — the adjoint identity ⟨Wx, r⟩ = ⟨x, Wᵀr⟩.
  `Wᵀr` depends on the weights only: computed once per weight matrix (the
  *probe*), it turns every later check into `O(b·(n + k))` against the
  product's `O(b·n·k)` (Freivalds, 1977). Two refinements make it a
  verdict rather than a heuristic:

    * **exact arithmetic for the comparison** — `ŷ·r` and `x·(Wᵀr)` are
      computed exactly over the dyadic rationals (the cells of
      `Vapor.Amalgam`), so the only slack is the substrate's own rounding;
    * **a proved tolerance, not a tuned one** — Higham's Lemma 3.1 (in
      `proofs/Vapor/Higham.lean`): any conforming substrate, any summation
      order, fused or not, satisfies `|ŷᵢⱼ − yᵢⱼ| ≤ γₖ·Σₗ|xᵢₗ||Wⱼₗ| + 2k·η`
      (η = 2⁻¹²⁶ covers flush-to-zero), plus the operands a DAZ substrate
      may read as zero. Projected on `|r|`, the bound costs the same
      `O(b·(n + k))` through `|W|ᵀ|r|`. A row whose discrepancy exceeds it
      **cannot** have come from a correct substrate.

  `r` has integer entries `±[1, 2²⁰]` drawn from a seed: a single corrupted
  element is caught whenever `|δ|·|rⱼ| > 2·tol` (always, if `δ` is above
  the rounding envelope); several conspiring elements cancel with
  probability ≈ 2⁻²⁰ per row, and not at all against an adversary who does
  not know the seed. For `s8` weights and activations (the int8 GEMM,
  `s32` results) the check is exact: zero tolerance.

  What it does **not** see, and says: a corruption smaller than the
  rounding envelope is indistinguishable from rounding — measured per bit
  position by `sensitivity/4` (the low mantissa bits of large outputs go
  unseen; sign, exponent and high mantissa bits never do).
  """
  import Bitwise
  alias Vapor.{Amalgam, Tensor}
  alias Vapor.Verify.Dyadic, as: D

  defmodule Probe do
    @moduledoc "What one weight matrix needs for every later check: `Wᵀr`, `|W|ᵀ|r|`, and the seed."
    defstruct [:dtype, :n, :k, :seed, :r, :wr, :awr, :asub, :rsum, :wmax, :digest]
  end

  @rmax 1 <<< 20
  @s 149

  # ----------------------------------------------------------------- probe

  @doc """
  The probe of a weight matrix `W : [n, k]` (`:f32`, `:bf16` or `:s8`).
  Option `seed:` (default 1) — keep it secret to make cancellation
  unforgeable. Costs one pass over `W` (`O(n·k)` exact integer work).
  """
  def probe(%Tensor{shape: [n, k], dtype: dt} = w, opts \\ []) when dt in [:f32, :bf16, :s8] do
    seed = Keyword.get(opts, :seed, 1)
    r = draws(seed, n)
    rows = rows(w)
    zero = List.duplicate(0, k)

    {wr, awr, asub, wmax} =
      Enum.zip(rows, r)
      |> Enum.reduce({zero, zero, zero, 0}, fn {row, rj}, {wr, awr, asub, wmax} ->
        ar = abs(rj)

        {wr, awr, asub} =
          Enum.zip_with([row, wr, awr, asub], fn [v, a, b, c] ->
            {a + v * rj, b + abs(v) * ar, if(subnormal?(dt, v), do: c + abs(v) * ar, else: c)}
          end)
          |> unzip3()

        {wr, awr, asub, Enum.reduce(row, wmax, &max(abs(&1), &2))}
      end)

    %Probe{dtype: dt, n: n, k: k, seed: seed, r: r, wr: wr, awr: awr, asub: asub, rsum: Enum.sum(Enum.map(r, &abs/1)), wmax: wmax,
           digest: :crypto.hash(:sha256, w.data)}
  end

  defp unzip3(list) do
    {a, b, c} = Enum.reduce(list, {[], [], []}, fn {x, y, z}, {a, b, c} -> {[x | a], [y | b], [z | c]} end)
    {Enum.reverse(a), Enum.reverse(b), Enum.reverse(c)}
  end

  @doc "The probe's integer vector `r` (`n` entries in `±[1, 2²⁰]`), a pure function of the seed."
  def draws(seed, n) do
    {xs, _} =
      Enum.map_reduce(1..n//1, seed * 0x9E37_79B9_7F4A_7C15 &&& 0xFFFF_FFFF_FFFF_FFFF, fn _, s ->
        {v, s} = Tensor.splitmix(s)
        mag = (v &&& @rmax - 1) + 1
        {if((v >>> 63 &&& 1) == 1, do: -mag, else: mag), s}
      end)

    xs
  end

  # rows of W as scaled integers (f32/bf16: multiples of 2^-149; s8: themselves)
  defp rows(%Tensor{dtype: :s8, shape: [_n, k]} = w), do: w |> Tensor.to_list() |> Enum.chunk_every(k)

  defp rows(%Tensor{shape: [_n, k]} = w) do
    w |> Tensor.widen() |> Tensor.to_list() |> Enum.map(&finite_cell!/1) |> Enum.chunk_every(k)
  end

  defp finite_cell!(bits) do
    case Amalgam.cell(:f32, bits) do
      :nzero -> 0
      c when is_integer(c) -> c
      _ -> raise ArgumentError, "a weight matrix with a non-finite entry has no envelope to check against"
    end
  end

  defp subnormal?(:s8, _), do: false
  defp subnormal?(_, v), do: v != 0 and abs(v) < 1 <<< 23

  # ----------------------------------------------------------------- assay

  @doc """
  Check a claimed `y = x·Wᵀ` against a probe. `x : [b, k]`, `y : [b, n]`
  (`f32`, or `s8`/`s32` for an `s8` probe). Returns `{:ok, report}` when
  every row is within its proved envelope, `{:corrupt, report}` otherwise.

  `report.rows` lists, per row, `%{row, verdict, ratio}` with `verdict` one
  of `:ok`, `:corrupt` (outside the envelope, or a non-finite output that
  finite inputs could not produce), `:unchecked` (non-finite inputs, or
  outputs that may legitimately overflow — never silently counted as ok);
  `ratio` is discrepancy ÷ tolerance (`< 1` passes).
  """
  def assay(%Probe{} = p, %Tensor{shape: [b, k]} = x, %Tensor{shape: [b, n]} = y) when k == p.k and n == p.n do
    xs = x |> as_ints(p.dtype) |> Enum.chunk_every(k)
    ys = y |> as_ints(p.dtype) |> Enum.chunk_every(n)
    rows = Enum.zip_with([xs, ys, 0..(b - 1)], fn [xr, yr, i] -> row(p, xr, yr, i) end)
    corrupt = Enum.filter(rows, &(&1.verdict == :corrupt))
    report = %{rows: rows, corrupt: Enum.map(corrupt, & &1.row), unchecked: for(r <- rows, r.verdict == :unchecked, do: r.row),
               worst: rows |> Enum.map(& &1.ratio) |> Enum.reject(&is_nil/1) |> Enum.max(fn -> 0.0 end), seed: p.seed}
    if corrupt == [], do: {:ok, report}, else: {:corrupt, report}
  end

  def assay(%Probe{} = p, %Tensor{shape: xs}, %Tensor{shape: ys}),
    do: raise(ArgumentError, "probe for W:[#{p.n}, #{p.k}] given x:#{inspect(xs)}, y:#{inspect(ys)}")

  @doc "Probe and assay in one call (for a single check; reuse the probe for many)."
  def check(w, x, y, opts \\ []), do: assay(probe(w, opts), x, y)

  defp as_ints(%Tensor{dtype: dt} = t, :s8) when dt in [:s8, :s32], do: Tensor.to_list(t)
  defp as_ints(%Tensor{} = t, :s8), do: raise(ArgumentError, "an s8 probe checks s8 activations and s32 results, not #{t.dtype}")
  defp as_ints(%Tensor{} = t, _), do: t |> Tensor.widen() |> Tensor.to_list() |> Enum.map(&Amalgam.cell(:f32, &1))

  # exact integer check
  defp row(%Probe{dtype: :s8} = p, xr, yr, i) do
    lhs = dot(yr, p.r)
    rhs = dot(xr, p.wr)
    if lhs == rhs, do: %{row: i, verdict: :ok, ratio: 0.0}, else: %{row: i, verdict: :corrupt, ratio: :infinity, discrepancy: lhs - rhs}
  end

  defp row(%Probe{} = p, xr, yr, i) do
    cond do
      Enum.any?(xr, &(not is_integer(&1) and &1 != :nzero)) ->
        %{row: i, verdict: :unchecked, ratio: nil, why: :nonfinite_input}

      Enum.any?(yr, &(not is_integer(&1) and &1 != :nzero)) ->
        # finite inputs overflow only if Σ|x|·max|W| reaches the largest float
        if may_overflow?(p, xr), do: %{row: i, verdict: :unchecked, ratio: nil, why: :may_overflow},
                                 else: %{row: i, verdict: :corrupt, ratio: :infinity, why: :nonfinite_output}

      true ->
        xr = Enum.map(xr, &zero/1)
        yr = Enum.map(yr, &zero/1)
        # scale 2^-298 on both sides
        lhs = dot(yr, p.r) <<< @s
        rhs = dot(xr, p.wr)
        diff = abs(lhs - rhs)
        tol = tolerance(p, xr)
        ratio = if tol == 0, do: if(diff == 0, do: 0.0, else: :infinity), else: ratio(diff, tol)
        verdict = if diff <= tol, do: :ok, else: :corrupt
        %{row: i, verdict: verdict, ratio: ratio}
    end
  end

  defp zero(:nzero), do: 0
  defp zero(v), do: v

  defp dot(a, b), do: Enum.zip_reduce(a, b, 0, fn u, v, acc -> acc + u * v end)

  # γₖ·Σ|x||W|ᵀ|r| + DAZ terms + 2k·η·Σ|r|, at scale 2^-298, rounded up
  defp tolerance(p, xr) do
    ax = Enum.map(xr, &abs/1)
    s1 = dot(ax, p.awr)
    daz_w = dot(ax, p.asub)
    daz_x = Enum.zip_reduce(ax, p.awr, 0, fn a, w, acc -> if a != 0 and a < 1 <<< 23, do: acc + a * w, else: acc end)
    {g, ge} = D.gamma(p.k)
    gs1 = ceil_shift(g * s1, -ge)
    eta = 2 * p.k * p.rsum <<< (2 * @s - 126)
    gs1 + daz_w + daz_x + eta
  end

  defp ceil_shift(v, sh) do
    q = v >>> sh
    if q <<< sh == v, do: q, else: q + 1
  end

  defp may_overflow?(p, xr) do
    sx = xr |> Enum.map(&zero/1) |> Enum.map(&abs/1) |> Enum.sum()
    # Σ|x|·max|W| ≥ 2^128 (both at scale 2^-149)
    sx * p.wmax >= 1 <<< (128 + 2 * @s)
  end

  defp ratio(diff, tol) do
    # a float, for reports; exactness lives in the comparison above
    shift = max(max(bitlen(diff), bitlen(tol)) - 60, 0)
    (diff >>> shift) / max(tol >>> shift, 1)
  end

  # within 8 bits of the bit length: enough to scale a ratio for a report
  defp bitlen(0), do: 0
  defp bitlen(v), do: byte_size(:binary.encode_unsigned(v)) * 8

  # ------------------------------------------------------- sensitivity --

  @doc """
  What the cupel can and cannot see, measured: flip bit `b` (0–31) of one
  output element in each of `trials` correct products, and count the
  detections. Returns `[{bit, detected, trials}]`. Correct products come
  from `runner.(x)` (default: the exact oracle).
  """
  def sensitivity(%Tensor{} = w, xs, opts \\ []) when is_list(xs) do
    p = probe(w, Keyword.take(opts, [:seed]))
    runner = Keyword.get(opts, :runner, &oracle_linear(w, &1))
    bits = Keyword.get(opts, :bits, 0..31)
    correct = Enum.map(xs, fn x -> {x, runner.(x)} end)

    for b <- bits do
      hits =
        Enum.with_index(correct)
        |> Enum.count(fn {{x, y}, t} ->
          [rows, cols] = y.shape
          pos = rem(t * 7919, rows * cols)
          match?({:corrupt, _}, assay(p, x, flip(y, pos, b)))
        end)

      {b, hits, length(correct)}
    end
  end

  @doc "Flip bit `b` of element `pos` of an f32 tensor — a simulated silicon fault."
  def flip(%Tensor{dtype: :f32, data: d} = t, pos, b) do
    <<pre::binary-size(pos * 4), v::32-little, rest::binary>> = d
    %{t | data: <<pre::binary, bxor(v, 1 <<< b)::32-little, rest::binary>>}
  end

  def flip(%Tensor{dtype: :s32, data: d} = t, pos, b) do
    <<pre::binary-size(pos * 4), v::32-little, rest::binary>> = d
    %{t | data: <<pre::binary, bxor(v, 1 <<< b)::32-little, rest::binary>>}
  end

  @doc "`x·Wᵀ` by the exact oracle — the reference a substrate is compared with."
  def oracle_linear(%Tensor{shape: [_n, k]} = w, %Tensor{shape: [b, k]} = x) do
    p = Vapor.Program.new(y: Vapor.Algebra.Term.linear(Vapor.Algebra.Term.input(:x, :f32, [b, k]), Vapor.Algebra.Term.const(w)))
    Vapor.Runtime.Oracle.eval_program(p, %{x: x}).y
  end

  @doc "`x·Wᵀ` in exact integers for `s8` operands (`s32` result) — the int8 GEMM's definition."
  def int_linear(%Tensor{dtype: :s8, shape: [n, k]} = w, %Tensor{dtype: :s8, shape: [b, k]} = x) do
    wr = w |> Tensor.to_list() |> Enum.chunk_every(k)
    out = for xr <- Enum.chunk_every(Tensor.to_list(x), k), wj <- wr, do: dot(xr, wj)
    Tensor.from_list(:s32, [b, n], out)
  end
end
