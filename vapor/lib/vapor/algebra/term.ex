defmodule Vapor.Algebra.Term do
  @moduledoc """
  The free many-sorted term algebra F_Σ(X) — *symbolic*, hence emittable.

  The predecessor carried Elixir closures inside terms (`{:iota, fn ... end, t}`),
  which no emitter can lower to machine code. Every operator here is a
  name with a fixed, substrate-independent semantics (`Vapor.Runtime.Oracle`).

  Terms (the generators of Σ, spec Definition 2.1):

    * `{:input, name, dtype, shape}`   — variable; dims may be `{:dyn, sym, max}`
    * `{:const, %Vapor.Tensor{}}`      — airlocked constant
    * `{:ew, op, [arg]}`               — ι: elementwise f32 homomorphism; `op` is
                                         a primitive or a canonical function
                                         (`Vapor.Canon`: exp, rcp, rsqrt, div,
                                         sigmoid, silu, max, min)
    * `{:reduce, :sum | :max, x}`      — canonical 16-lane reduction of the last
                                         axis (kept with extent 1)
    * `{:linear, x, w}`                — y = x·Wᵀ, W : f32[n,k] (canonical per-row
                                         order), x : f32[k] or f32[b,k]
    * `{:linear_masked, x, w, m}`      — row-predicated `linear`: row `i` is
                                         `x_i·Wᵀ` when `m_i ≠ 0` and `+0` otherwise
                                         (`m : f32[b,1]`; the sign of a zero mask
                                         is ignored). Bit-identical to `linear` on
                                         the active rows; inactive rows read no
                                         weight (sparse mixture-of-experts dispatch)
    * `{:linear_grouped, x, w, g}`     — block-diagonal `linear` with `g` groups:
                                         `x : f32[b, g·k]`, `W : f32[g·n, k]`,
                                         `y[:, i·n + j] = x[:, i·k …]·W[i·n + j]`
                                         (each a canonical dot product — the bits
                                         of `g` separate `linear`s); per-head
                                         projections such as MLA's absorbed
                                         query and value maps
    * `{:gather_row, table, idx}`      — rows of `table : f32[V,d]` at `idx : s32[T]`
    * `{:rope, x, cos, sin, pos, h}`   — rotary embedding of `x : f32[T, h·dh]`
                                         (rotate-half pairs `j, j + dh/2`),
                                         tables `f32[S, dh/2]`, `pos : s32[T]`
    * `{:kv_write, cache, pos, rows}`  — `cache : f32[S,n]` with row `pos[t]` ←
                                         `rows[t]` (a functional update; lowered
                                         in place when the old cache is dead)
    * `{:attention, q, k, v, pos, {h, hkv}}` — causal softmax attention of
                                         `q : f32[T, h·dh]` over cache rows
                                         `0 … pos[t]` of `k, v : f32[S, hkv·dh]`
                                         (grouped-query when `hkv < h`), scores
                                         scaled by 1/√dh — or by an explicit
                                         binary32 `s` with `{h, hkv, {:scale, s}}`;
                                         a trailing `{:window, w}` keeps only the
                                         last `w` positions (sliding window)
    * `{:kv_write_paged, pool, table, slot, pos, rows, page}` — paged cache:
                                         row `t` goes to logical row `pos[t]` of
                                         sequence `slot[t]`, i.e. pool row
                                         `table[slot, pos/page]·page + pos mod page`
                                         (skipped when any index is out of range)
    * `{:attention_paged, q, kpool, vpool, table, slot, pos, {h, hkv, page}}` —
                                         `attention` over the logical rows
                                         `0 … pos[t]` of sequence `slot[t]`
                                         (bit-identical to the contiguous form)
    * `{:transpose, x}`                — 2-D transpose
    * `{:reshape, x, shape}`           — the same row-major values under another
                                         static shape (an exact relabelling:
                                         im2col rows become convolution rows)
    * `{:sample, logits, params}`      — per row: greedy when `params[·,0] = +0`,
                                         else a categorical draw at temperature
                                         `1/params[·,0]` with uniform `params[·,1]`
    * `{:qgemv, w, x}`                 — contraction with an `:sb4x` matrix,
                                         `x : f32[k]` or `f32[b,k]` (row-wise)
    * `{:qgemv_masked, w, x, m}`       — row-predicated `qgemv`: row `i` is
                                         `qgemv(W, x_i)` when `m_i ≠ 0`, `+0`
                                         otherwise (`x : f32[b,k]`, `m : f32[b,1]`);
                                         the 4-bit twin of `linear_masked`
    * `{:gemm_i8, a, w}`               — contraction over (ℤ, +, ×): C = A·Wᵀ, s8 → s32

  An `:ew` argument is a term or `{:splat, bits}` (a broadcast binary32).
  Operands of `:ew` broadcast NumPy-style at equal rank: an extent-1 axis
  stretches to the other operand's extent (`[r,1]` against `[r,c]` is a
  per-row scalar, `[1,c]` a shared row). Nothing broadcasts implicitly
  across ranks.

  Indices are read as unsigned 32-bit and clamped to the table (`gather_row`,
  `rope`, `attention`); `kv_write` ignores rows whose position is outside the
  cache. Every operator is total: no input can make a kernel address memory
  outside its buffers, and the oracle defines the same result.

  Semi-dynamic dimensions: `{:dyn, :S, 4096}` is a symbolic extent bound at
  run time and certified at its maximum (every resource and error bound is
  monotone in the extent), so one certificate covers every `S ≤ max`.
  """
  import Bitwise
  alias Vapor.{Rejection, Tensor}

  defguardp has_window(heads)
            when is_tuple(heads) and tuple_size(heads) >= 3 and is_tuple(elem(heads, tuple_size(heads) - 1)) and
                   elem(elem(heads, tuple_size(heads) - 1), 0) == :window

  @ew_arity Map.merge(%{add: 2, sub: 2, mul: 2, fma: 3, neg: 1, relu: 1}, Vapor.Canon.functions())
  def ew_ops, do: Map.keys(@ew_arity)
  def ew_arity(op), do: Map.fetch!(@ew_arity, op)

  # ----------------------------------------------------------- constructors --

  def input(name, dtype, shape), do: {:input, name, dtype, shape}

  @doc """
  The leaf that refers to a let-binding `name = term` (`Vapor.Program`): an
  input of the term's sort that the program itself supplies.
  """
  def ref(name, term) do
    {:ok, {dt, shape}} = infer(term)
    {:input, name, dt, shape}
  end
  def const(%Tensor{} = t), do: {:const, t}
  def splat(x) when is_float(x) or is_integer(x), do: {:splat, Vapor.F32.from_float(x)}
  def ew(op, args) when is_list(args), do: {:ew, op, args}
  def add(a, b), do: ew(:add, [a, b])
  def sub(a, b), do: ew(:sub, [a, b])
  def mul(a, b), do: ew(:mul, [a, b])
  def neg(a), do: ew(:neg, [a])
  def relu(a), do: ew(:relu, [a])
  @doc "a·b + c. Canonical policy: two roundings; `:fast`: fused, one rounding."
  def fma(a, b, c), do: ew(:fma, [a, b, c])
  def exp(a), do: ew(:exp, [a])
  def rcp(a), do: ew(:rcp, [a])
  def rsqrt(a), do: ew(:rsqrt, [a])
  def divide(a, b), do: ew(:div, [a, b])
  def sigmoid(a), do: ew(:sigmoid, [a])
  def silu(a), do: ew(:silu, [a])
  def max(a, b), do: ew(:max, [a, b])
  def tanh(a), do: ew(:tanh, [a])
  def gelu_tanh(a), do: ew(:gelu_tanh, [a])
  @doc "GELU, exact (erf) form — PyTorch's default `gelu`."
  def gelu(a), do: ew(:gelu, [a])
  @doc "Natural logarithm (IEEE specials; under an ulp on normal operands)."
  def log(a), do: ew(:log, [a])
  @doc """
  softplus(x) = ln(1 + eˣ), PyTorch's form (β 1, threshold 20) — Mamba's Δ:
  `x` above 20, else log1p(exp(x)), the log1p by Goldberg's correction
  `u·ln(w)/(w − 1)` with `w = 1 + u` (and `u` itself where `w` rounds to
  1) — within a few ulps (measured in the test suite). A composition of
  canonical nodes, not one microprogram: each node fits a kernel's
  registers on every target. `exp`'s operand is clamped, so a discarded
  branch never overflows.
  """
  def softplus(x) do
    u = exp(ew(:min, [x, splat(20.0)]))
    w = add(splat(1.0), u)
    d = sub(w, splat(1.0))
    tiny = splat(:math.pow(2, -24))
    r = divide(mul(u, log(w)), sel(d, tiny, splat(1.0), d))
    sel(splat(20.0), x, x, sel(d, tiny, u, r))
  end
  @doc "(a < b) ? x : y, elementwise with broadcasting (ordered compare; NaN selects y)."
  def sel(a, b, x, y), do: ew(:sel, [a, b, x, y])
  def min(a, b), do: ew(:min, [a, b])
  def reduce(op, x) when op in [:sum, :max], do: {:reduce, op, x}
  def linear(x, w), do: {:linear, x, w}

  @doc """
  `linear` predicated per row: rows whose mask is zero (either sign) are `+0`
  and cost nothing — no weight is read for them. On the active rows the bits
  are those of `linear/2` (the same canonical dot product).
  """
  def linear_masked(x, w, m), do: {:linear_masked, x, w, m}

  @doc """
  `g` independent linear maps side by side: group `i` maps columns
  `i·k … i·k + k − 1` of `x` by rows `i·n … i·n + n − 1` of `W`.
  """
  def linear_grouped(x, w, g) when is_integer(g) and g > 0, do: {:linear_grouped, x, w, g}
  def gather_row(table, idx), do: {:gather_row, table, idx}
  def rope(x, cos, sin, pos, heads), do: {:rope, x, cos, sin, pos, heads}
  def kv_write(cache, pos, rows), do: {:kv_write, cache, pos, rows}
  @doc """
  Causal attention. `scale` (a number) replaces 1/√dh; `window` (a positive
  integer) restricts row `t` to the last `window` positions,
  `max(0, p − window + 1) … p` — the sliding window of Mistral, Gemma 2/3
  and Phi-3, in the same canonical order (only the range changes).
  """
  def attention(q, k, v, pos, heads, kv_heads, scale \\ nil, window \\ nil),
    do: {:attention, q, k, v, pos, {heads, kv_heads} |> with_scale(scale) |> with_window(window)}

  def kv_write_paged(pool, table, slot, pos, rows, page), do: {:kv_write_paged, pool, table, slot, pos, rows, page}

  def attention_paged(q, kpool, vpool, table, slot, pos, heads, kv_heads, page, scale \\ nil, window \\ nil),
    do: {:attention_paged, q, kpool, vpool, table, slot, pos, {heads, kv_heads, page} |> with_scale(scale) |> with_window(window)}

  # nil keeps the default 1/√dh (and the term's original shape); a number is
  # rounded to binary32 once, here, and travels as bits
  defp with_scale(heads, nil), do: heads
  defp with_scale(heads, s) when is_number(s), do: Tuple.insert_at(heads, tuple_size(heads), {:scale, Vapor.F32.from_float(s)})

  defp with_window(heads, nil), do: heads
  defp with_window(heads, w) when is_integer(w), do: Tuple.insert_at(heads, tuple_size(heads), {:window, w})

  @doc """
  The binary32 bits of an attention node's score scale: explicit
  (`{:scale, bits}` in its head tuple) or 1/√dh.
  """
  def attention_scale_bits(heads, dh) do
    Enum.find_value(Tuple.to_list(heads), Vapor.Runtime.Oracle.attention_scale(dh), fn
      {:scale, b} -> b
      _ -> nil
    end)
  end

  @doc "An attention node's sliding window (`{:window, w}` in its head tuple), or `nil`."
  def attention_window(heads) do
    Enum.find_value(Tuple.to_list(heads), fn
      {:window, w} -> w
      _ -> nil
    end)
  end

  @doc "The head tuple without its options: `{h, hkv}` or `{h, hkv, page}`."
  def heads_of(heads), do: heads |> Tuple.to_list() |> Enum.reject(&is_tuple/1) |> List.to_tuple()

  @doc "Next-token choice per row of `logits` with `params = (1/T, u)` (see `Vapor.Runtime.Oracle`)."
  def sample(logits, params), do: {:sample, logits, params}

  @doc "2-D transpose (exact: a permutation of the values)."
  def transpose(x), do: {:transpose, x}

  @doc "The values of `x` (row-major) under a static `shape` of the same size."
  def reshape(x, shape) when is_list(shape), do: {:reshape, x, shape}

  def qgemv(w, x), do: {:qgemv, w, x}
  def qgemv_masked(w, x, m), do: {:qgemv_masked, w, x, m}
  def gemm_i8(a, w), do: {:gemm_i8, a, w}

  # --------------------------------------------------------- shapes & dims --

  def dyn(sym, max) when is_atom(sym) and is_integer(max) and max > 0, do: {:dyn, sym, max}

  @doc "Upper bound of a dimension (certification extent)."
  def dim_max({:dyn, _, m}), do: m
  def dim_max(n) when is_integer(n), do: n

  def shape_max(s), do: Enum.map(s, &dim_max/1)

  defp dim?({:dyn, s, m}), do: is_atom(s) and is_integer(m) and m > 0
  defp dim?(n), do: is_integer(n) and n > 0

  # ------------------------------------------------------ Rung 1: sorting --

  @doc """
  Infer `{dtype, shape}` or return the Axiom 1 counterexample (the violating
  node, the bound, and a repair candidate).
  """
  @spec infer(term) :: {:ok, {atom, list}} | {:error, Rejection.t()}
  def infer({:input, _n, dt, s} = t) do
    if dt in Tensor.dtypes() and is_list(s) and s != [] and Enum.all?(s, &dim?/1),
      do: {:ok, {dt, s}},
      else: reject(t, "well-formed input sort", "declare a base dtype and positive dims")
  end

  def infer({:const, %Tensor{dtype: dt, shape: s}}), do: {:ok, {dt, s}}

  def infer({:ew, op, args} = t) do
    with {:ok, n} <- Map.fetch(@ew_arity, op) |> ok_or(t, "op ∈ #{inspect(ew_ops())}"),
         true <- length(args) == n || reject(t, "arity(#{op}) = #{n}", "fix operand count"),
         {:ok, sorts} <- infer_all(Enum.reject(args, &match?({:splat, _}, &1))),
         true <- sorts != [] || reject(t, "at least one tensor operand", "splat-only ew is a constant"),
         true <- Enum.all?(sorts, &match?({:f32, _}, &1)) || reject(t, "operands of sort f32", "insert an explicit cast") do
      case broadcast_shapes(Enum.map(sorts, &elem(&1, 1))) do
        {:ok, s} -> {:ok, {:f32, s}}
        :error -> reject(t, "operands broadcast at equal rank (extent 1 stretches)", "reshape or reduce operands to compatible extents")
      end
    end
  end

  def infer({:reduce, op, x} = t) when op in [:sum, :max] do
    with {:ok, {:f32, s}} <- infer(x) |> expect_f32(t, "x : f32[…, n]"),
         n = List.last(s),
         true <- (is_integer(n) and rem(n, 16) == 0) ||
                   reject(t, "reduced extent static and ≡ 0 (mod 16), got #{inspect(n)}", "pad the axis to a multiple of 16") do
      {:ok, {:f32, List.replace_at(s, -1, 1)}}
    end
  end

  def infer({:gather_row, table, idx} = t) do
    with {:ok, {dt, [_v, d]}} <- infer(table) |> shape2(t, "table : f32[V,d]"),
         true <- dt in [:f32, :bf16] || reject(t, "table : f32 or bf16", "convert the table"),
         true <- (dt == :f32 or rem(d, 16) == 0) || reject(t, "bf16 rows of 16·n values", "keep this table in f32"),
         true <- is_integer(d) || reject(t, "static row width", "fix the row width"),
         {:ok, {:s32, [n]}} <- infer(idx) |> vec_of(:s32, t, "idx : s32[T]") do
      {:ok, {:f32, [n, d]}}
    end
  end

  def infer({:rope, x, cos, sin, pos, h} = t) do
    with {:ok, {:f32, [n, w]}} <- infer(x) |> shape2(t, "x : f32[T, h·dh]"),
         true <- (is_integer(h) and h > 0 and is_integer(w) and rem(w, 2 * h) == 0) ||
                   reject(t, "h·dh = #{inspect(w)} with dh even", "check the head count"),
         half = div(w, 2 * h),
         {:ok, {:f32, [s, ^half]}} <- infer(cos) |> sort_is(fn {dt, sh} -> dt == :f32 and match?([m, ^half] when is_integer(m), sh) end, t, "cos : f32[S, #{half}]"),
         {:ok, _} <- infer(sin) |> sort_is(&(&1 == {:f32, [s, half]}), t, "sin : f32[#{s}, #{half}]"),
         {:ok, _} <- infer(pos) |> sort_is(&(&1 == {:s32, [n]}), t, "pos : s32[#{inspect(n)}] (one per row of x)") do
      {:ok, {:f32, [n, w]}}
    end
  end

  def infer({:kv_write, cache, pos, rows} = t) do
    with {:ok, {:f32, [s, w]}} <- infer(cache) |> shape2(t, "cache : f32[S,n]"),
         true <- (is_integer(s) and is_integer(w)) || reject(t, "static cache extents", "size the cache at its maximum"),
         {:ok, {:f32, [n, ^w]}} <- infer(rows) |> sort_is(fn {dt, sh} -> dt == :f32 and match?([_, ^w], sh) end, t, "rows : f32[T, #{w}]"),
         {:ok, _} <- infer(pos) |> sort_is(&(&1 == {:s32, [n]}), t, "pos : s32[#{inspect(n)}]") do
      {:ok, {:f32, [s, w]}}
    end
  end

  # paged KV: a pool of P pages of `page` rows; sequence `slot` owns pages
  # table[slot, 0..MP-1]; its logical row j lives at page table[slot, j/page],
  # offset j mod page
  def infer({:kv_write_paged, pool, table, slot, pos, rows, page} = t) do
    with :ok <- page_ok(page, t),
         {:ok, {:f32, [pr, w]}} <- infer(pool) |> shape2(t, "pool : f32[P·page, n]"),
         true <- (is_integer(pr) and is_integer(w) and rem(pr, page) == 0) || reject(t, "static pool of whole pages", "size the pool as P·page rows"),
         {:ok, {:s32, [_, mp]}} <- infer(table) |> shape2(t, "table : s32[NS, MP]"),
         true <- is_integer(mp) || reject(t, "static table", "fix the pages per sequence"),
         {:ok, {:f32, [n, ^w]}} <- infer(rows) |> sort_is(fn {dt, sh} -> dt == :f32 and match?([_, ^w], sh) end, t, "rows : f32[T, #{w}]"),
         {:ok, _} <- infer(pos) |> sort_is(&(&1 == {:s32, [n]}), t, "pos : s32[#{inspect(n)}]"),
         {:ok, _} <- infer(slot) |> sort_is(&(&1 == {:s32, [n]}), t, "slot : s32[#{inspect(n)}]") do
      {:ok, {:f32, [pr, w]}}
    end
  end

  def infer({:attention_paged, _q, _kp, _vp, _table, _slot, _pos, heads} = t) when has_window(heads),
    do: check_window(t, 7)

  def infer({:attention_paged, _q, _kp, _vp, _table, _slot, _pos, {h, hkv, page, {:scale, b}}} = t) do
    if is_integer(b) and b >= 0 and b < 0x1_0000_0000,
      do: infer(put_elem(t, 7, {h, hkv, page})),
      else: reject(t, "a binary32 scale", "pass the scale as a number")
  end

  def infer({:attention_paged, q, kp, vp, table, slot, pos, {h, hkv, page}} = t) do
    with :ok <- page_ok(page, t),
         {:ok, {:f32, [pr, kw]}} <- infer(kp) |> shape2(t, "kpool : f32[P·page, hkv·dh]"),
         true <- (is_integer(pr) and rem(pr, page) == 0) || reject(t, "static pool of whole pages", "size the pool as P·page rows"),
         {:ok, _} <- infer(vp) |> sort_is(&(&1 == {:f32, [pr, kw]}), t, "vpool : f32[#{pr}, #{kw}]"),
         {:ok, {:s32, [_, mp]}} <- infer(table) |> shape2(t, "table : s32[NS, MP]"),
         true <- is_integer(mp) || reject(t, "static table", "fix the pages per sequence"),
         {:ok, {:f32, [n, w]}} <- infer({:attention, q, {:input, :"$k", :f32, [mp * page, kw]}, {:input, :"$v", :f32, [mp * page, kw]}, pos, {h, hkv}}),
         {:ok, _} <- infer(slot) |> sort_is(&(&1 == {:s32, [n]}), t, "slot : s32[#{inspect(n)}]") do
      {:ok, {:f32, [n, w]}}
    end
  end

  def infer({:reshape, x, shape} = t) do
    with {:ok, {dt, xs}} <- infer(x),
         true <- (dt in [:f32, :s32] and Enum.all?(xs, &is_integer/1)) || reject(t, "a static f32 or s32 operand", "fix the operand's extents"),
         true <- (shape != [] and Enum.all?(shape, &(is_integer(&1) and &1 > 0)) and Enum.product(shape) == Enum.product(xs)) ||
                   reject(t, "a static shape of #{Enum.product(xs)} elements, got #{inspect(shape)}", "keep the element count") do
      {:ok, {dt, shape}}
    end
  end

  def infer({:transpose, x} = t) do
    with {:ok, {:f32, [r, c]}} <- infer(x) |> shape2(t, "x : f32[R, C]"),
         true <- (is_integer(r) and is_integer(c)) || reject(t, "static extents", "fix the shape") do
      {:ok, {:f32, [c, r]}}
    end
  end

  def infer({:sample, logits, params} = t) do
    with {:ok, {:f32, [b, v]}} <- infer(logits) |> shape2(t, "logits : f32[B, V]"),
         true <- (is_integer(v) and rem(v, 16) == 0) || reject(t, "V static and ≡ 0 (mod 16)", "pad the vocabulary"),
         {:ok, _} <- infer(params) |> sort_is(&(&1 == {:f32, [b, 2]}), t, "params : f32[#{inspect(b)}, 2] (1/T, u)") do
      {:ok, {:s32, [b]}}
    end
  end

  def infer({:attention, _q, _k, _v, _pos, heads} = t) when has_window(heads), do: check_window(t, 5)

  def infer({:attention, _q, _k, _v, _pos, {h, hkv, {:scale, b}}} = t) do
    if is_integer(b) and b >= 0 and b < 0x1_0000_0000,
      do: infer(put_elem(t, 5, {h, hkv})),
      else: reject(t, "a binary32 scale", "pass the scale as a number")
  end

  def infer({:attention, q, k, v, pos, {h, hkv}} = t) do
    with true <- (is_integer(h) and is_integer(hkv) and hkv > 0 and rem(h, hkv) == 0) ||
                   reject(t, "h a multiple of hkv", "check the (grouped) head counts"),
         {:ok, {:f32, [n, w]}} <- infer(q) |> shape2(t, "q : f32[T, h·dh]"),
         true <- (is_integer(w) and rem(w, h) == 0 and rem(div(w, h), 16) == 0) ||
                   reject(t, "dh = (h·dh)/h static and ≡ 0 (mod 16)", "pad the head dimension"),
         kw = hkv * div(w, h),
         {:ok, {:f32, [s, ^kw]}} <- infer(k) |> sort_is(fn {dt, sh} -> dt == :f32 and match?([m, ^kw] when is_integer(m), sh) end, t, "k : f32[S, #{kw}]"),
         {:ok, _} <- infer(v) |> sort_is(&(&1 == {:f32, [s, kw]}), t, "v : f32[#{s}, #{kw}]"),
         {:ok, _} <- infer(pos) |> sort_is(&(&1 == {:s32, [n]}), t, "pos : s32[#{inspect(n)}]") do
      {:ok, {:f32, [n, w]}}
    end
  end

  def infer({:linear, x, w} = t) do
    with {:ok, {_, ws}} <- infer(w) |> weights_f32(t, "W : f32[n,k] or bf16[n,k]"),
         [n, k] <- ws,
         true <- (is_integer(k) and rem(k, 16) == 0) || reject(t, "k static and ≡ 0 (mod 16)", "pad the contraction axis"),
         {:ok, {:f32, xs}} <- infer(x) |> expect_f32(t, "x : f32[k] or f32[b,k]"),
         true <- (length(xs) in [1, 2] and List.last(xs) == k) ||
                   reject(t, "x : f32[#{k}] or f32[b,#{k}], got #{inspect(xs)}", "align the contraction axis") do
      {:ok, {:f32, List.replace_at(xs, -1, n)}}
    else
      [_ | _] -> reject(t, "W : f32[n,k]", "reshape W to rank 2")
      e -> e
    end
  end

  def infer({:linear_masked, x, w, m} = t) do
    with {:ok, {:f32, ys}} <- infer({:linear, x, w}),
         true <- length(ys) == 2 || reject(t, "x : f32[b,k] (one mask entry per row)", "use linear/2 for a single row"),
         [b, _] = ys,
         {:ok, _} <- infer(m) |> sort_is(&(&1 == {:f32, [b, 1]}), t, "m : f32[#{inspect(b)}, 1]") do
      {:ok, {:f32, ys}}
    end
  end

  def infer({:linear_grouped, x, w, g} = t) do
    with {:ok, {_, ws}} <- infer(w) |> weights_f32(t, "W : f32[g·n,k] or bf16[g·n,k]"),
         [gn, k] <- ws,
         true <- (is_integer(k) and rem(k, 16) == 0) || reject(t, "k static and ≡ 0 (mod 16)", "pad the contraction axis"),
         true <- (is_integer(gn) and rem(gn, g) == 0) || reject(t, "W rows a multiple of g = #{g}", "check the group count"),
         {:ok, {:f32, xs}} <- infer(x) |> expect_f32(t, "x : f32[g·k] or f32[b,g·k]"),
         true <- (length(xs) in [1, 2] and List.last(xs) == g * k) ||
                   reject(t, "x : f32[b,#{g * k}], got #{inspect(xs)}", "align the grouped contraction axis") do
      {:ok, {:f32, List.replace_at(xs, -1, gn)}}
    else
      [_ | _] -> reject(t, "W : f32[g·n,k]", "reshape W to rank 2")
      e -> e
    end
  end

  def infer({:qgemv, w, x} = t) do
    with {:ok, {wd, [rows, k]}} <- infer(w),
         true <- wd in [:sb4, :sb4x] || reject(t, "W : sb4[rows,k]", "quantize with Vapor.Quant.Sb4"),
         {:ok, {:f32, xs}} <- infer(x) |> f32_rows(t),
         true <- List.last(xs) == k || reject(t, "contraction axis #{k} = #{inspect(List.last(xs))}", "align K") do
      {:ok, {:f32, Enum.drop(xs, -1) ++ [rows]}}
    end
  end

  def infer({:qgemv_masked, w, x, m} = t) do
    with {:ok, {:f32, ys}} <- infer({:qgemv, w, x}),
         true <- length(ys) == 2 || reject(t, "x : f32[b,k] (one mask entry per row)", "use qgemv/2 for a single row"),
         [b, _] = ys,
         {:ok, _} <- infer(m) |> sort_is(&(&1 == {:f32, [b, 1]}), t, "m : f32[#{inspect(b)}, 1]") do
      {:ok, {:f32, ys}}
    end
  end

  def infer({:gemm_i8, a, w} = t) do
    with {:ok, {:s8, [m, k]}} <- infer(a) |> expect(t, "A : s8[m,k]"),
         {:ok, {:s8, [n, k2]}} <- infer(w) |> expect(t, "W : s8[n,k]"),
         true <- k2 == k || reject(t, "contraction axis #{inspect(k)} = #{inspect(k2)}", "align K") do
      {:ok, {:s32, [m, n]}}
    end
  end

  def infer(t), do: reject(t, "generator of Σ", "use Vapor.Algebra.Term constructors")

  # the window is the last element of the head tuple; checked, then dropped
  defp check_window(t, i) do
    heads = elem(t, i)
    {:window, w} = elem(heads, tuple_size(heads) - 1)

    if is_integer(w) and w > 0,
      do: infer(put_elem(t, i, Tuple.delete_at(heads, tuple_size(heads) - 1))),
      else: reject(t, "a positive window", "pass the sliding window as a positive integer")
  end

  defp infer_all(ts) do
    Enum.reduce_while(ts, {:ok, []}, fn t, {:ok, acc} ->
      case infer(t) do
        {:ok, s} -> {:cont, {:ok, acc ++ [s]}}
        e -> {:halt, e}
      end
    end)
  end

  defp expect({:ok, {d, [_, _]}} = ok, _t, _b) when d == :s8, do: ok
  defp expect({:error, _} = e, _t, _b), do: e
  defp expect(_, t, b), do: reject(t, b, "cast operands to s8")

  defp page_ok(page, t) do
    if is_integer(page) and page > 0 and (page &&& (page - 1)) == 0,
      do: :ok,
      else: reject(t, "page size a power of two", "choose 16, 32, …")
  end

  defp shape2({:ok, {_, [_, _]}} = ok, _t, _b), do: ok
  defp shape2({:error, _} = e, _t, _b), do: e
  defp shape2(_, t, b), do: reject(t, b, "reshape the operand")

  defp vec_of({:ok, {dt, [_]}} = ok, dt, _t, _b), do: ok
  defp vec_of({:error, _} = e, _dt, _t, _b), do: e
  defp vec_of(_, _dt, t, b), do: reject(t, b, "pass one index per row")

  defp sort_is({:ok, sort} = ok, pred, t, b), do: if(pred.(sort), do: ok, else: reject(t, b, "fix the operand's dtype and extents"))
  defp sort_is({:error, _} = e, _pred, _t, _b), do: e

  # weights may be stored as bf16: they mean their (exact) f32 widening
  defp weights_f32({:ok, {dt, _}} = ok, _t, _b) when dt in [:f32, :bf16], do: ok
  defp weights_f32(other, t, b), do: expect_f32(other, t, b)

  defp expect_f32({:ok, {:f32, _}} = ok, _t, _b), do: ok
  defp expect_f32({:error, _} = e, _t, _b), do: e
  defp expect_f32(_, t, b), do: reject(t, b, "use binary32 operands")

  @doc "NumPy broadcasting at equal rank: extent-1 axes stretch."
  def broadcast_shapes([s | rest]) do
    Enum.reduce_while(rest, {:ok, s}, fn r, {:ok, acc} ->
      if length(r) != length(acc) do
        {:halt, :error}
      else
        dims = Enum.zip_with(acc, r, fn a, b -> cond do
            a == b -> a
            a == 1 -> b
            b == 1 -> a
            true -> :error
          end end)

        if :error in dims, do: {:halt, :error}, else: {:cont, {:ok, dims}}
      end
    end)
  end

  defp f32_rows({:ok, {:f32, s}} = ok, _t) when length(s) in [1, 2], do: ok
  defp f32_rows({:error, _} = e, _t), do: e
  defp f32_rows(_, t), do: reject(t, "x : f32[k] or f32[b,k]", "reshape x")

  defp ok_or({:ok, v}, _t, _b), do: {:ok, v}
  defp ok_or(:error, t, b), do: reject(t, b, "choose a supported operator")

  defp reject(node, bound, repair),
    do: {:error, %Rejection{node: node, bound: bound, repair: repair}}

  # ---------------------------------------------------------- traversal --

  @doc "Children of a term (splats are leaves of `:ew`, not terms)."
  def children({:ew, _, args}), do: Enum.reject(args, &match?({:splat, _}, &1))
  def children({:reduce, _, x}), do: [x]
  def children({:linear, x, w}), do: [x, w]
  def children({:linear_masked, x, w, m}), do: [x, w, m]
  def children({:linear_grouped, x, w, _g}), do: [x, w]
  def children({:gather_row, table, idx}), do: [table, idx]
  def children({:rope, x, c, s, p, _h}), do: [x, c, s, p]
  def children({:kv_write, cache, pos, rows}), do: [cache, pos, rows]
  def children({:attention, q, k, v, pos, _h}), do: [q, k, v, pos]
  def children({:sample, l, p}), do: [l, p]
  def children({:transpose, x}), do: [x]
  def children({:reshape, x, _}), do: [x]
  def children({:kv_write_paged, pool, table, slot, pos, rows, _}), do: [pool, table, slot, pos, rows]
  def children({:attention_paged, q, kp, vp, table, slot, pos, _}), do: [q, kp, vp, table, slot, pos]
  def children({:qgemv, w, x}), do: [w, x]
  def children({:qgemv_masked, w, x, m}), do: [w, x, m]
  def children({:gemm_i8, a, w}), do: [a, w]
  def children(_), do: []

  def leaf?({:input, _, _, _}), do: true
  def leaf?({:const, _}), do: true
  def leaf?(_), do: false

  @doc "Children-first enumeration of the distinct subterms of a DAG (hash-consed)."
  def postorder(roots) when is_list(roots) do
    {order, _} = Enum.reduce(roots, {[], MapSet.new()}, &visit/2)
    Enum.reverse(order)
  end

  def postorder(root), do: postorder([root])

  defp visit(t, {acc, seen}) do
    if MapSet.member?(seen, t) do
      {acc, seen}
    else
      {acc, seen} = Enum.reduce(children(t), {acc, seen}, &visit/2)
      {[t | acc], MapSet.put(seen, t)}
    end
  end

end
