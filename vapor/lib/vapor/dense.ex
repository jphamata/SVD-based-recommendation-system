defmodule Vapor.Dense do
  @moduledoc """
  Dense linear algebra in binary64 for the workbench's solvers
  (docs/BANCADA.md §6): LU with partial pivoting (real and complex), the
  symmetric eigenproblem by cyclic Jacobi (and the generalized one,
  K φ = λ M φ, by Cholesky reduction), conjugate gradients for sparse
  symmetric positive systems, the tridiagonal (Thomas) solve, and least
  squares by Householder QR.

  Matrices are lists of rows (lists of floats); complex numbers are
  `{re, im}`. Every solve can return its **residual** ‖Ax − b‖∞ / (‖A‖∞‖x‖∞ +
  ‖b‖∞), which the solvers report next to their results — the backward
  error is the certificate a reader can check without trusting the
  factorisation.
  """

  # ------------------------------------------------------------- basics --

  def zeros(n, m), do: List.duplicate(List.duplicate(0.0, m), n)
  def identity(n), do: for(i <- 0..(n - 1), do: for(j <- 0..(n - 1), do: if(i == j, do: 1.0, else: 0.0)))
  def matvec(a, x), do: Enum.map(a, &dot(&1, x))
  def dot(a, b), do: dot(a, b, 0.0)
  defp dot([x | a], [y | b], acc), do: dot(a, b, acc + x * y)
  defp dot([], [], acc), do: acc
  def transpose([]), do: []
  def transpose(a), do: Enum.zip_with(a, & &1)
  def matmul(a, b), do: (bt = transpose(b); Enum.map(a, fn r -> Enum.map(bt, &dot(r, &1)) end))
  def norm_inf(v) when is_list(v), do: Enum.reduce(v, 0.0, fn x, m -> max(m, absx(x)) end)
  defp absx({re, im}), do: :math.sqrt(re * re + im * im)
  defp absx(x), do: abs(x)
  def mat_norm_inf(a), do: a |> Enum.map(fn r -> r |> Enum.map(&absx/1) |> Enum.sum() end) |> Enum.max(fn -> 0.0 end)
  def axpy(a, x, y), do: Enum.zip_with(x, y, &(a * &1 + &2))
  def sub(x, y), do: Enum.zip_with(x, y, &(&1 - &2))

  @doc "The normwise backward error of x as a solution of Ax = b."
  def residual(a, x, b) do
    r = sub(matvec(a, x), b) |> norm_inf()
    r / max(mat_norm_inf(a) * norm_inf(x) + norm_inf(b), 1.0e-300)
  end

  # ------------------------------------------------------------------ LU --

  @doc """
  Solve A x = b by Gaussian elimination with partial pivoting:
  `{:ok, x}` or `{:error, {:singular, column}}` (a pivot below 10⁻¹⁴ of
  the column's scale). `b` may be a vector or a matrix (a list of rows; the solution is then a matrix X with A X = B).
  """
  def solve(a, b) do
    n = length(a)
    multi = match?([[_ | _] | _], b)
    rhs = if multi, do: b, else: Enum.map(b, &[&1])
    aug = Enum.zip_with(a, rhs, &(&1 ++ &2)) |> Enum.map(&List.to_tuple/1)
    scale = Enum.map(a, fn r -> r |> Enum.map(&abs/1) |> Enum.max(fn -> 0.0 end) end) |> Enum.max(fn -> 1.0 end)

    case eliminate(aug, 0, n, max(scale, 1.0e-300)) do
      {:ok, u} ->
        m = tuple_size(hd(u)) - n
        xs = back(Enum.reverse(u), n, m)
        {:ok, if(multi, do: xs, else: Enum.map(xs, &hd/1))}

      err ->
        err
    end
  end

  defp eliminate(rows, k, n, _s) when k == n, do: {:ok, rows}

  defp eliminate(rows, k, n, s) do
    {done, rest} = Enum.split(rows, k)
    {piv, pi} = rest |> Enum.with_index() |> Enum.max_by(fn {r, _} -> abs(elem(r, k)) end)

    if abs(elem(piv, k)) < 1.0e-14 * s do
      {:error, {:singular, k}}
    else
      rest = List.delete_at(rest, pi)
      p = elem(piv, k)
      w = tuple_size(piv)

      rest =
        Enum.map(rest, fn r ->
          f = elem(r, k) / p
          if f == 0.0, do: r, else: elim_row(r, piv, f, k, w)
        end)

      eliminate(done ++ [piv | rest], k + 1, n, s)
    end
  end

  defp elim_row(r, piv, f, k, w) do
    l = for j <- 0..(w - 1), do: (if j < k, do: elem(r, j), else: (if j == k, do: 0.0, else: elem(r, j) - f * elem(piv, j)))
    List.to_tuple(l)
  end

  # back substitution over the reversed upper-triangular rows: m right-hand sides
  defp back(rows_rev, n, m) do
    Enum.reduce(rows_rev, %{}, fn r, xs ->
      i = n - map_size(xs) - 1
      x = for c <- 0..(m - 1) do
        s = Enum.reduce((i + 1)..(n - 1)//1, elem(r, n + c), fn j, acc -> acc - elem(r, j) * Enum.at(xs[j], c) end)
        s / elem(r, i)
      end
      Map.put(xs, i, x)
    end)
    |> then(fn xs -> for i <- 0..(n - 1), do: xs[i] end)
  end

  @doc "The inverse of a square matrix: `{:ok, inv}` or `{:error, _}`."
  def inverse(a), do: solve(a, identity(length(a)))

  # ------------------------------------------------------------ complex --

  def cadd({a, b}, {c, d}), do: {a + c, b + d}
  def csub({a, b}, {c, d}), do: {a - c, b - d}
  def cmul({a, b}, {c, d}), do: {a * c - b * d, a * d + b * c}
  def cdiv({a, b}, {c, d}), do: (q = c * c + d * d; {(a * c + b * d) / q, (b * c - a * d) / q})
  def cabs({a, b}), do: :math.sqrt(a * a + b * b)
  def carg({a, b}), do: :math.atan2(b, a)

  @doc "Solve a complex system (entries `{re, im}`) by pivoted elimination: `{:ok, x}` or `{:error, _}`."
  def csolve(a, b) do
    n = length(a)
    rows = Enum.zip_with(a, b, fn r, x -> List.to_tuple(r ++ [x]) end)
    scale = a |> List.flatten() |> Enum.map(&cabs/1) |> Enum.max(fn -> 1.0 end)

    with {:ok, u} <- celim(rows, 0, n, max(scale, 1.0e-300)) do
      xs =
        u |> Enum.reverse() |> Enum.reduce(%{}, fn r, xs ->
          i = n - map_size(xs) - 1
          s = Enum.reduce((i + 1)..(n - 1)//1, elem(r, n), fn j, acc -> csub(acc, cmul(elem(r, j), xs[j])) end)
          Map.put(xs, i, cdiv(s, elem(r, i)))
        end)

      {:ok, for(i <- 0..(n - 1), do: xs[i])}
    end
  end

  defp celim(rows, k, n, _s) when k == n, do: {:ok, rows}

  defp celim(rows, k, n, s) do
    {done, rest} = Enum.split(rows, k)
    {piv, pi} = rest |> Enum.with_index() |> Enum.max_by(fn {r, _} -> cabs(elem(r, k)) end)

    if cabs(elem(piv, k)) < 1.0e-14 * s do
      {:error, {:singular, k}}
    else
      rest = List.delete_at(rest, pi)
      p = elem(piv, k)
      w = tuple_size(piv)
      rest = Enum.map(rest, fn r ->
        f = cdiv(elem(r, k), p)
        for(j <- 0..(w - 1), do: (if j < k, do: elem(r, j), else: (if j == k, do: {0.0, 0.0}, else: csub(elem(r, j), cmul(f, elem(piv, j)))))) |> List.to_tuple()
      end)
      celim(done ++ [piv | rest], k + 1, n, s)
    end
  end

  # -------------------------------------------------------- tridiagonal --

  @doc "Thomas algorithm: sub-diagonal `a` (a₀ unused), diagonal `b`, super-diagonal `c` (cₙ₋₁ unused), right side `d`."
  def tridiag(a, b, c, d) do
    {cp, dp} =
      Enum.zip([a, b, c, d])
      |> Enum.reduce({[], []}, fn {ai, bi, ci, di}, {cps, dps} ->
        case cps do
          [] -> {[ci / bi], [di / bi]}
          [cprev | _] ->
            m = bi - ai * cprev
            {[ci / m | cps], [(di - ai * hd(dps)) / m | dps]}
        end
      end)

    [xl | _] = dp
    Enum.zip(tl(cp), tl(dp)) |> Enum.reduce([xl], fn {ci, di}, [xn | _] = acc -> [di - ci * xn | acc] end)
  end

  # ----------------------------------------------------------- symmetric --

  @doc """
  Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations:
  `{values (ascending), vectors (columns, as a list of eigenvectors)}`.
  Converges quadratically; off-diagonal mass below 10⁻²⁴ of the total.
  """
  def eigh(a) do
    n = length(a)
    m = a |> Enum.map(&List.to_tuple/1) |> List.to_tuple()
    v = identity(n) |> Enum.map(&List.to_tuple/1) |> List.to_tuple()
    total = a |> List.flatten() |> Enum.map(&(&1 * &1)) |> Enum.sum()
    {m, v} = sweeps(m, v, n, max(total, 1.0e-300), 0)
    pairs = for i <- 0..(n - 1), do: {elem(elem(m, i), i), for(k <- 0..(n - 1), do: elem(elem(v, k), i))}
    pairs = Enum.sort_by(pairs, &elem(&1, 0))
    {Enum.map(pairs, &elem(&1, 0)), Enum.map(pairs, &elem(&1, 1))}
  end

  defp g(m, i, j), do: elem(elem(m, i), j)
  defp s(m, i, j, x), do: put_elem(m, i, put_elem(elem(m, i), j, x))

  defp sweeps(m, v, n, total, k) do
    off = for(i <- 0..(n - 1), j <- 0..(n - 1), i != j, do: g(m, i, j) * g(m, i, j)) |> Enum.sum()
    if off <= 1.0e-24 * total or k > 60 or n < 2 do
      {m, v}
    else
      {m, v} = Enum.reduce(for(p <- 0..(n - 2), q <- (p + 1)..(n - 1), do: {p, q}), {m, v}, fn {p, q}, {m, v} -> rotate(m, v, n, p, q) end)
      sweeps(m, v, n, total, k + 1)
    end
  end

  defp rotate(m, v, n, p, q) do
    apq = g(m, p, q)
    if abs(apq) < 1.0e-300 do
      {m, v}
    else
      theta = (g(m, q, q) - g(m, p, p)) / (2 * apq)
      t = (if theta >= 0, do: 1.0, else: -1.0) / (abs(theta) + :math.sqrt(theta * theta + 1))
      c = 1 / :math.sqrt(t * t + 1)
      sn = t * c
      # A' = Jᵀ A J on rows/columns p, q
      m = Enum.reduce(0..(n - 1), m, fn k, m ->
        akp = g(m, k, p); akq = g(m, k, q)
        m |> s(k, p, c * akp - sn * akq) |> s(k, q, sn * akp + c * akq)
      end)
      m = Enum.reduce(0..(n - 1), m, fn k, m ->
        apk = g(m, p, k); aqk = g(m, q, k)
        m |> s(p, k, c * apk - sn * aqk) |> s(q, k, sn * apk + c * aqk)
      end)
      v = Enum.reduce(0..(n - 1), v, fn k, v ->
        vkp = g(v, k, p); vkq = g(v, k, q)
        v |> s(k, p, c * vkp - sn * vkq) |> s(k, q, sn * vkp + c * vkq)
      end)
      {m, v}
    end
  end

  @doc "Cholesky factor L (A = L Lᵀ) of a symmetric positive definite matrix: `{:ok, l}` or `{:error, :not_positive_definite}`."
  def cholesky(a) do
    n = length(a)
    at = a |> Enum.map(&List.to_tuple/1) |> List.to_tuple()

    l =
      Enum.reduce(0..(n - 1), %{}, fn j, l ->
        sjj = g(at, j, j) - Enum.sum(for k <- 0..(j - 1)//1, do: l[{j, k}] * l[{j, k}])
        if sjj <= 0, do: throw(:npd)
        ljj = :math.sqrt(sjj)
        l = Map.put(l, {j, j}, ljj)
        Enum.reduce((j + 1)..(n - 1)//1, l, fn i, l ->
          Map.put(l, {i, j}, (g(at, i, j) - Enum.sum(for k <- 0..(j - 1)//1, do: l[{i, k}] * l[{j, k}])) / ljj)
        end)
      end)

    {:ok, for(i <- 0..(n - 1), do: for(j <- 0..(n - 1), do: Map.get(l, {i, j}, 0.0)))}
  catch
    :npd -> {:error, :not_positive_definite}
  end

  @doc """
  The generalized symmetric eigenproblem K φ = λ M φ (M positive
  definite): `{:ok, {values, vectors}}`, vectors M-orthonormal.
  """
  def geigh(k, m) do
    with {:ok, l} <- cholesky(m), {:ok, li} <- inverse(l) do
      c = li |> matmul(k) |> matmul(transpose(li))
      c = for {r, i} <- Enum.with_index(c), do: for({x, j} <- Enum.with_index(r), do: 0.5 * (x + Enum.at(Enum.at(c, j), i)))
      {vals, ys} = eigh(c)
      lit = transpose(li)
      {:ok, {vals, Enum.map(ys, &matvec(lit, &1))}}
    end
  end

  # ---------------------------------------------------------------- CG --

  @doc """
  Conjugate gradients for A x = b with A symmetric positive definite,
  given as a function `x ↦ A x` (so sparse operators never materialise).
  `{x, iterations, relative residual}`.
  """
  def cg(apply_a, b, opts \\ []) do
    tol = Keyword.get(opts, :tol, 1.0e-10)
    maxit = Keyword.get(opts, :max_iter, 10 * length(b))
    x = Keyword.get(opts, :x0, List.duplicate(0.0, length(b)))
    r = sub(b, apply_a.(x))
    bn = :math.sqrt(dot(b, b)) |> max(1.0e-300)
    cg_loop(apply_a, x, r, r, dot(r, r), bn, tol, maxit, 0)
  end

  defp cg_loop(a, x, r, p, rr, bn, tol, maxit, k) do
    if k >= maxit or :math.sqrt(rr) / bn < tol, do: {x, k, :math.sqrt(rr) / bn}, else: cg_step(a, x, r, p, rr, bn, tol, maxit, k)
  end

  defp cg_step(a, x, r, p, rr, bn, tol, maxit, k) do
    ap = a.(p)
    alpha = rr / dot(p, ap)
    x = axpy(alpha, p, x)
    r = axpy(-alpha, ap, r)
    rr2 = dot(r, r)
    p = axpy(rr2 / rr, p, r)
    cg_loop(a, x, r, p, rr2, bn, tol, maxit, k + 1)
  end

  # ------------------------------------------------------- least squares --

  @doc "Least squares min ‖A x − b‖₂ by Householder QR: `{:ok, x}` or `{:error, :rank_deficient}`."
  def lstsq(a, b) do
    m = length(a)
    n = length(hd(a))
    cols = transpose(a)
    {r_cols, qtb} = householder(cols, b, 0, n, m)
    r = transpose(r_cols)
    rr = Enum.take(r, n) |> Enum.map(&Enum.take(&1, n))
    if Enum.any?(0..(n - 1), fn i -> abs(Enum.at(Enum.at(rr, i), i)) < 1.0e-12 * (mat_norm_inf(a) + 1.0e-300) end) do
      {:error, :rank_deficient}
    else
      solve(rr, Enum.take(qtb, n))
    end
  end

  defp householder(cols, b, k, n, _m) when k == n, do: {cols, b}

  defp householder(cols, b, k, n, m) do
    col = Enum.at(cols, k)
    xk = Enum.drop(col, k)
    alpha = -sgn(hd(xk)) * :math.sqrt(dot(xk, xk))
    v = [hd(xk) - alpha | tl(xk)]
    vv = dot(v, v)
    reflect = fn y ->
      if vv == 0, do: y, else: (
        {top, bot} = Enum.split(y, k)
        f = 2 * dot(v, bot) / vv
        top ++ Enum.zip_with(bot, v, &(&1 - f * &2)))
    end
    householder(Enum.map(cols, reflect), reflect.(b), k + 1, n, m)
  end

  defp sgn(x) when x < 0, do: -1.0
  defp sgn(_), do: 1.0

  # ------------------------------------------------------ sparse SPD, banded --

  @doc """
  Solve K x = f for a sparse symmetric positive definite K given as a map
  `{i, j} => v` (both triangles or one, summed symmetrically is the
  caller's business: entries are taken as given for i ≥ j), by a
  reverse Cuthill–McKee ordering and a banded Cholesky factorisation.
  `{:ok, x, %{bandwidth, n}}` or `{:error, {:not_positive_definite, dof}}`.
  """
  def sparse_spd_solve(k, f) do
    n = length(f)
    adj = Enum.reduce(k, %{}, fn {{i, j}, _}, a -> if i == j, do: a, else: a |> Map.update(i, [j], &[j | &1]) |> Map.update(j, [i], &[i | &1]) end)
    perm = rcm(n, adj)
    inv = perm |> Enum.with_index() |> Map.new()
    bw = Enum.reduce(k, 0, fn {{i, j}, _}, m -> max(m, abs(inv[i] - inv[j])) end)
    # lower band, row r holds columns r-bw .. r as a map
    # the caller gives both triangles; the lower one in the new ordering is kept
    rows = Enum.reduce(k, %{}, fn {{i, j}, v}, acc ->
      {r, c} = {inv[i], inv[j]}
      if r >= c, do: Map.update(acc, {r, c}, v, &(&1 + v)), else: acc
    end)
    fp = Enum.map(perm, &Enum.at(f, &1))
    case band_chol(rows, n, bw) do
      {:ok, l} ->
        y = band_forward(l, fp, n, bw)
        x = band_back(l, y, n, bw)
        xt = List.to_tuple(x)
        {:ok, for(i <- 0..(n - 1), do: elem(xt, inv[i])), %{bandwidth: bw, n: n}}
      err -> err
    end
  end

  defp rcm(n, adj) do
    deg = fn v -> length(Map.get(adj, v, [])) end
    {order, _} =
      Enum.reduce(Enum.sort_by(0..(n - 1), deg), {[], MapSet.new()}, fn start, {order, seen} ->
        if MapSet.member?(seen, start), do: {order, seen}, else: bfs([start], MapSet.put(seen, start), order, adj, deg)
      end)
    # built by prepending: already the reverse of the Cuthill–McKee order
    order
  end

  defp bfs([], seen, order, _adj, _deg), do: {order, seen}
  defp bfs([v | q], seen, order, adj, deg) do
    nb = Map.get(adj, v, []) |> Enum.uniq() |> Enum.reject(&MapSet.member?(seen, &1)) |> Enum.sort_by(deg)
    bfs(q ++ nb, Enum.reduce(nb, seen, &MapSet.put(&2, &1)), [v | order], adj, deg)
  end

  defp band_chol(a, n, bw) do
    l =
      Enum.reduce(0..(n - 1), %{}, fn j, l ->
        lo = max(0, j - bw)
        s = Map.get(a, {j, j}, 0.0) - Enum.reduce(lo..(j - 1)//1, 0.0, fn k, acc -> (x = Map.get(l, {j, k}, 0.0); acc + x * x) end)
        if s <= 0, do: throw({:npd, j})
        d = :math.sqrt(s)
        l = Map.put(l, {j, j}, d)
        Enum.reduce((j + 1)..min(n - 1, j + bw)//1, l, fn i, l ->
          lo2 = max(0, i - bw)
          v = Map.get(a, {i, j}, 0.0) - Enum.reduce(max(lo, lo2)..(j - 1)//1, 0.0, fn k, acc -> acc + Map.get(l, {i, k}, 0.0) * Map.get(l, {j, k}, 0.0) end)
          if v == 0.0, do: l, else: Map.put(l, {i, j}, v / d)
        end)
      end)
    {:ok, l}
  catch
    {:npd, j} -> {:error, {:not_positive_definite, j}}
  end

  defp band_forward(l, b, n, bw) do
    Enum.reduce(0..(n - 1), {%{}, List.to_tuple(b)}, fn i, {y, bt} ->
      s = Enum.reduce(max(0, i - bw)..(i - 1)//1, elem(bt, i), fn k, acc -> acc - Map.get(l, {i, k}, 0.0) * y[k] end)
      {Map.put(y, i, s / l[{i, i}]), bt}
    end)
    |> elem(0) |> then(fn y -> for i <- 0..(n - 1), do: y[i] end)
  end

  defp band_back(l, y, n, bw) do
    yt = List.to_tuple(y)
    Enum.reduce((n - 1)..0//-1, %{}, fn i, x ->
      s = Enum.reduce((i + 1)..min(n - 1, i + bw)//1, elem(yt, i), fn k, acc -> acc - Map.get(l, {k, i}, 0.0) * x[k] end)
      Map.put(x, i, s / l[{i, i}])
    end)
    |> then(fn x -> for i <- 0..(n - 1), do: x[i] end)
  end
end
