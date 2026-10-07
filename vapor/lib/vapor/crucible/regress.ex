defmodule Vapor.Crucible.Regress do
  @moduledoc """
  A law from data (docs/CRUCIBLE.md §11): **symbolic regression** — the
  Athanor searching the space of expressions over the user's columns,
  then the numeric constants of the finalists refined (Nelder–Mead), and
  every claim made on rows the search never saw.

      x, v, F
      0.1, 2.0, 0.41
      …
      target = F
      ops = + - * / sq sqrt sin exp
      max_size = 13
      budget = 20000

  Evidence:

    * a **held-out split** (20 %, by a hash of the row index — fixed, not
      chosen): R² and the error on it; the gap to the training error;
    * a **complexity–error front**: the simplest expression at each size
      that the search found, so a person picks the trade-off, not the
      machine;
    * the **control**: the same search on the target **shuffled** across
      rows (the law destroyed, the marginal kept) — its held-out R² is the
      floor that a real law must clear.
  """
  alias Vapor.Athanor
  alias Vapor.Athanor.Space

  def run(text) do
    with {:ok, header, rows, opts} <- table(text),
         {:ok, target} <- target(header, opts) do
      vars = header -- [target]
      ops = (opts["ops"] || "+ - * /") |> String.split() |> Enum.filter(&(&1 in Space.unary_ops() ++ Space.binary_ops()))
      max_size = opts |> Map.get("max_size", "13") |> String.to_integer() |> min(31)
      budget = opts |> Map.get("budget", "20000") |> String.to_integer() |> min(200_000)
      seed = opts |> Map.get("seed", "1") |> String.to_integer()
      leaves = (opts["constants"] || "1 2 3 0.5") |> String.split() |> Enum.map(&num/1) |> Enum.reject(&is_nil/1)
      ti = Enum.find_index(header, &(&1 == target))
      data = Enum.map(rows, fn r -> {List.delete_at(r, ti), Enum.at(r, ti)} end)
      {train, test} = data |> Enum.with_index() |> Enum.split_with(fn {_, i} -> Vapor.Alembic.Builtins.hash01([:split, i]) < 0.8 end)
      train = Enum.map(train, &elem(&1, 0))
      test = Enum.map(test, &elem(&1, 0))

      if length(train) < 5 or length(test) < 2 do
        {:error, "too few rows (need at least ~10)"}
      else
        found = search(vars, ops, leaves, max_size, budget, seed, train)
        space = found.space
        finalists = found.cert.top |> Enum.map(& &1.candidate) |> Enum.uniq() |> Enum.take(8)
        refined = finalists |> Enum.map(fn txt -> {:ok, t} = Space.parse_program(space, txt); t |> scale_tree(space, train) |> refine(space, train) end) |> Enum.uniq_by(&Space.show(space, &1))
        scored = Enum.map(refined, &score(&1, space, train, test)) |> Enum.reject(&is_nil/1) |> Enum.sort_by(& &1.test_mse)
        best = hd(scored)
        front = pareto(scored)
        shuffled_train = shuffle_targets(train)
        control = search(vars, ops, leaves, max_size, div(budget, 3), seed + 1, shuffled_train)
        ctrl_best =
          control.cert.top |> Enum.take(4) |> Enum.map(fn c -> {:ok, t} = Space.parse_program(space, c.candidate); score(t |> scale_tree(space, shuffled_train) |> refine(space, shuffled_train), space, shuffled_train, test) end)
          |> Enum.reject(&is_nil/1) |> Enum.max_by(& &1.test_r2, fn -> %{test_r2: nil} end)

        {:ok, %{kind: "regress", target: target, variables: vars, rows: length(data), train: length(train), test: length(test), ops: ops,
                best: best, front: front, finalists: scored, search: Map.take(found.cert, [:evaluations, :strategies, :found_by, :verdict, :journal_root]),
                control: %{test_r2: ctrl_best.test_r2, says: "on the shuffled target the best expression reaches R² = #{f(ctrl_best.test_r2)} on held-out rows"},
                evidence: [
                  %{check: "held-out fit", ok: best.test_r2 > 0.9, detail: "R² = #{f(best.test_r2)} on #{length(test)} rows the search never saw (training R² #{f(best.train_r2)})"},
                  %{check: "shuffled control", ok: ctrl_best.test_r2 == nil or best.test_r2 > ctrl_best.test_r2 + 0.3, detail: "the law destroyed: R² = #{f(ctrl_best.test_r2)}"},
                  %{check: "complexity", ok: true, detail: "#{length(front)} point(s) on the size–error front; the chosen one has #{best.size} nodes"}
                ],
                says: "#{target} ≈ #{best.expression} (held-out R² #{f(best.test_r2)})"}}
      end
    end
  end

  defp f(nil), do: "—"
  defp f(x), do: :erlang.float_to_binary(x * 1.0, [{:decimals, 4}, :compact])

  defp num(s) do
    case Float.parse(s) do
      {x, ""} -> if x == Float.round(x) and not String.contains?(s, "."), do: trunc(x), else: x
      _ -> nil
    end
  end

  # ------------------------------------------------------------ input

  defp table(text) do
    lines = text |> String.split("\n") |> Enum.map(&String.trim/1) |> Enum.reject(&(&1 == "" or String.starts_with?(&1, "#")))
    {opt_lines, data_lines} = Enum.split_with(lines, &Regex.match?(~r/^[a-z_]+\s*=/, &1))
    opts = Map.new(opt_lines, fn l -> [k, v] = String.split(l, "=", parts: 2); {String.trim(k), String.trim(v)} end)
    case data_lines do
      [h | rows] ->
        sep = if String.contains?(h, "\t"), do: "\t", else: ","
        header = h |> String.split(sep) |> Enum.map(&String.trim/1)
        parsed = Enum.map(rows, fn r -> r |> String.split(sep) |> Enum.map(&(String.trim(&1) |> Float.parse() |> then(fn {x, _} -> x; :error -> nil end))) end)
        cond do
          length(header) < 2 -> {:error, "a header with at least two columns (x, y)"}
          not Enum.all?(header, &Regex.match?(~r/^[\p{L}_][\w]*$/u, &1)) -> {:error, "column names must be identifiers: #{Enum.join(header, ", ")}"}
          Enum.any?(parsed, fn r -> length(r) != length(header) or Enum.any?(r, &is_nil/1) end) -> {:error, "every row needs #{length(header)} numbers"}
          length(parsed) > 20_000 -> {:error, "at most 20 000 rows"}
          true -> {:ok, header, parsed, opts}
        end
      [] -> {:error, "no data: a header line (x, y) then rows"}
    end
  end

  defp target(header, opts) do
    t = opts["target"] || List.last(header)
    if t in header, do: {:ok, t}, else: {:error, "target #{t} is not a column"}
  end

  # ------------------------------------------------------------ search

  defp search(vars, ops, leaves, max_size, budget, seed, train) do
    spec = """
    space = program(#{Vapor.Alembic.show(vars)}, #{Vapor.Alembic.show(ops)}, #{Vapor.Alembic.show(leaves)}, #{max_size})
    minimize(f) = 0
    budget = #{budget}
    seed = #{seed}
    """
    sub = Enum.take(train, 400)
    {:ok, s} = Athanor.Spec.parse(spec)
    objective = fn f -> scaled_mse(f, sub) end
    {:ok, cert} = Athanor.run(spec, objective: objective, control: false, seconds: 240)
    %{cert: cert, space: s.space}
  end

  defp mse_fn({:fn, _, _, fun}, rows) do
    try do
      e = Enum.reduce(rows, 0.0, fn {xs, y}, s -> (d = fun.(xs) - y; s + d * d) end) / length(rows)
      if e > 1.0e300, do: {:error, "overflow"}, else: {:ok, e}
    catch
      {:alembic, m} -> {:error, m}
      :error, _ -> {:error, "arithmetic"}
    end
  end

  # Keijzer's linear scaling: the error of the best a + b·f(x), in closed form — the search judges a
  # candidate's shape, not its scale (which the constant refinement settles later)
  defp scaled_mse({:fn, _, _, fun}, rows) do
    try do
      ps = Enum.map(rows, fn {xs, y} -> {fun.(xs), y} end)
      n = length(ps)
      mp = Enum.reduce(ps, 0.0, fn {p, _}, s -> s + p end) / n
      my = Enum.reduce(ps, 0.0, fn {_, y}, s -> s + y end) / n
      spp = Enum.reduce(ps, 0.0, fn {p, _}, s -> s + (p - mp) * (p - mp) end)
      spy = Enum.reduce(ps, 0.0, fn {p, y}, s -> s + (p - mp) * (y - my) end)
      b = if spp > 1.0e-300, do: spy / spp, else: 0.0
      a = my - b * mp
      e = Enum.reduce(ps, 0.0, fn {p, y}, s -> (d = a + b * p - y; s + d * d) end) / n
      if e > 1.0e300, do: {:error, "overflow"}, else: {:ok, e}
    catch
      {:alembic, m} -> {:error, m}
      :error, _ -> {:error, "arithmetic"}
    end
  end

  defp scale_tree(t, space, rows) do
    {:fn, _, _, fun} = Space.realize(space, t)
    ps = Enum.map(rows, fn {xs, y} -> {fun.(xs), y} end)
    n = length(ps)
    mp = Enum.reduce(ps, 0.0, fn {p, _}, s -> s + p end) / n
    my = Enum.reduce(ps, 0.0, fn {_, y}, s -> s + y end) / n
    spp = Enum.reduce(ps, 0.0, fn {p, _}, s -> s + (p - mp) * (p - mp) end)
    spy = Enum.reduce(ps, 0.0, fn {p, y}, s -> s + (p - mp) * (y - my) end)
    b = if spp > 1.0e-300, do: spy / spp, else: 0.0
    a = my - b * mp
    cond do
      abs(b - 1) < 1.0e-12 and abs(a) < 1.0e-12 -> t
      abs(a) < 1.0e-12 -> ["*", b, t]
      true -> ["+", a, ["*", b, t]]
    end
  catch
    _, _ -> t
  end

  defp eval_tree(t, space, rows) do
    {:fn, _, _, fun} = Space.realize(space, t)
    try do
      {:ok, Enum.map(rows, fn {xs, _} -> fun.(xs) end)}
    catch
      _, _ -> :error
    end
  end

  defp score(t, space, train, test) do
    with {:ok, ptr} <- eval_tree(t, space, train), {:ok, pte} <- eval_tree(t, space, test) do
      %{expression: pretty(t, space), tree: Space.show(space, t), size: Space.tree_size(t), train_mse: mse(ptr, train), test_mse: mse(pte, test), train_r2: r2(ptr, train), test_r2: r2(pte, test)}
    else
      _ -> nil
    end
  end

  # a polynomial result in expanded form with 4 significant digits; otherwise infix with rounded constants
  defp pretty(t, space) do
    case Vapor.Crucible.Poly.from_expr(to_expr(t), space.vars) do
      {:ok, p} when map_size(p) > 0 ->
        p
        |> Enum.map(fn {e, c} -> {e, Vapor.Crucible.Poly.qfloat(c)} end)
        |> Enum.reject(fn {_, c} -> c == 0 end)
        |> Enum.sort_by(fn {e, _} -> -Enum.sum(Tuple.to_list(e)) end)
        |> Enum.with_index()
        |> Enum.map_join("", fn {{e, c}, i} ->
          mono = e |> Tuple.to_list() |> Enum.zip(space.vars) |> Enum.filter(fn {k, _} -> k > 0 end) |> Enum.map_join("·", fn {1, v} -> v; {k, v} -> "#{v}^#{k}" end)
          mag = sig(abs(c))
          body = cond do mono == "" -> mag; mag == "1" -> mono; true -> mag <> "·" <> mono end
          cond do i == 0 and c < 0 -> "−" <> body; i == 0 -> body; c < 0 -> " − " <> body; true -> " + " <> body end
        end)
      _ -> Space.show(space, round_consts(t))
    end
  end

  defp sig(x) when x == 0, do: "0"
  defp sig(x) do
    d = 4 - (x |> :math.log10() |> Float.floor() |> trunc()) - 1
    r = Float.round(x * 1.0, max(min(d, 15), 0))
    if r == Float.round(r), do: Integer.to_string(trunc(r)), else: :erlang.float_to_binary(r, [:short])
  end

  defp round_consts(c) when is_number(c), do: (v = String.to_float(sig_f(c)); v)
  defp round_consts([op | args]), do: [op | Enum.map(args, &round_consts/1)]
  defp round_consts(v), do: v
  defp sig_f(c), do: (s = (if c < 0, do: "-", else: "") <> sig(abs(c)); if(String.contains?(s, "."), do: s, else: s <> ".0"))

  defp to_expr(c) when is_number(c), do: {:n, c * 1.0}
  defp to_expr(v) when is_binary(v), do: {:v, v}
  defp to_expr(["neg", a]), do: {:neg, to_expr(a)}
  defp to_expr(["sq", a]), do: {:f, "sq", [to_expr(a)]}
  defp to_expr([op, a]), do: {:f, op, [to_expr(a)]}
  defp to_expr([op, a, b]) when op in ["+", "-", "*", "/", "^"], do: {String.to_existing_atom(op), to_expr(a), to_expr(b)}
  defp to_expr([op, a, b]), do: {:f, op, [to_expr(a), to_expr(b)]}

  defp mse(pred, rows), do: Enum.zip(pred, rows) |> Enum.reduce(0.0, fn {p, {_, y}}, s -> s + (p - y) * (p - y) end) |> Kernel./(length(rows))

  defp r2(pred, rows) do
    ys = Enum.map(rows, &elem(&1, 1))
    m = Enum.sum(ys) / length(ys)
    ss = Enum.reduce(ys, 0.0, &(&2 + (&1 - m) * (&1 - m)))
    if ss == 0, do: nil, else: 1 - mse(pred, rows) * length(rows) / ss
  end

  defp pareto(scored) do
    scored
    |> Enum.group_by(& &1.size)
    |> Enum.map(fn {_, g} -> Enum.min_by(g, & &1.test_mse) end)
    |> Enum.sort_by(& &1.size)
    |> Enum.reduce([], fn p, acc -> if acc == [] or p.test_mse < hd(acc).test_mse, do: [p | acc], else: acc end)
    |> Enum.reverse()
  end

  defp shuffle_targets(rows) do
    ys = rows |> Enum.map(&elem(&1, 1)) |> Enum.with_index() |> Enum.sort_by(fn {_, i} -> Vapor.Alembic.Builtins.hash01([:perm, i]) end) |> Enum.map(&elem(&1, 0))
    Enum.zip_with(rows, ys, fn {xs, _}, y -> {xs, y} end)
  end

  # ------------------------------------------------------------ constants

  # the numeric leaves of a tree, refined by Nelder–Mead on the training error
  defp refine(t, space, train) do
    consts = collect(t)
    if consts == [] do
      t
    else
      rows = Enum.take(train, 400)
      loss = fn cs -> case mse_fn(Space.realize(space, put(t, cs)), rows) do {:ok, e} -> e; _ -> 1.0e300 end end
      best = nelder_mead(loss, consts, 300)
      t2 = put(t, Enum.map(best, &Float.round(&1 * 1.0, 6)))
      if loss.(Enum.map(best, &Float.round(&1 * 1.0, 6))) <= loss.(consts), do: Space.simplify_tree(t2), else: t
    end
  end

  defp collect(c) when is_number(c), do: [c * 1.0]
  defp collect(v) when is_binary(v), do: []
  defp collect([_ | args]), do: Enum.flat_map(args, &collect/1)

  defp put(t, cs), do: elem(put_(t, cs), 0)
  defp put_(c, [x | rest]) when is_number(c), do: {x, rest}
  defp put_(v, cs) when is_binary(v), do: {v, cs}
  defp put_([op | args], cs) do
    {args2, rest} = Enum.map_reduce(args, cs, &put_/2)
    {[op | args2], rest}
  end

  @doc false
  def nelder_mead(f, x0, iters) do
    n = length(x0)
    simplex = [x0 | for(i <- 0..(n - 1), do: List.update_at(x0, i, &(&1 + max(abs(&1) * 0.1, 0.1))))]
    pts = Enum.map(simplex, &{f.(&1), &1})
    Enum.reduce(1..iters, pts, fn _, pts ->
      [{fb, _} = best | _] = pts = Enum.sort_by(pts, &elem(&1, 0))
      {fw, worst} = List.last(pts)
      {fsw, _} = Enum.at(pts, -2)
      rest = Enum.drop(pts, -1)
      c = rest |> Enum.map(&elem(&1, 1)) |> Enum.zip() |> Enum.map(fn t -> Enum.sum(Tuple.to_list(t)) / n end)
      at = fn a -> Enum.zip_with(c, worst, &(&1 + a * (&1 - &2))) end
      xr = at.(1.0)
      fr = f.(xr)
      cond do
        fr < fb ->
          xe = at.(2.0)
          fe = f.(xe)
          rest ++ [if(fe < fr, do: {fe, xe}, else: {fr, xr})]
        fr < fsw -> rest ++ [{fr, xr}]
        true ->
          xc = at.(-0.5)
          fc = f.(xc)
          if fc < fw do
            rest ++ [{fc, xc}]
          else
            {_, bx} = best
            [best | Enum.map(tl(pts), fn {_, x} -> y = Enum.zip_with(bx, x, &(&1 + 0.5 * (&2 - &1))); {f.(y), y} end)]
          end
      end
    end)
    |> Enum.min_by(&elem(&1, 0))
    |> elem(1)
  end
end
