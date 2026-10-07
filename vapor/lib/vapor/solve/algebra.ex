defmodule Vapor.Solve.Algebra do
  @moduledoc """
  Algebraic problems typed in as text (docs/BANCADA.md §4): systems of
  nonlinear equations (Newton with the symbolic Jacobian and a
  backtracking line search; with `search = [a, b]`, every root a
  quasi-random multistart finds in the box, deduplicated), nonlinear
  least-squares fits (Levenberg–Marquardt, with standard errors from the
  covariance, R² and the residuals) and constrained minimisation
  (BFGS inside an augmented Lagrangian, with the KKT residuals reported).

  Each answer comes with what lets a reader judge it: the residual of
  every equation at the root, the parameters' uncertainties, the
  stationarity and feasibility of the optimum.
  """
  alias Vapor.{Dense, Expr, Solve}

  @id "[\\p{L}_][\\p{L}\\p{N}_]*"

  # ============================================================ nonlinear

  @doc """
  A nonlinear system:

      unknowns x = 2, y = 0.5
      x^2 + y^2 = 4
      x*y = 1
      search = [-3, 3]      # optional: find every root in the box

  `{:ok, %{roots: [%{values, residual, iterations}], equations, …}}`.
  """
  def nonlinear(text) do
    with {:ok, p} <- parse_system(text) do
      names = p.unknowns
      res = Enum.map(p.equations, fn {l, r} -> Expr.simplify({:-, l, r}) end)
      f = Expr.compile(res, names)
      jac = Expr.compile(for(r <- res, u <- names, do: Expr.diff(r, u)), names)
      n = length(names)
      newton = fn x0 -> newton(f, fn x -> jac.(List.to_tuple(x)) |> Enum.chunk_every(n) end, x0, 0) end

      starts =
        case p.search do
          nil -> [p.guess]
          {lo, hi} -> [p.guess | for(k <- 1..60, do: for(d <- 0..(n - 1), do: lo + (hi - lo) * halton(k, Enum.at([2, 3, 5, 7, 11, 13, 17, 19], rem(d, 8)))))]
        end

      roots =
        starts
        |> Enum.map(newton)
        |> Enum.filter(&match?({:ok, _}, &1))
        |> Enum.map(&elem(&1, 1))
        |> Enum.reduce([], fn r, acc -> if Enum.any?(acc, &close?(&1.x, r.x)), do: acc, else: acc ++ [r] end)
        |> Enum.filter(fn r -> p.search == nil or Enum.all?(r.x, fn v -> v >= elem(p.search, 0) - 1.0e-9 and v <= elem(p.search, 1) + 1.0e-9 end) end)
        |> Enum.sort_by(& &1.x)

      if roots == [] do
        {:error, "Newton did not converge from #{if p.search, do: "#{length(starts)} starts", else: "the guess"} (try another guess, or search = [a, b])"}
      else
        {:ok, %{unknowns: names, equations: Enum.map(res, &(Expr.to_text(&1) <> " = 0")),
                roots: Enum.map(roots, fn r -> %{values: Map.new(Enum.zip(names, r.x)), residual: r.res, iterations: r.its, residuals: r.each} end),
                starts: length(starts)}}
      end
    end
  end

  defp close?(a, b), do: Enum.zip(a, b) |> Enum.all?(fn {x, y} -> abs(x - y) <= 1.0e-7 * max(1.0, abs(x)) end)

  @doc false
  def halton(k, b), do: halton(k, b, 1 / b, 0.0)
  defp halton(0, _b, _f, acc), do: acc
  defp halton(k, b, f, acc), do: halton(div(k, b), b, f / b, acc + f * rem(k, b))

  defp newton(f, jac, x, its) do
    fx = safe(fn -> f.(List.to_tuple(x)) end)
    cond do
      fx == :error -> :error
      norm(fx) < 1.0e-12 -> {:ok, %{x: x, res: norm(fx), its: its, each: fx}}
      its >= 100 -> :error
      true ->
        with j when j != :error <- safe(fn -> jac.(x) end),
             {:ok, dx} <- solve_or_lsq(j, Enum.map(fx, &(-&1))) do
          step = Enum.reduce_while([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 1.0e-2, 1.0e-3], nil, fn a, _ ->
            xn = Enum.zip_with(x, dx, &(&1 + a * &2))
            case safe(fn -> f.(List.to_tuple(xn)) end) do
              :error -> {:cont, nil}
              fn_ -> if norm(fn_) < (1 - 1.0e-4 * a) * norm(fx), do: {:halt, xn}, else: {:cont, nil}
            end
          end)
          case step do
            nil -> if norm(fx) < 1.0e-9, do: {:ok, %{x: x, res: norm(fx), its: its, each: fx}}, else: :error
            xn -> newton(f, jac, xn, its + 1)
          end
        else
          _ -> :error
        end
    end
  end

  defp solve_or_lsq(j, b) do
    if length(j) == length(hd(j)), do: Dense.solve(j, b), else: Dense.lstsq(j, b)
  end

  defp safe(fun) do
    fun.()
  rescue
    ArithmeticError -> :error
  end

  defp norm(v), do: Enum.reduce(v, 0.0, &max(abs(&1), &2))

  defp parse_system(text) do
    stmts = Solve.statements(text)
    {decl, rest} = Enum.split_with(stmts, fn {s, _} -> s =~ ~r/^(unknowns?|inc[oó]gnitas?|solve|resolva)\b/iu end)

    with [{d, _} | _] <- decl |> (fn [] -> {:error, "declare the unknowns: unknowns x = 1, y = 2"}; l -> l end).(),
         {:ok, unknowns, guess} <- parse_decl(d) do
      Enum.reduce_while(rest, {:ok, %{unknowns: unknowns, guess: guess, equations: [], params: %{}, search: nil}}, fn {s, line}, {:ok, acc} ->
        cond do
          m = Regex.run(~r/^(?:search|busca)\s*=\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\]$/u, s) ->
            with {:ok, a} <- const(Enum.at(m, 1), acc.params), {:ok, b} <- const(Enum.at(m, 2), acc.params) do
              {:cont, {:ok, %{acc | search: {min(a, b), max(a, b)}}}}
            else
              {:error, w} -> {:halt, {:error, "line #{line}: #{w}"}}
            end

          true ->
            case String.split(s, ~r/(?<![<>=!])=(?!=)/, parts: 2) do
              [l, r] ->
                with {:ok, lt} <- Expr.parse(l), {:ok, rt} <- Expr.parse(r) do
                  sub = Map.new(acc.params, fn {k, v} -> {k, {:n, v}} end)
                  {lt, rt} = {Expr.subst(lt, sub), Expr.subst(rt, sub)}
                  vs = Expr.vars(lt) ++ Expr.vars(rt)
                  cond do
                    Enum.any?(vs, &(&1 in unknowns)) ->
                      extra = Enum.reject(Enum.uniq(vs), &(&1 in unknowns))
                      if extra == [], do: {:cont, {:ok, %{acc | equations: acc.equations ++ [{lt, rt}]}}}, else: {:halt, {:error, "line #{line}: unknown name(s) #{Enum.join(extra, ", ")}"}}

                    match?({:v, _}, lt) ->
                      case const(r, acc.params) do
                        {:ok, v} -> {:cont, {:ok, %{acc | params: Map.put(acc.params, elem(lt, 1), v)}}}
                        {:error, w} -> {:halt, {:error, "line #{line}: #{w}"}}
                      end

                    true -> {:halt, {:error, "line #{line}: no unknown in #{inspect(s)}"}}
                  end
                else
                  {:error, w} -> {:halt, {:error, "line #{line}: #{w}"}}
                end

              _ -> {:halt, {:error, "line #{line}: expected an equation a = b"}}
            end
        end
      end)
      |> case do
        {:ok, %{equations: []}} -> {:error, "no equations"}
        {:ok, p} -> if length(p.equations) < length(p.unknowns), do: {:error, "#{length(p.unknowns)} unknowns but #{length(p.equations)} equation(s)"}, else: {:ok, p}
        e -> e
      end
    end
  end

  defp parse_decl(d) do
    body = Regex.replace(~r/^(unknowns?|inc[oó]gnitas?|solve|resolva)\s*:?\s*/iu, d, "")
    items = body |> String.split(",") |> Enum.map(&String.trim/1) |> Enum.reject(&(&1 == ""))

    Enum.reduce_while(items, {:ok, [], []}, fn it, {:ok, ns, gs} ->
      case Regex.run(~r/^(#{@id})\s*(?:=\s*(.+))?$/u, it) do
        [_, n] -> {:cont, {:ok, ns ++ [n], gs ++ [1.0]}}
        [_, n, g] -> (case const(g, %{}) do {:ok, v} -> {:cont, {:ok, ns ++ [n], gs ++ [v]}}; e -> {:halt, e} end)
        _ -> {:halt, {:error, "bad unknown #{inspect(it)}"}}
      end
    end)
    |> case do
      {:ok, [], _} -> {:error, "no unknowns declared"}
      other -> other
    end
  end

  defp const(s, params) do
    with {:ok, t} <- Expr.parse(s) do
      case Expr.vars(t) -- Map.keys(params) do
        [] -> (try do {:ok, Expr.eval(t, params)} rescue _ -> {:error, "cannot evaluate #{s}"} end)
        m -> {:error, "unknown name(s) #{Enum.join(m, ", ")}"}
      end
    end
  end

  # ================================================================== fit

  @doc """
  A nonlinear least-squares fit:

      fit y = a*exp(-b*x) + c
      a = 3; b = 1; c = 0          # starting values (default 1)
      data
      x, y
      0, 3.02
      0.5, 2.31
      …

  `{:ok, %{params: %{name => %{value, stderr}}, r2, rmse, residuals, fitted, iterations, dof}}`.
  """
  def fit(text) do
    with {:ok, spec} <- parse_fit(text) do
      %{response: yname, model: model, params: pnames, guess: p0, columns: cols, rows: rows} = spec
      inputs = Expr.vars(model) -- pnames
      bad = Enum.reject(inputs, &(&1 in cols))

      cond do
        bad != [] -> {:error, "the model uses #{Enum.join(bad, ", ")}, which is neither a parameter nor a data column (#{Enum.join(cols, ", ")})"}
        yname not in cols -> {:error, "the response #{yname} is not a data column (#{Enum.join(cols, ", ")})"}
        length(rows) <= length(pnames) -> {:error, "#{length(rows)} data rows for #{length(pnames)} parameters: need more data than parameters"}
        true -> lm(spec, inputs, model, pnames, p0, yname, cols, rows)
      end
    end
  end

  defp lm(spec, inputs, model, pnames, p0, yname, cols, rows) do
    names = pnames ++ inputs
    f = Expr.compile(model, names)
    grad = Expr.compile(Enum.map(pnames, &Expr.diff(model, &1)), names)
    ci = Map.new(Enum.with_index(cols))
    xs = Enum.map(rows, fn r -> Enum.map(inputs, &Enum.at(r, ci[&1])) end)
    ys = Enum.map(rows, &Enum.at(&1, ci[yname]))
    resid = fn p -> Enum.zip_with(xs, ys, fn x, y -> y - f.(List.to_tuple(p ++ x)) end) end
    jac = fn p -> Enum.map(xs, fn x -> grad.(List.to_tuple(p ++ x)) end) end
    ssr = fn r -> Enum.reduce(r, 0.0, &(&1 * &1 + &2)) end

    {p, its, conv} = lm_loop(p0, resid, jac, ssr, 1.0e-3, 0)
    r = resid.(p)
    s = ssr.(r)
    n = length(ys)
    k = length(pnames)
    dof = n - k
    j = jac.(p)
    jtj = Dense.matmul(Dense.transpose(j), j)
    cov = case Dense.inverse(jtj) do {:ok, inv} -> Enum.map(inv, fn row -> Enum.map(row, &(&1 * s / max(dof, 1))) end); _ -> nil end
    mean = Enum.sum(ys) / n
    sst = Enum.reduce(ys, 0.0, &((&1 - mean) ** 2 + &2))
    stderr = if cov, do: for(i <- 0..(k - 1), do: :math.sqrt(max(Enum.at(Enum.at(cov, i), i), 0.0))), else: List.duplicate(nil, k)

    {:ok, %{model: "#{yname} = #{Expr.to_text(model)}", params: Map.new(Enum.zip([pnames, p, stderr]), fn {nm, v, e} -> {nm, %{value: v, stderr: e}} end),
            order: pnames, r2: if(sst > 0, do: 1 - s / sst, else: nil), rmse: :math.sqrt(s / n), ssr: s, dof: dof, iterations: its, converged: conv,
            aic: n * :math.log(max(s / n, 1.0e-300)) + 2 * k, residuals: r, inputs: inputs, data: %{x: xs, y: ys}, fitted: Enum.map(xs, fn x -> f.(List.to_tuple(p ++ x)) end),
            covariance: cov, response: yname, rows: length(rows), columns: spec.columns}}
  rescue
    ArithmeticError -> {:error, "the model could not be evaluated at some data point (division by zero or a domain error)"}
  end

  defp lm_loop(p, resid, jac, ssr, lam, its) do
    r = resid.(p)
    s = ssr.(r)
    j = jac.(p)
    # J is ∂f/∂p; residual is y − f: the step solves (JᵀJ + λ diag JᵀJ) δ = Jᵀ r
    jt = Dense.transpose(j)
    jtj = Dense.matmul(jt, j)
    g = Dense.matvec(jt, r)
    damped = for {row, i} <- Enum.with_index(jtj), do: for({x, c} <- Enum.with_index(row), do: if(i == c, do: x + lam * max(x, 1.0e-12), else: x))

    cond do
      its >= 300 -> {p, its, false}
      true ->
        case Dense.solve(damped, g) do
          {:ok, d} ->
            pn = Enum.zip_with(p, d, &(&1 + &2))
            sn = try do ssr.(resid.(pn)) rescue ArithmeticError -> :infinity end
            cond do
              sn != :infinity and sn < s ->
                if (s - sn) <= 1.0e-14 * max(s, 1.0e-300) or Dense.norm_inf(d) <= 1.0e-12 * (1 + Dense.norm_inf(p)),
                  do: {pn, its + 1, true}, else: lm_loop(pn, resid, jac, ssr, max(lam / 3, 1.0e-12), its + 1)
              lam > 1.0e12 -> {p, its, Dense.norm_inf(g) < 1.0e-8 * max(s, 1.0)}
              true -> lm_loop(p, resid, jac, ssr, lam * 4, its + 1)
            end

          _ -> lm_loop(p, resid, jac, ssr, lam * 4, its + 1)
        end
    end
  end

  defp parse_fit(text) do
    lines = text |> String.split("\n") |> Enum.map(&(&1 |> String.split("#", parts: 2) |> hd() |> String.trim())) |> Enum.reject(&(&1 == ""))
    {head, data} = Enum.split_while(lines, &(not (&1 =~ ~r/^(data|dados)\s*:?$/iu)))
    data = Enum.drop(data, 1)

    {fits, others} = Enum.split_with(head, &(&1 =~ ~r/^(fit|ajuste|ajustar)\b/iu))

    with [fit_line | _] <- (if fits == [], do: {:error, "write the model: fit y = …"}, else: fits),
         [_, y, rhs] <- Regex.run(~r/^(?:fit|ajuste|ajustar)\s+(#{@id})\s*=\s*(.+)$/iu, fit_line) || {:error, "write the model as: fit y = expression"},
         {:ok, model} <- Expr.parse(rhs),
         {:ok, guesses} <- parse_guesses(others),
         {:ok, cols, rows} <- parse_table(data) do
      pnames = (Expr.vars(model) -- cols) |> Enum.sort()
      {:ok, %{response: y, model: model, params: pnames, guess: Enum.map(pnames, &Map.get(guesses, &1, 1.0)), columns: cols, rows: rows}}
    end
  end

  defp parse_guesses(lines) do
    lines |> Enum.flat_map(&String.split(&1, ";")) |> Enum.map(&String.trim/1) |> Enum.reject(&(&1 == ""))
    |> Enum.reduce_while({:ok, %{}}, fn s, {:ok, acc} ->
      case Regex.run(~r/^(#{@id})\s*=\s*(.+)$/u, s) do
        [_, n, v] -> (case const(v, %{}) do {:ok, x} -> {:cont, {:ok, Map.put(acc, n, x)}}; e -> {:halt, e} end)
        _ -> {:halt, {:error, "not understood: #{inspect(s)}"}}
      end
    end)
  end

  @doc false
  # a table: a header of names, then rows of numbers separated by commas, semicolons, tabs or spaces
  def parse_table([]), do: {:error, "no data: write a line 'data', a header (x, y) and rows"}

  def parse_table([header | rows]) do
    split = fn l -> l |> String.split(~r/[,;\t ]+/, trim: true) end
    cols = split.(header)
    if Enum.all?(cols, &(&1 =~ ~r/^#{@id}$/u)) do
      parsed = Enum.with_index(rows, 2) |> Enum.map(fn {r, i} ->
        vals = split.(r) |> Enum.map(&Float.parse(String.replace(&1, ~r/^\./, "0.")))
        if length(vals) == length(cols) and Enum.all?(vals, &match?({_, ""}, &1)), do: Enum.map(vals, &elem(&1, 0)), else: {:bad, i}
      end)
      case Enum.find(parsed, &match?({:bad, _}, &1)) do
        nil -> {:ok, cols, parsed}
        {:bad, i} -> {:error, "data row #{i} does not have #{length(cols)} numbers"}
      end
    else
      {:error, "the first data line must name the columns (e.g. x, y)"}
    end
  end

  # ============================================================= minimize

  @doc """
  Minimise (or maximise) an expression:

      minimize (1 - x)^2 + 100*(y - x^2)^2
      from x = -1.2, y = 1
      subject to x + y <= 1.5      # optional; also >=, = (equality)

  BFGS on the augmented Lagrangian; `{:ok, %{x, value, kkt, iterations,
  history, converged, verdict}}`. A constraint that bounds one variable by a
  number (`r >= 0.01`, `h <= 10`, `0 <= x`) is a **box bound**, kept exactly
  by projection at every step (projected BFGS) rather than penalised — so
  the search never leaves the region where the model makes sense (a
  negative radius, say). The verdict says whether the KKT conditions hold;
  an objective unbounded below in the region searched is reported as such,
  never as an answer.
  """
  def minimize(text) do
    stmts = Solve.statements(text)

    with [{obj_line, _} | _] <- Enum.filter(stmts, fn {s, _} -> s =~ ~r/^(minimi[sz]e|minimi[sz]ar|maximi[sz]e|maximi[sz]ar)\b/iu end),
         sign = if(obj_line =~ ~r/^maximi/iu, do: -1.0, else: 1.0),
         {:ok, obj} <- Expr.parse(Regex.replace(~r/^(minimi[sz]e|minimi[sz]ar|maximi[sz]e|maximi[sz]ar)\s*:?\s*/iu, obj_line, "")),
         {:ok, start} <- start_point(stmts, Expr.vars(obj)),
         {:ok, cons} <- constraints(stmts) do
      vars = (Expr.vars(obj) ++ Enum.flat_map(cons, fn {_, t} -> Expr.vars(t) end)) |> Enum.uniq() |> Enum.sort()
      {box, cons} = split_bounds(cons, vars)
      x0 = Enum.map(vars, &Map.get(start, &1, 0.0))
      solve_al(Expr.simplify({:*, {:n, sign}, obj}), sign, cons, vars, x0, box)
    else
      [] -> {:error, "write: minimize <expression>"}
      e -> e
    end
  end

  defp start_point(stmts, _vars) do
    case Enum.find(stmts, fn {s, _} -> s =~ ~r/^(from|de|a partir de|start)\b/iu end) do
      nil -> {:ok, %{}}
      {s, _} -> (with {:ok, ns, gs} <- parse_decl(Regex.replace(~r/^(from|de|a partir de|start)\s*/iu, s, "unknowns ")), do: {:ok, Map.new(Enum.zip(ns, gs))})
    end
  end

  defp constraints(stmts) do
    stmts
    |> Enum.filter(fn {s, _} -> s =~ ~r/^(subject to|s\.t\.|sujeito a|st)\b/iu end)
    |> Enum.flat_map(fn {s, _} -> Regex.replace(~r/^(subject to|s\.t\.|sujeito a|st)\s*:?\s*/iu, s, "") |> String.split(",") end)
    |> Enum.map(&String.trim/1) |> Enum.reject(&(&1 == ""))
    |> Enum.reduce_while({:ok, []}, fn c, {:ok, acc} ->
      parsed =
        cond do
          String.contains?(c, "<=") -> (([l, r] = String.split(c, "<=", parts: 2)); {:ineq, l, r})
          String.contains?(c, "≤") -> (([l, r] = String.split(c, "≤", parts: 2)); {:ineq, l, r})
          String.contains?(c, ">=") -> (([l, r] = String.split(c, ">=", parts: 2)); {:ineq, r, l})
          String.contains?(c, "≥") -> (([l, r] = String.split(c, "≥", parts: 2)); {:ineq, r, l})
          String.contains?(c, "=") -> (([l, r] = String.split(c, "=", parts: 2)); {:eq, l, r})
          true -> nil
        end

      with {kind, l, r} <- parsed || {:error, "constraint #{inspect(c)}: use <=, >= or ="},
           {:ok, lt} <- Expr.parse(l), {:ok, rt} <- Expr.parse(r) do
        {:cont, {:ok, acc ++ [{kind, Expr.simplify({:-, lt, rt})}]}}
      else
        e -> {:halt, e}
      end
    end)
  end

  # constraints `v − c ≤ 0` or `c − v ≤ 0` with v a variable and c a number are box bounds
  defp split_bounds(cons, vars) do
    lo0 = Map.new(vars, &{&1, :neg}); hi0 = Map.new(vars, &{&1, :pos})
    {lo, hi, rest} =
      Enum.reduce(cons, {lo0, hi0, []}, fn {kind, t} = c, {lo, hi, rest} ->
        with :ineq <- kind, {:ok, v, a, b} <- affine1(t, vars) do
          # a·v + b ≤ 0
          cond do
            a > 0 -> {lo, Map.update!(hi, v, &tighter(&1, -b / a, :hi)), rest}
            a < 0 -> {Map.update!(lo, v, &tighter(&1, -b / a, :lo)), hi, rest}
            true -> {lo, hi, rest ++ [c]}
          end
        else
          _ -> {lo, hi, rest ++ [c]}
        end
      end)
    bounds = Enum.map(vars, fn v -> {num_or(lo[v], :neg), num_or(hi[v], :pos)} end)
    {bounds, rest}
  end

  defp num_or(x, _d) when is_number(x), do: x * 1.0
  defp num_or(_, d), do: d
  defp tighter(cur, v, :hi), do: if(is_number(cur), do: min(cur, v), else: v)
  defp tighter(cur, v, :lo), do: if(is_number(cur), do: max(cur, v), else: v)

  # t = a·v + b with one variable and numeric a, b (checked by evaluation at two points)
  defp affine1(t, vars) do
    case Expr.vars(t) do
      [v] ->
        if v in vars do
          d = Expr.simplify(Expr.diff(t, v))
          if Expr.vars(d) == [] do
            a = Expr.eval(d, %{})
            b = Expr.eval(t, %{v => 0.0})
            if is_number(a) and is_number(b) and abs(Expr.eval(t, %{v => 1.7}) - (a * 1.7 + b)) < 1.0e-9 * (1 + abs(b)), do: {:ok, v, a, b}, else: :no
          else
            :no
          end
        else
          :no
        end
      _ -> :no
    end
  rescue
    _ -> :no
  end

  defp box_free?(box), do: Enum.all?(box, &(&1 == {:neg, :pos}))

  defp project(x, box), do: Enum.zip_with(x, box, fn xi, {lo, hi} -> xi |> then(&if(is_number(lo), do: max(&1, lo), else: &1)) |> then(&if(is_number(hi), do: min(&1, hi), else: &1)) end)

  # the gradient with the components that push against an active bound removed
  defp proj_grad(g, x, box) do
    Enum.zip_with([g, x, box], fn [gi, xi, {lo, hi}] ->
      cond do
        is_number(lo) and xi <= lo and gi > 0 -> 0.0
        is_number(hi) and xi >= hi and gi < 0 -> 0.0
        true -> gi
      end
    end)
  end

  defp solve_al(obj, sign, cons, vars, x0, box) do
    x0 = project(x0, box)
    f = Expr.compile(obj, vars)
    gf = Expr.compile(Enum.map(vars, &Expr.diff(obj, &1)), vars)
    cs = Enum.map(cons, fn {k, t} -> {k, Expr.compile(t, vars), Expr.compile(Enum.map(vars, &Expr.diff(t, &1)), vars)} end)
    lam0 = List.duplicate(0.0, length(cs))

    stat = fn xx, lam ->
      tx = List.to_tuple(xx)
      Enum.zip(cs, lam) |> Enum.reduce(gf.(tx), fn {{_, _, gc}, l}, acc -> Enum.zip_with(acc, gc.(tx), &(&1 + l * &2)) end) |> proj_grad(xx, box) |> Dense.norm_inf()
    end

    {x, lam, its, hist, _} =
      Enum.reduce_while(1..40, {x0, lam0, 0, [], {10.0, :infinity}}, fn outer, {x, lam, its, hist, {mu, prev}} ->
        phi = fn xx -> al_value(f, cs, lam, mu, xx) end
        dphi = fn xx -> al_grad(gf, cs, lam, mu, xx) end
        {xn, k} = bfgs(phi, dphi, x, 500, box)
        tx = List.to_tuple(xn)
        lam2 = Enum.zip_with(cs, lam, fn {kind, c, _}, l -> (v = c.(tx); if kind == :eq, do: l + mu * v, else: max(0.0, l + mu * v)) end)
        feas = feasibility(cs, tx)
        hist = hist ++ [%{outer: outer, value: sign * f.(tx), infeasibility: feas, penalty: mu}]
        # the penalty grows only when feasibility stalls (Bertsekas' rule), so it never swamps the objective
        mu2 = if prev != :infinity and feas > 0.25 * prev, do: min(mu * 10, 1.0e8), else: mu
        done = cs == [] or (feas < 1.0e-9 and stat.(xn, lam2) < 1.0e-7)
        diverged = Enum.any?(xn, &(abs(&1) > 1.0e12)) or abs(f.(tx)) > 1.0e100
        if done or diverged, do: {:halt, {xn, lam2, its + k, hist, nil}}, else: {:cont, {xn, lam2, its + k, hist, {mu2, feas}}}
      end)

    tx = List.to_tuple(x)
    g = gf.(tx)
    lag = Enum.zip(cs, lam) |> Enum.reduce(g, fn {{_, _, gc}, l}, acc -> Enum.zip_with(acc, gc.(tx), &(&1 + l * &2)) end) |> proj_grad(x, box)
    st = Dense.norm_inf(lag)
    feas = feasibility(cs, tx)
    scale = max(1.0, Dense.norm_inf(g))
    unbounded = Enum.any?(x, &(abs(&1) > 1.0e12)) or abs(f.(tx)) > 1.0e100
    converged = not unbounded and feas < 1.0e-6 and st < 1.0e-5 * scale
    verdict =
      cond do
        unbounded -> "diverged: the objective decreases without bound in the region searched — bound the variables (e.g. r >= 0.01)"
        converged -> "KKT conditions hold: a local #{if sign > 0, do: "minimum", else: "maximum"}"
        feas >= 1.0e-6 -> "not converged: the constraints are violated by #{Float.round(feas, 8)}"
        true -> "not converged: stationarity #{:erlang.float_to_binary(st, decimals: 10)}"
      end
    bounds_text = Enum.zip(vars, box) |> Enum.flat_map(fn {v, {lo, hi}} -> (if is_number(lo), do: ["#{v} ≥ #{Expr.num(lo)}"], else: []) ++ (if is_number(hi), do: ["#{v} ≤ #{Expr.num(hi)}"], else: []) end)

    {:ok, %{x: Map.new(Enum.zip(vars, x)), value: sign * f.(tx), vars: vars, iterations: its, history: hist, multipliers: lam, bounds: bounds_text,
            converged: converged, verdict: verdict,
            kkt: %{stationarity: st, infeasibility: feas,
                   complementarity: Enum.zip(cs, lam) |> Enum.map(fn {{k, c, _}, l} -> if k == :eq, do: 0.0, else: abs(l * c.(tx)) end) |> Enum.max(fn -> 0.0 end)},
            constraints: Enum.map(cons, fn {k, t} -> "#{Expr.to_text(t)} #{if k == :eq, do: "=", else: "≤"} 0" end), objective: Expr.to_text(obj), sense: if(sign > 0, do: "min", else: "max")}}
  rescue
    ArithmeticError -> {:error, "the objective or a constraint could not be evaluated along the way (domain error)"}
  end

  defp feasibility(cs, tx), do: cs |> Enum.map(fn {k, c, _} -> v = c.(tx); if k == :eq, do: abs(v), else: max(v, 0.0) end) |> Enum.max(fn -> 0.0 end)

  defp al_value(f, cs, lam, mu, x) do
    tx = List.to_tuple(x)
    Enum.zip(cs, lam) |> Enum.reduce(f.(tx), fn {{k, c, _}, l}, acc ->
      v = c.(tx)
      if k == :eq, do: acc + l * v + mu / 2 * v * v, else: acc + (max(0.0, l + mu * v) ** 2 - l * l) / (2 * mu)
    end)
  end

  defp al_grad(gf, cs, lam, mu, x) do
    tx = List.to_tuple(x)
    Enum.zip(cs, lam) |> Enum.reduce(gf.(tx), fn {{k, c, gc}, l}, acc ->
      v = c.(tx)
      w = if k == :eq, do: l + mu * v, else: max(0.0, l + mu * v)
      Enum.zip_with(acc, gc.(tx), &(&1 + w * &2))
    end)
  end

  @doc false
  # BFGS with an Armijo backtracking line search; returns {x, iterations}
  # with `box` (a list of {lo, hi}, :neg/:pos for none): projected BFGS — the
  # directions of active bounds are frozen, the step is projected back into the box
  def bfgs(f, g, x, maxit, box \\ nil) do
    n = length(x)
    box = box || List.duplicate({:neg, :pos}, n)
    x = project(x, box)
    bfgs_loop(f, g, x, f.(x), g.(x), Dense.identity(n), 0, maxit, box)
  end

  defp bfgs_loop(f, g, x, fx, gx, h, k, maxit, box) do
    pg = proj_grad(gx, x, box)
    if k >= maxit or Dense.norm_inf(pg) < 1.0e-10 or fx < -1.0e100 do
      {x, k}
    else
      active = Enum.zip_with(pg, gx, fn a, b -> a == 0.0 and b != 0.0 end)
      d = Dense.matvec(h, pg) |> Enum.map(&(-&1)) |> Enum.zip_with(active, fn di, a -> if a, do: 0.0, else: di end)
      slope = Dense.dot(pg, d)
      {d, slope, h} = if slope >= 0, do: (dd = Enum.map(pg, &(-&1)); {dd, Dense.dot(pg, dd), Dense.identity(length(x))}), else: {d, slope, h}

      case line(f, x, fx, d, slope, 1.0, 0, box) do
        nil -> {x, k}
        {xn, fxn} ->
          gn = g.(xn)
          s = Enum.zip_with(xn, x, &(&1 - &2))
          y = Enum.zip_with(gn, gx, &(&1 - &2))
          sy = Dense.dot(s, y)
          h = if sy > 1.0e-14, do: bfgs_update(h, s, y, sy), else: h
          if abs(fxn - fx) <= 1.0e-15 * max(abs(fx), 1.0) and Dense.norm_inf(s) < 1.0e-14, do: {xn, k + 1}, else: bfgs_loop(f, g, xn, fxn, gn, h, k + 1, maxit, box)
      end
    end
  end

  defp line(f, x, fx, d, slope, a, k, box) do
    if k > 40 do
      nil
    else
      xn = Enum.zip_with(x, d, &(&1 + a * &2)) |> project(box)
      # the Armijo test along the projected path: the step actually taken replaces a·d
      dec = if box_free?(box), do: a * slope, else: Dense.dot(Enum.zip_with(xn, x, &(&1 - &2)), Enum.map(d, &(&1 * slope / max(Dense.dot(d, d), 1.0e-300))))
      case (try do f.(xn) rescue ArithmeticError -> nil end) do
        nil -> line(f, x, fx, d, slope, a / 2, k + 1, box)
        fxn -> if fxn <= fx + 1.0e-4 * dec and xn != x, do: {xn, fxn}, else: line(f, x, fx, d, slope, a / 2, k + 1, box)
      end
    end
  end

  defp bfgs_update(h, s, y, sy) do
    hy = Dense.matvec(h, y)
    yhy = Dense.dot(y, hy)
    rho = 1 / sy
    for {hi, si, hyi} <- Enum.zip([h, s, hy]) do
      for {hij, sj, hyj} <- Enum.zip([hi, s, hy]) do
        hij - rho * (hyi * sj + si * hyj) + (rho * rho * yhy + rho) * si * sj
      end
    end
  end
end
