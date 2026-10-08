defmodule Vapor.Finance.Backtest do
  @moduledoc """
  Backtests with **noise gates** (docs/FINANCE.md §7).

  A backtest is a machine for producing Sharpe ratios; try enough
  variants and one will look good on pure noise. The industry's two
  silent failures are **look-ahead** (the signal used information it
  would not have had) and **selection** (the best of many trials reported
  as if it were the only one). This module answers both with numbers that
  could have come out otherwise:

  1. **Prefix invariance (look-ahead certificate).** The signal is
     re-computed on truncated histories `data[0..c]` at several cut points
     and must equal, bit for bit, the full-history signal up to `c`. It is
     a black-box test of the whole pipeline: an operator that peeks
     (`lead`, a full-sample `center`/`normalize`, a typo) fails it, with
     the first offending day. No trust in the operators is needed.
  2. **Deflated Sharpe ratio** (Bailey & López de Prado 2014): the
     probability that the true Sharpe exceeds the best one expected from
     N trials of pure noise, corrected for skewness and kurtosis.
  3. **Probability of backtest overfitting** (Bailey, Borwein, López de
     Prado & Zhu 2017) by combinatorially symmetric cross-validation: how
     often the in-sample winner ranks below the median out of sample.
  4. **White's Reality Check** with Politis–Romano's stationary
     bootstrap: is the best mean return explained by the luck of the
     maximum?

  Text:

      data = ar1 n=2520 phi=0.08 sigma=0.01 seed=3      # or gbm / t / csv (lines date,close)
      sweep fast = 5..30 step 5
      sweep slow = 40..120 step 20
      signal = sign(ema(close, fast) - ema(close, slow))
      cost = 5bp
      trials = 1        # trials made before this file (added to the sweep's)

  The position held over (t, t+1] is the signal at t, clipped to [−1, 1];
  costs are charged on |Δposition|.
  """
  alias Vapor.Finance.Num

  @max_trials 400
  @max_n 20_000

  # ================================================================= parse

  def run(text, opts \\ []) do
    with {:ok, spec} <- parse(text), {:ok, data} <- data(spec), {:ok, ast} <- expr(spec.signal) do
      evaluate(spec, data, ast, opts)
    end
  end

  def parse(text) do
    lines = text |> String.split("\n") |> Enum.with_index(1) |> Enum.map(fn {l, i} -> {l |> String.split("#") |> hd() |> String.trim(), i} end) |> Enum.reject(&(elem(&1, 0) == ""))
    init = %{data: nil, csv: [], sweeps: [], signal: nil, cost: 0.0, trials: 0, annual: 252}
    {spec, _} =
      Enum.reduce(lines, {init, false}, fn {l, _i}, {s, in_csv} ->
        cond do
          in_csv and l =~ ~r/^[\d\-\/.]+\s*[,;\s]\s*-?[\d.]+/ -> {%{s | csv: s.csv ++ [l]}, true}
          m = Regex.run(~r/^data\s*=\s*(\w+)(.*)$/i, l) ->
            kind = String.downcase(Enum.at(m, 1))
            kv = Regex.scan(~r/(\w+)\s*=\s*([-\d.eE]+)/, Enum.at(m, 2)) |> Map.new(fn [_, k, v] -> {k, num(v)} end)
            {%{s | data: {kind, kv}}, kind == "csv"}
          m = Regex.run(~r/^(?:sweep|varrer)\s+(\w+)\s*=\s*(-?[\d.]+)\s*\.\.\s*(-?[\d.]+)(?:\s+step\s+([\d.]+))?$/i, l) ->
            [_, name, a, b | st] = m
            {%{s | sweeps: s.sweeps ++ [{name, num(a), num(b), num(List.first(st) || "1")}]}, false}
          m = Regex.run(~r/^(?:signal|sinal)\s*=\s*(.+)$/i, l) -> {%{s | signal: Enum.at(m, 1)}, false}
          m = Regex.run(~r/^(?:cost|custo)\s*=\s*([\d.]+)\s*(bp|bps|%)?$/i, l) ->
            v = num(Enum.at(m, 1)); u = Enum.at(m, 2, "")
            {%{s | cost: case String.downcase(u || "") do "%" -> v / 100; "" -> v; _ -> v / 10_000 end}, false}
          m = Regex.run(~r/^(?:trials|tentativas)\s*=\s*(\d+)$/i, l) -> {%{s | trials: String.to_integer(Enum.at(m, 1))}, false}
          m = Regex.run(~r/^(?:annual|anual|periods)\s*=\s*(\d+)$/i, l) -> {%{s | annual: String.to_integer(Enum.at(m, 1))}, false}
          in_csv -> {s, true}
          true -> {Map.update(s, :unknown, [l], &(&1 ++ [l])), false}
        end
      end)
    cond do
      Map.has_key?(spec, :unknown) -> {:error, "not understood: #{inspect(hd(spec.unknown))} — data, sweep, signal, cost, trials"}
      spec.signal == nil -> {:error, "signal = … is missing"}
      spec.data == nil -> {:error, "data = gbm | ar1 | t | csv is missing"}
      true -> {:ok, spec}
    end
  end

  defp num(s), do: (case Float.parse(s) do {v, _} -> v; :error -> 0.0 end)

  # ================================================================= data

  @doc false
  def data(%{data: {"csv", _}, csv: rows}) do
    closes = for r <- rows, [_, v | _] = String.split(r, ~r/\s*[,;\s]\s*/), {x, _} = Float.parse(v), do: x
    cond do
      length(closes) < 60 -> {:error, "at least 60 prices are needed"}
      length(closes) > @max_n -> {:error, "at most #{@max_n} prices"}
      Enum.any?(closes, &(&1 <= 0)) -> {:error, "prices must be positive"}
      true -> {:ok, %{close: closes, source: "csv"}}
    end
  end

  def data(%{data: {kind, kv}}) when kind in ["gbm", "ar1", "t"] do
    n = trunc(Map.get(kv, "n", 2520)) |> max(100) |> min(@max_n)
    seed = trunc(Map.get(kv, "seed", 1))
    sigma = Map.get(kv, "sigma", 0.01); mu = Map.get(kv, "mu", 0.0) / 252
    z = Num.normals(seed, n)
    rets =
      case kind do
        "gbm" -> Enum.map(z, &(mu + sigma * &1))
        "ar1" ->
          phi = Map.get(kv, "phi", 0.05)
          {rs, _} = Enum.map_reduce(z, 0.0, fn e, prev -> r = mu + phi * prev + sigma * :math.sqrt(1 - phi * phi) * e; {r, r} end)
          rs
        "t" ->
          # Student t with ν = 4 scaled to σ: Z / √(χ²₄/4) · σ/√2
          nu = Map.get(kv, "nu", 4.0)
          k = trunc(nu)
          extra = Num.normals(seed + 1_000_003, n * k) |> Enum.chunk_every(k)
          Enum.zip_with(z, extra, fn e, zs -> chi = Enum.reduce(zs, 0.0, &(&1 * &1 + &2)); mu + sigma * :math.sqrt((nu - 2) / nu) * e / :math.sqrt(chi / nu) end)
      end
    {closes, _} = Enum.map_reduce(rets, 100.0, fn r, p -> q = p * :math.exp(r); {q, q} end)
    {:ok, %{close: [100.0 | closes], source: kind, params: kv}}
  end

  def data(%{data: {k, _}}), do: {:error, "data kind #{inspect(k)}: gbm, ar1, t or csv"}

  # ====================================================== the signal language

  @doc """
  Parse a signal expression: numbers, names, calls f(a, …), unary −,
  `* /`, `+ −`, comparisons `< > <= >=` (1 or 0), `and`/`or`.
  """
  def expr(text) do
    case tokens(text) do
      {:ok, toks} ->
        case parse_or(toks) do
          {:ok, ast, []} -> {:ok, ast}
          {:ok, _, [t | _]} -> {:error, "unexpected #{inspect(t)} in the signal"}
          e -> e
        end
      e -> e
    end
  end

  defp tokens(s) do
    re = ~r/\s*(?:(\d+\.?\d*(?:[eE][-+]?\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(<=|>=|[-+*\/(),<>]))/
    scan = fn scan, rest, acc ->
      rest = String.trim_leading(rest)
      if rest == "" do
        {:ok, Enum.reverse(acc)}
      else
        case Regex.run(re, rest, return: :index) do
          [{0, len} | groups] ->
            tok = case groups do
              [{a, l} | _] when a >= 0 and l > 0 -> {:num, num(binary_part(rest, a, l))}
              [_, {a, l} | _] when a >= 0 and l > 0 -> (w = binary_part(rest, a, l); if(w in ["and", "or", "e", "ou"], do: {:op, w}, else: {:id, w}))
              [_, _, {a, l}] -> {:op, binary_part(rest, a, l)}
            end
            scan.(scan, binary_part(rest, len, byte_size(rest) - len), [tok | acc])
          _ -> {:error, "cannot read the signal at #{inspect(String.slice(rest, 0, 12))}"}
        end
      end
    end
    scan.(scan, s, [])
  end

  defp parse_or(t), do: binary(t, &parse_cmp/1, %{"or" => :or, "ou" => :or, "and" => :and, "e" => :and})
  defp parse_cmp(t), do: binary(t, &parse_add/1, %{"<" => :<, ">" => :>, "<=" => :<=, ">=" => :>=})
  defp parse_add(t), do: binary(t, &parse_mul/1, %{"+" => :+, "-" => :-})
  defp parse_mul(t), do: binary(t, &parse_un/1, %{"*" => :*, "/" => :/})

  defp binary(t, sub, ops) do
    with {:ok, l, rest} <- sub.(t), do: binary_rest(l, rest, sub, ops)
  end

  defp binary_rest(l, [{:op, o} | rest] = all, sub, ops) do
    case ops do
      %{^o => op} -> with({:ok, r, rest2} <- sub.(rest), do: binary_rest({op, l, r}, rest2, sub, ops))
      _ -> {:ok, l, all}
    end
  end
  defp binary_rest(l, rest, _, _), do: {:ok, l, rest}

  defp parse_un([{:op, "-"} | t]), do: with({:ok, a, r} <- parse_un(t), do: {:ok, {:neg, a}, r})
  defp parse_un(t), do: parse_atom(t)

  defp parse_atom([{:num, x} | r]), do: {:ok, {:n, x}, r}
  defp parse_atom([{:id, f}, {:op, "("} | r]) do
    with {:ok, args, r2} <- parse_args(r, []), do: {:ok, {:f, String.downcase(f), args}, r2}
  end
  defp parse_atom([{:id, v} | r]), do: {:ok, {:v, v}, r}
  defp parse_atom([{:op, "("} | r]) do
    case parse_or(r) do
      {:ok, a, [{:op, ")"} | r2]} -> {:ok, a, r2}
      {:ok, _, _} -> {:error, "a parenthesis is not closed"}
      e -> e
    end
  end
  defp parse_atom([t | _]), do: {:error, "unexpected #{inspect(t)}"}
  defp parse_atom([]), do: {:error, "the signal ends too early"}

  defp parse_args([{:op, ")"} | r], acc), do: {:ok, Enum.reverse(acc), r}
  defp parse_args(t, acc) do
    case parse_or(t) do
      {:ok, a, [{:op, ","} | r]} -> parse_args(r, [a | acc])
      {:ok, a, [{:op, ")"} | r]} -> {:ok, Enum.reverse([a | acc]), r}
      {:ok, _, _} -> {:error, "expected , or ) in a call"}
      e -> e
    end
  end

  # ================================================ evaluation over series

  @causal ~w(lag diff pct ret logret sma ema std zscore rmax rmin rsi sign abs log exp clip min max if)
  @peeking ~w(lead center normalize)
  def functions, do: %{causal: @causal, non_causal_accepted: @peeking}

  @doc false
  # series are tuples of floats or nil (warm-up); scalars are numbers
  def eval_series(ast, env) do
    case ast do
      {:n, x} -> x
      {:v, v} -> (case Map.fetch(env, v) do {:ok, x} -> x; :error -> throw({:unknown, v}) end)
      {:neg, a} -> map1(eval_series(a, env), &(-&1))
      {op, a, b} when op in [:+, :-, :*, :/, :<, :>, :<=, :>=, :and, :or] -> map2(eval_series(a, env), eval_series(b, env), op_fun(op))
      {:f, f, args} -> call(f, Enum.map(args, &eval_series(&1, env)))
    end
  end

  defp op_fun(:+), do: &(&1 + &2)
  defp op_fun(:-), do: &(&1 - &2)
  defp op_fun(:*), do: &(&1 * &2)
  defp op_fun(:/), do: fn a, b -> if b == 0, do: nil, else: a / b end
  defp op_fun(:<), do: &bool(&1 < &2)
  defp op_fun(:>), do: &bool(&1 > &2)
  defp op_fun(:<=), do: &bool(&1 <= &2)
  defp op_fun(:>=), do: &bool(&1 >= &2)
  defp op_fun(:and), do: &bool(&1 != 0 and &2 != 0)
  defp op_fun(:or), do: &bool(&1 != 0 or &2 != 0)
  defp bool(true), do: 1.0
  defp bool(false), do: 0.0

  defp map1(x, f) when is_tuple(x), do: x |> Tuple.to_list() |> Enum.map(&(&1 && f.(&1))) |> List.to_tuple()
  defp map1(x, f), do: f.(x)

  defp map2(a, b, f) when is_tuple(a) and is_tuple(b), do: Enum.zip_with(Tuple.to_list(a), Tuple.to_list(b), &(&1 && &2 && f.(&1, &2))) |> List.to_tuple()
  defp map2(a, b, f) when is_tuple(a), do: map1(a, &f.(&1, b))
  defp map2(a, b, f) when is_tuple(b), do: map1(b, &f.(a, &1))
  defp map2(a, b, f), do: f.(a, b)

  defp int!(k) when is_number(k) and k >= 1, do: trunc(k)
  defp int!(k), do: throw({:arg, "a window must be a number ≥ 1 (got #{inspect(k)})"})

  defp call("lag", [x, k]), do: shift(x, int!(k))
  defp call("lag", [x]), do: shift(x, 1)
  defp call("diff", [x | k]), do: map2(x, shift(x, int!(List.first(k, 1))), &(&1 - &2))
  defp call(f, [x | k]) when f in ["pct", "ret"], do: map2(x, shift(x, int!(List.first(k, 1))), fn a, b -> if b == 0, do: nil, else: a / b - 1 end)
  defp call("logret", [x | k]), do: map2(x, shift(x, int!(List.first(k, 1))), fn a, b -> if a > 0 and b > 0, do: :math.log(a / b), else: nil end)
  defp call("sma", [x, n]), do: rolling(x, int!(n), fn w -> Enum.sum(w) / length(w) end)
  defp call("std", [x, n]), do: rolling(x, int!(n), &Num.std/1)
  defp call("rmax", [x, n]), do: rolling(x, int!(n), &Enum.max/1)
  defp call("rmin", [x, n]), do: rolling(x, int!(n), &Enum.min/1)
  defp call("zscore", [x, n]), do: (m = call("sma", [x, n]); s = call("std", [x, n]); map2(map2(x, m, &(&1 - &2)), s, fn a, b -> if b == 0, do: nil, else: a / b end))
  defp call("ema", [x, n]), do: ema(x, int!(n))
  defp call("rsi", [x, n]) do
    d = call("diff", [x, 1])
    up = ema(map1(d, &max(&1, 0.0)), int!(n)); dn = ema(map1(d, &max(-&1, 0.0)), int!(n))
    map2(up, dn, fn u, v -> if u + v == 0, do: 50.0, else: 100 * u / (u + v) end)
  end
  defp call("sign", [x]), do: map1(x, fn v -> cond do v > 0 -> 1.0; v < 0 -> -1.0; true -> 0.0 end end)
  defp call("abs", [x]), do: map1(x, &abs/1)
  defp call("log", [x]), do: map1(x, fn v -> if v > 0, do: :math.log(v), else: nil end)
  defp call("exp", [x]), do: map1(x, &:math.exp/1)
  defp call("clip", [x, lo, hi]), do: map1(x, &(&1 |> max(lo) |> min(hi)))
  defp call("min", [a, b]), do: map2(a, b, &min/2)
  defp call("max", [a, b]), do: map2(a, b, &max/2)
  defp call("if", [c, a, b]), do: map2(map2(c, a, &{&1, &2}), b, fn {cc, aa}, bb -> if cc != 0, do: aa, else: bb end)
  # accepted because people write them — and caught by the prefix-invariance certificate
  defp call("lead", [x | k]), do: shift(x, -int!(List.first(k, 1)))
  defp call("center", [x]), do: (vs = valid(x); m = Num.mean(vs); map1(x, &(&1 - m)))
  defp call("normalize", [x]), do: (vs = valid(x); m = Num.mean(vs); s = Num.std(vs); map1(x, &((&1 - m) / max(s, 1.0e-300))))
  defp call(f, args), do: throw({:arg, "unknown function #{f}/#{length(args)} (causal: #{Enum.join(@causal, ", ")})"})

  defp valid(x) when is_tuple(x), do: x |> Tuple.to_list() |> Enum.reject(&is_nil/1)
  defp valid(x), do: [x]

  defp shift(x, k) when is_tuple(x) do
    n = tuple_size(x)
    List.to_tuple(for i <- 0..(n - 1), do: (j = i - k; if(j >= 0 and j < n, do: elem(x, j), else: nil)))
  end
  defp shift(x, _), do: x

  defp rolling(x, n, f) when is_tuple(x) do
    l = Tuple.to_list(x)
    {out, _} = Enum.map_reduce(l, :queue.new(), fn v, q ->
      q = :queue.in(v, q); q = if :queue.len(q) > n, do: elem(:queue.out(q), 1), else: q
      w = :queue.to_list(q)
      {if(length(w) == n and Enum.all?(w, &(&1 != nil)), do: f.(w), else: nil), q}
    end)
    List.to_tuple(out)
  end
  defp rolling(x, _, _), do: x

  defp ema(x, n) when is_tuple(x) do
    a = 2 / (n + 1)
    {out, _} = Enum.map_reduce(Tuple.to_list(x), {nil, 0}, fn
      nil, st -> {nil, st}
      v, {nil, c} -> {if(n <= 1, do: v, else: nil), {v, c + 1}}
      v, {e, c} -> e2 = a * v + (1 - a) * e; {if(c + 1 >= n, do: e2, else: nil), {e2, c + 1}}
    end)
    List.to_tuple(out)
  end
  defp ema(x, _), do: x

  # =================================================================== run

  defp signal_for(ast, closes, params) do
    env = Map.merge(%{"close" => List.to_tuple(closes)}, params)
    try do
      case eval_series(ast, env) do
        s when is_tuple(s) -> {:ok, s}
        c when is_number(c) -> {:ok, List.to_tuple(List.duplicate(c * 1.0, length(closes)))}
      end
    catch
      {:unknown, v} -> {:error, "unknown name #{v} (close, or a swept parameter)"}
      {:arg, w} -> {:error, w}
    end
  end

  @doc false
  # strategy returns for (t, t+1]: position = clip(signal_t), cost on |Δposition|
  def strategy(sig, closes, cost) do
    cl = List.to_tuple(closes); n = tuple_size(cl)
    {rets, _} =
      Enum.map_reduce(0..(n - 2), 0.0, fn t, prev ->
        p = case elem(sig, t) do nil -> 0.0; v -> v |> max(-1.0) |> min(1.0) end
        r = elem(cl, t + 1) / elem(cl, t) - 1
        {p * r - cost * abs(p - prev), p}
      end)
    rets
  end

  defp grid(sweeps) do
    Enum.reduce(sweeps, [%{}], fn {name, a, b, st}, acc ->
      vals = Stream.iterate(a, &(&1 + st)) |> Enum.take_while(&(&1 <= b + 1.0e-9))
      for m <- acc, v <- vals, do: Map.put(m, name, v)
    end)
  end

  defp evaluate(spec, data, ast, _opts) do
    trials = grid(spec.sweeps)
    cond do
      length(trials) > @max_trials -> {:error, "#{length(trials)} trials: at most #{@max_trials}"}
      true ->
        closes = data.close
        runs =
          Enum.reduce_while(trials, {:ok, []}, fn params, {:ok, acc} ->
            case signal_for(ast, closes, params) do
              {:ok, sig} -> {:cont, {:ok, [{params, sig, strategy(sig, closes, spec.cost)} | acc]}}
              e -> {:halt, e}
            end
          end)
        with {:ok, runs} <- runs do
          runs = Enum.reverse(runs)
          srs = Enum.map(runs, fn {_, _, r} -> sharpe(r) end)
          {best_params, best_sig, best} = Enum.at(runs, srs |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1))
          n_trials = length(runs) + spec.trials
          lookahead = prefix_invariance(ast, closes, best_params, best_sig)
          dsr = deflated_sharpe(best, srs, n_trials)
          pbo = if length(runs) >= 2, do: pbo(Enum.map(runs, &elem(&1, 2))), else: nil
          rc = reality_check(Enum.map(runs, &elem(&1, 2)), seed: 11)
          bh = strategy(List.to_tuple(List.duplicate(1.0, length(closes))), closes, 0.0)
          ann = spec.annual
          gates = [
            %{gate: "no look-ahead (prefix invariance)", pass: lookahead.clean, detail: lookahead.detail},
            %{gate: "deflated Sharpe > 0.95", pass: dsr.dsr > 0.95, detail: "DSR = #{f3(dsr.dsr)} with N = #{n_trials} trials"},
            %{gate: "PBO < 0.5", pass: pbo == nil or pbo.pbo < 0.5, detail: if(pbo, do: "PBO = #{f3(pbo.pbo)} over #{pbo.splits} splits", else: "one trial: not applicable")},
            %{gate: "Reality Check p < 0.05", pass: rc.p_value < 0.05, detail: "p = #{f3(rc.p_value)} (#{rc.bootstraps} stationary bootstraps)"}]
          {:ok, %{source: data.source, observations: length(closes), trials: length(runs), declared_trials: spec.trials, cost: spec.cost,
                  best: %{params: best_params, stats: stats(best, ann)}, buy_and_hold: stats(bh, ann),
                  equity: equity(best) |> thin(400), buy_and_hold_equity: equity(bh) |> thin(400), drawdown: drawdown(best) |> thin(400),
                  trial_table: Enum.zip(runs, srs) |> Enum.map(fn {{p, _, _}, s} -> %{params: p, sharpe_annual: s * :math.sqrt(ann)} end),
                  lookahead: lookahead, deflated_sharpe: dsr, pbo: pbo, reality_check: rc, gates: gates,
                  verdict: if(Enum.all?(gates, & &1.pass), do: "signal: every gate passed", else: "noise or flawed: #{gates |> Enum.reject(& &1.pass) |> Enum.map_join("; ", & &1.gate)} failed")}}
        end
    end
  end

  defp f3(x), do: :erlang.float_to_binary(x * 1.0, decimals: 3)

  defp thin(xs, m) do
    n = length(xs)
    if n <= m, do: xs, else: (step = n / m; for(i <- 0..(m - 1), do: Enum.at(xs, trunc(i * step))) ++ [List.last(xs)])
  end

  def sharpe(rets) do
    sd = Num.std(rets)
    if sd == 0.0, do: 0.0, else: Num.mean(rets) / sd
  end

  def stats(rets, ann) do
    sr = sharpe(rets); {sk, ku} = Num.moments(rets)
    eq = equity(rets)
    %{sharpe_annual: sr * :math.sqrt(ann), return_annual: Num.mean(rets) * ann, vol_annual: Num.std(rets) * :math.sqrt(ann),
      max_drawdown: drawdown(rets) |> Enum.min(fn -> 0.0 end), final_equity: List.last(eq), skew: sk, kurtosis: ku,
      hit_rate: Enum.count(rets, &(&1 > 0)) / max(Enum.count(rets, &(&1 != 0)), 1)}
  end

  defp equity(rets), do: rets |> Enum.scan(1.0, fn r, e -> e * (1 + r) end)

  defp drawdown(rets) do
    {dd, _} = Enum.map_reduce(equity(rets), 1.0, fn e, peak -> pk = max(peak, e); {e / pk - 1, pk} end)
    dd
  end

  # ---------------------------------------------------- the certificates

  @doc """
  Prefix invariance: the signal recomputed on `close[0..c]` must equal the
  full-history signal on [0, c] for every cut point c (eight, spread over
  the history). The first t where they differ is the look-ahead.
  """
  def prefix_invariance(ast, closes, params, full) do
    n = length(closes)
    cuts = for k <- 1..8, do: max(10, div(n * k, 9))
    bad =
      Enum.find_value(cuts, fn c ->
        {:ok, part} = signal_for(ast, Enum.take(closes, c + 1), params)
        Enum.find_value(0..c, fn t -> if elem(part, t) != elem(full, t), do: {c, t, elem(full, t), elem(part, t)} end)
      end)
    case bad do
      nil -> %{clean: true, cuts: cuts, detail: "identical on #{length(cuts)} truncated histories"}
      {c, t, a, b} ->
        v = fn nil -> "no value"; x -> :erlang.float_to_binary(x * 1.0, [:short]) end
        %{clean: false, cuts: cuts, cut: c, day: t, full: a, truncated: b,
          detail: "day #{t}: #{v.(a)} with the full history, #{v.(b)} when the data stops at day #{c} — the signal uses the future"}
    end
  end

  @doc """
  Probabilistic Sharpe ratio of a return series against a benchmark
  Sharpe `sr0` (per period): Φ((ŜR − SR₀)√(T − 1) / √(1 − γ₃ŜR + (γ₄ − 1)/4·ŜR²)).
  """
  def psr(rets, sr0) do
    t = length(rets); sr = sharpe(rets); {g3, g4} = Num.moments(rets)
    den = 1 - g3 * sr + (g4 - 1) / 4 * sr * sr
    Num.ncdf((sr - sr0) * :math.sqrt(t - 1) / :math.sqrt(max(den, 1.0e-12)))
  end

  @euler 0.5772156649015329

  @doc """
  Deflated Sharpe ratio: the PSR against SR₀ = √V[ŜRₙ]·((1 − γ)Φ⁻¹(1 − 1/N) + γΦ⁻¹(1 − 1/(Ne))),
  the expected maximum Sharpe of N unskilled trials (V from the trials;
  with one trial, SR₀ = 0 and the DSR is the PSR).
  """
  def deflated_sharpe(best, srs, n) do
    v = if length(srs) > 1, do: Num.var(srs), else: 1 / max(length(best) - 1, 1)
    sr0 = if n > 1, do: :math.sqrt(v) * ((1 - @euler) * Num.ninv(1 - 1 / n) + @euler * Num.ninv(1 - 1 / (n * :math.exp(1)))), else: 0.0
    %{dsr: psr(best, sr0), psr: psr(best, 0.0), sr_best: sharpe(best), sr0_expected_max: sr0, trials: n, trial_sr_variance: v}
  end

  @doc """
  Probability of backtest overfitting by CSCV: the T × N matrix of trial
  returns is cut into S = 16 blocks; for each of the C(16, 8) = 12 870
  halvings, the in-sample best trial's out-of-sample rank ω gives
  λ = logit(ω); PBO = P(λ ≤ 0).
  """
  def pbo(trial_rets, s \\ 16) do
    t = length(hd(trial_rets)); n = length(trial_rets)
    blk = div(t, s)
    # per block and trial: Σr, Σr², count — so each split's Sharpe is a sum of blocks
    per =
      for r <- trial_rets do
        tup = List.to_tuple(r)
        for b <- 0..(s - 1) do
          xs = for i <- (b * blk)..(b * blk + blk - 1), do: elem(tup, i)
          {Enum.sum(xs), Enum.reduce(xs, 0.0, &(&1 * &1 + &2)), blk}
        end |> List.to_tuple()
      end |> List.to_tuple()
    sr_of = fn trial, blocks ->
      {a, b, c} = Enum.reduce(blocks, {0.0, 0.0, 0}, fn bi, {a, b, c} -> {x, y, z} = elem(elem(per, trial), bi); {a + x, b + y, c + z} end)
      m = a / c; v = (b - c * m * m) / max(c - 1, 1)
      if v <= 0, do: 0.0, else: m / :math.sqrt(v)
    end
    all = Enum.to_list(0..(s - 1))
    combos = combinations(all, div(s, 2))
    lambdas =
      for is <- combos do
        os = all -- is
        isr = for k <- 0..(n - 1), do: sr_of.(k, is)
        best = isr |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1)
        osr = for k <- 0..(n - 1), do: sr_of.(k, os)
        mine = Enum.at(osr, best)
        rank = Enum.count(osr, &(&1 < mine)) + 0.5 * (Enum.count(osr, &(&1 == mine)) - 1) + 1
        w = rank / (n + 1)
        :math.log(w / (1 - w))
      end
    %{pbo: Enum.count(lambdas, &(&1 <= 0)) / length(lambdas), splits: length(lambdas), blocks: s,
      logit_histogram: histogram(lambdas, -4.0, 4.0, 32)}
  end

  defp combinations(_, 0), do: [[]]
  defp combinations([], _), do: []
  defp combinations([h | t], k), do: Enum.map(combinations(t, k - 1), &[h | &1]) ++ combinations(t, k)

  defp histogram(xs, lo, hi, bins) do
    w = (hi - lo) / bins
    counts = Enum.reduce(xs, List.duplicate(0, bins), fn x, c -> i = trunc((min(max(x, lo), hi - 1.0e-9) - lo) / w); List.update_at(c, i, &(&1 + 1)) end)
    for {c, i} <- Enum.with_index(counts), do: %{from: lo + i * w, to: lo + (i + 1) * w, count: c}
  end

  @doc """
  White's Reality Check: the best trial's mean return against the
  bootstrap distribution of max_k (mean*_k − mean_k), resampling days by
  Politis–Romano's stationary bootstrap (mean block length 10).
  """
  def reality_check(trial_rets, opts \\ []) do
    b = Keyword.get(opts, :bootstraps, 300); seed = Keyword.get(opts, :seed, 1); q = 1 / Keyword.get(opts, :block, 10)
    t = length(hd(trial_rets))
    tups = Enum.map(trial_rets, &List.to_tuple/1)
    means = Enum.map(trial_rets, &Num.mean/1)
    stat = :math.sqrt(t) * Enum.max(means)
    us = Num.uniforms(seed, b * t * 2) |> List.to_tuple()
    exceed =
      Enum.count(0..(b - 1), fn bi ->
        {idx, _} = Enum.map_reduce(0..(t - 1), nil, fn i, prev ->
          u1 = elem(us, (bi * t + i) * 2); u2 = elem(us, (bi * t + i) * 2 + 1)
          j = if prev == nil or u1 < q, do: trunc(u2 * t), else: rem(prev + 1, t)
          {j, j}
        end)
        boot = Enum.zip(tups, means) |> Enum.map(fn {tp, m} -> Enum.reduce(idx, 0.0, &(&2 + elem(tp, &1))) / t - m end)
        :math.sqrt(t) * Enum.max(boot) >= stat
      end)
    %{p_value: exceed / b, bootstraps: b, statistic: stat}
  end
end
