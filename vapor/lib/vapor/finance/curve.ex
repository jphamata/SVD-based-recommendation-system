defmodule Vapor.Finance.Curve do
  @moduledoc """
  Discount curves bootstrapped from the instruments a desk actually quotes
  (docs/FINANCAS.md §3) — typed as text:

      date = 2025-01-02
      calendar = anbima
      basis = du252
      interpolation = flat_forward
      di1 F26 = 15.02%
      ltn 2027-01-01 price = 732.15
      ntnf 2031-01-01 price = 865.27
      deposit 2025-04-01 = 12.15% simple
      zero 2029-01-02 = 13.4%
      bond 2030-06-15 coupon = 4% freq = 2 price = 98.5 clean
      swap 5y = 4.10% freq = 1

  The bootstrap places one node at each instrument's last cash flow and
  solves (Brent) for the discount factor that reprices it, with the earlier
  nodes held. Interpolation: `flat_forward` (log-linear discount — the
  ANBIMA/B3 convention for the DI curve) or `linear_zero` (linear in the
  continuously compounded zero rate). The **certificate** reprices every
  instrument from the finished curve and reports the worst error; the
  forward rates between nodes are listed, and a negative forward is
  flagged (an arbitrage in the quotes, or a typo).

  `fit nss` adds a Nelson–Siegel–Svensson fit of the bootstrapped zero
  rates (β by exact least squares on a grid of τ₁, τ₂; then refined).
  """
  alias Vapor.Finance.{Calendar, Num}

  @ntnf_coupon 1000 * (:math.sqrt(1.1) - 1)

  # ------------------------------------------------------------- parsing

  @doc "Bootstrap a curve from text: `{:ok, %{nodes, instruments, certificate, forwards, …}}`."
  def run(text) do
    with {:ok, spec} <- parse(text), {:ok, curve} <- bootstrap(spec) do
      {:ok, curve |> report(spec)}
    end
  end

  def parse(text) do
    lines = text |> String.split("\n") |> Enum.with_index(1) |> Enum.map(fn {l, i} -> {l |> String.split("#") |> hd() |> String.trim(), i} end) |> Enum.reject(&(elem(&1, 0) == ""))
    init = %{date: nil, calendar: :anbima, basis: :bus252, interpolation: :flat_forward, compounding: :annual, instruments: [], fit: false}
    Enum.reduce_while(lines, {:ok, init}, fn {l, i}, {:ok, s} ->
      case line(l, s) do
        {:ok, s} -> {:cont, {:ok, s}}
        {:error, w} -> {:halt, {:error, "line #{i}: #{w}"}}
      end
    end)
    |> case do
      {:ok, %{date: nil}} -> {:error, "date = YYYY-MM-DD (the valuation date) is missing"}
      {:ok, %{instruments: []}} -> {:error, "no instruments (di1, ltn, ntnf, deposit, zero, bond, swap)"}
      {:ok, s} -> {:ok, %{s | instruments: Enum.reverse(s.instruments)}}
      e -> e
    end
  end

  defp line(l, s) do
    low = String.downcase(l)
    cond do
      m = Regex.run(~r/^(?:date|data)\s*=\s*(\d{4}-\d{2}-\d{2})$/i, l) -> with({:ok, d} <- Date.from_iso8601(Enum.at(m, 1)), do: {:ok, %{s | date: d}})
      m = Regex.run(~r/^(?:calendar|calendário|calendario)\s*=\s*(\S+)$/i, l) -> with({:ok, c} <- Calendar.parse_calendar(Enum.at(m, 1)), do: {:ok, %{s | calendar: c}})
      m = Regex.run(~r/^(?:basis|base)\s*=\s*(.+)$/i, l) -> with({:ok, b} <- Calendar.parse_basis(Enum.at(m, 1)), do: {:ok, %{s | basis: b}})
      m = Regex.run(~r/^(?:interpolation|interpolação|interpolacao)\s*=\s*(\S+)$/i, l) ->
        case String.downcase(Enum.at(m, 1)) do
          x when x in ["flat_forward", "flatforward", "log_linear", "loglinear", "exponencial"] -> {:ok, %{s | interpolation: :flat_forward}}
          x when x in ["linear_zero", "linear"] -> {:ok, %{s | interpolation: :linear_zero}}
          x -> {:error, "interpolation #{x}: flat_forward or linear_zero"}
        end
      m = Regex.run(~r/^(?:compounding|capitalização|capitalizacao)\s*=\s*(\S+)$/i, l) -> with({:ok, c} <- compounding(Enum.at(m, 1)), do: {:ok, %{s | compounding: c}})
      low =~ ~r/^fit\s+nss$/ -> {:ok, %{s | fit: true}}
      true -> with({:ok, inst} <- instrument(l, s), do: {:ok, %{s | instruments: [Map.put(inst, :text, l) | s.instruments]}})
    end
  end

  defp compounding(x) do
    case String.downcase(x) do
      c when c in ["annual", "anual", "exponential", "exponencial"] -> {:ok, :annual}
      c when c in ["continuous", "contínua", "continua"] -> {:ok, :continuous}
      c when c in ["simple", "simples", "linear"] -> {:ok, :simple}
      c -> {:error, "compounding #{c}: annual, continuous or simple"}
    end
  end

  defp rate(s) do
    s = String.trim(s)
    case Float.parse(String.replace(s, ",", ".")) do
      {v, "%"} -> {:ok, v / 100}
      {v, ""} -> {:ok, if(abs(v) > 1, do: v / 100, else: v)}
      _ -> {:error, "not a rate: #{s}"}
    end
  end

  defp num(s), do: (case Float.parse(String.trim(s) |> String.replace(",", ".")) do {v, _} -> {:ok, v}; :error -> {:error, "not a number: #{s}"} end)

  defp opts(rest) do
    Regex.scan(~r/(\w+)\s*=\s*([^\s]+)/u, rest) |> Map.new(fn [_, k, v] -> {String.downcase(k), v} end)
    |> Map.put("_flags", Regex.scan(~r/\b(simple|simples|continuous|contínua|clean|limpo|dirty|sujo|annual|anual)\b/i, rest) |> Enum.map(&String.downcase(hd(&1))))
  end

  defp flag_comp(o, default) do
    cond do
      Enum.any?(o["_flags"], &(&1 in ["simple", "simples"])) -> :simple
      Enum.any?(o["_flags"], &(&1 in ["continuous", "contínua"])) -> :continuous
      Enum.any?(o["_flags"], &(&1 in ["annual", "anual"])) -> :annual
      true -> default
    end
  end

  defp instrument(l, s) do
    cond do
      m = Regex.run(~r/^di1\s*([fghjkmnquvxz]\d{2})\s*=?\s*([\d.,]+%?)$/i, l) ->
        with {:ok, mat} <- Calendar.di1_maturity(Enum.at(m, 1), s.date.year), {:ok, r} <- rate(Enum.at(m, 2)),
             do: {:ok, %{kind: :di1, maturity: mat, rate: r, code: String.upcase(Enum.at(m, 1))}}

      m = Regex.run(~r/^(deposit|depósito|deposito|zero)\s+(\d{4}-\d{2}-\d{2})\s*=\s*([-\d.,]+%?)(.*)$/i, l) ->
        with {:ok, d} <- Date.from_iso8601(Enum.at(m, 2)), {:ok, r} <- rate(Enum.at(m, 3)) do
          {:ok, %{kind: :zero, maturity: d, rate: r, compounding: flag_comp(opts(Enum.at(m, 4)), if(String.downcase(Enum.at(m, 1)) == "zero", do: s.compounding, else: s.compounding))}}
        end

      m = Regex.run(~r/^ltn\s+(\d{4}-\d{2}-\d{2})\s+(?:price|preço|preco|pu)\s*=\s*([\d.,]+)$/i, l) ->
        with {:ok, d} <- Date.from_iso8601(Enum.at(m, 1)), {:ok, p} <- num(Enum.at(m, 2)), do: {:ok, %{kind: :ltn, maturity: d, price: p}}

      m = Regex.run(~r/^ntn-?f\s+(\d{4}-\d{2}-\d{2})\s+(?:price|preço|preco|pu)\s*=\s*([\d.,]+)$/i, l) ->
        with {:ok, d} <- Date.from_iso8601(Enum.at(m, 1)), {:ok, p} <- num(Enum.at(m, 2)), do: {:ok, %{kind: :ntnf, maturity: d, price: p}}

      m = Regex.run(~r/^bond\s+(\d{4}-\d{2}-\d{2})(.*)$/i, l) ->
        o = opts(Enum.at(m, 2))
        with {:ok, d} <- Date.from_iso8601(Enum.at(m, 1)), {:ok, c} <- rate(o["coupon"] || o["cupom"] || "0"),
             {:ok, p} <- num(o["price"] || o["preço"] || o["preco"] || "nil"),
             {freq, _} <- Integer.parse(o["freq"] || "2") do
          clean = Enum.any?(o["_flags"], &(&1 in ["clean", "limpo"]))
          {:ok, %{kind: :bond, maturity: d, coupon: c, freq: freq, price: p, clean: clean}}
        else
          _ -> {:error, "bond YYYY-MM-DD coupon = 4% freq = 2 price = 98.5 [clean]"}
        end

      m = Regex.run(~r/^swap\s+(\d+)\s*([ymd])\s*=\s*([-\d.,]+%?)(.*)$/i, l) ->
        o = opts(Enum.at(m, 4))
        with {:ok, r} <- rate(Enum.at(m, 3)), {freq, _} <- Integer.parse(o["freq"] || "1") do
          n = String.to_integer(Enum.at(m, 1))
          months = case String.downcase(Enum.at(m, 2)) do "y" -> 12 * n; "m" -> n; "d" -> nil end
          mat = if months, do: Calendar.adjust(s.calendar, Calendar.add_months(s.date, months), :modified_following), else: Calendar.add_business_days(s.calendar, s.date, n)
          {:ok, %{kind: :swap, maturity: mat, rate: r, freq: freq, tenor_months: months || 0}}
        end

      true -> {:error, "not understood: #{inspect(l)} — di1 F26 = 15%, ltn/ntnf DATE price = …, deposit/zero DATE = r%, bond DATE coupon = … price = …, swap 5y = r%"}
    end
  end

  # --------------------------------------------------------- the pricing

  @doc false
  def tau(spec, d), do: Calendar.year_fraction(spec.basis, spec.date, d, spec.calendar)

  defp df_of_rate(r, t, :annual), do: :math.pow(1 + r, -t)
  defp df_of_rate(r, t, :continuous), do: :math.exp(-r * t)
  defp df_of_rate(r, t, :simple), do: 1 / (1 + r * t)

  @doc false
  def rate_of_df(df, t, _) when t <= 0, do: (_ = df; 0.0)
  def rate_of_df(df, t, :annual), do: :math.pow(df, -1 / t) - 1
  def rate_of_df(df, t, :continuous), do: -:math.log(df) / t
  def rate_of_df(df, t, :simple), do: (1 / df - 1) / t

  # cash flows {date, amount} of an instrument, and its target value (what the flows must be worth)
  @doc false
  def flows(%{kind: :di1, maturity: m, rate: r}, spec), do: {[{m, 100_000.0}], 100_000.0 * df_of_rate(r, tau(spec, m), :annual)}
  def flows(%{kind: :zero, maturity: m, rate: r, compounding: c}, spec), do: {[{m, 1.0}], df_of_rate(r, tau(spec, m), c)}
  # Treasury bills and notes pay on the next business day when the date is a holiday (DU counted to it)
  def flows(%{kind: :ltn, maturity: m, price: p}, _spec), do: {[{Calendar.adjust(:anbima, m, :following), 1000.0}], p}

  def flows(%{kind: :ntnf, maturity: m, price: p}, spec) do
    # coupons on 1 January and 1 July, paid on the next business day; principal with the last coupon
    dates = coupon_dates(m, 6, spec.date) |> Enum.map(&Calendar.adjust(:anbima, &1, :following))
    cfs = Enum.map(dates, &{&1, @ntnf_coupon})
    {List.update_at(cfs, -1, fn {d, c} -> {d, c + 1000.0} end), p}
  end

  def flows(%{kind: :bond, maturity: m, coupon: c, freq: f, price: p, clean: clean}, spec) do
    months = div(12, f)
    dates = coupon_dates(m, months, spec.date)
    cpn = 100 * c / f
    cfs = Enum.map(dates, &{&1, cpn}) |> List.update_at(-1, fn {d, x} -> {d, x + 100.0} end)
    # a clean quote: add the accrued interest since the previous coupon date (in the curve's basis)
    accrued =
      if clean do
        nxt = hd(dates); prv = Calendar.add_months(nxt, -months)
        cpn * Calendar.year_fraction(spec.basis, prv, spec.date, spec.calendar) / Calendar.year_fraction(spec.basis, prv, nxt, spec.calendar)
      else
        0.0
      end
    {cfs, p + accrued}
  end

  def flows(%{kind: :swap, maturity: m, rate: r, freq: f, tenor_months: tm}, spec) do
    # single-curve par swap: the fixed leg Σ r·α·D(tᵢ) plus the notional at the end is worth par;
    # the schedule rolls forward from the valuation date (unadjusted), each date adjusted
    months = div(12, f)
    dates =
      if tm > 0,
        do: for(k <- 1..max(div(tm, months), 1), do: Calendar.adjust(spec.calendar, Calendar.add_months(spec.date, k * months), :modified_following)),
        else: [m]
    {cfs, _} = Enum.map_reduce(dates, spec.date, fn d, prev -> {{d, r * Calendar.year_fraction(spec.basis, prev, d, spec.calendar)}, d} end)
    {List.update_at(cfs, -1, fn {d, x} -> {d, x + 1.0} end), 1.0}
  end

  # payment dates strictly after `after`, rolling back from maturity by `months`
  defp coupon_dates(mat, months, start) do
    Stream.iterate(0, &(&1 + 1)) |> Enum.reduce_while([], fn k, acc ->
      d = Calendar.add_months(mat, -k * months)
      if Date.compare(d, start) == :gt, do: {:cont, [d | acc]}, else: {:halt, acc}
    end)
  end

  # ------------------------------------------------------- the curve

  @doc "Discount factor at year-fraction t from nodes [{t, D}] (t₀ = 0, D = 1 first)."
  def df(nodes, t, interp) do
    cond do
      t <= 0 -> 1.0
      true ->
        {left, right} = bracket(nodes, t)
        case {left, right} do
          {{t1, d1}, nil} ->
            # beyond the last node: the last forward held flat
            {t0, d0} = prev_node(nodes, t1)
            f = if t1 > t0, do: :math.log(d1 / d0) / (t1 - t0), else: 0.0
            d1 * :math.exp(f * (t - t1))
          {{t0, d0}, {t1, d1}} ->
            w = (t - t0) / (t1 - t0)
            case interp do
              :flat_forward -> :math.exp((1 - w) * :math.log(d0) + w * :math.log(d1))
              :linear_zero ->
                z0 = if t0 > 0, do: -:math.log(d0) / t0, else: (if t1 > 0, do: -:math.log(d1) / t1, else: 0.0)
                z1 = -:math.log(d1) / t1
                :math.exp(-((1 - w) * z0 + w * z1) * t)
            end
        end
    end
  end

  defp bracket(nodes, t) do
    case Enum.split_while(nodes, fn {ti, _} -> ti < t end) do
      {ls, [r | _]} -> {List.last(ls) || {0.0, 1.0}, r}
      {ls, []} -> {List.last(ls), nil}
    end
  end

  defp prev_node(nodes, t1) do
    nodes |> Enum.filter(fn {t, _} -> t < t1 end) |> List.last() || {0.0, 1.0}
  end

  defp pv({cfs, _target}, nodes, spec), do: Enum.reduce(cfs, 0.0, fn {d, a}, s -> s + a * df(nodes, tau(spec, d), spec.interpolation) end)

  def bootstrap(spec) do
    insts =
      spec.instruments
      |> Enum.map(fn i -> {fl, tg} = flows(i, spec); Map.merge(i, %{flows: fl, target: tg, t: tau(spec, elem(List.last(fl), 0))}) end)
      |> Enum.sort_by(& &1.t)

    dup = insts |> Enum.chunk_by(& &1.t) |> Enum.find(&(length(&1) > 1))
    cond do
      dup -> {:error, "two instruments end at the same point (#{Enum.map_join(dup, " and ", & &1.text)}): keep one"}
      Enum.any?(insts, &(&1.t <= 0)) -> {:error, "an instrument matures on or before the valuation date"}
      true ->
        Enum.reduce_while(insts, {:ok, [{0.0, 1.0}]}, fn i, {:ok, nodes} ->
          g = fn d -> pv({i.flows, i.target}, nodes ++ [{i.t, d}], spec) - i.target end
          case Num.brent(g, 1.0e-6, 3.0, 1.0e-15) do
            {:ok, d} -> {:cont, {:ok, nodes ++ [{i.t, d}]}}
            {:error, _} -> {:halt, {:error, "no discount factor in (0, 3] reprices #{i.text} — the quote is inconsistent with the shorter instruments"}}
          end
        end)
        |> case do
          {:ok, nodes} -> {:ok, %{nodes: nodes, instruments: insts}}
          e -> e
        end
    end
  end

  defp report(%{nodes: nodes, instruments: insts}, spec) do
    comp = spec.compounding
    date_of = fn t -> Enum.find(insts, &(&1.t == t)) end
    node_rows =
      for {t, d} <- tl(nodes) do
        i = date_of.(t)
        %{date: Date.to_iso8601(elem(List.last(i.flows), 0)), t: t, df: d, zero: rate_of_df(d, t, comp), instrument: i.text}
      end

    {fwds, _} =
      Enum.map_reduce(tl(nodes), hd(nodes), fn {t, d}, {t0, d0} ->
        f = rate_of_df(d / d0, t - t0, comp)
        {%{from: t0, to: t, forward: f}, {t, d}}
      end)

    reprice =
      for i <- insts do
        model = pv({i.flows, i.target}, nodes, spec)
        rel = abs(model - i.target) / max(abs(i.target), 1.0e-12)
        %{instrument: i.text, quote_value: i.target, model_value: model, rel_error: rel}
      end

    worst = reprice |> Enum.map(& &1.rel_error) |> Enum.max()
    neg = Enum.filter(fwds, &(&1.forward < 0))
    tmax = elem(List.last(nodes), 0)
    dense = for k <- 0..120, t = tmax * k / 120, t > 0, do: (d = df(nodes, t, spec.interpolation); %{t: t, zero: rate_of_df(d, t, comp), df: d})
    inst_fwd = for k <- 1..240, t = tmax * k / 240, do: %{t: t, forward: inst_forward(nodes, t, spec.interpolation, comp)}

    base = %{date: Date.to_iso8601(spec.date), calendar: spec.calendar, basis: spec.basis, interpolation: spec.interpolation, compounding: comp,
             nodes: node_rows, forwards: fwds, dense: dense, forward_curve: inst_fwd,
             certificate: %{reprice: reprice, max_rel_error: worst, repriced: worst < 1.0e-10, negative_forwards: length(neg),
                            verdict: cond do
                              worst >= 1.0e-10 -> "an instrument is not repriced: the curve does not hold"
                              neg != [] -> "every instrument repriced, but #{length(neg)} forward(s) are negative — check the quotes"
                              true -> "every instrument repriced to #{:erlang.float_to_binary(worst, [{:scientific, 1}])}; all forwards positive"
                            end}}
    if spec.fit, do: Map.put(base, :nss, nss(Enum.map(node_rows, &{&1.t, &1.zero}))), else: base
  end

  defp inst_forward(nodes, t, interp, comp) do
    h = 1.0e-4
    d1 = df(nodes, t, interp); d2 = df(nodes, t + h, interp)
    rate_of_df(d2 / d1, h, comp)
  end

  # ------------------------------------------------------ Nelson–Siegel–Svensson

  @doc """
  Fit z(t) = β₀ + β₁·L(t/τ₁) + β₂·(L(t/τ₁) − e^(−t/τ₁)) + β₃·(L(t/τ₂) − e^(−t/τ₂)),
  L(x) = (1 − e^(−x))/x, to points [{t, z}]: the β exactly by least squares
  for each (τ₁, τ₂) on a grid, the best pair refined by Nelder–Mead.
  """
  def nss(points) do
    pts = Enum.filter(points, fn {t, _} -> t > 0 end)
    if length(pts) < 4 do
      %{error: "at least four points are needed for Nelson–Siegel–Svensson"}
    else
      sse = fn [l1, l2] ->
        if l1 <= 0.02 or l2 <= 0.02 or l1 > 30 or l2 > 30 do
          {1.0e9, nil}
        else
          x = for {t, _} <- pts, do: nss_row(t, l1, l2)
          y = for {_, z} <- pts, do: z
          case Vapor.Dense.lstsq(x, y) do
            {:ok, b} -> {Enum.zip_with(Vapor.Dense.matvec(x, b), y, &((&1 - &2) * (&1 - &2))) |> Enum.sum(), b}
            _ -> {1.0e9, nil}
          end
        end
      end
      grid = for a <- [0.25, 0.5, 1.0, 2.0, 3.0, 5.0], b <- [0.5, 1.0, 2.0, 5.0, 8.0, 12.0], a < b, do: [a, b]
      start = Enum.min_by(grid, &elem(sse.(&1), 0))
      {best, _, _} = Num.nelder_mead(fn p -> elem(sse.(p), 0) end, start, step: 0.2, tol: 1.0e-14)
      {e, beta} = sse.(best)
      rmse = :math.sqrt(e / length(pts))
      %{beta: beta, tau: best, rmse: rmse, fitted: for({t, z} <- pts, do: %{t: t, zero: z, nss: Vapor.Dense.dot(nss_row(t, Enum.at(best, 0), Enum.at(best, 1)), beta)})}
    end
  end

  defp nss_row(t, l1, l2) do
    lf = fn x -> if x < 1.0e-8, do: 1 - x / 2, else: (1 - :math.exp(-x)) / x end
    a = lf.(t / l1); b = lf.(t / l2)
    [1.0, a, a - :math.exp(-t / l1), b - :math.exp(-t / l2)]
  end
end
