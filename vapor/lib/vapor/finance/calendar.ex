defmodule Vapor.Finance.Calendar do
  @moduledoc """
  Business-day calendars and day-count conventions (docs/FINANCAS.md §2).

  Every holiday is a **rule**, not a downloaded list: fixed dates, nth
  weekday of a month, and the moveable feasts from the Gregorian computus
  (Easter by the anonymous algorithm of 1876). The rules are pinned
  against QuantLib's calendars day by day over a century
  (`financas_test.exs`), and the special closings that no rule produces
  (a day of mourning, a hurricane) are data, listed with their reason.

  | calendar | what it is |
  |---|---|
  | `:anbima` | Brazilian national settlement calendar (ANBIMA; B3's DU/252 counts) — Carnival Monday and Tuesday, Good Friday, Corpus Christi, and the fixed national holidays (20 November from 2024) |
  | `:nyse` | New York Stock Exchange — observed-date rules, MLK, Juneteenth (2022–), Good Friday, and listed special closings |
  | `:target` | TARGET2 (euro settlement) |
  | `:weekends` | Saturdays and Sundays only |

  Day counts (`year_fraction/4`): `:bus252` (DU/252), `:act360`,
  `:act365f`, `:thirty360` (US bond basis), `:thirty_e360` (Eurobond),
  `:act_act_isda`.
  """

  @calendars [:anbima, :nyse, :target, :weekends]
  def calendars, do: @calendars

  # ------------------------------------------------------------ the computus

  @doc "Easter Sunday of a Gregorian year (Meeus/Jones/Butcher)."
  def easter(y) do
    a = rem(y, 19); b = div(y, 100); c = rem(y, 100)
    d = div(b, 4); e = rem(b, 4); f = div(b + 8, 25); g = div(b - f + 1, 3)
    h = rem(19 * a + b - d - g + 15, 30)
    i = div(c, 4); k = rem(c, 4)
    l = rem(32 + 2 * e + 2 * i - h - k, 7)
    m = div(a + 11 * h + 22 * l, 451)
    month = div(h + l - 7 * m + 114, 31)
    day = rem(h + l - 7 * m + 114, 31) + 1
    Date.new!(y, month, day)
  end

  defp nth_weekday(y, m, dow, n) do
    first = Date.new!(y, m, 1)
    shift = rem(dow - Date.day_of_week(first) + 7, 7)
    Date.add(first, shift + 7 * (n - 1))
  end

  defp last_weekday(y, m, dow) do
    last = Date.new!(y, m, Calendar.ISO.days_in_month(y, m))
    Date.add(last, -rem(Date.day_of_week(last) - dow + 7, 7))
  end

  def weekend?(d), do: Date.day_of_week(d) in [6, 7]

  # --------------------------------------------------------------- holidays

  @doc "The holidays of `cal` in year `y` that fall on weekdays, sorted, each with its name."
  def holidays(cal, y) when cal in @calendars do
    rules(cal, y) |> Enum.reject(fn {d, _} -> weekend?(d) or d.year != y end) |> Enum.uniq_by(&elem(&1, 0)) |> Enum.sort_by(&Date.to_gregorian_days(elem(&1, 0)))
  end

  defp rules(:weekends, _), do: []

  defp rules(:anbima, y) do
    e = easter(y)
    [{Date.new!(y, 1, 1), "Confraternização Universal"},
     {Date.add(e, -48), "Carnaval (segunda)"}, {Date.add(e, -47), "Carnaval (terça)"},
     {Date.add(e, -2), "Sexta-feira Santa"},
     {Date.new!(y, 4, 21), "Tiradentes"}, {Date.new!(y, 5, 1), "Dia do Trabalho"},
     {Date.add(e, 60), "Corpus Christi"},
     {Date.new!(y, 9, 7), "Independência"}, {Date.new!(y, 10, 12), "Nossa Senhora Aparecida"},
     {Date.new!(y, 11, 2), "Finados"}, {Date.new!(y, 11, 15), "Proclamação da República"},
     {Date.new!(y, 12, 25), "Natal"}] ++
      if(y >= 2024, do: [{Date.new!(y, 11, 20), "Dia Nacional de Zumbi e da Consciência Negra (Lei 14.759/2023)"}], else: [])
  end

  defp rules(:target, y) do
    e = easter(y)
    # Good Friday, Easter Monday, 1 May and 26 December from 2000; 31 December in 1998, 1999 and 2001 only
    [{Date.new!(y, 1, 1), "New Year's Day"}, {Date.new!(y, 12, 25), "Christmas Day"}] ++
      if(y >= 2000, do: [{Date.add(e, -2), "Good Friday"}, {Date.add(e, 1), "Easter Monday"}, {Date.new!(y, 5, 1), "Labour Day"}, {Date.new!(y, 12, 26), "Christmas Holiday"}], else: []) ++
      if(y in [1998, 1999, 2001], do: [{Date.new!(y, 12, 31), "TARGET closing (#{y})"}], else: [])
  end

  defp rules(:nyse, y) do
    e = easter(y)
    obs = fn d -> case Date.day_of_week(d) do 6 -> Date.add(d, -1); 7 -> Date.add(d, 1); _ -> d end end
    # New Year's Day: a Sunday moves to Monday; a Saturday is NOT observed on the Friday before (NYSE rule 7.2)
    ny = Date.new!(y, 1, 1)
    ny = if Date.day_of_week(ny) == 7, do: Date.add(ny, 1), else: ny
    [{ny, "New Year's Day"},
     {Date.add(e, -2), "Good Friday"},
     {nth_weekday(y, 2, 1, 3), if(y >= 1971, do: "Washington's Birthday", else: "Washington's Birthday")},
     {last_weekday(y, 5, 1), "Memorial Day"},
     {obs.(Date.new!(y, 7, 4)), "Independence Day"},
     {nth_weekday(y, 9, 1, 1), "Labor Day"},
     {nth_weekday(y, 11, 4, 4), "Thanksgiving Day"},
     {obs.(Date.new!(y, 12, 25)), "Christmas Day"}] ++
      if(y >= 1998, do: [{nth_weekday(y, 1, 1, 3), "Martin Luther King, Jr. Day"}], else: []) ++
      if(y >= 2022, do: [{obs.(Date.new!(y, 6, 19)), "Juneteenth"}], else: []) ++
      Enum.filter(nyse_special(), fn {d, _} -> d.year == y end)
  end

  # closings no rule produces (NYSE notices); kept as data with their reason
  defp nyse_special do
    [{~D[1994-04-27], "Day of mourning: President Nixon"}, {~D[2001-09-11], "September 11 attacks"}, {~D[2001-09-12], "September 11 attacks"}, {~D[2001-09-13], "September 11 attacks"}, {~D[2001-09-14], "September 11 attacks"},
     {~D[2004-06-11], "Day of mourning: President Reagan"}, {~D[2007-01-02], "Day of mourning: President Ford"},
     {~D[2012-10-29], "Hurricane Sandy"}, {~D[2012-10-30], "Hurricane Sandy"},
     {~D[2018-12-05], "Day of mourning: President G. H. W. Bush"}, {~D[2025-01-09], "Day of mourning: President Carter"}]
  end

  @doc "Is `d` a business day of `cal`?"
  def business_day?(cal, d), do: not weekend?(d) and not holiday?(cal, d)

  def holiday?(cal, d), do: Enum.any?(holiday_set(cal, d.year), &(&1 == d))

  # holidays of a year, memoized in the process dictionary (rules are cheap, but day loops call this often)
  defp holiday_set(cal, y) do
    key = {__MODULE__, cal, y}
    case Process.get(key) do
      nil -> (s = holidays(cal, y) |> Enum.map(&elem(&1, 0)) |> MapSet.new(); Process.put(key, s); s)
      s -> s
    end
  end

  @doc "Business days in [from, to) — the DU of the Brazilian market when `cal = :anbima`. Negative when to < from."
  def business_days(cal, from, to) do
    cond do
      Date.compare(to, from) == :lt -> -business_days(cal, to, from)
      true ->
        n = Date.diff(to, from)
        # whole weeks at once, then the rest day by day; holidays subtracted per year
        full = div(n, 7)
        rest = Enum.count(0..(rem(n, 7) - 1)//1, fn k -> not weekend?(Date.add(from, full * 7 + k)) end)
        weekdays = full * 5 + rest
        hol = for y <- from.year..to.year, d <- holiday_set(cal, y), Date.compare(d, from) != :lt, Date.compare(d, to) == :lt, do: d
        weekdays - length(hol)
    end
  end

  @doc "Move a date by business-day convention: `:following`, `:modified_following`, `:preceding`, `:modified_preceding`, `:unadjusted`."
  def adjust(_cal, d, :unadjusted), do: d
  def adjust(cal, d, :following), do: (if business_day?(cal, d), do: d, else: adjust(cal, Date.add(d, 1), :following))
  def adjust(cal, d, :preceding), do: (if business_day?(cal, d), do: d, else: adjust(cal, Date.add(d, -1), :preceding))
  def adjust(cal, d, :modified_following), do: (f = adjust(cal, d, :following); if f.month != d.month, do: adjust(cal, d, :preceding), else: f)
  def adjust(cal, d, :modified_preceding), do: (p = adjust(cal, d, :preceding); if p.month != d.month, do: adjust(cal, d, :following), else: p)

  @doc "Add `n` business days (n may be negative); from a holiday, counting starts at the adjusted date."
  def add_business_days(cal, d, 0), do: adjust(cal, d, :following)
  def add_business_days(cal, d, n) when n > 0, do: Enum.reduce(1..n, d, fn _, x -> adjust(cal, Date.add(x, 1), :following) end)
  def add_business_days(cal, d, n) when n < 0, do: Enum.reduce(1..-n, d, fn _, x -> adjust(cal, Date.add(x, -1), :preceding) end)

  @doc "Add months, end-of-month aware (Jan 31 + 1M = Feb 28/29)."
  def add_months(%Date{year: y, month: m, day: d}, k) do
    t = y * 12 + (m - 1) + k
    {yy, mm} = {div(t, 12), rem(t, 12) + 1}
    Date.new!(yy, mm, min(d, Calendar.ISO.days_in_month(yy, mm)))
  end

  # ------------------------------------------------------------- day counts

  @doc "Year fraction between two dates in a day-count convention (the calendar only matters for `:bus252`)."
  def year_fraction(basis, d1, d2, cal \\ :anbima)
  def year_fraction(:bus252, d1, d2, cal), do: business_days(cal, d1, d2) / 252
  def year_fraction(:act360, d1, d2, _), do: Date.diff(d2, d1) / 360
  def year_fraction(:act365f, d1, d2, _), do: Date.diff(d2, d1) / 365

  def year_fraction(:thirty360, d1, d2, _) do
    # US bond basis (ISDA 2006 4.16(f)): D1 = 31 → 30; D2 = 31 and D1 ≥ 30 → 30
    dd1 = if d1.day == 31, do: 30, else: d1.day
    dd2 = if d2.day == 31 and dd1 >= 30, do: 30, else: d2.day
    (360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (dd2 - dd1)) / 360
  end

  def year_fraction(:thirty_e360, d1, d2, _) do
    dd1 = min(d1.day, 30); dd2 = min(d2.day, 30)
    (360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (dd2 - dd1)) / 360
  end

  def year_fraction(:act_act_isda, d1, d2, _) do
    cond do
      Date.compare(d2, d1) == :lt -> -year_fraction(:act_act_isda, d2, d1, nil)
      d1.year == d2.year -> Date.diff(d2, d1) / days_in_year(d1.year)
      true ->
        first = Date.diff(Date.new!(d1.year + 1, 1, 1), d1) / days_in_year(d1.year)
        last = Date.diff(d2, Date.new!(d2.year, 1, 1)) / days_in_year(d2.year)
        first + (d2.year - d1.year - 1) + last
    end
  end

  defp days_in_year(y), do: if(Calendar.ISO.leap_year?(y), do: 366, else: 365)

  @bases [:bus252, :act360, :act365f, :thirty360, :thirty_e360, :act_act_isda]
  def bases, do: @bases

  @doc "Parse a calendar or basis name (`anbima`, `b3`, `nyse`, `target`; `du252`, `act/360`, `30/360`, …)."
  def parse_calendar(s) do
    case s |> to_string() |> String.downcase() |> String.trim() do
      x when x in ["anbima", "b3", "brazil", "brasil", "br"] -> {:ok, :anbima}
      x when x in ["nyse", "us", "eua"] -> {:ok, :nyse}
      x when x in ["target", "target2", "eur", "euro"] -> {:ok, :target}
      x when x in ["weekends", "fins de semana", "none"] -> {:ok, :weekends}
      x -> {:error, "calendar #{inspect(x)}: anbima (= b3), nyse, target or weekends"}
    end
  end

  def parse_basis(s) do
    case s |> to_string() |> String.downcase() |> String.replace(~r/\s+/, "") do
      x when x in ["bus252", "du252", "du/252", "business252", "252"] -> {:ok, :bus252}
      x when x in ["act360", "act/360", "actual/360"] -> {:ok, :act360}
      x when x in ["act365", "act365f", "act/365", "act/365f", "actual/365(fixed)"] -> {:ok, :act365f}
      x when x in ["30/360", "thirty360", "30360", "bondbasis"] -> {:ok, :thirty360}
      x when x in ["30e/360", "thirty_e360", "eurobond"] -> {:ok, :thirty_e360}
      x when x in ["act/act", "actact", "act_act_isda", "act/actisda"] -> {:ok, :act_act_isda}
      x -> {:error, "basis #{inspect(x)}: du252, act/360, act/365f, 30/360, 30e/360 or act/act"}
    end
  end

  @doc """
  The month codes of futures (F G H J K M N Q U V X Z) and the B3 rule for
  the DI1 maturity: the first business day of the contract month.
  """
  @codes %{"F" => 1, "G" => 2, "H" => 3, "J" => 4, "K" => 5, "M" => 6, "N" => 7, "Q" => 8, "U" => 9, "V" => 10, "X" => 11, "Z" => 12}
  def di1_maturity(code, ref_year \\ nil) do
    case Regex.run(~r/^(?:DI1)?([FGHJKMNQUVXZ])(\d{2})$/i, String.trim(code)) do
      [_, m, yy] ->
        y = 2000 + String.to_integer(yy)
        y = if ref_year && y < ref_year - 50, do: y + 100, else: y
        {:ok, adjust(:anbima, Date.new!(y, @codes[String.upcase(m)], 1), :following)}
      _ -> {:error, "a DI1 code is a month letter and two digits (F26 = January 2026)"}
    end
  end
end
