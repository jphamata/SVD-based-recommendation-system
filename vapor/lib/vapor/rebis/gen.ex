defmodule Vapor.Rebis.Gen do
  @moduledoc """
  Circuits with a known meaning, written as netlist text (so they pass
  through the same parser as anyone's): two structurally different adders
  that must be equivalent, a multiplier, and the same adder with a
  **trojan** — an output bit flipped when the inputs hit a rare trigger,
  the case that simulation cannot find and a proof cannot miss.
  """

  @doc "Ripple-carry adder of two `n`-bit words: outputs `s0…s(n−1)`, `cout`."
  def ripple(n, opts \\ []) do
    body =
      for i <- 0..(n - 1) do
        c = if i == 0, do: "0", else: "c#{i}"
        "s#{i}#{suffix(opts, i)} = a#{i} ^ b#{i} ^ #{c}\nc#{i + 1} = maj(a#{i}, b#{i}, #{c})"
      end

    header(n) <> Enum.join(body, "\n") <> "\ncout = c#{n}\n" <> trojan(n, opts)
  end

  @doc "Kogge–Stone parallel-prefix adder (log-depth carries): the same function, another structure."
  def kogge_stone(n) do
    base = for i <- 0..(n - 1), do: "g0_#{i} = a#{i} & b#{i}\np0_#{i} = a#{i} ^ b#{i}"
    levels = Stream.iterate(1, &(&1 * 2)) |> Enum.take_while(&(&1 < n))

    {lines, last} =
      levels
      |> Enum.with_index(1)
      |> Enum.reduce({[], 0}, fn {d, l}, {acc, _} ->
        ls =
          for i <- 0..(n - 1) do
            if i >= d do
              "g#{l}_#{i} = g#{l - 1}_#{i} | (p#{l - 1}_#{i} & g#{l - 1}_#{i - d})\np#{l}_#{i} = p#{l - 1}_#{i} & p#{l - 1}_#{i - d}"
            else
              "g#{l}_#{i} = g#{l - 1}_#{i} | 0\np#{l}_#{i} = p#{l - 1}_#{i} & 1"
            end
          end

        {acc ++ ls, l}
      end)

    sums = for i <- 0..(n - 1), do: if(i == 0, do: "s0 = p0_0", else: "s#{i} = p0_#{i} ^ g#{last}_#{i - 1}")
    header(n) <> Enum.join(base ++ lines ++ sums, "\n") <> "\ncout = g#{last}_#{n - 1}\n"
  end

  @doc "Shift-and-add multiplier `n×n → 2n` bits: outputs `m0…m(2n−1)`, inputs `a*`, `b*`."
  def multiplier(n) do
    ins = "input " <> Enum.map_join(0..(n - 1), " ", &"a#{&1}") <> " " <> Enum.map_join(0..(n - 1), " ", &"b#{&1}") <> "\n"
    outs = "output " <> Enum.map_join(0..(2 * n - 1), " ", &"m#{&1}") <> "\n"
    pp = for i <- 0..(n - 1), j <- 0..(n - 1), do: "pp#{i}_#{j} = a#{j} & b#{i}"

    # the accumulator is a list of wire names, bit 0 first; row i adds pp_i shifted by i
    acc0 = for j <- 0..(n - 1), do: "pp0_#{j}"

    {lines, acc} =
      Enum.reduce(1..(n - 1)//1, {[], acc0}, fn i, {lines, acc} ->
        width = max(length(acc), i + n)
        addend = fn j -> if j >= i and j - i < n, do: "pp#{i}_#{j - i}", else: "0" end
        bit = fn j -> Enum.at(acc, j, "0") end

        {ls, sums, carry} =
          Enum.reduce(0..(width - 1), {[], [], "0"}, fn j, {ls, sums, c} ->
            s = "r#{i}_#{j}"
            k = "k#{i}_#{j + 1}"
            {ls ++ ["#{s} = #{bit.(j)} ^ #{addend.(j)} ^ #{c}", "#{k} = maj(#{bit.(j)}, #{addend.(j)}, #{c})"], sums ++ [s], k}
          end)

        {lines ++ ls, sums ++ [carry]}
      end)

    outs_l = for k <- 0..(2 * n - 1), do: "m#{k} = " <> Enum.at(acc, k, "0")
    ins <> outs <> Enum.join(pp ++ lines ++ outs_l, "\n") <> "\n"
  end

  defp header(n) do
    "input " <> Enum.map_join(0..(n - 1), " ", &"a#{&1}") <> " " <> Enum.map_join(0..(n - 1), " ", &"b#{&1}") <> "\n" <>
      "output " <> Enum.map_join(0..(n - 1), " ", &"s#{&1}") <> " cout\n"
  end

  # with a trojan, s0 is computed as s0_clean and then flipped on the trigger
  defp suffix(opts, 0), do: if(Keyword.has_key?(opts, :trojan), do: "_clean", else: "")
  defp suffix(_, _), do: ""

  defp trojan(n, opts) do
    case Keyword.get(opts, :trojan) do
      nil ->
        ""

      trigger when is_integer(trigger) ->
        # the trigger: a = trigger exactly (all n bits of a), so 1 pattern in 2ⁿ of a
        lits = for i <- 0..(n - 1), do: if((Bitwise.>>>(trigger, i) |> Bitwise.band(1)) == 1, do: "a#{i}", else: "~a#{i}")
        "trig = " <> Enum.join(lits, " & ") <> "\ns0 = s0_clean ^ trig\n"
    end
  end
end
