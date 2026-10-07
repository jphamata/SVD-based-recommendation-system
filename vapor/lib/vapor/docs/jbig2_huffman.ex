defmodule Vapor.Docs.JBIG2.Huffman do
  @moduledoc """
  JBIG2's Huffman coding (T.88 Annex B): the fifteen standard tables, the
  code tables a stream may carry (segment type 53, B.2), the assignment of
  prefix codes from prefix lengths (B.3) and the decoding of a value (B.4)
  from a bit reader that reads most significant bit first.

  A table is a list of lines `{kind, preflen, rangelen, rangelow}` with
  `kind` one of `:normal | :lower | :upper | :oob`; a line of prefix length
  0 has no code. A decoded table maps codes to lines.
  """
  import Bitwise

  # {PREFLEN, RANGELEN, RANGELOW} lines, then the lower and upper range
  # lines, then OOB's prefix length when the table has one (T.88 B.5)
  @standard %{
    1 => {[{1, 4, 0}, {2, 8, 16}, {3, 16, 272}], {0, -1}, {3, 65_808}, nil},
    2 => {[{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 3, 3}, {5, 6, 11}], {0, -1}, {6, 75}, 6},
    3 => {[{8, 8, -256}, {1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 3, 3}, {5, 6, 11}], {8, -257}, {7, 75}, 6},
    4 => {[{1, 0, 1}, {2, 0, 2}, {3, 0, 3}, {4, 3, 4}, {5, 6, 12}], {0, -1}, {5, 76}, nil},
    5 => {[{7, 8, -255}, {1, 0, 1}, {2, 0, 2}, {3, 0, 3}, {4, 3, 4}, {5, 6, 12}], {7, -256}, {6, 76}, nil},
    6 => {[{5, 10, -2048}, {4, 9, -1024}, {4, 8, -512}, {4, 7, -256}, {5, 6, -128}, {5, 5, -64}, {4, 5, -32}, {2, 7, 0},
           {3, 7, 128}, {3, 8, 256}, {4, 9, 512}, {4, 10, 1024}], {6, -2049}, {6, 2048}, nil},
    7 => {[{4, 9, -1024}, {3, 8, -512}, {4, 7, -256}, {5, 6, -128}, {5, 5, -64}, {4, 5, -32}, {4, 5, 0}, {5, 5, 32},
           {5, 6, 64}, {4, 7, 128}, {3, 8, 256}, {3, 9, 512}, {3, 10, 1024}], {5, -1025}, {5, 2048}, nil},
    8 => {[{8, 3, -15}, {9, 1, -7}, {8, 1, -5}, {9, 0, -3}, {7, 0, -2}, {4, 0, -1}, {2, 1, 0}, {5, 0, 2}, {6, 0, 3},
           {3, 4, 4}, {6, 1, 20}, {4, 4, 22}, {4, 5, 38}, {5, 6, 70}, {5, 7, 134}, {6, 7, 262}, {7, 8, 390}, {6, 10, 646}],
          {9, -16}, {9, 1670}, 2},
    9 => {[{8, 4, -31}, {9, 2, -15}, {8, 2, -11}, {9, 1, -7}, {7, 1, -5}, {4, 1, -3}, {3, 1, -1}, {3, 1, 1}, {5, 1, 3},
           {6, 1, 5}, {3, 5, 7}, {6, 2, 39}, {4, 5, 43}, {4, 6, 75}, {5, 7, 139}, {5, 8, 267}, {6, 8, 523}, {7, 9, 779},
           {6, 11, 1291}], {9, -32}, {9, 3339}, 2},
    10 => {[{7, 4, -21}, {8, 0, -5}, {7, 0, -4}, {5, 0, -3}, {2, 2, -2}, {5, 0, 2}, {6, 0, 3}, {7, 0, 4}, {8, 0, 5},
            {2, 6, 6}, {5, 5, 70}, {6, 5, 102}, {6, 6, 134}, {6, 7, 198}, {6, 8, 326}, {6, 9, 582}, {6, 10, 1094},
            {7, 11, 2118}], {8, -22}, {8, 4166}, 2},
    11 => {[{1, 0, 1}, {2, 1, 2}, {4, 0, 4}, {4, 1, 5}, {5, 1, 7}, {5, 2, 9}, {6, 2, 13}, {7, 2, 17}, {7, 3, 21},
            {7, 4, 29}, {7, 5, 45}, {7, 6, 77}], {0, 0}, {7, 141}, nil},
    12 => {[{1, 0, 1}, {2, 0, 2}, {3, 1, 3}, {5, 0, 5}, {5, 1, 6}, {6, 1, 8}, {7, 0, 10}, {7, 1, 11}, {7, 2, 13},
            {7, 3, 17}, {7, 4, 25}, {8, 5, 41}], {0, 0}, {8, 73}, nil},
    13 => {[{1, 0, 1}, {3, 0, 2}, {4, 0, 3}, {5, 0, 4}, {4, 1, 5}, {3, 3, 7}, {6, 1, 15}, {6, 2, 17}, {6, 3, 21},
            {6, 4, 29}, {6, 5, 45}, {7, 6, 77}], {0, 0}, {7, 141}, nil},
    14 => {[{3, 0, -2}, {3, 0, -1}, {1, 0, 0}, {3, 0, 1}, {3, 0, 2}], {0, 0}, {0, 3}, nil},
    15 => {[{7, 4, -24}, {6, 2, -8}, {5, 1, -4}, {4, 0, -2}, {3, 0, -1}, {1, 0, 0}, {3, 0, 1}, {4, 0, 2}, {5, 1, 3},
            {6, 2, 5}, {7, 4, 9}], {7, -25}, {7, 25}, nil}
  }

  @doc "Standard table B.`n` (1–15), built."
  def standard(n) when is_map_key(@standard, n), do: n |> lines() |> build()

  @doc "The lines of standard table B.`n`."
  def lines(n) do
    {normal, {lp, llow}, {up, ulow}, oob} = Map.fetch!(@standard, n)
    Enum.map(normal, fn {p, r, low} -> {:normal, p, r, low} end) ++
      [{:lower, lp, 32, llow}, {:upper, up, 32, ulow}] ++ if(oob, do: [{:oob, oob, 0, 0}], else: [])
  end

  @doc """
  Assign prefix codes (B.3) and index them by `{length, code}`. Returns
  `%{codes: %{{len, code} => line}, maxlen}`.
  """
  def build(lines) do
    maxlen = lines |> Enum.map(&elem(&1, 1)) |> Enum.max(fn -> 0 end)
    count = lines |> Enum.map(&elem(&1, 1)) |> Enum.frequencies() |> Map.put(0, 0)

    # FIRSTCODE[len] = (FIRSTCODE[len − 1] + LENCOUNT[len − 1]) · 2, LENCOUNT[0] = 0
    {codes, _} =
      Enum.reduce(1..maxlen//1, {%{}, 0}, fn len, {acc, prev_first} ->
        first = (prev_first + Map.get(count, len - 1, 0)) <<< 1

        {acc, _} =
          Enum.reduce(lines, {acc, first}, fn line, {a, code} ->
            if elem(line, 1) == len, do: {Map.put(a, {len, code}, line), code + 1}, else: {a, code}
          end)

        {acc, first}
      end)

    %{codes: codes, maxlen: maxlen}
  end

  # ------------------------------------------------------------ the reader

  @doc "A bit reader over a binary (MSB first)."
  def reader(bin), do: %{bin: bin, pos: 0}

  @doc "Read `n` bits as an unsigned integer: `{value, reader}` (zeros past the end, as T.88's decoders do)."
  def bits(r, 0), do: {0, r}

  def bits(%{bin: bin, pos: pos} = r, n) do
    total = bit_size(bin)

    v =
      if pos + n <= total do
        <<_::size(pos), v::size(n), _::bitstring>> = bin
        v
      else
        avail = max(total - pos, 0)
        skip = min(pos, total)
        <<_::size(skip), head::size(avail), _::bitstring>> = bin
        head <<< (n - avail)
      end

    {v, %{r | pos: pos + n}}
  end

  @doc "Skip to the next byte boundary."
  def align(%{pos: pos} = r), do: %{r | pos: (pos + 7) &&& bnot(7)}

  @doc "The byte offset of a byte-aligned reader."
  def byte_pos(%{pos: pos}), do: div(pos + 7, 8)

  @doc "Advance a byte-aligned reader by `n` bytes."
  def skip_bytes(r, n), do: %{r | pos: byte_pos(r) * 8 + 8 * n}

  @doc "Decode one value (B.4): `{integer | :oob, reader}`."
  def decode(%{codes: codes, maxlen: maxlen}, r), do: prefix(codes, maxlen, r, 0, 0)

  defp prefix(codes, maxlen, r, len, code) do
    if len >= maxlen, do: raise(ArgumentError, "no Huffman code of the table matches the stream")
    {b, r} = bits(r, 1)
    code = code <<< 1 ||| b
    len = len + 1

    case Map.fetch(codes, {len, code}) do
      {:ok, {:oob, _, _, _}} -> {:oob, r}
      {:ok, {:normal, _, rl, low}} -> {v, r} = bits(r, rl); {low + v, r}
      {:ok, {:lower, _, rl, low}} -> {v, r} = bits(r, rl); {low - v, r}
      {:ok, {:upper, _, rl, low}} -> {v, r} = bits(r, rl); {low + v, r}
      :error -> prefix(codes, maxlen, r, len, code)
    end
  end

  # ------------------------------------------------------- code tables (B.2)

  @doc "Read a code table segment (type 53): `{:ok, table}` or `{:error, why}`."
  def table_segment(<<flags, low::signed-32, high::signed-32, rest::binary>>) do
    htoob = flags &&& 1
    htps = (flags >>> 1 &&& 7) + 1
    htrs = (flags >>> 4 &&& 7) + 1

    if low >= high do
      {:error, "a code table with HTLOW ≥ HTHIGH"}
    else
      {lines, r} = table_lines(reader(rest), htps, htrs, low, high, [])
      {lp, r} = bits(r, htps)
      {up, r} = bits(r, htps)
      {oob, _r} = if htoob == 1, do: bits(r, htps), else: {nil, r}
      lines = lines ++ [{:lower, lp, 32, low - 1}, {:upper, up, 32, high}] ++ if(oob, do: [{:oob, oob, 0, 0}], else: [])
      {:ok, build(lines)}
    end
  end

  def table_segment(_), do: {:error, "a truncated code table"}

  defp table_lines(r, _ps, _rs, cur, high, acc) when cur >= high, do: {Enum.reverse(acc), r}
  defp table_lines(_r, _ps, _rs, _cur, _high, acc) when length(acc) > 65_536, do: raise(ArgumentError, "a code table with too many lines")

  defp table_lines(r, ps, rs, cur, high, acc) do
    {p, r} = bits(r, ps)
    {rl, r} = bits(r, rs)
    table_lines(r, ps, rs, cur + (1 <<< rl), high, [{:normal, p, rl, cur} | acc])
  end

  # ---------------------------------------------- symbol ID table (7.4.3.1.7)

  @doc """
  Read the symbol ID Huffman table of a text region: 35 run-code lengths
  (4 bits each), then `nsyms` code lengths coded with them (32 repeats the
  previous length 3–6 times, 33 writes 3–10 zeros, 34 writes 11–138 zeros),
  then the table built from those lengths (value = the symbol's index).
  `{table, reader}` — the reader is left at the next byte boundary.
  """
  def symbol_id_table(r, nsyms) do
    {runlens, r} = Enum.map_reduce(0..34, r, fn _, r -> bits(r, 4) end)
    runtab = runlens |> Enum.with_index() |> Enum.map(fn {p, i} -> {:normal, p, 0, i} end) |> build()

    {lens, r} = id_lengths(runtab, r, nsyms, [], 0)
    tab = lens |> Enum.with_index() |> Enum.map(fn {p, i} -> {:normal, p, 0, i} end) |> build()
    {tab, align(r)}
  end

  defp id_lengths(_rt, r, n, acc, count) when count >= n, do: {acc |> Enum.reverse() |> Enum.take(n), r}

  defp id_lengths(rt, r, n, acc, count) do
    {code, r} = decode(rt, r)

    {more, r} =
      cond do
        code in 0..31 -> {[code], r}
        code == 32 -> {k, r} = bits(r, 2); {List.duplicate(List.first(acc) || raise(ArgumentError, "run code 32 with no previous length"), k + 3), r}
        code == 33 -> {k, r} = bits(r, 3); {List.duplicate(0, k + 3), r}
        code == 34 -> {k, r} = bits(r, 7); {List.duplicate(0, k + 11), r}
        true -> raise ArgumentError, "a symbol ID run code out of range"
      end

    id_lengths(rt, r, n, Enum.reverse(more) ++ acc, count + length(more))
  end
end
