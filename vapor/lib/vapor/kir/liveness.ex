defmodule Vapor.KIR.Liveness do
  @moduledoc """
  Live intervals over selected machine code (the input of linear scan).

  Machine instructions are `{op, defs, uses, attrs}`; registers are
  `{:vr, id, class, size}` (virtual) or `{:pr, class, index}` (precoloured).
  `attrs` may carry `label: l`, `br: l` (with `uncond: true` for jumps),
  `ret: true`, and `ec: true` (early clobber: the definition is live at the
  *use* point, so it can never share a register with an operand — required
  by RVV widening/extension instructions and multi-register expansions).

  Instruction `i` has a use point `2i` and a def point `2i + 1`. Liveness is
  the standard backward dataflow fixpoint over the CFG (so values carried
  around loop back-edges are live across the whole loop); an interval is the
  hull `[start, stop)` of every point where the register is live.
  """

  @type interval :: %{reg: tuple, start: non_neg_integer, stop: pos_integer}

  @spec intervals([tuple]) :: [interval]
  def intervals(code) do
    code = List.to_tuple(code)
    n = tuple_size(code)
    blocks = blocks(code, n)
    succ = successors(blocks, code)

    uses_defs =
      Map.new(blocks, fn {s, e} ->
        {ub, db} =
          Enum.reduce(s..e, {MapSet.new(), MapSet.new()}, fn i, {u, d} ->
            {_op, defs, uses, _} = elem(code, i)
            u = MapSet.union(u, MapSet.difference(vset(uses), d))
            {u, MapSet.union(d, vset(defs))}
          end)

        {s, {ub, db}}
      end)

    live_out = fixpoint(blocks, succ, uses_defs, Map.new(blocks, fn {s, _} -> {s, MapSet.new()} end))

    Enum.reduce(blocks, %{}, fn {s, e}, acc ->
      out = Map.fetch!(live_out, s)
      acc = Enum.reduce(out, acc, &extend(&2, &1, 2 * e + 1))

      {acc, live} =
        Enum.reduce(e..s//-1, {acc, out}, fn i, {acc, live} ->
          {_op, defs, uses, attrs} = elem(code, i)
          dpt = if attrs[:ec] == true, do: 2 * i, else: 2 * i + 1
          acc = Enum.reduce(vlist(defs), acc, &extend(&2, &1, dpt))
          acc = Enum.reduce(vlist(uses), acc, &extend(&2, &1, 2 * i))
          live = live |> MapSet.difference(vset(defs)) |> MapSet.union(vset(uses))
          {acc, live}
        end)

      Enum.reduce(live, acc, &extend(&2, &1, 2 * s))
    end)
    |> Enum.map(fn {reg, {a, b}} -> %{reg: reg, start: a, stop: b + 1} end)
    |> Enum.sort_by(&{&1.start, &1.stop})
  end

  defp extend(acc, reg, p), do: Map.update(acc, reg, {p, p}, fn {a, b} -> {min(a, p), max(b, p)} end)

  defp vlist(regs), do: Enum.filter(regs, &match?({:vr, _, _, _}, &1))
  defp vset(regs), do: MapSet.new(vlist(regs))

  # basic blocks: start at labels and after control transfers
  defp blocks(code, n) do
    starts =
      Enum.reduce(0..(n - 1), MapSet.new([0]), fn i, st ->
        {_, _, _, a} = elem(code, i)
        st = if a[:label] != nil, do: MapSet.put(st, i), else: st
        if (a[:br] != nil or a[:ret] == true) and i + 1 < n, do: MapSet.put(st, i + 1), else: st
      end)
      |> Enum.sort()

    Enum.zip(starts, tl(starts) ++ [n]) |> Enum.map(fn {s, nx} -> {s, nx - 1} end)
  end

  defp successors(blocks, code) do
    label_at =
      for {s, _} <- blocks, {_, _, _, a} = elem(code, s), a[:label] != nil, into: %{},
        do: {a[:label], s}

    starts = MapSet.new(Enum.map(blocks, &elem(&1, 0)))

    Map.new(blocks, fn {s, e} ->
      {_, _, _, a} = elem(code, e)
      fall = if MapSet.member?(starts, e + 1) and a[:uncond] != true and a[:ret] != true, do: [e + 1], else: []
      br = if a[:br] != nil, do: [Map.fetch!(label_at, a[:br])], else: []
      {s, fall ++ br}
    end)
  end

  defp fixpoint(blocks, succ, ud, live_out) do
    {changed, lo} =
      Enum.reduce(Enum.reverse(blocks), {false, live_out}, fn {s, _e}, {ch, lo} ->
        out =
          succ
          |> Map.fetch!(s)
          |> Enum.reduce(MapSet.new(), fn t, acc -> MapSet.union(acc, live_in(t, ud, lo)) end)

        if MapSet.equal?(out, Map.fetch!(lo, s)), do: {ch, lo}, else: {true, Map.put(lo, s, out)}
      end)

    if changed, do: fixpoint(blocks, succ, ud, lo), else: lo
  end

  defp live_in(s, ud, lo) do
    {u, d} = Map.fetch!(ud, s)
    MapSet.union(u, MapSet.difference(Map.fetch!(lo, s), d))
  end
end
