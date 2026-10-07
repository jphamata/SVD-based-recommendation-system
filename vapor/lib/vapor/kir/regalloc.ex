defmodule Vapor.KIR.RegAlloc do
  @moduledoc """
  Linear-scan register allocation with *register groups* (RVV LMUL, and its
  generalisation to AVX2/NEON unrolled strips), and **no spilling**.

  Each interval asks for a block of `size ∈ {1, 2, 4, 8}` consecutive
  physical registers starting at a multiple of `size` (RVV 1.0 §3.4.2
  register-group alignment; imposed uniformly so a single checker covers
  every ISA). Allocation is first-fit over a preference order — caller-saved
  registers first, then blocks whose buddy half is already occupied (keeping
  large aligned blocks whole), then the ISA's order.

  There is deliberately no spill path: when a block cannot be found the
  allocator returns the pressure point, and the *cut sweep* either lowers the
  group factor or closes the fused region there. Every accepted allocation is
  then re-validated by `Vapor.Extracted.check_alloc/2`, a checker proven sound
  in Lean (`proofs/Vapor/RegAlloc.lean`: an accepted assignment never maps two
  simultaneously-live values to overlapping registers) and extracted to
  Elixir — so "spill-free and clobber-free" is established per region by a
  verified checker, not assumed of the heuristic.
  """

  @type file :: %{count: pos_integer, order: [non_neg_integer], callee_saved: [non_neg_integer]}
  @type result :: %{assign: %{tuple => non_neg_integer}, callee_used: %{atom => [non_neg_integer]}}

  @spec allocate([map], %{atom => file}, %{tuple => non_neg_integer}) ::
          {:ok, result} | {:error, map}
  def allocate(intervals, files, pins \\ %{}) do
    intervals
    |> Enum.group_by(fn %{reg: {:vr, _, class, _}} -> class end)
    |> Enum.reduce_while({:ok, %{assign: %{}, callee_used: %{}}}, fn {class, ivs}, {:ok, acc} ->
      file = Map.fetch!(files, class)

      case scan(ivs, file, pins) do
        {:ok, assign} ->
          used =
            for {{:vr, _, _, sz}, p} <- assign, i <- p..(p + sz - 1), i in file.callee_saved,
                uniq: true, do: i

          {:cont,
           {:ok,
            %{acc | assign: Map.merge(acc.assign, assign),
                    callee_used: Map.put(acc.callee_used, class, Enum.sort(used))}}}

        {:error, info} ->
          {:halt, {:error, Map.put(info, :class, class)}}
      end
    end)
  end

  defp scan(ivs, file, pins) do
    rank = file.order |> Enum.with_index() |> Map.new()
    callee = MapSet.new(file.callee_saved)
    pinned = for iv <- ivs, Map.has_key?(pins, iv.reg), do: {iv, Map.fetch!(pins, iv.reg)}

    Enum.sort_by(ivs, &{&1.start, &1.stop})
    |> Enum.reduce_while({:ok, %{}, []}, fn iv, {:ok, assign, active} ->
      active = Enum.filter(active, fn {a, _p} -> a.stop > iv.start end)
      {:vr, _, _, size} = iv.reg
      busy = MapSet.new(for {a, p} <- active, {:vr, _, _, s} = a.reg, i <- p..(p + s - 1), do: i)

      blocked =
        for {piv, p} <- pinned, piv.reg != iv.reg, piv.start < iv.stop, iv.start < piv.stop,
            {:vr, _, _, s} = piv.reg, i <- p..(p + s - 1), into: MapSet.new(), do: i

      free? = fn s -> Enum.all?(s..(s + size - 1), &(Map.has_key?(rank, &1) and not MapSet.member?(busy, &1) and not MapSet.member?(blocked, &1))) end

      choice =
        case Map.fetch(pins, iv.reg) do
          {:ok, p} ->
            if free?.(p), do: p, else: nil

          :error ->
            file.order
            |> Enum.filter(&(rem(&1, size) == 0 and free?.(&1)))
            |> Enum.min_by(
              fn s ->
                span = s..(s + size - 1)
                callee_n = Enum.count(span, &MapSet.member?(callee, &1))
                buddy = s - rem(s, 2 * size)
                other = if buddy == s, do: s + size, else: buddy
                buddy_busy = Enum.any?(other..(other + size - 1), &MapSet.member?(busy, &1))
                {callee_n, if(buddy_busy, do: 0, else: 1), Map.fetch!(rank, s)}
              end,
              fn -> nil end
            )
        end

      if choice == nil do
        {:halt,
         {:error,
          %{at: iv.start, need: size, reg: iv.reg,
            live: length(active), live_regs: MapSet.size(busy)}}}
      else
        {:cont, {:ok, Map.put(assign, iv.reg, choice), [{iv, choice} | active]}}
      end
    end)
    |> case do
      {:ok, assign, _} -> {:ok, assign}
      err -> err
    end
  end

  @doc """
  Checker input: per register file, the list of `{start, stop, phys, size}`
  quadruples (half-open live ranges) — exactly the representation of
  `Vapor.RegAlloc.checkAlloc` in Lean.
  """
  def placements(intervals, assign) do
    intervals
    |> Enum.group_by(fn %{reg: {:vr, _, c, _}} -> c end)
    |> Map.new(fn {c, ivs} ->
      {c, for(%{reg: {:vr, _, _, s} = r, start: a, stop: b} <- ivs, do: {a, b, Map.fetch!(assign, r), s})}
    end)
  end
end
