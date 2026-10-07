defmodule Vapor.Arbiter do
  @moduledoc """
  Section 5 — the mechanical crest arbiter and the analytic benchmark.

  Work is *counted* from the schedule and the emitted code — FLOPs `F`,
  bytes moved `Q`, and dynamic instructions `I` (the static instruction count
  of each hot loop body, as emitted, times its trip count) — hardware is
  *declared* (peak `π`, bandwidth `β`, sustained issue rate `ι`, per-call and
  per-dispatch overheads: datasheet figures in a profile), and time is
  *predicted* by a three-roof roofline

      t = Σ max(F/π, Q/β, I/ι) + overheads

  The issue roof matters: a kernel that must decode 4-bit weights is often
  bound by instruction issue long before bandwidth, and a two-roof model
  mispredicts it several-fold. No warm-up, no profiling: the same inputs give
  the same decision on every node (Theorem 5.1).

  Routing picks the substrate with the smallest predicted time. This
  refines the specification's Eq. 3, which additionally required the
  intensity `F/Q` to reach the accelerator's crest `φ = π/β`: once the
  prediction contains every roof and every overhead, that guard can only
  ever select the *slower* substrate, so it is reported (as `crest`) but no
  longer decides.

  Separation of concerns: counted work is a property of the program and its
  code, so it goes into the (portable, co-signable) certificate; profiles
  are properties of a deployment, so the decision is taken at run time on
  each node from its own declared profiles.
  """
  alias Vapor.Compiled

  defmodule Profile do
    @moduledoc "Declared hardware envelope of one substrate."
    @enforce_keys [:name, :peak_flops, :bandwidth]
    defstruct [:name, :peak_flops, :bandwidth, issue_rate: nil, vlen: 128,
               call_overhead_s: 0.0, dispatch_overhead_s: 0.0]
  end

  @doc """
  Default declared profiles: one x86-64 AVX2 core (worker is single-threaded:
  2 FMA ports × 8 lanes × 2 FLOP × 3 GHz; ~20 GB/s streaming) and a
  discrete-class Vulkan device. Operators override these with datasheet
  values for their hardware; they are *inputs*, never measurements.
  """
  def default_profiles do
    %{
      native: %Profile{name: :native, peak_flops: 96.0e9, bandwidth: 20.0e9, issue_rate: 9.0e9,
                       call_overhead_s: 30.0e-6},
      fabric: %Profile{name: :fabric, peak_flops: 20.0e12, bandwidth: 500.0e9,
                       call_overhead_s: 200.0e-6, dispatch_overhead_s: 8.0e-6}
    }
  end

  @doc """
  Counted work of every scheduled call at the given extents: FLOPs, bytes,
  and hot-loop dynamic instructions for the code of `isa` (default: host).
  """
  def work(%Compiled{} = c, dims, isa \\ Vapor.Runtime.Substrates.host_isa(), vlen \\ 128) do
    kernels(c, dims, isa, vlen) ++ state_feedback(c, dims)
  end

  # recurrent programs copy each fed-back output into its state input per step
  defp state_feedback(%Compiled{state: []}, _dims), do: []

  defp state_feedback(c, dims) do
    bytes = c.state |> Enum.map(fn {_in, out} -> 2 * Compiled.nbytes(c, out, dims) end) |> Enum.sum()
    [%{kernel: :state_feedback, flops: 0, bytes: bytes, instructions: 0}]
  end

  defp kernels(c, dims, isa, vlen) do
    Enum.map(c.schedule, fn %{kernel: key, args: args} ->
      resolved = Enum.map(args, &Compiled.resolve_arg(c, &1, dims))
      imms = for {:imm, v} <- resolved, do: v
      slot_bytes = for({:slot, id} <- resolved, do: Compiled.nbytes(c, id, dims)) |> Enum.sum()

      flops =
        case {key, imms} do
          {{:ew, spec}, [r, c]} -> r * c * Enum.sum(Enum.map(spec.ops, fn {_, op, _} -> ew_flops(op) end))
          {{:reduce, _}, [r, c]} -> r * c
          {g, [n, k, b, _ldy]} when g in [:gemv_f32, :gemv_bf16] -> 2 * n * k * b
          # row-predicated: certified at its worst case, every row active
          {{:gemv_masked, _}, [n, k, b, _ldy]} -> 2 * n * k * b
          {{:gemv_grouped, _}, [n, k, b, g | _]} -> 2 * n * k * b * g
          {:rope, [t, h, half, _s]} -> 6 * t * h * half
          # attention at its maximal context (L = S): scores, softmax, weighted sum
          {{:attention, _}, [t, hkv, g, s, dh | _]} -> t * hkv * g * s * (4 * dh + 40)
          {{:attention_paged, _, _}, [t, hkv, g, s, dh | _]} -> t * hkv * g * s * (4 * dh + 40)
          {:sb_sums, [nsub]} -> 32 * nsub
          # predicated: certified at its worst case, every row active
          {g, [rows, nsb, b, _ldy]} when g in [:gemv_sb4, :gemv_sb4_masked] -> b * (rows * nsb * 256 * 3 + rows * nsb * 8 * 6)
          {:gemm_i8, [m, n, k]} -> 2 * m * n * k
          {:sample, [b, v]} -> b * v * 30
          {:transpose, _} -> 0
          _ -> 0
        end

      code = get_in(c.code, [isa, :codes, key])
      %{kernel: key, flops: flops, bytes: slot_bytes, instructions: instructions(code, key, imms, vlen)}
    end)
  end

  # hot-loop body counts × trip counts (a lower bound on dynamic instructions)
  defp instructions(nil, _key, _imms, _vlen), do: 0

  defp instructions(code, key, imms, vlen) do
    strip = code.loops |> Enum.find_value(0, fn {{:l, {:strip, _}}, n} -> n; _ -> nil end)
    lanes = fn bytes_per_elem -> lanes(code.isa, code.g, bytes_per_elem, vlen) end

    case {key, imms} do
      {{:ew, _}, [r, c]} -> r * div(c, lanes.(4)) * strip
      {{:reduce, _}, [r, c]} -> r * div(c, 16) * Map.get(code.loops, {:l, :chunk}, 0)
      {g, [n, k, b, _ldy]} when g in [:gemv_f32, :gemv_bf16] or (is_tuple(g) and elem(g, 0) == :gemv_masked) ->
        rr = Map.get(code.variant, :rows_per_iter, 1)
        blk = Map.get(code.loops, {:l, {:blk, :k}}, 0)
        one = Map.get(code.loops, {:l, {:one, :k}}, 0)
        b * (div(n, rr) * blk + rem(n, rr) * one) * div(k, 16)
      {:gather_row, [t, _v, d]} -> t * div(d, lanes.(4)) * strip
      {:gather_row_bf16, [t, _v, d]} -> t * div(d, 16) * Map.get(code.loops, {:l, :chunk}, 0)
      {:rope, [t, h, half, _s]} -> t * h * div(half, lanes.(4)) * strip
      {{:kv_write, _}, [t, _s, n]} -> t * div(n, lanes.(4)) * strip
      {{:kv_write_paged, _}, [t, _ns, _mp, _p, n]} -> t * div(n, lanes.(4)) * strip
      {{:attention_paged, sc, _}, [t, hkv, g, s, dh | _]} -> instructions(code, {:attention, sc}, [t, hkv, g, s, dh], vlen)
      {{:attention, _}, [t, hkv, g, s, dh | _]} ->
        # per (row, head) at L = S: the per-key score loop (one inner chunk
        # loop copy is part of its static body), the remaining chunk passes,
        # the weighted-sum loop per output chunk, and the four O(L) passes
        lp = &Map.get(code.loops, {:l, &1}, 0)
        strips = code.loops |> Enum.filter(&match?({{:l, {:strip, _}}, _}, &1)) |> Enum.map(&elem(&1, 1)) |> Enum.sum()
        c = div(dh, 16)
        per_head = s * (lp.(:sj) + (c - 1) * lp.(:sc) + c * lp.(:oj)) + div(s, 16) * (lp.(:mx) + lp.(:sum)) +
                     div(s, lanes.(4)) * strips
        t * hkv * g * per_head
      {:sb_sums, [nsub]} -> nsub * Map.get(code.loops, {:l, :loop}, 0)
      {:gemm_i8, [m, n, k]} -> m * n * div(k, lanes.(1)) * strip
      {g, [rows, nsb, b, _ldy]} when g in [:gemv_sb4, :gemv_sb4_masked] ->
        r = Map.get(code.variant, :rows_per_iter, 1)
        blk = Map.get(code.loops, {:l, {:blk, :sb}}, 0)
        one = Map.get(code.loops, {:l, {:one, :sb}}, 0)
        full = if r == 1, do: 0, else: div(rows, r)
        b * (full * blk + (rows - full * r) * one) * nsb
      _ -> 0
    end
  end

  # floating-point operations per element of a primitive (integer bit
  # manipulation and selects are counted by the issue roof, not as FLOPs)
  defp ew_flops(:fma), do: 2
  defp ew_flops(op) when op in [:add, :sub, :mul], do: 1
  defp ew_flops(_), do: 0

  defp lanes(:x86_64, g, 4, _), do: 8 * g
  defp lanes(:x86_64, g, 1, _), do: 8 * g
  defp lanes(:x86_64_avx512, g, _, _), do: 16 * g
  defp lanes(:aarch64, g, 4, _), do: 4 * g
  defp lanes(:aarch64, g, 1, _), do: 8 * g
  defp lanes(:riscv64, g, 4, vlen), do: div(vlen, 32) * g
  defp lanes(:riscv64, g, 1, vlen), do: div(vlen, 8) * min(g, 2)
  defp lanes(_, _, _, _), do: 1

  @doc "Predicted seconds on a profile for `iterations` steps."
  def predict(%Profile{} = p, work, iterations \\ 1) do
    per_step =
      work
      |> Enum.map(fn %{flops: f, bytes: q} = w ->
        issue = if p.issue_rate, do: Map.get(w, :instructions, 0) / p.issue_rate, else: 0.0
        Enum.max([f / p.peak_flops, q / p.bandwidth, issue]) + p.dispatch_overhead_s
      end)
      |> Enum.sum()

    p.call_overhead_s + iterations * per_step
  end

  @doc "Aggregate arithmetic intensity F/Q."
  def intensity(work) do
    {f, q} = Enum.reduce(work, {0, 0}, fn w, {f, q} -> {f + w.flops, q + w.bytes} end)
    f / max(q, 1)
  end

  @doc """
  Theorem 5.1 (refined): the substrate with the smallest predicted time.
  `profiles` maps substrate ids to declared profiles.
  """
  def decide(work, iterations \\ 1, profiles \\ default_profiles()) do
    predicted = Map.new(profiles, fn {id, p} -> {id, predict(p, work, iterations)} end)
    {target, _} = Enum.min_by(predicted, fn {_, t} -> t end)
    fabric = profiles[:fabric]

    %{target: target, intensity: intensity(work), predicted_s: predicted,
      crest_fabric: if(fabric, do: fabric.peak_flops / fabric.bandwidth)}
  end

  @doc """
  Declared profile for a Vulkan device, keyed by the vendor id the daemon
  reports. Mesa's software rasterisers (llvmpipe/lavapipe, vendor 0x10005)
  get a CPU-class profile including per-run pipeline construction; unknown
  devices fall back to the conservative discrete default. Operators extend
  this table with datasheet values.
  """
  def fabric_profile(%{vendor: 0x10005}),
    do: %Profile{name: :fabric, peak_flops: 40.0e9, bandwidth: 10.0e9, call_overhead_s: 5.0e-3,
                 dispatch_overhead_s: 150.0e-6}

  def fabric_profile(_), do: default_profiles().fabric
end
