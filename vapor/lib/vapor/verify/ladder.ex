defmodule Vapor.Verify.Ladder do
  @moduledoc """
  Section 10 — the verification ladder. Acceptance produces a signed
  certificate; rejection produces a `Vapor.Rejection` (Axiom 1).

  | rung | establishes | how |
  |---|---|---|
  | 1 | sorts, shapes, state feedback | `Vapor.Program.check/1` |
  | — | exact rewriting (ε = 0) | `Vapor.Compile.Rewrite` |
  | 2 | admission | cut sweep + allocation accepted by the Lean-extracted checker on every ISA; int8 no-wrap via extracted `admissible/3` at maximal extents; SPIR-V resource limits |
  | 3 | adjoint identity `⟨Wx, v⟩ = ⟨x, Wᵀv⟩` | compiled linear kernels vs. an exact transpose computation (layout/transposition errors) |
  | 4 | differential oracle | every CPU substrate bit-identical to the exact-arithmetic oracle |
  | 5 | parity + envelope | FNV-1a digests equal across all substrates (integers; floats under `:canonical`), every output inside the rigorous error envelope (all policies) |
  | 6 | proof-carrying code | Ed25519-signed certificate; quorum by co-signature |

  Rungs 3–5 execute deterministic probe inputs (seeded from the program
  digest) at probe extents; bounds are established at the maximal extents,
  which the monotonicity lemmas (`admissible_mono`, `withinEnvelope_mono`)
  show covers every smaller extent.
  """
  alias Vapor.{Arbiter, Bundle, Certificate, Compiled, Program, Rejection, Tensor}
  alias Vapor.Algebra.Term
  alias Vapor.Compile.{Lower, Rewrite}
  alias Vapor.Quant.Sb4
  alias Vapor.Runtime.{Dispatch, Native}
  alias Vapor.Verify.{Digest, Dyadic, Envelope}

  @spec certify(Program.t(), keyword) :: {:ok, Compiled.t()} | {:error, Rejection.t()}
  def certify(%Program{} = program, opts \\ []) do
    policy = Keyword.get(opts, :policy, :canonical)
    substrates = Keyword.get(opts, :substrates, Vapor.Runtime.Substrates.list())
    keys = List.wrap(Keyword.get(opts, :key, []))

    with :ok <- Program.check(program),
         :ok <- policy_admits(program, policy),
         {rewritten, n_rewrites} = Rewrite.rewrite(program, policy),
         :ok <- Program.check(rewritten),
         {:ok, c} <- Lower.lower(rewritten, Keyword.take(opts, [:targets]) ++ [policy: policy]),
         {:ok, admission} <- admission(c),
         probe = probe(c, opts),
         {:ok, adjoint} <- adjoint(c, probe, substrates),
         {:ok, oracle} <- Native.run_oracle(c, probe.env, probe.run),
         {:ok, runs} <- execute(c, probe, substrates),
         :ok <- differential(runs, oracle),
         {:ok, parity} <- parity(c, runs, oracle, probe) do
      payload =
        Map.merge(Bundle.digests(c), %{
          policy: policy,
          # the canonical semantics the bits follow (2: correctly rounded ÷)
          semantics: Vapor.Canon.version(),
          lean_sources: Vapor.Extracted.source_digest(),
          rewrites: n_rewrites,
          regions: c.regions,
          admission: admission,
          adjoint: adjoint,
          parity: parity,
          work: counted_work(c),
          extents: max_extents(c)
        })

      cert = Enum.reduce(keys, %Certificate{payload: payload}, &Certificate.sign(&2, &1))
      {:ok, %{c | certificate: cert}}
    end
  end

  # canonical functions are defined by their microprograms: their outputs
  # are certified by bit parity, which only the canonical policy promises
  defp policy_admits(_program, :canonical), do: :ok

  defp policy_admits(program, :fast) do
    analytic = [:max, :min]

    case program
         |> Program.order()
         |> Enum.find(&match?({:ew, op, _} when is_atom(op), &1) and Vapor.Canon.function?(elem(&1, 1)) and elem(&1, 1) not in analytic) do
      nil -> :ok
      node -> {:error, Rejection.new(node, "no a-priori envelope for canonical functions under :fast",
                                     "certify with policy :canonical (bit parity with the oracle)")}
    end
  end

  # ------------------------------------------------------------- rung 2 --

  defp admission(%Compiled{} = c) do
    registers =
      Map.new(c.code, fn {isa, %{codes: codes}} ->
        {isa,
         Map.new(codes, fn {key, code} ->
           {kernel_name(key),
            %{group_factor: code.g, bytes: byte_size(code.bin), callee_saved: code.callee_used,
              live_ranges: code.placements |> Map.values() |> Enum.map(&length/1) |> Enum.sum(),
              checker: :accepted}}
         end)}
      end)

    with {:ok, ints} <- int_admission(c),
         :ok <- spirv_limits(c) do
      {:ok, %{registers: registers, integer_no_wrap: ints}}
    end
  end

  defp int_admission(c) do
    c.program
    |> Program.order()
    |> Enum.filter(&match?({:gemm_i8, _, _}, &1))
    |> Enum.reduce_while({:ok, []}, fn {:gemm_i8, a, w} = node, {:ok, acc} ->
      {:ok, {_, [_, k]}} = Term.infer(a)
      k = Term.dim_max(k)
      {amax, wmax} = {max_abs(Program.constant(c.program, a) || a), max_abs(Program.constant(c.program, w) || w)}

      if Vapor.Extracted.admissible(k, amax, wmax) do
        {:cont, {:ok, acc ++ [%{k: k, a_max: amax, w_max: wmax, bound: k * amax * wmax, limit: 2_147_483_648}]}}
      else
        chunk = div(2_147_483_647, max(amax * wmax, 1))

        {:halt,
         {:error, Rejection.new(node, "K·|A|max·|W|max = #{k * amax * wmax} < 2³¹ (Theorem 7.1)",
                                "strip-mine K into chunks of at most #{chunk}, or rescale operands")}}
      end
    end)
  end

  defp max_abs({:const, %Tensor{} = t}), do: Tensor.max_abs(t)
  defp max_abs({:input, _, :s8, _}), do: 128

  defp spirv_limits(c) do
    bad = Enum.find(c.spirv, fn {_k, %{nbind: b, npush: p}} -> b > 16 or p > 32 end)
    if bad, do: {:error, Rejection.new(elem(bad, 0), "≤ 16 bindings, ≤ 128 B push constants", "split the region")}, else: :ok
  end

  # ------------------------------------------------------------- probes --

  @doc false
  def probe(%Compiled{} = c, opts) do
    seed = Digest.fnv1a64(Vapor.Canonical.encode(c.program))
    dims_override = Keyword.get(opts, :probe_dims, %{})
    recurrent? = c.program.state != []
    t_count = if recurrent?, do: Keyword.get(opts, :probe_iterations, 3), else: 1
    state = MapSet.new(Enum.map(c.program.state, &elem(&1, 0)))

    inputs = Program.inputs(c.program)

    env =
      inputs
      |> Enum.with_index()
      |> Map.new(fn {{:input, name, dt, shape}, i} ->
        concrete = Enum.map(shape, fn
          {:dyn, s, max} -> Map.get(dims_override, s, min(max, 13))
          n -> n
        end)

        seq? = recurrent? and not MapSet.member?(state, name)
        full = if seq?, do: [t_count | concrete], else: concrete
        {name, random(dt, full, seed + i)}
      end)

    seq = for {:input, name, _, _} <- inputs, recurrent?, not MapSet.member?(state, name), do: name
    run = if recurrent?, do: [iterations: t_count, sequence: seq], else: []
    %{env: env, run: run, seed: seed}
  end

  defp random(:f32, shape, seed), do: Tensor.random(:f32, shape, seed)
  defp random(:s8, shape, seed), do: Tensor.random(:s8, shape, seed)
  defp random(:s32, shape, seed), do: Tensor.random(:s32, shape, seed)

  # ------------------------------------------------------------- rung 3 --

  defp adjoint(c, probe, substrates) do
    primary = Enum.find(substrates, &(&1.kind == :native)) || Enum.find(substrates, &(&1.kind == :oracle))

    # weights bound by let-bindings are the same constants: substitute them
    k = &(Program.constant(c.program, &1) || &1)

    c.program
    |> Program.order()
    |> Enum.flat_map(fn
      {:qgemv, w, x} -> [{:qgemv, k.(w), x}]
      {:gemm_i8, a, w} -> [{:gemm_i8, a, k.(w)}]
      {:linear, x, w} -> [{:linear, x, k.(w)}]
      {:linear_masked, x, w, m} -> [{:linear_masked, x, k.(w), m}]
      _ -> []
    end)
    |> Enum.filter(fn
      {:qgemv, {:const, _}, _} -> true
      {:gemm_i8, _, {:const, _}} -> true
      {:linear, _, {:const, _}} -> true
      {:linear_masked, _, {:const, _}, _} -> true
      _ -> false
    end)
    |> Enum.uniq()
    |> Enum.reduce_while({:ok, []}, fn node, {:ok, acc} ->
      case adjoint_node(node, c.policy, probe.seed, primary) do
        {:ok, r} -> {:cont, {:ok, acc ++ [r]}}
        {:error, _} = e -> {:halt, e}
      end
    end)
  end

  defp adjoint_node({:qgemv, {:const, w} = wc, _x} = node, policy, seed, sub) do
    wx = Sb4.to_exec(w)
    [rows, k] = wx.shape
    xin = Term.input(:x, :f32, [k])
    {:ok, c} = Lower.lower(Program.new(y: Term.qgemv(wc, xin)), policy: policy)
    xv = Tensor.random(:f32, [k], seed + 101)
    v = for i <- 1..rows, do: if(rem(elem(Tensor.splitmix(seed + i), 0), 2) == 0, do: 1, else: -1)

    with {:ok, %{outputs: %{y: y}}} <- Dispatch.run_on(sub, c, %{x: xv}, []) do
      [bounds] = Envelope.bounds(c, %{x: xv})
      lhs = Enum.zip_reduce(Tensor.to_list(y), v, Dyadic.zero(), fn b, s, acc -> Dyadic.add(acc, Dyadic.mul(Dyadic.of_f32(b), Dyadic.of_int(s))) end)

      # Wᵀv computed column-wise from the declared dequantisation, exactly
      rhs = transpose_dot(wx, v, xv)

      tol =
        Enum.zip_reduce(bounds[:y], v, Dyadic.zero(), fn b, _s, acc ->
          Dyadic.add(acc, bound_value(b))
        end)

      if Dyadic.le?(Dyadic.abs(Dyadic.sub(lhs, rhs)), tol),
        do: {:ok, %{node: :qgemv, rows: rows, k: k, substrate: sub.id, identity: :holds}},
        else: {:error, Rejection.new(node, "⟨Wx, v⟩ = ⟨x, Wᵀv⟩ within the envelope", "kernel layout or transposition error")}
    end
  end

  defp adjoint_node({:linear, _x, {:const, w} = wc} = node, policy, seed, sub) do
    [rows, k] = w.shape
    {:ok, c} = Lower.lower(Program.new(y: Term.linear(Term.input(:x, :f32, [k]), wc)), policy: policy)
    xv = Tensor.random(:f32, [k], seed + 111)
    v = for i <- 1..rows, do: if(rem(elem(Tensor.splitmix(seed + 7 * i), 0), 2) == 0, do: 1, else: -1)

    with {:ok, %{outputs: %{y: y}}} <- Dispatch.run_on(sub, c, %{x: xv}, []) do
      [bounds] = Envelope.bounds(c, %{x: xv})
      lhs = Enum.zip_reduce(Tensor.to_list(y), v, Dyadic.zero(), fn b, s, acc -> Dyadic.add(acc, Dyadic.mul(Dyadic.of_f32(b), Dyadic.of_int(s))) end)

      # Wᵀv column by column, exactly, then ⟨x, Wᵀv⟩
      wrows = Tensor.widen(w).data |> Vapor.F32.decode() |> Enum.chunk_every(k)
      wtv =
        Enum.zip_reduce(wrows, v, List.duplicate(Dyadic.zero(), k), fn row, s, acc ->
          Enum.zip_with(acc, row, fn a, b -> Dyadic.add(a, Dyadic.mul(Dyadic.of_int(s), Dyadic.of_f32(b))) end)
        end)

      rhs = Enum.zip_reduce(Tensor.to_list(xv), wtv, Dyadic.zero(), fn xb, t, acc -> Dyadic.add(acc, Dyadic.mul(Dyadic.of_f32(xb), t)) end)
      tol = Enum.reduce(bounds[:y], Dyadic.zero(), fn b, acc -> Dyadic.add(acc, bound_value(b)) end)

      if Dyadic.le?(Dyadic.abs(Dyadic.sub(lhs, rhs)), tol),
        do: {:ok, %{node: :linear, rows: rows, k: k, substrate: sub.id, identity: :holds}},
        else: {:error, Rejection.new(node, "⟨Wx, v⟩ = ⟨x, Wᵀv⟩ within the envelope", "kernel layout or transposition error")}
    end
  end

  # the predicated kernel on two rows, mask (1, 0): the active row must
  # satisfy the adjoint identity, the inactive row must be +0 bit for bit
  defp adjoint_node({:linear_masked, _x, {:const, w} = wc, _m} = node, policy, seed, sub) do
    [rows, k] = w.shape
    prog = Program.new(y: Term.linear_masked(Term.input(:x, :f32, [2, k]), wc, Term.input(:m, :f32, [2, 1])))
    {:ok, c} = Lower.lower(prog, policy: policy)
    xv = Tensor.random(:f32, [2, k], seed + 121)
    mv = Tensor.new(:f32, [2, 1], <<0x3F800000::32-little, 0x80000000::32-little>>)
    v = for i <- 1..rows, do: if(rem(elem(Tensor.splitmix(seed + 11 * i), 0), 2) == 0, do: 1, else: -1)

    with {:ok, %{outputs: %{y: y}}} <- Dispatch.run_on(sub, c, %{x: xv, m: mv}, []) do
      {y0, y1} = Enum.split(Tensor.to_list(y), rows)
      [bounds] = Envelope.bounds(c, %{x: xv, m: mv})
      lhs = Enum.zip_reduce(y0, v, Dyadic.zero(), fn b, s, acc -> Dyadic.add(acc, Dyadic.mul(Dyadic.of_f32(b), Dyadic.of_int(s))) end)
      wrows = Tensor.widen(w).data |> Vapor.F32.decode() |> Enum.chunk_every(k)

      wtv =
        Enum.zip_reduce(wrows, v, List.duplicate(Dyadic.zero(), k), fn row, s, acc ->
          Enum.zip_with(acc, row, fn a, b -> Dyadic.add(a, Dyadic.mul(Dyadic.of_int(s), Dyadic.of_f32(b))) end)
        end)

      rhs = Enum.zip_reduce(Enum.take(Tensor.to_list(xv), k), wtv, Dyadic.zero(), fn xb, t, acc -> Dyadic.add(acc, Dyadic.mul(Dyadic.of_f32(xb), t)) end)
      tol = bounds[:y] |> Enum.take(rows) |> Enum.reduce(Dyadic.zero(), fn b, acc -> Dyadic.add(acc, bound_value(b)) end)

      cond do
        Enum.any?(y1, &(&1 != 0)) ->
          {:error, Rejection.new(node, "inactive rows are +0", "the predicated kernel wrote a masked row")}

        Dyadic.le?(Dyadic.abs(Dyadic.sub(lhs, rhs)), tol) ->
          {:ok, %{node: :linear_masked, rows: rows, k: k, substrate: sub.id, identity: :holds}}

        true ->
          {:error, Rejection.new(node, "⟨Wx, v⟩ = ⟨x, Wᵀv⟩ within the envelope", "kernel layout or transposition error")}
      end
    end
  end

  defp adjoint_node({:gemm_i8, a, {:const, w} = wc} = node, _policy, seed, sub) do
    {:ok, {:s8, [m, k]}} = Term.infer(a)
    m = Term.dim_max(m) |> min(13)
    [n, _] = w.shape
    {:ok, c} = Lower.lower(Program.new(c: Term.gemm_i8(Term.input(:a, :s8, [m, k]), wc)))
    av = Tensor.random(:s8, [m, k], seed + 202)
    vv = Tensor.random(:s8, [m, n], seed + 203)

    with {:ok, %{outputs: %{c: out}}} <- Dispatch.run_on(sub, c, %{a: av}, []) do
      lhs = Enum.zip_reduce(Tensor.to_list(out), Tensor.to_list(vv), 0, fn x, y, s -> s + x * y end)
      # ⟨A·Wᵀ, V⟩_F = ⟨A, V·W⟩_F, the right side computed without the kernel
      vw = matmul(Enum.chunk_every(Tensor.to_list(vv), n), Enum.chunk_every(Tensor.to_list(w), k))
      rhs = Enum.zip_reduce(Tensor.to_list(av), List.flatten(vw), 0, fn x, y, s -> s + x * y end)

      if lhs == rhs,
        do: {:ok, %{node: :gemm_i8, m: m, n: n, k: k, substrate: sub.id, identity: :exact}},
        else: {:error, Rejection.new(node, "⟨AWᵀ, V⟩ = ⟨A, VW⟩ exactly", "kernel layout or transposition error")}
    end
  end

  defp matmul(vrows, wrows) do
    # (V·W)[i][kk] = Σ_j V[i][j]·W[j][kk]
    wt = List.to_tuple(wrows)
    for vr <- vrows do
      vr
      |> Enum.with_index()
      |> Enum.reduce(List.duplicate(0, length(elem(wt, 0))), fn {s, j}, acc ->
        Enum.zip_with(acc, elem(wt, j), &(&1 + s * &2))
      end)
    end
  end

  defp transpose_dot(wx, v, xv) do
    xs = xv |> Tensor.to_list() |> Enum.map(&Dyadic.of_f32/1) |> List.to_tuple()
    [rows, _k] = wx.shape

    Enum.zip(0..(rows - 1), v)
    |> Enum.reduce(Dyadic.zero(), fn {r, s}, acc ->
      row = Tensor.row(wx, r)

      {contrib, _} =
        for <<blk::binary-152 <- row>>, reduce: {Dyadic.zero(), 0} do
          {a, sg} ->
            {q, ab} = Sb4.exec_block(blk)

            Enum.with_index(ab)
            |> Enum.reduce({a, sg}, fn {{alpha, beta}, sl}, {a, sg} ->
              a =
                Enum.reduce(0..31, a, fn i, a ->
                  w = Vapor.F32.mul(alpha, Vapor.F32.from_float(elem(q, 32 * sl + i)))
                  coef = Dyadic.add(Dyadic.of_f32(w), Dyadic.of_f32(beta))
                  Dyadic.add(a, Dyadic.mul(coef, elem(xs, 32 * sg + i)))
                end)

              {a, sg + 1}
            end)
        end

      Dyadic.add(acc, Dyadic.mul(Dyadic.of_int(s), contrib))
    end)
  end

  defp bound_value({:running, _v, e}), do: e
  defp bound_value({:wilkinson, _v, s, a, n}), do: Dyadic.add(Dyadic.mul(Dyadic.gamma(n), s), a)

  # --------------------------------------------------------- rungs 4 & 5 --

  defp execute(c, probe, substrates) do
    substrates
    |> Enum.reject(&(&1.kind == :oracle))
    |> Enum.filter(&Compiled.runs_on?(c, &1))
    |> Enum.reduce_while({:ok, []}, fn sub, {:ok, acc} ->
      case Dispatch.run_on(sub, c, probe.env, probe.run) do
        {:ok, r} -> {:cont, {:ok, acc ++ [{sub, r}]}}
        {:error, reason} -> {:halt, {:error, Rejection.new({:substrate, sub.id}, {:execution_failed, reason}, "exclude the substrate or fix the kernel")}}
      end
    end)
  end

  defp differential(runs, oracle) do
    Enum.reduce_while(runs, :ok, fn
      {%{kind: :native} = sub, r}, :ok ->
        if same?(r, oracle),
          do: {:cont, :ok},
          else: {:halt, {:error, Rejection.new({:substrate, sub.id}, "bit-identical to the oracle (Rung 4)", "backend miscompilation: report the kernel")}}

      _, :ok ->
        {:cont, :ok}
    end)
  end

  defp same?(r, o), do: r.outputs == o.outputs and steps(r) == steps(o)
  defp steps(%{steps: []} = r), do: [r.outputs]
  defp steps(%{steps: s}), do: s

  defp parity(c, runs, oracle, probe) do
    bounds = Envelope.bounds(c, probe.env, probe.run)
    all = [{%{id: :oracle, kind: :oracle}, oracle} | runs]
    int_outputs = for {name, id} <- c.outputs, c.slots[id].dtype == :s32, do: name

    digests =
      Map.new(all, fn {sub, r} ->
        {sub.id, r |> steps() |> Enum.map(fn m -> Map.new(m, fn {k, t} -> {k, Digest.tensor(t)} end) end)}
      end)

    ref = digests[:oracle]
    strict = c.policy == :canonical

    mismatched =
      for {id, d} <- digests, id != :oracle,
          mismatch = diff(d, ref, if(strict, do: :all, else: int_outputs)), mismatch != [],
          do: {id, mismatch}

    envelope =
      Enum.reduce_while(all, :ok, fn {sub, r}, :ok ->
        case check_envelope(steps(r), bounds, int_outputs) do
          :ok -> {:cont, :ok}
          {:error, info} -> {:halt, {:error, sub.id, info}}
        end
      end)

    cond do
      mismatched != [] ->
        {:error, Rejection.new({:parity, mismatched}, "FNV-1a digests equal across substrates (Rung 5)",
                               "a substrate deviates from the canonical order; exclude it or use policy :fast")}

      match?({:error, _, _}, envelope) ->
        {:error, sub, info} = envelope
        {:error, Rejection.new({:envelope, sub, info}, "every output inside the certified envelope", "numerical nonconformance")}

      true ->
        maxima =
          bounds
          |> List.last()
          |> Map.new(fn {name, bs} ->
            {name, if(Envelope.analytic?(bs), do: Dyadic.to_float(Envelope.max_error(bs)), else: :bit_parity)}
          end)

        {:ok,
         %{substrates: Enum.map(all, &elem(&1, 0).id) |> Enum.sort(),
           digests: ref,
           bit_identical: if(strict, do: :all_outputs, else: int_outputs),
           envelope_max_abs_error: maxima}}
    end
  end

  defp diff(d, ref, which) do
    Enum.zip(d, ref)
    |> Enum.with_index()
    |> Enum.flat_map(fn {{a, b}, t} ->
      for {name, h} <- a, which == :all or name in which, h != b[name], do: {t, name}
    end)
  end

  defp check_envelope(steps, bounds, _int_outputs) do
    Enum.zip(steps, bounds)
    |> Enum.reduce_while(:ok, fn {outs, bs}, :ok ->
      Enum.reduce_while(outs, :ok, fn {name, t}, :ok ->
        case Envelope.check(bs[name], t) do
          :ok -> {:cont, :ok}
          {:error, info} -> {:halt, {:error, Map.put(info, :output, name)}}
        end
      end)
      |> case do
        :ok -> {:cont, :ok}
        e -> {:halt, e}
      end
    end)
  end

  # ----------------------------------------------------------- dispatch --

  # hardware-independent: FLOPs and bytes per step, and the dynamic
  # instruction count of the emitted code per ISA, at the maximal extents
  defp counted_work(c) do
    dims = max_dims(c)
    per_isa = Map.new(c.code, fn {isa, _} -> {isa, Arbiter.work(c, dims, isa)} end)
    any = per_isa |> Map.values() |> hd()

    %{flops: Enum.sum(Enum.map(any, & &1.flops)), bytes: Enum.sum(Enum.map(any, & &1.bytes)),
      instructions: Map.new(per_isa, fn {isa, w} -> {isa, Enum.sum(Enum.map(w, & &1.instructions))} end)}
  end

  defp max_dims(c) do
    c.slots
    |> Map.values()
    |> Enum.flat_map(& &1.shape)
    |> Enum.flat_map(fn
      {:dyn, s, m} -> [{s, m}]
      _ -> []
    end)
    |> Map.new()
  end

  defp max_extents(c), do: max_dims(c)

  defp kernel_name({:ew, spec}), do: "ew/" <> Digest.hex64(Digest.fnv1a64(Vapor.Canonical.encode(spec)))
  defp kernel_name(k) when is_atom(k), do: Atom.to_string(k)
  defp kernel_name(t) when is_tuple(t), do: t |> Tuple.to_list() |> Enum.map_join("/", &name_part/1)

  defp name_part(a) when is_atom(a), do: Atom.to_string(a)
  defp name_part(i) when is_integer(i), do: Integer.to_string(i, 16)
end
