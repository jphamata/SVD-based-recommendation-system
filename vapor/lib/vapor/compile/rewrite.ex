defmodule Vapor.Compile.Rewrite do
  @moduledoc """
  Section 2.2 — equational rewriting `t ⟶ t′` with `⟦t⟧ = ⟦t′⟧` *bit for bit*
  (static error budget ε = 0). Only identities that hold for every IEEE-754
  finite input, signed zeros and subnormals included, are admitted — each
  proved in Lean against an IEEE-754 model built from the bit patterns
  (`proofs/Vapor/Binary32.lean`: the only correctly rounded result is `x`):

    * `neg(neg x) → x` (`neg_neg`)
    * `x · 1.0 → x`, `1.0 · x → x` (`mul_one_unique`, `one_mul_unique`)
    * `x + (−0.0) → x`, `(−0.0) + x → x` (`add_negZero_unique`) — but **not**
      `x + 0.0 → x`: `(−0) + (+0) = +0` (`add_posZero_refuted`)
    * `x − (+0.0) → x` (`sub_posZero_unique`) — but not `x − (−0.0)`
    * `relu(relu x) → relu x`
    * constant folding: a node whose operands are all constants is evaluated
      by the oracle (the declared semantics itself), under the program's
      float policy — never reassociated, never approximated.

  NaN is outside the claim, deliberately: substrates disagree on NaN bits
  (x86 quiets a signalling NaN and keeps its payload; RISC-V returns the
  canonical NaN; Arm in default-NaN mode too), so no rewrite — and no
  kernel — is bit-exact on NaN across substrates. The oracle refuses
  non-finite values, so certificates speak of finite executions only.
  (0.7 said "including NaN" here; that was wrong, and is tested now.)

  Rewriting is innermost-first, bounded, and hash-consed through the DAG.
  """
  alias Vapor.Program
  alias Vapor.Runtime.Oracle

  @one 0x3F80_0000
  @pzero 0x0000_0000
  @nzero 0x8000_0000

  @spec rewrite(Program.t(), atom, pos_integer) :: {Program.t(), non_neg_integer}
  def rewrite(%Program{outputs: outs, lets: lets} = p, policy \\ :canonical, budget \\ 10_000) do
    {memo, count} =
      p
      |> Program.order()
      |> Enum.reduce({%{}, 0}, fn t, {memo, n} ->
        t2 = rebuild(t, memo)
        {t3, k} = simplify(t2, policy, budget - n)
        {Map.put(memo, t, t3), n + k}
      end)

    # binding bodies are rewritten in place; references keep their names
    new = fn t -> Map.get(memo, t, t) end

    {%{p | outputs: Enum.map(outs, fn {name, t} -> {name, new.(t)} end),
           lets: Enum.map(lets, fn {name, t} -> {name, new.(t)} end)}, count}
  end

  defp rebuild({:ew, op, args}, memo), do: {:ew, op, Enum.map(args, &sub(&1, memo))}
  defp rebuild({:qgemv, w, x}, memo), do: {:qgemv, sub(w, memo), sub(x, memo)}
  defp rebuild({:gemm_i8, a, w}, memo), do: {:gemm_i8, sub(a, memo), sub(w, memo)}
  defp rebuild(t, _memo), do: t

  defp sub({:splat, _} = s, _memo), do: s
  defp sub(t, memo), do: Map.get(memo, t, t)

  defp simplify(t, _policy, budget) when budget <= 0, do: {t, 0}

  defp simplify(t, policy, budget) do
    case step(t, policy) do
      {:ok, t2} ->
        {t3, k} = simplify(t2, policy, budget - 1)
        {t3, k + 1}

      :none ->
        {t, 0}
    end
  end

  defp step({:ew, :neg, [{:ew, :neg, [x]}]}, _), do: {:ok, x}
  defp step({:ew, :relu, [{:ew, :relu, _} = r]}, _), do: {:ok, r}
  defp step({:ew, :mul, [x, {:splat, @one}]}, _) when elem(x, 0) != :splat, do: {:ok, x}
  defp step({:ew, :mul, [{:splat, @one}, x]}, _) when elem(x, 0) != :splat, do: {:ok, x}
  defp step({:ew, :add, [x, {:splat, @nzero}]}, _) when elem(x, 0) != :splat, do: {:ok, x}
  defp step({:ew, :add, [{:splat, @nzero}, x]}, _) when elem(x, 0) != :splat, do: {:ok, x}
  defp step({:ew, :sub, [x, {:splat, @pzero}]}, _) when elem(x, 0) != :splat, do: {:ok, x}

  defp step({op, _, _} = t, policy) when op in [:ew, :qgemv, :gemm_i8] do
    if foldable?(t), do: {:ok, {:const, Oracle.eval(t, %{}, policy)}}, else: :none
  end

  defp step(_, _), do: :none

  defp foldable?({:ew, _, args}),
    do: Enum.all?(args, &(match?({:splat, _}, &1) or match?({:const, _}, &1))) and
          Enum.any?(args, &match?({:const, _}, &1))

  defp foldable?({:qgemv, {:const, _}, {:const, _}}), do: true
  defp foldable?({:gemm_i8, {:const, _}, {:const, _}}), do: true
  defp foldable?(_), do: false
end
