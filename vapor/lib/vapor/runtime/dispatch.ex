defmodule Vapor.Runtime.Dispatch do
  @moduledoc """
  Resilient execution (Axiom 3, end to end).

  The arbiter's choice is tried first; every failure — a crashed worker or
  daemon, a lost device, an emulator fault, an unavailable tier — is
  recorded and the unit is rerouted down the chain

      fabric → metal → host native (AVX-512, then the base ISA) → RVV interpreter → oracle

  A substrate whose admission (`Vapor.Substrate`) does not allow the
  program's policy is skipped: a device measured as envelope-bound never
  receives a canonical program, a refused one receives nothing.

  The oracle cannot fail for a certified program, so a certified unit always
  completes; the trace says which substrate produced the result and why the
  others were skipped. Every substrate on the chain computes the *same*
  certified semantics (bit-identical under the canonical policy), so
  rerouting never changes the answer.
  """
  alias Vapor.Compiled
  alias Vapor.Runtime.{Fabric, Native, Substrates}

  @doc "Run on one substrate descriptor."
  def run_on(%{kind: :native, server: s, isa: isa, mode: mode}, %Compiled{} = c, env, opts),
    do: Native.run(s, c, env, [isa: isa, mode: mode] ++ opts)

  def run_on(%{kind: :fabric, server: s}, %Compiled{} = c, env, opts), do: Fabric.run(s, c, env, opts)
  def run_on(%{kind: :oracle}, %Compiled{} = c, env, opts), do: Native.run_oracle(c, env, opts)

  @doc """
  Run with failover. `:prefer` is `:fabric` or `:native` (normally the
  arbiter's decision); `:substrates` overrides discovery.
  Returns `{:ok, result, trace}`.
  """
  def run(%Compiled{} = c, env, opts \\ []) do
    subs = Keyword.get(opts, :substrates, Substrates.list())
    chain = chain(subs, Keyword.get(opts, :prefer, :native), c)
    run_opts = Keyword.drop(opts, [:substrates, :prefer])

    Enum.reduce_while(chain, {:error, :no_substrate, []}, fn sub, {:error, _, trace} ->
      case safe(fn -> run_on(sub, c, env, run_opts) end) do
        {:ok, res} -> {:halt, {:ok, Map.put(res, :substrate, sub.id), Enum.reverse([{sub.id, :ok} | trace])}}
        {:error, reason} -> {:cont, {:error, reason, [{sub.id, {:failed, reason}} | trace]}}
      end
    end)
    |> case do
      {:ok, _, _} = ok -> ok
      {:error, reason, trace} -> {:error, reason, Enum.reverse(trace)}
    end
  end

  defp chain(subs, prefer, c) do
    by = Map.new(subs, &{&1.id, &1})
    host_ok = Map.has_key?(c.code, Substrates.host_isa())

    order =
      if prefer == :fabric,
        do: [:fabric, :metal, :host_avx512, :host, :rvv_emulated, :oracle],
        else: [:host_avx512, :host, :fabric, :metal, :rvv_emulated, :oracle]

    order
    |> Enum.reject(&(&1 == :host and not host_ok))
    |> Enum.flat_map(fn id -> List.wrap(by[id]) end)
    |> Enum.filter(&(Compiled.runs_on?(c, &1) and Vapor.Substrate.allowed?(&1, c.policy)))
  end

  defp safe(f) do
    f.()
  catch
    :exit, reason -> {:error, {:exit, reason}}
  end
end
