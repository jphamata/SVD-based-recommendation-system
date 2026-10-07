defmodule Vapor do
  @moduledoc """
  vapor — certified, reproducible tensor synthesis for the BEAM.

      program = Vapor.Program.new(y: Vapor.Algebra.Term.qgemv(w, x))
      {:ok, compiled} = Vapor.compile(program, key: Vapor.Certificate.keygen())
      {:ok, result, trace} = Vapor.run(compiled, %{x: x})

  `compile/2` discharges the six-rung ladder (`Vapor.Verify.Ladder`) and
  returns machine code for x86-64/AArch64/RISC-V, SPIR-V for Vulkan, and a
  signed certificate; `run/3` executes on the arbiter's substrate with
  transparent failover (`Vapor.Runtime.Dispatch`). Nothing generated ever
  executes inside the BEAM.
  """
  alias Vapor.{Compiled, Program}
  alias Vapor.Runtime.Dispatch

  @doc """
  Certify and compile. Options: `:policy` (`:canonical` | `:fast`), `:key`
  (one or more `Vapor.Certificate.keygen/0` keys), `:substrates`,
  `:probe_dims`, `:probe_iterations`, `:targets`.
  """
  @spec compile(Program.t(), keyword) :: {:ok, Compiled.t()} | {:error, Vapor.Rejection.t()}
  def compile(%Program{} = p, opts \\ []), do: Vapor.Verify.Ladder.certify(p, opts)

  @doc """
  Execute a compiled program. For recurrent programs every non-state input
  carries a leading sequence dimension `T`; the whole loop runs behind one
  crossing and `:on_emit` streams each step. Returns `{:ok, result, trace}`.
  """
  def run(%Compiled{} = c, env, opts \\ []) do
    state = MapSet.new(Enum.map(c.program.state, &elem(&1, 0)))

    seq_opts =
      if c.program.state == [] do
        []
      else
        seq = for {:input, n, _, _} <- Program.inputs(c.program), not MapSet.member?(state, n), do: n
        t = env |> Map.fetch!(hd(seq)) |> Map.fetch!(:shape) |> hd()
        [iterations: t, sequence: seq]
      end

    prefer = Keyword.get_lazy(opts, :prefer, fn -> decide(c, env, seq_opts) end)
    Dispatch.run(c, env, [prefer: prefer] ++ seq_opts ++ Keyword.delete(opts, :prefer))
  end

  @doc """
  The arbiter's choice for this node: counted work of the compiled program
  (at the bound extents) against this node's declared profiles.
  """
  def decide(%Compiled{} = c, env, seq_opts \\ []) do
    state = MapSet.new(Enum.map(c.program.state, &elem(&1, 0)))
    step_env = Map.new(env, fn {k, v} ->
      {k, if(k in Keyword.get(seq_opts, :sequence, []) and not MapSet.member?(state, k), do: Vapor.Runtime.Native.slice(v, 0), else: v)}
    end)

    {:ok, dims} = Compiled.dims(c, step_env)
    work = Vapor.Arbiter.work(c, dims)
    subs = Vapor.Runtime.Substrates.list()
    defaults = Vapor.Arbiter.default_profiles()

    profiles =
      %{native: defaults.native}
      |> then(fn p ->
        case Enum.find(subs, &(&1.id == :fabric)) do
          nil -> p
          %{server: f} -> Map.put(p, :fabric, Vapor.Arbiter.fabric_profile(Vapor.Runtime.Fabric.info(f)))
        end
      end)

    Vapor.Arbiter.decide(work, Keyword.get(seq_opts, :iterations, 1), profiles).target
  end
end
