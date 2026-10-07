defmodule Vapor.Modal.Runner do
  @moduledoc """
  Run a program on a substrate: the native worker when one is given
  (`worker:`), the exact oracle otherwise — the same bits either way, in
  the canonical policy. Lowered programs are cached per runner process by
  the program's digest, so a pipeline that calls the same codec many times
  compiles it once.
  """
  alias Vapor.Program
  alias Vapor.Runtime.{Native, Oracle, Substrates}

  @doc "`%{output => Tensor}`."
  def run(%Program{} = p, env, opts \\ []) do
    case Keyword.get(opts, :worker) do
      nil ->
        Oracle.eval_program(p, env)

      w ->
        comp = lowered(p)
        {:ok, got} = Native.run(w, comp, env, isa: Keyword.get(opts, :isa, Substrates.host_isa()), mode: :native)
        got.outputs
    end
  end

  defp lowered(p) do
    key = {__MODULE__, :erlang.phash2(p)}

    case Process.get(key) do
      nil ->
        {:ok, comp} = Vapor.Compile.Lower.lower(p)
        Process.put(key, comp)
        comp

      comp ->
        comp
    end
  end

  @doc "A native worker when the host has one, else `nil` (the oracle)."
  def worker do
    case Substrates.binary("vapor-worker", "native") do
      nil -> nil
      exe -> with({:ok, w} <- Vapor.Runtime.Worker.start_link(exec: [exe]), do: w, else: (_ -> nil))
    end
  end
end
