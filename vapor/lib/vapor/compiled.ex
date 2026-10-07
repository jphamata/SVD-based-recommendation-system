defmodule Vapor.Compiled do
  @moduledoc """
  The result of lowering: value slots, the kernel schedule, and one linked
  code blob per ISA (plus SPIR-V modules when the fabric tier is targeted).
  Schedules are symbolic in the semi-dynamic dimensions; `bind/2` resolves
  them against concrete inputs.
  """
  alias Vapor.{Rejection, Tensor}

  defstruct [:program, :policy, :slots, :kernels, :schedule, :code, :outputs, :state,
             :regions, inplace: MapSet.new(), spirv: %{}, certificate: nil]

  @type t :: %__MODULE__{}

  @doc """
  Unify the declared input sorts with concrete tensors, returning the
  dimension environment `%{sym => extent}` (`partial: true` lets static
  inputs be omitted). A dynamic extent above its
  certified maximum is rejected (the certificate would not cover it).
  """
  @spec dims(t, %{atom => Tensor.t()}, keyword) :: {:ok, map} | {:error, Rejection.t()}
  def dims(%__MODULE__{slots: slots}, env, opts \\ []) do
    partial = Keyword.get(opts, :partial, false)

    slots
    |> Map.values()
    |> Enum.filter(&match?({:input, _}, &1.role))
    |> Enum.reduce_while({:ok, %{}}, fn %{role: {:input, name}, dtype: dt, shape: shape}, {:ok, d} ->
      case Map.fetch(env, name) do
        # `partial: true` (sessions): an omitted input keeps its contents
        # and must have a static shape
        :error when partial and is_list(shape) ->
          if Enum.all?(shape, &is_integer/1),
            do: {:cont, {:ok, d}},
            else: {:halt, {:error, Rejection.new({:input, name}, "given (its shape is dynamic)", "pass it to every step")}}

        {:ok, %Tensor{dtype: ^dt, shape: actual}} when length(actual) == length(shape) ->
          case unify(shape, actual, d) do
            {:ok, d} -> {:cont, {:ok, d}}
            {:error, why} -> {:halt, {:error, Rejection.new({:input, name}, why, "respect the declared extents")}}
          end

        _ ->
          {:halt, {:error, Rejection.new({:input, name}, "bound to #{dt}#{inspect(shape, charlists: :as_lists)}", "bind every input")}}
      end
    end)
  end

  defp unify(decl, actual, d) do
    Enum.zip(decl, actual)
    |> Enum.reduce_while({:ok, d}, fn
      {n, n}, acc when is_integer(n) -> {:cont, acc}
      {{:dyn, s, max}, a}, {:ok, d} when a <= max ->
        case Map.get(d, s, a) do
          ^a -> {:cont, {:ok, Map.put(d, s, a)}}
          other -> {:halt, {:error, "#{s} bound to both #{other} and #{a}"}}
        end
      {{:dyn, s, max}, a}, _ -> {:halt, {:error, "#{s} = #{a} exceeds certified maximum #{max}"}}
      {n, a}, _ -> {:halt, {:error, "extent #{a} ≠ declared #{n}"}}
    end)
  end

  @doc "Concrete shape of a slot under a dimension environment."
  def shape(%__MODULE__{slots: slots}, id, dims) do
    Enum.map(slots[id].shape, fn
      {:dyn, s, _} -> Map.fetch!(dims, s)
      n -> n
    end)
  end

  def nbytes(c, id, dims), do: Tensor.nbytes(c.slots[id].dtype, shape(c, id, dims))

  @doc "Resolve a symbolic kernel argument to a u64 or a slot reference."
  def resolve_arg(c, {:numel, id}, dims), do: {:imm, Enum.product(shape(c, id, dims))}
  def resolve_arg(c, {:rows, id}, dims), do: {:imm, c |> shape(id, dims) |> Enum.drop(-1) |> Enum.product()}
  def resolve_arg(c, {:cols, id}, dims), do: {:imm, c |> shape(id, dims) |> List.last()}
  def resolve_arg(c, {:dim, id, i}, dims), do: {:imm, c |> shape(id, dims) |> List.to_tuple() |> elem(i)}
  def resolve_arg(_c, {:imm, n}, _dims), do: {:imm, n}
  def resolve_arg(_c, {:slot, id}, _dims), do: {:slot, id}

  @doc "Whether the fabric has a SPIR-V module for every scheduled kernel."
  def fabric_complete?(%__MODULE__{kernels: ks, spirv: sp}), do: Enum.all?(Map.keys(ks), &Map.has_key?(sp, &1))

  @doc "Whether a substrate descriptor can run this program at all."
  def runs_on?(%__MODULE__{} = c, %{kind: :fabric}), do: fabric_complete?(c)
  def runs_on?(%__MODULE__{} = c, %{kind: :native, isa: isa}), do: Map.has_key?(c.code, isa)
  def runs_on?(_c, %{kind: :oracle}), do: true

end
