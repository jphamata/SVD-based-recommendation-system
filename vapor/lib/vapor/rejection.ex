defmodule Vapor.Rejection do
  @moduledoc """
  Axiom 1, rejection side: an explicit counterexample — the violating node,
  the exceeded bound, and an algebraic repair candidate. Compilation never
  raises on a well-typed request; it accepts with a certificate or rejects
  with one of these.
  """
  @enforce_keys [:node, :bound, :repair]
  defstruct [:node, :bound, :repair]

  @type t :: %__MODULE__{node: term(), bound: term(), repair: term()}

  def new(node, bound, repair), do: %__MODULE__{node: node, bound: bound, repair: repair}
end
