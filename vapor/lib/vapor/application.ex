defmodule Vapor.Application do
  @moduledoc """
  The control-plane supervision tree. The substrate supervisor owns the
  worker and fabric OS processes; any of them may crash and be restarted
  without touching the rest of the node (Axiom 3). Configure with
  `config :vapor, substrates: [cross: true, fabric: true, sandbox: true]`.
  """
  use Application

  @impl true
  def start(_type, _args) do
    opts = Application.get_env(:vapor, :substrates, [])
    children = [{Vapor.Runtime.Substrates, opts}, {Registry, keys: :unique, name: Vapor.Athanor.Registry},
                {DynamicSupervisor, strategy: :one_for_one, name: Vapor.Athanor.Sessions}]
    Supervisor.start_link(children, strategy: :one_for_one, name: Vapor.Supervisor)
  end
end
