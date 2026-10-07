defmodule Vapor.MixProject do
  use Mix.Project

  # The control plane has no external dependencies: no JSON library, no HTTP
  # client, no vendor runtime binding can permeate the tree (Axioms 4/5).
  def project do
    [
      app: :vapor,
      version: "0.14.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: [],
      description: "vapor — certified, reproducible tensor synthesis for the BEAM",
      package: [licenses: ["ISC"]]
    ]
  end

  def application do
    # inets/ssl/public_key: OTP's HTTP client, for the remote agent backends
    # (Vapor.Agent.HTTP) — part of Erlang/OTP, not dependencies; declared so
    # Mix ≥ 1.15 keeps them on the code path
    [extra_applications: [:logger, :crypto, :inets, :ssl, :public_key], mod: {Vapor.Application, []}]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]
end
