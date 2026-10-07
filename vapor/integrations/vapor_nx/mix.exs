defmodule VaporNx.MixProject do
  use Mix.Project

  # Nx (≥ 1.0, Elixir ≥ 1.17). With VAPOR_ECO=<dir> holding source checkouts
  # (nx, complex, telemetry) it builds with no package registry.
  def project do
    [
      app: :vapor_nx,
      version: "0.3.0",
      elixir: "~> 1.17",
      deps: deps(),
      description: "Nx.Defn compiler to vapor: certified, bit-identical across CPUs and GPUs (a fragment of Nx)"
    ]
  end

  def application, do: [extra_applications: [:logger]]

  defp deps do
    case System.get_env("VAPOR_ECO") do
      nil ->
        [{:vapor, path: "../.."}, {:nx, "~> 1.0"}]

      eco ->
        [{:vapor, path: "../.."},
         {:nx, path: Path.join(eco, "nx/nx"), override: true},
         {:complex, path: Path.join(eco, "complex"), override: true},
         {:telemetry, path: Path.join(eco, "telemetry"), override: true}]
    end
  end
end
