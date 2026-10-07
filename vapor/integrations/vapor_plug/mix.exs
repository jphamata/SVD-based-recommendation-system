defmodule VaporPlug.MixProject do
  use Mix.Project

  # vapor's core has no dependencies; this adapter depends on Plug (and, in
  # tests only, Bandit for a real HTTP server). With VAPOR_ECO=<dir> holding
  # source checkouts (plug, mime, plug_crypto, telemetry, bandit,
  # thousand_island, hpax, websock), it builds with no package registry.
  def project do
    [
      app: :vapor_plug,
      version: "0.3.0",
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      description: "vapor's OpenAI-compatible API as a Plug (Phoenix, Bandit, Cowboy)"
    ]
  end

  def application, do: [extra_applications: [:logger]]

  # the tests reuse vapor's tiny-model helpers
  defp elixirc_paths(:test), do: ["lib", Path.expand("../../test/support", __DIR__)]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    case System.get_env("VAPOR_ECO") do
      nil ->
        [{:vapor, path: "../.."}, {:plug, "~> 1.16"}, {:bandit, "~> 1.5", only: :test}]

      eco ->
        src = fn name -> Path.join(eco, name) end

        [{:vapor, path: "../.."},
         {:plug, path: src.("plug"), override: true},
         {:mime, path: src.("mime"), override: true},
         {:plug_crypto, path: src.("plug_crypto"), override: true},
         {:telemetry, path: src.("telemetry"), override: true},
         {:bandit, path: src.("bandit"), only: :test},
         {:thousand_island, path: src.("thousand_island"), only: :test, override: true},
         {:hpax, path: src.("hpax"), only: :test, override: true},
         {:websock, path: src.("websock"), only: :test, override: true}]
    end
  end
end
