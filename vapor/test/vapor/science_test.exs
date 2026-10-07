defmodule Vapor.ScienceTest do
  @moduledoc """
  The science laboratory (docs/CIENCIA.md): every experiment against its
  closed-form or published reference, with the control that must fail —
  quantum (coherent state, tunneling), relativity (gyration with γ, E×B),
  a tokamak equilibrium (Solov'ev), Hartree–Fock (H₂, HeH⁺), a
  Lennard-Jones liquid, Wright–Fisher fixation, phylogenetics, HP folding.
  """
  use ExUnit.Case, async: true
  @moduletag timeout: 900_000

  for name <- Vapor.Science.experiments() do
    test "#{name}" do
      r = Vapor.Science.run(unquote(name))
      assert r.pass, "#{r.name}: value #{inspect(r.value)}, reference #{inspect(r.reference)}, control #{inspect(r.control)} — #{r.threshold}"
    end
  end
end
