defmodule Vapor.Main.ScienceCli do
  @moduledoc false
  # vapor crucible — open science from the terminal
  import Vapor.Main
  alias Vapor.Crucible

  def run(argv) do
    case opts(argv, [example: :boolean]) do
      :usage -> 2
      {:ok, _o, []} ->
        out(bold("vapor crucible KIND [FILE|-]") <> dim("   (--example prints a starting point)"))
        Enum.each(Crucible.kinds(), fn k -> out("  #{String.pad_trailing(k.kind, 12)} #{k.about}") end)
        0
      {:ok, o, [kind | rest]} ->
        cond do
          Crucible.example(kind) == nil -> err("crucible: unknown domain #{kind} (vapor crucible lists them)"); 2
          o[:example] -> out(Crucible.example(kind)); 0
          true ->
            case read_input(List.first(rest)) do
              {:ok, text} ->
                case Crucible.run(kind, text) do
                  {:ok, r} ->
                    if json?(o), do: emit_json(r), else: out(render(r))
                    if Enum.all?(r[:evidence] || [], & &1.ok), do: 0, else: 1
                  {:error, e} -> err("crucible #{kind}: #{e}"); 3
                end
              {:error, e} -> err("crucible: " <> e); 3
            end
        end
    end
  end

  defp render(r) do
    head = bold("crucible · #{r.kind}") <> dim("  #{r[:ms]} ms")
    says = r[:says] || ""
    extra =
      case r.kind do
        "laws" -> Enum.map(r.laws, &("  " <> good("conserved ") <> &1.law <> dim("   #{&1.status}")))
        "quantum" -> Enum.map(r.states, fn s -> "  E#{s.index} = #{fmt(s.energy)} " <> dim("± #{fmt(s.error_estimate)}  order #{fmt(s.observed_order)}") end)
        "phylogeny" -> ["  " <> r.newick | Enum.map(r.splits, fn sp -> "  " <> dim("#{round(sp.support * 100)} %  ") <> Enum.join(sp.taxa, " ") end)]
        "regress" -> Enum.map(r.front, fn p -> "  " <> dim("size #{p.size}  R² #{fmt(p.test_r2)}  ") <> p.expression end)
        "hamiltonian" -> Enum.map(r.equations, &("  " <> &1)) ++ Enum.map(r.laws || [], &("  " <> good("conserved ") <> &1.law))
        "molecule" -> ["  orbital energies: " <> Enum.map_join(r.orbital_energies, ", ", &fmt/1)]
        _ -> []
      end
    ev = Enum.map(r[:evidence] || [], fn e -> "  #{if e.ok, do: good("✓"), else: bad("✗")} #{bold(e.check)}: #{e.detail}" end)
    Enum.join([head, says] ++ extra ++ ev, "\n")
  end

  defp fmt(nil), do: "—"
  defp fmt(x) when is_float(x), do: :erlang.float_to_binary(x, [{:decimals, 6}, :compact])
  defp fmt(x), do: to_string(x)
end
