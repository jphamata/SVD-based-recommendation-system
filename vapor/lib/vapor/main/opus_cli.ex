defmodule Vapor.Main.OpusCli do
  @moduledoc false
  # vapor rebis | aludel | tabula | cupel | amalgam — the 0.15 desks from the terminal (docs/CLI.md §0.15)
  import Vapor.Main
  alias Vapor.Console.Lab15

  # ------------------------------------------------------------------ rebis

  def rebis(argv) do
    case opts(argv, [spec: :string, qubits: :integer, seed: :integer]) do
      :usage -> 2
      {:ok, _o, []} -> out(rebis_help()); 0
      {:ok, o, ["equiv", a, b]} -> with2(a, b, &done(Lab15.rebis(%{"op" => "equivalent", "a" => &1, "b" => &2}), o, :rebis))
      {:ok, o, ["anf", a]} -> with1(a, &done(Lab15.rebis(%{"op" => "anf", "a" => &1}), o, :anf))
      {:ok, _o, ["aiger", a]} -> with1(a, fn t -> case Lab15.rebis(%{"op" => "aiger", "a" => t}) do {:ok, r} -> out(String.trim_trailing(r.aiger)); 0; {:error, e} -> err("rebis: " <> e); 3 end end)
      {:ok, o, ["identity", a]} ->
        if o[:spec], do: with1(a, &done(Lab15.rebis(%{"op" => "identity", "a" => &1, "spec" => o[:spec]}), o, :rebis)), else: (err("rebis identity: --spec 'm[16] = a[8] * b[8]'"); 2)
      {:ok, o, ["stabilizer", a]} ->
        with1(a, &done(Lab15.rebis(%{"op" => "stabilizer", "a" => &1, "n" => o[:qubits] || 2, "seed" => o[:seed] || 1}), o, :stabilizer))
      {:ok, _, _} -> err(rebis_help()); 2
    end
  end

  defp rebis_help do
    bold("vapor rebis") <> "  circuits over GF(2)\n" <>
      "  vapor rebis equiv A B                       the same function? (netlist or AIGER; a proof or a counterexample)\n" <>
      "  vapor rebis anf FILE                        the algebraic normal form of each output\n" <>
      "  vapor rebis identity FILE --spec 'm[16] = a[8] * b[8]'   a word-level identity, by algebra over ℤ\n" <>
      "  vapor rebis stabilizer FILE --qubits N      a Clifford circuit, by the stabilizer tableau\n" <>
      "  vapor rebis aiger FILE                      the netlist as AIGER ASCII"
  end

  # ------------------------------------------------------------------ aludel

  def aludel(argv) do
    case opts(argv, [vars: :string, box: :string, strict: :boolean, depth: :integer]) do
      :usage -> 2
      {:ok, _o, []} -> out(aludel_help()); 0
      {:ok, o, ["decide", poly]} ->
        with {:ok, box} <- parse_box(o[:box]) do
          done(Lab15.aludel(%{"op" => "decide", "vars" => o[:vars] || "x, y", "poly" => poly, "box" => box, "sense" => if(o[:strict], do: "pos", else: "nonneg"),
                              "depth" => o[:depth] || 20}), o, :aludel)
        else
          {:error, e} -> err("aludel: " <> e); 2
        end

      # a JSON request, as the console and MCP send it (barriers, synthesis)
      {:ok, o, [file]} ->
        with1(file, fn t ->
          case Vapor.JSON.decode(t) do
            {:ok, req} when is_map(req) -> done(Lab15.aludel(req), o, :aludel)
            _ -> err("aludel: #{file} is not a JSON request (see vapor aludel)"); 3
          end
        end)

      {:ok, _, _} -> err(aludel_help()); 2
    end
  end

  defp aludel_help do
    bold("vapor aludel") <> "  polynomial claims on boxes, decided exactly\n" <>
      "  vapor aludel decide 'x^2 - x + 1/4' --vars x --box '0,1'        p ≥ 0 on the box (--strict: p > 0)\n" <>
      "  vapor aludel REQUEST.json                                         any request: {\"op\": \"barrier\", \"vars\": \"x, y\", \"field\": [...], ...}"
  end

  defp parse_box(nil), do: {:error, "--box 'lo,hi;lo,hi' (one interval per variable)"}

  defp parse_box(s) do
    pairs = s |> String.split(";", trim: true) |> Enum.map(&String.split(&1, ",", trim: true))
    if Enum.all?(pairs, &match?([_, _], &1)), do: {:ok, Enum.map(pairs, fn [a, b] -> [String.trim(a), String.trim(b)] end)}, else: {:error, "--box 'lo,hi;lo,hi'"}
  end

  # ------------------------------------------------------------------ tabula

  def tabula(argv) do
    case opts(argv, [facts: :string]) do
      :usage -> 2
      {:ok, _o, []} -> out(bold("vapor tabula FILE [--facts a,b]") <> "  a contract: antinomies with their scenarios, proofs of consistency, gaps; positions under the facts"); 0
      {:ok, o, [file]} ->
        facts = if o[:facts], do: o[:facts] |> String.split(",", trim: true) |> Map.new(&{String.trim(&1), true}), else: nil
        with1(file, &done(Lab15.tabula(%{"text" => &1, "facts" => facts}), o, :tabula))
      {:ok, _, _} -> 2
    end
  end

  # ------------------------------------------------------------------ cupel

  def cupel(argv) do
    case opts(argv, [n: :integer, k: :integer, seed: :integer, trials: :integer, bit: :integer]) do
      :usage -> 2
      {:ok, o, []} -> done(Lab15.cupel(o |> Keyword.drop([:json]) |> Map.new(fn {k, v} -> {to_string(k), v} end)), o, :cupel)
      {:ok, _, _} -> err("vapor cupel [--n N --k K --seed S --trials T --bit B]"); 2
    end
  end

  # ------------------------------------------------------------------ amalgam

  def amalgam(argv) do
    case opts(argv, [f32: :boolean]) do
      :usage -> 2
      {:ok, o, rest} ->
        with1(List.first(rest) || "-", &done(Lab15.amalgam(%{"numbers" => &1, "format" => if(o[:f32], do: "f32", else: "f64")}), o, :amalgam))
    end
  end

  # ------------------------------------------------------------------ shared

  defp with1(path, f) do
    case read_input(path) do
      {:ok, t} -> f.(t)
      {:error, e} -> err(e); 3
    end
  end

  defp with2(a, b, f) do
    with {:ok, ta} <- read_input(a), {:ok, tb} <- read_input(b) do
      f.(ta, tb)
    else
      {:error, e} -> err(e); 3
    end
  end

  # 0 positive · 1 negative · 3 bad input
  defp done({:error, e}, _o, desk), do: (err("#{desk}: #{e}"); 3)

  defp done({:ok, r}, o, desk) do
    if json?(o), do: emit_json(r), else: human(desk, r)
    code(desk, r)
  end

  defp code(_desk, %{verdict: v}) when v in ["equivalent", "proved", "certified", "consistent"], do: 0
  defp code(_desk, %{verdict: _}), do: 1
  defp code(_desk, _), do: 0

  defp human(:rebis, %{verdict: "equivalent"} = r), do: out(good("equivalent") <> "  " <> dim("#{r.evidence.method} · #{r.ms} ms") <> "\n" <> fmt_map(r.evidence))
  defp human(:rebis, %{verdict: "different"} = r) do
    out(bad("different") <> "  " <> dim("found by #{r.method} · #{r.ms} ms"))
    out("  inputs: " <> words(r.counterexample))
    diff = for {k, v} <- r.a_out, r.b_out[k] != v, do: "#{k}: #{v} vs #{r.b_out[k]}"
    out("  outputs that differ: " <> Enum.join(diff, ", "))
  end
  defp human(:rebis, %{verdict: "proved"} = r), do: out(good("proved") <> "  " <> r.spec <> dim("  (#{r.stats.substitutions} substitutions, peak #{r.stats.peak_terms} terms, #{r.stats.ms} ms)"))
  defp human(:rebis, %{verdict: "refuted"} = r), do: out(bad("refuted") <> "  " <> r.spec <> "  — the two sides differ by #{r.value} at the inputs at 1: " <> Enum.join(for({k, 1} <- r.counterexample, do: k), " "))
  defp human(:rebis, r), do: out(warn(r.verdict) <> "  " <> to_string(r[:why]))
  defp human(:anf, r), do: Enum.each(r.outputs, &out(bold(&1.output) <> dim("  degree #{&1.degree}, #{&1.terms} terms") <> "\n  " <> &1.text))
  defp human(:stabilizer, r) do
    out(bold("#{r.measured} measurements") <> "  " <> Enum.join(r.outcomes, ""))
    out(dim("random: #{Enum.count(r.kinds, &(&1 == :random))}, deterministic: #{Enum.count(r.kinds, &(&1 == :deterministic))}"))
    if r.stabilizers, do: out("stabilizers: " <> Enum.join(r.stabilizers, " "))
  end
  defp human(:aludel, %{verdict: "certified"} = r), do: out(good("certified") <> "  " <> dim("#{r.cells} cells, depth #{r.depth}, witness #{r.witness.bits} bits, replayed: #{r.replayed}"))
  defp human(:aludel, %{verdict: "refuted"} = r), do: out(bad("refuted") <> "  at " <> fmt_map(r.point) <> " the value is #{r.value}")
  defp human(:aludel, %{verdict: "exhausted"} = r), do: out(warn("exhausted") <> "  the budget ran out on the cell " <> inspect(r.cell) <> dim("  (#{r.cells} cells)"))
  defp human(:aludel, %{conditions: cs} = r) do
    out((if r.verdict == "proved", do: good("proved"), else: warn(r.verdict)) <> "  B = " <> r.barrier)
    Enum.each(cs, fn c -> out("  #{if c.result.verdict == "certified", do: good("✓"), else: bad("✗")} #{c.claim} — #{c.result.verdict}") end)
  end
  defp human(:aludel, r), do: out(fmt_map(r))
  defp human(:tabula, r) do
    out((if r.verdict == "consistent", do: good("consistent"), else: bad("antinomies")) <> dim("  #{length(r.clauses)} clauses, #{length(r.checked_pairs)} pairs proved never to clash"))
    Enum.each(r.findings, fn f -> out("  #{bad("✗")} #{Enum.join(f.clauses, " × ")} (#{f.party}): #{f.why}\n     when " <> scenario(f.scenario)) end)
    Enum.each(r.resolved, fn f -> out("  #{good("✓")} #{Enum.join(f.clauses, " × ")}: #{f.why} — #{f.prevails} prevails") end)
    Enum.each(r.silences, fn s -> out("  #{warn("…")} nothing governs #{s.party} #{s.action} when " <> scenario(s.scenario)) end)
    if r.positions, do: positions(r.positions)
  end
  defp human(:cupel, r) do
    out(bold("bit  detected"))
    Enum.each(r.profile, fn p -> out(String.pad_leading("#{p.bit}", 3) <> "  " <> String.duplicate("█", p.detected) <> String.duplicate("·", p.trials - p.detected) <> " #{p.detected}/#{p.trials}") end)
    out(dim("int8: #{r.int8.bits_detected}/32 bits caught (exact); #{r.check_cost}"))
  end
  defp human(:amalgam, r) do
    Enum.each(r.naive, fn n -> out(String.pad_trailing(n.order, 16) <> n.value <> dim("  #{n.bits}")) end)
    out(bold(String.pad_trailing("amalgam", 16) <> r.amalgam.value) <> dim("  #{r.amalgam.bits} — the exact sum #{r.exact}, rounded once"))
    out(dim("#{r.distinct_naive} different results from #{length(r.naive)} orders of the same #{r.count} numbers"))
  end

  # ports named prefix0, prefix1, … read as words (a = 0xDEADBEEF); the rest as name=bit
  defp words(asg) do
    {indexed, single} =
      Enum.split_with(asg, fn {k, _} -> Regex.match?(~r/^[A-Za-z_]+\d+$/, k) end)

    ws =
      indexed
      |> Enum.group_by(fn {k, _} -> String.replace(k, ~r/\d+$/, "") end)
      |> Enum.sort()
      |> Enum.map(fn {p, bits} ->
        v = Enum.reduce(bits, 0, fn {k, b}, acc -> Bitwise.bor(acc, Bitwise.bsl(b, String.to_integer(String.replace_prefix(k, p, "")))) end)
        "#{p} = 0x#{Integer.to_string(v, 16)}"
      end)

    Enum.join(ws ++ Enum.map(Enum.sort(single), fn {k, v} -> "#{k}=#{v}" end), ", ")
  end

  defp positions(%{error: why}), do: out(bad("facts: ") <> why)

  defp positions(p) do
    norm = fn n -> "#{n.id}: #{n.party} #{Vapor.Tabula.modality_text(n.modality)} #{n.action}" <> if(n[:counterparty], do: " (to #{n.counterparty})", else: "") end
    out(bold("\nin force"))
    Enum.each(p.active, &out("  " <> norm.(&1)))
    Enum.each(p.overridden, &out(dim("  #{&1} — overridden")))
    Enum.each(p.claims, &out("  #{&1.holder} may claim from #{&1.against}: #{&1.action}" <> dim(" (#{&1.from})")))
    Enum.each(p.clashes, &out("  #{bad("✗")} #{Enum.join(&1.clauses, " × ")}: #{&1.why}"))
  end

  defp scenario(sc), do: sc |> Enum.filter(fn {_, v} -> v end) |> Enum.map(&elem(&1, 0)) |> then(&if(&1 == [], do: "no fact holds", else: Enum.join(&1, " and ")))
  defp fmt_map(m), do: m |> Enum.map_join(", ", fn {k, v} -> "#{k}: #{show(v)}" end)
  defp show(v) when is_binary(v), do: v
  defp show(v) when is_atom(v), do: Atom.to_string(v)
  defp show(v), do: inspect(v)
end
