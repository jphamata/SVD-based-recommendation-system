defmodule Vapor.Main.ForgeCli do
  @moduledoc false
  # vapor logic | qalib | recommend | palingenesis — the 0.17 verbs (docs/CLI.md §0.17)
  import Vapor.Main
  alias Vapor.{Logic, Palingenesis, Qalib, Recommend}

  # ------------------------------------------------------------------ logic

  def logic(argv) do
    case opts(argv, []) do
      :usage -> 2
      {:ok, _o, []} -> out(logic_help()); 0
      {:ok, o, ["check", claim, proposal]} ->
        with {:ok, t} <- read_input(claim), {:ok, pj} <- read_input(proposal), {:ok, prop} <- json_map(pj) do
          case Logic.check(t, prop) do
            {:ok, r} -> report(r, o, r.accepted)
            {:error, why} -> err("logic: " <> to_string(why)); 3
          end
        else
          {:error, why} -> err("logic: " <> why); 3
        end

      {:ok, o, [file]} ->
        with {:ok, t} <- read_input(file) do
          case Logic.run(t) do
            {:ok, r} -> report(r, o, positive?(r))
            {:error, why} -> err("logic: " <> to_string(why)); 3
          end
        else
          {:error, why} -> err("logic: " <> why); 3
        end

      {:ok, _, _} -> err(logic_help()); 2
    end
  end

  defp logic_help do
    bold("vapor logic FILE") <> "  a claim settled with its certificate: SAT/DRUP, formulas, combinatorics, rewriting,\n" <>
      "                   Gröbner, linear (exact simplex), integer linear (branch and bound), causal (ID algorithm)\n" <>
      "  vapor logic check FILE PROPOSAL.json   anyone's proposal, decided by the checker alone"
  end

  defp json_map(text) do
    case Vapor.JSON.decode(text) do
      {:ok, m} when is_map(m) -> {:ok, m}
      _ -> {:error, "the proposal is not a JSON object"}
    end
  end

  # positive: proved, optimal, satisfiable, identifiable… negative: refuted, infeasible, not identifiable
  defp positive?(%{verdict: v}) when is_binary(v), do: v in ["optimal", "unbounded", "sat", "valid", "satisfiable", "equivalent", "proved", "complete"] or String.starts_with?(v, "proved")
  defp positive?(%{kind: "causal", results: rs}), do: Enum.all?(rs, &(&1.verdict in ["identifiable", "valid", "d-separated"]))
  defp positive?(_), do: true

  defp report(r, o, positive) do
    if json?(o), do: emit_json(r), else: out(Vapor.JSON.encode(jsonable(r)))
    if positive, do: 0, else: 1
  end

  # ------------------------------------------------------------------ qalib

  def qalib(argv) do
    case opts(argv, [style: :string, module: :string]) do
      :usage -> 2
      {:ok, _o, []} -> out(qalib_help()); 0

      {:ok, o, ["map", file]} ->
        style = if o[:style] == "nand", do: :nand, else: :cells

        with {:ok, t} <- read_input(file), {:ok, c} <- Qalib.circuit(t), {:ok, v, stats} <- Qalib.to_verilog(c, style: style, module: o[:module] || "vapor_top") do
          # the written netlist is read back and proved equal before it is printed
          case Qalib.certify(c, v) do
            {:equivalent, ev} -> IO.write(v); err(dim("qalib: #{stats.cells} cells #{inspect(stats.by_type)}; proved equal to the source (#{ev.method})")); 0
            other -> err("qalib: the mapped netlist is not equivalent to its source: #{inspect(other)}"); 4
          end
        else
          {:error, why} -> err("qalib: " <> to_string(why)); 3
        end

      {:ok, o, ["check", spec, impl]} ->
        with {:ok, a} <- read_input(spec), {:ok, b} <- read_input(impl) do
          case Qalib.certify(a, b) do
            {:equivalent, ev} -> emit(%{verdict: "equivalent", evidence: ev}, o, "equivalent (#{ev.method})"); 0
            {:different, d} -> emit(%{verdict: "different", counterexample: d.counterexample, spec: d.a, impl: d.b}, o, "DIFFERENT at #{inspect(d.counterexample)}"); 1
            {:unknown, why} -> err("qalib: unknown: " <> why); 4
            {:error, why} -> err("qalib: " <> to_string(why)); 3
          end
        else
          {:error, why} -> err("qalib: " <> why); 3
        end

      {:ok, _, _} -> err(qalib_help()); 2
    end
  end

  defp qalib_help do
    bold("vapor qalib") <> "  netlists to sky130 cells and back, every step proved\n" <>
      "  vapor qalib map FILE [--style cells|nand] [--module NAME]   a circuit (Rebis, AIGER, BLIF, Verilog) as sky130_fd_sc_hd Verilog\n" <>
      "  vapor qalib check SPEC IMPL                                 the same function? (exit 0) or a counterexample (exit 1)"
  end

  defp emit(v, o, line), do: if(json?(o), do: emit_json(v), else: out(line))

  # -------------------------------------------------------------- recommend

  def recommend(argv) do
    case opts(argv, [top: :integer, seed: :integer, test: :float]) do
      :usage -> 2
      {:ok, _o, []} -> out(bold("vapor recommend RATINGS.csv [--top N] [--seed S] [--test 0.2]") <> "  matrix factorisation with baselines, a paired test and a shuffled control"); 0

      {:ok, o, [file]} ->
        with {:ok, t} <- read_input(file), {:ok, d} <- Recommend.parse_csv(t),
             {:ok, r} <- Recommend.evaluate(d, Keyword.take(o, [:top, :seed, :test])) do
          if json?(o) do
            emit_json(r)
          else
            out(bold("#{r.verdict}") <> "  rank #{r.rank}, λ #{r.lambda}, #{r.train} training / #{r.test} test ratings")
            out("  test RMSE: model #{f(r.rmse.model)} · biases #{f(r.rmse.biases)} · mean #{f(r.rmse.mean)} · shuffled control #{f(r.rmse.shuffled_control)}")
            out("  paired against the biases: #{r.paired.better}/#{r.paired.of} ratings better, p = #{f(r.paired.p_value)}")
            for {u, items} <- Enum.sort(r.recommendations), items != [], do: out("  #{u}: " <> Enum.map_join(items, ", ", fn {i, s} -> "#{i} (#{f(s)})" end))
          end

          if r.verdict == "signal", do: 0, else: 1
        else
          {:error, why} -> err("recommend: " <> to_string(why)); 3
        end

      {:ok, _, _} -> err("vapor recommend RATINGS.csv"); 2
    end
  end

  defp f(x) when is_float(x), do: :erlang.float_to_binary(x, [{:decimals, 4}, :compact])
  defp f(x), do: to_string(x)

  # ----------------------------------------------------------- palingenesis

  def palingenesis(argv) do
    case opts(argv, [plank: :string, from: :string, anchors: :string, targets: :string, epsilon: :float, out: :string, blend: :float]) do
      :usage -> 2
      {:ok, _o, []} -> out(palingenesis_help()); 0

      {:ok, o, ["planks", dir]} ->
        if jailed?(), do: (err("palingenesis: models are not read in the console's terminal"); 2), else: planks(dir, o)

      {:ok, o, ["try", dir]} ->
        cond do
          jailed?() -> err("palingenesis: models are not read in the console's terminal"); 2
          o[:plank] == nil or o[:from] == nil or o[:anchors] == nil -> err(palingenesis_help()); 2
          true -> try_plank(dir, o)
        end

      {:ok, _, _} -> err(palingenesis_help()); 2
    end
  end

  defp palingenesis_help do
    bold("vapor palingenesis") <> "  renew a model plank by plank, through the gates\n" <>
      "  vapor palingenesis planks MODEL                     the planks and their Merkle roots\n" <>
      "  vapor palingenesis try MODEL --plank P --from DONOR --anchors A.txt [--targets T.txt] [--epsilon 0.05]\n" <>
      "                         [--blend t] [--out DIR]     DONOR's plank P through the drift brake (anchors: one text per line)\n" <>
      "                                                     and the target test; --out writes the accepted model and its lineage"
  end

  defp planks(dir, o) do
    case Vapor.Lock.open(dir) do
      {:ok, m} ->
        ps = Palingenesis.planks(m.weights)
        rows = for {p, names} <- Enum.sort(ps), do: %{plank: p, tensors: length(names), root: Base.encode16(Palingenesis.plank_root(m.weights, names), case: :lower)}
        if json?(o), do: emit_json(%{root: Palingenesis.root(m.weights, ps), planks: rows}), else: Enum.each(rows, &out("#{String.pad_trailing(&1.plank, 44)} #{&1.tensors}  #{String.slice(&1.root, 0, 16)}…"))
        0

      {:error, why} -> err("palingenesis: #{inspect(why)}"); 3
    end
  end

  defp try_plank(dir, o) do
    with {:ok, m} <- Vapor.Lock.open(dir),
         {:ok, donor} <- Vapor.Lock.open(o[:from]),
         {:ok, names} <- Map.fetch(Palingenesis.planks(m.weights), o[:plank]) |> then(&if(&1 == :error, do: {:error, "no plank #{o[:plank]}"}, else: &1)),
         {:ok, anchors} <- lines(o[:anchors], m.tokenizer),
         {:ok, targets} <- (if o[:targets], do: lines(o[:targets], m.tokenizer), else: {:ok, []}) do
      name = {:cli, System.unique_integer()}
      {:ok, _} = Palingenesis.launch(name, %{spec: m.spec, weights: m.weights})
      mode = if o[:blend], do: {:blend, o[:blend]}, else: :replace
      res = Palingenesis.propose(name, o[:plank], Map.take(donor.weights, names), anchors: anchors, targets: targets, epsilon: o[:epsilon] || 0.05, mode: mode)
      Palingenesis.retire(name)

      case res do
        {:ok, g, report} ->
          if o[:out], do: write(o[:out], dir, g)
          emit(%{verdict: "admitted", generation: g.gen, root: g.root, drift: report.drift, target: report.target}, o,
               "admitted: drift max #{f(report.drift.max)} ≤ ε #{f(report.drift.epsilon)}" <> if(report.target, do: "; target #{f(report.target.bits_old)} → #{f(report.target.bits_new)} bits/token, p = #{f(report.target.p_value)}", else: ""))
          0

        {:error, report} ->
          emit(Map.put(report, :verdict, "refused"), o, "refused at the #{report.gate} gate: #{report.reason}")
          1
      end
    else
      {:error, why} -> err("palingenesis: #{if is_binary(why), do: why, else: inspect(why)}"); 3
    end
  end

  defp lines(path, tk) do
    with {:ok, t} <- read_input(path) do
      seqs = t |> String.split("\n", trim: true) |> Enum.map(&Vapor.Tokenizer.encode(tk, &1, add_bos: false)) |> Enum.filter(&(length(&1) >= 2))
      if seqs == [], do: {:error, "#{path}: no line long enough to measure"}, else: {:ok, seqs}
    end
  end

  defp write(out_dir, src, g) do
    File.mkdir_p!(out_dir)
    tensors = g.weights |> Enum.filter(fn {k, _} -> is_binary(k) end) |> Map.new()
    Vapor.Ingest.Safetensors.write(Path.join(out_dir, "model.safetensors"), tensors)
    for f <- ~w(config.json tokenizer.json tokenizer_config.json generation_config.json), File.exists?(Path.join(src, f)), do: File.cp!(Path.join(src, f), Path.join(out_dir, f))
    File.write!(Path.join(out_dir, "lineage.json"), Vapor.JSON.encode(jsonable(Enum.reverse(Enum.map(g.records, & &1.payload)))))
  end
end
