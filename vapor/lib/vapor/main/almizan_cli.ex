defmodule Vapor.Main.AlmizanCli do
  @moduledoc false
  # `vapor wzn` — Almizan from the terminal. The manifesto's ikseer verbs,
  # each mapped onto something that runs: check (distil the obligations and
  # decide them), show (a projection), hash, run, transmute (vapor's
  # compiler, a circuit, a Lean theorem), assay (binary32 against exact ℚ),
  # abjad.
  import Vapor.Main
  alias Vapor.Almizan
  alias Vapor.Almizan.{Abjad, Lower}

  @usage """
  vapor wzn check FILE [--lib DIR]          decide every obligation: proved · refuted (with the point) · unknown
  vapor wzn show FILE [--arabic|--latin]    the program in one script (the same tree, the same hash)
  vapor wzn hash FILE                       the program's identity: SHA-256 of its tree, whatever the script
  vapor wzn run FILE CLAIM [ARGS…]          evaluate (exact over ℚ, binary32 as vapor rounds it); burhān claims only once proved
  vapor wzn transmute FILE CLAIM [--to vapor|aiger|lean] [--out DIR]
                                            vapor's compiler (machine code per target) · a circuit · a Lean 4 theorem
  vapor wzn assay FILE CLAIM [--points N]   the binary32 program on vapor's oracle against exact ℚ, in ulps
  vapor wzn abjad [WORD]                    a word's abjad value, and why it cannot be an address
  FILE may be written in the Latin or the Arabic projection. Exit: 0 all proved · 1 something refuted or unknown · 3 bad input.
  """

  def run([]), do: (out(@usage); 0)
  def run(["help" | _]), do: (out(@usage); 0)

  def run([cmd | rest]) do
    case opts(rest, [lib: :string, arabic: :boolean, latin: :boolean, to: :string, out: :string, points: :integer, depth: :integer]) do
      :usage -> 2
      {:ok, o, args} -> verb(cmd, args, o)
    end
  end

  defp load(file, o) do
    with {:ok, text} <- read_input(file),
         {:ok, m} <- Almizan.parse(text) do
      lib = o[:lib] || (file not in [nil, "-"] && Path.dirname(file)) || "."
      if Enum.any?(m["decls"], &Map.has_key?(&1, "import")), do: Almizan.link(m, lib), else: {:ok, m}
    end
  end

  defp verb("check", [file], o) do
    case load(file, o) do
      {:ok, m} ->
        rs = Almizan.check(m, Keyword.take(o, [:depth]))
        if json?(o), do: emit_json(%{hash: Almizan.hash(m), claims: rs}), else: Enum.each(rs, &out(line(&1)))
        if Enum.all?(rs, &(&1.verdict in ["proved", "none"])), do: 0, else: 1

      {:error, e} -> (err("wzn: " <> e); 3)
    end
  end

  defp verb("show", [file], o) do
    with {:ok, text} <- read_input(file), {:ok, m} <- Almizan.parse(text) do
      IO.write(Almizan.print(m, if(o[:arabic], do: :arabic, else: :latin)))
      0
    else
      {:error, e} -> (err("wzn: " <> e); 3)
    end
  end

  defp verb("hash", [file], o) do
    with {:ok, text} <- read_input(file), {:ok, m} <- Almizan.parse(text) do
      h = Almizan.hash(m)
      if json?(o), do: emit_json(%{hash: h}), else: out(h)
      0
    else
      {:error, e} -> (err("wzn: " <> e); 3)
    end
  end

  defp verb("run", [file, claim | args], o) do
    with {:ok, m} <- load(file, o),
         {:ok, r} <- Almizan.run(m, claim, args) do
      v = Almizan.show(r.value)
      if json?(o), do: emit_json(Map.put(r, :value, v)), else: out(v <> if(r[:hash], do: dim("  " <> r.hash), else: ""))
      0
    else
      {:error, e} -> (err("wzn: " <> to_string(e)); 1)
    end
  rescue
    e in ArgumentError -> (err("wzn: " <> Exception.message(e)); 3)
  end

  defp verb("transmute", [file, claim], o) do
    with {:ok, m} <- load(file, o) do
      case o[:to] || "vapor" do
        "vapor" ->
          case Lower.transmute(m, claim) do
            {:ok, t} ->
              if o[:out] do
                File.mkdir_p!(o[:out])
                for {target, code} <- t.compiled.code, do: File.write!(Path.join(o[:out], "#{claim}.#{target}.bin"), flat(code))
              end

              if json?(o), do: emit_json(Map.drop(t, [:program, :compiled])), else: Enum.each(t.targets, &out("#{String.pad_trailing(&1.target, 14)} #{String.pad_leading(Integer.to_string(&1.bytes), 6)} bytes  #{dim(String.slice(&1.sha256, 0, 16))}"))
              0

            {:error, e} -> (err("wzn: #{e}"); 1)
          end

        "aiger" -> emit_text(Lower.aiger(m, claim), :aiger, o)
        "lean" -> emit_text(Lower.lean(m, claim), nil, o)
        other -> (err("wzn: --to vapor, aiger or lean, not #{other}"); 2)
      end
    else
      {:error, e} -> (err("wzn: " <> e); 3)
    end
  end

  defp verb("assay", [file, claim], o) do
    with {:ok, m} <- load(file, o), {:ok, a} <- Lower.assay(m, claim, o[:points] || 256) do
      if json?(o), do: emit_json(a), else: out("#{a.points} points · #{a.exact_points} exact · worst #{a.max_ulps} ulp · mean #{Float.round(a.mean_ulps, 3)} ulp")
      0
    else
      {:error, e} -> (err("wzn: #{e}"); 3)
    end
  end

  defp verb("abjad", words, o) do
    c = Abjad.collisions()
    vals = for w <- words, do: %{word: w, value: Abjad.value(w)}

    if json?(o) do
      emit_json(%{words: vals, roots: c})
    else
      Enum.each(vals, &out("#{&1.word}  #{&1.value}"))
      out(dim("of #{c.roots} three-letter roots, #{c.sharing} share their value with another (#{Float.round(c.fraction_sharing * 100, 2)} %); largest class #{c.largest_class} — a value, not an address"))
    end

    0
  end

  defp verb(_, _, _), do: (err(@usage); 2)

  defp emit_text({:ok, %{aiger: a}}, :aiger, _o), do: (IO.write(a); 0)
  defp emit_text({:ok, text}, nil, _o) when is_binary(text), do: (IO.write(text); 0)
  defp emit_text({:error, e}, _, _o), do: (err("wzn: #{e}"); 1)

  defp flat(%{bin: b}) when is_binary(b), do: b
  defp flat(b) when is_binary(b), do: b
  defp flat(m) when is_map(m), do: m |> Enum.sort() |> Enum.map(fn {_, v} -> flat(v) end) |> IO.iodata_to_binary()
  defp flat(_), do: ""

  defp line(r) do
    mark = case r.verdict do "proved" -> good("✓ proved"); "refuted" -> bad("✗ refuted"); "unknown" -> warn("? unknown"); _ -> dim("· no claim") end
    cex = if r[:counterexample], do: "\n    at " <> Enum.map_join(r.counterexample, ", ", fn {k, v} -> "#{k} = #{Almizan.show(v)}" end), else: ""
    "#{mark}  #{bold(r.claim)} #{dim("(#{r.root}, #{r.wazn})")}  #{r.detail}" <> if(r.decider, do: dim("  [#{r.decider}]"), else: "") <> cex
  end
end
