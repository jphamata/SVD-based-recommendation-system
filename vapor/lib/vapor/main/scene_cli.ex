defmodule Vapor.Main.SceneCli do
  @moduledoc false
  # vapor scene — scenes as documents, edited by operations in a pipeline
  #   vapor scene new [IMAGE] [--w W --h H]       > s.json
  #   vapor scene edit s.json 'add circle sun {…}' > s2.json   (or ops from stdin with -)
  #   vapor scene direct s.json "night, rain"      > s3.json   (a model, or the vocabulary)
  #   vapor scene export s.json                    > s.html
  #   vapor scene card | ops TEXT
  import Vapor.Main
  alias Vapor.Scene.Ops

  def run(argv) do
    case opts(argv, [w: :integer, h: :integer, model: :string, title: :string, bg: :string, seed: :integer]) do
      :usage -> 2
      {:ok, o, ["new" | rest]} -> new(o, rest)
      {:ok, _o, ["edit", file | ops]} -> edit(file, ops)
      {:ok, o, ["direct", file | words]} -> direct(o, file, Enum.join(words, " "))
      {:ok, o, ["export", file | _]} -> export(o, file)
      {:ok, _o, ["card" | _]} -> out(Ops.card()); 0
      {:ok, _o, ["ops" | text]} ->
        {:ok, ops, probs} = Ops.parse(Enum.join(text, " "))
        emit_json(%{ops: ops, problems: probs})
        if probs == [], do: 0, else: 1
      _ -> err("usage: vapor scene new [IMAGE] | edit FILE OPS… | direct FILE WORDS… | export FILE | card"); 2
    end
  end

  defp new(o, []) do
    bg = if o[:bg], do: String.split(o[:bg], ","), else: ["#0b1424", "#2a3550"]
    emit_json(Ops.blank(w: o[:w] || 960, h: o[:h] || 600, bg: bg, seed: o[:seed] || 1))
    0
  end

  defp new(_o, [image | _]) do
    with {:ok, bytes} <- read_input(image),
         {:ok, s} <- Vapor.Console.Lab11.scene_analyze(Path.basename(image), Base.encode64(bytes)) do
      emit_json(s |> Map.drop([:ms]) |> Map.put(:ops, []) |> Map.put(:seed, 1))
      0
    else
      {:error, e} -> err("scene: #{inspect(e)}"); 3
    end
  end

  defp load(file) do
    with {:ok, t} <- read_input(file) do
      case Vapor.JSON.decode(String.trim(t)) do
        {:ok, %{} = s} -> {:ok, s}
        _ -> {:error, "#{file}: not a scene (JSON object)"}
      end
    end
  end

  defp edit(file, ops) do
    with {:ok, scene} <- load(file) do
      text = Enum.join(ops, "\n")
      {:ok, s2, probs} = Ops.apply_text(scene, text)
      Enum.each(probs, &err(warn("scene: " <> &1)))
      emit_json(s2)
      if probs == [], do: 0, else: 1
    else
      {:error, e} -> err("scene: " <> e); 3
    end
  end

  defp direct(o, file, words) do
    with {:ok, scene} <- load(file) do
      mind = case o[:model] do nil -> Vapor.Mind.from_env(); m -> elem(Vapor.Mind.parse(m), 1) end
      {ops, probs, via} =
        case mind && Vapor.Mind.direct(mind, scene, words) do
          {:ok, r} -> {r.ops, r.problems, "model"}
          _ ->
            r = Vapor.Scene.direct(words)
            {r.ops, Enum.map(r.unknown, &"not understood: #{&1}"), "vocabulary"}
        end
      err(dim("scene: directed by the #{via}; #{length(ops)} operation(s)"))
      Enum.each(probs, &err(warn("scene: " <> &1)))
      emit_json(Map.update(scene, "ops", ops, &(&1 ++ ops)))
      0
    else
      {:error, e} -> err("scene: " <> e); 3
    end
  end

  defp export(o, file) do
    with {:ok, scene} <- load(file) do
      out(Vapor.Scene.standalone(Vapor.JSON.encode(scene), o[:title] || "vapor — scene"))
      0
    else
      {:error, e} -> err("scene: " <> e); 3
    end
  end
end
