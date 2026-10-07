defmodule Vapor.Main.MindCli do
  @moduledoc false
  # vapor mind — a language model from the terminal, always checked
  import Vapor.Main
  alias Vapor.Mind

  def run(argv) do
    case opts(argv, [model: :string, kind: :string]) do
      :usage -> 2
      {:ok, o, [cmd | rest]} -> with_model(o, fn m -> cmd(cmd, m, rest, o) end)
      {:ok, _o, []} -> err("usage: vapor mind ask QUESTION | formalize [TEXT|-] | transcript   (--model anthropic:MODEL | openai:MODEL[@URL] | script:FILE, or VAPOR_MIND)"); 2
    end
  end

  defp with_model(o, f) do
    m = case o[:model] do nil -> {:ok, Mind.from_env()}; s -> Mind.parse(s) end
    case m do
      {:ok, nil} -> err("mind: no model configured — set VAPOR_MIND (anthropic:MODEL, openai:MODEL[@URL], script:FILE) or pass --model"); 3
      {:ok, model} -> f.(model)
      {:error, e} -> err("mind: " <> e); 3
    end
  end

  defp cmd("ask", m, words, o) do
    q = if words in [[], ["-"]], do: elem(read_input("-"), 1), else: Enum.join(words, " ")
    case Mind.ask(m, q) do
      {:ok, a} -> if json?(o), do: emit_json(%{answer: a, transcript: Mind.transcript(m)}), else: out(a); 0
      {:error, e} -> err("mind: " <> e); 4
    end
  end

  defp cmd("formalize", m, words, o) do
    text = if words in [[], ["-"]], do: elem(read_input("-"), 1), else: Enum.join(words, " ")
    kind = case o[:kind] do "game" -> :game; "search" -> :search; _ -> :auto end
    case Mind.formalize(m, text, kind) do
      {:ok, r} ->
        if json?(o) do
          emit_json(Map.put(r, :transcript, Mind.transcript(m)))
        else
          out(r.program)
          err(dim("# #{r.kind}, loaded after #{r.attempts} attempt(s)"))
          if r.back_translation, do: err(dim("# read back: ") <> r.back_translation)
          Enum.each(r.notes, &err(warn("# note: " <> &1)))
        end
        0
      {:error, why, log} ->
        if json?(o), do: emit_json(%{error: why, attempts: log}), else: err("mind: " <> why)
        1
    end
  end

  defp cmd(other, _m, _w, _o), do: (err("mind: unknown command #{other} — ask or formalize"); 2)
end
