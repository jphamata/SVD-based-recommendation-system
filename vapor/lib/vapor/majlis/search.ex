defmodule Vapor.Majlis.Search do
  @moduledoc """
  Search across every conversation: BM25 (k₁ = 1.2, b = 0.75) over the words
  of every message — any script, by Unicode letter and number classes — and,
  for each hit, the threads whose current path holds it.
  """

  @doc "`[%{id, role, score, snippet, threads}]`, best first."
  def run(nodes, threads, path_fun, query, k) do
    q = terms(query) |> Enum.uniq()

    if q == [] do
      []
    else
      docs = for {id, n} <- nodes, do: {id, n, terms(n["content"])}
      n_docs = max(length(docs), 1)
      avg = docs |> Enum.map(&length(elem(&1, 2))) |> Enum.sum() |> Kernel./(n_docs)
      df = for t <- q, into: %{}, do: {t, Enum.count(docs, fn {_, _, ts} -> t in ts end)}

      on_paths =
        for {tid, t} <- threads, id <- path_fun.(t["head"]), reduce: %{} do
          acc -> Map.update(acc, id, [tid], &[tid | &1])
        end

      docs
      |> Enum.map(fn {id, n, ts} ->
        freq = Enum.frequencies(ts)
        len = length(ts)

        score =
          Enum.reduce(q, 0.0, fn t, acc ->
            f = Map.get(freq, t, 0)
            idf = :math.log(1 + (n_docs - df[t] + 0.5) / (df[t] + 0.5))
            acc + if(f == 0, do: 0.0, else: idf * f * 2.2 / (f + 1.2 * (1 - 0.75 + 0.75 * len / max(avg, 1.0))))
          end)

        {id, n, score}
      end)
      |> Enum.filter(fn {_, _, s} -> s > 0 end)
      |> Enum.sort_by(fn {id, _, s} -> {-s, id} end)
      |> Enum.take(k)
      |> Enum.map(fn {id, n, s} -> %{id: id, role: n["role"], score: Float.round(s, 4), snippet: snippet(n["content"], q), threads: Enum.sort(Map.get(on_paths, id, []))} end)
    end
  end

  @doc "The words of a text, lowercased."
  def terms(text), do: Regex.scan(~r/[\p{L}\p{N}]+/u, String.downcase(to_string(text))) |> List.flatten()

  defp snippet(text, q) do
    lower = String.downcase(text)

    at =
      Enum.find_value(q, 0, fn t ->
        case :binary.match(lower, t) do
          {pos, _} -> pos
          :nomatch -> nil
        end
      end)

    start = max(at - 60, 0)
    # cut on a character boundary
    pre = binary_part(text, 0, start) |> String.length()
    String.slice(text, pre, 180)
  end
end
