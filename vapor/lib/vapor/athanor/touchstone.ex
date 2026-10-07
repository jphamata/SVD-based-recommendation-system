defmodule Vapor.Athanor.Touchstone do
  @moduledoc """
  **Touchstone** — the stone on which gold was rubbed to tell it from brass:
  the independent check of an Athanor certificate (docs/ATHANOR.md §5).

  It does not trust the search, the strategies, the model or the person
  who proposed: it reads the problem again, puts the claimed candidate back
  through the space's membership test and the verifier, and compares. On
  request it also re-runs the exhaustive enumeration behind an
  "optimal" or "proved" verdict, or replays the whole search from its seed
  and outside proposals and compares the journal's root hash.
  """
  alias Vapor.Alembic
  alias Vapor.Athanor
  alias Vapor.Athanor.{Space, Spec}

  @doc """
  Check `cert` (a certificate map, string keys accepted) against the
  problem `text`. Options: `full: true` re-runs the enumeration behind an
  optimality or proof claim; `replay: true` re-runs the search and compares
  the journal root. `%{verified, checks: [%{check, ok, detail}]}`.
  """
  def verify(text, cert, opts \\ []) do
    cert = keys(cert)
    with {:ok, spec} <- Spec.parse(text, seed: cert["seed"], budget: cert["evaluations"]) do
      checks =
        [hash_check(spec, cert)] ++ best_checks(spec, cert) ++
          if(Keyword.get(opts, :full), do: [full_check(spec, cert)], else: []) ++
          if(Keyword.get(opts, :replay), do: [replay_check(text, cert)], else: [])

      checks = Enum.reject(checks, &is_nil/1)
      {:ok, %{verified: Enum.all?(checks, & &1.ok), checks: checks, verdict: cert["verdict"]}}
    end
  end

  defp keys(m) when is_map(m), do: Map.new(m, fn {k, v} -> {to_string(k), keys(v)} end)
  defp keys(l) when is_list(l), do: Enum.map(l, &keys/1)
  defp keys(v), do: v

  defp hash_check(spec, cert) do
    same = cert["spec_hash"] == spec.hash
    %{check: "problem", ok: same, detail: if(same, do: "the problem text is the one the certificate was made for", else: "the problem text differs from the one certified (hash #{String.slice(to_string(cert["spec_hash"]), 0, 12)}…)")}
  end

  defp best_checks(_spec, %{"best" => nil} = cert) do
    if cert["reason"] == "exhausted", do: [], else: [%{check: "best", ok: true, detail: "no candidate claimed"}]
  end

  defp best_checks(spec, %{"best" => best} = cert) do
    text = best["candidate"]
    parsed = if spec.space.kind == :program, do: Space.parse_program(spec.space, text), else: with({:ok, v} <- Alembic.literal(text), do: Space.check(spec.space, v))

    case parsed do
      {:error, m} -> [%{check: "membership", ok: false, detail: "the claimed candidate is not in the space: #{m}"}]
      {:ok, x} ->
        member = %{check: "membership", ok: true, detail: "#{text} belongs to #{spec.space.text}"}
        r = Athanor.evaluate(spec, %{x: x, key: text})
        [member, value_check(spec, cert, best, r)]
    end
  end

  defp best_checks(_spec, _cert), do: [%{check: "best", ok: false, detail: "the certificate names no best candidate"}]

  defp value_check(%{sense: :claim}, _cert, _best, r) do
    ok = r.status == :ok and r.refutes
    %{check: "counterexample", ok: ok, detail: if(ok, do: "claim(x) is false at the candidate: it refutes the claim", else: "claim(x) is not false at the candidate")}
  end

  defp value_check(_spec, _cert, best, r) do
    claimed = best["value"]
    cond do
      r.status != :ok -> %{check: "value", ok: false, detail: "the candidate is #{r.status} under the verifier#{if r[:error], do: ": " <> r.error, else: ""}"}
      claimed == nil -> %{check: "value", ok: true, detail: "valid under the verifier"}
      same?(r.value, claimed) -> %{check: "value", ok: true, detail: "the verifier gives #{Alembic.show(r.value)}, as claimed"}
      true -> %{check: "value", ok: false, detail: "the verifier gives #{Alembic.show(r.value)}, the certificate claims #{inspect(claimed)}"}
    end
  end

  defp same?(a, b) when is_number(a) and is_number(b), do: a == b or abs(a - b) <= 1.0e-12 * max(abs(a), abs(b))
  defp same?(a, b), do: a == b

  defp full_check(spec, cert) do
    if cert["reason"] != "exhausted" do
      %{check: "exhaustive", ok: true, detail: "no optimality or proof claim to re-check"}
    else
      size = spec.space.size
      spec = %{spec | budget: size}
      results = spec.space |> Space.enumerate() |> Stream.map(fn x -> Athanor.evaluate(spec, %{x: Space.canon(spec.space, x), key: nil}) end)
      ok = Enum.filter(results, &(&1.status == :ok))
      count = Enum.count(results)

      case spec.sense do
        :claim ->
          refuted = Enum.find(ok, & &1.refutes)
          %{check: "exhaustive", ok: refuted == nil, detail: if(refuted, do: "re-enumeration found a counterexample", else: "re-enumerated #{count} candidates: the claim holds for all")}

        :find ->
          %{check: "exhaustive", ok: ok == [], detail: "re-enumerated #{count} candidates: #{length(ok)} valid"}

        _ ->
          best = Enum.max_by(ok, & &1.fitness, fn -> nil end)
          claimed = cert["best"]["value"]
          good = best != nil and same?(best.value, claimed)
          %{check: "exhaustive", ok: good, detail: "re-enumerated #{count} candidates: the optimum is #{best && Alembic.show(best.value)}#{if good, do: ", as claimed", else: ", not #{inspect(claimed)}"}"}
      end
    end
  end

  defp replay_check(text, cert) do
    injected = Enum.reject(cert["outside_proposals"] || [], &(to_string(&1["source"]) == "start"))
    {:ok, spec} = Spec.parse(text, seed: cert["seed"], budget: cert["evaluations"])
    r = Athanor.init(spec, control: false)
    by_round = Enum.group_by(injected, & &1["round"])
    r = replay_rounds(r, by_round)
    same = r.journal == cert["journal_root"]
    %{check: "replay", ok: same, detail: if(same, do: "the search, replayed from its seed and outside proposals, gives the same journal root", else: "the replayed journal root differs (#{String.slice(r.journal, 0, 16)}…)")}
  end

  defp replay_rounds(%{status: :running} = r, by_round) do
    r =
      case Map.get(by_round, r.round) do
        nil -> r
        props ->
          vals = Enum.map(props, &parse_candidate(r.spec, &1["candidate"])) |> Enum.reject(&is_nil/1)
          sources = props |> Enum.map(&(&1["source"] |> to_string())) |> hd()
          Athanor.propose(r, vals, source_atom(sources))
      end
    replay_rounds(Athanor.step(r, 1), Map.delete(by_round, r.round - 1))
  end

  defp replay_rounds(r, _), do: r

  defp source_atom("mind"), do: :mind
  defp source_atom("start"), do: :start
  defp source_atom(_), do: :human

  defp parse_candidate(spec, text) do
    if spec.space.kind == :program do
      case Space.parse_program(spec.space, text) do {:ok, t} -> t; _ -> nil end
    else
      case Alembic.literal(text) do {:ok, v} -> v; _ -> nil end
    end
  end
end
