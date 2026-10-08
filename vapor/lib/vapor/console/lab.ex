defmodule Vapor.Console.Lab do
  @moduledoc """
  The substrate airlock's verdicts for the console (the 0.10 round's other
  laboratories — chaos, digital twin, networks, the training receipt — were
  fixed demonstrations and left the product in 0.16; DIRECTIVE §19).
  """
  alias Vapor.Runtime.Substrates

  # computed once; concurrent first requests wait for the one computing (a
  # worker holds one session: two at once would drop each other's)
  defp memo(key, f) do
    case :persistent_term.get({__MODULE__, :memo, key}, nil) do
      nil ->
        :global.trans({{__MODULE__, :lab}, self()}, fn ->
          case :persistent_term.get({__MODULE__, :memo, key}, nil) do
            nil -> v = f.(); :persistent_term.put({__MODULE__, :memo, key}, v); v
            v -> v
          end
        end, [node()], :infinity)

      v ->
        v
    end
  end

  # ------------------------------------------------------------- airlock --

  @doc "Every substrate present, admitted by measurement (memoized for the server's life)."
  def substrates do
    memo(:substrates, fn ->
      for s <- Substrates.list() do
        a =
          try do
            Vapor.Substrate.admit(s)
          rescue
            e -> {:error, Exception.message(e)}
          catch
            _, why -> {:error, inspect(why)}
          end

        case a do
          %Vapor.Substrate.Admission{} = a ->
            %{id: s.id, kind: s.kind, isa: Map.get(s, :isa), mode: Map.get(s, :mode), device: a.device, verdict: a.verdict, fingerprint: a.fingerprint,
              reasons: a.reasons, probes: Enum.map(Enum.sort(a.probes), fn {name, p} -> %{name: name, equal: p.equal, envelope: p.envelope, ulps: Map.get(p, :ulps, %{})} end)}

          {:error, why} ->
            %{id: s.id, kind: s.kind, verdict: :unavailable, reasons: [to_string(why)]}
        end
      end
    end)
  end
end
