defmodule Vapor.Tlog.Holder do
  @moduledoc """
  The server's transparency log (`Vapor.Tlog`) and its signing key, held by a
  process. With `path:` the log is the append-only file there (reopened if it
  exists) and the key lives next to it (`PATH.key`, created `0600`); without,
  the log is in memory with a fresh key — a log for this run only, which a
  reader can still witness while it lasts.
  """
  use Agent
  alias Vapor.Tlog
  alias Vapor.Tlog.Note

  @doc "Options: `path`, `origin` (default `\"vapor.local/console\"`), `signer` (a note signer key)."
  def start_link(opts \\ []) do
    Agent.start_link(fn -> init(opts) end)
  end

  defp init(opts) do
    origin = Keyword.get(opts, :origin, "vapor.local/console")

    case Keyword.get(opts, :path) do
      nil ->
        k = Keyword.get_lazy(opts, :signer, fn -> Note.keygen(origin).signer end)
        %{log: Tlog.new(origin), signer: k, verifier: verifier(k)}

      path ->
        key = path <> ".key"

        signer =
          Keyword.get_lazy(opts, :signer, fn ->
            if File.exists?(key) do
              key |> File.read!() |> String.trim()
            else
              k = Note.keygen(origin).signer
              File.write!(key, k <> "\n")
              File.chmod!(key, 0o600)
              k
            end
          end)

        log =
          if File.exists?(path) do
            {:ok, l} = Tlog.open(path)
            l
          else
            Tlog.new(origin, path: path)
          end

        %{log: log, signer: signer, verifier: verifier(signer)}
    end
  end

  # the verifier key of a signer key string (the public half, re-derived)
  defp verifier("PRIVATE+KEY+" <> rest) do
    [name, id, b64] = String.split(rest, "+", parts: 3)
    {:ok, <<alg, seed::binary-32>>} = Base.decode64(b64)
    {pub, _} = :crypto.generate_key(:eddsa, :ed25519, seed)
    "#{name}+#{id}+#{Base.encode64(<<alg, pub::binary>>)}"
  end

  def get(h), do: Agent.get(h, & &1, 60_000)

  @doc "Append `entry`; returns its receipt."
  def anchor(h, entry) do
    Agent.get_and_update(h, fn st ->
      {log, r} = Tlog.anchor(st.log, entry, st.signer)
      {r, %{st | log: log}}
    end, 60_000)
  end
end
