defmodule Mix.Tasks.Vapor.Archive do
  @shortdoc "Sign, verify and replay vapor archives (signed manifests, 0.13)"
  @moduledoc """
      mix vapor.archive sign FILE.zip --key KEYFILE [--out SIGNED.zip]
      mix vapor.archive verify FILE.zip [--trusted KEYFILE.pub]…
      mix vapor.archive replay FILE.zip

  `sign` adds `signature.json` (Ed25519 over the manifest, the operator key
  of `mix vapor.audit keygen`); the archive's identity does not change.
  `verify` checks every file against the manifest and, with `--trusted`,
  refuses an archive that is unsigned, signed by another key, or whose
  signature does not hold. `replay` recomputes a deterministic kind and
  compares.
  """
  use Mix.Task
  alias Vapor.Archive

  @impl true
  def run(argv) do
    {o, args, _} = OptionParser.parse(argv, strict: [key: :string, out: :string, trusted: :keep])
    Mix.Task.run("app.start")
    case args do
      ["sign", file] ->
        key = Archive.load_key(o[:key] || Mix.raise("--key KEYFILE"))
        {:ok, z} = Archive.sign(File.read!(file), key)
        out = o[:out] || String.replace_suffix(file, ".zip", ".signed.zip")
        File.write!(out, z)
        Mix.shell().info("signed by #{Vapor.Certificate.key_id(key.public)}: #{out}")
      ["verify", file] ->
        trusted = case Keyword.get_values(o, :trusted) do [] -> nil; ks -> Enum.map(ks, &Archive.load_key(&1).public) end
        case Archive.verify(File.read!(file), trusted: trusted) do
          {:ok, b} -> Mix.shell().info("intact: #{b.manifest["kind"]} #{b.id}" <> if(b.signature, do: " · signed by #{b.signature.key_id}#{if b.signature.trusted, do: " (trusted)", else: ""}", else: " · unsigned"))
          {:error, why} -> (Mix.shell().error("refused: #{inspect(why)}"); exit({:shutdown, 1}))
        end
      ["replay", file] ->
        case Archive.replay(File.read!(file)) do
          {:ok, :same} -> Mix.shell().info("replayed: the same result")
          other -> (Mix.shell().error("#{inspect(other, limit: 5)}"); exit({:shutdown, 1}))
        end
      _ -> Mix.raise("usage: mix vapor.archive sign|verify|replay FILE.zip …")
    end
  end
end
