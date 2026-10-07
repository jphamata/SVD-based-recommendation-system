defmodule Vapor.Tlog.Note do
  @moduledoc """
  Signed notes (C2SP `signed-note`, the format of `golang.org/x/mod/sumdb/note`)
  and witness cosignatures (C2SP `tlog-cosignature/v1`), with Ed25519.

  A note is UTF-8 text ending in a newline, a blank line, and one line per
  signature: `— <name> <base64(key id ‖ signature)>`. The key id is the
  first four bytes of `SHA-256(name ‖ "\\n" ‖ alg ‖ public key)`; `alg` is
  `0x01` for a plain Ed25519 note signature and `0x04` for a
  `cosignature/v1`, whose signed message is
  `"cosignature/v1\\ntime <t>\\n" ‖ note text` and whose signature bytes are
  `key id ‖ t (8 bytes, big-endian) ‖ Ed25519 signature`.

  Keys travel as the standard strings: a verifier key
  `<name>+<hex key id>+<base64(alg ‖ public key)>` and a signer key
  `PRIVATE+KEY+<name>+<hex key id>+<base64(alg ‖ seed)>`.
  """

  @ed25519 0x01
  @cosig 0x04

  @doc """
  A fresh key pair for `name`; `kind: :note` (a log, default) or
  `:cosignature` (a witness). Returns `%{signer: string, verifier: string}`.
  """
  def keygen(name, kind \\ :note) do
    valid_name!(name)
    {pub, priv} = :crypto.generate_key(:eddsa, :ed25519)
    alg = alg(kind)
    id = key_id(name, alg, pub)

    %{signer: "PRIVATE+KEY+#{name}+#{hex(id)}+#{Base.encode64(<<alg, priv::binary>>)}",
      verifier: "#{name}+#{hex(id)}+#{Base.encode64(<<alg, pub::binary>>)}"}
  end

  defp alg(:note), do: @ed25519
  defp alg(:cosignature), do: @cosig

  defp valid_name!(name) do
    if name == "" or String.contains?(name, ["+", " ", "\n", "\t"]) or not String.valid?(name),
      do: raise(ArgumentError, "a key name is non-empty UTF-8 without spaces, newlines or '+'")
  end

  @doc "The name a verifier or signer key string carries."
  def key_name("PRIVATE+KEY+" <> rest), do: rest |> String.split("+") |> hd()
  def key_name(vkey), do: vkey |> String.split("+") |> hd()

  defp key_id(name, alg, pub), do: binary_part(:crypto.hash(:sha256, <<name::binary, "\n", alg, pub::binary>>), 0, 4)
  defp hex(id), do: Base.encode16(id, case: :lower)

  defp parse_vkey(vkey) do
    with [name, id_hex, b64] <- String.split(vkey, "+", parts: 3),
         {:ok, id} <- Base.decode16(id_hex, case: :lower),
         {:ok, <<alg, pub::binary-32>>} <- Base.decode64(b64),
         true <- alg in [@ed25519, @cosig] and key_id(name, alg, pub) == id do
      {:ok, %{name: name, id: id, alg: alg, pub: pub, key: vkey}}
    else
      _ -> {:error, {:bad_verifier_key, vkey}}
    end
  end

  defp parse_skey("PRIVATE+KEY+" <> rest) do
    with [name, id_hex, b64] <- String.split(rest, "+", parts: 3),
         {:ok, id} <- Base.decode16(id_hex, case: :lower),
         {:ok, <<alg, priv::binary-32>>} <- Base.decode64(b64) do
      {:ok, %{name: name, id: id, alg: alg, priv: priv}}
    else
      _ -> {:error, :bad_signer_key}
    end
  end

  defp parse_skey(_), do: {:error, :bad_signer_key}

  @doc "Sign `text` (must end in a newline) with plain note signers: the note."
  def sign(text, signers) when is_binary(text) do
    unless String.ends_with?(text, "\n") and String.valid?(text), do: raise(ArgumentError, "note text is UTF-8 ending in a newline")

    lines =
      for s <- signers do
        {:ok, %{name: n, id: id, alg: @ed25519, priv: priv}} = parse_skey(s)
        sig = :crypto.sign(:eddsa, :none, text, [priv, :ed25519])
        sig_line(n, id <> sig)
      end

    text <> "\n" <> Enum.join(lines)
  end

  @doc """
  Add a `cosignature/v1` by `signer` (a witness key) to `note`, at time `t`
  (seconds since the epoch; default now). The note's existing signatures
  are kept; the witness signs the text only.
  """
  def cosign(note, signer, t \\ System.os_time(:second)) do
    {:ok, %{name: n, id: id, alg: @cosig, priv: priv}} = parse_skey(signer)
    {text, _} = split_note(note)
    sig = :crypto.sign(:eddsa, :none, cosig_message(text, t), [priv, :ed25519])
    note <> sig_line(n, <<id::binary, t::64, sig::binary>>)
  end

  defp cosig_message(text, t), do: "cosignature/v1\ntime #{t}\n" <> text

  defp sig_line(name, bytes), do: "— #{name} #{Base.encode64(bytes)}\n"

  @doc "The text of a note and its signature lines (unverified)."
  def split_note(note) do
    # the signature block follows the last blank line
    case :binary.matches(note, "\n\n") do
      [] ->
        {note, []}

      ms ->
        {pos, _} = List.last(ms)
        text = binary_part(note, 0, pos + 1)
        sigs = binary_part(note, pos + 2, byte_size(note) - pos - 2)
        {text, String.split(sigs, "\n", trim: true)}
    end
  end

  @doc """
  Open `note` against `keys` (verifier key strings): `{:ok, %{text, signers}}`
  where `signers` lists every valid signature by a known key
  (`%{name, key, kind: :note | :cosignature, time}`); signatures by unknown
  keys are ignored (as the format requires); a known key id whose signature
  fails is an error; a note with no valid signature is an error.
  """
  def open(note, keys) do
    {text, lines} = split_note(note)

    with {:ok, vkeys} <- parse_all(keys) do
      result =
        Enum.reduce_while(lines, {:ok, []}, fn line, {:ok, acc} ->
          case check_line(line, text, vkeys) do
            :unknown -> {:cont, {:ok, acc}}
            {:ok, s} -> {:cont, {:ok, [s | acc]}}
            {:error, _} = e -> {:halt, e}
          end
        end)

      case result do
        {:ok, []} -> {:error, :no_known_signature}
        {:ok, signers} -> {:ok, %{text: text, signers: Enum.reverse(signers)}}
        e -> e
      end
    end
  end

  defp parse_all(keys) do
    Enum.reduce_while(keys, {:ok, []}, fn k, {:ok, acc} ->
      case parse_vkey(k) do
        {:ok, v} -> {:cont, {:ok, [v | acc]}}
        e -> {:halt, e}
      end
    end)
  end

  defp check_line("— " <> rest, text, vkeys) do
    with [name, b64] <- String.split(rest, " "),
         {:ok, <<id::binary-4, body::binary>>} <- Base.decode64(b64) do
      case Enum.find(vkeys, &(&1.name == name and &1.id == id)) do
        nil -> :unknown
        %{alg: @ed25519, pub: pub, key: k} -> verify_plain(body, text, pub, name, k)
        %{alg: @cosig, pub: pub, key: k} -> verify_cosig(body, text, pub, name, k)
      end
    else
      _ -> {:error, :malformed_signature}
    end
  end

  defp check_line(_, _, _), do: {:error, :malformed_signature}

  defp verify_plain(<<sig::binary-64>>, text, pub, name, k) do
    if :crypto.verify(:eddsa, :none, text, sig, [pub, :ed25519]),
      do: {:ok, %{name: name, key: k, kind: :note, time: nil}},
      else: {:error, {:bad_signature, name}}
  end

  defp verify_plain(_, _, _, name, _), do: {:error, {:bad_signature, name}}

  defp verify_cosig(<<t::64, sig::binary-64>>, text, pub, name, k) do
    if :crypto.verify(:eddsa, :none, cosig_message(text, t), sig, [pub, :ed25519]),
      do: {:ok, %{name: name, key: k, kind: :cosignature, time: t}},
      else: {:error, {:bad_signature, name}}
  end

  defp verify_cosig(_, _, _, name, _), do: {:error, {:bad_signature, name}}
end
