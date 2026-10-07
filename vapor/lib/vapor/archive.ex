defmodule Vapor.Archive do
  @moduledoc """
  Save and export, for everything: one format for any result vapor
  produces — a chart read, a proof, a discovered algorithm, a simulation,
  a living scene, a trained policy.

  An archive is a zip with a `manifest.json` and its files. The manifest
  names the `kind`, the vapor version and semantics, the **recipe** (the
  request that produced the result, as data) and the SHA-256 of every
  file; the archive's identity is the SHA-256 of the manifest's canonical
  JSON (sorted keys), so the same result is the same archive on any
  machine, and one changed byte anywhere is caught (`verify/1`).

  `replay/1` runs the recipe again and compares: for the kinds whose
  producers are deterministic (most of vapor), an archive is not a claim but
  a re-checkable computation. Only kinds registered here can be replayed —
  an archive names a kind, never a function to call, so opening one runs
  nothing it chose.

  **Signed archives** (0.13). Integrity alone catches corruption, not a
  coherent lie: whoever rewrites a result can rewrite the manifest too. For
  the kinds that cannot be recomputed (a whole scene, a measurement) the
  answer is a signature: `sign/2` adds `signature.json` — Ed25519 over
  `"vapor-archive-v1\n" ‖ manifest.json`, with the key id — and
  `verify(zip, trusted: [public keys])` refuses an archive that is
  unsigned, signed by an unknown key, or whose signature does not hold.
  The signature sits outside the manifest, so the archive's identity does
  not change when it is signed. Keys are the operator keys of `mix
  vapor.audit keygen` (`load_key/1`).
  """

  @version 1

  # an archive is untrusted bytes: bounded before anything is inflated
  @max_entries 512
  @max_bytes 256 * 1024 * 1024

  # kind → {module, function}: a pure producer `fun(recipe) :: {:ok, result} | {:error, _}`
  @replayable %{
    "prove.geometry" => {Vapor.Prove, :replay},
    "prove.homology" => {Vapor.Prove, :replay},
    "science" => {Vapor.Science, :replay},
    "scene" => {Vapor.Scene, :replay},
    # 0.13: the finance desk's deterministic tasks (timings scrubbed)
    "finance.arbitrage" => {Vapor.Finance, :replay},
    "finance.backtest" => {Vapor.Finance, :replay},
    "finance.book" => {Vapor.Finance, :replay},
    "finance.calendar" => {Vapor.Finance, :replay},
    "finance.curve" => {Vapor.Finance, :replay},
    "finance.exchange" => {Vapor.Finance, :replay},
    "finance.micro" => {Vapor.Finance, :replay},
    "finance.options" => {Vapor.Finance, :replay},
    "finance.portfolio" => {Vapor.Finance, :replay},
    "finance.risk" => {Vapor.Finance, :replay}
  }

  @doc "The kinds whose recipes `replay/1` can run."
  def replayable, do: Map.keys(@replayable)

  @doc """
  Pack a result: `kind` (a string), `recipe` (JSON-able: what produced it),
  `result` (JSON-able), `files` (`%{name => binary}`, e.g. PNG, GLB, SVG,
  HTML). Returns `%{id, manifest, zip}` — `zip` is the archive's bytes.
  """
  def pack(kind, recipe, result, files \\ %{}) when is_binary(kind) do
    result_json = Vapor.JSON.encode(plain(result))
    files = Map.put(files, "result.json", result_json)

    manifest = %{
      "archive" => @version,
      "kind" => kind,
      "vapor" => to_string(Application.spec(:vapor, :vsn) || "dev"),
      "semantics" => Vapor.Canon.version(),
      "recipe" => plain(recipe),
      "files" => Map.new(files, fn {n, b} -> {n, %{"sha256" => sha(b), "bytes" => byte_size(b)}} end)
    }

    mjson = Vapor.JSON.encode(manifest)
    entries = [{~c"manifest.json", mjson} | for({n, b} <- Enum.sort(files), do: {String.to_charlist(n), b})]
    {:ok, {_, zip}} = :zip.create(~c"archive.zip", entries, [:memory])
    %{id: sha(mjson), manifest: manifest, zip: zip}
  end

  @doc """
  Open and check an archive's bytes: `{:ok, %{id, manifest, result, files}}`
  or `{:error, why}` — a file whose hash does not match, a file not in the
  manifest, a manifest missing.
  """
  def verify(zip, opts \\ []) when is_binary(zip) do
    with {:ok, entries} <- unzip(zip),
         {:ok, mjson} <- Map.fetch(entries, "manifest.json") |> or_error(:no_manifest),
         {:ok, m} <- decode_manifest(mjson),
         :ok <- check_files(m, Map.drop(entries, ["manifest.json", "signature.json"])),
         {:ok, rjson} <- Map.fetch(entries, "result.json") |> or_error(:no_result),
         {:ok, result} <- Vapor.JSON.decode(rjson) |> or_error(:bad_result),
         {:ok, sig} <- check_signature(mjson, entries["signature.json"], Keyword.get(opts, :trusted)) do
      {:ok, %{id: sha(mjson), manifest: m, result: result, signature: sig, files: Map.drop(entries, ["manifest.json", "result.json", "signature.json"])}}
    end
  end

  @sig_domain "vapor-archive-v1\n"

  @doc "Sign an archive with an Ed25519 key (`%{public, private}`): the same archive with `signature.json` added."
  def sign(zip, %{public: pub, private: priv}) do
    with {:ok, entries} <- unzip(zip), {:ok, mjson} <- Map.fetch(entries, "manifest.json") |> or_error(:no_manifest) do
      sig = :crypto.sign(:eddsa, :none, @sig_domain <> mjson, [priv, :ed25519])
      sj = Vapor.JSON.encode(%{"alg" => "ed25519", "domain" => String.trim(@sig_domain), "public" => Base.encode64(pub), "key_id" => Vapor.Certificate.key_id(pub),
                               "manifest_sha256" => sha(mjson), "signature" => Base.encode64(sig)})
      entries = entries |> Map.put("signature.json", sj)
      list = [{~c"manifest.json", mjson} | for({n, b} <- Enum.sort(Map.delete(entries, "manifest.json")), do: {String.to_charlist(n), b})]
      {:ok, {_, z}} = :zip.create(~c"archive.zip", list, [:memory])
      {:ok, z}
    end
  end

  @doc "Read an operator key file (base64 of private ‖ public, as `mix vapor.audit keygen` writes) or a `.pub` file."
  def load_key(path) do
    case path |> File.read!() |> String.trim() |> Base.decode64!() do
      <<priv::binary-32, pub::binary-32>> -> %{private: priv, public: pub}
      <<pub::binary-32>> -> %{public: pub}
    end
  end

  # nil trusted list: report the signature, require nothing; a list: require a valid signature by one of them
  defp check_signature(_mjson, nil, nil), do: {:ok, nil}
  defp check_signature(_mjson, nil, _trusted), do: {:error, :unsigned}

  defp check_signature(mjson, sj, trusted) do
    with {:ok, %{"public" => p64, "signature" => s64}} <- Vapor.JSON.decode(sj) |> or_error(:bad_signature_file),
         {:ok, pub} <- Base.decode64(p64) |> or_error(:bad_signature_file),
         {:ok, sig} <- Base.decode64(s64) |> or_error(:bad_signature_file) do
      valid = byte_size(pub) == 32 and :crypto.verify(:eddsa, :none, @sig_domain <> mjson, sig, [pub, :ed25519])
      info = %{key_id: Vapor.Certificate.key_id(pub), valid: valid, trusted: trusted != nil and pub in trusted}
      cond do
        not valid -> {:error, :bad_signature}
        trusted != nil and pub not in trusted -> {:error, {:untrusted_key, info.key_id}}
        true -> {:ok, info}
      end
    end
  end

  @doc """
  Verify, run the recipe again and compare the results' canonical JSON:
  `{:ok, :same}`, `{:error, {:differs, fresh_result}}`, or `{:error,
  {:not_replayable, kind}}`.
  """
  def replay(zip) do
    with {:ok, b} <- verify(zip),
         {:ok, {mod, fun}} <- Map.fetch(@replayable, b.manifest["kind"]) |> or_error({:not_replayable, b.manifest["kind"]}),
         {:ok, fresh} <- apply(mod, fun, [b.manifest["kind"], b.manifest["recipe"]]) do
      fresh = fresh |> plain() |> Vapor.JSON.encode() |> Vapor.JSON.decode() |> elem(1)
      if fresh == b.result, do: {:ok, :same}, else: {:error, {:differs, fresh}}
    end
  end

  @doc "Run a replayable kind's producer on a recipe: what `replay/1` compares against."
  def produce(kind, recipe) do
    case Map.fetch(@replayable, kind) do
      {:ok, {mod, fun}} -> apply(mod, fun, [kind, recipe])
      :error -> {:error, {:not_replayable, kind}}
    end
  end

  defp decode_manifest(mjson) do
    case Vapor.JSON.decode(mjson) do
      {:ok, %{"kind" => k, "files" => %{} = files} = m} when is_binary(k) ->
        if Enum.all?(files, fn {_, f} -> is_map(f) and is_binary(f["sha256"]) end), do: {:ok, m}, else: {:error, :bad_manifest}

      _ ->
        {:error, :bad_manifest}
    end
  end

  defp check_files(m, entries) do
    listed = m["files"]

    cond do
      Map.keys(listed) |> Enum.sort() != Map.keys(entries) |> Enum.sort() -> {:error, :files_differ_from_manifest}
      Enum.all?(listed, fn {n, %{"sha256" => h}} -> sha(entries[n]) == h end) -> :ok
      true -> {:error, {:tampered, for({n, %{"sha256" => h}} <- listed, sha(entries[n]) != h, do: n)}}
    end
  end

  defp unzip(zip) do
    # The central directory is read first, and every entry is inflated by
    # hand with a running count: a small zip that claims (or hides)
    # gigabytes is refused at the bound, never inflated past it.
    with {:ok, [_comment | dir]} <- :zip.list_dir(zip),
         :ok <- if(length(dir) <= @max_entries, do: :ok, else: {:error, :too_many_entries}),
         {:ok, list} <- inflate_all(zip, dir),
         names = Enum.map(list, &elem(&1, 0)),
         :ok <- if(length(Enum.uniq(names)) == length(names), do: :ok, else: {:error, :duplicate_entries}) do
      {:ok, Map.new(list)}
    else
      {:error, why} when why in [:too_many_entries, :too_large, :duplicate_entries] -> {:error, why}
      {:error, why} when is_atom(why) -> {:error, {:not_an_archive, why}}
      _ -> {:error, {:not_an_archive, :unreadable}}
    end
  end

  defp inflate_all(zip, dir) do
    Enum.reduce_while(dir, {:ok, [], 0}, fn {:zip_file, name, _info, _c, offset, csize}, {:ok, acc, total} ->
      case entry(zip, offset, csize, @max_bytes - total) do
        {:ok, bytes} -> {:cont, {:ok, [{to_string(name), bytes} | acc], total + byte_size(bytes)}}
        err -> {:halt, err}
      end
    end)
    |> case do
      {:ok, acc, _} -> {:ok, Enum.reverse(acc)}
      err -> err
    end
  end

  # local header: signature, version, flags, method (offset 8), …, name and extra lengths (26, 28)
  defp entry(zip, offset, csize, budget) do
    with <<_::binary-size(offset), 0x50, 0x4B, 0x03, 0x04, _::binary-size(4), method::little-16, _::binary-size(16), nl::little-16, xl::little-16, rest::binary>> <- zip,
         <<_::binary-size(nl + xl), data::binary-size(csize), _::binary>> <- rest do
      case method do
        0 -> if byte_size(data) <= budget, do: {:ok, data}, else: {:error, :too_large}
        8 -> inflate(data, budget)
        m -> {:error, {:compression, m}}
      end
    else
      _ -> {:error, :bad_entry}
    end
  end

  defp inflate(data, budget) do
    z = :zlib.open()

    try do
      :ok = :zlib.inflateInit(z, -15)
      inflate_loop(z, :zlib.safeInflate(z, data), [], 0, budget)
    rescue
      _ -> {:error, :bad_deflate}
    after
      :zlib.close(z)
    end
  end

  defp inflate_loop(z, {state, out}, acc, n, budget) do
    n = n + IO.iodata_length(out)

    cond do
      n > budget -> {:error, :too_large}
      state == :finished -> {:ok, IO.iodata_to_binary(Enum.reverse([out | acc]))}
      true -> inflate_loop(z, :zlib.safeInflate(z, []), [out | acc], n, budget)
    end
  end

  defp or_error({:ok, v}, _), do: {:ok, v}
  defp or_error(:error, why), do: {:error, why}
  defp or_error({:error, _}, why), do: {:error, why}

  defp sha(b), do: Base.encode16(:crypto.hash(:sha256, b), case: :lower)

  @doc "A term made JSON-able: tuples → lists, atoms (not booleans/nil) → strings, structs → maps."
  def plain(%{__struct__: _} = s), do: s |> Map.from_struct() |> plain()
  def plain(m) when is_map(m), do: Map.new(m, fn {k, v} -> {to_string(k), plain(v)} end)
  def plain(l) when is_list(l), do: Enum.map(l, &plain/1)
  def plain(t) when is_tuple(t), do: t |> Tuple.to_list() |> plain()
  def plain(a) when is_atom(a) and a not in [true, false, nil], do: Atom.to_string(a)
  def plain(f) when is_float(f), do: if(f != f or abs(f) == :infinity, do: nil, else: f)
  def plain(x), do: x
end
