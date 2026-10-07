defmodule Mix.Tasks.Vapor.Audit do
  @shortdoc "Pack signed, offline-verifiable audit dossiers from the system's evidence; verify them"
  @moduledoc """
      mix vapor.audit keygen KEYFILE                      # Ed25519 key: KEYFILE (private) and KEYFILE.pub
      mix vapor.audit export --out FILE.vdossier --key KEYFILE [--cosign KEYFILE]…
                             [--system NAME] [--version V] [--provider P] [--date YYYY-MM-DD] [--notes TEXT]
                             [--certificate FILE]… [--quality FILE] [--model PATH]
                             [--journal FILE --attestation FILE]…
                             [--receipt FILE --entry TEXT --log-key VKEY]…
                             [--document PATH[:CLAUSE,CLAUSE…]]…
                             [--html] [--pdf]
      mix vapor.audit verify FILE [--trusted KEYFILE.pub]… [--quorum N] [--log-key VKEY]…
      mix vapor.audit demo [--out DIR]                     # a complete dossier from this checkout

  `export` packs the evidence into one dossier (`Vapor.Audit`): a manifest
  of every item (kind, size, SHA-256, its verification summary, the clauses
  of the EU AI Act and ISO/IEC 42001 it is relevant to), a Merkle root,
  and Ed25519 signatures over the canonical manifest; `--html` also writes
  `FILE.html` (the dossier verifying itself in a browser, offline) and
  `--pdf` `FILE.pdf` (a printable report carrying the dossier as an
  attachment). Files: a certificate as written by `Vapor.Certificate.encode/1`,
  a journal by `Vapor.Agent.Journal.encode/1`, an attestation and a log
  receipt as JSON, the quality report as `docs/bench/quality.json`.

  `verify` rechecks a dossier (or a PDF report carrying one) from its
  bytes — hashes, root, signatures (only `--trusted` keys count, when
  given), each item by its own rules, log anchors with `--log-key` — and
  exits 1 on any failure, naming it.

  `demo` builds one from this checkout: a compiled program's certificate,
  an attested journal, a transparency-log receipt, the quality report, a
  model contract — dossier, HTML and PDF in `--out` (default
  `_build/audit-demo`).

  A dossier authenticates evidence; it is not a conformity assessment or a
  legal opinion (docs/AUDITORIA.md).
  """
  use Mix.Task
  alias Vapor.{Audit, Certificate}

  @switches [out: :string, key: :string, cosign: :keep, system: :string, version: :string, provider: :string, date: :string,
             notes: :string, certificate: :keep, quality: :string, model: :string, journal: :keep, attestation: :keep,
             receipt: :keep, entry: :keep, log_key: :keep, document: :keep, html: :boolean, pdf: :boolean, trusted: :keep,
             quorum: :integer]

  @impl true
  def run(argv) do
    {o, args, _} = OptionParser.parse(argv, strict: @switches)
    Mix.Task.run("app.start")

    case args do
      ["keygen", file] -> keygen(file)
      ["export"] -> export(o)
      ["verify", file] -> verify(file, o)
      ["demo"] -> demo(o[:out] || "_build/audit-demo")
      _ -> Mix.raise("usage: mix vapor.audit keygen KEYFILE | export --out FILE --key KEYFILE … | verify FILE | demo")
    end
  end

  defp keygen(file) do
    k = Certificate.keygen()
    File.write!(file, Base.encode64(k.private <> k.public) <> "\n")
    File.chmod!(file, 0o600)
    File.write!(file <> ".pub", Base.encode64(k.public) <> "\n")
    Mix.shell().info("key #{Certificate.key_id(k.public)}: #{file} (private), #{file}.pub")
  end

  defp read_key(file) do
    <<priv::binary-32, pub::binary-32>> = file |> File.read!() |> String.trim() |> Base.decode64!()
    %{private: priv, public: pub}
  end

  defp read_pub(file), do: file |> File.read!() |> String.trim() |> Base.decode64!()

  defp export(o) do
    out = o[:out] || Mix.raise("--out FILE is required")
    key = read_key(o[:key] || Mix.raise("--key KEYFILE is required (mix vapor.audit keygen KEYFILE)"))

    items =
      Enum.map(Keyword.get_values(o, :certificate), fn f ->
        {:ok, c} = Certificate.decode(File.read!(f))
        Audit.certificate(c, Path.basename(f))
      end) ++
        Enum.map(Enum.zip(Keyword.get_values(o, :journal), Keyword.get_values(o, :attestation)), fn {jf, af} ->
          {:ok, j} = Vapor.Agent.Journal.decode(File.read!(jf))
          att = af |> File.read!() |> Vapor.JSON.decode!() |> decode_attestation()
          Audit.journal(j, att, Path.basename(jf))
        end) ++
        Enum.map(Enum.zip([Keyword.get_values(o, :receipt), Keyword.get_values(o, :entry), Keyword.get_values(o, :log_key)]), fn {rf, e, k} ->
          Audit.tlog_receipt(Vapor.JSON.decode!(File.read!(rf)), e, k, Path.basename(rf))
        end) ++
        if(q = o[:quality], do: [Audit.quality(File.read!(q), Path.basename(q))], else: []) ++
        if(m = o[:model], do: [model_item(m)], else: []) ++
        Enum.map(Keyword.get_values(o, :document), fn spec ->
          {path, clauses} = case String.split(spec, ":", parts: 2) do
            [p, cs] -> {p, String.split(cs, ",")}
            [p] -> {p, nil}
          end
          Audit.document(Path.basename(path), File.read!(path), clauses)
        end)

    if items == [], do: Mix.raise("no evidence given")
    system = for {k, v} <- [name: o[:system], version: o[:version], provider: o[:provider]], v, into: %{}, do: {k, v}
    d = Audit.build(items, system: system, date: o[:date] || Date.to_iso8601(Date.utc_today()), notes: o[:notes] || "")
    d = Enum.reduce([o[:key] | Keyword.get_values(o, :cosign)], d, fn kf, d -> Audit.sign(d, read_key(kf)) end)
    write(d, out, o[:html], o[:pdf])
    _ = key
  end

  defp decode_attestation(m), do: m |> Map.update("key", nil, &Base.decode64!/1) |> Map.update("sig", nil, &Base.decode64!/1)

  defp model_item(path) do
    case Vapor.Lock.open(path) do
      {:ok, m} -> Audit.contract(m.spec, Path.basename(path))
      {:error, r} -> Mix.raise("#{path}: #{inspect(r)}")
    end
  end

  defp write(d, out, html, pdf) do
    File.mkdir_p!(Path.dirname(out))
    File.write!(out, Audit.encode(d))
    if html, do: File.write!(out <> ".html", Audit.html(d))
    if pdf, do: File.write!(out <> ".pdf", Audit.pdf(d))

    Mix.shell().info("dossier #{out}: #{length(d.blobs)} items, root #{d.manifest["root"]}, #{length(d.signatures)} signature(s)" <>
                       if(html, do: ", #{out}.html", else: "") <> if(pdf, do: ", #{out}.pdf", else: ""))
  end

  defp verify(file, o) do
    trusted = case Keyword.get_values(o, :trusted) do
      [] -> :any
      fs -> Enum.map(fs, &read_pub/1)
    end

    opts = [trusted: trusted, quorum: o[:quorum] || 1] ++ case Keyword.get_values(o, :log_key) do
      [] -> []
      ks -> [log_keys: ks]
    end

    {verdict, r} = Audit.verify(File.read!(file), opts)

    if Map.has_key?(r, :items) do
      for it <- r.items do
        mark = if it.hash and it.check in [:ok, :na], do: "ok  ", else: "FAIL"
        Mix.shell().info("#{mark} #{it.i + 1}. [#{it.kind}] #{it.name} — hash #{if it.hash, do: "matches", else: "ALTERED"}, check #{inspect(it.check)}")
      end

      for s <- r.signatures, do: Mix.shell().info("#{if s.valid and s.trusted, do: "ok  ", else: "--  "} signature #{s.key_id}: #{if s.valid, do: "valid", else: "INVALID"}#{if s.trusted, do: "", else: " (not trusted)"}")
      for a <- r.anchors, do: Mix.shell().info("#{if a.check == :ok, do: "ok  ", else: "--  "} log anchor ##{a.index}: #{inspect(a.check)}")
      Mix.shell().info("root #{if r.root, do: "recomputed", else: "DIFFERS"}; quorum #{elem(r.quorum, 0)}/#{elem(r.quorum, 1)}")
    end

    if verdict == :ok, do: Mix.shell().info("VERIFIED"), else: (Mix.shell().error("NOT VERIFIED: #{inspect(Map.get(r, :reason, "see above"))}"); exit({:shutdown, 1}))
  end

  # ------------------------------------------------------------------ demo --

  defp demo(dir) do
    File.mkdir_p!(dir)
    quality = Path.join(["docs", "bench", "quality.json"])
    %{dossier: d, signer: key, log_key: log_key} = Audit.demo(quality: if(File.exists?(quality), do: File.read!(quality)))
    out = Path.join(dir, "dossier.vdossier")
    write(d, out, true, true)
    File.write!(Path.join(dir, "log.vkey"), log_key.verifier <> "\n")
    File.write!(Path.join(dir, "signer.pub"), Base.encode64(key.public) <> "\n")
    Mix.shell().info("verify: mix vapor.audit verify #{out} --trusted #{Path.join(dir, "signer.pub")} --log-key \"$(cat #{Path.join(dir, "log.vkey")})\"")
  end
end
