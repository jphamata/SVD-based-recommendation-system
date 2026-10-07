defmodule Vapor.Audit do
  @moduledoc """
  **Audit dossiers** — the evidence an AI system already produces, packed
  into one signed, self-verifying file a regulator, an auditor or a
  customer can check **offline, without trusting whoever sends it**.

  vapor's outputs are evidence by construction: compilation certificates
  (`Vapor.Certificate`), agent journals with their attestations
  (`Vapor.Agent.Journal`), transparency-log receipts (`Vapor.Tlog`), the
  quality report with its controls (`mix vapor.quality`), fusion receipts,
  model contracts from the airlock. What an audit asks for is that they be
  *found, complete, unaltered and attributable* — so a dossier is:

    * a **manifest** — the system's identity, every item's kind, name,
      size, SHA-256, its own verification summary, and the clauses it is
      evidence for — and the items' bytes;
    * a **Merkle root** over the items (RFC 6962 leaves), so any one item
      can later be shown without the rest;
    * **Ed25519 signatures** over the canonical CBOR of the manifest
      (`Vapor.Canonical`), with co-signatures from independent nodes;
    * optionally **anchors**: transparency-log receipts for the root.

  `verify/2` rechecks *everything from the bytes*: hashes, root,
  signatures, and each item by its own rules (a certificate's signatures,
  a journal's hash chain and attestation, a log receipt's inclusion proof
  and checkpoint, a quality report's verdicts). `html/1` renders the
  dossier as one offline page that does the same checks in the browser
  (SHA-256 and Ed25519 in WebCrypto; the page decodes the *signed* bytes
  with its own CBOR reader, so what it shows is what was signed). `pdf/1`
  renders a printable report with the dossier attached
  (`/EmbeddedFiles`); `extract/1` takes it back out.

  **What this is not.** A dossier collects and authenticates evidence; it
  is not a conformity assessment, a legal opinion or a certification. The
  clause mapping (`clauses/0`) points each kind of evidence at the
  provisions it is *relevant* to — EU AI Act (Regulation (EU) 2024/1689)
  Articles 11–15 and 19 and Annex IV, ISO/IEC 42001:2023 Annex A — and is
  data in the manifest: replace it with your own reading (`clauses:`).
  """
  alias Vapor.{Canonical, Certificate, Merkle}
  alias Vapor.Agent.Journal

  @format "vapor-dossier/1"

  @clauses %{
    "AIA-11" => "EU AI Act Art. 11 — technical documentation",
    "AIA-12" => "EU AI Act Art. 12 — record-keeping (automatic logs)",
    "AIA-13" => "EU AI Act Art. 13 — transparency and information to deployers",
    "AIA-15" => "EU AI Act Art. 15 — accuracy, robustness and cybersecurity",
    "AIA-19" => "EU AI Act Art. 19 — automatically generated logs (keeping)",
    "AIA-IV-2b" => "EU AI Act Annex IV §2(b) — design specifications of the system",
    "AIA-IV-2g" => "EU AI Act Annex IV §2(g) — validation and testing procedures and metrics",
    "AIA-IV-4" => "EU AI Act Annex IV §4 — appropriateness of the performance metrics",
    "ISO42001-A.6.2.3" => "ISO/IEC 42001 A.6.2.3 — documentation of AI system design and development",
    "ISO42001-A.6.2.4" => "ISO/IEC 42001 A.6.2.4 — AI system verification and validation",
    "ISO42001-A.6.2.7" => "ISO/IEC 42001 A.6.2.7 — AI system technical documentation",
    "ISO42001-A.6.2.8" => "ISO/IEC 42001 A.6.2.8 — AI system recording of event logs"
  }

  @by_kind %{
    "certificate" => ["AIA-15", "AIA-IV-2g", "ISO42001-A.6.2.4"],
    "journal" => ["AIA-12", "AIA-19", "ISO42001-A.6.2.8"],
    "tlog_receipt" => ["AIA-12", "AIA-19", "ISO42001-A.6.2.8"],
    "quality" => ["AIA-15", "AIA-IV-2g", "AIA-IV-4", "ISO42001-A.6.2.4"],
    "contract" => ["AIA-11", "AIA-13", "AIA-IV-2b", "ISO42001-A.6.2.7"],
    "merge_receipt" => ["AIA-11", "ISO42001-A.6.2.3"],
    "document" => ["AIA-11", "ISO42001-A.6.2.7"]
  }

  @disclaimer "This dossier collects and authenticates evidence produced by the system. It is not a conformity assessment, a legal opinion or a certification; the clause mapping states which provisions each kind of evidence is relevant to, and can be replaced."

  @doc "The clause identifiers and titles a dossier maps evidence to (data, replaceable per dossier)."
  def clauses, do: @clauses

  @doc "The default kinds → clauses mapping."
  def mapping, do: @by_kind

  # ----------------------------------------------------------------- items --

  @doc """
  An item of evidence. `kind` is one of `certificate`, `journal`,
  `tlog_receipt`, `quality`, `contract`, `merge_receipt`, `document`;
  `bytes` is what is hashed and stored; `meta` a small map shown beside it.
  Prefer the constructors below, which check the evidence as they wrap it.
  """
  def item(kind, name, bytes, meta \\ %{}, clauses \\ nil) when is_binary(bytes),
    do: %{kind: to_string(kind), name: name, bytes: bytes, meta: meta, clauses: clauses || Map.get(@by_kind, to_string(kind), [])}

  @doc "A compilation certificate (`Vapor.Certificate`)."
  def certificate(%Certificate{} = c, name) do
    item(:certificate, name, Certificate.encode(c),
         %{"signatures" => length(c.signatures), "program" => hexish(get_in(c.payload, [:program]) || get_in(c.payload, ["program"]))})
  end

  @doc "An agent run: its journal and the node's attestation (`Vapor.Agent.Journal.attest/2`)."
  def journal(%Journal{} = j, attestation, name) do
    bytes = Canonical.encode({:vapor_journal_evidence, 1, Journal.encode(j), attestation})
    item(:journal, name, bytes, %{"run_id" => j.run_id, "events" => length(j.events), "root" => Journal.root(j)})
  end

  @doc "A transparency-log receipt for `entry` (`Vapor.Tlog.receipt/3`), with the log's verifier key."
  def tlog_receipt(receipt, entry, verifier_key, name) do
    bytes = Canonical.encode({:vapor_tlog_evidence, 1, receipt, entry, verifier_key})
    item(:tlog_receipt, name, bytes, %{"index" => receipt["index"], "size" => receipt["size"]})
  end

  @doc "A quality report (`docs/bench/quality.json`, as written by `mix vapor.quality`)."
  def quality(json_bytes, name) do
    meta =
      case Vapor.JSON.decode(json_bytes) do
        {:ok, r} -> quality_summary(r)
        _ -> %{"error" => "not JSON"}
      end

    item(:quality, name, json_bytes, meta)
  end

  @doc "A model's contract from the airlock (`Vapor.Lock.Spec`): family, interface, features, sizes."
  def contract(%Vapor.Lock.Spec{} = s, name) do
    c = %{"family" => to_string(s.family), "interface" => to_string(s.interface), "features" => Enum.map(s.features, &to_string/1),
          "vocab" => s.vocab, "max_pos" => s.max_pos}
    item(:contract, name, Vapor.JSON.encode(c), c)
  end

  @doc "Any other document (technical documentation, a risk assessment…), with the clauses it serves."
  def document(name, bytes, clauses \\ nil), do: item(:document, name, bytes, %{"bytes" => byte_size(bytes)}, clauses)

  defp hexish(nil), do: nil
  defp hexish(b) when is_binary(b) and byte_size(b) == 32, do: Base.encode16(b, case: :lower)
  defp hexish(b) when is_binary(b), do: b
  defp hexish(other), do: inspect(other)

  defp quality_summary(r) do
    checks = all_checks(r)
    %{"checks" => length(checks), "passed" => Enum.count(checks, &(&1["pass"] == true)), "substrate" => r["substrate"]}
  end

  defp all_checks(r) do
    r
    |> Map.values()
    |> Enum.flat_map(fn
      %{"checks" => cs} when is_list(cs) -> cs
      xs when is_list(xs) -> Enum.flat_map(xs, fn %{"checks" => cs} when is_list(cs) -> cs; _ -> [] end)
      _ -> []
    end)
    |> Enum.filter(&is_map/1)
  end

  # --------------------------------------------------------------- dossier --

  @doc """
  Build a dossier from items. Options: `system` (a map — name, version,
  provider, purpose…), `date` (a string the caller gives: the dossier has
  no clock, so the same evidence gives the same bytes), `clauses` (a map
  replacing `clauses/0`), `notes`.
  """
  def build(items, opts \\ []) do
    leaves = Enum.map(items, &Merkle.leaf(:crypto.hash(:sha256, &1.bytes)))

    manifest = %{
      "format" => @format,
      "system" => stringify(Keyword.get(opts, :system, %{})),
      "date" => Keyword.get(opts, :date, ""),
      "notes" => Keyword.get(opts, :notes, ""),
      "clauses" => Keyword.get(opts, :clauses, @clauses),
      "disclaimer" => @disclaimer,
      "root" => Base.encode16(Merkle.root(leaves), case: :lower),
      "items" =>
        items
        |> Enum.with_index()
        |> Enum.map(fn {it, i} ->
          %{"i" => i, "kind" => it.kind, "name" => it.name, "size" => byte_size(it.bytes),
            "sha256" => Base.encode16(:crypto.hash(:sha256, it.bytes), case: :lower), "meta" => stringify(it.meta), "clauses" => it.clauses}
        end)
    }

    %{manifest: manifest, blobs: Enum.map(items, & &1.bytes), signatures: [], anchors: []}
  end

  defp stringify(m) when is_map(m), do: Map.new(m, fn {k, v} -> {to_string(k), stringify(v)} end)
  defp stringify(l) when is_list(l), do: Enum.map(l, &stringify/1)
  defp stringify(a) when is_atom(a) and a not in [nil, true, false], do: Atom.to_string(a)
  defp stringify(v), do: v

  @doc "The bytes a signature covers: the canonical CBOR of the manifest."
  def signed_bytes(%{manifest: m}), do: Canonical.encode({:vapor_dossier_manifest, 1, m})

  @doc "Sign (or co-sign) with an Ed25519 key (`Vapor.Certificate.keygen/0`)."
  def sign(d, %{public: pub, private: priv}) do
    sig = :crypto.sign(:eddsa, :none, signed_bytes(d), [priv, :ed25519])
    %{d | signatures: Enum.uniq_by(d.signatures ++ [%{"key" => pub, "sig" => sig}], & &1["key"])}
  end

  @doc """
  Anchor the dossier's root in a transparency log (`Vapor.Tlog`): returns
  `{log, dossier}` with the receipt added to the (unsigned) anchors — it
  verifies on its own against the log's key.
  """
  def anchor(d, log, signer) do
    {log, receipt} = Vapor.Tlog.anchor(log, anchor_entry(d), signer)
    {log, %{d | anchors: d.anchors ++ [receipt]}}
  end

  defp anchor_entry(d), do: "vapor-dossier " <> d.manifest["root"] <> " " <> Base.encode16(:crypto.hash(:sha256, signed_bytes(d)), case: :lower)

  @doc "The dossier as one file (canonical CBOR)."
  def encode(d), do: Canonical.encode({:vapor_dossier, 1, d.manifest, d.blobs, d.signatures, d.anchors})

  @doc "Read a dossier file (or a PDF report carrying one)."
  def decode("%PDF" <> _ = pdf) do
    case extract(pdf) do
      {:ok, bin} -> decode(bin)
      e -> e
    end
  end

  def decode(bin) do
    case Canonical.decode(bin, atoms: :existing) do
      {:ok, {:vapor_dossier, 1, m, blobs, sigs, anchors}} when is_map(m) and is_list(blobs) ->
        {:ok, %{manifest: m, blobs: blobs, signatures: sigs, anchors: anchors}}

      {:ok, _} -> {:error, :not_a_dossier}
      e -> e
    end
  end

  # ---------------------------------------------------------------- verify --

  @doc """
  Verify a dossier from its bytes (or a decoded dossier). Options:
  `trusted` (public keys whose signatures count; default every signer),
  `quorum` (distinct trusted signatures required; 1), `log_keys`
  (verifier keys for anchors and log receipts). Returns `{:ok, report}`
  when everything holds, else `{:error, report}` — the report names every
  failure: `%{ok, root, signatures, items: [%{i, name, kind, hash, check}], anchors}`.
  """
  def verify(bin_or_dossier, opts \\ [])

  def verify(bin, opts) when is_binary(bin) do
    case decode(bin) do
      {:ok, d} -> verify(d, opts)
      {:error, why} -> {:error, %{ok: false, reason: why}}
    end
  end

  def verify(%{manifest: m, blobs: blobs} = d, opts) do
    known_atoms()
    trusted = Keyword.get(opts, :trusted, :any)
    quorum = Keyword.get(opts, :quorum, 1)
    msg = signed_bytes(d)

    sigs =
      Enum.map(d.signatures, fn %{"key" => pub, "sig" => sig} ->
        valid = :crypto.verify(:eddsa, :none, msg, sig, [pub, :ed25519])
        %{key_id: Certificate.key_id(pub), valid: valid, trusted: trusted == :any or pub in trusted}
      end)

    good_sigs = Enum.count(sigs, &(&1.valid and &1.trusted))
    items = Map.get(m, "items", [])

    item_reports =
      if length(items) != length(blobs) do
        [%{i: -1, name: "(all)", kind: "-", hash: false, check: {:fail, "#{length(blobs)} blobs for #{length(items)} items"}}]
      else
        Enum.zip(items, blobs)
        |> Enum.map(fn {it, b} ->
          hash = Base.encode16(:crypto.hash(:sha256, b), case: :lower) == it["sha256"] and byte_size(b) == it["size"]
          %{i: it["i"], name: it["name"], kind: it["kind"], size: it["size"], clauses: it["clauses"] || [], hash: hash,
            check: check_item(it["kind"], b, opts)}
        end)
      end

    root = Base.encode16(Merkle.root(Enum.map(blobs, &Merkle.leaf(:crypto.hash(:sha256, &1)))), case: :lower) == m["root"]

    anchors =
      Enum.map(d.anchors, fn r ->
        case Keyword.get(opts, :log_keys) do
          nil -> %{index: r["index"], check: :not_checked}
          keys ->
            case Vapor.Tlog.verify_receipt(r, anchor_entry(d), log_keys: keys) do
              {:ok, cp} -> %{index: r["index"], check: :ok, origin: cp.origin}
              {:error, why} -> %{index: r["index"], check: {:fail, inspect(why)}}
            end
        end
      end)

    ok = m["format"] == @format and root and good_sigs >= quorum and Enum.all?(item_reports, &(&1.hash and &1.check in [:ok, :na])) and
           Enum.all?(anchors, &(&1.check in [:ok, :not_checked]))

    report = %{ok: ok, format: m["format"], root: root, signatures: sigs, quorum: {good_sigs, quorum}, items: item_reports, anchors: anchors}
    if ok, do: {:ok, report}, else: {:error, report}
  end

  @doc """
  A complete dossier from this checkout, for demonstration and for the
  console: a compiled program's certificate (the six-rung ladder), an
  attested agent journal, the certificate anchored in a transparency log,
  a model contract, and `opts[:quality]` (a quality report's JSON) when
  given — signed by a fresh key and a witness, anchored in a fresh log.
  Returns `%{dossier, signer, witness, log_key}`.
  """
  def demo(opts \\ []) do
    alias Vapor.Algebra.Term, as: T
    key = Certificate.keygen()
    witness = Certificate.keygen()

    w = Vapor.Quant.Sb4.quantize(Vapor.Tensor.random(:f32, [32, 256], 7, scale: 0.3))
    prog = Vapor.Program.new(y: T.silu(T.qgemv(T.const(w), T.input(:x, :f32, [256]))))
    {:ok, compiled} = Vapor.compile(prog, key: key)

    j =
      Journal.new("demo-run")
      |> Journal.append("input", %{"text" => "summarise the contract's penalty clause"})
      |> Journal.append("tool_call", %{"name" => "search", "arguments" => %{"q" => "multa contratual"}})
      |> Journal.append("tool_result", %{"name" => "search", "result" => "contrato.pdf#p3: multa de 2%"})
      |> Journal.append("answer", %{"text" => "A multa é de 2% (contrato.pdf, p. 3)."})

    log_key = Vapor.Tlog.Note.keygen("audit.demo/log")
    entry = "certificate " <> Canonical.hex_digest(Certificate.canonical(compiled.certificate))
    {log, receipt} = Vapor.Tlog.anchor(Vapor.Tlog.new("audit.demo/log"), entry, log_key.signer)

    {:ok, c} = Vapor.Model.Config.from_map(%{"model_type" => "qwen2", "vocab_size" => 96, "hidden_size" => 64, "intermediate_size" => 96,
                                             "num_hidden_layers" => 2, "num_attention_heads" => 4, "num_key_value_heads" => 2,
                                             "max_position_embeddings" => 32, "rms_norm_eps" => 1.0e-5, "rope_theta" => 10_000.0})

    items =
      [certificate(compiled.certificate, "compiled program (6-rung ladder)"), journal(j, Journal.attest(j, key), "agent run demo-run"),
       tlog_receipt(receipt, entry, log_key.verifier, "certificate anchored in the log"), contract(Vapor.Lock.spec(c), "model contract (qwen2, tiny)")] ++
        if(q = opts[:quality], do: [quality(q, "quality report (mix vapor.quality)")], else: [])

    d = build(items, system: %{name: "vapor demo", version: to_string(Application.spec(:vapor, :vsn)), provider: "this checkout"},
              date: Keyword.get(opts, :date, Date.to_iso8601(Date.utc_today())), notes: "generated by Vapor.Audit.demo/1")
    d = d |> sign(key) |> sign(witness)
    {_log, d} = anchor(d, log, log_key.signer)
    %{dossier: d, signer: key, witness: witness, log_key: log_key}
  end

  # evidence is decoded refusing unknown atoms (untrusted input must not
  # grow the atom table): the atoms vapor itself writes exist once its
  # modules are loaded — in a fresh VM (a verifier node) they may not be yet
  defp known_atoms do
    unless :persistent_term.get({__MODULE__, :atoms}, false) do
      for mod <- Application.spec(:vapor, :modules) || [], do: Code.ensure_loaded(mod)
      :persistent_term.put({__MODULE__, :atoms}, true)
    end
  end

  # each kind re-checked by its own rules, from its bytes
  defp check_item("certificate", b, opts) do
    case Certificate.decode(b) do
      {:ok, c} ->
        keys = Keyword.get(opts, :certificate_keys) || Enum.map(c.signatures, &elem(&1, 0))
        if c.signatures != [] and Certificate.verify(c, keys, 1) == :ok, do: :ok, else: {:fail, "certificate signatures"}

      _ ->
        {:fail, "not a certificate"}
    end
  end

  defp check_item("journal", b, _opts) do
    with {:ok, {:vapor_journal_evidence, 1, jbin, att}} <- Canonical.decode(b),
         {:ok, j} <- Journal.decode(jbin),
         :ok <- Journal.verify(j),
         true <- Journal.attested?(j, att, [att["key"]]) || {:error, :attestation} do
      :ok
    else
      {:error, why} -> {:fail, "journal: #{inspect(why)}"}
      other -> {:fail, "journal: #{inspect(other)}"}
    end
  end

  defp check_item("tlog_receipt", b, opts) do
    with {:ok, {:vapor_tlog_evidence, 1, receipt, entry, vkey}} <- Canonical.decode(b),
         keys = Keyword.get(opts, :log_keys, [vkey]),
         {:ok, _} <- Vapor.Tlog.verify_receipt(receipt, entry, log_keys: keys) do
      :ok
    else
      {:error, why} -> {:fail, "log receipt: #{inspect(why)}"}
      other -> {:fail, "log receipt: #{inspect(other)}"}
    end
  end

  defp check_item("quality", b, _opts) do
    case Vapor.JSON.decode(b) do
      {:ok, r} ->
        cs = all_checks(r)
        failed = Enum.reject(cs, &(&1["pass"] == true))
        if cs != [] and failed == [], do: :ok, else: {:fail, "#{length(failed)} of #{length(cs)} quality checks fail"}

      _ ->
        {:fail, "quality report is not JSON"}
    end
  end

  defp check_item(_kind, _b, _opts), do: :na

  # ------------------------------------------------------------------ PDF --

  @doc """
  A printable PDF report of the dossier (Helvetica, A4), with the dossier
  file attached (`/EmbeddedFiles`, `dossier.vdossier`). The PDF itself is
  not PAdES-signed: the evidence inside it is, and `verify/2` (or the HTML
  page) checks it.
  """
  def pdf(d, opts \\ []) do
    title = Keyword.get(opts, :title, "Audit dossier")
    m = d.manifest
    sys = m["system"] || %{}

    lines =
      [{:h1, title},
       {:p, "#{sys["name"] || "system"} #{sys["version"] || ""} — #{m["date"]}"},
       {:mono, "root   " <> m["root"]},
       {:mono, "format " <> m["format"]},
       {:sp, nil},
       {:h2, "Signatures"}] ++
        Enum.map(d.signatures, fn s -> {:mono, "Ed25519 " <> Certificate.key_id(s["key"])} end) ++
        [{:sp, nil}, {:h2, "Evidence"}] ++
        Enum.flat_map(m["items"], fn it ->
          [{:p, "#{it["i"] + 1}. [#{it["kind"]}] #{it["name"]} (#{it["size"]} bytes)"},
           {:mono, "   sha256 " <> it["sha256"]},
           {:small, "   " <> Enum.map_join(it["meta"] || %{}, "  ", fn {k, v} -> "#{k}: #{inspect_short(v)}" end)},
           {:small, "   relevant to: " <> Enum.join(it["clauses"] || [], ", ")}]
        end) ++
        [{:sp, nil}, {:h2, "Clauses"}] ++
        (for {id, t} <- Enum.sort(m["clauses"] || %{}) do
           n = Enum.count(m["items"], &(id in (&1["clauses"] || [])))
           {:p, "#{id} — #{t}  [#{n} item#{if n == 1, do: "", else: "s"}]"}
         end) ++
        [{:sp, nil}, {:small, m["disclaimer"]},
         {:small, "Verify: mix vapor.audit verify <this file>  (the attached dossier.vdossier is checked: hashes, Merkle root, signatures, each item)."}]

    write_pdf(lines, encode(d))
  end

  defp inspect_short(v) when is_binary(v), do: if(String.length(v) > 40, do: String.slice(v, 0, 16) <> "…", else: v)
  defp inspect_short(v), do: inspect(v)

  # a minimal PDF: pages of text lines in the standard Helvetica/Courier
  # fonts (WinAnsi), and one embedded file
  defp write_pdf(lines, attachment) do
    {pw, ph, margin} = {595, 842, 56}

    {pages, cur, _y} =
      Enum.reduce(lines, {[], [], ph - margin}, fn line, {pages, cur, y} ->
        {font, size, lead} =
          case line do
            {:h1, _} -> {"F2", 18, 26}
            {:h2, _} -> {"F2", 12, 20}
            {:mono, _} -> {"F3", 8, 12}
            {:small, _} -> {"F1", 7.5, 11}
            {:sp, _} -> {"F1", 6, 8}
            _ -> {"F1", 10, 14}
          end

        texts = case line do
          {:sp, _} -> []
          {_, t} -> wrap(t || "", if(font == "F3", do: 92, else: round(480 / (size * 0.5))))
        end

        Enum.reduce(if(texts == [], do: [""], else: texts), {pages, cur, y}, fn t, {pages, cur, y} ->
          {pages, cur, y} = if y - lead < margin, do: {pages ++ [cur], [], ph - margin}, else: {pages, cur, y}
          op = "BT /#{font} #{size} Tf #{margin} #{y - lead} Td (#{pdf_escape(t)}) Tj ET"
          {pages, cur ++ [op], y - lead}
        end)
      end)

    pages = pages ++ [cur]
    n = length(pages)
    # objects: 1 catalog, 2 pages, 3–5 fonts, 6 embedded file, 7 filespec, then page + content pairs
    page_objs = for i <- 0..(n - 1), do: {8 + 2 * i, 9 + 2 * i}

    objs =
      [{1, "<< /Type /Catalog /Pages 2 0 R /Names << /EmbeddedFiles << /Names [(dossier.vdossier) 7 0 R] >> >> /PageMode /UseAttachments >>"},
       {2, "<< /Type /Pages /Kids [#{Enum.map_join(page_objs, " ", fn {p, _} -> "#{p} 0 R" end)}] /Count #{n} >>"},
       {3, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"},
       {4, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"},
       {5, "<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>"},
       {6, ["<< /Type /EmbeddedFile /Subtype /application#2Fcbor /Length #{byte_size(attachment)} /Params << /Size #{byte_size(attachment)} >> >>\nstream\n", attachment, "\nendstream"]},
       {7, "<< /Type /Filespec /F (dossier.vdossier) /UF (dossier.vdossier) /EF << /F 6 0 R >> /Desc (vapor audit dossier) >>"}] ++
        Enum.flat_map(Enum.zip(page_objs, pages), fn {{p, c}, ops} ->
          content = Enum.join(ops, "\n")
          [{p, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 #{pw} #{ph}] /Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> /Contents #{c} 0 R >>"},
           {c, ["<< /Length #{byte_size(content)} >>\nstream\n", content, "\nendstream"]}]
        end)

    head = "%PDF-1.7\n%\xE2\xE3\xCF\xD3\n"

    {body, offsets, _} =
      Enum.reduce(Enum.sort_by(objs, &elem(&1, 0)), {[], [], byte_size(head)}, fn {num, o}, {acc, offs, pos} ->
        chunk = IO.iodata_to_binary(["#{num} 0 obj\n", o, "\nendobj\n"])
        {[acc, chunk], [pos | offs], pos + byte_size(chunk)}
      end)

    offsets = Enum.reverse(offsets)
    xref_at = byte_size(head) + IO.iodata_length(body)
    size = length(objs) + 1

    IO.iodata_to_binary([head, body, "xref\n0 #{size}\n0000000000 65535 f \n",
                         Enum.map(offsets, &(String.pad_leading(Integer.to_string(&1), 10, "0") <> " 00000 n \n")),
                         "trailer\n<< /Size #{size} /Root 1 0 R >>\nstartxref\n#{xref_at}\n%%EOF\n"])
  end

  defp wrap(text, width) do
    text
    |> String.split(" ")
    |> Enum.reduce([""], fn w, [cur | rest] ->
      cond do
        cur == "" -> [w | rest]
        String.length(cur) + 1 + String.length(w) <= width -> [cur <> " " <> w | rest]
        true -> [w, cur | rest]
      end
    end)
    |> Enum.reverse()
  end

  # WinAnsi: Latin-1 bytes for what fits (accents included), "?" otherwise
  defp pdf_escape(t) do
    t
    |> String.replace("—", "-")
    |> String.replace("…", "...")
    |> String.to_charlist()
    |> Enum.map(fn
      c when c in [?(, ?), ?\\] -> [?\\, c]
      c when c < 256 -> c
      _ -> ??
    end)
    |> :erlang.list_to_binary()
  end

  @doc "The dossier attached to a PDF report (`pdf/2`): `{:ok, bytes}`."
  def extract(pdf) do
    with {pos, _} <- :binary.match(pdf, "/Type /EmbeddedFile"),
         {spos, _} <- :binary.match(pdf, "stream\n", scope: {pos, byte_size(pdf) - pos}),
         [_, len] <- Regex.run(~r/\/Length (\d+)/, binary_part(pdf, pos, spos - pos)) do
      len = String.to_integer(len)
      data = binary_part(pdf, spos + 7, len)
      flate = binary_part(pdf, pos, spos - pos) =~ "FlateDecode"
      {:ok, if(flate, do: :zlib.uncompress(data), else: data)}
    else
      _ -> {:error, :no_dossier_attached}
    end
  rescue
    _ -> {:error, :no_dossier_attached}
  end

  # ----------------------------------------------------------------- HTML --

  @doc """
  The dossier as one offline HTML page that verifies it in the browser:
  SHA-256 of every item against the manifest, the Merkle root, and each
  Ed25519 signature over the signed bytes (WebCrypto); the manifest shown
  is decoded *from the signed bytes* by the page's own CBOR reader. No
  network, no script from elsewhere. Items are embedded (base64).
  """
  def html(d) do
    data = %{
      "signed" => Base.encode64(signed_bytes(d)),
      "blobs" => Enum.map(d.blobs, &Base.encode64/1),
      "sigs" => Enum.map(d.signatures, fn s -> %{"key" => Base.encode64(s["key"]), "sig" => Base.encode64(s["sig"]),
                                                   "id" => Certificate.key_id(s["key"])} end),
      "anchors" => length(d.anchors)
    }

    json = data |> Vapor.JSON.encode() |> String.replace("</", "<\\/")
    String.replace(page_template(), "__DOSSIER__", json)
  end

  defp page_template do
    ~S"""
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audit dossier</title>
<style>
:root{--bg:#f6f4ef;--fg:#1d1c1a;--mut:#6b675e;--card:#fffdf8;--line:#e3ded2;--ok:#1f7a4d;--bad:#b3261e;--acc:#2a4d8f;--chip:#ece7db}
@media (prefers-color-scheme:dark){:root{--bg:#141412;--fg:#ece9e1;--mut:#a29d91;--card:#1c1b18;--line:#2d2b26;--ok:#5cc28f;--bad:#ff8a80;--acc:#8fb1ff;--chip:#26241f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 16px 64px}
h1{font:600 26px/1.2 ui-serif,Georgia,serif;margin:0 0 4px}h2{font:600 15px/1.3 ui-sans-serif,system-ui;letter-spacing:.04em;text-transform:uppercase;color:var(--mut);margin:32px 0 10px}
.sub{color:var(--mut)}.mono{font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}
.verdict{display:flex;gap:14px;align-items:center;margin:22px 0;padding:16px 18px;border-radius:14px;border:1px solid var(--line);background:var(--card)}
.dot{width:14px;height:14px;border-radius:50%;background:var(--mut);flex:none}.ok .dot{background:var(--ok)}.bad .dot{background:var(--bad)}
.verdict b{font-size:17px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.k{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
td,th{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-size:13.5px}th{font-weight:600;color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:0}td.s-ok,td.s-bad{white-space:nowrap}.chip{display:inline-block;background:var(--chip);border-radius:999px;padding:1px 8px;margin:2px 3px 2px 0;font-size:11.5px}
.s-ok{color:var(--ok);font-weight:600}.s-bad{color:var(--bad);font-weight:600}.s-mut{color:var(--mut)}
.matrix td.c{text-align:center}.matrix .y{color:var(--ok);font-weight:700}.foot{margin-top:34px;color:var(--mut);font-size:12.5px}
details summary{cursor:pointer;color:var(--acc)}
</style></head><body><main>
<h1 id="t">Audit dossier</h1><div class="sub" id="sys"></div>
<div class="verdict" id="v"><span class="dot"></span><div><b id="vt">Checking…</b><div class="sub" id="vd">hashes, Merkle root and signatures are being recomputed in this browser</div></div></div>
<div class="grid" id="facts"></div>
<h2>Evidence</h2><table id="items"><thead><tr><th>#</th><th>Item</th><th>SHA-256</th><th>Hash</th></tr></thead><tbody></tbody></table>
<h2>Signatures</h2><table id="sigs"><thead><tr><th>Key</th><th>Ed25519</th></tr></thead><tbody></tbody></table>
<h2>Clauses × evidence</h2><table class="matrix" id="mx"><thead></thead><tbody></tbody></table>
<div class="foot" id="disc"></div>
<details class="foot"><summary>What this page checks</summary><p>The manifest shown here is decoded by this page from the exact bytes that were signed (canonical CBOR). Every item's SHA-256 is recomputed and compared with the manifest; the Merkle root (RFC 6962 leaves over the items' hashes) is recomputed; each Ed25519 signature is checked over the signed bytes with the browser's WebCrypto. Item-specific checks (certificate signatures, journal hash chains, log receipts) run in <span class="mono">mix vapor.audit verify</span>.</p></details>
</main>
<script>
const D = __DOSSIER__;
const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const hex = a => Array.from(a, x => x.toString(16).padStart(2, "0")).join("");
const sha = async b => new Uint8Array(await crypto.subtle.digest("SHA-256", b));
const cat = (...as) => { const n = as.reduce((s, a) => s + a.length, 0), o = new Uint8Array(n); let p = 0; for (const a of as) { o.set(a, p); p += a.length } return o };
// canonical CBOR, the vapor profile (byte strings, tags 39 atoms / 30305 tuples)
function cbor(buf) {
  let p = 0; const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength), td = new TextDecoder();
  const arg = info => info < 24 ? info : info === 24 ? buf[p++] : info === 25 ? (p += 2, dv.getUint16(p - 2)) : info === 26 ? (p += 4, dv.getUint32(p - 4)) : (p += 8, Number(dv.getBigUint64(p - 8)));
  function item() {
    const b = buf[p++], major = b >> 5, info = b & 31;
    if (b === 0xf6) return null; if (b === 0xf5) return true; if (b === 0xf4) return false;
    if (b === 0xf9) { p += 2; const h = dv.getUint16(p - 2), s = h >> 15 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023; return s * (e === 0 ? f * 2 ** -24 : e === 31 ? (f ? NaN : Infinity) : (1 + f / 1024) * 2 ** (e - 15)) }
    if (b === 0xfa) { p += 4; return dv.getFloat32(p - 4) } if (b === 0xfb) { p += 8; return dv.getFloat64(p - 8) }
    const n = arg(info);
    switch (major) {
      case 0: return n; case 1: return -1 - n;
      case 2: { const s = buf.slice(p, p + n); p += n; return { bytes: s, toString() { return td.decode(s) } } }
      case 3: { const s = td.decode(buf.slice(p, p + n)); p += n; return s }
      case 4: { const a = []; for (let i = 0; i < n; i++) a.push(item()); return a }
      case 5: { const m = new Map(); for (let i = 0; i < n; i++) { const k = item(); m.set(String(k), item()) } return m }
      case 6: { const v = item(); return n === 39 ? { atom: v } : n === 30305 ? { tuple: v } : v }
    }
  }
  return item();
}
const s = v => v == null ? "" : typeof v === "object" && v.bytes ? new TextDecoder().decode(v.bytes) : String(v);
const esc = t => s(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
async function leaf(h) { return sha(cat(new Uint8Array([0]), h)) }
async function node(a, b) { return sha(cat(new Uint8Array([1]), a, b)) }
async function mroot(hs) { if (hs.length === 0) return sha(new Uint8Array()); if (hs.length === 1) return hs[0]; let k = 1; while (k * 2 < hs.length) k *= 2; return node(await mroot(hs.slice(0, k)), await mroot(hs.slice(k))) }
(async () => {
  const signed = b64(D.signed), top = cbor(signed), m = top.tuple[2];
  const get = (mp, k) => mp instanceof Map ? mp.get(k) : undefined;
  const sys = get(m, "system") || new Map();
  document.getElementById("t").textContent = "Audit dossier — " + (s(get(sys, "name")) || "system");
  document.getElementById("sys").textContent = [s(get(sys, "version")), s(get(sys, "provider")), s(get(m, "date"))].filter(Boolean).join(" · ");
  const items = get(m, "items") || [];
  let allOk = true; const leaves = [];
  const tb = document.querySelector("#items tbody");
  for (let i = 0; i < items.length; i++) {
    const it = items[i], blob = b64(D.blobs[i] || ""), h = await sha(blob);
    leaves.push(await leaf(h));
    const ok = hex(h) === s(get(it, "sha256")) && blob.length === get(it, "size"); allOk = allOk && ok;
    const meta = get(it, "meta"), metaTxt = meta instanceof Map ? Array.from(meta).map(([k, v]) => `${esc(k)}: ${esc(v)}`).join(" · ") : "";
    const cl = (get(it, "clauses") || []).map(c => `<span class="chip">${esc(c)}</span>`).join("");
    tb.insertAdjacentHTML("beforeend", `<tr><td>${i + 1}</td><td><b>${esc(get(it, "name"))}</b> <span class="chip">${esc(get(it, "kind"))}</span><div class="sub" style="font-size:12.5px">${metaTxt}</div><div>${cl}</div></td><td class="mono">${esc(get(it, "sha256"))}</td><td class="${ok ? "s-ok" : "s-bad"}">${ok ? "✓ matches" : "✗ altered"}</td></tr>`);
  }
  const root = hex(await mroot(leaves)), rootOk = root === s(get(m, "root")); allOk = allOk && rootOk;
  const st = document.querySelector("#sigs tbody"); let goodSigs = 0, noCrypto = false;
  for (const g of D.sigs) {
    let txt = "", cls = "s-mut";
    try {
      const k = await crypto.subtle.importKey("raw", b64(g.key), { name: "Ed25519" }, false, ["verify"]);
      const ok = await crypto.subtle.verify("Ed25519", k, b64(g.sig), signed);
      txt = ok ? "✓ valid" : "✗ invalid"; cls = ok ? "s-ok" : "s-bad"; if (ok) goodSigs++; else allOk = false;
    } catch (e) { txt = "not checkable in this browser (no Ed25519 in WebCrypto)"; noCrypto = true }
    st.insertAdjacentHTML("beforeend", `<tr><td class="mono">${esc(g.id)}</td><td class="${cls}">${txt}</td></tr>`);
  }
  if (D.sigs.length === 0) { st.insertAdjacentHTML("beforeend", `<tr><td colspan="2" class="s-bad">unsigned</td></tr>`); allOk = false }
  const facts = [["Merkle root", `<span class="mono">${esc(get(m, "root"))}</span><div class="${rootOk ? "s-ok" : "s-bad"}">${rootOk ? "✓ recomputed" : "✗ differs"}</div>`],
                 ["Items", items.length], ["Signatures", `${goodSigs} valid of ${D.sigs.length}`], ["Format", esc(get(m, "format"))], ["Log anchors", D.anchors]];
  document.getElementById("facts").innerHTML = facts.map(([k, v]) => `<div class="card"><div class="k">${k}</div><div>${v}</div></div>`).join("");
  const clauses = get(m, "clauses") || new Map(), ids = Array.from(clauses.keys()).sort();
  document.querySelector("#mx thead").innerHTML = `<tr><th>Clause</th>${items.map((_, i) => `<th>${i + 1}</th>`).join("")}</tr>`;
  document.querySelector("#mx tbody").innerHTML = ids.map(id => `<tr><td><b>${esc(id)}</b><div class="sub" style="font-size:12px">${esc(clauses.get(id))}</div></td>${items.map(it => (get(it, "clauses") || []).map(s).includes(id) ? `<td class="c y">●</td>` : `<td class="c s-mut">·</td>`).join("")}</tr>`).join("");
  document.getElementById("disc").textContent = s(get(m, "disclaimer"));
  const v = document.getElementById("v");
  v.className = "verdict " + (allOk ? "ok" : "bad");
  document.getElementById("vt").textContent = allOk ? (noCrypto ? "Hashes and root verified; signatures not checkable here" : "Verified in this browser") : "Verification failed";
  document.getElementById("vd").textContent = allOk ? `${items.length} items match their hashes, the Merkle root recomputes, ${goodSigs} signature(s) valid over the signed manifest.` : "Something in this dossier was altered or is unsigned — see the rows marked ✗.";
})();
</script></body></html>
"""
  end
end
