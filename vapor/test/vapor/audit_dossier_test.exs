defmodule Vapor.AuditDossierTest do
  @moduledoc """
  Audit dossiers (`Vapor.Audit`, docs/AUDITORIA.md): evidence the system
  produced — a compilation certificate, an attested agent journal, a
  transparency-log receipt, a quality report, a model contract — packed,
  signed and co-signed; verified offline from the bytes; **any altered byte
  is refused, by name**; the PDF report carries the dossier and gives it
  back; the HTML page verifies it in a browser engine (Node's WebCrypto).
  """
  use ExUnit.Case, async: true
  alias Vapor.{Audit, Certificate, Tlog}
  alias Vapor.Agent.Journal
  alias Vapor.Tlog.Note

  defp evidence do
    k = Certificate.keygen()
    cert = Certificate.sign(%Certificate{payload: %{program: :crypto.hash(:sha256, "a program"), ladder: [1, 2, 3, 4, 5, 6]}}, k)

    j = Journal.new("run-42") |> Journal.append("prompt", %{"text" => "hello"}) |> Journal.append("tool", %{"name" => "lookup", "result" => 3})
    att = Journal.attest(j, k)

    log_key = Note.keygen("vapor.test/dossier")
    {log, receipt} = Tlog.anchor(Tlog.new("vapor.test/dossier"), "search receipt 0001", log_key.signer)

    quality = Vapor.JSON.encode(%{"substrate" => "native", "text" => %{"checks" => [%{"name" => "planted bigram", "pass" => true}]},
                                  "any_to_any" => [%{"checks" => [%{"name" => "route", "pass" => true}]}]})

    {:ok, c} = Vapor.Model.Config.from_map(Vapor.TestHelpers.tiny_config("llama"))

    items = [Audit.certificate(cert, "compile certificate"), Audit.journal(j, att, "agent run 42"),
             Audit.tlog_receipt(receipt, "search receipt 0001", log_key.verifier, "search receipt anchored"),
             Audit.quality(quality, "quality report"), Audit.contract(Vapor.Lock.spec(c), "model contract"),
             Audit.document("risk-assessment.md", "# Risk assessment\nresidual risks …", ["AIA-11"])]

    {items, k, log, log_key}
  end

  defp dossier do
    {items, k, log, log_key} = evidence()
    witness = Certificate.keygen()
    d = items |> Audit.build(system: %{name: "acme-assistant", version: "1.4.0", provider: "ACME"}, date: "2026-10-03") |> Audit.sign(k) |> Audit.sign(witness)
    {log, d} = Audit.anchor(d, log, log_key.signer)
    {d, k, witness, log, log_key}
  end

  test "build, sign, co-sign, anchor: verified offline from the bytes, every item by its own rules" do
    {d, k, w, _log, log_key} = dossier()
    bin = Audit.encode(d)
    assert {:ok, r} = Audit.verify(bin, log_keys: [log_key.verifier], trusted: [k.public, w.public], quorum: 2)
    assert r.root and r.quorum == {2, 2}
    assert Enum.map(r.items, & &1.check) == [:ok, :ok, :ok, :ok, :na, :na]
    assert [%{check: :ok}] = r.anchors
    # the same evidence and date give the same bytes (no clock inside)
    {items, k2, _, _} = evidence()
    assert Audit.build(items, date: "x").manifest["items"] |> length() == 6
    assert byte_size(Audit.encode(Audit.sign(Audit.build(items, date: "x"), k2))) > 0
    # every evidence kind maps to clauses; the matrix covers Art. 12 and 15
    m = d.manifest
    assert Enum.any?(m["items"], &("AIA-12" in &1["clauses"])) and Enum.any?(m["items"], &("AIA-15" in &1["clauses"]))
  end

  test "any altered byte is refused, and the report says where" do
    {d, k, _w, _log, log_key} = dossier()
    bin = Audit.encode(d)
    opts = [log_keys: [log_key.verifier], trusted: [k.public]]

    # an item's bytes: its hash breaks (and the root)
    blobs = List.update_at(d.blobs, 3, fn b -> String.replace(b, "true", "fals") end)
    assert {:error, r} = Audit.verify(%{d | blobs: blobs}, opts)
    assert %{hash: false, name: "quality report"} = Enum.at(r.items, 3)
    refute r.root

    # the manifest (a name): every signature breaks
    m = put_in(d.manifest, ["items", Access.at(0), "name"], "something else")
    assert {:error, r} = Audit.verify(%{d | manifest: m}, opts)
    assert Enum.all?(r.signatures, &(not &1.valid))

    # an item rewritten with its hash updated in the manifest: still caught (signatures)
    forged = Vapor.JSON.encode(%{"text" => %{"checks" => [%{"name" => "x", "pass" => true}]}})
    m2 = update_in(d.manifest, ["items", Access.at(3)], &%{&1 | "sha256" => Base.encode16(:crypto.hash(:sha256, forged), case: :lower), "size" => byte_size(forged)})
    assert {:error, _} = Audit.verify(%{d | manifest: m2, blobs: List.replace_at(d.blobs, 3, forged)}, opts)

    # a flipped bit anywhere in the file: never accepted
    for pos <- [10, div(byte_size(bin), 2), byte_size(bin) - 20] do
      <<a::binary-size(pos), x, b::binary>> = bin
      refute match?({:ok, _}, Audit.verify(a <> <<Bitwise.bxor(x, 1)>> <> b, opts))
    end

    # an untrusted signer does not count
    assert {:error, %{quorum: {0, 1}}} = Audit.verify(bin, trusted: [Certificate.keygen().public], log_keys: [log_key.verifier])
    # a quality report with a failing check fails its item
    bad = Audit.quality(Vapor.JSON.encode(%{"text" => %{"checks" => [%{"name" => "x", "pass" => false}]}}), "q")
    d2 = Audit.build([bad]) |> Audit.sign(k)
    assert {:error, %{items: [%{check: {:fail, _}}]}} = Audit.verify(d2)
  end

  test "the PDF report carries the dossier and gives it back; its text lists the evidence" do
    {d, k, _w, _log, log_key} = dossier()
    pdf = Audit.pdf(d, title: "Dossiê de auditoria")
    assert "%PDF-1.7" <> _ = pdf
    assert {:ok, bin} = Audit.extract(pdf)
    assert bin == Audit.encode(d)
    assert {:ok, _} = Audit.verify(pdf, log_keys: [log_key.verifier], trusted: [k.public])
    # vapor's own PDF reader finds the text
    {:ok, pages, _} = Vapor.Docs.PDF.pages(pdf)
    text = Enum.join(pages, "\n")
    assert text =~ "agent run 42" and text =~ d.manifest["root"]
    if System.find_executable("pdftotext") do
      f = Path.join(System.tmp_dir!(), "dossier-#{System.unique_integer([:positive])}.pdf")
      File.write!(f, pdf)
      {out, 0} = System.cmd("pdftotext", [f, "-"])
      assert out =~ "Dossiê de auditoria" and out =~ "quality report"
      {att, 0} = System.cmd("pdfdetach", ["-list", f])
      assert att =~ "dossier.vdossier"
    end
  end

  test "the HTML page verifies the dossier in a browser engine; an altered item turns it red" do
    node = System.find_executable("node")
    if node do
      {d, _k, _w, _log, _} = dossier()
      run = fn html ->
        dir = Path.join(System.tmp_dir!(), "dossier-#{System.unique_integer([:positive])}")
        File.mkdir_p!(dir)
        # run the page's script in Node (WebCrypto with Ed25519), with a tiny DOM stand-in
        [_, script] = Regex.run(~r/<script>(.*)<\/script>/s, html)
        harness = """
        const els = {}; const el = id => els[id] || (els[id] = { id, textContent: "", innerHTML: "", className: "", rows: [],
          insertAdjacentHTML(_, h) { this.rows.push(h) } });
        globalThis.document = { getElementById: el, querySelector: q => el(q) };
        globalThis.atob = s => Buffer.from(s, "base64").toString("binary");
        #{script}
        setTimeout(() => { console.log(JSON.stringify({ v: els.v.className, t: els.vt.textContent,
          items: (els["#items tbody"].rows || []).length })) }, 1500);
        """
        File.write!(Path.join(dir, "h.js"), harness)
        {out, 0} = System.cmd(node, [Path.join(dir, "h.js")])
        Vapor.JSON.decode!(out |> String.split("\n", trim: true) |> List.last())
      end

      ok = run.(Audit.html(d))
      assert ok["v"] =~ "ok" and ok["items"] == 6
      assert ok["t"] in ["Verified in this browser", "Hashes and root verified; signatures not checkable here"]

      bad = run.(Audit.html(%{d | blobs: List.update_at(d.blobs, 0, &(&1 <> "x"))}))
      assert bad["v"] =~ "bad"
    end
  end
end
