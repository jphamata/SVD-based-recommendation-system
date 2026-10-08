defmodule Vapor.Quality.Round08 do
  @moduledoc """
  Quality checks for the 0.8 round, in the suite's discipline: every check
  has a value, a **control** that a broken or naive implementation would
  produce, and a threshold that separates them.

  | check | value | control (must fail) |
  |---|---|---|
  | table structure | ICDAR-2013 adjacency F1 of 12 scanned tables (4 styles, unseen fonts) | the 0.7 reading (lines in reading order, one column) |
  | merged cells | adjacency F1 on the grid tables (headers spanning rows and columns) | the same grids with span detection off |
  | cell text | mean cell CER, typed decoding (column types and shapes) | the free reading of the same cells |
  | GPU sessions | logits of a decode loop, resident GPU session vs CPU session; recordings replayed | the bits of a session whose state was reset between steps |
  | sparse 4-bit experts | output bits of a 4-bit MoE, predicated vs dense | the same model with one expert's weights changed (the comparison sees it) |
  | Mamba-2 | logits against transformers' own (shipped fixture), per-group and whole-width gated norm | each setting against the other's reference |
  | JBIG2 | generic-region and symbol streams decoded to the encoder's bitmap (SHA-256) | the same streams with the template misdeclared |
  | audit dossier | a dossier verifies offline; any altered byte is refused | the altered dossier itself (must not verify) |
  | shards across nodes | logits of a model split over BEAM nodes = one node | a shard computed with another node's slice |

  Rows are added by the modules of the round as they land; `run/1` returns
  `%{checks, tables}`.
  """
  import Bitwise
  alias Vapor.Vision.{OCR, Table}

  @priv "priv/quality"

  def run(opts \\ []) do
    w = Keyword.get(opts, :worker)
    {tables, table_checks} = tables(w)
    checks = List.flatten([table_checks, jbig2(w), audit(), cluster(w), mamba2(w), gpu(w), sb4(w)])
    %{checks: checks, tables: tables}
  end

  @doc false
  def __check__(f, w), do: (case f do :mamba2 -> mamba2(w); :gpu -> gpu(w); :sb4 -> sb4(w) end)

  defp check(name, value, control, threshold, pass), do: %{name: name, value: value, control: control, threshold: threshold, pass: pass}
  defp p("priv/" <> rel), do: Path.join(to_string(:code.priv_dir(:vapor)), rel)
  defp p(rel), do: Path.expand(rel)

  # ---------------------------------------------------- shards across nodes --

  # a model split over peer nodes = one worker, bit for bit, including after
  # a node is lost; the control computes with the shards in the wrong places
  defp cluster(nil), do: []

  defp cluster(_w) do
    with true <- Node.alive?() or start_dist(),
         paths = Enum.flat_map(:code.get_path(), &[~c"-connect_all", ~c"false", ~c"-pa", &1]) |> Enum.drop(2),
         peers when length(peers) == 3 <- start_peers(paths) do
      try do
        alias Vapor.Shard.Cluster
        x = Vapor.Tensor.random(:f32, [4, 256], 31)
        w = Vapor.Tensor.random(:f32, [120, 256], 32, scale: 0.1)
        {:ok, wk} = Vapor.Runtime.Worker.start_link(exec: [Vapor.Runtime.Substrates.binary("vapor-worker", "native")])
        {:ok, ref} = Vapor.Shard.linear([wk], x, w)
        {:ok, c} = Cluster.start(Enum.map(peers, &elem(&1, 1)))
        {:ok, c} = Cluster.load(c, :w, w)
        {:ok, y, c, _} = Cluster.linear(c, :w, x)
        # control: each node given another node's slice (the rows rotated by one shard)
        rot = Vapor.Tensor.new(:f32, [120, 256], binary_part(w.data, 40 * 256 * 4, 80 * 256 * 4) <> binary_part(w.data, 0, 40 * 256 * 4))
        {:ok, c} = Cluster.load(c, :rot, rot)
        {:ok, ys, c, _} = Cluster.linear(c, :rot, x)
        :peer.stop(elem(hd(peers), 0))
        {:ok, y2, _, rep} = Cluster.linear(c, :w, x)
        same = y == ref and y2 == ref

        [check("shards across BEAM nodes: a matrix split over 3 peer nodes = one worker, and again after a node is lost (#{length(rep.replaced)} shard moved)",
               same, "#{ys == ref} (shards in the wrong places)", "identical, control differs", same and ys != ref)]
      after
        for {pid, _} <- peers, do: (try do :peer.stop(pid) catch _, _ -> :ok end)
      end
    else
      _ -> []
    end
  end

  # Distributed Erlang needs epmd; the product never shells out to start it
  # (test/vapor/audit_test.exs), so without it the cluster check is not run
  defp start_dist do
    match?({:ok, _}, Node.start(:"vapor_quality_#{System.unique_integer([:positive])}@127.0.0.1", :longnames))
  rescue
    _ -> false
  end

  defp start_peers(paths) do
    for i <- 1..3 do
      {:ok, pid, node} = :peer.start(%{name: :"vapor_q08_#{i}_#{System.unique_integer([:positive])}", host: ~c"127.0.0.1", longnames: true, args: paths})
      {:ok, _} = :erpc.call(node, Application, :ensure_all_started, [:vapor])
      {pid, node}
    end
  rescue
    _ -> []
  end

  # ------------------------------------------------------------- audit --

  # a dossier verifies offline; every single-byte alteration of the file
  # (a sample spread over it) is refused
  defp audit do
    k = Vapor.Certificate.keygen()
    j = Vapor.Agent.Journal.new("q08") |> Vapor.Agent.Journal.append("input", %{"text" => "x"}) |> Vapor.Agent.Journal.append("answer", %{"text" => "y"})
    cert = Vapor.Certificate.sign(%Vapor.Certificate{payload: %{program: :crypto.hash(:sha256, "p")}}, k)
    items = [Vapor.Audit.certificate(cert, "certificate"), Vapor.Audit.journal(j, Vapor.Agent.Journal.attest(j, k), "run"),
             Vapor.Audit.document("doc.md", String.duplicate("evidence ", 50))]
    bin = items |> Vapor.Audit.build(date: "2026-10-03") |> Vapor.Audit.sign(k) |> Vapor.Audit.encode()
    ok = match?({:ok, _}, Vapor.Audit.verify(bin, trusted: [k.public]))
    positions = Enum.take_every(0..(byte_size(bin) - 1), max(div(byte_size(bin), 200), 1))

    accepted =
      Enum.count(positions, fn pos ->
        <<a::binary-size(pos), x, b::binary>> = bin
        match?({:ok, _}, Vapor.Audit.verify(a <> <<bxor(x, 0x01)>> <> b, trusted: [k.public]))
      end)

    [check("audit dossier: verifies offline (hashes, Merkle root, Ed25519, each item); single-bit alterations accepted (of #{length(positions)} spread over the file)",
           "#{ok} · #{accepted}", "the altered files themselves", "verifies, 0 accepted", ok and accepted == 0)]
  end

  # ------------------------------------------------------------- JBIG2 --

  # every fixture against jbig2dec's bitmap; the control misdeclares the
  # generic template of every generic region (a decoder that ignored the
  # template would still "match")
  defp jbig2(w) do
    dir = p("test/fixtures/docs/jbig2")

    if File.dir?(dir) do
      {:ok, man} = Vapor.JSON.decode(File.read!(Path.join(dir, "manifest.json")))
      sha = fn %{w: bw, h: bh} = bm -> :crypto.hash(:sha256, "P4\n#{bw} #{bh}\n" <> Vapor.Docs.JBIG2.packed(bm)) |> Base.encode16(case: :lower) end

      run = fn tamper ->
        Enum.count(man, fn {_, %{"files" => files, "sha256" => want}} ->
          bins = Enum.map(files, &tamper.(File.read!(Path.join(dir, &1))))
          r = case bins do
            [one] -> Vapor.Docs.JBIG2.decode(one)
            [g, pg] -> Vapor.Docs.JBIG2.decode(pg, g)
          end

          match?({:ok, _}, r) and sha.(elem(r, 1)) == want
        end)
      end

      good = run.(& &1)
      bad = run.(&misdeclare/1)
      generic = Enum.count(man, fn {n, _} -> String.starts_with?(n, "generic") or String.starts_with?(n, "enc_generic") or String.starts_with?(n, "enc_tpgd") end)

      [check("JBIG2: streams decoded to jbig2dec's bitmap (generic T0–T3, MMR, refinement, text in 8 corner modes, refinement/aggregate dictionaries, jbig2enc PDF mode)",
             "#{good}/#{map_size(man)}", "#{bad}/#{map_size(man)} (generic template misdeclared)",
             "all, control misses every generic stream", good == map_size(man) and bad <= map_size(man) - generic)] ++ jbig2_page(dir, w)
    else
      []
    end
  end

  # flip GBTEMPLATE (bits 1–2 of the generic region flags) in every
  # immediate generic region segment of a stream
  defp misdeclare(bin) do
    segs = Vapor.Docs.JBIG2.segments(bin)

    if segs == [] do
      bin
    else
      Enum.reduce(segs, bin, fn s, acc ->
        if s.type in [38, 39] and byte_size(s.data) > 17 and (:binary.at(s.data, 17) &&& 1) == 0 do
          case :binary.match(acc, s.data) do
            {at, _} ->
              <<pre::binary-size(at + 17), f, post::binary>> = acc
              pre <> <<bxor(f, 0b010)>> <> post
            :nomatch -> acc
          end
        else
          acc
        end
      end)
    end
  end

  # a scanned table in a JBIG2 PDF reads like the same page as PNG
  defp jbig2_page(_dir, nil), do: []

  defp jbig2_page(dir, w) do
    pdf = Path.join(dir, "enc_pdf.pdf")

    if File.exists?(pdf) do
      {:ok, %{1 => [img]}, _} = Vapor.Docs.PDF.images(File.read!(pdf), [1])
      {:ok, r} = OCR.read(img, worker: w)
      {:ok, png} = Vapor.Docs.Pictures.read(:png, File.read!(p(@priv <> "/tables/t03_inner.png")))
      {:ok, ref} = OCR.read(png.image, worker: w)
      dims = fn rr -> Enum.map(rr.tables, &{&1.rows, &1.cols}) end
      same = dims.(r) == dims.(ref) and dims.(r) != []
      cer = OCR.cer(r.text, ref.text)

      [check("JBIG2 → OCR: a scanned table in a JBIG2 PDF (globals + page), structure and text against the same page as PNG",
             "#{inspect(dims.(r))} · CER #{Float.round(cer * 100, 2)} %", "#{inspect(dims.(ref))} (the PNG)", "same table, CER < 1 %",
             same and cer < 0.01)]
    else
      []
    end
  end

  # ------------------------------------------------------------- tables --

  @doc """
  Every table of a fixture directory (`priv/quality/tables` by default)
  read by `OCR.read/2` and scored by `Table.score/2`, with the controls:
  `%{rows: [%{name, style, kind, dims, truth_dims, f1, exact, cer, free_cer, lines_f1, nospan_f1, tesseract_cer}], mean}`.
  Options: `names` (a subset), `worker`.
  """
  def evaluate_tables(dir \\ p(@priv <> "/tables"), opts \\ []) do
    w = Keyword.get(opts, :worker)
    {:ok, truth} = Vapor.JSON.decode(File.read!(Path.join(dir, "truth.json")))
    tess = case File.read(Path.join(dir, "tesseract.json")) do
      {:ok, b} -> elem(Vapor.JSON.decode(b), 1)["cells"]
      _ -> %{}
    end

    names = Keyword.get(opts, :names, truth |> Map.keys() |> Enum.sort())

    rows =
      for name <- names do
        t = truth[name]
        {:ok, pic} = Vapor.Docs.Pictures.read(:png, File.read!(Path.join(dir, name <> ".png")))
        tcells = for c <- t["table"]["cells"], do: %{row: c["row"], col: c["col"], rowspan: c["rowspan"], colspan: c["colspan"], text: c["text"], box: List.to_tuple(c["box"])}

        {:ok, r} = OCR.read(pic.image, worker: w)
        tb = List.first(r.tables)
        s = if tb, do: Table.score(tb.cells, tcells), else: %{f1: 0.0, exact: 0.0, cer: 1.0}
        free = if tb, do: tb.cells |> Enum.map(&%{&1 | text: Map.get(&1, :free, &1.text)}) |> Table.score(tcells), else: %{cer: 1.0}

        # control 1: the 0.7 reading — lines in reading order, as one column
        {:ok, flat} = OCR.read(pic.image, worker: w, tables: false)
        {bx0, by0, bx1, by1} = List.to_tuple(t["table"]["box"])

        lines =
          flat.lines
          |> Enum.filter(fn %{box: {x0, y0, x1, y1}} -> cx = (x0 + x1) / 2; cy = (y0 + y1) / 2; cx > bx0 and cx < bx1 and cy > by0 and cy < by1 end)
          |> Enum.with_index()
          |> Enum.map(fn {l, i} -> %{row: i, col: 0, rowspan: 1, colspan: 1, box: l.box, text: l.text} end)

        # control 2: the grid without merged cells (structure only)
        {nospan, _} = pic.image |> Vapor.Vision.Segment.gray() |> Vapor.Vision.Segment.ink() |> Vapor.Vision.Segment.components() |> Table.detect(spans: false)
        nospan_f1 = case nospan do
          [ns | _] -> Table.score(Enum.map(ns.cells, &Map.put(&1, :text, if(&1.comps == [], do: "", else: "x"))), tcells).f1
          _ -> 0.0
        end

        tcer =
          case tess[name] do
            nil -> nil
            cells -> Enum.zip(cells, tcells) |> Enum.map(fn {h, c} -> min(OCR.cer(h, c.text), 1.0) end) |> then(&(Enum.sum(&1) / length(&1)))
          end

        %{name: name, style: t["style"], font: t["font"], kind: tb && tb.kind, dims: tb && {tb.rows, tb.cols},
          truth_dims: {t["table"]["rows"], t["table"]["cols"]}, f1: s.f1, exact: s.exact, cer: s.cer, free_cer: free.cer,
          lines_f1: Table.score(lines, tcells).f1, nospan_f1: nospan_f1, tesseract_cer: tcer}
      end

    mean = fn k, rs -> rs = Enum.reject(rs, &is_nil(&1[k])); if rs == [], do: nil, else: Enum.sum(Enum.map(rs, & &1[k])) / length(rs) end
    grids = Enum.filter(rows, &(&1.style == "grid"))

    %{rows: rows,
      mean: %{f1: mean.(:f1, rows), exact: mean.(:exact, rows), cer: mean.(:cer, rows), free_cer: mean.(:free_cer, rows),
              lines_f1: mean.(:lines_f1, rows), nospan_f1: mean.(:nospan_f1, rows), tesseract_cer: mean.(:tesseract_cer, rows),
              grid_f1: mean.(:f1, grids), grid_nospan_f1: mean.(:nospan_f1, grids)}}
  end

  defp tables(nil), do: {nil, []}

  defp tables(w) do
    if File.dir?(p(@priv <> "/tables")) do
      e = evaluate_tables(p(@priv <> "/tables"), worker: w)
      m = e.mean
      pct = fn x -> "#{Float.round(x * 100, 1)} %" end

      {e,
       [check("tables: structure, ICDAR-2013 adjacency F1 of 12 scanned tables (grid, inner, booktabs, rules under rows)",
              Float.round(m.f1, 3), "#{Float.round(m.lines_f1, 3)} (the 0.7 reading: lines in order, one column)", "≥ 0.95, control ≤ 0.6",
              m.f1 >= 0.95 and m.lines_f1 <= 0.6),
        check("tables: merged cells, adjacency F1 on the grids (header cells spanning rows and columns)",
              Float.round(m.grid_f1, 3), "#{Float.round(m.grid_nospan_f1, 3)} (span detection off)", "≥ 0.95, control < value",
              m.grid_f1 >= 0.95 and m.grid_nospan_f1 < m.grid_f1),
        check("tables: mean cell CER, column types and shapes (Tesseract on perfectly cropped cells: #{if m.tesseract_cer, do: pct.(m.tesseract_cer), else: "—"})",
              pct.(m.cer), "#{pct.(m.free_cer)} (free reading of the same cells)", "< 8 %, control > value",
              m.cer < 0.08 and m.free_cer > m.cer)]}
    else
      {nil, []}
    end
  end

  # ------------------------------------------------------------- Mamba-2 --

  # transformers' own logits, shipped (priv/quality/mamba2, written by
  # test/python/hf_mamba2.py): each gated-norm setting must match its
  # reference and miss the other's — the comparison discriminates
  defp mamba2(w) do
    dir = p("priv/quality/mamba2/mamba2-g2")

    if File.dir?(dir) do
      {:ok, ref} = Vapor.Ingest.Safetensors.read(Path.join(dir, "reference.safetensors"))
      prompt = Vapor.Tensor.to_list(ref["prompt"])
      rows = fn m, worker ->
        {:ok, g} = Vapor.Recurrent.open(m.spec, m.weights, worker: worker)
        {_, _, rows} = Vapor.Recurrent.prefill(g, prompt)
        Vapor.Recurrent.close(g)
        rows
      end
      {:ok, grouped} = Vapor.Lock.open(dir)
      {:ok, whole} = Vapor.Lock.open(dir, gated_norm: :whole)
      want = fn k -> ref[k] |> Vapor.Tensor.to_floats() |> Enum.chunk_every(grouped.spec.vocab) end
      og = rows.(grouped, nil)
      ow = rows.(whole, nil)
      fl = &Enum.map(&1, fn r -> Vapor.Sampler.floats(r) end)
      {eg, ew} = {rel(fl.(og), want.("logits_grouped")), rel(fl.(ow), want.("logits"))}
      {cg, cw} = {rel(fl.(og), want.("logits")), rel(fl.(ow), want.("logits_grouped"))}
      native = if w, do: rows.(grouped, w) == og and rows.(whole, w) == ow, else: true

      [check("Mamba-2 (2 groups, Δ clamped) = transformers' chunked scan: per-group norm vs mamba_ssm's formula, whole-width vs transformers (max relative logit error); native bits = oracle",
             "#{sci(eg)} / #{sci(ew)}#{if w, do: ", native #{native}", else: ""}", "#{sci(cg)} / #{sci(cw)} (each against the other's reference)",
             "< 1e-5, control > 0.05", eg < 1.0e-5 and ew < 1.0e-5 and cg > 0.05 and cw > 0.05 and native)]
    else
      []
    end
  end

  defp rel(rows, want) do
    Enum.zip(rows, want)
    |> Enum.map(fn {a, b} -> (Enum.zip_with(a, b, &abs(&1 - &2)) |> Enum.max()) / (b |> Enum.map(&abs/1) |> Enum.max()) end)
    |> Enum.max()
  end

  defp sci(x), do: :io_lib.format("~.2e", [x]) |> to_string()

  # ------------------------------------------------------------------ GPU --

  # a decode loop in a resident GPU session = the CPU session, bit for bit;
  # the control: the GPU loop with its KV state reset before every token
  defp gpu(nil), do: []

  defp gpu(w) do
    alias Vapor.Runtime.{Fabric, Session, Substrates}
    bin = Substrates.binary("vapor-fabric", "native")

    with true <- bin != nil,
         {:ok, f} <- Fabric.start_link(exec: [bin]),
         %{ready: true} = dev <- Fabric.info(f) do
      {c, ws} = Vapor.Bench.Round08.tiny("llama", %{"vocab_size" => 256, "hidden_size" => 64, "intermediate_size" => 128, "num_hidden_layers" => 2,
                                                     "num_attention_heads" => 4, "num_key_value_heads" => 2})
      {:ok, prog} = Vapor.Model.Decoder.program(c, ws, max_seq: 64)
      {:ok, comp} = Vapor.Compile.Lower.lower(prog)
      ids = &Vapor.Tensor.from_list(:s32, [length(&1)], &1)
      loop = fn server, opts, reset ->
        {:ok, s} = Session.open(server, comp, opts)
        rows = for i <- 0..11 do
          s2 = if reset and i > 0, do: (Session.close(s); elem(Session.open(server, comp, opts), 1)), else: s
          {:ok, %{logits: l}, m} = Session.step(s2, %{tok: ids.([rem(i * 13 + 5, 256)]), pos: ids.([i])}, [:logits])
          {l.data, m.counters[:gpu_recording_reused]}
        end
        rows
      end
      cpu = loop.(w, [isa: Substrates.host_isa()], false) |> Enum.map(&elem(&1, 0))
      g = loop.(f, [isa: :spirv], false)
      reset = loop.(f, [isa: :spirv], true) |> Enum.map(&elem(&1, 0))
      GenServer.stop(f)
      same = Enum.map(g, &elem(&1, 0)) == cpu
      reused = Enum.count(g, &(elem(&1, 1) == 1))

      [check("GPU resident session (#{dev[:name]}): 12 decode steps = the CPU session, bit for bit; recorded command buffers replayed",
             "#{same}, #{reused}/12 replayed", "#{reset == cpu} (KV state reset each token)", "identical, control differs", same and reset != cpu)]
    else
      _ -> []
    end
  end

  # ------------------------------------------------------- sparse 4-bit --

  # a 4-bit MoE: predicated (row-masked) GEMV = dense, bit for bit; the
  # control changes a selected expert's weights, which the comparison sees
  defp sb4(nil), do: []

  defp sb4(w) do
    alias Vapor.Runtime.{Native, Substrates}
    {c, ws} = Vapor.Bench.Round08.tiny("mixtral", %{"vocab_size" => 256, "hidden_size" => 256, "intermediate_size" => 256, "num_hidden_layers" => 1,
                                                     "num_attention_heads" => 4, "num_key_value_heads" => 2, "num_local_experts" => 4,
                                                     "num_experts_per_tok" => 2})
    toks = [3, 17, 99, 200]
    env = Map.merge(Vapor.Model.Decoder.empty_caches(c, 16), %{tok: Vapor.Tensor.from_list(:s32, [4], toks), pos: Vapor.Tensor.from_list(:s32, [4], [0, 1, 2, 3])})
    run = fn ws, mode ->
      {:ok, p} = Vapor.Model.Decoder.program(c, ws, max_seq: 16, moe: mode, quantize: :sb4)
      {:ok, comp} = Vapor.Compile.Lower.lower(p)
      {:ok, r} = Native.run(w, comp, env, isa: Substrates.host_isa(), mode: :native)
      r.outputs.logits.data
    end
    dense = run.(ws, :dense)
    sparse = run.(ws, :sparse)
    # every expert's down projection scaled: whichever the router selects, the output must change
    bumped = Map.new(ws, fn {k, t} ->
      if is_binary(k) and k =~ ~r/experts\.\d+\.w2/, do: {k, Vapor.Tensor.from_list(:f32, t.shape, Enum.map(Vapor.Tensor.to_floats(t), &(&1 * 1.5)))}, else: {k, t}
    end)
    control = run.(bumped, :sparse)

    [check("sparse 4-bit experts: predicated qgemv_masked = dense sb4 MoE, bit for bit (4 tokens, top-2 of 4)",
           "#{sparse == dense}", "#{control == dense} (experts' down projections × 1.5)", "identical, control differs", sparse == dense and control != dense)]
  end

end
