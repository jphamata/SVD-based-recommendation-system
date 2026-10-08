defmodule Vapor.Console do
  @moduledoc """
  The operator console: a web page served by `Vapor.Serve` at `/` (and
  `/ui`), and the JSON endpoints it uses — usable from any client.

  | endpoint | what it does |
  |---|---|
  | `GET /v1/vapor/info` | the served model's contract (`Vapor.Lock.Spec`: family, lineage, interface, widths, features), the registered adapters, the library's status |
  | `GET /v1/vapor/library` | files (path, kind, size, SHA-256), warnings, counts, root |
  | `POST /v1/vapor/library` | `{"name", "data"}` (base64): ingest through the document airlock (`Vapor.Docs`) |
  | `POST /v1/vapor/search` | `{"query", "k"}`: passages with provenance, inclusion proofs, the library root and a receipt — the receipt anchored in the transparency log (`tlog`) |
  | `POST /v1/vapor/verify` | a search result's `{query, k, method, library_receipt}`: recomputed against the library |
  | `POST /v1/vapor/citations` | `{"answer", "query", "k"}`: every `<quote src="i">…</quote>` checked verbatim against the sources |
  | `POST /v1/vapor/search_image` | `{"name", "data"}`: visually similar pictures (with thumbnails) |
  | `POST /v1/vapor/quality` | `{"text"}`: the calibrated text gate's verdict (`Vapor.Quality.Text`) and the thresholds |
  | `GET /v1/vapor/thumb?doc=…` | a PNG thumbnail of an indexed picture |
  | `POST /v1/vapor/ocr` | `{"name", "data", "script"}` (PNG, JPEG, PPM or PDF): the text of the picture or of the PDF's scanned pages — lines with boxes, confidences and each character's confidence (`Vapor.Vision.OCR`); figures with captions and charts' data; `script` `latin` (default), `arabic`, `cyrillic`, `cursive` (refused: no handwriting reader is shipped), `zh`, `ja`, `ko` or `math` (a formula → LaTeX) |
  | `GET /v1/vapor/tlog` | the transparency log: origin, size, root, the signed checkpoint, the verifier key, the latest entries |
  | `POST /v1/vapor/tlog` | `{"text"}`: anchor a note; its receipt (index, inclusion proof, checkpoint) |
  | `GET /v1/vapor/tlog/proof?index=i` | entry `i` with its receipt against the current tree |
  | `GET /v1/vapor/tlog/consistency?from=m` | the RFC 9162 proof that the current tree extends the one of `m` entries |
  | `GET /v1/vapor/substrates` | every substrate present, admitted by measurement: verdict (canonical, envelope, refused), numerical fingerprint, the probes (`Vapor.Substrate`) |

  The page is one self-contained HTML file (`priv/console/index.html`): no
  CDN, no fonts to fetch, works offline.
  """
  alias Vapor.Docs.{Library, Pictures}
  alias Vapor.{JSON, Lock, Tensor}
  alias Vapor.Quality.Text

  defmodule Holder do
    @moduledoc "The console's library, held by a process (one per server)."
    use Agent
    def start_link(opts \\ []), do: Agent.start_link(fn -> Library.new(opts) end)
    def get(h), do: Agent.get(h, & &1, 60_000)
    def update(h, f), do: Agent.get_and_update(h, f, 600_000)
  end

  @doc "Handle a console request, or `:pass`."
  def handle(sock, %{method: :GET, path: p}, _ctx) when p in ["/", "/ui", "/ui/"], do: page(sock)
  def handle(sock, %{method: :GET, path: "/favicon.ico"}, _ctx), do: asset(sock, "favicon.ico", "image/x-icon")
  def handle(sock, %{method: :GET, path: "/logo.svg"}, _ctx), do: asset(sock, "logo.svg", "image/svg+xml")
  def handle(sock, %{method: :GET, path: "/manifest.webmanifest"}, _ctx), do: asset(sock, "manifest.webmanifest", "application/manifest+json")
  def handle(sock, %{method: :GET, path: "/icon-" <> n}, _ctx) when n in ["192.png", "512.png"], do: asset(sock, "icon-" <> n, "image/png")
  def handle(sock, %{method: :GET, path: "/" <> js}, _ctx) when js in ["workbench.js", "gpu_tracer.js", "market.js", "athanor.js", "opus.js"], do: asset(sock, js, "text/javascript; charset=utf-8")

  def handle(sock, %{method: :GET, path: "/v1/vapor/info"}, ctx) do
    {spec, context} =
      case ctx.engine && Vapor.Engine.info(ctx.engine) do
        %{config: c} = i -> {c |> Lock.spec() |> spec_json(), i[:max_seq]}
        _ -> {nil, nil}
      end

    substrate = case ctx.engine && Vapor.Engine.info(ctx.engine) do
      %{substrate: sub} -> Map.take(sub, [:kind, :device, :staged, :device_local, :resident_bytes])
      _ -> nil
    end

    json(sock, 200, %{model: ctx.model, spec: spec, context: context, substrate: substrate, adapters: Enum.map(Lock.adapters(), &Lock.id/1), library: Library.stats(lib(ctx)),
                      version: to_string(Application.spec(:vapor, :vsn) || "")})
  end

  def handle(sock, %{method: :GET, path: "/v1/vapor/library"}, ctx) do
    l = lib(ctx)
    json(sock, 200, %{stats: Library.stats(l), files: l.files, warnings: l.warnings})
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/library", body: body}, ctx) do
    with {:ok, %{"name" => name, "data" => b64}} when is_binary(name) and is_binary(b64) <- JSON.decode(body),
         {:ok, bytes} <- Base.decode64(b64, ignore: :whitespace) |> ok_or("data: base64") do
      result = Holder.update(ctx.library, fn l ->
        case Library.add(l, {Path.basename(name), bytes}) do
          {:ok, l2, rep} -> {{:ok, rep}, l2}
          err -> {err, l}
        end
      end)

      case result do
        {:ok, rep} -> json(sock, 200, %{report: Map.drop(rep, [:file]) |> Map.put(:file, rep[:file]), stats: Library.stats(lib(ctx))})
        {:error, %Vapor.Rejection{} = r} -> json(sock, 422, %{error: %{message: "#{r.bound}", node: inspect(r.node), repair: r.repair}})
      end
    else
      _ -> bad(sock, "body: {\"name\": …, \"data\": base64}")
    end
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/search", body: body}, ctx) do
    with {:ok, %{"query" => q} = req} when is_binary(q) <- JSON.decode(body) do
      r = Library.search(lib(ctx), q, k: clamp(req["k"], 1, 20, 5))
      out = search_json(r)
      # the search's receipt is anchored in the server's transparency log
      json(sock, 200, Map.put(out, :tlog, anchor(ctx, search_entry(out))))
    else
      _ -> bad(sock, "body: {\"query\": text, \"k\": n}")
    end
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/verify", body: body}, ctx) do
    with {:ok, %{"query" => q, "k" => k, "library_receipt" => rec} = req} <- JSON.decode(body) do
      l = lib(ctx)
      again = Library.search(l, q, k: k, method: method(req["method"]))
      same = again.library_receipt == rec
      json(sock, 200, %{verified: same, library_root: again.library_root, recomputed_receipt: again.library_receipt,
                        reason: if(same, do: nil, else: "the library or the ranking changed since that search")})
    else
      _ -> bad(sock, "body: a search result")
    end
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/citations", body: body}, ctx) do
    with {:ok, %{"answer" => a, "query" => q} = req} when is_binary(a) and is_binary(q) <- JSON.decode(body) do
      r = Library.search(lib(ctx), q, k: clamp(req["k"], 1, 20, 5))
      checks = Vapor.RAG.check_citations(a, r.hits, r.root)
      json(sock, 200, %{citations: checks, all_verbatim: Enum.all?(checks, &(&1.verbatim and &1.member)), library_root: r.library_root})
    else
      _ -> bad(sock, "body: {\"answer\", \"query\"}")
    end
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/search_image", body: body}, ctx) do
    with {:ok, %{"name" => name, "data" => b64}} <- JSON.decode(body),
         {:ok, bytes} <- Base.decode64(b64, ignore: :whitespace) |> ok_or("data: base64"),
         kind = Vapor.Docs.sniff(name, bytes),
         {:ok, %{image: img}} <- Pictures.read(kind, bytes) |> ok_or("a PNG, JPEG or PPM picture") do
      hits = Library.search_image(lib(ctx), img, k: 6)
      json(sock, 200, %{hits: hits})
    else
      {:ok, _} -> bad(sock, "a decodable picture (PNG, JPEG or PPM)")
      {:error, why} when is_binary(why) -> bad(sock, why)
      _ -> bad(sock, "body: {\"name\", \"data\": base64}")
    end
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/quality", body: body}, _ctx) do
    with {:ok, %{"text" => t}} when is_binary(t) <- JSON.decode(body) do
      {p, g} = gate()
      j = Text.judge(t, p, g)
      json(sock, 200, %{verdict: j.verdict, noise: j[:noise], collapse: j[:collapse], metrics: j.metrics,
                        gate: %{len: g.len, noise: thresholds(g.noise), collapse: thresholds(g.collapse)},
                        reference: "vapor's Portuguese docs (priv/quality); a gate judges language statistics, not meaning"})
    else
      _ -> bad(sock, "body: {\"text\": …}")
    end
  end

  def handle(sock, %{method: :GET, path: "/v1/vapor/thumb?" <> qs}, ctx) do
    doc = URI.decode_query(qs)["doc"]

    case Enum.find(lib(ctx).images, &(&1.doc == doc)) do
      nil -> json(sock, 404, %{error: %{message: "no such picture"}})
      %{image: img} -> raw(sock, 200, "image/png", Vapor.Modal.Image.png(thumb(img), 1))
    end
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/ocr", body: body}, _ctx) do
    with {:ok, %{"name" => name, "data" => b64}} <- JSON.decode(body),
         {:ok, bytes} <- Base.decode64(b64, ignore: :whitespace) |> ok_or("data: base64"),
         {:ok, pages} <- ocr_inputs(name, bytes) do
      t0 = System.monotonic_time(:millisecond)
      pdf? = Vapor.Docs.sniff(name, bytes) == :pdf

      script = Map.get(JSON.decode(body) |> elem(1), "script", "latin")

      out =
        for {label, img} <- pages do
          case read_script(script, img) do
            {:ok, r} ->
              {label, page} = case label do {:page, i} -> {"page #{i}", i}; l -> {l, nil} end
              %{label: label, page: page, width: img.w, height: img.h, text: r.text, confidence: r.confidence,
                blocks: Enum.map(r.blocks, &Tuple.to_list/1), image: if(pdf?, do: page_image(img)),
                tables: Enum.map(r[:tables] || [], &table_json/1), block_kinds: r[:block_kinds] || [],
                figures: Enum.map(r[:figures] || [], &figure_json/1), script: script, latex: r[:latex], direction: r[:direction],
                lines: Enum.map(r.lines, fn l -> %{box: Tuple.to_list(l.box), block: Map.get(l, :block), text: l.text, greedy: Map.get(l, :greedy, l.text),
                                                   confidence: Map.get(l, :confidence, 1.0), visual: Map.get(l, :visual),
                                                   chars: Enum.map(Map.get(l, :chars, []), &%{c: &1.char, p: Float.round((&1[:p] || &1[:score] || 1.0) * 1.0, 3)})} end)}

            {:error, %Vapor.Rejection{} = r} ->
              %{label: (case label do {:page, i} -> "page #{i}"; l -> l end), error: r.bound, repair: r.repair}

            {:error, why} ->
              %{label: (case label do {:page, i} -> "page #{i}"; l -> l end), error: "nothing read (#{inspect(why, limit: 4)})"}
          end
        end

      # which reader and decoder this script used, said as they are
      {reader, lm} = reader_of(script)

      json(sock, 200, %{pages: out, ms: System.monotonic_time(:millisecond) - t0, model: reader,
                        decoder: lm && %{order: lm.lm.order, weight: lm.weight, beam: lm.beam, gate: lm.gate}})
    else
      {:error, why} when is_binary(why) -> bad(sock, why)
      _ -> bad(sock, "body: {\"name\", \"data\": base64}")
    end
  end

  # ------------------------------------------------------ transparency log --

  def handle(sock, %{method: :GET, path: "/v1/vapor/tlog"}, %{tlog: h}) when h != nil do
    %{log: log, signer: k, verifier: v} = Vapor.Tlog.Holder.get(h)
    entries = for i <- max(0, log.size - 50)..(log.size - 1)//1, do: entry_json(log, i)

    json(sock, 200, %{origin: log.origin, size: log.size, root: Base.encode16(Vapor.Tlog.root(log), case: :lower),
                      checkpoint: Vapor.Tlog.checkpoint(log, k), verifier: v, entries: Enum.reverse(entries)})
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/tlog", body: body}, %{tlog: h} = ctx) when h != nil do
    case JSON.decode(body) do
      {:ok, %{"text" => text}} when is_binary(text) and byte_size(text) in 1..65_536 ->
        json(sock, 200, %{receipt: anchor(ctx, "vapor-note/1\n" <> text)})

      _ ->
        bad(sock, "body: {\"text\": up to 64 KiB}")
    end
  end

  def handle(sock, %{method: :GET, path: "/v1/vapor/tlog/proof?" <> qs}, %{tlog: h}) when h != nil do
    %{log: log, signer: k} = Vapor.Tlog.Holder.get(h)

    case Integer.parse(URI.decode_query(qs)["index"] || "") do
      {i, ""} when i >= 0 and i < log.size ->
        json(sock, 200, Map.merge(entry_json(log, i), %{receipt: log |> Vapor.Tlog.receipt(i, k) |> Map.update!("proof", &hexes/1)}))

      _ ->
        bad(sock, "index: an entry of the log")
    end
  end

  # the proof that today's log extends the tree of `from` entries a reader saw
  def handle(sock, %{method: :GET, path: "/v1/vapor/tlog/consistency?" <> qs}, %{tlog: h}) when h != nil do
    %{log: log, signer: k} = Vapor.Tlog.Holder.get(h)

    case Integer.parse(URI.decode_query(qs)["from"] || "") do
      {m, ""} when m >= 1 and m <= log.size ->
        json(sock, 200, %{from: m, size: log.size, checkpoint: Vapor.Tlog.checkpoint(log, k),
                          proof: Enum.map(Vapor.Tlog.consistency(log, m), &Base.encode16(&1, case: :lower))})

      _ ->
        bad(sock, "from: a size between 1 and the log's size")
    end
  end

  # audit dossiers: verify one (a .vdossier or a PDF report carrying it), or build the demonstration one
  def handle(sock, %{method: :POST, path: "/v1/vapor/audit", body: body}, _ctx) do
    with {:ok, %{"data" => b64} = req} <- JSON.decode(body),
         {:ok, bytes} <- Base.decode64(b64, ignore: :whitespace) |> ok_or("data: base64") do
      json(sock, 200, audit_json(bytes, req))
    else
      {:error, why} when is_binary(why) -> bad(sock, why)
      _ -> bad(sock, "body: {\"data\": base64 of a .vdossier or its PDF report, \"log_key\"?: verifier key}")
    end
  end

  # ------------------------------------------------------------- studio --

  def handle(sock, %{method: :GET, path: "/v1/vapor/studio/nodes"}, _ctx),
    do: json(sock, 200, %{nodes: Vapor.Studio.catalogue(), templates: Vapor.Studio.Templates.all()})

  def handle(sock, %{method: :POST, path: "/v1/vapor/studio/run", body: body}, ctx) do
    with {:ok, %{"graph" => g}} <- JSON.decode(body) do
      case Vapor.Studio.run(g, worker: studio_worker(), dir: studio_dir(ctx), cache: studio_cache()) do
        {:ok, r} ->
          nodes = Map.new(r.order, fn id ->
            rc = r.receipts[id]
            outs = Map.new(r.outputs[id] || %{}, fn {port, v} -> (fn d -> {port, preview_memo(d, v) |> Map.put(:digest, d)} end).(rc.outputs[port].digest) end)
            {id, %{outputs: outs, cached: id in r.cached, ms: Map.get(r.ms, id, 0)}}
          end)

          json(sock, 200, %{root: r.root, executed: r.executed, cached: r.cached, order: r.order, nodes: nodes})

        {:error, why} ->
          json(sock, 422, %{error: rejection_json(why)})
      end
    else
      _ -> bad(sock, "body: {\"graph\": {\"nodes\": {…}}}")
    end
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/studio/verify", body: body}, ctx) do
    with {:ok, %{"graph" => g, "root" => root}} <- JSON.decode(body) do
      case Vapor.Studio.verify(g, root, worker: studio_worker(), dir: studio_dir(ctx)) do
        :ok -> json(sock, 200, %{verified: true, root: root})
        {:error, {:root, _, got}} -> json(sock, 200, %{verified: false, root: root, got: got})
        {:error, why} -> json(sock, 422, %{error: rejection_json(why)})
      end
    else
      _ -> bad(sock, "body: {\"graph\", \"root\"}")
    end
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/studio/comfy", body: body}, _ctx) do
    with {:ok, %{"workflow" => wf}} <- JSON.decode(body) do
      case Vapor.Studio.Comfy.import(wf) do
        {:ok, g, notes} -> json(sock, 200, %{graph: g, notes: notes})
        {:error, why} -> json(sock, 422, %{error: rejection_json(why)})
      end
    else
      _ -> bad(sock, "body: {\"workflow\": ComfyUI API-format prompt}")
    end
  end

  # ---------------------------------------------- the 0.10 laboratories --

  def handle(sock, %{method: :GET, path: "/v1/vapor/substrates"}, _ctx), do: json(sock, 200, %{substrates: Vapor.Console.Lab.substrates()})

  # ---- the 0.11 laboratories (Vapor.Console.Lab11)

  def handle(sock, %{method: :POST, path: "/v1/vapor/" <> route, body: body}, _ctx)
      when route in ~w(sketch archive archive/check) do
    alias Vapor.Console.Lab11, as: L

    case JSON.decode(body) do
      {:ok, req} when is_map(req) ->
        result =
          case route do
            "sketch" -> L.sketch(req["name"] || "sketch.png", req["data"] || "", req["mode"] || "vector", req)
            "archive" -> L.archive(req["kind"], req["recipe"], req["result"])
            "archive/check" -> L.archive_check(req["data"] || "")
          end

        case result do
          {:ok, %{error: msg}} -> bad(sock, msg)
          {:ok, v} -> json(sock, 200, v)
          {:error, why} when is_binary(why) -> bad(sock, why)
          {:error, why} -> bad(sock, inspect(why))
        end

      _ ->
        bad(sock, "body: a JSON object")
    end
  end

  # ---- the 0.12 laboratories (Vapor.Console.Lab12)

  def handle(sock, %{method: :POST, path: "/v1/vapor/" <> route, body: body}, _ctx)
      when route in ~w(solve engineering logic chess shogi go mnk poker protein render) do
    alias Vapor.Console.Lab12, as: L

    case JSON.decode(body) do
      {:ok, req} when is_map(req) ->
        result =
          case route do
            "solve" -> L.solve(req)
            "engineering" -> L.engineering(req)
            "logic" -> L.logic(req)
            "chess" -> L.chess(req)
            "shogi" -> L.shogi(req)
            "go" -> L.go(req)
            "mnk" -> L.mnk(req)
            "poker" -> L.poker(req)
            "protein" -> L.protein(req)
            "render" -> L.render(req)
          end

        case result do
          {:ok, v} -> json(sock, 200, v)
          {:error, why} when is_binary(why) -> bad(sock, why)
          {:error, why} -> bad(sock, inspect(why))
          other -> bad(sock, inspect(other))
        end

      _ ->
        bad(sock, "body: a JSON object")
    end
  end

  # ---- the 0.13 desks (Vapor.Console.Lab13): finance and the trading desk

  def handle(sock, %{method: :POST, path: "/v1/vapor/finance", body: body}, _ctx) do
    case JSON.decode(body) do
      {:ok, req} when is_map(req) ->
        case Vapor.Console.Lab13.finance(req) do
          {:ok, v} -> json(sock, 200, v)
          {:error, why} -> bad(sock, why)
        end
      _ -> bad(sock, "body: a JSON object")
    end
  end

  # ---- the 0.14 open workspace (Vapor.Console.Lab14): Alembic, the Athanor, games, the Crucible, the Assay
  # the reference tracer's self-check (white and gradient furnaces, the biased estimator as control)
  def handle(sock, %{method: :GET, path: "/v1/vapor/render/furnace"}, _ctx), do: json(sock, 200, Vapor.Console.Lab12.furnace())

  def handle(sock, %{method: :GET, path: "/v1/vapor/workspace"}, ctx), do: json(sock, 200, Vapor.Main.jsonable(Vapor.Console.Lab14.info(ctx)))

  def handle(sock, %{method: :GET, path: "/v1/vapor/athanor/" <> rest}, _ctx) do
    {id, since} = case String.split(rest, "?since=", parts: 2) do [i, n] -> {i, (case Integer.parse(n) do {k, _} -> k; :error -> 0 end)}; [i] -> {i, 0} end
    respond14(sock, Vapor.Console.Lab14.athanor_action(id, %{"action" => "snapshot", "since" => since}))
  end

  def handle(sock, %{method: :POST, path: "/v1/vapor/" <> route, body: body}, ctx)
      when route in ["alembic", "athanor", "athanor/verify", "game", "crucible", "assay", "formalize", "detect"] or
             (byte_size(route) > 8 and binary_part(route, 0, 8) == "athanor/") do
    alias Vapor.Console.Lab14, as: L
    case JSON.decode(body) do
      {:ok, req} when is_map(req) ->
        result =
          case route do
            "alembic" -> L.alembic(req)
            "athanor" -> with({:ok, id} <- L.athanor_start(req, ctx), do: L.athanor_action(id, %{"action" => "snapshot"}))
            "athanor/verify" -> L.verify(req)
            "game" -> L.game(req)
            "crucible" -> L.crucible(req)
            "assay" -> L.assay(req)
            "formalize" -> L.formalize(req, ctx)
            "detect" -> {:ok, L.detect(to_string(req["text"] || ""))}
            "athanor/" <> id -> L.athanor_action(id, req)
          end
        respond14(sock, result)
      _ -> bad(sock, "body: a JSON object")
    end
  end

  # ---- the 0.15 Opus desks (Vapor.Console.Lab15): Rebis, Aludel, Tabula, Cupel, Amalgam
  def handle(sock, %{method: :GET, path: "/v1/vapor/opus"}, _ctx), do: json(sock, 200, Vapor.Main.jsonable(Vapor.Console.Lab15.info()))

  def handle(sock, %{method: :POST, path: "/v1/vapor/" <> route, body: body}, _ctx) when route in ~w(rebis aludel tabula cupel amalgam) do
    alias Vapor.Console.Lab15, as: L

    case JSON.decode(body) do
      {:ok, req} when is_map(req) ->
        result =
          case route do
            "rebis" -> L.rebis(req)
            "aludel" -> L.aludel(req)
            "tabula" -> L.tabula(req)
            "cupel" -> L.cupel(req)
            "amalgam" -> L.amalgam(req)
          end

        respond14(sock, result)

      _ ->
        bad(sock, "body: a JSON object")
    end
  end

  def handle(sock, req, ctx), do: Vapor.Console.Hall.handle(sock, req, ctx)

  defp respond14(sock, {:ok, r}), do: json(sock, 200, Vapor.Main.jsonable(r))
  defp respond14(sock, {:error, %{} = e}), do: json(sock, 422, %{error: Vapor.Main.jsonable(e)})
  defp respond14(sock, {:error, e}), do: json(sock, 422, %{error: %{message: to_string(e)}})

  defp studio_worker, do: if(Vapor.Runtime.Substrates.binary("vapor-worker", "native"), do: Vapor.Vision.OCR.worker())
  defp studio_dir(ctx), do: Map.get(ctx, :studio_dir) || System.get_env("VAPOR_STUDIO_DIR") || File.cwd!()

  # one cache for the server's lifetime: an edited graph re-runs only what changed
  defp studio_cache do
    case Process.whereis(Vapor.Console.StudioCache) do
      nil ->
        case Vapor.Studio.Cache.start_link(name: Vapor.Console.StudioCache, max: 256) do
          {:ok, pid} -> Process.unlink(pid); pid
          {:error, {:already_started, pid}} -> pid
        end

      pid ->
        pid
    end
  end

  # one reader per script: Latin (printed), Arabic, Cyrillic, cursive Latin (refused), Chinese, Japanese, Korean, a formula
  defp read_script(script, img) do
    case script do
      s when s in ["arabic", "cyrillic", "cursive"] -> Vapor.Vision.OCR.read(img, model: String.to_atom(s), digitize: false)
      s when s in ["zh", "ja", "ko"] ->
        with {:ok, pack} <- Vapor.Vision.CJK.default(String.to_atom(s)),
             {:ok, r} <- Vapor.Vision.CJK.read(pack, img, worker: Vapor.Vision.OCR.worker()) do
          {:ok, Map.merge(r, %{confidence: 1.0, blocks: [], tables: [], block_kinds: []})}
        end

      "math" ->
        with {:ok, r} <- Vapor.Vision.Math.read(img) do
          {:ok, %{text: r.latex, latex: r.latex, confidence: r.confidence, lines: [], blocks: [], tables: [], block_kinds: []}}
        end

      _ -> Vapor.Vision.OCR.read(img, digitize: true)
    end
  end

  defp reader_of(script) do
    case script do
      s when s in ["arabic", "cyrillic"] ->
        lm = with {:ok, m} <- Vapor.Vision.OCR.default(String.to_atom(s)), do: Vapor.Vision.OCR.language_model(m, []), else: (_ -> nil)
        {"priv/ocr-#{s} (vapor_encoder, head: rows, CTC)", lm}

      "cursive" -> {"none (refused)", nil}
      s when s in ["zh", "ja", "ko"] -> {"priv/ocr-cjk-#{s} (directional element features, nearest class; character language model)", nil}
      "math" -> {"priv/math (symbol templates, structure by geometry)", nil}

      _ ->
        lm = with {:ok, m} <- Vapor.Vision.OCR.default(), do: Vapor.Vision.OCR.language_model(m, []), else: (_ -> nil)
        {"priv/ocr (vapor_encoder, head: rows, CTC)", lm}
    end
  end

  defp figure_json(f) do
    %{box: Tuple.to_list(f.box), kind: f.kind, caption: f[:caption] && %{text: f.caption.text, box: Tuple.to_list(f.caption.box)},
      refused: f[:refused] && inspect(f.refused),
      data: f[:data] && %{x: Map.take(f.data.x, [:scale]), y: Map.take(f.data.y, [:scale]),
                          series: Enum.map(f.data.series, fn s -> %{color: s.color, kind: s.kind, points: Enum.map(s.points, fn {x, y} -> [x, y] end),
                                                                   labels: for(b <- Map.get(s, :bars, []), b.label != nil, do: [b.x, b.label])} end)}}
  end

  defp rejection_json(%Vapor.Rejection{} = r), do: %{node: inspect(r.node), expected: r.bound, repair: r.repair}
  defp rejection_json(other), do: %{node: nil, expected: inspect(other), repair: nil}

  # previews are a function of the value, so of its digest: made once
  defp preview_memo(digest, v) do
    t = case :ets.whereis(:vapor_console_previews) do
      :undefined ->
        # owned by the studio cache's process, which lives as long as the server
        Agent.get(studio_cache(), fn st ->
          if :ets.whereis(:vapor_console_previews) == :undefined, do: :ets.new(:vapor_console_previews, [:named_table, :public, :set])
          st
        end)

        :vapor_console_previews

      ref -> ref
    end

    case :ets.lookup(t, digest) do
      [{_, p}] -> p
      [] ->
        p = preview(v)
        if :ets.info(t, :size) > 512, do: :ets.delete_all_objects(t)
        :ets.insert(t, {digest, p})
        p
    end
  end

  # what the canvas shows of a value: a picture, a clip, a sound, a mesh's render, or the value
  @preview_px 320
  defp preview(%Vapor.Modal.Image{} = img), do: %{kind: "image", info: Vapor.Studio.Value.describe(img), src: page_image(fit(img, @preview_px))}

  defp preview(%Vapor.Studio.Video{frames: fr} = v) do
    small = %{v | frames: fr |> Enum.take(240) |> Enum.map(&fit(&1, 240))}
    %{kind: "video", info: Vapor.Studio.Value.describe(v), src: "data:image/gif;base64," <> Base.encode64(Vapor.Media.GIF.encode(small.frames, fps: v.fps))}
  end

  defp preview(%Vapor.Modal.Audio{} = a) do
    %{kind: "audio", info: Vapor.Studio.Value.describe(a), src: "data:audio/wav;base64," <> Base.encode64(Vapor.Modal.Audio.encode(a))}
  end

  defp preview(%{__struct__: Vapor.Geom.Mesh} = m) do
    %{kind: "mesh", info: Vapor.Studio.Value.describe(m), src: page_image(Vapor.Geom.render(m, width: 240, height: 240))}
  end

  defp preview(%Tensor{} = t), do: %{kind: "tensor", info: Vapor.Studio.Value.describe(t)}
  defp preview(v) when is_binary(v), do: %{kind: "text", info: Vapor.Studio.Value.describe(v), value: String.slice(v, 0, 2000)}
  defp preview(v) when is_number(v), do: %{kind: "number", info: Vapor.Studio.Value.describe(v), value: v}
  defp preview(v), do: %{kind: "json", info: %{type: "json"}, value: v |> JSON.encode() |> String.slice(0, 2000)}

  defp audit_json(bytes, req) do
    opts = if k = req["log_key"], do: [log_keys: [k]], else: []
    {verdict, r} = Vapor.Audit.verify(bytes, opts)

    m = case Vapor.Audit.decode(bytes) do
      {:ok, d} -> d.manifest
      _ -> %{}
    end

    %{ok: verdict == :ok, reason: r[:reason] && inspect(r[:reason]), format: r[:format], root_ok: r[:root], root: m["root"],
      system: m["system"], date: m["date"], notes: m["notes"], items: r[:items] || [], signatures: r[:signatures] || [],
      quorum: r[:quorum], anchors: r[:anchors] || [], clauses: Vapor.Audit.clauses(), bytes: byte_size(bytes)}
  end

  defp table_json(t) do
    %{kind: t.kind, box: Tuple.to_list(t.box), rows: t.rows, cols: t.cols, header_rows: t.header_rows, confidence: t[:confidence],
      markdown: t[:markdown] || Vapor.Vision.Table.to_markdown(t), html: Vapor.Vision.Table.to_html(t), csv: Vapor.Vision.Table.to_csv(t),
      cells: Enum.map(t.cells, fn c -> %{row: c.row, col: c.col, rowspan: c[:rowspan] || 1, colspan: c[:colspan] || 1, box: Tuple.to_list(c.box),
                                         text: c[:text] || "", confidence: c[:confidence]} end)}
  end

  defp anchor(%{tlog: nil}, _entry), do: nil
  defp anchor(%{tlog: h}, entry), do: h |> Vapor.Tlog.Holder.anchor(entry) |> Map.update!("proof", &hexes/1)

  defp hexes(b64s), do: Enum.map(b64s, &(&1 |> Base.decode64!() |> Base.encode16(case: :lower)))

  # what a search commits to, as readable text: the query and the receipt
  # anyone can recompute against the library (`POST /v1/vapor/verify`)
  defp search_entry(r) do
    "vapor-search/1\n" <> JSON.encode(%{query: r.query, k: r.k, method: to_string(r.method), library_root: r.library_root,
                                         library_receipt: r.library_receipt})
  end

  defp entry_json(log, i) do
    e = Vapor.Tlog.entry(log, i)
    kind = e |> :binary.split("\n") |> hd() |> String.slice(0, 40)

    %{index: i, kind: kind, leaf: Base.encode16(Vapor.Tlog.leaf_hash(e), case: :lower), bytes: byte_size(e),
      text: if(String.valid?(e), do: String.slice(e, 0, 2000), else: nil), data: Base.encode64(e)}
  end

  # a page out of a PDF has no picture in the browser: a PNG of it, at most 1100 px wide
  defp page_image(img), do: "data:image/png;base64," <> Base.encode64(Vapor.Modal.Image.png(fit(img, 1100), 1))

  defp fit(%Vapor.Modal.Image{w: w} = img, max) when w <= max, do: img
  defp fit(%Vapor.Modal.Image{w: w, h: h, c: c, px: px}, max) do
    s = w / max
    {nw, nh} = {max, max(round(h / s), 1)}

    vals =
      for y <- 0..(nh - 1), x <- 0..(nw - 1), ch <- 0..(c - 1) do
        # the darkest of the source pixels under this one: thin strokes survive
        {sx0, sy0} = {trunc(x * s), trunc(y * s)}
        {sx1, sy1} = {min(trunc((x + 1) * s), w) - 1, min(trunc((y + 1) * s), h) - 1}
        for(yy <- sy0..max(sy1, sy0), xx <- sx0..max(sx1, sx0), do: elem(px, (yy * w + xx) * c + ch)) |> Enum.min()
      end

    Vapor.Modal.Image.new(nw, nh, c, vals)
  end

  defp ocr_inputs(name, bytes) do
    case Vapor.Docs.sniff(name, bytes) do
      :pdf ->
        with {:ok, pages, _} <- Vapor.Docs.PDF.pages(bytes),
             empty = for({t, i} <- Enum.with_index(pages, 1), String.trim(t) == "", do: i),
             {:ok, imgs, _} <- Vapor.Docs.PDF.images(bytes, empty) do
          case for({i, list} <- Enum.sort(imgs), img <- list, do: {{:page, i}, img}) do
            [] -> {:error, "no scanned page in this PDF (every page has a text layer, or its images are not decodable)"}
            l -> {:ok, Enum.take(l, 8)}
          end
        else
          {:error, %Vapor.Rejection{bound: b}} -> {:error, b}
        end

      kind when kind in [:png, :jpeg, :pnm] ->
        case Pictures.read(kind, bytes) do
          {:ok, %{image: img}} -> {:ok, [{name, img}]}
          {:ok, pic} -> {:error, "pixels not decoded: #{Enum.join(List.wrap(pic[:warnings]), "; ")}"}
          {:error, %Vapor.Rejection{bound: b}} -> {:error, b}
        end

      other ->
        {:error, "#{other}: OCR reads PNG, JPEG, PPM and scanned PDFs"}
    end
  end

  # ------------------------------------------------------------ helpers --

  defp lib(ctx), do: Holder.get(ctx.library)

  defp page(sock) do
    case page_html() do
      {:ok, html} -> raw(sock, 200, "text/html; charset=utf-8", html)
      {:error, path} -> json(sock, 404, %{error: %{message: "console page missing (#{path})"}})
    end
  end

  @doc """
  The console page as served: `index.html` with its sibling scripts
  (`<script src="/workbench.js">`, `/gpu_tracer.js`) inlined, so the page
  stays one self-contained document — it can be saved and opened offline.
  """
  def page_html do
    dir = Path.join([to_string(:code.priv_dir(:vapor)), "console"])
    path = Path.join(dir, "index.html")

    case File.read(path) do
      {:ok, html} ->
        {:ok, Regex.replace(~r{<script src="/([a-z_]+\.js)"></script>}, html, fn whole, name ->
          case File.read(Path.join(dir, name)) do
            {:ok, js} -> "<script>\n" <> String.replace(js, "</script", "<\\/script") <> "\n</script>"
            _ -> whole
          end
        end)}
      _ -> {:error, path}
    end
  end

  @doc false
  # the page, with the token kept in an HttpOnly cookie (the address bar keeps no secret after this)
  def login({m, ref} = sock, token) do
    if function_exported?(m, :send_body, 5) do
      {:ok, html} = page_html()
      m.send_body(ref, 200, "text/html; charset=utf-8", html,
                  [{"set-cookie", "vapor_token=#{token}; HttpOnly; SameSite=Strict; Path=/"}])
    else
      json(sock, 406, %{error: %{message: "this transport serves JSON only"}})
    end
  end

  defp asset(sock, name, ctype) do
    case File.read(Path.join([to_string(:code.priv_dir(:vapor)), "console", name])) do
      {:ok, bin} -> raw(sock, 200, ctype, bin)
      _ -> json(sock, 404, %{error: %{message: "#{name} missing"}})
    end
  end

  defp raw({m, ref} = sock, status, ctype, body) do
    if Code.ensure_loaded?(m) and function_exported?(m, :send_body, 5),
      do: m.send_body(ref, status, ctype, body, []),
      else: json(sock, 406, %{error: %{message: "this transport serves JSON only"}})
  end

  defp json({m, ref}, status, obj), do: m.json(ref, status, JSON.encode(Vapor.Quality.Report.plain(obj)), [])
  defp bad(sock, msg), do: json(sock, 400, %{error: %{message: msg, type: "invalid_request_error"}})

  defp ok_or({:ok, v}, _), do: {:ok, v}
  defp ok_or(:error, why), do: {:error, why}
  defp ok_or({:error, _}, why), do: {:error, why}

  defp clamp(n, lo, hi, _d) when is_integer(n), do: n |> max(lo) |> min(hi)
  defp clamp(_, _lo, _hi, d), do: d

  defp method("dense"), do: :dense
  defp method("hybrid"), do: :hybrid
  defp method(_), do: :bm25

  defp spec_json(s) do
    s |> Map.from_struct() |> Map.drop([:config, :adapter]) |> Map.put(:adapter, inspect(s.adapter))
  end

  defp search_json(r) do
    %{query: r.query, k: r.k, method: r.method, root: Base.encode16(r.root, case: :lower), receipt: r.receipt,
      library_root: r.library_root, library_receipt: r.library_receipt,
      hits: Enum.map(r.hits, fn h ->
        Map.take(h, [:doc, :text, :rank, :score, :id, :start, :stop, :file, :file_sha256, :container, :container_sha256])
        |> Map.put(:proof, Enum.map(h.proof, &proof_step/1))
      end)}
  end

  defp proof_step({side, hash}) when is_binary(hash), do: %{side: side, hash: Base.encode16(hash, case: :lower)}
  defp proof_step(hash) when is_binary(hash), do: Base.encode16(hash, case: :lower)
  defp proof_step(other), do: inspect(other)

  defp thresholds(g), do: %{t_noise: g.t_noise, t_natural: g.t_natural, margin: g.margin}

  # the text gate, calibrated once per VM on the frozen corpus
  defp gate do
    case :persistent_term.get({__MODULE__, :gate}, nil) do
      nil ->
        {ref, hold} = Vapor.Quality.Suite.corpus_pt_raw()
        p = Text.profile(ref)
        {:ok, g} = Text.gate(p, hold, len: 120, count: 24)
        :persistent_term.put({__MODULE__, :gate}, {p, g})
        {p, g}

      pg ->
        pg
    end
  end

  # at most 160 px on the long side (nearest neighbour)
  defp thumb(%Vapor.Modal.Image{w: w, h: h, c: c} = img) do
    s = max(w, h) / 160
    if s <= 1 do
      img
    else
      {tw, th} = {max(1, round(w / s)), max(1, round(h / s))}
      vals = for y <- 0..(th - 1), x <- 0..(tw - 1), ch <- 0..(c - 1), do: Vapor.Modal.Image.at(img, min(trunc(x * s), w - 1), min(trunc(y * s), h - 1), ch)
      Vapor.Modal.Image.new(tw, th, c, vals)
    end
  end
end
