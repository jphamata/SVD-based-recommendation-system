defmodule Mix.Tasks.Vapor.Rag do
  @shortdoc "Index files (zip, PDF, Office, images…) into a verifiable library, search it, check a result"
  @moduledoc """
      mix vapor.rag index LIB PATH...            # files or directories (recursive), through the document airlock
      mix vapor.rag search LIB "pergunta" [--k 5] [--json]
      mix vapor.rag image LIB foto.png [--k 5]   # visually similar pictures
      mix vapor.rag show LIB                     # files, warnings, root

  `LIB` is one file (`Vapor.Docs.Library`, in the BEAM's external term
  format, read back with `:safe`). Every hit prints the path through
  containers down to the page, its score, and the SHA-256 of the file it
  came from; `--json` prints the whole result with inclusion proofs and the
  receipt, for `Vapor.Docs.Library.verify/2` anywhere.
  """
  use Mix.Task
  alias Vapor.Docs.Library

  @impl true
  def run(argv) do
    Mix.Task.run("app.start")
    {o, args, _} = OptionParser.parse(argv, strict: [k: :integer, json: :boolean])

    args = Enum.map(args, &Vapor.CLI.utf8_arg/1)

    case args do
      ["index", lib | paths] when paths != [] -> index(lib, paths)
      ["search", lib, query] -> search(load!(lib), query, o)
      ["image", lib, img] -> image(load!(lib), img, o)
      ["show", lib] -> show(load!(lib))
      _ -> Mix.raise("usage: mix vapor.rag index LIB PATH… | search LIB QUERY | image LIB PNG | show LIB")
    end
  end

  defp index(out, paths) do
    lib = if File.regular?(out), do: load!(out), else: Library.new()

    lib =
      paths
      |> Enum.flat_map(fn p -> if File.dir?(p), do: Path.wildcard(Path.join(p, "**/*")) |> Enum.filter(&File.regular?/1) |> Enum.sort(), else: [p] end)
      |> Enum.reduce(lib, fn p, lib ->
        case Library.add(lib, p) do
          {:ok, lib, %{duplicate: d}} -> Mix.shell().info("  = #{d} (already in the library)"); lib
          {:ok, lib, rep} ->
            Mix.shell().info("  + #{p}: #{rep.added} passages, #{rep.images} pictures")
            for w <- rep.warnings, do: Mix.shell().info("      #{w}")
            lib
          {:error, r} -> Mix.shell().error("  ✗ #{p}: #{r.bound}"); lib
        end
      end)

    File.write!(out, :erlang.term_to_binary({:vapor_library, 1, lib}, compressed: 6))
    s = Library.stats(lib)
    Mix.shell().info("#{out}: #{s.files} files, #{s.passages} passages, #{s.images} pictures, root #{s.root}")
  end

  defp load!(path) do
    # :safe decoding creates no atom: the modules that name them are loaded first
    for m <- [Vapor.Docs, Vapor.Docs.Library, Vapor.Docs.Pictures, Vapor.Docs.PDF, Vapor.Docs.Office, Vapor.Docs.Zip,
              Vapor.Docs.Markup, Vapor.RAG, Vapor.Merkle, Vapor.Modal.Image, Vapor.Tensor, Vapor.Program, Vapor.Algebra.Term],
        do: Code.ensure_loaded(m)

    case File.read(path) do
      {:ok, bin} ->
        case :erlang.binary_to_term(bin, [:safe]) do
          {:vapor_library, 1, %Library{} = lib} -> lib
          _ -> Mix.raise("#{path}: not a vapor library")
        end

      {:error, why} ->
        Mix.raise("#{path}: #{inspect(why)}")
    end
  rescue
    ArgumentError -> Mix.raise("#{path}: not a vapor library (or written by a newer vapor)")
  end

  defp search(lib, query, o) do
    r = Library.search(lib, query, k: o[:k] || 5)

    if o[:json] do
      IO.puts(Vapor.Quality.Report.json(%{r | root: Base.encode16(r.root, case: :lower)}))
    else
      if r.hits == [], do: Mix.shell().info("no passage matches #{inspect(query)}")

      for h <- r.hits do
        Mix.shell().info("#{h.rank}. #{h.doc}   (score #{Float.round(h.score, 3)}, file #{String.slice(h.container_sha256 || "", 0, 12)}…)")
        Mix.shell().info("   " <> (h.text |> String.replace("\n", " ") |> String.slice(0, 220)))
      end

      Mix.shell().info("library #{r.library_root}\nreceipt #{r.library_receipt}")
    end
  end

  defp image(lib, path, o) do
    {:ok, name, bytes} = {:ok, Path.basename(path), File.read!(path)}

    case Vapor.Docs.Pictures.read(Vapor.Docs.sniff(name, bytes), bytes) do
      {:ok, %{image: img}} -> for h <- Library.search_image(lib, img, k: o[:k] || 5), do: Mix.shell().info("#{Float.round(h.score, 3)}  #{h.doc}")
      {:ok, _} -> Mix.raise("#{path}: pixels not decodable here (PNG and PPM are)")
      {:error, r} -> Mix.raise("#{path}: #{r.bound}")
    end
  end

  defp show(lib) do
    for f <- lib.files, do: Mix.shell().info("#{String.pad_trailing(to_string(f.kind), 9)} #{String.pad_leading(Integer.to_string(f.bytes), 9)}  #{String.slice(f.sha256, 0, 12)}  #{f.path}")
    for w <- lib.warnings, do: Mix.shell().info("warning: #{w}")
    s = Library.stats(lib)
    Mix.shell().info("#{s.files} files, #{s.passages} passages, #{s.chunks} chunks, #{s.images} pictures, root #{s.root}")
  end
end
