defmodule Vapor.Docs.Zip do
  @moduledoc """
  A zip reader that cannot be bombed: the central directory is parsed here,
  every member is inflated in chunks (`:zlib.safeInflate/2`) with the
  remaining budget as a hard cap, and its CRC-32 and size are checked
  against the header — a member that inflates beyond what it declared is a
  rejection, whatever the header says. Encrypted members are skipped with a
  warning; Zip64 archives are refused by name.

  The flavour of an archive comes from its members, not its name: Word
  (`word/document.xml`), Excel (`xl/workbook.xml`), PowerPoint
  (`ppt/presentation.xml`), OpenDocument and EPUB (the `mimetype` member).
  """
  alias Vapor.Docs
  alias Vapor.Rejection

  @doc "`:zip | :docx | :xlsx | :pptx | :odf | :epub` from the member names (and `mimetype`)."
  def flavour(bytes) do
    case entries(bytes) do
      {:ok, es} ->
        names = MapSet.new(es, & &1.name)

        cond do
          MapSet.member?(names, "word/document.xml") -> :docx
          MapSet.member?(names, "xl/workbook.xml") -> :xlsx
          MapSet.member?(names, "ppt/presentation.xml") -> :pptx
          MapSet.member?(names, "mimetype") ->
            case member(bytes, Enum.find(es, &(&1.name == "mimetype")), 256) do
              {:ok, "application/epub+zip" <> _} -> :epub
              {:ok, "application/vnd.oasis.opendocument." <> _} -> :odf
              _ -> :zip
            end
          true -> :zip
        end

      _ ->
        :zip
    end
  end

  @doc false
  def extract(kind, name, bytes, depth, opts, budget, acc) do
    with {:ok, es} <- entries(bytes),
         :ok <- need(length(es) <= opts[:max_entries], name, "at most #{opts[:max_entries]} members (#{length(es)})"),
         declared = es |> Enum.map(& &1.usize) |> Enum.sum(),
         :ok <- need(declared + :counters.get(budget, 1) <= opts[:max_total], name,
                     "declared expansion #{declared} bytes within the ingest budget #{opts[:max_total]} (a zip bomb?)"),
         :ok <- ratios(name, es, opts[:max_ratio]) do
      files = Enum.reject(es, &(String.ends_with?(&1.name, "/") or String.starts_with?(&1.name, "__MACOSX/") or Path.basename(&1.name) == ".DS_Store"))
      {enc, files} = Enum.split_with(files, & &1.encrypted)
      acc = Docs.add(acc, [], Enum.map(enc, &"#{name}!/#{&1.name}: encrypted member, skipped"))

      case kind do
        k when k in [:docx, :xlsx, :pptx, :odf] -> office(k, name, bytes, files, opts, budget, acc)
        :epub -> members(name, bytes, Enum.filter(files, &(Path.extname(&1.name) in ~w(.xhtml .html .htm))), depth, opts, budget, acc)
        :zip -> members(name, bytes, files, depth, opts, budget, acc)
      end
    end
  end

  # members over 1 MiB that compress beyond max_ratio are refused unread
  defp ratios(name, es, max) do
    case Enum.find(es, &(&1.usize > 1_048_576 and &1.usize > max * max(&1.csize, 1))) do
      nil -> :ok
      e -> need(false, name, "#{e.name}: a compression ratio ≤ #{max}× (declares #{e.usize} bytes from #{e.csize}: a zip bomb?)")
    end
  end

  defp members(name, bytes, files, depth, opts, budget, acc) do
    Enum.reduce_while(files, {:ok, acc}, fn e, {:ok, acc} ->
      left = opts[:max_total] - :counters.get(budget, 1)

      with {:ok, data} <- member(bytes, e, min(left, opts[:max_bytes])),
           {:ok, acc} <- Docs.visit("#{name}!/#{e.name}", data, depth + 1, opts, budget, acc) do
        {:cont, {:ok, acc}}
      else
        err -> {:halt, err}
      end
    end)
  end

  # Office documents: the XML members the reader needs, inflated within budget
  defp office(kind, name, bytes, files, opts, budget, acc) do
    wanted = Enum.filter(files, &(Path.extname(&1.name) in ~w(.xml .rels)))

    Enum.reduce_while(wanted, {:ok, %{}}, fn e, {:ok, m} ->
      case member(bytes, e, opts[:max_bytes]) do
        {:ok, data} -> :counters.add(budget, 1, byte_size(data)); {:cont, {:ok, Map.put(m, e.name, data)}}
        err -> {:halt, err}
      end
    end)
    |> case do
      {:ok, xml} -> {:ok, Docs.office(kind, name, xml, acc)}
      err -> err
    end
  end

  # ----------------------------------------------------------- the format --

  @doc "Central-directory entries: `%{name, method, crc, csize, usize, offset, encrypted}`."
  def entries(bytes) do
    with {:ok, eocd} <- eocd(bytes),
         <<"PK", 5, 6, _disk::16-little, _cd_disk::16-little, _n_here::16-little, n::16-little, cd_size::32-little,
           cd_off::32-little, _::binary>> <- binary_part(bytes, eocd, byte_size(bytes) - eocd),
         :ok <- need(n != 0xFFFF and cd_off != 0xFFFF_FFFF, "zip", "a classic zip (Zip64 is not supported)"),
         true <- cd_off + cd_size <= byte_size(bytes) || bad("the central directory inside the file") do
      central(binary_part(bytes, cd_off, cd_size), n, [])
    else
      {:error, _} = e -> e
      _ -> bad("a zip end-of-central-directory record")
    end
  end

  defp eocd(bytes) do
    from = max(0, byte_size(bytes) - 65_557)
    tail = binary_part(bytes, from, byte_size(bytes) - from)

    case :binary.matches(tail, <<"PK", 5, 6>>) do
      [] -> bad("a zip end-of-central-directory record")
      ms -> {:ok, from + elem(List.last(ms), 0)}
    end
  end

  defp central(_bin, 0, acc), do: {:ok, Enum.reverse(acc)}

  defp central(<<"PK", 1, 2, _made::16, _need::16, flags::16-little, method::16-little, _t::16, _d::16, crc::32-little,
                 csize::32-little, usize::32-little, nl::16-little, xl::16-little, cl::16-little, _disk::16, _ia::16, _ea::32,
                 off::32-little, name::binary-size(nl), _x::binary-size(xl), _c::binary-size(cl), rest::binary>>, n, acc) do
    e = %{name: Docs.utf8(name), method: method, crc: crc, csize: csize, usize: usize, offset: off, encrypted: Bitwise.band(flags, 1) == 1}
    central(rest, n - 1, [e | acc])
  end

  defp central(_, _, _), do: bad("well-formed central-directory entries")

  @doc "Inflate one member, never producing more than `cap` bytes; CRC and size checked."
  def member(bytes, e, cap) do
    with <<"PK", 3, 4, _::binary-size(22), nl::16-little, xl::16-little, _::binary>> <- binary_part(bytes, e.offset, min(30, byte_size(bytes) - e.offset)),
         start = e.offset + 30 + nl + xl,
         true <- start + e.csize <= byte_size(bytes) || bad("member #{e.name} inside the file"),
         raw = binary_part(bytes, start, e.csize),
         {:ok, data} <- decode(e.method, raw, min(cap, e.usize), e.name),
         true <- (byte_size(data) == e.usize and :erlang.crc32(data) == e.crc) || bad("member #{e.name} matching its declared size and CRC") do
      {:ok, data}
    else
      {:error, _} = err -> err
      _ -> bad("a local header for #{e.name}")
    end
  end

  defp decode(0, raw, cap, name), do: if(byte_size(raw) <= cap, do: {:ok, raw}, else: over(name, cap))

  defp decode(8, raw, cap, name) do
    z = :zlib.open()
    :ok = :zlib.inflateInit(z, -15)

    try do
      inflate(z, :zlib.safeInflate(z, raw), [], 0, cap, name)
    after
      :zlib.close(z)
    end
  end

  defp decode(m, _raw, _cap, name), do: bad("stored or deflated members (#{name} uses method #{m})")

  defp inflate(z, {state, out}, acc, n, cap, name) do
    n = n + IO.iodata_length(out)

    cond do
      n > cap -> over(name, cap)
      state == :finished -> {:ok, IO.iodata_to_binary(Enum.reverse([out | acc]))}
      true -> inflate(z, :zlib.safeInflate(z, []), [out | acc], n, cap, name)
    end
  end

  defp over(name, cap), do: {:error, Rejection.new({:zip, name}, "at most #{cap} inflated bytes (it inflates beyond its header or the budget: a zip bomb?)", "refuse the archive")}

  defp bad(b), do: {:error, Rejection.new(:zip, b, "check the archive")}
  defp need(true, _n, _b), do: :ok
  defp need(false, n, b), do: {:error, Rejection.new({:zip, n}, b, "refuse the archive or raise the limits")}
end
