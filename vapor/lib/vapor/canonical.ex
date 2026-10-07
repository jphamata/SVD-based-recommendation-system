defmodule Vapor.Canonical do
  @moduledoc """
  Canonical bytes of a term: deterministic CBOR (RFC 8949 §4.2 — preferred
  serialisation, definite lengths — with the length-first map-key order of
  §4.2.3, i.e. RFC 7049 canonical CBOR, which `cbor2.dumps(x,
  canonical=True)` also produces), the encoding under every signature and
  every content address in vapor.

  Why not `:erlang.term_to_binary(t, [:deterministic])`: OTP documents that
  option as stable only *within* one release, so two verifier nodes on
  different OTP versions could compute different bytes for the same payload
  — and never reach a co-signing quorum — and nothing outside the BEAM can
  check a signature over it. Deterministic CBOR is a fixed, standard,
  language-independent function of the value: a Python, Rust or JavaScript
  verifier recomputes the same bytes with any conforming library.

  The mapping (the *vapor profile* of CBOR):

  | term | CBOR |
  |---|---|
  | `nil`, `true`, `false` | simple values 22, 21, 20 |
  | integers | major types 0/1, shortest form; beyond 64 bits tags 2/3 (bignums, no leading zero bytes) |
  | floats | the shortest of binary16/32/64 that holds the value exactly |
  | binaries | byte strings (major 2) |
  | other atoms | tag 39 ("identifier") over the name as text |
  | lists | arrays |
  | tuples | tag 30305 over an array |
  | maps (structs included) | maps, keys ordered by their encoded bytes: shorter first, then bytewise |

  Pids, references, ports and functions have no canonical form and raise.
  `decode/2` inverts `encode/1` on everything it produces, refusing
  non-canonical input (so a value has exactly one accepted encoding).
  """

  @tuple_tag 30_305
  @atom_tag 39

  @doc "Deterministic CBOR bytes of `term`."
  @spec encode(term) :: binary
  def encode(term), do: term |> enc() |> IO.iodata_to_binary()

  @doc "SHA-256 of the canonical bytes."
  def digest(term), do: :crypto.hash(:sha256, encode(term))

  @doc "Lower-case hex SHA-256 of the canonical bytes."
  def hex_digest(term), do: Base.encode16(digest(term), case: :lower)

  defp enc(nil), do: <<0xF6>>
  defp enc(true), do: <<0xF5>>
  defp enc(false), do: <<0xF4>>
  defp enc(n) when is_integer(n) and n >= 0 and n < 0x1_0000_0000_0000_0000, do: head(0, n)
  defp enc(n) when is_integer(n) and n < 0 and n >= -0x1_0000_0000_0000_0000, do: head(1, -1 - n)
  defp enc(n) when is_integer(n) and n > 0, do: [head(6, 2), bytes(:binary.encode_unsigned(n))]
  defp enc(n) when is_integer(n), do: [head(6, 3), bytes(:binary.encode_unsigned(-1 - n))]
  defp enc(x) when is_float(x), do: float(x)
  defp enc(b) when is_binary(b), do: bytes(b)
  defp enc(a) when is_atom(a), do: [head(6, @atom_tag), text(Atom.to_string(a))]
  defp enc(l) when is_list(l), do: [head(4, length(l)) | Enum.map(l, &enc/1)]
  defp enc(t) when is_tuple(t), do: [head(6, @tuple_tag), enc(Tuple.to_list(t))]

  defp enc(%{} = m) do
    pairs =
      m
      |> Map.to_list()
      |> Enum.map(fn {k, v} -> {encode(k), v} end)
      |> Enum.sort_by(fn {kb, _} -> {byte_size(kb), kb} end)

    [head(5, length(pairs)) | Enum.map(pairs, fn {kb, v} -> [kb, enc(v)] end)]
  end

  defp enc(other), do: raise(ArgumentError, "no canonical encoding for #{inspect(other)}")

  defp bytes(b), do: [head(2, byte_size(b)), b]
  defp text(s), do: [head(3, byte_size(s)), s]

  defp head(major, n) when n < 24, do: <<major::3, n::5>>
  defp head(major, n) when n < 0x100, do: <<major::3, 24::5, n::8>>
  defp head(major, n) when n < 0x10000, do: <<major::3, 25::5, n::16>>
  defp head(major, n) when n < 0x1_0000_0000, do: <<major::3, 26::5, n::32>>
  defp head(major, n), do: <<major::3, 27::5, n::64>>

  # the shortest IEEE width that holds x exactly (preferred serialisation)
  defp float(x) do
    cond do
      fits?(x, 16) -> <<0xF9, x::float-16>>
      fits?(x, 32) -> <<0xFA, x::float-32>>
      true -> <<0xFB, x::float-64>>
    end
  end

  defp fits?(x, 16) do
    <<y::float-16>> = <<x::float-16>>
    y === x
  rescue
    _ -> false
  end

  defp fits?(x, 32) do
    <<y::float-32>> = <<x::float-32>>
    y === x
  rescue
    _ -> false
  end

  # ------------------------------------------------------------- decode --

  @doc """
  Decode canonical bytes. Options: `atoms: :existing` (default — unknown
  atom names are refused, so untrusted input cannot exhaust the atom table)
  or `atoms: :create`.
  """
  @spec decode(binary, keyword) :: {:ok, term} | {:error, term}
  def decode(bin, opts \\ []) when is_binary(bin) do
    mode = Keyword.get(opts, :atoms, :existing)

    case dec(bin, mode) do
      {:ok, term, ""} ->
        # exactly one accepted encoding per value
        if encode(term) == bin, do: {:ok, term}, else: {:error, :not_canonical}

      {:ok, _, _rest} ->
        {:error, :trailing_bytes}

      {:error, _} = e ->
        e
    end
  rescue
    e in [ArgumentError, MatchError, FunctionClauseError] -> {:error, {:malformed, Exception.message(e)}}
  end

  defp dec(<<0xF6, r::binary>>, _), do: {:ok, nil, r}
  defp dec(<<0xF5, r::binary>>, _), do: {:ok, true, r}
  defp dec(<<0xF4, r::binary>>, _), do: {:ok, false, r}
  defp dec(<<0xF9, x::float-16, r::binary>>, _), do: {:ok, x, r}
  defp dec(<<0xFA, x::float-32, r::binary>>, _), do: {:ok, x, r}
  defp dec(<<0xFB, x::float-64, r::binary>>, _), do: {:ok, x, r}

  defp dec(<<major::3, info::5, r::binary>>, mode) do
    with {:ok, n, r} <- arg(info, r) do
      case major do
        0 -> {:ok, n, r}
        1 -> {:ok, -1 - n, r}
        2 -> take(r, n)
        3 -> take(r, n)
        4 -> items(r, n, mode, [])
        5 -> pairs(r, n, mode, %{})
        6 -> tagged(n, r, mode)
        _ -> {:error, {:unsupported_major, major}}
      end
    end
  end

  defp dec(<<>>, _), do: {:error, :truncated}

  defp arg(n, r) when n < 24, do: {:ok, n, r}
  defp arg(24, <<n::8, r::binary>>), do: {:ok, n, r}
  defp arg(25, <<n::16, r::binary>>), do: {:ok, n, r}
  defp arg(26, <<n::32, r::binary>>), do: {:ok, n, r}
  defp arg(27, <<n::64, r::binary>>), do: {:ok, n, r}
  defp arg(_, _), do: {:error, :bad_argument}

  defp take(r, n) when byte_size(r) >= n, do: {:ok, binary_part(r, 0, n), binary_part(r, n, byte_size(r) - n)}
  defp take(_, _), do: {:error, :truncated}

  defp items(r, 0, _mode, acc), do: {:ok, Enum.reverse(acc), r}

  defp items(r, n, mode, acc) do
    with {:ok, x, r} <- dec(r, mode), do: items(r, n - 1, mode, [x | acc])
  end

  defp pairs(r, 0, _mode, acc), do: {:ok, acc, r}

  defp pairs(r, n, mode, acc) do
    with {:ok, k, r} <- dec(r, mode),
         {:ok, v, r} <- dec(r, mode) do
      pairs(r, n - 1, mode, Map.put(acc, k, v))
    end
  end

  defp tagged(2, r, mode), do: with({:ok, b, r} <- dec(r, mode), do: {:ok, :binary.decode_unsigned(b), r})
  defp tagged(3, r, mode), do: with({:ok, b, r} <- dec(r, mode), do: {:ok, -1 - :binary.decode_unsigned(b), r})
  defp tagged(@tuple_tag, r, mode), do: with({:ok, l, r} when is_list(l) <- dec(r, mode), do: {:ok, List.to_tuple(l), r})

  defp tagged(@atom_tag, r, mode) do
    with {:ok, s, r} when is_binary(s) <- dec(r, mode) do
      case mode do
        :create -> {:ok, String.to_atom(s), r}
        :existing -> {:ok, String.to_existing_atom(s), r}
      end
    end
  end

  defp tagged(tag, _r, _mode), do: {:error, {:unsupported_tag, tag}}
end
