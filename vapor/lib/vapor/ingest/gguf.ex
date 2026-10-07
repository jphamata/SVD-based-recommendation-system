defmodule Vapor.Ingest.GGUF do
  @moduledoc """
  The GGUF airlock (llama.cpp's container, versions 2 and 3):

      "GGUF" | version:u32 | n_tensors:u64 | n_kv:u64
      n_kv × (key:str, type:u32, value)          metadata
      n_tensors × (name:str, n_dims:u32, dims:u64[n_dims], ggml_type:u32, offset:u64)
      padding to `general.alignment` (default 32)
      tensor data

  Strings are `u64 length + bytes`; values are the 13 GGUF types (integers,
  floats, bool, string, and arrays of any of them).

  Nothing is trusted: every count is checked against the bytes that remain
  in the file before anything is allocated (an array of `n` elements needs
  at least `n` bytes; `n` strings at least `8n`), strings must be UTF-8,
  keys must be unique, arrays nest at most 2 deep, tensor dimensions are
  positive, and every tensor's data lies inside the file, aligned, without
  overlapping another. Only the metadata and the tensor index are parsed
  here — from a prefix of the file that grows until it suffices, so a
  multi-gigabyte model is not read to learn its vocabulary.
  """
  import Bitwise
  alias Vapor.Rejection

  @magic "GGUF"
  @types %{0 => :u8, 1 => :i8, 2 => :u16, 3 => :i16, 4 => :u32, 5 => :i32, 6 => :f32, 7 => :bool,
           8 => :string, 9 => :array, 10 => :u64, 11 => :i64, 12 => :f64}
  @min_size %{u8: 1, i8: 1, u16: 2, i16: 2, u32: 4, i32: 4, f32: 4, bool: 1, string: 8, array: 12,
              u64: 8, i64: 8, f64: 8}

  # ggml tensor types: {name, block elements, block bytes}
  @ggml %{0 => {:f32, 1, 4}, 1 => {:f16, 1, 2}, 2 => {:q4_0, 32, 18}, 3 => {:q4_1, 32, 20},
          6 => {:q5_0, 32, 22}, 7 => {:q5_1, 32, 24}, 8 => {:q8_0, 32, 34}, 9 => {:q8_1, 32, 40},
          10 => {:q2_k, 256, 84}, 11 => {:q3_k, 256, 110}, 12 => {:q4_k, 256, 144},
          13 => {:q5_k, 256, 176}, 14 => {:q6_k, 256, 210}, 15 => {:q8_k, 256, 292},
          24 => {:i8, 1, 1}, 25 => {:i16, 1, 2}, 26 => {:i32, 1, 4}, 27 => {:i64, 1, 8},
          28 => {:f64, 1, 8}, 30 => {:bf16, 1, 2}}

  @type t :: %{version: 2 | 3, metadata: %{String.t() => term}, types: %{String.t() => atom | {:array, atom}},
               tensors: [map], data_start: non_neg_integer}

  @doc "Parse and check the header, metadata and tensor index of a GGUF file."
  @spec read(Path.t()) :: {:ok, t} | {:error, Rejection.t()}
  def read(path) do
    with {:ok, %{size: size}} <- File.stat(path),
         {:ok, f} <- File.open(path, [:read, :binary]) do
      try do
        grow(f, path, size, min(size, 1 <<< 20))
      after
        File.close(f)
      end
    else
      {:error, %Rejection{}} = e -> e
      {:error, why} -> reject(path, "readable file (#{inspect(why)})")
    end
  end

  # parse the prefix; if it ends too early, read four times more (≤ size)
  defp grow(f, path, size, n) do
    {:ok, bin} = :file.pread(f, 0, n)

    try do
      parse(bin, size)
    catch
      :short when n < size -> grow(f, path, size, min(size, n * 4))
      :short -> reject(path, "a complete header (the file ends inside it)")
      {:bad, bound} -> reject(path, bound)
    end
  end

  defp parse(<<@magic, version::32-little, nt::64-little, nkv::64-little, rest::binary>> = bin, size) do
    unless version in [2, 3], do: throw({:bad, "GGUF version 2 or 3 (got #{version})"})
    room = size - 24
    unless nkv * 12 <= room and nt * 28 <= room, do: throw({:bad, "#{nkv} keys and #{nt} tensors fit in the file"})

    {kvs, rest} = times(nkv, rest, [], fn r -> kv(r, size) end)
    keys = Enum.map(kvs, &elem(&1, 0))
    if length(Enum.uniq(keys)) != length(keys), do: throw({:bad, "unique metadata keys"})
    meta = Map.new(kvs, fn {k, v, _} -> {k, v} end)
    types = Map.new(kvs, fn {k, _, t} -> {k, t} end)

    {infos, rest} = times(nt, rest, [], &tensor_info/1)
    align = Map.get(meta, "general.alignment", 32)
    unless is_integer(align) and align > 0 and (align &&& (align - 1)) == 0, do: throw({:bad, "general.alignment a power of two"})

    consumed = byte_size(bin) - byte_size(rest)
    data_start = consumed + rem(align - rem(consumed, align), align)
    tensors = check_tensors(infos, data_start, size, align)
    {:ok, %{version: version, metadata: meta, types: types, tensors: tensors, data_start: data_start}}
  end

  defp parse(bin, _size) when byte_size(bin) < 24, do: throw(:short)
  defp parse(_bin, _size), do: throw({:bad, "GGUF magic"})

  defp times(0, rest, acc, _f), do: {Enum.reverse(acc), rest}

  defp times(n, rest, acc, f) do
    {x, rest} = f.(rest)
    times(n - 1, rest, [x | acc], f)
  end

  defp kv(bin, size) do
    {key, rest} = string(bin)

    case rest do
      <<t::32-little, rest::binary>> ->
        t = type(t)
        {v, rest2} = value(t, rest, size, 0)
        # arrays keep their element type (a writer must reproduce it: llama.cpp checks)
        t = if t == :array, do: (<<et::32-little, _::binary>> = rest; {:array, type(et)}), else: t
        {{key, v, t}, rest2}

      _ ->
        throw(:short)
    end
  end

  defp type(t), do: Map.get(@types, t) || throw({:bad, "known GGUF value type (got #{t})"})

  defp string(<<n::64-little, rest::binary>>) do
    case rest do
      <<s::binary-size(n), rest::binary>> ->
        if String.valid?(s), do: {s, rest}, else: throw({:bad, "UTF-8 strings"})

      _ ->
        throw(:short)
    end
  end

  defp string(_), do: throw(:short)

  defp value(:u8, <<v::8, r::binary>>, _, _), do: {v, r}
  defp value(:i8, <<v::signed-8, r::binary>>, _, _), do: {v, r}
  defp value(:u16, <<v::16-little, r::binary>>, _, _), do: {v, r}
  defp value(:i16, <<v::signed-16-little, r::binary>>, _, _), do: {v, r}
  defp value(:u32, <<v::32-little, r::binary>>, _, _), do: {v, r}
  defp value(:i32, <<v::signed-32-little, r::binary>>, _, _), do: {v, r}
  defp value(:u64, <<v::64-little, r::binary>>, _, _), do: {v, r}
  defp value(:i64, <<v::signed-64-little, r::binary>>, _, _), do: {v, r}
  defp value(:f32, <<v::32-little-bits, r::binary>>, _, _), do: {float(v, 32), r}
  defp value(:f64, <<v::64-little-bits, r::binary>>, _, _), do: {float(v, 64), r}
  defp value(:bool, <<v::8, r::binary>>, _, _) when v in [0, 1], do: {v == 1, r}
  defp value(:bool, <<_::8, _::binary>>, _, _), do: throw({:bad, "booleans are 0 or 1"})
  defp value(:string, bin, _, _), do: string(bin)

  defp value(:array, <<t::32-little, n::64-little, r::binary>>, size, depth) do
    et = type(t)
    if depth >= 2, do: throw({:bad, "arrays nest at most 2 deep"})
    if n * @min_size[et] > size, do: throw({:bad, "array of #{n} #{et} fits in the file"})
    array(et, n, r, size, depth + 1)
  end

  defp value(_t, _bin, _, _), do: throw(:short)

  # fixed-width numeric arrays are decoded in one comprehension
  defp array(et, n, r, _size, _depth) when et in [:u8, :i8, :u16, :i16, :u32, :i32, :u64, :i64, :f32, :f64] do
    w = @min_size[et]

    case r do
      <<raw::binary-size(n * w), rest::binary>> -> {numbers(et, raw), rest}
      _ -> throw(:short)
    end
  end

  defp array(et, n, r, size, depth), do: times(n, r, [], &value(et, &1, size, depth))

  defp numbers(:u8, raw), do: :binary.bin_to_list(raw)
  defp numbers(:i8, raw), do: for(<<v::signed-8 <- raw>>, do: v)
  defp numbers(:u16, raw), do: for(<<v::16-little <- raw>>, do: v)
  defp numbers(:i16, raw), do: for(<<v::signed-16-little <- raw>>, do: v)
  defp numbers(:u32, raw), do: for(<<v::32-little <- raw>>, do: v)
  defp numbers(:i32, raw), do: for(<<v::signed-32-little <- raw>>, do: v)
  defp numbers(:u64, raw), do: for(<<v::64-little <- raw>>, do: v)
  defp numbers(:i64, raw), do: for(<<v::signed-64-little <- raw>>, do: v)
  defp numbers(:f32, raw), do: for(<<v::32-little-bits <- raw>>, do: float(v, 32))
  defp numbers(:f64, raw), do: for(<<v::64-little-bits <- raw>>, do: float(v, 64))

  # non-finite floats are kept as their bit pattern (a metadata value, not arithmetic)
  defp float(bits, 32) do
    case bits do
      <<v::float-32-little>> -> v
      <<b::32-little>> -> {:nonfinite_f32, b}
    end
  end

  defp float(bits, 64) do
    case bits do
      <<v::float-64-little>> -> v
      <<b::64-little>> -> {:nonfinite_f64, b}
    end
  end

  defp tensor_info(bin) do
    {name, rest} = string(bin)

    case rest do
      <<nd::32-little, rest::binary>> when nd in 1..4 ->
        case rest do
          <<dims::binary-size(nd * 8), t::32-little, off::64-little, rest::binary>> ->
            dims = for <<d::64-little <- dims>>, do: d
            {%{name: name, dims: dims, type: t, offset: off}, rest}

          _ ->
            throw(:short)
        end

      <<nd::32-little, _::binary>> ->
        throw({:bad, "tensor #{inspect(name)} has 1–4 dimensions (got #{nd})"})

      _ ->
        throw(:short)
    end
  end

  defp check_tensors(infos, data_start, size, align) do
    names = Enum.map(infos, & &1.name)
    if length(Enum.uniq(names)) != length(names), do: throw({:bad, "unique tensor names"})

    checked =
      Enum.map(infos, fn %{name: n, dims: dims, type: t, offset: off} = i ->
        {kind, blk, bytes} = Map.get(@ggml, t) || throw({:bad, "tensor #{inspect(n)}: known ggml type (got #{t})"})
        unless Enum.all?(dims, &(&1 > 0)), do: throw({:bad, "tensor #{inspect(n)}: positive dimensions"})
        unless rem(hd(dims), blk) == 0, do: throw({:bad, "tensor #{inspect(n)}: rows of whole #{kind} blocks"})
        len = div(Enum.product(dims), blk) * bytes
        unless rem(off, align) == 0, do: throw({:bad, "tensor #{inspect(n)}: aligned offset"})
        unless data_start + off + len <= size, do: throw({:bad, "tensor #{inspect(n)}: data inside the file"})
        %{i | type: kind} |> Map.put(:bytes, len)
      end)

    checked
    |> Enum.sort_by(& &1.offset)
    |> Enum.reduce(0, fn %{name: n, offset: off, bytes: len}, at ->
      if off < at, do: throw({:bad, "tensor #{inspect(n)}: no overlap"})
      off + len
    end)

    checked
  end

  # ------------------------------------------------------------- writing --

  @doc """
  Write a GGUF (version 3) file: `metadata` is a list of `{key, value}`
  (written in that order), `tensors` a list of `{name, ggml_type, dims,
  bytes}` with `dims` in ggml order (innermost first) and `bytes` the
  payload. Value types are inferred — integers `u32` (`i32` if negative,
  `u64`/`i64` if wide), floats `f32`, booleans, strings, and homogeneous
  lists as arrays — or given as `{type, value}` (`{:array, type, list}`).
  The result passes `read/1` (tested), and gguf-py reads it.
  """
  def write(path, metadata, tensors, align \\ 32) do
    ids = Map.new(@ggml, fn {id, {kind, _, _}} -> {kind, id} end)
    tids = Map.new(@types, fn {id, t} -> {t, id} end)

    {index, _} =
      Enum.map_reduce(tensors, 0, fn {name, kind, dims, bytes}, off ->
        {_, blk, bb} = @ggml[ids[kind]]
        ^bytes = binary_part(bytes, 0, div(Enum.product(dims), blk) * bb)
        entry = [str(name), <<length(dims)::32-little>>, Enum.map(dims, &<<&1::64-little>>), <<ids[kind]::32-little, off::64-little>>]
        {entry, off + pad(byte_size(bytes), align)}
      end)

    kvs = Enum.map(metadata, fn {k, v} -> [str(k), typed(v, tids)] end)
    head = IO.iodata_to_binary([@magic, <<3::32-little, length(tensors)::64-little, length(metadata)::64-little>>, kvs, index])
    head = [head, :binary.copy(<<0>>, pad(byte_size(head), align) - byte_size(head))]
    data = Enum.map(tensors, fn {_, _, _, b} -> [b, :binary.copy(<<0>>, pad(byte_size(b), align) - byte_size(b))] end)
    File.write(path, [head, data])
  end

  defp pad(n, a), do: n + rem(a - rem(n, a), a)
  defp str(s), do: [<<byte_size(s)::64-little>>, s]

  defp typed(v, tids) do
    {t, payload} = encode(v)
    [<<tids[t]::32-little>>, payload]
  end

  defp encode({{:array, et}, xs}), do: encode({:array, et, xs})
  defp encode({:array, et, xs}), do: {:array, [<<Map.new(@types, fn {i, t} -> {t, i} end)[et]::32-little, length(xs)::64-little>>, Enum.map(xs, &elem(encode({et, &1}), 1))]}
  defp encode({t, v}) when t in [:u8, :i8, :u16, :i16, :u32, :i32, :u64, :i64, :f32, :f64, :bool, :string], do: {t, scalar(t, v)}
  defp encode(v) when is_boolean(v), do: {:bool, scalar(:bool, v)}
  defp encode(v) when is_binary(v), do: {:string, str(v)}
  defp encode(v) when is_float(v), do: {:f32, scalar(:f32, v)}
  defp encode(v) when is_integer(v), do: (t = int_type([v]); {t, scalar(t, v)})

  defp encode(xs) when is_list(xs) do
    et =
      cond do
        Enum.all?(xs, &is_integer/1) -> int_type(xs)
        Enum.all?(xs, &is_number/1) -> :f32
        Enum.all?(xs, &is_binary/1) -> :string
        Enum.all?(xs, &is_boolean/1) -> :bool
      end

    encode({:array, et, xs})
  end

  defp int_type(xs) do
    {lo, hi} = Enum.min_max(xs, fn -> {0, 0} end)

    cond do
      lo >= 0 and hi < 1 <<< 32 -> :u32
      lo >= -(1 <<< 31) and hi < 1 <<< 31 -> :i32
      lo >= 0 -> :u64
      true -> :i64
    end
  end

  defp scalar(:string, v), do: str(v)
  defp scalar(:bool, v), do: if(v, do: <<1>>, else: <<0>>)
  defp scalar(:f32, {:nonfinite_f32, b}), do: <<b::32-little>>
  defp scalar(:f64, {:nonfinite_f64, b}), do: <<b::64-little>>
  defp scalar(:f32, v), do: <<v::float-32-little>>
  defp scalar(:f64, v), do: <<v::float-64-little>>
  defp scalar(t, v) when t in [:u8, :i8], do: <<v::8>>
  defp scalar(t, v) when t in [:u16, :i16], do: <<v::16-little>>
  defp scalar(t, v) when t in [:u32, :i32], do: <<v::32-little>>
  defp scalar(t, v) when t in [:u64, :i64], do: <<v::64-little>>

  defp reject(path, bound), do: {:error, Rejection.new({:gguf, path}, bound, "re-export the file")}
end
