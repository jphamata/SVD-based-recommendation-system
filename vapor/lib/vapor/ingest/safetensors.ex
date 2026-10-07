defmodule Vapor.Ingest.Safetensors do
  @moduledoc """
  The safetensors airlock: `[u64 header length][JSON header][data]`.

  Nothing from the file is trusted until the header has been checked in
  full (Axiom 5):

    * the header is bounded (≤ 100 MiB) and is a JSON object without
      duplicate keys (`Vapor.JSON` refuses them);
    * every tensor names a known dtype, a shape of non-negative integers and
      `data_offsets = [begin, end]` inside the data section with
      `end − begin = numel · sizeof(dtype)`;
    * the tensors tile the data section exactly — no overlap and no hole
      (the format's own rule against polyglot files).

  Only then is data read. Every dtype of the format is recognised (sizes
  checked, sub-byte ones by bits); vapor's tensors are binary32, `s32`,
  `s8` or `u8`, and each stored dtype maps onto them by a stated rule:

  | stored | vapor | rule |
  |---|---|---|
  | `F32` `I32` `I8` `U8` | same | the bytes |
  | `F16` `BF16` | `f32` | exact widening (`bf16` a 16-bit shift, `f16` incl. subnormals) |
  | `F8_E4M3` `F8_E4M3FNUZ` `F8_E5M2` `F8_E5M2FNUZ` `F8_E8M0` | `f32` | exact widening (a 256-entry table per format) |
  | `F64` | `f32` | rounded to nearest-even (overflow to ±∞), as numpy and torch |
  | `I16` `U16` `BOOL` | `s32` | exact; `BOOL` bytes must be 0 or 1 |
  | `I64` `U32` `U64` | `s32` | exact when every value fits, else refused |
  | `F4` `F6_E2M3` `F6_E3M2` `C64` | — | recognised and refused by name |

  Writing stores `f32`, `s32`, `s8`, `u8`, and narrows binary32 to `BF16`
  or `F16` on request (round to nearest-even, NaN canonical), as torch
  does. Checkpoints larger than a shard limit are written in the Hugging
  Face sharded form (`write_sharded/4`). A rejection names the tensor and
  the violated bound.
  """
  import Bitwise
  alias Vapor.{Rejection, Tensor}

  @max_header 100 * 1024 * 1024
  # bits per element of every dtype the format defines
  @bits %{"F64" => 64, "F32" => 32, "F16" => 16, "BF16" => 16, "F8_E4M3" => 8, "F8_E4M3FNUZ" => 8, "F8_E5M2" => 8,
          "F8_E5M2FNUZ" => 8, "F8_E8M0" => 8, "F6_E2M3" => 6, "F6_E3M2" => 6, "F4" => 4, "C64" => 64,
          "I64" => 64, "I32" => 32, "I16" => 16, "I8" => 8, "U64" => 64, "U32" => 32, "U16" => 16, "U8" => 8, "BOOL" => 8}
  @readable ~w(F64 F32 F16 BF16 F8_E4M3 F8_E4M3FNUZ F8_E5M2 F8_E5M2FNUZ F8_E8M0 I64 I32 I16 I8 U64 U32 U16 U8 BOOL)

  @type entry :: %{name: String.t(), dtype: String.t(), shape: [non_neg_integer], offsets: {non_neg_integer, non_neg_integer}}

  @doc "Validated index of a file: entries (sorted by offset), metadata and the data section start."
  @spec index(Path.t()) :: {:ok, %{entries: [entry], metadata: map, data_start: non_neg_integer}} | {:error, Rejection.t()}
  def index(path) do
    with {:ok, size} <- file_size(path),
         {:ok, f} <- File.open(path, [:read, :binary]) do
      try do
        index_open(f, path, size)
      after
        File.close(f)
      end
    else
      {:error, %Rejection{}} = e -> e
      {:error, why} -> reject(path, "readable file (#{inspect(why)})", "check the path")
    end
  end

  defp index_open(f, path, size) do
    with {:ok, <<hlen::64-little>>} <- pread(f, 0, 8, path, size),
         :ok <- check(hlen <= @max_header and 8 + hlen <= size, path, "header length #{hlen} within the file and ≤ 100 MiB"),
         {:ok, hbin} <- pread(f, 8, hlen, path, size),
         {:ok, header} <- json(hbin, path),
         {:ok, entries} <- entries(header, size - 8 - hlen, path) do
      {:ok, %{entries: entries, metadata: Map.get(header, "__metadata__", %{}), data_start: 8 + hlen}}
    end
  end

  @doc """
  Read tensors (all, or those named in `:only`). Returns
  `{:ok, %{name => Vapor.Tensor}}` or the first rejection. With
  `bf16: :keep`, `BF16` tensors stay bfloat16 (vapor's `:bf16`, which
  means its exact widening) instead of being widened on load.
  """
  def read(path, opts \\ []) do
    only = opts[:only] && MapSet.new(opts[:only])
    keep? = Keyword.get(opts, :bf16) == :keep

    with {:ok, %{entries: entries, data_start: ds}} <- index(path),
         {:ok, f} <- File.open(path, [:read, :binary]) do
      try do
        entries
        |> Enum.filter(&(only == nil or MapSet.member?(only, &1.name)))
        |> Enum.reduce_while({:ok, %{}}, fn e, {:ok, acc} ->
          {b, en} = e.offsets
          # a 0-dimensional tensor (a scalar: CLIP's logit_scale) is read as [1]
          e = if e.shape == [], do: %{e | shape: [1]}, else: e

          with :ok <- check(e.dtype in @readable, e.name, "dtype ∈ #{inspect(@readable)}, got #{e.dtype}"),
               {:ok, raw} <- pread(f, ds + b, en - b, path, ds + en) do
            case if(keep? and e.dtype == "BF16", do: Tensor.new(:bf16, e.shape, raw), else: tensor(e.dtype, e.shape, raw)) do
              {:error, bound} -> {:halt, reject(e.name, bound, "convert the tensor before export")}
              t -> {:cont, {:ok, Map.put(acc, e.name, t)}}
            end
          else
            err -> {:halt, err}
          end
        end)
      after
        File.close(f)
      end
    end
  end

  @doc """
  Write tensors (`f32`, `s32`, `s8`, `u8`) in canonical order (by name); the
  header is space-padded so the data section starts 8-byte aligned.
  Option `:as` stores `f32` tensors narrowed: `"BF16"` or `"F16"` for all
  of them, or a function `name -> "F32" | "BF16" | "F16"`.
  """
  def write(path, tensors, metadata \\ %{}, opts \\ []) do
    File.write(path, encode(tensors, metadata, opts))
  end

  @doc "The bytes of a safetensors file (see `write/4`)."
  def encode(tensors, metadata \\ %{}, opts \\ []) do
    as = as_fun(opts)

    stored =
      tensors
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.map(fn {name, %Tensor{} = t} -> {name, t.shape, stored(t, as.(name))} end)

    [header(Enum.map(stored, fn {name, shape, {dt, data}} -> {name, dt, shape, byte_size(data)} end), metadata)
     | Enum.map(stored, fn {_, _, {_, data}} -> data end)]
  end

  @doc """
  The header of a safetensors file — length prefix, JSON, space padding to
  8 bytes — for `[{name, dtype, shape, bytes}]` in the order their data
  will follow. With it a file can be written one tensor at a time
  (`Vapor.Merge.stream/3`) and come out byte for byte as `encode/3`.
  """
  def header(entries, metadata \\ %{}) do
    {h, _} =
      Enum.reduce(entries, {%{}, 0}, fn {name, dt, shape, n}, {h, off} ->
        {Map.put(h, name, %{"dtype" => dt, "shape" => shape, "data_offsets" => [off, off + n]}), off + n}
      end)

    json = h |> then(&if(map_size(metadata) > 0, do: Map.put(&1, "__metadata__", metadata), else: &1)) |> Vapor.JSON.encode()
    pad = rem(8 - rem(byte_size(json), 8), 8)
    json = json <> String.duplicate(" ", pad)
    <<byte_size(json)::64-little, json::binary>>
  end

  @doc false
  # {dtype name, bytes} of a tensor as stored under `as`
  def stored_as(t, as), do: stored(t, as)

  @doc """
  Every tensor of a checkpoint directory (single `model.safetensors` or
  sharded with its index) without reading data: `{:ok, %{name => {file,
  data_start, entry}}}`, for `read_entry/3`.
  """
  def catalog(dir) do
    single = Path.join(dir, "model.safetensors")
    index = Path.join(dir, "model.safetensors.index.json")

    files =
      cond do
        File.regular?(single) -> {:ok, [single]}
        File.regular?(index) ->
          with {:ok, bin} <- File.read(index), {:ok, %{"weight_map" => m}} when is_map(m) <- Vapor.JSON.decode(bin) do
            {:ok, m |> Map.values() |> Enum.uniq() |> Enum.sort() |> Enum.map(&Path.join(dir, Path.basename(&1)))}
          else
            _ -> reject(index, "an index with a weight_map object", "re-export the checkpoint")
          end
        true -> reject(dir, "model.safetensors, single or sharded", "download the safetensors weights")
      end

    with {:ok, fs} <- files do
      Enum.reduce_while(fs, {:ok, %{}}, fn f, {:ok, acc} ->
        case index(f) do
          {:ok, %{entries: es, data_start: ds}} -> {:cont, {:ok, Enum.reduce(es, acc, &Map.put(&2, &1.name, {f, ds, &1}))}}
          err -> {:halt, err}
        end
      end)
    end
  end

  @doc "Read one catalogued tensor (as `read/2` would): `{:ok, tensor}` or a rejection."
  def read_entry(file, ds, %{name: name, dtype: dt, shape: shape, offsets: {b, e}}) do
    shape = if shape == [], do: [1], else: shape

    with :ok <- check(dt in @readable, name, "dtype ∈ #{inspect(@readable)}, got #{dt}"),
         {:ok, f} <- File.open(file, [:read, :binary]) do
      try do
        with {:ok, raw} <- pread(f, ds + b, e - b, file, ds + e) do
          case tensor(dt, shape, raw) do
            {:error, bound} -> reject(name, bound, "convert the tensor before export")
            t -> {:ok, t}
          end
        end
      after
        File.close(f)
      end
    end
  end

  defp stored(%Tensor{dtype: :f32, data: d}, "BF16"), do: {"BF16", f32_to_bf16(d)}
  defp stored(%Tensor{dtype: :f32, data: d}, "F16"), do: {"F16", f32_to_f16(d)}
  defp stored(%Tensor{dtype: :bf16} = t, "F32"), do: {"F32", Tensor.widen(t).data}
  defp stored(%Tensor{dtype: :bf16, data: d}, "BF16"), do: {"BF16", d}
  defp stored(%Tensor{dtype: :bf16} = t, "F16"), do: stored(Tensor.widen(t), "F16")
  defp stored(%Tensor{dtype: dt, data: d}, _), do: {dtype_name(dt), d}

  @doc """
  The Hugging Face sharded form: tensors grouped in name order into files
  of at most `max_bytes` (a tensor larger than that is a shard of its own),
  `model-0000k-of-0000n.safetensors`, plus `model.safetensors.index.json`
  (`metadata.total_size`, `weight_map`). A single shard is written as plain
  `model.safetensors`. Options as `write/4`. Returns the file names.
  """
  def write_sharded(dir, tensors, max_bytes, opts \\ []) do
    meta = Keyword.get(opts, :metadata, %{"format" => "pt"})
    size = fn {name, t} -> byte_size(elem(stored(t, as_fun(opts).(name)), 1)) end
    sorted = Enum.sort_by(tensors, &elem(&1, 0))
    shards = shard_plan(sorted, size, max_bytes)

    File.mkdir_p!(dir)
    n = length(shards)

    if n <= 1 do
      :ok = write(Path.join(dir, "model.safetensors"), Map.new(List.flatten(shards)), meta, opts)
      {:ok, ["model.safetensors"]}
    else
      names = for k <- 1..n, do: "model-#{pad5(k)}-of-#{pad5(n)}.safetensors"

      for {file, shard} <- Enum.zip(names, shards), do: :ok = write(Path.join(dir, file), Map.new(shard), meta, opts)

      index = %{"metadata" => %{"total_size" => tensors |> Enum.map(size) |> Enum.sum()},
                "weight_map" => for({file, shard} <- Enum.zip(names, shards), {name, _} <- shard, into: %{}, do: {name, file})}

      :ok = File.write(Path.join(dir, "model.safetensors.index.json"), Vapor.JSON.encode(index))
      {:ok, names ++ ["model.safetensors.index.json"]}
    end
  end

  @doc false
  # items in name order cut into files of at most max_bytes (a bigger item alone)
  def shard_plan(sorted, size, max_bytes) do
    Enum.chunk_while(sorted, {[], 0}, fn kv, {acc, n} ->
      b = size.(kv)
      if acc != [] and n + b > max_bytes, do: {:cont, Enum.reverse(acc), {[kv], b}}, else: {:cont, {[kv | acc], n + b}}
    end, fn {[], _} -> {:cont, {[], 0}}; {acc, _} -> {:cont, Enum.reverse(acc), {[], 0}} end)
  end

  defp as_fun(opts) do
    case Keyword.get(opts, :as, "F32") do
      f when is_function(f, 1) -> f
      dt -> fn _ -> dt end
    end
  end

  defp pad5(k), do: k |> Integer.to_string() |> String.pad_leading(5, "0")

  # ---------------------------------------------------------------- header --

  defp entries(header, data_len, path) when is_map(header) do
    header
    |> Map.delete("__metadata__")
    |> Enum.reduce_while({:ok, []}, fn {name, spec}, {:ok, acc} ->
      case entry(name, spec) do
        {:ok, e} -> {:cont, {:ok, [e | acc]}}
        err -> {:halt, err}
      end
    end)
    |> case do
      {:ok, es} -> tiling(Enum.sort_by(es, & &1.offsets), data_len, path)
      err -> err
    end
  end

  defp entries(_header, _len, path), do: reject(path, "header is a JSON object", "re-export the file")

  defp entry(name, %{"dtype" => dt, "shape" => shape, "data_offsets" => [b, e]})
       when is_binary(dt) and is_list(shape) and is_integer(b) and is_integer(e) do
    with :ok <- check(Map.has_key?(@bits, dt), name, "known dtype, got #{inspect(dt)}"),
         :ok <- check(Enum.all?(shape, &(is_integer(&1) and &1 >= 0)), name, "shape of non-negative integers"),
         :ok <- check(b >= 0 and e >= b, name, "0 ≤ begin ≤ end"),
         bits = Enum.product(shape) * @bits[dt],
         :ok <- check(rem(bits, 8) == 0, name, "a whole number of bytes (#{bits} bits of #{dt})"),
         :ok <- check(e - b == div(bits, 8), name,
                      "end − begin = numel·sizeof(#{dt}) = #{div(bits, 8)}, got #{e - b}") do
      {:ok, %{name: name, dtype: dt, shape: shape, offsets: {b, e}}}
    end
  end

  defp entry(name, _), do: reject(name, "entry {dtype, shape, data_offsets: [begin, end]}", "re-export the file")

  # sorted by offset, the tensors must cover [0, data_len) exactly
  defp tiling(entries, data_len, path) do
    Enum.reduce_while(entries, {:ok, 0}, fn %{name: n, offsets: {b, e}}, {:ok, at} ->
      cond do
        b < at -> {:halt, reject(n, "tensors do not overlap (starts at #{b}, previous ends at #{at})", "re-export the file")}
        b > at -> {:halt, reject(n, "no hole in the data section (gap #{at}…#{b})", "re-export the file")}
        true -> {:cont, {:ok, e}}
      end
    end)
    |> case do
      {:ok, ^data_len} -> {:ok, entries}
      {:ok, end_} -> reject(path, "tensors cover the data section (#{end_} of #{data_len} bytes)", "re-export the file")
      err -> err
    end
  end

  # ----------------------------------------------------------------- data --

  defp tensor("F32", shape, raw), do: Tensor.new(:f32, shape, raw)
  defp tensor("I32", shape, raw), do: Tensor.new(:s32, shape, raw)
  defp tensor("I8", shape, raw), do: Tensor.new(:s8, shape, raw)
  defp tensor("U8", shape, raw), do: Tensor.new(:u8, shape, raw)
  defp tensor("BF16", shape, raw), do: Tensor.new(:f32, shape, bf16_to_f32(raw))
  defp tensor("F16", shape, raw), do: Tensor.new(:f32, shape, f16_to_f32(raw))
  defp tensor("F64", shape, raw), do: Tensor.new(:f32, shape, f64_to_f32(raw))
  defp tensor("F8_" <> _ = dt, shape, raw), do: Tensor.new(:f32, shape, f8_to_f32(dt, raw))
  defp tensor("I16", shape, raw), do: Tensor.new(:s32, shape, for(<<x::signed-16-little <- raw>>, into: <<>>, do: <<x::32-little>>))
  defp tensor("U16", shape, raw), do: Tensor.new(:s32, shape, for(<<x::16-little <- raw>>, into: <<>>, do: <<x::32-little>>))

  defp tensor("BOOL", shape, raw) do
    if Enum.all?(:binary.bin_to_list(raw), &(&1 <= 1)),
      do: Tensor.new(:s32, shape, for(<<x <- raw>>, into: <<>>, do: <<x::32-little>>)),
      else: {:error, "BOOL bytes are 0 or 1"}
  end

  defp tensor(dt, shape, raw) when dt in ["I64", "U32", "U64"] do
    {w, signed} = %{"I64" => {64, true}, "U32" => {32, false}, "U64" => {64, false}}[dt]
    xs = if signed, do: for(<<x::signed-little-size(w) <- raw>>, do: x), else: for(<<x::little-size(w) <- raw>>, do: x)

    if Enum.all?(xs, &(&1 >= -2_147_483_648 and &1 <= 2_147_483_647)),
      do: Tensor.new(:s32, shape, for(x <- xs, into: <<>>, do: <<x::32-little>>)),
      else: {:error, "#{dt} values within s32 (vapor's integer type)"}
  end

  @doc """
  IEEE binary64 → binary32, round to nearest-even (overflow to ±∞,
  NaN kept quiet with the payload's high bits), as numpy's `astype`.
  """
  def f64_to_f32(raw) do
    for <<b::64-little <- raw>>, into: <<>> do
      case <<b::64>> do
        <<x::float-64>> -> <<x::float-32-little>>
        _ -> <<f64_special(b)::32-little>>
      end
    end
  end

  defp f64_special(b) do
    s = (b >>> 32) &&& 0x8000_0000
    m = b &&& 0xF_FFFF_FFFF_FFFF
    if m == 0, do: s ||| 0x7F80_0000, else: s ||| 0x7FC0_0000 ||| (m >>> 29)
  end

  # value of an 8-bit float pattern as binary32 bits: {exponent bits,
  # mantissa bits, bias, special rule}
  @f8 %{"F8_E4M3" => {4, 3, 7, :fn}, "F8_E4M3FNUZ" => {4, 3, 8, :fnuz}, "F8_E5M2" => {5, 2, 15, :ieee},
        "F8_E5M2FNUZ" => {5, 2, 16, :fnuz}}

  @doc false
  # NaN patterns as torch widens them: E4M3FN keeps the mantissa bits under
  # an all-ones exponent; FNUZ and E8M0 give 0x7F800001
  def f8_bits("F8_E8M0", x), do: if(x == 0xFF, do: 0x7F80_0001, else: if(x == 0, do: 0x0040_0000, else: x <<< 23))

  def f8_bits(dt, x) do
    {eb, mb, bias, rule} = @f8[dt]
    s = (x &&& 0x80) <<< 24
    e = x >>> mb &&& (1 <<< eb) - 1
    m = x &&& (1 <<< mb) - 1
    emax = (1 <<< eb) - 1

    cond do
      rule == :fnuz and x == 0x80 -> 0x7F80_0001
      rule == :fn and e == emax and m == (1 <<< mb) - 1 -> s ||| 0x7F80_0000 ||| m <<< (23 - mb)
      rule == :ieee and e == emax and m == 0 -> s ||| 0x7F80_0000
      rule == :ieee and e == emax -> s ||| 0x7FC0_0000 ||| m <<< (23 - mb)
      e == 0 and m == 0 -> s
      e == 0 -> small(s, m, mb, 1 - bias)
      true -> s ||| (e - bias + 127) <<< 23 ||| m <<< (23 - mb)
    end
  end

  # subnormal m·2^(e−mb): normalise the leading one
  defp small(s, m, mb, e) when m < 1 <<< mb, do: small(s, m <<< 1, mb, e - 1)
  defp small(s, m, mb, e), do: s ||| (e + 127) <<< 23 ||| (m &&& (1 <<< mb) - 1) <<< (23 - mb)

  defp f8_to_f32(dt, raw) do
    t = table({:f8, dt}, fn -> for x <- 0..255, into: <<>>, do: <<f8_bits(dt, x)::32-little>> end)
    for <<x <- raw>>, into: <<>>, do: binary_part(t, 4 * x, 4)
  end

  @doc "bfloat16 → binary32, exact: the 16 high bits of the binary32 pattern."
  def bf16_to_f32(raw), do: for(<<lo, hi <- raw>>, into: <<>>, do: <<0, 0, lo, hi>>)

  @doc "IEEE binary16 → binary32, exact (subnormals normalised, ±∞ and NaN kept)."
  def f16_to_f32(raw) do
    # all 65536 widenings (256 KiB), built once per node
    t = table(:f16, fn -> for h <- 0..0xFFFF, into: <<>>, do: <<f16_bits(h)::32-little>> end)
    for <<h::16-little <- raw>>, into: <<>>, do: binary_part(t, 4 * h, 4)
  end

  # a widening table, computed once and kept in persistent_term
  defp table(key, build) do
    case :persistent_term.get({__MODULE__, key}, nil) do
      nil -> (t = build.(); :persistent_term.put({__MODULE__, key}, t); t)
      t -> t
    end
  end

  @doc "binary32 → bfloat16, round to nearest-even (every NaN → 0x7FC0), as torch."
  def f32_to_bf16(raw) do
    for <<b::32-little <- raw>>, into: <<>> do
      h =
        if (b &&& 0x7FFF_FFFF) > 0x7F80_0000,
          do: 0x7FC0,
          else: (b + 0x7FFF + (b >>> 16 &&& 1)) >>> 16

      <<h::16-little>>
    end
  end

  @doc "binary32 → IEEE binary16, round to nearest-even (overflow to ±∞, NaN → sign·0x7E00), as torch."
  def f32_to_f16(raw), do: for(<<b::32-little <- raw>>, into: <<>>, do: <<f16_of(b)::16-little>>)

  defp f16_of(b) do
    s = (b >>> 16) &&& 0x8000
    e = b >>> 23 &&& 0xFF
    m = b &&& 0x7F_FFFF

    cond do
      e == 0xFF and m != 0 -> s ||| 0x7E00
      e == 0xFF -> s ||| 0x7C00
      true ->
        # the exact value is M·2^(E) with M the 24-bit significand; rescale
        # to units of the target's last place and round half to even
        {sig, ex} = if e == 0, do: {m, -149}, else: {m ||| 0x80_0000, e - 150}
        unit = if e - 127 < -14, do: -24, else: e - 127 - 10
        shift = unit - ex
        q = rne(sig, shift)
        # q is the magnitude in units of 2^unit; renormalise into fields
        cond do
          unit == -24 -> s ||| q
          q >= 0x800 -> pack(s, unit + 1, q >>> 1)
          true -> pack(s, unit, q)
        end
    end
  end

  defp rne(sig, shift) when shift <= 0, do: sig <<< -shift

  defp rne(sig, shift) do
    q = sig >>> shift
    r = sig &&& (1 <<< shift) - 1
    half = 1 <<< (shift - 1)
    if r > half or (r == half and (q &&& 1) == 1), do: q + 1, else: q
  end

  # q·2^unit with 0x400 ≤ q < 0x800 (or q == 0x400 after a carry): exponent field
  defp pack(s, unit, q) do
    e = unit + 10 + 15
    if e >= 31, do: s ||| 0x7C00, else: s ||| e <<< 10 ||| (q &&& 0x3FF)
  end

  def f16_bits(h) do
    s = (h &&& 0x8000) <<< 16
    e = h >>> 10 &&& 0x1F
    m = h &&& 0x3FF

    cond do
      e == 0 and m == 0 -> s
      e == 0 -> subnormal(s, m, -14)
      e == 31 -> s ||| 0x7F80_0000 ||| m <<< 13
      true -> s ||| (e - 15 + 127) <<< 23 ||| m <<< 13
    end
  end

  # value m·2⁻²⁴: shift the leading one into the implicit position
  defp subnormal(s, m, e) when m < 0x400, do: subnormal(s, m <<< 1, e - 1)
  defp subnormal(s, m, e), do: s ||| (e + 127) <<< 23 ||| (m &&& 0x3FF) <<< 13

  defp dtype_name(:f32), do: "F32"
  defp dtype_name(:s32), do: "I32"
  defp dtype_name(:s8), do: "I8"
  defp dtype_name(:u8), do: "U8"

  # --------------------------------------------------------------- helpers --

  defp file_size(path) do
    case File.stat(path) do
      {:ok, %{size: s}} when s >= 8 -> {:ok, s}
      {:ok, _} -> reject(path, "file of at least 8 bytes", "check the file")
      {:error, why} -> reject(path, "readable file (#{inspect(why)})", "check the path")
    end
  end

  defp pread(_f, _off, 0, _path, _limit), do: {:ok, <<>>}

  defp pread(f, off, n, path, _limit) do
    case :file.pread(f, off, n) do
      {:ok, bin} when byte_size(bin) == n -> {:ok, bin}
      _ -> reject(path, "#{n} bytes at offset #{off}", "the file is truncated")
    end
  end

  defp json(bin, path) do
    case Vapor.JSON.decode(bin) do
      {:ok, v} -> {:ok, v}
      {:error, {pos, why}} -> reject(path, "header is valid JSON (byte #{pos}: #{why})", "re-export the file")
    end
  end

  defp check(true, _node, _bound), do: :ok
  defp check(false, node, bound), do: reject(node, bound, "re-export the file")

  defp reject(node, bound, repair), do: {:error, Rejection.new({:safetensors, node}, bound, repair)}
end
