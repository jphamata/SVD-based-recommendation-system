defmodule Vapor.IngestTest do
  @moduledoc """
  Phase P2, the airlocks: RFC 8259 JSON, safetensors and `config.json`.
  Everything read from a file is checked before it is used, and every
  refusal names what was violated.
  """
  use ExUnit.Case, async: true
  import Bitwise
  alias Vapor.{JSON, Rejection, Tensor}
  alias Vapor.Ingest.Safetensors, as: ST
  alias Vapor.Model.Config
  import Vapor.TestHelpers

  @tmp Path.join(System.tmp_dir!(), "vapor-ingest-test")

  setup_all do
    File.mkdir_p!(@tmp)
    on_exit(fn -> File.rm_rf!(@tmp) end)
  end

  # ------------------------------------------------------------------ JSON --

  @valid [
    ~s({"a":[1,-0,2.5e-3,1E+2,true,false,null],"b":{"c":"\\u00e9\\ud83d\\ude00\\n"}}),
    ~s( [ ] ),
    ~s("\\"\\\\\\/\\b\\f\\n\\r\\t"),
    ~s(-0.0),
    ~s(123456789012345678901234567890),
    ~s(1.7976931348623157e308),
    ~s(5e-324),
    ~s({"é":"ü","":0}),
    ~s([[[[[[[[[[[]]]]]]]]]]])
  ]

  @invalid [
    {~s([1,]), "trailing comma in an array"},
    {~s({"a":1,}), "trailing comma in an object"},
    {~s({"a":1,"a":2}), "duplicate key"},
    {~s(01), "leading zero"},
    {~s(1.), "empty fraction"},
    {~s(.5), "no integer part"},
    {~s(1e), "empty exponent"},
    {~s(+1), "plus sign"},
    {~s("\\ud800"), "unpaired high surrogate"},
    {~s("\\udc00"), "unpaired low surrogate"},
    {~s("a\tb"), "raw control character"},
    {~s("\\x"), "invalid escape"},
    {~s(NaN), "NaN is not JSON"},
    {~s(1e400), "overflow"},
    {~s([1] 2), "trailing data"},
    {~s({"a" 1}), "missing colon"},
    {~s("abc), "unterminated string"},
    {String.duplicate("[", 600) <> String.duplicate("]", 600), "nesting depth"}
  ]

  test "JSON: valid documents decode; invalid ones are refused with a byte position" do
    for doc <- @valid, do: assert({:ok, _} = JSON.decode(doc), doc)

    assert JSON.decode!(~s({"a":[1,2.5,"\\u00e9\\ud83d\\ude00"],"b":null})) ==
             %{"a" => [1, 2.5, "é😀"], "b" => nil}

    for {doc, why} <- @invalid do
      assert {:error, {pos, msg}} = JSON.decode(doc), why
      assert is_integer(pos) and pos >= 0 and is_binary(msg)
    end
  end

  test "JSON: encode ∘ decode is the identity on random documents (keys sorted, deterministic)" do
    :rand.seed(:exsss, {3, 1, 4})

    for _ <- 1..300 do
      v = random_json(4)
      text = JSON.encode(v)
      assert JSON.decode!(text) == v
      assert JSON.encode(JSON.decode!(text)) == text
    end
  end

  @tag :python
  test "JSON: same values as Python's json module on valid documents, same refusals on invalid ones" do
    :rand.seed(:exsss, {2, 7, 1})
    docs = @valid ++ Enum.map(1..200, fn _ -> JSON.encode(random_json(4)) end)

    # Python re-serialises each document canonically; both parses must agree
    script = """
    import json, sys
    def strict(s):
        return json.loads(s, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)),
                          object_pairs_hook=lambda ps: dict(ps) if len(set(k for k, _ in ps)) == len(ps) else (_ for _ in ()).throw(ValueError("dup")))
    for line in sys.stdin.read().split("\\x00")[:-1]:
        try:
            v = strict(line)
            if isinstance(v, float) and v != v or v in (float("inf"), float("-inf")): raise ValueError("nonfinite")
            print("ok " + json.dumps(v, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
        except (ValueError, RecursionError) as e:
            print("err")
    """

    input = Enum.map_join(docs ++ Enum.map(@invalid, &elem(&1, 0)), &(&1 <> <<0>>))
    lines = py!(script, [], input) |> String.split("\n", trim: true)

    {oks, errs} = Enum.split(lines, length(docs))

    for {doc, "ok " <> canon} <- Enum.zip(docs, oks) do
      assert JSON.decode!(doc) == JSON.decode!(canon), doc
    end

    assert length(oks) == length(docs) and Enum.all?(oks, &String.starts_with?(&1, "ok "))
    # Python refuses what we refuse (1e400 → inf, dup via the hook) — except
    # lone surrogates, which RFC 8259 §8.2 leaves unpredictable: Python keeps
    # them in a str, we refuse (an Elixir string is valid UTF-8), and our
    # depth bound, which is deliberately below Python's recursion limit
    for {{doc, why}, got} <- Enum.zip(@invalid, errs), not String.contains?(why, ["surrogate", "nesting"]) do
      assert got == "err", "python accepted #{inspect(doc)} (#{why})"
    end
  end

  defp random_json(0), do: random_scalar()

  defp random_json(d) do
    case :rand.uniform(4) do
      1 -> for _ <- 1..:rand.uniform(4), do: random_json(d - 1)
      2 -> Map.new(1..:rand.uniform(4), fn _ -> {random_string(), random_json(d - 1)} end)
      _ -> random_scalar()
    end
  end

  defp random_scalar do
    case :rand.uniform(7) do
      1 -> :rand.uniform(1 <<< 70) - (1 <<< 69)
      2 -> (:rand.uniform() - 0.5) * :math.pow(10, :rand.uniform(80) - 40)
      3 -> random_string()
      4 -> true
      5 -> false
      6 -> nil
      7 -> 0.0
    end
  end

  defp random_string do
    alphabet = [?a, ?Z, ?0, ?\s, ?", ?\\, ?/, ?\n, ?\t, 0x01, 0x7F, ?é, ?€, 0x1F600, 0xFFFF]
    for _ <- 1..:rand.uniform(6), into: "", do: <<Enum.random(alphabet)::utf8>>
  end

  # ----------------------------------------------------------- safetensors --

  test "safetensors: write/read round trip, 8-byte aligned data, subset reads" do
    ts = %{"b.weight" => Tensor.random(:f32, [3, 5], 1), "a" => Tensor.from_list(:s32, [4], [1, -2, 3, 1 <<< 30]),
           "q" => Tensor.random(:s8, [7], 2), "u" => Tensor.random(:u8, [2, 2], 3)}
    path = Path.join(@tmp, "rt.safetensors")
    :ok = ST.write(path, ts, %{"format" => "pt"})

    assert {:ok, ^ts} = ST.read(path)
    assert {:ok, %{"a" => _} = one} = ST.read(path, only: ["a"])
    assert map_size(one) == 1
    {:ok, idx} = ST.index(path)
    assert rem(idx.data_start, 8) == 0
    assert idx.metadata == %{"format" => "pt"}
    assert Enum.map(idx.entries, & &1.name) == ["a", "b.weight", "q", "u"]
  end

  test "half precision widens exactly: bf16 by a shift, f16 over all 65536 patterns" do
    # every finite f16 pattern against the BEAM's own IEEE half decoding
    for h <- 0..0xFFFF, (h >>> 10 &&& 0x1F) != 31 do
      <<x::float-16>> = <<h::16>>
      <<want::32>> = <<x::float-32>>
      assert ST.f16_bits(h) == want, "f16 #{Integer.to_string(h, 16)}"
    end

    assert ST.f16_bits(0x7C00) == 0x7F80_0000
    assert ST.f16_bits(0xFC00) == 0xFF80_0000
    assert (ST.f16_bits(0x7E00) &&& 0x7FC0_0000) == 0x7FC0_0000

    assert ST.bf16_to_f32(<<0x80, 0x3F, 0x00, 0xC0>>) == <<0, 0, 0x80, 0x3F, 0, 0, 0x00, 0xC0>>
  end

  @tag :python
  test "safetensors: f16 widening equals numpy's, pattern for pattern" do
    out = py!("""
    import numpy as np, sys
    h = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32).view(np.uint32)
    sys.stdout.write(h.tobytes().hex())
    """)

    want = Base.decode16!(out, case: :lower)
    got = ST.f16_to_f32(for h <- 0..0xFFFF, into: <<>>, do: <<h::16-little>>)
    # NaN payloads: numpy keeps them the same way (quiet bit and payload shifted)
    assert got == want
  end

  test "safetensors: malformed headers are refused, naming the tensor and the bound" do
    good = %{"dtype" => "F32", "shape" => [2], "data_offsets" => [0, 8]}

    cases = [
      {%{"t" => %{good | "data_offsets" => [0, 12]}}, <<0::64>>, "numel·sizeof"},
      {%{"t" => %{good | "dtype" => "F8"}}, <<0::64>>, "known dtype"},
      {%{"t" => %{good | "shape" => [-2]}}, <<0::64>>, "non-negative"},
      {%{"t" => good, "u" => %{good | "data_offsets" => [4, 12]}}, <<0::96>>, "overlap"},
      {%{"t" => good, "u" => %{good | "data_offsets" => [12, 20]}}, <<0::160>>, "no hole"},
      {%{"t" => good}, <<0::128>>, "cover the data section"},
      {%{"t" => %{"dtype" => "F32"}}, <<0::64>>, "entry"}
    ]

    for {header, data, expect} <- cases do
      path = Path.join(@tmp, "bad.safetensors")
      h = JSON.encode(header)
      File.write!(path, [<<byte_size(h)::64-little>>, h, data])
      assert {:error, %Rejection{bound: bound}} = ST.read(path)
      assert bound =~ expect, "#{inspect(header)}: #{bound}"
    end

    raw = fn bin -> path = Path.join(@tmp, "raw.safetensors"); File.write!(path, bin); ST.index(path) end
    assert {:error, %Rejection{bound: "file of at least 8 bytes"}} = raw.(<<1, 2, 3>>)
    assert {:error, %Rejection{bound: "header length" <> _}} = raw.(<<1_000_000::64-little, "{}">>)
    assert {:error, %Rejection{bound: "header is valid JSON" <> _}} = raw.(<<6::64-little, ~s({"a":1)>>)
    assert {:error, %Rejection{bound: "header is valid JSON" <> _}} = raw.(<<13::64-little, ~s({"a":1,"a":1})>>)
    assert {:error, %Rejection{bound: "header is a JSON object"}} = raw.(<<2::64-little, "[]">>)
  end

  @tag :torch
  test "safetensors: reads what the reference library writes (F32, BF16, F16, I32) bit for bit" do
    path = Path.join(@tmp, "ref.safetensors")

    out = py!("""
    import sys, torch
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5, 7, generator=g) * 3
    t = {"f32": x, "bf16": x.to(torch.bfloat16), "f16": x.to(torch.float16), "i32": torch.arange(-3, 9, dtype=torch.int32),
         "sub": torch.tensor([6e-8, -6e-5, 65504.0], dtype=torch.float16)}
    save_file(t, sys.argv[1], metadata={"k": "v"})
    for k in ["f32", "bf16", "f16", "sub"]:
        print(k, t[k].float().numpy().tobytes().hex())
    print("i32", t["i32"].numpy().tobytes().hex())
    """, [path])

    {:ok, ts} = ST.read(path)
    {:ok, idx} = ST.index(path)
    assert idx.metadata == %{"k" => "v"}

    for line <- String.split(out, "\n", trim: true) do
      [k, hex] = String.split(line, " ")
      assert ts[k].data == Base.decode16!(hex, case: :lower), k
    end

    assert ts["i32"].dtype == :s32 and ts["bf16"].dtype == :f32 and ts["bf16"].shape == [5, 7]
  end

  @tag :torch
  test "safetensors: every dtype torch writes reads as torch converts it (all 8-bit float patterns, F64 rounding, integers)" do
    path = Path.join(@tmp, "all.safetensors")

    out = py!("""
    import sys, torch
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(1)
    b = torch.arange(256, dtype=torch.uint8)
    f64 = torch.cat([torch.randn(4096, generator=g, dtype=torch.float64) * 10 ** torch.randint(-50, 40, (4096,), generator=g),
                     torch.tensor([0.0, -0.0, 1e300, -1e300, 3.4028235677973366e38, 1.401298464324817e-45, 7.006492321624086e-46,
                                   float("inf"), float("-inf"), 2.0 ** -149 * 1.5, 1 + 2.0 ** -24], dtype=torch.float64)])
    t = {"e4m3": b.clone().view(torch.float8_e4m3fn), "e5m2": b.clone().view(torch.float8_e5m2),
         "e4m3fnuz": b.clone().view(torch.float8_e4m3fnuz), "e5m2fnuz": b.clone().view(torch.float8_e5m2fnuz),
         "e8m0": b.clone().view(torch.float8_e8m0fnu), "f64": f64,
         "i64": torch.tensor([-2**31, 2**31 - 1, 0, -5], dtype=torch.int64), "i16": torch.arange(-300, 300, 7, dtype=torch.int16),
         "u16": torch.tensor([0, 1, 65535, 300], dtype=torch.uint16), "u32": torch.tensor([0, 2**31 - 1], dtype=torch.uint32),
         "b": torch.tensor([True, False, True])}
    save_file(t, sys.argv[1])
    for k in ["e4m3", "e5m2", "e4m3fnuz", "e5m2fnuz", "e8m0", "f64"]:
        print(k, t[k].float().numpy().tobytes().hex())
    for k in ["i64", "i16", "u16", "u32", "b"]:
        print(k, t[k].to(torch.int32).numpy().tobytes().hex())
    """, [path])

    {:ok, ts} = ST.read(path)

    for line <- String.split(out, "\n", trim: true) do
      [k, hex] = String.split(line, " ")
      assert ts[k].data == Base.decode16!(hex, case: :lower), k
    end

    assert ts["e4m3"].dtype == :f32 and ts["i64"].dtype == :s32
  end

  @tag :torch
  # NaN is compared as NaN: torch's own paths disagree on its bits (0x7FC0 vs
  # 0xFFFF), and no certified program carries one
  test "safetensors: narrowing to BF16 and F16 equals torch's (ties, subnormals, overflow; NaN stays NaN) and torch reads our files" do
    path = Path.join(@tmp, "narrow.safetensors")
    edge = [0, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0001, 0xFFA0_0000, 0x3F80_8000, 0x3F81_8000, 0x3F80_7FFF,
            0x477F_F000, 0x477F_EFFF, 0x4780_0000, 0x3880_0000, 0x3380_0000, 0x3300_0001, 0x3300_0000, 0x387F_C000,
            0x0000_0001, 0x7F7F_FFFF, 0x0080_0000, 0x3C00_1000, 0x3C00_3000]
    :rand.seed(:exsss, {9, 9, 9})
    words = edge ++ for(_ <- 1..20_000, do: :rand.uniform(1 <<< 32) - 1)
    x = Tensor.new(:f32, [length(words)], for(w <- words, into: <<>>, do: <<w::32-little>>))
    :ok = ST.write(path, %{"bf" => x, "h" => x, "f" => x}, %{}, as: fn "bf" -> "BF16"; "h" -> "F16"; _ -> "F32" end)

    out = py!("""
    import sys, torch
    from safetensors.torch import load_file
    t = load_file(sys.argv[1])
    f = t["f"]
    print(t["bf"].dtype, t["h"].dtype)
    nan = torch.isnan(f)
    for k, dt in [("bf", torch.bfloat16), ("h", torch.float16)]:
        assert torch.isnan(t[k][nan]).all()
        print(t[k][~nan].view(torch.int16).numpy().tobytes().hex(), f[~nan].to(dt).view(torch.int16).numpy().tobytes().hex())
    """, [path])

    [types, bf, h] = String.split(out, "\n", trim: true)
    assert types == "torch.bfloat16 torch.float16"
    [ours, theirs] = String.split(bf, " ")
    assert ours == theirs
    [ours, theirs] = String.split(h, " ")
    assert ours == theirs
  end

  test "safetensors: values outside vapor's types are refused by tensor; sub-byte formats are known and refused" do
    raw = fn name, dt, shape, data ->
      h = JSON.encode(%{name => %{"dtype" => dt, "shape" => shape, "data_offsets" => [0, byte_size(data)]}})
      path = Path.join(@tmp, "dt.safetensors")
      File.write!(path, [<<byte_size(h)::64-little>>, h, data])
      ST.read(path)
    end

    assert {:error, %Rejection{node: {:safetensors, "big"}, bound: "I64 values within s32" <> _}} = raw.("big", "I64", [1], <<1 <<< 40::64-little>>)
    assert {:error, %Rejection{bound: "BOOL bytes are 0 or 1"}} = raw.("b", "BOOL", [2], <<1, 2>>)
    assert {:error, %Rejection{bound: "dtype ∈" <> _}} = raw.("q", "F4", [4], <<0, 0>>)
    assert {:error, %Rejection{bound: "a whole number of bytes" <> _}} = raw.("q", "F4", [3], <<0, 0>>)
    assert {:ok, %{"q" => %Tensor{dtype: :s32}}} = raw.("q", "U64", [1], <<7::64-little>>)
  end

  test "safetensors: the sharded form round-trips through the model loader" do
    {:ok, c} = Vapor.Model.Config.from_map(Vapor.TestHelpers.tiny_config("llama"))
    w = Vapor.TestHelpers.tiny_weights(c)
    dir = Path.join(@tmp, "sharded")
    {:ok, files} = ST.write_sharded(dir, w, 20_000)
    assert "model.safetensors.index.json" in files and length(files) > 3
    assert {:ok, ^w} = Vapor.Model.weights(dir)
  end

  # ---------------------------------------------------------------- config --

  test "config: the three families map onto one decoder with explicit differences" do
    {:ok, l} = Config.from_map(tiny_config("llama", %{"attention_bias" => true}))
    {:ok, m} = Config.from_map(tiny_config("mistral", %{"sliding_window" => 4096}))
    {:ok, q} = Config.from_map(tiny_config("qwen2"))

    assert {l.qkv_bias, l.o_bias, l.tie} == {true, true, false}
    assert {m.qkv_bias, m.o_bias, m.sliding_window} == {false, false, 4096}
    assert {q.qkv_bias, q.o_bias, q.tie} == {true, false, true}
    assert {q.head_dim, q.kv_heads} == {16, 2}

    {:ok, s} =
      Config.from_map(tiny_config("llama", %{"rope_scaling" => %{"rope_type" => "llama3", "factor" => 8.0,
        "low_freq_factor" => 1.0, "high_freq_factor" => 4.0, "original_max_position_embeddings" => 8192}}))

    assert s.rope_scaling == {:llama3, 8.0, 1.0, 4.0, 8192}
    refute Config.window_binds?(m, 4096)
    assert Config.window_binds?(%{m | sliding_window: 16}, 32)
  end

  test "config: to_map is the inverse of from_map; the GGUF-only forms are refused" do
    variants = [
      tiny_config("llama", %{"attention_bias" => true, "bos_token_id" => 1, "eos_token_id" => [2, 3]}),
      tiny_config("llama", %{"rope_scaling" => %{"rope_type" => "llama3", "factor" => 8.0, "low_freq_factor" => 1.0,
                                                 "high_freq_factor" => 4.0, "original_max_position_embeddings" => 8192}}),
      tiny_config("llama", %{"rope_scaling" => %{"type" => "linear", "factor" => 2.0}, "num_key_value_heads" => 4}),
      tiny_config("mistral", %{"sliding_window" => 4096}),
      tiny_config("qwen2", %{"sliding_window" => 32_768, "use_sliding_window" => false, "rope_theta" => 1_000_000.0})
    ]

    for v <- variants do
      {:ok, c} = Config.from_map(v)
      {:ok, m} = Config.to_map(c)
      assert {:ok, ^c} = Config.from_map(m)
      # and through JSON, as written to disk
      assert {:ok, ^c} = m |> JSON.encode() |> JSON.decode() |> elem(1) |> Config.from_map()
    end

    {:ok, c} = Config.from_map(tiny_config("llama"))
    assert {:error, %Rejection{node: {:config, "rope_scaling"}}} = Config.to_map(%{c | rope_scaling: :freq_factors})
    assert {:error, %Rejection{node: {:config, "attention_bias"}}} = Config.to_map(%{c | qkv_bias: true})
  end

  test "a model written as a Hugging Face directory opens to the same configuration and weights (F32 exact, BF16 rounded)" do
    {:ok, c} = Config.from_map(tiny_config("llama", %{"attention_bias" => true}))
    w = tiny_weights(c)

    for {dtype, shard} <- [{"F32", 5_000_000_000}, {"F32", 30_000}, {"BF16", 30_000}] do
      dir = Path.join(@tmp, "hf-#{dtype}-#{shard}")
      :ok = Vapor.Model.write(dir, c, w, dtype: dtype, shard_bytes: shard)
      {:ok, m} = Vapor.Model.open(dir)
      assert m.config == c
      assert File.exists?(Path.join(dir, "model.safetensors.index.json")) == (shard < 1_000_000)

      for {k, t} <- w do
        want = if dtype == "BF16" and length(t.shape) == 2, do: %{t | data: ST.bf16_to_f32(ST.f32_to_bf16(t.data))}, else: t
        assert m.weights[k] == want, "#{dtype} #{k}"
      end
    end
  end

  test "config: unsupported architectures and features are refused by field" do
    refuse = fn over, field ->
      assert {:error, %Rejection{node: {:config, ^field}}} = Config.from_map(tiny_config("llama", over))
    end

    refuse.(%{"model_type" => "gemma2"}, "model_type")
    refuse.(%{"hidden_act" => "quick_gelu"}, "hidden_act")
    # exact GELU is a canonical function now (Vapor.Canon :gelu)
    assert {:ok, %Config{act: :gelu}} = Config.from_map(tiny_config("llama", %{"hidden_act" => "gelu"}))
    refuse.(%{"mlp_bias" => true}, "mlp_bias")
    refuse.(%{"rope_scaling" => %{"rope_type" => "dynamic", "factor" => 4.0}}, "rope_scaling")
    refuse.(%{"rope_scaling" => %{"rope_type" => "longrope", "factor" => 4.0}}, "rope_scaling")
    refuse.(%{"num_key_value_heads" => 3}, "num_key_value_heads")
    refuse.(%{"head_dim" => 24}, "head_dim")
    refuse.(%{"hidden_size" => -64}, "hidden_size")
    refuse.(%{"vocab_size" => nil}, "vocab_size")

    path = Path.join(@tmp, "config.json")
    File.write!(path, JSON.encode(tiny_config("qwen2")))
    assert {:ok, %Config{arch: "qwen2"}} = Config.load(path)
    File.write!(path, "{")
    assert {:error, %Rejection{}} = Config.load(path)
  end
end
