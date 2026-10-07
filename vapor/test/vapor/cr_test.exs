defmodule Vapor.CRTest do
  @moduledoc """
  Correctly rounded elementary functions (`Vapor.CR`) and the constants
  built from them: the RoPE tables of a certified program are a function of
  the configuration alone — the same bits on every host, whatever its libm.
  """
  use ExUnit.Case, async: true
  alias Vapor.{CR, F32}
  alias Vapor.Model.{Config, Llama}
  import Vapor.TestHelpers

  defp f32(x), do: F32.to_float(F32.from_float(x))

  test "exact cases and symmetries" do
    assert CR.cos_f32(0.0) == 1.0 and CR.sin_f32(0.0) == 0.0
    assert CR.pow_f32(10_000.0, 0.0) == 1.0 and CR.pow_f32(10_000.0, 1.0) == 10_000.0
    assert CR.pow_f32(10_000.0, 0.5) == 100.0
    assert CR.pow_f64(16.0, -0.5) == 0.25
    assert CR.log_f64(1.0) == 0.0
    assert CR.exp_f64(0.0) == 1.0

    :rand.seed(:exsss, {3, 1, 4})

    for _ <- 1..500 do
      x = f32(:rand.uniform() * 4096)
      assert CR.sin_f32(-x) == -CR.sin_f32(x)
      assert CR.cos_f32(-x) == CR.cos_f32(x)
      # results are binary32 values
      assert f32(CR.cos_f32(x)) == CR.cos_f32(x)
    end
  end

  test "the binary64 fast path never decides differently from the exact integer path" do
    :rand.seed(:exsss, {2, 7, 1})

    for _ <- 1..1500 do
      x = f32(:rand.uniform() * :math.pow(2, :rand.uniform(36) - 16))
      assert CR.cos_f32(x) == CR.trig_exact(:cos, x, 24), "cos #{x}"
      assert CR.sin_f32(x) == CR.trig_exact(:sin, x, 24), "sin #{x}"
    end
  end

  # pinned digests: tables computed by the definition, not by a host's libm
  test "RoPE tables are host-independent constants (pinned SHA-256)" do
    digest = fn {c, s} -> Base.encode16(:crypto.hash(:sha256, c.data <> s.data), case: :lower) end

    c = %Config{arch: "llama", vocab: 16, hidden: 64, intermediate: 16, layers: 1, heads: 1, kv_heads: 1, head_dim: 64,
                eps: 1.0e-5, rope_theta: 10_000.0, max_pos: 512, tie: true}

    assert digest.(Llama.rope_tables(c, 512)) == "dfcd0bc0381604390ce99887809f290d7526b6a9b6ceb7c9205ff06b2db188a7"

    c3 = %{c | rope_theta: 500_000.0, rope_scaling: {:llama3, 8.0, 1.0, 4.0, 128}}
    assert digest.(Llama.rope_tables(c3, 512)) == "ce7c404fcad7d06ba887dc7748ac726021712cca02a79179f2b5907c5979bed5"

    {:ok, cy} =
      Config.from_map(%{"model_type" => "qwen3", "vocab_size" => 16, "hidden_size" => 64, "intermediate_size" => 16,
                        "num_hidden_layers" => 1, "num_attention_heads" => 1, "head_dim" => 64, "max_position_embeddings" => 2048,
                        "rope_parameters" => %{"rope_type" => "yarn", "factor" => 4.0, "original_max_position_embeddings" => 512,
                                               "rope_theta" => 1.0e6}})

    assert digest.(Llama.rope_tables(cy, 2048)) == "4f7eb44370260b986e67da202f3090fb36ffc7b0c0b2a33fb0016a17bde94f5b"
  end

  @tag :mpmath
  test "correct rounding against mpmath at 300 bits: cos, sin, pow (binary32), log, exp (binary64)" do
    :rand.seed(:exsss, {9, 9, 9})
    angles = for _ <- 1..4000, do: f32(:rand.uniform() * :math.pow(2, :rand.uniform(40) - 20))
    pows = for b <- [10_000.0, 500_000.0, 1.0e6, 3.0], i <- 0..63, do: {b, f32(f32(2.0 * i) / 128)}
    logs = [0.5, 2.0, 3.0, 10.0, 1.0e-7, 12.789, 40.0, 1.0000001, 0.707, 4.0]

    lines =
      Enum.map(angles, &"T #{fmt(&1)} #{fmt(CR.cos_f32(&1))} #{fmt(CR.sin_f32(&1))}") ++
        Enum.map(pows, fn {b, e} -> "P #{fmt(b)} #{fmt(e)} #{fmt(CR.pow_f32(b, e))}" end) ++
        Enum.map(logs, &"L #{fmt(&1)} #{fmt(CR.log_f64(&1))} #{fmt(CR.exp_f64(&1 / 7))} #{fmt(&1 / 7)}")

    out =
      py!("""
      import sys, mpmath as mp
      mp.mp.prec = 300
      def r(v, p):
          with mp.workprec(p):
              return +v
      bad = []
      for line in sys.stdin:
          k, *xs = line.split()
          xs = [mp.mpf(float(x)) for x in xs]
          if k == "T":
              a, c, s = xs
              if c != r(mp.cos(a), 24) or s != r(mp.sin(a), 24): bad.append(line)
          elif k == "P":
              b, e, y = xs
              if y != r(mp.power(b, e), 24): bad.append(line)
          else:
              x, l, e, x7 = xs
              if l != r(mp.log(x), 53) or e != r(mp.exp(x7), 53): bad.append(line)
      print(len(bad)); print("".join(bad[:5]))
      """, [], Enum.join(lines, "\n") <> "\n")

    assert String.starts_with?(out, "0\n"), out
  end

  # transformers computes cos/sin in PyTorch's binary32 (not correctly
  # rounded): the tables agree to within an ulp, mostly exactly — two ulp
  # with YaRN, whose attention factor multiplies an already rounded cos
  @tag :torch
  test "RoPE tables (default, YaRN) agree with transformers' rotary embeddings to an ulp" do
    for {name, rope, bound} <- [{"default", %{"rope_type" => "default", "rope_theta" => 1.0e4}, 1},
                                {"yarn", %{"rope_type" => "yarn", "factor" => 4.0, "original_max_position_embeddings" => 128, "rope_theta" => 1.0e6}, 2}] do
      cfg = %{"model_type" => "qwen3", "vocab_size" => 16, "hidden_size" => 64, "intermediate_size" => 16, "num_hidden_layers" => 1,
              "num_attention_heads" => 1, "num_key_value_heads" => 1, "head_dim" => 64, "max_position_embeddings" => 512,
              "rope_parameters" => rope}

      {:ok, c} = Config.from_map(cfg)
      {cos, sin} = Llama.rope_tables(c, 512)

      out =
        py!("""
        import sys, json, torch
        from transformers import AutoConfig
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        cfg = json.loads(sys.argv[1]); mt = cfg.pop("model_type")
        rot = Qwen3RotaryEmbedding(AutoConfig.for_model(mt, **cfg))
        pos = torch.arange(512)[None]
        cos, sin = rot(torch.zeros(1, 512, 64), pos)
        half = cos.shape[-1] // 2
        print((cos[0, :, :half].contiguous().numpy().tobytes() + sin[0, :, :half].contiguous().numpy().tobytes()).hex())
        """, [Vapor.JSON.encode(cfg)])

      theirs = out |> String.trim() |> Base.decode16!(case: :lower)
      ours = cos.data <> sin.data
      ulps = for {<<a::32-little>>, <<b::32-little>>} <- Enum.zip(chunks(ours), chunks(theirs)), do: abs(key(a) - key(b))
      assert Enum.max(ulps) <= bound, "#{name}: worst #{Enum.max(ulps)} ulp"
      IO.puts("\n  RoPE #{name}: #{Float.round(100 * Enum.count(ulps, &(&1 == 0)) / length(ulps), 2)}% of entries bit-equal to transformers")
      exact = Enum.count(ulps, &(&1 == 0)) / length(ulps)
      assert exact > 0.9, "#{name}: only #{exact} exactly equal"
    end
  end

  defp chunks(bin), do: for(<<w::binary-4 <- bin>>, do: w)
  defp key(b), do: if(Bitwise.band(b, 0x8000_0000) != 0, do: -Bitwise.band(b, 0x7FFF_FFFF), else: b)
  defp fmt(x), do: :erlang.float_to_binary(x * 1.0, [:short])
end
