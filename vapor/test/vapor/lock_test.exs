defmodule Vapor.LockTest do
  @moduledoc """
  The model airlock: one owner per checkpoint, contracts checked at the
  lock, tiers (alias as data, blueprint, topology), registration at run
  time, diagnostics — and the architectural rule that the core holds no
  reference to any model family.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Lock, Program, Rejection, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Lock.{Alias, Spec}
  alias Vapor.Lock.Adapters.Encoder
  alias Vapor.Model.{Config, Llama}
  alias Vapor.Runtime.{Native, Oracle, Worker}
  import Vapor.TestHelpers

  @tmp Path.join(System.tmp_dir!(), "vapor-lock-test")

  setup do
    on_exit(fn -> for id <- ["toy", "mylm", "renamed", "loop_a", "loop_b"], do: Lock.unregister(id) end)
    :ok
  end

  defp env(c, toks) do
    n = length(toks)
    Map.merge(Llama.empty_caches(c, 16), %{tok: Tensor.from_list(:s32, [n], toks), pos: Tensor.from_list(:s32, [n], Enum.to_list(0..(n - 1)))})
  end

  # ------------------------------------------------------------ the decoder --

  test "the decoder through the lock builds the very program of the direct builder" do
    for {arch, over} <- [{"llama", %{"attention_bias" => true}}, {"mistral", %{}}, {"qwen2", %{}}] do
      map = tiny_config(arch, over)
      {:ok, c} = Config.from_map(map)
      ws = tiny_weights(c)
      {:ok, spec, ws2} = Lock.from_map(map, ws)
      assert %Spec{interface: :causal_lm, family: ^arch, vocab: 96, width: 64} = spec
      assert {:ok, p} = Lock.build(spec, ws2, max_seq: 16)
      assert {:ok, ^p} = Llama.program(c, ws, max_seq: 16)
      # an admitted configuration is as good as its spec
      assert {:ok, ^p} = Lock.build(c, ws, max_seq: 16)
    end
  end

  test "zero state comes from the program, not the model" do
    {:ok, c} = Config.from_map(tiny_config("qwen2"))
    {:ok, p} = Lock.build(c, tiny_weights(c), max_seq: 16)
    assert Lock.zero_state(p) == Llama.empty_caches(c, 16)
  end

  test "a checkpoint nobody claims is refused with its near misses and the repair" do
    {:ok, c} = Config.from_map(tiny_config("llama"))
    ws = tiny_weights(c)
    assert {:error, %Rejection{node: {:lock, "model_type \"phi9\""}, bound: bound}} = Lock.from_map(tiny_config("llama", %{"model_type" => "phi9"}), ws)
    assert bound =~ "decoder: the tensors follow the pre-norm decoder layout"
    assert {:error, %Rejection{}} = Lock.from_map(%{"model_type" => "nothing"}, %{})
  end

  test "explain: every adapter's answer, the winner, missing and unused tensors" do
    map = tiny_config("qwen2")
    {:ok, c} = Config.from_map(map)
    ws = tiny_weights(c) |> Map.delete("model.norm.weight") |> Map.put("extra.weight", Tensor.random(:f32, [2, 2], 1))
    r = Lock.explain(%{source: :memory, config: map}, ws)
    assert Enum.any?(r.claims, &(&1.adapter == "decoder" and &1.answer == {:claim, 100}))
    assert {:ok, %{family: "qwen2"}} = r.admitted
    assert [{"model.norm.weight", [64], nil}] = r.tensors.missing
    assert r.tensors.unused == ["extra.weight"]
  end

  # ------------------------------------------------------------- contracts --

  defmodule Broken do
    @moduledoc false
    @behaviour Vapor.Lock.Adapter
    def id, do: "toy"
    def claim(%{config: %{"model_type" => "toy"}}), do: {:claim, 100}
    def claim(_), do: :no
    def admit(_m, ws, _o), do: {:ok, spec(:toy), ws}
    def owns?(:toy), do: true
    def owns?(_), do: false
    def spec(:toy), do: %Spec{adapter: __MODULE__, family: "toy", interface: :causal_lm, config: :toy, vocab: 32, width: 16}
    # logits of the wrong width, and no pos input
    def build(_s, _w, _o), do: {:ok, Program.new(logits: T.linear(T.gather_row(T.const(Tensor.random(:f32, [32, 16], 1)), T.input(:tok, :s32, [4])), T.const(Tensor.random(:f32, [48, 16], 2))))}
  end

  test "a program that breaks its declared contract is stopped at the lock" do
    assert {:ok, "toy"} = Lock.register(Broken)
    {:ok, spec, _} = Lock.from_map(%{"model_type" => "toy"}, %{})
    assert {:error, %Rejection{node: {:contract, :pos}}} = Lock.build(spec, %{})
  end

  test "registration: modules are checked, registered adapters come first, unregister restores" do
    assert {:error, %Rejection{}} = Lock.register(Enum)
    assert {:error, %Rejection{node: {:alias, "like"}}} = Lock.register(%{"id" => "x", "model_type" => "x"})

    # an override of a built-in family, by data: llama checkpoints now report family "mylm"
    assert {:ok, "mylm"} = Lock.register(%{"id" => "mylm", "model_type" => "mistral", "like" => "llama"})
    {:ok, c} = Config.from_map(tiny_config("mistral"))
    {:ok, spec, _} = Lock.from_map(tiny_config("mistral"), tiny_weights(c))
    assert spec.family == "mylm" and spec.lineage == ["mylm", "llama"]
    Lock.unregister("mylm")
    {:ok, spec, _} = Lock.from_map(tiny_config("mistral"), tiny_weights(c))
    assert spec.family == "mistral"
  end

  test "alias loops are refused, not followed forever" do
    {:ok, _} = Lock.register(%{"id" => "loop_a", "model_type" => "aaa", "like" => "bbb"})
    {:ok, _} = Lock.register(%{"id" => "loop_b", "model_type" => "bbb", "like" => "aaa"})
    assert {:error, %Rejection{node: {:alias, _}}} = Lock.from_map(%{"model_type" => "aaa"}, %{})
  end

  # ------------------------------------------------------- tier 1: data alias --

  # a Phi-3 checkpoint made from Llama weights: q/k/v and gate/up fused as HF does
  defp phi3(over \\ %{}) do
    map = tiny_config("llama", %{"num_key_value_heads" => 2})
    {:ok, c} = Config.from_map(map)
    ws = tiny_weights(c, 3)

    fused =
      Enum.reduce(0..(c.layers - 1), ws, fn l, acc ->
        p = "model.layers.#{l}."
        cat = fn names -> names |> Enum.map(&acc[p <> &1]) |> then(fn [t | _] = ts -> Tensor.new(:f32, [Enum.sum(Enum.map(ts, &hd(&1.shape))), List.last(t.shape)], Enum.map_join(ts, & &1.data)) end) end
        qkv = cat.(~w(self_attn.q_proj.weight self_attn.k_proj.weight self_attn.v_proj.weight))
        gu = cat.(~w(mlp.gate_proj.weight mlp.up_proj.weight))

        acc
        |> Map.drop(Enum.map(~w(self_attn.q_proj.weight self_attn.k_proj.weight self_attn.v_proj.weight mlp.gate_proj.weight mlp.up_proj.weight), &(p <> &1)))
        |> Map.merge(%{(p <> "self_attn.qkv_proj.weight") => qkv, (p <> "mlp.gate_up_proj.weight") => gu})
      end)

    pmap = map |> Map.drop(["attention_bias"]) |> Map.merge(%{"model_type" => "phi3", "sliding_window" => nil}) |> Map.merge(over)
    {map, c, ws, pmap, fused}
  end

  test "phi3 (a data alias): fused projections split by the lock, the same bits as the Llama program" do
    {_map, c, ws, pmap, fused} = phi3()
    assert {:ok, spec, split} = Lock.from_map(pmap, fused)
    assert spec.family == "phi3" and spec.lineage == ["phi3", "llama"]
    assert split == ws
    {:ok, p} = Lock.build(spec, split, max_seq: 16)
    {:ok, ref} = Llama.program(c, ws, max_seq: 16)
    e = env(c, [3, 9, 27, 81, 50])
    assert Oracle.eval_program(p, e).logits == Oracle.eval_program(ref, e).logits
  end

  test "phi3: what the alias does not cover is refused by field" do
    {_, _, _, pmap, fused} = phi3()
    # partial rotary (Phi-4-mini) is built since 0.5.0; a factor that leaves an odd rotary width is refused,
    # and so is any rope key whose meaning is not implemented — at the top level or nested (transformers ≥ 5)
    assert {:ok, %{config: %{rotary_dim: 8}}, _} = Lock.from_map(Map.put(pmap, "partial_rotary_factor", 0.5), fused)
    assert {:ok, %{config: %{rotary_dim: 8}}, _} =
             Lock.from_map(Map.put(pmap, "rope_parameters", %{"rope_type" => "default", "rope_theta" => 1.0e4, "partial_rotary_factor" => 0.5}), fused)
    assert {:error, %Rejection{node: {:config, "partial_rotary_factor"}}} = Lock.from_map(Map.put(pmap, "partial_rotary_factor", 0.1875), fused)
    assert {:error, %Rejection{node: {:config, "rope_parameters.mrope_section"}}} =
             Lock.from_map(Map.put(pmap, "rope_parameters", %{"rope_type" => "default", "mrope_section" => [2, 2]}), fused)
    assert {:error, %Rejection{node: {:config, "rope_scaling"}}} =
             Lock.from_map(Map.put(pmap, "rope_scaling", %{"type" => "longrope", "short_factor" => [], "long_factor" => []}), fused)

    bad = Map.update!(fused, "model.layers.0.self_attn.qkv_proj.weight", fn t -> %{t | shape: [hd(t.shape) - 16, 64], data: binary_part(t.data, 0, (hd(t.shape) - 16) * 64 * 4)} end)
    assert {:error, %Rejection{node: {:weight, "model.layers.0.self_attn.qkv_proj.weight"}}} = Lock.from_map(pmap, bad)
  end

  test "an alias from JSON: renamed config keys and tensors, no code" do
    File.mkdir_p!(@tmp)
    path = Path.join(@tmp, "renamed.json")

    File.write!(path, Vapor.JSON.encode(%{
      "id" => "renamed", "model_type" => "renamed_lm", "like" => "qwen2",
      "config" => %{"rename" => %{"n_layer" => "num_hidden_layers", "d_model" => "hidden_size"}},
      "tensors" => %{"rename" => [["tr.h.{l}.ln1.weight", "model.layers.{l}.input_layernorm.weight"], ["tr.wte.weight", "model.embed_tokens.weight"]]}
    }))

    assert {:ok, ["renamed"]} = Lock.register_json(path)
    map = tiny_config("qwen2")
    {:ok, c} = Config.from_map(map)
    ws = tiny_weights(c)
    theirs = map |> Map.drop(["num_hidden_layers", "hidden_size"]) |> Map.merge(%{"model_type" => "renamed_lm", "n_layer" => 2, "d_model" => 64})

    ren = ws |> Map.delete("model.embed_tokens.weight") |> Map.put("tr.wte.weight", ws["model.embed_tokens.weight"])
    ren = Enum.reduce(0..1, ren, fn l, acc -> {t, acc} = Map.pop(acc, "model.layers.#{l}.input_layernorm.weight"); Map.put(acc, "tr.h.#{l}.ln1.weight", t) end)

    assert {:ok, spec, back} = Lock.from_map(theirs, ren)
    assert spec.family == "renamed" and back == ws
  end

  test "the documented example alias (examples/eclusa/meu_llama.json) admits a renamed, fused Llama" do
    on_exit(fn -> Lock.unregister("meu_llama") end)
    assert {:ok, ["meu_llama"]} = Lock.register_json(Path.expand("../../examples/eclusa/meu_llama.json", __DIR__))
    {_map, c, ws, _pmap, fused} = phi3()
    map = tiny_config("llama", %{"num_key_value_heads" => 2})

    theirs =
      map
      |> Map.drop(~w(num_hidden_layers hidden_size num_attention_heads num_key_value_heads intermediate_size rms_norm_eps attention_bias))
      |> Map.merge(%{"model_type" => "meu_llama_v2", "n_layer" => 2, "n_embd" => 64, "n_head" => 4, "n_kv_head" => 2,
                     "ffn_dim" => 96, "norm_eps" => 1.0e-5})

    ren = %{"model.embed_tokens.weight" => "transformer.wte.weight", "model.norm.weight" => "transformer.ln_f.weight"}
    per = %{"input_layernorm.weight" => "ln_1.weight", "post_attention_layernorm.weight" => "ln_2.weight",
            "self_attn.o_proj.weight" => "attn.out.weight", "mlp.down_proj.weight" => "mlp.down.weight",
            "self_attn.qkv_proj.weight" => "attn.qkv.weight", "mlp.gate_up_proj.weight" => "mlp.gate_up.weight"}

    theirs_ws =
      Map.new(fused, fn {k, t} ->
        case {ren[k], Regex.run(~r/^model\.layers\.(\d+)\.(.+)$/, k)} do
          {nil, [_, l, rest]} -> {"transformer.h.#{l}." <> Map.get(per, rest, rest), t}
          {nil, nil} -> {k, t}
          {new, _} -> {new, t}
        end
      end)

    assert {:ok, spec, back} = Lock.from_map(theirs, theirs_ws)
    assert spec.family == "meu_llama" and back == ws
    assert {:error, %Rejection{node: {:config, "alibi"}}} = Lock.from_map(Map.put(theirs, "alibi", true), theirs_ws)
    _ = c
  end

  test "the built-in alias descriptors are valid" do
    for d <- Alias.builtin(), do: assert({:ok, _} = Alias.validate(d))
  end

  # ------------------------------------------------------- tier 2: blueprint --

  defp granite do
    map = tiny_config("llama", %{"model_type" => "granite", "attention_bias" => true, "tie_word_embeddings" => true,
                                 "embedding_multiplier" => 3.0, "attention_multiplier" => 0.125,
                                 "residual_multiplier" => 0.22, "logits_scaling" => 4.0})
    {:ok, c} = Config.from_map(%{map | "model_type" => "llama"})
    {map, tiny_weights(c, 5)}
  end

  test "granite (a blueprint): the multipliers become decoder knobs; config.json round trips" do
    {map, ws} = granite()
    {:ok, spec, _} = Lock.from_map(map, ws)
    assert spec.family == "granite"
    c = spec.config
    assert {c.embed_scale, c.attn_scale, c.residual_scale, c.logit_divisor} == {3.0, 0.125, 0.22, 4.0}
    assert {:ok, %{"model_type" => "granite", "residual_multiplier" => 0.22}} = Config.to_map(c)

    # the knobs change the program; a plain llama of the same weights differs
    {:ok, p} = Lock.build(spec, ws, max_seq: 16)
    {:ok, plain} = Lock.build(%{c | embed_scale: nil, attn_scale: nil, residual_scale: nil, logit_divisor: nil, arch: "llama"}, ws, max_seq: 16)
    e = env(c, [1, 2, 3])
    refute Oracle.eval_program(p, e).logits == Oracle.eval_program(plain, e).logits
    assert {:error, %Rejection{node: {:config, "residual_multiplier"}}} = Lock.from_map(Map.put(map, "residual_multiplier", "x"), ws)
  end

  test "GGUF export refuses what GGUF's llama cannot carry (it used to write it silently)" do
    {map, ws} = granite()
    {:ok, spec, _} = Lock.from_map(map, ws)
    path = Path.join(System.tmp_dir!(), "vapor-granite.gguf")
    assert {:error, %Rejection{node: {:gguf, "granite"}, bound: b}} = Vapor.Model.GGUF.write(path, spec.config, ws)
    assert b =~ "residual_scale"
    refute File.exists?(path)

    {:ok, q3} = Config.from_map(tiny_config("qwen3", %{"head_dim" => 16}))
    assert {:error, %Rejection{node: {:gguf, "qwen3"}}} = Vapor.Model.GGUF.write(path, q3, tiny_weights(q3))
  end

  # --------------------------------------------------- python references --

  defp reference(mode, ws, cfg, input) do
    File.mkdir_p!(@tmp)
    wp = Path.join(@tmp, "w-#{System.unique_integer([:positive])}.safetensors")
    cp = wp <> ".json"
    :ok = Vapor.Ingest.Safetensors.write(wp, Map.new(ws, fn {k, t} -> {k, Tensor.widen(t)} end))
    File.write!(cp, Vapor.JSON.encode(cfg))
    out = py!(File.read!(Path.expand("../python/np_reference.py", __DIR__)), [mode, wp, cp], Vapor.JSON.encode(input))
    {:ok, got} = Vapor.JSON.decode(out)
    got
  end

  defp rel_err(a, b) do
    {fa, fb} = {List.flatten(a), List.flatten(b)}
    scale = fb |> Enum.map(&abs/1) |> Enum.max()
    (Enum.zip_with(fa, fb, &abs(&1 - &2)) |> Enum.max()) / scale
  end

  @tag :python
  test "phi3 and granite against independent NumPy references (float64)" do
    toks = [5, 17, 3, 90, 44, 1]

    {_, c, _, pmap, fused} = phi3()
    {:ok, spec, ws} = Lock.from_map(pmap, fused)
    {:ok, p} = Lock.build(spec, ws, max_seq: 16)
    got = Oracle.eval_program(p, env(c, toks)).logits |> Tensor.to_floats() |> Enum.chunk_every(96)
    assert rel_err(got, reference("decoder", fused, pmap, %{tokens: toks})["logits"]) < 2.0e-6

    {gmap, gws} = granite()
    {:ok, gs, gws2} = Lock.from_map(gmap, gws)
    {:ok, gp} = Lock.build(gs, gws2, max_seq: 16)
    got = Oracle.eval_program(gp, env(gs.config, toks)).logits |> Tensor.to_floats() |> Enum.chunk_every(96)
    assert rel_err(got, reference("decoder", gws, gmap, %{tokens: toks})["logits"]) < 2.0e-6
  end

  # ---------------------------------------------------- tier 3: topologies --

  defp tiny_vit do
    cfg = %{"model_type" => "vit", "hidden_size" => 32, "num_hidden_layers" => 2, "num_attention_heads" => 2,
            "intermediate_size" => 64, "image_size" => 8, "patch_size" => 4, "num_channels" => 3, "hidden_act" => "gelu",
            "layer_norm_eps" => 1.0e-12, "qkv_bias" => true}
    r = fn shape, s -> Tensor.random(:f32, shape, s, scale: 0.3) end
    pre = "vit."

    lay = for l <- 0..1, {n, shape} <- [{"layernorm_before.weight", [32]}, {"layernorm_before.bias", [32]}, {"layernorm_after.weight", [32]},
                                        {"layernorm_after.bias", [32]}, {"attention.attention.query.weight", [32, 32]}, {"attention.attention.query.bias", [32]},
                                        {"attention.attention.key.weight", [32, 32]}, {"attention.attention.key.bias", [32]},
                                        {"attention.attention.value.weight", [32, 32]}, {"attention.attention.value.bias", [32]},
                                        {"attention.output.dense.weight", [32, 32]}, {"attention.output.dense.bias", [32]},
                                        {"intermediate.dense.weight", [64, 32]}, {"intermediate.dense.bias", [64]},
                                        {"output.dense.weight", [32, 64]}, {"output.dense.bias", [32]}], do: {pre <> "encoder.layer.#{l}." <> n, shape}

    names = [{pre <> "embeddings.cls_token", [1, 1, 32]}, {pre <> "embeddings.position_embeddings", [1, 5, 32]},
             {pre <> "embeddings.patch_embeddings.projection.weight", [32, 3, 4, 4]}, {pre <> "embeddings.patch_embeddings.projection.bias", [32]},
             {pre <> "layernorm.weight", [32]}, {pre <> "layernorm.bias", [32]}, {"classifier.weight", [5, 32]}, {"classifier.bias", [5]}] ++ lay

    ws = names |> Enum.with_index(70) |> Map.new(fn {{n, s}, i} -> {n, r.(s, i)} end)
    # layer-norm weights near 1
    ws = Map.new(ws, fn {n, t} -> if String.ends_with?(n, "norm.weight") or String.ends_with?(n, "layernorm_before.weight") or String.ends_with?(n, "layernorm_after.weight"),
                                    do: {n, Tensor.from_list(:f32, t.shape, Enum.map(Tensor.to_floats(t), &(&1 + 1.0)))}, else: {n, t} end)
    {cfg, ws}
  end

  defp vit_image, do: Vapor.Modal.Image.new(8, 8, 3, Vapor.Modal.Rng.uniform(5, 192))

  test "a ViT through the encoder topology: contract, [CLS], bidirectional attention, classifier" do
    {cfg, ws} = tiny_vit()
    {:ok, spec, ws} = Lock.from_map(cfg, ws)
    assert %Spec{interface: :encoder, rows: 5, width: 32, family: "vit"} = spec
    {:ok, p} = Lock.build(spec, ws)
    e = Encoder.input(spec, Vapor.Modal.Image.patches(vit_image(), 4, normalize: {0.5, 0.5}))
    out = Oracle.eval_program(p, e)
    assert out.hidden.shape == [5, 32] and out.logits.shape == [1, 5]
  end

  @tag :python
  test "the ViT against an independent NumPy reference (convolution computed as a convolution)" do
    {cfg, ws0} = tiny_vit()
    {:ok, spec, ws} = Lock.from_map(cfg, ws0)
    {:ok, p} = Lock.build(spec, ws)
    img = vit_image()
    out = Oracle.eval_program(p, Encoder.input(spec, Vapor.Modal.Image.patches(img, 4, normalize: {0.5, 0.5})))
    chw = for ch <- 0..2, do: (for y <- 0..7, do: (for x <- 0..7, do: (Vapor.Modal.Image.at(img, x, y, ch) - 0.5) / 0.5))
    ref = reference("vit", ws0, cfg, %{pixels: chw})
    assert rel_err(out.hidden |> Tensor.to_floats() |> Enum.chunk_every(32), ref["hidden"]) < 2.0e-6
    assert rel_err(Tensor.to_floats(out.logits), ref["logits"]) < 2.0e-6
  end

  test "bidirectional = causal with the last row as horizon: permutation equivariance and padding invariance" do
    d = 32
    r = fn shape, s -> Tensor.random(:f32, shape, s, scale: 0.3) end
    lay = for l <- 0..1, {n, s} <- [{"norm1.weight", [d]}, {"norm1.bias", [d]}, {"norm2.weight", [d]}, {"norm2.bias", [d]}, {"q.weight", [d, d]},
                                    {"q.bias", [d]}, {"k.weight", [d, d]}, {"k.bias", [d]}, {"v.weight", [d, d]}, {"v.bias", [d]},
                                    {"o.weight", [d, d]}, {"o.bias", [d]}, {"fc1.weight", [64, d]}, {"fc1.bias", [64]},
                                    {"fc2.weight", [d, 64]}, {"fc2.bias", [d]}], do: {"layers.#{l}.#{n}", s}
    zero_pos = Tensor.from_list(:f32, [6, d], List.duplicate(0.0, 6 * d))
    ws = ([{"embed.weight", [d, 16]}, {"embed.bias", [d]}, {"norm.weight", [d]}, {"norm.bias", [d]}] ++ lay)
         |> Enum.with_index(9) |> Map.new(fn {{n, s}, i} -> {n, r.(s, i)} end) |> Map.put("pos", zero_pos)
    cfg = %{"model_type" => "vapor_encoder", "rows" => 6, "row_width" => 16, "hidden_size" => d, "num_hidden_layers" => 2,
            "num_attention_heads" => 2, "intermediate_size" => 64, "norm" => "layer"}
    {:ok, spec, ws} = Lock.from_map(cfg, ws)
    {:ok, p} = Lock.build(spec, ws)

    rows = Tensor.random(:f32, [4, 16], 3)
    hid = fn rows -> Oracle.eval_program(p, Encoder.input(spec, rows)).hidden |> Tensor.to_floats() |> Enum.chunk_every(d) |> Enum.take(4) end
    base = hid.(rows)

    # without positions, permuting the rows permutes the outputs (every row sees
    # the same set) — up to rounding: the softmax sums its keys in row order
    perm = [2, 0, 3, 1]
    permuted = Tensor.new(:f32, [4, 16], Enum.map_join(perm, &Tensor.row(rows, &1)))
    diff = Enum.zip_with(List.flatten(hid.(permuted)), List.flatten(Enum.map(perm, &Enum.at(base, &1))), &abs(&1 - &2)) |> Enum.max()
    assert diff < 1.0e-5

    # a causal horizon (row t sees 0…t) breaks it: the first row then sees only itself
    causal = %{Encoder.input(spec, rows) | horizon: Tensor.from_list(:s32, [6], [0, 1, 2, 3, 3, 3])}
    first = Oracle.eval_program(p, causal).hidden |> Tensor.to_floats() |> Enum.take(d)
    assert Enum.zip_with(first, hd(base), &abs(&1 - &2)) |> Enum.max() > 1.0e-3

    # the padding rows are never seen: their content does not reach the real rows
    e = Encoder.input(spec, rows)
    junk = %{e | rows: Tensor.new(:f32, [6, 16], binary_part(e.rows.data, 0, 4 * 64) <> Tensor.random(:f32, [2, 16], 99).data)}
    assert Oracle.eval_program(p, junk).hidden |> Tensor.to_floats() |> Enum.chunk_every(d) |> Enum.take(4) == base
  end

  @tag :native
  test "encoder, codec and projector programs: the native worker = the oracle, bit for bit" do
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    {cfg, ws} = tiny_vit()
    {:ok, spec, ws} = Lock.from_map(cfg, ws)
    {:ok, p} = Lock.build(spec, ws)
    e = Encoder.input(spec, Vapor.Modal.Image.patches(vit_image(), 4, normalize: {0.5, 0.5}))
    {:ok, comp} = Vapor.Compile.Lower.lower(p)
    {:ok, ref} = Native.run_oracle(comp, e)

    for isa <- Vapor.Runtime.Substrates.host_isas() do
      {:ok, got} = Native.run(w, comp, e, isa: isa, mode: :native)
      assert got.outputs == ref.outputs, inspect(isa)
    end
  end

  # ---------------------------------------- architecture: no family in the core --

  @core [Vapor.Engine, Vapor.Engine.Pool, Vapor.Embed, Vapor.Serve, Vapor.Speculative, Vapor.RAG, Vapor.Merge, Vapor.Console,
         Vapor.Docs, Vapor.Docs.Library, Vapor.Docs.PDF, Vapor.Docs.Zip, Vapor.Docs.Office, Vapor.Docs.Pictures,
         Vapor.Modal.Runner, Vapor.Modal.Text, Vapor.Modal.Hub, Vapor.Quality.Text, Vapor.Quality.Signal, Vapor.Quality.Gate,
         Vapor.Program, Vapor.Compile.Lower, Vapor.Compile.Rewrite, Vapor.Runtime.Native, Vapor.Runtime.Oracle,
         Vapor.Runtime.Session, Vapor.Runtime.Dispatch, Vapor.Verify.Ladder, Vapor.Algebra.Term, Vapor.KIR.Kernels,
         Vapor.Emit.X86, Vapor.Emit.ARM, Vapor.Emit.RVV, Vapor.Emit.SpirV, Vapor.Recurrent, Vapor.Speculative.Tree,
         Vapor.Shard, Vapor.Spatial, Vapor.Tlog, Vapor.Runtime.Plan]

  @families [Vapor.Model.Llama, Vapor.Model.Config, Vapor.Model.GGUF, Vapor.Lock.Adapters.Decoder, Vapor.Lock.Adapters.Granite,
             Vapor.Lock.Adapters.Encoder, Vapor.Lock.Adapters.Codec, Vapor.Lock.Adapters.Linear, Vapor.Lock.Alias,
             Vapor.Lock.Adapters.Mamba, Vapor.Lock.Adapters.Whisper, Vapor.Lock.Adapters.VAE, Vapor.Lock.Adapters.DiT,
             Vapor.Lock.Adapters.MLP]

  test "the core's compiled modules reference no model family and no adapter (only Vapor.Lock)" do
    for mod <- @core do
      {:module, _} = Code.ensure_loaded(mod)
      {:ok, {_, [atoms: atoms]}} = :beam_lib.chunks(:code.which(mod), [:atoms])
      refs = for {_, a} <- atoms, a in @families, do: a
      assert refs == [], "#{inspect(mod)} references #{inspect(refs)}"
    end
  end
end
