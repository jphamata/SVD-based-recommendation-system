defmodule Vapor.LatentAttentionTest do
  @moduledoc """
  Multi-head latent attention with the **compressed cache** (DeepSeek's
  absorbed form, `mla: :latent`, the default): one cache row of
  `kv_lora_rank + qk_rope_head_dim` floats per token and layer, shared by
  every head — the query is mapped into latent space per head and the
  value up-projection applied after attention (`linear_grouped/3`), so no
  per-head key or value is ever materialised.

  Same real arithmetic as the expanded form (`mla: :expanded`), different
  rounding (another association): the two are compared within tolerance
  and on greedy decisions, and the latent program is itself bit-identical
  on every substrate, prefill = cached decoding. Parity with transformers
  is `Vapor.FrontierHFTest` (deepseek, deepseek-yarn), which now runs the
  latent form.
  """
  use ExUnit.Case, async: false
  alias Vapor.Tensor
  alias Vapor.Compile.Lower
  alias Vapor.Model.{Config, Llama}
  alias Vapor.Runtime.{Dispatch, Native, Oracle, Substrates, Worker}
  import Vapor.TestHelpers

  @over %{"q_lora_rank" => 32, "kv_lora_rank" => 32, "qk_nope_head_dim" => 16, "qk_rope_head_dim" => 16,
          "v_head_dim" => 24, "n_routed_experts" => 8, "num_experts_per_tok" => 3, "moe_intermediate_size" => 32,
          "n_shared_experts" => 1, "first_k_dense_replace" => 1, "n_group" => 4, "topk_group" => 2,
          "routed_scaling_factor" => 2.5, "num_key_value_heads" => 4, "max_position_embeddings" => 64,
          "rope_parameters" => %{"rope_type" => "yarn", "factor" => 4.0, "original_max_position_embeddings" => 16,
                                 "rope_theta" => 1.0e4, "mscale" => 1.0, "mscale_all_dim" => 0.707}}
  @s 16
  @toks [1, 95, 7, 7, 42, 0]

  setup_all do
    {:ok, c} = Config.from_map(tiny_config("deepseek_v3", @over))
    {:ok, c: c, ws: tiny_weights(c, 3)}
  end

  defp env(c, mode, toks, p0 \\ 0, caches \\ nil) do
    n = length(toks)
    Map.merge(caches || Llama.empty_caches(c, @s, mla: mode),
              %{tok: Tensor.from_list(:s32, [n], toks), pos: Tensor.from_list(:s32, [n], Enum.to_list(p0..(p0 + n - 1)))})
  end

  defp lower!(p) do
    {:ok, c} = Lower.lower(p)
    c
  end

  test "the cache: one row of kv_lora + rope floats per token and layer, no values cache", %{c: c, ws: ws} do
    {:ok, p} = Llama.program(c, ws, max_seq: @s)
    assert Keyword.keys(p.state) == [:k0, :k1]
    assert Llama.cache_floats(c) == 32 + 16
    assert Llama.cache_floats(c, mla: :expanded) == 2 * c.heads * c.head_dim
    # DeepSeek-V3's shapes: 576 floats against 128 heads × 192 × 2
    v3 = %{c | heads: 128, head_dim: 192, kv_heads: 128, mla: %{c.mla | kv_lora: 512, rope: 64}}
    assert {Llama.cache_floats(v3), Llama.cache_floats(v3, mla: :expanded)} == {576, 49_152}
  end

  test "latent and expanded forms agree within tolerance and on every argmax (oracle)", %{c: c, ws: ws} do
    outs =
      for mode <- [:expanded, :latent] do
        {:ok, p} = Llama.program(c, ws, max_seq: @s, mla: mode)
        Oracle.eval_program(p, env(c, mode, @toks)).logits |> Tensor.to_floats()
      end

    [a, b] = outs
    scale = a |> Enum.map(&abs/1) |> Enum.max()
    assert (Enum.zip_with(a, b, &abs(&1 - &2)) |> Enum.max()) <= 1.0e-5 * scale
    argmax = fn l -> l |> Enum.chunk_every(c.vocab) |> Enum.map(fn r -> r |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1) end) end
    assert argmax.(a) == argmax.(b)
  end

  @tag :native
  @tag timeout: 900_000
  test "latent MLA: host ISAs, RVV interpreter and fabric bit-identical; prefill = cached decoding", %{c: c, ws: ws} do
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    {:ok, p} = Llama.program(c, ws, max_seq: @s)
    comp = lower!(p)
    e = env(c, :latent, @toks)
    {:ok, ref} = Native.run_oracle(comp, e)

    for isa <- Substrates.host_isas(), do: assert(elem(Native.run(w, comp, e, isa: isa, mode: :native), 1).outputs == ref.outputs)
    {:ok, emu} = Native.run(w, comp, e, isa: :riscv64, mode: :emulate, poison: true, vlen: 256)
    assert emu.outputs == ref.outputs

    case Enum.find(Substrates.list(), &(&1.kind == :fabric)) do
      nil -> :ok
      fabric -> assert elem(Dispatch.run_on(fabric, comp, e, []), 1).outputs == ref.outputs
    end

    {rows, _} =
      Enum.map_reduce(Enum.with_index(@toks), Llama.empty_caches(c, @s), fn {t, i}, caches ->
        {:ok, got} = Native.run(w, comp, env(c, :latent, [t], i, caches), isa: Substrates.host_isa(), mode: :native)
        {got.outputs.logits.data, Map.new(caches, fn {k, _} -> {k, got.outputs[:"#{k}_next"]} end)}
      end)

    assert IO.iodata_to_binary(rows) == ref.outputs.logits.data
  end

  @tag :native
  test "the paged engine serves the latent form (one pool per layer)", %{c: c, ws: ws} do
    {:ok, e} = Vapor.Engine.start_link(config: c, weights: ws, max_seq: 32, sequences: 2, page: 8, step_tokens: 8)
    {:ok, ids, _, _} = Vapor.Engine.complete(e, [1, 95, 7], max_tokens: 6, temperature: 0.0)
    assert length(ids) == 6

    # the same greedy tokens as the contiguous latent program
    {:ok, p} = Llama.program(c, ws, max_seq: 32)

    {want, _} =
      Enum.map_reduce(1..6, {[1, 95, 7], Llama.empty_caches(c, 32), 0}, fn _, {toks, caches, p0} ->
        out = Oracle.eval_program(p, Map.merge(caches, %{tok: Tensor.from_list(:s32, [length(toks)], toks),
                                                         pos: Tensor.from_list(:s32, [length(toks)], Enum.to_list(p0..(p0 + length(toks) - 1)))}))
        row = out.logits |> Tensor.to_floats() |> Enum.take(-c.vocab)
        t = row |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1)
        {t, {[t], Map.new(caches, fn {k, _} -> {k, out[:"#{k}_next"]} end), p0 + length(toks)}}
      end)

    assert ids == want
  end
end
