defmodule Vapor.FrontierTest do
  @moduledoc """
  Frontier families — Qwen3, Qwen3-MoE, Mixtral, Gemma 3, DeepSeek-V3 —
  built from the algebra's existing operators (selector contractions,
  rank-and-select mixtures, latent attention by layout; see
  `Vapor.Model.Decoder`), hence bit-identical on every substrate with no new
  machine code; and, against Hugging Face transformers, within the same
  declared tolerance as the Llama family (`hf_parity_test`).
  """
  use ExUnit.Case, async: false
  alias Vapor.{Rejection, Tensor}
  alias Vapor.Compile.Lower
  alias Vapor.Ingest.Safetensors
  alias Vapor.Model.{Config, Decoder}
  alias Vapor.Runtime.{Dispatch, Native, Substrates, Worker}
  import Vapor.TestHelpers

  @s 16

  @families [
    {"qwen3", %{"head_dim" => 16}},
    {"qwen3_moe", %{"head_dim" => 16, "num_experts" => 6, "num_experts_per_tok" => 2, "moe_intermediate_size" => 32,
                    "norm_topk_prob" => true, "mlp_only_layers" => [0]}},
    {"mixtral", %{"num_local_experts" => 4, "num_experts_per_tok" => 2}},
    {"gemma3_text", %{"head_dim" => 16, "sliding_window" => 64, "query_pre_attn_scalar" => 24,
                      "layer_types" => ["sliding_attention", "full_attention"], "final_logit_softcapping" => 30.0,
                      "hidden_activation" => "gelu_pytorch_tanh",
                      "rope_parameters" => %{"full_attention" => %{"rope_type" => "linear", "factor" => 8.0, "rope_theta" => 1.0e6},
                                             "sliding_attention" => %{"rope_type" => "default", "rope_theta" => 1.0e4}}}},
    {"deepseek_v3", %{"q_lora_rank" => 32, "kv_lora_rank" => 32, "qk_nope_head_dim" => 16, "qk_rope_head_dim" => 16,
                      "v_head_dim" => 24, "n_routed_experts" => 8, "num_experts_per_tok" => 3, "moe_intermediate_size" => 32,
                      "n_shared_experts" => 1, "first_k_dense_replace" => 1, "n_group" => 4, "topk_group" => 2,
                      "routed_scaling_factor" => 2.5, "num_key_value_heads" => 4, "max_position_embeddings" => 64,
                      "rope_parameters" => %{"rope_type" => "yarn", "factor" => 4.0, "original_max_position_embeddings" => 16,
                                             "rope_theta" => 1.0e4, "mscale" => 1.0, "mscale_all_dim" => 0.707}}}
  ]

  defp model(arch, over, opts \\ []) do
    {:ok, c} = Config.from_map(tiny_config(arch, over))
    ws = Keyword.get_lazy(opts, :weights, fn -> tiny_weights(c, 3) end)
    {:ok, p} = Decoder.program(c, ws, [max_seq: @s] ++ Keyword.delete(opts, :weights))
    {c, p}
  end

  defp env(c, toks, p0 \\ 0, caches \\ nil) do
    n = length(toks)
    Map.merge(caches || Decoder.empty_caches(c, @s),
              %{tok: Tensor.from_list(:s32, [n], toks), pos: Tensor.from_list(:s32, [n], Enum.to_list(p0..(p0 + n - 1)))})
  end

  defp lower!(p) do
    {:ok, c} = Lower.lower(p)
    c
  end

  @toks [1, 95, 7, 7, 42, 0]

  # ------------------------------------------------------------ configs --

  test "configurations: every family parses; unsupported variants are rejections naming the field" do
    for {arch, over} <- @families, do: assert({:ok, %Config{arch: ^arch}} = Config.from_map(tiny_config(arch, over)))

    gemma2ish = tiny_config("gemma3_text", Map.put(elem(Enum.at(@families, 3), 1), "attn_logit_softcapping", 50.0))
    assert {:error, %Rejection{node: {:config, "attn_logit_softcapping"}}} = Config.from_map(gemma2ish)

    dyn = tiny_config("qwen3", %{"head_dim" => 16, "rope_parameters" => %{"rope_type" => "dynamic", "factor" => 2.0}})
    assert {:error, %Rejection{node: {:config, "rope_scaling"}, bound: "rope_type default, linear, llama3 or yarn" <> _}} = Config.from_map(dyn)

    biased = tiny_config("deepseek_v3", Map.put(elem(List.last(@families), 1), "attention_bias", true))
    assert {:error, %Rejection{node: {:config, "attention_bias"}}} = Config.from_map(biased)

    bad_groups = tiny_config("deepseek_v3", Map.put(elem(List.last(@families), 1), "topk_group", 1))
    assert {:error, %Rejection{node: {:config, "n_group/topk_group"}}} = Config.from_map(bad_groups)

    # frontier families are written back as read: from_map ∘ to_map = id
    for {arch, over} <- @families do
      {:ok, c} = Config.from_map(tiny_config(arch, over))
      {:ok, m} = Config.to_map(c)
      assert Config.from_map(m) == {:ok, c}
    end
  end

  test "the MLA head layout: rotate-half pairs are both rotary or both pass-through, every slot used once" do
    {:ok, c} = Config.from_map(tiny_config("deepseek_v3", elem(List.last(@families), 1)))
    {nope, rope} = Decoder.mla_layout(c)
    %{nope: dn, rope: dr} = c.mla
    half = div(c.head_dim, 2)
    slots = Enum.map(0..(dn - 1), nope) ++ Enum.map(0..(dr - 1), rope)
    assert length(Enum.uniq(slots)) == dn + dr and Enum.all?(slots, &(&1 in 0..(c.head_dim - 1)))
    rot = MapSet.new(Enum.map(0..(dr - 1), rope))
    # partner of every rotary slot is rotary; interleaved pairs (2p, 2p+1) land on (j, j + half)
    assert Enum.all?(rot, fn j -> MapSet.member?(rot, if(j < half, do: j + half, else: j - half)) end)
    assert Enum.all?(0..(div(dr, 2) - 1), fn p -> rope.(2 * p + 1) - rope.(2 * p) == half end)
  end

  # --------------------------------------------------------- substrates --

  @tag :native
  @tag timeout: 900_000
  test "every frontier family: host ISAs and the poisoned RVV interpreter bit-identical to the oracle" do
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))

    for {arch, over} <- @families do
      {c, p} = model(arch, over)
      comp = lower!(p)
      e = env(c, @toks)
      {:ok, ref} = Native.run_oracle(comp, e)

      for isa <- Substrates.host_isas() do
        {:ok, got} = Native.run(w, comp, e, isa: isa, mode: :native)
        assert got.outputs == ref.outputs, "#{arch} on #{isa}"
      end

      {:ok, emu} = Native.run(w, comp, e, isa: :riscv64, mode: :emulate, poison: true, vlen: 256)
      assert emu.outputs == ref.outputs, "#{arch} on the RVV interpreter"
    end
  end

  @tag :qemu
  @tag timeout: 1_800_000
  test "latent attention and expert routing under QEMU (AArch64 NEON, RVV VLEN 128) bit-identical to the oracle" do
    for {target, isa} <- [{:aarch64, :aarch64}, {{:riscv64, 128}, :riscv64}] do
      {:ok, w} = Worker.start_link(exec: worker_exec(target))

      for {arch, over} <- Enum.filter(@families, &(elem(&1, 0) in ["qwen3_moe", "deepseek_v3", "gemma3_text"])) do
        {c, p} = model(arch, over)
        comp = lower!(p)
        e = env(c, Enum.take(@toks, 3))
        {:ok, ref} = Native.run_oracle(comp, e)
        {:ok, got} = Native.run(w, comp, e, isa: isa, mode: :native)
        assert got.outputs == ref.outputs, "#{arch} on #{inspect(target)}"
      end
    end
  end

  @tag :vulkan
  @tag timeout: 900_000
  test "every frontier family on the Vulkan fabric bit-identical to the oracle" do
    fabric = Enum.find(Substrates.list(), &(&1.kind == :fabric))

    for {arch, over} <- @families do
      {c, p} = model(arch, over)
      comp = lower!(p)
      e = env(c, Enum.take(@toks, 4))
      {:ok, ref} = Native.run_oracle(comp, e)
      {:ok, got} = Dispatch.run_on(fabric, comp, e, [])
      assert got.outputs == ref.outputs, "#{arch} on the fabric"
    end
  end

  @tag :native
  test "prefill = cached decoding, bit for bit, through routers and latent attention" do
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))

    for {arch, over} <- @families, arch in ["qwen3_moe", "deepseek_v3", "gemma3_text"] do
      {c, p} = model(arch, over)
      comp = lower!(p)
      {:ok, all} = Native.run(w, comp, env(c, @toks), isa: Substrates.host_isa(), mode: :native)

      {rows, _} =
        Enum.map_reduce(Enum.with_index(@toks), nil, fn {t, i}, caches ->
          {:ok, got} = Native.run(w, comp, env(c, [t], i, caches), isa: Substrates.host_isa(), mode: :native)
          next = for l <- 0..(c.layers - 1), name <- Decoder.cache_names(c, l), into: %{}, do: {name, got.outputs[:"#{name}_next"]}
          {got.outputs.logits.data, next}
        end)

      assert IO.iodata_to_binary(rows) == all.outputs.logits.data, arch
    end
  end

  # Selection, not multiplication by a zero gate: an expert no row selects
  # may hold NaN or ∞ weights and not one output bit changes — so a runtime
  # that skips unselected experts computes the dense definition exactly.
  @tag :native
  test "mixture of experts: unselected experts contribute nothing — not even NaN" do
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))

    for {arch, over} <- @families, arch in ["qwen3_moe", "mixtral", "deepseek_v3"] do
      {:ok, c} = Config.from_map(tiny_config(arch, over))
      ws = tiny_weights(c, 3)
      {_, p} = model(arch, over, weights: ws)
      layer = Enum.find_index(c.moe.sparse, & &1)
      tok = [42]

      # which experts the one row selects, read from the router's ranks
      ranks = for e <- 0..(c.moe.experts - 1), do: :"layers.#{layer}.rank#{e}"
      probe = %{p | outputs: p.outputs ++ Enum.map(ranks, fn n -> {n, Vapor.Algebra.Term.input(n, :f32, [Vapor.Algebra.Term.dyn(:t, @s), 1])} end)}
      {:ok, ref} = Native.run(w, lower!(probe), env(c, tok), isa: Substrates.host_isa(), mode: :native)
      selected = for {n, e} <- Enum.with_index(ranks), (ref.outputs[n] |> Tensor.to_floats() |> hd()) < c.moe.top_k, do: e
      assert length(selected) == c.moe.top_k
      idle = Enum.find(0..(c.moe.experts - 1), &(&1 not in selected))

      for bad <- [0x7FC0_0000, 0x7F80_0000] do
        poisoned =
          Map.new(ws, fn {k, t} ->
            if String.contains?(k, "layers.#{layer}.") and String.contains?(k, "experts.#{idle}."),
              do: {k, Tensor.new(:f32, t.shape, :binary.copy(<<bad::32-little>>, Enum.product(t.shape)))},
              else: {k, t}
          end)

        {_, pp} = model(arch, over, weights: poisoned)
        {:ok, got} = Native.run(w, lower!(pp), env(c, tok), isa: Substrates.host_isa(), mode: :native)
        assert got.outputs.logits.data == ref.outputs.logits.data, "#{arch}: expert #{idle} poisoned with 0x#{Integer.to_string(bad, 16)}"
      end
    end
  end
end

defmodule Vapor.FrontierHFTest do
  @moduledoc """
  The frontier families against Hugging Face transformers (PyTorch,
  float32): small random checkpoints from `test/python/hf_frontier.py`,
  under the tolerance of `hf_parity_test` — and no checkpoint tensor left
  unread.
  """
  use ExUnit.Case, async: false
  alias Vapor.Tensor
  alias Vapor.Compile.Lower
  alias Vapor.Ingest.Safetensors
  alias Vapor.Model.Decoder
  alias Vapor.Runtime.{Native, Substrates, Worker}
  import Vapor.TestHelpers

  @moduletag :torch
  @moduletag timeout: 900_000

  defp lower!(p) do
    {:ok, c} = Lower.lower(p)
    c
  end

  # -------------------------------------------------- transformers oracle --

  @variants ~w(qwen3 qwen3-yarn qwen3-moe qwen3-moe-norm mixtral gemma3 gemma3-window mistral-window deepseek deepseek-yarn)
  @tol 1.0e-5

  setup_all do
    root = Path.join(System.tmp_dir!(), "vapor-frontier-#{System.unique_integer([:positive])}")
    script = Path.expand("../python/hf_frontier.py", __DIR__)

    for v <- @variants do
      File.mkdir_p!(Path.join(root, v))
      py!(File.read!(script), [Path.join(root, v), v, "7"])
    end

    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    on_exit(fn -> File.rm_rf!(root) end)
    {:ok, root: root, worker: w}
  end

  for v <- @variants do
    test "#{v}: every checkpoint tensor is read, prefill within tolerance, greedy decoding identical", %{root: root, worker: w} do
      dir = Path.join(root, unquote(v))
      {:ok, ref} = Safetensors.read(Path.join(dir, "reference.safetensors"))
      {:ok, m} = Vapor.Model.open(dir)
      # nothing in the checkpoint is silently ignored
      expected = m.config |> Decoder.expected_weights() |> Enum.map(&elem(&1, 0)) |> Enum.sort()
      assert expected == m.weights |> Map.keys() |> Enum.filter(&is_binary/1) |> Enum.sort()

      {:ok, %{config: c, program: p}} = Vapor.Model.load(dir, max_seq: 32)
      comp = lower!(p)
      prompt = Tensor.to_list(ref["prompt"])
      step = fn caches, toks, p0 ->
        n = length(toks)
        e = Map.merge(caches, %{tok: Tensor.from_list(:s32, [n], toks), pos: Tensor.from_list(:s32, [n], Enum.to_list(p0..(p0 + n - 1)))})
        {:ok, got} = Native.run(w, comp, e, isa: Substrates.host_isa(), mode: :native)
        {got.outputs.logits, Map.new(caches, fn {k, _} -> {k, got.outputs[:"#{k}_next"]} end)}
      end

      {logits, caches} = step.(Decoder.empty_caches(c, 32), prompt, 0)
      want = Tensor.to_floats(ref["logits"])
      scale = want |> Enum.map(&abs/1) |> Enum.max()
      err = Enum.zip_with(Tensor.to_floats(logits), want, &abs(&1 - &2)) |> Enum.max()
      assert err <= @tol * scale, "max |Δ| = #{err}, allowed #{@tol * scale}"

      greedy = ref["greedy"] |> Tensor.to_list() |> Enum.drop(length(prompt))
      [_, vsz] = logits.shape

      {got, _} =
        Enum.map_reduce(Enum.with_index(greedy), {logits, caches}, fn {_, i}, {lg, cs} ->
          row = lg |> Tensor.to_floats() |> Enum.take(-vsz)
          {tok, gap} = argmax_gap(row)
          next = if i < length(greedy) - 1, do: step.(cs, [tok], length(prompt) + i), else: {nil, cs}
          {{tok, gap, row |> Enum.map(&abs/1) |> Enum.max()}, next}
        end)

      mismatch = Enum.zip(got, greedy) |> Enum.find_index(fn {{t, _, _}, u} -> t != u end)

      if mismatch do
        {_, gap, sc} = Enum.at(got, mismatch)
        assert gap <= 2 * @tol * sc, "diverged at step #{mismatch} with a clear margin #{gap}"
      end
    end
  end

  defp argmax_gap(xs) do
    {best, i} = xs |> Enum.with_index() |> Enum.max_by(&elem(&1, 0), fn a, b -> a > b end)
    {i, best - (xs |> List.delete_at(i) |> Enum.max())}
  end
end
