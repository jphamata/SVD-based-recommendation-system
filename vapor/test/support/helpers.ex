defmodule Vapor.TestHelpers do
  @moduledoc false
  alias Vapor.{Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Quant.Sb4

  def worker_exec(:host), do: [Vapor.Runtime.Substrates.binary("vapor-worker", "native")]
  def worker_exec(:aarch64), do: ["qemu-aarch64", Vapor.Runtime.Substrates.binary("vapor-worker", "aarch64-linux")]

  def worker_exec({:riscv64, vlen}),
    do: ["qemu-riscv64", "-cpu", "rv64,v=true,vlen=#{vlen},zfh=true,vext_spec=v1.0", Vapor.Runtime.Substrates.binary("vapor-worker", "riscv64-linux")]

  @doc "The affine SSM block used across the suite: sb4 projections, recurrent state."
  def ssm_block(d \\ 512, ds \\ 256) do
    win = T.const(Sb4.quantize(Tensor.random(:f32, [ds, d], 11)))
    wout = T.const(Sb4.quantize(Tensor.random(:f32, [d, ds], 12)))
    decay = T.const(Tensor.random(:f32, [ds], 13, scale: 0.5))
    x = T.input(:x, :f32, [d])
    h = T.input(:h, :f32, [ds])
    hn = T.fma(decay, h, T.qgemv(win, x))
    Program.new([y: T.qgemv(wout, hn), h_next: hn], state: [h: :h_next])
  end

  def ssm_env(t, d \\ 512, ds \\ 256),
    do: %{x: Tensor.random(:f32, [t, d], 21), h: Tensor.from_list(:f32, [ds], List.duplicate(0.0, ds))}

  @doc "A long elementwise chain that forces the cut sweep to split."
  def ew_chain(len \\ 12) do
    x = T.input(:x, :f32, [T.dyn(:n, 4096)])
    y = T.input(:y, :f32, [T.dyn(:n, 4096)])

    chain =
      Enum.reduce(1..len, x, fn i, acc ->
        case rem(i, 4) do
          0 -> T.fma(acc, y, T.splat(0.25 * i))
          1 -> T.relu(T.sub(acc, y))
          2 -> T.mul(acc, T.add(y, T.splat(1.5)))
          3 -> T.neg(T.add(acc, x))
        end
      end)

    Program.new(out: chain)
  end

  @doc "cos/sin tables f32[S, dh/2] (θ = 10000), rounded once from binary64."
  def rope_tables(s, dh, theta \\ 10_000.0) do
    half = div(dh, 2)
    ang = for p <- 0..(s - 1), i <- 0..(half - 1), do: p / :math.pow(theta, 2 * i / dh)
    {Tensor.from_list(:f32, [s, half], Enum.map(ang, &:math.cos/1)), Tensor.from_list(:f32, [s, half], Enum.map(ang, &:math.sin/1))}
  end

  @doc """
  A pre-norm attention block over an embedding table, with KV caches:
  gather → RMSNorm → q/k/v → RoPE → cache writes → GQA attention → out
  projection → residual. `recurrent: true` feeds the caches back as state.
  """
  def attention_block(opts \\ []) do
    {d, h, hkv, s, v} = {64, 4, 2, 32, 97}
    dh = div(d, h)
    t = Keyword.get(opts, :t, T.dyn(:t, 16))
    w = fn n, k, seed -> T.const(Tensor.random(:f32, [n, k], seed, scale: 0.25)) end
    {cos, sin} = rope_tables(s, dh)
    tok = T.input(:tok, :s32, [t])
    pos = T.input(:pos, :s32, [t])
    x = T.gather_row(T.const(Tensor.random(:f32, [v, d], 31)), tok)
    ms = T.mul(T.reduce(:sum, T.mul(x, x)), T.splat(1 / d))
    xn = T.mul(T.const(Tensor.random(:f32, [1, d], 32)), T.mul(x, T.rsqrt(T.add(ms, T.splat(1.0e-5)))))
    q = T.rope(T.linear(xn, w.(d, d, 33)), T.const(cos), T.const(sin), pos, h)
    k = T.rope(T.linear(xn, w.(hkv * dh, d, 34)), T.const(cos), T.const(sin), pos, hkv)
    vv = T.linear(xn, w.(hkv * dh, d, 35))
    # contiguous caches, or `paged: {page, pages, sequences}` pools + table
    {kn, vn, att} =
      case Keyword.get(opts, :paged) do
        nil ->
          kn = T.kv_write(T.input(:k, :f32, [s, hkv * dh]), pos, k)
          vn = T.kv_write(T.input(:v, :f32, [s, hkv * dh]), pos, vv)
          {kn, vn, T.attention(q, kn, vn, pos, h, hkv)}

        {page, pages, ns} ->
          table = T.input(:table, :s32, [ns, div(s, page)])
          slot = T.input(:slot, :s32, [t])
          kn = T.kv_write_paged(T.input(:k, :f32, [pages * page, hkv * dh]), table, slot, pos, k, page)
          vn = T.kv_write_paged(T.input(:v, :f32, [pages * page, hkv * dh]), table, slot, pos, vv, page)
          {kn, vn, T.attention_paged(q, kn, vn, table, slot, pos, h, hkv, page)}
      end

    y = T.add(x, T.linear(att, w.(d, d, 36)))
    out = [y: y, k_next: kn, v_next: vn]

    if Keyword.get(opts, :recurrent, false),
      do: Program.new(out, state: [k: :k_next, v: :v_next]),
      else: Program.new(out)
  end

  def attention_env(n, opts \\ []) do
    zeros = Tensor.from_list(:f32, [32, 32], List.duplicate(0.0, 32 * 32))
    toks = Tensor.random(:s32, [n], 41, max: 97)
    base = Keyword.get(opts, :base, 0)
    %{tok: toks, pos: Tensor.from_list(:s32, [n], Enum.to_list(base..(base + n - 1))), k: zeros, v: zeros}
  end

  def gemm_program(k \\ 203, n \\ 17) do
    a = T.input(:a, :s8, [T.dyn(:m, 64), k])
    Program.new(c: T.gemm_i8(a, T.const(Tensor.random(:s8, [n, k], 5))))
  end

  # ------------------------------------------------------------ python --

  @doc """
  The interpreter used by the differential tiers: `$VAPOR_PYTHON` or
  `python3`, and whether it can import `mods`.
  """
  def python, do: System.get_env("VAPOR_PYTHON") || System.find_executable("python3")

  def python?(mods) do
    py = python()
    py != nil and match?({_, 0}, System.cmd(py, ["-c", "import " <> Enum.join(mods, ", ")], stderr_to_stdout: true))
  end

  @py_env [{"TQDM_DISABLE", "1"}, {"HF_HUB_DISABLE_PROGRESS_BARS", "1"}, {"TRANSFORMERS_VERBOSITY", "error"}]

  @doc "Run a Python script (stdin: `input`); returns stdout or raises with the output."
  def py!(script, args \\ [], input \\ nil) do
    dir = Path.join(System.tmp_dir!(), "vapor-py-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    file = Path.join(dir, "s.py")
    File.write!(file, script)
    argv = [file | args]

    {out, code} =
      if input do
        inp = Path.join(dir, "in")
        File.write!(inp, input)
        System.cmd("sh", ["-c", ~s(exec "$0" "$@" < "#{inp}"), python() | argv], env: @py_env)
      else
        System.cmd(python(), argv, env: @py_env)
      end

    File.rm_rf!(dir)
    if code != 0, do: raise("python failed (#{code}):\n#{out}")
    out
  end

  # ------------------------------------------------------------ models --

  @doc "A config map of a tiny model of the given family (random weights come from `tiny_weights/2`)."
  def tiny_config(arch, over \\ %{}) do
    Map.merge(
      %{"model_type" => arch, "vocab_size" => 96, "hidden_size" => 64, "intermediate_size" => 96,
        "num_hidden_layers" => 2, "num_attention_heads" => 4, "num_key_value_heads" => 2,
        "max_position_embeddings" => 32, "rms_norm_eps" => 1.0e-5, "rope_theta" => 10_000.0,
        "tie_word_embeddings" => arch == "qwen2", "hidden_act" => "silu"},
      over
    )
  end

  @doc """
  A byte-level BPE tokenizer with no merges (GPT-2 byte aliases, ChatML
  specials `<|im_start|>` 256, `<|im_end|>` 257 = EOS, `<|endoftext|>` 258):
  every text encodes, byte for byte — a real tokenizer for tests that need
  no vocabulary file.
  """
  def byte_bpe_tokenizer do
    printable = Enum.to_list(?!..?~) ++ Enum.to_list(0xA1..0xAC) ++ Enum.to_list(0xAE..0xFF)

    {alias, _} =
      Enum.reduce(0..255, {%{}, 0}, fn b, {m, n} ->
        if b in printable, do: {Map.put(m, b, b), n}, else: {Map.put(m, b, 256 + n), n + 1}
      end)

    {:ok, tk} =
      Vapor.Tokenizer.from_gguf(%{
        "tokenizer.ggml.model" => "gpt2", "tokenizer.ggml.pre" => "qwen2",
        "tokenizer.ggml.tokens" => Enum.map(0..255, &<<alias[&1]::utf8>>) ++ ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
        "tokenizer.ggml.token_type" => List.duplicate(1, 256) ++ [3, 3, 3], "tokenizer.ggml.merges" => [],
        "tokenizer.ggml.eos_token_id" => 257, "tokenizer.ggml.add_bos_token" => false})

    tk
  end

  @doc "Random f32 weights with Hugging Face names for a `Vapor.Model.Config` (any family)."
  def tiny_weights(%Vapor.Model.Config{} = c, seed \\ 1) do
    c
    |> Vapor.Model.Llama.expected_weights()
    |> Enum.with_index(seed * 1000)
    |> Map.new(fn
      {{name, shape, :norm}, i} ->
        # norm weights near 1 (near 0 for Gemma's 1 + w), as trained models have them
        r = Tensor.random(:f32, shape, i, scale: 0.2) |> Tensor.to_floats()
        {name, Tensor.from_list(:f32, shape, Enum.map(r, &(&1 + if(c.norm_offset, do: 0.0, else: 1.0))))}

      {{name, shape, kind}, i} ->
        scale = case {kind, name} do
          {:bias, _} -> 0.1
          {:vector, _} -> 0.1
          {_, "model.embed_tokens.weight"} -> 1.0
          _ -> 0.2
        end

        {name, Tensor.random(:f32, shape, i, scale: scale)}
    end)
  end

  # ------------------------------------------------------ P1 programs --

  # program inputs: squares must stay finite (the oracle certifies finite executions)
  @edge [-87.0001, -87.0, -86.9, -50.0, -1.0e-30, 0.0, 1.0e-30, 1.0e-7, 0.5, 1.0, 2.0,
         20.0, 87.9, 88.0, 88.1, 1.0e18, -1.0e18, 1.2e-38, 1.0e-45]

  defp fx(n, seed, scale) do
    rand = Tensor.random(:f32, [n - length(@edge)], seed, scale: scale) |> Tensor.to_list()
    Tensor.new(:f32, [n], Vapor.F32.encode([0x8000_0000 | Enum.map(tl(@edge), &Vapor.F32.from_float/1)] ++ rand))
  end

  @doc "The P1 canonical-function programs with their inputs."
  def canon_programs do
    x = T.input(:x, :f32, [T.dyn(:n, 2048)])
    m = T.input(:m, :f32, [7, 48])
    v = T.input(:v, :f32, [48])
    w = T.const(Tensor.random(:f32, [37, 48], 5))
    g = T.const(Tensor.random(:f32, [1, 48], 6))
    wb = T.const(Tensor.to_bf16(Tensor.random(:f32, [37, 48], 11, scale: 2.0)))

    rms =
      T.mul(T.mul(m, T.rsqrt(T.add(T.mul(T.reduce(:sum, T.mul(m, m)), T.splat(1 / 48)), T.splat(1.0e-5)))), g)

    e = T.exp(T.sub(m, T.reduce(:max, m)))
    softmax = T.mul(e, T.rcp(T.reduce(:sum, e)))

    [
      {"functions", Program.new(e: T.exp(x), r: T.rcp(x), q: T.rsqrt(T.mul(x, x)), s: T.silu(x),
                                g: T.sigmoid(x), d: T.divide(x, T.add(T.mul(x, x), T.splat(1.0))),
                                mx: T.max(x, T.neg(x)), mn: T.min(x, T.splat(0.25))),
       %{x: fx(1003, 1, 30.0)}},
      # frontier-model functions: tanh (soft-capping), GELU-tanh (Gemma),
      # selection with column/row broadcasting (MoE routing masks)
      {"frontier functions (tanh, gelu_tanh, sel)",
       Program.new(t: T.tanh(x), g: T.gelu_tanh(x), s: T.sel(x, T.splat(0.5), T.mul(x, x), T.neg(x)),
                   b: T.sel(m, T.reduce(:max, m), T.splat(1.0), T.splat(0.0)),
                   r: T.sel(T.reduce(:sum, m), m, m, T.splat(-2.0))),
       %{x: fx(1003, 12, 6.0), m: Tensor.random(:f32, [7, 48], 13, scale: 2.0)}},
      {"rmsnorm (row/col broadcast, sum)", Program.new(y: rms), %{m: Tensor.random(:f32, [7, 48], 2, scale: 3.0)}},
      {"softmax (max, exp, sum, rcp)", Program.new(y: softmax), %{m: Tensor.random(:f32, [7, 48], 3, scale: 8.0)}},
      # widest group factor with broadcast operands in every position: a
      # lanewise destination must never overwrite a register all lanes read
      {"broadcast operands at full width",
       Program.new(y: T.mul(T.sub(T.splat(2.0), x), T.add(x, T.splat(0.5))), z: T.mul(m, T.reduce(:sum, m))),
       %{x: fx(1003, 9, 4.0), m: Tensor.random(:f32, [7, 48], 10)}},
      {"linear f32, batch and vector", Program.new(y: T.linear(m, w), z: T.silu(T.linear(v, w))),
       %{m: Tensor.random(:f32, [7, 48], 4), v: Tensor.random(:f32, [48], 5)}},
      # bfloat16 storage: widened in the kernels, rows past the table clamp
      {"bf16 weights: linear and gather",
       Program.new(y: T.linear(m, wb), z: T.linear(v, wb),
                   r: T.gather_row(T.const(Tensor.to_bf16(Tensor.random(:f32, [9, 32], 8, scale: 3.0))), T.input(:i, :s32, [4]))),
       %{m: Tensor.random(:f32, [7, 48], 6), v: Tensor.random(:f32, [48], 7), i: Tensor.from_list(:s32, [4], [0, 8, 3, -1])}}
    ]
  end

end
