defmodule Vapor.Bench do
  @moduledoc """
  The measurement apparatus (phase P6). `run/1` measures and writes
  `BENCH.md`, `roofline.svg` and `engine.svg` into a directory:

    * the machine: CPU, cores, ISA extensions, available event counters;
    * kernels, per host ISA (AVX2, AVX-512), 1 thread and all cores: time inside the worker (minimum of
      repeated steps of a resident session), counted FLOPs and bytes
      (`Vapor.Arbiter.work/2`), achieved GFLOP/s and GB/s, the arbiter's
      prediction, CPU time from the task-clock counter, speedup;
    * the roofline: memory roof *measured* (a streaming kernel), compute
      roof the declared peak of the native profile;
    * the engine: decode throughput against concurrent sequences, threads
      and ISA (continuous batching), prefill throughput;
    * accuracy: ULP histograms of the canonical functions against binary64;
    * the tokenizer: encoding speed on a real vocabulary (if fixtures exist).

  Every figure is measured when the report is generated; nothing is copied
  from elsewhere. Wall-clock numbers depend on the machine and its load.
  """
  alias Vapor.{Arbiter, Canon, Engine, F32, Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Bench.SVG
  alias Vapor.Compile.Lower
  alias Vapor.Model.Config
  alias Vapor.Runtime.{Session, Substrates, Worker}

  @reps 15

  def run(opts \\ []) do
    out = Keyword.get(opts, :out, "docs/bench")
    File.mkdir_p!(out)
    cores = System.schedulers_online()
    threads = Enum.uniq([1, cores])
    log = Keyword.get(opts, :log, &IO.puts/1)

    log.("machine…")
    machine = machine()
    log.("kernels…")
    isas = Substrates.host_isas()
    kernels = for {name, prog, env} <- kernel_cases(), do: kernel(name, prog, env, threads, isas)
    bw = kernels |> Enum.find(&(&1.name =~ "stream")) |> Map.fetch!(:rows) |> Enum.map(& &1.gbs) |> Enum.max()
    peak = machine.peak
    log.("engine…")
    engine = engine(threads, isas, Keyword.get(opts, :engine_requests, [1, 2, 4, 8]))
    log.("accuracy…")
    ulps = ulp_histograms(Keyword.get(opts, :samples, 20_000))
    tok = tokenizer_speed()

    points =
      for %{name: n, rows: rows} <- kernels, r = List.last(rows), r.flops > 0,
          do: {"#{n} (#{isa_name(r.isa)}, #{r.threads} thr)", r.flops / max(r.bytes, 1), r.gflops}

    File.write!(Path.join(out, "roofline.svg"), SVG.roofline("Kernels against the roofline (this machine)", points, bw, peak))

    File.write!(Path.join(out, "engine.svg"),
                SVG.lines("Decode throughput, continuous batching", "concurrent sequences", "tokens/s",
                          engine.decode |> Enum.map(& &1.requests) |> Enum.uniq(),
                          for(i <- isas, t <- threads, do: {"#{isa_name(i)}, #{t} thread#{if t > 1, do: "s"}", for(r <- engine.decode, r.threads == t and r.isa == i, do: {r.requests, r.tps})})))

    File.write!(Path.join(out, "BENCH.md"), report(machine, kernels, bw, peak, engine, ulps, tok))
    log.("wrote #{out}/BENCH.md, roofline.svg, engine.svg")
    :ok
  end

  # --------------------------------------------------------------- machine --

  defp machine do
    info = File.read!("/proc/cpuinfo")
    model = Regex.run(~r/model name\s*:\s*(.+)/, info, capture: :all_but_first) |> List.first() || "unknown"
    flags = Regex.run(~r/flags\s*:\s*(.+)/, info, capture: :all_but_first) |> List.first("") |> String.split()
    isa = Enum.filter(~w(avx2 fma avx512f avx512bw avx512vl avx512_bf16 avx512_fp16 amx_tile), &(&1 in flags))
    {:ok, w} = Worker.start_link(exec: [Substrates.binary("vapor-worker", "native")])
    c = Vapor.Compile.Lower.lower(Program.new(y: T.neg(T.input(:x, :f32, [16])))) |> elem(1)
    {:ok, r} = Vapor.Runtime.Native.run(w, c, %{x: Tensor.random(:f32, [16], 1)}, isa: Substrates.host_isa(), mode: :native)

    mhz = Regex.run(~r/cpu MHz\s*:\s*([0-9.]+)/, info, capture: :all_but_first) |> List.first("2000") |> Float.parse() |> elem(0)

    # theoretical f32 peak of the widest code vapor emits here: two FMA
    # pipes × lanes (8 AVX2, 16 AVX-512) × 2 FLOP per cycle, every core
    lanes = if Substrates.best_isa() == :x86_64_avx512, do: 16, else: 8
    %{cpu: model, cores: System.schedulers_online(), isa: isa, otp: System.otp_release(), elixir: System.version(),
      ghz: mhz / 1000, peak: 4 * lanes * mhz / 1000 * System.schedulers_online(), flop_cycle: 4 * lanes,
      best: isa_name(Substrates.best_isa()),
      counters: Map.keys(r.counters) |> Enum.sort(), date: Date.utc_today() |> Date.to_iso8601()}
  end

  # --------------------------------------------------------------- kernels --

  defp kernel_cases do
    x = fn n -> T.input(:x, :f32, [n]) end
    rows = fn b, n -> T.input(:x, :f32, [b, n]) end
    s = 1024
    {h, hkv, dh} = {8, 4, 64}
    q = T.input(:q, :f32, [1, h * dh])
    pos = T.input(:pos, :s32, [1])
    kc = T.input(:k, :f32, [s, hkv * dh])
    vc = T.input(:v, :f32, [s, hkv * dh])
    att_env = %{q: Tensor.random(:f32, [1, h * dh], 1), pos: Tensor.from_list(:s32, [1], [s - 1]),
                k: Tensor.random(:f32, [s, hkv * dh], 2), v: Tensor.random(:f32, [s, hkv * dh], 3)}
    table = T.input(:table, :s32, [1, div(s, 16)])
    slot = T.input(:slot, :s32, [1])
    perm = Enum.shuffle(0..(div(s, 16) - 1))

    [
      {"stream y = x + 0.5 (16 M)", Program.new(y: T.add(x.(16_777_216), T.splat(0.5))), %{x: fast(16_777_216)}},
      {"GEMV f32 2048²", Program.new(y: T.linear(rows.(1, 2048), T.const(fast([2048, 2048])))), %{x: Tensor.random(:f32, [1, 2048], 5)}},
      {"GEMV bf16 2048²", Program.new(y: T.linear(rows.(1, 2048), T.const(Tensor.to_bf16(fast([2048, 2048]))))), %{x: Tensor.random(:f32, [1, 2048], 5)}},
      {"linear f32 512² × 64 rows", Program.new(y: T.linear(rows.(64, 512), T.const(fast([512, 512])))), %{x: Tensor.random(:f32, [64, 512], 6)}},
      {"GEMV sb4 1024×4096", Program.new(y: T.qgemv(T.const(Vapor.Quant.Sb4.quantize(fast([1024, 4096]))), rows.(1, 4096))),
       %{x: Tensor.random(:f32, [1, 4096], 7)}},
      {"x·silu(x) fused (4 M)", Program.new(y: T.mul(x.(4_194_304), T.silu(x.(4_194_304)))), %{x: fast(4_194_304)}},
      {"attention S=1024 (8 h, 4 kv, dh 64)", Program.new(y: T.attention(q, kc, vc, pos, h, hkv)), att_env},
      {"attention paged S=1024, page 16", Program.new(y: T.attention_paged(q, kc, vc, table, slot, pos, h, hkv, 16)),
       Map.merge(att_env, %{table: Tensor.from_list(:s32, [1, div(s, 16)], perm), slot: Tensor.from_list(:s32, [1], [0])})},
      {"sample V=32000 (T=1)", Program.new(y: T.sample(T.input(:l, :f32, [1, 32_000]), T.input(:p, :f32, [1, 2]))),
       %{l: Tensor.random(:f32, [1, 32_000], 8, scale: 5.0), p: Tensor.from_list(:f32, [1, 2], [1.0, 0.5])}}
    ]
  end

  # uniform [-1, 1) f32 from random bits (fast for large tensors)
  defp fast(n) when is_integer(n), do: fast([n])

  defp fast(shape) do
    n = Enum.product(shape)
    data = for <<b::32 <- :rand.bytes(4 * n)>>, into: <<>>, do: <<(Bitwise.band(b, 0x7F_FFFF) - 4_194_304) / 4_194_304::float-32-little>>
    Tensor.new(:f32, shape, data)
  end

  defp isa_name(:x86_64), do: "AVX2"
  defp isa_name(:x86_64_avx512), do: "AVX-512"
  defp isa_name(i), do: Atom.to_string(i)

  defp kernel(name, prog, env, threads, isas) do
    {:ok, c} = Lower.lower(prog)
    {:ok, dims} = Vapor.Compiled.dims(c, env)
    work = Arbiter.work(c, dims)
    flops = work |> Enum.map(& &1.flops) |> Enum.sum()
    bytes = work |> Enum.map(& &1.bytes) |> Enum.sum()
    predicted = Arbiter.predict(Arbiter.default_profiles().native, work) * 1.0e3

    rows =
      for isa <- isas, t <- threads do
        {:ok, w} = Worker.start_link(exec: [Substrates.binary("vapor-worker", "native")], threads: t)
        {:ok, s} = Session.open(w, c, isa: isa)
        # inputs are written once: later steps find them in place (rewriting
        # them every step would measure the copy and its cache placement)
        _ = Session.step(s, env, [])
        runs = for _ <- 1..@reps, do: elem(Session.step(s, %{}, []), 2)
        best = Enum.min_by(runs, & &1.elapsed_ns)
        Session.close(s)
        GenServer.stop(w)
        sec = best.elapsed_ns / 1.0e9

        %{isa: isa, threads: t, ms: sec * 1.0e3, gflops: flops / sec / 1.0e9, gbs: bytes / sec / 1.0e9, flops: flops, bytes: bytes,
          cpu_ms: Map.get(best.counters, :task_clock_ns, 0) / 1.0e6}
      end

    %{name: name, rows: rows, predicted_ms: predicted}
  end

  # ---------------------------------------------------------------- engine --

  defp engine(threads, isas, requests) do
    {:ok, c} =
      Config.from_map(%{"model_type" => "llama", "vocab_size" => 32_000, "hidden_size" => 256, "intermediate_size" => 768,
                        "num_hidden_layers" => 4, "num_attention_heads" => 8, "num_key_value_heads" => 4,
                        "max_position_embeddings" => 1024, "rms_norm_eps" => 1.0e-5, "tie_word_embeddings" => true})

    ws = model_weights(c)
    prompt = fn i -> for j <- 1..16, do: rem(i * 7919 + j * 104_729, 32_000) end

    decode =
      for isa <- isas, t <- threads, n <- requests do
        {:ok, e} = Engine.start_link(config: c, weights: ws, max_seq: 128, page: 16, sequences: max(n, 1), step_tokens: 64, threads: t, isa: isa)
        _ = Engine.complete(e, prompt.(0), max_tokens: 4)
        t0 = System.monotonic_time()
        tasks = for i <- 1..n, do: Task.async(fn -> Engine.complete(e, prompt.(i), max_tokens: 48, temperature: 0.8, seed: i) end)
        toks = tasks |> Enum.map(&Task.await(&1, 600_000)) |> Enum.map(fn {:ok, ids, _, _} -> length(ids) end) |> Enum.sum()
        dt = System.convert_time_unit(System.monotonic_time() - t0, :native, :microsecond) / 1.0e6
        GenServer.stop(e)
        %{isa: isa, threads: t, requests: n, tokens: toks, seconds: dt, tps: toks / dt}
      end

    # data parallelism instead of intra-op threads: one replica per core,
    # one thread each, at the widest ISA
    cores = Enum.max(threads)

    # bfloat16 weight storage: same tokens as f32 over the rounded weights,
    # half the bytes per step
    storage =
      for st <- [:f32, :bf16], n <- [1, Enum.max(requests)] do
        isa = List.last(isas)
        {:ok, e} = Engine.start_link(config: c, weights: ws, max_seq: 128, page: 16, sequences: n, step_tokens: 64,
                                     threads: cores, isa: isa, storage: st)
        _ = Engine.complete(e, prompt.(0), max_tokens: 4)
        t0 = System.monotonic_time()
        tasks = for i <- 1..n, do: Task.async(fn -> Engine.complete(e, prompt.(i), max_tokens: 48, temperature: 0.8, seed: i) end)
        toks = tasks |> Enum.map(&Task.await(&1, 600_000)) |> Enum.map(fn {:ok, ids, _, _} -> length(ids) end) |> Enum.sum()
        dt = System.convert_time_unit(System.monotonic_time() - t0, :native, :microsecond) / 1.0e6
        GenServer.stop(e)
        %{storage: st, isa: isa, threads: cores, requests: n, tokens: toks, seconds: dt, tps: toks / dt}
      end

    replicas =
      for n <- requests, cores > 1, n >= cores do
        {:ok, e} = Engine.start_link(config: c, weights: ws, max_seq: 128, page: 16, sequences: max(div(n, cores), 1), step_tokens: 64,
                                     threads: cores, replicas: cores)
        _ = Engine.complete(e, prompt.(0), max_tokens: 4)
        t0 = System.monotonic_time()
        tasks = for i <- 1..n, do: Task.async(fn -> Engine.complete(e, prompt.(i), max_tokens: 48, temperature: 0.8, seed: i) end)
        toks = tasks |> Enum.map(&Task.await(&1, 600_000)) |> Enum.map(fn {:ok, ids, _, _} -> length(ids) end) |> Enum.sum()
        dt = System.convert_time_unit(System.monotonic_time() - t0, :native, :microsecond) / 1.0e6
        GenServer.stop(e)
        %{replicas: cores, requests: n, tokens: toks, seconds: dt, tps: toks / dt}
      end

    prefill =
      for isa <- isas, t <- threads do
        {:ok, e} = Engine.start_link(config: c, weights: ws, max_seq: 512, page: 16, sequences: 1, step_tokens: 256, threads: t, isa: isa)
        _ = Engine.complete(e, prompt.(0), max_tokens: 1)
        long = for j <- 1..256, do: rem(j * 31, 32_000)
        t0 = System.monotonic_time()
        {:ok, _, _, _} = Engine.complete(e, long, max_tokens: 1)
        dt = System.convert_time_unit(System.monotonic_time() - t0, :native, :microsecond) / 1.0e6
        GenServer.stop(e)
        %{isa: isa, threads: t, tokens: 256, seconds: dt, tps: 256 / dt}
      end

    %{config: c, decode: decode, prefill: prefill, replicas: replicas, storage: storage}
  end

  defp model_weights(c) do
    {d, qw, kvw, ff} = {c.hidden, c.heads * c.head_dim, c.kv_heads * c.head_dim, c.intermediate}
    scale = fn t, s -> Tensor.new(:f32, t.shape, for(<<x::float-32-little <- t.data>>, into: <<>>, do: <<x * s::float-32-little>>)) end
    ones = fn n -> Tensor.from_list(:f32, [n], List.duplicate(1.0, n)) end

    Map.merge(
      %{"model.embed_tokens.weight" => scale.(fast([c.vocab, d]), 0.05), "model.norm.weight" => ones.(d)},
      Map.new(
        for l <- 0..(c.layers - 1),
            {n, shape} <- [{"self_attn.q_proj.weight", [qw, d]}, {"self_attn.k_proj.weight", [kvw, d]},
                           {"self_attn.v_proj.weight", [kvw, d]}, {"self_attn.o_proj.weight", [d, qw]},
                           {"mlp.gate_proj.weight", [ff, d]}, {"mlp.up_proj.weight", [ff, d]}, {"mlp.down_proj.weight", [d, ff]},
                           {"input_layernorm.weight", [d]}, {"post_attention_layernorm.weight", [d]}],
            do: {"model.layers.#{l}.#{n}", if(length(shape) == 1, do: ones.(d), else: scale.(fast(shape), 1 / :math.sqrt(List.last(shape))))}
      )
    )
  end

  # -------------------------------------------------------------- accuracy --

  defp ulp_histograms(n) do
    :rand.seed(:exsss, {1, 2, 3})
    key = fn b -> if Bitwise.band(b, 0x8000_0000) != 0, do: -Bitwise.band(b, 0x7FFF_FFFF), else: b end

    cases = [
      {:exp, "e^x, x ∈ [−87, 88]", fn -> -87 + :rand.uniform() * 175 end, &:math.exp/1},
      {:rcp, "1/x, |x| ∈ [2⁻¹⁰⁰, 2¹⁰⁰]", fn -> (if :rand.uniform() < 0.5, do: -1, else: 1) * :math.pow(2, -100 + :rand.uniform() * 200) end, &(1 / &1)},
      {:rsqrt, "1/√x, x ∈ [2⁻¹²⁰, 2¹²⁰]", fn -> :math.pow(2, -120 + :rand.uniform() * 240) end, &(1 / :math.sqrt(&1))},
      {:sigmoid, "σ(x), x ∈ [−40, 40]", fn -> -40 + :rand.uniform() * 80 end, &(1 / (1 + :math.exp(-&1)))},
      {:silu, "x·σ(x), x ∈ [−40, 40]", fn -> -40 + :rand.uniform() * 80 end, &(&1 / (1 + :math.exp(-&1)))}
    ]

    for {f, label, gen, ref} <- cases do
      run = Canon.compile(f)

      hist =
        Enum.reduce(1..n, %{}, fn _, h ->
          xb = F32.from_float(gen.())
          u = abs(key.(run.([xb])) - key.(F32.from_float(ref.(F32.to_float(xb)))))
          Map.update(h, min(u, 3), 1, &(&1 + 1))
        end)

      {label, n, for(k <- 0..3, do: Map.get(hist, k, 0))}
    end
  end

  defp tokenizer_speed do
    path = "test/fixtures/vocab/ggml-vocab-llama-bpe.gguf"

    if File.exists?(path) do
      {:ok, g} = Vapor.Ingest.GGUF.read(path)
      {:ok, tk} = Vapor.Tokenizer.from_gguf(g.metadata)
      text = String.duplicate(File.read!("README.md"), 8)
      _ = Vapor.Tokenizer.encode(tk, String.slice(text, 0, 2000))
      {us, ids} = :timer.tc(fn -> Vapor.Tokenizer.encode(tk, text) end)
      %{bytes: byte_size(text), tokens: length(ids), tps: length(ids) / (us / 1.0e6)}
    end
  end

  # ---------------------------------------------------------------- report --

  defp report(m, kernels, bw, peak, engine, ulps, tok) do
    f = &SVG.fmt/1

    kernel_rows =
      Enum.flat_map(kernels, fn %{name: n, rows: rows, predicted_ms: p} ->
        base = hd(rows).ms

        Enum.map(rows, fn r ->
          "| #{n} | #{isa_name(r.isa)} | #{r.threads} | #{f.(r.ms)} | #{f.(r.cpu_ms)} | #{f.(r.gflops)} | #{f.(r.gbs)} | #{f.(base / r.ms)}× | #{f.(p)} |"
        end)
      end)

    decode_rows = for r <- engine.decode, do: "| #{isa_name(r.isa)} | #{r.threads} | #{r.requests} | #{r.tokens} | #{f.(r.seconds)} | #{f.(r.tps)} |"
    storage_rows = for r <- engine.storage, do: "| #{r.storage} | #{isa_name(r.isa)} | #{r.threads} | #{r.requests} | #{r.tokens} | #{f.(r.seconds)} | #{f.(r.tps)} |"
    replica_rows = for r <- engine.replicas, do: "| #{r.replicas} × 1 | #{r.requests} | #{r.tokens} | #{f.(r.seconds)} | #{f.(r.tps)} |"
    prefill_rows = for r <- engine.prefill, do: "| #{isa_name(r.isa)} | #{r.threads} | #{r.tokens} | #{f.(r.seconds * 1000)} | #{f.(r.tps)} |"

    ulp_rows =
      for {label, n, [u0, u1, u2, u3]} <- ulps,
          do: "| #{label} | #{n} | #{Float.round(100 * u0 / n, 2)} % | #{Float.round(100 * u1 / n, 2)} % | #{Float.round(100 * u2 / n, 2)} % | #{Float.round(100 * u3 / n, 2)} % |"

    c = engine.config

    """
    # Medições — vapor

    Gerado por `mix vapor.bench` em #{m.date}. Tudo abaixo foi medido nesta
    máquina, na hora; tempos de parede dependem da carga do host.

    ## Máquina

    | | |
    |---|---|
    | CPU | #{m.cpu} |
    | núcleos (BEAM schedulers) | #{m.cores} |
    | extensões relevantes | #{Enum.join(m.isa, ", ")} |
    | contadores de eventos disponíveis | #{if m.counters == [], do: "nenhum", else: Enum.join(m.counters, ", ")} |
    | OTP / Elixir | #{m.otp} / #{m.elixir} |

    Contadores de hardware (ciclos, instruções, cache) só aparecem quando o
    kernel expõe uma PMU; aqui #{if :cycles in m.counters, do: "aparecem", else: "não há PMU (VM) — o tempo de CPU vem do contador de software task-clock"}.

    ## Kernels

    Tempo dentro do worker (mínimo de #{@reps} passos de uma sessão residente,
    entradas já residentes),
    trabalho contado pelo árbitro, previsão do perfil nativo declarado.
    Resultados bit-idênticos para qualquer número de threads e entre AVX2 e
    AVX-512 (testado); a aceleração é relativa à primeira linha (AVX2, 1 thread).

    | kernel | ISA | threads | ms | CPU ms | GFLOP/s | GB/s | aceleração | previsto ms |
    |---|---|---|---|---|---|---|---|---|
    #{Enum.join(kernel_rows, "\n")}

    ![roofline](roofline.svg)

    Teto de memória medido: **#{f.(bw)} GB/s** (kernel de streaming, todos os
    núcleos); teto de cômputo teórico do código #{m.best} emitido: #{m.flop_cycle} FLOP/ciclo ×
    #{f.(m.ghz)} GHz × #{m.cores} núcleos = #{f.(peak)} GFLOP/s (duas unidades FMA; a política
    canônica arredonda produto e soma separadamente, então seu teto é metade).

    ## Motor (lote contínuo, KV paginado)

    Modelo Llama de pesos aleatórios: vocabulário #{c.vocab}, largura #{c.hidden},
    #{c.layers} camadas, #{c.heads} cabeças (#{c.kv_heads} KV), f32; prompts de 16 tokens,
    48 tokens gerados por requisição, amostragem no substrato (T = 0,8).

    | ISA | threads | requisições simultâneas | tokens | s | tokens/s |
    |---|---|---|---|---|---|
    #{Enum.join(decode_rows, "\n")}

    ![engine](engine.svg)

    Armazenamento dos pesos em bfloat16 (`storage: :bf16`: metade dos bytes
    lidos por passo; mesmos bits que o programa f32 sobre os pesos arredondados — testado):

    | pesos | ISA | threads | requisições simultâneas | tokens | s | tokens/s |
    |---|---|---|---|---|---|---|
    #{Enum.join(storage_rows, "\n")}

    Paralelismo de dados em vez de threads intra-operação (`replicas:`, um
    motor por núcleo com uma thread cada, uma compilação e as mesmas páginas
    de pesos; mesmos tokens por requisição — testado):

    | réplicas × threads | requisições simultâneas | tokens | s | tokens/s |
    |---|---|---|---|---|
    #{Enum.join(replica_rows, "\n")}

    Prefill (um prompt de 256 tokens, primeiro token):

    | ISA | threads | tokens | ms | tokens/s |
    |---|---|---|---|---|
    #{Enum.join(prefill_rows, "\n")}

    ## Exatidão das funções canônicas (ULP contra binary64)

    | função | amostras | 0 ULP | 1 ULP | 2 ULP | ≥ 3 ULP |
    |---|---|---|---|---|---|
    #{Enum.join(ulp_rows, "\n")}

    ## Tokenizador

    #{if tok, do: "Vocabulário do Llama 3 (128 256 tokens), README × 8: #{tok.bytes} bytes → #{tok.tokens} tokens, **#{f.(tok.tps / 1000)} mil tokens/s** num núcleo da BEAM.", else: "(sem `make fixtures`: não medido)"}
    """
  end
end
