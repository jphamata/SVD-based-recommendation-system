defmodule Vapor.Bench.Round08 do
  @moduledoc """
  Measurements for the 0.8 round (`mix vapor.bench --round08`, written to
  `docs/bench/ROUND08.md`): what each new capability costs or saves on this
  machine — and, next to every speed, whether the bits held.

    * GPU sessions: a decode loop on the Vulkan fabric as one-shot RUNs
      (every token re-creates pipelines and moves the KV cache both ways)
      against a resident session (direct and staged memory), and the CPU
      session; the engine's throughput on each substrate;
    * sparse 4-bit experts: a Mixtral-shaped sb4 model, dense against
      predicated dispatch (same output bits checked);

  The GPU here is **lavapipe** (Mesa's Vulkan on the same two CPU cores):
  its times measure the machinery — what a session removes from every
  step — not what a discrete GPU would deliver.
  """
  alias Vapor.Tensor
  alias Vapor.Compile.Lower
  alias Vapor.Model.{Config, Llama}
  alias Vapor.Runtime.{Fabric, Native, Session, Substrates, Worker}

  @reps 5

  def run(opts \\ []) do
    out = Keyword.get(opts, :out, "docs/bench")
    log = Keyword.get(opts, :log, &IO.puts/1)
    File.mkdir_p!(out)
    sections = Keyword.get(opts, :sections, [:gpu, :sb4])

    parts =
      for sec <- sections do
        log.("#{sec}…")
        section(sec, log)
      end

    File.write!(Path.join(out, "ROUND08.md"), header() <> Enum.join(parts, "\n"))
    :ok
  end

  defp header do
    """
    # Medições da rodada 0.8

    Regeneráveis com `mix vapor.bench --round08`. Máquina: #{machine()}.
    A GPU é o **lavapipe** (Vulkan da Mesa nos mesmos núcleos da CPU): os
    tempos de GPU medem a maquinaria — o que a sessão residente tira de
    cada passo —, não o que uma GPU discreta entregaria. Pesos aleatórios:
    medidas da máquina, não da qualidade (essa é `mix vapor.quality`).

    """
  end

  defp machine do
    cpu =
      case File.read("/proc/cpuinfo") do
        {:ok, s} -> Regex.run(~r/model name\s*:\s*(.*)/, s, capture: :all_but_first) |> List.wrap() |> List.first("?")
        _ -> "?"
      end

    "#{cpu}, #{System.schedulers_online()} vCPUs, OTP #{System.otp_release()}"
  end

  defp section(:gpu, _log), do: gpu_report(gpu())
  defp section(:sb4, _log), do: sb4_report(sb4())

  # ------------------------------------------------------------------ GPU --

  @doc false
  def tiny(arch, over) do
    {:ok, c} =
      Config.from_map(Map.merge(%{"model_type" => arch, "vocab_size" => 2048, "max_position_embeddings" => 512,
                                  "rms_norm_eps" => 1.0e-5, "rope_theta" => 10_000.0, "hidden_act" => "silu"}, over))

    ws =
      for {name, shape, kind} <- Llama.expected_weights(c), into: %{} do
        t = Tensor.random(:f32, shape, :erlang.phash2(name), scale: if(kind == :norm, do: 0.05, else: 0.2))
        {name, if(kind == :norm, do: Tensor.from_list(:f32, shape, Enum.map(Tensor.to_floats(t), &(&1 + 1.0))), else: t)}
      end

    {c, ws}
  end

  defp ids(xs), do: Tensor.from_list(:s32, [length(xs)], xs)
  defp median(xs), do: xs |> Enum.sort() |> Enum.at(div(length(xs), 2))

  def gpu do
    fabric_bin = Substrates.binary("vapor-fabric", "native")
    {c, ws} = tiny("llama", %{"hidden_size" => 256, "intermediate_size" => 512, "num_hidden_layers" => 4,
                              "num_attention_heads" => 8, "num_key_value_heads" => 2})
    s = 256
    {:ok, p} = Llama.program(c, ws, max_seq: s)
    {:ok, comp} = Lower.lower(p)
    prompt = Enum.map(1..32, &rem(&1 * 37, 2000))
    steps = 32

    {:ok, w} = Worker.start_link(exec: [Substrates.binary("vapor-worker", "native")], threads: System.schedulers_online())
    {:ok, f} = Fabric.start_link(exec: [fabric_bin])

    session_run = fn server, opts ->
      {:ok, sess} = Session.open(server, comp, opts)
      {:ok, %{logits: l0}, _} = Session.step(sess, %{tok: ids(prompt), pos: ids(Enum.to_list(0..31))}, [:logits])

      {rows, metas} =
        Enum.map_reduce(0..(steps - 1), [], fn i, metas ->
          t0 = System.monotonic_time(:microsecond)
          {:ok, %{logits: l}, m} = Session.step(sess, %{tok: ids([rem(i * 11, 2000)]), pos: ids([32 + i])}, [:logits])
          {l.data, [{System.monotonic_time(:microsecond) - t0, m} | metas]}
        end)

      Session.close(sess)
      metas = Enum.reverse(metas)
      %{rows: [binary_part(l0.data, 31 * c.vocab * 4, c.vocab * 4) | rows],
        ms: median(Enum.map(metas, &elem(&1, 0))) / 1000, bytes: elem(List.last(metas), 1).counters[:gpu_host_bytes],
        reused: Enum.count(metas, fn {_, m} -> m.counters[:gpu_recording_reused] == 1 end)}
    end

    cpu = session_run.(w, isa: Substrates.host_isa())
    direct = session_run.(f, isa: :spirv)
    staged = session_run.(f, isa: :spirv, staging: true)

    # the same loop as one-shot RUNs: caches cross both ways every token
    caches0 = Llama.empty_caches(c, s)
    {:ok, r0} = Fabric.run(f, comp, Map.merge(caches0, %{tok: ids(prompt), pos: ids(Enum.to_list(0..31))}))
    names = for l <- 0..(c.layers - 1), n <- Llama.cache_names(c, l), do: n
    next = fn r -> Map.new(names, &{&1, r.outputs[:"#{&1}_next"]}) end

    {run_rows, run_times} =
      Enum.map_reduce(0..(steps - 1), {next.(r0), []}, fn i, {caches, ts} ->
        t0 = System.monotonic_time(:microsecond)
        {:ok, r} = Fabric.run(f, comp, Map.merge(caches, %{tok: ids([rem(i * 11, 2000)]), pos: ids([32 + i])}))
        {r.outputs.logits.data, {next.(r), [System.monotonic_time(:microsecond) - t0 | ts]}}
      end)
      |> then(fn {rows, {_, ts}} -> {rows, ts} end)

    run_bytes = Enum.sum(for n <- names, do: 2 * byte_size(caches0[n].data)) + 8 + c.vocab * 4
    run = %{ms: median(run_times) / 1000, bytes: run_bytes,
            rows: [binary_part(r0.outputs.logits.data, 31 * c.vocab * 4, c.vocab * 4) | run_rows]}

    engine = engine_bench(c, ws, f)
    GenServer.stop(f)

    %{cpu: cpu, direct: direct, staged: staged, run: run, steps: steps,
      same_bits: cpu.rows == direct.rows and direct.rows == staged.rows and staged.rows == run.rows, engine: engine,
      shape: "Llama, largura 256, 4 camadas, 8 cabeças (2 KV), vocabulário 2048, contexto 256"}
  end

  defp engine_bench(c, ws, f) do
    reqs = for i <- 1..4, do: {Enum.map(1..24, &rem(&1 * (7 + i), 2000)), [max_tokens: 32, temperature: 0.0]}
    base = [config: c, weights: ws, max_seq: 128, page: 16, sequences: 4, step_tokens: 64]

    for {label, extra} <- [{"CPU (worker nativo)", [threads: System.schedulers_online()]}, {"GPU (sessão residente)", [isa: :spirv, fabric: f]}] do
      {:ok, e} = Vapor.Engine.start_link(base ++ extra)
      t0 = System.monotonic_time(:microsecond)
      refs = for {pr, o} <- reqs, do: elem(Vapor.Engine.generate(e, pr, o), 1)
      outs = for r <- refs, do: Vapor.Engine.collect(r, [], 600_000)
      us = System.monotonic_time(:microsecond) - t0
      toks = outs |> Enum.map(fn {:ok, ids, _, _} -> length(ids) end) |> Enum.sum()
      GenServer.stop(e)
      %{label: label, tokens: toks, tps: toks / (us / 1.0e6), outs: outs}
    end
  end

  defp gpu_report(g) do
    [e_cpu, e_gpu] = g.engine

    """
    ## 1. Sessões residentes na GPU (P0 da 0.7)

    #{g.shape}; um *prompt* de 32 tokens, depois #{g.steps} passos de um token.
    Tempo de parede por passo, medido na BEAM (inclui o protocolo), mediana:

    | caminho | ms/token | bytes host↔dispositivo por token | gravações reaproveitadas |
    |---|---:|---:|---:|
    | GPU, um `RUN` por token (sem sessão: pipelines e cache KV a cada passo) | #{fmt(g.run.ms)} | #{g.run.bytes} | — |
    | GPU, sessão residente, memória direta | #{fmt(g.direct.ms)} | #{g.direct.bytes} | #{g.direct.reused}/#{g.steps} |
    | GPU, sessão residente, *staging* (caminho de GPU discreta) | #{fmt(g.staged.ms)} | #{g.staged.bytes} | #{g.staged.reused}/#{g.steps} |
    | CPU, sessão no worker nativo | #{fmt(g.cpu.ms)} | — | — |

    Mesmos bits nos quatro caminhos (logits de cada passo): **#{if g.same_bits, do: "sim", else: "NÃO"}**.
    A sessão corta #{fmt(g.run.ms / max(g.direct.ms, 0.001))}× o tempo por token e
    #{round(g.run.bytes / max(g.direct.bytes, 1))}× o tráfego: o passo move os ids e uma linha de logits,
    não o cache.

    **Motor** (4 pedidos simultâneos, 24 tokens de *prompt*, 32 gerados, guloso):

    | substrato | tokens | tokens/s |
    |---|---:|---:|
    | #{e_cpu.label} | #{e_cpu.tokens} | #{fmt(e_cpu.tps)} |
    | #{e_gpu.label} | #{e_gpu.tokens} | #{fmt(e_gpu.tps)} |

    Mesmos tokens nos dois: **#{if e_cpu.outs == e_gpu.outs, do: "sim", else: "NÃO"}**. No lavapipe a GPU
    é a própria CPU, então a comparação de vazão diz pouco sobre GPU real; o que
    ela prova é que o motor serve inteiro no Vulkan, com os bits da CPU.
    """
  end

  # ------------------------------------------------------------- sparse 4-bit --

  def sb4 do
    {c, ws} = tiny("mixtral", %{"vocab_size" => 512, "hidden_size" => 512, "intermediate_size" => 512, "num_hidden_layers" => 2,
                                "num_attention_heads" => 8, "num_key_value_heads" => 2, "num_local_experts" => 8,
                                "num_experts_per_tok" => 2})
    {:ok, w} = Worker.start_link(exec: [Substrates.binary("vapor-worker", "native")], threads: System.schedulers_online())
    isa = Substrates.best_isa()

    rows =
      for t <- [1, 8] do
        toks = Enum.map(1..t, &rem(&1 * 37, 500))
        e = Map.merge(Llama.empty_caches(c, 64), %{tok: ids(toks), pos: ids(Enum.to_list(0..(t - 1)))})

        [{dm, d, dr}, {sm, s, sr}] =
          for mode <- [:dense, :sparse] do
            {:ok, p} = Llama.program(c, ws, max_seq: 64, moe: mode, quantize: :sb4)
            comp = elem(Lower.lower(p), 1)
            runs = for _ <- 1..@reps, do: elem(Native.run(w, comp, e, isa: isa, mode: :native), 1)
            {:ok, emu} = Native.run(w, comp, e, isa: :riscv64, mode: :emulate, vlen: 256)
            {median(Enum.map(runs, & &1.elapsed_ns)) / 1.0e6, hd(runs).outputs, emu.retired}
          end

        %{tokens: t, dense_ms: dm, sparse_ms: sm, dense_retired: dr, sparse_retired: sr, same_bits: d == s}
      end

    %{rows: rows, isa: isa, shape: "Mixtral, largura 512, 8 especialistas (top-2), 2 camadas, pesos sb4 (4,75 bits/peso)"}
  end

  defp sb4_report(r) do
    """
    ## 2. Especialistas esparsos em 4 bits (`qgemv_masked`)

    #{r.shape}; ISA `#{r.isa}`. Tempo no worker (mediana de #{@reps}) e instruções
    retiradas contadas exatamente pelo interpretador RVV (VLEN 256):

    | tokens | denso ms | esparso ms | × | instruções denso | instruções esparso | mesmos bits |
    |---:|---:|---:|---:|---:|---:|:---:|
    #{Enum.map_join(r.rows, "\n", fn x -> "| #{x.tokens} | #{fmt(x.dense_ms)} | #{fmt(x.sparse_ms)} | #{fmt(x.dense_ms / x.sparse_ms)} | #{x.dense_retired} | #{x.sparse_retired} | #{if x.same_bits, do: "sim", else: "NÃO"} |" end)}

    Com top-2 de 8, cada token lê 1/4 dos especialistas; o resto do modelo
    (atenção, roteador, cabeça) não muda — o ganho total é menor que 4×.
    """
  end

  defp fmt(x) when is_float(x), do: :erlang.float_to_binary(x, decimals: 2)
  defp fmt(x), do: to_string(x)
end
