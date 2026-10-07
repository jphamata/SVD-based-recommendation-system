# examples/ssm_block.exs — an affine state-space block, 4-bit quantised,
# from term to certified machine code to resilient streaming inference.
#
#     mix run examples/ssm_block.exs
#
#   u_t = W_in · x_t              (sb4 GEMV, d_state × d_model)
#   h_t = a ⊙ h_{t−1} + u_t       (the σ = 2 affine monoid, fused elementwise)
#   y_t = W_out · h_t             (sb4 GEMV, d_model × d_state)
alias Vapor.{Arbiter, Bundle, Certificate, Program, Tensor}
alias Vapor.Algebra.Term, as: T
alias Vapor.Quant.Sb4
alias Vapor.Runtime.{Dispatch, Fabric, Native, Substrates}
alias Vapor.Verify.{Envelope, Digest}

d_model = 1024
d_state = 512
tokens = 64

line = fn -> IO.puts(String.duplicate("─", 78)) end
section = fn t -> line.(); IO.puts(t); line.() end

# ---------------------------------------------------------------- model --
section.("1. Model: affine SSM block, weights quantised to :sb4 (4.6875 bit/w storage)")
w_in = Sb4.quantize(Tensor.random(:f32, [d_state, d_model], 11))
w_out = Sb4.quantize(Tensor.random(:f32, [d_model, d_state], 12))
decay = Tensor.random(:f32, [d_state], 13, scale: 0.5)
x = T.input(:x, :f32, [d_model])
h = T.input(:h, :f32, [d_state])
h_next = T.fma(T.const(decay), h, T.qgemv(T.const(w_in), x))
program = Program.new([y: T.qgemv(T.const(w_out), h_next), h_next: h_next], state: [h: :h_next])

f32_bytes = 4 * (d_model * d_state * 2)
sb4_bytes = byte_size(w_in.data) + byte_size(w_out.data)
IO.puts("weights: #{2 * d_model * d_state} params, #{sb4_bytes} B as :sb4 vs #{f32_bytes} B as f32 (#{Float.round(f32_bytes / sb4_bytes, 2)}× smaller)")

# ------------------------------------------------------------ certify --
section.("2. Certify: the six-rung ladder on every substrate this machine has")
IO.puts("substrates: #{Substrates.list() |> Enum.map(& &1.id) |> inspect()}")
[k_a, k_b, k_c] = for _ <- 1..3, do: Certificate.keygen()
{t_cert, {:ok, compiled}} = :timer.tc(fn -> Vapor.compile(program, key: k_a) end)
p = compiled.certificate.payload
IO.puts("certified in #{div(t_cert, 1000)} ms (policy #{p.policy}, #{p.rewrites} exact rewrites, regions #{inspect(p.regions)})")

for {isa, ks} <- p.admission.registers do
  desc = ks |> Enum.map(fn {k, v} -> "#{String.slice(k, 0, 12)}: g=#{v.group_factor} #{v.bytes}B" end) |> Enum.join(", ")
  IO.puts("  rung 2  #{isa}: allocation accepted by the Lean-extracted checker — #{desc}")
end

for a <- p.adjoint, do: IO.puts("  rung 3  ⟨Wx,v⟩ = ⟨x,Wᵀv⟩ for #{a.rows}×#{a.k} qgemv on #{a.substrate}: #{a.identity}")
IO.puts("  rung 4  every CPU substrate bit-identical to the exact-arithmetic oracle")
IO.puts("  rung 5  parity across #{inspect(p.parity.substrates)}: #{inspect(p.parity.bit_identical)}")
IO.puts("          certified max |error| vs exact math: #{inspect(p.parity.envelope_max_abs_error)}")

# quorum: an independent node recompiles and co-signs the identical payload
{:ok, independent} = Vapor.compile(program, key: k_b)
{:ok, cert} = Certificate.cosign(compiled.certificate, independent.certificate, k_b)
compiled = %{compiled | certificate: cert}
IO.puts("  rung 6  Ed25519 signatures: #{length(cert.signatures)} (independent payloads identical: #{Certificate.canonical(independent.certificate) == Certificate.canonical(cert)})")

# ------------------------------------------------------------- deploy --
section.("3. Deploy: ship the bundle; the edge verifies signatures + code hashes, no ladder")
bundle = Bundle.pack(compiled)
{t_edge, {:ok, edge}} = :timer.tc(fn -> Bundle.unpack(bundle, [k_a.public, k_b.public, k_c.public], 2) end)
IO.puts("bundle #{byte_size(bundle)} B verified (2-of-3 quorum) in #{Float.round(t_edge / 1000, 2)} ms at the edge")

# -------------------------------------------------------------- infer --
section.("4. Infer: #{tokens} tokens, one BEAM crossing, streamed per token")
env = %{x: Tensor.random(:f32, [tokens, d_model], 21), h: Tensor.from_list(:f32, [d_state], List.duplicate(0.0, d_state))}
parent = self()
{:ok, res, trace} = Vapor.run(edge, env, on_emit: fn t, outs -> if t < 3, do: send(parent, {:tok, t, outs}) end)

for _ <- 1..3 do
  receive do
    {:tok, t, outs} ->
      [y0 | _] = Tensor.to_floats(outs[:y])
      IO.puts("  token #{t}: y[0] = #{Float.round(y0, 6)}  (streamed before the sequence finished)")
  end
end

IO.puts("ran on #{res.substrate}; trace #{inspect(trace)}")

# ----------------------------------------------------------- analytic --
section.("5. Analytic benchmark: counted work, declared hardware, predicted vs observed")
work = Arbiter.work(edge, %{})
profiles =
  case Substrates.get(:fabric) do
    nil -> %{native: Arbiter.default_profiles().native}
    %{server: f} -> %{native: Arbiter.default_profiles().native, fabric: Arbiter.fabric_profile(Fabric.info(f))}
  end
d = Arbiter.decide(work, tokens, profiles)
bytes_tok = Enum.sum(Enum.map(work, & &1.bytes))
flops_tok = Enum.sum(Enum.map(work, & &1.flops))
instr_tok = Enum.sum(Enum.map(work, & &1.instructions))
IO.puts("per token (counted): #{flops_tok} FLOP, #{bytes_tok} B, #{instr_tok} hot-loop instructions (#{Vapor.Runtime.Substrates.host_isa()})")
IO.puts("intensity #{Float.round(d.intensity, 3)} FLOP/B; arbiter (three-roof, declared profiles) → #{d.target}")
for {id, t} <- Enum.sort(d.predicted_s), do: IO.puts("  predicted #{String.pad_trailing(to_string(id), 27)} #{:io_lib.format("~9.3f", [t * 1.0e3])} ms")

observe = fn sub ->
  {:ok, r} = Dispatch.run_on(sub, edge, env, iterations: tokens, sequence: [:x])
  r.elapsed_ns / 1.0e6
end

for sub <- Substrates.list(), sub.kind != :oracle do
  ms = observe.(sub)
  gbs = bytes_tok * tokens / (ms / 1.0e3) / 1.0e9
  label = "observed(uncertified) #{sub.id}"
  IO.puts("  #{String.pad_trailing(label, 36)} #{:io_lib.format("~9.3f", [ms])} ms  (#{Float.round(gbs, 2)} GB/s effective)")
end

# ---------------------------------------------------------- envelope --
section.("6. Numerics: bit parity and the certified envelope on this run")
short = %{env | x: Tensor.new(:f32, [8, d_model], binary_part(env.x.data, 0, 8 * d_model * 4))}
{:ok, ref} = Native.run_oracle(edge, short, iterations: 8, sequence: [:x])
{:ok, host} = Dispatch.run_on(Substrates.get(:host), edge, short, iterations: 8, sequence: [:x])
IO.puts("host vs oracle, 8 tokens: #{if host.steps == ref.steps, do: "bit-identical", else: "DIFFERENT"}  (digest y@7 #{Digest.tensor(List.last(host.steps)[:y])})")
bounds = Envelope.bounds(edge, short, iterations: 8, sequence: [:x])
ok = Enum.zip(host.steps, bounds) |> Enum.all?(fn {o, b} -> Envelope.check(b[:y], o[:y]) == :ok end)
IO.puts("every y inside its rigorous a-priori envelope: #{ok}")

# ---------------------------------------------------------- resilience --
section.("7. Resilience: the GPU driver crashes mid-service; the unit is rerouted")

case Substrates.get(:fabric) do
  nil ->
    IO.puts("(no Vulkan device here — the fabric tier is absent, the native tier serves)")

  %{server: f} ->
    {:ok, test_fabric} = Fabric.start_link(exec: [Substrates.binary("vapor-fabric", "native"), "--fault-injection"])
    {:error, crash} = Fabric.inject_fault(test_fabric)
    IO.puts("injected driver fault → #{inspect(crash)}; daemon restarts: #{Fabric.info(test_fabric).restarts}")
    dead = spawn(fn -> :ok end)
    subs = [%{id: :fabric, kind: :fabric, isa: :spirv, mode: :gpu, server: dead} | Enum.reject(Substrates.list(), &(&1.id == :fabric))]
    {:ok, r2, trace2} = Dispatch.run(edge, short, substrates: subs, prefer: :fabric, iterations: 8, sequence: [:x])
    IO.puts("unit rerouted: #{inspect(Enum.map(trace2, fn {id, st} -> {id, if(st == :ok, do: :ok, else: :failed)} end))}")
    IO.puts("rerouted result bit-identical to the oracle: #{r2.steps == ref.steps}")
    _ = f
end

line.()
