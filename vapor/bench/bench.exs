# bench/bench.exs — the analytic benchmark, kernel by kernel.
#
# Counted quantities (FLOPs, bytes, hot-loop instructions of the emitted
# code) and the three-roof prediction from *declared* profiles come first;
# observed wall time inside the worker is printed next to them and is
# labelled uncertified. Each row runs `reps` iterations behind one crossing.
alias Vapor.{Arbiter, Program, Tensor}
alias Vapor.Algebra.Term, as: T
alias Vapor.Compile.Lower
alias Vapor.Quant.Sb4
alias Vapor.Runtime.{Dispatch, Substrates}

host = Substrates.get(:host) || raise "build the worker first: make native"
p = Arbiter.default_profiles().native
reps = 32

cases = [
  {"gemv sb4 4096×4096", fn ->
     w = T.const(Sb4.quantize(Tensor.random(:f32, [4096, 4096], 1)))
     x = T.input(:x, :f32, [4096])
     {Program.new([y: T.qgemv(w, x), x_next: x], state: [x: :x_next]), %{x: Tensor.random(:f32, [4096], 2)}}
   end},
  {"ew fma 1M (fused region)", fn ->
     a = T.input(:a, :f32, [1_048_576])
     b = T.const(Tensor.random(:f32, [1_048_576], 3))
     {Program.new([a_next: T.fma(a, b, T.splat(0.5))], state: [a: :a_next]), %{a: Tensor.random(:f32, [1_048_576], 4)}}
   end},
  {"gemm i8 64×1024·1024ᵀ", fn ->
     a = T.input(:a, :s8, [64, 1024])
     w = T.const(Tensor.random(:s8, [1024, 1024], 5))
     {Program.new([c: T.gemm_i8(a, w), a_next: a], state: [a: :a_next]), %{a: Tensor.random(:s8, [64, 1024], 6)}}
   end}
]

IO.puts(String.pad_trailing("kernel", 28) <> "   FLOP/it      B/it   instr/it   predicted  observed(uncertified)")

for {name, build} <- cases do
  {prog, env} = build.()
  {:ok, c} = Lower.lower(prog)
  work = Arbiter.work(c, %{})
  predicted = Arbiter.predict(p, work, reps) * 1.0e3
  {:ok, r} = Dispatch.run_on(host, c, env, iterations: reps, sequence: [], stream: false)
  observed = r.elapsed_ns / 1.0e6
  sum = fn k -> work |> Enum.map(&Map.fetch!(&1, k)) |> Enum.sum() end

  row = :io_lib.format("~10.3e~10.3e~11.3e ~8.3f ms  ~8.3f ms",
                       [sum.(:flops) * 1.0, sum.(:bytes) * 1.0, sum.(:instructions) * 1.0, predicted, observed])
  IO.puts(String.pad_trailing(name, 28) <> IO.iodata_to_binary(row))
end
