defmodule Vapor.Substrate.Kit do
  @moduledoc """
  **The admission battery as a portable kit** — for devices vapor does not
  drive itself.

  A Tenstorrent card is reached through Tenstorrent's own compiler stack
  (`tt-xla`, a PJRT plugin that compiles StableHLO for Tensix); a TPU or a
  GPU through theirs. `write/1` exports every probe of `Vapor.Substrate`
  that has a StableHLO form (`Vapor.Export.StableHLO`) into a directory —
  module, inputs, manifest — together with `run_kit.py`, which runs them
  on any PJRT device through JAX. `judge/1` reads the outputs back and
  admits the device exactly as a local one: bit for bit against the
  oracle, inside or outside the rigorous envelope, with the fingerprint
  (contraction, flush-to-zero, operand precision, reduction order…).

      mix vapor.substrate kit DIR                  # here
      python3 DIR/run_kit.py DIR --platform tt     # on the Tenstorrent machine
      mix vapor.substrate judge DIR                # here again: the admission

  Probes without a StableHLO form (sampling, the 4-bit GEMV) are listed in
  the kit as not exported; the verdict covers what was run.
  """
  alias Vapor.{JSON, Substrate, Tensor}
  alias Vapor.Export.StableHLO

  @doc "Write the kit into `dir`. Returns `%{exported: [name], skipped: [{name, why}]}`."
  def write(dir) do
    File.mkdir_p!(dir)

    {exported, skipped} =
      Enum.reduce(Substrate.probes(), {[], []}, fn {name, prog, env, _ro}, {ex, sk} ->
        case StableHLO.export(prog, env: env) do
          {:ok, m} ->
            pd = Path.join(dir, to_string(name))
            File.mkdir_p!(pd)
            File.write!(Path.join(pd, "module.mlir"), m.mlir)

            ins =
              for {n, dt, s} <- m.inputs do
                File.write!(Path.join(pd, "in_#{n}.bin"), env[n].data)
                %{name: n, dtype: dt, shape: s, file: "in_#{n}.bin"}
              end

            outs = Enum.map(m.outputs, fn {n, dt, s} -> %{name: n, dtype: dt, shape: s} end)
            File.write!(Path.join(pd, "manifest.json"), JSON.encode(%{inputs: ins, outputs: outs}))
            {[name | ex], sk}

          {:error, r} ->
            {ex, [{name, r.bound} | sk]}
        end
      end)

    exported = Enum.reverse(exported)
    skipped = Enum.reverse(skipped)

    File.write!(Path.join(dir, "kit.json"),
      JSON.encode(%{vapor: to_string(Application.spec(:vapor, :vsn)), semantics: Vapor.Canon.version(),
                    probes: exported, not_exported: Map.new(skipped, fn {n, why} -> {n, why} end)}))

    File.write!(Path.join(dir, "run_kit.py"), run_kit())
    File.write!(Path.join(dir, "stablehlo_run.py"), runner())
    %{exported: exported, skipped: skipped}
  end

  @doc "Admit the device that ran the kit in `dir` (`%Vapor.Substrate.Admission{}`)."
  def judge(dir) do
    {:ok, kit} = JSON.decode(File.read!(Path.join(dir, "kit.json")))
    names = Map.new(Substrate.probes(), fn {n, _, _, _} -> {to_string(n), n} end)

    results =
      for p <- kit["probes"], name = Map.fetch!(names, p), into: %{} do
        pd = Path.join(dir, p)
        {:ok, man} = JSON.decode(File.read!(Path.join(pd, "manifest.json")))

        outs =
          Enum.reduce_while(man["outputs"], %{}, fn o, acc ->
            f = Path.join(pd, "out_#{o["name"]}.bin")

            if File.exists?(f),
              do: {:cont, Map.put(acc, String.to_existing_atom(o["name"]), Tensor.new(String.to_existing_atom(o["dtype"]), o["shape"], File.read!(f)))},
              else: {:halt, :missing}
          end)

        {name, if(outs == :missing, do: {:error, :not_run}, else: {:ok, outs})}
      end

    device =
      case File.read(Path.join(dir, "device.json")) do
        {:ok, b} ->
          case JSON.decode(b) do
            {:ok, d} -> "#{d["kind"]} (#{d["platform"]}, PJRT via JAX #{d["jax"]})"
            _ -> "PJRT device"
          end

        _ ->
          "PJRT device"
      end

    Substrate.admit(%{id: :pjrt}, results: results, device: device)
  end

  # the runner, embedded so that a kit directory is self-contained
  @runner_src File.read!(Path.expand("../../../test/python/stablehlo_run.py", __DIR__))
  @external_resource Path.expand("../../../test/python/stablehlo_run.py", __DIR__)
  defp runner, do: @runner_src

  defp run_kit do
    ~S'''
    """Run vapor's admission kit on a PJRT device (JAX): every probe directory
    holds a StableHLO module, its inputs and a manifest; outputs are written
    next to them. Then, on the machine with vapor: mix vapor.substrate judge DIR

    usage: python3 run_kit.py DIR [--platform cpu|tt|tpu|cuda|METAL]
    """
    import json, os, subprocess, sys
    d = sys.argv[1]
    extra = sys.argv[2:]
    kit = json.load(open(os.path.join(d, "kit.json")))
    here = os.path.dirname(os.path.abspath(__file__))
    for p in kit["probes"]:
        r = subprocess.run([sys.executable, os.path.join(here, "stablehlo_run.py"), os.path.join(d, p)] + extra)
        print(("ok      " if r.returncode == 0 else "FAILED  ") + p)
    # the device of the first probe that ran names the run
    for p in kit["probes"]:
        f = os.path.join(d, p, "device.json")
        if os.path.exists(f):
            open(os.path.join(d, "device.json"), "w").write(open(f).read())
            break
    '''
  end
end
