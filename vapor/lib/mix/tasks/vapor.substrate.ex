defmodule Mix.Tasks.Vapor.Substrate do
  @shortdoc "Admit the execution substrates by measurement (the substrate airlock)"
  @moduledoc """
      mix vapor.substrate                      # admit every substrate of this machine
      mix vapor.substrate kit DIR              # the battery as StableHLO, for a PJRT device
      mix vapor.substrate judge DIR [--sign KEYFILE]   # admit the device that ran the kit

  Each substrate runs the known-answer battery of `Vapor.Substrate` and is
  admitted as canonical (the oracle's bits), envelope-bound (FMA, another
  reduction order, flush-to-zero — inside the rigorous bound) or refused;
  the fingerprint says what the device does. `kit` writes the battery for
  devices vapor reaches only through their vendor's compiler — Tenstorrent
  (`tt-xla`), TPU, GPUs — via PJRT (`python3 DIR/run_kit.py DIR --platform
  tt`); `judge` reads the outputs back. `--sign KEYFILE` signs the
  admission record with the operator's Ed25519 key (the format of `mix
  vapor.audit keygen`), so the signature says *who* admitted the device
  (`Vapor.Substrate.attest/2`); a fresh throwaway key would say nothing.
  """
  use Mix.Task
  alias Vapor.Substrate

  @impl true
  def run(["kit", dir | _]) do
    Mix.Task.run("app.start")
    %{exported: ex, skipped: sk} = Substrate.Kit.write(dir)
    Mix.shell().info("kit in #{dir}: #{length(ex)} probes (#{Enum.join(ex, ", ")})")
    for {n, why} <- sk, do: Mix.shell().info("  not exported: #{n} — #{why}")
    Mix.shell().info("next: python3 #{Path.join(dir, "run_kit.py")} #{dir} --platform <cpu|tt|tpu|cuda>, then mix vapor.substrate judge #{dir}")
  end

  def run(["judge", dir | rest]) do
    Mix.Task.run("app.start")
    a = Substrate.Kit.judge(dir)
    report(a)

    with [_, file | _] <- Enum.drop_while(rest, &(&1 != "--sign")) do
      <<priv::binary-32, pub::binary-32>> = file |> File.read!() |> String.trim() |> Base.decode64!()
      key = %{private: priv, public: pub}
      {bytes, sig} = Substrate.attest(a, key)
      Mix.shell().info("record #{Base.encode16(:crypto.hash(:sha256, bytes), case: :lower)}\n  key #{Base.encode16(key.public, case: :lower)}\n  signature #{Base.encode16(sig, case: :lower)}")
    else
      [] -> :ok
      _ -> Mix.raise("--sign KEYFILE (an Ed25519 key from mix vapor.audit keygen)")
    end
  end

  def run(_argv) do
    Mix.Task.run("app.start")
    for s <- Vapor.Runtime.Substrates.list(), do: s |> Substrate.admit() |> report()
  end

  defp report(a) do
    fp = a.fingerprint

    flags =
      [contraction: fp.contraction, "flush-to-zero": fp.flush_to_zero, "denormals-are-zero": fp.denormals_are_zero]
      |> Enum.filter(&(elem(&1, 1) == true))
      |> Enum.map(&elem(&1, 0))

    Mix.shell().info(String.pad_trailing("#{a.substrate}", 14) <> String.pad_trailing("#{a.verdict}", 11) <> "#{a.device}")
    Mix.shell().info("              significand #{fp.significand_bits} bits, reduction #{fp.reduction_order}" <>
                       if(flags == [], do: "", else: ", " <> Enum.join(flags, ", ")))
    for r <- a.reasons, a.verdict == :refused, do: Mix.shell().info("              ✗ #{r}")
  end
end
