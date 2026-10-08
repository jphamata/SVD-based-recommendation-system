defmodule Vapor.MSLShim do
  @moduledoc false
  # The "device" behind `vapor-metal-sim`: a directory of shared objects that
  # clang built from the exact MSL text `Vapor.Emit.MSL` emits (plus the
  # ten-line shim in test/support/msl), keyed by the source's SHA-256 —
  # what `newLibraryWithSource:` would compile on a Mac. The build flags
  # are the device's arithmetic: one rounding per operation (conforming),
  # FMA contraction, or flush-to-zero — the last two are the non-conforming
  # devices the substrate airlock must catch.
  alias Vapor.Compiled
  alias Vapor.Emit.MSL

  @shim Path.expand("msl", __DIR__)

  def available?, do: System.find_executable("clang++") != nil and Vapor.Runtime.Substrates.binary("vapor-metal-sim", "native") != nil

  def flags(:conforming), do: ["-O2", "-ffp-contract=off"]
  def flags(:contracting), do: ["-O2", "-ffp-contract=fast", "-mfma"]
  def flags(:ftz), do: ["-O2", "-ffp-contract=off", "-DVAPOR_SHIM_FTZ"]

  def device_name(:conforming), do: "MSL shim — clang, one rounding per operation (CPU)"
  def device_name(:contracting), do: "MSL shim — clang -ffp-contract=fast (fuses a·b+c)"
  def device_name(:ftz), do: "MSL shim — flush-to-zero (MXCSR FTZ|DAZ)"

  # The objects need nothing from the C++ runtime: linking none keeps them loadable by any glibc
  # loader, including a worker built against another libc than the system clang's (Nix on Ubuntu).
  @link ["-std=c++17", "-shared", "-fPIC", "-nostdlib++", "-static-libgcc", "-Wno-unknown-attributes"]

  @doc "The cache directory of a device mode (created on demand), keyed by its build flags."
  def dir(mode) do
    tag = :crypto.hash(:sha256, :erlang.term_to_binary({@link, flags(mode)})) |> Base.encode16(case: :lower) |> binary_part(0, 8)
    d = Path.join(System.tmp_dir!(), "vapor-msl-#{mode}-#{tag}")
    File.mkdir_p!(d)
    File.write!(Path.join(d, "device"), device_name(mode) <> "\n")
    d
  end

  @doc "Compile every kernel of a compiled program for a device mode."
  def ensure(%Compiled{} = c, mode) do
    keys = c.schedule |> Enum.map(& &1.kernel) |> Enum.uniq()
    ensure_keys(keys, c.policy, mode)
  end

  def ensure_keys(keys, policy, mode) do
    d = dir(mode)

    keys
    |> Enum.map(&MSL.compile(&1, policy))
    |> Enum.reject(&is_nil/1)
    |> Enum.uniq_by(& &1.src)
    |> Task.async_stream(fn %{src: src} -> build(d, src, mode) end, timeout: 300_000, max_concurrency: System.schedulers_online())
    |> Enum.each(fn {:ok, :ok} -> :ok end)
  end

  def build(d, src, mode) do
    hex = :crypto.hash(:sha256, src) |> Base.encode16(case: :lower)
    so = Path.join(d, hex <> ".so")

    if File.exists?(so) do
      :ok
    else
      cpp = Path.join(d, hex <> ".cpp")
      File.write!(cpp, src)
      tmp = so <> ".#{System.unique_integer([:positive])}"
      args = @link ++ ["-I", @shim] ++ flags(mode) ++ ["-o", tmp, cpp]
      {out, rc} = System.cmd("clang++", args, stderr_to_stdout: true)
      if rc != 0, do: raise("clang++ failed for #{cpp}:\n#{out}")
      File.rename!(tmp, so)
      :ok
    end
  end

  @doc "A `Vapor.Runtime.Fabric` on `vapor-metal-sim` for a device mode."
  def start(mode, opts \\ []) do
    exe = Vapor.Runtime.Substrates.binary("vapor-metal-sim", "native")
    Vapor.Runtime.Fabric.start_link([exec: [exe, "--cache", dir(mode)] ++ Keyword.get(opts, :args, [])] ++ Keyword.drop(opts, [:args]))
  end
end
