defmodule Vapor.Runtime.Substrates do
  @moduledoc """
  Discovery and supervision of execution substrates.

  | id              | process                         | executes                      |
  |-----------------|---------------------------------|-------------------------------|
  | `:host`         | `vapor-worker` (native)         | host ISA machine code         |
  | `:host_avx512`  | same worker                     | AVX-512 code (x86-64-v4 hosts)|
  | `:rvv_emulated` | same worker, emulate mode       | RVV code, in-tree interpreter |
  | `:aarch64_qemu` | worker under `qemu-aarch64`     | NEON code (cross validation)  |
  | `:riscv64_qemu` | worker under `qemu-riscv64`     | RVV code (cross validation)   |
  | `:fabric`       | `vapor-fabric` (Vulkan)         | SPIR-V                        |
  | `:metal`        | `vapor-metal` (macOS)           | MSL (the same kernels)        |
  | `:oracle`       | none (in the BEAM, pure)        | declared semantics            |

  Binaries are looked up in `priv/` (installed by `make native`) and then in
  `native/zig-out/<target>/bin`. A missing tier is simply absent: hardware is
  a performance tier, never a correctness tier — and an accelerator is
  admitted by measurement before it computes anything canonical
  (`Vapor.Substrate`).
  """
  use Supervisor

  # (FreeBSD's ports name the ISAs amd64 and arm64: "amd64-portbld-freebsd14.0")
  @host_isa (case :erlang.system_info(:system_architecture) |> List.to_string() do
               "x86_64" <> _ -> :x86_64
               "amd64" <> _ -> :x86_64
               "aarch64" <> _ -> :aarch64
               "arm64" <> _ -> :aarch64
               "riscv64" <> _ -> :riscv64
               _ -> :unknown
             end)

  def host_isa, do: @host_isa

  @v4 ~w(avx512f avx512vl avx512bw avx512dq avx512cd bmi2)

  @doc """
  Every code ISA the host executes natively: the base ISA, plus
  `:x86_64_avx512` on x86-64-v4 machines (the CPU flags the kernel
  reports — which also means it saves the zmm state).
  """
  def host_isas do
    case :persistent_term.get({__MODULE__, :isas}, nil) do
      nil ->
        isas = [@host_isa] ++ if(@host_isa == :x86_64 and v4?(), do: [:x86_64_avx512], else: [])
        :persistent_term.put({__MODULE__, :isas}, isas)
        isas

      isas ->
        isas
    end
  end

  @doc "The widest ISA the host executes (the performance default)."
  def best_isa, do: List.last(host_isas())

  defp v4? do
    case File.read("/proc/cpuinfo") do
      {:ok, info} ->
        flags = Regex.run(~r/^flags\s*:\s*(.*)$/m, info, capture: :all_but_first) |> List.wrap() |> List.first("") |> String.split()
        Enum.all?(@v4, &(&1 in flags))

      _ ->
        false
    end
  end

  def start_link(opts \\ []), do: Supervisor.start_link(__MODULE__, opts, name: __MODULE__)

  @impl true
  def init(opts) do
    children =
      [
        worker_child(:host, native_worker(), Keyword.get(opts, :sandbox, true)),
        fabric_child(Keyword.get(opts, :fabric, true)),
        metal_child(Keyword.get(opts, :metal, true))
      ] ++
        if Keyword.get(opts, :cross, false) do
          [qemu_child(:aarch64_qemu, "qemu-aarch64", [], "aarch64-linux"),
           qemu_child(:riscv64_qemu, "qemu-riscv64", ["-cpu", "rv64,v=true,vlen=256,zfh=true,vext_spec=v1.0"], "riscv64-linux")]
        else
          []
        end

    Supervisor.init(Enum.reject(children, &is_nil/1), strategy: :one_for_one)
  end

  @doc "Available substrates as descriptors `%{id, kind, isa, mode, server}`."
  def list do
    running =
      case Process.whereis(__MODULE__) do
        nil -> []
        _ -> Supervisor.which_children(__MODULE__) |> Enum.flat_map(fn {id, pid, _, _} -> if is_pid(pid), do: [{id, pid}], else: [] end)
      end
      |> Map.new()

    host =
      case running[:host] do
        nil -> []
        pid -> [%{id: :host, kind: :native, isa: @host_isa, mode: :native, server: pid}] ++
                 for(isa <- tl(host_isas()), do: %{id: :"host_#{isa |> Atom.to_string() |> String.replace("x86_64_", "")}", kind: :native, isa: isa, mode: :native, server: pid}) ++
                 [%{id: :rvv_emulated, kind: :native, isa: :riscv64, mode: :emulate, server: pid}]
      end

    cross =
      Enum.flat_map([aarch64_qemu: :aarch64, riscv64_qemu: :riscv64], fn {id, isa} ->
        case running[id] do
          nil -> []
          pid -> [%{id: id, kind: :native, isa: isa, mode: :native, server: pid}]
        end
      end)

    fabric =
      case running[:fabric] do
        nil -> []
        pid -> if Vapor.Runtime.Fabric.info(pid).ready, do: [%{id: :fabric, kind: :fabric, isa: :spirv, mode: :gpu, server: pid}], else: []
      end

    metal =
      case running[:metal] do
        nil -> []
        pid -> if Vapor.Runtime.Fabric.info(pid).ready, do: [%{id: :metal, kind: :fabric, isa: :msl, mode: :gpu, server: pid}], else: []
      end

    host ++ cross ++ fabric ++ metal ++ [%{id: :oracle, kind: :oracle, isa: :oracle, mode: :exact, server: nil}]
  end

  def get(id), do: Enum.find(list(), &(&1.id == id))

  # ------------------------------------------------------------- children --

  defp worker_child(_id, nil, _), do: nil

  defp worker_child(id, path, sandbox),
    do: Supervisor.child_spec({Vapor.Runtime.Worker, exec: [path], sandbox: sandbox}, id: id)

  defp fabric_child(false), do: nil

  defp fabric_child(true) do
    case binary("vapor-fabric", target_dir()) do
      nil -> nil
      path -> Supervisor.child_spec({Vapor.Runtime.Fabric, exec: [path]}, id: :fabric)
    end
  end

  # Metal exists only on macOS; `vapor-metal-sim` (its Linux stand-in) is
  # started by tests, never discovered
  defp metal_child(false), do: nil

  defp metal_child(true) do
    case :os.type() == {:unix, :darwin} and binary("vapor-metal", target_dir()) do
      path when is_binary(path) -> Supervisor.child_spec({Vapor.Runtime.Fabric, exec: [path]}, id: :metal)
      _ -> nil
    end
  end

  defp qemu_child(id, qemu, args, target) do
    with q when is_binary(q) <- System.find_executable(qemu),
         w when is_binary(w) <- binary("vapor-worker", target) do
      Supervisor.child_spec({Vapor.Runtime.Worker, exec: [q | args] ++ [w]}, id: id)
    else
      _ -> nil
    end
  end

  defp native_worker, do: binary("vapor-worker", target_dir())

  defp target_dir, do: "native"

  @doc false
  def binary(name, target) do
    priv = case :code.priv_dir(:vapor) do
      {:ok, dir} -> [Path.join([dir, "bin", name])]
      _ -> []
    end

    root = Path.expand("../../..", __DIR__)
    # $VAPOR_BIN has the zig-out layout (<target>/bin/<name>); `nix build` produces it
    prebuilt = for dir <- List.wrap(System.get_env("VAPOR_BIN")), do: Path.join([dir, target, "bin", name])

    candidates =
      (if target == "native", do: priv, else: []) ++
        [Path.join([root, "native/zig-out", target, "bin", name]) | prebuilt]
    Enum.find(candidates, &File.exists?/1)
  end
end
