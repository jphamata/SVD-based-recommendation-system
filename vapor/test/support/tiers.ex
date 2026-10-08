defmodule Vapor.TestTiers do
  @moduledoc false
  # the tooling probes, apart from the helpers: every test file depends on
  # them (test_helper.exs), so they name only what they need (the test
  # cache's keys follow what a file can reach)

  @doc """
  The tiers this machine cannot run (their tooling is missing): the tags
  `test_helper.exs` excludes, and part of `mix vapor.test`'s cache keys.
  """
  def excluded do
    alias Vapor.Runtime.Substrates

    have = fn exe -> System.find_executable(exe) != nil end
    worker = Substrates.binary("vapor-worker", "native")
    fabric? = Substrates.list() |> Enum.any?(&(&1.id == :fabric))

    cross? =
      have.("qemu-riscv64") and have.("qemu-aarch64") and
        Substrates.binary("vapor-worker", "riscv64-linux") != nil and
        Substrates.binary("vapor-worker", "aarch64-linux") != nil

    py? = &python?/1

    # headless Chromium through Playwright (the console's pages, the GPU tracer on SwiftShader)
    playwright? =
      have.("node") and
        match?({_, 0}, System.cmd("node", ["-e", "try { require('playwright') } catch { require('/opt/node22/lib/node_modules/playwright') }"], stderr_to_stdout: true))

      [native: worker == nil,
       python: not py?.(["json", "numpy"]),
       mpmath: not py?.(["mpmath"]),
       cbor2: not py?.(["cbor2", "cryptography"]),
       jinja2: not py?.(["jinja2"]),
       mcp: not py?.(["mcp.server.mcpserver"]),
       snarkjs: not (System.get_env("VAPOR_SNARKJS") != nil and System.find_executable("node") != nil),
       torch: not py?.(["torch", "transformers", "safetensors"]),
       diffusers: not py?.(["torch", "diffusers", "safetensors"]),
       openai: not py?.(["openai"]),
       gguf_py: not py?.(["gguf", "numpy"]),
       llama_cpp: not (System.get_env("VAPOR_LLAMA_CPP") != nil and py?.(["torch", "transformers", "gguf"]) and
                         File.exists?(Path.expand("../fixtures/vocab/ggml-vocab-llama-bpe.gguf", __DIR__))),
       llama_cpp_lib: not (py?.(["llama_cpp", "numpy"]) and File.exists?(Path.expand("../fixtures/vocab/ggml-vocab-qwen2.gguf", __DIR__))),
       vocab: not File.exists?(Path.expand("../fixtures/vocab/ggml-vocab-llama-spm.gguf", __DIR__)),
       clip_vocab: not (File.exists?(Path.expand("../fixtures/vocab/bpe_simple_vocab_16e6.txt.gz", __DIR__)) and
                          py?.(["tokenizers", "transformers"])),
       hf_tokenizers: not (File.exists?(Path.expand("../fixtures/vocab/ggml-vocab-llama-spm.gguf", __DIR__)) and
                             py?.(["tokenizers", "transformers", "gguf"])), qemu: not cross?, vulkan: not fabric?,
       binutils: not (have.("objdump") and have.("aarch64-linux-gnu-objdump") and have.("riscv64-linux-gnu-as")),
       spirv_tools: not have.("spirv-val"), lean: not have.("lake"), ffmpeg: not have.("ffmpeg"),
       trimesh: not py?.(["trimesh", "numpy"]), pillow: not py?.(["PIL", "numpy"]),
       msl_shim: not Vapor.MSLShim.available?(), zig: not have.("zig"), jax: not py?.(["jax", "numpy"]), networkx: not py?.(["networkx", "scipy"]), scipy: not py?.(["scipy", "numpy"]), chess_py: not py?.(["chess"]), shogi_py: not py?.(["shogi"]), tmtools: not py?.(["tmtools", "numpy"]), biopython: not py?.(["Bio.Align"]), quantlib: not py?.(["QuantLib"]), ngspice: not have.("ngspice"), simplefix: not py?.(["simplefix"]), playwright: not playwright?]
      |> Enum.filter(fn {_, skip} -> skip end)
      |> Enum.map(&elem(&1, 0))
  end

  @doc false
  def python, do: System.get_env("VAPOR_PYTHON") || System.find_executable("python3")

  @doc false
  def python?(mods) do
    py = python()
    py != nil and match?({_, 0}, System.cmd(py, ["-c", "import " <> Enum.join(mods, ", ")], stderr_to_stdout: true))
  end
end
