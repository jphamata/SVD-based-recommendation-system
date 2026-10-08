# vapor — build and verification entry points.
MIX  ?= mix
ZIG  ?= zig
LAKE ?= lake
OPT  ?= ReleaseSafe

.PHONY: freebsd all native metal cross proofs extract test test-core integrations notebook example bench quality slides fixtures e2e technical clean

all: native proofs test

## native: vapor-worker + vapor-fabric for this host, installed into priv/
native:
	cd native && $(ZIG) build -Dtarget=native -Doptimize=$(OPT) --prefix zig-out/native
	mkdir -p priv/bin
	cp native/zig-out/native/bin/vapor-worker native/zig-out/native/bin/vapor-fabric native/zig-out/native/bin/vapor-metal-sim priv/bin/

## metal: the Metal daemon for macOS (Apple Silicon and Intel), cross-compiled
## from any host — no SDK needed (Metal and libobjc are opened at run time)
metal:
	cd native && $(ZIG) build -Dtarget=aarch64-macos -Doptimize=$(OPT) --prefix zig-out/aarch64-macos
	cd native && $(ZIG) build -Dtarget=x86_64-macos -Doptimize=$(OPT) --prefix zig-out/x86_64-macos

## freebsd: the CPU worker for FreeBSD (x86-64 and AArch64), Capsicum-confined,
## cross-compiled from any host (zig ships FreeBSD's libc headers)
freebsd:
	cd native && $(ZIG) build -Dtarget=x86_64-freebsd -Doptimize=$(OPT) --prefix zig-out/x86_64-freebsd
	cd native && $(ZIG) build -Dtarget=aarch64-freebsd -Doptimize=$(OPT) --prefix zig-out/aarch64-freebsd

## cross: AArch64 and RISC-V workers (run under qemu-user for cross validation)
cross:
	cd native && $(ZIG) build -Dtarget=aarch64-linux -Doptimize=$(OPT) --prefix zig-out/aarch64-linux
	cd native && $(ZIG) build -Dtarget=riscv64-linux -Doptimize=$(OPT) --prefix zig-out/riscv64-linux

## proofs: build the Lean development (core Lean only; must be warning-free)
proofs:
	cd proofs && $(LAKE) build

## extract: regenerate lib/vapor/extracted.ex and its Lean-computed conformance suite
extract: proofs
	cd proofs && $(LAKE) exe vapor-extract ../lib/vapor/extracted.ex ../test/vapor/extracted_conformance_test.exs

## test: every tier whose tooling is present (see test/test_helper.exs)
test:
	$(MIX) test

## test-core: control plane only (no worker, QEMU, Vulkan, binutils or Lean)
test-core:
	$(MIX) test --exclude native --exclude qemu --exclude vulkan --exclude binutils --exclude spirv_tools --exclude lean

## integrations: the Plug/Phoenix and Nx adapters (each its own Mix project,
## Elixir >= 1.15 / 1.17; with VAPOR_ECO=<dir of source checkouts> no Hex)
integrations: native
	cd integrations/vapor_plug && $(MIX) deps.get && $(MIX) test
	cd integrations/vapor_nx && $(MIX) deps.get && $(MIX) test

## notebook: run the Livebook tour's cells in order (also part of `make test`)
notebook: native
	$(MIX) run scripts/run_livemd.exs notebooks/vapor_tour.livemd

example:
	$(MIX) run examples/ssm_block.exs

bench:
	$(MIX) run bench/bench.exs

## quality: the output-quality benchmark (calibrated noise gates, planted
## models, every any-to-any route, fusion, substrate parity); writes
## docs/bench/QUALITY.md, quality.json and the PNG/WAV gallery; exits 1 on
## any failed check
quality:
	$(MIX) vapor.quality

## fixtures: real vocabularies (llama.cpp's vocab-only GGUFs and their
## Hugging Face-generated test vectors), pinned by SHA-256
FIXTURE_URL ?= https://raw.githubusercontent.com/ggml-org/llama.cpp/master/models
# CLIP's BPE merges (OpenAI, MIT): the CLIP tokenizer is rebuilt from them
CLIP_BPE_URL ?= https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz
fixtures:
	mkdir -p test/fixtures/vocab
	cd test/fixtures/vocab && for f in $$(awk '{print $$2}' ../../fixtures.sha256); do \
	  [ -f $$f ] || curl -fsSL -o $$f $(FIXTURE_URL)/$$f || exit 1; done
	cd test/fixtures/vocab && sha256sum -c ../../fixtures.sha256
	cd test/fixtures/vocab && { [ -f bpe_simple_vocab_16e6.txt.gz ] || curl -fsSL -o bpe_simple_vocab_16e6.txt.gz $(CLIP_BPE_URL); } && \
	  echo "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a  bpe_simple_vocab_16e6.txt.gz" | sha256sum -c

## e2e: the whole pipeline on a demo checkpoint (random weights, real Qwen2
## vocabulary): airlocks → tokenizer → compiler → engine → HTTP server → client,
## from the safetensors directory, its q8_0 GGUF export, and that GGUF exported
## back to a sharded bf16 safetensors directory
E2E_DIR ?= _build/e2e-model
e2e: native fixtures
	$(MIX) vapor.demo_model $(E2E_DIR)
	$(MIX) vapor.generate --model $(E2E_DIR) --prompt "Olá, mundo!" --max-tokens 32 --temperature 0.8 --seed 1
	$(MIX) vapor.export --model $(E2E_DIR) --out $(E2E_DIR).q8_0.gguf --type q8_0
	$(MIX) vapor.generate --model $(E2E_DIR).q8_0.gguf --prompt "Olá, mundo!" --max-tokens 32 --temperature 0
	$(MIX) vapor.export --model $(E2E_DIR).q8_0.gguf --out $(E2E_DIR).bf16 --dtype bf16 --shard-size 20000000
	$(MIX) vapor.generate --model $(E2E_DIR).bf16 --storage bf16 --prompt "Olá, mundo!" --max-tokens 32 --temperature 0
	sh scripts/e2e_serve.sh $(E2E_DIR)
	sh scripts/e2e_serve.sh $(E2E_DIR).q8_0.gguf

## slides: the Beamer deck (pdflatex twice for the page references)
slides:
	cd slides && pdflatex -interaction=nonstopmode -halt-on-error vapor.tex >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error vapor.tex >/dev/null

## technical: the technical monograph and slides, the system as it is (lualatex twice; DejaVu fonts)
technical:
	cd technical && for f in monograph slides; do lualatex -interaction=nonstopmode -halt-on-error $$f.tex >/dev/null && lualatex -interaction=nonstopmode -halt-on-error $$f.tex >/dev/null || exit 1; done

clean:
	rm -rf _build priv/bin native/zig-out native/.zig-cache proofs/.lake
