# flake.nix — vapor: build, test tiers and slides.
#
#   nix develop               shell with every test tier's tooling (make native test)
#   nix develop .#slides      TeX Live for slides/ (make slides)
#   nix build                 vapor-worker/vapor-fabric, zig-out layout: $out/<target>/bin
#   nix flake check           control plane + native tier against the built binaries
#
# Tiers are enabled by test/test_helper.exs exactly when their tools are on PATH.
{
  description = "vapor — certified tensor compiler: BEAM → x86-64 / AArch64 / RVV 1.0 / SPIR-V";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f system nixpkgs.legacyPackages.${system});

      # zig 0.16 is what native/build.zig targets
      zigOf = pkgs: pkgs.zig_0_16 or pkgs.zig;

      # Zig triples: explicit gnu target → Zig's bundled glibc stubs, so the
      # build needs no system libc; autoPatchelf then points the fabric at
      # Nix's loader and adds the (dlopen'ed) Vulkan loader to its RUNPATH.
      hostTriple = system: { x86_64-linux = "x86_64-linux-gnu"; aarch64-linux = "aarch64-linux-gnu"; }.${system};

      # The binutils oracle calls Debian-style triplets (riscv64-linux-gnu-as);
      # Nix names them riscv64-unknown-linux-gnu-*. Alias, do not patch tests.
      triplet = pkgs: name: cross: pkgs.runCommand "${name}-binutils" { } ''
        mkdir -p $out/bin
        bu=${cross.buildPackages.binutils-unwrapped}
        for t in $bu/bin/${cross.stdenv.targetPlatform.config}-*; do
          ln -s "$t" "$out/bin/${name}-''${t##*/${cross.stdenv.targetPlatform.config}-}"
        done
      '';
    in
    {
      packages = forAll (system: pkgs:
        let zig = zigOf pkgs; in rec {
          native = pkgs.stdenv.mkDerivation {
            pname = "vapor-native";
            version = "0.2.0";
            src = ./native;
            nativeBuildInputs = [ zig pkgs.autoPatchelfHook ];
            runtimeDependencies = [ pkgs.vulkan-loader ];
            dontConfigure = true;
            buildPhase = ''
              export ZIG_GLOBAL_CACHE_DIR=$TMPDIR/zig-global ZIG_LOCAL_CACHE_DIR=$TMPDIR/zig-local
              build() { zig build -Dtarget=$1 -Dcpu=baseline -Doptimize=ReleaseSafe --prefix out/$2; }
              build ${hostTriple system} native
              build aarch64-linux aarch64-linux
              build riscv64-linux riscv64-linux
            '';
            installPhase = ''
              mkdir -p $out/native/bin $out/aarch64-linux/bin $out/riscv64-linux/bin
              cp out/native/bin/vapor-worker out/native/bin/vapor-fabric $out/native/bin/
              # cross workers are static and libc-free: they run under qemu-user as-is
              cp out/aarch64-linux/bin/vapor-worker $out/aarch64-linux/bin/
              cp out/riscv64-linux/bin/vapor-worker $out/riscv64-linux/bin/
            '';
            doCheck = true;
            checkPhase = "zig build test -Dtarget=${hostTriple system} -Dcpu=baseline";
          };
          default = native;
        });

      devShells = forAll (system: pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            beamPackages.elixir beamPackages.erlang      # core (top-level elixir/erlang are deprecated)
            gnumake zip (zigOf pkgs)                     # native
            qemu                                         # qemu-aarch64 / qemu-riscv64
            vulkan-loader vulkan-tools mesa              # fabric on lavapipe
            spirv-tools                                  # spirv-val
            elan                                         # Lean 4, version from proofs/lean-toolchain
            poppler-utils qpdf ghostscript               # document airlock: pdftotext oracle, fixture producers
            (python3.withPackages (ps: with ps; [numpy pillow]))   # NumPy references, Pillow PNG oracle
          ] ++ lib.optionals (system == "x86_64-linux") [
            binutils                                     # objdump -m i386:x86-64
            (triplet pkgs "aarch64-linux-gnu" pkgsCross.aarch64-multiplatform)
            (triplet pkgs "riscv64-linux-gnu" pkgsCross.riscv64)
          ];
          shellHook = ''
            export MIX_HOME=$PWD/.nix-mix HEX_HOME=$PWD/.nix-hex ERL_AFLAGS="-kernel shell_history enabled"
            # the fabric dlopen()s libvulkan.so.1
            export LD_LIBRARY_PATH=${pkgs.vulkan-loader}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
            # no system Vulkan driver (non-NixOS, CI): fall back to Mesa's lavapipe
            if [ ! -d /run/opengl-driver/share/vulkan/icd.d ] && [ -z "''${VK_DRIVER_FILES:-}" ]; then
              export VK_DRIVER_FILES=$(ls ${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.*.json 2>/dev/null | head -n1)
            fi
          '';
        };

        slides = pkgs.mkShell {
          packages = [
            pkgs.gnumake
            (pkgs.texliveMedium.withPackages (ps: with ps; [ beamer pgf pgfplots abntex2 booktabs listings xcolor latexmk ]))   # slides/, thesis/, technical/
            pkgs.dejavu_fonts   # technical/: the monospace and the Arabic script, found by luaotfload
          ];
          shellHook = ''
            export OSFONTDIR=${pkgs.dejavu_fonts}/share/fonts''${OSFONTDIR:+:$OSFONTDIR}
          '';
        };
      });

      checks = forAll (system: pkgs: {
        default = pkgs.stdenv.mkDerivation {
          name = "vapor-check";
          src = ./.;
          nativeBuildInputs = [ pkgs.beamPackages.elixir pkgs.beamPackages.erlang ];
          dontConfigure = true;
          buildPhase = ''
            export HOME=$TMPDIR MIX_HOME=$TMPDIR/mix HEX_HOME=$TMPDIR/hex
            export VAPOR_BIN=${self.packages.${system}.native}
            mix test --exclude qemu --exclude vulkan --exclude binutils --exclude spirv_tools --exclude lean
          '';
          installPhase = "touch $out";
        };
      });

      formatter = forAll (system: pkgs: pkgs.nixfmt);
    };
}
