//! vapor native tier: `zig build` produces
//!   vapor-worker  — static, libc-free native substrate (+ RVV interpreter)
//!   vapor-fabric  — Vulkan compute daemon (links libc for the loader)
//!   vapor-metal   — Metal compute daemon (macOS targets; MSL modules)
//!   vapor-metal-sim — the same daemon on Linux over a CPU stand-in for
//!                   Metal (tests: MSL compiled by clang through a shim)
//! Cross builds: `zig build -Dtarget=aarch64-linux` / `-Dtarget=riscv64-linux`;
//! `-Dtarget=aarch64-macos` / `x86_64-macos` build the Darwin pair;
//! `-Dtarget=x86_64-freebsd` / `aarch64-freebsd` the FreeBSD worker
//! (libc, Capsicum).
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const darwin = target.result.os.tag.isDarwin();

    if (target.result.os.tag == .freebsd) {
        // the CPU worker on FreeBSD: its libc, W^X code pages by mprotect,
        // Capsicum capability mode for the sandbox. (The Vulkan fabric and
        // the Metal daemon are not built for FreeBSD.)
        const fw = b.addExecutable(.{
            .name = "vapor-worker",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/worker.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
                .strip = optimize != .Debug,
            }),
        });
        b.installArtifact(fw);
        return;
    }

    // The GPU daemon for unified-memory devices: Metal on macOS, the CPU
    // stand-in on Linux. Links libc (dlopen of the framework or the modules).
    const metal = b.addExecutable(.{
        .name = if (darwin) "vapor-metal" else "vapor-metal-sim",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/metal.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .strip = optimize != .Debug,
        }),
    });
    b.installArtifact(metal);
    if (darwin) {
        // the CPU worker on macOS: libSystem (no stable syscall ABI there),
        // MAP_JIT code pages, no seccomp (reported as such in HELLO)
        const dw = b.addExecutable(.{
            .name = "vapor-worker",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/worker.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
                .strip = optimize != .Debug,
            }),
        });
        b.installArtifact(dw);
        return; // the Vulkan fabric is a Linux program
    }

    const worker = b.addExecutable(.{
        .name = "vapor-worker",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/worker.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = false,
            .strip = optimize != .Debug,
        }),
    });
    b.installArtifact(worker);

    // The fabric daemon links the system libc so that the Vulkan loader
    // (and its ICDs) can be dlopen'ed; nothing Vulkan is linked at build time.
    const fabric = b.addExecutable(.{
        .name = "vapor-fabric",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/fabric.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .strip = optimize != .Debug,
        }),
    });
    b.installArtifact(fabric);

    const tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/worker.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    const run_tests = b.addRunArtifact(tests);
    b.step("test", "run native unit tests").dependOn(&run_tests.step);
}
