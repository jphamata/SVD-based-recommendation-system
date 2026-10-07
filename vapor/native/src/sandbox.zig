//! seccomp-BPF confinement of the native worker.
//!
//! Once installed, the process may only read/write its stdio, map/unmap/
//! protect memory, open files read-only for zero-copy weight mapping, read
//! the monotonic clock, and exit. Any other system call (execve, socket,
//! fork, ptrace, kill, …) terminates the worker with SIGSYS — so generated
//! machine code running inside it cannot escape into the host even if it is
//! wrong or malicious. `PR_SET_NO_NEW_PRIVS` is set first, as the kernel
//! requires for unprivileged filters, and the audit architecture is checked
//! so a foreign syscall ABI (x32, compat) cannot bypass the allowlist.

const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;

const SockFilter = extern struct { code: u16, jt: u8, jf: u8, k: u32 };
const SockFprog = extern struct { len: u16, filter: [*]const SockFilter };

const LD_W_ABS: u16 = 0x20;
const JEQ_K: u16 = 0x15;
const RET_K: u16 = 0x06;
const RET_ALLOW: u32 = 0x7fff0000;
const RET_KILL_PROCESS: u32 = 0x80000000;

fn stmt(code: u16, k: u32) SockFilter {
    return .{ .code = code, .jt = 0, .jf = 0, .k = k };
}

fn jeq(k: u32, jt: u8, jf: u8) SockFilter {
    return .{ .code = JEQ_K, .jt = jt, .jf = jf, .k = k };
}

// AUDIT_ARCH_* = EM_* | __AUDIT_ARCH_64BIT | __AUDIT_ARCH_LE (linux/audit.h)
const audit_arch: u32 = switch (builtin.cpu.arch) {
    .x86_64 => 0xC000_003E,
    .aarch64 => 0xC000_00B7,
    .riscv64 => 0xC000_00F3,
    else => @compileError("unsupported architecture"),
};

const allowed = blk: {
    const S = linux.SYS;
    const list = [_]S{
        .read,        .write,  .openat, .close,       .lseek,
        .mmap,        .munmap, .mprotect, .exit,      .exit_group,
        .rt_sigreturn, .clock_gettime, .rt_sigprocmask, .getpid, .gettid,
        .setitimer,   .futex,
    };
    break :blk list;
};

pub const Status = enum(u8) { off = 0, on = 1, unsupported = 2 };

/// seccomp on Linux, Capsicum on FreeBSD; elsewhere (Darwin) the worker
/// reports that it runs unconfined (HELLO's sandbox byte), and the BEAM
/// records it.
pub const install = switch (builtin.os.tag) {
    .linux => installLinux,
    .freebsd => installCapsicum,
    else => installNone,
};

// FreeBSD Capsicum (sys/capsicum.h): rights are CAPRIGHT(0, bit) = 1<<57 | bit
const capsicum = struct {
    const CapRights = extern struct { cr_rights: [2]u64 };
    extern "c" fn __cap_rights_init(version: c_int, rights: *CapRights, ...) *CapRights;
    extern "c" fn cap_rights_limit(fd: c_int, rights: *const CapRights) c_int;
    extern "c" fn cap_enter() c_int;
    extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
    fn right(bit: u64) u64 {
        return (1 << 57) | bit;
    }
};

/// Capability mode (Capsicum): after `cap_enter` the process holds no
/// global namespace at all — no `open` by path, no sockets, no `execve`, no
/// new processes, no signals to others. What it keeps is what it already
/// holds: its stdio, its memory, and one directory descriptor on `/`
/// limited to look-up, read, seek, stat and read-only mapping, through
/// which weight files are opened (`sys.root_fd`, `openat`): the same
/// policy as the Linux seccomp filter, by capabilities instead of a
/// system-call list. (Compiled for FreeBSD and type-checked here; vapor's
/// tests have no FreeBSD host to run it on — said in docs/TODO.md.)
fn installCapsicum() Status {
    const sys = @import("sys.zig");
    const fd = capsicum.open("/", 0x0002_0000 | 0x0010_0000); // O_RDONLY | O_DIRECTORY | O_CLOEXEC
    if (fd < 0) return .off;
    var r: capsicum.CapRights = undefined;
    const READ = capsicum.right(0x1);
    const SEEK = capsicum.right(0x4 | 0x8);
    const MMAP_R = capsicum.right(0x10) | SEEK | READ;
    const LOOKUP = capsicum.right(0x400);
    const FSTAT = capsicum.right(0x8_0000);
    _ = capsicum.__cap_rights_init(0, &r, READ, SEEK, MMAP_R, LOOKUP, FSTAT, @as(u64, 0));
    if (capsicum.cap_rights_limit(fd, &r) != 0) return .off;
    if (capsicum.cap_enter() != 0) return .off;
    sys.root_fd = fd;
    return .on;
}

fn installNone() Status {
    return .unsupported;
}

fn installLinux() Status {
    const extra: usize = if (builtin.cpu.arch == .riscv64) 1 else 0;
    const n = 5 + allowed.len + extra;
    var prog: [n]SockFilter = undefined;
    // [0] load arch, [1] arch == ours ? continue : kill, [2] load nr
    prog[0] = stmt(LD_W_ABS, 4);
    prog[1] = jeq(audit_arch, 0, @intCast(n - 4));
    prog[2] = stmt(LD_W_ABS, 0);
    inline for (allowed, 0..) |sys, i| {
        // on match jump to the ALLOW at the end
        prog[3 + i] = jeq(@intCast(@intFromEnum(sys)), @intCast(n - 2 - (3 + i)), 0);
    }
    if (extra == 1) {
        const idx = 3 + allowed.len;
        prog[idx] = jeq(259, @intCast(n - 2 - idx), 0); // riscv_flush_icache
    }
    prog[n - 2] = stmt(RET_K, RET_KILL_PROCESS);
    prog[n - 1] = stmt(RET_K, RET_ALLOW);

    if (linux.errno(linux.prctl(@intFromEnum(linux.PR.SET_NO_NEW_PRIVS), 1, 0, 0, 0)) != .SUCCESS)
        return .unsupported;
    const fprog = SockFprog{ .len = @intCast(n), .filter = &prog };
    // TSYNC: the filter applies to every thread of the pool, not only this one
    const rc = linux.seccomp(linux.SECCOMP.SET_MODE_FILTER, 1, &fprog);
    return if (linux.errno(rc) == .SUCCESS) .on else .unsupported;
}
