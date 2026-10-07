//! The few operating-system calls the native tier makes, for Linux, Darwin
//! and FreeBSD.
//!
//! On Linux every call is a raw system call: the worker stays static and
//! libc-free, exactly as before. Darwin has no stable system-call ABI — the
//! kernel interface is libSystem — so there the same functions go through
//! the C library (and `vapor-worker`/`vapor-metal` link it). FreeBSD goes
//! through its libc too (its syscall ABI is stable, but libc is the
//! documented interface), with `_umtx_op` for the futex and, once the
//! Capsicum sandbox is entered, files opened relative to the one directory
//! descriptor kept for it (`root_fd`). Nothing above this file names an
//! operating system.

const std = @import("std");
const builtin = @import("builtin");

pub const is_linux = builtin.os.tag == .linux;
pub const is_darwin = builtin.os.tag.isDarwin();
pub const is_freebsd = builtin.os.tag == .freebsd;

/// FreeBSD in capability mode: the read-only directory descriptor through
/// which weight files are still reachable (`openat`, relative paths); -1
/// before the sandbox, and always on the other systems.
pub var root_fd: i32 = -1;

const fbsd = struct {
    extern "c" fn openat(fd: c_int, path: [*:0]const u8, flags: c_int, ...) c_int;
};

// open read-only: relative to the capability directory when there is one
fn openRead(z: [*:0]const u8) c_int {
    if (is_freebsd and root_fd >= 0) {
        var p = z;
        while (p[0] == '/') p += 1;
        return fbsd.openat(root_fd, p, 0x0000 | 0x0010_0000); // O_RDONLY | O_CLOEXEC
    }
    return c.open(z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
}
const linux = std.os.linux;
const c = std.c;

pub const Error = error{ Io, OutOfMemory, Truncated };

/// The smallest page the target can have (16 KiB on Apple Silicon).
pub const page = std.heap.page_size_min;

/// read(2) with EINTR retried; 0 at end of file.
pub fn read(fd: i32, buf: []u8) Error!usize {
    while (true) {
        if (is_linux) {
            const rc = linux.read(fd, buf.ptr, buf.len);
            switch (linux.errno(rc)) {
                .SUCCESS => return rc,
                .INTR => continue,
                else => return error.Io,
            }
        } else {
            const rc = c.read(fd, buf.ptr, buf.len);
            if (rc >= 0) return @intCast(rc);
            if (c.errno(rc) == .INTR) continue;
            return error.Io;
        }
    }
}

/// write(2) with EINTR retried; the count actually written.
pub fn write(fd: i32, buf: []const u8) Error!usize {
    while (true) {
        if (is_linux) {
            const rc = linux.write(fd, buf.ptr, buf.len);
            switch (linux.errno(rc)) {
                .SUCCESS => return rc,
                .INTR => continue,
                else => return error.Io,
            }
        } else {
            const rc = c.write(fd, buf.ptr, buf.len);
            if (rc >= 0) return @intCast(rc);
            if (c.errno(rc) == .INTR) continue;
            return error.Io;
        }
    }
}

/// Fresh zeroed pages.
pub fn mapAnon(len: usize) Error![]align(page) u8 {
    const n = std.mem.alignForward(usize, @max(len, 1), page);
    if (is_linux) {
        const rc = linux.mmap(null, n, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        if (linux.errno(rc) != .SUCCESS) return error.OutOfMemory;
        const p: [*]align(page) u8 = @ptrFromInt(rc);
        return p[0..n];
    } else {
        const p = c.mmap(null, n, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        if (p == c.MAP_FAILED) return error.OutOfMemory;
        const q: [*]align(page) u8 = @ptrCast(@alignCast(p));
        return q[0..n];
    }
}

pub fn unmap(mem: []align(page) u8) void {
    if (is_linux) _ = linux.munmap(mem.ptr, mem.len) else _ = c.munmap(mem.ptr, mem.len);
}

/// Map `len` bytes of the file at `path` copy-on-write (pages rounded up to
/// `align_to`). The file must be at least `len` bytes long.
pub fn mapFile(path: []const u8, len: u64, align_to: u64) Error![]align(page) u8 {
    var z: [4096]u8 = undefined;
    if (path.len >= z.len) return error.Io;
    @memcpy(z[0..path.len], path);
    z[path.len] = 0;
    const map_len = std.mem.alignForward(usize, @max(len, 1), @max(align_to, page));
    if (is_linux) {
        const fd_rc = linux.openat(linux.AT.FDCWD, @ptrCast(&z), .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
        if (linux.errno(fd_rc) != .SUCCESS) return error.Io;
        const fd: i32 = @intCast(fd_rc);
        defer _ = linux.close(fd);
        const size = linux.lseek(fd, 0, linux.SEEK.END);
        if (linux.errno(size) != .SUCCESS or len > size) return error.Truncated;
        const rc = linux.mmap(null, map_len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE }, fd, 0);
        if (linux.errno(rc) != .SUCCESS) return error.OutOfMemory;
        const p: [*]align(page) u8 = @ptrFromInt(rc);
        return p[0..map_len];
    } else {
        const fd = openRead(@ptrCast(&z));
        if (fd < 0) return error.Io;
        defer _ = c.close(fd);
        const size = c.lseek(fd, 0, c.SEEK.END);
        if (size < 0 or len > @as(u64, @intCast(size))) return error.Truncated;
        const p = c.mmap(null, map_len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE }, fd, 0);
        if (p == c.MAP_FAILED) return error.OutOfMemory;
        const q: [*]align(page) u8 = @ptrCast(@alignCast(p));
        return q[0..map_len];
    }
}

pub fn monotonicNs() u64 {
    if (is_linux) {
        var ts: linux.timespec = undefined;
        _ = linux.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    } else {
        var ts: c.timespec = undefined;
        _ = c.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    }
}

pub fn exit(code: u8) noreturn {
    if (is_linux) linux.exit_group(code) else c._exit(code);
}

/// Map `len` bytes of `path` starting at `off` (any offset), private: a
/// writable buffer is copy-on-write and the file never changes. Returns the
/// whole mapping; the data starts at `off % page` within it.
pub fn mapFileAt(path: []const u8, off: u64, len: u64, writable: bool) Error![]align(page) u8 {
    var z: [4096]u8 = undefined;
    if (path.len >= z.len) return error.Io;
    @memcpy(z[0..path.len], path);
    z[path.len] = 0;
    const page_off = off & (page - 1);
    const map_len = std.mem.alignForward(usize, page_off + len, page);
    if (is_linux) {
        const fd_rc = linux.openat(linux.AT.FDCWD, @ptrCast(&z), .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
        if (linux.errno(fd_rc) != .SUCCESS) return error.Io;
        const fd: i32 = @intCast(fd_rc);
        defer _ = linux.close(fd);
        const size = linux.lseek(fd, 0, linux.SEEK.END);
        if (linux.errno(size) != .SUCCESS or off + len > size) return error.Truncated;
        const rc = linux.mmap(null, map_len, .{ .READ = true, .WRITE = writable }, .{ .TYPE = .PRIVATE }, fd, @intCast(off - page_off));
        if (linux.errno(rc) != .SUCCESS) return error.OutOfMemory;
        const p: [*]align(page) u8 = @ptrFromInt(rc);
        return p[0..map_len];
    } else {
        const fd = openRead(@ptrCast(&z));
        if (fd < 0) return error.Io;
        defer _ = c.close(fd);
        const size = c.lseek(fd, 0, c.SEEK.END);
        if (size < 0 or off + len > @as(u64, @intCast(size))) return error.Truncated;
        const p = c.mmap(null, map_len, .{ .READ = true, .WRITE = writable }, .{ .TYPE = .PRIVATE }, fd, @intCast(off - page_off));
        if (p == c.MAP_FAILED) return error.OutOfMemory;
        const q: [*]align(page) u8 = @ptrCast(@alignCast(p));
        return q[0..map_len];
    }
}

// ------------------------------------------------------------ generated code --

extern fn __clear_cache(start: *anyopaque, end: *anyopaque) callconv(.c) void;
const darwin_jit = struct {
    extern "c" fn pthread_jit_write_protect_np(enabled: c_int) void;
    extern "c" fn sys_icache_invalidate(start: *anyopaque, len: usize) void;
};

/// Pages holding `code`, never writable and executable at once (W^X).
/// Linux and FreeBSD: written, then flipped to read+execute (`mprotect`),
/// the instruction cache synchronised where the ISA needs it. Darwin (Apple
/// Silicon forbids executing pages that were not mapped for a JIT): a
/// `MAP_JIT` mapping, written while this thread's view of it is writable
/// (`pthread_jit_write_protect_np(0)`), then switched back to execute-only
/// and the instruction cache invalidated. `executable = false` gives plain
/// readable pages (code for the RVV interpreter is data).
pub fn loadCode(code: []const u8, executable: bool) Error![]align(page) u8 {
    if (!is_darwin or !executable) {
        const m = try mapAnon(code.len);
        @memcpy(m[0..code.len], code);
        if (executable) {
            // Linux and FreeBSD: written, then read+execute (W^X)
            const ok = if (is_linux)
                linux.errno(linux.mprotect(m.ptr, m.len, .{ .READ = true, .EXEC = true })) == .SUCCESS
            else
                c.mprotect(m.ptr, m.len, .{ .READ = true, .EXEC = true }) == 0;
            if (!ok) {
                unmap(m);
                return error.OutOfMemory;
            }
            if (builtin.cpu.arch != .x86_64) __clear_cache(m.ptr, m.ptr + code.len);
        }
        return m;
    }
    const n = std.mem.alignForward(usize, @max(code.len, 1), page);
    const p = c.mmap(null, n, .{ .READ = true, .WRITE = true, .EXEC = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true, .JIT = true }, -1, 0);
    if (p == c.MAP_FAILED) return error.OutOfMemory;
    const m: [*]align(page) u8 = @ptrCast(@alignCast(p));
    if (builtin.cpu.arch == .aarch64) darwin_jit.pthread_jit_write_protect_np(0);
    @memcpy(m[0..code.len], code);
    if (builtin.cpu.arch == .aarch64) {
        darwin_jit.pthread_jit_write_protect_np(1);
        darwin_jit.sys_icache_invalidate(m, code.len);
    }
    return m[0..n];
}

// --------------------------------------------------------------- watchdog --

const ItimerVal = extern struct { interval_sec: isize, interval_usec: isize, value_sec: isize, value_usec: isize };
const darwin_timer = struct { // setitimer(2) of libc: Darwin and FreeBSD
    extern "c" fn setitimer(which: c_int, new: *const ItimerVal, old: ?*ItimerVal) c_int;
};

/// SIGALRM in `ms` milliseconds (0 disarms): the watchdog for generated code
/// that does not return. Darwin's `timeval` has a 32-bit microsecond field,
/// passed here in a full word (little-endian: the same bytes); FreeBSD's is
/// a full word.
pub fn armTimer(ms: u32) void {
    const v = ItimerVal{ .interval_sec = 0, .interval_usec = 0, .value_sec = @intCast(ms / 1000), .value_usec = @intCast((ms % 1000) * 1000) };
    if (is_linux) _ = linux.syscall3(.setitimer, 0, @intFromPtr(&v), 0) else _ = darwin_timer.setitimer(0, &v, null);
}

// ------------------------------------------------------------------ futex --

const darwin_ulock = struct {
    extern "c" fn __ulock_wait(op: u32, addr: ?*const anyopaque, value: u64, timeout_us: u32) c_int;
    extern "c" fn __ulock_wake(op: u32, addr: ?*const anyopaque, value: u64) c_int;
};
const UL_COMPARE_AND_WAIT: u32 = 1;
const ULF_WAKE_ALL: u32 = 0x100;
const ULF_NO_ERRNO: u32 = 0x0100_0000;

// FreeBSD: sys/umtx.h
const fbsd_umtx = struct {
    extern "c" fn _umtx_op(obj: ?*anyopaque, op: c_int, val: c_ulong, uaddr: ?*anyopaque, uaddr2: ?*anyopaque) c_int;
};
const UMTX_OP_WAIT_UINT_PRIVATE: c_int = 15;
const UMTX_OP_WAKE_PRIVATE: c_int = 16;

/// Sleep while `*addr == expect` (spurious wake-ups allowed).
pub fn futexWait(addr: *const u32, expect: u32) void {
    if (is_linux)
        _ = linux.futex_4arg(addr, .{ .cmd = .WAIT, .private = true }, expect, null)
    else if (is_freebsd)
        _ = fbsd_umtx._umtx_op(@constCast(addr), UMTX_OP_WAIT_UINT_PRIVATE, expect, null, null)
    else
        _ = darwin_ulock.__ulock_wait(UL_COMPARE_AND_WAIT | ULF_NO_ERRNO, addr, expect, 0);
}

/// Wake up to `n` sleepers on `addr` (Darwin: one, or all).
pub fn futexWake(addr: *const u32, n: u32) void {
    if (is_linux)
        _ = linux.futex_3arg(addr, .{ .cmd = .WAKE, .private = true }, n)
    else if (is_freebsd)
        _ = fbsd_umtx._umtx_op(@constCast(addr), UMTX_OP_WAKE_PRIVATE, n, null, null)
    else
        _ = darwin_ulock.__ulock_wake(UL_COMPARE_AND_WAIT | ULF_NO_ERRNO | (if (n > 1) ULF_WAKE_ALL else 0), addr, 0);
}
