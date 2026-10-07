//! Hardware and software event counters (perf_event_open), when the kernel
//! and the machine provide them. Each thread of the worker opens its own
//! counters before the seccomp filter is installed (perf_event_open is not
//! on the allowlist); afterwards only `read` is needed. A counter the host
//! refuses (no PMU in a VM, perf_event_paranoid, a container policy) is
//! simply absent from the report — never an error.
const std = @import("std");
const linux = std.os.linux;

pub const Id = enum(u8) { cycles = 1, instructions = 2, cache_misses = 3, task_clock_ns = 4, page_faults = 5, context_switches = 6 };

const events = [_]struct { id: Id, type: linux.PERF.TYPE, config: u64 }{
    .{ .id = .cycles, .type = .HARDWARE, .config = 0 },
    .{ .id = .instructions, .type = .HARDWARE, .config = 1 },
    .{ .id = .cache_misses, .type = .HARDWARE, .config = 3 },
    .{ .id = .task_clock_ns, .type = .SOFTWARE, .config = 1 },
    .{ .id = .page_faults, .type = .SOFTWARE, .config = 2 },
    .{ .id = .context_switches, .type = .SOFTWARE, .config = 3 },
};

pub const n_events = 6;
comptime {
    if (events.len != n_events) @compileError("n_events");
}
const builtin = @import("builtin");
const have = builtin.os.tag == .linux;
const max_threads = 64;

var fds: [max_threads][n_events]i32 = [_][n_events]i32{[_]i32{-1} ** n_events} ** max_threads;
var used = std.atomic.Value(u32).init(0);

/// Open this thread's counters (user-space events of the calling thread).
pub fn openForThisThread(slot: u32) void {
    if (!have or slot >= max_threads) return;
    for (events, 0..) |e, i| {
        var attr = linux.perf_event_attr{ .type = e.type, .config = e.config };
        attr.flags.exclude_kernel = true;
        attr.flags.exclude_hv = true;
        const rc = linux.perf_event_open(&attr, 0, -1, -1, 0);
        fds[slot][i] = if (linux.errno(rc) == .SUCCESS) @intCast(rc) else -1;
    }
    _ = used.fetchAdd(1, .acq_rel);
}

/// Sum of every thread's counter, per event (`null` when unavailable).
pub fn read() [n_events]?u64 {
    var out: [n_events]?u64 = [_]?u64{null} ** n_events;
    const n = @min(used.load(.acquire), max_threads);
    var t: u32 = 0;
    while (t < n) : (t += 1) {
        for (0..n_events) |i| {
            const fd = fds[t][i];
            if (fd < 0) continue;
            var v: u64 = 0;
            const rc = linux.read(fd, @ptrCast(&v), 8);
            if (linux.errno(rc) == .SUCCESS and rc == 8) out[i] = (out[i] orelse 0) + v;
        }
    }
    return out;
}

/// `(id, delta)` pairs for the counters present in both readings.
pub fn encode(before: [n_events]?u64, after: [n_events]?u64, buf: *[1 + 9 * n_events]u8) []const u8 {
    var n: u8 = 0;
    var at: usize = 1;
    for (events, 0..) |e, i| {
        if (before[i] != null and after[i] != null) {
            buf[at] = @intFromEnum(e.id);
            std.mem.writeInt(u64, buf[at + 1 ..][0..8], after[i].? -% before[i].?, .little);
            at += 9;
            n += 1;
        }
    }
    buf[0] = n;
    return buf[0..at];
}
