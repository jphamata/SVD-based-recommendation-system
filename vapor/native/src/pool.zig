//! Intra-operation data parallelism for the worker: a fixed pool of threads
//! created at HELLO, *before* the seccomp filter is installed (the filter is
//! then synchronised onto every thread, and `clone` stays forbidden).
//!
//! A job runs one function on every thread with its index (the caller is
//! index 0). Hand-off and completion use a generation counter and a pending
//! count: waiters spin briefly (a kernel call is typically microseconds) and
//! then sleep on a futex, so an idle pool costs nothing.
//!
//! Determinism is the caller's: work is split into contiguous row ranges
//! whose results do not depend on the split (vapor's kernels compute every
//! row with the same instructions in the same order).
const std = @import("std");
const sys = @import("sys.zig");
const counters = @import("counters.zig");

pub const max_threads = 64;

pub const Job = struct {
    ctx: *const anyopaque,
    run: *const fn (ctx: *const anyopaque, index: u32, count: u32) void,
};

var count: u32 = 1;
var gen = std.atomic.Value(u32).init(0);
var pending = std.atomic.Value(u32).init(0);
var job: Job = undefined;
var ready = std.atomic.Value(u32).init(0);

// Waiters spin for up to `spin_ns` before sleeping on the futex: a VM's
// futex wake-up costs ~100 µs, more than a whole decode-step kernel, and
// the calls of a step (and the steps of a busy engine) arrive back to back.
const spin_ns: u64 = 2_000_000;

fn nowNs() u64 {
    return sys.monotonicNs();
}

/// Spin (checking the clock every 64 rounds) until `v` differs from
/// `seen` or the budget is spent, then sleep on the futex.
fn await(v: *const std.atomic.Value(u32), seen: u32) u32 {
    const t0 = nowNs();
    var spins: u32 = 0;
    var x = v.load(.acquire);
    while (x == seen) : (spins += 1) {
        if (spins & 63 == 63 and nowNs() - t0 > spin_ns) {
            futexWait(v, seen);
        } else std.atomic.spinLoopHint();
        x = v.load(.acquire);
    }
    return x;
}

fn futexWait(v: *const std.atomic.Value(u32), expect: u32) void {
    sys.futexWait(&v.raw, expect);
}

fn futexWake(v: *const std.atomic.Value(u32), n: u32) void {
    sys.futexWake(&v.raw, n);
}

fn loop(index: u32, fp_init: *const fn () void) void {
    fp_init();
    counters.openForThisThread(index);
    _ = ready.fetchAdd(1, .release);
    var seen: u32 = 0;
    while (true) {
        seen = await(&gen, seen);
        job.run(job.ctx, index, count);
        if (pending.fetchSub(1, .acq_rel) == 1) futexWake(&pending, 1);
    }
}

/// Start `n - 1` helper threads (so `n` run a job). `fp_init` establishes a
/// thread's floating-point environment before it runs anything.
pub fn start(n: u32, fp_init: *const fn () void) !void {
    const want = @min(@max(n, 1), max_threads);
    var i: u32 = 1;
    while (i < want) : (i += 1) {
        const t = try std.Thread.spawn(.{ .stack_size = 256 * 1024 }, loop, .{ i, fp_init });
        t.detach();
        count = i + 1;
    }
    // every thread has finished starting (its remaining system calls are
    // futex waits) before the caller installs the seccomp filter
    while (ready.load(.acquire) != count - 1) std.atomic.spinLoopHint();
}

pub fn size() u32 {
    return count;
}

/// Run `j` on every thread (the caller included) and return when all are done.
pub fn run(j: Job) void {
    if (count == 1) return j.run(j.ctx, 0, 1);
    job = j;
    pending.store(count - 1, .release);
    _ = gen.fetchAdd(1, .release);
    futexWake(&gen, max_threads);
    j.run(j.ctx, 0, count);
    var p = pending.load(.acquire);
    while (p != 0) p = await(&pending, p);
}
