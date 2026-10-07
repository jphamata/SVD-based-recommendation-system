//! vapor-worker — the native substrate, *outside* the BEAM.
//!
//! The predecessor jumped into generated machine code from a NIF, so one
//! illegal instruction or wild store killed the whole Erlang VM. Here that
//! code runs in this separate, unprivileged, seccomp-confined process,
//! spawned and supervised by an OTP port. A fault kills only the worker; the
//! BEAM observes the exit status (128 + signal), restarts it, and reroutes
//! the unit — never a node loss.
//!
//! Two execution modes per RUN:
//!   * native  — the code blob is mapped W^X (written RW, then flipped to
//!               R+X) and called as `void k(const uint64_t *args)`;
//!   * emulate — the blob is RV64GCV machine code interpreted by `rvemu`
//!               with every memory access bounds-checked; faults and
//!               illegal instructions come back as ERR frames.
//!
//! Protocol (stdio, `{packet, 4}`; see lib/vapor/runtime/worker.ex).

const std = @import("std");

/// Faults must surface as the real signal (SIGSEGV/SIGILL/SIGBUS) so the
/// BEAM can classify them; Zig's handler would turn them into SIGABRT.
pub const std_options: std.Options = .{ .enable_segfault_handler = false };
const builtin = @import("builtin");
const sys = @import("sys.zig");
const proto = @import("proto.zig");
const rvemu = @import("rvemu.zig");
const sandbox = @import("sandbox.zig");
const pool = @import("pool.zig");
const counters = @import("counters.zig");

const version: u32 = 3; // 3: sessions (OPEN, STEP, CLOSE)

const Op = enum(u8) { hello = 1, run = 2, emit = 3, done = 4, err = 5, open = 6, step = 7, close = 8 };

const ErrCode = enum(u32) {
    illegal_instruction = 1,
    memory_fault = 2,
    fuel_exhausted = 3,
    misaligned_fetch = 4,
    bad_frame = 5,
    io = 6,
    unsupported = 7,
};

const Kernel = *const fn ([*]const u64) callconv(.c) void;

const max_args = 20;

const Arg = union(enum) {
    imm: u64,
    buf: struct { idx: u32, off: u64 },
    iter: struct { idx: u32, base: u64, stride: u64 },
};

/// Partition descriptor: the call may run as contiguous chunks of the
/// count in argument `count_idx` (multiples of `grain`), each chunk with the
/// pointer arguments advanced by `stride` bytes per unit and each thread's
/// scratch arguments offset by `index · bytes`.
const Split = struct {
    count_idx: u8,
    /// 255, or an argument that must equal 1 for this split to apply
    guard_idx: u8,
    grain: u32,
    nptr: u8,
    ptrs: [max_args]struct { idx: u8, stride: u64 },
    nscr: u8,
    scr: [2]struct { idx: u8, bytes: u64 },
};

const Call = struct { entry: u32, nargs: u32, args: [max_args]Arg, splits: [2]Split, nsplit: u8 };
const Emit = struct { buf: u32, base: u64, stride: u64, len: u64 };
const Copy = struct { src: u32, soff: u64, dst: u32, doff: u64, len: u64 };

/// A run's tables (buffers, calls, emits, …) are sized by the frame, not by
/// compile-time maxima: a model has hundreds of buffers and calls. Each
/// count is first checked against the bytes left in the frame (every entry
/// occupies at least `min_bytes` of it), so a hostile count cannot make the
/// worker allocate more than a small multiple of what it was sent.
const Table = proto.Table;

const Buf = struct {
    mem: []u8 = &.{},
    writable: bool = false,
    region: ?proto.Region = null,
    mapped: ?[]align(sys.page) u8 = null,
};

fn arch() u8 {
    return switch (builtin.cpu.arch) {
        .x86_64 => 1,
        .aarch64 => 2,
        .riscv64 => 3,
        else => 0,
    };
}

pub fn main() void {
    while (true) {
        const frame = proto.readFrame(0) catch |e| switch (e) {
            error.Eof => sys.exit(0),
            else => sys.exit(3),
        };
        defer frame.region.free();
        const payload = frame.region.mem[0..frame.len];
        dispatch(payload) catch |e| {
            // Reply-level failure (malformed frame, I/O): report and go on.
            const code: ErrCode = switch (e) {
                error.Io => .io,
                else => .bad_frame,
            };
            sendErr(code, 0, 0, @errorName(e));
        };
    }
}

fn dispatch(payload: []const u8) !void {
    var c = proto.Cursor{ .buf = payload };
    const op = try c.int(u8);
    switch (op) {
        @intFromEnum(Op.hello) => {
            const flags = try c.int(u32);
            // threads exist before the filter; afterwards clone is forbidden
            const threads: u32 = c.int(u32) catch 1;
            if (pool.size() == 1) {
                counters.openForThisThread(0);
                if (threads > 1) pool.start(threads, canonicalFpEnv) catch {};
            }
            const st: sandbox.Status = if (flags & 1 != 0) sandbox.install() else .off;
            const reply = [_]u8{ @intFromEnum(Op.hello), arch(), @intFromEnum(st) } ++ proto.le(u32, version) ++
                proto.le(u32, pool.size());
            try proto.writeFrame(1, &.{&reply});
        },
        @intFromEnum(Op.run) => try run(&c),
        @intFromEnum(Op.open) => try open(&c),
        @intFromEnum(Op.step) => try step(&c),
        @intFromEnum(Op.close) => {
            closeSession();
            try done(0, 0, &.{0}, &.{}, &c);
        },
        else => return error.Truncated,
    }
}

fn sendErr(code: ErrCode, pc: u64, word: u32, msg: []const u8) void {
    const head = [_]u8{@intFromEnum(Op.err)} ++ proto.le(u32, @intFromEnum(code)) ++ proto.le(u64, pc) ++ proto.le(u32, word);
    proto.writeFrame(1, &.{ &head, msg }) catch sys.exit(4);
}

/// Code, buffers and emulator state: what a RUN builds and drops, and what a
/// session (OPEN … STEP* … CLOSE) keeps between steps.
const Machine = struct {
    code: proto.Region,
    code_len: u32,
    native: bool,
    bufs: Table(Buf),
    spans: Table(rvemu.Span),
    args_region: proto.Region,
    stack: proto.Region,
    hart: rvemu.Hart,

    fn args(self: *Machine) [*]u64 {
        return @ptrCast(@alignCast(self.args_region.mem.ptr));
    }

    fn deinit(self: *Machine) void {
        for (self.bufs.items) |b| {
            if (b.region) |r| r.free();
            if (b.mapped) |m| sys.unmap(m);
        }
        self.bufs.free();
        self.spans.free();
        self.args_region.free();
        self.stack.free();
        self.code.free();
    }
};

const Setup = union(enum) { ok: Machine, refused: []const u8 };

/// Parse `mode vlen flags fuel [deadline] code buffers` and build a machine.
fn setup(c: *proto.Cursor, with_deadline: bool, deadline: *u32) !Setup {
    const mode = try c.int(u8);
    const vlen = try c.int(u32);
    const flags = try c.int(u32);
    const fuel = try c.int(u64);
    deadline.* = if (with_deadline) try c.int(u32) else 0;
    const code_len = try c.int(u32);
    const code_src = try c.bytes(code_len);
    if (mode > 1) return .{ .refused = "unknown execution mode" };
    const native = mode == 0;
    const vlenb: u32 = if (vlen == 0) 16 else vlen / 8;
    if (!native and (vlenb < 16 or vlenb > rvemu.max_vlenb or vlenb & (vlenb - 1) != 0))
        return .{ .refused = "VLEN must be a power of two in [128, 512]" };

    // ---- buffers ----
    const nbuf = try c.int(u32);
    var bufs = try Table(Buf).init(c, nbuf, 10);
    @memset(bufs.items, .{});
    errdefer {
        for (bufs.items) |b| {
            if (b.region) |r| r.free();
            if (b.mapped) |m| sys.unmap(m);
        }
        bufs.free();
    }
    for (bufs.items) |*b| {
        const kind = try c.int(u8);
        b.writable = (try c.int(u8)) != 0;
        const len = try c.int(u64);
        switch (kind) {
            0, 1 => {
                const r = try proto.Region.alloc(len + 64);
                b.region = r;
                b.mem = r.mem[0..len];
                if (kind == 1) @memcpy(b.mem, try c.bytes(len));
            },
            2 => {
                const plen = try c.int(u16);
                const path = try c.bytes(plen);
                const off = try c.int(u64);
                b.mapped = try sys.mapFileAt(path, off, len, b.writable);
                const page_off = off & (sys.page - 1);
                b.mem = b.mapped.?[page_off .. page_off + len];
            },
            else => return error.Truncated,
        }
    }

    // ---- code: W^X, the page is never writable and executable at once ----
    const code = proto.Region{ .mem = try sys.loadCode(code_src, native) };
    errdefer code.free();

    // ---- emulator state (mode 1) ----
    const args_region = try proto.Region.alloc(max_args * 8);
    errdefer args_region.free();
    const stack = try proto.Region.alloc(64 * 1024);
    errdefer stack.free();
    const spans = try Table(rvemu.Span).init(c, nbuf + 2, 0);
    for (bufs.items, 0..) |b, i| spans.items[i] = .{ .base = @intFromPtr(b.mem.ptr), .len = b.mem.len, .writable = b.writable };
    spans.items[nbuf] = .{ .base = @intFromPtr(args_region.mem.ptr), .len = args_region.mem.len, .writable = false };
    spans.items[nbuf + 1] = .{ .base = @intFromPtr(stack.mem.ptr), .len = stack.mem.len, .writable = true };

    return .{ .ok = .{
        .code = code,
        .code_len = code_len,
        .native = native,
        .bufs = bufs,
        .spans = spans,
        .args_region = args_region,
        .stack = stack,
        .hart = .{
            .vlenb = vlenb,
            .code = code.mem[0..code_len],
            .code_base = @intFromPtr(code.mem.ptr),
            .spans = spans.items,
            .fuel = fuel,
            .poison = flags & 1 != 0,
        },
    } };
}

/// Parse a call list and check every reference against the machine's
/// buffers (Axiom 5): nothing is executed until all of it is in bounds.
fn parseCalls(c: *proto.Cursor, m: *const Machine) !Table(Call) {
    const ncall = try c.int(u32);
    const table = try Table(Call).init(c, ncall, 8);
    errdefer table.free();
    const bufs = m.bufs.items;
    for (table.items) |*k| {
        k.entry = try c.int(u32);
        k.nargs = try c.int(u32);
        if (k.nargs > max_args or k.entry >= m.code_len) return error.Truncated;
        for (k.args[0..k.nargs]) |*a| {
            a.* = switch (try c.int(u8)) {
                0 => .{ .imm = try c.int(u64) },
                1 => .{ .buf = .{ .idx = try c.int(u32), .off = try c.int(u64) } },
                2 => .{ .iter = .{ .idx = try c.int(u32), .base = try c.int(u64), .stride = try c.int(u64) } },
                else => return error.Truncated,
            };
            switch (a.*) {
                .imm => {},
                .buf => |b| if (b.idx >= bufs.len or b.off > bufs[b.idx].mem.len) return error.Truncated,
                .iter => |b| if (b.idx >= bufs.len) return error.Truncated,
            }
        }
        k.nsplit = try c.int(u8);
        if (k.nsplit > 2) return error.Truncated;
        for (k.splits[0..k.nsplit]) |*slot| {
            var sp: Split = undefined;
            sp.count_idx = try c.int(u8);
            sp.guard_idx = try c.int(u8);
            if (sp.guard_idx != 255 and (sp.guard_idx >= k.nargs or k.args[sp.guard_idx] != .imm)) return error.Truncated;
            sp.grain = @max(try c.int(u32), 1);
            sp.nptr = try c.int(u8);
            if (sp.count_idx >= k.nargs or k.args[sp.count_idx] != .imm or sp.nptr > sp.ptrs.len) return error.Truncated;
            for (sp.ptrs[0..sp.nptr]) |*q| {
                q.* = .{ .idx = try c.int(u8), .stride = try c.int(u64) };
                if (q.idx >= k.nargs or k.args[q.idx] == .imm) return error.Truncated;
            }
            sp.nscr = try c.int(u8);
            if (sp.nscr > sp.scr.len) return error.Truncated;
            for (sp.scr[0..sp.nscr]) |*q| {
                q.* = .{ .idx = try c.int(u8), .bytes = try c.int(u64) };
                // every thread's scratch lies inside the buffer
                const a = if (q.idx < k.nargs) k.args[q.idx] else return error.Truncated;
                switch (a) {
                    .buf => |b| if (b.off + q.bytes * pool.size() > bufs[b.idx].mem.len) return error.Truncated,
                    else => return error.Truncated,
                }
            }
            slot.* = sp;
        }
    }
    return table;
}

/// Per-iteration arguments stay inside their buffer for every iteration.
fn checkIters(calls: []const Call, bufs: []const Buf, iters: u32) !void {
    if (iters == 0) return;
    for (calls) |k| for (k.args[0..k.nargs]) |a| switch (a) {
        .iter => |b| if (b.base + b.stride * (iters - 1) > bufs[b.idx].mem.len) return error.Truncated,
        else => {},
    };
}

fn resolve(k: *const Call, bufs: []const Buf, t: u32, out: [*]u64) void {
    for (k.args[0..k.nargs], 0..) |a, j| out[j] = switch (a) {
        .imm => |v| v,
        .buf => |b| @intFromPtr(bufs[b.idx].mem.ptr) + b.off,
        .iter => |b| @intFromPtr(bufs[b.idx].mem.ptr) + b.base + b.stride * t,
    };
}

const Chunk = struct { m: *const Machine, k: *const Call, sp: *const Split, t: u32 };

/// One thread's share of a split call: a contiguous range of whole grains.
fn chunk(ctx: *const anyopaque, index: u32, n: u32) void {
    const job: *const Chunk = @ptrCast(@alignCast(ctx));
    const k = job.k;
    const sp = job.sp.*;
    var a: [max_args]u64 = undefined;
    resolve(k, job.m.bufs.items, job.t, &a);
    const total = a[sp.count_idx];
    const grains = (total + sp.grain - 1) / sp.grain;
    const per = (grains + n - 1) / n * sp.grain;
    const r0 = @min(total, per * index);
    const r1 = @min(total, r0 + per);
    if (r0 >= r1) return;
    a[sp.count_idx] = r1 - r0;
    for (sp.ptrs[0..sp.nptr]) |q| a[q.idx] += r0 * q.stride;
    for (sp.scr[0..sp.nscr]) |q| a[q.idx] += index * q.bytes;
    const f: Kernel = @ptrFromInt(@intFromPtr(job.m.code.mem.ptr) + k.entry);
    f(&a);
}

/// The first descriptor whose guard holds and whose count has at least two
/// grains (fewer is not worth a hand-off).
fn applicable(k: *const Call, args: [*]const u64) ?*const Split {
    for (k.splits[0..k.nsplit]) |*sp| {
        if (sp.guard_idx != 255 and args[sp.guard_idx] != 1) continue;
        if (args[sp.count_idx] >= 2 * @as(u64, sp.grain)) return sp;
    }
    return null;
}

/// Execute one pass over the calls (iteration `t`); an emulator fault is
/// reported as an ERR frame and returns false. Native calls with a
/// partition descriptor and at least two grains per thread run on the pool.
fn execute(m: *Machine, calls: []const Call, t: u32) bool {
    const args = m.args();
    const bufs = m.bufs.items;
    for (calls) |*k| {
        resolve(k, bufs, t, args);
        if (m.native and pool.size() > 1) {
            if (applicable(k, args)) |sp| {
                const job = Chunk{ .m = m, .k = k, .sp = sp, .t = t };
                pool.run(.{ .ctx = &job, .run = chunk });
                continue;
            }
        }
        if (m.native) {
            const f: Kernel = @ptrFromInt(@intFromPtr(m.code.mem.ptr) + k.entry);
            f(args);
        } else {
            m.hart.call(k.entry, @intFromPtr(args), @intFromPtr(m.stack.mem.ptr) + m.stack.mem.len) catch |e| {
                const ec: ErrCode = switch (e) {
                    error.IllegalInstruction => .illegal_instruction,
                    error.MemoryFault => .memory_fault,
                    error.FuelExhausted => .fuel_exhausted,
                    error.MisalignedFetch => .misaligned_fetch,
                };
                sendErr(ec, m.hart.fault_pc -% m.hart.code_base, m.hart.fault_word, @errorName(e));
                return false;
            };
        }
    }
    return true;
}

/// The canonical results assume IEEE defaults: round-to-nearest-even,
/// subnormals preserved, exceptions masked — established before every run
/// so no inherited state can change a single bit. Native code cannot be
/// interrupted cooperatively, so a wall-clock deadline arms SIGALRM, whose
/// default action ends the process (the BEAM sees exit status 128+14).
fn begin(m: *const Machine, deadline_ms: u32) void {
    if (m.native) canonicalFpEnv();
    armWatchdog(deadline_ms);
}

/// DONE: timing, retired instructions (emulated), event counters, then
/// `(len, bytes)` per slice.
fn done(elapsed: u64, retired: u64, events: []const u8, slices: []const []const u8, c: *const proto.Cursor) !void {
    const parts = try Table([]const u8).init(c, @intCast(2 * slices.len + 2), 0);
    defer parts.free();
    const lens = try Table([8]u8).init(c, @intCast(slices.len), 0);
    defer lens.free();
    const head = [_]u8{@intFromEnum(Op.done)} ++ proto.le(u64, elapsed) ++ proto.le(u64, retired);
    parts.items[0] = &head;
    parts.items[1] = events;
    for (slices, 0..) |sl, i| {
        lens.items[i] = proto.le(u64, sl.len);
        parts.items[2 + 2 * i] = &lens.items[i];
        parts.items[3 + 2 * i] = sl;
    }
    try proto.writeFrame(1, parts.items);
}

/// RUN: a stateless unit — machine, calls, iterations, emits, state copies,
/// results — built, executed and dropped.
fn run(c: *proto.Cursor) !void {
    var deadline_ms: u32 = 0;
    var m = switch (try setup(c, true, &deadline_ms)) {
        .ok => |mm| mm,
        .refused => |why| return sendErr(.unsupported, 0, 0, why),
    };
    defer m.deinit();
    const bufs = m.bufs.items;

    const call_table = try parseCalls(c, &m);
    defer call_table.free();
    const iters = try c.int(u32);
    try checkIters(call_table.items, bufs, iters);

    const nemit = try c.int(u32);
    const emit_table = try Table(Emit).init(c, nemit, 28);
    defer emit_table.free();
    const emits = emit_table.items;
    for (emits) |*e| e.* = .{ .buf = try c.int(u32), .base = try c.int(u64), .stride = try c.int(u64), .len = try c.int(u64) };
    const ncopy = try c.int(u32);
    const copy_table = try Table(Copy).init(c, ncopy, 32);
    defer copy_table.free();
    const copies = copy_table.items;
    for (copies) |*k| k.* = .{ .src = try c.int(u32), .soff = try c.int(u64), .dst = try c.int(u32), .doff = try c.int(u64), .len = try c.int(u64) };
    const nret = try c.int(u32);
    const ret_table = try Table([]const u8).init(c, nret, 4);
    defer ret_table.free();
    for (ret_table.items) |*r| {
        const i = try c.int(u32);
        if (i >= bufs.len) return error.Truncated;
        r.* = bufs[i].mem;
    }
    for (emits) |e| {
        if (e.buf >= bufs.len) return error.Truncated;
        if (iters > 0 and e.base + e.stride * (iters - 1) + e.len > bufs[e.buf].mem.len) return error.Truncated;
    }
    for (copies) |k| {
        if (k.src >= bufs.len or k.dst >= bufs.len) return error.Truncated;
        if (k.soff + k.len > bufs[k.src].mem.len or k.doff + k.len > bufs[k.dst].mem.len) return error.Truncated;
    }
    const part_table = try Table([]const u8).init(c, nemit + 1, 0);
    defer part_table.free();
    const parts = part_table.items;

    begin(&m, deadline_ms);
    defer armWatchdog(0);
    const ev0 = counters.read();
    const t0 = proto.monotonicNs();
    var t: u32 = 0;
    while (t < iters) : (t += 1) {
        if (!execute(&m, call_table.items, t)) return;
        if (nemit > 0) {
            const head = [_]u8{@intFromEnum(Op.emit)} ++ proto.le(u32, t);
            parts[0] = &head;
            for (emits, 1..) |e, i| {
                const o = e.base + e.stride * t;
                parts[i] = bufs[e.buf].mem[o .. o + e.len];
            }
            try proto.writeFrame(1, parts[0 .. nemit + 1]);
        }
        for (copies) |k|
            @memmove(bufs[k.dst].mem[k.doff .. k.doff + k.len], bufs[k.src].mem[k.soff .. k.soff + k.len]);
    }
    const elapsed = proto.monotonicNs() - t0;
    var evbuf: [1 + 9 * counters.n_events]u8 = undefined;
    try done(elapsed, m.hart.retired, counters.encode(ev0, counters.read(), &evbuf), ret_table.items, c);
}

// ------------------------------------------------------------ sessions --
//
// OPEN builds a machine and keeps it: weights stay mapped and KV caches stay
// in worker memory across decode steps, so a step moves only its inputs in
// and its selected outputs out. A crash ends the session with the process;
// the BEAM learns of it from the exit status like any other fault.

var session: ?Machine = null;

fn closeSession() void {
    if (session) |*m| m.deinit();
    session = null;
}

/// OPEN: `mode vlen flags fuel code buffers` (a RUN's machine part).
fn open(c: *proto.Cursor) !void {
    closeSession();
    var deadline_ms: u32 = 0;
    switch (try setup(c, false, &deadline_ms)) {
        .ok => |m| session = m,
        .refused => |why| return sendErr(.unsupported, 0, 0, why),
    }
    try done(0, 0, &.{0}, &.{}, c);
}

/// STEP: `deadline writes calls returns [copies]` with
/// writes  = n × (buf:u32, off:u64, len:u64, bytes)
/// returns = n × (buf:u32, off:u64, len:u64)
/// copies  = n × (src:u32, soff:u64, dst:u32, doff:u64, len:u64), optional:
///           state feedback (`h ← h_next`) after the pass, as in RUN — a
///           recurrent model's state stays in the session between steps
fn step(c: *proto.Cursor) !void {
    const m = if (session) |*s| s else return sendErr(.bad_frame, 0, 0, "no open session");
    const bufs = m.bufs.items;
    const deadline_ms = try c.int(u32);

    const nw = try c.int(u32);
    if (@as(usize, nw) * 20 > c.buf.len - c.pos) return error.Truncated;
    var w: u32 = 0;
    while (w < nw) : (w += 1) {
        const i = try c.int(u32);
        const off = try c.int(u64);
        const len = try c.int(u64);
        const bytes = try c.bytes(len);
        if (i >= bufs.len or off + len > bufs[i].mem.len or !bufs[i].writable) return error.Truncated;
        @memcpy(bufs[i].mem[off .. off + len], bytes);
    }

    const call_table = try parseCalls(c, m);
    defer call_table.free();
    try checkIters(call_table.items, bufs, 1);

    const nret = try c.int(u32);
    const ret_table = try Table([]const u8).init(c, nret, 20);
    defer ret_table.free();
    for (ret_table.items) |*r| {
        const i = try c.int(u32);
        const off = try c.int(u64);
        const len = try c.int(u64);
        if (i >= bufs.len or off + len > bufs[i].mem.len) return error.Truncated;
        r.* = bufs[i].mem[off .. off + len];
    }

    const ncopy: u32 = if (c.pos < c.buf.len) try c.int(u32) else 0;
    const copy_table = try Table(Copy).init(c, ncopy, 32);
    defer copy_table.free();
    const copies = copy_table.items;
    for (copies) |*k| {
        k.* = .{ .src = try c.int(u32), .soff = try c.int(u64), .dst = try c.int(u32), .doff = try c.int(u64), .len = try c.int(u64) };
        if (k.src >= bufs.len or k.dst >= bufs.len or !bufs[k.dst].writable) return error.Truncated;
        if (k.soff + k.len > bufs[k.src].mem.len or k.doff + k.len > bufs[k.dst].mem.len) return error.Truncated;
    }

    begin(m, deadline_ms);
    defer armWatchdog(0);
    const ev0 = counters.read();
    const t0 = proto.monotonicNs();
    if (!execute(m, call_table.items, 0)) return;
    for (copies) |k|
        @memmove(bufs[k.dst].mem[k.doff .. k.doff + k.len], bufs[k.src].mem[k.soff .. k.soff + k.len]);
    const elapsed = proto.monotonicNs() - t0;
    var evbuf: [1 + 9 * counters.n_events]u8 = undefined;
    try done(elapsed, m.hart.retired, counters.encode(ev0, counters.read(), &evbuf), ret_table.items, c);
}

fn canonicalFpEnv() void {
    switch (builtin.cpu.arch) {
        .x86_64 => {
            const mxcsr: u32 = 0x1F80; // all exceptions masked, RNE, FTZ = DAZ = 0
            asm volatile ("ldmxcsr (%[p])"
                :
                : [p] "r" (&mxcsr),
                : .{ .memory = true });
        },
        .aarch64 => asm volatile ("msr fpcr, xzr"), // RNE, FZ = DN = AHP = 0
        .riscv64 => asm volatile ("csrwi frm, 0"), // dynamic rounding mode = RNE
        else => {},
    }
}

fn armWatchdog(ms: u32) void {
    sys.armTimer(ms);
}

test {
    _ = rvemu;
}
