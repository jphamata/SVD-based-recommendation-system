//! vapor-metal — the GPU daemon for unified-memory devices (Apple Metal).
//!
//! It speaks the fabric protocol (see fabric.zig and
//! lib/vapor/runtime/fabric.ex) frame for frame — HELLO, RUN, OPEN/STEP/CLOSE
//! — with one difference: modules are Metal Shading Language source, not
//! SPIR-V words (`Vapor.Emit.MSL` translates the same kernel library), and
//! the HELLO reply ends with a format byte (1 = MSL) that tells the BEAM so.
//!
//! On unified memory there is nothing to stage: every buffer is shared, the
//! host writes inputs and reads outputs in place, and read-only weight
//! files are wrapped without a copy. A RUN records, for each iteration, the
//! window copies (blit), the dispatches (one serial compute encoder: each
//! dispatch sees the previous one's writes), the emit copies and the state
//! copies — the order of fabric.zig — commits once, and waits with a
//! deadline (a hung GPU is reported as a lost device; the BEAM respawns us).
//!
//! The device work is behind a small backend interface:
//!   * `mtl.zig`    — Metal through the Objective-C runtime (macOS);
//!   * `mslsim.zig` — a Linux stand-in that runs the same MSL on the CPU,
//!     from shared objects clang built out of the exact source text (tests
//!     only: it never compiles anything itself).
//! Everything in this file — parsing, validation, the plan of a RUN, the
//! sessions — is the same code in both, and is what the Linux tests execute.

const std = @import("std");
const sys = @import("sys.zig");
const proto = @import("proto.zig");
const be = if (sys.is_darwin) @import("mtl.zig") else @import("mslsim.zig");

pub const std_options: std.Options = .{ .enable_segfault_handler = false };

const Op = enum(u8) { hello = 1, run = 2, emit = 3, done = 4, err = 5, fault = 9, open = 10, step = 11, close = 12 };

const max_bind = 16;
const max_push = 32;
const format_msl: u8 = 1;

var info: be.Info = undefined;
var fault_injection = false;
var init_error: []const u8 = "";

pub fn main(init: std.process.Init) void {
    const args = init.minimal.args.toSlice(init.arena.allocator()) catch &.{};
    var opts = be.Options{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a: []const u8 = args[i];
        if (std.mem.eql(u8, a, "--fault-injection")) fault_injection = true else if (i + 1 < args.len and be.option(&opts, a, args[i + 1])) i += 1 else _ = be.flag(&opts, a);
    }
    const ready = if (be.init(opts)) |inf| blk: {
        info = inf;
        break :blk true;
    } else |e| blk: {
        init_error = @errorName(e);
        break :blk false;
    };

    while (true) {
        const frame = proto.readFrame(0) catch |e| switch (e) {
            error.Eof => sys.exit(0),
            else => sys.exit(3),
        };
        defer frame.region.free();
        const pool = be.poolPush();
        defer be.poolPop(pool);
        dispatch(frame.region.mem[0..frame.len], ready) catch |e| {
            if (e == error.DeviceLost) {
                sendErr(2, 0, "device lost");
                sys.exit(5);
            }
            sendErr(1, 0, @errorName(e));
        };
    }
}

fn sendErr(code: u32, res: i32, msg: []const u8) void {
    const head = [_]u8{@intFromEnum(Op.err)} ++ proto.le(u32, code) ++ proto.le(u64, @as(u64, @bitCast(@as(i64, res)))) ++ proto.le(u32, 0);
    const detail = be.lastError();
    proto.writeFrame(1, &.{ &head, msg, if (detail.len > 0) ": " else "", detail }) catch sys.exit(4);
}

fn dispatch(payload: []const u8, ready: bool) !void {
    var c = proto.Cursor{ .buf = payload };
    switch (try c.int(u8)) {
        @intFromEnum(Op.hello) => {
            _ = try c.int(u32);
            const name = if (ready) info.name else init_error;
            const head = [_]u8{ @intFromEnum(Op.hello), @intFromBool(ready) } ++
                proto.le(u32, 0) ++ proto.le(u32, if (ready) info.vendor else 0) ++ proto.le(u32, 0) ++
                // denormals are not declared but measured (Vapor.Substrate); unified memory imports host files
                [_]u8{ 0, 1, 0 } ++ proto.le(u16, @intCast(name.len));
            try proto.writeFrame(1, &.{ &head, name, &[_]u8{format_msl} });
        },
        @intFromEnum(Op.run) => {
            if (!ready) return error.Unavailable;
            try run(&c);
        },
        @intFromEnum(Op.open) => {
            if (!ready) return error.Unavailable;
            try openSession(&c);
        },
        @intFromEnum(Op.step) => {
            if (!ready) return error.Unavailable;
            try stepSession(&c);
        },
        @intFromEnum(Op.close) => {
            const sid = try c.int(u32);
            if (sid < max_sessions) sessions[sid].deinit();
            try proto.writeFrame(1, &.{&([_]u8{@intFromEnum(Op.done)} ++ proto.le(u64, 0) ++ proto.le(u64, 0) ++ [_]u8{0})});
        },
        @intFromEnum(Op.fault) => {
            if (!fault_injection) return error.FaultInjectionDisabled;
            const p: *volatile u32 = @ptrFromInt(0x10);
            p.* = 0xDEAD;
        },
        else => return error.Truncated,
    }
}

// ----------------------------------------------------------- shared parts --

const Table = proto.Table;
const BufSpec = struct { kind: u8, writable: bool = true, len: u64, data: []const u8 = &.{}, path: []const u8 = &.{} };
const Module = struct { src: []const u8, nbind: u32, npush: u32 };
const Bind = union(enum) {
    buf: u32,
    window: struct { idx: u32, base: u64, stride: u64, len: u64, win: u32 },
};
const Disp = struct { module: u32, groups: [3]u32, npush: u32, push: [max_push]u32, nbind: u32, binds: [max_bind]Bind };
const Emit = struct { buf: u32, base: u64, stride: u64, len: u64, stage: u32 };
const Copy = struct { src: u32, soff: u64, dst: u32, doff: u64, len: u64 };

fn parseModules(c: *proto.Cursor) !Table(Module) {
    const nmod = try c.int(u32);
    const mods = try Table(Module).init(c, nmod, 12);
    errdefer mods.free();
    for (mods.items) |*m| {
        const len = try c.int(u32);
        m.src = try c.bytes(len);
        m.nbind = try c.int(u32);
        m.npush = try c.int(u32);
        if (m.nbind > max_bind or m.npush > max_push) return error.Truncated;
    }
    return mods;
}

fn parseBuffers(c: *proto.Cursor) !Table(BufSpec) {
    const nbuf = try c.int(u32);
    const specs = try Table(BufSpec).init(c, nbuf, 10);
    errdefer specs.free();
    for (specs.items) |*b| {
        const kind = try c.int(u8);
        const writable = try c.int(u8) != 0;
        const len = try c.int(u64);
        b.* = .{ .kind = kind, .writable = writable, .len = len };
        switch (kind) {
            0 => {},
            1 => b.data = try c.bytes(len),
            2 => {
                b.path = try c.bytes(try c.int(u16));
                if (try c.int(u64) != 0) return error.Truncated;
            },
            else => return error.Truncated,
        }
    }
    return specs;
}

/// A frame buffer as a device buffer: zeroed, filled, or a weight file
/// (wrapped without a copy when it is read-only, copied otherwise).
fn makeBuffer(b: BufSpec) !be.Buffer {
    return switch (b.kind) {
        0 => try be.newBuffer(b.len),
        1 => blk: {
            const buf = try be.newBuffer(b.len);
            @memcpy(buf.ptr[0..b.len], b.data);
            break :blk buf;
        },
        else => blk: {
            const m = try sys.mapFile(b.path, b.len, be.wrap_align);
            if (!b.writable) {
                if (be.wrap(m, b.len)) |w| break :blk w else |_| {}
            }
            defer sys.unmap(m);
            const buf = try be.newBuffer(b.len);
            @memcpy(buf.ptr[0..b.len], m[0..b.len]);
            break :blk buf;
        },
    };
}

// ---------------------------------------------------------------- RUN --

const RunState = struct {
    bufs: []be.Buffer = &.{},
    nbuf: u32 = 0,
    pipes: []be.Pipeline = &.{},
    npipe: u32 = 0,
    regions: [2]?proto.Region = .{ null, null },

    fn deinit(s: *RunState) void {
        for (s.pipes[0..s.npipe]) |p| be.freePipeline(p);
        for (s.bufs[0..s.nbuf]) |b| be.freeBuffer(b);
        for (s.regions) |r| if (r) |x| x.free();
    }
};

fn run(c: *proto.Cursor) !void {
    var s = RunState{};
    defer s.deinit();
    const deadline_ms = try c.int(u32);

    const mods = try parseModules(c);
    defer mods.free();
    const nmod: u32 = @intCast(mods.items.len);
    const specs = try parseBuffers(c);
    defer specs.free();
    const nbuf: u32 = @intCast(specs.items.len);

    const ndisp = try c.int(u32);
    const disps = try Table(Disp).init(c, ndisp, 24);
    defer disps.free();
    var nwin: u32 = 0;
    for (disps.items) |*dp| {
        dp.module = try c.int(u32);
        if (dp.module >= nmod) return error.Truncated;
        for (&dp.groups) |*g| g.* = try c.int(u32);
        dp.npush = try c.int(u32);
        if (dp.npush != mods.items[dp.module].npush) return error.Truncated;
        for (dp.push[0..dp.npush]) |*p| p.* = try c.int(u32);
        dp.nbind = try c.int(u32);
        if (dp.nbind != mods.items[dp.module].nbind) return error.Truncated;
        for (dp.binds[0..dp.nbind]) |*bd| {
            bd.* = switch (try c.int(u8)) {
                1 => .{ .buf = try c.int(u32) },
                2 => .{ .window = .{ .idx = try c.int(u32), .base = try c.int(u64), .stride = try c.int(u64), .len = try c.int(u64), .win = 0 } },
                else => return error.Truncated,
            };
            switch (bd.*) {
                .buf => |i| if (i >= nbuf) return error.Truncated,
                .window => |*w| {
                    if (w.idx >= nbuf) return error.Truncated;
                    w.win = nbuf + nwin;
                    nwin += 1;
                },
            }
        }
    }

    const iters = try c.int(u32);
    const nemit = try c.int(u32);
    const emits = try Table(Emit).init(c, nemit, 28);
    defer emits.free();
    for (emits.items, 0..) |*e, k| {
        e.* = .{ .buf = try c.int(u32), .base = try c.int(u64), .stride = try c.int(u64), .len = try c.int(u64), .stage = nbuf + nwin + @as(u32, @intCast(k)) };
        if (e.buf >= nbuf) return error.Truncated;
    }
    const ncopy = try c.int(u32);
    const copies = try Table(Copy).init(c, ncopy, 32);
    defer copies.free();
    for (copies.items) |*k| {
        k.* = .{ .src = try c.int(u32), .soff = try c.int(u64), .dst = try c.int(u32), .doff = try c.int(u64), .len = try c.int(u64) };
        if (k.src >= nbuf or k.dst >= nbuf) return error.Truncated;
    }
    const nret = try c.int(u32);
    const rets = try Table(u32).init(c, nret, 4);
    defer rets.free();
    for (rets.items) |*r| {
        r.* = try c.int(u32);
        if (r.* >= nbuf) return error.Truncated;
    }

    // ---- buffers: the frame's, then a window per windowed binding, then emit stages ----
    const all = try proto.Region.alloc(@as(usize, nbuf + nwin + nemit) * @sizeOf(be.Buffer));
    s.regions[0] = all;
    s.bufs = @as([*]be.Buffer, @ptrCast(@alignCast(all.mem.ptr)))[0 .. nbuf + nwin + nemit];
    for (specs.items, 0..) |b, i| {
        s.bufs[i] = try makeBuffer(b);
        s.nbuf += 1;
    }
    for (disps.items) |dp| for (dp.binds[0..dp.nbind]) |bd| switch (bd) {
        .window => |w| {
            s.bufs[w.win] = try be.newBuffer(w.len);
            s.nbuf += 1;
        },
        .buf => {},
    };
    for (emits.items) |e| {
        s.bufs[e.stage] = try be.newBuffer(e.len * @max(iters, 1));
        s.nbuf += 1;
    }

    // ---- pipelines ----
    const pr = try proto.Region.alloc(@as(usize, @max(nmod, 1)) * @sizeOf(be.Pipeline));
    s.regions[1] = pr;
    s.pipes = @as([*]be.Pipeline, @ptrCast(@alignCast(pr.mem.ptr)))[0..nmod];
    for (mods.items, 0..) |m, k| {
        s.pipes[k] = try be.compile(m.src, m.nbind, m.npush);
        s.npipe += 1;
    }

    // ---- record: for t in iterations — windows, dispatches, emits, state copies ----
    var cmd = try be.begin();
    var t: u32 = 0;
    var binds: [max_bind]be.Buffer = undefined;
    while (t < iters) : (t += 1) {
        for (disps.items) |dp| {
            for (dp.binds[0..dp.nbind], 0..) |bd, k| switch (bd) {
                .window => |w| {
                    be.copy(&cmd, s.bufs[w.idx], w.base + w.stride * t, s.bufs[w.win], 0, w.len);
                    binds[k] = s.bufs[w.win];
                },
                .buf => |x| binds[k] = s.bufs[x],
            };
            be.dispatch(&cmd, s.pipes[dp.module], dp.groups, dp.push[0..dp.npush], binds[0..dp.nbind]);
        }
        for (emits.items) |e| be.copy(&cmd, s.bufs[e.buf], e.base + e.stride * t, s.bufs[e.stage], e.len * t, e.len);
        for (copies.items) |k| be.copy(&cmd, s.bufs[k.src], k.soff, s.bufs[k.dst], k.doff, k.len);
    }
    const t0 = proto.monotonicNs();
    try be.commitWait(&cmd, deadline_ms);
    const elapsed = proto.monotonicNs() - t0;

    // ---- emits (per iteration), then DONE with the returned buffers ----
    const parts = try Table([]const u8).init(c, @max(nemit + 1, 1 + 2 * nret), 0);
    defer parts.free();
    t = 0;
    while (t < iters and nemit > 0) : (t += 1) {
        const head = [_]u8{@intFromEnum(Op.emit)} ++ proto.le(u32, t);
        parts.items[0] = &head;
        for (emits.items, 1..) |e, i| parts.items[i] = s.bufs[e.stage].ptr[e.len * t .. e.len * (t + 1)];
        try proto.writeFrame(1, parts.items[0 .. nemit + 1]);
    }
    const lens = try Table([8]u8).init(c, nret, 0);
    defer lens.free();
    const head = [_]u8{@intFromEnum(Op.done)} ++ proto.le(u64, elapsed) ++ proto.le(u64, 0) ++ [_]u8{0};
    parts.items[0] = &head;
    for (rets.items, 0..) |r, i| {
        lens.items[i] = proto.le(u64, s.bufs[r].len);
        parts.items[1 + 2 * i] = &lens.items[i];
        parts.items[2 + 2 * i] = s.bufs[r].ptr[0..s.bufs[r].len];
    }
    try proto.writeFrame(1, parts.items[0 .. 1 + 2 * nret]);
}

// ------------------------------------------------------------ sessions --
//
// The GPU twin of the worker's sessions, on unified memory: OPEN creates
// the program's buffers at their certified maximal extents and its
// pipelines once; a STEP writes its inputs in place, encodes the schedule
// resolved for the step's extents, commits, and returns the outputs asked
// for. Metal command buffers are transient (they cannot be resubmitted),
// so a step is always encoded afresh — counter 7 (recording reused) is 0.

const max_sessions = 8;

const Sess = struct {
    live: bool = false,
    bufs: []be.Buffer = &.{},
    writable: []bool = &.{},
    nbuf: u32 = 0,
    pipes: []be.Pipeline = &.{},
    npipe: u32 = 0,
    regions: [3]?proto.Region = .{ null, null, null },
    bytes: u64 = 0,

    fn deinit(s: *Sess) void {
        for (s.pipes[0..s.npipe]) |p| be.freePipeline(p);
        for (s.bufs[0..s.nbuf]) |b| be.freeBuffer(b);
        for (s.regions) |r| if (r) |x| x.free();
        s.* = .{};
    }
};

var sessions: [max_sessions]Sess = [_]Sess{.{}} ** max_sessions;

fn openSession(c: *proto.Cursor) !void {
    _ = try c.int(u32); // deadline: OPEN is bounded by the BEAM-side timeout
    _ = try c.int(u32); // flags: bit 0 (force staging) has no meaning on unified memory
    var sid: u32 = 0;
    while (sid < max_sessions and sessions[sid].live) sid += 1;
    if (sid == max_sessions) return error.TooManySessions;
    const s = &sessions[sid];
    buildSession(s, c) catch |e| {
        s.deinit();
        return e;
    };
    s.live = true;
    const head = [_]u8{@intFromEnum(Op.open)} ++ proto.le(u32, sid) ++ [_]u8{ 0, 1 } ++ proto.le(u64, s.bytes);
    try proto.writeFrame(1, &.{&head});
}

fn buildSession(s: *Sess, c: *proto.Cursor) !void {
    const mods = try parseModules(c);
    defer mods.free();
    const specs = try parseBuffers(c);
    defer specs.free();
    const nbuf = specs.items.len;

    const wr = try proto.Region.alloc(@max(nbuf, 1) * @sizeOf(bool));
    s.regions[2] = wr;
    s.writable = @as([*]bool, @ptrCast(wr.mem.ptr))[0..nbuf];
    const all = try proto.Region.alloc(@max(nbuf, 1) * @sizeOf(be.Buffer));
    s.regions[0] = all;
    s.bufs = @as([*]be.Buffer, @ptrCast(@alignCast(all.mem.ptr)))[0..nbuf];
    for (specs.items, 0..) |b, i| {
        s.writable[i] = b.writable;
        s.bufs[i] = try makeBuffer(b);
        s.nbuf += 1;
        s.bytes += b.len;
    }
    const pr = try proto.Region.alloc(@max(mods.items.len, 1) * @sizeOf(be.Pipeline));
    s.regions[1] = pr;
    s.pipes = @as([*]be.Pipeline, @ptrCast(@alignCast(pr.mem.ptr)))[0..mods.items.len];
    for (mods.items, 0..) |m, k| {
        s.pipes[k] = try be.compile(m.src, m.nbind, m.npush);
        s.npipe += 1;
    }
}

const Io = struct { buf: u32, off: u64, len: u64, data: []const u8 = &.{} };
const SDisp = struct { module: u32, groups: [3]u32, npush: u32, push: [max_push]u32, nbind: u32, binds: [max_bind]u32 };

/// STEP: `sid deadline writes returns dispatches copies` (fabric.zig's layout).
fn stepSession(c: *proto.Cursor) !void {
    const sid = try c.int(u32);
    if (sid >= max_sessions or !sessions[sid].live) return error.NoSession;
    const s = &sessions[sid];
    const deadline_ms = try c.int(u32);
    const nb = s.nbuf;

    const nw = try c.int(u32);
    const writes = try Table(Io).init(c, nw, 20);
    defer writes.free();
    var host_bytes: u64 = 0;
    for (writes.items) |*w| {
        w.* = .{ .buf = try c.int(u32), .off = try c.int(u64), .len = try c.int(u64) };
        w.data = try c.bytes(w.len);
        if (w.buf >= nb or !s.writable[w.buf] or w.off + w.len > s.bufs[w.buf].len) return error.Truncated;
        host_bytes += w.len;
    }
    const nr = try c.int(u32);
    const rets = try Table(Io).init(c, nr, 20);
    defer rets.free();
    for (rets.items) |*r| {
        r.* = .{ .buf = try c.int(u32), .off = try c.int(u64), .len = try c.int(u64) };
        if (r.buf >= nb or r.off + r.len > s.bufs[r.buf].len) return error.Truncated;
        host_bytes += r.len;
    }
    const nd = try c.int(u32);
    const disps = try Table(SDisp).init(c, nd, 24);
    defer disps.free();
    for (disps.items) |*dp| {
        dp.module = try c.int(u32);
        if (dp.module >= s.npipe) return error.Truncated;
        for (&dp.groups) |*g| g.* = try c.int(u32);
        dp.npush = try c.int(u32);
        if (dp.npush > max_push) return error.Truncated;
        for (dp.push[0..dp.npush]) |*p| p.* = try c.int(u32);
        dp.nbind = try c.int(u32);
        if (dp.nbind > max_bind) return error.Truncated;
        for (dp.binds[0..dp.nbind]) |*b| {
            b.* = try c.int(u32);
            if (b.* >= nb) return error.Truncated;
        }
    }
    const nc = try c.int(u32);
    const copies = try Table(Copy).init(c, nc, 32);
    defer copies.free();
    for (copies.items) |*k| {
        k.* = .{ .src = try c.int(u32), .soff = try c.int(u64), .dst = try c.int(u32), .doff = try c.int(u64), .len = try c.int(u64) };
        if (k.src >= nb or k.dst >= nb or !s.writable[k.dst]) return error.Truncated;
        if (k.soff + k.len > s.bufs[k.src].len or k.doff + k.len > s.bufs[k.dst].len) return error.Truncated;
    }

    const t0 = proto.monotonicNs();
    for (writes.items) |w| @memcpy(s.bufs[w.buf].ptr[w.off .. w.off + w.len], w.data);
    var cmd = try be.begin();
    var binds: [max_bind]be.Buffer = undefined;
    for (disps.items) |dp| {
        for (dp.binds[0..dp.nbind], 0..) |b, k| binds[k] = s.bufs[b];
        be.dispatch(&cmd, s.pipes[dp.module], dp.groups, dp.push[0..dp.npush], binds[0..dp.nbind]);
    }
    for (copies.items) |k| be.copy(&cmd, s.bufs[k.src], k.soff, s.bufs[k.dst], k.doff, k.len);
    try be.commitWait(&cmd, deadline_ms);
    const elapsed = proto.monotonicNs() - t0;

    const parts = try Table([]const u8).init(c, 1 + 2 * nr, 0);
    defer parts.free();
    const lens = try Table([8]u8).init(c, nr, 0);
    defer lens.free();
    var counters_blk: [1 + 18]u8 = undefined;
    counters_blk[0] = 2;
    counters_blk[1] = 7;
    @memcpy(counters_blk[2..10], &proto.le(u64, 0));
    counters_blk[10] = 8;
    @memcpy(counters_blk[11..19], &proto.le(u64, host_bytes));
    const head = [_]u8{@intFromEnum(Op.done)} ++ proto.le(u64, elapsed) ++ proto.le(u64, 0) ++ counters_blk;
    parts.items[0] = &head;
    for (rets.items, 0..) |r, i| {
        lens.items[i] = proto.le(u64, r.len);
        parts.items[1 + 2 * i] = &lens.items[i];
        parts.items[2 + 2 * i] = s.bufs[r.buf].ptr[r.off .. r.off + r.len];
    }
    try proto.writeFrame(1, parts.items[0 .. 1 + 2 * nr]);
}
