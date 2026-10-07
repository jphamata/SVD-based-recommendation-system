//! The Metal backend of `vapor-metal` (macOS): Metal through the
//! Objective-C runtime, with nothing linked but libSystem.
//!
//! libobjc and the Metal framework are opened at run time (`dlopen`), the
//! classes and selectors looked up by name, and every message is an
//! `objc_msgSend` cast to the exact C signature of that method — the
//! documented way to call Objective-C from C. No SDK, no Swift, no
//! Objective-C compiler: the daemon cross-compiles from Linux
//! (`zig build -Dtarget=aarch64-macos`).
//!
//! Floating point: modules are compiled with `MTLMathModeSafe` (macOS 15+:
//! "no transformation that could affect the results") or, before it,
//! `fastMathEnabled = NO`. That is a request, not a proof: `Vapor.Substrate`
//! measures on arrival whether the device contracts, flushes subnormals or
//! departs from the oracle on any kernel, and admits it accordingly.
//!
//! Status, said plainly: this file is compiled for aarch64-macos and
//! x86_64-macos by `make metal`, but it has never been executed — there is
//! no Mac where it was written. Everything around it (protocol, plans,
//! sessions, the MSL itself) is executed on Linux through `mslsim.zig`.

const std = @import("std");
const sys = @import("sys.zig");

pub const Options = struct {};
pub const Info = struct { name: []const u8, vendor: u32 };

pub fn option(_: *Options, _: []const u8, _: []const u8) bool {
    return false;
}

pub fn flag(_: *Options, _: []const u8) bool {
    return false;
}

const Id = ?*anyopaque;
const Sel = ?*anyopaque;
const MTLSize = extern struct { width: u64, height: u64, depth: u64 };

var msg_send: *const anyopaque = undefined;
var get_class: *const fn ([*:0]const u8) callconv(.c) Id = undefined;
var sel_register: *const fn ([*:0]const u8) callconv(.c) Sel = undefined;
var pool_push: *const fn () callconv(.c) ?*anyopaque = undefined;
var pool_pop: *const fn (?*anyopaque) callconv(.c) void = undefined;

var device: Id = null;
var queue: Id = null;
var name_buf: [256]u8 = undefined;
var err_buf: [1024]u8 = undefined;
var err_len: usize = 0;

fn Fn(comptime R: type, comptime A: type) type {
    const f = @typeInfo(A).@"struct".fields;
    return switch (f.len) {
        0 => fn (Id, Sel) callconv(.c) R,
        1 => fn (Id, Sel, f[0].type) callconv(.c) R,
        2 => fn (Id, Sel, f[0].type, f[1].type) callconv(.c) R,
        3 => fn (Id, Sel, f[0].type, f[1].type, f[2].type) callconv(.c) R,
        4 => fn (Id, Sel, f[0].type, f[1].type, f[2].type, f[3].type) callconv(.c) R,
        5 => fn (Id, Sel, f[0].type, f[1].type, f[2].type, f[3].type, f[4].type) callconv(.c) R,
        else => @compileError("objc message with more than five arguments"),
    };
}

/// `[target sel_name args…]`, typed by the caller.
fn send(comptime R: type, target: Id, comptime sel_name: [:0]const u8, args: anytype) R {
    const f: *const Fn(R, @TypeOf(args)) = @ptrCast(@alignCast(msg_send));
    return @call(.auto, f, .{ target, sel_register(sel_name) } ++ args);
}

fn sym(lib: *anyopaque, name: [:0]const u8) !*anyopaque {
    return std.c.dlsym(lib, name) orelse error.MissingSymbol;
}

pub fn lastError() []const u8 {
    const e = err_buf[0..err_len];
    err_len = 0;
    return e;
}

fn noteError(err: Id) void {
    err_len = 0;
    if (err == null) return;
    const desc = send(Id, err, "localizedDescription", .{});
    if (desc == null) return;
    const s = send(?[*:0]const u8, desc, "UTF8String", .{}) orelse return;
    const t = std.mem.sliceTo(s, 0);
    const n = @min(t.len, err_buf.len);
    @memcpy(err_buf[0..n], t[0..n]);
    err_len = n;
}

pub fn init(_: Options) !Info {
    const objc = std.c.dlopen("/usr/lib/libobjc.A.dylib", .{ .NOW = true }) orelse return error.NoObjCRuntime;
    msg_send = try sym(objc, "objc_msgSend");
    get_class = @ptrCast(@alignCast(try sym(objc, "objc_getClass")));
    sel_register = @ptrCast(@alignCast(try sym(objc, "sel_registerName")));
    pool_push = @ptrCast(@alignCast(try sym(objc, "objc_autoreleasePoolPush")));
    pool_pop = @ptrCast(@alignCast(try sym(objc, "objc_autoreleasePoolPop")));

    const metal = std.c.dlopen("/System/Library/Frameworks/Metal.framework/Metal", .{ .NOW = true }) orelse return error.NoMetalFramework;
    const create: *const fn () callconv(.c) Id = @ptrCast(@alignCast(try sym(metal, "MTLCreateSystemDefaultDevice")));
    device = create();
    if (device == null) {
        // headless sessions: the first of all devices
        const all: *const fn () callconv(.c) Id = @ptrCast(@alignCast(try sym(metal, "MTLCopyAllDevices")));
        const arr = all();
        if (arr != null and send(u64, arr, "count", .{}) > 0) device = send(Id, arr, "firstObject", .{});
    }
    if (device == null) return error.NoMetalDevice;
    queue = send(Id, device, "newCommandQueue", .{});
    if (queue == null) return error.NoCommandQueue;

    var name: []const u8 = "Apple GPU";
    const ns = send(Id, device, "name", .{});
    if (ns != null) if (send(?[*:0]const u8, ns, "UTF8String", .{})) |s| {
        const t = std.mem.sliceTo(s, 0);
        const n = @min(t.len, name_buf.len);
        @memcpy(name_buf[0..n], t[0..n]);
        name = name_buf[0..n];
    };
    return .{ .name = name, .vendor = 0x106B };
}

// ------------------------------------------------------------- buffers --

/// Apple Silicon pages are 16 KiB: a no-copy buffer needs that alignment.
pub const wrap_align: u64 = 16384;

pub const Buffer = struct { obj: Id, ptr: [*]u8, len: u64, map: ?[]align(sys.page) u8 = null };

pub fn newBuffer(len: u64) !Buffer {
    const obj = send(Id, device, "newBufferWithLength:options:", .{ @as(u64, @max(len, 4)), @as(u64, 0) }); // shared storage
    if (obj == null) return error.OutOfDeviceMemory;
    const p = send(?[*]u8, obj, "contents", .{}) orelse return error.OutOfDeviceMemory;
    @memset(p[0..@max(len, 4)], 0);
    return .{ .obj = obj, .ptr = p, .len = len };
}

/// A mapped weight file as a device buffer, without a copy.
pub fn wrap(m: []align(sys.page) u8, len: u64) !Buffer {
    if (@intFromPtr(m.ptr) % wrap_align != 0 or m.len % wrap_align != 0) return error.Unaligned;
    const obj = send(Id, device, "newBufferWithBytesNoCopy:length:options:deallocator:", .{ @as(*anyopaque, m.ptr), @as(u64, m.len), @as(u64, 0), @as(Id, null) });
    if (obj == null) return error.NoHostImport;
    return .{ .obj = obj, .ptr = m.ptr, .len = len, .map = m };
}

pub fn freeBuffer(b: Buffer) void {
    send(void, b.obj, "release", .{});
    if (b.map) |m| sys.unmap(m);
}

// ------------------------------------------------------------ pipelines --

pub const Pipeline = struct { pso: Id, nbind: u32, npush: u32 };

fn nsString(bytes: []const u8) Id {
    const alloc = send(Id, get_class("NSString"), "alloc", .{});
    return send(Id, alloc, "initWithBytes:length:encoding:", .{ @as(*const anyopaque, bytes.ptr), @as(u64, bytes.len), @as(u64, 4) }); // UTF-8
}

/// The kernel's name: the identifier after `kernel void `.
fn kernelName(src: []const u8) ![]const u8 {
    const key = "kernel void ";
    const at = (std.mem.indexOf(u8, src, key) orelse return error.NoKernel) + key.len;
    var end = at;
    while (end < src.len and (std.ascii.isAlphanumeric(src[end]) or src[end] == '_')) end += 1;
    if (end == at) return error.NoKernel;
    return src[at..end];
}

pub fn compile(src: []const u8, nbind: u32, npush: u32) !Pipeline {
    const text = nsString(src);
    defer send(void, text, "release", .{});
    const opts = send(Id, send(Id, get_class("MTLCompileOptions"), "alloc", .{}), "init", .{});
    defer send(void, opts, "release", .{});
    if (send(u8, opts, "respondsToSelector:", .{sel_register("setMathMode:")}) != 0)
        send(void, opts, "setMathMode:", .{@as(i64, 0)}) // MTLMathModeSafe
    else
        send(void, opts, "setFastMathEnabled:", .{@as(u8, 0)});

    var err: Id = null;
    const lib = send(Id, device, "newLibraryWithSource:options:error:", .{ text, opts, &err });
    if (lib == null) {
        noteError(err);
        return error.CompileFailed;
    }
    defer send(void, lib, "release", .{});
    const fname = nsString(try kernelName(src));
    defer send(void, fname, "release", .{});
    const func = send(Id, lib, "newFunctionWithName:", .{fname});
    if (func == null) return error.NoKernel;
    defer send(void, func, "release", .{});
    const pso = send(Id, device, "newComputePipelineStateWithFunction:error:", .{ func, &err });
    if (pso == null) {
        noteError(err);
        return error.PipelineFailed;
    }
    if (send(u64, pso, "maxTotalThreadsPerThreadgroup", .{}) < 64) return error.ThreadgroupTooSmall;
    return .{ .pso = pso, .nbind = nbind, .npush = npush };
}

pub fn freePipeline(p: Pipeline) void {
    send(void, p.pso, "release", .{});
}

// ------------------------------------------------------------- commands --

pub fn poolPush() usize {
    return @intFromPtr(pool_push());
}

pub fn poolPop(p: usize) void {
    pool_pop(@ptrFromInt(p));
}

const Kind = enum { none, compute, blit };
pub const Cmd = struct { cb: Id, enc: Id = null, kind: Kind = .none };

pub fn begin() !Cmd {
    const cb = send(Id, queue, "commandBuffer", .{});
    if (cb == null) return error.NoCommandBuffer;
    return .{ .cb = cb };
}

fn encoder(cmd: *Cmd, kind: Kind) Id {
    if (cmd.kind != kind) {
        if (cmd.enc != null) send(void, cmd.enc, "endEncoding", .{});
        cmd.enc = if (kind == .compute) send(Id, cmd.cb, "computeCommandEncoder", .{}) else send(Id, cmd.cb, "blitCommandEncoder", .{});
        cmd.kind = kind;
    }
    return cmd.enc;
}

pub fn copy(cmd: *Cmd, src: Buffer, soff: u64, dst: Buffer, doff: u64, len: u64) void {
    if (len == 0) return;
    const enc = encoder(cmd, .blit);
    send(void, enc, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:", .{ src.obj, soff, dst.obj, doff, len });
}

pub fn dispatch(cmd: *Cmd, pipe: Pipeline, groups: [3]u32, push: []const u32, binds: []const Buffer) void {
    const enc = encoder(cmd, .compute);
    send(void, enc, "setComputePipelineState:", .{pipe.pso});
    for (binds, 0..) |b, k| send(void, enc, "setBuffer:offset:atIndex:", .{ b.obj, @as(u64, 0), @as(u64, k) });
    if (push.len > 0) send(void, enc, "setBytes:length:atIndex:", .{ @as(*const anyopaque, push.ptr), @as(u64, push.len * 4), @as(u64, binds.len) });
    send(void, enc, "dispatchThreadgroups:threadsPerThreadgroup:", .{
        MTLSize{ .width = groups[0], .height = groups[1], .depth = groups[2] },
        MTLSize{ .width = 64, .height = 1, .depth = 1 },
    });
}

/// Commit and wait for completion, at most `deadline_ms` (0: no limit). A
/// command buffer that errors or outlives its deadline is a lost device.
pub fn commitWait(cmd: *Cmd, deadline_ms: u32) !void {
    if (cmd.enc != null) send(void, cmd.enc, "endEncoding", .{});
    cmd.enc = null;
    cmd.kind = .none;
    send(void, cmd.cb, "commit", .{});
    const t0 = sys.monotonicNs();
    while (true) {
        const status = send(u64, cmd.cb, "status", .{}); // 4 completed, 5 error
        if (status == 4) return;
        if (status == 5) {
            noteError(send(Id, cmd.cb, "error", .{}));
            return error.DeviceLost;
        }
        if (deadline_ms != 0 and sys.monotonicNs() - t0 > @as(u64, deadline_ms) * 1_000_000) return error.DeviceLost;
        const ts = std.c.timespec{ .sec = 0, .nsec = 20_000 };
        _ = std.c.nanosleep(&ts, null);
    }
}
