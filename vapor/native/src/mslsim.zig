//! The Linux stand-in for Metal behind `vapor-metal-sim` (tests only).
//!
//! It runs the very MSL text the BEAM sends — the source a Mac would hand to
//! `newLibraryWithSource:` — on the CPU. It compiles nothing itself: a
//! module is found by the SHA-256 of its source in a cache directory
//! (`--cache DIR` or `VAPOR_MSL_CACHE`) as `DIR/<sha256>.so`, a shared object
//! clang built from that exact text with a ten-line shim for the Metal types
//! (test/support/msl). How that directory was built is the "device": with
//! one rounding per operation it is a conforming device; built with FMA
//! contraction or with flush-to-zero it is a non-conforming one, which is
//! how `Vapor.Substrate`'s admission is tested against devices that break
//! the rules. `DIR/device` (one line) names it.
//!
//! Every dispatch runs the whole grid on the calling thread, in order, so a
//! command "buffer" is executed as it is recorded — the serial order Metal
//! guarantees inside one encoder and across encoders.

const std = @import("std");
const sys = @import("sys.zig");

pub const Options = struct { cache: []const u8 = "" };
pub const Info = struct { name: []const u8, vendor: u32 };

pub fn option(o: *Options, name: []const u8, value: []const u8) bool {
    if (std.mem.eql(u8, name, "--cache")) {
        o.cache = value;
        return true;
    }
    return false;
}

pub fn flag(_: *Options, _: []const u8) bool {
    return false;
}

var cache_dir: [1024]u8 = undefined;
var cache_len: usize = 0;
var name_buf: [256]u8 = undefined;
var err_buf: [1200]u8 = undefined;
var err_len: usize = 0;

pub fn lastError() []const u8 {
    const e = err_buf[0..err_len];
    err_len = 0;
    return e;
}

fn setError(comptime fmt: []const u8, args: anytype) void {
    const s = std.fmt.bufPrint(&err_buf, fmt, args) catch err_buf[0..0];
    err_len = s.len;
}

pub fn init(o: Options) !Info {
    var dir: []const u8 = o.cache;
    if (dir.len == 0) {
        const e = std.c.getenv("VAPOR_MSL_CACHE") orelse return error.NoModuleCache;
        dir = std.mem.sliceTo(e, 0);
    }
    if (dir.len == 0 or dir.len >= cache_dir.len - 80) return error.NoModuleCache;
    @memcpy(cache_dir[0..dir.len], dir);
    cache_len = dir.len;

    // the device's name: DIR/device, or a generic one
    var name: []const u8 = "Metal stand-in: MSL on the CPU (clang shim)";
    var p: [1100]u8 = undefined;
    const path = try std.fmt.bufPrint(&p, "{s}/device", .{dir});
    if (sys.mapFile(path, 1, sys.page)) |m| {
        defer sys.unmap(m);
        const line = std.mem.sliceTo(m[0..@min(m.len, name_buf.len)], 0);
        const one = std.mem.trimEnd(u8, if (std.mem.indexOfScalar(u8, line, '\n')) |i| line[0..i] else line, " \r\t");
        if (one.len > 0) {
            @memcpy(name_buf[0..one.len], one);
            name = name_buf[0..one.len];
        }
    } else |_| {}
    return .{ .name = name, .vendor = 0 };
}

// ------------------------------------------------------------- buffers --

pub const wrap_align: u64 = sys.page;

pub const Buffer = struct { ptr: [*]u8, len: u64, mem: []align(sys.page) u8 };

pub fn newBuffer(len: u64) !Buffer {
    const m = try sys.mapAnon(@max(len, 4));
    return .{ .ptr = m.ptr, .len = len, .mem = m };
}

/// A mapped file used in place (unified memory: no copy).
pub fn wrap(m: []align(sys.page) u8, len: u64) !Buffer {
    return .{ .ptr = m.ptr, .len = len, .mem = m };
}

pub fn freeBuffer(b: Buffer) void {
    sys.unmap(b.mem);
}

// ------------------------------------------------------------ pipelines --

const Entry = *const fn ([*]?*anyopaque, [*]const u32, u32, u32, u32) callconv(.c) void;
pub const Pipeline = struct { entry: Entry, nbind: u32, npush: u32 };

pub fn compile(src: []const u8, nbind: u32, npush: u32) !Pipeline {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(src, &digest, .{});
    var p: [1200]u8 = undefined;
    const hex = std.fmt.bytesToHex(digest, .lower);
    const path = try std.fmt.bufPrintZ(&p, "{s}/{s}.so", .{ cache_dir[0..cache_len], hex });
    const h = std.c.dlopen(path, .{ .NOW = true }) orelse {
        setError("module not built for this device: {s}", .{path});
        return error.ModuleNotCompiled;
    };
    const f = std.c.dlsym(h, "vapor_entry") orelse return error.ModuleNotCompiled;
    return .{ .entry = @ptrCast(@alignCast(f)), .nbind = nbind, .npush = npush };
}

pub fn freePipeline(_: Pipeline) void {}

// ------------------------------------------------------------- commands --

pub const Cmd = struct {};

pub fn poolPush() usize {
    return 0;
}

pub fn poolPop(_: usize) void {}

pub fn begin() !Cmd {
    return .{};
}

pub fn copy(_: *Cmd, src: Buffer, soff: u64, dst: Buffer, doff: u64, len: u64) void {
    @memmove(dst.ptr[doff .. doff + len], src.ptr[soff .. soff + len]);
}

pub fn dispatch(_: *Cmd, pipe: Pipeline, groups: [3]u32, push: []const u32, binds: []const Buffer) void {
    var b: [16]?*anyopaque = [_]?*anyopaque{null} ** 16;
    for (binds, 0..) |x, k| b[k] = x.ptr;
    const none = [_]u32{0};
    pipe.entry(&b, if (push.len > 0) push.ptr else &none, groups[0], groups[1], groups[2]);
}

pub fn commitWait(_: *Cmd, _: u32) !void {}
