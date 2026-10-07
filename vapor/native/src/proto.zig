//! Framing shared by the worker and the fabric daemon.
//!
//! The BEAM talks to both over stdio with Erlang's `{packet, 4}` framing:
//! a 4-byte big-endian length, then the payload. Inside a payload every
//! integer is little-endian. A process that dies mid-frame cannot
//! desynchronise the stream: the BEAM sees the port close, never a partial
//! message it would misparse (Axiom 3).

const std = @import("std");
const sys = @import("sys.zig");

pub const Error = error{ Eof, Io, Truncated, TooLarge, OutOfMemory };

/// Frames above this size are refused (the length prefix is untrusted).
pub const max_frame: usize = 1 << 30;

fn readAll(fd: i32, buf: []u8) Error!void {
    var got: usize = 0;
    while (got < buf.len) {
        const n = sys.read(fd, buf[got..]) catch return error.Io;
        if (n == 0) return if (got == 0) error.Eof else error.Truncated;
        got += n;
    }
}

fn writeAll(fd: i32, buf: []const u8) Error!void {
    var put: usize = 0;
    while (put < buf.len) put += sys.write(fd, buf[put..]) catch return error.Io;
}

/// Page-backed byte region (mmap/munmap; no allocator state; on Linux no libc).
pub const Region = struct {
    mem: []align(sys.page) u8,

    pub fn alloc(len: usize) Error!Region {
        return .{ .mem = sys.mapAnon(len) catch return error.OutOfMemory };
    }

    pub fn free(self: Region) void {
        sys.unmap(self.mem);
    }
};

/// Read one frame into a fresh region; the caller frees it.
pub fn readFrame(fd: i32) Error!struct { region: Region, len: usize } {
    var hdr: [4]u8 = undefined;
    try readAll(fd, &hdr);
    const len = std.mem.readInt(u32, &hdr, .big);
    if (len > max_frame) return error.TooLarge;
    const r = try Region.alloc(len);
    errdefer r.free();
    try readAll(fd, r.mem[0..len]);
    return .{ .region = r, .len = len };
}

/// Write one frame assembled from parts (gathered, then a single length).
pub fn writeFrame(fd: i32, parts: []const []const u8) Error!void {
    var total: usize = 0;
    for (parts) |p| total += p.len;
    var hdr: [4]u8 = undefined;
    std.mem.writeInt(u32, &hdr, @intCast(total), .big);
    try writeAll(fd, &hdr);
    for (parts) |p| try writeAll(fd, p);
}

/// Bounds-checked little-endian cursor over an untrusted payload.
pub const Cursor = struct {
    buf: []const u8,
    pos: usize = 0,

    pub fn int(self: *Cursor, comptime T: type) Error!T {
        const n = @sizeOf(T);
        if (self.buf.len - self.pos < n) return error.Truncated;
        const v = std.mem.readInt(T, self.buf[self.pos..][0..n], .little);
        self.pos += n;
        return v;
    }

    pub fn bytes(self: *Cursor, n: usize) Error![]const u8 {
        if (self.buf.len - self.pos < n) return error.Truncated;
        const s = self.buf[self.pos .. self.pos + n];
        self.pos += n;
        return s;
    }
};

pub fn le(comptime T: type, v: T) [@sizeOf(T)]u8 {
    var b: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &b, v, .little);
    return b;
}

pub fn monotonicNs() u64 {
    return sys.monotonicNs();
}

/// A frame-sized table of `n` items in its own region; `min_bytes` is the
/// smallest encoding of one item, so a hostile count cannot allocate more
/// than the frame could describe.
pub fn Table(comptime T: type) type {
    return struct {
        region: Region,
        items: []T,

        pub fn init(c: *const Cursor, n: u32, min_bytes: usize) !@This() {
            if (@as(usize, n) * min_bytes > c.buf.len - c.pos) return error.Truncated;
            const r = try Region.alloc(@as(usize, n) * @sizeOf(T));
            const p: [*]T = @ptrCast(@alignCast(r.mem.ptr));
            return .{ .region = r, .items = p[0..n] };
        }

        pub fn free(self: @This()) void {
            self.region.free();
        }
    };
}
