//! vapor-fabric — Substrate II: the Vulkan compute daemon.
//!
//! A separate, unprivileged OS process owns the GPU driver. The BEAM talks
//! to it over stdio with `{packet, 4}` framing; if the driver faults
//! (SIGSEGV/SIGBUS inside libvulkan or the ICD) or the device is lost, only
//! this process dies and the BEAM reroutes the unit to the native substrate
//! (Axiom 3). The full headless compute path is implemented:
//!
//!   instance → physical device (compute queue) → logical device
//!   → buffers (host-visible, or zero-copy import of mmapped weight files
//!     via VK_EXT_external_memory_host) → shader modules from the SPIR-V
//!   words the control plane emitted → descriptor-set layouts, pipeline
//!   layouts with push constants, compute pipelines → descriptor pool/sets
//!   → one command buffer per RUN (dispatches, barriers, iteration windows,
//!   state-feedback copies, emit staging) → queue submit → fence wait with
//!   a deadline → read-back.
//!
//! On `VK_EXT_external_memory_host` vs `VK_KHR_external_memory_fd`: an
//! OPAQUE_FD handle can only import memory previously *exported by the same
//! driver*; an arbitrary memfd is not importable that way. Host-pointer
//! import of a shared mapping is the mechanism that actually delivers
//! zero-copy from files written by the BEAM.

const std = @import("std");

/// Faults must surface as the real signal (SIGSEGV/SIGILL/SIGBUS) so the
/// BEAM can classify them; Zig's handler would turn them into SIGABRT.
pub const std_options: std.Options = .{ .enable_segfault_handler = false };
const linux = std.os.linux;
const proto = @import("proto.zig");
const vk = @import("vk.zig");

const Op = enum(u8) { hello = 1, run = 2, emit = 3, done = 4, err = 5, fault = 9, open = 10, step = 11, close = 12 };

// per dispatch: bindings and push constants (Vulkan's guaranteed minimum
// of 128 push-constant bytes); every table of a unit is sized by its frame
const max_bind = 16;
const max_push = 32;

var fns: vk.Fns = undefined;
var instance: ?vk.Instance = null;
var pdev: vk.PhysicalDevice = undefined;
var device: ?vk.Device = null;
var queue: ?vk.Queue = null;
var qfamily: u32 = 0;
var mem_props: vk.PhysicalDeviceMemoryProperties = undefined;
var ext_host = false;
var host_align: u64 = 4096;
var coop_i8 = false;
var denorm32 = false;
var props2: vk.PhysicalDeviceProperties2 = .{};
var fault_injection = false;
var init_error: []const u8 = "";

pub fn main(init: std.process.Init) void {
    const args = init.minimal.args.toSlice(init.arena.allocator()) catch &.{};
    for (args) |a| {
        if (std.mem.eql(u8, a, "--fault-injection")) fault_injection = true;
    }
    const ready = if (setup()) true else |e| blk: {
        init_error = @errorName(e);
        break :blk false;
    };

    while (true) {
        const frame = proto.readFrame(0) catch |e| switch (e) {
            error.Eof => linux.exit_group(0),
            else => linux.exit_group(3),
        };
        defer frame.region.free();
        dispatch(frame.region.mem[0..frame.len], ready) catch |e| {
            if (e == error.DeviceLost) {
                sendErr(2, vk.ERROR_DEVICE_LOST, "device lost");
                linux.exit_group(5); // the device is unusable; let the BEAM respawn us
            }
            sendErr(1, 0, @errorName(e));
        };
    }
}

fn sendErr(code: u32, vkres: i32, msg: []const u8) void {
    const head = [_]u8{@intFromEnum(Op.err)} ++ proto.le(u32, code) ++ proto.le(u64, @as(u64, @bitCast(@as(i64, vkres)))) ++ proto.le(u32, 0);
    proto.writeFrame(1, &.{ &head, msg }) catch linux.exit_group(4);
}

fn check(r: vk.Result) !void {
    if (r == vk.ERROR_DEVICE_LOST) return error.DeviceLost;
    if (r == vk.ERROR_OUT_OF_DEVICE_MEMORY or r == vk.ERROR_OUT_OF_HOST_MEMORY) return error.OutOfDeviceMemory;
    if (r < 0) return error.VulkanError;
}

// ------------------------------------------------------------- device setup --

fn setup() !void {
    const lib = std.c.dlopen("libvulkan.so.1", .{ .NOW = true }) orelse return error.NoVulkanLoader;
    const gipa = std.c.dlsym(lib, "vkGetInstanceProcAddr") orelse return error.NoVulkanLoader;
    fns = .{ .getInstanceProcAddr = @ptrCast(@alignCast(gipa)) };
    fns.createInstance = @ptrCast(fns.getInstanceProcAddr(null, "vkCreateInstance") orelse return error.NoVulkanLoader);

    const app = vk.ApplicationInfo{ .pApplicationName = "vapor-fabric", .apiVersion = (1 << 22) | (3 << 12) };
    try check(fns.createInstance(&.{ .pApplicationInfo = &app }, null, &instance));
    try fns.loadInstance(instance.?);

    var n: u32 = 0;
    try check(fns.enumeratePhysicalDevices(instance.?, &n, null));
    if (n == 0) return error.NoPhysicalDevice;
    var devs: [16]vk.PhysicalDevice = undefined;
    n = @min(n, 16);
    try check(fns.enumeratePhysicalDevices(instance.?, &n, &devs));

    // first device with a compute queue
    var found = false;
    outer: for (devs[0..n]) |d| {
        var qn: u32 = 0;
        fns.getPhysicalDeviceQueueFamilyProperties(d, &qn, null);
        var qf: [32]vk.QueueFamilyProperties = undefined;
        qn = @min(qn, 32);
        fns.getPhysicalDeviceQueueFamilyProperties(d, &qn, &qf);
        for (qf[0..qn], 0..) |q, i| if (q.queueFlags & vk.QUEUE_COMPUTE != 0) {
            pdev = d;
            qfamily = @intCast(i);
            found = true;
            break :outer;
        };
    }
    if (!found) return error.NoComputeQueue;

    // properties: float controls, external host memory alignment
    var fc = vk.FloatControlsProperties{};
    var eh = vk.ExternalMemoryHostProperties{ .pNext = &fc };
    props2 = .{ .pNext = &eh };
    fns.getPhysicalDeviceProperties2(pdev, &props2);
    props2.pNext = null;
    denorm32 = fc.flags[4] != 0;
    fns.getPhysicalDeviceMemoryProperties(pdev, &mem_props);

    // extensions
    var en: u32 = 0;
    try check(fns.enumerateDeviceExtensionProperties(pdev, null, &en, null));
    var exts: [512]vk.ExtensionProperties = undefined;
    en = @min(en, 512);
    try check(fns.enumerateDeviceExtensionProperties(pdev, null, &en, &exts));
    var has_coop_ext = false;
    for (exts[0..en]) |e| {
        const name = std.mem.sliceTo(&e.name, 0);
        if (std.mem.eql(u8, name, "VK_EXT_external_memory_host")) ext_host = true;
        if (std.mem.eql(u8, name, "VK_KHR_cooperative_matrix")) has_coop_ext = true;
    }
    if (ext_host) host_align = @max(eh.minImportedHostPointerAlignment, 1);

    // cooperative matrix: need exactly s8·s8 → s32, 16×16×16, subgroup scope
    if (has_coop_ext) if (fns.getCoopMatrixProperties) |f| {
        var cn: u32 = 0;
        if (f(pdev, &cn, null) == vk.SUCCESS and cn > 0) {
            var cp: [64]vk.CooperativeMatrixProperties = [_]vk.CooperativeMatrixProperties{.{}} ** 64;
            cn = @min(cn, 64);
            if (f(pdev, &cn, &cp) == vk.SUCCESS) for (cp[0..cn]) |p| {
                if (p.MSize == 16 and p.NSize == 16 and p.KSize == 16 and p.AType == 3 and p.BType == 3 and
                    p.CType == 5 and p.ResultType == 5 and p.scope == 3) coop_i8 = true;
            };
        }
    };

    // logical device
    const prio = [_]f32{1.0};
    const qci = [_]vk.DeviceQueueCreateInfo{.{ .queueFamilyIndex = qfamily, .pQueuePriorities = &prio }};
    var names: [2][*:0]const u8 = undefined;
    var nn: u32 = 0;
    if (ext_host) {
        names[nn] = "VK_EXT_external_memory_host";
        nn += 1;
    }
    const f16i8 = vk.Float16Int8Features{};
    const s8 = vk.Storage8BitFeatures{ .pNext = &f16i8 };
    const coopf = vk.CooperativeMatrixFeatures{ .pNext = &s8 };
    if (coop_i8) {
        names[nn] = "VK_KHR_cooperative_matrix";
        nn += 1;
    }
    try check(fns.createDevice(pdev, &.{
        .pNext = if (coop_i8) &coopf else null,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = nn,
        .ppEnabledExtensionNames = &names,
    }, null, &device));
    try fns.loadDevice(device.?);
    fns.getDeviceQueue(device.?, qfamily, 0, &queue);
}

fn dispatch(payload: []const u8, ready: bool) !void {
    var c = proto.Cursor{ .buf = payload };
    switch (try c.int(u8)) {
        @intFromEnum(Op.hello) => {
            _ = try c.int(u32);
            const p = &props2.properties;
            const name = if (ready) std.mem.sliceTo(p[20..276], 0) else init_error;
            const head = [_]u8{ @intFromEnum(Op.hello), @intFromBool(ready) } ++
                proto.le(u32, std.mem.readInt(u32, p[0..4], .little)) ++
                proto.le(u32, std.mem.readInt(u32, p[8..12], .little)) ++
                proto.le(u32, std.mem.readInt(u32, p[12..16], .little)) ++
                [_]u8{ @intFromBool(denorm32), @intFromBool(ext_host), @intFromBool(coop_i8) } ++
                proto.le(u16, @intCast(name.len));
            try proto.writeFrame(1, &.{ &head, name });
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
            // Simulated driver crash: a wild store, exactly what a buggy ICD does.
            const p: *volatile u32 = @ptrFromInt(0x10);
            p.* = 0xDEAD;
        },
        else => return error.Truncated,
    }
}

// ---------------------------------------------------------------- RUN --

const Buf = struct {
    buffer: vk.Handle = 0,
    memory: vk.Handle = 0,
    map: [*]u8 = undefined,
    mapped: bool = false,
    len: u64 = 0,
    file_map: ?[]align(4096) u8 = null,
};

/// A buffer as the frame describes it (created after the whole unit parsed).
const BufSpec = struct { kind: u8, len: u64, data: []const u8 = &.{}, path: []const u8 = &.{} };

const Bind = union(enum) {
    buf: u32,
    window: struct { idx: u32, base: u64, stride: u64, len: u64, win: u32 },
};

const Disp = struct {
    module: u32,
    groups: [3]u32,
    npush: u32,
    push: [max_push]u32,
    nbind: u32,
    binds: [max_bind]Bind,
};

const Module = struct { words: []const u8, nbind: u32, npush: u32 };
const Pipe = struct { shader: vk.Handle = 0, dsl: vk.Handle = 0, layout: vk.Handle = 0, pipe: vk.Handle = 0 };
const Emit = struct { buf: u32, base: u64, stride: u64, len: u64, stage: u32 };
const Copy = struct { src: u32, soff: u64, dst: u32, doff: u64, len: u64 };

const Table = proto.Table;

/// Everything a unit created; `deinit` releases exactly what exists.
const State = struct {
    bufs: []Buf = &.{}, // the frame's buffers, then iteration windows, then emit stages
    nbuf: u32 = 0, // created so far
    pipes: []Pipe = &.{},
    regions: [2]?proto.Region = .{ null, null }, // backing of bufs and pipes, freed last
    pool: vk.Handle = 0,
    cmd_pool: vk.Handle = 0,
    fence: vk.Handle = 0,

    fn deinit(s: *State) void {
        const d = device.?;
        if (s.fence != 0) fns.destroyFence(d, s.fence, null);
        if (s.cmd_pool != 0) fns.destroyCommandPool(d, s.cmd_pool, null);
        if (s.pool != 0) fns.destroyDescriptorPool(d, s.pool, null);
        for (s.pipes) |p| {
            if (p.pipe != 0) fns.destroyPipeline(d, p.pipe, null);
            if (p.layout != 0) fns.destroyPipelineLayout(d, p.layout, null);
            if (p.dsl != 0) fns.destroyDescriptorSetLayout(d, p.dsl, null);
            if (p.shader != 0) fns.destroyShaderModule(d, p.shader, null);
        }
        for (s.bufs[0..s.nbuf]) |b| {
            if (b.buffer != 0) fns.destroyBuffer(d, b.buffer, null);
            if (b.memory != 0) fns.freeMemory(d, b.memory, null);
            if (b.file_map) |m| _ = linux.munmap(m.ptr, m.len);
        }
        for (s.regions) |r| if (r) |x| x.free();
    }
};

fn memoryType(bits: u32, want: u32) !u32 {
    var i: u32 = 0;
    while (i < mem_props.memoryTypeCount) : (i += 1) {
        if (bits & (@as(u32, 1) << @intCast(i)) != 0 and mem_props.memoryTypes[i].propertyFlags & want == want) return i;
    }
    return error.NoMemoryType;
}

/// Host-visible, coherent storage buffer (the portable path).
fn hostBuffer(len: u64) !Buf {
    const d = device.?;
    var b = Buf{ .len = len };
    const size = @max(std.mem.alignForward(u64, len, 4), 4);
    try check(fns.createBuffer(d, &.{ .size = size, .usage = vk.BUF_STORAGE | vk.BUF_TRANSFER_SRC | vk.BUF_TRANSFER_DST }, null, &b.buffer));
    var req: vk.MemoryRequirements = undefined;
    fns.getBufferMemoryRequirements(d, b.buffer, &req);
    const mt = try memoryType(req.memoryTypeBits, vk.MEM_HOST_VISIBLE | vk.MEM_HOST_COHERENT);
    try check(fns.allocateMemory(d, &.{ .allocationSize = req.size, .memoryTypeIndex = mt }, null, &b.memory));
    try check(fns.bindBufferMemory(d, b.buffer, b.memory, 0));
    var p: ?*anyopaque = null;
    try check(fns.mapMemory(d, b.memory, 0, vk.WHOLE_SIZE, 0, &p));
    b.map = @ptrCast(p.?);
    b.mapped = true;
    return b;
}

/// Zero-copy: import an mmapped weight file as device memory.
fn importFile(path: []const u8, len: u64) !Buf {
    const m = try mapFile(path, len);
    errdefer _ = linux.munmap(m.ptr, m.len);
    const d = device.?;
    var b = Buf{ .len = len, .file_map = m, .map = m.ptr, .mapped = true };
    const get = fns.getMemoryHostPointerProperties orelse return error.NoHostImport;
    var hp = vk.MemoryHostPointerProperties{};
    try check(get(d, vk.HANDLE_HOST_ALLOCATION, m.ptr, &hp));
    const ext = vk.ExternalMemoryBufferCreateInfo{};
    try check(fns.createBuffer(d, &.{ .pNext = &ext, .size = m.len, .usage = vk.BUF_STORAGE | vk.BUF_TRANSFER_SRC }, null, &b.buffer));
    var req: vk.MemoryRequirements = undefined;
    fns.getBufferMemoryRequirements(d, b.buffer, &req);
    const mt = try memoryType(req.memoryTypeBits & hp.memoryTypeBits, 0);
    const imp = vk.ImportMemoryHostPointerInfo{ .pHostPointer = m.ptr };
    check(fns.allocateMemory(d, &.{ .pNext = &imp, .allocationSize = m.len, .memoryTypeIndex = mt }, null, &b.memory)) catch |e| {
        fns.destroyBuffer(d, b.buffer, null);
        return e;
    };
    try check(fns.bindBufferMemory(d, b.buffer, b.memory, 0));
    return b;
}

fn mapFile(path: []const u8, len: u64) ![]align(4096) u8 {
    var z: [4096]u8 = undefined;
    if (path.len >= z.len) return error.TooLarge;
    @memcpy(z[0..path.len], path);
    z[path.len] = 0;
    const fd_rc = linux.openat(linux.AT.FDCWD, @ptrCast(&z), .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    if (linux.errno(fd_rc) != .SUCCESS) return error.Io;
    const fd: i32 = @intCast(fd_rc);
    defer _ = linux.close(fd);
    const size = linux.lseek(fd, 0, linux.SEEK.END);
    if (linux.errno(size) != .SUCCESS or len > size) return error.Truncated;
    const map_len = std.mem.alignForward(usize, @max(len, 1), @max(host_align, 4096));
    const rc = linux.mmap(null, map_len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE }, fd, 0);
    if (linux.errno(rc) != .SUCCESS) return error.OutOfMemory;
    const p: [*]align(4096) u8 = @ptrFromInt(rc);
    return p[0..map_len];
}

fn barrier(cb: vk.CommandBuffer) void {
    const all = vk.ACCESS_SHADER_READ | vk.ACCESS_SHADER_WRITE | vk.ACCESS_TRANSFER_READ | vk.ACCESS_TRANSFER_WRITE;
    const mb = [_]vk.MemoryBarrier{.{ .srcAccessMask = vk.ACCESS_SHADER_WRITE | vk.ACCESS_TRANSFER_WRITE, .dstAccessMask = all }};
    const stages = vk.PIPE_STAGE_COMPUTE | vk.PIPE_STAGE_TRANSFER;
    fns.cmdPipelineBarrier(cb, stages, stages, 0, 1, &mb, 0, null, 0, null);
}

fn copy(cb: vk.CommandBuffer, src: vk.Handle, soff: u64, dst: vk.Handle, doff: u64, len: u64) void {
    const r = [_]vk.BufferCopy{.{ .srcOffset = soff, .dstOffset = doff, .size = len }};
    fns.cmdCopyBuffer(cb, src, dst, 1, &r);
}

fn run(c: *proto.Cursor) !void {
    const d = device.?;
    var s = State{};
    defer s.deinit();

    const deadline_ms = try c.int(u32);

    // ---- parse the whole unit first; tables are sized by the frame ----
    const nmod = try c.int(u32);
    const mods = try Table(Module).init(c, nmod, 12);
    defer mods.free();
    for (mods.items) |*m| {
        const len = try c.int(u32);
        m.words = try c.bytes(len);
        m.nbind = try c.int(u32);
        m.npush = try c.int(u32);
        if (m.nbind > max_bind or m.npush > max_push or len % 4 != 0) return error.Truncated;
    }

    const nbuf = try c.int(u32);
    const specs = try Table(BufSpec).init(c, nbuf, 10);
    defer specs.free();
    for (specs.items) |*b| {
        const kind = try c.int(u8);
        _ = try c.int(u8); // writability is enforced by the SPIR-V NonWritable decoration
        const len = try c.int(u64);
        b.* = .{ .kind = kind, .len = len };
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

    const ndisp = try c.int(u32);
    const disps = try Table(Disp).init(c, ndisp, 24);
    defer disps.free();
    var total_binds: u32 = 0;
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
        total_binds += dp.nbind;
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
    const all = try proto.Region.alloc(@as(usize, nbuf + nwin + nemit) * @sizeOf(Buf));
    s.regions[0] = all;
    const bp: [*]Buf = @ptrCast(@alignCast(all.mem.ptr));
    s.bufs = bp[0 .. nbuf + nwin + nemit];
    for (specs.items, 0..) |b, i| {
        s.bufs[i] = switch (b.kind) {
            0 => try hostBuffer(b.len),
            1 => try hostBuffer(b.len),
            else => if (ext_host) importFile(b.path, b.len) catch try copyFile(b.path, b.len) else try copyFile(b.path, b.len),
        };
        s.nbuf += 1;
        switch (b.kind) {
            0 => @memset(s.bufs[i].map[0..b.len], 0),
            1 => @memcpy(s.bufs[i].map[0..b.len], b.data),
            else => {},
        }
    }
    for (disps.items) |dp| for (dp.binds[0..dp.nbind]) |bd| switch (bd) {
        .window => |w| {
            s.bufs[w.win] = try hostBuffer(w.len);
            s.nbuf += 1;
        },
        .buf => {},
    };
    for (emits.items) |e| {
        s.bufs[e.stage] = try hostBuffer(e.len * @max(iters, 1));
        s.nbuf += 1;
    }

    // ---- pipelines: shader module → set layout → pipeline layout → compute pipeline ----
    const pipes = try proto.Region.alloc(@as(usize, nmod) * @sizeOf(Pipe));
    s.regions[1] = pipes;
    const pp: [*]Pipe = @ptrCast(@alignCast(pipes.mem.ptr));
    s.pipes = pp[0..nmod];
    for (s.pipes) |*p| p.* = .{};
    for (mods.items, s.pipes) |m, *pl| {
        // frame payloads carry no alignment guarantee; SPIR-V words must be 4-aligned
        const aligned = try proto.Region.alloc(m.words.len);
        defer aligned.free();
        @memcpy(aligned.mem[0..m.words.len], m.words);
        const words: [*]const u32 = @ptrCast(aligned.mem.ptr);
        try check(fns.createShaderModule(d, &.{ .codeSize = m.words.len, .pCode = words }, null, &pl.shader));
        var lb: [max_bind]vk.DescriptorSetLayoutBinding = undefined;
        for (0..m.nbind) |k| lb[k] = .{ .binding = @intCast(k) };
        try check(fns.createDescriptorSetLayout(d, &.{ .bindingCount = m.nbind, .pBindings = &lb }, null, &pl.dsl));
        const pcr = [_]vk.PushConstantRange{.{ .size = @max(m.npush, 1) * 4 }};
        const sl = [_]vk.Handle{pl.dsl};
        try check(fns.createPipelineLayout(d, &.{ .pSetLayouts = &sl, .pushConstantRangeCount = if (m.npush > 0) 1 else 0, .pPushConstantRanges = &pcr }, null, &pl.layout));
        const ci = [_]vk.ComputePipelineCreateInfo{.{ .stage = .{ .module = pl.shader }, .layout = pl.layout }};
        try check(fns.createComputePipelines(d, 0, 1, &ci, null, @ptrCast(&pl.pipe)));
    }

    // ---- descriptor sets, one per dispatch ----
    const ps = [_]vk.DescriptorPoolSize{.{ .descriptorCount = @max(total_binds, 1) }};
    try check(fns.createDescriptorPool(d, &.{ .maxSets = @max(ndisp, 1), .pPoolSizes = &ps }, null, &s.pool));
    const sets = try Table(vk.Handle).init(c, ndisp, 0);
    defer sets.free();
    for (disps.items, sets.items) |dp, *set| {
        const sl = [_]vk.Handle{s.pipes[dp.module].dsl};
        try check(fns.allocateDescriptorSets(d, &.{ .descriptorPool = s.pool, .pSetLayouts = &sl }, set));
        var infos: [max_bind]vk.DescriptorBufferInfo = undefined;
        var writes: [max_bind]vk.WriteDescriptorSet = undefined;
        for (dp.binds[0..dp.nbind], 0..) |bd, k| {
            const bi = switch (bd) {
                .buf => |x| x,
                .window => |w| w.win,
            };
            infos[k] = .{ .buffer = s.bufs[bi].buffer };
            writes[k] = .{ .dstSet = set.*, .dstBinding = @intCast(k), .pBufferInfo = &infos[k] };
        }
        fns.updateDescriptorSets(d, dp.nbind, &writes, 0, null);
    }

    // ---- record: for t in iterations — windows, dispatches, emits, state copies ----
    try check(fns.createCommandPool(d, &.{ .queueFamilyIndex = qfamily }, null, &s.cmd_pool));
    var cb: ?vk.CommandBuffer = null;
    try check(fns.allocateCommandBuffers(d, &.{ .commandPool = s.cmd_pool }, &cb));
    try check(fns.beginCommandBuffer(cb.?, &.{}));
    var t: u32 = 0;
    while (t < iters) : (t += 1) {
        for (disps.items, sets.items) |dp, set| {
            var any_window = false;
            for (dp.binds[0..dp.nbind]) |bd| switch (bd) {
                .window => |w| {
                    copy(cb.?, s.bufs[w.idx].buffer, w.base + w.stride * t, s.bufs[w.win].buffer, 0, w.len);
                    any_window = true;
                },
                .buf => {},
            };
            if (any_window) barrier(cb.?);
            const pl = s.pipes[dp.module];
            fns.cmdBindPipeline(cb.?, vk.PIPELINE_BIND_COMPUTE, pl.pipe);
            const one = [_]vk.Handle{set};
            fns.cmdBindDescriptorSets(cb.?, vk.PIPELINE_BIND_COMPUTE, pl.layout, 0, 1, &one, 0, null);
            if (dp.npush > 0) fns.cmdPushConstants(cb.?, pl.layout, vk.STAGE_COMPUTE, 0, dp.npush * 4, &dp.push);
            fns.cmdDispatch(cb.?, dp.groups[0], dp.groups[1], dp.groups[2]);
            barrier(cb.?);
        }
        for (emits.items) |e| copy(cb.?, s.bufs[e.buf].buffer, e.base + e.stride * t, s.bufs[e.stage].buffer, e.len * t, e.len);
        for (copies.items) |k| copy(cb.?, s.bufs[k.src].buffer, k.soff, s.bufs[k.dst].buffer, k.doff, k.len);
        barrier(cb.?);
    }
    // make device writes visible to the host before read-back
    const hb = [_]vk.MemoryBarrier{.{ .srcAccessMask = vk.ACCESS_SHADER_WRITE | vk.ACCESS_TRANSFER_WRITE, .dstAccessMask = vk.ACCESS_HOST_READ }};
    fns.cmdPipelineBarrier(cb.?, vk.PIPE_STAGE_COMPUTE | vk.PIPE_STAGE_TRANSFER, vk.PIPE_STAGE_HOST, 0, 1, &hb, 0, null, 0, null);
    try check(fns.endCommandBuffer(cb.?));

    try check(fns.createFence(d, &.{}, null, &s.fence));
    const t0 = proto.monotonicNs();
    const cbs = [_]vk.CommandBuffer{cb.?};
    const si = [_]vk.SubmitInfo{.{ .pCommandBuffers = &cbs }};
    try check(fns.queueSubmit(queue.?, 1, &si, s.fence));
    const fences = [_]vk.Handle{s.fence};
    const timeout_ns: u64 = if (deadline_ms == 0) std.math.maxInt(u64) else @as(u64, deadline_ms) * 1_000_000;
    const wr = fns.waitForFences(d, 1, &fences, 1, timeout_ns);
    if (wr == vk.TIMEOUT) return error.DeviceLost; // a hung queue is treated as a lost device
    try check(wr);
    const elapsed = proto.monotonicNs() - t0;

    // ---- emits (per iteration), then DONE with the returned buffers ----
    const parts = try Table([]const u8).init(c, @max(nemit + 1, 1 + 2 * nret), 0);
    defer parts.free();
    t = 0;
    while (t < iters and nemit > 0) : (t += 1) {
        const head = [_]u8{@intFromEnum(Op.emit)} ++ proto.le(u32, t);
        parts.items[0] = &head;
        for (emits.items, 1..) |e, i| parts.items[i] = s.bufs[e.stage].map[e.len * t .. e.len * (t + 1)];
        try proto.writeFrame(1, parts.items[0 .. nemit + 1]);
    }
    const lens = try Table([8]u8).init(c, nret, 0);
    defer lens.free();
    // no event counters on the fabric: an empty counter block
    const head = [_]u8{@intFromEnum(Op.done)} ++ proto.le(u64, elapsed) ++ proto.le(u64, 0) ++ [_]u8{0};
    parts.items[0] = &head;
    for (rets.items, 0..) |r, i| {
        lens.items[i] = proto.le(u64, s.bufs[r].len);
        parts.items[1 + 2 * i] = &lens.items[i];
        parts.items[2 + 2 * i] = s.bufs[r].map[0..s.bufs[r].len];
    }
    try proto.writeFrame(1, parts.items[0 .. 1 + 2 * nret]);
}

fn copyFile(path: []const u8, len: u64) !Buf {
    const m = try mapFile(path, len);
    defer _ = linux.munmap(m.ptr, m.len);
    const b = try hostBuffer(len);
    @memcpy(b.map[0..len], m[0..len]);
    return b;
}

// ------------------------------------------------------------ sessions --
//
// A RUN builds and destroys everything — buffers, pipelines, descriptor
// sets, a command buffer — around one unit: right for a one-shot program,
// ruinous for an autoregressive loop, where every token would re-create
// the pipelines and re-upload the weights and the KV cache. A *session* is
// the GPU twin of the worker's: OPEN creates the program's buffers at their
// certified maximal extents (weights and caches resident on the device),
// its pipelines once; a STEP writes only its inputs, replays the schedule
// resolved for the step's extents and reads back only the outputs asked for.
//
// Memory, by what the device offers:
//   * direct — a DEVICE_LOCAL|HOST_VISIBLE type exists (integrated GPUs,
//     resizable BAR, lavapipe): buffers live there, the host writes inputs
//     and reads outputs in place.
//   * staged — device-local memory the host cannot map (a discrete GPU
//     without BAR, or a direct allocation that ran out): every host↔device
//     byte goes through one host-visible staging buffer, the copies are
//     recorded in the step's command buffer, and weights are uploaded once
//     at OPEN. `flags` bit 0 forces this path (so it is testable anywhere).
//
// Recorded command buffers are cached per session, keyed by the exact bytes
// of everything that shapes the recording (dispatch geometry, push
// constants, bindings, copies, the write/return layout): a decode loop
// whose step has the same extents as the last one submits an already
// recorded buffer — no re-recording, no descriptor updates. Entries are
// compared byte for byte, never by hash alone.
//
// Bits are those of RUN: the same SPIR-V, the same dispatches, the same
// barriers. Only where buffers live and when commands are recorded changes.

const max_sessions = 8;
const cache_slots = 8;
const staging_chunk: u64 = 64 << 20;

const Recorded = struct {
    key: ?proto.Region = null,
    key_len: usize = 0,
    pool: vk.Handle = 0,
    cb: ?vk.CommandBuffer = null,
    last_use: u64 = 0,
};

const Sess = struct {
    live: bool = false,
    staged: bool = false,
    device_local: bool = false,
    bufs: []Buf = &.{},
    writable: []bool = &.{},
    nbuf: u32 = 0,
    pipes: []Pipe = &.{},
    regions: [3]?proto.Region = .{ null, null, null },
    cmd_pool: vk.Handle = 0,
    fence: vk.Handle = 0,
    staging: Buf = .{},
    staging_cap: u64 = 0,
    cache: [cache_slots]Recorded = [_]Recorded{.{}} ** cache_slots,
    tick: u64 = 0,
    bytes: u64 = 0,

    fn dropCache(s: *Sess) void {
        const d = device.?;
        for (&s.cache) |*e| {
            if (e.cb) |cb| fns.freeCommandBuffers(d, s.cmd_pool, 1, &[_]vk.CommandBuffer{cb});
            if (e.pool != 0) fns.destroyDescriptorPool(d, e.pool, null);
            if (e.key) |k| k.free();
            e.* = .{};
        }
    }

    fn freeStaging(s: *Sess) void {
        const d = device.?;
        if (s.staging.buffer != 0) fns.destroyBuffer(d, s.staging.buffer, null);
        if (s.staging.memory != 0) fns.freeMemory(d, s.staging.memory, null);
        s.staging = .{};
        s.staging_cap = 0;
    }

    fn deinit(s: *Sess) void {
        if (!s.live and s.nbuf == 0 and s.cmd_pool == 0) return;
        const d = device.?;
        s.dropCache();
        s.freeStaging();
        if (s.fence != 0) fns.destroyFence(d, s.fence, null);
        if (s.cmd_pool != 0) fns.destroyCommandPool(d, s.cmd_pool, null);
        for (s.pipes) |p| {
            if (p.pipe != 0) fns.destroyPipeline(d, p.pipe, null);
            if (p.layout != 0) fns.destroyPipelineLayout(d, p.layout, null);
            if (p.dsl != 0) fns.destroyDescriptorSetLayout(d, p.dsl, null);
            if (p.shader != 0) fns.destroyShaderModule(d, p.shader, null);
        }
        for (s.bufs[0..s.nbuf]) |b| {
            if (b.buffer != 0) fns.destroyBuffer(d, b.buffer, null);
            if (b.memory != 0) fns.freeMemory(d, b.memory, null);
            if (b.file_map) |m| _ = linux.munmap(m.ptr, m.len);
        }
        for (s.regions) |r| if (r) |x| x.free();
        s.* = .{};
    }
};

var sessions: [max_sessions]Sess = [_]Sess{.{}} ** max_sessions;

/// A memory type with all of `want`, preferring one that also has `prefer`.
fn memoryTypePref(bits: u32, want: u32, prefer: u32) !u32 {
    return memoryType(bits, want | prefer) catch memoryType(bits, want);
}

fn hasDirectType() bool {
    var i: u32 = 0;
    const f = vk.MEM_DEVICE_LOCAL | vk.MEM_HOST_VISIBLE | vk.MEM_HOST_COHERENT;
    while (i < mem_props.memoryTypeCount) : (i += 1) if (mem_props.memoryTypes[i].propertyFlags & f == f) return true;
    return false;
}

/// A session buffer: mapped device-local memory (direct) or unmapped
/// device-local memory (staged).
fn sessionBuffer(len: u64, staged: bool, device_local: *bool) !Buf {
    const d = device.?;
    var b = Buf{ .len = len };
    const size = @max(std.mem.alignForward(u64, len, 4), 4);
    try check(fns.createBuffer(d, &.{ .size = size, .usage = vk.BUF_STORAGE | vk.BUF_TRANSFER_SRC | vk.BUF_TRANSFER_DST }, null, &b.buffer));
    errdefer fns.destroyBuffer(d, b.buffer, null);
    var req: vk.MemoryRequirements = undefined;
    fns.getBufferMemoryRequirements(d, b.buffer, &req);
    const mt = if (staged)
        try memoryTypePref(req.memoryTypeBits, 0, vk.MEM_DEVICE_LOCAL)
    else
        try memoryTypePref(req.memoryTypeBits, vk.MEM_HOST_VISIBLE | vk.MEM_HOST_COHERENT, vk.MEM_DEVICE_LOCAL);
    if (mem_props.memoryTypes[mt].propertyFlags & vk.MEM_DEVICE_LOCAL == 0) device_local.* = false;
    try check(fns.allocateMemory(d, &.{ .allocationSize = req.size, .memoryTypeIndex = mt }, null, &b.memory));
    errdefer fns.freeMemory(d, b.memory, null);
    try check(fns.bindBufferMemory(d, b.buffer, b.memory, 0));
    if (!staged) {
        var p: ?*anyopaque = null;
        try check(fns.mapMemory(d, b.memory, 0, vk.WHOLE_SIZE, 0, &p));
        b.map = @ptrCast(p.?);
        b.mapped = true;
    }
    return b;
}

/// Grow the staging buffer to at least `need` bytes (cached recordings
/// name the old one, so they are dropped).
fn ensureStaging(s: *Sess, need: u64) !void {
    if (need <= s.staging_cap) return;
    s.dropCache();
    s.freeStaging();
    var cap: u64 = 1 << 16;
    while (cap < need) cap *= 2;
    s.staging = try hostBuffer(cap);
    s.staging_cap = cap;
}

fn beginOnce(s: *Sess) !vk.CommandBuffer {
    var cb: ?vk.CommandBuffer = null;
    try check(fns.allocateCommandBuffers(device.?, &.{ .commandPool = s.cmd_pool }, &cb));
    try check(fns.beginCommandBuffer(cb.?, &.{}));
    return cb.?;
}

fn submitWait(s: *Sess, cb: vk.CommandBuffer, deadline_ms: u32) !void {
    const d = device.?;
    const cbs = [_]vk.CommandBuffer{cb};
    const si = [_]vk.SubmitInfo{.{ .pCommandBuffers = &cbs }};
    const fences = [_]vk.Handle{s.fence};
    try check(fns.resetFences(d, 1, &fences));
    try check(fns.queueSubmit(queue.?, 1, &si, s.fence));
    const timeout_ns: u64 = if (deadline_ms == 0) std.math.maxInt(u64) else @as(u64, deadline_ms) * 1_000_000;
    const wr = fns.waitForFences(d, 1, &fences, 1, timeout_ns);
    if (wr == vk.TIMEOUT) return error.DeviceLost;
    try check(wr);
}

fn endSubmitFree(s: *Sess, cb: vk.CommandBuffer) !void {
    try check(fns.endCommandBuffer(cb));
    defer fns.freeCommandBuffers(device.?, s.cmd_pool, 1, &[_]vk.CommandBuffer{cb});
    try submitWait(s, cb, 0);
}

/// Staged upload of `bytes` into `dst` at `off`, in staging-sized chunks.
fn upload(s: *Sess, dst: vk.Handle, off: u64, bytes: []const u8) !void {
    var done_: u64 = 0;
    while (done_ < bytes.len) {
        const n = @min(bytes.len - done_, staging_chunk);
        try ensureStaging(s, n);
        @memcpy(s.staging.map[0..n], bytes[done_ .. done_ + n]);
        const cb = try beginOnce(s);
        copy(cb, s.staging.buffer, 0, dst, off + done_, n);
        try endSubmitFree(s, cb);
        done_ += n;
    }
}

/// OPEN: `deadline flags modules buffers` (a RUN's module and buffer
/// tables). Replies `OPEN sid staged device_local resident_bytes`.
fn openSession(c: *proto.Cursor) !void {
    _ = try c.int(u32); // deadline: OPEN is bounded by the BEAM-side timeout
    const flags = try c.int(u32);
    const start = c.pos;

    var sid: u32 = 0;
    while (sid < max_sessions and sessions[sid].live) sid += 1;
    if (sid == max_sessions) return error.TooManySessions;

    const force_staged = flags & 1 != 0;
    var staged = force_staged or !hasDirectType();
    const s = &sessions[sid];
    buildSession(s, c, staged) catch |e| {
        s.deinit();
        if (e != error.OutOfDeviceMemory or staged) return e;
        // the mappable device-local heap is small (a BAR window): stage instead
        staged = true;
        c.pos = start;
        try buildSession(s, c, true);
    };
    s.live = true;
    const head = [_]u8{ @intFromEnum(Op.open) } ++ proto.le(u32, sid) ++
        [_]u8{ @intFromBool(s.staged), @intFromBool(s.device_local) } ++ proto.le(u64, s.bytes);
    try proto.writeFrame(1, &.{&head});
}

fn buildSession(s: *Sess, c: *proto.Cursor, staged: bool) !void {
    const d = device.?;
    s.staged = staged;
    s.device_local = true;

    const nmod = try c.int(u32);
    const mods = try Table(Module).init(c, nmod, 12);
    defer mods.free();
    for (mods.items) |*m| {
        const len = try c.int(u32);
        m.words = try c.bytes(len);
        m.nbind = try c.int(u32);
        m.npush = try c.int(u32);
        if (m.nbind > max_bind or m.npush > max_push or len % 4 != 0) return error.Truncated;
    }
    const nbuf = try c.int(u32);
    const specs = try Table(BufSpec).init(c, nbuf, 10);
    defer specs.free();
    const wr = try proto.Region.alloc(@as(usize, nbuf) * @sizeOf(bool));
    s.regions[2] = wr;
    s.writable = @as([*]bool, @ptrCast(wr.mem.ptr))[0..nbuf];
    for (specs.items, 0..) |*b, i| {
        const kind = try c.int(u8);
        s.writable[i] = try c.int(u8) != 0;
        const len = try c.int(u64);
        b.* = .{ .kind = kind, .len = len };
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

    try check(fns.createCommandPool(d, &.{ .queueFamilyIndex = qfamily, .flags = 0x2 }, null, &s.cmd_pool)); // RESET_COMMAND_BUFFER
    try check(fns.createFence(d, &.{}, null, &s.fence));

    // ---- buffers, resident until CLOSE ----
    const all = try proto.Region.alloc(@as(usize, @max(nbuf, 1)) * @sizeOf(Buf));
    s.regions[0] = all;
    s.bufs = @as([*]Buf, @ptrCast(@alignCast(all.mem.ptr)))[0..nbuf];
    for (specs.items, 0..) |b, i| {
        if (!staged and b.kind == 2 and ext_host and !s.writable[i]) {
            // read-only weights on a mappable device: import the shared file, no copy
            if (importFile(b.path, b.len)) |imp| {
                s.bufs[i] = imp;
                s.nbuf += 1;
                s.bytes += b.len;
                continue;
            } else |_| {}
        }
        s.bufs[i] = try sessionBuffer(b.len, staged, &s.device_local);
        s.nbuf += 1;
        s.bytes += b.len;
        const dst = s.bufs[i];
        switch (b.kind) {
            0 => if (staged) {
                const cb = try beginOnce(s);
                fns.cmdFillBuffer(cb, dst.buffer, 0, vk.WHOLE_SIZE, 0);
                try endSubmitFree(s, cb);
            } else @memset(dst.map[0..b.len], 0),
            1 => if (staged) try upload(s, dst.buffer, 0, b.data) else @memcpy(dst.map[0..b.len], b.data),
            else => {
                const m = try mapFile(b.path, b.len);
                defer _ = linux.munmap(m.ptr, m.len);
                if (staged) try upload(s, dst.buffer, 0, m[0..b.len]) else @memcpy(dst.map[0..b.len], m[0..b.len]);
            },
        }
    }

    // ---- pipelines, once ----
    const pipes = try proto.Region.alloc(@as(usize, @max(nmod, 1)) * @sizeOf(Pipe));
    s.regions[1] = pipes;
    s.pipes = @as([*]Pipe, @ptrCast(@alignCast(pipes.mem.ptr)))[0..nmod];
    for (s.pipes) |*p| p.* = .{};
    for (mods.items, s.pipes) |m, *pl| {
        const aligned = try proto.Region.alloc(m.words.len);
        defer aligned.free();
        @memcpy(aligned.mem[0..m.words.len], m.words);
        const words: [*]const u32 = @ptrCast(aligned.mem.ptr);
        try check(fns.createShaderModule(d, &.{ .codeSize = m.words.len, .pCode = words }, null, &pl.shader));
        var lb: [max_bind]vk.DescriptorSetLayoutBinding = undefined;
        for (0..m.nbind) |k| lb[k] = .{ .binding = @intCast(k) };
        try check(fns.createDescriptorSetLayout(d, &.{ .bindingCount = m.nbind, .pBindings = &lb }, null, &pl.dsl));
        const pcr = [_]vk.PushConstantRange{.{ .size = @max(m.npush, 1) * 4 }};
        const sl = [_]vk.Handle{pl.dsl};
        try check(fns.createPipelineLayout(d, &.{ .pSetLayouts = &sl, .pushConstantRangeCount = if (m.npush > 0) 1 else 0, .pPushConstantRanges = &pcr }, null, &pl.layout));
        const ci = [_]vk.ComputePipelineCreateInfo{.{ .stage = .{ .module = pl.shader }, .layout = pl.layout }};
        try check(fns.createComputePipelines(d, 0, 1, &ci, null, @ptrCast(&pl.pipe)));
    }
}

const Io = struct { buf: u32, off: u64, len: u64, data: []const u8 = &.{}, stage: u64 = 0 };
const SDisp = struct { module: u32, groups: [3]u32, npush: u32, push: [max_push]u32, nbind: u32, binds: [max_bind]u32 };

/// STEP: `sid deadline writes returns dispatches copies` with
///   writes  = n × (buf:u32, off:u64, len:u64, bytes)
///   returns = n × (buf:u32, off:u64, len:u64)
///   dispatch = module:u32 gx gy gz npush push[npush] nbind binds[nbind]:u32
///   copies  = n × (src:u32, soff:u64, dst:u32, doff:u64, len:u64), after the pass
/// Replies DONE (elapsed, counters: 7 = recording reused, 8 = host bytes) with the returns.
fn stepSession(c: *proto.Cursor) !void {
    const d = device.?;
    const sid = try c.int(u32);
    if (sid >= max_sessions or !sessions[sid].live) return error.NoSession;
    const s = &sessions[sid];
    const deadline_ms = try c.int(u32);
    const nb = s.nbuf;

    // ---- writes and returns: validated, laid out in staging ----
    const nw = try c.int(u32);
    const writes = try Table(Io).init(c, nw, 20);
    defer writes.free();
    var stage_at: u64 = 0;
    var host_bytes: u64 = 0;
    for (writes.items) |*w| {
        w.* = .{ .buf = try c.int(u32), .off = try c.int(u64), .len = try c.int(u64) };
        w.data = try c.bytes(w.len);
        if (w.buf >= nb or !s.writable[w.buf] or w.off + w.len > s.bufs[w.buf].len) return error.Truncated;
        w.stage = stage_at;
        stage_at = std.mem.alignForward(u64, stage_at + w.len, 16);
        host_bytes += w.len;
    }
    const tail_start = c.pos;
    const nr = try c.int(u32);
    const rets = try Table(Io).init(c, nr, 20);
    defer rets.free();
    for (rets.items) |*r| {
        r.* = .{ .buf = try c.int(u32), .off = try c.int(u64), .len = try c.int(u64) };
        if (r.buf >= nb or r.off + r.len > s.bufs[r.buf].len) return error.Truncated;
        r.stage = stage_at;
        stage_at = std.mem.alignForward(u64, stage_at + r.len, 16);
        host_bytes += r.len;
    }
    const nd = try c.int(u32);
    const disps = try Table(SDisp).init(c, nd, 24);
    defer disps.free();
    for (disps.items) |*dp| {
        dp.module = try c.int(u32);
        if (dp.module >= s.pipes.len) return error.Truncated;
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
    if (s.staged) try ensureStaging(s, @max(stage_at, 16));

    // ---- the recording's key: write layout (not data) + everything after it ----
    const tail = c.buf[tail_start..c.pos];
    const key_len = 4 + nw * 20 + tail.len;
    const key = try proto.Region.alloc(key_len);
    var key_owned = true;
    defer if (key_owned) key.free();
    {
        var at: usize = 0;
        @memcpy(key.mem[0..4], &proto.le(u32, nw));
        at = 4;
        for (writes.items) |w| {
            @memcpy(key.mem[at..][0..4], &proto.le(u32, w.buf));
            @memcpy(key.mem[at + 4 ..][0..8], &proto.le(u64, w.off));
            @memcpy(key.mem[at + 12 ..][0..8], &proto.le(u64, w.len));
            at += 20;
        }
        @memcpy(key.mem[at .. at + tail.len], tail);
    }

    s.tick += 1;
    var hit: ?*Recorded = null;
    for (&s.cache) |*e| if (e.key) |k| {
        if (e.key_len == key_len and std.mem.eql(u8, k.mem[0..key_len], key.mem[0..key_len])) {
            hit = e;
            break;
        }
    };
    const reused = hit != null;
    const entry = hit orelse blk: {
        // evict the least recently used slot and record into it
        var victim = &s.cache[0];
        for (&s.cache) |*e| if (e.key == null or e.last_use < victim.last_use) {
            victim = e;
            if (e.key == null) break;
        };
        if (victim.cb) |cb| fns.freeCommandBuffers(d, s.cmd_pool, 1, &[_]vk.CommandBuffer{cb});
        if (victim.pool != 0) fns.destroyDescriptorPool(d, victim.pool, null);
        if (victim.key) |k| k.free();
        victim.* = .{};
        try record(s, victim, writes.items, rets.items, disps.items, copies.items);
        victim.key = key;
        victim.key_len = key_len;
        key_owned = false;
        break :blk victim;
    };
    entry.last_use = s.tick;

    // ---- host → device, submit, device → host ----
    const t0 = proto.monotonicNs();
    for (writes.items) |w| {
        if (s.staged) @memcpy(s.staging.map[w.stage .. w.stage + w.len], w.data) else @memcpy(s.bufs[w.buf].map[w.off .. w.off + w.len], w.data);
    }
    try submitWait(s, entry.cb.?, deadline_ms);
    const elapsed = proto.monotonicNs() - t0;

    const parts = try Table([]const u8).init(c, 1 + 2 * nr, 0);
    defer parts.free();
    const lens = try Table([8]u8).init(c, nr, 0);
    defer lens.free();
    var counters_blk: [1 + 18]u8 = undefined;
    counters_blk[0] = 2;
    counters_blk[1] = 7;
    @memcpy(counters_blk[2..10], &proto.le(u64, @intFromBool(reused)));
    counters_blk[10] = 8;
    @memcpy(counters_blk[11..19], &proto.le(u64, host_bytes));
    const head = [_]u8{@intFromEnum(Op.done)} ++ proto.le(u64, elapsed) ++ proto.le(u64, 0) ++ counters_blk;
    parts.items[0] = &head;
    for (rets.items, 0..) |r, i| {
        lens.items[i] = proto.le(u64, r.len);
        parts.items[1 + 2 * i] = &lens.items[i];
        parts.items[2 + 2 * i] = if (s.staged) s.staging.map[r.stage .. r.stage + r.len] else s.bufs[r.buf].map[r.off .. r.off + r.len];
    }
    try proto.writeFrame(1, parts.items[0 .. 1 + 2 * nr]);
}

fn record(s: *Sess, e: *Recorded, writes: []const Io, rets: []const Io, disps: []const SDisp, copies: []const Copy) !void {
    const d = device.?;
    var total_binds: u32 = 0;
    for (disps) |dp| total_binds += dp.nbind;
    const ps = [_]vk.DescriptorPoolSize{.{ .descriptorCount = @max(total_binds, 1) }};
    try check(fns.createDescriptorPool(d, &.{ .maxSets = @max(@as(u32, @intCast(disps.len)), 1), .pPoolSizes = &ps }, null, &e.pool));

    var cb: ?vk.CommandBuffer = null;
    try check(fns.allocateCommandBuffers(d, &.{ .commandPool = s.cmd_pool }, &cb));
    e.cb = cb;
    try check(fns.beginCommandBuffer(cb.?, &.{ .flags = 0 })); // reusable: no ONE_TIME_SUBMIT

    if (s.staged and writes.len > 0) {
        for (writes) |w| copy(cb.?, s.staging.buffer, w.stage, s.bufs[w.buf].buffer, w.off, w.len);
        barrier(cb.?);
    }
    for (disps) |dp| {
        var set: vk.Handle = 0;
        const pl = s.pipes[dp.module];
        const sl = [_]vk.Handle{pl.dsl};
        try check(fns.allocateDescriptorSets(d, &.{ .descriptorPool = e.pool, .pSetLayouts = &sl }, &set));
        var infos: [max_bind]vk.DescriptorBufferInfo = undefined;
        var wds: [max_bind]vk.WriteDescriptorSet = undefined;
        for (dp.binds[0..dp.nbind], 0..) |bi, k| {
            infos[k] = .{ .buffer = s.bufs[bi].buffer };
            wds[k] = .{ .dstSet = set, .dstBinding = @intCast(k), .pBufferInfo = &infos[k] };
        }
        fns.updateDescriptorSets(d, dp.nbind, &wds, 0, null);
        fns.cmdBindPipeline(cb.?, vk.PIPELINE_BIND_COMPUTE, pl.pipe);
        const one = [_]vk.Handle{set};
        fns.cmdBindDescriptorSets(cb.?, vk.PIPELINE_BIND_COMPUTE, pl.layout, 0, 1, &one, 0, null);
        if (dp.npush > 0) fns.cmdPushConstants(cb.?, pl.layout, vk.STAGE_COMPUTE, 0, dp.npush * 4, &dp.push);
        fns.cmdDispatch(cb.?, dp.groups[0], dp.groups[1], dp.groups[2]);
        barrier(cb.?);
    }
    for (copies) |k| copy(cb.?, s.bufs[k.src].buffer, k.soff, s.bufs[k.dst].buffer, k.doff, k.len);
    if (copies.len > 0) barrier(cb.?);
    if (s.staged) {
        for (rets) |r| copy(cb.?, s.bufs[r.buf].buffer, r.off, s.staging.buffer, r.stage, r.len);
        if (rets.len > 0) barrier(cb.?);
    }
    const hb = [_]vk.MemoryBarrier{.{ .srcAccessMask = vk.ACCESS_SHADER_WRITE | vk.ACCESS_TRANSFER_WRITE, .dstAccessMask = vk.ACCESS_HOST_READ }};
    fns.cmdPipelineBarrier(cb.?, vk.PIPE_STAGE_COMPUTE | vk.PIPE_STAGE_TRANSFER, vk.PIPE_STAGE_HOST, 0, 1, &hb, 0, null, 0, null);
    try check(fns.endCommandBuffer(cb.?));
}
