//! In-tree RV64GCV (RVV 1.0) interpreter — the execution substrate for
//! RISC-V vector code on hosts that are not RISC-V.
//!
//! Scope is exactly the instruction subset vapor's RVV backend emits (plus
//! the obvious neighbours of each form). It is not a general emulator; it is
//! a *checked* one:
//!
//!   * every load/store is bounds-checked against the buffers the caller
//!     bound (no access outside them can happen — a fault is an error value,
//!     never a signal);
//!   * every instruction fetch must lie inside the code blob;
//!   * an unknown or reserved encoding is `IllegalInstruction`, reported with
//!     its pc and word — the interpreter never raises SIGILL;
//!   * a fuel budget bounds execution time.
//!
//! Semantics follow the RVV 1.0 and F specifications: RNE arithmetic, fused
//! `vfmacc`/`fmadd`, canonical NaN results, NaN-boxing of single-precision
//! scalars, register-group alignment checks, `vill` on unsupported vtype.
//! With `poison` set, tail/mask-agnostic elements are overwritten with all
//! ones (a legal RVV behaviour), which proves kernels never depend on them.
//! It is differential-tested against QEMU's RVV implementation running the
//! same machine code.

const std = @import("std");

pub const Fault = error{ IllegalInstruction, MemoryFault, FuelExhausted, MisalignedFetch };

pub const Span = struct { base: u64, len: u64, writable: bool };

pub const max_vlenb = 64; // VLEN ≤ 512
const canonical_nan: u32 = 0x7FC0_0000;
const return_sentinel: u64 = 0;

pub const Hart = struct {
    x: [32]u64 = [_]u64{0} ** 32,
    f: [32]u64 = [_]u64{0} ** 32,
    v: [32 * max_vlenb]u8 = [_]u8{0} ** (32 * max_vlenb),
    vlenb: u32,
    vl: u64 = 0,
    vtype: u64 = 1 << 63,
    pc: u64 = 0,
    code: []const u8,
    code_base: u64,
    spans: []const Span,
    fuel: u64,
    retired: u64 = 0,
    poison: bool = false,
    fault_pc: u64 = 0,
    fault_word: u32 = 0,

    /// Run the function at `entry` with a0 = `a0` until it returns.
    pub fn call(self: *Hart, entry: u64, a0: u64, sp: u64) Fault!void {
        self.x[1] = return_sentinel;
        self.x[2] = sp;
        self.x[10] = a0;
        self.pc = self.code_base + entry;
        while (self.pc != return_sentinel) {
            if (self.fuel != 0 and self.retired >= self.fuel) return error.FuelExhausted;
            const w = try self.fetch();
            self.step(w) catch |e| {
                self.fault_pc = self.pc;
                self.fault_word = w;
                return e;
            };
            self.x[0] = 0;
            self.retired += 1;
        }
    }

    fn fetch(self: *Hart) Fault!u32 {
        if (self.pc & 3 != 0) return error.MisalignedFetch;
        if (self.pc < self.code_base or self.pc - self.code_base + 4 > self.code.len) {
            self.fault_pc = self.pc;
            return error.MemoryFault;
        }
        const off = self.pc - self.code_base;
        return std.mem.readInt(u32, self.code[off..][0..4], .little);
    }

    // ------------------------------------------------------------ memory --

    fn check(self: *Hart, addr: u64, n: u64, write: bool) Fault!void {
        for (self.spans) |s| {
            if (addr >= s.base and n <= s.len and addr - s.base <= s.len - n) {
                if (write and !s.writable) return error.MemoryFault;
                return;
            }
        }
        return error.MemoryFault;
    }

    fn load(self: *Hart, comptime T: type, addr: u64) Fault!T {
        try self.check(addr, @sizeOf(T), false);
        const p: *align(1) const T = @ptrFromInt(addr);
        return p.*;
    }

    fn store(self: *Hart, comptime T: type, addr: u64, val: T) Fault!void {
        try self.check(addr, @sizeOf(T), true);
        const p: *align(1) T = @ptrFromInt(addr);
        p.* = val;
    }

    // ------------------------------------------------------------ scalar --

    fn step(self: *Hart, w: u32) Fault!void {
        const op = w & 0x7F;
        const rd: u5 = @truncate(w >> 7);
        const f3: u3 = @truncate(w >> 12);
        const rs1: u5 = @truncate(w >> 15);
        const rs2: u5 = @truncate(w >> 20);
        const f7: u7 = @truncate(w >> 25);
        const next = self.pc + 4;
        const a = self.x[rs1];
        const b = self.x[rs2];

        switch (op) {
            0x37 => self.x[rd] = sext(w & 0xFFFF_F000, 32),
            0x13 => {
                const imm = immI(w);
                self.x[rd] = switch (f3) {
                    0 => a +% imm,
                    1 => if ((w >> 26) == 0) a << @truncate((w >> 20) & 63) else return error.IllegalInstruction,
                    4 => a ^ imm,
                    5 => switch (w >> 26) {
                        0 => a >> @truncate((w >> 20) & 63),
                        0x10 => @bitCast(@as(i64, @bitCast(a)) >> @truncate((w >> 20) & 63)),
                        else => return error.IllegalInstruction,
                    },
                    6 => a | imm,
                    7 => a & imm,
                    else => return error.IllegalInstruction,
                };
            },
            0x1B => {
                if (f3 != 0) return error.IllegalInstruction;
                self.x[rd] = sext((a +% immI(w)) & 0xFFFF_FFFF, 32);
            },
            0x33 => self.x[rd] = switch (@as(u10, f7) << 3 | f3) {
                0 => a +% b,
                0x100 => a -% b,
                8 => a *% b,
                1 => a << @truncate(b & 63),
                4 => a ^ b,
                5 => a >> @truncate(b & 63),
                6 => a | b,
                7 => a & b,
                else => return error.IllegalInstruction,
            },
            0x03 => {
                const ea = a +% immI(w);
                self.x[rd] = switch (f3) {
                    0 => sext(try self.load(u8, ea), 8),
                    1 => sext(try self.load(u16, ea), 16),
                    2 => sext(try self.load(u32, ea), 32),
                    3 => try self.load(u64, ea),
                    4 => try self.load(u8, ea),
                    5 => try self.load(u16, ea),
                    6 => try self.load(u32, ea),
                    else => return error.IllegalInstruction,
                };
            },
            0x23 => {
                const ea = a +% immS(w);
                switch (f3) {
                    0 => try self.store(u8, ea, @truncate(b)),
                    1 => try self.store(u16, ea, @truncate(b)),
                    2 => try self.store(u32, ea, @truncate(b)),
                    3 => try self.store(u64, ea, b),
                    else => return error.IllegalInstruction,
                }
            },
            0x63 => {
                const sa: i64 = @bitCast(a);
                const sb: i64 = @bitCast(b);
                const taken = switch (f3) {
                    0 => a == b,
                    1 => a != b,
                    4 => sa < sb,
                    5 => sa >= sb,
                    6 => a < b,
                    7 => a >= b,
                    else => return error.IllegalInstruction,
                };
                if (taken) {
                    self.pc +%= immB(w);
                    return;
                }
            },
            0x6F => {
                self.x[rd] = next;
                self.pc +%= immJ(w);
                return;
            },
            0x67 => {
                if (f3 != 0) return error.IllegalInstruction;
                const t = (a +% immI(w)) & ~@as(u64, 1);
                self.x[rd] = next;
                self.pc = t;
                return;
            },
            0x07 => switch (f3) {
                2 => self.f[rd] = box(try self.load(u32, a +% immI(w))),
                3 => self.f[rd] = try self.load(u64, a +% immI(w)),
                else => try self.vload(w, rd, rs1, f3),
            },
            0x27 => switch (f3) {
                2 => try self.store(u32, a +% immS(w), @truncate(self.f[rs2])),
                3 => try self.store(u64, a +% immS(w), self.f[rs2]),
                else => try self.vstore(w, rd, rs1, f3),
            },
            0x53 => try self.opfp(w, rd, f3, rs1, rs2, f7),
            0x43 => {
                if ((w >> 25) & 3 != 0 or !rneOk(f3)) return error.IllegalInstruction;
                const rs3: u5 = @truncate(w >> 27);
                self.f[rd] = box(canon(@mulAdd(f32, unboxS(self.f[rs1]), unboxS(self.f[rs2]), unboxS(self.f[rs3]))));
            },
            0x57 => try self.opv(w, rd, f3, rs1, rs2),
            else => return error.IllegalInstruction,
        }
        self.pc = next;
    }

    fn opfp(self: *Hart, w: u32, rd: u5, f3: u3, rs1: u5, rs2: u5, f7: u7) Fault!void {
        _ = w;
        switch (f7) {
            0x00, 0x04, 0x08 => {
                if (!rneOk(f3)) return error.IllegalInstruction;
                const x = unboxS(self.f[rs1]);
                const y = unboxS(self.f[rs2]);
                self.f[rd] = box(canon(switch (f7) {
                    0x00 => x + y,
                    0x04 => x - y,
                    else => x * y,
                }));
            },
            0x68 => {
                if (!rneOk(f3)) return error.IllegalInstruction;
                const src: u32 = @truncate(self.x[rs1]);
                const r: f32 = switch (rs2) {
                    0 => @floatFromInt(@as(i32, @bitCast(src))),
                    1 => @floatFromInt(src),
                    else => return error.IllegalInstruction,
                };
                self.f[rd] = box(@bitCast(r));
            },
            0x78 => {
                if (f3 != 0 or rs2 != 0) return error.IllegalInstruction;
                self.f[rd] = box(@truncate(self.x[rs1]));
            },
            0x70 => {
                if (f3 != 0 or rs2 != 0) return error.IllegalInstruction;
                self.x[rd] = sext(@as(u32, @truncate(self.f[rs1])), 32);
            },
            else => return error.IllegalInstruction,
        }
    }

    // ------------------------------------------------------------ vector --

    const VType = struct { sew: u32, lmul: u32, ta: bool, ma: bool };

    fn vt(self: *Hart) Fault!VType {
        if (self.vtype >> 63 != 0) return error.IllegalInstruction;
        const vlmul: u3 = @truncate(self.vtype);
        const vsew: u3 = @truncate(self.vtype >> 3);
        return .{
            .sew = @as(u32, 8) << vsew,
            .lmul = @as(u32, 1) << vlmul,
            .ta = (self.vtype >> 6) & 1 != 0,
            .ma = (self.vtype >> 7) & 1 != 0,
        };
    }

    fn vlmax(self: *Hart, t: VType) u64 {
        return @as(u64, t.lmul) * self.vlenb * 8 / t.sew;
    }

    fn setvtype(self: *Hart, vtypei: u64, avl: ?u64, rd: u5, keep_vl: bool) void {
        const vlmul = vtypei & 7;
        const vsew = (vtypei >> 3) & 7;
        const ok = vlmul <= 3 and vsew <= 3 and vtypei >> 8 == 0;
        if (!ok) {
            self.vtype = 1 << 63;
            self.vl = 0;
            self.x[rd] = 0;
            return;
        }
        const t = VType{ .sew = @as(u32, 8) << @truncate(vsew), .lmul = @as(u32, 1) << @truncate(vlmul), .ta = false, .ma = false };
        const max = self.vlmax(t);
        if (keep_vl) {
            // vsetvli x0, x0: legal only if VLMAX is unchanged by the new ratio
            if (self.vtype >> 63 != 0 or self.vlmax(self.vt() catch unreachable) != max) {
                self.vtype = 1 << 63;
                self.vl = 0;
                return;
            }
        } else {
            self.vl = @min(avl.?, max);
        }
        self.vtype = vtypei;
        self.x[rd] = self.vl;
    }

    fn group(self: *Hart, reg: u5, emul: u32) Fault!void {
        _ = self;
        if (emul > 8 or reg % emul != 0 or @as(u32, reg) + emul > 32) return error.IllegalInstruction;
    }

    fn eptr(self: *Hart, comptime T: type, reg: u5, i: u64) *align(1) T {
        const off = @as(usize, reg) * self.vlenb + @as(usize, @intCast(i)) * @sizeOf(T);
        return @ptrCast(&self.v[off]);
    }

    fn mbit(self: *Hart, reg: u5, i: u64) bool {
        const byte = self.v[@as(usize, reg) * self.vlenb + @as(usize, @intCast(i / 8))];
        return (byte >> @truncate(i % 8)) & 1 != 0;
    }

    fn setmbit(self: *Hart, reg: u5, i: u64, val: bool) void {
        const p = &self.v[@as(usize, reg) * self.vlenb + @as(usize, @intCast(i / 8))];
        const m: u8 = @as(u8, 1) << @truncate(i % 8);
        p.* = if (val) p.* | m else p.* & ~m;
    }

    fn active(self: *Hart, vm: bool, i: u64) bool {
        return vm or self.mbit(0, i);
    }

    /// Tail/mask-agnostic poisoning (all ones) for elements the policy frees.
    fn finish(self: *Hart, comptime T: type, vd: u5, t: VType, emul: u32, vm: bool) void {
        if (!self.poison) return;
        const n = @as(u64, emul) * self.vlenb / @sizeOf(T);
        if (t.ma and !vm) {
            var i: u64 = 0;
            while (i < self.vl) : (i += 1) if (!self.mbit(0, i)) {
                self.eptr(T, vd, i).* = ~@as(T, 0);
            };
        }
        if (t.ta) {
            var i: u64 = self.vl;
            while (i < n) : (i += 1) self.eptr(T, vd, i).* = ~@as(T, 0);
        }
    }

    fn vload(self: *Hart, w: u32, vd: u5, rs1: u5, width: u3) Fault!void {
        const vm = (w >> 25) & 1 != 0;
        if ((w >> 26) != 0 or ((w >> 20) & 31) != 0) return error.IllegalInstruction; // unit-stride, nf=0
        const t = try self.vt();
        const eew: u32 = switch (width) {
            0 => 8,
            5 => 16,
            6 => 32,
            7 => 64,
            else => return error.IllegalInstruction,
        };
        const emul = @max(1, eew * t.lmul / t.sew);
        try self.group(vd, emul);
        const base = self.x[rs1];
        var i: u64 = 0;
        while (i < self.vl) : (i += 1) {
            if (!self.active(vm, i)) continue;
            switch (eew) {
                8 => self.eptr(u8, vd, i).* = try self.load(u8, base + i),
                16 => self.eptr(u16, vd, i).* = try self.load(u16, base + 2 * i),
                32 => self.eptr(u32, vd, i).* = try self.load(u32, base + 4 * i),
                else => self.eptr(u64, vd, i).* = try self.load(u64, base + 8 * i),
            }
        }
        switch (eew) {
            8 => self.finish(u8, vd, t, emul, vm),
            16 => self.finish(u16, vd, t, emul, vm),
            32 => self.finish(u32, vd, t, emul, vm),
            else => self.finish(u64, vd, t, emul, vm),
        }
    }

    fn vstore(self: *Hart, w: u32, vs3: u5, rs1: u5, width: u3) Fault!void {
        const vm = (w >> 25) & 1 != 0;
        if ((w >> 26) != 0 or ((w >> 20) & 31) != 0) return error.IllegalInstruction;
        const t = try self.vt();
        const eew: u32 = switch (width) {
            0 => 8,
            5 => 16,
            6 => 32,
            7 => 64,
            else => return error.IllegalInstruction,
        };
        try self.group(vs3, @max(1, eew * t.lmul / t.sew));
        const base = self.x[rs1];
        var i: u64 = 0;
        while (i < self.vl) : (i += 1) {
            if (!self.active(vm, i)) continue;
            switch (eew) {
                8 => try self.store(u8, base + i, self.eptr(u8, vs3, i).*),
                16 => try self.store(u16, base + 2 * i, self.eptr(u16, vs3, i).*),
                32 => try self.store(u32, base + 4 * i, self.eptr(u32, vs3, i).*),
                else => try self.store(u64, base + 8 * i, self.eptr(u64, vs3, i).*),
            }
        }
    }

    fn opv(self: *Hart, w: u32, vd: u5, f3: u3, rs1: u5, vs2: u5) Fault!void {
        if (f3 == 7) {
            if (w >> 31 == 0) { // vsetvli
                const keep = rs1 == 0 and vd == 0;
                const avl: ?u64 = if (keep) null else if (rs1 == 0) std.math.maxInt(u64) else self.x[rs1];
                self.setvtype((w >> 20) & 0x7FF, avl, vd, keep);
            } else if (w >> 30 == 3) { // vsetivli
                self.setvtype((w >> 20) & 0x3FF, rs1, vd, false);
            } else {
                self.setvtype(self.x[vs2], if (rs1 == 0) std.math.maxInt(u64) else self.x[rs1], vd, false);
            }
            return;
        }

        const funct6 = w >> 26;
        const vm = (w >> 25) & 1 != 0;
        const vs1 = rs1;
        const t = try self.vt();
        const vl = self.vl;
        var i: u64 = 0;

        switch (f3) {
            // OPIVV
            0 => switch (funct6) {
                0x17 => { // vmv.v.v (vm=1) / vmerge.vvm (vm=0)
                    try self.group(vd, t.lmul);
                    try self.group(vs1, t.lmul);
                    if (!vm) try self.group(vs2, t.lmul);
                    try self.lanes2(t, vd, vs2, vs1, vm, .merge);
                },
                0x00 => try self.lanes2(t, vd, vs2, vs1, vm, .add),
                0x02 => try self.lanes2(t, vd, vs2, vs1, vm, .sub),
                0x09 => try self.lanes2(t, vd, vs2, vs1, vm, .@"and"),
                0x0B => try self.lanes2(t, vd, vs2, vs1, vm, .xor),
                else => return error.IllegalInstruction,
            },
            // OPIVX: vadd/vsub/vrsub/vand/vxor .vx, vmv.v.x
            4 => {
                try self.group(vd, t.lmul);
                const kind: LaneX = switch (funct6) {
                    0x00 => .add,
                    0x02 => .sub,
                    0x03 => .rsub,
                    0x09 => .@"and",
                    0x0B => .xor,
                    0x17 => if (vm and vs2 == 0) .mv else return error.IllegalInstruction,
                    else => return error.IllegalInstruction,
                };
                if (kind != .mv) try self.group(vs2, t.lmul);
                const x = self.x[rs1];
                switch (t.sew) {
                    8 => self.lanesX(u8, t, vd, vs2, @truncate(x), vm, kind),
                    16 => self.lanesX(u16, t, vd, vs2, @truncate(x), vm, kind),
                    32 => self.lanesX(u32, t, vd, vs2, @truncate(x), vm, kind),
                    else => self.lanesX(u64, t, vd, vs2, x, vm, kind),
                }
            },
            // OPFVV
            1 => {
                if (t.sew != 32) return error.IllegalInstruction;
                switch (funct6) {
                    0x00, 0x02, 0x24, 0x2C, 0x09 => {
                        try self.group(vd, t.lmul);
                        try self.group(vs1, t.lmul);
                        try self.group(vs2, t.lmul);
                        while (i < vl) : (i += 1) {
                            if (!self.active(vm, i)) continue;
                            const x = self.eptr(u32, vs2, i).*;
                            const y = self.eptr(u32, vs1, i).*;
                            const d = self.eptr(u32, vd, i);
                            d.* = switch (funct6) {
                                0x00 => canon(asF(x) + asF(y)),
                                0x02 => canon(asF(x) - asF(y)),
                                0x24 => canon(asF(x) * asF(y)),
                                0x2C => canon(@mulAdd(f32, asF(y), asF(x), asF(d.*))),
                                else => (x & 0x7FFF_FFFF) | (~y & 0x8000_0000),
                            };
                        }
                        self.finish(u32, vd, t, t.lmul, vm);
                    },
                    0x12 => { // VFUNARY0
                        try self.group(vd, t.lmul);
                        try self.group(vs2, t.lmul);
                        while (i < vl) : (i += 1) {
                            if (!self.active(vm, i)) continue;
                            const x = self.eptr(u32, vs2, i).*;
                            const r: f32 = switch (vs1) {
                                2 => @floatFromInt(x),
                                3 => @floatFromInt(@as(i32, @bitCast(x))),
                                else => return error.IllegalInstruction,
                            };
                            self.eptr(u32, vd, i).* = @bitCast(r);
                        }
                        self.finish(u32, vd, t, t.lmul, vm);
                    },
                    0x10 => { // vfmv.f.s
                        if (vs1 != 0 or !vm) return error.IllegalInstruction;
                        self.f[vd] = box(self.eptr(u32, vs2, 0).*);
                    },
                    0x1B, 0x19 => { // vmflt.vv, vmfle.vv
                        try self.group(vs1, t.lmul);
                        try self.group(vs2, t.lmul);
                        while (i < vl) : (i += 1) {
                            if (!self.active(vm, i)) continue;
                            const x = asF(self.eptr(u32, vs2, i).*);
                            const y = asF(self.eptr(u32, vs1, i).*);
                            self.setmbit(vd, i, if (funct6 == 0x1B) x < y else x <= y);
                        }
                        self.maskTail(vd);
                    },
                    else => return error.IllegalInstruction,
                }
            },
            // OPMVV
            2 => switch (funct6) {
                0x00 => { // vredsum.vs
                    try self.group(vs2, t.lmul);
                    if (vl == 0) return;
                    switch (t.sew) {
                        8 => try self.redsum(u8, t, vd, vs2, vs1, vm),
                        16 => try self.redsum(u16, t, vd, vs2, vs1, vm),
                        32 => try self.redsum(u32, t, vd, vs2, vs1, vm),
                        else => try self.redsum(u64, t, vd, vs2, vs1, vm),
                    }
                },
                0x10 => { // vmv.x.s
                    if (vs1 != 0 or !vm) return error.IllegalInstruction;
                    self.x[vd] = switch (t.sew) {
                        8 => sext(self.eptr(u8, vs2, 0).*, 8),
                        16 => sext(self.eptr(u16, vs2, 0).*, 16),
                        32 => sext(self.eptr(u32, vs2, 0).*, 32),
                        else => self.eptr(u64, vs2, 0).*,
                    };
                },
                0x12 => { // vzext/vsext .vf2/.vf4
                    const frac: u32 = switch (vs1) {
                        4, 5 => 4,
                        6, 7 => 2,
                        else => return error.IllegalInstruction,
                    };
                    const signed = vs1 & 1 == 1;
                    if (t.sew / frac < 8) return error.IllegalInstruction;
                    try self.group(vd, t.lmul);
                    try self.group(vs2, @max(1, t.lmul / frac));
                    if (overlaps(vd, t.lmul, vs2, @max(1, t.lmul / frac))) return error.IllegalInstruction;
                    while (i < vl) : (i += 1) {
                        if (!self.active(vm, i)) continue;
                        const src: u64 = switch (t.sew / frac) {
                            8 => if (signed) sext(self.eptr(u8, vs2, i).*, 8) else self.eptr(u8, vs2, i).*,
                            16 => if (signed) sext(self.eptr(u16, vs2, i).*, 16) else self.eptr(u16, vs2, i).*,
                            else => if (signed) sext(self.eptr(u32, vs2, i).*, 32) else self.eptr(u32, vs2, i).*,
                        };
                        switch (t.sew) {
                            16 => self.eptr(u16, vd, i).* = @truncate(src),
                            32 => self.eptr(u32, vd, i).* = @truncate(src),
                            else => self.eptr(u64, vd, i).* = src,
                        }
                    }
                    switch (t.sew) {
                        16 => self.finish(u16, vd, t, t.lmul, vm),
                        32 => self.finish(u32, vd, t, t.lmul, vm),
                        else => self.finish(u64, vd, t, t.lmul, vm),
                    }
                },
                0x1D => { // vmnand.mm
                    while (i < vl) : (i += 1) self.setmbit(vd, i, !(self.mbit(vs2, i) and self.mbit(vs1, i)));
                    if (self.poison) {
                        i = vl;
                        while (i < self.vlenb * 8) : (i += 1) self.setmbit(vd, i, true);
                    }
                },
                0x3B, 0x35 => { // vwmul.vv (signed), vwadd.wv
                    if (t.sew > 32 or t.lmul > 4) return error.IllegalInstruction;
                    const wide = 2 * t.lmul;
                    try self.group(vd, wide);
                    try self.group(vs1, t.lmul);
                    if (funct6 == 0x3B) {
                        try self.group(vs2, t.lmul);
                        if (overlaps(vd, wide, vs2, t.lmul) or overlaps(vd, wide, vs1, t.lmul)) return error.IllegalInstruction;
                    } else {
                        try self.group(vs2, wide);
                        if (overlaps(vd, wide, vs1, t.lmul)) return error.IllegalInstruction;
                    }
                    switch (t.sew) {
                        8 => try self.widen(u8, u16, i8, i16, t, funct6, vd, vs2, vs1, vm),
                        16 => try self.widen(u16, u32, i16, i32, t, funct6, vd, vs2, vs1, vm),
                        else => try self.widen(u32, u64, i32, i64, t, funct6, vd, vs2, vs1, vm),
                    }
                },
                else => return error.IllegalInstruction,
            },
            // OPIVI
            3 => {
                const imm5: u64 = sext(@as(u64, rs1), 5);
                switch (funct6) {
                    0x17, 0x09, 0x28, 0x00, 0x25 => {
                        try self.group(vd, t.lmul);
                        if (!(funct6 == 0x17 and vm)) try self.group(vs2, t.lmul);
                        switch (t.sew) {
                            8 => self.lanesImm(u8, t, funct6, vd, vs2, imm5, rs1, vm),
                            16 => self.lanesImm(u16, t, funct6, vd, vs2, imm5, rs1, vm),
                            32 => self.lanesImm(u32, t, funct6, vd, vs2, imm5, rs1, vm),
                            else => self.lanesImm(u64, t, funct6, vd, vs2, imm5, rs1, vm),
                        }
                    },
                    0x0F => { // vslidedown.vi
                        try self.group(vd, t.lmul);
                        try self.group(vs2, t.lmul);
                        const max = self.vlmax(t);
                        switch (t.sew) {
                            8 => self.slidedown(u8, t, vd, vs2, rs1, max, vm),
                            16 => self.slidedown(u16, t, vd, vs2, rs1, max, vm),
                            32 => self.slidedown(u32, t, vd, vs2, rs1, max, vm),
                            else => self.slidedown(u64, t, vd, vs2, rs1, max, vm),
                        }
                    },
                    else => return error.IllegalInstruction,
                }
            },
            // OPFVF
            5 => {
                if (t.sew != 32) return error.IllegalInstruction;
                const fs = unboxS(self.f[rs1]);
                const fbits: u32 = @bitCast(fs);
                switch (funct6) {
                    0x24, 0x00, 0x02, 0x27, 0x2C => { // vfmul/vfadd/vfsub/vfrsub/vfmacc .vf
                        try self.group(vd, t.lmul);
                        try self.group(vs2, t.lmul);
                        while (i < vl) : (i += 1) {
                            if (!self.active(vm, i)) continue;
                            const x = asF(self.eptr(u32, vs2, i).*);
                            const d = self.eptr(u32, vd, i);
                            d.* = canon(switch (funct6) {
                                0x24 => x * fs,
                                0x00 => x + fs,
                                0x02 => x - fs,
                                0x27 => fs - x,
                                else => @mulAdd(f32, fs, x, asF(d.*)),
                            });
                        }
                        self.finish(u32, vd, t, t.lmul, vm);
                    },
                    0x17 => { // vfmv.v.f (vm=1) / vfmerge.vfm (vm=0)
                        try self.group(vd, t.lmul);
                        if (vm and vs2 != 0) return error.IllegalInstruction;
                        if (!vm) try self.group(vs2, t.lmul);
                        while (i < vl) : (i += 1) {
                            self.eptr(u32, vd, i).* = if (vm or self.mbit(0, i)) fbits else self.eptr(u32, vs2, i).*;
                        }
                        self.finish(u32, vd, t, t.lmul, true);
                    },
                    0x1B, 0x19, 0x1F => { // vmflt.vf, vmfle.vf, vmfge.vf
                        try self.group(vs2, t.lmul);
                        while (i < vl) : (i += 1) {
                            if (!self.active(vm, i)) continue;
                            const x = asF(self.eptr(u32, vs2, i).*);
                            self.setmbit(vd, i, switch (funct6) {
                                0x1B => x < fs,
                                0x19 => x <= fs,
                                else => x >= fs,
                            });
                        }
                        self.maskTail(vd);
                    },
                    0x1D => { // vmfgt.vf
                        try self.group(vs2, t.lmul);
                        while (i < vl) : (i += 1) {
                            if (!self.active(vm, i)) continue;
                            self.setmbit(vd, i, asF(self.eptr(u32, vs2, i).*) > fs);
                        }
                        if (self.poison) {
                            i = vl;
                            while (i < self.vlenb * 8) : (i += 1) self.setmbit(vd, i, true);
                        }
                    },
                    0x10 => { // vfmv.s.f
                        if (vl > 0) self.eptr(u32, vd, 0).* = fbits;
                    },
                    else => return error.IllegalInstruction,
                }
            },
            // OPMVX
            6 => switch (funct6) {
                0x10 => { // vmv.s.x
                    if (vs2 != 0 or !vm) return error.IllegalInstruction;
                    if (vl > 0) switch (t.sew) {
                        8 => self.eptr(u8, vd, 0).* = @truncate(self.x[rs1]),
                        16 => self.eptr(u16, vd, 0).* = @truncate(self.x[rs1]),
                        32 => self.eptr(u32, vd, 0).* = @truncate(self.x[rs1]),
                        else => self.eptr(u64, vd, 0).* = self.x[rs1],
                    };
                },
                else => return error.IllegalInstruction,
            },
            else => return error.IllegalInstruction,
        }
    }

    /// Mask destinations are always tail-agnostic (RVV 1.0 §5.3): poison them.
    fn maskTail(self: *Hart, vd: u5) void {
        if (!self.poison) return;
        var i: u64 = self.vl;
        while (i < self.vlenb * 8) : (i += 1) self.setmbit(vd, i, true);
    }

    const LaneX = enum { add, sub, rsub, @"and", xor, mv };

    fn lanesX(self: *Hart, comptime T: type, t: VType, vd: u5, vs2: u5, x: T, vm: bool, kind: LaneX) void {
        var i: u64 = 0;
        while (i < self.vl) : (i += 1) {
            if (!self.active(vm, i)) continue;
            const d = self.eptr(T, vd, i);
            const a = if (kind == .mv) 0 else self.eptr(T, vs2, i).*;
            d.* = switch (kind) {
                .add => a +% x,
                .sub => a -% x,
                .rsub => x -% a,
                .@"and" => a & x,
                .xor => a ^ x,
                .mv => x,
            };
        }
        self.finish(T, vd, t, t.lmul, vm);
    }

    const Lane2 = enum { add, sub, @"and", xor, merge };

    fn lanes2(self: *Hart, t: VType, vd: u5, vs2: u5, vs1: u5, vm: bool, kind: Lane2) Fault!void {
        switch (t.sew) {
            8 => self.lanes2T(u8, t, vd, vs2, vs1, vm, kind),
            16 => self.lanes2T(u16, t, vd, vs2, vs1, vm, kind),
            32 => self.lanes2T(u32, t, vd, vs2, vs1, vm, kind),
            else => self.lanes2T(u64, t, vd, vs2, vs1, vm, kind),
        }
    }

    fn lanes2T(self: *Hart, comptime T: type, t: VType, vd: u5, vs2: u5, vs1: u5, vm: bool, kind: Lane2) void {
        var i: u64 = 0;
        while (i < self.vl) : (i += 1) {
            const d = self.eptr(T, vd, i);
            switch (kind) {
                .merge => d.* = if (vm or self.mbit(0, i)) self.eptr(T, vs1, i).* else self.eptr(T, vs2, i).*,
                else => if (self.active(vm, i)) {
                    const a = self.eptr(T, vs2, i).*;
                    const b = self.eptr(T, vs1, i).*;
                    d.* = switch (kind) {
                        .add => a +% b,
                        .sub => a -% b,
                        .@"and" => a & b,
                        .xor => a ^ b,
                        .merge => unreachable,
                    };
                },
            }
        }
        self.finish(T, vd, t, t.lmul, if (kind == .merge) true else vm);
    }

    fn lanesImm(self: *Hart, comptime T: type, t: VType, funct6: u32, vd: u5, vs2: u5, imm: u64, uimm: u5, vm: bool) void {
        const bits = @bitSizeOf(T);
        const sh: std.math.Log2Int(T) = @truncate(@as(u64, uimm) & (bits - 1));
        var i: u64 = 0;
        while (i < self.vl) : (i += 1) {
            const d = self.eptr(T, vd, i);
            switch (funct6) {
                0x17 => d.* = if (vm or self.mbit(0, i)) @truncate(imm) else self.eptr(T, vs2, i).*,
                else => if (self.active(vm, i)) {
                    const x = self.eptr(T, vs2, i).*;
                    d.* = switch (funct6) {
                        0x09 => x & @as(T, @truncate(imm)),
                        0x28 => x >> sh,
                        0x25 => x << sh,
                        else => x +% @as(T, @truncate(imm)),
                    };
                },
            }
        }
        self.finish(T, vd, t, t.lmul, if (funct6 == 0x17) true else vm);
    }

    fn slidedown(self: *Hart, comptime T: type, t: VType, vd: u5, vs2: u5, off: u64, max: u64, vm: bool) void {
        var i: u64 = 0;
        while (i < self.vl) : (i += 1) {
            if (!self.active(vm, i)) continue;
            self.eptr(T, vd, i).* = if (i + off < max) self.eptr(T, vs2, i + off).* else 0;
        }
        self.finish(T, vd, t, t.lmul, vm);
    }

    fn redsum(self: *Hart, comptime T: type, t: VType, vd: u5, vs2: u5, vs1: u5, vm: bool) Fault!void {
        var acc: T = self.eptr(T, vs1, 0).*;
        var i: u64 = 0;
        while (i < self.vl) : (i += 1) if (self.active(vm, i)) {
            acc +%= self.eptr(T, vs2, i).*;
        };
        self.eptr(T, vd, 0).* = acc;
        if (self.poison and t.ta) {
            var k: u64 = 1;
            while (k < self.vlenb / @sizeOf(T)) : (k += 1) self.eptr(T, vd, k).* = ~@as(T, 0);
        }
    }

    fn widen(self: *Hart, comptime N: type, comptime W: type, comptime SN: type, comptime SW: type, t: VType, funct6: u32, vd: u5, vs2: u5, vs1: u5, vm: bool) Fault!void {
        var i: u64 = 0;
        while (i < self.vl) : (i += 1) {
            if (!self.active(vm, i)) continue;
            const b: SW = @as(SN, @bitCast(self.eptr(N, vs1, i).*));
            const d = self.eptr(W, vd, i);
            if (funct6 == 0x3B) {
                const a: SW = @as(SN, @bitCast(self.eptr(N, vs2, i).*));
                d.* = @bitCast(a *% b);
            } else {
                const a: SW = @bitCast(self.eptr(W, vs2, i).*);
                d.* = @bitCast(a +% b);
            }
        }
        self.finish(W, vd, t, 2 * t.lmul, vm);
    }
};

fn overlaps(a: u5, na: u32, b: u5, nb: u32) bool {
    return @as(u32, a) < @as(u32, b) + nb and @as(u32, b) < @as(u32, a) + na;
}

fn rneOk(rm: u3) bool {
    return rm == 0 or rm == 7; // static RNE, or dynamic with frm = RNE (reset value)
}

fn sext(v: u64, comptime bits: u7) u64 {
    const sh: u6 = @intCast(64 - @as(u7, bits));
    return @bitCast(@as(i64, @bitCast(v << sh)) >> sh);
}

fn immI(w: u32) u64 {
    return sext(w >> 20, 12);
}

fn immS(w: u32) u64 {
    return sext(((w >> 25) << 5) | ((w >> 7) & 31), 12);
}

fn immB(w: u32) u64 {
    const v = ((w >> 31) << 12) | (((w >> 7) & 1) << 11) | (((w >> 25) & 0x3F) << 5) | (((w >> 8) & 0xF) << 1);
    return sext(v, 13);
}

fn immJ(w: u32) u64 {
    const v = ((w >> 31) << 20) | (((w >> 12) & 0xFF) << 12) | (((w >> 20) & 1) << 11) | (((w >> 21) & 0x3FF) << 1);
    return sext(v, 21);
}

fn box(bits: u32) u64 {
    return @as(u64, 0xFFFF_FFFF_0000_0000) | bits;
}

/// Unbox a single-precision value; improperly boxed inputs read as the canonical NaN.
fn unboxS(reg: u64) f32 {
    const bits: u32 = if (reg >> 32 == 0xFFFF_FFFF) @truncate(reg) else canonical_nan;
    return @bitCast(bits);
}

fn asF(bits: u32) f32 {
    return @bitCast(bits);
}

fn canon(x: f32) u32 {
    return if (std.math.isNan(x)) canonical_nan else @bitCast(x);
}

test "immediates round-trip" {
    // addi a0, a0, -1  = 0xfff50513
    try std.testing.expectEqual(@as(u64, @bitCast(@as(i64, -1))), immI(0xfff50513));
    // beq x0, x0, +8 = 0x00000463
    try std.testing.expectEqual(@as(u64, 8), immB(0x00000463));
    // jal x0, -4 = 0xffdff06f
    try std.testing.expectEqual(@as(u64, @bitCast(@as(i64, -4))), immJ(0xffdff06f));
}
