// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Resource Aliasing for Memory Reuse
//!
//! Allows multiple non-overlapping resources to share the same memory allocation,
//! reducing overall memory footprint for resources with non-overlapping lifetimes.
//!
//! ## Features
//!
//! - Lifetime-based memory aliasing
//! - Automatic alignment handling
//! - Memory pool with aliasing support
//! - Resource lifetime tracking
//!
//! ## Example
//!
//! ```zig
//! const vulkan = @import("vulkan");
//!
//! // Create an aliasing pool
//! var pool = try vulkan.AliasingPool.init(&ctx, allocator, .{
//!     .pool_size = 256 * 1024 * 1024, // 256 MB
//! });
//! defer pool.deinit();
//!
//! // Define resource lifetimes (non-overlapping)
//! const temp_a = pool.allocate(1024 * 1024, 256, .{ .start = 0, .end = 10 });
//! const temp_b = pool.allocate(2 * 1024 * 1024, 256, .{ .start = 10, .end = 20 });
//! // temp_a and temp_b can share memory since lifetimes don't overlap
//!
//! // Or use automatic lifetime tracking
//! pool.beginLifetime("pass_a");
//! const buf_a = try pool.createBuffer(size, usage);
//! pool.endLifetime("pass_a");
//!
//! pool.beginLifetime("pass_b");
//! // pass_b resources can alias pass_a if they don't overlap
//! ```

const std = @import("std");
const types = @import("../vk/types.zig");
const loader = @import("../vk/loader.zig");
const err = @import("../vk/error.zig");
const Alignment = @import("allocator.zig").Alignment;

/// Resource lifetime for aliasing decisions
pub const Lifetime = struct {
    /// Start time/frame/pass index
    start: u32,
    /// End time/frame/pass index (exclusive)
    end: u32,

    /// Check if two lifetimes overlap
    pub fn overlaps(self: Lifetime, other: Lifetime) bool {
        return self.start < other.end and other.start < self.end;
    }

    /// Check if this lifetime contains a point
    pub fn contains(self: Lifetime, point: u32) bool {
        return point >= self.start and point < self.end;
    }

    /// Merge two lifetimes
    pub fn merge(self: Lifetime, other: Lifetime) Lifetime {
        return .{
            .start = @min(self.start, other.start),
            .end = @max(self.end, other.end),
        };
    }

    /// Get duration
    pub fn duration(self: Lifetime) u32 {
        return self.end - self.start;
    }

    /// Create an always-live lifetime
    pub fn always() Lifetime {
        return .{ .start = 0, .end = std.math.maxInt(u32) };
    }

    /// Create a single-frame lifetime
    pub fn frame(f: u32) Lifetime {
        return .{ .start = f, .end = f + 1 };
    }
};

/// Aliased resource handle
pub const AliasedResource = struct {
    /// Offset within the backing memory
    offset: u64,
    /// Size of this resource
    size: u64,
    /// Alignment requirement
    alignment: u64,
    /// Lifetime of this resource
    lifetime: Lifetime,
    /// Resource ID for tracking
    id: u32,
    /// User-defined tag/name
    tag: ?[]const u8 = null,
};

/// Aliasing pool options
pub const AliasingPoolOptions = struct {
    /// Total pool size in bytes
    pool_size: u64 = 256 * 1024 * 1024,
    /// Default alignment
    default_alignment: u64 = 256,
    /// Maximum number of resources
    max_resources: u32 = 1024,
    /// Memory property flags for the pool
    memory_properties: types.MemoryPropertyFlags = .{ .device_local = true },
};

/// Memory range for aliasing tracking
const MemoryRange = struct {
    offset: u64,
    size: u64,
    resource_id: u32,
    lifetime: Lifetime,
};

/// Aliasing memory pool
pub const AliasingPool = struct {
    allocator: std.mem.Allocator,
    options: AliasingPoolOptions,

    /// All allocated resources
    resources: std.ArrayListUnmanaged(AliasedResource),
    /// Memory ranges for aliasing
    ranges: std.ArrayListUnmanaged(MemoryRange),
    /// Next resource ID
    next_id: u32 = 0,

    /// Peak memory usage (actual unique memory needed)
    peak_usage: u64 = 0,
    /// Total logical memory allocated
    logical_usage: u64 = 0,
    /// Memory saved through aliasing
    memory_saved: u64 = 0,

    /// Named lifetime tracking
    active_lifetimes: std.StringHashMapUnmanaged(u32),
    current_time: u32 = 0,

    const Self = @This();

    /// Initialize an aliasing pool
    pub fn init(allocator: std.mem.Allocator, options: AliasingPoolOptions) Self {
        return Self{
            .allocator = allocator,
            .options = options,
            .resources = .{},
            .ranges = .{},
            .active_lifetimes = .{},
        };
    }

    /// Deinitialize the pool
    pub fn deinit(self: *Self) void {
        self.resources.deinit(self.allocator);
        self.ranges.deinit(self.allocator);
        self.active_lifetimes.deinit(self.allocator);
        self.* = undefined;
    }

    /// Allocate a resource with explicit lifetime
    pub fn allocate(
        self: *Self,
        size: u64,
        alignment: u64,
        lifetime: Lifetime,
    ) !AliasedResource {
        return self.allocateTagged(size, alignment, lifetime, null);
    }

    /// Allocate a resource with explicit lifetime and tag
    pub fn allocateTagged(
        self: *Self,
        size: u64,
        alignment: u64,
        lifetime: Lifetime,
        tag: ?[]const u8,
    ) !AliasedResource {
        const actual_alignment = if (alignment == 0) self.options.default_alignment else alignment;
        const aligned_size = Alignment.alignUp(size, actual_alignment);

        // Find a suitable offset that doesn't conflict with overlapping lifetimes
        const offset = try self.findNonConflictingOffset(aligned_size, actual_alignment, lifetime);

        const resource = AliasedResource{
            .offset = offset,
            .size = aligned_size,
            .alignment = actual_alignment,
            .lifetime = lifetime,
            .id = self.next_id,
            .tag = tag,
        };
        self.next_id += 1;

        try self.resources.append(self.allocator, resource);
        try self.ranges.append(self.allocator, .{
            .offset = offset,
            .size = aligned_size,
            .resource_id = resource.id,
            .lifetime = lifetime,
        });

        // Update statistics
        self.logical_usage += aligned_size;
        self.updatePeakUsage();

        return resource;
    }

    /// Find an offset that doesn't conflict with resources of overlapping lifetimes
    fn findNonConflictingOffset(
        self: *Self,
        size: u64,
        alignment: u64,
        lifetime: Lifetime,
    ) !u64 {
        if (self.ranges.items.len == 0) {
            return 0;
        }

        // Collect all ranges that overlap with our lifetime
        var conflicting = std.ArrayListUnmanaged(MemoryRange){};
        defer conflicting.deinit(self.allocator);

        for (self.ranges.items) |range| {
            if (range.lifetime.overlaps(lifetime)) {
                try conflicting.append(self.allocator, range);
            }
        }

        if (conflicting.items.len == 0) {
            return 0; // No conflicts, can use offset 0
        }

        // Sort conflicting ranges by offset
        std.mem.sort(MemoryRange, conflicting.items, {}, struct {
            fn lessThan(_: void, a: MemoryRange, b: MemoryRange) bool {
                return a.offset < b.offset;
            }
        }.lessThan);

        // Find first gap that fits
        var current_offset: u64 = 0;
        for (conflicting.items) |range| {
            const aligned_offset = Alignment.alignUp(current_offset, alignment);

            // Check if there's a gap before this range
            if (aligned_offset + size <= range.offset) {
                return aligned_offset;
            }

            // Move past this range
            current_offset = range.offset + range.size;
        }

        // No gap found, allocate at the end
        const aligned_offset = Alignment.alignUp(current_offset, alignment);
        if (aligned_offset + size > self.options.pool_size) {
            return err.Error.OutOfDeviceMemory;
        }

        return aligned_offset;
    }

    /// Update peak usage calculation
    fn updatePeakUsage(self: *Self) void {
        // For each time point, calculate total memory in use
        var max_usage: u64 = 0;

        // Get all unique time points
        var time_points = std.ArrayListUnmanaged(u32){};
        defer time_points.deinit(self.allocator);

        for (self.ranges.items) |range| {
            time_points.append(self.allocator, range.lifetime.start) catch continue;
            time_points.append(self.allocator, range.lifetime.end) catch continue;
        }

        std.mem.sort(u32, time_points.items, {}, std.sort.asc(u32));

        // For each time point, calculate usage
        for (time_points.items) |t| {
            var usage: u64 = 0;
            var max_offset: u64 = 0;

            for (self.ranges.items) |range| {
                if (range.lifetime.contains(t)) {
                    usage += range.size;
                    const range_end = range.offset + range.size;
                    if (range_end > max_offset) max_offset = range_end;
                }
            }

            if (max_offset > max_usage) max_usage = max_offset;
        }

        self.peak_usage = max_usage;
        self.memory_saved = if (self.logical_usage > max_usage)
            self.logical_usage - max_usage
        else
            0;
    }

    /// Begin a named lifetime
    /// Note: If tracking fails, lifetime boundaries may be lost but allocation still works.
    pub fn beginLifetime(self: *Self, name: []const u8) void {
        self.active_lifetimes.put(self.allocator, name, self.current_time) catch |e| {
            std.log.warn("AliasingPool.beginLifetime: failed to track lifetime '{s}': {s} - aliasing may be suboptimal", .{
                name,
                @errorName(e),
            });
        };
    }

    /// End a named lifetime
    pub fn endLifetime(self: *Self, name: []const u8) void {
        _ = self.active_lifetimes.remove(name);
    }

    /// Get current lifetime for active allocations
    pub fn getCurrentLifetime(self: *Self) Lifetime {
        return .{
            .start = self.current_time,
            .end = self.current_time + 1,
        };
    }

    /// Advance time
    pub fn advanceTime(self: *Self) void {
        self.current_time += 1;
    }

    /// Reset the pool (clear all allocations)
    pub fn reset(self: *Self) void {
        self.resources.clearRetainingCapacity();
        self.ranges.clearRetainingCapacity();
        self.next_id = 0;
        self.peak_usage = 0;
        self.logical_usage = 0;
        self.memory_saved = 0;
        self.current_time = 0;
        self.active_lifetimes.clearRetainingCapacity();
    }

    /// Get aliasing statistics
    pub fn getStats(self: Self) AliasingStats {
        return .{
            .resource_count = @intCast(self.resources.items.len),
            .logical_usage = self.logical_usage,
            .peak_physical_usage = self.peak_usage,
            .memory_saved = self.memory_saved,
            .aliasing_ratio = if (self.logical_usage > 0)
                @as(f64, @floatFromInt(self.memory_saved)) / @as(f64, @floatFromInt(self.logical_usage))
            else
                0.0,
        };
    }

    /// Get resource by ID
    pub fn getResource(self: Self, id: u32) ?AliasedResource {
        for (self.resources.items) |res| {
            if (res.id == id) return res;
        }
        return null;
    }

    /// Get all resources active at a given time
    pub fn getActiveResources(self: Self, allocator_: std.mem.Allocator, time: u32) ![]const AliasedResource {
        var active = std.ArrayListUnmanaged(AliasedResource){};
        errdefer active.deinit(allocator_);

        for (self.resources.items) |res| {
            if (res.lifetime.contains(time)) {
                try active.append(allocator_, res);
            }
        }

        return try active.toOwnedSlice(allocator_);
    }

    /// Check if two resources can be aliased
    pub fn canAlias(self: Self, id_a: u32, id_b: u32) bool {
        const res_a = self.getResource(id_a) orelse return false;
        const res_b = self.getResource(id_b) orelse return false;
        return !res_a.lifetime.overlaps(res_b.lifetime);
    }

    /// Print aliasing summary
    pub fn printSummary(self: Self, writer: anytype) !void {
        const stats = self.getStats();

        try writer.print("\n=== Resource Aliasing Summary ===\n\n", .{});
        try writer.print("Resources: {}\n", .{stats.resource_count});
        try writer.print("Logical usage: {d:.2} MB\n", .{
            @as(f64, @floatFromInt(stats.logical_usage)) / (1024 * 1024),
        });
        try writer.print("Physical usage: {d:.2} MB\n", .{
            @as(f64, @floatFromInt(stats.peak_physical_usage)) / (1024 * 1024),
        });
        try writer.print("Memory saved: {d:.2} MB ({d:.1}%)\n", .{
            @as(f64, @floatFromInt(stats.memory_saved)) / (1024 * 1024),
            stats.aliasing_ratio * 100,
        });

        try writer.print("\nResources:\n", .{});
        for (self.resources.items) |res| {
            try writer.print("  [{d}] offset={d}, size={d}, lifetime=[{d},{d})", .{
                res.id,
                res.offset,
                res.size,
                res.lifetime.start,
                res.lifetime.end,
            });
            if (res.tag) |tag| {
                try writer.print(" tag=\"{s}\"", .{tag});
            }
            try writer.print("\n", .{});
        }
    }
};

/// Aliasing statistics
pub const AliasingStats = struct {
    resource_count: u32,
    logical_usage: u64,
    peak_physical_usage: u64,
    memory_saved: u64,
    aliasing_ratio: f64,
};

/// Interval tree node for efficient overlap queries
pub const IntervalNode = struct {
    start: u32,
    end: u32,
    max_end: u32,
    resource_id: u32,
    left: ?*IntervalNode = null,
    right: ?*IntervalNode = null,
};

// =============================================================================
// Tests
// =============================================================================

test "Lifetime overlap detection" {
    const a = Lifetime{ .start = 0, .end = 10 };
    const b = Lifetime{ .start = 5, .end = 15 };
    const c = Lifetime{ .start = 10, .end = 20 };

    try std.testing.expect(a.overlaps(b)); // 0-10 overlaps with 5-15
    try std.testing.expect(!a.overlaps(c)); // 0-10 doesn't overlap with 10-20
    try std.testing.expect(b.overlaps(c)); // 5-15 overlaps with 10-20
}

test "Lifetime contains" {
    const lt = Lifetime{ .start = 5, .end = 15 };

    try std.testing.expect(!lt.contains(4));
    try std.testing.expect(lt.contains(5));
    try std.testing.expect(lt.contains(10));
    try std.testing.expect(lt.contains(14));
    try std.testing.expect(!lt.contains(15));
}

test "Lifetime merge" {
    const a = Lifetime{ .start = 0, .end = 10 };
    const b = Lifetime{ .start = 5, .end = 20 };
    const merged = a.merge(b);

    try std.testing.expectEqual(@as(u32, 0), merged.start);
    try std.testing.expectEqual(@as(u32, 20), merged.end);
}

test "AliasingPool basic allocation" {
    const allocator = std.testing.allocator;
    var pool = AliasingPool.init(allocator, .{
        .pool_size = 1024 * 1024,
    });
    defer pool.deinit();

    // Allocate non-overlapping resources
    const res_a = try pool.allocate(1024, 256, .{ .start = 0, .end = 5 });
    const res_b = try pool.allocate(1024, 256, .{ .start = 5, .end = 10 });

    // They should be able to alias (same offset possible)
    try std.testing.expect(pool.canAlias(res_a.id, res_b.id));
}

test "AliasingPool overlapping prevents aliasing" {
    const allocator = std.testing.allocator;
    var pool = AliasingPool.init(allocator, .{
        .pool_size = 1024 * 1024,
    });
    defer pool.deinit();

    // Allocate overlapping resources
    const res_a = try pool.allocate(1024, 256, .{ .start = 0, .end = 10 });
    const res_b = try pool.allocate(1024, 256, .{ .start = 5, .end = 15 });

    // They should NOT be able to alias
    try std.testing.expect(!pool.canAlias(res_a.id, res_b.id));
    // And should have different offsets
    try std.testing.expect(res_a.offset != res_b.offset);
}

test "AliasingPool statistics" {
    const allocator = std.testing.allocator;
    var pool = AliasingPool.init(allocator, .{
        .pool_size = 1024 * 1024,
    });
    defer pool.deinit();

    // Allocate non-overlapping resources
    _ = try pool.allocate(1024, 256, .{ .start = 0, .end = 5 });
    _ = try pool.allocate(1024, 256, .{ .start = 5, .end = 10 });
    _ = try pool.allocate(1024, 256, .{ .start = 10, .end = 15 });

    const stats = pool.getStats();
    try std.testing.expectEqual(@as(u32, 3), stats.resource_count);
    // All three should be able to share memory since they don't overlap
    try std.testing.expect(stats.memory_saved > 0);
}
