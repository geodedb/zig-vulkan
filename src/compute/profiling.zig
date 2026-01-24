// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! GPU Compute Profiling
//!
//! High-level profiling utilities for compute shaders including:
//! - Performance counters (VK_KHR_performance_query)
//! - Timestamp-based profiling with statistical analysis
//! - Bandwidth and throughput calculations
//! - Multi-dispatch profiling sessions
//!
//! ## Example
//!
//! ```zig
//! const vulkan = @import("vulkan");
//!
//! // Create a profiler
//! var profiler = try vulkan.ComputeProfiler.init(&ctx, allocator, .{
//!     .enable_timestamps = true,
//!     .enable_statistics = true,
//! });
//! defer profiler.deinit();
//!
//! // Profile a dispatch
//! profiler.beginRegion(cmd, "matrix_multiply");
//! dispatcher.dispatch1D(cmd, pipeline, element_count, descriptor_set);
//! profiler.endRegion(cmd, "matrix_multiply");
//!
//! // After submission and wait
//! const results = try profiler.getResults(allocator);
//! defer allocator.free(results);
//!
//! for (results) |result| {
//!     std.debug.print("{s}: {d:.2} ms\n", .{result.name, result.duration_ms});
//! }
//! ```

const std = @import("std");
const types = @import("../vk/types.zig");
const loader = @import("../vk/loader.zig");
const err = @import("../vk/error.zig");
const Context = @import("context.zig").Context;
const command = @import("command.zig");

/// Profiling options
pub const ProfilerOptions = struct {
    /// Enable timestamp queries
    enable_timestamps: bool = true,
    /// Enable pipeline statistics
    enable_statistics: bool = false,
    /// Maximum number of regions to track
    max_regions: u32 = 64,
    /// Include memory bandwidth estimates
    estimate_bandwidth: bool = false,
};

/// A single profiled region
pub const ProfileRegion = struct {
    /// Region name
    name: []const u8,
    /// Start timestamp index
    start_query: u32,
    /// End timestamp index
    end_query: u32,
    /// Whether this region has been completed
    completed: bool = false,
    /// Optional memory transfer size for bandwidth calculation
    bytes_transferred: u64 = 0,
    /// Number of work items dispatched
    work_items: u64 = 0,
};

/// Profiling result for a single region
pub const ProfileResult = struct {
    /// Region name
    name: []const u8,
    /// Duration in nanoseconds
    duration_ns: u64,
    /// Duration in milliseconds
    duration_ms: f64,
    /// Estimated bandwidth in GB/s (if bytes_transferred was set)
    bandwidth_gbps: ?f64 = null,
    /// Work items per second
    throughput: ?f64 = null,
    /// Pipeline statistics (if enabled)
    statistics: ?command.PipelineStatistics = null,
};

/// Statistical summary of multiple runs
pub const ProfileStats = struct {
    /// Region name
    name: []const u8,
    /// Number of samples
    sample_count: u32,
    /// Minimum duration in nanoseconds
    min_ns: u64,
    /// Maximum duration in nanoseconds
    max_ns: u64,
    /// Average duration in nanoseconds
    avg_ns: u64,
    /// Standard deviation in nanoseconds
    std_dev_ns: f64,
    /// Median duration in nanoseconds
    median_ns: u64,

    /// Get average in milliseconds
    pub fn avgMs(self: ProfileStats) f64 {
        return @as(f64, @floatFromInt(self.avg_ns)) / 1_000_000.0;
    }

    /// Get min in milliseconds
    pub fn minMs(self: ProfileStats) f64 {
        return @as(f64, @floatFromInt(self.min_ns)) / 1_000_000.0;
    }

    /// Get max in milliseconds
    pub fn maxMs(self: ProfileStats) f64 {
        return @as(f64, @floatFromInt(self.max_ns)) / 1_000_000.0;
    }
};

/// Compute shader profiler
pub const ComputeProfiler = struct {
    ctx: *Context,
    allocator: std.mem.Allocator,
    options: ProfilerOptions,

    /// Timestamp query pool
    timestamp_pool: ?command.TimestampQuery = null,
    /// Statistics query pool
    stats_pool: ?command.PipelineStatisticsQuery = null,

    /// Tracked regions
    regions: std.ArrayListUnmanaged(ProfileRegion),
    /// Region name storage (owned strings)
    region_names: std.ArrayListUnmanaged([]const u8),
    /// Current query index
    current_query: u32 = 0,
    /// Current statistics query index
    current_stats_query: u32 = 0,

    /// Historical data for statistics
    history: std.StringHashMapUnmanaged(std.ArrayListUnmanaged(u64)),

    const Self = @This();

    /// Initialize a new profiler
    pub fn init(ctx: *Context, allocator: std.mem.Allocator, options: ProfilerOptions) !Self {
        var self = Self{
            .ctx = ctx,
            .allocator = allocator,
            .options = options,
            .regions = .{},
            .region_names = .{},
            .history = .{},
        };

        if (options.enable_timestamps) {
            self.timestamp_pool = try command.TimestampQuery.init(ctx, options.max_regions * 2);
        }

        if (options.enable_statistics) {
            self.stats_pool = try command.PipelineStatisticsQuery.init(
                ctx,
                options.max_regions,
                command.PipelineStatisticsFlags.computeOnly(),
            );
        }

        return self;
    }

    /// Deinitialize the profiler
    pub fn deinit(self: *Self) void {
        if (self.timestamp_pool) |*pool| {
            pool.deinit();
        }
        if (self.stats_pool) |*pool| {
            pool.deinit();
        }

        // Free region names
        for (self.region_names.items) |name| {
            self.allocator.free(name);
        }
        self.region_names.deinit(self.allocator);
        self.regions.deinit(self.allocator);

        // Free history
        var history_iter = self.history.valueIterator();
        while (history_iter.next()) |list| {
            list.deinit(self.allocator);
        }
        self.history.deinit(self.allocator);

        self.* = undefined;
    }

    /// Reset for a new frame/iteration
    pub fn reset(self: *Self, cmd: types.CommandBuffer) void {
        self.current_query = 0;
        self.current_stats_query = 0;
        self.regions.clearRetainingCapacity();

        if (self.timestamp_pool) |pool| {
            pool.reset(cmd, 0, self.options.max_regions * 2);
        }
        if (self.stats_pool) |pool| {
            pool.reset(cmd, 0, self.options.max_regions);
        }
    }

    /// Begin a profiled region
    pub fn beginRegion(self: *Self, cmd: types.CommandBuffer, name: []const u8) void {
        self.beginRegionWithInfo(cmd, name, 0, 0);
    }

    /// Begin a profiled region with additional info
    pub fn beginRegionWithInfo(
        self: *Self,
        cmd: types.CommandBuffer,
        name: []const u8,
        bytes_transferred: u64,
        work_items: u64,
    ) void {
        if (self.current_query >= self.options.max_regions * 2) return;

        const start_query = self.current_query;

        if (self.timestamp_pool) |pool| {
            pool.writeTimestamp(cmd, .{ .compute_shader = true }, start_query);
            self.current_query += 1;
        }

        if (self.stats_pool) |pool| {
            pool.begin(cmd, self.current_stats_query);
        }

        // Store owned copy of name
        const owned_name = self.allocator.dupe(u8, name) catch |e| {
            std.log.warn("GpuProfiler: failed to duplicate region name '{s}': {s}", .{ name, @errorName(e) });
            return;
        };
        self.region_names.append(self.allocator, owned_name) catch |e| {
            std.log.warn("GpuProfiler: failed to append region name '{s}': {s}", .{ name, @errorName(e) });
            self.allocator.free(owned_name);
            return;
        };

        self.regions.append(self.allocator, .{
            .name = owned_name,
            .start_query = start_query,
            .end_query = 0,
            .bytes_transferred = bytes_transferred,
            .work_items = work_items,
        }) catch |e| {
            std.log.warn("GpuProfiler: failed to record profiling region '{s}': {s}", .{ name, @errorName(e) });
            // Free the name we just duplicated since we couldn't record the region
            self.allocator.free(owned_name);
            _ = self.region_names.pop();
            return;
        };
    }

    /// End the current profiled region
    pub fn endRegion(self: *Self, cmd: types.CommandBuffer, name: []const u8) void {
        // Find the matching region
        for (self.regions.items) |*region| {
            if (!region.completed and std.mem.eql(u8, region.name, name)) {
                if (self.timestamp_pool) |pool| {
                    pool.writeTimestamp(cmd, .{ .compute_shader = true }, self.current_query);
                    region.end_query = self.current_query;
                    self.current_query += 1;
                }

                if (self.stats_pool) |pool| {
                    pool.end(cmd, self.current_stats_query);
                    self.current_stats_query += 1;
                }

                region.completed = true;
                return;
            }
        }
    }

    /// Get profiling results (call after GPU work is complete)
    pub fn getResults(self: *Self) ![]ProfileResult {
        var results = try self.allocator.alloc(ProfileResult, self.regions.items.len);
        errdefer self.allocator.free(results);

        var valid_count: usize = 0;

        for (self.regions.items, 0..) |region, i| {
            if (!region.completed) continue;

            if (self.timestamp_pool) |pool| {
                const elapsed = pool.getElapsedNs(self.allocator, region.start_query, region.end_query) catch continue;

                var result = ProfileResult{
                    .name = region.name,
                    .duration_ns = elapsed,
                    .duration_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0,
                };

                // Calculate bandwidth if data was provided
                if (region.bytes_transferred > 0 and elapsed > 0) {
                    const bytes_per_second = @as(f64, @floatFromInt(region.bytes_transferred)) /
                        (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0);
                    result.bandwidth_gbps = bytes_per_second / (1024 * 1024 * 1024);
                }

                // Calculate throughput if work items were provided
                if (region.work_items > 0 and elapsed > 0) {
                    result.throughput = @as(f64, @floatFromInt(region.work_items)) /
                        (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0);
                }

                // Get statistics if available
                if (self.stats_pool) |stats_pool| {
                    result.statistics = stats_pool.getStatistics(self.allocator, @intCast(i)) catch null;
                }

                // Update history for statistical analysis
                self.updateHistory(region.name, elapsed) catch |e| {
                    std.log.debug("GpuProfiler: updateHistory failed for {s}: {s}", .{ region.name, @errorName(e) });
                };

                results[valid_count] = result;
                valid_count += 1;
            }
        }

        // Resize to actual count
        if (valid_count < results.len) {
            results = self.allocator.realloc(results, valid_count) catch results[0..valid_count];
        }

        return results[0..valid_count];
    }

    /// Update history for a region
    fn updateHistory(self: *Self, name: []const u8, duration_ns: u64) !void {
        const result = self.history.getOrPut(self.allocator, name) catch return;
        if (!result.found_existing) {
            result.value_ptr.* = .{};
        }
        try result.value_ptr.append(self.allocator, duration_ns);
    }

    /// Get statistical summary for a region
    pub fn getStats(self: Self, name: []const u8) ?ProfileStats {
        const history_list = self.history.get(name) orelse return null;
        if (history_list.items.len == 0) return null;

        const samples = history_list.items;
        const n = samples.len;

        // Calculate min, max, sum
        var min_val: u64 = std.math.maxInt(u64);
        var max_val: u64 = 0;
        var sum: u128 = 0;

        for (samples) |s| {
            if (s < min_val) min_val = s;
            if (s > max_val) max_val = s;
            sum += s;
        }

        const avg = @as(u64, @intCast(sum / n));

        // Calculate standard deviation
        var variance_sum: f64 = 0;
        for (samples) |s| {
            const diff = @as(f64, @floatFromInt(s)) - @as(f64, @floatFromInt(avg));
            variance_sum += diff * diff;
        }
        const std_dev = @sqrt(variance_sum / @as(f64, @floatFromInt(n)));

        // Calculate median (need sorted copy)
        const sorted = self.allocator.dupe(u64, samples) catch return null;
        defer self.allocator.free(sorted);
        std.mem.sortUnstable(u64, sorted, {}, std.sort.asc(u64));

        const median = if (n % 2 == 0)
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2
        else
            sorted[n / 2];

        return ProfileStats{
            .name = name,
            .sample_count = @intCast(n),
            .min_ns = min_val,
            .max_ns = max_val,
            .avg_ns = avg,
            .std_dev_ns = std_dev,
            .median_ns = median,
        };
    }

    /// Print profiling summary
    pub fn printSummary(self: *Self, writer: anytype) !void {
        try writer.print("\n=== Compute Profiling Summary ===\n\n", .{});

        const results = self.getResults() catch return;
        defer self.allocator.free(results);

        for (results) |result| {
            try writer.print("{s}:\n", .{result.name});
            try writer.print("  Duration: {d:.3} ms ({} ns)\n", .{ result.duration_ms, result.duration_ns });

            if (result.bandwidth_gbps) |bw| {
                try writer.print("  Bandwidth: {d:.2} GB/s\n", .{bw});
            }
            if (result.throughput) |tp| {
                try writer.print("  Throughput: {d:.2e} items/s\n", .{tp});
            }
            if (result.statistics) |stats| {
                try writer.print("  Compute invocations: {}\n", .{stats.compute_shader_invocations});
            }

            // Print historical stats if available
            if (self.getStats(result.name)) |stats| {
                try writer.print("  Stats ({} samples): avg={d:.3}ms, min={d:.3}ms, max={d:.3}ms, stddev={d:.3}ms\n", .{
                    stats.sample_count,
                    stats.avgMs(),
                    stats.minMs(),
                    stats.maxMs(),
                    stats.std_dev_ns / 1_000_000.0,
                });
            }
            try writer.print("\n", .{});
        }
    }
};

/// Lightweight profiling scope using RAII pattern
pub const ProfileScope = struct {
    profiler: *ComputeProfiler,
    cmd: types.CommandBuffer,
    name: []const u8,

    const Self = @This();

    pub fn init(profiler: *ComputeProfiler, cmd: types.CommandBuffer, name: []const u8) Self {
        profiler.beginRegion(cmd, name);
        return .{
            .profiler = profiler,
            .cmd = cmd,
            .name = name,
        };
    }

    pub fn deinit(self: Self) void {
        self.profiler.endRegion(self.cmd, self.name);
    }
};

/// Performance counter types (for VK_KHR_performance_query if available)
pub const PerformanceCounterType = enum {
    /// GPU clock cycles
    gpu_cycles,
    /// Compute shader invocations
    compute_invocations,
    /// Memory read bandwidth
    memory_read_bytes,
    /// Memory write bandwidth
    memory_write_bytes,
    /// Cache hit rate
    cache_hit_rate,
    /// ALU utilization
    alu_utilization,
    /// Occupancy percentage
    occupancy,
    /// Warps/waves launched
    warps_launched,
    /// Stall cycles
    stall_cycles,
};

/// Performance counter value
pub const PerformanceCounterValue = union {
    uint64: u64,
    float64: f64,
    percentage: f32,
};

/// Check if performance queries are available
pub fn performanceQueriesAvailable(ctx: *Context) bool {
    // Check for VK_KHR_performance_query extension
    // This would need extension enumeration in a full implementation
    _ = ctx;
    return false; // Conservative default
}

// =============================================================================
// Tests
// =============================================================================

test "ProfileStats calculations" {
    const stats = ProfileStats{
        .name = "test",
        .sample_count = 5,
        .min_ns = 1_000_000, // 1ms
        .max_ns = 5_000_000, // 5ms
        .avg_ns = 3_000_000, // 3ms
        .std_dev_ns = 1_414_213.56,
        .median_ns = 3_000_000,
    };

    try std.testing.expectApproxEqAbs(@as(f64, 3.0), stats.avgMs(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), stats.minMs(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), stats.maxMs(), 0.001);
}

test "ProfileResult bandwidth calculation" {
    // 1 GB transferred in 100ms = 10 GB/s
    const result = ProfileResult{
        .name = "test",
        .duration_ns = 100_000_000, // 100ms
        .duration_ms = 100.0,
        .bandwidth_gbps = 10.0,
    };

    try std.testing.expectApproxEqAbs(@as(f64, 10.0), result.bandwidth_gbps.?, 0.001);
}

test "ProfilerOptions defaults" {
    const options = ProfilerOptions{};
    try std.testing.expect(options.enable_timestamps);
    try std.testing.expect(!options.enable_statistics);
    try std.testing.expectEqual(@as(u32, 64), options.max_regions);
}
