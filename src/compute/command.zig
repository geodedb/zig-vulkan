// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Vulkan Command Buffer Management
//!
//! Provides command buffer recording and submission utilities for compute operations.

const std = @import("std");
const types = @import("../vk/types.zig");
const loader = @import("../vk/loader.zig");
const err = @import("../vk/error.zig");
const Context = @import("context.zig").Context;

/// Command buffer wrapper for single-use recording
pub const CommandBuffer = struct {
    handle: types.CommandBuffer,
    pool: types.CommandPool,
    device: types.Device,
    dispatch: loader.DeviceDispatch,

    const Self = @This();

    /// Allocate a command buffer from the context's pool
    pub fn init(ctx: *Context) err.Error!Self {
        const alloc_info = types.CommandBufferAllocateInfo{
            .command_pool = ctx.command_pool,
            .level = .primary,
            .command_buffer_count = 1,
        };

        var handle: types.CommandBuffer = undefined;
        const result = ctx.device_dispatch.vkAllocateCommandBuffers.?(ctx.device, &alloc_info, @ptrCast(&handle));
        if (result != .success) return err.fromVkResult(result);

        return Self{
            .handle = handle,
            .pool = ctx.command_pool,
            .device = ctx.device,
            .dispatch = ctx.device_dispatch,
        };
    }

    /// Free the command buffer
    pub fn deinit(self: *Self) void {
        self.dispatch.vkFreeCommandBuffers.?(self.device, self.pool, 1, @ptrCast(&self.handle));
        self.* = undefined;
    }

    /// Begin recording commands
    pub fn begin(self: Self) err.Error!void {
        const begin_info = types.CommandBufferBeginInfo{
            .flags = .{ .one_time_submit = true },
        };

        const result = self.dispatch.vkBeginCommandBuffer.?(self.handle, &begin_info);
        if (result != .success) return err.fromVkResult(result);
    }

    /// End recording commands
    pub fn end(self: Self) err.Error!void {
        const result = self.dispatch.vkEndCommandBuffer.?(self.handle);
        if (result != .success) return err.fromVkResult(result);
    }

    /// Reset the command buffer for reuse
    pub fn reset(self: Self) err.Error!void {
        const result = self.dispatch.vkResetCommandBuffer.?(self.handle, .{});
        if (result != .success) return err.fromVkResult(result);
    }

    /// Get the raw handle for direct Vulkan calls
    pub fn getHandle(self: Self) types.CommandBuffer {
        return self.handle;
    }
};

/// Fence for CPU-GPU synchronization
pub const Fence = struct {
    handle: types.Fence,
    device: types.Device,
    dispatch: loader.DeviceDispatch,

    const Self = @This();

    /// Create a fence
    pub fn init(ctx: *Context, signaled: bool) err.Error!Self {
        const create_info = types.FenceCreateInfo{
            .flags = if (signaled) .{ .signaled = true } else .{},
        };

        var handle: types.Fence = undefined;
        const result = ctx.device_dispatch.vkCreateFence.?(ctx.device, &create_info, null, &handle);
        if (result != .success) return err.fromVkResult(result);

        return Self{
            .handle = handle,
            .device = ctx.device,
            .dispatch = ctx.device_dispatch,
        };
    }

    /// Destroy the fence
    pub fn deinit(self: *Self) void {
        self.dispatch.vkDestroyFence.?(self.device, self.handle, null);
        self.* = undefined;
    }

    /// Wait for the fence to become signaled
    pub fn wait(self: Self, timeout_ns: u64) err.Error!bool {
        const result = self.dispatch.vkWaitForFences.?(self.device, 1, @ptrCast(&self.handle), 1, timeout_ns);
        return switch (result) {
            .success => true,
            .timeout => false,
            else => err.fromVkResult(result),
        };
    }

    /// Wait indefinitely for the fence
    pub fn waitForever(self: Self) err.Error!void {
        const result = self.dispatch.vkWaitForFences.?(self.device, 1, @ptrCast(&self.handle), 1, std.math.maxInt(u64));
        if (result != .success) return err.fromVkResult(result);
    }

    /// Reset the fence to unsignaled state
    pub fn reset(self: Self) err.Error!void {
        const result = self.dispatch.vkResetFences.?(self.device, 1, @ptrCast(&self.handle));
        if (result != .success) return err.fromVkResult(result);
    }

    /// Check if fence is signaled (non-blocking)
    pub fn isSignaled(self: Self) bool {
        return self.dispatch.vkGetFenceStatus.?(self.device, self.handle) == .success;
    }
};

/// Semaphore for GPU-GPU synchronization
pub const Semaphore = struct {
    handle: types.Semaphore,
    device: types.Device,
    dispatch: loader.DeviceDispatch,

    const Self = @This();

    /// Create a semaphore
    pub fn init(ctx: *Context) err.Error!Self {
        const create_info = types.SemaphoreCreateInfo{};

        var handle: types.Semaphore = undefined;
        const result = ctx.device_dispatch.vkCreateSemaphore.?(ctx.device, &create_info, null, &handle);
        if (result != .success) return err.fromVkResult(result);

        return Self{
            .handle = handle,
            .device = ctx.device,
            .dispatch = ctx.device_dispatch,
        };
    }

    /// Destroy the semaphore
    pub fn deinit(self: *Self) void {
        self.dispatch.vkDestroySemaphore.?(self.device, self.handle, null);
        self.* = undefined;
    }
};

/// Queue submission helper
pub const Submission = struct {
    /// Submit a single command buffer and wait for completion
    pub fn submitAndWait(ctx: *Context, cmd: CommandBuffer) err.Error!void {
        var fence = try Fence.init(ctx, false);
        defer fence.deinit();

        try submit(ctx, cmd, null, null, fence);
        try fence.waitForever();
    }

    /// Submit a command buffer with optional synchronization
    pub fn submit(
        ctx: *Context,
        cmd: CommandBuffer,
        wait_semaphore: ?Semaphore,
        signal_semaphore: ?Semaphore,
        fence: ?Fence,
    ) err.Error!void {
        var wait_stage = types.PipelineStageFlags{ .compute_shader = true };

        const submit_info = types.SubmitInfo{
            .wait_semaphore_count = if (wait_semaphore != null) 1 else 0,
            .p_wait_semaphores = if (wait_semaphore) |ws| @ptrCast(&ws.handle) else null,
            .p_wait_dst_stage_mask = if (wait_semaphore != null) @ptrCast(&wait_stage) else null,
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&cmd.handle),
            .signal_semaphore_count = if (signal_semaphore != null) 1 else 0,
            .p_signal_semaphores = if (signal_semaphore) |ss| @ptrCast(&ss.handle) else null,
        };

        const result = ctx.device_dispatch.vkQueueSubmit.?(
            ctx.compute_queue,
            1,
            @ptrCast(&submit_info),
            if (fence) |f| f.handle else null,
        );
        if (result != .success) return err.fromVkResult(result);
    }

    /// Wait for the compute queue to become idle
    pub fn waitQueueIdle(ctx: *Context) err.Error!void {
        const result = ctx.device_dispatch.vkQueueWaitIdle.?(ctx.compute_queue);
        if (result != .success) return err.fromVkResult(result);
    }
};

/// Query pool for timestamps
/// GPU Query API
///
/// The query API provides two main query types with a consistent interface:
///
/// ## Query Types
///
/// - `TimestampQuery`: Measures GPU execution time
/// - `PipelineStatisticsQuery`: Counts shader invocations and other metrics
///
/// ## Common Usage Pattern
///
/// All query types follow this lifecycle:
/// 1. `init()` - Create the query pool
/// 2. `reset()` - Reset queries before use (in command buffer)
/// 3. Record operations with query markers
/// 4. `getResults()` / `getStatistics()` - Retrieve results after execution
/// 5. `deinit()` - Clean up resources
///
/// ## Example
///
/// ```zig
/// // Timestamp query for measuring kernel execution time
/// var timestamps = try TimestampQuery.init(&ctx, 2);
/// defer timestamps.deinit();
///
/// try cmd.begin();
/// timestamps.reset(cmd.handle, 0, 2);
/// timestamps.writeTimestamp(cmd.handle, .{ .top_of_pipe = true }, 0);
/// // ... dispatch compute work ...
/// timestamps.writeTimestamp(cmd.handle, .{ .bottom_of_pipe = true }, 1);
/// try cmd.end();
///
/// // After execution
/// const elapsed = try timestamps.getElapsedNs(allocator, 0, 1);
/// ```
pub const TimestampQuery = struct {
    handle: types.QueryPool,
    device: types.Device,
    dispatch: loader.DeviceDispatch,
    query_count: u32,
    timestamp_period: f32,

    const Self = @This();

    /// Create a timestamp query pool
    pub fn init(ctx: *Context, query_count: u32) err.Error!Self {
        if (ctx.timestamp_valid_bits == 0) {
            return err.Error.FeatureNotPresent;
        }

        const create_info = types.QueryPoolCreateInfo{
            .query_type = .timestamp,
            .query_count = query_count,
        };

        var handle: types.QueryPool = undefined;
        const result = ctx.device_dispatch.vkCreateQueryPool.?(ctx.device, &create_info, null, &handle);
        if (result != .success) return err.fromVkResult(result);

        return Self{
            .handle = handle,
            .device = ctx.device,
            .dispatch = ctx.device_dispatch,
            .query_count = query_count,
            .timestamp_period = ctx.timestamp_period,
        };
    }

    /// Destroy the query pool
    pub fn deinit(self: *Self) void {
        self.dispatch.vkDestroyQueryPool.?(self.device, self.handle, null);
        self.* = undefined;
    }

    /// Reset queries (must be done before use)
    pub fn reset(self: Self, cmd: types.CommandBuffer, first_query: u32, count: u32) void {
        self.dispatch.vkCmdResetQueryPool.?(cmd, self.handle, first_query, count);
    }

    /// Write a timestamp
    pub fn writeTimestamp(self: Self, cmd: types.CommandBuffer, stage: types.PipelineStageFlags, query_index: u32) void {
        self.dispatch.vkCmdWriteTimestamp.?(cmd, stage, self.handle, query_index);
    }

    /// Get timestamp results (returns timestamps in nanoseconds)
    pub fn getResults(self: Self, allocator: std.mem.Allocator, first_query: u32, count: u32) err.Error![]u64 {
        const results = allocator.alloc(u64, count) catch return err.Error.OutOfHostMemory;
        errdefer allocator.free(results);

        const result = self.dispatch.vkGetQueryPoolResults.?(
            self.device,
            self.handle,
            first_query,
            count,
            count * @sizeOf(u64),
            @ptrCast(results.ptr),
            @sizeOf(u64),
            .{ .result_64 = true, .wait = true },
        );

        if (result != .success) {
            allocator.free(results);
            return err.fromVkResult(result);
        }

        // Convert to nanoseconds using timestamp period
        // Validate and cap to prevent overflow/NaN issues
        for (results) |*ts| {
            const raw_ns = @as(f64, @floatFromInt(ts.*)) * self.timestamp_period;
            // Check for valid finite value within u64 range
            if (std.math.isNan(raw_ns) or std.math.isInf(raw_ns) or raw_ns < 0) {
                ts.* = 0; // Invalid timestamp, treat as zero
            } else if (raw_ns >= @as(f64, @floatFromInt(std.math.maxInt(u64)))) {
                ts.* = std.math.maxInt(u64); // Cap at max u64
            } else {
                ts.* = @intFromFloat(raw_ns);
            }
        }

        return results;
    }

    /// Get elapsed time between two queries in nanoseconds
    pub fn getElapsedNs(self: Self, allocator: std.mem.Allocator, start_query: u32, end_query: u32) err.Error!u64 {
        const results = try self.getResults(allocator, start_query, end_query - start_query + 1);
        defer allocator.free(results);

        const start_ts = results[0];
        const end_ts = results[results.len - 1];

        return if (end_ts >= start_ts) end_ts - start_ts else 0;
    }
};

/// Pipeline statistics flags for compute queries
pub const PipelineStatisticsFlags = packed struct {
    input_assembly_vertices: bool = false,
    input_assembly_primitives: bool = false,
    vertex_shader_invocations: bool = false,
    geometry_shader_invocations: bool = false,
    geometry_shader_primitives: bool = false,
    clipping_invocations: bool = false,
    clipping_primitives: bool = false,
    fragment_shader_invocations: bool = false,
    tessellation_control_patches: bool = false,
    tessellation_evaluation_invocations: bool = false,
    compute_shader_invocations: bool = false,
    _reserved: u21 = 0,

    /// Stats relevant for compute shaders
    pub fn computeOnly() PipelineStatisticsFlags {
        return .{ .compute_shader_invocations = true };
    }

    /// All available statistics
    pub fn all() PipelineStatisticsFlags {
        return .{
            .input_assembly_vertices = true,
            .input_assembly_primitives = true,
            .vertex_shader_invocations = true,
            .geometry_shader_invocations = true,
            .geometry_shader_primitives = true,
            .clipping_invocations = true,
            .clipping_primitives = true,
            .fragment_shader_invocations = true,
            .tessellation_control_patches = true,
            .tessellation_evaluation_invocations = true,
            .compute_shader_invocations = true,
        };
    }

    /// Count number of enabled statistics
    pub fn count(self: PipelineStatisticsFlags) u32 {
        var n: u32 = 0;
        if (self.input_assembly_vertices) n += 1;
        if (self.input_assembly_primitives) n += 1;
        if (self.vertex_shader_invocations) n += 1;
        if (self.geometry_shader_invocations) n += 1;
        if (self.geometry_shader_primitives) n += 1;
        if (self.clipping_invocations) n += 1;
        if (self.clipping_primitives) n += 1;
        if (self.fragment_shader_invocations) n += 1;
        if (self.tessellation_control_patches) n += 1;
        if (self.tessellation_evaluation_invocations) n += 1;
        if (self.compute_shader_invocations) n += 1;
        return n;
    }
};

/// Pipeline statistics results
pub const PipelineStatistics = struct {
    input_assembly_vertices: u64 = 0,
    input_assembly_primitives: u64 = 0,
    vertex_shader_invocations: u64 = 0,
    geometry_shader_invocations: u64 = 0,
    geometry_shader_primitives: u64 = 0,
    clipping_invocations: u64 = 0,
    clipping_primitives: u64 = 0,
    fragment_shader_invocations: u64 = 0,
    tessellation_control_patches: u64 = 0,
    tessellation_evaluation_invocations: u64 = 0,
    compute_shader_invocations: u64 = 0,
};

/// Query pool for pipeline statistics
pub const PipelineStatisticsQuery = struct {
    handle: types.QueryPool,
    device: types.Device,
    dispatch: loader.DeviceDispatch,
    query_count: u32,
    statistics_flags: PipelineStatisticsFlags,
    stats_per_query: u32,

    const Self = @This();

    /// Create a pipeline statistics query pool
    pub fn init(ctx: *Context, query_count: u32, statistics_flags: PipelineStatisticsFlags) err.Error!Self {
        const stats_count = statistics_flags.count();
        if (stats_count == 0) {
            return err.Error.InvalidValue;
        }

        const create_info = types.QueryPoolCreateInfo{
            .query_type = .pipeline_statistics,
            .query_count = query_count,
            .pipeline_statistics = @bitCast(statistics_flags),
        };

        var handle: types.QueryPool = undefined;
        const result = ctx.device_dispatch.vkCreateQueryPool.?(ctx.device, &create_info, null, &handle);
        if (result != .success) return err.fromVkResult(result);

        return Self{
            .handle = handle,
            .device = ctx.device,
            .dispatch = ctx.device_dispatch,
            .query_count = query_count,
            .statistics_flags = statistics_flags,
            .stats_per_query = stats_count,
        };
    }

    /// Destroy the query pool
    pub fn deinit(self: *Self) void {
        self.dispatch.vkDestroyQueryPool.?(self.device, self.handle, null);
        self.* = undefined;
    }

    /// Reset queries (must be done before use)
    pub fn reset(self: Self, cmd: types.CommandBuffer, first_query: u32, count: u32) void {
        self.dispatch.vkCmdResetQueryPool.?(cmd, self.handle, first_query, count);
    }

    /// Begin a statistics query
    pub fn begin(self: Self, cmd: types.CommandBuffer, query_index: u32) void {
        self.dispatch.vkCmdBeginQuery.?(cmd, self.handle, query_index, 0);
    }

    /// End a statistics query
    pub fn end(self: Self, cmd: types.CommandBuffer, query_index: u32) void {
        self.dispatch.vkCmdEndQuery.?(cmd, self.handle, query_index);
    }

    /// Get raw results for a query
    pub fn getRawResults(self: Self, allocator: std.mem.Allocator, query_index: u32) err.Error![]u64 {
        const results = allocator.alloc(u64, self.stats_per_query) catch return err.Error.OutOfHostMemory;
        errdefer allocator.free(results);

        const result = self.dispatch.vkGetQueryPoolResults.?(
            self.device,
            self.handle,
            query_index,
            1,
            self.stats_per_query * @sizeOf(u64),
            @ptrCast(results.ptr),
            @sizeOf(u64),
            .{ .result_64 = true, .wait = true },
        );

        if (result != .success and result != .not_ready) {
            allocator.free(results);
            return err.fromVkResult(result);
        }

        return results;
    }

    /// Get parsed pipeline statistics for a query
    pub fn getStatistics(self: Self, allocator: std.mem.Allocator, query_index: u32) err.Error!PipelineStatistics {
        const raw = try self.getRawResults(allocator, query_index);
        defer allocator.free(raw);

        var stats = PipelineStatistics{};
        var idx: usize = 0;

        // Results are returned in order of bit flags
        if (self.statistics_flags.input_assembly_vertices) {
            stats.input_assembly_vertices = raw[idx];
            idx += 1;
        }
        if (self.statistics_flags.input_assembly_primitives) {
            stats.input_assembly_primitives = raw[idx];
            idx += 1;
        }
        if (self.statistics_flags.vertex_shader_invocations) {
            stats.vertex_shader_invocations = raw[idx];
            idx += 1;
        }
        if (self.statistics_flags.geometry_shader_invocations) {
            stats.geometry_shader_invocations = raw[idx];
            idx += 1;
        }
        if (self.statistics_flags.geometry_shader_primitives) {
            stats.geometry_shader_primitives = raw[idx];
            idx += 1;
        }
        if (self.statistics_flags.clipping_invocations) {
            stats.clipping_invocations = raw[idx];
            idx += 1;
        }
        if (self.statistics_flags.clipping_primitives) {
            stats.clipping_primitives = raw[idx];
            idx += 1;
        }
        if (self.statistics_flags.fragment_shader_invocations) {
            stats.fragment_shader_invocations = raw[idx];
            idx += 1;
        }
        if (self.statistics_flags.tessellation_control_patches) {
            stats.tessellation_control_patches = raw[idx];
            idx += 1;
        }
        if (self.statistics_flags.tessellation_evaluation_invocations) {
            stats.tessellation_evaluation_invocations = raw[idx];
            idx += 1;
        }
        if (self.statistics_flags.compute_shader_invocations) {
            stats.compute_shader_invocations = raw[idx];
            idx += 1;
        }

        return stats;
    }

    /// Get compute shader invocations for a query (convenience method)
    pub fn getComputeInvocations(self: Self, allocator: std.mem.Allocator, query_index: u32) err.Error!u64 {
        const stats = try self.getStatistics(allocator, query_index);
        return stats.compute_shader_invocations;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "Submission struct exists" {
    // Basic compilation test
    _ = Submission.submitAndWait;
    _ = Submission.submit;
    _ = Submission.waitQueueIdle;
}
