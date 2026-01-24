// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! High-Level Compute Dispatch API
//!
//! Provides ergonomic compute dispatch with automatic workgroup sizing,
//! barriers, and common compute patterns.
//!
//! ## Features
//!
//! - Automatic workgroup size calculation based on problem size
//! - Device-aware optimal workgroup sizing
//! - Common dispatch patterns (1D, 2D, 3D)
//! - Barrier helpers for compute synchronization
//!
//! ## Example
//!
//! ```zig
//! var dispatcher = ComputeDispatcher.init(&ctx);
//!
//! // Dispatch 1M elements with automatic workgroup sizing
//! dispatcher.dispatch1D(cmd, pipeline, 1_000_000, descriptor_set);
//!
//! // Dispatch 2D grid (e.g., image processing)
//! dispatcher.dispatch2D(cmd, pipeline, 1920, 1080, descriptor_set);
//! ```

const std = @import("std");
const types = @import("../vk/types.zig");
const loader = @import("../vk/loader.zig");
const err = @import("../vk/error.zig");
const Context = @import("context.zig").Context;
const ComputePipeline = @import("pipeline.zig").ComputePipeline;
const Buffer = @import("buffer.zig").Buffer;

/// Workgroup size configuration
pub const WorkgroupSize = struct {
    x: u32 = 256,
    y: u32 = 1,
    z: u32 = 1,

    /// Create a 1D workgroup size
    pub fn size1D(x: u32) WorkgroupSize {
        return .{ .x = x, .y = 1, .z = 1 };
    }

    /// Create a 2D workgroup size
    pub fn size2D(x: u32, y: u32) WorkgroupSize {
        return .{ .x = x, .y = y, .z = 1 };
    }

    /// Create a 3D workgroup size
    pub fn size3D(x: u32, y: u32, z: u32) WorkgroupSize {
        return .{ .x = x, .y = y, .z = z };
    }

    /// Total invocations per workgroup
    /// Returns null on overflow (workgroup size too large)
    pub fn totalInvocations(self: WorkgroupSize) u32 {
        const xy = std.math.mul(u32, self.x, self.y) catch return std.math.maxInt(u32);
        return std.math.mul(u32, xy, self.z) catch std.math.maxInt(u32);
    }
};

/// Dispatch dimensions for automatic workgroup calculation
pub const DispatchDimensions = struct {
    /// Total elements in X dimension
    total_x: u32,
    /// Total elements in Y dimension (1 for 1D dispatch)
    total_y: u32 = 1,
    /// Total elements in Z dimension (1 for 1D/2D dispatch)
    total_z: u32 = 1,

    /// Create 1D dispatch dimensions
    pub fn dim1D(total: u32) DispatchDimensions {
        return .{ .total_x = total, .total_y = 1, .total_z = 1 };
    }

    /// Create 2D dispatch dimensions
    pub fn dim2D(width: u32, height: u32) DispatchDimensions {
        return .{ .total_x = width, .total_y = height, .total_z = 1 };
    }

    /// Create 3D dispatch dimensions
    pub fn dim3D(x: u32, y: u32, z: u32) DispatchDimensions {
        return .{ .total_x = x, .total_y = y, .total_z = z };
    }

    /// Total elements across all dimensions
    pub fn totalElements(self: DispatchDimensions) u64 {
        return @as(u64, self.total_x) * self.total_y * self.total_z;
    }
};

/// Computed dispatch parameters
pub const DispatchParams = struct {
    /// Number of workgroups in X
    groups_x: u32,
    /// Number of workgroups in Y
    groups_y: u32,
    /// Number of workgroups in Z
    groups_z: u32,
    /// Workgroup size used
    workgroup_size: WorkgroupSize,
    /// Whether dispatch dimensions exactly divide workgroup size
    is_exact: bool,

    /// Total number of workgroups
    /// Returns maxInt(u64) on overflow
    pub fn totalGroups(self: DispatchParams) u64 {
        const xy = std.math.mul(u64, @as(u64, self.groups_x), self.groups_y) catch return std.math.maxInt(u64);
        return std.math.mul(u64, xy, self.groups_z) catch std.math.maxInt(u64);
    }

    /// Total invocations (may be more than requested due to rounding)
    /// Returns maxInt(u64) on overflow
    pub fn totalInvocations(self: DispatchParams) u64 {
        return std.math.mul(u64, self.totalGroups(), self.workgroup_size.totalInvocations()) catch std.math.maxInt(u64);
    }
};

/// High-level compute dispatcher
pub const ComputeDispatcher = struct {
    device: types.Device,
    dispatch: loader.DeviceDispatch,
    max_workgroup_size: [3]u32,
    max_workgroup_invocations: u32,
    max_compute_workgroup_count: [3]u32,
    subgroup_size: u32,

    const Self = @This();

    /// Initialize dispatcher from context
    pub fn init(ctx: *Context) Self {
        const limits = ctx.device_properties.limits;
        return .{
            .device = ctx.device,
            .dispatch = ctx.device_dispatch,
            .max_workgroup_size = limits.max_compute_work_group_size,
            .max_workgroup_invocations = limits.max_compute_work_group_invocations,
            .max_compute_workgroup_count = limits.max_compute_work_group_count,
            .subgroup_size = 32, // Default, could be queried from VK 1.1+
        };
    }

    /// Calculate optimal dispatch parameters for given dimensions
    pub fn calculateDispatch(
        self: Self,
        dims: DispatchDimensions,
        preferred_workgroup: ?WorkgroupSize,
    ) DispatchParams {
        const wg = preferred_workgroup orelse self.optimalWorkgroupSize(dims);

        // Calculate number of workgroups (round up)
        const groups_x = (dims.total_x + wg.x - 1) / wg.x;
        const groups_y = (dims.total_y + wg.y - 1) / wg.y;
        const groups_z = (dims.total_z + wg.z - 1) / wg.z;

        // Clamp to device limits
        const final_x = @min(groups_x, self.max_compute_workgroup_count[0]);
        const final_y = @min(groups_y, self.max_compute_workgroup_count[1]);
        const final_z = @min(groups_z, self.max_compute_workgroup_count[2]);

        return .{
            .groups_x = final_x,
            .groups_y = final_y,
            .groups_z = final_z,
            .workgroup_size = wg,
            .is_exact = (dims.total_x % wg.x == 0) and
                (dims.total_y % wg.y == 0) and
                (dims.total_z % wg.z == 0),
        };
    }

    /// Get optimal workgroup size for given dimensions
    pub fn optimalWorkgroupSize(self: Self, dims: DispatchDimensions) WorkgroupSize {
        // For 1D dispatch
        if (dims.total_y == 1 and dims.total_z == 1) {
            // Use 256 or max allowed, prefer multiple of subgroup size
            const preferred = @min(@as(u32, 256), self.max_workgroup_invocations);
            const aligned = (preferred / self.subgroup_size) * self.subgroup_size;
            return WorkgroupSize.size1D(@max(aligned, self.subgroup_size));
        }

        // For 2D dispatch (common for image processing)
        if (dims.total_z == 1) {
            // Use 16x16 or similar square workgroup
            // sqrt(max_workgroup_invocations) is always <= sqrt(2^32) ≈ 65536, fits in u32
            // Handle edge case where max_workgroup_invocations could be 0
            if (self.max_workgroup_invocations == 0) {
                return WorkgroupSize.size2D(1, 1);
            }
            const max_per_dim = std.math.sqrt(@as(f64, @floatFromInt(self.max_workgroup_invocations)));
            const max_per_dim_capped = @min(max_per_dim, @as(f64, std.math.maxInt(u32)));
            const size = @max(@as(u32, 1), @min(@as(u32, 16), @as(u32, @intFromFloat(max_per_dim_capped))));
            return WorkgroupSize.size2D(size, size);
        }

        // For 3D dispatch
        // Use 8x8x4 or similar
        // cbrt(max_workgroup_invocations) is always <= cbrt(2^32) ≈ 1625, fits in u32
        const cube_root_f = @floor(std.math.cbrt(@as(f64, @floatFromInt(self.max_workgroup_invocations))));
        const cube_root_capped = @min(cube_root_f, @as(f64, std.math.maxInt(u32)));
        const cube_root: u32 = @intFromFloat(cube_root_capped);
        const size = @min(@as(u32, 8), cube_root);
        return WorkgroupSize.size3D(size, size, @max(size / 2, 1));
    }

    /// Dispatch 1D compute work
    ///
    /// Automatically calculates workgroup count for the given element count.
    /// The shader should use gl_GlobalInvocationID.x to index elements.
    pub fn dispatch1D(
        self: Self,
        cmd: types.CommandBuffer,
        pipeline: ComputePipeline,
        element_count: u32,
        descriptor_set: types.DescriptorSet,
    ) void {
        const params = self.calculateDispatch(DispatchDimensions.dim1D(element_count), null);
        self.dispatchWithParams(cmd, pipeline, params, descriptor_set);
    }

    /// Dispatch 2D compute work (e.g., image processing)
    ///
    /// Uses 16x16 workgroups by default for good occupancy on most GPUs.
    pub fn dispatch2D(
        self: Self,
        cmd: types.CommandBuffer,
        pipeline: ComputePipeline,
        width: u32,
        height: u32,
        descriptor_set: types.DescriptorSet,
    ) void {
        const params = self.calculateDispatch(DispatchDimensions.dim2D(width, height), null);
        self.dispatchWithParams(cmd, pipeline, params, descriptor_set);
    }

    /// Dispatch 3D compute work (e.g., volume processing)
    pub fn dispatch3D(
        self: Self,
        cmd: types.CommandBuffer,
        pipeline: ComputePipeline,
        x: u32,
        y: u32,
        z: u32,
        descriptor_set: types.DescriptorSet,
    ) void {
        const params = self.calculateDispatch(DispatchDimensions.dim3D(x, y, z), null);
        self.dispatchWithParams(cmd, pipeline, params, descriptor_set);
    }

    /// Dispatch with explicit workgroup size
    pub fn dispatchWithWorkgroup(
        self: Self,
        cmd: types.CommandBuffer,
        pipeline: ComputePipeline,
        dims: DispatchDimensions,
        workgroup: WorkgroupSize,
        descriptor_set: types.DescriptorSet,
    ) void {
        const params = self.calculateDispatch(dims, workgroup);
        self.dispatchWithParams(cmd, pipeline, params, descriptor_set);
    }

    /// Dispatch with pre-calculated parameters
    pub fn dispatchWithParams(
        self: Self,
        cmd: types.CommandBuffer,
        pipeline: ComputePipeline,
        params: DispatchParams,
        descriptor_set: types.DescriptorSet,
    ) void {
        pipeline.bind(cmd);
        pipeline.bindDescriptorSet(cmd, descriptor_set);
        self.dispatch.vkCmdDispatch.?(cmd, params.groups_x, params.groups_y, params.groups_z);
    }

    /// Dispatch with push constants
    pub fn dispatch1DWithPushConstants(
        self: Self,
        cmd: types.CommandBuffer,
        pipeline: ComputePipeline,
        element_count: u32,
        descriptor_set: types.DescriptorSet,
        comptime T: type,
        push_constants: *const T,
    ) void {
        const params = self.calculateDispatch(DispatchDimensions.dim1D(element_count), null);
        pipeline.bind(cmd);
        pipeline.bindDescriptorSet(cmd, descriptor_set);
        pipeline.pushConstants(cmd, T, push_constants);
        self.dispatch.vkCmdDispatch.?(cmd, params.groups_x, params.groups_y, params.groups_z);
    }

    /// Insert compute-to-compute barrier (between dependent dispatches)
    pub fn barrierComputeToCompute(self: Self, cmd: types.CommandBuffer) void {
        const memory_barrier = types.MemoryBarrier{
            .src_access_mask = .{ .shader_write = true },
            .dst_access_mask = .{ .shader_read = true },
        };

        self.dispatch.vkCmdPipelineBarrier.?(
            cmd,
            .{ .compute_shader = true },
            .{ .compute_shader = true },
            0,
            1,
            @ptrCast(&memory_barrier),
            0,
            null,
            0,
            null,
        );
    }

    /// Insert full memory barrier (ensures all writes visible)
    pub fn barrierFull(self: Self, cmd: types.CommandBuffer) void {
        const memory_barrier = types.MemoryBarrier{
            .src_access_mask = .{ .shader_write = true, .transfer_write = true },
            .dst_access_mask = .{ .shader_read = true, .transfer_read = true, .host_read = true },
        };

        self.dispatch.vkCmdPipelineBarrier.?(
            cmd,
            .{ .compute_shader = true, .transfer = true },
            .{ .compute_shader = true, .transfer = true, .host = true },
            0,
            1,
            @ptrCast(&memory_barrier),
            0,
            null,
            0,
            null,
        );
    }

    /// Get device limits for workgroups
    pub fn getLimits(self: Self) struct {
        max_workgroup_size: [3]u32,
        max_workgroup_invocations: u32,
        max_workgroup_count: [3]u32,
    } {
        return .{
            .max_workgroup_size = self.max_workgroup_size,
            .max_workgroup_invocations = self.max_workgroup_invocations,
            .max_workgroup_count = self.max_compute_workgroup_count,
        };
    }
};

/// Batch dispatch for multiple compute passes
pub const DispatchBatch = struct {
    cmd: types.CommandBuffer,
    dispatcher: ComputeDispatcher,
    passes_recorded: u32,

    const Self = @This();

    /// Create a new dispatch batch
    pub fn init(cmd: types.CommandBuffer, ctx: *Context) Self {
        return .{
            .cmd = cmd,
            .dispatcher = ComputeDispatcher.init(ctx),
            .passes_recorded = 0,
        };
    }

    /// Add a 1D dispatch pass
    pub fn add1D(
        self: *Self,
        pipeline: ComputePipeline,
        element_count: u32,
        descriptor_set: types.DescriptorSet,
    ) void {
        if (self.passes_recorded > 0) {
            self.dispatcher.barrierComputeToCompute(self.cmd);
        }
        self.dispatcher.dispatch1D(self.cmd, pipeline, element_count, descriptor_set);
        self.passes_recorded += 1;
    }

    /// Add a 2D dispatch pass
    pub fn add2D(
        self: *Self,
        pipeline: ComputePipeline,
        width: u32,
        height: u32,
        descriptor_set: types.DescriptorSet,
    ) void {
        if (self.passes_recorded > 0) {
            self.dispatcher.barrierComputeToCompute(self.cmd);
        }
        self.dispatcher.dispatch2D(self.cmd, pipeline, width, height, descriptor_set);
        self.passes_recorded += 1;
    }

    /// Finalize batch with final barrier
    pub fn finish(self: *Self) void {
        if (self.passes_recorded > 0) {
            self.dispatcher.barrierFull(self.cmd);
        }
    }
};

// =============================================================================
// Tests
// =============================================================================

test "WorkgroupSize calculations" {
    const wg = WorkgroupSize.size2D(16, 16);
    try std.testing.expectEqual(@as(u32, 256), wg.totalInvocations());

    const wg1d = WorkgroupSize.size1D(64);
    try std.testing.expectEqual(@as(u32, 64), wg1d.totalInvocations());
}

test "DispatchDimensions calculations" {
    const dims = DispatchDimensions.dim2D(1920, 1080);
    try std.testing.expectEqual(@as(u64, 1920 * 1080), dims.totalElements());
}

test "DispatchParams calculations" {
    const params = DispatchParams{
        .groups_x = 120,
        .groups_y = 68,
        .groups_z = 1,
        .workgroup_size = WorkgroupSize.size2D(16, 16),
        .is_exact = true,
    };
    try std.testing.expectEqual(@as(u64, 120 * 68), params.totalGroups());
    try std.testing.expectEqual(@as(u64, 120 * 68 * 256), params.totalInvocations());
}
