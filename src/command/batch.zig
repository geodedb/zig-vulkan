// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Batch Command Recording
//!
//! Multi-dispatch recording helpers with automatic barriers and resource tracking.
//! Simplifies recording multiple compute dispatches with proper synchronization.
//!
//! Features:
//! - Fluent API for batch recording
//! - Automatic barrier insertion between dispatches
//! - Resource access tracking
//! - Indirect dispatch support
//! - Command buffer reuse

const std = @import("std");
const types = @import("../vk/types.zig");
const Context = @import("../compute/context.zig").Context;

/// Resource access tracking for automatic barriers
pub const ResourceAccess = struct {
    resource: Resource,
    stage: types.PipelineStageFlags = .{ .compute_shader = true },
    access: types.AccessFlags,
    layout: ?types.ImageLayout = null, // For images

    pub const Resource = union(enum) {
        buffer: types.Buffer,
        image: types.Image,
    };
};

/// Dispatch command configuration
pub const DispatchConfig = struct {
    /// Pipeline to bind (null to use current)
    pipeline: ?types.Pipeline = null,

    /// Pipeline layout
    layout: ?types.PipelineLayout = null,

    /// Descriptor sets to bind
    descriptor_sets: []const types.DescriptorSet = &.{},
    first_set: u32 = 0,

    /// Dynamic offsets for dynamic buffers
    dynamic_offsets: []const u32 = &.{},

    /// Push constant data
    push_constants: ?[]const u8 = null,
    push_offset: u32 = 0,

    /// Dispatch dimensions
    workgroups: [3]u32 = .{ 1, 1, 1 },

    /// Resources accessed (for automatic barriers)
    reads: []const ResourceAccess = &.{},
    writes: []const ResourceAccess = &.{},

    /// Optional label for debugging
    label: ?[]const u8 = null,
};

/// Indirect dispatch configuration
pub const IndirectDispatchConfig = struct {
    pipeline: ?types.Pipeline = null,
    layout: ?types.PipelineLayout = null,
    descriptor_sets: []const types.DescriptorSet = &.{},
    first_set: u32 = 0,
    dynamic_offsets: []const u32 = &.{},
    push_constants: ?[]const u8 = null,
    push_offset: u32 = 0,

    /// Indirect buffer containing dispatch parameters
    indirect_buffer: types.Buffer,
    indirect_offset: u64 = 0,

    reads: []const ResourceAccess = &.{},
    writes: []const ResourceAccess = &.{},
    label: ?[]const u8 = null,
};

/// Batch command recorder for compute dispatches
pub const BatchRecorder = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    ctx: *Context,
    cmd: types.CommandBuffer,

    /// State tracking
    current_pipeline: ?types.Pipeline = null,
    current_layout: ?types.PipelineLayout = null,

    /// Resource state tracking for barriers
    resource_states: std.AutoHashMapUnmanaged(usize, ResourceState),

    /// Statistics
    dispatch_count: u32 = 0,
    barrier_count: u32 = 0,

    /// Debug mode
    insert_labels: bool = false,

    const ResourceState = struct {
        last_stage: types.PipelineStageFlags,
        last_access: types.AccessFlags,
        last_layout: ?types.ImageLayout,
    };

    pub fn init(allocator: std.mem.Allocator, ctx: *Context, cmd: types.CommandBuffer) Self {
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .cmd = cmd,
            .resource_states = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.resource_states.deinit(self.allocator);
    }

    /// Enable debug labels for profiling tools
    pub fn enableLabels(self: *Self) *Self {
        self.insert_labels = true;
        return self;
    }

    /// Record a compute dispatch
    pub fn dispatch(self: *Self, config: DispatchConfig) !*Self {
        // Insert barriers for resource hazards
        try self.insertBarriers(config.reads, config.writes);

        // Begin debug label if enabled
        if (self.insert_labels) {
            if (config.label) |label| {
                self.beginLabel(label, .{ 0.2, 0.6, 0.8, 1.0 });
            }
        }

        // Bind pipeline if changed
        if (config.pipeline) |pipeline| {
            if (pipeline != self.current_pipeline) {
                self.ctx.device_dispatch.cmdBindPipeline(self.cmd, .compute, pipeline);
                self.current_pipeline = pipeline;
            }
        }

        // Update layout
        if (config.layout) |layout| {
            self.current_layout = layout;
        }

        // Bind descriptor sets
        if (config.descriptor_sets.len > 0) {
            if (self.current_layout) |layout| {
                self.ctx.device_dispatch.cmdBindDescriptorSets(
                    self.cmd,
                    .compute,
                    layout,
                    config.first_set,
                    @intCast(config.descriptor_sets.len),
                    config.descriptor_sets.ptr,
                    @intCast(config.dynamic_offsets.len),
                    if (config.dynamic_offsets.len > 0) config.dynamic_offsets.ptr else null,
                );
            }
        }

        // Push constants
        if (config.push_constants) |data| {
            if (self.current_layout) |layout| {
                self.ctx.device_dispatch.cmdPushConstants(
                    self.cmd,
                    layout,
                    .{ .compute = true },
                    config.push_offset,
                    @intCast(data.len),
                    data.ptr,
                );
            }
        }

        // Dispatch
        self.ctx.device_dispatch.cmdDispatch(
            self.cmd,
            config.workgroups[0],
            config.workgroups[1],
            config.workgroups[2],
        );

        // End debug label
        if (self.insert_labels and config.label != null) {
            self.endLabel();
        }

        // Update resource states
        try self.updateResourceStates(config.reads, config.writes);

        self.dispatch_count += 1;
        return self;
    }

    /// Record an indirect dispatch
    pub fn dispatchIndirect(self: *Self, config: IndirectDispatchConfig) !*Self {
        try self.insertBarriers(config.reads, config.writes);

        if (self.insert_labels) {
            if (config.label) |label| {
                self.beginLabel(label, .{ 0.8, 0.6, 0.2, 1.0 });
            }
        }

        if (config.pipeline) |pipeline| {
            if (pipeline != self.current_pipeline) {
                self.ctx.device_dispatch.cmdBindPipeline(self.cmd, .compute, pipeline);
                self.current_pipeline = pipeline;
            }
        }

        if (config.layout) |layout| {
            self.current_layout = layout;
        }

        if (config.descriptor_sets.len > 0) {
            if (self.current_layout) |layout| {
                self.ctx.device_dispatch.cmdBindDescriptorSets(
                    self.cmd,
                    .compute,
                    layout,
                    config.first_set,
                    @intCast(config.descriptor_sets.len),
                    config.descriptor_sets.ptr,
                    @intCast(config.dynamic_offsets.len),
                    if (config.dynamic_offsets.len > 0) config.dynamic_offsets.ptr else null,
                );
            }
        }

        if (config.push_constants) |data| {
            if (self.current_layout) |layout| {
                self.ctx.device_dispatch.cmdPushConstants(
                    self.cmd,
                    layout,
                    .{ .compute = true },
                    config.push_offset,
                    @intCast(data.len),
                    data.ptr,
                );
            }
        }

        // Validate indirect offset is 4-byte aligned (Vulkan spec requirement)
        if (config.indirect_offset % 4 != 0) {
            std.log.err("Indirect dispatch offset {d} is not 4-byte aligned (Vulkan spec violation)", .{config.indirect_offset});
            return error.InvalidAlignment;
        }

        self.ctx.device_dispatch.cmdDispatchIndirect(
            self.cmd,
            config.indirect_buffer,
            config.indirect_offset,
        );

        if (self.insert_labels and config.label != null) {
            self.endLabel();
        }

        try self.updateResourceStates(config.reads, config.writes);

        self.dispatch_count += 1;
        return self;
    }

    /// Insert explicit memory barrier
    pub fn barrier(self: *Self, src_stage: types.PipelineStageFlags, dst_stage: types.PipelineStageFlags, src_access: types.AccessFlags, dst_access: types.AccessFlags) *Self {
        const memory_barrier = types.MemoryBarrier{
            .s_type = .memory_barrier,
            .p_next = null,
            .src_access_mask = src_access,
            .dst_access_mask = dst_access,
        };

        self.ctx.device_dispatch.cmdPipelineBarrier(
            self.cmd,
            src_stage,
            dst_stage,
            .{},
            1,
            @ptrCast(&memory_barrier),
            0,
            null,
            0,
            null,
        );

        self.barrier_count += 1;
        return self;
    }

    /// Insert compute-to-compute barrier (common case)
    pub fn computeBarrier(self: *Self) *Self {
        return self.barrier(
            .{ .compute_shader_bit = true },
            .{ .compute_shader_bit = true },
            .{ .shader_write_bit = true },
            .{ .shader_read_bit = true, .shader_write_bit = true },
        );
    }

    /// Insert buffer barrier
    pub fn bufferBarrier(
        self: *Self,
        buffer: types.Buffer,
        offset: u64,
        size: u64,
        src_stage: types.PipelineStageFlags,
        dst_stage: types.PipelineStageFlags,
        src_access: types.AccessFlags,
        dst_access: types.AccessFlags,
    ) *Self {
        const buffer_barrier = types.BufferMemoryBarrier{
            .s_type = .buffer_memory_barrier,
            .p_next = null,
            .src_access_mask = src_access,
            .dst_access_mask = dst_access,
            .src_queue_family_index = types.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = types.QUEUE_FAMILY_IGNORED,
            .buffer = buffer,
            .offset = offset,
            .size = size,
        };

        self.ctx.device_dispatch.cmdPipelineBarrier(
            self.cmd,
            src_stage,
            dst_stage,
            .{},
            0,
            null,
            1,
            @ptrCast(&buffer_barrier),
            0,
            null,
        );

        self.barrier_count += 1;
        return self;
    }

    /// Fill buffer with value
    pub fn fillBuffer(self: *Self, buffer: types.Buffer, offset: u64, size: u64, value: u32) *Self {
        self.ctx.device_dispatch.cmdFillBuffer(self.cmd, buffer, offset, size, value);
        return self;
    }

    /// Copy buffer
    pub fn copyBuffer(
        self: *Self,
        src: types.Buffer,
        dst: types.Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    ) *Self {
        const region = types.BufferCopy{
            .src_offset = src_offset,
            .dst_offset = dst_offset,
            .size = size,
        };

        self.ctx.device_dispatch.cmdCopyBuffer(
            self.cmd,
            src,
            dst,
            1,
            @ptrCast(&region),
        );

        return self;
    }

    /// Update buffer
    pub fn updateBuffer(self: *Self, buffer: types.Buffer, offset: u64, data: []const u8) *Self {
        self.ctx.device_dispatch.cmdUpdateBuffer(
            self.cmd,
            buffer,
            offset,
            @intCast(data.len),
            data.ptr,
        );
        return self;
    }

    /// Execute multiple identical dispatches (multi-draw pattern)
    pub fn dispatchMultiple(self: *Self, config: DispatchConfig, count: u32) !*Self {
        // Bind once
        if (config.pipeline) |pipeline| {
            if (pipeline != self.current_pipeline) {
                self.ctx.device_dispatch.cmdBindPipeline(self.cmd, .compute, pipeline);
                self.current_pipeline = pipeline;
            }
        }

        if (config.layout) |layout| {
            self.current_layout = layout;
        }

        if (config.descriptor_sets.len > 0) {
            if (self.current_layout) |layout| {
                self.ctx.device_dispatch.cmdBindDescriptorSets(
                    self.cmd,
                    .compute,
                    layout,
                    config.first_set,
                    @intCast(config.descriptor_sets.len),
                    config.descriptor_sets.ptr,
                    @intCast(config.dynamic_offsets.len),
                    if (config.dynamic_offsets.len > 0) config.dynamic_offsets.ptr else null,
                );
            }
        }

        // Dispatch multiple times
        for (0..count) |_| {
            self.ctx.device_dispatch.cmdDispatch(
                self.cmd,
                config.workgroups[0],
                config.workgroups[1],
                config.workgroups[2],
            );
            self.dispatch_count += 1;
        }

        try self.updateResourceStates(config.reads, config.writes);

        return self;
    }

    fn insertBarriers(
        self: *Self,
        reads: []const ResourceAccess,
        writes: []const ResourceAccess,
    ) !void {
        // Check reads for write-after-read hazards
        for (reads) |access| {
            const key = resourceKey(access.resource);
            if (self.resource_states.get(key)) |state| {
                if (hasWriteAccess(state.last_access)) {
                    try self.insertResourceBarrier(access, state);
                }
            }
        }

        // Check writes for read-after-write and write-after-write hazards
        for (writes) |access| {
            const key = resourceKey(access.resource);
            if (self.resource_states.get(key)) |state| {
                try self.insertResourceBarrier(access, state);
            }
        }
    }

    fn insertResourceBarrier(self: *Self, access: ResourceAccess, state: ResourceState) !void {
        const memory_barrier = types.MemoryBarrier{
            .s_type = .memory_barrier,
            .p_next = null,
            .src_access_mask = state.last_access,
            .dst_access_mask = access.access,
        };

        self.ctx.device_dispatch.cmdPipelineBarrier(
            self.cmd,
            state.last_stage,
            access.stage,
            .{},
            1,
            @ptrCast(&memory_barrier),
            0,
            null,
            0,
            null,
        );

        self.barrier_count += 1;
    }

    fn updateResourceStates(
        self: *Self,
        reads: []const ResourceAccess,
        writes: []const ResourceAccess,
    ) !void {
        for (reads) |access| {
            const key = resourceKey(access.resource);
            try self.resource_states.put(self.allocator, key, .{
                .last_stage = access.stage,
                .last_access = access.access,
                .last_layout = access.layout,
            });
        }

        for (writes) |access| {
            const key = resourceKey(access.resource);
            try self.resource_states.put(self.allocator, key, .{
                .last_stage = access.stage,
                .last_access = access.access,
                .last_layout = access.layout,
            });
        }
    }

    fn beginLabel(self: *Self, name: []const u8, color: [4]f32) void {
        // Vulkan requires null-terminated strings. Use sentinel-terminated slice if available,
        // otherwise use a stack buffer with null terminator for safety.
        var name_buf: [256:0]u8 = undefined;
        const name_z: [*:0]const u8 = if (name.len < name_buf.len) blk: {
            @memcpy(name_buf[0..name.len], name);
            name_buf[name.len] = 0;
            break :blk @ptrCast(&name_buf);
        } else blk: {
            // Truncate long names to fit buffer
            @memcpy(name_buf[0 .. name_buf.len - 1], name[0 .. name_buf.len - 1]);
            name_buf[name_buf.len - 1] = 0;
            break :blk @ptrCast(&name_buf);
        };

        const label_info = types.DebugUtilsLabelEXT{
            .s_type = .debug_utils_label_ext,
            .p_next = null,
            .p_label_name = name_z,
            .color = color,
        };
        self.ctx.device_dispatch.cmdBeginDebugUtilsLabelEXT(self.cmd, &label_info);
    }

    fn endLabel(self: *Self) void {
        self.ctx.device_dispatch.cmdEndDebugUtilsLabelEXT(self.cmd);
    }

    /// Get recording statistics
    pub fn getStats(self: Self) struct {
        dispatch_count: u32,
        barrier_count: u32,
        tracked_resources: usize,
    } {
        return .{
            .dispatch_count = self.dispatch_count,
            .barrier_count = self.barrier_count,
            .tracked_resources = self.resource_states.count(),
        };
    }
};

/// Reusable command buffer for batched recording
pub const ReusableCommandBuffer = struct {
    const Self = @This();

    ctx: *Context,
    pool: types.CommandPool,
    buffer: types.CommandBuffer,
    recorded: bool = false,

    pub fn init(ctx: *Context, pool: types.CommandPool) !Self {
        const alloc_info = types.CommandBufferAllocateInfo{
            .s_type = .command_buffer_allocate_info,
            .p_next = null,
            .command_pool = pool,
            .level = .primary,
            .command_buffer_count = 1,
        };

        var buffer: types.CommandBuffer = undefined;
        const result = ctx.device_dispatch.allocateCommandBuffers(
            ctx.device,
            &alloc_info,
            @ptrCast(&buffer),
        );

        if (result != .success) {
            return error.CommandBufferAllocationFailed;
        }

        return .{
            .ctx = ctx,
            .pool = pool,
            .buffer = buffer,
        };
    }

    pub fn deinit(self: *Self) void {
        self.ctx.device_dispatch.freeCommandBuffers(
            self.ctx.device,
            self.pool,
            1,
            @ptrCast(&self.buffer),
        );
    }

    /// Begin recording (resets if already recorded)
    pub fn begin(self: *Self) !*Self {
        if (self.recorded) {
            const reset_result = self.ctx.device_dispatch.resetCommandBuffer(self.buffer, .{});
            if (reset_result != .success) {
                return error.CommandBufferResetFailed;
            }
        }

        const begin_info = types.CommandBufferBeginInfo{
            .s_type = .command_buffer_begin_info,
            .p_next = null,
            .flags = .{},
            .p_inheritance_info = null,
        };

        const result = self.ctx.device_dispatch.beginCommandBuffer(self.buffer, &begin_info);
        if (result != .success) {
            return error.CommandBufferBeginFailed;
        }

        self.recorded = false;
        return self;
    }

    /// Begin for one-time submit
    pub fn beginOneTime(self: *Self) !*Self {
        if (self.recorded) {
            const reset_result = self.ctx.device_dispatch.resetCommandBuffer(self.buffer, .{});
            if (reset_result != .success) {
                return error.CommandBufferResetFailed;
            }
        }

        const begin_info = types.CommandBufferBeginInfo{
            .s_type = .command_buffer_begin_info,
            .p_next = null,
            .flags = .{ .one_time_submit_bit = true },
            .p_inheritance_info = null,
        };

        const result = self.ctx.device_dispatch.beginCommandBuffer(self.buffer, &begin_info);
        if (result != .success) {
            return error.CommandBufferBeginFailed;
        }

        self.recorded = false;
        return self;
    }

    /// End recording
    pub fn end(self: *Self) !types.CommandBuffer {
        const result = self.ctx.device_dispatch.endCommandBuffer(self.buffer);
        if (result != .success) {
            return error.CommandBufferEndFailed;
        }
        self.recorded = true;
        return self.buffer;
    }

    /// Get the command buffer for recording
    pub fn getBuffer(self: Self) types.CommandBuffer {
        return self.buffer;
    }
};

// Helper functions
fn resourceKey(resource: ResourceAccess.Resource) usize {
    return switch (resource) {
        .buffer => |b| @intFromEnum(b),
        .image => |i| @intFromEnum(i),
    };
}

fn hasWriteAccess(access: types.AccessFlags) bool {
    return access.shader_write_bit or
        access.transfer_write_bit or
        access.host_write_bit or
        access.memory_write_bit;
}

// Tests
test "BatchRecorder basic" {
    // Would need mock context for full test
    const Stats = struct {
        dispatch_count: u32,
        barrier_count: u32,
        tracked_resources: usize,
    };
    const stats: Stats = .{
        .dispatch_count = 0,
        .barrier_count = 0,
        .tracked_resources = 0,
    };
    try std.testing.expectEqual(@as(u32, 0), stats.dispatch_count);
}
