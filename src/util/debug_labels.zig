// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Debug Labels Integration
//!
//! Automatic debug labels for RenderDoc, NSight, and other GPU profiling tools.
//! Provides hierarchical labeling of GPU work for easy navigation and analysis.
//!
//! Features:
//! - Automatic scope-based labels
//! - Color-coded categories
//! - Object naming for resources
//! - Marker insertion for events
//! - Hierarchical label stacks

const std = @import("std");
const types = @import("../vk/types.zig");
const Context = @import("../compute/context.zig").Context;

/// Predefined colors for common categories
pub const Colors = struct {
    pub const compute: [4]f32 = .{ 0.2, 0.6, 0.9, 1.0 }; // Blue
    pub const transfer: [4]f32 = .{ 0.9, 0.6, 0.2, 1.0 }; // Orange
    pub const barrier: [4]f32 = .{ 0.9, 0.2, 0.2, 1.0 }; // Red
    pub const memory: [4]f32 = .{ 0.2, 0.9, 0.2, 1.0 }; // Green
    pub const descriptor: [4]f32 = .{ 0.6, 0.2, 0.9, 1.0 }; // Purple
    pub const pipeline: [4]f32 = .{ 0.9, 0.9, 0.2, 1.0 }; // Yellow
    pub const resource: [4]f32 = .{ 0.2, 0.9, 0.9, 1.0 }; // Cyan
    pub const custom: [4]f32 = .{ 0.7, 0.7, 0.7, 1.0 }; // Gray
};

/// Debug label category for automatic coloring
pub const Category = enum {
    compute,
    transfer,
    barrier,
    memory,
    descriptor,
    pipeline,
    resource,
    custom,

    pub fn getColor(self: Category) [4]f32 {
        return switch (self) {
            .compute => Colors.compute,
            .transfer => Colors.transfer,
            .barrier => Colors.barrier,
            .memory => Colors.memory,
            .descriptor => Colors.descriptor,
            .pipeline => Colors.pipeline,
            .resource => Colors.resource,
            .custom => Colors.custom,
        };
    }
};

/// Debug utilities manager
pub const DebugUtils = struct {
    const Self = @This();

    ctx: *Context,
    enabled: bool,
    label_stack: std.ArrayListUnmanaged([]const u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, ctx: *Context) Self {
        // Check if debug utils extension is available
        const enabled = ctx.device_api.cmdBeginDebugUtilsLabelEXT != null;

        return .{
            .ctx = ctx,
            .enabled = enabled,
            .label_stack = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.label_stack.deinit(self.allocator);
    }

    /// Check if debug labels are enabled
    pub fn isEnabled(self: Self) bool {
        return self.enabled;
    }

    // ============ Queue Labels ============

    /// Begin a queue label region
    pub fn beginQueueLabel(
        self: *Self,
        queue: types.Queue,
        name: []const u8,
        category: Category,
    ) void {
        if (!self.enabled) return;

        const label_info = types.DebugUtilsLabelEXT{
            .s_type = .debug_utils_label_ext,
            .p_next = null,
            .p_label_name = name.ptr,
            .color = category.getColor(),
        };

        self.ctx.device_api.queueBeginDebugUtilsLabelEXT(queue, &label_info);
        self.label_stack.append(self.allocator, name) catch |e| {
            std.log.debug("DebugUtils.beginQueueLabel: failed to track label stack: {s} - endQueueLabel may be unbalanced", .{
                @errorName(e),
            });
        };
    }

    /// End a queue label region
    pub fn endQueueLabel(self: *Self, queue: types.Queue) void {
        if (!self.enabled) return;

        self.ctx.device_api.queueEndDebugUtilsLabelEXT(queue);
        if (self.label_stack.items.len > 0) {
            _ = self.label_stack.pop();
        }
    }

    /// Insert a queue marker
    pub fn insertQueueLabel(
        self: *Self,
        queue: types.Queue,
        name: []const u8,
        category: Category,
    ) void {
        if (!self.enabled) return;

        const label_info = types.DebugUtilsLabelEXT{
            .s_type = .debug_utils_label_ext,
            .p_next = null,
            .p_label_name = name.ptr,
            .color = category.getColor(),
        };

        self.ctx.device_api.queueInsertDebugUtilsLabelEXT(queue, &label_info);
    }

    // ============ Command Buffer Labels ============

    /// Begin a command buffer label region
    pub fn beginCmdLabel(
        self: *Self,
        cmd: types.CommandBuffer,
        name: []const u8,
        category: Category,
    ) void {
        if (!self.enabled) return;

        const label_info = types.DebugUtilsLabelEXT{
            .s_type = .debug_utils_label_ext,
            .p_next = null,
            .p_label_name = name.ptr,
            .color = category.getColor(),
        };

        self.ctx.device_api.cmdBeginDebugUtilsLabelEXT(cmd, &label_info);
    }

    /// Begin with custom color
    pub fn beginCmdLabelWithColor(
        self: *Self,
        cmd: types.CommandBuffer,
        name: []const u8,
        color: [4]f32,
    ) void {
        if (!self.enabled) return;

        const label_info = types.DebugUtilsLabelEXT{
            .s_type = .debug_utils_label_ext,
            .p_next = null,
            .p_label_name = name.ptr,
            .color = color,
        };

        self.ctx.device_api.cmdBeginDebugUtilsLabelEXT(cmd, &label_info);
    }

    /// End a command buffer label region
    pub fn endCmdLabel(self: *Self, cmd: types.CommandBuffer) void {
        if (!self.enabled) return;
        self.ctx.device_api.cmdEndDebugUtilsLabelEXT(cmd);
    }

    /// Insert a command buffer marker
    pub fn insertCmdLabel(
        self: *Self,
        cmd: types.CommandBuffer,
        name: []const u8,
        category: Category,
    ) void {
        if (!self.enabled) return;

        const label_info = types.DebugUtilsLabelEXT{
            .s_type = .debug_utils_label_ext,
            .p_next = null,
            .p_label_name = name.ptr,
            .color = category.getColor(),
        };

        self.ctx.device_api.cmdInsertDebugUtilsLabelEXT(cmd, &label_info);
    }

    // ============ Object Naming ============

    /// Set debug name for a buffer
    pub fn setBufferName(self: *Self, buffer: types.Buffer, name: []const u8) void {
        self.setObjectName(.buffer, @intFromEnum(buffer), name);
    }

    /// Set debug name for an image
    pub fn setImageName(self: *Self, image: types.Image, name: []const u8) void {
        self.setObjectName(.image, @intFromEnum(image), name);
    }

    /// Set debug name for a pipeline
    pub fn setPipelineName(self: *Self, pipeline: types.Pipeline, name: []const u8) void {
        self.setObjectName(.pipeline, @intFromEnum(pipeline), name);
    }

    /// Set debug name for a descriptor set
    pub fn setDescriptorSetName(self: *Self, set: types.DescriptorSet, name: []const u8) void {
        self.setObjectName(.descriptor_set, @intFromEnum(set), name);
    }

    /// Set debug name for a command buffer
    pub fn setCommandBufferName(self: *Self, cmd: types.CommandBuffer, name: []const u8) void {
        self.setObjectName(.command_buffer, @intFromEnum(cmd), name);
    }

    /// Set debug name for a queue
    pub fn setQueueName(self: *Self, queue: types.Queue, name: []const u8) void {
        self.setObjectName(.queue, @intFromEnum(queue), name);
    }

    /// Set debug name for a shader module
    pub fn setShaderModuleName(self: *Self, module: types.ShaderModule, name: []const u8) void {
        self.setObjectName(.shader_module, @intFromEnum(module), name);
    }

    /// Set debug name for any object
    pub fn setObjectName(
        self: *Self,
        object_type: types.ObjectType,
        handle: u64,
        name: []const u8,
    ) void {
        if (!self.enabled) return;

        const name_info = types.DebugUtilsObjectNameInfoEXT{
            .s_type = .debug_utils_object_name_info_ext,
            .p_next = null,
            .object_type = object_type,
            .object_handle = handle,
            .p_object_name = name.ptr,
        };

        _ = self.ctx.device_api.setDebugUtilsObjectNameEXT(self.ctx.device, &name_info);
    }

    /// Set debug tag for an object (arbitrary data)
    pub fn setObjectTag(
        self: *Self,
        object_type: types.ObjectType,
        handle: u64,
        tag_name: u64,
        data: []const u8,
    ) void {
        if (!self.enabled) return;

        const tag_info = types.DebugUtilsObjectTagInfoEXT{
            .s_type = .debug_utils_object_tag_info_ext,
            .p_next = null,
            .object_type = object_type,
            .object_handle = handle,
            .tag_name = tag_name,
            .tag_size = data.len,
            .p_tag = data.ptr,
        };

        _ = self.ctx.device_api.setDebugUtilsObjectTagEXT(self.ctx.device, &tag_info);
    }
};

/// Scoped debug label (RAII-style)
pub const ScopedLabel = struct {
    debug: *DebugUtils,
    cmd: types.CommandBuffer,

    pub fn init(
        debug: *DebugUtils,
        cmd: types.CommandBuffer,
        name: []const u8,
        category: Category,
    ) ScopedLabel {
        debug.beginCmdLabel(cmd, name, category);
        return .{
            .debug = debug,
            .cmd = cmd,
        };
    }

    pub fn deinit(self: *const ScopedLabel) void {
        self.debug.endCmdLabel(self.cmd);
    }
};

/// Helper for scoped label usage
pub fn scopedLabel(
    debug: *DebugUtils,
    cmd: types.CommandBuffer,
    name: []const u8,
    category: Category,
) ScopedLabel {
    return ScopedLabel.init(debug, cmd, name, category);
}

/// Labeled command recorder with automatic debug labels
pub const LabeledRecorder = struct {
    const Self = @This();

    debug: *DebugUtils,
    cmd: types.CommandBuffer,
    depth: u32 = 0,

    pub fn init(debug: *DebugUtils, cmd: types.CommandBuffer) Self {
        return .{
            .debug = debug,
            .cmd = cmd,
        };
    }

    /// Begin a labeled section
    pub fn begin(self: *Self, name: []const u8, category: Category) *Self {
        self.debug.beginCmdLabel(self.cmd, name, category);
        self.depth += 1;
        return self;
    }

    /// End current section
    pub fn end(self: *Self) *Self {
        if (self.depth > 0) {
            self.debug.endCmdLabel(self.cmd);
            self.depth -= 1;
        }
        return self;
    }

    /// Insert a marker
    pub fn marker(self: *Self, name: []const u8, category: Category) *Self {
        self.debug.insertCmdLabel(self.cmd, name, category);
        return self;
    }

    /// End all open sections
    pub fn endAll(self: *Self) void {
        while (self.depth > 0) {
            self.debug.endCmdLabel(self.cmd);
            self.depth -= 1;
        }
    }

    /// Labeled dispatch
    pub fn labeledDispatch(
        self: *Self,
        name: []const u8,
        x: u32,
        y: u32,
        z: u32,
    ) *Self {
        self.debug.beginCmdLabel(self.cmd, name, .compute);
        self.debug.ctx.device_api.cmdDispatch(self.cmd, x, y, z);
        self.debug.endCmdLabel(self.cmd);
        return self;
    }

    /// Labeled barrier
    pub fn labeledBarrier(
        self: *Self,
        name: []const u8,
        src_stage: types.PipelineStageFlags,
        dst_stage: types.PipelineStageFlags,
        src_access: types.AccessFlags,
        dst_access: types.AccessFlags,
    ) *Self {
        self.debug.beginCmdLabel(self.cmd, name, .barrier);

        const memory_barrier = types.MemoryBarrier{
            .s_type = .memory_barrier,
            .p_next = null,
            .src_access_mask = src_access,
            .dst_access_mask = dst_access,
        };

        self.debug.ctx.device_api.cmdPipelineBarrier(
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

        self.debug.endCmdLabel(self.cmd);
        return self;
    }
};

/// Format a label with frame number
pub fn frameLabel(
    buffer: []u8,
    comptime fmt: []const u8,
    frame: u64,
    args: anytype,
) []const u8 {
    return std.fmt.bufPrint(buffer, "[F{d}] " ++ fmt, .{frame} ++ args) catch buffer[0..0];
}

/// Format a label with timing info
pub fn timedLabel(
    buffer: []u8,
    comptime fmt: []const u8,
    duration_ns: u64,
    args: anytype,
) []const u8 {
    const duration_ms = @as(f64, @floatFromInt(duration_ns)) / 1_000_000.0;
    return std.fmt.bufPrint(buffer, fmt ++ " ({d:.2}ms)", args ++ .{duration_ms}) catch buffer[0..0];
}

// Tests
test "Category colors" {
    const compute_color = Category.compute.getColor();
    try std.testing.expectEqual(@as(f32, 0.2), compute_color[0]);
    try std.testing.expectEqual(@as(f32, 0.6), compute_color[1]);
    try std.testing.expectEqual(@as(f32, 0.9), compute_color[2]);
    try std.testing.expectEqual(@as(f32, 1.0), compute_color[3]);
}

test "frameLabel formatting" {
    var buffer: [128]u8 = undefined;
    const label = frameLabel(&buffer, "Compute Pass {s}", 42, .{"main"});
    try std.testing.expectEqualStrings("[F42] Compute Pass main", label);
}

test "timedLabel formatting" {
    var buffer: [128]u8 = undefined;
    const label = timedLabel(&buffer, "GPU Work", 1_500_000, .{});
    try std.testing.expectEqualStrings("GPU Work (1.50ms)", label);
}
