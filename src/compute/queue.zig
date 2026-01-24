// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Multi-Queue Management
//!
//! Provides support for multiple queue families enabling:
//! - Async compute with transfer overlap
//! - Dedicated transfer queues for data streaming
//! - Better GPU utilization through parallel queue submission
//!
//! ## Queue Types
//!
//! - **Compute Queue**: Primary queue for compute shader dispatch
//! - **Transfer Queue**: Dedicated queue for memory transfers (if available)
//! - **Async Compute Queue**: Secondary compute queue for overlapping work
//!
//! ## Example
//!
//! ```zig
//! var queue_manager = try QueueManager.init(&ctx);
//! defer queue_manager.deinit();
//!
//! // Submit transfer on dedicated queue while compute is running
//! try queue_manager.submitTransfer(transfer_cmd, null, transfer_done);
//! try queue_manager.submitCompute(compute_cmd, null, compute_done);
//!
//! // Wait for both to complete
//! try queue_manager.waitAll();
//! ```

const std = @import("std");
const types = @import("../vk/types.zig");
const loader = @import("../vk/loader.zig");
const err = @import("../vk/error.zig");
const Context = @import("context.zig").Context;
const Fence = @import("command.zig").Fence;
const Semaphore = @import("command.zig").Semaphore;
const CommandBuffer = @import("command.zig").CommandBuffer;

/// Queue family information
pub const QueueFamilyInfo = struct {
    /// Queue family index
    family_index: u32,
    /// Queue index within the family
    queue_index: u32,
    /// Queue handle
    queue: types.Queue,
    /// Command pool for this queue family
    command_pool: types.CommandPool,
    /// Capabilities
    supports_compute: bool,
    supports_transfer: bool,
    supports_graphics: bool,
    /// Number of timestamp bits (0 if timestamps not supported)
    timestamp_valid_bits: u32,
};

/// Queue selection preference
pub const QueuePreference = enum {
    /// Any queue that supports the operation
    any,
    /// Prefer dedicated queue (e.g., transfer-only queue)
    dedicated,
    /// Prefer async queue (different from main compute queue)
    async_capable,
};

/// Multi-queue manager for async operations
pub const QueueManager = struct {
    allocator: std.mem.Allocator,
    device: types.Device,
    device_dispatch: loader.DeviceDispatch,

    /// Primary compute queue (same as Context.compute_queue)
    compute_queue: QueueFamilyInfo,
    /// Dedicated transfer queue (if available, different family)
    transfer_queue: ?QueueFamilyInfo,
    /// Async compute queue (if available, same family different index or different family)
    async_compute_queue: ?QueueFamilyInfo,

    /// Whether async transfer is available (dedicated transfer queue exists)
    has_async_transfer: bool,
    /// Whether async compute is available (multiple compute queues)
    has_async_compute: bool,

    const Self = @This();

    /// Initialize the queue manager from a context
    ///
    /// Discovers available queues and creates command pools for each queue family.
    /// If multiple queues are available, enables async operations.
    pub fn init(ctx: *Context) err.Error!Self {
        var self = Self{
            .allocator = ctx.allocator,
            .device = ctx.device,
            .device_dispatch = ctx.device_dispatch,
            .compute_queue = undefined,
            .transfer_queue = null,
            .async_compute_queue = null,
            .has_async_transfer = false,
            .has_async_compute = false,
        };

        // Primary compute queue from context
        self.compute_queue = QueueFamilyInfo{
            .family_index = ctx.compute_queue_family,
            .queue_index = 0,
            .queue = ctx.compute_queue,
            .command_pool = ctx.command_pool,
            .supports_compute = true,
            .supports_transfer = true, // Compute queues always support transfer
            .supports_graphics = false,
            .timestamp_valid_bits = ctx.timestamp_valid_bits,
        };

        // Try to find dedicated transfer queue
        self.transfer_queue = try self.findDedicatedTransferQueue(ctx);
        self.has_async_transfer = self.transfer_queue != null;

        // Try to find async compute queue (secondary compute queue)
        self.async_compute_queue = try self.findAsyncComputeQueue(ctx);
        self.has_async_compute = self.async_compute_queue != null;

        return self;
    }

    /// Deinitialize and release resources
    pub fn deinit(self: *Self) void {
        // Note: We don't destroy the primary compute queue's command pool
        // as it's owned by the Context

        if (self.transfer_queue) |*tq| {
            self.device_dispatch.vkDestroyCommandPool.?(self.device, tq.command_pool, null);
        }

        if (self.async_compute_queue) |*aq| {
            // Only destroy if it has its own command pool (different family)
            if (aq.family_index != self.compute_queue.family_index) {
                self.device_dispatch.vkDestroyCommandPool.?(self.device, aq.command_pool, null);
            }
        }

        self.* = undefined;
    }

    /// Allocate a command buffer for the specified queue type
    pub fn allocateCommandBuffer(self: *Self, queue_type: QueueType) err.Error!CommandBuffer {
        const pool = switch (queue_type) {
            .compute => self.compute_queue.command_pool,
            .transfer => if (self.transfer_queue) |tq| tq.command_pool else self.compute_queue.command_pool,
            .async_compute => if (self.async_compute_queue) |aq| aq.command_pool else self.compute_queue.command_pool,
        };

        const alloc_info = types.CommandBufferAllocateInfo{
            .command_pool = pool,
            .level = .primary,
            .command_buffer_count = 1,
        };

        var handle: types.CommandBuffer = undefined;
        const result = self.device_dispatch.vkAllocateCommandBuffers.?(self.device, &alloc_info, @ptrCast(&handle));
        if (result != .success) return err.fromVkResult(result);

        return CommandBuffer{
            .handle = handle,
            .pool = pool,
            .device = self.device,
            .dispatch = self.device_dispatch,
        };
    }

    /// Submit work to the compute queue
    pub fn submitCompute(
        self: *Self,
        cmd: CommandBuffer,
        wait_semaphore: ?Semaphore,
        signal_semaphore: ?Semaphore,
        fence: ?Fence,
    ) err.Error!void {
        try self.submitToQueue(self.compute_queue.queue, cmd, wait_semaphore, signal_semaphore, fence);
    }

    /// Submit work to the transfer queue (falls back to compute queue if unavailable)
    pub fn submitTransfer(
        self: *Self,
        cmd: CommandBuffer,
        wait_semaphore: ?Semaphore,
        signal_semaphore: ?Semaphore,
        fence: ?Fence,
    ) err.Error!void {
        const queue = if (self.transfer_queue) |tq| tq.queue else self.compute_queue.queue;
        try self.submitToQueue(queue, cmd, wait_semaphore, signal_semaphore, fence);
    }

    /// Submit work to the async compute queue (falls back to main compute if unavailable)
    pub fn submitAsyncCompute(
        self: *Self,
        cmd: CommandBuffer,
        wait_semaphore: ?Semaphore,
        signal_semaphore: ?Semaphore,
        fence: ?Fence,
    ) err.Error!void {
        const queue = if (self.async_compute_queue) |aq| aq.queue else self.compute_queue.queue;
        try self.submitToQueue(queue, cmd, wait_semaphore, signal_semaphore, fence);
    }

    /// Submit to a specific queue
    fn submitToQueue(
        self: *Self,
        queue: types.Queue,
        cmd: CommandBuffer,
        wait_semaphore: ?Semaphore,
        signal_semaphore: ?Semaphore,
        fence: ?Fence,
    ) err.Error!void {
        var wait_stage = types.PipelineStageFlags{ .transfer = true };

        const submit_info = types.SubmitInfo{
            .wait_semaphore_count = if (wait_semaphore != null) 1 else 0,
            .p_wait_semaphores = if (wait_semaphore) |ws| @ptrCast(&ws.handle) else null,
            .p_wait_dst_stage_mask = if (wait_semaphore != null) @ptrCast(&wait_stage) else null,
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&cmd.handle),
            .signal_semaphore_count = if (signal_semaphore != null) 1 else 0,
            .p_signal_semaphores = if (signal_semaphore) |ss| @ptrCast(&ss.handle) else null,
        };

        const result = self.device_dispatch.vkQueueSubmit.?(
            queue,
            1,
            @ptrCast(&submit_info),
            if (fence) |f| f.handle else null,
        );
        if (result != .success) return err.fromVkResult(result);
    }

    /// Wait for all queues to become idle
    pub fn waitAll(self: *Self) err.Error!void {
        var result = self.device_dispatch.vkQueueWaitIdle.?(self.compute_queue.queue);
        if (result != .success) return err.fromVkResult(result);

        if (self.transfer_queue) |tq| {
            result = self.device_dispatch.vkQueueWaitIdle.?(tq.queue);
            if (result != .success) return err.fromVkResult(result);
        }

        if (self.async_compute_queue) |aq| {
            if (aq.queue != self.compute_queue.queue) {
                result = self.device_dispatch.vkQueueWaitIdle.?(aq.queue);
                if (result != .success) return err.fromVkResult(result);
            }
        }
    }

    /// Wait for the compute queue to become idle
    pub fn waitCompute(self: *Self) err.Error!void {
        const result = self.device_dispatch.vkQueueWaitIdle.?(self.compute_queue.queue);
        if (result != .success) return err.fromVkResult(result);
    }

    /// Wait for the transfer queue to become idle
    pub fn waitTransfer(self: *Self) err.Error!void {
        const queue = if (self.transfer_queue) |tq| tq.queue else self.compute_queue.queue;
        const result = self.device_dispatch.vkQueueWaitIdle.?(queue);
        if (result != .success) return err.fromVkResult(result);
    }

    /// Get queue capabilities
    pub fn getCapabilities(self: Self) QueueCapabilities {
        return .{
            .has_dedicated_transfer = self.has_async_transfer,
            .has_async_compute = self.has_async_compute,
            .compute_queue_count = if (self.has_async_compute) 2 else 1,
            .transfer_queue_family = if (self.transfer_queue) |tq| tq.family_index else self.compute_queue.family_index,
            .async_compute_queue_family = if (self.async_compute_queue) |aq| aq.family_index else self.compute_queue.family_index,
        };
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    fn findDedicatedTransferQueue(self: *Self, ctx: *Context) err.Error!?QueueFamilyInfo {
        // Get queue family properties
        const queue_families = ctx.instance_dispatch.getPhysicalDeviceQueueFamilyProperties(
            ctx.physical_device,
            ctx.allocator,
        ) catch return null;
        defer ctx.allocator.free(queue_families);

        // Look for a transfer-only queue (no compute, no graphics)
        for (queue_families, 0..) |qf, i| {
            const idx: u32 = @intCast(i);
            if (idx == ctx.compute_queue_family) continue; // Skip main compute family

            if (qf.queue_flags.transfer and !qf.queue_flags.compute and !qf.queue_flags.graphics) {
                // Found dedicated transfer queue
                return try self.createQueueInfo(ctx, idx, qf);
            }
        }

        // Look for any transfer queue in a different family
        for (queue_families, 0..) |qf, i| {
            const idx: u32 = @intCast(i);
            if (idx == ctx.compute_queue_family) continue;

            if (qf.queue_flags.transfer) {
                return try self.createQueueInfo(ctx, idx, qf);
            }
        }

        return null;
    }

    fn findAsyncComputeQueue(self: *Self, ctx: *Context) err.Error!?QueueFamilyInfo {
        const queue_families = ctx.instance_dispatch.getPhysicalDeviceQueueFamilyProperties(
            ctx.physical_device,
            ctx.allocator,
        ) catch return null;
        defer ctx.allocator.free(queue_families);

        // Note: The device is created with queue_count=1 in Context.init(), so we cannot
        // get a second queue from the same family. Attempting vkGetDeviceQueue with index > 0
        // when only 1 queue was created would be undefined behavior per Vulkan spec.
        // To use multiple queues from the same family, Context.init() would need to be
        // modified to request multiple queues during device creation.

        // Try to find a different compute-capable family
        for (queue_families, 0..) |qf, i| {
            const idx: u32 = @intCast(i);
            if (idx == ctx.compute_queue_family) continue;

            if (qf.queue_flags.compute) {
                return try self.createQueueInfo(ctx, idx, qf);
            }
        }

        return null;
    }

    fn createQueueInfo(self: *Self, _: *Context, family_index: u32, qf: types.QueueFamilyProperties) err.Error!QueueFamilyInfo {
        // Get queue handle
        var queue: types.Queue = undefined;
        self.device_dispatch.vkGetDeviceQueue.?(self.device, family_index, 0, &queue);

        // Create command pool for this family
        const pool_info = types.CommandPoolCreateInfo{
            .flags = .{ .reset_command_buffer = true },
            .queue_family_index = family_index,
        };

        var command_pool: types.CommandPool = undefined;
        const result = self.device_dispatch.vkCreateCommandPool.?(self.device, &pool_info, null, &command_pool);
        if (result != .success) return err.fromVkResult(result);

        return QueueFamilyInfo{
            .family_index = family_index,
            .queue_index = 0,
            .queue = queue,
            .command_pool = command_pool,
            .supports_compute = qf.queue_flags.compute,
            .supports_transfer = qf.queue_flags.transfer,
            .supports_graphics = qf.queue_flags.graphics,
            .timestamp_valid_bits = qf.timestamp_valid_bits,
        };
    }
};

/// Queue type for operations
pub const QueueType = enum {
    /// Primary compute queue
    compute,
    /// Transfer queue (dedicated if available)
    transfer,
    /// Async compute queue
    async_compute,
};

/// Queue capabilities summary
pub const QueueCapabilities = struct {
    /// Whether a dedicated transfer queue is available
    has_dedicated_transfer: bool,
    /// Whether async compute is available
    has_async_compute: bool,
    /// Number of compute queues available
    compute_queue_count: u32,
    /// Queue family index for transfer operations
    transfer_queue_family: u32,
    /// Queue family index for async compute
    async_compute_queue_family: u32,
};

/// Timeline semaphore for advanced synchronization
pub const TimelineSemaphore = struct {
    handle: types.Semaphore,
    device: types.Device,
    dispatch: loader.DeviceDispatch,
    current_value: u64,

    const Self = @This();

    /// Create a timeline semaphore with initial value
    pub fn init(ctx: *Context, initial_value: u64) err.Error!Self {
        const type_info = types.SemaphoreTypeCreateInfo{
            .semaphore_type = .timeline,
            .initial_value = initial_value,
        };

        const create_info = types.SemaphoreCreateInfo{
            .p_next = &type_info,
        };

        var handle: types.Semaphore = undefined;
        const result = ctx.device_dispatch.vkCreateSemaphore.?(ctx.device, &create_info, null, &handle);
        if (result != .success) return err.fromVkResult(result);

        return Self{
            .handle = handle,
            .device = ctx.device,
            .dispatch = ctx.device_dispatch,
            .current_value = initial_value,
        };
    }

    /// Destroy the timeline semaphore
    pub fn deinit(self: *Self) void {
        self.dispatch.vkDestroySemaphore.?(self.device, self.handle, null);
        self.* = undefined;
    }

    /// Signal the semaphore to a new value from the CPU
    pub fn signal(self: *Self, value: u64) err.Error!void {
        if (self.dispatch.vkSignalSemaphore == null) {
            return err.Error.FeatureNotPresent;
        }

        const signal_info = types.SemaphoreSignalInfo{
            .semaphore = self.handle,
            .value = value,
        };

        const result = self.dispatch.vkSignalSemaphore.?(self.device, &signal_info);
        if (result != .success) return err.fromVkResult(result);

        self.current_value = value;
    }

    /// Wait for the semaphore to reach a value
    pub fn wait(self: *Self, value: u64, timeout_ns: u64) err.Error!bool {
        if (self.dispatch.vkWaitSemaphores == null) {
            return err.Error.FeatureNotPresent;
        }

        const wait_info = types.SemaphoreWaitInfo{
            .semaphore_count = 1,
            .p_semaphores = @ptrCast(&self.handle),
            .p_values = @ptrCast(&value),
        };

        const result = self.dispatch.vkWaitSemaphores.?(self.device, &wait_info, timeout_ns);
        return switch (result) {
            .success => true,
            .timeout => false,
            else => err.fromVkResult(result),
        };
    }

    /// Get the current counter value
    pub fn getValue(self: Self) err.Error!u64 {
        if (self.dispatch.vkGetSemaphoreCounterValue == null) {
            return err.Error.FeatureNotPresent;
        }

        var value: u64 = undefined;
        const result = self.dispatch.vkGetSemaphoreCounterValue.?(self.device, self.handle, &value);
        if (result != .success) return err.fromVkResult(result);
        return value;
    }

    /// Get the next value for signaling
    pub fn nextValue(self: *Self) u64 {
        self.current_value += 1;
        return self.current_value;
    }
};

/// Async transfer operation
pub const AsyncTransfer = struct {
    queue_manager: *QueueManager,
    fence: Fence,
    command_buffer: CommandBuffer,
    completed: bool,

    const Self = @This();

    /// Check if the transfer is complete
    pub fn isComplete(self: *Self) bool {
        if (self.completed) return true;
        self.completed = self.fence.isSignaled();
        return self.completed;
    }

    /// Wait for the transfer to complete
    pub fn wait(self: *Self) err.Error!void {
        if (self.completed) return;
        try self.fence.waitForever();
        self.completed = true;
    }

    /// Clean up resources
    pub fn deinit(self: *Self) void {
        var fence = self.fence;
        fence.deinit();
        var cmd = self.command_buffer;
        cmd.deinit();
        self.* = undefined;
    }
};

// =============================================================================
// Cross-Queue Synchronization with Ownership Transfer
// =============================================================================

/// Buffer ownership transfer between queue families
pub const BufferOwnershipTransfer = struct {
    /// Source queue family index
    src_queue_family: u32,
    /// Destination queue family index
    dst_queue_family: u32,
    /// Buffer handle
    buffer: types.Buffer,
    /// Offset within buffer
    offset: u64,
    /// Size of region (VK_WHOLE_SIZE for entire buffer)
    size: u64,

    /// Check if ownership transfer is actually needed
    pub fn isNeeded(self: BufferOwnershipTransfer) bool {
        return self.src_queue_family != self.dst_queue_family;
    }
};

/// Image ownership transfer between queue families
pub const ImageOwnershipTransfer = struct {
    /// Source queue family index
    src_queue_family: u32,
    /// Destination queue family index
    dst_queue_family: u32,
    /// Image handle
    image: types.Image,
    /// Image layout before transfer
    old_layout: types.ImageLayout,
    /// Image layout after transfer
    new_layout: types.ImageLayout,
    /// Subresource range
    subresource_range: types.ImageSubresourceRange,

    /// Check if ownership transfer is actually needed
    pub fn isNeeded(self: ImageOwnershipTransfer) bool {
        return self.src_queue_family != self.dst_queue_family;
    }
};

/// Cross-queue synchronization coordinator
///
/// Manages buffer and image ownership transfers between queue families
/// with proper barrier insertion and synchronization.
///
/// ## Usage
///
/// ```zig
/// var sync = QueueSync.init(allocator);
/// defer sync.deinit();
///
/// // Transfer buffer ownership from compute to transfer queue
/// try sync.addBufferTransfer(.{
///     .src_queue_family = compute_family,
///     .dst_queue_family = transfer_family,
///     .buffer = my_buffer,
///     .offset = 0,
///     .size = VK_WHOLE_SIZE,
/// });
///
/// // Record release barrier on source queue
/// try sync.recordReleaseBarriers(&src_cmd);
///
/// // Submit and signal semaphore...
///
/// // Record acquire barrier on destination queue
/// try sync.recordAcquireBarriers(&dst_cmd);
/// ```
pub const QueueSync = struct {
    allocator: std.mem.Allocator,
    buffer_transfers: std.ArrayListUnmanaged(BufferOwnershipTransfer),
    image_transfers: std.ArrayListUnmanaged(ImageOwnershipTransfer),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .buffer_transfers = .{},
            .image_transfers = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.buffer_transfers.deinit(self.allocator);
        self.image_transfers.deinit(self.allocator);
    }

    /// Add a buffer ownership transfer
    pub fn addBufferTransfer(self: *Self, transfer: BufferOwnershipTransfer) !void {
        if (transfer.isNeeded()) {
            try self.buffer_transfers.append(self.allocator, transfer);
        }
    }

    /// Add an image ownership transfer
    pub fn addImageTransfer(self: *Self, transfer: ImageOwnershipTransfer) !void {
        if (transfer.isNeeded()) {
            try self.image_transfers.append(self.allocator, transfer);
        }
    }

    /// Clear all pending transfers
    pub fn clear(self: *Self) void {
        self.buffer_transfers.clearRetainingCapacity();
        self.image_transfers.clearRetainingCapacity();
    }

    /// Record release barriers on source command buffer
    /// Call this before submitting to the source queue
    pub fn recordReleaseBarriers(
        self: Self,
        cmd: *CommandBuffer,
        src_access: types.AccessFlags,
        _: types.AccessFlags, // dst_access not used for release barriers
        src_stage: types.PipelineStageFlags,
        dst_stage: types.PipelineStageFlags,
    ) void {
        if (self.buffer_transfers.items.len == 0 and self.image_transfers.items.len == 0) return;

        // Process barriers in batches to handle large transfer counts
        const BATCH_SIZE = 16;

        // Process buffer barriers in batches
        var buf_offset: usize = 0;
        while (buf_offset < self.buffer_transfers.items.len) {
            var buffer_barriers: [BATCH_SIZE]types.BufferMemoryBarrier = undefined;
            const remaining = self.buffer_transfers.items.len - buf_offset;
            const batch_count = @min(remaining, BATCH_SIZE);

            for (self.buffer_transfers.items[buf_offset..][0..batch_count], 0..) |t, i| {
                buffer_barriers[i] = types.BufferMemoryBarrier{
                    .src_access_mask = src_access,
                    .dst_access_mask = .{}, // No access needed for release
                    .src_queue_family_index = t.src_queue_family,
                    .dst_queue_family_index = t.dst_queue_family,
                    .buffer = t.buffer,
                    .offset = t.offset,
                    .size = t.size,
                };
            }

            cmd.dispatch.vkCmdPipelineBarrier.?(
                cmd.handle,
                src_stage,
                dst_stage,
                .{},
                0,
                null,
                @intCast(batch_count),
                &buffer_barriers,
                0,
                null,
            );

            buf_offset += batch_count;
        }

        // Process image barriers in batches
        var img_offset: usize = 0;
        while (img_offset < self.image_transfers.items.len) {
            var image_barriers: [BATCH_SIZE]types.ImageMemoryBarrier = undefined;
            const remaining = self.image_transfers.items.len - img_offset;
            const batch_count = @min(remaining, BATCH_SIZE);

            for (self.image_transfers.items[img_offset..][0..batch_count], 0..) |t, i| {
                image_barriers[i] = types.ImageMemoryBarrier{
                    .src_access_mask = src_access,
                    .dst_access_mask = .{}, // No access needed for release
                    .old_layout = t.old_layout,
                    .new_layout = t.old_layout, // Keep same layout for release
                    .src_queue_family_index = t.src_queue_family,
                    .dst_queue_family_index = t.dst_queue_family,
                    .image = t.image,
                    .subresource_range = t.subresource_range,
                };
            }

            cmd.dispatch.vkCmdPipelineBarrier.?(
                cmd.handle,
                src_stage,
                dst_stage,
                .{},
                0,
                null,
                0,
                null,
                @intCast(batch_count),
                &image_barriers,
            );

            img_offset += batch_count;
        }
    }

    /// Record acquire barriers on destination command buffer
    /// Call this after waiting for the source queue's semaphore
    pub fn recordAcquireBarriers(
        self: Self,
        cmd: *CommandBuffer,
        _: types.AccessFlags, // src_access not used for acquire barriers
        dst_access: types.AccessFlags,
        src_stage: types.PipelineStageFlags,
        dst_stage: types.PipelineStageFlags,
    ) void {
        if (self.buffer_transfers.items.len == 0 and self.image_transfers.items.len == 0) return;

        // Process barriers in batches to handle large transfer counts
        const BATCH_SIZE = 16;

        // Process buffer barriers in batches
        var buf_offset: usize = 0;
        while (buf_offset < self.buffer_transfers.items.len) {
            var buffer_barriers: [BATCH_SIZE]types.BufferMemoryBarrier = undefined;
            const remaining = self.buffer_transfers.items.len - buf_offset;
            const batch_count = @min(remaining, BATCH_SIZE);

            for (self.buffer_transfers.items[buf_offset..][0..batch_count], 0..) |t, i| {
                buffer_barriers[i] = types.BufferMemoryBarrier{
                    .src_access_mask = .{}, // No access needed for acquire
                    .dst_access_mask = dst_access,
                    .src_queue_family_index = t.src_queue_family,
                    .dst_queue_family_index = t.dst_queue_family,
                    .buffer = t.buffer,
                    .offset = t.offset,
                    .size = t.size,
                };
            }

            cmd.dispatch.vkCmdPipelineBarrier.?(
                cmd.handle,
                src_stage,
                dst_stage,
                .{},
                0,
                null,
                @intCast(batch_count),
                &buffer_barriers,
                0,
                null,
            );

            buf_offset += batch_count;
        }

        // Process image barriers in batches
        var img_offset: usize = 0;
        while (img_offset < self.image_transfers.items.len) {
            var image_barriers: [BATCH_SIZE]types.ImageMemoryBarrier = undefined;
            const remaining = self.image_transfers.items.len - img_offset;
            const batch_count = @min(remaining, BATCH_SIZE);

            for (self.image_transfers.items[img_offset..][0..batch_count], 0..) |t, i| {
                image_barriers[i] = types.ImageMemoryBarrier{
                    .src_access_mask = .{}, // No access needed for acquire
                    .dst_access_mask = dst_access,
                    .old_layout = t.old_layout,
                    .new_layout = t.new_layout, // Apply layout transition
                    .src_queue_family_index = t.src_queue_family,
                    .dst_queue_family_index = t.dst_queue_family,
                    .image = t.image,
                    .subresource_range = t.subresource_range,
                };
            }

            cmd.dispatch.vkCmdPipelineBarrier.?(
                cmd.handle,
                src_stage,
                dst_stage,
                .{},
                0,
                null,
                0,
                null,
                @intCast(batch_count),
                &image_barriers,
            );

            img_offset += batch_count;
        }
    }

    /// Check if any transfers are pending
    pub fn hasPendingTransfers(self: Self) bool {
        return self.buffer_transfers.items.len > 0 or self.image_transfers.items.len > 0;
    }
};

/// High-level cross-queue operation coordinator
///
/// Simplifies the process of transferring work between queues with
/// automatic barrier management and semaphore synchronization.
pub const CrossQueueCoordinator = struct {
    queue_manager: *QueueManager,
    sync: QueueSync,
    transfer_semaphore: ?Semaphore,

    const Self = @This();

    pub fn init(queue_manager: *QueueManager, allocator: std.mem.Allocator) Self {
        return Self{
            .queue_manager = queue_manager,
            .sync = QueueSync.init(allocator),
            .transfer_semaphore = null,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.transfer_semaphore) |*sem| {
            sem.deinit();
        }
        self.sync.deinit();
    }

    /// Get or create the transfer semaphore
    pub fn getSemaphore(self: *Self, ctx: *Context) err.Error!Semaphore {
        if (self.transfer_semaphore) |sem| {
            return sem;
        }

        const create_info = types.SemaphoreCreateInfo{};
        var handle: types.Semaphore = undefined;
        const result = ctx.device_dispatch.vkCreateSemaphore.?(ctx.device, &create_info, null, &handle);
        if (result != .success) return err.fromVkResult(result);

        self.transfer_semaphore = Semaphore{
            .handle = handle,
            .device = ctx.device,
            .dispatch = ctx.device_dispatch,
        };

        return self.transfer_semaphore.?;
    }

    /// Execute a buffer transfer from compute queue to transfer queue
    pub fn transferToTransferQueue(
        self: *Self,
        ctx: *Context,
        buffer: types.Buffer,
        src_cmd: *CommandBuffer,
        dst_cmd: *CommandBuffer,
    ) err.Error!void {
        const caps = self.queue_manager.getCapabilities();

        try self.sync.addBufferTransfer(.{
            .src_queue_family = ctx.compute_queue_family,
            .dst_queue_family = caps.transfer_queue_family,
            .buffer = buffer,
            .offset = 0,
            .size = std.math.maxInt(u64), // VK_WHOLE_SIZE
        });

        // Record release barrier on compute queue
        self.sync.recordReleaseBarriers(
            src_cmd,
            .{ .shader_write = true },
            .{ .transfer_read = true },
            .{ .compute_shader = true },
            .{ .transfer = true },
        );

        // Record acquire barrier on transfer queue
        self.sync.recordAcquireBarriers(
            dst_cmd,
            .{ .shader_write = true },
            .{ .transfer_read = true },
            .{ .compute_shader = true },
            .{ .transfer = true },
        );

        self.sync.clear();
    }

    /// Execute a buffer transfer from transfer queue back to compute queue
    pub fn transferToComputeQueue(
        self: *Self,
        ctx: *Context,
        buffer: types.Buffer,
        src_cmd: *CommandBuffer,
        dst_cmd: *CommandBuffer,
    ) err.Error!void {
        const caps = self.queue_manager.getCapabilities();

        try self.sync.addBufferTransfer(.{
            .src_queue_family = caps.transfer_queue_family,
            .dst_queue_family = ctx.compute_queue_family,
            .buffer = buffer,
            .offset = 0,
            .size = std.math.maxInt(u64), // VK_WHOLE_SIZE
        });

        // Record release barrier on transfer queue
        self.sync.recordReleaseBarriers(
            src_cmd,
            .{ .transfer_write = true },
            .{ .shader_read = true },
            .{ .transfer = true },
            .{ .compute_shader = true },
        );

        // Record acquire barrier on compute queue
        self.sync.recordAcquireBarriers(
            dst_cmd,
            .{ .transfer_write = true },
            .{ .shader_read = true },
            .{ .transfer = true },
            .{ .compute_shader = true },
        );

        self.sync.clear();
    }
};

// =============================================================================
// Tests
// =============================================================================

test "BufferOwnershipTransfer isNeeded" {
    const transfer_needed = BufferOwnershipTransfer{
        .src_queue_family = 0,
        .dst_queue_family = 1,
        .buffer = undefined,
        .offset = 0,
        .size = 1024,
    };
    try std.testing.expect(transfer_needed.isNeeded());

    const transfer_not_needed = BufferOwnershipTransfer{
        .src_queue_family = 0,
        .dst_queue_family = 0,
        .buffer = undefined,
        .offset = 0,
        .size = 1024,
    };
    try std.testing.expect(!transfer_not_needed.isNeeded());
}

test "QueueSync init/deinit" {
    var sync = QueueSync.init(std.testing.allocator);
    defer sync.deinit();

    try std.testing.expect(!sync.hasPendingTransfers());
}

test "QueueType enum values" {
    try std.testing.expectEqual(@intFromEnum(QueueType.compute), 0);
    try std.testing.expectEqual(@intFromEnum(QueueType.transfer), 1);
    try std.testing.expectEqual(@intFromEnum(QueueType.async_compute), 2);
}

test "QueueCapabilities defaults" {
    const caps = QueueCapabilities{
        .has_dedicated_transfer = false,
        .has_async_compute = false,
        .compute_queue_count = 1,
        .transfer_queue_family = 0,
        .async_compute_queue_family = 0,
    };
    try std.testing.expect(!caps.has_dedicated_transfer);
    try std.testing.expect(!caps.has_async_compute);
}
