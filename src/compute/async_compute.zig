// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Async Compute Coordination
//!
//! Higher-level primitives for overlapping compute and transfer operations
//! with automatic synchronization and resource tracking.
//!
//! ## Features
//!
//! - Automatic barrier insertion between dependent operations
//! - Overlapping compute/transfer on separate queues
//! - Timeline semaphore coordination
//! - Double/triple buffering support
//! - Streaming compute patterns
//!
//! ## Example
//!
//! ```zig
//! const vulkan = @import("vulkan");
//!
//! // Create an async compute coordinator
//! var coordinator = try vulkan.AsyncCoordinator.init(&ctx, allocator, .{
//!     .frame_count = 2,  // Double buffering
//! });
//! defer coordinator.deinit();
//!
//! // Submit overlapping work
//! coordinator.beginFrame();
//!
//! // Upload data (transfer queue)
//! coordinator.scheduleTransfer(staging_buf, gpu_buf, .upload);
//!
//! // Compute work (compute queue, waits on transfer)
//! coordinator.scheduleCompute(pipeline, descriptor_set, workgroups);
//!
//! // Download results (transfer queue, waits on compute)
//! coordinator.scheduleTransfer(gpu_buf, readback_buf, .download);
//!
//! coordinator.endFrame();
//! ```

const std = @import("std");
const types = @import("../vk/types.zig");
const loader = @import("../vk/loader.zig");
const err = @import("../vk/error.zig");
const Context = @import("context.zig").Context;
const command = @import("command.zig");
const queue_mod = @import("queue.zig");
const Buffer = @import("buffer.zig").Buffer;

/// Async coordinator options
pub const CoordinatorOptions = struct {
    /// Number of frames for buffering (2 = double, 3 = triple)
    frame_count: u32 = 2,
    /// Use timeline semaphores for coordination (if available)
    use_timeline_semaphores: bool = true,
    /// Maximum pending operations per frame
    max_pending_ops: u32 = 64,
    /// Prefer async transfer queue
    prefer_async_transfer: bool = true,
};

/// Operation type for scheduling
pub const OperationType = enum {
    /// Data upload (CPU -> GPU)
    upload,
    /// Data download (GPU -> CPU)
    download,
    /// GPU-to-GPU copy
    copy,
    /// Compute dispatch
    compute,
    /// Memory barrier
    barrier,
};

/// Resource state for tracking
pub const ResourceState = enum {
    undefined,
    transfer_src,
    transfer_dst,
    compute_read,
    compute_write,
    compute_read_write,
    host_read,
    host_write,
};

/// Pending operation
pub const PendingOperation = struct {
    op_type: OperationType,
    /// Timeline value this operation produces
    signal_value: u64 = 0,
    /// Timeline values this operation waits on
    wait_values: [4]u64 = .{ 0, 0, 0, 0 },
    wait_count: u32 = 0,
    /// Resources involved
    src_buffer: ?*const Buffer = null,
    dst_buffer: ?*const Buffer = null,
    /// Size and offset for transfers
    size: u64 = 0,
    offset: u64 = 0,
    /// Workgroup counts for compute
    workgroups: [3]u32 = .{ 1, 1, 1 },
};

/// Frame synchronization data
pub const FrameSync = struct {
    /// Command buffers for this frame
    compute_cmd: ?command.CommandBuffer = null,
    transfer_cmd: ?command.CommandBuffer = null,
    /// Fence for frame completion
    fence: ?command.Fence = null,
    /// Timeline semaphore value for this frame
    timeline_value: u64 = 0,
    /// Whether this frame is in flight
    in_flight: bool = false,
};

/// Async compute coordinator
pub const AsyncCoordinator = struct {
    ctx: *Context,
    allocator: std.mem.Allocator,
    options: CoordinatorOptions,

    /// Frame synchronization data
    frames: []FrameSync,
    /// Current frame index
    current_frame: u32 = 0,
    /// Global timeline counter
    timeline_counter: u64 = 0,

    /// Timeline semaphore for coordination
    timeline_semaphore: ?queue_mod.TimelineSemaphore = null,

    /// Pending operations for current frame
    pending_ops: std.ArrayListUnmanaged(PendingOperation),

    /// Resource state tracking
    resource_states: std.AutoHashMapUnmanaged(usize, ResourceState),

    /// Transfer queue (if available separately)
    transfer_queue: ?types.Queue = null,
    has_async_transfer: bool = false,

    /// Deadlock detection: track which queue is waiting on which resource
    /// Key: resource ID, Value: queue index that owns it
    resource_owners: std.AutoHashMapUnmanaged(usize, u32),

    const Self = @This();

    /// Initialize the async coordinator
    pub fn init(ctx: *Context, allocator: std.mem.Allocator, options: CoordinatorOptions) !Self {
        var self = Self{
            .ctx = ctx,
            .allocator = allocator,
            .options = options,
            .frames = undefined,
            .pending_ops = .{},
            .resource_states = .{},
            .resource_owners = .{},
        };

        // Allocate frame sync data
        self.frames = try allocator.alloc(FrameSync, options.frame_count);
        errdefer allocator.free(self.frames);

        // Initialize frames
        for (self.frames) |*frame| {
            frame.* = .{};
            frame.fence = try command.Fence.init(ctx, true); // Signaled initially
        }

        // Create timeline semaphore if requested and available
        if (options.use_timeline_semaphores) {
            self.timeline_semaphore = queue_mod.TimelineSemaphore.init(ctx, 0) catch null;
        }

        // Check for async transfer queue
        self.checkAsyncTransfer();

        return self;
    }

    /// Deinitialize the coordinator
    /// Note: Cleanup proceeds even if waitAll fails to prevent resource leaks.
    pub fn deinit(self: *Self) void {
        // Wait for all frames to complete
        self.waitAll() catch |e| {
            std.log.warn("AsyncComputeCoordinator.deinit: waitAll failed: {s}", .{@errorName(e)});
        };

        // Free frame resources
        for (self.frames) |*frame| {
            if (frame.compute_cmd) |*cmd| cmd.deinit();
            if (frame.transfer_cmd) |*cmd| cmd.deinit();
            if (frame.fence) |*fence| fence.deinit();
        }
        self.allocator.free(self.frames);

        if (self.timeline_semaphore) |*sem| {
            sem.deinit();
        }

        self.pending_ops.deinit(self.allocator);
        self.resource_states.deinit(self.allocator);
        self.resource_owners.deinit(self.allocator);
        self.* = undefined;
    }

    /// Check for potential deadlocks in the pending operations
    /// A deadlock can occur when:
    /// - Queue A is waiting for resource R1 owned by Queue B
    /// - Queue B is waiting for resource R2 owned by Queue A
    ///
    /// Returns true if a potential deadlock is detected
    pub fn detectDeadlock(self: *Self) bool {
        // Build dependency graph: which queue is waiting on which other queue
        // queue_waits_on[q1] = set of queues that q1 is waiting on
        const max_queues: usize = 4; // compute, transfer, sparse binding, etc.
        var waits_on: [max_queues]std.StaticBitSet(max_queues) = .{std.StaticBitSet(max_queues).initEmpty()} ** max_queues;

        // Analyze pending operations to build wait graph
        for (self.pending_ops.items) |op| {
            const src_queue: u32 = if (op.op_type == .upload or op.op_type == .download or op.op_type == .copy)
                1 // transfer queue
            else
                0; // compute queue

            // Check what resources this operation needs
            if (op.src_buffer) |buf| {
                const buf_id = @intFromPtr(buf);
                if (self.resource_owners.get(buf_id)) |owner_queue| {
                    if (owner_queue != src_queue) {
                        // This queue waits on another queue
                        waits_on[src_queue].set(owner_queue);
                    }
                }
            }
            if (op.dst_buffer) |buf| {
                const buf_id = @intFromPtr(buf);
                if (self.resource_owners.get(buf_id)) |owner_queue| {
                    if (owner_queue != src_queue) {
                        waits_on[src_queue].set(owner_queue);
                    }
                }
            }
        }

        // Check for cycles: if A waits on B and B waits on A, that's a deadlock
        for (0..max_queues) |q1| {
            for (0..max_queues) |q2| {
                if (q1 != q2 and waits_on[q1].isSet(q2) and waits_on[q2].isSet(q1)) {
                    std.log.err("[AsyncCoordinator] Potential deadlock detected!", .{});
                    std.log.err("  Queue {d} waits on Queue {d}", .{ q1, q2 });
                    std.log.err("  Queue {d} waits on Queue {d}", .{ q2, q1 });
                    std.log.err("  This creates a circular wait condition", .{});
                    return true;
                }
            }
        }

        return false;
    }

    /// Claim ownership of a resource for a specific queue
    /// Used for deadlock detection
    pub fn claimResource(self: *Self, resource_id: usize, queue_id: u32) !void {
        try self.resource_owners.put(self.allocator, resource_id, queue_id);
    }

    /// Release ownership of a resource
    pub fn releaseResource(self: *Self, resource_id: usize) void {
        _ = self.resource_owners.remove(resource_id);
    }

    fn checkAsyncTransfer(self: *Self) void {
        // Check if we have a dedicated transfer queue
        // This would query queue families in a full implementation
        self.has_async_transfer = self.options.prefer_async_transfer and
            self.ctx.transfer_queue != null;
        if (self.has_async_transfer) {
            self.transfer_queue = self.ctx.transfer_queue;
        }
    }

    /// Begin a new frame
    pub fn beginFrame(self: *Self) !void {
        const frame = &self.frames[self.current_frame];

        // Wait for this frame slot to be available
        if (frame.in_flight) {
            if (frame.fence) |fence| {
                try fence.waitForever();
                try fence.reset();
            }
            frame.in_flight = false;
        }

        // Reset command buffers
        if (frame.compute_cmd) |*cmd| {
            try cmd.reset();
        } else {
            frame.compute_cmd = try command.CommandBuffer.init(self.ctx);
        }

        if (self.has_async_transfer) {
            if (frame.transfer_cmd) |*cmd| {
                try cmd.reset();
            } else {
                frame.transfer_cmd = try command.CommandBuffer.init(self.ctx);
            }
        }

        // Begin recording
        if (frame.compute_cmd) |cmd| {
            try cmd.begin();
        }
        if (frame.transfer_cmd) |cmd| {
            try cmd.begin();
        }

        // Clear pending ops
        self.pending_ops.clearRetainingCapacity();
    }

    /// Schedule a buffer transfer
    pub fn scheduleTransfer(
        self: *Self,
        src: *const Buffer,
        dst: *const Buffer,
        op: OperationType,
    ) !void {
        try self.scheduleTransferRegion(src, dst, op, 0, @min(src.size, dst.size));
    }

    /// Schedule a partial buffer transfer
    pub fn scheduleTransferRegion(
        self: *Self,
        src: *const Buffer,
        dst: *const Buffer,
        op: OperationType,
        offset: u64,
        size: u64,
    ) !void {
        const frame = &self.frames[self.current_frame];

        // Get the appropriate command buffer
        const cmd = if (self.has_async_transfer and op != .compute)
            frame.transfer_cmd orelse frame.compute_cmd
        else
            frame.compute_cmd;

        if (cmd == null) {
            std.log.warn("AsyncCompute.recordCopy: no command buffer available for frame {d}", .{self.current_frame});
            return err.Error.NotReady;
        }

        // Insert barrier if needed based on resource state
        try self.insertBarrierIfNeeded(cmd.?.handle, src, .transfer_src);
        try self.insertBarrierIfNeeded(cmd.?.handle, dst, .transfer_dst);

        // Record the copy command
        const copy_region = types.BufferCopy{
            .src_offset = offset,
            .dst_offset = offset,
            .size = size,
        };

        self.ctx.device_dispatch.vkCmdCopyBuffer.?(
            cmd.?.handle,
            src.handle,
            dst.handle,
            1,
            @ptrCast(&copy_region),
        );

        // Update resource states
        try self.resource_states.put(self.allocator, @intFromPtr(src), .transfer_src);
        try self.resource_states.put(self.allocator, @intFromPtr(dst), .transfer_dst);

        // Track pending operation
        self.timeline_counter += 1;
        try self.pending_ops.append(self.allocator, .{
            .op_type = op,
            .signal_value = self.timeline_counter,
            .src_buffer = src,
            .dst_buffer = dst,
            .size = size,
            .offset = offset,
        });
    }

    /// Schedule a compute dispatch
    pub fn scheduleCompute(
        self: *Self,
        workgroups_x: u32,
        workgroups_y: u32,
        workgroups_z: u32,
    ) !void {
        const frame = &self.frames[self.current_frame];
        const cmd = frame.compute_cmd orelse return err.Error.Unknown;

        // The caller should have already bound pipeline and descriptors
        self.ctx.device_dispatch.vkCmdDispatch.?(
            cmd.handle,
            workgroups_x,
            workgroups_y,
            workgroups_z,
        );

        // Track pending operation
        self.timeline_counter += 1;
        try self.pending_ops.append(self.allocator, .{
            .op_type = .compute,
            .signal_value = self.timeline_counter,
            .workgroups = .{ workgroups_x, workgroups_y, workgroups_z },
        });
    }

    /// Insert a memory barrier
    pub fn barrier(self: *Self, src_stage: types.PipelineStageFlags, dst_stage: types.PipelineStageFlags) void {
        const frame = &self.frames[self.current_frame];
        const cmd = frame.compute_cmd orelse return;

        const memory_barrier = types.MemoryBarrier{
            .src_access_mask = .{ .shader_write = true, .transfer_write = true },
            .dst_access_mask = .{ .shader_read = true, .transfer_read = true },
        };

        self.ctx.device_dispatch.vkCmdPipelineBarrier.?(
            cmd.handle,
            src_stage,
            dst_stage,
            0,
            1,
            @ptrCast(&memory_barrier),
            0,
            null,
            0,
            null,
        );
    }

    /// Insert barrier if resource state requires it
    fn insertBarrierIfNeeded(
        self: *Self,
        cmd: types.CommandBuffer,
        buffer: *const Buffer,
        target_state: ResourceState,
    ) !void {
        const current_state = self.resource_states.get(@intFromPtr(buffer)) orelse .undefined;

        // Determine if barrier is needed
        const needs_barrier = switch (current_state) {
            .undefined => false,
            .transfer_src => target_state != .transfer_src,
            .transfer_dst => true, // Always barrier after write
            .compute_write, .compute_read_write => true, // Always barrier after write
            .compute_read => target_state == .compute_write or target_state == .transfer_dst,
            .host_write => true,
            .host_read => target_state == .host_write or target_state == .transfer_dst,
        };

        if (needs_barrier) {
            const buffer_barrier = types.BufferMemoryBarrier{
                .src_access_mask = self.stateToAccessMask(current_state, true),
                .dst_access_mask = self.stateToAccessMask(target_state, false),
                .src_queue_family_index = types.QUEUE_FAMILY_IGNORED,
                .dst_queue_family_index = types.QUEUE_FAMILY_IGNORED,
                .buffer = buffer.handle,
                .offset = 0,
                .size = types.WHOLE_SIZE,
            };

            self.ctx.device_dispatch.vkCmdPipelineBarrier.?(
                cmd,
                self.stateToStageFlags(current_state),
                self.stateToStageFlags(target_state),
                0,
                0,
                null,
                1,
                @ptrCast(&buffer_barrier),
                0,
                null,
            );
        }
    }

    fn stateToAccessMask(self: Self, state: ResourceState, is_src: bool) types.AccessFlags {
        _ = self;
        _ = is_src;
        return switch (state) {
            .undefined => .{},
            .transfer_src => .{ .transfer_read = true },
            .transfer_dst => .{ .transfer_write = true },
            .compute_read => .{ .shader_read = true },
            .compute_write => .{ .shader_write = true },
            .compute_read_write => .{ .shader_read = true, .shader_write = true },
            .host_read => .{ .host_read = true },
            .host_write => .{ .host_write = true },
        };
    }

    fn stateToStageFlags(self: Self, state: ResourceState) types.PipelineStageFlags {
        _ = self;
        return switch (state) {
            .undefined => .{ .top_of_pipe = true },
            .transfer_src, .transfer_dst => .{ .transfer = true },
            .compute_read, .compute_write, .compute_read_write => .{ .compute_shader = true },
            .host_read, .host_write => .{ .host = true },
        };
    }

    /// End the current frame and submit
    pub fn endFrame(self: *Self) !void {
        const frame = &self.frames[self.current_frame];

        // End command buffer recording
        if (frame.compute_cmd) |cmd| {
            try cmd.end();
        }
        if (frame.transfer_cmd) |cmd| {
            try cmd.end();
        }

        // Submit with synchronization
        if (self.has_async_transfer) {
            // Submit transfer first, then compute
            try self.submitTransfer(frame);
            try self.submitCompute(frame);
        } else {
            // Just submit compute (which includes transfers)
            try self.submitCompute(frame);
        }

        frame.in_flight = true;

        // Advance to next frame
        self.current_frame = (self.current_frame + 1) % @as(u32, @intCast(self.frames.len));
    }

    fn submitTransfer(self: *Self, frame: *FrameSync) !void {
        if (frame.transfer_cmd == null) {
            // No transfer commands recorded this frame - skip submit
            return;
        }

        const submit_info = types.SubmitInfo{
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&frame.transfer_cmd.?.handle),
        };

        const result = self.ctx.device_dispatch.vkQueueSubmit.?(
            self.transfer_queue orelse self.ctx.compute_queue,
            1,
            @ptrCast(&submit_info),
            null,
        );
        if (result != .success) return err.fromVkResult(result);
    }

    fn submitCompute(self: *Self, frame: *FrameSync) !void {
        if (frame.compute_cmd == null) {
            // No compute commands recorded this frame - skip submit
            return;
        }

        const submit_info = types.SubmitInfo{
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&frame.compute_cmd.?.handle),
        };

        const result = self.ctx.device_dispatch.vkQueueSubmit.?(
            self.ctx.compute_queue,
            1,
            @ptrCast(&submit_info),
            if (frame.fence) |f| f.handle else null,
        );
        if (result != .success) return err.fromVkResult(result);
    }

    /// Wait for all pending work to complete
    pub fn waitAll(self: *Self) !void {
        for (self.frames) |*frame| {
            if (frame.in_flight) {
                if (frame.fence) |fence| {
                    try fence.waitForever();
                }
                frame.in_flight = false;
            }
        }
    }

    /// Wait for the current frame to complete
    pub fn waitCurrentFrame(self: *Self) !void {
        const frame = &self.frames[self.current_frame];
        if (frame.in_flight) {
            if (frame.fence) |fence| {
                try fence.waitForever();
            }
        }
    }

    /// Get the number of frames in flight
    pub fn framesInFlight(self: Self) u32 {
        var count: u32 = 0;
        for (self.frames) |frame| {
            if (frame.in_flight) count += 1;
        }
        return count;
    }

    /// Get the current frame index
    pub fn getCurrentFrame(self: Self) u32 {
        return self.current_frame;
    }

    /// Check if async transfer is being used
    pub fn hasAsyncTransfer(self: Self) bool {
        return self.has_async_transfer;
    }
};

/// Simple double-buffer helper for streaming compute
pub const StreamBuffer = struct {
    buffers: [2]Buffer,
    current: u32 = 0,

    const Self = @This();

    /// Get the buffer for reading (previous frame's output)
    pub fn readBuffer(self: *Self) *Buffer {
        return &self.buffers[1 - self.current];
    }

    /// Get the buffer for writing (current frame's output)
    pub fn writeBuffer(self: *Self) *Buffer {
        return &self.buffers[self.current];
    }

    /// Swap buffers
    pub fn swap(self: *Self) void {
        self.current = 1 - self.current;
    }
};

// =============================================================================
// Signal/Wait Abstraction
// =============================================================================

/// Named synchronization point
pub const SyncPoint = struct {
    name: []const u8,
    timeline_value: u64,
    frame_index: u32,
    signaled: bool = false,
};

/// Signal/Wait event for high-level synchronization
pub const SyncEvent = struct {
    /// Unique event ID
    id: u32,
    /// Human-readable name
    name: []const u8,
    /// Associated sync point
    sync_point: SyncPoint,
    /// Operations that must complete before this event
    dependencies: std.ArrayListUnmanaged(u32),
};

/// High-level synchronization manager with signal/wait semantics
pub const SyncManager = struct {
    allocator: std.mem.Allocator,
    ctx: *Context,
    coordinator: *AsyncCoordinator,

    /// Named sync points
    sync_points: std.StringHashMapUnmanaged(SyncPoint),

    /// Pending signals
    pending_signals: std.ArrayListUnmanaged(SyncPoint),

    /// Pending waits
    pending_waits: std.ArrayListUnmanaged(struct {
        point_name: []const u8,
        /// Callback that receives optional user_data - implementations must handle null safely
        callback: ?*const fn (?*anyopaque) void,
        user_data: ?*anyopaque,
    }),

    /// Global timeline counter
    next_timeline: u64 = 1,

    const Self = @This();

    /// Initialize the sync manager
    pub fn init(allocator: std.mem.Allocator, ctx: *Context, coordinator: *AsyncCoordinator) Self {
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .coordinator = coordinator,
            .sync_points = .{},
            .pending_signals = .{},
            .pending_waits = .{},
        };
    }

    /// Deinitialize
    pub fn deinit(self: *Self) void {
        self.sync_points.deinit(self.allocator);
        self.pending_signals.deinit(self.allocator);
        self.pending_waits.deinit(self.allocator);
    }

    /// Create a new sync point (signals when current work completes)
    pub fn signal(self: *Self, name: []const u8) !void {
        const timeline_value = self.next_timeline;
        self.next_timeline += 1;

        const point = SyncPoint{
            .name = name,
            .timeline_value = timeline_value,
            .frame_index = self.coordinator.current_frame,
            .signaled = false,
        };

        try self.sync_points.put(self.allocator, name, point);
        try self.pending_signals.append(self.allocator, point);

        // Insert memory barrier to ensure visibility
        self.coordinator.barrier(
            .{ .compute_shader = true, .transfer = true },
            .{ .compute_shader = true, .transfer = true },
        );
    }

    /// Wait for a sync point to be signaled
    pub fn wait(self: *Self, name: []const u8) !void {
        const point = self.sync_points.get(name) orelse return error.SyncPointNotFound;

        if (point.signaled) {
            return; // Already signaled, no need to wait
        }

        // If we're in the same frame, just insert a barrier
        if (point.frame_index == self.coordinator.current_frame) {
            self.coordinator.barrier(
                .{ .compute_shader = true, .transfer = true },
                .{ .compute_shader = true, .transfer = true },
            );
            return;
        }

        // Cross-frame wait - use timeline semaphore if available
        if (self.coordinator.timeline_semaphore) |sem| {
            try sem.waitHost(point.timeline_value, std.math.maxInt(u64));
        } else {
            // Fall back to fence wait
            try self.coordinator.waitAll();
        }
    }

    /// Wait with callback (non-blocking registration)
    /// The callback receives optional user_data and must handle null safely.
    pub fn waitAsync(
        self: *Self,
        name: []const u8,
        callback: *const fn (?*anyopaque) void,
        user_data: ?*anyopaque,
    ) !void {
        try self.pending_waits.append(self.allocator, .{
            .point_name = name,
            .callback = callback,
            .user_data = user_data,
        });
    }

    /// Check and process pending waits
    /// Note: Callbacks receive user_data which may be null. Callbacks must handle null safely.
    pub fn processPendingWaits(self: *Self) void {
        var i: usize = 0;
        while (i < self.pending_waits.items.len) {
            const wait_info = self.pending_waits.items[i];
            if (self.sync_points.get(wait_info.point_name)) |point| {
                if (point.signaled) {
                    if (wait_info.callback) |cb| {
                        // Pass user_data directly - callback signature accepts optional pointer
                        // Callback implementations must handle the null case appropriately
                        cb(wait_info.user_data);
                    }
                    _ = self.pending_waits.orderedRemove(i);
                    continue;
                }
            }
            i += 1;
        }
    }

    /// Mark sync points as signaled after frame completion
    pub fn markSignaled(self: *Self, frame_index: u32) void {
        var iter = self.sync_points.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.frame_index == frame_index and !entry.value_ptr.signaled) {
                entry.value_ptr.signaled = true;
            }
        }
    }

    /// Clear all sync points for a frame (when reusing)
    pub fn clearFrame(self: *Self, frame_index: u32) void {
        var to_remove = std.ArrayListUnmanaged([]const u8){};
        defer to_remove.deinit(self.allocator);

        var iter = self.sync_points.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.frame_index == frame_index) {
                to_remove.append(self.allocator, entry.key_ptr.*) catch {
                    // Log allocation failure - sync point won't be removed this frame
                    // but will eventually be cleaned up or overwritten
                    std.log.warn("TimelineSemaphoreSync: failed to track sync point for removal", .{});
                    continue;
                };
            }
        }

        for (to_remove.items) |key| {
            _ = self.sync_points.remove(key);
        }
    }

    /// Get current timeline value
    pub fn getTimelineValue(self: Self) u64 {
        return self.next_timeline - 1;
    }

    /// Check if a sync point exists and is signaled
    pub fn isSignaled(self: Self, name: []const u8) bool {
        if (self.sync_points.get(name)) |point| {
            return point.signaled;
        }
        return false;
    }
};

/// GPU Event wrapper for fine-grained GPU-side synchronization
///
/// VkEvent allows efficient intra-queue synchronization where one command
/// signals an event and later commands wait on it without CPU round-trip.
///
/// ## Usage
///
/// ```zig
/// // Create an event
/// var event = try GpuEvent.init(&ctx, allocator, "my_event");
/// defer event.deinit();
///
/// // In command buffer:
/// event.cmdSet(cmd, .{ .compute_shader = true });  // Signal after compute
/// // ... more compute commands ...
/// event.cmdWait(cmd, .{ .compute_shader = true }, .{ .compute_shader = true });
/// ```
pub const GpuEvent = struct {
    ctx: *Context,
    allocator: std.mem.Allocator,
    handle: types.Event,
    name: []const u8,
    is_device_only: bool = false,

    const Self = @This();

    /// Create a GPU event
    pub fn init(ctx: *Context, allocator: std.mem.Allocator, name: []const u8) !Self {
        return initWithFlags(ctx, allocator, name, .{});
    }

    /// Create a device-only GPU event (more efficient, but can't be queried from host)
    pub fn initDeviceOnly(ctx: *Context, allocator: std.mem.Allocator, name: []const u8) !Self {
        return initWithFlags(ctx, allocator, name, .{ .device_only = true });
    }

    /// Create a GPU event with custom flags
    pub fn initWithFlags(
        ctx: *Context,
        allocator: std.mem.Allocator,
        name: []const u8,
        flags: types.EventCreateFlags,
    ) !Self {
        const create_info = types.EventCreateInfo{
            .flags = flags,
        };

        var event_handle: types.Event = undefined;
        const result = ctx.device_dispatch.vkCreateEvent.?(
            ctx.device,
            &create_info,
            null,
            &event_handle,
        );
        if (result != .success) return err.fromVkResult(result);

        return .{
            .ctx = ctx,
            .allocator = allocator,
            .handle = event_handle,
            .name = name,
            .is_device_only = flags.device_only,
        };
    }

    /// Destroy the event
    pub fn deinit(self: *Self) void {
        if (self.ctx.device_dispatch.vkDestroyEvent) |destroy| {
            destroy(self.ctx.device, self.handle, null);
        }
        self.* = undefined;
    }

    /// Set the event from the host (not available for device-only events)
    pub fn setHost(self: *Self) !void {
        if (self.is_device_only) return error.DeviceOnlyEvent;

        const result = self.ctx.device_dispatch.vkSetEvent.?(self.ctx.device, self.handle);
        if (result != .success) return err.fromVkResult(result);
    }

    /// Reset the event from the host (not available for device-only events)
    pub fn resetHost(self: *Self) !void {
        if (self.is_device_only) return error.DeviceOnlyEvent;

        const result = self.ctx.device_dispatch.vkResetEvent.?(self.ctx.device, self.handle);
        if (result != .success) return err.fromVkResult(result);
    }

    /// Query event status from host (not available for device-only events)
    pub fn getStatus(self: Self) !bool {
        if (self.is_device_only) return error.DeviceOnlyEvent;

        const result = self.ctx.device_dispatch.vkGetEventStatus.?(self.ctx.device, self.handle);
        return switch (result) {
            .event_set => true,
            .event_reset => false,
            else => err.fromVkResult(result),
        };
    }

    /// Record command to set (signal) the event in command buffer
    pub fn cmdSet(self: Self, cmd: types.CommandBuffer, stage_mask: types.PipelineStageFlags) void {
        if (self.ctx.device_dispatch.vkCmdSetEvent) |set_event| {
            set_event(cmd, self.handle, stage_mask);
        }
    }

    /// Record command to reset the event in command buffer
    pub fn cmdReset(self: Self, cmd: types.CommandBuffer, stage_mask: types.PipelineStageFlags) void {
        if (self.ctx.device_dispatch.vkCmdResetEvent) |reset_event| {
            reset_event(cmd, self.handle, stage_mask);
        }
    }

    /// Record command to wait on the event in command buffer
    pub fn cmdWait(
        self: Self,
        cmd: types.CommandBuffer,
        src_stage_mask: types.PipelineStageFlags,
        dst_stage_mask: types.PipelineStageFlags,
    ) void {
        self.cmdWaitWithBarrier(cmd, src_stage_mask, dst_stage_mask, null, null, null);
    }

    /// Record command to wait on the event with memory barriers
    pub fn cmdWaitWithBarrier(
        self: Self,
        cmd: types.CommandBuffer,
        src_stage_mask: types.PipelineStageFlags,
        dst_stage_mask: types.PipelineStageFlags,
        memory_barriers: ?[]const types.MemoryBarrier,
        buffer_barriers: ?[]const types.BufferMemoryBarrier,
        image_barriers: ?[]const types.ImageMemoryBarrier,
    ) void {
        if (self.ctx.device_dispatch.vkCmdWaitEvents) |wait_events| {
            wait_events(
                cmd,
                1,
                @ptrCast(&self.handle),
                src_stage_mask,
                dst_stage_mask,
                if (memory_barriers) |mb| @intCast(mb.len) else 0,
                if (memory_barriers) |mb| mb.ptr else null,
                if (buffer_barriers) |bb| @intCast(bb.len) else 0,
                if (buffer_barriers) |bb| bb.ptr else null,
                if (image_barriers) |ib| @intCast(ib.len) else 0,
                if (image_barriers) |ib| ib.ptr else null,
            );
        }
    }

    /// Get the raw Vulkan handle
    pub fn getHandle(self: Self) types.Event {
        return self.handle;
    }
};

/// Batch of GPU events for multi-event synchronization
pub const GpuEventBatch = struct {
    events: std.ArrayListUnmanaged(GpuEvent),
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .events = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.events.items) |*event| {
            event.deinit();
        }
        self.events.deinit(self.allocator);
    }

    /// Add an event to the batch
    pub fn add(self: *Self, event: GpuEvent) !void {
        try self.events.append(self.allocator, event);
    }

    /// Create and add a new event
    pub fn createAndAdd(self: *Self, ctx: *Context, name: []const u8) !*GpuEvent {
        const event = try GpuEvent.init(ctx, self.allocator, name);
        try self.events.append(self.allocator, event);
        return &self.events.items[self.events.items.len - 1];
    }

    /// Set all events in command buffer
    pub fn cmdSetAll(self: Self, cmd: types.CommandBuffer, stage_mask: types.PipelineStageFlags) void {
        for (self.events.items) |event| {
            event.cmdSet(cmd, stage_mask);
        }
    }

    /// Wait on all events in command buffer
    /// Handles batches larger than 64 events by issuing multiple wait commands
    pub fn cmdWaitAll(
        self: Self,
        cmd: types.CommandBuffer,
        src_stage_mask: types.PipelineStageFlags,
        dst_stage_mask: types.PipelineStageFlags,
    ) void {
        if (self.events.items.len == 0) return;

        const wait_events = self.events.items[0].ctx.device_dispatch.vkCmdWaitEvents orelse return;

        // Process events in batches of 64 (Vulkan implementation limit on some drivers)
        const BATCH_SIZE = 64;
        var offset: usize = 0;

        while (offset < self.events.items.len) {
            var handles: [BATCH_SIZE]types.Event = undefined;
            const remaining = self.events.items.len - offset;
            const batch_count = @min(remaining, BATCH_SIZE);

            for (self.events.items[offset..][0..batch_count], 0..) |event, i| {
                handles[i] = event.handle;
            }

            wait_events(
                cmd,
                @intCast(batch_count),
                &handles,
                src_stage_mask,
                dst_stage_mask,
                0,
                null,
                0,
                null,
                0,
                null,
            );

            offset += batch_count;
        }
    }

    /// Get the number of events
    pub fn count(self: Self) usize {
        return self.events.items.len;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "CoordinatorOptions defaults" {
    const options = CoordinatorOptions{};
    try std.testing.expectEqual(@as(u32, 2), options.frame_count);
    try std.testing.expect(options.use_timeline_semaphores);
    try std.testing.expectEqual(@as(u32, 64), options.max_pending_ops);
}

test "ResourceState transitions" {
    // Just test enum values exist
    try std.testing.expect(@intFromEnum(ResourceState.undefined) == 0);
    try std.testing.expect(@intFromEnum(ResourceState.compute_write) != @intFromEnum(ResourceState.compute_read));
}

test "StreamBuffer swap" {
    var stream = StreamBuffer{
        .buffers = undefined,
        .current = 0,
    };

    try std.testing.expectEqual(@as(u32, 0), stream.current);
    stream.swap();
    try std.testing.expectEqual(@as(u32, 1), stream.current);
    stream.swap();
    try std.testing.expectEqual(@as(u32, 0), stream.current);
}
