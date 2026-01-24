// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Vulkan Compute Buffers
//!
//! High-level buffer management for GPU compute operations.
//! Provides easy-to-use abstractions for common buffer patterns.
//!
//! ## Features
//!
//! - Automatic memory type selection based on usage pattern
//! - Host-visible memory auto-mapping for staging/readback buffers
//! - Size validation against device limits (debug builds)
//! - Type-safe read/write operations
//!
//! ## Buffer Types
//!
//! - `storage`: GPU read/write, device-local memory
//! - `uniform`: Constant data, host-visible for updates
//! - `staging`: CPU to GPU transfers, host-visible
//! - `readback`: GPU to CPU transfers, host-cached
//!
//! ## Thread Safety
//!
//! Buffers are **not** thread-safe. Concurrent access to the same buffer
//! from multiple threads requires external synchronization.
//!
//! | Operation | Thread Safety |
//! |-----------|---------------|
//! | Buffer.init | Thread-safe (allocates new buffer) |
//! | Buffer.deinit | Not thread-safe (must not be in use) |
//! | Buffer.writeSlice | Not thread-safe |
//! | Buffer.readSlice | Not thread-safe |
//! | Buffer.fill | Not thread-safe |
//! | Buffer.getMappedPtr | Not thread-safe |
//!
//! ### Safe Multi-Threading Patterns
//!
//! - Create separate buffers per thread for write operations
//! - Use staging buffers with command buffer synchronization
//! - Protect shared buffer access with mutexes
//! - Use fences to ensure GPU operations complete before CPU access
//!
//! ## Example
//!
//! ```zig
//! // Create a storage buffer for compute operations
//! var buffer = try Buffer.init(&ctx, 1024 * 1024, .storage);
//! defer buffer.deinit(&ctx);
//!
//! // Create staging buffer for data upload
//! var staging = try Buffer.init(&ctx, 1024 * 1024, .staging);
//! defer staging.deinit(&ctx);
//!
//! // Write data to staging buffer
//! try staging.writeSlice(f32, data);
//! ```

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../vk/types.zig");
const loader = @import("../vk/loader.zig");
const err = @import("../vk/error.zig");
const mem_alloc = @import("../util/allocator.zig");
const pooled_alloc = @import("../util/pooled_allocator.zig");
const Context = @import("context.zig").Context;

/// Buffer usage patterns
pub const BufferUsage = enum {
    /// Storage buffer for compute shaders (read/write)
    storage,
    /// Uniform buffer for constant data
    uniform,
    /// Staging buffer for CPU to GPU transfers
    staging,
    /// Readback buffer for GPU to CPU transfers
    readback,
};

/// GPU buffer wrapper
pub const Buffer = struct {
    handle: types.Buffer,
    memory: mem_alloc.MemoryBlock,
    size: u64,
    usage: BufferUsage,
    device: types.Device,
    dispatch: loader.DeviceDispatch,

    const Self = @This();

    /// Create a new buffer
    ///
    /// Allocates a GPU buffer with automatic memory type selection based on usage.
    ///
    /// ## Parameters
    /// - `ctx`: Compute context
    /// - `size`: Buffer size in bytes (must be > 0)
    /// - `usage`: Buffer usage pattern
    ///
    /// ## Errors
    /// - `InvalidArgument`: Size is 0 or exceeds device limits
    /// - `OutOfDeviceMemory`: GPU memory allocation failed
    /// - `OutOfHostMemory`: Host memory allocation failed
    ///
    /// ## Size Limits
    /// - Storage buffers: `max_storage_buffer_range` (typically 2GB+)
    /// - Uniform buffers: `max_uniform_buffer_range` (typically 64KB)
    pub fn init(
        ctx: *Context,
        size: u64,
        usage: BufferUsage,
    ) err.Error!Self {
        // Validate size
        if (size == 0) {
            return err.Error.InvalidArgument;
        }

        // Validate against device limits in debug builds
        if (builtin.mode == .Debug) {
            const limits = ctx.device_properties.limits;
            const max_size: u64 = switch (usage) {
                .storage => limits.max_storage_buffer_range,
                .uniform => limits.max_uniform_buffer_range,
                .staging, .readback => std.math.maxInt(u32),
            };
            if (size > max_size) {
                std.log.warn("[Vulkan] Buffer size {d} exceeds max for {s} ({d})", .{
                    size,
                    @tagName(usage),
                    max_size,
                });
                return err.Error.InvalidArgument;
            }
        }

        const vk_usage = getVkUsage(usage);
        const mem_usage = getMemUsage(usage);

        const buffer_info = types.BufferCreateInfo{
            .size = size,
            .usage = vk_usage,
            .sharing_mode = .exclusive,
        };

        var handle: types.Buffer = undefined;
        const result = ctx.device_dispatch.vkCreateBuffer.?(ctx.device, &buffer_info, null, &handle);
        if (result != .success) return err.fromVkResult(result);
        errdefer ctx.device_dispatch.vkDestroyBuffer.?(ctx.device, handle, null);

        // Get memory requirements
        var mem_reqs: types.MemoryRequirements = undefined;
        ctx.device_dispatch.vkGetBufferMemoryRequirements.?(ctx.device, handle, &mem_reqs);

        // Allocate memory
        const memory = try ctx.gpu_allocator.allocate(mem_reqs, mem_usage);
        errdefer ctx.gpu_allocator.free(memory);

        // Bind memory to buffer
        const bind_result = ctx.device_dispatch.vkBindBufferMemory.?(ctx.device, handle, memory.memory, 0);
        if (bind_result != .success) return err.fromVkResult(bind_result);

        return Self{
            .handle = handle,
            .memory = memory,
            .size = size,
            .usage = usage,
            .device = ctx.device,
            .dispatch = ctx.device_dispatch,
        };
    }

    /// Get maximum buffer size for a usage type
    ///
    /// Returns the maximum allowed buffer size based on device limits.
    pub fn getMaxSize(ctx: *Context, usage: BufferUsage) u64 {
        const limits = ctx.device_properties.limits;
        return switch (usage) {
            .storage => limits.max_storage_buffer_range,
            .uniform => limits.max_uniform_buffer_range,
            .staging, .readback => std.math.maxInt(u32),
        };
    }

    /// Destroy the buffer
    pub fn deinit(self: *Self, ctx: *Context) void {
        self.dispatch.vkDestroyBuffer.?(self.device, self.handle, null);
        ctx.gpu_allocator.free(self.memory);
        self.* = undefined;
    }

    /// Check if buffer is host-visible (can be mapped)
    pub fn isHostVisible(self: Self) bool {
        return self.memory.mapped_ptr != null;
    }

    /// Get mapped memory pointer (if available)
    pub fn getMappedPtr(self: Self) ?[*]u8 {
        return if (self.memory.mapped_ptr) |ptr| @ptrCast(ptr) else null;
    }

    /// Write data to the buffer (requires host-visible memory)
    ///
    /// ## Errors
    /// - `InvalidArgument`: Data too large for buffer
    /// - `MemoryMapFailed`: Buffer is not host-visible or usage incompatible
    pub fn write(self: Self, data: []const u8) err.Error!void {
        if (data.len > self.size) return err.Error.InvalidArgument;

        // Validate usage compatibility - only staging and uniform buffers support host writes
        if (self.usage != .staging and self.usage != .uniform) {
            if (builtin.mode == .Debug) {
                std.log.warn("[Vulkan] write called on {s} buffer - use staging buffer for host writes", .{@tagName(self.usage)});
            }
            return err.Error.MemoryMapFailed;
        }

        const ptr = self.getMappedPtr() orelse return err.Error.MemoryMapFailed;
        @memcpy(ptr[0..data.len], data);
    }

    /// Write typed data to the buffer
    ///
    /// ## Errors
    /// - `InvalidArgument`: Data too large for buffer or size overflow
    /// - `MemoryMapFailed`: Buffer is not host-visible or usage incompatible
    pub fn writeSlice(self: Self, comptime T: type, data: []const T) err.Error!void {
        // Check for multiplication overflow
        const byte_size = std.math.mul(usize, data.len, @sizeOf(T)) catch return err.Error.InvalidArgument;
        if (byte_size > self.size) return err.Error.InvalidArgument;

        // Validate usage compatibility - only staging and uniform buffers support host writes
        if (self.usage != .staging and self.usage != .uniform) {
            if (builtin.mode == .Debug) {
                std.log.warn("[Vulkan] writeSlice called on {s} buffer - use staging buffer for host writes", .{@tagName(self.usage)});
            }
            return err.Error.MemoryMapFailed;
        }

        const ptr = self.getMappedPtr() orelse return err.Error.MemoryMapFailed;
        // Validate alignment before cast to prevent undefined behavior on strict architectures
        if (@alignOf(T) > 1 and !std.mem.isAligned(@intFromPtr(ptr), @alignOf(T))) {
            std.log.warn("[Vulkan] writeSlice: buffer address not aligned for type (required {d})", .{@alignOf(T)});
            return err.Error.InvalidArgument;
        }
        const typed_ptr: [*]T = @ptrCast(@alignCast(ptr));
        @memcpy(typed_ptr[0..data.len], data);
    }

    /// Read data from the buffer (requires host-visible memory)
    ///
    /// ## Errors
    /// - `InvalidArgument`: Destination too large for buffer
    /// - `MemoryMapFailed`: Buffer is not host-visible or usage incompatible
    pub fn read(self: Self, dest: []u8) err.Error!void {
        if (dest.len > self.size) return err.Error.InvalidArgument;

        // Validate usage compatibility - only readback, staging, and uniform buffers support host reads
        if (self.usage != .readback and self.usage != .staging and self.usage != .uniform) {
            if (builtin.mode == .Debug) {
                std.log.warn("[Vulkan] read called on {s} buffer - use readback buffer for host reads", .{@tagName(self.usage)});
            }
            return err.Error.MemoryMapFailed;
        }

        const ptr = self.getMappedPtr() orelse return err.Error.MemoryMapFailed;
        @memcpy(dest, ptr[0..dest.len]);
    }

    /// Read typed data from the buffer
    ///
    /// ## Errors
    /// - `InvalidArgument`: Destination too large for buffer or size overflow
    /// - `MemoryMapFailed`: Buffer is not host-visible or usage incompatible
    pub fn readSlice(self: Self, comptime T: type, dest: []T) err.Error!void {
        // Check for multiplication overflow
        const byte_size = std.math.mul(usize, dest.len, @sizeOf(T)) catch return err.Error.InvalidArgument;
        if (byte_size > self.size) return err.Error.InvalidArgument;

        // Validate usage compatibility - only readback, staging, and uniform buffers support host reads
        if (self.usage != .readback and self.usage != .staging and self.usage != .uniform) {
            if (builtin.mode == .Debug) {
                std.log.warn("[Vulkan] readSlice called on {s} buffer - use readback buffer for host reads", .{@tagName(self.usage)});
            }
            return err.Error.MemoryMapFailed;
        }

        const ptr = self.getMappedPtr() orelse return err.Error.MemoryMapFailed;
        // Validate alignment before cast to prevent undefined behavior on strict architectures
        if (@alignOf(T) > 1 and !std.mem.isAligned(@intFromPtr(ptr), @alignOf(T))) {
            std.log.warn("[Vulkan] readSlice: buffer address not aligned for type (required {d})", .{@alignOf(T)});
            return err.Error.InvalidArgument;
        }
        const typed_ptr: [*]const T = @ptrCast(@alignCast(ptr));
        @memcpy(dest, typed_ptr[0..dest.len]);
    }

    /// Fill buffer with a pattern (via command buffer)
    /// Note: Vulkan requires size to be a multiple of 4 bytes, or VK_WHOLE_SIZE.
    /// Using VK_WHOLE_SIZE here to fill the entire buffer safely.
    pub fn fill(self: Self, cmd: types.CommandBuffer, value: u32) void {
        // Use VK_WHOLE_SIZE to avoid issues with non-4-byte-aligned buffer sizes
        // Vulkan spec: "If size is not equal to VK_WHOLE_SIZE, size must be a multiple of 4"
        self.dispatch.vkCmdFillBuffer.?(cmd, self.handle, 0, types.WHOLE_SIZE, value);
    }

    /// Flush writes to make them visible to the device
    ///
    /// Required for non-coherent memory after CPU writes before GPU reads.
    /// For coherent memory (most staging/readback buffers), this is a no-op.
    pub fn flush(self: Self, offset: u64, size: u64) err.Error!void {
        if (!self.memory.requires_flush) return;

        const flush_range = types.MappedMemoryRange{
            .memory = self.memory.memory,
            .offset = offset,
            .size = size,
        };

        const result = self.dispatch.vkFlushMappedMemoryRanges.?(self.device, 1, @ptrCast(&flush_range));
        if (result != .success) return err.fromVkResult(result);
    }

    /// Flush the entire buffer
    pub fn flushAll(self: Self) err.Error!void {
        return self.flush(0, types.WHOLE_SIZE);
    }

    /// Invalidate reads to get latest data from device
    ///
    /// Required for non-coherent memory after GPU writes before CPU reads.
    /// For coherent memory (most staging/readback buffers), this is a no-op.
    pub fn invalidate(self: Self, offset: u64, size: u64) err.Error!void {
        if (!self.memory.requires_flush) return;

        const invalidate_range = types.MappedMemoryRange{
            .memory = self.memory.memory,
            .offset = offset,
            .size = size,
        };

        const result = self.dispatch.vkInvalidateMappedMemoryRanges.?(self.device, 1, @ptrCast(&invalidate_range));
        if (result != .success) return err.fromVkResult(result);
    }

    /// Invalidate the entire buffer
    pub fn invalidateAll(self: Self) err.Error!void {
        return self.invalidate(0, types.WHOLE_SIZE);
    }

    /// Check if buffer requires explicit flush/invalidate
    pub fn requiresFlush(self: Self) bool {
        return self.memory.requires_flush;
    }
};

/// Buffer transfer operations
pub const Transfer = struct {
    /// Copy data between buffers
    pub fn copy(
        dispatch: loader.DeviceDispatch,
        cmd: types.CommandBuffer,
        src: Buffer,
        dst: Buffer,
        size: ?u64,
    ) void {
        const copy_size = size orelse @min(src.size, dst.size);
        const region = types.BufferCopy{
            .src_offset = 0,
            .dst_offset = 0,
            .size = copy_size,
        };
        dispatch.vkCmdCopyBuffer.?(cmd, src.handle, dst.handle, 1, @ptrCast(&region));
    }

    /// Copy with offsets
    pub fn copyRegion(
        dispatch: loader.DeviceDispatch,
        cmd: types.CommandBuffer,
        src: Buffer,
        dst: Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    ) void {
        const region = types.BufferCopy{
            .src_offset = src_offset,
            .dst_offset = dst_offset,
            .size = size,
        };
        dispatch.vkCmdCopyBuffer.?(cmd, src.handle, dst.handle, 1, @ptrCast(&region));
    }

    /// Insert a buffer memory barrier
    pub fn barrier(
        dispatch: loader.DeviceDispatch,
        cmd: types.CommandBuffer,
        buf: Buffer,
        src_stage: types.PipelineStageFlags,
        dst_stage: types.PipelineStageFlags,
        src_access: types.AccessFlags,
        dst_access: types.AccessFlags,
    ) void {
        const buf_barrier = types.BufferMemoryBarrier{
            .src_access_mask = src_access,
            .dst_access_mask = dst_access,
            .buffer = buf.handle,
            .offset = 0,
            .size = buf.size,
        };

        dispatch.vkCmdPipelineBarrier.?(
            cmd,
            src_stage,
            dst_stage,
            0,
            0,
            null,
            1,
            @ptrCast(&buf_barrier),
            0,
            null,
        );
    }

    /// Common barrier: after compute write, before host read
    pub fn barrierComputeToHost(
        dispatch: loader.DeviceDispatch,
        cmd: types.CommandBuffer,
        buf: Buffer,
    ) void {
        barrier(
            dispatch,
            cmd,
            buf,
            .{ .compute_shader = true },
            .{ .host = true },
            .{ .shader_write = true },
            .{ .host_read = true },
        );
    }

    /// Common barrier: after host write, before compute read
    pub fn barrierHostToCompute(
        dispatch: loader.DeviceDispatch,
        cmd: types.CommandBuffer,
        buf: Buffer,
    ) void {
        barrier(
            dispatch,
            cmd,
            buf,
            .{ .host = true },
            .{ .compute_shader = true },
            .{ .host_write = true },
            .{ .shader_read = true },
        );
    }

    /// Common barrier: after transfer, before compute
    pub fn barrierTransferToCompute(
        dispatch: loader.DeviceDispatch,
        cmd: types.CommandBuffer,
        buf: Buffer,
    ) void {
        barrier(
            dispatch,
            cmd,
            buf,
            .{ .transfer = true },
            .{ .compute_shader = true },
            .{ .transfer_write = true },
            .{ .shader_read = true },
        );
    }

    /// Common barrier: after compute, before transfer
    pub fn barrierComputeToTransfer(
        dispatch: loader.DeviceDispatch,
        cmd: types.CommandBuffer,
        buf: Buffer,
    ) void {
        barrier(
            dispatch,
            cmd,
            buf,
            .{ .compute_shader = true },
            .{ .transfer = true },
            .{ .shader_write = true },
            .{ .transfer_read = true },
        );
    }
};

/// Get Vulkan buffer usage flags for a usage pattern
fn getVkUsage(usage: BufferUsage) types.BufferUsageFlags {
    return switch (usage) {
        .storage => .{ .storage_buffer = true, .transfer_src = true, .transfer_dst = true },
        .uniform => .{ .uniform_buffer = true, .transfer_dst = true },
        .staging => .{ .transfer_src = true },
        .readback => .{ .transfer_dst = true },
    };
}

/// Get memory usage for a buffer usage pattern
fn getMemUsage(usage: BufferUsage) mem_alloc.MemoryUsage {
    return switch (usage) {
        .storage => .gpu_only,
        .uniform => .cpu_to_gpu,
        .staging => .cpu_to_gpu,
        .readback => .gpu_to_cpu,
    };
}

// =============================================================================
// Tests
// =============================================================================

test "getVkUsage returns correct flags" {
    const storage = getVkUsage(.storage);
    try std.testing.expect(storage.storage_buffer);
    try std.testing.expect(storage.transfer_src);
    try std.testing.expect(storage.transfer_dst);

    const staging = getVkUsage(.staging);
    try std.testing.expect(staging.transfer_src);
    try std.testing.expect(!staging.storage_buffer);
}

test "getMemUsage returns correct usage" {
    try std.testing.expectEqual(mem_alloc.MemoryUsage.gpu_only, getMemUsage(.storage));
    try std.testing.expectEqual(mem_alloc.MemoryUsage.cpu_to_gpu, getMemUsage(.staging));
    try std.testing.expectEqual(mem_alloc.MemoryUsage.gpu_to_cpu, getMemUsage(.readback));
}

// =============================================================================
// Pooled Buffer (uses suballocation from PooledAllocator)
// =============================================================================

/// GPU buffer that uses pooled memory allocation with suballocation
///
/// This buffer type is more efficient for many small allocations as it
/// pools memory and suballocates from larger chunks. Use this when you
/// have many buffers of similar sizes or when allocation performance
/// is critical.
///
/// ## Example
///
/// ```zig
/// var allocator = PooledAllocator.init(device, dispatch, mem_props, host_alloc, .{});
/// defer allocator.deinit();
///
/// var buffer = try PooledBuffer.init(&allocator, &ctx, 1024, .storage);
/// defer buffer.deinit(&allocator);
/// ```
pub const PooledBuffer = struct {
    handle: types.Buffer,
    allocation: pooled_alloc.Suballocation,
    size: u64,
    usage: BufferUsage,
    device: types.Device,
    dispatch: loader.DeviceDispatch,

    const Self = @This();

    /// Create a new buffer using pooled allocation
    ///
    /// Uses the provided PooledAllocator to suballocate memory from
    /// larger memory pools, reducing Vulkan allocation overhead.
    pub fn init(
        allocator: *pooled_alloc.PooledAllocator,
        ctx: *Context,
        size: u64,
        usage: BufferUsage,
    ) err.Error!Self {
        // Validate size
        if (size == 0) {
            return err.Error.InvalidArgument;
        }

        // Validate against device limits in debug builds
        if (builtin.mode == .Debug) {
            const limits = ctx.device_properties.limits;
            const max_size: u64 = switch (usage) {
                .storage => limits.max_storage_buffer_range,
                .uniform => limits.max_uniform_buffer_range,
                .staging, .readback => std.math.maxInt(u32),
            };
            if (size > max_size) {
                std.log.warn("[Vulkan] Buffer size {d} exceeds max for {s} ({d})", .{
                    size,
                    @tagName(usage),
                    max_size,
                });
                return err.Error.InvalidArgument;
            }
        }

        const vk_usage = getVkUsage(usage);
        const mem_usage = getMemUsage(usage);

        const buffer_info = types.BufferCreateInfo{
            .size = size,
            .usage = vk_usage,
            .sharing_mode = .exclusive,
        };

        var handle: types.Buffer = undefined;
        const result = ctx.device_dispatch.vkCreateBuffer.?(ctx.device, &buffer_info, null, &handle);
        if (result != .success) return err.fromVkResult(result);
        errdefer ctx.device_dispatch.vkDestroyBuffer.?(ctx.device, handle, null);

        // Get memory requirements
        var mem_reqs: types.MemoryRequirements = undefined;
        ctx.device_dispatch.vkGetBufferMemoryRequirements.?(ctx.device, handle, &mem_reqs);

        // Allocate from pool
        const allocation = try allocator.allocate(mem_reqs, mem_usage);
        errdefer allocator.free(allocation);

        // Bind memory to buffer at the allocation offset
        const bind_result = ctx.device_dispatch.vkBindBufferMemory.?(
            ctx.device,
            handle,
            allocation.memory,
            allocation.offset,
        );
        if (bind_result != .success) return err.fromVkResult(bind_result);

        return Self{
            .handle = handle,
            .allocation = allocation,
            .size = size,
            .usage = usage,
            .device = ctx.device,
            .dispatch = ctx.device_dispatch,
        };
    }

    /// Destroy the buffer and return memory to pool
    pub fn deinit(self: *Self, allocator: *pooled_alloc.PooledAllocator) void {
        self.dispatch.vkDestroyBuffer.?(self.device, self.handle, null);
        allocator.free(self.allocation);
        self.* = undefined;
    }

    /// Check if buffer is host-visible (can be mapped)
    pub fn isHostVisible(self: Self) bool {
        return self.allocation.mapped_ptr != null;
    }

    /// Get mapped memory pointer (if available)
    pub fn getMappedPtr(self: Self) ?[*]u8 {
        return if (self.allocation.mapped_ptr) |ptr| @ptrCast(ptr) else null;
    }

    /// Write data to the buffer (requires host-visible memory)
    ///
    /// ## Errors
    /// - `InvalidArgument`: Data too large for buffer
    /// - `MemoryMapFailed`: Buffer is not host-visible or usage incompatible
    pub fn write(self: Self, data: []const u8) err.Error!void {
        if (data.len > self.size) return err.Error.InvalidArgument;

        // Validate usage compatibility - only staging and uniform buffers support host writes
        if (self.usage != .staging and self.usage != .uniform) {
            if (builtin.mode == .Debug) {
                std.log.warn("[Vulkan] write called on {s} buffer - use staging buffer for host writes", .{@tagName(self.usage)});
            }
            return err.Error.MemoryMapFailed;
        }

        const ptr = self.getMappedPtr() orelse return err.Error.MemoryMapFailed;
        @memcpy(ptr[0..data.len], data);
    }

    /// Write typed data to the buffer
    ///
    /// ## Errors
    /// - `InvalidArgument`: Data too large for buffer or size overflow
    /// - `MemoryMapFailed`: Buffer is not host-visible or usage incompatible
    pub fn writeSlice(self: Self, comptime T: type, data: []const T) err.Error!void {
        // Check for multiplication overflow
        const byte_size = std.math.mul(usize, data.len, @sizeOf(T)) catch return err.Error.InvalidArgument;
        if (byte_size > self.size) return err.Error.InvalidArgument;

        // Validate usage compatibility - only staging and uniform buffers support host writes
        if (self.usage != .staging and self.usage != .uniform) {
            if (builtin.mode == .Debug) {
                std.log.warn("[Vulkan] writeSlice called on {s} buffer - use staging buffer for host writes", .{@tagName(self.usage)});
            }
            return err.Error.MemoryMapFailed;
        }

        const ptr = self.getMappedPtr() orelse return err.Error.MemoryMapFailed;
        // Validate alignment before cast to prevent undefined behavior on strict architectures
        if (@alignOf(T) > 1 and !std.mem.isAligned(@intFromPtr(ptr), @alignOf(T))) {
            std.log.warn("[Vulkan] writeSlice: buffer address not aligned for type (required {d})", .{@alignOf(T)});
            return err.Error.InvalidArgument;
        }
        const typed_ptr: [*]T = @ptrCast(@alignCast(ptr));
        @memcpy(typed_ptr[0..data.len], data);
    }

    /// Read data from the buffer (requires host-visible memory)
    ///
    /// ## Errors
    /// - `InvalidArgument`: Destination too large for buffer
    /// - `MemoryMapFailed`: Buffer is not host-visible or usage incompatible
    pub fn read(self: Self, dest: []u8) err.Error!void {
        if (dest.len > self.size) return err.Error.InvalidArgument;

        // Validate usage compatibility - only readback, staging, and uniform buffers support host reads
        if (self.usage != .readback and self.usage != .staging and self.usage != .uniform) {
            if (builtin.mode == .Debug) {
                std.log.warn("[Vulkan] read called on {s} buffer - use readback buffer for host reads", .{@tagName(self.usage)});
            }
            return err.Error.MemoryMapFailed;
        }

        const ptr = self.getMappedPtr() orelse return err.Error.MemoryMapFailed;
        @memcpy(dest, ptr[0..dest.len]);
    }

    /// Read typed data from the buffer
    ///
    /// ## Errors
    /// - `InvalidArgument`: Destination too large for buffer or size overflow
    /// - `MemoryMapFailed`: Buffer is not host-visible or usage incompatible
    pub fn readSlice(self: Self, comptime T: type, dest: []T) err.Error!void {
        // Check for multiplication overflow
        const byte_size = std.math.mul(usize, dest.len, @sizeOf(T)) catch return err.Error.InvalidArgument;
        if (byte_size > self.size) return err.Error.InvalidArgument;

        // Validate usage compatibility - only readback, staging, and uniform buffers support host reads
        if (self.usage != .readback and self.usage != .staging and self.usage != .uniform) {
            if (builtin.mode == .Debug) {
                std.log.warn("[Vulkan] readSlice called on {s} buffer - use readback buffer for host reads", .{@tagName(self.usage)});
            }
            return err.Error.MemoryMapFailed;
        }

        const ptr = self.getMappedPtr() orelse return err.Error.MemoryMapFailed;
        // Validate alignment before cast to prevent undefined behavior on strict architectures
        if (@alignOf(T) > 1 and !std.mem.isAligned(@intFromPtr(ptr), @alignOf(T))) {
            std.log.warn("[Vulkan] readSlice: buffer address not aligned for type (required {d})", .{@alignOf(T)});
            return err.Error.InvalidArgument;
        }
        const typed_ptr: [*]const T = @ptrCast(@alignCast(ptr));
        @memcpy(dest, typed_ptr[0..dest.len]);
    }

    /// Fill buffer with a pattern (via command buffer)
    /// Note: Vulkan requires size to be a multiple of 4 bytes, or VK_WHOLE_SIZE.
    /// Using VK_WHOLE_SIZE here to fill the entire buffer safely.
    pub fn fill(self: Self, cmd: types.CommandBuffer, value: u32) void {
        // Use VK_WHOLE_SIZE to avoid issues with non-4-byte-aligned buffer sizes
        // Vulkan spec: "If size is not equal to VK_WHOLE_SIZE, size must be a multiple of 4"
        self.dispatch.vkCmdFillBuffer.?(cmd, self.handle, 0, types.WHOLE_SIZE, value);
    }

    /// Get the underlying Vulkan buffer handle
    pub fn getHandle(self: Self) types.Buffer {
        return self.handle;
    }

    /// Get allocation info (for debugging/statistics)
    pub fn getAllocationInfo(self: Self) struct {
        memory: types.DeviceMemory,
        offset: u64,
        size: u64,
        is_dedicated: bool,
    } {
        return .{
            .memory = self.allocation.memory,
            .offset = self.allocation.offset,
            .size = self.allocation.size,
            .is_dedicated = self.allocation.is_dedicated,
        };
    }
};

/// Transfer operations for pooled buffers
pub const PooledTransfer = struct {
    /// Copy data between pooled buffers
    pub fn copy(
        dispatch: loader.DeviceDispatch,
        cmd: types.CommandBuffer,
        src: PooledBuffer,
        dst: PooledBuffer,
        size: ?u64,
    ) void {
        const copy_size = size orelse @min(src.size, dst.size);
        const region = types.BufferCopy{
            .src_offset = 0,
            .dst_offset = 0,
            .size = copy_size,
        };
        dispatch.vkCmdCopyBuffer.?(cmd, src.handle, dst.handle, 1, @ptrCast(&region));
    }

    /// Insert a buffer memory barrier for pooled buffer
    pub fn barrier(
        dispatch: loader.DeviceDispatch,
        cmd: types.CommandBuffer,
        buf: PooledBuffer,
        src_stage: types.PipelineStageFlags,
        dst_stage: types.PipelineStageFlags,
        src_access: types.AccessFlags,
        dst_access: types.AccessFlags,
    ) void {
        const buf_barrier = types.BufferMemoryBarrier{
            .src_access_mask = src_access,
            .dst_access_mask = dst_access,
            .buffer = buf.handle,
            .offset = 0,
            .size = buf.size,
        };

        dispatch.vkCmdPipelineBarrier.?(
            cmd,
            src_stage,
            dst_stage,
            0,
            0,
            null,
            1,
            @ptrCast(&buf_barrier),
            0,
            null,
        );
    }
};

test "PooledBuffer usage" {
    // NOTE: Full functional tests require a Vulkan context and pooled allocator
    // which cannot be created in unit tests. This placeholder ensures the type
    // compiles correctly and has the expected interface.
    // Integration tests with actual GPU are in tests/integration.zig.
    _ = PooledBuffer.init;
    _ = PooledBuffer.deinit;
    _ = PooledBuffer.write;
    _ = PooledBuffer.read;
    _ = PooledBuffer.writeSlice;
    _ = PooledBuffer.readSlice;
    _ = PooledBuffer.isHostVisible;
}
