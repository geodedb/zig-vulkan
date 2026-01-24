// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! GPU Memory Allocator
//!
//! This module provides memory allocation strategies for Vulkan device memory.
//! It supports simple bump allocation with memory type selection based on
//! required properties.

const std = @import("std");
const types = @import("../vk/types.zig");
const loader = @import("../vk/loader.zig");
const err = @import("../vk/error.zig");

/// Memory allocation strategy
pub const AllocationStrategy = enum {
    /// Allocate a new block for each allocation (simple, but wasteful)
    simple,
    /// Pool allocations together (more efficient, requires tracking)
    pooled,
};

/// Memory usage hints for allocation
pub const MemoryUsage = enum {
    /// Memory for GPU-only access (fastest for GPU operations)
    gpu_only,
    /// Memory for CPU to GPU transfers (staging buffers)
    cpu_to_gpu,
    /// Memory for GPU to CPU transfers (readback buffers)
    gpu_to_cpu,
    /// Memory accessible by both CPU and GPU (convenience, may be slower)
    cpu_and_gpu,
};

/// Memory block tracking
pub const MemoryBlock = struct {
    memory: types.DeviceMemory,
    size: u64,
    memory_type_index: u32,
    mapped_ptr: ?*anyopaque = null,
    usage: MemoryUsage,
    /// Whether this memory requires explicit flush/invalidate (non-coherent)
    requires_flush: bool = false,

    /// Check if this block is currently mapped
    pub fn isMapped(self: MemoryBlock) bool {
        return self.mapped_ptr != null;
    }

    /// Check if this memory is coherent (no flush/invalidate needed)
    pub fn isCoherent(self: MemoryBlock) bool {
        return !self.requires_flush;
    }
};

/// Statistics for debugging
pub const AllocatorStats = struct {
    total_allocated: u64 = 0,
    allocation_count: u64 = 0,
    peak_allocated: u64 = 0,
};

/// Simple memory allocator that creates one allocation per request
pub const SimpleAllocator = struct {
    device: types.Device,
    dispatch: loader.DeviceDispatch,
    memory_properties: types.PhysicalDeviceMemoryProperties,
    allocations: std.ArrayListUnmanaged(MemoryBlock),
    host_allocator: std.mem.Allocator,
    stats: AllocatorStats = .{},

    const Self = @This();
    pub const Stats = AllocatorStats;

    /// Initialize the allocator
    pub fn init(
        device: types.Device,
        dispatch: loader.DeviceDispatch,
        memory_properties: types.PhysicalDeviceMemoryProperties,
        host_allocator: std.mem.Allocator,
    ) Self {
        return Self{
            .device = device,
            .dispatch = dispatch,
            .memory_properties = memory_properties,
            .allocations = .{},
            .host_allocator = host_allocator,
        };
    }

    /// Deinitialize the allocator and free all memory
    pub fn deinit(self: *Self) void {
        for (self.allocations.items) |block| {
            if (block.mapped_ptr != null) {
                self.dispatch.vkUnmapMemory.?(self.device, block.memory);
            }
            self.dispatch.vkFreeMemory.?(self.device, block.memory, null);
        }
        self.allocations.deinit(self.host_allocator);
        self.* = undefined;
    }

    /// Allocate memory for a buffer
    pub fn allocate(
        self: *Self,
        requirements: types.MemoryRequirements,
        usage: MemoryUsage,
    ) err.Error!MemoryBlock {
        const required_properties = getRequiredProperties(usage);

        const memory_type_index = self.findMemoryType(
            requirements.memory_type_bits,
            required_properties,
        ) orelse return err.Error.NoSuitableMemoryType;

        const alloc_info = types.MemoryAllocateInfo{
            .allocation_size = requirements.size,
            .memory_type_index = memory_type_index,
        };

        var memory: types.DeviceMemory = undefined;
        const result = self.dispatch.vkAllocateMemory.?(self.device, &alloc_info, null, &memory);
        if (result != .success) return err.fromVkResult(result);

        const mem_props = self.memory_properties.memory_types[memory_type_index].property_flags;
        const is_host_visible = isHostVisible(mem_props);
        const is_coherent = mem_props.host_coherent;

        var block = MemoryBlock{
            .memory = memory,
            .size = requirements.size,
            .memory_type_index = memory_type_index,
            .usage = usage,
            .requires_flush = is_host_visible and !is_coherent,
        };

        // Auto-map host-visible memory
        if (is_host_visible) {
            var ptr: ?*anyopaque = undefined;
            const map_result = self.dispatch.vkMapMemory.?(self.device, memory, 0, requirements.size, 0, &ptr);
            if (map_result == .success) {
                block.mapped_ptr = ptr;
            }
        }

        self.allocations.append(self.host_allocator, block) catch {
            self.dispatch.vkFreeMemory.?(self.device, memory, null);
            return err.Error.OutOfHostMemory;
        };

        // Update stats
        self.stats.total_allocated += requirements.size;
        self.stats.allocation_count += 1;
        if (self.stats.total_allocated > self.stats.peak_allocated) {
            self.stats.peak_allocated = self.stats.total_allocated;
        }

        return block;
    }

    /// Free a specific memory block
    pub fn free(self: *Self, block: MemoryBlock) void {
        if (block.mapped_ptr != null) {
            self.dispatch.vkUnmapMemory.?(self.device, block.memory);
        }
        self.dispatch.vkFreeMemory.?(self.device, block.memory, null);

        // Remove from tracking
        for (self.allocations.items, 0..) |item, i| {
            if (@intFromPtr(item.memory) == @intFromPtr(block.memory)) {
                _ = self.allocations.swapRemove(i);
                break;
            }
        }

        self.stats.total_allocated -|= block.size; // Saturating subtract to prevent underflow
    }

    /// Find a suitable memory type
    fn findMemoryType(
        self: Self,
        type_bits: u32,
        required_properties: types.MemoryPropertyFlags,
    ) ?u32 {
        // Vulkan spec guarantees memory_type_count <= VK_MAX_MEMORY_TYPES (32)
        // Validate this to ensure u5 cast is safe for bit shift operation
        const type_count = self.memory_properties.memory_type_count;
        if (type_count > 32) {
            std.log.err("Invalid memory_type_count {d} exceeds Vulkan maximum of 32", .{type_count});
            return null;
        }
        for (0..type_count) |i| {
            // Safe cast: i < type_count <= 32, so i fits in u5 (0-31)
            const idx: u5 = @intCast(i);
            if ((type_bits & (@as(u32, 1) << idx)) != 0) {
                const props = self.memory_properties.memory_types[i].property_flags;
                if (hasRequiredProperties(props, required_properties)) {
                    return @intCast(i);
                }
            }
        }
        return null;
    }

    /// Get allocation statistics
    pub fn getStats(self: Self) Stats {
        return self.stats;
    }
};

/// Get required memory properties for a usage pattern
fn getRequiredProperties(usage: MemoryUsage) types.MemoryPropertyFlags {
    return switch (usage) {
        .gpu_only => .{ .device_local = true },
        .cpu_to_gpu => .{ .host_visible = true, .host_coherent = true },
        .gpu_to_cpu => .{ .host_visible = true, .host_cached = true },
        .cpu_and_gpu => .{ .host_visible = true, .host_coherent = true },
    };
}

/// Check if memory properties include required properties
fn hasRequiredProperties(
    available: types.MemoryPropertyFlags,
    required: types.MemoryPropertyFlags,
) bool {
    if (required.device_local and !available.device_local) return false;
    if (required.host_visible and !available.host_visible) return false;
    if (required.host_coherent and !available.host_coherent) return false;
    if (required.host_cached and !available.host_cached) return false;
    return true;
}

/// Check if memory type is host visible
fn isHostVisible(props: types.MemoryPropertyFlags) bool {
    return props.host_visible;
}

/// Alignment utilities
pub const Alignment = struct {
    /// Align a size up to the given alignment
    /// Uses checked arithmetic to prevent overflow on very large sizes
    pub fn alignUp(size: u64, alignment: u64) u64 {
        if (alignment == 0) return size;
        // Use checked addition to detect overflow
        const sum = std.math.add(u64, size, alignment - 1) catch {
            // On overflow, return max aligned value
            return std.math.maxInt(u64) - (std.math.maxInt(u64) % alignment);
        };
        return (sum / alignment) * alignment;
    }

    /// Align a size up to the given alignment, returning null on overflow
    /// Use this when overflow should be treated as an allocation failure
    pub fn alignUpChecked(size: u64, alignment: u64) ?u64 {
        if (alignment == 0) return size;
        const sum = std.math.add(u64, size, alignment - 1) catch return null;
        return (sum / alignment) * alignment;
    }

    /// Check if a value is aligned
    pub fn isAligned(value: u64, alignment: u64) bool {
        if (alignment == 0) return true;
        return (value % alignment) == 0;
    }

    /// Get the offset needed to align a value
    pub fn alignmentOffset(value: u64, alignment: u64) u64 {
        if (alignment == 0) return 0;
        const remainder = value % alignment;
        if (remainder == 0) return 0;
        return alignment - remainder;
    }

    /// Safely add offset + size with overflow checking
    /// Returns null if the addition would overflow
    pub fn addChecked(a: u64, b: u64) ?u64 {
        return std.math.add(u64, a, b) catch null;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "Alignment.alignUp" {
    try std.testing.expectEqual(@as(u64, 256), Alignment.alignUp(200, 256));
    try std.testing.expectEqual(@as(u64, 256), Alignment.alignUp(256, 256));
    try std.testing.expectEqual(@as(u64, 512), Alignment.alignUp(257, 256));
    try std.testing.expectEqual(@as(u64, 0), Alignment.alignUp(0, 256));
}

test "Alignment.isAligned" {
    try std.testing.expect(Alignment.isAligned(256, 256));
    try std.testing.expect(Alignment.isAligned(512, 256));
    try std.testing.expect(!Alignment.isAligned(257, 256));
    try std.testing.expect(Alignment.isAligned(0, 256));
}

test "Alignment.alignmentOffset" {
    try std.testing.expectEqual(@as(u64, 56), Alignment.alignmentOffset(200, 256));
    try std.testing.expectEqual(@as(u64, 0), Alignment.alignmentOffset(256, 256));
    try std.testing.expectEqual(@as(u64, 255), Alignment.alignmentOffset(257, 256));
}

test "getRequiredProperties returns correct flags" {
    const gpu_only = getRequiredProperties(.gpu_only);
    try std.testing.expect(gpu_only.device_local);
    try std.testing.expect(!gpu_only.host_visible);

    const cpu_to_gpu = getRequiredProperties(.cpu_to_gpu);
    try std.testing.expect(cpu_to_gpu.host_visible);
    try std.testing.expect(cpu_to_gpu.host_coherent);
}

test "hasRequiredProperties" {
    const available = types.MemoryPropertyFlags{
        .device_local = true,
        .host_visible = true,
        .host_coherent = true,
    };

    const required1 = types.MemoryPropertyFlags{ .device_local = true };
    try std.testing.expect(hasRequiredProperties(available, required1));

    const required2 = types.MemoryPropertyFlags{ .host_cached = true };
    try std.testing.expect(!hasRequiredProperties(available, required2));
}
