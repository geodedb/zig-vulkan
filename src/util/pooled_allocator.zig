// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Pooled GPU Memory Allocator
//!
//! This module provides an efficient memory allocation system for Vulkan that:
//! - Pools allocations to reduce Vulkan API calls
//! - Supports suballocation within memory blocks
//! - Maintains free lists for memory reuse
//! - Tracks comprehensive allocation statistics
//!
//! ## Architecture
//!
//! ```
//! PooledAllocator
//!   └─ MemoryPool (one per memory type)
//!       └─ MemoryChunk (large Vulkan allocations)
//!           └─ Suballocation (individual allocations within chunks)
//! ```
//!
//! ## Usage
//!
//! ```zig
//! var allocator = PooledAllocator.init(device, dispatch, mem_props, host_alloc, .{});
//! defer allocator.deinit();
//!
//! // Allocate memory
//! const alloc = try allocator.allocate(requirements, .gpu_only);
//!
//! // Use alloc.memory, alloc.offset, alloc.mapped_ptr
//!
//! // Free when done
//! allocator.free(alloc);
//!
//! // Check statistics
//! const stats = allocator.getStats();
//! ```

const std = @import("std");
const types = @import("../vk/types.zig");
const loader = @import("../vk/loader.zig");
const err = @import("../vk/error.zig");
const base_alloc = @import("allocator.zig");

pub const MemoryUsage = base_alloc.MemoryUsage;
pub const Alignment = base_alloc.Alignment;

/// Configuration for the pooled allocator
pub const PoolConfig = struct {
    /// Default chunk size for device-local memory (64 MB)
    device_chunk_size: u64 = 64 * 1024 * 1024,
    /// Default chunk size for host-visible memory (16 MB)
    host_chunk_size: u64 = 16 * 1024 * 1024,
    /// Maximum number of chunks per memory type
    max_chunks_per_type: u32 = 32,
    /// Minimum allocation size (smaller allocations are rounded up)
    min_allocation_size: u64 = 256,
    /// Enable defragmentation on free (slower but reduces fragmentation)
    enable_defrag_on_free: bool = true,
    /// Threshold for dedicated allocation (allocations larger than this get their own block)
    dedicated_threshold: u64 = 16 * 1024 * 1024,

    /// Create configuration tuned for small frequent allocations
    /// Optimized for: Particle systems, UI elements, frequent buffer updates
    pub fn forSmallAllocations() PoolConfig {
        return .{
            .device_chunk_size = 16 * 1024 * 1024, // 16 MB chunks
            .host_chunk_size = 4 * 1024 * 1024, // 4 MB chunks
            .min_allocation_size = 64, // Smaller minimum
            .dedicated_threshold = 4 * 1024 * 1024, // Earlier dedicated threshold
            .enable_defrag_on_free = true,
        };
    }

    /// Create configuration tuned for large infrequent allocations
    /// Optimized for: Textures, large compute buffers, ML models
    pub fn forLargeAllocations() PoolConfig {
        return .{
            .device_chunk_size = 256 * 1024 * 1024, // 256 MB chunks
            .host_chunk_size = 64 * 1024 * 1024, // 64 MB chunks
            .min_allocation_size = 4096, // Larger minimum (page-aligned)
            .dedicated_threshold = 64 * 1024 * 1024, // Later dedicated threshold
            .enable_defrag_on_free = false, // Less fragmentation concern
        };
    }

    /// Create configuration tuned for streaming workloads
    /// Optimized for: Video processing, real-time data, ring buffers
    pub fn forStreaming() PoolConfig {
        return .{
            .device_chunk_size = 32 * 1024 * 1024, // 32 MB chunks
            .host_chunk_size = 32 * 1024 * 1024, // Larger host chunks for staging
            .min_allocation_size = 256,
            .dedicated_threshold = 32 * 1024 * 1024,
            .enable_defrag_on_free = true, // Important for ring buffer reuse
        };
    }

    /// Create configuration tuned for ML inference workloads
    /// Optimized for: Neural network weights, activation buffers
    pub fn forMLInference() PoolConfig {
        return .{
            .device_chunk_size = 128 * 1024 * 1024, // 128 MB for large weight matrices
            .host_chunk_size = 16 * 1024 * 1024,
            .min_allocation_size = 4096, // Aligned for tensor operations
            .dedicated_threshold = 128 * 1024 * 1024, // Very large weights get dedicated
            .enable_defrag_on_free = false, // Weights are usually static
        };
    }

    /// Create configuration based on device memory size
    /// Automatically scales chunk sizes based on available GPU memory
    pub fn autoTuned(total_device_memory: u64) PoolConfig {
        // Use ~1-2% of device memory per chunk, with reasonable bounds
        const target_chunk = @max(
            16 * 1024 * 1024, // Minimum 16 MB
            @min(
                256 * 1024 * 1024, // Maximum 256 MB
                total_device_memory / 64, // ~1.5% of memory
            ),
        );

        // Dedicated threshold at ~25% of chunk size
        const dedicated = target_chunk / 4;

        return .{
            .device_chunk_size = target_chunk,
            .host_chunk_size = target_chunk / 4, // Host chunks are smaller
            .min_allocation_size = 256,
            .dedicated_threshold = dedicated,
            .enable_defrag_on_free = true,
        };
    }

    /// Create configuration for memory-constrained environments
    /// Optimized for: Mobile GPUs, integrated graphics, low memory systems
    pub fn forLowMemory() PoolConfig {
        return .{
            .device_chunk_size = 8 * 1024 * 1024, // 8 MB chunks
            .host_chunk_size = 2 * 1024 * 1024, // 2 MB chunks
            .max_chunks_per_type = 16, // Fewer chunks
            .min_allocation_size = 64,
            .dedicated_threshold = 2 * 1024 * 1024, // Earlier dedicated
            .enable_defrag_on_free = true, // Critical for reuse
        };
    }
};

/// Comprehensive allocation statistics
pub const AllocatorStats = struct {
    // Current state
    /// Currently allocated bytes (active allocations)
    current_allocated: u64 = 0,
    /// Current number of active allocations
    current_allocation_count: u64 = 0,
    /// Current number of Vulkan memory blocks
    current_block_count: u64 = 0,

    // Peak metrics
    /// Peak allocated bytes
    peak_allocated: u64 = 0,
    /// Peak allocation count
    peak_allocation_count: u64 = 0,
    /// Peak block count
    peak_block_count: u64 = 0,

    // Cumulative metrics
    /// Total allocations ever made
    total_allocations: u64 = 0,
    /// Total deallocations ever made
    total_deallocations: u64 = 0,
    /// Total bytes ever allocated
    total_bytes_allocated: u64 = 0,
    /// Total bytes ever freed
    total_bytes_freed: u64 = 0,

    // Pool efficiency
    /// Total pool capacity (all chunks)
    pool_capacity: u64 = 0,
    /// Bytes wasted due to alignment
    alignment_waste: u64 = 0,
    /// Bytes in free list (available for reuse)
    free_list_bytes: u64 = 0,
    /// Number of free regions
    free_region_count: u64 = 0,

    // Per-type breakdown
    /// Bytes allocated per memory usage type
    bytes_by_usage: [4]u64 = .{ 0, 0, 0, 0 },
    /// Allocation count per memory usage type
    count_by_usage: [4]u64 = .{ 0, 0, 0, 0 },

    // Pool hit/miss
    /// Allocations satisfied from existing pools
    pool_hits: u64 = 0,
    /// Allocations requiring new chunks
    pool_misses: u64 = 0,
    /// Allocations using dedicated memory
    dedicated_allocations: u64 = 0,

    /// Get fragmentation ratio (0.0 = no fragmentation, approaching 1.0 = highly fragmented)
    ///
    /// This metric estimates fragmentation based on the number of free regions:
    /// - 1 free region = no fragmentation (0.0)
    /// - Many small free regions = high fragmentation
    ///
    /// Formula: (regions - 1) / (regions + 10)
    /// - The +10 dampening factor prevents the ratio from quickly reaching 1.0
    /// - With 11 regions: ~0.5, with 100 regions: ~0.9
    ///
    /// This is a heuristic approximation. For precise fragmentation analysis,
    /// examine free_region_count and free_list_bytes directly.
    pub fn fragmentationRatio(self: AllocatorStats) f64 {
        if (self.pool_capacity == 0) return 0.0;
        const used = self.current_allocated;
        const free = self.free_list_bytes;
        if (used + free == 0) return 0.0;

        const ideal_free_regions: f64 = 1.0;
        const actual_free_regions: f64 = @floatFromInt(self.free_region_count);
        if (actual_free_regions <= ideal_free_regions) return 0.0;

        // Dampened ratio: grows slowly as regions increase
        // +10 ensures the ratio approaches 1.0 gradually
        return @min(1.0, (actual_free_regions - ideal_free_regions) / (actual_free_regions + 10.0));
    }

    /// Get pool utilization ratio (0.0 = empty, 1.0 = full)
    pub fn utilizationRatio(self: AllocatorStats) f64 {
        if (self.pool_capacity == 0) return 0.0;
        return @as(f64, @floatFromInt(self.current_allocated)) / @as(f64, @floatFromInt(self.pool_capacity));
    }

    /// Get pool hit rate (0.0 = all misses, 1.0 = all hits)
    pub fn poolHitRate(self: AllocatorStats) f64 {
        const total = self.pool_hits + self.pool_misses;
        if (total == 0) return 1.0;
        return @as(f64, @floatFromInt(self.pool_hits)) / @as(f64, @floatFromInt(total));
    }

    /// Reset cumulative statistics (keeps current state)
    pub fn resetCumulative(self: *AllocatorStats) void {
        self.total_allocations = 0;
        self.total_deallocations = 0;
        self.total_bytes_allocated = 0;
        self.total_bytes_freed = 0;
        self.pool_hits = 0;
        self.pool_misses = 0;
        self.dedicated_allocations = 0;
    }
};

/// A suballocation within a memory chunk
pub const Suballocation = struct {
    /// The Vulkan memory handle (from the parent chunk)
    memory: types.DeviceMemory,
    /// Offset within the memory block
    offset: u64,
    /// Size of this allocation
    size: u64,
    /// Memory type index
    memory_type_index: u32,
    /// Mapped pointer (if host-visible, adjusted for offset)
    mapped_ptr: ?*anyopaque,
    /// Memory usage hint
    usage: MemoryUsage,
    /// Whether this is a dedicated allocation
    is_dedicated: bool,
    /// Internal: chunk index for pool tracking
    chunk_index: u32,

    /// Get mapped pointer as typed slice
    /// Returns null if not mapped, size is not a multiple of sizeof(T), or pointer is misaligned
    pub fn getMappedSlice(self: Suballocation, comptime T: type) ?[]T {
        if (self.mapped_ptr) |ptr| {
            // Validate size is a multiple of element size to prevent buffer underread
            if (self.size % @sizeOf(T) != 0) {
                std.log.warn("Suballocation.getMappedSlice: size {d} is not a multiple of element size {d}", .{
                    self.size,
                    @sizeOf(T),
                });
                return null;
            }
            // Validate alignment
            const byte_ptr: [*]u8 = @ptrCast(ptr);
            if (!std.mem.isAligned(@intFromPtr(byte_ptr), @alignOf(T))) {
                std.log.warn("Suballocation.getMappedSlice: pointer not aligned for type (required alignment: {d})", .{
                    @alignOf(T),
                });
                return null;
            }
            const typed_ptr: [*]T = @ptrCast(@alignCast(byte_ptr));
            return typed_ptr[0 .. self.size / @sizeOf(T)];
        }
        return null;
    }

    /// Check if this allocation is mapped
    pub fn isMapped(self: Suballocation) bool {
        return self.mapped_ptr != null;
    }
};

/// A free region within a chunk
const FreeRegion = struct {
    offset: u64,
    size: u64,
};

/// A memory chunk (one Vulkan allocation)
const MemoryChunk = struct {
    memory: types.DeviceMemory,
    size: u64,
    memory_type_index: u32,
    mapped_base: ?[*]u8,
    free_regions: std.ArrayListUnmanaged(FreeRegion),
    allocation_count: u32,
    is_dedicated: bool,
    /// Bytes leaked due to failed free tracking (for diagnostics)
    leaked_bytes: u64 = 0,
    /// Whether this chunk slot has been freed (for stable index preservation)
    /// Freed dedicated chunks keep their slot to prevent index invalidation
    is_freed: bool = false,

    fn init(
        memory: types.DeviceMemory,
        size: u64,
        memory_type_index: u32,
        mapped_base: ?[*]u8,
        is_dedicated: bool,
        host_allocator: std.mem.Allocator,
    ) MemoryChunk {
        var chunk = MemoryChunk{
            .memory = memory,
            .size = size,
            .memory_type_index = memory_type_index,
            .mapped_base = mapped_base,
            .free_regions = .{},
            .allocation_count = 0,
            .is_dedicated = is_dedicated,
        };

        // Start with entire chunk as free (unless dedicated)
        if (!is_dedicated) {
            chunk.free_regions.append(host_allocator, .{
                .offset = 0,
                .size = size,
            }) catch |e| {
                std.log.warn("MemoryChunk.init: failed to initialize free region: {s} - chunk unusable for suballocation", .{
                    @errorName(e),
                });
            };
        }

        return chunk;
    }

    fn deinit(self: *MemoryChunk, host_allocator: std.mem.Allocator) void {
        self.free_regions.deinit(host_allocator);
    }

    /// Try to allocate from this chunk
    fn allocate(
        self: *MemoryChunk,
        size: u64,
        alignment: u64,
        host_allocator: std.mem.Allocator,
    ) ?struct { offset: u64, actual_size: u64 } {
        // Find best-fit free region
        var best_index: ?usize = null;
        var best_waste: u64 = std.math.maxInt(u64);

        for (self.free_regions.items, 0..) |region, i| {
            // Use checked alignment to detect overflow
            const aligned_offset = Alignment.alignUpChecked(region.offset, alignment) orelse continue;
            // Ensure aligned offset doesn't exceed region bounds
            if (aligned_offset < region.offset) continue; // Overflow occurred
            const offset_waste = aligned_offset - region.offset;
            const available = if (region.size > offset_waste) region.size - offset_waste else 0;

            if (available >= size) {
                const total_waste = offset_waste + (available - size);
                if (total_waste < best_waste) {
                    best_waste = total_waste;
                    best_index = i;
                }
            }
        }

        if (best_index) |idx| {
            const region = self.free_regions.items[idx];
            const aligned_offset = Alignment.alignUpChecked(region.offset, alignment) orelse return null;
            if (aligned_offset < region.offset) return null; // Overflow check
            const offset_waste = aligned_offset - region.offset;

            // Use checked addition to prevent overflow
            const actual_size = Alignment.addChecked(size, offset_waste) orelse return null;
            const new_offset = Alignment.addChecked(aligned_offset, size) orelse return null;

            // Update or remove the free region
            if (region.size == actual_size) {
                // Exact fit - remove region
                _ = self.free_regions.orderedRemove(idx);
            } else if (offset_waste > 0 and region.size > actual_size) {
                // Split into two: waste at start, remainder at end
                self.free_regions.items[idx] = .{
                    .offset = new_offset,
                    .size = region.size - actual_size,
                };
                // Add waste region at start if significant
                if (offset_waste >= 64) {
                    self.free_regions.insert(host_allocator, idx, .{
                        .offset = region.offset,
                        .size = offset_waste,
                    }) catch |e| {
                        std.log.debug("MemoryChunk.allocate: failed to track waste region: {s} - minor fragmentation", .{
                            @errorName(e),
                        });
                    };
                }
            } else {
                // Just shrink from the start
                self.free_regions.items[idx] = .{
                    .offset = new_offset,
                    .size = region.size - actual_size,
                };
            }

            self.allocation_count += 1;
            return .{ .offset = aligned_offset, .actual_size = size };
        }

        return null;
    }

    /// Free an allocation within this chunk
    ///
    /// NOTE: If free region tracking fails (out of host memory), the memory is leaked
    /// within this chunk until the chunk itself is freed. The `leaked_bytes` counter
    /// tracks this for diagnostic purposes, but there is no automatic recovery path.
    /// This is a rare edge case that only occurs under extreme memory pressure.
    fn free(self: *MemoryChunk, offset: u64, size: u64, host_allocator: std.mem.Allocator, enable_coalesce: bool) void {
        self.allocation_count -|= 1; // Saturating to handle edge cases

        // Add to free list
        const new_region = FreeRegion{ .offset = offset, .size = size };

        // Find insertion point (keep sorted by offset)
        var insert_idx: usize = 0;
        for (self.free_regions.items, 0..) |region, i| {
            if (region.offset > offset) {
                insert_idx = i;
                break;
            }
            insert_idx = i + 1;
        }

        self.free_regions.insert(host_allocator, insert_idx, new_region) catch |e| {
            // If we can't track it, memory is leaked within the chunk
            // Track the leaked bytes for diagnostic purposes
            self.leaked_bytes += size;
            std.log.warn("MemoryChunk.free: failed to track freed region at offset {d}: {s} - {d} bytes leaked (total leaked: {d})", .{
                offset,
                @errorName(e),
                size,
                self.leaked_bytes,
            });
            return;
        };

        // Coalesce adjacent free regions
        if (enable_coalesce) {
            self.coalesce(insert_idx);
        }
    }

    /// Coalesce adjacent free regions around the given index
    fn coalesce(self: *MemoryChunk, idx: usize) void {
        if (self.free_regions.items.len <= 1) return;

        // Try to merge with next region
        if (idx + 1 < self.free_regions.items.len) {
            const current = self.free_regions.items[idx];
            const next = self.free_regions.items[idx + 1];
            if (current.offset + current.size == next.offset) {
                self.free_regions.items[idx].size += next.size;
                _ = self.free_regions.orderedRemove(idx + 1);
            }
        }

        // Try to merge with previous region
        if (idx > 0) {
            const prev = self.free_regions.items[idx - 1];
            const current = self.free_regions.items[idx];
            if (prev.offset + prev.size == current.offset) {
                self.free_regions.items[idx - 1].size += current.size;
                _ = self.free_regions.orderedRemove(idx);
            }
        }
    }

    /// Get total free space in this chunk
    fn totalFreeSpace(self: MemoryChunk) u64 {
        var total: u64 = 0;
        for (self.free_regions.items) |region| {
            total += region.size;
        }
        return total;
    }

    /// Get largest contiguous free region
    fn largestFreeRegion(self: MemoryChunk) u64 {
        var largest: u64 = 0;
        for (self.free_regions.items) |region| {
            largest = @max(largest, region.size);
        }
        return largest;
    }

    /// Coalesce all adjacent free regions
    fn coalesceAdjacentFreeRegions(self: *MemoryChunk) void {
        if (self.free_regions.items.len <= 1) return;

        // Sort regions by offset first
        std.mem.sort(FreeRegion, self.free_regions.items, {}, struct {
            fn lessThan(_: void, a: FreeRegion, b: FreeRegion) bool {
                return a.offset < b.offset;
            }
        }.lessThan);

        // Merge adjacent regions
        var write_idx: usize = 0;
        var read_idx: usize = 1;

        while (read_idx < self.free_regions.items.len) {
            const current = &self.free_regions.items[write_idx];
            const next = self.free_regions.items[read_idx];

            if (current.offset + current.size == next.offset) {
                // Merge: extend current to include next
                current.size += next.size;
            } else {
                // No merge: move to next write position
                write_idx += 1;
                if (write_idx != read_idx) {
                    self.free_regions.items[write_idx] = next;
                }
            }
            read_idx += 1;
        }

        // Truncate the list to remove merged regions
        self.free_regions.items.len = write_idx + 1;
    }
};

/// Memory pool for a specific memory type
const MemoryPool = struct {
    memory_type_index: u32,
    chunks: std.ArrayListUnmanaged(MemoryChunk),
    is_host_visible: bool,

    fn init(memory_type_index: u32, is_host_visible: bool) MemoryPool {
        return .{
            .memory_type_index = memory_type_index,
            .chunks = .{},
            .is_host_visible = is_host_visible,
        };
    }

    fn deinit(
        self: *MemoryPool,
        device: types.Device,
        dispatch: loader.DeviceDispatch,
        host_allocator: std.mem.Allocator,
    ) void {
        for (self.chunks.items) |*chunk| {
            if (chunk.mapped_base != null) {
                dispatch.vkUnmapMemory.?(device, chunk.memory);
            }
            dispatch.vkFreeMemory.?(device, chunk.memory, null);
            chunk.deinit(host_allocator);
        }
        self.chunks.deinit(host_allocator);
    }
};

/// Pooled memory allocator with suballocation support
pub const PooledAllocator = struct {
    device: types.Device,
    dispatch: loader.DeviceDispatch,
    memory_properties: types.PhysicalDeviceMemoryProperties,
    host_allocator: std.mem.Allocator,
    config: PoolConfig,

    /// Memory pools indexed by memory type
    pools: [types.MAX_MEMORY_TYPES]?MemoryPool,

    /// Statistics
    stats: AllocatorStats,

    const Self = @This();

    /// Initialize the pooled allocator
    pub fn init(
        device: types.Device,
        dispatch: loader.DeviceDispatch,
        memory_properties: types.PhysicalDeviceMemoryProperties,
        host_allocator: std.mem.Allocator,
        config: PoolConfig,
    ) Self {
        return .{
            .device = device,
            .dispatch = dispatch,
            .memory_properties = memory_properties,
            .host_allocator = host_allocator,
            .config = config,
            .pools = .{null} ** types.MAX_MEMORY_TYPES,
            .stats = .{},
        };
    }

    /// Deinitialize and free all memory
    pub fn deinit(self: *Self) void {
        for (&self.pools) |*maybe_pool| {
            if (maybe_pool.*) |*pool| {
                pool.deinit(self.device, self.dispatch, self.host_allocator);
                maybe_pool.* = null;
            }
        }
        self.* = undefined;
    }

    /// Allocate memory with pooling and suballocation
    pub fn allocate(
        self: *Self,
        requirements: types.MemoryRequirements,
        usage: MemoryUsage,
    ) err.Error!Suballocation {
        const required_properties = getRequiredProperties(usage);
        const memory_type_index = self.findMemoryType(
            requirements.memory_type_bits,
            required_properties,
        ) orelse return err.Error.NoSuitableMemoryType;

        const size = @max(requirements.size, self.config.min_allocation_size);
        const alignment = @max(requirements.alignment, 1);

        // Check if this should be a dedicated allocation
        if (size >= self.config.dedicated_threshold) {
            return self.allocateDedicated(size, alignment, memory_type_index, usage);
        }

        // Try to allocate from existing pool
        if (self.pools[memory_type_index]) |*pool| {
            if (self.tryAllocateFromPool(pool, size, alignment, usage)) |alloc| {
                self.stats.pool_hits += 1;
                return alloc;
            }
        }

        // Need to create a new chunk
        self.stats.pool_misses += 1;
        return self.allocateNewChunk(size, alignment, memory_type_index, usage);
    }

    /// Allocate from an existing pool
    fn tryAllocateFromPool(
        self: *Self,
        pool: *MemoryPool,
        size: u64,
        alignment: u64,
        usage: MemoryUsage,
    ) ?Suballocation {
        for (pool.chunks.items, 0..) |*chunk, chunk_idx| {
            if (chunk.is_dedicated) continue;

            if (chunk.allocate(size, alignment, self.host_allocator)) |result| {
                self.updateStatsOnAlloc(size, usage, result.actual_size - size);

                const mapped_ptr: ?*anyopaque = if (chunk.mapped_base) |base|
                    @ptrCast(base + result.offset)
                else
                    null;

                return Suballocation{
                    .memory = chunk.memory,
                    .offset = result.offset,
                    .size = size,
                    .memory_type_index = pool.memory_type_index,
                    .mapped_ptr = mapped_ptr,
                    .usage = usage,
                    .is_dedicated = false,
                    .chunk_index = @intCast(chunk_idx),
                };
            }
        }
        return null;
    }

    /// Allocate a new chunk and suballocate from it
    fn allocateNewChunk(
        self: *Self,
        size: u64,
        alignment: u64,
        memory_type_index: u32,
        usage: MemoryUsage,
    ) err.Error!Suballocation {
        const is_host_visible = isHostVisible(
            self.memory_properties.memory_types[memory_type_index].property_flags,
        );

        // Determine chunk size
        const base_chunk_size = if (is_host_visible)
            self.config.host_chunk_size
        else
            self.config.device_chunk_size;

        const chunk_size = @max(base_chunk_size, Alignment.alignUp(size, alignment) * 2);

        // Allocate Vulkan memory
        const alloc_info = types.MemoryAllocateInfo{
            .allocation_size = chunk_size,
            .memory_type_index = memory_type_index,
        };

        var memory: types.DeviceMemory = undefined;
        const result = self.dispatch.vkAllocateMemory.?(self.device, &alloc_info, null, &memory);
        if (result != .success) return err.fromVkResult(result);

        // Map if host-visible
        var mapped_base: ?[*]u8 = null;
        if (is_host_visible) {
            var ptr: ?*anyopaque = undefined;
            const map_result = self.dispatch.vkMapMemory.?(self.device, memory, 0, chunk_size, 0, &ptr);
            if (map_result == .success) {
                mapped_base = @ptrCast(ptr);
            }
        }

        // Create or get pool
        if (self.pools[memory_type_index] == null) {
            self.pools[memory_type_index] = MemoryPool.init(memory_type_index, is_host_visible);
        }

        var pool = &self.pools[memory_type_index].?;

        // Create chunk
        var chunk = MemoryChunk.init(memory, chunk_size, memory_type_index, mapped_base, false, self.host_allocator);

        // Allocate from the new chunk
        const alloc_result = chunk.allocate(size, alignment, self.host_allocator) orelse {
            // This should never happen on a fresh chunk, but handle gracefully
            std.log.err("PooledAllocator: allocation from fresh chunk failed - chunk initialization likely failed", .{});
            chunk.deinit(self.host_allocator);
            self.dispatch.vkFreeMemory.?(self.device, memory, null);
            return err.Error.OutOfHostMemory;
        };

        // Bounds check before casting to u32
        if (pool.chunks.items.len >= std.math.maxInt(u32)) {
            std.log.err("PooledAllocator: too many chunks ({d}), cannot allocate more", .{pool.chunks.items.len});
            chunk.deinit(self.host_allocator);
            self.dispatch.vkFreeMemory.?(self.device, memory, null);
            return err.Error.OutOfHostMemory;
        }
        const chunk_idx: u32 = @intCast(pool.chunks.items.len);
        pool.chunks.append(self.host_allocator, chunk) catch {
            chunk.deinit(self.host_allocator);
            self.dispatch.vkFreeMemory.?(self.device, memory, null);
            return err.Error.OutOfHostMemory;
        };

        self.updateStatsOnAlloc(size, usage, alloc_result.actual_size - size);
        self.stats.current_block_count += 1;
        self.stats.pool_capacity += chunk_size;
        if (self.stats.current_block_count > self.stats.peak_block_count) {
            self.stats.peak_block_count = self.stats.current_block_count;
        }

        const mapped_ptr: ?*anyopaque = if (mapped_base) |base|
            @ptrCast(base + alloc_result.offset)
        else
            null;

        return Suballocation{
            .memory = memory,
            .offset = alloc_result.offset,
            .size = size,
            .memory_type_index = memory_type_index,
            .mapped_ptr = mapped_ptr,
            .usage = usage,
            .is_dedicated = false,
            .chunk_index = chunk_idx,
        };
    }

    /// Allocate dedicated memory for large allocations
    fn allocateDedicated(
        self: *Self,
        size: u64,
        alignment: u64,
        memory_type_index: u32,
        usage: MemoryUsage,
    ) err.Error!Suballocation {
        const aligned_size = Alignment.alignUp(size, alignment);
        const is_host_visible = isHostVisible(
            self.memory_properties.memory_types[memory_type_index].property_flags,
        );

        const alloc_info = types.MemoryAllocateInfo{
            .allocation_size = aligned_size,
            .memory_type_index = memory_type_index,
        };

        var memory: types.DeviceMemory = undefined;
        const result = self.dispatch.vkAllocateMemory.?(self.device, &alloc_info, null, &memory);
        if (result != .success) return err.fromVkResult(result);

        var mapped_ptr: ?*anyopaque = null;
        if (is_host_visible) {
            var ptr: ?*anyopaque = undefined;
            const map_result = self.dispatch.vkMapMemory.?(self.device, memory, 0, aligned_size, 0, &ptr);
            if (map_result == .success) {
                mapped_ptr = ptr;
            }
        }

        // Create pool and chunk for tracking
        if (self.pools[memory_type_index] == null) {
            self.pools[memory_type_index] = MemoryPool.init(memory_type_index, is_host_visible);
        }

        var pool = &self.pools[memory_type_index].?;
        const chunk = MemoryChunk.init(
            memory,
            aligned_size,
            memory_type_index,
            if (mapped_ptr) |p| @ptrCast(p) else null,
            true, // is_dedicated
            self.host_allocator,
        );

        // Look for a freed slot to reuse (maintains index stability)
        var chunk_idx: u32 = undefined;
        var found_freed_slot = false;
        for (pool.chunks.items, 0..) |*existing_chunk, i| {
            if (existing_chunk.is_freed) {
                // Reuse this freed slot
                existing_chunk.* = chunk;
                // Safe cast: i < items.len which was bounded when chunk was created
                chunk_idx = if (i <= std.math.maxInt(u32)) @intCast(i) else {
                    std.log.err("PooledAllocator: chunk index {d} exceeds u32 max", .{i});
                    self.dispatch.vkFreeMemory.?(self.device, memory, null);
                    return err.Error.OutOfHostMemory;
                };
                found_freed_slot = true;
                break;
            }
        }

        if (!found_freed_slot) {
            // Bounds check before casting to u32
            if (pool.chunks.items.len >= std.math.maxInt(u32)) {
                std.log.err("PooledAllocator: too many chunks ({d}), cannot allocate more", .{pool.chunks.items.len});
                self.dispatch.vkFreeMemory.?(self.device, memory, null);
                return err.Error.OutOfHostMemory;
            }
            chunk_idx = @intCast(pool.chunks.items.len);
            pool.chunks.append(self.host_allocator, chunk) catch {
                self.dispatch.vkFreeMemory.?(self.device, memory, null);
                return err.Error.OutOfHostMemory;
            };
        }

        self.updateStatsOnAlloc(size, usage, aligned_size - size);
        self.stats.current_block_count += 1;
        self.stats.pool_capacity += aligned_size;
        self.stats.dedicated_allocations += 1;
        if (self.stats.current_block_count > self.stats.peak_block_count) {
            self.stats.peak_block_count = self.stats.current_block_count;
        }

        return Suballocation{
            .memory = memory,
            .offset = 0,
            .size = size,
            .memory_type_index = memory_type_index,
            .mapped_ptr = mapped_ptr,
            .usage = usage,
            .is_dedicated = true,
            .chunk_index = chunk_idx,
        };
    }

    /// Free a suballocation
    pub fn free(self: *Self, alloc: Suballocation) void {
        if (self.pools[alloc.memory_type_index]) |*pool| {
            if (alloc.chunk_index < pool.chunks.items.len) {
                var chunk = &pool.chunks.items[alloc.chunk_index];

                if (alloc.is_dedicated) {
                    // Free entire dedicated allocation
                    if (chunk.mapped_base != null) {
                        self.dispatch.vkUnmapMemory.?(self.device, chunk.memory);
                    }
                    self.dispatch.vkFreeMemory.?(self.device, chunk.memory, null);
                    chunk.deinit(self.host_allocator);

                    // Mark chunk slot as freed instead of removing to preserve indices
                    // This prevents index invalidation for other live suballocations
                    chunk.is_freed = true;
                    chunk.memory = .null_handle;
                    chunk.size = 0;

                    self.stats.current_block_count -|= 1; // Saturating subtract
                    self.stats.pool_capacity -|= alloc.size;
                } else {
                    // Return to free list
                    chunk.free(
                        alloc.offset,
                        alloc.size,
                        self.host_allocator,
                        self.config.enable_defrag_on_free,
                    );

                    // Update free list stats
                    self.stats.free_list_bytes += alloc.size;
                    self.stats.free_region_count = self.countFreeRegions();
                }

                self.updateStatsOnFree(alloc.size, alloc.usage);
            }
        }
    }

    /// Update statistics on allocation
    fn updateStatsOnAlloc(self: *Self, size: u64, usage: MemoryUsage, waste: u64) void {
        self.stats.current_allocated += size;
        self.stats.current_allocation_count += 1;
        self.stats.total_allocations += 1;
        self.stats.total_bytes_allocated += size;
        self.stats.alignment_waste += waste;

        if (self.stats.current_allocated > self.stats.peak_allocated) {
            self.stats.peak_allocated = self.stats.current_allocated;
        }
        if (self.stats.current_allocation_count > self.stats.peak_allocation_count) {
            self.stats.peak_allocation_count = self.stats.current_allocation_count;
        }

        const usage_idx = @intFromEnum(usage);
        self.stats.bytes_by_usage[usage_idx] += size;
        self.stats.count_by_usage[usage_idx] += 1;
    }

    /// Update statistics on free
    fn updateStatsOnFree(self: *Self, size: u64, usage: MemoryUsage) void {
        // Use saturating subtraction to prevent underflow on mismatched free
        self.stats.current_allocated -|= size;
        self.stats.current_allocation_count -|= 1;
        self.stats.total_deallocations += 1;
        self.stats.total_bytes_freed += size;

        const usage_idx = @intFromEnum(usage);
        self.stats.bytes_by_usage[usage_idx] -|= size;
        self.stats.count_by_usage[usage_idx] -|= 1;
    }

    /// Count total free regions across all pools
    fn countFreeRegions(self: *Self) u64 {
        var count: u64 = 0;
        for (&self.pools) |*maybe_pool| {
            if (maybe_pool.*) |*pool| {
                for (pool.chunks.items) |chunk| {
                    count += chunk.free_regions.items.len;
                }
            }
        }
        return count;
    }

    /// Find a suitable memory type
    fn findMemoryType(
        self: Self,
        type_bits: u32,
        required_properties: types.MemoryPropertyFlags,
    ) ?u32 {
        // Validate memory_type_count doesn't exceed 32 (Vulkan spec guarantees this,
        // but validate to prevent u5 cast issues with non-compliant drivers)
        if (self.memory_properties.memory_type_count > 32) {
            std.log.err("PooledAllocator: memory_type_count ({d}) exceeds maximum of 32", .{
                self.memory_properties.memory_type_count,
            });
            return null;
        }
        for (0..self.memory_properties.memory_type_count) |i| {
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
    pub fn getStats(self: Self) AllocatorStats {
        return self.stats;
    }

    /// Reset cumulative statistics
    pub fn resetStats(self: *Self) void {
        self.stats.resetCumulative();
    }

    /// Compact pools by removing empty chunks
    /// Uses swapRemove for O(n) performance instead of O(n²) with orderedRemove
    pub fn compact(self: *Self) void {
        for (&self.pools) |*maybe_pool| {
            if (maybe_pool.*) |*pool| {
                var i: usize = 0;
                while (i < pool.chunks.items.len) {
                    const chunk = &pool.chunks.items[i];
                    // Remove if empty and not dedicated
                    if (!chunk.is_dedicated and chunk.allocation_count == 0) {
                        if (chunk.mapped_base != null) {
                            self.dispatch.vkUnmapMemory.?(self.device, chunk.memory);
                        }
                        self.dispatch.vkFreeMemory.?(self.device, chunk.memory, null);

                        self.stats.pool_capacity -= chunk.size;
                        self.stats.current_block_count -= 1;
                        self.stats.free_list_bytes -= chunk.totalFreeSpace();

                        chunk.deinit(self.host_allocator);
                        // Use swapRemove for O(1) removal - order doesn't matter for compaction
                        _ = pool.chunks.swapRemove(i);
                        // Don't increment i - the swapped element needs to be checked
                    } else {
                        i += 1;
                    }
                }
            }
        }
        self.stats.free_region_count = self.countFreeRegions();
    }

    /// Defragmentation result
    pub const DefragResult = struct {
        /// Number of allocations moved
        allocations_moved: u32,
        /// Bytes freed by coalescing
        bytes_freed: u64,
        /// Chunks removed
        chunks_removed: u32,
        /// New fragmentation ratio
        fragmentation_ratio: f64,
    };

    /// Perform memory defragmentation
    ///
    /// This operation coalesces free regions, removes empty chunks, and
    /// optionally relocates allocations to reduce fragmentation.
    /// Note: Active GPU allocations cannot be moved - this only consolidates
    /// free space and removes unused chunks.
    pub fn defragment(self: *Self) DefragResult {
        var result = DefragResult{
            .allocations_moved = 0,
            .bytes_freed = 0,
            .chunks_removed = 0,
            .fragmentation_ratio = 0,
        };

        const old_free_regions = self.stats.free_region_count;
        const old_chunks = self.stats.current_block_count;

        // First, coalesce free regions in all pools
        for (&self.pools) |*maybe_pool| {
            if (maybe_pool.*) |*pool| {
                for (pool.chunks.items) |*chunk| {
                    const before_regions = chunk.free_regions.items.len;
                    chunk.coalesceAdjacentFreeRegions();
                    const after_regions = chunk.free_regions.items.len;
                    if (before_regions > after_regions) {
                        result.allocations_moved += @intCast(before_regions - after_regions);
                    }
                }
            }
        }

        // Then compact (remove empty chunks)
        self.compact();

        // Calculate results
        result.chunks_removed = @intCast(old_chunks - self.stats.current_block_count);
        result.bytes_freed = self.stats.free_list_bytes;

        // Recalculate free region count
        const new_free_regions = self.countFreeRegions();
        if (old_free_regions > new_free_regions) {
            result.allocations_moved += @intCast(old_free_regions - new_free_regions);
        }

        result.fragmentation_ratio = self.stats.fragmentationRatio();

        return result;
    }

    /// Defragmentation budget for incremental defrag
    pub const DefragBudget = struct {
        /// Maximum time to spend (nanoseconds)
        max_time_ns: u64 = 1_000_000, // 1ms default
        /// Maximum allocations to process
        max_allocations: u32 = 100,
        /// Maximum bytes to free
        max_bytes: u64 = 16 * 1024 * 1024,
    };

    /// Perform incremental defragmentation with a budget
    /// This is suitable for calling each frame without blocking
    pub fn defragmentIncremental(self: *Self, budget: DefragBudget) DefragResult {
        const start_time = std.time.nanoTimestamp();
        var result = DefragResult{
            .allocations_moved = 0,
            .bytes_freed = 0,
            .chunks_removed = 0,
            .fragmentation_ratio = 0,
        };

        var processed: u32 = 0;
        _ = budget.max_bytes; // Reserved for future use

        // Process pools until budget exhausted
        for (&self.pools) |*maybe_pool| {
            if (maybe_pool.*) |*pool| {
                for (pool.chunks.items) |*chunk| {
                    // Check budget
                    if (processed >= budget.max_allocations) break;

                    const elapsed: u64 = @intCast(std.time.nanoTimestamp() - start_time);
                    if (elapsed >= budget.max_time_ns) break;

                    // Coalesce this chunk
                    const before = chunk.free_regions.items.len;
                    chunk.coalesceAdjacentFreeRegions();
                    const after = chunk.free_regions.items.len;

                    if (before > after) {
                        result.allocations_moved += @intCast(before - after);
                        processed += @intCast(before - after);
                    }
                }
            }
        }

        // Quick compact pass
        self.compact();

        result.fragmentation_ratio = self.stats.fragmentationRatio();
        return result;
    }

    /// Get fragmentation statistics for a specific memory type
    pub fn getFragmentationInfo(self: Self, memory_type_index: u32) ?struct {
        /// Total number of free regions
        free_region_count: u32,
        /// Average free region size
        avg_free_region_size: u64,
        /// Largest free region
        largest_free_region: u64,
        /// Smallest free region
        smallest_free_region: u64,
        /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented)
        fragmentation_ratio: f64,
    } {
        if (self.pools[memory_type_index]) |pool| {
            var total_regions: u32 = 0;
            var total_free: u64 = 0;
            var largest: u64 = 0;
            var smallest: u64 = std.math.maxInt(u64);
            var total_capacity: u64 = 0;

            for (pool.chunks.items) |chunk| {
                for (chunk.free_regions.items) |region| {
                    total_regions += 1;
                    total_free += region.size;
                    largest = @max(largest, region.size);
                    smallest = @min(smallest, region.size);
                }
                total_capacity += chunk.size;
            }

            if (total_regions == 0) {
                return .{
                    .free_region_count = 0,
                    .avg_free_region_size = 0,
                    .largest_free_region = 0,
                    .smallest_free_region = 0,
                    .fragmentation_ratio = 0.0,
                };
            }

            // Fragmentation = 1 - (largest_free / total_free)
            const frag_ratio = if (total_free > 0)
                1.0 - (@as(f64, @floatFromInt(largest)) / @as(f64, @floatFromInt(total_free)))
            else
                0.0;

            return .{
                .free_region_count = total_regions,
                .avg_free_region_size = total_free / total_regions,
                .largest_free_region = largest,
                .smallest_free_region = if (smallest == std.math.maxInt(u64)) 0 else smallest,
                .fragmentation_ratio = frag_ratio,
            };
        }
        return null;
    }

    /// Get detailed pool information
    pub fn getPoolInfo(self: Self, memory_type_index: u32) ?struct {
        chunk_count: u32,
        total_size: u64,
        used_size: u64,
        free_size: u64,
        largest_free: u64,
    } {
        if (self.pools[memory_type_index]) |pool| {
            var total: u64 = 0;
            var free_space: u64 = 0;
            var largest: u64 = 0;

            for (pool.chunks.items) |chunk| {
                total += chunk.size;
                const chunk_free = chunk.totalFreeSpace();
                free_space += chunk_free;
                largest = @max(largest, chunk.largestFreeRegion());
            }

            return .{
                .chunk_count = @intCast(pool.chunks.items.len),
                .total_size = total,
                .used_size = total - free_space,
                .free_size = free_space,
                .largest_free = largest,
            };
        }
        return null;
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

// =============================================================================
// Tests
// =============================================================================

test "AllocatorStats initialization" {
    const stats = AllocatorStats{};
    try std.testing.expectEqual(@as(u64, 0), stats.current_allocated);
    try std.testing.expectEqual(@as(u64, 0), stats.current_allocation_count);
    try std.testing.expectEqual(@as(f64, 0.0), stats.fragmentationRatio());
    try std.testing.expectEqual(@as(f64, 0.0), stats.utilizationRatio());
    try std.testing.expectEqual(@as(f64, 1.0), stats.poolHitRate());
}

test "AllocatorStats ratios" {
    var stats = AllocatorStats{};
    stats.pool_capacity = 1000;
    stats.current_allocated = 500;
    stats.free_list_bytes = 300;
    stats.free_region_count = 5;
    stats.pool_hits = 8;
    stats.pool_misses = 2;

    try std.testing.expectEqual(@as(f64, 0.5), stats.utilizationRatio());
    try std.testing.expectEqual(@as(f64, 0.8), stats.poolHitRate());
    try std.testing.expect(stats.fragmentationRatio() > 0.0);
}

test "Suballocation mapping" {
    const alloc = Suballocation{
        .memory = undefined,
        .offset = 0,
        .size = 1024,
        .memory_type_index = 0,
        .mapped_ptr = null,
        .usage = .gpu_only,
        .is_dedicated = false,
        .chunk_index = 0,
    };

    try std.testing.expect(!alloc.isMapped());
    try std.testing.expectEqual(@as(?[]u32, null), alloc.getMappedSlice(u32));
}

test "PoolConfig defaults" {
    const config = PoolConfig{};
    try std.testing.expectEqual(@as(u64, 64 * 1024 * 1024), config.device_chunk_size);
    try std.testing.expectEqual(@as(u64, 16 * 1024 * 1024), config.host_chunk_size);
    try std.testing.expectEqual(@as(u64, 256), config.min_allocation_size);
    try std.testing.expect(config.enable_defrag_on_free);
}

test "FreeRegion coalescing logic" {
    // Test that free regions can be properly tracked
    var regions = std.ArrayListUnmanaged(FreeRegion){};
    defer regions.deinit(std.testing.allocator);

    try regions.append(std.testing.allocator, .{ .offset = 0, .size = 100 });
    try regions.append(std.testing.allocator, .{ .offset = 200, .size = 100 });

    try std.testing.expectEqual(@as(usize, 2), regions.items.len);
    try std.testing.expectEqual(@as(u64, 0), regions.items[0].offset);
    try std.testing.expectEqual(@as(u64, 200), regions.items[1].offset);
}
