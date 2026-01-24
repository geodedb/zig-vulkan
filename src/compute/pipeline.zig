// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Vulkan Compute Pipelines
//!
//! High-level compute pipeline management including shader modules,
//! descriptor set layouts, and pipeline creation.
//!
//! ## Features
//!
//! - SPIR-V shader module loading with validation
//! - Descriptor set layout and pool management
//! - Compute pipeline creation
//! - Descriptor set updates with bounds checking
//!
//! ## SPIR-V Validation
//!
//! In debug builds, shader modules are validated for:
//! - Correct magic number (0x07230203)
//! - Valid SPIR-V version (1.0 - 1.6)
//! - Valid ID bound
//! - Zero schema field
//!
//! ## Example
//!
//! ```zig
//! // Load shader from embedded SPIR-V
//! var shader = try ShaderModule.initFromBytes(&ctx, @embedFile("compute.spv"));
//! defer shader.deinit();
//!
//! // Create descriptor layout
//! var layout = try DescriptorSetLayout.init(&ctx, &.{
//!     .{ .binding = 0, .descriptor_type = .storage_buffer },
//!     .{ .binding = 1, .descriptor_type = .storage_buffer },
//! });
//! defer layout.deinit();
//!
//! // Create pipeline
//! var pipeline = try ComputePipeline.init(&ctx, shader, "main", layout, null);
//! defer pipeline.deinit();
//! ```

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../vk/types.zig");
const loader = @import("../vk/loader.zig");
const err = @import("../vk/error.zig");
const validation = @import("../util/validation.zig");
const hot_reload = @import("../util/hot_reload.zig");
const Context = @import("context.zig").Context;
const Buffer = @import("buffer.zig").Buffer;

/// Maximum shader/cache file size (100MB).
/// Prevents loading unreasonably large files that would exhaust memory.
/// This limit applies to SPIR-V shader files and pipeline cache files.
const max_shader_file_size: usize = 100 * 1024 * 1024;

/// SPIR-V shader module
pub const ShaderModule = struct {
    handle: types.ShaderModule,
    device: types.Device,
    dispatch: loader.DeviceDispatch,

    const Self = @This();

    /// Create a shader module from SPIR-V bytecode
    pub fn initFromSpirv(ctx: *Context, spirv: []const u32) err.Error!Self {
        if (!isValidSpirv(spirv)) {
            return err.Error.InvalidSpirv;
        }

        const create_info = types.ShaderModuleCreateInfo{
            .code_size = spirv.len * @sizeOf(u32),
            .p_code = spirv.ptr,
        };

        var handle: types.ShaderModule = undefined;
        const result = ctx.device_dispatch.vkCreateShaderModule.?(ctx.device, &create_info, null, &handle);
        if (result != .success) return err.fromVkResult(result);

        return Self{
            .handle = handle,
            .device = ctx.device,
            .dispatch = ctx.device_dispatch,
        };
    }

    /// Create a shader module from SPIR-V bytes (for embedded data)
    pub fn initFromBytes(ctx: *Context, bytes: []const u8) err.Error!Self {
        if (bytes.len % 4 != 0) {
            return err.Error.InvalidSpirv;
        }

        const spirv: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, bytes));
        return initFromSpirv(ctx, spirv);
    }

    /// Create a shader module from a file
    pub fn initFromFile(ctx: *Context, path: []const u8) err.Error!Self {
        const file = std.fs.cwd().openFile(path, .{}) catch return err.Error.InvalidSpirv;
        defer file.close();

        const stat = file.stat() catch return err.Error.InvalidSpirv;
        if (stat.size % 4 != 0 or stat.size > max_shader_file_size) {
            return err.Error.InvalidSpirv;
        }

        const buffer = ctx.allocator.alloc(u8, stat.size) catch return err.Error.OutOfHostMemory;
        defer ctx.allocator.free(buffer);

        const bytes_read = file.readAll(buffer) catch return err.Error.InvalidSpirv;
        if (bytes_read != stat.size) return err.Error.InvalidSpirv;

        return initFromBytes(ctx, buffer);
    }

    /// Destroy the shader module
    pub fn deinit(self: *Self) void {
        self.dispatch.vkDestroyShaderModule.?(self.device, self.handle, null);
        self.* = undefined;
    }
};

/// Descriptor binding specification
pub const DescriptorBinding = struct {
    binding: u32,
    descriptor_type: types.DescriptorType,
    count: u32 = 1,
};

/// Descriptor set layout
pub const DescriptorSetLayout = struct {
    handle: types.DescriptorSetLayout,
    device: types.Device,
    dispatch: loader.DeviceDispatch,
    bindings: []const DescriptorBinding,

    const Self = @This();

    /// Create a descriptor set layout
    ///
    /// Validates bindings against device limits in debug builds:
    /// - max_per_stage_descriptor_storage_buffers
    /// - max_per_stage_descriptor_uniform_buffers
    /// - max_per_stage_descriptor_sampled_images
    /// - max_per_stage_descriptor_storage_images
    pub fn init(ctx: *Context, bindings: []const DescriptorBinding) err.Error!Self {
        // Validate against device limits in debug builds
        if (builtin.mode == .Debug) {
            const limits = ctx.device_properties.limits;
            var storage_buffer_count: u32 = 0;
            var uniform_buffer_count: u32 = 0;
            var sampled_image_count: u32 = 0;
            var storage_image_count: u32 = 0;

            for (bindings) |b| {
                switch (b.descriptor_type) {
                    .storage_buffer, .storage_buffer_dynamic => {
                        storage_buffer_count += b.count;
                    },
                    .uniform_buffer, .uniform_buffer_dynamic => {
                        uniform_buffer_count += b.count;
                    },
                    .sampled_image, .combined_image_sampler => {
                        sampled_image_count += b.count;
                    },
                    .storage_image => {
                        storage_image_count += b.count;
                    },
                    else => {},
                }
            }

            if (storage_buffer_count > limits.max_per_stage_descriptor_storage_buffers) {
                std.log.err("[Vulkan] DescriptorSetLayout exceeds device limits: {d} storage buffers > max {d}", .{
                    storage_buffer_count,
                    limits.max_per_stage_descriptor_storage_buffers,
                });
                return err.Error.InvalidArgument;
            }
            if (uniform_buffer_count > limits.max_per_stage_descriptor_uniform_buffers) {
                std.log.err("[Vulkan] DescriptorSetLayout exceeds device limits: {d} uniform buffers > max {d}", .{
                    uniform_buffer_count,
                    limits.max_per_stage_descriptor_uniform_buffers,
                });
                return err.Error.InvalidArgument;
            }
            if (sampled_image_count > limits.max_per_stage_descriptor_sampled_images) {
                std.log.err("[Vulkan] DescriptorSetLayout exceeds device limits: {d} sampled images > max {d}", .{
                    sampled_image_count,
                    limits.max_per_stage_descriptor_sampled_images,
                });
                return err.Error.InvalidArgument;
            }
            if (storage_image_count > limits.max_per_stage_descriptor_storage_images) {
                std.log.err("[Vulkan] DescriptorSetLayout exceeds device limits: {d} storage images > max {d}", .{
                    storage_image_count,
                    limits.max_per_stage_descriptor_storage_images,
                });
                return err.Error.InvalidArgument;
            }
        }

        var vk_bindings = std.ArrayListUnmanaged(types.DescriptorSetLayoutBinding){};
        defer vk_bindings.deinit(ctx.allocator);

        for (bindings) |b| {
            vk_bindings.append(ctx.allocator, .{
                .binding = b.binding,
                .descriptor_type = b.descriptor_type,
                .descriptor_count = b.count,
                .stage_flags = .{ .compute = true },
            }) catch return err.Error.OutOfHostMemory;
        }

        const create_info = types.DescriptorSetLayoutCreateInfo{
            .binding_count = @intCast(vk_bindings.items.len),
            .p_bindings = vk_bindings.items.ptr,
        };

        var handle: types.DescriptorSetLayout = undefined;
        const result = ctx.device_dispatch.vkCreateDescriptorSetLayout.?(ctx.device, &create_info, null, &handle);
        if (result != .success) return err.fromVkResult(result);

        return Self{
            .handle = handle,
            .device = ctx.device,
            .dispatch = ctx.device_dispatch,
            .bindings = bindings,
        };
    }

    /// Destroy the layout
    pub fn deinit(self: *Self) void {
        self.dispatch.vkDestroyDescriptorSetLayout.?(self.device, self.handle, null);
        self.* = undefined;
    }
};

/// Descriptor pool for allocating descriptor sets
pub const DescriptorPool = struct {
    handle: types.DescriptorPool,
    device: types.Device,
    dispatch: loader.DeviceDispatch,

    const Self = @This();

    /// Create a descriptor pool
    pub fn init(ctx: *Context, max_sets: u32, pool_sizes: []const types.DescriptorPoolSize) err.Error!Self {
        const create_info = types.DescriptorPoolCreateInfo{
            .flags = .{ .free_descriptor_set = true },
            .max_sets = max_sets,
            .pool_size_count = @intCast(pool_sizes.len),
            .p_pool_sizes = pool_sizes.ptr,
        };

        var handle: types.DescriptorPool = undefined;
        const result = ctx.device_dispatch.vkCreateDescriptorPool.?(ctx.device, &create_info, null, &handle);
        if (result != .success) return err.fromVkResult(result);

        return Self{
            .handle = handle,
            .device = ctx.device,
            .dispatch = ctx.device_dispatch,
        };
    }

    /// Create a simple pool for storage buffers
    pub fn initSimple(ctx: *Context, max_sets: u32, max_storage_buffers: u32) err.Error!Self {
        const pool_sizes = [_]types.DescriptorPoolSize{
            .{ .type = .storage_buffer, .descriptor_count = max_storage_buffers },
            .{ .type = .uniform_buffer, .descriptor_count = max_sets },
        };
        return init(ctx, max_sets, &pool_sizes);
    }

    /// Allocate a descriptor set
    pub fn allocate(self: Self, layout: DescriptorSetLayout) err.Error!types.DescriptorSet {
        const alloc_info = types.DescriptorSetAllocateInfo{
            .descriptor_pool = self.handle,
            .descriptor_set_count = 1,
            .p_set_layouts = @ptrCast(&layout.handle),
        };

        var set: types.DescriptorSet = undefined;
        const result = self.dispatch.vkAllocateDescriptorSets.?(self.device, &alloc_info, @ptrCast(&set));
        if (result != .success) return err.fromVkResult(result);

        return set;
    }

    /// Free a descriptor set
    pub fn free(self: Self, set: types.DescriptorSet) void {
        const result = self.dispatch.vkFreeDescriptorSets.?(self.device, self.handle, 1, @ptrCast(&set));
        if (result != .success) {
            std.log.warn("DescriptorPool.free: vkFreeDescriptorSets failed: {s}", .{@tagName(result)});
        }
    }

    /// Destroy the pool
    pub fn deinit(self: *Self) void {
        self.dispatch.vkDestroyDescriptorPool.?(self.device, self.handle, null);
        self.* = undefined;
    }
};

/// Push constant range specification
pub const PushConstantRange = struct {
    offset: u32 = 0,
    size: u32,
};

/// Specialization constant entry
/// Maps a SPIR-V constant ID to a value in the data block
pub const SpecializationEntry = struct {
    /// The SPIR-V constant ID (from layout(constant_id = X))
    id: u32,
    /// Byte offset into the data block
    offset: u32,
    /// Size in bytes
    size: u32,
};

/// Specialization constants configuration
/// Allows compile-time optimization of shaders with constant values
pub const SpecializationConfig = struct {
    /// Array of specialization entries
    entries: []const SpecializationEntry,
    /// Raw data block containing all constant values
    data: []const u8,
};

/// Pipeline cache for faster pipeline creation
///
/// Caches compiled pipeline data to disk, dramatically speeding up
/// subsequent pipeline creation. The cache is automatically populated
/// when pipelines are created with a cache, and can be saved/loaded.
pub const PipelineCache = struct {
    handle: types.PipelineCache,
    device: types.Device,
    dispatch: loader.DeviceDispatch,

    const Self = @This();

    /// Create an empty pipeline cache
    pub fn init(ctx: *Context) err.Error!Self {
        const create_info = types.PipelineCacheCreateInfo{
            .initial_data_size = 0,
            .p_initial_data = null,
        };

        var handle: types.PipelineCache = undefined;
        const result = ctx.device_dispatch.vkCreatePipelineCache.?(ctx.device, &create_info, null, &handle);
        if (result != .success) return err.fromVkResult(result);

        return Self{
            .handle = handle,
            .device = ctx.device,
            .dispatch = ctx.device_dispatch,
        };
    }

    /// Create a pipeline cache from saved data
    pub fn initFromData(ctx: *Context, data: []const u8) err.Error!Self {
        const create_info = types.PipelineCacheCreateInfo{
            .initial_data_size = data.len,
            .p_initial_data = data.ptr,
        };

        var handle: types.PipelineCache = undefined;
        const result = ctx.device_dispatch.vkCreatePipelineCache.?(ctx.device, &create_info, null, &handle);
        if (result != .success) return err.fromVkResult(result);

        return Self{
            .handle = handle,
            .device = ctx.device,
            .dispatch = ctx.device_dispatch,
        };
    }

    /// Load cache from file
    pub fn initFromFile(ctx: *Context, path: []const u8) err.Error!Self {
        const file = std.fs.cwd().openFile(path, .{}) catch {
            // File doesn't exist, create empty cache
            return init(ctx);
        };
        defer file.close();

        const stat = file.stat() catch return init(ctx);
        if (stat.size == 0 or stat.size > max_shader_file_size) {
            return init(ctx);
        }

        const buffer = ctx.allocator.alloc(u8, stat.size) catch return err.Error.OutOfHostMemory;
        defer ctx.allocator.free(buffer);

        const bytes_read = file.readAll(buffer) catch return init(ctx);
        if (bytes_read != stat.size) return init(ctx);

        return initFromData(ctx, buffer);
    }

    /// Get cache data for saving
    pub fn getData(self: Self, allocator: std.mem.Allocator) err.Error![]u8 {
        var size: usize = 0;
        var result = self.dispatch.vkGetPipelineCacheData.?(self.device, self.handle, &size, null);
        if (result != .success) return err.fromVkResult(result);

        const data = allocator.alloc(u8, size) catch return err.Error.OutOfHostMemory;
        errdefer allocator.free(data);

        result = self.dispatch.vkGetPipelineCacheData.?(self.device, self.handle, &size, data.ptr);
        if (result != .success) return err.fromVkResult(result);

        return data;
    }

    /// Save cache to file
    pub fn saveToFile(self: Self, allocator: std.mem.Allocator, path: []const u8) err.Error!void {
        const data = try self.getData(allocator);
        defer allocator.free(data);

        const file = std.fs.cwd().createFile(path, .{}) catch return err.Error.Unknown;
        defer file.close();

        file.writeAll(data) catch return err.Error.Unknown;
    }

    /// Merge another cache into this one
    pub fn merge(self: Self, other: Self) err.Error!void {
        const result = self.dispatch.vkMergePipelineCaches.?(
            self.device,
            self.handle,
            1,
            @ptrCast(&other.handle),
        );
        if (result != .success) return err.fromVkResult(result);
    }

    /// Destroy the cache
    pub fn deinit(self: *Self) void {
        self.dispatch.vkDestroyPipelineCache.?(self.device, self.handle, null);
        self.* = undefined;
    }
};

/// Persistent pipeline cache with automatic disk save/load
///
/// This wrapper around PipelineCache provides:
/// - Automatic loading from disk on init
/// - Automatic saving to disk on deinit
/// - Cache validation against the current device
/// - Statistics tracking
/// - Graceful handling of corrupted cache files
///
/// ## Usage
///
/// ```zig
/// var cache = try PersistentPipelineCache.init(ctx, .{
///     .cache_path = "pipeline_cache.bin",
///     .auto_save = true,
/// });
/// defer cache.deinit();
///
/// // Use cache.getHandle() for pipeline creation
/// var pipeline = try ComputePipeline.initWithCache(ctx, shader, "main", layout, null, cache.getHandle());
/// ```
pub const PersistentPipelineCache = struct {
    inner: PipelineCache,
    ctx: *Context,
    config: PersistentCacheConfig,
    stats: CacheStats,
    loaded_from_disk: bool,

    const Self = @This();

    /// Configuration for persistent pipeline cache
    pub const PersistentCacheConfig = struct {
        /// Path to cache file
        cache_path: []const u8,
        /// Automatically save cache on deinit
        auto_save: bool = true,
        /// Maximum cache file size (bytes)
        max_file_size: usize = max_shader_file_size,
        /// Validate cache header against current device
        validate_header: bool = true,
        /// Delete cache if validation fails
        delete_on_invalid: bool = true,
        /// Create parent directories if needed
        create_dirs: bool = true,
    };

    /// Cache statistics
    pub const CacheStats = struct {
        /// Whether cache was loaded from disk
        loaded_from_disk: bool = false,
        /// Size of loaded cache data
        loaded_size: usize = 0,
        /// Number of pipelines created with this cache
        pipelines_created: u32 = 0,
        /// Number of cache hits (estimated)
        estimated_hits: u32 = 0,
        /// Load time in nanoseconds
        load_time_ns: i128 = 0,
        /// Whether cache was validated successfully
        validated: bool = false,
    };

    /// Initialize persistent cache
    pub fn init(ctx: *Context, config: PersistentCacheConfig) err.Error!Self {
        const start_time = std.time.nanoTimestamp();
        var stats = CacheStats{};

        // Try to load from disk
        const inner = blk: {
            if (config.create_dirs) {
                // Ensure parent directory exists
                if (std.fs.path.dirname(config.cache_path)) |dir| {
                    std.fs.cwd().makePath(dir) catch |e| {
                        std.log.debug("PipelineCache: could not create directory {s}: {s}", .{ dir, @errorName(e) });
                    };
                }
            }

            // Try loading existing cache
            const file = std.fs.cwd().openFile(config.cache_path, .{}) catch {
                // File doesn't exist, create empty cache
                break :blk try PipelineCache.init(ctx);
            };
            defer file.close();

            const file_stat = file.stat() catch {
                break :blk try PipelineCache.init(ctx);
            };

            if (file_stat.size == 0 or file_stat.size > config.max_file_size) {
                if (config.delete_on_invalid) {
                    std.fs.cwd().deleteFile(config.cache_path) catch |e| {
                        std.log.debug("PipelineCache: could not delete invalid cache: {s}", .{@errorName(e)});
                    };
                }
                break :blk try PipelineCache.init(ctx);
            }

            const buffer = ctx.allocator.alloc(u8, file_stat.size) catch {
                break :blk try PipelineCache.init(ctx);
            };
            defer ctx.allocator.free(buffer);

            const bytes_read = file.readAll(buffer) catch {
                break :blk try PipelineCache.init(ctx);
            };

            if (bytes_read != file_stat.size) {
                break :blk try PipelineCache.init(ctx);
            }

            // Validate cache header if requested
            if (config.validate_header) {
                if (!validateCacheHeader(buffer, ctx)) {
                    if (config.delete_on_invalid) {
                        std.fs.cwd().deleteFile(config.cache_path) catch |e| {
                            std.log.debug("PipelineCache: could not delete invalid cache after header validation: {s}", .{@errorName(e)});
                        };
                    }
                    break :blk try PipelineCache.init(ctx);
                }
                stats.validated = true;
            }

            stats.loaded_from_disk = true;
            stats.loaded_size = bytes_read;

            break :blk PipelineCache.initFromData(ctx, buffer) catch {
                // Invalid cache data, create empty
                if (config.delete_on_invalid) {
                    std.fs.cwd().deleteFile(config.cache_path) catch |e| {
                        std.log.debug("PipelineCache: could not delete invalid cache after load failure: {s}", .{@errorName(e)});
                    };
                }
                break :blk try PipelineCache.init(ctx);
            };
        };

        stats.load_time_ns = std.time.nanoTimestamp() - start_time;

        return Self{
            .inner = inner,
            .ctx = ctx,
            .config = config,
            .stats = stats,
            .loaded_from_disk = stats.loaded_from_disk,
        };
    }

    /// Validate pipeline cache header against device properties
    fn validateCacheHeader(data: []const u8, ctx: *Context) bool {
        // Pipeline cache header format (Vulkan spec):
        // - 4 bytes: header size
        // - 4 bytes: header version (should be VK_PIPELINE_CACHE_HEADER_VERSION_ONE = 1)
        // - 4 bytes: vendor ID
        // - 4 bytes: device ID
        // - 16 bytes: pipeline cache UUID

        if (data.len < 32) return false;

        const header_size = std.mem.readInt(u32, data[0..4], .little);
        if (header_size < 32) return false;

        const header_version = std.mem.readInt(u32, data[4..8], .little);
        if (header_version != 1) return false; // VK_PIPELINE_CACHE_HEADER_VERSION_ONE

        const vendor_id = std.mem.readInt(u32, data[8..12], .little);
        const device_id = std.mem.readInt(u32, data[12..16], .little);

        // Compare with current device
        if (vendor_id != ctx.device_properties.vendor_id) return false;
        if (device_id != ctx.device_properties.device_id) return false;

        // Compare pipeline cache UUID
        const cache_uuid = data[16..32];
        if (!std.mem.eql(u8, cache_uuid, &ctx.device_properties.pipeline_cache_uuid)) return false;

        return true;
    }

    /// Get the underlying pipeline cache handle
    pub fn getHandle(self: Self) types.PipelineCache {
        return self.inner.handle;
    }

    /// Get the underlying PipelineCache
    pub fn getInner(self: *Self) *PipelineCache {
        return &self.inner;
    }

    /// Record that a pipeline was created with this cache
    pub fn recordPipelineCreated(self: *Self) void {
        self.stats.pipelines_created += 1;
        // Estimate cache hit if we loaded from disk
        if (self.loaded_from_disk) {
            self.stats.estimated_hits += 1;
        }
    }

    /// Save cache to disk
    pub fn save(self: Self) err.Error!void {
        const data = try self.inner.getData(self.ctx.allocator);
        defer self.ctx.allocator.free(data);

        if (self.config.create_dirs) {
            if (std.fs.path.dirname(self.config.cache_path)) |dir| {
                std.fs.cwd().makePath(dir) catch |e| {
                    std.log.debug("PipelineCache.save: could not create directory: {s}", .{@errorName(e)});
                };
            }
        }

        const file = std.fs.cwd().createFile(self.config.cache_path, .{}) catch {
            return err.Error.Unknown;
        };
        defer file.close();

        file.writeAll(data) catch {
            return err.Error.Unknown;
        };
    }

    /// Get cache statistics
    pub fn getStats(self: Self) CacheStats {
        return self.stats;
    }

    /// Merge another cache into this one
    pub fn merge(self: Self, other: PipelineCache) err.Error!void {
        return self.inner.merge(other);
    }

    /// Deinit and optionally save
    /// Note: If auto_save is enabled and save fails, the error is logged but cleanup proceeds.
    pub fn deinit(self: *Self) void {
        if (self.config.auto_save) {
            self.save() catch |e| {
                std.log.warn("PersistentPipelineCache.deinit: auto-save failed: {s}", .{@errorName(e)});
            };
        }
        self.inner.deinit();
    }
};

/// Compute pipeline
pub const ComputePipeline = struct {
    pipeline: types.Pipeline,
    layout: types.PipelineLayout,
    device: types.Device,
    device_dispatch: loader.DeviceDispatch,

    const Self = @This();

    /// Create a compute pipeline
    pub fn init(
        ctx: *Context,
        shader: ShaderModule,
        entry_point: [*:0]const u8,
        descriptor_layout: DescriptorSetLayout,
        push_constant_range: ?PushConstantRange,
    ) err.Error!Self {
        // Create pipeline layout
        var push_range: types.PushConstantRange = undefined;
        const push_range_ptr: ?[*]const types.PushConstantRange = if (push_constant_range) |pcr| blk: {
            push_range = .{
                .stage_flags = .{ .compute = true },
                .offset = pcr.offset,
                .size = pcr.size,
            };
            break :blk @ptrCast(&push_range);
        } else null;

        const layout_info = types.PipelineLayoutCreateInfo{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&descriptor_layout.handle),
            .push_constant_range_count = if (push_constant_range != null) 1 else 0,
            .p_push_constant_ranges = push_range_ptr,
        };

        var layout: types.PipelineLayout = undefined;
        var result = ctx.device_dispatch.vkCreatePipelineLayout.?(ctx.device, &layout_info, null, &layout);
        if (result != .success) return err.fromVkResult(result);
        errdefer ctx.device_dispatch.vkDestroyPipelineLayout.?(ctx.device, layout, null);

        // Create pipeline
        const stage_info = types.PipelineShaderStageCreateInfo{
            .stage = .{ .compute = true },
            .module = shader.handle,
            .p_name = entry_point,
        };

        const pipeline_info = types.ComputePipelineCreateInfo{
            .stage = stage_info,
            .layout = layout,
        };

        var pipeline: types.Pipeline = undefined;
        result = ctx.device_dispatch.vkCreateComputePipelines.?(
            ctx.device,
            null,
            1,
            @ptrCast(&pipeline_info),
            null,
            @ptrCast(&pipeline),
        );
        if (result != .success) return err.fromVkResult(result);

        return Self{
            .pipeline = pipeline,
            .layout = layout,
            .device = ctx.device,
            .device_dispatch = ctx.device_dispatch,
        };
    }

    /// Create a compute pipeline with pipeline cache
    ///
    /// Using a pipeline cache can significantly speed up pipeline creation
    /// on subsequent runs, as the driver can reuse compiled shader code.
    pub fn initWithCache(
        ctx: *Context,
        shader: ShaderModule,
        entry_point: [*:0]const u8,
        descriptor_layout: DescriptorSetLayout,
        push_constant_range: ?PushConstantRange,
        cache: PipelineCache,
    ) err.Error!Self {
        // Create pipeline layout
        var push_range: types.PushConstantRange = undefined;
        const push_range_ptr: ?[*]const types.PushConstantRange = if (push_constant_range) |pcr| blk: {
            push_range = .{
                .stage_flags = .{ .compute = true },
                .offset = pcr.offset,
                .size = pcr.size,
            };
            break :blk @ptrCast(&push_range);
        } else null;

        const layout_info = types.PipelineLayoutCreateInfo{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&descriptor_layout.handle),
            .push_constant_range_count = if (push_constant_range != null) 1 else 0,
            .p_push_constant_ranges = push_range_ptr,
        };

        var layout: types.PipelineLayout = undefined;
        var result = ctx.device_dispatch.vkCreatePipelineLayout.?(ctx.device, &layout_info, null, &layout);
        if (result != .success) return err.fromVkResult(result);
        errdefer ctx.device_dispatch.vkDestroyPipelineLayout.?(ctx.device, layout, null);

        // Create pipeline with cache
        const stage_info = types.PipelineShaderStageCreateInfo{
            .stage = .{ .compute = true },
            .module = shader.handle,
            .p_name = entry_point,
        };

        const pipeline_info = types.ComputePipelineCreateInfo{
            .stage = stage_info,
            .layout = layout,
        };

        var pipeline: types.Pipeline = undefined;
        result = ctx.device_dispatch.vkCreateComputePipelines.?(
            ctx.device,
            cache.handle, // Use pipeline cache
            1,
            @ptrCast(&pipeline_info),
            null,
            @ptrCast(&pipeline),
        );
        if (result != .success) return err.fromVkResult(result);

        return Self{
            .pipeline = pipeline,
            .layout = layout,
            .device = ctx.device,
            .device_dispatch = ctx.device_dispatch,
        };
    }

    /// Create a compute pipeline with specialization constants
    ///
    /// Specialization constants allow compile-time optimization by providing constant
    /// values that the shader compiler can use for optimization (loop unrolling,
    /// branch elimination, etc.).
    ///
    /// ## Parameters
    /// - `ctx`: Compute context
    /// - `shader`: Compiled shader module
    /// - `entry_point`: Shader entry point name (typically "main")
    /// - `descriptor_layout`: Descriptor set layout for shader bindings
    /// - `push_constant_range`: Optional push constant configuration
    /// - `specialization`: Specialization constants configuration
    ///
    /// ## Example
    /// ```zig
    /// // Define specialization data
    /// const SpecData = extern struct {
    ///     workgroup_size: u32,
    ///     use_fast_path: u32, // 0 or 1 for bool
    /// };
    /// const spec_data = SpecData{ .workgroup_size = 256, .use_fast_path = 1 };
    ///
    /// const spec_config = SpecializationConfig{
    ///     .entries = &.{
    ///         .{ .id = 0, .offset = 0, .size = 4 },  // workgroup_size
    ///         .{ .id = 1, .offset = 4, .size = 4 },  // use_fast_path
    ///     },
    ///     .data = std.mem.asBytes(&spec_data),
    /// };
    ///
    /// var pipeline = try ComputePipeline.initWithSpecialization(
    ///     &ctx, shader, "main", layout, null, spec_config
    /// );
    /// ```
    pub fn initWithSpecialization(
        ctx: *Context,
        shader: ShaderModule,
        entry_point: [*:0]const u8,
        descriptor_layout: DescriptorSetLayout,
        push_constant_range: ?PushConstantRange,
        specialization: SpecializationConfig,
    ) err.Error!Self {
        // Create pipeline layout
        var push_range: types.PushConstantRange = undefined;
        const push_range_ptr: ?[*]const types.PushConstantRange = if (push_constant_range) |pcr| blk: {
            push_range = .{
                .stage_flags = .{ .compute = true },
                .offset = pcr.offset,
                .size = pcr.size,
            };
            break :blk @ptrCast(&push_range);
        } else null;

        const layout_info = types.PipelineLayoutCreateInfo{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&descriptor_layout.handle),
            .push_constant_range_count = if (push_constant_range != null) 1 else 0,
            .p_push_constant_ranges = push_range_ptr,
        };

        var layout: types.PipelineLayout = undefined;
        var result = ctx.device_dispatch.vkCreatePipelineLayout.?(ctx.device, &layout_info, null, &layout);
        if (result != .success) return err.fromVkResult(result);
        errdefer ctx.device_dispatch.vkDestroyPipelineLayout.?(ctx.device, layout, null);

        // Build specialization map entries (zero-initialized for safety)
        var spec_entries: [32]types.SpecializationMapEntry = std.mem.zeroes([32]types.SpecializationMapEntry);
        const entry_count = @min(specialization.entries.len, 32);

        // Warn if specialization entries are being truncated
        if (builtin.mode == .Debug and specialization.entries.len > 32) {
            std.log.warn("[Vulkan] Specialization entries truncated: {d} entries provided, max is 32", .{
                specialization.entries.len,
            });
        }

        for (specialization.entries[0..entry_count], 0..) |entry, i| {
            spec_entries[i] = .{
                .constant_id = entry.id,
                .offset = entry.offset,
                .size = entry.size,
            };
        }

        const spec_info = types.SpecializationInfo{
            .map_entry_count = @intCast(entry_count),
            .p_map_entries = &spec_entries,
            .data_size = specialization.data.len,
            .p_data = specialization.data.ptr,
        };

        // Create pipeline with specialization
        const stage_info = types.PipelineShaderStageCreateInfo{
            .stage = .{ .compute = true },
            .module = shader.handle,
            .p_name = entry_point,
            .p_specialization_info = &spec_info,
        };

        const pipeline_info = types.ComputePipelineCreateInfo{
            .stage = stage_info,
            .layout = layout,
        };

        var pipeline: types.Pipeline = undefined;
        result = ctx.device_dispatch.vkCreateComputePipelines.?(
            ctx.device,
            null,
            1,
            @ptrCast(&pipeline_info),
            null,
            @ptrCast(&pipeline),
        );
        if (result != .success) return err.fromVkResult(result);

        return Self{
            .pipeline = pipeline,
            .layout = layout,
            .device = ctx.device,
            .device_dispatch = ctx.device_dispatch,
        };
    }

    /// Bind the pipeline to a command buffer
    pub fn bind(self: Self, cmd: types.CommandBuffer) void {
        self.device_dispatch.vkCmdBindPipeline.?(cmd, .compute, self.pipeline);
    }

    /// Bind descriptor sets
    pub fn bindDescriptorSet(self: Self, cmd: types.CommandBuffer, set: types.DescriptorSet) void {
        self.device_dispatch.vkCmdBindDescriptorSets.?(
            cmd,
            .compute,
            self.layout,
            0,
            1,
            @ptrCast(&set),
            0,
            null,
        );
    }

    /// Push constants
    pub fn pushConstants(self: Self, cmd: types.CommandBuffer, comptime T: type, data: *const T) void {
        self.device_dispatch.vkCmdPushConstants.?(
            cmd,
            self.layout,
            .{ .compute = true },
            0,
            @sizeOf(T),
            data,
        );
    }

    /// Dispatch compute workgroups
    pub fn dispatch(self: Self, cmd: types.CommandBuffer, x: u32, y: u32, z: u32) void {
        self.device_dispatch.vkCmdDispatch.?(cmd, x, y, z);
    }

    /// Dispatch compute workgroups indirectly from a buffer
    ///
    /// The buffer must contain a DispatchIndirectCommand structure at the given offset.
    /// This enables GPU-driven workloads where the dispatch parameters are computed
    /// by previous compute passes.
    ///
    /// ## Parameters
    /// - `cmd`: Command buffer to record the dispatch
    /// - `buffer`: Buffer containing DispatchIndirectCommand
    /// - `offset`: Byte offset into the buffer (must be a multiple of 4)
    ///
    /// ## Example
    /// ```zig
    /// // Prepare indirect buffer with dispatch parameters
    /// var indirect_buf = try Buffer.init(&ctx, @sizeOf(DispatchIndirectCommand), .storage);
    /// defer indirect_buf.deinit(&ctx);
    ///
    /// // GPU writes dispatch parameters...
    ///
    /// // Dispatch using buffer contents
    /// pipeline.dispatchIndirect(cmd, indirect_buf, 0);
    /// ```
    pub fn dispatchIndirect(self: Self, cmd: types.CommandBuffer, buffer: Buffer, offset: u64) void {
        self.device_dispatch.vkCmdDispatchIndirect.?(cmd, buffer.handle, offset);
    }

    /// Dispatch compute workgroups indirectly using a raw buffer handle
    pub fn dispatchIndirectRaw(self: Self, cmd: types.CommandBuffer, buffer: types.Buffer, offset: u64) void {
        self.device_dispatch.vkCmdDispatchIndirect.?(cmd, buffer, offset);
    }

    /// Destroy the pipeline
    pub fn deinit(self: *Self) void {
        self.device_dispatch.vkDestroyPipeline.?(self.device, self.pipeline, null);
        self.device_dispatch.vkDestroyPipelineLayout.?(self.device, self.layout, null);
        self.* = undefined;
    }
};

/// Maximum number of bindings in a single update call
pub const MAX_DESCRIPTOR_BINDINGS: usize = 16;

/// Simple descriptor binding (binding index, buffer, and type)
pub const SimpleDescriptorBinding = struct {
    binding: u32,
    buffer: Buffer,
    descriptor_type: types.DescriptorType,
};

/// Descriptor binding for updates with optional offset/range
pub const DescriptorBufferBinding = struct {
    binding: u32,
    buffer: Buffer,
    descriptor_type: types.DescriptorType,
    /// Optional offset within buffer (default 0)
    offset: u64 = 0,
    /// Optional range (default is entire buffer from offset)
    range: ?u64 = null,
};

/// Update a descriptor set with buffer bindings
///
/// Binds buffers to descriptor set bindings. In debug builds, validates:
/// - Number of bindings doesn't exceed MAX_DESCRIPTOR_BINDINGS (16)
/// - Buffer sizes are within device limits for the descriptor type
/// - Binding indices are unique
///
/// ## Parameters
/// - `dispatch`: Device dispatch table
/// - `device`: Vulkan device
/// - `set`: Descriptor set to update
/// - `bindings`: Array of binding specifications
///
/// ## Example
/// ```zig
/// updateDescriptorSet(ctx.device_dispatch, ctx.device, descriptor_set, &.{
///     .{ .binding = 0, .buffer = input_buffer, .descriptor_type = .storage_buffer },
///     .{ .binding = 1, .buffer = output_buffer, .descriptor_type = .storage_buffer },
/// });
/// ```
///
/// ## API Variants
///
/// This module provides three descriptor update functions:
/// - `updateDescriptorSet`: Simple bindings (recommended for most cases)
/// - `updateDescriptorSetWithContext`: Context-aware with automatic validation
/// - `updateDescriptorSetAdvanced`: Full control over offsets and ranges
///
/// Use `updateDescriptorSetAdvanced` when you need:
/// - Partial buffer bindings (offset != 0)
/// - Explicit range specification
/// - Maximum performance without validation overhead
pub fn updateDescriptorSet(
    dispatch: loader.DeviceDispatch,
    device: types.Device,
    set: types.DescriptorSet,
    bindings: []const SimpleDescriptorBinding,
) void {
    updateDescriptorSetEx(dispatch, device, set, bindings, null);
}

/// Update a descriptor set with context for automatic validation
///
/// Recommended function that uses the Context to automatically validate
/// buffer sizes against device limits in debug builds.
pub fn updateDescriptorSetWithContext(
    ctx: *Context,
    set: types.DescriptorSet,
    bindings: []const SimpleDescriptorBinding,
) void {
    updateDescriptorSetEx(ctx.device_dispatch, ctx.device, set, bindings, ctx.device_properties.limits);
}

/// Update a descriptor set with buffer bindings and optional validation
///
/// Extended version that accepts device limits for bounds checking.
/// When `limits` is provided and validation is enabled (debug builds),
/// validates buffer sizes against device limits.
///
/// Note: Prefer `updateDescriptorSetWithContext` which automatically provides limits.
pub fn updateDescriptorSetEx(
    dispatch: loader.DeviceDispatch,
    device: types.Device,
    set: types.DescriptorSet,
    bindings: []const SimpleDescriptorBinding,
    limits: ?types.PhysicalDeviceLimits,
) void {
    var buffer_infos: [MAX_DESCRIPTOR_BINDINGS]types.DescriptorBufferInfo = undefined;
    var writes: [MAX_DESCRIPTOR_BINDINGS]types.WriteDescriptorSet = undefined;

    const count = @min(bindings.len, MAX_DESCRIPTOR_BINDINGS);

    // Debug validation
    if (builtin.mode == .Debug) {
        // Warn if bindings are being truncated
        if (bindings.len > MAX_DESCRIPTOR_BINDINGS) {
            std.log.warn("[Vulkan] Descriptor bindings truncated: {d} bindings provided, max is {d}", .{
                bindings.len,
                MAX_DESCRIPTOR_BINDINGS,
            });
        }

        // Check for duplicate bindings
        for (bindings[0..count], 0..) |b1, i| {
            for (bindings[i + 1 .. count]) |b2| {
                if (b1.binding == b2.binding) {
                    std.log.warn("[Vulkan] Duplicate descriptor binding {d}", .{b1.binding});
                }
            }
        }

        // Validate buffer sizes against limits
        if (limits) |lim| {
            for (bindings[0..count]) |b| {
                const max_range: u64 = switch (b.descriptor_type) {
                    .storage_buffer, .storage_buffer_dynamic => lim.max_storage_buffer_range,
                    .uniform_buffer, .uniform_buffer_dynamic => lim.max_uniform_buffer_range,
                    else => std.math.maxInt(u32),
                };
                if (b.buffer.size > max_range) {
                    std.log.warn("[Vulkan] Buffer at binding {d} exceeds max range ({d} > {d})", .{
                        b.binding,
                        b.buffer.size,
                        max_range,
                    });
                }
            }
        }
    }

    for (bindings[0..count], 0..) |b, i| {
        buffer_infos[i] = .{
            .buffer = b.buffer.handle,
            .offset = 0,
            .range = b.buffer.size,
        };
        writes[i] = .{
            .dst_set = set,
            .dst_binding = b.binding,
            .descriptor_count = 1,
            .descriptor_type = b.descriptor_type,
            .p_buffer_info = &buffer_infos[i],
        };
    }

    dispatch.vkUpdateDescriptorSets.?(device, @intCast(count), &writes, 0, null);
}

/// Update a descriptor set with advanced binding options
///
/// Provides full control over buffer offsets and ranges.
pub fn updateDescriptorSetAdvanced(
    dispatch: loader.DeviceDispatch,
    device: types.Device,
    set: types.DescriptorSet,
    bindings: []const DescriptorBufferBinding,
) void {
    var buffer_infos: [MAX_DESCRIPTOR_BINDINGS]types.DescriptorBufferInfo = undefined;
    var writes: [MAX_DESCRIPTOR_BINDINGS]types.WriteDescriptorSet = undefined;

    const count = @min(bindings.len, MAX_DESCRIPTOR_BINDINGS);

    // Warn if bindings are being truncated (debug builds only)
    if (builtin.mode == .Debug and bindings.len > MAX_DESCRIPTOR_BINDINGS) {
        std.log.warn("[Vulkan] Advanced descriptor bindings truncated: {d} bindings provided, max is {d}", .{
            bindings.len,
            MAX_DESCRIPTOR_BINDINGS,
        });
    }

    for (bindings[0..count], 0..) |b, i| {
        // Validate offset doesn't exceed buffer size to prevent underflow
        if (b.offset > b.buffer.size) {
            std.log.err("[Vulkan] Descriptor binding {d}: offset ({d}) exceeds buffer size ({d})", .{
                b.binding,
                b.offset,
                b.buffer.size,
            });
            return; // Skip invalid binding
        }
        const range = b.range orelse (b.buffer.size - b.offset);
        buffer_infos[i] = .{
            .buffer = b.buffer.handle,
            .offset = b.offset,
            .range = range,
        };
        writes[i] = .{
            .dst_set = set,
            .dst_binding = b.binding,
            .descriptor_count = 1,
            .descriptor_type = b.descriptor_type,
            .p_buffer_info = &buffer_infos[i],
        };
    }

    dispatch.vkUpdateDescriptorSets.?(device, @intCast(count), &writes, 0, null);
}

/// Validate SPIR-V bytecode
///
/// Performs validation of SPIR-V shader code:
/// - Magic number verification (0x07230203)
/// - Version validation (1.0 - 1.6)
/// - ID bound validation
/// - Schema validation (must be 0)
///
/// In debug builds, additional instruction scanning is performed.
fn isValidSpirv(spirv_code: []const u32) bool {
    const result = validation.validateSpirv(spirv_code);
    return result == .valid;
}

/// Get detailed SPIR-V validation information
///
/// Returns a `SpirvInfo` struct with version, bound, and validation status.
/// Useful for debugging shader loading issues.
pub fn validateSpirvDetailed(spirv_code: []const u32) validation.SpirvInfo {
    return validation.validateSpirvDetailed(spirv_code);
}

// =============================================================================
// Hot-Reloadable Pipeline
// =============================================================================

/// Configuration for hot-reloadable pipelines
pub const HotReloadConfig = struct {
    /// Path to the SPIR-V shader file
    shader_path: []const u8,
    /// Shader entry point (typically "main")
    entry_point: [*:0]const u8 = "main",
    /// Poll interval for file changes (in nanoseconds)
    poll_interval_ns: u64 = 500_000_000, // 500ms default
    /// Whether to use content hash for change detection (slower but more reliable)
    use_content_hash: bool = false,
    /// Optional callback when shader is reloaded
    on_reload: ?*const fn (success: bool, user_data: ?*anyopaque) void = null,
    /// User data for callbacks
    user_data: ?*anyopaque = null,
};

/// Hot-reloadable compute pipeline with automatic shader file watching
///
/// Wraps a ComputePipeline with automatic shader file watching and
/// pipeline recreation when the shader source changes.
///
/// ## Example
/// ```zig
/// var hot_pipeline = try HotReloadablePipeline.init(&ctx, layout, push_range, .{
///     .shader_path = "shaders/compute.spv",
///     .entry_point = "main",
/// });
/// defer hot_pipeline.deinit();
///
/// // In main loop
/// try hot_pipeline.poll(); // Check for shader changes
/// hot_pipeline.bind(cmd);  // Bind current pipeline
/// ```
pub const HotReloadablePipeline = struct {
    ctx: *Context,
    allocator: std.mem.Allocator,

    /// Current active pipeline
    pipeline: ComputePipeline,
    /// Pending pipeline (being compiled)
    pending_pipeline: ?ComputePipeline = null,

    /// Shader watcher for file monitoring
    watcher: hot_reload.ShaderWatcher,

    /// Configuration
    config: HotReloadConfig,

    /// Descriptor layout (needed for recreation)
    descriptor_layout: DescriptorSetLayout,
    /// Push constant range (needed for recreation)
    push_constant_range: ?PushConstantRange,

    /// Statistics
    reload_count: u32 = 0,
    last_reload_time_ns: i128 = 0,
    last_reload_success: bool = true,

    const Self = @This();

    /// Initialize a hot-reloadable pipeline
    pub fn init(
        ctx: *Context,
        allocator: std.mem.Allocator,
        descriptor_layout: DescriptorSetLayout,
        push_constant_range: ?PushConstantRange,
        config: HotReloadConfig,
    ) !Self {
        // Load initial shader
        const shader = try ShaderModule.initFromFile(ctx, config.shader_path);
        errdefer shader.deinit();

        // Create initial pipeline
        const pipeline = try ComputePipeline.init(
            ctx,
            shader,
            config.entry_point,
            descriptor_layout,
            push_constant_range,
        );
        errdefer pipeline.deinit();

        // Destroy shader module (pipeline has compiled it)
        var shader_mut = shader;
        shader_mut.deinit();

        // Set up file watcher
        var watcher = hot_reload.ShaderWatcher.init(allocator);
        errdefer watcher.deinit();

        watcher.poll_interval_ns = config.poll_interval_ns;

        // Watch the shader file
        try watcher.watch(config.shader_path, .{
            .use_content_hash = config.use_content_hash,
        });

        return Self{
            .ctx = ctx,
            .allocator = allocator,
            .pipeline = pipeline,
            .watcher = watcher,
            .config = config,
            .descriptor_layout = descriptor_layout,
            .push_constant_range = push_constant_range,
        };
    }

    /// Poll for shader file changes and reload if needed
    /// Returns the number of successful reloads (0 or 1)
    pub fn poll(self: *Self) !u32 {
        const changes = try self.watcher.poll();
        if (changes == 0) return 0;

        // File changed - attempt to reload
        return self.reload();
    }

    /// Force reload the shader (useful for testing or manual triggers)
    pub fn reload(self: *Self) u32 {
        const start_time = std.time.nanoTimestamp();

        // Try to load and compile new shader
        const new_shader = ShaderModule.initFromFile(self.ctx, self.config.shader_path) catch |e| {
            std.log.warn("[HotReload] Failed to load shader: {}", .{e});
            self.notifyReload(false);
            return 0;
        };

        const new_pipeline = ComputePipeline.init(
            self.ctx,
            new_shader,
            self.config.entry_point,
            self.descriptor_layout,
            self.push_constant_range,
        ) catch |e| {
            var shader_mut = new_shader;
            shader_mut.deinit();
            std.log.warn("[HotReload] Failed to create pipeline: {}", .{e});
            self.notifyReload(false);
            return 0;
        };

        // Clean up shader module
        var shader_mut = new_shader;
        shader_mut.deinit();

        // Wait for GPU to finish using old pipeline
        const wait_result = self.ctx.device_dispatch.vkDeviceWaitIdle.?(self.ctx.device);
        if (wait_result != .success) {
            std.log.warn("[HotReload] vkDeviceWaitIdle failed: {s} - proceeding with pipeline swap", .{@tagName(wait_result)});
        }

        // Swap pipelines
        var old_pipeline = self.pipeline;
        self.pipeline = new_pipeline;
        old_pipeline.deinit();

        // Update stats
        self.reload_count += 1;
        self.last_reload_time_ns = start_time;
        self.last_reload_success = true;

        std.log.info("[HotReload] Shader reloaded successfully (reload #{d})", .{self.reload_count});
        self.notifyReload(true);

        return 1;
    }

    fn notifyReload(self: *Self, success: bool) void {
        self.last_reload_success = success;
        if (self.config.on_reload) |callback| {
            callback(success, self.config.user_data);
        }
    }

    /// Bind the current pipeline to a command buffer
    pub fn bind(self: Self, cmd: types.CommandBuffer) void {
        self.pipeline.bind(cmd);
    }

    /// Bind with descriptor set
    pub fn bindWithDescriptors(self: Self, cmd: types.CommandBuffer, set: types.DescriptorSet) void {
        self.pipeline.bindWithDescriptors(cmd, set);
    }

    /// Push constants
    pub fn pushConstants(self: Self, cmd: types.CommandBuffer, comptime T: type, data: *const T) void {
        self.pipeline.pushConstants(cmd, T, data);
    }

    /// Dispatch
    pub fn dispatch(self: Self, cmd: types.CommandBuffer, x: u32, y: u32, z: u32) void {
        self.pipeline.dispatch(cmd, x, y, z);
    }

    /// Get the raw pipeline handle (for advanced use)
    pub fn getHandle(self: Self) types.Pipeline {
        return self.pipeline.pipeline;
    }

    /// Get the pipeline layout
    pub fn getLayout(self: Self) types.PipelineLayout {
        return self.pipeline.layout;
    }

    /// Pause file watching
    pub fn pauseWatching(self: *Self) void {
        self.watcher.paused = true;
    }

    /// Resume file watching
    pub fn resumeWatching(self: *Self) void {
        self.watcher.paused = false;
    }

    /// Check if watching is paused
    pub fn isWatchingPaused(self: Self) bool {
        return self.watcher.paused;
    }

    /// Get reload statistics
    pub fn getStats(self: Self) struct {
        reload_count: u32,
        last_reload_success: bool,
        files_watched: usize,
    } {
        return .{
            .reload_count = self.reload_count,
            .last_reload_success = self.last_reload_success,
            .files_watched = self.watcher.entries.items.len,
        };
    }

    /// Clean up
    pub fn deinit(self: *Self) void {
        self.watcher.deinit();
        self.pipeline.deinit();
        self.* = undefined;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "isValidSpirv" {
    // Valid SPIR-V: magic, version 1.0, generator=0, bound=1, schema=0
    const valid = [_]u32{ 0x07230203, 0x00010000, 0, 1, 0 };
    try std.testing.expect(isValidSpirv(&valid));

    const invalid = [_]u32{ 0x12345678, 0, 0, 0, 0 };
    try std.testing.expect(!isValidSpirv(&invalid));

    const too_short = [_]u32{0x07230203};
    try std.testing.expect(!isValidSpirv(&too_short));

    // Zero bound is invalid
    const zero_bound = [_]u32{ 0x07230203, 0x00010000, 0, 0, 0 };
    try std.testing.expect(!isValidSpirv(&zero_bound));
}

test "PushConstantRange defaults" {
    const range = PushConstantRange{ .size = 16 };
    try std.testing.expectEqual(@as(u32, 0), range.offset);
    try std.testing.expectEqual(@as(u32, 16), range.size);
}

test "DescriptorBinding defaults" {
    const binding = DescriptorBinding{
        .binding = 0,
        .descriptor_type = .storage_buffer,
    };
    try std.testing.expectEqual(@as(u32, 1), binding.count);
}
