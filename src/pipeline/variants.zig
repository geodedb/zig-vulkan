// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Shader Variants
//!
//! Specialization constant combinations with pipeline caching.
//! Allows efficient creation and reuse of shader variants with different
//! specialization constant values.
//!
//! Features:
//! - Type-safe specialization constant definitions
//! - Automatic pipeline variant caching
//! - Variant enumeration and iteration
//! - Lazy compilation support
//! - Variant usage statistics

const std = @import("std");
const types = @import("../vk/types.zig");
const Context = @import("../compute/context.zig").Context;

/// Specialization constant type
pub const SpecType = enum {
    bool_type,
    int32,
    uint32,
    float32,
};

/// Specialization constant definition
pub const SpecConstantDef = struct {
    id: u32,
    name: []const u8,
    spec_type: SpecType,
    default_value: SpecValue,
};

/// Specialization constant value
pub const SpecValue = union(SpecType) {
    bool_type: bool,
    int32: i32,
    uint32: u32,
    float32: f32,

    pub fn toBytes(self: SpecValue) [4]u8 {
        return switch (self) {
            .bool_type => |v| std.mem.toBytes(@as(u32, if (v) 1 else 0)),
            .int32 => |v| std.mem.toBytes(v),
            .uint32 => |v| std.mem.toBytes(v),
            .float32 => |v| std.mem.toBytes(v),
        };
    }

    pub fn getSize(self: SpecValue) usize {
        return switch (self) {
            .bool_type => @sizeOf(u32), // VK_TRUE/VK_FALSE are u32
            .int32 => @sizeOf(i32),
            .uint32 => @sizeOf(u32),
            .float32 => @sizeOf(f32),
        };
    }
};

/// Key for variant cache lookup
pub const VariantKey = struct {
    values: []const SpecValue,

    pub fn hash(self: VariantKey) u64 {
        var h = std.hash.Wyhash.init(0);
        for (self.values) |v| {
            h.update(&v.toBytes());
        }
        return h.final();
    }

    pub fn eql(self: VariantKey, other: VariantKey) bool {
        if (self.values.len != other.values.len) return false;
        for (self.values, other.values) |a, b| {
            if (!std.meta.eql(a.toBytes(), b.toBytes())) return false;
        }
        return true;
    }
};

/// Cached pipeline variant
pub const PipelineVariant = struct {
    pipeline: types.Pipeline,
    key_hash: u64,
    use_count: u64 = 0,
    last_used_frame: u64 = 0,
};

/// Shader variant manager for a single shader
pub const ShaderVariantManager = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    ctx: *Context,

    /// Shader module
    shader_module: types.ShaderModule,
    entry_point: []const u8,

    /// Pipeline layout
    layout: types.PipelineLayout,

    /// Specialization constant definitions
    constants: []const SpecConstantDef,

    /// Cached variants (hash -> variant)
    variants: std.AutoHashMapUnmanaged(u64, PipelineVariant),

    /// Owned spec constant values for default variant
    default_values: []SpecValue,

    /// Statistics
    total_compilations: u64 = 0,
    cache_hits: u64 = 0,
    current_frame: u64 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        ctx: *Context,
        shader_module: types.ShaderModule,
        entry_point: []const u8,
        layout: types.PipelineLayout,
        constants: []const SpecConstantDef,
    ) !Self {
        // Store default values
        const default_values = try allocator.alloc(SpecValue, constants.len);
        for (constants, 0..) |c, i| {
            default_values[i] = c.default_value;
        }

        const owned_entry = try allocator.dupe(u8, entry_point);

        return .{
            .allocator = allocator,
            .ctx = ctx,
            .shader_module = shader_module,
            .entry_point = owned_entry,
            .layout = layout,
            .constants = constants,
            .variants = .{},
            .default_values = default_values,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.variants.iterator();
        while (iter.next()) |entry| {
            self.ctx.device_api.destroyPipeline(
                self.ctx.device,
                entry.value_ptr.pipeline,
                null,
            );
        }
        self.variants.deinit(self.allocator);
        self.allocator.free(self.default_values);
        self.allocator.free(self.entry_point);
    }

    /// Get or create a variant with the given specialization values
    pub fn getVariant(self: *Self, values: []const SpecValue) !types.Pipeline {
        const key = VariantKey{ .values = values };
        const key_hash = key.hash();

        // Check cache
        if (self.variants.getPtr(key_hash)) |variant| {
            variant.use_count += 1;
            variant.last_used_frame = self.current_frame;
            self.cache_hits += 1;
            return variant.pipeline;
        }

        // Create new variant
        const pipe = try self.compilePipeline(values);

        try self.variants.put(self.allocator, key_hash, .{
            .pipeline = pipe,
            .key_hash = key_hash,
            .use_count = 1,
            .last_used_frame = self.current_frame,
        });

        self.total_compilations += 1;
        return pipe;
    }

    /// Get default variant
    pub fn getDefault(self: *Self) !types.Pipeline {
        return self.getVariant(self.default_values);
    }

    /// Pre-compile a set of variants
    pub fn precompile(self: *Self, variant_sets: []const []const SpecValue) !void {
        for (variant_sets) |values| {
            _ = try self.getVariant(values);
        }
    }

    /// Advance frame counter (for LRU tracking)
    pub fn advanceFrame(self: *Self) void {
        self.current_frame += 1;
    }

    /// Evict unused variants (LRU)
    pub fn evictUnused(self: *Self, max_age: u64) u32 {
        var evicted: u32 = 0;
        var to_remove = std.ArrayListUnmanaged(u64){};
        defer to_remove.deinit(self.allocator);

        var iter = self.variants.iterator();
        while (iter.next()) |entry| {
            const age = self.current_frame - entry.value_ptr.last_used_frame;
            if (age > max_age) {
                to_remove.append(self.allocator, entry.key_ptr.*) catch continue;
            }
        }

        for (to_remove.items) |key| {
            if (self.variants.fetchRemove(key)) |entry| {
                self.ctx.device_api.destroyPipeline(
                    self.ctx.device,
                    entry.value.pipeline,
                    null,
                );
                evicted += 1;
            }
        }

        return evicted;
    }

    fn compilePipeline(self: *Self, values: []const SpecValue) !types.Pipeline {
        // Build specialization map entries
        const entries = try self.allocator.alloc(types.SpecializationMapEntry, self.constants.len);
        defer self.allocator.free(entries);

        // Build data buffer
        var data_size: u32 = 0;
        for (values) |v| {
            data_size += @intCast(v.getSize());
        }

        const data = try self.allocator.alloc(u8, data_size);
        defer self.allocator.free(data);

        var offset: u32 = 0;
        for (self.constants, 0..) |c, i| {
            const value = if (i < values.len) values[i] else c.default_value;
            const size: u32 = @intCast(value.getSize());

            entries[i] = .{
                .constant_id = c.id,
                .offset = offset,
                .size = size,
            };

            const bytes = value.toBytes();
            @memcpy(data[offset..][0..size], bytes[0..size]);
            offset += size;
        }

        const spec_info = types.SpecializationInfo{
            .map_entry_count = @intCast(entries.len),
            .p_map_entries = entries.ptr,
            .data_size = data_size,
            .p_data = data.ptr,
        };

        const stage_info = types.PipelineShaderStageCreateInfo{
            .s_type = .pipeline_shader_stage_create_info,
            .p_next = null,
            .flags = 0,
            .stage = .{ .compute = true },
            .module = self.shader_module,
            .p_name = self.entry_point.ptr,
            .p_specialization_info = &spec_info,
        };

        const create_info = types.ComputePipelineCreateInfo{
            .s_type = .compute_pipeline_create_info,
            .p_next = null,
            .flags = .{},
            .stage = stage_info,
            .layout = self.layout,
            .base_pipeline_handle = null,
            .base_pipeline_index = -1,
        };

        var pipe: types.Pipeline = undefined;
        const result = self.ctx.device_api.createComputePipelines(
            self.ctx.device,
            .null_handle,
            1,
            @ptrCast(&create_info),
            null,
            @ptrCast(&pipe),
        );

        if (result != .success) {
            return error.PipelineCreationFailed;
        }

        return pipe;
    }

    /// Get statistics
    pub fn getStats(self: Self) struct {
        variant_count: usize,
        total_compilations: u64,
        cache_hits: u64,
        hit_rate: f32,
    } {
        const total = self.total_compilations + self.cache_hits;
        return .{
            .variant_count = self.variants.count(),
            .total_compilations = self.total_compilations,
            .cache_hits = self.cache_hits,
            .hit_rate = if (total > 0) @as(f32, @floatFromInt(self.cache_hits)) / @as(f32, @floatFromInt(total)) else 0.0,
        };
    }
};

/// Builder for creating variant managers
pub const VariantBuilder = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    constants: std.ArrayListUnmanaged(SpecConstantDef),

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .constants = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.constants.deinit(self.allocator);
    }

    /// Add a boolean constant
    pub fn addBool(self: *Self, id: u32, name: []const u8, default: bool) !*Self {
        try self.constants.append(self.allocator, .{
            .id = id,
            .name = name,
            .spec_type = .bool_type,
            .default_value = .{ .bool_type = default },
        });
        return self;
    }

    /// Add an integer constant
    pub fn addInt(self: *Self, id: u32, name: []const u8, default: i32) !*Self {
        try self.constants.append(self.allocator, .{
            .id = id,
            .name = name,
            .spec_type = .int32,
            .default_value = .{ .int32 = default },
        });
        return self;
    }

    /// Add an unsigned integer constant
    pub fn addUint(self: *Self, id: u32, name: []const u8, default: u32) !*Self {
        try self.constants.append(self.allocator, .{
            .id = id,
            .name = name,
            .spec_type = .uint32,
            .default_value = .{ .uint32 = default },
        });
        return self;
    }

    /// Add a float constant
    pub fn addFloat(self: *Self, id: u32, name: []const u8, default: f32) !*Self {
        try self.constants.append(self.allocator, .{
            .id = id,
            .name = name,
            .spec_type = .float32,
            .default_value = .{ .float32 = default },
        });
        return self;
    }

    /// Build the variant manager
    pub fn build(
        self: *Self,
        ctx: *Context,
        shader_module: types.ShaderModule,
        entry_point: []const u8,
        layout: types.PipelineLayout,
    ) !ShaderVariantManager {
        return ShaderVariantManager.init(
            self.allocator,
            ctx,
            shader_module,
            entry_point,
            layout,
            self.constants.items,
        );
    }
};

/// Multi-shader variant system
pub const VariantSystem = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    ctx: *Context,
    managers: std.StringHashMapUnmanaged(*ShaderVariantManager),

    pub fn init(allocator: std.mem.Allocator, ctx: *Context) Self {
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .managers = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.managers.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.managers.deinit(self.allocator);
    }

    /// Register a shader with variants
    pub fn registerShader(
        self: *Self,
        name: []const u8,
        shader_module: types.ShaderModule,
        entry_point: []const u8,
        layout: types.PipelineLayout,
        constants: []const SpecConstantDef,
    ) !void {
        const manager = try self.allocator.create(ShaderVariantManager);
        manager.* = try ShaderVariantManager.init(
            self.allocator,
            self.ctx,
            shader_module,
            entry_point,
            layout,
            constants,
        );

        const owned_name = try self.allocator.dupe(u8, name);
        try self.managers.put(self.allocator, owned_name, manager);
    }

    /// Get variant for a registered shader
    pub fn getVariant(self: *Self, shader_name: []const u8, values: []const SpecValue) !types.Pipeline {
        const manager = self.managers.get(shader_name) orelse return error.ShaderNotFound;
        return manager.getVariant(values);
    }

    /// Get default variant
    pub fn getDefault(self: *Self, shader_name: []const u8) !types.Pipeline {
        const manager = self.managers.get(shader_name) orelse return error.ShaderNotFound;
        return manager.getDefault();
    }

    /// Advance frame for all managers
    pub fn advanceFrame(self: *Self) void {
        var iter = self.managers.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.advanceFrame();
        }
    }

    /// Evict unused variants from all managers
    pub fn evictUnused(self: *Self, max_age: u64) u32 {
        var total: u32 = 0;
        var iter = self.managers.iterator();
        while (iter.next()) |entry| {
            total += entry.value_ptr.*.evictUnused(max_age);
        }
        return total;
    }

    /// Get total statistics
    pub fn getStats(self: Self) struct {
        shader_count: usize,
        total_variants: usize,
        total_compilations: u64,
        total_cache_hits: u64,
    } {
        var total_variants: usize = 0;
        var total_compilations: u64 = 0;
        var total_cache_hits: u64 = 0;

        var iter = self.managers.iterator();
        while (iter.next()) |entry| {
            const stats = entry.value_ptr.*.getStats();
            total_variants += stats.variant_count;
            total_compilations += stats.total_compilations;
            total_cache_hits += stats.cache_hits;
        }

        return .{
            .shader_count = self.managers.count(),
            .total_variants = total_variants,
            .total_compilations = total_compilations,
            .total_cache_hits = total_cache_hits,
        };
    }
};

// Tests
test "SpecValue toBytes" {
    const bool_val = SpecValue{ .bool_type = true };
    const bytes = bool_val.toBytes();
    try std.testing.expectEqual(@as(u32, 1), std.mem.bytesToValue(u32, &bytes));

    const int_val = SpecValue{ .int32 = -42 };
    const int_bytes = int_val.toBytes();
    try std.testing.expectEqual(@as(i32, -42), std.mem.bytesToValue(i32, &int_bytes));
}

test "VariantKey hash" {
    const values1 = [_]SpecValue{
        .{ .uint32 = 64 },
        .{ .bool_type = true },
    };
    const values2 = [_]SpecValue{
        .{ .uint32 = 64 },
        .{ .bool_type = true },
    };
    const values3 = [_]SpecValue{
        .{ .uint32 = 128 },
        .{ .bool_type = true },
    };

    const key1 = VariantKey{ .values = &values1 };
    const key2 = VariantKey{ .values = &values2 };
    const key3 = VariantKey{ .values = &values3 };

    try std.testing.expectEqual(key1.hash(), key2.hash());
    try std.testing.expect(key1.hash() != key3.hash());
}

test "VariantBuilder" {
    var builder = VariantBuilder.init(std.testing.allocator);
    defer builder.deinit();

    _ = try builder.addBool(0, "USE_FAST_PATH", true);
    _ = try builder.addUint(1, "WORKGROUP_SIZE", 64);
    _ = try builder.addFloat(2, "THRESHOLD", 0.5);

    try std.testing.expectEqual(@as(usize, 3), builder.constants.items.len);
}
