// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Descriptor Set Templates
//!
//! Provides cached descriptor update templates for faster descriptor set updates.
//! Instead of rebuilding VkWriteDescriptorSet arrays each frame, templates allow
//! pre-computing the update structure and only changing the actual resource handles.
//!
//! Features:
//! - Pre-compiled update templates
//! - Batch updates with minimal overhead
//! - Support for push descriptors
//! - Template caching by layout

const std = @import("std");
const types = @import("../vk/types.zig");
const Context = @import("../compute/context.zig").Context;

/// Entry in a descriptor update template
pub const TemplateEntry = struct {
    /// Binding index in the descriptor set
    binding: u32,
    /// Array element within the binding
    array_element: u32 = 0,
    /// Number of descriptors to update
    descriptor_count: u32 = 1,
    /// Type of descriptor
    descriptor_type: types.DescriptorType,
    /// Offset in the data blob where this entry's data starts
    offset: usize,
    /// Stride between array elements (0 for single descriptors)
    stride: usize = 0,
};

/// Data for buffer descriptor updates
pub const BufferData = struct {
    buffer: ?types.Buffer = null,
    offset: u64 = 0,
    range: u64 = 0, // 0 means whole size
};

/// Data for image descriptor updates
pub const ImageData = struct {
    sampler: ?types.Sampler = null,
    image_view: ?types.ImageView = null,
    image_layout: types.ImageLayout = .general,
};

/// Data for texel buffer views
pub const TexelBufferData = struct {
    buffer_view: ?types.BufferView = null,
};

/// Union of all descriptor data types
pub const DescriptorData = union {
    buffer: BufferData,
    image: ImageData,
    texel_buffer: TexelBufferData,
};

/// Pre-compiled descriptor update template
pub const UpdateTemplate = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    ctx: *Context,

    /// Vulkan descriptor update template handle
    handle: ?types.DescriptorUpdateTemplate = null,

    /// Template entries for reference
    entries: []TemplateEntry,

    /// Associated descriptor set layout
    layout: types.DescriptorSetLayout,

    /// Size of the data blob needed for updates
    data_size: usize,

    /// Name for debugging
    name: []const u8,

    pub fn init(
        allocator: std.mem.Allocator,
        ctx: *Context,
        layout: types.DescriptorSetLayout,
        entries: []const TemplateEntry,
        name: []const u8,
    ) !Self {
        // Calculate total data size and validate entries
        var data_size: usize = 0;
        for (entries) |entry| {
            const entry_size = switch (entry.descriptor_type) {
                .uniform_buffer,
                .storage_buffer,
                .uniform_buffer_dynamic,
                .storage_buffer_dynamic,
                => @sizeOf(BufferData),

                .sampler,
                .combined_image_sampler,
                .sampled_image,
                .storage_image,
                .input_attachment,
                => @sizeOf(ImageData),

                .uniform_texel_buffer,
                .storage_texel_buffer,
                => @sizeOf(TexelBufferData),

                else => return error.UnsupportedDescriptorType,
            };

            const end_offset = entry.offset + entry_size * entry.descriptor_count;
            if (end_offset > data_size) {
                data_size = end_offset;
            }
        }

        // Copy entries
        const owned_entries = try allocator.alloc(TemplateEntry, entries.len);
        @memcpy(owned_entries, entries);

        // Copy name
        const owned_name = try allocator.dupe(u8, name);

        // Build Vulkan template entries
        const vk_entries = try allocator.alloc(types.DescriptorUpdateTemplateEntry, entries.len);
        defer allocator.free(vk_entries);

        for (entries, 0..) |entry, i| {
            vk_entries[i] = .{
                .dst_binding = entry.binding,
                .dst_array_element = entry.array_element,
                .descriptor_count = entry.descriptor_count,
                .descriptor_type = entry.descriptor_type,
                .offset = entry.offset,
                .stride = if (entry.stride > 0) entry.stride else getDefaultStride(entry.descriptor_type),
            };
        }

        // Create the template
        const create_info = types.DescriptorUpdateTemplateCreateInfo{
            .s_type = .descriptor_update_template_create_info,
            .p_next = null,
            .flags = 0,
            .descriptor_update_entry_count = @intCast(vk_entries.len),
            .p_descriptor_update_entries = vk_entries.ptr,
            .template_type = .descriptor_set,
            .descriptor_set_layout = layout,
            .pipeline_bind_point = .compute, // Not used for descriptor_set type
            .pipeline_layout = null,
            .set = 0,
        };

        var handle: ?types.DescriptorUpdateTemplate = null;
        const result = ctx.device_api.createDescriptorUpdateTemplate(
            ctx.device,
            &create_info,
            null,
            @ptrCast(&handle),
        );

        if (result != .success) {
            allocator.free(owned_entries);
            allocator.free(owned_name);
            return error.TemplateCreationFailed;
        }

        return .{
            .allocator = allocator,
            .ctx = ctx,
            .handle = handle,
            .entries = owned_entries,
            .layout = layout,
            .data_size = data_size,
            .name = owned_name,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.handle) |h| {
            self.ctx.device_api.destroyDescriptorUpdateTemplate(
                self.ctx.device,
                h,
                null,
            );
        }
        self.allocator.free(self.entries);
        self.allocator.free(self.name);
        self.* = undefined;
    }

    /// Update a descriptor set using this template
    pub fn update(self: *Self, descriptor_set: types.DescriptorSet, data: []const u8) void {
        // Runtime bounds check (debug.assert is stripped in Release)
        if (data.len < self.data_size) {
            std.log.err("DescriptorTemplate.update: data buffer too small ({d} < {d} bytes required)", .{
                data.len,
                self.data_size,
            });
            return;
        }

        self.ctx.device_api.updateDescriptorSetWithTemplate(
            self.ctx.device,
            descriptor_set,
            self.handle,
            @ptrCast(data.ptr),
        );
    }

    fn getDefaultStride(desc_type: types.DescriptorType) usize {
        return switch (desc_type) {
            .uniform_buffer,
            .storage_buffer,
            .uniform_buffer_dynamic,
            .storage_buffer_dynamic,
            => @sizeOf(BufferData),

            .sampler,
            .combined_image_sampler,
            .sampled_image,
            .storage_image,
            .input_attachment,
            => @sizeOf(ImageData),

            .uniform_texel_buffer,
            .storage_texel_buffer,
            => @sizeOf(TexelBufferData),

            else => @sizeOf(BufferData), // Fallback
        };
    }
};

/// Builder for constructing update templates
pub const TemplateBuilder = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    entries: std.ArrayListUnmanaged(TemplateEntry),
    current_offset: usize = 0,

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .entries = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.entries.deinit(self.allocator);
    }

    /// Add a buffer binding
    pub fn addBuffer(
        self: *Self,
        binding: u32,
        desc_type: types.DescriptorType,
    ) !*Self {
        try self.entries.append(self.allocator, .{
            .binding = binding,
            .descriptor_type = desc_type,
            .offset = self.current_offset,
            .descriptor_count = 1,
        });
        self.current_offset += @sizeOf(BufferData);
        return self;
    }

    /// Add a buffer array binding
    pub fn addBufferArray(
        self: *Self,
        binding: u32,
        desc_type: types.DescriptorType,
        count: u32,
    ) !*Self {
        try self.entries.append(self.allocator, .{
            .binding = binding,
            .descriptor_type = desc_type,
            .offset = self.current_offset,
            .descriptor_count = count,
            .stride = @sizeOf(BufferData),
        });
        self.current_offset += @sizeOf(BufferData) * count;
        return self;
    }

    /// Add an image/sampler binding
    pub fn addImage(
        self: *Self,
        binding: u32,
        desc_type: types.DescriptorType,
    ) !*Self {
        try self.entries.append(self.allocator, .{
            .binding = binding,
            .descriptor_type = desc_type,
            .offset = self.current_offset,
            .descriptor_count = 1,
        });
        self.current_offset += @sizeOf(ImageData);
        return self;
    }

    /// Add an image array binding
    pub fn addImageArray(
        self: *Self,
        binding: u32,
        desc_type: types.DescriptorType,
        count: u32,
    ) !*Self {
        try self.entries.append(self.allocator, .{
            .binding = binding,
            .descriptor_type = desc_type,
            .offset = self.current_offset,
            .descriptor_count = count,
            .stride = @sizeOf(ImageData),
        });
        self.current_offset += @sizeOf(ImageData) * count;
        return self;
    }

    /// Build the template
    pub fn build(
        self: *Self,
        ctx: *Context,
        layout: types.DescriptorSetLayout,
        name: []const u8,
    ) !UpdateTemplate {
        return UpdateTemplate.init(
            self.allocator,
            ctx,
            layout,
            self.entries.items,
            name,
        );
    }

    /// Get the required data size for updates
    pub fn getDataSize(self: Self) usize {
        return self.current_offset;
    }
};

/// Helper for building update data blobs
/// All write methods include bounds checking to prevent buffer overflows.
pub const UpdateDataBuilder = struct {
    const Self = @This();

    data: []u8,
    offset: usize = 0,

    pub fn init(buffer: []u8) Self {
        return .{ .data = buffer };
    }

    /// Write buffer data at current offset
    /// Panics if write would exceed buffer capacity
    pub fn writeBuffer(self: *Self, buffer: ?types.Buffer, offset: u64, range: u64) *Self {
        const end = self.offset + @sizeOf(BufferData);
        if (end > self.data.len) {
            @panic("UpdateDataBuilder.writeBuffer: buffer overflow");
        }
        const buf_data = BufferData{
            .buffer = buffer,
            .offset = offset,
            .range = range,
        };
        @memcpy(
            self.data[self.offset..][0..@sizeOf(BufferData)],
            std.mem.asBytes(&buf_data),
        );
        self.offset = end;
        return self;
    }

    /// Write image data at current offset
    /// Panics if write would exceed buffer capacity
    pub fn writeImage(
        self: *Self,
        sampler: ?types.Sampler,
        image_view: ?types.ImageView,
        layout: types.ImageLayout,
    ) *Self {
        const end = self.offset + @sizeOf(ImageData);
        if (end > self.data.len) {
            @panic("UpdateDataBuilder.writeImage: buffer overflow");
        }
        const img_data = ImageData{
            .sampler = sampler,
            .image_view = image_view,
            .image_layout = layout,
        };
        @memcpy(
            self.data[self.offset..][0..@sizeOf(ImageData)],
            std.mem.asBytes(&img_data),
        );
        self.offset = end;
        return self;
    }

    /// Write buffer data at specific offset
    /// Panics if write would exceed buffer capacity
    pub fn writeBufferAt(self: *Self, at: usize, buffer: ?types.Buffer, offset: u64, range: u64) *Self {
        const end = at + @sizeOf(BufferData);
        if (end > self.data.len) {
            @panic("UpdateDataBuilder.writeBufferAt: buffer overflow");
        }
        const buf_data = BufferData{
            .buffer = buffer,
            .offset = offset,
            .range = range,
        };
        @memcpy(
            self.data[at..][0..@sizeOf(BufferData)],
            std.mem.asBytes(&buf_data),
        );
        return self;
    }

    /// Write image data at specific offset
    /// Panics if write would exceed buffer capacity
    pub fn writeImageAt(
        self: *Self,
        at: usize,
        sampler: ?types.Sampler,
        image_view: ?types.ImageView,
        layout: types.ImageLayout,
    ) *Self {
        const end = at + @sizeOf(ImageData);
        if (end > self.data.len) {
            @panic("UpdateDataBuilder.writeImageAt: buffer overflow");
        }
        const img_data = ImageData{
            .sampler = sampler,
            .image_view = image_view,
            .image_layout = layout,
        };
        @memcpy(
            self.data[at..][0..@sizeOf(ImageData)],
            std.mem.asBytes(&img_data),
        );
        return self;
    }

    /// Get remaining capacity in bytes
    pub fn remaining(self: Self) usize {
        return self.data.len -| self.offset;
    }

    /// Check if there's enough space for a buffer entry
    pub fn canWriteBuffer(self: Self) bool {
        return self.remaining() >= @sizeOf(BufferData);
    }

    /// Check if there's enough space for an image entry
    pub fn canWriteImage(self: Self) bool {
        return self.remaining() >= @sizeOf(ImageData);
    }

    /// Reset for reuse
    pub fn reset(self: *Self) *Self {
        self.offset = 0;
        return self;
    }
};

/// Cache for descriptor update templates
///
/// Ownership model: Each UpdateTemplate owns its name string. The cache uses
/// template.name as the hash map key, creating a lifecycle dependency where
/// the key is valid as long as the template exists. When removing templates,
/// always use remove() which handles cleanup correctly.
pub const TemplateCache = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    ctx: *Context,
    /// Maps template names to template pointers. Keys point to template-owned strings.
    templates: std.StringHashMapUnmanaged(*UpdateTemplate),

    pub fn init(allocator: std.mem.Allocator, ctx: *Context) Self {
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .templates = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.templates.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.templates.deinit(self.allocator);
    }

    /// Get or create a template
    pub fn getOrCreate(
        self: *Self,
        name: []const u8,
        layout: types.DescriptorSetLayout,
        entries: []const TemplateEntry,
    ) !*UpdateTemplate {
        if (self.templates.get(name)) |template| {
            return template;
        }

        const template = try self.allocator.create(UpdateTemplate);
        template.* = try UpdateTemplate.init(self.allocator, self.ctx, layout, entries, name);

        try self.templates.put(self.allocator, template.name, template);
        return template;
    }

    /// Get an existing template
    pub fn get(self: Self, name: []const u8) ?*UpdateTemplate {
        return self.templates.get(name);
    }

    /// Remove a template from cache
    pub fn remove(self: *Self, name: []const u8) void {
        if (self.templates.fetchRemove(name)) |entry| {
            entry.value.deinit();
            self.allocator.destroy(entry.value);
        }
    }

    /// Get number of cached templates
    pub fn count(self: Self) usize {
        return self.templates.count();
    }
};

/// Push descriptor update helper
pub const PushDescriptorUpdater = struct {
    const Self = @This();

    ctx: *Context,
    writes: std.ArrayListUnmanaged(types.WriteDescriptorSet),
    buffer_infos: std.ArrayListUnmanaged(types.DescriptorBufferInfo),
    image_infos: std.ArrayListUnmanaged(types.DescriptorImageInfo),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, ctx: *Context) Self {
        return .{
            .ctx = ctx,
            .allocator = allocator,
            .writes = .{},
            .buffer_infos = .{},
            .image_infos = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.writes.deinit(self.allocator);
        self.buffer_infos.deinit(self.allocator);
        self.image_infos.deinit(self.allocator);
    }

    /// Add a buffer to push
    pub fn pushBuffer(
        self: *Self,
        binding: u32,
        desc_type: types.DescriptorType,
        buffer: types.Buffer,
        offset: u64,
        range: u64,
    ) !void {
        const info_idx = self.buffer_infos.items.len;
        try self.buffer_infos.append(self.allocator, .{
            .buffer = buffer,
            .offset = offset,
            .range = range,
        });

        try self.writes.append(self.allocator, .{
            .s_type = .write_descriptor_set,
            .p_next = null,
            .dst_set = null, // Not used for push descriptors
            .dst_binding = binding,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = desc_type,
            .p_image_info = null,
            .p_buffer_info = @ptrCast(&self.buffer_infos.items[info_idx]),
            .p_texel_buffer_view = null,
        });
    }

    /// Add an image to push
    pub fn pushImage(
        self: *Self,
        binding: u32,
        desc_type: types.DescriptorType,
        sampler: types.Sampler,
        image_view: types.ImageView,
        layout: types.ImageLayout,
    ) !void {
        const info_idx = self.image_infos.items.len;
        try self.image_infos.append(self.allocator, .{
            .sampler = sampler,
            .image_view = image_view,
            .image_layout = layout,
        });

        try self.writes.append(self.allocator, .{
            .s_type = .write_descriptor_set,
            .p_next = null,
            .dst_set = null, // Not used for push descriptors
            .dst_binding = binding,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = desc_type,
            .p_image_info = @ptrCast(&self.image_infos.items[info_idx]),
            .p_buffer_info = null,
            .p_texel_buffer_view = null,
        });
    }

    /// Execute push descriptor update
    pub fn push(
        self: *Self,
        cmd: types.CommandBuffer,
        layout: types.PipelineLayout,
        set: u32,
    ) void {
        if (self.writes.items.len == 0) return;

        self.ctx.device_api.cmdPushDescriptorSetKHR(
            cmd,
            .compute,
            layout,
            set,
            @intCast(self.writes.items.len),
            self.writes.items.ptr,
        );
    }

    /// Reset for next frame
    pub fn reset(self: *Self) void {
        self.writes.clearRetainingCapacity();
        self.buffer_infos.clearRetainingCapacity();
        self.image_infos.clearRetainingCapacity();
    }
};

// Tests
test "TemplateBuilder basic" {
    var builder = TemplateBuilder.init(std.testing.allocator);
    defer builder.deinit();

    _ = try builder.addBuffer(0, .uniform_buffer);
    _ = try builder.addBuffer(1, .storage_buffer);
    _ = try builder.addImage(2, .combined_image_sampler);

    try std.testing.expectEqual(@as(usize, 3), builder.entries.items.len);
    try std.testing.expectEqual(
        @sizeOf(BufferData) * 2 + @sizeOf(ImageData),
        builder.getDataSize(),
    );
}

test "UpdateDataBuilder" {
    var buffer: [256]u8 = undefined;
    var data_builder = UpdateDataBuilder.init(&buffer);

    _ = data_builder.writeBuffer(null, 0, 1024);
    try std.testing.expectEqual(@sizeOf(BufferData), data_builder.offset);

    _ = data_builder.writeImage(null, null, .general);
    try std.testing.expectEqual(@sizeOf(BufferData) + @sizeOf(ImageData), data_builder.offset);

    _ = data_builder.reset();
    try std.testing.expectEqual(@as(usize, 0), data_builder.offset);
}
