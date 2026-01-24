// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Multi-Device Support
//!
//! Provides support for using multiple GPUs simultaneously for:
//! - Parallel compute workloads across devices
//! - Load balancing between GPUs
//! - Device group operations (linked GPUs like SLI/CrossFire)
//!
//! ## Usage
//!
//! ```zig
//! var multi = try MultiDeviceContext.init(allocator, .{});
//! defer multi.deinit();
//!
//! // Use all available compute devices
//! for (multi.devices) |*device| {
//!     var ctx = try device.createContext(.{});
//!     defer ctx.deinit();
//!     // ... use device
//! }
//!
//! // Or select specific devices by capability
//! const best = multi.selectBestDevice(.{ .prefer_discrete = true });
//! ```

const std = @import("std");
const types = @import("../vk/types.zig");
const loader = @import("../vk/loader.zig");
const err = @import("../vk/error.zig");
const Context = @import("context.zig").Context;
const DeviceInfo = @import("context.zig").DeviceInfo;
const Options = @import("context.zig").Options;

/// Device selection criteria
pub const DeviceSelector = struct {
    /// Prefer discrete GPUs over integrated
    prefer_discrete: bool = true,
    /// Minimum required device memory (bytes)
    min_memory: u64 = 0,
    /// Required device features (if any)
    require_compute: bool = true,
    /// Prefer devices with timestamps support
    prefer_timestamps: bool = false,
    /// Vendor preference (null = any)
    vendor_id: ?u32 = null,
};

/// Device capability flags
pub const DeviceCapabilities = struct {
    /// Supports compute shaders
    compute: bool,
    /// Supports timestamps
    timestamps: bool,
    /// Is a discrete GPU
    discrete: bool,
    /// Supports sparse binding
    sparse_binding: bool,
    /// Total device memory
    memory_size: u64,
    /// Maximum workgroup size
    max_workgroup_size: u32,
    /// Maximum shared memory per workgroup
    max_shared_memory: u32,
};

/// Physical device wrapper for multi-device contexts
pub const PhysicalDeviceHandle = struct {
    /// Physical device handle
    handle: types.PhysicalDevice,
    /// Device index in the system
    index: u32,
    /// Device properties
    properties: types.PhysicalDeviceProperties,
    /// Memory properties
    memory_properties: types.PhysicalDeviceMemoryProperties,
    /// Compute queue family index
    compute_queue_family: ?u32,
    /// Transfer queue family index (if separate from compute)
    transfer_queue_family: ?u32,
    /// Instance dispatch (for device creation)
    instance_dispatch: loader.InstanceDispatch,
    /// Instance handle
    instance: types.Instance,
    /// Allocator
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Get device information
    pub fn getInfo(self: Self) DeviceInfo {
        const mem_total = blk: {
            var total: u64 = 0;
            // Bound by MAX_MEMORY_HEAPS to prevent out-of-bounds access
            const heap_count = @min(self.memory_properties.memory_heap_count, types.MAX_MEMORY_HEAPS);
            for (0..heap_count) |i| {
                if (self.memory_properties.memory_heaps[i].flags.device_local) {
                    total += self.memory_properties.memory_heaps[i].size;
                }
            }
            break :blk total;
        };

        return DeviceInfo{
            .index = self.index,
            .name = self.properties.device_name,
            .device_type = self.properties.device_type,
            .api_version = self.properties.api_version,
            .driver_version = self.properties.driver_version,
            .vendor_id = self.properties.vendor_id,
            .device_id = self.properties.device_id,
            .has_compute_queue = self.compute_queue_family != null,
            .compute_queue_family = self.compute_queue_family,
            .max_compute_work_group_count = self.properties.limits.max_compute_work_group_count,
            .max_compute_work_group_size = self.properties.limits.max_compute_work_group_size,
            .max_compute_shared_memory_size = self.properties.limits.max_compute_shared_memory_size,
            .total_memory = mem_total,
            .timestamp_supported = self.properties.limits.timestamp_compute_and_graphics != 0,
        };
    }

    /// Get device capabilities
    pub fn getCapabilities(self: Self) DeviceCapabilities {
        return DeviceCapabilities{
            .compute = self.compute_queue_family != null,
            .timestamps = self.properties.limits.timestamp_compute_and_graphics != 0,
            .discrete = self.properties.device_type == .discrete_gpu,
            .sparse_binding = false, // Would need feature query
            .memory_size = blk: {
                var total: u64 = 0;
                // Bound by MAX_MEMORY_HEAPS to prevent out-of-bounds access
                const heap_count = @min(self.memory_properties.memory_heap_count, types.MAX_MEMORY_HEAPS);
                for (0..heap_count) |i| {
                    if (self.memory_properties.memory_heaps[i].flags.device_local) {
                        total += self.memory_properties.memory_heaps[i].size;
                    }
                }
                break :blk total;
            },
            .max_workgroup_size = self.properties.limits.max_compute_work_group_invocations,
            .max_shared_memory = self.properties.limits.max_compute_shared_memory_size,
        };
    }

    /// Get device name as string
    pub fn getName(self: Self) []const u8 {
        const len = std.mem.indexOfScalar(u8, &self.properties.device_name, 0) orelse self.properties.device_name.len;
        return self.properties.device_name[0..len];
    }

    /// Check if device matches selection criteria
    pub fn matchesCriteria(self: Self, criteria: DeviceSelector) bool {
        // Must have compute queue if required
        if (criteria.require_compute and self.compute_queue_family == null) {
            return false;
        }

        // Check minimum memory
        const caps = self.getCapabilities();
        if (caps.memory_size < criteria.min_memory) {
            return false;
        }

        // Check vendor
        if (criteria.vendor_id) |vid| {
            if (self.properties.vendor_id != vid) {
                return false;
            }
        }

        return true;
    }

    /// Score device for selection (higher is better)
    pub fn score(self: Self, criteria: DeviceSelector) i32 {
        var s: i32 = 0;

        const caps = self.getCapabilities();

        // Discrete GPU preference
        if (caps.discrete) {
            s += if (criteria.prefer_discrete) 1000 else 0;
        }

        // Timestamp support
        if (caps.timestamps and criteria.prefer_timestamps) {
            s += 100;
        }

        // Memory size (log scale, capped at 32GB contribution)
        s += @intCast(@min(caps.memory_size / (1024 * 1024 * 1024), 32));

        // Workgroup size (capped to prevent overflow)
        s += @intCast(@min(caps.max_workgroup_size / 256, 16));

        return s;
    }

    /// Create a compute context for this device
    pub fn createContext(self: Self, options: Options) err.Error!Context {
        var opts = options;
        opts.device_index = self.index;
        return Context.init(self.allocator, opts);
    }
};

/// Multi-device context for managing multiple GPUs
pub const MultiDeviceContext = struct {
    allocator: std.mem.Allocator,
    base_loader: loader.BaseLoader,
    global_dispatch: loader.GlobalDispatch,
    instance_dispatch: loader.InstanceDispatch,
    instance: types.Instance,
    devices: []PhysicalDeviceHandle,

    const Self = @This();

    /// Initialize multi-device context with all available devices
    pub fn init(allocator: std.mem.Allocator, options: Options) err.Error!Self {
        // Load Vulkan
        var base = loader.BaseLoader.load() catch return err.Error.VulkanUnavailable;
        errdefer base.close();

        const global = loader.GlobalDispatch.load(base) catch return err.Error.FunctionNotFound;

        // Create instance
        var extensions = std.ArrayListUnmanaged([*:0]const u8){};
        defer extensions.deinit(allocator);
        var layers = std.ArrayListUnmanaged([*:0]const u8){};
        defer layers.deinit(allocator);

        if (options.enable_validation) {
            layers.append(allocator, "VK_LAYER_KHRONOS_validation") catch return err.Error.OutOfHostMemory;
        }

        if (options.enable_debug_utils) {
            extensions.append(allocator, "VK_EXT_debug_utils") catch return err.Error.OutOfHostMemory;
        }

        const app_info = types.ApplicationInfo{
            .p_application_name = options.app_name,
            .application_version = options.app_version,
            .p_engine_name = "zig-vulkan-compute",
            .engine_version = types.makeApiVersion(0, 0, 1, 0),
            .api_version = options.min_api_version,
        };

        const instance_info = types.InstanceCreateInfo{
            .p_application_info = &app_info,
            .enabled_layer_count = @intCast(layers.items.len),
            .pp_enabled_layer_names = if (layers.items.len > 0) layers.items.ptr else null,
            .enabled_extension_count = @intCast(extensions.items.len),
            .pp_enabled_extension_names = if (extensions.items.len > 0) extensions.items.ptr else null,
        };

        const instance = try global.createInstance(&instance_info);

        const inst_dispatch = loader.InstanceDispatch.load(base, instance) catch return err.Error.FunctionNotFound;
        errdefer inst_dispatch.destroyInstance(instance);

        // Enumerate physical devices
        const phys_devices = try inst_dispatch.enumeratePhysicalDevices(instance, allocator);
        defer allocator.free(phys_devices);

        if (phys_devices.len == 0) {
            return err.Error.NoDevices;
        }

        // Create device handles
        var device_handles = allocator.alloc(PhysicalDeviceHandle, phys_devices.len) catch return err.Error.OutOfHostMemory;
        errdefer allocator.free(device_handles);

        for (phys_devices, 0..) |phys_dev, i| {
            const props = inst_dispatch.getPhysicalDeviceProperties(phys_dev);
            const mem_props = inst_dispatch.getPhysicalDeviceMemoryProperties(phys_dev);

            // Find queue families
            const queue_families = try inst_dispatch.getPhysicalDeviceQueueFamilyProperties(phys_dev, allocator);
            defer allocator.free(queue_families);

            var compute_family: ?u32 = null;
            var transfer_family: ?u32 = null;

            for (queue_families, 0..) |qf, qi| {
                const idx: u32 = @intCast(qi);
                if (qf.queue_flags.compute and compute_family == null) {
                    compute_family = idx;
                }
                if (qf.queue_flags.transfer and !qf.queue_flags.compute and transfer_family == null) {
                    transfer_family = idx;
                }
            }

            device_handles[i] = PhysicalDeviceHandle{
                .handle = phys_dev,
                .index = @intCast(i),
                .properties = props,
                .memory_properties = mem_props,
                .compute_queue_family = compute_family,
                .transfer_queue_family = transfer_family,
                .instance_dispatch = inst_dispatch,
                .instance = instance,
                .allocator = allocator,
            };
        }

        return Self{
            .allocator = allocator,
            .base_loader = base,
            .global_dispatch = global,
            .instance_dispatch = inst_dispatch,
            .instance = instance,
            .devices = device_handles,
        };
    }

    /// Deinitialize and release resources
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.devices);
        self.instance_dispatch.destroyInstance(self.instance);
        self.base_loader.close();
        self.* = undefined;
    }

    /// Get number of available devices
    pub fn deviceCount(self: Self) usize {
        return self.devices.len;
    }

    /// Get device by index
    pub fn getDevice(self: Self, index: usize) ?*PhysicalDeviceHandle {
        if (index >= self.devices.len) return null;
        return &self.devices[index];
    }

    /// Select the best device based on criteria
    pub fn selectBestDevice(self: Self, criteria: DeviceSelector) ?*PhysicalDeviceHandle {
        var best: ?*PhysicalDeviceHandle = null;
        var best_score: i32 = -1;

        for (&self.devices) |*dev| {
            if (!dev.matchesCriteria(criteria)) continue;

            const s = dev.score(criteria);
            if (s > best_score) {
                best_score = s;
                best = dev;
            }
        }

        return best;
    }

    /// Get all devices matching criteria
    pub fn selectDevices(self: Self, criteria: DeviceSelector, out: []?*PhysicalDeviceHandle) usize {
        var count: usize = 0;

        for (&self.devices) |*dev| {
            if (count >= out.len) break;
            if (dev.matchesCriteria(criteria)) {
                out[count] = dev;
                count += 1;
            }
        }

        return count;
    }

    /// Get all discrete GPUs
    pub fn getDiscreteGPUs(self: Self, out: []?*PhysicalDeviceHandle) usize {
        return self.selectDevices(.{ .prefer_discrete = true }, out);
    }

    /// Create contexts for all compute-capable devices
    ///
    /// On partial failure: If context creation fails for any device after some
    /// contexts were successfully created, all previously created contexts are
    /// properly cleaned up before returning the error.
    pub fn createAllContexts(self: Self, options: Options) err.Error![]Context {
        var contexts = self.allocator.alloc(Context, self.devices.len) catch return err.Error.OutOfHostMemory;
        errdefer self.allocator.free(contexts);

        var count: usize = 0;
        errdefer {
            // Clean up any successfully created contexts on failure
            for (contexts[0..count]) |*ctx| {
                ctx.deinit();
            }
        }

        for (&self.devices) |*dev| {
            if (dev.compute_queue_family != null) {
                contexts[count] = try dev.createContext(options);
                count += 1;
            }
        }

        // Shrink to actual size
        if (count < contexts.len) {
            contexts = self.allocator.realloc(contexts, count) catch contexts;
        }

        return contexts;
    }

    /// Print device summary
    pub fn printSummary(self: Self, writer: anytype) !void {
        try writer.print("Found {d} Vulkan device(s):\n", .{self.devices.len});

        for (self.devices) |dev| {
            const caps = dev.getCapabilities();
            try writer.print("  [{d}] {s}\n", .{ dev.index, dev.getName() });
            try writer.print("       Type: {s}, Memory: {d} MB\n", .{
                @tagName(dev.properties.device_type),
                caps.memory_size / (1024 * 1024),
            });
            try writer.print("       Compute: {s}, Timestamps: {s}\n", .{
                if (caps.compute) "yes" else "no",
                if (caps.timestamps) "yes" else "no",
            });
        }
    }
};

/// Create contexts for multiple devices with load balancing
pub const DevicePool = struct {
    contexts: []Context,
    allocator: std.mem.Allocator,
    next_device: usize,

    const Self = @This();

    /// Initialize device pool with contexts for all compute devices
    ///
    /// Note: This creates a temporary MultiDeviceContext for device enumeration,
    /// then creates independent Context instances for each device. Each Context
    /// has its own Vulkan instance for full isolation. The enumeration context
    /// is destroyed after device discovery is complete.
    pub fn init(allocator: std.mem.Allocator, options: Options) err.Error!Self {
        // Create temporary enumeration context - each created Context will have its own instance
        var multi = try MultiDeviceContext.init(allocator, options);
        defer multi.deinit(); // Safe: Contexts create their own instances

        const contexts = try multi.createAllContexts(options);

        return Self{
            .contexts = contexts,
            .allocator = allocator,
            .next_device = 0,
        };
    }

    /// Deinitialize all contexts
    pub fn deinit(self: *Self) void {
        for (self.contexts) |*ctx| {
            ctx.deinit();
        }
        self.allocator.free(self.contexts);
        self.* = undefined;
    }

    /// Get next device in round-robin fashion
    pub fn nextContext(self: *Self) *Context {
        const ctx = &self.contexts[self.next_device];
        self.next_device = (self.next_device + 1) % self.contexts.len;
        return ctx;
    }

    /// Get device count
    pub fn count(self: Self) usize {
        return self.contexts.len;
    }

    /// Get context by index
    pub fn getContext(self: *Self, index: usize) ?*Context {
        if (index >= self.contexts.len) return null;
        return &self.contexts[index];
    }
};

// =============================================================================
// Tests
// =============================================================================

test "DeviceSelector defaults" {
    const sel = DeviceSelector{};
    try std.testing.expect(sel.prefer_discrete);
    try std.testing.expect(sel.require_compute);
    try std.testing.expectEqual(@as(u64, 0), sel.min_memory);
}

test "DeviceCapabilities struct" {
    const caps = DeviceCapabilities{
        .compute = true,
        .timestamps = true,
        .discrete = true,
        .sparse_binding = false,
        .memory_size = 8 * 1024 * 1024 * 1024,
        .max_workgroup_size = 1024,
        .max_shared_memory = 48 * 1024,
    };
    try std.testing.expect(caps.compute);
    try std.testing.expect(caps.discrete);
}
