// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Vulkan Compute Context
//!
//! High-level abstraction for GPU compute operations. Manages Vulkan instance,
//! device selection, queues, and provides a clean API for compute workloads.
//!
//! ## Thread Safety
//!
//! The Context struct follows these thread-safety rules:
//!
//! | Operation | Thread Safety |
//! |-----------|---------------|
//! | Context.init | Not thread-safe (use once at startup) |
//! | Context.deinit | Not thread-safe (use once at shutdown) |
//! | Context.deviceInfo | Thread-safe (read-only) |
//! | Context.deviceExtensions | Thread-safe (read-only) |
//! | Context.waitIdle | Thread-safe (blocks all queues) |
//! | Queue submission | Thread-safe with same queue |
//!
//! ### Recommended Patterns
//!
//! - Create Context once at application startup
//! - Share Context across threads for read-only operations
//! - Use separate CommandBuffers per thread
//! - Synchronize queue access with fences/semaphores
//!
//! ### Not Thread-Safe
//!
//! - Modifying context options
//! - Destroying resources while in use by GPU
//! - Multiple threads submitting to the same queue without synchronization

const std = @import("std");
const types = @import("../vk/types.zig");
const loader = @import("../vk/loader.zig");
const err = @import("../vk/error.zig");
const mem_alloc = @import("../util/allocator.zig");

/// Debug statistics collected during context lifetime
pub const DebugStats = struct {
    /// Total validation errors
    errors: u64 = 0,
    /// Total validation warnings
    warnings: u64 = 0,
    /// Total info messages
    info_messages: u64 = 0,
    /// Total debug messages
    debug_messages: u64 = 0,
    /// Performance warnings (subset of warnings)
    performance_warnings: u64 = 0,
    /// Whether any critical errors occurred
    has_critical_errors: bool = false,

    /// Reset all statistics
    pub fn reset(self: *DebugStats) void {
        self.* = .{};
    }

    /// Get total message count
    pub fn totalMessages(self: DebugStats) u64 {
        return self.errors + self.warnings + self.info_messages + self.debug_messages;
    }
};

/// Debug severity filter
pub const DebugSeverity = packed struct {
    verbose: bool = false,
    info: bool = false,
    warning: bool = true,
    err: bool = true,

    pub const all = DebugSeverity{ .verbose = true, .info = true, .warning = true, .err = true };
    pub const warnings_and_errors = DebugSeverity{ .verbose = false, .info = false, .warning = true, .err = true };
    pub const errors_only = DebugSeverity{ .verbose = false, .info = false, .warning = false, .err = true };
};

/// Debug message type filter
pub const DebugMessageType = packed struct {
    general: bool = true,
    validation: bool = true,
    performance: bool = true,
    device_address_binding: bool = false,

    pub const all = DebugMessageType{ .general = true, .validation = true, .performance = true, .device_address_binding = true };
    pub const validation_only = DebugMessageType{ .general = false, .validation = true, .performance = false, .device_address_binding = false };
};

/// Configuration options for context creation
pub const Options = struct {
    /// Application name for Vulkan instance
    app_name: ?[*:0]const u8 = null,
    /// Application version
    app_version: u32 = 0,
    /// Enable Vulkan validation layers (debug builds)
    enable_validation: bool = false,
    /// Enable debug utils extension for detailed debugging
    enable_debug_utils: bool = false,
    /// Prefer discrete GPU over integrated
    prefer_discrete: bool = true,
    /// Specific device index to use (overrides prefer_discrete)
    device_index: ?u32 = null,
    /// Enable timestamp queries for benchmarking
    enable_timestamps: bool = false,
    /// Minimum Vulkan API version required
    min_api_version: u32 = types.API_VERSION_1_0,
    /// Custom log function (defaults to std.log)
    log_fn: ?LogFn = null,
    /// Debug severity filter (only used when enable_debug_utils is true)
    debug_severity: DebugSeverity = DebugSeverity.warnings_and_errors,
    /// Debug message types to capture
    debug_message_types: DebugMessageType = .{},
    /// Whether to collect debug statistics
    collect_debug_stats: bool = true,

    pub const LogFn = *const fn (Level, []const u8) void;
    pub const Level = enum { debug, info, warn, err };
};

/// Result of probing the system for Vulkan availability
pub const ProbeResult = struct {
    /// Whether the Vulkan loader is available
    loader_present: bool = false,
    /// Vulkan instance version (0 if not available)
    instance_version: u32 = 0,
    /// Number of physical devices found
    device_count: u32 = 0,
    /// Device information for each found device
    devices: []DeviceInfo = &.{},
    /// Any error encountered during probing
    probe_error: ?err.Error = null,

    /// Check if Vulkan is fully available for compute
    pub fn isAvailable(self: ProbeResult) bool {
        return self.loader_present and self.device_count > 0 and self.probe_error == null;
    }

    /// Get the best device for compute workloads
    pub fn getBestComputeDevice(self: ProbeResult) ?*const DeviceInfo {
        var best: ?*const DeviceInfo = null;
        for (self.devices) |*dev| {
            if (!dev.has_compute_queue) continue;
            if (best == null) {
                best = dev;
                continue;
            }
            // Prefer discrete GPUs
            if (dev.device_type == .discrete_gpu and best.?.device_type != .discrete_gpu) {
                best = dev;
            }
        }
        return best;
    }

    pub fn deinit(self: *ProbeResult, allocator: std.mem.Allocator) void {
        allocator.free(self.devices);
        self.* = .{};
    }
};

/// Information about a physical device
pub const DeviceInfo = struct {
    /// Device index
    index: u32,
    /// Device name
    name: [types.MAX_PHYSICAL_DEVICE_NAME_SIZE]u8,
    /// Device type
    device_type: types.PhysicalDeviceType,
    /// Vulkan API version supported
    api_version: u32,
    /// Driver version
    driver_version: u32,
    /// Vendor ID
    vendor_id: u32,
    /// Device ID
    device_id: u32,
    /// Whether device has a compute-capable queue
    has_compute_queue: bool,
    /// Compute queue family index
    compute_queue_family: ?u32,
    /// Maximum compute work group count
    max_compute_work_group_count: [3]u32,
    /// Maximum compute work group size
    max_compute_work_group_size: [3]u32,
    /// Maximum compute shared memory size
    max_compute_shared_memory_size: u32,
    /// Total device memory (approximate)
    total_memory: u64,
    /// Whether timestamps are supported
    timestamp_supported: bool,

    pub fn getName(self: *const DeviceInfo) []const u8 {
        const len = std.mem.indexOfScalar(u8, &self.name, 0) orelse self.name.len;
        return self.name[0..len];
    }
};

/// Main compute context
pub const Context = struct {
    allocator: std.mem.Allocator,
    options: Options,

    // Vulkan handles
    base_loader: loader.BaseLoader,
    global_dispatch: loader.GlobalDispatch,
    instance_dispatch: loader.InstanceDispatch,
    device_dispatch: loader.DeviceDispatch,

    instance: types.Instance,
    physical_device: types.PhysicalDevice,
    device: types.Device,
    compute_queue: types.Queue,
    compute_queue_family: u32,

    // Optional transfer queue (if available on separate queue family)
    transfer_queue: ?types.Queue = null,
    transfer_queue_family: ?u32 = null,
    has_async_transfer: bool = false,

    // Properties cache
    device_properties: types.PhysicalDeviceProperties,
    memory_properties: types.PhysicalDeviceMemoryProperties,

    // Resource management
    command_pool: types.CommandPool,
    gpu_allocator: mem_alloc.SimpleAllocator,

    // Debug utilities (optional)
    debug_messenger: ?types.DebugUtilsMessengerEXT = null,
    /// Debug statistics - NOTE: This field is NOT populated by the debug messenger callback.
    /// The Vulkan spec doesn't allow modifying the messenger after creation, so p_user_data
    /// cannot be set to point to this field. This field exists for manual tracking or
    /// future implementation via thread-local storage. Use getDebugStats() to read.
    debug_stats: DebugStats = .{},

    // Timestamp support
    timestamp_valid_bits: u32 = 0,
    timestamp_period: f32 = 0,

    const Self = @This();

    /// Vulkan object types for debug naming
    pub const ObjectType = enum(u32) {
        unknown = 0,
        instance = 1,
        physical_device = 2,
        device = 3,
        queue = 4,
        semaphore = 5,
        command_buffer = 6,
        fence = 7,
        device_memory = 8,
        buffer = 9,
        image = 10,
        event = 11,
        query_pool = 12,
        buffer_view = 13,
        image_view = 14,
        shader_module = 15,
        pipeline_cache = 16,
        pipeline_layout = 17,
        render_pass = 18,
        pipeline = 19,
        descriptor_set_layout = 20,
        sampler = 21,
        descriptor_pool = 22,
        descriptor_set = 23,
        framebuffer = 24,
        command_pool = 25,
    };

    /// Probe the system for Vulkan availability without creating a context
    pub fn probe(allocator: std.mem.Allocator) ProbeResult {
        var result = ProbeResult{};

        // Try to load the Vulkan library
        var base = loader.BaseLoader.load() catch |e| {
            result.probe_error = switch (e) {
                error.VulkanUnavailable => err.Error.VulkanUnavailable,
                else => err.Error.Unknown,
            };
            return result;
        };
        defer base.close();
        result.loader_present = true;

        // Load global functions
        const global = loader.GlobalDispatch.load(base) catch |e| {
            result.probe_error = switch (e) {
                error.FunctionNotFound => err.Error.FunctionNotFound,
                else => err.Error.Unknown,
            };
            return result;
        };

        // Get instance version
        result.instance_version = global.enumerateInstanceVersion() orelse types.API_VERSION_1_0;

        // Create temporary instance to enumerate devices
        const app_info = types.ApplicationInfo{
            .api_version = types.API_VERSION_1_0,
        };
        const instance_info = types.InstanceCreateInfo{
            .p_application_info = &app_info,
        };

        const instance = global.createInstance(&instance_info) catch |e| {
            result.probe_error = e;
            return result;
        };

        // Load instance dispatch
        const inst_dispatch = loader.InstanceDispatch.load(base, instance) catch |e| {
            result.probe_error = switch (e) {
                error.FunctionNotFound => err.Error.FunctionNotFound,
                else => err.Error.Unknown,
            };
            // Destroy instance using manually loaded vkDestroyInstance
            const PfnDestroyInstance = *const fn (types.Instance, ?*const anyopaque) callconv(.c) void;
            if (base.getInstanceProcAddr(instance, "vkDestroyInstance")) |proc| {
                const destroy: PfnDestroyInstance = @ptrCast(proc);
                destroy(instance, null);
            }
            return result;
        };
        defer inst_dispatch.destroyInstance(instance);

        // Enumerate physical devices
        const devices = inst_dispatch.enumeratePhysicalDevices(instance, allocator) catch |e| {
            result.probe_error = e;
            return result;
        };
        defer allocator.free(devices);

        result.device_count = @intCast(devices.len);

        // Gather device info
        var device_infos = allocator.alloc(DeviceInfo, devices.len) catch {
            result.probe_error = err.Error.OutOfHostMemory;
            return result;
        };

        for (devices, 0..) |phys_dev, i| {
            const props = inst_dispatch.getPhysicalDeviceProperties(phys_dev);
            const queue_families = inst_dispatch.getPhysicalDeviceQueueFamilyProperties(phys_dev, allocator) catch {
                device_infos[i] = std.mem.zeroes(DeviceInfo);
                continue;
            };
            defer allocator.free(queue_families);

            const mem_props = inst_dispatch.getPhysicalDeviceMemoryProperties(phys_dev);

            var compute_family: ?u32 = null;
            for (queue_families, 0..) |qf, qi| {
                if (qf.queue_flags.compute) {
                    compute_family = @intCast(qi);
                    break;
                }
            }

            var total_mem: u64 = 0;
            // Vulkan spec: memory_heap_count <= VK_MAX_MEMORY_HEAPS (16)
            const heap_count = @min(mem_props.memory_heap_count, types.MAX_MEMORY_HEAPS);
            for (0..heap_count) |hi| {
                if (mem_props.memory_heaps[hi].flags.device_local) {
                    total_mem += mem_props.memory_heaps[hi].size;
                }
            }

            device_infos[i] = DeviceInfo{
                .index = @intCast(i),
                .name = props.device_name,
                .device_type = props.device_type,
                .api_version = props.api_version,
                .driver_version = props.driver_version,
                .vendor_id = props.vendor_id,
                .device_id = props.device_id,
                .has_compute_queue = compute_family != null,
                .compute_queue_family = compute_family,
                .max_compute_work_group_count = props.limits.max_compute_work_group_count,
                .max_compute_work_group_size = props.limits.max_compute_work_group_size,
                .max_compute_shared_memory_size = props.limits.max_compute_shared_memory_size,
                .total_memory = total_mem,
                .timestamp_supported = props.limits.timestamp_compute_and_graphics != 0,
            };
        }

        result.devices = device_infos;
        return result;
    }

    /// Initialize a compute context for GPU compute operations.
    ///
    /// This creates a Vulkan instance, selects a physical device, creates a logical device,
    /// and sets up command pools for compute operations.
    ///
    /// ## Parameters
    /// - `allocator`: Memory allocator for internal allocations. Must remain valid for the
    ///   lifetime of the Context. Used for command buffers, descriptor sets, and internal
    ///   data structures.
    /// - `options`: Configuration options including device selection, validation layers,
    ///   and API version requirements. See `Options` for details.
    ///
    /// ## Returns
    /// A fully initialized Context ready for compute operations, or an error.
    ///
    /// ## Errors
    /// - `VulkanUnavailable`: Vulkan library could not be loaded
    /// - `FunctionNotFound`: Required Vulkan functions not available
    /// - `IncompatibleDriver`: Driver doesn't support requested API version
    /// - `NoSuitableDevice`: No GPU meets the selection criteria
    /// - `NoComputeQueue`: Selected device has no compute-capable queue
    /// - `OutOfHostMemory`: Memory allocation failed
    /// - `OutOfDeviceMemory`: Device memory allocation failed
    ///
    /// ## Lifecycle
    /// Call `deinit()` when done to release all Vulkan resources. The allocator must
    /// remain valid until `deinit()` is called.
    ///
    /// ## Example
    /// ```zig
    /// var ctx = try Context.init(allocator, .{
    ///     .enable_validation = true,
    ///     .device_index = null, // Auto-select best device
    /// });
    /// defer ctx.deinit();
    /// ```
    pub fn init(allocator: std.mem.Allocator, options: Options) err.Error!Self {
        // Load Vulkan library
        var base = loader.BaseLoader.load() catch return err.Error.VulkanUnavailable;
        errdefer base.close();

        // Load global dispatch
        const global = loader.GlobalDispatch.load(base) catch return err.Error.FunctionNotFound;

        // Check instance version
        const inst_version = global.enumerateInstanceVersion() orelse types.API_VERSION_1_0;
        if (inst_version < options.min_api_version) {
            return err.Error.IncompatibleDriver;
        }

        // Prepare extensions and layers
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

        // Create instance
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

        // Load instance dispatch
        const inst_dispatch = loader.InstanceDispatch.load(base, instance) catch return err.Error.FunctionNotFound;
        errdefer inst_dispatch.destroyInstance(instance);

        // Set up debug messenger if requested
        var debug_messenger: ?types.DebugUtilsMessengerEXT = null;
        if (options.enable_debug_utils and inst_dispatch.vkCreateDebugUtilsMessengerEXT != null) {
            const sev = options.debug_severity;
            const msg = options.debug_message_types;
            const debug_info = types.DebugUtilsMessengerCreateInfoEXT{
                .message_severity = .{
                    .verbose = sev.verbose,
                    .info = sev.info,
                    .warning = sev.warning,
                    .err = sev.err,
                },
                .message_type = .{
                    .general = msg.general,
                    .validation = msg.validation,
                    .performance = msg.performance,
                },
                .pfn_user_callback = debugCallback,
                // Note: p_user_data is null - debug_stats field is not updated by the callback.
                // The messenger cannot be updated after creation per Vulkan spec.
                // Use external validation layer logging or implement custom stats tracking.
            };
            debug_messenger = inst_dispatch.createDebugUtilsMessengerEXT(instance, &debug_info) catch null;
        }

        // Enumerate and select physical device
        const devices = try inst_dispatch.enumeratePhysicalDevices(instance, allocator);
        defer allocator.free(devices);

        if (devices.len == 0) {
            return err.Error.NoDevices;
        }

        var selected_device: ?types.PhysicalDevice = null;
        var selected_queue_family: u32 = 0;

        if (options.device_index) |idx| {
            if (idx < devices.len) {
                const queue_families = try inst_dispatch.getPhysicalDeviceQueueFamilyProperties(devices[idx], allocator);
                defer allocator.free(queue_families);
                for (queue_families, 0..) |qf, qi| {
                    if (qf.queue_flags.compute) {
                        selected_device = devices[idx];
                        selected_queue_family = @intCast(qi);
                        break;
                    }
                }
            }
        } else {
            // Auto-select best device
            var best_score: i32 = -1;
            for (devices) |dev| {
                const props = inst_dispatch.getPhysicalDeviceProperties(dev);
                const queue_families = try inst_dispatch.getPhysicalDeviceQueueFamilyProperties(dev, allocator);
                defer allocator.free(queue_families);

                var compute_family: ?u32 = null;
                for (queue_families, 0..) |qf, qi| {
                    if (qf.queue_flags.compute) {
                        compute_family = @intCast(qi);
                        break;
                    }
                }

                if (compute_family == null) continue;

                var score: i32 = 0;
                if (props.device_type == .discrete_gpu) {
                    score += if (options.prefer_discrete) 1000 else 0;
                } else if (props.device_type == .integrated_gpu) {
                    score += if (!options.prefer_discrete) 1000 else 500;
                }

                if (score > best_score) {
                    best_score = score;
                    selected_device = dev;
                    selected_queue_family = compute_family.?;
                }
            }
        }

        const physical_device = selected_device orelse return err.Error.NoComputeQueue;
        const device_properties = inst_dispatch.getPhysicalDeviceProperties(physical_device);
        const memory_properties = inst_dispatch.getPhysicalDeviceMemoryProperties(physical_device);

        // Get timestamp info
        var timestamp_valid_bits: u32 = 0;
        var timestamp_period: f32 = 0;
        if (options.enable_timestamps) {
            const queue_families = try inst_dispatch.getPhysicalDeviceQueueFamilyProperties(physical_device, allocator);
            defer allocator.free(queue_families);
            if (selected_queue_family < queue_families.len) {
                timestamp_valid_bits = queue_families[selected_queue_family].timestamp_valid_bits;
                timestamp_period = device_properties.limits.timestamp_period;
            }
        }

        // Create logical device
        const queue_priorities = [_]f32{1.0};
        const queue_create_info = types.DeviceQueueCreateInfo{
            .queue_family_index = selected_queue_family,
            .queue_count = 1,
            .p_queue_priorities = &queue_priorities,
        };

        const device_info = types.DeviceCreateInfo{
            .queue_create_info_count = 1,
            .p_queue_create_infos = @ptrCast(&queue_create_info),
        };

        const device = try inst_dispatch.createDevice(physical_device, &device_info);

        // Load device dispatch
        const dev_dispatch = loader.DeviceDispatch.load(inst_dispatch, device) catch return err.Error.FunctionNotFound;
        errdefer dev_dispatch.vkDestroyDevice.?(device, null);

        // Get compute queue
        var compute_queue: types.Queue = undefined;
        dev_dispatch.vkGetDeviceQueue.?(device, selected_queue_family, 0, &compute_queue);

        // Create command pool
        const pool_info = types.CommandPoolCreateInfo{
            .flags = .{ .reset_command_buffer = true },
            .queue_family_index = selected_queue_family,
        };

        var command_pool: types.CommandPool = undefined;
        const pool_result = dev_dispatch.vkCreateCommandPool.?(device, &pool_info, null, &command_pool);
        if (pool_result != .success) {
            return err.fromVkResult(pool_result);
        }

        // Initialize GPU allocator
        const gpu_allocator = mem_alloc.SimpleAllocator.init(device, dev_dispatch, memory_properties, allocator);

        return Self{
            .allocator = allocator,
            .options = options,
            .base_loader = base,
            .global_dispatch = global,
            .instance_dispatch = inst_dispatch,
            .device_dispatch = dev_dispatch,
            .instance = instance,
            .physical_device = physical_device,
            .device = device,
            .compute_queue = compute_queue,
            .compute_queue_family = selected_queue_family,
            .device_properties = device_properties,
            .memory_properties = memory_properties,
            .command_pool = command_pool,
            .gpu_allocator = gpu_allocator,
            .debug_messenger = debug_messenger,
            .timestamp_valid_bits = timestamp_valid_bits,
            .timestamp_period = timestamp_period,
        };
    }

    /// Deinitialize the context and release all resources
    pub fn deinit(self: *Self) void {
        // Wait for device to be idle
        _ = self.device_dispatch.vkDeviceWaitIdle.?(self.device);

        // Clean up GPU allocator
        self.gpu_allocator.deinit();

        // Destroy command pool
        self.device_dispatch.vkDestroyCommandPool.?(self.device, self.command_pool, null);

        // Destroy device
        self.device_dispatch.vkDestroyDevice.?(self.device, null);

        // Destroy debug messenger if present
        if (self.debug_messenger) |messenger| {
            self.instance_dispatch.destroyDebugUtilsMessengerEXT(self.instance, messenger);
        }

        // Destroy instance
        self.instance_dispatch.destroyInstance(self.instance);

        // Close loader
        self.base_loader.close();

        self.* = undefined;
    }

    /// Get device information
    pub fn deviceInfo(self: Self) DeviceInfo {
        return DeviceInfo{
            .index = 0,
            .name = self.device_properties.device_name,
            .device_type = self.device_properties.device_type,
            .api_version = self.device_properties.api_version,
            .driver_version = self.device_properties.driver_version,
            .vendor_id = self.device_properties.vendor_id,
            .device_id = self.device_properties.device_id,
            .has_compute_queue = true,
            .compute_queue_family = self.compute_queue_family,
            .max_compute_work_group_count = self.device_properties.limits.max_compute_work_group_count,
            .max_compute_work_group_size = self.device_properties.limits.max_compute_work_group_size,
            .max_compute_shared_memory_size = self.device_properties.limits.max_compute_shared_memory_size,
            .total_memory = blk: {
                var total: u64 = 0;
                // Vulkan spec: memory_heap_count <= VK_MAX_MEMORY_HEAPS (16)
                const heap_count = @min(self.memory_properties.memory_heap_count, types.MAX_MEMORY_HEAPS);
                for (0..heap_count) |i| {
                    if (self.memory_properties.memory_heaps[i].flags.device_local) {
                        total += self.memory_properties.memory_heaps[i].size;
                    }
                }
                break :blk total;
            },
            .timestamp_supported = self.timestamp_valid_bits > 0,
        };
    }

    /// Get cached physical device properties
    pub fn getPhysicalDeviceProperties(self: Self) types.PhysicalDeviceProperties {
        return self.device_properties;
    }

    /// Get timestamp period in nanoseconds per tick
    pub fn getTimestampPeriod(self: Self) f32 {
        return self.timestamp_period;
    }

    /// Check if GPU timestamps are supported
    pub fn supportsTimestamps(self: Self) bool {
        return self.timestamp_valid_bits > 0;
    }

    /// Wait for device to become idle
    pub fn waitIdle(self: Self) err.Error!void {
        const result = self.device_dispatch.vkDeviceWaitIdle.?(self.device);
        if (result != .success) return err.fromVkResult(result);
    }

    /// Get debug statistics (only meaningful when debug_utils is enabled)
    pub fn getDebugStats(self: Self) DebugStats {
        return self.debug_stats;
    }

    /// Reset debug statistics
    pub fn resetDebugStats(self: *Self) void {
        self.debug_stats.reset();
    }

    /// Set a debug name for a Vulkan object
    /// This is useful for debugging and validation messages
    /// Only works when enable_debug_utils is true
    pub fn setDebugName(self: Self, object_type: ObjectType, handle: u64, name: [*:0]const u8) void {
        if (!self.options.enable_debug_utils) return;
        if (self.instance_dispatch.vkSetDebugUtilsObjectNameEXT == null) return;

        const name_info = types.DebugUtilsObjectNameInfoEXT{
            .object_type = @enumFromInt(@intFromEnum(object_type)),
            .object_handle = handle,
            .p_object_name = name,
        };
        _ = self.instance_dispatch.vkSetDebugUtilsObjectNameEXT.?(self.device, &name_info);
    }

    /// Set debug name for a buffer
    pub fn setBufferName(self: Self, buffer: types.Buffer, name: [*:0]const u8) void {
        self.setDebugName(.buffer, @intFromPtr(buffer), name);
    }

    /// Set debug name for a pipeline
    pub fn setPipelineName(self: Self, pipeline: types.Pipeline, name: [*:0]const u8) void {
        self.setDebugName(.pipeline, @intFromPtr(pipeline), name);
    }

    /// Set debug name for a descriptor set
    pub fn setDescriptorSetName(self: Self, set: types.DescriptorSet, name: [*:0]const u8) void {
        self.setDebugName(.descriptor_set, @intFromPtr(set), name);
    }

    /// Set debug name for a command buffer
    pub fn setCommandBufferName(self: Self, cmd: types.CommandBuffer, name: [*:0]const u8) void {
        self.setDebugName(.command_buffer, @intFromPtr(cmd), name);
    }

    /// Set debug name for a shader module
    pub fn setShaderModuleName(self: Self, shader: types.ShaderModule, name: [*:0]const u8) void {
        self.setDebugName(.shader_module, @intFromPtr(shader), name);
    }

    /// Insert a debug label into a command buffer (for GPU debuggers like RenderDoc)
    pub fn insertDebugLabel(self: Self, cmd_buffer: types.CommandBuffer, label: [*:0]const u8, color: ?[4]f32) void {
        if (!self.options.enable_debug_utils) return;
        if (self.instance_dispatch.vkCmdInsertDebugUtilsLabelEXT == null) return;

        const label_info = types.DebugUtilsLabelEXT{
            .p_label_name = label,
            .color = color orelse [4]f32{ 1.0, 1.0, 1.0, 1.0 },
        };
        self.instance_dispatch.vkCmdInsertDebugUtilsLabelEXT.?(cmd_buffer, &label_info);
    }

    /// Begin a debug label region in a command buffer
    pub fn beginDebugLabel(self: Self, cmd_buffer: types.CommandBuffer, label: [*:0]const u8, color: ?[4]f32) void {
        if (!self.options.enable_debug_utils) return;
        if (self.instance_dispatch.vkCmdBeginDebugUtilsLabelEXT == null) return;

        const label_info = types.DebugUtilsLabelEXT{
            .p_label_name = label,
            .color = color orelse [4]f32{ 1.0, 1.0, 1.0, 1.0 },
        };
        self.instance_dispatch.vkCmdBeginDebugUtilsLabelEXT.?(cmd_buffer, &label_info);
    }

    /// End a debug label region in a command buffer
    pub fn endDebugLabel(self: Self, cmd_buffer: types.CommandBuffer) void {
        if (!self.options.enable_debug_utils) return;
        if (self.instance_dispatch.vkCmdEndDebugUtilsLabelEXT == null) return;

        self.instance_dispatch.vkCmdEndDebugUtilsLabelEXT.?(cmd_buffer);
    }

    /// Check if debug utilities are available and enabled
    pub fn hasDebugUtils(self: Self) bool {
        return self.options.enable_debug_utils and self.debug_messenger != null;
    }

    /// Debug callback for validation messages
    ///
    /// Note: user_data is always null because the Vulkan spec doesn't allow modifying
    /// the debug messenger after creation. Statistics tracking would require thread-local
    /// storage or a global atomic counter approach. Use the logging output for diagnostics.
    fn debugCallback(
        severity: types.DebugUtilsMessageSeverityFlagsEXT,
        _: types.DebugUtilsMessageTypeFlagsEXT,
        callback_data: *const types.DebugUtilsMessengerCallbackDataEXT,
        _: ?*anyopaque, // Always null - see note above
    ) callconv(.c) u32 {
        const message = if (callback_data.p_message) |msg| std.mem.span(msg) else "(no message)";

        // Log the message at appropriate level
        if (severity.err) {
            std.log.err("[Vulkan] {s}", .{message});
        } else if (severity.warning) {
            std.log.warn("[Vulkan] {s}", .{message});
        } else if (severity.info) {
            std.log.info("[Vulkan] {s}", .{message});
        } else {
            std.log.debug("[Vulkan] {s}", .{message});
        }

        return 0; // VK_FALSE - don't abort
    }
};

// =============================================================================
// Tests
// =============================================================================

test "ProbeResult.isAvailable" {
    var result = ProbeResult{};
    try std.testing.expect(!result.isAvailable());

    result.loader_present = true;
    result.device_count = 1;
    try std.testing.expect(result.isAvailable());

    result.probe_error = err.Error.Unknown;
    try std.testing.expect(!result.isAvailable());
}

test "Options defaults" {
    const opts = Options{};
    try std.testing.expect(!opts.enable_validation);
    try std.testing.expect(opts.prefer_discrete);
    try std.testing.expectEqual(opts.device_index, null);
}
