// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Vulkan Dynamic Loader
//!
//! This module provides dynamic loading of the Vulkan library at runtime,
//! enabling graceful degradation when Vulkan is not available.
//!
//! The loader searches for the Vulkan library in standard locations:
//! - Linux: libvulkan.so.1
//! - Windows: vulkan-1.dll
//! - macOS: libvulkan.1.dylib, libMoltenVK.dylib

const std = @import("std");
const builtin = @import("builtin");
const types = @import("types.zig");

// Use platform-specific dynamic loading (std.DynLib has issues with function pointer calls)
const c = if (builtin.os.tag == .windows) struct {
    pub extern "kernel32" fn LoadLibraryA(lpLibFileName: [*:0]const u8) ?*anyopaque;
    pub extern "kernel32" fn FreeLibrary(hLibModule: *anyopaque) c_int;
    pub extern "kernel32" fn GetProcAddress(hModule: *anyopaque, lpProcName: [*:0]const u8) ?*anyopaque;

    pub inline fn dlopen(path: [*:0]const u8, _: c_int) ?*anyopaque {
        return LoadLibraryA(path);
    }
    pub inline fn dlclose(handle: *anyopaque) c_int {
        return FreeLibrary(handle);
    }
    pub inline fn dlsym(handle: *anyopaque, symbol: [*:0]const u8) ?*anyopaque {
        return GetProcAddress(handle, symbol);
    }
    pub const RTLD_NOW: c_int = 0; // Unused on Windows
} else struct {
    pub extern "c" fn dlopen(path: [*:0]const u8, mode: c_int) ?*anyopaque;
    pub extern "c" fn dlclose(handle: *anyopaque) c_int;
    pub extern "c" fn dlsym(handle: *anyopaque, symbol: [*:0]const u8) ?*anyopaque;
    pub const RTLD_NOW: c_int = 2;
};

/// Errors that can occur during Vulkan loading
pub const LoadError = error{
    /// Vulkan library could not be found on this system
    VulkanUnavailable,
    /// Failed to load a required function from the Vulkan library
    FunctionNotFound,
    /// Memory allocation failed
    OutOfMemory,
};

/// Platform-specific library names
const LibraryNames = switch (builtin.os.tag) {
    .linux, .freebsd, .openbsd, .netbsd => &[_][:0]const u8{
        "libvulkan.so.1",
        "libvulkan.so",
        // Full paths as fallback
        "/lib/x86_64-linux-gnu/libvulkan.so.1",
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
        "/usr/lib/libvulkan.so.1",
    },
    .windows => &[_][:0]const u8{
        "vulkan-1.dll",
    },
    .macos, .ios, .tvos => &[_][:0]const u8{
        "libvulkan.1.dylib",
        "libvulkan.dylib",
        "libMoltenVK.dylib",
    },
    else => &[_][:0]const u8{},
};

/// Global function pointer types
pub const PfnGetInstanceProcAddr = *const fn (?types.Instance, [*:0]const u8) callconv(.c) ?PfnVoidFunction;
pub const PfnVoidFunction = *const fn () callconv(.c) void;

/// Base loader that holds the dynamic library handle and vkGetInstanceProcAddr
pub const BaseLoader = struct {
    handle: *anyopaque,
    vkGetInstanceProcAddr: PfnGetInstanceProcAddr,

    const Self = @This();

    /// Attempt to load the Vulkan library
    pub fn load() LoadError!Self {
        for (LibraryNames) |name| {
            const handle = c.dlopen(name.ptr, c.RTLD_NOW) orelse continue;
            const get_proc_ptr = c.dlsym(handle, "vkGetInstanceProcAddr") orelse {
                _ = c.dlclose(handle);
                continue;
            };
            const get_proc: PfnGetInstanceProcAddr = @ptrCast(@alignCast(get_proc_ptr));
            return Self{
                .handle = handle,
                .vkGetInstanceProcAddr = get_proc,
            };
        }
        return LoadError.VulkanUnavailable;
    }

    /// Close the library handle
    pub fn close(self: *Self) void {
        _ = c.dlclose(self.handle);
        self.* = undefined;
    }

    /// Get a function pointer using vkGetInstanceProcAddr
    pub fn getInstanceProcAddr(self: Self, instance: ?types.Instance, name: [*:0]const u8) ?PfnVoidFunction {
        return self.vkGetInstanceProcAddr(instance, name);
    }

    /// Check if Vulkan is available on this system
    pub fn isAvailable() bool {
        var loader = load() catch return false;
        loader.close();
        return true;
    }
};

/// Global (pre-instance) function dispatch table
pub const GlobalDispatch = struct {
    vkCreateInstance: PfnCreateInstance = null,
    vkEnumerateInstanceVersion: ?PfnEnumerateInstanceVersion = null,
    vkEnumerateInstanceExtensionProperties: PfnEnumerateInstanceExtensionProperties = null,
    vkEnumerateInstanceLayerProperties: PfnEnumerateInstanceLayerProperties = null,

    const Self = @This();

    pub const PfnCreateInstance = ?*const fn (
        *const types.InstanceCreateInfo,
        ?*const anyopaque,
        *types.Instance,
    ) callconv(.c) types.Result;

    pub const PfnEnumerateInstanceVersion = *const fn (*u32) callconv(.c) types.Result;

    pub const PfnEnumerateInstanceExtensionProperties = ?*const fn (
        ?[*:0]const u8,
        *u32,
        ?[*]types.ExtensionProperties,
    ) callconv(.c) types.Result;

    pub const PfnEnumerateInstanceLayerProperties = ?*const fn (
        *u32,
        ?[*]types.LayerProperties,
    ) callconv(.c) types.Result;

    /// Load global functions from the base loader
    pub fn load(base: BaseLoader) LoadError!Self {
        var self = Self{};

        self.vkCreateInstance = @ptrCast(base.getInstanceProcAddr(null, "vkCreateInstance"));
        if (self.vkCreateInstance == null) return LoadError.FunctionNotFound;

        self.vkEnumerateInstanceVersion = @ptrCast(base.getInstanceProcAddr(null, "vkEnumerateInstanceVersion"));
        // Optional - Vulkan 1.0 doesn't have this

        self.vkEnumerateInstanceExtensionProperties = @ptrCast(base.getInstanceProcAddr(null, "vkEnumerateInstanceExtensionProperties"));
        if (self.vkEnumerateInstanceExtensionProperties == null) return LoadError.FunctionNotFound;

        self.vkEnumerateInstanceLayerProperties = @ptrCast(base.getInstanceProcAddr(null, "vkEnumerateInstanceLayerProperties"));
        if (self.vkEnumerateInstanceLayerProperties == null) return LoadError.FunctionNotFound;

        return self;
    }

    /// Create a Vulkan instance
    pub fn createInstance(
        self: Self,
        create_info: *const types.InstanceCreateInfo,
    ) VulkanError!types.Instance {
        var instance: types.Instance = undefined;
        const result = self.vkCreateInstance.?(create_info, null, &instance);
        if (result != .success) return mapVulkanError(result);
        return instance;
    }

    /// Get the Vulkan instance version (1.1+)
    pub fn enumerateInstanceVersion(self: Self) ?u32 {
        if (self.vkEnumerateInstanceVersion) |func| {
            var version: u32 = undefined;
            if (func(&version) == .success) return version;
        }
        return null;
    }

    /// Enumerate available instance extensions
    pub fn enumerateInstanceExtensionProperties(
        self: Self,
        allocator: std.mem.Allocator,
        layer_name: ?[*:0]const u8,
    ) VulkanError![]types.ExtensionProperties {
        var count: u32 = 0;
        var result = self.vkEnumerateInstanceExtensionProperties.?(layer_name, &count, null);
        if (result != .success and result != .incomplete) return mapVulkanError(result);

        if (count == 0) return &[_]types.ExtensionProperties{};

        const props = allocator.alloc(types.ExtensionProperties, count) catch return VulkanError.OutOfHostMemory;
        result = self.vkEnumerateInstanceExtensionProperties.?(layer_name, &count, props.ptr);
        if (result != .success) {
            allocator.free(props);
            return mapVulkanError(result);
        }

        return props[0..count];
    }

    /// Enumerate available instance layers
    pub fn enumerateInstanceLayerProperties(
        self: Self,
        allocator: std.mem.Allocator,
    ) VulkanError![]types.LayerProperties {
        var count: u32 = 0;
        var result = self.vkEnumerateInstanceLayerProperties.?(&count, null);
        if (result != .success and result != .incomplete) return mapVulkanError(result);

        if (count == 0) return &[_]types.LayerProperties{};

        const props = allocator.alloc(types.LayerProperties, count) catch return VulkanError.OutOfHostMemory;
        result = self.vkEnumerateInstanceLayerProperties.?(&count, props.ptr);
        if (result != .success) {
            allocator.free(props);
            return mapVulkanError(result);
        }

        return props[0..count];
    }
};

/// Instance-level function dispatch table
pub const InstanceDispatch = struct {
    vkDestroyInstance: PfnDestroyInstance = null,
    vkEnumeratePhysicalDevices: PfnEnumeratePhysicalDevices = null,
    vkGetPhysicalDeviceProperties: PfnGetPhysicalDeviceProperties = null,
    vkGetPhysicalDeviceFeatures: PfnGetPhysicalDeviceFeatures = null,
    vkGetPhysicalDeviceQueueFamilyProperties: PfnGetPhysicalDeviceQueueFamilyProperties = null,
    vkGetPhysicalDeviceMemoryProperties: PfnGetPhysicalDeviceMemoryProperties = null,
    vkCreateDevice: PfnCreateDevice = null,
    vkEnumerateDeviceExtensionProperties: PfnEnumerateDeviceExtensionProperties = null,
    vkGetDeviceProcAddr: PfnGetDeviceProcAddr = null,

    // Debug utils (optional extension)
    vkCreateDebugUtilsMessengerEXT: ?PfnCreateDebugUtilsMessengerEXT = null,
    vkDestroyDebugUtilsMessengerEXT: ?PfnDestroyDebugUtilsMessengerEXT = null,
    vkSetDebugUtilsObjectNameEXT: ?PfnSetDebugUtilsObjectNameEXT = null,
    vkCmdBeginDebugUtilsLabelEXT: ?PfnCmdBeginDebugUtilsLabelEXT = null,
    vkCmdEndDebugUtilsLabelEXT: ?PfnCmdEndDebugUtilsLabelEXT = null,
    vkCmdInsertDebugUtilsLabelEXT: ?PfnCmdInsertDebugUtilsLabelEXT = null,

    // Vulkan 1.1+ properties/features query (optional, for extensions)
    vkGetPhysicalDeviceProperties2: ?PfnGetPhysicalDeviceProperties2 = null,
    vkGetPhysicalDeviceFeatures2: ?PfnGetPhysicalDeviceFeatures2 = null,
    vkGetPhysicalDeviceMemoryProperties2: ?PfnGetPhysicalDeviceMemoryProperties2 = null,

    // VK_KHR_cooperative_matrix (optional)
    vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR: ?PfnGetPhysicalDeviceCooperativeMatrixPropertiesKHR = null,

    const Self = @This();

    pub const PfnDestroyInstance = ?*const fn (types.Instance, ?*const anyopaque) callconv(.c) void;
    pub const PfnEnumeratePhysicalDevices = ?*const fn (types.Instance, *u32, ?[*]types.PhysicalDevice) callconv(.c) types.Result;
    pub const PfnGetPhysicalDeviceProperties = ?*const fn (types.PhysicalDevice, *types.PhysicalDeviceProperties) callconv(.c) void;
    pub const PfnGetPhysicalDeviceFeatures = ?*const fn (types.PhysicalDevice, *types.PhysicalDeviceFeatures) callconv(.c) void;
    pub const PfnGetPhysicalDeviceQueueFamilyProperties = ?*const fn (types.PhysicalDevice, *u32, ?[*]types.QueueFamilyProperties) callconv(.c) void;
    pub const PfnGetPhysicalDeviceMemoryProperties = ?*const fn (types.PhysicalDevice, *types.PhysicalDeviceMemoryProperties) callconv(.c) void;
    pub const PfnCreateDevice = ?*const fn (types.PhysicalDevice, *const types.DeviceCreateInfo, ?*const anyopaque, *types.Device) callconv(.c) types.Result;
    pub const PfnEnumerateDeviceExtensionProperties = ?*const fn (types.PhysicalDevice, ?[*:0]const u8, *u32, ?[*]types.ExtensionProperties) callconv(.c) types.Result;
    pub const PfnGetDeviceProcAddr = ?*const fn (types.Device, [*:0]const u8) callconv(.c) ?PfnVoidFunction;
    pub const PfnCreateDebugUtilsMessengerEXT = *const fn (types.Instance, *const types.DebugUtilsMessengerCreateInfoEXT, ?*const anyopaque, *types.DebugUtilsMessengerEXT) callconv(.c) types.Result;
    pub const PfnDestroyDebugUtilsMessengerEXT = *const fn (types.Instance, types.DebugUtilsMessengerEXT, ?*const anyopaque) callconv(.c) void;
    pub const PfnSetDebugUtilsObjectNameEXT = *const fn (types.Device, *const types.DebugUtilsObjectNameInfoEXT) callconv(.c) types.Result;
    pub const PfnCmdBeginDebugUtilsLabelEXT = *const fn (types.CommandBuffer, *const types.DebugUtilsLabelEXT) callconv(.c) void;
    pub const PfnCmdEndDebugUtilsLabelEXT = *const fn (types.CommandBuffer) callconv(.c) void;
    pub const PfnCmdInsertDebugUtilsLabelEXT = *const fn (types.CommandBuffer, *const types.DebugUtilsLabelEXT) callconv(.c) void;
    pub const PfnGetPhysicalDeviceProperties2 = *const fn (types.PhysicalDevice, *types.PhysicalDeviceProperties2) callconv(.c) void;
    pub const PfnGetPhysicalDeviceFeatures2 = *const fn (types.PhysicalDevice, *types.PhysicalDeviceFeatures2) callconv(.c) void;
    pub const PfnGetPhysicalDeviceMemoryProperties2 = *const fn (types.PhysicalDevice, *types.PhysicalDeviceMemoryProperties2) callconv(.c) void;
    pub const PfnGetPhysicalDeviceCooperativeMatrixPropertiesKHR = *const fn (types.PhysicalDevice, *u32, ?[*]types.CooperativeMatrixPropertiesKHR) callconv(.c) types.Result;

    /// Load instance functions
    pub fn load(base: BaseLoader, instance: types.Instance) LoadError!Self {
        var self = Self{};

        self.vkDestroyInstance = @ptrCast(base.getInstanceProcAddr(instance, "vkDestroyInstance"));
        if (self.vkDestroyInstance == null) return LoadError.FunctionNotFound;

        self.vkEnumeratePhysicalDevices = @ptrCast(base.getInstanceProcAddr(instance, "vkEnumeratePhysicalDevices"));
        if (self.vkEnumeratePhysicalDevices == null) return LoadError.FunctionNotFound;

        self.vkGetPhysicalDeviceProperties = @ptrCast(base.getInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties"));
        if (self.vkGetPhysicalDeviceProperties == null) return LoadError.FunctionNotFound;

        self.vkGetPhysicalDeviceFeatures = @ptrCast(base.getInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures"));
        if (self.vkGetPhysicalDeviceFeatures == null) return LoadError.FunctionNotFound;

        self.vkGetPhysicalDeviceQueueFamilyProperties = @ptrCast(base.getInstanceProcAddr(instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
        if (self.vkGetPhysicalDeviceQueueFamilyProperties == null) return LoadError.FunctionNotFound;

        self.vkGetPhysicalDeviceMemoryProperties = @ptrCast(base.getInstanceProcAddr(instance, "vkGetPhysicalDeviceMemoryProperties"));
        if (self.vkGetPhysicalDeviceMemoryProperties == null) return LoadError.FunctionNotFound;

        self.vkCreateDevice = @ptrCast(base.getInstanceProcAddr(instance, "vkCreateDevice"));
        if (self.vkCreateDevice == null) return LoadError.FunctionNotFound;

        self.vkEnumerateDeviceExtensionProperties = @ptrCast(base.getInstanceProcAddr(instance, "vkEnumerateDeviceExtensionProperties"));
        if (self.vkEnumerateDeviceExtensionProperties == null) return LoadError.FunctionNotFound;

        self.vkGetDeviceProcAddr = @ptrCast(base.getInstanceProcAddr(instance, "vkGetDeviceProcAddr"));
        if (self.vkGetDeviceProcAddr == null) return LoadError.FunctionNotFound;

        // Optional extension functions (VK_EXT_debug_utils)
        self.vkCreateDebugUtilsMessengerEXT = @ptrCast(base.getInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        self.vkDestroyDebugUtilsMessengerEXT = @ptrCast(base.getInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        self.vkSetDebugUtilsObjectNameEXT = @ptrCast(base.getInstanceProcAddr(instance, "vkSetDebugUtilsObjectNameEXT"));
        self.vkCmdBeginDebugUtilsLabelEXT = @ptrCast(base.getInstanceProcAddr(instance, "vkCmdBeginDebugUtilsLabelEXT"));
        self.vkCmdEndDebugUtilsLabelEXT = @ptrCast(base.getInstanceProcAddr(instance, "vkCmdEndDebugUtilsLabelEXT"));
        self.vkCmdInsertDebugUtilsLabelEXT = @ptrCast(base.getInstanceProcAddr(instance, "vkCmdInsertDebugUtilsLabelEXT"));

        // Vulkan 1.1+ functions (optional, needed for extension queries)
        self.vkGetPhysicalDeviceProperties2 = @ptrCast(base.getInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2"));
        self.vkGetPhysicalDeviceFeatures2 = @ptrCast(base.getInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2"));
        self.vkGetPhysicalDeviceMemoryProperties2 = @ptrCast(base.getInstanceProcAddr(instance, "vkGetPhysicalDeviceMemoryProperties2"));

        // VK_KHR_cooperative_matrix extension (optional)
        self.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = @ptrCast(base.getInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));

        return self;
    }

    /// Destroy a Vulkan instance
    pub fn destroyInstance(self: Self, instance: types.Instance) void {
        self.vkDestroyInstance.?(instance, null);
    }

    /// Enumerate physical devices
    pub fn enumeratePhysicalDevices(
        self: Self,
        instance: types.Instance,
        allocator: std.mem.Allocator,
    ) VulkanError![]types.PhysicalDevice {
        var count: u32 = 0;
        var result = self.vkEnumeratePhysicalDevices.?(instance, &count, null);
        if (result != .success and result != .incomplete) return mapVulkanError(result);

        if (count == 0) return &[_]types.PhysicalDevice{};

        const devices = allocator.alloc(types.PhysicalDevice, count) catch return VulkanError.OutOfHostMemory;
        result = self.vkEnumeratePhysicalDevices.?(instance, &count, devices.ptr);
        if (result != .success) {
            allocator.free(devices);
            return mapVulkanError(result);
        }

        return devices[0..count];
    }

    /// Get physical device properties
    pub fn getPhysicalDeviceProperties(self: Self, device: types.PhysicalDevice) types.PhysicalDeviceProperties {
        var props: types.PhysicalDeviceProperties = undefined;
        self.vkGetPhysicalDeviceProperties.?(device, &props);
        return props;
    }

    /// Get physical device features
    pub fn getPhysicalDeviceFeatures(self: Self, device: types.PhysicalDevice) types.PhysicalDeviceFeatures {
        var features: types.PhysicalDeviceFeatures = undefined;
        self.vkGetPhysicalDeviceFeatures.?(device, &features);
        return features;
    }

    /// Get queue family properties
    pub fn getPhysicalDeviceQueueFamilyProperties(
        self: Self,
        device: types.PhysicalDevice,
        allocator: std.mem.Allocator,
    ) VulkanError![]types.QueueFamilyProperties {
        var count: u32 = 0;
        self.vkGetPhysicalDeviceQueueFamilyProperties.?(device, &count, null);

        if (count == 0) return &[_]types.QueueFamilyProperties{};

        const props = allocator.alloc(types.QueueFamilyProperties, count) catch return VulkanError.OutOfHostMemory;
        self.vkGetPhysicalDeviceQueueFamilyProperties.?(device, &count, props.ptr);

        return props[0..count];
    }

    /// Get memory properties
    pub fn getPhysicalDeviceMemoryProperties(self: Self, device: types.PhysicalDevice) types.PhysicalDeviceMemoryProperties {
        var props: types.PhysicalDeviceMemoryProperties = undefined;
        self.vkGetPhysicalDeviceMemoryProperties.?(device, &props);
        return props;
    }

    /// Create a logical device
    pub fn createDevice(
        self: Self,
        physical_device: types.PhysicalDevice,
        create_info: *const types.DeviceCreateInfo,
    ) VulkanError!types.Device {
        var device: types.Device = undefined;
        const result = self.vkCreateDevice.?(physical_device, create_info, null, &device);
        if (result != .success) return mapVulkanError(result);
        return device;
    }

    /// Create debug utils messenger (if extension is available)
    pub fn createDebugUtilsMessengerEXT(
        self: Self,
        instance: types.Instance,
        create_info: *const types.DebugUtilsMessengerCreateInfoEXT,
    ) VulkanError!types.DebugUtilsMessengerEXT {
        if (self.vkCreateDebugUtilsMessengerEXT) |func| {
            var messenger: types.DebugUtilsMessengerEXT = undefined;
            const result = func(instance, create_info, null, &messenger);
            if (result != .success) return mapVulkanError(result);
            return messenger;
        }
        return VulkanError.ExtensionNotPresent;
    }

    /// Destroy debug utils messenger
    pub fn destroyDebugUtilsMessengerEXT(
        self: Self,
        instance: types.Instance,
        messenger: types.DebugUtilsMessengerEXT,
    ) void {
        if (self.vkDestroyDebugUtilsMessengerEXT) |func| {
            func(instance, messenger, null);
        }
    }
};

/// Device-level function dispatch table
pub const DeviceDispatch = struct {
    // Core device functions
    vkDestroyDevice: ?*const fn (types.Device, ?*const anyopaque) callconv(.c) void = null,
    vkGetDeviceQueue: ?*const fn (types.Device, u32, u32, *types.Queue) callconv(.c) void = null,
    vkDeviceWaitIdle: ?*const fn (types.Device) callconv(.c) types.Result = null,

    // Memory
    vkAllocateMemory: ?*const fn (types.Device, *const types.MemoryAllocateInfo, ?*const anyopaque, *types.DeviceMemory) callconv(.c) types.Result = null,
    vkFreeMemory: ?*const fn (types.Device, types.DeviceMemory, ?*const anyopaque) callconv(.c) void = null,
    vkMapMemory: ?*const fn (types.Device, types.DeviceMemory, u64, u64, u32, *?*anyopaque) callconv(.c) types.Result = null,
    vkUnmapMemory: ?*const fn (types.Device, types.DeviceMemory) callconv(.c) void = null,
    vkFlushMappedMemoryRanges: ?*const fn (types.Device, u32, [*]const types.MappedMemoryRange) callconv(.c) types.Result = null,
    vkInvalidateMappedMemoryRanges: ?*const fn (types.Device, u32, [*]const types.MappedMemoryRange) callconv(.c) types.Result = null,

    // Buffers
    vkCreateBuffer: ?*const fn (types.Device, *const types.BufferCreateInfo, ?*const anyopaque, *types.Buffer) callconv(.c) types.Result = null,
    vkDestroyBuffer: ?*const fn (types.Device, types.Buffer, ?*const anyopaque) callconv(.c) void = null,
    vkGetBufferMemoryRequirements: ?*const fn (types.Device, types.Buffer, *types.MemoryRequirements) callconv(.c) void = null,
    vkBindBufferMemory: ?*const fn (types.Device, types.Buffer, types.DeviceMemory, u64) callconv(.c) types.Result = null,

    // Command pools and buffers
    vkCreateCommandPool: ?*const fn (types.Device, *const types.CommandPoolCreateInfo, ?*const anyopaque, *types.CommandPool) callconv(.c) types.Result = null,
    vkDestroyCommandPool: ?*const fn (types.Device, types.CommandPool, ?*const anyopaque) callconv(.c) void = null,
    vkResetCommandPool: ?*const fn (types.Device, types.CommandPool, types.CommandPoolResetFlags) callconv(.c) types.Result = null,
    vkAllocateCommandBuffers: ?*const fn (types.Device, *const types.CommandBufferAllocateInfo, [*]types.CommandBuffer) callconv(.c) types.Result = null,
    vkFreeCommandBuffers: ?*const fn (types.Device, types.CommandPool, u32, [*]const types.CommandBuffer) callconv(.c) void = null,
    vkBeginCommandBuffer: ?*const fn (types.CommandBuffer, *const types.CommandBufferBeginInfo) callconv(.c) types.Result = null,
    vkEndCommandBuffer: ?*const fn (types.CommandBuffer) callconv(.c) types.Result = null,
    vkResetCommandBuffer: ?*const fn (types.CommandBuffer, types.CommandBufferResetFlags) callconv(.c) types.Result = null,

    // Command buffer commands
    vkCmdBindPipeline: ?*const fn (types.CommandBuffer, types.PipelineBindPoint, types.Pipeline) callconv(.c) void = null,
    vkCmdBindDescriptorSets: ?*const fn (types.CommandBuffer, types.PipelineBindPoint, types.PipelineLayout, u32, u32, [*]const types.DescriptorSet, u32, ?[*]const u32) callconv(.c) void = null,
    vkCmdDispatch: ?*const fn (types.CommandBuffer, u32, u32, u32) callconv(.c) void = null,
    vkCmdDispatchIndirect: ?*const fn (types.CommandBuffer, types.Buffer, u64) callconv(.c) void = null,
    vkCmdCopyBuffer: ?*const fn (types.CommandBuffer, types.Buffer, types.Buffer, u32, [*]const types.BufferCopy) callconv(.c) void = null,
    vkCmdFillBuffer: ?*const fn (types.CommandBuffer, types.Buffer, u64, u64, u32) callconv(.c) void = null,
    vkCmdUpdateBuffer: ?*const fn (types.CommandBuffer, types.Buffer, u64, u64, *const anyopaque) callconv(.c) void = null,
    vkCmdPipelineBarrier: ?*const fn (types.CommandBuffer, types.PipelineStageFlags, types.PipelineStageFlags, types.DependencyFlags, u32, ?[*]const types.MemoryBarrier, u32, ?[*]const types.BufferMemoryBarrier, u32, ?[*]const types.ImageMemoryBarrier) callconv(.c) void = null,
    vkCmdPushConstants: ?*const fn (types.CommandBuffer, types.PipelineLayout, types.ShaderStageFlags, u32, u32, *const anyopaque) callconv(.c) void = null,
    vkCmdWriteTimestamp: ?*const fn (types.CommandBuffer, types.PipelineStageFlags, types.QueryPool, u32) callconv(.c) void = null,
    vkCmdResetQueryPool: ?*const fn (types.CommandBuffer, types.QueryPool, u32, u32) callconv(.c) void = null,

    // Synchronization
    vkCreateFence: ?*const fn (types.Device, *const types.FenceCreateInfo, ?*const anyopaque, *types.Fence) callconv(.c) types.Result = null,
    vkDestroyFence: ?*const fn (types.Device, types.Fence, ?*const anyopaque) callconv(.c) void = null,
    vkResetFences: ?*const fn (types.Device, u32, [*]const types.Fence) callconv(.c) types.Result = null,
    vkGetFenceStatus: ?*const fn (types.Device, types.Fence) callconv(.c) types.Result = null,
    vkWaitForFences: ?*const fn (types.Device, u32, [*]const types.Fence, u32, u64) callconv(.c) types.Result = null,
    vkCreateSemaphore: ?*const fn (types.Device, *const types.SemaphoreCreateInfo, ?*const anyopaque, *types.Semaphore) callconv(.c) types.Result = null,
    vkDestroySemaphore: ?*const fn (types.Device, types.Semaphore, ?*const anyopaque) callconv(.c) void = null,

    // Events (fine-grained GPU synchronization)
    vkCreateEvent: ?*const fn (types.Device, *const types.EventCreateInfo, ?*const anyopaque, *types.Event) callconv(.c) types.Result = null,
    vkDestroyEvent: ?*const fn (types.Device, types.Event, ?*const anyopaque) callconv(.c) void = null,
    vkGetEventStatus: ?*const fn (types.Device, types.Event) callconv(.c) types.Result = null,
    vkSetEvent: ?*const fn (types.Device, types.Event) callconv(.c) types.Result = null,
    vkResetEvent: ?*const fn (types.Device, types.Event) callconv(.c) types.Result = null,
    vkCmdSetEvent: ?*const fn (types.CommandBuffer, types.Event, types.PipelineStageFlags) callconv(.c) void = null,
    vkCmdResetEvent: ?*const fn (types.CommandBuffer, types.Event, types.PipelineStageFlags) callconv(.c) void = null,
    vkCmdWaitEvents: ?*const fn (types.CommandBuffer, u32, [*]const types.Event, types.PipelineStageFlags, types.PipelineStageFlags, u32, ?[*]const types.MemoryBarrier, u32, ?[*]const types.BufferMemoryBarrier, u32, ?[*]const types.ImageMemoryBarrier) callconv(.c) void = null,

    // Timeline semaphores (Vulkan 1.2+)
    vkSignalSemaphore: ?*const fn (types.Device, *const types.SemaphoreSignalInfo) callconv(.c) types.Result = null,
    vkWaitSemaphores: ?*const fn (types.Device, *const types.SemaphoreWaitInfo, u64) callconv(.c) types.Result = null,
    vkGetSemaphoreCounterValue: ?*const fn (types.Device, types.Semaphore, *u64) callconv(.c) types.Result = null,

    // Queue submission
    vkQueueSubmit: ?*const fn (types.Queue, u32, [*]const types.SubmitInfo, ?types.Fence) callconv(.c) types.Result = null,
    vkQueueWaitIdle: ?*const fn (types.Queue) callconv(.c) types.Result = null,

    // Descriptors
    vkCreateDescriptorSetLayout: ?*const fn (types.Device, *const types.DescriptorSetLayoutCreateInfo, ?*const anyopaque, *types.DescriptorSetLayout) callconv(.c) types.Result = null,
    vkDestroyDescriptorSetLayout: ?*const fn (types.Device, types.DescriptorSetLayout, ?*const anyopaque) callconv(.c) void = null,
    vkCreateDescriptorPool: ?*const fn (types.Device, *const types.DescriptorPoolCreateInfo, ?*const anyopaque, *types.DescriptorPool) callconv(.c) types.Result = null,
    vkDestroyDescriptorPool: ?*const fn (types.Device, types.DescriptorPool, ?*const anyopaque) callconv(.c) void = null,
    vkResetDescriptorPool: ?*const fn (types.Device, types.DescriptorPool, u32) callconv(.c) types.Result = null,
    vkAllocateDescriptorSets: ?*const fn (types.Device, *const types.DescriptorSetAllocateInfo, [*]types.DescriptorSet) callconv(.c) types.Result = null,
    vkFreeDescriptorSets: ?*const fn (types.Device, types.DescriptorPool, u32, [*]const types.DescriptorSet) callconv(.c) types.Result = null,
    vkUpdateDescriptorSets: ?*const fn (types.Device, u32, [*]const types.WriteDescriptorSet, u32, ?[*]const types.CopyDescriptorSet) callconv(.c) void = null,

    // Pipelines
    vkCreateShaderModule: ?*const fn (types.Device, *const types.ShaderModuleCreateInfo, ?*const anyopaque, *types.ShaderModule) callconv(.c) types.Result = null,
    vkDestroyShaderModule: ?*const fn (types.Device, types.ShaderModule, ?*const anyopaque) callconv(.c) void = null,
    vkCreatePipelineLayout: ?*const fn (types.Device, *const types.PipelineLayoutCreateInfo, ?*const anyopaque, *types.PipelineLayout) callconv(.c) types.Result = null,
    vkDestroyPipelineLayout: ?*const fn (types.Device, types.PipelineLayout, ?*const anyopaque) callconv(.c) void = null,
    vkCreateComputePipelines: ?*const fn (types.Device, ?types.PipelineCache, u32, [*]const types.ComputePipelineCreateInfo, ?*const anyopaque, [*]types.Pipeline) callconv(.c) types.Result = null,
    vkDestroyPipeline: ?*const fn (types.Device, types.Pipeline, ?*const anyopaque) callconv(.c) void = null,

    // Query pools
    vkCreateQueryPool: ?*const fn (types.Device, *const types.QueryPoolCreateInfo, ?*const anyopaque, *types.QueryPool) callconv(.c) types.Result = null,
    vkDestroyQueryPool: ?*const fn (types.Device, types.QueryPool, ?*const anyopaque) callconv(.c) void = null,
    vkGetQueryPoolResults: ?*const fn (types.Device, types.QueryPool, u32, u32, usize, *anyopaque, u64, types.QueryResultFlags) callconv(.c) types.Result = null,

    const Self = @This();

    /// Load device functions
    pub fn load(inst_dispatch: InstanceDispatch, device: types.Device) LoadError!Self {
        var self = Self{};

        const get_proc = inst_dispatch.vkGetDeviceProcAddr orelse return LoadError.FunctionNotFound;

        // Helper to load a function
        const loader = struct {
            fn get(gp: @TypeOf(get_proc), dev: types.Device, name: [*:0]const u8) ?PfnVoidFunction {
                return gp(dev, name);
            }
        };

        // Core functions
        self.vkDestroyDevice = @ptrCast(loader.get(get_proc, device, "vkDestroyDevice"));
        self.vkGetDeviceQueue = @ptrCast(loader.get(get_proc, device, "vkGetDeviceQueue"));
        self.vkDeviceWaitIdle = @ptrCast(loader.get(get_proc, device, "vkDeviceWaitIdle"));

        // Memory
        self.vkAllocateMemory = @ptrCast(loader.get(get_proc, device, "vkAllocateMemory"));
        self.vkFreeMemory = @ptrCast(loader.get(get_proc, device, "vkFreeMemory"));
        self.vkMapMemory = @ptrCast(loader.get(get_proc, device, "vkMapMemory"));
        self.vkUnmapMemory = @ptrCast(loader.get(get_proc, device, "vkUnmapMemory"));
        self.vkFlushMappedMemoryRanges = @ptrCast(loader.get(get_proc, device, "vkFlushMappedMemoryRanges"));
        self.vkInvalidateMappedMemoryRanges = @ptrCast(loader.get(get_proc, device, "vkInvalidateMappedMemoryRanges"));

        // Buffers
        self.vkCreateBuffer = @ptrCast(loader.get(get_proc, device, "vkCreateBuffer"));
        self.vkDestroyBuffer = @ptrCast(loader.get(get_proc, device, "vkDestroyBuffer"));
        self.vkGetBufferMemoryRequirements = @ptrCast(loader.get(get_proc, device, "vkGetBufferMemoryRequirements"));
        self.vkBindBufferMemory = @ptrCast(loader.get(get_proc, device, "vkBindBufferMemory"));

        // Command pools and buffers
        self.vkCreateCommandPool = @ptrCast(loader.get(get_proc, device, "vkCreateCommandPool"));
        self.vkDestroyCommandPool = @ptrCast(loader.get(get_proc, device, "vkDestroyCommandPool"));
        self.vkResetCommandPool = @ptrCast(loader.get(get_proc, device, "vkResetCommandPool"));
        self.vkAllocateCommandBuffers = @ptrCast(loader.get(get_proc, device, "vkAllocateCommandBuffers"));
        self.vkFreeCommandBuffers = @ptrCast(loader.get(get_proc, device, "vkFreeCommandBuffers"));
        self.vkBeginCommandBuffer = @ptrCast(loader.get(get_proc, device, "vkBeginCommandBuffer"));
        self.vkEndCommandBuffer = @ptrCast(loader.get(get_proc, device, "vkEndCommandBuffer"));
        self.vkResetCommandBuffer = @ptrCast(loader.get(get_proc, device, "vkResetCommandBuffer"));

        // Command buffer commands
        self.vkCmdBindPipeline = @ptrCast(loader.get(get_proc, device, "vkCmdBindPipeline"));
        self.vkCmdBindDescriptorSets = @ptrCast(loader.get(get_proc, device, "vkCmdBindDescriptorSets"));
        self.vkCmdDispatch = @ptrCast(loader.get(get_proc, device, "vkCmdDispatch"));
        self.vkCmdDispatchIndirect = @ptrCast(loader.get(get_proc, device, "vkCmdDispatchIndirect"));
        self.vkCmdCopyBuffer = @ptrCast(loader.get(get_proc, device, "vkCmdCopyBuffer"));
        self.vkCmdFillBuffer = @ptrCast(loader.get(get_proc, device, "vkCmdFillBuffer"));
        self.vkCmdUpdateBuffer = @ptrCast(loader.get(get_proc, device, "vkCmdUpdateBuffer"));
        self.vkCmdPipelineBarrier = @ptrCast(loader.get(get_proc, device, "vkCmdPipelineBarrier"));
        self.vkCmdPushConstants = @ptrCast(loader.get(get_proc, device, "vkCmdPushConstants"));
        self.vkCmdWriteTimestamp = @ptrCast(loader.get(get_proc, device, "vkCmdWriteTimestamp"));
        self.vkCmdResetQueryPool = @ptrCast(loader.get(get_proc, device, "vkCmdResetQueryPool"));

        // Synchronization
        self.vkCreateFence = @ptrCast(loader.get(get_proc, device, "vkCreateFence"));
        self.vkDestroyFence = @ptrCast(loader.get(get_proc, device, "vkDestroyFence"));
        self.vkResetFences = @ptrCast(loader.get(get_proc, device, "vkResetFences"));
        self.vkGetFenceStatus = @ptrCast(loader.get(get_proc, device, "vkGetFenceStatus"));
        self.vkWaitForFences = @ptrCast(loader.get(get_proc, device, "vkWaitForFences"));
        self.vkCreateSemaphore = @ptrCast(loader.get(get_proc, device, "vkCreateSemaphore"));
        self.vkDestroySemaphore = @ptrCast(loader.get(get_proc, device, "vkDestroySemaphore"));

        // Events
        self.vkCreateEvent = @ptrCast(loader.get(get_proc, device, "vkCreateEvent"));
        self.vkDestroyEvent = @ptrCast(loader.get(get_proc, device, "vkDestroyEvent"));
        self.vkGetEventStatus = @ptrCast(loader.get(get_proc, device, "vkGetEventStatus"));
        self.vkSetEvent = @ptrCast(loader.get(get_proc, device, "vkSetEvent"));
        self.vkResetEvent = @ptrCast(loader.get(get_proc, device, "vkResetEvent"));
        self.vkCmdSetEvent = @ptrCast(loader.get(get_proc, device, "vkCmdSetEvent"));
        self.vkCmdResetEvent = @ptrCast(loader.get(get_proc, device, "vkCmdResetEvent"));
        self.vkCmdWaitEvents = @ptrCast(loader.get(get_proc, device, "vkCmdWaitEvents"));

        // Timeline semaphores (Vulkan 1.2+, optional)
        self.vkSignalSemaphore = @ptrCast(loader.get(get_proc, device, "vkSignalSemaphore"));
        self.vkWaitSemaphores = @ptrCast(loader.get(get_proc, device, "vkWaitSemaphores"));
        self.vkGetSemaphoreCounterValue = @ptrCast(loader.get(get_proc, device, "vkGetSemaphoreCounterValue"));

        // Queue submission
        self.vkQueueSubmit = @ptrCast(loader.get(get_proc, device, "vkQueueSubmit"));
        self.vkQueueWaitIdle = @ptrCast(loader.get(get_proc, device, "vkQueueWaitIdle"));

        // Descriptors
        self.vkCreateDescriptorSetLayout = @ptrCast(loader.get(get_proc, device, "vkCreateDescriptorSetLayout"));
        self.vkDestroyDescriptorSetLayout = @ptrCast(loader.get(get_proc, device, "vkDestroyDescriptorSetLayout"));
        self.vkCreateDescriptorPool = @ptrCast(loader.get(get_proc, device, "vkCreateDescriptorPool"));
        self.vkDestroyDescriptorPool = @ptrCast(loader.get(get_proc, device, "vkDestroyDescriptorPool"));
        self.vkResetDescriptorPool = @ptrCast(loader.get(get_proc, device, "vkResetDescriptorPool"));
        self.vkAllocateDescriptorSets = @ptrCast(loader.get(get_proc, device, "vkAllocateDescriptorSets"));
        self.vkFreeDescriptorSets = @ptrCast(loader.get(get_proc, device, "vkFreeDescriptorSets"));
        self.vkUpdateDescriptorSets = @ptrCast(loader.get(get_proc, device, "vkUpdateDescriptorSets"));

        // Pipelines
        self.vkCreateShaderModule = @ptrCast(loader.get(get_proc, device, "vkCreateShaderModule"));
        self.vkDestroyShaderModule = @ptrCast(loader.get(get_proc, device, "vkDestroyShaderModule"));
        self.vkCreatePipelineLayout = @ptrCast(loader.get(get_proc, device, "vkCreatePipelineLayout"));
        self.vkDestroyPipelineLayout = @ptrCast(loader.get(get_proc, device, "vkDestroyPipelineLayout"));
        self.vkCreateComputePipelines = @ptrCast(loader.get(get_proc, device, "vkCreateComputePipelines"));
        self.vkDestroyPipeline = @ptrCast(loader.get(get_proc, device, "vkDestroyPipeline"));

        // Query pools
        self.vkCreateQueryPool = @ptrCast(loader.get(get_proc, device, "vkCreateQueryPool"));
        self.vkDestroyQueryPool = @ptrCast(loader.get(get_proc, device, "vkDestroyQueryPool"));
        self.vkGetQueryPoolResults = @ptrCast(loader.get(get_proc, device, "vkGetQueryPoolResults"));

        return self;
    }
};

/// Vulkan errors mapped to Zig errors
pub const VulkanError = error{
    OutOfHostMemory,
    OutOfDeviceMemory,
    InitializationFailed,
    DeviceLost,
    MemoryMapFailed,
    LayerNotPresent,
    ExtensionNotPresent,
    FeatureNotPresent,
    IncompatibleDriver,
    TooManyObjects,
    FormatNotSupported,
    FragmentedPool,
    OutOfPoolMemory,
    InvalidExternalHandle,
    Fragmentation,
    InvalidOpaqueCaptureAddress,
    SurfaceLost,
    NativeWindowInUse,
    OutOfDate,
    Unknown,
};

/// Map VkResult to VulkanError
pub fn mapVulkanError(result: types.Result) VulkanError {
    return switch (result) {
        .error_out_of_host_memory => VulkanError.OutOfHostMemory,
        .error_out_of_device_memory => VulkanError.OutOfDeviceMemory,
        .error_initialization_failed => VulkanError.InitializationFailed,
        .error_device_lost => VulkanError.DeviceLost,
        .error_memory_map_failed => VulkanError.MemoryMapFailed,
        .error_layer_not_present => VulkanError.LayerNotPresent,
        .error_extension_not_present => VulkanError.ExtensionNotPresent,
        .error_feature_not_present => VulkanError.FeatureNotPresent,
        .error_incompatible_driver => VulkanError.IncompatibleDriver,
        .error_too_many_objects => VulkanError.TooManyObjects,
        .error_format_not_supported => VulkanError.FormatNotSupported,
        .error_fragmented_pool => VulkanError.FragmentedPool,
        .error_out_of_pool_memory => VulkanError.OutOfPoolMemory,
        .error_invalid_external_handle => VulkanError.InvalidExternalHandle,
        .error_fragmentation => VulkanError.Fragmentation,
        .error_invalid_opaque_capture_address => VulkanError.InvalidOpaqueCaptureAddress,
        .error_surface_lost_khr => VulkanError.SurfaceLost,
        .error_native_window_in_use_khr => VulkanError.NativeWindowInUse,
        .error_out_of_date_khr => VulkanError.OutOfDate,
        else => VulkanError.Unknown,
    };
}

// =============================================================================
// Tests
// =============================================================================

test "BaseLoader.isAvailable" {
    // This test just checks that the function runs without crashing
    _ = BaseLoader.isAvailable();
}

test "LibraryNames is not empty on supported platforms" {
    if (builtin.os.tag == .linux or
        builtin.os.tag == .windows or
        builtin.os.tag == .macos)
    {
        try std.testing.expect(LibraryNames.len > 0);
    }
}
