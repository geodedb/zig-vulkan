// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Mock Vulkan Layer for Unit Testing
//!
//! Provides a mock Vulkan implementation that allows testing without
//! actual GPU hardware. Useful for CI/CD pipelines and unit tests.
//!
//! ## Features
//!
//! - Simulated Vulkan device with configurable properties
//! - Controllable error injection for testing error handling
//! - Memory tracking for leak detection
//! - Command buffer recording validation
//!
//! ## Usage
//!
//! ```zig
//! const mock = @import("vulkan").testing.mock;
//!
//! // Create a mock context
//! var ctx = mock.MockContext.init(allocator, .{});
//! defer ctx.deinit();
//!
//! // Configure error injection
//! ctx.injectError(.allocateMemory, .OutOfDeviceMemory);
//!
//! // Run tests...
//! ```

const std = @import("std");
const vk_types = @import("../vk/types.zig");
const vk_err = @import("../vk/error.zig");

/// Mock device properties for testing
pub const MockDeviceProperties = struct {
    device_name: [256]u8 = [_]u8{0} ** 256,
    device_type: vk_types.PhysicalDeviceType = .discrete_gpu,
    vendor_id: u32 = 0x10DE, // NVIDIA
    device_id: u32 = 0x1234,
    api_version: u32 = vk_types.API_VERSION_1_2,

    /// Total device memory in bytes
    total_memory: u64 = 8 * 1024 * 1024 * 1024, // 8GB
    /// Heap count
    heap_count: u32 = 2,
    /// Memory type count
    memory_type_count: u32 = 4,

    /// Maximum allocations
    max_allocations: u32 = 4096,
    /// Maximum descriptor sets
    max_descriptor_sets: u32 = 4096,
    /// Maximum compute work group count
    max_compute_work_group_count: [3]u32 = .{ 65535, 65535, 65535 },
    /// Maximum compute work group size
    max_compute_work_group_size: [3]u32 = .{ 1024, 1024, 64 },

    /// Subgroup size
    subgroup_size: u32 = 32,
    /// Subgroup supported stages
    subgroup_supported_stages: u32 = 0x00000020, // COMPUTE_BIT

    pub fn setDeviceName(self: *MockDeviceProperties, name: []const u8) void {
        const len = @min(name.len, 255);
        @memcpy(self.device_name[0..len], name[0..len]);
        self.device_name[len] = 0;
    }
};

/// Mock memory heap
pub const MockMemoryHeap = struct {
    size: u64,
    used: u64 = 0,
    flags: u32 = 0,
};

/// Mock memory allocation
pub const MockAllocation = struct {
    size: u64,
    heap_index: u32,
    memory_type: u32,
    mapped: bool = false,
    map_offset: u64 = 0,
    map_size: u64 = 0,
};

/// Operation types for error injection
pub const MockOperation = enum {
    createInstance,
    createDevice,
    allocateMemory,
    createBuffer,
    createShaderModule,
    createPipeline,
    createDescriptorPool,
    createCommandPool,
    queueSubmit,
    waitForFences,
    mapMemory,
    flushMemory,
};

/// Error injection configuration
pub const ErrorInjection = struct {
    operation: MockOperation,
    error_code: vk_err.Error,
    /// Number of times to inject (0 = infinite)
    count: u32 = 1,
    /// Delay before injecting (in operation count)
    delay: u32 = 0,

    current_count: u32 = 0,
    current_delay: u32 = 0,
};

/// Mock Vulkan context for testing
pub const MockContext = struct {
    allocator: std.mem.Allocator,
    properties: MockDeviceProperties,

    // Resource tracking
    allocations: std.AutoHashMap(u64, MockAllocation),
    next_handle: u64 = 1,
    total_allocated: u64 = 0,

    // Memory heaps
    heaps: [16]MockMemoryHeap = undefined,

    // Error injection
    injected_errors: std.ArrayListUnmanaged(ErrorInjection) = .{},

    // Statistics
    stats: MockStats = .{},

    // Device lost simulation
    device_lost: bool = false,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, properties: MockDeviceProperties) Self {
        var heaps: [16]MockMemoryHeap = undefined;
        heaps[0] = .{ .size = properties.total_memory, .flags = 1 }; // DEVICE_LOCAL
        heaps[1] = .{ .size = 256 * 1024 * 1024, .flags = 0 }; // HOST_VISIBLE

        return Self{
            .allocator = allocator,
            .properties = properties,
            .allocations = std.AutoHashMap(u64, MockAllocation).init(allocator),
            .heaps = heaps,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocations.deinit();
        self.injected_errors.deinit(self.allocator);
    }

    /// Inject an error for a specific operation
    pub fn injectError(self: *Self, operation: MockOperation, error_code: vk_err.Error) !void {
        try self.injected_errors.append(self.allocator, .{
            .operation = operation,
            .error_code = error_code,
        });
    }

    /// Inject an error with configuration
    pub fn injectErrorEx(self: *Self, config: ErrorInjection) !void {
        try self.injected_errors.append(self.allocator, config);
    }

    /// Clear all error injections
    pub fn clearErrors(self: *Self) void {
        self.injected_errors.clearRetainingCapacity();
    }

    /// Simulate device lost
    pub fn simulateDeviceLost(self: *Self) void {
        self.device_lost = true;
    }

    /// Reset device lost state
    pub fn resetDeviceLost(self: *Self) void {
        self.device_lost = false;
    }

    /// Check if an error should be injected
    fn checkInjectedError(self: *Self, operation: MockOperation) ?vk_err.Error {
        if (self.device_lost) {
            return vk_err.Error.DeviceLost;
        }

        var i: usize = 0;
        while (i < self.injected_errors.items.len) {
            var injection = &self.injected_errors.items[i];
            if (injection.operation == operation) {
                // Check delay
                if (injection.current_delay < injection.delay) {
                    injection.current_delay += 1;
                    i += 1;
                    continue;
                }

                // Inject error
                injection.current_count += 1;
                const error_code = injection.error_code;

                // Remove if count reached
                if (injection.count > 0 and injection.current_count >= injection.count) {
                    _ = self.injected_errors.orderedRemove(i);
                } else {
                    i += 1;
                }

                return error_code;
            } else {
                i += 1;
            }
        }

        return null;
    }

    // Mock Vulkan operations

    /// Mock memory allocation
    pub fn allocateMemory(self: *Self, size: u64, memory_type: u32) (vk_err.Error || std.mem.Allocator.Error)!u64 {
        if (self.checkInjectedError(.allocateMemory)) |err| {
            return err;
        }

        const heap_index: u32 = if (memory_type < 2) 0 else 1;

        // Check heap capacity
        if (self.heaps[heap_index].used + size > self.heaps[heap_index].size) {
            return vk_err.Error.OutOfDeviceMemory;
        }

        // Allocate
        const handle = self.next_handle;
        self.next_handle += 1;

        try self.allocations.put(handle, .{
            .size = size,
            .heap_index = heap_index,
            .memory_type = memory_type,
        });

        self.heaps[heap_index].used += size;
        self.total_allocated += size;
        self.stats.allocation_count += 1;
        self.stats.total_allocated += size;

        return handle;
    }

    /// Mock memory free
    pub fn freeMemory(self: *Self, handle: u64) void {
        if (self.allocations.get(handle)) |alloc| {
            self.heaps[alloc.heap_index].used -= alloc.size;
            self.total_allocated -= alloc.size;
            self.stats.free_count += 1;
            _ = self.allocations.remove(handle);
        }
    }

    /// Mock buffer creation
    pub fn createBuffer(self: *Self, size: u64) vk_err.Error!u64 {
        if (self.checkInjectedError(.createBuffer)) |err| {
            return err;
        }

        const handle = self.next_handle;
        self.next_handle += 1;
        self.stats.buffer_count += 1;

        _ = size;
        return handle;
    }

    /// Mock shader module creation
    pub fn createShaderModule(self: *Self, code: []const u32) vk_err.Error!u64 {
        if (self.checkInjectedError(.createShaderModule)) |err| {
            return err;
        }

        // Validate SPIR-V magic number
        if (code.len < 5 or code[0] != 0x07230203) {
            return vk_err.Error.InvalidSpirv;
        }

        const handle = self.next_handle;
        self.next_handle += 1;
        self.stats.shader_count += 1;

        return handle;
    }

    /// Mock queue submit
    pub fn queueSubmit(self: *Self) vk_err.Error!void {
        if (self.checkInjectedError(.queueSubmit)) |err| {
            return err;
        }

        self.stats.submit_count += 1;
    }

    /// Get memory budget info (for memory pressure testing)
    pub fn getMemoryBudget(self: Self) MockMemoryBudget {
        return .{
            .heap_count = self.properties.heap_count,
            .heaps = .{
                .{ .budget = self.heaps[0].size, .usage = self.heaps[0].used },
                .{ .budget = self.heaps[1].size, .usage = self.heaps[1].used },
                .{ .budget = 0, .usage = 0 },
                .{ .budget = 0, .usage = 0 },
            },
        };
    }

    /// Get mock statistics
    pub fn getStats(self: Self) MockStats {
        return self.stats;
    }

    /// Check for memory leaks
    pub fn hasMemoryLeaks(self: Self) bool {
        return self.allocations.count() > 0;
    }

    /// Get leak count
    pub fn getLeakCount(self: Self) usize {
        return self.allocations.count();
    }
};

/// Mock memory budget
pub const MockMemoryBudget = struct {
    heap_count: u32,
    heaps: [4]struct {
        budget: u64,
        usage: u64,
    },
};

/// Mock operation statistics
pub const MockStats = struct {
    allocation_count: u64 = 0,
    free_count: u64 = 0,
    total_allocated: u64 = 0,
    buffer_count: u64 = 0,
    shader_count: u64 = 0,
    pipeline_count: u64 = 0,
    submit_count: u64 = 0,
    device_lost_count: u64 = 0,
};

// =============================================================================
// Test Utilities
// =============================================================================

/// Create a minimal valid SPIR-V module for testing
pub fn createMinimalSpirv(allocator: std.mem.Allocator) ![]u32 {
    const spirv = try allocator.alloc(u32, 10);
    spirv[0] = 0x07230203; // Magic
    spirv[1] = 0x00010300; // Version 1.3
    spirv[2] = 0x00000000; // Generator
    spirv[3] = 0x00000020; // Bound = 32
    spirv[4] = 0x00000000; // Schema
    // OpCapability Shader
    spirv[5] = 0x00020011;
    spirv[6] = 0x00000001;
    // OpMemoryModel Logical GLSL450
    spirv[7] = 0x0003000E;
    spirv[8] = 0x00000000;
    spirv[9] = 0x00000001;
    return spirv;
}

/// Assert that a mock context has no memory leaks
pub fn assertNoLeaks(ctx: *MockContext) !void {
    if (ctx.hasMemoryLeaks()) {
        std.debug.print("Memory leak detected: {} allocation(s) not freed\n", .{ctx.getLeakCount()});
        return error.MemoryLeak;
    }
}

// =============================================================================
// Tests
// =============================================================================

test "MockContext basic operations" {
    var ctx = MockContext.init(std.testing.allocator, .{});
    defer ctx.deinit();

    // Allocate memory
    const handle = try ctx.allocateMemory(1024, 0);
    try std.testing.expect(handle > 0);
    try std.testing.expect(ctx.total_allocated == 1024);

    // Free memory
    ctx.freeMemory(handle);
    try std.testing.expect(ctx.total_allocated == 0);
    try std.testing.expect(!ctx.hasMemoryLeaks());
}

test "MockContext error injection" {
    var ctx = MockContext.init(std.testing.allocator, .{});
    defer ctx.deinit();

    // Inject OOM error
    try ctx.injectError(.allocateMemory, vk_err.Error.OutOfDeviceMemory);

    // Allocation should fail
    const result = ctx.allocateMemory(1024, 0);
    try std.testing.expectError(vk_err.Error.OutOfDeviceMemory, result);
}

test "MockContext device lost simulation" {
    var ctx = MockContext.init(std.testing.allocator, .{});
    defer ctx.deinit();

    // Simulate device lost
    ctx.simulateDeviceLost();

    // All operations should return DeviceLost
    try std.testing.expectError(vk_err.Error.DeviceLost, ctx.allocateMemory(1024, 0));
    try std.testing.expectError(vk_err.Error.DeviceLost, ctx.createBuffer(1024));
    try std.testing.expectError(vk_err.Error.DeviceLost, ctx.queueSubmit());

    // Reset and operations should succeed
    ctx.resetDeviceLost();
    const handle = try ctx.allocateMemory(1024, 0);
    ctx.freeMemory(handle);
}

test "MockContext SPIR-V validation" {
    var ctx = MockContext.init(std.testing.allocator, .{});
    defer ctx.deinit();

    // Valid SPIR-V
    const valid_spirv = try createMinimalSpirv(std.testing.allocator);
    defer std.testing.allocator.free(valid_spirv);

    const handle = try ctx.createShaderModule(valid_spirv);
    try std.testing.expect(handle > 0);

    // Invalid SPIR-V
    const invalid_spirv = [_]u32{ 0, 0, 0, 0, 0 };
    try std.testing.expectError(vk_err.Error.InvalidSpirv, ctx.createShaderModule(&invalid_spirv));
}

test "MockContext memory budget" {
    var ctx = MockContext.init(std.testing.allocator, .{});
    defer ctx.deinit();

    const budget = ctx.getMemoryBudget();
    try std.testing.expect(budget.heap_count == 2);
    try std.testing.expect(budget.heaps[0].usage == 0);

    // Allocate and check budget
    const handle = try ctx.allocateMemory(1024 * 1024, 0);
    const budget2 = ctx.getMemoryBudget();
    try std.testing.expect(budget2.heaps[0].usage == 1024 * 1024);

    ctx.freeMemory(handle);
}

test "MockContext leak detection" {
    var ctx = MockContext.init(std.testing.allocator, .{});
    defer ctx.deinit();

    // No leaks initially
    try std.testing.expect(!ctx.hasMemoryLeaks());

    // Allocate without freeing
    _ = try ctx.allocateMemory(1024, 0);

    // Should detect leak
    try std.testing.expect(ctx.hasMemoryLeaks());
    try std.testing.expect(ctx.getLeakCount() == 1);
}
