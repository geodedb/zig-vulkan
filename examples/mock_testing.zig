//! Mock Testing Example
//!
//! Demonstrates how to use the mock Vulkan layer for unit testing
//! without requiring actual GPU hardware.

const std = @import("std");
const vulkan = @import("vulkan");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Mock Testing Layer Example\n", .{});
    std.debug.print("==========================\n\n", .{});

    // Create mock context with custom properties
    var mock = vulkan.MockContext.init(allocator, .{
        .device_type = .discrete_gpu,
        .vendor_id = 0x10DE, // NVIDIA
        .total_memory = 8 * 1024 * 1024 * 1024, // 8GB
        .max_compute_work_group_size = .{ 1024, 1024, 64 },
    });
    defer mock.deinit();

    std.debug.print("Mock Device Configuration:\n", .{});
    std.debug.print("  Type: discrete_gpu\n", .{});
    std.debug.print("  Memory: 8 GB\n", .{});
    std.debug.print("  Max workgroup: 1024x1024x64\n\n", .{});

    // Test 1: Basic memory allocation
    std.debug.print("Test 1: Memory Allocation\n", .{});
    {
        const handle1 = try mock.allocateMemory(1024 * 1024, 0); // 1MB device local
        const handle2 = try mock.allocateMemory(512 * 1024, 2); // 512KB host visible

        std.debug.print("  Allocated 1MB device-local: handle={}\n", .{handle1});
        std.debug.print("  Allocated 512KB host-visible: handle={}\n", .{handle2});
        std.debug.print("  Total allocated: {} bytes\n", .{mock.total_allocated});

        mock.freeMemory(handle1);
        mock.freeMemory(handle2);
        std.debug.print("  After free: {} bytes\n", .{mock.total_allocated});
        std.debug.print("  Memory leaks: {}\n\n", .{mock.hasMemoryLeaks()});
    }

    // Test 2: Error injection
    std.debug.print("Test 2: Error Injection\n", .{});
    {
        // Inject OOM on next allocation
        try mock.injectError(.allocateMemory, vulkan.Error.OutOfDeviceMemory);

        const result = mock.allocateMemory(1024, 0);
        if (result) |_| {
            std.debug.print("  ERROR: Should have failed!\n", .{});
        } else |err| {
            std.debug.print("  Caught expected error: {}\n", .{err});
        }

        // Next allocation should succeed (error consumed)
        const handle = try mock.allocateMemory(1024, 0);
        std.debug.print("  Next allocation succeeded: handle={}\n", .{handle});
        mock.freeMemory(handle);
    }

    // Test 3: Device lost simulation
    std.debug.print("\nTest 3: Device Lost Simulation\n", .{});
    {
        mock.simulateDeviceLost();
        std.debug.print("  Simulated device lost\n", .{});

        // All operations should fail
        if (mock.allocateMemory(1024, 0)) |_| {
            std.debug.print("  ERROR: Should have failed!\n", .{});
        } else |err| {
            std.debug.print("  allocateMemory: {}\n", .{err});
        }

        if (mock.createBuffer(4096)) |_| {
            std.debug.print("  ERROR: Should have failed!\n", .{});
        } else |err| {
            std.debug.print("  createBuffer: {}\n", .{err});
        }

        if (mock.queueSubmit()) |_| {
            std.debug.print("  ERROR: Should have failed!\n", .{});
        } else |_| {
            std.debug.print("  queueSubmit: DeviceLost\n", .{});
        }

        mock.resetDeviceLost();
        std.debug.print("  Device reset - operations resume\n", .{});
    }

    // Test 4: SPIR-V validation
    std.debug.print("\nTest 4: SPIR-V Validation\n", .{});
    {
        // Valid SPIR-V
        const valid_spirv = try vulkan.testing.createMinimalSpirv(allocator);
        defer allocator.free(valid_spirv);

        const handle = try mock.createShaderModule(valid_spirv);
        std.debug.print("  Valid SPIR-V accepted: handle={}\n", .{handle});

        // Invalid SPIR-V (wrong magic number)
        const invalid_spirv = [_]u32{ 0xDEADBEEF, 0, 0, 0, 0 };
        if (mock.createShaderModule(&invalid_spirv)) |_| {
            std.debug.print("  ERROR: Should have rejected invalid SPIR-V!\n", .{});
        } else |err| {
            std.debug.print("  Invalid SPIR-V rejected: {}\n", .{err});
        }
    }

    // Test 5: Memory budget tracking
    std.debug.print("\nTest 5: Memory Budget Tracking\n", .{});
    {
        const initial = mock.getMemoryBudget();
        std.debug.print("  Initial device heap usage: {} / {} bytes\n", .{
            initial.heaps[0].usage,
            initial.heaps[0].budget,
        });

        // Allocate some memory
        const handle = try mock.allocateMemory(64 * 1024 * 1024, 0); // 64MB

        const after = mock.getMemoryBudget();
        std.debug.print("  After 64MB alloc: {} / {} bytes\n", .{
            after.heaps[0].usage,
            after.heaps[0].budget,
        });

        mock.freeMemory(handle);
    }

    // Test 6: Leak detection
    std.debug.print("\nTest 6: Leak Detection\n", .{});
    {
        _ = try mock.allocateMemory(4096, 0); // Intentional leak
        _ = try mock.allocateMemory(8192, 0); // Another leak

        std.debug.print("  Allocations without free\n", .{});
        std.debug.print("  Memory leaks detected: {}\n", .{mock.hasMemoryLeaks()});
        std.debug.print("  Leak count: {}\n", .{mock.getLeakCount()});
    }

    // Show final stats
    std.debug.print("\nFinal Statistics:\n", .{});
    const stats = mock.getStats();
    std.debug.print("  Total allocations: {}\n", .{stats.allocation_count});
    std.debug.print("  Total frees: {}\n", .{stats.free_count});
    std.debug.print("  Buffers created: {}\n", .{stats.buffer_count});
    std.debug.print("  Shaders created: {}\n", .{stats.shader_count});
    std.debug.print("  Queue submits: {}\n", .{stats.submit_count});

    std.debug.print("\nExample completed.\n", .{});
}
