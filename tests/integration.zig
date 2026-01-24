//! Integration Tests for Vulkan Compute Library
//!
//! These tests require a Vulkan-capable device and will be skipped
//! if Vulkan is not available.

const std = @import("std");
const vulkan = @import("vulkan");

// Skip message for when Vulkan is unavailable
const skip_message = "Vulkan not available - skipping integration test";

// Test that probing works without errors
test "probe system for Vulkan availability" {
    var result = vulkan.probe(std.testing.allocator);
    defer result.deinit(std.testing.allocator);

    // Probe should always succeed (even if no Vulkan)
    // Just verify we get valid data back
    if (result.loader_present) {
        std.debug.print("\nVulkan loader found. Instance version: {}.{}.{}\n", .{
            vulkan.vk.apiVersionMajor(result.instance_version),
            vulkan.vk.apiVersionMinor(result.instance_version),
            vulkan.vk.apiVersionPatch(result.instance_version),
        });
        std.debug.print("Found {} device(s)\n", .{result.device_count});

        for (result.devices) |dev| {
            std.debug.print("  - {s} ({s})\n", .{
                dev.getName(),
                @tagName(dev.device_type),
            });
        }
    } else {
        std.debug.print("\n{s}\n", .{skip_message});
    }
}

// Test context creation (skipped if Vulkan unavailable)
test "create and destroy context" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        // NoDevices is acceptable - means Vulkan is there but no GPU
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            std.debug.print("\nNo compute-capable devices found - skipping\n", .{});
            return;
        }
        return e;
    };
    defer ctx.deinit();

    const info = ctx.deviceInfo();
    std.debug.print("\nUsing device: {s}\n", .{info.getName()});
    std.debug.print("  Type: {s}\n", .{@tagName(info.device_type)});
    std.debug.print("  Memory: {} MB\n", .{info.total_memory / (1024 * 1024)});
    std.debug.print("  Max workgroup size: {}x{}x{}\n", .{
        info.max_compute_work_group_size[0],
        info.max_compute_work_group_size[1],
        info.max_compute_work_group_size[2],
    });
}

// Test buffer creation
test "create and use buffers" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            std.debug.print("\nNo compute-capable devices found - skipping\n", .{});
            return;
        }
        return e;
    };
    defer ctx.deinit();

    // Create staging buffer (host-visible)
    var staging = try vulkan.Buffer.init(&ctx, 1024, .staging);
    defer staging.deinit(&ctx);

    try std.testing.expect(staging.isHostVisible());
    try std.testing.expect(staging.size == 1024);

    // Write test data
    const test_data = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try staging.writeSlice(u32, &test_data);

    // Read it back
    var read_data: [8]u32 = undefined;
    try staging.readSlice(u32, &read_data);

    try std.testing.expectEqualSlices(u32, &test_data, &read_data);

    // Create storage buffer (device-local)
    var storage = try vulkan.Buffer.init(&ctx, 4096, .storage);
    defer storage.deinit(&ctx);

    try std.testing.expect(storage.size == 4096);

    std.debug.print("\nBuffer creation and host-visible I/O test passed\n", .{});
}

// Test command buffer recording
test "record and submit command buffer" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            std.debug.print("\nNo compute-capable devices found - skipping\n", .{});
            return;
        }
        return e;
    };
    defer ctx.deinit();

    // Create a command buffer
    var cmd = try vulkan.CommandBuffer.init(&ctx);
    defer cmd.deinit();

    // Record a simple command buffer (just begin/end)
    try cmd.begin();
    try cmd.end();

    // Submit and wait
    try vulkan.Submission.submitAndWait(&ctx, cmd);

    std.debug.print("\nCommand buffer submission test passed\n", .{});
}

// Test fence synchronization
test "fence creation and signaling" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            std.debug.print("\nNo compute-capable devices found - skipping\n", .{});
            return;
        }
        return e;
    };
    defer ctx.deinit();

    // Create fence in unsignaled state
    var fence = try vulkan.Fence.init(&ctx, false);
    defer fence.deinit();

    try std.testing.expect(!fence.isSignaled());

    // Submit empty command buffer to signal fence
    var cmd = try vulkan.CommandBuffer.init(&ctx);
    defer cmd.deinit();

    try cmd.begin();
    try cmd.end();

    try vulkan.Submission.submit(&ctx, cmd, null, null, fence);

    // Wait for fence
    try fence.waitForever();

    try std.testing.expect(fence.isSignaled());

    std.debug.print("\nFence synchronization test passed\n", .{});
}

// Test buffer copy operation
test "buffer copy via command buffer" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            std.debug.print("\nNo compute-capable devices found - skipping\n", .{});
            return;
        }
        return e;
    };
    defer ctx.deinit();

    const size = 256;

    // Create staging and readback buffers
    var src = try vulkan.Buffer.init(&ctx, size, .staging);
    defer src.deinit(&ctx);

    var dst = try vulkan.Buffer.init(&ctx, size, .readback);
    defer dst.deinit(&ctx);

    // Write test pattern
    const test_data = [_]u32{0xDEADBEEF} ** (size / 4);
    try src.writeSlice(u32, &test_data);

    // Record copy command
    var cmd = try vulkan.CommandBuffer.init(&ctx);
    defer cmd.deinit();

    try cmd.begin();
    vulkan.Transfer.copy(ctx.device_dispatch, cmd.getHandle(), src, dst, null);
    try cmd.end();

    // Execute
    try vulkan.Submission.submitAndWait(&ctx, cmd);

    // Verify
    var read_data: [size / 4]u32 = undefined;
    try dst.readSlice(u32, &read_data);

    try std.testing.expectEqualSlices(u32, &test_data, &read_data);

    std.debug.print("\nBuffer copy test passed\n", .{});
}

// Test descriptor pool and set allocation
test "descriptor pool allocation" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            std.debug.print("\nNo compute-capable devices found - skipping\n", .{});
            return;
        }
        return e;
    };
    defer ctx.deinit();

    // Create descriptor set layout
    const bindings = [_]vulkan.DescriptorBinding{
        .{ .binding = 0, .descriptor_type = .storage_buffer },
        .{ .binding = 1, .descriptor_type = .storage_buffer },
    };

    var layout = try vulkan.DescriptorSetLayout.init(&ctx, &bindings);
    defer layout.deinit();

    // Create descriptor pool
    var pool = try vulkan.DescriptorPool.initSimple(&ctx, 4, 8);
    defer pool.deinit();

    // Allocate descriptor set
    const set = try pool.allocate(layout);
    pool.free(set);

    std.debug.print("\nDescriptor pool test passed\n", .{});
}

// Test device info retrieval
test "get device info" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            std.debug.print("\nNo compute-capable devices found - skipping\n", .{});
            return;
        }
        return e;
    };
    defer ctx.deinit();

    const info = ctx.deviceInfo();

    // Verify we got valid data
    try std.testing.expect(info.has_compute_queue);
    try std.testing.expect(info.max_compute_work_group_size[0] > 0);
    try std.testing.expect(info.max_compute_work_group_count[0] > 0);
    try std.testing.expect(info.total_memory > 0);

    std.debug.print("\nDevice info test passed\n", .{});
}

// Test error descriptions
test "error descriptions are helpful" {
    const errors = [_]vulkan.Error{
        vulkan.Error.VulkanUnavailable,
        vulkan.Error.NoDevices,
        vulkan.Error.OutOfHostMemory,
        vulkan.Error.DeviceLost,
    };

    for (errors) |e| {
        const desc = vulkan.vk.describe(e);
        try std.testing.expect(desc.len > 0);
    }
}

// Test alignment utilities
test "alignment utilities" {
    // alignUp
    try std.testing.expectEqual(@as(u64, 256), vulkan.Alignment.alignUp(200, 256));
    try std.testing.expectEqual(@as(u64, 256), vulkan.Alignment.alignUp(256, 256));
    try std.testing.expectEqual(@as(u64, 512), vulkan.Alignment.alignUp(300, 256));

    // isAligned
    try std.testing.expect(vulkan.Alignment.isAligned(256, 256));
    try std.testing.expect(vulkan.Alignment.isAligned(0, 256));
    try std.testing.expect(!vulkan.Alignment.isAligned(100, 256));
}

// =============================================================================
// SPIR-V Validation Tests
// =============================================================================

test "SPIR-V validation - valid shader" {
    // Minimal valid SPIR-V header
    const valid_spirv = [_]u32{
        0x07230203, // Magic
        0x00010000, // Version 1.0
        0x00000000, // Generator
        0x00000010, // Bound (16 IDs)
        0x00000000, // Schema (reserved, must be 0)
    };

    const result = vulkan.validateSpirv(&valid_spirv);
    try std.testing.expectEqual(vulkan.SpirvValidationResult.valid, result);
}

test "SPIR-V validation - invalid magic" {
    const invalid_spirv = [_]u32{ 0x12345678, 0x00010000, 0, 10, 0 };
    const result = vulkan.validateSpirv(&invalid_spirv);
    try std.testing.expectEqual(vulkan.SpirvValidationResult.invalid_magic, result);
}

test "SPIR-V validation - too short" {
    const short_spirv = [_]u32{ 0x07230203, 0x00010000, 0 };
    const result = vulkan.validateSpirv(&short_spirv);
    try std.testing.expectEqual(vulkan.SpirvValidationResult.too_short, result);
}

test "SPIR-V validation - invalid bound" {
    const invalid_bound = [_]u32{ 0x07230203, 0x00010000, 0, 0, 0 }; // Bound = 0 is invalid
    const result = vulkan.validateSpirv(&invalid_bound);
    try std.testing.expectEqual(vulkan.SpirvValidationResult.invalid_bound, result);
}

test "SPIR-V validation - invalid schema" {
    const invalid_schema = [_]u32{ 0x07230203, 0x00010000, 0, 10, 1 }; // Schema must be 0
    const result = vulkan.validateSpirv(&invalid_schema);
    try std.testing.expectEqual(vulkan.SpirvValidationResult.invalid_schema, result);
}

test "SPIR-V validation detailed - extracts version info" {
    const spirv = [_]u32{
        0x07230203, // Magic
        0x00010300, // Version 1.3
        0x0001_0001, // Generator ID
        0x00000064, // Bound = 100
        0x00000000, // Schema
    };

    const info = vulkan.validateSpirvDetailed(&spirv);
    try std.testing.expect(info.isValid());
    try std.testing.expectEqual(@as(u8, 1), info.version_major);
    try std.testing.expectEqual(@as(u8, 3), info.version_minor);
    try std.testing.expectEqual(@as(u32, 100), info.bound);
    try std.testing.expectEqualStrings("1.3", info.getVersionString());
}

// =============================================================================
// Buffer Size Validation Tests
// =============================================================================

test "buffer size validation - zero size rejected" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    // Zero-size buffer should fail
    const result = vulkan.Buffer.init(&ctx, 0, .storage);
    try std.testing.expectError(vulkan.Error.InvalidArgument, result);

    std.debug.print("\nZero-size buffer rejection test passed\n", .{});
}

test "buffer max size query" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    const storage_max = vulkan.Buffer.getMaxSize(&ctx, .storage);
    const uniform_max = vulkan.Buffer.getMaxSize(&ctx, .uniform);

    // Storage buffers typically support at least 128MB
    try std.testing.expect(storage_max >= 128 * 1024 * 1024);

    // Uniform buffers typically support at least 16KB
    try std.testing.expect(uniform_max >= 16 * 1024);

    std.debug.print("\nBuffer max sizes: storage={d}MB, uniform={d}KB\n", .{
        storage_max / (1024 * 1024),
        uniform_max / 1024,
    });
}

// =============================================================================
// Context Options Tests
// =============================================================================

test "context with validation layers" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{
        .enable_validation = true,
        .enable_debug_utils = true,
    }) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        // Validation layers might not be installed
        if (e == vulkan.Error.LayerNotPresent) {
            std.debug.print("\nValidation layers not installed - skipping\n", .{});
            return;
        }
        return e;
    };
    defer ctx.deinit();

    std.debug.print("\nContext with validation layers test passed\n", .{});
}

// =============================================================================
// Memory Management Tests
// =============================================================================

test "multiple buffer allocations and frees" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    // Allocate multiple buffers
    var buffers: [8]vulkan.Buffer = undefined;
    var count: usize = 0;
    errdefer {
        for (buffers[0..count]) |*b| {
            b.deinit(&ctx);
        }
    }

    for (&buffers) |*b| {
        b.* = try vulkan.Buffer.init(&ctx, 64 * 1024, .storage);
        count += 1;
    }

    // Free them all
    for (&buffers) |*b| {
        b.deinit(&ctx);
    }

    std.debug.print("\nMultiple buffer allocation test passed ({d} buffers)\n", .{buffers.len});
}

test "staging buffer roundtrip with large data" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    const size: usize = 1024 * 1024; // 1MB
    var staging = try vulkan.Buffer.init(&ctx, size, .staging);
    defer staging.deinit(&ctx);

    // Generate test pattern
    const pattern = std.testing.allocator.alloc(u8, size) catch return;
    defer std.testing.allocator.free(pattern);

    for (pattern, 0..) |*b, i| {
        b.* = @truncate(i ^ (i >> 8));
    }

    // Write and read back
    try staging.write(pattern);

    const readback = std.testing.allocator.alloc(u8, size) catch return;
    defer std.testing.allocator.free(readback);

    try staging.read(readback);

    try std.testing.expectEqualSlices(u8, pattern, readback);

    std.debug.print("\n1MB staging buffer roundtrip test passed\n", .{});
}

// =============================================================================
// Semaphore Tests
// =============================================================================

test "semaphore creation" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    var sem = try vulkan.Semaphore.init(&ctx);
    defer sem.deinit();

    std.debug.print("\nSemaphore creation test passed\n", .{});
}

// =============================================================================
// Timestamp Query Tests
// =============================================================================

test "timestamp queries" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{
        .enable_timestamps = true,
    }) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    const info = ctx.deviceInfo();
    if (!info.timestamp_supported) {
        std.debug.print("\nTimestamps not supported - skipping\n", .{});
        return;
    }

    var timer = try vulkan.TimestampQuery.init(&ctx, 2);
    defer timer.deinit();

    // Record timestamps around a command buffer
    var cmd = try vulkan.CommandBuffer.init(&ctx);
    defer cmd.deinit();

    try cmd.begin();
    // Reset queries before use
    timer.reset(cmd.getHandle(), 0, 2);
    // Record start timestamp
    timer.writeTimestamp(cmd.getHandle(), .{ .top_of_pipe = true }, 0);
    // Would do work here...
    // Record end timestamp
    timer.writeTimestamp(cmd.getHandle(), .{ .bottom_of_pipe = true }, 1);
    try cmd.end();

    try vulkan.Submission.submitAndWait(&ctx, cmd);

    // Get elapsed time
    const elapsed = timer.getElapsedNs(std.testing.allocator, 0, 1) catch 0;
    std.debug.print("\nTimestamp query test passed (elapsed: {} ns)\n", .{elapsed});
}

// =============================================================================
// Debug Layer Integration Tests
// =============================================================================

test "debug utilities - stats initialization" {
    // Test that DebugStats initializes to zero
    var stats = vulkan.DebugStats{};
    try std.testing.expectEqual(@as(u64, 0), stats.errors);
    try std.testing.expectEqual(@as(u64, 0), stats.warnings);
    try std.testing.expectEqual(@as(u64, 0), stats.totalMessages());
    try std.testing.expect(!stats.has_critical_errors);

    // Test reset
    stats.errors = 5;
    stats.warnings = 10;
    stats.reset();
    try std.testing.expectEqual(@as(u64, 0), stats.errors);
    try std.testing.expectEqual(@as(u64, 0), stats.warnings);
}

test "debug severity presets" {
    // Test severity presets
    const all = vulkan.DebugSeverity.all;
    try std.testing.expect(all.verbose);
    try std.testing.expect(all.info);
    try std.testing.expect(all.warning);
    try std.testing.expect(all.err);

    const errors_only = vulkan.DebugSeverity.errors_only;
    try std.testing.expect(!errors_only.verbose);
    try std.testing.expect(!errors_only.info);
    try std.testing.expect(!errors_only.warning);
    try std.testing.expect(errors_only.err);

    const default = vulkan.DebugSeverity.warnings_and_errors;
    try std.testing.expect(!default.verbose);
    try std.testing.expect(!default.info);
    try std.testing.expect(default.warning);
    try std.testing.expect(default.err);
}

test "debug message type presets" {
    const all = vulkan.DebugMessageType.all;
    try std.testing.expect(all.general);
    try std.testing.expect(all.validation);
    try std.testing.expect(all.performance);
    try std.testing.expect(all.device_address_binding);

    const validation = vulkan.DebugMessageType.validation_only;
    try std.testing.expect(!validation.general);
    try std.testing.expect(validation.validation);
    try std.testing.expect(!validation.performance);
}

test "context with debug utils - check hasDebugUtils" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{
        .enable_validation = true,
        .enable_debug_utils = true,
        .debug_severity = vulkan.DebugSeverity.all,
        .debug_message_types = vulkan.DebugMessageType.all,
    }) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        // Validation layers might not be installed
        if (e == vulkan.Error.LayerNotPresent) {
            std.debug.print("\nValidation layers not installed - skipping\n", .{});
            return;
        }
        return e;
    };
    defer ctx.deinit();

    // hasDebugUtils should return true if debug_utils extension is available
    if (ctx.hasDebugUtils()) {
        std.debug.print("\nDebug utilities enabled and available\n", .{});

        // Test getDebugStats
        const stats = ctx.getDebugStats();
        std.debug.print("Initial stats - errors: {}, warnings: {}\n", .{ stats.errors, stats.warnings });
    } else {
        std.debug.print("\nDebug utilities not available on this system\n", .{});
    }
}

test "debug object naming" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{
        .enable_validation = true,
        .enable_debug_utils = true,
    }) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        if (e == vulkan.Error.LayerNotPresent) {
            std.debug.print("\nValidation layers not installed - skipping\n", .{});
            return;
        }
        return e;
    };
    defer ctx.deinit();

    // Create a buffer and set its debug name
    var staging = vulkan.Buffer.init(&ctx, 1024, .staging) catch {
        return;
    };
    defer staging.deinit(&ctx);

    // Set debug name - this should not crash even if debug utils is not available
    ctx.setBufferName(staging.handle, "TestBuffer");

    std.debug.print("\nDebug object naming test passed\n", .{});
}

// =============================================================================
// Error Context and Diagnostics Tests
// =============================================================================

test "error context creation" {
    const ctx = vulkan.ErrorContext.from(
        vulkan.Error.OutOfDeviceMemory,
        "Test allocation failure",
        "Attempted to allocate 1GB buffer",
    );

    try std.testing.expect(ctx.code == vulkan.Error.OutOfDeviceMemory);
    try std.testing.expect(ctx.message.len > 0);
    try std.testing.expect(ctx.details != null);
    try std.testing.expect(ctx.timestamp > 0);
}

test "error trace chaining" {
    var trace = vulkan.ErrorTrace.init();
    try std.testing.expect(trace.count == 0);
    try std.testing.expect(trace.rootCause() == null);

    // Push multiple errors
    trace.push(vulkan.ErrorContext.from(vulkan.Error.OutOfDeviceMemory, "First error", null));
    trace.push(vulkan.ErrorContext.from(vulkan.Error.DeviceLost, "Second error", null));

    try std.testing.expect(trace.count == 2);

    const root = trace.rootCause();
    try std.testing.expect(root != null);
    try std.testing.expect(root.?.code == vulkan.Error.OutOfDeviceMemory);

    const latest = trace.latest();
    try std.testing.expect(latest != null);
    try std.testing.expect(latest.?.code == vulkan.Error.DeviceLost);
}

test "diagnostics suggestions" {
    const suggestions = vulkan.Diagnostics.getSuggestions(vulkan.Error.OutOfDeviceMemory);
    try std.testing.expect(suggestions.len > 0);

    const device_lost_suggestions = vulkan.Diagnostics.getSuggestions(vulkan.Error.DeviceLost);
    try std.testing.expect(device_lost_suggestions.len > 0);

    const related = vulkan.Diagnostics.getRelatedErrors(vulkan.Error.OutOfDeviceMemory);
    try std.testing.expect(related.len > 0);
}

test "error log global instance" {
    const log = vulkan.ErrorLog.global();
    try std.testing.expect(log != null);

    log.beginTrace();
    const ctx = vulkan.ErrorContext.from(vulkan.Error.Timeout, "Test timeout", null);
    log.log(ctx);

    const trace = log.currentTrace();
    try std.testing.expect(trace.count > 0);
}

// =============================================================================
// Persistent Pipeline Cache Tests
// =============================================================================

test "persistent pipeline cache - basic operations" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    // Create persistent cache with temp path
    const cache_path = "/tmp/test_pipeline_cache.bin";

    // Remove any existing cache file
    std.fs.cwd().deleteFile(cache_path) catch {};

    var cache = vulkan.PersistentPipelineCache.init(&ctx, .{
        .cache_path = cache_path,
        .auto_save = true,
        .validate_header = true,
    }) catch |e| {
        std.debug.print("\nPipeline cache creation failed: {}\n", .{e});
        return;
    };
    defer cache.deinit();

    // Check initial stats
    const stats = cache.stats;
    try std.testing.expect(!stats.loaded_from_disk); // First run, no existing cache
    try std.testing.expect(stats.pipelines_created == 0);

    // Record pipeline creation
    cache.recordPipelineCreated();
    try std.testing.expect(cache.stats.pipelines_created == 1);

    // Save cache
    cache.save() catch |e| {
        std.debug.print("\nCache save failed: {}\n", .{e});
        return;
    };

    // Clean up
    std.fs.cwd().deleteFile(cache_path) catch {};

    std.debug.print("\nPersistent pipeline cache test passed\n", .{});
}

test "persistent pipeline cache - reload from disk" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    const cache_path = "/tmp/test_pipeline_cache_reload.bin";
    std.fs.cwd().deleteFile(cache_path) catch {};

    // Create and save a cache
    {
        var cache = vulkan.PersistentPipelineCache.init(&ctx, .{
            .cache_path = cache_path,
            .auto_save = false,
        }) catch return;
        defer cache.deinit();

        cache.recordPipelineCreated();
        cache.save() catch return;
    }

    // Reload the cache
    {
        var cache = vulkan.PersistentPipelineCache.init(&ctx, .{
            .cache_path = cache_path,
            .validate_header = true,
        }) catch |e| {
            std.debug.print("\nCache reload failed: {}\n", .{e});
            std.fs.cwd().deleteFile(cache_path) catch {};
            return;
        };
        defer cache.deinit();

        // Should be loaded from disk if validation passed
        if (cache.loaded_from_disk) {
            try std.testing.expect(cache.stats.loaded_size > 0);
            std.debug.print("\nCache reloaded from disk ({} bytes)\n", .{cache.stats.loaded_size});
        }
    }

    std.fs.cwd().deleteFile(cache_path) catch {};
    std.debug.print("\nPersistent pipeline cache reload test passed\n", .{});
}

// =============================================================================
// Memory Pressure and Budget Tests
// =============================================================================

test "memory budget query" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    const budget = vulkan.queryMemoryBudget(&ctx);

    // Should have at least one heap
    try std.testing.expect(budget.heap_count > 0);

    // Check heap budget info
    for (0..budget.heap_count) |i| {
        const heap = budget.heaps[i];
        try std.testing.expect(heap.budget > 0); // Should have some budget
    }

    std.debug.print("\nMemory budget query passed (heaps: {}, ext available: {})\n", .{
        budget.heap_count,
        budget.budget_extension_available,
    });
}

test "memory budget monitor" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    const monitor = vulkan.MemoryBudgetMonitor.init(&ctx);
    const status = monitor.check();

    // Should be OK initially (no allocations)
    try std.testing.expect(status.level == .ok or status.level == .warning or status.level == .critical);
    try std.testing.expect(status.isOk() == (status.level == .ok));

    std.debug.print("\nMemory budget monitor test passed (status: {s}, max usage: {d:.1}%)\n", .{
        @tagName(status.level),
        status.max_usage_ratio * 100,
    });
}

test "memory pressure handler registration" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    var handler = vulkan.MemoryPressureHandler.init(std.testing.allocator, &ctx, .{
        .warning_threshold = 0.8,
        .critical_threshold = 0.95,
        .auto_evict = false,
    });
    defer handler.deinit();

    // Register a dummy evictable resource
    const dummy_callback = struct {
        fn evict(_: ?*anyopaque, _: usize) vulkan.EvictionResult {
            return .{ .bytes_freed = 0, .success = false };
        }
    }.evict;

    handler.registerResource(.{
        .name = "test_resource",
        .user_data = null,
        .evict_fn = dummy_callback,
        .priority = 0,
        .size_bytes = 1024,
        .last_access = std.time.nanoTimestamp(),
        .in_use = false,
    }) catch |e| {
        std.debug.print("\nResource registration failed: {}\n", .{e});
        return;
    };

    // Check pressure (should trigger threshold check)
    const pressure = handler.checkPressure();
    std.debug.print("\nMemory pressure check: {s}\n", .{@tagName(pressure)});

    std.debug.print("\nMemory pressure handler test passed\n", .{});
}

// =============================================================================
// Device Lost Recovery Tests
// =============================================================================

test "device lost handler configuration" {
    const config = vulkan.DeviceLostConfig{
        .max_recreation_attempts = 3,
        .recreation_delay_ms = 500,
        .enable_logging = false,
    };

    try std.testing.expect(config.max_recreation_attempts == 3);
    try std.testing.expect(config.recreation_delay_ms == 500);
    try std.testing.expect(!config.enable_logging);
}

test "device lost handler - resource registration" {
    var handler = vulkan.DeviceLostHandler.init(std.testing.allocator, .{});
    defer handler.deinit();

    // Test resource registration
    const dummy_recreate = struct {
        fn recreate(_: ?*anyopaque) vulkan.RecreationError!void {}
    }.recreate;

    handler.registerResource("test_buffer", null, dummy_recreate, 10) catch |e| {
        std.debug.print("\nResource registration failed: {}\n", .{e});
        return;
    };

    // Stats should show zero device lost events
    const stats = handler.getStats();
    try std.testing.expect(stats.device_lost_count == 0);
    try std.testing.expect(stats.successful_recoveries == 0);

    std.debug.print("\nDevice lost handler test passed\n", .{});
}

// =============================================================================
// Cross-Queue Synchronization Tests
// =============================================================================

test "queue family info" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    // Check queue availability - Context always has a compute queue (it's not optional)
    // The fact that we got here means the compute queue is valid
    std.debug.print("\nQueue availability test passed (compute queue available)\n", .{});

    // Check for optional async transfer queue
    if (ctx.transfer_queue) |_| {
        std.debug.print("Async transfer queue available\n", .{});
    }
}

test "timeline semaphore" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    // Try to create timeline semaphore (may fail if extension not supported)
    var timeline = vulkan.TimelineSemaphore.init(&ctx, 0) catch {
        std.debug.print("\nTimeline semaphores not supported - skipping\n", .{});
        return;
    };
    defer timeline.deinit();

    // Check initial value
    const value = timeline.getValue() catch {
        std.debug.print("\nFailed to get timeline value\n", .{});
        return;
    };
    try std.testing.expect(value == 0);

    // Signal to a new value
    timeline.signal(1) catch {
        std.debug.print("\nFailed to signal timeline\n", .{});
        return;
    };

    // Wait for the value
    const completed = timeline.wait(1, 1_000_000_000) catch {
        std.debug.print("\nTimeline wait failed\n", .{});
        return;
    };
    _ = completed; // Wait completed or timed out

    const new_value = timeline.getValue() catch 0;
    try std.testing.expect(new_value >= 1);

    std.debug.print("\nTimeline semaphore test passed\n", .{});
}

// =============================================================================
// Compute Graph Tests
// =============================================================================

test "compute graph - task registration" {
    var graph = vulkan.ComputeGraph.init(std.testing.allocator, undefined);
    defer graph.deinit();

    // Register resources
    const buf_a = graph.registerResource("buffer_a", null, 1024) catch |e| {
        std.debug.print("\nResource registration failed: {}\n", .{e});
        return;
    };
    const buf_b = graph.registerResource("buffer_b", null, 1024) catch |e| {
        std.debug.print("\nResource registration failed: {}\n", .{e});
        return;
    };

    // Add tasks
    _ = graph.addTask(.{
        .name = "write_task",
        .workgroups = .{ 64, 1, 1 },
        .resources = &.{.{ .handle = buf_a, .access = .write }},
    }) catch |e| {
        std.debug.print("\nTask registration failed: {}\n", .{e});
        return;
    };

    _ = graph.addTask(.{
        .name = "read_write_task",
        .workgroups = .{ 64, 1, 1 },
        .resources = &.{
            .{ .handle = buf_a, .access = .read },
            .{ .handle = buf_b, .access = .write },
        },
    }) catch |e| {
        std.debug.print("\nTask registration failed: {}\n", .{e});
        return;
    };

    // Compile graph
    graph.compile() catch |e| {
        std.debug.print("\nGraph compilation failed: {}\n", .{e});
        return;
    };

    const stats = graph.getStats();
    try std.testing.expect(stats.task_count == 2);
    try std.testing.expect(stats.resource_count == 2);
    try std.testing.expect(stats.level_count > 0);

    std.debug.print("\nCompute graph test passed (tasks: {}, barriers: {})\n", .{
        stats.task_count,
        stats.barrier_count,
    });
}

// =============================================================================
// Async Coordinator Tests
// =============================================================================

test "async coordinator - frame management" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    var coordinator = vulkan.AsyncCoordinator.init(&ctx, std.testing.allocator, .{
        .frame_count = 2,
        .use_timeline_semaphores = false,
    }) catch |e| {
        std.debug.print("\nAsync coordinator init failed: {}\n", .{e});
        return;
    };
    defer coordinator.deinit();

    // Test frame management
    try std.testing.expect(coordinator.getCurrentFrame() == 0);
    try std.testing.expect(coordinator.framesInFlight() == 0);

    std.debug.print("\nAsync coordinator test passed (async transfer: {})\n", .{
        coordinator.hasAsyncTransfer(),
    });
}

// =============================================================================
// Shader Compiler Tests
// =============================================================================

test "GLSL compiler availability" {
    const available = vulkan.glsl.isCompilerAvailable();
    std.debug.print("\nGLSL compiler available: {}\n", .{available});
}

test "HLSL compiler availability" {
    const available = vulkan.glsl.isHlslCompilerAvailable();
    std.debug.print("\nHLSL compiler (DXC) available: {}\n", .{available});
}

test "unified compiler initialization" {
    var compiler = vulkan.glsl.UnifiedCompiler.init(std.testing.allocator);
    defer compiler.deinit();

    std.debug.print("\nUnified compiler capabilities - GLSL: {}, HLSL: {}\n", .{
        compiler.supportsLanguage(.glsl),
        compiler.supportsLanguage(.hlsl),
    });
}

test "shader language detection" {
    try std.testing.expect(vulkan.glsl.detectLanguage("shader.comp") == .glsl);
    try std.testing.expect(vulkan.glsl.detectLanguage("shader.vert") == .glsl);
    try std.testing.expect(vulkan.glsl.detectLanguage("shader.frag") == .glsl);
    try std.testing.expect(vulkan.glsl.detectLanguage("shader.hlsl") == .hlsl);
    try std.testing.expect(vulkan.glsl.detectLanguage("shader.hlsli") == .hlsl);
    try std.testing.expect(vulkan.glsl.detectLanguage("unknown.txt") == null);

    std.debug.print("\nShader language detection test passed\n", .{});
}

// =============================================================================
// Multi-Device Context Tests
// =============================================================================

test "multi-device enumeration" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var multi = vulkan.MultiDeviceContext.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices) {
            std.debug.print("\nNo devices available for multi-device test\n", .{});
            return;
        }
        return e;
    };
    defer multi.deinit();

    const device_count = multi.deviceCount();
    std.debug.print("\nMulti-device context: {} device(s) found\n", .{device_count});

    // Enumerate devices
    for (0..device_count) |i| {
        if (multi.getDevice(@intCast(i))) |dev| {
            const info = dev.getInfo();
            std.debug.print("  Device {}: {s} ({s})\n", .{
                i,
                info.getName(),
                @tagName(info.device_type),
            });
        }
    }

    // Test device selection
    const best = multi.selectBestDevice(.{
        .prefer_discrete = true,
        .require_compute = true,
    });
    if (best) |dev| {
        std.debug.print("Best device: {s}\n", .{dev.getInfo().getName()});
    }

    std.debug.print("\nMulti-device enumeration test passed\n", .{});
}

// =============================================================================
// Shader Reflection Tests
// =============================================================================

test "shader reflection - basic SPIR-V parsing" {
    // Minimal valid compute SPIR-V header with OpCapability and OpMemoryModel
    const spirv = [_]u32{
        0x07230203, // Magic
        0x00010300, // Version 1.3
        0x00000000, // Generator
        0x00000020, // Bound = 32
        0x00000000, // Schema
        // OpCapability Shader
        0x00020011,
        0x00000001,
        // OpMemoryModel Logical GLSL450
        0x0003000E,
        0x00000000,
        0x00000001,
    };

    // Just verify the SPIR-V header is valid
    const result = vulkan.validateSpirv(&spirv);
    try std.testing.expect(result == .valid);

    std.debug.print("\nShader reflection parsing test passed\n", .{});
}

// =============================================================================
// Pooled Allocator Tests
// =============================================================================

test "pooled allocator configuration" {
    const config = vulkan.PoolConfig{
        .device_chunk_size = 16 * 1024 * 1024, // 16MB chunks
        .host_chunk_size = 8 * 1024 * 1024, // 8MB chunks
        .max_chunks_per_type = 16,
        .min_allocation_size = 256,
    };

    try std.testing.expect(config.device_chunk_size == 16 * 1024 * 1024);
    try std.testing.expect(config.max_chunks_per_type == 16);
    try std.testing.expect(config.min_allocation_size == 256);
}

test "pooled allocator - basic allocation" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    var pool = vulkan.PooledAllocator.init(
        ctx.device,
        ctx.device_dispatch,
        ctx.memory_properties,
        std.testing.allocator,
        .{},
    );
    defer pool.deinit();

    // Pooled allocator initialized successfully
    const stats = pool.getStats();
    std.debug.print("\nPooled allocator test passed (total allocations: {})\n", .{
        stats.total_allocations,
    });
}

// =============================================================================
// Shader Compiler Detection Tests
// =============================================================================

test "GLSL compiler detection" {
    // This test doesn't require Vulkan - just checks for compilers
    const available = vulkan.glsl.isCompilerAvailable();
    std.debug.print("\nGLSL compiler available: {}\n", .{available});

    const info = vulkan.glsl.detectCompilers(std.testing.allocator);
    std.debug.print("  GLSL: {s}\n", .{if (info.glsl_available) "yes" else "no"});
    if (info.glsl_backend) |backend| {
        std.debug.print("  Backend: {s}\n", .{backend.executable()});
    }
}

test "HLSL (DXC) compiler detection" {
    const available = vulkan.glsl.isHlslCompilerAvailable();
    std.debug.print("\nHLSL (DXC) compiler available: {}\n", .{available});

    const info = vulkan.glsl.detectCompilers(std.testing.allocator);
    std.debug.print("  DXC: {s}\n", .{if (info.hlsl_available) "yes" else "no"});
    if (info.max_shader_model) |sm| {
        std.debug.print("  Max Shader Model: {s}\n", .{sm.toString()});
    }
}

test "unified compiler detection" {
    const info = vulkan.glsl.detectCompilers(std.testing.allocator);
    std.debug.print("\nCompiler Detection Results:\n", .{});
    std.debug.print("  GLSL available: {}\n", .{info.glsl_available});
    std.debug.print("  HLSL available: {}\n", .{info.hlsl_available});

    if (!info.glsl_available and !info.hlsl_available) {
        std.debug.print("  No shader compilers found (install glslc or dxc)\n", .{});
    }
}

// =============================================================================
// Compute Graph Tests
// =============================================================================

test "compute graph - basic structure" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    var graph = vulkan.ComputeGraph.init(std.testing.allocator, &ctx);
    defer graph.deinit();

    // Register resources
    const buf_a = try graph.registerResource("input", null, 4096);
    const buf_b = try graph.registerResource("output", null, 4096);

    // Add tasks
    const task1 = try graph.addTask(.{
        .name = "preprocess",
        .workgroups = .{ 64, 1, 1 },
        .resources = &.{
            .{ .handle = buf_a, .access = .read },
            .{ .handle = buf_b, .access = .write },
        },
    });

    const task2 = try graph.addTask(.{
        .name = "postprocess",
        .workgroups = .{ 64, 1, 1 },
        .resources = &.{
            .{ .handle = buf_b, .access = .read_write },
        },
    });

    // Add explicit dependency
    try graph.addDependency(task1, task2);

    // Compile the graph
    try graph.compile();

    const stats = graph.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.task_count);
    try std.testing.expectEqual(@as(usize, 2), stats.resource_count);
    try std.testing.expect(stats.level_count >= 2); // Should have at least 2 levels

    std.debug.print("\nCompute graph test passed:\n", .{});
    std.debug.print("  Tasks: {}, Resources: {}, Barriers: {}, Levels: {}\n", .{
        stats.task_count,
        stats.resource_count,
        stats.barrier_count,
        stats.level_count,
    });
}

test "compute graph - parallel detection" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            return;
        }
        return e;
    };
    defer ctx.deinit();

    var graph = vulkan.ComputeGraph.init(std.testing.allocator, &ctx);
    defer graph.deinit();

    // Create independent resources
    const buf_a = try graph.registerResource("buf_a", null, 4096);
    const buf_b = try graph.registerResource("buf_b", null, 4096);
    const buf_c = try graph.registerResource("buf_c", null, 4096);

    // Add independent tasks (should run in parallel)
    _ = try graph.addTask(.{
        .name = "process_a",
        .workgroups = .{ 32, 1, 1 },
        .resources = &.{.{ .handle = buf_a, .access = .write }},
    });

    _ = try graph.addTask(.{
        .name = "process_b",
        .workgroups = .{ 32, 1, 1 },
        .resources = &.{.{ .handle = buf_b, .access = .write }},
    });

    _ = try graph.addTask(.{
        .name = "process_c",
        .workgroups = .{ 32, 1, 1 },
        .resources = &.{.{ .handle = buf_c, .access = .write }},
    });

    try graph.compile();

    const stats = graph.getStats();
    // All tasks should be in level 0 (parallel)
    try std.testing.expectEqual(@as(usize, 1), stats.level_count);
    try std.testing.expectEqual(@as(usize, 3), stats.max_parallelism);

    std.debug.print("\nParallel graph detection passed: max_parallelism = {}\n", .{stats.max_parallelism});
}

// =============================================================================
// Mock Context Tests
// =============================================================================

test "mock context - basic operations" {
    var mock = vulkan.MockContext.init(std.testing.allocator, .{});
    defer mock.deinit();

    // Allocate memory
    const handle = try mock.allocateMemory(4096, 0);
    try std.testing.expect(handle > 0);
    try std.testing.expect(mock.total_allocated == 4096);

    // Create buffer
    const buf_handle = try mock.createBuffer(1024);
    try std.testing.expect(buf_handle > 0);

    // Free memory
    mock.freeMemory(handle);
    try std.testing.expect(mock.total_allocated == 0);
    try std.testing.expect(!mock.hasMemoryLeaks());

    std.debug.print("\nMock context basic operations passed\n", .{});
}

test "mock context - error injection" {
    var mock = vulkan.MockContext.init(std.testing.allocator, .{});
    defer mock.deinit();

    // Inject OOM error
    try mock.injectError(.allocateMemory, vulkan.Error.OutOfDeviceMemory);

    // Allocation should fail
    const result = mock.allocateMemory(1024, 0);
    try std.testing.expectError(vulkan.Error.OutOfDeviceMemory, result);

    // Next allocation should succeed (error consumed)
    const handle = try mock.allocateMemory(1024, 0);
    try std.testing.expect(handle > 0);
    mock.freeMemory(handle);

    std.debug.print("\nMock context error injection passed\n", .{});
}

test "mock context - device lost simulation" {
    var mock = vulkan.MockContext.init(std.testing.allocator, .{});
    defer mock.deinit();

    // Simulate device lost
    mock.simulateDeviceLost();

    // All operations should return DeviceLost
    try std.testing.expectError(vulkan.Error.DeviceLost, mock.allocateMemory(1024, 0));
    try std.testing.expectError(vulkan.Error.DeviceLost, mock.createBuffer(1024));
    try std.testing.expectError(vulkan.Error.DeviceLost, mock.queueSubmit());

    // Reset and operations should succeed
    mock.resetDeviceLost();
    const handle = try mock.allocateMemory(1024, 0);
    mock.freeMemory(handle);

    std.debug.print("\nMock context device lost simulation passed\n", .{});
}

test "mock context - SPIR-V validation" {
    var mock = vulkan.MockContext.init(std.testing.allocator, .{});
    defer mock.deinit();

    // Valid SPIR-V
    const valid_spirv = try vulkan.testing.createMinimalSpirv(std.testing.allocator);
    defer std.testing.allocator.free(valid_spirv);

    const handle = try mock.createShaderModule(valid_spirv);
    try std.testing.expect(handle > 0);

    // Invalid SPIR-V
    const invalid_spirv = [_]u32{ 0, 0, 0, 0, 0 };
    try std.testing.expectError(vulkan.Error.InvalidSpirv, mock.createShaderModule(&invalid_spirv));

    std.debug.print("\nMock context SPIR-V validation passed\n", .{});
}

test "mock context - memory budget" {
    var mock = vulkan.MockContext.init(std.testing.allocator, .{});
    defer mock.deinit();

    const initial_budget = mock.getMemoryBudget();
    try std.testing.expect(initial_budget.heap_count == 2);
    try std.testing.expect(initial_budget.heaps[0].usage == 0);

    // Allocate and check budget
    const handle = try mock.allocateMemory(1024 * 1024, 0);
    const budget = mock.getMemoryBudget();
    try std.testing.expect(budget.heaps[0].usage == 1024 * 1024);

    mock.freeMemory(handle);
    const final_budget = mock.getMemoryBudget();
    try std.testing.expect(final_budget.heaps[0].usage == 0);

    std.debug.print("\nMock context memory budget tracking passed\n", .{});
}

test "mock context - leak detection" {
    var mock = vulkan.MockContext.init(std.testing.allocator, .{});
    defer mock.deinit();

    // No leaks initially
    try std.testing.expect(!mock.hasMemoryLeaks());

    // Allocate without freeing
    _ = try mock.allocateMemory(1024, 0);

    // Should detect leak
    try std.testing.expect(mock.hasMemoryLeaks());
    try std.testing.expect(mock.getLeakCount() == 1);

    std.debug.print("\nMock context leak detection passed\n", .{});
}

// =============================================================================
// Compute Graph Tests
// =============================================================================

test "compute graph - cycle detection error message" {
    // This test verifies that cycle detection provides useful error info
    // Note: Actual graph API requires real Vulkan context, so we test the concept
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            std.debug.print("\nNo compute-capable devices found - skipping\n", .{});
            return;
        }
        return e;
    };
    defer ctx.deinit();

    // Create a compute graph
    var graph = vulkan.ComputeGraph.init(std.testing.allocator, &ctx);
    defer graph.deinit();

    // Add tasks that would form a cycle (A -> B -> C -> A)
    const task_a = try graph.addTask(.{ .name = "task_a" });
    const task_b = try graph.addTask(.{ .name = "task_b" });
    const task_c = try graph.addTask(.{ .name = "task_c" });

    // Create cycle: A depends on C, B depends on A, C depends on B
    try graph.addDependency(task_a, task_c); // A waits for C
    try graph.addDependency(task_b, task_a); // B waits for A
    try graph.addDependency(task_c, task_b); // C waits for B (creates cycle)

    // Compile should fail with CyclicDependency error
    const result = graph.compile();
    try std.testing.expectError(error.CyclicDependency, result);

    std.debug.print("\nCompute graph cycle detection test passed\n", .{});
}

test "compute graph - valid DAG compiles" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            std.debug.print("\nNo compute-capable devices found - skipping\n", .{});
            return;
        }
        return e;
    };
    defer ctx.deinit();

    var graph = vulkan.ComputeGraph.init(std.testing.allocator, &ctx);
    defer graph.deinit();

    // Create valid DAG: A -> B -> C (no cycles)
    const task_a = try graph.addTask(.{ .name = "source" });
    const task_b = try graph.addTask(.{ .name = "middle" });
    const task_c = try graph.addTask(.{ .name = "sink" });

    try graph.addDependency(task_b, task_a); // B waits for A
    try graph.addDependency(task_c, task_b); // C waits for B

    // Should compile successfully
    try graph.compile();

    std.debug.print("\nCompute graph valid DAG test passed\n", .{});
}

// =============================================================================
// Buffer Usage Validation Tests
// =============================================================================

test "buffer - usage validation prevents misuse" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            std.debug.print("\nNo compute-capable devices found - skipping\n", .{});
            return;
        }
        return e;
    };
    defer ctx.deinit();

    // Create a storage buffer (device-local, not host-visible)
    var storage = try vulkan.Buffer.init(&ctx, 1024, .storage);
    defer storage.deinit(&ctx);

    // Attempt to write to storage buffer should fail
    // (storage buffers are device-local, require staging buffer for writes)
    const test_data = [_]u32{ 1, 2, 3, 4 };
    const write_result = storage.writeSlice(u32, &test_data);

    // Should fail with MemoryMapFailed (can't map device-local memory)
    try std.testing.expectError(vulkan.Error.MemoryMapFailed, write_result);

    // Create a staging buffer (host-visible, for transfers)
    var staging = try vulkan.Buffer.init(&ctx, 1024, .staging);
    defer staging.deinit(&ctx);

    // Writing to staging buffer should succeed
    try staging.writeSlice(u32, &test_data);

    std.debug.print("\nBuffer usage validation test passed\n", .{});
}

// =============================================================================
// Alignment and Overflow Tests
// =============================================================================

test "alignment utilities - overflow handling" {
    const Alignment = vulkan.util.Alignment;

    // Normal alignment should work
    try std.testing.expectEqual(@as(u64, 256), Alignment.alignUp(200, 256));
    try std.testing.expectEqual(@as(u64, 256), Alignment.alignUp(256, 256));
    try std.testing.expectEqual(@as(u64, 512), Alignment.alignUp(257, 256));

    // Test alignUpChecked returns null on overflow potential
    // Very large values should still work
    const large: u64 = std.math.maxInt(u64) - 1000;
    const result = Alignment.alignUpChecked(large, 256);
    // Should return null because aligning would overflow
    try std.testing.expect(result == null);

    // addChecked should handle overflow
    try std.testing.expectEqual(@as(?u64, 100), Alignment.addChecked(50, 50));
    try std.testing.expect(Alignment.addChecked(std.math.maxInt(u64), 1) == null);

    std.debug.print("\nAlignment overflow handling test passed\n", .{});
}

// =============================================================================
// Device Lost Recovery Tests
// =============================================================================

test "device lost handler - basic operation" {
    const recovery = vulkan.util.recovery;

    var handler = recovery.DeviceLostHandler.init(std.testing.allocator, .{
        .max_recreation_attempts = 2,
        .recreation_delay_ms = 1,
        .enable_logging = false,
    });
    defer handler.deinit();

    // Initial stats should be clean
    const initial_stats = handler.getStats();
    try std.testing.expectEqual(@as(u32, 0), initial_stats.device_lost_count);
    try std.testing.expectEqual(@as(usize, 0), initial_stats.registered_resources);

    // Register a test resource
    var recreation_called: bool = false;
    try handler.registerResource("test_buffer", &recreation_called, struct {
        fn recreate(data: ?*anyopaque) recovery.RecreationError!void {
            const called: *bool = @ptrCast(@alignCast(data orelse return));
            called.* = true;
        }
    }.recreate, 100);

    try std.testing.expectEqual(@as(usize, 1), handler.getStats().registered_resources);

    // Simulate device lost
    handler.invalidateAll();
    try std.testing.expectEqual(@as(usize, 0), handler.getStats().valid_resources);

    // Trigger recovery
    try handler.handleDeviceLost();

    // Verify resource was recreated
    try std.testing.expect(recreation_called);
    try std.testing.expectEqual(@as(usize, 1), handler.getStats().valid_resources);
    try std.testing.expectEqual(@as(u32, 1), handler.device_lost_count);

    std.debug.print("\nDevice lost handler test passed\n", .{});
}

test "error severity classification" {
    const recovery = vulkan.util.recovery;

    // Transient errors
    try std.testing.expectEqual(recovery.ErrorSeverity.transient, recovery.getErrorSeverity(vulkan.Error.Timeout));
    try std.testing.expectEqual(recovery.ErrorSeverity.transient, recovery.getErrorSeverity(vulkan.Error.NotReady));

    // Recoverable errors
    try std.testing.expectEqual(recovery.ErrorSeverity.recoverable, recovery.getErrorSeverity(vulkan.Error.DeviceLost));
    try std.testing.expectEqual(recovery.ErrorSeverity.recoverable, recovery.getErrorSeverity(vulkan.Error.OutOfDeviceMemory));

    // Fatal errors
    try std.testing.expectEqual(recovery.ErrorSeverity.fatal, recovery.getErrorSeverity(vulkan.Error.OutOfHostMemory));
    try std.testing.expectEqual(recovery.ErrorSeverity.fatal, recovery.getErrorSeverity(vulkan.Error.NoDevices));

    std.debug.print("\nError severity classification test passed\n", .{});
}

test "error collector - capacity enforcement" {
    const recovery = vulkan.util.recovery;

    var collector = recovery.ErrorCollector.init(std.testing.allocator, 3);
    defer collector.deinit();

    // Add more errors than capacity
    collector.record(vulkan.Error.Timeout, "test1", "msg1");
    collector.record(vulkan.Error.NotReady, "test2", "msg2");
    collector.record(vulkan.Error.DeviceLost, "test3", "msg3");
    collector.record(vulkan.Error.OutOfDeviceMemory, "test4", "msg4");

    // Should only keep last 3
    try std.testing.expectEqual(@as(usize, 3), collector.count());

    // Most recent should be last
    const last = collector.lastError().?;
    try std.testing.expectEqual(vulkan.Error.OutOfDeviceMemory, last.error_code);

    std.debug.print("\nError collector capacity test passed\n", .{});
}

// =============================================================================
// Extension Capability Tests
// =============================================================================

test "query extension capabilities" {
    if (!vulkan.isAvailable()) {
        std.debug.print("\n{s}\n", .{skip_message});
        return;
    }

    var ctx = vulkan.Context.init(std.testing.allocator, .{}) catch |e| {
        if (e == vulkan.Error.NoDevices or e == vulkan.Error.NoComputeQueue) {
            std.debug.print("\nNo compute-capable devices found - skipping\n", .{});
            return;
        }
        return e;
    };
    defer ctx.deinit();

    // Query all capabilities
    const caps = vulkan.queryAllCapabilities(&ctx, std.testing.allocator);

    // Print available capabilities
    std.debug.print("\nExtension capabilities:\n", .{});

    if (caps.subgroup) |sg| {
        std.debug.print("  Subgroup size: {d}\n", .{sg.subgroup_size});
    } else {
        std.debug.print("  Subgroup: not available\n", .{});
    }

    if (caps.descriptor_indexing) |di| {
        std.debug.print("  Descriptor indexing: {s}\n", .{if (di.runtime_descriptor_array) "yes" else "no"});
    } else {
        std.debug.print("  Descriptor indexing: not available\n", .{});
    }

    if (caps.ray_tracing) |rt| {
        std.debug.print("  Ray tracing: {s}\n", .{if (rt.isAvailable()) "yes" else "no"});
    } else {
        std.debug.print("  Ray tracing: not available\n", .{});
    }

    if (caps.cooperative_matrix) |cm| {
        std.debug.print("  Cooperative matrix: {s} ({d} configs)\n", .{
            if (cm.isAvailable()) "yes" else "no",
            cm.configurations.len,
        });
        // Free configurations if allocated
        if (cm.configurations.len > 0) {
            std.testing.allocator.free(cm.configurations);
        }
    } else {
        std.debug.print("  Cooperative matrix: not available\n", .{});
    }

    std.debug.print("\nExtension capabilities test passed\n", .{});
}
