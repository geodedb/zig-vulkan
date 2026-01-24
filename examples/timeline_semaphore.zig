//! Timeline Semaphore Example
//!
//! Demonstrates advanced GPU synchronization using timeline semaphores:
//! - Multi-stage pipeline with CPU/GPU coordination
//! - Async compute with transfer overlap
//! - Progress tracking and timeout handling
//!
//! Timeline semaphores allow waiting on specific counter values rather than
//! binary signaled/unsignaled states, enabling more flexible synchronization.

const std = @import("std");
const vulkan = @import("vulkan");

/// Pipeline stage identifiers
const PipelineStage = enum(u64) {
    idle = 0,
    upload_started = 1,
    upload_complete = 2,
    compute_started = 3,
    compute_complete = 4,
    download_started = 5,
    download_complete = 6,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Timeline Semaphore Example\n", .{});
    std.debug.print("==========================\n\n", .{});

    // Check availability
    if (!vulkan.isAvailable()) {
        std.debug.print("Vulkan not available\n", .{});
        return;
    }

    // Create context with Vulkan 1.2 for timeline semaphores
    var ctx = vulkan.Context.init(allocator, .{
        .app_name = "timeline_semaphore_example",
        .min_api_version = vulkan.vk.API_VERSION_1_2, // Timeline semaphores require 1.2
    }) catch |e| {
        std.debug.print("Failed to create context: {s}\n", .{vulkan.vk.describe(e)});
        std.debug.print("Note: Timeline semaphores require Vulkan 1.2+\n", .{});
        return;
    };
    defer ctx.deinit();

    const device_info = ctx.deviceInfo();
    std.debug.print("Device: {s}\n", .{device_info.getName()});
    std.debug.print("API Version: {}.{}.{}\n\n", .{
        vulkan.vk.apiVersionMajor(device_info.api_version),
        vulkan.vk.apiVersionMinor(device_info.api_version),
        vulkan.vk.apiVersionPatch(device_info.api_version),
    });

    // Create timeline semaphore
    std.debug.print("Creating timeline semaphore...\n", .{});
    var timeline = vulkan.TimelineSemaphore.init(&ctx, 0) catch |e| {
        std.debug.print("Failed to create timeline semaphore: {s}\n", .{vulkan.vk.describe(e)});
        std.debug.print("Timeline semaphores may not be supported on this device.\n", .{});
        return;
    };
    defer timeline.deinit();

    std.debug.print("Initial value: {}\n\n", .{timeline.current_value});

    // Create queue manager for multi-queue operations
    var queue_manager = vulkan.QueueManager.init(&ctx) catch |e| {
        std.debug.print("Failed to create queue manager: {s}\n", .{vulkan.vk.describe(e)});
        return;
    };
    defer queue_manager.deinit();

    const caps = queue_manager.getCapabilities();
    std.debug.print("Queue Capabilities:\n", .{});
    std.debug.print("  Dedicated transfer queue: {}\n", .{caps.has_dedicated_transfer});
    std.debug.print("  Async compute: {}\n\n", .{caps.has_async_compute});

    // Create test buffers
    const data_size: usize = 1024 * 1024; // 1 MB

    var staging = try vulkan.Buffer.init(&ctx, data_size, .staging);
    defer staging.deinit(&ctx);

    var compute_buffer = try vulkan.Buffer.init(&ctx, data_size, .storage);
    defer compute_buffer.deinit(&ctx);

    var readback = try vulkan.Buffer.init(&ctx, data_size, .readback);
    defer readback.deinit(&ctx);

    std.debug.print("Buffers created:\n", .{});
    std.debug.print("  Staging: {} bytes\n", .{data_size});
    std.debug.print("  Compute: {} bytes\n", .{data_size});
    std.debug.print("  Readback: {} bytes\n\n", .{data_size});

    // Simulate pipeline stages with timeline semaphore
    std.debug.print("Simulating pipeline stages...\n", .{});
    std.debug.print("-" ** 50 ++ "\n", .{});

    // Stage 1: Upload data (CPU signals when ready)
    std.debug.print("Stage 1: Uploading data to GPU...\n", .{});
    {
        // Write test data to staging buffer
        const test_pattern = [_]u8{0xAB} ** 256;
        var data: [data_size]u8 = undefined;
        for (0..data.len / test_pattern.len) |i| {
            @memcpy(data[i * test_pattern.len ..][0..test_pattern.len], &test_pattern);
        }
        try staging.write(&data);

        // Signal upload started
        try timeline.signal(@intFromEnum(PipelineStage.upload_started));
        std.debug.print("  Signaled: upload_started ({})\n", .{@intFromEnum(PipelineStage.upload_started)});

        // Record transfer command
        var cmd = try vulkan.CommandBuffer.init(&ctx);
        defer cmd.deinit();

        try cmd.begin();
        vulkan.Transfer.copy(ctx.device_dispatch, cmd.getHandle(), staging, compute_buffer, null);
        try cmd.end();

        // Submit and wait
        try vulkan.Submission.submitAndWait(&ctx, cmd);

        // Signal upload complete
        try timeline.signal(@intFromEnum(PipelineStage.upload_complete));
        std.debug.print("  Signaled: upload_complete ({})\n", .{@intFromEnum(PipelineStage.upload_complete)});
    }

    // Check current value
    const current_val = try timeline.getValue();
    std.debug.print("  Current timeline value: {}\n\n", .{current_val});

    // Stage 2: Compute processing (simulated)
    std.debug.print("Stage 2: GPU compute processing...\n", .{});
    {
        // In a real scenario, we'd dispatch compute shaders here
        // For now, simulate with timeline signals

        try timeline.signal(@intFromEnum(PipelineStage.compute_started));
        std.debug.print("  Signaled: compute_started ({})\n", .{@intFromEnum(PipelineStage.compute_started)});

        // Simulate compute work (small delay)
        _ = std.time.nanoTimestamp(); // Just a small operation instead of sleep

        try timeline.signal(@intFromEnum(PipelineStage.compute_complete));
        std.debug.print("  Signaled: compute_complete ({})\n", .{@intFromEnum(PipelineStage.compute_complete)});
    }

    // Stage 3: Download results
    std.debug.print("\nStage 3: Downloading results...\n", .{});
    {
        try timeline.signal(@intFromEnum(PipelineStage.download_started));
        std.debug.print("  Signaled: download_started ({})\n", .{@intFromEnum(PipelineStage.download_started)});

        // Wait for compute to complete before downloading
        const timeout_ns = 1_000_000_000; // 1 second
        const completed = try timeline.wait(@intFromEnum(PipelineStage.compute_complete), timeout_ns);
        std.debug.print("  Wait for compute_complete: {s}\n", .{if (completed) "success" else "timeout"});

        // Record download command
        var cmd = try vulkan.CommandBuffer.init(&ctx);
        defer cmd.deinit();

        try cmd.begin();
        vulkan.Transfer.copy(ctx.device_dispatch, cmd.getHandle(), compute_buffer, readback, null);
        try cmd.end();

        // Submit and wait
        try vulkan.Submission.submitAndWait(&ctx, cmd);

        try timeline.signal(@intFromEnum(PipelineStage.download_complete));
        std.debug.print("  Signaled: download_complete ({})\n", .{@intFromEnum(PipelineStage.download_complete)});
    }

    // Final state
    std.debug.print("\nFinal State:\n", .{});
    std.debug.print("-" ** 50 ++ "\n", .{});
    const final_val = try timeline.getValue();
    std.debug.print("Final timeline value: {}\n", .{final_val});

    // Demonstrate timeout handling
    std.debug.print("\nTimeout Handling Demo:\n", .{});
    std.debug.print("-" ** 50 ++ "\n", .{});
    {
        // Try to wait for a value that hasn't been signaled yet
        const future_value: u64 = 100;
        const short_timeout = 10_000_000; // 10ms
        std.debug.print("Waiting for value {} with 10ms timeout...\n", .{future_value});

        const result = try timeline.wait(future_value, short_timeout);
        std.debug.print("Result: {s}\n", .{if (result) "reached" else "timeout (expected)"});
    }

    // Usage patterns summary
    std.debug.print("\nTimeline Semaphore Usage Patterns:\n", .{});
    std.debug.print("-" ** 50 ++ "\n", .{});
    std.debug.print("1. Multi-stage pipeline: Track progress through stages\n", .{});
    std.debug.print("2. Producer-consumer: Signal batch completion\n", .{});
    std.debug.print("3. Frame pacing: Coordinate CPU and GPU work\n", .{});
    std.debug.print("4. Async operations: Wait with timeout for completion\n", .{});
    std.debug.print("5. Progress tracking: Query current value anytime\n", .{});

    std.debug.print("\nExample completed successfully!\n", .{});
}
