//! Multi-Device Compute Example
//!
//! Demonstrates using multiple Vulkan devices (GPUs) for parallel compute operations.
//! This pattern is useful for:
//! - Utilizing multiple discrete GPUs in a system
//! - Distributing workloads across integrated and discrete GPUs
//! - Maximizing throughput for embarrassingly parallel problems
//!
//! ## Hardware Requirements
//!
//! - Multiple Vulkan-capable devices (e.g., integrated + discrete GPU)
//! - Each device must support compute queues
//!
//! ## Usage
//!
//! ```bash
//! zig build example-multi-device
//! ./zig-out/bin/multi_device
//! ```
//!
//! ## Architecture
//!
//! This example creates separate contexts for each device, allowing fully
//! independent resource management and command submission. Synchronization
//! between devices must be handled at the application level (e.g., using
//! host synchronization after readback).

const std = @import("std");
const vulkan = @import("vulkan");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Multi-Device Compute Example\n", .{});
    std.debug.print("=============================\n\n", .{});

    // Check Vulkan availability
    if (!vulkan.isAvailable()) {
        std.debug.print("Vulkan is not available on this system.\n", .{});
        return;
    }

    // Probe for devices
    var probe = vulkan.probe(allocator);
    defer probe.deinit(allocator);

    if (!probe.isAvailable()) {
        std.debug.print("No Vulkan devices found.\n", .{});
        return;
    }

    const device_count = probe.devices.len;
    std.debug.print("Found {} Vulkan device(s)\n\n", .{device_count});

    if (device_count < 2) {
        std.debug.print("Multi-device example requires at least 2 devices.\n", .{});
        std.debug.print("Demonstrating device enumeration instead:\n\n", .{});

        for (probe.devices, 0..) |device, i| {
            std.debug.print("Device {}: {s}\n", .{ i, device.getName() });
            std.debug.print("  Type: {s}\n", .{@tagName(device.device_type)});
            std.debug.print("  Memory: {} MB\n\n", .{device.total_memory / (1024 * 1024)});
        }
        return;
    }

    // Create contexts for multiple devices
    std.debug.print("Creating contexts for {} devices...\n\n", .{device_count});

    var contexts = std.ArrayListUnmanaged(vulkan.Context){};
    defer {
        for (contexts.items) |*ctx| {
            ctx.deinit();
        }
        contexts.deinit(allocator);
    }

    // Initialize each device
    for (probe.devices, 0..) |device, i| {
        std.debug.print("Initializing Device {}: {s}\n", .{ i, device.getName() });

        var ctx = vulkan.Context.init(allocator, .{
            .device_index = @intCast(i),
        }) catch |e| {
            std.debug.print("  Failed to create context: {}\n\n", .{e});
            continue;
        };

        const info = ctx.deviceInfo();
        std.debug.print("  Type: {s}\n", .{@tagName(info.device_type)});
        std.debug.print("  Memory: {} MB\n", .{info.total_memory / (1024 * 1024)});
        std.debug.print("  Max Workgroup: {}x{}x{}\n\n", .{
            info.max_compute_work_group_size[0],
            info.max_compute_work_group_size[1],
            info.max_compute_work_group_size[2],
        });

        try contexts.append(allocator, ctx);
    }

    if (contexts.items.len < 2) {
        std.debug.print("Could not initialize multiple devices for compute.\n", .{});
        return;
    }

    std.debug.print("Successfully initialized {} device contexts!\n\n", .{contexts.items.len});

    // Demonstrate multi-device work distribution strategy
    printMultiDeviceStrategy();
}

fn printMultiDeviceStrategy() void {
    std.debug.print(
        \\Multi-Device Work Distribution Strategies:
        \\==========================================
        \\
        \\1. Data Parallelism (Recommended for large datasets)
        \\   - Split input data evenly across devices
        \\   - Each device processes its partition independently
        \\   - Combine results on host after completion
        \\
        \\   Example: Image batch processing
        \\   - Device 0: Process images 0-999
        \\   - Device 1: Process images 1000-1999
        \\
        \\2. Pipeline Parallelism (For multi-stage processing)
        \\   - Each device handles a different processing stage
        \\   - Data flows from one device to the next
        \\   - Useful when stages have different compute requirements
        \\
        \\   Example: ML inference pipeline
        \\   - Device 0 (Fast GPU): Preprocessing + first layers
        \\   - Device 1 (Large GPU): Deep layers + postprocessing
        \\
        \\3. Heterogeneous Computing (Mixed device types)
        \\   - Integrated GPU: Small, latency-sensitive tasks
        \\   - Discrete GPU: Large, throughput-oriented tasks
        \\
        \\   Example: Real-time + batch processing
        \\   - Integrated: UI rendering, small updates
        \\   - Discrete: Background compute, large batches
        \\
        \\Implementation Notes:
        \\- Each device has independent memory spaces
        \\- Data transfer between devices goes through host memory
        \\- Use async operations to overlap transfer and compute
        \\- Consider PCIe bandwidth when planning data movement
        \\
        \\Code Pattern:
        \\
        \\    // Split work across devices
        \\    const work_per_device = total_work / device_count;
        \\
        \\    for (contexts.items) |*ctx, device_idx| {{
        \\        const start = device_idx * work_per_device;
        \\        const end = start + work_per_device;
        \\
        \\        // Create device-local buffers
        \\        var buffer = try vulkan.Buffer.init(ctx, buffer_size, .storage);
        \\        defer buffer.deinit(ctx);
        \\
        \\        // Upload partition of data
        \\        var staging = try vulkan.Buffer.init(ctx, buffer_size, .staging);
        \\        defer staging.deinit(ctx);
        \\        try staging.writeSlice(f32, data[start..end]);
        \\
        \\        // Submit work...
        \\    }}
        \\
    , .{});
}
