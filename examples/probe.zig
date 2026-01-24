//! Probe Example
//!
//! Demonstrates how to check for Vulkan availability and enumerate devices.
//! This example does not require any GPU compute capabilities.

const std = @import("std");
const vulkan = @import("vulkan");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Vulkan Compute Library - System Probe\n", .{});
    std.debug.print("=====================================\n\n", .{});

    // Quick availability check
    std.debug.print("Quick check: Vulkan loader ", .{});
    if (vulkan.isAvailable()) {
        std.debug.print("✓ available\n\n", .{});
    } else {
        std.debug.print("✗ NOT available\n", .{});
        std.debug.print("\nVulkan is not available on this system.\n", .{});
        std.debug.print("This may be because:\n", .{});
        std.debug.print("  - Vulkan drivers are not installed\n", .{});
        std.debug.print("  - The Vulkan loader (libvulkan) is not available\n", .{});
        std.debug.print("  - Running in a container without GPU access\n", .{});
        return;
    }

    // Detailed probe
    var result = vulkan.probe(allocator);
    defer result.deinit(allocator);

    if (result.probe_error) |err| {
        std.debug.print("Probe error: {s}\n", .{vulkan.vk.describe(err)});
        return;
    }

    // Instance version
    std.debug.print("Vulkan Instance Version: {}.{}.{}\n", .{
        vulkan.vk.apiVersionMajor(result.instance_version),
        vulkan.vk.apiVersionMinor(result.instance_version),
        vulkan.vk.apiVersionPatch(result.instance_version),
    });

    std.debug.print("Devices Found: {}\n\n", .{result.device_count});

    if (result.device_count == 0) {
        std.debug.print("No Vulkan devices found.\n", .{});
        std.debug.print("The Vulkan loader is present but no GPUs were detected.\n", .{});
        return;
    }

    // Device details
    for (result.devices, 0..) |dev, i| {
        std.debug.print("Device {}: {s}\n", .{ i, dev.getName() });
        std.debug.print("  Type: {s}\n", .{@tagName(dev.device_type)});
        std.debug.print("  API Version: {}.{}.{}\n", .{
            vulkan.vk.apiVersionMajor(dev.api_version),
            vulkan.vk.apiVersionMinor(dev.api_version),
            vulkan.vk.apiVersionPatch(dev.api_version),
        });
        std.debug.print("  Vendor ID: 0x{X:0>4}\n", .{dev.vendor_id});
        std.debug.print("  Device ID: 0x{X:0>4}\n", .{dev.device_id});
        std.debug.print("  Compute Queue: {s}\n", .{if (dev.has_compute_queue) "✓ yes" else "✗ no"});

        if (dev.has_compute_queue) {
            std.debug.print("  Compute Queue Family: {}\n", .{dev.compute_queue_family.?});
        }

        std.debug.print("  Total Memory: {} MB\n", .{dev.total_memory / (1024 * 1024)});
        std.debug.print("  Max Workgroup Size: {}x{}x{}\n", .{
            dev.max_compute_work_group_size[0],
            dev.max_compute_work_group_size[1],
            dev.max_compute_work_group_size[2],
        });
        std.debug.print("  Max Workgroup Count: {}x{}x{}\n", .{
            dev.max_compute_work_group_count[0],
            dev.max_compute_work_group_count[1],
            dev.max_compute_work_group_count[2],
        });
        std.debug.print("  Shared Memory: {} KB\n", .{dev.max_compute_shared_memory_size / 1024});
        std.debug.print("  Timestamps: {s}\n", .{if (dev.timestamp_supported) "✓ supported" else "✗ not supported"});
        std.debug.print("\n", .{});
    }

    // Recommendation
    if (result.getBestComputeDevice()) |best| {
        std.debug.print("Recommended Device: {s}\n", .{best.getName()});
    } else {
        std.debug.print("No compute-capable device found.\n", .{});
    }

    // Test context creation
    std.debug.print("\nTesting context creation... ", .{});

    var ctx = vulkan.Context.init(allocator, .{}) catch |e| {
        std.debug.print("✗ failed: {s}\n", .{vulkan.vk.describe(e)});
        return;
    };
    defer ctx.deinit();

    std.debug.print("✓ success\n", .{});

    const info = ctx.deviceInfo();
    std.debug.print("Using: {s}\n", .{info.getName()});
}
